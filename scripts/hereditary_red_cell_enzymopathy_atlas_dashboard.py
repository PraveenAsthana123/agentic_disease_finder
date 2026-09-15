"""Hereditary Red Cell Enzymopathy Atlas — 8-Gene Reference
G6PD-PKLR-HK1-GPI-PGK1-TPI1-ALDOA-PFKM
PPP / Embden-Meyerhof Glycolysis Enzymopathies
320 patients (8 x 40), seeds 2838-2845.
Endpoints: /api/hereditary-red-cell-enzymopathy-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "G6PD",
        "protein": (
            "G6PD -- Xq28 XLR -- 515aa -- Glucose-6-Phosphate-Dehydrogenase-59kDa-"
            "Pentose-Phosphate-Pathway-Rate-Limiting-Step-NADPH-Glutathione-Oxidative-Defence-"
            "Most-Common-Human-Enzymopathy-500-Million-Worldwide-"
            "OMIM-Gene-305900-Disease-G6PD-Deficiency-G6PDd-300908"
        ),
        "locus": "Xq28",
        "protein_size": (
            "515 aa / 59 kDa (G6PD homodimer; catalyses first and rate-limiting step of pentose phosphate pathway "
            "(PPP); glucose-6-phosphate + NADP+ → 6-phosphoglucono-δ-lactone + NADPH; "
            "NADPH is essential for regenerating reduced glutathione (GSH via glutathione reductase); "
            "GSH protects erythrocytes from oxidative damage (H2O2 scavenging); "
            "X-linked recessive: males fully affected (hemizygous); females mosaic (lyonisation); "
            "WHO classification: Class I = CNSHA (rare); Class II-III = episodic favism/drug-induced; "
            "Class IV = normal activity; Class V = increased activity; "
            "~500 million people affected worldwide — most common human enzymopathy; "
            "protective heterozygosity against Plasmodium falciparum malaria)"
        ),
        "inheritance": (
            "X-LINKED RECESSIVE (XLR) — hemizygous males affected; "
            "EPIDEMIOLOGY: "
            "  ~500 million worldwide; highest prevalence sub-Saharan Africa, Mediterranean, Middle East, South/Southeast Asia; "
            "  Africa: A- variant (202A + 376G substitutions) ~13% of males; "
            "  Mediterranean: B- variant (563T; c.563C>T, p.Ser188Phe; class II; severe); "
            "  Sephardic Jewish, Sardinian, Greek: Mediterranean variant; "
            "  Southeast Asia: Mahidol, Viangchan, Canton variants; "
            "FEMALE CARRIERS: "
            "  Heterozygous females: mosaic (lyonisation) → variable from unaffected to affected; "
            "  Homozygous females possible in high-prevalence populations; "
            "  X-inactivation skewing → carrier females may have intermediate enzyme activity; "
            "WHO CLASS SYSTEM: "
            "  Class I: <10% residual activity + CNSHA (chronic, non-episodic); rare; e.g. G6PD Canton severe variants; "
            "  Class II: <10% activity but ONLY episodic hemolysis with triggers (favism/drugs); most Mediterranean variant; "
            "  Class III: 10-60% activity; episodic hemolysis (milder triggers); Africa A-; "
            "  Class IV: 60-150% normal; no hemolysis; polymorphic variants; "
            "  Class V: >150% normal activity; no disease; rare; "
            "PLASMODIUM PROTECTION: "
            "  Heterozygous females (and hemizygous males with moderate variants) have partial protection against severe malaria; "
            "  Explains global distribution pattern"
        ),
        "disease_category": (
            "G6PD DEFICIENCY — OMIM 300908; "
            "SPECTRUM (WHO CLASS): "
            "  CLASS I (CNSHA only): "
            "    Chronic compensated or severe non-spherocytic hemolytic anemia; "
            "    Jaundice, splenomegaly, gallstones even without triggers; "
            "    Hb typically 7-11 g/dL at baseline; "
            "  CLASS II-III (Episodic hemolysis — common): "
            "    BASELINE: Hb often near-normal (10-14 g/dL); reticulocytes mildly elevated; "
            "    ACUTE CRISIS: "
            "      Hemolytic triggers: fava beans (vicine/convicine → H2O2), drugs (primaquine, dapsone, "
            "        rasburicase, nitrofurantoin, methylene blue), infection, neonatal period; "
            "      Back pain, dark urine (haemoglobinuria), jaundice, falling Hb; "
            "      Heinz bodies: oxidised denatured haemoglobin inside erythrocytes → splenic removal → "
            "        extravascular hemolysis + occasional intravascular; "
            "      Reticulocytes rise sharply during crisis recovery; "
            "NEONATAL JAUNDICE: "
            "  Major manifestation regardless of class; "
            "  G6PD neonatal jaundice → kernicterus → leading preventable cause of bilirubin encephalopathy; "
            "  Screening: neonatal G6PD enzyme assay or molecular testing in high-prevalence populations"
        ),
        "disease_pathway": (
            "PENTOSE PHOSPHATE PATHWAY (PPP) / NADPH / GLUTATHIONE OXIDATIVE DEFENCE: "
            "NORMAL G6PD FUNCTION: "
            "  PPP step 1: G6PD + G6P + NADP+ → 6-phosphoglucono-δ-lactone + NADPH; "
            "  PPP step 2: 6-phosphogluconolactonase → 6-phosphogluconate; "
            "  PPP step 3: 6-phosphogluconate dehydrogenase + NADP+ → ribulose-5-phosphate + NADPH + CO2; "
            "  2 NADPH per glucose-6-phosphate; "
            "  NADPH → glutathione reductase → reduced glutathione (GSH from GSSG); "
            "  GSH → glutathione peroxidase → scavenges H2O2 and lipid hydroperoxides; "
            "  Erythrocytes: NO mitochondria → PPP = SOLE source of NADPH; "
            "G6PD DEFICIENCY (LOF): "
            "  ↓ NADPH → ↓ GSH → accumulated H2O2 and reactive oxygen species; "
            "  Haemoglobin oxidation → methemoglobin → Heinz body formation; "
            "  Heinz bodies bind RBC membrane → rigidity → extravascular splenic sequestration; "
            "EPISODIC TRIGGER MECHANISM (Classes II-III): "
            "  Favism: vicine/convicine (fava bean) → intestinally metabolised → divicine/isouramil → "
            "    → directly generates H2O2 → overwhelms residual G6PD → hemolytic crisis; "
            "  Primaquine/dapsone: quinone metabolites → cycling H2O2 generation; "
            "  Infection: phagocyte oxidative burst generates ROS → overwhelms G6PD-deficient RBCs nearby; "
            "G6PD IN CLASS I (CNSHA): "
            "  Residual G6PD barely functional even at rest → insufficient basal NADPH → "
            "  → constitutive Heinz body formation → chronic ongoing hemolysis"
        ),
        "pathognomonic": (
            "EPISODIC HEMOLYSIS TRIGGERED BY FAVA BEANS / PRIMAQUINE / INFECTION — "
            "PATHOGNOMONIC CLINICAL PATTERN FOR G6PD CLASS II-III; "
            "FAVISM (G6PD-SPECIFIC PATHOGNOMONIC): "
            "  Hemolytic crisis within 24-72h of fava bean ingestion; "
            "  Unique to G6PD deficiency; no other hereditary hemolytic anemia triggered by fava beans; "
            "  Mediterranean variant (class II) most severe favism; "
            "HEINZ BODIES ON SUPRAVITAL STAIN: "
            "  Brilliant cresyl blue stain: irregular intracellular inclusions (oxidised Hb clumps); "
            "  Bite cells on Romanowsky stain (Heinz bodies removed by spleen leaving bite mark); "
            "  Only visible during/just after acute hemolytic episode; "
            "ENZYME ASSAY TIMING — CRITICAL DIAGNOSTIC TRAP: "
            "  G6PD enzyme assay: MUST BE PERFORMED AFTER HEMOLYTIC CRISIS HAS RESOLVED (≥4-6 weeks); "
            "  DURING crisis: reticulocytes surge → reticulocytes have NORMAL or near-normal G6PD → "
            "    falsely normal assay result; old G6PD-deficient cells already hemolysed; "
            "  RULE: normal enzyme result during crisis does NOT exclude G6PD deficiency; "
            "  Repeat assay in steady state; or molecular analysis (unaffected by reticulocyte surge); "
            "MALE PATIENT + MEDITERRANEAN/AFRICAN ANCESTRY + BACK PAIN + DARK URINE: "
            "  Classic presentation — haemoglobinuria in G6PD hemolytic crisis; "
            "NEONATAL JAUNDICE WITHOUT ISOIMMUNISATION + HIGH-PREVALENCE POPULATION: "
            "  G6PD deficiency is a major cause of neonatal jaundice → phototherapy/exchange transfusion risk; "
            "  Newborn screening in high-prevalence populations mandatory"
        ),
        "treatment": (
            "NO SPECIFIC CURATIVE TREATMENT — AVOIDANCE OF TRIGGERS IS KEYSTONE: "
            "TRIGGER AVOIDANCE (PRIMARY PREVENTION): "
            "  Avoid fava beans (all class II-III), primaquine, dapsone, rasburicase, "
            "    nitrofurantoin, methylene blue (at therapeutic doses); "
            "  Infection treatment: prompt antimicrobials (infection itself triggers hemolysis); "
            "  G6PD-SAFE ANTIMALARIAL: artemisinin derivatives OK at standard doses; "
            "    primaquine/tafenoquine CONTRAINDICATED in class II-III; "
            "ACUTE HEMOLYTIC EPISODE MANAGEMENT: "
            "  Hydration (haemoglobinuria → renal protection); "
            "  Folic acid supplementation; "
            "  Packed red cell transfusion for severe anemia (Hb <7 g/dL or symptomatic); "
            "  Neonatal jaundice: phototherapy; exchange transfusion for severe hyperbilirubinaemia; "
            "SPLENECTOMY: NOT recommended (extravascular hemolysis mainly spleen, but splenectomy "
            "  carries infection risk and does not cure underlying enzyme deficiency; "
            "  limited benefit except in refractory Class I CNSHA); "
            "CLASS I (CNSHA) SPECIFIC: "
            "  Regular folate; transfusion support as needed; iron chelation if transfusion-dependent; "
            "  Splenectomy in select severe cases; "
            "  Gene therapy: early phase clinical trials (lentiviral G6PD delivery to HSCs); "
            "GENETIC COUNSELLING: X-linked; maternal carrier testing; prenatal counselling"
        ),
        "seed": 2838,
        "pt_vars": {
            "hb_range": (10.0, 14.0),
            "retic_range": (2.0, 6.0),
            "ldh_range": (200, 400),
            "bili_range": (20, 50),
            "splenomegaly_pct": 40,
            "gallstones_pct": 25,
            "aplastic_crisis_pct": 5,
            "neuro_pct": 0,
            "myopathy_pct": 0,
            "rhabdo_pct": 0,
            "transfusion_dependent_pct": 8,
            "splenectomy_pct": 2,
            "enzyme_activity_range": (0, 10),
        }
    },
    {
        "gene": "PKLR",
        "protein": (
            "PKLR -- 1q22 AR -- 574aa -- Pyruvate-Kinase-L-R-62kDa-"
            "Glycolysis-Step-10-Phosphoenolpyruvate-to-Pyruvate-ATP-Production-"
            "Most-Common-Hereditary-CNSHA-PK-Deficiency-"
            "OMIM-Gene-609712-Disease-PK-Deficiency-Hemolytic-Anemia-266200"
        ),
        "locus": "1q22",
        "protein_size": (
            "574 aa (L-isoform) / 62 kDa (R-isoform is erythrocyte-specific, same gene PKLR, "
            "different promoter; L-isoform = liver; R-isoform = erythrocyte; "
            "homotetramer; catalyses step 10 of glycolysis: PEP + ADP → pyruvate + ATP; "
            "rate-limiting ATP-generating step in RBCs; "
            "PKLR LOF → ATP depletion → rigid RBC membrane → extravascular hemolysis; "
            "2,3-DPG accumulation (key clinical pearl); "
            "mitapivat (AG-348) allosteric activator, FDA approved 2022; "
            "R479W most common European mutation)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function) — PK Deficiency; "
            "EPIDEMIOLOGY: "
            "  Most common hereditary CNSHA worldwide (~1:20,000 Northern Europeans); "
            "  Most common non-immune non-spherocytic hereditary hemolytic anemia; "
            "  Northern European Caucasian: R479W (p.Arg479Trp) founder mutation most common; "
            "  Amish founder: p.Arg479Trp homozygous; "
            "MECHANISM — PKLR R-ISOFORM: "
            "  Two promoters in PKLR gene: "
            "    Promoter 1 (liver promoter): L-isoform (1537bp leader, exons 1-11); "
            "    Promoter 2 (erythroid promoter): R-isoform (different exon 1, exons 2-11, 574 aa); "
            "  Coding mutations in exons 2-11: affect BOTH L and R isoforms; "
            "  Promoter mutations: affect only one isoform; "
            "ATP DEPLETION → CNSHA: "
            "  No pyruvate kinase activity → PEP cannot form pyruvate → no ATP from step 10; "
            "  Erythrocyte ATP depletion → Na+/K+ ATPase failure → cell swells → membrane rigidity; "
            "  Rigid RBCs trapped in spleen → extravascular hemolysis; "
            "2,3-DPG PARADOX (PATHOGNOMONIC METABOLIC FEATURE): "
            "  PK blockade → PEP accumulation → 2,3-DPG accumulates (Rapoport-Luebering shunt); "
            "  ELEVATED 2,3-DPG → right-shifts O2 dissociation curve (decreased O2 affinity); "
            "  RIGHT-SHIFT: Hb releases O2 more readily to tissues at any given pO2; "
            "  CLINICAL CONSEQUENCE: patients TOLERATE LOWER HAEMOGLOBIN than expected; "
            "  PK-deficient patient with Hb 7 g/dL may be minimally symptomatic; "
            "  SPLENECTOMY PARADOX: after splenectomy, reticulocytes (high G6PD, low PK sensitivity) survive better "
            "    → RBC count rises but 2,3-DPG effect lost with normal RBCs → "
            "    → patients may FEEL WORSE after splenectomy despite higher Hb"
        ),
        "disease_category": (
            "PYRUVATE KINASE DEFICIENCY — HEMOLYTIC ANEMIA (PK-HAN) — OMIM 266200; "
            "HAEMATOLOGICAL PROFILE: "
            "  Hb: 7-11 g/dL (wide range; some severe neonates, some mild adults); "
            "  Reticulocytes: 8-25% (markedly elevated; proportional to hemolysis severity); "
            "  LDH: 400-800 U/L (hemolysis marker); "
            "  Bilirubin: 40-100 μmol/L (indirect; jaundice); "
            "  Blood film: echinocytes (spiculated cells = pyknocytes in neonates); "
            "  Splenomegaly: 85% (chronic extravascular hemolysis → spleen enlarges); "
            "  Gallstones: 60% (bilirubin gallstones from chronic hemolysis); "
            "CLINICAL: "
            "  Neonatal jaundice: 50-60% (phototherapy or exchange transfusion often required); "
            "  Childhood: variable — some require regular transfusions; others compensate; "
            "  PARVOVIRUS B19 APLASTIC CRISIS: B19 → arrests erythropoiesis → acute Hb fall → "
            "    life-threatening emergency (hospitalisation, transfusion); "
            "IRON OVERLOAD: "
            "  Occurs even WITHOUT transfusion (increased iron absorption from ineffective erythropoiesis); "
            "  Transfusion-dependent patients: faster iron loading; "
            "  Liver biopsy or MRI T2* for iron quantification"
        ),
        "disease_pathway": (
            "EMBDEN-MEYERHOF GLYCOLYSIS — PYRUVATE KINASE STEP (STEP 10) — ATP PRODUCTION DEFECT: "
            "NORMAL GLYCOLYSIS IN ERYTHROCYTES: "
            "  Step 10: PEP + ADP → pyruvate + ATP (via pyruvate kinase R-isoform); "
            "  Major ATP generation step; RBCs lack mitochondria → entirely glycolytic for ATP; "
            "  ATP powers: Na+/K+ ATPase (cell volume), Ca2+-ATPase (calcium pump), membrane deformability; "
            "PKLR LOF — ATP DEPLETION: "
            "  PEP blocked → cannot form pyruvate → ATP generation fails at step 10; "
            "  Residual ATP from step 7 (phosphoglycerate kinase/PGK1) — insufficient; "
            "  Na+/K+ ATPase fails → Na+ influx + K+ loss → cell dehydration and rigidity; "
            "  Membrane lipid asymmetry disrupted → phosphatidylserine externalisation → "
            "    → macrophage recognition → extravascular destruction; "
            "2,3-DPG SHUNT MECHANISM: "
            "  PEP accumulation upstream → back-pressure → 2,3-DPG accumulation via Rapoport-Luebering; "
            "  2,3-DPG binds β-haemoglobin subunit T-state interface → stabilises deoxy-Hb; "
            "  Right-shifts O2-Hb dissociation curve: lower pO2 required for Hb O2 release → better tissue O2; "
            "MITAPIVAT MECHANISM: "
            "  AG-348 (mitapivat): allosteric activator of PK-R dimer interface → "
            "    → activates residual PK-R enzyme in heterozygous/compound heterozygous patients; "
            "  Increases pyruvate/ATP → reduces 2,3-DPG (right-shift corrected) → Hb rises; "
            "  FDA approved 2022 for adult non-transfusion-dependent PK deficiency"
        ),
        "pathognomonic": (
            "2,3-DPG ELEVATION + RIGHT-SHIFTED O2 DISSOCIATION CURVE + CNSHA + SPLENOMEGALY: "
            "ELEVATED 2,3-DPG (PATHOGNOMONIC METABOLIC HALLMARK): "
            "  Erythrocyte 2,3-DPG markedly elevated (2-3x upper limit); "
            "  Unique among glycolytic enzymopathies — only PK deficiency causes this reliably; "
            "  Explains why PK-deficient patients TOLERATE HAEMOGLOBIN LEVELS that would be symptomatic in other anemias; "
            "  Clinical pearl: PK patient with Hb 7 g/dL asymptomatic → don't over-transfuse; "
            "SPLENECTOMY PARADOX — CLINICAL ALERT: "
            "  Post-splenectomy: Hb rises (longer-lived RBCs), but reticulocytes fall → "
            "  → 2,3-DPG effect reduces → patients may report WORSE symptoms despite higher Hb; "
            "  Explain this to patients before splenectomy decision; "
            "ECHINOCYTES ON BLOOD FILM: "
            "  Spiculated cells = PK-characteristic morphology (not spherocytes as in HS); "
            "  Distinguishes PK deficiency from hereditary spherocytosis; "
            "PARVOVIRUS B19 APLASTIC CRISIS: "
            "  In any CNSHA patient with sudden severe Hb drop → reticulocytopenia → "
            "    B19 IgM serology immediately; "
            "  PK-deficient patients particularly vulnerable; "
            "  Immunocompetent patients: crisis self-limited ~10 days; transfusion bridge; "
            "ENZYME ASSAY: "
            "  PK-R enzyme activity markedly reduced (<25% in most affected patients); "
            "  Assay in erythrocytes (not leukocytes — use M isoform); "
            "  Perform fluorescent spot test (screening) then quantitative assay for confirmation"
        ),
        "treatment": (
            "MITAPIVAT (AG-348, PYRUKYND) — FDA APPROVED 2022 — DISEASE-MODIFYING: "
            "  Allosteric PK-R activator; oral twice daily; "
            "  Hb rise in non-transfusion-dependent adults; reduced transfusion burden; "
            "  Clinical trials: paediatric and transfusion-dependent populations ongoing; "
            "  FIRST disease-modifying therapy for PK deficiency; "
            "SUPPORTIVE CARE: "
            "  Folic acid supplementation (high turnover erythropoiesis); "
            "  Packed red cell transfusion for severe anemia (Hb <6-7 g/dL or symptomatic); "
            "  Parvovirus B19 crisis: transfusion bridge + await resolution; "
            "SPLENECTOMY: "
            "  Reduces transfusion requirement in transfusion-dependent patients; "
            "  Performs well for spleen-mediated extravascular hemolysis; "
            "  TIMING: defer until >5 years to preserve immune function; vaccines mandatory pre-splenectomy; "
            "  COUNSEL patients about post-splenectomy 2,3-DPG paradox; "
            "IRON MANAGEMENT: "
            "  Serum ferritin monitoring every 6-12 months; "
            "  Iron chelation: deferasirox, deferoxamine for significant iron overload; "
            "  Avoid iron supplementation (overload risk even without transfusion); "
            "NEONATAL: "
            "  Phototherapy for jaundice; exchange transfusion for severe hyperbilirubinaemia; "
            "HSCT: curative; reserved for severely affected children with matched sibling donor; "
            "GENE THERAPY: clinical trials ongoing (lentiviral PK-R delivery)"
        ),
        "seed": 2839,
        "pt_vars": {
            "hb_range": (7.0, 11.0),
            "retic_range": (8.0, 25.0),
            "ldh_range": (400, 800),
            "bili_range": (40, 100),
            "splenomegaly_pct": 85,
            "gallstones_pct": 60,
            "aplastic_crisis_pct": 20,
            "neuro_pct": 0,
            "myopathy_pct": 0,
            "rhabdo_pct": 0,
            "transfusion_dependent_pct": 35,
            "splenectomy_pct": 40,
            "enzyme_activity_range": (5, 25),
        }
    },
    {
        "gene": "HK1",
        "protein": (
            "HK1 -- 10q22.1 AR -- 917aa -- Hexokinase-1-100kDa-"
            "Glycolysis-Step-1-Glucose-to-Glucose-6-Phosphate-"
            "Erythrocyte-Specific-HK1-E-N-Terminal-7aa-Insert-"
            "OMIM-Gene-142600-Disease-HK1-Deficiency-CNSHA-235700"
        ),
        "locus": "10q22.1",
        "protein_size": (
            "917 aa / 100 kDa (hexokinase 1; catalyses step 1 of glycolysis: glucose + ATP → glucose-6-phosphate + ADP; "
            "erythrocyte-specific isoform HK1-E: N-terminal 7 amino acid insert unique to RBCs; "
            "HK1 most important hexokinase isoform in erythrocytes; "
            "inhibited by its own product G6P (product inhibition regulates flux); "
            "HK1 LOF → immediate glycolytic failure at step 1 → ATP depletion → hemolysis; "
            "2,3-DPG LOW in HK1 deficiency (unlike PKLR — key DDx); "
            "rare; fewer than 100 described cases in literature)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function) — HK1 deficiency; "
            "MECHANISM: "
            "  HK1 is step 1 of glycolysis: glucose + ATP → glucose-6-phosphate (G6P) + ADP; "
            "  G6P is the entry point for both glycolysis (energy, ATP) and PPP (NADPH, oxidative defence); "
            "  HK1 LOF → no G6P formation → neither glycolysis nor PPP can proceed; "
            "  Combined ATP depletion + (secondary) reduced NADPH; "
            "  Most severe upstream block in glycolysis; "
            "2,3-DPG DISTINGUISHING FEATURE: "
            "  HK1 deficiency: G6P absent → glycolysis entirely blocked from step 1; "
            "  2,3-DPG CANNOT accumulate (requires glycolytic intermediates downstream of G6P); "
            "  2,3-DPG LOW → left-shifts O2 dissociation curve (high O2 affinity) → "
            "    → poor tissue O2 delivery despite whatever hemoglobin is present; "
            "  OPPOSITE of PKLR (2,3-DPG elevated in PKLR, low in HK1); "
            "SEVERITY: "
            "  Generally more severe CNSHA than PKLR; "
            "  Splenomegaly universal; iron overload common; "
            "  Neonatal onset typical; often transfusion-dependent"
        ),
        "disease_category": (
            "HEXOKINASE 1 DEFICIENCY — CNSHA — OMIM 235700; "
            "HAEMATOLOGICAL PROFILE: "
            "  Hb: 7-10 g/dL (often severe CNSHA); "
            "  Reticulocytes: 10-20% (markedly elevated); "
            "  LDH: 500-900 U/L (significant hemolysis); "
            "  Bilirubin: 50-120 μmol/L; "
            "  Splenomegaly: 90% (near-universal); "
            "  Gallstones: 65%; "
            "  2,3-DPG: LOW (opposite of PKLR); "
            "CLINICAL: "
            "  Onset: neonatal jaundice (severe) or early childhood; "
            "  Chronic hemolysis: severe; often transfusion-dependent from infancy; "
            "  Iron overload: significant (transfusional + increased absorption); "
            "DIAGNOSIS: "
            "  HK enzyme activity absent/markedly reduced in RBCs; "
            "  Molecular analysis of HK1; "
            "  2,3-DPG low (helps DDx from PKLR where elevated)"
        ),
        "disease_pathway": (
            "EMBDEN-MEYERHOF GLYCOLYSIS — HEXOKINASE STEP 1 — TOTAL GLYCOLYTIC FAILURE: "
            "NORMAL HK1 FUNCTION: "
            "  Glucose enters RBC via GLUT1 (facilitated diffusion); "
            "  HK1: glucose + ATP → G6P + ADP (irreversible; committed step of glycolysis); "
            "  Product inhibition: G6P inhibits HK1 when abundant (flux regulation); "
            "  G6P flows into: glycolysis (→ ATP) AND pentose phosphate pathway (→ NADPH); "
            "HK1 LOF: "
            "  No G6P → glycolysis blocked at step 1 → no downstream ATP or NADPH; "
            "  Combined glycolytic failure + oxidative stress susceptibility; "
            "  Erythrocyte membrane pump failure → rigid cells → splenic trapping; "
            "EXTRAVASCULAR HEMOLYSIS: "
            "  Spleen: rigid HK1-deficient RBCs held in splenic cords → macrophage phagocytosis; "
            "  Splenomegaly from chronic high hemolytic load; "
            "2,3-DPG ABSENCE MECHANISM: "
            "  Normal 2,3-DPG synthesis requires glycolytic intermediate 1,3-BPG (after G6P); "
            "  HK1 block at step 1 → no 1,3-BPG available → no 2,3-DPG production; "
            "  Low 2,3-DPG → haemoglobin RETAINS oxygen (left-shift) → impaired tissue O2 delivery; "
            "  This exacerbates the clinical anemia beyond what Hb level suggests"
        ),
        "pathognomonic": (
            "SEVERE CNSHA WITH 2,3-DPG LOW — DDx FROM PKLR (2,3-DPG ELEVATED): "
            "2,3-DPG LOW (KEY DISTINGUISHING FEATURE): "
            "  HK1: 2,3-DPG LOW (glycolysis blocked before 1,3-BPG entry into Rapoport-Luebering shunt); "
            "  PKLR: 2,3-DPG HIGH (glycolysis blocked after 1,3-BPG → 2,3-DPG accumulates); "
            "  This 2,3-DPG dichotomy is the pivotal DDx tool between HK1 and PKLR deficiency; "
            "ABSENT HK ENZYME ACTIVITY ON RBC ENZYME PANEL: "
            "  Complete or near-complete absence of HK activity in red cells; "
            "  Normal leukocyte HK (uses HK2/HK3 isoforms — not affected by HK1 RBC-specific mutation); "
            "  RBC-specific assay essential; "
            "UNIVERSAL SPLENOMEGALY: "
            "  HK1 → extravascular hemolysis → splenomegaly in nearly all patients; "
            "  Giant spleen not uncommon; "
            "SEVERE NEONATAL JAUNDICE REQUIRING EXCHANGE TRANSFUSION: "
            "  HK1 deficiency typically presents more severely neonatally than PKLR; "
            "EARLY TRANSFUSION DEPENDENCE: "
            "  Many HK1-deficient patients require regular transfusions from infancy"
        ),
        "treatment": (
            "NO SPECIFIC APPROVED THERAPY — SUPPORTIVE MANAGEMENT: "
            "TRANSFUSION SUPPORT: "
            "  Regular packed RBC transfusions for severe anemia; "
            "  Target Hb >8 g/dL (lower threshold acceptable given 2,3-DPG is already LOW — not right-shifted); "
            "IRON CHELATION: "
            "  Aggressive chelation required: deferasirox preferred; deferoxamine for severe overload; "
            "  Monitor ferritin 3-monthly; MRI liver T2* annually; "
            "SPLENECTOMY: "
            "  Partial benefit (reduces extravascular destruction); "
            "  Defer until >5 years; post-splenectomy vaccines; "
            "  Less benefit than in hereditary spherocytosis (enzyme absent, not membrane defect); "
            "FOLIC ACID: daily supplementation for high-turnover erythropoiesis; "
            "NEONATAL MANAGEMENT: "
            "  Phototherapy; exchange transfusion for severe neonatal jaundice; "
            "STEM CELL TRANSPLANTATION: "
            "  Curative; for severely transfusion-dependent patients; "
            "  Preferably before iron overload becomes irreversible organ damage; "
            "GENE THERAPY: theoretical; in preclinical development"
        ),
        "seed": 2840,
        "pt_vars": {
            "hb_range": (7.0, 10.0),
            "retic_range": (10.0, 20.0),
            "ldh_range": (500, 900),
            "bili_range": (50, 120),
            "splenomegaly_pct": 90,
            "gallstones_pct": 65,
            "aplastic_crisis_pct": 15,
            "neuro_pct": 0,
            "myopathy_pct": 0,
            "rhabdo_pct": 0,
            "transfusion_dependent_pct": 40,
            "splenectomy_pct": 45,
            "enzyme_activity_range": (5, 20),
        }
    },
    {
        "gene": "GPI",
        "protein": (
            "GPI -- 19q13.11 AR -- 558aa -- Glucose-6-Phosphate-Isomerase-63kDa-"
            "Glycolysis-Step-2-G6P-to-Fructose-6-Phosphate-"
            "2nd-Most-Common-Non-Spherocytic-CNSHA-After-PKLR-"
            "OMIM-Gene-172400-Disease-GPI-Deficiency-CNSHA-613470"
        ),
        "locus": "19q13.11",
        "protein_size": (
            "558 aa / 63 kDa (GPI homodimer; phosphoglucose isomerase; catalyses glycolysis step 2: "
            "glucose-6-phosphate → fructose-6-phosphate (reversible isomerisation); "
            "also known as autocrine motility factor (AMF) when secreted by tumour cells — different function; "
            "biallelic AR mutations → deficient GPI in RBCs → glycolytic arrest at step 2 → ATP depletion; "
            "2nd most common hereditary non-spherocytic CNSHA after PKLR; "
            "French founder: p.Arg347His; severity variable; neurological involvement in rare severe cases)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function) — GPI deficiency; "
            "MECHANISM: "
            "  GPI catalyses G6P ↔ Fructose-6-phosphate (reversible); "
            "  GPI LOF → G6P accumulates, fructose-6-phosphate deficient; "
            "  Glycolysis arrested at step 2 → ATP depletion → hemolysis; "
            "  G6P may overflow into PPP → NADPH relatively preserved (unlike HK1 where both fail); "
            "  2,3-DPG: normal or mildly increased (downstream glycolytic intermediates still formed partially); "
            "SEVERITY: "
            "  Variable — mild chronic compensation to severe CNSHA; "
            "  Neurological involvement in most severely affected (see disease_category); "
            "EPIDEMIOLOGY: "
            "  ~50 described families; truly rare; "
            "  French founder p.Arg347His; "
            "GPI DUAL FUNCTION: "
            "  Intracellular: glycolytic enzyme; "
            "  Secreted as AMF (autocrine motility factor): promotes tumour cell motility — "
            "    clinically irrelevant in GPI-deficient hemolytic disease"
        ),
        "disease_category": (
            "GPI DEFICIENCY — CNSHA — OMIM 613470; "
            "HAEMATOLOGICAL PROFILE: "
            "  Hb: 8-11 g/dL (moderate-severe CNSHA); "
            "  Reticulocytes: 8-20%; "
            "  LDH: 400-700 U/L; "
            "  Bilirubin: 40-90 μmol/L; "
            "  Splenomegaly: 80%; "
            "  Gallstones: 55%; "
            "  2,3-DPG: normal or mildly elevated (partial downstream glycolysis preserved via overflow through fructose pathway); "
            "NEUROLOGICAL INVOLVEMENT (SEVERE CASES): "
            "  Intellectual disability, developmental delay, ataxia, pyramidal signs in ~20-30% of severely affected; "
            "  Mechanism: GPI required in neurons (glycolytic ATP in brain); "
            "  Not universal — depends on severity of mutation and residual enzyme activity; "
            "CLINICAL: "
            "  Neonatal jaundice common; "
            "  Variable severity of hemolytic anemia throughout life; "
            "  Relatively mild platelet function — normal"
        ),
        "disease_pathway": (
            "EMBDEN-MEYERHOF GLYCOLYSIS — GPI STEP 2 — FRUCTOSE-6-PHOSPHATE DEFICIENCY: "
            "NORMAL GPI: "
            "  G6P ↔ F6P (reversible isomerisation; GPI homodimer); "
            "  F6P flows forward: F6P → F-1,6-BP (via PFK-1) → ultimately to ATP at steps 7+10; "
            "  G6P also enters PPP (PPP entry does NOT require GPI); "
            "GPI LOF: "
            "  G6P accumulates → cannot be converted to F6P → glycolytic arrest at step 2; "
            "  G6P excess → increased PPP flux (G6PD step) → NADPH partially preserved; "
            "  But downstream glycolytic ATP generation fails → ATP depletion → RBC rigidity → hemolysis; "
            "  F6P deficiency → cannot generate F-1,6-BP → no downstream substrate for aldolase (ALDOA) or PFK; "
            "2,3-DPG STATUS: "
            "  Partial preservation because: "
            "    some F6P still formed by residual GPI activity in heterozygote cells; "
            "    accessory pathways may contribute minor F6P; "
            "  2,3-DPG not as severely reduced as HK1; "
            "EXTRAVASCULAR HEMOLYSIS: "
            "  Same mechanism as other glycolytic defects: ATP depletion → membrane failure → spleen trapping"
        ),
        "pathognomonic": (
            "HEREDITARY CNSHA + LOW GPI ENZYME ACTIVITY IN RBCs — DDx PKLR BY ENZYME PANEL: "
            "GPI ENZYME ASSAY: "
            "  GPI activity markedly reduced in erythrocytes (not leukocytes — use RBC-specific assay); "
            "  Quantitative enzyme assay in RBCs: standard haematology enzyme panel includes GPI; "
            "  Normal GPI in leukocytes (leukocytes have alternative isoform contribution); "
            "DDx FROM PKLR: "
            "  Both: CNSHA, splenomegaly, gallstones, elevated bilirubin; "
            "  PKLR: 2,3-DPG elevated, echinocytes on film; "
            "  GPI: 2,3-DPG normal/mild, no specific morphological change; "
            "  Enzyme panel distinguishes definitively; "
            "FRENCH FOUNDER MUTATION p.Arg347His: "
            "  Most common European GPI mutation; Sanger/NGS identifies; "
            "NEUROLOGICAL FEATURES IN SEVERE GPI DEFICIENCY: "
            "  CNSHA + developmental delay/intellectual disability + no other metabolic cause → "
            "    include GPI in enzyme panel; "
            "  Not present in milder cases; severity-dependent; "
            "NO SPECIFIC MORPHOLOGY: "
            "  GPI-deficient RBCs: no pathognomonic morphology; "
            "  Polychromasia + mild normocytic anemia = non-specific hemolytic picture"
        ),
        "treatment": (
            "NO SPECIFIC APPROVED THERAPY — SUPPORTIVE: "
            "TRANSFUSION SUPPORT: "
            "  For symptomatic severe anemia (Hb <7 g/dL) or aplastic crisis; "
            "  Folic acid supplementation; "
            "SPLENECTOMY: "
            "  Beneficial for transfusion-dependent patients; "
            "  Reduces hemolytic rate; "
            "  Post-splenectomy vaccines + penicillin prophylaxis; "
            "IRON CHELATION: "
            "  Transfusional iron overload management; "
            "NEUROLOGICAL MANAGEMENT (IF PRESENT): "
            "  Supportive developmental intervention; "
            "  Seizure management with anti-epileptics; "
            "HSCT: "
            "  Curative for severe GPI deficiency with neurological involvement; "
            "  Corrects haematological but NOT neurological component (neurons persist with GPI LOF); "
            "GENETIC COUNSELLING: AR, 25% risk per pregnancy"
        ),
        "seed": 2841,
        "pt_vars": {
            "hb_range": (8.0, 11.0),
            "retic_range": (8.0, 20.0),
            "ldh_range": (400, 700),
            "bili_range": (40, 90),
            "splenomegaly_pct": 80,
            "gallstones_pct": 55,
            "aplastic_crisis_pct": 15,
            "neuro_pct": 25,
            "myopathy_pct": 0,
            "rhabdo_pct": 0,
            "transfusion_dependent_pct": 30,
            "splenectomy_pct": 35,
            "enzyme_activity_range": (5, 25),
        }
    },
    {
        "gene": "PGK1",
        "protein": (
            "PGK1 -- Xq21.1 XLR -- 417aa -- Phosphoglycerate-Kinase-1-45kDa-"
            "Glycolysis-Step-7-1,3-BPG-to-3-Phosphoglycerate-ATP-Production-"
            "CNSHA-Myopathy-CNS-Triad-Only-Multisystem-Glycolytic-Enzymopathy-"
            "OMIM-Gene-311800-Disease-PGK1-Deficiency-300653"
        ),
        "locus": "Xq21.1",
        "protein_size": (
            "417 aa / 45 kDa (phosphoglycerate kinase 1; X-linked; catalyses glycolysis step 7: "
            "1,3-bisphosphoglycerate + ADP → 3-phosphoglycerate + ATP; first ATP-generating step in glycolysis; "
            "single isoform encoded by PGK1 (ubiquitous); "
            "PGK2 = testis-specific (not affected by PGK1 mutations); "
            "X-linked recessive: males severely affected, females mosaic; "
            "UNIQUE TRIAD: CNSHA + MYOPATHY + CNS involvement; "
            "only glycolytic enzymopathy affecting all three systems; "
            "exertional rhabdomyolysis major complication)"
        ),
        "inheritance": (
            "X-LINKED RECESSIVE (XLR) — males fully affected, females mosaic (variable); "
            "MECHANISM: "
            "  PGK1 catalyses step 7: 1,3-BPG + ADP → 3-PG + ATP; "
            "  PGK1 is first substrate-level ATP generation step in glycolysis; "
            "  PGK1 LOF → no ATP from step 7 → step 10 ATP only if PEP reaches PK; "
            "  However: 1,3-BPG also shunts to 2,3-DPG (Rapoport-Luebering) → 2,3-DPG elevated; "
            "  ATP depletion → RBC rigidity → extravascular hemolysis; "
            "MULTISYSTEM INVOLVEMENT — UNIQUE FEATURE: "
            "  Erythrocytes: CNSHA (glycolytic ATP depletion); "
            "  Muscle: skeletal muscle highly glycolytic → PGK1 LOF → ATP depletion during exercise → "
            "    myopathy + rhabdomyolysis; "
            "  CNS: brain glucose metabolism via glycolysis → PGK1 LOF → neuronal energy failure → "
            "    intellectual disability, seizures, behavioural difficulties; "
            "  ONLY PGK1 among glycolytic enzymopathies causes all three systems; "
            "GENOTYPE-PHENOTYPE: "
            "  p.Arg206Pro: predominantly CNS phenotype (intellectual disability, seizures); "
            "  p.Asp268Asn: predominantly myopathy phenotype (exertional rhabdomyolysis, minimal CNS)"
        ),
        "disease_category": (
            "PGK1 DEFICIENCY — CNSHA + MYOPATHY + CNS TRIAD — OMIM 300653; "
            "TRIAD (ALL THREE NOT ALWAYS PRESENT — VARIABLE): "
            "  1. CNSHA: "
            "     Hb 8-12 g/dL; reticulocytes 6-18%; LDH elevated; bilirubin elevated; splenomegaly; "
            "     Often mild compared to PKLR/HK1; "
            "  2. MYOPATHY: "
            "     Exercise intolerance, myalgia, muscle weakness; "
            "     EXERTIONAL RHABDOMYOLYSIS — major complication: "
            "       Acute muscle breakdown with exercise → myoglobinuria → renal failure risk; "
            "       CK markedly elevated (10,000-100,000+ U/L during crisis); "
            "     Fixed proximal weakness in severe cases; "
            "  3. CNS: "
            "     Intellectual disability (mild to severe); "
            "     Seizures; behavioural problems; emotional lability; "
            "     Pyramidal signs in severe cases; "
            "CLINICAL VARIANTS: "
            "  Pure hemolytic: rare (CNSHA without myopathy/CNS); "
            "  Hemolytic + CNS (no myopathy); "
            "  Hemolytic + myopathy (no CNS) — most common combined phenotype; "
            "  All three = severe genotype"
        ),
        "disease_pathway": (
            "EMBDEN-MEYERHOF GLYCOLYSIS — PGK1 STEP 7 — MULTISYSTEM ATP DEFICIENCY: "
            "NORMAL PGK1: "
            "  Step 7: 1,3-BPG + ADP → 3-PG + ATP (first ATP produced in glycolysis); "
            "  Also: 1,3-BPG enters Rapoport-Luebering shunt → 2,3-DPG (only in erythrocytes); "
            "  Proceeds to step 8 (phosphoglycerate mutase) → step 9 (enolase) → step 10 (PK); "
            "PGK1 LOF IN ERYTHROCYTES: "
            "  Step 7 ATP absent → only step 10 ATP available → overall ATP depletion; "
            "  1,3-BPG diverts to 2,3-DPG (Rapoport-Luebering overflow) → 2,3-DPG elevated; "
            "  Similar to PKLR: elevated 2,3-DPG → right-shift → relatively better tissue O2; "
            "PGK1 LOF IN MUSCLE: "
            "  Skeletal muscle: no Rapoport-Luebering shunt (only in RBCs); "
            "  During exercise: glycolytic demand ↑ → PGK1 failure → rapid ATP depletion → "
            "    → anaerobic threshold reached immediately → exercise intolerance; "
            "    → rhabdomyolysis: myosin pump failure → Ca2+ influx → proteolysis → myoglobin release; "
            "PGK1 LOF IN NEURONS: "
            "  Brain relies heavily on glycolysis for fast ATP (though also OXPHOS); "
            "  Neurons: PGK1 LOF → reduced synaptic ATP → altered ion gradients → seizure threshold lowered; "
            "  Developmental: neuronal migration/differentiation requires glycolytic ATP in foetal brain"
        ),
        "pathognomonic": (
            "CNSHA + MYOPATHY + CNS TRIAD IN AN X-LINKED PATTERN — "
            "UNIQUE TO PGK1 AMONG ALL GLYCOLYTIC ENZYMOPATHIES: "
            "TRIAD UNIQUENESS: "
            "  PGK1 is the ONLY glycolytic enzyme deficiency that causes all three: CNSHA + myopathy + CNS; "
            "  TPI1 causes CNSHA + neurodegeneration (but no myopathy); "
            "  ALDOA causes CNSHA + myopathy + CNS (similar triad — differentiate by enzyme assay); "
            "  PFKM causes myopathy + mild CNSHA (no CNS); "
            "EXERTIONAL RHABDOMYOLYSIS: "
            "  Acute exercise → myoglobinuria (cola-coloured urine) → acute kidney injury risk; "
            "  CK >10,000 U/L; "
            "  X-linked: young males with rhabdomyolysis + hemolysis → PGK1 assay mandatory; "
            "X-LINKED INHERITANCE + HAEMOLYSIS + MUSCLE/CNS: "
            "  Males: full phenotype; "
            "  Mothers: carriers may have mild hemolysis; "
            "PGK1 ENZYME ASSAY MANDATORY IN: "
            "  Any male with unexplained hemolysis + myopathy (± CNS): "
            "    rule out PGK1 on multi-enzyme haematology panel; "
            "  Standard haematology enzyme panel includes PGK1; "
            "GENOTYPE-PHENOTYPE COUNSELLING: "
            "  p.Arg206Pro variants: prepare CNS management team; "
            "  p.Asp268Asn: prepare myopathy + rhabdomyolysis management"
        ),
        "treatment": (
            "MULTISYSTEM MANAGEMENT — NO CURATIVE APPROVED THERAPY: "
            "HEMOLYTIC ANEMIA: "
            "  Folic acid supplementation; "
            "  Transfusion for severe anemia; "
            "  Splenectomy: moderate benefit; timing >5 years; "
            "MYOPATHY MANAGEMENT (KEY PRIORITY): "
            "  AVOID strenuous exercise and triggers for rhabdomyolysis; "
            "  Hydration before exercise; oral carbohydrates pre-exercise (glucose supplementation); "
            "  RHABDOMYOLYSIS EPISODE: "
            "    IV fluid resuscitation — aggressive hydration 200-300 mL/kg/day; "
            "    Monitor urine output, renal function (creatinine, BUN); "
            "    Urinary alkalinisation (sodium bicarbonate) to prevent myoglobin precipitation; "
            "    Avoid nephrotoxins; "
            "    Dialysis if acute kidney injury develops; "
            "CNS MANAGEMENT: "
            "  Seizures: anti-epileptic drugs (standard); "
            "  Intellectual disability: early intervention, special education; "
            "  Behavioural support; "
            "HSCT: "
            "  Curative for haematological component; "
            "  Does NOT reverse neurological or muscle manifestations (PGK1 persists in those cells); "
            "GENE THERAPY: preclinical"
        ),
        "seed": 2842,
        "pt_vars": {
            "hb_range": (8.0, 12.0),
            "retic_range": (6.0, 18.0),
            "ldh_range": (350, 700),
            "bili_range": (30, 80),
            "splenomegaly_pct": 75,
            "gallstones_pct": 45,
            "aplastic_crisis_pct": 10,
            "neuro_pct": 60,
            "myopathy_pct": 80,
            "rhabdo_pct": 40,
            "transfusion_dependent_pct": 25,
            "splenectomy_pct": 30,
            "enzyme_activity_range": (5, 20),
        }
    },
    {
        "gene": "TPI1",
        "protein": (
            "TPI1 -- 12p13.31 AR -- 286aa -- Triosephosphate-Isomerase-1-27kDa-Homodimer-"
            "Glycolysis-Step-5-DHAP-to-Glyceraldehyde-3-Phosphate-"
            "MOST-SEVERE-Glycolytic-Enzymopathy-CNSHA-Neurodegeneration-Fatal-Childhood-"
            "OMIM-Gene-190450-Disease-TPI1-Deficiency-615512"
        ),
        "locus": "12p13.31",
        "protein_size": (
            "286 aa / 27 kDa (subunit; active form is homodimer); catalyses glycolysis step 5: "
            "DHAP ↔ glyceraldehyde-3-phosphate (GAP) (reversible); "
            "aldolase (step 4) splits F-1,6-BP into DHAP + GAP; TPI1 converts DHAP → GAP; "
            "without TPI1: only one GAP molecule per glucose (50% glycolytic efficiency); "
            "DHAP accumulates — neurotoxic; "
            "most severe glycolytic enzymopathy in terms of neurological prognosis; "
            "Glu104Asp (Ashkenazi Jewish founder compound heterozygote) = less severe phenotype; "
            "homozygous severe mutations: death by 6 years without HSCT)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function) — TPI1 deficiency; "
            "MECHANISM: "
            "  TPI1 converts DHAP → GAP (the actual substrate for step 6, G3P dehydrogenase); "
            "  Aldolase (step 4) produces DHAP + GAP in equal amounts; "
            "  TPI1 LOF → DHAP cannot be converted to GAP → "
            "    → DHAP accumulates to toxic levels; "
            "    → only 1 of 2 GAP per glucose enters step 6 (50% reduction in glycolytic flux); "
            "DHAP ACCUMULATION — NEUROTOXIC: "
            "  DHAP is the most critical substrate accumulating; "
            "  DHAP methylglyoxal → advanced glycation end-products; "
            "  Mitochondrial toxicity from DHAP overload; "
            "  NEURONAL DEATH: DHAP-derived methylglyoxal → oxidative stress + protein glycation → "
            "    progressive neurodegeneration; "
            "SEVERITY (MOST SEVERE GLYCOLYTIC ENZYMOPATHY): "
            "  Spinocerebellar degeneration + CNSHA + neutrophil dysfunction; "
            "  Median age of death without HSCT: ~6 years (reported range 1-15 years); "
            "  HSCT cures haematological but NOT neurological component; "
            "Glu104Asp (p.Glu104Asp): "
            "  Most common mutation; compound heterozygous Ashkenazi; "
            "  Less severe than most homozygous; some survive to adulthood"
        ),
        "disease_category": (
            "TPI1 DEFICIENCY — CNSHA + NEURODEGENERATION + NEUTROPHIL DYSFUNCTION — OMIM 615512; "
            "TRIAD: "
            "  1. CNSHA (hemolytic anemia): "
            "     Hb 7-10 g/dL; reticulocytes 8-20%; LDH 400-800; bilirubin 40-100; "
            "     Splenomegaly 85%; gallstones 50%; "
            "  2. PROGRESSIVE NEURODEGENERATIVE DISEASE: "
            "     Spinocerebellar degeneration (onset infancy/early childhood); "
            "     Progressive: truncal hypotonia → limb spasticity → loss of ambulation; "
            "     Dysarthria, dysphagia; choreoathetosis in some; "
            "     Cognitive impairment; "
            "     MRI: progressive cerebellar atrophy, white matter changes; "
            "  3. NEUTROPHIL DYSFUNCTION: "
            "     Increased susceptibility to bacterial infections; "
            "     Neutrophil TPI1 activity reduced → phagocytic ATP depletion → impaired killing; "
            "NATURAL HISTORY WITHOUT TREATMENT: "
            "  Median death age ~6 years; respiratory failure from neurological deterioration; "
            "  Infections major cause of mortality"
        ),
        "disease_pathway": (
            "EMBDEN-MEYERHOF GLYCOLYSIS — TPI1 STEP 5 — DHAP TOXICITY + 50% FLUX REDUCTION: "
            "NORMAL TPI1: "
            "  Aldolase (step 4): F-1,6-BP → DHAP + GAP; "
            "  TPI1 (step 5): DHAP → GAP (rapid equilibrium; K_eq strongly favours GAP formation); "
            "  Both GAP molecules from one glucose proceed through steps 6-10 → 2x ATP generation; "
            "TPI1 LOF: "
            "  DHAP cannot be converted to GAP → DHAP accumulates dramatically; "
            "  One GAP molecule still generated by aldolase → 50% glycolytic flux preserved; "
            "  DHAP neurotoxicity pathway: "
            "    DHAP → methylglyoxal (non-enzymatic); "
            "    Methylglyoxal → glyoxal → oxidative stress → lipid peroxidation; "
            "    MGO reacts with proteins/DNA → advanced glycation end-products (AGEs); "
            "    AGE-RAGE signalling → neuroinflammation → neuronal apoptosis; "
            "ERYTHROCYTE EFFECT: "
            "  DHAP accumulation + 50% ATP depletion → RBC rigidity → hemolysis; "
            "BRAIN VULNERABILITY: "
            "  Neurons = high glucose consumers; "
            "  DHAP + reduced ATP → synergistic neurodegeneration; "
            "  Cerebellum particularly vulnerable (high metabolic demand; Purkinje cell ATP sensitivity)"
        ),
        "pathognomonic": (
            "CNSHA + PROGRESSIVE SPINOCEREBELLAR NEURODEGENERATION IN CHILDHOOD — "
            "MOST SEVERE GLYCOLYTIC ENZYMOPATHY: "
            "TPI ENZYME ACTIVITY DRAMATICALLY REDUCED: "
            "  TPI enzyme activity <5% of normal in RBCs AND leukocytes (not just erythrocytes); "
            "  TPI1 ubiquitously expressed — leukocyte TPI also low; "
            "  Distinguishes TPI1 from other glycolytic enzymopathies (most only reduced in RBCs); "
            "ERYTHROCYTE DHAP ELEVATION: "
            "  Markedly elevated DHAP in erythrocytes (RBC metabolomics); "
            "  Direct evidence of metabolic block at TPI1 step; "
            "SPINOCEREBELLAR DEGENERATION + HEMOLYTIC ANEMIA: "
            "  Combination pathognomonic for TPI1 in a child with CNSHA; "
            "  No other common glycolytic enzymopathy causes spinocerebellar degeneration; "
            "  DDx: ALDOA causes similar triad but with myopathy; GPI causes mild CNS in severe cases; "
            "FATAL CHILDHOOD COURSE WITHOUT HSCT: "
            "  Rapid neurological deterioration + CNSHA + recurrent infections; "
            "  Diagnosis of TPI1 → URGENT HSCT EVALUATION (haematological benefit confirmed; neuro timing critical); "
            "  HSCT DOES NOT CURE NEUROLOGICAL COMPONENT — brain TPI1 persists; "
            "  Rationale for HSCT: cure anemia, reduce infection risk, but neuro inexorable"
        ),
        "treatment": (
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT) — URGENTLY FOR HAEMATOLOGICAL COMPONENT: "
            "HSCT: "
            "  Curative for CNSHA and neutrophil dysfunction; "
            "  Does NOT halt or reverse neurodegeneration (neurons not replaced by HSCT); "
            "  TIMING CRITICAL: HSCT before major neurological deterioration preferred; "
            "  Sibling matched donor preferred; haploidentical accepted given severity; "
            "  Consider HSCT in all TPI1-confirmed infants with severe phenotype; "
            "SUPPORTIVE HEMOLYTIC ANEMIA: "
            "  Transfusion support; folic acid; "
            "  Splenectomy: limited benefit (neurological outcome not changed); "
            "NEUROLOGICAL MANAGEMENT (SUPPORTIVE ONLY): "
            "  Anti-epileptic drugs for seizures; "
            "  Physiotherapy, speech therapy; "
            "  Nutritional support (dysphagia); "
            "  Palliative planning for progressive course; "
            "INFECTION PROPHYLAXIS: "
            "  Pneumococcal, meningococcal, HiB vaccines; "
            "  Low threshold for antibiotics; "
            "  PCP prophylaxis in immunocompromised; "
            "GENE THERAPY: theoretical; very early research"
        ),
        "seed": 2843,
        "pt_vars": {
            "hb_range": (7.0, 10.0),
            "retic_range": (8.0, 20.0),
            "ldh_range": (400, 800),
            "bili_range": (40, 100),
            "splenomegaly_pct": 85,
            "gallstones_pct": 50,
            "aplastic_crisis_pct": 15,
            "neuro_pct": 95,
            "myopathy_pct": 0,
            "rhabdo_pct": 0,
            "transfusion_dependent_pct": 50,
            "splenectomy_pct": 30,
            "enzyme_activity_range": (1, 10),
        }
    },
    {
        "gene": "ALDOA",
        "protein": (
            "ALDOA -- 16p11.2 AR -- 364aa -- Aldolase-A-39kDa-"
            "Glycolysis-Step-4-Fructose-1,6-Bisphosphate-to-DHAP-plus-GAP-"
            "CNSHA-Myopathy-Intellectual-Disability-Triad-"
            "OMIM-Gene-103850-Disease-ALDOA-Deficiency-200350"
        ),
        "locus": "16p11.2",
        "protein_size": (
            "364 aa / 39 kDa (aldolase A tetramer; muscle/brain/erythrocyte isoform; "
            "catalyses glycolysis step 4: fructose-1,6-bisphosphate → DHAP + glyceraldehyde-3-phosphate; "
            "aldolase B = liver (hereditary fructose intolerance — different gene, different disease); "
            "aldolase C = brain-specific (provides partial compensation in neurons); "
            "ALDOA LOF → CNSHA + myopathy + intellectual disability; "
            "CK elevated (myopathy); EMG myopathic; p.Arg303His most common; "
            "very rare — fewer than 30 documented cases)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function) — ALDOA deficiency; "
            "MECHANISM: "
            "  ALDOA catalyses step 4: F-1,6-BP → DHAP + GAP; "
            "  This is the first carbon-splitting step in glycolysis; "
            "  ALDOA LOF → F-1,6-BP accumulates; DHAP and GAP deficient; "
            "  ATP generation downstream (steps 7+10) severely reduced; "
            "ISOFORM SPECIFICITY: "
            "  ALDOA: expressed in muscle, brain, erythrocytes — ALL THREE affected; "
            "  ALDOB: liver (hereditary fructose intolerance) — not involved in ALDOA deficiency; "
            "  ALDOC: brain-specific — provides partial compensation in neurons → "
            "    brain less severely affected than muscle/RBCs; "
            "  Explains: myopathy + CNSHA most severe; intellectual disability variable/moderate; "
            "SIMILAR TO TPI1 TRIAD BUT WITH MYOPATHY (NOT NEURODEGENERATION): "
            "  TPI1: CNSHA + neurodegeneration + infection susceptibility; "
            "  ALDOA: CNSHA + myopathy (prominent) + intellectual disability (moderate); "
            "  Both: AR; both very rare; distinguished by enzyme assay + ERK/CK"
        ),
        "disease_category": (
            "ALDOA DEFICIENCY — CNSHA + MYOPATHY + INTELLECTUAL DISABILITY — OMIM 200350; "
            "TRIAD: "
            "  1. CNSHA: "
            "     Hb 8-11 g/dL; reticulocytes 6-18%; LDH 400-750; bilirubin 35-90; "
            "     Splenomegaly 75%; gallstones 50%; "
            "  2. MYOPATHY (EXERTIONAL): "
            "     Exercise intolerance; exertional myalgia; "
            "     RHABDOMYOLYSIS: acute episodes after strenuous activity → myoglobinuria; "
            "     CK markedly elevated (myopathic, especially after exercise); "
            "     EMG: myopathic pattern; muscle biopsy: fibre atrophy; "
            "     ALDOA absent in muscle on immunohistochemistry; "
            "  3. INTELLECTUAL DISABILITY / PSYCHOMOTOR RETARDATION: "
            "     Variable severity (mild to moderate); "
            "     Some developmental delay; "
            "     Partial compensation from ALDOC in neurons; "
            "     Not progressive neurodegeneration (unlike TPI1); "
            "HAEMOLYTIC FEATURES dominate clinical picture in most cases; "
            "NEONATAL PRESENTATION: jaundice, severe anemia"
        ),
        "disease_pathway": (
            "EMBDEN-MEYERHOF GLYCOLYSIS — ALDOLASE A STEP 4 — CARBON SPLITTING FAILURE: "
            "NORMAL ALDOA: "
            "  F-1,6-BP → DHAP + GAP (aldol cleavage); "
            "  Both 3-carbon products enter lower glycolysis (via TPI1 and step 6 respectively); "
            "  2 DHAP/GAP pairs per glucose → feeds steps 6-10 for full ATP yield; "
            "ALDOA LOF: "
            "  F-1,6-BP accumulates upstream; DHAP + GAP severely deficient; "
            "  Steps 6-10 receive minimal substrate → ATP generation fails; "
            "  Accumulation of F-1,6-BP (does not have same neurotoxicity as DHAP in TPI1); "
            "MUSCLE-SPECIFIC MECHANISM: "
            "  Skeletal muscle = primarily ALDOA-expressing; no isoform backup; "
            "  Exercise → high glycolytic demand → ALDOA failure → rapid ATP depletion → "
            "    → myosin ATPase pump failure → Ca2+ influx → muscle fibre destruction → rhabdomyolysis; "
            "BRAIN MECHANISM: "
            "  ALDOC partially compensates in neurons → milder ID vs myopathy; "
            "  F-1,6-BP accumulation → some fructose-1,6-bisphosphatase activity (gluconeogenesis direction) → "
            "    → indirect mitochondrial substrate availability → partial compensation; "
            "ERYTHROCYTE: "
            "  No ALDOC/ALDOB backup in RBCs → full ALDOA deficiency → ATP depletion → CNSHA"
        ),
        "pathognomonic": (
            "CNSHA + EXERTIONAL MYOPATHY + RHABDOMYOLYSIS + INTELLECTUAL DISABILITY — "
            "TRIAD SIMILAR TO PGK1 BUT AUTOSOMAL RECESSIVE: "
            "ALDOA ENZYME ASSAY IN RBCs AND MUSCLE: "
            "  Markedly reduced/absent ALDOA in erythrocytes AND muscle; "
            "  ALDOB (liver) and ALDOC (brain) unaffected — measured on specific substrate panels; "
            "  RBC enzyme panel: ALDOA assay low → diagnosis; "
            "CK ELEVATION + MYOPATHIC EMG + RHABDOMYOLYSIS: "
            "  CK >1000 U/L at rest; >50,000 after exercise; "
            "  Myoglobinuria in rhabdomyolysis episodes; "
            "DDx PGK1 (both: CNSHA + myopathy + CNS, X-linked vs AR): "
            "  PGK1: X-linked; PGK1 enzyme low; p.Arg206Pro/Asp268Asn; "
            "  ALDOA: autosomal recessive; ALDOA enzyme low; p.Arg303His; "
            "  Both: exertional rhabdomyolysis; "
            "F-1,6-BP ACCUMULATION: "
            "  Quantitative metabolomics: F-1,6-BP elevated in RBCs; "
            "  Specific for ALDOA (and upstream PFKM deficiency — but PFKM has different phenotype); "
            "MUSCLE BIOPSY ALDOA IMMUNOSTAINING: "
            "  Absent ALDOA immunostaining on immunohistochemistry — specific and diagnostic"
        ),
        "treatment": (
            "SUPPORTIVE MANAGEMENT — NO CURATIVE APPROVED THERAPY: "
            "HEMOLYTIC ANEMIA: "
            "  Transfusion for severe anemia; "
            "  Folic acid; "
            "  Splenectomy for transfusion-dependent patients; "
            "MYOPATHY / RHABDOMYOLYSIS PREVENTION: "
            "  Strict avoidance of strenuous exercise; "
            "  Gradual graded activity programmes; "
            "  Pre-exercise glucose/carbohydrate supplementation; "
            "  ACUTE RHABDOMYOLYSIS: "
            "    IV aggressive hydration; urinary alkalinisation; renal monitoring; "
            "    ICU management if oliguric; "
            "INTELLECTUAL DISABILITY: "
            "  Early educational intervention; occupational therapy; "
            "  Speech and language therapy; "
            "  Behavioural support; "
            "IRON CHELATION: "
            "  For iron overload from transfusions; "
            "HSCT: "
            "  Curative for haematological component; "
            "  Limited data in ALDOA; "
            "  Does not address myopathy or CNS (ALDOA persists in those tissues)"
        ),
        "seed": 2844,
        "pt_vars": {
            "hb_range": (8.0, 11.0),
            "retic_range": (6.0, 18.0),
            "ldh_range": (400, 750),
            "bili_range": (35, 90),
            "splenomegaly_pct": 75,
            "gallstones_pct": 50,
            "aplastic_crisis_pct": 10,
            "neuro_pct": 70,
            "myopathy_pct": 85,
            "rhabdo_pct": 45,
            "transfusion_dependent_pct": 30,
            "splenectomy_pct": 35,
            "enzyme_activity_range": (5, 20),
        }
    },
    {
        "gene": "PFKM",
        "protein": (
            "PFKM -- 12q13.11 AR -- 780aa -- Phosphofructokinase-M-Subunit-85kDa-"
            "Glycolysis-Step-3-Fructose-6-Phosphate-to-Fructose-1,6-Bisphosphate-"
            "Tarui-Disease-GSD-Type-VII-CNSHA-Myopathy-Hyperuricemia-No-Second-Wind-"
            "OMIM-Gene-610681-Disease-GSD7-232800"
        ),
        "locus": "12q13.11",
        "protein_size": (
            "780 aa / 85 kDa (M-subunit of phosphofructokinase; PFK is a tetramer; "
            "muscle PFK = M4 tetramer; RBC PFK = mixed M and L subunit tetramers (M2L2, ML3, L4); "
            "liver PFK = L4 tetramer; "
            "PFKM LOF: muscle PFK = M4 → complete loss; RBC PFK = only M2L2/ML3 lost, L4 preserved; "
            "→ PARTIAL RBC PFK deficiency (50-75% residual L-tetramer activity in RBCs); "
            "→ MILD compensated hemolysis in RBCs vs SEVERE myopathy in muscle; "
            "Ashkenazi founders: p.Gln7* and p.Arg232*; "
            "Tarui disease = GSD type VII)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function PFKM) — Tarui disease / GSD VII; "
            "MECHANISM: "
            "  PFK catalyses step 3: F6P + ATP → F-1,6-BP + ADP (key regulatory step, irreversible); "
            "  PFK is the MAJOR allosteric rate-limiting regulatory enzyme of glycolysis; "
            "  PFKM encodes the muscle M-subunit; PFKL = liver L-subunit; "
            "TISSUE-SPECIFIC PHENOTYPE: "
            "  MUSCLE (M4 tetramer): "
            "    PFKM LOF → complete absence of PFK in muscle → severe glycolytic block → myopathy; "
            "  ERYTHROCYTES (mixed M/L tetramers): "
            "    M-subunit lost → M2L2, ML3 tetramers absent → only L4 remains; "
            "    L4 tetramer has ~50% residual activity → partial PFK in RBCs; "
            "    MILD COMPENSATED HEMOLYSIS — not severe CNSHA; "
            "  RESULT: muscle phenotype >> RBC phenotype in Tarui disease; "
            "HYPERURICEMIA/GOUT: "
            "  Exercise → ischemic muscle → purine nucleotide breakdown → uric acid release; "
            "  DISTINCTIVE FEATURE of Tarui disease (not seen in McArdle/GSD5)"
        ),
        "disease_category": (
            "TARUI DISEASE / GSD TYPE VII — CNSHA (MILD) + MYOPATHY + HYPERURICEMIA — OMIM 232800; "
            "TRIAD: "
            "  1. MYOPATHY (DOMINANT FEATURE): "
            "     Exercise intolerance, painful muscle cramps; "
            "     NO SECOND WIND (PATHOGNOMONIC DDx from McArdle/GSD5); "
            "     RHABDOMYOLYSIS after intense exercise (myoglobinuria, CK elevation); "
            "     Fixed weakness: proximal in some patients (older adults); "
            "  2. CNSHA (MILD — COMPENSATED): "
            "     Hb 10-14 g/dL (compensated, often near-normal); "
            "     Reticulocytes 5-15%; LDH 350-700; bilirubin 30-80; "
            "     Splenomegaly 60%; gallstones 50%; "
            "     Mild elevation unconjugated bilirubin at baseline; "
            "  3. HYPERURICEMIA / GOUT: "
            "     Elevated serum uric acid from ischemic muscle purine release; "
            "     Gout attacks possible; "
            "     NOT seen in McArdle disease (useful DDx); "
            "ISCHEMIC FOREARM EXERCISE TEST: "
            "  NO rise in venous lactate after ischemic forearm exercise; "
            "  Ammonia RISES normally; "
            "  (McArdle/GSD5 also has no lactate rise → need enzyme assays to distinguish)"
        ),
        "disease_pathway": (
            "EMBDEN-MEYERHOF GLYCOLYSIS — PFK STEP 3 — KEY REGULATORY BLOCK + ISOFORM-SPECIFIC SEVERITY: "
            "NORMAL PFK FUNCTION: "
            "  Step 3 (key committed regulatory step): F6P + ATP → F-1,6-BP + ADP; "
            "  PFK is allosterically inhibited by: ATP, citrate (high energy state); "
            "  PFK is allosterically activated by: AMP, ADP, fructose-2,6-bisphosphate (F2,6BP); "
            "  Regulates glycolytic flux to match energy demand; "
            "PFKM LOF IN MUSCLE: "
            "  M4 tetramer absent → NO residual PFK activity in skeletal muscle; "
            "  F6P cannot be converted to F-1,6-BP → complete glycolytic block; "
            "  Muscle relies exclusively on glycolysis during intense exercise → "
            "    → no ATP from glycolysis → immediate exhaustion, cramps, rhabdomyolysis; "
            "NO SECOND WIND MECHANISM (PATHOGNOMONIC DDx): "
            "  McArdle (GSD5, PYGM): blocked at glycogenolysis → "
            "    → after 10min: fatty acid oxidation + FFA mobilisation kicks in → SECOND WIND (exercise tolerance improves); "
            "  Tarui (PFKM): blocked at PFK (downstream of glycogenolysis) → "
            "    → even if glycogen is mobilised, glucose-6-phosphate cannot pass step 3; "
            "    → NO second wind (glycolysis permanently blocked regardless of substrate availability); "
            "PFKM LOF IN ERYTHROCYTES (PARTIAL): "
            "  M-subunit lost; L4 tetramers preserved → ~50% RBC PFK activity; "
            "  Mild hemolysis (compensated) + mild reticulocytosis; "
            "HYPERURICEMIA MECHANISM: "
            "  Exercise → ischemic ATP breakdown → AMP → adenosine → inosine → hypoxanthine → xanthine → uric acid; "
            "  PFKM: more ischemia per exercise bout → more purine turnover → hyperuricemia"
        ),
        "pathognomonic": (
            "MYOPATHY + NO SECOND WIND + MILD COMPENSATED HEMOLYSIS + HYPERURICEMIA — "
            "PATHOGNOMONIC TARUI DISEASE (GSD VII) CONSTELLATION: "
            "NO SECOND WIND (KEY DDx FROM McARDLE GSD5): "
            "  McArdle (PYGM): exercise → cramp → REST 5-10 min → second wind → exercise tolerance improves; "
            "  Tarui (PFKM): exercise → cramp → REST → NO second wind; "
            "  Clinical test: standardised 12-minute exercise test → Tarui never recovers to second wind; "
            "ISCHEMIC FOREARM EXERCISE TEST: "
            "  NO venous lactate rise after ischemic forearm exercise; "
            "  Ammonia RISES (purine release confirms exercise was adequate); "
            "  McArdle also has no lactate rise → further differentiation requires: "
            "    aldolase, LDH, phosphoglycerate mutase isoform testing; "
            "    PFKM: aldolase + LDH also fail (downstream of PFK block); "
            "    McArdle: aldolase + LDH normal (block is glycogenolysis, not glycolysis itself); "
            "PARTIAL RBC PFK DEFICIENCY (30-50% ACTIVITY): "
            "  Mild not severe hemolysis; "
            "  Compensated with modest reticulocytosis; "
            "  UNIQUE: no other severe myopathy has concurrent hemolysis — Tarui signature; "
            "HYPERURICEMIA + GOUT IN YOUNG ATHLETE WITH EXERCISE INTOLERANCE: "
            "  Gout-like joint symptoms + myopathy → Tarui disease in differential; "
            "ASHKENAZI JEWISH ANCESTRY: "
            "  p.Gln7* and p.Arg232* founders → ethnic-specific founders for Tarui"
        ),
        "treatment": (
            "NO SPECIFIC CURATIVE THERAPY — PRIMARILY AVOIDANCE + SUPPORTIVE: "
            "MYOPATHY MANAGEMENT: "
            "  AVOID: intense anaerobic exercise, isometric loading; "
            "  SAFE: moderate aerobic exercise (fatty acid oxidation pathway unaffected); "
            "  Pre-exercise sucrose (sucrose → glucose + fructose → fructose bypasses PFK via aldose pathway) — "
            "    WORSENS exercise tolerance (sucrose loading paradox in Tarui — opposite of McArdle); "
            "    AVOID sucrose before exercise in Tarui; "
            "  Fructose + glucose supplementation NOT beneficial (cannot pass PFK block); "
            "RHABDOMYOLYSIS MANAGEMENT: "
            "  IV hydration; urine alkalinisation; renal monitoring; "
            "  Acute kidney injury risk; ICU management if severe; "
            "HYPERURICEMIA / GOUT: "
            "  Allopurinol (XO inhibitor) for hyperuricemia/gout prophylaxis; "
            "  Colchicine for acute gout attacks; "
            "  Low-purine diet; "
            "HEMOLYTIC ANEMIA: "
            "  Folic acid supplementation; "
            "  Rarely requires transfusion (compensated in most); "
            "  Splenectomy: not generally indicated (mild hemolysis); "
            "SUCROSE LOADING PARADOX: "
            "  McArdle: sucrose pre-exercise IMPROVES tolerance; "
            "  Tarui: sucrose pre-exercise WORSENS tolerance (sucrose → fructose → fructose-1-phosphate → "
            "    traps available phosphate → less ATP available → worse ischemia) — EDUCATE PATIENTS"
        ),
        "seed": 2845,
        "pt_vars": {
            "hb_range": (10.0, 14.0),
            "retic_range": (5.0, 15.0),
            "ldh_range": (350, 700),
            "bili_range": (30, 80),
            "splenomegaly_pct": 60,
            "gallstones_pct": 50,
            "aplastic_crisis_pct": 5,
            "neuro_pct": 5,
            "myopathy_pct": 100,
            "rhabdo_pct": 70,
            "transfusion_dependent_pct": 10,
            "splenectomy_pct": 15,
            "enzyme_activity_range": (30, 50),
        }
    },
]

DEFINITIONS = {
    "definitions": [
        {
            "term": "CNSHA (Chronic Non-Spherocytic Hemolytic Anemia)",
            "definition": (
                "Hereditary hemolytic anemia without spherocytes on blood film; "
                "caused by erythrocyte enzyme deficiencies (PKLR, HK1, GPI, PGK1, TPI1, ALDOA, PFKM) "
                "or membrane disorders; "
                "distinguished from hereditary spherocytosis (spectrin/ankyrin defects, spherocytes present); "
                "features: chronic jaundice, splenomegaly, gallstones, reticulocytosis; "
                "diagnosis requires RBC enzyme panel (not blood film morphology alone); "
                "severity: episodic (G6PD class II-III) to severe CNSHA (TPI1, HK1) to fatal childhood (TPI1)"
            )
        },
        {
            "term": "Favism (G6PD-specific trigger)",
            "definition": (
                "Acute hemolytic crisis specifically triggered by ingestion of fava beans (Vicia faba); "
                "UNIQUE TO G6PD DEFICIENCY (no other red cell enzymopathy); "
                "mechanism: fava bean glycosides vicine and convicine → intestinal hydrolysis → "
                "divicine + isouramil → directly generate H2O2 and reactive oxygen species → "
                "overwhelm residual G6PD NADPH → glutathione depletion → haemoglobin oxidation → Heinz bodies; "
                "onset: 24-72 hours after ingestion; "
                "severity: Mediterranean variant (class II) most severe; "
                "management: avoid fava beans; supportive transfusion if Hb drops severely"
            )
        },
        {
            "term": "2,3-DPG Paradox (PKLR-specific metabolic hallmark)",
            "definition": (
                "In PKLR deficiency: PEP cannot be converted to pyruvate → PEP accumulates → "
                "Rapoport-Luebering shunt: 1,3-BPG → 2,3-DPG increases markedly; "
                "2,3-DPG binds β-haemoglobin subunit in T-state → stabilises deoxy-Hb → "
                "right-shifts O2-Hb dissociation curve; "
                "RIGHT-SHIFT: haemoglobin releases O2 to tissues at lower pO2; "
                "CLINICAL PARADOX: PK-deficient patients tolerate haemoglobin levels "
                "(e.g. Hb 7 g/dL) that would cause severe symptoms in other anemias; "
                "DDx tool: 2,3-DPG ELEVATED = PKLR; 2,3-DPG LOW = HK1 (block before 1,3-BPG); "
                "Post-splenectomy: 2,3-DPG paradox partially lost → patients may feel worse despite higher Hb"
            )
        },
        {
            "term": "Aplastic Crisis (Parvovirus B19)",
            "definition": (
                "Acute temporary arrest of erythropoiesis caused by Parvovirus B19 infection; "
                "B19 infects erythroid progenitors (globoside receptor on erythroid precursors) → "
                "cytopathic effect → cessation of RBC production for ~7-10 days; "
                "in normal individuals: unnoticed (RBC lifespan 120 days, 7-day gap minor); "
                "in CNSHA patients (RBC lifespan 7-30 days): 7-10 days of zero production → "
                "catastrophic Hb fall → life-threatening emergency; "
                "features: sudden severe anemia, reticulocytopenia (hallmark), fever, rash in some; "
                "diagnosis: B19 IgM serology + PCR; "
                "management: transfusion bridge; self-limited ~10 days in immunocompetent; "
                "CNSHA patients need annual flu vaccine + prompt infection treatment"
            )
        },
        {
            "term": "Heinz Bodies (G6PD-specific oxidative haemoglobin denaturation)",
            "definition": (
                "Intracellular inclusions of oxidised, denatured, precipitated haemoglobin in erythrocytes; "
                "mechanism: oxidative stress (H2O2) → haemoglobin oxidation → methemoglobin → "
                "irreversible globin chain cross-linking → precipitates as Heinz body; "
                "SPECIFIC TO G6PD DEFICIENCY and other oxidative stress states "
                "(congenital Heinz body hemolytic anemia — unstable haemoglobins also cause Heinz bodies); "
                "detection: supravital stain (brilliant cresyl blue) — Romanowsky stains CANNOT show Heinz bodies; "
                "bite cells: Heinz bodies removed by spleen → leave a 'bite mark' on Romanowsky stain; "
                "timing: Heinz bodies appear DURING hemolytic crisis; disappear in recovery (affected cells hemolysed)"
            )
        },
        {
            "term": "Ischemic Forearm Exercise Test (Tarui PFKM diagnosis)",
            "definition": (
                "Diagnostic test for glycogen storage diseases affecting muscle glycolysis; "
                "protocol: forearm ischemia (inflated BP cuff) + repeated handgrip exercise → "
                "release cuff → venous blood sampled at 1, 3, 5, 10 minutes post-exercise; "
                "NORMAL response: venous LACTATE rises ≥3-fold (anaerobic glycolysis produces lactate); "
                "PFKM (Tarui) or GSD5 (McArdle/PYGM): NO lactate rise (glycolytic block); "
                "AMMONIA: rises normally in all (purine breakdown marker confirms exercise was adequate); "
                "DDx Tarui from McArdle: "
                "  further isoenzyme analysis needed (aldolase, LDH, PGM activity); "
                "  Tarui: aldolase, LDH also fail (downstream of PFK); "
                "  McArdle: aldolase, LDH normal (block at glycogenolysis, not glycolysis); "
                "Non-ischemic version: safer; adequate for most screening"
            )
        },
        {
            "term": "G6PD Enzyme Assay Timing (critical diagnostic trap)",
            "definition": (
                "G6PD enzyme assay MUST be performed ≥4-6 weeks AFTER hemolytic crisis has resolved; "
                "DURING crisis: haemolytic episode destroys oldest G6PD-deficient RBCs preferentially; "
                "surviving cells = youngest (reticulocytes) → reticulocytes have highest G6PD activity "
                "(young cells have higher enzyme levels regardless of genotype); "
                "result: G6PD enzyme assay performed DURING crisis may be falsely NORMAL; "
                "DIAGNOSTIC TRAP: normal G6PD assay during crisis DOES NOT exclude G6PD deficiency; "
                "solutions: "
                "  1. Repeat assay after full recovery (4-6 weeks); "
                "  2. Fluorescent spot test (screening) same limitation; "
                "  3. MOLECULAR ANALYSIS (gene sequencing) not affected by reticulocyte surge → "
                "     gold standard for diagnosis during acute crisis or female carrier testing"
            )
        },
        {
            "term": "No Second Wind Phenomenon (PFKM/Tarui disease DDx from McArdle)",
            "definition": (
                "Second wind: phenomenon in McArdle disease (GSD V, PYGM deficiency) where "
                "after ~10 minutes of exercise, fatty acid oxidation supplements glycolytic failure → "
                "exercise tolerance spontaneously improves (second wind); "
                "mechanism: fatty acid mobilisation + free fatty acid oxidation in OXPHOS bypasses "
                "glycogenolysis block; "
                "TARUI DISEASE (PFKM LOF): NO second wind; "
                "mechanism: PFK blocks glycolysis DOWNSTREAM of glycogenolysis; "
                "even if glycogen → glucose-6-phosphate (via phosphorylase/PGM), "
                "glucose-6-phosphate cannot pass PFK step → no second wind; "
                "clinical test: standardised exercise test + monitor for second wind; "
                "CRITICAL DDx: Tarui = no second wind; McArdle = second wind present; "
                "both diseases: no venous lactate rise on forearm ischemia test"
            )
        },
        {
            "term": "DHAP Neurotoxicity (TPI1 pathomechanism)",
            "definition": (
                "DHAP = dihydroxyacetone phosphate; normal glycolytic intermediate; "
                "TPI1 normally converts DHAP → GAP (glyceraldehyde-3-phosphate) rapidly; "
                "TPI1 LOF: DHAP accumulates dramatically in all TPI1-expressing cells; "
                "DHAP neurotoxicity mechanism: "
                "  DHAP → methylglyoxal (non-enzymatic fragmentation); "
                "  methylglyoxal → reactive advanced glycation end-products (AGEs); "
                "  AGE-RAGE receptor signalling → neuroinflammation; "
                "  mitochondrial toxicity from DHAP overload → ATP deficiency in neurons; "
                "  protein glycation → impaired protein function; "
                "consequence: progressive spinocerebellar neurodegeneration; "
                "LEUKOCYTE TPI also reduced (unlike other glycolytic enzymopathies where only RBC-specific); "
                "HSCT cures haematological component but cannot reverse established neurodegeneration"
            )
        },
        {
            "term": "Mitapivat (AG-348) — FDA 2022 Pyruvate Kinase Activator",
            "definition": (
                "Mitapivat (Pyrukynd, AG-348): first approved disease-modifying oral therapy for PKLR deficiency; "
                "FDA approved August 2022 for non-transfusion-dependent PK deficiency in adults; "
                "mechanism: allosteric activator binding to PK-R dimer interface → "
                "activates residual mutant PK-R enzyme in heterozygous/compound heterozygous patients; "
                "effects: increased pyruvate + ATP production → reduced 2,3-DPG → "
                "left-shift O2 curve correction → rising Hb (mean +1.5-2 g/dL in trials); "
                "reduced need for transfusion in non-transfusion-dependent patients; "
                "clinical trials: ACTIVATE and ACTIVATE-T trials; "
                "note: activates residual enzyme → requires at least partial residual PK-R activity; "
                "null/homozygous severe mutations: less benefit; "
                "trials ongoing for transfusion-dependent and paediatric populations"
            )
        },
    ],
    "standards": [
        "Bolton-Maggs PHB et al. Guidelines for the diagnosis and management of hereditary spherocytosis. Br J Haematol. 2012",
        "Beutler E. G6PD Deficiency. Blood. 1994; WHO Working Group G6PD Deficiency Classification 1989",
        "Grace RF et al. Clinical spectrum of pyruvate kinase deficiency: data from the Pyruvate Kinase Deficiency Natural History Study. Blood. 2018",
        "Fermo E, Vercellati C, Marcello AP et al. Red cell enzymopathies. Int J Lab Hematol. 2020",
        "Zanella A, Fermo E, Bianchi P, Valentini G. Red cell pyruvate kinase deficiency: molecular and clinical aspects. Br J Haematol. 2005",
        "OMIM: G6PD 305900, PKLR 609712/266200, HK1 142600/235700, GPI 172400/613470, PGK1 311800/300653, TPI1 190450/615512, ALDOA 103850/200350, PFKM 610681/232800",
        "Tarui S et al. Phosphofructokinase deficiency in skeletal muscle. Biochem Biophys Res Commun. 1965",
        "Gnerer JP et al. Wasting away in Wonderland: TPI deficiency. Genetics. 2006",
        "Luzzatto L et al. Glucose-6-phosphate dehydrogenase deficiency. N Engl J Med. 2020",
        "Glader B. Hereditary hemolytic anemias due to red cell enzyme disorders. In: Greer's Wintrobe's Clinical Hematology. 2019",
    ]
}


def _make_patients(gene_data):
    rng = random.Random(gene_data["seed"])
    pts = []
    pv = gene_data["pt_vars"]

    hb_lo, hb_hi = pv["hb_range"]
    retic_lo, retic_hi = pv["retic_range"]
    ldh_lo, ldh_hi = pv["ldh_range"]
    bili_lo, bili_hi = pv["bili_range"]
    ea_lo, ea_hi = pv["enzyme_activity_range"]

    for i in range(40):
        hb = round(rng.uniform(hb_lo, hb_hi), 1)
        retic = round(rng.uniform(retic_lo, retic_hi), 1)
        ldh = round(rng.uniform(ldh_lo, ldh_hi), 0)
        bili = round(rng.uniform(bili_lo, bili_hi), 1)
        enzyme_activity = round(rng.uniform(ea_lo, ea_hi), 1)
        age_diag = round(rng.uniform(0.0, 20.0), 1)
        sex = rng.choice(["M", "F"])

        splenomegaly = rng.random() < pv["splenomegaly_pct"] / 100
        gallstones = rng.random() < pv["gallstones_pct"] / 100
        aplastic_crisis = rng.random() < pv["aplastic_crisis_pct"] / 100
        neuro = rng.random() < pv["neuro_pct"] / 100
        myopathy = rng.random() < pv["myopathy_pct"] / 100
        rhabdo = rng.random() < pv["rhabdo_pct"] / 100
        transfusion_dependent = rng.random() < pv["transfusion_dependent_pct"] / 100
        splenectomy = rng.random() < pv["splenectomy_pct"] / 100

        pts.append({
            "patient_id": f"{gene_data['gene']}-{i+1:03d}",
            "gene": gene_data["gene"],
            "sex": sex,
            "age_at_diagnosis_years": age_diag,
            "hemoglobin_g_dl": hb,
            "reticulocyte_pct": retic,
            "ldh_u_l": ldh,
            "bilirubin_umol_l": bili,
            "enzyme_activity_pct_normal": enzyme_activity,
            "splenomegaly": splenomegaly,
            "gallstones": gallstones,
            "aplastic_crisis": aplastic_crisis,
            "neuro_involvement": neuro,
            "myopathy": myopathy,
            "rhabdomyolysis": rhabdo,
            "transfusion_dependent": transfusion_dependent,
            "splenectomy": splenectomy,
            "seed": gene_data["seed"],
        })
    return pts


def generate_overview():
    total_patients = 0
    all_genes = []

    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        total_patients += len(patients)

        splenomegaly_n = sum(1 for p in patients if p["splenomegaly"])
        neuro_n = sum(1 for p in patients if p["neuro_involvement"])
        myopathy_n = sum(1 for p in patients if p["myopathy"])
        rhabdo_n = sum(1 for p in patients if p["rhabdomyolysis"])
        transfusion_n = sum(1 for p in patients if p["transfusion_dependent"])
        aplastic_n = sum(1 for p in patients if p["aplastic_crisis"])

        all_genes.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "n_patients": len(patients),
            "median_hb_g_dl": round(
                sorted(p["hemoglobin_g_dl"] for p in patients)[len(patients) // 2], 1
            ),
            "mean_retic_pct": round(
                sum(p["reticulocyte_pct"] for p in patients) / len(patients), 1
            ),
            "mean_ldh_u_l": round(
                sum(p["ldh_u_l"] for p in patients) / len(patients), 0
            ),
            "pct_splenomegaly": round(splenomegaly_n / len(patients) * 100, 1),
            "pct_neuro": round(neuro_n / len(patients) * 100, 1),
            "pct_myopathy": round(myopathy_n / len(patients) * 100, 1),
            "pct_rhabdo": round(rhabdo_n / len(patients) * 100, 1),
            "pct_transfusion_dependent": round(transfusion_n / len(patients) * 100, 1),
            "pct_aplastic_crisis": round(aplastic_n / len(patients) * 100, 1),
            "mean_enzyme_activity_pct": round(
                sum(p["enzyme_activity_pct_normal"] for p in patients) / len(patients), 1
            ),
            "seed": gene_data["seed"],
        })

    return {
        "atlas": "Hereditary Red Cell Enzymopathy Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Red Cell Enzymopathy Reference — "
            "G6PD·PKLR·HK1·GPI·PGK1·TPI1·ALDOA·PFKM"
        ),
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "gene_summaries": all_genes,
        "seeds": "2838-2845",
        "pathway_categories": [
            {
                "pathway": "Pentose Phosphate Pathway — Oxidative Defence (G6PD)",
                "genes": ["G6PD"],
                "note": (
                    "G6PD = PPP rate-limiting step; NADPH → glutathione → H2O2 scavenging; "
                    "most common human enzymopathy (~500 million worldwide); "
                    "XLR; episodic hemolysis (class II-III) or CNSHA (class I); "
                    "favism (fava beans) pathognomonic trigger; Heinz bodies on supravital stain; "
                    "CRITICAL: enzyme assay after crisis, not during (reticulocyte surge falsely normalises result)"
                ),
            },
            {
                "pathway": "Embden-Meyerhof Glycolysis — Upper Steps + Phosphotransfer (PKLR, HK1, PGK1)",
                "genes": ["PKLR", "HK1", "PGK1"],
                "note": (
                    "PKLR (step 10): most common CNSHA; 2,3-DPG elevated → right-shift → tolerates low Hb; "
                    "mitapivat FDA 2022; echinocytes; R479W most common; "
                    "HK1 (step 1): severe CNSHA; 2,3-DPG LOW (opposite of PKLR — key DDx); "
                    "PGK1 (step 7): XLR; UNIQUE TRIAD: CNSHA + myopathy + CNS; "
                    "only glycolytic enzyme deficiency with all three systems; exertional rhabdomyolysis"
                ),
            },
            {
                "pathway": "Embden-Meyerhof Glycolysis — Central Steps (GPI, TPI1, ALDOA, PFKM)",
                "genes": ["GPI", "TPI1", "ALDOA", "PFKM"],
                "note": (
                    "GPI (step 2): 2nd most common non-spherocytic CNSHA after PKLR; French p.Arg347His founder; "
                    "TPI1 (step 5): most SEVERE glycolytic enzymopathy; DHAP neurotoxic accumulation; "
                    "  spinocerebellar neurodegeneration + CNSHA; fatal childhood without HSCT; "
                    "ALDOA (step 4): CNSHA + exertional myopathy + intellectual disability; ALDOA absent in muscle; "
                    "PFKM (step 3) = Tarui/GSD7: myopathy dominant + mild CNSHA + hyperuricemia; "
                    "  NO second wind (DDx McArdle); partial RBC PFK (L4 tetramers preserved)"
                ),
            },
        ],
        "critical_distinctions": [
            "G6PD vs OTHER ENZYMOPATHIES: G6PD = episodic (NOT chronic) for class II-III; only triggered by oxidant stress (fava beans, drugs, infection); 500M worldwide vs other enzymopathies rare; XLR vs AR for most others",
            "PKLR vs HK1 (both: CNSHA, splenomegaly, similar severity): PKLR = 2,3-DPG ELEVATED (right-shift, tolerate low Hb); HK1 = 2,3-DPG LOW (left-shift, poor O2 delivery); enzyme panel distinguishes definitively",
            "PKLR 2,3-DPG PARADOX: do NOT over-transfuse PK-deficient patients; 2,3-DPG elevation compensates for low Hb; aggressive transfusion suppresses erythropoiesis and 2,3-DPG advantage",
            "G6PD ASSAY TRAP: normal G6PD enzyme assay DURING crisis DOES NOT exclude G6PD deficiency; reticulocytes have normal G6PD; repeat assay 4-6 weeks after crisis OR use molecular testing",
            "PKLR SPLENECTOMY PARADOX: post-splenectomy Hb rises but patients may feel worse; 2,3-DPG effect lost with older circulating RBCs; counsel patients pre-splenectomy",
            "TPI1 vs ALDOA (both: CNSHA + multisystem): TPI1 = PROGRESSIVE NEURODEGENERATION (spinocerebellar) + neutrophil dysfunction, fatal childhood; ALDOA = myopathy + intellectual disability (not neurodegenerative); DHAP elevated in TPI1",
            "PGK1 vs ALDOA (both: CNSHA + myopathy + CNS): PGK1 = X-LINKED (males); ALDOA = AUTOSOMAL RECESSIVE; enzyme assay panel distinguishes; PGK1 p.Arg206Pro = CNS-dominant; PGK1 p.Asp268Asn = myopathy-dominant",
            "PFKM (TARUI) vs McARDLE (GSD5 PYGM): both = no venous lactate on forearm ischemia test; TARUI: NO SECOND WIND + hemolysis + hyperuricemia; McARDLE: SECOND WIND present + no hemolysis + no hyperuricemia; ischemic forearm test only screen — isoenzyme panel differentiates",
            "TPI1 HSCT: cures haematological component (anemia, infections) but DOES NOT halt neurodegeneration; HSCT timing critical — before major neurological deterioration for maximum benefit",
            "MITAPIVAT (PKLR ONLY): allosteric PK-R activator; requires residual PK-R enzyme activity; less benefit for null/homozygous severe mutations; raises Hb + reduces 2,3-DPG correction in non-transfusion-dependent adults",
        ],
    }


def generate_breakdown():
    result = []
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        result.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "n_patients": len(patients),
            "patients": patients[:5],
        })
    return {"genes": result, "total": len(ATLAS_GENES), "seeds": "2838-2845"}


def generate_definitions():
    return {
        "atlas": "Hereditary Red Cell Enzymopathy Atlas",
        "definitions": DEFINITIONS["definitions"],
        "standards": DEFINITIONS["standards"],
        "gene_count": len(ATLAS_GENES),
        "seeds": "2838-2845",
    }
