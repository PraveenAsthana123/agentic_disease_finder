#!/usr/bin/env python3
"""GSD-Atlas — Complete 10-Gene Glycogen Storage Disorder Atlas
G6PC · SLC37A4 · AGL · GBE1 · PYGM · PYGL · PFKM · PHKA2 · SLC2A2 · GYS2
400-patient aggregate cohort (10 × 40, seeds 868–877)

Glycogen Storage Disorders (GSD) facts:
  - Inherited defects in enzymes or transporters of glycogen synthesis, degradation, or regulation
  - Collectively affect ~1 in 20,000-40,000 individuals
  - Wide spectrum: from fatal neonatal forms (GBE1 classic) to essentially asymptomatic (Hartnup analogy: PYGL, PHKA2)
  - GSD Ia (G6PC) is the prototypical hepatic GSD: raw cornstarch is cornerstone therapy
  - NEVER give glucose boluses in GSD Ia/Ib (G6P trapped; worsens lactic acidosis)
  - GSD V (PYGM/McArdle): sucrose trick prevents exercise-induced myoglobinuria
  - GSD VII (PFKM/Tarui): high-carbohydrate PARADOX — carbs WORSEN exercise tolerance
  - GSD 0a (GYS2): OPPOSITE of all others — cannot STORE glycogen (not excess storage)

ATLAS SCOPE (10 classic GSD genes):
  Glucose-6-phosphatase system:
    G6PC      — G6Pase-alpha → GSD Ia / Von Gierke type a (17q21.31)
    SLC37A4   — G6P translocase → GSD Ib / Von Gierke type b (11q23.3)
  Glycogen debranching:
    AGL       — Amylo-1,6-glucosidase/4-alpha-glucanotransferase → GSD IIIa/b / Cori-Forbes (1p21.2)
  Glycogen branching:
    GBE1      — Glycogen branching enzyme → GSD IV / Andersen (3p12.2)
  Muscle glycogen phosphorylase:
    PYGM      — Myophosphorylase → GSD V / McArdle (11q13.1)
  Liver glycogen phosphorylase:
    PYGL      — Liver phosphorylase → GSD VI / Hers (14q22.1)
  Phosphofructokinase:
    PFKM      — PFK muscle isoform → GSD VII / Tarui (12q13.3)
  Phosphorylase kinase:
    PHKA2     — PhK alpha2 subunit (liver) → GSD IXa / X-linked (Xp22.13)
  GLUT2 glucose transporter:
    SLC2A2    — GLUT2 → GSD XI / Fanconi-Bickel (3q26.2)
  Glycogen synthase:
    GYS2      — Liver glycogen synthase → GSD 0a (12p12.1)

CRITICAL CLINICAL RULES:
  1. G6PC (GSD Ia): NEVER glucose boluses — glucose → G6P → trapped → worsens lactic acidosis;
     cornstarch 1-2g/kg q4-6h is cornerstone; NO fructose/galactose (worsen hypoglycemia);
     lactic acidosis is PRIMARY (not secondary); hepatocellular adenoma >90% by 30y → HCC 10-15%.
  2. SLC37A4 (GSD Ib): Same biochemistry as GSD Ia PLUS NEUTROPENIA (ANC <1500);
     G-CSF MANDATORY; IBD (Crohn's-like) unique to GSD Ib.
  3. AGL (GSD IIIa/b): HIGH PROTEIN DIET (25-30%) — different from GSD I;
     liver transplant does NOT improve muscle disease in IIIa (different enzymopathy source).
  4. GBE1 (GSD IV): polyglucosan (amylopectin-like) accumulation; liver Tx CURATIVE for hepatic form ONLY;
     adult polyglucosan body disease (APBD) = neuromuscular form — liver Tx NOT helpful.
  5. PYGM (McArdle, GSD V): McArdle SIGN — venous lactate FAILS to rise with ischemic exercise;
     sucrose trick (35g oral) before exercise prevents myoglobinuria; SECOND WIND at 7-10 min.
  6. PYGL (GSD VI): GENERALLY BENIGN — resolves spontaneously at puberty; cornstarch if needed.
  7. PFKM (GSD VII, Tarui): HEMOLYTIC ANEMIA in ALL patients (PFK-M in RBCs);
     HIGH CARBOHYDRATE PARADOX — carbs worsen exercise tolerance (G6P inhibits residual PFK).
  8. PHKA2 (GSD IXa): X-LINKED; generally benign; resolves in adulthood.
  9. SLC2A2 (GSD XI, Fanconi-Bickel): galactose-free + fructose-restricted diet;
     FANCONI SYNDROME — glucosuria with normoglycemia, aminoaciduria, phosphaturia, rickets.
  10. GYS2 (GSD 0a): UNIQUE — cannot STORE glycogen; fasting KETOTIC hypoglycemia + postprandial HYPERGLYCEMIA;
      NO glycogen accumulation (opposite of other GSDs); high protein diet + cornstarch for nocturnal HG.

COHORT: 10 × 40 = 400 patient slots (seeds 868–877; gene-specific seeds)
"""

import random

SEED_BASE = 868

# ── All 10 Glycogen Storage Disorder genes ────────────────────────────────────
GSD_GENES = [
    # ── G6PC — Glucose-6-Phosphatase Catalytic Subunit (GSD Ia / Von Gierke type a) ──
    {
        "gene": "G6PC", "alias": "G6PC — Glucose-6-Phosphatase Catalytic Subunit (GSD Ia / Von Gierke type a)",
        "aa": "357 aa", "kDa": "36.5 kDa",
        "gene_class": "phosphatase",
        "locus": "17q21.31", "omim_gene": 613742,
        "phenotype": "GSD Ia — hepatomegaly, doll face, fasting hypoglycemia, hyperlactatemia, hypertriglyceridemia, hyperuricemia; adenoma risk",
        "disease": (
            "G6PC biallelic loss → Glycogen Storage Disease type Ia (GSD Ia, Von Gierke disease type a, OMIM #232200). "
            "G6PC encodes glucose-6-phosphatase catalytic subunit alpha (G6Pase-α, 357aa, 36.5 kDa), the endoplasmic "
            "reticulum (ER) lumen-facing phosphatase that dephosphorylates glucose-6-phosphate (G6P) → free glucose + "
            "phosphate; free glucose then exits via GLUT2. G6Pase is the FINAL COMMON PATHWAY for both glycogenolysis "
            "and gluconeogenesis hepatic glucose output. Deficiency → G6P accumulates in liver and kidney → glycogen "
            "accumulates (hepatomegaly, nephromegaly) → fasting hypoglycemia (cannot release free glucose). "
            "Compensatory metabolic storm: excess G6P → glycolysis → pyruvate → lactate (primary lactic acidosis, "
            "NOT secondary); → pentose phosphate pathway → fatty acid synthesis → hypertriglyceridemia; "
            "→ hypoxanthine release → hyperuricemia (gout). Classic presentation: doll face (fat pads), protuberant "
            "abdomen, short stature, fasting hypoglycemia/seizures in infancy. NO fructose/galactose (cannot be "
            "converted to free glucose without G6Pase → worsen hypoglycemia). Hepatocellular adenoma (HCA) develops "
            "in >90% by age 30y; malignant transformation to HCC 10-15%. Incidence: 1 in 100,000."
        ),
        "inheritance": "Autosomal recessive. G6PC 17q21.31. Incidence ~1 in 100,000. Pan-ethnic; Ashkenazi Jewish and Chinese enrichment.",
        "hallmark": (
            "G6PC/GSD Ia HALLMARKS: (1) CORNSTARCH CORNERSTONE — raw uncooked cornstarch (1-2 g/kg q4-6h "
            "in infants, q6h in older children; slowly absorbed α-amylase digestion) maintains fasting normoglycemia; "
            "cooked starch digests too fast; extended-release cornstarch (Glycosade) for overnight dosing; "
            "(2) NEVER GLUCOSE BOLUSES — IV bolus glucose is contraindicated in acute hypoglycemia management "
            "because glucose → G6P (via glucokinase) → trapped → exacerbates lactic acidosis and glycogen accumulation; "
            "use continuous glucose infusion (GIR 8-10 mg/kg/min) instead; "
            "(3) NO FRUCTOSE / GALACTOSE — fructose enters as F6P → F1,6-BP → glycolysis → more G6P; "
            "galactose enters as Gal1P → UDPGal → glycogen → cannot be exported as free glucose; "
            "both worsen hypoglycemia and lactic acidosis; eliminate from diet; "
            "(4) PRIMARY LACTIC ACIDOSIS — excess G6P → glycolysis → pyruvate → lactate; "
            "this is PRIMARY (enzyme-driven), not secondary to tissue hypoxia; "
            "plasma lactate 2-12 mmol/L fasting (normal <2); "
            "(5) HEPATOCELLULAR ADENOMA — develops in >90% by age 30y (glycogen/lipid accumulation + "
            "growth factor dysregulation); surveillance ultrasound every 6-12 months; "
            "AFP rising → HCC (10-15% malignant transformation); "
            "(6) KIDNEY DISEASE — renal tubular dysfunction with normoglycaemic glucosuria, "
            "later CKD (focal segmental glomerulosclerosis) due to hyperfiltration + purine nephropathy; "
            "(7) GOUT — hyperuricemia from increased purine catabolism + reduced renal clearance (lactate competes); "
            "allopurinol if symptomatic; "
            "(8) LIPID-LOWERING — hypertriglyceridemia → pancreatitis risk; fibrates/ω-3 used; "
            "(9) LIVER TRANSPLANT — corrects metabolic disease completely; adenoma → HCC = absolute indication; "
            "(10) GENE THERAPY — AAV-G6PC (DTX401) Phase 3 (Ultragenyx) — reduces hypoglycemia events"
        ),
        "key_ddx": (
            "G6PC DDx: (1) SLC37A4 (GSD Ib) — identical biochemistry + NEUTROPENIA (ANC <1500) + IBD; "
            "distinguish by neutrophil count and sequencing; (2) GSD III (AGL) — fasting ketosis (CAN make ketones); "
            "NO lactic acidosis; high protein improves (different from GSD I); (3) GSD VI/IX (PYGL/PHKA2) — "
            "much milder; NO significant lactic acidosis/hyperuricemia; resolves puberty; "
            "(4) Fructose-1,6-bisphosphatase deficiency — episodic hypoglycemia/lactic acidosis with fasting or "
            "fructose; normal glycogen; hepatomegaly absent or mild; "
            "(5) Fatty liver / glycogen hepatopathy — no metabolic storm; normal lactate"
        ),
        "diet_treatment": "Raw cornstarch 1-2g/kg q4-6h (continuous glucose provision); NO fructose/galactose; GIR 8-10 mg/kg/min IV for sick day; allopurinol (gout); lipid-lowering; liver Tx if HCA/HCC; AAV gene therapy (DTX401 Phase 3)",
        "gene_therapy_status": "DTX401 (AAV8-G6PC) Phase 3 trial (Ultragenyx); significant reduction in hypoglycemia events; first IND-treated patients reported 2021-2024",
        "critical_ci": (
            "CRITICAL CI: (1) Glucose boluses in hypoglycemia — worsen lactic acidosis (G6P trapped); use GIR drip; "
            "(2) Fructose or galactose in diet — cannot be exported as free glucose; worsens metabolic control; "
            "(3) Liver biopsy before metabolic stabilisation — coagulopathy risk if in crisis; "
            "(4) Missing adenoma surveillance — HCA → HCC without AFP/US monitoring; "
            "(5) Prescribing SGLT2 inhibitors — reduce glucosuria which is renal compensatory in GSD Ia; may worsen volume status"
        ),
        "nbs_marker": "Not in standard NBS; clinical presentation (hypoglycemia, hepatomegaly, lactic acidosis in infancy); blood glucose, lactate, triglycerides, uric acid panel; glycogen gene panel/WES",
        "key_biomarker": "Fasting plasma glucose (low); plasma lactate (HIGH, 2-12 mmol/L, PRIMARY); plasma triglycerides (HIGH); plasma uric acid (HIGH); urine uric acid; liver glycogen on biopsy (if done)",
        "severity_spectrum": "Classic severe (neonatal/infantile hypoglycemia + full metabolic storm) → Moderate (later onset, partial residual activity) → Mild (rare, significant residual)",
        "founder_variant": "p.Arg83Cys (R83C) — Ashkenazi Jewish, ~70% of Ashkenazi alleles; p.Gln347Ter (Q347X) — pan-ethnic null; p.Tyr128Cys — Chinese/Asian",
        "key_variants": [
            "p.Arg83Cys (R83C) — Ashkenazi Jewish founder, ~70% AJ alleles",
            "p.Gln347Ter (Q347X) — pan-ethnic null, classic",
            "p.Tyr128Cys (Y128C) — Chinese/East Asian",
            "p.His119Leu (H119L) — European, moderate",
            "p.Val166Gly — European, severe",
        ],
        "seed": SEED_BASE + 0,
    },
    # ── SLC37A4 — Glucose-6-Phosphate Translocase (GSD Ib / Von Gierke type b) ──
    {
        "gene": "SLC37A4", "alias": "SLC37A4 — Glucose-6-Phosphate Translocase (G6PT) (GSD Ib / Von Gierke type b)",
        "aa": "429 aa", "kDa": "46.4 kDa",
        "gene_class": "transporter",
        "locus": "11q23.3", "omim_gene": 602671,
        "phenotype": "GSD Ib — all GSD Ia features + NEUTROPENIA (ANC <1500) + inflammatory bowel disease (IBD/Crohn's-like)",
        "disease": (
            "SLC37A4 biallelic loss → GSD type Ib (OMIM #232220). SLC37A4 encodes the glucose-6-phosphate "
            "translocase (G6PT, 429aa, 46.4 kDa), an ER membrane antiporter that transports G6P from the cytosol "
            "into the ER lumen where G6Pase-α (G6PC) dephosphorylates it. Without G6PT, cytosolic G6P cannot access "
            "G6Pase → identical biochemical phenotype to GSD Ia (hypoglycemia, lactic acidosis, hypertriglyceridemia, "
            "hyperuricemia, hepatomegaly). The DISTINGUISHING FEATURE of GSD Ib (absent in GSD Ia): "
            "NEUTROPENIA (absolute neutrophil count consistently <1500/µL) — SLC37A4 is expressed in "
            "neutrophils/macrophages and is required for normal neutrophil function; its loss → neutrophil apoptosis, "
            "functional impairment, and neutropenia → severe recurrent infections (bacterial, fungal). "
            "IBD (Crohn's-like inflammatory bowel disease) occurs in ~75% of GSD Ib patients, likely from neutrophil "
            "dysfunction → altered gut immunity + gut microbiome dysregulation. Incidence: 1 in 500,000-1,000,000 "
            "(rarer than GSD Ia)."
        ),
        "inheritance": "Autosomal recessive. SLC37A4 11q23.3. Incidence ~1 in 500,000-1,000,000. Rarer than GSD Ia.",
        "hallmark": (
            "SLC37A4/GSD Ib HALLMARKS: (1) GSD Ia BIOCHEMISTRY — identical hepatic/renal phenotype: "
            "hypoglycemia, lactic acidosis, hypertriglyceridemia, hyperuricemia, hepatomegaly, adenoma risk; "
            "SAME dietary management as GSD Ia (cornstarch, no fructose/galactose, GIR for sick day); "
            "(2) NEUTROPENIA — the unique feature distinguishing GSD Ib from GSD Ia; "
            "ANC typically <1500/µL consistently; neutrophil function also impaired (even when count "
            "transiently normal); susceptibility to bacterial and fungal infections; "
            "(3) G-CSF (GRANULOCYTE COLONY-STIMULATING FACTOR) — MANDATORY in all GSD Ib; "
            "filgrastim or lenograstim titrated to ANC >1000-1500/µL; reduces infection risk and IBD severity; "
            "evidence shows G-CSF also improves IBD symptoms in GSD Ib (independent of neutrophil count); "
            "(4) INFLAMMATORY BOWEL DISEASE — Crohn's-like panenteric IBD in ~75%; "
            "may precede or follow hepatic diagnosis; recurrent oral ulcers, perianal disease, "
            "abdominal pain; responds partially to G-CSF; standard IBD therapies used adjunctively; "
            "(5) EMPAGLIFLOZIN (SGLT2 inhibitor) — emerging therapy for GSD Ib: "
            "empagliflozin prevents 1,5-anhydroglucitol-6-phosphate (1,5-AG6P) accumulation "
            "(a toxic sugar phosphate from dietary 1,5-anhydroglucitol → G6PT transport substrate) in neutrophils; "
            "reduces neutrophil dysfunction and IBD in clinical case series/trials; "
            "(6) ADENOMA RISK — same as GSD Ia (>90% by 30y); AFP/US surveillance mandatory"
        ),
        "key_ddx": (
            "SLC37A4 DDx: (1) G6PC (GSD Ia) — identical biochemistry; NO neutropenia in GSD Ia; "
            "ANC normal in GSD Ia; distinguish by ANC and sequencing; "
            "(2) Kostmann syndrome / severe congenital neutropenia (ELANE, HAX1) — neutropenia but "
            "NO hepatic metabolic disease; normal glucose/lactate; "
            "(3) Glycogen hepatopathy — hepatomegaly, glycogen excess; normal glucose/lactate; normal ANC; "
            "(4) Congenital DBA/neutropenia syndromes — bone marrow failure patterns; no metabolic storm"
        ),
        "diet_treatment": "Same as GSD Ia: cornstarch q4-6h, NO fructose/galactose, GIR sick day; PLUS G-CSF (filgrastim) MANDATORY for neutropenia; empagliflozin (SGLT2i) for IBD/neutrophil dysfunction; standard IBD therapies as adjunct",
        "gene_therapy_status": "No approved gene therapy; G6PT enzyme replacement not feasible (membrane protein); DTX401 (G6PC) does NOT help GSD Ib — separate gene",
        "critical_ci": (
            "CRITICAL CI: (1) Glucose boluses — same CI as GSD Ia (G6P trapping); "
            "(2) Withholding G-CSF — neutropenia is mandatory therapy indication; "
            "severe infections can be fatal without G-CSF; "
            "(3) Not screening for IBD in every GSD Ib patient — IBD occurs in ~75%; "
            "colonoscopy/endoscopy at diagnosis and symptom-driven; "
            "(4) Using DTX401 (G6PC gene therapy) for GSD Ib — wrong gene; "
            "(5) Standard SGLT2 inhibitor dosing without GSD metabolic monitoring — risk of DKA-like ketoacidosis"
        ),
        "nbs_marker": "Not in standard NBS; same as GSD Ia (hypoglycemia, lactic acidosis) PLUS neutropenia on CBC",
        "key_biomarker": "ANC <1500/µL (distinguishing from GSD Ia); CBC differential; plasma glucose (low); plasma lactate (HIGH); plasma triglycerides (HIGH); uric acid (HIGH); G6PT gene sequencing",
        "severity_spectrum": "Classic severe (full metabolic storm + severe neutropenia + IBD) → Moderate (partial G6PT, milder neutropenia, later IBD) → Mild (rare residual activity)",
        "founder_variant": "p.Trp118Arg (W118R) — pan-ethnic common; p.Gly339Cys (G339C) — European; p.1211-1212delCT — truncating null",
        "key_variants": [
            "p.Trp118Arg (W118R) — pan-ethnic, classic null",
            "p.Gly339Cys (G339C) — European, classic",
            "c.1211_1212delCT — frameshift null, severe",
            "p.Gln240His (Q240H) — Asian, moderate",
            "p.Arg415Cys — European, severe neutropenia",
        ],
        "seed": SEED_BASE + 1,
    },
    # ── AGL — Glycogen Debranching Enzyme (GSD IIIa/b / Cori-Forbes disease) ──
    {
        "gene": "AGL", "alias": "AGL — Amylo-1,6-Glucosidase/4-Alpha-Glucanotransferase (GSD IIIa/b / Cori-Forbes Disease)",
        "aa": "1532 aa", "kDa": "174.8 kDa",
        "gene_class": "glucosidase",
        "locus": "1p21.2", "omim_gene": 610860,
        "phenotype": "GSD IIIa/b — hepatomegaly, fasting ketotic hypoglycemia, myopathy (IIIa), elevated CK; HIGH PROTEIN diet unlike GSD I",
        "disease": (
            "AGL biallelic loss → GSD type III (Cori disease / Forbes disease, OMIM #232400). AGL encodes "
            "the glycogen debranching enzyme (GDE, 1532aa, 174.8 kDa) — the LARGEST GSD gene product — which "
            "possesses two distinct catalytic activities: (1) 4-alpha-glucanotransferase (transferase) activity "
            "that transfers a 3-unit glucan chain from a branch to the main chain; and (2) amylo-1,6-glucosidase "
            "activity that cleaves the single glucose residue at the branch point (limit dextrin cleavage). "
            "GDE is required to completely degrade glycogen beyond the outer branches. Deficiency → "
            "short-chain glycogen (LIMIT DEXTRIN) accumulates in liver, muscle (IIIa), or liver only (IIIb). "
            "GSD IIIa: liver + MUSCLE (85% of GSD III) — progressive myopathy + CK elevation + "
            "hepatomegaly + fasting hypoglycemia with KETOSIS (unlike GSD I: CAN make ketones because "
            "G6P → pyruvate → ketogenesis is intact). GSD IIIb: liver only (no muscle involvement, 15%). "
            "Lactic acidosis is MILD or ABSENT (contrast with GSD I: severe lactic acidosis). "
            "Liver cirrhosis develops over decades (25-30%); hepatocellular carcinoma reported. "
            "Incidence: 1 in 85,000."
        ),
        "inheritance": "Autosomal recessive. AGL 1p21.2. Incidence ~1 in 85,000. GSD IIIa (liver+muscle, 85%) vs GSD IIIb (liver only, 15%).",
        "hallmark": (
            "AGL/GSD IIIa-b HALLMARKS: (1) LIMIT DEXTRIN ACCUMULATION — short-chain glycogen (limit dextrin) "
            "accumulates (vs long-chain polyglucosan in GBE1/GSD IV); specific histological pattern; "
            "(2) HIGH PROTEIN DIET — 25-30% protein, 25% fat, 45% complex carbohydrate; "
            "protein → gluconeogenesis → bypasses glycogen block; "
            "THIS IS DIFFERENT FROM GSD I where protein is unrestricted and complex carbs are limited; "
            "high protein diet reduces hypoglycemia AND may reduce myopathy in GSD IIIa; "
            "(3) FASTING KETOTIC HYPOGLYCEMIA — unlike GSD I, GSD III CAN make ketones "
            "(glycogen → G6P → acetyl-CoA → ketogenesis is intact); "
            "fasting → ketosis + hypoglycemia (limit dextrin cannot fully mobilise glucose); "
            "(4) MYOPATHY (IIIa only) — progressive skeletal muscle myopathy in adults; "
            "proximal muscle weakness; elevated CK (often >2000 IU/L in IIIa); "
            "liver transplant does NOT improve muscle disease — the hepatic transplant provides GDE only in liver; "
            "muscle disease continues to progress post-transplant in IIIa; "
            "(5) LIVER CIRRHOSIS — progressive fibrosis in 25-30%; liver transplant corrects hepatic disease "
            "but muscle disease (IIIa) continues; decision for Tx must weigh this; "
            "(6) MILD/ABSENT LACTIC ACIDOSIS — unlike GSD I (severe primary lactic acidosis); "
            "GSD III does not have massively elevated G6P → glycolysis is not as flooded; "
            "this is a KEY DDx clue; "
            "(7) CORNSTARCH — used if hypoglycemia symptomatic (unlike GSD I, not always mandatory); "
            "(8) IIIa vs IIIb — AGL exon 3 mutations (p.Gln6Ter, p.Asp6Ter) exclusively cause IIIb "
            "(liver only) due to specific transcript expression"
        ),
        "key_ddx": (
            "AGL DDx: (1) G6PC/GSD Ia — severe lactic acidosis (primary), NO ketosis, NO fructose/galactose; "
            "AGL: mild/no lactic acidosis, KETOTIC hypoglycemia, high protein helps; "
            "(2) GBE1/GSD IV — LONG-chain polyglucosan (not limit dextrin); liver cirrhosis earlier; "
            "distinguished by glycogen morphology and sequencing; "
            "(3) PYGM/GSD V (McArdle) — exercise intolerance not fasting hypoglycemia; no hepatomegaly; "
            "(4) Ketotic hypoglycemia idiopathic — no hepatomegaly, normal glycogen, normal CK; "
            "(5) POMPE disease (GAA) — lysosomal glycogen accumulation; acid alpha-glucosidase deficiency; "
            "vacuolar myopathy on biopsy; enzyme assay diagnostic"
        ),
        "diet_treatment": "High protein diet (25-30% protein); complex carbohydrates (limit simple sugars); cornstarch if symptomatic hypoglycemia; raw cornstarch nocturnal; liver transplant for cirrhosis/HCC (muscle disease NOT improved in IIIa)",
        "gene_therapy_status": "No approved gene therapy; AAV-AGL hepatic trials in development; high protein diet evidence-based",
        "critical_ci": (
            "CRITICAL CI: (1) Applying GSD I dietary restrictions (no fructose/galactose) to GSD III — "
            "GSD III can tolerate galactose and fructose (different enzyme block); "
            "(2) Expecting liver transplant to cure muscle disease in IIIa — liver Tx corrects hepatic GDE "
            "but muscle GDE remains deficient; counsel family pre-Tx; "
            "(3) Not trialling high protein diet — this is the primary therapeutic tool in GSD III, "
            "not just cornstarch; "
            "(4) Confusing limit dextrin (GSD III) with polyglucosan (GSD IV) on biopsy — "
            "different morphology, different enzyme, different treatment"
        ),
        "nbs_marker": "Not in standard NBS; elevated transaminases + hepatomegaly in infancy; plasma glucose/ketones; CK (IIIa); AGL enzyme in leukocytes; gene sequencing",
        "key_biomarker": "Plasma CK (elevated in IIIa, normal IIIb); fasting glucose (low) + ketones (HIGH — unlike GSD I); plasma lactate (mild or normal); liver glycogen (limit dextrin pattern on EM); AGL enzyme activity in leukocytes or fibroblasts",
        "severity_spectrum": "GSD IIIa classic (liver + progressive muscle myopathy) → GSD IIIa mild → GSD IIIb (liver only, no muscle, better prognosis)",
        "founder_variant": "p.Gln6Ter (AGL exon 3, liver-only IIIb, North African Jewish); p.Arg864Ter — pan-ethnic null (IIIa); p.Trp1327Ter — European",
        "key_variants": [
            "p.Gln6Ter — North African Jewish, exon 3, GSD IIIb (liver only)",
            "p.Arg864Ter — pan-ethnic null, GSD IIIa",
            "p.Trp1327Ter — European, IIIa, severe",
            "p.4529insA — pan-ethnic frameshift, IIIa",
            "p.Glu1542Ter — European, IIIa",
        ],
        "seed": SEED_BASE + 2,
    },
    # ── GBE1 — Glycogen Branching Enzyme (GSD IV / Andersen disease) ──
    {
        "gene": "GBE1", "alias": "GBE1 — Glycogen Branching Enzyme (GSD IV / Andersen Disease / Polyglucosan Body Disease)",
        "aa": "702 aa", "kDa": "80.1 kDa",
        "gene_class": "transferase",
        "locus": "3p12.2", "omim_gene": 607839,
        "phenotype": "GSD IV — polyglucosan (amylopectin-like) accumulation; classic: fatal liver cirrhosis → liver Tx; adult APBD: motor/cognitive, NO hepatic",
        "disease": (
            "GBE1 biallelic loss → GSD type IV (Andersen disease, OMIM #232500) or Adult Polyglucosan Body Disease "
            "(APBD, OMIM #263570). GBE1 encodes the glycogen branching enzyme (GBE, 702aa, 80.1 kDa), which transfers "
            "alpha-1,4-linked glucan chains (minimum 6 residues) to alpha-1,6-branch points, creating the highly "
            "branched structure of normal glycogen. Deficiency → abnormal glycogen accumulates with few branch points "
            "and long outer chains — resembling AMYLOPECTIN (plant starch structure) or POLYGLUCOSAN BODIES. "
            "This poorly soluble glycogen accumulates in liver, muscle, and nervous system. "
            "CLASSIC FATAL HEPATIC FORM (most common, typically null/severe mutations): progressive liver cirrhosis "
            "by age 1-5y, portal hypertension, liver failure, death in childhood without liver transplant. "
            "Cardiomyopathy in some. NEUROMUSCULAR FORM (hypomorphic/partial loss): exercise intolerance, "
            "myopathy, neuropathy, NO hepatic disease. ADULT POLYGLUCOSAN BODY DISEASE (APBD): progressive "
            "upper + lower motor neuron disease, cognitive decline, neurogenic bladder, NO/minimal liver involvement; "
            "liver transplant NOT curative for this form. NEONATAL FORM: hydrops fetalis, fetal demise. "
            "Incidence: 1 in 500,000-1,000,000."
        ),
        "inheritance": "Autosomal recessive. GBE1 3p12.2. Incidence ~1 in 500,000-1,000,000. Multiple phenotypic forms based on residual GBE activity.",
        "hallmark": (
            "GBE1/GSD IV HALLMARKS: (1) POLYGLUCOSAN ACCUMULATION — abnormal glycogen with long unbranched "
            "chains, resembling amylopectin; periodic acid-Schiff (PAS) positive, diastase-RESISTANT "
            "(unlike normal glycogen which is diastase-sensitive); periodic acid-Schiff/diastase (PAS-D) test "
            "on liver biopsy is diagnostic; "
            "(2) CLASSIC HEPATIC FORM — progressive liver fibrosis → cirrhosis by 1-5y; "
            "portal hypertension, ascites, liver failure; no effective medical therapy; "
            "LIVER TRANSPLANT is CURATIVE for the hepatic form — corrects metabolic disease in liver; "
            "however, polyglucosan accumulation continues in extrahepatic tissues (muscle, neurons); "
            "neurological progression may occur post-Tx in some; "
            "(3) LIVER TX DOES NOT CURE NEUROMUSCULAR/APBD FORM — liver transplant does not affect "
            "the neuromuscular polyglucosan accumulation; not indicated for adult APBD; "
            "(4) ADULT POLYGLUCOSAN BODY DISEASE (APBD) — partial GBE1 loss → adult-onset progressive "
            "motor neuron disease (upper + lower), cognitive decline, neurogenic bladder; "
            "nerve biopsy: polyglucosan bodies in axons (PAS-D positive); "
            "Jewish Ashkenazi founder p.Tyr329Ser enriched in APBD; "
            "(5) NEONATAL HYDROPS FORM — complete GBE1 loss in utero → fetal hydrops, "
            "cardiomyopathy, polyhydramnios, stillbirth or neonatal death; "
            "(6) GBE ENZYME ASSAY — leukocytes or fibroblasts; <10% activity in classic form; "
            "10-30% residual in APBD/neuromuscular"
        ),
        "key_ddx": (
            "GBE1 DDx: (1) AGL (GSD IIIa/b) — limit dextrin (short-chain, diastase-sensitive) vs "
            "polyglucosan (long-chain, diastase-RESISTANT); AGL: ketotic hypoglycemia + high protein helps; "
            "(2) Lafora disease (EPM2A/NHLRC1) — polyglucosan (Lafora bodies) but in brain; progressive "
            "myoclonic epilepsy; onset ~15y; genetic and clinical difference; "
            "(3) ALPHA-GLUCAN STORAGE DISORDERS — Pompe (GAA): lysosomal; acid phosphatase stain positive; "
            "(4) Cryptogenic cirrhosis — PAS-D stain on biopsy crucial; GBE enzyme assay; "
            "(5) Neurological APBD DDx: ALS, PLS, PPMS — no polyglucosan on nerve biopsy; "
            "normal GBE enzyme in those conditions"
        ),
        "diet_treatment": "No specific dietary treatment corrects the block; liver transplant for classic hepatic form (before cirrhosis decompensation); supportive for neuromuscular/APBD form; no proven pharmacological therapy",
        "gene_therapy_status": "No approved gene therapy; AAV-GBE1 in preclinical models; enzyme replacement not available; liver Tx remains definitive for hepatic form",
        "critical_ci": (
            "CRITICAL CI: (1) Offering liver transplant to APBD patients — liver Tx does not halt "
            "neurological polyglucosan accumulation; not indicated; "
            "(2) Missing the diagnosis by not doing PAS-D on liver biopsy — "
            "H&E alone may show cirrhosis without identifying the cause; PAS-D mandatory; "
            "(3) Confusing with AGL/GSD III — different glycogen morphology, different enzyme, "
            "different dietary response; GBE1: no benefit from high protein diet; "
            "(4) Missing APBD in adult patients with progressive motor/cognitive disease — "
            "nerve biopsy + GBE enzyme assay + GBE1 sequencing needed"
        ),
        "nbs_marker": "Not in standard NBS; elevated LFTs in infancy (hepatic form); hepatomegaly; GBE enzyme in leukocytes; gene sequencing",
        "key_biomarker": "Liver biopsy: PAS-D positive, diastase-RESISTANT polyglucosan bodies (pathognomonic); GBE enzyme activity in leukocytes/fibroblasts (<10% classic, 10-30% APBD); nerve biopsy (APBD): axonal polyglucosan bodies",
        "severity_spectrum": "Neonatal hydrops (lethal) → Classic hepatic (cirrhosis <5y, fatal without Tx) → Neuromuscular childhood (myopathy) → APBD (adult progressive, no hepatic)",
        "founder_variant": "p.Tyr329Ser — Ashkenazi Jewish APBD founder (~40% of APBD alleles); p.Phe257Leu — North European hepatic form",
        "key_variants": [
            "p.Tyr329Ser (Y329S) — Ashkenazi Jewish, APBD form",
            "p.Phe257Leu — North European, hepatic classic",
            "p.Arg524Gln — hepatic, severe",
            "p.Leu224Pro — hepatic, neonatal hydrops form",
            "p.Arg515His — neuromuscular, partial activity",
        ],
        "seed": SEED_BASE + 3,
    },
    # ── PYGM — Muscle Glycogen Phosphorylase (GSD V / McArdle disease) ──
    {
        "gene": "PYGM", "alias": "PYGM — Muscle Glycogen Phosphorylase (Myophosphorylase) (GSD V / McArdle Disease)",
        "aa": "842 aa", "kDa": "97.3 kDa",
        "gene_class": "phosphorylase",
        "locus": "11q13.1", "omim_gene": 608455,
        "phenotype": "GSD V — exercise intolerance, myalgia, cramps, myoglobinuria with exercise; McArdle sign; sucrose trick; second wind; NO hepatic disease",
        "disease": (
            "PYGM biallelic loss → GSD type V (McArdle disease, OMIM #232600). PYGM encodes muscle glycogen "
            "phosphorylase (myophosphorylase, 842aa, 97.3 kDa), the key enzyme that liberates G1P from "
            "muscle glycogen during exercise. Muscle cannot access its glycogen during exercise → "
            "anaerobic glycolysis fails → energy crisis during muscle contraction. Clinical: "
            "exercise intolerance (rapid fatigue, myalgia, painful cramps) from the first minutes of "
            "exercise; MYOGLOBINURIA with strenuous exercise (rhabdomyolysis → reddish-brown urine → "
            "acute kidney injury risk in severe episodes). SECOND WIND PHENOMENON: after 7-10 minutes of "
            "moderate aerobic exercise, symptoms improve as fatty acid oxidation and hepatic glycogenolysis "
            "provide alternative substrate (working muscle shifts from glycogen to blood glucose + FFA). "
            "Baseline CK markedly elevated (500-2000 IU/L), surging dramatically with exercise. "
            "NO HEPATIC DISEASE (liver phosphorylase = PYGL, different gene). Normal intelligence. "
            "Symptoms typically begin in childhood but diagnosis often delayed to adulthood. "
            "p.Arg50Ter (R50X) is the most common European null variant (~80% of European alleles). "
            "Incidence: 1 in 100,000 (Northern Europeans); underdiagnosed."
        ),
        "inheritance": "Autosomal recessive. PYGM 11q13.1. Incidence ~1 in 100,000. Most common in Northern Europeans. p.Arg50Ter dominant European allele.",
        "hallmark": (
            "PYGM/McArdle HALLMARKS: (1) McARDLE SIGN (FOREARM ISCHEMIC EXERCISE TEST) — "
            "inflate BP cuff above systolic pressure on arm, patient squeezes a ball for 1 min; "
            "NORMAL: venous lactate rises 3-5x baseline AND ammonia rises 3-5x; "
            "McARDLE: venous lactate FAILS TO RISE (no anaerobic glycolysis from muscle glycogen) "
            "but ammonia RISES normally (purine nucleotide cycle intact) — pathognomonic for all "
            "myophosphorylase-related GSD (PYGM, PYGL); "
            "(2) SUCROSE TRICK — 35g oral sucrose 5 minutes before exercise; sucrose → "
            "glucose + fructose → blood glucose rises → muscles can use blood glucose during exercise; "
            "PREVENTS MYOGLOBINURIA and improves exercise tolerance; evidence-based; "
            "(3) SECOND WIND — the defining clinical hallmark; after 7-10 min of aerobic exercise, "
            "symptoms dramatically improve; hepatic glycogenolysis + fat oxidation compensate; "
            "patients learn to pace exercise to trigger second wind; "
            "(4) MYOGLOBINURIA — with strenuous exercise; reddish-brown urine; "
            "risk of acute kidney injury; aggressive IV hydration MANDATORY during rhabdomyolysis; "
            "avoid exercise to exhaustion; "
            "(5) BASELINE CK ELEVATION — 500-2000 IU/L at rest; surges to 10,000-100,000+ with exertion; "
            "(6) AEROBIC EXERCISE TRAINING — progressive aerobic conditioning is THERAPEUTIC; "
            "aerobic exercise increases GLUT4, fatty acid transport → better exercise tolerance long-term; "
            "(7) HIGH-PROTEIN DIET — 25% protein provides gluconeogenic substrate; "
            "(8) p.Arg50Ter (R50X) — ~80% European alleles; carrier frequency 1 in 120 Northern Europeans; "
            "(9) VITAMIN B6 (pyridoxine) supplementation — no significant benefit (cofactor not deficient in PYGM)"
        ),
        "key_ddx": (
            "PYGM DDx: (1) PFKM (GSD VII/Tarui) — same exercise intolerance + myoglobinuria + lactate fails "
            "to rise in ischemic test; BUT Tarui: HEMOLYTIC ANEMIA (PFKM in RBCs) + HIGH-CARB PARADOX; "
            "distinguish by enzyme assay or sequencing; (2) Mitochondrial myopathy (CPEO, mtDNA deletion) — "
            "fixed weakness + ptosis; lactate often elevated at rest; (3) CPT2 deficiency — "
            "fasting-triggered myoglobinuria (not exercise alone); C16 acylcarnitine elevated; "
            "(4) VLCAD deficiency — fasting + cold trigger; C14:1 acylcarnitine elevated; "
            "(5) Inflammatory myopathy (polymyositis/dermatomyositis) — subacute weakness + EMG changes; "
            "creatine kinase elevated differently; autoantibodies"
        ),
        "diet_treatment": "High protein (25%) + moderate complex carb; sucrose trick (35g before exercise); progressive aerobic training; aggressive IV hydration for myoglobinuria; avoid exercise to exhaustion",
        "gene_therapy_status": "No approved gene therapy; aerobic exercise training Level A evidence; sucrose trick Level B evidence; gene therapy preclinical models (PYGM AAV)",
        "critical_ci": (
            "CRITICAL CI: (1) Isometric exercise without sucrose trick — triggers severe rhabdomyolysis; "
            "(2) High carbohydrate loading before exercise — NOT beneficial (contrast with Tarui where it is harmful); "
            "sucrose trick timed correctly (5 min before) is different from carb loading; "
            "(3) Delaying IV hydration during myoglobinuria — rhabdomyolysis-induced AKI preventable with "
            "prompt IV normal saline; "
            "(4) Prescribing creatine without evidence — some benefit; not mandatory; vitamin B6 not beneficial; "
            "(5) Not measuring ischemic forearm test lactate — missed diagnosis; ammonia alone insufficient"
        ),
        "nbs_marker": "Not in standard NBS; exercise intolerance + myoglobinuria in childhood/adulthood; elevated baseline CK; ischemic forearm test; PYGM gene sequencing",
        "key_biomarker": "Forearm ischemic exercise test: lactate FAILS to rise, ammonia rises normally (pathognomonic); serum CK (500-2000 IU/L baseline, much higher post-exercise); urine myoglobin (with exercise); muscle biopsy: absent myophosphorylase histochemistry",
        "severity_spectrum": "Mild (exercise intolerance only, no myoglobinuria) → Moderate (frequent myoglobinuria) → Severe (recurrent AKI) — rare fixed weakness in elderly patients",
        "founder_variant": "p.Arg50Ter (R50X) — Northern European, ~80% of European alleles; p.Gly205Ser — Spanish/Mediterranean",
        "key_variants": [
            "p.Arg50Ter (R50X) — Northern European founder, ~80% European alleles",
            "p.Gly205Ser — Spanish/Mediterranean, common",
            "p.Lys542Thr — North African",
            "p.Glu462Gly — European, moderate",
            "p.Trp798Ter — null, severe",
        ],
        "seed": SEED_BASE + 4,
    },
    # ── PYGL — Liver Glycogen Phosphorylase (GSD VI / Hers disease) ──
    {
        "gene": "PYGL", "alias": "PYGL — Liver Glycogen Phosphorylase (GSD VI / Hers Disease)",
        "aa": "847 aa", "kDa": "97.2 kDa",
        "gene_class": "phosphorylase",
        "locus": "14q22.1", "omim_gene": 613741,
        "phenotype": "GSD VI — GENERALLY BENIGN: hepatomegaly, mild hypoglycemia, hyperlipidemia; NO lactic acidosis; resolves puberty",
        "disease": (
            "PYGL biallelic loss → GSD type VI (Hers disease, OMIM #232700). PYGL encodes liver glycogen "
            "phosphorylase (847aa, 97.2 kDa), which is activated by phosphorylase kinase (PhK) and AMP, "
            "initiating glycogenolysis in the liver. Deficiency → hepatic glycogen cannot be mobilized → "
            "mild/moderate hepatomegaly from glycogen accumulation + mild fasting hypoglycemia. "
            "GENERALLY BENIGN DISORDER: the degree of metabolic compromise is much less than GSD Ia/Ib/III "
            "because gluconeogenesis (which does not require glycogen phosphorylase) is intact, providing "
            "alternative glucose. Lactic acidosis is ABSENT or minimal (contrast with GSD I). "
            "Hyperuricemia is ABSENT (contrast with GSD I). Elevated transaminases in childhood. "
            "REMARKABLE FEATURE: spontaneous resolution at puberty/adolescence (hepatomegaly resolves, "
            "transaminases normalise) — hormonal changes may upregulate alternative pathways. "
            "Long-term prognosis is EXCELLENT. Incidence: ~1 in 65,000-100,000 (may be underdiagnosed "
            "due to mild phenotype)."
        ),
        "inheritance": "Autosomal recessive. PYGL 14q22.1. Incidence ~1 in 65,000-100,000. GENERALLY BENIGN with excellent long-term prognosis.",
        "hallmark": (
            "PYGL/GSD VI HALLMARKS: (1) HEPATOMEGALY without significant functional liver disease — "
            "liver enlarged due to glycogen accumulation; transaminases mildly elevated; "
            "NO cirrhosis (unlike GSD III/IV); "
            "(2) MILD FASTING HYPOGLYCEMIA — typically only with prolonged fasting; "
            "mild ketonemia (gluconeogenesis intact → some ketogenesis preserved); "
            "NO seizures in typical cases; "
            "(3) NO LACTIC ACIDOSIS — key distinguishing feature from GSD Ia/Ib; "
            "plasma lactate NORMAL or minimally elevated; "
            "(4) NO HYPERURICEMIA — absent (contrast with GSD Ia/Ib); "
            "(5) SPONTANEOUS RESOLUTION — hepatomegaly and hypoglycemia resolve at puberty/adolescence "
            "in most patients; mechanism unclear (may relate to hormonal changes in glycogenolytic demands "
            "or epigenetic adaptation); this is the single most important prognostic feature of GSD VI; "
            "(6) CORNSTARCH — used only if symptomatic hypoglycemia; NOT mandatory in all patients; "
            "dose adjustments as patient grows; "
            "(7) DDx from PHKA2 (GSD IXa) — identical clinical phenotype but different gene "
            "(PYGL = AR enzyme; PHKA2 = X-linked PhK regulator); phosphorylase enzyme activity needed "
            "to distinguish; gene sequencing is definitive; "
            "(8) LIVER BIOPSY — glycogen-laden hepatocytes; PAS-D positive, diastase-sensitive "
            "(normal glycogen structure, unlike GSD IV)"
        ),
        "key_ddx": (
            "PYGL DDx: (1) PHKA2 (GSD IXa) — identical clinical phenotype; PHKA2 = X-linked; "
            "PYGL = AR; enzyme assay: PhK normal + phosphorylase low in PYGL; "
            "PhK low in PHKA2; sequencing definitive; "
            "(2) G6PC (GSD Ia) — severe lactic acidosis, hyperuricemia, adenoma risk; "
            "ABSENT in GSD VI; (3) AGL (GSD IIIa/b) — CK elevated in IIIa, myopathy; "
            "GSD VI has no muscle disease; (4) Transient hepatomegaly of infancy — "
            "resolves without metabolic sequelae; (5) NAFLD — BMI, insulin resistance, "
            "normal glycogen enzyme panel"
        ),
        "diet_treatment": "Cornstarch if symptomatic hypoglycemia (not mandatory); avoid prolonged fasting; high-complex-carb diet; frequent meals; NO specific metabolic restriction (no need for fructose/galactose avoidance); spontaneous resolution expected at puberty",
        "gene_therapy_status": "No gene therapy needed (generally benign; resolves spontaneously); dietary management sufficient in most",
        "critical_ci": (
            "CRITICAL CI: (1) Applying GSD I restrictions (no fructose/galactose) — not needed in GSD VI; "
            "overtreatment causes unnecessary dietary restriction; "
            "(2) Not reassuring families of benign prognosis — GSD VI resolves; avoid over-medicalisation; "
            "(3) Missing the diagnosis and attributing hepatomegaly to viral hepatitis — "
            "enzyme assay/sequencing needed; "
            "(4) Not distinguishing from GSD IX (PHKA2) — management same but inheritance differs "
            "(X-linked in PHKA2 affects genetic counselling)"
        ),
        "nbs_marker": "Not in standard NBS; incidental hepatomegaly on examination; mildly elevated transaminases; fasting glucose/ketones; liver phosphorylase enzyme activity; PYGL sequencing",
        "key_biomarker": "Liver phosphorylase activity (low in leukocytes/liver); plasma glucose (mild fasting hypoglycemia); plasma lactate (NORMAL); plasma triglycerides (mild elevation); liver biopsy: glycogen-laden hepatocytes, PAS-D positive, diastase-sensitive",
        "severity_spectrum": "Mild transient hepatomegaly with hypoglycemia (resolves puberty) → Moderate symptomatic (cornstarch needed) → Rare severe (persistent hypoglycemia into adulthood)",
        "founder_variant": "No single dominant founder; p.Gly564Asp — European; p.Arg405Gln — pan-ethnic",
        "key_variants": [
            "p.Gly564Asp — European, moderate",
            "p.Arg405Gln — pan-ethnic",
            "p.Lys628Ter — null, severe",
            "p.Trp493Ter — European null",
            "p.Ile684Thr — hypomorphic, mild",
        ],
        "seed": SEED_BASE + 5,
    },
    # ── PFKM — Phosphofructokinase Muscle Isoform (GSD VII / Tarui disease) ──
    {
        "gene": "PFKM", "alias": "PFKM — Phosphofructokinase Muscle Isoform (GSD VII / Tarui Disease)",
        "aa": "780 aa", "kDa": "85.0 kDa",
        "gene_class": "kinase",
        "locus": "12q13.3", "omim_gene": 610681,
        "phenotype": "GSD VII — exercise intolerance + myoglobinuria; HEMOLYTIC ANEMIA (all patients); HIGH CARBOHYDRATE PARADOX — carbs WORSEN exercise; Ashkenazi Jewish founder",
        "disease": (
            "PFKM biallelic loss → GSD type VII (Tarui disease, OMIM #232800). PFKM encodes the muscle "
            "isoform of phosphofructokinase (PFK-M, 780aa, 85.0 kDa), the key regulatory enzyme of "
            "glycolysis: fructose-6-phosphate → fructose-1,6-bisphosphate (ATP-consuming, rate-limiting step). "
            "PFKM is expressed in BOTH skeletal muscle AND RED BLOOD CELLS (RBCs, which use PFKM + PFKL "
            "to form the tetrameric RBC PFK). Deficiency → (1) MUSCLE: identical to McArdle but glycolysis "
            "blocked earlier (before glycogen phosphorylase even acts) → exercise intolerance, myalgia, "
            "cramps, myoglobinuria; FOREARM TEST: lactate FAILS to rise (no glycolysis in muscle), "
            "ammonia rises (same as PYGM). (2) RBCs: compensated HEMOLYTIC ANEMIA in ALL PFKM patients "
            "(reticulocytosis, elevated bilirubin, mildly low Hgb). UNIQUE CLINICAL FEATURE: "
            "HIGH CARBOHYDRATE PARADOX — glucose or fructose ingestion before exercise WORSENS exercise "
            "tolerance (glucose → glucose-6-phosphate → fructose-6-phosphate → accumulates before the "
            "PFKM block → inhibits remaining PFK activity + hexokinase → less substrate available). "
            "Ashkenazi Jewish founder mutations. Incidence: very rare globally, ~1 in 1,000,000."
        ),
        "inheritance": "Autosomal recessive. PFKM 12q13.3. Very rare globally (~1 in 1,000,000). Enriched in Ashkenazi Jewish. Hemolysis in ALL patients.",
        "hallmark": (
            "PFKM/GSD VII HALLMARKS: (1) HEMOLYTIC ANEMIA — ALL PFKM patients have compensated "
            "hemolysis (reticulocytosis 5-15%, elevated indirect bilirubin, mildly decreased Hgb ~10-12 g/dL); "
            "erythrocyte PFK (mixed PFKM/PFKL) is reduced to ~50% of normal in heterozygotes, "
            "<10% in homozygotes; RBC osmotic fragility increased; "
            "(2) HIGH CARBOHYDRATE PARADOX — THE UNIQUE CLINICAL HALLMARK: "
            "eating carbohydrates before exercise WORSENS symptoms (opposite of McArdle where sucrose helps); "
            "mechanism: glucose → G6P → F6P → accumulates before the PFK-M block → "
            "competitive inhibition of residual PFK + hexokinase feedback inhibition; "
            "glycogen breakdown is relatively UNAFFECTED by dietary carbs (glycogen → G1P → G6P → F6P → "
            "same block); net effect: high glucose REDUCES alternative glycogen-derived substrate efficiency; "
            "(3) EXERCISE INTOLERANCE + MYOGLOBINURIA — same pattern as McArdle/PYGM; "
            "strenuous exercise → rhabdomyolysis → myoglobinuria → AKI risk; "
            "(4) FOREARM ISCHEMIC TEST: lactate FAILS to rise (glycolysis blocked at PFK); "
            "ammonia rises (same as PYGM — purine cycle intact); "
            "(5) SECOND WIND — present but less prominent than McArdle; "
            "(6) FASTING PARADOX — short fasting IMPROVES exercise tolerance in PFKM "
            "(fasting increases FFA + gluconeogenesis → bypasses glycolytic block); "
            "(7) ASHKENAZI JEWISH FOUNDER — p.Asp181del + p.Arg7Gln-splice are the two founder mutations; "
            "(8) PFK ASSAY — erythrocyte PFK activity near-absent in homozygotes; "
            "muscle biopsy: PAS-D positive glycogen + absent PFK histochemistry"
        ),
        "key_ddx": (
            "PFKM DDx: (1) PYGM (McArdle, GSD V) — exercise intolerance + myoglobinuria + lactate FAILS to rise; "
            "KEY DIFFERENCE: McArdle has NO hemolysis; sucrose HELPS McArdle (WORSENS PFKM); "
            "enzyme assay (muscle PFK vs myophosphorylase) and sequencing distinguish; "
            "(2) PHKA2 (GSD IXa) — X-linked, hepatic, no myoglobinuria, no hemolysis; "
            "(3) Hereditary hemolytic anemias (G6PD deficiency, PK deficiency) — hemolysis but "
            "NO exercise myopathy; different RBC enzyme deficiencies; "
            "(4) Mitochondrial myopathy — resting lactate elevated; fixed weakness; mtDNA or OXPHOS panel"
        ),
        "diet_treatment": "AVOID carbohydrate loading before exercise (WORSENS); fasting or low-carb state BEFORE exercise may help; fat-rich pre-exercise meal; progressive aerobic training; aggressive IV hydration for myoglobinuria; folate supplementation for hemolysis",
        "gene_therapy_status": "No approved gene therapy; dietary carbohydrate timing management is primary; aerobic training beneficial",
        "critical_ci": (
            "CRITICAL CI: (1) Sucrose trick before exercise — contraindicated (OPPOSITE of McArdle); "
            "worsens exercise intolerance in PFKM; counsel patients explicitly; "
            "(2) Carbohydrate loading before exercise — HIGH-CARB PARADOX; worsens symptoms; "
            "(3) Missing hemolytic anemia — ALL PFKM patients have compensated hemolysis; "
            "CBC is mandatory; not diagnosing anemia leaves folate unsubstituted; "
            "(4) Using the same pre-exercise advice as McArdle (sucrose = G6P → more F6P → worse in PFKM)"
        ),
        "nbs_marker": "Not in standard NBS; exercise intolerance + hemolytic anemia; erythrocyte PFK assay; forearm ischemic test; PFKM sequencing",
        "key_biomarker": "Erythrocyte PFK activity (absent in homozygotes); serum bilirubin (elevated indirect — hemolysis); reticulocytosis; serum CK (elevated baseline, surge with exercise); forearm ischemic test: lactate fails to rise",
        "severity_spectrum": "Hemolytic only (heterozygotes — no myopathy) → Classic (hemolysis + exercise myopathy, most patients) → Infantile polyglucosan form (rare, severe, generalised myopathy, cardiomyopathy)",
        "founder_variant": "p.Asp181del (Ashkenazi Jewish, ~67% AJ alleles) + c.237+1G>A (Arg7Gln-splice) (Ashkenazi Jewish, ~33% AJ alleles)",
        "key_variants": [
            "p.Asp181del — Ashkenazi Jewish founder, ~67% AJ alleles",
            "c.237+1G>A (splice) — Ashkenazi Jewish founder, ~33% AJ alleles",
            "p.Val647Gly — non-Jewish, European",
            "p.Cys114Tyr — pan-ethnic, hemolysis predominant",
            "p.Arg255Ter — null, infantile severe form",
        ],
        "seed": SEED_BASE + 6,
    },
    # ── PHKA2 — Phosphorylase Kinase Alpha2 Subunit (GSD IXa / X-linked PhK deficiency) ──
    {
        "gene": "PHKA2", "alias": "PHKA2 — Phosphorylase Kinase Alpha2 Subunit (GSD IXa / X-Linked Liver Phosphorylase Kinase Deficiency)",
        "aa": "1235 aa", "kDa": "138.5 kDa",
        "gene_class": "kinase",
        "locus": "Xp22.13", "omim_gene": 311870,
        "phenotype": "GSD IXa — X-LINKED; hepatomegaly, mild hypoglycemia, short stature; GENERALLY BENIGN; resolves adulthood",
        "disease": (
            "PHKA2 hemizygous loss → GSD type IXa (Liver Phosphorylase Kinase Deficiency X-linked, OMIM #306000). "
            "PHKA2 encodes the alpha2 subunit of phosphorylase kinase (PhK, 1235aa, 138.5 kDa), which activates "
            "liver glycogen phosphorylase (PYGL) by phosphorylation. Without PhK activity, PYGL remains inactive "
            "→ mild hepatic glycogen accumulation. X-LINKED INHERITANCE: PHKA2 on Xp22.13 → hemizygous males "
            "affected; heterozygous females may have mild hepatomegaly. This distinguishes GSD IXa from GSD VI "
            "(PYGL, AR) which is clinically identical but autosomal recessive. GSD IXa is GENERALLY BENIGN: "
            "hepatomegaly (from mild glycogen accumulation), mildly elevated transaminases, short stature, "
            "mild fasting hypoglycemia with mild ketonemia; NO significant lactic acidosis; NO hyperuricemia; "
            "liver fibrosis is rare/absent. Most patients RESOLVE SPONTANEOUSLY IN ADULTHOOD — height normalises, "
            "hepatomegaly resolves, liver function normalises. GSD IXa is the MOST COMMON liver GSD. "
            "Incidence: ~1 in 100,000 males. PhK gene family: PHKA2 (X-linked, liver) vs PHKB (AR, liver+heart) "
            "vs PHKG2 (AR, liver, more severe; risk of fibrosis) — locus distinguishes."
        ),
        "inheritance": "X-linked recessive. PHKA2 Xp22.13. Hemizygous males affected; heterozygous females mild/carrier. Most common liver GSD. Incidence ~1 in 100,000 males.",
        "hallmark": (
            "PHKA2/GSD IXa HALLMARKS: (1) X-LINKED — critical for genetic counselling; "
            "mother is obligate carrier (typically asymptomatic or mild hepatomegaly); "
            "sons have 50% chance; daughters have 50% chance of being carriers; "
            "PHKB (liver+heart, AR) and PHKG2 (liver, AR, fibrosis risk) are DIFFERENT genes and autosomal recessive; "
            "(2) GENERALLY BENIGN — the most important clinical message; "
            "hepatomegaly and mild hypoglycemia are the primary findings; "
            "lactic acidosis, hyperuricemia, myopathy, cirrhosis are NOT expected; "
            "(3) SPONTANEOUS RESOLUTION IN ADULTHOOD — hepatomegaly, transaminases, height all "
            "normalise after puberty; cornstarch is a temporary measure; "
            "(4) CORNSTARCH — used if symptomatic fasting hypoglycemia (NOT mandatory in all); "
            "dose adjusted as child grows; can be discontinued as symptoms resolve; "
            "(5) FASTING KETOTIC HYPOGLYCEMIA — mild; responds to frequent feeds and cornstarch; "
            "NO glucose bolus issues (glycogen can still mobilize via gluconeogenesis and residual PhK); "
            "(6) PhK ENZYME ASSAY — erythrocyte PhK activity reduced; distinguishes from PYGL deficiency "
            "(where PhK is normal but phosphorylase is low); "
            "(7) PHKA2 vs PHKB vs PHKG2 — PHKG2 deficiency has FIBROSIS RISK (up to cirrhosis); "
            "clinically similar but liver biopsy shows fibrosis; PHKG2 not X-linked (AR); "
            "genetic testing mandatory to distinguish; "
            "(8) PHKA2 GENOTYPE-PHENOTYPE — null mutations generally more symptomatic than missense; "
            "some missense cause hepatic only with complete adult resolution"
        ),
        "key_ddx": (
            "PHKA2 DDx: (1) PYGL (GSD VI) — clinically identical; both benign and resolve at puberty; "
            "distinguish by PhK enzyme (normal in PYGL, low in PHKA2), inheritance (PYGL=AR, PHKA2=X-linked), "
            "and sequencing; (2) PHKG2 (GSD IXc) — X-linked vs AR; PHKG2 has fibrosis risk; "
            "liver biopsy + PHKG2 sequencing; (3) GSD Ia/Ib — severe lactic acidosis, hyperuricemia; "
            "ABSENT in GSD IXa; (4) AGL/GSD III — CK elevated (IIIa), no resolution at puberty; "
            "(5) Viral hepatitis — serology; (6) Fatty liver — BMI, insulin resistance"
        ),
        "diet_treatment": "Cornstarch if symptomatic (not mandatory in all); frequent meals; high-complex-carb diet; spontaneous resolution expected; NO dietary restrictions (unlike GSD Ia); reassurance is key",
        "gene_therapy_status": "No gene therapy needed (benign, resolves); dietary management is temporary and sufficient",
        "critical_ci": (
            "CRITICAL CI: (1) Not doing genetic testing — cannot distinguish PHKA2 from PYGL or PHKG2 "
            "clinically (PHKG2 has fibrosis risk requiring different follow-up); "
            "(2) Not counselling X-linked inheritance — mother is carrier; daughters at 50% carrier risk; "
            "(3) Over-treating with lifelong cornstarch — resolves at puberty; adjust/stop treatment; "
            "(4) Missing PHKB (AR) or PHKG2 in female patients — PHKA2 causes mild disease in females "
            "(carriers); severe disease in females suggests PHKB or PHKG2 (AR)"
        ),
        "nbs_marker": "Not in standard NBS; hepatomegaly on examination; mildly elevated transaminases + fasting hypoglycemia; PhK enzyme assay in erythrocytes; PHKA2 gene sequencing",
        "key_biomarker": "Erythrocyte PhK activity (low/absent); plasma glucose (mild fasting hypoglycemia); plasma lactate (NORMAL or mildly elevated); CK (NORMAL in PHKA2); liver biopsy: glycogen-laden hepatocytes, PAS-D positive",
        "severity_spectrum": "Mild hepatomegaly only (most males; resolves adulthood) → Moderate (frequent hypoglycemia, cornstarch needed) → Rare severe (persistent into adulthood, very rare)",
        "founder_variant": "No dominant single founder; diverse mutations across PHKA2; missense variants in regulatory domains; p.Arg1042Trp — European; c.3582+1G>T — splice, European",
        "key_variants": [
            "p.Arg1042Trp — European, moderate phenotype",
            "c.3582+1G>T — splice, European, classic",
            "p.Glu1124Lys — Asian, mild",
            "p.Arg43Gln — null-equivalent, severe (rare)",
            "p.Gly1017Asp — kinase domain, classic",
        ],
        "seed": SEED_BASE + 7,
    },
    # ── SLC2A2 — GLUT2 (GSD XI / Fanconi-Bickel syndrome) ──
    {
        "gene": "SLC2A2", "alias": "SLC2A2 — GLUT2 Facilitative Glucose/Galactose Transporter (GSD XI / Fanconi-Bickel Syndrome)",
        "aa": "524 aa", "kDa": "57.3 kDa",
        "gene_class": "transporter",
        "locus": "3q26.2", "omim_gene": 138160,
        "phenotype": "GSD XI / Fanconi-Bickel — hepatorenal glycogen accumulation + RENAL FANCONI SYNDROME; galactose-free + fructose-restricted diet; postprandial hyperglycemia + fasting hypoglycemia",
        "disease": (
            "SLC2A2 biallelic loss → GSD type XI / Fanconi-Bickel syndrome (FBS, OMIM #227810). SLC2A2 "
            "encodes GLUT2 (glucose transporter 2, 524aa, 57.3 kDa), a bidirectional high-Km facilitative "
            "transporter for glucose, galactose, fructose, and glucosamine, expressed in hepatocytes, "
            "pancreatic beta-cells, renal proximal tubules, and intestinal epithelium. GLUT2 is unique "
            "in being a HIGH-Km transporter (glucose Km ~17 mM), functioning as a glucose sensor/overflow "
            "system. Deficiency → (1) LIVER: glucose and galactose cannot exit hepatocytes → trapped → "
            "hepatomegaly with glycogen/fat accumulation + postprandial hyperglycemia (cannot release glucose "
            "from liver after meals); fasting hypoglycemia (cannot absorb hepatic glucose effectively); "
            "(2) KIDNEY: GLUT2 mediates proximal tubular glucose reabsorption → loss → RENAL FANCONI "
            "SYNDROME: glucosuria with normoglycemia, aminoaciduria, phosphaturia → phosphopenic rickets, "
            "tubular acidosis; (3) PANCREAS: beta-cell glucose sensing impaired (GLUT2 is the glucose sensor "
            "coupler); (4) GALACTOSE: GLUT2 transports galactose → cannot be reabsorbed from renal tubules → "
            "galactosuria; cannot be fully metabolised in liver → accumulates → worsens glycogen load. "
            "Incidence: very rare (<200 cases worldwide; enriched Middle Eastern, Turkish, Mediterranean populations)."
        ),
        "inheritance": "Autosomal recessive. SLC2A2 3q26.2. Very rare (< 200 cases worldwide). Enriched Middle Eastern, Turkish, Japanese, Mediterranean.",
        "hallmark": (
            "SLC2A2/Fanconi-Bickel HALLMARKS: (1) HEPATORENAL GLYCOGEN ACCUMULATION — "
            "hepatomegaly (massive in infancy) from glycogen + lipid; nephromegaly from tubular glycogen; "
            "(2) RENAL FANCONI SYNDROME — the DEFINING CLINICAL FEATURE of FBS: "
            "GLUCOSURIA with NORMOGLYCEMIA (glucose in urine despite normal or low blood glucose), "
            "aminoaciduria (all amino acids), phosphaturia (→ phosphopenic RICKETS), "
            "bicarbonuria (→ tubular acidosis), uricosuria; "
            "treat with phosphate supplementation + vitamin D for rickets; "
            "(3) GALACTOSE-FREE + FRUCTOSE-RESTRICTED DIET — galactose is transported via GLUT2 in liver; "
            "without GLUT2, galactose cannot exit hepatocytes → accumulates → worsens glycogen load; "
            "fructose similarly trapped; eliminate both from diet; "
            "(4) POSTPRANDIAL HYPERGLYCEMIA + FASTING HYPOGLYCEMIA — GLUT2 bidirectional defect: "
            "cannot release glucose from liver after meals → postprandial hyperglycemia; "
            "cannot supply hepatic glucose to circulation during fasting → fasting hypoglycemia; "
            "this DUAL glycaemic phenotype is pathognomonic for FBS; "
            "(5) FAILURE TO THRIVE — a cardinal feature; poor growth from infancy; "
            "metabolic acidosis (tubular) + phosphopenic rickets + recurrent hypoglycemia all contribute; "
            "(6) PANCREATIC BETA-CELL GLUCOSE SENSING — GLUT2 loss → impaired glucose sensing; "
            "insulin secretion dysregulated (postprandial insulin response blunted); "
            "(7) CORNSTARCH — for fasting hypoglycemia; frequent meals mandatory; "
            "(8) LONG-TERM PROGNOSIS — generally good with management; "
            "hepatomegaly decreases with age; Fanconi syndrome may improve; renal function preserved if rickets treated"
        ),
        "key_ddx": (
            "SLC2A2 DDx: (1) Galactosemia (GALT/GALE) — galactosemia type 1/2; galactose-1-phosphate elevated; "
            "no glycogen accumulation pattern; different enzyme; "
            "(2) G6PC (GSD Ia) — hepatomegaly + hypoglycemia; primary lactic acidosis (absent in FBS); "
            "NO Fanconi syndrome in GSD Ia; "
            "(3) Dent disease (CLCN5) — proximal tubular dysfunction + nephrolithiasis; "
            "no hepatic glycogen disease; different transport defect; "
            "(4) Lowe syndrome (OCRL) — proximal tubular disease + cataracts + hypotonia; "
            "X-linked; no glycogen; (5) Fructose intolerance (ALDOB) — fructose-triggered severe hypoglycemia; "
            "no tubular disease pattern with normoglycaemic glucosuria"
        ),
        "diet_treatment": "Galactose-free + fructose-restricted diet (MANDATORY); frequent meals; raw cornstarch (fasting hypoglycemia); phosphate supplementation + vitamin D for rickets (Fanconi); bicarbonate for acidosis; manage tubular losses",
        "gene_therapy_status": "No approved gene therapy; dietary management is cornerstone; supportive for Fanconi syndrome (mineral replacement)",
        "critical_ci": (
            "CRITICAL CI: (1) Allowing galactose in diet — galactose cannot exit hepatocytes without GLUT2; "
            "accumulates → worsens hepatomegaly and glycogen load; "
            "(2) Missing Fanconi syndrome — phosphaturia → rickets → pathological fractures if untreated; "
            "glucosuria with normoglycemia is the clinical key; "
            "(3) Interpreting glucosuria as diabetes — FBS: glucosuria at NORMAL blood glucose; "
            "diabetes: glucosuria at HIGH blood glucose; different; "
            "(4) Not replacing phosphate for rickets — severe metabolic bone disease without phosphate"
        ),
        "nbs_marker": "Not in standard NBS; failure to thrive + hepatomegaly + glucosuria with normoglycemia in infancy; urine glucose/protein/amino acids/phosphate; SLC2A2 sequencing",
        "key_biomarker": "Glucosuria with normoglycemia (PATHOGNOMONIC of renal Fanconi + FBS context); aminoaciduria (all AAs); phosphaturia + rickets on X-ray; plasma glucose: postprandial HIGH + fasting LOW; liver biopsy: massive hepatomegaly with glycogen+lipid",
        "severity_spectrum": "Severe neonatal (massive hepatomegaly, severe Fanconi, failure to thrive) → Moderate (manageable with diet + mineral supplementation) → Mild (late-diagnosed adults, milder Fanconi)",
        "founder_variant": "p.Gly318Arg — Middle Eastern/Turkish; p.Leu203Pro — Japanese; p.Arg44Ter — pan-ethnic null",
        "key_variants": [
            "p.Gly318Arg — Middle Eastern/Turkish, common",
            "p.Leu203Pro — Japanese, common",
            "p.Arg44Ter — null, severe",
            "p.Arg498Cys — European, moderate",
            "p.Trp444Ter — null, classic",
        ],
        "seed": SEED_BASE + 8,
    },
    # ── GYS2 — Liver Glycogen Synthase (GSD 0a) ──
    {
        "gene": "GYS2", "alias": "GYS2 — Liver Glycogen Synthase (GSD 0a / Glycogen Synthase Deficiency)",
        "aa": "703 aa", "kDa": "80.4 kDa",
        "gene_class": "synthase",
        "locus": "12p12.1", "omim_gene": 138571,
        "phenotype": "GSD 0a — UNIQUE: CANNOT STORE GLYCOGEN; fasting KETOTIC hypoglycemia + postprandial HYPERGLYCEMIA; NO glycogen accumulation; excellent prognosis",
        "disease": (
            "GYS2 biallelic loss → GSD type 0a (Liver Glycogen Synthase Deficiency, OMIM #240600). GYS2 "
            "encodes liver glycogen synthase (GS2, 703aa, 80.4 kDa), which catalyses the final step in "
            "glycogen synthesis: transfers glucose from UDP-glucose to the growing glycogen chain via "
            "alpha-1,4-glycosidic bonds. UNIQUE AMONG ALL GSDs: GSD 0a is the ONLY GSD where glycogen "
            "CANNOT BE SYNTHESISED (not merely degraded/accumulated). The liver has NO glycogen reserve. "
            "Clinical paradox: (1) FASTING KETOTIC HYPOGLYCEMIA — no glycogen to mobilize → blood glucose "
            "falls with fasting → compensatory ketogenesis (ketosis is primary defense when glycogen is "
            "absent → brain uses ketone bodies); hypoglycemia episodes typically mild to moderate; "
            "(2) POSTPRANDIAL HYPERGLYCEMIA — glucose absorbed from meals CANNOT be stored as glycogen → "
            "transient hyperglycemia; may mimic early diabetes or impaired glucose tolerance; "
            "(3) HbA1c may be LOW-NORMAL to NORMAL despite postprandial hyperglycemia (episodes are brief "
            "and transient; HbA1c reflects 3-month average → may not capture brief postprandial spikes); "
            "NO hepatomegaly (no glycogen accumulation); normal liver function; normal intelligence. "
            "Excellent prognosis with dietary management. Rare; prevalence unknown; "
            "likely underdiagnosed (may present only as ketotic hypoglycemia in infancy)."
        ),
        "inheritance": "Autosomal recessive. GYS2 12p12.1. Very rare (exact prevalence unknown; likely underdiagnosed). Excellent prognosis.",
        "hallmark": (
            "GYS2/GSD 0a HALLMARKS: (1) OPPOSITE OF ALL OTHER GSDs — cannot STORE glycogen; "
            "NO glycogen accumulation; NO hepatomegaly; liver is NORMAL in size; "
            "this is the most important conceptual point in the entire GSD field; "
            "(2) FASTING KETOTIC HYPOGLYCEMIA — primary presentation in infants/children; "
            "morning seizures or lethargy (overnight fast); urine ketones HIGH (primary adaptation); "
            "plasma glucose low during fasting; hypoglycemia is the clinical trigger for diagnosis; "
            "(3) POSTPRANDIAL HYPERGLYCEMIA — glucose absorbed after meals cannot be buffered into glycogen; "
            "glucose spikes 1-2h post-meal; may mimic T2DM or impaired glucose tolerance in adults; "
            "(4) DUAL GLYCAEMIC PHENOTYPE — fasting LOW + postprandial HIGH is pathognomonic; "
            "no other common condition causes this exact dual pattern; "
            "(5) HIGH PROTEIN DIET + CORNSTARCH — high protein diet provides gluconeogenic substrate "
            "(alanine, glutamine) to maintain fasting glucose; raw cornstarch for nocturnal hypoglycemia; "
            "(6) HbA1C — may be paradoxically LOW (postprandial spikes are brief → time-averaged glucose "
            "may be near normal); HbA1c alone is NOT adequate for monitoring GSD 0a; "
            "(7) NO LIVER BIOPSY NEEDED — no pathological glycogen to find; liver enzyme normal; "
            "(8) COGNITIVE OUTCOME — excellent if hypoglycemia treated promptly; "
            "(9) ADULT PROGNOSIS — hypoglycemia typically improves as fasting tolerance increases with age; "
            "postprandial hyperglycemia may persist but generally benign"
        ),
        "key_ddx": (
            "GYS2 DDx: (1) Ketotic hypoglycemia of childhood — common, idiopathic; "
            "no postprandial hyperglycemia; normal glycogen synthesis; resolves by puberty; "
            "(2) Type 1 diabetes — postprandial hyperglycemia but HIGH HbA1c; autoantibodies present; "
            "NO fasting ketotic hypoglycemia (pre-insulin); different pattern; "
            "(3) GSD Ia — severe lactic acidosis, hepatomegaly, hyperuricemia; absent in GSD 0a; "
            "(4) Fatty acid oxidation disorders (MCAD) — fasting ketotic hypoglycemia; "
            "BUT: acylcarnitine profile elevated; no postprandial hyperglycemia; "
            "(5) Insulinoma — fasting hypoglycemia but with high insulin and LOW ketones; "
            "GSD 0a: hypoglycemia WITH high ketones"
        ),
        "diet_treatment": "High protein diet (provides gluconeogenic substrate for fasting); raw cornstarch for nocturnal fasting prevention; frequent meals; avoid prolonged fasting; diet typically sufficient for management; excellent prognosis",
        "gene_therapy_status": "No gene therapy pursued (dietary management highly effective; benign disorder); cornstarch + high protein is evidence-based",
        "critical_ci": (
            "CRITICAL CI: (1) Diagnosing as T2DM and prescribing insulin/metformin — GSD 0a postprandial "
            "hyperglycemia is benign and transient; insulin causes profound hypoglycemia; "
            "(2) Prolonged fasting in GSD 0a — no glycogen reserve; "
            "hypoglycemia occurs faster than in normal individuals; "
            "(3) Missing the fasting ketotic hypoglycemia component — if only postprandial glucose "
            "abnormality detected, GSD 0a may be missed; "
            "(4) Liver biopsy to investigate 'hepatic storage disease' — biopsy is NORMAL in GSD 0a; "
            "wasted procedure with risk; molecular diagnosis is definitive"
        ),
        "nbs_marker": "Not in standard NBS; morning seizures/lethargy with hypoglycemia + ketonuria in infancy; fasting glucose + ketones + blood glucose curve (fast + mixed meal); GYS2 sequencing",
        "key_biomarker": "Fasting glucose (LOW) + urine ketones (HIGH) simultaneously (ketotic hypoglycemia); postprandial glucose (HIGH — spikes post-meal); HbA1c (may be paradoxically normal/low); NO hepatomegaly; GYS2 sequencing",
        "severity_spectrum": "Neonatal hypoglycemia (severe fasting intolerance) → Childhood ketotic hypoglycemia (classic presentation) → Adult mild postprandial hyperglycemia (benign, often resolved)",
        "founder_variant": "No dominant founder; diverse private mutations worldwide; p.Asp464Asn — European hypomorphic; p.Arg385Ter — null, classic",
        "key_variants": [
            "p.Arg385Ter — null, classic, severe fasting HG",
            "p.Asp464Asn — European, hypomorphic, mild",
            "p.Gly304Asp — Asian, moderate",
            "p.Trp543Ter — null, severe",
            "p.Val303Ile — hypomorphic, mild postprandial",
        ],
        "seed": SEED_BASE + 9,
    },
]


# ── Patient cohort generator ───────────────────────────────────────────────────
def _make_patients_for_gene(g):
    rng = random.Random(g["seed"])
    gene = g["gene"]
    patients = []
    n = 40
    for i in range(n):
        pid = f"GSD-{gene}-{i+1:03d}"
        # Gene-specific simulation
        if gene == "G6PC":
            age_dx = rng.choice(
                [rng.uniform(0, 0.3)] * 25 +   # neonatal/infant hypoglycemia
                [rng.uniform(0.3, 2)] * 10 +    # infant hepatomegaly
                [rng.uniform(2, 10)] * 5        # late clinical
            )
            subtype = rng.choice(["classic_severe"] * 20 + ["moderate"] * 12 + ["mild_residual"] * 8)
            tx_controlled = rng.random() < 0.72  # cornstarch/diet
            has_adenoma = age_dx < 5 and rng.random() < 0.30  # adenoma eventually
            neurological = False
            deceased = not tx_controlled and rng.random() < 0.12
        elif gene == "SLC37A4":
            age_dx = rng.choice(
                [rng.uniform(0, 0.5)] * 22 +   # neonatal hypoglycemia + infection
                [rng.uniform(0.5, 3)] * 13 +    # hepatomegaly + neutropenia
                [rng.uniform(3, 15)] * 5        # IBD discovery
            )
            subtype = rng.choice(["classic_neutropenia_ibd"] * 20 + ["neutropenia_mild_ibd"] * 12 + ["hepatic_predominant"] * 8)
            tx_controlled = rng.random() < 0.68
            has_adenoma = rng.random() < 0.25
            neurological = False
            deceased = not tx_controlled and rng.random() < 0.10
        elif gene == "AGL":
            age_dx = rng.choice(
                [rng.uniform(0.3, 2)] * 18 +   # infant hepatomegaly
                [rng.uniform(2, 15)] * 15 +     # childhood CK rise / myopathy
                [rng.uniform(15, 40)] * 7       # adult myopathy dominant
            )
            subtype = rng.choice(["iiia_liver_muscle"] * 34 + ["iiib_liver_only"] * 6)
            tx_controlled = rng.random() < 0.62  # high protein diet
            has_adenoma = False
            neurological = subtype == "iiia_liver_muscle" and rng.random() < 0.45  # myopathy
            deceased = rng.random() < 0.05  # cirrhosis/HCC rare
        elif gene == "GBE1":
            age_dx = rng.choice(
                [rng.uniform(0, 0.5)] * 15 +   # neonatal hepatic/hydrops
                [rng.uniform(0.5, 5)] * 12 +    # infant liver failure
                [rng.uniform(20, 60)] * 13      # adult APBD
            )
            subtype = rng.choice(["classic_hepatic_fatal"] * 15 + ["neuromuscular_childhood"] * 10 + ["apbd_adult"] * 12 + ["neonatal_hydrops"] * 3)
            tx_controlled = subtype == "classic_hepatic_fatal" and rng.random() < 0.60  # liver Tx
            has_adenoma = False
            neurological = subtype in ("apbd_adult", "neuromuscular_childhood") and rng.random() < 0.85
            deceased = (subtype == "neonatal_hydrops") or (subtype == "classic_hepatic_fatal" and not tx_controlled and rng.random() < 0.50)
        elif gene == "PYGM":
            age_dx = rng.choice(
                [rng.uniform(5, 20)] * 15 +    # childhood exercise symptoms
                [rng.uniform(20, 45)] * 20 +    # adult delay in diagnosis
                [rng.uniform(0, 5)] * 5         # early childhood severe
            )
            subtype = rng.choice(["classic_exercise_intolerance"] * 25 + ["with_myoglobinuria"] * 12 + ["mild_only"] * 3)
            tx_controlled = rng.random() < 0.70  # sucrose trick + aerobic training
            has_adenoma = False
            neurological = False
            deceased = subtype == "with_myoglobinuria" and rng.random() < 0.05  # AKI
        elif gene == "PYGL":
            age_dx = rng.choice(
                [rng.uniform(0.3, 3)] * 25 +   # infant hepatomegaly
                [rng.uniform(3, 15)] * 12 +     # childhood elevated LFTs
                [rng.uniform(15, 40)] * 3       # adult incidental
            )
            subtype = rng.choice(["mild_hepatomegaly"] * 25 + ["moderate_hypoglycemia"] * 12 + ["resolved_adult"] * 3)
            tx_controlled = rng.random() < 0.88  # very good prognosis
            has_adenoma = False
            neurological = False
            deceased = False
        elif gene == "PFKM":
            age_dx = rng.choice(
                [rng.uniform(5, 25)] * 20 +    # exercise intolerance onset
                [rng.uniform(25, 50)] * 15 +    # adult hemolysis discovery
                [rng.uniform(0, 5)] * 5         # infantile polyglucosan
            )
            subtype = rng.choice(["classic_exercise_hemolysis"] * 25 + ["hemolysis_only"] * 8 + ["infantile_severe"] * 4 + ["myopathy_dominant"] * 3)
            tx_controlled = rng.random() < 0.65  # dietary timing management
            has_adenoma = False
            neurological = subtype == "infantile_severe" and rng.random() < 0.80
            deceased = subtype == "infantile_severe" and rng.random() < 0.30
        elif gene == "PHKA2":
            age_dx = rng.choice(
                [rng.uniform(0.3, 3)] * 25 +   # infant hepatomegaly
                [rng.uniform(3, 12)] * 12 +     # childhood elevated transaminases
                [rng.uniform(12, 40)] * 3       # adult incidental (carrier female)
            )
            subtype = rng.choice(["mild_hepatomegaly"] * 25 + ["moderate_hypoglycemia"] * 12 + ["resolved_adult"] * 3)
            tx_controlled = rng.random() < 0.90  # very good prognosis
            has_adenoma = False
            neurological = False
            deceased = False
        elif gene == "SLC2A2":
            age_dx = rng.choice(
                [rng.uniform(0, 1)] * 25 +     # neonatal/infant Fanconi + hepatomegaly
                [rng.uniform(1, 10)] * 12 +     # childhood rickets + failure to thrive
                [rng.uniform(10, 40)] * 3       # adult late diagnosis
            )
            subtype = rng.choice(["classic_fanconi_hepatomegaly"] * 22 + ["rickets_predominant"] * 12 + ["mild_fanconi"] * 6)
            tx_controlled = rng.random() < 0.68  # diet + mineral replacement
            has_adenoma = False
            neurological = False
            deceased = not tx_controlled and rng.random() < 0.08
        else:  # GYS2 — GSD 0a
            age_dx = rng.choice(
                [rng.uniform(0, 1)] * 18 +     # neonatal/infant ketotic HG
                [rng.uniform(1, 10)] * 17 +     # childhood morning HG
                [rng.uniform(10, 40)] * 5       # adult postprandial HG discovered
            )
            subtype = rng.choice(["ketotic_hypoglycemia_classic"] * 22 + ["dual_hyperhypo"] * 12 + ["mild_postprandial"] * 6)
            tx_controlled = rng.random() < 0.85  # high protein + cornstarch effective
            has_adenoma = False
            neurological = False
            deceased = False

        patients.append({
            "pid": pid,
            "gene": gene,
            "subtype": subtype,
            "age_dx_y": round(max(0.0, age_dx), 2),
            "tx_controlled": tx_controlled,
            "has_adenoma": has_adenoma,
            "neurological": neurological,
            "deceased": deceased,
        })
    return patients


ALL_PATIENTS = []
for _g in GSD_GENES:
    _pts = _make_patients_for_gene(_g)
    _g["n_patients"] = len(_pts)
    _g["patients"] = _pts
    ALL_PATIENTS.extend(_pts)


# ─── API: get_overview ──────────────────────────────────────────────────────────
def get_overview():
    total = len(ALL_PATIENTS)
    n_tx = sum(1 for p in ALL_PATIENTS if p["tx_controlled"])
    n_adenoma = sum(1 for p in ALL_PATIENTS if p["has_adenoma"])
    n_neuro = sum(1 for p in ALL_PATIENTS if p["neurological"])
    n_deceased = sum(1 for p in ALL_PATIENTS if p["deceased"])

    gene_summary = []
    for g in GSD_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "locus": g["locus"],
            "gene_class": g["gene_class"],
            "n_patients": g["n_patients"],
            "diet_treatment": g["diet_treatment"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "pct_tx": round(100 * sum(1 for p in pts if p["tx_controlled"]) / len(pts), 1),
            "pct_adenoma": round(100 * sum(1 for p in pts if p["has_adenoma"]) / len(pts), 1),
            "pct_neuro": round(100 * sum(1 for p in pts if p["neurological"]) / len(pts), 1),
            "pct_deceased": round(100 * sum(1 for p in pts if p["deceased"]) / len(pts), 1),
        })

    return {
        "atlas": "GSD-Atlas — Complete 10-Gene Glycogen Storage Disorder Atlas",
        "n_genes": len(GSD_GENES),
        "n_patients": total,
        "seeds": [g["seed"] for g in GSD_GENES],
        "genes_covered": [g["gene"] for g in GSD_GENES],
        "gene_classes": {
            "phosphatase":   ["G6PC"],
            "transporter":   ["SLC37A4", "SLC2A2"],
            "glucosidase":   ["AGL"],
            "transferase":   ["GBE1"],
            "phosphorylase": ["PYGM", "PYGL"],
            "kinase":        ["PFKM", "PHKA2"],
            "synthase":      ["GYS2"],
        },
        "aggregate_clinical": {
            "pct_tx_controlled": round(100 * n_tx / total, 1),
            "pct_adenoma":       round(100 * n_adenoma / total, 1),
            "pct_neurological":  round(100 * n_neuro / total, 1),
            "pct_deceased":      round(100 * n_deceased / total, 1),
        },
        "gene_summary": gene_summary,
        "critical_clinical_rules": [
            "G6PC (GSD Ia): NEVER glucose boluses (G6P trapped → worsens lactic acidosis); cornstarch 1-2g/kg q4-6h is cornerstone; NO fructose/galactose; lactic acidosis is PRIMARY not secondary; hepatocellular adenoma >90% by 30y → HCC 10-15%; AFP + ultrasound every 6-12 months mandatory",
            "SLC37A4 (GSD Ib): IDENTICAL to GSD Ia biochemistry PLUS NEUTROPENIA (ANC <1500) — G-CSF MANDATORY; IBD (Crohn's-like) in ~75%; empagliflozin emerging for neutrophil/IBD; DTX401 (G6PC gene therapy) does NOT treat GSD Ib",
            "AGL (GSD IIIa/b): HIGH PROTEIN DIET (25-30%) is the primary treatment — different from GSD I; liver transplant does NOT improve muscle disease in IIIa (muscle GDE remains deficient); limit dextrin (short-chain, diastase-sensitive) distinguishes from GSD IV polyglucosan",
            "GBE1 (GSD IV): polyglucosan (diastase-RESISTANT amylopectin-like) accumulation; liver transplant CURATIVE for classic hepatic form; liver Tx does NOT help APBD (adult neuromuscular form); Ashkenazi Jewish p.Tyr329Ser enriched in APBD",
            "PYGM (McArdle, GSD V): McArdle SIGN — lactate FAILS to rise (ammonia rises normally) on forearm ischemic test; SUCROSE TRICK (35g oral, 5 min before exercise) prevents myoglobinuria; SECOND WIND at 7-10 min aerobic exercise; p.Arg50Ter in ~80% European alleles",
            "PYGL (GSD VI): GENERALLY BENIGN — resolves spontaneously at puberty/adolescence; NO lactic acidosis or hyperuricemia (key DDx from GSD Ia); cornstarch only if symptomatic hypoglycemia; excellent long-term prognosis",
            "PFKM (GSD VII, Tarui): HEMOLYTIC ANEMIA in ALL patients (RBC PFK uses PFKM); HIGH CARBOHYDRATE PARADOX — carbs WORSEN exercise tolerance (OPPOSITE of McArdle sucrose trick); Ashkenazi Jewish p.Asp181del founder; forearm test same as PYGM (lactate fails to rise)",
            "PHKA2 (GSD IXa): X-LINKED (Xp22.13) — different from all other listed GSDs (AR); GENERALLY BENIGN; resolves adulthood; PHKG2 (AR) has fibrosis risk — must distinguish by genetics; most common liver GSD",
            "SLC2A2 (GSD XI, Fanconi-Bickel): GALACTOSE-FREE + FRUCTOSE-RESTRICTED diet MANDATORY; FANCONI SYNDROME — glucosuria with normoglycemia + aminoaciduria + phosphaturia + rickets; DUAL GLYCAEMIC PHENOTYPE (fasting LOW + postprandial HIGH) is pathognomonic",
            "GYS2 (GSD 0a): UNIQUE — CANNOT STORE GLYCOGEN; fasting KETOTIC hypoglycemia + postprandial HYPERGLYCEMIA; NO hepatomegaly; high protein diet + cornstarch; do NOT give insulin for postprandial hyperglycemia (causes profound hypoglycemia); HbA1c may be paradoxically normal",
        ],
    }


# ─── API: get_breakdown ─────────────────────────────────────────────────────────
def get_breakdown():
    gene_rows = []
    for g in GSD_GENES:
        pts = g["patients"]
        gene_rows.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "gene_class": g["gene_class"],
            "n_patients": g["n_patients"],
            "seed": g["seed"],
            "phenotype": g["phenotype"],
            "inheritance": g["inheritance"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "diet_treatment": g["diet_treatment"],
            "gene_therapy_status": g["gene_therapy_status"],
            "critical_ci": g["critical_ci"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "key_variants": g["key_variants"],
            # Aggregate from cohort
            "pct_tx": round(100 * sum(1 for p in pts if p["tx_controlled"]) / len(pts), 1),
            "pct_adenoma": round(100 * sum(1 for p in pts if p["has_adenoma"]) / len(pts), 1),
            "pct_neuro": round(100 * sum(1 for p in pts if p["neurological"]) / len(pts), 1),
            "pct_deceased": round(100 * sum(1 for p in pts if p["deceased"]) / len(pts), 1),
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })
    return {
        "genes": gene_rows,
        "total": len(GSD_GENES),
        "total_patients": len(ALL_PATIENTS),
    }


# ─── API: get_definitions ─────────────────────────────────────────────────────
def get_definitions():
    return {
        "atlas": "GSD-Atlas — Complete 10-Gene Glycogen Storage Disorder Atlas",
        "gsd_overview": {
            "full_name": "Glycogen Storage Diseases — inherited defects in enzymes or transporters of glycogen synthesis, degradation, or regulation → glycogen accumulation or absence → hypoglycemia, myopathy, hepatomegaly, or specific organ failure",
            "genes_in_atlas": 10,
            "total_known_gsd_types": "~20 defined GSD types (types 0–XV and variants)",
            "collective_incidence": "Collectively ~1 in 20,000-40,000; GSD I most common hepatic GSD (1 in 100,000); GSD V most common muscle GSD (1 in 100,000)",
            "inheritance_note": "Most autosomal recessive; PHKA2 (GSD IXa) is X-linked recessive; GBE1 adult form has founder variants in Ashkenazi Jewish",
            "nbs_note": "None in standard expanded NBS (no specific acylcarnitine or amino acid marker); clinical presentation drives diagnosis; gene panel/WES is diagnostic gold standard",
        },
        "definitions": [
            {
                "term": "GSD — Glycogen Storage Disease",
                "definition": "A group of inherited disorders caused by defects in enzymes or transporters involved in glycogen metabolism. Collectively ~20 types, classified by enzyme defect and tissue involved. Hepatic GSDs (G6PC, SLC37A4, AGL, GBE1, PYGL, PHKA2, SLC2A2, GYS2) present with hepatomegaly and hypoglycemia. Muscle GSDs (PYGM, PFKM, AGL in IIIa) present with exercise intolerance and myopathy. GSD 0a (GYS2) is unique — cannot store glycogen → no accumulation.",
            },
            {
                "term": "Cornstarch — Raw Uncooked Cornstarch Therapy",
                "definition": "Raw (uncooked) cornstarch is a slowly digested, slowly absorbed complex glucose polymer (amylopectin). It acts as a sustained-release glucose source due to incomplete gelatinisation (raw starch resists rapid alpha-amylase digestion). Used in GSD Ia (G6PC), GSD Ib (SLC37A4), GSD IIIa/b (AGL, if needed), GSD VI (PYGL, if symptomatic), GSD IXa (PHKA2, if symptomatic), and GSD 0a (GYS2, for nocturnal hypoglycemia). Cooked starch digests rapidly and is NOT equivalent. Glycosade (waxy maize starch) is an extended-release formulation for overnight use.",
            },
            {
                "term": "McArdle Sign — Forearm Ischemic Exercise Test",
                "definition": "A bedside diagnostic test for muscle glycogenolysis defects (GSD V/PYGM, GSD VII/PFKM). Procedure: inflate BP cuff above systolic on forearm; patient squeezes a ball for 60 seconds; release cuff; sample venous blood at 0, 1, 3, 6 min for lactate and ammonia. Normal: venous lactate rises ≥3× baseline AND ammonia rises. McArdle/Tarui result: lactate FAILS to rise (no anaerobic glycolysis from muscle glycogen/PFK block) BUT ammonia rises normally (purine nucleotide cycle intact, indicating the patient exercised). A double-fail (lactate AND ammonia both fail to rise) suggests inadequate effort — not a GSD. Pathognomonic for PYGM and PFKM deficiencies.",
            },
            {
                "term": "Sucrose Trick — PYGM McArdle Pre-Exercise Sucrose",
                "definition": "35g of oral sucrose 5 minutes before exercise in PYGM (McArdle, GSD V). Mechanism: sucrose → glucose + fructose → blood glucose rises → working muscle can use blood glucose as the primary fuel (bypassing the myophosphorylase block). Prevents exercise-induced myoglobinuria and improves exercise tolerance. Evidence-based (Level B). CONTRAINDICATED in PFKM (GSD VII/Tarui) — carbohydrates worsen exercise tolerance due to the high-carbohydrate paradox (G6P accumulates before PFK block).",
            },
            {
                "term": "Second Wind Phenomenon — PYGM McArdle",
                "definition": "Characteristic of GSD V (McArdle disease). At 7-10 minutes of continuous moderate aerobic exercise, symptoms (myalgia, fatigue, cramps) dramatically improve. Mechanism: hepatic glycogenolysis provides blood glucose; free fatty acid mobilisation accelerates; the working muscle shifts from relying on (unavailable) muscle glycogen to blood-borne glucose and FFA as primary fuels. Patients are instructed to 'pace through' the first 7-10 minutes to trigger second wind. Aerobic exercise training (progressive conditioning) amplifies this response and is therapeutic.",
            },
            {
                "term": "High Carbohydrate Paradox — PFKM GSD VII Tarui",
                "definition": "Unique to GSD VII (PFKM/Tarui disease). Carbohydrate ingestion BEFORE exercise WORSENS exercise tolerance. Mechanism: glucose → G6P → F6P → F6P accumulates BEFORE the PFK-M block → inhibits remaining PFK activity → also feedback-inhibits hexokinase → net reduction in substrate flow through glycolysis. Additionally, blood glucose suppresses FFA mobilisation (insulin response) — the only alternative fuel for PFKM muscle is blocked. Therefore, PFKM patients perform BETTER in a fasted state (more FFA available). CONTRAST: McArdle (PYGM) — sucrose trick helps (blood glucose bypasses glycogen block). PFKM: sucrose is contraindicated.",
            },
            {
                "term": "Glucose-6-Phosphate (G6P) — Trapped Metabolite in GSD Ia/Ib",
                "definition": "G6P is the central metabolite at the junction of glycogenolysis, gluconeogenesis, glycolysis, and the pentose phosphate pathway in the liver. In GSD Ia (G6PC) and GSD Ib (SLC37A4), G6P cannot be dephosphorylated to free glucose and cannot exit the ER or hepatocyte. Accumulation causes: (1) glycogen synthesis from excess G6P → hepatomegaly; (2) overflow into glycolysis → pyruvate → primary lactic acidosis; (3) overflow into pentose phosphate pathway → uric acid synthesis + NADPH/FA synthesis → hyperuricemia + hypertriglyceridemia. Glucose boluses (IV) convert rapidly to G6P → worsen all three pathways → CONTRAINDICATED.",
            },
            {
                "term": "G-CSF — Granulocyte Colony-Stimulating Factor — MANDATORY in GSD Ib",
                "definition": "Filgrastim or lenograstim (recombinant G-CSF) is MANDATORY therapy for GSD Ib (SLC37A4). Mechanism: SLC37A4 expressed in neutrophils; G6PT loss → neutrophil functional impairment and apoptosis → neutropenia (ANC <1500/µL) → severe recurrent bacterial and fungal infections. G-CSF stimulates neutrophil production and maturation, increasing ANC and improving neutrophil function. G-CSF also independently reduces IBD severity in GSD Ib (mechanism not fully elucidated — may relate to neutrophil gut trafficking). Target ANC 1000-1500/µL. NOT needed in GSD Ia (G6PC) — no neutropenia in GSD Ia.",
            },
            {
                "term": "Limit Dextrin — GSD III Glycogen Accumulation",
                "definition": "Limit dextrin is the short-chain glycogen that remains after glycogen phosphorylase has degraded the outer chains to within 4 residues of each branch point (the limit of phosphorylase action). In GSD IIIa/b (AGL/glycogen debranching enzyme deficiency), the debranching enzyme that removes these residual branch glucoses is absent → limit dextrin accumulates in liver (both IIIa and IIIb) and muscle (IIIa only). On biopsy: PAS-D positive but diastase-SENSITIVE (normal glycogen structure, just truncated). CONTRAST with GSD IV (GBE1): polyglucosan bodies (diastase-RESISTANT, abnormal branching).",
            },
            {
                "term": "Polyglucosan Body — GSD IV GBE1 Amylopectin-like Glycogen",
                "definition": "Polyglucosan bodies are abnormal glucose polymers with very few branch points and long outer chains — resembling amylopectin (plant starch) in structure, unlike normal animal glycogen (highly branched). In GSD IV (GBE1 deficiency), the branching enzyme that creates alpha-1,6 branch points is absent → glycogen lacks branches → long linear chains form polyglucosan bodies. Histochemistry: PAS-D positive, diastase-RESISTANT (diastase cannot digest poorly-branched long chains efficiently). Found in liver, muscle, and neurons (APBD). CONTRAST with limit dextrin (GSD III): short-chain, diastase-sensitive.",
            },
            {
                "term": "Fanconi-Bickel Syndrome — GSD XI (SLC2A2/GLUT2)",
                "definition": "Fanconi-Bickel syndrome (GSD XI) is caused by biallelic loss of SLC2A2 (GLUT2), the high-Km bidirectional glucose transporter in liver, kidney, intestine, and pancreatic beta-cells. The renal Fanconi syndrome component: GLUT2 mediates proximal tubular glucose reabsorption; without GLUT2, glucose is lost in urine at NORMAL blood glucose (normoglycaemic glucosuria — the diagnostic key). Also: aminoaciduria, phosphaturia (→ rickets), bicarbonuria (→ tubular acidosis). Galactose-free diet mandatory (galactose cannot exit liver without GLUT2). The dual glycaemic phenotype (fasting hypoglycemia + postprandial hyperglycemia) is pathognomonic.",
            },
            {
                "term": "GSD 0 — Glycogen Synthase Deficiency (GYS2)",
                "definition": "GSD type 0a is caused by GYS2 (liver glycogen synthase) biallelic loss. It is the conceptual opposite of all other GSDs — glycogen CANNOT BE SYNTHESISED rather than accumulated. Key features: NO hepatomegaly; fasting KETOTIC hypoglycemia (no glycogen reserve → ketogenesis as primary defense); postprandial HYPERGLYCEMIA (glucose cannot be stored as glycogen → transiently spills into blood after meals). HbA1c may be paradoxically normal or low. High protein diet + cornstarch is therapeutic. NEVER give insulin for postprandial hyperglycemia — will cause severe fasting hypoglycemia.",
            },
            {
                "term": "Hepatocellular Adenoma — GSD Ia/Ib Complication",
                "definition": "Hepatocellular adenomas (HCAs) develop in >90% of GSD Ia (G6PC) patients by age 30y and are also seen in GSD Ib (SLC37A4). Mechanism: chronic glycogen accumulation + lipid deposition + growth factor dysregulation (HNF1A and Wnt pathway alterations). Multiple adenomas common. Risk of malignant transformation to hepatocellular carcinoma (HCC): 10-15%, especially in late-treated or non-compliant patients. AFP elevation + new nodular changes = alarm for HCC. Surveillance: liver ultrasound + AFP every 6-12 months. HCA ≥3cm or rising AFP → surgical resection or liver transplant consideration. Hepatic gene therapy (DTX401) aims to reduce adenoma development.",
            },
            {
                "term": "Phosphorylase Kinase (PhK) — GSD IXa PHKA2 Activator",
                "definition": "Phosphorylase kinase (PhK) is a heterotetrameric kinase complex (alpha, beta, gamma, delta subunits) that activates glycogen phosphorylase (PYGL in liver, PYGM in muscle) by phosphorylation at Ser14. Without PhK, liver phosphorylase remains in the inactive b-form → glycogenolysis is impaired → mild glycogen accumulation → hepatomegaly and mild hypoglycemia. PHKA2 encodes the liver-specific alpha2 regulatory subunit (X-linked, Xp22.13). PHKB (AR, liver+heart) and PHKG2 (AR, liver, fibrosis risk in PHKG2) are additional PhK subunit genes causing GSD IX variants. All cause similar clinical phenotype but different inheritance and severity.",
            },
            {
                "term": "Normoglycaemic Glucosuria — Fanconi-Bickel Hallmark",
                "definition": "The presence of glucose in the urine at NORMAL blood glucose concentration. This is the cardinal diagnostic feature of renal Fanconi syndrome in GSD XI (SLC2A2/GLUT2 deficiency) and distinguishes it from diabetic glycosuria (glucosuria at HIGH blood glucose). Mechanism in GSD XI: GLUT2 normally mediates reabsorption of filtered glucose in the proximal tubule; GLUT2 loss → glucose cannot be reabsorbed → spills into urine at normoglycaemia. The Fanconi syndrome also includes aminoaciduria (all amino acids), phosphaturia (→ rickets), and bicarbonuria (→ tubular acidosis). Not to be confused with benign familial glucosuria (SLC5A2/SGLT2 deficiency) which has no Fanconi syndrome.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== GSD Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"Tx controlled: {ov['aggregate_clinical']['pct_tx_controlled']}%")
    print(f"Neurological: {ov['aggregate_clinical']['pct_neurological']}%")
    print(f"Deceased: {ov['aggregate_clinical']['pct_deceased']}%")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
