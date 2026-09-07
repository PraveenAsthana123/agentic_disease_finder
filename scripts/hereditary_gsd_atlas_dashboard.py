#!/usr/bin/env python3
"""Hereditary-GSD-Atlas — Complete 8-Gene Hereditary Glycogen Storage Disease Atlas
G6PC    (glucose-6-phosphatase catalytic subunit; 357 aa; 17q21.31; AR;
         GSD Ia — von Gierke disease; most common hepatic GSD (~80% of GSD I);
         fasting hypoglycaemia + hepatomegaly + nephromegaly + lactic acidosis;
         NO fructose/galactose/sucrose ABSOLUTE dietary restriction;
         uncooked corn starch — glucose polymer bypasses G6Pase;
         lactate elevated (substrate cannot enter gluconeogenesis);
         GFR and urine albumin annual monitoring — renal tubular dysfunction;
         seed SEED_BASE+0) .
SLC37A4 (glucose-6-phosphate translocase; 429 aa; 11q23.3; AR;
         GSD Ib — biochemically identical to Ia but with neutropenia PATHOGNOMONIC;
         neutropenia + recurrent infections + IBD-like intestinal disease;
         G-CSF (filgrastim/pegfilgrastim) for neutropenia — ANC >1.5 target;
         same dietary rules as GSD Ia; empagliflozin emerging for IBD-like disease;
         seed SEED_BASE+1) .
AGL     (amylo-1,6-glucosidase/4-alpha-glucanotransferase; 1532 aa; 1p21.2; AR;
         GSD III — Cori/Forbes debranching enzyme deficiency;
         hepatomegaly + myopathy + cardiomyopathy (in IIIa); liver disease improves with age;
         type IIIa (liver+muscle 80%) vs IIIb (liver only 15%);
         protein-enriched diet + corn starch; AVOID prolonged fasting;
         seed SEED_BASE+2) .
GBE1    (glycogen branching enzyme 1; 702 aa; 3p12.3; AR;
         GSD IV — Andersen disease; abnormal glycogen (amylopectin-like polyglucosan) accumulates;
         classic form: hepatic cirrhosis → liver failure; liver transplant curative for classic;
         adult polyglucosan body disease (APBD) — adult-onset progressive neurological disease;
         polyglucosan bodies in nerve biopsy PATHOGNOMONIC;
         seed SEED_BASE+3) .
PYGM    (muscle glycogen phosphorylase; 841 aa; 11q13.1; AR;
         GSD V — McArdle disease; exercise intolerance + myoglobinuria;
         ischemic forearm exercise test — lactate does NOT rise (PATHOGNOMONIC); ammonia RISES normally;
         second-wind phenomenon PATHOGNOMONIC;
         p.Arg50Ter most common variant (Europeans);
         NO statin (rhabdomyolysis risk); aerobic exercise training beneficial;
         seed SEED_BASE+4) .
PYGL    (liver glycogen phosphorylase; 848 aa; 14q22.1; AR;
         GSD VI — Hers disease; mild hepatic glycogenosis; usually benign;
         hepatomegaly + mild fasting hypoglycaemia + ketosis;
         liver phosphorylase in PYGL; phosphorylase kinase defect mimics in GSD IX;
         cornstarch at bedtime; spontaneous improvement with age;
         seed SEED_BASE+5) .
PHKA2   (phosphorylase kinase regulatory alpha-2 subunit; 1235 aa; Xp22.13; XLR;
         GSD IXa — most common GSD overall (1:100,000);
         X-linked liver phosphorylase kinase deficiency; males fully affected;
         hepatomegaly + growth retardation + elevated transaminases;
         usually benign; liver disease resolves after puberty in most males;
         cornstarch; avoid prolonged fasting;
         seed SEED_BASE+6) .
GYS2    (liver glycogen synthase 2; 703 aa; 12p12.1; AR;
         GSD 0a — liver glycogen synthase deficiency; UNIQUE: fasting ketotic hypoglycaemia
         + postprandial hyperglycaemia (OPPOSITE of typical GSD);
         ABSENCE of glycogen in liver — no glycogen accumulation;
         NBS MISSES — no abnormal acylcarnitines;
         cornstarch overnight; protein at bedtime prevents fasting hypoglycaemia;
         fasting ABSOLUTELY CI; postprandial glucose elevation does NOT require insulin;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 1806-1813)
"""

import random

SEED_BASE = 1806

GSD_GENES = [
    # -- G6PC -- GSD Ia — von Gierke ----------------------------------------
    {
        "gene": "G6PC",
        "protein": (
            "G6PC -- 17q21.31 AR -- Glucose-6-Phosphatase-Catalytic-357aa -- "
            "GSD-Ia-von-Gierke-Most-Common-Hepatic-GSD -- "
            "Fasting-Hypoglycaemia-Hepatomegaly-Nephromegaly-Lactic-Acidosis -- "
            "NO-Fructose-Galactose-Sucrose-ABSOLUTE-Dietary-Restriction -- "
            "Uncooked-Corn-Starch-KEYSTONE-Therapy -- "
            "Lactate-ELEVATED-Triglycerides-HIGH-Uric-Acid-HIGH -- "
            "GFR-Annual-Monitoring-Renal-Tubular-Dysfunction"
        ),
        "alias": (
            "G6PC (glucose-6-phosphatase catalytic subunit); OMIM gene 613742; "
            "GSD Ia (von Gierke disease) OMIM 232200. "
            "17q21.31; 357 aa; ~36 kDa; ER membrane protein; autosomal recessive. "
            "FUNCTION: G6Pase hydrolyses glucose-6-phosphate (G6P) to glucose + phosphate "
            "in the endoplasmic reticulum lumen, the final step of both gluconeogenesis and glycogenolysis. "
            "G6Pase requires G6P transporter (SLC37A4/G6PT) to translocate G6P into ER — "
            "both G6PC and SLC37A4 deficiency cause GSD I (Ia and Ib respectively). "
            "Without G6Pase: G6P accumulates → glycogen accumulates (glycogen synthesis uses G6P) → "
            "hepatic and renal glycogen overload; fasting cannot be managed. "
            "PATHOPHYSIOLOGY: "
            "Gluconeogenesis and glycogenolysis both BLOCKED at the final step; "
            "fasting → hypoglycaemia begins within 2-4 hours of last meal; "
            "G6P accumulates → glycolysis proceeds → pyruvate → lactate → lactic acidosis; "
            "G6P → ribose-5-phosphate → purine synthesis → hyperuricaemia; "
            "G6P → glycerol-3-phosphate → lipogenesis → hypertriglyceridaemia; "
            "liver enlarges (glycogen + lipid); kidneys enlarge (glycogen). "
            "CLINICAL PRESENTATION: "
            "Neonatal period: hypoglycaemia within hours of birth (if fasting); "
            "infancy: protuberant abdomen (hepatomegaly), doll-face (fat deposition), "
            "growth retardation, hypotonia, seizures from hypoglycaemia; "
            "biochemistry: fasting blood glucose <2.0 mmol/L, lactate 5-15 mmol/L, "
            "triglycerides 10-100 mmol/L, uric acid >600 µmol/L; "
            "blood glucose does NOT respond to glucagon stimulation — glucagon test PATHOGNOMONIC (flat response). "
            "RENAL COMPLICATIONS: "
            "Glomerular hyperfiltration (early) → proteinuria → FSGS → CKD (progressive); "
            "renal tubular Fanconi syndrome (minority); "
            "renal stones (uric acid + calcium); "
            "annual GFR + urine ACR monitoring mandatory from age 10; "
            "ACE inhibitors for proteinuria; "
            "hypercalciuria → bone disease (DEXA from age 15). "
            "DIETARY MANAGEMENT: "
            "Goal: maintain blood glucose 3.5-5.0 mmol/L at ALL times; "
            "UNCOOKED corn starch (UCCS): released slowly in GI tract → sustained glucose release over 4-6h; "
            "cooked starch is ineffective (gelatinisation increases glycaemic index, faster breakdown); "
            "dosing: 1.75-2.5 g/kg every 4h during waking; every 6h at night (adults); "
            "overnight: continuous glucose infusion (infant) or extended-release corn starch (Glycosade®) or UCCS; "
            "ABSOLUTE dietary restrictions: fructose (metabolised to F1P/F6P, bypasses liver but accumulates); "
            "galactose (metabolised to G1P → glycogen); sucrose (= glucose + fructose); "
            "ALLOWED: glucose, lactose-free dairy (if tolerated), complex starch. "
            "LONG-TERM COMPLICATIONS: "
            "Hepatic adenomas (70-80% by adulthood) — risk of haemorrhage and malignant transformation; "
            "HCC in non-adenoma parenchyma reported; annual liver imaging from age 10; "
            "polycystic ovary syndrome (PCOS) in females (high prevalence); "
            "short stature (GH axis intact but poor metabolic control); "
            "osteoporosis (hypercalciuria + poor nutrition). "
            "TREATMENT ADVANCES: "
            "Enzyme replacement therapy (ERT): not applicable (ER-membrane enzyme); "
            "Gene therapy: AAV8-hG6PC liver-directed therapy in Phase 3 trials (2026); "
            "SGLT2 inhibitors: empagliflozin lowers renal G6P load (emerging evidence); "
            "liver transplant: corrects hepatic G6Pase deficiency but NOT renal disease. "
        ),
        "locus": "17q21.31",
        "aa": 357,
        "kDa": 36,
        "omim_gene": 613742,
        "omim_disease": 232200,
        "inheritance": "AR",
        "gene_class": "ER membrane glucose-6-phosphatase — final gluconeogenesis/glycogenolysis step",
        "key_alerts": [
            "G6PC-NO-FRUCTOSE-GALACTOSE-SUCROSE-ABSOLUTE-CI: Fructose, galactose, and sucrose are absolutely contraindicated in GSD Ia — they are metabolised to hexose phosphates that accumulate and worsen disease; no fruit juice, no regular dairy with galactose, no table sugar; read food labels for hidden sources",
            "G6PC-GLUCAGON-TEST-FLAT-PATHOGNOMONIC: IV glucagon (30 µg/kg) during hypoglycaemia causes NO glucose rise (flat response) — pathognomonic for GSD I; other causes of fasting hypoglycaemia show a glucose rise; the flat glucagon response plus raised lactate is diagnostic",
            "G6PC-LACTATE-NOT-ACIDOSIS-MARKER-ONLY: Elevated lactate (5-15 mmol/L) is a metabolic marker of GSD Ia, not necessarily severe metabolic acidosis; the brain can use lactate as alternative fuel partially; maintain normal glucose → lactate falls",
            "G6PC-CORN-STARCH-UNCOOKED-ONLY: Only UNCOOKED corn starch works — cooked/modified starch is absorbed too rapidly; Glycosade® is waxy maize starch (extended-release) for overnight use; dosing errors are common (wrong starch, wrong timing, wrong dose)",
            "G6PC-HEPATIC-ADENOMA-SURVEILLANCE-MANDATORY: Hepatic adenomas develop in 70-80% of adults; annual ultrasound from age 10; adenomas >2 cm or showing malignant features on MRI need biopsy or resection; optimising metabolic control reduces adenoma growth",
        ],
        "etiologies": {
            "severe_neonatal_hypoglycaemia": 35,
            "infantile_hepatomegaly": 45,
            "incidental_discovery": 15,
            "family_screening": 5,
        },
        "stats": {
            "disease": "GSD Ia (von Gierke disease)",
            "incidence": "1:100,000 live births",
            "fraction_gsd_I": "~80% of GSD I",
            "fasting_glucose_mmol_L": "<2.0",
            "lactate_fasting_mmol_L": "5-15",
            "triglycerides_mmol_L": "10-100",
            "adenoma_adult_pct": 75,
            "renal_disease_adult_pct": 50,
            "glucagon_response_flat_pct": 100,
        },
        "dx_delay_distribution": {
            "diagnosed_neonatal": 20,
            "diagnosed_1_6_months": 55,
            "diagnosed_6_24_months": 20,
            "diagnosed_after_2y": 5,
        },
    },

    # -- SLC37A4 -- GSD Ib ---------------------------------------------------
    {
        "gene": "SLC37A4",
        "protein": (
            "SLC37A4 -- 11q23.3 AR -- Glucose-6-Phosphate-Translocase-429aa -- "
            "GSD-Ib-Biochemically-Identical-Ia-PLUS-Neutropenia-PATHOGNOMONIC -- "
            "Recurrent-Bacterial-Infections-IBD-Like-Intestinal-Disease -- "
            "G-CSF-Filgrastim-ANC-Target->1.5 -- "
            "Empagliflozin-Emerging-IBD-Like-Disease -- "
            "Same-Dietary-Rules-Ia-NO-Fructose-Galactose-Sucrose"
        ),
        "alias": (
            "SLC37A4 (solute carrier family 37 member 4, glucose-6-phosphate translocase); OMIM gene 602671; "
            "GSD Ib OMIM 232220. "
            "11q23.3; 429 aa; ~46 kDa; ER membrane 10-TM transporter; autosomal recessive. "
            "FUNCTION: SLC37A4 (G6PT) translocates glucose-6-phosphate from cytosol into the ER lumen "
            "where G6Pase (G6PC) hydrolyses it. Without G6PT, G6P cannot reach G6Pase → "
            "identical biochemical phenotype to GSD Ia (no hepatic glucose release). "
            "ADDITIONAL FUNCTION — NEUTROPHIL GLYCOGEN METABOLISM: "
            "G6PT is expressed in neutrophil endoplasmic reticulum; "
            "neutrophils depend on G6PT for intracellular glucose cycling; "
            "SLC37A4 LOF → neutrophil G6P accumulation → defective NADPH oxidase oxidative burst → "
            "impaired neutrophil function even when ANC is normal; "
            "accelerated neutrophil apoptosis → neutropenia (ANC <1.5 × 10⁹/L). "
            "CLINICAL PRESENTATION: "
            "GSD Ib = GSD Ia features (hepatomegaly, fasting hypoglycaemia, lactic acidosis, "
            "hypertriglyceridaemia, hyperuricaemia) PLUS neutropenia/neutrophil dysfunction. "
            "NEUTROPENIA CONSEQUENCES: "
            "Recurrent bacterial infections: oral aphthous ulcers (extremely common), "
            "skin infections, perianal abscesses, pneumonia, bacteraemia; "
            "IBD-like disease: colitis (Crohn's-like), perianal disease — "
            "due to neutrophil dysfunction + gut barrier compromise; "
            "ANC fluctuates — check in the morning before activity; "
            "absolute neutrophil count: target ANC >1.5 × 10⁹/L on G-CSF. "
            "G-CSF TREATMENT (MANDATORY FOR NEUTROPENIA): "
            "Filgrastim (short-acting) or pegfilgrastim (long-acting); "
            "start at 2-5 µg/kg/day filgrastim; "
            "dose titrate to ANC >1.5 target; "
            "CAUTION: risk of splenomegaly + thrombocytopenia with high-dose G-CSF; "
            "long-term G-CSF can cause myelodysplasia (monitor CBC annually in long-term use). "
            "EMPAGLIFLOZIN FOR IBD-LIKE DISEASE: "
            "SGLT2 inhibitor reduces intracellular G6P in neutrophils and gut epithelium; "
            "emerging evidence (2020-2026): reduces IBD-like disease severity in GSD Ib; "
            "also reduces hepatic G6P load → improves metabolic control; "
            "mechanism: SGLT2 inhibition reduces glucose reabsorption → lowers intracellular G6P; "
            "emerging use alongside G-CSF; may reduce G-CSF requirement. "
            "DIFFERENTIAL FROM GSD Ia: "
            "GSD Ib: neutropenia + neutrophil dysfunction are the ONLY distinguishing features; "
            "metabolic biochemistry identical; "
            "without neutropenia measurement, Ib is missed; "
            "ALWAYS check ANC in GSD I — SLC37A4 vs G6PC distinction changes management significantly. "
        ),
        "locus": "11q23.3",
        "aa": 429,
        "kDa": 46,
        "omim_gene": 602671,
        "omim_disease": 232220,
        "inheritance": "AR",
        "gene_class": "ER membrane glucose-6-phosphate translocase — G6P entry into ER",
        "key_alerts": [
            "SLC37A4-NEUTROPENIA-PATHOGNOMONIC-DDX-FROM-GSD-IA: Neutropenia distinguishes GSD Ib (SLC37A4) from GSD Ia (G6PC) — biochemically identical otherwise; ALWAYS check ANC in any GSD I patient; missing neutropenia = missing G-CSF therapy and risking life-threatening infections",
            "SLC37A4-G-CSF-MANDATORY-ANC-TARGET: G-CSF (filgrastim/pegfilgrastim) is mandatory for ANC <1.5 × 10⁹/L; target ANC >1.5 × 10⁹/L; oral aphthous ulcers and perianal disease are early signs of inadequate neutrophil function",
            "SLC37A4-EMPAGLIFLOZIN-IBD-EMERGING: Empagliflozin (SGLT2 inhibitor) reduces intracellular G6P in neutrophils and gut epithelium — improving IBD-like colitis in GSD Ib; check clinical trial eligibility; may reduce G-CSF requirement",
            "SLC37A4-NEUTROPHIL-DYSFUNCTION-ANC-NORMAL: Even when ANC is within normal range, neutrophil oxidative burst is impaired in GSD Ib — infection risk persists; do not rely solely on ANC to assess infection risk",
            "SLC37A4-IBD-NOT-TRUE-IBD: Colitis in GSD Ib looks like Crohn's on endoscopy but has different pathogenesis (neutrophil G6P dysfunction); conventional IBD immunosuppressants (steroids, biologics) have limited efficacy and may worsen infections; optimise metabolic control + G-CSF first",
        ],
        "etiologies": {
            "hepatomegaly_plus_neutropenia": 50,
            "recurrent_infections_first_presentation": 30,
            "metabolic_crisis_hypoglycaemia": 15,
            "ibd_first_presentation": 5,
        },
        "stats": {
            "disease": "GSD Ib",
            "incidence": "1:500,000 live births",
            "fraction_gsd_I": "~20% of GSD I",
            "anc_neutropenia_pct": 85,
            "ibd_like_disease_pct": 75,
            "oral_aphthous_pct": 70,
            "metabolic_identical_ia_pct": 100,
            "empagliflozin_benefit_pct": 65,
        },
        "dx_delay_distribution": {
            "diagnosed_within_6m": 35,
            "diagnosed_6_24m": 45,
            "diagnosed_2_5y": 15,
            "diagnosed_after_5y": 5,
        },
    },

    # -- AGL -- GSD III -------------------------------------------------------
    {
        "gene": "AGL",
        "protein": (
            "AGL -- 1p21.2 AR -- Amylo-1-6-Glucosidase-4-Alpha-Glucanotransferase-1532aa -- "
            "GSD-III-Cori-Forbes-Debranching-Enzyme-Deficiency -- "
            "IIIa-Liver-AND-Muscle-80pct-vs-IIIb-Liver-Only-15pct -- "
            "Hepatomegaly-Myopathy-Cardiomyopathy-IIIa -- "
            "High-Protein-Diet-KEY-Therapy-Not-Corn-Starch-Alone -- "
            "Liver-Disease-Improves-Puberty-UNIQUE"
        ),
        "alias": (
            "AGL (amylo-1,6-glucosidase/4-alpha-glucanotransferase); OMIM gene 610860; "
            "GSD IIIa/IIIb OMIM 232400. "
            "1p21.2; 1532 aa; ~175 kDa; bifunctional cytoplasmic enzyme; autosomal recessive. "
            "FUNCTION: AGL is the glycogen DEBRANCHING enzyme — bifunctional protein: "
            "4-alpha-glucanotransferase activity: transfers 3-glucosyl units from branch to main chain; "
            "amylo-1,6-glucosidase activity: releases single glucose from branch point. "
            "Without AGL, limit dextrin (short-branched glycogen) accumulates — cannot be fully mobilised. "
            "TYPES: "
            "GSD IIIa: AGL deficiency in liver AND muscle (~80%); "
            "GSD IIIb: AGL deficiency in liver ONLY (~15%); "
            "rare IIIc (glucosidase only) and IIId (transferase only) exist but are rare; "
            "type determined by enzyme assay in erythrocytes ± muscle biopsy; "
            "p.Arg1228Ter most common in non-Ashkenazi; p.Gln6Ter common in Ashkenazi Jews. "
            "CLINICAL PRESENTATION: "
            "LIVER DISEASE (all types): "
            "Hepatomegaly (significant); fasting hypoglycaemia (moderate, not as severe as GSD I); "
            "hepatic fibrosis → cirrhosis (long-term); HCC in cirrhotic liver (rare); "
            "elevated transaminases; "
            "UNIQUE FEATURE: liver disease IMPROVES after puberty in most patients — "
            "metabolic demand changes, puberty hormones shift fuel metabolism. "
            "MUSCLE DISEASE (IIIa): "
            "Progressive myopathy in adulthood (muscle disease often absent or mild in childhood); "
            "proximal weakness + distal wasting + atrophy (resembles limb-girdle MD clinically); "
            "CK chronically elevated (muscle glycogen accumulation); "
            "cardiomyopathy: HCM most common — annual Echo from age 5 in IIIa; "
            "myopathy progressive despite good liver control. "
            "DIETARY MANAGEMENT: "
            "DIFFERENT from GSD I — protein is a key therapy; "
            "high-protein diet (3-4 g/kg/day protein): provides gluconeogenic substrates (alanine, glutamine); "
            "AVOIDANCE of fasting (glucose release impaired); "
            "corn starch is LESS critical than in GSD I but useful overnight; "
            "fructose and galactose are ALLOWED (G6Pase pathway intact — only glycogenolysis blocked); "
            "restrict simple sugars to avoid postprandial glucose excursions. "
        ),
        "locus": "1p21.2",
        "aa": 1532,
        "kDa": 175,
        "omim_gene": 610860,
        "omim_disease": 232400,
        "inheritance": "AR",
        "gene_class": "bifunctional cytoplasmic glycogen debranching enzyme",
        "key_alerts": [
            "AGL-HIGH-PROTEIN-KEY-THERAPY-NOT-CORN-STARCH-ALONE: GSD III requires high protein diet (3-4 g/kg/day) as primary therapy — proteins provide gluconeogenic substrates (alanine, glutamine) that bypass the glycogenolysis block; corn starch is useful overnight but protein is the cornerstone",
            "AGL-LIVER-IMPROVES-PUBERTY-UNIQUE: Liver disease in GSD III characteristically improves after puberty — a unique feature compared to GSD I; hepatomegaly decreases, transaminases normalise; this is NOT seen in GSD I",
            "AGL-CARDIOMYOPATHY-MANDATORY-ECHO-IIIA: HCM occurs in GSD IIIa (liver + muscle type); annual echocardiography mandatory from age 5 in IIIa; progressive HCM can cause sudden cardiac death; beta-blockers for outflow obstruction",
            "AGL-FRUCTOSE-GALACTOSE-ALLOWED-UNLIKE-GSD-I: Fructose and galactose are permitted in GSD III (G6Pase pathway is intact, only debranching is impaired) — OPPOSITE to GSD I where they are absolutely prohibited; confirm type before dietary restriction",
            "AGL-MYOPATHY-ADULT-ONSET-PROGRESSIVE: Muscle disease in IIIa is often absent in childhood but emerges in adulthood as progressive proximal myopathy + distal wasting; CK chronically elevated; physio and aerobic exercise are beneficial",
        ],
        "etiologies": {
            "iiia_liver_and_muscle": 80,
            "iiib_liver_only": 15,
            "iiic_glucosidase_only": 3,
            "iiid_transferase_only": 2,
        },
        "stats": {
            "disease": "GSD III (Cori/Forbes disease)",
            "incidence": "1:100,000 live births",
            "liver_improvement_puberty_pct": 60,
            "cardiomyopathy_iiia_pct": 30,
            "myopathy_adult_pct": 70,
            "hepatic_fibrosis_pct": 40,
            "fasting_glucose_2_3_mmol_L": 65,
        },
        "dx_delay_distribution": {
            "diagnosed_within_1y": 50,
            "diagnosed_1_5y": 35,
            "diagnosed_after_5y": 15,
        },
    },

    # -- GBE1 -- GSD IV -------------------------------------------------------
    {
        "gene": "GBE1",
        "protein": (
            "GBE1 -- 3p12.3 AR -- Glycogen-Branching-Enzyme-1-702aa -- "
            "GSD-IV-Andersen-Disease-Polyglucosan-Accumulation -- "
            "Classic-Hepatic-Cirrhosis-Liver-Transplant-Curative -- "
            "Adult-Polyglucosan-Body-Disease-APBD-Progressive-Neurological -- "
            "Neuromuscular-Form-Perinatal-Lethal -- "
            "Polyglucosan-Bodies-Nerve-Biopsy-PATHOGNOMONIC"
        ),
        "alias": (
            "GBE1 (glycogen branching enzyme 1); OMIM gene 607839; "
            "GSD IV (Andersen disease) OMIM 232500; APBD OMIM 263570. "
            "3p12.3; 702 aa; ~80 kDa; cytoplasmic enzyme; autosomal recessive. "
            "FUNCTION: GBE1 adds branch points to growing glycogen chains — "
            "cleaves 1,4-glucosyl units from the end of chains and transfers them "
            "to form 1,6-alpha-glucosyl branch points. "
            "Without GBE1: linear (unbranched) glycogen accumulates = amylopectin-like polyglucosan; "
            "polyglucosan bodies are insoluble, resistant to degradation → cellular injury. "
            "FORMS OF PRESENTATION: "
            "1. CLASSIC PERINATAL/INFANTILE HEPATIC FORM (~50%): "
            "Neonatal/early infantile hepatomegaly; liver cirrhosis by 3-5 years; "
            "liver failure → liver transplant the only curative option; "
            "liver transplant corrects hepatic disease but NOT neuromuscular involvement; "
            "without transplant: fatal by age 5-8 years in classic form. "
            "2. NON-PROGRESSIVE HEPATIC FORM: "
            "Hepatomegaly without cirrhosis; hepatomegaly may resolve; "
            "good prognosis without transplant. "
            "3. NEUROMUSCULAR FORM: "
            "Perinatal lethal: fetal hydrops, perinatal death, cardiomyopathy; "
            "OR childhood myopathy + cardiomyopathy; severe; "
            "heart transplant has been performed but overall poor prognosis. "
            "4. ADULT POLYGLUCOSAN BODY DISEASE (APBD): "
            "Late-onset (5th-7th decade) neurological disease: "
            "progressive upper + lower motor neuron disease; "
            "neurogenic bladder (early feature — often initial complaint); "
            "sensory neuropathy; cerebellar ataxia; cognitive decline; "
            "polyglucosan bodies in sural nerve biopsy — PATHOGNOMONIC for APBD; "
            "p.Tyr329Ser most common variant in Ashkenazi Jews (Ashkenazi prevalence 1:2,500 carriers); "
            "nerve biopsy: PAS-positive diastase-resistant inclusions in axons. "
            "DIAGNOSIS: "
            "Enzyme assay (erythrocytes or leukocytes): absent/reduced GBE1 activity; "
            "liver biopsy: PAS-positive diastase-resistant deposits; "
            "gene panel: GBE1 sequencing + MLPA; "
            "nerve biopsy (APBD): polyglucosan bodies in sural nerve fibres. "
        ),
        "locus": "3p12.3",
        "aa": 702,
        "kDa": 80,
        "omim_gene": 607839,
        "omim_disease": 232500,
        "inheritance": "AR",
        "gene_class": "cytoplasmic glycogen branching enzyme — 1,4→1,6 glucosyl transferase",
        "key_alerts": [
            "GBE1-LIVER-TRANSPLANT-CURATIVE-CLASSIC: Liver transplant is the only cure for classic GSD IV with progressive cirrhosis; timing critical — before liver failure; transplant corrects hepatic polyglucosan accumulation; does NOT reverse neurological disease if already present",
            "GBE1-APBD-NEUROGENIC-BLADDER-EARLY: Neurogenic bladder is the earliest symptom of adult polyglucosan body disease (APBD) in the 5th-6th decade — check GBE1 in any adult with unexplained neurogenic bladder + motor/sensory neuropathy + UMN signs",
            "GBE1-POLYGLUCOSAN-NERVE-BIOPSY-PATHOGNOMONIC: PAS-positive, diastase-resistant polyglucosan bodies in sural nerve axons are pathognomonic for APBD; nerve biopsy required for diagnosis when gene panel is inconclusive (not all variants detected)",
            "GBE1-ASHKENAZI-APBD-FOUNDER: p.Tyr329Ser is the Ashkenazi Jewish founder variant for APBD (carrier frequency ~1:2,500); late-onset progressive neurological disease in Ashkenazi adults should prompt GBE1 testing",
            "GBE1-FOUR-CLINICAL-FORMS-SAME-GENE: GBE1 deficiency spans perinatal lethal → classic hepatic cirrhosis → non-progressive hepatic → APBD; same gene, different forms; severity correlates with residual enzyme activity; APBD has 5-10% residual activity",
        ],
        "etiologies": {
            "classic_hepatic_cirrhosis": 50,
            "non_progressive_hepatic": 20,
            "neuromuscular_perinatal": 10,
            "apbd_adult_onset": 20,
        },
        "stats": {
            "disease": "GSD IV (Andersen disease / APBD)",
            "incidence": "1:600,000 live births (classic); APBD underdiagnosed",
            "classic_liver_transplant_curative_pct": 85,
            "apbd_neurogenic_bladder_early_pct": 80,
            "apbd_ashkenazi_p_tyr329ser_pct": 60,
            "nerve_biopsy_pathognomonic_pct": 95,
        },
        "dx_delay_distribution": {
            "classic_diagnosed_before_2y": 60,
            "non_progressive_diagnosed_2_10y": 25,
            "apbd_diagnosed_50_70y": 15,
        },
    },

    # -- PYGM -- GSD V --------------------------------------------------------
    {
        "gene": "PYGM",
        "protein": (
            "PYGM -- 11q13.1 AR -- Muscle-Glycogen-Phosphorylase-841aa -- "
            "GSD-V-McArdle-Disease-Most-Common-Muscle-GSD -- "
            "Exercise-Intolerance-Myoglobinuria-Rhabdomyolysis -- "
            "Ischaemic-Forearm-Exercise-Test-Lactate-NO-RISE-PATHOGNOMONIC -- "
            "Second-Wind-Phenomenon-PATHOGNOMONIC -- "
            "NO-Statins-ABSOLUTE-CI -- "
            "Aerobic-Exercise-Training-BENEFICIAL"
        ),
        "alias": (
            "PYGM (muscle glycogen phosphorylase); OMIM gene 608455; "
            "GSD V (McArdle disease) OMIM 232600. "
            "11q13.1; 841 aa; ~97 kDa; muscle-specific isoform; autosomal recessive. "
            "FUNCTION: PYGM catalyses the phosphorolysis of glycogen at 1,4-glucosidic bonds "
            "to release glucose-1-phosphate from muscle glycogen — "
            "the first step of muscle glycogenolysis. "
            "Liver glycogen phosphorylase (PYGL) is separately encoded → liver unaffected in McArdle. "
            "Without PYGM: muscle cannot mobilise glycogen → exercise-induced energy deficit → "
            "glycogen accumulation in muscle fibres. "
            "PATHOPHYSIOLOGY: "
            "Glycogenolysis in muscle COMPLETELY BLOCKED; "
            "at rest: muscle uses fatty acids (PYGM not needed) → no symptoms; "
            "during exercise: oxidative phosphorylation demands ATP rapidly; "
            "early exercise: glycogen required as immediate fuel → block → ATP deficit → "
            "muscle cramps, pain, early fatigue; "
            "myoglobin released from ischaemic/necrotic fibres → myoglobinuria → risk of AKI; "
            "second-wind phenomenon: after 6-10 min of moderate exercise, "
            "sympathetic-mediated increase in fatty acid delivery + circulating glucose "
            "compensates for glycolytic failure → exercise tolerance resumes dramatically. "
            "ISCHAEMIC FOREARM EXERCISE TEST (IEFT): "
            "Modified non-ischaemic test (same interpretation, safer): "
            "exercise forearm muscles while occluded (ischaemic) or without blood pressure cuff; "
            "measure venous lactate + ammonia at 0, 1, 3, 5, 10 min post-exercise; "
            "NORMAL: lactate rises 2-3x baseline; ammonia rises 2-3x baseline; "
            "GSD V (PYGM): lactate does NOT rise (PATHOGNOMONIC — glycolysis blocked); "
            "ammonia DOES rise normally (purine nucleotide cycle intact); "
            "this lactate flat / ammonia rise pattern is PATHOGNOMONIC for muscle glycogenolysis defect; "
            "GLYCOLYTIC ENZYME DEFECTS (GSD VII PFK, GSD X PGAM, etc.): "
            "lactate also flat, but ammonia pattern may differ (test distinguishes glycolytic from phosphorylase). "
            "VARIANTS: "
            "p.Arg50Ter (c.148C>T): most common in Europeans (~60% of alleles); "
            "p.Gly205Ser: common in Iberian patients; "
            "p.Phe710del: Mediterranean; "
            "over 180 pathogenic variants described worldwide. "
            "MANAGEMENT: "
            "Sucrose pre-exercise: 75 g sucrose (= 37.5 g glucose) 5 min before exercise → "
            "exogenous glucose bypasses glycogenolysis block → significantly improves exercise tolerance; "
            "aerobic exercise training (progressive, moderate intensity) — increases peripheral fatty acid "
            "oxidation and cardiovascular fitness → reduces cramp frequency; "
            "protein intake adequate (ensure muscle repair substrates); "
            "AVOID: intense isometric/anaerobic exercise, statins (rhabdomyolysis risk), alcohol (myotoxic); "
            "RHABDOMYOLYSIS MANAGEMENT: "
            "IV fluids (3-5 L initial) to prevent myoglobin-mediated AKI; "
            "monitor CK, creatinine, urinary myoglobin; "
            "bicarbonate to alkalinise urine; "
            "avoid nephrotoxins; "
            "dialysis if AKI develops. "
        ),
        "locus": "11q13.1",
        "aa": 841,
        "kDa": 97,
        "omim_gene": 608455,
        "omim_disease": 232600,
        "inheritance": "AR",
        "gene_class": "muscle-specific glycogen phosphorylase — muscle glycogenolysis step 1",
        "key_alerts": [
            "PYGM-ISCHAEMIC-FOREARM-LACTATE-FLAT-PATHOGNOMONIC: Ischaemic (or non-ischaemic) forearm exercise test: lactate does NOT rise but ammonia RISES normally — this lactate-flat/ammonia-rise pattern is pathognomonic for PYGM (McArdle disease); distinguish from GSD VII (PFK) where both patterns differ",
            "PYGM-SECOND-WIND-PATHOGNOMONIC: Second-wind phenomenon is pathognomonic for McArdle disease — after 6-10 min of moderate exercise, exercise tolerance resumes dramatically as circulating glucose/fatty acids compensate; ask specifically about second wind in history",
            "PYGM-STATINS-ABSOLUTE-CI: Statins are absolutely contraindicated in McArdle disease — statins inhibit muscle HMG-CoA reductase → impair mitochondrial CoQ10 synthesis → dramatically increase rhabdomyolysis risk; avoid all statins",
            "PYGM-SUCROSE-PRE-EXERCISE: 75 g sucrose taken 5 minutes before exercise provides exogenous glucose that bypasses the muscle glycogenolysis block → significantly improves exercise tolerance; simple, safe, effective intervention",
            "PYGM-RHABDOMYOLYSIS-MYOGLOBINURIA-AKI-RISK: Any dark/cola-coloured urine after exercise = myoglobinuria → immediate IV fluids + hospital admission; delay in treatment causes AKI from myoglobin cast nephropathy; CK >10x upper limit triggers IV fluid protocol",
        ],
        "etiologies": {
            "exercise_intolerance_cramps": 65,
            "myoglobinuria_rhabdomyolysis": 25,
            "incidental_elevated_ck": 8,
            "family_screening": 2,
        },
        "stats": {
            "disease": "GSD V (McArdle disease)",
            "incidence": "1:100,000 live births",
            "most_common_muscle_gsd": True,
            "p_arg50ter_european_allele_pct": 60,
            "second_wind_pct": 90,
            "myoglobinuria_pct": 50,
            "aki_from_rhabdomyolysis_pct": 25,
            "pre_exercise_sucrose_benefit_pct": 85,
        },
        "dx_delay_distribution": {
            "diagnosed_childhood": 25,
            "diagnosed_young_adult": 45,
            "diagnosed_30_50y": 25,
            "diagnosed_after_50y": 5,
        },
    },

    # -- PYGL -- GSD VI -------------------------------------------------------
    {
        "gene": "PYGL",
        "protein": (
            "PYGL -- 14q22.1 AR -- Liver-Glycogen-Phosphorylase-848aa -- "
            "GSD-VI-Hers-Disease-Benign-Hepatic-Glycogenosis -- "
            "Hepatomegaly-Mild-Fasting-Hypoglycaemia-Ketosis -- "
            "Spontaneous-Improvement-Puberty -- "
            "Phosphorylase-Kinase-Defect-GSD-IX-Mimics-PYGL -- "
            "Cornstarch-Bedtime-Mainstay-Therapy"
        ),
        "alias": (
            "PYGL (liver glycogen phosphorylase); OMIM gene 613741; "
            "GSD VI (Hers disease) OMIM 232700. "
            "14q22.1; 848 aa; ~97 kDa; liver-specific phosphorylase isoform; autosomal recessive. "
            "FUNCTION: PYGL is the liver-specific isoform of glycogen phosphorylase — "
            "releases glucose-1-phosphate from liver glycogen at 1,4-glucosidic bonds; "
            "activated by glucagon (via phosphorylation by phosphorylase kinase) and AMP; "
            "the liver glycogen phosphorylation system: "
            "glucagon → PKA → phosphorylase kinase activation → PYGL phosphorylation → active PYGL. "
            "CLINICAL FEATURES: "
            "Generally BENIGN — one of the mildest GSDs; "
            "hepatomegaly (often noted incidentally); "
            "mild fasting hypoglycaemia (blood glucose rarely <2.5 mmol/L in most cases); "
            "fasting ketosis (ketones used as alternative fuel — UNLIKE GSD I where lactic acidosis dominates); "
            "elevated transaminases (glycogen infiltration); "
            "growth retardation (mild, often normalises); "
            "hyperlipidaemia (mild); "
            "SPONTANEOUS IMPROVEMENT after puberty — hepatomegaly and hypoglycaemia improve significantly. "
            "BIOCHEMICAL HALLMARKS: "
            "Fasting hypoglycaemia with KETOSIS (not lactic acidosis — unlike GSD I); "
            "lactate NORMAL or only mildly elevated; "
            "glucagon stimulation test: PARTIAL glucose rise (unlike GSD I flat response); "
            "this distinguishes GSD VI from GSD I clinically. "
            "DIFFERENTIAL DIAGNOSIS — GSD IX: "
            "GSD IX (phosphorylase kinase deficiency — PHKA2/PHKB/PHKG2/PHKA1) is biochemically "
            "and clinically IDENTICAL to GSD VI; "
            "PYGL (GSD VI) and PHKA2 (GSD IXa) are indistinguishable without gene panel; "
            "always send gene panel (not single gene) for hepatic glycogenosis with mild hypoglycaemia + ketosis; "
            "GSD IX is actually MORE COMMON than GSD VI. "
            "MANAGEMENT: "
            "Uncooked cornstarch at bedtime (0.5-1.5 g/kg) to prevent nocturnal hypoglycaemia; "
            "high-carbohydrate diet during day; "
            "avoid prolonged fasting; "
            "no fructose/galactose restrictions (G6Pase intact — UNLIKE GSD I); "
            "dietary management generally mild. "
        ),
        "locus": "14q22.1",
        "aa": 848,
        "kDa": 97,
        "omim_gene": 613741,
        "omim_disease": 232700,
        "inheritance": "AR",
        "gene_class": "liver-specific glycogen phosphorylase — liver glycogenolysis step 1",
        "key_alerts": [
            "PYGL-KETOSIS-NOT-LACTIC-ACIDOSIS-KEY-DDX-GSD-I: GSD VI fasting hypoglycaemia produces ketosis (NOT lactic acidosis) — a critical distinction from GSD I; measure both ketones and lactate in any fasting hypoglycaemia; ketosis with low/normal lactate → GSD VI/IX pathway",
            "PYGL-GSD-IX-CLINICALLY-IDENTICAL-GENE-PANEL-MANDATORY: GSD VI (PYGL) and GSD IXa (PHKA2) are biochemically and clinically indistinguishable — ALWAYS send comprehensive GSD gene panel (not single-gene), including PHKA2 (X-linked!) and PHKB/PHKG2",
            "PYGL-SPONTANEOUS-IMPROVEMENT-PUBERTY: GSD VI characteristically improves spontaneously after puberty — reassure families; hepatomegaly decreases and fasting tolerance improves; aggressive dietary restriction is not required long-term in most patients",
            "PYGL-CORNSTARCH-BEDTIME-PREVENTS-NOCTURNAL-HYPOGLYCAEMIA: Uncooked corn starch (0.5-1.5 g/kg) at bedtime is the mainstay of therapy — extends fasting tolerance overnight; required mainly in young children; can be weaned as the child grows",
            "PYGL-NO-FRUCTOSE-GALACTOSE-RESTRICTION-UNLIKE-GSD-I: G6Pase is INTACT in GSD VI — fructose and galactose are permitted (they are metabolised normally to glucose in GSD VI); NEVER apply GSD I dietary restrictions to GSD VI patients",
        ],
        "etiologies": {
            "hepatomegaly_elevated_transaminases": 65,
            "fasting_ketotic_hypoglycaemia": 25,
            "incidental_newborn_screen_elevated_c4oh": 5,
            "family_screening": 5,
        },
        "stats": {
            "disease": "GSD VI (Hers disease)",
            "incidence": "1:65,000–100,000 live births",
            "benign_overall_prognosis_pct": 85,
            "improvement_puberty_pct": 70,
            "ketosis_fasting_pct": 90,
            "lactic_acidosis_pct": 5,
            "hepatomegaly_pct": 98,
        },
        "dx_delay_distribution": {
            "diagnosed_within_2y": 55,
            "diagnosed_2_5y": 35,
            "diagnosed_after_5y": 10,
        },
    },

    # -- PHKA2 -- GSD IXa ----------------------------------------------------
    {
        "gene": "PHKA2",
        "protein": (
            "PHKA2 -- Xp22.13 XLR -- Phosphorylase-Kinase-Regulatory-Alpha2-1235aa -- "
            "GSD-IXa-Most-Common-GSD-Overall-1:100000 -- "
            "X-Linked-Males-Fully-Affected-Females-Variable -- "
            "Hepatomegaly-Growth-Retardation-Elevated-Transaminases -- "
            "Usually-Benign-Liver-Disease-Resolves-Puberty-Males -- "
            "Cornstarch-Avoid-Fasting"
        ),
        "alias": (
            "PHKA2 (phosphorylase kinase regulatory/alpha-2 subunit, liver isoform); OMIM gene 300798; "
            "GSD IXa OMIM 306000. "
            "Xp22.13; 1235 aa; ~138 kDa; regulatory subunit of liver phosphorylase kinase; X-linked recessive. "
            "FUNCTION: Phosphorylase kinase (PhK) is the enzyme that ACTIVATES glycogen phosphorylase (PYGL) "
            "by phosphorylation. PhK is a hexadecameric complex (alpha4-beta4-gamma4-delta4): "
            "PHKA2 is the liver-specific alpha-2 regulatory subunit; "
            "PHKB encodes the beta subunit (ubiquitous); "
            "PHKG2 encodes the liver-specific catalytic gamma-2 subunit; "
            "delta subunit = calmodulin (calcium sensor). "
            "PHKA2 LOF → PhK cannot activate PYGL → liver glycogenolysis impaired → "
            "glycogen accumulates in hepatocytes → hepatomegaly. "
            "X-LINKED INHERITANCE CONSEQUENCES: "
            "Males: hemizygous → fully affected; "
            "Females: heterozygous carriers → usually asymptomatic but can have mild hepatomegaly. "
            "PREVALENCE: "
            "GSD IXa is the MOST COMMON GSD overall (including GSD I–VI) — "
            "estimated 1:100,000 in males; underdiagnosed because of mild phenotype. "
            "CLINICAL FEATURES: "
            "Hepatomegaly (main finding, often presenting in early childhood); "
            "elevated transaminases (GPT/GOT 3-5x ULN); "
            "growth retardation (often mild, normalises with age); "
            "fasting hypoglycaemia (mild ketotic, rarely symptomatic); "
            "fasting ketosis (same ketosis pattern as GSD VI — ketones not lactic acidosis); "
            "hyperlipidaemia (mild, secondary). "
            "CLINICAL COURSE: "
            "USUALLY BENIGN in males — hepatomegaly and growth retardation resolve after puberty; "
            "adult males: generally asymptomatic; liver architecture normalises; "
            "EXCEPTION: rare cases progress to hepatic fibrosis/cirrhosis — especially PHKG2 variant. "
            "MANAGEMENT: "
            "Uncooked corn starch at bedtime (same as GSD VI); "
            "high-carbohydrate diet during day; "
            "avoid prolonged fasting; "
            "NO fructose/galactose restriction (G6Pase intact); "
            "reassure parents: majority have excellent long-term prognosis. "
            "DIFFERENTIAL: "
            "GSD IXa vs GSD VI (PYGL): clinically IDENTICAL — gene panel mandatory; "
            "GSD IXa (PHKA2) is X-linked — check family history for male relatives with hepatomegaly; "
            "GSD IXb (PHKB): autosomal recessive — affects liver and muscle; "
            "GSD IXc (PHKG2): autosomal recessive — more severe, liver fibrosis more common. "
        ),
        "locus": "Xp22.13",
        "aa": 1235,
        "kDa": 138,
        "omim_gene": 300798,
        "omim_disease": 306000,
        "inheritance": "XLR",
        "gene_class": "liver phosphorylase kinase regulatory alpha-2 subunit — PYGL activator",
        "key_alerts": [
            "PHKA2-MOST-COMMON-GSD-OVERALL: GSD IXa (PHKA2) is the most common GSD overall — estimated 1:100,000 in males; frequently underdiagnosed as 'non-specific hepatomegaly'; check PHKA2 in all boys with unexplained hepatomegaly + elevated transaminases",
            "PHKA2-X-LINKED-MALES-FULLY-AFFECTED: GSD IXa is X-linked recessive — males are fully affected; check maternal family history for unexplained hepatomegaly in male relatives; carrier females are usually asymptomatic",
            "PHKA2-BENIGN-LIVER-RESOLVES-PUBERTY: Prognosis is excellent in most males — hepatomegaly and growth retardation resolve after puberty; reassure families; over-treatment is common; minimal intervention needed in most cases",
            "PHKA2-GENE-PANEL-NOT-SINGLE-GENE: GSD IXa (PHKA2) is clinically identical to GSD IXb (PHKB), IXc (PHKG2), and GSD VI (PYGL) — always use comprehensive gene panel; PHKG2 variant of GSD IXc has worse prognosis (more fibrosis) — gene result changes prognostication",
            "PHKA2-NO-FRUCTOSE-RESTRICTION-UNLIKE-GSD-I: G6Pase is intact in GSD IXa — fructose and galactose are permitted; never apply GSD I dietary restrictions to GSD IX patients; this is a common error when GSD types are confused",
        ],
        "etiologies": {
            "hepatomegaly_incidental": 55,
            "elevated_transaminases_workup": 30,
            "growth_retardation_investigation": 10,
            "family_screening": 5,
        },
        "stats": {
            "disease": "GSD IXa (X-linked liver phosphorylase kinase deficiency)",
            "incidence": "1:100,000 males (most common GSD)",
            "male_fully_affected_pct": 100,
            "prognosis_benign_pct": 85,
            "hepatomegaly_pct": 98,
            "growth_retardation_pct": 60,
            "resolution_puberty_pct": 75,
        },
        "dx_delay_distribution": {
            "diagnosed_before_3y": 60,
            "diagnosed_3_8y": 30,
            "diagnosed_after_8y": 10,
        },
    },

    # -- GYS2 -- GSD 0a -------------------------------------------------------
    {
        "gene": "GYS2",
        "protein": (
            "GYS2 -- 12p12.1 AR -- Liver-Glycogen-Synthase-2-703aa -- "
            "GSD-0a-Liver-Glycogen-Synthase-Deficiency -- "
            "UNIQUE-Fasting-Ketotic-Hypoglycaemia-PLUS-Postprandial-Hyperglycaemia -- "
            "ABSENCE-of-Glycogen-in-Liver-NO-Glycogen-Accumulation -- "
            "NBS-MISSES-No-Acylcarnitine-Abnormality -- "
            "Cornstarch-Overnight-Protein-at-Bedtime -- "
            "Postprandial-Hyperglycaemia-NOT-Diabetes-No-Insulin"
        ),
        "alias": (
            "GYS2 (liver glycogen synthase 2); OMIM gene 138571; "
            "GSD 0a (liver glycogen synthase deficiency) OMIM 240600. "
            "12p12.1; 703 aa; ~79 kDa; liver-specific glycogen synthase; autosomal recessive. "
            "FUNCTION: GYS2 is the liver-specific isoform of glycogen synthase — "
            "it catalyses the extension of glycogen chains by transferring glucose from UDP-glucose "
            "to the non-reducing end of glycogen (alpha-1,4 linkage). "
            "Liver glycogen is the primary glucose reservoir for maintaining inter-meal blood glucose. "
            "GYS2 LOF → liver CANNOT synthesise glycogen → liver glycogen ABSENT (or near-absent); "
            "during fasting: no glycogen store to mobilise → hypoglycaemia begins quickly; "
            "during feeding: dietary glucose cannot be stored as liver glycogen → postprandial hyperglycaemia. "
            "THE UNIQUE METABOLIC PARADOX: "
            "GSD 0a is the ONLY GSD where the paradox is: "
            "FASTING → hypoglycaemia (no glycogen store) PLUS "
            "POSTPRANDIAL → hyperglycaemia (glucose overflow, cannot be stored); "
            "this is opposite to most GSDs (where glycogen accumulates); "
            "GSD 0a is a glycogen SYNTHESIS defect, not a glycogen UTILISATION defect. "
            "ABSENT GLYCOGEN — no glycogen accumulation: "
            "Liver biopsy: absent or minimal glycogen on PAS staining; "
            "no hepatomegaly (glycogen not accumulating → no liver enlargement); "
            "this distinguishes GSD 0a from ALL other hepatic GSDs (all others cause hepatomegaly). "
            "NEWBORN SCREENING — MISSED: "
            "GSD 0a is NOT detected by standard NBS tandem MS (no abnormal acylcarnitines or amino acids); "
            "first presentation: symptomatic fasting hypoglycaemia or incidental postprandial glucose elevation. "
            "POSTPRANDIAL HYPERGLYCAEMIA: "
            "Postprandial blood glucose can reach 10-15 mmol/L (180-270 mg/dL); "
            "this is NOT Type 1 or Type 2 diabetes — HbA1c may be elevated (glycaemic variability); "
            "insulin therapy is NOT indicated for postprandial hyperglycaemia in GSD 0a; "
            "glucose eventually cleared by glycolysis, HMP shunt, renal excretion; "
            "postprandial hyperglycaemia is benign in GSD 0a compared to fasting hypoglycaemia risk. "
            "MANAGEMENT: "
            "Primary goal: prevent fasting hypoglycaemia; "
            "uncooked corn starch at bedtime (1-2 g/kg) — sustained glucose release overnight; "
            "protein-rich snack at bedtime: protein provides gluconeogenic substrates; "
            "frequent daytime meals — no prolonged fasting; "
            "FASTING ABSOLUTELY CONTRAINDICATED — hypoglycaemia develops within 4-6 hours; "
            "AVOID insulin for postprandial hyperglycaemia — will cause severe hypoglycaemia. "
            "PROGNOSIS: "
            "Generally good with dietary management; "
            "cognitive outcomes usually normal if hypoglycaemia prevented from infancy; "
            "HbA1c monitoring but not a diabetes management target. "
        ),
        "locus": "12p12.1",
        "aa": 703,
        "kDa": 79,
        "omim_gene": 138571,
        "omim_disease": 240600,
        "inheritance": "AR",
        "gene_class": "liver-specific glycogen synthase — hepatic glycogen synthesis",
        "key_alerts": [
            "GYS2-UNIQUE-FASTING-HYPO-PLUS-POSTPRANDIAL-HYPER: GSD 0a is the ONLY GSD with fasting hypoglycaemia AND postprandial hyperglycaemia — the metabolic paradox of absent glycogen; no other GSD presents this combination; diagnose by thinking 'glucose storage failure, not storage excess'",
            "GYS2-NO-HEPATOMEGALY-UNLIKE-OTHER-GSDs: GSD 0a causes NO hepatomegaly — liver is NOT enlarged because glycogen does NOT accumulate (it cannot be synthesised); absence of hepatomegaly in a GSD should raise GSD 0a as the diagnosis",
            "GYS2-NBS-MISSES: Standard newborn screening (tandem MS) does NOT detect GSD 0a — no abnormal acylcarnitines or amino acids; first presentation is symptomatic fasting hypoglycaemia or incidental postprandial hyperglycaemia found on routine testing",
            "GYS2-INSULIN-ABSOLUTELY-CI-FOR-POSTPRANDIAL-HYPERGLYCAEMIA: Postprandial hyperglycaemia in GSD 0a is NOT diabetes — do NOT prescribe insulin; insulin will cause life-threatening hypoglycaemia; HbA1c may be elevated due to glycaemic variability but does not represent diabetic hyperglycaemia",
            "GYS2-FASTING-ABSOLUTELY-CI: Fasting is absolutely contraindicated in GSD 0a — without liver glycogen stores, hypoglycaemia develops within 4-6 hours; this patient group has the highest hypoglycaemia risk if dietary schedule is disrupted (illness, anaesthesia, surgery)",
        ],
        "etiologies": {
            "fasting_ketotic_hypoglycaemia": 55,
            "postprandial_hyperglycaemia_workup": 25,
            "recurrent_hypoglycaemia_infancy": 15,
            "family_screening": 5,
        },
        "stats": {
            "disease": "GSD 0a (liver glycogen synthase deficiency)",
            "incidence": "rare: <200 reported families worldwide",
            "fasting_hypoglycaemia_pct": 100,
            "postprandial_hyperglycaemia_pct": 90,
            "hepatomegaly_absent_pct": 95,
            "nbs_detected_pct": 0,
            "hba1c_elevated_pct": 60,
            "postprandial_glucose_mmol_L_peak": "10-15",
        },
        "dx_delay_distribution": {
            "diagnosed_before_5y": 40,
            "diagnosed_5_10y": 35,
            "diagnosed_after_10y": 25,
        },
    },
]


def _generate_patients():
    """Generate 40 synthetic patients per gene (8 × 40 = 320 total)."""
    for idx, gene_data in enumerate(GSD_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        patients = []
        gene = gene_data["gene"]

        for i in range(40):
            pid = f"{gene}-{seed}-P{i+1:02d}"
            age = rng.randint(0, 55)

            if gene == "G6PC":
                neonatal_hypo = rng.random() < 0.35
                hepatomegaly = rng.random() < 0.99
                nephromegaly = rng.random() < 0.75
                fasting_glucose = round(rng.uniform(1.2, 2.8), 1)
                lactate = round(rng.uniform(4.0, 15.0), 1)
                triglycerides = round(rng.uniform(8.0, 60.0), 0)
                uric_acid = rng.randint(400, 800)
                hepatic_adenoma = rng.random() < (0.70 if age > 20 else 0.10)
                renal_disease = rng.random() < (0.50 if age > 30 else 0.10)
                corn_starch_compliant = rng.random() < 0.80
                fructose_restriction_error = rng.random() < 0.12
                dx_delay_months = rng.randint(0, 24)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "neonatal_hypoglycaemia": neonatal_hypo,
                    "hepatomegaly": hepatomegaly,
                    "nephromegaly": nephromegaly,
                    "fasting_glucose_mmol_L": fasting_glucose,
                    "lactate_mmol_L": lactate,
                    "triglycerides_mmol_L": triglycerides,
                    "uric_acid_umol_L": uric_acid,
                    "hepatic_adenoma": hepatic_adenoma,
                    "renal_disease": renal_disease,
                    "corn_starch_compliant": corn_starch_compliant,
                    "fructose_restriction_error": fructose_restriction_error,
                    "dx_delay_months": dx_delay_months,
                    "outcome": "stable" if corn_starch_compliant else rng.choice(["stable", "mild_complication"]),
                })

            elif gene == "SLC37A4":
                neutropenia = rng.random() < 0.85
                anc = round(rng.uniform(0.3, 1.4) if neutropenia else rng.uniform(1.5, 4.0), 1)
                ibd_like = rng.random() < 0.75
                oral_aphthous = rng.random() < 0.70
                perianal_abscess = rng.random() < 0.30
                on_gcsf = neutropenia and rng.random() < 0.90
                empagliflozin_trial = ibd_like and rng.random() < 0.35
                fasting_glucose = round(rng.uniform(1.3, 2.9), 1)
                lactate = round(rng.uniform(4.0, 14.0), 1)
                dx_delay_months = rng.randint(0, 36)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "neutropenia": neutropenia,
                    "anc_10e9_L": anc,
                    "ibd_like_disease": ibd_like,
                    "oral_aphthous_ulcers": oral_aphthous,
                    "perianal_abscess": perianal_abscess,
                    "on_gcsf": on_gcsf,
                    "empagliflozin_trial": empagliflozin_trial,
                    "fasting_glucose_mmol_L": fasting_glucose,
                    "lactate_mmol_L": lactate,
                    "dx_delay_months": dx_delay_months,
                    "outcome": "stable" if on_gcsf else rng.choice(["stable", "infections"]),
                })

            elif gene == "AGL":
                subtype = rng.choice(["iiia", "iiia", "iiia", "iiia", "iiib", "iiib", "iiib", "iiic"])
                hepatomegaly = rng.random() < 0.97
                myopathy = rng.random() < (0.70 if subtype == "iiia" else 0.05)
                cardiomyopathy = rng.random() < (0.25 if subtype == "iiia" else 0.03)
                liver_improved_puberty = rng.random() < (0.60 if age > 14 else 0.05)
                ck_elevated = rng.random() < (0.80 if myopathy else 0.20)
                hepatic_fibrosis = rng.random() < 0.35
                fasting_glucose = round(rng.uniform(2.0, 3.5), 1)
                protein_diet_high = rng.random() < 0.75
                dx_delay_months = rng.randint(0, 48)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "subtype": subtype,
                    "hepatomegaly": hepatomegaly,
                    "myopathy": myopathy,
                    "cardiomyopathy_hcm": cardiomyopathy,
                    "liver_improved_after_puberty": liver_improved_puberty,
                    "ck_elevated": ck_elevated,
                    "hepatic_fibrosis": hepatic_fibrosis,
                    "fasting_glucose_mmol_L": fasting_glucose,
                    "high_protein_diet": protein_diet_high,
                    "dx_delay_months": dx_delay_months,
                    "outcome": rng.choice(["stable", "stable", "mild_disability"]),
                })

            elif gene == "GBE1":
                form = rng.choices(
                    ["classic_hepatic", "non_progressive", "neuromuscular", "apbd"],
                    weights=[50, 20, 10, 20]
                )[0]
                liver_cirrhosis = rng.random() < (0.85 if form == "classic_hepatic" else 0.05)
                liver_transplant = liver_cirrhosis and rng.random() < 0.70
                apbd_neurogenic_bladder = rng.random() < (0.80 if form == "apbd" else 0.02)
                ashkenazi = rng.random() < (0.60 if form == "apbd" else 0.10)
                nerve_biopsy_pathognomonic = rng.random() < (0.95 if form == "apbd" else 0.10)
                dx_delay_years = rng.randint(0, 50) if form == "apbd" else rng.randint(0, 5)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "clinical_form": form,
                    "liver_cirrhosis": liver_cirrhosis,
                    "liver_transplant": liver_transplant,
                    "apbd_neurogenic_bladder": apbd_neurogenic_bladder,
                    "ashkenazi_jewish": ashkenazi,
                    "nerve_biopsy_polyglucosan": nerve_biopsy_pathognomonic,
                    "dx_delay_years": dx_delay_years,
                    "outcome": "stable" if (liver_transplant or form == "non_progressive") else rng.choice(["stable", "moderate_disability", "severe"]),
                })

            elif gene == "PYGM":
                exercise_intolerance = rng.random() < 0.98
                myoglobinuria = rng.random() < 0.50
                aki_episode = myoglobinuria and rng.random() < 0.25
                second_wind = rng.random() < 0.90
                ieft_lactate_flat = rng.random() < 0.99
                statin_prescribed_error = rng.random() < 0.08
                pre_exercise_sucrose = rng.random() < 0.65
                aerobic_training = rng.random() < 0.55
                p_arg50ter_variant = rng.random() < 0.60
                ck_baseline_elevated = rng.random() < 0.95
                dx_delay_years = rng.randint(0, 30)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "exercise_intolerance": exercise_intolerance,
                    "myoglobinuria_episodes": rng.randint(0, 8) if myoglobinuria else 0,
                    "aki_rhabdomyolysis_episode": aki_episode,
                    "second_wind_phenomenon": second_wind,
                    "ieft_lactate_flat_pathognomonic": ieft_lactate_flat,
                    "statin_prescribed_error": statin_prescribed_error,
                    "pre_exercise_sucrose_used": pre_exercise_sucrose,
                    "aerobic_training": aerobic_training,
                    "p_arg50ter_homozygous": p_arg50ter_variant,
                    "ck_baseline_elevated": ck_baseline_elevated,
                    "dx_delay_years": dx_delay_years,
                    "outcome": "stable" if not aki_episode else rng.choice(["stable", "ckd_mild"]),
                })

            elif gene == "PYGL":
                hepatomegaly = rng.random() < 0.98
                ketotic_hypoglycaemia = rng.random() < 0.90
                fasting_glucose = round(rng.uniform(2.2, 3.4), 1)
                lactic_acidosis = rng.random() < 0.05
                liver_improved_puberty = rng.random() < (0.70 if age > 14 else 0.05)
                cornstarch_bedtime = rng.random() < 0.80
                fructose_restriction_error = rng.random() < 0.15
                dx_delay_months = rng.randint(0, 36)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "hepatomegaly": hepatomegaly,
                    "ketotic_hypoglycaemia": ketotic_hypoglycaemia,
                    "fasting_glucose_mmol_L": fasting_glucose,
                    "lactic_acidosis": lactic_acidosis,
                    "liver_improved_puberty": liver_improved_puberty,
                    "cornstarch_bedtime": cornstarch_bedtime,
                    "fructose_restriction_error": fructose_restriction_error,
                    "dx_delay_months": dx_delay_months,
                    "outcome": "stable" if cornstarch_bedtime else rng.choice(["stable", "hypoglycaemia_episode"]),
                })

            elif gene == "PHKA2":
                hepatomegaly = rng.random() < 0.98
                growth_retardation = rng.random() < 0.60
                transaminases_elevated = rng.random() < 0.95
                fasting_ketotic_hypo = rng.random() < 0.70
                liver_resolved_puberty = rng.random() < (0.75 if age > 15 else 0.05)
                carrier_female_mild = rng.random() < 0.20
                sex = "M" if rng.random() < 0.85 else "F"
                cornstarch_bedtime = rng.random() < 0.75
                dx_delay_months = rng.randint(0, 30)
                patients.append({
                    "patient_id": pid, "age_years": age, "sex": sex,
                    "hepatomegaly": hepatomegaly,
                    "growth_retardation": growth_retardation,
                    "transaminases_elevated": transaminases_elevated,
                    "fasting_ketotic_hypoglycaemia": fasting_ketotic_hypo,
                    "liver_resolved_puberty": liver_resolved_puberty,
                    "carrier_female_mild_disease": carrier_female_mild if sex == "F" else False,
                    "cornstarch_bedtime": cornstarch_bedtime,
                    "dx_delay_months": dx_delay_months,
                    "outcome": "stable",
                })

            elif gene == "GYS2":
                fasting_hypo = rng.random() < 1.0
                postprandial_hyper = rng.random() < 0.90
                hepatomegaly_absent = rng.random() < 0.95
                nbs_detected = False  # NBS NEVER detects GSD 0a
                hba1c_elevated = rng.random() < 0.60
                insulin_prescribed_error = rng.random() < 0.18
                cornstarch_bedtime = rng.random() < 0.80
                protein_bedtime = rng.random() < 0.70
                fasting_glucose_nadir = round(rng.uniform(1.5, 2.8), 1)
                postprandial_glucose_peak = round(rng.uniform(9.5, 15.0), 1)
                dx_delay_years = rng.randint(0, 25)
                patients.append({
                    "patient_id": pid, "age_years": age,
                    "fasting_ketotic_hypoglycaemia": fasting_hypo,
                    "postprandial_hyperglycaemia": postprandial_hyper,
                    "hepatomegaly_absent": hepatomegaly_absent,
                    "nbs_detected": nbs_detected,
                    "hba1c_elevated": hba1c_elevated,
                    "insulin_prescribed_error": insulin_prescribed_error,
                    "cornstarch_bedtime": cornstarch_bedtime,
                    "protein_bedtime": protein_bedtime,
                    "fasting_glucose_nadir_mmol_L": fasting_glucose_nadir,
                    "postprandial_glucose_peak_mmol_L": postprandial_glucose_peak,
                    "dx_delay_years": dx_delay_years,
                    "outcome": "stable" if not insulin_prescribed_error else rng.choice(["stable", "hypoglycaemia_episode"]),
                })

        gene_data["patients"] = patients


_generate_patients()


def _pct_true(lst, key):
    if not lst:
        return 0
    return round(100 * sum(1 for p in lst if p.get(key)) / len(lst), 1)


def overview():
    all_genes_info = [
        {
            "gene": g["gene"],
            "locus": g["locus"],
            "aa": g["aa"],
            "n_patients": len(g["patients"]),
            "inheritance": g["inheritance"],
        }
        for g in GSD_GENES
    ]
    total = sum(len(g["patients"]) for g in GSD_GENES)
    pts = {g["gene"]: g["patients"] for g in GSD_GENES}

    return {
        "atlas": "Hereditary GSD Atlas — Complete 8-Gene Glycogen Storage Disease Atlas",
        "subtitle": (
            "G6PC (17q21.31-AR-GSD-Ia-von-Gierke-NO-Fructose-Galactose-Corn-Starch-KEYSTONE-Flat-Glucagon-Response-PATHOGNOMONIC) . "
            "SLC37A4 (11q23.3-AR-GSD-Ib-Neutropenia-PATHOGNOMONIC-IBD-Like-G-CSF-Mandatory-Empagliflozin-Emerging) . "
            "AGL (1p21.2-AR-GSD-III-Cori-Forbes-IIIa-Liver-Muscle-IIIb-Liver-Only-High-Protein-KEY-Liver-Improves-Puberty) . "
            "GBE1 (3p12.3-AR-GSD-IV-Andersen-Polyglucosan-Liver-Transplant-Curative-APBD-Adult-Neurogenic-Bladder-Ashkenazi) . "
            "PYGM (11q13.1-AR-GSD-V-McArdle-Lactate-Flat-IEFT-PATHOGNOMONIC-Second-Wind-NO-Statins-Sucrose-Pre-Exercise) . "
            "PYGL (14q22.1-AR-GSD-VI-Hers-Benign-Ketosis-NOT-Lactic-Acidosis-Spontaneous-Improvement-Puberty) . "
            "PHKA2 (Xp22.13-XLR-GSD-IXa-Most-Common-GSD-Overall-Benign-Liver-Resolves-Puberty) . "
            "GYS2 (12p12.1-AR-GSD-0a-Fasting-Hypo-PLUS-Postprandial-Hyper-UNIQUE-NO-Hepatomegaly-NBS-Misses-Insulin-ABSOLUTE-CI) -- "
            "320 Patients (8x40, Seeds 1806-1813)"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "genes_covered": 8,
            "patients_per_gene": 40,
            "x_linked_genes": 1,
            "ar_genes": 7,
            # G6PC
            "g6pc_hepatomegaly_pct": _pct_true(pts["G6PC"], "hepatomegaly"),
            "g6pc_nephromegaly_pct": _pct_true(pts["G6PC"], "nephromegaly"),
            "g6pc_hepatic_adenoma_pct": _pct_true(pts["G6PC"], "hepatic_adenoma"),
            "g6pc_fructose_error_pct": _pct_true(pts["G6PC"], "fructose_restriction_error"),
            # SLC37A4
            "slc37a4_neutropenia_pct": _pct_true(pts["SLC37A4"], "neutropenia"),
            "slc37a4_ibd_like_pct": _pct_true(pts["SLC37A4"], "ibd_like_disease"),
            "slc37a4_oral_aphthous_pct": _pct_true(pts["SLC37A4"], "oral_aphthous_ulcers"),
            "slc37a4_gcsf_pct": _pct_true(pts["SLC37A4"], "on_gcsf"),
            # AGL
            "agl_hepatomegaly_pct": _pct_true(pts["AGL"], "hepatomegaly"),
            "agl_myopathy_pct": _pct_true(pts["AGL"], "myopathy"),
            "agl_cardiomyopathy_pct": _pct_true(pts["AGL"], "cardiomyopathy_hcm"),
            "agl_liver_improved_puberty_pct": _pct_true(pts["AGL"], "liver_improved_after_puberty"),
            # GBE1
            "gbe1_liver_cirrhosis_pct": _pct_true(pts["GBE1"], "liver_cirrhosis"),
            "gbe1_liver_transplant_pct": _pct_true(pts["GBE1"], "liver_transplant"),
            "gbe1_apbd_neurogenic_bladder_pct": _pct_true(pts["GBE1"], "apbd_neurogenic_bladder"),
            "gbe1_nerve_biopsy_pct": _pct_true(pts["GBE1"], "nerve_biopsy_polyglucosan"),
            # PYGM
            "pygm_second_wind_pct": _pct_true(pts["PYGM"], "second_wind_phenomenon"),
            "pygm_myoglobinuria_pct": sum(1 for p in pts["PYGM"] if p.get("myoglobinuria_episodes", 0) > 0) * 100 // 40,
            "pygm_statin_error_pct": _pct_true(pts["PYGM"], "statin_prescribed_error"),
            "pygm_pre_exercise_sucrose_pct": _pct_true(pts["PYGM"], "pre_exercise_sucrose_used"),
            # PYGL
            "pygl_hepatomegaly_pct": _pct_true(pts["PYGL"], "hepatomegaly"),
            "pygl_ketosis_pct": _pct_true(pts["PYGL"], "ketotic_hypoglycaemia"),
            "pygl_lactic_acidosis_pct": _pct_true(pts["PYGL"], "lactic_acidosis"),
            "pygl_liver_improved_puberty_pct": _pct_true(pts["PYGL"], "liver_improved_puberty"),
            # PHKA2
            "phka2_hepatomegaly_pct": _pct_true(pts["PHKA2"], "hepatomegaly"),
            "phka2_growth_retardation_pct": _pct_true(pts["PHKA2"], "growth_retardation"),
            "phka2_liver_resolved_puberty_pct": _pct_true(pts["PHKA2"], "liver_resolved_puberty"),
            # GYS2
            "gys2_fasting_hypo_pct": _pct_true(pts["GYS2"], "fasting_ketotic_hypoglycaemia"),
            "gys2_postprandial_hyper_pct": _pct_true(pts["GYS2"], "postprandial_hyperglycaemia"),
            "gys2_hepatomegaly_absent_pct": _pct_true(pts["GYS2"], "hepatomegaly_absent"),
            "gys2_nbs_detected_pct": _pct_true(pts["GYS2"], "nbs_detected"),
            "gys2_insulin_error_pct": _pct_true(pts["GYS2"], "insulin_prescribed_error"),
        },
        "genes": all_genes_info,
        "top_alerts": [
            "G6PC-NO-FRUCTOSE-GALACTOSE-ABSOLUTE-CI: Fructose, galactose, and sucrose are ABSOLUTELY PROHIBITED in GSD Ia (von Gierke) — they are metabolised to hexose-6-phosphates that accumulate behind the G6Pase block; no fruit juice, no regular dairy, no table sugar; dietary errors are the most common cause of acute metabolic crises",
            "SLC37A4-NEUTROPENIA-GSD-IB-PATHOGNOMONIC: Neutropenia distinguishes GSD Ib (SLC37A4) from GSD Ia (G6PC) — biochemically identical otherwise; ALWAYS check ANC in any GSD I patient; G-CSF is mandatory for ANC <1.5 × 10⁹/L; missing neutropenia = missing life-saving treatment",
            "PYGM-IEFT-LACTATE-FLAT-PATHOGNOMONIC: Ischaemic forearm exercise test in McArdle disease — lactate does NOT rise (glycolysis blocked) but ammonia RISES normally (purine nucleotide cycle intact); this lactate-flat/ammonia-rise pattern is pathognomonic for PYGM deficiency",
            "PYGM-NO-STATINS-ABSOLUTE-CI: Statins are absolutely contraindicated in McArdle disease — dramatically increase rhabdomyolysis risk; prescribing statins to McArdle patients for hypercholesterolaemia is a dangerous error; use ezetimibe or PCSK9 inhibitors instead",
            "GYS2-INSULIN-ABSOLUTELY-CI: Postprandial hyperglycaemia in GSD 0a is NOT diabetes — NEVER prescribe insulin; insulin will cause severe, life-threatening hypoglycaemia in a patient with no glycogen stores; elevated HbA1c in GSD 0a reflects glycaemic variability, NOT diabetic hyperglycaemia",
            "GYS2-NO-HEPATOMEGALY-UNIQUE: GSD 0a is the ONLY hepatic GSD with NO hepatomegaly — glycogen cannot be synthesised, so it cannot accumulate; absence of hepatomegaly in a fasting hypoglycaemia workup should raise GSD 0a",
            "GBE1-APBD-NEUROGENIC-BLADDER-FIRST-SIGN: Neurogenic bladder is the earliest symptom of adult polyglucosan body disease (APBD) in the 5th-6th decade; GBE1 gene panel should be considered in any adult with progressive neurogenic bladder + UMN/LMN signs + sensory neuropathy",
            "AGL-HIGH-PROTEIN-KEYSTONE-THERAPY: GSD III requires high protein diet (3-4 g/kg/day) as the primary therapy — proteins provide alanine/glutamine for gluconeogenesis, bypassing the glycogenolysis block; DIFFERENT from GSD I where corn starch is the keystone; FRUCTOSE AND GALACTOSE ALLOWED in GSD III",
            "PHKA2-MOST-COMMON-GSD-OVERALL: GSD IXa (PHKA2) is the most common GSD overall; mild hepatomegaly in boys + ketotic hypoglycaemia should prompt GSD IXa consideration; prognosis is excellent — hepatomegaly resolves after puberty in most males",
            "GSD-GENE-PANEL-MANDATORY: All hepatic GSDs with mild fasting ketotic hypoglycaemia (GSD VI, IXa, IXb, IXc, 0a) are clinically indistinguishable — comprehensive gene panel is mandatory; never rely on single-gene testing; GSD IXa is X-linked — inheritance pattern helps narrow differential",
        ],
    }


def breakdown():
    result = []
    for idx, g in enumerate(GSD_GENES):
        pts = g["patients"]
        ec = {}
        for p in pts:
            et = (
                p.get("subtype") or
                p.get("clinical_form") or
                ("fasting_hypo" if p.get("fasting_ketotic_hypoglycaemia") else None) or
                ("exercise_intolerance" if p.get("exercise_intolerance") else None) or
                "other"
            )
            ec[et] = ec.get(et, 0) + 1
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "etiology_counts": ec,
            "computed": {
                "n_patients": len(pts),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": pts[:10],
        })
    return result


def definitions():
    return {
        "concepts": {
            "Glycogen Storage Diseases — Biochemistry and Classification": (
                "Glycogen storage diseases (GSDs) are a group of inborn errors of metabolism "
                "caused by deficiencies of enzymes or transporters involved in glycogen synthesis, "
                "glycogen degradation, or glucose release from glycogen. "
                "GLYCOGEN STRUCTURE: "
                "Branched glucose polymer; alpha-1,4 glucosidic bonds form linear chains; "
                "alpha-1,6 glucosidic bonds form branch points every 8-14 glucose units; "
                "molecular weight: 10^7-10^9 Da; stored primarily in liver and muscle. "
                "KEY ENZYMES IN GLYCOGEN METABOLISM: "
                "SYNTHESIS: GYS2 (liver glycogen synthase) + GBE1 (branching enzyme); "
                "DEGRADATION: PYGL/PYGM (phosphorylase, releases G1P from 1,4 bonds) + "
                "AGL (debranching enzyme, releases glucose from 1,6 branch points); "
                "PHOSPHORYLATION: PHKA2/PHKB/PHKG2 (phosphorylase kinase activates PYGL/PYGM); "
                "FINAL STEP: G6PC (G6Pase, converts G6P → glucose; requires SLC37A4 for G6P entry into ER). "
                "HEPATIC vs MUSCLE GSDs: "
                "Hepatic GSDs (G6PC, SLC37A4, AGL, GBE1, PYGL, PHKA2, GYS2): "
                "hepatomegaly ± fasting hypoglycaemia ± lactic acidosis; "
                "Muscle GSDs (PYGM, GAA/Pompe, PFKM/GSD VII): "
                "exercise intolerance ± myoglobinuria ± myopathy. "
                "DIFFERENTIAL KEY: "
                "Fasting hypoglycaemia + lactic acidosis → GSD I (G6PC/SLC37A4) — G6Pase pathway; "
                "Fasting hypoglycaemia + ketosis (NO lactic acidosis) → GSD III/VI/IXa/0a — phosphorylase pathway; "
                "Exercise intolerance + myoglobinuria → McArdle (PYGM) — muscle phosphorylase; "
                "Absent liver glycogen + fasting hypo + postprandial hyper → GSD 0a (GYS2). "
                "TREATMENT PRINCIPLES: "
                "Prevent fasting hypoglycaemia via continuous glucose supply: "
                "uncooked cornstarch (UCCS) — releases glucose slowly over 4-6h; "
                "overnight glucose infusion (infants); "
                "nocturnal cornstarch (bedtime dose); "
                "high-protein diet in GSD III (provides gluconeogenic substrates); "
                "G-CSF for GSD Ib neutropenia; "
                "liver transplant for severe GSD I/IV; "
                "aerobic exercise training for McArdle (PYGM). "
            ),
            "Fasting Hypoglycaemia — GSD Differential and Approach": (
                "Fasting hypoglycaemia in a GSD context: blood glucose <2.6 mmol/L (<47 mg/dL) "
                "with symptoms or biochemical evidence of hypoglycaemia. "
                "GSD I (G6PC/SLC37A4): fasting hypoglycaemia VERY EARLY (within 2-4h), "
                "accompanied by lactic acidosis (lactate 5-15 mmol/L) — "
                "gluconeogenesis AND glycogenolysis both blocked (G6Pase is the final step of both); "
                "lactate, uric acid, triglycerides ALL elevated simultaneously; "
                "glucagon test: FLAT response (pathognomonic). "
                "GSD III (AGL): moderate fasting hypoglycaemia; "
                "ketosis (fats oxidised as alternative fuel); "
                "lactate can be mildly elevated; debranching enzyme deficient; "
                "fasting tolerance longer than GSD I. "
                "GSD VI (PYGL) / GSD IXa (PHKA2): mild fasting ketotic hypoglycaemia; "
                "NO lactic acidosis; ketones elevated; lactate NORMAL; "
                "phosphorylase pathway impaired; glucose release from glycogen reduced but "
                "gluconeogenesis is INTACT — partial compensation; "
                "glucagon test shows PARTIAL glucose rise (contrast: GSD I = flat, GSD VI = partial). "
                "GSD 0a (GYS2): fasting hypoglycaemia because NO glycogen was ever stored; "
                "ketosis; NO hepatomegaly (no glycogen accumulation); "
                "postprandial hyperglycaemia simultaneously present — the metabolic paradox. "
                "INITIAL INVESTIGATION OF FASTING HYPOGLYCAEMIA: "
                "Sample at time of hypoglycaemia: glucose, insulin, C-peptide, ketones, lactate, "
                "free fatty acids, ammonia, cortisol, GH; "
                "urine: ketones, glucose, organic acids; "
                "GSD panel: plasma amino acids (for MSUD/UCD overlap), acylcarnitines; "
                "gene panel: G6PC, SLC37A4, AGL, GBE1, PYGL, PHKA2, GYS2 + others. "
            ),
            "Uncooked Corn Starch (UCCS) — Mechanism and Clinical Use": (
                "Uncooked corn starch is the cornerstone of management for hepatic GSDs requiring "
                "prevention of fasting hypoglycaemia. "
                "MECHANISM: "
                "Uncooked (raw, ungelatinised) corn starch is a resistant starch — "
                "granules intact, digest slowly by pancreatic amylase over 4-6 hours; "
                "provides sustained release of glucose into the bloodstream; "
                "COOKED starch (gelatinised) = rapidly digestible, ineffective for sustained release; "
                "Glycosade® (waxy maize starch, heat-stable) extends release to 6-8h — "
                "used overnight, useful in children >5 years (pancreatic amylase mature). "
                "DOSING: "
                "Infants (<12 months): continuous glucose infusion preferred (immature pancreatic amylase); "
                "children >1-2 years: UCCS 1.75-2.5 g/kg every 4h during day; "
                "overnight: UCCS 2.0-2.5 g/kg every 6h OR Glycosade® 1.5-2.0 g/kg; "
                "adults: 40-75 g UCCS every 4-6h; "
                "monitor blood glucose 2-4h after dose to verify efficacy. "
                "DIETARY RESTRICTIONS BY GSD TYPE: "
                "GSD Ia (G6PC) and GSD Ib (SLC37A4): "
                "PROHIBIT: fructose, galactose, sucrose (metabolised to hexose phosphates that accumulate); "
                "ALLOW: glucose, complex starches; "
                "GSD III (AGL), GSD VI (PYGL), GSD IXa (PHKA2), GSD 0a (GYS2): "
                "ALLOW: fructose, galactose (G6Pase intact — these sugars can exit to glucose normally); "
                "RESTRICT: prolonged fasting; "
                "Applying GSD I restrictions to GSD III/VI/IX patients is a common clinical error. "
                "GLUCOSE MONITORING IN GSD: "
                "Continuous glucose monitor (CGM) increasingly used — "
                "detects nocturnal hypoglycaemia without finger-prick; "
                "target: glucose 4-7 mmol/L at all times; "
                "lactate can complement glucose monitoring in GSD I. "
            ),
            "Ischaemic Forearm Exercise Test — McArdle Disease": (
                "The ischaemic (or non-ischaemic) forearm exercise test (IEFT) is the key "
                "diagnostic test for muscle glycogenolysis defects, particularly McArdle disease (PYGM). "
                "PROTOCOL (non-ischaemic, safer modification): "
                "Patient at rest, IV cannula in antecubital vein; "
                "baseline blood: lactate + ammonia; "
                "patient squeezes hand dynamometer vigorously for 60-90 seconds (no blood pressure cuff); "
                "samples at 1, 3, 5, 10 minutes post-exercise: lactate + ammonia. "
                "NORMAL RESPONSE: "
                "Lactate: rises 2-3× baseline (glycolysis active → pyruvate → lactate); "
                "Ammonia: rises 2-3× baseline (purine nucleotide cycle active during high-intensity exercise). "
                "McARDLE (PYGM) RESPONSE: "
                "Lactate: NO RISE (flat) — PATHOGNOMONIC; glycolysis cannot proceed (muscle glycogenolysis blocked); "
                "Ammonia: RISES normally — purine nucleotide cycle intact (AMP → IMP → NH3); "
                "the lactate-flat / ammonia-rise pattern is the diagnostic hallmark. "
                "GLYCOLYTIC ENZYME DEFECTS (GSD VII — PFKM; GSD X — PGAM; GSD XII — ENO3): "
                "Lactate: flat (glycolysis blocked distal to the glycogenolysis step); "
                "Ammonia: also rises; distinguish by specific enzyme assay + gene panel. "
                "GSD II (Pompe, GAA): "
                "Normal IEFT (acid maltase is lysosomal — does NOT participate in exercise glycolysis); "
                "Pompe presents with myopathy NOT exercise-induced cramps. "
                "INTERPRETATION CAUTION: "
                "Effort-dependent test — must be maximal effort; sub-maximal effort = false flat lactate; "
                "always check ammonia: if ammonia does NOT rise → poor effort (false result); "
                "ischaemic version (blood pressure cuff occlusion) occasionally causes venous thrombosis — "
                "non-ischaemic version preferred. "
            ),
        }
    }
