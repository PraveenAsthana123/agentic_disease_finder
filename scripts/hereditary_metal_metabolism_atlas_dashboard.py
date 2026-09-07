#!/usr/bin/env python3
"""Hereditary-Metal-Metabolism-Atlas — Complete 8-Gene Hereditary Metal/Trace Element Metabolism Atlas
ATP7B   (Wilson-Disease; 1465 aa; 13q14.3; AR;
         P-type Cu-ATPase — copper transport into bile and ceruloplasmin;
         Kayser-Fleischer rings PATHOGNOMONIC on slit-lamp;
         D-penicillamine / trientine / zinc acetate;
         p.His1069Gln 40% of European alleles;
         seed SEED_BASE+0) .
ATP7A   (Menkes Disease; 1500 aa; Xq21.1; XLR;
         P-type Cu-ATPase — Cu export from enterocytes/placenta;
         Pili torti + kinky/steely hair PATHOGNOMONIC;
         Cu-histidine SQ within 4-6 WEEKS of birth;
         Occipital horn syndrome milder allelic variant;
         seed SEED_BASE+1) .
HFE     (Hereditary Hemochromatosis Type 1; 348 aa; 6p22.2; AR;
         HLA-linked MHC class I; p.Cys282Tyr (C282Y) 80-85% of HH alleles;
         2nd-3rd MCP arthropathy PATHOGNOMONIC early;
         phlebotomy 500 mL weekly until ferritin <50 µg/L CURATIVE;
         seed SEED_BASE+2) .
HAMP    (Juvenile Hemochromatosis Type 2B / Hepcidin; 84 aa; 19q13.12; AR;
         hepcidin — master iron regulatory hormone;
         CARDIAC involvement EARLIEST and most life-threatening;
         juvenile onset 2nd-3rd decade;
         seed SEED_BASE+3) .
SLC40A1 (Ferroportin Disease / Hemochromatosis Type 4; 570 aa; 2q32.2; AD;
         AUTOSOMAL DOMINANT — sole cellular iron exporter;
         Type 4A (LoF) macrophage iron, LOW-NORMAL TS;
         Type 4B (hepcidin resistance) HIGH TS — treatment differs;
         seed SEED_BASE+4) .
CP      (Aceruloplasminemia; 1058 aa; 3q25.1; AR;
         ceruloplasmin multicopper ferroxidase;
         TRIAD: DM + retinal degeneration + neurodegeneration PATHOGNOMONIC;
         ceruloplasmin ABSENT; serum iron LOW despite iron overload paradox;
         seed SEED_BASE+5) .
SLC30A10 (Hypermanganesemia with Dystonia 1; 485 aa; 1q41; AR;
          manganese exporter — liver bile export + intestine;
          POLYCYTHEMIA PATHOGNOMONIC in Mn overload;
          MRI T1 HIGH signal basal ganglia PATHOGNOMONIC;
          EDTA chelation;
          seed SEED_BASE+6) .
TMPRSS6  (Iron-Refractory Iron Deficiency Anemia / IRIDA; 811 aa; 22q12.3; AR;
           matriptase-2 serine protease — cleaves HJV, reduces hepcidin;
           ORAL IRON COMPLETELY INEFFECTIVE PATHOGNOMONIC;
           hepcidin constitutively high;
           IV iron needed but partial response;
           seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1838–1845)
"""

import random

SEED_BASE = 1838

METAL_GENES = [
    # -- ATP7B -- Wilson Disease -----------------------------------------------
    {
        "gene": "ATP7B",
        "protein": (
            "ATP7B -- 13q14.3 AR -- P-type-Cu-ATPase-1465aa -- "
            "Wilson-Disease-Copper-Transport-Bile-Ceruloplasmin -- "
            "Kayser-Fleischer-Rings-Slit-Lamp-PATHOGNOMONIC -- "
            "Serum-Ceruloplasmin-VERY-LOW-<0.1g/L-95pct -- "
            "D-Penicillamine-Trientine-Zinc-Acetate-Liver-Tx-Hepatic-Only -- "
            "p.His1069Gln-40pct-European-Alleles"
        ),
        "alias": (
            "ATP7B (ATPase copper transporting beta); OMIM gene 606882; "
            "Wilson disease OMIM 277900. "
            "13q14.3; 1465 aa; ~165 kDa; trans-Golgi network and hepatocyte canalicular membrane; "
            "P1B-type ATPase (Cu-transporting); autosomal recessive. "
            "FUNCTION: ATP7B pumps copper into the trans-Golgi network for incorporation into "
            "ceruloplasmin (the major serum copper-binding protein), and exports excess copper "
            "into bile for faecal excretion. It is the hepatocyte's primary mechanism for "
            "preventing copper accumulation. "
            "In ATP7B deficiency: copper accumulates in the LIVER first, then spills into "
            "plasma (as non-ceruloplasmin-bound copper = toxic free copper) → deposits in "
            "BRAIN (basal ganglia, thalamus, brainstem), KIDNEY (Fanconi syndrome), "
            "CORNEA (Kayser-Fleischer rings), and RED BLOOD CELLS (Coombs-negative haemolytic anaemia). "
            "CERULOPLASMIN: "
            "Serum ceruloplasmin VERY LOW (<0.1 g/L) in ~95% of Wilson patients; "
            "ceruloplasmin is low because ATP7B failure prevents copper incorporation "
            "into apoceruloplasmin → apoceruloplasmin is degraded → total ceruloplasmin falls; "
            "EXCEPTION: in acute liver failure or inflammation, ceruloplasmin can be "
            "falsely normal (it is an acute phase reactant). "
            "KAYSER-FLEISCHER RINGS: "
            "Copper deposits in Descemet membrane of peripheral cornea = KF rings; "
            "golden-brown to greenish ring visible on slit-lamp examination; "
            "PRESENT in ~95% of neuropsychiatric Wilson; "
            "ABSENT in up to 50% of hepatic-only Wilson (liver disease before neurological involvement); "
            "slit-lamp examination is MANDATORY — KF rings invisible to the naked eye; "
            "KF rings resolve with treatment. "
            "PATHOPHYSIOLOGY AND PHENOTYPE: "
            "1. HEPATIC DISEASE (often first): "
            "ranges from asymptomatic elevation of transaminases → "
            "chronic active hepatitis → cirrhosis → "
            "acute liver failure (ALF); "
            "acute Wilson ALF: haemolysis (Coombs-negative, intravascular) + "
            "rapid jaundice + coagulopathy + low ALP (ALF with LOW ALP = Wilson red flag); "
            "copper release from necrotic hepatocytes → rapid systemic copper load. "
            "2. NEUROPSYCHIATRIC DISEASE: "
            "dysarthria (slurred speech), dystonia (limb or bulbar), "
            "tremor (wing-beating or resting), drooling, personality change; "
            "psychiatric: psychosis, depression, anxiety, personality change "
            "(often first presentation in young adults); "
            "MRI brain: T2/FLAIR high signal in lenticular nuclei (putamen, globus pallidus), "
            "thalamus, midbrain; "
            "face-of-the-giant panda sign (midbrain MRI). "
            "3. RENAL: Fanconi syndrome (proximal tubular dysfunction): "
            "glycosuria + aminoaciduria + phosphaturia + uricosuria + nephrocalcinosis. "
            "DIAGNOSIS: "
            "24h urine copper: >100 µg/day baseline diagnostic; "
            ">1600 µg/day (>25 µmol/day) after penicillamine challenge = confirmatory; "
            "serum ceruloplasmin <0.1 g/L; "
            "slit-lamp: KF rings; "
            "liver biopsy: hepatic copper >250 µg/g dry weight (definitive); "
            "ATP7B gene sequencing (>500 mutations known); "
            "Leipzig score ≥4 = diagnosis confirmed. "
            "TREATMENT: "
            "D-PENICILLAMINE (copper chelator): first-line in many centres; "
            "mobilises hepatic copper → urinary excretion; "
            "TERATOGENIC — CI in pregnancy; "
            "side effects: nephrotic syndrome, drug lupus, myasthenia, thrombocytopenia; "
            "TRIENTINE (triethylenetetramine): "
            "copper chelator with fewer side effects than penicillamine; "
            "PREFERRED in pregnancy and neuropsychiatric Wilson; "
            "standard first-line in North America and increasingly in Europe; "
            "ZINC ACETATE: "
            "competes with copper for intestinal absorption (induces metallothionein in enterocytes); "
            "maintenance therapy; slower onset; does NOT chelate; "
            "safe in pregnancy; "
            "Liver transplant: CURATIVE for hepatic Wilson disease "
            "(including acute liver failure); "
            "no established benefit for neuropsychiatric Wilson in isolation; "
            "transplant corrects the metabolic defect (donor liver has normal ATP7B). "
            "COPPER-RICH FOODS RESTRICTION during treatment initiation: "
            "shellfish (especially oysters), liver/offal, chocolate, nuts, mushrooms; "
            "restriction less critical once chelation established. "
            "KEY CLINICAL FACTS: "
            "Coombs-negative haemolytic anaemia + liver disease in young person = Wilson until proven otherwise; "
            "low ALP in setting of acute liver failure = PATHOGNOMONIC Wilson; "
            "neuropsychiatric Wilson: do not miss — often diagnosed as primary psychiatric disorder; "
            "p.His1069Gln (c.3207C>A): most common European variant (~40% of European alleles); "
            "prognosis with treatment started before cirrhosis: excellent."
        ),
        "age_of_onset": "5-35 years (hepatic typically earlier; neuropsychiatric teens-early adulthood)",
        "inheritance": "AR",
        "locus": "13q14.3",
        "protein_size": "1465 aa",
        "key_biomarker": "Serum ceruloplasmin <0.1 g/L; 24h urine Cu >100 µg/day; slit-lamp KF rings",
        "pathognomonic": "Kayser-Fleischer rings (slit-lamp) + low ceruloplasmin + elevated urine copper in young adult",
        "treatment": "D-penicillamine (CI pregnancy) OR trientine OR zinc acetate (maintenance); liver transplant curative for hepatic Wilson",
        "critical_flags": [
            "KF-RINGS-SLIT-LAMP-MANDATORY — absent in hepatic-only Wilson (up to 50%); never diagnose without slit-lamp",
            "CERULOPLASMIN-VERY-LOW-95pct — <0.1 g/L; can be falsely normal in acute phase/inflammation",
            "D-PENICILLAMINE-TERATOGENIC-CI-PREGNANCY — switch to trientine if pregnant",
            "TRIENTINE-PREFERRED-PREGNANCY-NEUROPSYCHIATRIC — fewer side effects; standard North America",
            "COPPER-RICH-FOODS-CI-INITIATION — shellfish, liver, chocolate, nuts restricted at start",
            "LIVER-TRANSPLANT-HEPATIC-ONLY — curative for liver disease; no proven benefit for neuropsychiatric",
            "p.His1069Gln-40pct-EUROPEAN-ALLELES — most common European mutation; sequence ATP7B",
            "ACUTE-LIVER-FAILURE-LOW-ALP-COOMBS-NEG-HAEMOLYSIS-PATHOGNOMONIC — Wilson ALF red flag",
        ],
    },
    # -- ATP7A -- Menkes Disease ------------------------------------------------
    {
        "gene": "ATP7A",
        "protein": (
            "ATP7A -- Xq21.1 XLR -- P-type-Cu-ATPase-1500aa -- "
            "Menkes-Disease-Cu-Export-Enterocytes-Placenta -- "
            "Pili-Torti-Kinky-Steely-Hair-PATHOGNOMONIC -- "
            "Cu-Histidine-SQ-Within-4-6-Weeks-Birth -- "
            "LOW-Serum-Cu-LOW-Ceruloplasmin-Both-Low -- "
            "Occipital-Horn-Syndrome-Milder-Allelic-Variant"
        ),
        "alias": (
            "ATP7A (ATPase copper transporting alpha); OMIM gene 300011; "
            "Menkes disease OMIM 309400; "
            "Occipital horn syndrome OMIM 304150. "
            "Xq21.1; 1500 aa; ~180 kDa; trans-Golgi network and plasma membrane; "
            "P1B-type Cu-ATPase; X-linked recessive. "
            "FUNCTION: ATP7A exports copper from enterocytes into the portal circulation "
            "after dietary absorption, and from the placenta to the fetus. "
            "In the brain, ATP7A supplies copper to cuproenzymes including: "
            "cytochrome c oxidase (mitochondrial respiration), "
            "dopamine-beta-hydroxylase (norepinephrine synthesis), "
            "lysyl oxidase (collagen/elastin crosslinking), "
            "peptidylglycine amidating monooxygenase, "
            "superoxide dismutase. "
            "In ATP7A deficiency: "
            "Copper is absorbed from the gut but CANNOT be exported from enterocytes "
            "→ copper trapped in gut (and in other non-hepatic cells); "
            "SYSTEMIC COPPER DEFICIENCY despite dietary copper ingestion; "
            "liver is copper-deficient; serum copper low; ceruloplasmin low. "
            "CONTRAST WITH WILSON (ATP7B): "
            "Wilson = copper ACCUMULATION (hepatocyte cannot export); "
            "Menkes = copper DEFICIENCY in blood/organs (enterocyte cannot export to blood); "
            "both: serum ceruloplasmin low; "
            "but Wilson: serum copper eventually elevated (non-ceruloplasmin bound); "
            "Menkes: both serum copper AND ceruloplasmin low. "
            "PATHOGNOMONIC — PILI TORTI: "
            "Twisted, depigmented (grey-white-silvery), brittle hair = PILI TORTI; "
            "also called kinky hair, steely hair, or monilethrix-like; "
            "caused by deficiency of lysyl oxidase (copper-dependent) "
            "→ impaired keratin crosslinking in hair follicle; "
            "hallmark clinical sign; visible in first weeks-months of life. "
            "CONNECTIVE TISSUE FAILURE (lysyl oxidase copper-dependent): "
            "aortic and arterial tortuosity (tortuous cerebral arteries on MRI angiography); "
            "subdural haematoma (bridging vein rupture due to arterial fragility — "
            "MIMICS NON-ACCIDENTAL INJURY / NAI); "
            "bladder diverticula; "
            "skin laxity and hyperextensibility; "
            "joint hypermobility. "
            "NEUROLOGICAL MANIFESTATIONS: "
            "Seizures (often refractory, onset 2-3 months); "
            "severe intellectual disability; "
            "hypotonia progressing to spasticity; "
            "loss of milestones (regression); "
            "MRI brain: Leigh-like pattern (basal ganglia signal change); "
            "progressive cerebellar and cerebral atrophy; "
            "without treatment: death usually by age 3 years. "
            "TREATMENT: "
            "COPPER-HISTIDINE (Cu-histidine) subcutaneous injections: "
            "must be initiated within 4-6 WEEKS OF BIRTH before irreversible neurodegeneration; "
            "newborn screening + rapid diagnosis is essential; "
            "Cu-histidine bypasses the enterocyte block (parenteral route); "
            "outcome even with early treatment is limited — some neurological damage inevitable; "
            "patients with NORMAL or NEAR-NORMAL baseline EEG at diagnosis have better outcomes; "
            "genetic diagnosis in utero allows planned immediate postnatal treatment. "
            "OCCIPITAL HORN SYNDROME (OHS): "
            "allelic milder variant of ATP7A mutations; "
            "connective tissue features predominant: "
            "occipital horns (calcifications at tendon insertion), "
            "bladder diverticula, arterial tortuosity, skin/joint laxity; "
            "MINIMAL neurological involvement (compared to classic Menkes); "
            "these patients can survive to adulthood; "
            "also called X-linked cutis laxa. "
            "EPIDEMIOLOGY: "
            "~1:100,000-300,000 live male births; "
            "X-linked recessive: virtually all affected are male; "
            "females are carriers (usually unaffected) — rare female cases reported with skewed X-inactivation; "
            "de novo mutations in ~30% of cases (no family history). "
            "DIAGNOSIS: "
            "Serum copper: very low (<10 µg/dL; normal 70-140 µg/dL); "
            "ceruloplasmin: very low (<0.1 g/L); "
            "plasma catecholamines: dopamine high, norepinephrine low "
            "(dopamine-beta-hydroxylase Cu-dependent); "
            "MRI brain: early changes, arterial tortuosity on MRA; "
            "ATP7A gene sequencing (mutations throughout the gene); "
            "hair microscopy: pili torti on light microscopy. "
            "KEY CLINICAL FACTS: "
            "Menkes = copper deficiency; Wilson = copper excess; "
            "pili torti = hallmark; inspect hair under microscope in any hypotonic infant; "
            "subdural haematoma in Menkes → child protection alert (mimics NAI) — "
            "copper studies before assuming NAI in male infant with subdural + loose joints; "
            "4-6 week treatment window is critical — early genetic diagnosis from family history."
        ),
        "age_of_onset": "Neonatal-infantile (2-3 months for seizures; symptoms from birth)",
        "inheritance": "XLR",
        "locus": "Xq21.1",
        "protein_size": "1500 aa",
        "key_biomarker": "Serum Cu very low + ceruloplasmin very low; plasma catecholamines (high dopamine, low NE)",
        "pathognomonic": "Pili torti (kinky hair) + low Cu + low ceruloplasmin + neurodegeneration in male infant",
        "treatment": "Cu-histidine SQ within 4-6 weeks of birth; outcome poor without very early treatment",
        "critical_flags": [
            "PILI-TORTI-KINKY-HAIR-PATHOGNOMONIC — twisted brittle depigmented hair; inspect under microscope",
            "XLR-MALES-AFFECTED-FEMALES-CARRIERS — de novo in 30%; all presenting patients are male",
            "CU-HISTIDINE-SQ-WITHIN-4-6-WEEKS-BIRTH — irreversible neurodegeneration after this window",
            "LOW-CU-LOW-CERULOPLASMIN-BOTH — unlike Wilson where ceruloplasmin low but Cu tissue high",
            "AORTIC-TORTUOSITY-BLADDER-DIVERTICULA-CI-ELECTIVE-SURGERY — connective tissue fragility",
            "SUBDURAL-HAEMATOMA-MIMICS-NAI — do Cu studies before child protection in male infant",
            "CONNECTIVE-TISSUE-FAILURE-LYSYL-OXIDASE — collagen/elastin crosslinking impaired",
            "OCCIPITAL-HORN-SYNDROME-MILDER-ALLELIC — connective tissue predominant; neurologically spared",
        ],
    },
    # -- HFE -- Hereditary Hemochromatosis Type 1 --------------------------------
    {
        "gene": "HFE",
        "protein": (
            "HFE -- 6p22.2 AR -- HLA-linked-MHC-Class-I-Protein-348aa -- "
            "HH-Type-1-C282Y-80-85pct-HH-Alleles-Northern-European-Founder -- "
            "2nd-3rd-MCP-Arthropathy-PATHOGNOMONIC-Early -- "
            "Phlebotomy-500mL-Weekly-Ferritin-<50-µg/L-CURATIVE -- "
            "Transferrin-Saturation->45pct-Screen -- "
            "Alcohol-CI-VitaminC-Megadoses-CI"
        ),
        "alias": (
            "HFE (homeostatic iron regulator); OMIM gene 613609; "
            "hereditary hemochromatosis type 1 OMIM 235200. "
            "6p22.2; 348 aa; ~40 kDa; HLA-linked, non-classical MHC class I protein; "
            "expressed in hepatocytes and duodenal crypt cells; autosomal recessive. "
            "FUNCTION: HFE forms a complex with beta-2 microglobulin and interacts with "
            "transferrin receptor 1 (TfR1) and TfR2 on the hepatocyte surface. "
            "This HFE/TfR2 complex senses plasma transferrin saturation and "
            "signals upregulation of hepcidin (HAMP) transcription. "
            "Without functional HFE: "
            "the hepatocyte cannot adequately sense iron loading → "
            "hepcidin is not upregulated appropriately → "
            "ferroportin (SLC40A1) on enterocytes and macrophages is not degraded → "
            "UNRESTRICTED dietary iron absorption → progressive iron accumulation. "
            "COMMON VARIANTS: "
            "p.Cys282Tyr (C282Y): most common HH allele; "
            "80-85% of HH cases in Northern Europeans; "
            "Northern European founder mutation (Viking/Celtic origin); "
            "C282Y frequency: ~1:10 carrier in Northern Europeans; "
            "C282Y homozygotes: ~1:200-300 Northern Europeans = "
            "MOST COMMON AUTOSOMAL RECESSIVE DISORDER in this population; "
            "C282Y disrupts a disulfide bond in HFE → protein misfolded → "
            "cannot reach cell surface → cannot interact with TfR2; "
            "p.His63Asp (H63D): milder; only clinically significant in compound heterozygosity C282Y/H63D; "
            "H63D homozygotes: rarely cause clinical disease. "
            "PHENOTYPIC PENETRANCE: "
            "LOW — only ~30% of C282Y homozygous MALES develop clinical disease; "
            "females: even lower penetrance (oestrogen promotes iron loss via menstruation); "
            "many C282Y homozygotes identified by family cascade remain asymptomatic. "
            "CLINICAL FEATURES (chronological): "
            "Early (2nd-3rd decade): asymptomatic with elevated transferrin saturation; "
            "arthropathy — 2nd and 3rd MCP joints (metacarpophalangeal) PATHOGNOMONIC early sign "
            "(iron deposits in synovium; chondrocalcinosis on X-ray); "
            "Intermediate (4th-5th decade): "
            "fatigue, hepatomegaly, elevated transaminases; "
            "diabetes mellitus (pancreatic iron deposition); "
            "hypogonadism (pituitary and gonadal iron); "
            "Late: "
            "liver cirrhosis (and HCC risk, even after phlebotomy); "
            "cardiomyopathy and arrhythmias; "
            "CLASSIC TRIAD (late, often referred to in exams): "
            "cirrhosis + diabetes mellitus + bronze skin (hyperpigmentation) — "
            "LATE manifestations, rarely seen if screened early. "
            "DIAGNOSIS: "
            "Transferrin saturation (TS) >45%: primary screening test "
            "(earlier than ferritin rises; TS rises first); "
            "serum ferritin >1000 µg/L: associated with organ damage (biopsy mandatory); "
            "HFE genotyping: C282Y/C282Y confirms diagnosis in most cases; "
            "liver biopsy: required if ferritin >1000 µg/L or clinical signs of cirrhosis "
            "(to assess fibrosis and hepatic iron concentration); "
            "MRI liver: T2* sequence for iron quantification (non-invasive). "
            "TREATMENT: "
            "PHLEBOTOMY: 500 mL (~250 mg iron) per session; "
            "weekly until ferritin <50 µg/L; "
            "then maintenance phlebotomy 2-4x per year; "
            "CURATIVE if started before cirrhosis; "
            "once cirrhosis established: HCC surveillance mandatory 6-monthly (AFP + USS); "
            "RESTRICTIONS: "
            "ALCOHOL ABSOLUTELY CI — additive hepatotoxicity; "
            "VITAMIN C MEGADOSES CI — mobilises iron from stores → acute cardiac toxicity "
            "(can precipitate arrhythmia); "
            "RED MEAT and iron-rich foods: moderate restriction; "
            "avoid iron supplements; "
            "family cascade: all first-degree relatives need HFE genotyping + TS. "
            "KEY CLINICAL FACTS: "
            "C282Y = most common AR disorder in Northern Europeans; "
            "TS >45% + ferritin elevated = screen for HFE mutations; "
            "phlebotomy is cheap, effective, and curative pre-cirrhosis; "
            "arthropathy does NOT improve with phlebotomy (different from other manifestations); "
            "liver HCC risk persists even after successful phlebotomy and iron depletion — "
            "surveillance mandatory for all who had cirrhosis."
        ),
        "age_of_onset": "4th-5th decade (males); later in females (menstruation protective); TS elevated earlier",
        "inheritance": "AR",
        "locus": "6p22.2",
        "protein_size": "348 aa",
        "key_biomarker": "Transferrin saturation >45% (first screening test); serum ferritin elevated; HFE C282Y/C282Y genotype",
        "pathognomonic": "2nd-3rd MCP arthropathy in C282Y homozygote + elevated TS + ferritin",
        "treatment": "Phlebotomy 500 mL weekly until ferritin <50 µg/L; alcohol and Vitamin C megadoses CI; HCC surveillance post-cirrhosis",
        "critical_flags": [
            "C282Y-MOST-COMMON-AR-DISORDER-EUROPEANS — 1:200-300 Northern European homozygotes",
            "TRANSFERRIN-SATURATION->45pct-SCREEN — first rises before ferritin; do not screen ferritin alone",
            "PHLEBOTOMY-CURATIVE-PRE-CIRRHOSIS — 500 mL weekly until ferritin <50; maintenance 2-4x/year",
            "2ND-3RD-MCP-ARTHROPATHY-PATHOGNOMONIC — early sign; does NOT improve with phlebotomy",
            "ALCOHOL-ABSOLUTELY-CI — additive hepatotoxicity; zero tolerance",
            "VITAMIN-C-MEGADOSES-CI-CARDIAC — mobilises iron → acute arrhythmia/cardiac toxicity",
            "LIVER-BIOPSY-MANDATORY-FERRITIN->1000 — assess fibrosis; HCC surveillance post-cirrhosis",
            "PHENOTYPIC-PENETRANCE-LOW-30pct-MALES — many C282Y homozygotes asymptomatic",
        ],
    },
    # -- HAMP -- Juvenile Hemochromatosis Type 2B --------------------------------
    {
        "gene": "HAMP",
        "protein": (
            "HAMP -- 19q13.12 AR -- Hepcidin-Antimicrobial-Peptide-84aa -- "
            "Juvenile-Hemochromatosis-Type-2B-Most-Severe-HH -- "
            "CARDIAC-Involvement-EARLIEST-Leading-Cause-of-Death -- "
            "Onset-2nd-3rd-Decade-Juvenile-Unlike-HFE-4th-5th -- "
            "Hypogonadism-Primary-Amenorrhoea-Early -- "
            "Aggressive-Phlebotomy-Urgent-Deferoxamine-Cardiac-Crisis"
        ),
        "alias": (
            "HAMP (hepcidin antimicrobial peptide); OMIM gene 606464; "
            "juvenile hemochromatosis type 2B OMIM 613313. "
            "19q13.12; 84 aa precursor protein; ~25 aa active peptide (cleaved from 84aa precursor); "
            "secreted by liver; autosomal recessive. "
            "HAMP encodes the 84-amino-acid preprohepcidin — "
            "SMALLEST of all hereditary HH-associated proteins; "
            "cleaved to yield the 25-amino-acid bioactive hepcidin peptide. "
            "FUNCTION: Hepcidin is the master hormone of iron homeostasis. "
            "Hepcidin binds ferroportin (SLC40A1) on cell surfaces → "
            "induces ferroportin internalisation and degradation → "
            "blocks iron export from: enterocytes (dietary iron absorbed but not released), "
            "macrophages (recycled iron from haemoglobin not released), "
            "hepatocytes (stored iron not released). "
            "When hepcidin is appropriately high (iron replete): iron export blocked. "
            "When hepcidin is appropriately low (iron deficient): iron export allowed. "
            "In HAMP deficiency: "
            "hepcidin absent/very low → ferroportin constitutively active → "
            "UNRESTRICTED iron absorption and macrophage iron release → "
            "RAPID massive iron accumulation in ALL organs. "
            "SEVERITY: "
            "JHH (types 2A HJV and 2B HAMP) = MOST SEVERE hereditary hemochromatosis; "
            "accumulates iron FAR faster than HFE (type 1) — "
            "symptoms in 2nd-3rd decade vs 4th-5th for HFE. "
            "JHH type 2A (hemojuvelin HFE2/HJV) is MORE COMMON than type 2B (HAMP). "
            "CARDIAC INVOLVEMENT — EARLIEST AND MOST LIFE-THREATENING: "
            "dilated cardiomyopathy: systolic dysfunction, reduced ejection fraction; "
            "arrhythmias: atrial fibrillation, ventricular arrhythmia, heart block; "
            "congestive heart failure; "
            "CARDIAC IRON DEPOSITION is the leading cause of death in untreated JHH; "
            "unlike HFE HH where cardiac involvement is a late complication, "
            "in JHH cardiac disease may present before obvious hepatic disease. "
            "MRI cardiac T2*: iron deposition quantification — MANDATORY in JHH. "
            "HYPOGONADISM (early and often first non-cardiac presentation): "
            "pituitary iron deposition → gonadotrophin secretion impaired → "
            "primary amenorrhoea (females), delayed puberty, delayed menarche; "
            "testicular failure (males); "
            "hypogonadotropic hypogonadism (low LH/FSH + low oestrogen/testosterone). "
            "FERRITIN: "
            "Ferritin >1000 µg/L typically by late 2nd decade; "
            "transferrin saturation markedly elevated (>80% common). "
            "DIAGNOSIS: "
            "Young patient (teens/early 20s) with unexplained cardiac disease + hypogonadism + "
            "markedly elevated ferritin + TS = JHH until proven otherwise; "
            "HFE2/HJV and HAMP sequencing; "
            "MRI liver + cardiac T2*: iron quantification; "
            "exclude HFE type 1 first (most common). "
            "TREATMENT: "
            "Aggressive phlebotomy URGENT — much faster iron accumulation requires aggressive depletion; "
            "if severe cardiac disease: deferoxamine (IV or SQ infusion — iron chelation); "
            "deferoxamine + phlebotomy combined in severe cases; "
            "cardiac management: antiarrhythmics, heart failure treatment, ICD if VT; "
            "gonadotrophin replacement for hypogonadism; "
            "early diagnosis before cardiac complications is critical. "
            "KEY CLINICAL FACTS: "
            "Hepcidin is the 'master key' of iron metabolism — its absence opens all iron export pathways; "
            "JHH type 2B (HAMP) vs type 2A (HJV) — same phenotype, different gene; "
            "juvenile (2nd-3rd decade) presentation with cardiac + hypogonadism = JHH alarm; "
            "ferritin >1000 in a teenager = JHH red flag; "
            "cardiac MRI T2* mandatory — cardiac death preventable if diagnosed early."
        ),
        "age_of_onset": "2nd-3rd decade (juvenile, unlike HFE 4th-5th decade)",
        "inheritance": "AR",
        "locus": "19q13.12",
        "protein_size": "84 aa",
        "key_biomarker": "Ferritin >1000 µg/L in 2nd decade; TS >80%; cardiac MRI T2* iron deposition",
        "pathognomonic": "Juvenile-onset cardiac disease + hypogonadism + very high ferritin in teen/young adult",
        "treatment": "Aggressive phlebotomy URGENT; deferoxamine if severe cardiac disease; cardiac monitoring mandatory",
        "critical_flags": [
            "JUVENILE-ONSET-2ND-3RD-DECADE — cardiac + hypogonadism in teenager = JHH alarm",
            "CARDIAC-INVOLVEMENT-EARLIEST-LEADING-KILLER — dilated CMP + arrhythmia; MRI T2* cardiac mandatory",
            "HYPOGONADISM-PRIMARY-AMENORRHOEA-EARLY — pituitary iron; low LH/FSH; delayed puberty",
            "HEPCIDIN-84aa-SMALLEST-HORMONE — master iron regulator; 25aa active peptide",
            "AGGRESSIVE-PHLEBOTOMY-URGENT — faster iron accumulation than HFE; start immediately",
            "DEFEROXAMINE-CARDIAC-CRISIS — chelation if severe cardiac iron; combine with phlebotomy",
            "HFE2-HJV-MORE-COMMON-HH2A — HJV type 2A more common; same phenotype as HAMP type 2B",
            "FERRITIN->1000-2ND-DECADE-ALARM — JHH red flag; aggressively treat before cardiac damage",
        ],
    },
    # -- SLC40A1 -- Ferroportin Disease (HH Type 4) ------------------------------
    {
        "gene": "SLC40A1",
        "protein": (
            "SLC40A1 -- 2q32.2 AD -- Ferroportin-Iron-Exporter-570aa -- "
            "Hemochromatosis-Type-4-Ferroportin-Disease -- "
            "AUTOSOMAL-DOMINANT-Unlike-All-Other-HH-Forms -- "
            "Type4A-LoF-Macrophage-Iron-LOW-TS -- "
            "Type4B-Hepcidin-Resistance-HIGH-TS -- "
            "Distinguish-4A-vs-4B-CRITICAL-Different-Treatment"
        ),
        "alias": (
            "SLC40A1 (solute carrier family 40 member 1; ferroportin; IREG1; MTP1); "
            "OMIM gene 604653; "
            "hemochromatosis type 4 / ferroportin disease OMIM 606069. "
            "2q32.2; 570 aa; ~62 kDa; 12-transmembrane iron exporter; "
            "expressed on basolateral surface of enterocytes, macrophages, hepatocytes, placenta; "
            "AUTOSOMAL DOMINANT. "
            "FUNCTION: Ferroportin (SLC40A1) is the SOLE known cellular iron exporter in mammals. "
            "It exports iron from: "
            "enterocytes (dietary iron into portal blood), "
            "macrophages (recycled iron from haemoglobin into plasma), "
            "hepatocytes (stored iron into plasma). "
            "Ferroportin is regulated by hepcidin: "
            "hepcidin binds ferroportin → internalisation → degradation → iron export stopped. "
            "Without ferroportin binding (hepcidin resistance mutations): "
            "iron export continues unchecked. "
            "AUTOSOMAL DOMINANT: "
            "Unlike HFE, HFE2, TFR2, and HAMP (all AR), "
            "SLC40A1 disease is AUTOSOMAL DOMINANT (heterozygous mutations cause disease). "
            "Why dominant: "
            "Loss-of-function (LoF) mutations: haploinsufficiency — one normal allele insufficient; "
            "Gain-of-function (GoF/hepcidin-resistance): the mutant ferroportin is resistant to "
            "hepcidin binding — the mutant protein acts in a dominant-negative or gain-of-function manner. "
            "TWO DISTINCT PHENOTYPES BY MUTATION TYPE — CRITICAL TO DISTINGUISH: "
            "TYPE 4A (Loss-of-function / classic ferroportin disease): "
            "ferroportin cannot export iron efficiently → "
            "iron trapped in macrophages (reticuloendothelial system — Kupffer cells); "
            "SERUM FINDINGS: HIGH ferritin + LOW-NORMAL transferrin saturation (TS); "
            "ANAEMIA may be present (iron trapped in macrophages, not available for erythropoiesis); "
            "liver biopsy: iron deposits in KUPFFER CELLS (not hepatocytes) — DIAGNOSTIC; "
            "DO NOT over-phlebotomize — risk of anaemia; "
            "therapeutic phlebotomy limited by tolerance (Hb must remain >12 g/dL). "
            "TYPE 4B (Gain-of-function / hepcidin resistance): "
            "ferroportin mutation prevents hepcidin from binding → "
            "ferroportin remains active despite high hepcidin → "
            "unrestricted iron export from gut → hepatocyte iron loading; "
            "SERUM FINDINGS: HIGH ferritin + HIGH transferrin saturation (like HFE HH); "
            "liver biopsy: iron deposits in HEPATOCYTES (not Kupffer cells); "
            "treatment: same as HFE type 1 (phlebotomy to ferritin <50 µg/L); "
            "TYPE 4A vs 4B DISTINCTION: "
            "TS: low-normal (4A) vs high (4B); "
            "biopsy: Kupffer cell iron (4A) vs hepatocyte iron (4B); "
            "functional hepcidin test can distinguish; "
            "this distinction is critical: "
            "Type 4A → cautious limited phlebotomy; "
            "Type 4B → aggressive phlebotomy (same as HFE HH). "
            "CLINICAL COURSE: "
            "Type 4A: typically milder; ferritin can be very high without same organ damage risk "
            "as hepatocyte iron loading; anaemia complicates treatment; "
            "Type 4B: similar to HFE HH but earlier onset (AD, only one hit needed); "
            "liver disease, diabetes, cardiac involvement in severe cases. "
            "DIAGNOSIS: "
            "Family history: autosomal dominant pattern (parent affected); "
            "serum ferritin + TS (key to distinguishing 4A vs 4B); "
            "SLC40A1 sequencing; "
            "liver biopsy if ferritin >1000 µg/L (Kupffer vs hepatocyte distribution). "
            "KEY CLINICAL FACTS: "
            "Ferroportin disease = ONLY HH that is autosomal dominant; "
            "Type 4A — high ferritin, low TS — macrophage iron — cautious phlebotomy; "
            "Type 4B — high ferritin, high TS — hepatocyte iron — aggressive phlebotomy; "
            "getting this wrong can cause anaemia (over-treating 4A) or under-treat 4B."
        ),
        "age_of_onset": "Variable; 4th-6th decade typical; earlier than HFE due to single hit (AD)",
        "inheritance": "AD",
        "locus": "2q32.2",
        "protein_size": "570 aa",
        "key_biomarker": "Ferritin HIGH; TS low-normal (Type 4A) OR TS high (Type 4B); liver biopsy Kupffer vs hepatocyte iron",
        "pathognomonic": "Autosomal dominant hemochromatosis + high ferritin; TS distinguishes 4A (low) from 4B (high)",
        "treatment": "Type 4A: cautious limited phlebotomy (anaemia risk); Type 4B: aggressive phlebotomy same as HFE type 1",
        "critical_flags": [
            "AD-DOMINANT-UNLIKE-OTHER-HH — only hemochromatosis that is autosomal dominant",
            "TYPE4A-MACROPHAGE-IRON-LOW-TS — high ferritin + low-normal TS; Kupffer cell iron on biopsy",
            "TYPE4B-HEPCIDIN-RESISTANCE-HIGH-TS — high ferritin + high TS; hepatocyte iron on biopsy",
            "DO-NOT-OVER-PHLEBOTOMIZE-TYPE4A-ANAEMIA — iron trapped in macrophages not available for RBC",
            "KUPFFER-CELL-IRON-4A-HEPATOCYTE-4B — biopsy distribution distinguishes; CRITICAL distinction",
            "AUTOSOMAL-DOMINANT-HETEROZYGOUS — one mutation sufficient; family cascade: 50% risk to offspring",
            "FERRITIN-HIGH-TS-DETERMINES-TYPE — TS is the key first test to determine treatment approach",
            "FUNCTIONAL-HEPCIDIN-TEST-DISTINGUISHES — research tool to confirm 4A vs 4B if biopsy equivocal",
        ],
    },
    # -- CP -- Aceruloplasminemia -----------------------------------------------
    {
        "gene": "CP",
        "protein": (
            "CP -- 3q25.1 AR -- Ceruloplasmin-Multicopper-Ferroxidase-1058aa -- "
            "Aceruloplasminemia-Ceruloplasmin-ABSENT-Not-Low -- "
            "TRIAD-DM+Retinal-Degeneration+Neurodegeneration-PATHOGNOMONIC -- "
            "Serum-Iron-LOW-Paradox-Despite-Total-Body-Iron-Overload -- "
            "MRI-T2-Low-Signal-Basal-Ganglia-Iron-Deposition -- "
            "Deferoxamine-IV-Chelation-FFP-Temporary-Replacement"
        ),
        "alias": (
            "CP (ceruloplasmin); OMIM gene 117700; "
            "aceruloplasminemia OMIM 604290. "
            "3q25.1; 1058 aa; ~132 kDa; secreted plasma glycoprotein; "
            "multicopper ferroxidase (contains 6-7 copper atoms per molecule); "
            "synthesised primarily in liver; autosomal recessive. "
            "FUNCTION: Ceruloplasmin is the major copper-transport protein in plasma, "
            "but its most critical enzymatic function is as a FERROXIDASE: "
            "it oxidises Fe2+ (ferrous, soluble, toxic) → Fe3+ (ferric, insoluble, transferrin-bound); "
            "this oxidation step is ESSENTIAL for iron to bind transferrin and be transported to tissues; "
            "without ceruloplasmin ferroxidase: "
            "Fe2+ cannot be oxidised → iron CANNOT exit cells via ferroportin efficiently "
            "(ferroportin exports Fe2+ but oxidation to Fe3+ is needed to bind transferrin in plasma); "
            "iron accumulates INSIDE cells in liver, brain, retina, pancreas. "
            "NOTE: CP is a copper protein but the DISEASE is an IRON storage disorder. "
            "DISTINCT FROM WILSON DISEASE: "
            "Wilson = ATP7B (copper-ATPase) → copper accumulation (copper disorder); "
            "Aceruloplasminemia = CP (ferroxidase) → IRON accumulation (iron disorder, not copper); "
            "both have absent ceruloplasmin; "
            "no Kayser-Fleischer rings in aceruloplasminemia; "
            "no hepatic copper accumulation. "
            "PATHOGNOMONIC TRIAD (adult onset, 4th-5th decade): "
            "1. DIABETES MELLITUS: pancreatic islet iron deposition → beta-cell destruction; "
            "often the FIRST presentation; type 1-like (immune-negative); "
            "2. RETINAL DEGENERATION: pigmentary retinopathy; visual loss; "
            "subretinal pigmentation; blurred vision; "
            "3. NEURODEGENERATION: cerebellar ataxia (gait, coordination); "
            "parkinsonism (bradykinesia, rigidity, tremor); "
            "dementia / cognitive decline; "
            "chorea, dystonia; "
            "progressive over years to decades. "
            "SERUM FINDINGS — PARADOX: "
            "Ceruloplasmin: ABSENT (undetectable, not just low) — "
            "this is pathognomonic; even in Wilson, ceruloplasmin is LOW but not undetectable; "
            "Serum IRON: LOW (paradox — total body iron is HIGH); "
            "reason: iron cannot be oxidised to Fe3+ → cannot bind transferrin → "
            "iron trapped in cells; transferrin saturation: low-normal or normal; "
            "Serum FERRITIN: VERY HIGH (reflecting total body iron overload); "
            "Anaemia: present despite iron overload (iron cannot be mobilised for erythropoiesis). "
            "MRI BRAIN: "
            "T2 LOW signal (dark) in basal ganglia (especially putamen, globus pallidus), "
            "thalamus, cerebellar dentate nuclei, cortex — "
            "iron is paramagnetic: T2 signal decreases with iron accumulation; "
            "T1 may show high signal (blood products); "
            "this MRI pattern is DIAGNOSTIC in the context of the triad. "
            "DIAGNOSIS: "
            "Absent serum ceruloplasmin; "
            "elevated serum ferritin; "
            "low serum iron; "
            "MRI brain T2 low signal in basal ganglia; "
            "CP gene sequencing; "
            "liver biopsy: hepatic iron elevated; "
            "EXCLUDE Wilson (check hepatic Cu, ATP7B sequencing). "
            "TREATMENT: "
            "DEFEROXAMINE (desferrioxamine) IV/SQ infusion — iron chelation; "
            "removes iron from brain and other organs (crosses blood-brain barrier poorly but "
            "systemic chelation reduces iron burden); "
            "fresh frozen plasma (FFP): temporary ceruloplasmin replacement; "
            "effect transient (t½ of ceruloplasmin ~5 days); "
            "used in acute situations; "
            "recombinant ceruloplasmin: experimental; "
            "zinc supplementation: reduces dietary iron absorption; "
            "no curative therapy available; "
            "neurological progression can be slowed but not reversed. "
            "KEY CLINICAL FACTS: "
            "DM + retinal degeneration + cerebellar ataxia in adult = aceruloplasminemia DDx; "
            "ceruloplasmin ABSENT (not just low) + serum iron LOW + ferritin HIGH = diagnostic; "
            "anaemia despite iron overload = paradox explained by ferroxidase deficiency; "
            "MRI T2 basal ganglia hypointensity = key imaging sign; "
            "not a copper disorder despite ceruloplasmin being a copper protein."
        ),
        "age_of_onset": "4th-5th decade (adult onset; DM often first in 4th decade)",
        "inheritance": "AR",
        "locus": "3q25.1",
        "protein_size": "1058 aa",
        "key_biomarker": "Ceruloplasmin ABSENT/undetectable; serum iron LOW; ferritin VERY HIGH; MRI T2 low signal basal ganglia",
        "pathognomonic": "TRIAD: DM + retinal degeneration + cerebellar ataxia/neurodegeneration + absent ceruloplasmin",
        "treatment": "Deferoxamine IV chelation; FFP temporary ceruloplasmin replacement; zinc reduces absorption; no cure",
        "critical_flags": [
            "TRIAD-DM+RETINAL-DEGENERATION+NEURODEGENERATION-PATHOGNOMONIC — adult onset 4th-5th decade",
            "CERULOPLASMIN-ABSENT-NOT-LOW — undetectable; Wilson has LOW but not absent",
            "SERUM-IRON-LOW-PARADOX-DESPITE-IRON-OVERLOAD — ferroxidase absent; Fe2+ trapped in cells",
            "MRI-T2-LOW-SIGNAL-BASAL-GANGLIA — iron paramagnetic; key imaging; put/GP/thalamus/cerebellum",
            "DEFEROXAMINE-IV-CHELATION — primary treatment; slows but does not reverse neurodegeneration",
            "ANAEMIA-WITH-IRON-OVERLOAD-PARADOX — iron inaccessible for erythropoiesis",
            "DISTINCT-FROM-WILSON-NO-KF-NO-COPPER — iron disorder NOT copper despite ceruloplasmin absent",
            "FFP-TEMPORARY-CERULOPLASMIN-REPLACEMENT — t½ ~5 days; acute management only",
        ],
    },
    # -- SLC30A10 -- Hypermanganesemia with Dystonia 1 ---------------------------
    {
        "gene": "SLC30A10",
        "protein": (
            "SLC30A10 -- 1q41 AR -- Mn-Exporter-Liver-Intestine-485aa -- "
            "Hypermanganesemia-with-Dystonia-1-HMNDYT1 -- "
            "POLYCYTHEMIA-PATHOGNOMONIC-Mn-Induced-EPO-Increase -- "
            "MRI-T1-HIGH-Signal-Basal-Ganglia-Mn-Paramagnetic-PATHOGNOMONIC -- "
            "EDTA-Chelation-Treatment-Dietary-Mn-Restriction -- "
            "Childhood-Onset-Dystonia-Parkinsonism"
        ),
        "alias": (
            "SLC30A10 (solute carrier family 30 member 10; zinc transporter 10 / ZnT10); "
            "OMIM gene 611146; "
            "hypermanganesemia with dystonia 1 (HMNDYT1) OMIM 613280. "
            "1q41; 485 aa; ~54 kDa; Golgi/plasma membrane Mn2+ exporter; "
            "expressed in liver (biliary Mn export), intestine (faecal Mn export); "
            "autosomal recessive. "
            "NOTE: SLC30A10 was initially annotated as a zinc transporter but "
            "its physiologically critical function is MANGANESE export. "
            "FUNCTION: SLC30A10 exports manganese (Mn2+) from hepatocytes into bile "
            "(primary route of Mn excretion) and from enterocytes into the intestinal lumen "
            "(limiting Mn absorption). "
            "Mn is a physiologically essential trace element — cofactor for: "
            "arginase, glutamine synthetase, superoxide dismutase (MnSOD), pyruvate carboxylase. "
            "Normal Mn homeostasis depends entirely on hepatobiliary excretion "
            "(no efficient renal route for Mn, unlike other metals). "
            "In SLC30A10 deficiency: "
            "hepatocyte Mn export into bile abolished → Mn accumulates in LIVER; "
            "intestinal Mn absorption not limited → systemic Mn overload; "
            "Mn deposits in BRAIN (basal ganglia — globus pallidus preferentially), "
            "liver, and bone. "
            "PATHOGNOMONIC — POLYCYTHEMIA: "
            "Manganese is a potent stimulus for hepatic erythropoietin (EPO) production; "
            "hepatic Mn accumulation → marked EPO increase → secondary polycythemia; "
            "elevated haemoglobin, haematocrit, RBC count; "
            "polycythemia in a child with movement disorder = HMNDYT1 until proven otherwise; "
            "this polycythemia mimics iron-deficiency pattern on RBC indices "
            "(microcytic hypochromic) because Mn competes with iron in some pathways — "
            "BUT patient is NOT truly iron deficient. "
            "MRI BRAIN — PATHOGNOMONIC: "
            "T1 HIGH signal (bright/hyperintense) in basal ganglia (especially globus pallidus, "
            "subthalamic nucleus, substantia nigra); "
            "Mn is PARAMAGNETIC on T1 MRI — T1 bright signal = Mn deposition; "
            "CONTRAST WITH iron disorders: iron causes T2 LOW signal (dark); "
            "Mn causes T1 HIGH signal (bright). "
            "CLINICAL FEATURES: "
            "Childhood onset (usually 2-15 years); "
            "MOVEMENT DISORDER: dystonia (focal or generalised), bradykinesia, rigidity; "
            "PARKINSONISM: tremor, masked facies, shuffling gait; "
            "LIVER DISEASE: hepatomegaly, cirrhosis in advanced cases; "
            "no cognitive impairment initially (contrast with Wilson neuropsychiatric); "
            "progressive course; early treatment prevents neurological deterioration. "
            "BLOOD MANGANESE: very high (>2 µg/L; normal <1 µg/L; "
            "some patients >100-fold elevated). "
            "DIAGNOSIS: "
            "Blood Mn: very high; "
            "MRI brain: T1 high signal basal ganglia; "
            "liver: hepatomegaly, iron-deficiency-like picture on RBC indices; "
            "polycythemia; "
            "SLC30A10 sequencing; "
            "rule out: occupational Mn exposure (welders — similar MRI but different blood tests), "
            "total parenteral nutrition (Mn excess in TPN). "
            "TREATMENT: "
            "CaEDTA (calcium disodium EDTA) IV chelation — chelates Mn effectively; "
            "administered IV or IM; reduces blood Mn and neurological symptoms; "
            "trientine (not as effective as CaEDTA for Mn); "
            "deferiprone: also chelates Mn (off-label); "
            "DIETARY Mn RESTRICTION: eliminate Mn-rich foods "
            "(tea, nuts, whole grains, legumes, leafy vegetables); "
            "liver transplant: corrects hepatic Mn handling; "
            "reported beneficial for hepatic disease; neurological benefit less certain. "
            "KEY CLINICAL FACTS: "
            "T1 bright basal ganglia + polycythemia + dystonia in child = HMNDYT1; "
            "CaEDTA chelation can dramatically improve symptoms if started early; "
            "occupational Mn toxicity (welders): same MRI but blood Mn elevated only during exposure; "
            "dietary Mn restriction alone insufficient for severe disease; "
            "SLC30A10 = Mn exporter (despite 'ZnT10' misnomer in older literature)."
        ),
        "age_of_onset": "Childhood 2-15 years (movement disorder onset)",
        "inheritance": "AR",
        "locus": "1q41",
        "protein_size": "485 aa",
        "key_biomarker": "Blood Mn very high (>2 µg/L normal <1); MRI T1 HIGH signal basal ganglia; polycythemia",
        "pathognomonic": "POLYCYTHEMIA + T1 hyperintense basal ganglia (Mn) + childhood dystonia + blood Mn very high",
        "treatment": "CaEDTA IV chelation; dietary Mn restriction; deferiprone; liver transplant for severe hepatic disease",
        "critical_flags": [
            "POLYCYTHEMIA-PATHOGNOMONIC-Mn-OVERLOAD — elevated Hb/Hct from Mn-driven EPO; child with movement disorder",
            "T1-HIGH-SIGNAL-BASAL-GANGLIA-MRI-Mn — Mn paramagnetic on T1 (bright); contrast iron (T2 dark)",
            "CHILDHOOD-ONSET-DYSTONIA-PARKINSONISM — 2-15 years; global pallidus preferential deposition",
            "EDTA-CHELATION-TREATMENT — CaEDTA IV most effective; dramatic improvement if early",
            "BLOOD-Mn-VERY-HIGH — >2 µg/L (normal <1); rule out occupational exposure",
            "LIVER-CIRRHOSIS-Mn — primary Mn organ; transplant corrects hepatic handling",
            "DIETARY-Mn-RESTRICTION — tea, nuts, grains, legumes restricted; adjunct to chelation",
            "DEFERIPRONE-CHELATES-Mn — off-label alternative or add-on to CaEDTA",
        ],
    },
    # -- TMPRSS6 -- IRIDA -------------------------------------------------------
    {
        "gene": "TMPRSS6",
        "protein": (
            "TMPRSS6 -- 22q12.3 AR -- Matriptase-2-Serine-Protease-811aa -- "
            "Iron-Refractory-Iron-Deficiency-Anemia-IRIDA -- "
            "ORAL-IRON-COMPLETELY-INEFFECTIVE-PATHOGNOMONIC -- "
            "Hepcidin-Constitutively-HIGH-HJV-Not-Cleaved -- "
            "IV-Iron-Needed-Partial-Response-Hepcidin-Blocks-Macrophage-Release -- "
            "Childhood-Onset-1-3-Years-Sequencing-Mandatory"
        ),
        "alias": (
            "TMPRSS6 (transmembrane serine protease 6 / matriptase-2); OMIM gene 609862; "
            "iron-refractory iron deficiency anemia (IRIDA) OMIM 206200. "
            "22q12.3; 811 aa; ~90 kDa; type II transmembrane serine protease; "
            "expressed on hepatocyte surface; autosomal recessive. "
            "FUNCTION: TMPRSS6 (matriptase-2) is a cell-surface serine protease that "
            "cleaves hemojuvelin (HJV/HFE2) from the hepatocyte surface. "
            "HJV is a bone morphogenetic protein (BMP) co-receptor that promotes "
            "HAMP (hepcidin) transcription via the BMP/SMAD pathway. "
            "In iron deficiency: TMPRSS6 cleaves HJV → reduces BMP signalling → "
            "reduces hepcidin → ferroportin remains active → iron absorption and release. "
            "In TMPRSS6 deficiency: "
            "HJV cannot be cleaved → HJV remains on hepatocyte surface → "
            "BMP/SMAD signalling constitutively high → "
            "HEPCIDIN constitutively HIGH (cannot be suppressed even in iron deficiency); "
            "high hepcidin → ferroportin degraded → "
            "iron CANNOT be absorbed from gut → "
            "iron CANNOT be released from macrophages → "
            "iron deficiency anaemia despite attempted iron supplementation. "
            "PATHOGNOMONIC — ORAL IRON COMPLETELY INEFFECTIVE: "
            "oral iron supplementation completely fails (iron cannot be absorbed from gut "
            "because ferroportin on enterocytes is degraded by high hepcidin); "
            "this is the hallmark: microcytic hypochromic anaemia with "
            "COMPLETE NON-RESPONSE to repeated oral iron trials; "
            "OTHER CAUSES of iron deficiency DO respond to oral iron "
            "(coeliac, blood loss, etc.); "
            "resistance to ALL forms of oral iron is diagnostic. "
            "IV IRON — PARTIAL RESPONSE: "
            "IV iron (ferric carboxymaltose, ferumoxytol, iron sucrose) bypasses gut block; "
            "iron reaches circulation; "
            "BUT hepcidin still prevents macrophage iron release → "
            "response is INCOMPLETE — Hb rises but returns when IV iron stops; "
            "requires repeated IV iron infusions; "
            "experimental: anti-hepcidin antibodies, hepcidin antagonists in trials. "
            "CLINICAL FEATURES: "
            "Microcytic hypochromic anaemia: "
            "low MCV, low MCH, low MCHC; "
            "low serum iron, low transferrin saturation, low ferritin; "
            "elevated total iron-binding capacity (TIBC); "
            "hepcidin level: INAPPROPRIATELY HIGH for degree of iron deficiency "
            "(in classic IDA, hepcidin should be undetectable; in IRIDA it is measurable/elevated); "
            "Children: typically diagnosed at age 1-3 years; "
            "when weaning onto solid food, iron intake should meet requirements but anaemia persists. "
            "DIAGNOSIS: "
            "IRIDA vs celiac disease, blood loss, dietary deficiency: "
            "all fail oral iron → IRIDA; "
            "celiac antibodies negative; "
            "stool occult blood negative; "
            "dietary history reviewed; "
            "hepcidin inappropriately high (urine or serum hepcidin-25); "
            "IV iron trial: partial response confirms; "
            "TMPRSS6 gene sequencing: MANDATORY for definitive diagnosis; "
            "exclude: chronic disease anaemia (ACD — hepcidin high but appropriately); "
            "the key difference: in IRIDA, hepcidin is high in the context of iron deficiency "
            "(inappropriate); in ACD, hepcidin is high in the context of normal/high stores (appropriate). "
            "TREATMENT: "
            "IV iron infusions: repeated every 2-4 months (only definitive treatment available); "
            "haematology specialist review; "
            "AVOID repeated oral iron trials — no benefit and adds mucosal irritation; "
            "experimental: erythropoietin + IV iron; hepcidin inhibitors in clinical trials; "
            "monitor: CBC, reticulocytes, transferrin saturation, ferritin after each infusion. "
            "KEY CLINICAL FACTS: "
            "IRIDA = iron deficiency that cannot be corrected by oral iron; "
            "microcytic anaemia in child not responding to oral iron = IRIDA DDx; "
            "hepcidin inappropriately high = the mechanistic key; "
            "IV iron is the treatment but hepcidin block limits macrophage iron mobilisation; "
            "TMPRSS6 sequencing is mandatory — genetic test closes the diagnosis; "
            "do not stop at 'dietary iron deficiency' without excluding IRIDA by genetics."
        ),
        "age_of_onset": "Infancy to early childhood (1-3 years when diagnosed; symptoms from weaning)",
        "inheritance": "AR",
        "locus": "22q12.3",
        "protein_size": "811 aa",
        "key_biomarker": "Microcytic hypochromic anaemia; low TS; hepcidin INAPPROPRIATELY HIGH for iron deficiency; complete failure of oral iron",
        "pathognomonic": "Microcytic anaemia + complete oral iron failure + inappropriately high hepcidin + partial IV iron response",
        "treatment": "IV iron (repeated infusions); avoid oral iron; TMPRSS6 sequencing mandatory; hepcidin inhibitors experimental",
        "critical_flags": [
            "ORAL-IRON-COMPLETELY-INEFFECTIVE-PATHOGNOMONIC — ferroportin degraded; cannot absorb oral iron",
            "IV-IRON-NEEDED-PARTIAL-RESPONSE — bypasses gut but hepcidin still blocks macrophage release",
            "HEPCIDIN-INAPPROPRIATELY-HIGH — should be undetectable in iron deficiency; measurable = IRIDA",
            "MICROCYTIC-HYPOCHROMIC-LOW-TS — low MCV/MCH; low serum iron/TS; high TIBC",
            "PARENTERAL-IRON-PREFERRED — IV route only effective; oral iron contraindicated (futile)",
            "SEQUENCING-MANDATORY — TMPRSS6 gene sequencing for definitive diagnosis",
            "DO-NOT-REPEAT-ORAL-IRON-TRIALS — no benefit; mucosal irritation; delay correct diagnosis",
            "CHILDHOOD-ONSET-1-3-YEARS — diagnosis typically at weaning; anaemia refractory from outset",
        ],
    },
]


def _make_patients(gene_data, seed):
    rng = random.Random(seed)
    ages = [rng.randint(0, 65) for _ in range(40)]
    sexes = [rng.choice(["M", "F"]) for _ in range(40)]
    gene = gene_data["gene"]
    inheritance = gene_data["inheritance"]
    severity_choices = ["mild", "moderate", "severe"]
    # severity weights reflecting each gene's clinical phenotype
    sev_weights = {
        "ATP7B":    [25, 45, 30],   # Wilson — spectrum from asymptomatic to ALF
        "ATP7A":    [10, 20, 70],   # Menkes — severe neurodegeneration
        "HFE":      [45, 40, 15],   # HH1 — low penetrance; often mild if screened
        "HAMP":     [10, 30, 60],   # JHH — severe cardiac; early onset
        "SLC40A1":  [35, 45, 20],   # Ferroportin — spectrum 4A vs 4B
        "CP":       [15, 40, 45],   # Aceruloplasminemia — progressive neurological
        "SLC30A10": [20, 35, 45],   # HMNDYT1 — progressive if untreated
        "TMPRSS6":  [30, 50, 20],   # IRIDA — anaemia but manageable with IV Fe
    }
    weights = sev_weights.get(gene, [25, 45, 30])
    patients = []
    for i in range(40):
        severity = rng.choices(severity_choices, weights=weights)[0]
        age = ages[i]
        sex = sexes[i]
        on_treatment = rng.random() > 0.20
        # gene-specific complication flags
        hepatic_disease = gene in ("ATP7B", "HFE", "HAMP", "SLC40A1", "CP", "SLC30A10") and rng.random() > 0.40
        neurological = gene in ("ATP7B", "ATP7A", "CP", "SLC30A10") and rng.random() > 0.45
        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis": age,
            "sex": sex,
            "severity": severity,
            "inheritance": inheritance,
            "on_treatment": on_treatment,
            "hepatic_disease": hepatic_disease,
            "neurological_involvement": neurological,
            "key_biomarker_abnormal": True,
            "family_cascade": rng.random() > 0.45,
        })
    return patients


def _build_cohort():
    all_patients = []
    for idx, gene_data in enumerate(METAL_GENES):
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

    on_treatment = sum(1 for p in COHORT if p.get("on_treatment"))
    cascade = sum(1 for p in COHORT if p.get("family_cascade"))
    hepatic = sum(1 for p in COHORT if p.get("hepatic_disease"))
    neuro = sum(1 for p in COHORT if p.get("neurological_involvement"))

    genes_covered = len(METAL_GENES)
    ar_genes = sum(1 for g in METAL_GENES if g["inheritance"] == "AR")
    ad_genes = sum(1 for g in METAL_GENES if g["inheritance"] == "AD")
    xlr_genes = sum(1 for g in METAL_GENES if g["inheritance"] == "XLR")

    return {
        "atlas": (
            "Hereditary-Metal-Metabolism-Atlas — Complete 8-Gene Hereditary Metal & "
            "Trace Element Metabolism Disorders Atlas"
        ),
        "subtitle": (
            "ATP7B (Wilson-Disease-KF-Rings-PATHOGNOMONIC-D-Penicillamine-Trientine-Zinc-Liver-Tx) · "
            "ATP7A (Menkes-Kinky-Hair-XLR-Cu-Histidine-SQ-4-6-Weeks) · "
            "HFE (HH1-C282Y-Most-Common-AR-European-Phlebotomy-Curative-MCP-Arthropathy) · "
            "HAMP (JHH-Hepcidin-Deficiency-Juvenile-Cardiac-EARLIEST-KILLER-Hypogonadism) · "
            "SLC40A1 (Ferroportin-AD-Type4A-Macrophage-Type4B-Hepcidin-Resistance) · "
            "CP (Aceruloplasminemia-DM+Retina+Neuro-TRIAD-Ceruloplasmin-ABSENT) · "
            "SLC30A10 (Hypermanganesemia-Polycythemia-PATHOGNOMONIC-T1-High-BG-EDTA) · "
            "TMPRSS6 (IRIDA-Oral-Iron-COMPLETELY-INEFFECTIVE-PATHOGNOMONIC-IV-Iron-Partial) — "
            f"320 Patients (8×40, Seeds {SEED_BASE}–{SEED_BASE+7})"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
        "aggregate_stats": {
            "genes_covered": genes_covered,
            "ar_genes": ar_genes,
            "ad_genes": ad_genes,
            "xlr_genes": xlr_genes,
            "patients_per_gene": total // genes_covered,
            "on_treatment_pct": round(on_treatment / total * 100, 1),
            "family_cascade_pct": round(cascade / total * 100, 1),
            "hepatic_disease_pct": round(hepatic / total * 100, 1),
            "neurological_involvement_pct": round(neuro / total * 100, 1),
            "severity_mild_pct": round(severity_counts["mild"] / total * 100, 1),
            "severity_moderate_pct": round(severity_counts["moderate"] / total * 100, 1),
            "severity_severe_pct": round(severity_counts["severe"] / total * 100, 1),
        },
        "genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "disorder": g["pathognomonic"].split("+")[0].strip(),
                "treatment": g["treatment"],
                "n_patients": gene_counts.get(g["gene"], 0),
                "key_biomarker": g["key_biomarker"],
            }
            for g in METAL_GENES
        ],
        "top_alerts": [
            flag
            for g in METAL_GENES
            for flag in (g["critical_flags"][:2])
        ],
        "critical_treatment_alerts": [
            flag
            for g in METAL_GENES
            for flag in g["critical_flags"]
        ],
    }


def breakdown():
    per_gene = {}
    for g in METAL_GENES:
        gene = g["gene"]
        pts = [p for p in COHORT if p["gene"] == gene]
        mild     = sum(1 for p in pts if p["severity"] == "mild")
        moderate = sum(1 for p in pts if p["severity"] == "moderate")
        severe   = sum(1 for p in pts if p["severity"] == "severe")
        on_treatment = sum(1 for p in pts if p.get("on_treatment"))
        hepatic  = sum(1 for p in pts if p.get("hepatic_disease"))
        neuro    = sum(1 for p in pts if p.get("neurological_involvement"))
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
            "on_treatment": on_treatment,
            "on_treatment_pct": round(on_treatment / len(pts) * 100, 1) if pts else 0,
            "hepatic_disease": hepatic,
            "neurological_involvement": neuro,
            "family_cascade": sum(1 for p in pts if p.get("family_cascade")),
            "protein_description": g["protein"],
            "age_of_onset": g["age_of_onset"],
        }
    return {
        "atlas": "Hereditary-Metal-Metabolism-Atlas — Per-Gene Breakdown",
        "genes": per_gene,
        "aggregate": {
            "total_patients": len(COHORT),
            "total_genes": len(METAL_GENES),
            "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
            "all_inheritance": list({g["inheritance"] for g in METAL_GENES}),
        },
    }


def definitions():
    defs = {}
    for g in METAL_GENES:
        defs[g["gene"]] = g["alias"]

    defs["Metal & Trace Element Metabolism — Overview and Regulatory Pathways"] = (
        "Hereditary disorders of metal and trace element metabolism arise from defects in the "
        "proteins that transport, export, oxidise, or regulate copper (Cu), iron (Fe), and "
        "manganese (Mn) — the three metals covered in this atlas. "
        "COPPER HOMEOSTASIS: "
        "Dietary Cu absorbed in duodenum/proximal jejunum; "
        "CTR1 (SLC31A1) on enterocyte apical membrane imports Cu2+ → "
        "ATOX1 chaperone delivers Cu to ATP7A on basolateral surface → "
        "ATP7A exports Cu into portal blood; "
        "hepatocyte: Cu imported via CTR1 → "
        "ATP7B in trans-Golgi: "
        "(a) incorporates Cu into ceruloplasmin (holoCeruloplasmin secreted); "
        "(b) exports excess Cu into bile → faecal excretion; "
        "Defects: ATP7B (Wilson — Cu accumulation in liver/brain) · ATP7A (Menkes — Cu trapped in enterocytes/brain starved). "
        "IRON HOMEOSTASIS AND HEPCIDIN AXIS: "
        "Dietary Fe2+ imported by DMT1 in enterocyte → "
        "ferroportin (SLC40A1) exports Fe2+ at basolateral surface → "
        "ceruloplasmin / hephaestin oxidises Fe2+ → Fe3+ → binds transferrin → "
        "delivered to erythroid precursors, liver, other tissues. "
        "Hepcidin regulation: "
        "Iron replete → BMP/SMAD pathway (HJV co-receptor) → HAMP transcription → "
        "hepcidin secreted → binds SLC40A1 → internalisation/degradation → Fe export blocked. "
        "HFE/TfR2 complex senses transferrin saturation → modulates hepcidin. "
        "TMPRSS6: cleaves HJV → reduces hepcidin (iron-deficiency signal). "
        "Defects: HFE (inadequate hepcidin → iron overload HH1) · "
        "HAMP (hepcidin absent → JHH juvenile severe) · "
        "SLC40A1 (ferroportin — 4A LoF macrophage iron; 4B GoF hepcidin resistance) · "
        "CP (ferroxidase absent → Fe2+ cannot be oxidised → iron trapped in cells) · "
        "TMPRSS6 (matriptase-2 absent → hepcidin constitutively high → IRIDA). "
        "MANGANESE HOMEOSTASIS: "
        "Dietary Mn absorbed in gut; no renal route; only hepatobiliary excretion; "
        "SLC30A10 exports Mn from hepatocytes into bile; "
        "Defect: SLC30A10 absent → Mn accumulates in liver + brain (HMNDYT1). "
        "KEY BIOCHEMICAL DISCRIMINATORS: "
        "COPPER disorders: check serum ceruloplasmin + 24h urine Cu + slit-lamp; "
        "IRON overload: check TS + ferritin + HFE genotype; "
        "IRON deficiency refractory: check hepcidin + oral vs IV iron response; "
        "MANGANESE: blood Mn level + MRI T1 basal ganglia."
    )

    defs["Hemochromatosis — Classification of All Types"] = (
        "Hereditary hemochromatosis (HH) is classified by gene and pathomechanism: "
        "TYPE 1 (HFE) — most common; autosomal recessive; C282Y/C282Y; "
        "4th-5th decade; MCP arthropathy early; phlebotomy curative. "
        "TYPE 2A (HFE2/HJV) — juvenile; most common JHH gene; AR; "
        "same severe phenotype as 2B; onset 2nd-3rd decade; cardiac + hypogonadism. "
        "TYPE 2B (HAMP) — juvenile; hepcidin-encoding gene; AR; "
        "same phenotype as 2A; 84aa; cardiac LEADING KILLER. "
        "TYPE 3 (TFR2) — intermediate severity; AR; "
        "TfR2 senses transferrin saturation; adult onset similar to HFE HH. "
        "TYPE 4A (SLC40A1 LoF) — ferroportin; AUTOSOMAL DOMINANT; "
        "macrophage iron; high ferritin + LOW-NORMAL TS; cautious phlebotomy. "
        "TYPE 4B (SLC40A1 GoF/hepcidin resistance) — ferroportin; AD; "
        "hepatocyte iron; high ferritin + HIGH TS; aggressive phlebotomy. "
        "KEY: Type 4 is AD (all others AR); Type 4A vs 4B treatment is OPPOSITE; "
        "TS is the most discriminating first test between 4A and 4B. "
        "SECONDARY HEMOCHROMATOSIS: "
        "transfusional iron overload (thalassaemia major, sickle cell); "
        "parenteral iron excess; haemolytic anaemias; "
        "treatment = chelation (deferoxamine, deferiprone, deferasirox)."
    )

    defs["Copper Disorders — Wilson vs Menkes Comparison"] = (
        "Wilson disease (ATP7B) and Menkes disease (ATP7A) are both P1B-type Cu-ATPase defects "
        "but cause OPPOSITE copper phenotypes: "
        "WILSON (ATP7B — 13q14.3 — AR): "
        "ATP7B absent → hepatocyte cannot export Cu into bile → Cu ACCUMULATES; "
        "organs: liver (cirrhosis, ALF) + brain (neuropsychiatric) + cornea (KF rings) + kidney (Fanconi); "
        "serum: ceruloplasmin LOW (<0.1 g/L), 24h urine Cu HIGH (>100 µg/day); "
        "KF rings (slit-lamp, 95% neuropsychiatric Wilson); "
        "treatment: chelation (D-penicillamine or trientine) or zinc (maintenance); liver Tx curative for hepatic. "
        "MENKES (ATP7A — Xq21.1 — XLR): "
        "ATP7A absent → enterocyte cannot export Cu into portal blood → Cu TRAPPED IN GUT; "
        "systemic Cu DEFICIENT (brain, connective tissue starved of Cu); "
        "serum: ceruloplasmin LOW (both ceruloplasmin and serum Cu low); "
        "pili torti (pathognomonic kinky hair); aortic tortuosity; subdural haematoma; "
        "neurodegeneration from birth; "
        "treatment: Cu-histidine SQ within 4-6 weeks of birth. "
        "SHARED FEATURE: ceruloplasmin low. "
        "KEY DIFFERENCE: Wilson = Cu EXCESS in liver/brain; Menkes = Cu DEFICIENCY in blood/brain. "
        "DDx: Wilson presents in teens/adulthood; Menkes in neonates/infants (male only)."
    )

    defs["Hepcidin Axis — HAMP / HFE / TMPRSS6 / SLC40A1 Interactions"] = (
        "Hepcidin (encoded by HAMP) is the central regulator of iron homeostasis. "
        "HEPCIDIN PRODUCTION (liver): "
        "BMP signalling (HJV co-receptor) → SMAD1/5/8 → HAMP transcription; "
        "HFE/TfR2 complex: senses transferrin saturation → modulates HAMP; "
        "TMPRSS6: cleaves HJV → reduces BMP signalling → reduces hepcidin "
        "(the 'iron brake' releaser — allows iron absorption in iron deficiency). "
        "HEPCIDIN ACTION: "
        "binds ferroportin (SLC40A1) → internalisation → degradation → "
        "blocks iron export from enterocytes, macrophages, hepatocytes. "
        "DISORDERS OF HEPCIDIN EXCESS: "
        "TMPRSS6 deficiency (IRIDA) → hepcidin constitutively high → oral iron useless; "
        "iron-refractory iron deficiency anaemia → IV iron only. "
        "DISORDERS OF HEPCIDIN INSUFFICIENCY: "
        "HAMP deficiency → hepcidin absent → unrestricted iron → JHH (most severe); "
        "HFE deficiency → inadequate hepcidin response → HH type 1 (milder); "
        "HFE2/HJV deficiency → inadequate hepcidin → JHH type 2A (severe). "
        "FERROPORTIN DEFECTS (SLC40A1): "
        "Type 4A (LoF): ferroportin cannot export iron — macrophage iron trapped; "
        "Type 4B (GoF): ferroportin resistant to hepcidin binding — unrestricted export. "
        "DIAGNOSTIC HEPCIDIN USE: "
        "Hepcidin HIGH + iron deficiency → IRIDA (TMPRSS6); "
        "Hepcidin LOW + iron overload → HAMP/HFE2/HFE deficiency. "
        "Serum or urine hepcidin-25 assay: specialist laboratories; "
        "not yet routine clinical standard in all centres."
    )

    defs["MRI Signal Patterns in Metal Storage Disorders"] = (
        "MRI is a critical non-invasive tool for diagnosing and monitoring metal storage disorders. "
        "IRON — T2/T2* EFFECT: "
        "Iron is strongly paramagnetic → REDUCES T2 and T2* relaxation → "
        "T2 signal DECREASED (dark/hypointense) in iron-laden organs; "
        "brain: T2 dark in basal ganglia, thalamus (aceruloplasminemia, neurodegeneration with brain iron accumulation); "
        "liver: T2* MRI for hepatic iron quantification (replaces biopsy in many cases); "
        "cardiac: T2* <20 ms = significant cardiac iron (JHH, secondary hemochromatosis). "
        "MANGANESE — T1 EFFECT: "
        "Mn is paramagnetic on T1 → INCREASES T1 signal → "
        "T1 signal INCREASED (bright/hyperintense) in basal ganglia; "
        "globus pallidus most affected in HMNDYT1 (SLC30A10); "
        "T1 bright basal ganglia = Mn deposition (not iron); "
        "DDx: hepatic encephalopathy (Mn accumulation in cirrhosis), "
        "hyperalimentation (excess Mn in TPN), occupational exposure. "
        "COPPER — Wilson disease: "
        "T2 high signal in lenticular nuclei, thalamus, brainstem ('face of the giant panda' midbrain sign); "
        "T1 variable; "
        "Cu is not strongly paramagnetic but causes inflammation/oedema → T2 changes. "
        "PRACTICAL RULE: "
        "T2 DARK basal ganglia = IRON → aceruloplasminemia, NBIA syndromes; "
        "T1 BRIGHT basal ganglia = MANGANESE → HMNDYT1, TPN, cirrhosis; "
        "T2 BRIGHT lenticular nuclei = Wilson disease (copper)."
    )

    return {
        "atlas": "Hereditary-Metal-Metabolism-Atlas — Clinical Definitions",
        "definitions": defs,
        "total_genes": len(METAL_GENES),
        "total_definition_entries": len(defs),
    }
