#!/usr/bin/env python3
"""Hemoglobinopathy-Atlas — Complete 8-Gene Hereditary Red Cell Disorder Atlas
HBB     (Sickle cell disease / Beta-thalassemia; ~147 aa (mature beta-globin); 11p15.4;
         HbSS — vasoocclusion, ACS #1 mortality; Casgevy FDA Dec 2023 (CRISPR);
         hydroxyurea (HbF); voxelotor; crizanlizumab; penicillin prophylaxis birth) ·
HBA1    (Alpha-thalassemia; ~142 aa; 16p13.3; deletional; MLPA essential;
         Hb H disease (3-gene deletion) → chronic hemolysis;
         Hb Bart's hydrops fetalis (4-gene deletion) → FATAL — intrauterine transfusion) ·
G6PD    (G6PD deficiency; ~515 aa; Xq28; X-linked; 400M worldwide;
         favism — fava beans; PRIMAQUINE/DAPSONE/RASBURICASE ABSOLUTELY CI;
         test AFTER episode not during — reticulocytes falsely elevate activity;
         methylene blue CI in G6PD deficiency) ·
PKLR    (Pyruvate kinase deficiency; ~574 aa; 1q22; most common non-spherocytic hemolytic;
         2,3-DPG ELEVATED (right-shifts O2 curve — relatively protective);
         mitapivat FDA 2022 — first enzyme activator; splenectomy) ·
ANK1    (Hereditary spherocytosis type 1; ~1881 aa; 8p11.21;
         MCHC ELEVATED PATHOGNOMONIC; EMA binding test BEST;
         splenectomy curative — do AFTER 6 yrs; folate lifelong; parvovirus B19 aplasia) ·
SPTA1   (Hereditary pyropoikilocytosis / elliptocytosis; ~3054 aa; 1q23.1;
         HPP — SEVERE hemolysis; MCV 50-70 fL; heat instability at 45°C;
         AR HPP + AD HE allelic; Africa + Mediterranean) ·
SLC4A1  (Band 3 / AE1; ~911 aa; 17q21.31; Southeast Asian Ovalocytosis (SAO)
         27bp in-frame deletion — protects vs cerebral malaria; AR: distal RTA1 + hemolysis) ·
PIEZO1  (Dehydrated HS / Hereditary xerocytosis; ~2521 aa; 16q24.3; AD GOF;
         SPLENECTOMY ABSOLUTELY CI — life-threatening DVT/PE/portal-vein thrombosis;
         ektacytometry diagnostic; stomatocytes + elevated ferritin without transfusions)
320-patient aggregate cohort (8 × 40, seeds 1166–1173)
"""

import random

SEED_BASE = 1166

HEMOGLOBINOPATHY_GENES = [
    # ── HBB — Sickle cell disease / Beta-thalassemia ────────────────────────
    {
        "gene": "HBB",
        "protein": "Beta-globin (Hb)",
        "alias": (
            "HBB; OMIM gene 141900; 11p15.4; 147 aa (mature); SCD (HbSS) OMIM #603903; "
            "Beta-thalassemia major OMIM #613985; AR (HbSS, HbS/Beta-thal) or AD (HbAS carrier); "
            "SCD global prevalence ~300,000 births/yr; beta-thal major ~60,000 births/yr; "
            "p.Glu6Val (HbS); p.Glu6Lys (HbC); beta0/beta+ promoter/splice variants (beta-thal)"
        ),
        "aa": "147 aa (mature beta-globin)",
        "kDa": "~16 kDa",
        "gene_class": (
            "Oxygen-transport globin subunit; forms alpha2-beta2 tetramer (HbA = 2α+2β); "
            "p.Glu6Val (HbS) → hydrophobic Val at position 6 → deoxyHbS polymerises → "
            "sickle-shaped RBCs; Casgevy (exagamglogene autotemcel) and Lyfgenia (lovotibeglogene "
            "autotemcel) both FDA-approved Dec 2023; beta-globin locus 11p15.4; "
            "gene clusters: epsilon-Ggamma-Agamma-delta-beta (5' to 3')"
        ),
        "locus": "11p15.4",
        "omim_gene": 141900,
        "omim_disease": 603903,
        "phenotype": (
            "Sickle cell disease (HbSS): vasoocclusion crises (VOC); acute chest syndrome (ACS) — "
            "#1 cause of mortality; stroke 11% by age 20; priapism; avascular necrosis of femoral head; "
            "splenic sequestration crisis (infants); aplastic crisis (parvovirus B19); "
            "chronic haemolytic anaemia (Hb 6-9 g/dL); Hb electrophoresis: HbS >85%, no HbA; "
            "Beta-thal major: transfusion-dependent anaemia; iron overload → cardiomyopathy, "
            "cirrhosis, endocrinopathy; luspatercept FDA 2020; betibeglogene FDA 2022"
        ),
        "hallmark": (
            "ACUTE CHEST SYNDROME — new infiltrate on CXR + fever/respiratory symptoms — "
            "#1 CAUSE OF MORTALITY in SCD; exchange transfusion target HbS <30%; "
            "Casgevy (exagamglogene autotemcel) — CRISPR BCL11A — FDA Dec 2023 — "
            "FIRST FDA-APPROVED CRISPR THERAPY for both SCD and beta-thal major"
        ),
        "disease": (
            "HBB encodes the beta-globin subunit of haemoglobin (HbA = α2β2). "
            "NORMAL FUNCTION: Beta-globin pairs with alpha-globin dimers (HBA1/HBA2) to form "
            "the functional oxygen-transport tetramer. Fetal haemoglobin (HbF = α2γ2, encoded by "
            "HBG1/HBG2) predominates at birth and suppresses within 6 months as beta-globin "
            "expression increases (HbF switch); high HbF is protective in SCD and beta-thal. "
            "PATHOMECHANISM — SCD: p.Glu6Val (rs334) substitution produces HbS (hydrophobic "
            "valine replaces charged glutamate at position 6 of beta-chain); deoxyHbS molecules "
            "polymerise into rigid fibres → sickle morphology → vasoocclusion, haemolysis. "
            "Endothelial activation, neutrophil-platelet-RBC adhesion cascades amplify ischaemia. "
            "PATHOMECHANISM — Beta-thalassaemia: >300 variants (beta0 = complete LOF: nonsense, "
            "frameshift, splice-site abolishing beta-globin; beta+ = reduced: promoter, mild splice). "
            "Beta0/beta0 → transfusion-dependent (thal major); beta+/beta0 → thal intermedia; "
            "beta+/beta+ → mild anaemia/trait; excess alpha-chains precipitate → ineffective "
            "erythropoiesis + haemolytic anaemia; extramedullary haematopoiesis. "
            "CLINICAL — SCD: Vasoocclusive crisis (VOC) — bone pain (most common), "
            "dactylitis (infants, 6 months — FIRST presentation); acute chest syndrome (ACS) — "
            "new CXR infiltrate + 2 of: fever, cough, dyspnoea, hypoxia, chest pain — "
            "exchange transfusion + broad-spectrum ABx + bronchodilators; "
            "Stroke — primary prevention transfusion programme (TCD Doppler ≥200 cm/s); "
            "Splenic sequestration (infants — spleen enlarges rapidly + hypovolaemia; "
            "parents taught to palpate spleen daily); Aplastic crisis (parvovirus B19 — "
            "10-day bone marrow suppression; transfusion support; household contacts avoid exposure); "
            "Priapism (aspiration + pseudoephedrine; exchange transfusion if >4 hours); "
            "Avascular necrosis femoral/humeral head (core decompression / arthroplasty). "
            "NEWBORN SCREENING: Hb electrophoresis or HPLC — all states/countries; "
            "penicillin prophylaxis from 2 months (until 5 yrs minimum); pneumococcal vaccines. "
            "TREATMENT: Hydroxyurea (HbF inducer; reduces VOC 50%, ACS 50%, mortality; "
            "requires monitoring CBC for myelosuppression; offer to all SCD ≥9 months); "
            "Voxelotor (Oxbryta — anti-sickling haemoglobin modifier; raises Hb 1 g/dL; "
            "used with hydroxyurea); Crizanlizumab (Adakveo — anti-P-selectin MAb; "
            "reduces VOC 45%); L-glutamine (reduces oxidative stress); "
            "Allogeneic HSCT — curative but requires matched sibling; "
            "Casgevy (exagamglogene autotemcel; CRISPR-Cas9 BCL11A enhancer → de-represses HbF; "
            "FDA Dec 2023; 97% VOC-free at 12 months in trials); "
            "Lyfgenia (lovotibeglogene autotemcel; lentiviral HbAT87Q; FDA Dec 2023). "
            "BETA-THAL TREATMENT: Hypertransfusion (target Hb 9-10 g/dL, HbA >70%); "
            "Iron chelation — deferasirox (oral, hepatotoxic — LFTs); deferoxamine (IV); "
            "Luspatercept (Reblozyl — TGFβ trap; FDA 2020 — reduces transfusion burden 33%); "
            "Betibeglogene (Zynteglo — lentiviral HBB gene; FDA 2022 — functional cure 89%); "
            "Casgevy FDA Dec 2023 for beta-thal major; HSCT curative."
        ),
        "treatment_alert": (
            "ACUTE CHEST SYNDROME: exchange transfusion (target HbS <30%) + incentive spirometry "
            "+ broad-spectrum antibiotics (atypicals covered: levofloxacin or ceftriaxone+azithromycin); "
            "simple top-up transfusion is INSUFFICIENT for severe ACS. "
            "HYDROXYUREA causes transient myelosuppression — hold if ANC <2.0×10⁹/L or platelets <80×10⁹/L. "
            "IRON CHELATION mandatory when serum ferritin >1000 ng/mL or LIC >7 mg/g dry weight (MRI T2*). "
            "CASGEVY: patient must have severe SCD (≥2 VOC/yr or prior ACS/stroke) or transfusion-dependent thal; "
            "requires myeloablative busulfan conditioning — fertility counselling and gonadal cryopreservation mandatory."
        ),
        "key_ddx": (
            "HbSS vs HbSC vs HbS/beta-thal: Hb electrophoresis/HPLC (no HbA in HbSS; HbA present in HbS/beta+); "
            "HbC disease: target cells, mild anaemia, milder course; "
            "G6PD deficiency mimics SCD crises (check G6PD before primaquine); "
            "hereditary spherocytosis: spherocytes + MCHC elevated (not sickle crisis); "
            "beta-thal major vs HbE/beta-thal: SE Asia; HbA2 always >3.5% in beta-thal carrier"
        ),
        "gfr_pattern": "Not primary renal disease; sickle cell nephropathy — hyperfiltration early → CKD late (ACEi)",
        "proteinuria_pattern": "Sickle cell nephropathy: microalbuminuria → proteinuria; nephrotic in FSGS variant",
        "primary_complication": "Acute chest syndrome (SCD mortality #1); iron overload cardiomyopathy (beta-thal)",
    },

    # ── HBA1 — Alpha-thalassaemia ────────────────────────────────────────────
    {
        "gene": "HBA1",
        "protein": "Alpha-globin 1 (HBA1)",
        "alias": (
            "HBA1; OMIM gene 141800; 16p13.3; 142 aa; Alpha-thalassaemia OMIM #604131; "
            "HBA1 + HBA2 (duplicated alpha-globin genes on 16p13.3 — 4 alleles total); "
            "primarily DELETIONAL (--/αα vs -α/αα); MLPA essential — Sanger misses deletions; "
            "Hb H (3-gene del, --/-α) OMIM #613978; Hb Bart's hydrops fetalis (4-gene del) FATAL"
        ),
        "aa": "142 aa (alpha-globin)",
        "kDa": "~15 kDa",
        "gene_class": (
            "Oxygen-transport globin subunit; alpha-globin pairs with beta-globin (HbA = α2β2); "
            "also pairs with gamma-globin (HbF = α2γ2) and delta-globin (HbA2 = α2δ2); "
            "HBA1 and HBA2 located in tandem on 16p13.3; "
            "deletion genotypes: --/αα = alpha-thal-1 trait; -α/αα = alpha-thal-2 (silent carrier); "
            "--/-- = Hb Bart's hydrops fetalis (LETHAL); --/-α = Hb H disease (chronic haemolysis)"
        ),
        "locus": "16p13.3",
        "omim_gene": 141800,
        "omim_disease": 604131,
        "phenotype": (
            "Silent carrier (-α/αα): clinically silent; microcytosis mild; HbA2 LOW (vs ELEVATED in beta-thal); "
            "Alpha-thal-1 trait (--/αα): mild microcytic anaemia; HbA2 normal-LOW; "
            "Hb H disease (--/-α): moderate haemolytic anaemia (Hb 7-10 g/dL); splenomegaly; "
            "Hb H inclusion bodies (brilliant cresyl blue); Hb H 5-30% on electrophoresis; "
            "Hb Bart's hydrops fetalis (--/--): LETHAL — massive hydrops, heart failure, placentomegaly; "
            "pure gamma-chains (Hb Bart's = γ4) — extremely high O2 affinity; no oxygen delivery"
        ),
        "hallmark": (
            "HbA2 LOW (or normal) — CRITICAL DDx from beta-thalassaemia where HbA2 is ELEVATED >3.5%; "
            "MLPA IS MANDATORY — Sanger/sequencing MISSES the large deletions causing alpha-thal; "
            "Hb Bart's hydrops fetalis is an OBSTETRIC EMERGENCY — intrauterine transfusion "
            "(IUT) from 20 weeks if --/-- detected prenatally; without IUT → stillbirth or neonatal death; "
            "brilliant cresyl blue smear — Hb H inclusions like 'golf balls' in Hb H disease"
        ),
        "disease": (
            "Alpha-thalassaemia results from decreased or absent alpha-globin chain production. "
            "NORMAL: Each person has 4 alpha-globin gene copies (2 on each chromosome 16 — "
            "HBA2 (5') and HBA1 (3') in tandem). Deletions are the predominant mutation. "
            "GENOTYPE-PHENOTYPE: "
            "(1) -α/αα (1-gene deletion = silent carrier): no clinical disease; microcytosis minimal; "
            "MCV borderline; HbA2 normal or slightly LOW — key distinguisher from beta-thal trait; "
            "(2) --/αα OR -α/-α (2-gene deletion = alpha-thal-1 trait or homozygous alpha-thal-2): "
            "mild microcytic hypochromic anaemia; HbA2 low-normal; ferritin normal (NOT iron deficiency); "
            "(3) --/-α (3-gene deletion = Hb H disease): moderate chronic haemolytic anaemia "
            "(Hb 7-10 g/dL); excess beta-chains form beta4 tetramers = Hb H; "
            "Hb H detected on electrophoresis (fast-migrating) or brilliant cresyl blue "
            "(inclusion bodies 'golf balls'); splenomegaly; intermittent transfusions; "
            "(4) --/-- (4-gene deletion = Hb Bart's hydrops fetalis): LETHAL; "
            "excess gamma-chains form gamma4 tetramers = Hb Bart's (extremely high O2 affinity → "
            "tissue hypoxia); severe hydrops (anasarca, ascites, pleural/pericardial effusions); "
            "high-output cardiac failure in utero; stillbirth or early neonatal death without IUT. "
            "DIAGNOSIS: HbA2 INTERPRETATION IS CRITICAL — alpha-thal trait → HbA2 LOW or NORMAL "
            "(key DDx from beta-thal where HbA2 ELEVATED >3.5%); MLPA (Multiplex Ligation-Probe "
            "Amplification) IS THE REQUIRED TEST — common Southeast Asian deletion (--SEA), "
            "Mediterranean (--MED), Filipino (--FIL), Thai (--THAI) all MISSED by Sanger; "
            "HPLC alone insufficient — must measure HbA2 AND perform MLPA for deletions; "
            "prenatal diagnosis: CVS (11-13 weeks) or amniocentesis (16+ weeks) for at-risk couples. "
            "SOUTHEAST ASIAN PREVALENCE: --SEA deletion (carrier frequency 4-10% in SE Asia); "
            "Mediterranean: --MED; African: single-gene deletions predominant (rarely hydrops). "
            "MANAGEMENT: Hb H disease — folate 5 mg/day; avoid oxidant drugs (trigger haemolysis); "
            "splenectomy for severe cases; iron chelation if transfusion-dependent; "
            "Hb Bart's — IUT recommended from ~20 weeks if --/-- detected; "
            "survival to term possible with aggressive IUT; long-term transfusion + chelation needed. "
            "POINT MUTATION variants: Constant Spring (HbCS) — non-deletional HBA2 chain termination "
            "variant; compound --/αCSα → Hb H-CS disease (more severe than Hb H); "
            "HbCS causes haemolytic anaemia even as double heterozygote with --/αα."
        ),
        "treatment_alert": (
            "MLPA IS MANDATORY — never rely on Sanger alone for alpha-thalassaemia; "
            "large deletions (--SEA, --MED, --FIL) will be missed. "
            "HbA2 LOW (NOT elevated) in alpha-thal trait — if HbA2 >3.5%, reconsider beta-thal. "
            "Hb Bart's hydrops fetalis: counsel both parents; IUT only option for survival; "
            "maternal risk: mirror syndrome (maternal hydrops). "
            "AVOID IRON SUPPLEMENTATION in alpha-thal trait without confirmed iron deficiency "
            "(ferritin + serum iron) — iron overload risk."
        ),
        "key_ddx": (
            "Alpha-thal trait vs iron deficiency: both microcytic; ferritin/serum iron distinguishes; "
            "Alpha-thal trait vs beta-thal trait: HbA2 LOW vs ELEVATED >3.5% — KEY DISTINGUISHER; "
            "Hb H disease vs haemolytic anaemia: Hb H inclusions (brilliant cresyl blue); HPLC; "
            "Hb Bart's on HPLC at birth (neonatal screening); MLPA confirms genotype"
        ),
        "gfr_pattern": "Not primarily renal; Hb H disease: mild CKD from chronic haemolysis (rarely significant)",
        "proteinuria_pattern": "Not prominent; haemoglobinuria during haemolytic crises",
        "primary_complication": "Hb Bart's hydrops fetalis (LETHAL without IUT); iron overload (Hb H, transfusion-dependent)",
    },

    # ── G6PD — G6PD Deficiency ───────────────────────────────────────────────
    {
        "gene": "G6PD",
        "protein": "Glucose-6-phosphate dehydrogenase",
        "alias": (
            "G6PD; OMIM gene 305900; Xq28; 515 aa; G6PD deficiency OMIM #300908; "
            "X-linked; most common red cell enzymopathy; ~400 million affected worldwide; "
            "G6PD*Mediterranean (c.563C>T, Ser188Phe) — class II (severe, <10% activity); "
            "G6PD*A- (c.202G>A+c.376A>G) — class III (10-60% activity, African variant); "
            "G6PD*B — wild type; G6PD*Canton (c.1376G>T) — Asia, class II"
        ),
        "aa": "515 aa",
        "kDa": "~59 kDa monomer",
        "gene_class": (
            "Pentose phosphate pathway (PPP) enzyme; catalyses oxidation of glucose-6-phosphate "
            "to 6-phosphoglucono-delta-lactone, generating NADPH; NADPH regenerates glutathione "
            "(GSH) via glutathione reductase; GSH protects RBCs from oxidative haemolysis; "
            "RBCs have NO mitochondria → entirely dependent on PPP/NADPH for oxidative defence; "
            "functional enzyme is a dimer (active) or tetramer"
        ),
        "locus": "Xq28",
        "omim_gene": 305900,
        "omim_disease": 300908,
        "phenotype": (
            "Usually clinically silent until oxidative stress; favism (haemolysis after fava beans); "
            "drug-induced haemolysis: primaquine, dapsone, rasburicase, nitrofurantoin, "
            "high-dose aspirin; infection-induced haemolysis (commonest trigger); "
            "neonatal jaundice (first days of life — risk kernicterus); "
            "chronic non-spherocytic haemolytic anaemia (CNSHA) — class I (rare, severe variants); "
            "Heinz bodies; bite cells; blister cells on blood film during haemolytic episode"
        ),
        "hallmark": (
            "TEST G6PD ACTIVITY AFTER HAEMOLYTIC EPISODE NOT DURING — reticulocytes are young "
            "RBCs with higher G6PD activity → falsely normal result during active haemolysis; "
            "wait 3-6 months post-episode for accurate testing; "
            "METHYLENE BLUE IS CONTRAINDICATED IN G6PD DEFICIENCY — methylene blue is used to treat "
            "methaemoglobinaemia but requires G6PD/NADPH to be reduced back; in G6PD def → "
            "cannot reduce methylene blue → worsens methaemoglobinaemia + causes haemolysis; "
            "RASBURICASE CI — uric acid oxidase generates H2O2 → severe haemolysis in G6PD def"
        ),
        "disease": (
            "G6PD deficiency is the most prevalent human enzyme deficiency (~400 million affected). "
            "NORMAL FUNCTION: G6PD catalyses the first step of the pentose phosphate pathway (PPP): "
            "glucose-6-phosphate + NADP⁺ → 6-phosphoglucono-delta-lactone + NADPH. "
            "NADPH is the sole reducing agent in RBCs (no mitochondria); NADPH → glutathione "
            "reductase → reduced glutathione (GSH); GSH neutralises H₂O₂ and oxidant stress; "
            "G6PD-deficient RBCs → insufficient NADPH → GSH depleted → oxidative damage → "
            "Heinz body formation (denatured Hb) → splenic trapping (pitting) or intravascular "
            "haemolysis. "
            "WHO CLASSIFICATION: "
            "Class I (CNSHA): enzyme activity <1% of normal; severe chronic haemolysis even at rest; "
            "rare variants (G6PD*Harilaou, G6PD*Santiago de Cuba); "
            "Class II (<10%): severe episodic haemolysis; Mediterranean (most common class II); "
            "Class III (10-60%): episodic haemolysis with oxidant triggers; G6PD*A- (African); "
            "Class IV (60-150%): normal activity — clinically irrelevant. "
            "X-LINKAGE: Males hemizygous (one copy) → directly affected; "
            "Females heterozygous — LYONISATION (X-inactivation) creates mosaic of normal + "
            "deficient RBCs; heterozygous females can be clinically affected if predominantly "
            "unfavourable inactivation (especially if febrile/septic). "
            "TRIGGERS: (1) Fava beans (vicine, convicine — pro-oxidants; Mediterranean G6PD*Med); "
            "(2) DRUGS: primaquine, tafenoquine, rasburicase, dapsone, nitrofurantoin, "
            "methylene blue, high-dose aspirin, phenazopyridine, nalidixic acid; "
            "(3) Infections — commonest trigger worldwide (oxidative burst of phagocytes); "
            "(4) Neonatal period — phototherapy risk (bilirubin load + oxidant); "
            "kernicterus risk higher than beta-thal or SCD neonates. "
            "BLOOD FILM: Heinz bodies (EDTA not fixed — requires crystal violet/brilliant cresyl blue); "
            "bite cells (splenic pitting of Heinz bodies); blister cells; reticulocytosis; "
            "haemoglobinuria (dark urine). "
            "DIAGNOSIS: G6PD enzyme activity assay; fluorescent spot test (qualitative — screening); "
            "quantitative: spectrophotometric; DNA mutation analysis (large panels). "
            "TIMING: Must wait 2-3 months post-haemolytic episode (reticulocytes have higher enzyme "
            "activity — falsely normal); or test FAMILY MEMBERS to establish genotype. "
            "MALARIA PROTECTION: G6PD deficiency confers heterozygous advantage against Plasmodium "
            "falciparum; falciparum-infected G6PD-deficient RBCs are cleared faster. "
            "TREATMENT: Supportive (transfusion for severe haemolysis); avoid all oxidant triggers; "
            "neonatal jaundice — phototherapy + exchange transfusion; no specific pharmacotherapy; "
            "gene therapy trials ongoing."
        ),
        "treatment_alert": (
            "PRIMAQUINE AND TAFENOQUINE ARE ABSOLUTELY CONTRAINDICATED — must screen G6PD before "
            "prescribing either drug for Plasmodium vivax radical cure (liver hypnozoites); "
            "quantitative G6PD assay required (qualitative misses heterozygous females); "
            "RASBURICASE ABSOLUTELY CI — G6PD must be screened before tumour lysis prophylaxis; "
            "METHYLENE BLUE CI IN G6PD DEFICIENCY — alternative for metHb: ascorbic acid; "
            "DAPSONE CI — check G6PD before PCP prophylaxis or leprosy treatment. "
            "Do NOT test G6PD during acute haemolysis — wait 2-3 months for accurate result."
        ),
        "key_ddx": (
            "G6PD def vs immune haemolysis: DAT (Coombs) negative in G6PD def; "
            "G6PD def vs pyruvate kinase def: both non-spherocytic; PK def → 2,3-DPG elevated; "
            "G6PD def vs hereditary spherocytosis: no spherocytes in G6PD def (except during crisis); "
            "G6PD def vs HbSS: blood film (sickling vs Heinz bodies); electrophoresis; "
            "CNSHA (class I G6PD) vs other congenital haemolytic anaemias: enzyme assay"
        ),
        "gfr_pattern": "Not primarily renal; haemoglobinuria during acute episodes → AKI risk",
        "proteinuria_pattern": "Haemoglobinuria (dark urine) during haemolytic crises; AKI in severe attacks",
        "primary_complication": "Favism haemolytic crisis; neonatal kernicterus; AKI from haemoglobinuria",
    },

    # ── PKLR — Pyruvate Kinase Deficiency ────────────────────────────────────
    {
        "gene": "PKLR",
        "protein": "Pyruvate kinase liver/red blood cell (PKRL/PK-R)",
        "alias": (
            "PKLR; OMIM gene 609712; 1q22; 574 aa (PK-R isoform); "
            "PK deficiency OMIM #266200; AR; most common congenital non-spherocytic haemolytic "
            "anaemia (after G6PD def which is enzymopathy, not spherocytic); ~1 in 20,000 Europeans; "
            "p.Arg486Trp most common European variant; p.Arg510Gln; "
            "PKLR encodes both hepatic (PKL) and erythroid (PKR) isoforms via different promoters"
        ),
        "aa": "574 aa (PK-R isoform)",
        "kDa": "~63 kDa",
        "gene_class": (
            "Glycolytic enzyme (Embden-Meyerhof pathway); catalyses conversion of "
            "phosphoenolpyruvate (PEP) + ADP → pyruvate + ATP; in RBCs this generates ATP "
            "for Na⁺/K⁺-ATPase (RBC cation homeostasis) and membrane integrity; "
            "PK deficiency → ATP depletion → rigid RBCs → splenic haemolysis; "
            "2,3-DPG (2,3-bisphosphoglycerate) ACCUMULATES proximal to the block → "
            "right-shifts Hb-O2 dissociation curve → RELATIVELY PROTECTIVE (better O2 delivery "
            "at given Hb level); reticulocytes have mitochondria → use oxidative phosphorylation "
            "to survive despite PK deficiency — splenomegaly worsens haemolysis (splenic hypoxia "
            "eliminates mitochondrial advantage)"
        ),
        "locus": "1q22",
        "omim_gene": 609712,
        "omim_disease": 266200,
        "phenotype": (
            "Variable severity: neonatal hyperbilirubinaemia + transfusion-dependent in severe; "
            "chronic non-spherocytic haemolytic anaemia; Hb 6-12 g/dL; "
            "splenomegaly (compensatory erythropoiesis + trapping); gallstones (calcium bilirubinate); "
            "aplastic crisis (parvovirus B19); reticulocytosis (may paradoxically RISE post-splenectomy "
            "as reticulocytes previously destroyed in spleen survive); "
            "iron overload (even without transfusions — ineffective erythropoiesis increases iron absorption)"
        ),
        "hallmark": (
            "2,3-DPG ELEVATED — right-shifts Hb-O2 curve → patients tolerate lower Hb better than "
            "expected clinically (pulse oximetry may overestimate functional delivery); "
            "PARADOXICAL RETICULOCYTOSIS POST-SPLENECTOMY — reticulocyte count rises (not falls) "
            "after splenectomy because reticulocytes (which depend on mitochondria) were preferentially "
            "destroyed in the hypoxic spleen; this is PATHOGNOMONIC for PK deficiency; "
            "MITAPIVAT (Pyrukynd) FDA approved 2022 — first enzyme activator for PK deficiency"
        ),
        "disease": (
            "PKLR encodes two isoforms: PKL (hepatic) and PKR (erythroid) via tissue-specific promoters. "
            "NORMAL FUNCTION: PK-R catalyses the final ATP-generating step of glycolysis "
            "(PEP + ADP → pyruvate + ATP); mature RBCs have no mitochondria and rely entirely "
            "on glycolysis for ATP production. "
            "PATHOMECHANISM: PK-R deficiency → reduced ATP → impaired Na⁺/K⁺-ATPase → "
            "RBC dehydration + rigidity → sequestration in spleen → extravascular haemolysis. "
            "2,3-DPG ACCUMULATION (proximal metabolite builds up distal to block) → allosterically "
            "reduces Hb-O2 affinity (right-shift of dissociation curve) → relatively better O2 "
            "delivery to tissues at equivalent Hb level — patients may be less symptomatic than "
            "their low Hb suggests. "
            "RETICULOCYTES: Have mitochondria → can compensate via oxidative phosphorylation; "
            "but in the hypoxic spleen (low O2 tension), mitochondrial compensation fails → "
            "reticulocytes preferentially destroyed in spleen; post-splenectomy → reticulocytes "
            "survive in peripheral blood → reticulocyte count rises paradoxically (PATHOGNOMONIC). "
            "DIAGNOSIS: Fluorescent spot test (qualitative); quantitative PK enzyme assay "
            "(normalise to RBC count not Hb — reticulocytes have higher PK activity, can "
            "give falsely normal result if highly reticulocytic); DNA panel (>300 variants known). "
            "CLINICAL: Neonatal jaundice requiring exchange transfusion in severe cases; "
            "chronic compensated haemolysis in mild cases (diagnosed later); splenomegaly; "
            "gallstones (pigment); aplastic crisis (parvovirus B19 — emergency transfusion); "
            "iron overload from ineffective erythropoiesis + increased intestinal iron absorption "
            "(even in non-transfused patients — monitor ferritin and MRI liver iron). "
            "TREATMENT: Chronic transfusions + chelation in severe cases; folate supplementation; "
            "splenectomy: reduces transfusion need; best in children ≥5 yrs (post-vaccines); "
            "Mitapivat (Pyrukynd, Agios; FDA 2022) — small-molecule allosteric PK-R activator; "
            "increases PK-R affinity for PEP → more ATP → reduces haemolysis; "
            "improves Hb 1.5-2 g/dL; reduces transfusion burden; hepatotoxicity monitoring; "
            "gene therapy trials (lentiviral PKLR) in progress (early success)."
        ),
        "treatment_alert": (
            "MITAPIVAT (FDA 2022) — monitor LFTs (hepatotoxicity risk); do NOT stop abruptly "
            "(haemolytic rebound); trial enrollment ongoing for paediatric dosing. "
            "SPLENECTOMY: Do AFTER age 5 (vaccination: pneumococcal, Hib, meningococcal B+ACWY "
            "at least 4 weeks prior); post-splenectomy penicillin prophylaxis 2 years minimum; "
            "reticulocyte count RISING post-splenectomy is EXPECTED and PATHOGNOMONIC — "
            "do not interpret as treatment failure. "
            "IRON OVERLOAD: Monitor ferritin and MRI liver iron even in non-transfused patients "
            "(ineffective erythropoiesis increases hepcidin suppression → iron absorption increases)."
        ),
        "key_ddx": (
            "PK def vs G6PD def: both non-spherocytic; G6PD def episodic+trigger-related; "
            "PK def chronic + 2,3-DPG elevated; G6PD activity vs PK activity assays; "
            "PK def vs hereditary spherocytosis: no spherocytes in PK def; "
            "PK def vs Diamond-Blackfan: pure RBC aplasia vs haemolytic; "
            "PK def vs autoimmune HA: DAT positive vs negative; "
            "reticulocyte paradox post-splenectomy is PATHOGNOMONIC for PK def"
        ),
        "gfr_pattern": "Not primarily renal; haemoglobinuria/haemolysis can cause AKI in severe crises",
        "proteinuria_pattern": "Not prominent; haemoglobinuria during aplastic/haemolytic crises",
        "primary_complication": "Transfusion-dependent haemolytic anaemia; iron overload; aplastic crisis (parvovirus B19)",
    },

    # ── ANK1 — Hereditary Spherocytosis type 1 ──────────────────────────────
    {
        "gene": "ANK1",
        "protein": "Ankyrin-1 (Ank1)",
        "alias": (
            "ANK1; OMIM gene 612641; 8p11.21; ~1881 aa; Hereditary Spherocytosis type 1 (HS1) OMIM #182900; "
            "AD (de novo or familial) or AR; most common congenital haemolytic anaemia in northern Europeans "
            "(1:2,000); ANK1 anchors spectrin-actin cytoskeleton to band 3 (AE1/SLC4A1) protein; "
            "other HS genes: SPTB (ANK1-spectrin interaction), EPB41 (4.1R), SPTA1, SLC4A1"
        ),
        "aa": "~1881 aa",
        "kDa": "~206 kDa",
        "gene_class": (
            "Membrane adapter protein (ankyrin superfamily); three functional domains: "
            "N-terminal membrane domain (24 ANK repeats binding band 3/AE1 and other membrane proteins); "
            "spectrin-binding domain (central); C-terminal regulatory domain (death domain homology); "
            "ANK1 bridges band 3 (integral membrane) to spectrin-actin cytoskeleton; "
            "loss → reduced spectrin density → membrane instability → spherocyte formation"
        ),
        "locus": "8p11.21",
        "omim_gene": 612641,
        "omim_disease": 182900,
        "phenotype": (
            "Mild to severe haemolytic anaemia; spherocytes on blood film; MCHC ELEVATED (>36 g/dL) — "
            "PATHOGNOMONIC; MCV low; reticulocytosis; splenomegaly; jaundice neonatal and episodic; "
            "gallstones (calcium bilirubinate) by 2nd decade; aplastic crisis (parvovirus B19); "
            "megaloblastic crisis (folate deficiency in compensated chronic haemolysis); "
            "severity spectrum: silent carrier → severe transfusion-dependent (rare)"
        ),
        "hallmark": (
            "MCHC ELEVATED (>36 g/dL) IS PATHOGNOMONIC — due to relative dehydration of spherocytes "
            "(small surface area:volume ratio); "
            "EMA (eosin-5-maleimide) binding test IS THE BEST SCREENING TEST — reduced EMA binding "
            "to band 3 on flow cytometry; superior to osmotic fragility (fewer false positives/negatives); "
            "splenectomy is CURATIVE for haemolysis but GALLSTONES must be managed first "
            "(cholecystectomy concurrent or prior to splenectomy); "
            "splenectomy deferred until age ≥6 years (infection risk from asplenia)"
        ),
        "disease": (
            "Hereditary spherocytosis (HS) is the most common inherited haemolytic anaemia in "
            "northern Europeans (prevalence 1:2,000). ANK1 accounts for approximately 50-60% of HS. "
            "NORMAL FUNCTION: Ankyrin-1 (Ank1) anchors the spectrin-actin cytoskeletal network "
            "to the lipid bilayer via band 3 (AE1/SLC4A1 — the chloride/bicarbonate exchanger); "
            "this spectrin-ankyrin-band3 linkage provides the RBC with deformability, stability, "
            "and resilience during passage through the spleen. "
            "PATHOMECHANISM: ANK1 LOF variants (frameshift, nonsense, splice — haploinsufficiency) "
            "reduce spectrin density on the membrane → membrane lipid bilayer loss → "
            "spherocyte formation (spherical rather than biconcave disc); "
            "spherocytes are trapped and destroyed in the spleen (narrow inter-endothelial slits "
            "2-3 µm require normal deformability) → extravascular haemolysis. "
            "Spherocytes also have higher intracellular [Na⁺] and relative dehydration → "
            "elevated MCHC (hallmark of all spherocytes). "
            "DIAGNOSIS: Blood film (spherocytes — no central pallor); MCHC elevated (>36 g/dL); "
            "MCV low-normal; reticulocytes elevated; unconjugated bilirubin elevated; "
            "LDH elevated; haptoglobin low; "
            "EMA (eosin-5-maleimide) binding flow cytometry — BEST test: EMA covalently binds "
            "to Lys430 of band 3; in HS (regardless of which gene) EMA fluorescence reduced ≥16%; "
            "Cryohaemolysis test (alternative); osmotic fragility (older, less specific); "
            "DAT (Coombs) NEGATIVE (key DDx from autoimmune HS). "
            "FAMILY HISTORY: AD with variable penetrance; new mutations common; "
            "parents should be tested. "
            "COMPLICATIONS: Neonatal jaundice (phototherapy ± exchange transfusion); "
            "gallstones (calcium bilirubinate — common by teens; cholecystectomy); "
            "aplastic crisis (parvovirus B19 — 10-14 day marrow suppression; transfusion support; "
            "family contacts avoid exposure to pregnant women); "
            "megaloblastic crisis (folate depletion in high turnover — supplement with folate 1 mg/day); "
            "haemolytic crisis (infection, fever — increased haemolysis + compensated). "
            "SPLENECTOMY: Cures haemolysis and anaemia; deferred until age ≥6 yrs; "
            "VACCINATIONS MANDATORY 4 weeks prior: pneumococcal (PPV23 + PCV13), "
            "Hib, meningococcal B + ACWY; post-splenectomy penicillin V (or amoxicillin) prophylaxis "
            "≥2 years (lifelong in high-risk); emergency antibiotic plan for febrile illness. "
            "PARTIAL SPLENECTOMY considered in moderate HS to preserve some immune function. "
            "FOLATE: 1 mg/day (children) or 5 mg/day (adults) lifelong in haemolytic HS. "
            "SEVERE HS: Chronic transfusions + chelation until splenectomy."
        ),
        "treatment_alert": (
            "SPLENECTOMY COMPLICATIONS: OPSI (Overwhelming Post-Splenectomy Infection) — "
            "S. pneumoniae #1 cause; also N. meningitidis, H. influenzae; lifelong penicillin "
            "prophylaxis; medical alert card/bracelet; immediate antibiotics for ANY fever >38.5°C. "
            "PARVOVIRUS B19 APLASTIC CRISIS: Avoid contact with pregnant women/immunocompromised "
            "during crisis (highly infectious); RBC transfusion support for 10-14 days. "
            "GALLSTONES: Assess biliary status (USS) before splenectomy; concurrent cholecystectomy "
            "if stones present. "
            "FOLATE: Lifelong supplementation — megaloblastic crisis prevents. "
            "Do NOT confuse HS with autoimmune HA: DAT is NEGATIVE in HS."
        ),
        "key_ddx": (
            "HS vs autoimmune HA: DAT positive in AIHA, negative in HS; "
            "HS vs microangiopathic HA: blood film (schistocytes in MAHa, spherocytes in HS); "
            "HS vs dehydrated HS (PIEZO1): EMA test (reduced in HS, stomatocytes in DHS); "
            "ANK1 HS vs SPTB/EPB41/SLC4A1 HS: EMA test positive for all; gene panel distinguishes; "
            "HS vs ABO incompatibility neonatal: DAT positive in ABO"
        ),
        "gfr_pattern": "Not primarily renal; chronic haemolysis → pigment nephropathy (rare); CKD post-aplastic crisis",
        "proteinuria_pattern": "Not prominent; haemoglobinuria rare",
        "primary_complication": "Gallstones; aplastic crisis (parvovirus B19); OPSI post-splenectomy",
    },

    # ── SPTA1 — Hereditary Pyropoikilocytosis / Elliptocytosis ──────────────
    {
        "gene": "SPTA1",
        "protein": "Spectrin alpha chain (α-spectrin)",
        "alias": (
            "SPTA1; OMIM gene 182860; 1q23.1; ~3054 aa; Hereditary elliptocytosis (HE) OMIM #611804; "
            "Hereditary pyropoikilocytosis (HPP) OMIM #266140; "
            "HE — AD; HPP — AR (compound heterozygote with SPTA1 alpha-LELY allele); "
            "most severe hereditary RBC membrane disorder; alpha-LELY (Low Expression LYon) — "
            "silent common polymorphism compounding severity when trans to pathogenic variant"
        ),
        "aa": "~3054 aa",
        "kDa": "~280 kDa",
        "gene_class": (
            "RBC membrane cytoskeletal protein (spectrin family); alpha-spectrin pairs with "
            "beta-spectrin (SPTB) to form α1β1 heterodimers → associate head-to-head to form "
            "spectrin tetramers; tetramers linked by actin-protein 4.1 junctional complexes; "
            "spectrin network provides RBC deformability + biconcave shape; "
            "SPTA1 mutations disrupt alpha-beta spectrin interactions or horizontal tetramer contacts → "
            "membrane fragmentation under shear stress → elliptocytes or poikilocytes"
        ),
        "locus": "1q23.1",
        "omim_gene": 182860,
        "omim_disease": 611804,
        "phenotype": (
            "HE (heterozygote): mild/compensated; elliptocytes >25% on film; usually not anaemic; "
            "no treatment; very common in Africa (malaria protection); "
            "HPP (compound heterozygote SPTA1 pathogenic + alpha-LELY): SEVERE haemolytic anaemia; "
            "MCV 50-70 fL (markedly LOW); microspherocytes, fragments, triangular cells; "
            "Hb 6-8 g/dL; neonatal transfusion-dependent; heat instability: RBCs fragment at 45°C "
            "(normal RBCs fragment only at 49°C) — PATHOGNOMONIC TEST for HPP; "
            "splenomegaly; bone deformities from extramedullary haematopoiesis"
        ),
        "hallmark": (
            "HEAT INSTABILITY TEST: HPP RBCs fragment at 45°C vs 49°C for normal RBCs — "
            "PATHOGNOMONIC for hereditary pyropoikilocytosis; "
            "MCV 50-70 fL in HPP — SEVERELY LOW (lowest MCV of any haemolytic condition); "
            "ALPHA-LELY allele (c.6531-12C>T, also called alphaLELY) is a common low-expression "
            "polymorphism that DOES NOT CAUSE DISEASE alone but in TRANS with an SPTA1 pathogenic "
            "variant → insufficient normal alpha-spectrin → HPP (not just HE); "
            "parents of HPP proband: one parent has HE + SPTA1 pathogenic, other parent has alphaLELY — "
            "KEY GENETIC COUNSELLING POINT"
        ),
        "disease": (
            "SPTA1 encodes alpha-spectrin, the largest cytoskeletal protein of the RBC membrane. "
            "NORMAL FUNCTION: Alpha-spectrin (α-SP) associates with beta-spectrin (SPTB) to form "
            "α1β1 heterodimers that self-associate head-to-head into tetramers; these tetramers "
            "cross-link via actin-protein 4.1 junctional complexes to create the submembranous "
            "spectrin network, providing mechanical stability and deformability to the RBC. "
            "Alpha-spectrin is synthesised in 3-4× molar excess over beta-spectrin; "
            "heterozygous LOF is usually well-compensated (only 1 pathogenic allele; "
            "excess normal alpha-spectrin available). "
            "PATHOMECHANISM — HE: Heterozygous mutations at alpha-beta spectrin contact sites "
            "or tetramerisation sites → weakened horizontal spectrin contacts → under shear stress "
            "→ elliptical (not round) RBC shape; usually mild or silent haemolysis. "
            "PATHOMECHANISM — HPP: Compound heterozygosity of pathogenic SPTA1 variant + "
            "alpha-LELY allele (common polymorphism c.6531-12C>T, creating cryptic splice site → "
            "exon 46 skipping → reduced alpha-spectrin expression ~65% of normal); "
            "in TRANS: one allele produces reduced/abnormal alpha-spectrin + other allele produces "
            "less spectrin (alpha-LELY) → overall severe spectrin deficiency → "
            "severe membrane fragmentation → micropoikilocytosis → SEVERE haemolytic anaemia. "
            "BLOOD FILM: HPP: microspherocytes + bizarre fragmented RBCs (triangular, budding, "
            "helmet cells) + elliptocytes; VERY LOW MCV (50-70 fL); "
            "HE: elliptocytes >25-75% (rod-shaped); mild or no anaemia. "
            "HEAT INSTABILITY TEST (HPP): Incubate blood at 45°C for 15 min; HPP RBCs fragment "
            "(pathognomonic at 45°C; normal RBCs require 49°C); in vitro test at home possible "
            "(place blood sample in water bath). "
            "EKTACYTOMETRY (osmotic gradient): defines deformability curve — characteristic for HPP "
            "vs HE vs other membranopathies. "
            "ALPHA-LELY: Present in ~15-30% of African population; alone causes no disease; "
            "in combination with pathogenic SPTA1 mutation → HPP phenotype; "
            "DNA testing essential to identify alpha-LELY status. "
            "MANAGEMENT: HE: folate; no specific treatment; avoid extreme oxidant stress; "
            "HPP: neonatal transfusion support; folate; iron chelation if transfusion-loaded; "
            "splenectomy by age 5-6 yrs (reduces transfusion need substantially in HPP); "
            "post-splenectomy: vaccines + penicillin prophylaxis; "
            "bone marrow suppression (parvovirus B19) → aplastic crisis requiring transfusion."
        ),
        "treatment_alert": (
            "SPLENECTOMY IN HPP: Significantly reduces transfusion requirement; "
            "vaccinate + penicillin prophylaxis as for HS splenectomy (same OPSI risk). "
            "ALPHA-LELY GENETIC COUNSELLING: Two HE parents with different SPTA1 variants "
            "can have HPP offspring (compound het) — must test both alleles in family members; "
            "risk to sibling: 25% HPP if one parent has pathogenic SPTA1 + other has alpha-LELY in trans. "
            "MCV 50-70 fL in HPP — do NOT attribute to iron deficiency without ferritin/serum iron; "
            "iron supplementation is unnecessary and potentially harmful in HPP."
        ),
        "key_ddx": (
            "HPP vs iron deficiency anaemia: MCV similarly low; HPP has haemolysis markers; ferritin normal/elevated; "
            "HPP vs HE: HPP far more severe; heat instability at 45°C; MCV lower; "
            "HE vs ovalocytosis (SLC4A1 SAO): SAO ovalocytes (stiff); HE elliptocytes (flexible); "
            "HPP vs thalassaemia: Hb electrophoresis; HbA2 normal in HPP; "
            "HPP vs microangiopathic HA: HPP chronic + congenital; MAHa acute + acquired"
        ),
        "gfr_pattern": "Not primarily renal; chronic haemolysis → intermittent haemoglobinuria; AKI rare",
        "proteinuria_pattern": "Not prominent",
        "primary_complication": "Severe neonatal haemolysis requiring exchange transfusion (HPP); aplastic crisis",
    },

    # ── SLC4A1 — Band 3 / Southeast Asian Ovalocytosis ──────────────────────
    {
        "gene": "SLC4A1",
        "protein": "Anion exchanger 1 (Band 3 / AE1)",
        "alias": (
            "SLC4A1; OMIM gene 109270; 17q21.31; 911 aa; "
            "Hereditary spherocytosis type 3 (HS3) OMIM #270970 (AR); "
            "Hereditary stomatocytosis/Southeast Asian ovalocytosis (SAO) OMIM #166900 (AD); "
            "Distal renal tubular acidosis type 1 + haemolytic anaemia OMIM #611590 (AR); "
            "SAO 27bp in-frame deletion (codons 400-408) — band 3 transmembrane domain; "
            "Most common cause of ovalocytosis in Southeast Asia (carrier frequency up to 12%)"
        ),
        "aa": "911 aa",
        "kDa": "~100 kDa monomer (80 kDa core + glycosylation)",
        "gene_class": (
            "Anion exchanger (SLC4A family); integral transmembrane glycoprotein; "
            "two functional domains: N-terminal cytoplasmic domain (binds ankyrin/ankyrin-spectrin "
            "network + aldolase/GAPDH/haemoglobin); C-terminal transmembrane domain (14 TM spans; "
            "Cl⁻/HCO₃⁻ exchanger — exports HCO₃⁻ from RBCs in lungs, imports in tissues); "
            "most abundant membrane protein (1 million copies/RBC); essential for CO₂ transport"
        ),
        "locus": "17q21.31",
        "omim_gene": 109270,
        "omim_disease": 166900,
        "phenotype": (
            "SAO (AD heterozygote): ovalocytes (oval/elliptical RBCs) 10-40%; mild or no anaemia; "
            "increased RBC rigidity; PROTECTION against cerebral malaria (P. falciparum) and "
            "P. vivax invasion; usually clinically silent; "
            "SAO homozygote (--/--): LETHAL in utero (hydrops fetalis); "
            "HS type 3 (AR): similar to ANK1 HS; spherocytes; haemolytic anaemia; "
            "AR distal RTA1 + haemolytic anaemia: incomplete RTA → hypokalaemia + nephrocalcinosis "
            "+ renal stones + growth retardation; Hb 7-10 g/dL"
        ),
        "hallmark": (
            "SOUTHEAST ASIAN OVALOCYTOSIS (SAO) PROTECTS AGAINST CEREBRAL MALARIA — "
            "SAO RBCs resist Plasmodium falciparum invasion; PfEMP3 cannot bind rigid SAO band 3; "
            "SAO cells resist cerebral cytoadhesion; "
            "SAO HOMOZYGOTE IS LETHAL — if both parents are SAO carriers, "
            "25% of offspring are --/-- → hydrops/lethal; PRENATAL DIAGNOSIS essential in "
            "SAO-endemic SE Asia; "
            "AR dRTA1 + haemolytic anaemia: consider SLC4A1 in patients with "
            "haemolytic anaemia + hypokalaemia + hyperchloraemic metabolic acidosis + nephrocalcinosis"
        ),
        "disease": (
            "SLC4A1 (Band 3 / AE1) is the principal anion transporter and structural anchor of "
            "the RBC membrane. "
            "NORMAL FUNCTION: N-terminal cytoplasmic domain anchors the spectrin-actin cytoskeleton "
            "via ankyrin (ANK1); C-terminal transmembrane domain performs Cl⁻/HCO₃⁻ exchange "
            "(critical for CO₂ transport from tissues to lungs); band 3 also forms functional "
            "complexes with GAPDH, aldolase, haemoglobin, and carbonic anhydrase II. "
            "SOUTHEAST ASIAN OVALOCYTOSIS (SAO): "
            "27bp in-frame deletion (codons 400-408; AUA→_ in transmembrane segment 3-4 junction) → "
            "band 3 protein with 9-amino acid deletion in TM domain → rigid, stiff membrane → "
            "ovalocyte shape (oval/elliptical, with transverse ridges or central 'bar'); "
            "SAO RBCs have MARKEDLY INCREASED RIGIDITY (10-fold higher than normal) but maintain "
            "membrane integrity; clinical phenotype usually MILD (compensated haemolysis or none); "
            "heterozygous SAO is NOT associated with significant anaemia in most carriers. "
            "MALARIA PROTECTION: SAO ovalocytes resist P. falciparum invasion because: "
            "(1) PfEMP3 ligand cannot bind rigid SAO band 3; "
            "(2) SAO RBCs resist knob formation needed for cytoadhesion; "
            "(3) Relative PROTECTION against cerebral malaria P. falciparum and P. vivax duffy-antigen; "
            "heterozygote frequency 4-12% in SE Asia (Indonesia, Philippines, Malaysia, Papua New Guinea). "
            "SAO HOMOZYGOTE LETHAL: --/-- → hydrops fetalis in utero; incompatible with life; "
            "AT-RISK COUPLES (both SAO carriers): prenatal testing essential. "
            "HS TYPE 3 (AR): Missense/truncating variants of SLC4A1 (distinct from SAO deletion) "
            "cause AR HS (rare); clinically similar to ANK1 HS; EMA test reduced. "
            "AR DISTAL RENAL TUBULAR ACIDOSIS TYPE 1 + HAEMOLYTIC ANAEMIA: "
            "Specific AR SLC4A1 variants (Southeast Asia: G701D, A858D; Africa: R602H) → "
            "abnormal band 3 mis-targeted to kidney collecting duct intercalated cells "
            "(normally secretes H⁺) → impaired acid secretion → hyperchloraemic metabolic "
            "acidosis + hypokalaemia; nephrocalcinosis; renal stones; growth retardation; "
            "co-existing haemolytic anaemia (spherocytic or stomatocytic). "
            "DIAGNOSIS: Blood film (ovalocytes with transverse bar — 'finger prints'); "
            "PCR for SAO 27bp deletion; EMA test (reduced in HS-type SLC4A1 mutations); "
            "urine pH: cannot acidify below 5.5 in dRTA; serum HCO₃⁻ low; serum K⁺ low; "
            "renal USS (nephrocalcinosis); genetic panel (SLC4A1 full sequencing + deletion PCR). "
            "TREATMENT: SAO — no treatment needed (usually benign); "
            "dRTA1 — bicarbonate/citrate supplementation (corrects acidosis + prevents stones); "
            "HS type 3 AR — as per ANK1 HS (splenectomy ± folate)."
        ),
        "treatment_alert": (
            "SAO HOMOZYGOSITY IS LETHAL — identify SAO couples in SE Asia; "
            "prenatal testing (CVS/amnio) if both parents SAO. "
            "dRTA1 + haemolytic anaemia: ALKALI THERAPY IS ESSENTIAL — prevents nephrocalcinosis "
            "and renal stones; monitor serum K⁺ (hypokalaemia can cause paralysis); "
            "potassium supplementation required. "
            "Do NOT mistake SAO ovalocytes for sickle cells or elliptocytes — SAO cells are oval "
            "with a characteristic transverse ridge (bar/ridge on film — pathognomonic). "
            "MALARIA PROPHYLAXIS: SAO carriers should still use antimalarial prophylaxis when "
            "travelling (protection is partial, not complete)."
        ),
        "key_ddx": (
            "SAO vs HE (SPTA1): SAO ovalocytes rigid + transverse bar; HE elliptocytes flexible; "
            "SAO vs sickle cells: no sickling; Hb electrophoresis normal; "
            "SAO dRTA1 vs primary Fanconi syndrome: SAO → distal only; no proximal tubular loss; "
            "SLC4A1 HS vs ANK1 HS: gene panel; both EMA-positive; similar clinical phenotype; "
            "SAO vs Plasmodium falciparum malaria protection: SAO reduces cerebral malaria risk but "
            "does NOT prevent infection — malaria prophylaxis still required"
        ),
        "gfr_pattern": "dRTA1: nephrocalcinosis → CKD progressive if untreated; correct acidosis early",
        "proteinuria_pattern": "dRTA1: mild tubular proteinuria; nephrocalcinosis → CKD → glomerular proteinuria late",
        "primary_complication": "SAO homozygosity (LETHAL); dRTA + nephrocalcinosis; HS-type haemolysis",
    },

    # ── PIEZO1 — Dehydrated HS / Hereditary Xerocytosis ─────────────────────
    {
        "gene": "PIEZO1",
        "protein": "PIEZO-type mechanosensitive ion channel component 1",
        "alias": (
            "PIEZO1; OMIM gene 611184; 16q24.3; ~2521 aa; "
            "Dehydrated hereditary stomatocytosis (DHS) / hereditary xerocytosis OMIM #194380; "
            "AD gain-of-function (GOF); ~50-150 million affected worldwide (Africa/Mediterranean); "
            "p.Arg2456His most common GOF variant; "
            "PIEZO1 also involved in hereditary generalised lymphatic dysplasia (GOF, AR) and "
            "red blood cell volume regulation; "
            "SPLENECTOMY ABSOLUTELY CONTRAINDICATED — life-threatening thromboembolic events"
        ),
        "aa": "~2521 aa",
        "kDa": "~288 kDa",
        "gene_class": (
            "Mechanosensitive ion channel (trimeric); largest known trimeric membrane protein; "
            "homotrimer of three 2521 aa subunits forming propeller-like blades; "
            "Ca²⁺-permeable non-selective cation channel; senses membrane tension; "
            "in RBCs: GOF → channel stays open too long → excess Ca²⁺ entry → "
            "Gardos channel (KCNN4) activation → K⁺ efflux → RBC dehydration → xerocytes; "
            "dehydrated RBCs → elevated MCHC but DIFFERENT mechanism from HS (no spectrin defect)"
        ),
        "locus": "16q24.3",
        "omim_gene": 611184,
        "omim_disease": 194380,
        "phenotype": (
            "Mild to moderate haemolytic anaemia; MCHC elevated (dehydrated RBCs); "
            "stomatocytes (mouth-shaped cells) on blood film; target cells; "
            "elevated MCHC + DECREASED MCV (dehydrated); reticulocytosis; "
            "elevated ferritin WITHOUT transfusions (ineffective erythropoiesis + increased iron absorption); "
            "hepatosplenomegaly; neonatal jaundice; perinatal oedema (paradoxical); "
            "DHS associated with hereditary generalised lymphatic dysplasia (lymphoedema) in some families"
        ),
        "hallmark": (
            "SPLENECTOMY IS ABSOLUTELY CONTRAINDICATED IN DHS/PIEZO1 — "
            "causes life-threatening thromboembolic events: DVT, PE, portal vein thrombosis, "
            "mesenteric vein thrombosis; mechanism: splenic sequestration of pro-thrombotic "
            "phosphatidylserine-exposing RBCs is lost post-splenectomy → "
            "massive thromboembolic burden; fatalities reported; "
            "EKTACYTOMETRY (osmotic gradient laser diffractometry) IS DIAGNOSTIC — "
            "characteristic leftward shift of Omin (minimum deformability point) with "
            "elevated EImax; this is PATHOGNOMONIC for DHS and distinguishes it from HS and HE; "
            "ELEVATED FERRITIN WITHOUT TRANSFUSIONS — iron overload in DHS even in non-transfused; "
            "monitor and chelate (deferasirox) when LIC elevated"
        ),
        "disease": (
            "PIEZO1 GOF variants cause Dehydrated Hereditary Stomatocytosis (DHS), also called "
            "hereditary xerocytosis — an AD red cell disorder. "
            "NORMAL FUNCTION: PIEZO1 is a mechanosensitive Ca²⁺-permeable ion channel that "
            "responds to membrane tension/shear stress; in RBCs, it regulates cell volume by "
            "briefly opening to allow Ca²⁺ entry, which then activates the Gardos channel "
            "(KCNN4, Ca²⁺-activated K⁺ channel) → transient K⁺ efflux → slight RBC dehydration. "
            "GOF PATHOMECHANISM: PIEZO1 GOF mutations (most commonly p.Arg2456His; "
            "also p.Glu756del, p.Arg2482Trp) → prolonged channel opening → "
            "excessive Ca²⁺ entry → sustained Gardos channel activation → "
            "excessive K⁺ and water loss → RBC dehydration → elevated MCHC → xerocyte (dry RBC). "
            "MOLECULAR: PIEZO1 homotrimer; 2521 aa with propeller-like peripheral blades and "
            "central ion channel pore; trimeric assembly requires all three subunits; "
            "GOF results in increased open probability or slower inactivation. "
            "BLOOD FILM: Stomatocytes (uniconcave cells with slit-like central pallor = 'mouth'); "
            "target cells; xerocytes; echinocytes; NOT spherocytes; MCHC elevated. "
            "EKTACYTOMETRY: Characteristic leftward shift of Omin AND high EImax; "
            "laser diffraction ektacytometry is the GOLD STANDARD diagnostic test; "
            "osmotic gradient ektacytometry profile is PATHOGNOMONIC for DHS/xerocytosis; "
            "distinguishes DHS from HS (different curve shape) and Hb variants. "
            "IRON OVERLOAD WITHOUT TRANSFUSIONS: Haemolysis → suppressed hepcidin → "
            "increased GI iron absorption → iron accumulation in liver + other organs; "
            "ferritin regularly elevated; LIC (liver iron concentration) on MRI; "
            "chelation with deferasirox when LIC >7 mg/g dry weight. "
            "THROMBOEMBOLIC RISK: DHS RBCs expose phosphatidylserine (PS) on outer leaflet "
            "(due to Ca²⁺ excess activating scramblase); PS-exposing RBCs are pro-thrombotic "
            "and pro-coagulant; normally spleen removes PS-exposing RBCs; "
            "splenectomy removes this protective filtration → catastrophic thromboembolic events. "
            "PERINATAL OEDEMA: Paradoxical oedema at birth (despite dehydrated RBCs); "
            "mechanism unclear — possibly related to lymphatic PIEZO1 GOF effects; resolves. "
            "LYMPHATIC DYSPLASIA: PIEZO1 GOF (AR, distinct from AD DHS) → "
            "hereditary generalised lymphatic dysplasia (lymphoedema, pleural effusions). "
            "DIAGNOSIS: Ektacytometry + MCHC elevated + stomatocytes + family history (AD); "
            "ferritin elevated; LDH elevated; haptoglobin low; reticulocytosis; "
            "EMA test (normal — distinguishes from HS); genetic panel (PIEZO1). "
            "MANAGEMENT: Folate supplementation; monitor ferritin/LIC; iron chelation when needed; "
            "splenectomy ABSOLUTELY CONTRAINDICATED; anticoagulation if thromboembolism; "
            "gene therapy and PIEZO1-channel inhibitor (GsMTx4) trials in early stages."
        ),
        "treatment_alert": (
            "SPLENECTOMY ABSOLUTELY CONTRAINDICATED IN DHS/PIEZO1 — "
            "multiple case reports of fatal portal vein thrombosis, PE, and mesenteric thrombosis "
            "after splenectomy; this is the most important clinical pearl in DHS; "
            "if splenectomy was performed pre-diagnosis, lifelong anticoagulation mandatory. "
            "IRON OVERLOAD: Monitor ferritin every 6 months; MRI liver iron if ferritin >500 ng/mL; "
            "deferasirox if LIC >7 mg/g dry weight; AVOID IRON SUPPLEMENTS (increases overload). "
            "EMA TEST IS NORMAL — differentiates DHS from HS; do ektacytometry for diagnosis. "
            "DISTINGUISH FROM HEREDITARY SPHEROCYTOSIS: Both have elevated MCHC but "
            "HS → spherocytes + EMA reduced; DHS → stomatocytes + EMA normal + ektacytometry pattern different."
        ),
        "key_ddx": (
            "DHS (PIEZO1) vs HS (ANK1/SPTB): MCHC elevated in both; spherocytes in HS; stomatocytes in DHS; "
            "EMA reduced in HS, normal in DHS; ektacytometry distinguishes definitively; "
            "DHS vs overhydrated stomatocytosis (RHAG): opposite ektacytometry curve; RHAG gene panel; "
            "DHS vs Hb CC/SC disease: stomatocytes also seen; Hb electrophoresis; "
            "DHS vs liver disease stomatocytosis: acquired; liver function tests; history; "
            "PIEZO1 GOF vs PIEZO1 LOF: DHS (GOF) vs lymphatic dysplasia (GOF AR); distinct phenotypes"
        ),
        "gfr_pattern": "Not primarily renal; haemolysis + iron overload → CKD if severe",
        "proteinuria_pattern": "Not prominent; iron overload can cause tubular proteinuria",
        "primary_complication": "Thromboembolic events (especially if splenectomy); iron overload; neonatal oedema",
    },
]


def _make_cohort(gene: dict, seed: int, n: int = 40) -> list:
    """Generate a synthetic patient cohort for a gene."""
    rng = random.Random(seed)
    g = gene["gene"]
    patients = []
    for i in range(n):
        age = rng.randint(1, 72)
        sex = rng.choice(["M", "F"])
        # Gene-specific severity and features
        if g == "HBB":
            subtype = rng.choice(["HbSS", "HbS/beta-thal", "beta-thal-major"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[20, 35, 45])[0]
            esrd = rng.random() < 0.12
            htn = rng.random() < 0.70
            drug_error = rng.random() < 0.22
            dx_delayed = rng.random() < 0.15  # newborn screening catches early
            transplant = rng.random() < 0.18
            adherent = rng.random() < 0.72
        elif g == "HBA1":
            subtype = rng.choice(["Hb H disease", "HbH-CS", "Alpha-thal trait", "Hb Barts survived"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[30, 40, 30])[0]
            esrd = rng.random() < 0.08
            htn = rng.random() < 0.30
            drug_error = rng.random() < 0.28
            dx_delayed = rng.random() < 0.45  # often missed (microcytosis attributed to iron def)
            transplant = rng.random() < 0.05
            adherent = rng.random() < 0.68
        elif g == "G6PD":
            subtype = rng.choice(["G6PD*Mediterranean", "G6PD*A-", "G6PD*Canton", "G6PD*B- other"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[45, 35, 20])[0]
            esrd = rng.random() < 0.05
            htn = rng.random() < 0.25
            drug_error = rng.random() < 0.38  # primaquine/rasburicase given without testing
            dx_delayed = rng.random() < 0.55  # often discovered only when drug given
            transplant = rng.random() < 0.03
            adherent = rng.random() < 0.60
        elif g == "PKLR":
            subtype = rng.choice(["Compound het R486W/other", "Homozygous R486W", "Severe neonatal", "Mild adult"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[25, 40, 35])[0]
            esrd = rng.random() < 0.06
            htn = rng.random() < 0.28
            drug_error = rng.random() < 0.12
            dx_delayed = rng.random() < 0.50
            transplant = rng.random() < 0.08
            adherent = rng.random() < 0.70
        elif g == "ANK1":
            subtype = rng.choice(["Mild HS", "Moderate HS", "Severe HS (AR)", "Neonatal jaundice onset"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[35, 45, 20])[0]
            esrd = rng.random() < 0.04
            htn = rng.random() < 0.22
            drug_error = rng.random() < 0.08
            dx_delayed = rng.random() < 0.30
            transplant = rng.random() < 0.04
            adherent = rng.random() < 0.82
        elif g == "SPTA1":
            subtype = rng.choice(["HE mild", "HPP compound het", "HPP neonatal severe", "HE with alphaLELY"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[30, 35, 35])[0]
            esrd = rng.random() < 0.05
            htn = rng.random() < 0.22
            drug_error = rng.random() < 0.10
            dx_delayed = rng.random() < 0.60  # HPP often misdiagnosed as IDA
            transplant = rng.random() < 0.05
            adherent = rng.random() < 0.72
        elif g == "SLC4A1":
            subtype = rng.choice(["SAO heterozygote", "HS type 3 AR", "dRTA1 + HA", "SAO + malaria coexistence"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[45, 35, 20])[0]
            esrd = rng.random() < 0.15  # dRTA1 → nephrocalcinosis → CKD
            htn = rng.random() < 0.28
            drug_error = rng.random() < 0.12
            dx_delayed = rng.random() < 0.50
            transplant = rng.random() < 0.06
            adherent = rng.random() < 0.74
        else:  # PIEZO1
            subtype = rng.choice(["DHS/xerocytosis mild", "DHS + iron overload", "DHS neonatal oedema", "DHS post-splenectomy"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[40, 40, 20])[0]
            esrd = rng.random() < 0.05
            htn = rng.random() < 0.25
            drug_error = rng.random() < 0.42  # splenectomy performed in error
            dx_delayed = rng.random() < 0.65
            transplant = rng.random() < 0.03
            adherent = rng.random() < 0.68

        patients.append({
            "id": f"{g}-{seed}-{i+1:03d}",
            "gene": g,
            "age": age,
            "sex": sex,
            "subtype": subtype,
            "severity": severity,
            "esrd": esrd,
            "hypertension": htn,
            "drug_error": drug_error,
            "dx_delayed": dx_delayed,
            "transplant": transplant,
            "adherent": adherent,
        })
    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    return {
        "n": n,
        "esrd_pct": round(100 * sum(p["esrd"] for p in patients) / n, 1),
        "htn_pct": round(100 * sum(p["hypertension"] for p in patients) / n, 1),
        "drug_error_pct": round(100 * sum(p["drug_error"] for p in patients) / n, 1),
        "dx_delayed_pct": round(100 * sum(p["dx_delayed"] for p in patients) / n, 1),
        "transplant_pct": round(100 * sum(p["transplant"] for p in patients) / n, 1),
        "adherent_pct": round(100 * sum(p["adherent"] for p in patients) / n, 1),
        "severity": {
            "Mild":     round(100 * sum(p["severity"] == "Mild"     for p in patients) / n, 1),
            "Moderate": round(100 * sum(p["severity"] == "Moderate" for p in patients) / n, 1),
            "Severe":   round(100 * sum(p["severity"] == "Severe"   for p in patients) / n, 1),
        },
    }


def _build_all_patients():
    all_patients = []
    for idx, gene in enumerate(HEMOGLOBINOPATHY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(gene, seed=seed, n=40)
        all_patients.extend(cohort)
    return all_patients


ALL_PATIENTS = _build_all_patients()


def get_overview() -> dict:
    n = len(ALL_PATIENTS)
    agg = _cohort_stats(ALL_PATIENTS)
    return {
        "atlas_name": "Hemoglobinopathy-Atlas",
        "atlas_subtitle": (
            "Complete 8-Gene Hereditary Red Cell Disorder Atlas — "
            "HBB · HBA1 · G6PD · PKLR · ANK1 · SPTA1 · SLC4A1 · PIEZO1"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1166–1173",
        "description": (
            "Comprehensive hereditary haematology reference covering the 8 most clinically "
            "significant inherited red cell disorders: sickle cell disease and beta-thalassaemia "
            "(HBB); alpha-thalassaemia (HBA1 — with critical MLPA requirement); G6PD deficiency "
            "(most common enzymopathy; 400 million worldwide; primaquine/rasburicase CI); "
            "pyruvate kinase deficiency (PKLR; mitapivat FDA 2022; 2,3-DPG elevated); "
            "hereditary spherocytosis type 1 (ANK1; MCHC elevated; EMA test; splenectomy curative); "
            "hereditary pyropoikilocytosis / elliptocytosis (SPTA1; HPP heat instability 45°C; "
            "alpha-LELY allele; MCV 50-70 fL); "
            "Band 3 / Southeast Asian ovalocytosis (SLC4A1; SAO malaria protection; "
            "SAO homozygote lethal; dRTA1); "
            "Dehydrated HS / xerocytosis (PIEZO1 GOF; SPLENECTOMY ABSOLUTELY CI; "
            "ektacytometry diagnostic; iron overload without transfusions). "
            "320-patient aggregate cohort, 8 × 40 patients, seeds 1166–1173."
        ),
        "drug_alerts": [
            {
                "type": "danger",
                "title": "PIEZO1 (DHS/Xerocytosis): SPLENECTOMY ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "Splenectomy in PIEZO1 DHS causes life-threatening thromboembolic events — "
                    "DVT, PE, portal vein thrombosis, mesenteric vein thrombosis; fatalities reported. "
                    "Manage DHS conservatively. If splenectomy was performed pre-diagnosis, "
                    "anticoagulate lifelong. Identify DHS by ektacytometry + EMA normal + MCHC elevated."
                ),
            },
            {
                "type": "danger",
                "title": "G6PD: PRIMAQUINE, TAFENOQUINE, RASBURICASE, DAPSONE — ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "Must screen G6PD activity before prescribing primaquine/tafenoquine (P. vivax "
                    "radical cure) or rasburicase (tumour lysis prophylaxis); "
                    "quantitative assay required (qualitative misses heterozygous females); "
                    "METHYLENE BLUE IS CI in G6PD deficiency (alternative: ascorbic acid for metHb); "
                    "DO NOT test G6PD during active haemolysis — false-normal from reticulocytes."
                ),
            },
            {
                "type": "danger",
                "title": "HBB (SCD): ACUTE CHEST SYNDROME — EXCHANGE TRANSFUSION, NOT SIMPLE TOP-UP",
                "body": (
                    "ACS = new CXR infiltrate + ≥2 of fever/cough/dyspnoea/hypoxia/chest pain — "
                    "#1 cause of mortality in SCD; exchange transfusion (target HbS <30%) + "
                    "incentive spirometry + broad-spectrum ABx (atypicals covered); "
                    "simple transfusion is INSUFFICIENT for severe ACS."
                ),
            },
            {
                "type": "warning",
                "title": "HBA1 (Alpha-Thal): MLPA IS MANDATORY — Sanger MISSES deletions",
                "body": (
                    "Large deletions (--SEA, --MED, --FIL, --THAI) causing Hb H disease and "
                    "Hb Bart's hydrops fetalis CANNOT be detected by Sanger sequencing; "
                    "MLPA is required. HbA2 is LOW (not elevated) in alpha-thal trait — "
                    "critical DDx from beta-thal where HbA2 >3.5%."
                ),
            },
            {
                "type": "warning",
                "title": "SLC4A1 (SAO): HOMOZYGOTE IS LETHAL — Prenatal Testing for At-Risk Couples",
                "body": (
                    "SAO (--/αα) homozygous offspring → hydrops fetalis → lethal; "
                    "prevalence of SAO carrier up to 12% in SE Asia; "
                    "both parents SAO → 25% risk lethal offspring; "
                    "CVS or amniocentesis mandatory in at-risk pregnancies."
                ),
            },
            {
                "type": "warning",
                "title": "ANK1 (HS): POST-SPLENECTOMY OPSI — Vaccination + Penicillin Mandatory",
                "body": (
                    "Splenectomy is curative for HS but OPSI risk (S. pneumoniae #1) lifelong; "
                    "vaccinate 4 weeks pre-op: PCV13 + PPV23, Hib, MenACWY + MenB; "
                    "penicillin V prophylaxis ≥2 years (lifelong in high-risk); "
                    "immediate antibiotics for ANY febrile episode post-splenectomy."
                ),
            },
        ],
        "critical_rules": [
            "PIEZO1 DHS: SPLENECTOMY ABSOLUTELY CI — thromboembolism fatalities reported; "
            "ektacytometry diagnostic; manage conservatively",
            "G6PD: Test AFTER haemolytic episode (not during) — reticulocytes falsely elevate activity; "
            "PRIMAQUINE/RASBURICASE CI without testing",
            "HBA1 alpha-thal: MLPA mandatory (Sanger misses deletions); HbA2 LOW in alpha-thal "
            "(NOT elevated — opposite of beta-thal)",
            "SLC4A1 SAO homozygote: LETHAL — prenatal testing if both parents carry SAO",
            "ANK1 HS: MCHC elevated PATHOGNOMONIC; EMA binding test BEST; "
            "splenectomy curative after age 6 + full vaccination",
            "HBB SCD: Casgevy (CRISPR BCL11A) FDA Dec 2023 — first FDA-approved CRISPR therapy; "
            "ACS = exchange transfusion NOT simple top-up",
            "PKLR: Paradoxical reticulocytosis post-splenectomy PATHOGNOMONIC; "
            "2,3-DPG elevated; mitapivat FDA 2022",
            "SPTA1 HPP: Heat instability at 45°C PATHOGNOMONIC; MCV 50-70 fL; "
            "alpha-LELY allele compounding severity",
        ],
        "kpis": [
            {"label": "Total Patients",    "value": str(n)},
            {"label": "Genes Covered",     "value": "8"},
            {"label": "Drug Error Rate",   "value": f"{agg['drug_error_pct']}%"},
            {"label": "Delayed Dx Rate",   "value": f"{agg['dx_delayed_pct']}%"},
            {"label": "ESRD Rate",         "value": f"{agg['esrd_pct']}%"},
            {"label": "Transplant Rate",   "value": f"{agg['transplant_pct']}%"},
            {"label": "Severe Cases",      "value": f"{agg['severity']['Severe']}%"},
            {"label": "Adherent",          "value": f"{agg['adherent_pct']}%"},
        ],
        "aggregate_clinical": {
            "esrd_pct":                agg["esrd_pct"],
            "hypertension_pct":        agg["htn_pct"],
            "transplant_rate_pct":     agg["transplant_pct"],
            "drug_error_pct":          agg["drug_error_pct"],
            "diagnosis_delayed_pct":   agg["dx_delayed_pct"],
            "surveillance_adherent_pct": agg["adherent_pct"],
            "severity_mild_pct":       agg["severity"]["Mild"],
            "severity_moderate_pct":   agg["severity"]["Moderate"],
            "severity_severe_pct":     agg["severity"]["Severe"],
        },
        "pathway_targets": {
            "HBB":    "Hb polymerisation (HbS) → voxelotor; HbF induction → hydroxyurea; BCL11A CRISPR → Casgevy",
            "HBA1":   "Globin chain imbalance → transfusion + chelation; IUT for Hb Bart's; MLPA diagnosis",
            "G6PD":   "NADPH deficiency → oxidative haemolysis; avoid triggers; gene therapy trials",
            "PKLR":   "PK-R enzyme → mitapivat (allosteric activator) FDA 2022; 2,3-DPG compensation",
            "ANK1":   "Spectrin-ankyrin-band3 defect → membrane vesiculation; splenectomy curative",
            "SPTA1":  "Spectrin alpha tetramerisation defect → membrane fragmentation; HPP heat instability",
            "SLC4A1": "Band 3 / AE1 → SAO malaria protection; dRTA1 alkali therapy; SAO homozygote lethal",
            "PIEZO1": "GOF Ca²⁺ channel → Gardos activation → xerocyte dehydration; "
                      "SPLENECTOMY CI; ektacytometry; chelation",
        },
        "disease_category_breakdown": {
            "Haemoglobinopathy (HBB/HBA1)": 25,
            "Enzymopathy (G6PD/PKLR)":       25,
            "Membranopathy (ANK1/SPTA1)":    25,
            "Band-3/Channel (SLC4A1/PIEZO1)": 25,
        },
    }


def get_breakdown() -> dict:
    genes = []
    for idx, gene_def in enumerate(HEMOGLOBINOPATHY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(gene_def, seed=seed, n=40)
        stats = _cohort_stats(cohort)
        genes.append({
            "gene":              gene_def["gene"],
            "protein":           gene_def["protein"],
            "alias":             gene_def["alias"],
            "aa":                gene_def["aa"],
            "kDa":               gene_def["kDa"],
            "locus":             gene_def["locus"],
            "omim_gene":         gene_def["omim_gene"],
            "omim_disease":      gene_def["omim_disease"],
            "gene_class":        gene_def["gene_class"],
            "inheritance": (
                "X-linked (G6PD hemizygous males; heterozygous females mosaic lyonisation)"
                if gene_def["gene"] == "G6PD"
                else "AD (most) / de novo; AR severe forms (ANK1/SLC4A1/SPTA1)"
                if gene_def["gene"] in ("ANK1", "SPTA1", "SLC4A1")
                else "AD GOF"
                if gene_def["gene"] == "PIEZO1"
                else "AR (HbSS compound het or homozygote; beta-thal major AR)"
                if gene_def["gene"] == "HBB"
                else "AR (HBA1 deletional; Hb H --/-α; Hb Bart's --/--)"
                if gene_def["gene"] == "HBA1"
                else "AR"
            ),
            "phenotype":         gene_def["phenotype"],
            "hallmark":          gene_def["hallmark"],
            "treatment_alert":   gene_def["treatment_alert"],
            "key_ddx":           gene_def["key_ddx"],
            "gfr_pattern":       gene_def["gfr_pattern"],
            "proteinuria_pattern": gene_def["proteinuria_pattern"],
            "primary_complication": gene_def["primary_complication"],
            "disease_detail":    gene_def["disease"],
            "cohort_n":          40,
            "cohort_stats":      {
                "esrd_pct":       stats["esrd_pct"],
                "htn_pct":        stats["htn_pct"],
                "drug_error_pct": stats["drug_error_pct"],
                "dx_delayed_pct": stats["dx_delayed_pct"],
                "transplant_pct": stats["transplant_pct"],
                "adherent_pct":   stats["adherent_pct"],
                "severity":       stats["severity"],
            },
        })
    return {"genes": genes, "n_genes": len(genes)}


def get_definitions() -> list:
    return [
        {
            "term": "Acute Chest Syndrome (ACS)",
            "full": "Acute Chest Syndrome — SCD #1 mortality",
            "explanation": (
                "New CXR infiltrate + ≥2 of: fever >38.5°C, cough, dyspnoea, hypoxia (O₂ sat <95%), "
                "or chest pain in SCD patient. #1 cause of mortality in SCD. "
                "Treatment: exchange transfusion (target HbS <30%) + incentive spirometry + "
                "broad-spectrum ABx (cover atypicals: ceftriaxone + azithromycin or levofloxacin) + "
                "bronchodilators. Simple top-up transfusion is insufficient for severe ACS."
            ),
        },
        {
            "term": "Casgevy (exagamglogene autotemcel)",
            "full": "First FDA-approved CRISPR therapy (Dec 2023)",
            "explanation": (
                "CRISPR-Cas9 gene editing targeting BCL11A enhancer in haematopoietic stem cells → "
                "de-represses HBG1/HBG2 → HbF reactivation (>20% HbF) → suppresses HbS polymerisation; "
                "FDA-approved Dec 8 2023 for SCD (≥12 yrs) and beta-thal major; "
                "97% VOC-free at 12 months (CLIMB-SCD-121); 93% transfusion-independent (beta-thal); "
                "requires myeloablative conditioning (busulfan); fertility counselling mandatory; "
                "manufactured from patient's own HSCs (autologous)."
            ),
        },
        {
            "term": "G6PD Testing Timing Rule",
            "full": "Test G6PD AFTER haemolytic episode, NOT during",
            "explanation": (
                "Reticulocytes (young RBCs) have higher G6PD enzyme activity than mature RBCs; "
                "during acute haemolysis, reticulocyte count rises markedly → "
                "G6PD activity assay may return FALSELY NORMAL; "
                "wait 2-3 months post-episode (or test family members) for accurate quantitative assay; "
                "qualitative fluorescent spot test also unreliable during haemolysis."
            ),
        },
        {
            "term": "MLPA (Multiplex Ligation-Dependent Probe Amplification)",
            "full": "Required for Alpha-Thalassaemia Diagnosis",
            "explanation": (
                "The common alpha-thalassaemia mutations are large DELETIONS (--SEA 17.5 kb Southeast Asian; "
                "--MED 20 kb Mediterranean; --FIL 30 kb Filipino; --THAI 35 kb Thai; -α3.7; -α4.2) that "
                "CANNOT be detected by Sanger sequencing or point mutation panels; "
                "MLPA detects copy number changes across HBA1 and HBA2; "
                "mandatory for all suspected alpha-thal and for couples at risk for Hb Bart's hydrops."
            ),
        },
        {
            "term": "HbA2 Interpretation",
            "full": "HbA2 LOW in alpha-thal; HbA2 HIGH in beta-thal — Critical DDx",
            "explanation": (
                "HbA2 (alpha2-delta2) is the key discriminator: "
                "Beta-thalassaemia trait → HbA2 ELEVATED >3.5% (delta2-chains pair with excess alpha-chains); "
                "Alpha-thalassaemia trait → HbA2 LOW or NORMAL (<2.5%); "
                "microcytosis + HbA2 LOW = alpha-thal (confirm MLPA) or iron deficiency (check ferritin); "
                "microcytosis + HbA2 ELEVATED >3.5% = beta-thal trait (Sanger/panel sufficient)."
            ),
        },
        {
            "term": "Ektacytometry (Osmotic Gradient Laser Diffractometry)",
            "full": "Gold Standard for RBC Membranopathy Diagnosis",
            "explanation": (
                "Measures RBC deformability index (EI) across an osmotic gradient; "
                "generates a characteristic profile curve; "
                "Omin (minimum EI point) and EImax (maximum deformability) identify disorder type: "
                "HS → Omin shifted LEFT with reduced EImax (dense spherocytes); "
                "DHS/PIEZO1 → Omin LEFTWARD + HIGH EImax (dehydrated + less deformable at low tonicity); "
                "HE/HPP → characteristic pattern; "
                "PATHOGNOMONIC for PIEZO1 DHS — distinguishes from HS definitively."
            ),
        },
        {
            "term": "EMA (Eosin-5-Maleimide) Binding Test",
            "full": "Best Screening Test for Hereditary Spherocytosis",
            "explanation": (
                "EMA covalently binds to Lys430 of band 3 (AE1/SLC4A1) on the RBC surface; "
                "in HS (any genetic cause — ANK1, SPTB, SLC4A1, SPTA1, EPB41) → "
                "band 3 expression reduced → EMA fluorescence DECREASED ≥16% vs normal controls; "
                "measured by flow cytometry; superior to osmotic fragility (fewer false positives/negatives); "
                "EMA test is NORMAL in DHS/PIEZO1 — important DDx from HS."
            ),
        },
        {
            "term": "Hb Bart's Hydrops Fetalis",
            "full": "4-Gene Alpha-Thalassaemia — Lethal Without IUT",
            "explanation": (
                "Complete absence of alpha-globin (--/--) → only gamma-chains form → "
                "Hb Bart's (γ4 tetramers) with extremely high O₂ affinity → "
                "no oxygen delivery to fetal tissues → hydrops fetalis (anasarca, effusions, "
                "massive placentomegaly); cardiac failure in utero; LETHAL without treatment; "
                "intrauterine transfusion (IUT) from ~20 weeks can produce liveborn infants "
                "requiring lifelong transfusions; maternal mirror syndrome (maternal hydrops) risk; "
                "most common in SE Asia (SAO + alpha-thal coinheritance)."
            ),
        },
        {
            "term": "2,3-DPG (2,3-Bisphosphoglycerate)",
            "full": "Elevated in PK Deficiency — Protective Right-Shift",
            "explanation": (
                "2,3-DPG is a glycolytic intermediate that allosterically reduces Hb-O₂ affinity "
                "(right-shifts the oxygen-haemoglobin dissociation curve → better O₂ delivery at "
                "lower PO₂); in PKLR deficiency, the metabolic block at PK causes accumulation "
                "of proximal intermediates including 2,3-DPG; "
                "result: patients tolerate lower Hb levels than expected because O₂ delivery is "
                "relatively better (right-shifted Hb curve)."
            ),
        },
        {
            "term": "Paradoxical Reticulocytosis Post-Splenectomy",
            "full": "Pathognomonic for PK Deficiency",
            "explanation": (
                "In PKLR deficiency, reticulocytes (which have mitochondria) can compensate "
                "via oxidative phosphorylation, but in the hypoxic microenvironment of the spleen, "
                "mitochondrial compensation fails → reticulocytes are preferentially destroyed in spleen; "
                "post-splenectomy: reticulocytes now survive in peripheral blood → "
                "reticulocyte count RISES (not falls) after splenectomy; "
                "this paradoxical reticulocytosis is PATHOGNOMONIC for PK deficiency."
            ),
        },
        {
            "term": "Southeast Asian Ovalocytosis (SAO)",
            "full": "SLC4A1 27bp In-Frame Deletion — Malaria Protection + Lethal Homozygote",
            "explanation": (
                "27bp in-frame deletion of codons 400-408 in SLC4A1 (band 3 / AE1) transmembrane domain → "
                "rigid ovalocytes with transverse bar; protects against Plasmodium falciparum cerebral "
                "malaria (PfEMP3 cannot bind; cytoadhesion impaired); "
                "heterozygotes clinically silent (mild/no anaemia); "
                "SAO HOMOZYGOTE (--/--) → lethal hydrops fetalis; "
                "carrier frequency 4-12% in SE Asia; "
                "PRENATAL TESTING MANDATORY if both parents are SAO carriers."
            ),
        },
        {
            "term": "Hydroxyurea",
            "full": "Ribonucleotide Reductase Inhibitor — HbF Inducer in SCD",
            "explanation": (
                "Hydroxyurea induces HbF (fetal haemoglobin) by increasing HBG1/HBG2 expression "
                "(mechanism involves nitric oxide → sGC → cGMP → HbF loci demethylation); "
                "reduces VOC frequency 50%, ACS 50%, hospitalisations, transfusion requirement, "
                "and mortality in SCD; "
                "dose: 15-35 mg/kg/day titrated to therapeutic response (MCV rise, HbF rise) "
                "while monitoring CBC (hold if ANC <2.0×10⁹/L, platelets <80×10⁹/L, Hb <4.5 g/dL); "
                "offer to all patients with SCD ≥9 months old."
            ),
        },
        {
            "term": "Mitapivat (Pyrukynd)",
            "full": "FDA 2022 — First PK-R Enzyme Activator",
            "explanation": (
                "Mitapivat (Agios Pharmaceuticals) is a small-molecule allosteric activator of "
                "pyruvate kinase R (PK-R); "
                "increases PK-R affinity for PEP → more ATP production → reduces haemolysis; "
                "ACTIVATE-PK trial: increases Hb 1.5-2 g/dL; reduces transfusion burden; "
                "FDA approved Feb 2022 for adults with PKLR deficiency; "
                "monitoring: LFTs (hepatotoxicity); CBC; "
                "do NOT stop abruptly — haemolytic rebound; taper dose; "
                "paediatric dosing trials ongoing."
            ),
        },
        {
            "term": "OPSI (Overwhelming Post-Splenectomy Infection)",
            "full": "Life-Threatening Infection Risk Post-Splenectomy",
            "explanation": (
                "Splenectomy → loss of splenic filtration of encapsulated bacteria → "
                "OPSI (fulminant sepsis within hours of onset); "
                "pathogens: S. pneumoniae (#1), N. meningitidis, H. influenzae type b, Capnocytophaga; "
                "mortality 50-70% once OPSI established; "
                "prevention: PCV13 + PPV23 + Hib + MenACWY + MenB vaccination ≥4 weeks pre-splenectomy; "
                "penicillin V prophylaxis ≥2 years (lifelong in high-risk/children); "
                "fever >38.5°C → immediate oral antibiotics (amoxicillin) + emergency evaluation; "
                "medical alert card/bracelet mandatory."
            ),
        },
        {
            "term": "HPP Heat Instability Test",
            "full": "Pathognomonic for Hereditary Pyropoikilocytosis",
            "explanation": (
                "Hereditary pyropoikilocytosis (HPP) RBCs fragment at 45°C (vs 49°C for normal RBCs); "
                "due to spectrin alpha-chain structural instability (SPTA1 mutations + alpha-LELY allele); "
                "performed by incubating blood at 45°C for 15 minutes and examining blood film; "
                "fragmentation at 45°C is PATHOGNOMONIC for HPP (not seen in HE, HS, or normal); "
                "MCV 50-70 fL (severely low) is characteristic of HPP."
            ),
        },
        {
            "term": "Alpha-LELY Allele",
            "full": "Low Expression Lyon (alphaLELY) — Severity Modifier in SPTA1",
            "explanation": (
                "c.6531-12C>T polymorphism in SPTA1 creates a cryptic splice site → "
                "skipping of exon 46 → ~35% reduced alpha-spectrin expression; "
                "ALONE: clinically silent (present in 15-30% of African population); "
                "IN TRANS with a pathogenic SPTA1 variant: compound heterozygote → "
                "insufficient normal alpha-spectrin → HPP (not just HE); "
                "parents of HPP proband: one parent has HE + pathogenic SPTA1, "
                "other parent has alpha-LELY — critical genetic counselling point."
            ),
        },
    ]


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(f"Atlas: {ov['atlas_name']}; Genes: {ov['n_genes']}; Patients: {ov['n_patients']}")
    print(f"Drug alerts: {len(ov['drug_alerts'])}; Critical rules: {len(ov['critical_rules'])}")
    bd = get_breakdown()
    print(f"\n=== BREAKDOWN ===")
    for g in bd["genes"]:
        cs = g["cohort_stats"]
        print(f"  {g['gene']:8s} | ESRD {cs['esrd_pct']:5.1f}% | DrugErr {cs['drug_error_pct']:5.1f}%"
              f" | DxDelay {cs['dx_delayed_pct']:5.1f}% | Severe {cs['severity']['Severe']:5.1f}%")
    defs = get_definitions()
    print(f"\n=== DEFINITIONS === ({len(defs)} terms)")
    for d in defs:
        print(f"  {d['term']}")
