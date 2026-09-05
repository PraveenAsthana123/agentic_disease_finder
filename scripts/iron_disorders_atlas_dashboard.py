#!/usr/bin/env python3
"""Iron-Disorders-Atlas — Complete 8-Gene Hereditary Iron Metabolism Disorders Atlas
HFE     (HFE protein; 348 aa; 6p21.3; HH Type 1; AR; most common hereditary hemochromatosis; C282Y
         homozygous ~1 in 200 Northern European; p.C282Y/p.H63D compound heterozygote — milder;
         transferrin saturation >45% + ferritin elevated; phlebotomy curative; do NOT use ferritin alone) ·
HJV     (Hemojuvelin / RGMc; 426 aa; 1q21.1; HH Type 2A; AR; juvenile hemochromatosis;
         severe early-onset cardiomyopathy and hypogonadism by age 30; hepcidin absent; phlebotomy + IV desferrioxamine) ·
HAMP    (Hepcidin antimicrobial peptide; 84 aa; 19q13.12; HH Type 2B; AR; hepcidin structural gene;
         identical juvenile phenotype to HJV; heart failure + hypogonadism; same urgent phlebotomy) ·
TFR2    (Transferrin receptor 2; 801 aa; 7q22.1; HH Type 3; AR; HFE-like adult presentation;
         HFE gene normal — TFR2 sequencing mandatory; serum hepcidin low; phlebotomy effective) ·
SLC40A1 (Ferroportin / SLC40A1; 571 aa; 2q32.2; HH Type 4; AD; two subtypes — Type 4A = ferroportin
         disease (macrophage iron, normal/low transferrin sat, venesection poorly tolerated) vs Type 4B
         (classical HH phenotype, venesection effective); differentiate before therapy) ·
TMPRSS6 (Matriptase-2; 811 aa; 22q12.3; IRIDA Iron-Refractory Iron Deficiency Anemia; AR;
         cannot suppress hepcidin in iron deficiency → oral iron completely fails; IV iron (ferric
         carboxymaltose) effective; oral iron + hepcidin test = diagnostic key) ·
CP      (Ceruloplasmin; 1065 aa; 3q25.1; Aceruloplasminemia; AR; progressive neurodegeneration +
         diabetes mellitus + retinal degeneration; serum ceruloplasmin absent; high ferritin with
         initially normal transferrin saturation; IV desferrioxamine; fresh frozen plasma short-term) ·
FTL     (Ferritin light chain; 175 aa; 19q13.33; Hyperferritinemia-Cataract Syndrome HHCS; AD;
         IRE mutation in 5-prime UTR of FTL → no iron overload but bilateral dense cataracts;
         serum ferritin 500-10000 but NO iron accumulation in organs; do NOT venesect — worsens anemia)
320-patient aggregate cohort (8 x 40, seeds 1286-1293)
"""

import random

SEED_BASE = 1286

IRON_GENES = [
    # ── HFE — Hereditary Hemochromatosis Type 1 ──────────────────────────────
    {
        "gene": "HFE",
        "protein": "HFE Protein (HFE)",
        "alias": (
            "HFE; OMIM gene 613609; HH1 #235200; 6p21.3; 348 aa; ~37 kDa; AR (biallelic); "
            "MHC class I-like protein; regulates transferrin receptor 1 (TFR1) iron sensing; "
            "p.C282Y (c.845G>A): disrupts disulfide bond → misfolded HFE cannot bind beta-2-microglobulin → "
            "absent surface expression → unregulated TFR1 iron uptake; p.H63D milder — retained surface expression; "
            "C282Y homozygous ~1:200 Northern European; most common AR disorder in Europeans; "
            "penetrance: 50% C282Y homozygous females and 25% males develop clinical disease; "
            "transferrin saturation >45% + ferritin >200 (F) or >300 (M) = diagnostic threshold; "
            "liver biopsy if ferritin >1000 or elevated transaminases — MRI T2* alternative"
        ),
        "aa": "348 aa",
        "kDa": "~37 kDa",
        "locus": "6p21.3",
        "omim_gene": 613609,
        "omim_disease": 235200,
        "inheritance": "AR (biallelic); C282Y homozygous most common; C282Y/H63D compound het milder; H63D homozygous rarely causes iron overload",
        "gene_class": (
            "HFE encodes an MHC class I-like protein that forms a heterodimer with beta-2-microglobulin (B2M) "
            "and interacts with transferrin receptor 1 (TFR1) at the cell surface of hepatocytes and enterocytes. "
            "HFE modulates TFR1 affinity for transferrin and regulates hepcidin expression via the BMP-SMAD pathway. "
            "p.C282Y disrupts the critical disulfide bond in the alpha-3 domain → misfolded protein trapped in "
            "endoplasmic reticulum → absent HFE-TFR1 interaction → unregulated hepatic iron uptake and failure "
            "to stimulate hepcidin → progressive iron accumulation in parenchymal cells (hepatocytes, cardiomyocytes, "
            "pancreatic beta cells, pituitary, joints). "
            "KEY DIAGNOSTIC DISTINCTION from secondary iron overload: ferritin elevated + transferrin saturation "
            ">45% (early) or >70% (overt); serum hepcidin low for degree of iron loading (opposite of anemia of "
            "chronic disease where hepcidin is high). "
            "GENETIC TESTING: HFE C282Y/H63D genotyping is diagnostic for HH1; if HFE genotype normal in HH "
            "phenotype → sequence HJV, HAMP, TFR2, SLC40A1."
        ),
        "phenotype": (
            "Classical HH Type 1: hepatomegaly → cirrhosis → HCC (7-200x risk if cirrhotic); "
            "cardiomyopathy (dilated or restrictive); diabetes mellitus (bronze diabetes = pancreatic iron); "
            "hypogonadotropic hypogonadism (pituitary iron — low LH/FSH/testosterone); "
            "arthropathy (2nd and 3rd MCP joints PATHOGNOMONIC — chondrocalcinosis, pseudogout); "
            "skin hyperpigmentation (bronzing from melanin + iron); "
            "fatigue and lethargy (most common presenting symptom); "
            "presentation typically age 40-60 in men (menstruation protective in women until menopause); "
            "women present later, less severely, less frequently clinical disease"
        ),
        "hallmark": (
            "ARTHROPATHY of 2nd/3rd MCP joints is PATHOGNOMONIC for hereditary hemochromatosis — "
            "other causes of iron overload rarely cause this specific joint pattern. "
            "Transferrin saturation >45% is the earliest biochemical marker (rises before ferritin). "
            "Ferritin alone is insufficient — it is an acute-phase reactant (elevated in inflammation, "
            "metabolic syndrome, liver disease, alcohol). Use BOTH transferrin saturation AND ferritin."
        ),
        "treatment_alerts": [
            "PHLEBOTOMY is curative if pre-cirrhotic: weekly 500 mL (250 mg iron) until ferritin 50-100 μg/L; then maintenance every 2-4 months.",
            "DO NOT use ferritin alone for diagnosis — transferrin saturation >45% is MANDATORY first marker.",
            "IRON CHELATION (desferrioxamine, deferasirox) rarely needed in HFE HH — phlebotomy always preferred.",
            "HCC SURVEILLANCE: 6-monthly liver ultrasound if cirrhosis established — HCC risk persists even after iron depletion.",
            "ARTHROPATHY does NOT improve with phlebotomy — iron already deposited in cartilage; manage symptomatically.",
            "FAMILY SCREENING: all first-degree relatives need HFE genotyping + transferrin saturation; C282Y homozygous relatives need phlebotomy.",
            "ALCOHOL EXACERBATES iron loading and liver disease in HFE HH — advise abstinence.",
        ],
        "key_ddx": (
            "HJV/HAMP HH Type 2 (juvenile onset, severe cardiomyopathy, age <30); "
            "TFR2 HH Type 3 (HFE normal, adult onset, check TFR2 sequence); "
            "SLC40A1 HH Type 4 (AD, macrophage iron in Type 4A vs classical in 4B); "
            "Secondary iron overload: transfusion dependence, dyserythropoiesis, alcohol — "
            "key DDx: HFE genotype, transferrin saturation pattern, family history; "
            "Metabolic hyperferritinemia: ferritin elevated but transferrin saturation NORMAL."
        ),
    },
    # ── HJV — Juvenile Hemochromatosis Type 2A ───────────────────────────────
    {
        "gene": "HJV",
        "protein": "Hemojuvelin / Repulsive Guidance Molecule C (HJV/RGMc)",
        "alias": (
            "HJV (OMIM gene 608374); HH2A #602390; 1q21.1; 426 aa; ~50 kDa; AR (biallelic); "
            "BMP co-receptor in liver → stimulates hepcidin via SMAD1/5/8; HJV null → absent BMP-hepcidin axis → "
            "most severe hepcidin deficiency of all HH types → massive iron loading; "
            "onset before age 30 (classic juvenile); cardiomyopathy leading cause of death; "
            "p.G320V commonest pathogenic variant; Mediterranean and Portuguese founder variants"
        ),
        "aa": "426 aa",
        "kDa": "~50 kDa",
        "locus": "1q21.1",
        "omim_gene": 608374,
        "omim_disease": 602390,
        "inheritance": "AR (biallelic); de novo very rare; consanguinity increases risk; both sexes equally affected",
        "gene_class": (
            "HJV (hemojuvelin, also called RGMc) is a GPI-anchored cell-surface BMP co-receptor expressed "
            "predominantly in hepatocytes and skeletal muscle. "
            "MECHANISM: HJV enhances BMP6/BMP2 binding to BMPR1/BMPR2 → SMAD1/5/8 phosphorylation → "
            "hepcidin (HAMP) transcription. This is the dominant pathway regulating hepcidin in response to "
            "iron stores. HJV LOF → absent SMAD-hepcidin signalling → serum hepcidin undetectable → "
            "constitutive ferroportin expression on enterocytes and macrophages → unconstrained dietary iron "
            "absorption and macrophage iron release → rapid parenchymal iron accumulation. "
            "SEVERITY: HJV HH is the most iron-loaded form of all hereditary hemochromatoses — "
            "serum ferritin may exceed 10,000 μg/L; liver iron concentration 10-50x normal; "
            "cardiomyopathy appears by age 20-30 (vs HFE HH where cardiac disease is rare pre-menopause). "
            "HEPCIDIN MEASUREMENT: serum hepcidin undetectable in HJV HH even with massive iron overload — "
            "distinguishes from secondary iron overload where hepcidin is elevated."
        ),
        "phenotype": (
            "Juvenile hemochromatosis: massive iron loading presenting age 10-30; "
            "cardiomyopathy (dilated/restrictive) — arrhythmia, heart failure, sudden cardiac death; "
            "hypogonadotropic hypogonadism (pubertal delay, amenorrhea in females, impotence in males); "
            "hepatic fibrosis/cirrhosis (less dominant than in HFE HH at this age); "
            "skin bronzing; diabetes mellitus; "
            "cardiac involvement distinguishes juvenile from adult-onset HH forms"
        ),
        "hallmark": (
            "CARDIOMYOPATHY before age 30 in iron overload = Juvenile HH until proven otherwise. "
            "Serum hepcidin UNDETECTABLE despite massive iron overload. "
            "Both sexes affected equally (no menstrual protection). "
            "Start phlebotomy URGENTLY — may need 2-3x weekly initially + IV desferrioxamine if "
            "cardiac function compromised (phlebotomy in severe cardiomyopathy can cause hypotension)."
        ),
        "treatment_alerts": [
            "URGENT PHLEBOTOMY: 2-3x weekly initially when safe; target ferritin <50 μg/L.",
            "IV DESFERRIOXAMINE: if cardiac compromise precludes phlebotomy — continuous SC infusion or IV.",
            "CARDIAC MONITORING: ECHO every 6 months until iron depleted; arrhythmia monitoring.",
            "ENDOCRINE: testosterone replacement (hypogonadism rarely reverses with iron depletion); GH if growth affected.",
            "HEPCIDIN AGONIST THERAPY (experimental): mini-hepcidin or TMPRSS6 inhibitors — trials ongoing.",
            "FAMILY SCREENING: siblings at 25% risk; HJV sequencing all siblings.",
        ],
        "key_ddx": (
            "HAMP HH Type 2B (identical phenotype — hepcidin structural gene; same management); "
            "HFE HH Type 1 (adult onset, less severe, C282Y homozygous); "
            "Cardiac sarcoidosis (no iron overload biochemistry); "
            "Dilated cardiomyopathy (ferritin normal); "
            "Secondary iron overload from thalassaemia (hemoglobin electrophoresis distinguishes)."
        ),
    },
    # ── HAMP — Hepcidin / Hereditary Hemochromatosis Type 2B ─────────────────
    {
        "gene": "HAMP",
        "protein": "Hepcidin Antimicrobial Peptide (HAMP)",
        "alias": (
            "HAMP (OMIM gene 606464); HH2B #602390 (same OMIM as HJV — allelic disease); "
            "19q13.12; 84 aa (25 aa bioactive peptide cleaved from 84 aa prepropeptide); ~9 kDa mature; "
            "AR (biallelic); hepcidin is the master iron hormone — hepatic peptide that binds ferroportin → "
            "ferroportin ubiquitination → endocytosis → degradation → stops intestinal iron absorption and "
            "macrophage iron release; HAMP null → constitutive ferroportin activity → juvenile HH identical to HJV HH"
        ),
        "aa": "84 aa",
        "kDa": "~9 kDa (mature bioactive peptide)",
        "locus": "19q13.12",
        "omim_gene": 606464,
        "omim_disease": 602390,
        "inheritance": "AR (biallelic); rare; reported in Italian, Portuguese, North African, and Asian families; consanguinity common",
        "gene_class": (
            "HAMP encodes hepcidin, the central regulator of systemic iron homeostasis. "
            "Hepcidin is a 25-amino-acid beta-defensin-like peptide secreted by hepatocytes in response to "
            "high iron stores, BMP6 signalling (via HJV-BMPR-SMAD axis), inflammation (IL-6→STAT3), and infection. "
            "MECHANISM OF ACTION: hepcidin binds ferroportin (SLC40A1) on the basolateral surface of "
            "duodenal enterocytes and on macrophage surfaces → ferroportin phosphorylation → "
            "ubiquitination → lysosomal degradation → trapped intracellular iron → "
            "reduced dietary iron absorption and reduced macrophage iron recycling. "
            "HAMP biallelic LOF → no functional hepcidin → ferroportin constitutively active → "
            "iron absorption unrestricted → same severe juvenile hemochromatosis as HJV HH. "
            "DIAGNOSTIC VALUE: serum hepcidin-25 (by mass spectrometry) is commercially available; "
            "in HH Types 1-3, hepcidin is inappropriately LOW for iron load; in Type 4A, hepcidin resistance. "
            "INFLAMMATORY ANEMIA: conversely, IL-6-driven hepcidin excess → ferroportin suppression → "
            "iron trapping → hypoferremia → anemia of chronic disease (high ferritin, low transferrin sat)."
        ),
        "phenotype": (
            "Juvenile hemochromatosis identical to HJV HH Type 2A: "
            "onset before age 30; cardiomyopathy; hypogonadotropic hypogonadism; "
            "hepatic iron loading with rapid progression to fibrosis; "
            "skin bronzing; diabetes mellitus; "
            "serum hepcidin undetectable; transferrin saturation >80%; ferritin >3000 μg/L typically"
        ),
        "hallmark": (
            "SERUM HEPCIDIN UNDETECTABLE in biallelic HAMP LOF — this is the direct assay of the "
            "missing gene product. Hepcidin measurement distinguishes HH2A (HJV — hepcidin absent "
            "because upstream BMP signalling fails) from HH2B (HAMP — hepcidin absent because the "
            "structural gene is mutant). Management identical; genetic differentiation matters for family cascade."
        ),
        "treatment_alerts": [
            "SAME TREATMENT as HJV HH: urgent phlebotomy 2-3x weekly; IV desferrioxamine if cardiac compromise.",
            "CARDIAC ECHO mandatory at diagnosis — early severe cardiomyopathy expected.",
            "GENETIC DISTINCTION from HJV HH requires HAMP + HJV sequencing — same phenotype, different gene.",
            "HEPCIDIN REPLACEMENT THERAPY (investigational): synthetic hepcidin analogues or mini-hepcidin peptides.",
            "REPRODUCTIVE COUNSELLING: hypogonadism rarely reverses; early testosterone/estrogen replacement; fertility options.",
        ],
        "key_ddx": (
            "HJV HH Type 2A (identical phenotype — upstream BMP co-receptor; HJV sequencing); "
            "HFE HH Type 1 (adult onset, C282Y homozygous, less severe); "
            "TMPRSS6-IRIDA (opposite: hepcidin inappropriately high, iron deficient not overloaded); "
            "Anemia of chronic inflammation (hepcidin high, transferrin sat low — distinguished biochemically)."
        ),
    },
    # ── TFR2 — Hereditary Hemochromatosis Type 3 ─────────────────────────────
    {
        "gene": "TFR2",
        "protein": "Transferrin Receptor 2 (TFR2)",
        "alias": (
            "TFR2 (OMIM gene 604720); HH3 #604250; 7q22.1; 801 aa; ~90 kDa; AR (biallelic); "
            "hepatic iron sensor that works alongside HFE; TFR2 binds diferric transferrin → signals "
            "hepcidin upregulation; TFR2 LOF → impaired hepatic iron sensing → inappropriately low hepcidin → "
            "progressive iron accumulation; adult presentation similar to HFE HH but HFE genotype NORMAL; "
            "Mediterranean and Middle Eastern families most reported; p.Y250X founder variant in Campania Italy"
        ),
        "aa": "801 aa",
        "kDa": "~90 kDa",
        "locus": "7q22.1",
        "omim_gene": 604720,
        "omim_disease": 604250,
        "inheritance": "AR (biallelic); rarer than HFE HH; Mediterranean, Middle Eastern, Asian reports; p.Y250X Italian founder",
        "gene_class": (
            "TFR2 encodes a type II transmembrane glycoprotein expressed predominantly in hepatocytes "
            "and erythroid precursors. Unlike TFR1, TFR2 has low affinity for transferrin and does not "
            "mediate cellular iron uptake. "
            "MECHANISM: TFR2 acts as a hepatic iron sensor — it binds diferric transferrin in proportion "
            "to serum iron saturation and signals (via interaction with HFE and activin receptor-like kinases) "
            "to stimulate BMP-SMAD pathway hepcidin transcription. "
            "TFR2 LOF → intact BMP pathway but absent transferrin-iron sensing → hepcidin remains low "
            "despite rising transferrin saturation → parenchymal iron accumulation. "
            "GENETIC PANEL: Any patient with HH phenotype (transferrin sat >45%, ferritin elevated, parenchymal "
            "iron on MRI) but NORMAL HFE C282Y/H63D genotype → MUST sequence TFR2 (and HJV, HAMP, SLC40A1). "
            "ERYTHROPOIESIS: TFR2 is also expressed in erythroid progenitors where it regulates EPO receptor "
            "signalling; TFR2 variants associate with elevated red blood cell parameters."
        ),
        "phenotype": (
            "Adult-onset hemochromatosis phenotypically similar to HFE HH Type 1: "
            "hepatomegaly → fibrosis/cirrhosis → HCC risk; "
            "cardiomyopathy (less severe than juvenile HH); "
            "diabetes mellitus; arthropathy (2nd/3rd MCP joints); "
            "skin bronzing; hypogonadism; "
            "onset typically 30-50 years; HFE genotype NORMAL — key diagnostic clue"
        ),
        "hallmark": (
            "HH PHENOTYPE + NORMAL HFE GENOTYPE → TFR2 sequencing MANDATORY. "
            "TFR2 HH is clinically indistinguishable from HFE HH Type 1 but HFE C282Y/H63D both negative. "
            "Serum hepcidin appropriately low for iron load. "
            "Response to phlebotomy identical to HFE HH — excellent if pre-cirrhotic."
        ),
        "treatment_alerts": [
            "PHLEBOTOMY: same protocol as HFE HH — weekly 500 mL until ferritin <50 μg/L; then maintenance.",
            "HFE GENOTYPE NORMAL: Do NOT assume no hereditary cause — expand panel to TFR2, HJV, HAMP, SLC40A1.",
            "HCC SURVEILLANCE: same as HFE HH if cirrhosis established.",
            "IRON MRI (T2*): non-invasive quantification of hepatic and cardiac iron loading preferred over biopsy.",
            "FAMILY SCREENING: first-degree relatives need TFR2 sequencing + iron studies.",
        ],
        "key_ddx": (
            "HFE HH Type 1 (C282Y homozygous — most common; TFR2 normal); "
            "SLC40A1 HH Type 4 (AD inheritance distinguishes — family history of AD transmission); "
            "HJV/HAMP HH Type 2 (juvenile onset age <30, more severe, hepcidin lower); "
            "Alcoholic liver disease with secondary iron (transferrin sat pattern, liver biopsy); "
            "Non-alcoholic fatty liver disease with hyperferritinemia (transferrin sat typically normal)."
        ),
    },
    # ── SLC40A1 — Ferroportin Disease / HH Type 4 ────────────────────────────
    {
        "gene": "SLC40A1",
        "protein": "Ferroportin (SLC40A1 / FPN1 / IREG1)",
        "alias": (
            "SLC40A1 (OMIM gene 604653); HH4 #606069; 2q32.2; 571 aa; ~62 kDa; AD (dominant); "
            "ferroportin is the ONLY known cellular iron exporter; expressed on enterocytes, macrophages, hepatocytes; "
            "TYPE 4A (ferroportin disease): LOF variants — reduced cell-surface ferroportin → macrophage iron trapping → "
            "normal/low transferrin saturation, elevated ferritin, poor venesection tolerance (drops hemoglobin); "
            "TYPE 4B (classical HH phenotype): GOF variants resist hepcidin binding → constitutive iron export → "
            "classical parenchymal HH; venesection effective; DISTINGUISH TYPE BEFORE TREATMENT"
        ),
        "aa": "571 aa",
        "kDa": "~62 kDa",
        "locus": "2q32.2",
        "omim_gene": 604653,
        "omim_disease": 606069,
        "inheritance": "AD (dominant); Type 4A = LOF haploinsufficiency or dominant-negative; Type 4B = GOF hepcidin-resistance; sporadic de novo possible",
        "gene_class": (
            "SLC40A1 encodes ferroportin, the sole known mammalian cellular iron export protein, expressed "
            "on the basolateral surface of duodenal enterocytes, liver Kupffer cells, splenic macrophages, "
            "and placental syncytiotrophoblasts. "
            "MECHANISM: ferroportin exports Fe2+ out of cells in coordination with membrane-bound hephaestin "
            "(intestine) or ceruloplasmin (liver/macrophages) which oxidises Fe2+ to Fe3+ for transferrin binding. "
            "Hepcidin binds ferroportin → internalisation → ubiquitination → degradation → reduced iron export. "
            "TYPE 4A (ferroportin disease, LOF): haploinsufficiency or dominant-negative → "
            "reduced ferroportin surface expression → iron trapped in macrophages/Kupffer cells → "
            "elevated ferritin (macrophage origin) but NORMAL or LOW transferrin saturation → "
            "iron-deficient erythropoiesis can coexist → venesection triggers anemia prematurely (use erythrocytapheresis). "
            "TYPE 4B (hepcidin-resistant GOF): ferroportin variants at hepcidin-binding surface (Y64N, N144H, D157G) → "
            "ferroportin cannot bind hepcidin → constitutive iron export → classical parenchymal HH → "
            "elevated transferrin saturation AND ferritin → venesection as effective as HFE HH."
        ),
        "phenotype": (
            "TYPE 4A: macrophage iron overload; elevated ferritin with normal/low transferrin saturation; "
            "splenomegaly (Kupffer cell iron); mild anemia; hepatic iron in Kupffer cells (sinusoidal) not hepatocytes; "
            "cardiovascular and endocrine complications less common than classical HH; "
            "TYPE 4B: parenchymal hemochromatosis identical to HFE HH — liver fibrosis/cirrhosis, "
            "cardiomyopathy, diabetes, arthropathy; elevated transferrin saturation AND ferritin; "
            "AD inheritance (father-to-son transmission distinguishes from X-linked)"
        ),
        "hallmark": (
            "AD INHERITANCE pattern (unlike AR HFE, HJV, HAMP, TFR2) + iron overload = SLC40A1 until proven otherwise. "
            "TYPE 4A hallmark: NORMAL or LOW transferrin saturation with HIGH ferritin — "
            "opposite of classical HH (high transferrin sat). "
            "MACROPHAGE PATTERN on liver biopsy (Perls stain Kupffer cells > hepatocytes) = Type 4A. "
            "VENESECTION CAUTION in Type 4A — monitor Hb closely; consider erythrocytapheresis."
        ),
        "treatment_alerts": [
            "DISTINGUISH TYPE 4A vs 4B BEFORE TREATMENT — transferrin saturation and liver biopsy pattern.",
            "TYPE 4A: erythrocytapheresis preferred over simple phlebotomy (removes iron without depleting red cells); low-dose venesection with Hb monitoring.",
            "TYPE 4B: phlebotomy as effective as HFE HH — weekly 500 mL; target ferritin <50.",
            "IRON CHELATION (deferasirox): effective in both subtypes if venesection poorly tolerated.",
            "AD FAMILY SCREENING: 50% of offspring at risk; offer genetic testing to all first-degree relatives.",
            "HEPCIDIN ASSAY: Type 4A — hepcidin may be elevated (body response to apparent iron-replete signal); Type 4B — hepcidin fails to suppress ferroportin.",
        ],
        "key_ddx": (
            "HFE HH Type 1 (AR, C282Y homozygous, both sexes similar penetrance, high transferrin sat always); "
            "Secondary hyperferritinemia (metabolic syndrome, alcohol, liver disease — transferrin sat usually normal); "
            "Anemia of chronic disease (low transferrin sat but also low ferritin relatively, or high with "
            "low reticulocyte count — distinguish from Type 4A which has high ferritin, normal Hb initially)."
        ),
    },
    # ── TMPRSS6 — IRIDA (Iron-Refractory Iron Deficiency Anemia) ─────────────
    {
        "gene": "TMPRSS6",
        "protein": "Matriptase-2 (TMPRSS6)",
        "alias": (
            "TMPRSS6 (OMIM gene 609862); IRIDA #206200; 22q12.3; 811 aa; ~90 kDa; AR (biallelic); "
            "type II transmembrane serine protease on hepatocyte surface; cleaves HJV ectodomain → reduces "
            "BMP-SMAD-hepcidin signalling when iron is deficient; TMPRSS6 LOF → HJV intact → hepcidin "
            "constitutively elevated even in iron deficiency → ferroportin degraded → iron trapped → "
            "oral iron FAILS to correct anemia; IV iron effective (bypasses ferroportin block at intestine); "
            "microcytic hypochromic anemia since infancy; serum hepcidin elevated despite low iron"
        ),
        "aa": "811 aa",
        "kDa": "~90 kDa",
        "locus": "22q12.3",
        "omim_gene": 609862,
        "omim_disease": 206200,
        "inheritance": "AR (biallelic); consanguinity increases risk; heterozygous carriers have mildly elevated hepcidin and borderline iron indices",
        "gene_class": (
            "TMPRSS6 encodes matriptase-2, a hepatocyte surface serine protease that negatively regulates "
            "the BMP-hepcidin axis specifically during iron deficiency. "
            "MECHANISM: in iron deficiency, TMPRSS6 cleaves membrane-bound HJV (hemojuvelin) → "
            "loss of HJV BMP co-receptor function → reduced SMAD1/5/8 signalling → reduced hepcidin transcription → "
            "ferroportin expression preserved → iron can be mobilised from intestine and macrophages. "
            "TMPRSS6 LOF → HJV permanently active → hepcidin constitutively elevated even in iron deficiency → "
            "ferroportin on enterocytes and macrophages degraded → oral iron cannot be absorbed → "
            "iron trapped in macrophages → refractory iron deficiency despite adequate oral iron supplementation. "
            "DIAGNOSTIC CLUE: serum hepcidin ELEVATED (or inappropriately normal/high) despite microcytic "
            "anemia and low serum ferritin — this is the opposite of all other causes of iron deficiency where "
            "hepcidin should be suppressed. "
            "TREATMENT ELEGANCE: IV iron (ferric carboxymaltose, iron sucrose) bypasses intestinal absorption → "
            "iron goes directly to bloodstream → erythropoiesis can proceed; serum iron corrects rapidly."
        ),
        "phenotype": (
            "IRIDA since infancy/childhood: microcytic hypochromic anemia (Hb 7-10 g/dL typically); "
            "low MCV, low MCH, low serum iron, low ferritin, high TIBC — typical iron deficiency; "
            "ORAL IRON FAILS completely or partially; "
            "serum hepcidin elevated (elevated or inappropriately normal for degree of iron deficiency); "
            "growth delay; fatigue; pallor; splenomegaly if chronic; "
            "FAMILY HISTORY: iron deficiency anemia refractory to oral iron in siblings"
        ),
        "hallmark": (
            "ORAL IRON COMPLETELY FAILS + ELEVATED HEPCIDIN in iron deficiency anemia = IRIDA (TMPRSS6) until excluded. "
            "Normal iron deficiency anemia: hepcidin suppressed (<10 ng/mL); "
            "IRIDA: hepcidin normal or elevated (>20 ng/mL) despite low iron — PATHOGNOMONIC inverse relationship. "
            "IV iron test: 500 mg IV iron → Hb rises 2+ g/dL within 2-4 weeks → confirms diagnosis."
        ),
        "treatment_alerts": [
            "ORAL IRON IS FUTILE: do not persist with oral iron beyond 4-6 weeks of failed trial.",
            "IV IRON (ferric carboxymaltose, iron sucrose, ferumoxytol) is the treatment of choice — effective.",
            "HEPCIDIN TEST: serum hepcidin (mass spectrometry) elevated/normal in iron deficiency = TMPRSS6 diagnostic.",
            "ANTI-TMPRSS6 siRNA (Nedosiran/investigational): reduces TMPRSS6 → reduces hepcidin → improves iron absorption — trials ongoing.",
            "MONITOR HEMOGLOBIN: IV iron every 3-6 months; Hb target 11-12 g/dL (partial correction expected, not full).",
            "GENETIC PANEL: all siblings of IRIDA proband should be screened; 25% risk if both parents carriers.",
        ],
        "key_ddx": (
            "Dietary iron deficiency anemia (hepcidin suppressed, responds to oral iron); "
            "Thalassaemia trait (microcytic, low MCV, normal ferritin, Hb electrophoresis abnormal); "
            "Anemia of chronic disease (elevated hepcidin but in setting of inflammation, ferritin elevated); "
            "Celiac disease iron malabsorption (responds to IV iron, anti-tTG Ab positive, biopsy confirms); "
            "Helicobacter pylori-associated iron deficiency (responds to eradication + oral iron)."
        ),
    },
    # ── CP — Aceruloplasminemia ───────────────────────────────────────────────
    {
        "gene": "CP",
        "protein": "Ceruloplasmin (CP)",
        "alias": (
            "CP (OMIM gene 117700); Aceruloplasminemia #604290; 3q25.1; 1065 aa; ~132 kDa; AR (biallelic); "
            "ferroxidase enzyme in plasma; converts Fe2+ to Fe3+ for transferrin loading; "
            "CP null → iron cannot be exported from cells (no ferroxidase) → iron accumulates in neurons, "
            "glia, pancreatic beta cells, liver, retina → progressive neurodegeneration (extrapyramidal + cerebellar) "
            "+ diabetes mellitus + retinal degeneration — TRIPLE TRIAD pathognomonic; "
            "serum ceruloplasmin absent (not just low); high ferritin with initially NORMAL transferrin saturation"
        ),
        "aa": "1065 aa",
        "kDa": "~132 kDa",
        "locus": "3q25.1",
        "omim_gene": 117700,
        "omim_disease": 604290,
        "inheritance": "AR (biallelic); rare; worldwide reports; heterozygous carriers have mildly low ceruloplasmin but no clinical disease",
        "gene_class": (
            "CP encodes ceruloplasmin, the principal copper-containing plasma ferroxidase. "
            "MECHANISM: ceruloplasmin is secreted by hepatocytes and is the main ferroxidase in plasma; "
            "it oxidises ferrous iron (Fe2+) to ferric iron (Fe3+) which is then loaded onto transferrin. "
            "Without ceruloplasmin ferroxidase activity, ferrous iron cannot be exported from cells: "
            "ferroportin exports Fe2+ but without oxidation to Fe3+, Fe2+ cannot bind transferrin → "
            "export is blocked → iron accumulates intracellularly. "
            "NEURONAL IRON: neurons cannot export iron without ceruloplasmin (GPI-anchored ceruloplasmin "
            "on astrocytes serves this function) → brain iron accumulation → Fenton reaction → "
            "hydroxyl radical generation → oxidative neuronal death → progressive neurodegeneration. "
            "PANCREATIC: beta-cell iron accumulation → insulin secretory failure → diabetes mellitus. "
            "RETINAL: RPE and photoreceptor iron → retinal degeneration. "
            "SERUM MARKERS: ferritin markedly elevated (800-10,000 μg/L); ceruloplasmin ABSENT (not measurable); "
            "transferrin saturation initially NORMAL or low — distinguishes from classical HH where "
            "transferrin saturation is HIGH; serum iron LOW or normal."
        ),
        "phenotype": (
            "TRIPLE TRIAD of aceruloplasminemia (usually manifest age 30-60): "
            "(1) NEURODEGENERATION: extrapyramidal (dystonia, chorea, blepharospasm) + cerebellar (ataxia); "
            "(2) DIABETES MELLITUS: insulin-deficient type; precedes neurodegeneration by years; "
            "(3) RETINAL DEGENERATION: pigmentary retinopathy → visual loss; "
            "MRI brain: hypointensity in basal ganglia, thalamus, dentate nucleus (T2* iron signal); "
            "mild microcytic anemia (serum iron low despite high ferritin); "
            "serum ceruloplasmin ABSENT on immunological assay"
        ),
        "hallmark": (
            "ABSENT SERUM CERULOPLASMIN + HIGH FERRITIN + NORMAL/LOW TRANSFERRIN SATURATION + "
            "PROGRESSIVE NEURODEGENERATION + DIABETES = Aceruloplasminemia PATHOGNOMONIC. "
            "MRI brain T2*: basal ganglia and cerebellar hypointensity = iron-laden neurons — "
            "distinguishes from other movement disorders. "
            "KEY DDx POINT: transferrin saturation is NORMAL or LOW (unlike classical HH where it is HIGH) — "
            "iron is trapped inside cells, not in circulation."
        ),
        "treatment_alerts": [
            "IV DESFERRIOXAMINE: most evidence; continuous SC or IV; reduces brain iron on MRI; slows neurological decline.",
            "DEFERIPRONE: crosses blood-brain barrier — promising for neurological iron; used in NBIA disorders.",
            "FRESH FROZEN PLASMA (FFP): short-term ceruloplasmin replacement; temporary ferroxidase activity; used as bridge.",
            "RECOMBINANT HUMAN CERULOPLASMIN: investigational; potential disease-modifying therapy.",
            "DIABETES MANAGEMENT: insulin-dependent; standard T1DM protocols.",
            "RETINAL MONITORING: ophthalmology annually; retinal degeneration does NOT reverse with chelation.",
            "NO BENEFIT from phlebotomy: serum iron is already low; venesection will worsen anemia.",
        ],
        "key_ddx": (
            "Wilson disease (low ceruloplasmin but copper disorder, not iron — liver disease, Kayser-Fleischer rings); "
            "NBIA disorders (neurodegeneration with brain iron accumulation — different genes, PKAN/BPAN/etc.); "
            "Classical HH (high transferrin saturation — opposite of aceruloplasminemia); "
            "Parkinson disease + DM (ceruloplasmin normal; no ferritin elevation); "
            "Pantothenate kinase deficiency PKAN (eye-of-tiger MRI sign; no ceruloplasmin abnormality)."
        ),
    },
    # ── FTL — Hyperferritinemia-Cataract Syndrome ────────────────────────────
    {
        "gene": "FTL",
        "protein": "Ferritin Light Chain (FTL)",
        "alias": (
            "FTL (OMIM gene 134790); Hyperferritinemia-Cataract Syndrome HHCS #600886; 19q13.33; "
            "175 aa; ~20 kDa; AD (dominant); pathogenic variants in the iron-responsive element (IRE) "
            "in the 5-prime UTR of FTL mRNA → IRE-IRP regulatory system disrupted → FTL mRNA translated "
            "constitutively even in low-iron state → ferritin L-chain overproduced → crystal deposition in "
            "lens → bilateral dense nuclear cataracts from childhood; NO systemic iron overload; "
            "serum ferritin 500-10,000 μg/L but transferrin saturation NORMAL; DO NOT venesect"
        ),
        "aa": "175 aa",
        "kDa": "~20 kDa",
        "locus": "19q13.33",
        "omim_gene": 134790,
        "omim_disease": 600886,
        "inheritance": "AD (dominant); high penetrance; de novo variants reported; >25 distinct IRE mutations described; founder variants in Southern European populations",
        "gene_class": (
            "FTL encodes the ferritin light chain (L-subunit), which forms heteropolymeric ferritin shells "
            "with the heavy chain (FTH1). Ferritin is the primary intracellular iron storage protein. "
            "REGULATORY MECHANISM: in low-iron states, iron-regulatory proteins (IRP1, IRP2) bind the "
            "iron-responsive element (IRE) in the 5' UTR of FTL mRNA → translational repression → "
            "less ferritin produced (appropriate — no need to store absent iron). "
            "PATHOMECHANISM: mutations in the FTL IRE → altered stem-loop structure → IRP cannot bind → "
            "FTL mRNA translated constitutively regardless of iron status → ferritin L-subunit overproduced → "
            "excess ferritin deposits in lens epithelium and fibres → crystal aggregates → "
            "bilateral dense cataracts (nuclear, posterior subcapsular, or congenital). "
            "SYSTEMIC: ferritin is elevated in serum (500-10,000 μg/L) but represents overproduced "
            "L-chain protein, NOT iron-laden ferritin — transferrin saturation is NORMAL, "
            "liver iron is NORMAL on MRI, no parenchymal iron accumulation. "
            "CRITICAL: venesection will NOT lower ferritin in HHCS (ferritin production is iron-independent "
            "due to the IRE mutation) and WILL cause iron deficiency anemia."
        ),
        "phenotype": (
            "BILATERAL CATARACTS from childhood or early adulthood: "
            "nuclear (characteristic crystalline deposits), posterior subcapsular, progressive; "
            "SERUM FERRITIN MARKEDLY ELEVATED (500-10,000 μg/L): "
            "discovered incidentally or during evaluation of cataracts; "
            "TRANSFERRIN SATURATION NORMAL: key distinguishing feature from HH; "
            "NO SYSTEMIC IRON OVERLOAD: liver MRI T2* normal; no hepatomegaly; no diabetes; no arthropathy; "
            "FAMILY HISTORY: cataract + high ferritin in parent or sibling (AD — 50% of offspring)"
        ),
        "hallmark": (
            "BILATERAL CATARACTS + VERY HIGH SERUM FERRITIN + NORMAL TRANSFERRIN SATURATION = "
            "Hyperferritinemia-Cataract Syndrome (HHCS) PATHOGNOMONIC triad. "
            "ABSOLUTE RULE: DO NOT VENESECT — will cause iron deficiency anemia without lowering ferritin. "
            "REASSURE patient: high ferritin does NOT indicate iron overload; organs are not damaged. "
            "TREATMENT: cataract surgery when visually significant (same as age-related cataracts)."
        ),
        "treatment_alerts": [
            "DO NOT VENESECT: venesection will cause iron deficiency anemia; ferritin will NOT decrease because production is iron-independent.",
            "CATARACT SURGERY: standard phacoemulsification when vision affected; good outcomes; cataracts recur if any residual lens tissue left.",
            "NO IRON CHELATION NEEDED: no systemic iron overload to treat.",
            "PATIENT EDUCATION: explain that high ferritin is protein overproduction, NOT iron overload — prevents iatrogenic harm.",
            "GENETIC TESTING: FTL IRE sequencing (5-prime UTR — standard clinical exome may NOT capture UTR variants; request specifically).",
            "FAMILY SCREENING: 50% of offspring affected; cataract + ferritin in relatives is diagnostic.",
        ],
        "key_ddx": (
            "Hereditary hemochromatosis HFE/TFR2 (high transferrin saturation + high ferritin — vs HHCS where sat NORMAL); "
            "Inflammatory hyperferritinemia (ferritin elevated in sepsis, malignancy, MAS — transferrin sat normal but "
            "context is illness, not cataracts); "
            "Adult-onset Still disease (very high ferritin, fever, arthritis — not cataracts); "
            "Metabolic hyperferritinemia (obesity, NAFLD, alcohol — ferritin 200-1000, transferrin sat borderline, no cataracts)."
        ),
    },
]


def _make_cohort(gene_info: dict, seed: int) -> list:
    """Generate a 40-patient cohort for one gene."""
    rng = random.Random(seed)
    gene = gene_info["gene"]
    pts = []
    for i in range(40):
        age = rng.randint(8, 72)
        sex = rng.choice(["M", "F", "M", "M"] if gene in ("HFE", "TFR2", "TFR2") else ["M", "F"])
        # Gene-specific phenotypic fields
        p: dict = {"id": i + 1, "age": age, "sex": sex}

        if gene == "HFE":
            p["genotype"] = rng.choice(["C282Y/C282Y"] * 7 + ["C282Y/H63D"] * 2 + ["H63D/H63D"])
            p["ferritin"] = rng.randint(400, 4000)
            p["transferrin_sat"] = rng.randint(48, 90)
            p["cirrhosis"] = rng.random() < 0.25
            p["diabetes"] = rng.random() < 0.30
            p["arthropathy_mcp"] = rng.random() < 0.60
            p["cardiomyopathy"] = rng.random() < 0.10
            p["phlebotomy_started"] = rng.random() < 0.85
            p["ferritin_post_phlebotomy"] = rng.randint(20, 90) if p["phlebotomy_started"] else None
        elif gene == "HJV":
            p["age_onset"] = rng.randint(10, 28)
            p["ferritin"] = rng.randint(3000, 12000)
            p["transferrin_sat"] = rng.randint(75, 98)
            p["cardiomyopathy"] = rng.random() < 0.75
            p["hypogonadism"] = rng.random() < 0.80
            p["hepcidin_undetectable"] = True
            p["phlebotomy_urgent"] = rng.random() < 0.90
            p["desferrioxamine_used"] = rng.random() < 0.55
        elif gene == "HAMP":
            p["age_onset"] = rng.randint(8, 25)
            p["ferritin"] = rng.randint(2500, 10000)
            p["hepcidin_serum"] = "undetectable"
            p["cardiomyopathy"] = rng.random() < 0.70
            p["hypogonadism"] = rng.random() < 0.75
            p["phlebotomy_urgent"] = rng.random() < 0.88
        elif gene == "TFR2":
            p["ferritin"] = rng.randint(350, 3500)
            p["transferrin_sat"] = rng.randint(45, 85)
            p["hfe_normal"] = True
            p["cirrhosis"] = rng.random() < 0.20
            p["diabetes"] = rng.random() < 0.25
            p["arthropathy_mcp"] = rng.random() < 0.50
            p["phlebotomy_started"] = rng.random() < 0.80
        elif gene == "SLC40A1":
            p["subtype"] = rng.choice(["4A"] * 3 + ["4B"] * 2)
            p["ferritin"] = rng.randint(500, 5000)
            p["transferrin_sat"] = rng.randint(15, 35) if p["subtype"] == "4A" else rng.randint(45, 85)
            p["macrophage_iron_pattern"] = p["subtype"] == "4A"
            p["phlebotomy_tolerated"] = rng.random() < 0.40 if p["subtype"] == "4A" else rng.random() < 0.85
            p["chelation_used"] = rng.random() < 0.35 if p["subtype"] == "4A" else rng.random() < 0.10
        elif gene == "TMPRSS6":
            p["hb_gdl"] = round(rng.uniform(6.5, 9.5), 1)
            p["mcv"] = rng.randint(58, 72)
            p["ferritin"] = rng.randint(3, 18)
            p["hepcidin_elevated"] = rng.random() < 0.90
            p["oral_iron_failed"] = True
            p["iv_iron_response"] = rng.random() < 0.88
            p["hb_post_iv_iron"] = round(p["hb_gdl"] + rng.uniform(1.5, 3.5), 1)
        elif gene == "CP":
            p["age_onset"] = rng.randint(28, 58)
            p["neurodegeneration"] = rng.random() < 0.85
            p["diabetes"] = rng.random() < 0.80
            p["retinal_degeneration"] = rng.random() < 0.70
            p["ceruloplasmin_absent"] = True
            p["ferritin"] = rng.randint(800, 8000)
            p["transferrin_sat"] = rng.randint(8, 25)
            p["desferrioxamine_used"] = rng.random() < 0.65
            p["mri_brain_hypointensity"] = rng.random() < 0.90
        elif gene == "FTL":
            p["cataract_bilateral"] = True
            p["ferritin"] = rng.randint(500, 9000)
            p["transferrin_sat"] = rng.randint(18, 38)
            p["no_iron_overload"] = True
            p["venesection_attempted_erroneously"] = rng.random() < 0.30
            p["anemia_from_erroneous_venesection"] = p["venesection_attempted_erroneously"] and rng.random() < 0.75
            p["cataract_surgery"] = rng.random() < 0.60
        pts.append(p)
    return pts


_ALL_COHORTS = {
    g["gene"]: _make_cohort(g, SEED_BASE + i)
    for i, g in enumerate(IRON_GENES)
}


def _pct(pts: list, key: str) -> float:
    vals = [p[key] for p in pts if key in p and p[key] is not None]
    return round(sum(bool(v) for v in vals) / len(vals) * 100, 1) if vals else 0.0


def _avg(pts: list, key: str) -> float:
    vals = [p[key] for p in pts if key in p and p[key] is not None and isinstance(p[key], (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def get_overview() -> dict:
    all_pts = [p for pts in _ALL_COHORTS.values() for p in pts]
    n = len(all_pts)
    return {
        "atlas_name": "Iron-Disorders-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Iron Metabolism Disorders Atlas",
        "n_patients": n,
        "n_genes": len(IRON_GENES),
        "seeds": "1286–1293",
        "genes": [g["gene"] for g in IRON_GENES],
        "diseases": {
            "HFE":     "Hereditary Hemochromatosis Type 1 (HH1) — most common",
            "HJV":     "Hereditary Hemochromatosis Type 2A — Juvenile (HH2A)",
            "HAMP":    "Hereditary Hemochromatosis Type 2B — Juvenile (HH2B)",
            "TFR2":    "Hereditary Hemochromatosis Type 3 (HH3)",
            "SLC40A1": "Hereditary Hemochromatosis Type 4A/4B — Ferroportin Disease",
            "TMPRSS6": "Iron-Refractory Iron Deficiency Anemia (IRIDA)",
            "CP":      "Aceruloplasminemia — neurodegeneration + DM + retinal degeneration",
            "FTL":     "Hyperferritinemia-Cataract Syndrome (HHCS) — DO NOT VENESECT",
        },
        "inheritance_pattern": {
            "AR": ["HFE", "HJV", "HAMP", "TFR2", "TMPRSS6", "CP"],
            "AD": ["SLC40A1", "FTL"],
        },
        "key_treatment_rules": [
            "HFE/HJV/HAMP/TFR2: Phlebotomy curative if pre-cirrhotic — transferrin saturation >45% triggers start",
            "SLC40A1 Type 4A: DO NOT venesect aggressively — macrophage iron, anemia risk; erythrocytapheresis preferred",
            "TMPRSS6-IRIDA: Oral iron FAILS — IV iron (ferric carboxymaltose) is treatment of choice",
            "CP-Aceruloplasminemia: IV Desferrioxamine; NO venesection (iron not in circulation)",
            "FTL-HHCS: DO NOT VENESECT — high ferritin is protein overproduction NOT iron overload; cataract surgery only",
            "HJV/HAMP Juvenile HH: URGENT phlebotomy + cardiac monitoring; age <30 + cardiomyopathy = emergency",
        ],
        "aggregate_clinical": {
            "hfe_hh1_cirrhosis_pct": _pct(_ALL_COHORTS["HFE"], "cirrhosis"),
            "hfe_arthropathy_mcp_pct": _pct(_ALL_COHORTS["HFE"], "arthropathy_mcp"),
            "hjv_cardiomyopathy_pct": _pct(_ALL_COHORTS["HJV"], "cardiomyopathy"),
            "hjv_hypogonadism_pct": _pct(_ALL_COHORTS["HJV"], "hypogonadism"),
            "tmprss6_oral_iron_fails_pct": _pct(_ALL_COHORTS["TMPRSS6"], "oral_iron_failed"),
            "tmprss6_iv_iron_response_pct": _pct(_ALL_COHORTS["TMPRSS6"], "iv_iron_response"),
            "cp_neurodegeneration_pct": _pct(_ALL_COHORTS["CP"], "neurodegeneration"),
            "cp_triple_triad_all_pct": round(
                sum(1 for p in _ALL_COHORTS["CP"] if p.get("neurodegeneration") and p.get("diabetes") and p.get("retinal_degeneration")) / 40 * 100, 1
            ),
            "ftl_erroneous_venesection_pct": _pct(_ALL_COHORTS["FTL"], "venesection_attempted_erroneously"),
            "ftl_anemia_from_venesection_pct": _pct(_ALL_COHORTS["FTL"], "anemia_from_erroneous_venesection"),
        },
    }


def get_breakdown() -> dict:
    out: dict = {}
    for ginfo in IRON_GENES:
        gene = ginfo["gene"]
        pts = _ALL_COHORTS[gene]
        stats: dict = {
            "n": len(pts),
            "sex_m_pct": round(sum(1 for p in pts if p.get("sex") == "M") / len(pts) * 100, 1),
        }
        if gene == "HFE":
            stats.update({
                "ferritin_mean": _avg(pts, "ferritin"),
                "transferrin_sat_mean": _avg(pts, "transferrin_sat"),
                "cirrhosis_pct": _pct(pts, "cirrhosis"),
                "diabetes_pct": _pct(pts, "diabetes"),
                "arthropathy_mcp_pct": _pct(pts, "arthropathy_mcp"),
                "phlebotomy_started_pct": _pct(pts, "phlebotomy_started"),
            })
        elif gene == "HJV":
            stats.update({
                "ferritin_mean": _avg(pts, "ferritin"),
                "age_onset_mean": _avg(pts, "age_onset"),
                "cardiomyopathy_pct": _pct(pts, "cardiomyopathy"),
                "hypogonadism_pct": _pct(pts, "hypogonadism"),
                "desferrioxamine_pct": _pct(pts, "desferrioxamine_used"),
            })
        elif gene == "HAMP":
            stats.update({
                "ferritin_mean": _avg(pts, "ferritin"),
                "age_onset_mean": _avg(pts, "age_onset"),
                "cardiomyopathy_pct": _pct(pts, "cardiomyopathy"),
                "hypogonadism_pct": _pct(pts, "hypogonadism"),
                "phlebotomy_urgent_pct": _pct(pts, "phlebotomy_urgent"),
            })
        elif gene == "TFR2":
            stats.update({
                "ferritin_mean": _avg(pts, "ferritin"),
                "transferrin_sat_mean": _avg(pts, "transferrin_sat"),
                "hfe_normal_pct": _pct(pts, "hfe_normal"),
                "cirrhosis_pct": _pct(pts, "cirrhosis"),
                "arthropathy_mcp_pct": _pct(pts, "arthropathy_mcp"),
            })
        elif gene == "SLC40A1":
            stats.update({
                "type_4a_pct": round(sum(1 for p in pts if p.get("subtype") == "4A") / len(pts) * 100, 1),
                "type_4b_pct": round(sum(1 for p in pts if p.get("subtype") == "4B") / len(pts) * 100, 1),
                "ferritin_mean": _avg(pts, "ferritin"),
                "phlebotomy_tolerated_pct": _pct(pts, "phlebotomy_tolerated"),
                "chelation_used_pct": _pct(pts, "chelation_used"),
            })
        elif gene == "TMPRSS6":
            stats.update({
                "hb_mean": _avg(pts, "hb_gdl"),
                "mcv_mean": _avg(pts, "mcv"),
                "ferritin_mean": _avg(pts, "ferritin"),
                "hepcidin_elevated_pct": _pct(pts, "hepcidin_elevated"),
                "oral_iron_failed_pct": _pct(pts, "oral_iron_failed"),
                "iv_iron_response_pct": _pct(pts, "iv_iron_response"),
                "hb_post_iv_mean": _avg(pts, "hb_post_iv_iron"),
            })
        elif gene == "CP":
            stats.update({
                "ferritin_mean": _avg(pts, "ferritin"),
                "transferrin_sat_mean": _avg(pts, "transferrin_sat"),
                "neurodegeneration_pct": _pct(pts, "neurodegeneration"),
                "diabetes_pct": _pct(pts, "diabetes"),
                "retinal_degeneration_pct": _pct(pts, "retinal_degeneration"),
                "mri_brain_hypointensity_pct": _pct(pts, "mri_brain_hypointensity"),
                "desferrioxamine_pct": _pct(pts, "desferrioxamine_used"),
            })
        elif gene == "FTL":
            stats.update({
                "ferritin_mean": _avg(pts, "ferritin"),
                "transferrin_sat_mean": _avg(pts, "transferrin_sat"),
                "bilateral_cataract_pct": _pct(pts, "cataract_bilateral"),
                "erroneous_venesection_pct": _pct(pts, "venesection_attempted_erroneously"),
                "iatrogenic_anemia_pct": _pct(pts, "anemia_from_erroneous_venesection"),
                "cataract_surgery_pct": _pct(pts, "cataract_surgery"),
            })
        out[gene] = {
            "gene": gene,
            "protein": ginfo["protein"],
            "aa": ginfo["aa"],
            "kDa": ginfo["kDa"],
            "locus": ginfo["locus"],
            "omim_gene": ginfo["omim_gene"],
            "omim_disease": ginfo["omim_disease"],
            "inheritance": ginfo["inheritance"],
            "gene_class": ginfo["gene_class"],
            "phenotype": ginfo["phenotype"],
            "hallmark": ginfo["hallmark"],
            "treatment_alerts": ginfo["treatment_alerts"],
            "key_ddx": ginfo["key_ddx"],
            "cohort_stats": stats,
        }
    return {"breakdown": out}


def get_definitions() -> dict:
    return {
        "definitions": {
            "Hereditary_Hemochromatosis": (
                "Hereditary hemochromatosis (HH) is a group of AR (and one AD) disorders of iron overload "
                "caused by deficient hepcidin production or action, leading to unregulated intestinal iron absorption "
                "and parenchymal iron deposition. "
                "Types: HH1 (HFE — most common, C282Y); HH2A (HJV — juvenile); HH2B (HAMP — juvenile); "
                "HH3 (TFR2 — adult HFE-like); HH4 (SLC40A1/Ferroportin — AD, two subtypes). "
                "Pathway: BMP6 → HJV (co-receptor) → BMPR1/2 → SMAD1/5/8 → HAMP → hepcidin → "
                "ferroportin degradation → reduced iron absorption and macrophage release."
            ),
            "Hepcidin_Iron_Axis": (
                "Hepcidin (HAMP) is the master iron regulator: "
                "HIGH hepcidin → ferroportin degraded → iron trapped in cells → low serum iron. "
                "LOW hepcidin → ferroportin active → iron exported → high serum iron. "
                "HEREDITARY HH (HFE/HJV/HAMP/TFR2): hepcidin LOW despite iron excess → "
                "  → continued iron absorption → parenchymal loading. "
                "IRIDA (TMPRSS6): hepcidin HIGH despite iron deficiency → "
                "  → ferroportin degraded → oral iron cannot be absorbed → IV iron bypasses block. "
                "INFLAMMATION: IL-6 → hepcidin HIGH → anemia of chronic disease (low serum iron, high ferritin). "
                "FTL-HHCS: hepcidin NORMAL, iron NORMAL — the elevated ferritin is protein overproduction."
            ),
            "HFE_C282Y_Genotype": (
                "p.C282Y (c.845G>A in HFE exon 4): disrupts Cys260-Cys203 disulfide bond in alpha-3 domain → "
                "misfolded protein retained in ER → absent cell-surface HFE-B2M-TFR1 complex. "
                "Homozygous C282Y (~1:200 Northern European) = highest-penetrance HH genotype: "
                "  50% of C282Y/C282Y females and 25% of males develop biochemical or clinical disease. "
                "p.H63D: Asp to Glu substitution; HFE reaches cell surface; lower risk — "
                "  H63D/H63D: very rarely causes HH alone; C282Y/H63D: mild HH in a minority. "
                "GENOTYPING: HFE C282Y + H63D is first-line (PCR-based, widely available); "
                "If HH phenotype with NORMAL HFE genotype → sequence TFR2, HJV, HAMP, SLC40A1."
            ),
            "Transferrin_Saturation_Rule": (
                "Transferrin saturation (TSAT) = serum iron / TIBC × 100%. "
                "NORMAL: 20-40%. "
                "DIAGNOSTIC for HH (Types 1, 2A, 2B, 3): TSAT >45% fasting = investigation threshold; "
                ">70% in overt classical HH. "
                "DO NOT USE FERRITIN ALONE: ferritin is an acute-phase reactant elevated in "
                "inflammation, alcohol, NAFLD, malignancy, infection — false positives common. "
                "ALWAYS test TSAT first, then ferritin to determine iron stores. "
                "EXCEPTION — Type 4A ferroportin disease: TSAT may be NORMAL or LOW despite high ferritin. "
                "EXCEPTION — Aceruloplasminemia: TSAT NORMAL or LOW despite very high ferritin. "
                "EXCEPTION — HHCS (FTL): TSAT NORMAL despite very high ferritin."
            ),
            "Juvenile_HH_Emergency": (
                "HJV and HAMP HH Types 2A/2B = JUVENILE HEMOCHROMATOSIS — medical emergency. "
                "Cardiomyopathy (dilated or restrictive) developing by age 20-30 is the LEADING CAUSE OF DEATH. "
                "CARDIAC MANIFESTATIONS: congestive heart failure, arrhythmias (AF, VT), sudden cardiac death. "
                "URGENT MANAGEMENT: "
                "  (1) Phlebotomy 2-3x weekly if cardiac function allows (EF >40%); "
                "  (2) IV desferrioxamine (DFO) 50-75 mg/kg/day SC/IV if EF <40% or Hb <10; "
                "  (3) Cardiac specialist involvement from day 1; "
                "  (4) Endocrinology for hypogonadism — testosterone/estrogen early. "
                "TARGET: ferritin <50 μg/L; transferrin saturation <30%."
            ),
            "SLC40A1_Type4A_vs_4B": (
                "FERROPORTIN DISEASE (HH Type 4) has TWO DISTINCT SUBTYPES requiring different treatment: "
                "TYPE 4A (LOF, ferroportin disease): "
                "  Mechanism: haploinsufficiency → reduced ferroportin surface → iron trapped in macrophages; "
                "  Iron pattern: HIGH ferritin + NORMAL/LOW TSAT + MACROPHAGE iron on liver biopsy; "
                "  Clinical: splenomegaly, mild anaemia, mild hepatic iron; "
                "  Treatment: erythrocytapheresis preferred; LOW-DOSE phlebotomy with Hb monitoring; "
                "  Venesection risk: premature anaemia — must monitor Hb 2-weekly; "
                "TYPE 4B (GOF, hepcidin-resistant): "
                "  Mechanism: ferroportin variant at hepcidin-binding site → hepcidin cannot suppress ferroportin; "
                "  Iron pattern: HIGH ferritin + HIGH TSAT + HEPATOCYTE iron (parenchymal); "
                "  Clinical: same as classical HFE HH; "
                "  Treatment: weekly phlebotomy 500 mL — as effective as HFE HH."
            ),
            "IRIDA_Oral_Iron_Failure": (
                "IRIDA (TMPRSS6): hepcidin ELEVATED despite iron deficiency → oral iron cannot be absorbed. "
                "ORAL IRON TRIAL: if 4-6 months of adequate oral iron (200 mg elemental/day) fails to raise "
                "Hb by >1 g/dL AND serum hepcidin is not suppressed → suspect IRIDA. "
                "CONFIRMATION: TMPRSS6 sequencing (biallelic pathogenic variants). "
                "IV IRON RESPONSE: ferric carboxymaltose 500-1000 mg IV → Hb rises 2-3 g/dL within 4 weeks → "
                "confirms that erythropoiesis is iron-responsive (the intestinal block is bypassed). "
                "MAINTENANCE: IV iron every 3-6 months; does not correct permanently — ongoing need. "
                "EXPERIMENTAL: nedosiran (TMPRSS6 siRNA) reduces matriptase-2 → reduces hepcidin → "
                "may improve oral iron absorption; in clinical trials."
            ),
            "Aceruloplasminemia_Triple_Triad": (
                "ACERULOPLASMINEMIA (CP biallelic LOF): "
                "PATHOGNOMONIC TRIPLE TRIAD: "
                "  (1) Progressive NEURODEGENERATION: extrapyramidal (dystonia, chorea, blepharospasm) + "
                "      cerebellar (ataxia, dysarthria) — usually onset age 40-60; "
                "  (2) DIABETES MELLITUS: insulin-dependent, onset before neurological symptoms; "
                "  (3) RETINAL DEGENERATION: pigmentary retinopathy → visual loss. "
                "BIOCHEMISTRY: Serum ceruloplasmin ABSENT (immunological assay, not just low); "
                "  Ferritin very high (800-10,000 μg/L); Transferrin saturation NORMAL or LOW. "
                "MRI BRAIN: T2* hypointensity in basal ganglia, thalamus, dentate nucleus = iron-laden neurons. "
                "TREATMENT: IV Desferrioxamine + consider Deferiprone (crosses BBB). "
                "DO NOT confuse with Wilson disease: Wilson = copper disorder with low ceruloplasmin BUT "
                "  transferrin saturation NORMAL, Kayser-Fleischer rings, liver disease, no iron overload."
            ),
            "HHCS_Do_Not_Venesect": (
                "HYPERFERRITINEMIA-CATARACT SYNDROME (FTL IRE mutation): "
                "ABSOLUTE RULE: DO NOT VENESECT. "
                "Mechanism of error: clinician sees ferritin 2000-8000 → assumes iron overload → "
                "starts phlebotomy → ferritin does NOT fall (production is iron-independent due to IRE mutation) → "
                "patient becomes iron-deficient with anemia → iatrogenic harm. "
                "CORRECT UNDERSTANDING: ferritin is elevated because the FTL mRNA is translated constitutively "
                "(IRP cannot repress it) — the iron content of the ferritin is NORMAL; organs have normal iron. "
                "DIAGNOSIS: cataract + ferritin >500 + NORMAL transferrin saturation + NORMAL liver MRI T2* = HHCS. "
                "CONFIRM: FTL 5-prime UTR sequencing (IRE mutations) — request specifically, not covered by standard exome. "
                "TREATMENT: cataract surgery when visually significant; no iron-specific therapy needed."
            ),
        }
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(f"Atlas: {ov['atlas_name']}")
    print(f"N patients: {ov['n_patients']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Aggregate: {json.dumps(ov['aggregate_clinical'], indent=2)}")
    print("\n=== BREAKDOWN (gene list) ===")
    bd = get_breakdown()
    for g, info in bd["breakdown"].items():
        print(f"  {g}: {info['cohort_stats']}")
    print("\n=== DEFINITIONS (keys) ===")
    df = get_definitions()
    for k in df["definitions"]:
        print(f"  - {k}")
