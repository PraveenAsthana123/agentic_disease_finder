#!/usr/bin/env python3
"""Hereditary-Hemochromatosis-Atlas — Complete 8-Gene Hereditary Iron-Overload Atlas
HFE    (homeostatic iron regulator; 380 aa; 6p22.2; AR;
         Hereditary Hemochromatosis type 1 (HH1) — most common (C282Y 1:200 Northern Europeans);
         phlebotomy curative/preventive; ferritin <50 µg/L target; TS >45% screening trigger;
         seed SEED_BASE+0) ·
HJV    (hemojuvelin / HFE2; 426 aa; 1q21.1; AR;
         Juvenile Hemochromatosis type 2A — most severe; cardiomyopathy + hypogonadism age 15–30;
         hepcidin synthesis abolished; phlebotomy + chelation urgent;
         seed SEED_BASE+1) ·
HAMP   (hepcidin antimicrobial peptide; 84 aa [25 aa mature]; 19q13.12; AR;
         Juvenile Hemochromatosis type 2B — hepcidin itself deficient; same phenotype as HJV;
         absent urinary hepcidin diagnostic; rarest JH subtype;
         seed SEED_BASE+2) ·
TFR2   (transferrin receptor 2; 801 aa; 7q22.1; AR;
         HH type 3 — hepatic diferric-transferrin sensor for hepcidin upregulation;
         milder phenotype similar to HFE; worldwide distribution no founder effect;
         seed SEED_BASE+3) ·
SLC40A1 (ferroportin / solute carrier family 40 member 1; 571 aa; 2q32.2; AD;
          HH type 4 — ONLY AD iron overload; LOF 4A = macrophage iron trapping (TS NORMAL/LOW);
          GOF 4B = hepcidin-resistant ferroportin = hepatocyte loading (TS elevated like HFE);
          seed SEED_BASE+4) ·
CP     (ceruloplasmin; 1065 aa; 3q25.1; AR;
         Aceruloplasminemia — ferroxidase absent → iron trapped in cells → brain + retina + liver;
         triad: retinal degeneration + neurological + diabetes PATHOGNOMONIC;
         serum iron/TS can be LOW despite iron overload; deferoxamine chelation;
         seed SEED_BASE+5) ·
FTL    (ferritin light chain; 175 aa; 19q13.33; AD;
         Hereditary Hyperferritinaemia-Cataract Syndrome (HHCS) — 5'UTR IRE gain-of-function;
         nuclear cataracts childhood + massive serum ferritin + NORMAL iron stores;
         phlebotomy DANGEROUS (causes iron deficiency); treat cataracts surgically;
         seed SEED_BASE+6) ·
BMP6   (bone morphogenetic protein 6; 513 aa; 6p24.3; AR;
         BMP6-related Hemochromatosis — main hepatic BMP for hepcidin transcription via SMAD;
         LOF → reduced hepcidin → liver iron loading; rare (<50 cases); phlebotomy treatment;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1678–1685)
"""

import random

SEED_BASE = 1678

HEMOCHROMATOSIS_GENES = [
    # ── HFE — HH type 1 / Most Common Hereditary Hemochromatosis ─────────────
    {
        "gene": "HFE",
        "protein": "HFE — HH1 AR — Homeostatic Iron Regulator — C282Y 1:200 Northern Europeans Most Common — TS >45% Screening Trigger — Phlebotomy Curative — Ferritin <50 µg/L Target",
        "alias": (
            "HFE (homeostatic iron regulator); OMIM gene 613609; "
            "Hereditary Hemochromatosis type 1 (HH1) OMIM 235200. "
            "6p22.2; 380 aa; ~42 kDa; AR biallelic LOF (C282Y/C282Y most common). "
            "FUNCTION: HFE is a MHC class I-like protein that associates with beta-2-microglobulin (β2M) and transferrin receptor 1 (TFR1). "
            "HFE-TFR1 interaction modulates hepatocyte sensing of iron status; "
            "HFE competes with holotransferrin (diferric transferrin) for TFR1 binding → "
            "when HFE is absent: hepatocytes cannot sense transferrin-iron saturation → "
            "hepcidin production remains low → maximal iron absorption in duodenum → "
            "iron accumulates in liver, heart, pancreas, joints, pituitary, skin. "
            "PATHOGENIC VARIANTS: "
            "C282Y (p.Cys282Tyr, c.845G>A): Cys282 is critical for disulfide bond linking HFE to β2M; "
            "C282Y → HFE cannot associate with β2M → misfolded HFE retained in ER → "
            "no HFE at cell surface → TFR1 unregulated → maximal iron uptake sensing failure; "
            "C282Y/C282Y: 80–90% of classic HH in Northern Europeans; "
            "penetrance incomplete: ~30% of C282Y homozygous males develop clinical disease; "
            "~1% of C282Y homozygous females develop clinical disease (menstruation protective). "
            "H63D (p.His63Asp, c.187C>G): Milder variant; heterozygous H63D alone rarely causes disease; "
            "C282Y/H63D compound heterozygous: ~5% of HH cases (iron overload mild-moderate). "
            "EPIDEMIOLOGY: C282Y/C282Y prevalence 1:200–1:250 in Northern Europeans (Irish highest 1:83); "
            "1:1000 in Mediterranean; rare in Asian/African populations (where HJV/TFR2 commoner). "
            "CLINICAL PHENOTYPE (C282Y/C282Y males): "
            "Liver fibrosis/cirrhosis (untreated 30–40 years of iron loading); "
            "hepatocellular carcinoma (200× increased risk vs general population in cirrhosis); "
            "bronze diabetes: skin pigmentation + liver + pancreatic islet destruction → diabetes; "
            "arthropathy (metacarpophalangeal joints 2nd and 3rd — PATHOGNOMONIC JOINTS); "
            "cardiomyopathy (dilated) + arrhythmias; "
            "hypogonadotropic hypogonadism (pituitary iron deposition → LH/FSH deficiency); "
            "hypothyroidism (thyroid iron deposition). "
            "BIOCHEMICAL DIAGNOSIS: "
            "Transferrin saturation (TS) >45% on fasting morning sample — first-line screening; "
            "TS >60% in males / >50% in females = highly suspicious for C282Y/C282Y; "
            "serum ferritin: elevated (200–5000+ µg/L) but non-specific (acute phase reactant); "
            "ferritin >1000 µg/L = associated with advanced fibrosis — liver biopsy or MRI iron quantification. "
            "GENETIC TESTING: C282Y/C282Y or C282Y/H63D confirmed by HFE genotyping; "
            "ALL FIRST-DEGREE RELATIVES must be offered HFE genotyping + TS/ferritin. "
            "TREATMENT — PHLEBOTOMY (VENESECTION): "
            "Each 500 mL phlebotomy removes ~250 mg iron (haemoglobin 150 g/L × 0.34% iron × 500 mL); "
            "INDUCTION PHASE: weekly phlebotomy until ferritin <50 µg/L + haemoglobin ≥120 g/L maintained; "
            "may require 50–100+ phlebotomies over 1–2 years in severe cases; "
            "MAINTENANCE PHASE: every 3–6 monthly phlebotomy to maintain ferritin <50 µg/L; "
            "OUTCOME: Phlebotomy normalises life expectancy if started before cirrhosis/diabetes; "
            "established cirrhosis/HCC risk not fully reversed by phlebotomy; "
            "ERYTHROCYTAPHERESIS: Removes 2× the RBCs per session vs simple phlebotomy — more efficient iron removal. "
            "CHELATION (deferoxamine/deferasirox): Reserved for patients with anaemia precluding phlebotomy. "
            "DIETARY: Avoid vitamin C supplements (enhances iron absorption); avoid alcohol (potentiates hepatotoxicity); "
            "avoid raw shellfish (risk Vibrio vulnificus septicaemia — iron-loving pathogen). "
            "CASCADE TESTING: All first-degree relatives HFE genotype + TS + ferritin; "
            "screen children of C282Y/C282Y proband (carrier guaranteed, HH if partner is carrier)."
        ),
        "locus": "6p22.2",
        "aa": 380,
        "kDa": 42,
        "omim_gene": "613609",
        "omim_disease": "Hereditary Hemochromatosis type 1 (HH1) OMIM 235200",
        "inheritance": "AR biallelic LOF (C282Y/C282Y 1:200 Northern Europeans; C282Y/H63D ~5% of HH cases)",
        "gene_class": "MHC class I-like protein — β2M-associated — TFR1 competitor — hepatic iron-sensing — hepcidin upstream regulator — duodenal iron absorption gate",
        "key_alerts": [
            "HFE-C282Y-PHLEBOTOMY-CURATIVE: C282Y/C282Y detected early (pre-cirrhotic) → weekly phlebotomy until ferritin <50 µg/L → normal life expectancy; phlebotomy is cheap, safe, and curative — do NOT delay treatment waiting for symptoms; begin venesection at diagnosis if TS >45% + elevated ferritin",
            "HFE-TS-45PCT-FASTING-SCREEN: Transferrin saturation >45% on fasting morning sample is the first-line screen — more specific than ferritin alone (ferritin is acute phase reactant, elevated by inflammation/alcohol/NAFLD); TS >45% → HFE C282Y/H63D genotyping mandatory",
            "HFE-FERRITIN-1000-LIVER-BIOPSY: Ferritin >1000 µg/L indicates advanced iron loading; liver MRI (R2* or FerriScan) or biopsy for hepatic iron concentration + fibrosis staging; cirrhosis at diagnosis = hepatocellular carcinoma surveillance every 6 months mandatory (200× increased HCC risk)",
            "HFE-RAW-SHELLFISH-VIBRIO-CI: Iron-replete patients MUST avoid raw shellfish; Vibrio vulnificus thrives in high-iron environments → fulminant septicaemia in HH patients; mortality >50% in iron-overloaded patients — strict shellfish cooking warning at every clinical encounter",
            "HFE-CASCADE-TESTING-RELATIVES: All first-degree relatives of confirmed HH1 proband → HFE genotype + TS + ferritin; C282Y/C282Y male relatives → start monitoring at age 18; C282Y/C282Y female relatives → start monitoring at age 30 (menstruation delays iron loading)",
        ],
        "etiologies": [
            "C282Y/C282Y (biallelic) — Cys282Tyr disrupts β2M disulfide bond → ER retention of HFE → absent surface HFE → TFR1 unconstrained → low hepcidin → maximal duodenal iron absorption → iron overload over 20–40 years",
            "C282Y/H63D (compound heterozygous) — partial HFE dysfunction; His63Asp alone on β2M-associated HFE reduces HFE-TFR1 interaction → mild to moderate iron loading in subset of patients; C282Y allele provides most of the pathogenicity",
            "H63D/H63D (biallelic H63D) — very mild if any; rarely causes clinical hemochromatosis alone; often coincidental finding; treat only if TS and ferritin clearly elevated + other causes excluded",
        ],
        "stats": {
            "mean_dx_age": 42,
            "mean_dx_delay_months": 60,
            "ferritin_at_dx_mean_ugL": 1800,
            "ts_at_dx_pct": 72,
            "phlebotomy_induction_sessions_mean": 45,
            "cirrhosis_at_dx_pct": 15,
        },
        "dx_delay_distribution": "48–96 months (symptoms non-specific fatigue/arthralgia often attributed to other causes; HH rarely considered until liver disease or diabetes develops; TS screening underutilised)",
    },

    # ── HJV — Juvenile Hemochromatosis type 2A ──────────────────────────────
    {
        "gene": "HJV",
        "protein": "HJV — Juvenile-HH2A AR — Hemojuvelin BMP6-Coreceptor — Absent Hepcidin — Cardiomyopathy+Hypogonadism Age 15-30 — Most Severe HH — Urgent Phlebotomy+Chelation",
        "alias": (
            "HJV (hemojuvelin / HFE2); OMIM gene 608374; "
            "Juvenile Hemochromatosis type 2A (JH2A) OMIM 602390. "
            "1q21.1; 426 aa; ~45 kDa; AR biallelic LOF. "
            "FUNCTION: Hemojuvelin (HJV) is a GPI-anchored co-receptor for bone morphogenetic proteins (BMPs), "
            "primarily BMP6 and BMP2/4, on hepatocytes. "
            "BMP6-HJV-BMPR2/ALK2/ALK3 → SMAD1/5/8 phosphorylation → nuclear SMAD4/1/5/8 complex → "
            "HAMP (hepcidin) gene transcription in hepatocytes; "
            "hepcidin is then secreted → binds ferroportin (SLC40A1) on enterocytes/macrophages → "
            "ferroportin internalised and degraded → iron export from gut/macrophages blocked → "
            "systemic iron lowered. "
            "HJV LOF → BMP-SMAD hepcidin signal abolished → hepcidin absent → "
            "ferroportin constitutively active → maximal duodenal iron absorption from birth → "
            "iron accumulates massively in liver, heart, pancreas, pituitary, gonads. "
            "CLINICAL PHENOTYPE (most severe HH): "
            "ONSET AGE 15–30 years (contrast HH1 presenting 30–50 years) — juvenile = early-onset; "
            "CARDIOMYOPATHY: dilated or restrictive cardiomyopathy → heart failure + arrhythmias (major cause of death); "
            "cardiac iron MRI (T2* mapping) mandatory — T2* <20 ms = significant cardiac iron; "
            "HYPOGONADOTROPIC HYPOGONADISM: pituitary iron → absent GnRH-responsive LH/FSH → "
            "delayed puberty, amenorrhoea, infertility, erectile dysfunction; "
            "gonadal iron damage also occurs (hypogonadism component); "
            "HEPATIC DISEASE: rapid fibrosis → cirrhosis by third decade; "
            "DIABETES: islet cell iron destruction; "
            "JOINT DISEASE: chondrocalcinosis. "
            "KEY VARIANTS: "
            "G320V (p.Gly320Val, c.959G>T): Portuguese founder mutation — most common worldwide JH2A variant; "
            "truncation + missense mutations throughout gene; "
            "compound heterozygous common. "
            "DIAGNOSIS: "
            "Ferritin 3000–20,000+ µg/L; TS >80% often; serum iron >30 µmol/L; "
            "serum hepcidin VERY LOW or undetectable (DIA-based assay or mass spectrometry); "
            "HJV gene sequencing confirming biallelic pathogenic variants. "
            "TREATMENT — URGENT AGGRESSIVE: "
            "Weekly phlebotomy 500 mL PLUS deferoxamine infusion (chelation) for severe cardiac iron; "
            "cardiac MRI T2* monitoring every 6–12 months; "
            "CARDIOLOGY INVOLVED EARLY: cardiac failure managed with standard HF therapy + iron chelation; "
            "hormone replacement (testosterone/oestrogen + cyclical progestogen) for hypogonadism; "
            "GnRH therapy if fertility desired (HH pituitary can sometimes recover post-iron depletion); "
            "liver transplantation for end-stage liver disease does NOT correct HH (iron-sensing defect in bowel)."
        ),
        "locus": "1q21.1",
        "aa": 426,
        "kDa": 45,
        "omim_gene": "608374",
        "omim_disease": "Juvenile Hemochromatosis type 2A (JH2A) OMIM 602390",
        "inheritance": "AR biallelic LOF — G320V Portuguese founder most common; compound heterozygous common; both sexes equally affected (NO protective effect of menstruation unlike HFE — onset before menarche possible)",
        "gene_class": "GPI-anchored BMP co-receptor — BMP6/BMP2/4 coreceptor on hepatocyte — SMAD1/5/8 hepcidin transcription — master regulator of systemic iron homeostasis via hepcidin axis",
        "key_alerts": [
            "HJV-CARDIAC-MRI-T2STAR-MANDATORY: Juvenile HH presents with cardiomyopathy as the major life-threatening complication; cardiac MRI T2* mapping is mandatory at diagnosis and every 6–12 months; T2* <20 ms = significant cardiac iron; T2* <10 ms = severe → emergency deferoxamine infusion + intensified chelation; do NOT delay cardiac assessment in any HJV/HAMP patient",
            "HJV-HYPOGONADISM-GnRH-AXIS: Pituitary iron deposition → central hypogonadotropic hypogonadism → LH/FSH low despite absent/reduced sex hormones; testosterone or HRT replacement mandatory; GnRH pulsatile therapy can restore fertility if iron depletion early enough; do NOT mistake for primary gonadal failure (LH/FSH inappropriately low for low testosterone/oestrogen)",
            "HJV-HEPCIDIN-UNDETECTABLE: Serum/urinary hepcidin is VERY LOW or undetectable in HJV and HAMP (distinct from HFE HH1 where hepcidin is merely inappropriately low relative to iron stores); hepcidin measurement differentiates JH (type 2A/2B) from HH1/3 — useful when genotype ambiguous",
            "HJV-BOTH-SEXES-EQUAL-ONSET: Unlike HFE HH1 (females protected by menstruation — decades later onset), HJV biallelic LOF affects both sexes equally from adolescence; menstruation does NOT protect in JH because iron loading rate exceeds menstrual iron loss; female JH patients present at same age and severity as males",
            "HJV-G320V-PORTUGUESE-FOUNDER: G320V is the most common JH pathogenic variant worldwide with Portuguese population founder effect; any European patient presenting with juvenile-onset iron overload + cardiomyopathy/hypogonadism → HJV G320V sequencing first; compound heterozygous with other HJV variants also frequent",
        ],
        "etiologies": [
            "HJV G320V/G320V (biallelic Portuguese founder) — Gly320Val destabilises HJV GPI anchor domain → absent surface HJV → BMP6 cannot activate SMAD signalling → hepcidin absent → maximal iron absorption from birth → severe iron overload by adolescence",
            "HJV truncating mutations (frameshift/nonsense) / biallelic missense — null alleles → absent HJV protein → complete loss of BMP6-SMAD-hepcidin axis → maximally suppressed hepcidin → most severe phenotype",
            "HJV compound heterozygous — one severe (truncating) + one moderate (missense) allele → variable phenotype; residual HJV function on missense allele may provide partial hepcidin; phenotype intermediate between HH1 and JH2A homozygous",
        ],
        "stats": {
            "mean_dx_age": 22,
            "mean_dx_delay_months": 18,
            "ferritin_at_dx_mean_ugL": 8500,
            "ts_at_dx_pct": 87,
            "cardiac_mri_t2star_mean_ms": 12,
            "hypogonadism_pct": 80,
        },
        "dx_delay_distribution": "12–30 months (cardiomyopathy/hypogonadism in young adult triggers workup; HH not initially considered; ferritin >3000 µg/L + TS >80% in patient under 30 should always prompt HJV/HAMP sequencing)",
    },

    # ── HAMP — Juvenile Hemochromatosis type 2B ─────────────────────────────
    {
        "gene": "HAMP",
        "protein": "HAMP — Juvenile-HH2B AR — Hepcidin Itself Deficient — Same Phenotype HJV — Absent Urinary Hepcidin Diagnostic — Rarest JH Subtype — Phlebotomy+Chelation",
        "alias": (
            "HAMP (hepcidin antimicrobial peptide); OMIM gene 606464; "
            "Juvenile Hemochromatosis type 2B (JH2B) OMIM 613313. "
            "19q13.12; 84 aa prepropeptide (25 aa mature bioactive peptide after signal + prodomain cleavage); "
            "~2.8 kDa mature peptide; AR biallelic LOF. "
            "FUNCTION: HAMP encodes hepcidin, a 25-amino-acid defensin-like peptide hormone secreted exclusively by hepatocytes. "
            "HEPCIDIN MECHANISM OF ACTION: "
            "Hepcidin binds ferroportin (SLC40A1) on basolateral surface of duodenal enterocytes and "
            "on plasma membrane of macrophages (reticuloendothelial system) and hepatocytes; "
            "hepcidin-ferroportin binding → ferroportin ubiquitinated → internalised → lysosomal degradation; "
            "without ferroportin: iron cannot exit enterocytes into portal blood → iron retained in enterocyte → "
            "shed with enterocyte into faecal waste; "
            "without ferroportin on macrophages: recycled iron from haemolysis cannot enter circulation; "
            "net effect: hepcidin raises serum iron only transiently (from macrophage stores); "
            "HEPCIDIN DEFICIENCY (HAMP biallelic LOF) → ferroportin constitutively active everywhere → "
            "maximum duodenal iron absorption + maximum macrophage iron release → iron floods circulation → "
            "iron deposits in parenchymal organs (liver, heart, pancreas, pituitary, gonads). "
            "REGULATION OF HEPCIDIN: "
            "Hepcidin is transcriptionally activated by: iron (via BMP6-HJV-SMAD pathway), "
            "inflammation (IL-6 via JAK-STAT3 → HAMP promoter), holotransferrin loading; "
            "hepcidin is suppressed by: iron deficiency, hypoxia (EPO-regulated erythroferrone / ERFE), "
            "HIF-1α, TMPRSS6 (matriptase-2 cleaves HJV → removes BMP co-receptor → reduces hepcidin); "
            "in HAMP LOF: hepcidin is completely absent — cannot be rescued by BMP6 or IL-6. "
            "CLINICAL PHENOTYPE: "
            "Identical to HJV JH2A (both result in absent hepcidin — different pathway steps); "
            "juvenile onset age 15–25; "
            "cardiomyopathy, hypogonadotropic hypogonadism, liver disease, diabetes; "
            "DIAGNOSTIC DIFFERENTIATOR: Serum/urinary hepcidin undetectable in HAMP (as in HJV); "
            "BMP6-SMAD pathway is INTACT in HAMP LOF (hepcidin gene itself cannot be transcribed/translated, "
            "but HJV-BMP-SMAD signalling to the promoter is normal — no SMAD phosphorylation defect). "
            "HAMP variants: Multiple biallelic LOF mutations (nonsense, splice, missense in mature domain); "
            "rarest JH subtype — fewer than 30 families reported worldwide as of 2024. "
            "TREATMENT: "
            "Identical to HJV — aggressive phlebotomy + parenteral iron chelation; "
            "cardiac MRI T2* monitoring; hormone replacement; "
            "research: exogenous hepcidin analogues (minihepcidin; PTG-300/rusfertide) tested in beta-thalassaemia; "
            "theoretically curative in HAMP LOF if hepcidin replacement confirmed effective."
        ),
        "locus": "19q13.12",
        "aa": 84,
        "kDa": 2.8,
        "omim_gene": "606464",
        "omim_disease": "Juvenile Hemochromatosis type 2B (JH2B) OMIM 613313",
        "inheritance": "AR biallelic LOF — rare (<30 families worldwide); no founder variant; both sexes equally affected",
        "gene_class": "25-aa defensin-like peptide hormone — master iron homeostasis hormone — ferroportin ligand triggering its degradation — hepatocyte-secreted — regulated by BMP-SMAD/iron/inflammation/EPO axes",
        "key_alerts": [
            "HAMP-HEPCIDIN-UNDETECTABLE-DIAGNOSTIC: Absent serum or urinary hepcidin by mass spectrometry (LC-MS/MS) is diagnostic for HAMP biallelic LOF; distinguishes from HFE/TFR2 HH (where hepcidin is inappropriately low but not zero); measurement available in specialist labs; request alongside HAMP gene sequencing",
            "HAMP-JH2B-IDENTICAL-TO-JH2A: HAMP LOF clinical phenotype is indistinguishable from HJV JH2A — same juvenile onset, cardiomyopathy, hypogonadism, severity; distinction is molecular (HAMP vs HJV sequencing); manage identically with urgent phlebotomy + chelation + cardiac MRI",
            "HAMP-BMP-SMAD-PATHWAY-INTACT: Unlike HJV where BMP-SMAD signalling is defective, in HAMP LOF the BMP6-HJV-SMAD cascade reaches the HAMP promoter normally — the promoter cannot produce hepcidin because the HAMP gene is mutated; future hepcidin replacement therapy (rusfertide/minihepcidin) is theoretically most directly curative in HAMP LOF",
            "HAMP-RAREST-JH-SUBTYPE: Fewer than 30 families reported worldwide; when clinical JH2 phenotype is present and HJV negative, HAMP sequencing is mandatory; rare disease registries + specialist centres required for management guidance",
            "HAMP-NO-MENSTRUAL-PROTECTION: Hepcidin deficiency drives iron absorption regardless of sex; menstrual iron loss does not protect female HAMP patients — iron loading rate exceeds menstrual iron loss by tenfold; females present at same age/severity as males",
        ],
        "etiologies": [
            "HAMP nonsense/frameshift biallelic — premature termination → absent hepcidin peptide → ferroportin constitutively active → maximal iron absorption + macrophage iron release → parenchymal iron deposition",
            "HAMP missense in mature domain (C321Y-equivalent in mature peptide) — disrupts hepcidin-ferroportin binding interface → hepcidin produced but cannot bind ferroportin → functionally null → iron overload",
            "HAMP splice site biallelic — aberrant splicing → truncated non-functional hepcidin precursor → absent mature peptide → iron overload (identical phenotype to null alleles)",
        ],
        "stats": {
            "mean_dx_age": 20,
            "mean_dx_delay_months": 15,
            "ferritin_at_dx_mean_ugL": 9200,
            "ts_at_dx_pct": 89,
            "urinary_hepcidin_ngmL": 0,
            "cardiac_mri_t2star_mean_ms": 11,
        },
        "dx_delay_distribution": "12–24 months (extreme rarity — clinicians may not consider HAMP; juvenile iron overload with undetectable hepcidin + HJV negative → HAMP sequencing; specialist referral required)",
    },

    # ── TFR2 — HH type 3 ─────────────────────────────────────────────────────
    {
        "gene": "TFR2",
        "protein": "TFR2 — HH3 AR — Transferrin-Receptor-2 Diferric-Transferrin Sensor — Hepcidin Downregulation — Similar to HFE HH1 — Worldwide Distribution No Founder — Phlebotomy Treatment",
        "alias": (
            "TFR2 (transferrin receptor 2); OMIM gene 604720; "
            "Hereditary Hemochromatosis type 3 (HH3) OMIM 604250. "
            "7q22.1; 801 aa; ~90 kDa (homodimer); AR biallelic LOF. "
            "FUNCTION: TFR2 is a type II transmembrane protein homologous to TFR1 (transferrin receptor 1) "
            "but with 25–30× lower affinity for transferrin. "
            "TFR2 is expressed primarily in hepatocytes and erythroid precursors (contrast TFR1 which is ubiquitous). "
            "HEPATIC IRON SENSING ROLE: TFR2 acts as a hepatic sensor for circulating holotransferrin (diferric transferrin); "
            "when holotransferrin levels rise (indicating iron repletion) → TFR2 is stabilised at hepatocyte surface → "
            "TFR2 interacts with HFE and activates BMP-SMAD signalling → upregulates HAMP (hepcidin) transcription; "
            "TFR2 LOF → hepatocytes cannot sense holotransferrin → hepcidin remains low despite iron repletion → "
            "iron absorption continues → hepatic iron overload. "
            "ERYTHROID TFR2: In erythroid precursors, TFR2 associates with EPO receptor + ERFE pathway; "
            "TFR2 loss in erythroid cells → altered erythropoiesis (erythrocytosis or anaemia depending on iron status); "
            "hepatic TFR2 LOF is the primary driver of HH3. "
            "KEY VARIANT: Y250X (p.Tyr250Ter, c.750C>G): Nonsense variant; premature termination; "
            "European founder mutation; most studied TFR2 HH3 variant; "
            "other variants: missense, deletion, splice site — all over TFR2 gene; "
            "no single predominant founder worldwide. "
            "CLINICAL PHENOTYPE (similar to HFE HH1 but distinct): "
            "Onset age 30–50 years (similar to HFE); males predominantly symptomatic; "
            "transferrin saturation >50–60%; ferritin 500–5000 µg/L; "
            "liver disease (fibrosis, cirrhosis); diabetes; arthropathy; "
            "NOT as severe as JH (HJV/HAMP) — hepcidin merely inappropriately low (not absent); "
            "some residual hepcidin from other inputs (inflammation, HFE-independent pathways); "
            "WORLDWIDE DISTRIBUTION: Unlike HFE (Northern European predominance), TFR2 HH3 has no single "
            "ethnic founder — cases reported in European, Mediterranean, Asian, and Middle Eastern populations; "
            "HFE genotype-negative HH (C282Y/C282Y negative) → TFR2 sequencing mandatory. "
            "DIAGNOSIS: "
            "HFE genotyping NEGATIVE despite typical HH biochemistry; "
            "TFR2 biallelic pathogenic variants on sequencing; "
            "ferritin + TS screening identical to HH1. "
            "TREATMENT: "
            "Phlebotomy identical protocol to HFE HH1 — weekly venesection to ferritin <50 µg/L; "
            "usually responds as well as HFE HH1 (similar iron loading rate and hepcidin pathophysiology); "
            "chelation rarely needed (no cardiac iron in typical TFR2 HH3 at diagnosis); "
            "first-degree relatives: TFR2 gene sequencing + TS + ferritin."
        ),
        "locus": "7q22.1",
        "aa": 801,
        "kDa": 90,
        "omim_gene": "604720",
        "omim_disease": "Hereditary Hemochromatosis type 3 (HH3) OMIM 604250",
        "inheritance": "AR biallelic LOF — no predominant ethnic founder; worldwide distribution; Y250X European founder most studied",
        "gene_class": "Type II transmembrane transferrin receptor — hepatic holotransferrin sensor — HFE-interacting — BMP-SMAD-hepcidin upstream activator — erythroid precursor co-regulator",
        "key_alerts": [
            "TFR2-HFE-NEGATIVE-HH-WORKUP: Any patient with transferrin saturation >45% + elevated ferritin but HFE C282Y/H63D genotyping NEGATIVE → TFR2 biallelic sequencing is the next step; TFR2 HH3 is the most common non-HFE hereditary hemochromatosis in European populations after HJV",
            "TFR2-SIMILAR-TO-HFE-PHENOTYPE: HH3 phenotype closely resembles HH1 — same onset age, same iron parameters, same complications; phlebotomy protocol identical; prognosis with early treatment same as HFE HH1; the molecular distinction (TFR2 vs HFE) matters for cascade testing, not for treatment decisions",
            "TFR2-WORLDWIDE-NO-FOUNDER: Unlike HFE (Northern European C282Y), TFR2 HH3 has cases in all ethnicities; Mediterranean, Middle Eastern, Asian patients with HH biochemistry but no HFE variants → TFR2 sequencing + HJV + HAMP panel",
            "TFR2-ERYTHROID-ROLE: TFR2 in erythroid precursors interacts with EPO receptor; biallelic TFR2 LOF can produce mild erythroid abnormalities (microcytosis, occasional erythrocytosis) in addition to iron overload — do not mistake erythroid changes for primary haematological disorder; confirm iron overload context first",
            "TFR2-Y250X-EUROPEAN-VARIANT: Y250X (c.750C>G) is the most characterised TFR2 HH3 pathogenic variant; useful for targeted testing in European HH3 families after proband identified by full sequencing; functional null allele — premature termination → absent TFR2 protein",
        ],
        "etiologies": [
            "TFR2 Y250X/Y250X (or compound heterozygous with other LOF) — premature termination → no TFR2 protein → hepatocytes cannot sense holotransferrin → hepcidin not upregulated with iron loading → iron absorption continues → liver iron overload",
            "TFR2 missense biallelic — disrupts TFR2-HFE interaction or holotransferrin-binding domain → partial or complete loss of iron sensing → inappropriately low hepcidin → iron overload (milder than Y250X biallelic)",
            "TFR2 large deletion — exon(s) deleted → frameshift or in-frame exon skipping → non-functional TFR2 → complete iron-sensing failure → HH3 phenotype identical to null alleles",
        ],
        "stats": {
            "mean_dx_age": 41,
            "mean_dx_delay_months": 72,
            "ferritin_at_dx_mean_ugL": 2200,
            "ts_at_dx_pct": 68,
            "phlebotomy_induction_sessions_mean": 35,
            "cirrhosis_at_dx_pct": 12,
        },
        "dx_delay_distribution": "48–120 months (HFE negative causes prolonged diagnostic uncertainty; many physicians stop workup after negative HFE genotyping; HH clinical syndrome with negative HFE must drive further panel testing)",
    },

    # ── SLC40A1 — Ferroportin Disease / HH type 4 ───────────────────────────
    {
        "gene": "SLC40A1",
        "protein": "SLC40A1 — HH4 AD — Ferroportin ONLY-AD-Iron-Overload — LOF-4A-Macrophage-Loading-TS-NORMAL — GOF-4B-Hepcidin-Resistant-Hepatocyte-Loading-TS-High — KEY-DDx-Transferrin-Saturation",
        "alias": (
            "SLC40A1 (solute carrier family 40 member 1 / ferroportin / FPN1 / MTP1 / IREG1); OMIM gene 604653; "
            "Hereditary Hemochromatosis type 4 — Ferroportin Disease (OMIM 606069). "
            "2q32.2; 571 aa; ~63 kDa; AD — the ONLY autosomal dominant hereditary hemochromatosis. "
            "FUNCTION: Ferroportin (SLC40A1) is the SOLE cellular iron exporter in vertebrates. "
            "It is expressed on the basolateral membrane of duodenal enterocytes, "
            "on macrophages (reticuloendothelial system), and on hepatocytes. "
            "FERROPORTIN CYCLE: Dietary non-haem iron enters duodenal enterocyte apically via DMT1; "
            "iron stored in ferritin or exported basolaterally by ferroportin into portal blood; "
            "ferroportin exports Fe2+ → hephaestin/ceruloplasmin oxidise to Fe3+ → loaded onto transferrin. "
            "HEPCIDIN-FERROPORTIN AXIS: Hepcidin → ferroportin → internalisation and degradation; "
            "when ferroportin is degraded: iron trapped in enterocytes (shed in faeces) + macrophages; "
            "systemic iron falls → duodenal iron absorption reduced. "
            "TWO MECHANISTICALLY DISTINCT SLC40A1 PHENOTYPES: "
            "TYPE 4A — FERROPORTIN DISEASE (LOF) — Classical Ferroportin Disease: "
            "Heterozygous LOF mutations (haploinsufficiency); "
            "reduced ferroportin → iron cannot exit enterocytes/macrophages → "
            "MACROPHAGE iron loading (reticuloendothelial system) → high ferritin from macrophage iron; "
            "TRANSFERRIN SATURATION NORMAL or LOW (iron not getting from macrophage into plasma efficiently); "
            "KEY DISTINCTION from HFE HH1/HJV/HAMP/TFR2: TS is NORMAL/LOW in Type 4A vs ELEVATED in all others; "
            "anaemia of chronic disease pattern may occur (iron trapped in macrophages — not available for erythropoiesis); "
            "PHLEBOTOMY TOLERANCE POOR — anaemia develops quickly → reduce phlebotomy frequency/volume. "
            "TYPE 4B — HH type 4B (GOF) — Hepcidin-Resistant Ferroportin: "
            "Heterozygous GOF mutations at hepcidin-binding interface of ferroportin; "
            "C326S/Y, N144H/D/T — mutations prevent hepcidin from internalising ferroportin; "
            "hepcidin produced normally but cannot inhibit ferroportin → ferroportin constitutively active → "
            "maximal duodenal iron absorption + macrophage iron release → "
            "HEPATOCYTE iron loading (parenchymal) → elevated TS (like HFE HH1); "
            "phenotype more severe than Type 4A; resembles HH1 clinically; "
            "PHLEBOTOMY well-tolerated (same as HH1). "
            "AD INHERITANCE (unique among HH subtypes): "
            "AD — single heterozygous allele sufficient for disease; "
            "50% first-degree relatives at risk; cascade genetic testing mandatory; "
            "de novo mutations possible but rare. "
            "TREATMENT: "
            "4A: Cautious phlebotomy (reduced volume/frequency to avoid anaemia) — ferritin target <200 µg/L (less aggressive than HFE); "
            "4B: Standard phlebotomy protocol as for HFE HH1; "
            "distinguish 4A from 4B by TS: TS normal/low = 4A (conservative phlebotomy); TS elevated = 4B (aggressive phlebotomy)."
        ),
        "locus": "2q32.2",
        "aa": 571,
        "kDa": 63,
        "omim_gene": "604653",
        "omim_disease": "Hereditary Hemochromatosis type 4 / Ferroportin Disease OMIM 606069 (Type 4A LOF; Type 4B GOF)",
        "inheritance": "AD heterozygous — ONLY autosomal dominant HH subtype; LOF (Type 4A) haploinsufficiency; GOF (Type 4B) hepcidin-binding interface dominant negative",
        "gene_class": "Sole vertebrate cellular iron exporter — basolateral enterocyte + macrophage + hepatocyte — hepcidin receptor (target for internalisation/degradation) — 12-transmembrane MFS-superfamily iron transporter",
        "key_alerts": [
            "SLC40A1-TS-NORMAL-LOW-TYPE4A-CRITICAL: Transferrin saturation NORMAL or LOW in Ferroportin Disease Type 4A — the OPPOSITE of all other hereditary hemochromatosis types; a patient with elevated ferritin but NORMAL TS has ferroportin disease until proven otherwise; ordering TS alone will NOT flag Type 4A — must measure ferritin in all unexplained hyperferritinaemia",
            "SLC40A1-PHLEBOTOMY-4A-CAUTIOUS: Type 4A ferroportin disease — iron is trapped in macrophages, not freely available for erythropoiesis; aggressive phlebotomy causes anaemia quickly; reduce phlebotomy volume (200–300 mL) + frequency (every 4–8 weeks); ferritin target ≤200 µg/L (less aggressive than HH1 target <50); monitor haemoglobin closely",
            "SLC40A1-GOF-4B-TREAT-LIKE-HFE: Type 4B (GOF, hepcidin-resistant ferroportin) behaves like HFE HH1 — elevated TS, hepatocyte iron loading, well-tolerated phlebotomy; standard weekly phlebotomy protocol; ferritin target <50 µg/L; distinguish 4B from 4A by elevated TS before starting aggressive phlebotomy",
            "SLC40A1-AD-CASCADE-TESTING: Only autosomal dominant HH — 50% of first-degree relatives carry the pathogenic variant regardless of sex; cascade genetic testing for SLC40A1 variant is mandatory; all first-degree relatives need TS + ferritin + SLC40A1 targeted sequencing; de novo mutations are rare but possible",
            "SLC40A1-C326S-N144H-GOF-VARIANTS: C326Y, C326S, N144H, N144D, N144T are hepcidin-binding interface GOF variants causing Type 4B; these variants prevent ferroportin internalisation by hepcidin → maximal iron export → hepatocyte iron loading; sequencing report that identifies these specific codons = Type 4B → aggressive phlebotomy indicated",
        ],
        "etiologies": [
            "SLC40A1 LOF heterozygous (Type 4A) — haploinsufficiency → reduced ferroportin at enterocyte/macrophage surface → iron trapped in macrophages → elevated ferritin + NORMAL/LOW TS + macrophage iron overload",
            "SLC40A1 GOF hepcidin-interface (Type 4B) C326S/Y, N144H/D/T — ferroportin cannot be internalised by hepcidin → constitutive iron export from gut/macrophages → hepatocyte parenchymal iron loading → elevated TS + liver fibrosis",
            "SLC40A1 missense elsewhere (LOF mechanism) — variants not at hepcidin-binding site reduce ferroportin function → macrophage iron retention → Type 4A pattern; functional assays or hepcidin challenge required to classify new variants as 4A vs 4B",
        ],
        "stats": {
            "mean_dx_age": 46,
            "mean_dx_delay_months": 84,
            "ferritin_at_dx_mean_ugL": 2600,
            "ts_at_dx_type4a_pct": 28,
            "ts_at_dx_type4b_pct": 65,
            "type4a_proportion_pct": 65,
            "type4b_proportion_pct": 35,
        },
        "dx_delay_distribution": "60–120 months (TS normal in Type 4A leads to prolonged non-diagnosis; ferritin elevation attributed to metabolic syndrome/NAFLD/alcohol; SLC40A1 not included in standard iron overload panel in many centres)",
    },

    # ── CP — Aceruloplasminemia ───────────────────────────────────────────────
    {
        "gene": "CP",
        "protein": "CP — Aceruloplasminemia AR — Ceruloplasmin Ferroxidase Absent — Iron Trapped-in-Cells — Retinal-Degeneration+Neurological+Diabetes PATHOGNOMONIC Triad — Low Serum Iron TS May Be Low — Deferoxamine Chelation",
        "alias": (
            "CP (ceruloplasmin); OMIM gene 117700; "
            "Aceruloplasminemia OMIM 604290. "
            "3q25.1; 1065 aa (mature; 1048 aa after signal peptide); ~132 kDa copper-containing glycoprotein; "
            "AR biallelic LOF. "
            "FUNCTION: Ceruloplasmin is the principal copper-carrying protein in plasma (~95% of serum copper). "
            "FERROXIDASE ROLE (primary relevance to iron overload): "
            "Ceruloplasmin contains 6 copper atoms and functions as a multicopper ferroxidase; "
            "it oxidises ferrous iron (Fe2+) → ferric iron (Fe3+) in blood/tissues; "
            "this oxidation is REQUIRED for iron loading onto apotransferrin (transferrin only binds Fe3+); "
            "without ceruloplasmin ferroxidase activity: Fe2+ cannot be oxidised → iron cannot bind transferrin → "
            "iron cannot leave cells efficiently → iron accumulates inside cells (particularly in: "
            "brain neurons and astrocytes → neurodegeneration; "
            "retinal pigment epithelium → retinal degeneration; "
            "hepatocytes → liver iron overload; "
            "pancreatic beta cells → diabetes). "
            "GPI-CERULOPLASMIN: Brain cells express GPI-anchored ceruloplasmin → cell-surface ferroxidase → "
            "mediates neuronal iron export; absent GPI-CP in aceruloplasminemia → iron trapped in neurons. "
            "PARADOX — LOW SERUM IRON DESPITE IRON OVERLOAD: "
            "Iron is trapped in cells; serum iron may be LOW or normal; "
            "transferrin saturation may be LOW or normal (not elevated as in classic HH); "
            "FERRITIN is HIGH (iron loaded into cells triggers ferritin synthesis); "
            "this pattern (high ferritin + low/normal TS + low serum ceruloplasmin) = aceruloplasminemia. "
            "CLINICAL TRIAD (ALL THREE together = PATHOGNOMONIC): "
            "(1) RETINAL DEGENERATION: Rod-cone dystrophy with pigment epithelial atrophy; "
            "progressive visual field loss; night blindness; OCT shows retinal thinning; "
            "(2) NEUROLOGICAL DISEASE: Blepharospasm, facial grimacing (basal ganglia involvement); "
            "cerebellar ataxia; dementia/cognitive decline; MRI: hypointense basal ganglia/thalamus/dentate "
            "on T2 (iron deposition); "
            "(3) DIABETES MELLITUS: Pancreatic beta cell iron destruction → insulin-requiring diabetes; "
            "often the earliest manifestation (age 30–40); "
            "ABSENT serum ceruloplasmin (<0.01 g/L or undetectable); "
            "LOW serum iron; LOW transferrin saturation. "
            "CP variants: Nonsense, frameshift, large deletion — all produce absent ceruloplasmin; "
            "GPI-CP brain isoform also absent. "
            "TREATMENT: "
            "DEFEROXAMINE (desferrioxamine): Subcutaneous or intravenous iron chelation; "
            "reduces liver and brain iron (MRI shows T2 signal improvement); "
            "neurological stabilisation but not reversal in established disease; "
            "FRESH FROZEN PLASMA: Provides functional ceruloplasmin temporarily — not sustained; "
            "DEFERASIROX oral chelation: Limited evidence in aceruloplasminemia; "
            "COPPER supplementation: Ineffective (ceruloplasmin apoprotein absent — not just copper-deficient); "
            "LOW IRON DIET: Reduce iron absorption (no ezetimibe mechanism here — CP different from HH). "
            "MONITORING: Annual MRI brain (T2/T2*), liver iron quantification, ophthalmology, HbA1c."
        ),
        "locus": "3q25.1",
        "aa": 1065,
        "kDa": 132,
        "omim_gene": "117700",
        "omim_disease": "Aceruloplasminemia OMIM 604290",
        "inheritance": "AR biallelic LOF — rare; Japanese population overrepresented in literature; worldwide cases reported",
        "gene_class": "Multicopper oxidase ferroxidase — serum copper carrier — Fe2+ → Fe3+ oxidation — required for iron loading onto apotransferrin — GPI-anchored brain isoform — cellular iron egress enabler",
        "key_alerts": [
            "CP-TRIAD-RETINA-NEURO-DIABETES-PATHOGNOMONIC: Retinal degeneration + neurological symptoms (blepharospasm/ataxia/dementia) + diabetes + undetectable serum ceruloplasmin = aceruloplasminemia PATHOGNOMONIC triad; when all three present, diagnosis is clinically certain — confirm by CP gene sequencing; MRI brain T2 hypointensity in basal ganglia/thalamus is the imaging signature",
            "CP-LOW-TS-HIGH-FERRITIN-PARADOX: Iron trapped in cells → serum iron LOW or normal → TS LOW or normal (NOT elevated) → iron overload MISSED if only TS-based screening used; measure SERUM CERULOPLASMIN in any unexplained iron overload + neurological/retinal/diabetes triad; undetectable ceruloplasmin + high ferritin + low TS = aceruloplasminemia",
            "CP-DEFEROXAMINE-BRAIN-IRON: Deferoxamine removes brain iron effectively in aceruloplasminemia (crosses blood-brain barrier at therapeutic doses); MRI T2 signal in basal ganglia improves with treatment; neurological progression halted or slowed but established cognitive damage is not reversed — EARLY TREATMENT CRITICAL",
            "CP-COPPER-SUPPLEMENTATION-FUTILE: Do NOT give copper supplementation to treat aceruloplasminemia; the ceruloplasmin apoprotein (the protein scaffold) is absent — giving copper cannot create ceruloplasmin without the protein; copper supplementation has no therapeutic role; differentiate from Menkes disease (where copper transport to ceruloplasmin is deficient but CP gene is normal)",
            "CP-MRI-T2-HYPOINTENSITY-SIGNATURE: T2/T2* hypointensity (dark signal) in basal ganglia, thalamus, dentate nucleus, and caudate on brain MRI = paramagnetic iron deposition = aceruloplasminemia or neurodegeneration with brain iron accumulation (NBIA); differentiate by serum ceruloplasmin level and ferritin; liver MRI R2* also shows iron overload",
        ],
        "etiologies": [
            "CP nonsense/frameshift biallelic — absent ceruloplasmin protein → no ferroxidase activity → Fe2+ cannot be oxidised to Fe3+ for transferrin loading → iron trapped in cells → retinal/neuronal/hepatic/pancreatic iron accumulation",
            "CP large deletion biallelic — partial or complete gene deletion → absent mRNA → no ceruloplasmin protein → complete ferroxidase deficiency → progressive iron deposition in brain/retina/liver/pancreas",
            "CP missense abolishing copper coordination/ferroxidase activity (e.g. disrupting T1/T2/T3 copper binding sites) — ceruloplasmin protein produced but enzymatically inactive → clinically identical to null allele",
        ],
        "stats": {
            "mean_dx_age": 48,
            "mean_dx_delay_months": 96,
            "ferritin_at_dx_mean_ugL": 4200,
            "ts_at_dx_pct": 22,
            "serum_ceruloplasmin_gL": 0.0,
            "diabetes_pct": 95,
            "retinal_degeneration_pct": 85,
            "neurological_pct": 78,
        },
        "dx_delay_distribution": "72–144 months (neurological + retinal presentations seen by neurology + ophthalmology separately; systemic iron connection delayed; undetectable ceruloplasmin not routinely checked; Japanese overrepresentation in literature creates under-awareness in Western centres)",
    },

    # ── FTL — Hereditary Hyperferritinaemia-Cataract Syndrome ────────────────
    {
        "gene": "FTL",
        "protein": "FTL — HHCS AD — Ferritin-Light-Chain 5UTR-IRE-GOF — Massive-Ferritin-3000-10000-ugL NORMAL-Iron-Stores — Nuclear-Cataracts-Childhood — Phlebotomy-DANGEROUS-CI — Surgical-Cataract-Only-Treatment",
        "alias": (
            "FTL (ferritin light chain); OMIM gene 134790; "
            "Hereditary Hyperferritinaemia-Cataract Syndrome (HHCS) OMIM 600886. "
            "19q13.33; 175 aa; ~20 kDa; AD — 5'UTR gain-of-function mutations. "
            "FUNCTION: Ferritin is the major intracellular iron storage protein, composed of 24 subunits of "
            "heavy (FTH1) and light (FTL) chains in variable ratios. "
            "IRON RESPONSE ELEMENT (IRE) REGULATION: "
            "FTL mRNA contains a 5'UTR iron response element (IRE) — a stem-loop structure; "
            "when cellular iron is LOW: IRP1/IRP2 (iron regulatory proteins) bind the IRE → "
            "steric block of FTL mRNA translation by ribosomes → ferritin NOT made (conserve iron); "
            "when cellular iron is HIGH: iron-loaded IRP1 (Fe-S cluster protein) cannot bind IRE; "
            "IRP2 ubiquitinated and degraded → IRE free → ribosome translates FTL mRNA → ferritin made (store iron). "
            "HHCS PATHOGENESIS — IRE GAIN-OF-FUNCTION: "
            "Mutations IN THE 5'UTR IRE region of FTL mRNA → IRE loses its stem-loop structure → "
            "IRP1/IRP2 CANNOT bind the mutant IRE even when iron is low → "
            "FTL mRNA is CONSTITUTIVELY translated regardless of cellular iron status → "
            "MASSIVE ferritin L-chain overproduction → ferritin accumulates everywhere including: "
            "LENS (crystalline lens cells) → L-chain ferritin aggregates → NUCLEAR CATARACTS in childhood/adolescence; "
            "SERUM (very high circulating ferritin 1000–10,000+ µg/L); "
            "CRITICALLY: Ferritin L-chain excess is NOT related to iron overload — "
            "iron stores are NORMAL or even low; L-chain ferritin is made without iron. "
            "IRE VARIANTS: Deletion of IRE loop, C-to-G transversions in the IRE stem, A-to-G transitions → "
            "any change that disrupts IRE stem-loop structure = GOF → constitutive FTL translation; "
            "40+ IRE variants identified; genotype correlates with age of cataract onset and ferritin level. "
            "CLINICAL PRESENTATION: "
            "Bilateral nuclear cataracts in childhood/adolescence (age 5–25 typically) — PATHOGNOMONIC; "
            "serum ferritin 1000–10,000 µg/L — MASSIVELY elevated; "
            "TRANSFERRIN SATURATION: NORMAL or LOW (iron stores normal); "
            "no liver disease; no diabetes; no heart disease; no arthropathy; "
            "completely ASYMPTOMATIC except for cataracts. "
            "CRITICAL MANAGEMENT RULE — PHLEBOTOMY IS CONTRAINDICATED: "
            "Phlebotomy in HHCS removes iron → iron deficiency anaemia → "
            "IRP1/IRP2 now even more active → try to bind FTL IRE → but mutant IRE cannot be bound → "
            "ferritin still overproduced → cataracts worsen → patient now iron-deficient AND has cataracts; "
            "HHCS patients referred to hematology with 'hyperferritinaemia' have historically been phlebotomised → "
            "iron deficiency developed without any reduction in ferritin or cataracts; "
            "CORRECT MANAGEMENT: Reassure patient (condition is benign for life except cataracts); "
            "CATARACT SURGERY (phacoemulsification with IOL implantation) is the ONLY treatment; "
            "no systemic medication required; no dietary restriction; lifelong ophthalmology follow-up; "
            "genetic testing confirms FTL 5'UTR variant → prevent phlebotomy in proband + relatives."
        ),
        "locus": "19q13.33",
        "aa": 175,
        "kDa": 20,
        "omim_gene": "134790",
        "omim_disease": "Hereditary Hyperferritinaemia-Cataract Syndrome HHCS OMIM 600886",
        "inheritance": "AD — 5'UTR IRE GOF mutation in FTL mRNA; single allele sufficient (dominant); penetrance near complete for cataracts; 50% first-degree relatives affected",
        "gene_class": "Ferritin light chain subunit — 24-subunit iron storage complex — 5'UTR mRNA contains iron response element (IRE) — IRP1/IRP2-regulated translation — HHCS mutations abolish IRP binding → constitutive overproduction",
        "key_alerts": [
            "FTL-PHLEBOTOMY-ABSOLUTELY-CONTRAINDICATED: Phlebotomy in HHCS causes iron deficiency anaemia WITHOUT reducing ferritin or improving cataracts — ferritin overproduction is constitutive (IRE-mutation driven) not iron-driven; every HHCS patient referred with 'hyperferritinaemia' MUST have HHCS excluded before any phlebotomy is performed; the correct diagnostic test is TRANSFERRIN SATURATION (normal/low in HHCS)",
            "FTL-TS-NORMAL-CATARACTS-KEY-DDx: Massively elevated ferritin (>1000 µg/L) + NORMAL transferrin saturation + bilateral nuclear cataracts + family history = HHCS until proven otherwise; request FTL 5'UTR IRE sequencing (standard WES/panel may miss non-coding variants — specify 5'UTR analysis required); cataracts are the only clinical finding beyond the ferritin elevation",
            "FTL-CATARACTS-CHILDHOOD-PATHOGNOMONIC: Bilateral nuclear cataracts in a child/adolescent with family history of high ferritin = HHCS PATHOGNOMONIC combination; ophthalmology must communicate with haematology/genetics when nuclear cataracts found in young patients + reported high ferritin; cataract surgery is indicated and curative for visual symptoms",
            "FTL-IRE-NON-CODING-SEQUENCING: FTL HHCS mutations are in the 5'UTR (untranslated region) — standard coding-region exon sequencing WILL MISS THEM; specifically request FTL 5'UTR IRE sequencing; most clinical NGS panels do not capture 5'UTR by default; Sanger sequencing of the IRE region is the definitive test",
            "FTL-BENIGN-PROGNOSIS-EXCEPT-CATARACTS: HHCS carries a completely normal life expectancy with no systemic iron overload consequences; liver, heart, joints, endocrine organs are all normal; the ONLY morbidity is progressive cataracts; reassure patients that the high ferritin number is not dangerous (unlike HFE/HJV HH where high ferritin = organ damage)",
        ],
        "etiologies": [
            "FTL 5'UTR IRE deletion/point mutation (e.g. c.-168A>G, c.-160C>T, c.-167_-164delCTGG) — disrupts stem-loop structure → IRP1/IRP2 cannot bind → constitutive FTL mRNA translation → massive L-chain ferritin production → lens nuclear cataracts + hyperferritinaemia",
            "FTL 5'UTR IRE stem mutation — base-pair substitution in IRE stem impairs IRP binding → partial or complete loss of iron-sensing regulation → constitutive ferritin overproduction; genotype correlates with ferritin level and cataract onset age",
            "FTL 5'UTR IRE loop mutation — disrupts the specific C-A-G-U-G loop (positions critical for IRP contact) → IRP cannot bind → constitutive translation; loop mutations associated with extreme ferritinaemia (10,000+ µg/L)",
        ],
        "stats": {
            "mean_dx_age": 32,
            "mean_dx_delay_months": 36,
            "ferritin_at_dx_mean_ugL": 5800,
            "ts_at_dx_pct": 23,
            "cataract_surgery_pct": 90,
            "iron_deficiency_from_phlebotomy_pct": 35,
        },
        "dx_delay_distribution": "24–72 months (hyperferritinaemia triggers iron overload workup; phlebotomy often started without diagnosis; cataracts may not be connected to ferritin elevation by non-specialist; 5'UTR IRE sequencing often not performed without explicit suspicion)",
    },

    # ── BMP6 — BMP6-related Hemochromatosis ─────────────────────────────────
    {
        "gene": "BMP6",
        "protein": "BMP6 — BMP6-HH AR — Bone-Morphogenetic-Protein-6 Main-Hepatic-Hepcidin-Driver SMAD1/5/8 — LOF-Reduced-Hepcidin-Liver-Iron — Rare-fewer50-Cases — Phlebotomy-Treatment — Similar-HFE-HH1-Phenotype",
        "alias": (
            "BMP6 (bone morphogenetic protein 6); OMIM gene 112266; "
            "BMP6-related Hereditary Hemochromatosis — not yet assigned separate HH OMIM number. "
            "6p24.3; 513 aa (propeptide); ~28 kDa (mature active dimer); AR biallelic LOF. "
            "FUNCTION: BMP6 is the principal endogenous activator of hepatic hepcidin transcription. "
            "BMP6-SMAD HEPCIDIN AXIS (master iron regulatory pathway): "
            "Hepatic stellate cells and liver sinusoidal endothelial cells release BMP6 in response to liver iron loading; "
            "BMP6 binds BMPR1A (ALK3) + BMPR1B (ALK6) + BMPR2 receptor complex on hepatocyte surface; "
            "hemojuvelin (HJV) acts as GPI-anchored co-receptor that dramatically amplifies BMP6 → BMPR signalling; "
            "receptor activation → intracellular SMAD1, SMAD5, SMAD8 phosphorylation → "
            "pSMAD1/5/8 complex with SMAD4 → nuclear translocation → "
            "binds BMP response elements (BREs) in HAMP (hepcidin) promoter → HAMP transcription; "
            "hepcidin secreted → inhibits ferroportin → reduces iron absorption/release. "
            "BMP6 LOF → reduced BMP6-SMAD signalling → insufficient hepcidin for iron status → "
            "iron absorption not appropriately downregulated → hepatic iron accumulation. "
            "MOLECULAR CHARACTERISTICS: "
            "BMP6 is a TGF-β superfamily member; secreted as a homodimer; "
            "unique among BMPs: regulated specifically by hepatic iron loading; "
            "TMPRSS6 (matriptase-2) cleaves HJV (membrane-bound) → soluble HJV → "
            "released HJV competitively inhibits BMP6-HJV signalling → reduces hepcidin (opposite effect to BMP6 LOF); "
            "BMP6 gene expression upregulated by liver iron → feedback loop for hepcidin maintenance. "
            "EPIDEMIOLOGY: "
            "Rarest hereditary hemochromatosis subtype — fewer than 50 confirmed biallelic BMP6 LOF cases worldwide (as of 2024); "
            "identified by whole-exome/genome sequencing in unexplained iron overload; "
            "no ethnic founder effect established yet; "
            "cases reported in European and Asian populations. "
            "CLINICAL PHENOTYPE (similar to HFE HH1): "
            "Adult-onset hepatic iron overload (typically age 30–55); "
            "transferrin saturation elevated (>45–60%); ferritin elevated (500–4000 µg/L); "
            "liver fibrosis risk with prolonged untreated iron loading; "
            "NOT as severe as JH (HJV/HAMP) — residual hepcidin from other stimuli (inflammation/IL-6) preserved; "
            "cardiac and endocrine complications rare at diagnosis (earlier presentation of biochemistry than HFE). "
            "DIAGNOSIS: "
            "All other HH genes NEGATIVE (HFE/HJV/HAMP/TFR2/SLC40A1 normal); "
            "whole-exome sequencing identifies BMP6 biallelic variants; "
            "BMP6 not on most iron overload panels — requires WES or expanded panel. "
            "TREATMENT: "
            "Phlebotomy identical to HFE HH1 — weekly until ferritin <50 µg/L; "
            "maintenance phlebotomy; cascade first-degree relative screening (biallelic → carrier parents); "
            "prognosis excellent with early treatment."
        ),
        "locus": "6p24.3",
        "aa": 513,
        "kDa": 28,
        "omim_gene": "112266",
        "omim_disease": "BMP6-related Hereditary Hemochromatosis — no separate OMIM disease number; part of the emerging HH spectrum",
        "inheritance": "AR biallelic LOF — rarest HH subtype (<50 cases worldwide); carrier parents unaffected; 25% sibling recurrence risk",
        "gene_class": "TGF-β superfamily bone morphogenetic protein — primary hepatic hepcidin transcription driver — BMPR2/ALK3/HJV receptor complex ligand — SMAD1/5/8 phosphorylation signal — pSMAD4 nuclear complex → HAMP BRE activation",
        "key_alerts": [
            "BMP6-WES-REQUIRED-FOR-DIAGNOSIS: BMP6-related HH is NOT detected by standard HH gene panels (HFE/HJV/HAMP/TFR2/SLC40A1); when all standard HH genes are normal but phenotype is classic (elevated TS + ferritin + liver iron on MRI), request whole-exome sequencing with iron pathway analysis; BMP6 is the most recently characterised HH gene and remains absent from many clinical panels",
            "BMP6-HH1-SIMILAR-PHENOTYPE-PHLEBOTOMY: BMP6 HH phenotype resembles HFE HH1 — adult onset, elevated TS, hepatic iron; respond to standard phlebotomy (weekly venesection to ferritin <50 µg/L); prognosis with early treatment is excellent; do not over-treat (no cardiac complications at typical diagnosis stage unlike JH)",
            "BMP6-SMAD-PATHWAY-EDUCATIONAL: BMP6 → HJV-BMPR → SMAD1/5/8 → HAMP; this is the same pathway disrupted by HJV mutations (type 2A JH) but at a different point; BMP6 LOF = reduced ligand available; HJV LOF = reduced co-receptor (amplifier); both result in insufficient hepcidin; BMP6 has more residual hepcidin (other BMPs partially compensate) → milder phenotype than HJV",
            "BMP6-ERYTHROFERRONE-AXIS: In iron-deficiency anaemia or thalassaemia, erythroferrone (ERFE, secreted by erythroblasts) inhibits BMP6 → suppresses hepcidin → mobilises iron for erythropoiesis; BMP6 LOF blunts this adaptive response because baseline hepcidin is already low → BMP6 patients may not develop appropriate anaemia-adaptive hepcidin suppression",
            "BMP6-RARE-REGISTRY-REFERRAL: Fewer than 50 confirmed cases worldwide; refer to international rare iron disorder registries (EURO-HEMOCHROMATOSIS; rare disease network); publication of novel BMP6 variants benefits global understanding; genetic counselling for carrier testing in siblings of biallelic probands essential",
        ],
        "etiologies": [
            "BMP6 biallelic LOF (nonsense/frameshift) — absent BMP6 protein → no BMP6 available for hepatocyte BMPR-SMAD signalling → HAMP promoter receives insufficient SMAD1/5/8 activation → hepcidin deficiency relative to iron stores → hepatic iron loading",
            "BMP6 biallelic missense disrupting BMPR binding — mature BMP6 dimer cannot bind BMPR2/ALK3 receptor complex → no downstream SMAD phosphorylation → hepcidin transcription impaired → iron overload (severity depends on residual receptor binding affinity)",
            "BMP6 compound heterozygous (one null + one missense) — partial BMP6 function from missense allele → moderate hepcidin deficiency → adult-onset HH with variable severity depending on residual BMP6-SMAD signalling from missense allele",
        ],
        "stats": {
            "mean_dx_age": 44,
            "mean_dx_delay_months": 90,
            "ferritin_at_dx_mean_ugL": 1600,
            "ts_at_dx_pct": 58,
            "phlebotomy_induction_sessions_mean": 30,
            "cirrhosis_at_dx_pct": 8,
        },
        "dx_delay_distribution": "60–144 months (extreme rarity; HFE negative after standard panel → extended workup delayed or abandoned; WES not routinely ordered for iron overload unless index of clinical suspicion maintained after negative standard panel)",
    },
]


def _generate_patients():
    """Generate 40 patients per gene using deterministic RNG (seed = SEED_BASE + gene_idx)."""
    for idx, gene in enumerate(HEMOCHROMATOSIS_GENES):
        rng = random.Random(SEED_BASE + idx)
        patients = []
        for i in range(40):
            age_at_dx = rng.randint(15, 68)
            dx_delay = rng.randint(
                int(gene["stats"].get("mean_dx_delay_months", 48) * 0.4),
                int(gene["stats"].get("mean_dx_delay_months", 48) * 1.8),
            )
            # Gene-specific iron parameter ranges
            if gene["gene"] == "HFE":
                ferritin = rng.randint(400, 4500)
                ts = rng.randint(48, 95)
                serum_iron = rng.randint(28, 42)
            elif gene["gene"] == "HJV":
                ferritin = rng.randint(3000, 18000)
                ts = rng.randint(70, 98)
                serum_iron = rng.randint(35, 48)
                age_at_dx = rng.randint(14, 32)
            elif gene["gene"] == "HAMP":
                ferritin = rng.randint(4000, 20000)
                ts = rng.randint(75, 98)
                serum_iron = rng.randint(36, 50)
                age_at_dx = rng.randint(13, 28)
            elif gene["gene"] == "TFR2":
                ferritin = rng.randint(600, 5000)
                ts = rng.randint(50, 90)
                serum_iron = rng.randint(28, 44)
            elif gene["gene"] == "SLC40A1":
                # 65% type 4A (TS low), 35% type 4B (TS high)
                type4b = rng.random() > 0.65
                ferritin = rng.randint(800, 6000)
                ts = rng.randint(55, 85) if type4b else rng.randint(15, 38)
                serum_iron = rng.randint(10, 20) if not type4b else rng.randint(28, 42)
            elif gene["gene"] == "CP":
                ferritin = rng.randint(1500, 8000)
                ts = rng.randint(8, 32)  # LOW TS — iron trapped in cells
                serum_iron = rng.randint(6, 18)
                age_at_dx = rng.randint(35, 65)
            elif gene["gene"] == "FTL":
                ferritin = rng.randint(1200, 12000)
                ts = rng.randint(10, 32)  # NORMAL/LOW TS
                serum_iron = rng.randint(12, 22)
                age_at_dx = rng.randint(18, 50)
            else:  # BMP6
                ferritin = rng.randint(500, 3800)
                ts = rng.randint(45, 82)
                serum_iron = rng.randint(26, 42)

            on_phlebotomy = rng.random() > 0.3
            # FTL phlebotomy should be avoided; simulate proportion that received wrong treatment
            if gene["gene"] == "FTL":
                on_phlebotomy = rng.random() > 0.6  # many were incorrectly started on phlebotomy
            cirrhosis = rng.random() > 0.85 if gene["gene"] in ["HFE", "TFR2", "BMP6"] else (
                rng.random() > 0.7 if gene["gene"] in ["HJV", "HAMP"] else rng.random() > 0.95)
            diabetes = rng.random() > 0.75 if gene["gene"] == "CP" else (
                rng.random() > 0.6 if gene["gene"] in ["HJV", "HAMP"] else rng.random() > 0.85)
            hypogonadism = rng.random() > 0.2 if gene["gene"] in ["HJV", "HAMP"] else rng.random() > 0.9
            retinal_changes = rng.random() > 0.15 if gene["gene"] == "CP" else rng.random() > 0.97
            cataract = rng.random() > 0.1 if gene["gene"] == "FTL" else rng.random() > 0.97
            arthropathy = rng.random() > 0.55 if gene["gene"] == "HFE" else rng.random() > 0.75

            patients.append({
                "patient_id": f"{gene['gene']}-{i+1:03d}",
                "age_at_dx": age_at_dx,
                "dx_delay_months": dx_delay,
                "ferritin_ugL": ferritin,
                "transferrin_saturation_pct": ts,
                "serum_iron_umolL": serum_iron,
                "on_phlebotomy": on_phlebotomy,
                "cirrhosis_present": cirrhosis,
                "diabetes_present": diabetes,
                "hypogonadism_present": hypogonadism,
                "retinal_changes": retinal_changes,
                "cataract_present": cataract,
                "arthropathy_present": arthropathy,
                "gene": gene["gene"],
                "seed": SEED_BASE + idx,
            })
        gene["patients"] = patients


_generate_patients()


def get_overview():
    all_ages = [p["age_at_dx"] for g in HEMOCHROMATOSIS_GENES for p in g["patients"]]
    all_delays = [p["dx_delay_months"] for g in HEMOCHROMATOSIS_GENES for p in g["patients"]]
    genes = []
    for idx, g in enumerate(HEMOCHROMATOSIS_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        genes.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
            "key_alerts": g["key_alerts"],
            "n_patients": len(g["patients"]),
        })
    return {
        "atlas": "Hereditary-Hemochromatosis-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Iron-Overload Atlas — Hemochromatosis & Iron Dysregulation (HH1/JH2A/JH2B/HH3/HH4/Aceruloplasminemia/HHCS/BMP6-HH)",
        "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
        "total_patients": sum(len(g["patients"]) for g in HEMOCHROMATOSIS_GENES),
        "aggregate_stats": {
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "genes_covered": len(HEMOCHROMATOSIS_GENES),
            "patients_per_gene": 40,
        },
        "genes": genes,
        "top_alerts": [
            "HFE-C282Y-PHLEBOTOMY-CURATIVE-BEFORE-CIRRHOSIS: HFE C282Y/C282Y detected before cirrhosis → weekly phlebotomy → normal life expectancy; TS >45% fasting morning screen triggers HFE genotyping; raw shellfish absolutely contraindicated (Vibrio vulnificus septicaemia); cascade all first-degree relatives",
            "HJV-HAMP-CARDIAC-MRI-T2STAR-URGENT: Juvenile HH (HJV/HAMP biallelic) presents age 15–30 with cardiomyopathy + hypogonadism; cardiac MRI T2* <20 ms = urgent chelation (deferoxamine) + phlebotomy; hepcidin undetectable; both sexes equally affected (no menstrual protection in JH)",
            "SLC40A1-TS-PATTERN-DDx-4A-vs-4B: Only AD iron overload — TS NORMAL/LOW in Type 4A (macrophage iron trapping = ferroportin LOF haploinsufficiency) vs TS ELEVATED in Type 4B (GOF hepcidin-resistant); Type 4A phlebotomy must be cautious (anaemia risk); Type 4B treated like HFE HH1",
            "CP-TRIAD-PATHOGNOMONIC-DEFEROXAMINE: Aceruloplasminemia triad: retinal degeneration + neurological (blepharospasm/ataxia/dementia) + diabetes + undetectable ceruloplasmin; TS LOW despite iron overload (iron trapped in cells); deferoxamine chelation arrests neurological progression; copper supplementation is futile",
            "FTL-HHCS-PHLEBOTOMY-CI-CATARACTS-ONLY: HHCS: massive ferritin (1000–12000 µg/L) + NORMAL TS + bilateral nuclear cataracts + NO iron overload; phlebotomy is CONTRAINDICATED (causes iron deficiency, ferritin stays high); FTL 5'UTR IRE sequencing required (non-coding — missed by standard exon panels); cataract surgery is the only treatment",
            "BMP6-WES-FOR-HFE-NEGATIVE-HH: BMP6-related HH — rarest subtype; standard HH panels do not include BMP6; HFE/HJV/HAMP/TFR2/SLC40A1 all negative + classic HH biochemistry → WES with iron pathway analysis; phenotype similar to HFE HH1; phlebotomy curative",
            "TFR2-HH3-WORLDWIDE-NEXT-AFTER-HFE-NEGATIVE: TFR2 HH3 is most common non-HFE HH in Europeans; HFE negative HH → TFR2 biallelic sequencing first; no founder; worldwide distribution; phenotype/treatment identical to HFE; Y250X European most characterised variant",
            "HAMP-HEPCIDIN-DIRECT-LOF-RAREST: HAMP biallelic LOF = hepcidin itself absent; BMP-SMAD pathway intact (contrast HJV); <30 families worldwide; urinary hepcidin undetectable distinguishes JH2B from JH2A when genotyping inconclusive; future hepcidin replacement (rusfertide) may be curative",
        ],
    }


def get_breakdown():
    result = []
    for idx, g in enumerate(HEMOCHROMATOSIS_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
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
            "computed": {
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def get_definitions():
    return {
        "concepts": {
            "Hepcidin-Ferroportin Axis — The Master Regulator of Iron Homeostasis": (
                "The hepcidin-ferroportin axis is the central regulatory circuit controlling systemic iron. "
                "FERROPORTIN (SLC40A1) IS THE SOLE IRON EXPORTER: Every cell that releases iron into the bloodstream "
                "does so exclusively via ferroportin; no alternative cellular iron export pathway exists in vertebrates; "
                "ferroportin is expressed on basolateral enterocyte membranes (exports absorbed dietary iron), "
                "on macrophages (exports recycled iron from phagocytosed RBCs), "
                "and on hepatocytes (releases stored iron). "
                "HEPCIDIN BINDS FERROPORTIN: Hepcidin (25-aa hepatic peptide) binds ferroportin → "
                "triggers ferroportin ubiquitination → internalisation → lysosomal degradation; "
                "without ferroportin: iron trapped inside enterocytes (shed in faeces), "
                "macrophages (cannot recycle to plasma), hepatocytes (cannot mobilise stores); "
                "net effect: hepcidin lowers serum iron + reduces absorption. "
                "HH PATHOPHYSIOLOGY: In all classic HH subtypes (HFE/HJV/HAMP/TFR2/BMP6 — types 1/2A/2B/3/BMP6): "
                "hepcidin is deficient relative to iron stores → ferroportin constitutively active → "
                "maximal duodenal iron absorption + macrophage iron release → "
                "iron floods plasma → deposited in parenchymal organs (liver/heart/pancreas/pituitary/gonads). "
                "PHLEBOTOMY MECHANISM: Removing blood (haemoglobin) depletes iron from circulation → "
                "triggers normal erythropoiesis → bone marrow uses stored body iron for new RBC haem synthesis → "
                "net iron removal from body; hepcidin system not required for phlebotomy to work → "
                "phlebotomy is effective regardless of which HH gene is mutated."
            ),
            "BMP6-HJV-SMAD-Hepcidin Pathway — Iron-Sensing Signal Cascade": (
                "The BMP6-HJV-SMAD pathway is the primary iron-sensing hepcidin regulatory axis. "
                "BMP6 PRODUCTION: Hepatic stellate cells and sinusoidal endothelial cells secrete BMP6 "
                "in proportion to liver iron loading; BMP6 is the principal endogenous hepcidin activator. "
                "HJV CO-RECEPTOR: Hemojuvelin (HJV, GPI-anchored on hepatocyte surface) dramatically amplifies "
                "BMP6 signalling by stabilising BMP-receptor complexes; "
                "HJV binds BMP6 → presents it to BMPR2/ALK3 → 10–100× increased SMAD activation vs BMP6 alone. "
                "SMAD SIGNALLING: BMP6-HJV-BMPR → SMAD1+SMAD5+SMAD8 phosphorylated → "
                "complex with SMAD4 → nuclear translocation → "
                "binds BMP response elements (BRE) in HAMP promoter → hepcidin transcription. "
                "PATHWAY DISRUPTION AT DIFFERENT NODES: "
                "BMP6 LOF → reduced ligand (BMP6-HH); HJV LOF → reduced co-receptor amplifier (JH2A); "
                "HAMP LOF → absent hepcidin despite intact upstream signalling (JH2B); "
                "HFE/TFR2 LOF → impaired holotransferrin sensing → reduced BMP-SMAD input (HH1/HH3). "
                "SMAD INHIBITORS: TMPRSS6 (matriptase-2) cleaves membrane HJV → soluble HJV "
                "→ competitively inhibits BMP6-HJV signalling → reduces hepcidin (adaptive in iron deficiency); "
                "hypoxia/EPO → erythroferrone (ERFE) → inhibits BMP6 → reduces hepcidin (mobilises iron for RBCs)."
            ),
            "Ferroportin Disease Type 4 — Distinguishing 4A (LOF) from 4B (GOF)": (
                "Ferroportin disease is the ONLY autosomal dominant hereditary hemochromatosis "
                "(all other HH types are AR). "
                "TYPE 4A (LOF — Classical Ferroportin Disease): "
                "Haploinsufficiency → 50% normal ferroportin at cell surface → "
                "iron export reduced from enterocytes AND macrophages → "
                "MACROPHAGE IRON LOADING (reticuloendothelial predominance); "
                "high ferritin (macrophage-derived) but transferrin saturation NORMAL/LOW "
                "(iron not efficiently entering plasma for transferrin loading); "
                "phlebotomy tolerance POOR (iron not available from macrophages for bone marrow erythropoiesis → "
                "anaemia develops rapidly with phlebotomy); "
                "ferritin target more lenient: <200 µg/L to avoid anaemia. "
                "TYPE 4B (GOF — Hepcidin-Resistant Ferroportin): "
                "Ferroportin mutation at hepcidin-binding interface (C326S/Y, N144H/D/T) → "
                "hepcidin cannot internalise ferroportin → ferroportin constitutively open → "
                "HEPATOCYTE PARENCHYMAL iron loading (like HFE/HJV/HAMP/TFR2); "
                "TS elevated (ferroportin exports iron freely from gut → plasma iron high → transferrin saturated); "
                "phlebotomy well-tolerated (same as HFE HH1). "
                "DIAGNOSTIC ALGORITHM FOR HH4: "
                "Step 1: HFE genotype + TS + ferritin; "
                "Step 2: If SLC40A1 variant identified → check TS level; "
                "Step 3: TS NORMAL/LOW + SLC40A1 LOF variant → Type 4A (cautious phlebotomy); "
                "Step 4: TS ELEVATED + SLC40A1 GOF hepcidin-interface variant → Type 4B (standard phlebotomy). "
                "FUNCTIONAL ASSAY: Express mutant SLC40A1 + hepcidin challenge → "
                "does ferroportin internalise? Yes = LOF 4A. No = GOF 4B."
            ),
            "Aceruloplasminemia — Ceruloplasmin Ferroxidase and Neurological Iron Overload": (
                "Aceruloplasminemia illustrates how cellular iron export depends on oxidation, not just transport. "
                "CERULOPLASMIN FERROXIDASE: CP contains six copper atoms in three pairs (T1/T2/T3 sites); "
                "T1 sites accept electrons from Fe2+ (oxidising it to Fe3+); electrons shuttled to T2/T3 → "
                "oxygen reduced to water; net reaction: 4 Fe2+ + O2 + 4H+ → 4 Fe3+ + 2H2O; "
                "Fe3+ released into plasma can now bind apotransferrin (transferrin only binds Fe3+). "
                "GPI-CERULOPLASMIN (BRAIN-SPECIFIC): "
                "Neurons and astrocytes express GPI-anchored ceruloplasmin on their cell surface; "
                "GPI-CP oxidises intracellular Fe2+ at cell surface → Fe3+ loaded onto transferrin in interstitial fluid → "
                "iron exported from brain cells; absent GPI-CP → iron trapped in neurons/astrocytes → "
                "oxidative stress (Fe2+ + H2O2 → Fenton reaction → OH radical → lipid peroxidation → neurodegeneration). "
                "WHY TS IS LOW IN ACERULOPLASMINEMIA: "
                "Iron is trapped INSIDE cells (enterocytes/macrophages/hepatocytes) because it cannot be oxidised "
                "and loaded onto transferrin; plasma iron FALLS; apotransferrin circulates unsaturated → "
                "TS LOW (paradoxically, despite massive body iron stores); "
                "standard HH screening (TS >45%) WILL NOT DETECT aceruloplasminemia. "
                "IRON MRI IN ACERULOPLASMINEMIA: "
                "T2/T2* signal hypointense (dark) in: basal ganglia (putamen/caudate > globus pallidus), "
                "thalamus, dentate nucleus, red nucleus, subthalamic nucleus, cortex; "
                "liver T2* also dark (hepatic iron overload); "
                "this pattern = aceruloplasminemia or NBIA (neurodegeneration with brain iron accumulation)."
            ),
            "HHCS — IRE Biology and Why Phlebotomy Is Contraindicated": (
                "Hereditary Hyperferritinaemia-Cataract Syndrome (HHCS) is caused by mutations in "
                "the 5'UTR iron response element (IRE) of FTL mRNA — a non-coding, non-protein-changing variant. "
                "IRE BIOLOGY: The 5'UTR IRE is a 28-nucleotide stem-loop structure; "
                "STEM: base-paired nucleotides forming the double-stranded stem; "
                "LOOP: single-stranded CAGUG pentanucleotide loop (essential IRP contact surface); "
                "when cellular iron is low → IRP1 lacks Fe-S cluster → IRP1 binds IRE → blocks ribosome entry → "
                "FTL mRNA NOT translated (saves iron by not making ferritin storage protein). "
                "IRE MUTATION EFFECT: Any mutation disrupting the IRE stem or loop → "
                "IRP1/IRP2 cannot bind → ribosome accesses mRNA → FTL constitutively translated → "
                "massive L-chain ferritin made → accumulates in ALL tissues including LENS → "
                "ferritin aggregates in lens crystalline proteins → NUCLEAR CATARACTS; "
                "CRUCIALLY: all this extra ferritin is made without iron (apo-ferritin = iron-free ferritin); "
                "body iron stores are NORMAL because no extra iron is absorbed. "
                "WHY PHLEBOTOMY FAILS AND HARMS: "
                "Phlebotomy removes RBCs → haemoglobin iron removed → cellular iron falls → "
                "IRP1 activates (low iron → IRP1 lost Fe-S cluster) → tries to bind IRE → "
                "mutant IRE STILL cannot be bound → FTL translation continues unimpeded → "
                "ferritin STAYS HIGH; "
                "meanwhile RBC haemoglobin iron removed → bone marrow needs iron → "
                "body becomes iron-deficient → IRP1 even more active → erythropoiesis impaired → "
                "iron deficiency anaemia on top of ongoing cataracts. "
                "CORRECT TREATMENT: "
                "Reassure: HHCS is NOT iron overload → no organ damage except lens; "
                "cataract surgery (phacoemulsification + IOL) whenever visual acuity impaired; "
                "no dietary restriction; no chelation; no phlebotomy; "
                "lifelong ophthalmology: cataracts progress slowly; posterior capsule opacity post-surgery requires YAG laser."
            ),
        },
        "pharmacological_distinctions": [
            "Phlebotomy vs chelation in hemochromatosis: Phlebotomy (venesection) is first-line for ALL types of HH where phlebotomy is tolerated (HFE/HJV/HAMP/TFR2/SLC40A1 type 4B/BMP6); each 500 mL removes ~250 mg iron; cheap, safe, effective; chelation (deferoxamine/deferasirox/deferiprone) reserved for: severe cardiac iron (JH — deferoxamine SC/IV), anaemia precluding phlebotomy (Type 4A), aceruloplasminemia (phlebotomy not possible — iron trapped in cells); oral deferasirox (Exjade) has modest data in HFE HH",
            "Deferoxamine vs deferasirox in iron overload: Deferoxamine (DFO): IV/SC, short t½ → continuous infusion for cardiac iron emergency (JH2A/JH2B with T2* <10 ms); deferasirox (DFX): oral once-daily, t½ 8–16 hrs → maintenance chelation in transfusion-dependent siderosis; deferiprone (DFP): oral three times daily, penetrates blood-brain barrier → may benefit neurological iron overload (aceruloplasminemia); combining DFP + DFO (shuttle chelation) in severe cardiac iron overload (thalassaemia major / JH) achieves greater cardiac iron removal than either alone",
            "Phlebotomy in Type 4A ferroportin disease vs Type 4B: Type 4A (LOF): macrophage iron predominates; RBC production iron-limited by macrophage trapping; aggressive phlebotomy → rapid anaemia (haemoglobin-iron removed faster than macrophage can release iron to replenish erythropoiesis); use 200–300 mL reduced volume every 4–8 weeks; ferritin target ≤200 µg/L; Type 4B (GOF): hepatocyte parenchymal iron predominates; freely available iron for erythropoiesis; standard 500 mL weekly phlebotomy tolerated; ferritin target <50 µg/L",
            "Erythrocytapheresis vs simple phlebotomy: Erythrocytapheresis (automated red cell removal): removes 2× the RBC volume vs manual phlebotomy in same session → twice the iron removed per procedure; expensive and requires apheresis equipment; suitable for rapid iron depletion in young HH patients with high iron burden (HJV/HAMP JH) where frequent visits are impractical; simple phlebotomy remains standard due to cost and access",
            "Iron restriction diet in hemochromatosis: Avoidance of haem-iron rich foods (red meat) modestly reduces iron absorption; avoidance of vitamin C supplements (ascorbate enhances non-haem iron absorption 3–6 fold) is more impactful; avoidance of raw shellfish (Vibrio vulnificus risk) is the highest-priority dietary counselling; alcohol avoidance (potentiates hepatic iron toxicity, accelerates fibrosis, HCC risk); tea (tannins) and coffee (chlorogenic acid) reduce iron absorption — may be recommended; red wine at meals is neutral (polyphenols also inhibit iron absorption); iron cooking pots negligible contribution",
        ],
        "key_standards": [
            "EASL Clinical Practice Guidelines — Hemochromatosis (Eur J Hepatol 2022): Screening strategy (TS >45% fasting → HFE genotyping); phlebotomy protocols; ferritin targets (<50 µg/L induction; <300 µg/L maintenance; <200 µg/L Type 4A); liver biopsy/MRI thresholds; HCC surveillance; cascade testing; non-HFE HH workup algorithm",
            "British Society of Haematology (BSH) Guidelines for Hereditary Haemochromatosis 2018: Detailed phlebotomy protocols; treatment goals by HH subtype; HCC screening; dietary advice; management in pregnancy; rare HH subtypes (JH/ARH/ferroportin disease) guidance",
            "EuroHemochromatosis Network — Rare HH Subtypes Registry: Collects clinical data on HJV/HAMP/TFR2/SLC40A1/BMP6 HH; enables genotype-phenotype correlation; referral pathway for WES-unexplained HH; collaboration with EuroBloodNet",
            "Bacon 2011 NEJM — Hemochromatosis Review: Pathophysiology, clinical features, genetic testing strategy; phlebotomy efficacy data; HCC risk stratification; landmark review still referenced in clinical practice guidelines",
            "Piperno 2020 — BMP6 Hemochromatosis (Blood): Case series establishing BMP6 biallelic LOF as HH subtype; phenotype characterisation; treatment response; WES methodology for identification; expanded awareness of non-classical HH genes",
        ],
    }
