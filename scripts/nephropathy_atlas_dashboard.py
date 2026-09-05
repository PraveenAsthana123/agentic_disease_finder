#!/usr/bin/env python3
"""Nephropathy-Atlas — Complete 8-Gene Hereditary Nephropathy Atlas
PKD1    (ADPKD type 1; ~4303 aa; 16p13.3; polycystin-1; AD; 1:400–1:1000;
         bilateral renal cysts; intracranial aneurysms — Berry aneurysm screening;
         tolvaptan FDA 2018 slows kidney growth; most common ESRD-causing hereditary disorder) ·
PKD2    (ADPKD type 2; ~968 aa; 4q22.1; polycystin-2 / TRPP2 channel; AD;
         later ESRD vs PKD1 (~10 yrs); milder phenotype; 15% of ADPKD) ·
COL4A5  (Alport syndrome X-linked / AS-XL; ~1454 aa; Xq22.3; collagen IV alpha5; XLR;
         haematuria + SNHL + anterior lenticonus PATHOGNOMONIC; ESRD males 30–40 yrs;
         ACEi/ARBs delay ESRD — start early even pre-proteinuria) ·
COL4A3  (Alport syndrome AR; ~1670 aa; 2q36.3; collagen IV alpha3; AR;
         most severe Alport; ESRD by 20–30 yrs; consanguinity) ·
NPHS1   (Congenital nephrotic syndrome Finnish type / CNF; ~1241 aa; 19q13.12;
         nephrin; AR; massive proteinuria at birth; elevated amniotic AFP; Fin-major+Fin-minor;
         transplant curative BUT 25% recurrence anti-nephrin antibody) ·
NPHS2   (Focal segmental glomerulosclerosis type 1 / FSGS1; ~383 aa; 1q25.2;
         podocin; AR; childhood steroid-resistant nephrotic; R138Q European founder;
         ACEi + cyclosporin/rituximab; low recurrence post-transplant unlike NPHS1) ·
UMOD    (Uromodulin-associated kidney disease / UAKD / FJHN / MCKD2; ~640 aa; 16p12.3;
         uromodulin / Tamm-Horsfall protein; AD; gout teens PATHOGNOMONIC; CKD slow;
         NSAIDs ABSOLUTELY CI; allopurinol for hyperuricemia) ·
HNF1B   (MODY5 / RCAD; ~557 aa; 17q12; hepatocyte nuclear factor 1-beta; AD;
         renal cysts + diabetes + hypomagnesaemia + pancreatic atrophy; 50%+ de novo;
         sulfonylureas INEFFECTIVE — insulin required; genitourinary anomalies in females)
320-patient aggregate cohort (8 × 40, seeds 1158–1165)
"""

import random

SEED_BASE = 1158

NEPHROPATHY_GENES = [
    # ── PKD1 — ADPKD type 1 ─────────────────────────────────────────────────
    {
        "gene": "PKD1",
        "protein": "Polycystin-1 (PC1)",
        "alias": (
            "PKD1; OMIM gene 601313; 16p13.3; ~4303 aa; ADPKD type 1 (OMIM #173900); "
            "AD; prevalence ~1:400–1:1000; largest known human gene (46 exons, ~14.5 kb mRNA); "
            "polycystic kidney disease — bilateral cysts + liver/pancreatic cysts; "
            "ESRD median ~55 yrs; intracranial aneurysms 5–10% (Berry aneurysm); "
            "tolvaptan FDA 2018 slows kidney growth; truncating variants → worse prognosis"
        ),
        "aa": "~4303 aa",
        "kDa": "~460 kDa",
        "gene_class": (
            "Mechanosensory receptor-channel complex (non-selective cation channel); "
            "11 transmembrane domains; REJ (receptor for egg jelly) domain; "
            "GPS cleavage site; primary cilia of renal tubular epithelium; "
            "regulates mTOR, cAMP, Wnt signalling pathways; interacts with PC2 (PKD2) "
            "at the C-terminal coiled-coil domain to form a functional receptor-channel unit"
        ),
        "locus": "16p13.3",
        "omim_gene": 601313,
        "omim_disease": 173900,
        "phenotype": (
            "ADPKD type 1 — bilateral progressive renal cysts → ESRD (~55 yrs); "
            "liver cysts (most common extrarenal, ~83%); pancreatic cysts; "
            "intracranial aneurysms (Berry, 5–10%); mitral valve prolapse (25%); "
            "colonic diverticula; abdominal hernia; hypertension early (RAA activation)"
        ),
        "disease": (
            "PKD1 encodes polycystin-1 (PC1), a large multimodal receptor expressed in "
            "the primary cilia of renal tubular epithelium. "
            "NORMAL FUNCTION: PC1 acts as a mechanosensor (detecting tubular fluid flow); "
            "it forms a receptor-channel complex with PC2 (PKD2/TRPP2) at the C-terminal "
            "coiled-coil domain — together regulating intracellular Ca²⁺, cAMP, mTOR, "
            "and Wnt/β-catenin pathways to control tubular cell proliferation, "
            "differentiation, and polarity. GPS auto-proteolysis generates PC1-NTF + PC1-CTF. "
            "PATHOMECHANISM: LOF variants in PKD1 → loss of PC1 function → "
            "'two-hit' somatic mutation model (germline LOF + somatic second-hit in individual "
            "cyst founder cells) → uncontrolled tubular cell proliferation + fluid secretion "
            "(cAMP-driven Cl⁻ secretion via CFTR) → bilateral cysts enlarge replacing "
            "functional parenchyma → CKD → ESRD. "
            "Activation of mTOR pathway in cyst lining cells → potential therapeutic target "
            "(everolimus trials — failed to improve eGFR despite reducing kidney volume). "
            "MOLECULAR FEATURES: PKD1 is complicated by six PSEUDO-GENES (PKD1P1-P6) on "
            "16p; next-generation sequencing must distinguish PKD1 from pseudogenes "
            "(requires long-range PCR or SMRT long-read sequencing); truncating variants "
            "(frameshift/nonsense/splice) → more severe disease / earlier ESRD than "
            "hypomorphic/missense variants (Cornec-Le Gall classification). "
            "CLINICAL FEATURES: Hypertension in 70–80% before GFR decline (RAA activation "
            "from cyst compression of intrarenal vasculature); gross haematuria (cyst "
            "haemorrhage — usually self-limited; CT to exclude RCC); flank pain (mass, "
            "haemorrhage, infection, nephrolithiasis — 20%); infected cysts (E. coli/Klebsiella — "
            "fluoroquinolones penetrate cysts better than aminoglycosides/beta-lactams). "
            "EXTRARENAL: Liver cysts (83%, more numerous in females — oestrogen-driven); "
            "pancreatic cysts; intracranial aneurysms (5–10% — Berry aneurysm; "
            "risk higher with positive family history of subarachnoid haemorrhage; "
            "screening MRA recommended if ≥1 affected first-degree relative or occupational "
            "risk; aneurysm ≥5mm or growing → intervention); "
            "mitral valve prolapse (25%); colonic diverticula. "
            "TREATMENT: Tolvaptan (V2-receptor antagonist → reduces cAMP-driven cyst growth; "
            "TEMPO 3:4 trial: 49% slower TKV growth, 26% slower eGFR decline; "
            "HEPATOTOXICITY — serious; LFTs monthly for first 18 months; "
            "ABSOLUTELY CI in liver disease / significant alcohol use); "
            "ACEi/ARBs: control BP and reduce proteinuria; "
            "hydration (>3L/day) — suppresses vasopressin (AVP) → reduces cAMP; "
            "dietary protein/sodium restriction; dialysis/transplant for ESRD. "
            "PROGNOSIS: Total kidney volume (TKV) on MRI is the best prognostic marker "
            "(Mayo Imaging Classification 1A-E: 1E = highest growth rate → fastest ESRD)."
        ),
        "inheritance": (
            "AD (autosomal dominant); 50% offspring affected; de novo ~5–10%; "
            "germline penetrance virtually 100% by age 80; "
            "truncating variants → worse prognosis; "
            "mosaicism can cause milder phenotype in apparent de novo cases"
        ),
        "hallmark": (
            "Bilateral renal cysts + hypertension before age 30 + family history; "
            "Mayo TKV classification for prognosis; intracranial Berry aneurysm; "
            "liver cysts in females (oestrogen-driven) most numerous; "
            "PKD1 truncating variant → ESRD ~10 yrs earlier than hypomorphic"
        ),
        "key_ddx": (
            "PKD2-ADPKD (same phenotype, later ESRD — genetic testing distinguishes); "
            "Tuberous sclerosis (TSC1/TSC2 — AML/renal cysts/cortical tubers; PKD1 and TSC2 "
            "are adjacent on 16p13 → contiguous gene syndrome); "
            "Von Hippel-Lindau (VHL — clear cell RCC + haemangioblastoma; not simple cysts); "
            "Acquired cystic kidney disease (ACKD — bilateral cysts in long-standing CKD/dialysis; "
            "no PKD1 mutation; RCC risk 10-fold elevated)"
        ),
        "treatment_alert": (
            "TOLVAPTAN: Serious hepatotoxicity risk — LFTs monthly for 18 months; "
            "ABSOLUTELY CI in liver disease or significant alcohol; "
            "aquaresis (polyuria/polydipsia/nocturia) — must have free water access; "
            "BERRY ANEURYSM: MRA screening if family history SAH or ≥1 affected first-degree relative; "
            "aneurysm ≥5mm → neurosurgical/endovascular assessment; "
            "INFECTED CYSTS: fluoroquinolones (ciprofloxacin/trimethoprim) — lipophilic, "
            "penetrate cysts; beta-lactams DO NOT penetrate — do not use for infected cysts"
        ),
        "seed": 1158,
        "cohort_n": 40,
        "esrd_rate": 0.55,
        "hypertension_rate": 0.78,
        "liver_cyst_rate": 0.83,
        "haematuria_rate": 0.45,
        "aneurysm_rate": 0.08,
        "drug_error_rate": 0.12,
        "diagnosis_delay_rate": 0.35,
        "transplant_rate": 0.30,
        "surveillance_adherent_rate": 0.72,
        "severity_weights": {"Mild": 0.20, "Moderate": 0.45, "Severe": 0.35},
        "primary_complication": "Progressive CKD/ESRD / intracranial aneurysm rupture",
        "gfr_pattern": "Declining from age 30s; ESRD median ~55 yrs (PKD1 truncating ~50 yrs)",
        "proteinuria_pattern": "Mild (< 1g/day typical; severe proteinuria rare)",
    },

    # ── PKD2 — ADPKD type 2 ─────────────────────────────────────────────────
    {
        "gene": "PKD2",
        "protein": "Polycystin-2 (PC2 / TRPP2)",
        "alias": (
            "PKD2; OMIM gene 173910; 4q22.1; ~968 aa; ADPKD type 2 (OMIM #613095); "
            "AD; ~15% of ADPKD cases; polycystin-2 is a TRP (transient receptor potential) channel (TRPP2); "
            "milder phenotype — ESRD ~65–70 yrs (~10 yrs later than PKD1); "
            "intracranial aneurysm risk lower; same tolvaptan treatment pathway"
        ),
        "aa": "~968 aa",
        "kDa": "~110 kDa",
        "gene_class": (
            "Transient receptor potential polycystin-2 (TRPP2) — Ca²⁺-permeable "
            "non-selective cation channel; 6 transmembrane domains; voltage-gated-like "
            "topology; EF-hand Ca²⁺-binding domain in C-terminus; "
            "localises to primary cilia, ER, and lateral plasma membrane; "
            "forms functional complex with PC1 (PKD1) via coiled-coil C-terminal interaction; "
            "Ca²⁺ signalling hub for cell proliferation and mechanosensation"
        ),
        "locus": "4q22.1",
        "omim_gene": 173910,
        "omim_disease": 613095,
        "phenotype": (
            "ADPKD type 2 — bilateral progressive renal cysts → ESRD (~65–70 yrs); "
            "liver cysts (less numerous than PKD1); intracranial aneurysms (lower risk than PKD1); "
            "hypertension (earlier than eGFR decline); haematuria; flank pain; "
            "overall milder disease burden vs PKD1"
        ),
        "disease": (
            "PKD2 encodes polycystin-2 (PC2 / TRPP2), a 6-transmembrane TRP-family "
            "non-selective cation channel that conducts Ca²⁺, Na⁺, and K⁺. "
            "NORMAL FUNCTION: PC2 localises to primary cilia and forms a complex with PC1 "
            "to function as a mechanosensory Ca²⁺-signalling channel in renal tubular "
            "epithelium — sensing tubular fluid flow and regulating intracellular Ca²⁺, "
            "mTOR, and Wnt pathways. PC2 also resides in ER membranes regulating "
            "Ca²⁺ store release. "
            "PATHOMECHANISM: LOF variants in PKD2 → loss of TRPP2 channel function → "
            "two-hit somatic second mutation in individual tubular cells → cyst initiation → "
            "cAMP-driven expansion (same mechanism as PKD1 but proceeds more slowly). "
            "PKD2 variants → generally truncating (frameshift/nonsense) — unlike PKD1 "
            "where missense vs truncating strongly predict severity. "
            "CLINICAL FEATURES: Indistinguishable from PKD1 on imaging but milder overall; "
            "ESRD develops ~10 years later (PKD2 median ~68 yrs vs PKD1 ~55 yrs); "
            "life expectancy closer to general population; intracranial aneurysm risk "
            "lower (~2–5% vs 5–10% in PKD1); liver cysts less numerous. "
            "DIAGNOSIS: Ultrasound criteria (Pei unified criteria): "
            "15–39 yrs: ≥3 unilateral or bilateral cysts; "
            "40–59 yrs: ≥2 cysts each kidney; "
            "≥60 yrs: ≥4 cysts each kidney. Genetic testing distinguishes PKD1 from PKD2. "
            "TREATMENT: Same as PKD1 — tolvaptan (V2 antagonist; REPRISE trial expanded "
            "to rapid progressors by eGFR decline ≥2.5 mL/min/1.73m²/yr); "
            "BP control ACEi/ARBs; hydration; dietary modifications. "
            "PROGNOSIS: Better than PKD1 — Mayo 1A-B phenotype most common in PKD2; "
            "GFR decline slower; transplant outcomes identical to PKD1."
        ),
        "inheritance": (
            "AD; 50% offspring; de novo ~5%; "
            "all truncating — genotype-phenotype less strong than PKD1; "
            "intra-familial variability present (modifier genes/environment)"
        ),
        "hallmark": (
            "ADPKD phenotype with LATER ESRD vs PKD1 (~10 yrs); "
            "molecular diagnosis distinguishes PKD1 from PKD2; "
            "milder liver cyst burden; lower intracranial aneurysm risk; "
            "Pei unified ultrasound criteria apply equally to PKD1 and PKD2"
        ),
        "key_ddx": (
            "PKD1-ADPKD (clinically identical — genetics distinguishes; earlier ESRD PKD1); "
            "ARPKD (PKHD1 — severe in children, congenital hepatic fibrosis; "
            "not bilateral adult simple cysts); "
            "Simple renal cysts (unilateral/few; no family history; no progression)"
        ),
        "treatment_alert": (
            "TOLVAPTAN: Same hepatotoxicity risk as PKD1 — LFTs mandatory monitoring; "
            "eligibility: Mayo class 1C-1E or eGFR decline ≥2.5/yr (REPRISE criteria); "
            "HYDRATION: >3L/day to suppress AVP — reduces cAMP-driven cyst expansion; "
            "avoid caffeine (adenosine receptor antagonist — raises cAMP); "
            "HIGH-SALT DIET worsens BP and cyst growth — restrict Na <2g/day"
        ),
        "seed": 1159,
        "cohort_n": 40,
        "esrd_rate": 0.35,
        "hypertension_rate": 0.72,
        "liver_cyst_rate": 0.60,
        "haematuria_rate": 0.38,
        "aneurysm_rate": 0.04,
        "drug_error_rate": 0.08,
        "diagnosis_delay_rate": 0.40,
        "transplant_rate": 0.20,
        "surveillance_adherent_rate": 0.70,
        "severity_weights": {"Mild": 0.35, "Moderate": 0.45, "Severe": 0.20},
        "primary_complication": "Progressive CKD/ESRD (milder, later than PKD1)",
        "gfr_pattern": "Slower decline than PKD1; ESRD median ~68 yrs",
        "proteinuria_pattern": "Mild-moderate; less than PKD1 at same TKV",
    },

    # ── COL4A5 — Alport Syndrome X-linked ───────────────────────────────────
    {
        "gene": "COL4A5",
        "protein": "Collagen type IV alpha-5 chain (COL4A5)",
        "alias": (
            "COL4A5; OMIM gene 303630; Xq22.3; ~1454 aa; Alport syndrome X-linked (OMIM #301050); "
            "XLR (males severely affected; females variable carriers); "
            "haematuria + SNHL + anterior lenticonus PATHOGNOMONIC; "
            "ESRD males 30–40 yrs; ACEi/ARBs MANDATORY early — delay ESRD by years; "
            "most common Alport form (~80%)"
        ),
        "aa": "~1454 aa",
        "kDa": "~161 kDa",
        "gene_class": (
            "Collagen type IV alpha-5 chain; triple-helix forming collagenous protein; "
            "Gly-X-Y repeat domain; 7S domain (N-terminal); NC1 domain (C-terminal — "
            "critical for alpha-345 trimer assembly); "
            "forms alpha-3/alpha-4/alpha-5 collagen IV heterotrimer network in "
            "glomerular basement membrane (GBM), cochlear basement membrane, "
            "and anterior lens capsule"
        ),
        "locus": "Xq22.3",
        "omim_gene": 303630,
        "omim_disease": 301050,
        "phenotype": (
            "Alport syndrome X-linked (males): persistent microhaematuria → nephrotic syndrome → "
            "ESRD 30–40 yrs; sensorineural hearing loss (SNHL) bilateral high-frequency; "
            "anterior lenticonus PATHOGNOMONIC (bulging of anterior lens — slit lamp); "
            "macular flecks; diffuse leiomyomatosis (rare, large COL4A5 deletions); "
            "Females: variable (carrier → microhaematuria only → frank Alport)"
        ),
        "disease": (
            "COL4A5 encodes the alpha-5 chain of type IV collagen, which forms an "
            "obligatory alpha-3/alpha-4/alpha-5 heterotrimer network in the GBM, "
            "cochlear (stria vascularis and spiral ligament), and anterior lens capsule basement membranes. "
            "NORMAL FUNCTION: The alpha345 collagen IV network provides the structural "
            "scaffold and charge-selective filtration barrier of the GBM. In the cochlea, "
            "it maintains the endocochlear potential. In the lens capsule, it supports "
            "lens architecture. "
            "PATHOMECHANISM: LOF variants in COL4A5 → defective alpha-5 chain → "
            "alpha345 heterotrimer cannot assemble → no alpha345 network in GBM → "
            "compensatory persistence of fetal alpha112 collagen IV → GBM thinning → "
            "then irregular thickening with multilamellation (electron microscopy: "
            "'basket-weave' or 'moth-eaten' appearance PATHOGNOMONIC on EM) → "
            "haematuria → progressive proteinuria → glomerulosclerosis → ESRD. "
            "RENAL PATHOLOGY: EM shows GBM thinning (early) → multilamellation "
            "(classic Alport basket-weave pattern); IHC shows absent COL4A5 staining "
            "in males (diagnostic). "
            "CLINICAL: Persistent microhaematuria UNIVERSAL from birth; macrohaematuria "
            "(with intercurrent URTIs — distinguish from IgA nephropathy); "
            "bilateral high-frequency SNHL (typically before ESRD — audiogram mandatory); "
            "anterior lenticonus (slit-lamp — bulging, oily-droplet appearance PATHOGNOMONIC); "
            "macular flecks/dot-and-fleck retinopathy (50% but rarely symptomatic). "
            "FEMALE CARRIERS: 25–30% develop significant proteinuria and CKD; "
            "10–15% reach ESRD by age 65 — not merely 'carriers'; "
            "treat and monitor like male Alport. "
            "DIAGNOSIS: COL4A5 sequencing (SNV + large deletion/duplication); "
            "kidney biopsy (EM basket-weave + IHC COL4A5 absent); "
            "skin biopsy (COL4A5 IHC on epidermal BM — simpler than kidney biopsy). "
            "TREATMENT: ACEi (ramipril/enalapril) FIRST-LINE — reduce proteinuria and "
            "delay ESRD; start early even in pre-proteinuria haematuria stage "
            "(Gross 2012 cohort: ACEi started before proteinuria → ESRD at 49 yrs vs "
            "33 yrs without treatment); ARBs second-line if ACEi not tolerated; "
            "sparsentan (dual endothelin/AT1-receptor antagonist) FDA 2023 for IgAN "
            "also used in Alport trials. "
            "ESRD: dialysis + transplant curative; anti-GBM disease post-transplant "
            "rare (~3–5% — anti-COL4A3 antibodies formed against 'foreign' GBM)."
        ),
        "inheritance": (
            "X-linked; XLR in males (severely affected — hemizygous); "
            "females (heterozygous) variably affected — 25–30% significant CKD; "
            "de novo COL4A5 mutations ~15%; "
            "deletions extending into COL4A6 → diffuse oesophageal/genital leiomyomatosis"
        ),
        "hallmark": (
            "Persistent microhaematuria from birth + SNHL + anterior lenticonus PATHOGNOMONIC; "
            "EM basket-weave GBM multilamellation; absent COL4A5 IHC on skin/kidney; "
            "ACEi started BEFORE proteinuria delays ESRD by ~16 years; "
            "female carriers: 25–30% develop significant CKD — not benign"
        ),
        "key_ddx": (
            "Thin basement membrane nephropathy / TBMN (COL4A3/4 heterozygous — only thin GBM, "
            "no multilamellation, good prognosis; overlaps COL4A3/4 heterozygous carriers); "
            "IgA nephropathy (macrohaematuria post-URTI; IgA deposits on IF — absent Alport; "
            "but Alport can co-exist with IgA mesangial deposits → confusing); "
            "Anti-GBM disease (linear IgG on IF; acute onset; COL4A3 NC1 antibodies; "
            "distinct from Alport but can complicate post-transplant)"
        ),
        "treatment_alert": (
            "ACEi/ARBs: START EARLY — even in pre-proteinuria haematuria stage; "
            "delay ESRD by up to 16 years in prospective cohort data; "
            "monitor K⁺ and GFR; "
            "ANTI-GBM POST-TRANSPLANT: rare (3–5%) but severe — "
            "plasma exchange + high-dose steroids + cyclophosphamide; "
            "FEMALE CARRIERS: Do NOT dismiss as 'just carriers' — 25–30% develop "
            "significant CKD and ESRD; monitor annually with urine protein and GFR"
        ),
        "seed": 1160,
        "cohort_n": 40,
        "esrd_rate": 0.65,
        "hypertension_rate": 0.75,
        "snhl_rate": 0.80,
        "anterior_lenticonus_rate": 0.25,
        "drug_error_rate": 0.15,
        "diagnosis_delay_rate": 0.50,
        "transplant_rate": 0.45,
        "surveillance_adherent_rate": 0.60,
        "severity_weights": {"Mild": 0.10, "Moderate": 0.35, "Severe": 0.55},
        "primary_complication": "ESRD 30–40 yrs (males) / SNHL / anterior lenticonus",
        "gfr_pattern": "Normal until teens/20s; rapid decline to ESRD 30–40 yrs (males)",
        "proteinuria_pattern": "Microhaematuria → progressive proteinuria → nephrotic range",
    },

    # ── COL4A3 — Alport Syndrome AR ─────────────────────────────────────────
    {
        "gene": "COL4A3",
        "protein": "Collagen type IV alpha-3 chain (COL4A3)",
        "alias": (
            "COL4A3; OMIM gene 120070; 2q36.3; ~1670 aa; Alport syndrome AR (OMIM #203780); "
            "AR; most severe Alport — ESRD before age 30; consanguinity; "
            "same alpha-345 collagen IV network as COL4A5; "
            "heterozygous carriers: thin GBM / TBMN — mostly benign but 15–30% progress to CKD; "
            "COL4A3 + COL4A4 adjacent on 2q36"
        ),
        "aa": "~1670 aa",
        "kDa": "~185 kDa",
        "gene_class": (
            "Collagen type IV alpha-3 chain; triple-helix Gly-X-Y repeat; "
            "NC1 domain dimerises (NC1 hexamer in assembled collagen IV network); "
            "forms obligatory alpha-3/alpha-4/alpha-5 heterotrimer in GBM; "
            "NC1 domain of COL4A3 = Goodpasture antigen "
            "(target of anti-GBM antibodies in Goodpasture's disease / RPGN)"
        ),
        "locus": "2q36.3",
        "omim_gene": 120070,
        "omim_disease": 203780,
        "phenotype": (
            "Alport syndrome AR — haematuria + SNHL + ESRD by 20–30 yrs; "
            "most severe Alport form; consanguinity common; "
            "anterior lenticonus (less frequent than XL form); "
            "COL4A3 NC1 = Goodpasture antigen — anti-GBM disease if transplanted "
            "with normal COL4A3-containing kidney (same as COL4A5 XL)"
        ),
        "disease": (
            "COL4A3 encodes the alpha-3 chain of type IV collagen, which is essential "
            "for forming the alpha-3/alpha-4/alpha-5 (alpha345) heterotrimer network "
            "in the GBM. COL4A3 and COL4A4 are adjacent on chromosome 2q36 and "
            "frequently deleted together in AR Alport. "
            "PATHOMECHANISM: Biallelic LOF in COL4A3 → failure of alpha345 network → "
            "identical GBM pathology to COL4A5 XL Alport (fetal alpha112 persistence → "
            "GBM thinning → basket-weave multilamellation on EM) → haematuria → "
            "progressive proteinuria → ESRD. "
            "AR Alport IS MORE SEVERE than XL in males: ESRD typically before age 25–30 "
            "(vs 30–40 for XL males). "
            "NC1 DOMAIN SIGNIFICANCE: The NC1 domain of the alpha-3 chain contains "
            "the Goodpasture antigen — epitopes recognised by anti-GBM autoantibodies "
            "in Goodpasture's disease. After renal transplantation, patients with "
            "biallelic COL4A3 mutations lack this antigen; the 'foreign' COL4A3 NC1 "
            "in the donated kidney can trigger de novo anti-GBM disease "
            "(3–5% of AR Alport transplants — monitor with anti-GBM antibody titres). "
            "HETEROZYGOUS CARRIERS (one mutant COL4A3 allele): "
            "Show thin GBM (TBMN) and microhaematuria; historically considered benign; "
            "BUT 15–30% develop significant proteinuria and CKD, especially males; "
            "important to counsel carriers and follow annually — not dismiss. "
            "CONSANGUINITY: AR Alport most common in consanguineous populations "
            "(Middle East, South Asia, North Africa). "
            "TREATMENT: ACEi/ARBs same as XL form — start at first proteinuria "
            "(or even at haematuria stage in families with known severe mutations); "
            "renal replacement therapy essential; transplant outcomes good; "
            "anti-GBM monitoring post-transplant mandatory. "
            "GENOTYPE-PHENOTYPE: Biallelic truncating > biallelic missense > compound heterozygous "
            "(truncating + missense) in terms of severity."
        ),
        "inheritance": (
            "AR; heterozygous COL4A3 = thin BM disease (TBMN) — haematuria carrier state; "
            "biallelic (homozygous or compound heterozygous) = AR Alport; "
            "consanguinity major risk factor; both COL4A3 and COL4A4 are on 2q36 "
            "and often deleted together"
        ),
        "hallmark": (
            "Most severe Alport form — ESRD by 20–25 yrs; consanguinity; "
            "basket-weave GBM EM + absent COL4A3 IHC; "
            "NC1 domain = Goodpasture antigen — anti-GBM post-transplant risk; "
            "heterozygous carriers: thin GBM (TBMN) — 15–30% develop CKD — monitor"
        ),
        "key_ddx": (
            "COL4A5 XL Alport (same phenotype — later ESRD; XL vs AR by genetics/family history); "
            "Goodpasture's disease (anti-COL4A3 NC1 antibodies; RPGN; linear IgG on IF; "
            "plasma exchange + cyclophosphamide; different from Alport but shares NC1 antigen); "
            "TBMN (COL4A3/4 heterozygous — thin GBM only; mostly benign; "
            "25–30% overlap with progressive CKD → hard to distinguish from AR Alport carrier)"
        ),
        "treatment_alert": (
            "ACEi/ARBs: start at first haematuria in severe genotypes — delay ESRD; "
            "ANTI-GBM POST-TRANSPLANT: monitor anti-COL4A3 NC1 antibodies after transplant; "
            "HETEROZYGOUS CARRIERS (TBMN): DO NOT dismiss — follow annually "
            "with urine ACR and eGFR; 15–30% develop CKD requiring treatment; "
            "AVOID NEPHROTOXINS (NSAIDs, aminoglycosides, IV contrast without hydration) "
            "in all Alport spectrum — accelerate GBM damage"
        ),
        "seed": 1161,
        "cohort_n": 40,
        "esrd_rate": 0.75,
        "hypertension_rate": 0.80,
        "snhl_rate": 0.72,
        "anterior_lenticonus_rate": 0.18,
        "drug_error_rate": 0.12,
        "diagnosis_delay_rate": 0.45,
        "transplant_rate": 0.55,
        "surveillance_adherent_rate": 0.58,
        "severity_weights": {"Mild": 0.05, "Moderate": 0.25, "Severe": 0.70},
        "primary_complication": "ESRD by age 20–30 / anti-GBM post-transplant",
        "gfr_pattern": "Rapid decline from teens; ESRD typically age 20–30",
        "proteinuria_pattern": "Progressive from microhaematuria → nephrotic-range proteinuria",
    },

    # ── NPHS1 — Congenital Nephrotic Syndrome Finnish type ──────────────────
    {
        "gene": "NPHS1",
        "protein": "Nephrin",
        "alias": (
            "NPHS1; OMIM gene 602716; 19q13.12; ~1241 aa; Congenital nephrotic syndrome Finnish type / CNF (OMIM #256300); "
            "AR; nephrin is the structural backbone of the glomerular slit diaphragm; "
            "massive proteinuria at birth (edematous foetus — elevated amniotic AFP); "
            "Fin-major (c.121_122insC, p.Leu41Phefs) + Fin-minor (c.3325C>T, p.Arg1109Ter) — Finnish founder mutations; "
            "transplant curative BUT 25% recurrence risk from anti-nephrin antibodies"
        ),
        "aa": "~1241 aa",
        "kDa": "~135 kDa",
        "gene_class": (
            "Immunoglobulin superfamily member (8 Ig-like domains + fibronectin type III + "
            "single transmembrane domain + short cytoplasmic tail); "
            "slit diaphragm structural backbone protein; "
            "connects adjacent podocyte foot processes across the filtration slit; "
            "interacts with podocin (NPHS2), CD2AP, NEPH1, ZO-1; "
            "activates PI3K/Akt and RAS/MAPK signalling; "
            "mechano-sensing and cytoskeletal regulation in podocytes"
        ),
        "locus": "19q13.12",
        "omim_gene": 602716,
        "omim_disease": 256300,
        "phenotype": (
            "Congenital nephrotic syndrome — massive proteinuria at birth (nephrotic at birth or within first 3 months); "
            "foetal oedema (ascites, pleural effusions — elevated amniotic AFP PATHOGNOMONIC); "
            "placenta large (>25% birth weight); hyperlipidaemia; "
            "thrombosis risk (low antithrombin III); infection risk (low IgG); "
            "untreated → ESRD and death by age 5 without dialysis/transplant"
        ),
        "disease": (
            "NPHS1 encodes nephrin, a ~135 kDa type I transmembrane protein of the "
            "immunoglobulin superfamily that forms the backbone of the glomerular "
            "filtration slit diaphragm — the critical size-selective barrier "
            "at the junction between adjacent podocyte foot processes. "
            "NORMAL FUNCTION: Nephrin molecules from adjacent foot processes interact "
            "homophilically (Ig-domain 'zipper') to form the filtration slit diaphragm; "
            "extracellular domains provide the physical barrier; "
            "cytoplasmic tail anchors to actin cytoskeleton via CD2AP, podocin (NPHS2), "
            "and activates survival signalling (PI3K/Akt) in podocytes. "
            "PATHOMECHANISM: LOF variants in NPHS1 → absent or dysfunctional nephrin → "
            "loss of slit diaphragm → massive unrestricted protein leakage through GBM → "
            "proteinuria at birth (all plasma proteins — albumin, immunoglobulins, "
            "clotting factors lost in urine). "
            "PERINATAL FEATURES: Elevated amniotic AFP (nephrin-deficient foetus leaks AFP "
            "into amniotic fluid before birth — prenatal diagnosis possible); "
            "placentomegaly (>25% birth weight — nephrotic oedema + fluid); "
            "foetal ascites + hydrops fetalis in severe cases. "
            "COMPLICATIONS: Thrombosis (antithrombin III loss → hypercoagulable); "
            "infection (IgG loss → hypogammaglobulinaemia → Gram-positive sepsis); "
            "malnutrition (albumin loss); hypothyroidism (TBG loss); "
            "delayed motor development. "
            "FINNISH FOUNDER MUTATIONS: Fin-major (c.121_122insC, p.Leu41Phefs*4 — "
            "78% of Finnish CNF alleles) and Fin-minor (c.3325C>T, p.Arg1109Ter — "
            "16% of Finnish alleles); gene frequency 1:200 in Finland. "
            "Outside Finland: hundreds of non-Finnish variants (missense, frameshift, nonsense). "
            "TREATMENT: Symptomatic management (albumin infusions, daily — 'albumin drip'; "
            "anticoagulation — heparin or warfarin; IgG replacement; diet high protein/calories); "
            "unilateral or bilateral nephrectomy (to stop protein loss before transplant) + "
            "peritoneal dialysis until transplant weight adequate (>8–10 kg); "
            "RENAL TRANSPLANT: curative — transplanted kidney has functional nephrin; "
            "RECURRENCE (25%): post-transplant de novo anti-nephrin antibodies in some patients "
            "→ focal segmental glomerulosclerosis → nephrotic syndrome recurrence "
            "→ treat with plasmapheresis + rituximab."
        ),
        "inheritance": (
            "AR; both parents heterozygous carriers (generally asymptomatic); "
            "Finnish prevalence 1:8,000 live births; "
            "carrier frequency 1:200 in Finland; "
            "worldwide prevalence 1:50,000–1:100,000; "
            "prenatal diagnosis via amniotic AFP + NPHS1 genotyping"
        ),
        "hallmark": (
            "Massive proteinuria at BIRTH (or within 3 months); "
            "elevated amniotic AFP PATHOGNOMONIC; large placenta (>25% birth weight); "
            "Fin-major + Fin-minor Finnish founder mutations; "
            "25% post-transplant recurrence (anti-nephrin antibodies) — counsel families pre-transplant"
        ),
        "key_ddx": (
            "NPHS2-FSGS (podocin — childhood onset 1–4 yrs, NOT congenital; steroid-resistant; "
            "R138Q European founder; low recurrence post-transplant); "
            "Diffuse mesangial sclerosis (WT1 mutation — DMS on biopsy; Wilms tumour risk); "
            "Congenital infection (TORCH — CMV/toxoplasma; serologies; no NPHS1 mutation); "
            "Neonatal alloimmune thrombocytopenia with renal involvement (platelet antibodies; "
            "different presentation)"
        ),
        "treatment_alert": (
            "ANTI-NEPHRIN ANTIBODY RECURRENCE: 25% post-transplant — plasmapheresis + rituximab; "
            "counsel families PRE-TRANSPLANT about recurrence risk; "
            "THROMBOSIS: anticoagulate (antithrombin III loss) — heparin/warfarin mandatory; "
            "INFECTION: IgG replacement (monthly IVIG) — low IgG from proteinuria loss; "
            "HYPOTHYROIDISM: TBG lost in urine → secondary hypothyroidism — screen and treat; "
            "DIALYSIS TIMING: bilateral nephrectomy + PD until transplant weight >8–10 kg"
        ),
        "seed": 1162,
        "cohort_n": 40,
        "esrd_rate": 0.95,
        "hypertension_rate": 0.70,
        "thrombosis_rate": 0.25,
        "infection_rate": 0.55,
        "drug_error_rate": 0.10,
        "diagnosis_delay_rate": 0.10,   # usually diagnosed at birth
        "transplant_rate": 0.80,
        "surveillance_adherent_rate": 0.78,
        "severity_weights": {"Mild": 0.02, "Moderate": 0.08, "Severe": 0.90},
        "primary_complication": "ESRD + thrombosis + infection (congenital nephrotic at birth)",
        "gfr_pattern": "Nephrotic at birth; ESRD within 5 yrs without transplant",
        "proteinuria_pattern": "Massive (>10–100 g/day) — all plasma proteins lost",
    },

    # ── NPHS2 — FSGS type 1 (Podocin) ────────────────────────────────────────
    {
        "gene": "NPHS2",
        "protein": "Podocin",
        "alias": (
            "NPHS2; OMIM gene 604766; 1q25.2; ~383 aa; Focal segmental glomerulosclerosis type 1 / FSGS1 (OMIM #600995); "
            "AR; childhood steroid-resistant nephrotic syndrome (SRNS); R138Q European founder; "
            "ACEi + cyclosporin/rituximab; low transplant recurrence risk (unlike NPHS1 25%); "
            "R229Q hypomorphic allele — compound heterozygous with truncating → milder adult FSGS"
        ),
        "aa": "~383 aa",
        "kDa": "~42 kDa",
        "gene_class": (
            "Stomatin/prohibitin/flotillin/HflK (SPFH) domain protein; "
            "cholesterol-interacting membrane protein; "
            "lipid raft-associated at the slit diaphragm; "
            "directly interacts with nephrin (NPHS1) cytoplasmic tail; "
            "anchors nephrin to lipid rafts; "
            "organises signalling complexes at the slit diaphragm; "
            "cytoplasmic C-terminus interacts with TRPC6 (mechanosensitive Ca²⁺ channel)"
        ),
        "locus": "1q25.2",
        "omim_gene": 604766,
        "omim_disease": 600995,
        "phenotype": (
            "FSGS type 1 — childhood onset (1–4 yrs) nephrotic syndrome (steroid-resistant); "
            "proteinuria + oedema + hypoalbuminaemia + hyperlipidaemia; "
            "FSGS on renal biopsy; progressive CKD → ESRD (age 10–30 yrs); "
            "no extrarenal manifestations; low transplant recurrence (unlike NPHS1)"
        ),
        "disease": (
            "NPHS2 encodes podocin, a ~42 kDa integral membrane protein that localises "
            "to podocyte foot process plasma membrane lipid rafts at the slit diaphragm. "
            "NORMAL FUNCTION: Podocin directly binds the cytoplasmic tail of nephrin (NPHS1) "
            "and anchors it to cholesterol-rich lipid rafts, stabilising nephrin signalling "
            "complexes. Podocin also interacts with TRPC6 (mechanosensitive Ca²⁺ channel), "
            "MEC-2 (C. elegans homologue), and CD2AP at the slit diaphragm. "
            "PATHOMECHANISM: LOF variants in NPHS2 → podocin absent/dysfunctional → "
            "nephrin cannot be properly anchored to lipid rafts → slit diaphragm "
            "destabilisation → podocyte foot process effacement → protein leak → "
            "FSGS pattern on biopsy. "
            "CLINICAL FEATURES: Typically presents at age 1–4 yrs with full nephrotic "
            "syndrome (proteinuria >3.5 g/1.73m²/day, oedema, hypoalbuminaemia, "
            "hyperlipidaemia); STEROID-RESISTANT (does NOT respond to prednisolone — "
            "this is the key diagnostic clue that triggers genetic testing); "
            "progressive CKD → ESRD age 10–30 yrs. "
            "FOUNDER MUTATION: R138Q (c.413G>A, p.Arg138Gln) — most common NPHS2 "
            "variant in European SRNS patients (~30% of alleles); "
            "R138Q/R138Q homozygous or R138Q/truncating compound heterozygous → classic childhood FSGS; "
            "R229Q hypomorphic allele (c.686G>A, p.Arg229Gln) — reduced function; "
            "compound heterozygous R229Q + truncating → adult-onset mild FSGS. "
            "BIOPSY: FSGS (tip or perihilar lesion most common); "
            "EM: foot process effacement; "
            "no immune deposits (unlike membranous nephropathy or IgAN). "
            "TREATMENT: ACEi (enalapril/ramipril) — reduce proteinuria and slow CKD; "
            "cyclosporin/tacrolimus (calcineurin inhibitors — reduce proteinuria in some; "
            "not curative — relapse on withdrawal); "
            "rituximab (anti-CD20 — emerging evidence in SRNS with T-cell pathology, "
            "but genetic NPHS2 FSGS may respond less); "
            "ESRD → dialysis + transplant; "
            "LOW TRANSPLANT RECURRENCE: unlike NPHS1 (25%), NPHS2 FSGS very low "
            "recurrence post-transplant (<3–5%) because the defect is intrinsic to podocin "
            "in the native kidney, not circulating factor-mediated. "
            "DISTINGUISH FROM PRIMARY FSGS: Primary FSGS (idiopathic) has 30–50% "
            "transplant recurrence due to circulating permeability factor; "
            "genetic NPHS2/NPHS1 FSGS has low recurrence — genetic testing crucial "
            "before transplant to counsel on recurrence risk."
        ),
        "inheritance": (
            "AR; compound heterozygous most common in European populations (R138Q + second allele); "
            "R229Q hypomorphic allele — compound heterozygous with truncating → adult FSGS; "
            "heterozygous carriers (one NPHS2 allele) generally asymptomatic"
        ),
        "hallmark": (
            "Childhood steroid-resistant nephrotic syndrome (SRNS) — genetic testing mandatory; "
            "R138Q European founder mutation (~30% alleles); "
            "FSGS on biopsy; low transplant recurrence (<5%) — "
            "distinguish from primary FSGS (30–50% recurrence); "
            "R229Q hypomorphic → adult-onset mild FSGS when compound heterozygous"
        ),
        "key_ddx": (
            "Primary/idiopathic FSGS (circulating permeability factor; 30–50% transplant recurrence; "
            "no NPHS2 mutation; may respond to plasmapheresis post-transplant); "
            "NPHS1 CNF (congenital nephrotic at birth; 25% recurrence; Fin-major/Fin-minor; different gene); "
            "Steroid-sensitive nephrotic syndrome (MCD — minimal change disease; "
            "responds to steroids; no genetic mutation); "
            "Secondary FSGS (HIV/heroin/obesity/reflux nephropathy — secondary cause)"
        ),
        "treatment_alert": (
            "STEROID RESISTANCE: do NOT escalate steroids — genetic testing first; "
            "prolonged steroid side effects (obesity, growth retardation, osteoporosis) "
            "are avoidable once genetic SRNS confirmed; "
            "TRANSPLANT RECURRENCE: <5% for NPHS2 — very low; "
            "counsel families: genetic FSGS ≠ primary FSGS in terms of recurrence; "
            "CYCLOSPORIN: partial responders — use lowest effective dose (nephrotoxic); "
            "RITUXIMAB: evidence in steroid-dependent MCD/primary FSGS is strong; "
            "evidence weaker in genetic NPHS2 FSGS — manage expectations"
        ),
        "seed": 1163,
        "cohort_n": 40,
        "esrd_rate": 0.70,
        "hypertension_rate": 0.65,
        "steroid_resistance_rate": 1.00,
        "drug_error_rate": 0.20,
        "diagnosis_delay_rate": 0.55,
        "transplant_rate": 0.50,
        "surveillance_adherent_rate": 0.65,
        "severity_weights": {"Mild": 0.10, "Moderate": 0.40, "Severe": 0.50},
        "primary_complication": "ESRD (childhood–young adult) / steroid toxicity from misdiagnosis",
        "gfr_pattern": "Progressive CKD from childhood; ESRD age 10–30 yrs",
        "proteinuria_pattern": "Nephrotic-range (>3.5 g/day) steroid-resistant from onset",
    },

    # ── UMOD — Uromodulin-associated kidney disease ──────────────────────────
    {
        "gene": "UMOD",
        "protein": "Uromodulin (Tamm-Horsfall protein / THP)",
        "alias": (
            "UMOD; OMIM gene 191845; 16p12.3; ~640 aa; Uromodulin-associated kidney disease / UAKD (OMIM #162000); "
            "AD (also called FJHN — familial juvenile hyperuricaemic nephropathy, or MCKD2 — medullary cystic kidney disease type 2); "
            "GOUT in teens PATHOGNOMONIC; slow tubulointerstitial CKD; "
            "NSAIDs ABSOLUTELY CI (worsen CKD); allopurinol for hyperuricaemia; no specific curative therapy"
        ),
        "aa": "~640 aa",
        "kDa": "~85 kDa (core); heavily glycosylated ~100–150 kDa native",
        "gene_class": (
            "GPI-anchored glycoprotein; exclusively produced by thick ascending limb (TAL) "
            "of loop of Henle; most abundant urinary protein in humans; "
            "ZP domain + EGF-like domains + 8 cysteines enabling disulphide bonds; "
            "polymerises into large supramolecular aggregates in urine; "
            "roles: antibacterial (traps uropathogenic E. coli), "
            "renal crystal inhibitor (prevents calcium oxalate stone formation), "
            "immune modulation (TLR4 signalling); "
            "mutant UMOD misfolds → ER retention → tubular epithelial cell stress"
        ),
        "locus": "16p12.3",
        "omim_gene": 191845,
        "omim_disease": 162000,
        "phenotype": (
            "Uromodulin-associated kidney disease (UAKD): "
            "hyperuricaemia → gout (often teens/20s — PATHOGNOMONIC for age); "
            "slowly progressive tubulointerstitial CKD; "
            "medullary cysts (small — 1–5 mm — not always detected on ultrasound); "
            "ESRD typically age 40–70 yrs; no extrarenal features; "
            "renal biopsy: tubular atrophy + interstitial fibrosis + medullary cysts"
        ),
        "disease": (
            "UMOD encodes uromodulin, also known as Tamm-Horsfall protein (THP), "
            "the most abundantly produced urinary protein in healthy adults (~50 mg/day). "
            "Uromodulin is exclusively synthesised by thick ascending limb (TAL) "
            "epithelial cells where it is GPI-anchored to the apical membrane, "
            "then cleaved by TMPRSS11A and released into urine. "
            "NORMAL FUNCTION: Uromodulin polymerises into large gel-like filaments "
            "in urine — antibacterial (traps type 1 fimbriated E. coli), crystal inhibitor "
            "(prevents calcium oxalate stone aggregation), and immune modulator. "
            "PATHOMECHANISM: Missense variants in UMOD (predominantly in EGF-like domain "
            "cysteine residues disrupting disulphide bonds → protein misfolding → "
            "ER retention → abnormal intracellular accumulation in TAL cells → "
            "ER stress → UPR (unfolded protein response) → tubular cell apoptosis → "
            "tubulointerstitial nephritis → progressive fibrosis → CKD. "
            "HYPERURICAEMIA: Loss of TAL function → impaired uric acid secretion → "
            "systemic hyperuricaemia → gout. "
            "GOUT IN TEENS: Uric acid gout presenting in the second or third decade "
            "of life is HIGHLY PATHOGNOMONIC for UAKD; primary gout in young adults "
            "should always prompt urine uromodulin measurement and UMOD sequencing. "
            "MEDULLARY CYSTS: Small (1–5 mm) cysts at corticomedullary junction; "
            "often missed on routine ultrasound; MRI more sensitive. "
            "URINE UROMODULIN: Low urinary uromodulin (from ER-retained mutant) "
            "is a functional biomarker — plasma uromodulin also low in UAKD. "
            "TREATMENT: Allopurinol (xanthine oxidase inhibitor) — reduces uric acid, "
            "prevents gout attacks, may slow CKD progression; "
            "NSAIDs ABSOLUTELY CONTRAINDICATED — accelerate tubulointerstitial injury "
            "and hasten ESRD (NSAIDs block prostaglandin-mediated afferent arteriolar "
            "dilatation → reduce GFR → tubular ischaemia — catastrophic in already "
            "compromised TAL); "
            "colchicine for acute gout flares (avoid NSAIDs); "
            "ACEi/ARBs for BP control and proteinuria reduction; "
            "ESRD: dialysis/transplant (transplant curative — donor kidney has normal UMOD). "
            "NO SPECIFIC THERAPY targeting UMOD misfolding currently approved; "
            "chemical chaperones (4-PBA) investigational."
        ),
        "inheritance": (
            "AD; ~100% penetrance for CKD and hyperuricaemia by adulthood; "
            "de novo mutations rare; "
            "interfamilial variability in age of ESRD; "
            "all pathogenic UMOD variants so far are missense (typically EGF-like cysteine residues) "
            "— truncating variants may not cause disease (different mechanism)"
        ),
        "hallmark": (
            "Gout in teens/young adults (before age 30) PATHOGNOMONIC for UAKD; "
            "low urinary + plasma uromodulin (ER retention biomarker); "
            "slowly progressive tubulointerstitial CKD with small medullary cysts; "
            "NSAIDs ABSOLUTELY CI — accelerate ESRD dramatically; "
            "allopurinol mandatory for hyperuricaemia/gout prevention"
        ),
        "key_ddx": (
            "Primary gout (age >30; no CKD at presentation; no UMOD mutation; "
            "metabolic syndrome/alcohol — risk factors absent in UAKD teens); "
            "Lesch-Nyhan syndrome (HPRT1 — X-linked; severe neurological + gout + self-injurious behaviour; "
            "distinguished by HPRT assay and HPRT1 sequencing); "
            "ADPKD/PKD2 (larger cysts; not medullary predominant; polycystin mutation; "
            "no hyperuricaemia as presenting feature)"
        ),
        "treatment_alert": (
            "NSAIDs ABSOLUTELY CONTRAINDICATED — can precipitate acute-on-chronic kidney injury; "
            "use colchicine or corticosteroids for gout flares instead; "
            "ALLOPURINOL: start early — reduces gout and may slow CKD progression; "
            "FEBUXOSTAT: alternative to allopurinol (XO inhibitor; adjust dose for eGFR); "
            "AVOID: contrast agents without pre-hydration, aminoglycosides, "
            "high-dose calcineurin inhibitors — all accelerate tubulointerstitial damage"
        ),
        "seed": 1164,
        "cohort_n": 40,
        "esrd_rate": 0.45,
        "hypertension_rate": 0.70,
        "gout_rate": 0.88,
        "medullary_cyst_rate": 0.55,
        "drug_error_rate": 0.30,
        "diagnosis_delay_rate": 0.65,
        "transplant_rate": 0.25,
        "surveillance_adherent_rate": 0.60,
        "severity_weights": {"Mild": 0.25, "Moderate": 0.50, "Severe": 0.25},
        "primary_complication": "Gout (teens) + slow CKD → ESRD 40–70 yrs / NSAID-triggered AKI",
        "gfr_pattern": "Slow decline over decades; ESRD typically age 40–70 yrs",
        "proteinuria_pattern": "Mild tubular proteinuria (low molecular weight); rarely nephrotic",
    },

    # ── HNF1B — MODY5 / RCAD ──────────────────────────────────────────────
    {
        "gene": "HNF1B",
        "protein": "Hepatocyte nuclear factor 1-beta (HNF-1β / TCF2)",
        "alias": (
            "HNF1B; OMIM gene 189907; 17q12; ~557 aa; MODY5 / Renal Cysts And Diabetes / RCAD (OMIM #137920); "
            "AD; 50%+ de novo (most common de novo MODY type); "
            "renal cysts (prenatal/childhood) + diabetes (young adult) + hypomagnesaemia + "
            "pancreatic atrophy + genitourinary anomalies; "
            "sulfonylureas INEFFECTIVE — insulin required; whole-gene deletion most common"
        ),
        "aa": "~557 aa",
        "kDa": "~68 kDa",
        "gene_class": (
            "POU-homeodomain transcription factor; "
            "binds DNA as homodimer or HNF1A/HNF1B heterodimer; "
            "POU-S (POUs-specific domain) + POU-H (POU-homeodomain); "
            "transactivation domain in N-terminus; "
            "expressed in kidney (tubular epithelium), liver, pancreas (ductal cells), "
            "Müllerian-derived organs (uterus, fallopian tube), biliary epithelium; "
            "regulates PKHD1 (fibrocystin/AR-PKD gene), UMOD, FXYD2 (Na/K-ATPase gamma subunit)"
        ),
        "locus": "17q12",
        "omim_gene": 189907,
        "omim_disease": 137920,
        "phenotype": (
            "MODY5 / RCAD: "
            "Renal cysts (prenatal — detected on antenatal US); "
            "MODY type 5 diabetes (young adult onset; insulin-requiring; "
            "hyperglycaemia often disproportionate to apparent beta-cell mass); "
            "hypomagnesaemia (renal wasting); "
            "pancreatic atrophy (body + tail) + exocrine insufficiency; "
            "genitourinary anomalies (uterus didelphys / bicornuate uterus in females — infertility); "
            "biliary anomalies; hepatic steatosis; raised liver enzymes; "
            "CKD (variable — some reach ESRD)"
        ),
        "disease": (
            "HNF1B encodes hepatocyte nuclear factor 1-beta (HNF-1β), a "
            "POU-homeodomain transcription factor expressed during embryogenesis in "
            "the kidney, liver, pancreas, and Müllerian-derived reproductive organs. "
            "NORMAL FUNCTION: HNF1B regulates transcription of multiple genes critical "
            "for tubular differentiation and function, including PKHD1 (fibrocystin, "
            "the AR-PKD gene), UMOD (uromodulin/TAL function), "
            "FXYD2 (Na/K-ATPase gamma subunit — controls renal Mg²⁺ reabsorption), "
            "and SLC12A1 (NKCC2 in TAL). HNF1B controls ductal morphogenesis in "
            "the pancreas (ductal tree) and kidney (ureteric bud branching). "
            "PATHOMECHANISM: LOF variants in HNF1B → defective tubular differentiation → "
            "renal cysts (typically affecting collecting ducts and distal nephron — "
            "unlike ADPKD which affects any segment); "
            "LOF in pancreatic ductal cells → reduced beta-cell neogenesis + "
            "exocrine atrophy → MODY5 diabetes (insulin-requiring; sulfonylureas fail "
            "because the defect is in beta-cell development, not insulin secretion coupling "
            "to sulphonylurea receptor); "
            "LOF in TAL/DCT (distal convoluted tubule) → impaired FXYD2 expression → "
            "Na/K-ATPase dysfunction → renal Mg²⁺ wasting → hypomagnesaemia → "
            "tetany, muscle weakness, arrhythmia risk. "
            "MOLECULAR: Whole-gene deletion (17q12 microdeletion) accounts for ~50% "
            "of HNF1B disease — detected by MLPA or SNP array (NOT found by sequencing alone). "
            "Point mutations (missense/nonsense in POU domains) in other 50%. "
            "De novo variants: >50% — 'RCAD appears as a new case without family history "
            "because parents are unaffected'. "
            "RENAL PHENOTYPE: Multicystic dysplastic kidney (unilateral in some); "
            "bilateral renal cysts; renal hypoplasia/dysplasia; oligomeganephronia; "
            "hyperuricaemia (UMOD regulation); CKD variable (10–30% reach ESRD). "
            "DIABETES: MODY5 — young adult onset (teens to 40s); usually requires insulin; "
            "C-peptide detectable (not type 1 autoimmune); HbA1c may be modestly elevated; "
            "pancreatic atrophy on MRI (body/tail) PATHOGNOMONIC for HNF1B. "
            "GENITOURINARY: Uterus didelphys, bicornuate uterus, vaginal aplasia "
            "in females (~38%) — infertility; male: vas deferens agenesis. "
            "TREATMENT: Insulin for MODY5 diabetes (sulfonylureas ineffective — "
            "beta-cell KATP channel mechanism intact but beta-cell mass/differentiation "
            "inadequate; sulfonylurea trial acceptable if uncertain diagnosis but expect failure); "
            "Magnesium supplementation (oral or IV for symptomatic hypomagnesaemia); "
            "ACEi/ARBs for CKD management; "
            "GU surveillance for renal function and genitourinary anomalies; "
            "genetic counselling critical (de novo ~50% — parents may be unaffected)."
        ),
        "inheritance": (
            "AD; 50%+ de novo mutations (no family history in majority of cases); "
            "whole-gene 17q12 deletion ~50% (requires MLPA/SNP array — missed by sequencing); "
            "point mutations in POU domains ~50%; "
            "interfamilial and intrafamilial variability in phenotypic expression"
        ),
        "hallmark": (
            "Renal cysts detected prenatally/childhood + MODY5 diabetes + hypomagnesaemia + "
            "pancreatic atrophy on MRI PATHOGNOMONIC triad; "
            "50%+ de novo (no family history); "
            "17q12 whole-gene deletion — detect by MLPA NOT sequencing; "
            "sulfonylureas INEFFECTIVE — insulin required; "
            "uterus didelphys/bicornuate in females (infertility)"
        ),
        "key_ddx": (
            "ADPKD (PKD1/PKD2 — adult-onset bilateral cysts; no diabetes; no pancreatic atrophy; "
            "no hypomagnesaemia; polycystin mutations); "
            "Other MODY types (HNF1A-MODY3 — good sulfonylurea response; no renal cysts; "
            "GCK-MODY2 — mild non-progressive hyperglycaemia; no treatment needed; "
            "MODY10 INS — variable; no cysts); "
            "Type 1 DM (GAD/ZnT8/IA-2 antibodies positive; DKA common; no cysts; "
            "pancreas normal or small globally not atrophied body/tail only)"
        ),
        "treatment_alert": (
            "SULFONYLUREAS INEFFECTIVE in MODY5 — insulin required; "
            "misclassification as T1DM or T2DM common → correct diagnosis critical; "
            "WHOLE-GENE DELETION: 17q12 microdeletion NOT detected by sequencing alone — "
            "order MLPA or SNP array in all suspected HNF1B cases; "
            "HYPOMAGNESAEMIA: symptomatic — tetany, seizures, arrhythmia; "
            "supplement proactively; monitor Mg²⁺ level; "
            "FEMALE GENITOURINARY ANOMALIES: gynaecological assessment mandatory — "
            "infertility investigation if reproductive age; "
            "RENAL SURVEILLANCE: annual eGFR + urine ACR — variable CKD trajectory"
        ),
        "seed": 1165,
        "cohort_n": 40,
        "esrd_rate": 0.25,
        "hypertension_rate": 0.60,
        "diabetes_rate": 0.85,
        "hypomagnesaemia_rate": 0.70,
        "pancreatic_atrophy_rate": 0.75,
        "gu_anomaly_rate": 0.38,
        "drug_error_rate": 0.35,    # sulfonylurea error high
        "diagnosis_delay_rate": 0.60,
        "transplant_rate": 0.15,
        "surveillance_adherent_rate": 0.62,
        "severity_weights": {"Mild": 0.30, "Moderate": 0.45, "Severe": 0.25},
        "primary_complication": "MODY5 misdiagnosed as T1/T2DM + CKD + infertility (genitourinary)",
        "gfr_pattern": "Variable — 10–30% reach ESRD; renal cysts stable or slowly progressive",
        "proteinuria_pattern": "Variable — tubular pattern (hypomagnesaemia) + mild glomerular",
    },
]


# ─── Patient cohort generator ──────────────────────────────────────────────────

def _gen_cohort():
    patients = []
    for gd in NEPHROPATHY_GENES:
        rng = random.Random(gd["seed"])
        gene = gd["gene"]
        for i in range(gd["cohort_n"]):
            age = rng.randint(4, 75)
            sex = rng.choice(["M", "F"])
            severity = rng.choices(
                list(gd["severity_weights"].keys()),
                weights=list(gd["severity_weights"].values()),
            )[0]

            esrd       = rng.random() < gd.get("esrd_rate", 0.40)
            htn        = rng.random() < gd.get("hypertension_rate", 0.65)
            drug_err   = rng.random() < gd.get("drug_error_rate", 0.12)
            dx_delayed = rng.random() < gd.get("diagnosis_delay_rate", 0.45)
            transplant = rng.random() < gd.get("transplant_rate", 0.25)
            adherent   = rng.random() < gd.get("surveillance_adherent_rate", 0.65)

            p = {
                "gene": gene,
                "patient_id": f"{gene}-{i+1:03d}",
                "age": age,
                "sex": sex,
                "severity": severity,
                "esrd": esrd,
                "hypertension": htn,
                "drug_error": drug_err,
                "diagnosis_delayed": dx_delayed,
                "received_transplant": transplant,
                "surveillance_adherent": adherent,
            }

            # Gene-specific fields
            if gene == "PKD1":
                p["liver_cysts"] = rng.random() < gd.get("liver_cyst_rate", 0.83)
                p["haematuria"]  = rng.random() < gd.get("haematuria_rate", 0.45)
                p["aneurysm"]    = rng.random() < gd.get("aneurysm_rate", 0.08)
            elif gene == "PKD2":
                p["liver_cysts"] = rng.random() < gd.get("liver_cyst_rate", 0.60)
                p["haematuria"]  = rng.random() < gd.get("haematuria_rate", 0.38)
                p["aneurysm"]    = rng.random() < gd.get("aneurysm_rate", 0.04)
            elif gene in ("COL4A5", "COL4A3"):
                p["snhl"]              = rng.random() < gd.get("snhl_rate", 0.78)
                p["anterior_lenticonus"] = rng.random() < gd.get("anterior_lenticonus_rate", 0.22)
                p["haematuria"]        = True
            elif gene == "NPHS1":
                p["thrombosis"]  = rng.random() < gd.get("thrombosis_rate", 0.25)
                p["infection"]   = rng.random() < gd.get("infection_rate", 0.55)
                p["congenital"]  = True
            elif gene == "NPHS2":
                p["steroid_resistant"] = True
                p["fsgs_on_biopsy"]    = True
            elif gene == "UMOD":
                p["gout"]          = rng.random() < gd.get("gout_rate", 0.88)
                p["medullary_cyst"] = rng.random() < gd.get("medullary_cyst_rate", 0.55)
            elif gene == "HNF1B":
                p["diabetes"]         = rng.random() < gd.get("diabetes_rate", 0.85)
                p["hypomagnesaemia"]  = rng.random() < gd.get("hypomagnesaemia_rate", 0.70)
                p["pancreatic_atrophy"] = rng.random() < gd.get("pancreatic_atrophy_rate", 0.75)
                p["gu_anomaly"]       = (sex == "F") and rng.random() < gd.get("gu_anomaly_rate", 0.38)

            patients.append(p)
    return patients


# ─── API response builders ─────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] = sev.get(p["severity"], 0) + 1

    esrd_n       = sum(1 for p in patients if p["esrd"])
    htn_n        = sum(1 for p in patients if p["hypertension"])
    drug_err_n   = sum(1 for p in patients if p["drug_error"])
    delay_n      = sum(1 for p in patients if p["diagnosis_delayed"])
    transplant_n = sum(1 for p in patients if p["received_transplant"])
    adherent_n   = sum(1 for p in patients if p["surveillance_adherent"])

    # Gene-level stats
    gene_stats = {}
    for gd in NEPHROPATHY_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        ng = len(gpts)
        gene_stats[gd["gene"]] = {
            "esrd_pct":       round(100 * sum(1 for p in gpts if p["esrd"]) / ng, 1),
            "htn_pct":        round(100 * sum(1 for p in gpts if p["hypertension"]) / ng, 1),
            "drug_error_pct": round(100 * sum(1 for p in gpts if p["drug_error"]) / ng, 1),
            "transplant_pct": round(100 * sum(1 for p in gpts if p["received_transplant"]) / ng, 1),
            "gfr_pattern":    gd["gfr_pattern"],
            "proteinuria":    gd["proteinuria_pattern"],
        }

    disease_cat = {
        "ADPKD type 1 (PKD1)":                    round(100 * 40 / n, 1),
        "ADPKD type 2 (PKD2)":                    round(100 * 40 / n, 1),
        "Alport X-linked (COL4A5)":               round(100 * 40 / n, 1),
        "Alport AR (COL4A3)":                     round(100 * 40 / n, 1),
        "Congenital Nephrotic Finnish (NPHS1)":   round(100 * 40 / n, 1),
        "FSGS type 1 / SRNS (NPHS2)":            round(100 * 40 / n, 1),
        "Uromodulin-assoc. KD / FJHN (UMOD)":    round(100 * 40 / n, 1),
        "MODY5 / RCAD (HNF1B)":                  round(100 * 40 / n, 1),
    }

    kpis = [
        {"label": "Total Patients",   "value": str(n)},
        {"label": "Genes Covered",    "value": "8"},
        {"label": "ESRD Rate",        "value": f"{round(100*esrd_n/n,1)}%"},
        {"label": "Hypertension",     "value": f"{round(100*htn_n/n,1)}%"},
        {"label": "Transplant Rate",  "value": f"{round(100*transplant_n/n,1)}%"},
        {"label": "Drug Error",       "value": f"{round(100*drug_err_n/n,1)}%"},
        {"label": "Delayed Dx",       "value": f"{round(100*delay_n/n,1)}%"},
        {"label": "Surveillance OK",  "value": f"{round(100*adherent_n/n,1)}%"},
    ]

    return {
        "atlas_name": "Nephropathy-Atlas",
        "atlas_subtitle": (
            "PKD1·PKD2·COL4A5·COL4A3·NPHS1·NPHS2·UMOD·HNF1B — "
            "320 patients (8×40, seeds 1158–1165)"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1158–1165",
        "description": (
            "Comprehensive atlas of 8 major hereditary nephropathy disorders: "
            "PKD1/ADPKD type 1 (AD; most common, 78% ADPKD; polycystin-1; bilateral cysts; "
            "tolvaptan FDA 2018 — HEPATOTOXICITY; Berry aneurysm 5–10%; ESRD ~55 yrs; "
            "truncating variants → worse prognosis; Mayo TKV classification 1A-E); "
            "PKD2/ADPKD type 2 (AD; 15% ADPKD; polycystin-2/TRPP2; milder; ESRD ~68 yrs; "
            "10 yrs later than PKD1; same tolvaptan pathway; lower aneurysm risk); "
            "COL4A5/Alport XL (XLR males; collagen IV alpha-5; haematuria + SNHL + "
            "anterior lenticonus PATHOGNOMONIC; ESRD males 30–40 yrs; ACEi started pre-proteinuria "
            "delays ESRD 16 yrs; females 25–30% develop CKD — not benign carriers); "
            "COL4A3/Alport AR (AR; most severe; ESRD by 20–30; consanguinity; "
            "NC1 = Goodpasture antigen — anti-GBM post-transplant risk); "
            "NPHS1/CNF (AR; nephrin; massive proteinuria at birth; elevated amniotic AFP; "
            "Fin-major/Fin-minor Finnish founder; transplant curative but 25% recurrence "
            "anti-nephrin antibody); "
            "NPHS2/FSGS1 (AR; podocin; childhood SRNS; R138Q European founder; "
            "steroid-resistant; low transplant recurrence <5% unlike primary FSGS 30–50%); "
            "UMOD/UAKD (AD; uromodulin/TAL; gout in teens PATHOGNOMONIC; slow CKD; "
            "NSAIDs ABSOLUTELY CI; allopurinol mandatory); "
            "HNF1B/MODY5-RCAD (AD; 50%+ de novo; renal cysts + diabetes + hypomagnesaemia + "
            "pancreatic atrophy + genitourinary anomalies; "
            "whole-gene 17q12 deletion — MLPA NOT sequencing; sulfonylureas INEFFECTIVE — insulin)."
        ),
        "aggregate_clinical": {
            "esrd_pct":             round(100 * esrd_n / n, 1),
            "hypertension_pct":     round(100 * htn_n / n, 1),
            "drug_error_pct":       round(100 * drug_err_n / n, 1),
            "diagnosis_delayed_pct": round(100 * delay_n / n, 1),
            "transplant_rate_pct":  round(100 * transplant_n / n, 1),
            "surveillance_adherent_pct": round(100 * adherent_n / n, 1),
            "severity_mild_pct":    round(100 * sev["Mild"] / n, 1),
            "severity_moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severity_severe_pct":  round(100 * sev["Severe"] / n, 1),
        },
        "drug_alerts": [
            {
                "type": "danger",
                "title": "TOLVAPTAN (PKD1/PKD2) — Hepatotoxicity: LFTs Monthly ×18 Months",
                "body": (
                    "Tolvaptan (V2-receptor antagonist; FDA 2018) slows ADPKD progression "
                    "(TEMPO 3:4: 49% slower TKV growth, 26% slower eGFR decline) but carries "
                    "serious hepatotoxicity risk. Monthly LFTs mandatory for first 18 months. "
                    "ABSOLUTELY CONTRAINDICATED in liver disease or significant alcohol use. "
                    "Aquaresis (polyuria/polydipsia/nocturia) expected — ensure free water access. "
                    "Eligibility: Mayo class 1C-1E or eGFR decline ≥2.5 mL/min/1.73m²/yr."
                ),
            },
            {
                "type": "danger",
                "title": "NSAIDs (UMOD/UAKD) — ABSOLUTELY CONTRAINDICATED — Accelerate ESRD",
                "body": (
                    "In UAKD (UMOD mutations), NSAIDs block prostaglandin-mediated afferent arteriolar "
                    "dilatation in already-compromised thick ascending limb → acute-on-chronic kidney "
                    "injury → accelerated ESRD. Gout flares (common, presenting in teens) MUST be "
                    "treated with colchicine or corticosteroids — NEVER NSAIDs. "
                    "Allopurinol reduces uric acid and prevents attacks long-term."
                ),
            },
            {
                "type": "danger",
                "title": "SULFONYLUREAS INEFFECTIVE in HNF1B / MODY5 — Insulin Required",
                "body": (
                    "HNF1B/MODY5 diabetes arises from impaired beta-cell differentiation and "
                    "reduced beta-cell neogenesis. The KATP channel (sulfonylurea target) may be "
                    "intact but functional beta-cell mass is critically reduced. Sulfonylureas "
                    "fail — insulin is required. Misdiagnosis as T2DM leads to years of ineffective "
                    "sulfonylurea therapy with poor glycaemic control. Renal cysts + young-onset "
                    "diabetes = genetic testing MANDATORY."
                ),
            },
            {
                "type": "warning",
                "title": "ACEi/ARBs (COL4A5/COL4A3 Alport) — Start BEFORE Proteinuria Appears",
                "body": (
                    "In Alport syndrome (COL4A5 XL / COL4A3 AR), ACEi/ARB therapy started "
                    "at the HAEMATURIA stage (before proteinuria develops) delays ESRD by up to "
                    "16 years (Gross 2012 cohort: treatment median ESRD 49 yrs vs 33 yrs untreated). "
                    "Do not wait for proteinuria — start at diagnosis of Alport regardless of "
                    "urine ACR. Monitor K⁺ and GFR."
                ),
            },
            {
                "type": "warning",
                "title": "HNF1B 17q12 Deletion — MLPA Required (Sequencing Misses 50% of Cases)",
                "body": (
                    "Whole-gene 17q12 microdeletion accounts for ~50% of HNF1B pathogenic variants "
                    "and is NOT detected by standard Sanger or NGS panel sequencing. "
                    "MLPA (multiplex ligation-dependent probe amplification) or chromosomal SNP array "
                    "MUST be ordered when HNF1B disease is suspected (renal cysts + diabetes + "
                    "hypomagnesaemia + pancreatic atrophy). Sequencing-only workup will miss half the diagnoses."
                ),
            },
            {
                "type": "warning",
                "title": "NPHS1 Post-Transplant — 25% Recurrence (Anti-Nephrin Antibodies)",
                "body": (
                    "Finnish congenital nephrotic syndrome (NPHS1) is caused by complete nephrin absence. "
                    "After renal transplantation, the 'foreign' nephrin in the donated kidney can trigger "
                    "de novo anti-nephrin IgG antibody formation in 25% of recipients → "
                    "transplant FSGS recurrence. Counsel families before transplant. "
                    "Treat recurrence with plasmapheresis + rituximab (anti-CD20)."
                ),
            },
        ],
        "critical_rules": [
            "PKD1/PKD2 Berry aneurysm: MRA screening if positive family history SAH or at-risk occupation; ≥5 mm → neurosurgical referral",
            "PKD infected cysts: fluoroquinolones (ciprofloxacin) penetrate cysts; beta-lactams DO NOT",
            "COL4A5 females: 25–30% develop significant CKD — NOT benign carriers; monitor annually",
            "COL4A3/COL4A5 anti-GBM post-transplant: monitor anti-GBM antibody titres; plasma exchange if positive",
            "NPHS2 FSGS: <5% transplant recurrence — vs primary FSGS 30–50%; genetic testing before transplant",
            "UMOD gout in teens: PATHOGNOMONIC — check urinary uromodulin + UMOD gene; NSAIDs ABSOLUTELY CI",
            "HNF1B females: uterus didelphys/bicornuate in 38% — gynaecological assessment mandatory",
            "NPHS1 congenital nephrotic: elevated amniotic AFP is prenatal diagnostic clue",
        ],
        "pathway_targets": {
            "PKD1": "mTOR (cyst growth) + cAMP/V2R (tolvaptan) + mTOR (everolimus—failed GFR endpoint)",
            "PKD2": "TRPP2 Ca²⁺ channel + cAMP → same V2R/tolvaptan pathway as PKD1",
            "COL4A5": "COL4A5 gene therapy (experimental) + RAAS (ACEi/ARB — standard of care)",
            "COL4A3": "RAAS (ACEi/ARB) + anti-GBM surveillance post-transplant",
            "NPHS1": "Slit diaphragm integrity + PI3K/Akt signalling + anti-nephrin plasmapheresis post-Tx",
            "NPHS2": "Lipid raft slit diaphragm complex + calcineurin inhibition (partial) + rituximab",
            "UMOD": "ER proteostasis (chemical chaperones 4-PBA — investigational) + xanthine oxidase (allopurinol)",
            "HNF1B": "Transcriptional regulation (no direct therapy) + KATP channel insulin pathway",
        },
        "kpis": kpis,
        "disease_category_breakdown": disease_cat,
        "gene_level_stats": gene_stats,
        "severity_distribution": sev,
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    genes = []
    for gd in NEPHROPATHY_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        ng = len(gpts)
        genes.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "gene_class": gd["gene_class"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "phenotype": gd["phenotype"],
            "hallmark": gd["hallmark"],
            "key_ddx": gd["key_ddx"],
            "treatment_alert": gd["treatment_alert"],
            "disease_detail": gd["disease"],
            "primary_complication": gd["primary_complication"],
            "gfr_pattern": gd["gfr_pattern"],
            "proteinuria_pattern": gd["proteinuria_pattern"],
            "cohort_n": ng,
            "cohort_stats": {
                "esrd_pct":       round(100 * sum(1 for p in gpts if p["esrd"]) / ng, 1),
                "htn_pct":        round(100 * sum(1 for p in gpts if p["hypertension"]) / ng, 1),
                "drug_error_pct": round(100 * sum(1 for p in gpts if p["drug_error"]) / ng, 1),
                "dx_delayed_pct": round(100 * sum(1 for p in gpts if p["diagnosis_delayed"]) / ng, 1),
                "transplant_pct": round(100 * sum(1 for p in gpts if p["received_transplant"]) / ng, 1),
                "adherent_pct":   round(100 * sum(1 for p in gpts if p["surveillance_adherent"]) / ng, 1),
                "severity": {
                    "Mild":     round(100 * sum(1 for p in gpts if p["severity"] == "Mild") / ng, 1),
                    "Moderate": round(100 * sum(1 for p in gpts if p["severity"] == "Moderate") / ng, 1),
                    "Severe":   round(100 * sum(1 for p in gpts if p["severity"] == "Severe") / ng, 1),
                },
            },
        })
    return {"genes": genes, "n_genes": len(genes), "n_patients": len(patients)}


def get_definitions() -> list:
    return [
        {
            "term": "ADPKD",
            "full": "Autosomal Dominant Polycystic Kidney Disease",
            "explanation": (
                "Most common hereditary kidney disease (1:400–1:1000). Caused by PKD1 (78%) or PKD2 (15%). "
                "Bilateral progressive renal cysts → ESRD. Extrarenal: liver cysts, intracranial aneurysms, "
                "MVP, diverticula. Tolvaptan (V2-antagonist) slows progression — hepatotoxicity risk."
            ),
        },
        {
            "term": "Polycystin-1 / Polycystin-2 Complex",
            "full": "PC1/PC2 mechanosensory Ca²⁺ channel complex in primary cilia",
            "explanation": (
                "PKD1 encodes PC1 (4303 aa, 11 TM domains, GPS cleavage site); PKD2 encodes PC2 (TRPP2 — "
                "6 TM TRP-family Ca²⁺ channel). The two form a receptor-channel complex in primary cilia "
                "that senses tubular fluid flow and regulates Ca²⁺, cAMP, mTOR, and Wnt signalling. "
                "Loss of either → cAMP-driven cyst expansion (V2-receptor antagonist = tolvaptan targets this)."
            ),
        },
        {
            "term": "Mayo TKV Classification",
            "full": "Mayo Imaging Classification 1A-E for ADPKD prognosis",
            "explanation": (
                "Total kidney volume (TKV) adjusted for height (height-adjusted TKV, HtTKV) on MRI/CT "
                "classifies ADPKD progression risk: 1A (slowest) → 1E (fastest, ≥8% TKV growth/year). "
                "Class 1C-1E eligible for tolvaptan. Best prognostic marker for ESRD timing."
            ),
        },
        {
            "term": "Alport Syndrome Basket-Weave GBM",
            "full": "Multilamellation of the glomerular basement membrane on electron microscopy",
            "explanation": (
                "Pathognomonic for Alport syndrome on EM: early GBM thinning → irregular thickening → "
                "multilamellation with splitting into multiple electron-dense laminae giving 'basket-weave' "
                "or 'moth-eaten' appearance. Caused by absence of alpha345 collagen IV network "
                "(COL4A3/4/5) → persistence of fetal alpha112 network → GBM remodelling failure. "
                "COL4A5 IHC absent on skin/kidney biopsy in XL males — diagnostic."
            ),
        },
        {
            "term": "Anterior Lenticonus",
            "full": "Anterior bulging of the central lens — Alport syndrome PATHOGNOMONIC",
            "explanation": (
                "Bilateral anterior protrusion of the lens nucleus through a weakened anterior lens capsule "
                "(basement membrane lacks COL4A5). Slit-lamp appearance: 'oil droplet' reflex. "
                "PATHOGNOMONIC for Alport syndrome — seen in ~20–30% of COL4A5 XL males. "
                "Progresses over time → oil-droplet reflex → anterior cortical opacities → cataract. "
                "Distinguishes Alport from thin BM disease (TBMN) which lacks lenticonus."
            ),
        },
        {
            "term": "Goodpasture Antigen / NC1 Domain (COL4A3)",
            "full": "COL4A3 NC1 domain — target of anti-GBM antibodies in Goodpasture's disease",
            "explanation": (
                "The NC1 domain of COL4A3 contains the 'Goodpasture antigen' — "
                "the epitope recognised by anti-GBM autoantibodies in Goodpasture's disease "
                "(RPGN + pulmonary haemorrhage). In AR Alport (biallelic COL4A3 LOF), "
                "patients lack COL4A3 NC1 entirely; after transplantation, the foreign COL4A3 NC1 "
                "in the donor kidney can trigger de novo anti-GBM disease (3–5%). "
                "Monitor post-transplant anti-GBM antibody titres."
            ),
        },
        {
            "term": "Slit Diaphragm",
            "full": "Glomerular filtration slit diaphragm — nephrin-podocin scaffold",
            "explanation": (
                "The slit diaphragm spans the ~40 nm filtration slit between adjacent podocyte foot processes. "
                "Structural backbone: nephrin (NPHS1) molecules from adjacent foot processes interact "
                "homophilically across the slit; podocin (NPHS2) anchors nephrin to cholesterol-rich "
                "lipid rafts. Loss of either → foot process effacement → proteinuria. "
                "NPHS1 loss → congenital nephrotic (massive); NPHS2 loss → childhood FSGS (steroid-resistant)."
            ),
        },
        {
            "term": "Fin-Major / Fin-Minor Mutations (NPHS1)",
            "full": "Finnish founder mutations in NPHS1 causing congenital nephrotic syndrome",
            "explanation": (
                "Fin-major: c.121_122insC (p.Leu41Phefs*4) — frameshift; accounts for 78% of Finnish CNF alleles. "
                "Fin-minor: c.3325C>T (p.Arg1109Ter) — nonsense; accounts for 16% of Finnish alleles. "
                "Together cover ~94% of Finnish CNF alleles. "
                "Gene frequency 1:200 in Finland (1:8,000 disease incidence). "
                "Non-Finnish cases have diverse non-founder mutations."
            ),
        },
        {
            "term": "SRNS — Steroid-Resistant Nephrotic Syndrome",
            "full": "Nephrotic syndrome failing to achieve remission with standard prednisolone",
            "explanation": (
                "Defined as failure to achieve complete remission (urine protein:creatinine <0.2) "
                "after ≥8 weeks of daily prednisolone (60 mg/m²/day). SRNS is the major indication "
                "for genetic testing — NPHS2 (podocin), NPHS1, WT1, LAMB2, TRPC6, INF2 mutations "
                "account for ~25% of childhood SRNS. Genetic SRNS has low transplant recurrence "
                "(<5%) vs primary FSGS (30–50%). Genetic diagnosis prevents unnecessary prolonged "
                "steroid courses with all associated toxicity."
            ),
        },
        {
            "term": "Uromodulin / Tamm-Horsfall Protein (UMOD)",
            "full": "Most abundant urinary protein — exclusively from thick ascending limb",
            "explanation": (
                "Uromodulin (THP) is a GPI-anchored glycoprotein produced exclusively by TAL epithelial "
                "cells; ~50 mg/day in healthy adults (most abundant urine protein). "
                "Functions: antibacterial (traps type 1 fimbriated E. coli), crystal inhibitor "
                "(prevents calcium oxalate stones), immune modulator (TLR4). "
                "UMOD mutations → misfolding → ER retention → TAL stress → tubulointerstitial nephritis → CKD. "
                "Low urinary + plasma uromodulin = functional biomarker of UAKD."
            ),
        },
        {
            "term": "UAKD / FJHN / MCKD2",
            "full": "Uromodulin-associated kidney disease / Familial Juvenile Hyperuricaemic Nephropathy / Medullary Cystic Kidney Disease type 2",
            "explanation": (
                "Three historical names for the same AD condition caused by UMOD mutations. "
                "FJHN emphasises the hyperuricaemia/gout in teens (most striking clinical clue); "
                "MCKD2 emphasises the small medullary cysts; UAKD is the current preferred term. "
                "Key clinical red flag: GOUT BEFORE AGE 30 → test urinary uromodulin → UMOD sequencing. "
                "NSAIDs for gout ABSOLUTELY CONTRAINDICATED — use colchicine."
            ),
        },
        {
            "term": "RCAD / MODY5",
            "full": "Renal Cysts And Diabetes syndrome / Maturity-Onset Diabetes of the Young type 5",
            "explanation": (
                "HNF1B spectrum disease: renal cysts (prenatal or childhood) + MODY5 diabetes "
                "(young adult; insulin-requiring) + hypomagnesaemia + pancreatic atrophy (body/tail on MRI) "
                "+ genitourinary anomalies (uterus didelphys 38% females). "
                "50%+ de novo mutations. Whole-gene 17q12 deletion in 50% — MLPA required. "
                "Sulfonylureas fail (beta-cell developmental defect) — insulin mandatory."
            ),
        },
        {
            "term": "17q12 Microdeletion (HNF1B)",
            "full": "Whole-gene deletion of HNF1B detected by MLPA or SNP array — not sequencing",
            "explanation": (
                "The 17q12 region contains HNF1B plus several neighbouring genes. "
                "Whole-gene HNF1B deletion accounts for ~50% of pathogenic HNF1B variants "
                "but is INVISIBLE to Sanger sequencing and most NGS panels (reads from both "
                "alleles present — hemizygous deletion not detected). "
                "MLPA (multiplex ligation-dependent probe amplification) or chromosomal SNP microarray "
                "is required. Standard clinical practice: order both sequencing AND MLPA when "
                "suspecting HNF1B disease."
            ),
        },
        {
            "term": "Anti-Nephrin Antibody Post-Transplant Recurrence (NPHS1)",
            "full": "De novo anti-nephrin IgG causing post-transplant FSGS recurrence in CNF recipients",
            "explanation": (
                "Patients with biallelic NPHS1 loss (CNF) have never expressed nephrin — "
                "the donated kidney introduces 'foreign' nephrin antigen. "
                "In ~25% of transplants, de novo anti-nephrin IgG forms → deposits on slit diaphragm → "
                "FSGS recurrence with nephrotic syndrome. Treat: plasmapheresis (remove antibody) "
                "+ rituximab (anti-CD20; deplete B cells). Counsel families pre-transplant."
            ),
        },
        {
            "term": "Tolvaptan V2-Receptor Antagonist",
            "full": "Vasopressin-2 receptor antagonist slowing ADPKD cyst expansion via cAMP reduction",
            "explanation": (
                "Vasopressin (AVP) binds V2-receptor on principal cells → cAMP elevation → CFTR activation → "
                "Cl⁻ secretion into cysts → fluid accumulation → cyst expansion. "
                "Tolvaptan blocks V2R → reduces cAMP → slows cyst growth (TEMPO 3:4: 49% slower TKV growth) "
                "and slows eGFR decline (26%). Aquaresis (polyuria/polydipsia) expected. "
                "Hepatotoxicity: serious/fatal; LFTs monthly ×18 months; CI in liver disease. "
                "Approved FDA 2018, EMA 2015. Patient must maintain >3L/day fluid intake."
            ),
        },
    ]


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(f"Overview: {ov['atlas_name']}, n={ov['n_patients']}, genes={ov['n_genes']}")
    bd = get_breakdown()
    print(f"Breakdown: {len(bd['genes'])} genes")
    defs = get_definitions()
    print(f"Definitions: {len(defs)} terms")
    print("OK — Nephropathy-Atlas dashboard self-test passed")
