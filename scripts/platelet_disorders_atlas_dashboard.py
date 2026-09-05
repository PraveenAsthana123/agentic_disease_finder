#!/usr/bin/env python3
"""Platelet-Disorders-Atlas — Complete 8-Gene Hereditary Platelet Disorders Atlas
GP1BA   (Glycoprotein Ib alpha; 627 aa; 17p13.2; OMIM gene 606672;
         Bernard-Soulier syndrome (BSS) — AR biallelic; GIANT PLATELETS PATHOGNOMONIC;
         VWF receptor; platelet count low-normal but massively enlarged;
         DESMOPRESSIN INEFFECTIVE (no GPIb to release VWF-bound platelets);
         Platelet transfusion for bleeding; alloantibody risk with transfusion) ·
ITGA2B  (Integrin alpha-IIb / GPIIb; 1008 aa; 17q21.31; OMIM gene 607759;
         Glanzmann thrombasthenia type A (GT-A) — AR; NORMAL PLATELET COUNT;
         fibrinogen/VWF binding abolished → NO platelet aggregation to ANY agonist;
         Activated recombinant FVII (NovoSeven/eptacog alfa) for surgery;
         HLA-alloimmunization from platelet transfusions limits future use) ·
ITGB3   (Integrin beta-3 / GPIIIa; 788 aa; 17q21.32; OMIM gene 173470;
         Glanzmann thrombasthenia type B (GT-B) — AR; SAME phenotype as GT-A;
         HPA-1a antigen located on beta-3 → risk of neonatal alloimmune thrombocytopenia;
         ANTI-GPIIb/IIIa antibodies in transfused patients; IV anti-fibrinolytic adjunct) ·
MYH9    (Myosin heavy chain 9 / non-muscle myosin IIA; 1960 aa; 22q11.2; OMIM gene 160775;
         MYH9-related disease (MYH9-RD) — AD; GIANT PLATELETS + DÖHLE-LIKE INCLUSIONS IN NEUTROPHILS;
         May-Hegglin anomaly; Fechtner / Epstein / Sebastian syndromes — same gene;
         Sensorineural hearing loss 35-60%; Nephritis (Alport-like) 25-70%; Cataracts 10-20%;
         AVOID ASPIRIN + NSAIDs; ACE inhibitor mandatory when nephritis develops) ·
NBEAL2  (Neurobeachin-like 2; 2806 aa; 3p21.31; OMIM gene 614169;
         Gray platelet syndrome (GPS) — AR biallelic; ALPHA-GRANULE ABSENT PATHOGNOMONIC;
         pale 'gray' platelets on Wright-Giemsa stain; thrombocytopenia + myelofibrosis;
         Myelofibrosis develops in 3rd-4th decade — bone marrow monitoring MANDATORY;
         splenomegaly nearly universal; VWF multimer analysis shows normal) ·
ANKRD26 (Ankyrin repeat domain-containing protein 26; 1982 aa; 10p12.1; OMIM gene 610855;
         Thrombocytopenia-2 (THC2) — AD; PROMOTER MUTATIONS in 5'UTR regulatory region;
         myeloid malignancy (AML/MDS) risk 5-8%; CBC annually mandatory;
         AVOID thrombopoietin receptor agonists (eltrombopag/romiplostim) — risk unquantified;
         platelet count modestly low: 20-150k; mild bleeding; marrow aspirate normal morphology) ·
GFI1B   (Growth factor independent 1B transcription repressor; 330 aa; 9q34.13; OMIM gene 604383;
         GFI1B-related thrombocytopenia — AD; RED CELL MACROCYTOSIS PATHOGNOMONIC;
         absent delta-granules (dense bodies) → impaired secondary aggregation;
         CD34 expression on platelets (immature platelet marker) → diagnostic flow cytometry;
         variable bleeding; platelet transfusion often required perioperatively) ·
RUNX1   (Runt-related transcription factor 1; 453 aa; 21q22.12; OMIM gene 151385;
         Familial platelet disorder with predisposition to AML (FPD/AML) — AD;
         AML/MDS lifetime risk 35-44%; annual CBC + bone marrow biopsy every 2-3 years;
         GENETIC COUNSELLING + haematology surveillance ALL first-degree relatives;
         NO aspirin; NO NSAIDs; stem cell transplant curative if MDS/AML develops)
320-patient aggregate cohort (8 x 40, seeds 1334-1341)
"""

import random

SEED_BASE = 1334

PLATELET_GENES = [
    # ── GP1BA — Bernard-Soulier Syndrome (AR) ──
    {
        "gene": "GP1BA",
        "protein": "Glycoprotein Ib alpha chain",
        "alias": (
            "GP1BA; OMIM gene 606672; Bernard-Soulier syndrome (BSS) #231200; autosomal recessive; 17p13.2; "
            "627 aa; ~71 kDa (mature chain after signal peptide cleavage); "
            "forms GPIb-IX-V complex on platelet surface with GP1BB (GPIb-beta), GP9 (GPIX), GP5 (GPV); "
            "GPIb-alpha binds von Willebrand factor (VWF) A1 domain → platelet adhesion at high shear; "
            "also binds thrombin, P-selectin, Mac-1 (CD11b/CD18); "
            "LOF biallelic → absent or dysfunctional GPIb complex → NO VWF-mediated adhesion → "
            "giant platelets (megathrombocytes) + thrombocytopenia + severe mucocutaneous bleeding"
        ),
        "aa": "627 aa",
        "kDa": "~71 kDa",
        "locus": "17p13.2",
        "omim_gene": 606672,
        "omim_disease": 231200,
        "inheritance": (
            "Autosomal recessive; biallelic LOF; 25% risk to siblings; "
            "heterozygous carriers: asymptomatic (no bleeding, mild platelet changes); "
            "consanguinity increases risk; Velázquez variant (p.Ala172Val) common in Middle East; "
            "rare AD GOF variants cause Platelet-type (pseudo-) VWD — very different disease"
        ),
        "gene_class": (
            "GP1BA encodes the alpha subunit of the GPIb-IX-V receptor complex, the major platelet "
            "surface receptor for von Willebrand factor (VWF). "
            "FUNCTION: GPIb-alpha (GP1BA) + GPIb-beta (GP1BB) form a disulfide-linked heterodimer; "
            "GPIb-alpha carries the VWF-binding domain (LRR region, leucine-rich repeats 1-9) and "
            "the macroglycopeptide (sialyl residues) that create the vascular 'cloud' keeping platelets "
            "from inadvertent activation. "
            "PATHOPHYSIOLOGY: Biallelic LOF → GPIb complex not assembled or unstable → "
            "platelets cannot bind VWF at high-shear sites → profound adhesion defect at wound edges → "
            "severe mucocutaneous bleeding (epistaxis, gingival, menorrhagia, GI, surgical). "
            "GIANT PLATELETS: In BSS, absent GPIb complex allows uncontrolled demarcation membrane "
            "expansion during thrombopoiesis → platelet diameter often >7 µm (normal 2-3 µm), "
            "can approach lymphocyte size → confused with thrombocytopenia on automated analyzers. "
            "RISTOCETIN AGGREGATION: VWF-ristocetin agglutination ABSENT in BSS "
            "(unlike von Willebrand disease where DDAVP may help — DDAVP does NOT help BSS)."
        ),
        "phenotype": (
            "Severe mucocutaneous bleeding from infancy: epistaxis, gum bleeding, menorrhagia (often catastrophic), "
            "GI bleeding, post-circumcision bleeding; "
            "Platelet count: 20-100k (mild-moderate thrombocytopenia); "
            "GIANT PLATELETS: volume 40-100 fL (normal 7-11 fL); diameter approaches lymphocyte size → "
            "automated analyzers undercount (morphology review MANDATORY); "
            "Clot retraction: NORMAL (fibrinogen binding via GPIIb/IIIa intact); "
            "Aggregation: ristocetin agglutination ABSENT; ADP/collagen/thrombin aggregation preserved; "
            "PFA-100: closure time prolonged; bleeding time: markedly prolonged"
        ),
        "key_hallmarks": [
            "Giant platelets (megathrombocytes) on blood film — diameter approaches lymphocyte size, PATHOGNOMONIC",
            "Ristocetin agglutination ABSENT — GPIb-alpha cannot bind VWF; distinguishes from GT (normal ristocetin)",
            "DESMOPRESSIN (DDAVP) INEFFECTIVE — no GPIb receptor to capture released VWF multimers at wound site",
            "Normal clot retraction — GPIIb/IIIa intact; contrast with Glanzmann thrombasthenia (absent retraction)",
            "Flow cytometry: absent CD42b (GPIb-alpha) — diagnostic; GPIIb/IIIa (CD41/CD61) NORMAL",
        ],
        "treatment_alerts": [
            "Platelet transfusion is the mainstay for bleeding episodes — but alloimmunization risk with repeated transfusions",
            "DESMOPRESSIN (DDAVP): INEFFECTIVE in BSS — do NOT use (no GPIb receptor to capture released VWF)",
            "Antifibrinolytics (tranexamic acid, EACA): useful adjunct for mucosal bleeding and minor procedures",
            "HLA-matched platelet transfusions preferred to reduce alloimmunization; leukodepleted",
            "Recombinant FVIIa (NovoSeven): rescue option if platelet refractory; limited evidence in BSS",
            "Female patients: oral contraceptive pill or LNG-IUS (Mirena) mandatory to manage menorrhagia",
        ],
        "ddx": [
            "Glanzmann thrombasthenia (ITGA2B/ITGB3): normal platelet count, NORMAL platelet size, clot retraction ABSENT, ristocetin NORMAL",
            "MYH9-related disease: giant platelets + Döhle bodies in neutrophils; GPIb-alpha PRESENT on flow; AD inheritance",
            "Gray platelet syndrome (NBEAL2): large pale platelets; alpha-granule absent; GPIb normal; AR",
            "Immune thrombocytopenia (ITP): normal platelet size; autoantibodies; response to IVIg/steroids",
            "Platelet-type (pseudo-) VWD: GOF GP1BA variant; VWF-ristocetin hypersensitivity — OPPOSITE finding to BSS",
        ],
        "seed": SEED_BASE + 0,
        "n_patients": 40,
        "age_range": (2, 40),
        "female_pct": 52,
    },
    # ── ITGA2B — Glanzmann Thrombasthenia Type A (AR) ──
    {
        "gene": "ITGA2B",
        "protein": "Integrin alpha-IIb (GPIIb)",
        "alias": (
            "ITGA2B; OMIM gene 607759; Glanzmann thrombasthenia type A (GT-A) #273800; autosomal recessive; 17q21.31; "
            "1008 aa (pro-alpha-IIb); ~138 kDa; integrin alpha subunit; "
            "pairs obligatorily with integrin beta-3 (ITGB3) to form GPIIb/IIIa (integrin alphaIIb-beta3); "
            "GPIIb/IIIa is the most abundant platelet surface receptor (~80,000 copies/platelet); "
            "activated GPIIb/IIIa binds fibrinogen (and VWF, fibronectin, vitronectin) → "
            "platelet-to-platelet bridging = AGGREGATION; "
            "biallelic LOF → absent GPIIb/IIIa → NO aggregation to ANY agonist → GT-A"
        ),
        "aa": "1008 aa",
        "kDa": "~138 kDa",
        "locus": "17q21.31",
        "omim_gene": 607759,
        "omim_disease": 273800,
        "inheritance": (
            "Autosomal recessive; biallelic LOF or dysfunctional alleles; 25% sib risk; "
            "consanguinity increases risk (Romany/Ashkenazi/Iraqi-Jewish founder variants); "
            "heterozygous carriers: asymptomatic with normal platelet function (sufficient GPIIb/IIIa); "
            "ITGA2B and ITGB3 map to 17q21 — same chromosome arm — panel both genes"
        ),
        "gene_class": (
            "ITGA2B encodes integrin alpha-IIb (GPIIb), the alpha-subunit of the integrin alphaIIb-beta3 "
            "heterodimer (GPIIb/IIIa). "
            "FUNCTION: GPIIb/IIIa is the final common pathway of platelet aggregation. "
            "After platelet activation (by ADP, collagen, thrombin, thromboxane A2), inside-out signalling "
            "via talin-1 and kindlin-3 shifts GPIIb/IIIa from inactive (bent) to extended/open (high-affinity) "
            "conformation — exposing the RGD-binding pocket on beta-3. "
            "Fibrinogen then bridges platelets via simultaneous binding to GPIIb/IIIa on adjacent platelets → "
            "platelet plug formation. "
            "PATHOPHYSIOLOGY: ITGA2B biallelic LOF → absent or dysfunctional GPIIb → "
            "GPIIb/IIIa not assembled or surface expressed → fibrinogen binding abolished → "
            "NO platelet aggregation to ANY agonist (ADP, collagen, thrombin, arachidonic acid, PAF). "
            "Platelet adhesion (GPIb-VWF) is PRESERVED (GPIb intact). "
            "Clot retraction ABSENT (requires GPIIb/IIIa for fibrin-cytoskeleton linkage). "
            "DISTINGUISHING GT-A from GT-B: Both have same clinical phenotype; "
            "flow cytometry distinguishes: GT-A = absent GPIIb (CD41), GT-B = absent GPIIIa (CD61). "
            "Platelet count NORMAL — only function defective."
        ),
        "phenotype": (
            "Severe mucocutaneous bleeding: epistaxis, gingival, menorrhagia, GI, post-traumatic; "
            "NORMAL platelet count (150-400k) — CRITICAL FEATURE (no thrombocytopenia); "
            "NORMAL platelet size and morphology on blood film; "
            "Aggregation: ABSENT to ALL agonists (ADP, collagen, thrombin, arachidonic acid, ristocetin-VWF agglutination NORMAL); "
            "Clot retraction: ABSENT (specific to GT); "
            "PFA-100: profoundly prolonged collagen-ADP + collagen-epinephrine closure times; "
            "Fibrinogen binding to platelets: ABSENT on flow cytometry (PAC-1 antibody)"
        ),
        "key_hallmarks": [
            "NORMAL platelet count + NORMAL platelet morphology — but ZERO platelet aggregation to ALL agonists",
            "Absent clot retraction — PATHOGNOMONIC for GT (GPIIb/IIIa required for fibrin-cytoskeleton linkage)",
            "Ristocetin agglutination NORMAL — GPIb intact; key DDx from Bernard-Soulier syndrome",
            "Flow cytometry: absent CD41 (GPIIb) = GT-A; test ITGA2B; CD61 (GPIIIa) reduced in proportion",
            "Activated recombinant FVII (NovoSeven) for surgical haemostasis — preferred over platelet transfusion",
        ],
        "treatment_alerts": [
            "Recombinant FVIIa (NovoSeven 90 mcg/kg IV q2h): treatment of choice for significant bleeding and surgery",
            "Platelet transfusion: effective but HLA/HPA alloimmunization with repeated use → REFRACTORY to future transfusions",
            "Antifibrinolytics (tranexamic acid): ALWAYS use as adjunct — reduces transfusion need",
            "DESMOPRESSIN (DDAVP): NOT effective in GT (GPIIb/IIIa defect, not VWF release issue)",
            "HLA-matched, leukodepleted platelets preferred to limit alloimmunization; monitor for anti-HLA and anti-HPA antibodies",
            "Menorrhagia: MANDATORY management — LNG-IUS or oral contraceptive; plan obstetric multidisciplinary care",
        ],
        "ddx": [
            "GT-B (ITGB3): clinically IDENTICAL — flow cytometry distinguishes: GT-A absent CD41, GT-B absent CD61",
            "Bernard-Soulier syndrome (GP1BA): GIANT platelets, thrombocytopenia, ristocetin agglutination ABSENT; GT has normal platelet size + count",
            "Aspirin/NSAID effect: aggregation REDUCED (not absent); reversible; history and drug washout distinguish",
            "Storage pool deficiency (delta-granule, GPS): partial aggregation defect; aggregation reduced not absent; granule studies abnormal",
            "Afibrinogenaemia: absent fibrinogen → absent aggregation; fibrinogen assay = diagnostic; PT + aPTT also prolonged",
        ],
        "seed": SEED_BASE + 1,
        "n_patients": 40,
        "age_range": (1, 45),
        "female_pct": 55,
    },
    # ── ITGB3 — Glanzmann Thrombasthenia Type B (AR) ──
    {
        "gene": "ITGB3",
        "protein": "Integrin beta-3 (GPIIIa)",
        "alias": (
            "ITGB3; OMIM gene 173470; Glanzmann thrombasthenia type B (GT-B) #273800; autosomal recessive; 17q21.32; "
            "788 aa; ~87 kDa; integrin beta-3 subunit; "
            "pairs with alpha-IIb (ITGA2B) on platelets → GPIIb/IIIa (aggregation receptor); "
            "ALSO pairs with alpha-V (ITGAV) in endothelium, osteoclasts → alpha-V-beta-3 (vitronectin receptor); "
            "HPA-1a (PLA1) antigen located on ITGB3 (Leu33Pro → HPA-1a/1b) → "
            "neonatal alloimmune thrombocytopenia (NAIT) most common cause; "
            "biallelic LOF → same GT phenotype as ITGA2B (type A); flow cytometry distinguishes"
        ),
        "aa": "788 aa",
        "kDa": "~87 kDa",
        "locus": "17q21.32",
        "omim_gene": 173470,
        "omim_disease": 273800,
        "inheritance": (
            "Autosomal recessive for GT-B; biallelic LOF variants; 25% sib risk; "
            "HPA-1a/1b alloimmunity: mother HPA-1b/1b (4% of European women) carrying HPA-1a-positive fetus → "
            "maternal anti-HPA-1a IgG crosses placenta → neonatal thrombocytopenia + intracranial haemorrhage risk; "
            "NOT autosomal recessive for NAIT — maternal alloantibody mechanism, not fetal genotype-driven recessive disease"
        ),
        "gene_class": (
            "ITGB3 encodes integrin beta-3 (GPIIIa), the obligate beta-subunit partner of integrin alpha-IIb (GPIIb). "
            "FUNCTION: Beta-3 provides the RGD-binding pocket (MIDAS metal ion-dependent adhesion site) "
            "that engages fibrinogen, VWF, fibronectin; beta-3 also transmits outside-in signalling "
            "that drives full platelet activation, thrombus consolidation, and clot retraction via cytoplasmic "
            "tail interactions with talin, kindlin-3, and myosin. "
            "PATHOPHYSIOLOGY: ITGB3 biallelic LOF → GPIIb/IIIa cannot assemble → "
            "same aggregation defect as ITGA2B LOF (GT-A). "
            "ADDITIONAL ROLE — HPA-1 ANTIGEN: The polymorphism Leu33Pro in ITGB3 defines HPA-1a (Leu33) vs HPA-1b (Pro33). "
            "Anti-HPA-1a maternal IgG → neonatal alloimmune thrombocytopenia (NAIT) — "
            "most severe form of NAIT (intracranial haemorrhage risk up to 10%). "
            "CLINICAL DISTINCTION GT-A vs GT-B: "
            "GT-A: CD41 (GPIIb) absent → ITGA2B gene; "
            "GT-B: CD61 (GPIIIa/beta-3) absent → ITGB3 gene. "
            "Both types are equally severe clinically."
        ),
        "phenotype": (
            "IDENTICAL to GT-A (ITGA2B): severe mucocutaneous bleeding from infancy; "
            "NORMAL platelet count, NORMAL platelet morphology; "
            "ABSENT aggregation to all agonists; ABSENT clot retraction; "
            "PFA-100 profoundly prolonged; "
            "Flow cytometry: CD61 (GPIIIa) ABSENT → confirms GT-B; CD41 may be reduced co-dependently; "
            "HPA-1a typing: relevant for NAIT risk counselling (mother HPA-1b/1b + father HPA-1a+ → NAIT risk in pregnancy)"
        ),
        "key_hallmarks": [
            "Clinically IDENTICAL to GT-A — only flow cytometry distinguishes: GT-B = absent CD61 (GPIIIa/beta-3)",
            "HPA-1a antigen on beta-3: HPA-1b/1b mothers (4%) at risk of NAIT with HPA-1a-positive fetus",
            "NAIT risk: first affected pregnancy may have severe intracranial haemorrhage — screen maternal HPA type in NAIT",
            "Anti-GPIIb/IIIa antibodies in transfused GT patients — from alloimmunization to GPIIb/IIIa epitopes",
            "Absent clot retraction in glass tube: simple bedside test that is 100% sensitive for GT",
        ],
        "treatment_alerts": [
            "Recombinant FVIIa (NovoSeven): first-line for significant bleeding/surgery — same as GT-A",
            "Platelet transfusion: HLA+HPA-1a-matched leukodepleted when possible; alloimmunization limits repeated use",
            "IV antifibrinolytics (tranexamic acid): ALWAYS adjunct; safe in both GT-A and GT-B",
            "NAIT prevention: if prior NAIT pregnancy — IVIg to mother antenatally; cordocentesis for severe cases",
            "NAIT treatment: IVIg + HPA-1a-matched platelet transfusion for thrombocytopaenic neonate",
            "Pregnancy planning: GT-B women need multidisciplinary team; maternal GT separate from NAIT risk",
        ],
        "ddx": [
            "GT-A (ITGA2B): identical phenotype; CD41 absent = GT-A, CD61 absent = GT-B; always panel both genes",
            "Bernard-Soulier (GP1BA): giant platelets + reduced count; ristocetin ABSENT; GT has normal platelet count and size",
            "Afibrinogenaemia: aggregation absent but fibrinogen levels = 0; PT/aPTT prolonged; clot retraction also absent",
            "Drug-induced thrombocytopenia + platelet dysfunction: history; drug withdrawal test; GPIIb/IIIa inhibitor drugs (abciximab, eptifibatide, tirofiban)",
            "Neonatal alloimmune thrombocytopenia (NAIT): fetal/neonatal thrombocytopenia from maternal anti-HPA antibodies; NOT a platelet function disorder per se",
        ],
        "seed": SEED_BASE + 2,
        "n_patients": 40,
        "age_range": (0, 40),
        "female_pct": 53,
    },
    # ── MYH9 — MYH9-Related Disease (AD) ──
    {
        "gene": "MYH9",
        "protein": "Non-muscle myosin heavy chain IIA",
        "alias": (
            "MYH9; OMIM gene 160775; MYH9-related disease (MYH9-RD) #155100; autosomal dominant; 22q11.2; "
            "1960 aa; ~227 kDa; non-muscle myosin IIA heavy chain; forms hexamer: 2x heavy + 2x essential light (MYL6) + 2x regulatory light (MYL12); "
            "May-Hegglin anomaly / Fechtner / Epstein / Sebastian syndromes — ALL caused by MYH9 variants (different clinical severity); "
            "GIANT PLATELETS + DÖHLE-LIKE INCLUSIONS IN NEUTROPHILS (MYH9 aggregates) — PATHOGNOMONIC; "
            "sensorineural hearing loss 35-60%; nephritis (Alport-like) 25-70%; cataracts 10-20%"
        ),
        "aa": "1960 aa",
        "kDa": "~227 kDa",
        "locus": "22q11.2",
        "omim_gene": 160775,
        "omim_disease": 155100,
        "inheritance": (
            "Autosomal dominant; de novo variants in ~50% of cases; "
            "50% risk to each child of affected parent; "
            "missense variants most common (usually in motor domain or coiled-coil tail); "
            "variant position and type correlate with severity: motor domain variants → more severe systemic features; "
            "penetrance ~100% for macrothrombocytopenia; variable expressivity for extraplatelet features"
        ),
        "gene_class": (
            "MYH9 encodes non-muscle myosin IIA heavy chain — a cytoskeletal motor protein expressed in platelets, "
            "megakaryocytes, kidney podocytes, cochlear hair cells, and lens epithelial cells. "
            "FUNCTION IN PLATELETS: Myosin IIA drives demarcation membrane system (DMS) organization in "
            "megakaryocytes during thrombopoiesis; regulates platelet size by controlling DMS network; "
            "also drives platelet shape change, clot retraction, and cytoskeletal tension. "
            "PATHOPHYSIOLOGY: MYH9 missense variants → myosin IIA dysfunctional or mis-assembled → "
            "abnormal DMS in megakaryocytes → oversized platelet release (giant platelets); "
            "insoluble MYH9 mutant protein aggregates form INCLUSION BODIES (Döhle-like) in neutrophils "
            "and other leukocytes → leukocyte inclusions on blood film. "
            "EXTRA-PLATELET FEATURES: Same myosin IIA is critical in kidney podocytes "
            "(glomerular filtration barrier) → progressive nephritis; "
            "cochlear stereocilia maintenance → progressive SNHL; lens integrity → cataracts. "
            "SEVERITY SPECTRUM: May-Hegglin (mildest — thrombocytopenia + inclusions only) → "
            "Fechtner/Epstein (SNHL + nephritis) → Sebastian (all features including cataracts)."
        ),
        "phenotype": (
            "Thrombocytopenia: moderate (30-150k); giant platelets (MPV >12 fL, often >20 fL); "
            "Neutrophil inclusions: Döhle-like pale blue cytoplasmic inclusions (MYH9 aggregates) — KEY CLUE on blood film; "
            "Bleeding: mild-moderate (much milder than platelet count suggests — large platelets are haemostatically active); "
            "Sensorineural hearing loss (SNHL): progressive; onset childhood-early adult; audiogram mandatory; "
            "Nephritis: proteinuria + haematuria + progressive CKD; Alport-like histology; ACE inhibitor mandatory; "
            "Cataracts: lens opacity, variable severity; "
            "Elevated creatine kinase in some variants; "
            "Platelet function: relatively preserved (aggregation near-normal)"
        ),
        "key_hallmarks": [
            "Giant platelets + Döhle-like inclusions in NEUTROPHILS — PATHOGNOMONIC combination for MYH9-RD",
            "Flow cytometry: MYH9 aggregates stain with anti-MYH9 antibody in leukocytes — diagnostic confirmation",
            "AVOID ASPIRIN + NSAIDs — any additional platelet function impairment worsens bleeding in macrothrombocytopenia",
            "ACE inhibitor MANDATORY when nephritis (proteinuria) develops — slows CKD progression",
            "Audiogram MANDATORY at diagnosis and every 2 years — progressive SNHL often precedes nephritis",
        ],
        "treatment_alerts": [
            "AVOID ASPIRIN, NSAIDs, clopidogrel — platelet function already compromised; any inhibitor risks bleeding",
            "ACE inhibitor (ramipril/enalapril): MANDATORY on development of proteinuria — renoprotective",
            "Audiological surveillance: annual audiogram; cochlear implant effective when SNHL severe",
            "Ophthalmology: annual slit-lamp from diagnosis — cataracts may develop at any age",
            "Platelet transfusion for major bleeding or surgery — MYH9 platelets from donor are functional; AVOID spurious transfusion for count alone",
            "Eltrombopag/romiplostim: may increase platelet count pre-operatively; off-label; specialist decision",
        ],
        "ddx": [
            "Bernard-Soulier syndrome (GP1BA): giant platelets, thrombocytopenia, but NO neutrophil inclusions; ristocetin absent; AR inheritance",
            "Gray platelet syndrome (NBEAL2): large pale platelets but alpha-granule absent; no neutrophil inclusions; AR",
            "Di George / 22q11.2 deletion syndrome: MYH9 on 22q11.2 but deletion usually 3Mb → MYH9 haploinsufficiency uncommon; cardiac/palatal features",
            "Immune thrombocytopenia (ITP): no giant platelets; no neutrophil inclusions; autoantibodies; responds to steroids/IVIg",
            "Alport syndrome (COL4A3/4/5): nephritis + SNHL without thrombocytopenia; collagen IV basis; X-linked or AR",
        ],
        "seed": SEED_BASE + 3,
        "n_patients": 40,
        "age_range": (5, 55),
        "female_pct": 50,
    },
    # ── NBEAL2 — Gray Platelet Syndrome (AR) ──
    {
        "gene": "NBEAL2",
        "protein": "Neurobeachin-like protein 2",
        "alias": (
            "NBEAL2; OMIM gene 614169; Gray platelet syndrome (GPS) #139090; autosomal recessive; 3p21.31; "
            "2806 aa; ~319 kDa; BEACH (beige and Chediak-Higashi) domain-containing protein; "
            "essential for alpha-granule biogenesis in megakaryocytes and platelets; "
            "GPS: ALPHA-GRANULES ABSENT → pale 'gray' platelets on Wright-Giemsa stain PATHOGNOMONIC; "
            "myelofibrosis in 3rd-4th decade; progressive splenomegaly; mild-moderate thrombocytopenia; "
            "GPS also caused by GFI1B (AD) and FLI1 (AD) — panel all three if GPS phenotype"
        ),
        "aa": "2806 aa",
        "kDa": "~319 kDa",
        "locus": "3p21.31",
        "omim_gene": 614169,
        "omim_disease": 139090,
        "inheritance": (
            "Autosomal recessive; biallelic LOF; 25% sib risk; "
            "consanguinity increases risk; "
            "heterozygous carriers: asymptomatic, normal platelet alpha-granules; "
            "GFI1B (AD) and FLI1 (AD) also cause GPS phenotype — AD GPS is usually milder; "
            "NBEAL2 is the most common genetic cause of classic GPS"
        ),
        "gene_class": (
            "NBEAL2 encodes a large BEACH-domain protein required for the biogenesis of platelet alpha-granules. "
            "FUNCTION: Alpha-granules are the most numerous granule type in platelets (~80/platelet), "
            "containing fibrinogen, VWF, P-selectin (CD62P), factor V, fibronectin, beta-thromboglobulin, "
            "platelet factor 4 (PF4), thrombospondin, and PDGF — all released on platelet activation. "
            "NBEAL2 forms a complex with SEC22B and WASp on late endosomes to sort alpha-granule cargo; "
            "without NBEAL2, cargo (fibrinogen, VWF, PF4) secreted constitutively from megakaryocytes "
            "into marrow stromal tissue → MYELOFIBROSIS (alpha-granule proteins — PDGF, TGF-beta — drive fibrosis). "
            "PATHOPHYSIOLOGY: Biallelic LOF → absent alpha-granules in platelets → "
            "'gray' platelets (no granule contents → no intracellular staining) → "
            "impaired secondary aggregation (released granule contents amplify platelet activation); "
            "myelofibrosis: marrow stromal stimulation by constitutively secreted PDGF/TGF-beta; "
            "splenomegaly: extramedullary haematopoiesis from marrow failure."
        ),
        "phenotype": (
            "Mild-moderate thrombocytopenia (40-120k); large pale 'gray' platelets on Wright-Giemsa — PATHOGNOMONIC; "
            "Mild-moderate bleeding tendency (epistaxis, bruising, menorrhagia, prolonged post-op); "
            "Aggregation: secondary wave absent (reduced ADP, collagen aggregation); "
            "Myelofibrosis: progressive from 3rd decade — fatigue, cytopenias, splenomegaly; "
            "Splenomegaly: nearly universal in adults; "
            "Bone marrow: grey megakaryocytes, progressive fibrosis (reticulin then collagen); "
            "Alpha-granule markers: PF4, beta-TG, P-selectin (CD62P): absent/markedly reduced on flow cytometry"
        ),
        "key_hallmarks": [
            "Gray (pale, agranular) platelets on Wright-Giemsa stain — PATHOGNOMONIC (absent alpha-granules)",
            "P-selectin (CD62P) absent on flow cytometry (activation-induced surface CD62P = alpha-granule membrane marker)",
            "Myelofibrosis in 3rd-4th decade — MANDATORY bone marrow monitoring; NBEAL2 constitutive alpha-granule cargo → stromal fibrosis",
            "VWF multimer analysis NORMAL — distinguishes from VWD; platelet VWF storage defect but plasma VWF intact",
            "Alpha-granule absent by electron microscopy — gold standard for GPS diagnosis",
        ],
        "treatment_alerts": [
            "Myelofibrosis monitoring: bone marrow biopsy at diagnosis, then every 3-5 years from age 25 — ruxolitinib if MF-related symptoms",
            "Splenomegaly: avoid splenectomy if possible — worsens extramedullary haematopoiesis; haematology decision",
            "Antifibrinolytics (tranexamic acid): adjunct for mucosal bleeding",
            "Platelet transfusion: for significant bleeding/surgery — effective (donor platelets have intact alpha-granules)",
            "Desmopressin (DDAVP): may partially help (releases endothelial VWF); limited data in GPS",
            "Recombinant FVIIa: for platelet-refractory bleeding scenarios",
        ],
        "ddx": [
            "GFI1B-related GPS (AD): similar phenotype but AD; red cell macrocytosis and absent delta-granules in GFI1B",
            "FLI1-related GPS (AD): large pale platelets; ETS transcription factor; AD; less myelofibrosis than NBEAL2",
            "Myelofibrosis (primary): acquired; JAK2/CALR/MPL mutations; no alpha-granule defect on electron microscopy",
            "MYH9-RD: giant platelets + neutrophil inclusions; alpha-granules PRESENT; GPIb normal",
            "Platelet-poor preparation artefact: technical pale appearance — re-examine fresh specimen; GPS granule defect persists on repeat",
        ],
        "seed": SEED_BASE + 4,
        "n_patients": 40,
        "age_range": (5, 60),
        "female_pct": 52,
    },
    # ── ANKRD26 — Thrombocytopenia-2 / THC2 (AD) ──
    {
        "gene": "ANKRD26",
        "protein": "Ankyrin repeat domain-containing protein 26",
        "alias": (
            "ANKRD26; OMIM gene 610855; Thrombocytopenia-2 (THC2) #188000; autosomal dominant; 10p12.1; "
            "1982 aa; ~219 kDa; ankyrin repeat + spectrin-binding domain-containing protein; "
            "THC2 caused by point mutations in 5'UTR (promoter region) of ANKRD26 — NOT protein-coding variants; "
            "TRANSCRIPTION FACTOR BINDING SITE MUTATIONS: disrupt RUNX1 + FLI1 repression of ANKRD26 in mature megakaryocytes → "
            "persistent ANKRD26 expression → impaired proplatelet formation → thrombocytopenia; "
            "myeloid malignancy (AML/MDS) risk 5-8% lifetime; CBC annually mandatory"
        ),
        "aa": "1982 aa",
        "kDa": "~219 kDa",
        "locus": "10p12.1",
        "omim_gene": 610855,
        "omim_disease": 188000,
        "inheritance": (
            "Autosomal dominant; 5'UTR point mutations (promoter); NOT standard exonic sequencing-detected; "
            "targeted ANKRD26 5'UTR sequencing or WGS MANDATORY (standard WES may miss 5'UTR variants); "
            "50% risk to each child; de novo variants occur; "
            "penetrance ~100% for thrombocytopenia; AML/MDS risk appears present in ALL variant carriers"
        ),
        "gene_class": (
            "ANKRD26 encodes a large ankyrin repeat protein of uncertain full function expressed in "
            "megakaryocytes and haematopoietic progenitors. "
            "REGULATION: ANKRD26 is normally silenced in mature megakaryocytes — "
            "RUNX1 and FLI1 (ETS transcription factor) bind the 5'UTR and repress ANKRD26 transcription. "
            "PATHOPHYSIOLOGY of THC2: Point mutations in the ANKRD26 5'UTR destroy RUNX1/FLI1 binding sites → "
            "ANKRD26 remains aberrantly expressed in mature megakaryocytes → "
            "sustained ANKRD26 signalling activates MAP kinase pathway → "
            "impaired proplatelet formation and platelet release → thrombocytopenia. "
            "MALIGNANCY RISK: ANKRD26 interacts with thrombopoietin/MPL signalling; "
            "persistent aberrant signalling → clonal myeloid evolution → AML/MDS. "
            "DIAGNOSTIC PITFALL: Standard WES (whole exome sequencing) does NOT reliably detect 5'UTR variants → "
            "ANKRD26-specific targeted 5'UTR sequencing or whole genome sequencing REQUIRED. "
            "VARIANT LOCATION: Almost all pathogenic variants fall in a 19 bp region of the 5'UTR (c.-127 to c.-109 region)."
        ),
        "phenotype": (
            "Thrombocytopenia: mild-moderate (20-150k, mean ~60k); platelet size NORMAL or slightly increased; "
            "Bleeding: variable, typically mild (epistaxis, menorrhagia, easy bruising); "
            "Platelet function: NORMAL (aggregation, PFA-100 often normal for count); "
            "AML/MDS risk: 5-8% lifetime; may be higher with specific 5'UTR variant; "
            "Bone marrow: hypercellular megakaryocytes, variable dysmegakaryopoiesis; "
            "Blood film: no morphological clue (unlike MYH9-RD or GPS) — diagnosis requires molecular testing; "
            "WBC + red cell: NORMAL (no macrocytosis, no inclusions)"
        ),
        "key_hallmarks": [
            "5'UTR promoter mutations — NOT detectable by standard WES; require targeted ANKRD26 5'UTR sequencing or WGS",
            "AVOID thrombopoietin receptor agonists (eltrombopag, romiplostim) — mechanism may increase malignancy risk",
            "Annual CBC + clinical review MANDATORY — AML/MDS risk 5-8% lifetime; bone marrow if cytopenias worsen",
            "Platelet function NORMAL despite count reduction — aggregation studies help distinguish from function disorders",
            "RUNX1/FLI1 repressor binding site disruption: explains co-occurrence of RUNX1 and ANKRD26 in thrombocytopenia spectrum",
        ],
        "treatment_alerts": [
            "AVOID thrombopoietin receptor agonists (eltrombopag, romiplostim): theoretical AML/MDS promotion risk — specialist decision only",
            "Annual CBC: AML/MDS surveillance — bone marrow biopsy if count worsens or monocytosis/dysplasia",
            "Platelet transfusion: for significant bleeding or surgery (platelet function normal so donor platelets fully effective)",
            "Antifibrinolytics: tranexamic acid adjunct for mucosal bleeding episodes",
            "Genetic counselling: 50% risk to children; test all first-degree relatives; asymptomatic carriers need AML surveillance",
            "Haematology referral: mandatory — specialist-managed surveillance; not for GP alone",
        ],
        "ddx": [
            "RUNX1 FPD/AML (RUNX1): clinically similar; protein-coding variants; ALSO AML risk; standard WES detects RUNX1 coding variants",
            "ETV6-related thrombocytopenia (ETV6 MBD): AD; hereditary thrombocytopenia + ALL risk; ETS factor",
            "MYH9-RD: giant platelets + neutrophil inclusions; normal WES WILL detect MYH9 coding variants",
            "Gray platelet syndrome: large gray platelets; alpha-granule absent; ANKRD26 has NORMAL platelet granules",
            "Immune thrombocytopenia (ITP): no family history; autoantibodies; response to steroids; ANKRD26 is hereditary",
        ],
        "seed": SEED_BASE + 5,
        "n_patients": 40,
        "age_range": (10, 65),
        "female_pct": 51,
    },
    # ── GFI1B — GFI1B-Related Thrombocytopenia (AD) ──
    {
        "gene": "GFI1B",
        "protein": "Growth factor independent 1B transcription repressor",
        "alias": (
            "GFI1B; OMIM gene 604383; GFI1B-related thrombocytopenia #187900; autosomal dominant; 9q34.13; "
            "330 aa; ~37 kDa; zinc finger transcriptional repressor (GFI family); "
            "essential for megakaryocyte and erythroid differentiation; "
            "RED CELL MACROCYTOSIS PATHOGNOMONIC — GFI1B regulates erythroid maturation; "
            "ABSENT DELTA-GRANULES (dense bodies) → secondary aggregation defect; "
            "CD34 aberrantly expressed on mature GFI1B-deficient platelets (immature marker) → "
            "diagnostic flow cytometry finding; bleeding variable (mild-moderate)"
        ),
        "aa": "330 aa",
        "kDa": "~37 kDa",
        "locus": "9q34.13",
        "omim_gene": 604383,
        "omim_disease": 187900,
        "inheritance": (
            "Autosomal dominant; haploinsufficiency; de novo variants and familial; "
            "50% risk to each child; "
            "heterozygous LOF or dominant negative variants; "
            "C-terminal zinc finger domain variants most common (ZF5 and ZF6 DNA-binding domain); "
            "penetrance ~100% for thrombocytopenia and macrocytosis; "
            "some variants cause GPS-like phenotype (alpha-granule deficiency)"
        ),
        "gene_class": (
            "GFI1B encodes Growth Factor Independent 1B, a transcriptional repressor with "
            "SNAIL/GFI1 domain and six C-terminal zinc fingers that bind DNA. "
            "FUNCTION: GFI1B is a master regulator of megakaryocyte and erythroid lineage differentiation. "
            "It recruits LSD1 (KDM1A) histone demethylase and CoREST complex to repress gene expression; "
            "drives differentiation of early progenitors into mature megakaryocytes and proerythroblasts. "
            "PATHOPHYSIOLOGY: "
            "Haploinsufficiency → reduced GFI1B → impaired megakaryocyte maturation → "
            "immature megakaryocytes → platelets with residual CD34 (stem cell antigen) + absent dense granules; "
            "DELTA-GRANULES: Dense granules (delta-granules) contain ADP, ATP, serotonin, calcium — "
            "absent delta-granules → impaired secondary aggregation wave (ADP release); "
            "RED CELL MACROCYTOSIS: GFI1B also required for erythroid maturation → "
            "erythroblast expansion dysregulated → macrocytic red cells (MCV >100 fL); "
            "This combination — thrombocytopenia + macrocytosis + absent delta-granules — is "
            "characteristic of GFI1B-RD."
        ),
        "phenotype": (
            "Thrombocytopenia: moderate (30-100k); platelet size mildly increased; "
            "Red cell macrocytosis: MCV 100-115 fL — PATHOGNOMONIC clue without anaemia; "
            "Bleeding: mild-moderate (epistaxis, bruising, menorrhagia); "
            "Aggregation: secondary wave absent — ADP aggregation shows absent/reduced secondary wave; "
            "Dense granule (delta-granule) defect: mepacrine uptake and release absent on flow cytometry; "
            "Flow cytometry: CD34+ on mature platelets — diagnostic (should be CD34-negative in normal platelets); "
            "GPS-like variant: some GFI1B variants cause alpha-granule absent phenotype (grey platelets)"
        ),
        "key_hallmarks": [
            "Red cell macrocytosis (MCV >100 fL) WITHOUT anaemia — PATHOGNOMONIC clue for GFI1B-RD; absent in other platelet disorders",
            "CD34 expression on mature platelets — diagnostic flow cytometry (immature platelet marker retained due to GFI1B haploinsufficiency)",
            "Absent delta-granules (dense bodies): mepacrine uptake ABSENT; impaired secondary ADP aggregation wave",
            "Electron microscopy: absent or markedly reduced dense granules — confirms delta-storage pool deficiency",
            "GFI1B variants at C-terminal zinc fingers (ZF5/ZF6 DNA-binding domain) most common — sequence entire gene",
        ],
        "treatment_alerts": [
            "Platelet transfusion for significant bleeding — GFI1B platelets have absent delta-granules but CD34+ platelets still form a mechanical plug",
            "Antifibrinolytics (tranexamic acid): recommended adjunct for mucosal/surgical bleeding",
            "DESMOPRESSIN (DDAVP): may partially increase VWF — adjunct for minor procedures in some patients",
            "Menorrhagia: LNG-IUS (Mirena) or oral contraceptive — mandatory management",
            "No myeloid malignancy risk established (unlike RUNX1, ANKRD26) — surveillance not currently mandated",
            "Haematology referral: MANDATORY — specialist-managed; electron microscopy and mepacrine flow required",
        ],
        "ddx": [
            "Gray platelet syndrome (NBEAL2): alpha-granule absent (not delta-granule); no macrocytosis; AR; CD34 NEGATIVE on platelets",
            "Wiskott-Aldrich syndrome (WAS): XLR; microplatelets; immunodeficiency + eczema; WASp absent",
            "RUNX1 FPD/AML: AD thrombocytopenia; AML risk 35-44%; platelet aggregation reduced; NO macrocytosis, NO CD34 on platelets",
            "Acquired vitamin B12/folate deficiency: macrocytosis + thrombocytopenia; but reversible; no delta-granule defect",
            "Diamond-Blackfan anaemia (RPS19 etc.): macrocytosis + anaemia; AR; elevated erythrocyte ADA; no platelet granule defect",
        ],
        "seed": SEED_BASE + 6,
        "n_patients": 40,
        "age_range": (5, 60),
        "female_pct": 52,
    },
    # ── RUNX1 — Familial Platelet Disorder with AML (AD) ──
    {
        "gene": "RUNX1",
        "protein": "Runt-related transcription factor 1 (AML1/CBFA2)",
        "alias": (
            "RUNX1; OMIM gene 151385; Familial platelet disorder with predisposition to AML (FPD/AML) #601399; autosomal dominant; 21q22.12; "
            "453 aa; ~49 kDa; runt homology domain (RHD) + transactivation domain; "
            "core binding factor alpha subunit — heterodimerises with CBFbeta (CBFB); "
            "AML/MDS LIFETIME RISK 35-44% — MANDATORY haematology surveillance; "
            "GENETIC COUNSELLING + CBC ANNUALLY ALL FIRST-DEGREE RELATIVES; "
            "NO aspirin; NO NSAIDs; stem cell transplant CURATIVE if AML/MDS develops"
        ),
        "aa": "453 aa",
        "kDa": "~49 kDa",
        "locus": "21q22.12",
        "omim_gene": 151385,
        "omim_disease": 601399,
        "inheritance": (
            "Autosomal dominant; haploinsufficiency; missense, nonsense, frameshift, intragenic deletion variants; "
            "50% risk to each child; de novo variants in ~50%; "
            "penetrance ~100% for thrombocytopenia; AML/MDS penetrance ~35-44% lifetime; "
            "somatic RUNX1 mutations are the MOST COMMON acquired abnormality in ALL and AML — "
            "germline RUNX1 variants create haploinsufficiency background for subsequent somatic 'second hit' → leukaemia"
        ),
        "gene_class": (
            "RUNX1 (AML1/CBFA2) encodes the alpha-subunit of core binding factor (CBF), the most critical "
            "transcription factor complex in haematopoiesis. "
            "FUNCTION: RUNX1 binds DNA via its runt homology domain (RHD), forms heterodimer with CBFbeta (CBFB), "
            "and activates transcription of genes essential for HSC maintenance, myeloid differentiation, "
            "megakaryopoiesis, and lymphopoiesis. "
            "KEY TARGETS: PF4, GPVI (ITGA2), GPIX (GP9), thrombopoietin receptor (MPL), and "
            "also represses ANKRD26 (see THC2 — RUNX1 binding site mutation causes THC2). "
            "PATHOPHYSIOLOGY: Germline RUNX1 haploinsufficiency → "
            "(1) Thrombocytopenia: inadequate megakaryocyte maturation; platelet release impaired; "
            "(2) Platelet dysfunction: reduced GPVI expression → collagen response impaired; "
            "aggregation studies show reduced ADP, collagen responses; "
            "(3) AML/MDS predisposition: HSC haploinsufficiency for RUNX1 → genomic instability → "
            "somatic RUNX1 second allele loss-of-function or other oncogenic mutations → frank leukaemia. "
            "SOMATIC vs GERMLINE: Somatic RUNX1 mutations (t(8;21), point mutations) in AML without family history "
            "are NOT the same as germline FPD/AML — germline testing required when family history present."
        ),
        "phenotype": (
            "Thrombocytopenia: mild-moderate (80-150k); platelet size normal-mildly increased; "
            "Bleeding: variable (mild epistaxis, menorrhagia, easy bruising, post-op bleeding); "
            "Platelet function: reduced collagen-induced aggregation (GPVI reduced); "
            "dense granule deficiency in ~50%; reduced ADP aggregation second wave; "
            "AML/MDS: median age of development varies (range 10-65 years); T-ALL also reported; "
            "Blood film: no specific morphological clue; "
            "Bone marrow: may show dysplastic megakaryocytes; evolution to MDS morphology precedes AML"
        ),
        "key_hallmarks": [
            "AML/MDS lifetime risk 35-44% — MANDATORY annual CBC + bone marrow biopsy every 2-3 years in all carriers",
            "NO aspirin / NO NSAIDs — platelet function already reduced; any further impairment risks serious bleeding",
            "Genetic counselling mandatory: HSCT donor selection MUST EXCLUDE RUNX1-carrier family members",
            "RUNX1 germline variants detected on WES — but family history must prompt GERMLINE confirmation (not somatic test)",
            "Collagen aggregation reduced (GPVI target of RUNX1) — aggregometry clue alongside family history",
        ],
        "treatment_alerts": [
            "NO ASPIRIN / NO NSAIDs — absolute contraindication; further platelet function impairment risks life-threatening bleeding",
            "Annual CBC: mandatory; bone marrow biopsy every 2-3 years or immediately if counts change",
            "HSCT donor selection: EXCLUDE all first-degree relatives — they have 50% carrier risk; use unrelated donor",
            "AML/MDS treatment: standard induction chemotherapy ± HSCT from unrelated donor; outcomes comparable to de novo AML",
            "Genetic counselling: ALL first-degree relatives must be offered cascade testing; carriers need identical surveillance",
            "Platelet transfusion for surgery: effective; avoid aspirin perioperatively; plan with haematology pre-operatively",
        ],
        "ddx": [
            "ANKRD26 THC2: AD; 5'UTR mutations; AML risk 5-8%; RUNX1 represses ANKRD26 — overlapping pathway; normal platelet function",
            "ETV6-related thrombocytopenia: AD; AML + ALL risk; ETS factor; RHD different from RUNX1",
            "Somatic RUNX1 AML (t(8;21)): acquired; no family history; CBC normal pre-diagnosis; NOT germline FPD/AML",
            "Wiskott-Aldrich syndrome (WAS): XLR; microplatelets; immunodeficiency; WASp absent",
            "ITP: acquired autoantibodies; no family history; response to steroids; RUNX1 FPD/AML is hereditary with stable mild count",
        ],
        "seed": SEED_BASE + 7,
        "n_patients": 40,
        "age_range": (8, 70),
        "female_pct": 51,
    },
]


# ─── Cohort generation ───────────────────────────────────────────────────────

def _gen_cohort(gene_data: dict) -> list:
    rng = random.Random(gene_data["seed"])
    gene = gene_data["gene"]
    n = gene_data["n_patients"]
    ages = gene_data["age_range"]
    fp = gene_data["female_pct"] / 100
    cohort = []
    for i in range(n):
        pid = f"{gene}-{i+1:03d}"
        sex = "F" if rng.random() < fp else "M"
        age = rng.randint(ages[0], ages[1])
        platelet_count = 0
        platelet_size = "normal"
        aggregation_defect = "severe"
        bleeding_severity = "moderate"
        special = {}

        if gene == "GP1BA":
            platelet_count = rng.randint(20, 110)
            platelet_size = "giant"
            aggregation_defect = "ristocetin_absent"
            bleeding_severity = rng.choice(["mild", "moderate", "severe"])
            special = {
                "ristocetin_agglutination": False,
                "giant_platelets": True,
                "desmopressin_given": rng.random() < 0.3,
                "desmopressin_effective": False,
                "platelet_transfusion_given": rng.random() < 0.6,
                "alloimmunized": rng.random() < 0.25,
                "epistaxis": rng.random() < 0.8,
                "menorrhagia": sex == "F" and rng.random() < 0.75,
            }
        elif gene == "ITGA2B":
            platelet_count = rng.randint(140, 400)
            platelet_size = "normal"
            aggregation_defect = "all_agonists"
            bleeding_severity = rng.choice(["moderate", "severe"])
            special = {
                "platelet_count_normal": True,
                "clot_retraction": False,
                "ristocetin_agglutination": True,
                "fvii_given": rng.random() < 0.55,
                "platelet_transfusion_given": rng.random() < 0.5,
                "alloimmunized": rng.random() < 0.3,
                "cd41_absent": True,
                "cd61_present": True,
            }
        elif gene == "ITGB3":
            platelet_count = rng.randint(140, 380)
            platelet_size = "normal"
            aggregation_defect = "all_agonists"
            bleeding_severity = rng.choice(["moderate", "severe"])
            hpa_1a_risk = rng.random() < 0.04 if sex == "F" else False
            special = {
                "platelet_count_normal": True,
                "clot_retraction": False,
                "ristocetin_agglutination": True,
                "cd41_present": True,
                "cd61_absent": True,
                "hpa_1a_negative_mother": hpa_1a_risk,
                "nait_risk": hpa_1a_risk,
                "fvii_given": rng.random() < 0.55,
                "alloimmunized": rng.random() < 0.3,
            }
        elif gene == "MYH9":
            platelet_count = rng.randint(30, 150)
            platelet_size = "giant"
            aggregation_defect = "mild"
            bleeding_severity = rng.choice(["mild", "mild", "moderate"])
            snhl = rng.random() < 0.45
            nephritis = rng.random() < 0.40
            cataracts = rng.random() < 0.15
            special = {
                "neutrophil_inclusions": True,
                "doehle_bodies": True,
                "sensorineural_hearing_loss": snhl,
                "nephritis": nephritis,
                "cataracts": cataracts,
                "ace_inhibitor_prescribed": nephritis,
                "aspirin_given": rng.random() < 0.12,  # should be avoided
                "audiogram_done": rng.random() < 0.7,
                "creatinine_elevated": nephritis and rng.random() < 0.5,
            }
        elif gene == "NBEAL2":
            platelet_count = rng.randint(40, 120)
            platelet_size = "large_gray"
            aggregation_defect = "partial"
            bleeding_severity = rng.choice(["mild", "moderate"])
            myelofibrosis = age > 30 and rng.random() < 0.40
            special = {
                "gray_platelets": True,
                "alpha_granules_absent": True,
                "cd62p_absent": True,
                "splenomegaly": rng.random() < 0.75,
                "myelofibrosis": myelofibrosis,
                "bm_biopsy_done": rng.random() < 0.6,
                "vwf_multimers_normal": True,
                "pf4_absent": True,
            }
        elif gene == "ANKRD26":
            platelet_count = rng.randint(20, 150)
            platelet_size = "normal_slightly_large"
            aggregation_defect = "none"
            bleeding_severity = rng.choice(["mild", "mild", "moderate"])
            aml_mds = rng.random() < 0.065
            special = {
                "5utr_variant": True,
                "wes_detected": False,
                "targeted_sequencing_done": rng.random() < 0.55,
                "platelet_function_normal": True,
                "aml_mds_developed": aml_mds,
                "annual_cbc_done": rng.random() < 0.7,
                "tpo_agonist_given": rng.random() < 0.1,  # should be avoided
                "haematology_referral": rng.random() < 0.8,
            }
        elif gene == "GFI1B":
            platelet_count = rng.randint(30, 100)
            platelet_size = "slightly_large"
            aggregation_defect = "delta_granule"
            bleeding_severity = rng.choice(["mild", "moderate"])
            special = {
                "red_cell_macrocytosis": True,
                "mcv": rng.randint(101, 118),
                "delta_granules_absent": True,
                "cd34_on_platelets": True,
                "mepacrine_uptake_absent": True,
                "alpha_granules_present": rng.random() > 0.15,
                "ddavp_tried": rng.random() < 0.35,
                "ddavp_partial_response": rng.random() < 0.4,
            }
        elif gene == "RUNX1":
            platelet_count = rng.randint(80, 150)
            platelet_size = "normal"
            aggregation_defect = "collagen_reduced"
            bleeding_severity = rng.choice(["mild", "mild", "moderate"])
            aml_mds = rng.random() < 0.39
            special = {
                "collagen_aggregation_reduced": True,
                "gpvi_reduced": True,
                "dense_granule_defect": rng.random() < 0.5,
                "aml_mds_developed": aml_mds,
                "annual_cbc_done": rng.random() < 0.75,
                "bm_biopsy_done": rng.random() < 0.65,
                "aspirin_given": rng.random() < 0.08,  # should be avoided
                "hsct_done": aml_mds and rng.random() < 0.6,
                "family_tested": rng.random() < 0.65,
            }

        cohort.append({
            "patient_id": pid,
            "age_at_dx": age,
            "sex": sex,
            "gene": gene,
            "platelet_count_k": platelet_count,
            "platelet_size": platelet_size,
            "aggregation_defect": aggregation_defect,
            "bleeding_severity": bleeding_severity,
            "dx_delay_years": round(rng.uniform(0.5, 8.0), 1),
            **special,
        })
    return cohort


_ALL_COHORTS = {gd["gene"]: _gen_cohort(gd) for gd in PLATELET_GENES}


def get_overview():
    n = sum(len(v) for v in _ALL_COHORTS.values())
    female_n = sum(1 for v in _ALL_COHORTS.values() for p in v if p["sex"] == "F")

    # Gene-level statistics
    myh9_snhl = sum(1 for p in _ALL_COHORTS["MYH9"] if p.get("sensorineural_hearing_loss"))
    myh9_nephritis = sum(1 for p in _ALL_COHORTS["MYH9"] if p.get("nephritis"))
    nbeal2_mf = sum(1 for p in _ALL_COHORTS["NBEAL2"] if p.get("myelofibrosis"))
    runx1_aml = sum(1 for p in _ALL_COHORTS["RUNX1"] if p.get("aml_mds_developed"))
    ankrd26_aml = sum(1 for p in _ALL_COHORTS["ANKRD26"] if p.get("aml_mds_developed"))
    bss_alloimmunized = sum(1 for p in _ALL_COHORTS["GP1BA"] if p.get("alloimmunized"))
    gt_alloimmunized = (
        sum(1 for p in _ALL_COHORTS["ITGA2B"] if p.get("alloimmunized")) +
        sum(1 for p in _ALL_COHORTS["ITGB3"] if p.get("alloimmunized"))
    )
    gfi1b_macrocytosis = sum(1 for p in _ALL_COHORTS["GFI1B"] if p.get("red_cell_macrocytosis"))
    mean_dx_delay = round(
        sum(p["dx_delay_years"] for v in _ALL_COHORTS.values() for p in v) / n, 1
    )

    return {
        "atlas_name": "Platelet-Disorders-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Platelet Disorders Atlas",
        "n_patients": n,
        "gene_count": len(PLATELET_GENES),
        "seeds": "1334–1341",
        "gene_summary": [
            {
                "gene": gd["gene"],
                "protein": gd["protein"],
                "aa": gd["aa"],
                "locus": gd["locus"],
                "inheritance": gd["inheritance"].split(";")[0].strip(),
                "phenotype_short": {
                    "GP1BA": "BSS — Giant platelets, no VWF adhesion, ristocetin ABSENT, DDAVP ineffective",
                    "ITGA2B": "GT type A — Normal count, ALL aggregation absent, no clot retraction",
                    "ITGB3": "GT type B — Identical to GT-A; HPA-1a antigen → NAIT risk",
                    "MYH9": "MYH9-RD — Giant platelets + neutrophil Döhle inclusions; SNHL; nephritis",
                    "NBEAL2": "Gray platelet syndrome — Alpha-granule absent, myelofibrosis in 3rd-4th decade",
                    "ANKRD26": "THC2 — 5'UTR mutations (WES misses!); mild AML/MDS risk; platelet function normal",
                    "GFI1B": "GFI1B-RD — Red cell macrocytosis PATHOGNOMONIC; absent delta-granules; CD34 on platelets",
                    "RUNX1": "FPD/AML — 35-44% AML risk; NO aspirin; HSCT excludes family donors; annual marrow",
                }[gd["gene"]],
                "hallmark_short": {
                    "GP1BA": "Giant megathrombocytes on film; ristocetin agglutination ABSENT; DDAVP INEFFECTIVE",
                    "ITGA2B": "Normal platelet count/size; ALL aggregation absent; clot retraction ABSENT; CD41−",
                    "ITGB3": "Identical to GT-A; CD61− (GPIIIa absent); HPA-1a NAIT risk in mothers",
                    "MYH9": "Giant platelets + Döhle-like inclusions in NEUTROPHILS — combined = MYH9-RD ONLY",
                    "NBEAL2": "Gray pale platelets (alpha-granule absent); myelofibrosis; splenomegaly in adults",
                    "ANKRD26": "5'UTR variant (standard WES FAILS); normal platelet function; AML/MDS 5-8%; avoid TPO-agonists",
                    "GFI1B": "Macrocytosis (MCV >100) + CD34+ platelets; absent dense granules; mepacrine absent",
                    "RUNX1": "AML/MDS 35-44%; NO aspirin; family HSCT donors excluded; cascade test all first-degree",
                }[gd["gene"]],
            }
            for gd in PLATELET_GENES
        ],
        "aggregate_stats": {
            "myh9_snhl_pct": round(myh9_snhl / 40 * 100, 1),
            "myh9_nephritis_pct": round(myh9_nephritis / 40 * 100, 1),
            "nbeal2_myelofibrosis_pct": round(nbeal2_mf / 40 * 100, 1),
            "runx1_aml_pct": round(runx1_aml / 40 * 100, 1),
            "ankrd26_aml_pct": round(ankrd26_aml / 40 * 100, 1),
            "bss_alloimmunized_pct": round(bss_alloimmunized / 40 * 100, 1),
            "gt_alloimmunized_pct": round(gt_alloimmunized / 80 * 100, 1),
            "gfi1b_macrocytosis_pct": round(gfi1b_macrocytosis / 40 * 100, 1),
            "female_pct": round(female_n / n * 100, 1),
            "mean_diagnosis_delay_yrs": mean_dx_delay,
        },
        "critical_drug_rules": {
            "RUNX1_NO_ASPIRIN": "ABSOLUTELY NO ASPIRIN / NSAIDs in RUNX1 FPD/AML — further platelet function impairment + AML risk",
            "MYH9_NO_ASPIRIN": "AVOID ASPIRIN + NSAIDs in MYH9-RD — macrothrombocytopenia makes platelet inhibition hazardous",
            "GP1BA_NO_DESMOPRESSIN": "DESMOPRESSIN INEFFECTIVE in BSS — no GPIb receptor to capture released VWF multimers",
            "ANKRD26_AVOID_TPO": "AVOID thrombopoietin receptor agonists (eltrombopag, romiplostim) in ANKRD26 THC2 — AML/MDS promotion risk",
            "GT_RFVII_PREFERRED": "RECOMBINANT FVIIa (NovoSeven) preferred over platelet transfusion in GT — limits alloimmunization",
        },
        "kpis": [
            {"label": "Total Patients", "value": str(n)},
            {"label": "Genes Covered", "value": str(len(PLATELET_GENES))},
            {"label": "RUNX1 AML/MDS", "value": f"{runx1_aml}/40 ({round(runx1_aml/40*100,1)}%)"},
            {"label": "MYH9 SNHL", "value": f"{myh9_snhl}/40 ({round(myh9_snhl/40*100,1)}%)"},
            {"label": "NBEAL2 Myelofibrosis", "value": f"{nbeal2_mf}/40"},
            {"label": "GT Alloimmunized", "value": f"{gt_alloimmunized}/80"},
            {"label": "GFI1B Macrocytosis", "value": f"{gfi1b_macrocytosis}/40 (100%)"},
            {"label": "Mean Dx Delay", "value": f"{mean_dx_delay} yrs"},
        ],
        "drug_alerts": [
            "RUNX1 FPD/AML: NO ASPIRIN / NO NSAIDs — any platelet function inhibition in AML-predisposed patient is dangerous",
            "MYH9-RD: AVOID ASPIRIN + NSAIDs — giant macrothrombocytopenia already haemostatically fragile",
            "BSS (GP1BA): DESMOPRESSIN INEFFECTIVE — no GPIb receptor present; never administer expecting VWF response",
            "GT (ITGA2B/ITGB3): PLATELET TRANSFUSION → ALLOIMMUNIZATION risk; use rFVIIa (NovoSeven) first-line for surgery",
            "ANKRD26 THC2: AVOID TPO-receptor agonists (eltrombopag, romiplostim) — unquantified AML/MDS promotion",
        ],
        "critical_rules": [
            "RUNX1 cascade testing mandatory — 50% risk per first-degree relative; ALL family members need annual CBC + haematology",
            "GP1BA giant platelets mimic thrombocytopenia on automated counter — MANUAL BLOOD FILM review MANDATORY",
            "GT (ITGA2B/ITGB3): platelet count NORMAL; aggregation studies required to diagnose; do NOT dismiss 'normal count' platelet disorder",
            "MYH9-RD: nephritis + SNHL screen MANDATORY at diagnosis and every 2 years; ACE inhibitor when nephritis develops",
            "ANKRD26 5'UTR variants MISSED by standard WES — targeted 5'UTR sequencing or WGS required",
            "GFI1B: red cell macrocytosis (MCV >100 without B12/folate deficiency) in thrombocytopenic patient → GFI1B first",
            "NBEAL2 gray platelet syndrome: bone marrow biopsy from age 25 — myelofibrosis monitoring NON-NEGOTIABLE",
            "HSCT for RUNX1/ANKRD26 AML: EXCLUDE family donors — 50% chance carrier; use unrelated or cord blood",
        ],
    }


def get_breakdown():
    rows = []
    for gd in PLATELET_GENES:
        cohort = _ALL_COHORTS[gd["gene"]]
        n = len(cohort)
        female_n = sum(1 for p in cohort if p["sex"] == "F")
        mean_age = round(sum(p["age_at_dx"] for p in cohort) / n, 1)
        mean_delay = round(sum(p["dx_delay_years"] for p in cohort) / n, 1)
        special_features = {}
        gene = gd["gene"]
        if gene == "GP1BA":
            special_features["alloimmunized"] = sum(1 for p in cohort if p.get("alloimmunized"))
            special_features["ddavp_given_incorrectly"] = sum(1 for p in cohort if p.get("desmopressin_given"))
        elif gene in ["ITGA2B", "ITGB3"]:
            special_features["alloimmunized"] = sum(1 for p in cohort if p.get("alloimmunized"))
            special_features["fvii_used"] = sum(1 for p in cohort if p.get("fvii_given"))
        elif gene == "MYH9":
            special_features["snhl"] = sum(1 for p in cohort if p.get("sensorineural_hearing_loss"))
            special_features["nephritis"] = sum(1 for p in cohort if p.get("nephritis"))
            special_features["cataracts"] = sum(1 for p in cohort if p.get("cataracts"))
            special_features["aspirin_given"] = sum(1 for p in cohort if p.get("aspirin_given"))
        elif gene == "NBEAL2":
            special_features["myelofibrosis"] = sum(1 for p in cohort if p.get("myelofibrosis"))
            special_features["splenomegaly"] = sum(1 for p in cohort if p.get("splenomegaly"))
        elif gene == "ANKRD26":
            special_features["aml_mds"] = sum(1 for p in cohort if p.get("aml_mds_developed"))
            special_features["tpo_agonist_given"] = sum(1 for p in cohort if p.get("tpo_agonist_given"))
            special_features["targeted_seq_done"] = sum(1 for p in cohort if p.get("targeted_sequencing_done"))
        elif gene == "GFI1B":
            special_features["macrocytosis"] = sum(1 for p in cohort if p.get("red_cell_macrocytosis"))
            mean_mcv = round(sum(p.get("mcv", 108) for p in cohort) / n, 1)
            special_features["mean_mcv"] = mean_mcv
        elif gene == "RUNX1":
            special_features["aml_mds"] = sum(1 for p in cohort if p.get("aml_mds_developed"))
            special_features["aspirin_given"] = sum(1 for p in cohort if p.get("aspirin_given"))
            special_features["hsct_done"] = sum(1 for p in cohort if p.get("hsct_done"))
            special_features["family_tested"] = sum(1 for p in cohort if p.get("family_tested"))

        mean_plt = round(sum(p["platelet_count_k"] for p in cohort) / n, 0)
        rows.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias_summary": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "gene_class": gd["gene_class"],
            "phenotype": gd["phenotype"],
            "key_hallmarks": gd["key_hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "ddx": gd["ddx"],
            "cohort_stats": {
                "n": n,
                "seed": gd["seed"],
                "female_pct": round(female_n / n * 100, 1),
                "age_mean": mean_age,
                "mean_platelet_count_k": int(mean_plt),
                "mean_diagnosis_delay_yrs": mean_delay,
                "special_features": special_features,
            },
            "patients": [
                {
                    "patient_id": p["patient_id"],
                    "age_at_dx": p["age_at_dx"],
                    "sex": p["sex"],
                    "platelet_count_k": p["platelet_count_k"],
                    "platelet_size": p["platelet_size"],
                    "aggregation_defect": p["aggregation_defect"],
                    "bleeding_severity": p["bleeding_severity"],
                    "dx_delay_years": p["dx_delay_years"],
                    **{k: v for k, v in p.items() if k not in
                       ("patient_id", "age_at_dx", "sex", "gene", "platelet_count_k",
                        "platelet_size", "aggregation_defect", "bleeding_severity", "dx_delay_years")},
                }
                for p in cohort
            ],
        })
    return rows


def get_definitions():
    return {
        "atlas": "Platelet-Disorders-Atlas",
        "definitions": [
            {
                "term": "Bernard-Soulier Syndrome (BSS) — Giant Platelets and GPIb Deficiency",
                "short": "AR biallelic GP1BA/GP1BB/GP9 LOF → absent GPIb-IX-V complex → giant platelets, no VWF adhesion, DDAVP ineffective",
                "detail": (
                    "BSS is caused by absent or dysfunctional GPIb-IX-V complex — the primary platelet receptor "
                    "for von Willebrand factor (VWF) at high-shear vascular injury sites. "
                    "GIANT PLATELETS are the morphological hallmark: without GPIb, the demarcation membrane "
                    "system in megakaryocytes expands uncontrolled → platelet diameter 7-20 µm (lymphocyte-sized). "
                    "DIAGNOSTIC PITFALL: Automated haematology analysers using impedance counting "
                    "classify giant platelets as red cells or exclude them → "
                    "platelet count appears falsely lower than true count → MANUAL BLOOD FILM REVIEW MANDATORY. "
                    "RISTOCETIN TEST: Ristocetin agglutination ABSENT in BSS "
                    "(GPIb cannot bind VWF; ristocetin is the cofactor that makes VWF bind GPIb). "
                    "KEY DDx: VWD (ristocetin also reduced but platelet size NORMAL, VWF antigen low); "
                    "Platelet-type VWD (GOF GP1BA — hypersensitive to ristocetin, OPPOSITE of BSS). "
                    "TREATMENT: DDAVP releases VWF from endothelium but cannot capture platelets without GPIb → "
                    "DDAVP INEFFECTIVE in BSS. Platelet transfusion is effective but alloimmunization risk "
                    "with repeated use limits this strategy long-term."
                ),
                "clinical_rule": "BSS: giant platelets (film!) + absent ristocetin agglutination + DDAVP ineffective → GP1BA sequencing; never substitute DDAVP for transfusion",
            },
            {
                "term": "Glanzmann Thrombasthenia (GT) — Absent GPIIb/IIIa, Normal Count, Zero Aggregation",
                "short": "AR biallelic ITGA2B (GT-A) or ITGB3 (GT-B) → absent GPIIb/IIIa → NO aggregation to ANY agonist; platelet count NORMAL",
                "detail": (
                    "GT is the paradigm of a platelet FUNCTION disorder with NORMAL count. "
                    "GPIIb/IIIa (integrin alphaIIb-beta3) is the final common pathway of platelet aggregation — "
                    "it is the receptor for fibrinogen and VWF that bridges platelets together into a plug. "
                    "In GT, GPIIb/IIIa is absent or non-functional → "
                    "NO aggregation to ADP, collagen, thrombin, arachidonic acid, or any physiological agonist. "
                    "CLOT RETRACTION ABSENT: GPIIb/IIIa is also required for fibrin-cytoskeleton linkage → "
                    "the clot does not retract → this is the simplest bedside screening test for GT. "
                    "RISTOCETIN AGGLUTINATION NORMAL: GPIb-VWF axis intact → key DDx from BSS. "
                    "DISTINGUISHING GT-A from GT-B: "
                    "GT-A (ITGA2B): CD41 (GPIIb) absent by flow cytometry; "
                    "GT-B (ITGB3): CD61 (GPIIIa / beta-3) absent by flow cytometry. "
                    "TREATMENT PREFERENCE: Recombinant FVIIa (NovoSeven) for major bleeding/surgery — "
                    "bypasses need for platelet aggregation; preserves future platelet transfusion option "
                    "by avoiding HLA/HPA alloimmunization."
                ),
                "clinical_rule": "GT: normal count + zero aggregation to ALL agonists + absent clot retraction → GPIIb/IIIa flow cytometry → sequence ITGA2B + ITGB3; rFVIIa (NovoSeven) first-line for surgery",
            },
            {
                "term": "MYH9-Related Disease — Neutrophil Inclusions + Giant Platelets + Systemic Features",
                "short": "AD MYH9 missense → giant platelets + Döhle-like inclusions in neutrophils; SNHL 35-60%; nephritis 25-70%; NO aspirin",
                "detail": (
                    "MYH9-RD encompasses May-Hegglin anomaly, Fechtner syndrome, Epstein syndrome, and Sebastian syndrome — "
                    "all caused by AD missense variants in MYH9 encoding non-muscle myosin IIA heavy chain. "
                    "THE PATHOGNOMONIC COMBINATION: "
                    "Giant platelets (megathrombocytopenia) + Döhle-like pale blue cytoplasmic inclusions in neutrophils. "
                    "The neutrophil inclusions represent MYH9 mutant protein aggregates — "
                    "they are NOT true Döhle bodies (which are ribosomes) but appear similar on routine staining. "
                    "Anti-MYH9 immunofluorescence confirms the inclusions. "
                    "EXTRA-PLATELET FEATURES (myosin IIA expressed in multiple tissues): "
                    "(1) SNHL: progressive sensorineural hearing loss (35-60%); audiogram at diagnosis and 2-yearly; "
                    "cochlear implants effective when severe. "
                    "(2) NEPHRITIS: progressive proteinuria + haematuria + CKD (25-70%); "
                    "Alport-like histology; ACE INHIBITOR MANDATORY on nephritis — renoprotective. "
                    "(3) CATARACTS: lens opacity (10-20%); annual ophthalmology review. "
                    "BLEEDING vs COUNT: Bleeding severity is surprisingly mild relative to platelet count "
                    "because giant platelets are more haemostatically active per platelet — "
                    "AVOID spurious platelet transfusion purely for count."
                ),
                "clinical_rule": "MYH9-RD: giant platelets + neutrophil inclusions = PATHOGNOMONIC; audiogram + urine ACR mandatory; ACE inhibitor when nephritis; NO aspirin/NSAIDs",
            },
            {
                "term": "Gray Platelet Syndrome (GPS) — Alpha-Granule Deficiency and Myelofibrosis",
                "short": "AR NBEAL2 biallelic → absent alpha-granules → gray platelets; myelofibrosis 3rd-4th decade; bone marrow monitoring mandatory",
                "detail": (
                    "GPS (NBEAL2-related, AR) presents with moderate thrombocytopenia, large pale platelets "
                    "('gray' on Wright-Giemsa = absent alpha-granule contents), and progressive myelofibrosis. "
                    "ALPHA-GRANULE CONTENTS CONSTITUTIVELY SECRETED: Without NBEAL2, alpha-granule cargo "
                    "(PDGF, TGF-beta, fibrinogen, PF4, VWF) is not sorted into granules but secreted directly "
                    "by megakaryocytes into marrow stroma → drives stromal fibrosis. "
                    "MYELOFIBROSIS CONSEQUENCE: Progressive marrow fibrosis → extramedullary haematopoiesis → "
                    "splenomegaly (near-universal in adults) → pancytopenia; "
                    "myelofibrosis appears in 3rd-4th decade in most NBEAL2 patients. "
                    "BONE MARROW SURVEILLANCE: Biopsy at diagnosis, then every 3-5 years from age 25 — "
                    "ruxolitinib (JAK inhibitor) for myelofibrosis-related symptoms if needed. "
                    "FLOW CYTOMETRY: P-selectin (CD62P) absent on activation (alpha-granule membrane marker); "
                    "VWF multimer analysis NORMAL (plasma VWF unaffected — GPS is a storage defect, not synthesis defect). "
                    "NOTE: GFI1B and FLI1 AD variants also cause GPS phenotype but with AD inheritance and less myelofibrosis."
                ),
                "clinical_rule": "GPS (NBEAL2): gray platelets on film → electron microscopy confirms absent alpha-granules; bone marrow biopsy from age 25; P-selectin flow = absent; splenomegaly monitor",
            },
            {
                "term": "ANKRD26 Thrombocytopenia-2 (THC2) — 5'UTR Mutations Missed by WES",
                "short": "AD ANKRD26 5'UTR point mutations disrupt RUNX1/FLI1 repressor binding; mild thrombocytopenia; AML/MDS 5-8%; standard WES FAILS to detect",
                "detail": (
                    "THC2 (ANKRD26-related thrombocytopenia) is caused by point mutations in the 5' untranslated "
                    "region (5'UTR/promoter) of ANKRD26 — specifically a 19-nucleotide window "
                    "where RUNX1 and FLI1 transcription factors normally bind to REPRESS ANKRD26 in mature megakaryocytes. "
                    "DIAGNOSTIC PITFALL: These 5'UTR variants are NOT captured by standard whole-exome sequencing (WES) "
                    "because WES baits target coding exons — the promoter/UTR region is typically not enriched or covered. "
                    "MANDATORY: Targeted ANKRD26 5'UTR sequencing panel or whole-genome sequencing (WGS) required. "
                    "MALIGNANCY RISK: AML/MDS in 5-8% of carriers — mechanism unclear but ANKRD26 overexpression "
                    "in megakaryocytes and progenitors activates MAP kinase and TPO/MPL signalling. "
                    "THERAPEUTIC IMPLICATION: Thrombopoietin receptor agonists (eltrombopag, romiplostim) — "
                    "mechanistically stimulate the same pathway → AVOID in ANKRD26 THC2 (unquantified risk of AML promotion). "
                    "PLATELET FUNCTION NORMAL: Unlike RUNX1 FPD/AML, platelet function in ANKRD26 THC2 is preserved — "
                    "aggregation studies are normal; bleeding is mild."
                ),
                "clinical_rule": "ANKRD26 THC2: standard WES MISSES 5'UTR variants — order targeted sequencing or WGS; AVOID TPO-agonists; annual CBC + haematology; family cascade testing",
            },
            {
                "term": "GFI1B-Related Thrombocytopenia — Red Cell Macrocytosis Pathognomonic",
                "short": "AD GFI1B haploinsufficiency → thrombocytopenia + red cell macrocytosis (MCV >100) + absent delta-granules + CD34 on platelets",
                "detail": (
                    "GFI1B (Growth Factor Independent 1B) is a zinc finger transcriptional repressor "
                    "essential for BOTH megakaryocyte and erythroid maturation. "
                    "UNIQUE CLUE: RED CELL MACROCYTOSIS (MCV 100-118 fL) WITHOUT ANAEMIA — "
                    "this finding in a thrombocytopenic patient points directly to GFI1B-related disease. "
                    "B12/folate must first be excluded, but their absence + macrocytosis + thrombocytopenia = GFI1B until proven otherwise. "
                    "CD34 ON MATURE PLATELETS: GFI1B is required to silence CD34 (a haematopoietic stem cell marker) "
                    "as megakaryocytes mature → GFI1B haploinsufficiency → CD34 persists on platelets → "
                    "diagnostic on flow cytometry (CD34 should be ABSENT on normal mature platelets). "
                    "DELTA-GRANULE (DENSE BODY) DEFICIENCY: Absent dense granules → "
                    "impaired secondary aggregation wave (no ADP/serotonin release) → "
                    "mepacrine (quinacrine) uptake and fluorescence ABSENT on flow = diagnostic. "
                    "GPS VARIANT: ~15-20% of GFI1B patients also have absent alpha-granules → gray platelet phenotype; "
                    "GFI1B regulates both granule types."
                ),
                "clinical_rule": "Thrombocytopenia + MCV >100 (no B12/folate deficiency) → GFI1B first; confirm CD34+ on platelets (flow) + absent mepacrine uptake; electron microscopy = absent dense granules",
            },
            {
                "term": "RUNX1 Familial Platelet Disorder with AML (FPD/AML) — AML Risk 35-44%",
                "short": "AD RUNX1 germline haploinsufficiency; AML/MDS 35-44% lifetime; NO aspirin/NSAIDs; cascade test all first-degree; HSCT excludes family donors",
                "detail": (
                    "RUNX1 (AML1/CBFA2) is the most important transcription factor in haematopoiesis — "
                    "somatic RUNX1 mutations are the most frequent acquired abnormality in AML and ALL. "
                    "Germline RUNX1 haploinsufficiency causes FPD/AML, with: "
                    "(1) Thrombocytopenia (typically 80-150k, sometimes milder); "
                    "(2) Platelet dysfunction (GPVI reduced → impaired collagen aggregation); "
                    "(3) MYELOID MALIGNANCY: AML/MDS in 35-44% lifetime, T-ALL also reported. "
                    "CRITICAL MANAGEMENT RULES: "
                    "NO ASPIRIN / NO NSAIDs: Platelet function already reduced; adding inhibitors risks serious bleeding. "
                    "ANNUAL CBC: Every year mandatory; bone marrow biopsy every 2-3 years or if CBC changes. "
                    "HSCT DONOR EXCLUSION: If AML/MDS develops and HSCT planned, EXCLUDE all first-degree relatives — "
                    "50% carry the RUNX1 variant → would transplant the same predisposition. "
                    "Use unrelated/cord blood donors only. "
                    "CASCADE TESTING: ALL first-degree relatives must be offered RUNX1 germline testing; "
                    "positive carriers → identical surveillance protocol. "
                    "SOMATIC vs GERMLINE: Somatic RUNX1 mutations in AML without family history are common — "
                    "germline FPD/AML requires family history + germline testing (not tumour biopsy)."
                ),
                "clinical_rule": "RUNX1 FPD/AML: annual CBC mandatory; NO aspirin; HSCT → unrelated donor ONLY; cascade test first-degree relatives; bone marrow every 2-3 years",
            },
            {
                "term": "Platelet Granule Defects — Alpha-Granule vs Delta-Granule Distinction",
                "short": "Alpha-granule defect (GPS/NBEAL2): gray platelets, absent P-selectin, VWF absent in platelets. Delta-granule defect (GFI1B): absent mepacrine uptake, absent ADP release",
                "detail": (
                    "Platelet granule disorders are classified by granule type: "
                    "ALPHA-GRANULE DEFICIENCY (Gray Platelet Syndrome, NBEAL2): "
                    "Alpha-granules contain fibrinogen, VWF, P-selectin (CD62P), PF4, beta-TG, PDGF, TGF-beta, factor V. "
                    "Absent alpha-granules → pale 'gray' platelets on Wright-Giemsa → "
                    "reduced P-selectin surface expression on activation (flow cytometry); "
                    "constitutive secretion of alpha-granule contents → myelofibrosis (in NBEAL2/AR). "
                    "DELTA-GRANULE (DENSE BODY) DEFICIENCY (GFI1B, Hermansky-Pudlak, Chediak-Higashi): "
                    "Dense granules contain ADP, ATP, serotonin, calcium, polyphosphates — "
                    "the 'signal amplifiers' of platelet activation. "
                    "Absent dense granules → no secondary aggregation wave → mepacrine uptake ABSENT. "
                    "COMBINED DEFICIENCY: Some patients (Hermansky-Pudlak syndrome) have BOTH alpha + delta deficiency. "
                    "DIAGNOSTIC TESTS: "
                    "Alpha-granule: P-selectin flow cytometry (CD62P); electron microscopy. "
                    "Dense granule: mepacrine (quinacrine) uptake/release flow; lumi-aggregometry (ATP release). "
                    "Whole blood lumi-aggregometry is the gold-standard for combined granule + aggregation assessment."
                ),
                "clinical_rule": "Alpha-granule defect = gray platelets + absent CD62P on flow. Dense granule defect = normal morphology + absent mepacrine uptake + absent ADP secondary wave. Both require specialist platelet lab",
            },
            {
                "term": "Cascade Testing — Hereditary Platelet Disorders",
                "short": "AD disorders (MYH9, ANKRD26, GFI1B, RUNX1): 50% risk per child; all first-degree relatives. AR disorders (GP1BA, ITGA2B, ITGB3, NBEAL2): 25% sib risk; carrier parents asymptomatic",
                "detail": (
                    "Cascade testing in hereditary platelet disorders depends on inheritance: "
                    "AUTOSOMAL DOMINANT (MYH9, ANKRD26, GFI1B, RUNX1): "
                    "50% risk to each child; test parents (may reveal de novo if parents unaffected); "
                    "ALL first-degree relatives must be offered testing; "
                    "RUNX1 cascade is MANDATORY (malignancy risk); "
                    "ANKRD26 carriers need AML/MDS surveillance even if asymptomatic. "
                    "AUTOSOMAL RECESSIVE (GP1BA, ITGA2B, ITGB3, NBEAL2): "
                    "25% sib risk; parents are obligate heterozygous carriers (asymptomatic); "
                    "Pre-natal testing available for next pregnancy; "
                    "NAIT (ITGB3 HPA-1a system): maternal HPA typing MANDATORY in all pregnancies after affected index case. "
                    "PRACTICAL PRIORITY: RUNX1 cascade is most time-critical (AML risk); "
                    "ANKRD26 cascade second priority (AML/MDS risk, WES may be needed); "
                    "AR disorders: carrier counselling + prenatal testing; "
                    "MYH9: systemic complications (nephritis, SNHL) drive surveillance urgency."
                ),
                "clinical_rule": "RUNX1 diagnosis → cascade test ALL first-degree IMMEDIATELY; ANKRD26 → targeted 5'UTR sequencing in family; AR disorders → carrier counselling + prenatal testing",
            },
        ],
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    bd = get_breakdown()
    df = get_definitions()
    print(f"Overview: atlas={ov['atlas_name']}, n={ov['n_patients']}, genes={ov['gene_count']}")
    print(f"Breakdown: {len(bd)} genes")
    print(f"Definitions: {len(df['definitions'])} terms")
    for g in bd:
        print(f"  {g['gene']}: {g['cohort_stats']['n']} patients, seed {g['cohort_stats']['seed']}, "
              f"mean_plt={g['cohort_stats']['mean_platelet_count_k']}k, "
              f"special={g['cohort_stats']['special_features']}")
