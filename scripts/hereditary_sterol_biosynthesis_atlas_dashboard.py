"""Hereditary Sterol-Biosynthesis Atlas — 8-Gene Reference
DHCR7-EBP-NSDHL-SC5D-DHCR24-LSS-MVK-SQLE
320 patients (8 x 40), seeds 2782-2789.
Endpoints: /api/hereditary-sterol-biosynthesis-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "DHCR7",
        "protein": (
            "DHCR7 -- 11q13.4 AR -- 475aa -- 7-Dehydrocholesterol-Reductase-54kDa-"
            "8-TM-Endoplasmic-Reticulum-Sterol-Reductase-Final-Step-Cholesterol-Synthesis-"
            "OMIM-Gene-602858-Disease-Smith-Lemli-Opitz-270400"
        ),
        "locus": "11q13.4",
        "protein_size": "475 aa / 54 kDa (8-TM ER-resident sterol reductase; NADPH-dependent; final step in the Kandutsch-Russell pathway: converts 7-dehydrocholesterol → cholesterol; loss → 7-DHC accumulates; 7-DHC is photoreactive generating oxysterols)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "DHCR7 (7-dehydrocholesterol reductase) catalyses the FINAL step in cholesterol biosynthesis: "
            "  7-dehydrocholesterol (7-DHC) + NADPH → cholesterol (via Kandutsch-Russell pathway); "
            "  Alternatively via Bloch pathway: desmosterol → cholesterol via DHCR24 (parallel final step); "
            "DHCR7 deficiency → 7-DHC accumulates (toxic); cholesterol deficient in all tissues; "
            "PREVALENCE: 1:20,000–1:40,000 live births (European); most common defect in post-squalene cholesterol synthesis; "
            "CARRIER FREQUENCY: ~1:50 in European populations; "
            "SEVERITY SPECTRUM: "
            "  Mild (Rutledge): syndactyly only + mild intellectual disability; "
            "  Classic: multiple congenital anomalies + moderate-severe ID; "
            "  Severe (lethal): holoprosencephaly, hydrops, early death; "
            "GENOTYPE-PHENOTYPE: biallelic null = lethal; W151X + mild allele = milder phenotype; "
            "IVS8-1G→C: severe splice allele; p.Thr93Met: most common mild allele (1% carrier rate in UK)"
        ),
        "disease_category": (
            "SMITH-LEMLI-OPITZ SYNDROME (SLO / RSH) — OMIM 270400; "
            "BIOCHEMICAL HALLMARKS: "
            "  7-dehydrocholesterol (7-DHC) elevated in plasma (normal <1 μmol/L; SLO often >100 μmol/L); "
            "  7-DHC:cholesterol ratio elevated; "
            "  Plasma cholesterol LOW or normal (dietary contribution); "
            "  Urine 7-DHC elevated (spot test); "
            "CLINICAL FEATURES — PATHOGNOMONIC: "
            "  2,3-TOE SYNDACTYLY — 2nd and 3rd toes fused (cutaneous+/-osseous) — PATHOGNOMONIC for SLO; "
            "  Found in >97% of SLO patients; absent in other cholesterol synthesis disorders; "
            "MAJOR FEATURES: "
            "  Microcephaly + brain malformations (holoprosencephaly in severe cases); "
            "  Intellectual disability (mild to severe); "
            "  Facial: ptosis, epicanthal folds, anteverted nares, micrognathia; "
            "  Cleft palate (30-50%); "
            "  Genital abnormalities (46,XY undervirilisation; ambiguous genitalia); "
            "  Cardiac defects (AVSD, VSD, TOF — 35%); "
            "  Renal (duplex kidneys, hydronephrosis — 30%); "
            "  Adrenal insufficiency (cholesterol = cortisol precursor); "
            "  Behavioural: autism spectrum features, self-injurious behaviour, sleep disorders; "
            "  Photosensitivity: 7-DHC absorbs UVA → oxysterol reactive intermediates → phototoxicity; "
            "DIAGNOSTIC PATHWAY: "
            "  1. Clinical suspicion: 2,3-toe syndactyly + multiple anomalies + ID; "
            "  2. Plasma 7-DHC measurement (GC-MS) → diagnostic; "
            "  3. DHCR7 sequencing (biallelic pathogenic variants); "
            "  4. Prenatal: amniocyte 7-DHC levels; fetal DNA DHCR7 if family mutation known"
        ),
        "disease_pathway": (
            "CHOLESTEROL BIOSYNTHESIS — DHCR7 FINAL STEP: "
            "KANDUTSCH-RUSSELL PATHWAY (predominant in adult tissues): "
            "  HMG-CoA → mevalonate → (FDPS/FDFT1) → squalene → (SQLE) → 2,3-oxidosqualene → "
            "  → (LSS) lanosterol → (DHCR24) dihydrolanosterol → ... → 7-dehydrocholesterol → "
            "  → (DHCR7) CHOLESTEROL; "
            "BLOCH PATHWAY (parallel): "
            "  ... → desmosterol → (DHCR24) → cholesterol; "
            "7-DHC TOXICITY: "
            "  7-DHC has 2 conjugated double bonds → UV absorbs → reactive oxysterols (DHCEO, 7-ketocholesterol); "
            "  Oxysterols: apoptosis, mitochondrial dysfunction, autophagy disruption; "
            "  Sunlight exposure → severe phototoxic skin damage in SLO; "
            "CHOLESTEROL ROLES IMPACTED: "
            "  Membrane fluidity and lipid raft formation (Sonic Hedgehog signalling requires cholesterol); "
            "  Steroid hormone synthesis (cortisol, sex hormones → adrenal + gonadal insufficiency); "
            "  Bile acid synthesis (reduced bile); "
            "  Myelination (myelin 70% cholesterol → CNS features); "
            "  Wnt / Hedgehog / smoothened signalling (developmental patterning → anomalies); "
            "TREATMENT: "
            "  Cholesterol supplementation (dietary cholesterol: eggs, meat, dairy); "
            "  Reduces 7-DHC somewhat via feedback; partial clinical benefit; "
            "  Simvastatin paradox: HMGCR inhibition → reduces 7-DHC (less substrate); may improve behaviour; "
            "    → statin REDUCES 7-DHC toxicity despite being expected to worsen cholesterol deficiency; "
            "  Sunscreen mandatory (photoprotection from 7-DHC phototoxicity); "
            "  Adrenal crisis prevention: stress dosing if adrenal insufficiency"
        ),
        "pathognomonic": (
            "2,3-TOE SYNDACTYLY — PATHOGNOMONIC FOR SMITH-LEMLI-OPITZ: "
            "  2nd and 3rd toes are fused (cutaneous webbing ± bony fusion); "
            "  Present in >97% of SLO cases; immediate diagnostic clue in neonate; "
            "  Distinguishes SLO from all other cholesterol synthesis disorders; "
            "  NOT found in CDPX2 (stippled epiphyses), CHILD (unilateral nevus), lathosterolosis, desmosterolosis; "
            "KEY DISTINGUISHING FEATURES vs OTHER STEROL DISORDERS: "
            "  SLO vs lathosterolosis: SLO = 2,3-syndactyly; lathosterolosis = liver disease + NO syndactyly; "
            "  SLO vs desmosterolosis: SLO = 2,3-syndactyly; desmosterolosis = short stature + thick calvariae; "
            "  SLO vs CDPX2: SLO = syndactyly + 7-DHC elevated; CDPX = stippled epiphyses + XLD females only; "
            "BIOCHEMICAL CONFIRMATION: "
            "  7-DHC elevation on plasma sterol profile (GC-MS) is DIAGNOSTIC; "
            "  No other condition raises 7-DHC this strikingly (mild elevation in AY9944 drug); "
            "  7-DHC:cholesterol ratio >0.3 strongly supports SLO; "
            "SIMVASTATIN PARADOX (SLO specific): "
            "  Statins HELP in SLO: reduce 7-DHC burden without always worsening total cholesterol; "
            "  Paradoxical because statin reduces cholesterol synthesis at HMG-CoA level → less 7-DHC upstream; "
            "  Behaviour improvement reported in some SLO patients on simvastatin"
        ),
        "key_facts": [
            "DHCR7-2-3-TOE-SYNDACTYLY-PATHOGNOMONIC",
            "DHCR7-7-DHC-ELEVATED-DIAGNOSTIC",
            "DHCR7-MOST-COMMON-STEROL-DISORDER",
            "DHCR7-CHOLESTEROL-SUPPLEMENTATION",
            "DHCR7-SIMVASTATIN-PARADOX-REDUCES-7DHC",
            "DHCR7-PHOTOSENSITIVITY-7DHC-UVA",
            "DHCR7-ADRENAL-INSUFFICIENCY-CORTISOL",
            "DHCR7-AUTISM-BEHAVIOURAL-FEATURES",
        ],
        "treatment": (
            "Dietary cholesterol supplementation (eggs, meat, dairy, infant formula); "
            "Simvastatin (reduces 7-DHC production — paradoxical benefit for behaviour/toxicity); "
            "Strict photoprotection (SPF50+, UV-blocking clothing — 7-DHC phototoxic under UVA); "
            "Adrenal insufficiency: hydrocortisone stress dosing + MRI-guided monitoring; "
            "Behavioural support + ASD therapies; "
            "Cleft palate repair, cardiac surgery, renal follow-up as indicated; "
            "Prenatal: offering GC-MS amniotic fluid sterol analysis + DHCR7 sequencing; "
            "Carrier testing offered to all first-degree relatives"
        ),
        "seed_base": 2782,
        "n_patients": 40,
    },
    {
        "gene": "EBP",
        "protein": (
            "EBP -- Xp11.23 XLD -- 230aa -- Emopamil-Binding-Protein-25kDa-"
            "5-TM-ER-Sterol-Δ8-Δ7-Isomerase-CDPX2-X-Linked-Dominant-"
            "OMIM-Gene-300205-Disease-CDPX2-302960"
        ),
        "locus": "Xp11.23",
        "protein_size": "230 aa / 25 kDa (5-TM ER-resident cholesterol biosynthesis enzyme; catalyses isomerisation of Δ8-sterols → Δ7-sterols in the post-lanosterol pathway; emopamil (calcium channel blocker) binds EBP — pharmacological tool; also known as Δ8-Δ7 sterol isomerase)",
        "inheritance": (
            "X-LINKED DOMINANT — heterozygous females (MOSAIC); hemizygous males (LETHAL in utero / neonatal); "
            "EBP encodes the sterol Δ8-Δ7 isomerase in the cholesterol biosynthesis post-lanosterol pathway; "
            "EBP deficiency → 8-dehydrocholesterol (8-DHC) and 8(9)-cholestenol accumulate → "
            "  these intermediates are incorporated into membranes → altered membrane properties → "
            "  defective Hedgehog signalling → limb + skin + lens patterning defects; "
            "X-LINKED DOMINANT MECHANISM: "
            "  Heterozygous FEMALES: somatic mosaicism (random X-inactivation); "
            "    cells expressing mutant allele → cholesterol synthesis defect; "
            "    cells expressing WT allele → complement; "
            "    Result: MOSAIC skin and skeletal phenotype along Blaschko lines; "
            "  Hemizygous MALES: no WT allele → complete EBP deficiency → "
            "    lethal mid-gestation (hydrops, skeletal dysplasia); rarely postnatal males (X-karyotype mosaic); "
            "CONRADI-HÜNERMANN-HAPPLE (CDPX2): OMIM 302960; "
            "HAPPLE refers to XLD disorders following Blaschko lines; "
            "EMOPAMIL BINDING: original name from EBP's binding to emopamil (calcium antagonist); unrelated to function"
        ),
        "disease_category": (
            "CDPX2 — CONRADI-HÜNERMANN-HAPPLE SYNDROME (X-LINKED CHONDRODYSPLASIA PUNCTATA TYPE 2) — OMIM 302960; "
            "BIOCHEMICAL HALLMARKS: "
            "  8-dehydrocholesterol (8-DHC) elevated in plasma; "
            "  8(9)-cholestenol elevated; "
            "  Ratio 8-DHC:total sterols elevated (GC-MS sterol profile diagnostic); "
            "  Accumulated abnormal sterols also in hair (hair sterol analysis useful); "
            "CLINICAL FEATURES — PATHOGNOMONIC: "
            "  STIPPLED EPIPHYSES (CHONDRODYSPLASIA PUNCTATA): "
            "    Calcification foci in cartilaginous epiphyses visible on X-ray in neonates; "
            "    Distribution: asymmetric, mosaic (reflects X-inactivation pattern); "
            "    PATHOGNOMONIC for CDPX2 in a female neonate with ichthyosis; "
            "    Stippling RESOLVES with age (disappears by 2-3 years); "
            "    Spine: coronal cleft vertebrae (split vertebral bodies on sagittal X-ray); "
            "MAJOR FEATURES (all mosaic = follow Blaschko lines): "
            "  ICHTHYOSIFORM ERYTHRODERMA at birth → coarse ichthyosis (hyperkeratosis in mosaic pattern); "
            "  ICY FOLLICULAR ATROPHODERMA: follicular atrophic pits (ice-pick scars) — late-onset adult feature; "
            "  ALOPECIA: patchy scarring alopecia (mosaic hair loss along Blaschko lines); "
            "  CATARACTS: sectoral lens opacities (60-80%); asymmetric; "
            "  ASYMMETRIC LIMB SHORTENING: rhizomelic shortening (proximal — humerus, femur) ± scoliosis; "
            "  Facial: flat nasal bridge, frontal bossing; "
            "MALES: usually lethal; rare liveborn males mosaic (47,XXY karyotype or somatic mosaic); "
            "DIAGNOSTIC PATHWAY: "
            "  1. Neonatal female: ichthyosis + stippled epiphyses on X-ray + cataracts; "
            "  2. Plasma/hair GC-MS sterol profile: 8-DHC/8(9)-cholestenol elevated; "
            "  3. EBP sequencing; "
            "  4. Skin biopsy: Blaschko-pattern enzyme activity mosaicism"
        ),
        "disease_pathway": (
            "CHOLESTEROL BIOSYNTHESIS — EBP STEROL ISOMERASE STEP: "
            "POST-LANOSTEROL PATHWAY: "
            "  Lanosterol → (multiple steps) → 8-dehydrocholesterol → (EBP Δ8→Δ7 isomerase) → "
            "  → 7-dehydrocholesterol → (DHCR7) → cholesterol; "
            "  ALTERNATIVE: lanosterol → ... → lathosterol → (SC5D) → 7-DHC → (DHCR7) → cholesterol; "
            "EBP STEP: "
            "  EBP isomerises the double bond at position 8 → position 7 (Δ8 → Δ7 shift); "
            "  This is required for DHCR7 to complete the final reduction step; "
            "  EBP deficiency → 8-DHC and 8(9)-cholestenol accumulate (cannot proceed to 7-DHC); "
            "DOWNSTREAM EFFECTS: "
            "  Cholesterol deficiency → impaired Hedgehog (SHH, IHH) signalling: "
            "    IHH (Indian Hedgehog) normally stimulates PTHLH in perichondrium → regulates endochondral bone formation; "
            "    IHH requires cholesterol modification for secretion + PTCH1 receptor binding; "
            "    IHH deficiency → disordered endochondral ossification → stippled epiphyses; "
            "  Cholesterol deficiency in skin → defective barrier → ichthyosis; "
            "  Cholesterol deficiency in lens → cataracts (lens cholesterol essential for transparency); "
            "MOSAICISM MECHANISM: "
            "  Random X-inactivation creates two cell populations: "
            "    WT-expressing cells: normal cholesterol synthesis → normal; "
            "    Mutant-expressing cells: 8-DHC accumulation + cholesterol deficiency → pathological; "
            "  Skin: Blaschko-pattern ichthyosis/atrophoderma matches X-inactivation clone boundaries; "
            "  Stippled epiphyses: mosaic cartilage cells (some WT, some mutant) → asymmetric calcification; "
            "TREATMENT: "
            "  No disease-modifying therapy; "
            "  Topical emollients for ichthyosis; "
            "  Ophthalmology: cataract extraction if visually significant; "
            "  Orthopaedics: scoliosis monitoring + surgery; limb-length discrepancy management; "
            "  Statins (theoretical: reduce intermediates upstream); evidence limited; "
            "  Genetic counselling: 50% daughters affected; sons typically lethal"
        ),
        "pathognomonic": (
            "STIPPLED EPIPHYSES (CHONDRODYSPLASIA PUNCTATA) IN NEONATAL FEMALE — PATHOGNOMONIC for CDPX2: "
            "  Asymmetric punctate calcifications in epiphyses on skeletal X-ray; "
            "  Present at birth, resolve by age 2-3 years (transient); "
            "  Combined with mosaic ichthyosis → highly specific for CDPX2; "
            "  NOTE: stippled epiphyses also in RCDP (rhizomelic CDP, PEX7 gene — AR, peroxisomal) and warfarin embryopathy; "
            "    CDPX2 distinguished: XLD female, asymmetric, 8-DHC elevated, EBP mutation; "
            "    RCDP distinguished: AR, plasmalogen deficiency, peroxisomal disease; "
            "MOSAIC BLASCHKO-PATTERN SKIN — PATHOGNOMONIC DISTRIBUTION: "
            "  Ichthyosis/atrophoderma following Blaschko lines (V-shapes on back, S-curves on trunk); "
            "  This distribution pattern = X-linked mosaicism in skin → points to EBP over autosomal disorders; "
            "GC-MS HAIR STEROL ANALYSIS: "
            "  8-DHC in hair shaft — can diagnose even in adults when skin is clear; "
            "  Hair sterol analysis: non-invasive, diagnostic even post-childhood when stippling resolves; "
            "KEY DDx: "
            "  CDPX1 (ARSE gene, Xp22.3): XLR; males only; stippling milder; normal skin; "
            "  CDPX2 vs SLO: CDPX2 = stippling + 8-DHC; SLO = 2,3-syndactyly + 7-DHC; "
            "  CDPX2 vs CHILD: CHILD = unilateral strictly midline; CHILD = NSDHL gene; "
            "  CDPX2 vs Congenital hypothyroidism: hypothyroid can have mild stippling but not 8-DHC elevation"
        ),
        "key_facts": [
            "EBP-STIPPLED-EPIPHYSES-PATHOGNOMONIC-CDPX2",
            "EBP-MOSAIC-BLASCHKO-FEMALE-ONLY",
            "EBP-MALES-LETHAL-IN-UTERO",
            "EBP-8-DHC-ELEVATED-DIAGNOSTIC",
            "EBP-STIPPLING-RESOLVES-AGE-2-3",
            "EBP-HAIR-STEROL-ADULT-DIAGNOSIS",
            "EBP-CATARACTS-SECTORAL-60-80PCT",
            "EBP-XLD-50PCT-DAUGHTERS-AFFECTED",
        ],
        "treatment": (
            "No disease-modifying therapy; "
            "Topical emollients + keratolytics for ichthyosis (urea creams, lactic acid); "
            "Ophthalmology: cataract monitoring → extraction if visually significant; "
            "Orthopaedics: scoliosis surveillance + surgery; limb length discrepancy management; "
            "Dermatology: sun protection (abnormal skin barrier); "
            "Statins: theoretical benefit (reduce 8-DHC upstream) — anecdotal reports; "
            "Genetic counselling: 50% daughters will be affected (XLD); recurrence risk counselling; "
            "Prenatal: chorionic villus sampling or amniocentesis for EBP mutation + sterol profile"
        ),
        "seed_base": 2783,
        "n_patients": 40,
    },
    {
        "gene": "NSDHL",
        "protein": (
            "NSDHL -- Xq28 XLD -- 374aa -- NAD(P)H-Steroid-Dehydrogenase-Like-Protein-41kDa-"
            "3-Hydroxysterol-C-4-Decarboxylase-ER-Membrane-Sterol-Demethylation-"
            "OMIM-Gene-300275-Disease-CHILD-308050-CK-309520"
        ),
        "locus": "Xq28",
        "protein_size": "374 aa / 41 kDa (ER-resident NAD(P)H-dependent 3-hydroxysterol-C-4 decarboxylase; catalyses C-4 demethylation step in post-lanosterol cholesterol biosynthesis; forms complex with SC4MOL/MSMO1; NSDHL removes C-4 methyl groups from sterol intermediates)",
        "inheritance": (
            "X-LINKED DOMINANT — hemizygous males (usually LETHAL in utero); heterozygous females (mosaic — viable); "
            "NSDHL: NADP(H) steroid dehydrogenase-like — part of C-4 sterol demethylation complex; "
            "C-4 demethylation removes two methyl groups at C-4 of lanosterol intermediates: "
            "  NSDHL (C-4 decarboxylase) acts with SC4MOL (C-4 methylsterol oxidase) in sequence; "
            "  Without NSDHL → C-4 methylated sterols accumulate (4α-methylcholesta-8,24-dien-3β-ol and related); "
            "TWO DISTINCT PHENOTYPES: "
            "  CHILD SYNDROME (Congenital Hemidysplasia with Ichthyosiform nevus and Limb Defects): "
            "    Females only (mosaic X-inactivation); "
            "    Inflammatory ichthyosiform skin nevus STRICTLY UNILATERAL (one body side); "
            "    Ipsilateral limb defects (hypoplasia to complete limb absence); "
            "    Skin follows sharp midline: abrupt cessation at body midline — PATHOGNOMONIC; "
            "  CK SYNDROME (males — hypomorphic NSDHL alleles): "
            "    X-linked intellectual disability + cerebral malformations + facial features in males; "
            "    Allelic to CHILD but hypomorphic (partial function retained); "
            "    Males survive because allele retains residual NSDHL enzymatic activity"
        ),
        "disease_category": (
            "CHILD SYNDROME — OMIM 308050 + CK SYNDROME — OMIM 309520; "
            "BIOCHEMICAL HALLMARKS: "
            "  4α-methylcholesta-8-en-3β-ol and related C-4-methylated sterol intermediates elevated; "
            "  Cholesterol synthesis: moderately reduced in skin cells (compensated by systemic circulation); "
            "  GC-MS sterol analysis of skin: C-4 methylated intermediates diagnostic; "
            "CHILD SYNDROME — CLINICAL FEATURES — PATHOGNOMONIC: "
            "  UNILATERAL INFLAMMATORY ICHTHYOSIFORM NEVUS STRICTLY STOPS AT BODY MIDLINE: "
            "    One of the most distinctive phenotypes in all of medicine; "
            "    Right-sided more common (2:1 right:left ratio — unknown mechanism); "
            "    Skin: yellow-orange waxy ichthyosiform scales; sometimes verrucous; "
            "    Periflexural accentuation (groin, axilla, popliteal fossa — called CHILD nevus); "
            "  IPSILATERAL LIMB DEFECTS: "
            "    Same side as skin nevus; variable severity (hypoplastic limb → complete absence); "
            "    Hypoplastic nails, reduced bone density ipsilateral; "
            "  INTERNAL ORGAN IPSILATERAL ABNORMALITIES: "
            "    Heart: pulmonic stenosis, ASD (ipsilateral); "
            "    Kidney: hypoplasia/aplasia ipsilateral; "
            "    Lung: ipsilateral hypoplasia; "
            "  IPSILATERAL CNS: brain hemisphere abnormality possible; "
            "  NO CONTRALATERAL INVOLVEMENT — midline is absolute; "
            "CK SYNDROME — CLINICAL FEATURES: "
            "  Males; intellectual disability (usually moderate-severe); "
            "  Hypotonia; small/slender stature; lissencephaly/pachygyria; "
            "  Relative macrocephaly; thin sparse hair; hypertelorism; "
            "  NO ichthyosis (skin is normal in CK syndrome); "
            "DIAGNOSTIC PATHWAY: "
            "  1. CHILD: unilateral ichthyosiform nevus stopping at midline → NSDHL sequencing; "
            "  2. Skin biopsy sterol analysis; "
            "  3. CK syndrome: male with ID + brain malformations → NSDHL + X-exome sequencing"
        ),
        "disease_pathway": (
            "CHOLESTEROL BIOSYNTHESIS — NSDHL C-4 DEMETHYLATION COMPLEX: "
            "POST-LANOSTEROL C-4 DEMETHYLATION: "
            "  Lanosterol has two methyl groups at C-4 position → must be removed; "
            "  Step 1: SC4MOL (C-4-methylsterol oxidase/MSMO1) — oxidises C-4 methyl group to carboxyl; "
            "  Step 2: NSDHL (3β-hydroxysterol-C-4 decarboxylase) — removes carboxyl → decarboxylation; "
            "  Step 3: MSMO1 repeat for second C-4 methyl group; "
            "  Product: 4-desmethylsterol (C-4 demethylated intermediate → progresses to cholesterol); "
            "NSDHL DEFICIENCY → C-4-METHYLATED STEROLS ACCUMULATE: "
            "  These intermediates are structurally abnormal → incorporated into membranes; "
            "  Abnormal membranes → Hedgehog signalling defect: "
            "    SHH cholesterol modification blocked → SHH secretion/gradient defective; "
            "    IHH disruption → disordered limb patterning (limb defects, ipsilateral); "
            "  Mosaicism: skin cells expressing mutant NSDHL → inflammatory reaction → ichthyosiform nevus; "
            "  The sharp MIDLINE = embryological left-right axis; mutant cell clones don't cross midline; "
            "STATIN TREATMENT FOR CHILD SYNDROME: "
            "  Topical simvastatin/lovastatin: reduces C-4 methyl sterol intermediates; "
            "  Topical statin application to CHILD nevus → dramatic clinical improvement reported (Happle 1996); "
            "  Mechanism: HMGCR inhibition → less substrate for NSDHL pathway → less toxic intermediate; "
            "  Also: cholesterol supplementation in topical vehicle (dual approach); "
            "  This is one of the few hereditary cholesterol synthesis disorders responsive to topical therapy"
        ),
        "pathognomonic": (
            "UNILATERAL ICHTHYOSIFORM NEVUS STRICTLY FOLLOWING THE BODY MIDLINE — PATHOGNOMONIC FOR CHILD SYNDROME: "
            "  The skin lesion stops ABRUPTLY at the MIDLINE: "
            "    Chest/abdomen: lesion on one side, completely normal on the other; "
            "    No gradual transition — sharp demarcation at midline; "
            "  This absolute midline boundary is unique to CHILD syndrome among all dermatoses; "
            "  Other mosaic skin disorders (incontinentia pigmenti, EBP/CDPX2) follow Blaschko lines — "
            "    Blaschko lines do NOT follow the midline; CHILD does; "
            "RIGHT-SIDE PREPONDERANCE (2:1): "
            "  Right-sided CHILD more common; mechanism unclear but reproducible finding; "
            "PERIFLEXURAL ACCENTUATION: "
            "  CHILD nevus is most pronounced in flexures (axilla, groin, neck, popliteal); "
            "  Can look like unilateral psoriasis or inverse psoriasis — biopsy + sterol analysis distinguishes; "
            "TOPICAL STATIN RESPONSE: "
            "  Application of simvastatin/lovastatin cream → dramatic improvement/resolution of CHILD nevus; "
            "  THIS TREATMENT RESPONSE IS UNIQUE TO CHILD SYNDROME (confirms NSDHL diagnosis); "
            "KEY DDx: "
            "  CHILD vs CDPX2: CHILD = strictly midline, NSDHL; CDPX2 = Blaschko lines, EBP, stippled epiphyses; "
            "  CHILD vs linear naevus sebaceous: NSLS = keratinocytic nevus; no sterol abnormality; no limb defect; "
            "  CHILD vs ILVEN (inflammatory linear verrucous epidermal nevus): ILVEN = pruritic, somatic mosaicism, no sterol abnormality"
        ),
        "key_facts": [
            "NSDHL-UNILATERAL-MIDLINE-PATHOGNOMONIC-CHILD",
            "NSDHL-IPSILATERAL-LIMB-DEFECTS",
            "NSDHL-MALES-LETHAL-CK-HYPOMORPHIC",
            "NSDHL-TOPICAL-STATIN-DRAMATIC-RESPONSE",
            "NSDHL-RIGHT-SIDED-2-TO-1-PREPONDERANCE",
            "NSDHL-C4-METHYLATED-STEROLS-DIAGNOSTIC",
            "NSDHL-PERIFLEXURAL-ACCENTUATION",
            "NSDHL-SHH-SIGNALLING-DEFECT-LIMBS",
        ],
        "treatment": (
            "CHILD syndrome: Topical simvastatin or lovastatin (0.5-2% in propylene glycol/ethanol vehicle) to CHILD nevus — dramatic response reported; "
            "Topical cholesterol supplementation (5% cholesterol ointment) — dual therapy with statin; "
            "Emollients for ichthyosis management; "
            "Orthopaedics: limb-length discrepancy, prosthetics for absent limb; "
            "Cardiology: echocardiogram (ipsilateral cardiac defects); "
            "Renal: ultrasound monitoring; "
            "CK syndrome: intellectual disability support, seizure management; neurological surveillance; "
            "Genetic counselling: XLD, 50% daughters affected"
        ),
        "seed_base": 2784,
        "n_patients": 40,
    },
    {
        "gene": "SC5D",
        "protein": (
            "SC5D -- 11q23.3 AR -- 299aa -- Sterol-C5-Desaturase-33kDa-"
            "5-TM-ER-Fatty-Acid-Hydroxylase-Superfamily-Δ5-Desaturation-Lathosterol-to-7-DHC-"
            "OMIM-Gene-604370-Disease-Lathosterolosis-607330"
        ),
        "locus": "11q23.3",
        "protein_size": "299 aa / 33 kDa (ER-resident Δ5-desaturase; fatty acid hydroxylase superfamily; introduces double bond at C-5 of lathosterol → 7-dehydrocholesterol; requires molecular oxygen + cytochrome b5 electron donor; rate-limiting for 7-DHC generation in Kandutsch-Russell pathway)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "SC5D (sterol C5-desaturase, also called SC5DL) introduces a double bond at the C-5 position of the sterol ring: "
            "  Lathosterol (5α-cholest-7-en-3β-ol) → 7-dehydrocholesterol (cholesta-5,7-dien-3β-ol); "
            "SC5D is the PENULTIMATE enzyme in Kandutsch-Russell pathway (before DHCR7); "
            "SC5D deficiency → lathosterol accumulates; 7-DHC cannot be formed; DHCR7 has no substrate; "
            "ULTRA-RARE: approximately 20-25 cases reported worldwide (as of 2025); "
            "FIRST CASE: 2007 (Krakowiak et al.); "
            "GENOTYPE-PHENOTYPE: all reported cases: biallelic loss-of-function; "
            "Heterozygous carriers: asymptomatic; elevated lathosterol:cholesterol ratio (useful for carrier detection)"
        ),
        "disease_category": (
            "LATHOSTEROLOSIS — OMIM 607330; "
            "BIOCHEMICAL HALLMARKS: "
            "  Lathosterol markedly elevated (plasma GC-MS); "
            "  7-dehydrocholesterol absent or greatly reduced (no SC5D → can't make 7-DHC); "
            "  Cholesterol low to normal (dietary contribution); "
            "  Lathosterol:cholesterol ratio elevated (diagnostic; also in heterozygotes mildly); "
            "  Bile acids: modified (lathosterol → lathocholate, a C-5 unsaturated bile acid — hepatotoxic); "
            "CLINICAL FEATURES: "
            "  LIVER DISEASE: progressive hepatic fibrosis and cirrhosis; "
            "    Lathocholate (abnormal bile acid from lathosterol) → hepatotoxicity; "
            "    Presentations range from neonatal cholestasis → cirrhosis → liver failure; "
            "  BRAIN MALFORMATIONS: polymicrogyria, lissencephaly; intellectual disability; "
            "  SKIN: ichthyosis (sometimes; variable); "
            "  DYSMORPHIC FEATURES: similar to SLO but NO 2,3-toe syndactyly; "
            "    micrognathia, limb anomalies; "
            "  GROWTH: failure to thrive + short stature; "
            "  NOTE: 2,3-toe syndactyly ABSENT (distinguishes from SLO/DHCR7); "
            "  Severity variable: some cases severe multisystem + others milder; "
            "DIAGNOSTIC PATHWAY: "
            "  1. Sterol profile (GC-MS plasma): lathosterol elevated; 7-DHC absent → strongly suggests SC5D; "
            "  2. SC5D gene sequencing (biallelic variants); "
            "  3. Enzymatic assay (fibroblasts: lathosterol → 7-DHC conversion assay); "
            "  4. Liver biopsy: fibrosis staging; hepatic sterol analysis"
        ),
        "disease_pathway": (
            "CHOLESTEROL BIOSYNTHESIS — SC5D PENULTIMATE STEP: "
            "KANDUTSCH-RUSSELL PATHWAY: "
            "  ... → lathosterol → (SC5D Δ5-desaturase) → 7-dehydrocholesterol → (DHCR7) → cholesterol; "
            "SC5D REACTION: "
            "  Lathosterol (one double bond at Δ7) → SC5D introduces second double bond at Δ5 → "
            "  Product: 7-dehydrocholesterol (two conjugated double bonds: Δ5 and Δ7); "
            "  Requires: O2, NADH, cytochrome b5; "
            "LATHOSTEROL TOXICITY: "
            "  Lathosterol is a saturated analog of cholesterol (one double bond instead of normal Δ5 position); "
            "  Poorly incorporates into lipid rafts; disrupts membrane function; "
            "  Importantly: lathosterol → lathocholate (bile acid analog) → "
            "    Lathocholate is hepatotoxic → accumulates in bile canaliculi → cholestasis → fibrosis; "
            "LIVER IS MOST SEVERELY AFFECTED: "
            "  High hepatic sterol flux → greatest lathocholate accumulation; "
            "  Liver disease is often the most clinically prominent and prognostically important feature; "
            "TREATMENT: "
            "  Dietary cholesterol supplementation (reduce endogenous synthesis flux); "
            "  Ursodeoxycholic acid (UDCA): hepatoprotection, reduces lathocholate hepatotoxicity; "
            "  Liver transplantation: corrects hepatic sterol synthesis defect; "
            "    Improves liver disease; may not correct CNS manifestations (brain synthesises own cholesterol); "
            "  Simvastatin: controversial (reduces lathosterol via HMGCR inhibition — may help); "
            "  Cholesterol supplementation + UDCA first-line; liver transplant for progressive disease"
        ),
        "pathognomonic": (
            "LATHOSTEROL ELEVATION WITHOUT 7-DHC ELEVATION — DIAGNOSTIC FOR LATHOSTEROLOSIS: "
            "  GC-MS sterol profile: elevated lathosterol + absent/low 7-DHC + cholesterol low; "
            "  This pattern distinguishes SC5D deficiency from: "
            "    DHCR7 deficiency (SLO): 7-DHC elevated; lathosterol normal; "
            "    EBP deficiency (CDPX2): 8-DHC elevated; lathosterol normal; "
            "    DHCR24 deficiency: desmosterol elevated; "
            "CLINICAL DISTINGUISHING FEATURES: "
            "  SC5D vs SLO (DHCR7): SC5D = NO 2,3-toe syndactyly; liver disease prominent; lathosterol elevated; "
            "  SC5D vs CDPX2 (EBP): SC5D = AR, both sexes, liver disease; CDPX2 = XLD females, skin stippling; "
            "  SC5D vs DHCR24: SC5D = lathosterol; DHCR24 = desmosterol, statins worsen; "
            "HEPATIC PRESENTATION: "
            "  Neonatal/infantile cholestasis + elevated lathosterol → suspect SC5D even before full phenotype; "
            "  Lathocholate (hepatotoxic bile acid from lathosterol) — marker of SC5D-related liver injury; "
            "LATHOSTEROL:CHOLESTEROL RATIO IN CARRIERS: "
            "  Heterozygous SC5D carriers have mildly elevated lathosterol:cholesterol ratio; "
            "  Useful for family cascade testing (distinguishes true carriers from non-carriers)"
        ),
        "key_facts": [
            "SC5D-LATHOSTEROL-ELEVATED-DIAGNOSTIC",
            "SC5D-NO-2-3-SYNDACTYLY-DDX-SLO",
            "SC5D-LIVER-DISEASE-LATHOCHOLATE",
            "SC5D-ULTRA-RARE-LT25-CASES",
            "SC5D-UDCA-HEPATOPROTECTION",
            "SC5D-LIVER-TRANSPLANT-OPTION",
            "SC5D-BRAIN-MALFORMATION-POLYMICROGYRIA",
            "SC5D-PENULTIMATE-STEP-BEFORE-DHCR7",
        ],
        "treatment": (
            "Dietary cholesterol supplementation (eggs, meat — reduce endogenous synthesis); "
            "Ursodeoxycholic acid (UDCA 10-15 mg/kg/day): hepatoprotection against lathocholate; "
            "Simvastatin: controversial but may reduce lathosterol accumulation; "
            "Liver transplantation: for progressive hepatic fibrosis/failure; corrects liver but not CNS defect; "
            "Nutritional support: fat-soluble vitamins (A,D,E,K) with cholestasis; "
            "Neurological support: physiotherapy, educational support for brain malformations; "
            "Genetic counselling: AR disorder, 25% recurrence risk"
        ),
        "seed_base": 2785,
        "n_patients": 40,
    },
    {
        "gene": "DHCR24",
        "protein": (
            "DHCR24 -- 1p32.3 AR -- 516aa -- 24-Dehydrocholesterol-Reductase-60kDa-"
            "FAD-Dependent-ER-Oxidoreductase-Desmosterol-to-Cholesterol-Bloch-Pathway-Final-Step-"
            "OMIM-Gene-606418-Disease-Desmosterolosis-602398"
        ),
        "locus": "1p32.3",
        "protein_size": "516 aa / 60 kDa (ER-resident FAD-dependent oxidoreductase; reduces Δ24 double bond of desmosterol → cholesterol; also reduces Δ24 double bonds at multiple upstream steps; DHCR24 acts at Bloch pathway final step AND at multiple upstream sterol intermediates; also known as seladin-1, 24-DHCR)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "DHCR24 (24-dehydrocholesterol reductase) reduces the C-24 double bond in: "
            "  BLOCH PATHWAY: desmosterol → cholesterol (final step); "
            "  ALSO: Δ24 double bond at multiple upstream positions in both Bloch and K-R pathways; "
            "DHCR24 deficiency → desmosterol accumulates + all Δ24-unsaturated intermediates; "
            "ULTRA-RARE: approximately 15 cases reported worldwide; "
            "FIRST CASE: 2001 (Waterham et al.); "
            "ALSO KNOWN AS: seladin-1 (selective Alzheimer's disease indicator-1) — DHCR24 is "
            "  downregulated in Alzheimer's disease (reduced brain cholesterol synthesis correlates); "
            "STATINS WORSEN: "
            "  DHCR24 deficiency + statin → HMGCR inhibited → less upstream flux → worsens cholesterol deficiency; "
            "  UNLIKE SLO where statins paradoxically help; "
            "  CRITICAL clinical distinction: NEVER give statins to DHCR24 deficiency patients"
        ),
        "disease_category": (
            "DESMOSTEROLOSIS — OMIM 602398; "
            "BIOCHEMICAL HALLMARKS: "
            "  Desmosterol markedly elevated in plasma, tissues (especially brain); "
            "  Cholesterol severely reduced in severe cases; "
            "  All Δ24-unsaturated sterol intermediates elevated; "
            "  GC-MS sterol profile: desmosterol peak — pathognomonic; "
            "CLINICAL FEATURES: "
            "  BRAIN: intellectual disability + brain malformations (variable: ACC, lissencephaly, small cerebellum); "
            "  GROWTH: intrauterine growth retardation + short stature; "
            "  THICK CALVARIAE: thickened skull vault (desmosterol alters bone mineralisation); "
            "  LIMB SHORTENING: variable limb length reduction; "
            "  CLEFT PALATE (50%); "
            "  FACIAL: flat nasal bridge, widely spaced teeth, frontal bossing; "
            "  NO 2,3-toe syndactyly (distinguishes from SLO); "
            "  NO stippled epiphyses (distinguishes from CDPX2); "
            "  CARDIAC: ASD/VSD (30-40%); "
            "  TOTAL BODY RESPONSE: widespread desmosterol incorporation into all membranes; "
            "  SEVERITY: variable; some with normal-ish cognitive outcome; others severe ID; "
            "DIAGNOSTIC PATHWAY: "
            "  1. Desmosterol elevated on GC-MS sterol profile → strongly suggests DHCR24; "
            "  2. DHCR24 sequencing (biallelic); "
            "  3. Enzymatic assay (desmosterol → cholesterol conversion in fibroblasts)"
        ),
        "disease_pathway": (
            "CHOLESTEROL BIOSYNTHESIS — DHCR24 BLOCH PATHWAY FINAL STEP + MULTIPLE EARLIER STEPS: "
            "BLOCH PATHWAY: "
            "  Lanosterol → (multiple steps) → zymosterol → (MVD/other steps) → 7-dehydrodesmosterol → "
            "  → (DHCR24 Δ24 reduction at zymosterol AND subsequent steps) → lathosterol/7-DHC → "
            "  → desmosterol → (DHCR24 FINAL Δ24 reduction) → CHOLESTEROL; "
            "DHCR24 IS PROMISCUOUS: "
            "  Reduces Δ24 double bond at multiple points along the Bloch pathway; "
            "  Not just the final step — multiple substrates; "
            "  Deficiency → build-up of ALL Δ24-unsaturated intermediates; "
            "DESMOSTEROL TOXICITY: "
            "  Desmosterol has a C-24 double bond → alters membrane packing vs cholesterol; "
            "  CNS: brain relies on endogenous cholesterol synthesis (blood-brain barrier = no plasma cholesterol enters); "
            "    Desmosterol accumulation disrupts: "
            "      Myelin formation (myelin cholesterol 70%); "
            "      Neuronal membrane signalling; "
            "      Hedgehog/Wnt developmental signalling; "
            "SELADIN-1 / ALZHEIMER CONNECTION: "
            "  DHCR24 (seladin-1) was identified as downregulated in selectively vulnerable neurons in Alzheimer's disease; "
            "  Reduced DHCR24 → accumulating desmosterol → altered neuronal cholesterol → amyloid processing?; "
            "  DHCR24 protects against oxidative stress (independent of cholesterol synthesis); "
            "TREATMENT CRITICAL POINT: "
            "  STATINS CONTRAINDICATED: unlike SLO (where statins help), statins worsen DHCR24 deficiency; "
            "  Rationale: statin reduces mevalonate flux → less substrate → even less cholesterol; "
            "  Dietary cholesterol supplementation (primary treatment); "
            "  No disease-modifying targeted therapy currently approved"
        ),
        "pathognomonic": (
            "DESMOSTEROL ELEVATION ON GC-MS STEROL PROFILE — DIAGNOSTIC FOR DESMOSTEROLOSIS: "
            "  Desmosterol peak on plasma GC-MS: characteristic retention time; "
            "  Desmosterol:cholesterol ratio dramatically elevated; "
            "  Absent/low 7-DHC and lathosterol: distinguishes from SLO and lathosterolosis; "
            "STATINS CONTRAINDICATED (critical clinical fact): "
            "  Unlike Smith-Lemli-Opitz (DHCR7) where statins paradoxically reduce 7-DHC and may help, "
            "  In DHCR24 deficiency statins WORSEN cholesterol deficiency — never prescribe; "
            "  This distinction is clinically critical when a patient with a sterol disorder is prescribed statins; "
            "THICK CALVARIAE (THICKENED SKULL VAULT): "
            "  Present on CT/MRI head: thick skull bones; "
            "  Relatively specific for desmosterolosis vs other sterol synthesis disorders; "
            "  Mechanism: desmosterol incorporation into bone matrix alters mineral density; "
            "KEY DDx: "
            "  DHCR24 vs DHCR7 (SLO): desmosterol vs 7-DHC; statins worsen vs statins help; thick calvaria vs 2,3-syndactyly; "
            "  DHCR24 vs SC5D: desmosterol vs lathosterol; liver disease in SC5D but not prominently in DHCR24; "
            "  DHCR24 vs EBP: AR vs XLD; desmosterol vs 8-DHC; no stippling in DHCR24; "
            "SELADIN-1 LINK: "
            "  DHCR24/seladin-1 downregulation in Alzheimer's disease neurons → connection to neurodegeneration; "
            "  Patients with biallelic LOF have complete absence — different from age-related partial downregulation"
        ),
        "key_facts": [
            "DHCR24-DESMOSTEROL-ELEVATED-DIAGNOSTIC",
            "DHCR24-STATINS-CONTRAINDICATED-WORSEN",
            "DHCR24-THICK-CALVARIAE-SKULL",
            "DHCR24-NO-2-3-SYNDACTYLY-DDX-SLO",
            "DHCR24-SELADIN-1-ALZHEIMER-LINK",
            "DHCR24-BLOCH-PATHWAY-FINAL-STEP",
            "DHCR24-ULTRA-RARE-LT15-CASES",
            "DHCR24-DIETARY-CHOLESTEROL-FIRST-LINE",
        ],
        "treatment": (
            "Dietary cholesterol supplementation (primary — reduced endogenous synthesis + exogenous source); "
            "STATINS ABSOLUTELY CONTRAINDICATED (worsen cholesterol deficiency — unlike SLO where they help); "
            "Fat-soluble vitamin supplementation (A,D,E,K); "
            "Supportive: cleft palate repair, cardiac surgery; "
            "Neurological: educational support, physiotherapy; "
            "Brain-directed cholesterol delivery: investigational (liposome-mediated CNS cholesterol delivery); "
            "Genetic counselling: AR, 25% recurrence risk; prenatal: desmosterol in amniotic fluid"
        ),
        "seed_base": 2786,
        "n_patients": 40,
    },
    {
        "gene": "LSS",
        "protein": (
            "LSS -- 21q22.3 AR -- 733aa -- Lanosterol-Synthase-83kDa-"
            "Oxidosqualene-Cyclase-Converts-2-3-Oxidosqualene-to-Lanosterol-"
            "First-Cyclic-Sterol-Cholesterol-Synthesis-"
            "OMIM-Gene-600909-Disease-Cataracts-Alopecia-617021"
        ),
        "locus": "21q22.3",
        "protein_size": "733 aa / 83 kDa (ER-resident oxidosqualene cyclase; catalyses cyclisation of 2,3-oxidosqualene → lanosterol; this is the FIRST ring-closure step — converting a linear precursor to the steroid ring system; no cofactor required; suicide inhibitor: oxysterol; widely expressed including lens epithelium and hair follicle)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF or hypomorphic; "
            "LSS (lanosterol synthase) performs the KEY RING-CYCLISATION step of cholesterol biosynthesis: "
            "  2,3-oxidosqualene → lanosterol (first tetracyclic sterol with protosterol structure); "
            "  This single step creates the steroid skeleton — prerequisite for ALL subsequent modifications; "
            "LSS deficiency → 2,3-oxidosqualene accumulates; NO lanosterol; NO downstream sterols; "
            "SEVERITY DEPENDS ON RESIDUAL ACTIVITY: "
            "  Complete null: likely embryonic lethal (cholesterol essential for embryogenesis); "
            "  Hypomorphic (partial activity): viable; tissue-specific manifestations where LSS is highly expressed; "
            "HIGHLY EXPRESSED TISSUES: lens epithelium, hair follicle dermal papilla; "
            "RARE: fewer than 20 families reported as of 2025; "
            "LANOSTEROL + CATARACTS LINK (2015 Science paper): "
            "  Adding lanosterol (the LSS product) dissolves protein aggregates in human cataracts in vitro; "
            "  Eye drops with lanosterol: Phase I/II studies ongoing; "
            "  Mechanistic insight: why LSS deficiency → cataracts (loss of lanosterol-mediated chaperoning)"
        ),
        "disease_category": (
            "LSS DEFICIENCY — CATARACTS + ALOPECIA — OMIM 617021 (non-syndromic cataracts 50); "
            "BIOCHEMICAL HALLMARKS: "
            "  2,3-oxidosqualene accumulation (GC-MS analysis); "
            "  Lanosterol absent or greatly reduced; "
            "  Downstream cholesterol: partially maintained by diet (liver/intestine may have residual LSS); "
            "  Lens-specific: lens normally relies almost exclusively on de novo cholesterol synthesis; "
            "CLINICAL FEATURES: "
            "  CATARACTS: congenital/infantile bilateral dense cataracts; "
            "    Early-onset (neonatal to infantile); visually significant requiring extraction; "
            "    Dense nuclear cataracts; "
            "    Lens epithelium has high LSS expression → very sensitive to LSS deficiency; "
            "  ALOPECIA: total or near-total alopecia (scalp ± eyebrows/lashes); "
            "    Hair follicle dermal papilla: high LSS expression; dermal papilla cholesterol essential for "
            "    Wnt signalling in hair cycle; LSS deficiency → hair follicle cycle arrest; "
            "    Alopecia can be congenital (no hair at birth) or progressive; "
            "  SPECTRUM: some patients cataracts only; some cataracts + alopecia; "
            "  GENERALLY NORMAL NEURODEVELOPMENT (distinguishing feature — no brain malformations); "
            "  No limb defects; no organ malformations (unlike SLO, desmosterolosis); "
            "  Skin: may have mild xerosis; no ichthyosis; "
            "DIAGNOSTIC PATHWAY: "
            "  1. Bilateral dense cataracts + alopecia in infant → suspect LSS; "
            "  2. Gene panel (cataract/alopecia genes): LSS sequencing; "
            "  3. Plasma/urine sterol profile: 2,3-oxidosqualene elevated; lanosterol absent; "
            "  4. Enzymatic assay: LSS activity in fibroblasts"
        ),
        "disease_pathway": (
            "CHOLESTEROL BIOSYNTHESIS — LSS KEY RING-CYCLISATION STEP: "
            "PRE-LANOSTEROL (MEVALONATE) PATHWAY: "
            "  Acetyl-CoA → HMG-CoA → (HMGCR) mevalonate → (kinases/decarboxylase) → "
            "  → isopentenyl-PP → geranyl-PP → farnesyl-PP → (FDFT1/squalene synthase) → squalene → "
            "  → (SQLE/squalene epoxidase) → 2,3-oxidosqualene → (LSS) → LANOSTEROL; "
            "  Lanosterol → (multiple steps) → cholesterol; "
            "LSS IS THE ENTRY POINT TO STEROL SYNTHESIS: "
            "  No LSS → no lanosterol → no downstream sterols (no cholesterol, ergosterol, bile acids, steroid hormones); "
            "  Complete LSS null would be lethal; viable patients have hypomorphic alleles; "
            "LANOSTEROL AS LENS CHAPERONE: "
            "  2015 Science paper (Zhao et al.): lanosterol reverses protein aggregation in cataractous lenses; "
            "  Crystallins (lens proteins) normally soluble; mutant crystallins aggregate → cataract; "
            "  Lanosterol: acts as hydrophobic chaperone → disaggregates crystallin aggregates; "
            "  LSS deficiency → no endogenous lanosterol → crystallins aggregate → cataract; "
            "  This explains why LSS mutations cause cataracts specifically (high lens LSS expression + no dietary lanosterol supply); "
            "HAIR FOLLICLE CHOLESTEROL REQUIREMENT: "
            "  Dermal papilla (hair follicle mesenchymal core): "
            "    Wnt signalling: requires cholesterol for LRP5/6-mediated angiogenesis + papilla cell growth; "
            "    LSS deficiency → dermal papilla cholesterol deficiency → hair follicle cannot cycle; "
            "    Alopecia (usually non-scarring, potentially reversible if treated); "
            "TREATMENT: "
            "  Cataract: early surgical extraction + optical correction (contact lenses/glasses); "
            "  Alopecia: topical lanosterol (investigational); topical cholesterol supplementation; "
            "  Dietary cholesterol: systemic supplementation (may partially correct somatic tissues); "
            "  Lanosterol eye drops: investigational (dissolving lens opacities in Phase I trials)"
        ),
        "pathognomonic": (
            "BILATERAL CONGENITAL CATARACTS + TOTAL ALOPECIA IN INFANT — HIGH SPECIFICITY FOR LSS DEFICIENCY: "
            "  This two-sign combination (cataracts + alopecia) with normal neurodevelopment and no major anomalies "
            "  is highly specific for LSS deficiency; "
            "LANOSTEROL DEFICIENCY CATARACT MECHANISM: "
            "  Lanosterol normally prevents crystallin aggregation; "
            "  LSS deficiency → no lanosterol → crystallins aggregate → dense nuclear cataracts early in life; "
            "  UNIQUE TO LSS: other cholesterol synthesis disorders don't specifically cause this degree of cataracts; "
            "    (EBP/CDPX2 can cause sectoral cataracts but mechanism different; SLO can have cataracts but minor); "
            "ALOPECIA DISTRIBUTION: "
            "  Total scalp alopecia; eyebrows/eyelashes may also be absent; "
            "  Distinguishes from common alopecia areata (patchy, autoimmune); "
            "  Distinguishes from ectodermal dysplasias (different genes, no sterol abnormality); "
            "NORMAL NEURODEVELOPMENT: "
            "  LSS deficiency notably spares the brain (unlike SLO, desmosterolosis, lathosterolosis); "
            "  Patients typically have normal IQ; "
            "  This is because: liver and intestine likely have residual LSS activity + dietary cholesterol; "
            "    brain relies on de novo synthesis but may have compensatory mechanisms; "
            "KEY DDx: "
            "  LSS vs congenital cataracts (CRYAA, GJA8 etc.): sterol profile normal in other cataract genes; "
            "  LSS vs alopecia areata: AA is autoimmune, episodic; LSS = congenital, stable, sterol abnormality; "
            "  LSS vs ectodermal dysplasia (EDA, EDAR): ED has teeth + sweat gland anomalies; sterol normal"
        ),
        "key_facts": [
            "LSS-CATARACTS-ALOPECIA-COMBINATION",
            "LSS-LANOSTEROL-CRYSTALLIN-CHAPERONE",
            "LSS-NORMAL-NEURODEVELOPMENT",
            "LSS-RING-CYCLISATION-FIRST-CYCLIC-STEROL",
            "LSS-LENS-HAIR-HIGH-EXPRESSION",
            "LSS-LANOSTEROL-EYE-DROPS-INVESTIGATIONAL",
            "LSS-2-3-OXIDOSQUALENE-ELEVATED",
            "LSS-HYPOMORPHIC-ALLELES-VIABLE",
        ],
        "treatment": (
            "Cataracts: early surgical extraction (within weeks of diagnosis) + optical correction; "
            "Lanosterol eye drops: investigational (Phase I/II — dissolving crystallin aggregates); "
            "Alopecia: topical cholesterol/lanosterol preparations (investigational); "
            "Dietary cholesterol supplementation (systemic support); "
            "Visual rehabilitation: contact lenses, glasses (early amblyopia prevention critical); "
            "Regular ophthalmology follow-up for posterior capsule opacification; "
            "Genetic counselling: AR, 25% recurrence risk"
        ),
        "seed_base": 2787,
        "n_patients": 40,
    },
    {
        "gene": "MVK",
        "protein": (
            "MVK -- 12q24.11 AR -- 396aa -- Mevalonate-Kinase-41kDa-"
            "Phosphorylates-Mevalonate-to-Mevalonate-5-Phosphate-GHMP-Kinase-Superfamily-"
            "Isoprenoid-Pathway-Upstream-of-Farnesyl-PP-Squalene-"
            "OMIM-Gene-251170-Disease-MVK-Def-610377-HIDS-260920-Mevalonic-Aciduria-610377"
        ),
        "locus": "12q24.11",
        "protein_size": "396 aa / 41 kDa (GHMP kinase superfamily; homodimer; phosphorylates mevalonic acid → mevalonate-5-phosphate using ATP; acts directly downstream of HMGCR; essential for isoprenoid pathway including cholesterol, dolichol, ubiquinone, geranylgeranyl-PP, farnesyl-PP biosynthesis)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF or hypomorphic; "
            "MVK acts immediately downstream of HMGCR (the rate-limiting statin-targeted enzyme): "
            "  Mevalonate → (MVK) → mevalonate-5-phosphate → (PMVK) → mevalonate-5-PP → "
            "  → (MVD) → isopentenyl-PP → (multiple steps) → farnesyl-PP → squalene → ... → cholesterol; "
            "MVK deficiency → mevalonate accumulates; ALL downstream isoprenoids deficient; "
            "NOT JUST CHOLESTEROL DEFICIENCY: "
            "  Farnesyl-PP deficit → impaired protein farnesylation (Ras, lamins); "
            "  Geranylgeranyl-PP deficit → impaired GTPase prenylation (Rac, Rho, Rab); "
            "  Dolichol deficit → impaired N-glycosylation; "
            "  Ubiquinone deficit → mitochondrial complex I–III function impaired; "
            "SEVERITY SPECTRUM: "
            "  Mevalonic aciduria (MA): severe, near-complete MVK deficiency (residual activity <1%); "
            "  HIDS (Hyper-IgD syndrome): mild, hypomorphic MVK (residual activity 1-10%); "
            "  Overlapping intermediate forms exist"
        ),
        "disease_category": (
            "MVK DEFICIENCY — MEVALONIC ACIDURIA (MA) + HYPER-IgD SYNDROME (HIDS): "
            "MEVALONIC ACIDURIA (MA) — OMIM 610377 (severe): "
            "  Biochemical: mevalonic acid markedly elevated in urine (10-100x normal); "
            "  Clinical: "
            "    Psychomotor retardation (severe intellectual disability); "
            "    Cerebellar ataxia (progressive); "
            "    Progressive dysmorphic features (micrognathia, malar hypoplasia, dysplastic ears); "
            "    Cataracts; "
            "    Hepatosplenomegaly; "
            "    Normocytic anaemia (haemolytic + dyserythropoietic); "
            "    Recurrent febrile crises (overlap with HIDS feature); "
            "    Growth failure; "
            "    Cardiomyopathy; "
            "    Death in childhood if untreated (severe form); "
            "HYPER-IgD SYNDROME (HIDS) — OMIM 260920 (mild/hypomorphic): "
            "  Biochemical: mild mevalonic aciduria (detectable during fever attacks only); "
            "  IgD markedly elevated (>100 IU/mL) — used as biomarker (not pathogenic); "
            "  PERIODIC FEVER SYNDROME: "
            "    Episodes: 3-7 days fever; frequency 4-6 per year; triggered by stress/infection/immunisation; "
            "    Associated: LYMPHADENOPATHY (cervical, prominent), abdominal pain, diarrhoea, vomiting; "
            "    Skin: macular/papular rash; arthralgia; headache; "
            "    Onset: childhood (usually <1 year first episode); "
            "    Attacks decrease in frequency after puberty (may remit spontaneously); "
            "  IgD >100 IU/mL: present in 80% of HIDS but elevated IgD is NOT specific; "
            "DIAGNOSTIC PATHWAY: "
            "  1. MA: urine mevalonic acid (significantly elevated always); MVK sequencing; "
            "  2. HIDS: urine mevalonic acid (elevated during fever attacks; normal between); "
            "     IgD + IgA elevated; MVK sequencing (I268T = most common HIDS allele, 80% of HIDS)"
        ),
        "disease_pathway": (
            "ISOPRENOID/MEVALONATE PATHWAY — MVK STEP (UPSTREAM OF CHOLESTEROL SYNTHESIS): "
            "MEVALONATE PATHWAY OVERVIEW: "
            "  Acetyl-CoA → HMG-CoA → (HMGCR) mevalonate → (MVK) mevalonate-5-P → (PMVK) mevalonate-5-PP → "
            "  → (MVD) isopentenyl-PP ← → dimethylallyl-PP → (FDPS) geranyl-PP → farnesyl-PP → "
            "    → (FDFT1) squalene → ... → cholesterol; "
            "    → geranylgeranyl-PP → prenylation of Rho/Rac/Rab GTPases; "
            "    → farnesyl-PP → prenylation of Ras/KRAS/lamins; "
            "    → dolichol → N-glycosylation; "
            "    → ubiquinone (CoQ10) → electron transport; "
            "WHY FEVER ATTACKS IN HIDS/MVK DEFICIENCY: "
            "  Mevalonate accumulation activates NLRP3 inflammasome directly → IL-1β, IL-18 secretion; "
            "  During fever (when MVK is thermolabile hypomorphic enzyme): "
            "    Heat → HIDS MVK enzyme activity further decreases → "
            "    Mevalonate surge → inflammasome activation → IL-1β spike → fever; "
            "  Vicious cycle: fever → more MVK inactivation → more mevalonate → more IL-1β → more fever; "
            "PROTEIN PRENYLATION DEFICIT: "
            "  Farnesyl-PP deficit → impaired Ras farnesylation → defective immune cell signalling; "
            "  Geranylgeranyl-PP deficit → impaired Rac1/RhoA → cytoskeleton dysfunction in leukocytes; "
            "  Dolichol deficit → defective glycoprotein folding; "
            "  These combined explain multi-system features of MA (severe form); "
            "TREATMENT: "
            "  HIDS: IL-1β inhibitors — ANAKINRA (IL-1Ra, daily subcutaneous) or CANAKINUMAB (anti-IL-1β, monthly); "
            "    Canakinumab: FDA/EMA approved for HIDS/TRAPS/CAPS (2016); "
            "    Reduce attack frequency and severity dramatically; "
            "  MA: no specific curative therapy; supportive; "
            "    Geranylgeraniol supplementation (GGsuppl): replenishes isoprenoids downstream; Phase I trials; "
            "    Simvastatin: PARADOXICALLY WORSENS (further blocks mevalonate production) — AVOID in MA; "
            "  HIDS: simvastatin also AVOIDED (reduces MVK substrate — worsens during attacks); "
            "  Bone marrow transplant: investigated for MA (corrects haematopoietic defects)"
        ),
        "pathognomonic": (
            "PERIODIC FEVER SYNDROME + MARKEDLY ELEVATED IgD >100 IU/mL + LYMPHADENOPATHY IN CHILD — HIDS: "
            "  CERVICAL LYMPHADENOPATHY is PROMINENT during attacks (more than other autoinflammatory syndromes); "
            "  Tender cervical nodes; "
            "  Combined with high IgD + periodic fever: characteristic HIDS pattern; "
            "URINE MEVALONIC ACID (KEY DIAGNOSTIC): "
            "  MA (severe): always markedly elevated; "
            "  HIDS: elevated during fever attack (often not elevated between attacks); "
            "    TIMING OF URINE COLLECTION CRITICAL: collect during fever attack for HIDS diagnosis; "
            "  GC-MS urine organic acid panel or targeted mevalonic acid assay; "
            "MVK I268T ALLELE — HIDS FOUNDER: "
            "  I268T (c.803T→C): most common HIDS allele; "
            "  Thermolabile: 37°C OK; 40°C (fever) → enzyme unfolds → activity crashes → mevalonate surge → attack; "
            "  Explains why fever triggers attacks: thermolabile MVK-I268T loses function precisely when fever starts; "
            "KEY DDx AMONG PERIODIC FEVER SYNDROMES: "
            "  HIDS vs FMF (MEFV gene): FMF = Mediterranean; serosal attacks; colchicine responsive; no high IgD; "
            "  HIDS vs TRAPS (TNFRSF1A gene): TRAPS = longer attacks (>1 week); periorbital oedema; myalgia; "
            "  HIDS vs PFAPA: PFAPA = very regular; responds to corticosteroids; no MVK mutation; "
            "  HIDS vs CAPS (NLRP3 gene): urticarial rash; cold-triggered; CNS features"
        ),
        "key_facts": [
            "MVK-HIDS-PERIODIC-FEVER-IgD-ELEVATED",
            "MVK-I268T-THERMOLABILE-FOUNDER-ALLELE",
            "MVK-URINE-MEVALONIC-ACID-DURING-ATTACK",
            "MVK-CERVICAL-LYMPHADENOPATHY-PROMINENT",
            "MVK-ANAKINRA-CANAKINUMAB-FDA-APPROVED",
            "MVK-MEVALONIC-ACIDURIA-SEVERE-FORM",
            "MVK-STATINS-WORSEN-AVOID",
            "MVK-NLRP3-INFLAMMASOME-IL1B-MECHANISM",
        ],
        "treatment": (
            "HIDS: Canakinumab (anti-IL-1β, 150mg SC every 4 weeks — FDA/EMA approved 2016 for HIDS); "
            "Anakinra (IL-1Ra, 1-2 mg/kg/day SC — effective alternative); "
            "NSAIDs: symptomatic during attacks; "
            "Colchicine: less effective than in FMF but sometimes used adjunctively; "
            "STATINS: CONTRAINDICATED (worsen MVK deficiency by reducing mevalonate substrate); "
            "MA (severe): geranylgeraniol supplementation (investigational, GG-OH restores prenylation); "
            "Bone marrow transplant: investigated for severe MA; "
            "Genetic counselling: AR, 25% recurrence risk"
        ),
        "seed_base": 2788,
        "n_patients": 40,
    },
    {
        "gene": "SQLE",
        "protein": (
            "SQLE -- 8q24.13 AR/AD -- 574aa -- Squalene-Epoxidase-Squalene-Monooxygenase-64kDa-"
            "FAD-Dependent-Microsomal-Oxidase-Converts-Squalene-to-2-3-Oxidosqualene-"
            "OMIM-Gene-602019-Disease-Alopecia-Hypotrichosis-617960"
        ),
        "locus": "8q24.13",
        "protein_size": "574 aa / 64 kDa (ER-resident FAD-dependent monooxygenase; uses molecular O2 + NADPH + FAD; converts squalene → 2,3-oxidosqualene — the substrate for LSS ring-cyclisation; SQLE is the rate-limiting step for squalene channelling into sterol synthesis; also known as squalene monooxygenase or SM)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → hereditary alopecia) or "
            "AUTOSOMAL DOMINANT (GOF amplification → squamous cell cancer susceptibility); "
            "SQLE (squalene epoxidase) catalyses: squalene + O2 + NADPH + FAD → 2,3-oxidosqualene; "
            "  This step introduces the oxygen to the squalene double bond → epoxide → "
            "  2,3-oxidosqualene then cyclises (LSS) → lanosterol; "
            "SQLE IS RATE-LIMITING FOR STEROL FLUX: "
            "  Terbinafine (antifungal) inhibits fungal SQLE (ergosterol synthesis); "
            "  Mammalian SQLE is similar but less sensitive; "
            "  NB-598 (experimental SQLE inhibitor): lipid-lowering agent candidate; "
            "BIALLELIC LOF → ALOPECIA: "
            "  Loss of SQLE → squalene accumulates; no 2,3-oxidosqualene → no lanosterol → no cholesterol; "
            "  Hair follicle highly sensitive (high SQLE expression in dermal papilla); "
            "  Squalene accumulation is itself toxic to hair follicle (squalene peroxides); "
            "GOF AMPLIFICATION → CANCER: "
            "  SQLE is amplified in ~10% of squamous cell carcinomas (head/neck, oesophagus); "
            "  Amplified SQLE → increased cholesterol synthesis → cancer growth advantage; "
            "  SQLE amplification: oncogene in some tumours (sterol flux drives cell proliferation)"
        ),
        "disease_category": (
            "SQLE DEFICIENCY — HEREDITARY ALOPECIA/HYPOTRICHOSIS — OMIM 617960; "
            "BIOCHEMICAL HALLMARKS: "
            "  Squalene markedly elevated in scalp/skin (sebum squalene elevated — measurable); "
            "  2,3-oxidosqualene absent or greatly reduced; "
            "  Lanosterol and downstream sterols: reduced (partially compensated by liver/dietary); "
            "  Sebum squalene content: usually 2-5% of total sebum; in SQLE deficiency: greatly elevated; "
            "CLINICAL FEATURES (BIALLELIC LOF): "
            "  ALOPECIA: congenital or infantile-onset alopecia; "
            "    Often complete scalp alopecia; eyebrows/eyelashes variable; "
            "    Non-scarring alopecia (hair follicle present but in arrest); "
            "    Squalene accumulation in scalp sebum → toxic squalene peroxides → follicular damage; "
            "  HYPOTRICHOSIS: generalised sparse hair (milder biallelic hypomorphic cases); "
            "  SYSTEMIC FEATURES: generally minimal (liver compensates via dietary/alternative synthesis); "
            "  SEBORRHOEA: increased sebum squalene → greasy/scaly scalp; "
            "  NO MAJOR ORGAN MALFORMATIONS (unlike SLO, desmosterolosis); "
            "  GENERALLY NORMAL NEURODEVELOPMENT; "
            "SQLE AMPLIFICATION / GOF (oncology context): "
            "  SQLE amplification: 8q24.13 amplicon in head+neck/oesophageal SCC; "
            "  Not a hereditary syndrome per se — somatic amplification in cancer; "
            "  Relevance: patients with known SQLE LOF who develop cancer — SQLE inhibitors potentially beneficial; "
            "DIAGNOSTIC PATHWAY: "
            "  1. Congenital/infantile alopecia + elevated scalp squalene; "
            "  2. SQLE sequencing (biallelic LOF alleles); "
            "  3. GC-MS scalp sebum: squalene:total lipid ratio elevated"
        ),
        "disease_pathway": (
            "CHOLESTEROL BIOSYNTHESIS — SQLE UPSTREAM STEP (SQUALENE EPOXIDATION): "
            "PATHWAY: "
            "  Farnesyl-PP → (FDFT1/squalene synthase) → squalene → (SQLE) → 2,3-oxidosqualene → "
            "  → (LSS) → lanosterol → ... → cholesterol; "
            "SQLE MECHANISM: "
            "  Squalene (6 double bonds, acyclic, C30) + O2 + NADPH + FAD → "
            "  → 2,3-oxidosqualene (one end epoxidised); "
            "  The 2,3-epoxide activates the molecule for LSS-catalysed cyclisation; "
            "SQUALENE ACCUMULATION (SQLE deficiency): "
            "  Squalene accumulates in sebum (skin surface lipid) — measurable; "
            "  Squalene itself is relatively non-toxic; "
            "  BUT: squalene + oxygen (UV, sebaceous gland) → squalene hydroperoxides; "
            "  Squalene peroxides → follicular inflammation → follicular occlusion → alopecia; "
            "  Same mechanism proposed for acne (squalene peroxidation in sebum → comedone + inflammation); "
            "SQLE AND ANTIFUNGAL TERBINAFINE: "
            "  Terbinafine is a competitive SQLE inhibitor (fungal SQLE > human SQLE selectivity); "
            "  In dermatophyte fungi: SQLE inhibition → squalene accumulates → fungal cell death; "
            "  Topical terbinafine: used for tinea pedis, onychomycosis; "
            "  Relevance: SQLE-deficient patients should probably avoid terbinafine (already SQLE-null); "
            "CANCER CONTEXT: "
            "  SQLE as oncogene (GOF amplification): cancer cells need cholesterol for proliferation; "
            "  Amplified SQLE → abundant sterol precursor → cholesterol → membranes + signalling lipids; "
            "  NB-598 (SQLE inhibitor): investigated as anti-cancer + lipid-lowering agent; "
            "  Relationship to SQLE biallelic deficiency: suggests SQLE inhibitors may have side effect of alopecia"
        ),
        "pathognomonic": (
            "CONGENITAL ALOPECIA + MARKEDLY ELEVATED SCALP SQUALENE — SQLE DEFICIENCY: "
            "  Squalene in scalp sebum: directly measurable (GC-MS sebum lipid analysis); "
            "  Elevated squalene:total-sebum ratio strongly suggests SQLE deficiency; "
            "  In other alopecia causes: squalene not elevated; "
            "TERBINAFINE INTERACTION — CLINICAL RELEVANCE: "
            "  SQLE-deficient patients: terbinafine (SQLE inhibitor) would further block residual SQLE; "
            "  Potentially exacerbate squalene accumulation and alopecia; "
            "  Clinical pearl: inquire about terbinafine use in unexplained alopecia; "
            "SQUALENE PEROXIDE HAIR FOLLICLE TOXICITY: "
            "  Squalene + UV/oxygen → 4,5-epoxy-2,3-dihydrosqualene (squalene monoepoxide) → "
            "    → comedo + perifollicular inflammation; "
            "  This chemistry occurs at the follicular opening where sebum + oxygen interface; "
            "KEY DDx: "
            "  SQLE vs LSS: both cause alopecia; LSS = cataracts also; LSS = 2,3-oxidosqualene elevated, squalene normal; "
            "    SQLE = squalene elevated; no cataracts; "
            "  SQLE vs alopecia areata: AA = patchy, autoimmune, no squalene elevation; "
            "  SQLE vs androgenetic alopecia: AGA = pattern loss, no congenital onset, no squalene elevation; "
            "  SQLE vs ectodermal dysplasia: ED = teeth + sweat glands + nails; sterol normal; "
            "SQLE AMPLIFICATION IN CANCER: "
            "  Somatic SQLE amplification at 8q24.13 → SQLE GOF → cancer growth; "
            "  Hereditary SQLE LOF patients (carriers): no clear cancer predisposition from LOF; "
            "  Cancer cells with SQLE amplification: sensitive to SQLE inhibitors (NB-598)"
        ),
        "key_facts": [
            "SQLE-CONGENITAL-ALOPECIA-SQUALENE-ELEVATED",
            "SQLE-TERBINAFINE-INHIBITOR-CAUTION",
            "SQLE-SQUALENE-PEROXIDE-FOLLICULAR-TOXICITY",
            "SQLE-CANCER-AMPLIFICATION-GOF-ONCOGENE",
            "SQLE-RATE-LIMITING-SQUALENE-EPOXIDATION",
            "SQLE-NB598-INHIBITOR-INVESTIGATIONAL",
            "SQLE-NO-CATARACTS-DDX-LSS",
            "SQLE-SEBUM-GC-MS-DIAGNOSTIC",
        ],
        "treatment": (
            "SQLE biallelic deficiency (alopecia): "
            "Scalp protection: UV-protective headwear (squalene peroxide formation reduced); "
            "Antioxidant topical preparations: vitamin E / idebenone (reduce squalene peroxide formation); "
            "Avoid terbinafine (SQLE inhibitor → further block residual activity); "
            "Dietary cholesterol supplementation (systemic support); "
            "Wigs/hairpieces for alopecia management; "
            "Topical minoxidil: may stimulate follicle survival (anecdotal); "
            "Investigational: NB-598 (SQLE inhibitor — for SQLE GOF cancer — not for deficiency); "
            "Genetic counselling: biallelic = AR; GOF amplification = somatic cancer context"
        ),
        "seed_base": 2789,
        "n_patients": 40,
    },
]


def _patient_data(gene_info: dict) -> list:
    seed = gene_info["seed_base"]
    rng = random.Random(seed)
    n = gene_info["n_patients"]
    gene = gene_info["gene"]
    rows = []
    for i in range(n):
        age = rng.randint(1, 55)
        sex = rng.choice(["M", "F"])
        rows.append({
            "patient_id": f"STEROL-{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age": age,
            "sex": sex,
            "plasma_cholesterol_mgdl": round(rng.uniform(40, 250), 1),
            "accumulated_sterol": gene_info["key_facts"][0].split("-")[1] if len(gene_info["key_facts"]) > 0 else "unknown",
            "inheritance": gene_info["inheritance"].split(";")[0].strip()[:60],
            "key_phenotype": gene_info["key_facts"][rng.randint(0, min(3, len(gene_info["key_facts"]) - 1))],
        })
    return rows


def generate_overview() -> dict:
    all_patients = []
    for g in ATLAS_GENES:
        all_patients.extend(_patient_data(g))

    total = len(all_patients)
    genes = [g["gene"] for g in ATLAS_GENES]

    avg_chol = round(sum(p["plasma_cholesterol_mgdl"] for p in all_patients) / total, 1)

    per_gene = {}
    for g in ATLAS_GENES:
        pts = [p for p in all_patients if p["gene"] == g["gene"]]
        per_gene[g["gene"]] = {
            "n": len(pts),
            "avg_cholesterol": round(sum(p["plasma_cholesterol_mgdl"] for p in pts) / len(pts), 1),
            "locus": g["locus"],
            "protein_size": g["protein_size"].split("(")[0].strip(),
            "disease": g["disease_category"].split("—")[0].strip()[:60],
        }

    return {
        "atlas": "Hereditary-Sterol-Biosynthesis-Atlas",
        "subtitle": "Complete 8-Gene Post-Squalene Cholesterol Biosynthesis & Isoprenoid Pathway Reference",
        "genes": genes,
        "total_patients": total,
        "seeds": "2782-2789",
        "summary": {
            "avg_plasma_cholesterol_mgdl": avg_chol,
            "per_gene": per_gene,
        },
        "biosynthesis_pathway_steps": {
            "step1_mevalonate_kinase": "MVK — phosphorylates mevalonate → mevalonate-5-P; upstream of squalene; HIDS/mevalonic aciduria",
            "step2_squalene_epoxidase": "SQLE — squalene → 2,3-oxidosqualene (FAD-dependent); terbinafine target; alopecia",
            "step3_lanosterol_synthase": "LSS — 2,3-oxidosqualene → lanosterol (FIRST ring closure); cataracts + alopecia",
            "step4_c4_demethylation_1": "NSDHL — C-4 decarboxylation (with SC4MOL); lanosterol → C-4-desmethyl intermediates; CHILD/CK",
            "step5_delta8_isomerase": "EBP — Δ8-sterol → Δ7-sterol isomerisation; 8-DHC → 7-DHC direction; CDPX2",
            "step6_delta5_desaturase": "SC5D — lathosterol → 7-dehydrocholesterol (Δ5 double bond introduction); Lathosterolosis",
            "step7_delta24_reductase": "DHCR24 — desmosterol → cholesterol (Bloch pathway final step, Δ24 reduction); Desmosterolosis",
            "step8_delta7_reductase": "DHCR7 — 7-dehydrocholesterol → cholesterol (Kandutsch-Russell final step); Smith-Lemli-Opitz",
        },
        "pathognomonic_signs": {
            "DHCR7": "2,3-toe syndactyly — PATHOGNOMONIC for Smith-Lemli-Opitz (>97% of cases); 7-DHC elevated",
            "EBP": "Stippled epiphyses + mosaic Blaschko-line ichthyosis in females — PATHOGNOMONIC CDPX2; 8-DHC elevated",
            "NSDHL": "Unilateral ichthyosiform nevus strictly stopping at midline — PATHOGNOMONIC CHILD syndrome",
            "SC5D": "Lathosterol elevated without 7-DHC elevation; prominent liver disease (lathocholate hepatotoxicity)",
            "DHCR24": "Desmosterol elevated; thick calvariae; STATINS ABSOLUTELY CONTRAINDICATED (worsen)",
            "LSS": "Bilateral congenital cataracts + total alopecia with NORMAL neurodevelopment",
            "MVK": "Periodic fever + cervical lymphadenopathy + IgD >100 IU/mL (HIDS); urine mevalonic acid elevated during attack",
            "SQLE": "Congenital alopecia + elevated scalp squalene (sebum GC-MS); terbinafine interaction risk",
        },
        "critical_statin_rules": {
            "DHCR7_SLO": "Statins PARADOXICALLY HELP — reduce 7-DHC accumulation (HMGCR inhibition → less substrate for DHCR7 pathway); behaviour improves",
            "DHCR24_Desmosterolosis": "Statins ABSOLUTELY CONTRAINDICATED — worsen cholesterol deficiency (reduce desmosterol substrate availability); clinical deterioration",
            "MVK_HIDS": "Statins CONTRAINDICATED — reduce mevalonate availability → worsen MVK deficiency → worse attacks",
            "EBP_CDPX2": "Statins: theoretical benefit (anecdotal reports); evidence limited; cautious use possible",
            "SC5D": "Simvastatin: controversial; may reduce lathosterol via HMGCR inhibition; use cautiously",
            "LSS": "No specific statin contraindication (LSS downstream of HMGCR); statins reduce flux → may worsen",
        },
        "cascade_testing": "Plasma/urine GC-MS sterol profile for all first-degree relatives; gene-specific: DHCR7 sequencing if 2,3-syndactyly; EBP if Blaschko-pattern skin female; NSDHL if unilateral nevus; MVK if periodic fever + high IgD",
        "registered": "2026-09-15",
    }


def generate_breakdown() -> dict:
    result = {}
    for g in ATLAS_GENES:
        pts = _patient_data(g)
        result[g["gene"]] = {
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"][:400],
            "disease_category": g["disease_category"][:600],
            "disease_pathway": g["disease_pathway"][:600],
            "pathognomonic": g["pathognomonic"][:500],
            "key_facts": g["key_facts"],
            "treatment": g["treatment"],
            "n_patients": len(pts),
            "seed": g["seed_base"],
            "sample_patients": pts[:5],
        }
    return result


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Sterol-Biosynthesis-Atlas Glossary",
        "terms": {
            "Cholesterol_Biosynthesis": "Multi-step pathway converting acetyl-CoA → cholesterol via mevalonate, squalene, lanosterol; two parallel final pathways: Kandutsch-Russell (7-DHC → cholesterol via DHCR7) and Bloch (desmosterol → cholesterol via DHCR24)",
            "Mevalonate_Pathway": "Acetyl-CoA → HMG-CoA → (HMGCR) mevalonate → (MVK) → (PMVK) → (MVD) → isopentenyl-PP → farnesyl-PP → squalene; produces ALL isoprenoids: cholesterol, dolichol, ubiquinone, farnesyl-PP, geranylgeranyl-PP",
            "Post_Squalene_Pathway": "Squalene → (SQLE) 2,3-oxidosqualene → (LSS) lanosterol → (NSDHL/SC4MOL) C-4 demethylation → (EBP) Δ8→Δ7 isomerisation → (SC5D) lathosterol → 7-DHC → (DHCR7) cholesterol; Bloch parallel: desmosterol → (DHCR24) cholesterol",
            "DHCR7": "7-dehydrocholesterol reductase (475aa, 11q13.4); final step Kandutsch-Russell; DHCR7 deficiency = Smith-Lemli-Opitz; 2,3-toe syndactyly PATHOGNOMONIC; 7-DHC elevated; simvastatin paradoxically reduces 7-DHC",
            "Smith_Lemli_Opitz": "SLO/RSH syndrome (OMIM 270400); DHCR7 biallelic LOF; 1:20,000 (most common sterol disorder); 2,3-toe syndactyly PATHOGNOMONIC; 7-DHC elevated; cholesterol deficiency; statin HELPS (reduce 7-DHC); dietary cholesterol supplement",
            "EBP": "Emopamil-binding protein (230aa, Xp11.23); Δ8→Δ7 sterol isomerase; XLD → CDPX2 (Conradi-Hünermann-Happle) in females (mosaic); males lethal; stippled epiphyses PATHOGNOMONIC; 8-DHC elevated; Blaschko-line skin",
            "CDPX2": "X-linked dominant chondrodysplasia punctata type 2 (OMIM 302960); EBP mutation; females only (mosaic); stippled epiphyses (transient) + ichthyosis (Blaschko) + cataracts + limb shortening; 8-DHC/8(9)-cholestenol elevated; hair sterol analysis for adult diagnosis",
            "NSDHL": "NAD(P)H steroid dehydrogenase-like (374aa, Xq28); C-4 decarboxylation step; XLD; females → CHILD syndrome; males → CK syndrome (hypomorphic) or lethal; CHILD = unilateral nevus strictly midline PATHOGNOMONIC; topical statin resolves CHILD nevus",
            "CHILD_Syndrome": "Congenital Hemidysplasia Ichthyosiform nevus Limb Defects (OMIM 308050); NSDHL mutation; XLD female; unilateral ichthyosiform nevus ABRUPTLY stopping at midline PATHOGNOMONIC; right-sided 2:1; topical simvastatin/lovastatin = dramatic response (unique to CHILD)",
            "CK_Syndrome": "X-linked ID + brain malformations in males (OMIM 309520); hypomorphic NSDHL alleles; normal skin (no CHILD nevus); allelic to CHILD but partial enzyme activity retained; males survive (unlike CHILD LOF in males = lethal)",
            "SC5D": "Sterol C5-desaturase (299aa, 11q23.3, AR); penultimate step Kandutsch-Russell: lathosterol → 7-DHC; SC5D deficiency = Lathosterolosis (OMIM 607330); ~20 cases; lathosterol elevated; liver disease (lathocholate toxic); NO 2,3-syndactyly",
            "Lathosterolosis": "SC5D deficiency (OMIM 607330); biallelic AR; lathosterol elevated; 7-DHC absent; liver fibrosis (lathocholate hepatotoxicity); brain malformations; ~20 cases; UDCA + dietary cholesterol; liver transplant for progressive disease",
            "Lathocholate": "Abnormal bile acid formed from lathosterol accumulating in SC5D deficiency; hepatotoxic; accumulates in bile canaliculi → cholestasis → fibrosis; UDCA treatment counteracts lathocholate toxicity",
            "DHCR24": "24-dehydrocholesterol reductase (516aa, 1p32.3, AR); Bloch pathway final step: desmosterol → cholesterol; DHCR24 deficiency = Desmosterolosis (OMIM 602398); ~15 cases; desmosterol elevated; statins ABSOLUTELY CONTRAINDICATED; thick calvariae; seladin-1",
            "Desmosterolosis": "DHCR24 deficiency (OMIM 602398); desmosterol elevated; ~15 cases; brain malformations; thick calvariae; statins CONTRAINDICATED (worsen — unlike SLO); dietary cholesterol first-line",
            "Seladin1": "DHCR24 alternative name (Selective Alzheimer's Disease Indicator-1); downregulated in Alzheimer's disease vulnerable neurons; DHCR24/seladin-1 may protect against oxidative stress; link between cholesterol biosynthesis and neurodegeneration",
            "LSS": "Lanosterol synthase (733aa, 21q22.3, AR); FIRST ring-cyclisation step: 2,3-oxidosqualene → lanosterol; LSS deficiency → cataracts + alopecia (OMIM 617021); 2,3-oxidosqualene elevated; lanosterol absent; normal neurodevelopment distinguishes from SLO/DHCR24",
            "Lanosterol_Chaperone": "Lanosterol disaggregates misfolded crystallin protein aggregates in lens (2015 Science paper); lens epithelium: high LSS expression → LSS deficiency → no lanosterol → crystallins aggregate → dense cataracts; lanosterol eye drops investigational",
            "MVK": "Mevalonate kinase (396aa, 12q24.11, AR); GHMP kinase; mevalonate → mevalonate-5-phosphate; upstream of farnesyl-PP and squalene; MVK deficiency → HIDS (mild) or mevalonic aciduria (severe); isoprenoids ALL deficient; statins WORSEN; canakinumab FDA-approved",
            "HIDS": "Hyper-IgD Syndrome (Hyperimmunoglobulinaemia D and periodic fever syndrome, OMIM 260920); hypomorphic MVK deficiency; I268T thermolabile allele; periodic fever + cervical lymphadenopathy + high IgD >100 IU/mL; urine mevalonic acid during attacks; canakinumab/anakinra effective",
            "Mevalonic_Aciduria": "Severe MVK deficiency (OMIM 610377); near-complete loss of MVK activity; urine mevalonic acid always elevated; psychomotor retardation + cerebellar ataxia + dysmorphic + anemia + hepatosplenomegaly; febrile crises; severe form; geranylgeraniol supplementation investigational",
            "NLRP3_Inflammasome_MVK": "Mevalonate directly activates NLRP3 inflammasome → IL-1β + IL-18 secretion; in HIDS: fever → thermolabile MVK-I268T inactivated → mevalonate surges → NLRP3 activation → IL-1β spike → fever attack (autocatalytic fever cycle); explains canakinumab (anti-IL-1β) efficacy",
            "SQLE": "Squalene epoxidase/monooxygenase (574aa, 8q24.13); FAD-dependent; squalene → 2,3-oxidosqualene; terbinafine inhibits fungal SQLE; biallelic LOF = alopecia (OMIM 617960); GOF amplification in squamous cell carcinoma; squalene elevated in LOF; NB-598 investigational inhibitor",
            "Terbinafine_SQLE": "Terbinafine (allylamine antifungal) is a competitive SQLE inhibitor (fungal SQLE >> human SQLE); in SQLE-deficient patients: terbinafine could further block residual SQLE → worsen squalene accumulation + alopecia; clinical pearl: inquire terbinafine use in unexplained alopecia",
            "Squalene_Peroxide_Toxicity": "Squalene + UV/oxygen → squalene hydroperoxides; these peroxides are pro-inflammatory at hair follicle opening; cause follicular occlusion + perifollicular inflammation → alopecia; same mechanism proposed for squalene-driven acne comedone formation",
            "Statin_Rule_Sterol_Disorders": "CRITICAL DISTINCTION: SLO (DHCR7) = statins HELP (paradox — reduce 7-DHC toxicity); Desmosterolosis (DHCR24) = statins CONTRAINDICATED (worsen); HIDS/MVK = statins CONTRAINDICATED (reduce mevalonate substrate); always check gene before prescribing statins in sterol disorders",
        }
    }
