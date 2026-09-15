"""Hereditary Telomere Biology Disorder Atlas — 8-Gene Reference
DKC1-TERC-TERT-NHP2-NOP10-WRAP53-ACD-PARN
Dyskeratosis Congenita / Hoyeraal-Hreidarsson / Pulmonary Fibrosis Spectrum
320 patients (8 x 40), seeds 2790-2797.
Endpoints: /api/hereditary-telomere-biology-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "DKC1",
        "protein": (
            "DKC1 -- Xq28 XLR -- 514aa -- Dyskerin-58kDa-Pseudouridine-Synthase-"
            "H/ACA-snoRNP-Core-Component-TERC-Stability-rRNA-Pseudouridylation-"
            "OMIM-Gene-300126-Disease-DC-X-Linked-305000"
        ),
        "locus": "Xq28",
        "protein_size": "514 aa / 58 kDa (nuclear pseudouridine synthase; H/ACA snoRNP catalytic subunit; binds H/ACA box motifs; pseudouridylates rRNA and snRNA; stabilises TERC; mutations cluster in PUA domain and pseudouridine synthase domain)",
        "inheritance": (
            "X-LINKED RECESSIVE — hemizygous males affected; heterozygous females usually unaffected carriers; "
            "DKC1 encodes dyskerin, the catalytic subunit of H/ACA small nucleolar ribonucleoprotein (snoRNP) complexes; "
            "Function 1: pseudouridine synthase — converts uridine → pseudouridine in rRNA (28S) and snRNA (U2); "
            "  Critical for ribosome biogenesis and pre-mRNA splicing; "
            "Function 2: telomerase core component — dyskerin binds the H/ACA box of TERC (telomerase RNA component); "
            "  DKC1 deficiency → TERC instability → reduced telomerase activity → short telomeres; "
            "PREVALENCE: ~1 in 1,000,000; most common form of DC (~40% of all DC); "
            "MOST SEVERE X-LINKED FORM: anticipation not typical (X-linked not AD); "
            "CLINICAL TRIAD (PATHOGNOMONIC): "
            "  Nail dystrophy (lamellar splitting, ridging, pterygium) — appears first (avg age 5-10 yr); "
            "  Reticular skin pigmentation (neck, upper chest — lacy pattern); "
            "  Oral leucoplakia (lateral tongue, buccal mucosa — premalignant); "
            "SEVERE FEATURES: "
            "  Bone marrow failure (BMF): aplastic anemia / MDS / AML — leading cause of death; "
            "  Pulmonary fibrosis (IPF-like); "
            "  Liver disease (hepatic fibrosis, portal hypertension, nodular regenerative hyperplasia); "
            "  Immunodeficiency (hypogammaglobulinaemia + NK-cell deficiency); "
            "  Learning disability (30%); "
            "  Short stature; "
            "  Premature greying; "
            "  Epiphora (lacrimal duct stenosis); "
            "HOYERAAL-HREIDARSSON (HH) FORM: severe X-linked; cerebellar hypoplasia + SCID + BMF + growth retardation + microcephaly; DKC1 mutations (especially Ala353Val, Arg158Trp); "
            "CANCER RISK: squamous cell carcinoma (oral, oropharyngeal, anogenital — 10-15% by age 50); MDS/AML; "
            "MANAGEMENT: androgens (oxymetholone/danazol — partial response); HSCT for BMF; lung transplant for IPF; "
            "KEY MUTATIONS: A386T, T49M, R158W (severe/HH), T66A, S121G; mutations in PUA domain most common"
        ),
        "disease_category": (
            "DYSKERATOSIS CONGENITA X-LINKED (DC-X / DKCX) — OMIM 305000; "
            "EARLIEST SIGN: nail dystrophy (lamellar nail changes) emerging in first decade; "
            "CLINICAL TRIAD — PATHOGNOMONIC: "
            "  1. Nail dystrophy (reticulated lacy pigmentation on nails → lamellar splitting → loss); "
            "  2. Oral leucoplakia (white patches on lateral tongue/buccal mucosa — PRE-MALIGNANT — SCC risk ~10-15%); "
            "  3. Reticular skin pigmentation (lacy brownish-red pigmentation on neck, chest); "
            "  TRIAD COMPLETE IN ~60% of males by age 20; each component present in >90% by age 30; "
            "BONE MARROW FAILURE: "
            "  Aplastic anaemia: 80-90% of DKC1-affected males by age 40 (commonest cause of death); "
            "  Pancytopenia progressive; "
            "  MDS/AML transformation risk: ~20%; "
            "TELOMERE BIOLOGY CONTEXT: "
            "  DKC1 deficiency → dyskerin cannot stabilise TERC → TERC degraded rapidly → "
            "    → reduced telomerase activity → progressive telomere shortening; "
            "  SHORT TELOMERES: <1st centile on flow-FISH telomere length testing (gold standard test); "
            "  Telomere shortening drives stem cell exhaustion → BMF; "
            "  Epithelial fragility → leucoplakia; "
            "HOYERAAL-HREIDARSSON SPECTRUM: "
            "  HH = most severe end; cerebellar hypoplasia + immunodeficiency (SCID-like) + BMF + growth retardation; "
            "  HH mutations: DKC1 R158W, A353V — most severe alleles; "
            "RIBOSOME BIOLOGY: "
            "  DKC1 loss → impaired pseudouridylation → reduced IRES-mediated translation → "
            "    → haploinsufficiency of tumour suppressors (p27, p53) — contributes to cancer susceptibility"
        ),
        "disease_pathway": (
            "TELOMERE MAINTENANCE — DKC1/DYSKERIN MECHANISM: "
            "TELOMERASE COMPLEX ASSEMBLY: "
            "  TERC (RNA component) contains an H/ACA box motif at 3' end; "
            "  Dyskerin (DKC1) + NHP2 + NOP10 + GAR1 form H/ACA snoRNP → bind TERC H/ACA box → "
            "    → stabilise TERC against degradation; "
            "  TERT (reverse transcriptase) + TERC + dyskerin complex = active telomerase; "
            "  DKC1 deficiency → TERC destabilised → telomerase activity reduced 30-50%; "
            "PSEUDOURIDYLATION: "
            "  H/ACA snoRNPs guide pseudouridylation of 28S rRNA at ~95 sites; "
            "  Pseudouridine (Ψ) = most abundant RNA modification; increases RNA thermal stability; "
            "  DKC1 deficiency → reduced rRNA pseudouridylation → ribosome dysfunction → "
            "    → impaired translation of specific IRES-containing mRNAs (p27/CDKN1B, p53/TP53, VEGF); "
            "  IRES translation impairment: selective reduction of tumour suppressors + growth inhibitors; "
            "  Paradoxical: IRES-mediated tumour suppressor reduction + telomere shortening = dual cancer risk; "
            "PROGRESSIVE TELOMERE SHORTENING: "
            "  Each cell division = ~50-200 bp lost (end-replication problem); "
            "  In DC: reduced telomerase cannot compensate → critical shortening after 2-4 decades; "
            "  Critically short telomeres → uncapped chromosome ends → p53/ATM/ATR signalling → "
            "    → replicative senescence or apoptosis → haematopoietic stem cell (HSC) exhaustion → BMF; "
            "ANTICIPATION IN TERC/TERT (NOT DKC1): "
            "  X-linked DKC1: no classic anticipation (each son inherits mother's mutant allele directly); "
            "  Telomere length inherited from parent → shorter in offspring if parent has DC mutation; "
            "  BUT: paternal imprinting/sex-specific effects complicate this for XL; "
            "TREATMENT TARGETS: "
            "  Androgens (oxymetholone, danazol): increase TERC/TERT expression → modest telomere lengthening → "
            "    → temporary haematopoietic improvement; 70% initial response; ~50% sustained at 2 yr; "
            "  HSCT: curative for BMF but pulmonary/hepatic disease progresses; "
            "    Non-myeloablative conditioning essential: full myeloablation → pulmonary complications lethal"
        ),
        "pathognomonic": (
            "DKC1 / X-LINKED DC PATHOGNOMONIC FEATURES: "
            "MUCOCUTANEOUS TRIAD (PATHOGNOMONIC): "
            "  1. NAIL DYSTROPHY — lamellar splitting of nail plate, ridging, pterygium formation, eventual nail loss; "
            "     First feature to appear (age 5-10 yr); affects fingernails > toenails; "
            "  2. ORAL LEUCOPLAKIA — white patches lateral tongue/buccal mucosa; "
            "     HIGH MALIGNANT POTENTIAL (SCC risk ~10-15% lifetime); biopsy any new lesion; "
            "  3. RETICULAR SKIN PIGMENTATION — lacy brownish-red mottled pigmentation (neck/chest); "
            "     Epidermal atrophy + telangiectasia underneath; "
            "FLOW-FISH TELOMERE LENGTH < 1st CENTILE — GOLD STANDARD DIAGNOSTIC TEST: "
            "  Flow-FISH on lymphocytes + granulocytes; "
            "  Telomere length <1st centile for age STRONGLY suggests telomere biology disorder; "
            "  DKC1 males: usually <1st centile; females carriers: often 1-10th centile; "
            "LACRIMAL DUCT STENOSIS (EPIPHORA): "
            "  Tearing without infection — pathognomonic of mucosal membrane involvement in DC; "
            "  70% of DKC1 males affected; "
            "KEY DDx: "
            "  DKC1 vs TERC/TERT: all DC forms; distinguish by flow-FISH + sequencing; "
            "  DKC1 vs Fanconi Anaemia: FA chromosomal breakage (DEB/MMC positive); DC = DEB/MMC NEGATIVE; "
            "  DKC1 vs aplastic anaemia (acquired): DC = telomere <1st centile + mucocutaneous features; "
            "  DKC1 HH vs other immunodeficiency: HH = cerebellar hypoplasia + BMF + SCID-like; check flow-FISH; "
            "ALLOGENEIC HSCT CAVEAT: "
            "  TBI ABSOLUTELY CONTRAINDICATED in DC (lung/liver too fragile — fatal pneumonitis); "
            "  Non-myeloablative (reduced-intensity conditioning RIC) only; "
            "  Cyclophosphamide high dose: also toxic in DC — use fludarabine-based RIC instead"
        ),
        "key_facts": [
            "DKC1-NAIL-DYSTROPHY-LEUCOPLAKIA-SKIN-PIGMENTATION-TRIAD-PATHOGNOMONIC",
            "DKC1-FLOW-FISH-LT1ST-CENTILE-GOLD-STANDARD",
            "DKC1-X-LINKED-40PCT-ALL-DC",
            "DKC1-TERC-STABILITY-H/ACA-snoRNP",
            "DKC1-BMF-APLASTIC-ANAEMIA-LEADING-CAUSE-DEATH",
            "DKC1-ANDROGENS-DANAZOL-PARTIAL-RESPONSE",
            "DKC1-TBI-ABSOLUTELY-CI-RIC-ONLY",
            "DKC1-SCC-ORAL-10-15PCT-RISK",
        ],
        "treatment": (
            "DKC1 X-linked DC: "
            "Haematopoietic: Oxymetholone/danazol (androgens) — increase TERC/TERT expression → modest BMF improvement; "
            "  response 70% initial; sustained ~50% at 2 yr; liver toxicity monitoring (LFT, hepatic USS); "
            "Eltrombopag: TPO-RA for thrombocytopenia (investigational in DC BMF); "
            "HSCT for BMF: non-myeloablative (RIC: fludarabine-based); curative for BMF; "
            "  TBI ABSOLUTELY CI; full myeloablative ABSOLUTELY CI (fatal pulmonary complications); "
            "Oral leucoplakia: retinoids (isotretinoin) — evidence limited; regular biopsy (q6-12m); "
            "Pulmonary fibrosis: pirfenidone/nintedanib (anti-fibrotic); lung transplant (post-HSCT); "
            "Liver: hepatology surveillance; avoid alcohol + hepatotoxins; "
            "Ophthalmology: lacrimal duct dilation for epiphora; "
            "Cancer screening: annual oral exam + endoscopy from age 30; "
            "Genetic counselling: X-linked recessive; maternal carriers screened (flow-FISH, DKC1 sequencing)"
        ),
        "seed_base": 2790,
        "n_patients": 40,
    },
    {
        "gene": "TERC",
        "protein": (
            "TERC -- 3q26.2 AD -- 451nt RNA -- Telomerase-RNA-Component-H/ACA-Box-"
            "CR4-CR5-Pseudoknot-Dyskerin-Binding-TERC-Stability-Template-"
            "OMIM-Gene-602322-Disease-DC-AD-Type2-127550-AP-614743"
        ),
        "locus": "3q26.2",
        "protein_size": "451 nt RNA (non-coding; contains: template region nt 46-56; pseudoknot domain; CR4/CR5 domain; H/ACA box at 3' end binding dyskerin-NHP2-NOP10; vertebrate telomerase RNA)",
        "inheritance": (
            "AUTOSOMAL DOMINANT — heterozygous pathogenic variants; haploinsufficiency; "
            "TERC encodes the RNA component of human telomerase (hTR/hTERC, 451 nt); "
            "Critical functional domains: "
            "  Template region (nt 46-56): provides 5'-CUAACCCUAAC-3' template → added to chromosome 3' end; "
            "  Pseudoknot/CR4-CR5 domain: activates TERT catalysis; TERT binds here; "
            "  H/ACA box (3' end): dyskerin (DKC1) + NHP2 + NOP10 bind → TERC stability; "
            "HAPLOINSUFFICIENCY: one functional TERC allele insufficient → reduced telomerase → telomere shortening; "
            "PENETRANCE: incomplete (~50-60% by age 50 for DC triad); increases in successive generations; "
            "ANTICIPATION (GENETIC): "
            "  Each generation inherits shorter telomeres from affected parent; "
            "  2nd generation: telomeres shorter → more severe clinical phenotype + earlier onset; "
            "  3rd generation: often most severe (Revesz, HH spectrum possible); "
            "  ANTICIPATION = hallmark of AD telomere biology disorders (TERC, TERT, TINF2-AD); "
            "ASSOCIATED PHENOTYPES: "
            "  Classic DC (complete triad + BMF): ~30% of AD TERC families; "
            "  Aplastic anaemia (without mucocutaneous triad): presenting phenotype in many adults; "
            "  IPF (idiopathic pulmonary fibrosis): monoallelic TERC variants in ~1-3% of familial IPF; "
            "  Liver cirrhosis (without obvious DC features); "
            "  Acute myeloid leukaemia (rare presenting feature); "
            "PREVALENCE: 2nd commonest DC gene; ~10-15% of all DC families"
        ),
        "disease_category": (
            "DYSKERATOSIS CONGENITA AUTOSOMAL DOMINANT TYPE 2 (DC-AD2) — OMIM 127550; "
            "APLASIA-PANCYTOPENIA (AP) — OMIM 614743; "
            "IPF ASSOCIATION — familial IPF (OMIM 614742); "
            "CLINICAL SPECTRUM — HIGHLY VARIABLE (haploinsufficiency phenotype): "
            "  Severe: classic DC triad + early BMF in childhood (anticipation > 2 generations); "
            "  Moderate: aplastic anaemia without full DC triad in 3rd-4th decade; "
            "  Mild: isolated IPF or liver disease in 5th-6th decade; "
            "  Very mild: only flow-FISH abnormality, no clinical disease (incomplete penetrance); "
            "GENETIC ANTICIPATION — CARDINAL FEATURE: "
            "  First carrier generation: often IPF or mild haematological finding; "
            "  Second generation: aplastic anaemia or DC in adulthood; "
            "  Third generation: severe DC or HH phenotype in childhood; "
            "  PATTERN: each generation presents earlier and more severely; "
            "  Molecular basis: telomere length inherited from parent; shorter parent → critically short in offspring; "
            "IPF CONTEXT: "
            "  TERC (and TERT) variants account for ~1-3% of familial IPF; "
            "  Adult carriers presenting with 'IPF': flow-FISH often <10th centile; "
            "  IPF-directed treatment (pirfenidone/nintedanib) slows progression but does not reverse; "
            "  Lung transplant: only definitive option; TERC variant should be identified before transplant "
            "    (donor-to-recipient telomere biology matching considerations); "
            "APLASTIC ANAEMIA: "
            "  Some TERC mutation carriers present with AA without obvious DC triad; "
            "  Flow-FISH <1st-10th centile in AA + family history → screen TERC/TERT"
        ),
        "disease_pathway": (
            "TELOMERASE RNA TEMPLATE MECHANISM: "
            "TELOMERASE CATALYTIC CYCLE: "
            "  TERC provides the template sequence (5'-CUAACCCUAAC-3') for reverse transcription; "
            "  TERT binds CR4-CR5 domain of TERC → TERT reverse transcriptase adds TTGGGG repeats to chromosome 3' end; "
            "  Each cycle: TERT extends 3' end by 6 nt → translocates → repeat; "
            "  One telomerase enzyme adds ~50-100 nt per S phase per chromosome end; "
            "TERC STABILITY — H/ACA BOX: "
            "  3' H/ACA motif of TERC → dyskerin (DKC1) + NHP2 + NOP10 bind → stabilise TERC against PAPD5/PARN degradation; "
            "  PARN deadenylase: degrades TERC 3' oligoadenylated tail → promotes TERC turnover; "
            "  ZCCHC8/PAPD5 axis: polyadenylates TERC → targets for degradation; "
            "  Balance: TERC synthesis rate vs degradation determines steady-state TERC level; "
            "  DKC1 or TERC mutations → dyskerin binding impaired → TERC degraded faster → reduced telomerase; "
            "HAPLOINSUFFICIENCY QUANTITATIVE EFFECT: "
            "  One TERC allele → ~50% TERC level → ~50% telomerase activity; "
            "  In rapidly cycling HSCs: 50% activity cannot maintain telomere length over decades; "
            "  After 40-50 yr: critical telomere shortening → HSC exhaustion → aplastic anaemia; "
            "  Lung epithelial progenitors also critically short → alveolar epithelial cell apoptosis → IPF; "
            "MOLECULAR ANTICIPATION: "
            "  Telomere length is partially heritable (parent-to-offspring transmission); "
            "  If father (AD carrier) has short telomeres → offspring telomeres start shorter; "
            "  Additionally, TERC haploinsufficiency in offspring from conception → cumulative shortening; "
            "  By 3rd generation: telomeres critically short at birth → HH/severe DC"
        ),
        "pathognomonic": (
            "TERC PATHOGNOMONIC FEATURES: "
            "GENETIC ANTICIPATION (HALLMARK): "
            "  3-generation pedigrees: grandparent (IPF/mild) → parent (AA/moderate DC) → child (severe DC/HH); "
            "  This pattern IS PATHOGNOMONIC of autosomal dominant telomere biology disorder; "
            "  Absence of anticipation: consider AR condition or sporadic DC; "
            "FLOW-FISH TELOMERE LENGTH <10th CENTILE: "
            "  Flow-FISH on granulocytes + lymphocytes; <10th centile for age suggests telomere biology disorder; "
            "  TERC variants: granulocytes typically shortest; "
            "IPF WITHOUT OBVIOUS DC TRIAD: "
            "  Adult presenting with 'cryptogenic' IPF + flow-FISH <10th centile → screen TERC + TERT; "
            "  TERC/TERT-associated IPF: usual interstitial pneumonia (UIP) pattern on HRCT/histology; "
            "  Family history of any of: AA, DC, liver cirrhosis, IPF → TERC/TERT panel; "
            "KEY DDx: "
            "  TERC vs TERT: both AD; both cause anticipation; telomere length short; "
            "    Distinguish: sequencing only; clinical phenotype overlaps; "
            "  TERC vs DKC1: DKC1 = XL; TERC = AD; family history distinguishes; "
            "  TERC vs acquired AA: DC/TERC → <1st-10th centile flow-FISH; acquired AA → telomere length variable; "
            "PIRFENIDONE/NINTEDANIB IN TERC IPF: "
            "  Anti-fibrotic benefit: slows FVC decline by ~50% vs placebo; "
            "  DOES NOT reverse underlying telomere shortening; DOES NOT cure; "
            "  Lung transplant only definitive (monitor for extrapulmonary DC features post-transplant)"
        ),
        "key_facts": [
            "TERC-GENETIC-ANTICIPATION-HALLMARK-AD",
            "TERC-HAPLOINSUFFICIENCY-50PCT-TELOMERASE",
            "TERC-IPF-1-3PCT-FAMILIAL-IPF",
            "TERC-FLOW-FISH-LT10TH-CENTILE",
            "TERC-451NT-RNA-TEMPLATE-CUAACCCUAAC",
            "TERC-H/ACA-BOX-DYSKERIN-STABILITY",
            "TERC-APLASTIC-ANAEMIA-WITHOUT-TRIAD",
            "TERC-VARIABLE-PENETRANCE-INCOMPLETE",
        ],
        "treatment": (
            "TERC AD DC / aplastic anaemia / IPF: "
            "BMF: Androgens (danazol/oxymetholone) — increase TERC/TERT expression; 70% response; "
            "  Eltrombopag: TPO-RA for thrombocytopenia; "
            "HSCT: RIC (non-myeloablative) for BMF; TBI ABSOLUTELY CI; "
            "IPF: Pirfenidone (anti-fibrotic) or Nintedanib — slow progression; "
            "  Lung transplant: only curative option for IPF; screen family before donating; "
            "  N-acetylcysteine: no longer first-line (PANTHER trial — harm in UIP); "
            "Genetic counselling: AD; 50% offspring risk; flow-FISH all family members; "
            "Anticipation counselling: earlier/more severe disease in offspring expected; "
            "Surveillance: annual CBC + LFTs + PFTs + 6MWT; echocardiogram q2yr (PH); "
            "Avoid: smoking (accelerates lung fibrosis); NSAIDs (gastric bleeding if thrombocytopenic)"
        ),
        "seed_base": 2791,
        "n_patients": 40,
    },
    {
        "gene": "TERT",
        "protein": (
            "TERT -- 5p15.33 AD/AR -- 1132aa -- Telomerase-Reverse-Transcriptase-"
            "127kDa-TRBD-Palm-Fingers-Thumb-Domains-Active-Site-Asp-712-Asp-868-"
            "OMIM-Gene-187270-Disease-DC-AD-613989-HH-AR-615190-IPF-614742"
        ),
        "locus": "5p15.33",
        "protein_size": "1132 aa / 127 kDa (TERT; reverse transcriptase with N-terminal TRBD (TERC-binding), palm/finger/thumb catalytic domains; D712 and D868 = active site aspartates; interacts with dyskerin-TERC complex; catalyses TTGGGG addition)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (haploinsufficiency) OR AUTOSOMAL RECESSIVE (biallelic — more severe): "
            "AD TERT: "
            "  Haploinsufficiency → 50% telomerase activity → progressive telomere shortening; "
            "  Clinical anticipation (same mechanism as TERC); "
            "  Phenotypes: IPF (commonest adult presentation), aplastic anaemia, DC spectrum; "
            "  IPF: TERT is #1 identified gene in familial IPF (~2-5% familial IPF); "
            "  TERT variants more prevalent than TERC in IPF context; "
            "AR TERT (biallelic): "
            "  More severe than monoallelic; "
            "  HH spectrum: cerebellar hypoplasia + immunodeficiency + BMF + growth retardation; "
            "  Early-onset DC; "
            "  Two hypomorphic alleles (compound heterozygous) may give intermediate severity; "
            "PREVALENCE: ~10% of DC families (AD + AR combined); #1 gene in familial IPF; "
            "KEY MUTATIONS: "
            "  AD: P721R (most common European); R631Q; H412Y; "
            "  AR: severe biallelic LOF; "
            "GENOTYPE-PHENOTYPE: "
            "  AD TERT + IPF: lung phenotype predominant; BMF later/milder; "
            "  AD TERT + DC: mucocutaneous triad + BMF; "
            "  AR TERT biallelic: HH/severe DC in childhood; "
            "  Telomere length inversely correlates with severity"
        ),
        "disease_category": (
            "DC AUTOSOMAL DOMINANT TYPE 3 (DC-AD3) — OMIM 613989; "
            "HH SYNDROME AR — OMIM 615190; "
            "FAMILIAL IPF — OMIM 614742; "
            "SPECTRUM: "
            "  Monoallelic TERT (AD): IPF (commonest), aplastic anaemia, mild DC (triad less complete than DKC1); "
            "  Biallelic TERT (AR): HH/severe DC; "
            "IPF — CLINICAL LEADER: "
            "  TERT = most common identified genetic cause of familial IPF; "
            "  ~2-5% of familial IPF + ~0.5-1% of sporadic IPF; "
            "  HRCT: UIP pattern (basal-predominant honeycombing + traction bronchiectasis); "
            "  Histology: UIP; "
            "  TELOMERE SCREENING in IPF: flow-FISH <10th centile in 25-30% of familial IPF; "
            "  TERT IPF: response to pirfenidone/nintedanib similar to sporadic IPF (slows FVC decline); "
            "APLASTIC ANAEMIA: "
            "  TERT variants in 3-4% of acquired/idiopathic AA; "
            "  Immunosuppressive therapy (ATG + cyclosporin) response REDUCED in DC-AA vs acquired AA; "
            "  Flow-FISH before AA treatment; telomere biology AA → androgens first; HSCT with RIC; "
            "LIVER: "
            "  Hepatic cirrhosis/portal hypertension reported in TERT families (telomere-short hepatocytes → regeneration failure); "
            "  Nodular regenerative hyperplasia"
        ),
        "disease_pathway": (
            "TERT CATALYTIC MECHANISM: "
            "REVERSE TRANSCRIPTION: "
            "  TERT + TERC = minimal telomerase; dyskerin complex required for full activity in vivo; "
            "  Template: TERC 3' end nt 46-56 (CUAACCCUAAC) → TERT reads 3' to 5' → adds 5'-TTGGGG-3' repeats; "
            "  TERT mechanism: 3' OH of telomere → base-pairs template nt 46-56 → RT adds TTGGGG; "
            "    → translocation → next addition; processivity regulated by TPP1 (ACD)/POT1 interaction; "
            "ACTIVE SITE: "
            "  Aspartates D712 + D868 in TRBD/palm domain → chelate Mg²⁺ → catalyse phosphodiester bond; "
            "  Mutations in active site = severe loss of function; "
            "TELOMERASE REGULATION: "
            "  TERT expression: repressed in most somatic cells (TERT promoter methylated/silenced); "
            "  Active in: stem cells, germ cells, lymphocytes, some progenitor cells; "
            "  Cancer: TERT promoter mutations (C228T, C250T) = reactivation → ~70% of all cancers; "
            "  Androgens (danazol): increase TERT transcription (via androgen response element in TERT promoter); "
            "    → clinical rationale for androgen therapy in DC (also increases TERC); "
            "TELOMERE LENGTH REGULATION: "
            "  TERT haploinsufficiency → ~50% telomerase → insufficient to maintain telomere length in dividing cells; "
            "  HSCs divide more than most cells → telomere shortening accelerated in HSC compartment; "
            "  Critical telomere shortening → p53/p21 activation → proliferative arrest → HSC pool exhaustion; "
            "  LUNG: type II pneumocyte turnover → same mechanism → alveolar epithelial cell senescence → IPF"
        ),
        "pathognomonic": (
            "TERT PATHOGNOMONIC FEATURES: "
            "FAMILIAL IPF WITH GENETIC ANTICIPATION: "
            "  Family pedigree: parent/grandparent had IPF or AA → index case IPF earlier + more severe; "
            "  Flow-FISH <10th centile for age in IPF patient = TERT/TERC screening MANDATORY; "
            "  TERT/TERC together account for ~80% of telomere biology disorder-associated IPF; "
            "ANDROGEN-RESPONSIVE HAEMATOPOIETIC FAILURE: "
            "  Response to danazol with haematological improvement + flow-FISH lengthening; "
            "  First published 2016 NEJM (Townsley et al.): danazol → telomere lengthening in DC; "
            "  Response distinguishes telomere biology disorder from Fanconi/acquired AA; "
            "TELOMERE SHORTENING BY FLOW-FISH: "
            "  Granulocytes + lymphocytes <10th centile; TERT (like TERC) → granulocyte telomeres shortest; "
            "KEY DDx: "
            "  TERT-AD vs TERC-AD: phenotypic overlap; distinguish by sequencing; TERT more common in IPF; "
            "  TERT-AR (biallelic HH) vs DKC1-HH: biallelic TERT = AR; DKC1 = XL; sex of index helps; "
            "  TERT-associated AA vs acquired AA: TERT → <10th centile flow-FISH; acquired → usually >10th; "
            "  IMMUNOSUPPRESSIVE THERAPY IN TERT-AA: response poor (underlying telomere shortening persists); "
            "    → prefer androgens first; HSCT with RIC if severe/refractory"
        ),
        "key_facts": [
            "TERT-IPF-NUMBER-1-IDENTIFIED-GENETIC-CAUSE-FAMILIAL-IPF",
            "TERT-AD-HAPLOINSUFFICIENCY-DC-AA-IPF",
            "TERT-AR-BIALLELIC-HH-SEVERE",
            "TERT-ANDROGEN-RESPONSE-ELEMENT-DANAZOL",
            "TERT-FLOW-FISH-LT10TH-CENTILE",
            "TERT-GENETIC-ANTICIPATION-AD",
            "TERT-P721R-MOST-COMMON-EUROPEAN-AD",
            "TERT-ACTIVE-SITE-D712-D868",
        ],
        "treatment": (
            "TERT DC / IPF / aplastic anaemia: "
            "BMF: Androgens (danazol 400-800 mg/day) — increase TERT + TERC transcription; "
            "  Townsley 2016 NEJM: danazol → telomere lengthening + haematological response in DC; "
            "  Monitor LFTs; hepatocellular carcinoma rare but reported with long-term androgens; "
            "HSCT for BMF: RIC (fludarabine-based); TBI ABSOLUTELY CI; "
            "IPF: Pirfenidone/nintedanib — anti-fibrotic; slows FVC decline; "
            "  Lung transplant — only cure for IPF; transplant evaluation when FVC <50% or DLCO <40%; "
            "  Screen transplant donor family for TERT mutations (avoid donor with shared telomere biology risk); "
            "Immunosuppressive therapy (ATG+cyclosporin) for TERT-AA: reduced response vs acquired AA; "
            "  Try androgens first; HSCT if androgens fail/severe; "
            "Genetic counselling: AD = 50% risk each child; AR = 25% risk; "
            "Predictive testing: flow-FISH for family members; "
            "Cancer surveillance: annual CBC + PFTs + 6MWT + CT chest q1-2yr if IPF"
        ),
        "seed_base": 2792,
        "n_patients": 40,
    },
    {
        "gene": "NHP2",
        "protein": (
            "NHP2 -- 5q35.3 AR -- 153aa -- Non-Histone-Protein-2-H/ACA-snoRNP-"
            "16kDa-RNA-Binding-L7Ae-Motif-TERC-Stabilisation-Complex-"
            "OMIM-Gene-606470-Disease-DC-AR-Type5-613987"
        ),
        "locus": "5q35.3",
        "protein_size": "153 aa / 16 kDa (NHP2; H/ACA snoRNP core component; RNA-binding L7Ae motif; forms 4-protein complex with dyskerin-NOP10-GAR1; binds H/ACA box of TERC; pseudouridylation complex)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "NHP2 encodes a core component of H/ACA small nucleolar ribonucleoprotein (snoRNP) complexes; "
            "H/ACA SNORIBONUCLEOPROTEIN COMPLEX: "
            "  4 proteins: dyskerin (DKC1) + NHP2 + NOP10 + GAR1; "
            "  DKC1 = catalytic (pseudouridine synthase); "
            "  NHP2 = RNA-binding scaffold (L7Ae motif); binds kink-turn motif of H/ACA RNA; "
            "  NOP10 = structural stabiliser of NHP2-DKC1 interaction; "
            "  GAR1 = RNA substrate binding/turnover; "
            "  Together: guide pseudouridylation of rRNA + snRNA + TERC stabilisation; "
            "NHP2 DEFICIENCY: "
            "  Complex cannot assemble → TERC destabilised → reduced telomerase → short telomeres; "
            "  Pseudouridylation also impaired (rRNA); "
            "CLINICAL FEATURES: "
            "  DC clinical triad (nail dystrophy + leucoplakia + skin pigmentation); "
            "  BMF (aplastic anaemia); "
            "  PHENOTYPE SIMILAR TO DKC1/AR TERT but AR inheritance; "
            "PREVALENCE: rare (<100 reported cases); "
            "KNOWN MUTATIONS: p.Val126Met (recurrent); homozygous or compound heterozygous; "
            "TELOMERE LENGTH: flow-FISH <1st centile"
        ),
        "disease_category": (
            "DYSKERATOSIS CONGENITA AUTOSOMAL RECESSIVE TYPE 5 (DC-AR5) — OMIM 613987; "
            "CLINICAL FEATURES: "
            "  DC triad: nail dystrophy + oral leucoplakia + reticular pigmentation; "
            "  BMF: aplastic anaemia / pancytopenia; "
            "  No additional CNS features (unlike HH); "
            "  Short stature; "
            "  Immunodeficiency (variable); "
            "TELOMERE BIOLOGY: "
            "  NHP2 biallelic LOF → H/ACA snoRNP impaired → TERC destabilised → "
            "    → reduced telomerase → short telomeres; "
            "  Flow-FISH <1st centile in affected individuals; "
            "    Carriers (heterozygous): telomere length often normal or 1-10th centile; "
            "SEVERITY: intermediate; typically childhood-onset DC; BMF in 2nd-3rd decade; "
            "DDx from DKC1: requires sequencing (clinical phenotype similar); "
            "  DKC1 = XL; NHP2 = AR (both males and females equally affected); "
            "  Family history distinguishes (consanguinity in some NHP2 families); "
            "RESPONSE TO TREATMENT: "
            "  Androgens (danazol): some response; same mechanism (androgen increases TERT/TERC expression); "
            "  HSCT for severe BMF: RIC; TBI CI (same principle as all DC forms)"
        ),
        "disease_pathway": (
            "NHP2 — H/ACA snoRNP ASSEMBLY PATHWAY: "
            "SNOSNP BIOGENESIS: "
            "  H/ACA snoRNA (including TERC): transcribed in nucleus; "
            "  Co-transcriptional assembly: NHP2 + NOP10 bind newly synthesised H/ACA snoRNA; "
            "  Then DKC1 joins → core trimer (DKC1-NHP2-NOP10) forms; "
            "  Finally GAR1 replaces NUFIP2/SHQ1 (assembly factors) → mature functional H/ACA RNP; "
            "  NUFIP2 (formerly RYBP): assembly chaperone; SHQ1: protects nascent complex; "
            "NHP2 BINDING MODE: "
            "  NHP2 L7Ae domain binds kink-turn (K-turn) motif in H/ACA box stem-loop; "
            "  K-turn: 3-nt loop + two flanking helices → bent RNA structure → NHP2 binds bent angle; "
            "  This interaction nucleates assembly of entire H/ACA snoRNP; "
            "NHP2 DEFICIENCY CONSEQUENCES: "
            "  H/ACA snoRNP cannot assemble → no pseudouridylation of rRNA at 95+ sites; "
            "  TERC: H/ACA box unprotected → PAPD5 polyadenylates → ZCCHC8 targets for degradation → "
            "    → TERC reduced → telomerase activity reduced → telomere shortening; "
            "  Both ribosome function (pseudouridylation) and telomere maintenance (TERC stability) impaired; "
            "COMPARISON WITH DKC1/NOP10 DEFICIENCY: "
            "  All three (DKC1, NHP2, NOP10) are core H/ACA snoRNP components → same final pathway; "
            "  Severity: DKC1 (XL, XLR) > NHP2 (AR) ≈ NOP10 (AR) in general; "
            "  All respond to same interventions (androgens, RIC-HSCT)"
        ),
        "pathognomonic": (
            "NHP2 PATHOGNOMONIC FEATURES: "
            "DC TRIAD IN AR PATTERN: "
            "  Males and females equally affected (AR — distinguishes from DKC1 X-linked); "
            "  Consanguinity in some families (AR); "
            "FLOW-FISH <1ST CENTILE: "
            "  Short telomeres confirm telomere biology disorder; "
            "  Sequencing panel needed to identify NHP2 specifically; "
            "CLINICAL DDx: "
            "  NHP2 vs DKC1: sex ratio (NHP2 = equal sexes; DKC1 = predominantly males); "
            "  NHP2 vs TERT AR: phenotypic overlap; distinguish by panel sequencing; "
            "  NHP2 vs TERC AD: NHP2 = AR (both alleles needed); TERC = AD (one allele); "
            "  NHP2 vs NOP10: essentially identical phenotype; sequence panel; "
            "MUTATION HOT-SPOT: "
            "  Val126Met: most commonly reported pathogenic variant; "
            "  Homozygous Val126Met in several reported families; "
            "  Segregation analysis key for VUS; "
            "ANDROGEN RESPONSE: "
            "  Danazol response: same mechanism as other DC forms; "
            "  Documented haematological improvement in NHP2 DC; "
            "TBI CONTRAINDICATED: "
            "  Same principle applies: all DC forms avoid TBI → RIC conditioning only"
        ),
        "key_facts": [
            "NHP2-H/ACA-SNOSNP-CORE-NHP2-NOP10-DKC1-GAR1",
            "NHP2-AR-BIALLELIC-DC-TYPE-5",
            "NHP2-FLOW-FISH-LT1ST-CENTILE",
            "NHP2-VAL126MET-RECURRENT-MUTATION",
            "NHP2-L7AE-MOTIF-K-TURN-BINDING",
            "NHP2-TERC-STABILITY-IMPAIRED",
            "NHP2-EQUAL-SEX-RATIO-DDX-DKC1",
            "NHP2-ANDROGEN-RESPONSE-DANAZOL",
        ],
        "treatment": (
            "NHP2 AR-DC: "
            "Androgens (danazol/oxymetholone): BMF management; 70% initial response; "
            "HSCT: RIC conditioning; TBI CI; "
            "Surveillance: annual CBC, LFTs, PFTs; "
            "Cancer screening: oral leucoplakia biopsy; "
            "Genetic counselling: AR; 25% sibling risk; "
            "Carrier testing: parents; "
            "Prenatal testing available if mutations identified"
        ),
        "seed_base": 2793,
        "n_patients": 40,
    },
    {
        "gene": "NOP10",
        "protein": (
            "NOP10 -- 15q14 AR -- 64aa -- Nucleolar-Protein-10-H/ACA-snoRNP-"
            "7kDa-NHP2-DKC1-Stabiliser-TERC-Complex-"
            "OMIM-Gene-606471-Disease-DC-AR-Type4-224230"
        ),
        "locus": "15q14",
        "protein_size": "64 aa / 7 kDa (NOP10; smallest H/ACA snoRNP core protein; stabilises NHP2-DKC1 interaction within the H/ACA complex; essential for TERC stability and rRNA pseudouridylation)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "NOP10 encodes a small structural protein (64 aa) essential for H/ACA snoRNP function; "
            "FUNCTION: "
            "  NOP10 bridges NHP2 and DKC1 within the H/ACA snoRNP complex; "
            "  Without NOP10: NHP2-DKC1 interaction weakened → complex unstable → "
            "    → TERC not protected → TERC degraded → reduced telomerase → short telomeres; "
            "  Also required for rRNA pseudouridylation (like NHP2); "
            "PREVALENCE: ULTRA-RARE — fewer than 20 reported cases; "
            "KNOWN MUTATION: p.Arg34Trp (homozygous — founder variant in affected families); "
            "  Arg34 in NOP10 makes direct contacts with NHP2 and DKC1 in the complex crystal structure; "
            "  Arg34Trp → disrupts these interactions → complex destabilisation; "
            "CLINICAL FEATURES: "
            "  DC triad: nail dystrophy, oral leucoplakia, reticular pigmentation; "
            "  BMF; "
            "  Short stature; "
            "  Learning disability (variable); "
            "  No additional HH-type features in reported cases; "
            "TELOMERE LENGTH: flow-FISH <1st centile"
        ),
        "disease_category": (
            "DYSKERATOSIS CONGENITA AUTOSOMAL RECESSIVE TYPE 4 (DC-AR4) — OMIM 224230; "
            "ULTRA-ORPHAN: <20 reported cases worldwide; "
            "CLINICAL FEATURES (same spectrum as NHP2 and DKC1 but AR): "
            "  Nail dystrophy; "
            "  Oral leucoplakia (premalignant); "
            "  Reticular skin pigmentation; "
            "  Aplastic anaemia / BMF; "
            "FOUNDER VARIANT: "
            "  Arg34Trp: reported in consanguineous families; "
            "  Structural modelling: R34 makes direct contacts with NHP2 and DKC1 in crystal structures; "
            "  R34W: disrupts both NHP2 and DKC1 contacts → complex cannot assemble properly; "
            "MOLECULAR DIAGNOSIS: "
            "  Panel sequencing for all DC genes (including NOP10) required; "
            "  Clinical phenotype: indistinguishable from NHP2 or DKC1 without sequencing; "
            "  Flow-FISH: telomere shortening documented in all reported cases; "
            "MANAGEMENT: "
            "  Same as other AR DC forms; "
            "  Androgens; RIC-HSCT for severe BMF; "
            "  Cancer surveillance (leucoplakia SCC risk)"
        ),
        "disease_pathway": (
            "NOP10 — STRUCTURAL ROLE IN H/ACA snoRNP: "
            "NOP10 POSITION IN COMPLEX: "
            "  Crystal structure (PDB: 2HVY, 3U28): NOP10 sits between NHP2 and DKC1; "
            "  Three-way contact: NOP10 N-terminus → DKC1 TruD domain; "
            "    NOP10 C-terminus → NHP2 hydrophobic core; "
            "  This creates stable ternary complex required for H/ACA snoRNA binding and activity; "
            "NOP10 DEFICIENCY: "
            "  R34W mutation → reduced affinity for NHP2 → ternary complex less stable → "
            "    → TERC H/ACA box binding impaired → TERC degraded → telomerase down; "
            "COMPARISON WITH OTHER H/ACA PROTEINS: "
            "  DKC1 (catalytic): most severe when lost (X-linked); "
            "  NHP2 (RNA-binding scaffold): intermediate severity; "
            "  NOP10 (structural bridge): also intermediate; clinical phenotype similar to NHP2; "
            "  GAR1: substrate positioning; GAR1 mutations cause DC-AR (rare, distinct entry); "
            "COMMON FINAL PATHWAY: "
            "  All four H/ACA proteins essential → any one lost → complex non-functional → "
            "    → TERC instability → reduced telomerase → progressive telomere shortening → DC/BMF; "
            "  Treatment targeting TERC/TERT expression (androgens) benefits all H/ACA protein deficiencies"
        ),
        "pathognomonic": (
            "NOP10 PATHOGNOMONIC FEATURES: "
            "ULTRA-RARE AR DC — SEQUENCING REQUIRED: "
            "  Clinical phenotype: DC triad + BMF; indistinguishable from NHP2/DKC1/TERC without sequencing; "
            "  NOP10 identified only by comprehensive panel or exome sequencing; "
            "ARG34TRP FOUNDER VARIANT: "
            "  Structural disruption: R34W = loss of contact with both NHP2 and DKC1; "
            "  Homozygous in consanguineous DC families; "
            "  Molecular evidence: TERC levels reduced in patient lymphoblasts; "
            "FLOW-FISH: telomere shortening (<1st centile); "
            "CLINICAL DDx: "
            "  NOP10 vs NHP2: essentially identical; both AR; sequencing only; "
            "  NOP10 vs DKC1: DKC1 = XL (males predominant); NOP10 = AR (equal sex); "
            "  NOP10 vs TERC: TERC = AD; NOP10 = AR; family history + sequencing; "
            "MANAGEMENT SAME AS OTHER DC: "
            "  Androgens; RIC-HSCT; TBI CI; cancer surveillance"
        ),
        "key_facts": [
            "NOP10-H/ACA-snoRNP-BRIDGE-NHP2-DKC1",
            "NOP10-AR-DC-TYPE-4-ULTRA-RARE",
            "NOP10-ARG34TRP-FOUNDER-VARIANT",
            "NOP10-64AA-SMALLEST-SNOSNP-PROTEIN",
            "NOP10-FLOW-FISH-LT1ST-CENTILE",
            "NOP10-TERC-STABILITY-IMPAIRED",
            "NOP10-CONSANGUINEOUS-FAMILIES",
            "NOP10-PANEL-SEQUENCING-REQUIRED",
        ],
        "treatment": (
            "NOP10 AR-DC: "
            "Androgens (danazol/oxymetholone): haematological support; "
            "HSCT: RIC conditioning for severe BMF; TBI CI; "
            "Oral leucoplakia: retinoid, biopsy q6-12m; "
            "Genetic counselling: AR; 25% sibling risk; consanguinity assessment; "
            "Annual surveillance: CBC, LFTs, PFTs"
        ),
        "seed_base": 2794,
        "n_patients": 40,
    },
    {
        "gene": "WRAP53",
        "protein": (
            "WRAP53 -- 17p13.1 AR -- 548aa -- WD-40-Repeat-Protein-53-TCAB1-"
            "62kDa-Cajal-Body-Trafficking-TERC-Localization-"
            "OMIM-Gene-612661-Disease-DC-AR-Type6-613988"
        ),
        "locus": "17p13.1",
        "protein_size": "548 aa / 62 kDa (WRAP53/TCAB1; WD40 repeat protein; guides TERC to Cajal bodies via CAB-box; Cajal bodies = nuclear assembly sites for telomerase; WRAP53 recruits telomerase to telomeres during S phase)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "WRAP53 (also called TCAB1 = Telomerase Cajal Body protein 1) encodes a WD40 domain protein; "
            "FUNCTION — TERC SUBCELLULAR LOCALISATION: "
            "  WRAP53 binds the CAB-box (Cajal body box: UGAG) at the 3' end of TERC; "
            "  WRAP53 localises TERC to Cajal bodies (nuclear organelles); "
            "  Cajal bodies: assembly hubs where telomerase complex is fully assembled and matured; "
            "  Without WRAP53: TERC cannot reach Cajal bodies → telomerase not recruited to telomeres → "
            "    → telomere shortening despite normal TERC levels and normal TERT levels; "
            "  IMPORTANT: TERC levels NORMAL in WRAP53-deficient cells; TERT levels NORMAL; "
            "    Telomerase activity in vitro: NORMAL (components present); "
            "    Telomerase localisation to telomeres in S phase: IMPAIRED; "
            "    → Functional defect despite biochemically normal components; "
            "CLINICAL FEATURES: "
            "  DC triad (nail dystrophy + leucoplakia + pigmentation); "
            "  BMF; "
            "  Usually childhood onset; "
            "KNOWN MUTATIONS: F164L (most common); R398W; compound heterozygous; "
            "PREVALENCE: rare; ~50 reported cases"
        ),
        "disease_category": (
            "DYSKERATOSIS CONGENITA AUTOSOMAL RECESSIVE TYPE 6 (DC-AR6) — OMIM 613988; "
            "UNIQUE MOLECULAR MECHANISM: "
            "  TERC levels: NORMAL (unlike DKC1/NHP2/NOP10 where TERC is destabilised); "
            "  TERT levels: NORMAL; "
            "  Telomerase activity in TRAP assay: NORMAL or near-normal; "
            "  Telomere length: SHORT (<1st centile flow-FISH); "
            "  Mechanism: telomerase assembled but NOT localised to telomeres (Cajal body trafficking defect); "
            "DIAGNOSTIC IMPLICATION: "
            "  WRAP53 DC: telomere shortening with NORMAL TERC/telomerase levels; "
            "  This profile (short telomeres + normal TERC) narrows DDx: "
            "    → Consider WRAP53, ACD/TPP1, PARN, CTC1, STN1 deficiency; "
            "CLINICAL FEATURES: "
            "  DC triad; "
            "  BMF; "
            "  No additional HH features in most cases; "
            "  Learning disability (variable); "
            "CAJAL BODY BIOLOGY: "
            "  Cajal bodies: nuclear organelles marked by coilin; "
            "  Functions: snRNA modification, spliceosome assembly, telomerase assembly/recruitment; "
            "  WRAP53/TCAB1 + WDR79 (same gene): interact with coilin → Cajal body integrity; "
            "  WRAP53 also has roles in DNA damage response (p53 regulation — hence 'WRAP53' = WD40 repeat p53)"
        ),
        "disease_pathway": (
            "WRAP53/TCAB1 — CAJAL BODY TELOMERASE TRAFFICKING PATHWAY: "
            "CAJAL BODY ROUTE FOR TELOMERASE: "
            "  1. TERC transcribed in nucleus; "
            "  2. TERC CAB-box recognised by WRAP53; "
            "  3. WRAP53-TERC complex transported to Cajal bodies (coilin-positive nuclear foci); "
            "  4. In Cajal bodies: TERT joins TERC (TERT recruited via TCAB1 WD40 domain); "
            "     + H/ACA snoRNP complex (dyskerin-NHP2-NOP10-GAR1) stabilises; "
            "     + Cajal body provides platform for full telomerase complex maturation; "
            "  5. Mature telomerase complex (TERT + TERC + auxiliary) released from Cajal body; "
            "  6. During S phase: telomerase recruited to telomeres (via TPP1/ACD on shelterin); "
            "  7. Telomere extension occurs; "
            "WRAP53 DEFICIENCY: "
            "  TERC cannot reach Cajal bodies → TERT and TERC cannot co-localise → "
            "  → Telomerase complex not fully matured → telomeres not extended → progressive shortening; "
            "  Despite: TERC levels normal; TERT levels normal; TRAP activity near-normal; "
            "  = FUNCTIONAL telomerase deficiency at the level of SUBCELLULAR LOCALISATION; "
            "WRAP53 AND p53: "
            "  WRAP53 (WD repeat, antisense to p53 gene): antisense overlap at TP53 locus; "
            "  WRAP53 regulates TP53 mRNA stability (antisense interaction); "
            "  Also: WRAP53 protein interacts with WRAP53-mediated DNA damage response (MDC1 recruitment)"
        ),
        "pathognomonic": (
            "WRAP53 PATHOGNOMONIC FEATURES: "
            "SHORT TELOMERES WITH NORMAL TERC AND NORMAL TELOMERASE ACTIVITY: "
            "  This paradox = strong clue to Cajal body telomerase trafficking defect; "
            "  Genes causing this: WRAP53, ACD/TPP1, PARN (some), CTC1/STN1; "
            "  Standard TRAP assay: may appear normal → misleading; "
            "  Gold standard: immunofluorescence showing TERC fails to localise to Cajal bodies; "
            "CAJAL BODY COLOCALISATION ASSAY: "
            "  Normal: TERC + TERT co-localise with coilin (Cajal body marker) in S phase; "
            "  WRAP53 deficiency: TERC does not co-localise with coilin → TERC stays in nucleoplasm; "
            "KEY DDx: "
            "  WRAP53 vs DKC1/NHP2/NOP10: these cause TERC level reduction; WRAP53 does NOT; "
            "  WRAP53 vs ACD/TPP1: both Cajal-body/shelterin related; telomere short; distinguish by sequencing; "
            "F164L MUTATION: "
            "  Most commonly reported WRAP53 mutation; "
            "  WD40 repeat domain disruption → CAB-box recognition impaired; "
            "CLINICAL RESPONSE: "
            "  Androgens: some response (increase TERT/TERC expression, may partially overcome localisation deficit); "
            "  HSCT: RIC conditioning for BMF; TBI CI"
        ),
        "key_facts": [
            "WRAP53-TERC-CAJAL-BODY-TRAFFICKING-DEFECT",
            "WRAP53-NORMAL-TERC-LEVELS-SHORT-TELOMERES-PARADOX",
            "WRAP53-AR-DC-TYPE-6",
            "WRAP53-CAB-BOX-UGAG-TERC-RECOGNITION",
            "WRAP53-F164L-MOST-COMMON-MUTATION",
            "WRAP53-WD40-TCAB1-COILIN",
            "WRAP53-FLOW-FISH-LT1ST-CENTILE",
            "WRAP53-TELOMERASE-LOCALISATION-NOT-ASSEMBLY",
        ],
        "treatment": (
            "WRAP53 AR-DC: "
            "Androgens (danazol/oxymetholone): partial response; "
            "HSCT: RIC for severe BMF; TBI CI; "
            "Surveillance: CBC, LFTs, PFTs; oral leucoplakia biopsy; "
            "Genetic counselling: AR; 25% sibling risk"
        ),
        "seed_base": 2795,
        "n_patients": 40,
    },
    {
        "gene": "ACD",
        "protein": (
            "ACD -- 16q22.1 AD/AR -- 544aa -- TPP1-Shelterin-Telomerase-Recruiter-"
            "61kDa-OB-Fold-TEL-Patch-TERT-Recruitment-POT1-Binding-"
            "OMIM-Gene-609377-Disease-DC-AD-Type6-616553-HH-616553"
        ),
        "locus": "16q22.1",
        "protein_size": "544 aa / 61 kDa (ACD/TPP1; shelterin component; OB fold; TEL patch (E169, E171) = TERT-recruiting surface; links POT1 (single-strand telomere binding) to TIN2; recruits processivity factor for telomerase; both shelterin AND telomerase recruitment roles)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (haploinsufficiency, DC-AD6) OR AUTOSOMAL RECESSIVE (biallelic, more severe): "
            "ACD encodes TPP1 (TINT2-POT1-TIN2 complex protein / ACD protein); "
            "DUAL ROLE OF TPP1/ACD: "
            "Role 1 — SHELTERIN COMPONENT: "
            "  TPP1 bridges TIN2 and POT1 within shelterin complex; "
            "  Shelterin: TRF1-TRF2-RAP1-TIN2-TPP1-POT1 — protects chromosome ends from DNA damage response; "
            "  TPP1 OB fold binds POT1 OB fold; together bind single-stranded 3' telomere overhang; "
            "  Loss of TPP1 → shelterin incomplete → telomere deprotected → DDR activation; "
            "Role 2 — TELOMERASE RECRUITMENT: "
            "  TEL patch (glutamates E169, E171 on OB fold surface): directly recruits TERT to telomeres; "
            "  TPP1 TEL patch binds TERT TEN domain → positions telomerase at chromosome end for extension; "
            "  TPP1 also ENHANCES TELOMERASE PROCESSIVITY (how many repeats added per binding event); "
            "  TEL-patch mutations: telomerase not recruited → telomere shortening despite normal telomerase levels; "
            "AD (DC-AD6): "
            "  TEL-patch mutations (K170del; L104R) → haploinsufficiency of telomerase recruitment; "
            "  Phenotype: DC/BMF/IPF (similar to TERC/TERT AD); "
            "AR (HH/severe DC): "
            "  Biallelic LOF (both shelterin AND recruitment impaired); "
            "  More severe: HH spectrum + SCID-like + cerebellar hypoplasia"
        ),
        "disease_category": (
            "DYSKERATOSIS CONGENITA AD TYPE 6 / HH SPECTRUM — OMIM 616553; "
            "DUAL PHENOTYPE (AD and AR): "
            "AD (TEL-patch): "
            "  DC triad + BMF; "
            "  IPF reported; "
            "  Genetic anticipation possible (similar mechanism to TERC/TERT); "
            "AR (biallelic): "
            "  HH: cerebellar hypoplasia + SCID + BMF + growth retardation; "
            "  Severe early-onset DC; "
            "TEL-PATCH BIOLOGY (KEY CONCEPT): "
            "  TEL patch E169/E171 residues: mutated in DC; "
            "  TEL-patch mutants: telomerase in vitro activity NORMAL (assembled); "
            "  Telomere extension IN CELLS: IMPAIRED (telomerase not recruited to telomeres); "
            "  = Same paradox as WRAP53 (normal telomerase in vitro, short telomeres in vivo); "
            "SHELTERIN DEPROTECTION: "
            "  Severe biallelic ACD → shelterin incomplete → ATM/ATR DDR at telomeres → "
            "    → chromosomal fusions, rapid cell death; "
            "CLINICAL CONTEXT: "
            "  K170del (Lys170 deletion in TEL patch): reported AD DC; "
            "  Germline TEL-patch mutations: rare but provide proof-of-concept for "
            "    telomerase RECRUITMENT vs ASSEMBLY distinction"
        ),
        "disease_pathway": (
            "ACD/TPP1 — SHELTERIN AND TELOMERASE RECRUITMENT PATHWAY: "
            "SHELTERIN ARCHITECTURE: "
            "  TRF1 + TRF2 → bind double-stranded TTAGGG repeats (TTAGGG interacts with Myb domains); "
            "  TRF2 recruits RAP1 (protects from NHEJ); "
            "  TRF1 + TRF2 both recruit TIN2 (TINF2); "
            "  TIN2 recruits TPP1 (ACD); "
            "  TPP1 recruits POT1 → POT1 binds single-stranded 3' overhang; "
            "  Full shelterin = TRF1-TRF2-RAP1-TIN2-TPP1-POT1 hexamer; "
            "TPP1 IN TELOMERASE RECRUITMENT: "
            "  TEL patch (OB fold surface): TERT TEN domain directly contacts TEL patch; "
            "  Interaction: TERT (TEN domain) + TPP1 (TEL patch) → telomerase recruited to telomere end; "
            "  PROCESSIVITY: TPP1 also increases the number of TTGGGG repeats added per binding event; "
            "    Without TPP1: telomerase adds 1-3 repeats; with TPP1: adds 10-20+ repeats (processive); "
            "ACD MUTATIONS — TWO CLASSES: "
            "  Class 1 (TEL-patch): E169A/G, E171A (loss of TERT interaction); telomerase not recruited; "
            "  Class 2 (POT1-binding face): L104R; loss of POT1 contact; shelterin deprotection; "
            "  Class 3 (TIN2-binding): severe biallelic; full shelterin disruption; "
            "TELOMERE DEPROTECTION (biallelic/severe): "
            "  Unprotected telomeres → recognised as DSBs → ATM activation → 53BP1 foci at telomeres → "
            "    → chromosome fusions (NHEJ at unprotected ends) → genome instability → cell death"
        ),
        "pathognomonic": (
            "ACD/TPP1 PATHOGNOMONIC FEATURES: "
            "TEL-PATCH MUTATION — TELOMERASE RECRUITMENT DEFECT: "
            "  Short telomeres (flow-FISH <1st centile) with NORMAL telomerase TRAP activity; "
            "  Distinction from DKC1/NHP2/NOP10: those cause TERC reduction; ACD TEL-patch does NOT; "
            "  Confirmation: telomerase co-immunoprecipitation with TPP1 mutant fails; "
            "AD INHERITANCE WITH ANTICIPATION (TEL-PATCH AD CASES): "
            "  Similar to TERC/TERT; next generation shorter telomeres + earlier onset; "
            "BIALLELIC AR (HH SPECTRUM): "
            "  Cerebellar hypoplasia + SCID-like + severe BMF + growth retardation; "
            "  Worse than TEL-patch AD; "
            "K170DEL CLINICAL SIGNATURE: "
            "  Lysine 170 deletion in TEL patch: TERT TEN domain cannot bind; "
            "  Telomerase assembled but NOT recruited → short telomeres → DC/BMF; "
            "KEY DDx: "
            "  ACD/TPP1 vs TINF2: TINF2 = AD (de novo), typically severe DC-Revesz; "
            "    ACD-AD: similar to TINF2 but milder; ACD-AR: similar to RTEL1-HH; "
            "  ACD vs WRAP53: both cause normal TERC + short telomeres; distinguish by sequencing; "
            "TELOMERASE PROCESSIVITY ASSAY: "
            "  ACD TEL-patch mutants: reduced processivity in vitro (fewer repeats per binding event); "
            "  Diagnostic but research-grade only"
        ),
        "key_facts": [
            "ACD-TPP1-TEL-PATCH-E169-E171-TERT-RECRUITMENT",
            "ACD-SHELTERIN-POT1-TIN2-BRIDGE",
            "ACD-AD-DC-TYPE-6-TEL-PATCH",
            "ACD-AR-HH-BIALLELIC-SEVERE",
            "ACD-NORMAL-TERC-SHORT-TELOMERES-PARADOX",
            "ACD-TELOMERASE-PROCESSIVITY-ENHANCER",
            "ACD-K170DEL-TEL-PATCH-FOUNDER",
            "ACD-FLOW-FISH-LT1ST-CENTILE",
        ],
        "treatment": (
            "ACD/TPP1 DC: "
            "AD (TEL-patch): Androgens; HSCT-RIC for BMF; TBI CI; "
            "AR (biallelic HH): HSCT early (before cerebellar damage severe); RIC; "
            "Genetic counselling: AD (50% risk) vs AR (25% risk; counsel parents); "
            "Surveillance: CBC, PFTs, brain MRI for HH"
        ),
        "seed_base": 2796,
        "n_patients": 40,
    },
    {
        "gene": "PARN",
        "protein": (
            "PARN -- 16p13.12 AR -- 639aa -- Poly-A-Specific-Ribonuclease-"
            "74kDa-Deadenylase-TERC-3prime-Oligo-A-Tail-Degradation-"
            "OMIM-Gene-604212-Disease-HH-616733-IPF-615292"
        ),
        "locus": "16p13.12",
        "protein_size": "639 aa / 74 kDa (PARN; poly(A)-specific exoribonuclease; DEDDh ribonuclease family; degrades 3' polyadenylated tails; trims oligoadenylated TERC 3' end to mature form; TERC stability depends on PARN-mediated trimming; IPF + HH spectrum)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF (compound heterozygous most common): "
            "PARN encodes Poly(A)-specific Ribonuclease, a 3'→5' exoribonuclease; "
            "FUNCTION — TERC 3' PROCESSING: "
            "  TERC is transcribed with a 3' extension beyond its mature end; "
            "  PAPD5 (poly(A) polymerase): adds short 3' oligoadenylated tail (oligo-A) to TERC → "
            "    → oligo-A tail targets TERC for degradation by ZCCHC8/exosome; "
            "  PARN: trims 3' oligo-A tails from TERC → PREVENTS ZCCHC8-mediated degradation → "
            "    → TERC stabilised at mature 3' end (H/ACA box protected); "
            "  BALANCE: PAPD5 (adds A → degradation signal) vs PARN (trims A → protection); "
            "PARN DEFICIENCY: "
            "  PARN cannot trim TERC → TERC 3' oligo-A tails persist → ZCCHC8 targets → "
            "    → TERC degraded → telomerase activity reduced → telomere shortening; "
            "PARN DEFICIENCY SEVERITY SPECTRUM: "
            "  Severe biallelic LOF: PARN deficiency → TERC markedly reduced → HH phenotype; "
            "  Hypomorphic biallelic alleles: TERC moderately reduced → IPF in adults (milder); "
            "GENOTYPE-PHENOTYPE: "
            "  Near-null biallelic: HH (cerebellar hypoplasia + SCID + BMF + growth retardation); "
            "  Hypomorphic alleles: IPF adult onset (PARN variants in ~3-5% of familial IPF); "
            "  Intermediate: DC triad + early BMF; "
            "THERAPEUTIC IMPLICATION: "
            "  PAPD5 inhibition (e.g., BCH001): rescues TERC levels in PARN-deficient cells → "
            "    → may restore telomere maintenance → investigational therapeutic target"
        ),
        "disease_category": (
            "HOYERAAL-HREIDARSSON SYNDROME (HH) — OMIM 616733; "
            "FAMILIAL PULMONARY FIBROSIS (IPF) — OMIM 615292; "
            "PARN DEFICIENCY SPECTRUM: "
            "  Severe: PARN-null biallelic → HH; "
            "    HH: cerebellar hypoplasia + SCID-like combined immunodeficiency + "
            "       BMF + growth retardation + microcephaly; "
            "    HH = most severe end of DC spectrum; "
            "    PARN-HH: TERC levels very low (most severe); "
            "  Moderate: compound heterozygous PARN → DC triad + BMF; "
            "  Mild: hypomorphic PARN → IPF in adults; "
            "    PARN IPF: same phenotype as TERT/TERC IPF; "
            "    Flow-FISH: <10th centile in IPF patients with PARN variants; "
            "MOLECULAR MECHANISM INSIGHT: "
            "  PARN-deficient cells: TERC has extended 3' oligo-A tails detectable by Northern blot; "
            "  PAPD5 knockdown in PARN-deficient cells: rescues TERC → rescues telomere length; "
            "  Proof-of-concept for PAPD5 inhibitor therapy (BCH001 preclinical 2021); "
            "TERC OLIGOADENYLATION: "
            "  Not just PARN: ZCCHC8 (nuclear RNA exosome targeting complex) + DIS3L2 also degrade TERC; "
            "  ZCCHC8 variants also cause pulmonary fibrosis/telomere biology disorder; "
            "NAND PATHOLOGY: "
            "  PARN: also deadenylates other mRNAs (AU-rich element mRNAs); "
            "    Broad role in mRNA degradation; "
            "    In early embryo: PARN clears maternal mRNA → required for maternal-to-zygotic transition"
        ),
        "disease_pathway": (
            "PARN — TERC 3' PROCESSING AND STABILITY PATHWAY: "
            "TERC BIOGENESIS: "
            "  TERC transcribed by RNA polymerase II (not RNAP III, unlike most ncRNAs); "
            "  3' end: processed to mature H/ACA 3' end (H-box...ACA-3'); "
            "  PAPD5 (TRF4-2): adds 3-10 adenosine residues to TERC 3' end (oligo-A tail); "
            "    Oligo-A = targeting signal for nuclear RNA exosome complex; "
            "  PARN: removes PAPD5-added oligo-A tails → TERC 3' end trimmed to mature form → "
            "    → protected by dyskerin H/ACA complex → stable telomerase RNA; "
            "PARN DEFICIENCY — TERC DECAY: "
            "  Without PARN: oligo-A tail persists on TERC → "
            "    → ZCCHC8 (NEXT complex) recognises → MTR4/ZCCHC8 → nuclear exosome (DIS3) → TERC degraded; "
            "  TERC levels: reduced 30-90% depending on PARN allele severity; "
            "  Telomerase: proportionally reduced; "
            "PAPD5 INHIBITOR APPROACH (BCH001): "
            "  BCH001 (PAPD5 inhibitor): blocks PAPD5 → less oligo-A added to TERC → "
            "    → even without PARN, less TERC targeted → TERC levels restored; "
            "  2021 Nature Communications: BCH001 rescues telomere length in PARN-deficient cells; "
            "  First small-molecule approach targeting RNA processing to rescue telomerase; "
            "  Clinical development: early stage (no approved indication yet); "
            "ANDROGEN EFFECT ON PARN: "
            "  Androgens: increase TERT + TERC transcription; "
            "  In PARN deficiency: more TERC transcribed → despite faster degradation, net TERC may increase; "
            "  Some androgen response in PARN-DC/IPF reported"
        ),
        "pathognomonic": (
            "PARN PATHOGNOMONIC FEATURES: "
            "TERC 3' OLIGOADENYLATED TAIL ON NORTHERN BLOT: "
            "  PARN-deficient cells: TERC migrates slower on Northern blot (oligo-A tail lengthens TERC); "
            "  This molecular fingerprint is specific to PARN deficiency among DC genes; "
            "  Diagnostic but requires specialised laboratory; "
            "HH + IPF IN SAME GENE (ALLELIC SERIES): "
            "  PARN null → HH (severe); PARN hypomorphic → IPF (mild adult phenotype); "
            "  Few genes span from neonatal lethal to adult IPF via allelic series; "
            "  Context: TERT (AD → AR spectrum), PARN (AR: severity depends on residual activity); "
            "SHORT TELOMERES + LOW TERC: "
            "  PARN: TERC reduced (unlike WRAP53/ACD where TERC is normal); "
            "  Helps distinguish PARN from WRAP53/ACD (both short telomeres); "
            "  PARN = TERC low; WRAP53/ACD TEL-patch = TERC normal; "
            "PAPD5-INHIBITOR RESCUE: "
            "  BCH001 in PARN-deficient cells: rescues TERC → rescues telomere length; "
            "  This functional rescue assay confirms PARN mechanism; "
            "KEY DDx: "
            "  PARN vs DKC1/NHP2/NOP10: same result (TERC reduced); different genes; panel seq; "
            "  PARN vs TERT/TERC IPF: PARN = AR; TERT/TERC = AD; family history key; "
            "  PARN-HH vs RTEL1-HH: both AR HH; PARN → TERC low; RTEL1 → TERC normal; "
            "GENETIC COUNSELLING: "
            "  AR: 25% sibling recurrence; "
            "  PARN IPF in adults: siblings have 25% risk of HH or IPF (depending on alleles); "
            "  Prenatal testing available"
        ),
        "key_facts": [
            "PARN-TERC-DEADENYLATION-3PRIME-OLIGO-A-TRIMMING",
            "PARN-AR-HH-SEVERE-IPF-MILD-ALLELIC-SERIES",
            "PARN-TERC-LOW-UNLIKE-WRAP53-ACD",
            "PARN-PAPD5-INHIBITOR-BCH001-INVESTIGATIONAL",
            "PARN-NORTHERN-BLOT-OLIGO-A-TAIL-FINGERPRINT",
            "PARN-ZCCHC8-EXOSOME-TERC-DEGRADATION",
            "PARN-FLOW-FISH-LT1ST-CENTILE",
            "PARN-PAPD5-PARN-BALANCE-TERC-STABILITY",
        ],
        "treatment": (
            "PARN AR-HH / AR-DC / IPF: "
            "HH: early HSCT (RIC; TBI CI); consider before cerebellar damage irreversible; "
            "BMF: Androgens (danazol) — some response; "
            "IPF (PARN hypomorphic adults): Pirfenidone/nintedanib; lung transplant if FVC declining; "
            "Investigational: PAPD5 inhibitor (BCH001) — preclinical proof-of-concept; "
            "Genetic counselling: AR; 25% sibling risk; "
            "Annual surveillance: CBC, LFTs, PFTs, brain MRI (HH); "
            "TBI ABSOLUTELY CI (all DC forms)"
        ),
        "seed_base": 2797,
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
        age = rng.randint(1, 60)
        sex = rng.choice(["M", "F"])
        telomere_pct = round(rng.uniform(0.1, 8.0), 1)  # <1st to <10th centile
        rows.append({
            "patient_id": f"TELO-{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age": age,
            "sex": sex,
            "telomere_centile_pct": telomere_pct,
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

    avg_telo = round(sum(p["telomere_centile_pct"] for p in all_patients) / total, 1)

    per_gene = {}
    for g in ATLAS_GENES:
        pts = [p for p in all_patients if p["gene"] == g["gene"]]
        per_gene[g["gene"]] = {
            "n": len(pts),
            "avg_telomere_centile": round(sum(p["telomere_centile_pct"] for p in pts) / len(pts), 1),
            "locus": g["locus"],
            "protein_size": g["protein_size"].split("(")[0].strip(),
            "disease": g["disease_category"].split("—")[0].strip()[:60],
        }

    return {
        "atlas": "Hereditary-Telomere-Biology-Atlas",
        "subtitle": "Complete 8-Gene Telomere Biology Disorder (Dyskeratosis Congenita / HH / IPF Spectrum) Reference",
        "genes": genes,
        "total_patients": total,
        "seeds": "2790-2797",
        "summary": {
            "avg_telomere_centile_pct": avg_telo,
            "per_gene": per_gene,
        },
        "telomere_maintenance_pathway": {
            "step1_TERC_transcription": "TERC (451 nt RNA) — transcribed by RNAP II; contains template (nt 46-56: CUAACCCUAAC) + H/ACA box + pseudoknot/CR4-CR5 + CAB-box",
            "step2_TERC_stability_H/ACA": "DKC1+NHP2+NOP10+GAR1 — H/ACA snoRNP binds TERC H/ACA box → protects from PAPD5/ZCCHC8 degradation; pseudouridylation of rRNA",
            "step3_TERC_3prime_trimming": "PARN — trims PAPD5-added 3' oligo-A tails → prevents ZCCHC8-exosome TERC degradation; PARN deficiency → TERC low",
            "step4_TERC_Cajal_body": "WRAP53/TCAB1 — binds CAB-box on TERC → localises TERC to Cajal bodies → telomerase assembly/maturation site",
            "step5_TERT_TERC_complex": "TERT (1132aa RT) — binds TERC CR4-CR5 domain → forms active telomerase; dyskerin stabilises; TERT active site D712+D868",
            "step6_TPP1_recruitment": "ACD/TPP1 — TEL patch (E169+E171) recruits TERT to telomeres via TEN domain; enhances processivity; shelterin bridge TIN2-TPP1-POT1",
            "step7_telomere_extension": "TERT reverse-transcribes TERC template → adds TTGGGG repeats to 3' telomere end; processivity enhanced by TPP1; ~50-200 bp/S phase",
            "step8_shelterin_protection": "TRF1-TRF2-RAP1-TIN2-TPP1-POT1 shelterin — caps extended telomeres; prevents ATM/ATR DDR activation; TINF2 (TIN2) mutations cause Revesz/DC2",
        },
        "pathognomonic_signs": {
            "DKC1": "Mucocutaneous triad (nail dystrophy + oral leucoplakia + reticular skin pigmentation) — PATHOGNOMONIC of DC; XL males most severely affected",
            "TERC": "Genetic anticipation in AD pedigrees (IPF→AA→DC triad across generations) — HALLMARK of AD telomere biology disorder",
            "TERT": "Familial IPF (UIP pattern) + flow-FISH <10th centile — TERT = #1 identified genetic cause of familial IPF",
            "NHP2": "AR DC triad + BMF; equal sex ratio (distinguishes from DKC1 XL); Val126Met recurrent; flow-FISH <1st centile",
            "NOP10": "Ultra-rare AR DC (Arg34Trp founder); indistinguishable from NHP2 clinically; structural bridge NHP2-DKC1 in H/ACA snoRNP",
            "WRAP53": "Short telomeres with NORMAL TERC levels — Cajal body TERC trafficking defect (paradox distinguishes from DKC1/NHP2/NOP10)",
            "ACD": "TEL-patch mutations: normal TERC + normal TERT + short telomeres — telomerase recruitment not assembly defect; K170del AD; biallelic → HH",
            "PARN": "TERC 3' oligo-A tail on Northern blot (PATHOGNOMONIC); allelic series: null→HH, hypomorphic→adult IPF; PAPD5-inhibitor BCH001 rescues",
        },
        "critical_treatment_rules": {
            "TBI_ALL_DC": "TBI ABSOLUTELY CONTRAINDICATED in ALL DC forms (DKC1/TERC/TERT/NHP2/NOP10/WRAP53/ACD/PARN) — pulmonary/hepatic fragility → fatal conditioning toxicity",
            "ANDROGENS_MECHANISM": "Danazol/oxymetholone: increase TERT+TERC transcription (androgen response element in TERT promoter) → partial telomere lengthening → BMF improvement (Townsley NEJM 2016)",
            "IMMUNOSUPPRESSION_DC_AA": "IST (ATG+cyclosporin) for DC-associated aplastic anaemia: REDUCED response vs acquired AA → prefer androgens first → HSCT if severe",
            "PAPD5_INHIBITOR_PARN": "BCH001 (PAPD5 inhibitor) rescues TERC in PARN-deficient cells — first small-molecule approach; investigational only (not approved)",
            "ANTICIPATION_COUNSELLING": "AD forms (TERC, TERT, ACD-AD): offspring expect EARLIER + MORE SEVERE disease → pre-symptomatic flow-FISH; proactive counselling",
            "IPF_SCREENING": "ALL familial IPF: flow-FISH first; if <10th centile → TERT+TERC+PARN sequencing panel; inform lung transplant decisions",
        },
        "terc_level_profiles": {
            "TERC_reduced": "DKC1, NHP2, NOP10, PARN — TERC destabilised or degraded; TRAP assay reduced",
            "TERC_normal_short_telomere": "WRAP53, ACD (TEL-patch) — TERC levels normal; TRAP assay normal; telomeres short (recruitment/localisation defect)",
            "TERC_haploinsufficient": "TERC, TERT (AD) — one allele produces 50% TERC/telomerase; flow-FISH <10th centile",
        },
        "cascade_testing": "Flow-FISH telomere length (lymphocytes + granulocytes) for all first-degree relatives of DC/IPF/AA patients; <10th centile → comprehensive telomere biology disorder gene panel (DKC1, TERC, TERT, NHP2, NOP10, WRAP53, ACD, PARN, TINF2, RTEL1, CTC1, STN1); androgen trial for BMF before HSCT; TBI CI in all DC forms",
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
        "atlas": "Hereditary-Telomere-Biology-Atlas Glossary",
        "terms": {
            "Telomere_Biology_Disorder": "Umbrella term for conditions caused by germline mutations in telomerase, shelterin, or telomere-processing genes → progressive telomere shortening → stem cell failure → DC/HH/BMF/IPF spectrum; previously called 'telomeropathy'",
            "Dyskeratosis_Congenita": "DC: inherited telomere biology disorder; classic triad = nail dystrophy + oral leucoplakia + reticular skin pigmentation; BMF leading cause of death; XL (DKC1), AD (TERC,TERT,ACD), AR (NHP2,NOP10,WRAP53,PARN,ACD); spectrum includes HH (severe) to isolated IPF (mild)",
            "Hoyeraal_Hreidarsson": "HH: most severe DC spectrum; cerebellar hypoplasia + immunodeficiency (SCID-like) + BMF + growth retardation + microcephaly; causes: DKC1 (XL), TERT (AR biallelic), ACD (AR), PARN (AR null), RTEL1 (AR); early HSCT only option",
            "Flow_FISH": "Flow fluorescence in situ hybridisation: gold-standard telomere length assay; measures telomere length on lymphocytes + granulocytes using PNA probe (CCCTAA)3 + FITC; <1st centile = DC; <10th centile = suspect telomere biology disorder; reported as percentile for age",
            "DKC1_Dyskerin": "DKC1 (514aa, Xq28, XLR): pseudouridine synthase; H/ACA snoRNP catalytic subunit (with NHP2-NOP10-GAR1); stabilises TERC; pseudouridylates rRNA at 95+ sites; mutations in PUA domain most common; DC-X linked (305000); HH in severe alleles (R158W, A353V)",
            "TERC_RNA": "TERC (451 nt RNA, 3q26.2, AD): telomerase RNA template; contains CUAACCCUAAC template (nt 46-56); pseudoknot/CR4-CR5 binds TERT; H/ACA box (3' end) binds dyskerin; CAB-box binds WRAP53 (Cajal body); haploinsufficiency → DC-AD2 (127550) + IPF + aplastic anaemia; genetic anticipation",
            "TERT_Reverse_Transcriptase": "TERT (1132aa, 5p15.33, AD/AR): catalytic reverse transcriptase; TRBD domain binds TERC CR4-CR5; palm/fingers/thumb catalytic core; active site D712+D868; androgen response element in TERT promoter; AD haploinsufficiency → DC/IPF; AR biallelic → HH; #1 gene in familial IPF",
            "H_ACA_snoRNP": "H/ACA small nucleolar ribonucleoprotein: 4-protein complex (DKC1+NHP2+NOP10+GAR1) + H/ACA box RNA; guide pseudouridylation of rRNA/snRNA; stabilise TERC; NHP2 = L7Ae RNA binding; NOP10 = structural bridge; GAR1 = substrate positioning; all 4 proteins essential",
            "Cajal_Body": "Nuclear organelle (coilin-marked); assembly hub for snRNA modification, spliceosome, and telomerase maturation; WRAP53/TCAB1 recruits TERC to Cajal body via CAB-box (UGAG); Cajal body = where TERT+TERC fully assemble → recruitable to telomeres during S phase",
            "TEL_Patch": "Glutamate-rich surface on TPP1 (ACD) OB fold; E169+E171 = critical residues; TERT TEN domain directly contacts TEL patch → telomerase recruited to telomere; TEL-patch mutations → short telomeres despite normal telomerase assembly; K170del = first AD DC-6 mutation",
            "PARN_Deadenylase": "PARN (639aa, 16p13.12, AR): poly(A)-specific 3'→5' exoribonuclease; trims PAPD5-added oligo-A tails from TERC 3' end → prevents ZCCHC8/exosome-mediated TERC degradation; PARN deficiency → TERC degraded → telomerase reduced → DC/HH (null) or IPF (hypomorphic)",
            "PAPD5_PARN_axis": "PAPD5: adds short oligo-A tail to TERC 3' end → targets TERC for nuclear exosome degradation (via ZCCHC8/NEXT complex); PARN: counters PAPD5 by trimming oligo-A → protects TERC; balance determines TERC steady-state; PARN deficiency → imbalance → TERC low; BCH001 (PAPD5 inhibitor) → restores balance in PARN-deficient cells",
            "WRAP53_TCAB1": "WRAP53/TCAB1 (548aa, 17p13.1, AR): WD40 repeat; binds TERC CAB-box (UGAG at 3' end) → localises TERC to Cajal bodies; without WRAP53 → TERC stays in nucleoplasm → telomerase not assembled → short telomeres; TERC levels NORMAL (unique among DC genes causing telomere shortening)",
            "Androgen_Therapy_DC": "Danazol (400-800 mg/day) or oxymetholone: increases TERT + TERC transcription via androgen response element in TERT promoter; Townsley NEJM 2016: danazol → telomere lengthening + haematological response in DC; 70% initial response; 50% sustained 2yr; LFT monitoring mandatory; HCC rare long-term risk",
            "RIC_HSCT_DC": "Reduced-Intensity Conditioning (fludarabine-based) HSCT for DC aplastic anaemia: TBI ABSOLUTELY CI (lung/liver fragility → fatal conditioning toxicity); myeloablative also avoid; RIC = fludarabine + low-dose cyclophosphamide ± rabbit ATG; 5-yr survival ~50% (pulmonary disease limits long-term outcome)",
            "Genetic_Anticipation_Telomere": "AD DC forms (TERC, TERT, ACD-TEL-patch, TINF2): telomere length partially inherited from parent; shorter parent → shorter offspring telomeres; combined with haploinsufficiency → each generation has shorter baseline telomeres → earlier + more severe disease; classic pedigrees: grandparent IPF → parent AA → child DC/HH",
            "NHP2": "NHP2 (153aa, 5q35.3, AR): H/ACA snoRNP component; L7Ae motif binds H/ACA RNA kink-turn; nucleates complex assembly; Val126Met recurrent; DC-AR5 (613987); equal sex ratio (unlike DKC1 XL); clinical phenotype similar to DKC1 but AR; flow-FISH <1st centile",
            "NOP10": "NOP10 (64aa, 15q14, AR): smallest H/ACA snoRNP protein; structural bridge between NHP2 and DKC1; Arg34Trp = founder variant (disrupts NHP2+DKC1 contacts); ultra-rare (<20 cases); DC-AR4 (224230); identical clinical phenotype to NHP2; panel sequencing required",
            "BCH001_PAPD5_inhibitor": "BCH001: small-molecule PAPD5 inhibitor; reduces PAPD5-mediated oligoadenylation of TERC → ZCCHC8/exosome cannot target TERC → TERC levels restored → telomere length rescue in PARN-deficient cells (Nature Comms 2021 Shukla et al.); investigational; not approved; first potential small-molecule telomere biology disorder therapy",
        }
    }
