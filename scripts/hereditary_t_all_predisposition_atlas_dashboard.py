#!/usr/bin/env python3
"""Hereditary-T-ALL-Predisposition-Atlas -- Complete 8-Gene Reference
ATM     (Ataxia-telangiectasia mutated; 3056aa; 11q22.3; AD/AR LOF;
         A-T (OMIM 208900) biallelic; T-ALL 75x HIGHEST biallelic A-T;
         IgA deficiency PATHOGNOMONIC biallelic; RADIOSENSITIVITY ABSOLUTE biallelic;
         TBI conditioning ABSOLUTELY CONTRAINDICATED biallelic;
         seed SEED_BASE+0).
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         LFS (OMIM 151623); T-ALL elevated; AVOID RADIATION ABSOLUTELY;
         WBMRI Toronto annually; cranial RT ABSOLUTELY CONTRAINDICATED LFS;
         seed SEED_BASE+1).
CDKN2A  (Cyclin-dependent kinase inhibitor 2A; 156aa; 9p21.3; AD LOF;
         FAMM (OMIM 606719); del9p21 T-ALL; p16INK4a + p14ARF dual tumour suppressors;
         CDK4/6 inhibitor palbociclib INACTIVE T-ALL; NOTCH1 somatic co-deletion T-ALL;
         seed SEED_BASE+2).
PTEN    (Phosphatase and tensin homologue; 403aa; 10q23.31; AD LOF;
         Cowden/PHTS (OMIM 158350); T-ALL ~5x via PI3K/mTOR; macrocephaly PATHOGNOMONIC;
         Lhermitte-Duclos PATHOGNOMONIC; everolimus/mTOR inhibitor active;
         seed SEED_BASE+3).
NF1     (Neurofibromin 1; 2839aa; 17q11.2; AD LOF;
         NF1 syndrome (OMIM 162200); T-ALL/T-LBL 5-10x via Ras/MAPK;
         café-au-lait macules PATHOGNOMONIC ≥6; MPNST 8-13%; selumetinib FDA2020;
         seed SEED_BASE+4).
NBN     (Nibrin/NBS1; 754aa; 8q21.3; AD/AR LOF;
         NBS (OMIM 251260); 657del5 Slavic founder 1:150; biallelic ALL 60x T-ALL predominant;
         RADIOSENSITIVITY ABSOLUTE biallelic; cranial RT ABSOLUTELY CONTRAINDICATED biallelic;
         AVOID alkylating agents biallelic;
         seed SEED_BASE+5).
RB1     (Retinoblastoma tumour suppressor; 928aa; 13q14.2; AD LOF;
         Hereditary Retinoblastoma (OMIM 180200); bilateral Rb PATHOGNOMONIC;
         secondary T-ALL/T-LBL post-RT 5x; CDK4/6 inhibitor INACTIVE RB1-null;
         avoid high-dose RT at T-ALL site; secondary malignancy WBMRI;
         seed SEED_BASE+6).
ETV6    (ETS-related transcription factor; 1236aa; 12p13.2; AD LOF;
         THC5 (OMIM 616216); T-ALL also elevated (not only B-ALL); thrombocytopenia co-phenotype;
         AVOID NSAIDs THC5 thrombocytopenia; ETV6::RUNX1 somatic t(12;21) DISTINCT from germline;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3558-3565)
"""
import random

SEED_BASE = 3558

ATLAS_GENES = [
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD/AR LOF -- 3056aa -- "
            "ATM-350kDa-PIK3-Kinase-"
            "DSB-Master-Sensor-"
            "A-T-Biallelic-T-ALL-75x-HIGHEST-"
            "RADIOSENSITIVITY-ABSOLUTE-Biallelic-"
            "IgA-Deficiency-PATHOGNOMONIC-A-T-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM T-ALL molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; serine/threonine PIK3-related kinase (PIKK family); "
            "  HEAT repeats (aa 1-1960) — scaffold; FAT domain (aa 1960-2566) — regulation; "
            "  Kinase domain (aa 2714-2962) — catalytic; FATC (aa 2963-3056) — essential; "
            "  Activated by DSBs → autophosphorylation Ser1981 → monomer → active; "
            "  Substrates: H2AX (γH2AX), CHK2, BRCA1, RAD51, MDM2, p53; "
            "A-T (ATAXIA-TELANGIECTASIA) AND T-ALL: "
            "  Biallelic ATM LOF = Ataxia-Telangiectasia (A-T; OMIM 208900); "
            "  A-T: T-cell ALL/lymphoma 75x HIGHEST — T-LINEAGE PREDOMINANT in A-T; "
            "  T-ALL predominates due to V(D)J recombination errors at TCRA/TCRD (14q11.2), TCRB (7q34), TCRG (7p14); "
            "  ATM required for resolving RAG1/RAG2-generated DSBs during TCR gene rearrangement; "
            "  ATM LOF → unresolved TCR DSBs → chromosomal translocations → T-ALL; "
            "  Characteristic chromosomal translocations in A-T: t(14;14)(q11;q32), inv(14)(q11q32), t(7;14) — T-LBL/T-ALL hotspots; "
            "  Monoallelic germline ATM: T-ALL modest elevation (2-3x, same as B-ALL); "
            "IgA DEFICIENCY — PATHOGNOMONIC: "
            "  Serum IgA undetectable or severely reduced in A-T biallelic = PATHOGNOMONIC; "
            "  IgG also low; total lymphocyte count reduced (CD4 T-cell lymphopenia); "
            "  AFP elevated in >95% A-T — pathognomonic screening test (elevated AFP from birth); "
            "  Cerebellar ataxia onset age 1-3yr; telangiectasias age 3-6yr; "
            "RADIOSENSITIVITY: "
            "  RADIOSENSITIVITY ABSOLUTE biallelic A-T: no RT, no TBI, no craniospinal RT; "
            "  TBI conditioning for SCT ABSOLUTELY CONTRAINDICATED biallelic ATM; "
            "  RT dose in A-T = catastrophic pulmonary fibrosis + myelosuppression + severe RT recall; "
            "  Monoallelic germline ATM: RT dose reduce 20-30% (partial ATM kinase haploinsufficiency); "
            "  Avoid standard cranial RT in A-T — CNS disease: IT triple therapy + nelarabine; "
            "T-ALL IMMUNOPHENOTYPE AND TREATMENT: "
            "  T-cell immunophenotype: TdT+CD3+(cytoplasmic)CD5+CD7+ (cytoplasmic CD3+ = T-lineage key); "
            "  T-LBL vs T-ALL distinction: <25% BM blasts = T-LBL (lymphoma); ≥25% BM blasts = T-ALL; "
            "  A-T T-ALL: nelarabine (FDA approved T-ALL/T-LBL — T-lineage specific purine analogue); "
            "  Nelarabine CI in CNS T-ALL with prior cranial RT (neurotoxicity risk amplified); "
            "  Nelarabine + intrathecal cytarabine: neurotoxicity risk — monitor closely; "
            "  Busulfan-based non-TBI conditioning ONLY if SCT required biallelic A-T; "
            "  Ceralasertib (ATRi) synthetic lethality ATM-LOF T-ALL — investigational; "
            "SURVEILLANCE: "
            "  A-T: AFP at diagnosis (pathognomonic); lymphocyte subsets + Ig levels annually; "
            "  WBMRI Toronto annually biallelic; annual CBC from age 2yr; no RT-based imaging A-T; "
            "  MRI brain (no CT — radiosensitivity); neurology + immunology co-management A-T"
        ),
        "syndrome": "A-T biallelic (OMIM 208900) / T-ALL 75x HIGHEST biallelic / IgA deficiency PATHOGNOMONIC / RADIOSENSITIVITY ABSOLUTE biallelic / TBI conditioning ABSOLUTELY CONTRAINDICATED",
        "inheritance": "AD/AR LOF",
        "all_risk": "ATM biallelic A-T: T-ALL 75x HIGHEST (T-lineage PREDOMINANT — V(D)J TCR recombination DSB hotspot); IgA deficiency + AFP elevated PATHOGNOMONIC biallelic; RADIOSENSITIVITY ABSOLUTE biallelic — no RT, no CRT, no TBI; monoallelic germline ATM: T-ALL 2-3x; RT reduce 20-30% monoallelic; ceralasertib ATRi synthetic lethality investigational; nelarabine FDA-approved T-ALL/T-LBL; avoid nelarabine + cranial RT (neurotoxicity)",
        "pathognomonic": "A-T: cerebellar ataxia (onset 1-3yr) + telangiectasias (onset 3-6yr) + IgA deficiency + AFP elevated >95% + T-ALL/T-LBL = A-T PATHOGNOMONIC; γH2AX foci absent/reduced ATM-deficient cells (DSB repair failure); RADIOSENSITIVITY testing (chromosomal breakage + clonogenic survival) confirms biallelic; t(14;14)(q11;q32) or inv(14)(q11q32) in T-cells = characteristic A-T chromosomal hotspot; serum IgA undetectable + lymphopenia + AFP elevated = A-T screening triad PATHOGNOMONIC",
        "key_avoid": "NEVER use RT in biallelic A-T (ABSOLUTE CONTRAINDICATION — catastrophic toxicity); no cranial RT, no craniospinal RT, no TBI conditioning biallelic ATM; reduce RT 20-30% monoallelic germline ATM; avoid nelarabine in CNS T-ALL with prior cranial RT (neurotoxicity amplified); do NOT combine nelarabine with intrathecal cytarabine without close neurotoxicity monitoring; do NOT order CT/PET-CT for cancer screening A-T biallelic (radiosensitivity — use MRI/ultrasound only); do NOT use standard mammography A-T biallelic (use MRI only); do NOT skip AFP + IgA screening in children with ataxia",
        "key_rule": "ATM T-ALL key rules: (1) biallelic A-T: T-ALL 75x HIGHEST — T-lineage ALL chemo WITHOUT any RT; (2) nelarabine FDA-approved T-ALL/T-LBL — T-lineage specific; (3) TBI conditioning ABSOLUTELY CONTRAINDICATED biallelic; (4) RADIOSENSITIVITY ABSOLUTE biallelic — monoallelic: RT reduce 20-30%; (5) AFP + IgA + ataxia = A-T PATHOGNOMONIC screening triad; (6) t(14;14) / inv(14) T-cell chromosomal hotspot = A-T characteristic; (7) busulfan-based non-TBI conditioning if SCT needed biallelic; (8) ceralasertib ATRi synthetic lethality investigational; (9) cascade testing MANDATORY first-degree relatives.",
        "surveillance": "A-T: AFP at diagnosis (pathognomonic — elevated >95%); serum IgA/IgG/IgM + lymphocyte subsets (CD4/CD8/NK) annually; WBMRI Toronto annually biallelic (no radiation imaging); CBC + flow cytometry from age 2yr; neurological examination annually; MRI brain (not CT); cascade testing first-degree relatives germline ATM; monoallelic: annual CBC + immunophenotyping from age 30yr; radiosensitivity testing at T-ALL diagnosis A-T; T-cell receptor gene FISH (t(14;14)/inv(14) characteristic)",
        "targeted_rx": "Nelarabine (FDA-approved T-ALL/T-LBL — T-lineage purine analogue; AVOID if prior cranial RT); busulfan-based non-TBI conditioning SCT biallelic; ceralasertib (AZD6738 ATRi Phase 2 — ATM-LOF synthetic lethality); PARP inhibitors (partial HRD monoallelic ATM); standard T-ALL protocols WITHOUT RT (BFM 2000, COG AALL0434 T-ALL arm); IT triple therapy (methotrexate + cytarabine + hydrocortisone) — REPLACES CRT; IVIG supplementation A-T immunodeficiency; avoid TBI; avoid CRT; cascade testing MANDATORY",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "Transcription-Factor-G1-Arrest-"
            "LFS-T-ALL-Elevated-B-ALL-15x-"
            "AVOID-RADIATION-ABSOLUTELY-LFS-"
            "Cranial-RT-ABSOLUTELY-CONTRAINDICATED-LFS-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 T-ALL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; N-terminal transactivation domain (aa 1-40) — MDM2/p300 interaction; "
            "  Proline-rich region (aa 40-92); DNA-binding domain (aa 100-290) — MOST mutations; "
            "  Tetramerisation domain (aa 325-356); C-terminal regulatory domain (aa 363-393); "
            "  R175H, R248Q, R248W, R249S, R273H, R282W = hotspot missense (GOF + dominant negative); "
            "LFS AND T-ALL: "
            "  Li-Fraumeni syndrome (LFS; OMIM 151623): germline TP53 LOF → pan-malignancy predisposition; "
            "  T-ALL elevated in LFS (less prominent than B-ALL 15x, but clinically significant); "
            "  LFS T-ALL: young adult/adolescent predominant (TALL age distribution differs from B-ALL); "
            "  TP53 somatic del17p in T-ALL: ~5-8% diagnosis; higher at relapse (acquired del17p); "
            "  LFS T-ALL: T-immunophenotype TdT+CD3+(cytopl)CD5+CD7+; NOTCH1 somatic co-occurring; "
            "RADIATION — CRITICAL LFS RULE: "
            "  AVOID RADIATION ABSOLUTELY in LFS (catastrophic secondary malignancies: sarcoma, brain tumour, leukemia); "
            "  Cranial radiation therapy (CRT) ABSOLUTELY CONTRAINDICATED in LFS T-ALL: second brain tumours (glioma/medulloblastoma); "
            "  Craniospinal RT ABSOLUTELY CONTRAINDICATED LFS T-ALL; "
            "  Modern CNS-directed T-ALL therapy: IT triple therapy (MTX + cytarabine + hydrocortisone) replaces CRT; "
            "  TBI conditioning for SCT ABSOLUTELY CONTRAINDICATED LFS; "
            "  Nelarabine FDA-approved T-ALL — safe in LFS (non-RT); "
            "T-ALL TREATMENT LFS: "
            "  LFS T-ALL: standard T-ALL chemotherapy backbone WITHOUT RT; "
            "  Replace CRT with intensified IT therapy (triple IT: MTX + cytarabine + hydrocortisone); "
            "  Nelarabine for relapsed/refractory T-ALL; venetoclax (BCL2i) emerging LFS T-ALL; "
            "  MDM2 inhibitors INEFFECTIVE in TP53-LOF T-ALL (MDM2i only active TP53-WT); "
            "SURVEILLANCE: "
            "  WBMRI Toronto protocol annually (brain/spine/whole body); "
            "  CBC + flow cytometry annually from age 5yr; breast MRI from 20yr; colonoscopy 25yr; "
            "  T-ALL LFS: somatic TP53/del17p at diagnosis AND relapse"
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) / T-ALL elevated / AVOID RADIATION ABSOLUTELY / cranial RT ABSOLUTELY CONTRAINDICATED LFS / WBMRI Toronto annually",
        "inheritance": "AD LOF",
        "all_risk": "TP53 LFS: T-ALL elevated; B-ALL 15x; LFS total cancer risk 50% by 30yr, 90% by 60yr; somatic TP53/del17p ~5-8% T-ALL at dx; AVOID RADIATION ABSOLUTELY (secondary malignancies catastrophic LFS); cranial RT ABSOLUTELY CONTRAINDICATED LFS; TBI ABSOLUTELY CONTRAINDICATED LFS; nelarabine FDA-approved T-ALL (safe non-RT); WBMRI Toronto annually; MDM2 inhibitors INEFFECTIVE TP53-LOF T-ALL",
        "pathognomonic": "LFS: sarcoma + early breast cancer + brain tumour + adrenocortical carcinoma + T-ALL/B-ALL + diverse family cancer history = LFS PATHOGNOMONIC; del17p13.1 on FISH + TP53 sequence loss in T-ALL blasts = somatic TP53 inactivation; LFS patient receiving cranial RT → second brain tumour risk extreme — CONTRAINDICATED; WBMRI Toronto protocol = standard LFS surveillance; TP53 germline R248W/R175H hotspot = most common LFS alleles",
        "key_avoid": "AVOID RADIATION ABSOLUTELY in LFS T-ALL (secondary malignancies catastrophic — multiple primaries); cranial RT ABSOLUTELY CONTRAINDICATED (LFS second brain tumours — glioma, medulloblastoma); craniospinal RT ABSOLUTELY CONTRAINDICATED LFS T-ALL; TBI conditioning SCT CONTRAINDICATED LFS; do NOT use MDM2 inhibitors (idasanutlin/navtemadlin) in TP53-LOF T-ALL (MDM2i ONLY effective TP53-WT — wrong mechanism LOF); do NOT delay WBMRI Toronto protocol annually; do NOT use nelarabine after prior cranial RT (neurotoxicity — AVOID in LFS as RT ALREADY CONTRAINDICATED)",
        "key_rule": "TP53 LFS T-ALL key rules: (1) AVOID RADIATION ABSOLUTELY — no CRT, no craniospinal RT, no TBI (LFS); (2) IT triple therapy (methotrexate + cytarabine + hydrocortisone) REPLACES cranial RT; (3) nelarabine FDA-approved T-ALL — safe in LFS (non-RT); (4) WBMRI Toronto annually — LFS whole-body cancer surveillance; (5) venetoclax emerging TP53-mutated T-ALL; (6) MDM2 inhibitors INEFFECTIVE TP53-LOF T-ALL; (7) cascade testing MANDATORY first-degree relatives; (8) LFS cancer registry enrollment.",
        "surveillance": "LFS: WBMRI Toronto annually (brain/spine/whole body); CBC + flow cytometry annually from age 5yr; TP53 germline confirmed panel; breast MRI annually from age 20yr; colonoscopy from 25yr; dermatology annually; somatic TP53/del17p at T-ALL diagnosis AND relapse; CBC + LDH monthly if prior T-ALL history (relapse); cascade testing first-degree relatives; LFS Registry (ISLFSG) enrollment recommended; T-cell immunophenotyping (TdT+CD3+CD5+CD7+) if CBC concerning",
        "targeted_rx": "Standard T-ALL protocols WITHOUT RT (COG AALL0434 T-ALL arm, BFM 2000); nelarabine (FDA-approved T-ALL/T-LBL — SAFE in LFS non-RT); IT triple therapy (methotrexate + cytarabine + hydrocortisone) REPLACES CRT in LFS; venetoclax (BCL2i — TP53-mutated T-ALL emerging role); bortezomib (proteasome inhibitor emerging T-ALL); avoid MDM2 inhibitors (LOF ineffective); avoid CRT/TBI (LFS ABSOLUTELY); cascade testing MANDATORY",
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 AD LOF -- 156aa -- "
            "p16-INK4A-17kDa-CDK4-6-Inhibitor-"
            "p14ARF-MDM2-p53-Axis-"
            "FAMM-del9p21-Somatic-T-ALL-"
            "NOTCH1-CDKN2A-Co-Deletion-"
            "Palbociclib-INACTIVE-p16-Null-OMIM-606719"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / ~17 kDa / 9p21.3 CDKN2A T-ALL molecular context: "
            "STRUCTURE: "
            "  156 aa / ~17 kDa (p16INK4a); CDKN2A locus encodes TWO tumour suppressors: "
            "  p16INK4a (156aa, exons 1α-2-3) — 4 ankyrin repeats; CDK4/CDK6 binding; "
            "  p14ARF (132aa, exon 1β-2-3) — MDM2 binding → p53 stabilisation (alternate reading frame); "
            "  p16INK4a competes with D-cyclins → inhibits CDK4/CDK6 → pRB not phosphorylated → G1 arrest; "
            "  p16INK4a LOF → unchecked CDK4/CDK6 → pRB hyperphosphorylation → E2F release → S phase; "
            "  p14ARF LOF → MDM2 unopposed → p53 degradation → p53 pathway loss (dual hit); "
            "del9p21 SOMATIC T-ALL: "
            "  del9p21 somatic VERY COMMON in T-ALL (30-40% at diagnosis — higher rate than B-ALL 20-35%); "
            "  Homozygous del9p21 typical in T-ALL (biallelic loss of both p16INK4a and p14ARF); "
            "  CDKN2A/CDKN2B co-deletion (p15INK4b adjacent) typical in T-ALL; "
            "  del9p21 T-ALL = independent adverse marker (shorter EFS); "
            "NOTCH1 AND CDKN2A CO-DELETION — T-ALL SPECIFIC: "
            "  NOTCH1 somatic mutations: 60-70% T-ALL (MOST COMMON T-ALL somatic driver); "
            "  NOTCH1 activating mutations/deletions drive T-ALL via HES1, MYC, mTOR upregulation; "
            "  CDKN2A del9p21 + NOTCH1 somatic = COMMON co-occurring T-ALL profile; "
            "  NOTCH1 somatic mutations are ACQUIRED and DISTINCT from germline CDKN2A LOF predisposition; "
            "  gamma-secretase inhibitors (GSI) target NOTCH1 pathway — investigational T-ALL; "
            "GERMLINE CDKN2A LOF — FAMM: "
            "  FAMM: Familial Atypical Multiple Mole Melanoma (OMIM 606719); "
            "  Germline CDKN2A LOF: B-ALL/T-ALL 5-10x; melanoma 28-67x HIGHEST; pancreatic 17-39x; "
            "  CDKN2A germline + somatic del9p21 T-ALL = dual-hit CDKN2A (worst T-ALL outcome); "
            "CDK4/6 INHIBITOR — KEY T-ALL RULE: "
            "  p16-null T-ALL: CDK4/6 inhibitor palbociclib INACTIVE (CDK4/6 already unrestrained); "
            "  Venetoclax (BCL2i) ACTIVE regardless CDKN2A status in T-ALL; "
            "TREATMENT: "
            "  CDKN2A-T-ALL: standard T-ALL protocols; palbociclib INACTIVE p16-null; venetoclax preferred; "
            "  del9p21 T-ALL: intensified arm; GSI investigational NOTCH1+ T-ALL; nelarabine for relapsed"
        ),
        "syndrome": "FAMM / del9p21 somatic T-ALL 30-40% / CDKN2A dual p16INK4a + p14ARF LOF / NOTCH1 co-deletion T-ALL / CDK4/6 inhibitor palbociclib INACTIVE p16-null / pancreatic 17-39x EUS",
        "inheritance": "AD LOF",
        "all_risk": "CDKN2A FAMM LOF: T-ALL/B-ALL 5-10x; del9p21 somatic 30-40% T-ALL — very common T-ALL cytogenetic deletion; NOTCH1 somatic (60-70% T-ALL) + CDKN2A del9p21 = common co-profile; p16-null T-ALL → palbociclib INACTIVE; venetoclax ACTIVE regardless; nelarabine FDA-approved T-ALL; melanoma 28-67x; pancreatic 17-39x EUS from 40yr MANDATORY",
        "pathognomonic": "FAMM: atypical melanoma >50 moles + family melanoma + pancreatic cancer = CDKN2A PATHOGNOMONIC; del9p21 on T-ALL FISH/SNP array (homozygous common T-ALL — both alleles lost) = adverse marker; p16 IHC loss in T-ALL blasts = CDKN2A inactivation; palbociclib INACTIVE p16-null T-ALL (expected — CDK4/6 already disinhibited); NOTCH1 somatic mutation + CDKN2A del9p21 in T-ALL blasts = COMMON co-profile (DISTINCT from germline CDKN2A LOF)",
        "key_avoid": "Do NOT use CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib) in p16INK4a-null T-ALL (INACTIVE — avoidable toxicity without benefit); do NOT miss pancreatic EUS surveillance FAMM (17-39x risk — EUS from 40yr MANDATORY); do NOT use CDKN2A IHC alone without FISH/SNP array (IHC may be false-negative hemizygous); do NOT confuse NOTCH1 somatic mutation (60-70% sporadic T-ALL) with germline CDKN2A LOF (FAMM hereditary — separate germline testing); do NOT skip melanoma surveillance FAMM families",
        "key_rule": "CDKN2A T-ALL key rules: (1) del9p21 somatic 30-40% T-ALL (higher than B-ALL) — test at diagnosis FISH/SNP array; (2) NOTCH1 somatic (60-70% T-ALL) + CDKN2A del9p21 = common co-profile (DISTINCT from germline CDKN2A FAMM); (3) p16-null T-ALL → CDK4/6 inhibitor palbociclib INACTIVE; (4) venetoclax ACTIVE regardless CDKN2A status; (5) nelarabine FDA-approved T-ALL — use for relapsed/refractory; (6) FAMM: pancreatic EUS from 40yr MANDATORY (17-39x risk); (7) cascade test FAMM families; melanoma surveillance annually from 25yr.",
        "surveillance": "FAMM: dermoscopy annually from 25yr; pancreatic EUS from 40yr (17-39x risk — 6-monthly if family pancreatic history); CBC + T-cell immunophenotyping from age 35yr; T-ALL FISH/SNP array (del9p21) at diagnosis; p16 IHC on T-ALL blasts; NOTCH1/FBXW7 somatic NGS panel at diagnosis; cascade testing first-degree relatives; CDKN2A germline panel in FAMM + T-ALL; abdominal MRI pancreas if EUS unavailable",
        "targeted_rx": "Standard T-ALL protocols (COG AALL0434, BFM 2000/2017); nelarabine (FDA-approved T-ALL — relapsed/refractory); venetoclax (BCL2i — ACTIVE regardless CDKN2A); AVOID CDK4/6 inhibitors (palbociclib/ribociclib) p16-null T-ALL (INACTIVE); gamma-secretase inhibitors (GSI PF-03084014 — NOTCH1-activated T-ALL investigational); bortezomib (proteasome inhibitor T-ALL investigational); wide-excision melanoma FAMM; pancreatic EUS + resection if EUS lesion FAMM; cascade testing MANDATORY",
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 AD LOF -- 403aa -- "
            "PTEN-47kDa-Phosphatase-"
            "PI3K-AKT-mTOR-Negative-Regulator-"
            "Cowden-PHTS-T-ALL-5x-PI3K-mTOR-Pathway-"
            "Macrocephaly-PATHOGNOMONIC-"
            "Lhermitte-Duclos-PATHOGNOMONIC-OMIM-158350"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / ~47 kDa / 10q23.31 PTEN T-ALL molecular context: "
            "STRUCTURE: "
            "  403 aa / ~47 kDa; N-terminal phosphatase domain (aa 7-185) — catalytic; "
            "  C-terminal C2 domain (aa 186-351) — membrane anchoring; "
            "  PDZ-binding motif (aa 400-403) — protein-protein interaction; "
            "  PTEN dephosphorylates PIP3 → PIP2 → negatively regulates PI3K/AKT/mTOR pathway; "
            "  PTEN LOF → unchecked PI3K → phospho-AKT → mTORC1/2 constitutive activation → T-cell proliferation; "
            "  PTEN is on the critical Notch1-PI3K axis in T-ALL pathogenesis; "
            "COWDEN/PHTS AND T-ALL: "
            "  Cowden syndrome / PTEN Hamartoma Tumour Syndrome (PHTS; OMIM 158350); "
            "  Germline PTEN LOF: T-ALL ~5x via constitutive PI3K/mTOR activation; "
            "  PTEN somatic loss in T-ALL: 5-10% T-ALL diagnosis (NOTCH1 pathway cooperation); "
            "  PTEN + NOTCH1 somatic co-loss = T-ALL with activated mTOR + HES1 signalling (compounded); "
            "  PHTS spectrum: Cowden (breast/thyroid/endometrial) + Bannayan-Riley-Ruvalcaba (polyps + macrocephaly) + Proteus syndrome; "
            "MACROCEPHALY — PATHOGNOMONIC: "
            "  Macrocephaly (OFC > 2 SD above mean) = PATHOGNOMONIC clinical hallmark PTEN germline; "
            "  Head circumference MANDATORY in any patient with hamartomas + T-ALL; "
            "  Macrocephaly + T-ALL in child = PTEN germline until proven otherwise; "
            "LHERMITTE-DUCLOS DISEASE — PATHOGNOMONIC: "
            "  Lhermitte-Duclos disease (dysplastic cerebellar gangliocytoma) = PATHOGNOMONIC PTEN; "
            "  Striped cerebellum MRI appearance = PATHOGNOMONIC; "
            "  Adults presenting with Lhermitte-Duclos = PTEN germline testing MANDATORY; "
            "MTOR INHIBITOR — ACTIVE: "
            "  Everolimus (mTOR inhibitor, FDA approved) ACTIVE in PTEN-deficient T-ALL; "
            "  Everolimus + vincristine/dexamethasone — PTEN-deficient T-ALL clinical trials; "
            "  Rapamycin/temsirolimus (mTORC1 inhibitors) — PTEN-LOF T-ALL sensitivity; "
            "  Idelalisib/PI3Kδ inhibitors — PTEN-LOF T-ALL investigational; "
            "  AKT inhibitors (capivasertib, ipatasertib) — PTEN-LOF T-ALL emerging; "
            "TREATMENT: "
            "  PTEN-deficient T-ALL: standard T-ALL backbone + mTOR inhibitor; "
            "  Nelarabine FDA-approved T-ALL — safe; standard RT protocol with haematology review; "
            "  No specific RT contraindication germline PTEN (no DNA repair defect); "
            "SURVEILLANCE: "
            "  Cowden/PHTS: annual thyroid ultrasound from age 7yr; breast MRI + mammography from 30yr; "
            "  Endometrial ultrasound + Pipelle biopsy from 35yr; colonoscopy from 35yr or 5yr before youngest; "
            "  Renal ultrasound annually from 40yr; CBC + T-cell immunophenotyping from early adulthood; "
            "  Brain MRI if cerebellar symptoms (Lhermitte-Duclos); head circumference monitoring from birth"
        ),
        "syndrome": "Cowden/PHTS (OMIM 158350) / T-ALL ~5x via PI3K/mTOR / macrocephaly PATHOGNOMONIC / Lhermitte-Duclos PATHOGNOMONIC / everolimus mTOR inhibitor ACTIVE",
        "inheritance": "AD LOF",
        "all_risk": "PTEN germline LOF Cowden/PHTS: T-ALL ~5x via constitutive PI3K/AKT/mTOR pathway activation; PTEN + NOTCH1 somatic co-loss = compounded mTOR + HES1 activation T-ALL; macrocephaly PATHOGNOMONIC (OFC >2 SD) + T-ALL = PTEN germline; Lhermitte-Duclos (dysplastic cerebellar gangliocytoma) PATHOGNOMONIC; everolimus/mTOR inhibitor ACTIVE PTEN-deficient T-ALL; PHTS multi-organ hamartoma syndrome",
        "pathognomonic": "PHTS: macrocephaly (OFC >2 SD above mean) + hamartomas (thyroid nodules/colon polyps/breast fibroadenoma) + family cancer = PATHOGNOMONIC PTEN germline; Lhermitte-Duclos disease (striped cerebellum MRI — dysplastic cerebellar gangliocytoma) = PATHOGNOMONIC PTEN; Bannayan-Riley-Ruvalcaba syndrome (macrocephaly + polyps + penile freckling) = PATHOGNOMONIC PTEN male; PTEN somatic loss + NOTCH1 activated T-ALL = compounded mTOR signalling; everolimus mTOR inhibitor response in PTEN-deficient T-ALL",
        "key_avoid": "Do NOT miss macrocephaly measurement in PHTS/T-ALL (OFC MANDATORY — PATHOGNOMONIC); do NOT miss Lhermitte-Duclos on MRI (PATHOGNOMONIC PTEN — request cerebellar sequences); do NOT delay breast/thyroid/endometrial surveillance PHTS (Cowden multi-organ risk); do NOT use CDK4/6 inhibitors in PTEN-null T-ALL without first confirming p16 status (CDKN2A may co-mutate); do NOT confuse PTEN somatic T-ALL loss (5-10% acquired) with germline PTEN LOF PHTS (hereditary — separate germline testing); do NOT skip PI3K/mTOR pathway inhibitor consideration in PTEN-deficient T-ALL",
        "key_rule": "PTEN T-ALL key rules: (1) T-ALL ~5x via PI3K/AKT/mTOR pathway — mTOR inhibitor (everolimus) ACTIVE PTEN-deficient T-ALL; (2) macrocephaly (OFC >2 SD) = PATHOGNOMONIC PTEN germline — measure head circumference MANDATORY; (3) Lhermitte-Duclos (striped cerebellum MRI) = PATHOGNOMONIC — MRI brain MANDATORY if symptoms; (4) PTEN + NOTCH1 somatic co-loss = compounded T-ALL (aggressive subtype); (5) nelarabine FDA-approved T-ALL — SAFE PTEN (no DNA repair defect); (6) PHTS multi-organ surveillance MANDATORY (thyroid/breast/endometrial/colon/renal); (7) AKT/PI3K inhibitors emerging PTEN-LOF T-ALL; (8) cascade testing MANDATORY PHTS families.",
        "surveillance": "PHTS/Cowden: thyroid ultrasound annually from age 7yr; breast MRI + mammography from 30yr (BRCA2 co-test if family history); endometrial ultrasound + Pipelle from 35yr; colonoscopy from 35yr; renal ultrasound from 40yr; CBC + T-cell immunophenotyping from early adulthood; brain MRI if cerebellar symptoms (Lhermitte-Duclos); head circumference monitoring from birth (macrocephaly); dermatological exam annually (mucocutaneous features PHTS); PTEN germline sequencing in T-ALL + macrocephaly/hamartoma",
        "targeted_rx": "Everolimus (mTOR inhibitor — FDA approved; ACTIVE PTEN-deficient T-ALL); temsirolimus (mTORC1 inhibitor — PTEN-LOF T-ALL); idelalisib (PI3Kδ inhibitor — investigational PTEN-LOF); capivasertib (AKT inhibitor — investigational PTEN-LOF); nelarabine (FDA-approved T-ALL/T-LBL — safe PTEN no DNA repair CI); standard T-ALL protocols (COG AALL0434, BFM 2000); bortezomib (proteasome inhibitor T-ALL); gamma-secretase inhibitors (NOTCH1 co-activation); cascade testing MANDATORY PHTS families",
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 AD LOF -- 2839aa -- "
            "Neurofibromin-319kDa-Ras-GAP-"
            "Ras-MAPK-Negative-Regulator-"
            "T-ALL-T-LBL-5-10x-Ras-MAPK-Pathway-"
            "Cafe-au-Lait-PATHOGNOMONIC-6plus-"
            "MPNST-8-13pct-Selumetinib-FDA2020-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / ~319 kDa / 17q11.2 NF1 T-ALL molecular context: "
            "STRUCTURE: "
            "  2839 aa / ~319 kDa; GAP-related domain GRD (aa 1198-1530) — Ras GTPase activating (catalytic); "
            "  PH domain, CRAL/TRIO domain, armadillo domain flanking GRD; "
            "  Neurofibromin = Ras GAP: accelerates GTP hydrolysis by Ras → Ras-GDP (inactive); "
            "  NF1 LOF → Ras-GTP accumulates (constitutively active) → RAF/MEK/ERK cascade → cell proliferation; "
            "  NF1 LOF also activates PI3K/AKT pathway (RAS → PI3K cross-talk); "
            "  SPRED1 and NF2 are parallel negative regulators (SPRED1 = Legius syndrome, NF2 = NF2 syndrome); "
            "NF1 SYNDROME AND T-ALL/T-LBL: "
            "  NF1 (Neurofibromatosis type 1; OMIM 162200): germline NF1 LOF → Ras/MAPK activation; "
            "  T-ALL/T-LBL 5-10x in NF1 syndrome via constitutive Ras/MEK/ERK signalling; "
            "  NF1 T-LBL: typically mediastinal mass + pleural effusion; T-LBL <25% BM blasts; "
            "  Ras/MAPK activation in T-cells → T-cell receptor signalling amplification → T-LBL/T-ALL; "
            "  NF1 somatic mutations in T-ALL: rare (NF1 primarily germline predisposition); "
            "CAFÉ-AU-LAIT — PATHOGNOMONIC: "
            "  Café-au-lait macules (CALMs) ≥6 (≥0.5cm prepubertal, ≥1.5cm postpubertal) = PATHOGNOMONIC NF1; "
            "  Axillary/inguinal freckling (Crowe sign) = PATHOGNOMONIC NF1; "
            "  Lisch nodules (iris hamartomas) on slit-lamp = PATHOGNOMONIC NF1 adult; "
            "  Plexiform neurofibromas (congenital) = PATHOGNOMONIC NF1; "
            "  ≥6 CALMs + family history NF1 = NF1 diagnosis before genetic test result; "
            "MPNST AND SELUMETINIB: "
            "  MPNST (malignant peripheral nerve sheath tumour): 8-13% lifetime NF1 risk; "
            "  MPNST = most serious NF1 complication (5-year survival ~50%); "
            "  Selumetinib (MEK1/2 inhibitor, Koselugo; FDA2020) = first FDA-approved NF1 therapy; "
            "  Selumetinib approved for NF1 plexiform neurofibromas (not T-ALL); "
            "  Cobimetinib/binimetinib (MEK inhibitors) — NF1-mutated tumours investigational; "
            "T-ALL TREATMENT NF1: "
            "  Standard T-ALL chemotherapy (no specific RT contraindication NF1 — no DNA repair defect); "
            "  Nelarabine FDA-approved T-ALL — safe NF1; "
            "  MEK inhibitors (trametinib/cobimetinib) — NF1-mutated T-ALL investigational; "
            "  Selumetinib: FDA-approved NF1 plexiform neurofibromas (not T-ALL — off-label); "
            "SURVEILLANCE: "
            "  Annual clinical exam: CALMs, plexiform neurofibromas, blood pressure, vision/hearing; "
            "  MRI brain/spine if neurological symptoms; no routine full-body MRI in NF1 (WBMRI not standard); "
            "  CBC + T-cell immunophenotyping annually from adolescence; "
            "  Ophthalmology annually (optic glioma risk); MPNST monitoring: MRI of enlarging neurofibromas"
        ),
        "syndrome": "NF1 syndrome (OMIM 162200) / T-ALL/T-LBL 5-10x via Ras/MAPK / café-au-lait ≥6 PATHOGNOMONIC / MPNST 8-13% / selumetinib FDA2020 (plexiform NF)",
        "inheritance": "AD LOF",
        "all_risk": "NF1 germline LOF: T-ALL/T-LBL 5-10x via constitutive Ras/MAPK activation; NF1 T-LBL: mediastinal mass + pleural effusion; café-au-lait ≥6 + axillary freckling PATHOGNOMONIC NF1; MPNST 8-13% lifetime (most serious NF1 complication); selumetinib FDA2020 plexiform NF (not T-ALL); MEK inhibitors investigational NF1-T-ALL; nelarabine FDA-approved T-ALL — safe NF1 (no DNA repair defect)",
        "pathognomonic": "NF1: ≥6 café-au-lait macules (≥0.5cm prepubertal/≥1.5cm postpubertal) + axillary/inguinal freckling (Crowe sign) + Lisch nodules (iris hamartomas slit-lamp) + plexiform neurofibromas = NF1 PATHOGNOMONIC (≥2 diagnostic criteria required); child with mediastinal T-LBL + ≥6 CALMs = NF1 until proven otherwise; MPNST arising in plexiform neurofibroma = NF1 PATHOGNOMONIC; MEK/ERK constitutive activation in NF1-mutated T-ALL (Ras pathway biomarker)",
        "key_avoid": "Do NOT miss ≥6 CALMs in T-ALL/T-LBL (NF1 PATHOGNOMONIC — skin exam MANDATORY at diagnosis); do NOT delay NF1 germline testing in mediastinal T-LBL with café-au-lait; do NOT use full-body PET-CT routinely NF1 (radiation risk and false-positive neurofibromas); do NOT miss MPNST (enlarging plexiform neurofibroma = MPNST until proven — MRI MANDATORY for enlarging); do NOT confuse NF1 T-ALL/T-LBL with sporadic T-ALL (NF1 germline predisposition — family testing); selumetinib is for plexiform NF only (NOT T-ALL approved) — do NOT use off-label without MDT discussion",
        "key_rule": "NF1 T-ALL/T-LBL key rules: (1) T-ALL/T-LBL 5-10x — examine CALMs (≥6), axillary freckling, Lisch nodules at T-ALL diagnosis; (2) NF1 T-LBL: mediastinal mass COMMON — CT chest at diagnosis; (3) MPNST 8-13% lifetime — MRI for enlarging neurofibromas; (4) selumetinib (Koselugo) FDA2020 = plexiform NF (NOT T-ALL — MEK inhibitors investigational T-ALL); (5) nelarabine FDA-approved T-ALL — safe NF1 (no DNA repair defect); (6) MEK/ERK pathway inhibitors (trametinib/cobimetinib) — NF1-mutated T-ALL investigational; (7) cascade testing MANDATORY NF1 family.",
        "surveillance": "NF1: annual clinical exam (CALMs count, plexiform NF, blood pressure, vision/hearing); ophthalmology annually (optic pathway glioma 15-20% NF1 — MRI orbit if vision change); MRI brain/spine if neurological symptoms; CBC + T-cell immunophenotyping annually from adolescence; MRI of enlarging neurofibromas (MPNST surveillance); selumetinib for inoperable plexiform NF (FDA2020 paediatric); annual dermatology (NF1 skin features); echocardiogram NF1 (cardiac neurofibromas/hypertension); cascade testing first-degree relatives",
        "targeted_rx": "Selumetinib (MEK1/2 inhibitor — FDA2020 NF1 plexiform neurofibromas; off-label T-ALL investigational); trametinib (MEK1/2 inhibitor — NF1-mutated tumours including T-ALL investigational); cobimetinib (MEK inhibitor — NF1-mutated T-ALL investigational); nelarabine (FDA-approved T-ALL/T-LBL — safe NF1); standard T-ALL protocols (COG AALL0434, BFM 2000); bortezomib (T-ALL); gamma-secretase inhibitors (NOTCH1 co-activation); everolimus (mTOR — Ras/PI3K cross-talk); cascade testing MANDATORY NF1 families",
    },
    {
        "gene": "NBN",
        "protein": (
            "NBN -- 8q21.3 AD/AR LOF -- 754aa -- "
            "NBS1-Nibrin-85kDa-MRN-Complex-"
            "DSB-Sensor-ATM-Recruiter-"
            "657del5-Slavic-Founder-1-in-150-"
            "ALL-60x-T-ALL-Predominant-Biallelic-"
            "Cranial-RT-ABSOLUTELY-CONTRAINDICATED-NBS-OMIM-251260"
        ),
        "locus": "8q21.3",
        "protein_size": (
            "754 aa / ~85 kDa / 8q21.3 NBN T-ALL molecular context: "
            "STRUCTURE: "
            "  754 aa / ~85 kDa; N-terminal FHA domain (aa 1-108) + BRCT1 (aa 114-197) + BRCT2 (aa 221-332) — phosphopeptide docking; "
            "  MRE11-binding domain (aa 665-693); ATM-interacting domain (aa 734-754); "
            "  MRN complex: MRE11-RAD50-NBS1 trimer; DSB sensor and ATM recruitment; "
            "  657del5 = exon 6 frameshift → p.K219fs; founder mutation Czech/Slovak/Polish; "
            "NBS (NIJMEGEN BREAKAGE SYNDROME) AND T-ALL: "
            "  NBS = biallelic NBN LOF → microcephaly + immunodeficiency + radiosensitivity + lymphoma; "
            "  NBS biallelic: ALL 60x HIGHEST — T-ALL PREDOMINANT in NBS (T-lineage V(D)J recombination errors); "
            "  T-cell predominant: TCR gene rearrangement DSBs unresolved by MRN → T-ALL chromosomal translocations; "
            "  Monoallelic 657del5: B-cell ALL 3-5x; T-ALL also elevated monoallelic (1:150 Czech/Slovak/Polish carrier); "
            "  NBS clinical triad: microcephaly + café au lait macules + immunodeficiency + ALL/lymphoma; "
            "  Characteristic NBS chromosomal breakpoints: inv(7)(p13q35), t(7;14)(q35;q11.2) = T-cell hotspots; "
            "CRANIAL RT — ABSOLUTELY CONTRAINDICATED NBS BIALLELIC: "
            "  Cranial radiation ABSOLUTELY CONTRAINDICATED in NBS biallelic: "
            "    (a) RADIOSENSITIVITY ABSOLUTE (MRN complex absent → no DSB repair); "
            "    (b) Catastrophic neurotoxicity in microcephalic NBS brain; "
            "    (c) Secondary brain malignancy extreme (irradiated MRN-deficient CNS); "
            "  Craniospinal RT ABSOLUTELY CONTRAINDICATED NBS biallelic; "
            "  TBI conditioning ABSOLUTELY CONTRAINDICATED NBS biallelic; "
            "  Nelarabine (FDA-approved T-ALL) safe in NBS if avoiding prior cranial RT; "
            "MONOALLELIC NBN: "
            "  657del5 monoallelic: T-cell ALL elevated (1:150 Czech/Slovak/Polish — COMMON); "
            "  Monoallelic NBN: RT reduce 20-30% (partial MRN dysfunction); "
            "ALKYLATING AGENTS — BIALLELIC NBS: "
            "  AVOID alkylating agents biallelic NBS (profound DNA repair defect — aplasia risk); "
            "  Cyclophosphamide, melphalan, busulfan DOSE REDUCE or AVOID biallelic NBS; "
            "  Monoallelic NBN: standard T-ALL chemotherapy with haematological monitoring; "
            "TREATMENT: "
            "  NBS T-ALL: IT triple therapy replaces CRT; nelarabine for T-ALL (avoid if prior CRT); "
            "  Busulfan-based non-TBI conditioning ONLY if SCT required biallelic; "
            "  PARP inhibitors: MRN LOF → HRD → olaparib/rucaparib sensitivity"
        ),
        "syndrome": "NBS biallelic (OMIM 251260) / 657del5 Slavic founder 1:150 / ALL 60x T-ALL PREDOMINANT biallelic / cranial RT ABSOLUTELY CONTRAINDICATED NBS biallelic / RADIOSENSITIVITY ABSOLUTE biallelic / AVOID alkylating agents biallelic",
        "inheritance": "AD/AR LOF",
        "all_risk": "NBN 657del5 monoallelic: T-cell ALL elevated (1:150 Czech/Slovak/Polish); NBS biallelic: ALL 60x HIGHEST T-ALL PREDOMINANT; RADIOSENSITIVITY ABSOLUTE biallelic; cranial RT ABSOLUTELY CONTRAINDICATED NBS biallelic (neurotoxicity + malignancy); AVOID alkylating agents biallelic; monoallelic: RT reduce 20-30%; nelarabine FDA-approved T-ALL safe if no prior CRT; 1:150 Eastern European — COMMON hereditary T-ALL carrier",
        "pathognomonic": "NBS biallelic: microcephaly + café au lait macules + immunodeficiency + 657del5 + T-ALL = NBS PATHOGNOMONIC; B-cell/T-cell ALL in Czech/Slovak/Polish child + family history = test NBN 657del5 FIRST; radiosensitivity testing (chromosomal breakage + clonogenic survival) confirms NBS biallelic; cranial RT in NBS = catastrophic outcome (neurotoxicity + secondary malignancy); IgG low + T-cell lymphopenia + microcephaly = NBS screening triad; inv(7) / t(7;14) T-cell chromosomal hotspot = NBS characteristic",
        "key_avoid": "NEVER use cranial RT in NBS biallelic (ABSOLUTELY CONTRAINDICATED — neurotoxicity + radiosensitivity + secondary malignancy); no craniospinal RT biallelic NBS; no TBI conditioning biallelic NBS; AVOID alkylating agents biallelic (cyclophosphamide, melphalan, busulfan — dose reduce or avoid); do NOT miss 657del5 in Eastern European T-ALL (COMMON 1:150 carrier); do NOT give nelarabine after prior cranial RT (neurotoxicity — NBS cranial RT already CONTRAINDICATED); monoallelic NBN: RT reduce 20-30%",
        "key_rule": "NBN 657del5 T-ALL key rules: (1) test FIRST in Czech/Slovak/Polish ALL patients (1:150 carrier frequency); (2) biallelic NBS: cranial RT ABSOLUTELY CONTRAINDICATED (neurotoxicity + RADIOSENSITIVITY); (3) AVOID alkylating agents biallelic NBS; (4) IT triple therapy replaces CRT biallelic NBS; (5) RADIOSENSITIVITY ABSOLUTE biallelic — no RT, no TBI; (6) nelarabine FDA-approved T-ALL — safe in NBS if no prior CRT; (7) monoallelic: RT reduce 20-30%; (8) PARP inhibitor sensitivity MRN HRD; (9) cascade test Eastern European T-ALL families.",
        "surveillance": "Annual CBC + lymphocyte immunophenotyping from age 2yr NBS; immunoglobulin levels (IgG/IgA/IgM) + lymphocyte subsets annually NBS; WBMRI Toronto biallelic NBS annually (no radiation-based imaging); NBN 657del5 founder test in Eastern European ALL families; cascade testing first-degree relatives; radiosensitivity testing (chromosomal breakage) at T-ALL diagnosis NBS; TCR FISH (inv(7)/t(7;14) characteristic NBS T-ALL); IVIG supplementation NBS immunodeficiency; PARP sensitivity testing at relapse",
        "targeted_rx": "Standard T-ALL protocols without RT (COG AALL0434, BFM 2000); nelarabine (FDA-approved T-ALL — safe NBS if no prior CRT); IT triple therapy (methotrexate + cytarabine + hydrocortisone) — REPLACES CRT biallelic NBS; busulfan-based non-TBI conditioning ONLY if SCT required biallelic; PARP inhibitors (olaparib/rucaparib — MRN HRD sensitivity); bortezomib (T-ALL); IVIG biallelic NBS (immunodeficiency); AVOID alkylating agents biallelic; cascade testing MANDATORY Eastern European T-ALL families",
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 AD LOF -- 928aa -- "
            "pRB-110kDa-Cell-Cycle-Checkpoint-"
            "E2F-Transcription-Factor-Regulator-"
            "Hereditary-Retinoblastoma-Bilateral-PATHOGNOMONIC-"
            "Secondary-T-ALL-T-LBL-Post-RT-5x-"
            "CDK4-6i-INACTIVE-RB1-null-OMIM-180200"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / ~110 kDa / 13q14.2 RB1 T-ALL molecular context: "
            "STRUCTURE: "
            "  928 aa / ~110 kDa; N-terminal domain (aa 1-378) — LXCXE binding pocket N; "
            "  Pocket domain: A box (aa 379-572) + B box (aa 646-775) — E2F binding; "
            "  C-terminal domain (aa 776-928) — CDK phosphorylation sites; "
            "  pRB binds E2F1-8 transcription factors in G1 → represses S phase gene programme; "
            "  CDK4/6-Cyclin D phosphorylates pRB → releases E2F → S phase entry; "
            "  RB1 LOF → constitutive E2F release → S phase → T-cell proliferation; "
            "HEREDITARY RETINOBLASTOMA AND SECONDARY T-ALL/T-LBL: "
            "  Hereditary Retinoblastoma (OMIM 180200): germline RB1 LOF → bilateral Rb; "
            "  Bilateral retinoblastoma = PATHOGNOMONIC germline RB1 (>95% bilateral = germline); "
            "  Secondary T-ALL/T-LBL post-RT 5x: hereditary Rb treated with cranial/orbital RT; "
            "  Radiation field includes anterior brain/orbit → irradiated T-progenitor cells in RT field; "
            "  T-ALL/T-LBL typically arising 5-15yr after RT for primary Rb (therapy-related); "
            "  RB1 LOF → E2F unrestricted → T-lymphoid proliferation (both germline and secondary RT-induced); "
            "  RB1 somatic deletion in T-ALL: ~5-10% T-ALL; RB1 at 13q14.2 occasional somatic deletion; "
            "BILATERAL RETINOBLASTOMA — PATHOGNOMONIC: "
            "  Bilateral Rb (leukocoria both eyes) = PATHOGNOMONIC germline RB1 LOF; "
            "  Unilateral Rb in child <1yr also suggests germline RB1 (onset-age criterion); "
            "  Trilateral Rb (bilateral Rb + pineoblastoma) = PATHOGNOMONIC hereditary Rb — highest T-ALL risk; "
            "  White pupil reflex (leukocoria) + family history = emergency paediatric ophthalmology referral; "
            "CDK4/6 INHIBITOR — INACTIVE RB1-NULL: "
            "  CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib) INACTIVE in RB1-null T-ALL; "
            "  CDK4/6i mechanism: activate pRB to arrest E2F — no pRB = no target = INACTIVE; "
            "  Venetoclax (BCL2i) ACTIVE regardless RB1 status; "
            "RT — SECONDARY MALIGNANCY RISK: "
            "  Avoid high-dose RT at T-ALL treatment site in Rb survivors (already had prior orbital/cranial RT); "
            "  Cumulative RT risk: pRB-deficient cells in irradiated fields → secondary sarcoma/T-ALL risk; "
            "  WBMRI Toronto protocol for secondary malignancy surveillance in hereditary Rb; "
            "TREATMENT: "
            "  Hereditary Rb-associated T-ALL: standard T-ALL chemotherapy; "
            "  Nelarabine FDA-approved T-ALL — SAFE in RB1 (no DNA repair defect); "
            "  Minimise cumulative RT at T-ALL site (prior RT field consideration); "
            "  CDK4/6 inhibitors INACTIVE; venetoclax preferred if BCL2 positive; "
            "SURVEILLANCE: "
            "  Hereditary Rb: ophthalmological exam under anaesthesia every 4-6wk infancy; "
            "  Annual physical exam for secondary malignancies; WBMRI Toronto annually; "
            "  CBC + T-cell immunophenotyping annually from age 10yr (secondary T-ALL surveillance)"
        ),
        "syndrome": "Hereditary Retinoblastoma (OMIM 180200) / bilateral Rb PATHOGNOMONIC / secondary T-ALL/T-LBL post-RT 5x / CDK4/6 inhibitor INACTIVE RB1-null / WBMRI secondary malignancy surveillance",
        "inheritance": "AD LOF",
        "all_risk": "RB1 germline LOF: secondary T-ALL/T-LBL post-RT 5x (therapy-related T-ALL arising 5-15yr after orbital/cranial RT for primary Rb); bilateral retinoblastoma PATHOGNOMONIC (>95% bilateral = germline RB1); CDK4/6 inhibitors INACTIVE RB1-null (no pRB target); venetoclax ACTIVE regardless; nelarabine FDA-approved T-ALL safe RB1; avoid high-dose RT at T-ALL site (cumulative RT risk secondary malignancy); WBMRI Toronto annually secondary malignancy surveillance",
        "pathognomonic": "Bilateral retinoblastoma (leukocoria both eyes) = PATHOGNOMONIC germline RB1 LOF (>95%); trilateral Rb (bilateral Rb + pineoblastoma) = PATHOGNOMONIC — HIGHEST secondary T-ALL risk; white pupil reflex (leukocoria) = emergency ophthalmological referral PATHOGNOMONIC Rb; child with T-ALL/T-LBL 5-15yr after Rb orbital/cranial RT = therapy-related secondary T-ALL PATHOGNOMONIC RB1; RB1 somatic del13q14 in T-ALL blasts = biallelic RB1 inactivation; CDK4/6 inhibitor palbociclib INACTIVE = RB1-null confirmed (no tumour response expected)",
        "key_avoid": "Do NOT use CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib) in RB1-null T-ALL (INACTIVE — pRB is the CDK4/6i target; RB1-null = no target); avoid high-dose RT at T-ALL treatment field in Rb survivors (cumulative RT risk secondary sarcoma/T-ALL in irradiated RB1-deficient field); do NOT miss bilateral Rb (PATHOGNOMONIC — emergency ophthalmology, do NOT delay); do NOT delay WBMRI Toronto annually hereditary Rb (secondary malignancy surveillance — sarcoma, T-ALL/T-LBL, other); do NOT confuse secondary therapy-related T-ALL (post-RT RB1) with de novo T-ALL (different treatment/germline implications)",
        "key_rule": "RB1 T-ALL key rules: (1) bilateral Rb = PATHOGNOMONIC germline RB1 LOF — emergency ophthalmology referral; (2) secondary T-ALL/T-LBL post-RT: 5x elevation 5-15yr after orbital/cranial RT for primary Rb; (3) CDK4/6 inhibitors (palbociclib/ribociclib) INACTIVE RB1-null — no target; (4) venetoclax ACTIVE regardless RB1 status; (5) nelarabine FDA-approved T-ALL — SAFE RB1 (no DNA repair defect); (6) minimise cumulative RT at T-ALL site (prior Rb RT field — secondary malignancy risk); (7) WBMRI Toronto annually hereditary Rb secondary malignancy surveillance; (8) cascade testing MANDATORY.",
        "surveillance": "Hereditary Rb: ophthalmological exam under anaesthesia every 4-6wk infancy; WBMRI Toronto annually (secondary sarcoma/T-ALL/T-LBL surveillance); annual physical exam for secondary malignancies; CBC + T-cell immunophenotyping annually from age 10yr (secondary T-ALL); bone marrow aspirate if CBC shows blasts; RB1 germline panel in bilateral Rb; skull MRI at Rb diagnosis (pineoblastoma trilateral Rb); cascade testing first-degree relatives; bone survey/MRI for bone sarcoma surveillance (RT-field secondary)",
        "targeted_rx": "Nelarabine (FDA-approved T-ALL/T-LBL — SAFE RB1; avoid if prior extensive cranial RT); venetoclax (BCL2i — ACTIVE regardless RB1 status); standard T-ALL protocols (COG AALL0434, BFM 2000); bortezomib (T-ALL); AVOID CDK4/6 inhibitors (palbociclib/ribociclib — INACTIVE RB1-null); minimise cumulative RT (prior Rb RT field — secondary malignancy); allogeneic SCT (high-risk secondary T-ALL in hereditary Rb — non-TBI conditioning preferred); cascade testing MANDATORY hereditary Rb",
    },
    {
        "gene": "ETV6",
        "protein": (
            "ETV6 -- 12p13.2 AD LOF -- 1236aa -- "
            "ETV6-57kDa-ETS-Transcription-Factor-"
            "Pointed-Domain-DNA-Binding-THC5-"
            "T-ALL-Also-Elevated-Not-Only-B-ALL-"
            "Thrombocytopenia-Co-Phenotype-AVOID-NSAIDs-"
            "ETV6-RUNX1-t12-21-Somatic-DISTINCT-Germline-OMIM-616216"
        ),
        "locus": "12p13.2",
        "protein_size": (
            "1236 aa / ~57 kDa / 12p13.2 ETV6 T-ALL molecular context: "
            "STRUCTURE: "
            "  1236 aa / ~57 kDa; Pointed (PNT) domain (aa 43-124) — ETS family dimerisation; "
            "  Central linker (aa 125-330) — regulatory; "
            "  ETS DNA-binding domain (aa 335-416) — sequence-specific DNA binding; "
            "  ETV6 represses transcription of target genes (VEGFR, PDGFRB, KIT) via Pointed-domain homodimerisation; "
            "  ETV6 LOF → derepression of haematopoietic expansion signals → lymphoid precursor expansion; "
            "THC5 SYNDROME AND T-ALL: "
            "  Thrombocytopenia-5 (THC5; OMIM 616216): germline ETV6 LOF → thrombocytopenia + ALL predisposition; "
            "  ETV6 germline LOF: B-ALL 10-15x MOST COMMON hereditary ALL (best known); "
            "  T-ALL also elevated (less prominent than B-ALL but CLINICALLY SIGNIFICANT — not B-ALL exclusive); "
            "  ETV6 is haematopoietic master regulator — LOF affects both B- and T-lineage progenitors; "
            "  T-ALL in THC5: TdT+CD3+(cytopl)+CD5+CD7+ immunophenotype; "
            "  ETV6 germline families: thrombocytopenia clustering + B-ALL AND T-ALL cases in kindred; "
            "THROMBOCYTOPENIA — CO-PHENOTYPE: "
            "  THC5 thrombocytopenia: platelet count 50-120 × 10^9/L lifelong; MPV normal or elevated; "
            "  Thrombocytopenia precedes ALL by years in ETV6 carriers; "
            "  AVOID NSAIDs in THC5 thrombocytopenia (aspirin, ibuprofen, naproxen — platelet dysfunction); "
            "  Platelet transfusion for invasive procedures THC5; "
            "ETV6::RUNX1 SOMATIC — CRITICAL DISTINCTION: "
            "  t(12;21)(p13;q22) ETV6::RUNX1 somatic = MOST COMMON somatic fusion childhood B-ALL (~25%); "
            "  Somatic ETV6::RUNX1 produces B-ALL predominantly (NOT T-ALL); "
            "  Germline ETV6 LOF ≠ somatic ETV6::RUNX1 — COMPLETELY DIFFERENT entities; "
            "  Germline ETV6 LOF predisposes BOTH B-ALL AND T-ALL (broader lineage effect); "
            "  THC5 thrombocytopenia family history needed to distinguish germline LOF from somatic fusion; "
            "T-ALL TREATMENT ETV6: "
            "  Standard T-ALL protocols (COG AALL0434 T-ALL arm, BFM 2000/2017); "
            "  Nelarabine FDA-approved T-ALL — SAFE ETV6 (no DNA repair defect); "
            "  Avoid NSAIDs/aspirin (THC5 thrombocytopenia); platelet support for procedures; "
            "  No ETV6-specific targeted therapy (2026); cascade testing MANDATORY; "
            "SURVEILLANCE: "
            "  Annual CBC from birth (thrombocytopenia + blast surveillance); "
            "  T-cell immunophenotyping (TdT+CD3+CD5+CD7+) if CBC concerning; "
            "  Cascade testing first-degree relatives MANDATORY (50% risk THC5 + ALL predisposition)"
        ),
        "syndrome": "THC5 / T-ALL also elevated (not only B-ALL) / thrombocytopenia co-phenotype / AVOID NSAIDs / ETV6::RUNX1 somatic t(12;21) DISTINCT from germline ETV6 LOF",
        "inheritance": "AD LOF",
        "all_risk": "ETV6 germline LOF THC5: T-ALL also elevated (BOTH B-ALL 10-15x AND T-ALL); thrombocytopenia co-phenotype (platelet 50-120 × 10^9/L lifelong); AVOID NSAIDs (aspirin/ibuprofen — platelet dysfunction THC5); ETV6::RUNX1 somatic t(12;21) DISTINCT entity (B-ALL somatic good prognosis — NOT T-ALL); nelarabine FDA-approved T-ALL safe ETV6; cascade testing MANDATORY",
        "pathognomonic": "THC5: thrombocytopenia (platelet 50-120 × 10^9/L) + T-ALL (or B-ALL) in child + family history ETV6 LOF = PATHOGNOMONIC germline ETV6; germline ETV6 + T-ALL (not only B-ALL) in kindred = broader lineage predisposition; somatic ETV6::RUNX1 t(12;21) = B-ALL fusion (NOT T-ALL) — germline ETV6 LOF must be distinguished by thrombocytopenia family history; ETV6 germline panel in ALL families with thrombocytopenia regardless of lineage (B or T); platelet count 50-120 × 10^9/L from birth in THC5 carriers",
        "key_avoid": "AVOID NSAIDs/aspirin in THC5 (thrombocytopenia — platelet dysfunction risk); do NOT confuse somatic ETV6::RUNX1 t(12;21) (B-ALL somatic fusion — good prognosis, B-lineage) with germline ETV6 LOF (hereditary T-ALL AND B-ALL predisposition); do NOT withhold germline ETV6 testing in T-ALL families with thrombocytopenia (T-ALL also predisposed — not only B-ALL); do NOT miss cascade testing in ETV6 family (50% inheritance risk — thrombocytopenia + ALL); do NOT use ETV6::RUNX1 somatic FISH result in T-ALL tumour to infer germline ETV6 status",
        "key_rule": "ETV6 T-ALL key rules: (1) T-ALL also elevated (germline ETV6 predisposes BOTH B-ALL AND T-ALL — not B-ALL exclusive); (2) AVOID NSAIDs/aspirin (THC5 thrombocytopenia — platelet dysfunction); (3) somatic ETV6::RUNX1 t(12;21) = B-ALL fusion (NOT T-ALL) — DISTINCT from germline ETV6 LOF; (4) nelarabine FDA-approved T-ALL — SAFE ETV6 (no DNA repair defect); (5) cascade testing MANDATORY first-degree relatives (50% risk THC5 + ALL predisposition); (6) no ETV6-specific targeted T-ALL therapy (2026).",
        "surveillance": "Annual CBC from birth (thrombocytopenia + blast surveillance); T-cell immunophenotyping (TdT+CD3+CD5+CD7+) if CBC concerning; ETV6 germline panel for T-ALL + B-ALL families with thrombocytopenia; somatic ETV6::RUNX1 FISH/RT-PCR on B-ALL tumour (prognostic, NOT germline); platelet count monitoring THC5 lifelong; cascade testing first-degree relatives; paediatric oncology referral if T-ALL diagnosed; avoid NSAIDs in THC5 carriers lifelong",
        "targeted_rx": "Standard T-ALL protocols (COG AALL0434 T-ALL arm, BFM 2000/2017); nelarabine (FDA-approved T-ALL/T-LBL — SAFE ETV6); platelet support THC5 for procedures; AVOID NSAIDs/aspirin (THC5 thrombocytopenia); bortezomib (T-ALL investigational); gamma-secretase inhibitors (NOTCH1 T-ALL); venetoclax (BCL2i); blinatumomab (CD3/CD19 — check lineage); tisagenlecleucel (relapsed CD19+ only — B-ALL lineage; check if T-ALL); cascade testing MANDATORY ETV6 families",
    },
]


def _make_patient(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)

    # Gene-specific T-ALL characteristics
    t_all_params = {
        "ATM":    dict(age_mu=10, age_sd=5,  all_prob=0.75, t_lbl_prob=0.20, del9p21_prob=0.25,
                       notch1_prob=0.55, craniospinal_rt_prob=0.05, hrd_prob=0.10,
                       msi_high_prob=0.03, nelarabine_eligible=True),
        "TP53":   dict(age_mu=22, age_sd=10, all_prob=0.40, t_lbl_prob=0.15, del9p21_prob=0.20,
                       notch1_prob=0.60, craniospinal_rt_prob=0.03, hrd_prob=0.08,
                       msi_high_prob=0.04, nelarabine_eligible=True),
        "CDKN2A": dict(age_mu=30, age_sd=12, all_prob=0.50, t_lbl_prob=0.20, del9p21_prob=0.80,
                       notch1_prob=0.65, craniospinal_rt_prob=0.18, hrd_prob=0.06,
                       msi_high_prob=0.05, nelarabine_eligible=True),
        "PTEN":   dict(age_mu=25, age_sd=12, all_prob=0.35, t_lbl_prob=0.20, del9p21_prob=0.15,
                       notch1_prob=0.55, craniospinal_rt_prob=0.20, hrd_prob=0.05,
                       msi_high_prob=0.05, nelarabine_eligible=True),
        "NF1":    dict(age_mu=14, age_sd=7,  all_prob=0.45, t_lbl_prob=0.35, del9p21_prob=0.18,
                       notch1_prob=0.50, craniospinal_rt_prob=0.22, hrd_prob=0.04,
                       msi_high_prob=0.04, nelarabine_eligible=True),
        "NBN":    dict(age_mu=10, age_sd=5,  all_prob=0.70, t_lbl_prob=0.25, del9p21_prob=0.22,
                       notch1_prob=0.52, craniospinal_rt_prob=0.05, hrd_prob=0.18,
                       msi_high_prob=0.03, nelarabine_eligible=True),
        "RB1":    dict(age_mu=15, age_sd=8,  all_prob=0.40, t_lbl_prob=0.30, del9p21_prob=0.15,
                       notch1_prob=0.50, craniospinal_rt_prob=0.20, hrd_prob=0.05,
                       msi_high_prob=0.03, nelarabine_eligible=True),
        "ETV6":   dict(age_mu=8,  age_sd=4,  all_prob=0.55, t_lbl_prob=0.18, del9p21_prob=0.20,
                       notch1_prob=0.55, craniospinal_rt_prob=0.25, hrd_prob=0.05,
                       msi_high_prob=0.03, nelarabine_eligible=True),
    }
    p = t_all_params[gene]

    # Age distribution with gene-specific ranges
    age_raw = rng.gauss(p["age_mu"], p["age_sd"])
    if gene in ("ATM", "NBN", "ETV6"):
        age = max(2, min(30, round(age_raw, 1)))
    elif gene in ("NF1", "RB1"):
        age = max(3, min(35, round(age_raw, 1)))
    elif gene == "TP53":
        age = max(10, min(50, round(age_raw, 1)))
    elif gene == "CDKN2A":
        age = max(15, min(60, round(age_raw, 1)))
    elif gene == "PTEN":
        age = max(10, min(55, round(age_raw, 1)))
    else:
        age = max(2, min(55, round(age_raw, 1)))

    has_t_all          = rng.random() < p["all_prob"]
    t_lbl              = rng.random() < p["t_lbl_prob"]  # T-LBL vs T-ALL distinction
    del9p21            = rng.random() < p["del9p21_prob"]
    notch1_somatic     = rng.random() < p["notch1_prob"]
    craniospinal_rt    = rng.random() < p["craniospinal_rt_prob"]
    hrd_positive       = rng.random() < p["hrd_prob"]
    msi_high           = rng.random() < p["msi_high_prob"]
    nelarabine_elig    = p["nelarabine_eligible"] and has_t_all

    # T-ALL immunophenotype detail
    immuno_stages = ["ETP-ALL", "early-T", "cortical-T", "mature-T", "T-LBL"]
    if gene == "ATM":
        # A-T: early T-cell predominant (ETP-ALL uncommon; cortical T common)
        stage_probs = [0.10, 0.20, 0.45, 0.15, 0.10]
    elif gene == "PTEN":
        # PTEN: early T + ETP-ALL (PI3K pathway early T-cell)
        stage_probs = [0.18, 0.22, 0.35, 0.15, 0.10]
    elif gene == "NF1":
        # NF1: T-LBL predominant (mediastinal T-LBL most common NF1 T-lineage)
        stage_probs = [0.05, 0.15, 0.30, 0.15, 0.35]
    elif gene == "RB1":
        # RB1: cortical T + T-LBL (secondary post-RT)
        stage_probs = [0.08, 0.15, 0.35, 0.15, 0.27]
    else:
        stage_probs = [0.10, 0.18, 0.42, 0.20, 0.10]

    cumulative = 0.0
    r = rng.random()
    t_stage = immuno_stages[-1]
    for st, sp in zip(immuno_stages, stage_probs):
        cumulative += sp
        if r < cumulative:
            t_stage = st
            break

    return {
        "gene":             gene,
        "seed":             seed,
        "age_at_dx":        age,
        "has_t_all":        has_t_all,
        "t_stage":          t_stage,
        "t_lbl":            t_lbl,
        "del9p21":          del9p21,
        "notch1_somatic":   notch1_somatic,
        "craniospinal_rt":  craniospinal_rt,
        "hrd_positive":     hrd_positive,
        "msi_high":         msi_high,
        "nelarabine_elig":  nelarabine_elig,
    }


def _build_cohorts() -> dict:
    cohorts = {}
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        cohorts[gene] = [_make_patient(gene, SEED_BASE + i + j * 8) for j in range(40)]
    return cohorts


def generate_overview() -> dict:
    cohorts = _build_cohorts()
    all_pts = [p for pts in cohorts.values() for p in pts]

    t_all_n            = sum(1 for p in all_pts if p["has_t_all"])
    t_lbl_n            = sum(1 for p in all_pts if p["t_lbl"])
    del9p21_n          = sum(1 for p in all_pts if p["del9p21"])
    notch1_n           = sum(1 for p in all_pts if p["notch1_somatic"])
    craniospinal_n     = sum(1 for p in all_pts if p["craniospinal_rt"])
    hrd_n              = sum(1 for p in all_pts if p["hrd_positive"])
    nelarabine_n       = sum(1 for p in all_pts if p["nelarabine_elig"])
    mean_age           = round(sum(p["age_at_dx"] for p in all_pts) / len(all_pts), 1)

    gene_summary = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        t_all_gene_n = sum(1 for p in pts if p["has_t_all"])
        t_lbl_gene_n = sum(1 for p in pts if p["t_lbl"])
        gene_summary.append({
            "gene":              gene,
            "locus":             gd["locus"],
            "n":                 len(pts),
            "t_all_n":           t_all_gene_n,
            "t_all_pct":         round(100 * t_all_gene_n / len(pts), 1),
            "t_lbl_n":           t_lbl_gene_n,
            "t_lbl_pct":         round(100 * t_lbl_gene_n / len(pts), 1),
            "mean_age_onset":    round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "del9p21_n":         sum(1 for p in pts if p["del9p21"]),
            "notch1_n":          sum(1 for p in pts if p["notch1_somatic"]),
            "craniospinal_rt_n": sum(1 for p in pts if p["craniospinal_rt"]),
            "hrd_positive_n":    sum(1 for p in pts if p["hrd_positive"]),
            "nelarabine_elig_n": sum(1 for p in pts if p["nelarabine_elig"]),
            "inheritance":       gd["inheritance"],
            "key_distinctions":  [
                gd["all_risk"][:80],
                gd["key_rule"][:80],
            ],
        })

    genes_detail = []
    for gd in ATLAS_GENES:
        genes_detail.append({
            "gene":             gd["gene"],
            "protein":          gd["protein"],
            "locus":            gd["locus"],
            "protein_size":     gd["protein_size"],
            "inheritance":      gd["inheritance"],
            "cancer_risk":      gd["all_risk"],
            "syndrome":         gd["syndrome"],
            "pathognomonic":    gd["pathognomonic"],
            "surveillance_key": gd["surveillance"][:160],
            "key_rule":         gd["key_rule"],
            "key_avoid":        gd["key_avoid"],
        })

    pathway_categories = [
        {"pathway": "DNA Damage Response (DDR)", "genes": ["ATM", "NBN"], "mechanism": "V(D)J TCR recombination DSB mis-repair → T-ALL chromosomal translocations"},
        {"pathway": "TP53 / Cell Cycle Checkpoint", "genes": ["TP53", "CDKN2A", "RB1"], "mechanism": "p53/pRB/p16 inactivation → G1/S checkpoint loss → T-lymphoid proliferation"},
        {"pathway": "PI3K/AKT/mTOR", "genes": ["PTEN"], "mechanism": "PTEN LOF → constitutive PI3K/AKT/mTOR → T-cell proliferation; cooperates with NOTCH1"},
        {"pathway": "Ras/MAPK", "genes": ["NF1"], "mechanism": "Neurofibromin LOF → Ras-GTP constitutive → MEK/ERK → T-cell lineage expansion → T-LBL/T-ALL"},
        {"pathway": "ETS Transcription Factor / Haematopoiesis", "genes": ["ETV6"], "mechanism": "ETV6 LOF → derepression of haematopoietic expansion signals → B-ALL (primary) AND T-ALL (secondary predisposition)"},
    ]

    critical_distinctions = [
        "T-LBL vs T-ALL: <25% bone marrow blasts = T-LBL (lymphoma, treated as lymphoma); ≥25% bone marrow blasts = T-ALL (leukaemia, treated as ALL); distinction MANDATORY — staging and treatment protocol choice depends on BM blast percentage; NF1 and RB1 most frequently present as T-LBL (mediastinal); ATM A-T and NBN NBS most frequently T-ALL (BM-predominant)",
        "NOTCH1 somatic (60-70% T-ALL) is ACQUIRED and DISTINCT from germline predisposition genes: NOTCH1 activating mutations are the most common SOMATIC driver in T-ALL but do NOT predispose hereditarily; germline risk genes are ATM/TP53/CDKN2A/PTEN/NF1/NBN/RB1/ETV6; NOTCH1 somatic + CDKN2A del9p21 = common co-profile in sporadic T-ALL (NOT hereditary unless germline CDKN2A LOF confirmed)",
        "Nelarabine FDA-approved T-ALL/T-LBL (T-lineage specific purine analogue): SAFE in most hereditary T-ALL predisposition syndromes; AVOID nelarabine in CNS T-ALL patients with prior cranial RT (neurotoxicity risk amplified); nelarabine + intrathecal cytarabine neurotoxicity risk — monitor closely; nelarabine is T-lineage selective (not used in B-ALL or AML)",
        "RADIOSENSITIVITY ABSOLUTE genes in T-ALL: ATM biallelic A-T (T-ALL 75x HIGHEST) and NBN biallelic NBS (ALL 60x T-ALL predominant) — NO RT, NO CRT, NO TBI; IT triple therapy (MTX + cytarabine + hydrocortisone) REPLACES cranial RT; busulfan-based non-TBI conditioning if SCT required; AVOID RADIATION ABSOLUTELY TP53 LFS (secondary malignancies catastrophic); RT reduce 20-30% ATM monoallelic and NBN monoallelic",
        "CDK4/6 inhibitor INACTIVE rule in T-ALL: palbociclib/ribociclib/abemaciclib INACTIVE in CDKN2A-null (p16-null) T-ALL (CDK4/6 already unrestrained — no target) AND in RB1-null T-ALL (pRB is the CDK4/6i endpoint — RB1-null = no target); venetoclax (BCL2i) ACTIVE regardless of CDKN2A or RB1 status in T-ALL; verify p16 IHC and RB1 IHC before prescribing CDK4/6 inhibitors",
        "Alkylating agent rules in hereditary T-ALL: AVOID alkylating agents biallelic NBN NBS (profound DNA repair defect — aplasia risk; cyclophosphamide, melphalan, busulfan dose reduce or avoid biallelic); monoallelic NBN: standard T-ALL protocols with monitoring; ATM biallelic: busulfan-based SCT conditioning only; ETV6/TP53/CDKN2A/PTEN/NF1/RB1: standard T-ALL chemotherapy backbone (no alkylating agent restriction specific to these genes)",
        "ETV6 germline LOF (THC5): T-ALL also elevated — NOT B-ALL exclusive: germline ETV6 LOF predisposes BOTH B-ALL (10-15x, most common) AND T-ALL (also elevated — haematopoietic master regulator affects both lineages); somatic ETV6::RUNX1 t(12;21) is B-ALL specific (good prognosis somatic fusion — distinct from germline ETV6 LOF); thrombocytopenia (platelet 50-120 × 10^9/L) + T-ALL in child = test ETV6 germline; AVOID NSAIDs in THC5 (platelet dysfunction)",
        "TBI conditioning contraindications in hereditary T-ALL: ABSOLUTELY CONTRAINDICATED in ATM biallelic A-T (RADIOSENSITIVITY ABSOLUTE); ABSOLUTELY CONTRAINDICATED in NBN biallelic NBS (RADIOSENSITIVITY ABSOLUTE); ABSOLUTELY CONTRAINDICATED in TP53 LFS (catastrophic secondary malignancies); RELATIVE CONTRAINDICATION in RB1 hereditary Rb (cumulative RT secondary malignancy risk); busulfan-based non-TBI conditioning is the safe alternative for all four",
        "V(D)J TCR recombination DSB hotspots and T-ALL: TCRA/TCRD (14q11.2), TCRB (7q34), TCRG (7p14) are chromosomal fragile sites during T-cell development; ATM and NBN are essential for repairing RAG1/RAG2-generated DSBs at these loci; ATM LOF → t(14;14)/inv(14)/t(7;14) characteristic translocations; NBN LOF → inv(7)(p13q35)/t(7;14)(q35;q11.2); these gene-specific translocations help diagnose A-T and NBS underlying T-ALL",
        "Hereditary T-ALL germline panel: T-ALL in context of ataxia/telangiectasia → ATM FIRST; T-ALL with macrocephaly/hamartomas → PTEN; T-ALL with café-au-lait macules/neurofibromas → NF1; T-ALL post-Rb orbital RT → RB1; T-ALL in Eastern European child + microcephaly → NBN 657del5; T-ALL in LFS family → TP53; T-ALL with FAMM melanoma family → CDKN2A; T-ALL + thrombocytopenia → ETV6; cascade testing MANDATORY first-degree relatives all eight genes; annual CBC + TdT/CD3/CD5/CD7 immunophenotyping from early childhood in at-risk carriers",
    ]

    return {
        "atlas":                    "Hereditary-T-ALL-Predisposition-Atlas",
        "subtitle":                 "8-Gene T-ALL/T-LBL Predisposition Reference — 320-Patient Aggregate",
        "genes":                    [gd["gene"] for gd in ATLAS_GENES],
        "seed_range":               f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":           len(all_pts),
        "genes_n":                  8,
        "t_all_total_n":            t_all_n,
        "t_all_total_pct":          round(100 * t_all_n / len(all_pts), 1),
        "mean_age_onset":           mean_age,
        "del9p21_n":                del9p21_n,
        "del9p21_pct":              round(100 * del9p21_n / len(all_pts), 1),
        "notch1_somatic_n":         notch1_n,
        "notch1_somatic_pct":       round(100 * notch1_n / len(all_pts), 1),
        "t_lbl_n":                  t_lbl_n,
        "t_lbl_pct":                round(100 * t_lbl_n / len(all_pts), 1),
        "craniospinal_rt_n":        craniospinal_n,
        "craniospinal_rt_pct":      round(100 * craniospinal_n / len(all_pts), 1),
        "hrd_positive_n":           hrd_n,
        "hrd_positive_pct":         round(100 * hrd_n / len(all_pts), 1),
        "nelarabine_eligible_n":    nelarabine_n,
        "nelarabine_eligible_pct":  round(100 * nelarabine_n / len(all_pts), 1),
        "gene_summary":             gene_summary,
        "genes_detail":             genes_detail,
        "pathway_categories":       pathway_categories,
        "critical_distinctions":    critical_distinctions,
        "gene_n":                   {gd["gene"]: 40 for gd in ATLAS_GENES},
    }


def generate_breakdown() -> dict:
    cohorts = _build_cohorts()
    _gene_pathog = {gd["gene"]: gd["pathognomonic"] for gd in ATLAS_GENES}
    _gene_surv   = {gd["gene"]: gd["surveillance"][:120] for gd in ATLAS_GENES}
    _gene_rule   = {gd["gene"]: gd["key_rule"] for gd in ATLAS_GENES}

    breakdown_array = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        t_all_n  = sum(1 for p in pts if p["has_t_all"])
        t_lbl_n  = sum(1 for p in pts if p["t_lbl"])
        breakdown_array.append({
            "gene":              gene,
            "locus":             gd["locus"],
            "n":                 len(pts),
            "mean_age_onset":    round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "all_pct":           round(100 * t_all_n / len(pts), 1),
            "t_lbl_pct":         round(100 * t_lbl_n / len(pts), 1),
            "del9p21_n":         sum(1 for p in pts if p["del9p21"]),
            "notch1_n":          sum(1 for p in pts if p["notch1_somatic"]),
            "craniospinal_rt_n": sum(1 for p in pts if p["craniospinal_rt"]),
            "hrd_positive_n":    sum(1 for p in pts if p["hrd_positive"]),
            "nelarabine_elig_n": sum(1 for p in pts if p["nelarabine_elig"]),
            "seed":              SEED_BASE + i,
            "pathognomonic":     _gene_pathog[gene],
            "surveillance_key":  _gene_surv[gene],
            "key_rule":          _gene_rule[gene],
        })
    return {
        "breakdown":       breakdown_array,
        "total_patients":  sum(len(v) for v in cohorts.values()),
    }


def generate_definitions() -> dict:
    defs = {}
    for gd in ATLAS_GENES:
        defs[gd["gene"]] = {
            "gene":          gd["gene"],
            "locus":         gd["locus"],
            "protein_size":  gd["protein_size"],
            "syndrome":      gd["syndrome"],
            "inheritance":   gd["inheritance"],
            "all_risk":      gd["all_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_avoid":     gd["key_avoid"],
            "key_rule":      gd["key_rule"],
            "surveillance":  gd["surveillance"],
            "targeted_rx":   gd["targeted_rx"],
        }

    key_clinical_distinctions = [
        "T-LBL vs T-ALL: <25% bone marrow blasts = T-LBL (lymphoma); ≥25% = T-ALL (leukaemia) — MANDATORY staging distinction; NF1 and RB1 most frequently T-LBL (mediastinal); ATM A-T and NBN NBS predominantly T-ALL; different protocols apply (lymphoma vs leukaemia backbone); both treated with T-lineage active regimens including nelarabine",
        "Nelarabine FDA approved T-ALL/T-LBL (T-lineage specific): safe in ATM (biallelic A-T), TP53 (LFS), CDKN2A (FAMM), PTEN (Cowden), NF1, NBN (biallelic NBS), RB1, ETV6 — no DNA repair CI in most genes; AVOID nelarabine in CNS T-ALL with prior cranial RT (neurotoxicity amplified); nelarabine + IT cytarabine neurotoxicity risk — monitor closely; T-lineage selective — not for B-ALL or AML",
        "RADIOSENSITIVITY ABSOLUTE — RT NEVER in ATM biallelic A-T and NBN biallelic NBS T-ALL: no cranial RT, no craniospinal RT, no TBI conditioning; IT triple therapy replaces CRT; busulfan non-TBI conditioning for SCT; AVOID RADIATION ABSOLUTELY TP53 LFS; RT reduce 20-30% ATM monoallelic and NBN monoallelic; RB1 hereditary Rb: minimise cumulative RT at T-ALL site; PTEN/NF1/CDKN2A/ETV6: no specific RT constraint beyond standard T-ALL protocols",
        "NOTCH1 somatic (60-70% T-ALL) DISTINCT from hereditary T-ALL: NOTCH1 activating somatic mutations are the most prevalent acquired T-ALL driver; NOTCH1 somatic ≠ hereditary predisposition gene; gamma-secretase inhibitors (GSI) target NOTCH1 pathway; CDKN2A del9p21 + NOTCH1 somatic co-occurrence = common T-ALL co-profile; PTEN somatic loss cooperates with NOTCH1 in T-ALL (PI3K + HES1 compounded); germline testing required separately from somatic panel",
        "CDK4/6 inhibitor INACTIVE: p16INK4a-null T-ALL (CDKN2A germline LOF or somatic del9p21 homozygous) → palbociclib/ribociclib INACTIVE (CDK4/6 already unrestrained — no target); RB1-null T-ALL → CDK4/6 inhibitors INACTIVE (pRB is the endpoint — RB1-null = no downstream brake); venetoclax (BCL2i) ACTIVE regardless of CDKN2A or RB1 status; confirm p16 IHC and RB1 IHC before CDK4/6 inhibitor consideration in any T-ALL",
        "ATM biallelic A-T T-ALL 75x HIGHEST — T-lineage PREDOMINANCE mechanism: ATM essential for repairing RAG1/RAG2-generated DSBs during V(D)J recombination at TCRA/TCRD (14q11.2), TCRB (7q34), TCRG (7p14); ATM LOF → unresolved TCR DSBs → chromosomal translocations t(14;14)/inv(14)/t(7;14) → T-ALL; IgA deficiency + AFP elevated + cerebellar ataxia = A-T PATHOGNOMONIC triad; monoallelic ATM T-ALL modest (2-3x); RADIOSENSITIVITY ABSOLUTE biallelic",
        "NBN 657del5 Slavic founder (1:150 Czech/Slovak/Polish): biallelic NBS ALL 60x T-ALL PREDOMINANT; cranial RT ABSOLUTELY CONTRAINDICATED biallelic (RADIOSENSITIVITY ABSOLUTE + microcephaly neurotoxicity + secondary malignancy); AVOID alkylating agents biallelic; monoallelic 657del5 T-ALL elevated (test FIRST in Eastern European T-ALL); inv(7)/t(7;14) characteristic NBS T-cell chromosomal hotspot; nelarabine SAFE if no prior CRT; PARP inhibitor sensitivity MRN HRD",
        "RB1 hereditary Rb secondary T-ALL/T-LBL: therapy-related T-ALL arising 5-15yr after orbital/cranial RT for primary Rb; bilateral retinoblastoma = PATHOGNOMONIC germline RB1 (>95%); CDK4/6 inhibitors INACTIVE RB1-null (no pRB target); venetoclax ACTIVE; minimise cumulative RT at T-ALL treatment site (RB1-deficient cells in irradiated fields → secondary sarcoma); WBMRI Toronto annually hereditary Rb secondary malignancy surveillance",
        "PTEN Cowden/PHTS T-ALL ~5x via PI3K/mTOR: macrocephaly (OFC >2 SD) = PATHOGNOMONIC PTEN germline; Lhermitte-Duclos disease (striped cerebellum MRI) = PATHOGNOMONIC PTEN; everolimus (mTOR inhibitor) ACTIVE in PTEN-deficient T-ALL; PTEN + NOTCH1 somatic co-loss = compounded mTOR + HES1 activation; AKT/PI3K inhibitors (capivasertib, idelalisib) emerging PTEN-LOF T-ALL; PHTS multi-organ surveillance (thyroid/breast/endometrial/colon/renal) MANDATORY",
        "ETV6 germline LOF T-ALL predisposition — not B-ALL exclusive: germline ETV6 LOF predisposes BOTH B-ALL (10-15x, most prominent) AND T-ALL (haematopoietic master regulator affects T-lineage progenitors too); somatic ETV6::RUNX1 t(12;21) = B-ALL somatic fusion (good prognosis, B-lineage specific — NOT T-ALL); THC5 thrombocytopenia (platelet 50-120 × 10^9/L) + T-ALL in child = test ETV6 germline; AVOID NSAIDs/aspirin in THC5 (platelet dysfunction); cascade testing MANDATORY ETV6 families regardless of ALL lineage",
    ]

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"T-ALL RISK: {gd['all_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-T-ALL-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": key_clinical_distinctions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"Total patients: {ov['total_patients']}, T-ALL n: {ov['t_all_total_n']} ({ov['t_all_total_pct']}%)")
    print(f"del9p21: {ov['del9p21_n']} ({ov['del9p21_pct']}%), NOTCH1 somatic: {ov['notch1_somatic_n']} ({ov['notch1_somatic_pct']}%)")
    print(f"T-LBL: {ov['t_lbl_n']} ({ov['t_lbl_pct']}%), craniospinal-RT: {ov['craniospinal_rt_n']} ({ov['craniospinal_rt_pct']}%)")
    print(f"HRD-positive: {ov['hrd_positive_n']} ({ov['hrd_positive_pct']}%), nelarabine-eligible: {ov['nelarabine_eligible_n']} ({ov['nelarabine_eligible_pct']}%)")
    print("=== BREAKDOWN ===")
    br = generate_breakdown()
    for b in br["breakdown"]:
        print(f"  {b['gene']}: T-ALL {b['all_pct']}%, T-LBL {b['t_lbl_pct']}%, age {b['mean_age_onset']}yr, del9p21 {b['del9p21_n']}, NOTCH1 {b['notch1_n']}, CRT {b['craniospinal_rt_n']}")
    print("=== DEFINITIONS ===")
    df = generate_definitions()
    print(f"Genes: {list(df['gene_defs'].keys())}")
    print(f"Key distinctions: {len(df['key_clinical_distinctions'])}")
