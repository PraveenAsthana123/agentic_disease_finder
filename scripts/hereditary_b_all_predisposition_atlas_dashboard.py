#!/usr/bin/env python3
"""Hereditary-B-ALL-Predisposition-Atlas -- Complete 8-Gene Reference
ETV6    (ETS-related transcription factor; 1236aa; 12p13.2; AD LOF;
         Thrombocytopenia-5 (THC5, OMIM 616216); B-ALL 10-15x MOST COMMON hereditary ALL gene;
         ETV6::RUNX1 t(12;21) somatic most common childhood ALL — DISTINCT from germline ETV6 LOF;
         early-onset B-ALL younger children; PAX5 fusion somatic co-occurrence;
         seed SEED_BASE+0).
PAX5    (B-cell specific transcription factor; 391aa; 9p13.2; AD LOF;
         autosomal dominant B-ALL (OMIM 613065); P80R founder mutation;
         40-80% lifetime B-ALL risk HIGHEST — MOST SPECIFIC familial B-ALL gene;
         B-cell differentiation block pro-B → pre-B stage;
         seed SEED_BASE+1).
IKZF1   (Ikaros zinc finger TF; 519aa; 7p12.2; AD haploinsufficiency;
         Philadelphia-like (Ph-like) ALL association; lymphoid lineage commitment regulator;
         somatic del7p12.2 in 30% B-ALL poor prognosis; emerging germline predisposition;
         seed SEED_BASE+2).
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         LFS (OMIM 151623); B-ALL 15x in LFS; T-ALL also elevated;
         AVOID RADIATION ABSOLUTELY (LFS — catastrophic secondary malignancies);
         cranial RT ABSOLUTELY CONTRAINDICATED (LFS second brain tumours);
         seed SEED_BASE+3).
ATM     (Ataxia-telangiectasia mutated; 3056aa; 11q22.3; AD/AR LOF;
         A-T (OMIM 208900) biallelic; monoallelic B-ALL 2-3x;
         biallelic A-T: T-ALL 75x HIGHEST, B-ALL elevated; IgA deficiency PATHOGNOMONIC biallelic;
         RADIOSENSITIVITY ABSOLUTE biallelic; TBI conditioning CI biallelic;
         seed SEED_BASE+4).
CDKN2A  (Cyclin-dependent kinase inhibitor 2A; 156aa; 9p21.3; AD LOF;
         FAMM (OMIM 606719); del9p21 somatic ALL 20-35%;
         germline CDKN2A LOF: B-ALL/T-ALL 5-10x; p16INK4a + p14ARF dual tumour suppressors;
         CDK4/6 inhibitor palbociclib INACTIVE CDKN2A-null ALL; melanoma 28-67x, pancreatic 17-39x;
         seed SEED_BASE+5).
NBN     (Nibrin/NBS1; 754aa; 8q21.3; AD/AR LOF;
         NBS (OMIM 251260); 657del5 Slavic founder (Czech/Slovak/Polish 1:150);
         monoallelic: B-cell ALL 3-5x; biallelic NBS: ALL 60x HIGHEST;
         RADIOSENSITIVITY ABSOLUTE biallelic; cranial RT NBS ABSOLUTELY CONTRAINDICATED;
         AVOID alkylating agents biallelic;
         seed SEED_BASE+6).
BRCA2   (Breast cancer susceptibility gene 2; 3418aa; 13q12.3; AD LOF;
         HBOC (OMIM 612555); FA-D1 biallelic (OMIM 605724);
         biallelic FA-D1: B-ALL/AML HIGHEST RISK in FA pathway (Wilms + AML + ALL);
         monoallelic: B-ALL 2x; AVOID ALKYLATING AGENTS biallelic;
         AVOID DNA-crosslinking chemotherapy biallelic; PARP inhibitor sensitivity monoallelic;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3550-3557)
"""
import random

SEED_BASE = 3550

ATLAS_GENES = [
    {
        "gene": "ETV6",
        "protein": (
            "ETV6 -- 12p13.2 AD LOF -- 1236aa -- "
            "ETV6-57kDa-ETS-Transcription-Factor-"
            "Pointed-Domain-DNA-Binding-THC5-"
            "B-ALL-10-15x-MOST-COMMON-Hereditary-ALL-"
            "ETV6-RUNX1-t12-21-Somatic-DISTINCT-Germline-"
            "Early-Onset-Childhood-B-ALL-OMIM-616216"
        ),
        "locus": "12p13.2",
        "protein_size": (
            "1236 aa / ~57 kDa / 12p13.2 ETV6 B-ALL molecular context: "
            "STRUCTURE: "
            "  1236 aa / ~57 kDa; Pointed (PNT) domain (aa 43-124) — ETS family dimerisation; "
            "  Central linker (aa 125-330) — regulatory; "
            "  ETS DNA-binding domain (aa 335-416) — sequence-specific DNA binding; "
            "  ETV6 represses transcription of target genes (VEGFR, PDGFRB, KIT) via Pointed-domain homodimerisation; "
            "  ETV6 LOF → derepression of haematopoietic expansion signals → B-cell precursor clonal expansion; "
            "THC5 SYNDROME: "
            "  Thrombocytopenia-5 (THC5; OMIM 616216): germline ETV6 LOF → thrombocytopenia + B-ALL predisposition; "
            "  Autosomal dominant; platelet count 50-120 × 10^9/L lifelong; MPV normal or elevated; "
            "  B-ALL risk 10-15x lifetime (MOST COMMON hereditary ALL predisposition gene identified to date); "
            "  ETV6 germline LOF families: multiple B-ALL cases in single kindred + thrombocytopenia clustering; "
            "ETV6::RUNX1 SOMATIC TRANSLOCATION — CRITICAL DISTINCTION: "
            "  t(12;21)(p13;q22) ETV6::RUNX1 somatic = MOST COMMON somatic fusion childhood B-ALL (~25%); "
            "  Somatic ETV6::RUNX1 ≠ germline ETV6 LOF — COMPLETELY DIFFERENT entities; "
            "  Somatic ETV6::RUNX1 = GOOD prognosis, excellent chemotherapy response; "
            "  Germline ETV6 LOF = HEREDITARY PREDISPOSITION — may also develop somatic ETV6::RUNX1 co-occurrence; "
            "B-ALL FEATURES: "
            "  ETV6 germline ALL: early childhood onset (median 3-8yr); B-ALL predominant; "
            "  IKZF1 deletion may co-occur somatically in ETV6-germline ALL; "
            "  ETV6 + PAX5 mutations may cluster in familial B-ALL families; "
            "TREATMENT: "
            "  Standard paediatric B-ALL protocols (BFM, COG AALL0434 variant); "
            "  ETV6::RUNX1-containing ALL (somatic co-occurrence): excellent response, standard risk; "
            "  Thrombocytopenia THC5: platelet transfusion for procedures; avoid aspirin; "
            "SURVEILLANCE: "
            "  Annual CBC from birth (thrombocytopenia monitoring); "
            "  B-ALL surveillance: CBC + lymphocyte immunophenotyping (TdT+CD19+CD10+) from infancy; "
            "  Cascade testing MANDATORY first-degree relatives (50% risk THC5 + B-ALL predisposition); "
            "  Bone marrow aspirate if CBC suggests blast crisis (TdT+CD19+CD10+ blasts on flow)"
        ),
        "syndrome": "THC5 / B-ALL 10-15x MOST COMMON hereditary ALL gene / ETV6::RUNX1 somatic DISTINCT from germline ETV6 LOF",
        "inheritance": "AD LOF",
        "all_risk": "ETV6 germline LOF: B-ALL 10-15x MOST COMMON hereditary ALL predisposition gene; THC5 thrombocytopenia co-phenotype; ETV6::RUNX1 somatic t(12;21) DISTINCT entity (somatic = good prognosis childhood ALL); early childhood onset; cascade testing MANDATORY first-degree relatives",
        "pathognomonic": "THC5: thrombocytopenia (platelet 50-120 × 10^9/L) + B-ALL in child + family history ETV6 LOF = PATHOGNOMONIC germline ETV6; germline ETV6 + somatic ETV6::RUNX1 co-occurrence in same tumour = dual ETV6 hit; ETV6::RUNX1 on somatic FISH/RT-PCR alone ≠ germline ETV6 — thrombocytopenia family history needed to distinguish; ETV6 germline panel in childhood ALL families with thrombocytopenia",
        "key_avoid": "Do NOT confuse somatic ETV6::RUNX1 t(12;21) (good-prognosis somatic fusion) with germline ETV6 LOF (hereditary predisposition); do NOT withhold germline ETV6 testing in childhood B-ALL + family thrombocytopenia; do NOT use aspirin/NSAIDs in THC5 (thrombocytopenia); do NOT miss cascade testing in ETV6 family (50% inheritance risk — thrombocytopenia + B-ALL); do NOT assume ETV6::RUNX1 somatic FISH result in tumour = germline ETV6 status",
        "key_rule": "ETV6 B-ALL key rules: (1) MOST COMMON hereditary ALL gene — test ETV6 germline in familial childhood B-ALL; (2) somatic ETV6::RUNX1 t(12;21) ≠ germline ETV6 LOF — COMPLETELY DIFFERENT (good prognosis somatic vs hereditary predisposition); (3) THC5 thrombocytopenia co-phenotype — cascade test for thrombocytopenia + B-ALL clustering; (4) standard paediatric ALL protocols (no ETV6-specific targeted therapy); (5) cascade testing MANDATORY first-degree relatives from birth.",
        "surveillance": "Annual CBC from birth (thrombocytopenia + blast surveillance); B-cell immunophenotyping (TdT+CD19+CD10+) if CBC concerning; ETV6 germline panel for familial childhood B-ALL + thrombocytopenia; somatic ETV6::RUNX1 FISH/RT-PCR on tumour cells (prognostic, NOT germline testing); platelet count monitoring THC5 lifelong; cascade testing first-degree relatives; paediatric oncology referral if B-ALL diagnosed",
        "targeted_rx": "Standard paediatric B-ALL protocols (COG AALL0434, BFM 2000/2017); ETV6::RUNX1 somatic co-occurrence = standard-risk ALL (excellent prognosis); imatinib/dasatinib if Ph+ component; blinatumomab (CD19+ ALL); tisagenlecleucel (CAR-T CD19 relapsed/refractory); nelarabine (T-ALL component); platelet support THC5; no ETV6-specific targeted therapy (2026); cascade testing MANDATORY",
    },
    {
        "gene": "PAX5",
        "protein": (
            "PAX5 -- 9p13.2 AD LOF -- 391aa -- "
            "PAX5-45kDa-B-Cell-Specific-TF-"
            "Paired-Box-Octapeptide-Homeodomain-"
            "B-ALL-40-80pct-HIGHEST-P80R-Founder-"
            "Familial-B-ALL-MOST-SPECIFIC-"
            "Pro-B-PreB-Differentiation-Block-OMIM-613065"
        ),
        "locus": "9p13.2",
        "protein_size": (
            "391 aa / ~45 kDa / 9p13.2 PAX5 B-ALL molecular context: "
            "STRUCTURE: "
            "  391 aa / ~45 kDa; N-terminal paired box domain (aa 16-142) — bipartite DNA binding (PAI + RED subdomains); "
            "  Octapeptide domain (aa 179-186) — transcriptional repression; "
            "  Partial homeodomain (aa 228-254); "
            "  Transactivation domain (aa 304-391); "
            "  PAX5 activates B-cell lineage genes (CD19, BLNK, IKAROS) and represses non-B-lineage genes; "
            "  PAX5 is the master regulator of B-cell identity — required for pro-B → pre-B differentiation; "
            "FAMILIAL B-ALL: "
            "  PAX5 AD LOF = MOST SPECIFIC germline gene for autosomal dominant familial B-ALL (OMIM 613065); "
            "  P80R (c.239C>G; p.Pro80Arg) in PAI subdomain = FOUNDER MUTATION in familial B-ALL pedigrees; "
            "  P80R impairs DNA binding → partial LOF → B-cell differentiation block pro-B → pre-B; "
            "  40-80% lifetime B-ALL risk — HIGHEST penetrance hereditary ALL gene; "
            "  Familial B-ALL kindreds: multiple B-ALL across generations = PAX5 P80R suspect; "
            "B-CELL DIFFERENTIATION MECHANISM: "
            "  PAX5 LOF → failure to activate B-cell lineage programmes → cells arrest at pro-B stage; "
            "  Pro-B progenitors: TdT+CD34+CD19+CD10-; PAX5-LOF block prevents CD10+ pre-B differentiation; "
            "  Arrested pro-B cells acquire additional mutations (IKZF1, ETV6, CDKN2A del9p21) → frank B-ALL; "
            "  B-ALL in PAX5-LOF patients: B-ALL (CD19+CD10+TdT+), cytogenetics variable; "
            "SOMATIC PAX5 MUTATIONS: "
            "  Somatic PAX5 mutations in 30% sporadic B-ALL (distinct from germline P80R); "
            "  Most frequent somatic target after ETV6 and RUNX1 in paediatric B-ALL; "
            "TREATMENT: "
            "  Standard paediatric B-ALL protocols; CAR-T tisagenlecleucel CD19 relapsed; "
            "  Blinatumomab bispecific T-cell engager CD19×CD3; "
            "  B-ALL MRD monitoring at day 8 / end induction critical; "
            "SURVEILLANCE: "
            "  Annual CBC + B-cell immunophenotyping from early childhood; "
            "  PAX5 germline P80R panel in familial B-ALL; "
            "  Cascade testing MANDATORY first-degree relatives; bone marrow aspirate if blast crisis"
        ),
        "syndrome": "Autosomal dominant familial B-ALL (OMIM 613065) / PAX5 P80R founder mutation / 40-80% lifetime B-ALL HIGHEST / MOST SPECIFIC familial B-ALL gene",
        "inheritance": "AD LOF",
        "all_risk": "PAX5 AD LOF germline: B-ALL 40-80% lifetime HIGHEST penetrance; P80R founder mutation familial B-ALL pedigrees; pro-B → pre-B differentiation block; MOST SPECIFIC familial B-ALL gene (analogous to POT1 for CLL); somatic PAX5 30% sporadic B-ALL DISTINCT from germline P80R; cascade testing MANDATORY",
        "pathognomonic": "Familial B-ALL cluster (≥2 first-degree relatives) + germline PAX5 P80R = PATHOGNOMONIC autosomal dominant familial B-ALL; PAX5 P80R in DNA-binding PAI subdomain + arrested pro-B immunophenotype (TdT+CD34+CD19+CD10-) = PAX5-LOF B-ALL; somatic PAX5 in 30% sporadic ALL ≠ germline PAX5 — pedigree analysis mandatory; B-ALL in successive generations = test PAX5 germline first",
        "key_avoid": "Do NOT assume familial B-ALL is sporadic clustering (PAX5 P80R hereditary, not coincidence); do NOT confuse somatic PAX5 (30% sporadic ALL) with germline PAX5 P80R (familial); do NOT skip cascade testing PAX5 families (40-80% lifetime risk each carrier offspring); do NOT use PAX5 somatic result from ALL tumour to infer germline status — blood germline testing required; do NOT delay diagnosis — early B-ALL detection = curative standard therapy",
        "key_rule": "PAX5 B-ALL key rules: (1) MOST SPECIFIC hereditary B-ALL gene — familial B-ALL cluster → PAX5 germline FIRST; (2) P80R = founder mutation — test in familial B-ALL pedigrees; (3) 40-80% lifetime B-ALL risk HIGHEST — earliest onset possible (toddler); (4) standard B-ALL chemotherapy protocols — no PAX5-specific targeted therapy; (5) somatic PAX5 (30% sporadic) ≠ germline PAX5 — separate entities; (6) cascade testing MANDATORY all first-degree relatives.",
        "surveillance": "Annual CBC + lymphocyte immunophenotyping (TdT+CD19+CD10+) from birth in PAX5 carriers; bone marrow aspirate if CBC shows >5% blasts; PAX5 P80R germline panel in familial B-ALL; somatic FISH/NGS panel at ALL diagnosis (not germline testing); MRD monitoring end-induction critical; cascade testing first-degree relatives; paediatric oncology familial ALL registry enrollment recommended",
        "targeted_rx": "Standard paediatric B-ALL protocols (COG AALL0434/0931, BFM 2017); tisagenlecleucel CAR-T (CD19+ relapsed/refractory ≥25kg); blinatumomab (bispecific CD19×CD3 — MRD-positive or relapsed); inotuzumab ozogamicin (CD22+ relapsed/refractory); dasatinib/imatinib if Ph+ component; venetoclax + steroids (BCL2 B-ALL); nelarabine (T-ALL); no PAX5-specific FDA-approved targeted therapy (2026); cascade testing MANDATORY",
    },
    {
        "gene": "IKZF1",
        "protein": (
            "IKZF1 -- 7p12.2 AD haploinsufficiency -- 519aa -- "
            "IKZF1-Ikaros-58kDa-Zinc-Finger-TF-"
            "Lymphoid-Lineage-Commitment-"
            "Philadelphia-Like-Ph-like-ALL-"
            "Somatic-del7p12-30pct-B-ALL-Poor-Prognosis-"
            "Emerging-Germline-Predisposition-OMIM-603023"
        ),
        "locus": "7p12.2",
        "protein_size": (
            "519 aa / ~58 kDa / 7p12.2 IKZF1 B-ALL molecular context: "
            "STRUCTURE: "
            "  519 aa / ~58 kDa (IKZF1/Ikaros); N-terminal zinc finger domain 1-4 (aa 140-323) — DNA binding; "
            "  C-terminal zinc finger domain 5-6 (aa 461-519) — dimerisation; "
            "  IKZF1 binds GGGAA and related motifs at lymphoid gene promoters; "
            "  Regulates lymphoid lineage commitment: HSC → CLP → pro-B → pre-B → B-cell; "
            "  IKZF1 LOF/haploinsufficiency → impaired B-lineage commitment → pro-B arrest → ALL predisposition; "
            "SOMATIC IKZF1 DELETION (del7p12.2): "
            "  Somatic del7p12.2 in 30% B-ALL (NOT same as germline IKZF1 LOF); "
            "  Somatic IKZF1 deletion = POOR PROGNOSIS B-ALL (independent adverse marker); "
            "  IKZF1-deleted ALL: increased MRD positivity, higher relapse rate, worse EFS; "
            "  IKZF1plus definition: IKZF1 deletion + concurrent CDKN2A/CDKN2B/PAX5/EBF1 deletion = ultra-high risk; "
            "PHILADELPHIA-LIKE (Ph-like) B-ALL: "
            "  Ph-like ALL = gene expression profile similar to BCR::ABL1 Ph+ ALL but lacks t(9;22); "
            "  IKZF1 deletion present in 80% Ph-like ALL; CRLF2 rearrangement in 50% Ph-like; "
            "  Ph-like ALL: JAK1/2 mutations, EPOR fusions, ABL-class fusions (ETV6::ABL1) → TKI-targetable; "
            "  CRLF2 rearrangement + JAK2 mutation: ruxolitinib JAK inhibitor — active Ph-like ALL; "
            "GERMLINE IKZF1: "
            "  Emerging germline IKZF1 haploinsufficiency predisposition — under active investigation (2026); "
            "  Germline IKZF1 LOF families: immunodeficiency + B-cell lymphopenia + B-ALL reported; "
            "  Somatic del7p12 in tumour ≠ germline IKZF1 — separate germline testing required; "
            "TREATMENT: "
            "  IKZF1-deleted B-ALL: intensified therapy (augmented BFM protocols); "
            "  Ph-like ALL: TKI (dasatinib/imatinib for ABL-class fusions; ruxolitinib for CRLF2/JAK); "
            "  IKZF1plus ultra-high risk: allogeneic SCT in first remission consideration; "
            "SURVEILLANCE: "
            "  Somatic IKZF1 deletion tested on ALL diagnosis FISH/NGS; germline IKZF1 panel in families; "
            "  MRD day 8/29/end-consolidation critical (IKZF1-deleted ALL higher MRD positivity)"
        ),
        "syndrome": "Ph-like ALL / somatic del7p12.2 poor prognosis 30% B-ALL / IKZF1plus ultra-high risk / emerging germline predisposition",
        "inheritance": "AD haploinsufficiency",
        "all_risk": "IKZF1 somatic del7p12.2: 30% B-ALL poor prognosis independent adverse marker; IKZF1plus ultra-high risk; Ph-like ALL: IKZF1 deletion in 80%; germline IKZF1 LOF emerging predisposition; JAK/CRLF2 Ph-like ALL: ruxolitinib TKI; ABL-class Ph-like: dasatinib/imatinib; IKZF1plus → allogeneic SCT consideration",
        "pathognomonic": "Ph-like gene expression profile + IKZF1 deletion + CRLF2 rearrangement + JAK2 mutation = Ph-like ALL PATHOGNOMONIC; IKZF1plus (IKZF1 del + CDKN2A/PAX5/EBF1 del) = ultra-high risk profile; somatic del7p12.2 on FISH + unmutated IGHV-like pre-B immunophenotype; germline IKZF1 LOF family + B-cell lymphopenia + B-ALL = emerging hereditary predisposition",
        "key_avoid": "Do NOT treat IKZF1-deleted ALL with standard-risk protocols (IKZF1 deletion = adverse risk — augmented or high-risk arm); do NOT miss Ph-like ALL (TKI-targetable subset — check CRLF2/JAK2/ABL-class fusions at diagnosis); do NOT confuse somatic del7p12.2 (30% B-ALL) with germline IKZF1 LOF (emerging hereditary); do NOT omit IKZF1plus co-deletion analysis (CDKN2A/CDKN2B/PAX5/EBF1 co-deletion = ultra-high risk requiring SCT consideration)",
        "key_rule": "IKZF1 B-ALL key rules: (1) somatic IKZF1 deletion = POOR PROGNOSIS — augmented/high-risk protocol; (2) IKZF1plus = ultra-high risk → allogeneic SCT in CR1 consideration; (3) Ph-like ALL: test CRLF2/JAK2/ABL-class fusions → TKI (ruxolitinib/dasatinib) add to chemotherapy backbone; (4) Ph-like = 25% adolescent/young adult B-ALL — do not miss; (5) germline IKZF1 under investigation — test in familial B-ALL + immunodeficiency; (6) MRD end-induction critical — IKZF1-deleted ALL higher relapse.",
        "surveillance": "Somatic FISH del7p12.2 at ALL diagnosis MANDATORY; CRLF2 FISH/MLPA + JAK2/STAT5 mutation NGS (Ph-like); ABL-class fusion RNA panel (FISH/RT-PCR) Ph-like; MRD day 8, day 29, end-consolidation (bone marrow flow cytometry + qPCR); germline IKZF1 sequencing in familial B-ALL + immunodeficiency; cascade testing first-degree relatives if germline confirmed; annual immunophenotyping if germline IKZF1 carrier",
        "targeted_rx": "Ruxolitinib JAK1/2 inhibitor (CRLF2 rearrangement / JAK2 mutation Ph-like ALL — COG AALL1231); dasatinib (ABL-class fusions Ph-like: ETV6::ABL1, NUP214::ABL1); imatinib (BCR::ABL1 Ph+, ABL1-class Ph-like); blinatumomab (CD19+ B-ALL MRD/relapsed); tisagenlecleucel CAR-T; venetoclax + steroids (BCL2); inotuzumab ozogamicin (CD22+); augmented BFM protocols IKZF1-deleted; allogeneic SCT IKZF1plus CR1",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "Transcription-Factor-G1-Arrest-"
            "LFS-B-ALL-15x-T-ALL-Elevated-"
            "AVOID-RADIATION-ABSOLUTELY-LFS-"
            "Cranial-RT-ABSOLUTELY-CONTRAINDICATED-LFS-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 B-ALL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; N-terminal transactivation domain (aa 1-40) — MDM2/p300 interaction; "
            "  Proline-rich region (aa 40-92); DNA-binding domain (aa 100-290) — MOST mutations; "
            "  Tetramerisation domain (aa 325-356); C-terminal regulatory domain (aa 363-393); "
            "  R175H, R248Q, R248W, R249S, R273H, R282W = hotspot missense (GOF + dominant negative); "
            "LFS AND B-ALL: "
            "  Li-Fraumeni syndrome (LFS; OMIM 151623): germline TP53 LOF → pan-malignancy predisposition; "
            "  B-ALL risk 15x in LFS — T-ALL also elevated; haematological malignancies account for ~10% LFS cancers; "
            "  LFS B-ALL: earlier onset (child to young adult), TP53 somatic del17p co-occurrence; "
            "  Somatic TP53/del17p in ALL: ~5% at diagnosis; higher relapse; therapy-related (post-chemo); "
            "RADIATION — CRITICAL LFS RULE: "
            "  AVOID RADIATION ABSOLUTELY in LFS (catastrophic secondary malignancies: sarcoma, brain tumour, further haematological); "
            "  Cranial radiation therapy (CRT) ABSOLUTELY CONTRAINDICATED in LFS: second brain tumours (WHO grade IV glioma); "
            "  Craniospinal RT CONTRAINDICATED LFS — spinal cord, bone, soft tissue secondary tumours; "
            "  Modern CNS-directed ALL therapy: intrathecal methotrexate + IT cytarabine + IT hydrocortisone (no CRT LFS); "
            "  TBI conditioning for SCT ABSOLUTELY CONTRAINDICATED LFS; "
            "TREATMENT: "
            "  LFS B-ALL: standard ALL chemotherapy backbone; replace CRT with intensified IT therapy; "
            "  TP53-mutated ALL: venetoclax (BCL2i) + MRD-directed strategy; "
            "  MDM2 inhibitors (idasanutlin/navtemadlin) in TP53-WT (NOT TP53-LOF ALL — MDM2i ineffective LOF); "
            "SURVEILLANCE: "
            "  WBMRI Toronto protocol annually (brain/spine/whole body); "
            "  CBC + flow cytometry annually from age 5yr LFS; breast MRI from 20yr; colonoscopy 25yr"
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) / B-ALL 15x / AVOID RADIATION ABSOLUTELY / cranial RT ABSOLUTELY CONTRAINDICATED LFS / WBMRI Toronto annually",
        "inheritance": "AD LOF",
        "all_risk": "TP53 LFS: B-ALL 15x; T-ALL also elevated; LFS total cancer risk 50% by 30yr, 90% by 60yr; somatic TP53/del17p ~5% ALL at dx → therapy-related risk; AVOID RADIATION ABSOLUTELY (secondary malignancies catastrophic LFS); cranial RT ABSOLUTELY CONTRAINDICATED LFS; WBMRI Toronto annually; IT triple therapy replaces CRT in LFS B-ALL",
        "pathognomonic": "LFS: sarcoma + early breast cancer + brain tumour + adrenocortical carcinoma + B-ALL + family tree of diverse cancers = LFS PATHOGNOMONIC; del17p13.1 on FISH + TP53 sequence loss in ALL blasts = somatic TP53 inactivation; LFS patient receiving cranial RT → second brain tumour risk extreme — contraindicated; WBMRI Toronto protocol = standard LFS surveillance; TP53 germline R248W/R175H hotspot = most common LFS alleles",
        "key_avoid": "AVOID RADIATION ABSOLUTELY in LFS (secondary malignancies catastrophic — multiple primary tumours); cranial RT ABSOLUTELY CONTRAINDICATED (LFS second brain tumours — glioma, medulloblastoma); craniospinal RT ABSOLUTELY CONTRAINDICATED LFS; TBI conditioning SCT CONTRAINDICATED LFS; do NOT use MDM2 inhibitors (idasanutlin) in TP53-LOF ALL (MDM2i ONLY effective TP53-WT — wrong mechanism LOF); do NOT delay WBMRI Toronto protocol annually",
        "key_rule": "TP53 LFS B-ALL key rules: (1) AVOID RADIATION ABSOLUTELY — no CRT, no craniospinal RT, no TBI (LFS); (2) IT triple therapy (methotrexate + cytarabine + hydrocortisone) replaces cranial RT; (3) WBMRI Toronto annually — LFS whole-body cancer surveillance; (4) venetoclax + MRD-directed strategy TP53-mutated ALL; (5) MDM2 inhibitors INEFFECTIVE in TP53-LOF ALL (MDM2i only works TP53-WT); (6) cascade test first-degree relatives LFS; (7) LFS cancer registry enrollment.",
        "surveillance": "LFS: WBMRI Toronto annually (brain/spine/whole body); CBC + flow cytometry annually from age 5yr; TP53 germline confirmed panel; breast MRI annually from age 20yr; colonoscopy from 25yr; dermatology annually; CBC + LDH monthly if prior ALL history (relapse); somatic TP53/del17p at ALL diagnosis AND relapse; cascade testing first-degree relatives; LFS Registry (ISLFSG) enrollment recommended",
        "targeted_rx": "Standard ALL chemotherapy (COG AALL0434/0931/1231 + BFM 2017); IT triple therapy (methotrexate + cytarabine + hydrocortisone) — REPLACES CRT in LFS; venetoclax (BCL2i — TP53-mutated ALL emerging role); blinatumomab (CD19+ relapsed/refractory); tisagenlecleucel CAR-T (relapsed ≥25kg CD19+); avoid MDM2 inhibitors (LOF ineffective); avoid CRT/TBI (LFS ABSOLUTELY); avoid craniospinal RT (LFS ABSOLUTELY); cascade testing MANDATORY",
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD/AR LOF -- 3056aa -- "
            "ATM-350kDa-PIK3-Kinase-"
            "DSB-Master-Sensor-"
            "A-T-Biallelic-T-ALL-75x-HIGHEST-B-ALL-Elevated-"
            "RADIOSENSITIVITY-ABSOLUTE-Biallelic-"
            "IgA-Deficiency-PATHOGNOMONIC-A-T-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM B-ALL molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; serine/threonine PIK3-related kinase (PIKK family); "
            "  HEAT repeats (aa 1-1960) — scaffold; FAT domain (aa 1960-2566) — regulation; "
            "  Kinase domain (aa 2714-2962) — catalytic; FATC (aa 2963-3056) — essential; "
            "  Activated by DSBs → autophosphorylation Ser1981 → monomer → active; "
            "  Substrates: H2AX (γH2AX), CHK2, BRCA1, RAD51, MDM2, p53; "
            "A-T (ATAXIA-TELANGIECTASIA) AND B-ALL: "
            "  Biallelic ATM LOF = Ataxia-Telangiectasia (A-T; OMIM 208900); "
            "  A-T: T-cell ALL/lymphoma 75x HIGHEST; B-cell ALL also markedly elevated; "
            "  T-ALL predominates in A-T (T-lineage DNA rearrangements during V(D)J recombination); "
            "  Monoallelic germline ATM: B-ALL/B-cell NHL 2-3x (less dramatic than CLL 4-5x); "
            "IgA DEFICIENCY — PATHOGNOMONIC: "
            "  Serum IgA undetectable or severely reduced in A-T biallelic = PATHOGNOMONIC; "
            "  IgG also low; total lymphocyte count reduced; "
            "  AFP elevated in >95% A-T — pathognomonic screening test; "
            "RADIOSENSITIVITY: "
            "  RADIOSENSITIVITY ABSOLUTE biallelic A-T: no RT, no TBI, no craniospinal RT; "
            "  TBI conditioning for SCT ABSOLUTELY CONTRAINDICATED biallelic ATM; "
            "  Monoallelic germline ATM: RT dose reduce 20-30% (partial ATM kinase haploinsufficiency); "
            "  CRT in A-T = catastrophic pulmonary fibrosis + myelosuppression + radiation recall; "
            "TREATMENT: "
            "  A-T ALL: ALL chemotherapy backbone WITHOUT RT (no CRT, no craniospinal, no TBI); "
            "  IT triple therapy replaces CRT; busulfan-based non-TBI conditioning for SCT; "
            "  Ceralasertib (ATRi) synthetic lethality ATM-LOF ALL — investigational; "
            "  PARP inhibitor sensitivity (ATM LOF HRD partial) monoallelic; "
            "SURVEILLANCE: "
            "  A-T: AFP at diagnosis; lymphocyte subsets + Ig levels annually; WBMRI Toronto annually; "
            "  Annual CBC from age 2yr; no RT cancer screening in A-T (radiosensitivity)"
        ),
        "syndrome": "A-T biallelic (OMIM 208900) / T-ALL 75x HIGHEST biallelic / B-ALL 2-3x monoallelic / IgA deficiency PATHOGNOMONIC / RADIOSENSITIVITY ABSOLUTE biallelic",
        "inheritance": "AD/AR LOF",
        "all_risk": "ATM biallelic A-T: T-ALL 75x HIGHEST; B-ALL also elevated; monoallelic germline ATM: B-ALL 2-3x; IgA deficiency PATHOGNOMONIC biallelic; elevated AFP PATHOGNOMONIC A-T; RADIOSENSITIVITY ABSOLUTE biallelic — no RT, no CRT, no TBI; monoallelic: RT reduce 20-30%; ceralasertib ATRi synthetic lethality investigational",
        "pathognomonic": "A-T: cerebellar ataxia + telangiectasias + IgA deficiency + AFP elevated >95% + T-ALL = A-T PATHOGNOMONIC; γH2AX foci absent/reduced in ATM-deficient cells; RADIOSENSITIVITY testing (chromosomal breakage + clonogenic survival) confirms biallelic; serum AFP >10 ng/mL in child with ataxia + CLL/lymphoma = A-T until proven otherwise; IgA undetectable + lymphopenia + AFP elevated = A-T screening PATHOGNOMONIC triad",
        "key_avoid": "NEVER use RT in biallelic A-T (ABSOLUTE CONTRAINDICATION — catastrophic); no cranial RT, no craniospinal RT, no TBI conditioning biallelic ATM; reduce RT 20-30% monoallelic germline ATM; avoid alkylating agents without haematology review in A-T (DNA repair defect → toxicity); do NOT use MDM2 inhibitors ATM-LOF (TP53 pathway co-activation — unpredictable); do NOT order standard mammography A-T biallelic (radiosensitivity — use MRI only)",
        "key_rule": "ATM B-ALL key rules: (1) biallelic A-T: T-ALL 75x HIGHEST — ALL chemotherapy WITHOUT any RT; (2) IT triple therapy REPLACES CRT; (3) TBI conditioning ABSOLUTELY CONTRAINDICATED biallelic; (4) RADIOSENSITIVITY ABSOLUTE biallelic — monoallelic: RT reduce 20-30%; (5) AFP + IgA = A-T screening PATHOGNOMONIC — test early; (6) monoallelic B-ALL 2-3x — standard RT cautiously reduced; (7) ceralasertib ATRi synthetic lethality investigational ATM-LOF.",
        "surveillance": "A-T: AFP at diagnosis (pathognomonic); lymphocyte subsets (CD4/CD8/NK) + serum IgA/IgG/IgM annually; WBMRI Toronto annually biallelic; CBC + flow cytometry from age 2yr; neurological examination annually; MRI brain (no CT — radiosensitivity); cascade testing first-degree relatives germline ATM; monoallelic: annual CBC + immunophenotyping from age 30yr; radiosensitivity testing at ALL diagnosis A-T",
        "targeted_rx": "Standard paediatric ALL protocols WITHOUT RT (COG, BFM); IT triple therapy (methotrexate + cytarabine + hydrocortisone) — REPLACES CRT; busulfan-based non-TBI conditioning SCT biallelic; ceralasertib (AZD6738 ATRi Phase 2 — ATM-LOF synthetic lethality); PARP inhibitors (olaparib/rucaparib — partial HRD monoallelic ATM); blinatumomab (CD19+ B-ALL); tisagenlecleucel CAR-T; IVIG supplementation A-T immunodeficiency; avoid TBI; avoid CRT; cascade testing MANDATORY",
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 AD LOF -- 156aa -- "
            "p16-INK4A-17kDa-CDK4-6-Inhibitor-"
            "p14ARF-MDM2-p53-Axis-"
            "FAMM-del9p21-Somatic-ALL-20-35pct-"
            "B-ALL-T-ALL-5-10x-"
            "Palbociclib-INACTIVE-p16-Null-OMIM-606719"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / ~17 kDa / 9p21.3 CDKN2A B-ALL molecular context: "
            "STRUCTURE: "
            "  156 aa / ~17 kDa (p16INK4a); CDKN2A locus encodes TWO tumour suppressors: "
            "  p16INK4a (156aa, exons 1α-2-3) — 4 ankyrin repeats; CDK4/CDK6 binding; "
            "  p14ARF (132aa, exon 1β-2-3) — MDM2 binding → p53 stabilisation (alternate reading frame); "
            "  p16INK4a competes with D-cyclins → inhibits CDK4/CDK6 → pRB not phosphorylated → G1 arrest; "
            "  p16INK4a LOF → unchecked CDK4/CDK6 → pRB hyperphosphorylation → E2F release → S phase; "
            "  p14ARF LOF → MDM2 unopposed → p53 degradation → p53 pathway loss; "
            "  CDKN2A deletion inactivates BOTH p16INK4a AND p14ARF simultaneously (dual tumour suppressor loss); "
            "del9p21 SOMATIC ALL: "
            "  del9p21 somatic = MOST COMMON cytogenetic deletion in B-ALL (20-35% at diagnosis); "
            "  CDKN2A/CDKN2B (p15INK4b adjacent) co-deletion typical in ALL (homozygous or hemizygous); "
            "  del9p21 in ALL = adverse marker; IKZF1plus includes CDKN2A co-deletion; "
            "  Homozygous del9p21 = complete p16INK4a loss; heterozygous = haploinsufficiency; "
            "GERMLINE CDKN2A LOF — FAMM: "
            "  FAMM: Familial Atypical Multiple Mole Melanoma (OMIM 606719); "
            "  Germline CDKN2A LOF: B-ALL/T-ALL 5-10x; melanoma 28-67x HIGHEST; pancreatic 17-39x; "
            "  CDKN2A germline + somatic del9p21 ALL = dual-hit CDKN2A (worst ALL outcome); "
            "CDK4/6 INHIBITOR — KEY RULE: "
            "  p16-null ALL: CDK4/6 inhibitor palbociclib INACTIVE (no p16 → CDK4/6 already unrestrained); "
            "  Same principle as CLL: CDKN2A-null = palbociclib cannot re-inhibit already-disinhibited CDK4/6; "
            "  Venetoclax ACTIVE regardless CDKN2A status; "
            "TREATMENT: "
            "  CDKN2A-ALL: standard ALL protocols; palbociclib INACTIVE p16-null; venetoclax preferred; "
            "  del9p21 ALL: intensified arm (IKZF1plus-type); abemaciclib investigational in p16-present ALL"
        ),
        "syndrome": "FAMM / del9p21 somatic ALL 20-35% / CDKN2A dual p16INK4a + p14ARF LOF / CDK4/6 inhibitor palbociclib INACTIVE p16-null / pancreatic 17-39x EUS",
        "inheritance": "AD LOF",
        "all_risk": "CDKN2A FAMM LOF: B-ALL/T-ALL 5-10x; del9p21 somatic 20-35% ALL — most common ALL cytogenetic deletion; germline + somatic del9p21 = dual-hit CDKN2A worst outcome; p16-null ALL → palbociclib INACTIVE; venetoclax ACTIVE regardless; melanoma 28-67x; pancreatic 17-39x EUS from 40yr MANDATORY",
        "pathognomonic": "FAMM: atypical melanoma >50 moles + family melanoma + pancreatic cancer = CDKN2A PATHOGNOMONIC; del9p21 on ALL FISH/SNP array (homozygous = both alleles, hemizygous = one) = most common ALL cytogenetics; p16 IHC loss in ALL blasts = CDKN2A inactivation; palbociclib failure p16-null ALL (expected — INACTIVE confirmed by IHC); IKZF1plus includes CDKN2A co-deletion = ultra-high risk ALL",
        "key_avoid": "Do NOT use CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib) in p16INK4a-null ALL (INACTIVE — avoidable toxicity); do NOT miss pancreatic EUS surveillance FAMM (17-39x risk — EUS from 40yr MANDATORY); do NOT use CDKN2A IHC alone without FISH/SNP array (IHC may be false-negative hemizygous); do NOT confuse del9p21 somatic (most common ALL) with germline CDKN2A LOF (FAMM hereditary — separate germline testing); do NOT skip melanoma surveillance FAMM families",
        "key_rule": "CDKN2A B-ALL key rules: (1) del9p21 somatic 20-35% ALL = most common ALL deletion — test ALL diagnosis FISH/SNP array; (2) p16-null ALL → CDK4/6 inhibitor palbociclib INACTIVE; (3) venetoclax ACTIVE regardless CDKN2A status; (4) FAMM germline: pancreatic EUS from 40yr MANDATORY (17-39x risk); (5) IKZF1plus (IKZF1 + CDKN2A co-del) = ultra-high risk → SCT consideration CR1; (6) cascade test FAMM families; melanoma surveillance annually from 25yr.",
        "surveillance": "FAMM: dermoscopy annually from 25yr; pancreatic EUS from 40yr (17-39x risk — 6-monthly if family pancreatic); CBC + immunophenotyping from age 35yr; ALL FISH/SNP array (del9p21) at diagnosis; p16 IHC on ALL blasts; NGS panel at diagnosis; cascade testing first-degree relatives; CDKN2A germline panel in FAMM + ALL; IKZF1plus co-deletion analysis at ALL diagnosis; abdominal MRI pancreas if EUS unavailable",
        "targeted_rx": "Standard ALL protocols (COG AALL0434/0931/1231, BFM 2017); venetoclax (BCL2i — ACTIVE regardless CDKN2A); AVOID CDK4/6 inhibitors (palbociclib/ribociclib) p16-null ALL (INACTIVE); blinatumomab (CD19+ relapsed/refractory); tisagenlecleucel CAR-T; inotuzumab ozogamicin (CD22+); allogeneic SCT CR1 IKZF1plus (IKZF1 + CDKN2A co-del ultra-high risk); wide-excision melanoma FAMM; pancreatic EUS + resection if EUS lesion FAMM; cascade testing MANDATORY",
    },
    {
        "gene": "NBN",
        "protein": (
            "NBN -- 8q21.3 AD/AR LOF -- 754aa -- "
            "NBS1-Nibrin-85kDa-MRN-Complex-"
            "DSB-Sensor-ATM-Recruiter-"
            "657del5-Slavic-Founder-1-in-150-"
            "B-Cell-ALL-3-5x-NBS-ALL-60x-HIGHEST-"
            "Cranial-RT-ABSOLUTELY-CONTRAINDICATED-NBS-OMIM-251260"
        ),
        "locus": "8q21.3",
        "protein_size": (
            "754 aa / ~85 kDa / 8q21.3 NBN B-ALL molecular context: "
            "STRUCTURE: "
            "  754 aa / ~85 kDa; N-terminal FHA domain (aa 1-108) + BRCT1 (aa 114-197) + BRCT2 (aa 221-332) — phosphopeptide docking; "
            "  MRE11-binding domain (aa 665-693); ATM-interacting domain (aa 734-754); "
            "  MRN complex: MRE11-RAD50-NBS1 trimer; DSB sensor and ATM recruitment; "
            "  657del5 = exon 6 frameshift → p.K219fs; founder mutation Czech/Slovak/Polish; "
            "NBS (NIJMEGEN BREAKAGE SYNDROME) AND B-ALL: "
            "  NBS = biallelic NBN LOF → microcephaly + immunodeficiency + radiosensitivity + lymphoma 100x; "
            "  Carrier frequency 657del5 = 1:150 Czech/Slovak/Polish (HIGHEST founder allele); "
            "  NBS biallelic: B-cell ALL 60x HIGHEST (most severe ALL predisposition in this atlas outside FA-D1); "
            "  NBS clinical triad: microcephaly + café au lait macules + immunodeficiency + ALL/lymphoma; "
            "  T-cell and B-cell both elevated biallelic NBS; "
            "CRANIAL RT — ABSOLUTELY CONTRAINDICATED NBS: "
            "  Cranial radiation ABSOLUTELY CONTRAINDICATED in NBS biallelic: "
            "    (a) RADIOSENSITIVITY ABSOLUTE (MRN complex absent → no DSB repair); "
            "    (b) Neurotoxicity catastrophic in microcephalic NBS brain; "
            "    (c) Secondary brain malignancy risk extreme (irradiated MRN-deficient CNS); "
            "  Craniospinal RT ABSOLUTELY CONTRAINDICATED NBS biallelic; "
            "  TBI conditioning ABSOLUTELY CONTRAINDICATED NBS biallelic; "
            "MONOALLELIC NBN: "
            "  657del5 monoallelic: B-cell ALL 3-5x (same as CLL 3-5x); NHL 60x monoallelic strongest signal; "
            "  Monoallelic NBN: RT reduce 20-30% (partial MRN dysfunction); "
            "ALKYLATING AGENTS: "
            "  AVOID alkylating agents biallelic NBS (profound DNA repair defect — aplasia risk); "
            "  Cyclophosphamide, melphalan, busulfan DOSE REDUCE or AVOID biallelic NBS; "
            "  Monoallelic NBN: standard ALL chemotherapy with haematological monitoring; "
            "TREATMENT: "
            "  NBS B-ALL: IT triple therapy replaces CRT; reduced-intensity chemotherapy biallelic; "
            "  Busulfan-based non-TBI conditioning ONLY if SCT required biallelic; "
            "  PARP inhibitors: MRN LOF → HRD → olaparib/rucaparib sensitivity"
        ),
        "syndrome": "NBS biallelic (OMIM 251260) / 657del5 Slavic founder 1:150 / B-cell ALL 3-5x monoallelic / NBS ALL 60x HIGHEST biallelic / cranial RT ABSOLUTELY CONTRAINDICATED NBS / AVOID alkylating agents biallelic",
        "inheritance": "AD/AR LOF",
        "all_risk": "NBN 657del5 monoallelic: B-cell ALL 3-5x; NBS biallelic: ALL 60x HIGHEST; RADIOSENSITIVITY ABSOLUTE biallelic; cranial RT ABSOLUTELY CONTRAINDICATED NBS (neurotoxicity + malignancy); AVOID alkylating agents biallelic; monoallelic: RT reduce 20-30%; 1:150 Czech/Slovak/Polish — COMMON Eastern European hereditary ALL; PARP inhibitor sensitivity (MRN LOF HRD)",
        "pathognomonic": "NBS biallelic: microcephaly + café au lait macules + immunodeficiency + 657del5 = NBS PATHOGNOMONIC; B-cell ALL/NHL in Czech/Slovak/Polish child + family history = test NBN 657del5 FIRST; radiosensitivity testing (chromosomal breakage + clonogenic survival) confirms NBS biallelic; cranial RT in NBS = catastrophic outcome (neurotoxicity + secondary malignancy); IgG low + T-cell lymphopenia + microcephaly = NBS screening triad",
        "key_avoid": "NEVER use cranial RT in NBS biallelic (ABSOLUTELY CONTRAINDICATED — neurotoxicity + radiosensitivity + secondary malignancy); no craniospinal RT biallelic NBS; no TBI conditioning biallelic NBS; AVOID alkylating agents biallelic (cyclophosphamide, melphalan, busulfan — dose reduce or avoid); do NOT miss 657del5 in Eastern European ALL (COMMON 1:150 carrier); do NOT confuse NBS biallelic (ALL 60x) with monoallelic 657del5 (ALL 3-5x); monoallelic: RT reduce 20-30%",
        "key_rule": "NBN 657del5 B-ALL key rules: (1) test FIRST in Czech/Slovak/Polish ALL patients (1:150 carrier freq); (2) biallelic NBS: cranial RT ABSOLUTELY CONTRAINDICATED (neurotoxicity + radiosensitivity); (3) AVOID alkylating agents biallelic; (4) IT triple therapy replaces CRT biallelic NBS; (5) RADIOSENSITIVITY ABSOLUTE biallelic — no RT, no TBI; (6) monoallelic: RT reduce 20-30%; (7) PARP inhibitor sensitivity MRN HRD; (8) cascade test family.",
        "surveillance": "Annual CBC + lymphocyte immunophenotyping from age 2yr NBS; immunoglobulin levels (IgG/IgA/IgM) + lymphocyte subsets annually NBS; WBMRI Toronto biallelic NBS annually (no radiation-based imaging); NBN 657del5 founder test in Eastern European ALL families; cascade testing first-degree relatives; radiosensitivity testing (chromosomal breakage) at ALL diagnosis NBS; PARP sensitivity testing at relapse; IVIG supplementation NBS immunodeficiency",
        "targeted_rx": "Standard ALL protocols without RT (COG, BFM); IT triple therapy (methotrexate + cytarabine + hydrocortisone) — REPLACES CRT biallelic NBS; busulfan-based non-TBI conditioning ONLY if SCT required biallelic; PARP inhibitors (olaparib/rucaparib — MRN HRD sensitivity); blinatumomab (CD19+ B-ALL); tisagenlecleucel CAR-T; venetoclax + steroids (BCL2); IVIG biallelic NBS (immunodeficiency); AVOID alkylating agents biallelic; cascade testing MANDATORY Eastern European families",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 AD LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-"
            "FANCD1-DNA-Binding-BRC-Repeats-"
            "FA-D1-Biallelic-B-ALL-AML-Wilms-HIGHEST-"
            "Monoallelic-B-ALL-2x-"
            "AVOID-ALKYLATING-AGENTS-Biallelic-OMIM-605724"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / ~384 kDa / 13q12.3 BRCA2 B-ALL molecular context: "
            "STRUCTURE: "
            "  3418 aa / ~384 kDa; N-terminal transactivation domain (BRCA2-PALB2 interaction aa 10-40); "
            "  BRC repeats ×8 (aa 1002-2085) — RAD51 monomer binding (pre-synaptic filament loading); "
            "  DNA-binding domain DBD (aa 2402-3186) — ssDNA/dsDNA junction binding; "
            "  C-terminal domain (aa 3187-3418) — CTP1, BRCA1-BRCA2-NBS1 complex; "
            "  BRCA2 escorts RAD51 to DSBs → strand invasion → homologous recombination; "
            "FANCONI ANAEMIA COMPLEMENTATION GROUP D1 (FA-D1): "
            "  BRCA2 = FANCD1 (Fanconi anaemia complementation group D1; OMIM 605724); "
            "  Biallelic BRCA2 LOF = FA-D1 = MOST SEVERE FA complementation group; "
            "  FA-D1 PATHOGNOMONIC phenotype: medulloblastoma + Wilms tumour + AML + B-ALL + developmental anomalies; "
            "  B-ALL/AML = HIGHEST RISK in FA pathway (FA-D1 dominant haematological malignancy risk); "
            "  FA-D1 onset: extremely early (median B-ALL age 2-5yr FA-D1 — youngest in this atlas); "
            "  DEB/MMC chromosomal breakage test = gold standard FA diagnosis; "
            "ALKYLATING AGENTS — ABSOLUTELY CONTRAINDICATED FA-D1 BIALLELIC: "
            "  Alkylating agents (cyclophosphamide, melphalan, busulfan, chlorambucil, bendamustine) ABSOLUTELY CONTRAINDICATED biallelic BRCA2 FA-D1; "
            "  DNA-crosslinking mechanism catastrophic in FA-D1 (interstrand crosslinks = lethal FA cells); "
            "  Cyclophosphamide — standard B-ALL backbone — AVOID biallelic FA-D1; "
            "  Melphalan TBI-conditioning CONTRAINDICATED FA-D1; "
            "  FA-D1 ALL: modified FA-specific ALL protocols (low-dose, non-crosslinking agents); "
            "MONOALLELIC BRCA2: "
            "  Monoallelic BRCA2 LOF: B-ALL 2x modest elevation; HRD phenotype in B-ALL cells; "
            "  PARP inhibitor sensitivity monoallelic BRCA2-ALL (olaparib/rucaparib investigational); "
            "  Standard ALL chemotherapy protocols monoallelic; HRD testing at ALL diagnosis; "
            "TREATMENT: "
            "  FA-D1 B-ALL: FA-specific modified ALL protocol (avoid DNA crosslinkers); "
            "  Monoallelic: standard ALL + PARP inhibitor sensitivity; "
            "  Stem cell transplant FA-D1: reduced-intensity conditioning MANDATORY (avoid alkylating agents)"
        ),
        "syndrome": "HBOC / FA-D1 biallelic PATHOGNOMONIC (OMIM 605724) / B-ALL/AML/Wilms HIGHEST biallelic / B-ALL 2x monoallelic / AVOID ALKYLATING AGENTS biallelic / PARP inhibitor sensitivity monoallelic",
        "inheritance": "AD LOF",
        "all_risk": "BRCA2 biallelic FA-D1: B-ALL/AML HIGHEST RISK in FA pathway; Wilms + medulloblastoma co-occur; onset 2-5yr YOUNGEST; AVOID ALKYLATING AGENTS biallelic (cyclophosphamide, melphalan, busulfan CONTRAINDICATED); DEB/MMC chromosomal breakage test PATHOGNOMONIC; monoallelic: B-ALL 2x; HRD-positive ALL → PARP inhibitor sensitivity; monoallelic standard ALL protocols",
        "pathognomonic": "FA-D1 biallelic: medulloblastoma + Wilms tumour + AML + B-ALL + DEB/MMC breakage test PATHOGNOMONIC; DEB/MMC chromosomal breakage test = gold standard FA (increased breakage with diepoxybutane/mitomycin C); FA-D1 = most severe FA group — haematological malignancy earliest onset; BRCA2 germline + ALL in HBOC family = hereditary ALL risk; HRD testing positive (genomic scar/LOH) in BRCA2-ALL = PARP inhibitor eligible",
        "key_avoid": "AVOID ALKYLATING AGENTS biallelic BRCA2 FA-D1 (DNA-crosslinking mechanism catastrophic — interstrand crosslinks lethal FA-D1 cells; cyclophosphamide, melphalan, busulfan, chlorambucil, bendamustine ALL CONTRAINDICATED biallelic); do NOT use standard ALL protocol cyclophosphamide backbone FA-D1 (use modified non-crosslinking FA-ALL protocol); do NOT use melphalan-TBI conditioning FA-D1 (CONTRAINDICATED); do NOT miss DEB/MMC chromosomal breakage test (FA gold standard diagnostic)",
        "key_rule": "BRCA2 B-ALL key rules: (1) FA-D1 biallelic: DEB/MMC chromosomal breakage test MANDATORY for FA diagnosis; (2) AVOID ALKYLATING AGENTS biallelic (cyclophosphamide/melphalan/busulfan CONTRAINDICATED); (3) use modified FA-ALL protocols (non-crosslinking agents: methotrexate, 6-MP, vincristine, steroids backbone); (4) monoallelic: standard ALL protocols + HRD testing + PARP inhibitor eligibility; (5) reduced-intensity NON-ALKYLATING conditioning if SCT required FA-D1; (6) cascade test HBOC families.",
        "surveillance": "FA-D1: DEB/MMC chromosomal breakage test at diagnosis (gold standard FA); annual CBC from birth (aplasia + blast surveillance); bone marrow aspirate annually from age 1yr FA-D1 (extremely early ALL onset); BRCA2 germline panel in HBOC families; HRD testing (genomic scar/LOH) at ALL diagnosis monoallelic; PARP inhibitor eligibility assessment at relapse; HBOC surveillance (breast MRI annually from 25yr monoallelic; prostate from 40yr); Wilms tumour ultrasound annually to 8yr FA-D1; cascade testing MANDATORY",
        "targeted_rx": "FA-D1 modified ALL protocol (methotrexate + 6-mercaptopurine + vincristine + dexamethasone — AVOID DNA crosslinkers); AVOID cyclophosphamide, melphalan, busulfan biallelic FA-D1; reduced-intensity non-alkylating conditioning SCT FA-D1; PARP inhibitors (olaparib/rucaparib — HRD-positive monoallelic BRCA2 ALL investigational); blinatumomab (CD19+ relapsed); tisagenlecleucel CAR-T; venetoclax + steroids (BCL2); platinum-based (monoallelic HRD sensitivity); cascade testing MANDATORY HBOC families",
    },
]


def _make_patient(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)

    # Gene-specific B-ALL characteristics
    all_params = {
        "ETV6":   dict(age_mu=7,  age_sd=3,  all_prob=0.80, t_12_21_prob=0.40, del9p21_prob=0.20,
                       del7p12_prob=0.10, ph_like_prob=0.05, tp53_del_prob=0.03,
                       hrd_prob=0.05, craniospinal_rt_prob=0.30, venetoclax_eligible=True),
        "PAX5":   dict(age_mu=6,  age_sd=4,  all_prob=0.70, t_12_21_prob=0.15, del9p21_prob=0.25,
                       del7p12_prob=0.12, ph_like_prob=0.08, tp53_del_prob=0.04,
                       hrd_prob=0.05, craniospinal_rt_prob=0.28, venetoclax_eligible=True),
        "IKZF1":  dict(age_mu=18, age_sd=10, all_prob=0.50, t_12_21_prob=0.08, del9p21_prob=0.30,
                       del7p12_prob=0.75, ph_like_prob=0.60, tp53_del_prob=0.05,
                       hrd_prob=0.06, craniospinal_rt_prob=0.25, venetoclax_eligible=True),
        "TP53":   dict(age_mu=28, age_sd=12, all_prob=0.35, t_12_21_prob=0.05, del9p21_prob=0.18,
                       del7p12_prob=0.15, ph_like_prob=0.10, tp53_del_prob=0.70,
                       hrd_prob=0.08, craniospinal_rt_prob=0.05, venetoclax_eligible=True),
        "ATM":    dict(age_mu=12, age_sd=7,  all_prob=0.65, t_12_21_prob=0.10, del9p21_prob=0.20,
                       del7p12_prob=0.18, ph_like_prob=0.12, tp53_del_prob=0.05,
                       hrd_prob=0.12, craniospinal_rt_prob=0.25, venetoclax_eligible=True),
        "CDKN2A": dict(age_mu=35, age_sd=12, all_prob=0.45, t_12_21_prob=0.06, del9p21_prob=0.80,
                       del7p12_prob=0.22, ph_like_prob=0.18, tp53_del_prob=0.10,
                       hrd_prob=0.07, craniospinal_rt_prob=0.20, venetoclax_eligible=True),
        "NBN":    dict(age_mu=9,  age_sd=5,  all_prob=0.70, t_12_21_prob=0.12, del9p21_prob=0.22,
                       del7p12_prob=0.15, ph_like_prob=0.10, tp53_del_prob=0.04,
                       hrd_prob=0.18, craniospinal_rt_prob=0.15, venetoclax_eligible=True),
        "BRCA2":  dict(age_mu=5,  age_sd=3,  all_prob=0.60, t_12_21_prob=0.10, del9p21_prob=0.18,
                       del7p12_prob=0.12, ph_like_prob=0.08, tp53_del_prob=0.05,
                       hrd_prob=0.70, craniospinal_rt_prob=0.12, venetoclax_eligible=True),
    }
    p = all_params[gene]

    # Age distribution with gene-specific ranges
    age_raw = rng.gauss(p["age_mu"], p["age_sd"])
    if gene in ("ETV6", "PAX5", "BRCA2"):
        age = max(1, min(20, round(age_raw, 1)))
    elif gene in ("NBN", "ATM"):
        age = max(2, min(30, round(age_raw, 1)))
    elif gene == "IKZF1":
        age = max(5, min(45, round(age_raw, 1)))
    elif gene == "TP53":
        age = max(10, min(55, round(age_raw, 1)))
    elif gene == "CDKN2A":
        age = max(15, min(60, round(age_raw, 1)))
    else:
        age = max(1, min(60, round(age_raw, 1)))

    has_all         = rng.random() < p["all_prob"]
    t_12_21         = rng.random() < p["t_12_21_prob"]   # somatic ETV6::RUNX1
    del9p21         = rng.random() < p["del9p21_prob"]   # somatic CDKN2A
    del7p12         = rng.random() < p["del7p12_prob"]   # somatic IKZF1
    ph_like         = rng.random() < p["ph_like_prob"]   # Ph-like gene expression
    tp53_del        = rng.random() < p["tp53_del_prob"]  # somatic TP53/del17p
    hrd_positive    = rng.random() < p["hrd_prob"]
    craniospinal_rt = rng.random() < p["craniospinal_rt_prob"]
    venetoclax_elig = p["venetoclax_eligible"] and has_all

    # Immunophenotype
    immuno_types = ["B-ALL", "T-ALL", "AML", "normal"]
    if gene == "ATM":
        immuno_probs = [0.50, 0.30, 0.05, 0.15]  # ATM: T-ALL elevated biallelic A-T
    elif gene == "CDKN2A":
        immuno_probs = [0.55, 0.25, 0.05, 0.15]  # CDKN2A: T-ALL also elevated
    else:
        immuno_probs = [0.72, 0.12, 0.06, 0.10]  # Default: predominantly B-ALL
    cumulative = 0.0; r = rng.random()
    immunophenotype = immuno_types[-1]
    for it, ip in zip(immuno_types, immuno_probs):
        cumulative += ip
        if r < cumulative:
            immunophenotype = it; break

    return {
        "gene":             gene,
        "seed":             seed,
        "age_at_dx":        age,
        "has_all":          has_all,
        "immunophenotype":  immunophenotype,
        "t_12_21":          t_12_21,
        "del9p21":          del9p21,
        "del7p12":          del7p12,
        "ph_like":          ph_like,
        "tp53_del":         tp53_del,
        "hrd_positive":     hrd_positive,
        "craniospinal_rt":  craniospinal_rt,
        "venetoclax_elig":  venetoclax_elig,
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

    all_n           = sum(1 for p in all_pts if p["has_all"])
    del9p21_n       = sum(1 for p in all_pts if p["del9p21"])
    ph_like_n       = sum(1 for p in all_pts if p["ph_like"])
    t_12_21_n       = sum(1 for p in all_pts if p["t_12_21"])
    craniospinal_n  = sum(1 for p in all_pts if p["craniospinal_rt"])
    hrd_n           = sum(1 for p in all_pts if p["hrd_positive"])
    venetoclax_n    = sum(1 for p in all_pts if p["venetoclax_elig"])
    mean_age        = round(sum(p["age_at_dx"] for p in all_pts) / len(all_pts), 1)

    gene_summary = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        all_gene_n = sum(1 for p in pts if p["has_all"])
        gene_summary.append({
            "gene":             gene,
            "locus":            gd["locus"],
            "n":                len(pts),
            "all_n":            all_gene_n,
            "all_pct":          round(100 * all_gene_n / len(pts), 1),
            "mean_age_onset":   round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "del9p21_n":        sum(1 for p in pts if p["del9p21"]),
            "ph_like_n":        sum(1 for p in pts if p["ph_like"]),
            "t_12_21_n":        sum(1 for p in pts if p["t_12_21"]),
            "craniospinal_rt_n": sum(1 for p in pts if p["craniospinal_rt"]),
            "inheritance":      gd["inheritance"],
            "key_distinctions": [
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
            "key_distinctions": [
                gd["all_risk"][:80],
                gd["key_avoid"][:80],
                gd["key_rule"][:80],
            ],
        })

    return {
        "atlas":                    "Hereditary-B-ALL-Predisposition-Atlas",
        "seed_range":               f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":           len(all_pts),
        "genes_n":                  8,
        "all_total_n":              all_n,
        "all_total_pct":            round(100 * all_n / len(all_pts), 1),
        "mean_age_onset":           mean_age,
        "del9p21_n":                del9p21_n,
        "del9p21_pct":              round(100 * del9p21_n / len(all_pts), 1),
        "ph_like_n":                ph_like_n,
        "ph_like_pct":              round(100 * ph_like_n / len(all_pts), 1),
        "t_12_21_n":                t_12_21_n,
        "t_12_21_pct":              round(100 * t_12_21_n / len(all_pts), 1),
        "craniospinal_rt_n":        craniospinal_n,
        "craniospinal_rt_pct":      round(100 * craniospinal_n / len(all_pts), 1),
        "hrd_positive_n":           hrd_n,
        "hrd_positive_pct":         round(100 * hrd_n / len(all_pts), 1),
        "venetoclax_eligible_n":    venetoclax_n,
        "venetoclax_eligible_pct":  round(100 * venetoclax_n / len(all_pts), 1),
        "gene_summary":             gene_summary,
        "genes_detail":             genes_detail,
    }


def generate_breakdown() -> dict:
    cohorts = _build_cohorts()
    _gene_pathog = {gd["gene"]: gd["pathognomonic"] for gd in ATLAS_GENES}
    _gene_surv   = {gd["gene"]: gd["surveillance"][:120] for gd in ATLAS_GENES}
    _gene_dist   = {gd["gene"]: [gd["all_risk"][:80], gd["key_rule"][:80]] for gd in ATLAS_GENES}
    _gene_rule   = {gd["gene"]: gd["key_rule"] for gd in ATLAS_GENES}

    breakdown_array = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        all_n = sum(1 for p in pts if p["has_all"])
        breakdown_array.append({
            "gene":              gene,
            "locus":             gd["locus"],
            "n":                 len(pts),
            "mean_age_onset":    round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "all_n":             all_n,
            "all_pct":           round(100 * all_n / len(pts), 1),
            "del9p21_n":         sum(1 for p in pts if p["del9p21"]),
            "ph_like_n":         sum(1 for p in pts if p["ph_like"]),
            "t_12_21_n":         sum(1 for p in pts if p["t_12_21"]),
            "del7p12_n":         sum(1 for p in pts if p["del7p12"]),
            "tp53_del_n":        sum(1 for p in pts if p["tp53_del"]),
            "hrd_positive_n":    sum(1 for p in pts if p["hrd_positive"]),
            "craniospinal_rt_n": sum(1 for p in pts if p["craniospinal_rt"]),
            "venetoclax_elig_n": sum(1 for p in pts if p["venetoclax_elig"]),
            "seed":              SEED_BASE + i,
            "key_distinctions":  _gene_dist[gene],
            "pathognomonic":     _gene_pathog[gene],
            "surveillance_key":  _gene_surv[gene],
            "key_rule":          _gene_rule[gene],
        })
    return {"breakdown": breakdown_array, "total_patients": sum(len(v) for v in cohorts.values())}


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

    key_distinctions = [
        "ETV6 germline LOF (hereditary ALL predisposition, THC5 thrombocytopenia) vs ETV6::RUNX1 somatic t(12;21) (most common childhood ALL somatic fusion — GOOD prognosis): germline ETV6 LOF = hereditary predisposition 10-15x; somatic ETV6::RUNX1 = acquired standard-risk childhood B-ALL; a tumour carrying somatic ETV6::RUNX1 does NOT mean the patient has germline ETV6 LOF — thrombocytopenia family history required to distinguish; full germline panel needed if family history",
        "PAX5 germline P80R (familial B-ALL MOST SPECIFIC, 40-80% lifetime risk) vs somatic PAX5 (30% sporadic B-ALL): germline PAX5 P80R = founder mutation in familial B-ALL pedigrees — hereditary autosomal dominant; somatic PAX5 mutations = most frequent somatic ALL alteration after ETV6/RUNX1 — acquired; familial B-ALL cluster (≥2 first-degree relatives) → test PAX5 P80R germline FIRST; somatic ALL result does not inform germline status",
        "IKZF1 germline haploinsufficiency (emerging predisposition, immunodeficiency + B-ALL) vs somatic del7p12.2 IKZF1 (30% B-ALL poor prognosis, Ph-like ALL): somatic del7p12 = most important adverse B-ALL cytogenetic marker; IKZF1plus (somatic IKZF1 + CDKN2A/PAX5/EBF1 co-del) = ultra-high risk → SCT CR1; germline IKZF1 haploinsufficiency = separate hereditary entity (immunodeficiency + B-ALL families); Ph-like ALL: IKZF1 del in 80% + CRLF2/JAK2 → ruxolitinib target",
        "Cranial RT contraindications in hereditary B-ALL: ABSOLUTE CI in TP53 germline (LFS — second brain tumours catastrophic); ATM biallelic A-T (RADIOSENSITIVITY ABSOLUTE); NBN biallelic NBS (RADIOSENSITIVITY ABSOLUTE + neurotoxicity microcephalic brain); AVOID RADIATION ABSOLUTELY LFS patients; IT triple therapy (methotrexate + cytarabine + hydrocortisone) REPLACES CRT in ALL three; TBI conditioning CONTRAINDICATED ATM biallelic, NBN biallelic, TP53 LFS; REDUCE RT 20-30% monoallelic ATM and monoallelic NBN",
        "Alkylating agent rules in hereditary B-ALL: ABSOLUTELY CONTRAINDICATED biallelic BRCA2 FA-D1 (cyclophosphamide, melphalan, busulfan — DNA crosslinks lethal FA-D1 cells); AVOID biallelic NBN NBS (alkylating agents profound bone marrow toxicity MRN-deficient); monoallelic BRCA2 and monoallelic NBN: standard ALL protocols with haematological monitoring; ETV6/PAX5/IKZF1: standard ALL chemotherapy backbone (no alkylating agent restriction); use modified FA-ALL protocols (methotrexate + 6-MP + vincristine + steroids) for FA-D1",
        "CDKN2A dual p16INK4a + p14ARF: CDK4/6 inhibitor palbociclib INACTIVE in p16-null ALL (CDK4/6 already disinhibited — additional inhibition no target); del9p21 somatic 20-35% ALL most common cytogenetic deletion (IKZF1plus co-del with CDKN2A = ultra-high risk); venetoclax ACTIVE regardless CDKN2A status; EUS pancreatic surveillance from 40yr MANDATORY FAMM (17-39x pancreatic risk); abemaciclib CDK4/6 inhibitor investigational ONLY in p16-present ALL (confirmed by IHC)",
        "BRCA2 FA-D1 biallelic vs BRCA2 monoallelic B-ALL: FA-D1 biallelic = B-ALL/AML HIGHEST RISK entire FA pathway (Wilms + medulloblastoma + AML + B-ALL); median onset 2-5yr YOUNGEST in atlas; AVOID ALL alkylating agents biallelic (DNA crosslinks lethal); DEB/MMC chromosomal breakage test PATHOGNOMONIC; monoallelic BRCA2 = B-ALL 2x modest; HRD-positive ALL → PARP inhibitor (olaparib/rucaparib) sensitivity; standard ALL protocols monoallelic (no alkylating restriction); biallelic vs monoallelic distinction CRITICAL for ALL chemotherapy choice",
        "ATM biallelic A-T vs ATM monoallelic B-ALL: biallelic A-T = T-cell ALL 75x HIGHEST (T-lineage V(D)J recombination errors predominate); B-ALL also elevated biallelic; monoallelic germline ATM = B-ALL 2-3x modest; IgA deficiency + AFP elevated + ataxia = A-T PATHOGNOMONIC screening triad; biallelic: RADIOSENSITIVITY ABSOLUTE (no RT, no CRT, no TBI); monoallelic: RT cautiously reduce 20-30%; ceralasertib ATRi synthetic lethality investigational both; PARP inhibitor sensitivity (partial HRD) monoallelic; IVIG A-T immunodeficiency supplementation",
        "NBN 657del5 biallelic NBS ALL vs monoallelic: biallelic NBS = ALL 60x HIGHEST; RADIOSENSITIVITY ABSOLUTE; cranial RT ABSOLUTELY CONTRAINDICATED NBS (neurotoxicity + radiosensitivity + secondary malignancy); monoallelic 657del5 = B-cell ALL 3-5x; RT reduce 20-30% monoallelic; AVOID alkylating agents biallelic NBS; 1:150 Czech/Slovak/Polish — test NBN 657del5 FIRST in Eastern European paediatric ALL; cranial RT NBS = catastrophic irreversible outcome — never administer",
        "Hereditary B-ALL germline panel cascade testing: familial childhood B-ALL or LFS → full panel: ETV6 (most common hereditary ALL gene + THC5), PAX5 P80R (most specific familial B-ALL), IKZF1 (emerging + Ph-like), TP53 (LFS — cranial RT CONTRAINDICATED), ATM (A-T — RT CONTRAINDICATED biallelic), CDKN2A (FAMM + del9p21), NBN 657del5 (Eastern European — cranial RT CONTRAINDICATED biallelic), BRCA2 (FA-D1 — alkylating agents CONTRAINDICATED biallelic); annual CBC + TdT/CD19/CD10 immunophenotyping from early childhood in at-risk carriers; early ALL detection = curative standard paediatric therapy",
    ]

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"ALL RISK: {gd['all_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": key_distinctions,
        "seed_range":                f"{SEED_BASE}–{SEED_BASE + 7}",
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"Total patients: {ov['total_patients']}, B-ALL n: {ov['all_total_n']} ({ov['all_total_pct']}%)")
    print(f"del9p21: {ov['del9p21_n']} ({ov['del9p21_pct']}%), Ph-like: {ov['ph_like_n']} ({ov['ph_like_pct']}%)")
    print(f"t(12;21) ETV6::RUNX1: {ov['t_12_21_n']} ({ov['t_12_21_pct']}%), craniospinal-RT: {ov['craniospinal_rt_n']} ({ov['craniospinal_rt_pct']}%)")
    print(f"HRD-positive: {ov['hrd_positive_n']} ({ov['hrd_positive_pct']}%), venetoclax-eligible: {ov['venetoclax_eligible_n']} ({ov['venetoclax_eligible_pct']}%)")
    print("=== BREAKDOWN ===")
    br = generate_breakdown()
    for b in br["breakdown"]:
        print(f"  {b['gene']}: B-ALL {b['all_pct']}% ({b['all_n']}/{b['n']}), age {b['mean_age_onset']}yr, del9p21 {b['del9p21_n']}, ph-like {b['ph_like_n']}, CRT {b['craniospinal_rt_n']}")
    print("=== DEFINITIONS ===")
    df = generate_definitions()
    print(f"Genes: {list(df['gene_defs'].keys())}")
