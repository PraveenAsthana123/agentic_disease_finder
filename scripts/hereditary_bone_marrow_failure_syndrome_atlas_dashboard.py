#!/usr/bin/env python3
"""Hereditary-Bone-Marrow-Failure-Syndrome-Atlas -- Complete 8-Gene Reference
Inherited BMF syndromes: Fanconi Anemia, Dyskeratosis Congenita, Diamond-Blackfan Anemia,
Shwachman-Diamond Syndrome, Congenital Amegakaryocytic Thrombocytopenia.

FANCA   (FA complementation group A; 1455aa; 16q24.3; AR LOF;
          Most-common-FA-group ~60% cases;
          DEB/MMC-chromosomal-instability-test PATHOGNOMONIC;
          AML-MDS-risk 20-30% lifetime; HSCT-curative-BMF-NOT-cancer;
          seed SEED_BASE+0).
FANCC   (FA complementation group C; 558aa; 9q22.32; AR LOF;
          IVS4+4A>T Ashkenazi-Jewish-founder-PATHOGNOMONIC;
          322delG most-common-non-Jewish;
          Earlier-onset-more-severe-aplasia;
          seed SEED_BASE+1).
FANCG   (FA complementation group G; 622aa; 9p13.3; AR LOF;
          XRCC9; highest-AML-MDS-risk-among-FA-groups;
          Poor-HSCT-outcomes-vs-other-FA-groups;
          seed SEED_BASE+2).
DKC1    (Dyskerin; 514aa; Xq28; XLR LOF;
          DC X-linked; H/ACA-snoRNP-pseudouridine-synthase;
          Mucocutaneous-triad PATHOGNOMONIC nail-dystrophy+oral-leukoplakia+reticulate-skin-pigmentation;
          IPF-dominant-cause-of-death;
          seed SEED_BASE+3).
TERC    (Telomerase-RNA-component; 3q26.2; AD LOF;
          DC/HH/AA; telomere-shortening; heterozygous-sufficient;
          Genetic-anticipation-over-generations PATHOGNOMONIC;
          Aplastic-anemia-20-30pct;
          seed SEED_BASE+4).
RPS19   (Ribosomal-protein-S19; 145aa; 19q13.2; AD LOF haploinsufficiency;
          Diamond-Blackfan-Anemia most-common-gene-25pct;
          Pure-red-cell-aplasia; macrocytosis; elevated-eADA PATHOGNOMONIC;
          Corticosteroids-60-80pct-response;
          seed SEED_BASE+5).
SBDS    (SBDS-protein; 250aa; 7q11.21; AR LOF;
          Shwachman-Diamond-Syndrome;
          Exocrine-pancreatic-insufficiency PATHOGNOMONIC at-birth;
          EIF6-displacement-function; AML-MDS-risk; metaphyseal-dysplasia;
          seed SEED_BASE+6).
MPL     (Thrombopoietin-receptor; 635aa; 1p34.2; AR LOF;
          Congenital-amegakaryocytic-thrombocytopenia-CAMT;
          W515X most-common-CAMT-mutation;
          Severe-thrombocytopenia→bicytopenia→pancytopenia;
          HSCT-ONLY-curative-option;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3694-3701)
"""
import random

SEED_BASE = 3694

ATLAS_GENES = [
    {
        "gene": "FANCA",
        "protein": (
            "FANCA -- 16q24.3 AR LOF -- 1455aa -- "
            "Fanconi-Anemia-complementation-group-A-most-common-FA-60pct-"
            "DEB-MMC-chromosomal-instability-PATHOGNOMONIC-"
            "HSCT-curative-BMF-NOT-cancer-risk-"
            "FANCA-OMIM-607139"
        ),
        "locus": "16q24.3",
        "protein_size": (
            "1455 aa / ~163 kDa / 16q24.3 FANCA Fanconi Anemia Complementation Group A molecular context: "
            "STRUCTURE: "
            "  1455 aa / ~163 kDa; nuclear protein; no enzymatic domain; scaffold/adaptor function; "
            "  FANCA anchors the FA core complex (E3 ubiquitin ligase) to chromatin at DNA interstrand crosslinks; "
            "  Core complex: FANCA-FANCB-FANCC-FANCE-FANCF-FANCG-FANCL-FANCM; "
            "  FANCA mutation → core complex destabilized → FANCD2/FANCI monoubiquitination absent; "
            "  Downstream repair effectors (FANCD1/BRCA2, FANCN/PALB2, FANCJ/BRIP1) not recruited; "
            "  Net effect: unrepaired ICLs → chromosomal breaks → clastogen sensitivity; "
            "CLINICAL PHENOTYPE: "
            "  Aplastic anemia (BMF): onset typically 5-10 years; pancytopenia; "
            "  Congenital anomalies (~75% patients): café-au-lait macules, short stature, radial ray defects "
            "    (absent/hypoplastic thumb; absent radius in severe cases), renal malformations (horseshoe kidney), "
            "    microcephaly, hypogonadism; VACTERL association overlap; "
            "  ~25% patients have NO congenital anomalies (normal phenotype = FA not excluded); "
            "DEB/MMC CHROMOSOMAL INSTABILITY TEST — PATHOGNOMONIC: "
            "  Diepoxybutane (DEB) or Mitomycin C (MMC) clastogenic stress test; "
            "  FA cells: excess chromosomal breaks, radials, exchanges vs normal controls; "
            "  Gold standard diagnostic: whole-blood chromosomal breakage (NOT fibroblast as first-line); "
            "  False-negative: somatic mosaicism (revertant cells); perform PHA-stimulated lymphocytes + fibroblasts; "
            "  Prenatal: DEB on amniocytes (avoid if mosaicism concern → fetal blood sampling); "
            "CANCER RISK: "
            "  AML/MDS: 20-30% lifetime (childhood predominant; MDS → AML evolution); "
            "  Solid tumors (post-HSCT): head/neck SCC, gynecologic cancers (HPV-driven); "
            "  Lifetime solid tumor risk: 30-40%; peaks age 20-40 years; "
            "  HSCT cures BMF but does NOT eliminate solid tumor cancer risk; "
            "  Post-HSCT surveillance: annual head/neck exam + annual gynecologic exam; "
            "HSCT (only curative for BMF): "
            "  HLA-matched sibling first choice (exclude via DEB/MMC + FANC genotyping); "
            "  Reduced-intensity conditioning (RIC) MANDATORY — standard myeloablative → fatal toxicity; "
            "  Fludarabine-based conditioning; avoid radiation (radiosensitivity); "
            "  Unrelated donor outcomes improving with RIC; 5-year OS ~70-80% with matched donor; "
            "ANDROGEN THERAPY (temporizing): "
            "  Oxymetholone or danazol: 50-60% response (partial, not curative); "
            "  Hepatocellular adenoma risk with androgens: monitor LFTs + hepatic imaging; "
            "  No evidence androgens prevent progression to MDS/AML; "
            "TESTING: "
            "  DEB/MMC clastogenic stress test (whole blood): diagnostic (sensitivity ~95%); "
            "  FANC gene panel (FANCA, FANCC, FANCG most common): molecular confirmation; "
            "  FANCA: 60% FA cases; >500 pathogenic variants described; large deletions common (MLPA mandatory); "
            "  Flow cytometry: FANCD2 monoubiquitination assay (functional complement-group assignment)"
        ),
        "syndrome": "Fanconi-Anemia / FA-Group-A / AR-LOF / BMF-Chromosomal-Instability-AML-MDS-Solid-Tumor",
        "inheritance": "AR LOF biallelic (compound heterozygous common); carriers increased breast/ovarian cancer risk",
        "bmf_risk": "Aplastic anemia onset 5-10yr; AML-MDS 20-30% lifetime; solid tumors 30-40% lifetime (HPV-driven SCC post-HSCT)",
        "pathognomonic": "DEB/MMC chromosomal instability test: excess breaks/radials PATHOGNOMONIC; radial ray defects; VACTERL overlap",
        "key_rule": "RIC conditioning MANDATORY for HSCT — standard myeloablative conditioning causes fatal toxicity in FA (radiosensitivity + crosslink hypersensitivity)",
        "key_avoid": "Standard myeloablative HSCT conditioning (fatal toxicity); radiation (radiosensitivity ABSOLUTE); assuming normal phenotype excludes FA (25% have none)",
        "surveillance": "Annual CBC (BMF monitoring); DEB/MMC siblings (cascade); annual head/neck exam post-HSCT; annual gynecologic exam; hepatic imaging if androgens",
        "targeted_rx": "HSCT-RIC (curative for BMF); oxymetholone/danazol (temporizing 50-60% response); G-CSF (neutropenia); EPO (anemia); avoid radiation",
    },
    {
        "gene": "FANCC",
        "protein": (
            "FANCC -- 9q22.32 AR LOF -- 558aa -- "
            "Fanconi-Anemia-group-C-IVS4+4A>T-Ashkenazi-founder-PATHOGNOMONIC-"
            "322delG-most-common-non-Jewish-"
            "Earlier-onset-severe-aplasia-"
            "FANCC-OMIM-227645"
        ),
        "locus": "9q22.32",
        "protein_size": (
            "558 aa / ~63 kDa / 9q22.32 FANCC Fanconi Anemia Complementation Group C molecular context: "
            "STRUCTURE: "
            "  558 aa / ~63 kDa; cytoplasmic anchor of FA core complex; "
            "  FANCC shuttles between cytoplasm and nucleus; binds FANCE; "
            "  Three functional domains: N-terminal FAAP100 interaction; central FANCF-binding; C-terminal FANCE-binding; "
            "  FANCC also has non-canonical roles: oxidative stress response, apoptosis regulation; "
            "FOUNDER MUTATIONS — DIAGNOSTIC PEARLS: "
            "  IVS4+4A>T (c.456+4A>T): Ashkenazi Jewish founder mutation; ~30% Ashkenazi FA; "
            "    heterozygous carrier frequency ~1/89 Ashkenazi Jews; "
            "    PATHOGNOMONIC in Ashkenazi: test for IVS4+4A>T FIRST in Ashkenazi patients; "
            "    Phenotype: severe; early BMF onset (median 5yr); high AML risk; "
            "  322delG (c.67delG): most common mutation in non-Jewish populations worldwide; "
            "    severe exon 1 frameshift; truncated/absent protein; severe phenotype; "
            "  IVS4+4A>T + 322delG together: most FA-C compound heterozygotes; "
            "CLINICAL PHENOTYPE (FA-C): "
            "  Generally WORSE prognosis vs FA-A: earlier BMF onset; higher AML risk; "
            "  Congenital anomalies: present in ~70%; café-au-lait, radial ray in severe cases; "
            "  Endocrine: growth hormone deficiency, hypothyroidism, insulin resistance; "
            "  Leukemia: FA-C has disproportionately high AML/MDS transformation rate vs FA-A; "
            "SOMATIC MOSAICISM (diagnostic trap): "
            "  ~15-20% FA patients have somatic mosaicism (reversion mutations restoring one allele); "
            "  Reverted cells may rescue CBC; patient appears NOT to have aplasia; "
            "  DEB/MMC on blood: false-negative if high revertant fraction; "
            "  Solution: DEB/MMC on both lymphocytes AND fibroblasts; "
            "  Mosaicism provides natural history model for gene therapy; "
            "FANCONI ANEMIA CARRIER SCREENING (Jewish population): "
            "  Ashkenazi Jewish carrier screening: include FANCC-IVS4+4A>T; "
            "  Expanded panels (JScreen, Dor Yeshorim) routinely include FANCC; "
            "  Pre-conception or early prenatal; IVF + PGT-M available; "
            "TESTING: "
            "  DEB/MMC (gold standard); FANCC mutation analysis (IVS4+4A>T + 322delG first in Ashkenazi); "
            "  FANC gene panel; complementation group assignment by FANCD2-Ub assay; "
            "  Siblings: MANDATORY DEB/MMC (potential HSCT donors)"
        ),
        "syndrome": "Fanconi-Anemia / FA-Group-C / AR-LOF / Ashkenazi-IVS4+4A>T-Founder / Earlier-BMF-Onset / High-AML-Risk",
        "inheritance": "AR LOF biallelic; IVS4+4A>T Ashkenazi carrier ~1/89; 322delG non-Jewish founder",
        "bmf_risk": "Early BMF onset (median ~5yr); AML-MDS rate higher than FA-A; solid tumors similar to FA-A",
        "pathognomonic": "IVS4+4A>T mutation in Ashkenazi Jewish patient with aplasia PATHOGNOMONIC FA-C; DEB/MMC excess breaks",
        "key_rule": "Test IVS4+4A>T FIRST in Ashkenazi Jewish patients with BMF — Ashkenazi founder mutation; somatic mosaicism causes false-negative DEB/MMC on blood alone",
        "key_avoid": "Relying on blood-only DEB/MMC (mosaicism false-negative); delaying HSCT past severe aplasia; non-RIC conditioning",
        "surveillance": "Annual CBC; serial DEB/MMC if mosaicism suspected; bone marrow biopsy annually (MDS/AML surveillance); sibling DEB/MMC",
        "targeted_rx": "HSCT-RIC (curative for BMF); oxymetholone/danazol (temporizing); eltrombopag (TPO-RA investigational); PGT-M for subsequent pregnancies",
    },
    {
        "gene": "FANCG",
        "protein": (
            "FANCG -- 9p13.3 AR LOF -- 622aa -- "
            "Fanconi-Anemia-group-G-XRCC9-"
            "Highest-AML-MDS-risk-among-FA-groups-"
            "Poor-HSCT-outcomes-relative-"
            "FANCG-OMIM-602956"
        ),
        "locus": "9p13.3",
        "protein_size": (
            "622 aa / ~68 kDa / 9p13.3 FANCG Fanconi Anemia Complementation Group G (XRCC9) molecular context: "
            "STRUCTURE: "
            "  622 aa / ~68 kDa; WD40 repeat protein (tetratricopeptide-repeat related); "
            "  FANCG stabilizes FANCA within the FA core complex; "
            "  FANCG TPR domain: interfaces with FANCA; FANCG absent → FANCA nuclear import reduced; "
            "  Also designated XRCC9 (X-ray repair cross-complementing protein 9); "
            "FA-G CLINICAL PHENOTYPE (KEY DISTINCTIONS): "
            "  Disproportionately HIGH AML/MDS risk vs FA-A and FA-C groups; "
            "  AML transformation rate estimated 20-35% lifetime in FA-G; "
            "  Phenotype: variable congenital anomalies; BMF onset similar to FA-A (5-10yr); "
            "  BUT: solid tumor risk profile ALSO ELEVATED vs FA-A; "
            "  Endocrine: growth retardation, hypogonadotropic hypogonadism more common; "
            "FA-G HSCT OUTCOMES: "
            "  HSCT outcomes in FA-G historically WORSE than FA-A with RIC conditioning; "
            "  Mechanism unclear: possibly greater clonal evolution pre-HSCT; "
            "  Higher TRM (treatment-related mortality) with unrelated donor; "
            "  Early HSCT before aplasia is severe or MDS-clonal-evolution preferred; "
            "MONOSOMY 7 / CLONAL EVOLUTION: "
            "  FA patients with monosomy 7 on BM cytogenetics = IMMINENT AML risk; "
            "  FA-G: monosomy 7 appears earlier and more commonly than FA-A; "
            "  Monosomy 7 in FA → URGENT HSCT referral; do NOT observe; "
            "  MDS in FA: classified by modified WHO criteria (FA-specific); "
            "  RAEB (refractory anemia excess blasts) in FA → AML within months without HSCT; "
            "CLASTOGEN SENSITIVITY ASSAY: "
            "  DEB/MMC: same diagnostic utility as FA-A/C; "
            "  FANCG complementation: functional assay — retroviral FANCG restores DEB resistance; "
            "  Genotype: >50 pathogenic FANCG variants; c.313G>T (p.Glu105Ter) founder Japan/Korea; "
            "TESTING: "
            "  DEB/MMC (diagnostic); BM biopsy + cytogenetics ANNUALLY (high AML evolution); "
            "  FANCG sequencing; complementation group assignment; "
            "  BM cytogenetics: monosomy 7 = emergency escalation to HSCT"
        ),
        "syndrome": "Fanconi-Anemia / FA-Group-G / XRCC9 / AR-LOF / Highest-AML-MDS-Risk-among-FA / Poor-HSCT-Outcomes",
        "inheritance": "AR LOF biallelic; Japanese/Korean founder c.313G>T (p.Glu105Ter)",
        "bmf_risk": "BMF onset 5-10yr; AML-MDS risk HIGHEST among FA groups 20-35%; monosomy 7 = URGENT HSCT",
        "pathognomonic": "Monosomy 7 on BM cytogenetics in FA patient = IMMINENT AML — URGENT HSCT; DEB/MMC excess breaks",
        "key_rule": "Annual BM biopsy with cytogenetics MANDATORY in FA-G — monosomy 7 triggers URGENT HSCT referral; HSCT BEFORE MDS/AML evolution",
        "key_avoid": "Observing monosomy 7 without urgent HSCT action; delaying HSCT in FA-G (worst outcomes after MDS); standard-intensity conditioning",
        "surveillance": "Annual BM biopsy + cytogenetics (monosomy 7 screen); annual CBC; sibling DEB/MMC; clonal evolution marker panel",
        "targeted_rx": "HSCT-RIC (curative; early preferred); G-CSF (short-term neutropenia bridge); androgens (temporizing only); urgent HSCT trigger: monosomy 7 or RAEB",
    },
    {
        "gene": "DKC1",
        "protein": (
            "DKC1 -- Xq28 XLR LOF -- 514aa -- "
            "Dyskeratosis-Congenita-X-linked-Zinsser-Engman-Cole-"
            "Mucocutaneous-triad-nail-dystrophy+oral-leukoplakia+reticulate-pigmentation-PATHOGNOMONIC-"
            "IPF-pulmonary-fibrosis-dominant-cause-of-death-"
            "DKC1-OMIM-305000"
        ),
        "locus": "Xq28",
        "protein_size": (
            "514 aa / ~58 kDa / Xq28 DKC1 Dyskerin molecular context: "
            "STRUCTURE: "
            "  514 aa / ~58 kDa; pseudouridine synthase / transferase superfamily; "
            "  DKC1 is the catalytic subunit of H/ACA small nucleolar ribonucleoproteins (snoRNPs); "
            "  H/ACA snoRNPs: pseudouridylate rRNA + snRNA; guide telomerase RNA (TERC) stability; "
            "  DKC1 mutations → TERC instability → telomere shortening → BMF/IPF/cancer; "
            "  DKC1 also functions in rRNA processing (ribosome biogenesis) — dual mechanism; "
            "  Pseudouridylation (psi): most abundant RNA modification; required for ribosome translational fidelity; "
            "MUCOCUTANEOUS TRIAD — PATHOGNOMONIC (classic DC): "
            "  1. Nail dystrophy: longitudinal ridging → onycholysis → complete nail loss; earliest sign (childhood); "
            "  2. Oral leukoplakia: white patches (often on lateral tongue/buccal mucosa); malignant transformation risk; "
            "  3. Reticulate skin pigmentation: lacy/reticulate hyperpigmentation on face/neck/upper chest; "
            "  All three TOGETHER: pathognomonic for Dyskeratosis Congenita; "
            "  Present in ~90% classic XL-DC males; may be incomplete in some; "
            "  Onset of triad: typically 5-15 years; BMF usually follows; "
            "SERIOUS COMPLICATIONS (beyond BMF): "
            "  Pulmonary fibrosis (IPF-like): DOMINANT CAUSE OF DEATH in DC; 20-30% lifetime; "
            "  Mechanism: telomere dysfunction in alveolar epithelial cells → progressive fibrosis; "
            "  Hepatic fibrosis/cirrhosis: portal hypertension; "
            "  Esophageal stenosis: dysphagia; may require dilation; "
            "  Urethral/lacrimal duct stenosis; "
            "  Avascular necrosis (hip); "
            "  HSCT-LUNG INTERACTION CRITICAL: "
            "    HSCT conditioning accelerates pulmonary fibrosis in DC patients; "
            "    Avoid bleomycin (pulmonary toxicity synergistic); "
            "    Post-HSCT: IPF remains — HSCT does NOT prevent or cure pulmonary fibrosis; "
            "CANCER RISK: "
            "  Mucocutaneous SCC (mouth, tongue, skin): 10-20x elevated; "
            "  GI carcinoma; AML/MDS after BMF; "
            "  Oral leukoplakia: regular oral surgery surveillance MANDATORY; "
            "TELOMERE LENGTH TESTING: "
            "  Flow-FISH (fluorescence in situ hybridization with flow cytometry): gold standard; "
            "  Lymphocyte telomere length <1st percentile for age: diagnostic; "
            "  Clinically available through specialized labs (National Cancer Institute); "
            "  qPCR telomere length: lower sensitivity but more accessible; "
            "ANDROGENS in DC: "
            "  Danazol 400-800mg/day: ~70% response in DC-related aplasia; "
            "  Danazol mechanism: upregulates TERT (telomerase reverse transcriptase) expression; "
            "  Monitoring: LFT, lipids, hepatic ultrasound; hepatocellular adenoma risk; "
            "TESTING: "
            "  Mucocutaneous triad clinical exam; CBC; BM biopsy; "
            "  Telomere length (flow-FISH): <1st percentile diagnostic; "
            "  DKC1 sequencing (Xq28); "
            "  Female carriers (X-linked): may have mild features; skewed X-inactivation"
        ),
        "syndrome": "Dyskeratosis-Congenita / X-linked-Zinsser-Engman-Cole / DKC1 / XLR-LOF / Mucocutaneous-Triad / IPF-Dominant-Death",
        "inheritance": "XLR LOF (males severely affected; females carriers with variable/mild features; skewed X-inactivation)",
        "bmf_risk": "BMF onset 10-20yr; AML-MDS-SCC risk elevated; IPF dominant cause of death 20-30%",
        "pathognomonic": "Mucocutaneous TRIAD (nail dystrophy + oral leukoplakia + reticulate pigmentation) PATHOGNOMONIC DC; telomere length <1st percentile diagnostic",
        "key_rule": "HSCT does NOT prevent pulmonary fibrosis — IPF continues post-transplant; avoid bleomycin; use RIC conditioning; monitor pulmonary function annually",
        "key_avoid": "Standard myeloablative conditioning (fatal toxicity); bleomycin (pulmonary synergy); delaying oral leukoplakia surveillance (SCC transformation); missing IPF monitoring",
        "surveillance": "Annual PFTs (IPF screening); annual oral exam (leukoplakia/SCC); annual CBC + BM biopsy; echocardiogram (pulmonary HTN); LFT if on androgens",
        "targeted_rx": "Danazol (70% aplasia response); HSCT-RIC (curative for BMF, NOT IPF); nintedanib/pirfenidone (IPF treatment); oral leukoplakia: surgical excision",
    },
    {
        "gene": "TERC",
        "protein": (
            "TERC -- 3q26.2 AD LOF -- 451nt-RNA-component -- "
            "Dyskeratosis-Congenita-Aplastic-Anemia-Hoyeraal-Hreidarsson-"
            "Genetic-anticipation-over-generations-PATHOGNOMONIC-"
            "Heterozygous-LOF-sufficient-for-disease-AD-"
            "TERC-OMIM-602322"
        ),
        "locus": "3q26.2",
        "protein_size": (
            "451 nucleotide RNA / non-protein-coding / 3q26.2 TERC Telomerase RNA Component molecular context: "
            "STRUCTURE: "
            "  451-nucleotide non-coding RNA; RNA component of telomerase holoenzyme; "
            "  Template region (nt 46-56): provides the AAUCCC template for telomere repeat addition by TERT; "
            "  H/ACA box: 3' pseudoknot; binds DKC1/NHP2/NOP10 complex for stability and Cajal body localization; "
            "  CR4-CR5 domain: binds TERT catalytic domain; required for telomerase activity; "
            "  Haploinsufficiency: single mutant TERC allele → ~50% telomerase activity → telomere shortening; "
            "  Each generation inherits shorter telomeres → EARLIER AND MORE SEVERE disease; "
            "GENETIC ANTICIPATION — PATHOGNOMONIC: "
            "  TERC AD LOF → disease severity WORSENS over successive generations; "
            "  Generation 1 (grandparent): mild BMF or aplastic anemia in 40s-50s; "
            "  Generation 2 (parent): moderate DC with aplasia in 20s-30s; "
            "  Generation 3 (proband): severe DC/HH with early-onset aplasia in childhood; "
            "  Mechanism: successive generations inherit shorter and shorter telomeres; "
            "  CLINICAL IMPLICATION: family pedigree across 3 generations can reveal anticipation pattern; "
            "  Hoyeraal-Hreidarsson (HH): severe cerebellar hypoplasia + enteropathy + IUGR + DC features; "
            "  HH is the most severe end of TERC/DKC1 spectrum; "
            "DISEASE SPECTRUM (TERC/telomeropathies): "
            "  Idiopathic aplastic anemia (IAA): 5-10% of 'idiopathic' AA are telomeropathy-driven; "
            "  IPF (idiopathic pulmonary fibrosis): TERC mutations in familial IPF; "
            "  Dyskeratosis Congenita: full triad ± aplasia; "
            "  Hoyeraal-Hreidarsson: severe prenatal/neonatal form; "
            "  SHORT TELOMERE PHENOTYPES overlap: liver cirrhosis, AA, IPF can each be presenting feature; "
            "TELOMERE TESTING IN TERC: "
            "  Flow-FISH: lymphocyte telomere length <1st percentile for age confirms telomeropathy; "
            "  TERC sequencing: variants in template/H-ACA/CR4-CR5 regions; deletions by MLPA; "
            "  Family testing: all first-degree relatives of TERC mutation carrier; "
            "  Older generations may have MINIMAL symptoms but carry mutation; "
            "TREATMENT: "
            "  Danazol/oxymetholone: upregulates TERT; 70% response in telomeropathy-aplasia; "
            "  HSCT-RIC: curative for BMF; does NOT cure IPF; RIC essential (radiosensitivity); "
            "  Avoid alkylating agents (additional telomere stress); "
            "  No lung transplant post-HSCT data — IPF management same as sporadic IPF; "
            "TESTING: "
            "  Telomere flow-FISH; TERC sequencing; family cascade testing mandatory"
        ),
        "syndrome": "Dyskeratosis-Congenita-Aplastic-Anemia / TERC-telomerase-RNA / AD-LOF / Genetic-Anticipation-PATHOGNOMONIC / IPF-AA",
        "inheritance": "AD LOF (heterozygous sufficient for disease; haploinsufficiency); anticipation — worsens each generation",
        "bmf_risk": "Aplastic anemia 20-30% (can be presenting feature); IPF 20-30%; AML/MDS lower than FA; solid tumors elevated",
        "pathognomonic": "Genetic anticipation across 3 generations (earlier/severer disease each generation) PATHOGNOMONIC telomeropathy; telomere <1st percentile diagnostic",
        "key_rule": "Family pedigree across 3 generations showing anticipation pattern = PATHOGNOMONIC telomeropathy — test TERC/TERT/DKC1; 5-10% 'idiopathic' aplastic anemia = telomeropathy",
        "key_avoid": "Missing anticipation pattern in family history (misdiagnosis as idiopathic AA); aggressive conditioning (avoid alkylators); not testing family members",
        "surveillance": "Annual PFTs; annual CBC; BM biopsy if cytopenias; cascade family telomere testing; echocardiogram (pulmonary HTN); liver ultrasound (cirrhosis)",
        "targeted_rx": "Danazol (TERT upregulation; 70% aplasia response); HSCT-RIC (BMF curative); anti-fibrotic (nintedanib/pirfenidone for IPF); etelcalcetide investigational",
    },
    {
        "gene": "RPS19",
        "protein": (
            "RPS19 -- 19q13.2 AD LOF haploinsufficiency -- 145aa -- "
            "Diamond-Blackfan-Anemia-most-common-gene-25pct-DBA-"
            "Pure-red-cell-aplasia-macrocytosis-elevated-eADA-PATHOGNOMONIC-"
            "Corticosteroids-60-80pct-first-response-"
            "RPS19-OMIM-105650"
        ),
        "locus": "19q13.2",
        "protein_size": (
            "145 aa / ~17 kDa / 19q13.2 RPS19 Ribosomal Protein S19 molecular context: "
            "STRUCTURE: "
            "  145 aa / ~17 kDa; component of the 40S small ribosomal subunit; "
            "  RPS19 interacts with RPS14 and RPS31; required for pre-18S rRNA processing (21S→18S); "
            "  Haploinsufficiency: 50% RPS19 → impaired 40S assembly → p53 pathway activation → erythroid apoptosis; "
            "  Erythroid progenitors (BFU-E, CFU-E) are SELECTIVELY hypersensitive to ribosome stress; "
            "  Other cell lineages: relatively spared (ribosome stress tolerance higher in non-erythroid cells); "
            "  ~25% of all DBA caused by RPS19; second most common: RPL5 (7%), RPL11 (5%); "
            "  Many RPS genes implicated: RPS7, RPS10, RPS15A, RPS17, RPS24, RPS26, RPS27A, RPS28, RPS29; "
            "DIAMOND-BLACKFAN ANEMIA — CLINICAL DIAGNOSIS: "
            "  Age of presentation: typically in first year of life (often 2-6 months); "
            "  Profound macrocytic anemia with RETICULOCYTOPENIA; "
            "  BM: pure red cell aplasia (erythroid precursors absent/severely reduced; myeloid and megakaryocytic NORMAL); "
            "  Fetal hemoglobin (HbF): elevated for age; i-antigen expression elevated; "
            "  ADA (erythrocyte adenosine deaminase): ELEVATED — PATHOGNOMONIC; "
            "    Normal range: 0.7-2.0 mU/g Hb; DBA: typically >3.0 mU/g Hb; "
            "    Elevated eADA seen in ALL RP-gene DBA (not just RPS19); "
            "    Used as screening test before genetic panel; "
            "  Congenital anomalies: 47% DBA — triphalangeal thumbs (PATHOGNOMONIC), cleft palate, cardiac defects (VSD), "
            "    short stature, web neck, Holt-Oram-like defects; "
            "    Triphalangeal thumb in a macrocytic anemia infant = PATHOGNOMONIC DBA; "
            "CORTICOSTEROID RESPONSE: "
            "  First-line: prednisolone 2mg/kg/day; "
            "  Response rate: 60-80% initially; some children become transfusion-independent; "
            "  Long-term steroid-free remission: 20-30% (spontaneous remission also occurs ~20%); "
            "  Steroid refractoriness: dose-limiting toxicity (growth, bone density, metabolic); "
            "  Minimal steroid threshold: taper to lowest dose maintaining Hb ≥10 g/dL; "
            "CANCER RISK (DBA): "
            "  AML, MDS, ALL: elevated 2-5x vs general population; "
            "  Osteosarcoma, colon cancer: elevated; "
            "  Lifetime cancer risk: ~20-30%; "
            "TREATMENT OPTIONS: "
            "  1. Corticosteroids (first-line): prednisolone 2mg/kg; "
            "  2. Transfusions (steroid-refractory or <1yr age; avoid steroids <1yr); "
            "    Iron chelation MANDATORY if >10-15 transfusions (desferrioxamine or deferasirox); "
            "  3. Leucine (amino acid): phase II data — modest erythroid stimulation; "
            "  4. Luspatercept (ACVR2B-Fc; TGF-beta trap): phase II DBA; stimulates late erythropoiesis; "
            "  5. HSCT: curative (matched sibling preferred); indicated for steroid/transfusion-refractory; "
            "TESTING: "
            "  CBC + reticulocyte count; BM biopsy (erythroid hypoplasia); "
            "  eADA (elevated: screening); HbF (elevated for age); "
            "  RP gene panel (RPS19 first, then full panel); "
            "  Array CGH/MLPA: copy number variants in RP genes; "
            "  Targeted variant: de novo in 45%, familial AD in 55%"
        ),
        "syndrome": "Diamond-Blackfan-Anemia / RPS19 / AD-LOF-haploinsufficiency / Pure-RCA / Elevated-eADA-PATHOGNOMONIC / Triphalangeal-Thumb",
        "inheritance": "AD LOF haploinsufficiency (45% de novo; 55% familial); variable expressivity in family members",
        "bmf_risk": "Pure red cell aplasia from infancy; AML-MDS-osteosarcoma risk elevated 2-5x; lifetime cancer risk 20-30%",
        "pathognomonic": "Elevated erythrocyte ADA (eADA >3 mU/g Hb) PATHOGNOMONIC DBA; triphalangeal thumb + macrocytic anemia in infant = DBA until proven otherwise",
        "key_rule": "Do NOT start corticosteroids before age 1 year — steroid side effects (growth suppression, metabolic) outweigh benefit in infants; transfusions + iron chelation until age 1",
        "key_avoid": "Steroids in infancy <1yr (growth harm); delaying iron chelation after 10-15 transfusions; missing eADA testing (diagnoses DBA before genetics available)",
        "surveillance": "Annual CBC; eADA + HbF (treatment response); annual growth/development (steroid effects); iron studies + ferritin if transfusion-dependent; echocardiogram (iron overload)",
        "targeted_rx": "Prednisolone 2mg/kg (first-line; 60-80% response); transfusions + deferasirox (steroid-refractory); leucine supplementation; luspatercept (investigational); HSCT (curative for refractory)",
    },
    {
        "gene": "SBDS",
        "protein": (
            "SBDS -- 7q11.21 AR LOF -- 250aa -- "
            "Shwachman-Diamond-Syndrome-"
            "Exocrine-pancreatic-insufficiency-at-birth-PATHOGNOMONIC-"
            "EIF6-displacement-ribosome-biogenesis-"
            "AML-MDS-risk-metaphyseal-dysplasia-"
            "SBDS-OMIM-260400"
        ),
        "locus": "7q11.21",
        "protein_size": (
            "250 aa / ~29 kDa / 7q11.21 SBDS Shwachman-Bodian-Diamond-Syndrome protein molecular context: "
            "STRUCTURE: "
            "  250 aa / ~29 kDa; cytoplasmic and nuclear protein; "
            "  SBDS cooperates with EFL1 (elongation factor-like 1) GTPase to displace eIF6 from nascent 60S ribosomal subunit; "
            "  eIF6 normally prevents premature 60S-40S joining; SBDS/EFL1 displace eIF6 → activate 60S → allow 80S assembly; "
            "  SBDS mutation → eIF6 remains bound → 80S ribosome assembly impaired → p53 → myeloid/exocrine apoptosis; "
            "  SBDS-pseudogene (SBDSP, 7q11.21): recombination between SBDS and SBDSP creates gene conversion mutations; "
            "  Most common mutations: c.183-184TA>CT (exon 2; p.Lys62Arg; ~75% alleles worldwide); "
            "    c.258+2T>C (exon 2 splice; ~10%); both arise from SBDS/SBDSP gene conversion; "
            "CLINICAL TRIAD OF SDS: "
            "  1. EXOCRINE PANCREATIC INSUFFICIENCY (EPI): PATHOGNOMONIC presenting feature at birth; "
            "    Lipase, amylase, elastase severely reduced; steatorrhea, failure to thrive, fat-soluble vitamin deficiency; "
            "    EPI IMPROVES with age in ~50% (pancreatic acinar cell recovery); "
            "    Diagnosis: fecal elastase <100 mcg/g (abnormal); pancreatic secretin stimulation test; "
            "    Treatment: PERT (pancreatic enzyme replacement therapy) until spontaneous improvement confirmed; "
            "  2. CYTOPENIAS (neutropenia predominant): "
            "    Neutropenia: cyclic or chronic; ANC <1.5 in 100% of SDS; ANC <0.5 in ~50%; "
            "    Neutropenia is intermittent (episodic) — ANC can normalize then drop; "
            "    Thrombocytopenia: 70%; anemia: 40%; "
            "    BMF (aplastic anemia): 10-15% develop severe aplasia requiring HSCT; "
            "  3. METAPHYSEAL DYSPLASIA: "
            "    Metaphyseal chondrodysplasia: irregular metaphyses (distal femur, proximal tibia); "
            "    Short stature; costochondral junction abnormality (ribs); "
            "    Dx: skeletal survey X-rays; "
            "AML/MDS RISK IN SDS: "
            "  MDS: 20-30% lifetime (isochromosome 7q [i(7q)] or del20q — SDS-SPECIFIC); "
            "  i(7q) in SDS: NOT always MDS-equivalent (may be benign clonal evolution in SDS); "
            "    CRITICAL DISTINCTION: i(7q) in non-SDS = high-risk MDS; in SDS = may be non-malignant; "
            "  del20q in SDS: similarly may be benign; "
            "  AML: 5-10% lifetime; "
            "  Annual BM biopsy + cytogenetics with dedicated SDS-cytogenetic interpretation MANDATORY; "
            "OTHER FEATURES: "
            "  Hepatomegaly + elevated LFTs: 70%; normalizes with age; "
            "  Cognitive-behavioral: learning disabilities, ADHD-spectrum; "
            "  Dental: caries, abnormal dentition; "
            "TESTING: "
            "  Fecal elastase (EPI diagnosis); CBC; SBDS sequencing + MLPA (gene conversion panel); "
            "  Skeletal survey; annual BM biopsy + cytogenetics with SDS-expert review; "
            "  Vitamin ADEK levels (fat-soluble vitamin deficiency from EPI)"
        ),
        "syndrome": "Shwachman-Diamond-Syndrome / SBDS / AR-LOF / Exocrine-Pancreatic-Insufficiency-at-Birth / Metaphyseal-Dysplasia / AML-MDS",
        "inheritance": "AR LOF biallelic; gene conversion from SBDSP pseudogene; c.183-184TA>CT ~75% alleles worldwide",
        "bmf_risk": "Neutropenia 100% (cyclic/chronic); BMF (aplastic) 10-15%; AML-MDS 20-30%; i(7q)/del20q may be benign in SDS context",
        "pathognomonic": "Exocrine pancreatic insufficiency at birth + neutropenia + metaphyseal dysplasia = PATHOGNOMONIC SDS triad; fecal elastase <100 mcg/g",
        "key_rule": "i(7q) or del20q in SDS BM cytogenetics ≠ automatic MDS — these are SDS-specific benign clonal events; require SDS-specialist BM interpretation, NOT standard MDS criteria",
        "key_avoid": "Overdiagnosing MDS in SDS based on i(7q)/del20q alone (SDS-specific benign clones); stopping PERT without confirming EPI improvement; missing fat-soluble vitamin supplementation",
        "surveillance": "Annual BM biopsy + cytogenetics (SDS-expert); annual fecal elastase (EPI monitoring); vitamins ADEK levels; skeletal survey in childhood; CBC monthly (neutropenia monitoring)",
        "targeted_rx": "PERT (pancreatic enzyme replacement — first-line EPI); fat-soluble vitamins ADEK; G-CSF (neutropenia: short-course, not continuous — MDS risk); HSCT (BMF/AML-MDS); eltrombopag (investigational)",
    },
    {
        "gene": "MPL",
        "protein": (
            "MPL -- 1p34.2 AR LOF -- 635aa -- "
            "Congenital-Amegakaryocytic-Thrombocytopenia-CAMT-"
            "W515X-most-common-CAMT-mutation-"
            "Severe-thrombocytopenia→bicytopenia→pancytopenia-progression-"
            "HSCT-ONLY-curative-option-"
            "MPL-OMIM-604498"
        ),
        "locus": "1p34.2",
        "protein_size": (
            "635 aa / ~84 kDa / 1p34.2 MPL Thrombopoietin Receptor (c-Mpl) molecular context: "
            "STRUCTURE: "
            "  635 aa / ~84 kDa; type I cytokine receptor; hematopoietic progenitor and platelet surface; "
            "  Extracellular: 2 cytokine receptor homology (CRH) domains; TPO binding domain; "
            "  Transmembrane domain: 22 aa; "
            "  Intracellular: JAK2-binding box (Box1/Box2); STAT5 activation downstream; "
            "  MPL LOF → JAK2/STAT5 signaling absent → megakaryocyte/platelet failure; "
            "  TPO (thrombopoietin): primary hematopoietic cytokine for platelet production; "
            "  MPL mutation → no TPO signal → amegakaryocytosis → severe thrombocytopenia; "
            "  MPL also required for HSC self-renewal (TPO-MPL → HSC quiescence/survival); "
            "  MPL LOF: HSC pool PROGRESSIVELY depleted → bicytopenia → pancytopenia → aplastic anemia; "
            "CAMT CLINICAL COURSE — CRITICAL PATTERN: "
            "  CAMT TYPE I (null MPL mutations: nonsense, frameshift, splice-site): "
            "    Severe thrombocytopenia at birth (platelets <20k consistently); "
            "    Progression to bicytopenia + pancytopenia by age 3-5 INVARIABLY; "
            "    HSCT MUST occur before severe aplasia develops; "
            "  CAMT TYPE II (missense MPL mutations: some residual MPL signaling): "
            "    Transient partial remission in thrombocytopenia possible (age 1-2 years); "
            "    Relapses into severe thrombocytopenia; "
            "    Progression to aplasia: later than Type I but still inevitable; "
            "    HSCT required: typically by 5-10 years; "
            "  TYPE I vs TYPE II DISTINCTION: "
            "    W515X (Trp515Ter): Type I null — c.1544G>A; most common CAMT mutation; "
            "    R102P: Type II missense; transient improvement phase; "
            "HEMORRHAGIC COMPLICATIONS: "
            "  Intracranial hemorrhage (ICH): 25-35% of CAMT patients; "
            "    Extremely high-risk first year of life; "
            "    ICH prevention: prophylactic platelet transfusions to maintain >50k in infancy; "
            "  GI bleeding, retinal hemorrhage; mucocutaneous bleeding; "
            "ELEVATED TPO — DIAGNOSTIC MARKER: "
            "  CAMT: serum TPO MARKEDLY elevated (no MPL signal → no TPO consumption); "
            "  Diagnostic ratio: TPO elevated vs normal platelet count expected (paradox); "
            "  TPO >1000 pg/mL typical in CAMT; normal range <100 pg/mL; "
            "  Differential from ITP: TPO ELEVATED in CAMT, normal/low in ITP; "
            "  CRITICAL: do NOT mistake for ITP and treat with IVIG/steroids (TPO test first); "
            "HSCT (ONLY CURATIVE OPTION): "
            "  CAMT has NO medical cure — HSCT is mandatory; no TPO-RA works (MPL absent); "
            "  TPO-receptor agonists (eltrombopag, romiplostim): INACTIVE in CAMT (no receptor); "
            "  HSCT before aplasia = best outcomes; "
            "  HLA-matched sibling donor preferred; unrelated donor (MUD) if no sibling; "
            "  RIC conditioning acceptable; OS >80% with early matched-donor HSCT; "
            "  Post-HSCT: platelet engraftment confirms functional MPL in donor cells; "
            "TESTING: "
            "  CBC + platelet count (severe isolated thrombocytopenia at birth); "
            "  BM biopsy: amegakaryocytosis (absent/severely reduced megakaryocytes) PATHOGNOMONIC; "
            "  Serum TPO level (markedly elevated); "
            "  MPL sequencing: W515X + full gene; "
            "  Exclude: ITP (normal BM megakaryocytes, normal TPO, maternal antibodies); "
            "  Siblings: typing for HSCT; cascade MPL testing"
        ),
        "syndrome": "Congenital-Amegakaryocytic-Thrombocytopenia / CAMT / MPL / AR-LOF / Severe-Thrombocytopenia→Aplasia / HSCT-Only-Cure",
        "inheritance": "AR LOF biallelic; W515X most common; Type I (null) severe > Type II (missense) milder course",
        "bmf_risk": "Severe thrombocytopenia from birth; Type I: pancytopenia by age 3-5; ICH risk 25-35%; aplastic anemia inevitable without HSCT",
        "pathognomonic": "Amegakaryocytosis on BM biopsy + markedly elevated serum TPO + severe neonatal thrombocytopenia = PATHOGNOMONIC CAMT; TPO-RAs (eltrombopag) INACTIVE in CAMT",
        "key_rule": "TPO-receptor agonists (eltrombopag, romiplostim) are INACTIVE in CAMT — MPL is absent; HSCT is the ONLY cure; measure serum TPO before treating neonatal thrombocytopenia with IVIG (to exclude CAMT from ITP)",
        "key_avoid": "Treating CAMT as ITP (steroids/IVIG/TPO-RAs — all ineffective); delaying HSCT until aplasia is severe (worse outcomes); missing ICH prevention in infancy (platelets <50k target)",
        "surveillance": "Weekly CBC in infancy (ICH risk); serum TPO (diagnostic); BM biopsy (amegakaryocytosis); HLA typing siblings; platelet transfusions PRN (ICH prevention target >50k first year)",
        "targeted_rx": "Platelet transfusions (ICH prophylaxis; target >50k first year of life); HSCT-RIC (only cure; before aplasia); NO TPO-RAs (inactive); NO steroids/IVIG (ineffective for CAMT)",
    },
]


def _make_cohort(gene_idx: int, seed: int) -> dict:
    """Generate deterministic 40-patient cohort for one BMF gene."""
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]
    n = 40

    ages = sorted([rng.randint(0, 25) for _ in range(n)])  # BMF often presents in childhood
    mean_age = round(sum(ages) / n, 1)

    # Gene-specific clinical parameters
    g = gene["gene"]
    if g in ("FANCA", "FANCC", "FANCG"):
        # Fanconi Anemia: aplastic anemia + AML risk
        aplasia_count = rng.randint(22, 32)
        aml_mds_count = rng.randint(6, 14)
        hsct_count = rng.randint(18, 28)
        anomaly_count = rng.randint(25, 35) if g != "FANCC" else rng.randint(26, 36)
        deb_positive = n  # all FA patients DEB positive
        androgen_rx = rng.randint(12, 22)
        return {
            "n": n, "mean_age": mean_age, "aplasia_count": aplasia_count,
            "aml_mds_count": aml_mds_count, "hsct_count": hsct_count,
            "anomaly_count": anomaly_count, "deb_positive": deb_positive,
            "androgen_rx": androgen_rx,
        }
    elif g == "DKC1":
        # DC X-linked: mucocutaneous triad + IPF
        triad_complete = rng.randint(30, 38)
        aplasia_count = rng.randint(20, 32)
        ipf_count = rng.randint(8, 14)
        oral_scc_count = rng.randint(4, 8)
        hsct_count = rng.randint(18, 26)
        danazol_response = rng.randint(24, 30)
        return {
            "n": n, "mean_age": mean_age, "triad_complete": triad_complete,
            "aplasia_count": aplasia_count, "ipf_count": ipf_count,
            "oral_scc_count": oral_scc_count, "hsct_count": hsct_count,
            "danazol_response": danazol_response,
        }
    elif g == "TERC":
        # TERC: aplastic anemia + IPF + anticipation
        aplasia_count = rng.randint(10, 16)
        ipf_count = rng.randint(8, 14)
        anticipation_families = rng.randint(15, 22)
        danazol_response = rng.randint(22, 30)
        hsct_count = rng.randint(10, 18)
        telomere_short = n  # all <1st percentile
        return {
            "n": n, "mean_age": mean_age, "aplasia_count": aplasia_count,
            "ipf_count": ipf_count, "anticipation_families": anticipation_families,
            "danazol_response": danazol_response, "hsct_count": hsct_count,
            "telomere_short": telomere_short,
        }
    elif g == "RPS19":
        # DBA: pure RCA + steroid response
        steroid_response = rng.randint(24, 32)
        transfusion_dependent = rng.randint(12, 20)
        anomaly_count = rng.randint(16, 22)
        hsct_count = rng.randint(8, 14)
        spontaneous_remission = rng.randint(6, 10)
        iron_overload = rng.randint(8, 14)
        return {
            "n": n, "mean_age": mean_age, "steroid_response": steroid_response,
            "transfusion_dependent": transfusion_dependent, "anomaly_count": anomaly_count,
            "hsct_count": hsct_count, "spontaneous_remission": spontaneous_remission,
            "iron_overload": iron_overload,
        }
    elif g == "SBDS":
        # SDS: EPI + neutropenia + AML
        epi_at_birth = rng.randint(38, 40)
        neutropenia_count = n
        aml_mds_count = rng.randint(8, 14)
        metaphyseal_dysplasia = rng.randint(28, 36)
        pert_on_rx = rng.randint(30, 38)
        i7q_del20q = rng.randint(10, 18)
        hsct_count = rng.randint(8, 14)
        return {
            "n": n, "mean_age": mean_age, "epi_at_birth": epi_at_birth,
            "neutropenia_count": neutropenia_count, "aml_mds_count": aml_mds_count,
            "metaphyseal_dysplasia": metaphyseal_dysplasia, "pert_on_rx": pert_on_rx,
            "i7q_del20q": i7q_del20q, "hsct_count": hsct_count,
        }
    else:  # MPL / CAMT
        # CAMT: severe thrombocytopenia → aplasia
        ich_count = rng.randint(8, 14)
        pancytopenia_evolved = rng.randint(22, 32)
        hsct_count = rng.randint(28, 36)
        tpo_markedly_elevated = n
        type1_null = rng.randint(24, 30)
        platelet_tx_dependent = n
        return {
            "n": n, "mean_age": mean_age, "ich_count": ich_count,
            "pancytopenia_evolved": pancytopenia_evolved, "hsct_count": hsct_count,
            "tpo_markedly_elevated": tpo_markedly_elevated, "type1_null": type1_null,
            "platelet_tx_dependent": platelet_tx_dependent,
        }


def _atlas_overview_stats() -> dict:
    """Aggregate cohort statistics across all 8 genes."""
    cohorts = {
        g["gene"]: _make_cohort(i, SEED_BASE + i)
        for i, g in enumerate(ATLAS_GENES)
    }
    total_n = sum(c["n"] for c in cohorts.values())
    hsct_total = sum(c.get("hsct_count", 0) for c in cohorts.values())

    # FA cohorts
    fa_aplasia = sum(cohorts[g].get("aplasia_count", 0) for g in ("FANCA", "FANCC", "FANCG"))
    fa_aml = sum(cohorts[g].get("aml_mds_count", 0) for g in ("FANCA", "FANCC", "FANCG"))

    return {
        "total_n": total_n,
        "genes": len(ATLAS_GENES),
        "hsct_total": hsct_total,
        "hsct_pct": round(100 * hsct_total / total_n, 1),
        "fa_aplasia": fa_aplasia,
        "fa_aml_mds": fa_aml,
        "dkc1_ipf": cohorts["DKC1"].get("ipf_count", 0),
        "terc_ipf": cohorts["TERC"].get("ipf_count", 0),
        "rps19_steroid_response": cohorts["RPS19"].get("steroid_response", 0),
        "mpl_ich": cohorts["MPL"].get("ich_count", 0),
        "sbds_epi": cohorts["SBDS"].get("epi_at_birth", 0),
    }


def generate_overview() -> dict:
    stats = _atlas_overview_stats()
    return {
        "atlas": "Hereditary-Bone-Marrow-Failure-Syndrome-Atlas",
        "subtitle": "Complete 8-Gene BMF Reference — FA / DC / DBA / SDS / CAMT",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": stats["total_n"],
        "seeds": f"{SEED_BASE}–{SEED_BASE + len(ATLAS_GENES) - 1}",
        "summary": {
            "total_n": stats["total_n"],
            "syndromes_covered": 5,
            "genes": stats["genes"],
            "hsct_required_n": stats["hsct_total"],
            "hsct_pct": stats["hsct_pct"],
            "fa_aplasia_n": stats["fa_aplasia"],
            "fa_aml_mds_n": stats["fa_aml_mds"],
            "dkc1_terc_ipf_total": stats["dkc1_ipf"] + stats["terc_ipf"],
            "rps19_steroid_response_n": stats["rps19_steroid_response"],
            "mpl_ich_n": stats["mpl_ich"],
            "sbds_epi_at_birth_n": stats["sbds_epi"],
        },
        "clinical_highlights": [
            "FA (FANCA/FANCC/FANCG): DEB/MMC chromosomal instability test PATHOGNOMONIC — RIC-HSCT mandatory; standard conditioning fatal",
            "DKC1 X-linked DC: mucocutaneous triad (nail dystrophy + oral leukoplakia + reticulate pigmentation) PATHOGNOMONIC; IPF dominant cause of death",
            "TERC AD: genetic anticipation over generations PATHOGNOMONIC — earlier/severer disease each generation; 5-10% 'idiopathic' AA = telomeropathy",
            "RPS19 DBA: elevated eADA PATHOGNOMONIC; triphalangeal thumb = DBA until proven; corticosteroids 60-80% response; NO steroids before age 1yr",
            "SBDS SDS: exocrine pancreatic insufficiency at birth PATHOGNOMONIC; i(7q) in SDS ≠ MDS (SDS-specific benign clone)",
            "MPL CAMT: amegakaryocytosis + markedly elevated TPO PATHOGNOMONIC; TPO-RAs (eltrombopag/romiplostim) INACTIVE — no MPL; HSCT only cure",
        ],
        "key_pharmacological_rules": [
            "FA-ALL: RIC conditioning MANDATORY for HSCT — standard myeloablative causes fatal toxicity (radiosensitivity + ICL-hypersensitivity)",
            "DKC1: danazol 70% response via TERT upregulation; HSCT does NOT cure IPF — IPF continues post-transplant",
            "TERC: danazol TERT upregulation effective; avoid alkylators (additional telomere stress); RIC HSCT",
            "RPS19-DBA: NO corticosteroids before age 1 year; iron chelation mandatory after 10-15 transfusions",
            "SBDS: G-CSF SHORT-COURSE only (chronic G-CSF may accelerate MDS in SDS); PERT until EPI confirmed to resolve",
            "MPL-CAMT: eltrombopag/romiplostim COMPLETELY INACTIVE (MPL absent); maintain platelets >50k in first year (ICH prevention)",
        ],
        "cascade_testing": "DEB/MMC siblings (FA donor eligibility); telomere flow-FISH family (DC/TERC); RP gene panel (DBA family); SBDS gene conversion panel; MPL W515X family testing",
    }


def generate_breakdown() -> dict:
    gene_data = {}
    for i, g in enumerate(ATLAS_GENES):
        cohort = _make_cohort(i, SEED_BASE + i)
        gene_data[g["gene"]] = cohort

    return {
        "atlas": "Hereditary-Bone-Marrow-Failure-Syndrome-Atlas",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "syndrome_groups": {
            "Fanconi_Anemia": {
                "genes": ["FANCA", "FANCC", "FANCG"],
                "diagnostic_test": "DEB/MMC chromosomal instability (pathognomonic); FANC gene panel",
                "hsct_conditioning": "RIC MANDATORY (fludarabine-based); avoid radiation",
                "cancer_surveillance": "Annual BM biopsy + cytogenetics; annual head/neck (post-HSCT); annual gynecology (post-HSCT)",
            },
            "Dyskeratosis_Congenita": {
                "genes": ["DKC1", "TERC"],
                "diagnostic_test": "Telomere flow-FISH <1st percentile; DKC1/TERC/TERT sequencing; mucocutaneous triad clinical",
                "treatment": "Danazol 70% aplasia response (TERT upregulation); HSCT-RIC; IPF: nintedanib/pirfenidone",
                "surveillance": "Annual PFTs; annual oral exam; annual CBC; annual BM biopsy",
            },
            "Diamond_Blackfan_Anemia": {
                "genes": ["RPS19"],
                "diagnostic_test": "Elevated eADA (>3 mU/g Hb); BM erythroid hypoplasia; RP gene panel",
                "treatment": "Prednisolone 2mg/kg (first-line ≥1yr); transfusions + iron chelation; HSCT refractory",
                "caveat": "No steroids <1yr; check eADA before genetics; triphalangeal thumb pathognomonic",
            },
            "Shwachman_Diamond_Syndrome": {
                "genes": ["SBDS"],
                "diagnostic_test": "Fecal elastase <100 mcg/g (EPI); SBDS gene conversion panel; skeletal survey",
                "treatment": "PERT (EPI); vitamins ADEK; G-CSF short-course; HSCT for BMF/AML-MDS",
                "caveat": "i(7q)/del20q in SDS ≠ MDS (SDS-specific benign clones); SDS-specialist BM interpretation required",
            },
            "Congenital_Amegakaryocytic_Thrombocytopenia": {
                "genes": ["MPL"],
                "diagnostic_test": "BM amegakaryocytosis; serum TPO markedly elevated; MPL sequencing (W515X)",
                "treatment": "Platelet transfusions (ICH prophylaxis >50k); HSCT only cure; no TPO-RAs (inactive)",
                "caveat": "TPO-RAs INACTIVE in CAMT (no receptor); measure TPO before IVIG/steroids (exclude CAMT from ITP)",
            },
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Bone-Marrow-Failure-Syndrome-Atlas",
        "definitions": [
            {
                "term": "Fanconi Anemia (FA)",
                "definition": (
                    "AR chromosomal instability syndrome caused by defective DNA interstrand crosslink (ICL) repair via the FA pathway. "
                    "Characterized by aplastic anemia (age 5-10yr), congenital anomalies (~75%), and cancer predisposition (AML/MDS + solid tumors). "
                    "FANCA (~60%), FANCC, FANCG most common complementation groups. "
                    "Diagnosis: DEB/MMC clastogenic stress test (excess chromosomal breaks/radials). "
                    "HSCT with RIC conditioning is curative for BMF; does NOT prevent solid tumor risk post-HSCT."
                ),
            },
            {
                "term": "DEB/MMC Chromosomal Instability Test",
                "definition": (
                    "Diepoxybutane (DEB) or Mitomycin C (MMC) clastogenic stress test. "
                    "FA cells show excess chromosomal breaks, radials, and exchanges vs normal controls. "
                    "Gold-standard diagnostic for Fanconi Anemia. "
                    "Perform on PHA-stimulated lymphocytes (whole blood) + fibroblasts to detect somatic mosaicism. "
                    "False-negative possible with high revertant (mosaic) cell fraction in blood alone."
                ),
            },
            {
                "term": "IVS4+4A>T (FANCC Ashkenazi Founder)",
                "definition": (
                    "c.456+4A>T splice-site mutation in FANCC; Ashkenazi Jewish founder mutation. "
                    "Carrier frequency ~1/89 Ashkenazi Jews. "
                    "Associated with severe FA-C phenotype: early BMF onset (median ~5yr), high AML risk. "
                    "Pathognomonic in Ashkenazi Jewish patients with aplastic anemia. "
                    "Screen for IVS4+4A>T FIRST in Ashkenazi patients before full panel."
                ),
            },
            {
                "term": "Dyskeratosis Congenita (DC) Mucocutaneous Triad",
                "definition": (
                    "Pathognomonic triad of classic DC: "
                    "(1) Nail dystrophy — longitudinal ridging → onycholysis → complete nail loss (earliest sign); "
                    "(2) Oral leukoplakia — white mucosal patches, malignant transformation risk; "
                    "(3) Reticulate skin pigmentation — lacy hyperpigmentation on face/neck/chest. "
                    "Present in ~90% classic XL-DC (DKC1) males. Onset typically 5-15yr. "
                    "All three together = PATHOGNOMONIC Dyskeratosis Congenita."
                ),
            },
            {
                "term": "Genetic Anticipation (TERC/Telomeropathy)",
                "definition": (
                    "TERC AD LOF telomeropathy shows anticipation: disease severity WORSENS across successive generations. "
                    "Generation 1 (grandparent): mild aplasia/BMF in 40s-50s. "
                    "Generation 2 (parent): moderate DC with aplasia in 20s-30s. "
                    "Generation 3 (proband): severe DC/Hoyeraal-Hreidarsson in childhood. "
                    "Mechanism: successive generations inherit shorter and shorter telomeres (no re-elongation in germline). "
                    "PATHOGNOMONIC pattern in a 3-generation family pedigree with worsening hematologic disease."
                ),
            },
            {
                "term": "Elevated eADA (Diamond-Blackfan Anemia)",
                "definition": (
                    "Erythrocyte adenosine deaminase (eADA) is elevated in ALL ribosomal protein gene DBA. "
                    "Normal: 0.7-2.0 mU/g Hb. DBA: typically >3.0 mU/g Hb. "
                    "PATHOGNOMONIC screening test for DBA before genetics are available. "
                    "Elevated eADA in a macrocytic anemia infant with reticulocytopenia = DBA until proven otherwise. "
                    "Triphalangeal thumb + macrocytic anemia in infant = additional PATHOGNOMONIC DBA feature."
                ),
            },
            {
                "term": "Exocrine Pancreatic Insufficiency (Shwachman-Diamond)",
                "definition": (
                    "PATHOGNOMONIC presenting feature of Shwachman-Diamond Syndrome at birth. "
                    "Lipase, amylase, elastase severely reduced → steatorrhea, failure to thrive, fat-soluble vitamin deficiency. "
                    "Diagnosis: fecal elastase <100 mcg/g (severely reduced). "
                    "EPI improves spontaneously in ~50% with age (pancreatic acinar recovery). "
                    "Treatment: PERT (pancreatic enzyme replacement therapy) until spontaneous improvement confirmed; vitamins ADEK."
                ),
            },
            {
                "term": "i(7q) in Shwachman-Diamond Syndrome",
                "definition": (
                    "Isochromosome 7q [i(7q)] is an SDS-SPECIFIC cytogenetic finding. "
                    "In non-SDS context: i(7q) = high-risk MDS marker. "
                    "In SDS: i(7q) may represent benign clonal evolution — NOT automatically equivalent to MDS. "
                    "SDS-specialist BM cytogenetic interpretation MANDATORY before MDS diagnosis in SDS patients. "
                    "del(20q) in SDS: similarly may be benign. Both require expert contextual interpretation."
                ),
            },
            {
                "term": "Congenital Amegakaryocytic Thrombocytopenia (CAMT) — MPL W515X",
                "definition": (
                    "AR LOF of thrombopoietin receptor (MPL) → amegakaryocytosis → severe neonatal thrombocytopenia. "
                    "W515X (c.1544G>A) most common mutation (Type I null — severe). "
                    "BM shows amegakaryocytosis (absent megakaryocytes) with normal other lineages initially. "
                    "Serum TPO MARKEDLY elevated (no MPL → no TPO consumption). "
                    "Type I: pancytopenia invariably by age 3-5yr. HSCT is the ONLY curative option."
                ),
            },
            {
                "term": "TPO-RAs Inactive in CAMT",
                "definition": (
                    "Thrombopoietin receptor agonists (eltrombopag, romiplostim, avatrombopag) are COMPLETELY INACTIVE in CAMT. "
                    "Mechanism: MPL gene is absent/non-functional → no TPO receptor → TPO-RAs have no target. "
                    "CRITICAL DISTINCTION from ITP where TPO-RAs are first/second-line. "
                    "Clinical pearl: if neonatal thrombocytopenia fails IVIG + steroids + romiplostim → measure TPO + BM biopsy → CAMT. "
                    "HSCT is the only treatment — no medical alternative exists."
                ),
            },
            {
                "term": "RIC Conditioning (FA-HSCT Mandatory)",
                "definition": (
                    "Reduced-intensity conditioning (RIC) is MANDATORY for HSCT in all Fanconi Anemia patients. "
                    "Standard myeloablative conditioning (high-dose busulfan, cyclophosphamide, TBI) causes FATAL toxicity in FA: "
                    "  — Radiosensitivity: TBI → catastrophic DNA damage in FA cells; "
                    "  — ICL hypersensitivity: alkylating agents (busulfan, cyclophosphamide) are crosslinkers → lethal FA-cell toxicity. "
                    "RIC regimen: fludarabine-based; low-dose cyclophosphamide (5-10mg/kg vs 200mg/kg standard). "
                    "AVOID radiation in FA for any indication (radiosensitivity ABSOLUTE)."
                ),
            },
            {
                "term": "Somatic Mosaicism in Fanconi Anemia",
                "definition": (
                    "~15-20% FA patients have somatic mosaicism: spontaneous reversion of one mutant FANCA/FANCC/FANCG allele in a hematopoietic stem cell. "
                    "Reverted cells outcompete FA cells → blood count may normalize. "
                    "DEB/MMC on blood: FALSE-NEGATIVE if revertant fraction is high. "
                    "Solution: DEB/MMC on BOTH lymphocytes AND fibroblasts. "
                    "Implication: normal CBC does NOT exclude FA; normal DEB on blood-only does not exclude FA."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
