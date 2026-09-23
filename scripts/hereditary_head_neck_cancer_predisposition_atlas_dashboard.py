#!/usr/bin/env python3
"""Hereditary-Head-and-Neck-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
FANCA   (FA complementation group A; 1455aa; 16q24.3; AR LOF;
         Fanconi Anaemia type A — MOST COMMON 65% of all FA;
         HNSCC 500-700x lifetime risk — SQUAMOUS CELL CARCINOMA ORAL/PHARYNX/LARYNX;
         ABSOLUTELY NO alkylating agents — cyclophosphamide, MMC, cisplatin LETHAL;
         ABSOLUTELY NO ionising radiation above diagnostic dose;
         seed SEED_BASE+0) .
FANCC   (FA complementation group C; 558aa; 9q22.32; AR LOF;
         Fanconi Anaemia type C — 9-15% of FA; Ashkenazi founder c.456+4A>T (IVS4);
         HNSCC 500-700x; genotype-phenotype: null allele severe / hypomorphic milder;
         seed SEED_BASE+1) .
FANCD2  (FA complementation group D2; 1471aa; 3p25.3; AR LOF;
         Fanconi Anaemia type D2 — ubiquitination sensor, BRCA2-independent checkpoint;
         HNSCC 500-700x; monoubiquitination (K561) by FANCI-FANCD2 complex essential;
         seed SEED_BASE+2) .
FANCG   (FA complementation group G / XRCC9; 622aa; 9p13.3; AR LOF;
         Fanconi Anaemia type G — 9% of FA; NO phenotypic stratification by allele;
         HNSCC 500-700x; FA core complex scaffold; NBS1 interaction unique to FANCG;
         seed SEED_BASE+3) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome — HNSCC 30-35% cumulative by age 50;
         AVOID RADIATION ABSOLUTELY — radiation-induced sarcoma/secondary malignancy;
         WB-MRI Toronto Protocol ANNUALLY MANDATORY from birth;
         seed SEED_BASE+4) .
ATM     (Ataxia Telangiectasia Mutated; 3056aa; 11q22.3;
         AR biallelic = Ataxia-Telangiectasia / AD heterozygous = 2-5x HNSCC;
         RADIATION HYPERSENSITIVITY — even heterozygotes; AVOID radical RT;
         oropharyngeal 3-4x; laryngeal 2x; IgA deficiency PATHOGNOMONIC in A-T;
         seed SEED_BASE+5) .
NBN     (Nibrin / NBS1; 754aa; 8q21.3; AR biallelic = Nijmegen Breakage Syndrome;
         Slavic founder c.657_661del5 — 90% of NBS; AD heterozygous = 2-3x HNSCC;
         RADIATION SENSITIVITY biallelic; MRN complex (MRE11-RAD50-NBN) essential;
         microcephaly + immunodeficiency PATHOGNOMONIC for NBS;
         seed SEED_BASE+6) .
CDKN2A  (Cyclin-dependent kinase inhibitor 2A / p16-INK4a; 156aa; 9p21.3; AD LOF;
         FAMMM Syndrome + HPV-negative HNSCC 5-10x; oral/oropharyngeal predominant;
         annual dermatology from 18yr MANDATORY; annual oral exam MANDATORY;
         pancreatic MRI/EUS from 40yr (pancreatic 20x risk);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3454-3461)
"""
import random

SEED_BASE = 3454

ATLAS_GENES = [
    {
        "gene": "FANCA",
        "protein": (
            "FANCA -- 16q24.3 Autosomal-Recessive-LOF -- 1455aa -- "
            "FA-Core-Complex-Scaffold-163kDa-MOST-COMMON-65pct-FA-"
            "HNSCC-500-700x-SCC-Oral-Pharynx-Larynx-"
            "ALKYLATING-ABSOLUTELY-CI-RADIATION-ABSOLUTELY-CI-"
            "Androgen-Therapy-Oxymetholone-Transient-BMF-Bridge-OMIM-607139"
        ),
        "locus": "16q24.3",
        "protein_size": (
            "1455 aa / 163 kDa / 16q24.3 FANCA head-neck cancer molecular context: "
            "STRUCTURE: "
            "  1455 aa / 163 kDa; N-terminal HEAT repeats scaffold FA core complex; "
            "  FANCG interaction surface (C-terminal); nuclear localisation signals (NLS) x2; "
            "  Somatic mutation hotspots: exon 11-17 deletion most common (Ashkenazi); "
            "  FA core complex: FANCA-FANCB-FANCC-FANCE-FANCF-FANCG-FANCL-FANCM; "
            "CANCER RISKS (HNSCC FOCUS): "
            "  HNSCC: 500-700x relative risk; lifetime risk >25% by age 40 in FA; "
            "  ORAL CAVITY: SCC floor of mouth / lateral tongue most common; "
            "  OROPHARYNX: tonsil/BOT; HPV co-infection amplifies risk further; "
            "  ONSET: median HNSCC diagnosis age 26yr vs sporadic 62yr; "
            "  OTHER CANCERS: AML (15-20% by 40yr); vulval SCC; oesophageal SCC; "
            "KEY MANAGEMENT (HNSCC): "
            "  Annual oral examination with white-light + VELscope fluorescence from age 10yr; "
            "  BIOPSY any oral mucosal lesion — high false-negative on clinical inspection; "
            "  ALKYLATING AGENTS ABSOLUTELY CI: cisplatin, carboplatin, cyclophosphamide, MMC; "
            "  RADIATION ABSOLUTELY CI: avoid >2Gy to head-neck; surgical excision preferred; "
            "  CETUXIMAB + surgery: platinum-free protocol for FA-HNSCC; "
            "  Docetaxel/5-FU: used but bone marrow tolerance limits dosing; "
            "  HSCT: curative for BMF but does NOT reduce HNSCC risk — increases it post-HSCT; "
            "  Annual FBC from birth: monitor aplastic anaemia trajectory"
        ),
        "syndrome": "Fanconi Anaemia type A (FA-A) — most common complementation group",
        "inheritance": "AR LOF (autosomal recessive loss-of-function)",
        "hnscc_risk": "500-700x relative risk; >25% cumulative lifetime (oral/pharyngeal/laryngeal SCC)",
        "pathognomonic": "DEB/MMC chromosomal fragility test PATHOGNOMONIC for FA (FANCA any group)",
        "key_avoid": "ALKYLATING AGENTS ABSOLUTELY CI (cisplatin/MMC/cyclophosphamide); RADIATION >2Gy ABSOLUTELY CI — use cetuximab + surgery protocol",
        "key_rule": "Annual oral exam with fluorescence from age 10yr MANDATORY; biopsy any white/red oral lesion; do NOT use cisplatin for HNSCC treatment",
        "surveillance": "Annual FBC + oral exam age 10yr+; gynaecological exam age 15yr+ (females); audiology (HSCT post-exposure); ophthalmology",
        "targeted_rx": "Cetuximab (EGFR inhibitor — platinum-free); Docetaxel/5-FU (reduced dose); Eltrombopag (BMF bridge); Luspatercept; HSCT (BMF curative); Oxymetholone (transient androgen bridge)",
    },
    {
        "gene": "FANCC",
        "protein": (
            "FANCC -- 9q22.32 Autosomal-Recessive-LOF -- 558aa -- "
            "FA-Core-Complex-FANCE-Chaperone-63kDa-9-15pct-FA-"
            "Ashkenazi-Founder-c456+4A>T-IVS4-70pct-Ashkenazi-FA-"
            "HNSCC-500-700x-Genotype-Phenotype-Null-Severe-Hypomorphic-Milder-"
            "ALKYLATING-ABSOLUTELY-CI-OMIM-227645"
        ),
        "locus": "9q22.32",
        "protein_size": (
            "558 aa / 63 kDa / 9q22.32 FANCC head-neck cancer molecular context: "
            "STRUCTURE: "
            "  558 aa / 63 kDa; WD40-like domain: FANCE chaperone interaction; "
            "  ER-localised fraction: GRP78 interaction (cytoprotection); "
            "  HSP70 interaction: anti-apoptotic function in addition to FA pathway; "
            "  FANCC lacks intrinsic enzymatic activity — scaffold/chaperone; "
            "KEY MUTATIONS: "
            "  c.456+4A>T (IVS4): Ashkenazi Jewish founder — 70% of Ashkenazi FA; mild BMF trajectory; "
            "  c.67delG (exon 1): null allele — severe phenotype, early BMF; "
            "  c.1399G>T (p.Asp467Tyr): Japanese founder; "
            "CANCER RISKS (HNSCC FOCUS): "
            "  HNSCC: 500-700x relative risk — identical class risk to FANCA; "
            "  GENOTYPE-PHENOTYPE: null allele (c.67delG) — median HNSCC age 22yr; "
            "                      hypomorphic (c.456+4A>T) — median HNSCC age 32yr; "
            "  HPV-positive HNSCC: FA cells cannot repair HPV-induced DSBs; "
            "KEY MANAGEMENT: "
            "  Identical to FANCA: annual oral fluorescence exam age 10yr+; "
            "  ALKYLATING AGENTS ABSOLUTELY CI; RADIATION ABSOLUTELY CI; "
            "  Cetuximab + surgery: FA-HNSCC standard protocol; "
            "  Founder testing: c.456+4A>T carrier screening in Ashkenazi families MANDATORY"
        ),
        "syndrome": "Fanconi Anaemia type C (FA-C) — Ashkenazi Jewish founder variant c.456+4A>T",
        "inheritance": "AR LOF (autosomal recessive loss-of-function)",
        "hnscc_risk": "500-700x relative risk; genotype-phenotype: null allele earliest onset (age 22yr median)",
        "pathognomonic": "DEB/MMC fragility test; FANCC protein absent on Western blot; c.456+4A>T genotype in Ashkenazi",
        "key_avoid": "ALKYLATING AGENTS ABSOLUTELY CI; CISPLATIN ABSOLUTELY CI; RADIATION >2Gy ABSOLUTELY CI",
        "key_rule": "Carrier screening c.456+4A>T MANDATORY in Ashkenazi families; annual oral exam from age 10yr; cetuximab-based treatment not platinum",
        "surveillance": "Annual FBC + oral fluorescence exam; gynaecological surveillance (females); audiometry post-HSCT",
        "targeted_rx": "Cetuximab; Eltrombopag (BMF bridge); Luspatercept; HSCT (curative for BMF, does NOT reduce HNSCC risk); Androgen therapy (oxymetholone — bridge)",
    },
    {
        "gene": "FANCD2",
        "protein": (
            "FANCD2 -- 3p25.3 Autosomal-Recessive-LOF -- 1471aa -- "
            "FA-Core-Complex-Downstream-Monoubiquitination-Sensor-K561-163kDa-"
            "BRCA2-INDEPENDENT-Checkpoint-HNSCC-500-700x-"
            "FANCI-FANCD2-Heterodimer-I-D2-Complex-Ubiquitin-Foci-"
            "ALKYLATING-ABSOLUTELY-CI-OMIM-227646"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "1471 aa / 163 kDa / 3p25.3 FANCD2 head-neck cancer molecular context: "
            "STRUCTURE: "
            "  1471 aa / 163 kDa; monoubiquitination site K561 (human) — critical activation; "
            "  FANCI-FANCD2 (I-D2) heterodimer: co-monoubiquitinated complex; "
            "  Chromatin-binding domain: sites at stalled replication forks; "
            "  BRCA2 interaction domain (C-terminal): independent of core complex; "
            "  Nuclear foci formation: only monoubiquitinated FANCD2 forms discrete foci; "
            "MOLECULAR UNIQUENESS: "
            "  FANCD2 is the key signalling node — all FA core complex pathways converge on D2; "
            "  FANCL (E3 ubiquitin ligase) + UBE2T (E2) monoubiquitinate K561; "
            "  Monoubiquitination is loss-of-function readout: cells from FA-D2 patients lack foci; "
            "  BRCA1 and BRCA2 act DOWNSTREAM of FANCD2 — FANCD2 is epistatic to BRCA pathway; "
            "CANCER RISKS (HNSCC FOCUS): "
            "  HNSCC: 500-700x relative risk; identical phenotypic class as FA-A/C/G; "
            "  Mixed complementation groups show same HNSCC spectrum; "
            "  Higher rate of biallelic null alleles (severe FA) in D2 group; "
            "KEY MANAGEMENT: "
            "  Western blot for FANCD2 monoubiquitination: diagnostic marker of pathway function; "
            "  Complementation group assignment: essential before HSCT donor matching; "
            "  ALKYLATING ABSOLUTELY CI; cetuximab protocol for HNSCC"
        ),
        "syndrome": "Fanconi Anaemia type D2 (FA-D2) — ubiquitination checkpoint sensor",
        "inheritance": "AR LOF (autosomal recessive loss-of-function)",
        "hnscc_risk": "500-700x relative risk; severe phenotype due to null allele predominance in FA-D2",
        "pathognomonic": "Absent FANCD2 monoubiquitination foci on IF staining; DEB fragility test positive",
        "key_avoid": "ALKYLATING AGENTS ABSOLUTELY CI; RADIATION ABSOLUTELY CI; do NOT use PARP inhibitors (worsen replication stress in FA cells)",
        "key_rule": "Western blot FANCD2 monoubiquitination essential for pathway diagnosis; HSCT donor must be FA-screened (complementation group testing); annual HNSCC surveillance from age 10yr",
        "surveillance": "Annual oral fluorescence exam; annual FBC; monthly self-examination oral cavity; gynaecological (females)",
        "targeted_rx": "Cetuximab (EGFR — platinum-free FA protocol); Eltrombopag; Luspatercept; HSCT; Danazol (androgen bridge)",
    },
    {
        "gene": "FANCG",
        "protein": (
            "FANCG -- 9p13.3 Autosomal-Recessive-LOF -- 622aa -- "
            "FA-Core-Complex-XRCC9-70kDa-9pct-FA-NBS1-Interaction-UNIQUE-"
            "NO-Missense-Mild-Phenotype-All-FANCG-LOF-Equally-Severe-"
            "HNSCC-500-700x-ALKYLATING-ABSOLUTELY-CI-OMIM-602956"
        ),
        "locus": "9p13.3",
        "protein_size": (
            "622 aa / 70 kDa / 9p13.3 FANCG head-neck cancer molecular context: "
            "STRUCTURE: "
            "  622 aa / 70 kDa; formerly XRCC9 (X-ray cross-complementation group 9); "
            "  7 TPR (tetratricopeptide repeat) motifs: scaffold interactions; "
            "  FANCA binding: C-terminal TPR7; unique among core complex members; "
            "  NBS1 interaction: ONLY FANCG physically contacts NBS1 — links FA and NBS pathways; "
            "GENOTYPE-PHENOTYPE UNIQUE FEATURE: "
            "  ALL FANCG LOF mutations produce equivalent severe phenotype — NO hypomorphic alleles; "
            "  Contrast to FANCC (mild hypomorphic) or FANCF (no genotype-phenotype); "
            "  This makes FANCG carrier testing equally important regardless of variant class; "
            "CANCER RISKS (HNSCC FOCUS): "
            "  HNSCC: 500-700x relative risk — identical to other complementation groups; "
            "  NBS1 connection: FANCG-deficient cells have impaired NBS1-dependent DSB processing; "
            "  HNSCC in FANCG predominantly: floor of mouth, lateral tongue, tonsil; "
            "KEY MANAGEMENT: "
            "  NBS pathway co-dysfunction: relevant for DDR profiling; "
            "  Annual oral exam + fluorescence from age 10yr MANDATORY; "
            "  ALKYLATING ABSOLUTELY CI; cetuximab HNSCC protocol"
        ),
        "syndrome": "Fanconi Anaemia type G (FA-G / XRCC9) — unique NBS1 pathway connection",
        "inheritance": "AR LOF (autosomal recessive loss-of-function)",
        "hnscc_risk": "500-700x relative risk; NO hypomorphic alleles — all LOF equally severe",
        "pathognomonic": "DEB fragility test; FANCG-specific complementation assay; absent FANCG protein (Western blot)",
        "key_avoid": "ALKYLATING AGENTS ABSOLUTELY CI (no exceptions for FANCG — unlike FANCC hypomorphic context); RADIATION ABSOLUTELY CI",
        "key_rule": "All FANCG LOF alleles have equivalent severe risk (no mild FANCG phenotype); annual oral exam from age 10yr mandatory; NBS1 co-testing considered in ambiguous DDR presentations",
        "surveillance": "Annual FBC + oral fluorescence exam; annual dermatology (skin SCC); gynaecological surveillance; audiometry post-HSCT",
        "targeted_rx": "Cetuximab; Eltrombopag; Luspatercept; HSCT; Oxymetholone/danazol",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "Tumour-Suppressor-43kDa-Guardian-of-Genome-"
            "LFS-HNSCC-30-35pct-Cumulative-Oral-Laryngeal-"
            "AVOID-RADIATION-ABSOLUTELY-Radiation-Induced-Sarcoma-Risk-"
            "WB-MRI-Toronto-Protocol-ANNUALLY-MANDATORY-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 head-neck cancer molecular context: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; N-terminal transactivation domain (aa 1-67): MDM2 binding; "
            "  Proline-rich region (aa 68-97): apoptosis regulation; "
            "  DNA-binding domain (aa 102-292): hotspot mutations (R175H, R248W, R273H, G245S); "
            "  Tetramerisation domain (aa 323-356): homo-tetramer (active form); "
            "  Regulatory domain (aa 363-393): post-translational modifications (Lys372-382); "
            "CANCER RISKS (HNSCC FOCUS): "
            "  HNSCC: 30-35% cumulative lifetime risk — oral cavity/larynx/hypopharynx predominant; "
            "  ONSET: LFS-HNSCC median age 28yr vs sporadic HNSCC median 62yr; "
            "  HPV-negative HNSCC: TP53 LOF commonest somatic driver (50-80% sporadic); "
            "  LARYNGEAL: 5-10x elevated in LFS families; "
            "  OROPHARYNGEAL: 2-3x (lower HPV-independent rate); "
            "  SECONDARY MALIGNANCY: radiation-induced sarcoma in radiation field — up to 30%; "
            "KEY MANAGEMENT: "
            "  WB-MRI Toronto Protocol: ANNUALLY from birth (LFS surveillance gold standard); "
            "  ANNUAL oral exam + nasopharyngoscopy from age 18yr; "
            "  AVOID RADIATION ABSOLUTELY for curative intent: surgery preferred for HNSCC; "
            "  If RT unavoidable (metastatic/palliative): informed consent re secondary sarcoma; "
            "  CETUXIMAB preferred over platinum/RT for LFS-HNSCC; "
            "  ANNUAL FDG-PET not recommended (false positives, radiation burden in LFS)"
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) — HNSCC is a core LFS tumour type",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "hnscc_risk": "30-35% cumulative lifetime; oral cavity + larynx + hypopharynx predominant",
        "pathognomonic": "TP53 germline variant confirmed + HNSCC diagnosis <40yr; radiation-induced sarcoma in treated LFS HNSCC",
        "key_avoid": "RADIATION ABSOLUTELY CI for curative HNSCC treatment — radiation-induced sarcoma 30%; use surgery + cetuximab protocol",
        "key_rule": "WB-MRI Toronto Protocol ANNUALLY from birth MANDATORY; avoid RT; annual nasopharyngoscopy + oral exam from age 18yr",
        "surveillance": "Annual WB-MRI (whole-body MRI); annual oral cavity exam + nasopharyngoscopy; annual breast MRI women 20-65yr; annual abdominal USS; annual dermatology",
        "targeted_rx": "Cetuximab (platinum-free HNSCC); Pembrolizumab (tumour-agnostic MSI-H/TMB-H); Nivolumab (anti-PD-1 HNSCC); APR-246/eprenetapopt (reactivates mutant p53 — investigational); PRIMA-1Met",
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AR-Biallelic=AT / AD-Heterozygous=2-5xHNSCC -- 3056aa -- "
            "PI3K-Like-Kinase-350kDa-DSB-Master-Sensor-"
            "AT-Biallelic-Cerebellar-Ataxia-Telangiectasia-IgA-Deficiency-PATHOGNOMONIC-"
            "RADIATION-HYPERSENSITIVITY-Even-Heterozygotes-"
            "Oropharyngeal-3-4x-Laryngeal-2x-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM head-neck cancer molecular context: "
            "STRUCTURE: "
            "  3056 aa / 350 kDa; FAT domain (aa 1981-2566): protein-protein interactions; "
            "  Kinase domain (aa 2712-2962): PI3K-like — phosphorylates H2AX (γH2AX), CHEK2, BRCA1; "
            "  FATC domain (aa 3024-3056): essential for kinase activity; "
            "  MRN complex recruits ATM to DSBs: MRE11-RAD50-NBN (NBS1) complex; "
            "BIALLELIC (A-T) vs HETEROZYGOUS PHENOTYPE: "
            "  BIALLELIC: Ataxia-Telangiectasia — cerebellar ataxia (progressive), telangiectasia (sclerae/skin), "
            "    IgA deficiency PATHOGNOMONIC, lymphoma/leukaemia 35-40% lifetime, HNSCC elevated; "
            "  HETEROZYGOUS (carrier): 2-5x HNSCC risk; 2-4x breast cancer; prostate 2-4x; "
            "    Radiation hypersensitivity clinically significant even in heterozygotes; "
            "CANCER RISKS (HNSCC FOCUS): "
            "  BIALLELIC: HNSCC ≥10x lifetime risk (on top of lymphoma dominant risk); "
            "  HETEROZYGOUS: oropharyngeal 3-4x RR; laryngeal 2x RR; oral cavity 2x RR; "
            "  HPV-negative HNSCC enriched in ATM carriers; "
            "KEY MANAGEMENT: "
            "  RADIATION SENSITIVITY: even heterozygotes show G2/M checkpoint impairment; "
            "  Reduce RT dose by 20-30% if unavoidable; preferably avoid radical RT; "
            "  Annual nasopharyngoscopy from age 30yr (heterozygous carriers); "
            "  Biallelic A-T: immunoglobulin replacement (IgA-IgG deficiency); physiotherapy"
        ),
        "syndrome": "Ataxia-Telangiectasia (biallelic) / ATM carrier syndrome (heterozygous) — HNSCC elevated",
        "inheritance": "AR biallelic = A-T; AD heterozygous = HNSCC/breast/prostate risk",
        "hnscc_risk": "Biallelic: ≥10x; Heterozygous: oropharyngeal 3-4x, laryngeal 2x, oral cavity 2x",
        "pathognomonic": "Cerebellar ataxia + telangiectasia + IgA deficiency = A-T biallelic PATHOGNOMONIC",
        "key_avoid": "RADIATION: reduce dose even in heterozygotes (G2/M impairment); no radical RT without informed consent; avoid MMC (FA-like sensitivity in biallelic)",
        "key_rule": "Radiation sensitivity applies to HETEROZYGOTES — reduce RT dose 20-30% if used; annual nasopharyngoscopy from age 30yr for carriers; IgA level mandatory before surgery (biallelic)",
        "surveillance": "Annual nasopharyngoscopy + oral exam; annual neurological assessment (biallelic); immunoglobulin levels annually; annual chest X-ray (lymphoma screening biallelic)",
        "targeted_rx": "Olaparib (ATM-mutant solid tumours — PROfound data); Pembrolizumab (HNSCC); Cetuximab; Immunoglobulin replacement (biallelic A-T); Ibuprofen (neuroprotective trial A-T)",
    },
    {
        "gene": "NBN",
        "protein": (
            "NBN -- 8q21.3 AR-Biallelic=NBS / AD-Heterozygous=2-3xHNSCC -- 754aa -- "
            "MRN-Complex-Nibrin-NBS1-85kDa-DSB-Sensing-ATM-Recruitment-"
            "Slavic-Founder-c657del5-90pct-NBS-Microcephaly-Immunodeficiency-PATHOGNOMONIC-"
            "Oropharyngeal-2-3x-Laryngeal-2x-RADIATION-SENSITIVITY-OMIM-602667"
        ),
        "locus": "8q21.3",
        "protein_size": (
            "754 aa / 85 kDa / 8q21.3 NBN head-neck cancer molecular context: "
            "STRUCTURE: "
            "  754 aa / 85 kDa; FHA domain (aa 1-108): BRCA1 binding, γH2AX interaction; "
            "  BRCT1 (aa 108-196) + BRCT2 (aa 217-323): phosphopeptide recognition; "
            "  MRE11 binding (aa 665-693): anchors to MRN complex; "
            "  ATM activation domain (aa 734-754): essential for ATM trans-autophosphorylation; "
            "BIALLELIC vs HETEROZYGOUS: "
            "  BIALLELIC (NBS): Nijmegen Breakage Syndrome — microcephaly (PATHOGNOMONIC, progressive), "
            "    combined immunodeficiency (IgA+IgG deficiency), radiation sensitivity; "
            "    lymphoma 50-60% by age 20yr; HNSCC elevated; "
            "  HETEROZYGOUS: HNSCC 2-3x RR; prostate 3-4x; "
            "  Slavic founder c.657_661del5 (657del5): 90% of NBS globally; "
            "KEY MUTATIONS: "
            "  c.657_661del5: Central European Slavic founder — Poland, Czech Rep, Slovakia; "
            "    truncating frameshift; complete loss of NBN protein; "
            "  c.698delA: second founder in some populations; "
            "CANCER RISKS (HNSCC FOCUS): "
            "  HETEROZYGOUS carrier: oropharyngeal 2-3x; laryngeal 2x; oral cavity 2x; "
            "  BIALLELIC: HNSCC elevated above carrier (lymphoma dominant first); "
            "KEY MANAGEMENT: "
            "  Radiation sensitivity in biallelic NBS: equivalent to A-T; reduce RT or avoid; "
            "  Carrier (heterozygous): reduce RT dose 20% if unavoidable; "
            "  Annual nasopharyngoscopy from age 30yr (heterozygous); "
            "  Immunoglobulin replacement (biallelic)"
        ),
        "syndrome": "Nijmegen Breakage Syndrome (biallelic) / NBN carrier syndrome — HNSCC elevated",
        "inheritance": "AR biallelic = NBS; AD heterozygous = HNSCC/prostate risk",
        "hnscc_risk": "Biallelic: significant (lymphoma dominant); Heterozygous: oropharyngeal 2-3x, laryngeal 2x",
        "pathognomonic": "Microcephaly + combined immunodeficiency + 657del5 genotype = NBS biallelic PATHOGNOMONIC",
        "key_avoid": "RADIATION in biallelic NBS: equivalent to A-T sensitivity (avoid or minimise); alkylating agents avoided in biallelic",
        "key_rule": "657del5 Slavic founder testing MANDATORY in Central/Eastern European patients; radiation sensitivity biallelic = A-T equivalent; annual nasopharyngoscopy from age 30yr carriers",
        "surveillance": "Annual nasopharyngoscopy + oral exam (carriers); lymphoma surveillance biallelic (annual chest CT/PET); immunoglobulin levels biallelic; neuropsychological (microcephaly)",
        "targeted_rx": "Olaparib (NBN-mutant tumours — emerging data); Pembrolizumab (HNSCC); Immunoglobulin replacement (biallelic); Cetuximab (HNSCC platinum-free)",
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa -- "
            "p16-INK4a-CDK4-6-Inhibitor-15kDa-FAMMM-Syndrome-"
            "HPV-Negative-HNSCC-5-10x-Oral-Oropharyngeal-Predominant-"
            "Annual-Dermatology-MANDATORY-Annual-Oral-Exam-MANDATORY-"
            "Pancreatic-MRI-EUS-40yr-20x-Risk-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / 15 kDa / 9p21.3 CDKN2A head-neck cancer molecular context: "
            "STRUCTURE: "
            "  156 aa / 15 kDa (p16-INK4a isoform); alternate reading frame p14-ARF also encoded; "
            "  4 ankyrin repeat domains: CDK4/CDK6 binding — competitive with cyclin D1; "
            "  Inhibits CDK4/CDK6: prevents Rb phosphorylation — G1/S checkpoint maintained; "
            "  p14-ARF: MDM2 inhibitor — stabilises p53 (different reading frame); "
            "  Key hotspot mutations: p.G101W (Australian/European founder); R24P (p16-CDK4 interface); "
            "  R24C/R24H on CDK4: GOF — bypasses p16 inhibition (CDK4 germline GOF predisposition); "
            "CANCER RISKS (HNSCC FOCUS): "
            "  HNSCC: 5-10x relative risk — HPV-negative oral cavity and oropharynx predominant; "
            "  ORAL CAVITY: floor of mouth, lateral tongue, buccal mucosa; "
            "  OROPHARYNGEAL: tonsil/BOT (HPV-negative subtype enriched); "
            "  ONSET: CDKN2A-HNSCC median age 42yr vs sporadic 62yr; "
            "  MELANOMA: 40-50x lifetime risk (dominant risk — annual skin surveillance); "
            "  PANCREATIC: 20x relative risk — EUS/MRI from age 40-50yr MANDATORY; "
            "  UVEAL MELANOMA: 2-3x elevated; "
            "KEY MANAGEMENT: "
            "  ANNUAL oral examination from age 18yr MANDATORY; "
            "  ANNUAL full-skin dermatological exam from age 18yr MANDATORY; "
            "  AVOID TANNING — absolute contraindication; "
            "  Pancreatic MRI or EUS annually from age 40-50yr; "
            "  Cetuximab appropriate for HNSCC (no specific CDKN2A-driven contraindication); "
            "  CDK4/6 inhibitors (palbociclib) — paradox in CDKN2A-null: may be LESS effective"
        ),
        "syndrome": "FAMMM Syndrome (Familial Atypical Multiple Mole Melanoma) + HPV-negative HNSCC predisposition",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "hnscc_risk": "HPV-negative HNSCC 5-10x; oral cavity + oropharynx; median onset 42yr vs sporadic 62yr",
        "pathognomonic": "Multiple atypical nevi + pancreatic cancer + HNSCC in family pedigree = FAMMM PATHOGNOMONIC",
        "key_avoid": "TANNING ABSOLUTE CI (UV exposure amplifies melanoma and possibly oral SCC risk); CDK4/6 inhibitors may be paradoxically less effective in CDKN2A-null HNSCC",
        "key_rule": "Annual oral exam from age 18yr MANDATORY; annual full-skin dermatology from age 18yr MANDATORY; pancreatic MRI/EUS from age 40yr; tanning is absolute contraindication",
        "surveillance": "Annual full-skin dermatological exam; annual oral examination with fluorescence; pancreatic MRI/EUS from age 40yr; annual ophthalmology (uveal melanoma); abdominal USS",
        "targeted_rx": "Pembrolizumab (KEYNOTE-048 HNSCC); Nivolumab (CheckMate-141 HNSCC); Cetuximab; Vemurafenib/dabrafenib (if co-occurring BRAF V600 somatic, not germline); CDK4/6i with caution",
    },
]

# ── Variants (5 per gene, seed-based) ────────────────────────────────────────
VARIANTS = {
    "FANCA": [
        ("c.3788_3790delTCT", "p.Phe1263del", "exon 38", "in-frame deletion, FA-core scaffold", 0.28),
        ("c.1115G>A", "p.Arg372His", "exon 11", "HEAT repeat, FANCG-binding surface", 0.22),
        ("c.2546delC", "p.Pro849fs", "exon 26", "frameshift, null allele — severe phenotype", 0.18),
        ("c.4275C>A", "p.Tyr1425*", "exon 43", "nonsense, C-terminal NLS loss", 0.16),
        ("c.190-1G>C", "p.?", "IVS1-1G>C", "splice acceptor — exon 2 skipping", 0.16),
    ],
    "FANCC": [
        ("c.456+4A>T", "p.?", "IVS4", "Ashkenazi founder splice — mild FA-C phenotype", 0.38),
        ("c.67delG", "p.Asp23fs", "exon 1", "null frameshift — severe phenotype", 0.25),
        ("c.1399G>T", "p.Asp467Tyr", "exon 14", "Japanese founder, structural disruption", 0.18),
        ("c.1177G>A", "p.Glu393Lys", "exon 12", "WD40-like region, FANCE binding impaired", 0.12),
        ("c.738T>G", "p.Tyr246*", "exon 8", "nonsense — protein absent", 0.07),
    ],
    "FANCD2": [
        ("c.1bC>T", "p.?", "exon 1", "start codon loss — no protein", 0.30),
        ("c.1681G>A", "p.Ala561Thr", "exon 18", "K561 ubiquitination domain disruption (adjacent)", 0.24),
        ("c.3350G>A", "p.Arg1117His", "exon 33", "BRCA2 interaction surface", 0.20),
        ("c.2444T>C", "p.Ile815Thr", "exon 25", "nuclear foci formation impaired", 0.15),
        ("c.1948+1G>T", "p.?", "IVS19", "splice donor — exon 19 skipping", 0.11),
    ],
    "FANCG": [
        ("c.1794G>A", "p.Trp598*", "exon 13", "FANCA-binding C-terminal — null", 0.32),
        ("c.307+1G>C", "p.?", "IVS3", "splice donor — exon 4 skipping", 0.25),
        ("c.1496G>A", "p.Arg499His", "exon 11", "TPR7 fold disruption, FANCA binding lost", 0.20),
        ("c.637C>T", "p.Arg213*", "exon 6", "nonsense — NBS1 interaction domain absent", 0.14),
        ("c.1835delA", "p.Lys612fs", "exon 13", "frameshift C-terminal — NLS loss", 0.09),
    ],
    "TP53": [
        ("c.817C>T", "p.Arg273Cys", "exon 8", "DNA contact hotspot — dominant negative", 0.30),
        ("c.524G>A", "p.Arg175His", "exon 5", "structural hotspot — partial GOF", 0.25),
        ("c.742C>T", "p.Arg248Trp", "exon 7", "DNA contact hotspot — dominant negative", 0.20),
        ("c.733G>A", "p.Gly245Ser", "exon 7", "structural hotspot — partial GOF", 0.15),
        ("c.1010G>T", "p.Arg337Leu", "exon 9", "tetramerisation domain — Brazilian founder region", 0.10),
    ],
    "ATM": [
        ("c.7271T>G", "p.Val2424Gly", "exon 50", "kinase domain PI3K-like — common LGO panel", 0.28),
        ("c.1066-6T>G", "p.?", "IVS10", "splice — partial exon 11 skip, kinase impaired", 0.22),
        ("c.2572T>C", "p.Ser858Pro", "exon 18", "MRN interaction surface", 0.20),
        ("c.8147T>C", "p.Phe2716Ser", "exon 55", "FATC domain — activation surface", 0.16),
        ("c.3161C>T", "p.Thr1054Met", "exon 22", "FAT domain — BRCA1 interaction", 0.14),
    ],
    "NBN": [
        ("c.657_661del5", "p.Lys219fs", "exon 6", "Slavic founder frameshift — 90% of NBS", 0.45),
        ("c.698delA", "p.Asp233fs", "exon 6", "second founder frameshift", 0.20),
        ("c.511A>G", "p.Ile171Val", "exon 5", "BRCT2 fold disruption — ATM recruitment impaired", 0.15),
        ("c.643C>T", "p.Arg215Trp", "exon 6", "BRCT1-BRCT2 linker — phosphopeptide recognition", 0.11),
        ("c.1900G>T", "p.Glu634*", "exon 12", "MRE11 binding domain — null C-terminal", 0.09),
    ],
    "CDKN2A": [
        ("c.301G>T", "p.Gly101Trp", "exon 2", "Australian/European founder — ankyrin repeat 3", 0.35),
        ("c.70G>C", "p.Ala24Pro", "exon 1", "CDK4/CDK6 binding surface disruption", 0.22),
        ("c.238C>T", "p.Pro80Leu", "exon 2", "ankyrin repeat 2 structural — common in FAMMM", 0.18),
        ("c.442G>A", "p.Val148Met", "exon 2", "ankyrin repeat 4 partial — mild reduced binding", 0.14),
        ("c.1_1insC", "p.?", "exon 1", "frameshift start — no p16-INK4a protein", 0.11),
    ],
}

# ── Phenotype generators ─────────────────────────────────────────────────────
HNSCC_SITES = ["oral cavity", "oropharynx", "larynx", "hypopharynx", "nasopharynx", "salivary gland"]
FA_SITES     = ["oral cavity", "oropharynx", "larynx", "oesophagus", "vulva"]
STAGE_OPTIONS = ["I", "II", "III", "IVA", "IVB"]
STATUS_OPTS   = ["active surveillance", "HNSCC treated", "AML treated", "BMF—HSCT candidate", "remission"]

def _patients_for_gene(gene_dict: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = gene_dict["gene"]
    syndrome = gene_dict["syndrome"]
    variants = VARIANTS[gene]

    gene_ages = {
        "FANCA": (15, 35), "FANCC": (18, 40), "FANCD2": (16, 36),
        "FANCG": (14, 38), "TP53": (20, 55), "ATM": (32, 60),
        "NBN": (28, 58), "CDKN2A": (30, 65),
    }
    age_lo, age_hi = gene_ages.get(gene, (20, 60))

    patients = []
    for i in range(n):
        age = rng.randint(age_lo, age_hi)
        sex = rng.choice(["M", "F"])
        var = rng.choices(variants, weights=[v[4] for v in variants])[0]
        allele2 = rng.choice(variants)[0] if gene in {"FANCA","FANCC","FANCD2","FANCG","NBN"} and rng.random() < 0.7 else "heterozygous"

        sites = FA_SITES if gene in {"FANCA","FANCC","FANCD2","FANCG"} else HNSCC_SITES
        has_hnscc = rng.random() < (0.45 if gene in {"FANCA","FANCC","FANCD2","FANCG"} else 0.30)
        hnscc_site = rng.choice(sites) if has_hnscc else None
        hnscc_stage = rng.choice(STAGE_OPTIONS) if has_hnscc else None

        bmf = rng.random() < 0.55 if gene in {"FANCA","FANCC","FANCD2","FANCG"} else False
        hsct_done = rng.random() < 0.35 if bmf else False

        melanoma = rng.random() < 0.35 if gene == "CDKN2A" else False
        aml = rng.random() < 0.18 if gene in {"FANCA","FANCC","FANCD2","FANCG"} else False

        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "syndrome": syndrome,
            "age": age,
            "sex": sex,
            "variant": var[0],
            "effect": var[1],
            "exon": var[2],
            "domain": var[3],
            "allele2": allele2,
            "has_hnscc": has_hnscc,
            "hnscc_site": hnscc_site,
            "hnscc_stage": hnscc_stage,
            "has_bmf": bmf,
            "hsct_done": hsct_done,
            "has_melanoma": melanoma,
            "has_aml": aml,
            "status": rng.choice(STATUS_OPTS),
        })
    return patients


def _all_patients() -> list:
    all_pts = []
    for gene_dict in ATLAS_GENES:
        idx = [g["gene"] for g in ATLAS_GENES].index(gene_dict["gene"])
        all_pts.extend(_patients_for_gene(gene_dict, SEED_BASE + idx))
    return all_pts


# ── API generators ────────────────────────────────────────────────────────────
def generate_overview() -> dict:
    pts = _all_patients()
    gene_counts = {}
    for p in pts:
        gene_counts.setdefault(p["gene"], {"gene": p["gene"], "n": 0, "hnscc": 0, "bmf": 0, "aml": 0})
        gene_counts[p["gene"]]["n"] += 1
        if p["has_hnscc"]: gene_counts[p["gene"]]["hnscc"] += 1
        if p["has_bmf"]:   gene_counts[p["gene"]]["bmf"] += 1
        if p["has_aml"]:   gene_counts[p["gene"]]["aml"] += 1

    hnscc_total = sum(1 for p in pts if p["has_hnscc"])
    bmf_total   = sum(1 for p in pts if p["has_bmf"])
    fa_genes    = {"FANCA","FANCC","FANCD2","FANCG"}
    fa_pts      = [p for p in pts if p["gene"] in fa_genes]
    alkylating_risk = len(fa_pts)  # all FA patients at risk of alkylating-agent CI

    return {
        "atlas": "Hereditary-Head-and-Neck-Cancer-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene FANCA-FANCC-FANCD2-FANCG-TP53-ATM-NBN-CDKN2A Reference",
        "total_patients": len(pts),
        "gene_cohorts": len(ATLAS_GENES),
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "hnscc_cases": hnscc_total,
        "hnscc_rate_pct": round(100 * hnscc_total / len(pts), 1),
        "bone_marrow_failure_cases": bmf_total,
        "alkylating_agent_ci_patients": alkylating_risk,
        "gene_summary": list(gene_counts.values()),
        "key_clinical_rules": [
            "FA genes (FANCA/C/D2/G): ALKYLATING AGENTS ABSOLUTELY CI — use cetuximab + surgery for HNSCC",
            "TP53 (LFS): AVOID RADIATION ABSOLUTELY — radiation-induced sarcoma 30%; WB-MRI Toronto Protocol ANNUALLY",
            "ATM heterozygotes: radiation hypersensitivity — reduce RT dose 20-30% if unavoidable",
            "NBN 657del5: Slavic founder — 90% of NBS cases; radiation sensitivity biallelic = A-T equivalent",
            "CDKN2A: annual oral exam + dermatology from age 18yr; pancreatic MRI/EUS from age 40yr; tanning ABSOLUTE CI",
            "FA-HNSCC: annual oral fluorescence exam from age 10yr; cetuximab NOT platinum for treatment",
        ],
        "fanconi_anemia_summary": {
            "complementation_groups": ["FA-A (FANCA, 65%)", "FA-C (FANCC, 9-15%)", "FA-D2 (FANCD2)", "FA-G (FANCG, 9%)"],
            "deb_mmc_test": "DEB/MMC chromosomal fragility — universal diagnostic test for ALL FA groups",
            "hnscc_risk_fold": "500-700x relative risk for oral/pharyngeal/laryngeal SCC",
            "treatment_rule": "NO alkylating agents; NO radical RT; cetuximab + surgery = FA-HNSCC standard protocol",
        },
    }


def generate_breakdown() -> dict:
    pts = _all_patients()
    # Per-gene breakdown
    per_gene = {}
    for g in ATLAS_GENES:
        gene = g["gene"]
        gpts = [p for p in pts if p["gene"] == gene]
        hnscc_pts = [p for p in gpts if p["has_hnscc"]]
        site_dist = {}
        for p in hnscc_pts:
            if p["hnscc_site"]:
                site_dist[p["hnscc_site"]] = site_dist.get(p["hnscc_site"], 0) + 1
        stage_dist = {}
        for p in hnscc_pts:
            if p["hnscc_stage"]:
                stage_dist[p["hnscc_stage"]] = stage_dist.get(p["hnscc_stage"], 0) + 1
        per_gene[gene] = {
            "gene": gene,
            "syndrome": g["syndrome"],
            "inheritance": g["inheritance"],
            "locus": g["locus"],
            "n": len(gpts),
            "hnscc_n": len(hnscc_pts),
            "hnscc_pct": round(100 * len(hnscc_pts) / len(gpts), 1) if gpts else 0,
            "hnscc_risk": g["hnscc_risk"],
            "hnscc_site_distribution": site_dist,
            "hnscc_stage_distribution": stage_dist,
            "key_avoid": g["key_avoid"],
            "key_rule": g["key_rule"],
            "targeted_rx": g["targeted_rx"],
            "surveillance": g["surveillance"],
            "top_variants": [
                {"variant": v[0], "effect": v[1], "exon": v[2], "domain": v[3], "freq": round(v[4], 2)}
                for v in VARIANTS[gene]
            ],
        }
    # Site distribution across all genes
    all_sites = {}
    for p in pts:
        if p["hnscc_site"]:
            all_sites[p["hnscc_site"]] = all_sites.get(p["hnscc_site"], 0) + 1
    return {
        "per_gene": list(per_gene.values()),
        "hnscc_site_distribution": all_sites,
        "fa_specific": {
            "alkylating_agent_ci_rule": "ABSOLUTELY CONTRAINDICATED: cisplatin, carboplatin, cyclophosphamide, MMC, oxaliplatin",
            "radiation_ci_rule": "ABSOLUTELY CONTRAINDICATED >2Gy: surgical excision preferred; cetuximab available for EGFR-positive HNSCC",
            "preferred_protocol": "Cetuximab (EGFR inhibitor) + surgery; docetaxel/5-FU reduced dose if systemic required",
            "hsct_note": "HSCT cures BMF but does NOT reduce HNSCC risk — HNSCC risk may increase post-HSCT (immunosuppression)",
        },
        "ddr_pathway_summary": {
            "FA_pathway": "FANCA→FANCC→FANCD2→FANCG: core complex → FANCD2/FANCI monoubiquitination → DSB repair",
            "ATM_pathway": "MRN (MRE11-RAD50-NBN) senses DSB → recruits ATM → phosphorylates H2AX/CHEK2/BRCA1",
            "p53_pathway": "ATM → CHK2 → TP53 stabilisation (MDM2 release) → G1 arrest / apoptosis",
            "CDKN2A_pathway": "p16-INK4a inhibits CDK4/CDK6 → Rb dephosphorylated → E2F suppressed → G1 checkpoint",
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Head-and-Neck-Cancer-Predisposition-Atlas",
        "genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein_function": g["protein"],
                "protein_size": g["protein_size"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "hnscc_risk": g["hnscc_risk"],
                "pathognomonic_features": g["pathognomonic"],
                "absolutely_avoid": g["key_avoid"],
                "mandatory_rule": g["key_rule"],
                "surveillance_protocol": g["surveillance"],
                "targeted_therapies": g["targeted_rx"],
                "variants": [
                    {"variant": v[0], "protein_effect": v[1], "location": v[2], "domain_impact": v[3]}
                    for v in VARIANTS[g["gene"]]
                ],
            }
            for g in ATLAS_GENES
        ],
        "key_clinical_concepts": {
            "fanconi_anemia_hnscc": (
                "FA cells cannot repair interstrand crosslinks (ICLs) — basis of DEB/MMC test AND alkylating CI. "
                "HNSCC risk 500-700x because oral/pharyngeal mucosa faces continuous replication stress. "
                "Annual oral fluorescence exam from age 10yr is the cornerstone of FA surveillance."
            ),
            "tp53_radiation_rule": (
                "LFS: germline TP53 LOF means radiotherapy induces new tumours (G1 checkpoint absent). "
                "Radiation-induced sarcoma 30% in treated LFS cases. "
                "ABSOLUTELY avoid radical RT; use surgery + cetuximab/immunotherapy for HNSCC."
            ),
            "atm_nbm_radiation_sensitivity": (
                "Both ATM and NBN encode DSB sensing/signalling proteins. "
                "Biallelic loss = A-T/NBS syndrome with extreme radiation sensitivity. "
                "Even heterozygous carriers have measurable G2/M checkpoint impairment — reduce RT dose 20-30%."
            ),
            "cdkn2a_dual_risk": (
                "CDKN2A encodes both p16-INK4a (CDK4/6 inhibitor) and p14-ARF (MDM2 inhibitor). "
                "FAMMM syndrome: melanoma dominant risk (40-50x). "
                "HNSCC: 5-10x HPV-negative oral/oropharyngeal. Pancreatic: 20x. "
                "CDK4/6 inhibitors may paradoxically fail in CDKN2A-null HNSCC (loss of target)."
            ),
            "deb_mmc_fragility_test": (
                "Diepoxybutane (DEB) or mitomycin C (MMC) chromosomal fragility test: "
                "gold standard for FA diagnosis across ALL complementation groups (FANCA/C/D2/G). "
                "Increased chromosomal breaks = positive = FA. "
                "Must be performed before complementation group assignment by sequencing."
            ),
        },
        "abbreviations": {
            "HNSCC": "Head and Neck Squamous Cell Carcinoma",
            "FA":    "Fanconi Anaemia",
            "A-T":   "Ataxia-Telangiectasia",
            "NBS":   "Nijmegen Breakage Syndrome",
            "FAMMM": "Familial Atypical Multiple Mole Melanoma",
            "LFS":   "Li-Fraumeni Syndrome",
            "ICL":   "Interstrand Crosslink",
            "DSB":   "Double-Strand Break",
            "DEB":   "Diepoxybutane (fragility test mutagen)",
            "MMC":   "Mitomycin C (fragility test mutagen)",
            "BMF":   "Bone Marrow Failure",
            "HSCT":  "Haematopoietic Stem Cell Transplantation",
            "WB-MRI": "Whole-Body MRI (Toronto Protocol for LFS surveillance)",
            "CI":    "Contraindicated",
            "ABSOLUTELY CI": "Absolute contraindication — do NOT use under any circumstances",
        },
    }


if __name__ == "__main__":
    import json
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"HNSCC cases: {ov['hnscc_cases']} ({ov['hnscc_rate_pct']}%)")
    print(f"Bone marrow failure cases: {ov['bone_marrow_failure_cases']}")
    print(f"Seeds: {ov['seeds']}")
    print("\nGene summary:")
    for gs in ov["gene_summary"]:
        print(f"  {gs['gene']:10s} n={gs['n']} HNSCC={gs['hnscc']} BMF={gs['bmf']} AML={gs['aml']}")
