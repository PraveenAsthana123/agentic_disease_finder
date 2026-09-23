#!/usr/bin/env python3
"""Hereditary-Neuroblastoma-Predisposition-Atlas -- Complete 8-Gene Reference
ALK     (Anaplastic Lymphoma Kinase; 1620aa; 2p23.2; AD GOF;
         Hereditary Neuroblastoma NBLST1 -- most common familial NB gene;
         R1275Q ~40%, F1174L ~25% aggressive, F1245C, D1091N hotspots;
         75-80% familial NB cases; Crizotinib FDA 2022 pediatric NB;
         ALK amplification SOMATIC distinct from germline GOF;
         seed SEED_BASE+0) .
PHOX2B  (Paired mesoderm homeobox protein 2B; 314aa; 4p13; AD;
         CCHS -- Ondine's curse -- congenital central hypoventilation;
         polyalanine expansion 20/27 (most common) NB 5-10% CCHS;
         NPARMs (non-polyalanine repeat mutations) = NB 50% PATHOGNOMONIC;
         life-long ventilatory support CCHS; tumor-only mutations sporadic NB;
         seed SEED_BASE+1) .
BARD1   (BRCA1-associated RING domain protein 1; 777aa; 2q35; AD LOF;
         NB susceptibility 2 (NBLST2/NBLS);
         BRCA1-BARD1 RING heterodimer E3 ubiquitin ligase complex;
         C557S and truncating variants -- 5-10% sporadic high-risk NB;
         1p36 LOH in 70% NB (BARD1 at 2q35 distinct from 1p36 locus);
         seed SEED_BASE+2) .
KIF1B   (Kinesin family member 1B; 1816aa; 1p36.22; AD LOF;
         NB susceptibility 1 -- 1p36 deletion region;
         kinesin motor protein; 1p36 LOH 70% NB;
         pheochromocytoma risk also (type 1 neuroendocrine);
         multiple endocrine neoplasia-like neurocristopathy phenotype;
         seed SEED_BASE+3) .
NF1     (Neurofibromin 1; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis type 1 + neuroblastoma 2-3x;
         cafe-au-lait macules ≥6 ≥15mm PATHOGNOMONIC;
         plexiform NF -> MPNST 8-13% PATHOGNOMONIC transformation;
         Selumetinib FDA 2020 symptomatic inoperable plexiform NF;
         AVOID RADIATION (secondary MPNST risk);
         seed SEED_BASE+4) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome;
         NB in LFS 5-10% -- anaplastic/relapse NB;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annual;
         ALT pathway at relapse TP53 + ATRX co-mutation;
         seed SEED_BASE+5) .
DICER1  (DICER1 ribonuclease III; 1922aa; 14q32.13; AD LOF;
         DICER1 syndrome;
         neuroblastoma 2-4x elevated risk DICER1;
         pleuropulmonary blastoma PATHOGNOMONIC (sibling screen CT chest <8yr);
         cervical ERMS PATHOGNOMONIC; AVOID radiation in children;
         seed SEED_BASE+6) .
BRCA2   (Breast cancer gene 2; 3418aa; 13q12.3; Biallelic AR LOF FA-D1 / AD LOF HBOC;
         FA complementation group D1 -- NB/solid tumour risk;
         bilateral Wilms PATHOGNOMONIC FA-D1;
         DEB/MMC chromosomal fragility PATHOGNOMONIC FA;
         SIBLING DONOR EXCLUSION MANDATORY; HRD sensitivity to cisplatin;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3270-3277)
"""
import random

SEED_BASE = 3270

ATLAS_GENES = [
    {
        "gene": "ALK",
        "protein": (
            "ALK -- 2p23.2 Autosomal-Dominant-GOF -- 1620aa -- "
            "Anaplastic-Lymphoma-Kinase-177kDa-RTK-NBLST1-"
            "R1275Q-F1174L-Hotspots-Crizotinib-FDA2022-OMIM-105590"
        ),
        "locus": "2p23.2",
        "protein_size": (
            "1620 aa / 177 kDa / 2p23.2 ALK encodes anaplastic lymphoma kinase (receptor tyrosine kinase): "
            "STRUCTURE: "
            "  1620 aa / 177 kDa; receptor tyrosine kinase (RTK) class; "
            "  Extracellular domain (aa 1-1037): MAM domains x2, LDLa domain; ligands: FAM150A/B; "
            "  Transmembrane domain (aa 1038-1059); "
            "  Intracellular kinase domain (aa 1116-1383): activation loop; "
            "  Normal expression: neural crest, developing nervous system; "
            "  Germline GOF mutations: constitutive kinase activation -> uncontrolled cell proliferation; "
            "  ALK amplification (somatic, DISTINCT from germline GOF): MYCN co-amplification in subset; "
            "HEREDITARY NEUROBLASTOMA (NBLST1): "
            "  OMIM 256700; ALK germline GOF most common familial NB gene (75-80% familial cases); "
            "  Autosomal dominant; incomplete penetrance (~57% by age 80); "
            "  R1275Q: ~40% of germline ALK mutations; kinase domain activation loop; intermediate aggressiveness; "
            "  F1174L: ~25% of germline ALK mutations; MOST AGGRESSIVE -- MYCN co-amplification frequent; "
            "    F1174L + MYCN amplification: nearly universal high-risk INRG M disease; "
            "    F1174L survival inferior: 5yr OS ~30% untreated; crizotinib essential; "
            "  F1245C, D1091N, I1250T, Y1278S: additional germline hotspots; "
            "  T1151M, G1128A: weaker activating variants; variable penetrance; "
            "SOMATIC ALK (DISTINCT FROM GERMLINE): "
            "  Somatic ALK mutation 6-8% sporadic NB (enriched in high-risk); "
            "  Somatic ALK amplification: distinct event, 3-4% NB; "
            "  Both germline + somatic ALK = IDENTICAL codons (R1275Q/F1174L most common both); "
            "  ALK FISH: amplification (somatic) vs copy-neutral GOF (germline/somatic mutation); "
            "  Panel-based sequencing MANDATORY: germline vs somatic ALK interpretation critical; "
            "CRIZOTINIB (ALK INHIBITOR -- FDA 2022 PEDIATRIC): "
            "  Crizotinib FDA approved 2022 pediatric ALK-positive neuroblastoma (relapsed/refractory); "
            "  COG ADVL0912: objective response 26% relapsed/refractory NB; "
            "  Lorlatinib (3rd-gen ALK/ROS1 inhibitor): F1174L resistant to crizotinib -- lorlatinib active; "
            "  Ceritinib, alectinib: 2nd-gen ALK inhibitors; less pediatric data; "
            "  First-line crizotinib in germline ALK high-risk NB: COG ANBL1232/1522 phase III ongoing; "
            "SURVEILLANCE (ALK GERMLINE): "
            "  Abdominal US + MIBG scan from birth/diagnosis until age 10yr (no consensus; SIOPEN recommends); "
            "  Catecholamines (urine HVA/VMA) 3-monthly; "
            "  MRI chest/abdomen/pelvis preferred over CT (lifetime radiation in young patients); "
            "  Cascade testing first-degree relatives (50% risk): germline ALK panel; "
        ),
        "inheritance": "Autosomal dominant GOF; penetrance ~57% cumulative by age 80; de novo mutations account for 10% familial NB; most high-risk alleles (F1174L) show near-complete penetrance",
        "cancer_risk": "Neuroblastoma: 75-80% familial NB cases ALK germline GOF; F1174L highest risk (nearly 100% penetrance + high-risk INRG M); R1275Q intermediate penetrance; additional ALK GOF alleles with variable penetrance",
        "pathognomonic": "Familial NB (2+ first-degree relatives, bilateral adrenal or multifocal NB) PATHOGNOMONIC hereditary NB -- ALK germline GOF #1 cause; F1174L + MYCN amplification somatic PATHOGNOMONIC poorest prognosis ALK-driven NB",
        "surveillance_key": "Urine HVA/VMA q3M; abdominal US q3M from birth until age 10yr; MIBG scan at diagnosis + restaging; germline ALK panel first-degree relatives; avoid CT for routine surveillance; crizotinib at relapsed/refractory stage; lorlatinib for F1174L crizotinib-resistant",
        "key_distinctions": [
            "ALK-GERMLINE-GOF-75-80PCT-FAMILIAL-NB-NBLST1",
            "F1174L-MOST-AGGRESSIVE-MYCN-CO-AMP-PATHOGNOMONIC",
            "R1275Q-MOST-COMMON-40PCT-GERMLINE",
            "CRIZOTINIB-FDA2022-PEDIATRIC-NB-ALK",
            "LORLATINIB-F1174L-CRIZOTINIB-RESISTANT",
            "ALK-AMPLIFICATION-SOMATIC-DISTINCT-FROM-GERMLINE-GOF",
        ],
    },
    {
        "gene": "PHOX2B",
        "protein": (
            "PHOX2B -- 4p13 Autosomal-Dominant -- 314aa -- "
            "Paired-Mesoderm-Homeobox-2B-34kDa-NC-TF-CCHS-Ondine-"
            "NPARMs-NB-50pct-PATHOGNOMONIC-Polyalanine-Repeats-OMIM-603851"
        ),
        "locus": "4p13",
        "protein_size": (
            "314 aa / 34 kDa / 4p13 PHOX2B encodes paired mesoderm homeobox protein 2B (neural crest TF): "
            "STRUCTURE: "
            "  314 aa / 34 kDa; homeodomain transcription factor; "
            "  N-terminal homeodomain (aa 101-163): DNA binding; "
            "  C-terminal polyalanine tract (aa 241-260): 20 alanine repeats normally (20/20); "
            "  Germline mutations expand polyalanine tract: 20/25 to 20/33 repeats; "
            "  PHOX2B controls autonomic nervous system development: brainstem respiratory centres; "
            "  PHOX2B regulates RET, HAND2, DBH transcription in neural crest derivatives; "
            "CONGENITAL CENTRAL HYPOVENTILATION SYNDROME (CCHS / ONDINE'S CURSE): "
            "  OMIM 209880; loss of autonomic control of breathing during sleep; "
            "  Polyalanine expansion mutations (PARMs): 20/25 minimal (CCHS only, sleep); "
            "    20/26-20/27: moderate CCHS + NB risk 5-10%; "
            "    20/28-20/33: severe CCHS + cardiac dysrhythmia + NB risk 15-20%; "
            "    All >24/27 repeat alleles: Hirschsprung disease risk 20%; "
            "  Non-polyalanine repeat mutations (NPARMs): "
            "    FRAMESHIFT: 50% NB risk PATHOGNOMONIC -- highest NB risk; "
            "    MISSENSE (R100L, E193K, etc): also elevated NB risk; "
            "    NPARMs = most severe CCHS phenotype; "
            "  CCHS management: life-long ventilatory support (NIV/tracheostomy/diaphragm pacemaker); "
            "  CCHS surveillance: annual 24-hr Holter (cardiac dysrhythmia); "
            "NEUROBLASTOMA IN CCHS (PHOX2B NB): "
            "  NB occurs in 5-10% CCHS overall (higher in NPARMs 50%); "
            "  Adrenal NB + thoracic/cervical ganglionic NB (neural crest origin); "
            "  MIBG scan from CCHS diagnosis + annually in NPARM patients; "
            "  Urine catecholamines (HVA/VMA) q3M all CCHS; "
            "  PHOX2B tumor-only missense in sporadic NB: somatic PHOX2B GOF (distinct mechanism); "
            "    Tumor-only PHOX2B not CCHS phenotype; do NOT diagnose CCHS from tumor tissue; "
            "TREATMENT (PHOX2B NB): "
            "  Standard NB risk stratification (INRG) by stage, age, histology, MYCN status; "
            "  CCHS + NB: anesthesia HIGH RISK (respiratory depression -- no sedation unsupervised); "
            "  Anti-GD2 therapy (dinutuximab, naxitamab) for high-risk/relapsed NB; "
        ),
        "inheritance": "Autosomal dominant; polyalanine expansions de novo in 97% (rarely inherited from affected mosaic parent); NPARMs de novo in ~90%; autosomal dominant transmission from affected parent documented in polyalanine PARM",
        "cancer_risk": "NB: 5-10% CCHS overall; NPARMs 50% NB risk PATHOGNOMONIC; 20/27+ polyalanine: 15-20% NB risk; adrenal + thoracic/cervical ganglionic NB; Hirschsprung 20% (20/28+); cardiac dysrhythmia 25% (20/28+)",
        "pathognomonic": "CCHS (congenital central hypoventilation) PATHOGNOMONIC PHOX2B; NPARMs + CCHS = NB 50% risk PATHOGNOMONIC; tumor-only PHOX2B somatic mutation in sporadic NB (NOT CCHS -- tumor tissue distinct from germline)",
        "surveillance_key": "Life-long NIV/tracheostomy CCHS; annual 24-hr Holter; urine HVA/VMA q3M; MIBG annually in NPARM CCHS; avoid sedatives/opioids without ventilatory support (respiratory centre failure); anesthesia HIGH RISK; Hirschsprung workup 20/28+",
        "key_distinctions": [
            "CCHS-ONDINE-PATHOGNOMONIC-PHOX2B",
            "NPARMs-NB-50PCT-HIGHEST-RISK-PATHOGNOMONIC",
            "POLYALANINE-20-27-NB-5-10PCT",
            "TUMOR-ONLY-PHOX2B-NOT-CCHS-DIAGNOSIS",
            "ANESTHESIA-HIGH-RISK-RESPIRATORY-CENTRE-FAILURE",
            "LIFE-LONG-VENTILATORY-SUPPORT-CCHS",
        ],
    },
    {
        "gene": "BARD1",
        "protein": (
            "BARD1 -- 2q35 Autosomal-Dominant-LOF -- 777aa -- "
            "BRCA1-Associated-RING-Domain-1-86kDa-E3-Ub-Ligase-"
            "NB-Susceptibility-2-NBLST2-C557S-Variant-OMIM-601593"
        ),
        "locus": "2q35",
        "protein_size": (
            "777 aa / 86 kDa / 2q35 BARD1 encodes BRCA1-associated RING domain protein 1 (E3 ubiquitin ligase component): "
            "STRUCTURE: "
            "  777 aa / 86 kDa; "
            "  RING domain (aa 40-100): obligate heterodimer with BRCA1 RING -> E3 ubiquitin ligase; "
            "  BRCA1-BARD1 RING complex: ubiquitinates H2A at DNA damage; "
            "  Ankyrin repeats (aa 426-540): protein-protein interactions; "
            "  BRCT domain (aa 575-777): phosphopeptide binding; DNA damage response; "
            "  BARD1 stabilises BRCA1 protein (prevents proteasomal degradation); "
            "  BARD1 independent of BRCA1: BARD1 nuclear localisation independent function; "
            "NEUROBLASTOMA SUSCEPTIBILITY 2 (NBLST2): "
            "  NB susceptibility locus at 2q35 (BARD1 gene); "
            "  Cys557Ser (C557S): most studied population variant -- moderately elevated NB risk; "
            "  Multiple loss-of-function variants: truncating, splicing -- 5-10% high-risk sporadic NB; "
            "  BARD1 expression reduced in NB: loss of DNA damage checkpoint -> aneuploidy; "
            "  1p36 LOH: 70% NB (KIF1B locus distinct -- 1p36 deletion includes multiple suppressors); "
            "  GWAS validation: BARD1 2q35 SNPs rs6435862, rs3768716 -- NB GWAS top hits; "
            "    OR ~1.5-2.0 for common variants; rare LOF variants = higher penetrance; "
            "BRCA1-BARD1 COMPLEX IN NB: "
            "  BRCA1-BARD1 complex: HR repair and mitotic spindle checkpoint; "
            "  BARD1 LOF -> BRCA1 instability -> HRD -> HRD sensitivity (cisplatin/PARP inhibitors); "
            "  Olaparib activity predicted in BARD1-LOF NB (HRD context); "
            "  Anti-GD2 (dinutuximab/naxitamab) standard for high-risk NB regardless of BARD1; "
            "SURVEILLANCE (BARD1): "
            "  No established surveillance protocol for BARD1 germline carriers; "
            "  Clinical guidance: urine HVA/VMA q3M first 5yr in carrier children; "
            "  Cascade testing first-degree relatives (BARD1 family); "
            "  BARD1 C557S: common variant -- population frequency ~15%; individual risk moderate; "
        ),
        "inheritance": "Autosomal dominant LOF; C557S common variant (population frequency ~15%); rare truncating/splicing LOF variants with higher penetrance; BARD1 biallelic LOF not well described (likely embryonic lethal)",
        "cancer_risk": "NB: 5-10% high-risk sporadic NB BARD1 LOF; C557S OR 1.5-2.0x NB risk (common variant); rare LOF variants higher penetrance; BARD1 expression loss correlated with high-risk INRG disease",
        "pathognomonic": "BARD1 2q35 GWAS top hit for NB susceptibility; BARD1 LOF + HRD phenotype = cisplatin sensitivity; rare BARD1 truncating variants in familial NB cluster",
        "surveillance_key": "Urine HVA/VMA q3M until age 5yr for BARD1 carriers; cascade family testing; no consensus CT/MRI screening protocol for BARD1 germline; cisplatin-based therapy preferred (HRD BARD1-LOF); consider olaparib in BARD1-LOF refractory NB (HRD)",
        "key_distinctions": [
            "BARD1-NB-SUSCEPTIBILITY-2-NBLST2",
            "C557S-COMMON-VARIANT-OR-2X-NB",
            "BARD1-BRCA1-RING-HETERODIMER-E3-LIGASE",
            "HRD-CISPLATIN-OLAPARIB-SENSITIVITY",
            "GWAS-2Q35-TOP-HIT-NB",
            "BARD1-LOF-5-10PCT-HIGH-RISK-SPORADIC-NB",
        ],
    },
    {
        "gene": "KIF1B",
        "protein": (
            "KIF1B -- 1p36.22 Autosomal-Dominant-LOF -- 1816aa -- "
            "Kinesin-Family-Member-1B-200kDa-Motor-Protein-"
            "NB-Susceptibility-1-1p36-LOH-70pct-NB-OMIM-605995"
        ),
        "locus": "1p36.22",
        "protein_size": (
            "1816 aa / 200 kDa / 1p36.22 KIF1B encodes kinesin family member 1B (plus-end microtubule motor protein): "
            "STRUCTURE: "
            "  1816 aa / 200 kDa; kinesin superfamily member; "
            "  Motor domain (aa 1-360): ATP hydrolysis + microtubule binding; anterograde transport; "
            "  Coiled-coil stalk: dimerisation; "
            "  Pleckstrin homology (PH) domain: cargo binding (mitochondria transport); "
            "  KIF1B transports mitochondria + synaptic vesicle precursors along axons; "
            "1p36 DELETION LOCUS IN NEUROBLASTOMA: "
            "  1p36 LOH present in 70% NB (most common chromosomal aberration in NB); "
            "  1p36 deletion: most critical in MYCN-amplified NB (1p del + MYCN = worst prognosis); "
            "  1p36.22 locus contains KIF1B tumour suppressor region; "
            "  KIF1B deleted allele: haploinsufficiency -> reduced apoptotic signalling in NB; "
            "KIF1B AS NB TUMOUR SUPPRESSOR: "
            "  Q598X truncating variant: identified in NB families (original Maris lab report 2007); "
            "  KIF1B heterozygous LOF -> insufficient pro-apoptotic signalling in sympathoadrenal NC; "
            "  KIF1B restores apoptosis in NB cell lines (colony suppression assays); "
            "NEUROENDOCRINE TUMOURS (KIF1B): "
            "  KIF1B 1p36 deletion: pheochromocytoma risk (1p LOH in ~30% pheo); "
            "  Neuroblastoma + pheochromocytoma clustering in KIF1B-LOF families; "
            "  Paraganglioma risk also (extra-adrenal pheo); "
            "  Neurocristopathy spectrum: adrenal NB, pheo, paraganglioma; "
            "SURVEILLANCE (KIF1B GERMLINE LOF): "
            "  Urine HVA/VMA q3M until age 10yr; "
            "  Metanephrines annually from age 5yr (pheo risk); "
            "  MIBG scan at diagnosis and annually; "
            "  Abdominal US q6M until age 10yr; "
            "  Panel-based germline testing in familial NB + pheo; "
        ),
        "inheritance": "Autosomal dominant LOF; Q598X and other truncating variants familial; 1p36 deletion somatic in NB tumours (distinct from germline LOF); incomplete penetrance germline variants",
        "cancer_risk": "NB: 1p36 LOH in 70% sporadic NB (somatic); germline KIF1B LOF in familial NB clusters; pheochromocytoma (1p LOH 30% pheo); paraganglioma risk; neurocristopathy spectrum including NB + pheo",
        "pathognomonic": "1p36 deletion in 70% NB PATHOGNOMONIC (somatic); 1p36 + MYCN amplification = worst prognosis NB; germline KIF1B LOF in NB + pheochromocytoma families (neurocristopathy spectrum)",
        "surveillance_key": "Urine HVA/VMA q3M; metanephrines annually from age 5yr; abdominal US q6M; MIBG scan at diagnosis; panel germline testing familial NB + pheo; 1p36 FISH/MLPA on tumour (standard staging); cascade family testing",
        "key_distinctions": [
            "1P36-LOH-70PCT-NB-PATHOGNOMONIC-SOMATIC",
            "KIF1B-NB-SUSCEPTIBILITY-1-GERMLINE-LOF",
            "PHEO-PARAGANGLIOMA-NEUROCRISTOPATHY-RISK",
            "1P36-MYCN-CO-DELETION-WORST-PROGNOSIS",
            "KIF1B-APOPTOSIS-SUPPRESSOR-SYMPATHOADRENAL",
            "Q598X-TRUNCATING-FAMILIAL-NB-CLUSTER",
        ],
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-NF1-CAL-Macules-PATHOGNOMONIC-"
            "MPNST-8-13pct-Selumetinib-FDA2020-AVOID-Radiation-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 encodes neurofibromin (RAS GTPase-activating protein): "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; one of largest tumour suppressor proteins; "
            "  GAP-related domain (GRD, aa 1189-1551): RAS-GAP activity; catalyses RAS-GTP -> RAS-GDP; "
            "  Sec14 domain: lipid binding, membrane targeting; "
            "  NF1 LOF -> constitutive RAS-MAPK activation -> cell proliferation; "
            "  NF1 second hit (somatic): biallelic LOF required for most NF1 tumours (Knudson 2-hit); "
            "NEUROFIBROMATOSIS TYPE 1 (NF1): "
            "  OMIM 162200; 1 in 3000 births; 50% de novo; "
            "  Cafe-au-lait macules ≥6 (≥5mm prepubertal, ≥15mm postpubertal) PATHOGNOMONIC; "
            "  Neurofibromas (cutaneous/subcutaneous): start puberty; "
            "  Plexiform NF: congenital; MPNST transformation 8-13% PATHOGNOMONIC (lifetime); "
            "  Lisch nodules (iris hamartomas) PATHOGNOMONIC (>90% adults); "
            "  Optic pathway glioma (OPG): 15-20% NF1; often asymptomatic; "
            "NEUROBLASTOMA IN NF1: "
            "  NF1 + NB: 2-3x elevated risk vs general population; "
            "  NB in NF1 predominantly adrenal origin; younger onset; "
            "  JMML (juvenile myelomonocytic leukaemia): 200x elevated in NF1 children (somatic NF1 + RAS); "
            "  NB DISTINCT from MPNST: neural crest derived but different lineage (sympathetic ganglion); "
            "SELUMETINIB (MEK INHIBITOR -- FDA 2020): "
            "  Selumetinib FDA approved 2020 for symptomatic inoperable plexiform NF in NF1 children ≥2yr; "
            "  MEK1/2 inhibitor: downstream of RAS pathway; shrinks plexiform NF 70-80% response; "
            "  Partial response in most: not curative; continuous dosing required; "
            "  NOT approved for NB treatment (no NB indication); MPNST: selumetinib Phase II ongoing; "
            "AVOID RADIATION (NF1): "
            "  Radiation -> secondary MPNST risk in NF1 (RAS pathway hyperactivation + radiation mutagenesis); "
            "  Optic pathway glioma: avoid cranial RT if possible (secondary MPNST); "
            "  NB treatment: standard chemotherapy; minimise RT in NF1 (secondary sarcoma risk); "
        ),
        "inheritance": "Autosomal dominant LOF; 50% de novo; full penetrance (>99%) for cafe-au-lait macules; variable expressivity for other features; biallelic NF1 LOF (somatic second hit) required for tumours",
        "cancer_risk": "MPNST 8-13% lifetime PATHOGNOMONIC (plexiform NF transformation); NB 2-3x elevated; optic glioma 15-20%; JMML 200x elevated in NF1 children; GI stromal tumour (GIST) elevated; breast cancer elevated (2x)",
        "pathognomonic": "Cafe-au-lait macules ≥6 ≥15mm PATHOGNOMONIC NF1; Lisch nodules PATHOGNOMONIC adults; plexiform NF -> MPNST transformation PATHOGNOMONIC (8-13% lifetime); JMML 200x elevated PATHOGNOMONIC NF1 children",
        "surveillance_key": "Annual full skin + ophthalmology exam; MRI brain/spine annually childhood (OPG); avoid radiation (secondary MPNST); selumetinib symptomatic inoperable PN; urine HVA/VMA q6M for NB surveillance; MPNST PET-FDG + whole-body MRI if suspected",
        "key_distinctions": [
            "CAFE-AU-LAIT-MACULES-GT6-PATHOGNOMONIC-NF1",
            "MPNST-8-13PCT-LIFETIME-PATHOGNOMONIC",
            "SELUMETINIB-FDA2020-PLEXIFORM-NF",
            "AVOID-RADIATION-SECONDARY-MPNST",
            "JMML-200X-ELEVATED-NF1-CHILDREN",
            "NB-2-3X-ELEVATED-NEURAL-CREST-ORIGIN",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-LFS-NB-Anaplastic-Relapse-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annual-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 encodes tumour suppressor protein p53: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; transcription factor; tetramer in active form; "
            "  N-terminal transactivation domains (TAD1 aa 1-40, TAD2 aa 40-61); "
            "  Proline-rich region (aa 63-97); "
            "  DNA-binding domain (DBD, aa 102-292): most mutations in cancer here; "
            "  Tetramerisation domain (aa 323-356): oligomerisation; "
            "  Regulatory domain (aa 363-393): post-translational modifications; "
            "  R175H, R248W, R248Q, R273H, R273C, R249S: hotspot GOF mutations (dominant negative); "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  OMIM 151623; ~50% germline TP53 LOF; "
            "  Classic LFS: sarcoma <45yr + brain tumour + adrenal cortical carcinoma + breast <45yr; "
            "  Chompret criteria (2015): early onset tumour <46yr + family history; "
            "  Childhood ACC (adrenocortical carcinoma): TP53 germline 50-70% paediatric ACC PATHOGNOMONIC; "
            "NB IN LI-FRAUMENI / TP53: "
            "  NB in LFS: 5-10% (less common than STS/brain but documented); "
            "  TP53 somatic mutations in relapsed NB: 10-15% relapse (ALT pathway + TP53 co-mutation); "
            "  ALT pathway (ATRX/DAXX LOF + TP53 LOF): relapse-associated anaplastic NB; "
            "  MYCN amplification + TP53 somatic: rare first presentation; "
            "AVOID RADIATION ABSOLUTELY (LFS): "
            "  Germline TP53 LFS: AVOID RADIATION ABSOLUTELY; "
            "  Radiation -> second primary malignancy risk: sarcoma in radiation field; "
            "  NB radiotherapy (involved-field): OMIT if germline TP53 confirmed; "
            "  Tandem HDCT/ASCT preferred over total-body irradiation (TBI) in TP53 LFS NB; "
            "WBMRI TORONTO ANNUAL: "
            "  Whole-body MRI (WBMRI) Toronto Protocol annually; NOT PET-CT/CT (radiation); "
            "  Brain MRI + abdominal/chest MRI included; "
            "  Annual WBMRI from diagnosis (lifelong from diagnosis of germline TP53); "
        ),
        "inheritance": "Autosomal dominant LOF; 50% de novo; dominant negative GOF hotspot variants (R175H, R248W, R248Q) worsen phenotype via tetramer poisoning; near-complete penetrance (>90% lifetime cancer risk)",
        "cancer_risk": "LFS: sarcoma 30-50% dominant; brain tumour 15-20%; ACC 3-5% (paediatric ACC 50-70% TP53 PATHOGNOMONIC); breast <45yr 25-35%; NB 5-10% (relapse-associated somatic TP53 more common than germline primary NB)",
        "pathognomonic": "Paediatric adrenocortical carcinoma PATHOGNOMONIC LFS (50-70% paediatric ACC = TP53 germline); R337H 1/300 South Brazilian founder mutation PATHOGNOMONIC; ALT pathway NB relapse (ATRX/DAXX + TP53 somatic co-mutation)",
        "surveillance_key": "WBMRI Toronto annually (NOT CT/PET); AVOID RADIATION ABSOLUTELY; omit NB radiotherapy if TP53 germline; tandem HDCT/ASCT preferred over TBI; annual rapid MRI brain + abdominal MRI; breast MRI from age 20yr",
        "key_distinctions": [
            "LFS-AVOID-RADIATION-ABSOLUTELY-NB",
            "WBMRI-TORONTO-ANNUALLY-NOT-CT",
            "PAEDIATRIC-ACC-50-70PCT-TP53-PATHOGNOMONIC",
            "ALT-PATHWAY-ATRX-TP53-RELAPSE-NB",
            "OMIT-NB-RT-IF-TP53-GERMLINE",
            "R337H-SOUTH-BRAZIL-1IN300-FOUNDER",
        ],
    },
    {
        "gene": "DICER1",
        "protein": (
            "DICER1 -- 14q32.13 Autosomal-Dominant-LOF -- 1922aa -- "
            "DICER1-219kDa-RNase-III-miRNA-Processor-PPB-PATHOGNOMONIC-"
            "Cervical-ERMS-PATHOGNOMONIC-AVOID-Radiation-CT-Chest-LT8yr-OMIM-606241"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "1922 aa / 219 kDa / 14q32.13 DICER1 encodes DICER1 (RNase III endoribonuclease / miRNA processor): "
            "STRUCTURE: "
            "  1922 aa / 219 kDa; RNase III family endonuclease; "
            "  Helicase domain (aa 1-605); DUF283 domain (aa 608-699); "
            "  PAZ domain (aa 768-875): dsRNA 3' end binding; "
            "  Linker (aa 876-1000); "
            "  RNase IIIa domain (aa 1000-1100): cleaves miRNA* strand; "
            "  RNase IIIb domain (aa 1218-1380): cleaves mature miRNA strand; "
            "    RNase IIIb hotspot mutations (E1705K, D1709N, E1813G): second somatic hit PATHOGNOMONIC; "
            "  DICER1 LOF -> loss of mature miRNA processing -> derepression of oncogenes; "
            "DICER1 SYNDROME: "
            "  OMIM 601200; pleiotropic tumour predisposition; "
            "  Pleuropulmonary blastoma (PPB): PATHOGNOMONIC DICER1 syndrome; "
            "    Type I (cystic <2yr): lowest mortality (~14%) -- DICER1 germline required; "
            "    Type II (mixed cystic/solid 2-4yr); Type III (solid >2yr): higher mortality; "
            "    PPB sibling screen: ALL siblings <8yr -> chest CT (even if DICER1 untested); "
            "  Cystic nephroma: PATHOGNOMONIC DICER1 -- anaplastic Wilms progression risk; "
            "  Cervical ERMS (embryonal rhabdomyosarcoma): PATHOGNOMONIC DICER1 (adolescent females); "
            "    Fertility-sparing surgery if possible; vaginal/cervical mass in teen = DICER1 screen; "
            "NB IN DICER1 SYNDROME: "
            "  NB risk 2-4x in DICER1 germline carriers (case series evidence); "
            "  Adrenal ganglioneuroblastoma (GNB) and NB documented in DICER1 families; "
            "  Mechanism: loss of miRNA processing -> derepressed MYCN/LIN28B/IGF2BP1; "
            "  DICER1 RNase IIIb somatic hotspot (second hit) required for tumour development; "
            "AVOID RADIATION IN CHILDREN (DICER1): "
            "  Developing organs at highest risk from RT in DICER1 patients; "
            "  CT surveillance replaced by MRI/US where possible; "
            "  AVOID RT even for PPB if alternative protocol feasible; "
        ),
        "inheritance": "Autosomal dominant LOF; germline DICER1 LOF (first hit) + somatic RNase IIIb hotspot (second hit) required for most DICER1 tumours; de novo mutations ~10%; penetrance variable by tumour type",
        "cancer_risk": "PPB (Type I-III) PATHOGNOMONIC; cystic nephroma PATHOGNOMONIC; cervical ERMS PATHOGNOMONIC; NB 2-4x; multinodular goitre (MNG) 75% females; Sertoli-Leydig cell tumour ovary (SLCT) PATHOGNOMONIC; pineoblastoma elevated",
        "pathognomonic": "PPB PATHOGNOMONIC DICER1 (Type I cystic <2yr -- sibling CT chest <8yr ALL); cervical ERMS (adolescent female) PATHOGNOMONIC DICER1; RNase IIIb somatic hotspot (E1705/D1709/E1813) PATHOGNOMONIC second hit",
        "surveillance_key": "Chest CT siblings <8yr (PPB risk); annual thyroid US from age 8yr; pelvic US adolescent females (SLCT/cervical ERMS); AVOID radiation children; nephrology follow-up cystic nephroma; urine HVA/VMA NB monitoring; NB MRI preferred over CT",
        "key_distinctions": [
            "PPB-TYPE-I-CYSTIC-PATHOGNOMONIC-DICER1",
            "CERVICAL-ERMS-PATHOGNOMONIC-DICER1",
            "CT-CHEST-SIBLINGS-LT-8YR-PPB",
            "AVOID-RADIATION-CHILDREN-DICER1",
            "RNASE-IIIB-HOTSPOT-SOMATIC-SECOND-HIT-PATHOGNOMONIC",
            "NB-2-4X-ELEVATED-DICER1-SYNDROME",
        ],
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Biallelic-AR-LOF-FA-D1 / AD-LOF-HBOC -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-FAD1-Paediatric-Solid-Tumours-"
            "Cisplatin-HRD-Sibling-Donor-Exclusion-MANDATORY-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 384 kDa / 13q12.3 BRCA2 encodes BRCA2 (HR repair scaffold; FA-D1 gene): "
            "STRUCTURE: "
            "  3418 aa / 384 kDa; large nuclear HR scaffold protein; "
            "  PALB2-binding domain (aa 10-40): nuclear targeting; "
            "  BRC repeats x8 (aa 1002-2085): RAD51 loading at DSBs; "
            "  DBD (aa 2402-3190): ssDNA/dsDNA binding; "
            "  Biallelic BRCA2 LOF: Fanconi anemia complementation group D1 (FA-D1/FANCD1); "
            "FANCONI ANEMIA D1 (FA-D1) PAEDIATRIC SOLID TUMOURS: "
            "  OMIM 605724; most severe FA subtype; childhood cancer onset 2-5yr; "
            "  Bilateral Wilms tumour: PATHOGNOMONIC FA-D1 (50-60%); "
            "  Medulloblastoma (SHH-subtype): 10-15% PATHOGNOMONIC; "
            "  Embryonal RMS: 5-10%; "
            "  ALL (acute lymphoblastic leukaemia): 20-30%; "
            "  NB: rare reported in FA-D1; part of broad paediatric solid tumour spectrum; "
            "  BMF (bone marrow failure): near universal FA-D1 (aplastic anaemia); "
            "  VACTERL association: vertebral + cardiac + TE fistula + limb anomalies; "
            "  DEB/MMC chromosomal fragility test PATHOGNOMONIC FA diagnosis; "
            "SIBLING DONOR EXCLUSION (FA-D1): "
            "  ALL potential HSCT donors must be DEB/MMC tested BEFORE donor evaluation; "
            "  Both parents obligate heterozygous BRCA2 carriers (HBOC risk for parents); "
            "  Siblings: 25% FA-D1 risk + 50% HBOC carrier risk; "
            "  Sibling FA-D1 cannot donate for HSCT; "
            "MONOALLELIC BRCA2 (HBOC): "
            "  Monoallelic AD LOF: breast 47-69%, ovarian 11-17%, pancreatic 3-5%, prostate 6-9%; "
            "  NB/paediatric solid tumour NOT established risk for monoallelic BRCA2; "
            "HRD SENSITIVITY: "
            "  FA-D1 NB: cisplatin preferred (reduced alkylating agents avoid BMF); "
            "  Olaparib: PARP inhibitor activity in BRCA2-mutant solid tumours; "
            "  Avoid alkylating agents (cyclophosphamide, ifosfamide) in FA-D1 (severe BMF); "
        ),
        "inheritance": "Biallelic AR LOF (FA-D1): compound heterozygous BRCA2 -- both parents obligate het; monoallelic AD LOF (HBOC): 50% transmission; de novo FA-D1 possible but rare (parents usually HBOC carriers)",
        "cancer_risk": "FA-D1 biallelic: bilateral Wilms PATHOGNOMONIC; medulloblastoma PATHOGNOMONIC; ALL 20-30%; embryonal RMS 5-10%; BMF near-universal; NB rare reported; monoallelic HBOC: breast 47-69%, ovarian 11-17%",
        "pathognomonic": "Bilateral Wilms PATHOGNOMONIC FA-D1; DEB/MMC chromosomal fragility PATHOGNOMONIC FA diagnosis; VACTERL + BMF + childhood solid tumour = FA-D1 phenotype; sibling donor exclusion MANDATORY",
        "surveillance_key": "DEB/MMC test ALL potential HSCT donors MANDATORY; sibling donor exclusion before HSCT; reduced alkylating agent protocol (avoid cyclophosphamide/ifosfamide FA-D1 BMF); cisplatin preferred; androgens bridge BMF; monoallelic BRCA2: PBSO age 40-45yr",
        "key_distinctions": [
            "FA-D1-BILATERAL-WILMS-PATHOGNOMONIC",
            "DEB-MMC-CHROMOSOMAL-FRAGILITY-PATHOGNOMONIC-FA",
            "SIBLING-DONOR-EXCLUSION-MANDATORY-HSCT",
            "AVOID-ALKYLATING-AGENTS-FA-D1-BMF",
            "CISPLATIN-HRD-SENSITIVITY-BRCA2",
            "VACTERL-BMF-CHILDHOOD-SOLID-TUMOUR-FA-D1",
        ],
    },
]

TUMOR_TYPES = {
    "ALK": {
        "Adrenal Neuroblastoma": 22,
        "Thoracic Ganglioneuroblastoma": 8,
        "Cervical NB": 4,
        "Abdominal Para-adrenal NB": 4,
        "Bilateral Adrenal NB": 2,
    },
    "PHOX2B": {
        "Adrenal Neuroblastoma (CCHS)": 18,
        "Thoracic NB (CCHS)": 10,
        "Cervical Ganglioneuroblastoma": 6,
        "Mediastinal NB": 6,
    },
    "BARD1": {
        "Adrenal NB High-Risk": 26,
        "Para-adrenal NB": 8,
        "Abdominal NB": 6,
    },
    "KIF1B": {
        "Adrenal Neuroblastoma": 20,
        "Pheochromocytoma": 10,
        "Para-adrenal NB": 6,
        "Paraganglioma": 4,
    },
    "NF1": {
        "Adrenal NB (NF1)": 16,
        "Plexiform NF -> MPNST": 12,
        "Thoracic NB": 8,
        "JMML": 4,
    },
    "TP53": {
        "Adrenal NB (relapse)": 14,
        "Adrenal NB (primary LFS)": 8,
        "Anaplastic NB (ALT)": 10,
        "Ganglioblastoma Relapse": 8,
    },
    "DICER1": {
        "Adrenal NB": 16,
        "Ganglioneuroblastoma": 10,
        "Pleuropulmonary Blastoma": 8,
        "Cervical ERMS": 6,
    },
    "BRCA2": {
        "Bilateral Wilms (FA-D1)": 16,
        "Medulloblastoma (FA-D1)": 10,
        "Adrenal NB (FA-D1)": 6,
        "Embryonal RMS (FA-D1)": 8,
    },
}

PATHOGENIC_VARIANTS = {
    "ALK": {
        "p.Arg1275Gln (R1275Q)": 16,
        "p.Phe1174Leu (F1174L)": 10,
        "p.Phe1245Cys (F1245C)": 6,
        "p.Asp1091Asn (D1091N)": 4,
        "p.Ile1250Thr (I1250T)": 4,
    },
    "PHOX2B": {
        "c.723delG (NPARM fs)": 12,
        "20/27 polyalanine": 14,
        "20/28 polyalanine": 8,
        "p.Arg100Leu (R100L)": 6,
    },
    "BARD1": {
        "p.Cys557Ser (C557S)": 18,
        "c.1921+1G>A (splice)": 10,
        "p.Gln564Ter (Q564X)": 8,
        "p.Leu622Phe": 4,
    },
    "KIF1B": {
        "p.Gln598Ter (Q598X)": 14,
        "1p36 del (MLPA)": 16,
        "c.1194+2T>C (splice)": 6,
        "p.Arg304Trp": 4,
    },
    "NF1": {
        "p.Arg1947Ter (R1947X)": 10,
        "c.2033del (fs)": 10,
        "Exon 1-6 del (MLPA)": 8,
        "p.Gln519Ter": 6,
        "p.Arg304Ter": 6,
    },
    "TP53": {
        "p.Arg248Trp (R248W)": 10,
        "p.Arg175His (R175H)": 8,
        "p.Arg273His (R273H)": 8,
        "p.Arg337His (R337H)": 10,
        "p.Pro151Ser": 4,
    },
    "DICER1": {
        "p.Glu1705Lys (E1705K)": 12,
        "p.Asp1709Asn (D1709N)": 10,
        "c.5438+1G>A (splice)": 8,
        "p.Leu1264Pro": 6,
        "p.Arg1412Ter": 4,
    },
    "BRCA2": {
        "p.Lys2729Thr (K2729T)": 8,
        "c.8954-2A>G (splice)": 10,
        "p.Asp2723His": 8,
        "p.Trp2626Ter": 8,
        "p.Asn991Ile": 6,
    },
}

TREATMENT_PROTOCOLS = {
    "ALK": [
        "Induction: COG ANBL1232 (high-risk NB): cisplatin + carboplatin + doxorubicin + etoposide + cyclophosphamide",
        "ALK targeted: Crizotinib FDA 2022 (relapsed/refractory ALK-mutant NB)",
        "Lorlatinib (3rd-gen ALK inhibitor): F1174L crizotinib-resistant allele",
        "Tandem HDCT/ASCT: thiotepa + cyclophosphamide then carboplatin + etoposide + melphalan",
        "Anti-GD2: Dinutuximab maintenance immunotherapy (COG ANBL0032 standard)",
        "MIBG therapy: 131I-MIBG FDA 2018 for relapsed/refractory MIBG-avid NB",
    ],
    "PHOX2B": [
        "NB (CCHS): standard INRG risk stratification chemotherapy",
        "CCHS ventilatory support: NIV (non-invasive ventilation) lifelong",
        "Diaphragm pacing: bilateral hemidiaphragm pacing (freedom from NIV)",
        "Anesthesia: HIGH RISK -- pre-procedure respiratory management plan MANDATORY",
        "Anti-GD2: dinutuximab (high-risk CCHS-NB)",
        "Annual 24-hr Holter: cardiac dysrhythmia monitoring (20/28+ repeats)",
    ],
    "BARD1": [
        "High-risk NB: COG ANBL1232 standard induction chemotherapy",
        "Cisplatin preferred: HRD in BARD1-LOF (BRCA1 complex partner)",
        "Olaparib compassionate use: BARD1-LOF NB with HRD (case basis)",
        "Anti-GD2: dinutuximab/naxitamab maintenance",
        "HDCT/ASCT consolidation: high-risk INRG M disease",
        "131I-MIBG: relapsed MIBG-avid NB",
    ],
    "KIF1B": [
        "NB: standard INRG risk-adapted chemotherapy",
        "Pheochromocytoma: alpha-blockade (phenoxybenzamine) then surgery",
        "Paraganglioma: surgical resection; 131I-MIBG for metastatic pheo",
        "Anti-GD2 (dinutuximab) for high-risk/relapsed NB",
        "HDCT/ASCT: high-risk NB consolidation",
    ],
    "NF1": [
        "NB in NF1: standard INRG risk-adapted chemotherapy",
        "Selumetinib: symptomatic inoperable plexiform NF (FDA 2020, ≥2yr)",
        "MPNST: ifosfamide + doxorubicin (first-line sarcoma chemotherapy)",
        "Avoid radiation: secondary MPNST risk in NF1",
        "JMML: HSCT curative (only curative for JMML NF1)",
        "Trametinib: MEK inhibitor clinical trials for MPNST (NF1-driven)",
    ],
    "TP53": [
        "NB: AVOID RADIATION ABSOLUTELY (LFS) -- omit involved-field RT",
        "Tandem HDCT/ASCT: preferred over total body irradiation (TBI) in LFS",
        "Standard induction: cisplatin + carboplatin + doxorubicin",
        "WBMRI annually: surveillance NOT CT/PET",
        "Anti-GD2 maintenance: dinutuximab (standard high-risk NB)",
        "LFS genetic counselling: all first-degree relatives 50% risk",
    ],
    "DICER1": [
        "NB: standard INRG risk-adapted chemotherapy",
        "Avoid radiation in children: developing organs radiation-sensitive",
        "PPB Type I: lung-sparing surgery where feasible; chemotherapy (VAC)",
        "Cervical ERMS: fertility-sparing surgery + chemotherapy",
        "Anti-GD2: dinutuximab for high-risk NB",
        "MIBG scan surveillance from diagnosis in DICER1 NB",
    ],
    "BRCA2": [
        "FA-D1 NB: MODIFIED CHEMOTHERAPY -- avoid alkylating agents (cyclophosphamide, ifosfamide): severe BMF",
        "Cisplatin preferred over cyclophosphamide in FA-D1 NB (HR-deficient sensitivity)",
        "HSCT: curative for BMF component (NOT solid tumours -- SIBLING DONOR EXCLUSION MANDATORY)",
        "DEB/MMC test: ALL potential HSCT donors MANDATORY pre-evaluation",
        "Bilateral Wilms (FA-D1): nephron-sparing surgery aim",
        "Olaparib: PARP inhibitor activity in BRCA2-deficient tumours",
        "Androgens: temporary BMF bridge before HSCT",
    ],
}

SURVEILLANCE_PROTOCOLS = {
    "ALK": [
        "Urine HVA/VMA + catecholamines: q3M from birth until age 10yr",
        "Abdominal US: q3M until age 10yr (adrenal surveillance)",
        "MIBG scan (123I): at diagnosis + restaging; consider annual first 5yr",
        "MRI chest/abdomen: preferred over CT (lifetime radiation reduction)",
        "Cascade testing: germline ALK panel all first-degree relatives",
        "ALK FISH tumour: amplification (somatic) vs GOF mutation (germline/somatic)",
    ],
    "PHOX2B": [
        "Life-long NIV/ventilatory support: CCHS; no sedation unsupervised",
        "Annual 24-hr Holter: cardiac dysrhythmia (20/28+ polyalanine repeat)",
        "Urine HVA/VMA q3M: NB surveillance (all CCHS)",
        "MIBG annually: NPARM patients (50% NB risk)",
        "Abdominal US q6M: adrenal surveillance in NPARM CCHS",
        "Hirschsprung work-up: 20/28+ repeat (20% Hirschsprung risk)",
    ],
    "BARD1": [
        "Urine HVA/VMA q3M until age 5yr for BARD1 LOF carriers",
        "Abdominal US q6M first 5yr: adrenal surveillance",
        "Cascade family testing: BARD1 germline LOF relatives",
        "No consensus CT/MRI protocol: individualise based on family history",
        "GWAS screening: BARD1 panel in NB research cohorts",
    ],
    "KIF1B": [
        "Urine HVA/VMA q3M until age 10yr: NB surveillance",
        "Plasma/urine metanephrines annually from age 5yr: pheochromocytoma",
        "Abdominal US q6M until age 10yr: adrenal surveillance",
        "MIBG scan at NB diagnosis: staging + pheo",
        "1p36 FISH/MLPA on tumour: standard NB staging panel",
        "Annual BP measurement: hypertension (pheo screening)",
    ],
    "NF1": [
        "Annual full dermatology + ophthalmology exam: CALM, Lisch nodules",
        "MRI brain/spine annually childhood: OPG + CNS gliomas",
        "Whole-body MRI q2yr adults: MPNST surveillance in PN",
        "Urine HVA/VMA q6M: NB surveillance (elevated risk)",
        "Avoid CT/radiation: secondary MPNST risk",
        "PET-FDG (or WB-MRI) if rapid plexiform NF growth: MPNST suspect",
    ],
    "TP53": [
        "WBMRI Toronto Protocol annually: whole-body MRI (NOT CT/PET)",
        "Brain MRI 6-monthly first 5yr then annually",
        "Annual clinical exam: skin (osteosarcoma/STS awareness)",
        "Breast MRI annually from age 20yr (LFS)",
        "AVOID ALL RADIATION for surveillance (no CT/PET-CT/DEXA)",
        "CASCADE: 50% risk first-degree relatives -- offer germline TP53 testing",
    ],
    "DICER1": [
        "Chest CT siblings <8yr: PPB risk screening MANDATORY",
        "Annual thyroid US from age 8yr: MNG (multinodular goitre)",
        "Pelvic/abdominal US annually adolescent females: SLCT, cervical ERMS",
        "Nephrology follow-up: cystic nephroma surveillance",
        "Urine HVA/VMA q6M: NB monitoring",
        "MRI preferred over CT in children: radiation minimisation",
    ],
    "BRCA2": [
        "DEB/MMC test ALL potential HSCT donors MANDATORY",
        "Sibling donor exclusion: before HSCT donor evaluation (ALL siblings tested)",
        "Modified FA chemotherapy: avoid cyclophosphamide/ifosfamide (severe BMF)",
        "Androgens bridge (oxymetholone): pre-HSCT BMF management",
        "Monoallelic BRCA2 HBOC: PBSO age 40-45yr (ovarian prevention)",
        "Cisplatin-based chemotherapy preferred (HRD BRCA2-LOF sensitivity)",
    ],
}


def _make_cohort(gene_idx: int, seed: int) -> list:
    """Generate 40-patient cohort for one neuroblastoma predisposition gene."""
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]["gene"]

    # Age-at-diagnosis ranges (years) by gene
    age_ranges = {
        "ALK":    (0.5, 6.0),
        "PHOX2B": (0.1, 3.0),
        "BARD1":  (0.5, 7.0),
        "KIF1B":  (0.5, 8.0),
        "NF1":    (1.0, 9.0),
        "TP53":   (1.5, 10.0),
        "DICER1": (0.5, 7.0),
        "BRCA2":  (0.5, 5.0),
    }
    lo, hi = age_ranges[gene]

    # INRG risk group distribution by gene
    inrg_weights = {
        "ALK":    ["L1", "L2", "MS", "M", "M", "M"],
        "PHOX2B": ["L1", "L2", "MS", "M", "M"],
        "BARD1":  ["L2", "M", "M", "M", "M"],
        "KIF1B":  ["L1", "L2", "L2", "M", "M"],
        "NF1":    ["L1", "L1", "L2", "M", "M"],
        "TP53":   ["L2", "M", "M", "M", "M"],
        "DICER1": ["L1", "L2", "L2", "M", "M"],
        "BRCA2":  ["L2", "L2", "M", "M", "M"],
    }

    # Histology distribution by gene
    hist_weights = {
        "ALK":    ["FH", "FH", "UH", "GNB", "MYCN-amp"],
        "PHOX2B": ["FH", "FH", "UH", "GNB"],
        "BARD1":  ["UH", "UH", "MYCN-amp", "FH"],
        "KIF1B":  ["FH", "FH", "UH", "GNB", "Pheo"],
        "NF1":    ["FH", "FH", "GNB", "MPNST", "UH"],
        "TP53":   ["UH", "UH", "ALT-pathway", "MYCN-amp", "FH"],
        "DICER1": ["FH", "GNB", "UH", "PPB"],
        "BRCA2":  ["Wilms-FA", "Medulloblastoma", "FH", "RMS-FA"],
    }

    # CR/response rates by gene (probability 0-1)
    cr_prob = {
        "ALK": 0.68,
        "PHOX2B": 0.72,
        "BARD1": 0.55,
        "KIF1B": 0.62,
        "NF1": 0.65,
        "TP53": 0.52,
        "DICER1": 0.70,
        "BRCA2": 0.60,
    }

    radiation_prob = {
        "ALK": 0.45,
        "PHOX2B": 0.38,
        "BARD1": 0.48,
        "KIF1B": 0.40,
        "NF1": 0.15,   # NF1: avoid radiation (secondary MPNST risk)
        "TP53": 0.0,   # TP53 LFS: AVOID RADIATION ABSOLUTELY
        "DICER1": 0.18, # Minimise radiation in children
        "BRCA2": 0.22,
    }

    targeted_prob = {
        "ALK": 0.72,   # Crizotinib / lorlatinib
        "PHOX2B": 0.20,
        "BARD1": 0.18,
        "KIF1B": 0.10,
        "NF1": 0.45,   # Selumetinib (PN); trametinib (MPNST)
        "TP53": 0.10,
        "DICER1": 0.15,
        "BRCA2": 0.30,  # Olaparib / cisplatin HRD
    }

    relapse_prob = {
        "ALK": 0.38,
        "PHOX2B": 0.32,
        "BARD1": 0.50,
        "KIF1B": 0.42,
        "NF1": 0.40,
        "TP53": 0.55,
        "DICER1": 0.35,
        "BRCA2": 0.48,
    }

    gtr_prob = {
        "ALK": 0.55,
        "PHOX2B": 0.48,
        "BARD1": 0.45,
        "KIF1B": 0.50,
        "NF1": 0.40,
        "TP53": 0.50,
        "DICER1": 0.55,
        "BRCA2": 0.45,
    }

    variant_pool = list(PATHOGENIC_VARIANTS[gene].keys())

    patients = []
    for pid in range(1, 41):
        age = round(rng.uniform(lo, hi), 1)
        inrg = rng.choice(inrg_weights[gene])
        hist = rng.choice(hist_weights[gene])
        cr = rng.random() < cr_prob[gene]
        radiation = rng.random() < radiation_prob[gene]
        targeted = rng.random() < targeted_prob[gene]
        relapse = rng.random() < relapse_prob[gene]
        gtr = rng.random() < gtr_prob[gene]
        variant = rng.choice(variant_pool)

        # Response
        if cr:
            response = "CR"
        elif rng.random() < 0.4:
            response = "PR"
        elif rng.random() < 0.5:
            response = "SD"
        else:
            response = "PD"

        # Treatment
        if gene == "ALK" and targeted:
            treatment = rng.choice([
                "Induction+Crizotinib", "Induction+Lorlatinib", "COG-ANBL1232+Crizotinib",
            ])
        elif gene == "PHOX2B":
            treatment = rng.choice([
                "Standard-Induction+NIV-Support", "High-Risk-NB-Protocol",
                "Anti-GD2-Maintenance", "HDCT-ASCT",
            ])
        elif gene == "BARD1":
            treatment = rng.choice([
                "Cisplatin-Based-Induction", "Standard-NB-Protocol", "Anti-GD2+HDCT",
            ])
        elif gene == "KIF1B":
            treatment = rng.choice([
                "Standard-NB-Induction", "Pheo-Alpha-Block-Surgery", "Anti-GD2-Maintenance",
            ])
        elif gene == "NF1":
            treatment = rng.choice([
                "Standard-NB-Induction", "Selumetinib-PN", "Ifosfamide-Dox-MPNST", "JMML-HSCT",
            ])
        elif gene == "TP53":
            treatment = rng.choice([
                "No-RT-Modified-Protocol", "Tandem-HDCT-ASCT-No-TBI",
                "Cisplatin-Carboplatin-NB", "Anti-GD2-No-RT",
            ])
        elif gene == "DICER1":
            treatment = rng.choice([
                "Standard-NB-Induction", "PPB-VAC-Protocol", "Cervical-ERMS-Surgery",
                "Anti-GD2-Maintenance",
            ])
        else:  # BRCA2
            treatment = rng.choice([
                "Modified-FA-No-Alkylating", "Cisplatin-Based-FA-D1",
                "HSCT-BMF-Curative", "Bilateral-Nephrectomy-Wilms",
            ])

        patients.append({
            "gene": gene,
            "patient_id": f"{gene}-{pid:03d}",
            "age_dx": age,
            "inrg_risk": inrg,
            "histology": hist,
            "cr_achieved": cr,
            "response": response,
            "radiation_received": radiation,
            "targeted_therapy": targeted,
            "relapse": relapse,
            "gtr_resection": gtr,
            "treatment": treatment,
            "variant": variant,
        })
    return patients


def generate_overview() -> dict:
    """Generate Hereditary-Neuroblastoma-Predisposition-Atlas overview."""
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(len(ATLAS_GENES))]
    total = sum(len(c) for c in cohorts)

    # Aggregate stats
    all_patients = [p for cohort in cohorts for p in cohort]
    cr_pct = round(100 * sum(1 for p in all_patients if p["cr_achieved"]) / total)
    targeted_pct = round(100 * sum(1 for p in all_patients if p["targeted_therapy"]) / total)
    radiation_pct = round(100 * sum(1 for p in all_patients if p["radiation_received"]) / total)
    relapse_pct = round(100 * sum(1 for p in all_patients if p["relapse"]) / total)
    gtr_pct = round(100 * sum(1 for p in all_patients if p["gtr_resection"]) / total)
    mean_age = round(sum(p["age_dx"] for p in all_patients) / total, 1)

    # Top tumor types across all genes
    tumor_counts: dict = {}
    for gdef in ATLAS_GENES:
        for ttype, n in TUMOR_TYPES[gdef["gene"]].items():
            tumor_counts[ttype] = tumor_counts.get(ttype, 0) + n
    top_tumors = dict(sorted(tumor_counts.items(), key=lambda x: -x[1])[:10])

    gene_summaries = []
    for i, (gdef, cohort) in enumerate(zip(ATLAS_GENES, cohorts)):
        ages = [p["age_dx"] for p in cohort]
        cr = round(100 * sum(p["cr_achieved"] for p in cohort) / len(cohort))
        rad = round(100 * sum(p["radiation_received"] for p in cohort) / len(cohort))
        tgt = round(100 * sum(p["targeted_therapy"] for p in cohort) / len(cohort))
        gene_summaries.append({
            "gene": gdef["gene"],
            "locus": gdef["locus"],
            "protein": gdef["protein"],
            "n_patients": len(cohort),
            "mean_age_dx": round(sum(ages) / len(ages), 1),
            "cr_pct": cr,
            "radiation_pct": rad,
            "targeted_pct": tgt,
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
        })

    return _json_safe({
        "atlas": "Hereditary-Neuroblastoma-Predisposition-Atlas",
        "subtitle": "Complete 8-Gene Reference: ALK-PHOX2B-BARD1-KIF1B-NF1-TP53-DICER1-BRCA2",
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_age_dx": mean_age,
        "cr_pct": cr_pct,
        "gtr_pct": gtr_pct,
        "targeted_pct": targeted_pct,
        "radiation_pct": radiation_pct,
        "relapse_pct": relapse_pct,
        "genes": [g["gene"] for g in ATLAS_GENES],
        "gene_summaries": gene_summaries,
        "top_tumor_types": top_tumors,
        "key_pathognomonic": {
            "ALK_F1174L": "F1174L + MYCN amplification somatic PATHOGNOMONIC poorest prognosis ALK-NB; lorlatinib required (crizotinib-resistant)",
            "ALK_FAMILIAL": "Familial NB (bilateral adrenal or 2+ first-degree relatives) PATHOGNOMONIC hereditary NB -- ALK #1 germline cause (75-80%)",
            "PHOX2B_NPARM": "NPARMs (frameshift/missense PHOX2B) = NB 50% risk PATHOGNOMONIC + most severe CCHS",
            "PHOX2B_CCHS": "CCHS (congenital central hypoventilation / Ondine's curse) PATHOGNOMONIC PHOX2B",
            "NF1_CAFE_AU_LAIT": "Cafe-au-lait macules ≥6 ≥15mm PATHOGNOMONIC NF1; plexiform NF -> MPNST 8-13% lifetime PATHOGNOMONIC",
            "TP53_AVOID_RT": "AVOID RADIATION ABSOLUTELY if TP53 LFS -- omit NB RT; tandem HDCT/ASCT preferred over TBI",
            "DICER1_PPB": "PPB Type I (cystic lung <2yr) PATHOGNOMONIC DICER1; ALL siblings <8yr -> chest CT MANDATORY",
            "BRCA2_FA_D1": "Bilateral Wilms PATHOGNOMONIC FA-D1 (biallelic BRCA2); sibling donor exclusion MANDATORY pre-HSCT",
        },
        "key_management_rules": [
            "ALK germline GOF (F1174L): Lorlatinib preferred (crizotinib-resistant allele); lorlatinib 3rd-gen ALK/ROS1 inhibitor",
            "PHOX2B NPARM: anesthesia HIGH RISK -- respiratory centre failure -- NIV mandatory at induction",
            "NF1: AVOID RADIATION -- secondary MPNST risk; selumetinib FDA 2020 for symptomatic inoperable plexiform NF",
            "TP53 LFS: AVOID RADIATION ABSOLUTELY -- omit NB involved-field RT; tandem HDCT/ASCT not TBI",
            "DICER1: chest CT ALL siblings <8yr (PPB risk); AVOID radiation in children",
            "BRCA2 FA-D1: AVOID alkylating agents (cyclophosphamide/ifosfamide) -- severe BMF; DEB/MMC ALL potential donors MANDATORY",
            "BARD1 LOF: HRD context -- prefer cisplatin-based induction; consider olaparib in refractory disease",
            "KIF1B 1p36: annual metanephrines from age 5yr (pheochromocytoma surveillance)",
        ],
        "clinical_pearls": [
            "ALK R1275Q (~40%) vs F1174L (~25%): F1174L MOST AGGRESSIVE -- MYCN co-amplification frequent; lorlatinib required",
            "PHOX2B tumor-only mutations in sporadic NB are SOMATIC -- do NOT diagnose CCHS from tumor tissue",
            "1p36 LOH present in 70% all NB tumours (somatic) -- KIF1B germline LOF is DISTINCT from somatic 1p36 deletion",
            "NF1-associated NB: JMML distinct (200x elevated) vs NB (2-3x) -- different haematologic vs solid lineage",
            "ALK germline GOF penetrance ~57% cumulative -- incomplete penetrance means unaffected parent can carry high-risk allele",
            "Anti-GD2 therapy (dinutuximab, naxitamab-gqgk FDA 2020) is standard for high-risk NB regardless of germline gene",
            "131I-MIBG (iobenguane FDA 2018) for relapsed MIBG-avid NB -- check MIBG avidity at diagnosis",
            "MYCN amplification is SOMATIC (not germline) -- does not define hereditary NB; ALK is the key germline gene",
        ],
    })


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Neuroblastoma-Predisposition-Atlas."""
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(len(ATLAS_GENES))]

    breakdown = []
    for i, (gdef, cohort) in enumerate(zip(ATLAS_GENES, cohorts)):
        ages = [p["age_dx"] for p in cohort]
        cr_pct = round(100 * sum(p["cr_achieved"] for p in cohort) / len(cohort))
        rad_pct = round(100 * sum(p["radiation_received"] for p in cohort) / len(cohort))
        tgt_pct = round(100 * sum(p["targeted_therapy"] for p in cohort) / len(cohort))
        rel_pct = round(100 * sum(p["relapse"] for p in cohort) / len(cohort))
        gtr_pct = round(100 * sum(p["gtr_resection"] for p in cohort) / len(cohort))

        # Top tumor types
        top_tumors = dict(
            sorted(TUMOR_TYPES[gdef["gene"]].items(), key=lambda x: -x[1])[:5]
        )
        # Top variants
        top_variants = dict(
            sorted(PATHOGENIC_VARIANTS[gdef["gene"]].items(), key=lambda x: -x[1])[:5]
        )
        # Top treatments
        from collections import Counter
        treat_ctr = Counter(p["treatment"] for p in cohort)
        top_treats = dict(treat_ctr.most_common(5))

        breakdown.append({
            "gene": gdef["gene"],
            "locus": gdef["locus"],
            "protein_size": gdef["protein_size"],
            "inheritance": gdef["inheritance"],
            "n_patients": len(cohort),
            "mean_age_dx": round(sum(ages) / len(ages), 1),
            "cr_pct": cr_pct,
            "radiation_pct": rad_pct,
            "radiation_n": sum(p["radiation_received"] for p in cohort),
            "targeted_pct": tgt_pct,
            "targeted_n": sum(p["targeted_therapy"] for p in cohort),
            "relapse_pct": rel_pct,
            "gtr_pct": gtr_pct,
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
            "surveillance_key": gdef["surveillance_key"],
            "key_distinctions": gdef["key_distinctions"],
            "top_tumor_types": top_tumors,
            "top_variants": top_variants,
            "top_treatments": top_treats,
            "treatment_protocols": TREATMENT_PROTOCOLS[gdef["gene"]],
            "surveillance_protocols": SURVEILLANCE_PROTOCOLS[gdef["gene"]],
            "patients": [
                {
                    "gene": p["gene"],
                    "age_dx": p["age_dx"],
                    "inrg_risk": p["inrg_risk"],
                    "histology": p["histology"],
                    "treatment": p["treatment"],
                    "response": p["response"],
                    "radiation_received": p["radiation_received"],
                    "targeted_therapy": p["targeted_therapy"],
                    "relapse": p["relapse"],
                    "gtr_resection": p["gtr_resection"],
                    "variant": p["variant"],
                }
                for p in cohort[:10]
            ],
        })

    return _json_safe({
        "atlas": "Hereditary-Neuroblastoma-Predisposition-Atlas",
        "breakdown": breakdown,
    })


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Neuroblastoma-Predisposition-Atlas."""
    definitions = {}
    for gdef in ATLAS_GENES:
        g = gdef["gene"]
        definitions[g] = {
            "gene": g,
            "locus": gdef["locus"],
            "protein_size": gdef["protein_size"],
            "inheritance": gdef["inheritance"],
            "cancer_risk": gdef["cancer_risk"],
            "pathognomonic": gdef["pathognomonic"],
            "surveillance_key": gdef["surveillance_key"],
            "key_distinctions": gdef["key_distinctions"],
            "pathogenic_variants": PATHOGENIC_VARIANTS[g],
            "tumor_types": TUMOR_TYPES[g],
            "treatment_protocols": TREATMENT_PROTOCOLS[g],
            "surveillance_protocols": SURVEILLANCE_PROTOCOLS[g],
        }

    return _json_safe({
        "atlas": "Hereditary-Neuroblastoma-Predisposition-Atlas",
        "definitions": definitions,
        "key_rules": {
            "ALK_FAMILIAL_NB_CRIZOTINIB": (
                "ALK germline GOF = #1 hereditary NB gene (75-80% familial NB cases); "
                "R1275Q (~40%) intermediate aggressiveness; F1174L (~25%) MOST AGGRESSIVE + MYCN co-amplification; "
                "Crizotinib FDA 2022 pediatric ALK-positive NB (relapsed/refractory); "
                "Lorlatinib 3rd-gen: F1174L crizotinib-resistant allele -- lorlatinib preferred; "
                "Panel sequencing MANDATORY: germline vs somatic ALK interpretation critical"
            ),
            "PHOX2B_CCHS_NPARM_ANESTHESIA": (
                "CCHS (Ondine's curse) PATHOGNOMONIC PHOX2B germline; "
                "NPARMs (frameshift/missense PHOX2B): NB 50% risk PATHOGNOMONIC -- highest NB risk in CCHS; "
                "Polyalanine 20/27: NB 5-10%; 20/28-20/33: NB 15-20% + cardiac dysrhythmia; "
                "ANESTHESIA HIGH RISK: respiratory centre failure -- NIV mandatory; "
                "Tumor-only PHOX2B = somatic in sporadic NB -- DO NOT diagnose CCHS from tumor tissue"
            ),
            "NF1_AVOID_RADIATION_SELUMETINIB": (
                "NF1 NB: 2-3x elevated risk; AVOID RADIATION (secondary MPNST); "
                "Selumetinib FDA 2020: symptomatic inoperable plexiform NF ≥2yr (MEK inhibitor); "
                "MPNST transformation 8-13% lifetime PATHOGNOMONIC (plexiform NF); "
                "JMML 200x elevated NF1 children: HSCT curative; "
                "Cafe-au-lait macules ≥6 ≥15mm PATHOGNOMONIC NF1 diagnosis"
            ),
            "TP53_LFS_AVOID_RADIATION_ABSOLUTELY": (
                "AVOID RADIATION ABSOLUTELY if TP53 germline LFS -- omit NB involved-field RT; "
                "Tandem HDCT/ASCT preferred over TBI (total body irradiation) in LFS; "
                "WBMRI Toronto Protocol annually NOT CT/PET; "
                "R337H: Brazilian founder mutation 1/300 carrier; "
                "ALT pathway at NB relapse: ATRX/DAXX + TP53 somatic co-mutation"
            ),
            "DICER1_PPB_SIBLINGS_CT_CHEST": (
                "PPB Type I (cystic lung <2yr) PATHOGNOMONIC DICER1 -- sibling chest CT ALL <8yr MANDATORY; "
                "Cervical ERMS (adolescent female) PATHOGNOMONIC DICER1; "
                "NB 2-4x elevated DICER1 syndrome; "
                "AVOID radiation children (developing organs); "
                "RNase IIIb hotspot (E1705/D1709/E1813) somatic second hit PATHOGNOMONIC"
            ),
            "BRCA2_FA_D1_SIBLING_DONOR_EXCLUSION": (
                "FA-D1 (biallelic BRCA2): bilateral Wilms PATHOGNOMONIC; medulloblastoma PATHOGNOMONIC; "
                "DEB/MMC chromosomal fragility PATHOGNOMONIC FA diagnosis -- ALL potential donors tested; "
                "SIBLING DONOR EXCLUSION MANDATORY: all siblings tested before HSCT evaluation; "
                "AVOID alkylating agents (cyclophosphamide/ifosfamide) in FA-D1: severe BMF; "
                "Cisplatin preferred (HRD sensitivity); olaparib PARP inhibitor in BRCA2-deficient tumours"
            ),
            "CASCADE_NEUROBLASTOMA_PREDISPOSITION": (
                "ALK/PHOX2B/NF1/TP53/DICER1: AD -- 50% risk first-degree relatives -- germline panel cascade; "
                "KIF1B: AD LOF -- 50% first-degree risk; 1p36 FISH/MLPA on tumour (standard staging); "
                "BARD1: AD LOF -- cascade testing in NB families; C557S common variant (individual risk moderate); "
                "BRCA2 FA-D1: BOTH parents obligate heterozygous BRCA2; siblings 25% FA-D1 + 50% HBOC; "
                "DEB/MMC ALL siblings MANDATORY pre-HSCT donor evaluation"
            ),
        },
        "cascade_testing_rule": (
            "ALK/NF1/TP53/DICER1: AD GOF/LOF -- 50% risk first-degree relatives -- germline panel testing. "
            "PHOX2B: AD -- 50% risk (PARMs) or mostly de novo (NPARMs ~90%); all children of CCHS parent tested. "
            "BARD1/KIF1B: AD LOF -- cascade NB family testing; C557S population screening not recommended. "
            "BRCA2 FA-D1: both parents obligate BRCA2 het (HBOC counselling); "
            "siblings 25% FA-D1 + 50% HBOC carrier; DEB/MMC ALL siblings before HSCT MANDATORY. "
            "Familial NB (≥2 first-degree relatives, bilateral/multifocal): ALK panel first then full germline NB panel."
        ),
    })


def _json_safe(obj):
    """Ensure all values are JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2)[:3000])
    print("\n--- breakdown (first gene) ---")
    bd = generate_breakdown()
    print(json.dumps(bd["breakdown"][0], indent=2)[:2000])
