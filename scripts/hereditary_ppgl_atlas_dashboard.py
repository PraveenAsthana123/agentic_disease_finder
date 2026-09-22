#!/usr/bin/env python3
"""Hereditary-Pheochromocytoma-Paraganglioma-Atlas — Complete 8-Gene PPGL Atlas
VHL     (Von Hippel-Lindau tumour suppressor; 213 aa; 3p25.3; AD LOF;
         VHL disease — PPGL + clear cell RCC + CNS/retinal hemangioblastoma;
         biallelic somatic loss required; clear cell RCC almost always VHL; seed SEED_BASE+0) ·
SDHB    (succinate dehydrogenase complex iron sulphur subunit B; 280 aa; 1p36.13; AD LOF;
         PPGL2 — HIGHEST malignancy risk ~40%; extra-adrenal + head-neck; succinate 2HG epigenetic; seed SEED_BASE+1) ·
SDHD    (succinate dehydrogenase complex subunit D; 159 aa; 11q23.1; AD LOF maternal imprinting;
         PPGL1 — head-neck paraganglioma predominant; PATERNAL transmission only clinically relevant; seed SEED_BASE+2) ·
SDHA    (succinate dehydrogenase complex flavoprotein subunit A; 664 aa; 5p15.33; AD LOF;
         PPGL5 + GIST + pituitary adenoma; catalytic subunit; lowest penetrance SDHx; seed SEED_BASE+3) ·
SDHC    (succinate dehydrogenase complex subunit C; 169 aa; 1q23.3; AD LOF;
         PPGL3 — mainly head-neck; lower malignancy than SDHB; integral membrane subunit; seed SEED_BASE+4) ·
SDHAF2  (succinate dehydrogenase complex assembly factor 2; 166 aa; 11q13.1; AD LOF paternal imprinting;
         PGL2 — exclusively head-neck paraganglioma; ultra-rare; paternal imprinting same as SDHD; seed SEED_BASE+5) ·
RET     (ret proto-oncogene; 1114 aa; 10q11.21; AD GOF;
         MEN2A/MEN2B/FMTC — MTC + pheochromocytoma + parathyroid; C634 highest PHEO risk;
         prophylactic thyroidectomy timing by codon; M918T MEN2B neonatal thyroidectomy; seed SEED_BASE+6) ·
MAX     (MYC associated factor X; 160 aa; 14q23.3; AD LOF paternal imprinting;
         bilateral adrenal pheochromocytoma; young males; paternal imprinting same as SDHD; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2982-2989)
"""
import random

SEED_BASE = 2982

ATLAS_GENES = [
    {
        "gene": "VHL",
        "protein": (
            "VHL -- 3p25.3 AD LOF -- 213aa -- Von-Hippel-Lindau-Tumour-Suppressor-"
            "24kDa-E3-Ubiquitin-Ligase-Adaptor-HIF1a-Ubiquitination-"
            "VHL-Disease-PPGL-ClearCellRCC-CNS-Retinal-Hemangioblastoma-OMIM-193300"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "213 aa / 24 kDa (VHL — von Hippel-Lindau tumour suppressor protein; "
            "FUNCTION: substrate recognition subunit of Cullin2-RING E3 ubiquitin ligase complex; "
            "  Targets HIF-1α (hypoxia-inducible factor 1α) for ubiquitin-mediated proteasomal degradation; "
            "  Under normoxia: prolyl hydroxylase (PHD) hydroxylates HIF-1α → VHL binds → ubiquitination → degradation; "
            "  Under hypoxia (or VHL LOF): HIF-1α not hydroxylated → not bound by VHL → HIF-1α accumulates; "
            "  HIF-1α target genes: VEGF, PDGF, EPO, GLUT1, CAIX — explaining VHL tumour angiogenesis and metabolism; "
            "VHL DISEASE (germline LOF + somatic second hit — biallelic loss required): "
            "  Type 1 VHL (no PHEO): large deletions/truncating → VHL1 disease; "
            "    Clear cell RCC: 25-45% lifetime risk; often bilateral/multifocal; "
            "    CNS hemangioblastoma: cerebellum, brainstem, spinal cord — 60-80% lifetime; "
            "    Retinal hemangioblastoma: 25-60%; can cause blindness; "
            "    Endolymphatic sac tumour: hearing loss; "
            "    Pancreatic cysts/NETs; "
            "  Type 2 VHL (with PHEO): missense mutations → PHEO risk + above; "
            "    Type 2A: PHEO + hemangioblastoma, low RCC risk; "
            "    Type 2B: PHEO + hemangioblastoma + HIGH RCC risk; "
            "    Type 2C: PHEO ONLY (Chuvash polycythaemia — R200W — causes erythrocytosis); "
            "encoded 3p25.3"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF + BIALLELIC SOMATIC SECOND HIT (Two-Hit Model): "
            "  GERMLINE: heterozygous LOF inherited → all cells haploinsufficient; "
            "  SOMATIC: second-hit loss of remaining allele in target tissue → tumour initiation; "
            "  AD INHERITANCE: 80-90% penetrance by age 65; de novo in 20%; "
            "  PHEO PHENOTYPE (VHL Type 2): "
            "    Usually bilateral/multifocal adrenal pheochromocytoma; "
            "    Often biochemically norepinephrine-dominant (extra-adrenal also norepinephrine); "
            "    Mean age at PHEO diagnosis: 30 years (younger than sporadic); "
            "    Malignancy rate: ~5% (lower than SDHB); "
            "  SURVEILLANCE PROTOCOL: "
            "    Annual: 24h urine/plasma metanephrines + catecholamines; "
            "    Annual MRI abdomen/pelvis; "
            "    Annual ophthalmology; "
            "    MRI brain/spine every 2 years; "
            "    Renal USS every 12 months (RCC); "
            "  GENETIC TESTING: "
            "    VHL sequencing + MLPA (large deletions in 20%); "
            "    MLPA MANDATORY even if sequencing negative"
        ),
        "disease_category": (
            "VHL-DISEASE-PPGL-CLEAR-CELL-RCC-HEMANGIOBLASTOMA: "
            "  BIOCHEMISTRY: plasma normetanephrine (norepinephrine-secreting); "
            "    CAIX positive on immunostaining; "
            "  TREATMENT: "
            "    PHEO/PGL: surgical resection; alpha-blockade pre-op; "
            "    RCC: partial nephrectomy preferred; sunitinib/belzutifan for advanced; "
            "    CNS hemangioblastoma: stereotactic radiosurgery / excision when growing; "
            "    Retinal: laser/anti-VEGF; "
            "  GENETIC TESTING: "
            "    VHL sequencing + MLPA; "
            "    Test first-degree relatives; "
            "    Somatic VHL: found in >90% clear cell RCC (even sporadic)"
        ),
        "disease_pathway": (
            "VHL LOF → HIF-1a ACCUMULATION → PSEUDOHYPOXIA → TUMOUR ANGIOGENESIS: "
            "  Normal: PHD hydroxylates Pro402/Pro564 of HIF-1α; "
            "    VHL-Elongin-B-C-CUL2-RBX1 complex recognises hydroxy-Pro; "
            "    HIF-1α polyubiquitinated → 26S proteasome degradation; "
            "  VHL LOF: HIF-1α not captured → accumulates; "
            "    HIF-1α dimerises with ARNT (HIF-1β) → transcription factor; "
            "    Activates HRE (hypoxia-response elements) target genes: "
            "      VEGF → angiogenesis → hemangioblastoma vascularisation; "
            "      EPO → erythrocytosis (Chuvash R200W); "
            "      GLUT1/PDK1 → Warburg metabolism; "
            "      PDGF-B → PDGFR → pericyte recruitment; "
            "      CAIX → acidic microenvironment; "
            "  Clear cell RCC almost always: biallelic VHL loss; "
            "  HIF target: miR-210 → SDH4 suppressed → secondary SDH dysfunction"
        ),
    },
    {
        "gene": "SDHB",
        "protein": (
            "SDHB -- 1p36.13 AD LOF -- 280aa -- Succinate-Dehydrogenase-Complex-"
            "Iron-Sulphur-Subunit-B-32kDa-3-Fe-S-Clusters-Electron-Relay-"
            "PPGL2-Highest-Malignancy-40pct-Extra-Adrenal-Succinate-2HG-Epigenetic-OMIM-185470"
        ),
        "locus": "1p36.13",
        "protein_size": (
            "280 aa / 32 kDa (SDHB — succinate dehydrogenase complex iron-sulphur subunit B; "
            "FUNCTION: electron relay subunit of SDH (Complex II, mitochondrial); "
            "  Contains 3 iron-sulphur clusters: [2Fe-2S], [4Fe-4S], [3Fe-4S]; "
            "  Transfers electrons from succinate oxidation (SDHA) to ubiquinone (SDHC/D anchor); "
            "  SDH = only enzyme in both TCA cycle (Complex II) and electron transport chain; "
            "  SDHA oxidises succinate → fumarate + FADH2 → electrons transferred via SDHB clusters → ubiquinone; "
            "SDHx LOF MECHANISM (all SDH subunits): "
            "  Succinate accumulates (cannot be oxidised to fumarate); "
            "  Succinate competitively inhibits α-ketoglutarate-dependent dioxygenases: "
            "    PHD enzymes (prolyl hydroxylases): cannot hydroxylate HIF-1α → pseudohypoxia; "
            "    TET enzymes (ten-eleven translocation): cannot demethylate DNA → DNA hypermethylation; "
            "    KDM (lysine demethylases): histone demethylation blocked → histone hypermethylation; "
            "  Result: pseudohypoxia + global epigenetic reprogramming (CIMP phenotype); "
            "SDHB CONSEQUENCE (PPGL2): "
            "  Extra-adrenal paraganglioma + head-neck PGL + adrenal PHEO; "
            "  HIGHEST malignancy risk of all PPGL genes: ~40% (vs SDHD 5%, VHL 5%); "
            "  Young age at presentation; often multifocal; "
            "  Succinate elevated in plasma/urine: marker; "
            "  encoded 1p36.13"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — SDHB / PPGL2: "
            "  PHENOTYPE: "
            "    Extra-adrenal paraganglioma (abdominal, thoracic): MOST CHARACTERISTIC; "
            "    Head-neck paraganglioma (carotid body, jugulotympanic, vagal); "
            "    Adrenal pheochromocytoma; "
            "    MALIGNANCY: 40% lifetime (highest in all hereditary PPGL); "
            "    Metastases: lymph nodes, liver, lung, bone; "
            "    Young onset: often before age 40; "
            "  BIOCHEMISTRY: "
            "    Normetanephrine dominant (norepinephrine-secreting): extra-adrenal pattern; "
            "    Plasma succinate: elevated (tumour metabolite); "
            "    Dopamine excess: some non-functioning PGLs (dopamine → normetanephrine); "
            "    SDHB IHC: ABSENT staining (loss of SDHB protein) — diagnostic on tumour tissue; "
            "  SURVEILLANCE (INTENSIVE given malignancy risk): "
            "    Annual plasma/urine metanephrines + succinate; "
            "    MRI skull base to pelvis every 1-2 years; "
            "    18F-DOPA PET: chromaffin tissue; 68Ga-DOTATATE PET: somatostatin receptors; "
            "    Whole-body MRI for metastatic disease; "
            "  TREATMENT: "
            "    Surgical resection first-line; "
            "    Malignant: PRRT (177Lu-DOTATATE); sunitinib; temozolomide/capecitabine (TMZ/CAP); "
            "    Cabozantinib for SDHB-mutated malignant PPGL (clinical trials)"
        ),
        "disease_category": (
            "SDHB-PPGL2-HIGHEST-MALIGNANCY-40pct-EXTRA-ADRENAL: "
            "  KEY RULE: any paraganglioma in young patient → test SDHB first; "
            "  DIAGNOSIS: SDHB IHC loss on tumour; germline SDHB sequencing + MLPA; "
            "  MALIGNANCY PREDICTOR: SDHB mutation + extra-adrenal location → highest risk; "
            "  FIRST-LINE IMAGING: 18F-FDG PET for malignant PPGL; 68Ga-DOTATATE for well-differentiated; "
            "  GENETIC TESTING: panel VHL-SDHB-SDHC-SDHD-SDHA-SDHAF2-RET-NF1-MAX; "
            "  PATHOLOGY: SDHB IHC absent = SDHx mutation (any subunit) — granular cytoplasmic signal lost"
        ),
        "disease_pathway": (
            "SDHB LOF → SUCCINATE ACCUMULATION → PSEUDOHYPOXIA + CIMP EPIGENOME: "
            "  SDHB LOF → SDH complex destabilised (all 4 subunits require each other); "
            "  Succinate → cannot be converted to fumarate → accumulates in matrix + cytoplasm; "
            "  Succinate exported → circulates in plasma (biomarker); "
            "  PHD inhibition: HIF-1α not hydroxylated → VHL cannot bind → HIF-1α stable → VEGF/EPO; "
            "  TET inhibition: 5-methylcytosine cannot be demethylated → CpG island hypermethylation; "
            "    Tumour suppressor promoters silenced → CpG island methylator phenotype (CIMP); "
            "  KDM inhibition: H3K4/K9/K27 hypermethylated → chromatin compaction; "
            "  NET: pseudohypoxia signal + epigenetic silencing → paraganglioma initiation"
        ),
    },
    {
        "gene": "SDHD",
        "protein": (
            "SDHD -- 11q23.1 AD LOF Maternal-Imprinting -- 159aa -- Succinate-Dehydrogenase-Complex-"
            "Subunit-D-17kDa-2-TM-Helices-Ubiquinone-Binding-"
            "PPGL1-Head-Neck-Paraganglioma-Paternal-Transmission-Only-OMIM-115310"
        ),
        "locus": "11q23.1",
        "protein_size": (
            "159 aa / 17 kDa (SDHD — succinate dehydrogenase complex subunit D; "
            "FUNCTION: small integral membrane subunit anchoring SDH to inner mitochondrial membrane; "
            "  Contains 2 transmembrane helices; provides ubiquinone-binding pocket; "
            "  Together with SDHC forms the 'anchor' that transfers electrons to ubiquinone; "
            "  Essential for SDH complex stability; without SDHD → entire complex disassembles; "
            "MATERNAL IMPRINTING (parent-of-origin effect): "
            "  SDHD gene lies within imprinted region 11q23; "
            "  Maternal allele imprinted (silenced) — only PATERNAL copy expressed; "
            "  CONSEQUENCE: "
            "    Paternal SDHD mutation → patient has LOF SDHD (maternal allele already silenced); "
            "    Maternal SDHD mutation → patient has NORMAL SDHD (paternal allele intact); "
            "    CLINICAL RULE: ONLY paternal SDHD mutation causes PPGL1; "
            "    Maternal mutation carriers do NOT develop disease (but can transmit paternal allele); "
            "    Cascade testing MUST assess parent of origin; "
            "PPGL1 DISEASE: "
            "  Head-neck paraganglioma predominantly (carotid body, jugulotympanic, vagal); "
            "  Adrenal pheochromocytoma less common; "
            "  Malignancy rate: ~5% (much lower than SDHB); "
            "  Multifocal head-neck PGL: 50% of cases; "
            "  encoded 11q23.1"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — MATERNAL IMPRINTING — SDHD / PPGL1: "
            "  CRITICAL RULE: ONLY paternal SDHD mutation causes disease; "
            "  PEDIGREE PATTERN: "
            "    Affected father → 50% of children at risk (50% inherit mutation); "
            "      Of those who inherit: ALL with paternally inherited mutation affected (high penetrance); "
            "    Carrier mother → 0% of children affected (maternal SDHD silenced); "
            "      BUT children inherit mutation → their CHILDREN at risk (paternally transmitted again); "
            "  PHENOTYPE: "
            "    Head-neck paraganglioma: carotid body (common), jugulotympanic, vagal; "
            "    Often BILATERAL and MULTIFOCAL (50%); "
            "    Presenting symptoms: pulsatile tinnitus, hearing loss, neck mass, cranial nerve palsies; "
            "    Most non-functional (don't secrete catecholamines); "
            "    Adrenal PHEO: ~10-20% (less than SDHB); "
            "  BIOCHEMISTRY: "
            "    Majority biochemically silent (non-secreting head-neck PGL); "
            "    Functioning: dopamine or norepinephrine; "
            "    Plasma/urine methoxytyramine (dopamine metabolite): useful for non-secreting; "
            "  SURVEILLANCE: "
            "    Annual plasma metanephrines + methoxytyramine; "
            "    MRI skull base to pelvis every 2 years; "
            "    68Ga-DOTATATE PET: best for head-neck PGL"
        ),
        "disease_category": (
            "SDHD-PPGL1-HEAD-NECK-PATERNAL-IMPRINTING-MULTIFOCAL: "
            "  CRITICAL DIAGNOSTIC RULE: ask about father's side; maternal SDHD = clinically silent; "
            "  HEAD-NECK IMAGING: MRI neck + skull base (not CT — avoid radiation in young patients); "
            "  FUNCTIONAL IMAGING: 68Ga-DOTATATE PET-CT for multicentric disease; "
            "  GENETIC COUNSELLING MANDATORY: imprinting explanation complex; "
            "  TREATMENT: "
            "    Observe small stable head-neck PGL; "
            "    Surgery: curative for localised; high cranial nerve risk for skull base; "
            "    Stereotactic radiotherapy: for inaccessible/growing skull base PGL; "
            "    PRRT (177Lu-DOTATATE): for progressive/malignant"
        ),
        "disease_pathway": (
            "SDHD LOF → SDH COMPLEX DESTABILISATION → SAME SUCCINATE ACCUMULATION PATH: "
            "  SDHD anchor subunit absent → SDH complex (I-II-III-IV chain) cannot dock at membrane; "
            "  SDHA/SDHB/SDHC subunits unstable without SDHD → degraded → whole complex absent; "
            "  Succinate accumulates → PHD inhibition → HIF pseudohypoxia; "
            "  TET inhibition → CIMP; "
            "  SDHB IHC: granular cytoplasmic staining ABSENT (all SDHx mutations lose SDHB IHC); "
            "  SDHD IHC: NOT used diagnostically (more variable than SDHB IHC); "
            "  HEAD-NECK PREDILECTION of SDHD: "
            "    Carotid body cells are O2-sensing paraganglia; "
            "    Constitutively active HIF pathway mimics chronic hypoxia → chemoreceptor expansion; "
            "    High O2 tension in carotid body → HIF more active when PHD/VHL impaired"
        ),
    },
    {
        "gene": "SDHA",
        "protein": (
            "SDHA -- 5p15.33 AD LOF -- 664aa -- Succinate-Dehydrogenase-Complex-"
            "Flavoprotein-Subunit-A-73kDa-FAD-Binding-Succinate-Oxidase-Catalytic-"
            "PPGL5-GIST-Pituitary-Adenoma-Lowest-Penetrance-SDHx-OMIM-600857"
        ),
        "locus": "5p15.33",
        "protein_size": (
            "664 aa / 73 kDa (SDHA — succinate dehydrogenase complex flavoprotein subunit A; "
            "FUNCTION: catalytic subunit of SDH (Complex II); "
            "  Contains FAD (flavin adenine dinucleotide) covalently bound via His99; "
            "  Oxidises succinate → fumarate: succinate + FAD → fumarate + FADH2; "
            "  FADH2 electrons transferred to SDHB iron-sulphur clusters → ubiquinone; "
            "  SDHA is the CATALYTIC SUBUNIT — without it no succinate oxidation occurs; "
            "  SDHA also has a structural role in TCA cycle; "
            "SDHA GERMLINE LOF (PPGL5): "
            "  Lowest penetrance of all SDHx genes (estimated 10-20% lifetime risk); "
            "  PPGL5 designation; "
            "  Associated with GIST (gastrointestinal stromal tumour) — SDH-deficient GIST; "
            "    SDH-deficient GIST: epigastric/stomach; young patients; imatinib-RESISTANT; "
            "    SDH-deficient GIST = biallelic SDHx loss (SDHA most common in this context); "
            "  Pituitary adenoma: first SDHx gene linked to pituitary tumours; "
            "  Renal cell carcinoma (rare); "
            "SDHA IHC (UNIQUE among SDHx): "
            "  SDHA IHC absent ONLY when SDHA itself is mutated; "
            "  SDHB IHC absent = any SDHx mutation (non-specific); "
            "  SDHA IHC POSITIVE + SDHB IHC absent → SDHB/SDHC/SDHD mutation (not SDHA); "
            "  SDHA IHC ABSENT → SDHA mutation confirmed; "
            "encoded 5p15.33"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — SDHA / PPGL5 + GIST: "
            "  PHENOTYPE: "
            "    PPGL: adrenal + extra-adrenal + head-neck (lower frequency than SDHB/SDHD); "
            "    SDH-DEFICIENT GIST: "
            "      Stomach predominant; young patients (< 40 years); "
            "      SDH-deficient (SDHB IHC absent + SDHA IHC absent if SDHA mutated); "
            "      IMATINIB-RESISTANT — does not respond to standard GIST therapy; "
            "      Treatment: sunitinib; temozolomide; everolimus; "
            "    Pituitary adenoma (usually non-functioning); "
            "    Penetrance LOW: ~10-20% (important for counselling — most carriers unaffected); "
            "  BIOCHEMISTRY: "
            "    Functioning PPGL: norepinephrine/epinephrine secreting; "
            "    Non-functioning: methoxytyramine; "
            "    SDH-deficient GIST: c-KIT/PDGFRA wild-type (biallelic SDH loss drives tumour); "
            "  SDHA IHC KEY RULE: "
            "    ABSENT SDHA IHC + absent SDHB IHC → SDHA germline mutation until proven otherwise; "
            "    PRESENT SDHA IHC + absent SDHB IHC → SDHB/SDHC/SDHD/SDHAF2 mutation; "
            "  SURVEILLANCE: "
            "    Annual plasma metanephrines; "
            "    Upper GI endoscopy every 3 years (GIST surveillance); "
            "    MRI pituitary every 3-5 years"
        ),
        "disease_category": (
            "SDHA-PPGL5-GIST-PITUITARY-LOWEST-PENETRANCE-SDHx: "
            "  IMATINIB-RESISTANT GIST → test SDHA/SDHB IHC; "
            "  SDHA IHC ABSENT = SDHA germline mutation specific (vs other SDHx); "
            "  LOW PENETRANCE: appropriate counselling → not all carriers develop cancer; "
            "  GENETIC TESTING: "
            "    SDHA sequencing + MLPA; "
            "    SDH panel (SDHB/SDHC/SDHD/SDHA) for all PPGL + all GIST in <40 years"
        ),
        "disease_pathway": (
            "SDHA LOF → CATALYTIC BLOCK → SUCCINATE ACCUMULATES → SDH-DEFICIENT TUMOURIGENESIS: "
            "  SDHA absent → SDH catalytic activity abolished; "
            "  Succinate cannot be oxidised → mitochondrial succinate export to cytoplasm; "
            "  Cytoplasmic succinate: PHD2 inhibition → HIF-1α → VEGF (pseudohypoxia); "
            "  TET2 inhibition → 5mC cannot be oxidised → CpG hypermethylation (CIMP); "
            "  SDH-deficient GIST: "
            "    c-KIT and PDGFRA usually wild-type (no conventional GIST driver); "
            "    Biallelic SDHA loss drives GIST via succinate-epigenetic mechanism; "
            "    Imatinib targets KIT/PDGFRA tyrosine kinase → ineffective without these drivers; "
            "  Second hit in SDHA: often somatic loss of heterozygosity (LOH) at 5p15.33"
        ),
    },
    {
        "gene": "SDHC",
        "protein": (
            "SDHC -- 1q23.3 AD LOF -- 169aa -- Succinate-Dehydrogenase-Complex-"
            "Subunit-C-18kDa-1-TM-Helix-Integral-Membrane-Ubiquinone-Stabilisation-"
            "PPGL3-Head-Neck-Low-Malignancy-Iron-Sulphur-Stability-OMIM-602413"
        ),
        "locus": "1q23.3",
        "protein_size": (
            "169 aa / 18 kDa (SDHC — succinate dehydrogenase complex subunit C; "
            "FUNCTION: integral membrane subunit of SDH complex; "
            "  Contains 1 transmembrane helix; works with SDHD to anchor SDH at inner mitochondrial membrane; "
            "  Coordinates a haem b group (between SDHC and SDHD transmembrane domains); "
            "  Haem b: does not participate in electron transfer but stabilises the membrane anchor; "
            "  SDHC maintains structural integrity of the SDHB iron-sulphur module; "
            "  Without SDHC: SDH complex disassembles; SDHB IHC lost; "
            "PPGL3 DISEASE (SDHC LOF): "
            "  Head-neck paraganglioma predominantly (carotid body, jugulotympanic, vagal); "
            "  Similar to SDHD but WITHOUT maternal imprinting; "
            "  Adrenal pheochromocytoma less common; "
            "  MALIGNANCY RATE: low (~2-5%) — lower than SDHB; "
            "  Less multifocal than SDHD; "
            "  Often presents as incidental neck mass or pulsatile tinnitus; "
            "encoded 1q23.3"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — SDHC / PPGL3: "
            "  NO IMPRINTING (unlike SDHD and SDHAF2): "
            "    Both maternal and paternal SDHC mutations cause disease; "
            "    IMPORTANT DISTINCTION from SDHD: full penetrance regardless of parent of origin; "
            "  PHENOTYPE: "
            "    Head-neck paraganglioma: carotid body, jugulotympanic, vagal; "
            "    Less multifocal than SDHD; "
            "    Adrenal pheochromocytoma: ~15-20%; "
            "    Malignancy rare (<5%); "
            "    Presentations: neck mass, pulsatile tinnitus, conductive hearing loss; "
            "  BIOCHEMISTRY: "
            "    Mostly non-functional (no catecholamine secretion); "
            "    Functioning: dopamine or norepinephrine; "
            "    SDHB IHC: absent (as for all SDHx); "
            "    SDHC IHC: not routinely used; "
            "  SURVEILLANCE: "
            "    Annual plasma metanephrines; "
            "    MRI neck/skull base every 2-3 years; "
            "    68Ga-DOTATATE PET for multicentric survey; "
            "  TREATMENT: "
            "    Head-neck PGL: observation if slow-growing; surgery or stereotactic RT; "
            "    Adrenal PHEO: laparoscopic adrenalectomy after alpha-blockade"
        ),
        "disease_category": (
            "SDHC-PPGL3-HEAD-NECK-LOW-MALIGNANCY-NO-IMPRINTING: "
            "  DISTINGUISHING FROM SDHD: "
            "    SDHC = no imprinting (both parent transmissions active); "
            "    SDHD = maternal imprinting (only paternal transmission active); "
            "    Both = head-neck PGL predominantly; "
            "  IMAGING: MRI neck + skull base first-line; "
            "  GENETIC TESTING: SDHC sequencing + MLPA; panel preferred"
        ),
        "disease_pathway": (
            "SDHC LOF → MEMBRANE ANCHOR LOSS → SDH COMPLEX DISASSEMBLY → SUCCINATE ACCUMULATION: "
            "  SDHC absent → SDHD cannot form proper membrane anchor; "
            "  Whole SDH complex (SDHA/SDHB/SDHC/SDHD) disassembles; "
            "  SDH activity abolished → succinate accumulates; "
            "  Same downstream pathway: PHD inhibition → HIF pseudohypoxia; "
            "  TET inhibition → CIMP phenotype; "
            "  SDHB IHC absent on tumour (indirect marker of any SDHx dysfunction); "
            "  Head-neck predilection: same oxygen-sensing paraganglia mechanism as SDHD"
        ),
    },
    {
        "gene": "SDHAF2",
        "protein": (
            "SDHAF2 -- 11q13.1 AD LOF Paternal-Imprinting -- 166aa -- Succinate-Dehydrogenase-"
            "Assembly-Factor-2-18kDa-FAD-Attachment-SDHA-Covalent-Cofactor-"
            "PGL2-Exclusively-Head-Neck-Ultra-Rare-Paternal-Imprinting-OMIM-613019"
        ),
        "locus": "11q13.1",
        "protein_size": (
            "166 aa / 18 kDa (SDHAF2 — succinate dehydrogenase complex assembly factor 2; "
            "FUNCTION: assembly factor for SDH complex biogenesis; "
            "  Enables covalent attachment of FAD cofactor to His99 of SDHA; "
            "  Without SDHAF2: FAD not covalently incorporated into SDHA → SDHA non-functional; "
            "  SDHA without FAD → entire SDH complex cannot assemble; "
            "  SDHAF2 required only during assembly (not present in mature complex); "
            "  Related to yeast Sdh5 (first identified in Saccharomyces cerevisiae); "
            "PATERNAL IMPRINTING (same mechanism as SDHD): "
            "  SDHAF2 lies near SDHD on 11q; "
            "  Maternal SDHAF2 allele: imprinted/silenced; "
            "  Only PATERNAL SDHAF2 mutation causes disease; "
            "  Same cascade counselling principles as SDHD; "
            "PGL2 DISEASE: "
            "  EXCLUSIVELY head-neck paraganglioma — no adrenal pheochromocytoma reported; "
            "  Ultra-rare: only a few families worldwide; "
            "  Original Dutch founder family (c.232G>T, p.Gly78Val — Westphal et al. 2007); "
            "  Early onset (second decade); multifocal head-neck PGL; "
            "  encoded 11q13.1"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — PATERNAL IMPRINTING — SDHAF2 / PGL2: "
            "  SAME IMPRINTING RULE AS SDHD: "
            "    ONLY paternal SDHAF2 mutation causes PGL2; "
            "    Maternal SDHAF2 mutation → carrier (silent); "
            "    Children of maternal carriers: at risk only if they transmit mutation paternally; "
            "  PHENOTYPE (ULTRA-RARE): "
            "    EXCLUSIVELY head-neck paraganglioma; "
            "    NO adrenal pheochromocytoma reported; "
            "    Onset: second decade; often multifocal; "
            "    Non-functioning (dopamine or silent); "
            "  FOUNDER MUTATION: "
            "    Dutch founder: c.232G>T (p.Gly78Val) in exon 2; "
            "    If Dutch/Scandinavian ancestry: test this variant first; "
            "  BIOCHEMISTRY: "
            "    Non-functioning predominantly; "
            "    Methoxytyramine elevated in some cases; "
            "    SDHB IHC: absent (indirect marker); "
            "  SURVEILLANCE: "
            "    Annual plasma metanephrines + methoxytyramine; "
            "    MRI skull base/neck every 2-3 years (no abdominal imaging required if exclusively HN); "
            "  GENETIC COUNSELLING: "
            "    Parent of origin MANDATORY — maternal SDHAF2 carriers reassured (not at risk); "
            "    Paternal carriers → annual surveillance from age 15"
        ),
        "disease_category": (
            "SDHAF2-PGL2-EXCLUSIVELY-HEAD-NECK-ULTRA-RARE-PATERNAL-IMPRINTING: "
            "  DISTINGUISHING FROM SDHD: "
            "    Both: paternal imprinting; both head-neck PGL; "
            "    SDHAF2: NO adrenal PHEO; SDHD: adrenal PHEO ~10-20%; "
            "  CLINICAL TIP: Dutch/Scandinavian ancestry + head-neck PGL + family history → test SDHAF2; "
            "  TREATMENT: same as SDHD head-neck PGL (observation vs surgery vs SBRT)"
        ),
        "disease_pathway": (
            "SDHAF2 LOF → FAD NOT ATTACHED TO SDHA → SDHA NON-FUNCTIONAL → SUCCINATE ACCUMULATES: "
            "  SDHAF2 absent → SDHA His99 cannot form covalent FAD bond; "
            "  Apo-SDHA (FAD-free): cannot oxidise succinate; "
            "  SDH complex assembly stalls; SDHB/SDHC/SDHD subunits degraded; "
            "  Same downstream: succinate → PHD inhibition → HIF pseudohypoxia; "
            "  TET inhibition → CIMP; "
            "  SDHB IHC absent on tumour tissue; "
            "  Unique biology: SDHAF2 is the ONLY SDHx gene not a structural subunit — assembly factor only"
        ),
    },
    {
        "gene": "RET",
        "protein": (
            "RET -- 10q11.21 AD GOF -- 1114aa -- RET-Proto-Oncogene-"
            "Receptor-Tyrosine-Kinase-120kDa-GDNF-Family-Ligand-Receptor-"
            "MEN2A-MEN2B-FMTC-MTC-Pheochromocytoma-Parathyroid-Prophylactic-Thyroidectomy-OMIM-171400"
        ),
        "locus": "10q11.21",
        "protein_size": (
            "1114 aa / 120 kDa (RET — ret proto-oncogene; receptor tyrosine kinase; "
            "FUNCTION: transmembrane RTK activated by GDNF family ligands (GDNF, NRTN, ARTN, PSPN) + co-receptor GFRα; "
            "  Activated RET dimerises → autophosphorylation → PI3K/AKT, MAPK/ERK, PLCγ signalling; "
            "  Essential for neural crest migration, kidney development, enteric nervous system; "
            "  GOF MUTATIONS: constitutive activation without ligand → oncogenic signalling; "
            "MUTATION-PHENOTYPE CORRELATION (MEN2A/2B risk stratification): "
            "  ATA-D (HIGHEST RISK — THYROIDECTOMY WITHIN 6 MONTHS / AGE 6 MONTHS): "
            "    M918T (exon 16): MEN2B — most aggressive; neonatal thyroidectomy in first months; "
            "  ATA-C (HIGH RISK — THYROIDECTOMY BY AGE 5): "
            "    C634F/Y/S/R/G/W (exon 11): MEN2A; PHEO 50%; PHPT 20%; marfanoid in MEN2B-like variants; "
            "  ATA-B (MODERATE RISK — THYROIDECTOMY BY AGE 5-10): "
            "    C609/C611/C618/C620 (exon 10): MEN2A; PHEO 30%; PHPT 20%; Hirschsprung association; "
            "    C630 (exon 11): MEN2A; PHEO 25%; "
            "  ATA-A (LOWEST RISK): "
            "    Other exon 13/14/15 mutations; lower penetrance; later surgery; "
            "encoded 10q11.21"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GOF — RET / MEN2A / MEN2B / FMTC: "
            "  MEN2A PHENOTYPE (most common — 75-80%): "
            "    Medullary thyroid carcinoma (MTC): 95% penetrance lifetime; C-cell hyperplasia → MTC; "
            "    Pheochromocytoma: 50% (bilateral adrenal; epinephrine-dominant); "
            "    Primary hyperparathyroidism (PHPT): 20-30% (multiglandular); "
            "    Codon 634 (C634): highest risk for all three components; "
            "  MEN2B PHENOTYPE (5% — most aggressive): "
            "    MTC: earliest onset (neonatal/infancy) — most aggressive; "
            "    PHEO: 50% bilateral; "
            "    Marfanoid habitus; mucosal neuromas (tongue, lips — PATHOGNOMONIC); "
            "    Medullated corneal nerve fibres (slit-lamp); "
            "    Intestinal ganglioneuromatosis; "
            "    MUTATION: M918T (>95% of MEN2B cases); "
            "    PROPHYLACTIC THYROIDECTOMY: within 6 months of life; "
            "  FMTC (Familial MTC alone): MTC without PHEO/PHPT; lower-risk mutations; "
            "  KEY RULE — EXCLUDE PHEO BEFORE THYROID SURGERY: "
            "    Screen plasma metanephrines/urine catecholamines BEFORE any neck surgery; "
            "    Operating on occult PHEO = hypertensive crisis → life-threatening; "
            "  BIOCHEMISTRY (MTC): "
            "    Calcitonin: elevated (tumour marker); pentagastrin/calcium stimulation test; "
            "    CEA: elevated in progressive MTC; "
            "    RET M918T: circulating tumour DNA monitoring"
        ),
        "disease_category": (
            "RET-MEN2A-MTC-PHEO-PHPT-PROPHYLACTIC-THYROIDECTOMY-BY-CODON: "
            "  KEY RULE 1: EXCLUDE PHEO BEFORE ANY RET NECK SURGERY — alpha-blockade first; "
            "  KEY RULE 2: M918T MEN2B → thyroidectomy within 6 months of life; "
            "  KEY RULE 3: codon 634 → thyroidectomy by age 5 (ATA-C); "
            "  DIAGNOSIS: calcitonin + CEA (MTC); plasma metanephrines (PHEO); calcium+PTH (PHPT); "
            "  TREATMENT: "
            "    MTC: total thyroidectomy + central neck dissection; "
            "    Advanced MTC: vandetanib (FDA 2011) or cabozantinib (FDA 2012); "
            "    PHEO: laparoscopic adrenalectomy after alpha-blockade (bilateral cortical-sparing); "
            "    PHPT: parathyroidectomy at time of thyroidectomy if 4D-CT/sestamibi confirmed"
        ),
        "disease_pathway": (
            "RET GOF → CONSTITUTIVE KINASE ACTIVATION → RAS-MAPK + PI3K-AKT → MTC INITIATION: "
            "  RET extracellular domain: cysteine residues (C634) normally form intramolecular disulphide; "
            "  C634 mutation: unpaired cysteine forms INTERMOLECULAR disulphide with other RET molecule; "
            "  Constitutive dimerisation → persistent kinase activation WITHOUT ligand; "
            "  M918T: kinase domain activation loop stabilised in active conformation; "
            "  Downstream: "
            "    RAS-RAF-MEK-ERK (MAPK): proliferation; "
            "    PI3K-AKT-mTOR: survival/metabolism; "
            "    PLCγ-PKC: migration/invasion; "
            "    STAT3: survival; "
            "  C-cell hyperplasia (precursor) → parafollicular C-cell transformation → MTC; "
            "  PHEO: chief cells of adrenal medulla (neural crest) same RET-driven transformation; "
            "  Vandetanib/cabozantinib: RET kinase inhibitors (ATA-classified)"
        ),
    },
    {
        "gene": "MAX",
        "protein": (
            "MAX -- 14q23.3 AD LOF Paternal-Imprinting -- 160aa -- MYC-Associated-Factor-X-"
            "18kDa-bHLHLZ-Myc-Network-MNT-MAD-Dimerisation-"
            "Bilateral-Adrenal-Pheochromocytoma-Young-Males-Paternal-Imprinting-OMIM-154950"
        ),
        "locus": "14q23.3",
        "protein_size": (
            "160 aa / 18 kDa (MAX — MYC associated factor X; "
            "FUNCTION: essential dimerisation partner in MYC network transcription factors; "
            "  Basic helix-loop-helix leucine zipper (bHLHLZ) protein; "
            "  MAX dimerises with: MYC (oncogenic), MAD/MNT (tumour suppressive); "
            "  MYC:MAX heterodimer activates target genes (E-box sequences): cell cycle, ribosome biogenesis; "
            "  MAD:MAX heterodimer represses same targets → antagonises MYC; "
            "  MAX homodimer: represses E-box targets (transcriptionally inactive); "
            "  BALANCE: MYC:MAX (pro-proliferative) vs MAD:MAX (anti-proliferative) determines cell fate; "
            "MAX LOF MECHANISM: "
            "  MAX absent → MYC cannot form functional heterodimer → MYC oncogenic activity impaired; "
            "  Paradox: MAX LOF actually increases MYC activity (indirect); "
            "  MAD/MNT cannot repress E-box targets without MAX → net pro-proliferative; "
            "  MAZ network dysregulated; "
            "PATERNAL IMPRINTING: "
            "  MAX lies within imprinted region at 14q23.3; "
            "  Only PATERNAL MAX mutation causes disease; "
            "  Similar to SDHD and SDHAF2 (all hereditary PPGL genes with imprinting); "
            "encoded 14q23.3"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — PATERNAL IMPRINTING — MAX / BILATERAL ADRENAL PHEO: "
            "  SAME IMPRINTING RULE: only paternal MAX mutation causes PPGL; "
            "  PHENOTYPE: "
            "    BILATERAL ADRENAL PHEOCHROMOCYTOMA: CHARACTERISTIC; "
            "    Young males predominantly (imprinting + sex-specific expression effects); "
            "    Often synchronous bilateral (not metachronous); "
            "    Catecholamine excess: epinephrine + norepinephrine (adrenal phenotype); "
            "    Malignancy risk: low (~5-10%); "
            "    Head-neck PGL: less common but described; "
            "  BIOCHEMISTRY: "
            "    Plasma metanephrine ELEVATED (epinephrine-secreting = adrenal origin); "
            "    Normetanephrine also elevated; "
            "    CT adrenals: bilateral adrenal masses in young patient → think MAX; "
            "  GENETIC COUNSELLING: "
            "    Parent of origin: maternal MAX = silent; paternal MAX = disease; "
            "    Young male with bilateral pheo + father affected → MAX highly likely; "
            "  SURVEILLANCE: "
            "    Annual plasma metanephrines (bilateral means both sides must be followed); "
            "    MRI adrenals annually; "
            "  TREATMENT: "
            "    Bilateral pheo: bilateral cortical-sparing adrenalectomy preferred; "
            "    Avoid total bilateral adrenalectomy (permanent Addison → lifelong steroid dependence); "
            "    Alpha-blockade (phenoxybenzamine 10-14 days) + volume loading before surgery; "
            "    Post-op adrenal insufficiency risk: monitor cortisol; steroid cover"
        ),
        "disease_category": (
            "MAX-BILATERAL-ADRENAL-PHEO-YOUNG-MALES-PATERNAL-IMPRINTING: "
            "  KEY RULE: young male + bilateral adrenal pheo → test MAX (+ VHL, SDHB, RET); "
            "  THREE IMPRINTED PPGL GENES: SDHD (11q23), SDHAF2 (11q13), MAX (14q23); "
            "  CORTICAL-SPARING SURGERY: avoid total bilateral adrenalectomy; "
            "  DIAGNOSIS: bilateral pheo on CT/MRI + plasma metanephrines elevated + germline testing; "
            "  GENETIC TESTING: "
            "    MAX sequencing; confirm parent of origin; "
            "    Somatic MAX mutations: also found in pheochromocytoma tumours (no germline)"
        ),
        "disease_pathway": (
            "MAX LOF → MYC NETWORK DEREGULATION → CHROMAFFIN CELL PROLIFERATION: "
            "  MAX LOF: cannot form MAD:MAX repressor complexes → E-box target genes de-repressed; "
            "  MYC target genes activated without MAX normally repressing them (indirect net effect); "
            "  E-box target genes: cyclin D2, CDK4, c-myc targets → cell cycle entry; "
            "  Chromaffin cells (adrenal medulla): normally quiescent; "
            "    MAX LOF → constitutive MYC network activation → proliferation; "
            "  EPIGENETIC LINK: "
            "    MAX LOF tumours: succinate pathway INDIRECTLY affected; "
            "    PRC2 (polycomb repressive complex 2) activity altered; "
            "  Bilateral predilection: adrenal medullary chromaffin cells bilaterally predisposed; "
            "  Young males: testicular/hormonal factors may modulate imprinting penetrance"
        ),
    },
]


def _make_patients(seed: int, gene: str) -> list:
    """Generate 40 synthetic PPGL patients per gene."""
    rng = random.Random(seed)

    gene_profiles = {
        "VHL":    dict(adrenal_pheo_pct=0.80, head_neck_pgl_pct=0.15, extra_adrenal_pct=0.30,
                       malignant_pct=0.05, bilateral_pct=0.55, norepinephrine_dom_pct=0.75),
        "SDHB":   dict(adrenal_pheo_pct=0.35, head_neck_pgl_pct=0.40, extra_adrenal_pct=0.65,
                       malignant_pct=0.40, bilateral_pct=0.20, norepinephrine_dom_pct=0.85),
        "SDHD":   dict(adrenal_pheo_pct=0.15, head_neck_pgl_pct=0.85, extra_adrenal_pct=0.20,
                       malignant_pct=0.05, bilateral_pct=0.10, norepinephrine_dom_pct=0.20),
        "SDHA":   dict(adrenal_pheo_pct=0.30, head_neck_pgl_pct=0.35, extra_adrenal_pct=0.40,
                       malignant_pct=0.10, bilateral_pct=0.15, norepinephrine_dom_pct=0.50),
        "SDHC":   dict(adrenal_pheo_pct=0.15, head_neck_pgl_pct=0.85, extra_adrenal_pct=0.15,
                       malignant_pct=0.03, bilateral_pct=0.10, norepinephrine_dom_pct=0.15),
        "SDHAF2": dict(adrenal_pheo_pct=0.00, head_neck_pgl_pct=1.00, extra_adrenal_pct=0.00,
                       malignant_pct=0.02, bilateral_pct=0.20, norepinephrine_dom_pct=0.05),
        "RET":    dict(adrenal_pheo_pct=0.50, head_neck_pgl_pct=0.05, extra_adrenal_pct=0.10,
                       malignant_pct=0.05, bilateral_pct=0.50, norepinephrine_dom_pct=0.30),
        "MAX":    dict(adrenal_pheo_pct=0.90, head_neck_pgl_pct=0.10, extra_adrenal_pct=0.10,
                       malignant_pct=0.08, bilateral_pct=0.75, norepinephrine_dom_pct=0.50),
    }
    p = gene_profiles.get(gene, gene_profiles["SDHB"])

    treatment_map = {
        "VHL":    ["alpha-blockade+surgery", "alpha-blockade+laparoscopic-adrenalectomy", "cortical-sparing-adrenalectomy"],
        "SDHB":   ["alpha-blockade+surgery", "surgery+PRRT-177Lu-DOTATATE", "TMZ-CAP+sunitinib"],
        "SDHD":   ["observation", "stereotactic-RT", "surgery+observation"],
        "SDHA":   ["alpha-blockade+surgery", "sunitinib+everolimus", "observation"],
        "SDHC":   ["observation", "surgery", "stereotactic-RT"],
        "SDHAF2": ["observation", "stereotactic-RT", "surgery"],
        "RET":    ["alpha-blockade+adrenalectomy+thyroidectomy", "vandetanib", "cabozantinib+thyroidectomy"],
        "MAX":    ["alpha-blockade+cortical-sparing-adrenalectomy", "bilateral-cortical-sparing", "alpha-blockade+surgery"],
    }
    treatments = treatment_map.get(gene, ["alpha-blockade+surgery"])

    mutation_map = {
        "VHL":    ["p.Arg167Trp", "p.Arg167Gln", "del_exon1-3", "p.Leu188Val", "p.Tyr112His"],
        "SDHB":   ["p.Ser163Pro", "p.Arg27Ter", "del_exon3", "p.Cys101Tyr", "p.Arg46Ter"],
        "SDHD":   ["p.Asp92Tyr", "p.Pro81Leu", "del_exon1", "p.Leu95Pro", "c.IVS1+1G>A"],
        "SDHA":   ["p.Arg31Ter", "p.Arg589Trp", "p.Glu383Lys", "p.Arg236Cys", "p.Phe576Ser"],
        "SDHC":   ["p.Arg133Ter", "p.Arg169Trp", "p.His112Arg", "c.IVS1+3A>G", "p.Pro143Ser"],
        "SDHAF2": ["p.Gly78Val", "p.Gly78Ala", "p.Arg19Ter", "c.232G>T", "p.Ala50Thr"],
        "RET":    ["p.Cys634Phe", "p.Cys634Tyr", "p.Met918Thr", "p.Cys620Arg", "p.Cys618Arg"],
        "MAX":    ["p.Arg60Gln", "p.His28Arg", "p.Leu13Pro", "p.Arg36Ter", "p.Ala69Val"],
    }
    mutations = mutation_map.get(gene, ["unknown"])

    patients = []
    for i in range(40):
        adrenal_pheo = rng.random() < p["adrenal_pheo_pct"]
        head_neck_pgl = rng.random() < p["head_neck_pgl_pct"] if not adrenal_pheo else False
        extra_adrenal = rng.random() < p["extra_adrenal_pct"] if not adrenal_pheo and not head_neck_pgl else False
        malignant = rng.random() < p["malignant_pct"]
        bilateral = rng.random() < p["bilateral_pct"] if adrenal_pheo else False
        norepi_dom = rng.random() < p["norepinephrine_dom_pct"]
        age_dx = rng.randint(18, 55) if gene not in ("RET",) else rng.randint(5, 50)
        treatment = rng.choice(treatments)
        mutation = rng.choice(mutations)
        tumour_size_cm = round(rng.uniform(1.5, 6.5), 1)
        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis": age_dx,
            "adrenal_pheo": adrenal_pheo,
            "head_neck_pgl": head_neck_pgl,
            "extra_adrenal_pgl": extra_adrenal,
            "malignant": malignant,
            "bilateral": bilateral,
            "norepinephrine_dominant": norepi_dom,
            "tumour_size_cm": tumour_size_cm,
            "treatment": treatment,
            "mutation": mutation,
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Pheochromocytoma-Paraganglioma-Atlas."""
    return {
        "atlas":       "Hereditary-Pheochromocytoma-Paraganglioma-Atlas",
        "subtitle":    "Complete 8-Gene Hereditary PPGL Reference Atlas (VHL-SDHB-SDHD-SDHA-SDHC-SDHAF2-RET-MAX)",
        "total_genes": len(ATLAS_GENES),
        "seed_range":  f"{SEED_BASE}–{SEED_BASE+7}",
        "total_patients": 320,
        "genes": [g["gene"] for g in ATLAS_GENES],
        "gene_loci": {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "VHL":    "AD LOF (biallelic somatic second hit required)",
            "SDHB":   "AD LOF",
            "SDHD":   "AD LOF — MATERNAL IMPRINTING (paternal transmission only clinically relevant)",
            "SDHA":   "AD LOF",
            "SDHC":   "AD LOF",
            "SDHAF2": "AD LOF — PATERNAL IMPRINTING (same as SDHD)",
            "RET":    "AD GOF (constitutive RTK activation)",
            "MAX":    "AD LOF — PATERNAL IMPRINTING (same as SDHD)",
        },
        "key_clinical_rules": [
            "SDHB-40%-MALIGNANCY: highest malignancy risk in all hereditary PPGL — intensive surveillance; extra-adrenal predilection; succinate→2HG epigenetic CIMP",
            "SDHD/SDHAF2/MAX-PATERNAL-IMPRINTING: ONLY paternal transmission causes disease; maternal carriers are phenotypically silent but can transmit to next generation",
            "VHL-CLEAR-CELL-RCC: VHL LOF = clear cell RCC (sporadic also >90%); RCC + PPGL + hemangioblastoma triad; biallelic somatic loss required",
            "RET-EXCLUDE-PHEO-BEFORE-SURGERY: ALWAYS screen plasma metanephrines before any neck surgery in MEN2; occult PHEO → hypertensive crisis",
            "RET-M918T-MEN2B-NEONATAL: prophylactic thyroidectomy within 6 months of life (ATA-D highest risk codon); mucosal neuromas + marfanoid PATHOGNOMONIC",
            "SDHAF2-HEAD-NECK-ONLY: no adrenal pheo ever reported in SDHAF2 families; Dutch founder Gly78Val; paternal imprinting same as SDHD",
            "MAX-BILATERAL-ADRENAL: bilateral adrenal PHEO in young male → think MAX (+ VHL, RET); cortical-sparing surgery preferred to avoid permanent Addison",
            "SDHx-SDHB-IHC: SDHB IHC absent = ANY SDHx mutation (SDHB/SDHC/SDHD/SDHA/SDHAF2); SDHA IHC absent = SDHA mutation specific",
            "SDHA-GIST-IMATINIB-RESISTANT: SDH-deficient GIST = SDHA/SDHB/SDHC germline; imatinib ineffective (no KIT/PDGFRA mutation); young patient + gastric GIST → test SDHx",
            "SDHB-FDG-PET-MALIGNANT: 18F-FDG PET for malignant PPGL; 68Ga-DOTATATE for well-differentiated SDHx; PRRT (177Lu-DOTATATE) for progressive metastatic",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for all 8 hereditary PPGL genes."""
    genes_data = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        mutations_seen = {}
        for p in pts:
            mutations_seen[p["mutation"]] = mutations_seen.get(p["mutation"], 0) + 1
        genes_data.append({
            "gene":               g["gene"],
            "locus":              g["locus"],
            "protein":            g["protein"],
            "protein_size":       g["protein_size"],
            "inheritance":        g["inheritance"],
            "disease_category":   g["disease_category"],
            "disease_pathway":    g["disease_pathway"],
            "n_patients":         len(pts),
            "mean_age_dx":        round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1),
            "adrenal_pheo_pct":   round(100 * sum(1 for p in pts if p["adrenal_pheo"]) / len(pts), 1),
            "head_neck_pgl_pct":  round(100 * sum(1 for p in pts if p["head_neck_pgl"]) / len(pts), 1),
            "extra_adrenal_pct":  round(100 * sum(1 for p in pts if p["extra_adrenal_pgl"]) / len(pts), 1),
            "malignant_pct":      round(100 * sum(1 for p in pts if p["malignant"]) / len(pts), 1),
            "bilateral_pct":      round(100 * sum(1 for p in pts if p["bilateral"]) / len(pts), 1),
            "norepinephrine_dom_pct": round(100 * sum(1 for p in pts if p["norepinephrine_dominant"]) / len(pts), 1),
            "mean_tumour_size_cm": round(sum(p["tumour_size_cm"] for p in pts) / len(pts), 1),
            "treatment_breakdown": treatments,
            "mutation_breakdown":  mutations_seen,
            "patients":            pts,
        })
    return {
        "atlas": "Hereditary-Pheochromocytoma-Paraganglioma-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Key clinical terms for Hereditary-Pheochromocytoma-Paraganglioma-Atlas."""
    definitions = [
        {
            "term": "SDHB (PPGL2) — Highest Malignancy Risk ~40%",
            "genes": ["SDHB"],
            "definition": (
                "SDHB PPGL2 — MALIGNANCY RISK 40%: highest in all hereditary PPGL genes. "
                "EXTRA-ADRENAL PREDILECTION: abdominal, thoracic paraganglioma most common. "
                "BIOCHEMISTRY: normetanephrine dominant (norepinephrine-secreting); plasma succinate elevated. "
                "SDHB IHC: granular cytoplasmic staining absent — indirect marker for ANY SDHx mutation. "
                "SURVEILLANCE: annual plasma metanephrines; MRI skull base to pelvis every 1-2 years; "
                "18F-FDG PET (malignant/metastatic); 68Ga-DOTATATE PET (well-differentiated). "
                "MALIGNANT PPGL TREATMENT: "
                "PRRT (177Lu-DOTATATE): first-line for SSTR-expressing malignant PPGL (NETTER-1 trial). "
                "Sunitinib: anti-VEGF; cabozantinib (RET/MET); TMZ/CAP chemotherapy. "
                "KEY RULE: SDHB mutation + extra-adrenal location = highest malignancy risk combination. "
                "SDHx CIMP: succinate inhibits TET enzymes → DNA hypermethylation → "
                "tumour suppressor silencing → epigenetic driver."
            ),
        },
        {
            "term": "SDHD / SDHAF2 / MAX — Paternal Imprinting: Only Paternal Transmission Causes Disease",
            "genes": ["SDHD", "SDHAF2", "MAX"],
            "definition": (
                "THREE PPGL GENES WITH PATERNAL IMPRINTING: SDHD (11q23.1), SDHAF2 (11q13.1), MAX (14q23.3). "
                "IMPRINTING RULE: maternal allele silenced → only PATERNAL copy expressed. "
                "CONSEQUENCE: "
                "  Paternal mutation → patient haploinsufficient → disease risk; "
                "  Maternal mutation → patient has TWO functional copies (paternal intact) → NO disease; "
                "  BUT maternal carrier can transmit mutation to children: "
                "    If that child transmits it paternally → grandchildren at risk (skip-generation appearance). "
                "SDHD (PPGL1): head-neck PGL + adrenal ~15%; multifocal (50%); malignancy 5%. "
                "SDHAF2 (PGL2): EXCLUSIVELY head-neck; no adrenal PHEO; ultra-rare; Dutch Gly78Val founder. "
                "MAX: BILATERAL ADRENAL pheo; young males; paternal imprinting. "
                "GENETIC COUNSELLING: "
                "  ALWAYS establish parent of origin; "
                "  Maternal SDHD/SDHAF2/MAX carriers: reassure they are not at personal risk; "
                "  Their children with paternally inherited mutation → annual surveillance from age 15. "
                "PEDIGREE CLUE: seemingly unaffected generation + affected grandchildren → imprinting."
            ),
        },
        {
            "term": "RET (MEN2A/MEN2B) — Exclude PHEO Before Any Neck Surgery + Codon-Guided Thyroidectomy",
            "genes": ["RET"],
            "definition": (
                "RET MEN2 — TWO CRITICAL RULES: "
                "RULE 1 — EXCLUDE PHEO BEFORE SURGERY: "
                "  ALWAYS screen plasma fractionated metanephrines (or 24h urine) BEFORE neck surgery; "
                "  Occult pheo during thyroidectomy → catecholamine crisis → fatal hypertension; "
                "  If PHEO present: alpha-blockade (phenoxybenzamine ≥10-14 days) → PHEO surgery → "
                "  THEN thyroidectomy/parathyroidectomy. "
                "RULE 2 — CODON-GUIDED PROPHYLACTIC THYROIDECTOMY: "
                "  ATA-D (M918T = MEN2B): thyroidectomy within 6 MONTHS OF LIFE; "
                "  ATA-C (C634F/Y/R/W): thyroidectomy by AGE 5 YEARS; "
                "  ATA-B (C609/C611/C618/C620/C630): thyroidectomy by age 5-10 years; "
                "  ATA-A (other mutations): thyroidectomy based on calcitonin trend; "
                "MEN2B PATHOGNOMONIC: mucosal neuromas (tongue, lips, eyelids) + marfanoid habitus. "
                "MTC TREATMENT: total thyroidectomy + central neck dissection; "
                "Advanced: vandetanib (FDA 2011) or cabozantinib (FDA 2012) — RET kinase inhibitors. "
                "RET mutation-specific inhibitors: selpercatinib (LOXO-292) for RET-altered tumours."
            ),
        },
        {
            "term": "VHL — Clear Cell RCC + PPGL + Hemangioblastoma Triad; MLPA Mandatory",
            "genes": ["VHL"],
            "definition": (
                "VHL DISEASE TRIAD: clear cell RCC + CNS/retinal hemangioblastoma + PPGL. "
                "VHL TYPE 2 (missense mutations): PHEO present; Type 1 (deletions/truncations): RCC dominant. "
                "TYPE 2C (R200W, p.Arg200Trp Chuvash): PHEO + ERYTHROCYTOSIS — unique VHL phenotype. "
                "CLEAR CELL RCC: almost always biallelic VHL loss (sporadic or germline); "
                "CAIX IHC positive; VHL pathway: HIF-1α → VEGF → angiogenesis (highly vascular tumour). "
                "MLPA MANDATORY: 20% of VHL families have large deletions (not detected by sequencing). "
                "PHEO IN VHL: "
                "  Usually bilateral; norepinephrine-dominant (extra-adrenal pattern); "
                "  Alpha-blockade before surgery; cortical-sparing preferred; "
                "  Malignancy rate ~5% (low compared to SDHB). "
                "SURVEILLANCE (comprehensive multiorgan): "
                "  Annual metanephrines; annual MRI abdomen/pelvis; "
                "  MRI brain/spine every 2 years; annual ophthalmology; "
                "  Pancreatic MRI every 2 years (cysts/NETs). "
                "RCC TREATMENT: partial nephrectomy (nephron-sparing); "
                "Advanced: sunitinib; belzutifan (HIF-2α inhibitor, FDA 2021 for VHL-associated RCC/PPGL/hemangioblastoma)."
            ),
        },
        {
            "term": "SDHA (PPGL5) + SDH-deficient GIST — SDHB IHC vs SDHA IHC Interpretation",
            "genes": ["SDHA"],
            "definition": (
                "SDHx IHC INTERPRETATION — CRITICAL RULE: "
                "  SDHB IHC absent: marker for ANY SDHx mutation (SDHB, SDHC, SDHD, SDHA, SDHAF2); "
                "  SDHA IHC absent: SPECIFIC for SDHA mutation only; "
                "  ALGORITHM: "
                "    SDHB IHC absent + SDHA IHC absent → SDHA germline mutation; "
                "    SDHB IHC absent + SDHA IHC present → SDHB/SDHC/SDHD/SDHAF2 mutation; "
                "    SDHB IHC present → non-SDHx PPGL (VHL, RET, NF1, MAX, or sporadic). "
                "SDHA PPGL5: "
                "  LOWEST penetrance SDHx (~10-20%); appropriate counselling — most carriers unaffected; "
                "  GIST + pituitary adenoma also associated. "
                "SDH-DEFICIENT GIST — IMATINIB-RESISTANT: "
                "  Stomach predominant; young patients (<40); "
                "  KIT and PDGFRA wild-type (standard GIST panel negative); "
                "  SDHB IHC absent on GIST tissue → SDH panel (SDHA/SDHB/SDHC) germline testing; "
                "  Imatinib INEFFECTIVE (targets KIT/PDGFRA, which are normal here); "
                "  Treatment: sunitinib; everolimus; temozolomide. "
                "SURVEILLANCE: annual metanephrines; upper GI endoscopy every 3 years; MRI pituitary every 3-5 years."
            ),
        },
        {
            "term": "SDHC (PPGL3) — Head-Neck PGL Without Imprinting (vs SDHD Same Phenotype)",
            "genes": ["SDHC"],
            "definition": (
                "SDHC PPGL3 vs SDHD PPGL1 — SAME PHENOTYPE, DIFFERENT INHERITANCE: "
                "SDHC: NO IMPRINTING — maternal OR paternal transmission causes disease (AD standard); "
                "SDHD: MATERNAL IMPRINTING — only paternal transmission active. "
                "BOTH: head-neck paraganglioma predominantly; adrenal PHEO rare; non-secreting majority. "
                "SDHC ADDITIONAL FEATURES: "
                "  Malignancy rate: ~2-5% (lowest in SDHx); "
                "  Less multifocal than SDHD (SDHD 50% multifocal); "
                "  Predominantly carotid body paraganglioma; jugulotympanic PGL. "
                "SURVEILLANCE: "
                "  Annual plasma metanephrines + methoxytyramine (dopamine metabolite — key for non-secreting); "
                "  MRI neck/skull base every 2-3 years; "
                "  68Ga-DOTATATE PET-CT: best modality for head-neck PGL and multicentric survey. "
                "TREATMENT: "
                "  Active surveillance for small, stable, non-secreting head-neck PGL; "
                "  Surgery: curative for localised; risk of CN palsy at skull base; "
                "  Stereotactic body radiotherapy (SBRT): growth control for surgically inaccessible."
            ),
        },
        {
            "term": "Hereditary PPGL — 8-Gene Differential Guide: Malignancy Risk + Tumour Location + Imprinting",
            "genes": ["VHL", "SDHB", "SDHD", "SDHA", "SDHC", "SDHAF2", "RET", "MAX"],
            "definition": (
                "PPGL GENE DIFFERENTIAL: "
                "MALIGNANCY RISK (high→low): SDHB (~40%) > SDHA (10-20%) > MAX (5-10%) > VHL (~5%) ≈ RET (5%) > SDHD (5%) > SDHC (2-5%) > SDHAF2 (2%); "
                "TUMOUR LOCATION: "
                "  SDHAF2: EXCLUSIVELY head-neck (no adrenal); "
                "  SDHD + SDHC: predominantly head-neck (adrenal < 20%); "
                "  MAX: predominantly bilateral ADRENAL (young males); "
                "  SDHB: predominantly EXTRA-ADRENAL (abdominal/thoracic) + head-neck; "
                "  VHL: predominantly adrenal (bilateral); hemangioblastoma + RCC also; "
                "  RET: adrenal (bilateral); MTC + PHPT also (MEN2A); "
                "IMPRINTING: "
                "  3 paternal-imprinted: SDHD, SDHAF2, MAX → only paternal mutation active; "
                "  No imprinting: VHL, SDHB, SDHA, SDHC, RET; "
                "BIOCHEMISTRY: "
                "  Adrenal PHEO: epinephrine + norepinephrine (MEN2A/RET, VHL, MAX); "
                "  Extra-adrenal + head-neck: norepinephrine + dopamine (no PNMT enzyme); "
                "FIRST-LINE IMAGING: "
                "  Functional: 68Ga-DOTATATE PET (SDHx, head-neck); 18F-FDG PET (malignant/SDHB); "
                "  Anatomic: MRI whole body preferred (no radiation, young patients, bilateral)."
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Pheochromocytoma-Paraganglioma-Atlas",
        "count":       len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:600])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Terms: {df['count']}")
