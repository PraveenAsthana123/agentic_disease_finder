#!/usr/bin/env python3
"""Hereditary-PHEO-PGL-SDH-Atlas — Complete 8-Gene Pheochromocytoma/Paraganglioma/SDH Syndrome Atlas
SDHB   (Succinate dehydrogenase iron-sulfur subunit B; 280 aa; 1p36.13; AD;
         Hereditary PGL4 / PHEO;
         HIGHEST MALIGNANT RISK 35-40% among all PHEO/PGL genes PATHOGNOMONIC;
         Extra-adrenal location most common; Cr-SSTR-PET mandatory;
         Lutetium-DOTATATE (Lu-PRRT) for metastatic;
         seed SEED_BASE+0) ·
SDHD   (Succinate dehydrogenase cytochrome b small subunit; 159 aa; 11q23.1; AD;
         Hereditary PGL1;
         PATERNAL IMPRINTING: disease ONLY if inherited from father -- UNIQUE inheritance mode;
         Head/neck PGL characteristic; bilateral; low malignant risk 5%;
         seed SEED_BASE+1) ·
SDHC   (Succinate dehydrogenase cytochrome b large subunit; 169 aa; 1q23.3; AD;
         Hereditary PGL3;
         Parasympathetic head/neck PGL; LOWEST malignant risk <5%;
         No adrenal involvement; non-functional in most;
         seed SEED_BASE+2) ·
SDHA   (Succinate dehydrogenase flavoprotein subunit A; 621 aa; 5p15.33; AD;
         Hereditary PGL5;
         Largest complex II subunit; GIST risk co-occurs;
         IHC loss of SDHA confirms pathogenic variant;
         seed SEED_BASE+3) ·
VHL    (von Hippel-Lindau tumour suppressor; 213 aa; 3p25.3; AD;
         VHL disease;
         CLEAR CELL RCC + CNS/retinal HEMANGIOBLASTOMAS + PHEO (type 2A/2B/2C) + PNET;
         Belzutifan (Welireg) FDA2021 -- first HIF-2α inhibitor for VHL tumours;
         seed SEED_BASE+4) ·
RET    (Ret proto-oncogene; 1114 aa; 10q11.21; AD GOF;
         MEN2A / MEN2B;
         MEDULLARY THYROID CANCER 100% LIFETIME PATHOGNOMONIC;
         PHEO 50% in codon 634 MEN2A; PROPHYLACTIC THYROIDECTOMY age-based on codon;
         Vandetanib / Cabozantinib FDA;
         seed SEED_BASE+5) ·
TMEM127 (Transmembrane protein 127; 238 aa; 2q11.2; AD;
          Hereditary PHEO;
          Bilateral adrenal PHEO characteristic;
          mTORC1 pathway; low malignant risk;
          IHC TMEM127 loss confirms;
          seed SEED_BASE+6) ·
MAX    (MYC-associated factor X; 236 aa; 14q23.3; AD;
         Hereditary PHEO;
         PATERNAL IMPRINTING LIKE SDHD -- second imprinted PHEO gene;
         Bilateral adrenal; young onset <30 years;
         cMyc pathway dysregulation;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3078-3085)
"""
import random

SEED_BASE = 3078

ATLAS_GENES = [
    {
        "gene": "SDHB",
        "protein": (
            "SDHB -- 1p36.13 Autosomal-Dominant-LOF -- 280aa -- Succinate-Dehydrogenase-"
            "Iron-Sulfur-Subunit-B-Complex-II-Krebs-Cycle-Highest-Malignant-Risk-PGL4-"
            "PHEO-Extra-Adrenal-SSTR-PET-Mandatory-OMIM-185470"
        ),
        "locus": "1p36.13",
        "protein_size": (
            "280 aa / 32 kDa (SDHB; iron-sulfur protein; complex II catalytic subunit; "
            "STRUCTURE: N-terminal mitochondrial targeting sequence; 3 iron-sulfur clusters "
            "(2Fe-2S, 3Fe-4S, 4Fe-4S); interacts with SDHC/SDHD heterodimer (membrane anchor); "
            "forms tetrameric complex II (SDHA-SDHB-SDHC-SDHD); "
            "FUNCTION: "
            "  Complex II (succinate-ubiquinone oxidoreductase) bridges Krebs cycle + respiratory chain; "
            "  SDHB: electron transfer from SDHA (succinate oxidation) to ubiquinone; "
            "  SDH LOF -> succinate accumulates -> HIF-1α stabilisation (pseudohypoxia); "
            "  Succinate accumulation inhibits prolyl hydroxylases -> prevents VHL-mediated HIF degradation; "
            "  Epigenetic: succinate inhibits TET dioxygenases and JmjC demethylases -> SDHx hypermethylator; "
            "HEREDITARY PGL4 (SDHB): "
            "  HIGHEST MALIGNANT RISK among all SDHx genes: 35-40% metastatic PHEO/PGL; "
            "  Malignancy defined by distant metastasis (not histology): bone, liver, lung, lymph nodes; "
            "  Extra-adrenal PGL common: 50-60% extra-adrenal (thoracic, abdominal, pelvic); "
            "  Biochemistry: predominantly norepinephrine and dopamine secreting; "
            "  Cr-SSTR-PET (68Ga-DOTATATE PET/CT): mandatory staging and surveillance; "
            "  MIBG scintigraphy: still used if SSTR-PET unavailable; "
            "  Lutetium-DOTATATE (Lu-PRRT): for SSTR-avid metastatic disease; "
            "  Sunitinib: anti-VEGFR -- for SSTR-negative or progressive metastatic; "
            "  Succinate-to-fumarate ratio: metabolic biomarker in urine/plasma; "
        ),
        "inheritance": (
            "Autosomal dominant LOF (de novo rare; mostly familial); "
            "SDHB 1p36.13; bi-allelic required for tumorigenesis (Knudson 2-hit); "
            "Penetrance: ~50-60% lifetime tumour risk; "
            "SDHB IHC: loss of SDHB AND SDHA staining in tumour confirms SDHx pathogenicity; "
        ),
        "disease_category": "Hereditary PGL4 / PHEO (SDHx spectrum)",
        "key_mutations": [
            "p.Asp92Tyr (Fe-S cluster coordination -- most common European)",
            "p.Pro197Arg (European founder; iron-sulfur cluster 2)",
            "p.Cys101Tyr (Fe-S cluster 1 -- common)",
            "p.Arg46Gln (European)",
            "Exon 1 deletion (large rearrangement -- requires MLPA)",
            "p.Trp200Ter (truncating -- LOF)",
            "p.Arg230Cys (Fe-S cluster 3)",
        ],
        "clinical_keys": [
            "HIGHEST MALIGNANT RISK among all PHEO/PGL genes: 35-40% metastatic -- most important SDHx fact",
            "Extra-adrenal location common (50-60%) -- chest/abdomen/pelvis; thoracic PGL from SDHB",
            "Cr-SSTR-PET (68Ga-DOTATATE): mandatory for initial staging, surveillance every 1-2 years",
            "SDHB IHC: loss of SDHB protein on tumour section confirms SDHx variant (not gene-specific)",
            "Lutetium-DOTATATE (Lu-PRRT) for SSTR-avid metastatic SDHB disease",
            "Sunitinib: VEGFR inhibitor for SSTR-negative or progressive metastatic disease",
            "Annual biochemistry: plasma/urine metanephrines + methoxytyramine (dopamine marker)",
            "Genetic testing triggers: young PHEO/PGL, extra-adrenal, bilateral, recurrent, family history",
            "MLPA mandatory: large exonic deletions account for 10-15% SDHB pathogenic variants",
        ],
    },
    {
        "gene": "SDHD",
        "protein": (
            "SDHD -- 11q23.1 Autosomal-Dominant-LOF-PATERNAL-IMPRINTING -- 159aa -- "
            "Succinate-Dehydrogenase-Cytochrome-b-Small-Subunit-PGL1-Head-Neck-PGL-"
            "Paternal-Inheritance-Only-UNIQUE-OMIM-602690"
        ),
        "locus": "11q23.1",
        "protein_size": (
            "159 aa / 17 kDa (SDHD; CYB-small; small transmembrane anchor subunit of complex II; "
            "STRUCTURE: 3 transmembrane helices; intramembrane histidines coordinate heme b; "
            "pairs with SDHC to form cytochrome b558 dimer; anchors SDHA-SDHB in inner mitochondrial membrane; "
            "FUNCTION: "
            "  SDHD (with SDHC) anchors SDH complex to inner mitochondrial membrane; "
            "  Ubiquinone reduction occurs at the SDHC/SDHD interface; "
            "  SDHD LOF -> succinate accumulation -> pseudohypoxia (HIF-1α) same as SDHB; "
            "HEREDITARY PGL1 (SDHD): "
            "  PATERNAL IMPRINTING: DISEASE ONLY IF VARIANT INHERITED FROM FATHER -- UNIQUE; "
            "  Maternal SDHD inheritance: does NOT cause disease (maternally imprinted locus 11q23); "
            "  Head and neck PGL characteristic: carotid body, glomus jugulare, glomus tympanicum; "
            "  Bilateral and multifocal head/neck PGL: very common (>50% bilateral carotid body); "
            "  Low malignant risk: ~5% metastatic (compared to SDHB 35-40%); "
            "  Non-functional in most: <20% secrete catecholamines; "
            "  Adrenal PHEO: rare in SDHD (unlike SDHB adrenal/extra-adrenal mix); "
            "SURVEILLANCE: "
            "  Annual MRI neck + Cr-SSTR-PET for multifocal disease; "
            "  'Watch and wait' acceptable for small asymptomatic head/neck PGL; "
            "  Treat if: growing, functional, compressive (cranial nerve); "
        ),
        "inheritance": (
            "Autosomal dominant LOF with PATERNAL IMPRINTING; "
            "SDHD 11q23.1 (imprinted locus -- 11p15 cluster imprinting extends to 11q in some models); "
            "Disease ONLY if paternal transmission: maternally inherited SDHD does NOT cause disease; "
            "TEST BOTH PARENTS: paternal vs maternal inheritance completely changes recurrence risk; "
            "Paternal: ~50% children affected; Maternal: 0% children affected; "
        ),
        "disease_category": "Hereditary PGL1 / Head-Neck PGL (SDHD paternal imprinting)",
        "key_mutations": [
            "p.Asp92Tyr (SDHD -- also in SDHB but different position; common European)",
            "p.His50Arg (transmembrane histidine -- heme coordination)",
            "p.Leu95Pro (transmembrane domain -- lipid bilayer packing)",
            "p.Pro81Leu (lipid bilayer contact)",
            "p.Arg38Ter (truncating -- LOF)",
            "c.149G>A (p.Gly50Glu -- founder Netherlands)",
            "exon 3 deletion (Netherlands founder PGL1 cluster)",
        ],
        "clinical_keys": [
            "PATERNAL IMPRINTING: disease ONLY if inherited from father -- UNIQUE genetic rule; test both parents",
            "Head and neck PGL characteristic: carotid body (most common), glomus jugulare, glomus tympanicum",
            "Bilateral and multifocal head/neck PGL >50%: MRI neck baseline + annual if multifocal",
            "Low malignant risk ~5%: reassure family; contrast with SDHB 35-40%",
            "Non-functional majority: plasma metanephrines normal in most; MRI is primary surveillance tool",
            "Netherlands SDHD founder cluster: 3 specific variants common in Dutch population",
            "Watch and wait: acceptable for small asymptomatic carotid body tumours -- no surgery unless symptomatic",
            "Maternal SDHD carrier: reassure -- zero risk of disease in children (imprinted locus)",
            "Annual Cr-SSTR-PET when bilateral/multifocal: assess extent + detect progression",
        ],
    },
    {
        "gene": "SDHC",
        "protein": (
            "SDHC -- 1q23.3 Autosomal-Dominant-LOF -- 169aa -- Succinate-Dehydrogenase-"
            "Cytochrome-b-Large-Subunit-PGL3-Parasympathetic-Head-Neck-PGL-Lowest-Malignant-"
            "Risk-OMIM-602413"
        ),
        "locus": "1q23.3",
        "protein_size": (
            "169 aa / 19 kDa (SDHC; CYB-large; large transmembrane anchor subunit of complex II; "
            "STRUCTURE: 4 transmembrane helices; intramembrane histidines coordinate second heme b; "
            "dimerises with SDHD to form cytochrome b558; provides SDHB binding interface; "
            "FUNCTION: "
            "  SDHC (with SDHD): anchors catalytic SDHA-SDHB dimer to inner mitochondrial membrane; "
            "  SDHC LOF -> complex II dissociation -> same pseudohypoxia cascade as other SDHx; "
            "  SDHC LOF: succinate accumulation -> HIF-1α -> VEGF -> angiogenesis in paraganglia; "
            "HEREDITARY PGL3 (SDHC): "
            "  LOWEST malignant risk among all SDHx: <5% metastatic; "
            "  Predominantly parasympathetic head and neck PGL; "
            "  No adrenal PHEO involvement (strongly parasympathetic tropism); "
            "  Carotid body tumour most common; glomus jugulare, glomus tympanicum; "
            "  Non-functional: >90% non-secretory; "
            "  SDHC IHC: SDHB staining LOST (marker for all SDHx); SDHA staining preserved; "
            "  Penetrance: estimated 40-50% lifetime, may be lower; "
            "  SDHC variants less commonly tested: ensure SDHC included in PGL panel; "
        ),
        "inheritance": (
            "Autosomal dominant LOF; SDHC 1q23.3; NO imprinting (unlike SDHD and MAX); "
            "Standard 50% transmission risk; penetrance 40-50% (lower than SDHB); "
            "SDHC: both sexes equally affected; "
        ),
        "disease_category": "Hereditary PGL3 / Head-Neck PGL (SDHC)",
        "key_mutations": [
            "p.Arg55Trp (transmembrane helix -- very rare)",
            "p.Arg97Ter (truncating -- LOF)",
            "p.Gln109Ter (truncating)",
            "p.Ser45Phe (transmembrane domain)",
            "Exon deletions (require MLPA)",
        ],
        "clinical_keys": [
            "LOWEST malignant risk among all SDHx genes (<5%): most benign SDHx mutation",
            "Parasympathetic head/neck PGL only: NO adrenal PHEO -- key differentiator from SDHB",
            "Non-functional >90%: plasma metanephrines normal; MRI neck is primary surveillance",
            "Carotid body tumour most common: painless neck mass at carotid bifurcation",
            "SDHB IHC loss in tumour: occurs with ALL SDHx variants -- not specific to SDHC",
            "Watch and wait for asymptomatic: small head/neck PGL can be observed with MRI",
            "SDHC often underdiagnosed: ensure SDHC exons included in targeted SDHx panel",
            "Bilateral multifocal less common than SDHD: usually unilateral in SDHC",
            "Annual MRI neck surveillance from diagnosis: detect new lesions + monitor growth",
        ],
    },
    {
        "gene": "SDHA",
        "protein": (
            "SDHA -- 5p15.33 Autosomal-Dominant-LOF -- 621aa -- Succinate-Dehydrogenase-"
            "Flavoprotein-Subunit-A-Complex-II-Largest-Subunit-PGL5-GIST-Risk-IHC-"
            "SDHA-Loss-Diagnostic-OMIM-600857"
        ),
        "locus": "5p15.33",
        "protein_size": (
            "621 aa / 70 kDa (SDHA; FP; flavoprotein subunit; complex II catalytic subunit; "
            "STRUCTURE: FAD-binding domain; capping domain; helical domain; C-terminal domain; "
            "contains FAD cofactor; covalently linked via His99 (humans); "
            "dimerises with SDHB at the matrix-facing interface; "
            "FUNCTION: "
            "  SDHA catalyses succinate oxidation to fumarate in Krebs cycle; "
            "  FAD reduction: succinate -> fumarate + FADH2; "
            "  Electrons transferred: SDHA -> SDHB -> ubiquinone via Fe-S clusters; "
            "  SDHA LOF -> complex II absence -> succinate accumulation -> pseudohypoxia; "
            "  SDHA is the ONLY complex II subunit with a unique IHC marker; "
            "IHC SDHA: "
            "  SDHA IHC: SDHA staining LOST = pathogenic SDHA variant confirmed; "
            "  SDHB IHC: SDHB staining LOST = any SDHx variant (SDHB/C/D not SDHA); "
            "  Combined IHC: if SDHA + SDHB both lost = SDHA variant; SDHB lost + SDHA preserved = SDHB/C/D; "
            "HEREDITARY PGL5 (SDHA): "
            "  PHEO and PGL: adrenal + extra-adrenal; similar tropism to SDHB; "
            "  SDHA-deficient GIST: co-occurrence with SDHx-GIST; Carney triad; "
            "  Biochemistry: predominantly norepinephrine secreting; "
            "  Malignant risk: intermediate 10-15%; lower than SDHB; "
            "  Less common than SDHB: probably under-ascertained; "
        ),
        "inheritance": (
            "Autosomal dominant LOF; SDHA 5p15.33; NO imprinting; "
            "Bi-allelic (germline + somatic 2nd hit) required for tumorigenesis; "
            "Standard 50% transmission; penetrance 30-40% estimated; "
            "SDHA IHC: loss of SDHA protein confirms pathogenic SDHA variant in tumour; "
        ),
        "disease_category": "Hereditary PGL5 / PHEO (SDHA; GIST co-risk)",
        "key_mutations": [
            "p.Arg31Ter (truncating -- LOF; common)",
            "p.Arg589Gln (C-terminal domain; SDHB interaction)",
            "p.Ile107Thr (FAD domain -- flavin binding)",
            "p.Glu524Lys (capping domain)",
            "p.Thr422Ile (helical domain)",
            "Large exon deletions (require MLPA)",
        ],
        "clinical_keys": [
            "IHC SDHA LOSS = SDHA variant confirmed: first SDHx IHC step; combined with SDHB IHC",
            "SDHA-deficient GIST: screen for synchronous GIST (abdominal imaging); Carney triad",
            "Intermediate malignant risk 10-15%: higher than SDHC/SDHD, lower than SDHB",
            "Adrenal and extra-adrenal PHEO/PGL: similar distribution to SDHB",
            "Norepinephrine predominantly secretory: plasma normetanephrine elevated",
            "Cr-SSTR-PET mandatory for staging: SSTR-avid in SDHA-related tumours",
            "SDHA often underpaneled: ensure SDHA included in all hereditary PHEO/PGL panels",
            "Krebs cycle connection: succinate-to-fumarate ratio in urine as biomarker",
            "Annual biochemistry (plasma normetanephrine) + imaging every 1-2 years",
        ],
    },
    {
        "gene": "VHL",
        "protein": (
            "VHL -- 3p25.3 Autosomal-Dominant-LOF -- 213aa -- Von-Hippel-Lindau-Tumour-"
            "Suppressor-E3-Ubiquitin-Adaptor-CRL2VHL-Clear-Cell-RCC-Hemangioblastoma-"
            "PHEO-PNET-Belzutifan-FDA2021-OMIM-193300"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "213 aa / 24 kDa (VHL; pVHL; von Hippel-Lindau protein; E3 ubiquitin ligase adaptor; "
            "STRUCTURE: alpha domain (Cullin-2 binding + ElonginC/B binding); beta domain (HIF-α binding); "
            "forms CRL2VHL E3 complex: VHL-ElonginC-ElonginB-Cullin2-RBX1; "
            "FUNCTION: "
            "  pVHL targets hydroxylated HIF-α subunits (HIF-1α, HIF-2α) for polyubiquitylation + degradation; "
            "  Prolyl hydroxylase (PHD1/2/3) hydroxylates HIF-α Pro402/564 (oxygen-dependent); "
            "  pVHL captures hydroxylated HIF-α -> CRL2VHL -> ubiquitin -> proteasome; "
            "  VHL LOF: HIF-2α accumulates (not degraded) -> VEGF, EPO, CAIX overexpression -> angiogenesis; "
            "  HIF-2α predominantly in clear cell RCC and VHL-associated tumours; "
            "VHL DISEASE (3 major components): "
            "  Type 1: NO PHEO; ccRCC + hemangioblastomas (CNS + retinal); "
            "  Type 2A: PHEO + hemangioblastomas; LOW ccRCC risk; Tyr98 hotspot; "
            "  Type 2B: PHEO + ccRCC + hemangioblastomas; HIGH ccRCC risk; "
            "  Type 2C: PHEO ONLY; NO hemangioblastomas or ccRCC; rarest; "
            "  Retinal hemangioblastoma: often FIRST presentation; annual ophthalmology mandatory; "
            "  CNS hemangioblastoma: cerebellum, brainstem, spinal cord; VEGF-driven; "
            "  Pancreatic cysts/NET: pancreatic involvement in ~70% VHL; "
            "BELZUTIFAN (Welireg, HIF-2α inhibitor): "
            "  FDA2021: first-in-class HIF-2α inhibitor; FDA-approved for VHL disease (ccRCC + HBL + pNET); "
            "  Mechanism: directly binds HIF-2α PAS-B pocket -> prevents HIF-2α/ARNT dimerisation; "
            "  Phase 3 LITESPARK-010: belzutifan vs everolimus in ccRCC; "
        ),
        "inheritance": (
            "Autosomal dominant LOF; VHL 3p25.3; NO imprinting; "
            "Classic 2-hit tumour suppressor; germline LOF + somatic LOF of 2nd allele in tumour; "
            "Penetrance: ~90% lifetime; VHL type (1, 2A, 2B, 2C) genotype-phenotype correlation; "
            "Type 2C (PHEO only): Tyr98His almost exclusively; "
        ),
        "disease_category": "VHL disease (ccRCC + CNS/Retinal Hemangioblastoma + PHEO + PNET)",
        "key_mutations": [
            "p.Tyr98His (type 2C -- PHEO only; almost exclusively)",
            "p.Asn78Ser (type 2A -- PHEO + HBL; low ccRCC)",
            "p.Arg167Gln (type 2B -- PHEO + ccRCC + HBL; high ccRCC)",
            "p.Arg167Trp (type 2A/2B -- HIF-α binding domain)",
            "p.Pro86Leu (type 2A -- beta domain PHEO cluster)",
            "Large deletion exon 1 (type 1 -- no PHEO; truncating)",
            "p.Arg161Ter (truncating -- type 1)",
        ],
        "clinical_keys": [
            "VHL type predicts PHEO risk: Type 2A/2B/2C have PHEO; Type 1 NO PHEO -- genotype essential",
            "RETINAL HEMANGIOBLASTOMA: annual ophthalmology from age 1; often first presentation; laser/anti-VEGF",
            "ccRCC surveillance: MRI abdomen every 1-2 years from age 15; intervention if >3cm",
            "CNS hemangioblastoma: MRI brain + spine every 1-2 years; symptom-driven surgery",
            "Pancreatic involvement ~70%: MRI pancreas (NET screen); somatostatin receptor if NET suspected",
            "Belzutifan (Welireg) FDA2021: HIF-2α inhibitor -- first non-surgical option for VHL tumours",
            "Phaeochromocytoma surveillance: annual plasma/urine metanephrines from age 5 in type 2",
            "PHEO in VHL usually benign: adrenal, bilateral; malignant <5%; resect if functional or growing",
            "Genetic testing: VHL pathogenic variant in ~80% classic VHL disease; large deletions need MLPA",
        ],
    },
    {
        "gene": "RET",
        "protein": (
            "RET -- 10q11.21 Autosomal-Dominant-GOF -- 1114aa -- Ret-Proto-Oncogene-"
            "Receptor-Tyrosine-Kinase-MEN2A-MEN2B-Medullary-Thyroid-Cancer-100pct-PHEO-50pct-"
            "Prophylactic-Thyroidectomy-Vandetanib-Cabozantinib-OMIM-164761"
        ),
        "locus": "10q11.21",
        "protein_size": (
            "1114 aa / 124 kDa (RET; receptor tyrosine kinase; GDNF family receptor; "
            "STRUCTURE: extracellular cadherin-like domain (ligand/coreceptor binding); "
            "cysteine-rich domain (key GOF cysteines: C634); single transmembrane; "
            "juxtamembrane domain; 2 intracellular kinase lobes (split kinase); C-terminal tail; "
            "FUNCTION: "
            "  RET: receptor for GDNF-family ligands (GDNF, neurturin, artemin, persephin); "
            "  Signalling: GFRα co-receptor + GDNF -> RET dimerisation -> trans-autophosphorylation; "
            "  Downstream: RAS-MAPK, PI3K-AKT, STAT3, PLC-gamma; "
            "  Required for: enteric nervous system development, renal morphogenesis, spermatogenesis; "
            "  MEN2 GOF: gain-of-function mutations -> constitutive RET dimerisation/activation; "
            "MEN2A (codon 634 most common): "
            "  MEDULLARY THYROID CANCER: 100% lifetime risk PATHOGNOMONIC; earliest/most penetrant; "
            "  PHEO: 50% in codon 634 MEN2A; bilateral adrenal; usually benign (5% malignant); "
            "  PHPT: 15-25% in MEN2A; mild hypercalcaemia (contrast MEN1 -- more severe); "
            "  Codon 634 (C634R, C634Y, C634F): highest PHEO risk; earliest thyroid cancer; "
            "  Codon 620 (rare): similar to 634; "
            "PROPHYLACTIC THYROIDECTOMY (MEN2A codon-based): "
            "  Codon 918 (MEN2B): thyroidectomy within 6 months of birth; highest risk category; "
            "  Codon 634 (MEN2A): thyroidectomy by age 5 years; very high risk; "
            "  Other MEN2A codons: by age 5 if calcitonin elevated; otherwise by age 10; "
            "TREATMENT: "
            "  Vandetanib (Caprelsa): RET + VEGFR inhibitor; FDA for MTC; "
            "  Cabozantinib (Cometriq): RET + MET + VEGFR inhibitor; FDA for MTC; "
            "  Selpercatinib (Retevmo): highly selective RET inhibitor; FDA2020; best RR for MTC; "
            "  Pralsetinib (Gavreto): selective RET inhibitor; "
        ),
        "inheritance": (
            "Autosomal dominant GOF; RET 10q11.21; de novo in ~10% MEN2B (codon 918 M918T); "
            "Familial in most MEN2A; strong genotype-phenotype correlation; "
            "ALL MEN2 patients: RET sequencing essential for surgical timing decisions; "
        ),
        "disease_category": "MEN2A / MEN2B (RET GOF) -- MTC + PHEO + PHPT",
        "key_mutations": [
            "p.Cys634Arg (C634R -- MEN2A most common; highest PHEO risk 50%)",
            "p.Cys634Tyr (C634Y -- MEN2A; extracellular cysteine)",
            "p.Met918Thr (M918T -- MEN2B; intracellular; de novo ~10%; highest thyroid risk)",
            "p.Cys620Arg (C620R -- MEN2A; rare)",
            "p.Cys618Arg (C618R -- FMTC/MEN2A)",
            "p.Val804Met (V804M -- FMTC; moderate risk; tyrosine kinase domain)",
            "p.Ala883Phe (A883F -- MEN2B; intracellular kinase)",
        ],
        "clinical_keys": [
            "MEDULLARY THYROID CANCER 100% LIFETIME PATHOGNOMONIC: all RET carriers need thyroidectomy",
            "PROPHYLACTIC THYROIDECTOMY timing by codon: M918T by 6 months; C634 by age 5; others by 10",
            "Codon 634: 50% PHEO risk -- annual plasma metanephrines from age 5; PHEO excluded BEFORE thyroidectomy",
            "PHEO BEFORE SURGERY: always screen for PHEO (plasma metanephrines) before any surgical procedure",
            "MEN2B features: marfanoid habitus + mucosal neuromas (tongue, lips) + corneal nerve thickening PATHOGNOMONIC",
            "Selpercatinib (Retevmo): most selective RET inhibitor FDA2020; first-line for advanced MTC",
            "Calcitonin + CEA: tumour markers for MTC surveillance; doubling time prognostic",
            "PHPT in MEN2A (15-25%): parathyroid hyperplasia; concurrent parathyroidectomy at thyroidectomy if elevated PTH",
            "Genotype phenotype critical: V804M low risk (thyroidectomy can wait to age 10); M918T highest risk (immediate)",
        ],
    },
    {
        "gene": "TMEM127",
        "protein": (
            "TMEM127 -- 2q11.2 Autosomal-Dominant-LOF -- 238aa -- Transmembrane-Protein-127-"
            "mTORC1-Negative-Regulator-Bilateral-Adrenal-PHEO-Low-Malignant-Risk-"
            "IHC-TMEM127-Loss-OMIM-613403"
        ),
        "locus": "2q11.2",
        "protein_size": (
            "238 aa / 25 kDa (TMEM127; transmembrane protein 127; mTOR pathway regulator; "
            "STRUCTURE: 3 putative transmembrane domains; N-terminus cytoplasmic; "
            "localises to late endosomes/lysosomes + plasma membrane; "
            "FUNCTION: "
            "  TMEM127 is a negative regulator of mTORC1 via Rag GTPase pathway; "
            "  TMEM127 LOF -> mTORC1 hyperactivation -> PI3K-AKT-mTOR signalling; "
            "  Links to VHL pathway: mTORC1 promotes HIF-1α translation (VHL prevents HIF-α stability); "
            "  IHC: TMEM127 protein loss in tumour confirms pathogenic variant; "
            "HEREDITARY PHEO (TMEM127): "
            "  Bilateral adrenal PHEO characteristic: common, often synchronous; "
            "  Predominantly adrenal: very rare extra-adrenal PGL; "
            "  Low malignant risk: <5% metastatic; "
            "  Late onset: mean age of diagnosis ~40-50 years (later than SDHx); "
            "  Biochemistry: predominantly epinephrine secreting (adrenal medullary origin); "
            "  Everolimus (mTOR inhibitor): rationale for TMEM127-related PHEO (clinical data limited); "
        ),
        "inheritance": (
            "Autosomal dominant LOF; TMEM127 2q11.2; de novo rare; mostly familial; "
            "Penetrance: ~50% estimated; may be under-ascertained (later onset, bilateral adrenal); "
            "TMEM127: mTOR pathway -- different from SDHx pseudohypoxia mechanism; "
        ),
        "disease_category": "Hereditary PHEO (TMEM127; mTOR pathway)",
        "key_mutations": [
            "p.Glu59Ter (truncating -- LOF)",
            "p.Arg69Ter (truncating)",
            "p.Arg117Trp (transmembrane domain)",
            "p.Tyr35Cys (N-terminal cytoplasmic)",
            "Exon deletions (require MLPA)",
        ],
        "clinical_keys": [
            "Bilateral adrenal PHEO characteristic: plan for bilateral adrenalectomy timing if synchronous",
            "Adrenal predominantly: extra-adrenal PGL very rare -- different from SDHB",
            "Low malignant risk <5%: benign course typical; long-term surveillance standard",
            "Epinephrine-secreting (adrenal medullary): plasma/urine metanephrine elevated",
            "mTOR pathway: everolimus rationale -- clinical studies limited but mechanism-based",
            "IHC TMEM127 loss: confirms pathogenic variant in tumour tissue",
            "Later onset ~40-50 years: TMEM127 carriers need lifelong biochemical surveillance",
            "MLPA mandatory: exonic deletions contribute to TMEM127 pathogenic variants",
            "Annual plasma metanephrines + adrenal MRI every 2-3 years: surveillance protocol",
        ],
    },
    {
        "gene": "MAX",
        "protein": (
            "MAX -- 14q23.3 Autosomal-Dominant-LOF-PATERNAL-IMPRINTING -- 236aa -- "
            "MYC-Associated-Factor-X-bHLH-LZ-Transcription-Factor-Bilateral-Adrenal-PHEO-"
            "Young-Onset-Second-Imprinted-PHEO-Gene-OMIM-154950"
        ),
        "locus": "14q23.3",
        "protein_size": (
            "236 aa / 22 kDa (MAX; MYC-associated factor X; bHLH-LZ transcription factor; "
            "STRUCTURE: basic helix-loop-helix leucine zipper (bHLH-LZ) domain; "
            "dimerises with MYC, MAD, MNT family proteins; obligate heterodimer partner; "
            "FUNCTION: "
            "  MAX is the obligate dimerisation partner of all MYC family proteins (c-MYC, N-MYC, L-MYC); "
            "  MYC-MAX heterodimer: transcriptional activator of MYC target genes (proliferation, apoptosis); "
            "  MAD-MAX / MNT-MAX: transcriptional repressor -- opposes MYC-MAX; "
            "  MAX LOF: disrupts MYC-MAX repression -> net MYC activity increase -> proliferation; "
            "  Mechanism: MAX LOF -> loss of MAD/MNT repression -> MYC-driven chromatin access; "
            "MAX-PHEO: "
            "  PATERNAL IMPRINTING: like SDHD -- disease ONLY if inherited from father; "
            "  This makes MAX the SECOND imprinted gene in hereditary PHEO (after SDHD); "
            "  Bilateral adrenal PHEO: predominant phenotype; synchronous bilateral common; "
            "  Young onset: mean age of diagnosis < 30 years; "
            "  Malignant risk: intermediate ~15-20%; "
            "  Biochemistry: norepinephrine and epinephrine secreting; "
            "  MYC pathway drug targets: BRD4 inhibitors (JQ1) -- investigational; "
        ),
        "inheritance": (
            "Autosomal dominant LOF with PATERNAL IMPRINTING; "
            "MAX 14q23.3; disease ONLY if variant inherited from father (like SDHD); "
            "Maternal MAX inheritance: does NOT cause disease; "
            "TEST BOTH PARENTS: paternal vs maternal inheritance determines recurrence risk; "
            "Paternal: ~50% children affected; Maternal: 0% children affected; "
        ),
        "disease_category": "Hereditary PHEO (MAX; MYC pathway; paternal imprinting)",
        "key_mutations": [
            "p.Arg60Gln (bHLH domain -- MYC binding interface; most common)",
            "p.Pro7Leu (N-terminal; HLH helix 1)",
            "p.Glu37Ter (truncating -- LOF)",
            "p.Tyr70Ter (truncating -- bHLH domain)",
            "p.Arg60Pro (bHLH domain; leucine zipper interface)",
            "Exon deletions (MLPA required)",
        ],
        "clinical_keys": [
            "PATERNAL IMPRINTING LIKE SDHD: MAX disease ONLY if inherited from father -- test both parents",
            "Second imprinted PHEO gene: MAX and SDHD are the TWO imprinted genes in hereditary PHEO",
            "Bilateral adrenal PHEO young onset <30 years: suspect MAX in young bilateral adrenal PHEO",
            "Intermediate malignant risk 15-20%: higher than TMEM127/VHL; lower than SDHB",
            "Norepinephrine and epinephrine secretory: plasma normetanephrine AND metanephrine elevated",
            "Maternal MAX carrier: reassure -- zero risk of disease in children (imprinted locus)",
            "MYC pathway: investigational MYC-pathway inhibitors (BRD4/JQ1) -- no approved agent yet",
            "Annual plasma metanephrines from age 15 if paternal MAX carrier: young onset mandates early start",
            "Cr-SSTR-PET for staging if malignant: MAX tumours show variable SSTR expression",
        ],
    },
]


# ─────────────────────────────────────────────────────
# PATIENT DATA GENERATOR
# ─────────────────────────────────────────────────────

def _generate_patients_for_gene(gene: str, seed: int, n: int = 40) -> list:
    """Generate a deterministic synthetic cohort for one gene."""
    rng = random.Random(seed)
    patients = []

    # Gene-specific phenotypic distributions (clinically accurate)
    gene_profiles = {
        "SDHB": {
            "extra_adrenal_rate": 0.55, "bilateral_rate": 0.15,
            "malignant_rate": 0.37, "functional_rate": 0.80,
            "head_neck_rate": 0.20, "adrenal_rate": 0.45,
            "hypertension_rate": 0.65, "mean_age": 38, "sd_age": 14,
            "mean_diag_age": 40, "sd_diag_age": 13,
        },
        "SDHD": {
            "extra_adrenal_rate": 0.05, "bilateral_rate": 0.55,
            "malignant_rate": 0.05, "functional_rate": 0.18,
            "head_neck_rate": 0.90, "adrenal_rate": 0.08,
            "hypertension_rate": 0.25, "mean_age": 42, "sd_age": 13,
            "mean_diag_age": 44, "sd_diag_age": 13,
        },
        "SDHC": {
            "extra_adrenal_rate": 0.02, "bilateral_rate": 0.20,
            "malignant_rate": 0.04, "functional_rate": 0.10,
            "head_neck_rate": 0.92, "adrenal_rate": 0.05,
            "hypertension_rate": 0.15, "mean_age": 48, "sd_age": 14,
            "mean_diag_age": 50, "sd_diag_age": 14,
        },
        "SDHA": {
            "extra_adrenal_rate": 0.40, "bilateral_rate": 0.20,
            "malignant_rate": 0.12, "functional_rate": 0.70,
            "head_neck_rate": 0.30, "adrenal_rate": 0.55,
            "hypertension_rate": 0.58, "mean_age": 40, "sd_age": 14,
            "mean_diag_age": 42, "sd_diag_age": 14,
            "gist_rate": 0.20,
        },
        "VHL": {
            "extra_adrenal_rate": 0.05, "bilateral_rate": 0.50,
            "malignant_rate": 0.04, "functional_rate": 0.35,
            "head_neck_rate": 0.03, "adrenal_rate": 0.90,
            "hypertension_rate": 0.40, "mean_age": 35, "sd_age": 12,
            "mean_diag_age": 36, "sd_diag_age": 12,
            "rcc_rate": 0.60, "hbl_rate": 0.75, "pnet_rate": 0.45,
            "retinal_hbl_rate": 0.55,
        },
        "RET": {
            "extra_adrenal_rate": 0.02, "bilateral_rate": 0.65,
            "malignant_rate": 0.05, "functional_rate": 0.90,
            "head_neck_rate": 0.02, "adrenal_rate": 0.95,
            "hypertension_rate": 0.75, "mean_age": 38, "sd_age": 15,
            "mean_diag_age": 40, "sd_diag_age": 15,
            "mtc_rate": 1.00, "phpt_rate": 0.20,
        },
        "TMEM127": {
            "extra_adrenal_rate": 0.03, "bilateral_rate": 0.60,
            "malignant_rate": 0.04, "functional_rate": 0.92,
            "head_neck_rate": 0.01, "adrenal_rate": 0.96,
            "hypertension_rate": 0.80, "mean_age": 45, "sd_age": 13,
            "mean_diag_age": 47, "sd_diag_age": 13,
        },
        "MAX": {
            "extra_adrenal_rate": 0.08, "bilateral_rate": 0.55,
            "malignant_rate": 0.17, "functional_rate": 0.88,
            "head_neck_rate": 0.05, "adrenal_rate": 0.88,
            "hypertension_rate": 0.78, "mean_age": 28, "sd_age": 10,
            "mean_diag_age": 29, "sd_diag_age": 10,
        },
    }

    # Per-gene mutations
    gene_mutations = {
        "SDHB": ["p.Asp92Tyr", "p.Pro197Arg", "p.Cys101Tyr", "p.Arg46Gln", "p.Trp200Ter", "ex1_del", "p.Arg230Cys"],
        "SDHD": ["p.Asp92Tyr", "p.His50Arg", "p.Leu95Pro", "c.149G>A", "p.Arg38Ter", "ex3_del", "p.Pro81Leu"],
        "SDHC": ["p.Arg55Trp", "p.Arg97Ter", "p.Gln109Ter", "p.Ser45Phe", "ex2_del"],
        "SDHA": ["p.Arg31Ter", "p.Arg589Gln", "p.Ile107Thr", "p.Glu524Lys", "p.Thr422Ile", "ex4_del"],
        "VHL": ["p.Tyr98His", "p.Asn78Ser", "p.Arg167Gln", "p.Arg167Trp", "p.Pro86Leu", "ex1_del", "p.Arg161Ter"],
        "RET": ["p.Cys634Arg", "p.Cys634Tyr", "p.Met918Thr", "p.Cys618Arg", "p.Val804Met", "p.Ala883Phe", "p.Cys620Arg"],
        "TMEM127": ["p.Glu59Ter", "p.Arg69Ter", "p.Arg117Trp", "p.Tyr35Cys", "ex3_del"],
        "MAX": ["p.Arg60Gln", "p.Pro7Leu", "p.Glu37Ter", "p.Tyr70Ter", "p.Arg60Pro", "ex2_del"],
    }

    profile = gene_profiles.get(gene, gene_profiles["SDHB"])
    mutations = gene_mutations.get(gene, ["p.unknown"])

    for pid in range(1, n + 1):
        age = max(10, int(rng.gauss(profile["mean_age"], profile["sd_age"])))
        diag_age = max(10, int(rng.gauss(profile["mean_diag_age"], profile["sd_diag_age"])))
        sex = rng.choice(["M", "F"])
        mutation = rng.choice(mutations)

        extra_adrenal   = rng.random() < profile["extra_adrenal_rate"]
        bilateral       = rng.random() < profile["bilateral_rate"]
        malignant       = rng.random() < profile["malignant_rate"]
        functional      = rng.random() < profile["functional_rate"]
        head_neck       = rng.random() < profile["head_neck_rate"]
        adrenal         = rng.random() < profile["adrenal_rate"]
        hypertension    = rng.random() < profile["hypertension_rate"]
        gist            = rng.random() < profile.get("gist_rate", 0.0)
        rcc             = rng.random() < profile.get("rcc_rate", 0.0)
        hbl             = rng.random() < profile.get("hbl_rate", 0.0)
        retinal_hbl     = rng.random() < profile.get("retinal_hbl_rate", 0.0)
        pnet            = rng.random() < profile.get("pnet_rate", 0.0)
        mtc             = rng.random() < profile.get("mtc_rate", 0.0)
        phpt            = rng.random() < profile.get("phpt_rate", 0.0)
        paternal_mode   = gene in ("SDHD", "MAX")

        if malignant:
            severity = "severe"
        elif bilateral or functional:
            severity = "moderate"
        else:
            severity = "mild"

        normetanephrine_mult = rng.uniform(2, 12) if functional else rng.uniform(0.8, 1.3)
        metanephrine_mult    = rng.uniform(2, 10) if (functional and adrenal) else rng.uniform(0.8, 1.2)
        dopamine_elevated    = gene == "SDHB" and rng.random() < 0.55

        patients.append({
            "patient_id": f"{gene}-{pid:03d}",
            "age": age,
            "age_at_diagnosis_yrs": diag_age,
            "sex": sex,
            "gene": gene,
            "mutation": mutation,
            "extra_adrenal": extra_adrenal,
            "bilateral": bilateral,
            "malignant": malignant,
            "functional": functional,
            "head_neck_pgl": head_neck,
            "adrenal_pheo": adrenal,
            "hypertension": hypertension,
            "gist": gist,
            "rcc": rcc,
            "hemangioblastoma": hbl,
            "retinal_hbl": retinal_hbl,
            "pnet": pnet,
            "mtc": mtc,
            "phpt": phpt,
            "paternal_imprinting_mode": paternal_mode,
            "severity": severity,
            "normetanephrine_xULN": round(normetanephrine_mult, 2),
            "metanephrine_xULN": round(metanephrine_mult, 2),
            "dopamine_elevated": dopamine_elevated,
            "sstr_pet_positive": rng.random() < (0.90 if not head_neck else 0.95),
            "mibg_positive": rng.random() < (0.65 if not malignant else 0.45),
            "ihc_sdhb_loss": True,  # All SDHx tumours lose SDHB on IHC
            "germline_variant": True,
        })
    return patients


# ─────────────────────────────────────────────────────
# API FUNCTIONS
# ─────────────────────────────────────────────────────

def generate_overview() -> dict:
    """Overview data for Hereditary-PHEO-PGL-SDH-Atlas."""
    return {
        "atlas":          "Hereditary-PHEO-PGL-SDH-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Pheochromocytoma/Paraganglioma/SDH Syndrome Atlas "
            "(SDHB-SDHD-SDHC-SDHA-VHL-RET-TMEM127-MAX)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "SDHB": (
                "AD LOF 1p36.13 (complex II Fe-S subunit; 280aa; PGL4; HIGHEST MALIGNANT RISK 35-40% among all PHEO/PGL genes; "
                "extra-adrenal location 50-60%; Cr-SSTR-PET mandatory; Lutetium-DOTATATE for metastatic; "
                "norepinephrine + dopamine secretory; MLPA required for exonic deletions)"
            ),
            "SDHD": (
                "AD LOF 11q23.1 PATERNAL IMPRINTING UNIQUE (cytochrome b small subunit; 159aa; PGL1; "
                "DISEASE ONLY IF INHERITED FROM FATHER -- unique inheritance; head/neck PGL 90%; bilateral carotid body >50%; "
                "low malignant risk ~5%; non-functional majority; maternal SDHD: NO disease)"
            ),
            "SDHC": (
                "AD LOF 1q23.3 (cytochrome b large subunit; 169aa; PGL3; LOWEST malignant risk <5% among all SDHx; "
                "parasympathetic head/neck PGL; no adrenal PHEO; non-functional >90%; "
                "often underpaneled -- ensure SDHC included in any PGL panel)"
            ),
            "SDHA": (
                "AD LOF 5p15.33 (flavoprotein subunit; 621aa; PGL5; largest complex II subunit; "
                "IHC SDHA LOSS confirms SDHA variant (only SDHx with gene-specific IHC); GIST co-risk 20%; "
                "adrenal + extra-adrenal; intermediate malignant risk 10-15%)"
            ),
            "VHL": (
                "AD LOF 3p25.3 (HIF-2α regulator E3 adaptor; 213aa; VHL disease; "
                "TYPE 1: NO PHEO; ccRCC + hemangioblastomas only; "
                "TYPE 2A: PHEO + HBL; low ccRCC; TYPE 2B: PHEO + ccRCC + HBL; TYPE 2C: PHEO ONLY; "
                "Belzutifan (Welireg) FDA2021 -- first HIF-2α inhibitor; retinal HBL first presentation)"
            ),
            "RET": (
                "AD GOF 10q11.21 (receptor tyrosine kinase; 1114aa; MEN2A/2B; "
                "MEDULLARY THYROID CANCER 100% LIFETIME PATHOGNOMONIC -- thyroidectomy timing by codon; "
                "PHEO 50% in codon 634; EXCLUDE PHEO BEFORE ANY SURGERY; "
                "Selpercatinib (Retevmo) FDA2020 most selective; prophylactic thyroidectomy age-based)"
            ),
            "TMEM127": (
                "AD LOF 2q11.2 (mTOR pathway regulator; 238aa; bilateral adrenal PHEO; "
                "mTORC1 negative regulator; IHC TMEM127 loss confirms; low malignant risk <5%; "
                "later onset ~45-50 years; predominantly adrenal; epinephrine secretory)"
            ),
            "MAX": (
                "AD LOF 14q23.3 PATERNAL IMPRINTING LIKE SDHD (bHLH-LZ TF; 236aa; bilateral adrenal PHEO; "
                "DISEASE ONLY IF INHERITED FROM FATHER -- second imprinted PHEO gene; young onset <30 years; "
                "intermediate malignant risk 15-20%; TEST BOTH PARENTS -- paternal vs maternal completely different risk)"
            ),
        },
        "key_clinical_rules": [
            "SDHB: HIGHEST MALIGNANT RISK (35-40%) -- Cr-SSTR-PET at diagnosis and every 1-2 years; Lutetium-DOTATATE for metastatic",
            "SDHD and MAX: PATERNAL IMPRINTING -- disease ONLY if variant inherited from father; always test both parents",
            "SDHD: head/neck PGL characteristic; maternal SDHD carriers: reassure -- zero disease risk to their children",
            "VHL TYPE determines PHEO risk: Type 1 = NO PHEO; Type 2A/2B/2C = PHEO present; genotype mandatory before surveillance",
            "RET MEN2: EXCLUDE PHEO BEFORE ANY OPERATION (thyroidectomy, parathyroidectomy) -- plasma metanephrines first",
            "RET codon-based prophylactic thyroidectomy: M918T by 6 months; C634 by age 5; others by age 10",
            "SDHA IHC LOSS confirms SDHA variant (SDHA-specific); SDHB IHC loss = any SDHx variant (not SDHA-specific)",
            "ALL SDHx genes: SDHB IHC loss in tumour confirms SDHx pathogenicity (all lose SDHB regardless of which SDHx mutated)",
            "SDHA-GIST risk: screen for synchronous GIST (abdominal imaging) in all SDHA carriers",
            "Biochemistry cascade: plasma metanephrines (screening) -> Cr-SSTR-PET (anatomy/staging) -> MLPA if no point mutation",
            "VHL belzutifan (Welireg) FDA2021: HIF-2α inhibitor -- first systemic option for VHL hemangioblastoma/ccRCC/pNET",
            "Selpercatinib (Retevmo) FDA2020: most selective RET inhibitor for advanced MTC in RET-GOF carriers",
        ],
        "gene_panel_note": (
            "Hereditary PHEO/PGL gene panel (2024): minimum SDHB, SDHD, SDHC, SDHA, VHL, RET, TMEM127, MAX, EPAS1 (HIF2A); "
            "extended panel adds: FH, EGLN1/2 (PHD1/2), KIF1B, ATRX, CSDE1, DNMT3A; "
            "Testing strategy: "
            "  Young (<40) or bilateral or extra-adrenal PHEO: full hereditary panel (SDHx + VHL + RET + TMEM127 + MAX); "
            "  Head/neck PGL: SDHD first (then SDHB, SDHC, SDHA); "
            "  Adrenal bilateral PHEO young: MAX and TMEM127 (paternal inheritance check for MAX); "
            "  MEN2 phenotype (MTC + PHEO): RET sequencing; "
            "  IHC first if tumour available: SDHA + SDHB IHC narrows molecular testing; "
            "Imaging: Cr-SSTR-PET (68Ga-DOTATATE) preferred for staging/surveillance; "
            "Biochemistry: plasma fractionated metanephrines + methoxytyramine (dopamine marker for SDHB); "
            "SDH epigenetic: succinate accumulation causes SDHx-hypermethylator phenotype in tumour; "
            "PATERNAL IMPRINTING: SDHD + MAX -- BOTH require parental testing to establish inheritance mode"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-PHEO-PGL-SDH-Atlas."""
    genes_data = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        seed = SEED_BASE + idx
        patients = _generate_patients_for_gene(gene, seed)
        n = len(patients)

        extra_adrenal_n  = sum(1 for p in patients if p["extra_adrenal"])
        bilateral_n      = sum(1 for p in patients if p["bilateral"])
        malignant_n      = sum(1 for p in patients if p["malignant"])
        functional_n     = sum(1 for p in patients if p["functional"])
        head_neck_n      = sum(1 for p in patients if p["head_neck_pgl"])
        adrenal_n        = sum(1 for p in patients if p["adrenal_pheo"])
        hypertension_n   = sum(1 for p in patients if p["hypertension"])
        gist_n           = sum(1 for p in patients if p["gist"])
        rcc_n            = sum(1 for p in patients if p["rcc"])
        hbl_n            = sum(1 for p in patients if p["hemangioblastoma"])
        retinal_hbl_n    = sum(1 for p in patients if p["retinal_hbl"])
        pnet_n           = sum(1 for p in patients if p["pnet"])
        mtc_n            = sum(1 for p in patients if p["mtc"])
        phpt_n           = sum(1 for p in patients if p["phpt"])
        sstr_pos_n       = sum(1 for p in patients if p["sstr_pet_positive"])
        severe_n         = sum(1 for p in patients if p["severity"] == "severe")
        moderate_n       = sum(1 for p in patients if p["severity"] == "moderate")
        mild_n           = sum(1 for p in patients if p["severity"] == "mild")
        mean_diag_age    = round(sum(p["age_at_diagnosis_yrs"] for p in patients) / n, 1)
        mutations_seen   = list({p["mutation"] for p in patients})

        clinical_notes = {
            "SDHB": "Highest malignant risk 35-40%. Extra-adrenal 50-60%. Cr-SSTR-PET + Lu-DOTATATE for metastatic. MLPA for exonic deletions. Norepinephrine + dopamine secretory.",
            "SDHD": "Paternal imprinting -- disease only if from father. Head/neck PGL 90%. Bilateral carotid body common. Low malignant risk 5%. Non-functional majority.",
            "SDHC": "Lowest malignant risk <5%. Parasympathetic head/neck only. No adrenal PHEO. Watch and wait for asymptomatic. Frequently underpaneled.",
            "SDHA": "SDHA IHC loss confirms variant (unique SDHx marker). GIST co-risk 20%. Intermediate malignant risk 10-15%. Adrenal + extra-adrenal.",
            "VHL": "Type 1/2A/2B/2C determines PHEO risk. Retinal HBL annual ophthalmology. ccRCC MRI. Belzutifan FDA2021 for VHL tumours (ccRCC/HBL/pNET).",
            "RET": "MTC 100% lifetime. Thyroidectomy timing by codon (M918T by 6mo; C634 by age 5). PHEO excluded BEFORE all operations. Selpercatinib most selective.",
            "TMEM127": "Bilateral adrenal PHEO characteristic. Epinephrine secretory. Low malignant risk. mTOR pathway. Later onset ~45 years. IHC TMEM127 loss confirms.",
            "MAX": "Paternal imprinting like SDHD. Bilateral adrenal PHEO young onset <30 years. Intermediate malignant risk 15-20%. Test both parents.",
        }

        genes_data.append({
            "gene":                  gene,
            "locus":                 gene_info["locus"],
            "n":                     n,
            "n_patients":            n,
            "severe_pct":            round(severe_n / n * 100, 1),
            "moderate_pct":          round(moderate_n / n * 100, 1),
            "mild_pct":              round(mild_n / n * 100, 1),
            "extra_adrenal_pct":     round(extra_adrenal_n / n * 100, 1),
            "bilateral_pct":         round(bilateral_n / n * 100, 1),
            "malignant_pct":         round(malignant_n / n * 100, 1),
            "functional_pct":        round(functional_n / n * 100, 1),
            "head_neck_pgl_pct":     round(head_neck_n / n * 100, 1),
            "adrenal_pheo_pct":      round(adrenal_n / n * 100, 1),
            "hypertension_pct":      round(hypertension_n / n * 100, 1),
            "gist_pct":              round(gist_n / n * 100, 1),
            "rcc_pct":               round(rcc_n / n * 100, 1),
            "hemangioblastoma_pct":  round(hbl_n / n * 100, 1),
            "retinal_hbl_pct":       round(retinal_hbl_n / n * 100, 1),
            "pnet_pct":              round(pnet_n / n * 100, 1),
            "mtc_pct":               round(mtc_n / n * 100, 1),
            "phpt_pct":              round(phpt_n / n * 100, 1),
            "sstr_pet_positive_pct": round(sstr_pos_n / n * 100, 1),
            "mean_age_dx_yrs":       mean_diag_age,
            "sample_mutations":      mutations_seen[:4],
            "protein":               gene_info["protein"],
            "inheritance":           gene_info["inheritance"][:220],
            "disease_category":      gene_info["disease_category"],
            "clinical_note":         clinical_notes.get(gene, ""),
        })

    return {
        "atlas": "Hereditary-PHEO-PGL-SDH-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-PHEO-PGL-SDH-Atlas."""
    definitions = [
        {
            "term": "Hereditary-PHEO-PGL-Syndrome-Overview",
            "definition": (
                "Hereditary pheochromocytoma/paraganglioma (PHEO/PGL) syndromes are "
                "autosomal dominant tumour predispositions caused by germline variants in "
                "SDHx (SDHB/SDHD/SDHC/SDHA), VHL, RET, TMEM127, MAX, FH, EPAS1 and others. "
                "Collectively: ~40% of apparently sporadic PHEO/PGL have a germline cause -- "
                "germline testing recommended for ALL PHEO/PGL patients (not just familial). "
                "SDHx mechanism: succinate accumulation inhibits prolyl hydroxylases -> HIF-1α "
                "stabilisation (pseudohypoxia) -> VEGF/EPO upregulation -> tumour angiogenesis. "
                "SDH epigenetics: succinate inhibits TET dioxygenases and JmjC demethylases -> "
                "SDHx-hypermethylator tumour phenotype. "
                "Biochemical diagnosis: plasma fractionated metanephrines (best screening test); "
                "include methoxytyramine for extra-adrenal/SDHB dopamine secretion. "
                "Functional imaging: Cr-SSTR-PET (68Ga-DOTATATE PET/CT) is preferred for staging "
                "and surveillance of hereditary PHEO/PGL (supersedes MIBG for most indications). "
                "IHC cascade: SDHA + SDHB IHC on tumour -> directs genetic testing efficiently."
            ),
        },
        {
            "term": "SDHx-Malignant-Risk-Ranking",
            "definition": (
                "Malignant PHEO/PGL = metastatic disease (distant spread to bone, liver, lung, LN). "
                "Malignant risk varies dramatically by gene -- the MOST IMPORTANT SDHx clinical fact: "
                "SDHB: 35-40% metastatic (HIGHEST) -- most important prognostic factor; "
                "SDHA: 10-15% metastatic (INTERMEDIATE); "
                "MAX: 15-20% metastatic (INTERMEDIATE); "
                "VHL type 2: <5% malignant; "
                "SDHD: ~5% malignant; "
                "SDHC: <5% malignant (LOWEST); "
                "TMEM127: <5% malignant. "
                "Treatment of metastatic disease: "
                "  (1) Lutetium-DOTATATE (Lu-PRRT): for SSTR-avid metastatic (preferred SDHB); "
                "  (2) MIBG therapy (Azedra): for MIBG-avid; SDHB less MIBG-avid than adrenal; "
                "  (3) Sunitinib: anti-VEGFR for SSTR-negative / progressive disease; "
                "  (4) Temozolomide: for SDHx-hypermethylated tumours (MGMT-unmethylated); "
                "Surveillance intensity: SDHB = highest frequency (annual SSTR-PET); "
                "SDHC/SDHD = less intensive (MRI + annual biochemistry sufficient)."
            ),
        },
        {
            "term": "Paternal-Imprinting-SDHD-MAX",
            "definition": (
                "SDHD (PGL1, 11q23.1) and MAX (14q23.3) are the two imprinted genes in "
                "hereditary PHEO/PGL -- UNIQUE mode of inheritance: "
                "DISEASE ONLY IF VARIANT INHERITED FROM FATHER. "
                "Maternal transmission: does NOT cause disease (maternally imprinted locus). "
                "Clinical implication: "
                "  Asymptomatic child of affected individual: ALWAYS determine if variant came from mother or father; "
                "  Paternal transmission: 50% recurrence risk to children -- full surveillance; "
                "  Maternal transmission: 0% disease risk to children -- can reassure; "
                "Testing protocol: "
                "  Step 1: identify pathogenic SDHD or MAX variant in proband; "
                "  Step 2: test BOTH parents; "
                "  Step 3: if mother is carrier (maternally imprinted): NO surveillance needed for children; "
                "  Step 4: if father is carrier (paternally imprinted): full PHEO/PGL surveillance for children; "
                "SDHD head/neck PGL phenotype + MAX bilateral adrenal young-onset phenotype: "
                "Both can be clinically silent until detected -- proactive surveillance essential."
            ),
        },
        {
            "term": "VHL-Disease-Classification-Treatment",
            "definition": (
                "VHL disease type determines PHEO risk and clinical management: "
                "Type 1: no PHEO; truncating/deletion variants; ccRCC + CNS/retinal hemangioblastomas; "
                "Type 2A: PHEO + hemangioblastomas; LOW ccRCC risk; p.Tyr98-associated variants; "
                "Type 2B: PHEO + ccRCC + hemangioblastomas; HIGH ccRCC risk; "
                "Type 2C: PHEO ONLY (rarest); p.Tyr98His almost exclusively; NO other VHL manifestations; "
                "Surveillance by manifestation: "
                "  Retinal hemangioblastoma: annual ophthalmology from age 1 (earliest/first manifestation); "
                "  CNS hemangioblastoma: MRI brain+spine every 1-2 years from age 11; "
                "  ccRCC: MRI abdomen every 1-2 years from age 15; intervention if >3cm; "
                "  PHEO (type 2): annual plasma/urine metanephrines from age 5; "
                "  Pancreatic: MRI abdomen (includes pancreas) simultaneously with renal; "
                "BELZUTIFAN (Welireg) FDA2021: "
                "  First-in-class HIF-2α inhibitor; VHL disease indication; "
                "  Approved for VHL-associated ccRCC, hemangioblastoma, pNET not requiring immediate surgery; "
                "  Mechanism: binds HIF-2α PAS-B -> blocks HIF-2α/ARNT dimerisation -> no transcriptional activation; "
                "  Phase 3 LITESPARK-010 trial: belzutifan vs everolimus in ccRCC."
            ),
        },
        {
            "term": "RET-MEN2-Thyroidectomy-Protocol",
            "definition": (
                "RET MEN2 -- PROPHYLACTIC THYROIDECTOMY timing is codon-specific: "
                "Highest risk (ATA D): p.Met918Thr (codon 918, MEN2B) -- "
                "  Thyroidectomy within first 6 months of life; de novo in ~10%; "
                "  MEN2B phenotype: marfanoid habitus + mucosal neuromas (tongue/lips/eyelids) + corneal nerve thickening; "
                "Very high risk (ATA C): p.Cys634 codons (634R/Y/F), p.Cys618/620 -- "
                "  Thyroidectomy by age 5 years; PHEO risk 50% for codon 634; "
                "High risk (ATA B): most other pathogenic MEN2A codons -- "
                "  Thyroidectomy by age 5-10 depending on calcitonin; "
                "Moderate risk (ATA A): p.Val804Met (V804M) -- "
                "  Thyroidectomy can be deferred to age 10 if calcitonin normal; PHEO risk low; "
                "PHEO EXCLUSION BEFORE ALL SURGERY: "
                "  Plasma fractionated metanephrines must be normal before thyroidectomy/parathyroidectomy; "
                "  Unrecognised PHEO during surgery: hypertensive crisis/death risk; "
                "MEN2A associated conditions: "
                "  PHPT: 15-20% (parathyroid hyperplasia); concurrent parathyroidectomy if PTH elevated; "
                "  Cutaneous lichen amyloidosis: scapular skin lesion; "
                "  Hirschsprung disease: rare association; "
                "Treatment of advanced MTC: "
                "  Selpercatinib (Retevmo) FDA2020: most selective RET inhibitor; best ORR ~79%; "
                "  Vandetanib (Caprelsa): RET + VEGFR2; FDA2011; "
                "  Cabozantinib (Cometriq): RET + MET + VEGFR2; FDA2012."
            ),
        },
        {
            "term": "IHC-SDH-Testing-Cascade",
            "definition": (
                "IHC (immunohistochemistry) for SDHx should be performed on ALL PHEO/PGL tumour tissue: "
                "Step 1 -- SDHB IHC: "
                "  SDHB protein loss in tumour = SDHx variant (SDHB, SDHD, SDHC OR SDHA pathogenic); "
                "  SDHB loss: indicates that ANY SDHx subunit is mutated; ALL lose SDHB IHC (SDHB degraded); "
                "  SDHB preserved: SDHx unlikely; direct to VHL, RET, TMEM127, MAX, FH, EPAS1; "
                "Step 2 -- SDHA IHC (if SDHB lost): "
                "  SDHA loss: confirms SDHA variant specifically (SDHA is the ONLY SDHx with gene-specific IHC); "
                "  SDHA preserved: SDHB, SDHD or SDHC variant -- proceed to sequence all three; "
                "Interpretation summary: "
                "  SDHB lost + SDHA lost = SDHA variant confirmed; test SDHA gene; "
                "  SDHB lost + SDHA preserved = SDHB/SDHD/SDHC variant; sequence SDHB first (highest impact); "
                "  SDHB preserved = non-SDHx; panel VHL/RET/TMEM127/MAX; "
                "Limitations: "
                "  False-negative SDHB IHC reported (equivocal staining); genetic testing still recommended; "
                "  SDHB IHC technical quality important: FFPE fixation affects antigenicity; "
                "MLPA: mandatory for all SDHx if point mutation not found; exonic deletions 10-15% of SDHB pathogenic variants."
            ),
        },
        {
            "term": "Biochemical-Surveillance-PHEO-PGL",
            "definition": (
                "Biochemical diagnosis and surveillance for hereditary PHEO/PGL: "
                "Gold standard: PLASMA FRACTIONATED METANEPHRINES (sensitivity ~97% for adrenal PHEO): "
                "  Metanephrine (MN): epinephrine metabolite -- adrenal medullary; elevated in RET/TMEM127/adrenal SDHB; "
                "  Normetanephrine (NMN): norepinephrine metabolite -- sympathetic/extra-adrenal; elevated in most PHEO; "
                "  Methoxytyramine (MTY): dopamine metabolite -- KEY MARKER for SDHB extra-adrenal; "
                "SDHB: elevated NMN + MTY combination highly specific for SDHB metastatic; "
                "SDHD/SDHC: often non-functional -- normetanephrine may be normal; MRI is primary surveillance; "
                "RET: predominantly epinephrine (adrenal) -- elevated metanephrine; "
                "VHL: mixed epinephrine/norepinephrine secretion; "
                "Functional imaging hierarchy: "
                "  Cr-SSTR-PET (68Ga-DOTATATE PET/CT): PREFERRED for staging + surveillance (superior to MIBG); "
                "  MIBG scintigraphy/SPECT: when SSTR-PET unavailable; less sensitive for SDHB extra-adrenal; "
                "  FDG-PET: for SDHB malignant/metastatic (more FDG-avid when less SSTR-avid); "
                "Annual surveillance protocol for hereditary PHEO/PGL carriers: "
                "  Plasma metanephrines + methoxytyramine annually; "
                "  Cr-SSTR-PET every 1-2 years (more frequent for SDHB malignant risk); "
                "  MRI abdomen/chest if extra-adrenal risk (SDHB); "
                "  MRI neck annually for SDHD/SDHC head/neck PGL."
            ),
        },
        {
            "term": "SDHB-Metastatic-Treatment-Algorithm",
            "definition": (
                "Metastatic SDHB PHEO/PGL treatment algorithm (2024): "
                "Assessment: Cr-SSTR-PET staging (grade SSTR uptake) + FDG-PET + MRI; "
                "Biochemistry: plasma metanephrines + methoxytyramine (dopamine biomarker); "
                "MIBG avidity assessment (if SSTR-PET equivocal): "
                "  SSTR-avid (most SDHB): Lutetium-DOTATATE (Lu-177-DOTATATE; Lutathera); "
                "  SSTR-negative: Sunitinib (anti-VEGFR) or CVD chemotherapy (cyclophosphamide/vincristine/dacarbazine); "
                "Lutetium-DOTATATE (Lu-PRRT): "
                "  177Lu-DOTATATE delivers targeted radiation to SSTR-expressing cells; "
                "  NETTER-1 trial: 79% ORR vs 3% for octreotide; established for NET; off-label PHEO/PGL; "
                "  PHEO/PGL response: 30-40% partial response; used at experienced centres; "
                "Sunitinib: "
                "  VEGFR1/2/3, PDGFR, RET inhibitor; approved for renal, GIST, pNET; "
                "  Phase 3 FIRSTMAPPP: sunitinib vs placebo in metastatic PHEO/PGL -- positive PFS; "
                "Temozolomide: "
                "  For SDHx-hypermethylated tumours with MGMT methylation; "
                "Emerging: HIF-2α inhibitors (belzutifan) -- rationale for succinate-driven HIF activation; "
                "Surgical debulking: consider for isolated/oligometastatic disease; "
                "All treatment decisions: multidisciplinary team at specialist centre mandatory."
            ),
        },
    ]

    return {
        "atlas": "Hereditary-PHEO-PGL-SDH-Atlas",
        "count": len(definitions),
        "definitions": definitions,
    }
