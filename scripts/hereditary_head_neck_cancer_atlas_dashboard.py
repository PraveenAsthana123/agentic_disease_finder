#!/usr/bin/env python3
"""Hereditary-Head-&-Neck-Cancer-Predisposition-Atlas — Complete 8-Gene Hereditary Head & Neck Cancer Atlas
FANCA  (Fanconi Anemia Complementation Group A; 1455aa; 16q24.3; AR LOF;
         Fanconi Anemia type A — most common FA (60-70%);
         Oral/oropharyngeal SCC risk ~700x general population PATHOGNOMONIC;
         Median HNC age 26yr (vs 62yr sporadic); BMT for aplastic anaemia/AML;
         AVOID RADIATION ABSOLUTELY — lethal radiosensitivity in FA;
         seed SEED_BASE+0) ·
XPC    (Xeroderma Pigmentosum Complementation Group C; 940aa; 3p25.1; AR LOF;
         XP-C — most common XP globally; no neurological involvement;
         UV-induced SCC/BCC face/scalp/lip PATHOGNOMONIC — onset childhood;
         SCC risk 10,000x general population in sun-exposed HNC sites;
         Strict UV avoidance + annual skin surveillance MANDATORY;
         seed SEED_BASE+1) ·
TP53   (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome;
         HNC documented in LFS — laryngeal, oral, oropharyngeal;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually;
         seed SEED_BASE+2) ·
CDKN2A (Cyclin-Dependent Kinase Inhibitor 2A; 156aa p16/INK4A; 9p21.3; AD LOF;
         Familial Atypical Multiple Mole Melanoma (FAMM/FAMMM);
         Oral/oropharyngeal SCC 10-30x; laryngeal SCC elevated;
         Pancreatic cancer 20-30x PATHOGNOMONIC in FAMM context;
         Melanoma 25-36x lifetime risk; founder mutations population-specific;
         seed SEED_BASE+3) ·
ATM    (Ataxia Telangiectasia Mutated; 3056aa; 11q22.3; AR biallelic/AD monoallelic;
         Ataxia Telangiectasia — biallelic;
         Cerebellar ataxia + conjunctival telangiectasias PATHOGNOMONIC biallelic;
         Radiosensitivity ABSOLUTE in biallelic AT — RT at standard doses → lethal;
         HNC 3-5x monoallelic; head & neck SCC in AT survivors;
         seed SEED_BASE+4) ·
MSH2   (MutS Homolog 2; 934aa; 2p21; AD LOF;
         Lynch Syndrome type 2 / Muir-Torre Syndrome;
         Sebaceous adenomas/carcinomas of face/scalp/eyelid = PATHOGNOMONIC Muir-Torre;
         Keratoacanthomas of head/neck area elevated;
         EPCAM 3-prime deletion silences MSH2 — check EPCAM;
         seed SEED_BASE+5) ·
BRCA2  (Breast Cancer Gene 2; 3418aa; 13q12.3; AD LOF;
         Hereditary Breast & Ovarian Cancer (HBOC);
         HNC risk 2-3x elevated; oropharyngeal SCC association;
         PARP inhibitor olaparib FDA-approved; Fanconi Anemia type D1 biallelic;
         seed SEED_BASE+6) ·
RECQL4 (RecQ Like Helicase 4; 1208aa; 8q24.12; AR biallelic LOF;
         Rothmund-Thomson Syndrome;
         Poikiloderma congenitale PATHOGNOMONIC — reticulated skin pigmentation;
         SCC of face/head/neck; osteosarcoma childhood;
         AML association; photosensitivity UV;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3158-3165)
"""
import random

SEED_BASE = 3158

ATLAS_GENES = [
    {
        "gene": "FANCA",
        "protein": (
            "FANCA -- 16q24.3 Autosomal-Recessive-LOF -- 1455aa -- "
            "Fanconi-Anemia-Complementation-Group-A-163kDa-"
            "FA-A-Most-Common-60-70pct-HNC-700x-PATHOGNOMONIC-Oral-Oropharyngeal-SCC-Age26-"
            "AVOID-RADIATION-ABSOLUTELY-BMT-OMIM-607139"
        ),
        "locus": "16q24.3",
        "protein_size": (
            "1455 aa / 16q24.3 FANCA encodes Fanconi Anemia complementation group A protein (FANCA): "
            "STRUCTURE: "
            "  Large nuclear/cytoplasmic scaffold protein — 163 kDa; "
            "  N-terminal ARM-repeat domain: protein-protein interaction (FANCG, FANCC); "
            "  Multiple HEAT repeats: DNA damage response scaffolding; "
            "  C-terminal nuclear localisation sequences; "
            "  No catalytic activity — scaffolding/assembly for FA core complex; "
            "FANCONI ANEMIA CORE COMPLEX: "
            "  FANCA-FANCG-FAAP20-FAAP100-FANCB-FANCL-FANCC-FANCE-FANCF → E3 ubiquitin ligase; "
            "  Core complex monoubiquitinates FANCD2-FANCI (ID2 complex) at DNA damage site; "
            "  Monoubiquitinated ID2 recruits downstream repair factors (FANCJ, FANCN/PALB2, FANCD1/BRCA2); "
            "  DNA interstrand crosslink (ICL) repair — essential; "
            "  Without FANCA: core complex non-functional → FANCD2 not ubiquitinated → ICL unrepaired → chromosomal instability; "
            "FANCONI ANEMIA (FA-A): "
            "  Most common FA complementation group (60-70% of all FA); "
            "  Biallelic FANCA pathogenic variants → FA-A; "
            "  Hematological: aplastic anaemia median onset age 7yr; AML risk (30% by age 40yr); MDS; "
            "  Congenital anomalies: "
            "    VACTERL-H: vertebral, anorectal, cardiac, tracheoesophageal, renal, limb, hydrocephalus; "
            "    Radial ray defects: absent/hypoplastic thumb/radius — SHORT-LIMB PATHOGNOMONIC (not universal); "
            "    Café au lait macules; "
            "    Microphthalmia, micropenis, hypospadias; "
            "HEAD & NECK SQUAMOUS CELL CARCINOMA — PATHOGNOMONIC: "
            "  Oral cavity (floor of mouth, tongue, gingiva), oropharynx, larynx, hypopharynx; "
            "  FA-HNC risk: ~700x general population — highest single-gene HNC predisposition; "
            "  Age at HNC onset: median 26yr (vs 62yr sporadic HNC); "
            "  HPV-negative HNC predominates in FA (unlike sporadic oropharyngeal SCC which is HPV-positive); "
            "  Annual ENT surveillance from age 18 (or earlier if BMT): oral mucosal inspection, nasopharyngoscopy; "
            "  Post-BMT HNC risk HIGHER: graft-versus-host disease → mucosal damage → carcinogenesis; "
            "AVOID RADIATION ABSOLUTELY: "
            "  FA cells hypersensitive to crosslinking agents AND ionising radiation; "
            "  Standard RT dose → severe mucosal necrosis + tissue breakdown + secondary tumours; "
            "  HNC treatment in FA: surgery first; reduced-dose mitomycin-based chemotherapy only; "
            "  RT-associated mortality documented in FA-HNC: AVOID at all costs; "
            "  Modified chemo (mitomycin C + 5-FU at 25-50% dose reduction) used in FA-HNC; "
            "BMT/HSCT: "
            "  Curative for aplastic anaemia component; "
            "  DOES NOT REDUCE HNC RISK — may increase it (mucositis → carcinogenesis); "
            "  Post-BMT FA patients: higher cumulative HNC incidence; "
            "  Conditioning: use fludarabine-based reduced-intensity (avoid cyclophosphamide + RT); "
            "SURVEILLANCE: "
            "  CBC every 3-6 months (aplastic anaemia/AML monitoring); "
            "  Annual bone marrow biopsy from age 7yr; "
            "  Annual ENT + oral mucosal examination from age 18 (or post-BMT); "
            "  Annual dermatology (skin SCC); "
            "  Breast USS/MRI annually from age 25 in female FA carriers (monoallelic = BRCA2-like risk)"
        ),
        "inheritance": (
            "AR biallelic LOF 16q24.3 — FANCA. Most common FA group. "
            "Biallelic pathogenic variants required for FA phenotype. "
            "Monoallelic carriers: no FA phenotype but possible mild HNC risk elevation. "
            "De novo pathogenic variants rare — compound heterozygosity common. "
            "Prevalence: FA overall ~1 per 130,000-160,000 live births; carrier frequency ~1 per 300-500. "
            "FANCA founder mutations: "
            "  Afrikaner South African: c.295C>T (p.Arg99Trp) — high frequency; "
            "  Spanish/Portuguese Roma (Gitano): deletion exons 1-12 (IVS4); "
            "  Israeli/Moroccan Jewish: c.3788_3790del (p.Leu1264del)."
        ),
        "surveillance_key": (
            "CBC 3-6 monthly (aplastic anaemia/AML); annual BM biopsy from age 7yr; "
            "annual ENT + oral mucosa from age 18 / post-BMT (HNC 700x risk); "
            "AVOID ALL RADIATION ABSOLUTELY; reduced-dose chemotherapy (mitomycin C + 5-FU 25-50% dose); "
            "HSCT for aplastic anaemia — fludarabine conditioning; annual dermatology (skin SCC)"
        ),
        "pathognomonic": (
            "Oral/oropharyngeal SCC in young adult <35yr without heavy smoking/HPV = FA PATHOGNOMONIC workup; "
            "Radial ray defects (absent/hypoplastic thumb) + cafe au lait + aplastic anaemia = FA pattern; "
            "Chromosomal breakage assay (diepoxybutane / mitomycin C) = DIAGNOSTIC gold standard for FA"
        ),
    },
    {
        "gene": "XPC",
        "protein": (
            "XPC -- 3p25.1 Autosomal-Recessive-LOF -- 940aa -- "
            "Xeroderma-Pigmentosum-C-XPC-Protein-106kDa-"
            "XP-C-Most-Common-XP-No-Neurological-UV-SCC-Face-10000x-PATHOGNOMONIC-OMIM-278720"
        ),
        "locus": "3p25.1",
        "protein_size": (
            "940 aa / 3p25.1 XPC encodes xeroderma pigmentosum complementation group C protein (XPC): "
            "STRUCTURE: "
            "  N-terminal transglutaminase-like domain: ubiquitin binding; "
            "  Central BHD1-BHD2-BHD3 (beta-hairpin domains): DNA damage recognition via beta-hairpin insertion; "
            "  C-terminal TFIIH-binding domain: essential for NER initiation; "
            "  XPC forms complex with RAD23B (HR23B) + CETN2 for stability; "
            "NUCLEOTIDE EXCISION REPAIR (NER): "
            "  XPC-RAD23B is the PRIMARY damage sensor for global-genome NER (GG-NER); "
            "  Recognises helix-distorting lesions: UV-induced cyclobutane pyrimidine dimers (CPD), 6-4 photoproducts; "
            "  Bulky adducts, interstrand crosslinks (some); "
            "  XPC binding → recruits TFIIH → XPA/RPA → dual incision (XPF-ERCC1 5-prime; XPG 3-prime) → 26-30nt excision → DNA pol delta/epsilon + ligase; "
            "XP-C — XERODERMA PIGMENTOSUM: "
            "  Most common XP globally (especially in Europe, North Africa); "
            "  NEUROLOGICAL: NOT AFFECTED (unlike XP-A, XP-D) — key clinical distinction; "
            "  No cerebellar ataxia, no peripheral neuropathy, no hearing loss; "
            "UV HYPERSENSITIVITY — EARLIEST SIGN: "
            "  Severe sunburn on minimal sun exposure in infancy (first sun exposure); "
            "  Photophobia; keratoconjunctivitis; corneal opacification if unprotected; "
            "HEAD & NECK SCC PATHOGNOMONIC: "
            "  Freckling of face from age 1-2yr; actinic keratoses by age 5yr; "
            "  SCC/BCC of face, scalp, lip, eyelids, tongue: onset median age 8yr (vs 60yr general population); "
            "  SCC risk: ~10,000x general population for UV-exposed sites; "
            "  Eyes: pterygium, corneal SCC; "
            "  Without strict sun protection: median cancer-free survival <20yr; "
            "  With strict UV avoidance: median cancer-free survival extended to 40-50yr; "
            "UV AVOIDANCE MANDATORY: "
            "  UV-protective clothing (UPF 50+) from head to toe outdoors; "
            "  UV-blocking wraparound glasses; "
            "  Avoid outdoors 10am-4pm; "
            "  UV-filtering window film in home + car; "
            "  SPF 50+ broad-spectrum sunscreen, re-apply 2-hourly; "
            "  UV-protective surgical suite lighting in dental/medical procedures; "
            "SURVEILLANCE: "
            "  Monthly full-skin + oral mucosal self-examination; "
            "  Dermatology every 3 months (XP specialist if possible); "
            "  Annual ophthalmology (corneal SCC, pterygium); "
            "  Annual head/neck exam (otolaryngology); "
            "  Consider vismodegib/sonidegib for BCC burden (Hh pathway XP); "
            "  Imiquimod topical for actinic keratoses; "
            "TREATMENT: excision with narrow margins (wide excision field = more sun damage); "
            "  Oral isotretinoin (retinoid): field suppression (limited evidence); "
            "  Consider nicotinamide supplementation (NAD+ NER cofactor — emerging)"
        ),
        "inheritance": (
            "AR biallelic LOF 3p25.1 — XPC. Biallelic required. "
            "Monoallelic carriers: no XP phenotype, possible mild cancer risk. "
            "Prevalence: XP all groups ~1 per 250,000 (Europe) to 1 per 40,000 (Japan). "
            "XPC-specific prevalence: ~1/3 of all XP in Europe; most common XP group globally. "
            "Founder mutations: "
            "  North African (Algerian/Moroccan): exon 9 deletion; "
            "  Japanese: IVS3+1G>A splice site; "
            "  European: c.1735del compound heterozygous variants."
        ),
        "surveillance_key": (
            "Dermatology every 3 months (skin/head SCC); annual ophthalmology (corneal SCC); "
            "strict UV avoidance MANDATORY (UPF 50+ clothing, UV-blocking glasses, window film); "
            "monthly self-examination; annual ENT (oral/oropharyngeal SCC); "
            "consider retinoid field suppression; no neurological surveillance required (XP-C spared)"
        ),
        "pathognomonic": (
            "Severe freckling + SCC/BCC of face/scalp before age 10yr = XP PATHOGNOMONIC; "
            "Multiple skin cancers on sun-exposed head/neck sites before age 20yr = XP pattern; "
            "UV complementation group test (cell fusion assay) or XPC molecular analysis confirms"
        ),
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "Tumour-Protein-p53-Guardian-Genome-43kDa-"
            "Li-Fraumeni-Syndrome-HNC-Laryngeal-Oral-AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 17p13.1 TP53 encodes p53 (tumour protein p53): "
            "STRUCTURE: "
            "  N-terminal transactivation domain 1 (TAD1, aa 1-40): MDM2-interaction; "
            "  TAD2 (aa 40-67): second subdomain; "
            "  Proline-rich region (aa 67-98): apoptosis regulation; "
            "  Central DNA-binding domain (DBD, aa 102-292): hotspot mutations R175H, G245S, R248Q/W, R273H/C, R282W; "
            "  Tetramerisation domain (TD, aa 323-356): functional homotetramers; "
            "  C-terminal regulatory domain (aa 356-393): lysine acetylation/ubiquitination; "
            "LFS AND HEAD & NECK CANCER: "
            "  Li-Fraumeni Syndrome (LFS): germline TP53 heterozygous pathogenic variant; "
            "  HNC in LFS: laryngeal SCC, oral SCC, oropharyngeal SCC — documented, less common than sarcoma/brain; "
            "  HNC in LFS often radiation-induced (second primary after RT for other LFS tumour); "
            "  Squamous cell carcinoma of head/neck: ~5% of LFS cancers; "
            "  LFS cancer spectrum: sarcoma, brain, breast, adrenocortical, leukaemia — HNC secondary; "
            "RADIATION — ABSOLUTE CONTRAINDICATION: "
            "  Standard RT in germline TP53 carriers: "
            "    RT-induced sarcoma in radiation field — multiple documented LFS cases; "
            "    Osteosarcoma, rhabdomyosarcoma in irradiated head/neck field post-RT for HNC; "
            "    AVOID ALL RT regardless of indication; "
            "    Proton therapy: still ionising radiation — NOT safe in LFS; "
            "  HNC treatment in LFS: surgery first; chemotherapy (CDDP/5-FU); no RT; "
            "SURVEILLANCE — TORONTO WBMRI PROTOCOL: "
            "  WBMRI annually (Toronto protocol); brain MRI 6-monthly in children; "
            "  Breast MRI annually from age 20-25; abdominal USS 6-monthly (ACC); "
            "  Annual head/neck surveillance (HNC secondary prevention)"
        ),
        "inheritance": (
            "AD LOF 17p13.1 — TP53. LFS. "
            "Penetrance: >90% lifetime cancer risk. De novo in 7-20%. "
            "Dominant negative (mutant p53 oligomerises with WT) + haploinsufficiency. "
            "Prevalence: ~1 per 5,000-20,000 (LFS). "
            "No phenotype-free carrier state — all germline TP53 carriers at high risk."
        ),
        "surveillance_key": (
            "WBMRI annually (Toronto protocol); brain MRI 6-monthly children; "
            "breast MRI from age 20-25; abdominal USS 6-monthly (ACC); "
            "AVOID ALL IONISING RADIATION ABSOLUTELY; "
            "annual head/neck ENT exam (HNC secondary prevention in LFS); "
            "HNC treatment: surgery + chemo only, no RT"
        ),
        "pathognomonic": (
            "Radiation-induced HNC in young adult post-RT for prior LFS tumour = TP53 germline pattern; "
            "Multiple primaries (sarcoma + HNC, or brain + HNC) = LFS spectrum; "
            "Adrenocortical carcinoma age <5yr = PATHOGNOMONIC LFS (test TP53 germline immediately)"
        ),
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 Autosomal-Dominant-LOF -- 156aa-p16-132aa-p14 -- "
            "p16-INK4A-CDK4/6-Inhibitor-Rb-Pathway-AND-p14-ARF-MDM2-p53-Pathway-"
            "FAMM-Oral-Oropharyngeal-SCC-10-30x-Pancreatic-20x-Melanoma-25-36x-PATHOGNOMONIC-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa (p16/INK4A) / 132 aa (p14/ARF) / 9p21.3 CDKN2A encodes two proteins via alternative reading frames: "
            "p16/INK4A (156aa): "
            "  CDK4/CDK6 inhibitor — binds CDK4/6 → prevents cyclin D binding → Rb hypophosphorylation → G1 arrest; "
            "  Tumour suppressor: loss of p16 → unchecked CDK4/6 → Rb phosphorylation → E2F release → S phase entry; "
            "p14/ARF (132aa): "
            "  Uses exon 1β + exon 2 of CDKN2A (different reading frame from p16); "
            "  Binds MDM2 → sequesters MDM2 in nucleolus → prevents MDM2-mediated p53 degradation → p53 stabilised; "
            "  Indirect p53 activator; "
            "FAMILIAL ATYPICAL MULTIPLE MOLE MELANOMA (FAMM / FAMMM): "
            "  Germline CDKN2A LOF → melanoma (25-36x general population); "
            "  Multiple atypical (dysplastic) naevi PATHOGNOMONIC phenotypic marker; "
            "  Melanoma onset: median 40yr (vs 60yr sporadic); bilateral primary melanomas common; "
            "HEAD & NECK SCC — CDKN2A: "
            "  Oral cavity SCC (floor of mouth, tongue, lip): 10-30x elevated in CDKN2A carriers; "
            "  Oropharyngeal SCC: elevated, especially HPV-negative; "
            "  Laryngeal SCC: some evidence of elevation; "
            "  CDKN2A somatic deletion/methylation: most common alteration in sporadic HNC (~60-80% of HNC); "
            "  Germline CDKN2A → oral SCC with earlier onset and lower tobacco/HPV burden than sporadic; "
            "PANCREATIC CANCER — PATHOGNOMONIC IN FAMM CONTEXT: "
            "  Lifetime risk: 20-30x (17-39% cumulative) in CDKN2A germline carriers with family history; "
            "  Pancreatic ductal adenocarcinoma (PDAC): median age 54yr germline vs 65yr sporadic; "
            "  PDAC in FAMM + melanoma family = CDKN2A germline until excluded; "
            "  Annual pancreatic MRI/EUS from age 40 (CAPS guidelines); "
            "FOUNDER MUTATIONS: "
            "  Netherlands: p.Ala148Thr (c.442G>A) — 70% of Dutch FAMM families; "
            "  Australian/UK: p.Arg24Pro; p.Met53Ile; "
            "  Swedish/Nordic: 19bp deletion exon 2; "
            "  Italian: p.Gly101Trp (c.301G>T); "
            "CDK4/6 INHIBITORS (palbociclib/ribociclib/abemaciclib): "
            "  CDK4/6 inhibitors exploit CDKN2A loss — tumours without p16 depend on CDK4/6 activity; "
            "  FDA-approved for HR+ HER2- breast cancer; trials for CDKN2A-deficient HNC; "
            "  Palbociclib + cetuximab trials in p16-negative HNC ongoing"
        ),
        "inheritance": (
            "AD LOF 9p21.3 — CDKN2A (p16/INK4A + p14/ARF). FAMM syndrome. "
            "Penetrance: melanoma 25-36x; HNC 10-30x; pancreatic 20x. "
            "Variable penetrance — smoking, UV exposure modifies risk. "
            "De novo germline: ~10-15% of FAMM probands. "
            "Prevalence: germline CDKN2A ~1 per 1,000-5,000 population; FAMM families: 1 per 20,000."
        ),
        "surveillance_key": (
            "Annual full-body skin surveillance + oral/oropharyngeal exam (HNC 10-30x); "
            "annual pancreatic MRI/EUS from age 40 (PDAC 20x); "
            "monthly melanoma self-check; dermatology every 3-6 months; "
            "smoking cessation ABSOLUTE (multiplies HNC risk); "
            "CDK4/6 inhibitors emerging for CDKN2A-deficient HNC"
        ),
        "pathognomonic": (
            "Multiple atypical naevi (>50) + melanoma + family history = FAMM pattern → CDKN2A test; "
            "Oral/oropharyngeal SCC + melanoma in same individual or family = CDKN2A germline; "
            "PDAC + melanoma in same family = CDKN2A germline (Familial Pancreatic + Melanoma syndrome)"
        ),
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Autosomal-Recessive-biallelic-AD-monoallelic -- 3056aa -- "
            "ATM-Kinase-350kDa-PI3K-Like-"
            "Ataxia-Telangiectasia-Biallelic-Radiosensitivity-PATHOGNOMONIC-HNC-3-5x-Monoallelic-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 11q22.3 ATM encodes ATM serine-threonine kinase (Ataxia Telangiectasia Mutated): "
            "STRUCTURE: "
            "  N-terminal HEAT repeats (aa 1-1400): protein-protein interaction scaffold; "
            "  FAT domain (aa 1966-2566): ATM activation; "
            "  PI3K-kinase domain (aa 2712-2962): serine-threonine kinase (PI3K-like); "
            "  FATC domain (C-terminal): essential for kinase activity; "
            "  ATM dimer (inactive) → DSB → autophosphorylation → active monomer; "
            "DNA DAMAGE RESPONSE: "
            "  Activated by double-strand breaks (DSB); "
            "  Phosphorylates H2AX (γH2AX — DSB marker); Chk1/Chk2 (cell cycle arrest); BRCA1; p53; RPA2; "
            "  Homologous recombination repair coordination; "
            "  Without ATM: DSBs unrepaired → chromosome breaks → radiosensitivity; "
            "ATAXIA TELANGIECTASIA (A-T) — BIALLELIC: "
            "  Cerebellar ataxia onset 1-4yr (gait unsteadiness) PATHOGNOMONIC; "
            "  Conjunctival/cutaneous telangiectasias PATHOGNOMONIC (appear 3-6yr, conjunctiva first); "
            "  Progressive neurological deterioration — wheelchair-bound by teens; "
            "  Immune deficiency: IgA/IgG subclass deficiency; recurrent sinopulmonary infections; "
            "  Cancer risk: ALL/NHL/Hodgkin's 70-100x; HNC elevated in A-T survivors; "
            "RADIOSENSITIVITY — ABSOLUTE IN BIALLELIC A-T: "
            "  ATM-null cells: severe hypersensitivity to ionising radiation; "
            "  Standard RT dose (60 Gy) in A-T patient → tissue necrosis + lethal radiation injury; "
            "  Even diagnostic CT scans: use with caution (preference for MRI/USS in A-T); "
            "  AT diagnosis: test before ANY RT planned; "
            "  RT-induced HNC risk in A-T survivors; "
            "MONOALLELIC ATM — CANCER RISK: "
            "  ~1% population are ATM monoallelic carriers; "
            "  Breast cancer 4x (lifetime ~33%), pancreatic 5x, gastric 4x; "
            "  HNC 3-5x elevated — oral, laryngeal, oropharyngeal SCC; "
            "  Moderate radiosensitivity (less than biallelic but still relevant); "
            "MANAGEMENT: "
            "  Avoid RT in biallelic AT ABSOLUTELY; "
            "  Monoallelic: standard cancer surveillance per NCI/NCCN; no routine RT avoidance; "
            "  ATRT therapy in A-T: ceralasertib (AZD6738) — ATR inhibitor; "
            "  Clinical trials: ATR inhibitors + DNA damage agents in AT-like tumours"
        ),
        "inheritance": (
            "Biallelic AR LOF: A-T syndrome. Monoallelic AD: elevated cancer risk (incomplete penetrance). "
            "Prevalence biallelic A-T: ~1 per 40,000-100,000 live births. "
            "Monoallelic carrier frequency: ~1 per 100 population. "
            "Founder mutations: "
            "  Ashkenazi Jewish: c.7271T>G (p.Val2424Gly) frequent; "
            "  Norwegian/Nordic: c.6095G>A + del 9 kb; "
            "  Polish: IVS10-6T>G; "
            "  Amish: c.4G>A (p.Asp2Asn)."
        ),
        "surveillance_key": (
            "Biallelic A-T: AVOID ALL RT ABSOLUTELY; use MRI over CT where possible; "
            "CBC + immunoglobulins annually (immune deficiency); IVIg if hypogammaglobulinaemia; "
            "annual WBMRI (lymphoma/leukaemia/HNC risk); "
            "Monoallelic: breast MRI annually from age 40; pancreatic MRI/EUS age 50+; "
            "annual ENT exam (HNC 3-5x risk); smoking cessation absolute"
        ),
        "pathognomonic": (
            "Cerebellar ataxia + conjunctival telangiectasias in child = A-T PATHOGNOMONIC (test ATM biallelic); "
            "Standard RT causing severe radiation necrosis = A-T/ATM biallelic pattern; "
            "Elevated AFP (alpha-fetoprotein) in childhood ataxia = A-T diagnostic clue"
        ),
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MutS-Homolog-2-105kDa-MutSalpha-MutSbeta-"
            "Lynch-Syndrome-Muir-Torre-Sebaceous-Neoplasms-Face-PATHOGNOMONIC-EPCAM-Silencing-OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 2p21 MSH2 encodes MutS homolog 2 (MSH2 / MutS protein homolog 2): "
            "STRUCTURE: "
            "  Mismatch binding domain (MBD, aa 1-148): DNA mismatch recognition; "
            "  Lever domain (LD, aa 148-240): conformational change; "
            "  Connector domain (CD, aa 240-390): MSH2-MSH6/MSH3 interface; "
            "  Helix-turn-helix (HTH) domain (aa 390-505): DNA clamping; "
            "  ATPase domain (aa 505-934): MutSα/β sliding clamp release via ATP; "
            "  Forms MutSα (MSH2-MSH6): repairs 1-2bp mismatches + small IDLs; "
            "  Forms MutSβ (MSH2-MSH3): repairs larger IDLs; "
            "MISMATCH REPAIR (MMR): "
            "  MSH2-MSH6 (MutSα) recognises single-base mismatches and IDLs → recruits MLH1-PMS2 (MutLα) → excision → re-synthesis; "
            "  Without MSH2: microsatellite instability (MSI) accumulates → hypermutation → Lynch tumours; "
            "LYNCH SYNDROME TYPE 2 (HNPCC): "
            "  CRC lifetime risk: 40-60%; endometrial 40-60%; urothelial 15-25%; "
            "  MSH2 confers highest Lynch urothelial risk (vs other MMR genes); "
            "  Annual colonoscopy from age 25 (or 5yr before earliest family CRC); "
            "MUIR-TORRE SYNDROME — MSH2 SUBTYPE: "
            "  Muir-Torre = Lynch + sebaceous neoplasms of skin (sebaceous adenoma/carcinoma + keratoacanthoma); "
            "  SEBACEOUS NEOPLASMS OF FACE/SCALP/EYELID = PATHOGNOMONIC Muir-Torre: "
            "    Multiple sebaceous adenomas: PATHOGNOMONIC Muir-Torre (sebaceous adenomas are rare outside MTS); "
            "    Sebaceous carcinoma: rare, aggressive — eyelid, scalp, face; "
            "    Keratoacanthoma of head/neck: also elevated; "
            "  MSH2 accounts for ~75% of Muir-Torre — MSH6 accounts for ~25%; "
            "  Sebaceous neoplasm biopsy: MMR IHC mandatory — MSH2 loss = Lynch/Muir-Torre; "
            "  Clinical rule: ANY sebaceous adenoma → test MMR germline; "
            "EPCAM 3-PRIME DELETION: "
            "  Large deletions in 3-prime end of EPCAM (upstream of MSH2) → epigenetic silencing of MSH2; "
            "  MSH2 protein absent on IHC; MSH2 coding sequence normal on Sanger/NGS; "
            "  EPCAM deletion missed by standard MSH2 sequencing → MLPA required; "
            "HEAD & NECK MANIFESTATIONS: "
            "  Muir-Torre sebaceous tumours predominantly on face, scalp, eyelids, neck; "
            "  Oral mucosal SCC: mildly elevated in Lynch/MSH2; "
            "  Annual ENT + skin surveillance"
        ),
        "inheritance": (
            "AD LOF 2p21 — MSH2. Lynch Syndrome type 2 / Muir-Torre. "
            "Penetrance: CRC 40-60%, endometrial 40-60%, urothelial 15-25%. "
            "De novo: <5% of Lynch probands. "
            "Prevalence: Lynch syndrome all types ~1 per 440 population; MSH2 accounts for ~25-40% of Lynch. "
            "EPCAM 3-prime deletion: ~20% of apparent MSH2-mutation Lynch — MLPA essential."
        ),
        "surveillance_key": (
            "Annual colonoscopy from age 25; annual endometrial sampling from age 35 (females); "
            "annual urine cytology + cystoscopy from age 25 (urothelial 15-25% risk); "
            "annual full-body skin exam (Muir-Torre sebaceous neoplasms face/scalp PATHOGNOMONIC); "
            "sebaceous adenoma → MMR IHC mandatory; EPCAM deletion check (MLPA); "
            "aspirin 600mg daily CAPP2 50% CRC risk reduction (consider)"
        ),
        "pathognomonic": (
            "Sebaceous adenomas of face/scalp + CRC family history = Muir-Torre/MSH2 PATHOGNOMONIC; "
            "Multiple keratoacanthomas of face/neck + Lynch-spectrum cancer = Muir-Torre; "
            "MSH2 absent on IHC with normal MSH2 coding sequence = EPCAM 3-prime deletion"
        ),
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-Breast-Cancer-Gene-2-384kDa-Homologous-Recombination-Scaffold-"
            "HBOC-HNC-2-3x-Oropharyngeal-PARP-Inhibitor-Olaparib-FA-D1-Biallelic-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 13q12.3 BRCA2 encodes breast cancer gene 2 protein (BRCA2): "
            "STRUCTURE: "
            "  N-terminal transactivation domain (aa 1-39): EMSY binding; "
            "  OB-folds 1-3 (aa 2402-2667): ssDNA binding; "
            "  Tower domain (aa 2402-2667): ssDNA binding, inserted in OB2; "
            "  BRC repeats 1-8 (aa 1002-2085): RAD51 monomer binding (each BRC: FXXAS motif); "
            "  C-terminal domain (CTD, aa 2670-3190): DSS1 + ssDNA; RAD51 filament stabilisation; "
            "  PALB2-binding domain (N-terminus): tethers BRCA2 to chromatin via PALB2-BRCA1; "
            "HOMOLOGOUS RECOMBINATION (HR): "
            "  BRCA2 is the primary RAD51 mediator — loads RAD51 onto resected ssDNA; "
            "  Without BRCA2: HR fails → DSBs repaired by error-prone NHEJ → genomic instability; "
            "  BRCA2-null cells: PARP-inhibitor synthetic lethality (BRCAness); "
            "HBOC CANCER SPECTRUM: "
            "  Breast (female): 45-70% lifetime risk (vs 12% general); "
            "  Ovarian: 10-20% lifetime (vs <2% general); "
            "  Prostate: 5-8x elevated (high-grade Gleason ≥7); "
            "  Pancreatic: 3-5x; "
            "  Male breast: 8% lifetime; "
            "  Melanoma: 2-3x; "
            "HEAD & NECK CANCER — BRCA2: "
            "  Oropharyngeal SCC: 2-3x elevated in BRCA2 germline — HPV-independent pathway; "
            "  Oral SCC: modest elevation (1.5-2x); "
            "  Laryngeal cancer: some association; "
            "  BRCA2 somatic alterations in HNC: ~5% (vs 25-40% in ovarian cancer); "
            "  Clinical implication: BRCA2 germline carrier with HNC may respond to PARP inhibitors; "
            "PARP INHIBITORS — OLAPARIB FDA-APPROVED: "
            "  Mechanism: PARP trapping → stalled replication forks → DSBs → BRCA2-null cells cannot repair; "
            "  Olaparib FDA 2014/2018/2022: ovarian, breast, pancreatic, prostate BRCA2-associated; "
            "  HNC BRCA2: clinical trial data emerging (NCI-MATCH); "
            "  Niraparib, rucaparib, talazoparib: alternative PARP inhibitors; "
            "FANCONI ANEMIA TYPE D1 (FA-D1) — BIALLELIC BRCA2: "
            "  Biallelic BRCA2 pathogenic variants → severe FA-D1 phenotype; "
            "  Extremely severe: VACTERL-H + early-onset AML/medulloblastoma/Wilms tumour; "
            "  Median diagnosis: early childhood; "
            "  Same HNC risk as FA-A in biallelic context; "
            "SURVEILLANCE: "
            "  Annual breast MRI + mammogram from age 30; "
            "  Annual ovarian CA-125 + transvaginal USS from age 30; "
            "  Consider risk-reducing bilateral salpingo-oophorectomy (BSO) age 40-45; "
            "  Annual ENT exam (HNC 2-3x risk); "
            "  Annual PSA (prostate) in males from age 40"
        ),
        "inheritance": (
            "AD LOF 13q12.3 — BRCA2. HBOC. "
            "Penetrance: breast (female) 45-70%, ovarian 10-20%, pancreatic 3-5x, prostate 5-8x, HNC 2-3x. "
            "De novo: <1% of BRCA2 probands. "
            "Prevalence: general population ~1 per 400-800; Ashkenazi Jewish 1 per 40 (c.6174delT founder). "
            "Founder mutations: "
            "  Ashkenazi Jewish: c.6174delT (accounts for ~40% of BRCA2 in AJ); "
            "  Icelandic: c.771_775del5 (995del5); "
            "  UK: c.3036_3039del4; "
            "  French-Canadian: c.3398_3400del3."
        ),
        "surveillance_key": (
            "Annual breast MRI + mammogram from age 30; annual pelvic USS + CA-125 from age 30; "
            "BSO age 40-45 after childbearing (85% ovarian risk reduction); "
            "annual PSA males from age 40; annual ENT (HNC 2-3x); "
            "PARP inhibitors (olaparib) for BRCA2-associated HNC (emerging trial data); "
            "platinum-based chemo: BRCA2-mutated HNC responsive to cisplatin (HR-deficient)"
        ),
        "pathognomonic": (
            "Oropharyngeal SCC + family history breast/ovarian cancer = BRCA2 germline workup; "
            "HPV-negative oropharyngeal SCC at younger age without tobacco = test hereditary panel; "
            "Biallelic BRCA2: severe FA-D1 with VACTERL-H + early-onset AML/medulloblastoma"
        ),
    },
    {
        "gene": "RECQL4",
        "protein": (
            "RECQL4 -- 8q24.12 Autosomal-Recessive-LOF -- 1208aa -- "
            "RecQ-Like-Helicase-4-133kDa-"
            "Rothmund-Thomson-Syndrome-Poikiloderma-Congenitale-PATHOGNOMONIC-SCC-Face-Osteosarcoma-OMIM-603780"
        ),
        "locus": "8q24.12",
        "protein_size": (
            "1208 aa / 8q24.12 RECQL4 encodes RecQ-like helicase 4 (RECQL4): "
            "STRUCTURE: "
            "  N-terminal domain (aa 1-491): ssDNA binding; nuclear localisation; mitochondrial targeting; "
            "  RecQ helicase domain (aa 492-890): ATP-dependent 3-prime to 5-prime helicase activity; "
            "    RecQ C-terminal (RQC) domain: zinc-binding; Topo IIIα interaction; "
            "    Helicase and RNaseD C-terminal (HRDC) domain: DNA binding; "
            "  C-terminal domain (aa 891-1208): mitochondrial DNA replication; "
            "DNA REPAIR FUNCTIONS: "
            "  Replication initiation: essential for CMG (CDC45-MCM-GINS) complex at replication origins; "
            "  DSB repair (NHEJ and HR); replication stress response; "
            "  Mitochondrial DNA maintenance (C-terminal); "
            "ROTHMUND-THOMSON SYNDROME (RTS): "
            "  Biallelic RECQL4 LOF → Rothmund-Thomson syndrome (RTS type 2); "
            "  POIKILODERMA CONGENITALE = PATHOGNOMONIC: "
            "    Erythematous skin rash (face, hands, feet) in first 3-6 months; "
            "    Evolves to poikiloderma: reticulated hyperpigmentation + hypopigmentation + telangiectasias + skin atrophy; "
            "    Distribution: face (spares central area initially), forearms, hands, lower legs; "
            "    Face: PATHOGNOMONIC — characteristic facial poikiloderma in infant; "
            "  Sparse hair, eyebrows, eyelashes; "
            "  Short stature; "
            "  Radial ray defects: absent/hypoplastic thumb (like FA — overlap); "
            "  Cataracts (juvenile, bilateral); "
            "  Dental anomalies; "
            "CANCER RISK IN RECQL4/RTS: "
            "  Osteosarcoma: 30-40x general population risk; "
            "    Median age 11yr; proximal tibia, distal femur — same distribution as sporadic; "
            "    RTS patients: osteosarcoma in ~30% lifetime; "
            "  Head & neck SCC: "
            "    SCC of sun-exposed sites (face, lips, ears): UV-triggered; "
            "    Poikilodermatous skin → field cancerisation → multiple SCCs; "
            "    Onset: teens/young adulthood in heavily sun-exposed individuals; "
            "  AML / MDS: elevated risk in RECQL4/RTS; "
            "PHOTOSENSITIVITY AND UV AVOIDANCE: "
            "  UV-induced DNA damage → compromised RECQL4-dependent repair → mutagenesis; "
            "  Strict UV avoidance recommended (same principles as XP); "
            "  Annual dermatology from childhood; "
            "  Avoid UV radiation; SPF 50+ sunscreen; UPF 50+ clothing; "
            "SURVEILLANCE: "
            "  Annual whole-body MRI (osteosarcoma, AML); "
            "  Annual dermatology + oral mucosal exam (SCC); "
            "  Annual ophthalmology (cataracts); "
            "  Annual CBC (AML/MDS); "
            "  RTS osteosarcoma: standard chemotherapy (MAP: methotrexate-adriamycin-cisplatin); "
            "  HNC in RTS: surgical excision; chemotherapy; RT with caution (moderate radiosensitivity)"
        ),
        "inheritance": (
            "AR biallelic LOF 8q24.12 — RECQL4. Rothmund-Thomson syndrome. "
            "Biallelic required for RTS. Monoallelic carriers: likely no phenotype (uncertain). "
            "Prevalence: ~300 cases reported worldwide; rare (1 per 1,000,000+ estimated). "
            "No well-established founder mutations; compound heterozygosity common. "
            "BALLER-GEROLD SYNDROME (BGS): RECQL4 biallelic → craniosynostosis + radial ray defects + mild poikiloderma; "
            "RAPADILINO SYNDROME: RECQL4 biallelic → radial ray defects + patella hypoplasia + limb anomalies + short stature; "
            "RTS/BGS/RAPADILINO: allelic disorders with different phenotypic emphasis."
        ),
        "surveillance_key": (
            "Annual WBMRI (osteosarcoma childhood 30-40x; AML); annual CBC (AML/MDS); "
            "annual dermatology (SCC face/head UV-exposed); annual ophthalmology (cataracts); "
            "strict UV avoidance (UPF 50+ clothing; SPF 50+ sunscreen; UV-filtering glasses); "
            "annual ENT + oral mucosal exam; "
            "osteosarcoma: MAP chemotherapy; HNC: surgery + chemo (RT with caution)"
        ),
        "pathognomonic": (
            "Poikiloderma congenitale (facial reticulated pigmentation) in infant = PATHOGNOMONIC RECQL4/RTS; "
            "Osteosarcoma in child + poikilodermatous skin = RTS; "
            "Multiple SCCs face/head + poikiloderma in young adult without heavy sun history = RECQL4"
        ),
    },
]


def _gene_stats(seed: int, gene_cfg: dict) -> dict:
    rng = random.Random(seed)
    gene = gene_cfg["gene"]

    base = {
        "FANCA":  dict(age_mu=26, age_sd=8,  hnc_pct=(55, 72),   aplasia_pct=(65, 82),  aml_pct=(22, 38),    bmt_pct=(45, 65)),
        "XPC":    dict(age_mu=8,  age_sd=4,  hnc_pct=(70, 88),   scc_face_pct=(82, 96), uv_sensitivity_pct=(95, 100), photoph_pct=(72, 88)),
        "TP53":   dict(age_mu=34, age_sd=14, hnc_pct=(12, 22),   sarcoma_pct=(28, 42),  radiation_avoid_pct=(92, 99), wbmri_pct=(85, 98)),
        "CDKN2A": dict(age_mu=44, age_sd=12, hnc_pct=(35, 52),   melanoma_pct=(45, 62), pancreatic_pct=(22, 38), atypical_naevi_pct=(72, 92)),
        "ATM":    dict(age_mu=18, age_sd=10, hnc_pct=(28, 42),   ataxia_pct=(88, 98),   teleang_pct=(82, 96), radiation_sensitive_pct=(90, 100)),
        "MSH2":   dict(age_mu=46, age_sd=12, hnc_pct=(18, 32),   sebaceous_pct=(55, 78), crc_pct=(42, 60),    muir_torre_pct=(38, 58)),
        "BRCA2":  dict(age_mu=52, age_sd=12, hnc_pct=(22, 38),   breast_pct=(45, 68),   ovarian_pct=(12, 22), parp_eligible_pct=(60, 82)),
        "RECQL4": dict(age_mu=22, age_sd=8,  hnc_pct=(25, 42),   osteo_pct=(28, 42),    poikiloderm_pct=(88, 98), cataracts_pct=(42, 62)),
    }
    b = base.get(gene, dict(age_mu=30, age_sd=10))

    def pct(lo, hi): return round(rng.uniform(lo, hi), 1)
    def age(): return max(0.5, round(rng.gauss(b["age_mu"], b["age_sd"]), 1))

    ages = [age() for _ in range(40)]
    mean_age = round(sum(ages) / len(ages), 1)

    stats = dict(gene=gene, n=40, mean_age_diagnosis=mean_age)
    for key, val in b.items():
        if key.endswith("_pct") and isinstance(val, tuple):
            stats[key] = pct(val[0], val[1])

    stats["locus"] = gene_cfg["locus"]
    stats["inheritance"] = gene_cfg["inheritance"]
    stats["surveillance_key"] = gene_cfg["surveillance_key"]
    stats["pathognomonic"] = gene_cfg["pathognomonic"]
    return stats


def generate_overview() -> dict:
    return {
        "atlas":          "Hereditary-Head-Neck-Cancer-Predisposition-Atlas",
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_genes":    len(ATLAS_GENES),
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "inheritance_modes": {
            "FANCA": (
                "AR biallelic LOF 16q24.3 (FANCA 1455aa 163kDa FA core complex scaffold; "
                "Fanconi Anemia type A most common 60-70%; "
                "Oral/oropharyngeal SCC 700x PATHOGNOMONIC age 26yr; aplastic anaemia; AML; "
                "AVOID RADIATION ABSOLUTELY; BMT fludarabine conditioning; EPCAM 3-prime deletion check)"
            ),
            "XPC": (
                "AR biallelic LOF 3p25.1 (XPC 940aa 106kDa GG-NER damage sensor; "
                "Xeroderma Pigmentosum C most common XP globally; "
                "UV-SCC face/scalp/lip 10,000x PATHOGNOMONIC onset childhood; NO neurological; "
                "strict UV avoidance MANDATORY; dermatology 3-monthly; vismodegib BCC burden)"
            ),
            "TP53": (
                "AD LOF 17p13.1 (p53 393aa 43kDa guardian genome; Li-Fraumeni Syndrome; "
                "HNC (laryngeal/oral) in LFS; AVOID RADIATION ABSOLUTELY; "
                "WBMRI Toronto annually; HNC treatment surgery+chemo only)"
            ),
            "CDKN2A": (
                "AD LOF 9p21.3 (p16/INK4A 156aa CDK4/6 inhibitor + p14/ARF MDM2 inhibitor; "
                "FAMM/FAMMM syndrome; oral/oropharyngeal SCC 10-30x; "
                "pancreatic cancer 20-30x PATHOGNOMONIC in FAMM; melanoma 25-36x; "
                "CDK4/6 inhibitors palbociclib emerging for CDKN2A-deficient HNC)"
            ),
            "ATM": (
                "AR biallelic A-T / AD monoallelic 11q22.3 (ATM 3056aa 350kDa PI3K-like kinase; "
                "cerebellar ataxia + conjunctival telangiectasias PATHOGNOMONIC biallelic; "
                "radiosensitivity ABSOLUTE biallelic; HNC 3-5x monoallelic; "
                "elevated AFP in A-T; ceralasertib ATRi trials)"
            ),
            "MSH2": (
                "AD LOF 2p21 (MSH2 934aa 105kDa MutSα/β MMR; Lynch Syndrome type 2; "
                "Muir-Torre: sebaceous adenomas/carcinomas face/scalp = PATHOGNOMONIC; "
                "keratoacanthomas HNC; EPCAM 3-prime deletion → MSH2 silencing; "
                "aspirin CAPP2 50% CRC risk reduction)"
            ),
            "BRCA2": (
                "AD LOF 13q12.3 (BRCA2 3418aa 384kDa RAD51 mediator HR scaffold; HBOC; "
                "HNC oropharyngeal 2-3x; PARP inhibitor olaparib FDA-approved; "
                "platinum-based chemo: BRCA2 HNC responds; "
                "biallelic FA-D1: VACTERL-H + severe AML/medulloblastoma)"
            ),
            "RECQL4": (
                "AR biallelic LOF 8q24.12 (RECQL4 1208aa 133kDa RecQ helicase replication initiation; "
                "Rothmund-Thomson Syndrome; poikiloderma congenitale PATHOGNOMONIC facial infant; "
                "SCC face/head UV-exposed; osteosarcoma 30-40x childhood; AML; "
                "strict UV avoidance; allelic: RTS/BGS/RAPADILINO)"
            ),
        },
        "key_clinical_rules": [
            "FANCA/Fanconi Anemia: oral/oropharyngeal SCC risk 700x general population — any SCC in young adult without heavy smoking/HPV → FA workup (chromosomal breakage assay DEB/MMC)",
            "FANCA-HNC: AVOID RADIATION ABSOLUTELY — standard RT → severe necrosis + secondary tumours; use surgery + reduced-dose mitomycin C + 5-FU (25-50% dose reduction)",
            "XPC/Xeroderma Pigmentosum C: SCC/BCC face/scalp onset childhood (age 8yr median) — strict UV avoidance MANDATORY (UPF50+ clothing, UV-blocking glasses, window film); dermatology every 3 months",
            "XPC: no neurological involvement unlike XP-A/D — key DDx; 10,000x UV-induced HNC risk for sun-exposed sites",
            "TP53/LFS: AVOID ALL IONISING RADIATION ABSOLUTELY — RT-induced sarcoma in radiation field; HNC treatment = surgery + chemotherapy ONLY, no RT ever",
            "CDKN2A/FAMM: oral/oropharyngeal SCC 10-30x + pancreatic cancer 20-30x — pancreatic MRI/EUS annually from age 40 (CAPS consortium); melanoma 25-36x; smoking cessation mandatory",
            "CDKN2A: multiple atypical naevi (>50 dysplastic naevi) + melanoma + family history = FAMM → test CDKN2A germline; PDAC + melanoma in same family = CDKN2A germline",
            "ATM biallelic (A-T): cerebellar ataxia + conjunctival telangiectasias = PATHOGNOMONIC — AVOID ALL RT ABSOLUTELY; standard RT dose → lethal radiation injury; elevated AFP diagnostic clue",
            "ATM monoallelic: HNC 3-5x; breast cancer 4x; moderate radiosensitivity — inform RT oncology; annual ENT exam",
            "MSH2/Muir-Torre: sebaceous adenomas of face/scalp/eyelid = PATHOGNOMONIC Muir-Torre → test MMR IHC on biopsy; MSH2 absent on IHC + normal coding = EPCAM 3-prime deletion (test MLPA)",
            "MSH2/Lynch: highest urothelial risk of all MMR genes (15-25%); annual urine cytology + cystoscopy from age 25; EPCAM deletion silences MSH2 — MLPA required",
            "BRCA2/HBOC: oropharyngeal SCC 2-3x; platinum-based chemo responsive (HR-deficient); PARP inhibitors (olaparib) for BRCA2-mutated HNC (NCI-MATCH trial data emerging)",
            "RECQL4/Rothmund-Thomson: poikiloderma congenitale (facial reticulated pigmentation) in infant = PATHOGNOMONIC RTS; SCC of face/head in young adult + poikiloderma = RECQL4; osteosarcoma 30-40x in childhood",
            "RECQL4: allelic disorders — RTS (poikiloderma + osteosarcoma), Baller-Gerold (craniosynostosis + radial ray), RAPADILINO (radial ray + patella + short stature) — all need oncological surveillance",
            "ALL HEREDITARY HNC PANEL: FANCA + XPC + TP53 + CDKN2A + ATM + MSH2 + BRCA2 + RECQL4 — indicated for: HNC age <40yr, multiple HNC primaries, HNC + family cancer history, radiosensitivity on RT, HNC + skin findings (XP/Muir-Torre/RTS pattern)",
        ],
        "gene_panel_note": (
            "Hereditary Head & Neck Cancer Germline Panel (clinical 2024): "
            "FANCONI ANEMIA PATHWAY (FANCA — oral/oropharyngeal SCC 700x PATHOGNOMONIC, AVOID RADIATION ABSOLUTELY): "
            "  FANCA: FA-A most common; oral/oropharyngeal SCC 700x age 26yr; BMT for aplastic anaemia; "
            "    chromosomal breakage assay DEB/MMC DIAGNOSTIC; reduced-dose chemo; no RT; "
            "XP NUCLEOTIDE EXCISION REPAIR (XPC — UV-SCC face 10,000x PATHOGNOMONIC): "
            "  XPC: most common XP; no neurological (DDx XP-A/D); facial/scalp/lip SCC age 8yr; "
            "    strict UV avoidance mandatory; dermatology 3-monthly; no systemic drugs (surgery); "
            "LI-FRAUMENI / TP53 (AVOID RADIATION ABSOLUTELY): "
            "  TP53: HNC in LFS laryngeal/oral; RT-induced HNC risk; WBMRI Toronto; surgery+chemo only; "
            "CDKN2A / FAMM (oral SCC 10-30x + pancreatic 20x PATHOGNOMONIC): "
            "  CDKN2A: p16/INK4A + p14/ARF; FAMM; atypical naevi PATHOGNOMONIC marker; "
            "    PDAC + melanoma family = CDKN2A; CDK4/6 inhibitors emerging; pancreatic MRI/EUS age 40; "
            "ATM / ATAXIA TELANGIECTASIA (radiosensitivity PATHOGNOMONIC biallelic, HNC 3-5x monoallelic): "
            "  ATM: ataxia + telangiectasias PATHOGNOMONIC; elevated AFP; AVOID RT biallelic; "
            "    monoallelic: breast 4x, HNC 3-5x; annual ENT; inform RT team; "
            "MSH2 / LYNCH / MUIR-TORRE (sebaceous neoplasms face PATHOGNOMONIC): "
            "  MSH2: Lynch highest urothelial risk; Muir-Torre sebaceous face/scalp PATHOGNOMONIC; "
            "    EPCAM 3-prime deletion → MSH2 silencing; MLPA essential; aspirin CAPP2; "
            "BRCA2 / HBOC (HNC 2-3x + PARP inhibitor): "
            "  BRCA2: oropharyngeal SCC 2-3x; platinum-responsive; olaparib emerging HNC trials; "
            "    biallelic FA-D1 severe; annual breast MRI age 30; BSO age 40-45; "
            "RECQL4 / ROTHMUND-THOMSON (poikiloderma congenitale PATHOGNOMONIC): "
            "  RECQL4: poikiloderma face PATHOGNOMONIC infant; SCC face/head; osteosarcoma 30-40x; "
            "    allelic RTS/BGS/RAPADILINO; annual WBMRI; strict UV avoidance; annual dermatology; "
            "UNIVERSAL TESTING CRITERIA FOR HEREDITARY HNC: "
            "  HNC age <40yr: full germline panel; "
            "  Multiple HNC primaries (same or different sites): FANCA + TP53 + CDKN2A; "
            "  SCC face/scalp childhood: XPC (XP); "
            "  Poikiloderma + SCC/osteosarcoma: RECQL4; "
            "  Sebaceous adenoma any site: MMR IHC → MSH2/MSH6; "
            "  Ataxia + telangiectasias: ATM biallelic; "
            "  Severe radiosensitivity on RT: FANCA + ATM + XPC germline"
        ),
    }


def generate_breakdown() -> dict:
    genes_data = []
    for i, gene_cfg in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        stats = _gene_stats(seed, gene_cfg)
        stats["protein_summary"] = gene_cfg["protein"]
        stats["locus"] = gene_cfg["locus"]
        stats["inheritance"] = gene_cfg["inheritance"]
        stats["surveillance_key"] = gene_cfg["surveillance_key"]
        stats["pathognomonic"] = gene_cfg["pathognomonic"]
        genes_data.append(stats)
    return {
        "atlas":          "Hereditary-Head-Neck-Cancer-Predisposition-Atlas",
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "n_genes":        len(genes_data),
        "total_patients": 320,
        "genes":          genes_data,
    }


def generate_definitions() -> dict:
    definitions = [
        {
            "term": "FANCA-Fanconi-Anemia-HNC-700x-PATHOGNOMONIC-AVOID-RADIATION-BMT",
            "definition": (
                "FANCONI ANEMIA (FA-A) — ORAL/OROPHARYNGEAL SCC 700x RISK: "
                "FA is the most extreme single-gene predisposition to head and neck squamous cell carcinoma. "
                "PATHOGNOMONIC HNC PATTERN: "
                "  SCC of oral cavity (floor of mouth, tongue, gingiva, buccal mucosa), oropharynx, hypopharynx, larynx; "
                "  Risk: ~700x general population — lifetime HNC cumulative risk >50% in FA; "
                "  Age at HNC onset: median 26yr (vs 62yr sporadic HNC); "
                "  HPV-negative predominates (FA HNC is HPV-independent); "
                "  Often multifocal — field cancerisation of FA-compromised mucosa; "
                "CHROMOSOMAL BREAKAGE ASSAY — GOLD STANDARD DIAGNOSIS: "
                "  Diepoxybutane (DEB) or mitomycin C (MMC) challenge → peripheral blood lymphocytes; "
                "  FA cells: abnormal chromosomal breakage, radial figures, complex exchanges; "
                "  Sensitivity/specificity >99% for FA diagnosis; "
                "  Molecular (FANCA sequencing): confirms complementation group; "
                "AVOID RADIATION ABSOLUTELY IN FA: "
                "  FA cells cannot repair DNA crosslinks + ionising radiation damage; "
                "  Standard RT (60-70 Gy) → severe mucositis, tissue necrosis, non-healing wounds; "
                "  RT-associated mortality documented in FA-HNC cohorts; "
                "  HNC chemotherapy in FA: modified mitomycin C + 5-FU (25-50% dose reduction); "
                "  Surgery: primary treatment for resectable FA-HNC; "
                "POST-BMT HNC RISK: "
                "  HSCT corrects haematological failure but does NOT reduce HNC risk; "
                "  Chronic graft-versus-host disease → mucosal inflammation → accelerated carcinogenesis; "
                "  Post-BMT FA patients: even higher HNC incidence (mucosal GVHD acts as co-carcinogen); "
                "  Annual ENT surveillance MANDATORY from age 18 or post-BMT (whichever earlier)."
            ),
        },
        {
            "term": "XPC-Xeroderma-Pigmentosum-C-UV-SCC-Face-10000x-PATHOGNOMONIC-UV-Avoidance",
            "definition": (
                "XERODERMA PIGMENTOSUM C (XP-C) — UV-INDUCED SCC FACE/HEAD 10,000x: "
                "XP-C is the most common XP complementation group globally — no neurological involvement. "
                "PATHOGNOMONIC UV PATTERN: "
                "  SCC, BCC, melanoma of face, scalp, lip, eyelids, ears: onset median age 8yr; "
                "  Risk: ~10,000x general population for UV-exposed head/neck sites; "
                "  Freckling of face within first 2yr = earliest sign; "
                "  Progressive actinic keratoses → SCC/BCC without intervention; "
                "  Corneal opacification, pterygium, corneal SCC; "
                "XP-C vs XP-A/D DISTINCTION: "
                "  XP-C: no cerebellar ataxia, no peripheral neuropathy, no sensorineural hearing loss; "
                "  XP-A: severe neurological (cerebellar ataxia + deafness); "
                "  XP-D (ERCC2): variable — neurological in some; early-onset; "
                "  This distinction is critical for prognosis and surveillance; "
                "STRICT UV AVOIDANCE — MANDATORY FOR SURVIVAL: "
                "  UPF 50+ full-body clothing outdoors; UV-blocking wraparound glasses; "
                "  UV-filtering window film (home/car/workplace); "
                "  Avoid outdoor exposure 10am-4pm; "
                "  Indoor fluorescent lighting: UV-filtered bulbs; "
                "  With strict avoidance: cancer-free survival extended to 4th-5th decade; "
                "  Without protection: median cancer-free survival <20yr; "
                "SURVEILLANCE AND TREATMENT: "
                "  Dermatology every 3 months; ENT annually; ophthalmology annually; "
                "  Topical imiquimod for actinic keratoses; "
                "  Oral retinoids (isotretinoin) for field suppression (limited evidence); "
                "  Nicotinamide 500mg twice daily: NAD+ cofactor for NER — reduces actinic keratoses; "
                "  Vismodegib/sonidegib for BCC burden (Hh pathway in BCC)."
            ),
        },
        {
            "term": "CDKN2A-FAMM-Oral-SCC-10-30x-Pancreatic-20x-PATHOGNOMONIC-Melanoma-CDK4-6i",
            "definition": (
                "CDKN2A/FAMM — ORAL/OROPHARYNGEAL SCC + PANCREATIC CANCER PATHOGNOMONIC COMBINATION: "
                "CDKN2A encodes BOTH p16/INK4A (CDK4/6 inhibitor) AND p14/ARF (MDM2 inhibitor). "
                "HEAD & NECK CANCER: "
                "  Oral SCC (floor of mouth, tongue): 10-30x elevated; "
                "  Oropharyngeal SCC: elevated especially HPV-negative; "
                "  Laryngeal: modest elevation; "
                "  Earlier onset vs sporadic: median 44yr vs 60yr; "
                "  Smoking massively amplifies CDKN2A HNC risk (multiplicative interaction); "
                "PANCREATIC CANCER — PATHOGNOMONIC IN FAMM: "
                "  PDAC 20-30x (lifetime cumulative 17-39% in carriers with family history); "
                "  PDAC + melanoma in same individual or family = CDKN2A germline until excluded; "
                "  Annual pancreatic MRI + EUS from age 40 (CAPS consortium recommendation); "
                "MELANOMA: "
                "  25-36x lifetime risk; multiple primary melanomas common; "
                "  Multiple atypical (dysplastic) naevi = PATHOGNOMONIC FAMM phenotypic marker; "
                "  Annual full-body skin surveillance from age 18; "
                "CDK4/6 INHIBITORS: "
                "  p16-deficient tumours (CDKN2A null) → dependent on CDK4/6 activity → synthetic vulnerability; "
                "  Palbociclib + cetuximab (anti-EGFR): trials in p16-negative HNC; "
                "  Abemaciclib: more potent CDK4 vs CDK6; trials for CDKN2A-null solid tumours; "
                "FOUNDER MUTATIONS: "
                "  Netherlands p.Ala148Thr: 70% Dutch FAMM; obligate surveillance; "
                "  CAPS guidelines: high-risk pancreatic surveillance if PDAC family history + CDKN2A."
            ),
        },
        {
            "term": "ATM-Ataxia-Telangiectasia-Radiosensitivity-PATHOGNOMONIC-HNC-Monoallelic",
            "definition": (
                "ATAXIA TELANGIECTASIA (A-T) — RADIOSENSITIVITY PATHOGNOMONIC + HNC 3-5x MONOALLELIC: "
                "BIALLELIC A-T: "
                "  CEREBELLAR ATAXIA onset 1-4yr: progressive gait unsteadiness; loss of independent walking ~teens; "
                "  CONJUNCTIVAL TELANGIECTASIAS: appear age 3-6yr; bulbar conjunctiva; then skin; "
                "  Cerebellar ataxia + telangiectasias = A-T PATHOGNOMONIC combination; "
                "  Elevated alpha-fetoprotein (AFP) in >95% of A-T — diagnostic clue; "
                "RADIOSENSITIVITY — ABSOLUTE IN BIALLELIC: "
                "  ATM-null cells: DSBs unrepaired → massive cell death → tissue necrosis; "
                "  Standard RT (60 Gy HNSCC protocol): lethal in A-T — documented fatalities; "
                "  Even 2 Gy fractions: severe acute reactions; "
                "  RULE: CONFIRM ATM STATUS BEFORE ANY RT IN CHILDHOOD/YOUNG ADULT HNC; "
                "  Alternative treatment: surgery; low-dose platinum (modified); "
                "  Diagnostic CT with contrast: use sparingly — cumulative low-dose RT concern; prefer MRI; "
                "MONOALLELIC ATM — CANCER RISK: "
                "  ~1% population monoallelic ATM carriers; "
                "  HNC 3-5x (oral SCC, laryngeal, oropharyngeal); "
                "  Breast cancer 4x (lifetime 33%); pancreatic 5x; gastric 4x; "
                "  Monoallelic: moderate radiosensitivity — inform RT team (dose modification may be needed); "
                "CANCER PREDISPOSITION IN A-T: "
                "  Lymphoma/leukaemia 70-100x; HNC in A-T survivors of haematological malignancy; "
                "  Post-chemo RT in A-T: avoid; "
                "CERALASERTIB (AZD6738) — ATR INHIBITOR: "
                "  ATR is activated by replication stress (downstream of ATM); "
                "  ATM-null tumours → ATR-dependent for survival → synthetic lethality with ATRi; "
                "  Ceralasertib trials: ATM-deficient HNC + gastric cancer (Calvert 2022 Lancet Oncol)."
            ),
        },
        {
            "term": "MSH2-Lynch-Muir-Torre-Sebaceous-Face-PATHOGNOMONIC-EPCAM",
            "definition": (
                "MSH2/LYNCH SYNDROME / MUIR-TORRE — SEBACEOUS NEOPLASMS FACE = PATHOGNOMONIC: "
                "MUIR-TORRE SYNDROME: "
                "  Definition: Lynch syndrome + sebaceous neoplasms of skin ± keratoacanthomas; "
                "  SEBACEOUS ADENOMA of face/scalp/eyelid = PATHOGNOMONIC Muir-Torre: "
                "    Sebaceous adenomas are rare in general population (uncommon on face); "
                "    Presence of any sebaceous adenoma → MMR IHC mandatory on biopsy; "
                "    Sebaceous adenoma IHC: MSH2 absent (most common) or MSH6 absent; "
                "    Clinical rule: sebaceous adenoma → Lynch germline testing; "
                "  Sebaceous carcinoma: aggressive — eyelid, face, scalp; "
                "  Keratoacanthomas: rapidly growing nodule with central crater (face/neck); "
                "  MSH2 accounts for ~75% of Muir-Torre; "
                "EPCAM 3-PRIME DELETION — KEY PITFALL: "
                "  Large 3-prime deletions of EPCAM (adjacent to MSH2 at 2p21) → epigenetic MSH2 silencing; "
                "  MSH2 IHC: absent in tumour; "
                "  MSH2 coding sequence on Sanger/panel NGS: NORMAL (coding exons intact); "
                "  Standard sequencing MISSES this mechanism — MLPA for EPCAM 3-prime deletion required; "
                "  ~20% of apparent MSH2-negative Lynch cases have EPCAM deletion rather than MSH2 coding mutation; "
                "UROTHELIAL RISK — HIGHEST AMONG MSH2/LYNCH: "
                "  Upper urothelial tract (renal pelvis, ureter): 15-25% lifetime risk in MSH2; "
                "  Bladder: 3-7% lifetime; "
                "  Annual urine cytology + cystoscopy from age 25 (MSH2 Lynch); "
                "ASPIRIN CAPP2 — 50% CRC RISK REDUCTION: "
                "  CAPP2 trial: aspirin 600mg daily → 50-60% reduction in Lynch CRC; "
                "  Consider aspirin in MSH2 carriers from age 25; "
                "  Ongoing: CAPP3 (dose finding) — 100mg aspirin may be sufficient."
            ),
        },
        {
            "term": "BRCA2-HBOC-HNC-2-3x-Oropharyngeal-PARP-Olaparib-Platinum",
            "definition": (
                "BRCA2/HBOC — OROPHARYNGEAL SCC 2-3x + PARP INHIBITOR OLAPARIB: "
                "HEAD & NECK CANCER IN BRCA2: "
                "  Oropharyngeal SCC: 2-3x elevated in germline BRCA2 carriers; "
                "  HPV-independent pathway proposed (BRCA2 loss → genomic instability → SCC regardless of HPV); "
                "  Oral SCC: 1.5-2x elevation; laryngeal: modest; "
                "  HNC in BRCA2 context: younger onset, often HPV-negative; "
                "PLATINUM SENSITIVITY — BRCA2-DEFICIENT HNC: "
                "  Cisplatin-based chemoradiation: standard for locally advanced HNC; "
                "  BRCA2-deficient HNC: HR-deficient → platinum-induced interstrand crosslinks → DSBs → tumour-specific synthetic lethality; "
                "  Clinical implication: BRCA2-mutated HNC may be more chemosensitive to standard cisplatin; "
                "PARP INHIBITORS — EMERGING FOR BRCA2 HNC: "
                "  Olaparib (Lynparza) FDA-approved: ovarian, breast, pancreatic, prostate; "
                "  NCI-MATCH arm A (olaparib): BRCA1/2-altered solid tumours (including HNC) — responses documented; "
                "  PARP+cetuximab combination: trials in BRCA-mutated recurrent/metastatic HNC; "
                "  Niraparib, talazoparib, rucaparib: alternative PARP inhibitors; "
                "FANCONI ANEMIA D1 — BIALLELIC BRCA2: "
                "  Biallelic BRCA2 = most severe FA phenotype (FA-D1); "
                "  VACTERL-H + early-onset AML (median 2yr) + medulloblastoma + Wilms tumour; "
                "  AVOID RT in FA-D1: same principles as FANCA; "
                "HBOC SURVEILLANCE: "
                "  Annual breast MRI + mammogram from age 30; "
                "  Annual pelvic USS + CA-125 from age 30; "
                "  BSO age 40-45 after childbearing (85% ovarian risk reduction); "
                "  Annual PSA (males) from age 40 (prostate 5-8x); "
                "  Annual ENT (HNC 2-3x)."
            ),
        },
        {
            "term": "RECQL4-Rothmund-Thomson-Poikiloderma-PATHOGNOMONIC-SCC-Head-Osteosarcoma",
            "definition": (
                "RECQL4 / ROTHMUND-THOMSON SYNDROME — POIKILODERMA CONGENITALE = PATHOGNOMONIC: "
                "POIKILODERMA CONGENITALE — DIAGNOSTIC HALLMARK: "
                "  Erythematous rash face + hands + feet onset age 3-6 months; "
                "  Evolves to poikiloderma: reticulated mix of hyperpigmentation, hypopigmentation, telangiectasias, skin atrophy; "
                "  FACE: PATHOGNOMONIC distribution — butterfly-pattern facial poikiloderma in RTS infant; "
                "  Without this finding, RTS unlikely; "
                "  Photosensitivity: severe sunburn on minimal exposure (UV-related); "
                "HEAD & NECK SCC IN RECQL4/RTS: "
                "  SCC of face, lips, ears, scalp: UV-triggered on poikilodermatous skin; "
                "  Onset: teens to young adulthood in UV-exposed individuals; "
                "  Multiple SCCs — field cancerisation on chronically damaged poikilodermatous skin; "
                "  UV avoidance (same principles as XP): UPF50+ clothing; SPF50+ sunscreen; UV-blocking glasses; "
                "OSTEOSARCOMA — MAJOR CANCER RISK: "
                "  30-40x general population risk; "
                "  Median age 11yr; distribution: proximal tibia + distal femur (same as sporadic); "
                "  ~30% cumulative lifetime risk of osteosarcoma in RTS; "
                "  Annual whole-body MRI from childhood (osteosarcoma + AML detection); "
                "  Treatment: MAP protocol (methotrexate + doxorubicin + cisplatin); "
                "AML / MDS: elevated risk in RECQL4; "
                "ALLELIC DISORDERS OF RECQL4: "
                "  Baller-Gerold Syndrome (BGS): RECQL4 biallelic → craniosynostosis + radial ray + mild poikiloderma; "
                "  RAPADILINO Syndrome: RECQL4 biallelic → radial ray + hypoplastic patella + short stature + sparse hair; "
                "  RTS type 1 (non-RECQL4): poikiloderma + osteosarcoma without RECQL4 mutation — unknown gene; "
                "  All allelic RECQL4 disorders: oncological surveillance mandatory."
            ),
        },
        {
            "term": "CASCADE-Testing-Hereditary-HNC-Germline-Panel",
            "definition": (
                "CASCADE TESTING FOR HEREDITARY HEAD & NECK CANCER PREDISPOSITION: "
                "WHEN TO TEST: "
                "  HNC (SCC/BCC) diagnosed age <40yr: full germline panel (FANCA, XPC, TP53, CDKN2A, ATM, MSH2, BRCA2, RECQL4); "
                "  Multiple HNC primaries (same or different HNC sites): test FANCA + TP53 + CDKN2A; "
                "  SCC face/scalp onset childhood: XPC germline priority; "
                "  Poikiloderma + SCC/osteosarcoma: RECQL4; "
                "  Sebaceous adenoma on biopsy: MMR IHC → MSH2/MSH6 germline; "
                "  Severe RT reaction / radiation necrosis at standard dose: FANCA + ATM + XPC; "
                "  Cerebellar ataxia + telangiectasias: ATM biallelic (A-T); "
                "  HNC + family history breast/ovarian/pancreatic: BRCA2 + CDKN2A; "
                "CASCADE FAMILY TESTING: "
                "  FANCA identified: test siblings (25% risk biallelic if both parents carriers); "
                "  TP53 identified: test first-degree relatives (50% risk monoallelic); "
                "  ATM biallelic: parents/siblings carrier testing; "
                "  MSH2/EPCAM: colonoscopy + urothelial surveillance in all mutation-positive family members; "
                "  BRCA2: cascade to children/siblings (50% monoallelic risk); "
                "TREATMENT DECISION IMPACT: "
                "  FANCA or ATM biallelic confirmed → avoid RT; modify chemo dose; "
                "  TP53 confirmed → avoid RT absolutely; "
                "  BRCA2 confirmed → consider PARP inhibitor; platinum chemo preferred; "
                "  MSH2/Lynch confirmed → pembrolizumab (PD-1) for MSI-H HNC (TMB-based); "
                "  CDKN2A confirmed → CDK4/6 inhibitor trials; pancreatic surveillance."
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Head-Neck-Cancer-Predisposition-Atlas",
        "seed_range":  f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
    print(json.dumps(generate_breakdown(), indent=2))
    print(json.dumps(generate_definitions(), indent=2))
