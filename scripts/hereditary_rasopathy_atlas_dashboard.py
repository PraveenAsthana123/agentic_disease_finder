#!/usr/bin/env python3
"""Hereditary-RASopathy-Atlas — Complete 8-Gene RAS-MAPK Pathway Atlas
PTPN11 (SHP2; 580 aa; 12q24.13; AD;
         Noonan syndrome type 1; ~50-70% of all Noonan;
         WEBBED NECK + PULMONARY STENOSIS + SHORT STATURE PATHOGNOMONIC;
         Lentigines/LEOPARD overlap; RAS-MAPK gain-of-function;
         seed SEED_BASE+0) ·
SOS1   (Son of Sevenless homolog 1; 1333 aa; 2p22.1; AD;
         Noonan syndrome type 4;
         NORMAL IQ distinguishes from other RASopathies;
         Ectodermal features prominent; mild-moderate cardiac;
         seed SEED_BASE+1) ·
RAF1   (RAF proto-oncogene serine/threonine-protein kinase; 648 aa; 3p25.2; AD;
         Noonan syndrome type 5;
         HYPERTROPHIC CARDIOMYOPATHY UNIQUE among RASopathies -- HCM 75%;
         Severe cardiac phenotype; PTPN11-negative Noonan with HCM -- RAF1;
         seed SEED_BASE+2) ·
BRAF   (B-Raf proto-oncogene; 766 aa; 7q34; AD;
         Cardio-Facio-Cutaneous syndrome type 1 (CFC1);
         CURLY/SPARSE HAIR + ICHTHYOSIS + KERATOSIS PILARIS PATHOGNOMONIC;
         More severe ID than Noonan; seizures 40-50%;
         seed SEED_BASE+3) ·
MAP2K1 (Mitogen-activated protein kinase kinase 1; 393 aa; 15q22.31; AD;
         Cardio-Facio-Cutaneous syndrome type 3 (CFC3);
         SEVERE ID; Noonan-like facies; MEK1 downstream of RAF;
         seed SEED_BASE+4) ·
HRAS   (Harvey rat sarcoma viral proto-oncogene; 189 aa; 11p15.5; AD;
         Costello syndrome;
         LOOSE REDUNDANT SKIN + PAPILLOMATA face/perianal PATHOGNOMONIC;
         HIGH CANCER 15% -- rhabdomyosarcoma + bladder carcinoma;
         FGFR crosstalk; Noonan overlap;
         seed SEED_BASE+5) ·
KRAS   (Kirsten rat sarcoma viral proto-oncogene; 189 aa; 12p12.1; AD;
         Noonan syndrome type 3 + CFC overlap;
         MOST SEVERE Noonan; HIGH AML RISK;
         G12V most common pathogenic variant;
         seed SEED_BASE+6) ·
LZTR1  (Leucine-zipper-like transcriptional regulator 1; 827 aa; 22q11.21;
         AD dominant-negative OR AR biallelic UNIQUE bidirectional inheritance;
         Noonan syndrome type 10 (Noonan 10);
         CBL-B/EGFR ubiquitin pathway;
         Schwannomatosis DDx in adults;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3070-3077)
"""
import random

SEED_BASE = 3070

ATLAS_GENES = [
    {
        "gene": "PTPN11",
        "protein": (
            "PTPN11 -- 12q24.13 Autosomal-Dominant-GOF -- 580aa -- SHP2-"
            "Protein-Tyrosine-Phosphatase-Non-Receptor-11-RAS-MAPK-Activator-"
            "Noonan-Syndrome-1-WEBBED-NECK-PULMONARY-STENOSIS-SHORT-STATURE-PATHOGNOMONIC-"
            "50-70pct-Noonan-LEOPARD-Overlap-OMIM-176876"
        ),
        "locus": "12q24.13",
        "protein_size": (
            "580 aa / 68 kDa (PTPN11; SHP2; protein tyrosine phosphatase non-receptor type 11; "
            "STRUCTURE: tandem SH2 domains (N-SH2, C-SH2) + catalytic PTP domain; "
            "FUNCTION: "
            "  SHP2 is a signal transducer phosphatase activated by receptor tyrosine kinases; "
            "  Normally autoinhibited: N-SH2 blocks PTP domain active site; "
            "  Activation: N-SH2 binds phospho-Tyr on receptors -- opens active site; "
            "  SHP2 dephosphorylates GAP-recruiting sites -- RAS stays GTP-loaded (active); "
            "  GOF mutations: N-SH2/C-SH2 interface disrupted -- constitutive activation; "
            "  Constitutively active SHP2 -- hyperactivates RAS-RAF-MEK-ERK pathway; "
            "NOONAN SYNDROME TYPE 1 (PTPN11): "
            "  PREVALENCE: PTPN11 accounts for ~50-70% of all Noonan syndrome; "
            "  PATHOGNOMONIC TRIAD: "
            "    (1) WEBBED NECK (pterygium colli): loose skin folds neck/nape; "
            "    (2) PULMONARY STENOSIS: dysplastic pulmonary valve; 50-70%; "
            "    (3) SHORT STATURE: height SDS -2 to -3; GH therapy approved; "
            "  CARDIAC: PS 65%; HCM ~15%; ASD 10-15%; "
            "  ID: mild-moderate ~50-70%; SOS1 has NORMAL IQ (key differentiator); "
            "  JMML: ~1-5% Noonan-PTPN11; haematology monitoring; "
            "  LEOPARD: PTP domain LOF variants -- lentigines + more HCM; "
            "  GROWTH HORMONE: FDA approved; cardiac echo BEFORE initiating; "
        ),
        "inheritance": (
            "Autosomal dominant GOF (de novo ~60%; familial ~40%); "
            "PTPN11 12q24.13; germline GOF in N-SH2 or C-SH2 domains; "
            "LEOPARD: PTP domain LOF -> dominant negative -> paradoxical RAS activation; "
        ),
        "disease_category": "Noonan Syndrome type 1 / RASopathy",
        "key_mutations": [
            "p.Asn308Asp (most common -- N-SH2; Noonan 1 archetypal)",
            "p.Tyr63Cys (N-SH2; common)",
            "p.Asp61Gly (N-SH2 interface)",
            "p.Gln79Arg (N-SH2)",
            "p.Thr468Met (PTP domain -- LEOPARD)",
            "p.Tyr279Cys (PTP domain -- LEOPARD, HCM)",
            "p.Glu76Lys (somatic -- JMML, NOT germline Noonan)",
        ],
        "clinical_keys": [
            "WEBBED NECK + PULMONARY STENOSIS + SHORT STATURE PATHOGNOMONIC NOONAN TRIAD",
            "50-70% of ALL Noonan syndrome -- test PTPN11 FIRST in Noonan panel",
            "SOS1 has NORMAL IQ -- key differentiator from PTPN11 Noonan",
            "Growth hormone FDA approved -- but CARDIAC ASSESSMENT FIRST (no GH if obstructive HCM)",
            "JMML risk: ~1-5% PTPN11 Noonan -- CBCs in early childhood",
            "LEOPARD syndrome: PTPN11 PTP domain mutations -- lentigines + more HCM",
            "Dysplastic pulmonary valve: balloon valvuloplasty less effective -- may need surgery",
            "Cryptorchidism 60-70% males -- orchidopexy before 18 months",
        ],
    },
    {
        "gene": "SOS1",
        "protein": (
            "SOS1 -- 2p22.1 Autosomal-Dominant-GOF -- 1333aa -- Son-of-Sevenless-Homolog-1-"
            "RAS-GEF-Guanine-Nucleotide-Exchange-Factor-Noonan-4-NORMAL-IQ-ECTODERMAL-FEATURES-"
            "MILD-CARDIAC-PULMONARY-STENOSIS-OMIM-182530"
        ),
        "locus": "2p22.1",
        "protein_size": (
            "1333 aa / 152 kDa (SOS1; Son of Sevenless homolog 1; "
            "STRUCTURE: DH domain + PH domain + REM domain + CDC25 (catalytic GEF) domain + PxxP motifs; "
            "FUNCTION: "
            "  SOS1 is the primary RAS guanine nucleotide exchange factor (GEF) in RTK signalling; "
            "  Activated downstream of RTK via GRB2-SOS1 complex; "
            "  SOS1 catalyses GDP->GTP exchange on RAS -> RAS activation; "
            "  GOF mutations: disrupt allosteric autoinhibition -> constitutive SOS1 -> RAS-GTP; "
            "NOONAN SYNDROME TYPE 4 (SOS1): "
            "  PREVALENCE: ~10-15% of Noonan syndrome (second most common after PTPN11); "
            "  KEY DISTINGUISHING FEATURE: NORMAL INTELLIGENCE (IQ); "
            "    SOS1-Noonan: majority have NORMAL IQ (mean ~100); "
            "    PTPN11-Noonan: ~50-70% have mild-moderate ID; "
            "    BRAF/MAP2K1-CFC: severe ID; "
            "    This is the MOST IMPORTANT clinical differentiator for SOS1; "
            "  ECTODERMAL FEATURES: "
            "    CURLY/SPARSE/WOOLLY HAIR; hyperkeratotic skin (keratosis pilaris variant); "
            "    Sparse eyebrows; these features more prominent than PTPN11; "
            "  CARDIAC: PS 30-40% (less than PTPN11); HCM rare (unlike RAF1); "
            "  SHORT STATURE: present; GH response good; HCM rare so GH safer; "
        ),
        "inheritance": (
            "Autosomal dominant GOF; de novo ~60%; familial ~40%; "
            "SOS1 2p22.1; missense GOF disrupts autoinhibition; "
            "Normal IQ in most -- important for family counselling; "
        ),
        "disease_category": "Noonan Syndrome type 4 / RASopathy",
        "key_mutations": [
            "p.Arg552Gly (most common; DH-PH interface)",
            "p.Thr266Ile (DH domain)",
            "p.Arg552Ser (DH-PH interface)",
            "p.Ile733Val (CDC25 domain)",
            "p.Met269Ile (DH domain)",
            "p.Phe890Leu (PxxP region)",
            "p.Leu550Pro (DH domain)",
        ],
        "clinical_keys": [
            "NORMAL IQ -- key differentiator from PTPN11 and BRAF/MAP2K1; counsel families accordingly",
            "ECTODERMAL FEATURES: curly/woolly hair + keratosis pilaris more prominent than PTPN11",
            "HCM rare in SOS1 -- growth hormone safer to initiate than in RAF1",
            "Pulmonary stenosis 30-40% (less common than PTPN11); ASD secondary",
            "10-15% of all Noonan syndrome -- test after PTPN11 negative",
            "SOS1 GOF: disrupts autoinhibition -> constitutive RAS-GEF activity",
            "MEK inhibitor trials: SOS1 included (binimetinib, trametinib)",
            "Learning difficulties in minority despite normal IQ -- neuropsychological assessment useful",
        ],
    },
    {
        "gene": "RAF1",
        "protein": (
            "RAF1 -- 3p25.2 Autosomal-Dominant-GOF -- 648aa -- RAF-Proto-Oncogene-1-"
            "Serine-Threonine-Protein-Kinase-RAS-Effector-MEK-ERK-Activator-"
            "Noonan-5-HYPERTROPHIC-CARDIOMYOPATHY-75pct-UNIQUE-RASOPATHY-CARDIAC-PHENOTYPE-OMIM-164760"
        ),
        "locus": "3p25.2",
        "protein_size": (
            "648 aa / 73 kDa (RAF1; cRAF; proto-oncogene serine/threonine kinase RAF1; "
            "STRUCTURE: CR1 (RAS binding + cysteine-rich domain) + CR2 (14-3-3 binding) + "
            "CR3 (kinase domain); "
            "FUNCTION: "
            "  RAF1 is a serine/threonine kinase; central component of RAS-RAF-MEK-ERK cascade; "
            "  RAS-GTP binds RAF1 CR1 -- membrane recruitment -- RAF1 activation; "
            "  Active RAF1 -- phosphorylates MEK1/2 -- ERK1/2 activation; "
            "  GOF mutations: disrupt 14-3-3 inhibitory binding -- constitutive kinase activity; "
            "NOONAN SYNDROME TYPE 5 (RAF1): "
            "  PREVALENCE: ~5% of Noonan syndrome; "
            "  DEFINING FEATURE -- HYPERTROPHIC CARDIOMYOPATHY (HCM): "
            "    HCM present in ~75% of RAF1-Noonan (vs ~15% PTPN11; virtually absent SOS1); "
            "    RAF1-HCM: most common single-gene cause of Noonan-HCM; "
            "    Obstructive HCM (HOCM): subaortic obstruction; LVOT gradient; neonatal onset possible; "
            "  PULMONARY STENOSIS: ~35% RAF1-Noonan; combined HCM+PS characteristic; "
            "  CLINICAL RULE: PTPN11-NEGATIVE NOONAN WITH HCM -- RAF1 MUST BE TESTED; "
            "  GH CAUTION: GH can promote ventricular hypertrophy -- CONTRAINDICATED in obstructive HCM; "
            "  MEK inhibitors (trametinib): reduce HCM in preclinical models; trials ongoing; "
        ),
        "inheritance": (
            "Autosomal dominant GOF; de novo majority; "
            "RAF1 3p25.2; GOF disrupts 14-3-3 autoinhibitory binding; "
            "High de novo rate; variable expressivity of HCM even within families; "
        ),
        "disease_category": "Noonan Syndrome type 5 / RASopathy-HCM",
        "key_mutations": [
            "p.Leu613Val (most common; CR2/14-3-3 binding; HCM 90%)",
            "p.Ser257Leu (CR2; HCM)",
            "p.Pro261Thr (CR2; HCM)",
            "p.Asp486Gly (kinase domain)",
            "p.Tyr340His (CR2; HCM; severe)",
            "p.Leu613Pro (CR2; severe HCM)",
            "p.Ser259Phe (CR2)",
        ],
        "clinical_keys": [
            "HYPERTROPHIC CARDIOMYOPATHY 75% -- UNIQUE to RAF1 among RASopathies; echo MANDATORY",
            "PTPN11-negative Noonan + HCM -- TEST RAF1 FIRST",
            "GROWTH HORMONE CONTRAINDICATED in obstructive HCM -- cardiac echo before initiating GH",
            "Obstructive HCM: beta-blockers first-line; septal myectomy for severe LVOT obstruction",
            "Combined HCM + pulmonary stenosis: characteristic RAF1 pattern",
            "MEK inhibitor trials (trametinib): targeting RAF->MEK to reduce HCM",
            "Neonatal HCM: may be severe and require intensive care at birth",
            "p.Leu613Val most common -- CR2 domain GOF; HCM in ~90% with this specific mutation",
        ],
    },
    {
        "gene": "BRAF",
        "protein": (
            "BRAF -- 7q34 Autosomal-Dominant-GOF -- 766aa -- B-Raf-Proto-Oncogene-"
            "Serine-Threonine-Kinase-MEK-ERK-Pathway-CFC1-CURLY-SPARSE-HAIR-"
            "ICHTHYOSIS-KERATOSIS-PILARIS-PATHOGNOMONIC-SEVERE-ID-SEIZURES-40-50pct-OMIM-164757"
        ),
        "locus": "7q34",
        "protein_size": (
            "766 aa / 84 kDa (BRAF; B-Raf serine/threonine kinase; "
            "STRUCTURE: N-terminal regulatory region + CR1 (RAS binding) + CR2 (14-3-3) + "
            "CR3 (kinase domain); "
            "FUNCTION: "
            "  BRAF is the most potent RAF isoform kinase; primary effector of RAS; "
            "  RAS-GTP -- BRAF -- MEK1/2 -- ERK1/2 (main axis); "
            "  GOF mutations: kinase domain (CR3) or DFG loop -- constitutive activity; "
            "  CFC BRAF mutations activate kinase WITHOUT oncogenic transformation; "
            "  Different from oncogenic BRAF V600E (melanoma) -- CFC mutations distinct; "
            "CARDIO-FACIO-CUTANEOUS SYNDROME TYPE 1 (BRAF): "
            "  PREVALENCE: BRAF most common CFC gene (~75% CFC); "
            "  PATHOGNOMONIC ECTODERMAL TRIAD: "
            "    (1) CURLY, SPARSE, OR ABSENT HAIR (friable, brittle, absent eyebrows/lashes); "
            "    (2) ICHTHYOSIS: dry scaly skin; keratinisation defect; "
            "    (3) KERATOSIS PILARIS: follicular keratosis arms/thighs/cheeks; "
            "  SEVERE ID: profound in many; absent speech majority; "
            "  SEIZURES: 40-50%; West syndrome/LGS; drug-resistant; "
            "  CARDIAC: PS 55%; HCM 35%; gastrostomy universal; "
            "  IMPORTANT: BRAF V600E (oncogenic somatic) NOT same as CFC germline BRAF; "
            "  Trametinib compassionate use: seizure reduction + skin improvement; "
        ),
        "inheritance": (
            "Autosomal dominant GOF; de novo in virtually all CFC cases; "
            "BRAF 7q34; GOF kinase domain or regulatory domain mutations; "
            "Distinct mutations from oncogenic BRAF V600E; "
        ),
        "disease_category": "Cardio-Facio-Cutaneous Syndrome type 1 / RASopathy",
        "key_mutations": [
            "p.Gln257Arg (kinase domain N-lobe -- most common CFC)",
            "p.Glu501Lys (kinase domain)",
            "p.Asn581Ser (DFG loop)",
            "p.Lys499Glu (kinase domain)",
            "p.Thr241Pro (CR2)",
            "p.Arg461Gly (kinase domain)",
            "p.Asn581Asp (DFG loop)",
        ],
        "clinical_keys": [
            "CURLY/SPARSE HAIR + ICHTHYOSIS + KERATOSIS PILARIS PATHOGNOMONIC ECTODERMAL CFC TRIAD",
            "SEVERE ID -- more profound than PTPN11/SOS1 Noonan; absent speech majority",
            "SEIZURES 40-50% -- West syndrome/LGS pattern; drug-resistant common",
            "BRAF accounts for ~75% of all CFC syndrome",
            "Gastrostomy often needed -- severe feeding difficulties from birth",
            "HCM 30-40% -- less than RAF1 but more than SOS1; echo mandatory",
            "Trametinib compassionate use: seizure reduction + ectodermal improvement",
            "BRAF V600E (somatic oncogenic) is NOT the same mutation as CFC germline BRAF -- do not conflate",
        ],
    },
    {
        "gene": "MAP2K1",
        "protein": (
            "MAP2K1 -- 15q22.31 Autosomal-Dominant-GOF -- 393aa -- Mitogen-Activated-Protein-Kinase-Kinase-1-"
            "MEK1-ERK-Activator-RAS-RAF-MEK-ERK-Cascade-CFC3-SEVERE-ID-NOONAN-LIKE-FACIES-"
            "CARDIO-FACIO-CUTANEOUS-3-OMIM-176872"
        ),
        "locus": "15q22.31",
        "protein_size": (
            "393 aa / 44 kDa (MAP2K1; MEK1; mitogen-activated protein kinase kinase 1; "
            "STRUCTURE: N-terminal regulatory helix (autoinhibitory) + kinase domain + DEF docking site; "
            "FUNCTION: "
            "  MEK1 is the dual-specificity kinase directly downstream of RAF isoforms; "
            "  RAF -- phosphorylates MEK1 on Ser218/Ser222 -- MEK1 activated; "
            "  Active MEK1 -- phosphorylates ERK1/2 -- nuclear signalling -- proliferation; "
            "  GOF mutations: disrupt N-terminal autoinhibitory helix -- constitutive MEK1 activity; "
            "CARDIO-FACIO-CUTANEOUS SYNDROME TYPE 3 (MAP2K1): "
            "  PREVALENCE: ~10-15% of CFC syndrome (MAP2K2 ~5%); "
            "  PHENOTYPE: overlaps with BRAF-CFC but typically somewhat milder; "
            "  SEVERE ID: similar to BRAF; absent speech; non-ambulant subset; "
            "  NOONAN-LIKE FACIES: hypertelorism; low-set ears; short stature; "
            "  ECTODERMAL: sparse/curly hair; keratosis pilaris (may be milder than BRAF); "
            "  CARDIAC: PS + HCM present; less severe than RAF1; "
            "  SEIZURES: 30-40% (similar to BRAF); gastrostomy common; "
            "  DIRECT MEK INHIBITOR TARGET: MAP2K1 GOF directly targeted by trametinib/binimetinib; "
        ),
        "inheritance": (
            "Autosomal dominant GOF; de novo in virtually all cases; "
            "MAP2K1 15q22.31; GOF disrupts autoinhibitory N-terminal helix; "
        ),
        "disease_category": "Cardio-Facio-Cutaneous Syndrome type 3 / RASopathy-MEK",
        "key_mutations": [
            "p.Tyr130Cys (autoinhibitory helix N-lobe -- most common CFC)",
            "p.Phe53Leu (N-terminal regulatory)",
            "p.Ile99Val (regulatory region)",
            "p.Ile103Thr (regulatory region)",
            "p.Cys121Ser (regulatory helix)",
            "p.Phe53Ser (N-terminal)",
            "p.Glu102Gly (regulatory helix)",
        ],
        "clinical_keys": [
            "SEVERE ID -- similar severity to BRAF; absent speech; non-ambulant subset",
            "MAP2K1 GOF directly targeted by MEK inhibitors -- trametinib most relevant gene target",
            "CFC type 3 -- test after BRAF negative in suspected CFC; BRAF>>MAP2K1 in frequency",
            "Seizures 30-40%; hypotonia universal; feeding difficulties; gastrostomy common",
            "Ectodermal features: sparse/curly hair + keratosis pilaris (may be milder than BRAF)",
            "Cardiac: PS + HCM present; less severe than RAF1",
            "CFC vs Noonan distinction: CFC has severe ID + ectodermal; Noonan milder + no ectodermal",
            "MEK1 is direct downstream target of all RAF isoforms -- inhibitors address entire RAS-RAF-MEK axis",
        ],
    },
    {
        "gene": "HRAS",
        "protein": (
            "HRAS -- 11p15.5 Autosomal-Dominant-GOF -- 189aa -- Harvey-Rat-Sarcoma-Viral-Proto-Oncogene-"
            "RAS-GTPase-Costello-Syndrome-LOOSE-REDUNDANT-SKIN-PAPILLOMATA-FACE-PERIANAL-PATHOGNOMONIC-"
            "CANCER-15pct-Rhabdomyosarcoma-Bladder-FGFR-CROSSTALK-OMIM-190020"
        ),
        "locus": "11p15.5",
        "protein_size": (
            "189 aa / 21 kDa (HRAS; H-RAS; Harvey rat sarcoma viral proto-oncogene; "
            "STRUCTURE: G-domain with 5 conserved elements (G1-G5) + effector loop (residues 30-40) + "
            "C-terminal CAAX motif (farnesylation); "
            "FUNCTION: "
            "  HRAS is a small GTPase -- binary switch: inactive (GDP) / active (GTP); "
            "  GOF mutations: impair intrinsic GTPase activity or GAP-mediated hydrolysis; "
            "  HRAS locked in GTP-bound active state; "
            "  Most common Costello mutations: HRAS codon 12/13 (G12S, G12A, G12V etc.); "
            "  HRAS p.Gly12Ser: most common Costello germline mutation (~85% Costello); "
            "  FGFR1/3 CROSSTALK: HRAS activates same ERK pathway as FGFR signalling; "
            "  Tipifarnib/lonafarnib: farnesyl transferase inhibitors -- block HRAS membrane localisation; "
            "COSTELLO SYNDROME (HRAS): "
            "  PATHOGNOMONIC FEATURES: "
            "    (1) LOOSE REDUNDANT SKIN: deep palmar/plantar creases; loose neck skin; pathognomonic; "
            "    (2) PAPILLOMATA: PERINASAL + PERIANAL distribution PATHOGNOMONIC; appear childhood; "
            "  CANCER RISK 15% LIFETIME: "
            "    RHABDOMYOSARCOMA + BLADDER CARCINOMA; surveillance MANDATORY; "
            "    Abdominal USS every 6-12 months to age 8; urinalysis every 6 months from age 10; "
            "  MULTIFOCAL ATRIAL TACHYCARDIA: characteristic arrhythmia; Holter monitoring; "
            "  HCM 30-50%; severe feeding difficulties; gastrostomy universal; "
        ),
        "inheritance": (
            "Autosomal dominant GOF; virtually all de novo; "
            "HRAS 11p15.5; GOF codon 12/13 mutations impair GTPase activity; "
            "HRAS p.Gly12Ser ~85% of Costello syndrome; "
        ),
        "disease_category": "Costello Syndrome / RASopathy-Cancer",
        "key_mutations": [
            "p.Gly12Ser (most common -- ~85% Costello; papillomata + cancer risk high)",
            "p.Gly12Ala (milder cardiac; lower cancer risk)",
            "p.Gly12Val (higher cancer risk; somatic overlap)",
            "p.Gly12Asp (severe)",
            "p.Gly12Cys (moderate)",
            "p.Gly13Cys (rare variant)",
            "p.Lys117Arg (severe; highest cancer risk)",
        ],
        "clinical_keys": [
            "LOOSE REDUNDANT SKIN (palmar/plantar deep creases) + PAPILLOMATA (face+perianal) PATHOGNOMONIC",
            "CANCER 15% lifetime -- rhabdomyosarcoma + bladder carcinoma; USS + urinalysis SURVEILLANCE MANDATORY",
            "Abdominal USS every 6-12 months to age 8; urinalysis every 6 months from age 10",
            "MULTIFOCAL ATRIAL TACHYCARDIA: characteristic arrhythmia; Holter monitoring essential",
            "p.Gly12Ser accounts for ~85% Costello -- test this first",
            "FGFR crosstalk: tipifarnib/lonafarnib (farnesyl transfer inhibitors) rationale",
            "Severe feeding difficulties -- gastrostomy/NG tube universal in early infancy",
            "HCM 30-50%; neonatal hypoglycaemia; growth hormone less effective than Noonan",
        ],
    },
    {
        "gene": "KRAS",
        "protein": (
            "KRAS -- 12p12.1 Autosomal-Dominant-GOF -- 189aa -- Kirsten-Rat-Sarcoma-Viral-Proto-Oncogene-"
            "RAS-GTPase-Noonan-3-CFC-Overlap-MOST-SEVERE-NOONAN-HIGH-AML-RISK-"
            "G12V-Most-Common-Pathogenic-OMIM-190070"
        ),
        "locus": "12p12.1",
        "protein_size": (
            "189 aa / 21 kDa (KRAS; K-RAS; Kirsten rat sarcoma viral proto-oncogene; "
            "STRUCTURE: G-domain (G1-G5 elements; P-loop/G1 residues 10-17) + "
            "switch I (30-40) + switch II (58-72) + CAAX farnesylation motif; "
            "FUNCTION: "
            "  KRAS is the most commonly mutated RAS in cancer (>80% pancreatic, 30% lung, 40% colon); "
            "  SAME GTPase switch mechanism as HRAS; "
            "  Germline KRAS GOF -- RASopathy (Noonan-3 or CFC phenotype); "
            "  Somatic KRAS G12D/G12V/G12C -- cancer (different variants and mechanism); "
            "  G12V MOST COMMON GERMLINE PATHOGENIC KRAS VARIANT; "
            "  KRAS-Noonan: MOST SEVERE form of Noonan syndrome; "
            "AML/LEUKAEMIA RISK: "
            "  HIGH AML RISK vs other Noonan genes; JMML elevated risk; "
            "  KRAS somatic G12D/V causes up to 20-30% of sporadic JMML; "
            "  Germline KRAS Noonan: haematopoietic malignancy monitoring MANDATORY; "
            "  CBCs at every visit; low threshold for haematology referral; "
            "CLINICAL RULE -- GERMLINE vs SOMATIC KRAS: "
            "  Somatic KRAS: cancer driver in tumour; germline KRAS: RASopathy in blood; "
            "  VAF ~50% in blood -> germline RASopathy; do NOT panic about cancer; "
        ),
        "inheritance": (
            "Autosomal dominant GOF; de novo majority; "
            "KRAS 12p12.1; GOF impairs GTPase + GAP interaction; "
            "Variable: same mutation may cause Noonan vs CFC phenotype; "
        ),
        "disease_category": "Noonan Syndrome type 3 / CFC-overlap / Most-Severe-RASopathy",
        "key_mutations": [
            "p.Gly12Val (most common germline pathogenic -- p.G12V; JMML overlap)",
            "p.Gly12Asp (severe; CFC-like)",
            "p.Thr58Ile (switch II; Noonan)",
            "p.Val14Ile (G1 loop; Noonan-like)",
            "p.Asp153Val (effector loop)",
            "p.Pro34Arg (switch I; severe)",
            "p.Leu19Phe (G1 loop)",
        ],
        "clinical_keys": [
            "MOST SEVERE NOONAN -- severe ID, cardiac disease, structural brain anomalies",
            "HIGH AML/JMML RISK -- haematology monitoring MANDATORY; CBCs regularly",
            "p.Gly12Val most common germline KRAS -- distinguish from somatic oncogenic p.G12V",
            "Germline KRAS blood p.G12V: RASopathy, NOT somatic cancer -- context critical",
            "CFC overlap: some KRAS mutations -> CFC phenotype (ectodermal + severe ID)",
            "KRAS accounts for ~1-2% Noonan; ~5% CFC -- rare but most severe",
            "Haematopoietic malignancy surveillance: CBC at each visit; low threshold for referral",
            "Cardiac: severe HCM + PS combination possible; intensive echo monitoring",
        ],
    },
    {
        "gene": "LZTR1",
        "protein": (
            "LZTR1 -- 22q11.21 AD-dominant-negative-OR-AR-biallelic-UNIQUE -- 827aa -- "
            "Leucine-Zipper-Like-Transcriptional-Regulator-1-"
            "CUL3-E3-Ubiquitin-Ligase-Adaptor-RAS-Ubiquitination-Pathway-"
            "Noonan-10-BIDIRECTIONAL-INHERITANCE-UNIQUE-Schwannomatosis-DDx-OMIM-600574"
        ),
        "locus": "22q11.21",
        "protein_size": (
            "827 aa / 92 kDa (LZTR1; leucine-zipper-like transcriptional regulator 1; "
            "STRUCTURE: BTB-POZ domain (N-terminal) + Kelch-like propeller domain (C-terminal); "
            "FUNCTION: "
            "  LZTR1 is a substrate adaptor for CUL3-RING E3 ubiquitin ligase complex; "
            "  CUL3-LZTR1 complex: ubiquitinates RAS proteins (KRAS, HRAS, MRAS) -> proteasomal degradation; "
            "  Loss of LZTR1 -> reduced RAS ubiquitination -> increased RAS activity; "
            "  LZTR1 is a NEGATIVE REGULATOR of RAS (distinct from GOF mutations in other genes); "
            "UNIQUE BIDIRECTIONAL INHERITANCE: "
            "  DOMINANT MODE (heterozygous): dominant-negative; single mutant poisons CUL3 complex; "
            "    MILDER phenotype; 50% recurrence risk; "
            "  RECESSIVE MODE (biallelic): true LOF both alleles; MORE SEVERE phenotype; "
            "    25% recurrence risk (both parents carriers); "
            "  PARENTAL TESTING MANDATORY: completely changes recurrence risk counselling; "
            "SCHWANNOMATOSIS-2 DDx: "
            "  Same gene; truncating LOF -> schwannomas (adult onset, no Noonan features); "
            "  Missense dominant-negative -> Noonan-10 (childhood, cardiac, ID); "
            "  Child with Noonan + family schwannomas -> LZTR1; "
            "22q11.21: adjacent to DiGeorge region; large 22q11.2 deletions may include LZTR1; "
        ),
        "inheritance": (
            "UNIQUE bidirectional: "
            "AD (heterozygous dominant-negative) OR AR (biallelic LOF); "
            "LZTR1 22q11.21; parental testing MANDATORY to distinguish; "
            "AD: 50% risk; AR: 25% risk; only common RASopathy gene with both AD and AR; "
        ),
        "disease_category": "Noonan Syndrome type 10 / RASopathy-Bidirectional-Inheritance",
        "key_mutations": [
            "p.Arg688Cys (BTB-POZ domain; dominant-negative; most common Noonan-10 het)",
            "p.Arg709Gln (BTB domain; dominant-negative)",
            "p.Tyr119Cys (Kelch domain; AR compound het)",
            "p.Gly248Arg (Kelch domain)",
            "Exon deletions (LOF; schwannomatosis)",
            "p.Tyr276Cys (Kelch domain)",
            "p.Arg344Trp (Kelch repeat)",
        ],
        "clinical_keys": [
            "UNIQUE BIDIRECTIONAL INHERITANCE: AD dominant-negative OR AR biallelic -- TEST BOTH PARENTS",
            "AD het: 50% recurrence risk; AR biallelic: 25% risk -- completely different counselling",
            "Schwannomatosis-2 DDx: truncating LZTR1 LOF causes schwannomas in adults -- same gene",
            "Child with Noonan + family schwannomas -> LZTR1; adult schwannomas + Noonan family -> LZTR1",
            "22q11.21 locus: large 22q11.2 deletions may include LZTR1 -- check CMA report",
            "Biallelic LZTR1 (AR mode): MORE SEVERE than heterozygous dominant-negative",
            "CUL3-LZTR1 ubiquitinates RAS (KRAS/HRAS/MRAS) -- negative regulator lost in Noonan-10",
            "Test LZTR1 in Noonan panel; prioritise if other common genes negative",
        ],
    },
]


def _generate_patients_for_gene(gene: str, seed: int) -> list:
    """Generate 40 synthetic patients for a single RASopathy gene."""
    rng = random.Random(seed)
    patients = []

    gene_params = {
        "PTPN11": {
            "iq_range": (50, 90), "epilepsy_rate": 0.10, "speech_absent_rate": 0.10,
            "walk_rate": 0.95, "cardiac_hcm_rate": 0.15, "ps_rate": 0.65,
            "short_stature_rate": 0.90, "webbed_neck_rate": 0.45, "loose_skin_rate": 0.05,
            "papillomata_rate": 0.02, "cancer_risk_rate": 0.03, "ectodermal_rate": 0.15,
            "autism_rate": 0.20, "dx_months": (1, 24), "severity_dist": (0.15, 0.50, 0.35),
            "refractory_epilepsy_rate": 0.05,
            "mutations": ["p.Asn308Asp", "p.Tyr63Cys", "p.Asp61Gly", "p.Gln79Arg",
                          "p.Thr468Met", "p.Tyr279Cys", "p.Glu76Lys"],
        },
        "SOS1": {
            "iq_range": (75, 110), "epilepsy_rate": 0.08, "speech_absent_rate": 0.05,
            "walk_rate": 0.98, "cardiac_hcm_rate": 0.05, "ps_rate": 0.35,
            "short_stature_rate": 0.85, "webbed_neck_rate": 0.30, "loose_skin_rate": 0.03,
            "papillomata_rate": 0.01, "cancer_risk_rate": 0.01, "ectodermal_rate": 0.50,
            "autism_rate": 0.12, "dx_months": (3, 36), "severity_dist": (0.05, 0.30, 0.65),
            "refractory_epilepsy_rate": 0.02,
            "mutations": ["p.Arg552Gly", "p.Thr266Ile", "p.Arg552Ser", "p.Ile733Val",
                          "p.Met269Ile", "p.Phe890Leu", "p.Leu550Pro"],
        },
        "RAF1": {
            "iq_range": (45, 85), "epilepsy_rate": 0.15, "speech_absent_rate": 0.15,
            "walk_rate": 0.90, "cardiac_hcm_rate": 0.75, "ps_rate": 0.35,
            "short_stature_rate": 0.88, "webbed_neck_rate": 0.40, "loose_skin_rate": 0.05,
            "papillomata_rate": 0.02, "cancer_risk_rate": 0.02, "ectodermal_rate": 0.25,
            "autism_rate": 0.18, "dx_months": (0, 18), "severity_dist": (0.25, 0.50, 0.25),
            "refractory_epilepsy_rate": 0.08,
            "mutations": ["p.Leu613Val", "p.Ser257Leu", "p.Pro261Thr", "p.Asp486Gly",
                          "p.Tyr340His", "p.Leu613Pro", "p.Ser259Phe"],
        },
        "BRAF": {
            "iq_range": (20, 60), "epilepsy_rate": 0.45, "speech_absent_rate": 0.55,
            "walk_rate": 0.70, "cardiac_hcm_rate": 0.35, "ps_rate": 0.55,
            "short_stature_rate": 0.92, "webbed_neck_rate": 0.35, "loose_skin_rate": 0.08,
            "papillomata_rate": 0.05, "cancer_risk_rate": 0.04, "ectodermal_rate": 0.90,
            "autism_rate": 0.35, "dx_months": (1, 18), "severity_dist": (0.55, 0.35, 0.10),
            "refractory_epilepsy_rate": 0.30,
            "mutations": ["p.Gln257Arg", "p.Glu501Lys", "p.Asn581Ser", "p.Lys499Glu",
                          "p.Thr241Pro", "p.Arg461Gly", "p.Asn581Asp"],
        },
        "MAP2K1": {
            "iq_range": (20, 60), "epilepsy_rate": 0.38, "speech_absent_rate": 0.50,
            "walk_rate": 0.72, "cardiac_hcm_rate": 0.28, "ps_rate": 0.45,
            "short_stature_rate": 0.90, "webbed_neck_rate": 0.32, "loose_skin_rate": 0.06,
            "papillomata_rate": 0.03, "cancer_risk_rate": 0.03, "ectodermal_rate": 0.80,
            "autism_rate": 0.32, "dx_months": (1, 18), "severity_dist": (0.50, 0.38, 0.12),
            "refractory_epilepsy_rate": 0.25,
            "mutations": ["p.Tyr130Cys", "p.Phe53Leu", "p.Ile99Val", "p.Ile103Thr",
                          "p.Cys121Ser", "p.Phe53Ser", "p.Glu102Gly"],
        },
        "HRAS": {
            "iq_range": (30, 70), "epilepsy_rate": 0.20, "speech_absent_rate": 0.30,
            "walk_rate": 0.80, "cardiac_hcm_rate": 0.40, "ps_rate": 0.40,
            "short_stature_rate": 0.95, "webbed_neck_rate": 0.40, "loose_skin_rate": 0.92,
            "papillomata_rate": 0.75, "cancer_risk_rate": 0.15, "ectodermal_rate": 0.60,
            "autism_rate": 0.22, "dx_months": (0, 12), "severity_dist": (0.40, 0.42, 0.18),
            "refractory_epilepsy_rate": 0.12,
            "mutations": ["p.Gly12Ser", "p.Gly12Ala", "p.Gly12Val", "p.Gly12Asp",
                          "p.Gly12Cys", "p.Gly13Cys", "p.Lys117Arg"],
        },
        "KRAS": {
            "iq_range": (20, 55), "epilepsy_rate": 0.30, "speech_absent_rate": 0.50,
            "walk_rate": 0.68, "cardiac_hcm_rate": 0.45, "ps_rate": 0.50,
            "short_stature_rate": 0.93, "webbed_neck_rate": 0.42, "loose_skin_rate": 0.10,
            "papillomata_rate": 0.08, "cancer_risk_rate": 0.12, "ectodermal_rate": 0.50,
            "autism_rate": 0.30, "dx_months": (0, 12), "severity_dist": (0.55, 0.35, 0.10),
            "refractory_epilepsy_rate": 0.18,
            "mutations": ["p.Gly12Val", "p.Gly12Asp", "p.Thr58Ile", "p.Val14Ile",
                          "p.Asp153Val", "p.Pro34Arg", "p.Leu19Phe"],
        },
        "LZTR1": {
            "iq_range": (45, 90), "epilepsy_rate": 0.12, "speech_absent_rate": 0.15,
            "walk_rate": 0.92, "cardiac_hcm_rate": 0.18, "ps_rate": 0.50,
            "short_stature_rate": 0.85, "webbed_neck_rate": 0.35, "loose_skin_rate": 0.03,
            "papillomata_rate": 0.01, "cancer_risk_rate": 0.05, "ectodermal_rate": 0.12,
            "autism_rate": 0.16, "dx_months": (2, 36), "severity_dist": (0.20, 0.48, 0.32),
            "refractory_epilepsy_rate": 0.06,
            "mutations": ["p.Arg688Cys", "p.Arg709Gln", "p.Tyr119Cys", "p.Gly248Arg",
                          "Exon deletion LOF", "p.Tyr276Cys", "p.Arg344Trp"],
        },
    }

    p = gene_params.get(gene, gene_params["PTPN11"])
    sev_w = p["severity_dist"]

    for i in range(40):
        iq = rng.randint(*p["iq_range"])
        severity = rng.choices(["severe", "moderate", "mild"], weights=sev_w, k=1)[0]
        sex = rng.choice(["F", "M"])
        dx_mo = rng.randint(*p["dx_months"])
        mutation = rng.choice(p["mutations"])

        cardiac_hcm = rng.random() < p["cardiac_hcm_rate"]
        pulmonary_stenosis = rng.random() < p["ps_rate"]
        short_stature = rng.random() < p["short_stature_rate"]
        webbed_neck = rng.random() < p["webbed_neck_rate"]
        loose_skin = rng.random() < p["loose_skin_rate"]
        papillomata = rng.random() < p["papillomata_rate"]
        cancer_risk_flag = rng.random() < p["cancer_risk_rate"]
        ectodermal_features = rng.random() < p["ectodermal_rate"]

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "sex": sex,
            "iq_estimate": iq,
            "severity": severity,
            "age_at_diagnosis_mo": dx_mo,
            "mutation": mutation,
            "epilepsy": rng.random() < p["epilepsy_rate"],
            "refractory_epilepsy": rng.random() < p["refractory_epilepsy_rate"],
            "speech_absent": rng.random() < p["speech_absent_rate"],
            "independent_walk": rng.random() < p["walk_rate"],
            "cardiac_hcm": cardiac_hcm,
            "pulmonary_stenosis": pulmonary_stenosis,
            "short_stature": short_stature,
            "webbed_neck": webbed_neck,
            "loose_skin": loose_skin,
            "papillomata": papillomata,
            "cancer_risk_flag": cancer_risk_flag,
            "ectodermal_features": ectodermal_features,
            "autism_features": rng.random() < p["autism_rate"],
            "somatic_variant": False,  # All RASopathy genes in this atlas are germline
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-RASopathy-Atlas."""
    return {
        "atlas":          "Hereditary-RASopathy-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary RAS-MAPK Pathway Atlas "
            "(PTPN11-SOS1-RAF1-BRAF-MAP2K1-HRAS-KRAS-LZTR1)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "PTPN11": (
                "AD GOF 12q24.13 (SHP2; ~50-70% all Noonan; WEBBED NECK + PULMONARY STENOSIS + SHORT STATURE "
                "PATHOGNOMONIC; HCM ~15% (much less than RAF1); JMML 1-5%; LEOPARD overlap -- PTP domain LOF; "
                "Growth hormone FDA approved; test PTPN11 FIRST in Noonan panel)"
            ),
            "SOS1": (
                "AD GOF 2p22.1 (RAS-GEF; ~10-15% Noonan; NORMAL IQ distinguishes from all other RASopathies; "
                "ectodermal features prominent (curly/woolly hair + keratosis pilaris); HCM rare; PS 30-40%; "
                "milder overall; growth hormone safer than RAF1)"
            ),
            "RAF1": (
                "AD GOF 3p25.2 (cRAF kinase; ~5% Noonan; HYPERTROPHIC CARDIOMYOPATHY 75% UNIQUE among RASopathies; "
                "PTPN11-negative Noonan + HCM -- TEST RAF1 FIRST; GH contraindicated if obstructive HCM; "
                "MEK inhibitor trials for HCM; p.Leu613Val most common)"
            ),
            "BRAF": (
                "AD GOF 7q34 (B-Raf kinase; ~75% CFC syndrome; CFC type 1; CURLY/SPARSE HAIR + ICHTHYOSIS + "
                "KERATOSIS PILARIS PATHOGNOMONIC ectodermal triad; severe ID; seizures 40-50%; gastrostomy; "
                "HCM 30-40%; BRAF V600E somatic oncogenic NOT same as CFC germline mutations)"
            ),
            "MAP2K1": (
                "AD GOF 15q22.31 (MEK1; ~10-15% CFC; CFC type 3; severe ID; Noonan-like facies + ectodermal; "
                "direct MEK inhibitor target (trametinib most relevant); seizures 30-40%; test after BRAF negative)"
            ),
            "HRAS": (
                "AD GOF 11p15.5 (H-RAS GTPase; Costello syndrome; LOOSE REDUNDANT SKIN (deep palmar/plantar creases) "
                "+ PAPILLOMATA face+perianal PATHOGNOMONIC; CANCER 15% (rhabdomyosarcoma + bladder); "
                "multifocal atrial tachycardia; p.Gly12Ser ~85% Costello; cancer surveillance MANDATORY)"
            ),
            "KRAS": (
                "AD GOF 12p12.1 (K-RAS GTPase; ~1-2% Noonan; MOST SEVERE Noonan; HIGH AML/JMML RISK; "
                "G12V most common germline pathogenic; CFC overlap; distinguish germline vs somatic; "
                "haematology monitoring mandatory)"
            ),
            "LZTR1": (
                "UNIQUE BIDIRECTIONAL: AD dominant-negative OR AR biallelic -- only RASopathy gene with both modes; "
                "22q11.21; Noonan-10; TEST BOTH PARENTS to determine AD vs AR (completely different recurrence risk); "
                "Schwannomatosis-2 DDx (same gene, truncating LOF -> schwannomas in adults)"
            ),
        },
        "key_clinical_rules": [
            "PTPN11 accounts for 50-70% of ALL Noonan syndrome -- test PTPN11 FIRST in any Noonan panel; RASopathy panel if negative",
            "SOS1-Noonan has NORMAL IQ -- the single most important differentiator; counsel families about normal cognitive expectation",
            "RAF1: HYPERTROPHIC CARDIOMYOPATHY 75% -- PTPN11-negative Noonan with HCM -- test RAF1 immediately; most common Noonan-HCM gene",
            "GROWTH HORMONE FDA approved for Noonan short stature BUT cardiac assessment first -- CONTRAINDICATED in obstructive RAF1-HCM",
            "BRAF/MAP2K1 cause CFC syndrome -- CURLY/SPARSE HAIR + ICHTHYOSIS + KERATOSIS PILARIS ectodermal triad distinguishes CFC from Noonan",
            "HRAS-Costello: CANCER SURVEILLANCE MANDATORY (15% lifetime) -- USS every 6-12 months to age 8; urinalysis every 6 months from age 10",
            "HRAS p.Gly12Ser accounts for ~85% Costello syndrome -- test this hotspot first when Costello features present",
            "KRAS germline p.Gly12Val: RASopathy NOT oncogenic cancer risk -- context critical; distinguish from somatic KRAS in tumours",
            "LZTR1: ALWAYS test both parents -- AD dominant-negative (50% risk) vs AR biallelic (25% risk) recurrence completely different",
            "LZTR1 Schwannomatosis-2 DDx: truncating LOF -> schwannomas (adult); missense dominant-negative -> Noonan (childhood); SAME gene different disease",
            "MEK inhibitors (trametinib, binimetinib, selumetinib): targeting RAS-RAF-MEK-ERK axis for all RASopathy genes -- clinical trials ongoing",
            "All 8 RASopathy genes = GERMLINE -- somatic_variant always False; all germline sequencing (not deep sequencing for somatic mosaics)",
        ],
        "gene_panel_note": (
            "RASopathy gene panel (2024): PTPN11, SOS1, RAF1, BRAF, MAP2K1, HRAS, KRAS, LZTR1; "
            "Extended panel: MAP2K2, NRAS, RIT1, RRAS, RRAS2, MRAS, CBL, SHOC2, SOS2 for comprehensive RASopathy; "
            "Testing strategy: "
            "  Classic Noonan (PS + short stature + facies): PTPN11 first -> SOS1 -> RAF1 -> KRAS -> LZTR1; "
            "  Noonan + HCM: RAF1 first (then PTPN11, RIT1, BRAF, MRAS); "
            "  CFC phenotype (ectodermal + severe ID): BRAF first -> MAP2K1 -> MAP2K2 -> KRAS; "
            "  Costello (loose skin + papillomata): HRAS p.Gly12Ser hotspot first; "
            "  LZTR1 Noonan: parental testing mandatory (AD vs AR distinction); "
            "Growth hormone: FDA approved for Noonan; cardiac assessment before initiation (HCM exclusion); "
            "MEK inhibitors: clinical trials enrolling for PTPN11/SOS1/RAF1/BRAF/MAP2K1/KRAS; "
            "Cancer surveillance: HRAS (rhabdomyosarcoma/bladder); KRAS (JMML/AML); PTPN11 (JMML); "
            "RAS-MAPK pathway: all 8 genes converge on ERK activation"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-RASopathy-Atlas."""
    genes_data = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        seed = SEED_BASE + idx
        patients = _generate_patients_for_gene(gene, seed)
        n = len(patients)

        epilepsy_n          = sum(1 for p in patients if p["epilepsy"])
        refractory_n        = sum(1 for p in patients if p["refractory_epilepsy"])
        speech_absent_n     = sum(1 for p in patients if p["speech_absent"])
        walk_n              = sum(1 for p in patients if p["independent_walk"])
        cardiac_hcm_n       = sum(1 for p in patients if p["cardiac_hcm"])
        ps_n                = sum(1 for p in patients if p["pulmonary_stenosis"])
        short_stature_n     = sum(1 for p in patients if p["short_stature"])
        webbed_neck_n       = sum(1 for p in patients if p["webbed_neck"])
        loose_skin_n        = sum(1 for p in patients if p["loose_skin"])
        papillomata_n       = sum(1 for p in patients if p["papillomata"])
        cancer_risk_n       = sum(1 for p in patients if p["cancer_risk_flag"])
        ectodermal_n        = sum(1 for p in patients if p["ectodermal_features"])
        autism_n            = sum(1 for p in patients if p["autism_features"])
        severe_n            = sum(1 for p in patients if p["severity"] == "severe")
        moderate_n          = sum(1 for p in patients if p["severity"] == "moderate")
        mild_n              = sum(1 for p in patients if p["severity"] == "mild")
        mean_iq             = round(sum(p["iq_estimate"] for p in patients) / n, 1)
        mean_age            = round(sum(p["age_at_diagnosis_mo"] for p in patients) / n, 1)
        mutations_seen      = list({p["mutation"] for p in patients})

        clinical_notes = {
            "PTPN11": "Most common Noonan gene (50-70%). PS 65% with dysplastic valve. HCM ~15%. JMML risk ~3%. LEOPARD overlap (PTP domain mutations).",
            "SOS1": "Normal IQ is the key differentiator. Ectodermal features (curly hair, keratosis pilaris). HCM rare -- GH therapy safest here. PS 35%.",
            "RAF1": "HCM 75% -- unique among all RASopathies. PTPN11-negative Noonan + HCM = test RAF1 first. GH contraindicated in obstructive HCM.",
            "BRAF": "CFC type 1 (~75% CFC). Ectodermal triad pathognomonic. Severe ID. Seizures 40-50%. Gastrostomy universal. BRAF V600E somatic NOT the same.",
            "MAP2K1": "CFC type 3 (~10-15% CFC). Severe ID. Direct MEK inhibitor target. Ectodermal features. Seizures 30-40%. Test after BRAF negative.",
            "HRAS": "Costello syndrome. Loose redundant skin + papillomata (face+perianal) pathognomonic. Cancer 15%: rhabdomyosarcoma + bladder. Surveillance MANDATORY.",
            "KRAS": "Most severe Noonan. High AML/JMML risk. p.Gly12Val most common germline. CFC overlap. Distinguish germline (RASopathy) from somatic (cancer).",
            "LZTR1": "Unique bidirectional: AD dominant-negative OR AR biallelic. TEST BOTH PARENTS -- 50% vs 25% recurrence risk completely different. Schwannomatosis-2 DDx.",
        }

        genes_data.append({
            "gene":                     gene,
            "locus":                    gene_info["locus"],
            "n":                        n,
            "n_patients":               n,
            "severe_pct":               round(severe_n / n * 100, 1),
            "moderate_pct":             round(moderate_n / n * 100, 1),
            "mild_pct":                 round(mild_n / n * 100, 1),
            "mean_iq":                  mean_iq,
            "epilepsy_pct":             round(epilepsy_n / n * 100, 1),
            "refractory_epilepsy_pct":  round(refractory_n / n * 100, 1),
            "speech_absent_pct":        round(speech_absent_n / n * 100, 1),
            "independent_walk_pct":     round(walk_n / n * 100, 1),
            "cardiac_hcm_pct":          round(cardiac_hcm_n / n * 100, 1),
            "ps_pct":                   round(ps_n / n * 100, 1),
            "short_stature_pct":        round(short_stature_n / n * 100, 1),
            "webbed_neck_pct":          round(webbed_neck_n / n * 100, 1),
            "loose_skin_pct":           round(loose_skin_n / n * 100, 1),
            "papillomata_pct":          round(papillomata_n / n * 100, 1),
            "cancer_risk_pct":          round(cancer_risk_n / n * 100, 1),
            "ectodermal_pct":           round(ectodermal_n / n * 100, 1),
            "autism_pct":               round(autism_n / n * 100, 1),
            "mean_age_dx_mo":           mean_age,
            "sample_mutations":         mutations_seen[:4],
            "protein":                  gene_info["protein"],
            "inheritance":              gene_info["inheritance"][:220],
            "disease_category":         gene_info["disease_category"],
            "clinical_note":            clinical_notes.get(gene, ""),
        })

    return {
        "atlas": "Hereditary-RASopathy-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-RASopathy-Atlas."""
    definitions = [
        {
            "term": "RASopathy Atlas: Classification, RAS-MAPK Pathway Architecture, and Diagnostic Framework",
            "genes": ["PTPN11", "SOS1", "RAF1", "BRAF", "MAP2K1", "HRAS", "KRAS", "LZTR1"],
            "definition": (
                "HEREDITARY RASopathy ATLAS -- OVERVIEW AND CLASSIFICATION: "
                "RAS-MAPK PATHWAY ARCHITECTURE (signal flow): "
                "  RECEPTOR TYROSINE KINASE (RTK) activation -> "
                "  GRB2/SOS1 (GEF) -> RAS GDP->GTP exchange -> RAS-GTP active; "
                "  HRAS/KRAS/NRAS -> RAF isoforms (BRAF, RAF1/CRAF, ARAF) -> "
                "  MEK1/2 (MAP2K1/MAP2K2) -> ERK1/2 -> nuclear transcription; "
                "  NEGATIVE REGULATORS: "
                "    SHP2 (PTPN11): activates RAS by dephosphorylating GAP-recruiting sites; "
                "    NF1 (neurofibromin): GAP for RAS -> RAS inactivation; "
                "    CBL/CBL-B: E3 ligase degrades RTKs; LZTR1 ubiquitinates RAS via CUL3; "
                "8-GENE CLASSIFICATION: "
                "  GROUP 1 -- SHP2/GEF LEVEL: "
                "    PTPN11 (SHP2 phosphatase GOF): dephosphorylates inhibitory sites -> RAS stays active; "
                "    SOS1 (RAS-GEF GOF): constitutive GDP->GTP exchange -> RAS always active; "
                "  GROUP 2 -- RAS LEVEL: "
                "    HRAS (GOF codon 12/13): impaired GTPase -> Costello + cancer; "
                "    KRAS (GOF codon 12/13): most severe Noonan + CFC + AML risk; "
                "  GROUP 3 -- RAF LEVEL: "
                "    RAF1/CRAF (GOF 14-3-3 binding): constitutive kinase -> HCM 75% unique; "
                "    BRAF (GOF kinase domain): CFC type 1 -> ectodermal + severe ID + seizures; "
                "  GROUP 4 -- MEK LEVEL: "
                "    MAP2K1/MEK1 (GOF autoinhibitory): constitutive ERK activation -> CFC type 3; "
                "  GROUP 5 -- RAS UBIQUITINATION: "
                "    LZTR1 (CUL3 adaptor LOF): reduced RAS ubiquitination -> elevated RAS; "
                "    UNIQUE: AD dominant-negative OR AR biallelic -- bidirectional inheritance; "
                "NOONAN vs CFC vs COSTELLO DIFFERENTIATION: "
                "  Noonan (PTPN11/SOS1/RAF1/KRAS/LZTR1): typical facies + PS + short stature; "
                "    IQ: normal (SOS1) to mild-moderate ID (PTPN11/RAF1); "
                "  CFC (BRAF/MAP2K1): Noonan-like facies + ECTODERMAL (curly hair/ichthyosis/KP) + severe ID; "
                "  Costello (HRAS): Noonan-like + LOOSE SKIN + PAPILLOMATA + cancer 15%; "
                "DIAGNOSTIC ALGORITHM: "
                "  STEP 1: Noonan facies + PS + no ectodermal: PTPN11 first -> SOS1 -> RAF1 -> KRAS -> LZTR1; "
                "  STEP 2: Noonan + HCM: RAF1 first (then PTPN11, RIT1, BRAF, MRAS); "
                "  STEP 3: Ectodermal features (curly hair/ichthyosis/KP) + severe ID: BRAF -> MAP2K1; "
                "  STEP 4: Loose skin + papillomata + cancer: HRAS (p.Gly12Ser hotspot); "
                "  STEP 5: LZTR1 found: TEST BOTH PARENTS (AD vs AR recurrence risk); "
                "MEK INHIBITOR CLINICAL TRIALS (ALL GENES): "
                "  Trametinib, binimetinib, selumetinib, cobimetinib -- targeting MEK1/2; "
                "  All 8 genes hyperactivate ERK via MEK; MEK inhibition = pathway-agnostic; "
                "  RASopathy trials: PTPN11/SOS1/RAF1/BRAF/MAP2K1/KRAS/HRAS -- enrolling; "
                "  Evidence: seizure reduction (BRAF/MAP2K1), HCM reduction (RAF1), ectodermal (BRAF)"
            ),
        },
        {
            "term": "Noonan Syndrome (PTPN11, SOS1, RAF1, KRAS, LZTR1) -- Cardiac Protocol and Growth Hormone Therapy",
            "genes": ["PTPN11", "SOS1", "RAF1", "KRAS", "LZTR1"],
            "definition": (
                "NOONAN SYNDROME -- CLINICAL PROTOCOL (PTPN11/SOS1/RAF1/KRAS/LZTR1): "
                "CARDIAC DISEASE -- GENE-SPECIFIC RATES: "
                "  PULMONARY STENOSIS (PS): "
                "    Most common cardiac defect overall; dysplastic valve (thickened, non-calcified); "
                "    PTPN11: 65%; SOS1: 35%; RAF1: 35%; KRAS: 50%; LZTR1: 50%; "
                "    DYSPLASTIC VALVE: balloon valvuloplasty LESS EFFECTIVE; surgical valvotomy preferred; "
                "  HYPERTROPHIC CARDIOMYOPATHY (HCM): "
                "    RAF1: 75% -- HIGHEST; obstructive common; PTPN11: 15%; KRAS: 45%; SOS1: 5%; LZTR1: 18%; "
                "    HCM management: beta-blockers first-line; disopyramide alternative; septal myectomy severe HOCM; "
                "    MEK inhibitors: reduce HCM in preclinical models (RAF1 target); "
                "  NEONATAL HCM: RAF1 + KRAS may be severe at birth; PICU level care; "
                "GROWTH HORMONE (GH) THERAPY -- FULL PROTOCOL: "
                "  FDA approved for Noonan syndrome; "
                "  DOSE: 0.066 mg/kg/day SC daily; titrate to IGF-1 SDS 0 to +2; "
                "  CONTRAINDICATIONS: "
                "    ABSOLUTE: active malignancy; "
                "    CAUTION: obstructive HCM (GH may worsen LVOT obstruction); "
                "    PROTOCOL: echocardiogram BEFORE GH initiation; RAF1-Noonan echo MANDATORY; "
                "    Monitor echo 6-monthly during GH if any HCM; "
                "  BENEFIT: 4-6 cm additional height gain; improved growth velocity; "
                "HAEMATOLOGICAL SURVEILLANCE: "
                "  PTPN11 JMML: 1-5% risk; CBCs infancy; haematology referral for cytopenias; "
                "  KRAS: HIGH AML/JMML risk -- CBCs every visit; haematology low threshold; "
                "DEVELOPMENTAL: "
                "  SOS1: NORMAL IQ -- standard schooling appropriate; "
                "  PTPN11: mild-moderate ID -- learning support; IEP; "
                "  RAF1/KRAS: mild-moderate ID (KRAS more severe); "
                "  LZTR1 het: mild ID; biallelic more severe; "
                "  All: speech therapy, physiotherapy, occupational therapy; "
                "CRYPTORCHIDISM (males): 60-70%; orchidopexy <18 months; "
                "LYMPHATICS: lymphoedema (lower limb), chylothorax (neonatal); "
                "OPHTHALMOLOGY: amblyopia, strabismus screening at diagnosis; "
                "LZTR1-SPECIFIC: biallelic more severe than het; parental testing MANDATORY; "
                "GH IN KRAS: extreme caution -- HCM severe; AML risk monitoring first"
            ),
        },
        {
            "term": "CFC Syndrome (BRAF, MAP2K1) and Costello Syndrome (HRAS) -- Ectodermal Features and Cancer Surveillance",
            "genes": ["BRAF", "MAP2K1", "HRAS"],
            "definition": (
                "CFC AND COSTELLO SYNDROMES -- CLINICAL PROTOCOL (BRAF/MAP2K1/HRAS): "
                "CARDIO-FACIO-CUTANEOUS (CFC) SYNDROME -- BRAF AND MAP2K1: "
                "  ECTODERMAL PATHOGNOMONIC TRIAD (distinguishes CFC from Noonan): "
                "    (1) CURLY/SPARSE/ABSENT HAIR (friable, absent eyebrows); "
                "    (2) ICHTHYOSIS: dry scaly skin; keratinisation defect; "
                "    (3) KERATOSIS PILARIS: follicular keratosis arms/thighs/face; "
                "  These three together: PATHOGNOMONIC for CFC; not present in classic Noonan; "
                "  NEUROLOGICAL: SEVERE to PROFOUND ID; absent speech majority; "
                "    SEIZURES: BRAF 40-50%; MAP2K1 30-40%; West syndrome/LGS; drug-resistant; "
                "    Hypotonia universal; structural brain anomalies; "
                "  FEEDING: gastrostomy in majority (BRAF > MAP2K1); reflux; aspiration; "
                "  CARDIAC: BRAF HCM 30-40% + PS 55%; MAP2K1 HCM 28% + PS 45%; "
                "  BRAF V600E IMPORTANT NOTE: somatic oncogenic variant in melanoma -- NOT same as CFC germline BRAF; "
                "    Vemurafenib (BRAF V600E inhibitor): NOT appropriate for CFC; "
                "    MEK inhibitors (trametinib): appropriate for CFC (targets downstream); "
                "TRAMETINIB IN CFC: "
                "  Compassionate use + clinical trials; seizure reduction; ectodermal improvement; "
                "  Cardiac hypertrophy reduction; trials enrolling; "
                "COSTELLO SYNDROME -- HRAS: "
                "  PATHOGNOMONIC FEATURES: "
                "    LOOSE REDUNDANT SKIN: deep palmar/plantar creases; loose neck skin; not in Noonan/CFC; "
                "    PAPILLOMATA: perinasal + perianal distribution PATHOGNOMONIC; childhood onset; benign; "
                "  p.Gly12Ser (~85% Costello): test first; "
                "CANCER SURVEILLANCE -- HRAS (MANDATORY): "
                "  LIFETIME CANCER RISK: 15%; "
                "  RHABDOMYOSARCOMA: embryonal; most common; childhood; "
                "    SURVEILLANCE: abdominal + pelvic USS every 6-12 months TO AGE 8; "
                "  BLADDER TRANSITIONAL CELL CARCINOMA: late childhood/adulthood; "
                "    SURVEILLANCE: urinalysis (dipstick) every 6 months FROM AGE 10; "
                "  NEUROBLASTOMA: AFP surveillance infancy; "
                "  p.Lys117Arg: HIGHEST cancer risk; more frequent monitoring; "
                "  p.Gly12Ala: LOWEST cancer risk among common Costello mutations; "
                "CARDIAC IN COSTELLO: "
                "  MULTIFOCAL ATRIAL TACHYCARDIA (MAT): characteristic; Holter monitoring essential; "
                "  HCM 30-50%; PS; echo mandatory; "
                "TIPIFARNIB/LONAFARNIB (HRAS-specific): "
                "  HRAS requires farnesylation (CAAX) for membrane anchoring; "
                "  Farnesyl transferase inhibitors block HRAS -- HRAS uniquely sensitive (not KRAS); "
                "  Clinical trials ongoing for Costello"
            ),
        },
        {
            "term": "LZTR1-Noonan10: Unique Bidirectional Inheritance -- Dominant vs Recessive Algorithm and LZTR1 Schwannomatosis DDx",
            "genes": ["LZTR1"],
            "definition": (
                "LZTR1-NOONAN10 -- BIDIRECTIONAL INHERITANCE AND SCHWANNOMATOSIS DDx PROTOCOL: "
                "WHY LZTR1 IS UNIQUE AMONG ALL RASopathy GENES: "
                "  ALL other Noonan/CFC/Costello genes: ONLY autosomal dominant; "
                "  LZTR1: BOTH autosomal dominant AND autosomal recessive; "
                "  SAME CLINICAL PRESENTATION but COMPLETELY DIFFERENT RECURRENCE RISKS; "
                "MECHANISM -- WHY TWO INHERITANCE MODES: "
                "  DOMINANT MODE (HETEROZYGOUS): "
                "    Mechanism: DOMINANT NEGATIVE (not haploinsufficiency); "
                "    Mutant LZTR1 poisons the CUL3 complex (one bad subunit = dysfunctional complex); "
                "    MILDER phenotype; 50% recurrence risk; "
                "  RECESSIVE MODE (BIALLELIC): "
                "    Mechanism: true LOF both alleles (compound het or homozygous); "
                "    MORE SEVERE phenotype; 25% recurrence risk; "
                "    Parents typically heterozygous carriers, clinically NORMAL; "
                "PARENTAL TESTING ALGORITHM (MANDATORY): "
                "  STEP 1: Child has LZTR1 variant confirmed; "
                "  STEP 2: Test BOTH biological parents: "
                "    RESULT A: One parent has SAME variant AND clinically AFFECTED with Noonan: "
                "      -> AUTOSOMAL DOMINANT; 50% recurrence risk; "
                "    RESULT B: Both parents het (same or different variants) AND clinically UNAFFECTED: "
                "      -> AUTOSOMAL RECESSIVE; 25% recurrence risk; "
                "    RESULT C: Neither parent has variant: de novo dominant; <1% recurrence; "
                "  STEP 3: Adjust counselling per inheritance mode; "
                "SCHWANNOMATOSIS-2 vs NOONAN-10 -- SAME GENE: "
                "  SCHWANNOMATOSIS-2 (truncating/frameshift LOF): "
                "    Multiple schwannomas (peripheral nerve sheath tumours); "
                "    Adult onset (20s-50s); pain; sensorimotor deficits; "
                "    NO vestibular schwannomas (cf NF2); "
                "  NOONAN-10 (missense GOF/dominant-negative): "
                "    Childhood Noonan features (cardiac, ID, facies, short stature); "
                "  KEY DDx: "
                "    Child + Noonan features + family schwannomas -> LZTR1; "
                "    Adult schwannomas + family Noonan history -> LZTR1; "
                "22q11.21 LOCUS: "
                "  Adjacent to DiGeorge region (22q11.2 deletion); "
                "  Large 22q11.2 deletions may include LZTR1 and TBX1 simultaneously; "
                "  CMA 22q11.2 deletion: check whether LZTR1 included in deleted interval; "
                "TREATMENT: "
                "  Cardiac: standard per lesion; echo annually; "
                "  Growth hormone: may be used; cardiac assessment first; "
                "  MEK inhibitors: clinical trials include LZTR1-Noonan; "
                "  Schwannomatosis surveillance (if LOF variant): MRI spine every 3 years; "
                "  Pain management: analgesics; surgical excision if symptomatic"
            ),
        },
    ]
    return {
        "atlas": "Hereditary-RASopathy-Atlas",
        "count": len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2)[:500])
