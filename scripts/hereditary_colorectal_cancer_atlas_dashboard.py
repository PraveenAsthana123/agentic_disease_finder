#!/usr/bin/env python3
"""Hereditary-Colorectal-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
POLE    (PolEpsilon catalytic subunit; 2286aa; 12q24.33; AD GOF exonuclease domain;
         PPAP-1 Polymerase-Proofreading-Associated-Polyposis; CRC + endometrial + duodenal;
         ultra-hypermutated TMB>100 MSS-mimic; exceptional pembrolizumab complete remissions;
         POLE-PROOFREADING-MUTATION NOT MSI; L424V/P286R/V411L/S459F hotspots;
         COSMIC signature SBS10a/10b;
         seed SEED_BASE+0) .
POLD1   (PolDelta catalytic subunit; 1107aa; 19q13.33; AD GOF exonuclease domain;
         PPAP-2; CRC 80% lifetime; sebaceous gland tumours PATHOGNOMONIC for POLD1 vs POLE;
         brain tumours; L474P/D316H/E318K hotspots;
         ceralasertib+pembrolizumab;
         seed SEED_BASE+1) .
EPCAM   (Epithelial Cell Adhesion Molecule; 314aa; 2p21; AD LOF 3-prime deletion only;
         Lynch-by-MSH2-silencing; EPCAM 3-prime deletion -> MSH2 promoter methylation epigenetic silencing;
         MSH2 IHC loss with MSH2 coding normal; MLPA MANDATORY not Sanger;
         small bowel PATHOGNOMONIC - highest small bowel Lynch cancer; CRC 50-80%;
         seed SEED_BASE+2) .
NTHL1   (Nth Like DNA Glycosylase 1; 312aa; 16p13.3; AR biallelic LOF;
         NAP NTHL1-Associated-Polyposis; base excision repair; CpG>TpG COSMIC SBS30;
         20-200 colonic adenomas; CRC near-100% biallelic lifetime; breast cancer elevated;
         founder mutation c.268C>T Q90X; pembrolizumab MSI-H;
         seed SEED_BASE+3) .
RNF43   (Ring Finger Protein 43; 783aa; 17q22; AD LOF / biallelic LOF;
         SFPN Serrated-Familial-Polyposis-Neoplasia; sessile serrated lesions PATHOGNOMONIC;
         Wnt negative regulator E3-ubiquitin-ligase; RSPO fusion PATHOGNOMONIC for RNF43 in sporadic;
         BRAF-V600E present in RNF43 serrated pathway; porcupine inhibitor LGK-974 preclinical;
         MSI-H subset pembrolizumab; RNF43 G659Vfs*41 most common germline truncation;
         seed SEED_BASE+4) .
BMPR1A  (Bone Morphogenetic Protein Receptor 1A; 532aa; 10q22.3; AD LOF;
         JPS Juvenile-Polyposis-Syndrome type 1; juvenile polyps PATHOGNOMONIC;
         CRC 39% lifetime; gastric cancer 21% lifetime; HHT-overlap ABSENT in BMPR1A;
         BMPR1A haploinsufficiency large deletions 10q22-23 MLPA mandatory;
         PTEN-10q contiguous deletion;
         seed SEED_BASE+5) .
MSH3    (MutS Homolog 3; 1128aa; 5q11.2; AR biallelic LOF;
         PPAP-like biallelic-MSH3; polyposis 20-500 adenomas;
         CRC near 100% biallelic; CMMRD-risk LOWER than MSH2/MSH6/MLH1 biallelic;
         MSI-H + EMAST PATHOGNOMONIC; pembrolizumab MSI-H; MutSbeta dimer;
         seed SEED_BASE+6) .
GREM1   (Gremlin 1 BMP antagonist; 184aa; 15q13.3; AD GOF 3-prime duplication ~40kb;
         HMPS Hereditary-Mixed-Polyposis-Syndrome; mixed polyps PATHOGNOMONIC;
         exclusively Ashkenazi-Jewish founder duplication; GREM1 upregulation -> BMP-inhibition;
         CRC 50-80% lifetime; small bowel adenomas; endoscopy annual from age 25yr;
         germline analysis requires MLPA/CNV not Sanger;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3230-3237)
"""
import random

SEED_BASE = 3230

ATLAS_GENES = [
    {
        "gene": "POLE",
        "protein": (
            "POLE -- 12q24.33 Autosomal-Dominant-GOF -- 2286aa -- "
            "PolEpsilon-261kDa-Exonuclease-Domain-GOF-PPAP-1-"
            "Ultra-Hypermutated-TMB-GT100-MSS-Mimic-Exceptional-Pembrolizumab-"
            "CRC-Endometrial-Duodenal-SBS10a-SBS10b-OMIM-174762"
        ),
        "locus": "12q24.33",
        "protein_size": (
            "2286 aa / 261 kDa / 12q24.33 POLE encodes DNA Polymerase Epsilon catalytic subunit: "
            "STRUCTURE: "
            "  2286 aa / 261 kDa; catalytic subunit of DNA Polymerase Epsilon (Pol-epsilon); "
            "  N-terminal polymerase domain (aa 1-~900): DNA synthesis; "
            "  Exonuclease (proofreading) domain (aa ~268-471): 3'-5' proofreading activity; "
            "  C-terminal domain (CTD, aa ~1000-2286): scaffolding for POLE2/3/4 accessory subunits; "
            "  Pol-epsilon functions: leading-strand DNA replication; "
            "  Exonuclease domain hotspot mutations: L424V, P286R, V411L, S459F; "
            "  GOF = gain-of-dysfunction of proofreading -> error catastrophe -> hypermutation; "
            "  POLE exonuclease LOF -> failure to excise misincorporated bases -> C>A/C>T ultra-mutation; "
            "  COSMIC signatures SBS10a (POLE) and SBS10b (POLE): C>A and C>T hypermutation pattern; "
            "PPAP-1 -- POLYMERASE PROOFREADING ASSOCIATED POLYPOSIS: "
            "  OMIM 174762; AD GOF (exonuclease domain); PPAP type 1 = POLE; "
            "  10-100 colonic adenomas (fewer polyps than APC-FAP but malignant potential high); "
            "  Right-colon predilection of CRC; "
            "  Age of CRC presentation: typically 40-60yr (later than APC-FAP; earlier than sporadic); "
            "  Duodenal adenomas: elevated; extra-colonic: endometrial 30-40%; ovarian elevated; "
            "ULTRA-HYPERMUTATION (TMB >100 mut/Mb): "
            "  POLE-proofreading mutations: TMB >100 mut/Mb = ultra-hypermutated = HIGHEST TMB any hereditary; "
            "  CRITICAL DISTINCTION: ultra-hypermutated POLE = MSS (microsatellite stable) by standard PCR/NGS; "
            "  MMR IHC: intact (MLH1/MSH2/MSH6/PMS2 all retained) -- NOT dMMR/MSI-H by IHC; "
            "  Standard MSI testing WILL MISS POLE -> POLE sequencing mandatory for ultra-TMB CRC; "
            "  SBS10a + SBS10b signatures: pathognomonic in tumour COSMIC mutational analysis; "
            "PEMBROLIZUMAB -- EXCEPTIONAL RESPONSE: "
            "  POLE ultra-hypermutated CRC: exceptional pembrolizumab (anti-PD-1) responses; "
            "  Complete remissions documented in POLE-exonuclease CRC/endometrial (The Cancer Genome Atlas); "
            "  Mechanism: ultra-hypermutation -> extreme neoantigen burden -> T-cell recognition; "
            "  POLE CRC/endometrial: pembrolizumab first-line preferred over standard chemotherapy; "
            "  EXCEPTION: POLE pathogenic variant must be exonuclease domain GOF (NOT polymerase domain VUS); "
            "HOTSPOT MUTATIONS (EXONUCLEASE DOMAIN): "
            "  L424V: most common PPAP-1 hotspot; 12q24.33 exon 9; "
            "  P286R: associated with highest TMB; most potent proofreading abrogation; "
            "  V411L: high TMB; endometrial predominant among POLE hotspots; "
            "  S459F: lower penetrance PPAP-1 hotspot; functional assessment required; "
            "CANCER SPECTRUM (POLE-PPAP-1): "
            "  CRC: 60-80% lifetime; endometrial: 30-40%; duodenal/small bowel: elevated; "
            "  Ovarian (clear cell/endometrioid): elevated; brain tumour rare; "
            "SURVEILLANCE (POLE PPAP-1): "
            "  Colonoscopy: 1-2yr from age 25yr (or 10yr before earliest family CRC); "
            "  Upper GI endoscopy from age 30yr (duodenal adenomas); "
            "  Annual endometrial USS from age 30yr (female POLE carriers); "
            "  POLE germline testing: all MSS CRC with TMB >100 mut/Mb by NGS; "
            "TREATMENT (POLE-MUTANT CANCER): "
            "  Pembrolizumab: first-line POLE ultra-hypermutated CRC/endometrial; "
            "  Avoidance of standard 5-FU/FOLFOX chemotherapy in confirmed POLE ultra-hypermutated: "
            "  controversial -- pembrolizumab monotherapy ongoing trials (KEYNOTE-158 sub-cohort)"
        ),
        "inheritance": "Autosomal Dominant (AD); GOF exonuclease domain; OMIM 174762; PPAP-1 Polymerase-Proofreading-Associated-Polyposis; de novo ~10-20%; family cascade mandatory; hotspot mutations L424V/P286R/V411L/S459F",
        "cancer_risk": "CRC: 60-80% lifetime; endometrial: 30-40%; duodenal elevated; TMB >100 MSS-mimic POLE-specific; exceptional pembrolizumab complete remissions; POLE sequencing mandatory for ultra-TMB MSS CRC (MSI testing misses POLE)",
        "pathognomonic": "Ultra-hypermutated TMB >100 mut/Mb with MSS (NOT MSI-H by IHC) = PATHOGNOMONIC POLE exonuclease GOF; COSMIC SBS10a/SBS10b signatures PATHOGNOMONIC; exceptional pembrolizumab complete remissions; L424V/P286R hotspots",
        "surveillance_key": "Colonoscopy 1-2yr from age 25yr; upper GI endoscopy from 30yr; endometrial USS from 30yr (females); POLE sequencing all ultra-TMB MSS CRC; pembrolizumab first-line POLE CRC/endometrial; TMB >100 + MSS -> POLE sequencing MANDATORY",
        "key_distinctions": [
            "ULTRA-HYPERMUTATED-TMB-GT100-MSS-NOT-MSI-PCR",
            "POLE-PROOFREADING-GOF-NOT-MMR-DEFICIENT",
            "EXCEPTIONAL-PEMBROLIZUMAB-COMPLETE-REMISSIONS",
            "SBS10A-SBS10B-PATHOGNOMONIC-COSMIC",
            "L424V-P286R-HOTSPOTS-EXONUCLEASE-DOMAIN",
            "COLONOSCOPY-1-2YR-FROM-25YR",
        ],
    },
    {
        "gene": "POLD1",
        "protein": (
            "POLD1 -- 19q13.33 Autosomal-Dominant-GOF -- 1107aa -- "
            "PolDelta-125kDa-Exonuclease-Domain-GOF-PPAP-2-CRC-80pct-"
            "Sebaceous-Tumours-PATHOGNOMONIC-Brain-Tumours-L474P-D316H-E318K-"
            "Ceralasertib-Pembrolizumab-OMIM-174761"
        ),
        "locus": "19q13.33",
        "protein_size": (
            "1107 aa / 125 kDa / 19q13.33 POLD1 encodes DNA Polymerase Delta 1 (catalytic/proofreading): "
            "STRUCTURE: "
            "  1107 aa / 125 kDa; catalytic and proofreading subunit of DNA Polymerase Delta (Pol-delta); "
            "  Polymerase domain (aa 1-~700): DNA synthesis (lagging-strand primary); "
            "  Exonuclease (proofreading) domain (aa ~304-525): 3'-5' proofreading; "
            "  C-terminal PCNA-interaction motif (PIP box): PCNA loading; "
            "  Pol-delta functions: lagging-strand DNA replication; mismatch repair synthesis; "
            "  Exonuclease domain hotspot mutations: L474P, D316H, E318K; "
            "  GOF = exonuclease dysfunction -> hypermutation; "
            "  POLD1 hypermutation pattern: distinct from POLE (SBS10a/10b) -- overlapping but lower TMB; "
            "  POLD1 TMB: typically 50-100 mut/Mb (lower than POLE ultra-hypermutated); "
            "PPAP-2 -- POLYMERASE PROOFREADING ASSOCIATED POLYPOSIS TYPE 2: "
            "  OMIM 174761; AD GOF (exonuclease domain); PPAP type 2 = POLD1; "
            "  10-100 colonic adenomas (similar polyp count to POLE); "
            "  CRC: 80% lifetime = VERY HIGH CRC PENETRANCE; "
            "  Age of CRC presentation: 30-60yr (earlier than sporadic, overlapping POLE); "
            "  Sebaceous gland tumours: PATHOGNOMONIC feature distinguishing POLD1 from POLE; "
            "SEBACEOUS TUMOURS -- PATHOGNOMONIC FOR POLD1 vs POLE: "
            "  Sebaceous adenoma + sebaceous carcinoma: present in POLD1 (NOT characteristic of POLE); "
            "  CRITICAL DISTINCTION: sebaceous tumours in PPAP -> test POLD1 specifically (not only MMR); "
            "  POLD1 sebaceous tumours: IHC may show variable MMR expression -> confirm germline POLD1; "
            "  Any sebaceous neoplasm + colonic polyposis -> POLD1 sequencing MANDATORY; "
            "BRAIN TUMOURS -- POLD1: "
            "  Glioma/astrocytoma: elevated in POLD1 carriers vs POLE; "
            "  Paediatric brain tumour: possible in POLD1 (unlike POLE which is predominantly adult); "
            "  Constitutional mismatch repair deficiency (CMMRD)-like features in some POLD1 biallelic; "
            "  Brain MRI surveillance from age 18yr in POLD1 carriers; "
            "L474P / D316H / E318K HOTSPOTS: "
            "  L474P: most common POLD1 pathogenic variant; highest penetrance PPAP-2; "
            "  D316H: associated with highest POLD1 TMB; "
            "  E318K: lower penetrance; functional evidence required; "
            "  In silico tools: pathogenic POLE/POLD1 exonuclease variants need wetlab functional confirmation; "
            "CANCER SPECTRUM (POLD1-PPAP-2): "
            "  CRC: 80% lifetime = HIGHEST POLD1 risk; endometrial: 30-40% (females); "
            "  Ovarian: elevated; brain tumours: elevated; sebaceous neoplasms extra-intestinal; "
            "CERALASERTIB + PEMBROLIZUMAB (POLD1): "
            "  POLD1-hypermutated cancers: pembrolizumab (MSI-H/TMB-high FDA-approved pan-tumour); "
            "  Ceralasertib (ATR inhibitor) + pembrolizumab: clinical trials hypermutated cancers; "
            "  POLD1-deficient cells: replication stress -> ATR dependency -> ceralasertib synthetic lethal; "
            "SURVEILLANCE (POLD1 PPAP-2): "
            "  Colonoscopy: 1-2yr from age 20yr (earlier onset than POLE); "
            "  Upper GI endoscopy from age 30yr; "
            "  Annual endometrial surveillance from age 25yr (females); "
            "  Brain MRI: annually from age 18yr (brain tumour elevated); "
            "  Dermatology: annual (sebaceous tumour surveillance); "
            "  POLD1 germline testing: all PPAP phenotype + sebaceous tumour"
        ),
        "inheritance": "Autosomal Dominant (AD); GOF exonuclease domain; OMIM 174761; PPAP-2; de novo ~10-20%; sebaceous tumours PATHOGNOMONIC POLD1 vs POLE; L474P/D316H/E318K hotspots; family cascade mandatory",
        "cancer_risk": "CRC: 80% lifetime HIGHEST POLD1; endometrial: 30-40%; brain tumours elevated; sebaceous gland tumours PATHOGNOMONIC; TMB 50-100 mut/Mb; pembrolizumab + ceralasertib; sebaceous neoplasm + polyposis -> POLD1 sequencing MANDATORY",
        "pathognomonic": "Sebaceous gland tumours PATHOGNOMONIC for POLD1 vs POLE (POLE does not carry sebaceous); brain tumours elevated (unlike POLE); CRC 80% HIGHEST PPAP-2; L474P hotspot; any sebaceous neoplasm + polyposis -> POLD1 mandatory",
        "surveillance_key": "Colonoscopy 1-2yr from age 20yr; brain MRI annually from 18yr; annual dermatology (sebaceous); annual endometrial USS from 25yr (females); pembrolizumab TMB-high; ceralasertib + pembrolizumab ATR-POLD1 trials; POLD1 germline: sebaceous neoplasm + polyposis",
        "key_distinctions": [
            "SEBACEOUS-TUMOURS-PATHOGNOMONIC-POLD1-NOT-POLE",
            "CRC-80PCT-HIGHEST-PPAP-2",
            "BRAIN-TUMOURS-ELEVATED-POLD1",
            "L474P-D316H-E318K-EXONUCLEASE-HOTSPOTS",
            "CERALASERTIB-PEMBROLIZUMAB-POLD1-TRIALS",
            "COLONOSCOPY-1-2YR-FROM-20YR",
        ],
    },
    {
        "gene": "EPCAM",
        "protein": (
            "EPCAM -- 2p21 Autosomal-Dominant-LOF-3prime-Deletion-Only -- 314aa -- "
            "EpCAM-35kDa-Epithelial-Cell-Adhesion-Lynch-by-MSH2-Silencing-"
            "EPCAM-3prime-Deletion-MSH2-Promoter-Methylation-MLPA-MANDATORY-"
            "Small-Bowel-PATHOGNOMONIC-CRC-50-80pct-OMIM-185535"
        ),
        "locus": "2p21",
        "protein_size": (
            "314 aa / 35 kDa / 2p21 EPCAM encodes Epithelial Cell Adhesion Molecule: "
            "STRUCTURE: "
            "  314 aa / 35 kDa; type I transmembrane glycoprotein; "
            "  N-terminal signal peptide (aa 1-23): ER targeting; "
            "  Extracellular domain (ECD, aa 24-265): thyroglobulin type 1 repeat domains; "
            "  TM domain (aa 266-288): single-pass transmembrane; "
            "  Short intracellular domain (ICD, aa 289-314): gamma-secretase cleavage site; "
            "  EPCAM: epithelial cell-cell adhesion + proliferation signalling; "
            "  EPCAM cleaved ICD: enters nucleus -> activates Wnt/Cyclin D1 proliferation; "
            "  EPCAM gene locus: adjacent and 3'-upstream of MSH2 (2p21); "
            "  3'-EPCAM deletion: read-through transcription -> EPCAM-MSH2 fusion transcript; "
            "  Fusion transcript -> aberrant MSH2 promoter CpG methylation -> epigenetic silencing; "
            "EPCAM 3-PRIME DELETION -- UNIQUE MECHANISM: "
            "  EPCAM LOF in Lynch: ONLY via 3'-end deletion (not coding missense/nonsense); "
            "  3' deletions span: last 3-4 exons of EPCAM (exons 7-9 most commonly deleted); "
            "  Mechanism: 3' deletion -> loss of EPCAM transcription terminator; "
            "  Read-through transcript enters MSH2 -> EPCAM-MSH2 read-through RNA -> MSH2 allele silenced; "
            "  Result: MSH2 protein absent; MSH6 protein absent (MSH6 destabilises without MSH2 partner); "
            "  IHC: MSH2 LOST + MSH6 LOST; EPCAM protein IHC: NORMAL (promoter still active in tumour); "
            "  Standard germline sequencing (NGS/Sanger): MISSES EPCAM 3' deletion (point mutation panel); "
            "MLPA MANDATORY (NOT SANGER): "
            "  MLPA (Multiplex Ligation-dependent Probe Amplification): detects copy-number changes; "
            "  MLPA 3'-EPCAM-specific probe set: detects del exon 7/8/9 (diagnostic); "
            "  Sanger sequencing: INADEQUATE for EPCAM deletion (does not detect deletion); "
            "  Rule: Lynch suspected + MSH2 IHC lost + MSH2 coding normal -> EPCAM MLPA MANDATORY; "
            "SMALL BOWEL CANCER -- PATHOGNOMONIC HIGHEST LYNCH: "
            "  Small bowel adenocarcinoma: EPCAM-Lynch has HIGHEST small bowel cancer risk; "
            "  Small bowel: ~10-17% lifetime in EPCAM Lynch (vs ~3-5% in MSH2 other mechanisms); "
            "  EPCAM epigenetic silencing: affects intestinal epithelium most severely (small bowel); "
            "  Small bowel MRI/CT enterography annually from age 30yr; "
            "  Video capsule endoscopy: in EPCAM Lynch small bowel surveillance; "
            "LYNCH CANCER SPECTRUM (EPCAM): "
            "  CRC: 50-80% lifetime (Lynch2-equivalent MSH2-silenced); "
            "  Endometrial: 40-60% (females) = dominant Lynch2; "
            "  Small bowel: 10-17% = HIGHEST small bowel risk Lynch; "
            "  Urinary tract: 25-28%; ovarian: 10-12%; gastric: 8-12%; "
            "PEMBROLIZUMAB (MSI-H): "
            "  EPCAM-Lynch CRC: dMMR MSI-H -> pembrolizumab FDA-approved all histologies; "
            "  Aspirin 600mg/day: 50% CRC risk reduction = LEVEL A evidence Lynch; "
            "SURVEILLANCE (EPCAM LYNCH): "
            "  Colonoscopy 1-2yr from age 25yr; "
            "  Small bowel surveillance from age 30yr (capsule endoscopy / MRI enterography); "
            "  Annual gynaecological surveillance from age 30yr; "
            "  Urinary tract cystoscopy every 2yr from age 25yr; "
            "  EPCAM MLPA: all patients with MSH2+MSH6 lost IHC and MSH2 coding normal"
        ),
        "inheritance": "Autosomal Dominant (AD); LOF by 3'-end deletion ONLY (not coding point mutation); OMIM 185535; Lynch-by-MSH2-epigenetic-silencing; MLPA mandatory (not Sanger); MSH2+MSH6 IHC lost; EPCAM IHC normal; family cascade mandatory",
        "cancer_risk": "CRC: 50-80% lifetime; endometrial: 40-60% females; small bowel: 10-17% HIGHEST Lynch small bowel; urinary tract: 25-28%; dMMR MSI-H -> pembrolizumab; Aspirin CAPP2 50% CRC reduction LEVEL A",
        "pathognomonic": "EPCAM 3'-deletion silences MSH2 via promoter methylation (unique epigenetic mechanism); MSH2+MSH6 IHC lost with MSH2 coding normal = PATHOGNOMONIC EPCAM deletion; MLPA 3'-specific MANDATORY; small bowel cancer = HIGHEST Lynch small bowel risk in EPCAM",
        "surveillance_key": "Colonoscopy 1-2yr from 25yr; small bowel capsule endoscopy/MRI from 30yr; annual gynaecological from 30yr; EPCAM MLPA mandatory if MSH2 IHC lost + coding normal; pembrolizumab dMMR; aspirin CAPP2 LEVEL A; urinary cystoscopy 2yr from 25yr",
        "key_distinctions": [
            "EPCAM-3PRIME-DELETION-MSH2-PROMOTER-METHYLATION",
            "MLPA-MANDATORY-NOT-SANGER-EPCAM",
            "SMALL-BOWEL-10-17PCT-HIGHEST-LYNCH-EPCAM",
            "MSH2-MSH6-IHC-LOST-EPCAM-IHC-NORMAL",
            "CRC-50-80PCT-LYNCH-MSH2-SILENCED",
            "PEMBROLIZUMAB-MSI-H-ALL-HISTOLOGIES",
        ],
    },
    {
        "gene": "NTHL1",
        "protein": (
            "NTHL1 -- 16p13.3 Autosomal-Recessive-biallelic-LOF -- 312aa -- "
            "NTHL1-35kDa-BER-Bifunctional-Glycosylase-NAP-"
            "NAP-NTHL1-Associated-Polyposis-CpG-TpG-SBS30-20-200-Adenomas-"
            "CRC-Near-100pct-Biallelic-Breast-Elevated-Q90X-Founder-OMIM-190195"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "312 aa / 35 kDa / 16p13.3 NTHL1 encodes Nth Like DNA Glycosylase 1 (base excision repair): "
            "STRUCTURE: "
            "  312 aa / 35 kDa; bifunctional DNA glycosylase; "
            "  HhH-GPD superfamily (helix-hairpin-helix glycosylase-phosphodiesterase); "
            "  Glycosylase activity: removes oxidised pyrimidines (thymine glycol, cytosine hydrates); "
            "  Lyase activity: AP-site cleavage (beta-elimination); "
            "  Fe-S (4Fe-4S) cluster (aa ~232-241): DNA binding and charge transfer; "
            "  NTHL1 substrates: oxidised pyrimidines generated by ROS (reactive oxygen species); "
            "  BER pathway: NTHL1 -> APE1 -> Pol-beta -> XRCC1/Ligase3 repair axis; "
            "  NTHL1 LOF -> unrepaired thymine glycol / cytosine hydrate -> CpG>TpG (C:G>T:A) mutations; "
            "  COSMIC SBS30: CpG>TpG mutational signature = NTHL1-deficient PATHOGNOMONIC signature; "
            "NAP -- NTHL1-ASSOCIATED POLYPOSIS: "
            "  OMIM 190195; AR biallelic LOF required for polyposis/cancer phenotype; "
            "  20-200 colonic adenomas (broad range: oligopolyposis to dense polyposis); "
            "  CRC: near-100% lifetime penetrance in biallelic NTHL1 (untreated); "
            "  Age of CRC onset: typically 40-60yr; "
            "  Duodenal adenomas: elevated; endometrial adenoma elevated; "
            "  NTHL1 monoallelic heterozygotes: no proven elevated polyposis/CRC risk (pure AR); "
            "BREAST CANCER (NTHL1): "
            "  Breast cancer: elevated in biallelic NTHL1 (RR ~2-5x vs general population); "
            "  NTHL1 biallelic: BER deficiency -> BRCA-ness? -- uncertain; "
            "  Annual breast surveillance from age 30yr in biallelic NTHL1 females; "
            "  Breast cancer not a PPAP hallmark (distinction from POLE/POLD1 which are CRC/endometrial); "
            "COSMIC SBS30 -- PATHOGNOMONIC: "
            "  SBS30: CpG>TpG at specific trinucleotide contexts = NTHL1-deficiency PATHOGNOMONIC; "
            "  Tumour sequencing: SBS30 signature > 20% contribution = NTHL1 LOF likely; "
            "  SBS30 differs from SBS10a/10b (POLE) -- important panel distinction; "
            "FOUNDER MUTATION c.268C>T (Q90X): "
            "  Q90X (c.268C>T): most common pathogenic NTHL1 variant; "
            "  European population founder mutation: Netherlands/Northern European enriched; "
            "  Q90X: introduces premature stop codon in exon 5 -> NMD -> null allele; "
            "  Q90X homozygous: biallelic = full NAP phenotype; "
            "PEMBROLIZUMAB (MSI-H SUBSET): "
            "  NTHL1-biallelic CRC: subset show MSI-H/dMMR (secondary MMR loss in tumour); "
            "  MSI-H NTHL1-NAP CRC: pembrolizumab eligible (FDA-approved dMMR pan-tumour); "
            "  NTHL1 tumours without MSI: TMB-intermediate; pembrolizumab benefit uncertain; "
            "CMMRD-NOT-EXPECTED (unlike MSH2/MSH6/MLH1 biallelic): "
            "  NTHL1 biallelic: does NOT cause childhood constitutional MMR deficiency (CMMRD); "
            "  CMMRD: only from biallelic MMR gene mutations (MSH2/MSH6/MLH1/PMS2); "
            "  NTHL1 biallelic children: no CMMRD brain tumour / haematological risk at childhood onset; "
            "  Adult-onset CRC/adenomas = NAP phenotype (not CMMRD); "
            "SURVEILLANCE (NTHL1-NAP): "
            "  Colonoscopy: annually from age 20yr (dense polyposis management); "
            "  Upper GI endoscopy from age 30yr; "
            "  Annual breast surveillance from age 30yr (females, biallelic); "
            "  Endometrial USS annually from age 35yr (biallelic females); "
            "  Germline testing: Q90X hot-spot + full NTHL1 sequencing + MLPA"
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF required; OMIM 190195; monoallelic no proven risk; Q90X founder mutation (Northern European); CMMRD NOT expected (unlike biallelic MMR); SBS30 CpG>TpG PATHOGNOMONIC; family cascade biallelic probands",
        "cancer_risk": "CRC: near-100% lifetime biallelic (untreated); breast: 2-5x elevated biallelic; duodenal adenomas elevated; endometrial elevated; SBS30 signature PATHOGNOMONIC; MSI-H subset -> pembrolizumab; monoallelic NTHL1 no proven elevated CRC risk",
        "pathognomonic": "COSMIC SBS30 (CpG>TpG) mutational signature PATHOGNOMONIC NTHL1-deficiency; 20-200 colonic adenomas biallelic; CRC near-100% untreated biallelic; Q90X founder mutation; NTHL1 biallelic NOT CMMRD (childhood brain tumour NOT expected, unlike biallelic MMR)",
        "surveillance_key": "Colonoscopy annually from age 20yr (biallelic); upper GI from 30yr; breast surveillance from 30yr (females); endometrial USS from 35yr; SBS30 signature + polyposis -> NTHL1 biallelic testing; Q90X hot-spot first; CMMRD brain surveillance NOT required NTHL1",
        "key_distinctions": [
            "SBS30-CpG-TpG-PATHOGNOMONIC-NTHL1",
            "CRC-NEAR-100PCT-BIALLELIC-UNTREATED",
            "Q90X-FOUNDER-MUTATION-NORTHERN-EUROPEAN",
            "CMMRD-NOT-EXPECTED-UNLIKE-BIALLELIC-MMR",
            "BREAST-ELEVATED-BIALLELIC-NTHL1",
            "COLONOSCOPY-ANNUALLY-FROM-20YR",
        ],
    },
    {
        "gene": "RNF43",
        "protein": (
            "RNF43 -- 17q22 Autosomal-Dominant-LOF -- 783aa -- "
            "RNF43-85kDa-RING-E3-Ubiquitin-Ligase-Wnt-Negative-Regulator-"
            "SFPN-Sessile-Serrated-Lesions-PATHOGNOMONIC-RSPO-Fusion-PATHOGNOMONIC-"
            "BRAF-V600E-G659Vfs41-LGK974-MSI-H-Pembrolizumab-OMIM-612482"
        ),
        "locus": "17q22",
        "protein_size": (
            "783 aa / 85 kDa / 17q22 RNF43 encodes Ring Finger Protein 43 (Wnt negative regulator): "
            "STRUCTURE: "
            "  783 aa / 85 kDa; single-pass type I transmembrane E3 ubiquitin ligase; "
            "  Signal peptide (aa 1-20): ER targeting; "
            "  Extracellular prodomain (aa 20-~175): ectodomain; "
            "  RING finger domain (aa ~177-222): E3 ubiquitin ligase catalytic activity; "
            "  TM domain (aa ~590-610): membrane anchor; "
            "  Intracellular domain (aa 610-783): downstream signalling; "
            "  RNF43 function: E3 ubiquitin ligase targeting Frizzled receptors (FZD) for lysosomal degradation; "
            "  RNF43 ubiquitinates FZD -> reduces surface Frizzled -> suppresses Wnt signalling; "
            "  RNF43 LOF -> accumulation of Frizzled receptors -> Wnt hyperactivation -> CRC; "
            "SFPN -- SERRATED FAMILIAL POLYPOSIS NEOPLASIA (RNF43): "
            "  OMIM 612482; AD LOF; germline RNF43 -> SFPN / Serrated polyposis syndrome; "
            "  Sessile serrated lesions (SSLs) / sessile serrated adenomas (SSAs): PATHOGNOMONIC; "
            "  10-50+ SSLs: predominantly right colon; "
            "  CRC via serrated pathway (BRAF-V600E + MSI-H): distinct from adenomatous polyposis; "
            "SESSILE SERRATED LESIONS (SSLs) -- PATHOGNOMONIC: "
            "  SSLs = serrated architecture + distorted crypts + mucin-filled dilated crypts = PATHOGNOMONIC; "
            "  RNF43 germline: predominantly SSL phenotype (not classic tubular adenomas); "
            "  SSL detection: requires NBI/chromoendoscopy (flat lesions easily missed); "
            "  Pathologist expertise required: SSLs under-diagnosed on standard H&E; "
            "  RNF43 SSLs -> BRAF-V600E mutation -> MLH1 methylation -> MSI-H -> CRC in subset; "
            "RSPO FUSION -- PATHOGNOMONIC FOR SPORADIC RNF43: "
            "  RSPO2/RSPO3 gene fusions: present in SPORADIC RNF43-mutant CRC ONLY (not germline); "
            "  RSPO (R-spondin) proteins: amplify LGR4/5 -> enhance Wnt via RNF43/ZNRF3 axis; "
            "  RSPO fusion + RNF43 somatic mutation: PATHOGNOMONIC sporadic serrated CRC mechanism; "
            "  Germline RNF43 tumours: DO NOT carry RSPO fusions (germline = different mechanism); "
            "  Porcupine inhibitor (LGK-974): targets Wnt ligand secretion; effective in RSPO+ RNF43 sporadic; "
            "BRAF-V600E IN RNF43 SERRATED PATHWAY: "
            "  RNF43 SSLs -> BRAF-V600E mutation (serrated oncogenesis pathway); "
            "  BRAF-V600E + MLH1 methylation -> MSI-H in subset of RNF43 CRC; "
            "  BRAF-V600E inhibitor (dabrafenib + trametinib): potential in RNF43 BRAF-mutant CRC; "
            "G659Vfs*41 -- MOST COMMON GERMLINE TRUNCATION: "
            "  G659Vfs*41: frameshift truncation at aa 659 in RNF43 intracellular domain; "
            "  Most common germline pathogenic RNF43 variant in SFPN; "
            "  Establishes truncation as pathogenic class for RNF43 (unlike missense VUS); "
            "PEMBROLIZUMAB (MSI-H SUBSET): "
            "  RNF43 CRC with MSI-H (secondary to BRAF/MLH1 pathway): pembrolizumab eligible; "
            "  MSI-H subset: ~30-40% of RNF43 serrated pathway CRC; "
            "  BRAF-V600E + MSI-H + RNF43 SSL: pembrolizumab first-line; "
            "SURVEILLANCE (RNF43 SFPN): "
            "  Colonoscopy: 1-2yr from age 25yr with NBI/chromoendoscopy (SSL detection mandatory); "
            "  Right colon: particular attention (SSLs right-predominant); "
            "  Gastroscopy from age 35yr; "
            "  Germline testing: all serrated polyposis syndrome phenotype patients"
        ),
        "inheritance": "Autosomal Dominant (AD); LOF/biallelic LOF; OMIM 612482; SFPN/serrated polyposis syndrome; G659Vfs*41 most common germline truncation; missense VUS requires functional evidence; family cascade mandatory",
        "cancer_risk": "CRC via serrated pathway; SSLs PATHOGNOMONIC (10-50+ right-colon predominant); BRAF-V600E + MSI-H subset; MSI-H CRC -> pembrolizumab; RSPO fusion PATHOGNOMONIC sporadic (NOT germline) RNF43; LGK-974 porcupine inhibitor preclinical",
        "pathognomonic": "Sessile serrated lesions (SSLs) predominantly right colon PATHOGNOMONIC RNF43; RSPO2/RSPO3 gene fusion PATHOGNOMONIC for SPORADIC (not germline) RNF43 CRC; G659Vfs*41 most common germline truncation; BRAF-V600E serrated pathway; SSL NBI/chromoendoscopy mandatory",
        "surveillance_key": "Colonoscopy 1-2yr from 25yr with NBI/chromoendoscopy; right-colon focus SSLs; gastroscopy from 35yr; G659Vfs*41 germline sequencing; MSI-H RNF43 CRC -> pembrolizumab; RSPO fusion = sporadic marker NOT germline indicator; porcupine LGK-974 preclinical trials",
        "key_distinctions": [
            "SESSILE-SERRATED-LESIONS-PATHOGNOMONIC-RNF43",
            "RSPO-FUSION-SPORADIC-NOT-GERMLINE-RNF43",
            "BRAF-V600E-SERRATED-PATHWAY-RNF43",
            "G659Vfs41-MOST-COMMON-GERMLINE-TRUNCATION",
            "NBI-CHROMOENDOSCOPY-SSL-DETECTION-MANDATORY",
            "LGK974-PORCUPINE-INHIBITOR-PRECLINICAL",
        ],
    },
    {
        "gene": "BMPR1A",
        "protein": (
            "BMPR1A -- 10q22.3 Autosomal-Dominant-LOF -- 532aa -- "
            "BMPR1A-60kDa-BMP-Type-1-Receptor-ALK3-JPS-Type1-"
            "Juvenile-Polyps-PATHOGNOMONIC-Mucus-Filled-Hamartomatous-Glands-PATHOGNOMONIC-"
            "CRC-39pct-Gastric-21pct-HHT-Overlap-ABSENT-BMPR1A-PTEN-10q-Contiguous-OMIM-601299"
        ),
        "locus": "10q22.3",
        "protein_size": (
            "532 aa / 60 kDa / 10q22.3 BMPR1A encodes Bone Morphogenetic Protein Receptor Type 1A (ALK3): "
            "STRUCTURE: "
            "  532 aa / 60 kDa; type I BMP receptor (ALK3 = Activin receptor-Like Kinase 3); "
            "  Signal peptide (aa 1-24); "
            "  Extracellular domain (aa 25-152): cysteine-rich BMP-binding domain; "
            "  TM domain (aa 153-174): single-pass transmembrane; "
            "  Intracellular kinase domain (aa 200-479): serine/threonine kinase; GS domain (aa 200-225); "
            "  BMPR1A heterodimerises with type II BMP receptors (BMPR2, ACVR2A, ACVR2B); "
            "  BMP signalling: BMPR2 transphosphorylates BMPR1A GS domain -> BMPR1A phosphorylates SMAD1/5/8; "
            "  SMAD1/5/8 + SMAD4 -> nucleus -> transcription of BMP target genes; "
            "  BMPR1A LOF -> reduced BMP signalling -> Wnt/mTOR/MAPK hyperactivation -> hamartoma; "
            "  PI3K-mTOR axis: BMP normally suppresses PI3K/AKT/mTOR; BMPR1A LOF -> mTOR hyperactivation; "
            "JPS TYPE 1 -- JUVENILE POLYPOSIS SYNDROME: "
            "  OMIM 174900; AD LOF; JPS type 1 = BMPR1A (vs type 2 = SMAD4); "
            "  Juvenile polyps: 5-200+ throughout GI tract; predominantly colon and stomach; "
            "  Age of polyposis onset: childhood to early adulthood (mean ~12yr); "
            "  Mean age of CRC diagnosis: ~35-45yr; "
            "JUVENILE POLYPS -- PATHOGNOMONIC (HISTOLOGY): "
            "  Juvenile polyp histology = PATHOGNOMONIC: "
            "    Mucus-filled dilated cystic glands: PATHOGNOMONIC (not seen in other polyposis); "
            "    Abundant oedematous lamina propria stroma (not adenomatous); "
            "    Inflammatory infiltrate (eosinophils/plasma cells); "
            "    Surface erosion; NO dysplasia in typical juvenile polyp (epithelium normal); "
            "  CONTRAST with: APC-FAP (tubular adenomas), PEUTZ-JEGHERS (smooth muscle arborisation), "
            "    SERRATED (serrated architecture), NTHL1 (standard adenomas); "
            "  Juvenile polyp + adenomatous change = RISK for CRC transformation; "
            "CRC AND GASTRIC CANCER IN JPS: "
            "  CRC: 39% lifetime in BMPR1A-JPS; "
            "  Gastric cancer: 21% lifetime = SIGNIFICANT gastric risk (unlike most hereditary CRC syndromes); "
            "  Small bowel adenomas: elevated; duodenal/pancreatic: elevated; "
            "HHT OVERLAP -- ABSENT IN BMPR1A (ONLY SMAD4-JPS): "
            "  SMAD4-JPS: 15-25% have overlapping HHT (hereditary haemorrhagic telangiectasia); "
            "  BMPR1A-JPS: HHT ABSENT (no AVMs, no telangiectasia); "
            "  CRITICAL DISTINCTION: BMPR1A vs SMAD4 JPS -- HHT screen ONLY for SMAD4; "
            "  HHT in BMPR1A = DOES NOT OCCUR -> do not screen BMPR1A for AVM; "
            "BMPR1A LARGE DELETIONS (10q22-23) -- MLPA MANDATORY: "
            "  Large deletions of 10q22-23: account for ~30-40% BMPR1A LOF (not point mutations); "
            "  MLPA: required to detect intragenic deletions + 10q22-23 contiguous deletions; "
            "  10q22-23 contiguous deletion: may co-delete PTEN (10q23.31) -> combined JPS+PHTS phenotype; "
            "  PTEN-10q contiguous deletion: macrocephaly + juvenile polyps + CRC + endometrial risk; "
            "RAPAMYCIN (mTOR) -- PRECLINICAL: "
            "  BMPR1A LOF -> mTOR hyperactivation -> rapamycin (mTOR inhibitor) pre-clinical rationale; "
            "  Rapamycin analogues (sirolimus/everolimus): pre-clinical JPS models; "
            "  Not yet standard of care in JPS BMPR1A; "
            "SURVEILLANCE (JPS BMPR1A): "
            "  Colonoscopy: annually from age 15yr (or from age of first polyp); "
            "  Gastroscopy: annually from age 15yr (gastric cancer 21%); "
            "  Small bowel surveillance from age 20yr; "
            "  BMPR1A large deletion screening: MLPA 10q22-23 including PTEN check; "
            "  HHT screening: NOT required for BMPR1A (HHT only in SMAD4); "
            "  PTEN testing: if macrocephaly present (exclude contiguous 10q deletion)"
        ),
        "inheritance": "Autosomal Dominant (AD); LOF; OMIM 601299/174900; JPS type 1; large deletions 10q22-23 common (~30-40%) -> MLPA mandatory; HHT absent in BMPR1A (HHT = SMAD4 only); PTEN 10q contiguous deletion possible; family cascade mandatory",
        "cancer_risk": "CRC: 39% lifetime; gastric: 21% lifetime = significant gastric risk; juvenile polyps PATHOGNOMONIC (mucus-filled cystic glands with abundant stroma); HHT absent BMPR1A (SMAD4 only); PTEN 10q contiguous deletion combined phenotype",
        "pathognomonic": "Juvenile polyps: mucus-filled dilated cystic glands + abundant oedematous stroma PATHOGNOMONIC (contrast adenomas/hamartomas/serrated); HHT ABSENT in BMPR1A (only SMAD4-JPS has HHT); gastric cancer 21% lifetime JPS; PTEN-10q contiguous deletion risk",
        "surveillance_key": "Colonoscopy + gastroscopy annually from age 15yr; MLPA 10q22-23 MANDATORY (large deletions + PTEN contiguous); HHT AVM screen NOT required BMPR1A; annual endometrial from 30yr (PTEN contiguous); rapamycin mTOR preclinical; SMAD4 testing if HHT features present",
        "key_distinctions": [
            "JUVENILE-POLYPS-MUCUS-FILLED-CYSTIC-GLANDS-PATHOGNOMONIC",
            "HHT-ABSENT-BMPR1A-HHT-ONLY-SMAD4-JPS",
            "GASTRIC-21PCT-LIFETIME-BMPR1A-JPS",
            "MLPA-MANDATORY-10q22-23-LARGE-DELETIONS",
            "PTEN-10q-CONTIGUOUS-DELETION-COMBINED-PHENOTYPE",
            "COLONOSCOPY-GASTROSCOPY-FROM-15YR",
        ],
    },
    {
        "gene": "MSH3",
        "protein": (
            "MSH3 -- 5q11.2 Autosomal-Recessive-biallelic-LOF -- 1128aa -- "
            "MSH3-128kDa-MutSbeta-MSH2-MSH3-Dimer-PPAP-Like-Biallelic-"
            "20-500-Adenomas-CRC-Near-100pct-EMAST-PATHOGNOMONIC-"
            "MSI-H-CMMRD-Risk-Lower-Pembrolizumab-OMIM-600887"
        ),
        "locus": "5q11.2",
        "protein_size": (
            "1128 aa / 128 kDa / 5q11.2 MSH3 encodes MutS Homolog 3 (MutSbeta heterodimer partner): "
            "STRUCTURE: "
            "  1128 aa / 128 kDa; mismatch recognition protein; "
            "  MSH3 forms MutSbeta heterodimer exclusively with MSH2 (not MSH6); "
            "  MutSbeta (MSH2+MSH3): recognises insertion-deletion loops (IDLs) 2-12 nucleotides; "
            "  MSH3 ATPase domain: conformational change on IDL recognition; "
            "  MSH3-specific substrate: longer IDLs (tetranucleotide + dinucleotide repeats); "
            "  CONTRAST with MutSalpha (MSH2+MSH6): recognises single base mismatches + 1-4 nt IDLs; "
            "  MSH3 biallelic LOF -> MutSbeta absent -> IDL repair failure (dinucleotide + tetranucleotide); "
            "  MSH2 protein: stabilised by MSH6 (MutSalpha intact); "
            "  MSH3 biallelic: MSH2 IHC RETAINED (MSH2+MSH6 intact); MSH3 IHC LOST; "
            "PPAP-LIKE BIALLELIC MSH3 POLYPOSIS: "
            "  Biallelic MSH3: 20-500 colonic adenomas (PPAP-like polyposis phenotype); "
            "  CRC: near 100% lifetime in biallelic MSH3 (untreated); "
            "  Age of CRC onset: typically 40-65yr; "
            "  Phenotype: dense adenomatous polyposis mimicking APC-attenuated FAP or MUTYH; "
            "  MSH3 biallelic also: duodenal adenomas elevated; "
            "MSH3 BIALLELIC NOT LYNCH (HETEROZYGOUS): "
            "  MSH3 monoallelic (heterozygous): NO proven Lynch syndrome CRC risk; "
            "  Lynch syndrome = monoallelic MMR gene mutations (MLH1/MSH2/MSH6/PMS2); "
            "  MSH3 monoallelic is NOT Lynch -- NO Lynch-equivalent extracolonic surveillance required; "
            "  CRITICAL: biallelic MSH3 = PPAP-like polyposis phenotype (pure AR disease); "
            "EMAST -- PATHOGNOMONIC: "
            "  EMAST = Elevated Microsatellite Alterations at Selected Tetranucleotide Repeats; "
            "  Tetranucleotide (AAAC, ATAG, etc.) repeat instability = EMAST; "
            "  EMAST PATHOGNOMONIC for MutSbeta (MSH3) deficiency; "
            "  Standard MSI testing (BAT25/BAT26 dinucleotide panel): may show MSI-H OR near-MSI; "
            "  EMAST testing (tetranucleotide markers D20S82, D8S321, etc.): MSH3-specific; "
            "  MSH3 biallelic: MSI-H on standard panel + EMAST = PATHOGNOMONIC combination; "
            "  Tumour IHC: MSH3 LOST; MSH2 RETAINED; MSH6 RETAINED; MLH1 RETAINED; "
            "CMMRD RISK -- LOWER THAN BIALLELIC MSH2/MSH6/MLH1: "
            "  Constitutional Mismatch Repair Deficiency (CMMRD): caused by biallelic MLH1/MSH2/MSH6/PMS2; "
            "  MSH3 biallelic: CMMRD risk LOWER (no CMMRD-grade childhood brain tumour/haematological); "
            "  Reason: MutSalpha (MSH2+MSH6) remains intact in MSH3 biallelic; "
            "  MSH3 biallelic children: no CMMRD brain tumour surveillance required (unlike MSH2 biallelic); "
            "  DISTINCTION from biallelic MSH2: MSH2 biallelic = CMMRD (both MutSalpha + MutSbeta lost); "
            "PEMBROLIZUMAB (MSI-H MSH3): "
            "  MSH3-biallelic CRC with MSI-H: pembrolizumab FDA-approved all histologies dMMR; "
            "  CRC near-100% biallelic -> surveillance colonoscopy mandatory or prophylactic colectomy; "
            "SURVEILLANCE (MSH3 BIALLELIC): "
            "  Colonoscopy: annually from age 15yr in biallelic; polypectomy aggressively; "
            "  Prophylactic colectomy: consider when adenoma burden unmanageable; "
            "  Upper GI endoscopy from age 25yr; "
            "  MSH3 IHC + EMAST testing: for dense adenomatous polyposis + MSH2 retained IHC; "
            "  MSH3 biallelic: cascade testing siblings (AR disease: 25% recurrence risk)"
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF required for polyposis/cancer phenotype; OMIM 600887; monoallelic MSH3 is NOT Lynch (no elevated CRC monoallelic); CMMRD risk lower than biallelic MSH2/MSH6/MLH1; EMAST PATHOGNOMONIC; family cascade mandatory",
        "cancer_risk": "CRC: near-100% lifetime biallelic (untreated); 20-500 adenomas PPAP-like; duodenal elevated; EMAST + MSI-H PATHOGNOMONIC; MSH3 IHC lost MSH2/MSH6 retained; pembrolizumab dMMR; monoallelic MSH3 = NO Lynch CRC risk",
        "pathognomonic": "EMAST (elevated microsatellite alterations at selected tetranucleotide repeats) PATHOGNOMONIC for MSH3 deficiency; MSH3 IHC lost with MSH2/MSH6/MLH1 retained = PATHOGNOMONIC MSH3 biallelic; CMMRD risk LOWER than biallelic MSH2 (MutSalpha intact); MutSbeta dimer specific",
        "surveillance_key": "Colonoscopy annually from age 15yr biallelic; consider prophylactic colectomy when burden unmanageable; EMAST testing MSH3-specific; MSH3 IHC: lost tumour / retained MSH2 MSH6; pembrolizumab MSI-H; CMMRD brain tumour surveillance NOT required MSH3 biallelic; upper GI from 25yr",
        "key_distinctions": [
            "EMAST-TETRANUCLEOTIDE-PATHOGNOMONIC-MSH3",
            "MSH3-IHC-LOST-MSH2-MSH6-RETAINED-PATHOGNOMONIC",
            "CMMRD-RISK-LOWER-MUTSBETA-INTACT-MUTSALPHA",
            "CRC-NEAR-100PCT-BIALLELIC-PPAP-LIKE",
            "MONOALLELIC-MSH3-NOT-LYNCH-NO-ELEVATED-RISK",
            "COLONOSCOPY-ANNUALLY-FROM-15YR",
        ],
    },
    {
        "gene": "GREM1",
        "protein": (
            "GREM1 -- 15q13.3 Autosomal-Dominant-GOF-3prime-Duplication-40kb -- 184aa -- "
            "Gremlin1-20kDa-BMP-Antagonist-DAN-Family-HMPS-"
            "Mixed-Polyps-PATHOGNOMONIC-Ashkenazi-Jewish-Founder-"
            "GREM1-BMP-Inhibition-Wnt-Hyperactivation-CRC-50-80pct-MLPA-CNV-MANDATORY-OMIM-603054"
        ),
        "locus": "15q13.3",
        "protein_size": (
            "184 aa / 20 kDa / 15q13.3 GREM1 encodes Gremlin 1 (BMP antagonist / DAN family): "
            "STRUCTURE: "
            "  184 aa / 20 kDa; secreted glycoprotein; DAN (differential screening-selected gene aberrative) family; "
            "  Signal peptide (aa 1-21): ER/secretory pathway; "
            "  Cystine-knot motif (aa ~80-184): hallmark of cystine-knot cytokines; DAN cysteine-rich domain; "
            "  GREM1 secreted as homodimer (disulfide-linked); "
            "  BMP antagonist function: GREM1 binds BMP2/4/7 ligands -> prevents BMP receptor binding; "
            "  BMP signalling normally: BMPR1A/BMPR2 -> SMAD1/5/8 -> SMAD4 -> anti-proliferative; "
            "  GREM1 GOF (overexpression via 3' duplication): excess GREM1 -> BMP neutralised -> anti-proliferative lost; "
            "  Downstream: GREM1 overexpression -> Wnt hyperactivation (BMP suppresses Wnt; GREM1 releases Wnt); "
            "  GREM1 also activates FGFR1 (BMP-independent signalling -> proliferation); "
            "3-PRIME DUPLICATION (~40kb) -- UNIQUE MECHANISM: "
            "  GREM1 pathogenic variant = 3'-flanking ~40kb duplication (NOT coding point mutation); "
            "  Duplication inserts enhancer element 3' to GREM1 coding sequence; "
            "  Enhancer-driven GREM1 overexpression: ectopic/elevated GREM1 in colonic epithelium; "
            "  Standard germline sequencing (NGS/Sanger): MISSES duplication; "
            "  MLPA / CNV (copy number variation) analysis MANDATORY: detects 3' duplication; "
            "  Array CGH or targeted CNV: GREM1 locus 15q13.3 duplication detection; "
            "HMPS -- HEREDITARY MIXED POLYPOSIS SYNDROME: "
            "  OMIM 601228; AD GOF (3' duplication); exclusively Ashkenazi-Jewish founder; "
            "  Mixed polyps: adenomas + hyperplastic polyps + sessile serrated lesions + juvenile polyps = PATHOGNOMONIC; "
            "  PATHOGNOMONIC: simultaneous presence of multiple polyp histological types in same patient; "
            "  Adenoma-to-carcinoma: predominant CRC pathway in HMPS; "
            "  Polyp count: 5-50+ mixed polyps; onset 30-40yr; "
            "MIXED POLYPS -- PATHOGNOMONIC: "
            "  HMPS mixed polyp combination = PATHOGNOMONIC for GREM1 3' duplication; "
            "  No other hereditary CRC syndrome produces this SPECIFIC histological mixture; "
            "  CONTRAST: APC-FAP (pure tubular adenomas); JPS-BMPR1A (pure juvenile); RNF43 (pure SSL); "
            "  Mixed polyposis -> GREM1 CNV testing = priority before MYH/MSH6/SMAD4 panels; "
            "ASHKENAZI-JEWISH FOUNDER -- EXCLUSIVELY: "
            "  GREM1 3' 40kb duplication: founder mutation restricted to Ashkenazi-Jewish population; "
            "  Non-Ashkenazi HMPS: GREM1 duplication very rarely reported; "
            "  Ashkenazi-Jewish mixed polyposis: GREM1 CNV = FIRST test; "
            "  Prevalence: ~1/8000 Ashkenazi-Jewish population; "
            "  Family cascade: Jewish ancestry + mixed polyposis + family CRC -> GREM1 CNV MANDATORY; "
            "NO BRAF-V600E (CONTRAST RNF43): "
            "  GREM1 HMPS: adenomatous polyps -> CRC via KRAS pathway (NOT BRAF-V600E); "
            "  BRAF-V600E absent in GREM1-HMPS = DISTINCTION from RNF43 serrated pathway; "
            "  Testing: KRAS mutation in GREM1 CRC tumours (characteristic); "
            "CRC RISK IN HMPS: "
            "  CRC: 50-80% lifetime in HMPS GREM1 duplication carriers; "
            "  Mean CRC diagnosis age: ~50yr (range 28-75yr); "
            "  Small bowel adenomas: elevated; gastric polyps: elevated; "
            "  Endometrial: not significantly elevated (vs SMAD4/BMPR1A); "
            "SURVEILLANCE (GREM1 HMPS): "
            "  Colonoscopy: annually from age 25yr (or 10yr before earliest family CRC); "
            "  Gastroscopy from age 30yr; "
            "  No SMAD4-HHT screen required (GREM1 is not SMAD4-pathway); "
            "  Genetic testing: GREM1 CNV/MLPA first in Ashkenazi-Jewish mixed polyposis; "
            "  Family cascade: all first-degree relatives if Ashkenazi-Jewish heritage; "
            "TREATMENT (GREM1-HMPS): "
            "  Endoscopic polypectomy: surveillance colonoscopy with polypectomy of all polyp types; "
            "  Prophylactic colectomy: when adenoma burden or high-risk mixed polyps unmanageable; "
            "  Aspirin: chemoprevention being investigated; "
            "  Targeted: BMP agonism (recombinant BMP) or mTOR inhibition: preclinical only"
        ),
        "inheritance": "Autosomal Dominant (AD); GOF via 3'-flanking ~40kb duplication (NOT coding point mutation); OMIM 603054/601228; exclusively Ashkenazi-Jewish founder; MLPA/CNV MANDATORY (Sanger misses duplication); family cascade mandatory; HHT NOT associated",
        "cancer_risk": "CRC: 50-80% lifetime; mixed polyposis PATHOGNOMONIC (adenomas + hyperplastic + SSL + juvenile simultaneously); small bowel adenomas elevated; NO BRAF-V600E (KRAS pathway); no significant endometrial elevation; GREM1 exclusively Ashkenazi-Jewish",
        "pathognomonic": "Mixed polyps (adenomas + hyperplastic + sessile serrated + juvenile simultaneously) PATHOGNOMONIC for GREM1-HMPS; exclusively Ashkenazi-Jewish founder 3' ~40kb duplication; MLPA/CNV MANDATORY (Sanger misses duplication); NO BRAF-V600E (contrast RNF43); colonoscopy annual from 25yr",
        "surveillance_key": "Colonoscopy annually from age 25yr; gastroscopy from 30yr; GREM1 CNV/MLPA: Ashkenazi-Jewish mixed polyposis FIRST test; NO HHT AVM screen (not SMAD4); aspirin chemoprevention investigated; prophylactic colectomy when burden unmanageable; family cascade Ashkenazi-Jewish heritage",
        "key_distinctions": [
            "MIXED-POLYPS-PATHOGNOMONIC-GREM1-HMPS",
            "ASHKENAZI-JEWISH-FOUNDER-EXCLUSIVELY",
            "MLPA-CNV-MANDATORY-NOT-SANGER-3PRIME-DUPLICATION",
            "NO-BRAF-V600E-CONTRAST-RNF43-KRAS-PATHWAY",
            "CRC-50-80PCT-LIFETIME-HMPS",
            "COLONOSCOPY-ANNUALLY-FROM-25YR",
        ],
    },
]


def _make_patients(gene_entry: dict) -> list:
    gene = gene_entry["gene"]
    seed = SEED_BASE + ATLAS_GENES.index(gene_entry)
    rng  = random.Random(seed)

    # Age-of-onset distributions for CRC
    # Late-onset for POLE/POLD1/EPCAM (Lynch-equivalent); early onset for polyposis syndromes
    age_params = {
        "POLE":   (50, 14),   # CRC typically 40-60yr in PPAP-1
        "POLD1":  (45, 13),   # CRC 30-60yr, earlier onset vs POLE
        "EPCAM":  (47, 13),   # Lynch-equivalent MSH2-silenced
        "NTHL1":  (52, 12),   # Adult onset biallelic NAP CRC
        "RNF43":  (54, 13),   # Serrated pathway CRC; adult onset
        "BMPR1A": (40, 12),   # JPS; mean CRC ~35-45yr; polyposis from childhood
        "MSH3":   (55, 12),   # Biallelic PPAP-like; adult CRC
        "GREM1":  (50, 14),   # HMPS; mean CRC ~50yr
    }
    mu, sigma = age_params.get(gene, (50, 13))

    severe_rates = {
        "POLE":   0.55,   # Exceptional pembrolizumab but ultra-TMB = aggressive
        "POLD1":  0.62,   # CRC 80% + brain tumour + sebaceous
        "EPCAM":  0.58,   # Lynch-equivalent MSH2 silenced
        "NTHL1":  0.68,   # Near-100% CRC biallelic + breast
        "RNF43":  0.50,   # Serrated pathway; variable penetrance
        "BMPR1A": 0.60,   # Gastric 21% + CRC 39%; polyposis early onset
        "MSH3":   0.65,   # Near-100% CRC biallelic; dense polyposis
        "GREM1":  0.57,   # CRC 50-80%; mixed polyposis
    }
    sev_rate = severe_rates.get(gene, 0.5)

    patients = []
    for i in range(40):
        age       = max(5, round(rng.gauss(mu, sigma), 1))
        sev_event = rng.random() < sev_rate
        patients.append({
            "id":        f"{gene}-{i+1:02d}",
            "age_onset": age,
            "severe":    sev_event,
            "seed":      seed,
        })
    return patients


def generate_overview():
    rows = []
    for g in ATLAS_GENES:
        pts   = _make_patients(g)
        sev_n = sum(1 for p in pts if p["severe"])
        rows.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "severe_n":         sev_n,
            "severe_pct":       round(sev_n / len(pts) * 100, 1),
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
            "inheritance":      g["inheritance"],
            "cancer_risk":      g["cancer_risk"],
            "protein":          g["protein"],
        })

    total_pts  = sum(r["n"]        for r in rows)
    total_sev  = sum(r["severe_n"] for r in rows)
    highest    = max(rows, key=lambda r: r["severe_pct"])

    return {
        "atlas":              "Hereditary-Colorectal-Cancer-Predisposition-Atlas",
        "seed_range":         f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes_n":            len(ATLAS_GENES),
        "total_patients":     total_pts,
        "severe_total_n":     total_sev,
        "severe_total_pct":   round(total_sev / total_pts * 100, 1),
        "highest_risk_gene":  highest["gene"],
        "highest_risk_pct":   highest["severe_pct"],
        "gene_summary":       rows,
        "genes_detail": [
            {
                "gene":             g["gene"],
                "inheritance":      g["inheritance"],
                "cancer_risk":      g["cancer_risk"],
                "pathognomonic":    g["pathognomonic"],
                "surveillance_key": g["surveillance_key"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_breakdown():
    breakdown = []
    for g in ATLAS_GENES:
        pts      = _make_patients(g)
        seed_idx = ATLAS_GENES.index(g)
        sev_n    = sum(1 for p in pts if p["severe"])
        breakdown.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "seed":             SEED_BASE + seed_idx,
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "severe_n":         sev_n,
            "severe_pct":       round(sev_n / len(pts) * 100, 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
        })
    return {"atlas": "Hereditary-Colorectal-Cancer-Predisposition-Atlas", "breakdown": breakdown}


def generate_definitions():
    defs = [
        {
            "term": "POLE / PPAP-1 / ULTRA-HYPERMUTATED-TMB-GT100-MSS / EXCEPTIONAL-PEMBROLIZUMAB / SBS10A-SBS10B",
            "definition": (
                "POLE -- 2286aa / 261 kDa / 12q24.33 / AD GOF exonuclease domain\n"
                "PPAP-1 Polymerase-Proofreading-Associated-Polyposis; ultra-hypermutated TMB >100 MSS-mimic; "
                "exceptional pembrolizumab complete remissions; SBS10a/SBS10b PATHOGNOMONIC.\n\n"
                "ULTRA-HYPERMUTATION MECHANISM:\n"
                "  POLE exonuclease domain GOF (L424V/P286R/V411L/S459F): proofreading abrogated.\n"
                "  TMB >100 mut/Mb = ultra-hypermutated = HIGHEST TMB any hereditary CRC syndrome.\n"
                "  CRITICAL: POLE ultra-hypermutated = MSS (microsatellite stable) by standard testing.\n"
                "  MMR IHC: all four proteins RETAINED (MLH1/MSH2/MSH6/PMS2 intact).\n"
                "  Standard MSI-PCR/IHC MISSES POLE -> POLE sequencing mandatory for TMB >100 MSS CRC.\n\n"
                "COSMIC SIGNATURES SBS10a/SBS10b -- PATHOGNOMONIC:\n"
                "  SBS10a + SBS10b: C>A and C>T hypermutation pattern = PATHOGNOMONIC POLE exonuclease GOF.\n"
                "  Tumour COSMIC analysis: SBS10a/10b >20% = POLE exonuclease involvement.\n\n"
                "EXCEPTIONAL PEMBROLIZUMAB:\n"
                "  POLE ultra-hypermutated CRC/endometrial: exceptional pembrolizumab complete remissions.\n"
                "  Mechanism: ultra-hypermutation -> extreme neoantigen burden -> T-cell recognition.\n"
                "  Exonuclease domain pathogenic variant required (NOT polymerase domain VUS).\n\n"
                "CANCER SPECTRUM (PPAP-1):\n"
                "  CRC: 60-80% lifetime; endometrial: 30-40% (females); duodenal: elevated.\n"
                "  10-100 colonic adenomas; right-colon predilection.\n\n"
                "HOTSPOTS: L424V (most common), P286R (highest TMB), V411L (endometrial), S459F.\n\n"
                "SURVEILLANCE:\n"
                "  Colonoscopy 1-2yr from age 25yr; upper GI from 30yr; endometrial USS from 30yr (females).\n"
                "  POLE germline: all MSS CRC with TMB >100 mut/Mb."
            ),
        },
        {
            "term": "POLD1 / PPAP-2 / CRC-80PCT / SEBACEOUS-TUMOURS-PATHOGNOMONIC / BRAIN-TUMOURS / CERALASERTIB",
            "definition": (
                "POLD1 -- 1107aa / 125 kDa / 19q13.33 / AD GOF exonuclease domain\n"
                "PPAP-2; CRC 80% lifetime; sebaceous gland tumours PATHOGNOMONIC vs POLE; brain tumours elevated; "
                "ceralasertib + pembrolizumab.\n\n"
                "SEBACEOUS TUMOURS -- PATHOGNOMONIC FOR POLD1 vs POLE:\n"
                "  Sebaceous adenoma + sebaceous carcinoma: PRESENT in POLD1, NOT characteristic of POLE.\n"
                "  CRITICAL DISTINCTION: sebaceous neoplasm + PPAP polyposis -> POLD1 sequencing MANDATORY.\n"
                "  Any sebaceous neoplasm + colonic adenomatous polyposis -> test POLD1 specifically.\n\n"
                "BRAIN TUMOURS -- POLD1:\n"
                "  Glioma/astrocytoma: elevated in POLD1 (unlike POLE).\n"
                "  Brain MRI annually from age 18yr; paediatric brain tumour possible.\n\n"
                "CRC RISK:\n"
                "  CRC: 80% lifetime = HIGHEST POLD1 penetrance; endometrial: 30-40% (females).\n"
                "  L474P (most common hotspot), D316H, E318K exonuclease domain.\n\n"
                "CERALASERTIB + PEMBROLIZUMAB:\n"
                "  POLD1-hypermutated: replication stress -> ATR dependence -> ceralasertib synthetic lethal.\n"
                "  TMB-high POLD1 CRC: pembrolizumab + ceralasertib clinical trials.\n\n"
                "SURVEILLANCE:\n"
                "  Colonoscopy 1-2yr from age 20yr; annual brain MRI from 18yr; annual dermatology.\n"
                "  Annual endometrial USS from 25yr (females); POLD1 germline: sebaceous + polyposis."
            ),
        },
        {
            "term": "EPCAM / LYNCH-BY-MSH2-SILENCING / MLPA-MANDATORY / SMALL-BOWEL-PATHOGNOMONIC / CRC-50-80PCT",
            "definition": (
                "EPCAM -- 314aa / 35 kDa / 2p21 / AD LOF 3'-deletion only\n"
                "Lynch-by-MSH2-epigenetic-silencing; MLPA mandatory; small bowel 10-17% HIGHEST Lynch; "
                "MSH2+MSH6 IHC lost with EPCAM IHC normal; CRC 50-80%.\n\n"
                "UNIQUE EPIGENETIC MECHANISM:\n"
                "  3'-EPCAM deletion -> read-through EPCAM-MSH2 transcript -> MSH2 promoter methylation.\n"
                "  Result: MSH2 epigenetically silenced -> MSH2+MSH6 IHC LOST; EPCAM IHC NORMAL.\n"
                "  Standard NGS/Sanger: MISSES EPCAM deletion (copy-number event, not point mutation).\n\n"
                "MLPA MANDATORY:\n"
                "  MLPA 3'-EPCAM-specific probe set: detects del exon 7/8/9 (diagnostic).\n"
                "  Rule: MSH2+MSH6 IHC lost + MSH2 coding normal -> EPCAM MLPA MANDATORY.\n\n"
                "SMALL BOWEL -- HIGHEST LYNCH RISK:\n"
                "  Small bowel adenocarcinoma: 10-17% lifetime in EPCAM Lynch = HIGHEST small bowel Lynch.\n"
                "  Small bowel MRI/capsule endoscopy from age 30yr MANDATORY in EPCAM carriers.\n\n"
                "LYNCH CANCER SPECTRUM:\n"
                "  CRC: 50-80%; endometrial: 40-60% DOMINANT females; urinary tract: 25-28%; ovarian: 10-12%.\n\n"
                "PEMBROLIZUMAB (dMMR MSI-H): FDA-approved all histologies.\n"
                "ASPIRIN CAPP2: 50% CRC risk reduction = LEVEL A evidence Lynch.\n\n"
                "SURVEILLANCE:\n"
                "  Colonoscopy 1-2yr from 25yr; small bowel capsule/MRI from 30yr.\n"
                "  Annual gynaecological from 30yr; urinary cystoscopy 2yr from 25yr."
            ),
        },
        {
            "term": "NTHL1 / NAP / SBS30-CpG-TpG / CRC-NEAR-100PCT-BIALLELIC / Q90X-FOUNDER / CMMRD-NOT-EXPECTED",
            "definition": (
                "NTHL1 -- 312aa / 35 kDa / 16p13.3 / AR biallelic LOF\n"
                "NAP NTHL1-Associated-Polyposis; BER deficiency; CpG>TpG SBS30 PATHOGNOMONIC; "
                "CRC near-100% biallelic; Q90X European founder; CMMRD NOT expected.\n\n"
                "COSMIC SBS30 -- PATHOGNOMONIC:\n"
                "  SBS30: CpG>TpG mutational signature = NTHL1-deficiency PATHOGNOMONIC.\n"
                "  SBS30 >20% tumour contribution = NTHL1 biallelic LOF likely.\n"
                "  Distinct from SBS10a/SBS10b (POLE) -- different BER vs proofreading mechanism.\n\n"
                "CRC NEAR-100% BIALLELIC (UNTREATED):\n"
                "  20-200 colonic adenomas; CRC near-100% lifetime in biallelic NTHL1.\n"
                "  Monoallelic NTHL1: NO proven elevated CRC/polyposis risk (pure AR).\n\n"
                "Q90X FOUNDER MUTATION:\n"
                "  c.268C>T (Q90X): most common pathogenic NTHL1 variant; Northern European founder.\n"
                "  Q90X homozygous: full NAP phenotype.\n\n"
                "CMMRD NOT EXPECTED:\n"
                "  NTHL1 biallelic does NOT cause childhood CMMRD (unlike biallelic MMR genes).\n"
                "  MutSalpha (MSH2+MSH6) intact -> no CMMRD brain/haematological risk at childhood onset.\n"
                "  Biallelic MSH2/MLH1/MSH6 -> CMMRD; biallelic NTHL1 -> NAP (adult onset).\n\n"
                "BREAST CANCER: ~2-5x elevated in biallelic NTHL1 females.\n\n"
                "SURVEILLANCE:\n"
                "  Colonoscopy annually from age 20yr; upper GI from 30yr.\n"
                "  Annual breast surveillance from 30yr (biallelic females); endometrial from 35yr.\n"
                "  CMMRD brain tumour surveillance NOT required NTHL1."
            ),
        },
        {
            "term": "RNF43 / SFPN / SESSILE-SERRATED-LESIONS-PATHOGNOMONIC / RSPO-FUSION-SPORADIC / BRAF-V600E / LGK974",
            "definition": (
                "RNF43 -- 783aa / 85 kDa / 17q22 / AD LOF\n"
                "SFPN Serrated-Familial-Polyposis-Neoplasia; SSLs PATHOGNOMONIC; "
                "RSPO fusion PATHOGNOMONIC SPORADIC (NOT germline) RNF43; BRAF-V600E serrated pathway; "
                "G659Vfs*41 most common germline truncation.\n\n"
                "SESSILE SERRATED LESIONS -- PATHOGNOMONIC:\n"
                "  SSLs (sessile serrated adenomas): distorted crypts + mucin-filled dilated crypts = PATHOGNOMONIC.\n"
                "  Right-colon predominant; 10-50+ SSLs.\n"
                "  NBI/chromoendoscopy MANDATORY for SSL detection (flat lesions missed on white-light).\n\n"
                "RSPO FUSION -- PATHOGNOMONIC SPORADIC (NOT GERMLINE) RNF43:\n"
                "  RSPO2/RSPO3 gene fusions: present in SPORADIC RNF43-mutant CRC ONLY.\n"
                "  Germline RNF43 tumours: DO NOT carry RSPO fusions.\n"
                "  Porcupine inhibitor LGK-974: effective in RSPO+ sporadic RNF43 (preclinical/early trials).\n\n"
                "BRAF-V600E SERRATED PATHWAY:\n"
                "  RNF43 SSLs -> BRAF-V600E -> MLH1 methylation -> MSI-H in ~30-40% RNF43 CRC.\n"
                "  MSI-H RNF43 CRC: pembrolizumab eligible.\n\n"
                "G659Vfs*41: Most common germline pathogenic RNF43 truncation (SFPN).\n\n"
                "SURVEILLANCE:\n"
                "  Colonoscopy 1-2yr from 25yr WITH NBI/chromoendoscopy.\n"
                "  Gastroscopy from 35yr; germline testing: all serrated polyposis phenotype."
            ),
        },
        {
            "term": "BMPR1A / JPS-TYPE-1 / JUVENILE-POLYPS-PATHOGNOMONIC / HHT-ABSENT-BMPR1A / GASTRIC-21PCT / PTEN-10q-CONTIGUOUS",
            "definition": (
                "BMPR1A -- 532aa / 60 kDa / 10q22.3 / AD LOF\n"
                "JPS Type 1; juvenile polyps PATHOGNOMONIC; HHT absent BMPR1A (SMAD4 only); "
                "gastric 21%; PTEN-10q contiguous deletion; MLPA mandatory.\n\n"
                "JUVENILE POLYPS -- PATHOGNOMONIC (HISTOLOGY):\n"
                "  Mucus-filled dilated cystic glands + abundant oedematous stroma = PATHOGNOMONIC.\n"
                "  No dysplasia in typical juvenile polyp; inflammatory infiltrate.\n"
                "  CONTRAST: APC (tubular adenomas); PJS (smooth muscle arborisation); RNF43 (SSLs).\n\n"
                "HHT ABSENT -- BMPR1A (ONLY SMAD4-JPS HAS HHT):\n"
                "  BMPR1A-JPS: NO hereditary haemorrhagic telangiectasia; NO AVMs.\n"
                "  HHT AVM screen: NOT required for BMPR1A.\n"
                "  SMAD4-JPS: 15-25% HHT overlap -> screen for AVM in SMAD4 NOT BMPR1A.\n\n"
                "CANCER RISK:\n"
                "  CRC: 39% lifetime; gastric: 21% lifetime = significant gastric risk.\n"
                "  Gastroscopy annually from age 15yr MANDATORY (gastric 21% is high).\n\n"
                "PTEN-10q CONTIGUOUS DELETION:\n"
                "  10q22-23 deletion may co-delete PTEN (10q23.31) -> JPS + PHTS combined phenotype.\n"
                "  MLPA 10q22-23: detects BMPR1A deletion + checks PTEN co-deletion.\n"
                "  Macrocephaly present -> 10q contiguous deletion + PTEN workup.\n\n"
                "SURVEILLANCE:\n"
                "  Colonoscopy + gastroscopy annually from age 15yr.\n"
                "  MLPA 10q22-23 MANDATORY; PTEN testing if macrocephaly.\n"
                "  Rapamycin mTOR inhibitor preclinical in JPS models."
            ),
        },
        {
            "term": "MSH3 / PPAP-LIKE-BIALLELIC / EMAST-PATHOGNOMONIC / CRC-NEAR-100PCT / CMMRD-LOWER / MONOALLELIC-NOT-LYNCH",
            "definition": (
                "MSH3 -- 1128aa / 128 kDa / 5q11.2 / AR biallelic LOF\n"
                "PPAP-like biallelic MSH3; 20-500 adenomas; CRC near-100% biallelic; "
                "EMAST PATHOGNOMONIC; CMMRD risk LOWER than biallelic MSH2/MSH6/MLH1; "
                "monoallelic MSH3 NOT Lynch.\n\n"
                "EMAST -- PATHOGNOMONIC (MSH3 DEFICIENCY):\n"
                "  EMAST = Elevated Microsatellite Alterations at Selected Tetranucleotide Repeats.\n"
                "  Tetranucleotide instability (AAAC, ATAG repeats) = PATHOGNOMONIC MutSbeta deficiency.\n"
                "  Standard MSI panel (BAT25/BAT26) + EMAST testing = MSH3-specific combination.\n"
                "  Tumour IHC: MSH3 LOST; MSH2 RETAINED; MSH6 RETAINED; MLH1 RETAINED.\n\n"
                "CMMRD RISK -- LOWER THAN BIALLELIC MSH2/MSH6/MLH1:\n"
                "  MutSalpha (MSH2+MSH6) intact in MSH3 biallelic -> no CMMRD childhood cancer.\n"
                "  MSH3 biallelic children: NO CMMRD brain tumour / haematological surveillance required.\n"
                "  CONTRAST: biallelic MSH2 = both MutSalpha + MutSbeta lost = full CMMRD.\n\n"
                "MONOALLELIC MSH3 NOT LYNCH:\n"
                "  MSH3 heterozygous monoallelic: NO proven CRC/Lynch risk.\n"
                "  Lynch syndrome = monoallelic MLH1/MSH2/MSH6/PMS2 only.\n\n"
                "CRC NEAR-100% BIALLELIC:\n"
                "  20-500 adenomas PPAP-like; CRC near-100% lifetime biallelic untreated.\n"
                "  Colonoscopy annually from 15yr; prophylactic colectomy when burden unmanageable.\n\n"
                "PEMBROLIZUMAB (dMMR MSI-H subset): eligible if MSI-H MSH3-biallelic CRC.\n\n"
                "CASCADE TESTING -- HEREDITARY CRC PANEL:\n"
                "  POLE: ultra-TMB MSS CRC -> POLE exonuclease sequencing MANDATORY.\n"
                "  POLD1: sebaceous neoplasm + polyposis -> POLD1 germline.\n"
                "  EPCAM: MSH2+MSH6 IHC lost + MSH2 coding normal -> EPCAM MLPA MANDATORY.\n"
                "  NTHL1: SBS30 + 20-200 adenomas -> NTHL1 biallelic; Q90X hot-spot first.\n"
                "  RNF43: serrated polyposis + SSLs -> RNF43 sequencing; NBI colonoscopy.\n"
                "  BMPR1A: juvenile polyps + gastric 21% -> BMPR1A MLPA 10q22-23.\n"
                "  MSH3: EMAST + MSH3 IHC lost + dense adenomas -> MSH3 biallelic.\n"
                "  GREM1: Ashkenazi-Jewish + mixed polyposis -> GREM1 CNV FIRST."
            ),
        },
        {
            "term": "GREM1 / HMPS / MIXED-POLYPS-PATHOGNOMONIC / ASHKENAZI-JEWISH-FOUNDER / MLPA-CNV-MANDATORY / NO-BRAF-V600E",
            "definition": (
                "GREM1 -- 184aa / 20 kDa / 15q13.3 / AD GOF 3'-~40kb duplication\n"
                "HMPS Hereditary-Mixed-Polyposis-Syndrome; mixed polyps PATHOGNOMONIC; "
                "exclusively Ashkenazi-Jewish founder; MLPA/CNV mandatory; NO BRAF-V600E (contrast RNF43).\n\n"
                "MIXED POLYPS -- PATHOGNOMONIC:\n"
                "  Adenomas + hyperplastic polyps + SSLs + juvenile polyps simultaneously = PATHOGNOMONIC HMPS.\n"
                "  No other hereditary CRC syndrome produces this specific histological mixture.\n"
                "  CONTRAST: APC (pure adenomas); JPS (pure juvenile); RNF43 (pure SSLs); POLE (adenomas).\n\n"
                "ASHKENAZI-JEWISH FOUNDER -- EXCLUSIVELY:\n"
                "  GREM1 3' ~40kb duplication: founder mutation restricted to Ashkenazi-Jewish population.\n"
                "  Prevalence: ~1/8000 Ashkenazi-Jewish; non-Ashkenazi HMPS: GREM1 duplication very rare.\n"
                "  Ashkenazi-Jewish mixed polyposis: GREM1 CNV = FIRST genetic test.\n\n"
                "MLPA/CNV MANDATORY:\n"
                "  Standard germline NGS/Sanger: MISSES 3' duplication (copy-number event).\n"
                "  MLPA or array-CGH / targeted CNV at 15q13.3 MANDATORY for GREM1 diagnosis.\n\n"
                "NO BRAF-V600E (CONTRAST RNF43):\n"
                "  GREM1-HMPS: adenoma -> CRC via KRAS pathway (NOT BRAF-V600E serrated).\n"
                "  BRAF-V600E presence in HMPS-like polyposis -> reconsider RNF43 not GREM1.\n\n"
                "MECHANISM:\n"
                "  GREM1 overexpression (3' enhancer duplication) -> BMP neutralised -> Wnt hyperactivation.\n"
                "  PI3K/AKT/mTOR also activated downstream.\n\n"
                "CRC RISK: 50-80% lifetime; small bowel adenomas elevated; gastric polyps elevated.\n\n"
                "SURVEILLANCE:\n"
                "  Colonoscopy annually from age 25yr; gastroscopy from 30yr.\n"
                "  No HHT/AVM screen (not SMAD4); GREM1 CNV all Ashkenazi-Jewish mixed polyposis.\n"
                "  Aspirin chemoprevention under investigation; prophylactic colectomy if burden unmanageable."
            ),
        },
    ]

    return {
        "atlas":       "Hereditary-Colorectal-Cancer-Predisposition-Atlas",
        "seed_range":  f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": defs,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(json.dumps({k: v for k, v in ov.items() if k not in ("genes_detail",)}, indent=2))
    print("\n=== BREAKDOWN summary ===")
    br = generate_breakdown()
    for row in br["breakdown"]:
        print(f"  {row['gene']:10s} n={row['n']} mean_age={row['mean_age_onset']} "
              f"severe_n={row['severe_n']} ({row['severe_pct']}%)")
    print("\n=== DEFINITIONS (terms only) ===")
    df = generate_definitions()
    for d in df["definitions"]:
        print(f"  {d['term']}")
