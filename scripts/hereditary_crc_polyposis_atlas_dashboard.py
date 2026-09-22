#!/usr/bin/env python3
"""Hereditary-CRC-Polyposis-Atlas — Complete 8-Gene Colorectal Cancer & Polyposis Syndrome Atlas
MLH1   (MutL homologue 1; 852 aa; 3p22.2; AD LOF;
         Lynch syndrome 1; most common MMR gene ~35-40% of Lynch;
         CRC lifetime risk 40-70%; endometrial 30-60%;
         MLH1 PROMOTER METHYLATION (somatic, sporadic) must be excluded before Lynch diagnosis;
         seed SEED_BASE+0) ·
MSH2   (MutS homologue 2; 934 aa; 2p21; AD LOF;
         Lynch syndrome 2; ~31% of Lynch;
         CRC 45-70%; endometrial 40-60%;
         SEBACEOUS NEOPLASMS (Muir-Torre) + GLIOBLASTOMA (Turcot) PATHOGNOMONIC;
         EPCAM deletion silences MSH2 promoter -- check EPCAM deletion if MSH2 IHC lost;
         seed SEED_BASE+1) ·
MSH6   (MutS homologue 6; 1360 aa; 2p16.3; AD LOF;
         Lynch syndrome 5; ~18% of Lynch;
         ENDOMETRIAL > CRC in females (endometrial 40-70%); reduced CRC penetrance vs MLH1/MSH2;
         MSI-L or MSS possible (frameshift repair intact); often missed on older panels;
         seed SEED_BASE+2) ·
PMS2   (PMS1 homologue 2; 862 aa; 7p22.1; AD LOF;
         Lynch syndrome 4; ~14% of Lynch; LOWEST penetrance among MMR genes;
         BIALLELIC PMS2 = CMMRD (Constitutional MMR Deficiency) -- childhood CNS + haematological + GI tumours;
         Pseudogenes (PMS2CL) require long-range PCR; standard NGS MISSES large PMS2 deletions;
         seed SEED_BASE+3) ·
APC    (Adenomatous polyposis coli; 2843 aa; 5q22.2; AD LOF;
         FAP / AFAP / Gardner syndrome / Turcot;
         >100 COLORECTAL ADENOMAS PATHOGNOMONIC; CRC 100% by age 40 if untreated;
         PROPHYLACTIC COLECTOMY MANDATORY by age 25-35;
         Spigelman staging for duodenal/ampullary surveillance;
         Desmoid tumours INTRA-ABDOMINAL (post-surgery) unpredictable;
         seed SEED_BASE+4) ·
MUTYH  (MutY DNA glycosylase; 546 aa; 1p34.1; AR;
         MAP (MUTYH-associated polyposis);
         BIALLELIC required for MAP; 10-100 adenomas; CRC risk 80% by age 70;
         Y179C (c.536A>G) + G396D (c.1187G>A) common European founders;
         Heterozygous carriers ~1.5-2x CRC risk only (NOT Lynch-level);
         seed SEED_BASE+5) ·
STK11  (Serine/threonine kinase 11; 433 aa; 19p13.3; AD LOF;
         Peutz-Jeghers syndrome (PJS);
         MUCOCUTANEOUS LENTIGINES perioral/buccal/digits PATHOGNOMONIC (appear birth-childhood, fade puberty);
         HAMARTOMATOUS POLYPS (small bowel >> colon, stomach); intussusception risk;
         cancer lifetime risk 93%: CRC, small bowel, pancreatic (132x relative risk), breast, ovarian;
         seed SEED_BASE+6) ·
SMAD4  (SMAD family member 4; 552 aa; 18q21.2; AD LOF;
         Juvenile Polyposis Syndrome (JPS) / JPS-HHT Overlap;
         JUVENILE POLYPS (smooth mucosa, stalk, expanded lamina propria) PATHOGNOMONIC;
         SMAD4-JPS: HHT overlap in ~22% -- check epistaxis + telangiectasia + pulmonary AVM;
         CRC lifetime risk 40%; gastric polyps frequent;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3086-3093)
"""
import random

SEED_BASE = 3086

ATLAS_GENES = [
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 852aa -- MutL-Homologue-1-MMR-Mismatch-Repair-"
            "Lynch-Syndrome-1-Most-Common-35-40pct-CRC-40-70pct-Lifetime-dMMR-MSI-H-Pembrolizumab-"
            "OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "852 aa / 84 kDa (MLH1; MutL alpha complex with PMS2; "
            "STRUCTURE: N-terminal ATPase domain; C-terminal dimerisation domain with PMS2; "
            "heterodimer MLH1-PMS2 = MutL alpha (most abundant MMR endonuclease complex); "
            "also forms MutL beta (MLH1-PMS1) and MutL gamma (MLH1-MLH3) -- minor complexes; "
            "FUNCTION: "
            "  Post-replication mismatch repair (MMR): recognises MutS-mismatch complex -> nicks strand -> excision; "
            "  MLH1 is the essential scaffold subunit: PMS2 stabilised by MLH1 (PMS2 absent in MLH1-LOF IHC); "
            "  MLH1-PMS2 IHC: concomitant loss of BOTH proteins = MLH1 germline OR somatic methylation; "
            "  MLH1 promoter hypermethylation (somatic): 15% of all CRC -- sporadic MSI-H; "
            "LYNCH SYNDROME 1 (LS1): "
            "  Most common Lynch gene: 35-40% of all Lynch families; "
            "  CRC lifetime risk: 40-70% (age 70); mean age diagnosis 44 years (vs 63 sporadic); "
            "  Endometrial: 30-60% lifetime; Lynch-associated CRC most common extracolonic in MLH1; "
            "  Right colon predominance (70% proximal to splenic flexure); "
            "AMSTERDAM II criteria: >=3 LS cancers in >=2 generations; 1st degree; 1 cancer < 50 years; "
            "Revised Bethesda criteria: MSI testing trigger (age <50; synchronous/metachronous Lynch tumour; "
            "  1st-degree CRC/endometrial <50; MSI-H pathology); "
            "CRITICAL FIRST STEP: MLH1 promoter methylation test on tumour tissue -- "
            "  Methylation positive + no germline MLH1 = sporadic; do NOT diagnose Lynch; "
            "  Methylation negative + IHC MLH1 loss = strongly suggests germline MLH1; "
            "  BRAF V600E somatic test (rapid): if positive = sporadic MSI-H (not Lynch); "
            "Pembrolizumab (Keytruda) FDA2017: first tissue-agnostic approval for dMMR/MSI-H tumours; "
            "Adjuvant: FOLFOX for stage III CRC; pembrolizumab replacing FOLFOX in stage III dMMR clinical trials; "
            "Immunotherapy response: dMMR tumours respond to PD-1 blockade due to high mutational burden + neoantigen load."
        ),
        "inheritance": (
            "AD LOF 3p22.2 -- Lynch Syndrome 1 (LS1). Penetrance 40-70% CRC lifetime. "
            "PMS2 stabilised by MLH1 -- IHC shows concomitant MLH1+PMS2 loss when MLH1 variant. "
            "MLH1 somatic methylation must be excluded (tumour methylation + BRAF V600E) before germline testing. "
            "Founder mutations: Finnish p.Ala681Thr; Dutch c.306+5G>A (splice); Ashkenazi IVS14-1G>A."
        ),
        "disease_category": "Lynch Syndrome / Mismatch Repair Deficiency",
        "patient_generator_params": {
            "crc_risk": 0.62, "endometrial_risk": 0.45, "ovarian_risk": 0.12,
            "gastric_risk": 0.13, "urinary_tract_risk": 0.11, "small_bowel_risk": 0.06,
            "msi_h_pct": 0.95, "synchronous_crc_risk": 0.10,
            "age_range": (28, 72), "mean_age_dx": 44,
            "severity_dist": {"severe": 0.30, "moderate": 0.50, "mild": 0.20},
        },
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- MutS-Homologue-2-MMR-MutS-alpha-"
            "Lynch-Syndrome-2-31pct-Lynch-SEBACEOUS-NEOPLASMS-Muir-Torre-PATHOGNOMONIC-"
            "EPCAM-Deletion-MSH2-Silencing-OMIM-609310"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 105 kDa (MSH2; MutS alpha complex with MSH6; also MutS beta with MSH3; "
            "STRUCTURE: N-terminal mismatch-binding domain; ATP-binding cassette; C-terminal dimerisation; "
            "MSH2-MSH6 (MutS alpha): recognises base/base mismatches + small insertions/deletions; "
            "MSH2-MSH3 (MutS beta): recognises larger insertion/deletion loops; "
            "FUNCTION: "
            "  MSH2 scaffold: MSH6 unstable without MSH2 (concomitant MSH2+MSH6 IHC loss in MSH2-LOF); "
            "  Primary mismatch sensor before recruiting MutL alpha (MLH1-PMS2) for incision; "
            "LYNCH SYNDROME 2 (LS2): "
            "  Second most common Lynch gene: ~31% of Lynch families; "
            "  CRC lifetime risk: 45-70%; mean age diagnosis 41-43 years; "
            "  Endometrial: 40-60% lifetime; "
            "  Extracolonic spectrum: urinary tract (transitional cell carcinoma), stomach, ovarian, small bowel; "
            "MUIR-TORRE SYNDROME (MTS): "
            "  Lynch variant with SEBACEOUS NEOPLASMS (sebaceous adenoma, sebaceoma, sebaceous carcinoma) PATHOGNOMONIC; "
            "  Sebaceous tumours on face/trunk; sebaceous carcinoma can be periocular (eyelid); "
            "  MSH2 most common (>50% Muir-Torre), MLH1 also; "
            "  Visceral malignancy concurrent; BRAF V600E negative; "
            "EPCAM (TACSTD1) DELETION -- MSH2 SILENCING: "
            "  3' EPCAM deletion removes EPCAM stop codon -> read-through into MSH2 promoter -> methylation silences MSH2; "
            "  EPCAM deletions account for ~1-3% of Lynch -- standard MSH2 sequencing MISSES this; "
            "  IHC: MSH2 loss pattern identical to MSH2 point mutation; "
            "  Testing: MSH2 negative + MSH2 sequencing/MLPA negative -> test EPCAM MLPA; "
            "TURCOT SYNDROME (VARIANT 1): "
            "  Lynch (MLH1/MSH2) + CNS glioblastoma; distinguish from FAP-Turcot (APC + medulloblastoma). "
            "IHC pattern: concomitant MSH2 + MSH6 loss (MSH6 destabilised without MSH2 scaffold)."
        ),
        "inheritance": (
            "AD LOF 2p21 -- Lynch Syndrome 2 (LS2). EPCAM 3' deletion must be tested if MSH2 sequencing/MLPA negative. "
            "MSH6 co-lost on IHC (MSH6 requires MSH2). Sebaceous neoplasm on ANY site = test for MSH2/MLH1 Lynch. "
            "Turcot variant: glioblastoma multiforme (not medulloblastoma like FAP-Turcot)."
        ),
        "disease_category": "Lynch Syndrome / Mismatch Repair Deficiency / Muir-Torre",
        "patient_generator_params": {
            "crc_risk": 0.60, "endometrial_risk": 0.50, "ovarian_risk": 0.14,
            "gastric_risk": 0.14, "urinary_tract_risk": 0.18, "small_bowel_risk": 0.07,
            "msi_h_pct": 0.97, "synchronous_crc_risk": 0.09,
            "sebaceous_neoplasm": 0.12, "glioblastoma": 0.05,
            "age_range": (26, 70), "mean_age_dx": 42,
            "severity_dist": {"severe": 0.32, "moderate": 0.50, "mild": 0.18},
        },
    },
    {
        "gene": "MSH6",
        "protein": (
            "MSH6 -- 2p16.3 Autosomal-Dominant-LOF -- 1360aa -- MutS-Homologue-6-MMR-"
            "Lynch-Syndrome-5-18pct-Lynch-ENDOMETRIAL-PREDOMINANT-Females-Reduced-CRC-Penetrance-"
            "MSI-L-MSS-Possible-OMIM-600678"
        ),
        "locus": "2p16.3",
        "protein_size": (
            "1360 aa / 160 kDa (MSH6; MutS alpha complex with MSH2; "
            "STRUCTURE: N-terminal PCNA-interacting PIP box; conserved mismatch-binding domain; "
            "ATPase/NBD domain for conformational change; C-terminal dimerisation with MSH2; "
            "FUNCTION: "
            "  MutS alpha (MSH2-MSH6): recognises base/base mismatches + 1-bp insertions; "
            "  MSH6-specific: critical for G/T mismatch repair (C to A transversions common in MSH6-deficient cells); "
            "  MSH6 unique to MutS alpha (not in MutS beta); "
            "LYNCH SYNDROME 5 (LS5): "
            "  Third most common Lynch gene: ~18% of Lynch families; "
            "  ENDOMETRIAL PREDOMINANCE IN FEMALES: "
            "    Endometrial cancer lifetime risk: 40-70% (HIGHER THAN MSH2 in females); "
            "    CRC lifetime risk: 10-25% (significantly LOWER than MLH1/MSH2); "
            "    Mean age of CRC diagnosis: ~55 years (later than MLH1/MSH2, closer to sporadic age); "
            "  MSH6 ovarian cancer risk: ~10-15% lifetime; "
            "MSI STATUS IN MSH6: "
            "  MSI testing (BAT26, BAT25 etc.): may show MSI-L (low) or MSS (microsatellite stable) in MSH6; "
            "  Reason: MSH6 repairs base/base mismatches but MSH2-MSH3 (intact) still repairs frameshift loops; "
            "  IMPLICATION: MSI-NEGATIVE CRC does NOT exclude MSH6 Lynch -- IHC mandatory if clinical suspicion; "
            "  Revised Bethesda criteria miss MSH6 patients due to late age + less MSI-H; "
            "IHC PATTERN: isolated MSH6 loss (MSH2 RETAINED -- MSH2 forms MutS beta backup with MSH3); "
            "  IHC MSH6 loss + MSH2 retained = MSH6 mutation CONFIRMED unless concurrent somatic methylation; "
            "Aspirin chemoprevention: CAPP2 trial -- aspirin 600mg reduces CRC in Lynch 2-fold (significant in MLH1/MSH2, "
            "  trend in MSH6); CAPP3 dose-finding ongoing. "
            "Colonoscopy surveillance: every 2 years from age 25 (ESGE/NICE: every 1-2yr from 25 for MSH6). "
        ),
        "inheritance": (
            "AD LOF 2p16.3 -- Lynch Syndrome 5. REDUCED CRC PENETRANCE -- may be dismissed as low risk. "
            "Endometrial > CRC in females. MSI testing unreliable for MSH6 -- always do IHC. "
            "Isolated MSH6 loss on IHC (MSH2 intact): MSH6 variant confirmed."
        ),
        "disease_category": "Lynch Syndrome / Mismatch Repair Deficiency",
        "patient_generator_params": {
            "crc_risk": 0.20, "endometrial_risk": 0.58, "ovarian_risk": 0.13,
            "gastric_risk": 0.07, "urinary_tract_risk": 0.09, "small_bowel_risk": 0.03,
            "msi_h_pct": 0.60, "synchronous_crc_risk": 0.05,
            "age_range": (35, 78), "mean_age_dx": 55,
            "severity_dist": {"severe": 0.20, "moderate": 0.52, "mild": 0.28},
        },
    },
    {
        "gene": "PMS2",
        "protein": (
            "PMS2 -- 7p22.1 Autosomal-Dominant-LOF -- 862aa -- PMS1-Homologue-2-MMR-"
            "Lynch-Syndrome-4-14pct-Lynch-LOWEST-Penetrance-Biallelic-CMMRD-Childhood-CNS-"
            "Pseudogene-PMS2CL-Long-Range-PCR-Required-OMIM-600259"
        ),
        "locus": "7p22.1",
        "protein_size": (
            "862 aa / 96 kDa (PMS2; MutL alpha with MLH1; "
            "STRUCTURE: N-terminal ATPase domain; endonuclease active site (metal-binding DQHA motif); "
            "C-terminal MLH1-binding region (MIP box); "
            "Pseudogene PMS2CL on 7p: 9 exons highly similar to PMS2 exons 12-15 -- causes NGS misalignment; "
            "FUNCTION: "
            "  MutL alpha (MLH1-PMS2): endonuclease incises displaced strand containing mismatch; "
            "  PMS2 carries the DQHA(X)2E(X)4E latent endonuclease motif activated by MutS sliding clamp; "
            "  PMS2 destabilised without MLH1 (PMS2 absent on IHC in MLH1 variants); "
            "LYNCH SYNDROME 4 (LS4): "
            "  Least common MMR Lynch gene: ~14% of Lynch; "
            "  LOWEST CRC penetrance of all MMR genes: CRC lifetime risk 15-20% (40-70% for MLH1/MSH2); "
            "  LOWEST endometrial penetrance: ~15% lifetime; "
            "  IHC PATTERN: ISOLATED PMS2 LOSS + MLH1 RETAINED (PMS2 is unstable without MLH1, "
            "    but MLH1 uses other partners when PMS2 absent -- MLH1 retained in PMS2 variants); "
            "    ISOLATED PMS2 LOSS = PMS2 GERMLINE UNTIL PROVEN OTHERWISE; "
            "BIALLELIC PMS2 (CMMRD) -- Constitutional MMR Deficiency: "
            "  Both PMS2 alleles lost in childhood (biallelic pathogenic variants); "
            "  CHILDHOOD MALIGNANCIES: brain (glioblastoma, glioma), colorectal polyps/cancer, haematological (lymphoma, leukaemia); "
            "  Cafe-au-lait macules: multiple (can mimic NF1, distinguish by absence of Lisch nodules/subcutaneous neurofibromas); "
            "  Consanguineous families; parents are obligate Lynch carriers (monoallelic); "
            "  TREATMENT: pembrolizumab for CMMRD cancers (MSI-H/dMMR); very high response rate; "
            "PMS2 TESTING CHALLENGE: "
            "  Pseudogene PMS2CL (7p13.3): 99% homologous to PMS2 exons 12-15; "
            "  Standard short-read NGS frequently misaligns to pseudogene -> false negatives exons 12-15; "
            "  SOLUTION: long-range PCR (LR-PCR) + Sanger sequencing exons 11-15; "
            "  MLPA: PMS2 MLPA probe set (P008) distinguishes PMS2 from PMS2CL; "
            "  Do NOT rely on standard NGS panel alone for PMS2 exons 12-15 coverage verification."
        ),
        "inheritance": (
            "AD LOF 7p22.1 -- Lynch Syndrome 4. LOWEST MMR penetrance. Isolated PMS2 IHC loss (MLH1 retained). "
            "Biallelic = CMMRD (childhood brain/haematologic/GI cancers; cafe-au-lait spots). "
            "Standard NGS MISSES exons 12-15 (pseudogene PMS2CL) -- LR-PCR required."
        ),
        "disease_category": "Lynch Syndrome / CMMRD (biallelic) / Mismatch Repair Deficiency",
        "patient_generator_params": {
            "crc_risk": 0.18, "endometrial_risk": 0.16, "ovarian_risk": 0.09,
            "gastric_risk": 0.06, "urinary_tract_risk": 0.08, "small_bowel_risk": 0.03,
            "msi_h_pct": 0.75, "synchronous_crc_risk": 0.04,
            "cmmrd_biallelic": 0.10, "cafe_au_lait": 0.18,
            "age_range": (28, 80), "mean_age_dx": 57,
            "severity_dist": {"severe": 0.18, "moderate": 0.52, "mild": 0.30},
        },
    },
    {
        "gene": "APC",
        "protein": (
            "APC -- 5q22.2 Autosomal-Dominant-LOF -- 2843aa -- Adenomatous-Polyposis-Coli-"
            "Wnt-Signalling-Tumour-Suppressor-FAP-AFAP-Gardner-Turcot-"
            "GT100-POLYPS-PATHOGNOMONIC-CRC-100pct-Untreated-Prophylactic-Colectomy-Mandatory-"
            "OMIM-611731"
        ),
        "locus": "5q22.2",
        "protein_size": (
            "2843 aa / 311 kDa (APC; tumour suppressor; Wnt pathway scaffold; "
            "STRUCTURE: N-terminal armadillo repeats (Axin-binding); SAMP repeats; "
            "central region: 20-aa repeats (beta-catenin binding, phosphorylation targets); "
            "Mutation Cluster Region (MCR, codons 1250-1464): hotspot for classic FAP; "
            "C-terminal: EB1/EB3 interaction (microtubule capture); PDZ-binding motif; "
            "FUNCTION: "
            "  Wnt destruction complex: APC-Axin-CK1-GSK3beta phosphorylates beta-catenin for ubiquitination; "
            "  LOF -> beta-catenin accumulates -> nuclear TCF/LEF -> oncogene transcription (c-MYC, cyclin D1); "
            "  Microtubule stabilisation at kinetochore: APC LOF -> chromosomal instability (CIN); "
            "FAMILIAL ADENOMATOUS POLYPOSIS (FAP): "
            "  CLASSIC FAP: >100 colorectal adenomatous polyps (mean 1000); onset ~16 years; "
            "  CRC LIFETIME RISK 100% IF UNTREATED by age 40 years; "
            "  PROPHYLACTIC COLECTOMY MANDATORY: timing when polyp burden uncontrollable or age 25-35; "
            "  Options: total colectomy + IRA (ileorectostomy) or proctocolectomy + IPAA (pouch); "
            "  IRA preferred if rectal sparing possible (codon-dependent rectal severity); "
            "ATTENUATED FAP (AFAP): "
            "  <100 polyps; codons 1-163 or 3' end (>1580) or exon 9 splice; CRC risk ~70%; "
            "  Later onset (30s-40s); can mimic MUTYH-MAP; colonoscopy surveillance may suffice; "
            "EXTRACOLONIC FEATURES: "
            "  Duodenal/periampullary adenomas: 90% of FAP; Spigelman staging I-IV governs management; "
            "    Spigelman IV or grade 3 dysplasia: prophylactic pancreaticoduodenectomy or ablation; "
            "  CONGENITAL HYPERTROPHY OF RPE (CHRPE): bilateral CHRPE (>=4 lesions) 70-80% FAP PATHOGNOMONIC; "
            "  DESMOID TUMOURS: 10-15% FAP; INTRA-ABDOMINAL desmoids post-colectomy (codon 1310-2011 highest risk); "
            "    Desmoid unpredictable: can obstruct bowel/ureters; "
            "    Treatment: sulindac + tamoxifen first; sorafenib/other TKI; surgical high morbidity; "
            "  GARDNER SYNDROME: FAP + epidermoid cysts + osteomas (mandible) + supernumerary teeth + CHRPE; "
            "  TURCOT SYNDROME (variant 2): FAP + CNS medulloblastoma (cerebellar); "
            "    Distinguish from Lynch-Turcot (glioblastoma); "
            "  GASTRIC POLYPS: fundic gland polyps (80% FAP; low malignant risk); "
            "ASPIRIN / NSAID: "
            "  Sulindac (NSAID): reduces polyp count (not CRC substitute -- must still do colectomy); "
            "  Celecoxib (COX-2 inhibitor): reduces duodenal + rectal polyp burden; "
            "  Aspirin 600mg: CAPP2 trial Lynch + FAP adjuvant; "
            "Genotype-phenotype: "
            "  Codon 1250-1464 (MCR): severe classic FAP + desmoid + CHRPE; "
            "  Codons 1310-2011: highest desmoid risk; "
            "  5' codons 1-163 / exon 9 splice / 3' >1580: AFAP."
        ),
        "inheritance": (
            "AD LOF 5q22.2 -- FAP/AFAP. CRC 100% lifetime if untreated (classic FAP). "
            "PROPHYLACTIC COLECTOMY MANDATORY. Duodenal surveillance with Spigelman staging. "
            "Desmoid risk highest codons 1310-2011. CHRPE bilateral >=4 lesions pathognomonic. "
            "AFAP (codons 1-163 or >1580): <100 polyps, later age, can mimic MAP."
        ),
        "disease_category": "Familial Adenomatous Polyposis / Hereditary Colorectal Cancer",
        "patient_generator_params": {
            "crc_risk": 0.95, "polyp_count_gt100": 0.80, "duodenal_adenoma": 0.90,
            "desmoid": 0.14, "chrpe": 0.75, "osteoma": 0.20, "sebaceous_cyst": 0.35,
            "gastric_fundic_polyps": 0.80, "thyroid_cancer": 0.02,
            "colectomy_done": 0.75,
            "age_range": (18, 65), "mean_age_dx": 32,
            "severity_dist": {"severe": 0.55, "moderate": 0.35, "mild": 0.10},
        },
    },
    {
        "gene": "MUTYH",
        "protein": (
            "MUTYH -- 1p34.1 Autosomal-Recessive -- 546aa -- MutY-DNA-Glycosylase-"
            "Base-Excision-Repair-MAP-MUTYH-Associated-Polyposis-10-100-ADENOMAS-BIALLELIC-"
            "CRC-80pct-Y179C-G396D-European-Founders-Heterozygous-NOT-Lynch-Level-OMIM-604933"
        ),
        "locus": "1p34.1",
        "protein_size": (
            "546 aa / 60 kDa (MUTYH; base excision repair glycosylase; "
            "STRUCTURE: N-terminal targeting sequence (mitochondria + nucleus); "
            "HhH-GPD superfamily DNA glycosylase domain; PCNA-interacting C-terminal; "
            "FUNCTION: "
            "  Base excision repair: removes adenine mispaired opposite 8-oxo-7,8-dihydroguanine (8-oxoG); "
            "  8-oxoG = most common oxidative DNA lesion; paired with A -> G:C transversion if uncorrected; "
            "  MUTYH LOF -> accumulation of G>T transversions (characteristic MUTYH-MAP mutational signature); "
            "  Mutational signature: G>T transversions in APC, KRAS (c.34G>T, p.Gly12Cys) PATHOGNOMONIC in MAP; "
            "MAP (MUTYH-ASSOCIATED POLYPOSIS): "
            "  BIALLELIC pathogenic variants required (AR inheritance unlike Lynch); "
            "  10-100 colorectal adenomatous polyps (overlap with AFAP); "
            "  CRC LIFETIME RISK: 80% by age 70 biallelic; "
            "  Onset age: typically 45-55 years (later than classic FAP); "
            "FOUNDER VARIANTS (European): "
            "  Y179C (c.536A>G) [formerly Y165C]: ~36% of MAP alleles in Europeans; "
            "  G396D (c.1187G>A) [formerly G382D]: ~37% of MAP alleles in Europeans; "
            "  Compound heterozygous Y179C/G396D most common European MAP; "
            "  Other populations: different variants (South Asian: p.Glu480del; Japanese: specific variants); "
            "HETEROZYGOUS MUTYH CARRIERS: "
            "  ~1.5-2x elevated CRC risk vs general population (NOT Lynch-equivalent ~40-70%); "
            "  DO NOT treat monoallelic MUTYH as Lynch-level risk -- risk counselling critical; "
            "  Standard colonoscopy surveillance same as average-risk but starting age 40; "
            "EXTRACOLONIC: "
            "  Duodenal adenomas (MAP): 4-17% (less than FAP); "
            "  CHRPE: uncommon in MAP; "
            "  Sebaceous gland tumours: rare but reported; "
            "MANAGEMENT: "
            "  Biallelic MUTYH: colonoscopy every 1-2 years from age 18-25; "
            "  If polyp burden uncontrollable (>30-50 polyps, high-grade dysplasia): consider colectomy; "
            "  SOMATIC APC MUTATIONS in MAP tumours tend to be G:C > T:A transversions (MUTYH signature). "
        ),
        "inheritance": (
            "AR (autosomal recessive) 1p34.1 -- MAP requires BIALLELIC variants. "
            "Y179C + G396D European founders (>70% of European MAP alleles). "
            "HETEROZYGOUS: 1.5-2x CRC risk only (NOT Lynch-level). "
            "Distinguish from AFAP (APC): AR vs AD; G>T transversion signature in MUTYH."
        ),
        "disease_category": "MUTYH-Associated Polyposis / Hereditary Colorectal Cancer (AR)",
        "patient_generator_params": {
            "crc_risk": 0.75, "polyp_count_10_100": 0.85, "duodenal_adenoma": 0.12,
            "biallelic_confirmed": 0.90, "y179c_allele": 0.70, "g396d_allele": 0.70,
            "age_range": (35, 72), "mean_age_dx": 50,
            "severity_dist": {"severe": 0.35, "moderate": 0.48, "mild": 0.17},
        },
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- Serine-Threonine-Kinase-11-"
            "Peutz-Jeghers-Syndrome-MUCOCUTANEOUS-LENTIGINES-PERIORAL-PATHOGNOMONIC-"
            "HAMARTOMATOUS-POLYPS-CANCER-93pct-Lifetime-PANCREATIC-132x-Relative-Risk-OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 50 kDa (STK11/LKB1; serine/threonine kinase; AMPK pathway master regulator; "
            "STRUCTURE: N-terminal domain; central kinase domain (Ser/Thr kinase active site); "
            "C-terminal domain (nuclear/cytoplasmic localisation); farnesylation site; "
            "FUNCTION: "
            "  Master upstream kinase of AMPK (AMP-activated protein kinase): "
            "    STK11 activates AMPK by phosphorylating Thr172 under energy stress (low ATP); "
            "    AMPK -> TSC1/TSC2 -> inhibits mTORC1 (STK11 = indirect mTOR suppressor); "
            "  p53-mediated apoptosis regulation; DNA damage response; mitotic spindle; "
            "  Epithelial polarity regulator (important for intestinal epithelial homeostasis); "
            "  Somatic STK11 loss in sporadic lung adenocarcinoma (KRAS-co-mutation common); "
            "PEUTZ-JEGHERS SYNDROME (PJS): "
            "  MUCOCUTANEOUS LENTIGINES (pigmented spots): "
            "    PERIORAL mucosa (buccal/labial) + facial skin AROUND mouth: PATHOGNOMONIC; "
            "    Digits (fingers/toes); perianal; "
            "    APPEAR IN CHILDHOOD/INFANCY (often birth-2 years); "
            "    FADE AFTER PUBERTY -- absence in adult does NOT exclude PJS; document by parent history; "
            "    Buccal mucosal lentigines PERSIST (do NOT fade -- check buccal mucosa even in adults); "
            "  HAMARTOMATOUS POLYPS: "
            "    Small bowel PREDOMINANT (jejunum most common): heterogeneous composition (arborising smooth muscle + epithelium); "
            "    Also in stomach, colon, rectum; "
            "    INTUSSUSCEPTION risk: jejuno-jejunal intussusception is major morbidity in childhood -- emergency surgery; "
            "    Capsule endoscopy / MR enteroclysis: polyp size-based surveillance (>1.5 cm -> intervention); "
            "CANCER RISKS IN PJS (lifetime, age 70): "
            "  All cancers combined: ~93% lifetime (Giardiello 2000 + Hearle 2006 meta-analysis); "
            "  CRC: 39% lifetime; "
            "  Small bowel: 13% lifetime; "
            "  GASTRIC: 29% lifetime; "
            "  PANCREATIC: 11-36% lifetime (RELATIVE RISK 132x GENERAL POPULATION -- highest single-gene pancreatic risk); "
            "  BREAST: 32-54% lifetime (surveillance mammography + MRI from age 25); "
            "  OVARIAN: 21% (sex cord tumour with annular tubules SCTAT 36%, bilateral, benign majority; "
            "    cervical adenocarcinoma minimal deviation (MDC/adenoma malignum) PATHOGNOMONIC female PJS; "
            "  LUNG: 7-17%; "
            "  Testicular (male): Sertoli cell tumours large cell calcifying (LCCSCT); "
            "MANAGEMENT: "
            "  Enteroscopy + colonoscopy every 2-3 years from age 8-10; "
            "  Gastroscopy every 2-3 years from age 8; "
            "  EUS/MRI abdomen for pancreatic surveillance from age 30-35 annually; "
            "  Mammography + breast MRI from age 25 (breast high risk); "
            "  Annual physical exam for gynaecological surveillance; "
            "  Intraoperative enteroscopy at laparotomy for intussusception to clear all polyps simultaneously."
        ),
        "inheritance": (
            "AD LOF 19p13.3 -- Peutz-Jeghers Syndrome. PERIORAL LENTIGINES PATHOGNOMONIC. "
            "Lentigines fade after puberty -- buccal mucosa persists. "
            "PANCREATIC CANCER RISK 132x general population -- EUS/MRI annually from age 30-35. "
            "Cervical MDC/adenoma malignum PATHOGNOMONIC female PJS (NOT on standard Pap smear -- MRI pelvis)."
        ),
        "disease_category": "Peutz-Jeghers Syndrome / Hamartomatous Polyposis",
        "patient_generator_params": {
            "crc_risk": 0.39, "small_bowel_cancer": 0.13, "gastric_cancer": 0.29,
            "pancreatic_cancer": 0.22, "breast_cancer": 0.40, "ovarian_cancer": 0.21,
            "lentigines_perioral": 0.92, "hamartomatous_polyps": 0.98,
            "intussusception_hx": 0.35,
            "age_range": (12, 70), "mean_age_dx": 38,
            "severity_dist": {"severe": 0.42, "moderate": 0.42, "mild": 0.16},
        },
    },
    {
        "gene": "SMAD4",
        "protein": (
            "SMAD4 -- 18q21.2 Autosomal-Dominant-LOF -- 552aa -- SMAD-Family-Member-4-"
            "TGFbeta-BMP-Signal-Transducer-Juvenile-Polyposis-Syndrome-JPS-"
            "JUVENILE-POLYPS-SMOOTH-MUCOSA-PATHOGNOMONIC-JPS-HHT-OVERLAP-22pct-"
            "CRC-40pct-SMAD4-IHC-Loss-OMIM-600993"
        ),
        "locus": "18q21.2",
        "protein_size": (
            "552 aa / 60 kDa (SMAD4/DPC4; Deleted in Pancreatic Cancer locus 4; "
            "STRUCTURE: N-terminal MH1 domain (DNA binding, Smad-binding element SBE recognition); "
            "Central linker region (phosphorylation, ubiquitination); "
            "C-terminal MH2 domain (SMAD-SMAD interaction, nuclear translocation); "
            "SMAD4 = COMMON MEDIATOR SMAD (co-SMAD): required for both TGF-β and BMP pathways; "
            "FUNCTION: "
            "  TGF-β pathway: TGF-βR2/R1 -> phospho-SMAD2/3 -> SMAD4 complex -> nucleus -> growth suppression; "
            "  BMP pathway: BMPRII/I -> phospho-SMAD1/5/8 -> SMAD4 complex -> nucleus -> vascular development; "
            "  Tumour suppressor: SMAD4 LOF -> TGF-β growth inhibition lost; "
            "  SOMATIC SMAD4 loss: >50% of pancreatic ductal adenocarcinoma; poor prognosis marker; "
            "JUVENILE POLYPOSIS SYNDROME (JPS): "
            "  JUVENILE POLYPS (hamartomatous NOT adenomatous): "
            "    PATHOGNOMONIC histology: smooth outer mucosa; long pedunculated stalk; "
            "    EXPANDED LAMINA PROPRIA with inflammatory cells, dilated glands (different from Peutz-Jeghers arborising smooth muscle); "
            "    Primarily colorectal (most common); also gastric (more severe in SMAD4); "
            "  Clinical diagnosis: >=5 juvenile polyps colorectum; OR juvenile polyps throughout GI tract; "
            "    OR any juvenile polyps + family history JPS; "
            "  SMAD4 vs BMPR1A: "
            "    SMAD4: causes 15-20% JPS; GASTRIC POLYPS MORE SEVERE (may require gastrectomy for uncontrollable gastric disease); "
            "    BMPR1A: causes 25-40% JPS; fewer gastric polyps; "
            "  CRC LIFETIME RISK: 40% by age 60; "
            "  GASTRIC CANCER: ~20% lifetime in SMAD4-JPS (higher than BMPR1A-JPS); "
            "JPS-HHT OVERLAP (SMAD4-SPECIFIC): "
            "  22% of SMAD4-JPS patients have HHT (hereditary haemorrhagic telangiectasia) features: "
            "  EPISTAXIS (nosebleeds): most common HHT symptom; "
            "  TELANGIECTASIA: lips, tongue, fingers; "
            "  PULMONARY AVM: 25-40% SMAD4-JPS -- MANDATORY ECHO BUBBLE STUDY at diagnosis; "
            "    Paradoxical embolism: stroke + brain abscess risk from right-to-left shunt via PAVM; "
            "  HEPATIC AVM: liver bruit; high-output cardiac failure; "
            "  CEREBRAL AVM: haemorrhagic stroke risk (screen with MRI brain); "
            "  SMAD4 IHC: loss of SMAD4 protein in polyps/tumour = SMAD4 variant confirmed; "
            "  SMAD4 somatic loss in sporadic CRC: do NOT report as Lynch; IHC loss in tumour ≠ germline; "
            "MANAGEMENT: "
            "  Annual colonoscopy from age 15 (or 5 years younger than youngest family cancer); "
            "  Upper GI endoscopy annually (gastric polyps in SMAD4); "
            "  Echo bubble study + cardiac echo at diagnosis for PAVM (SMAD4-JPS); "
            "  CT pulmonary angiography if echo positive for PAVM; "
            "  MRI brain for cerebral AVM; "
            "  Prophylactic colectomy when polyp burden uncontrolled (>50 polyps, HGD); "
            "  Prophylactic gastrectomy: consider in SMAD4-JPS with severe gastric disease."
        ),
        "inheritance": (
            "AD LOF 18q21.2 -- Juvenile Polyposis Syndrome (SMAD4). "
            "JPS-HHT OVERLAP in 22% -- ALWAYS do echo bubble study for PAVM at diagnosis. "
            "PAVM paradoxical embolism causes stroke/brain abscess. "
            "SMAD4 IHC loss in tumour confirms SMAD4 variant; GASTRIC polyps more severe than BMPR1A-JPS. "
            "Juvenile polyp histology (smooth mucosa, stalk, expanded lamina propria) distinguishes from FAP/MAP adenomas."
        ),
        "disease_category": "Juvenile Polyposis Syndrome / JPS-HHT Overlap",
        "patient_generator_params": {
            "crc_risk": 0.40, "gastric_cancer": 0.20, "small_bowel_risk": 0.06,
            "juvenile_polyps": 0.98, "gastric_polyps": 0.65, "hht_overlap": 0.22,
            "pavm": 0.28, "epistaxis": 0.40, "telangiectasia": 0.35,
            "age_range": (12, 68), "mean_age_dx": 34,
            "severity_dist": {"severe": 0.38, "moderate": 0.46, "mild": 0.16},
        },
    },
]


# ─── Patient generators ────────────────────────────────────────────────────────

def _generate_patients_for_gene(gene: str, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    params = next(g["patient_generator_params"] for g in ATLAS_GENES if g["gene"] == gene)

    mutations = {
        "MLH1":  ["c.1852_1853delinsGG (p.Lys618Glu)", "c.676C>T (p.Arg226*)", "c.2041G>A (p.Ala681Thr Finn founder)",
                   "c.677G>A (p.Arg226Gln)", "c.117G>A (p.Trp39*)", "c.1731+1G>A (splice)"],
        "MSH2":  ["c.942+3A>T (splice)", "c.1276G>T (p.Val426Leu)", "c.388_389del (p.Pro130fs)",
                   "EPCAM exon5-8 del (MSH2 methylation)", "c.1861C>T (p.Arg621*)", "c.2635-1G>T (splice)"],
        "MSH6":  ["c.3261dupC (p.Phe1088Leufs)", "c.3439-2A>G (splice)", "c.1168G>T (p.Glu390*)",
                   "c.3959_3962delCAAG", "c.2476A>T (p.Ile826Phe)", "c.116G>A (p.Trp39*)"],
        "PMS2":  ["c.2174+1G>A (splice, exon 12 region)", "c.989-1G>A (splice)", "c.137G>T (p.Ser46Ile)",
                   "c.1831dup (p.Met611Ilefs)", "c.215-2A>G (splice)", "c.886A>T (p.Lys296*)"],
        "APC":   ["c.3927_3931delAAAGA (p.Glu1309fs) MCR", "c.4348C>T (p.Arg1450*)", "c.3183_3187del",
                   "c.1A>T (p.Met1?)", "c.3927_3931del5 (p.Glu1309Aspfs)", "c.4666dup (p.Met1556Ilefs)"],
        "MUTYH": ["c.536A>G (p.Tyr179Cys) founder", "c.1187G>A (p.Gly396Asp) founder",
                   "c.1145G>A (p.Gly382Asp)", "c.928G>A (p.Gly310Ser)", "c.63_64insTATA (p.Asp22fs)", "c.1227_1228dup"],
        "STK11": ["c.1062delA (p.Glu355Aspfs)", "c.962+3_962+6del (splice)", "c.580C>T (p.Arg194*)",
                   "Large exon 1-10 del", "c.921G>A (p.Glu307=, splicing)", "c.290del (p.Gly97Alafs)"],
        "SMAD4": ["c.1162C>T (p.Arg388Cys)", "c.1081C>T (p.Arg361Cys)", "large deletion exon 8-11",
                   "c.748C>T (p.Arg250*)", "c.1546del (p.Trp516Cysfs)", "c.1244T>G (p.Met415Arg)"],
    }.get(gene, ["Pathogenic variant"])

    patients = []
    severity_dist = params["severity_dist"]
    sev_choices = ["severe"] * int(40 * severity_dist["severe"]) + \
                  ["moderate"] * int(40 * severity_dist["moderate"]) + \
                  ["mild"] * int(40 * severity_dist.get("mild", 0))
    while len(sev_choices) < 40:
        sev_choices.append("moderate")

    age_lo, age_hi = params.get("age_range", (25, 75))

    for i in range(n):
        age_dx = int(rng.gauss(params.get("mean_age_dx", 45), 12))
        age_dx = max(age_lo, min(age_hi, age_dx))
        sev = sev_choices[i]

        # MMR-specific fields
        crc_present = rng.random() < params.get("crc_risk", 0.3)
        endometrial = rng.random() < params.get("endometrial_risk", 0.1)
        msi_h = rng.random() < params.get("msi_h_pct", 0.7)
        synch_crc = rng.random() < params.get("synchronous_crc_risk", 0.05)

        # Polyposis-specific
        adenomas_gt100 = rng.random() < params.get("polyp_count_gt100", 0.0)
        duodenal = rng.random() < params.get("duodenal_adenoma", 0.0)
        desmoid = rng.random() < params.get("desmoid", 0.0)
        chrpe = rng.random() < params.get("chrpe", 0.0)
        colectomy = rng.random() < params.get("colectomy_done", 0.0)

        # PJS-specific
        lentigines = rng.random() < params.get("lentigines_perioral", 0.0)
        intussusception = rng.random() < params.get("intussusception_hx", 0.0)
        pancreatic_ca = rng.random() < params.get("pancreatic_cancer", 0.0)
        breast_ca = rng.random() < params.get("breast_cancer", 0.0)

        # JPS-specific
        juvenile_polyps = rng.random() < params.get("juvenile_polyps", 0.0)
        hht_features = rng.random() < params.get("hht_overlap", 0.0)
        pavm = rng.random() < params.get("pavm", 0.0)

        # MUTYH
        biallelic = rng.random() < params.get("biallelic_confirmed", 0.0)

        patients.append({
            "id": f"{gene}-{i+1:02d}",
            "gene": gene,
            "age_at_diagnosis_yrs": age_dx,
            "mutation": rng.choice(mutations),
            "severity": sev,
            "crc_present": crc_present,
            "endometrial_cancer": endometrial,
            "msi_high": msi_h,
            "synchronous_crc": synch_crc,
            "adenomas_gt100": adenomas_gt100,
            "duodenal_adenoma": duodenal,
            "desmoid_tumour": desmoid,
            "chrpe": chrpe,
            "colectomy_done": colectomy,
            "lentigines_perioral": lentigines,
            "intussusception_history": intussusception,
            "pancreatic_cancer": pancreatic_ca,
            "breast_cancer": breast_ca,
            "juvenile_polyps_present": juvenile_polyps,
            "hht_features": hht_features,
            "pulmonary_avm": pavm,
            "biallelic_mutyh": biallelic,
            "surveillance_endoscopy_current": rng.choice([True, True, True, False]),
        })
    return patients


# ─── API generators ────────────────────────────────────────────────────────────

def generate_overview() -> dict:
    """Overview data for Hereditary-CRC-Polyposis-Atlas."""
    return {
        "atlas":          "Hereditary-CRC-Polyposis-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Colorectal Cancer & Polyposis Syndrome Atlas "
            "(MLH1-MSH2-MSH6-PMS2-APC-MUTYH-STK11-SMAD4)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "MLH1": (
                "AD LOF 3p22.2 (MutL alpha scaffold; 852aa; Lynch Syndrome 1; most common 35-40% Lynch; "
                "CRC 40-70% lifetime; endometrial 30-60%; "
                "MLH1 PROMOTER METHYLATION somatic exclusion MANDATORY before Lynch diagnosis; "
                "BRAF V600E somatic = sporadic MSI-H; dMMR = pembrolizumab FDA2017)"
            ),
            "MSH2": (
                "AD LOF 2p21 (MutS alpha; 934aa; Lynch Syndrome 2; ~31% Lynch; "
                "SEBACEOUS NEOPLASMS Muir-Torre PATHOGNOMONIC; GLIOBLASTOMA Turcot variant; "
                "EPCAM 3' deletion silences MSH2 -- test EPCAM MLPA if MSH2 sequencing negative; "
                "IHC: concomitant MSH2 + MSH6 loss)"
            ),
            "MSH6": (
                "AD LOF 2p16.3 (MutS alpha; 1360aa; Lynch Syndrome 5; ~18% Lynch; "
                "ENDOMETRIAL > CRC in females (endometrial 40-70%); REDUCED CRC penetrance 10-25%; "
                "MSI-L or MSS possible -- IHC mandatory if clinical suspicion; "
                "IHC: ISOLATED MSH6 LOSS (MSH2 retained))"
            ),
            "PMS2": (
                "AD LOF 7p22.1 (MutL alpha; 862aa; Lynch Syndrome 4; ~14% Lynch; "
                "LOWEST MMR penetrance (CRC 15-20%); ISOLATED PMS2 LOSS on IHC (MLH1 retained); "
                "BIALLELIC PMS2 = CMMRD (childhood CNS + haematologic + GI cancers + cafe-au-lait); "
                "Standard NGS MISSES exons 12-15 -- long-range PCR required)"
            ),
            "APC": (
                "AD LOF 5q22.2 (Wnt beta-catenin; 2843aa; FAP/AFAP/Gardner/Turcot; "
                ">100 COLORECTAL POLYPS PATHOGNOMONIC; CRC 100% lifetime untreated (classic FAP); "
                "PROPHYLACTIC COLECTOMY MANDATORY by age 25-35; "
                "Duodenal Spigelman staging; DESMOID codons 1310-2011; CHRPE bilateral pathognomonic)"
            ),
            "MUTYH": (
                "AR (autosomal recessive) 1p34.1 (base excision repair; 546aa; MAP; "
                "BIALLELIC required for MAP; 10-100 adenomas; CRC 80% lifetime biallelic; "
                "Y179C + G396D European founders; "
                "HETEROZYGOUS ~2x CRC only -- NOT Lynch-level; counsel carefully)"
            ),
            "STK11": (
                "AD LOF 19p13.3 (AMPK kinase; 433aa; Peutz-Jeghers Syndrome; "
                "MUCOCUTANEOUS LENTIGINES PERIORAL PATHOGNOMONIC (fade puberty -- buccal mucosa persists); "
                "PANCREATIC CANCER 132x relative risk; cancer lifetime 93%; "
                "hamartomatous polyps small bowel; intussusception emergency risk)"
            ),
            "SMAD4": (
                "AD LOF 18q21.2 (co-SMAD TGF-beta/BMP; 552aa; Juvenile Polyposis Syndrome; "
                "JUVENILE POLYPS (smooth mucosa, stalk, lamina propria) PATHOGNOMONIC; "
                "JPS-HHT OVERLAP 22% -- ECHO BUBBLE STUDY MANDATORY (pulmonary AVM -> paradoxical embolism); "
                "GASTRIC POLYPS SEVERE in SMAD4 vs BMPR1A; CRC 40% lifetime)"
            ),
        },
        "key_clinical_rules": [
            "MLH1: EXCLUDE somatic methylation (tumour MLH1 methylation + BRAF V600E) BEFORE diagnosing Lynch -- 15% sporadic MSI-H mimic",
            "MSH2: if MSH2 sequencing/MLPA negative but IHC shows MSH2 loss -- TEST EPCAM MLPA (3' deletion silences MSH2)",
            "MSH6: MSI testing can be MSI-L or MSS -- always do IHC if endometrial cancer <60 or Lynch family",
            "PMS2: standard NGS MISSES exons 12-15 (pseudogene PMS2CL) -- always confirm with long-range PCR + MLPA",
            "PMS2 biallelic = CMMRD: childhood cancers (brain, haematologic, CRC); cafe-au-lait spots mimic NF1",
            "APC FAP: prophylactic colectomy MANDATORY; duodenal Spigelman staging IV = consider prophylactic surgery",
            "APC desmoid: intra-abdominal post-colectomy; codons 1310-2011 highest risk; sulindac + tamoxifen first line",
            "MUTYH: BIALLELIC required for MAP; HETEROZYGOUS = 2x CRC risk only -- do NOT counsel as Lynch-level",
            "STK11: LENTIGINES FADE after puberty -- examine buccal mucosa + ask parent for childhood photos; pancreatic MRI/EUS annually from 30-35",
            "SMAD4: ECHO BUBBLE STUDY mandatory at diagnosis for pulmonary AVM (22% JPS-HHT overlap) -- PAVM causes stroke/brain abscess",
            "dMMR/MSI-H: pembrolizumab FDA approved 2017 (tissue-agnostic); all Lynch-associated CRC eligible if MSI-H/dMMR confirmed",
            "IHC cascade: stain MLH1-MSH2-MSH6-PMS2 -- loss pattern predicts gene (isolated PMS2 loss = PMS2; concomitant MLH1+PMS2 = MLH1; concomitant MSH2+MSH6 = MSH2 or EPCAM)",
        ],
        "gene_panel_note": (
            "Hereditary CRC & Polyposis gene panel (2024): MINIMUM MLH1, MSH2, MSH6, PMS2, APC, MUTYH, STK11, SMAD4, BMPR1A, PTEN (PHTS), RNF43; "
            "Extended: POLE, POLD1, NTHL1, MSH3, AXIN2 (oligodontia-CRC), GREM1 (serrated polyposis); "
            "IMPORTANT: "
            "  PMS2: long-range PCR required for exons 12-15 (pseudogene); "
            "  MSH2: include EPCAM MLPA (3' deletion panel); "
            "  MLH1: tumour MLH1 methylation + BRAF V600E testing before germline; "
            "  MUTYH: report biallelic separately from monoallelic (different risk counselling); "
            "Surveillance summary: "
            "  Lynch (MLH1/MSH2): colonoscopy 1-2yr from age 25; endometrial sampling 30-35yr; aspirin 600mg CAPP2 data; "
            "  Lynch MSH6/PMS2: colonoscopy every 2yr from 30; endometrial >35; "
            "  FAP: colectomy by 25-35 (polyp burden); post-IRA: rectoscopy 3-6mo; Spigelman duodenal staging; "
            "  MAP: colonoscopy 1-2yr from 18-25; colectomy when uncontrollable; "
            "  PJS: small bowel surveillance (capsule/MR enteroclysis) + colonoscopy every 2-3yr from age 8; EUS/MRI pancreas from 30; "
            "  JPS: colonoscopy annually from 15; gastroscopy annually (SMAD4); echo bubble at diagnosis (SMAD4); "
            "Pembrolizumab: all dMMR/MSI-H CRC (Lynch + sporadic MSI-H) eligible for checkpoint blockade"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-CRC-Polyposis-Atlas."""
    genes_data = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        seed = SEED_BASE + idx
        patients = _generate_patients_for_gene(gene, seed)
        n = len(patients)

        crc_n           = sum(1 for p in patients if p["crc_present"])
        endometrial_n   = sum(1 for p in patients if p["endometrial_cancer"])
        msi_h_n         = sum(1 for p in patients if p["msi_high"])
        synch_n         = sum(1 for p in patients if p["synchronous_crc"])
        polyposis_n     = sum(1 for p in patients if p["adenomas_gt100"])
        duodenal_n      = sum(1 for p in patients if p["duodenal_adenoma"])
        desmoid_n       = sum(1 for p in patients if p["desmoid_tumour"])
        chrpe_n         = sum(1 for p in patients if p["chrpe"])
        colectomy_n     = sum(1 for p in patients if p["colectomy_done"])
        lentigines_n    = sum(1 for p in patients if p["lentigines_perioral"])
        intuss_n        = sum(1 for p in patients if p["intussusception_history"])
        pancreatic_n    = sum(1 for p in patients if p["pancreatic_cancer"])
        breast_n        = sum(1 for p in patients if p["breast_cancer"])
        juvenile_n      = sum(1 for p in patients if p["juvenile_polyps_present"])
        hht_n           = sum(1 for p in patients if p["hht_features"])
        pavm_n          = sum(1 for p in patients if p["pulmonary_avm"])
        biallelic_n     = sum(1 for p in patients if p["biallelic_mutyh"])
        severe_n        = sum(1 for p in patients if p["severity"] == "severe")
        moderate_n      = sum(1 for p in patients if p["severity"] == "moderate")
        mild_n          = sum(1 for p in patients if p["severity"] == "mild")
        mean_diag_age   = round(sum(p["age_at_diagnosis_yrs"] for p in patients) / n, 1)
        mutations_seen  = list({p["mutation"] for p in patients})

        clinical_notes = {
            "MLH1":  "MLH1 promoter methylation exclusion MANDATORY before Lynch dx. BRAF V600E somatic = sporadic. dMMR CRC 40-70% lifetime. Pembrolizumab FDA2017 for MSI-H/dMMR.",
            "MSH2":  "EPCAM 3' deletion silences MSH2 (test if sequencing negative). Muir-Torre: sebaceous neoplasms PATHOGNOMONIC. Turcot: glioblastoma (not medulloblastoma). CRC 45-70% lifetime.",
            "MSH6":  "Endometrial > CRC in females (40-70% endometrial). Reduced CRC penetrance 10-25%. MSI-L/MSS possible -- IHC mandatory. Isolated MSH6 IHC loss (MSH2 retained). Later age dx ~55yr.",
            "PMS2":  "Lowest MMR penetrance (CRC 15-20%). Isolated PMS2 IHC loss (MLH1 retained). LR-PCR required exons 12-15 (pseudogene PMS2CL). Biallelic = CMMRD (childhood malignancies).",
            "APC":   "FAP: CRC 100% untreated. PROPHYLACTIC COLECTOMY by 25-35 years. Duodenal Spigelman IV = surgery. DESMOID: codons 1310-2011. CHRPE bilateral >=4 pathognomonic. Gardner: osteomas + cysts.",
            "MUTYH": "BIALLELIC required for MAP. Y179C + G396D European founders. CRC 80% lifetime biallelic. HETEROZYGOUS: 2x CRC only (not Lynch-level). G>T transversion signature in APC/KRAS.",
            "STK11": "LENTIGINES PERIORAL PATHOGNOMONIC (fade puberty -- check buccal mucosa). Pancreatic 132x relative risk. EUS/MRI from age 30-35. Intussusception emergency. Cancer lifetime 93%.",
            "SMAD4": "JUVENILE POLYPS PATHOGNOMONIC (smooth mucosa, stalk, expanded LP). JPS-HHT overlap 22% -- ECHO BUBBLE MANDATORY for PAVM. Gastric polyps severe. CRC 40%; gastric 20% lifetime.",
        }

        genes_data.append({
            "gene":                    gene,
            "locus":                   gene_info["locus"],
            "n":                       n,
            "n_patients":              n,
            "severe_pct":              round(severe_n / n * 100, 1),
            "moderate_pct":            round(moderate_n / n * 100, 1),
            "mild_pct":                round(mild_n / n * 100, 1),
            "crc_pct":                 round(crc_n / n * 100, 1),
            "endometrial_pct":         round(endometrial_n / n * 100, 1),
            "msi_h_pct":               round(msi_h_n / n * 100, 1),
            "synchronous_crc_pct":     round(synch_n / n * 100, 1),
            "polyposis_gt100_pct":     round(polyposis_n / n * 100, 1),
            "duodenal_adenoma_pct":    round(duodenal_n / n * 100, 1),
            "desmoid_pct":             round(desmoid_n / n * 100, 1),
            "chrpe_pct":               round(chrpe_n / n * 100, 1),
            "colectomy_pct":           round(colectomy_n / n * 100, 1),
            "lentigines_pct":          round(lentigines_n / n * 100, 1),
            "intussusception_pct":     round(intuss_n / n * 100, 1),
            "pancreatic_cancer_pct":   round(pancreatic_n / n * 100, 1),
            "breast_cancer_pct":       round(breast_n / n * 100, 1),
            "juvenile_polyps_pct":     round(juvenile_n / n * 100, 1),
            "hht_features_pct":        round(hht_n / n * 100, 1),
            "pulmonary_avm_pct":       round(pavm_n / n * 100, 1),
            "biallelic_mutyh_pct":     round(biallelic_n / n * 100, 1),
            "mean_age_dx_yrs":         mean_diag_age,
            "sample_mutations":        mutations_seen[:4],
            "protein":                 gene_info["protein"],
            "inheritance":             gene_info["inheritance"][:250],
            "disease_category":        gene_info["disease_category"],
            "clinical_note":           clinical_notes.get(gene, ""),
        })

    return {
        "atlas": "Hereditary-CRC-Polyposis-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-CRC-Polyposis-Atlas."""
    definitions = [
        {
            "term": "MMR-IHC-Interpretation-Cascade",
            "definition": (
                "IHC (immunohistochemistry) for mismatch repair proteins -- interpretation algorithm: "
                "STAIN: MLH1 / MSH2 / MSH6 / PMS2 on CRC or endometrial tumour tissue; "
                "PATTERN 1 -- MLH1 + PMS2 BOTH ABSENT (MSH2 + MSH6 retained): "
                "  FIRST TEST: MLH1 PROMOTER METHYLATION (tumour tissue; bisulphite sequencing / methylation-specific PCR); "
                "  Methylation POSITIVE: sporadic MSI-H (EPIGENETIC silencing); "
                "    Next: BRAF V600E somatic test (BRAF V600E + methylation = almost certainly sporadic; no Lynch workup); "
                "  Methylation NEGATIVE: strongly suggests GERMLINE MLH1 pathogenic variant; "
                "    BRAF V600E negative + methylation negative: refer for MLH1 germline testing; "
                "PATTERN 2 -- MSH2 + MSH6 BOTH ABSENT (MLH1 + PMS2 retained): "
                "  Order: MSH2 germline sequencing + MLPA; "
                "  If MSH2 negative: EPCAM MLPA (3' deletion silences MSH2 promoter by read-through); "
                "  EPCAM deletion = Lynch-equivalent; cascade test family members for EPCAM deletion; "
                "PATTERN 3 -- ISOLATED MSH6 ABSENT (MLH1/MSH2/PMS2 retained): "
                "  MSH6 germline sequencing + MLPA; "
                "  Note: MSI testing may show MSI-L or MSS in MSH6 -- IHC is more sensitive; "
                "PATTERN 4 -- ISOLATED PMS2 ABSENT (MLH1/MSH2/MSH6 retained): "
                "  PMS2 germline sequencing + MLPA (P008 kit); "
                "  Exons 12-15: long-range PCR required (pseudogene PMS2CL interference); "
                "  ISOLATED PMS2 LOSS = PMS2 germline until proven otherwise; "
                "PATTERN 5 -- ALL FOUR ABSENT: "
                "  Unusual; consider somatic second-hit on MLH1/MSH2; repeat on different tissue block; "
                "PATTERN 6 -- ALL FOUR RETAINED: "
                "  MMR proficient (pMMR) = Lynch EXTREMELY UNLIKELY; still consider if very strong family history; "
                "  MSS CRC: worse response to pembrolizumab; standard chemotherapy. "
                "CLINICAL RULE: IHC is a SCREENING tool; germline confirmation required before Lynch diagnosis. "
                "False-negative IHC possible: poor fixation, heterogeneous loss, technical failure."
            ),
        },
        {
            "term": "MLH1-Somatic-Methylation-vs-Lynch-Algorithm",
            "definition": (
                "The MOST COMMON diagnostic error in Lynch workup: diagnosing Lynch when MSI-H is sporadic. "
                "15% OF ALL CRC IS MSI-H -- but only ~3% of all CRC is Lynch syndrome. "
                "Sporadic MSI-H mechanism: SOMATIC hypermethylation of MLH1 PROMOTER (epigenetic silencing); "
                "  Common in BRAF V600E CRC (serrated pathway tumours); older patients (mean age 72 vs 44 for Lynch); "
                "EXCLUSION PROTOCOL (mandatory before germline MLH1 testing): "
                "Step 1 -- MLH1 promoter methylation test on TUMOUR tissue: "
                "  Bisulphite sequencing or methylation-specific PCR; "
                "  POSITIVE: likely sporadic (confirm with BRAF V600E); do NOT proceed to germline without further evidence; "
                "Step 2 -- BRAF V600E somatic test (rapid PCR or NGS on tumour): "
                "  POSITIVE (+ methylation): essentially sporadic -- Lynch workup NOT indicated; "
                "  NEGATIVE (+ methylation): methylated Lynch is rare but possible; consider germline if strong family history; "
                "Step 3 -- Germline MLH1 sequencing + MLPA: only if both methylation-negative and BRAF-negative; "
                "IMPORTANT: Germline MLH1 variants in METHYLATED tumours: "
                "  Very rare BUT reported -- young patient (<40) + methylation + negative BRAF: still consider germline; "
                "CLINICAL OUTCOME: "
                "  Sporadic MSI-H (methylated, BRAF V600E positive): NO family surveillance, NO Lynch counselling; "
                "  STILL pembrolizumab-eligible (MSI-H regardless of cause); "
                "  Lynch confirmed: cascade testing family; prophylactic gynaecological surveillance; colonoscopy."
            ),
        },
        {
            "term": "FAP-APC-Management-Protocol",
            "definition": (
                "FAP (Familial Adenomatous Polyposis) management -- APC pathogenic variant confirmed: "
                "INITIAL ASSESSMENT: "
                "  Sigmoidoscopy/colonoscopy from age 10-12 for polyp onset surveillance; "
                "  If >20-30 adenomas: FAP confirmed -- refer to specialist centre; "
                "  Eye exam: CHRPE documentation (bilateral >=4 lesions confirms classic FAP genotype); "
                "COLECTOMY DECISION: "
                "  When polyp burden unmanageable or progressive dysplasia (HGD): "
                "  Option A -- Total colectomy + ileorectal anastomosis (IRA): "
                "    Appropriate if rectal sparing possible (<20 polyps in rectum); "
                "    Annual rectoscopy + rectal surveillance post-IRA for life (rectal stump cancer risk 10-25% at 20yr); "
                "  Option B -- Proctocolectomy + IPAA (ileal pouch-anal anastomosis): "
                "    For severe rectal polyposis; eliminates rectal cancer risk; "
                "    J-pouch complications (pouchitis, pouch failure); "
                "  Timing: recommend before age 25-30 (mean cancer age without colectomy = 40yr); "
                "DUODENAL/PERIAMPULLARY SURVEILLANCE (Spigelman staging): "
                "  Staging I (score 1-4): endoscopy every 5 years; "
                "  Staging II (score 5-6): endoscopy every 3 years; "
                "  Staging III (score 7-8): endoscopy every 1-2 years + consider endoscopic treatment; "
                "  Staging IV (score 9-12): prophylactic surgery (Whipple/PPPD) discussion; "
                "  Periampullary adenocarcinoma: second most common FAP cancer death after CRC; "
                "DESMOID MANAGEMENT: "
                "  First-line: sulindac 150-300mg BD + tamoxifen 40-120mg/day (anti-fibrotic); "
                "  Second-line: sorafenib (TKI); imatinib; chemotherapy (vinblastine/methotrexate or doxorubicin); "
                "  Surgery: high recurrence rate; bowel obstruction/ureteral obstruction may mandate intervention; "
                "  Genotype-specific: avoid prophylactic surgery in desmoid-prone codons if resection can be deferred; "
                "CHEMOPREVENTION: celecoxib reduces polyp burden (post-colectomy rectal/pouch polyps); sulindac adjunct; "
                "THYROID SURVEILLANCE: annual thyroid USS (2-3% thyroid cancer in FAP -- papillary variant)."
            ),
        },
        {
            "term": "Peutz-Jeghers-Surveillance-Protocol",
            "definition": (
                "Peutz-Jeghers Syndrome (PJS) -- STK11 pathogenic variant -- surveillance: "
                "DIAGNOSIS CRITERIA: "
                "  >=2 histologically confirmed PJ polyps; OR "
                "  Any number PJ polyps + family history PJS; OR "
                "  Characteristic mucocutaneous lentigines + family history; OR "
                "  Any PJ polyps + characteristic lentigines; "
                "LENTIGINES: perioral/buccal (persistent); labial; digital; perianal; "
                "  Key: buccal mucosa lentigines persist throughout life; perioral fade after puberty; "
                "  Ask parents to show childhood photographs if adult patient; "
                "SMALL BOWEL SURVEILLANCE: "
                "  Capsule endoscopy or MR enteroclysis from age 8-10; "
                "  Every 2-3 years; "
                "  Polypectomy: endoscopic if feasible; surgical if >1.5-2 cm or multiple; "
                "  Intraoperative enteroscopy (IOE) at laparotomy: clear ALL polyps simultaneously; "
                "GI SURVEILLANCE SCHEDULE: "
                "  Upper GI endoscopy (OGD): from age 8; every 2-3 years; "
                "  Colonoscopy: from age 8; every 2-3 years; "
                "  Capsule/MRE small bowel: from age 8; every 2-3 years; "
                "PANCREATIC SURVEILLANCE (HIGHEST PRIORITY): "
                "  PANCREATIC CANCER RELATIVE RISK 132x general population; "
                "  EUS + MRI abdomen annually from age 30-35; "
                "  Consider MRI from age 25 if family history pancreatic cancer; "
                "BREAST SURVEILLANCE: "
                "  Annual breast MRI + mammography from age 25 (high risk >20% lifetime); "
                "  Consider risk-reducing mastectomy discussion for very high-risk individuals; "
                "GYNAECOLOGICAL SURVEILLANCE (female): "
                "  Annual cervical smear (standard) + pelvic MRI for minimal deviation adenocarcinoma (MDC); "
                "  MDC/adenoma malignum NOT detected on standard Pap smear -- pelvic MRI recommended; "
                "  Annual transvaginal USS + CA-125; "
                "TESTICULAR (male): "
                "  Annual testicular USS: Sertoli cell tumour large cell calcifying (LCCSCT); "
                "  LCCSCT causes feminisation (aromatase excess) -- gynaecomastia, growth acceleration; "
                "LUNG: annual CT chest surveillance discussed from 30-40."
            ),
        },
        {
            "term": "JPS-SMAD4-HHT-Overlap-Protocol",
            "definition": (
                "JPS-HHT Overlap in SMAD4 pathogenic variants -- critical management: "
                "PREVALENCE: ~22% of SMAD4-JPS patients have features of HHT (hereditary haemorrhagic telangiectasia); "
                "HHT DIAGNOSIS (Curacao criteria, 3 of 4): "
                "  1. Spontaneous recurrent epistaxis; "
                "  2. Mucocutaneous telangiectasia (lips, oral mucosa, fingers, nose); "
                "  3. Visceral AVM (pulmonary, hepatic, cerebral, spinal); "
                "  4. First-degree family member HHT; "
                "MANDATORY INVESTIGATIONS at SMAD4-JPS DIAGNOSIS: "
                "  Transthoracic ECHO BUBBLE STUDY: contrast echocardiography for PAVM detection; "
                "    Positive: right-to-left shunt -> CT pulmonary angiography for PAVM characterisation; "
                "    PAVM embolisation: coil/plug embolisation for feeding vessels >3 mm; "
                "    Follow-up CTPA: 3-5 years; "
                "  CT PULMONARY ANGIOGRAPHY: if echo positive (or high clinical suspicion); "
                "    Untreated PAVM causes: stroke (paradoxical embolism), brain abscess (right-to-left bypass of pulmonary filter); "
                "  MRI BRAIN: cerebral AVM (haemorrhagic stroke risk); repeat every 5 years if clear; "
                "  LIVER IMAGING (USS/Doppler or CT): hepatic AVM -- hepatic arteriovenous shunting -> high-output cardiac failure; "
                "EPISTAXIS MANAGEMENT: "
                "  Humidification, moisturisers; SNARE trial data -- tranexamic acid; "
                "  Bevacizumab (anti-VEGF): emerging evidence for severe HHT epistaxis + AVMs; "
                "JPS-SPECIFIC GASTRIC DISEASE IN SMAD4: "
                "  Gastric polyps often SEVERE (mass of gastric polyps in childhood/adolescence); "
                "  Gastric polyposis may require repeated endoscopic debulking; "
                "  Consider prophylactic total/partial gastrectomy if: uncontrolled gastric bleeding, severe gastric polyposis, HGD; "
                "  Protein-losing gastropathy: hypoalbuminaemia from gastric polyp exudate; "
                "IMPORTANT DISTINCTION: "
                "  BMPR1A-JPS: rare HHT overlap; gastric polyps less severe; "
                "  SMAD4-JPS: more severe gastric disease + HHT overlap; echo bubble study essential."
            ),
        },
        {
            "term": "Lynch-Syndrome-Surveillance-and-Aspirin",
            "definition": (
                "Lynch Syndrome (MLH1/MSH2/MSH6/PMS2) cancer surveillance and chemoprevention: "
                "COLORECTAL SURVEILLANCE: "
                "  MLH1/MSH2: colonoscopy every 1-2 years from age 25; "
                "  MSH6/PMS2: colonoscopy every 2-3 years from age 30; "
                "  Annual colonoscopy vs 2-year: evidence mixed; ESGE 2023 recommends 1-2yr for MLH1/MSH2; "
                "  Right colon polyps predominate (70% proximal to splenic flexure); chromoendoscopy increases polyp detection; "
                "GYNAECOLOGICAL (women with Lynch): "
                "  MSH2 (highest endometrial risk): annual endometrial sampling from age 30-35; "
                "  MLH1/MSH6: annual endometrial sampling + transvaginal USS from 35; "
                "  RISK-REDUCING SURGERY: prophylactic hysterectomy + bilateral salpingo-oophorectomy (RRHBSO): "
                "    Discuss when reproductive wishes complete; eliminates endometrial + ovarian Lynch risk; "
                "    Associated CRC surveillance still required post-hysterectomy; "
                "ASPIRIN CHEMOPREVENTION: "
                "  CAPP2 TRIAL (600mg aspirin, double-blind RCT, Lynch patients): "
                "    Primary analysis: no significant reduction at 2 years; "
                "    LONG-TERM FOLLOW-UP (>10 years): 63% reduction in CRC incidence (significant); "
                "    Effect strongest in MLH1/MSH2; trend in MSH6; insufficient data PMS2; "
                "  DOSE: 600mg daily (CAPP2) vs lower doses studied in CAPP3; "
                "  MECHANISM: possibly via prostaglandin-mediated immunity + mismatch repair bypass; "
                "UPPER GI: "
                "  Gastric + small bowel cancer risk (especially MLH1/MSH2); "
                "  H. pylori eradication: recommended (Lynch + H. pylori = elevated gastric risk); "
                "  OGD: not routine for all Lynch; consider if Asian origin (higher gastric risk) or gastric symptoms; "
                "URINARY TRACT: "
                "  MSH2 highest urinary tract cancer risk (~18% lifetime); "
                "  Annual urine cytology from age 30-35 (MSH2); controversial evidence; "
                "PEMBROLIZUMAB: "
                "  All dMMR/MSI-H Lynch-associated CRC eligible; "
                "  KEYNOTE-177: pembrolizumab vs FOLFOX first-line in MSI-H mCRC -- significant PFS2 and OS benefit; "
                "  Adjuvant pembrolizumab (stage III dMMR CRC): KEYNOTE-177 adjuvant ongoing; "
                "  IMMUNOTHERAPY RESPONSE: Lynch CRC high TMB + neoantigen load -> excellent PD-1 response; "
                "  Endometrial Lynch: pembrolizumab + lenvatinib (Keytruda+Lenvima) FDA2019 for MSI-H endometrial; "
                "GENETIC COUNSELLING: "
                "  Predictive testing of first-degree relatives from age 18-25; "
                "  Amsterdam II criteria (clinical diagnosis) if genetic testing not available; "
                "  Cascade testing: insurance/coverage navigation needed in some jurisdictions."
            ),
        },
        {
            "term": "MUTYH-MAP-vs-APC-FAP-Differentiation",
            "definition": (
                "Differentiating MAP (MUTYH-associated polyposis) from APC-FAP/AFAP: "
                "CLINICAL FEATURES: "
                "  MAP: 10-100 colorectal adenomas; onset 45-55 years; AR inheritance; "
                "  AFAP: 10-100 colorectal adenomas; onset 30-45 years; AD inheritance; "
                "  Classic FAP: >100 polyps; onset 16+ years; AD inheritance; "
                "INHERITANCE CLUE: "
                "  MAP: AUTOSOMAL RECESSIVE -- affected siblings, unaffected parents (or parents as late-onset carriers); "
                "  AFAP: AUTOSOMAL DOMINANT -- one parent likely affected (but variable penetrance); "
                "  CONSANGUINITY: increases MAP risk; "
                "MOLECULAR SIGNATURE: "
                "  MAP: G:C > T:A TRANSVERSIONS in APC + KRAS (8-oxoguanine signature); "
                "    KRAS p.Gly12Cys (c.34G>T): specific MUTYH signature; "
                "    APC somatic mutations: G to T transversions rather than frameshift/nonsense; "
                "  FAP: APC truncating mutations (frameshift, nonsense, splice); "
                "TESTING ALGORITHM (10-100 polyps): "
                "  Step 1: APC full sequencing + MLPA (rule out AFAP); "
                "  Step 2: MUTYH biallelic sequencing (Y179C + G396D + full gene); "
                "  Step 3: if negative: consider NTHL1, MSH3, POLD1, POLE (alternative polyposis genes); "
                "BIALLELIC MUTYH MANAGEMENT (MAP): "
                "  Colonoscopy every 1-2 years from age 18-25; "
                "  When polyps >30-50 or HGD: colectomy (IRA or IPAA); "
                "  Post-colectomy rectal surveillance every 6 months; "
                "  Duodenal surveillance every 3-5 years (4-17% duodenal adenomas, less than FAP); "
                "MONOALLELIC MUTYH (carrier): "
                "  ~1.5-2x elevated CRC risk vs general population; "
                "  Colonoscopy screening from age 40 (same as average-to-moderate risk); "
                "  DO NOT treat as Lynch or FAP risk -- significant counselling difference; "
                "  Insurance and familial implications: partners of monoallelic carriers should be tested "
                "    (biallelic offspring risk = 25% if partner also carries MUTYH variant)."
            ),
        },
        {
            "term": "dMMR-MSI-Pembrolizumab-Protocol",
            "definition": (
                "dMMR/MSI-H -- Pembrolizumab (Keytruda) clinical protocol: "
                "TESTING INDICATIONS: "
                "  ALL CRC at diagnosis: universal MSI/MMR testing (NCCN, ESMO); "
                "  Endometrial cancer: all endometrial cancers at diagnosis; "
                "  Other cancers: MSI-H testing on any advanced solid tumour (tissue-agnostic FDA2017 indication); "
                "MSI TESTING METHODS: "
                "  IHC (MLH1/MSH2/MSH6/PMS2): rapid, widely available; functional readout; "
                "  PCR-based MSI (microsatellite panel: BAT25, BAT26, BAT40, D5S346, D2S123): gold standard; "
                "  NGS TMB (tumour mutational burden): >10 mut/Mb = TMB-high (FDA2020 approval); "
                "  CONCORDANCE: IHC + PCR MSI concordance ~95%; "
                "  Note: MSH6 variants may be MSI-L on older panels -- IHC preferred if clinical suspicion; "
                "PEMBROLIZUMAB INDICATIONS: "
                "  KEYNOTE-158 (2017): pembrolizumab for MSI-H/dMMR unresectable or metastatic solid tumours "
                "    (first tissue-agnostic FDA approval); "
                "  KEYNOTE-177 (2021): pembrolizumab FIRST-LINE vs FOLFOX for MSI-H mCRC; "
                "    PFS2: 54.0 vs 24.5 months (HR 0.59); OS trend favourable; "
                "  KEYNOTE-158 endometrial: pembrolizumab for MSI-H/dMMR endometrial cancer; "
                "RESPONSE MECHANISM: "
                "  dMMR tumours: high mutational burden -> high neoantigen load -> T-cell recognition; "
                "  PD-1 blockade: reverses exhausted TIL (tumour-infiltrating lymphocyte) anergy; "
                "  Lynch vs sporadic MSI-H: BOTH eligible; cause of dMMR does not change immunotherapy eligibility; "
                "TOXICITY: "
                "  Immune-related adverse events (irAEs): colitis (10-15%); thyroiditis; pneumonitis; hepatitis; "
                "  Lynch patients: discuss theoretical risk of stimulating autoimmunity in genetic predisposition context; "
                "RESISTANCE: "
                "  Primary resistance: low TIL infiltration (cold tumours); POLE exonuclease-domain mutations co-existent; "
                "  Secondary resistance: JAK1/JAK2 mutations (loss of IFN-gamma signalling); beta2-microglobulin loss."
            ),
        },
    ]

    return {
        "atlas": "Hereditary-CRC-Polyposis-Atlas",
        "count": len(definitions),
        "definitions": definitions,
    }
