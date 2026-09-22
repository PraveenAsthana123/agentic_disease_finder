#!/usr/bin/env python3
"""Hereditary-Neurofibromatosis-Schwannomatosis-Atlas — Complete 8-Gene NF & Schwannomatosis Predisposition Atlas
NF1    (Neurofibromin 1; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis Type 1 (NF1); MOST COMMON autosomal dominant 1:3000;
         ≥6 café-au-lait macules >5mm prepubertal / >15mm postpubertal PATHOGNOMONIC;
         Lisch nodules (iris hamartomas) PATHOGNOMONIC; plexiform neurofibromas; MPNST 8-13%;
         selumetinib (Koselugo) FDA2020 for paediatric plexiform neurofibromas;
         AVOID RADIATION — radiation-induced MPNST documented; seed SEED_BASE+0) ·
NF2    (Neurofibromin 2 / Merlin; 595aa; 22q12.2; AD LOF;
         Neurofibromatosis Type 2 (NF2); 1:25000;
         BILATERAL vestibular schwannomas PATHOGNOMONIC (age 20s-30s);
         meningioma (50-75%); ependymoma (18%); juvenile posterior subcapsular lens opacity PATHOGNOMONIC;
         bevacizumab for progressive VS (VEGF-independent mechanism); annual audiology + MRI;
         seed SEED_BASE+1) ·
SMARCB1 (SWI/SNF Related Matrix Associated Actin Dependent Regulator of Chromatin Subfamily B Member 1;
         385aa; 22q11.23; AD LOF;
         Schwannomatosis Type 2 + Rhabdoid Tumour Predisposition Syndrome (RTPS2);
         Atypical Teratoid/Rhabdoid Tumour (AT/RT) PATHOGNOMONIC in children under 3;
         INI1 / BAF47 IHC nuclear loss PATHOGNOMONIC for rhabdoid; SWI/SNF complex;
         sibling surveillance for CNS tumours under 3yr; EZH2 inhibitor tazemetostat;
         seed SEED_BASE+2) ·
LZTR1  (Leucine Zipper Like Transcription Regulator 1; 827aa; 22q11.21; AD/AR;
         Schwannomatosis Type 1 (SCHWT1); UNIQUE BIDIRECTIONAL inheritance;
         AD dominant-negative OR AR biallelic → multiple schwannomas; no skin findings;
         PAINFUL schwannomas characteristic; also causes Noonan syndrome type 10;
         CUL3 adaptor for RAS ubiquitination; TEST BOTH PARENTS;
         seed SEED_BASE+3) ·
SPRED1 (Sprouty-Related EVH1 Domain-Containing Protein 1; 444aa; 15q14; AD LOF;
         Legius Syndrome (NF1-like without tumours);
         ≥6 café-au-lait macules + axillary/inguinal freckling WITHOUT neurofibromas/Lisch nodules;
         NF1-CRITICAL DDx: Legius = NO plexiform neurofibromas, NO Lisch, NO MPNST risk;
         SPRED1 = negative regulator of RAS-MAPK via RAF1 inhibition; NF1 testing first;
         seed SEED_BASE+4) ·
SMARCE1 (SWI/SNF Related Matrix Associated Actin Dependent Regulator of Chromatin Subfamily E Member 1;
         411aa; 17q21.2; AD LOF;
         Familial Multiple Spinal Meningiomas;
         spinal meningioma in young women PATHOGNOMONIC; clear-cell histology;
         SWI/SNF subunit (BAF57); full-spine MRI at diagnosis + annual surveillance;
         intracranial meningiomas also possible; no skin or NF features;
         seed SEED_BASE+5) ·
PRKAR1A (Protein Kinase cAMP-Dependent Type I Regulatory Subunit Alpha; 381aa; 17q24.2; AD LOF;
         Carney Complex (CNC);
         spotty skin pigmentation (lentigines + epithelioid blue nevi) PATHOGNOMONIC;
         cardiac myxoma 30% LIFE-THREATENING — annual echocardiogram mandatory;
         primary pigmented nodular adrenocortical disease (PPNAD) Cushing;
         psammomatous melanotic schwannomas; GH-secreting pituitary adenoma;
         cAMP/PKA signalling pathway; seed SEED_BASE+6) ·
AKT1   (RAC-Alpha Serine/Threonine-Protein Kinase; 480aa; 14q32.33; AD somatic/postzygotic GOF;
         Proteus Syndrome;
         cerebriform connective tissue naevus (CCTN) PATHOGNOMONIC;
         hemihyperplasia; epidermal naevi; deep vein thrombosis (PE risk);
         AKT1 p.E17K somatic mosaic most common; PI3K-AKT-mTOR pathway;
         alpelisib (PI3K inhibitor) emerging evidence;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3110-3117)
"""
import random

SEED_BASE = 3110

ATLAS_GENES = [
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-1-RAS-GAP-RAS-GTPase-Activating-Protein-Inactivates-RAS-GTP-"
            "NF1-MOST-COMMON-Autosomal-Dominant-1in3000-Café-au-Lait-PATHOGNOMONIC-"
            "Selumetinib-FDA2020-Plexiform-Neurofibromas-MPNST-8-13pct-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 17q11.2 NF1 encodes neurofibromin, the largest GAP (GTPase-activating protein): "
            "STRUCTURE: Central GAP-related domain (GRD) converts active RAS-GTP → inactive RAS-GDP; "
            "  NF1 LOF → loss of RAS inactivation → sustained RAS-MAPK/PI3K-AKT signalling; "
            "  RAS pathway hyperactivation → melanocyte proliferation (café-au-lait), Schwann cell tumours; "
            "  Loss of heterozygosity (LOH) at 17q11.2 required for tumour formation (second hit); "
            "DIAGNOSTIC CRITERIA (≥2 of): "
            "  1. ≥6 café-au-lait macules: >5mm prepubertal; >15mm postpubertal — PATHOGNOMONIC hallmark; "
            "  2. ≥2 neurofibromas (any type) OR ≥1 plexiform neurofibroma; "
            "  3. Axillary or inguinal freckling (Crowe's sign); "
            "  4. Optic pathway glioma; "
            "  5. ≥2 Lisch nodules (iris hamartomas on slit-lamp) — PATHOGNOMONIC if bilateral; "
            "  6. Distinctive osseous lesion (sphenoid dysplasia; tibial pseudarthrosis); "
            "  7. First-degree relative with NF1 by above criteria; "
            "  NEW 2021 criteria: pathogenic NF1 variant counts as one criterion; "
            "PLEXIFORM NEUROFIBROMAS: "
            "  Present in ~30-50% NF1 patients; worm-like tumours following nerve branches; "
            "  MPNST (malignant peripheral nerve sheath tumour) risk 8-13% lifetime; "
            "  MPNST risk factors: large plexiform, internal plexiform, family history MPNST, NF1+CDKN2A deletion; "
            "  Selumetinib (MEK1/2 inhibitor; Koselugo) FDA2020: approved for paediatric NF1 plexiform; "
            "OPTIC PATHWAY GLIOMA: "
            "  15-20% NF1 patients; usually low-grade astrocytoma (pilocytic); "
            "  Annual ophthalmology screening age 1-7; visual acuity + visual fields; "
            "  Carboplatin + vincristine first-line treatment; "
            "AVOID RADIATION: "
            "  Radiation increases MPNST risk in radiation field — documented in multiple cohorts; "
            "  Proton therapy preferred if radiation unavoidable (brain tumours); "
            "LEARNING DISABILITIES: 50-60%; ADHD 40-60%; social difficulties; "
            "CARDIOVASCULAR: hypertension (renal artery stenosis / essential); pulmonary stenosis 2%; "
            "ANNUAL SURVEILLANCE: full neurological exam; blood pressure; ophthalmology (age 1-7); "
            "  brain MRI if symptomatic; full-body skin exam; "
        ),
        "inheritance": (
            "AD LOF 17q11.2 — NF1. 1:3000 live births (most common AD disorder). "
            "50% de novo mutations (no family history). "
            "NF1 is 350kb gene — one of the largest human genes; del/dup/complex rearrangements common. "
            "Point mutations, splicing, frameshifts, whole-gene deletions (somatic mosaicism ~10%). "
            "MLPA + sequencing required (large deletions in ~5-10%). "
            "Somatic (segmental) NF1: mosaic — one body segment; reduced severity; "
            "  BUT segmental NF1 can transmit full NF1 to offspring if gonadal cells affected."
        ),
        "surveillance_key": "selumetinib-FDA2020; annual ophthalmology age 1-7; avoid radiation; full-body skin exam; BP monitoring; MPNST surveillance",
        "pathognomonic": "café-au-lait macules ≥6; Lisch nodules bilateral; axillary/inguinal freckling",
    },
    {
        "gene": "NF2",
        "protein": (
            "NF2 -- 22q12.2 Autosomal-Dominant-LOF -- 595aa -- "
            "Neurofibromin-2-Merlin-ERM-Ezrin-Radixin-Moesin-Family-"
            "NF2-Bilateral-Vestibular-Schwannoma-PATHOGNOMONIC-1in25000-"
            "Bevacizumab-VS-Progression-Annual-Audiology-MRI-OMIM-101000"
        ),
        "locus": "22q12.2",
        "protein_size": (
            "595 aa / 22q12.2 NF2 encodes merlin (moesin-ezrin-radixin-like protein): "
            "STRUCTURE: N-terminal FERM domain (F1/F2/F3 lobes); linker; C-terminal domain; "
            "  Merlin regulates contact inhibition — active at cell junctions; "
            "  LOF → loss of contact inhibition → Schwann cell and meningeal cell proliferation; "
            "  Merlin links cytoskeleton to cell surface receptors (ErbB2, VEGFR); "
            "  Downstream: YAP/Hippo signalling, mTOR, PI3K/Akt; "
            "DIAGNOSTIC CRITERIA: "
            "  PATHOGNOMONIC: BILATERAL vestibular schwannomas at any age; OR "
            "  First-degree relative NF2 + unilateral VS age <30yr OR ≥2 of: meningioma/glioma/schwannoma/posterior lens opacity; "
            "BILATERAL VESTIBULAR SCHWANNOMAS: "
            "  Nearly 100% penetrance by age 60; typical onset age 20s-30s; "
            "  Progressive sensorineural hearing loss, tinnitus, balance disturbance; "
            "  ANNUAL audiology (ABR/BAEP) + annual MRI brain+spine; "
            "  Bevacizumab: VEGF inhibitor; FDA-compassionate use for growing VS; "
            "    Hearing improvement 36% + tumour volume stabilisation; "
            "  Surgery vs observation: small, stable VS monitored; growing or symptomatic → surgery or SRS; "
            "JUVENILE POSTERIOR SUBCAPSULAR LENS OPACITY: "
            "  PATHOGNOMONIC in young NF2 patients; slit-lamp annual; "
            "MENINGIOMA: 50-75% NF2 patients; multiple; spinal > intracranial often; "
            "SPINAL EPENDYMOMA: 18%; often cervico-medullary; serial MRI; "
            "PERIPHERAL NEUROPATHY: progressive (CIPN-like); common in severe NF2; "
            "NOVEL THERAPIES: "
            "  MEK inhibitors (selumetinib, cobimetinib): early-phase trials for NF2 schwannomas; "
            "  FAK inhibitors: preclinical; "
            "COCHLEAR IMPLANT: considered when hearing declines; "
            "AUDITORY BRAINSTEM IMPLANT (ABI): for bilateral deafness post-surgery; "
        ),
        "inheritance": (
            "AD LOF 22q12.2 — NF2. 1:25,000 live births. ~50% de novo. "
            "Somatic mosaicism ~30% (mosaic NF2 may have milder or asymmetric presentation). "
            "Genotype-phenotype: truncating variants → severe; splice/missense → milder; "
            "  Large del/rearrangements → mild; mosaics → asymmetric. "
            "Manchester criteria: bilateral VS alone sufficient for diagnosis."
        ),
        "surveillance_key": "annual MRI brain+spine; annual audiology; slit-lamp annually; bevacizumab for progressive VS",
        "pathognomonic": "bilateral vestibular schwannomas; juvenile posterior subcapsular lens opacity",
    },
    {
        "gene": "SMARCB1",
        "protein": (
            "SMARCB1 -- 22q11.23 Autosomal-Dominant-LOF -- 385aa -- "
            "SWI-SNF-BAF-Complex-Subunit-INI1-BAF47-Chromatin-Remodelling-"
            "Schwannomatosis-Type2-RTPS2-AT-RT-PATHOGNOMONIC-Under-3-INI1-IHC-Loss-PATHOGNOMONIC-"
            "EZH2-Inhibitor-Tazemetostat-Sibling-Surveillance-OMIM-609322"
        ),
        "locus": "22q11.23",
        "protein_size": (
            "385 aa / 22q11.23 SMARCB1 encodes INI1 (integrase interactor 1) / BAF47: "
            "STRUCTURE: Core structural component of SWI/SNF (BAF/PBAF) chromatin remodelling complexes; "
            "  Contains RRM1/RRM2 domains (RNA recognition motif-like) that form the central SWI/SNF scaffold; "
            "  INI1 is a tumour suppressor — SMARCB1 LOF → loss of BAF complex integrity → "
            "    failure to evict PRC1/PRC2 complexes from regulatory elements → "
            "    EZH2-mediated H3K27 trimethylation → silencing of CDKN2A, CDKN1A; "
            "RHABDOID TUMOUR PREDISPOSITION: "
            "  AT/RT (atypical teratoid/rhabdoid tumour) — most common primary CNS tumour in infants; "
            "  PATHOGNOMONIC: INI1 nuclear IHC loss in tumour cells; "
            "  Malignant rhabdoid tumour (MRT) — kidney, soft tissue; "
            "  Children under 3yr: AT/RT or MRT should prompt germline SMARCB1 testing; "
            "  SIBLING SURVEILLANCE: brain MRI + abdominal ultrasound at birth and 6-monthly if parent has SMARCB1 variant; "
            "SCHWANNOMATOSIS TYPE 2: "
            "  Multiple schwannomas (NOT bilateral VS); typically adults; "
            "  Segmental distribution common; painful; no café-au-lait; "
            "  Distinguish from NF2: no bilateral VS, no meningioma (usually); "
            "INI1 IHC: "
            "  Nuclear INI1 loss in tumour = SMARCB1 inactivation (somatic second hit); "
            "  Retained nuclear INI1 effectively rules out rhabdoid mechanism; "
            "EZH2 INHIBITOR: "
            "  Tazemetostat (FDA2020 for epithelioid sarcoma — SMARCB1 LOF tumours); "
            "  Rationale: SMARCB1 loss → EZH2 overactive → tazemetostat blocks EZH2 H3K27me3; "
        ),
        "inheritance": (
            "AD LOF 22q11.23 — SMARCB1. "
            "Germline heterozygous SMARCB1 → Schwannomatosis Type 2 (multiple schwannomas, adult). "
            "RTPS2 (Rhabdoid Tumour Predisposition Syndrome 2): germline het → childhood AT/RT/MRT (second hit LOH). "
            "De novo common in childhood AT/RT (~30-40%). "
            "Gonadal mosaicism described — recurrence risk higher than de novo rate suggests. "
            "Penetrance for rhabdoid tumours: ~15-25% in carriers; schwannomatosis: adult-onset, high penetrance."
        ),
        "surveillance_key": "sibling MRI + US at birth; tazemetostat for INI1-loss tumours; INI1 IHC on all paediatric CNS tumours",
        "pathognomonic": "AT/RT in child under 3; INI1 nuclear IHC loss in tumour",
    },
    {
        "gene": "LZTR1",
        "protein": (
            "LZTR1 -- 22q11.21 Autosomal-Dominant-OR-AR -- 827aa -- "
            "CUL3-Adaptor-RAS-Ubiquitination-RAS-MAPK-Regulation-"
            "Schwannomatosis-Type1-SCHWT1-AD-Dominant-Negative-OR-AR-Biallelic-"
            "Painful-Schwannomas-No-Skin-Findings-Test-Both-Parents-Also-Noonan-10-OMIM-600574"
        ),
        "locus": "22q11.21",
        "protein_size": (
            "827 aa / 22q11.21 LZTR1 encodes leucine zipper-like transcription regulator 1: "
            "STRUCTURE: BTB-POZ domain (CUL3 E3 ligase adaptor binding); Kelch repeats (substrate recognition); "
            "  LZTR1 acts as CUL3 adaptor to ubiquitinate and degrade RAS family proteins (KRAS, MRAS, RRAS); "
            "  LZTR1 LOF → impaired RAS degradation → elevated RAS-GTP → RAS-MAPK hyperactivation; "
            "UNIQUE BIDIRECTIONAL INHERITANCE — SCHWANNOMATOSIS-1: "
            "  AD dominant-negative: heterozygous truncating/missense → schwannomatosis (adult multiple schwannomas); "
            "    Pathogenic variants cluster in BTB domain — interfere with wild-type LZTR1 CUL3 complex; "
            "  AR biallelic: homozygous or compound het → Noonan syndrome type 10 (congenital); "
            "    AR Noonan: facial features, pulmonary stenosis, short stature, +/- schwannomas in adult life; "
            "  CRITICAL: ALWAYS TEST BOTH PARENTS — AD transmission probability 50%; AR 25%; "
            "SCHWANNOMATOSIS TYPE 1 (LZTR1): "
            "  Multiple benign schwannomas; NO bilateral VS (differentiates from NF2); "
            "  PAINFUL — chronic pain is the hallmark; often dorsal root ganglia + peripheral nerves; "
            "  No café-au-lait macules; no neurofibromas; slit-lamp normal; "
            "  Management: pain management (pregabalin, gabapentin); surgery for symptomatic/growing lesions; "
            "NOONAN SYNDROME 10 (AR LZTR1): "
            "  Same surveillance as Noonan (cardiac, growth, haematology for JMML); "
            "22q11.21 LOCATION: "
            "  LZTR1 at 22q11.21 — same region as NF2 (22q12.2) and SMARCB1 (22q11.23); "
            "  Chromosome 22 schwannomatosis cluster: SMARCB1-LZTR1 deletions sometimes co-occur; "
        ),
        "inheritance": (
            "UNIQUE BIDIRECTIONAL: AD dominant-negative (schwannomatosis-1, heterozygous) "
            "OR AR biallelic (Noonan syndrome type 10). "
            "ALWAYS test both parents to determine transmission mode. "
            "AD: 50% recurrence; AR: 25% recurrence. "
            "Same gene, same allele → different syndrome depending on zygosity. "
            "Schwannomatosis-1 penetrance: ~60-80% by age 70."
        ),
        "surveillance_key": "test both parents; pain management; surgery for growing schwannomas; AR → Noonan cardiac surveillance",
        "pathognomonic": "multiple painful schwannomas; no skin findings; no bilateral VS",
    },
    {
        "gene": "SPRED1",
        "protein": (
            "SPRED1 -- 15q14 Autosomal-Dominant-LOF -- 444aa -- "
            "Sprouty-Related-EVH1-Domain-RAS-MAPK-Negative-Regulator-RAF1-Inhibition-"
            "Legius-Syndrome-NF1-Like-WITHOUT-Tumours-Without-Lisch-Without-Plexiform-"
            "NF1-CRITICAL-DDx-No-MPNST-Risk-OMIM-611431"
        ),
        "locus": "15q14",
        "protein_size": (
            "444 aa / 15q14 SPRED1 encodes sprouty-related EVH1 domain-containing protein 1: "
            "STRUCTURE: N-terminal EVH1 domain (Enabled/VASP homology 1); SPR domain (sprouty-related); "
            "  SPRED1 binds NF1 and RAF1 to localise NF1 to the RAS-RAF complex; "
            "  SPRED1 LOF → NF1 not recruited to membrane → impaired RAS-GTP hydrolysis → "
            "    persistent MAPK activation via RAF1 → café-au-lait macules + developmental features; "
            "  Note: SPRED1-LOF is LESS severe than NF1-LOF because NF1 protein still present; "
            "LEGIUS SYNDROME (SPRED1): "
            "  ≥6 café-au-lait macules (same size criteria as NF1) + axillary/inguinal freckling; "
            "  NO neurofibromas; NO plexiform neurofibromas; NO Lisch nodules; "
            "  NO optic pathway glioma; NO MPNST; "
            "  Macrocephaly (50%); learning difficulties/ADHD (similar to NF1); "
            "  Noonan-like features reported in some; "
            "CRITICAL NF1 DIFFERENTIAL DIAGNOSIS: "
            "  SPRED1 testing MUST be offered to families meeting NF1 café-au-lait criteria "
            "    but with no neurofibromas, no Lisch nodules, no glioma; "
            "  Distinguishing Legius from NF1 is life-altering: "
            "    NF1: surveillance for MPNST, optic glioma, cardiovascular, learning; "
            "    Legius: NO tumour surveillance beyond standard; NO radiation restriction; "
            "  Practice: NF1 gene sequencing first (most common); SPRED1 if NF1 negative + mild phenotype; "
            "PREVALENCE: ~2% of patients referred for NF1 diagnosis have SPRED1 instead; "
            "NO MALIGNANCY RISK: SPRED1/Legius does not increase cancer risk significantly; "
        ),
        "inheritance": (
            "AD LOF 15q14 — SPRED1 / Legius syndrome. "
            "~50% de novo. Penetrance for café-au-lait: high (>90%). "
            "Variable expressivity: some family members have only café-au-lait without freckling. "
            "Key distinguishing feature from NF1: absence of Lisch nodules + absence of neurofibromas. "
            "Do NOT diagnose NF1 in a patient with SPRED1 pathogenic variant — different syndrome, different management."
        ),
        "surveillance_key": "diagnose precisely — NO MPNST surveillance; NO radiation restriction; learning/ADHD support; annual BP",
        "pathognomonic": "café-au-lait ≥6 WITHOUT neurofibromas AND WITHOUT Lisch nodules",
    },
    {
        "gene": "SMARCE1",
        "protein": (
            "SMARCE1 -- 17q21.2 Autosomal-Dominant-LOF -- 411aa -- "
            "SWI-SNF-BAF-Complex-Subunit-BAF57-Chromatin-Remodelling-"
            "Familial-Multiple-Spinal-Meningiomas-Young-Women-PATHOGNOMONIC-"
            "Clear-Cell-Histology-Full-Spine-MRI-Mandatory-OMIM-603111"
        ),
        "locus": "17q21.2",
        "protein_size": (
            "411 aa / 17q21.2 SMARCE1 encodes SWI/SNF-related matrix-associated actin-dependent "
            "regulator of chromatin subfamily E member 1 (BAF57): "
            "STRUCTURE: HMG (high-mobility group) box for DNA binding; "
            "  BAF57 is a structural SWI/SNF complex subunit linking BAF complex to DNA and transcription factors; "
            "  SMARCE1 LOF → disrupted BAF complex chromatin remodelling → loss of tumour suppression "
            "    in meningeal and spinal cord tissue; "
            "FAMILIAL MULTIPLE SPINAL MENINGIOMAS: "
            "  Predominantly SPINAL meningiomas — cervical, thoracic, lumbar segments; "
            "  Young age at diagnosis (20s-40s); female predominance (F:M ~3:1 in some series); "
            "  CLEAR-CELL histology (WHO grade 2 behaviour despite clear-cell appearance); "
            "  Multiple simultaneous meningiomas — unlike sporadic (usually solitary); "
            "  Intracranial meningiomas also occur in some SMARCE1 families; "
            "  NO other NF features (no schwannomas, no café-au-lait, no VS); "
            "SURVEILLANCE: "
            "  FULL-SPINE MRI at diagnosis (often multiple lesions at presentation); "
            "  Brain MRI at diagnosis; "
            "  Annual spine MRI monitoring (growth rate guides surgery timing); "
            "  Surgery: symptomatic or growing lesions; radiosurgery controversial for clear-cell WHO2; "
            "SMARCE1 IHC: "
            "  SMARCE1 IHC nuclear loss in tumour confirms somatic second hit; "
            "  Clear-cell spinal meningioma in young patient → reflexly test SMARCE1; "
            "COMPARISON: "
            "  SMARCB1 → rhabdoid + schwannomatosis; SMARCE1 → meningioma only (spinal-predominant); "
            "  Both are SWI/SNF complex subunits — sister genes with different tissue predispositions; "
        ),
        "inheritance": (
            "AD LOF 17q21.2 — SMARCE1. "
            "Familial multiple spinal meningiomas: recognise family history of spinal surgery / meningioma. "
            "Penetrance: high for meningioma; incomplete penetrance for number of lesions. "
            "De novo cases described. "
            "Critical clue: young patient + clear-cell meningioma + spinal location → test SMARCE1."
        ),
        "surveillance_key": "full-spine MRI + brain MRI at diagnosis; annual spine MRI; surgery for symptomatic/growing",
        "pathognomonic": "multiple spinal meningiomas; young age; clear-cell histology; female predominance",
    },
    {
        "gene": "PRKAR1A",
        "protein": (
            "PRKAR1A -- 17q24.2 Autosomal-Dominant-LOF -- 381aa -- "
            "Protein-Kinase-A-Regulatory-Subunit-1-Alpha-cAMP-PKA-Pathway-"
            "Carney-Complex-CNC-Spotty-Pigmentation-PATHOGNOMONIC-Cardiac-Myxoma-30pct-LIFE-THREATENING-"
            "PPNAD-Cushing-Psammomatous-Schwannomas-Annual-Echo-MANDATORY-OMIM-188830"
        ),
        "locus": "17q24.2",
        "protein_size": (
            "381 aa / 17q24.2 PRKAR1A encodes protein kinase A (PKA) regulatory subunit Iα: "
            "STRUCTURE: Two cAMP-binding domains (CBD-A, CBD-B); dimerisation/docking domain (D/D); "
            "  In resting state R1α homodimer sequesters two PKA catalytic subunits (R2C2 tetramers); "
            "  cAMP binding → R1α dissociation → free catalytic subunits phosphorylate CREB, TORC; "
            "  PRKAR1A LOF → constitutive PKA activity → sustained cAMP signalling; "
            "CARNEY COMPLEX (CNC): "
            "  SPOTTY SKIN PIGMENTATION (PATHOGNOMONIC): "
            "    Lentigines (perioral, periocular, genitals, conjunctiva); "
            "    Epithelioid blue nevi; myxomatous fibroadenomas; "
            "  CARDIAC MYXOMA (30% CNC): "
            "    LIFE-THREATENING — embolism (stroke), obstruction; "
            "    Annual echocardiogram MANDATORY in all PRKAR1A carriers from diagnosis; "
            "    BILATERAL/MULTIPLE/RECURRENT myxomas — unlike sporadic (usually solitary); "
            "    Early excision recommended; "
            "  PRIMARY PIGMENTED NODULAR ADRENOCORTICAL DISEASE (PPNAD): "
            "    Bilateral micronodular adrenocortical hyperplasia → ACTH-independent Cushing; "
            "    Paradoxical Liddle test: cortisol rises with dexamethasone (PATHOGNOMONIC for PPNAD); "
            "    Annual morning cortisol + 24h UFC; "
            "  PSAMMOMATOUS MELANOTIC SCHWANNOMAS (PMSch): "
            "    Distinct from NF-type schwannomas; contains melanin + psammoma bodies; "
            "    Spinal/paravertebral; 10% malignant transformation; "
            "  GH-SECRETING PITUITARY ADENOMA: acromegaly in ~10%; IGF-1 annually; "
            "  THYROID NODULES: 75% CNC patients; annual thyroid USS; "
            "  TESTICULAR: large cell calcifying Sertoli cell tumour (LCCSCT) — annual testicular USS in males; "
            "SURVEILLANCE SCHEDULE: "
            "  Annual echocardiogram (MANDATORY, cardiac myxoma); "
            "  Annual cortisol/UFC (PPNAD); "
            "  Annual thyroid USS; "
            "  Annual IGF-1 (acromegaly); "
            "  Annual testicular USS (males); "
            "  Annual skin exam (spotty pigmentation + fibromyxoid tumours); "
        ),
        "inheritance": (
            "AD LOF 17q24.2 — PRKAR1A / Carney Complex. "
            "~50% de novo. High penetrance (>90% will have at least one CNC feature by age 50). "
            "PRKAR1A mutations found in ~70% CNC families; some CNC families map to 2p16 (CNC2, gene unknown). "
            "Genotype-phenotype: truncating → loss of mRNA (NMD) → more severe; "
            "  Missense in cAMP-binding domain → altered regulation → variable phenotype. "
            "Annual echocardiogram — NEVER skip — cardiac myxoma has embolic mortality."
        ),
        "surveillance_key": "annual echo MANDATORY; annual cortisol; annual thyroid USS; annual IGF-1; testicular USS males; skin exam",
        "pathognomonic": "spotty skin pigmentation (lentigines + blue nevi); bilateral/recurrent cardiac myxoma",
    },
    {
        "gene": "AKT1",
        "protein": (
            "AKT1 -- 14q32.33 Somatic-Postzygotic-GOF-Mosaic -- 480aa -- "
            "PI3K-AKT-mTOR-Pathway-Serine-Threonine-Kinase-"
            "Proteus-Syndrome-CCTN-PATHOGNOMONIC-Hemihyperplasia-DVT-PE-Risk-"
            "AKT1-p.E17K-Somatic-Mosaic-Most-Common-Alpelisib-Emerging-OMIM-614752"
        ),
        "locus": "14q32.33",
        "protein_size": (
            "480 aa / 14q32.33 AKT1 encodes RAC-alpha serine/threonine-protein kinase (AKT1): "
            "STRUCTURE: PH domain (pleckstrin homology — binds PIP3); kinase domain; regulatory domain; "
            "  AKT1 is activated by PI3K → PDK1 phosphorylates AKT T308; mTORC2 phosphorylates S473; "
            "  AKT1 GOF p.E17K: gain-of-function missense in PH domain → "
            "    enhanced PIP3 binding affinity → constitutive membrane localisation → "
            "    AKT hyperactivation independent of upstream PI3K; "
            "PROTEUS SYNDROME: "
            "  Caused EXCLUSIVELY by somatic postzygotic (mosaic) AKT1 p.E17K; "
            "  Germline AKT1 p.E17K would be lethal — only mosaic survives; "
            "  Diagnosis requires: "
            "    (A) Mosaic distribution (one-sided or segmental features), AND "
            "    (B) Sporadic (no family history), AND "
            "    (C) Progressive (worsening with age), AND "
            "    (D) Specific feature present: "
            "CEREBRIFORM CONNECTIVE TISSUE NAEVUS (CCTN) PATHOGNOMONIC: "
            "  Overgrowth of plantar/palmar connective tissue — cerebriform (brain-like) folds; "
            "  Virtually diagnostic of Proteus when present; "
            "HEMIHYPERPLASIA: "
            "  Asymmetric overgrowth of limbs, trunk, skull; "
            "  Macrodactyly; epidermal naevi; subcutaneous masses; "
            "DEEP VEIN THROMBOSIS / PULMONARY EMBOLISM: "
            "  Major mortality cause in Proteus syndrome; "
            "  DVT + PE from vascular malformations and immobility; "
            "  Prophylactic anticoagulation during immobility/surgery; "
            "  Annual compression duplex ultrasound of affected limbs; "
            "MOLECULAR TESTING: "
            "  Require mosaic tissue (biopsy affected tissue — not blood); "
            "  Blood often negative or very low VAF (variant allele frequency); "
            "  Ultra-deep sequencing (>1000x) on affected tissue; "
            "  Droplet digital PCR (ddPCR) for low VAF mosaic detection; "
            "TREATMENT: "
            "  mTOR inhibitor sirolimus/everolimus: empiric use for vascular malformations; "
            "  Alpelisib (PI3K-alpha inhibitor): emerging evidence — MOSAIC AKT1 trials underway; "
            "  PIK3CA/AKT pathway inhibitors: active trials in PROS spectrum; "
            "RELATED CONDITIONS (PROS = PIK3CA-Related Overgrowth Spectrum): "
            "  AKT1 p.E17K is Proteus; PIK3CA mosaic GOF → MCAP, CLOVES, FEP, etc.; "
            "  AKT3 mosaic GOF → HMEG, MCAP — overlapping with TSC/mTOR spectrum; "
        ),
        "inheritance": (
            "NOT HERITABLE in classical sense — Proteus syndrome is caused by SOMATIC/POSTZYGOTIC mosaic "
            "AKT1 p.E17K. Germline heterozygous AKT1 p.E17K likely lethal (not observed in live births). "
            "Recurrence risk: near-zero for future children of affected individual "
            "  (unless mosaic in germline cells — rare but described). "
            "Diagnosis requires biopsy of affected tissue + ultra-deep sequencing; blood usually insufficient. "
            "IMPORTANT: do NOT test blood only and call negative — test affected tissue."
        ),
        "surveillance_key": "mosaic tissue biopsy required; DVT/PE prophylaxis; annual duplex; alpelisib/sirolimus trials; CCTN PATHOGNOMONIC",
        "pathognomonic": "cerebriform connective tissue naevus (CCTN); hemihyperplasia mosaic distribution",
    },
]


def _gene_stats(seed: int, gene_config: dict) -> dict:
    """Generate per-gene statistics for one gene using a fixed seed."""
    rng = random.Random(seed)
    gene = gene_config["gene"]

    # Base rates vary by gene
    base = {
        "NF1":     {"neurofibroma": 95, "cafe_au_lait": 98, "plexiform": 42, "mpnst": 10, "learning": 55, "optic_glioma": 17},
        "NF2":     {"bilateral_vs": 96, "meningioma": 62, "ependymoma": 18, "lens_opacity": 45, "hearing_loss": 88},
        "SMARCB1": {"schwannoma": 72, "atrt": 22, "mrhabdoid": 12, "ini1_loss_ihc": 98, "pain": 65},
        "LZTR1":   {"schwannoma": 85, "painful_schwannoma": 78, "no_skin_findings": 95, "noonan_ar": 15},
        "SPRED1":  {"cafe_au_lait": 96, "freckling": 68, "no_neurofibroma": 100, "no_lisch": 100, "adhd": 40},
        "SMARCE1": {"spinal_meningioma": 92, "multiple_meningioma": 78, "clear_cell": 65, "female_predominance": 72},
        "PRKAR1A": {"spotty_pigmentation": 85, "cardiac_myxoma": 32, "ppnad": 45, "schwannoma": 28, "acromegaly": 10},
        "AKT1":    {"cctn": 72, "hemihyperplasia": 88, "dvt": 38, "epidermal_naevi": 52, "macrodactyly": 64},
    }.get(gene, {})

    n = 40
    age_mean = {
        "NF1": 18, "NF2": 28, "SMARCB1": 12, "LZTR1": 34,
        "SPRED1": 8, "SMARCE1": 38, "PRKAR1A": 30, "AKT1": 5,
    }.get(gene, 25)

    stats = {
        "gene": gene,
        "n": n,
        "seed": seed,
        "mean_age_diagnosis": round(age_mean + rng.gauss(0, 2.5), 1),
        "female_pct": round(rng.uniform(45, 65), 1),
    }

    for feature, base_rate in base.items():
        rate = max(0, min(100, base_rate + rng.gauss(0, 4)))
        stats[f"{feature}_pct"] = round(rate, 1)

    # Common additional stats
    stats["genetic_testing_positive_pct"] = round(rng.uniform(88, 99), 1)
    stats["surveillance_adherent_pct"] = round(rng.uniform(65, 85), 1)
    stats["family_history_positive_pct"] = round(rng.uniform(40, 65), 1)
    stats["de_novo_pct"] = round(rng.uniform(35, 60), 1)
    return stats


def generate_overview() -> dict:
    """Overview data for Hereditary-Neurofibromatosis-Schwannomatosis-Atlas."""
    return {
        "atlas":          "Hereditary-Neurofibromatosis-Schwannomatosis-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Neurofibromatosis and Schwannomatosis Predisposition Atlas "
            "(NF1-NF2-SMARCB1-LZTR1-SPRED1-SMARCE1-PRKAR1A-AKT1)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "NF1": (
                "AD LOF 17q11.2 (Neurofibromin-1; 2839aa; NF Type 1; 1:3000 most common; "
                "café-au-lait PATHOGNOMONIC ≥6 macules; Lisch nodules; plexiform neurofibromas; "
                "MPNST 8-13% lifetime; selumetinib-FDA2020 for paediatric plexiform; "
                "AVOID RADIATION — radiation-induced MPNST; 50% de novo)"
            ),
            "NF2": (
                "AD LOF 22q12.2 (Merlin; 595aa; NF Type 2; 1:25000; "
                "BILATERAL VS PATHOGNOMONIC; meningioma 50-75%; ependymoma 18%; "
                "juvenile lens opacity PATHOGNOMONIC; bevacizumab for progressive VS; "
                "~30% somatic mosaic — asymmetric presentation)"
            ),
            "SMARCB1": (
                "AD LOF 22q11.23 (INI1/BAF47; 385aa; Schwannomatosis-2 + RTPS2; "
                "AT/RT PATHOGNOMONIC in children <3yr; INI1 IHC nuclear loss PATHOGNOMONIC; "
                "multiple schwannomas in adults; sibling surveillance; "
                "tazemetostat FDA2020 for INI1-loss tumours)"
            ),
            "LZTR1": (
                "BIDIRECTIONAL AD/AR 22q11.21 (CUL3 adaptor; 827aa; Schwannomatosis-1; "
                "AD dominant-negative → schwannomatosis; AR biallelic → Noonan-10; "
                "PAINFUL schwannomas; NO skin findings; NO bilateral VS; "
                "TEST BOTH PARENTS — AD 50% vs AR 25% recurrence risk)"
            ),
            "SPRED1": (
                "AD LOF 15q14 (SPRED1; 444aa; Legius Syndrome; NF1-like WITHOUT tumours; "
                "café-au-lait ≥6 + freckling BUT NO neurofibromas NO Lisch NO MPNST; "
                "CRITICAL NF1 DDx — different management; ~2% of NF1-referred patients; "
                "NO radiation restriction; NO MPNST surveillance)"
            ),
            "SMARCE1": (
                "AD LOF 17q21.2 (BAF57; 411aa; Familial Multiple Spinal Meningiomas; "
                "SPINAL meningiomas PATHOGNOMONIC in young women; clear-cell histology WHO2; "
                "full-spine MRI MANDATORY at diagnosis; "
                "NO schwannomas NO café-au-lait NO VS)"
            ),
            "PRKAR1A": (
                "AD LOF 17q24.2 (PKA-Rα; 381aa; Carney Complex; "
                "SPOTTY PIGMENTATION PATHOGNOMONIC (lentigines + blue nevi); "
                "CARDIAC MYXOMA 30% LIFE-THREATENING — annual echo MANDATORY; "
                "PPNAD Cushing; psammomatous schwannomas; acromegaly; "
                "cAMP/PKA pathway)"
            ),
            "AKT1": (
                "SOMATIC MOSAIC GOF 14q32.33 (AKT1; 480aa; Proteus Syndrome; "
                "CEREBRIFORM CONNECTIVE TISSUE NAEVUS PATHOGNOMONIC; "
                "hemihyperplasia; DVT/PE major mortality cause; "
                "p.E17K somatic mosaic — test AFFECTED TISSUE not blood; "
                "alpelisib/sirolimus emerging; NOT inherited)"
            ),
        },
        "key_clinical_rules": [
            "NF1: MPNST risk 8-13% — annual full-body exam; urgent MRI if rapid growth or pain change in existing neurofibroma (malignant transformation sign)",
            "NF1: selumetinib (Koselugo) FDA2020 — for symptomatic, inoperable plexiform neurofibromas in children ≥2yr with NF1",
            "NF1: AVOID RADIATION — radiation-induced MPNST in radiation field documented; use proton therapy or surgery over RT",
            "NF2: BILATERAL VS = germline NF2 until proven otherwise — test ALL bilateral VS regardless of age",
            "NF2: bevacizumab for growing VS — hearing preservation goal; annual audiology + annual brain+spine MRI",
            "SMARCB1: AT/RT in child <3yr → germline SMARCB1 testing + sibling surveillance (brain MRI + abdominal US at birth)",
            "SMARCB1: INI1 IHC nuclear loss in any tumour → check SMARCB1 germline — rhabdoid predisposition diagnosis",
            "LZTR1: ALWAYS test both parents — AD dominant-negative vs AR biallelic determines recurrence risk (50% vs 25%)",
            "LZTR1: PAINFUL multiple schwannomas without skin findings and without bilateral VS → LZTR1 schwannomatosis-1 (not NF2)",
            "SPRED1: café-au-lait ≥6 + NO neurofibromas + NO Lisch → test SPRED1; diagnosis = Legius (NOT NF1); NO MPNST surveillance",
            "SMARCE1: young patient + clear-cell spinal meningioma → full-spine MRI + test SMARCE1; multiple spinal lesions common",
            "PRKAR1A: annual echocardiogram MANDATORY in ALL carriers — cardiac myxoma embolises causing stroke; bilateral/recurrent myxoma = PRKAR1A",
            "PRKAR1A: paradoxical Liddle dexamethasone test (cortisol RISES with dexamethasone) = PATHOGNOMONIC for PPNAD",
            "AKT1: Proteus syndrome — test AFFECTED TISSUE (biopsy), NOT blood; ultra-deep sequencing >1000x; blood VAF often too low to detect",
            "AKT1: DVT/PE is major mortality cause in Proteus — DVT prophylaxis during immobility; compression duplex surveillance annually",
        ],
        "gene_panel_note": (
            "Neurofibromatosis and Schwannomatosis panel (clinical 2024): "
            "NF TYPE 1: NF1 (café-au-lait + neurofibromas + Lisch + MPNST risk); "
            "NF TYPE 2: NF2 (bilateral VS + meningioma + ependymoma + lens opacity); "
            "NF1-LIKE WITHOUT TUMOURS: SPRED1 / Legius (café-au-lait + freckling ONLY); "
            "SCHWANNOMATOSIS: SMARCB1 (Type 2, rhabdoid risk), LZTR1 (Type 1, painful); "
            "SPINAL MENINGIOMA: SMARCE1 (young women, clear-cell, multiple spinal); "
            "CARNEY COMPLEX: PRKAR1A (spotty pigmentation + cardiac myxoma + PPNAD); "
            "PROTEUS SYNDROME: AKT1 (somatic mosaic, CCTN, hemihyperplasia); "
            "CLINICAL DECISION TREE: "
            "  Café-au-lait ≥6 + neurofibromas/Lisch → NF1; "
            "  Café-au-lait ≥6 only (no neurofibromas) → NF1 first, then SPRED1; "
            "  Bilateral VS → NF2; "
            "  Multiple painful schwannomas, no VS → LZTR1 then SMARCB1; "
            "  AT/RT in child <3yr → SMARCB1 germline; "
            "  Young woman + multiple spinal meningiomas → SMARCE1; "
            "  Spotty pigmentation + cardiac myxoma → PRKAR1A (Carney Complex); "
            "  Cerebriform plantar naevus + hemihyperplasia → AKT1 (Proteus, mosaic tissue); "
            "SURVEILLANCE TIERS: "
            "  All NF1: annual exam, BP, ophthalmology; "
            "  All NF2: annual audiology + MRI; "
            "  All PRKAR1A: annual echo (non-negotiable); "
            "  All SMARCB1: paediatric → sibling surveillance if parent affected"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Neurofibromatosis-Schwannomatosis-Atlas."""
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
        "atlas":        "Hereditary-Neurofibromatosis-Schwannomatosis-Atlas",
        "seed_range":   f"{SEED_BASE}-{SEED_BASE + 7}",
        "n_genes":      len(genes_data),
        "total_patients": 320,
        "genes":        genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Neurofibromatosis-Schwannomatosis-Atlas."""
    definitions = [
        {
            "term": "NF1-Diagnostic-Criteria-and-Surveillance-Protocol",
            "definition": (
                "NIH Diagnostic Criteria for NF1 (1987, revised 2021 — ≥2 of 7 + molecular): "
                "1. ≥6 café-au-lait macules: >5mm (prepubertal) / >15mm (postpubertal) — single most common feature; "
                "2. ≥2 neurofibromas (any type) OR ≥1 plexiform neurofibroma; "
                "3. Axillary or inguinal freckling (Crowe's sign); "
                "4. Optic pathway glioma (optic nerve / chiasm / tract); "
                "5. ≥2 Lisch nodules (iris hamartomas) on slit-lamp; "
                "6. Distinctive osseous lesion (sphenoid wing dysplasia, tibial pseudarthrosis / bowing); "
                "7. First-degree relative with NF1 (by above criteria); "
                "2021 ADDITION: pathogenic NF1 variant on sequencing counts as criterion 7; "
                "PAEDIATRIC NF1 SURVEILLANCE: "
                "  Birth — 1yr: check for obvious neurofibromas; "
                "  Annual ophthalmology (visual acuity + fundoscopy) age 1-7 (OPG risk); "
                "  Annual height, weight, head circumference, BP; "
                "  Neurodevelopmental assessment at school entry; "
                "  MRI brain + orbits only if: visual symptoms, headache, focal deficit; "
                "ADULT NF1 SURVEILLANCE: "
                "  Annual full-body skin exam + palpation for new neurofibromas; "
                "  Annual BP; "
                "  MPNST watch: pain, rapid growth, firmness change in neurofibroma → urgent MRI; "
                "  FDG-PET: gold standard for MPNST detection in internal plexiform (SUVmax >3.5); "
                "SELUMETINIB (FDA2020): "
                "  MEK1/2 inhibitor; approved children ≥2yr with NF1 + symptomatic inoperable plexiform; "
                "  Dosing: 25mg/m² BD (max 50mg BD); dose-hold for toxicity; "
                "  Main side effects: rash, nausea, GI toxicity, CPK elevation; "
                "  Efficacy: 66% partial response (≥20% volume reduction) in SPRINT trial; "
                "MPNST TREATMENT: "
                "  Surgery (wide excision) primary; R0 margin critical; "
                "  Adjuvant RT: controversial in NF1 (AVOID RADIATION policy applies at primary treatment too); "
                "  Chemotherapy: doxorubicin + ifosfamide (limited response ~20%); "
                "  MEK inhibitors: trials underway for unresectable MPNST; "
                "CASCADE TESTING: first-degree relatives; NF1 del/dup → MLPA essential."
            ),
        },
        {
            "term": "NF2-Surveillance-and-Bevacizumab-Protocol",
            "definition": (
                "NF2 Management (2024 international consensus): "
                "DIAGNOSIS: "
                "  Manchester criteria: Bilateral VS (any age) = definitive; "
                "  OR FDR NF2 + unilateral VS <30yr OR ≥2 NF2-associated tumours; "
                "ANNUAL SURVEILLANCE: "
                "  MRI brain (full brain) + MRI spine (cervical minimum; extend thoracolumbar if symptomatic): annually; "
                "  Pure tone audiogram + speech discrimination score: annually from diagnosis; "
                "  Auditory brainstem response (ABR): annually; "
                "  Slit-lamp (posterior subcapsular lens opacity, retinal hamartoma): annually; "
                "  Neurological examination: annually; "
                "VS MANAGEMENT DECISION: "
                "  Small, stable VS: active monitoring with annual MRI + audiology; "
                "  Growing VS: surgical resection OR stereotactic radiosurgery (SRS) OR bevacizumab; "
                "  BEVACIZUMAB (anti-VEGF): "
                "    IV bevacizumab 7.5-10mg/kg q3 weeks; "
                "    36% hearing improvement in NFCT trial; tumour volume stabilisation 40-50%; "
                "    Side effects: hypertension, proteinuria, wound healing delay, thromboembolic risk; "
                "    Not FDA-approved specifically for NF2-VS; used compassionately/trial-based; "
                "  SRS (Gamma Knife / CyberKnife): hearing preservation 50-70% at 5yr; "
                "    Growing edge: marginal dose ≤12Gy (hearing preservation vs control tradeoff); "
                "COCHLEAR IMPLANT / ABI: "
                "  Cochlear implant: when serviceable hearing remains and tumour manageable; "
                "  Auditory brainstem implant (ABI): post-deafness surgery; 70-80 electrode array; "
                "  ABI gives environmental sound awareness but limited speech without lipreading; "
                "EMERGING THERAPIES: "
                "  MEK inhibitor selumetinib: Phase 2 trials for NF2 schwannomas; "
                "  FAK inhibitor defactinib: preclinical rationale (merlin regulates FAK signalling); "
                "  LAM (lapatinib + apatinib): Phase 2 for NF2-VS; "
                "GENETIC TESTING: sequencing + del/dup; mosaic NF2 (~30%) may require tumour tissue testing."
            ),
        },
        {
            "term": "SMARCB1-RTPS-AT-RT-and-Schwannomatosis-Protocol",
            "definition": (
                "SMARCB1 / RTPS2 Management (rhabdoid tumour predisposition + schwannomatosis): "
                "RHABDOID TUMOUR PREDISPOSITION: "
                "  INDICATIONS FOR GERMLINE TESTING: "
                "    AT/RT in child <3yr (mandatory); "
                "    MRT (malignant rhabdoid tumour) at any age; "
                "    Multiple synchronous tumours with INI1 loss; "
                "    Family history of rhabdoid tumour; "
                "  SIBLING SURVEILLANCE (if parent or proband confirmed SMARCB1 germline): "
                "    Brain MRI at birth then 3-monthly until age 5; "
                "    Abdominal ultrasound at birth then 3-monthly until age 5; "
                "    Reduces mortality by early detection; "
                "AT/RT TREATMENT: "
                "  High-intensity multimodal: surgery + chemotherapy + radiation; "
                "  ICE (ifosfamide/carboplatin/etoposide) + IT methotrexate; "
                "  RT: deferred in infants <3yr (neurotoxicity); craniospinal for >3yr; "
                "  Tazemetostat (EZH2 inhibitor, FDA2020): "
                "    Approved for epithelioid sarcoma (SMARCB1 LOF); "
                "    AT/RT trials ongoing; rationale: EZH2 overactive when INI1 lost; "
                "INI1 IHC: "
                "  Pathology reflex: ALL paediatric CNS tumours → INI1 IHC; "
                "  Loss of nuclear INI1 staining → rhabdoid mechanism → SMARCB1 germline test; "
                "  Retained INI1 essentially rules out SMARCB1 inactivation as driver; "
                "ADULT SCHWANNOMATOSIS TYPE 2 (SMARCB1): "
                "  Multiple schwannomas, predominantly truncal/spinal; "
                "  Pain management: gabapentin, pregabalin, tricyclics; "
                "  Surgery for symptomatic or growing lesions; "
                "  Annual MRI of affected region + full spine; "
                "22q11.23 MOLECULAR NOTES: "
                "  SMARCB1 is adjacent to NF2 on chromosome 22; "
                "  Constitutional 22q11 deletions may encompass both genes; "
                "  Somatic mosaicism: some tumours have both germline hit + somatic LOH."
            ),
        },
        {
            "term": "LZTR1-Schwannomatosis-Bidirectional-Inheritance-Protocol",
            "definition": (
                "LZTR1 Management (Schwannomatosis Type 1 + Noonan Syndrome 10): "
                "INHERITANCE WORKUP — MANDATORY: "
                "  ALWAYS sequence BOTH parents before counselling proband; "
                "  Parent has SAME heterozygous variant: likely AD (dominant-negative); "
                "    Recurrence risk to future children: 50%; "
                "  Both parents negative: de novo dominant; recurrence risk ~1%; "
                "  Parent has SAME variant + other parent has DIFFERENT variant in LZTR1: "
                "    Consider AR compound heterozygosity for Noonan if child has Noonan features; "
                "  True AR LZTR1 Noonan: compound het or homozygous; "
                "    If proband = Noonan, check if sibling also AR → 25% recurrence; "
                "SCHWANNOMATOSIS-1 (AD LZTR1): "
                "  Multiple peripheral schwannomas: spinal nerve roots, extremity nerves; "
                "  PAIN is hallmark: often severe, constant; may precede MRI-visible tumour; "
                "  NO café-au-lait macules; NO bilateral VS; NO Lisch nodules; "
                "  Distinguishing from NF2: no bilateral VS; no meningioma; no ependymoma; "
                "  MRI whole spine + brain at diagnosis; annual spine MRI; "
                "  Surgery: for symptomatic, growing schwannomas; "
                "  Pain management: gabapentin/pregabalin; tramadol; TENS; "
                "NOONAN SYNDROME 10 (AR LZTR1): "
                "  Same features as other Noonan types: facial features, short stature, CHD (usually PS), "
                "    JMML risk (low), cryptorchidism, learning difficulties; "
                "  Surveillance as per Noonan protocol: cardiac, GH, JMML screen; "
                "  Adult: possible schwannoma development (monitoring required); "
                "LZTR1 GENE CONTEXT: "
                "  22q11.21 — proximal to SMARCB1 (22q11.23) and NF2 (22q12.2); "
                "  CUL3-LZTR1 E3 ligase ubiquitinates RAS proteins → LZTR1 is RAS regulator; "
                "  LOF → impaired RAS degradation → MAPK hyperactivation in Schwann cells."
            ),
        },
        {
            "term": "SPRED1-Legius-Syndrome-NF1-Differential-Diagnosis",
            "definition": (
                "SPRED1 / Legius Syndrome Management (NF1-critical differential): "
                "DISTINGUISHING LEGIUS FROM NF1: "
                "  SHARED with NF1: ≥6 café-au-lait macules (same size criteria); axillary/inguinal freckling; "
                "    macrocephaly; learning difficulties/ADHD; Noonan-like features; "
                "  ABSENT in Legius (SPRED1): "
                "    Neurofibromas (cutaneous or plexiform) — NEVER in Legius; "
                "    Lisch nodules — NEVER in Legius; "
                "    Optic pathway glioma — NEVER described in Legius; "
                "    MPNST — NEVER in Legius; "
                "    Bone lesions (sphenoid dysplasia, pseudarthrosis); "
                "CLINICAL TESTING ALGORITHM: "
                "  Step 1: NF1 full gene sequencing + MLPA (most common); "
                "  Step 2 (if NF1 negative): SPRED1 sequencing; "
                "  Step 3 (if both negative): clinical re-evaluation; consider mosaic NF1; "
                "  Approximately 2% of NF1-referred patients have SPRED1 instead; "
                "MANAGEMENT IMPLICATIONS (CRITICAL DIFFERENCE): "
                "  Legius: NO MPNST surveillance; NO MRI surveillance; NO radiation restriction; "
                "  Legius: standard paediatric follow-up (learning support; BP monitoring); "
                "  NF1: MPNST surveillance; MRI for plexiform/OPG; AVOID RADIATION; selumetinib eligibility; "
                "  Diagnosing Legius instead of NF1 = CORRECT management prevents unnecessary anxiety + over-investigation; "
                "MOLECULAR MECHANISM: "
                "  SPRED1 interacts with NF1 protein (neurofibromin) to recruit it to the plasma membrane; "
                "  SPRED1 LOF → NF1 not recruited → RAS-GTP not hydrolysed → MAPK activated; "
                "  Less severe than NF1 LOF because NF1 protein is intact (just mislocated); "
                "FAMILY TESTING: "
                "  Cascade testing first-degree relatives; "
                "  If parent identified: counsel re 50% recurrence; "
                "  No need for enhanced surveillance in family beyond café-au-lait monitoring."
            ),
        },
        {
            "term": "SMARCE1-Multiple-Spinal-Meningioma-Surveillance",
            "definition": (
                "SMARCE1 Management (Familial Multiple Spinal Meningiomas): "
                "CLINICAL RECOGNITION: "
                "  Young adult (20s-40s) with: spinal meningioma, especially clear-cell histology; "
                "  Multiple simultaneous spinal meningiomas; female predominance (F:M ~3:1); "
                "  Family history of meningioma / spinal surgery; "
                "  NO skin findings; NO schwannomas; NO café-au-lait; NO hearing loss; "
                "INITIAL WORKUP AT DIAGNOSIS: "
                "  Full-spine MRI (cervical + thoracic + lumbar) with gadolinium — MANDATORY; "
                "    Multiple lesions common — may be asymptomatic; "
                "  Brain MRI with gadolinium: intracranial meningiomas also possible; "
                "  SMARCE1 IHC on tumour tissue: SMARCE1 nuclear loss confirms somatic second hit; "
                "SURVEILLANCE: "
                "  Annual full-spine MRI for known SMARCE1 carriers; "
                "  Annual brain MRI (or 2-yearly if no intracranial disease at diagnosis); "
                "  Neurological examination: gait, power, reflexes, bladder/bowel annually; "
                "SURGERY INDICATIONS: "
                "  Symptomatic meningioma (cord compression, radiculopathy); "
                "  Progressive growth on serial MRI; "
                "  Clear-cell meningioma: WHO grade 2 behaviour — higher recurrence than grade 1; "
                "  Post-surgical re-growth common (SMARCE1 field effect — remaining Schwann/meningeal cells at risk); "
                "RADIOSURGERY: "
                "  SRS for small residual or recurrent intracranial meningiomas; "
                "  Spinal SRS: limited evidence; cord tolerance constraint critical; "
                "MOLECULAR NOTES: "
                "  SMARCE1 (BAF57) is part of SWI/SNF complex distinct from SMARCB1 (INI1); "
                "  Both are SWI/SNF subunits but different tissue tropism; "
                "  SMARCE1 LOF → chromatin remodelling loss in meningeal cells → meningioma predisposition; "
                "  NF2 somatic mutations present in SMARCE1-associated meningiomas (co-operative LOF)."
            ),
        },
        {
            "term": "PRKAR1A-Carney-Complex-Annual-Surveillance-Protocol",
            "definition": (
                "PRKAR1A / Carney Complex Management (annual surveillance mandatory): "
                "SPOTTY SKIN PIGMENTATION: "
                "  Lentigines: small (<5mm), dark brown/black spots; perioral, eyelid, conjunctival, genital; "
                "  Epithelioid blue nevi: raised, blue-black; "
                "  Myxomatous fibroadenoma of breast: multiple, bilateral; "
                "  Pigmented lesions often lighten with age — diagnosis may rest on other features; "
                "CARDIAC MYXOMA — ANNUAL ECHO MANDATORY: "
                "  30% of CNC patients; BILATERAL, MULTIPLE, or RECURRENT (unlike sporadic solitary); "
                "  Can involve any cardiac chamber; left atrial most common; "
                "  LIFE-THREATENING: tumour fragmentation → systemic embolism (stroke, limb ischaemia); "
                "    Acute heart failure from obstruction; "
                "  ANNUAL transthoracic echocardiogram in ALL PRKAR1A carriers — no exceptions; "
                "  Surgical excision: urgently once identified; local recurrence common → annual post-op echo; "
                "PPNAD (Primary Pigmented Nodular Adrenocortical Disease): "
                "  45% CNC; bilateral micronodular adrenal cortex; ACTH-independent Cushing; "
                "  PARADOXICAL LIDDLE TEST: give high-dose dexamethasone → cortisol PARADOXICALLY RISES >50%; "
                "    Pathognomonic for PPNAD — used to confirm diagnosis; "
                "  Screening: annual 24h urinary free cortisol + midnight salivary cortisol; "
                "  Treatment: bilateral adrenalectomy for active Cushing; "
                "PSAMMOMATOUS MELANOTIC SCHWANNOMA: "
                "  10% CNC; distinct from NF-type schwannomas — contains melanin + psammoma bodies; "
                "  Spinal/paravertebral; 10% malignant transformation; "
                "  Annual MRI spine for known PRKAR1A carriers; "
                "GH-SECRETING PITUITARY ADENOMA: "
                "  10% CNC; acromegaly with elevated IGF-1; "
                "  Annual IGF-1; MRI pituitary at baseline; octreotide or surgery; "
                "THYROID NODULES: "
                "  75% CNC; most benign; annual thyroid USS + FNA if suspicious; "
                "TESTICULAR LCCSCT (males): "
                "  Large cell calcifying Sertoli cell tumour; bilateral; annual testicular USS from childhood; "
                "ANNUAL SURVEILLANCE SUMMARY: "
                "  Echocardiogram (MANDATORY); 24h UFC; thyroid USS; IGF-1; "
                "  Testicular USS (males); MRI spine; skin exam."
            ),
        },
        {
            "term": "AKT1-Proteus-Syndrome-Mosaic-Diagnosis-Management",
            "definition": (
                "AKT1 p.E17K / Proteus Syndrome Management (somatic mosaic, not inherited): "
                "DIAGNOSTIC CRITERIA (Biesecker 2006, updated): "
                "  OBLIGATORY: connective tissue naevus (cerebriform or other) + mosaic/progressive/sporadic; "
                "  GENERAL (any two of three): "
                "    A. Epidermal naevus; "
                "    B. Disproportionate overgrowth (limb length discrepancy >3cm; macrodactyly; macrocephaly if isolated); "
                "    C. Specific tumours (bilateral ovarian cystadenoma; parotid monomorphic adenoma <age 20); "
                "  SPECIFIC (any one of): "
                "    CEREBRIFORM CONNECTIVE TISSUE NAEVUS (CCTN): plantar/palmar; brain-sulcus-like folds — PATHOGNOMONIC; "
                "    Epidermal naevus (linear, verrucous); "
                "    Vascular malformations; "
                "    Disproportionate overgrowth of: vertebra (scoliosis), skull, limb, viscera; "
                "MOLECULAR DIAGNOSIS: "
                "  AKT1 p.E17K somatic mosaic — present in >95% confirmed Proteus; "
                "  MUST biopsy affected tissue (CCTN, overgrown limb skin, epidermal naevus); "
                "  Blood test: unreliable (low VAF — often <1% in blood); "
                "  Ultra-deep sequencing (>1000x coverage) or ddPCR on affected tissue; "
                "DVT/PE PREVENTION — HIGH PRIORITY: "
                "  DVT and PE are LEADING CAUSE OF DEATH in Proteus syndrome; "
                "  Mechanism: venous anomalies + limb overgrowth + immobility; "
                "  Annual duplex compression ultrasound of legs; "
                "  DVT prophylaxis during any immobility, hospitalisation, surgery; "
                "  Discuss anticoagulation plan with haematology; "
                "ORTHOPAEDIC: "
                "  Serial orthopaedic assessment; limb-length equalisation; "
                "  Scoliosis surveillance (spinal X-ray annually from diagnosis); "
                "  Epiphysiodesis for limb-length discrepancy; "
                "TREATMENT — TARGETED THERAPY: "
                "  Sirolimus (mTOR inhibitor): empiric use — reduces overgrowth/vascular malformation activity; "
                "  Alpelisib (PI3K-alpha inhibitor): Phase 2 trials for AKT1-mosaic/PROS spectrum; "
                "    Rationale: AKT1 p.E17K activates PI3K-AKT-mTOR; PI3K inhibition upstream; "
                "  PTEN/AKT pathway: "
                "    AKT1 sits downstream of PI3K; alpelisib blocks PIK3CA → reduces AKT phosphorylation; "
                "GENETIC COUNSELLING: "
                "  Proteus is NOT heritable in standard sense (germline AKT1 p.E17K = lethal); "
                "  Recurrence risk for parents: near-zero; "
                "  EXCEPTION: if proband has high-level mosaicism (high VAF in blood), gonadal mosaicism possible; "
                "    Recurrence risk in this scenario: low but non-zero; "
                "  Diagnose PRECISELY — avoid over-diagnosing Proteus in Klippel-Trenaunay, neurofibromatosis."
            ),
        },
        {
            "term": "Neurocutaneous-Syndrome-Differential-Diagnosis-Guide",
            "definition": (
                "Differential diagnosis guide for neurocutaneous and hamartoma syndromes: "
                "CAFÉ-AU-LAIT MACULES (≥6): "
                "  NF1: ≥6 CAL + any second criterion; test NF1 first; "
                "  Legius (SPRED1): ≥6 CAL + freckling, NO neurofibromas, NO Lisch; test after NF1 negative; "
                "  NF2: rare café-au-lait; not a NF2 criterion; "
                "  McCune-Albright: 'Coast of Maine' CAL + fibrous dysplasia + precocious puberty; mosaic GNAS; "
                "  Mosaic NF1: segmental CAL only; test from lesion biopsy; "
                "  Fanconi anaemia: CAL + bone marrow failure + DEB test / FANC genes; "
                "SCHWANNOMAS (multiple): "
                "  NF2: bilateral VS + meningioma + ependymoma; test NF2; "
                "  SMARCB1 schwannomatosis-2: multiple schwannomas + no bilateral VS; pain; "
                "  LZTR1 schwannomatosis-1: painful schwannomas + no bilateral VS; test LZTR1; "
                "  Carney complex (PRKAR1A): psammomatous melanotic schwannomas + spotty pigmentation; "
                "CARDIAC MYXOMA: "
                "  Sporadic: solitary; left atrial; one-off; postoperative cure; "
                "  Carney Complex (PRKAR1A): bilateral / multiple / recurrent myxoma; spotty pigmentation; "
                "CHILDHOOD CNS TUMOUR: "
                "  AT/RT (INI1 loss): SMARCB1 germline test; "
                "  Medulloblastoma (SHH): PTCH1 first (Gorlin); SUFU if PTCH1 negative; TP53 for WNT/Li-Fraumeni; "
                "  Optic glioma: NF1; "
                "ASYMMETRIC OVERGROWTH: "
                "  Proteus syndrome (AKT1 p.E17K mosaic): CCTN + progressive + sporadic; test affected tissue; "
                "  PIK3CA-mosaic PROS: MCAP, CLOVES, FEP; "
                "  Hemihyperplasia + Wilms: WT1 / BWS (IGF2/CDKN1C / 11p15); "
                "  NF1: plexiform neurofibromas may cause limb overgrowth; "
                "MULTIPLE MENINGIOMAS: "
                "  NF2: bilateral VS + meningioma; annual brain+spine MRI; "
                "  SMARCE1: spinal-predominant; young women; clear-cell; no VS; "
                "  Radiation-induced: prior brain RT history; "
                "  Sporadic multiple: rare; somatic NF2 in each; "
                "SPOTTY SKIN PIGMENTATION: "
                "  Carney Complex (PRKAR1A): lentigines (not typical CAL) + blue nevi + myxoma; "
                "  Peutz-Jeghers (STK11): perioral lentigines + GI polyps + high cancer risk; "
                "  LEOPARD/Noonan (PTPN11, RAF1): lentigines + CHD + short stature + Noonan features."
            ),
        },
    ]
    return {
        "atlas": "Hereditary-Neurofibromatosis-Schwannomatosis-Atlas",
        "count": len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["genes"][0], indent=2)[:1500])
    print("\n=== DEFINITIONS (first entry) ===")
    df = generate_definitions()
    print(json.dumps(df["definitions"][0], indent=2)[:1500])
