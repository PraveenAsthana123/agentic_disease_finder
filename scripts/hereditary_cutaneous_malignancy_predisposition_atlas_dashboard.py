#!/usr/bin/env python3
"""Hereditary-Cutaneous-Malignancy-Predisposition-Atlas -- Complete 8-Gene Reference
PTCH1   (Patched-1; 1447aa; 9q22.32; AD LOF;
         Gorlin Syndrome / NBCCS — BCC 100s-1000s lifetime PATHOGNOMONIC;
         Odontogenic keratocyst (OKC) PATHOGNOMONIC; calcified falx PATHOGNOMONIC;
         AVOID RADIATION ABSOLUTELY — BCC explosion post-RT;
         seed SEED_BASE+0) .
MSH2    (MutS Homolog 2; 935aa; 2p21; AD LOF;
         Muir-Torre Syndrome — sebaceous carcinoma/adenoma PATHOGNOMONIC;
         Lynch syndrome type 2 — sebaceous IHC MANDATORY cascade trigger;
         EPCAM 3-prime deletion — silent MSH2 silencing; MLPA MANDATORY;
         seed SEED_BASE+1) .
XPC     (Xeroderma Pigmentosum Complementation Group C; 940aa; 3p25.1; AR LOF;
         XP group C — most common XP type; SCC/BCC/melanoma 10,000x;
         NO neurodegeneration (XPC-specific — nucleotide excision repair global genome only);
         SUNLIGHT ABSOLUTE CI; strict photoprotection birth;
         seed SEED_BASE+2) .
ERCC2   (Excision Repair Cross-Complementation Group 2 / XPD; 760aa; 19q13.32; AR LOF;
         XP group D — SCC/BCC + progressive neurodegeneration (Cockayne overlap);
         de Sanctis-Cacchione syndrome (severe end); TFIIH subunit helicase;
         SUNLIGHT ABSOLUTE CI; neurodegeneration surveillance MANDATORY;
         seed SEED_BASE+3) .
CYLD    (Cylindromatosis Tumour Suppressor; 956aa; 16q12.1; AD LOF;
         Brooke-Spiegler syndrome — cylindromas PATHOGNOMONIC (scalp "turban tumor");
         Spiradenomas + trichoepitheliomas; malignant transformation 5-10%;
         Annual dermatological survey MANDATORY; MTOR inhibitors emerging;
         seed SEED_BASE+4) .
TP53    (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome — cutaneous SCC/BCC 5-10x, Bowen's disease elevated;
         AVOID RADIATION ABSOLUTELY; WB-MRI Toronto Protocol ANNUALLY;
         seed SEED_BASE+5) .
SUFU    (Suppressor of Fused; 484aa; 10q24.32; AD LOF;
         SHH pathway — adult BCC 5-10x (distinct from childhood brain tumor role);
         Meningioma predisposition; less severe than PTCH1; emerging gene;
         Vismodegib/sonidegib active (SHH pathway inhibition);
         seed SEED_BASE+6) .
PTCH2   (Patched-2; 1203aa; 1p32.3; AD LOF;
         Gorlin variant — BCC 5-20x elevated (less severe than PTCH1);
         Medulloblastoma rare (less than PTCH1); OKC less common;
         Annual dermatological examination MANDATORY from age 20yr;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3470-3477)
"""
import random

SEED_BASE = 3470

ATLAS_GENES = [
    {
        "gene": "PTCH1",
        "protein": (
            "PTCH1 -- 9q22.32 Autosomal-Dominant-LOF -- 1447aa -- "
            "Patched-1-Receptor-161kDa-12TM-SHH-Inhibitor-"
            "Gorlin-NBCCS-BCC-100s-1000s-Lifetime-OKC-PATHOGNOMONIC-"
            "Calcified-Falx-PATHOGNOMONIC-AVOID-RADIATION-ABSOLUTELY-"
            "Vismodegib-FDA2012-Sonidegib-FDA2015-OMIM-601309"
        ),
        "locus": "9q22.32",
        "protein_size": (
            "1447 aa / 161 kDa / 9q22.32 PTCH1 cutaneous cancer molecular context: "
            "STRUCTURE: "
            "  1447 aa / 161 kDa; 12-transmembrane domain Patched receptor; "
            "  Sterol-sensing domain (SSD): aa 450-550; mediates cholesterol-linked SMO inhibition; "
            "  PTCH1 tonically inhibits Smoothened (SMO) — hedgehog OFF state in basal epidermis; "
            "  SHH binding releases SMO → GLI1/2 activation → basal cell proliferation; "
            "MOLECULAR MECHANISM OF BCC: "
            "  Loss of PTCH1 → SMO constitutively active → GLI1/GLI2 nuclear translocation; "
            "  GLI1 target genes: CCND1, BCL2, SNAIL — proliferation + anti-apoptosis; "
            "  Radiation induces second somatic hit (LOH at 9q22) → BCC explosion post-RT; "
            "  OKC mechanism: jaw epithelium proliferation via unchecked GLI signalling; "
            "KEY MUTATIONS: "
            "  Frameshift/truncating (50%): most common — complete loss; "
            "  Missense SSD domain (25%): R680H, R816Q — SMO interaction disrupted; "
            "  Splice site 5% — exon skipping; "
            "  Somatic second hit: usually 9q22 LOH (~60%) or somatic point mutation; "
            "CANCER RISKS (lifetime): "
            "  BCC: 100s-1000s (median onset 20-30yr vs 60yr sporadic); "
            "  Medulloblastoma: desmoplastic 5% (first 5 years of life); "
            "  OKC (odontogenic keratocyst): jaw 74% (PATHOGNOMONIC); "
            "KEY MANAGEMENT: "
            "  Annual full-body skin examination from puberty MANDATORY; "
            "  AVOID ALL RADIATION — even diagnostic RT triggers BCC eruption; "
            "  Vismodegib (150mg/d) FDA2012 — BCC first systemic approval; "
            "  Sonidegib (200mg/d) FDA2015 — SMO inhibitor alternative; "
            "  Annual dental OPG from age 8yr (OKC surveillance); "
        ),
        "key_mutations": [
            {"variant": "c.1273del (frameshift)", "protein_effect": "Leu425fs — complete PTCH1 loss", "location": "Exon 8", "phenotype": "Classic Gorlin, early BCC onset"},
            {"variant": "p.Arg816Gln (missense)", "protein_effect": "SSD domain disruption, SMO not suppressed", "location": "Exon 16", "phenotype": "Variable expressivity, BCC + OKC"},
            {"variant": "p.Arg680His (missense)", "protein_effect": "SSD cholesterol-binding disruption", "location": "Exon 13", "phenotype": "Moderate, BCC + occasional meningioma"},
        ],
        "surveillance": [
            "Annual full-body skin exam from puberty",
            "Annual dental OPG from age 8yr (OKC)",
            "Annual echocardiogram in childhood (cardiac fibroma)",
            "Brain MRI in first 5yr of life (desmoplastic medulloblastoma)",
            "Ophthalmology annual (coloboma, glaucoma)",
        ],
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 935aa -- "
            "MutS-Homolog-2-105kDa-MutSalpha-MutSbeta-MMR-"
            "Muir-Torre-Sebaceous-Carcinoma-PATHOGNOMONIC-Lynch-Type2-"
            "Urothelial-14pct-HIGHEST-EPCAM-Silencing-MLPA-MANDATORY-"
            "Pembrolizumab-FDA2017-MSI-H-OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "935 aa / 105 kDa / 2p21 MSH2 cutaneous cancer molecular context: "
            "STRUCTURE: "
            "  935 aa / 105 kDa; MutSα (MSH2-MSH6) — mismatch recognition; "
            "  MutSβ (MSH2-MSH3) — insertion-deletion loop recognition; "
            "  ATPase domain N-terminal: fuels conformational change after mismatch binding; "
            "  Connector domain: MSH6/MSH3 heterodimerisation interface; "
            "MUIR-TORRE SYNDROME: "
            "  MSH2 most commonly mutated in Muir-Torre (~90%) vs MLH1 (~10%); "
            "  Sebaceous adenoma/carcinoma + internal malignancy = PATHOGNOMONIC Muir-Torre; "
            "  Sebaceous IHC: MSH2/MSH6 loss PATHOGNOMONIC — cascade germline testing MANDATORY; "
            "  Keratoacanthoma type (Muir-Torre variant): MSI-H large crateriform lesion; "
            "  MSI-H on sebaceous tumour: test ALL sebaceous tumours routinely regardless of age; "
            "KEY MUTATIONS (Muir-Torre-enriched): "
            "  p.Ala636Pro (c.1906G>C): Muir-Torre recurrent; MutSα interface; "
            "  p.Glu198Stop (c.592G>T): truncating; Lynch2/MT; "
            "  EPCAM 3-prime deletion: silences MSH2 via methylation — MSH2 coding intact but silenced; "
            "  MLPA at EPCAM 3-prime: MANDATORY (standard sequencing misses this deletion); "
            "CANCER RISKS (Muir-Torre/Lynch2): "
            "  Sebaceous carcinoma: rare (1 in 2M sporadic) — hereditary dominant pathway; "
            "  CRC: 25-40% lifetime; Endometrial: 30-40%; Urothelial: 10-14% (HIGHEST Lynch gene); "
            "  Ovarian: 5-10%; Gastric: 5%; Biliary: 2-4%; "
            "KEY MANAGEMENT: "
            "  Sebaceous IHC MSH2/MSH6 on ALL sebaceous tumours — no age threshold; "
            "  Annual colonoscopy from age 20-25yr; cystoscopy/cytology age 30-35yr; "
            "  Aspirin 600mg/day (CAPP2 — 50% CRC risk reduction Level A); "
        ),
        "key_mutations": [
            {"variant": "p.Ala636Pro (c.1906G>C)", "protein_effect": "MutSα dimerisation disrupted", "location": "Exon 12", "phenotype": "Muir-Torre enriched, sebaceous carcinoma"},
            {"variant": "EPCAM 3' deletion", "protein_effect": "MSH2 promoter methylation silencing (coding intact)", "location": "2p21 upstream", "phenotype": "Lynch2/MT, MSH2 protein absent IHC"},
            {"variant": "p.Glu198Stop", "protein_effect": "Truncation, NMD", "location": "Exon 4", "phenotype": "Classic Lynch2, CRC + endometrial dominant"},
        ],
        "surveillance": [
            "Sebaceous IHC MSH2/MSH6 on all sebaceous tumours (no age cut-off)",
            "Colonoscopy from age 20-25yr (2-yearly)",
            "Urine cytology + cystoscopy from age 30-35yr (annual)",
            "Gynaecological USS + endometrial biopsy from age 35yr (annual)",
            "Aspirin 600mg CAPP2 protocol",
        ],
    },
    {
        "gene": "XPC",
        "protein": (
            "XPC -- 3p25.1 Autosomal-Recessive-LOF -- 940aa -- "
            "Xeroderma-Pigmentosum-C-106kDa-GGR-NER-DNA-Damage-Sensor-"
            "SCC-BCC-Melanoma-10000x-NO-Neurodegeneration-XPC-SPECIFIC-"
            "SUNLIGHT-ABSOLUTE-CI-Strict-Photoprotection-Birth-"
            "OMIM-278720"
        ),
        "locus": "3p25.1",
        "protein_size": (
            "940 aa / 106 kDa / 3p25.1 XPC cutaneous cancer molecular context: "
            "STRUCTURE: "
            "  940 aa / 106 kDa; Global Genome NER (GGR-NER) lesion sensor; "
            "  Transglutaminase homology domain: DNA binding, TGH; "
            "  TFIIH-binding domain C-terminal: recruits TFIIH for lesion verification; "
            "  Forms RAD23B-XPC-CETN2 trimer — RAD23B stabilises XPC; "
            "XPC-SPECIFIC BIOLOGY: "
            "  XPC initiates GGR-NER (global genome): detects UV-induced CPDs/6-4PP genome-wide; "
            "  TC-NER (transcription-coupled) is INTACT in XPC patients — explains NO neurodegeneration; "
            "  Loss of GGR-NER: CPDs/6-4PP accumulate in non-transcribed DNA → SCC/BCC/melanoma; "
            "  NEURODEGENERATION ABSENT: TC-NER (ERCC2/CSB/CSA) protects neuronal cells; XPC has no TC-NER role; "
            "KEY MUTATIONS: "
            "  Poly-AT insertion IVS3 (c.1077-18A>G): African founder (~40% African XP-C); "
            "  p.Tyr776Stop (c.2328C>G): European truncating; "
            "  p.Arg579Stop: severe null, early carcinoma <5yr; "
            "CANCER RISKS: "
            "  SCC/BCC: 10,000x elevated vs age-matched population; "
            "  Cutaneous melanoma: 2,000x elevated; "
            "  Onset: first skin cancer median age 8yr (vs 60yr sporadic); "
            "  Ocular surface carcinoma (conjunctival): 10-15x; "
            "KEY MANAGEMENT: "
            "  Total UV avoidance from birth — 400nm cut-off UV-protective clothing; "
            "  Annual ophthalmological examination; "
            "  Monthly full-body skin examination; "
            "  Annual retinal UV screening; "
        ),
        "key_mutations": [
            {"variant": "c.1077-18A>G (IVS3 poly-AT)", "protein_effect": "African founder — exon skipping, truncation", "location": "Intron 3", "phenotype": "African XP-C, severe, early carcinoma"},
            {"variant": "p.Tyr776Stop (c.2328C>G)", "protein_effect": "C-terminal truncation, TFIIH binding lost", "location": "Exon 11", "phenotype": "European, XP-C classic"},
            {"variant": "p.Arg579Stop", "protein_effect": "Null, complete GGR-NER loss", "location": "Exon 9", "phenotype": "Severe, skin cancer <5yr"},
        ],
        "surveillance": [
            "Total UV avoidance from birth (400nm cut-off clothing, UV-blocking films)",
            "Monthly full-body skin examination (self + dermatologist)",
            "Annual ophthalmology (conjunctival carcinoma, corneal scarring)",
            "Annual oncology review",
            "Neurodevelopmental review (NOT neurodegeneration — XPC; confirm diagnosis vs ERCC2)",
        ],
    },
    {
        "gene": "ERCC2",
        "protein": (
            "ERCC2 -- 19q13.32 Autosomal-Recessive-LOF -- 760aa -- "
            "ERCC2-XPD-89kDa-TFIIH-XPD-Subunit-3prime-5prime-Helicase-"
            "XP-D-SCC-BCC-Melanoma-10000x-PLUS-Neurodegeneration-"
            "de-Sanctis-Cacchione-Cockayne-Overlap-SUNLIGHT-ABSOLUTE-CI-"
            "OMIM-278730"
        ),
        "locus": "19q13.32",
        "protein_size": (
            "760 aa / 89 kDa / 19q13.32 ERCC2 cutaneous cancer molecular context: "
            "STRUCTURE: "
            "  760 aa / 89 kDa; XPD subunit of TFIIH — 3'-5' helicase activity; "
            "  Iron-sulphur cluster domain (FeS): aa 302-340 — lesion verification helicase; "
            "  ARCH domain: aa 166-290 — DNA bubble opening coordination; "
            "  p44-binding C-terminal: stabilises TFIIH CAK module (MAT1/CDK7/cyclin H); "
            "ERCC2 DUAL ROLE — TFIIH IN BOTH NER AND TRANSCRIPTION: "
            "  NER: XPD helicase unwinds DNA around lesion — critical for both GGR and TC-NER; "
            "  Transcription: TFIIH promotes RNA Pol II elongation — neuronal transcription essential; "
            "  Loss of ERCC2 impairs BOTH NER (skin cancer) AND neuronal TFIIH (neurodegeneration); "
            "XPD GENOTYPE-PHENOTYPE CORRELATIONS: "
            "  Severe (null): XP + neurodegeneration (de Sanctis-Cacchione); "
            "  Moderate LOF: XP alone (SCC/BCC only, no neurodegeneration) — p.Arg683Trp type; "
            "  Partial LOF: COFS/Cockayne overlap — growth retardation + UV sensitivity; "
            "KEY MUTATIONS: "
            "  p.Arg683Trp (c.2047C>T): most common XPD worldwide — XP ± neurodegeneration; "
            "  p.Asp681Asn (c.2041G>A): less severe; "
            "  p.Ala282Val + c.2251G>A compound het: COFS overlap; "
            "CANCER RISKS: "
            "  SCC/BCC/melanoma: 10,000x (same order as XPC); "
            "  ADDITIONAL: neurological malignancy if de Sanctis-Cacchione overlap; "
            "KEY MANAGEMENT (distinguishing from XPC): "
            "  Check neurodegeneration: XPC = absent; ERCC2 = present in severe alleles; "
            "  Brain MRI baseline + annual if neurodegeneration features present; "
        ),
        "key_mutations": [
            {"variant": "p.Arg683Trp (c.2047C>T)", "protein_effect": "Helicase ATPase disrupted, most common XPD allele", "location": "Exon 22", "phenotype": "XP-D classic, SCC/BCC, ±neurodegeneration"},
            {"variant": "p.Asp681Asn (c.2041G>A)", "protein_effect": "FeS-proximal, partial helicase impairment", "location": "Exon 22", "phenotype": "Milder XP-D, later onset"},
            {"variant": "p.Ala282Val compound het", "protein_effect": "ARCH domain disruption, residual helicase 20%", "location": "Exon 8", "phenotype": "COFS/Cockayne overlap, early severe"},
        ],
        "surveillance": [
            "Total UV avoidance from birth",
            "Monthly full-body skin examination",
            "Annual neurodevelopmental assessment (ERCC2 neurodegeneration risk)",
            "Brain MRI baseline + annual if neurological features",
            "Annual ophthalmology",
        ],
    },
    {
        "gene": "CYLD",
        "protein": (
            "CYLD -- 16q12.1 Autosomal-Dominant-LOF -- 956aa -- "
            "Cylindromatosis-Tumour-Suppressor-109kDa-Deubiquitinase-NF-kB-"
            "Brooke-Spiegler-Cylindromas-PATHOGNOMONIC-Turban-Tumor-Scalp-"
            "Spiradenomas-Trichoepitheliomas-Malignant-5-10pct-Annual-Derm-MANDATORY-"
            "OMIM-132700"
        ),
        "locus": "16q12.1",
        "protein_size": (
            "956 aa / 109 kDa / 16q12.1 CYLD cutaneous cancer molecular context: "
            "STRUCTURE: "
            "  956 aa / 109 kDa; deubiquitinase (DUB); catalytic triad USP domain C-terminal; "
            "  3 CAP-Gly domains N-terminal (aa 1-350): microtubule binding; "
            "  Ubiquitin hydrolase (USP) domain: aa 590-956; K63-linked polyUb chains; "
            "CYLD MOLECULAR MECHANISM: "
            "  CYLD cleaves K63-linked polyubiquitin from TRAF2/TRAF6/RIP1 → NF-κB pathway inhibition; "
            "  LOF CYLD → constitutive NF-κB → BCL2 upregulation + hair follicle proliferation; "
            "  Cylindromas arise from outer root sheath of hair follicle — DUB loss in adnexal epithelium; "
            "  HDAC6 substrate: CYLD also deubiquitinates HDAC6 controlling aggresome pathway; "
            "  Second hit required (somatic): 16q12 LOH in tumour tissue; "
            "BROOKE-SPIEGLER CLINICAL SPECTRUM: "
            "  Classic Brooke-Spiegler: cylindromas + spiradenomas + trichoepitheliomas (all three); "
            "  Familial cylindromatosis: cylindromas only (same gene, modifier influence); "
            "  Multiple familial trichoepitheliomas: trichoepitheliomas only (same gene); "
            "  'Turban tumor': confluent scalp cylindromas — PATHOGNOMONIC Brooke-Spiegler; "
            "  Malignant transformation: 5-10% cylindromas → cylindrocarcinoma/spiradenocarcinoma; "
            "KEY MUTATIONS: "
            "  p.Arg758Stop (c.2272C>T): most common European — USP domain truncation; "
            "  p.Asp681Gly: catalytic Asp — DUB activity abolished; "
            "  Large deletions exons 10-11: 20% Brooke-Spiegler families; "
            "MANAGEMENT: "
            "  Annual full-body dermatological survey MANDATORY; "
            "  Surgical excision of rapidly enlarging/symptomatic lesions; "
            "  mTOR inhibitors (rapamycin topical/systemic): emerging data; "
        ),
        "key_mutations": [
            {"variant": "p.Arg758Stop (c.2272C>T)", "protein_effect": "USP domain truncation, DUB activity abolished", "location": "Exon 19", "phenotype": "Brooke-Spiegler classic, turban tumor"},
            {"variant": "p.Asp681Gly (catalytic)", "protein_effect": "Catalytic Asp lost, K63-deubiquitination abolished", "location": "Exon 17", "phenotype": "Classic, early onset, multiple adnexal tumours"},
            {"variant": "Deletion exons 10-11", "protein_effect": "CAP-Gly2 + intervening loss, destabilises DUB", "location": "Exon 10-11", "phenotype": "Familial cylindromatosis predominant"},
        ],
        "surveillance": [
            "Annual full-body dermatological survey (head-to-toe)",
            "Surgical excision of enlarging cylindromas (malignant transformation)",
            "Audiological assessment (periauricular cylindromas → canal obstruction)",
            "Ophthalmology if periorbital cylindromas",
            "Family cascade testing (first-degree relatives)",
        ],
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Guardian-of-Genome-LFS-Cutaneous-SCC-BCC-5-10x-"
            "Bowen-Disease-Elevated-AVOID-RADIATION-ABSOLUTELY-"
            "WB-MRI-Toronto-Protocol-ANNUALLY-MANDATORY-"
            "OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 cutaneous cancer molecular context: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; tetramer transcription factor; "
            "  Transactivation domain (TAD) I+II: aa 1-40 — MDM2 binding, p300/CBP; "
            "  Proline-rich domain: aa 40-92 — apoptosis vs cell-cycle arrest switch; "
            "  DNA binding domain (DBD): aa 94-292 — most hotspot mutations cluster here; "
            "  Tetramerisation domain: aa 325-356 — DN effect of germline LOF; "
            "LFS CUTANEOUS MANIFESTATIONS: "
            "  SCC/BCC: 5-10x elevated lifetime (secondary to radiation + UV accumulation); "
            "  Bowen's disease (SCC in situ): elevated — multiple lesions a LFS diagnostic clue; "
            "  Sarcomatoid transformation of pre-existing SCC/BCC: accelerated in LFS; "
            "  Radiation-induced skin cancer: explosive BCC/SCC post-RT (same mechanism as PTCH1); "
            "  Classic LFS dominant spectrum: sarcoma 50-60% + brain 25% + breast 25% + ACC 6%; "
            "KEY MUTATIONS: "
            "  p.Arg248Trp/Gln (c.742C>T/A): contact mutant + GOF; most common; "
            "  p.Arg175His (c.524G>A): structural + GOF; "
            "  p.Arg337His (c.1010G>A): Brazilian founder — APC exon 7-8 splice region; "
            "CRITICAL RULES: "
            "  AVOID RADIATION ABSOLUTELY — secondary malignancy acceleration + sarcoma; "
            "  WB-MRI Toronto Protocol: annual surveillance replaces CT/PET (no ionising radiation); "
            "  Informed consent before any X-ray / fluoroscopy procedure; "
        ),
        "key_mutations": [
            {"variant": "p.Arg248Trp (c.742C>T)", "protein_effect": "Contact mutant + dominant negative + GOF", "location": "DBD exon 7", "phenotype": "LFS classic, early SCC/BCC + sarcoma"},
            {"variant": "p.Arg175His (c.524G>A)", "protein_effect": "Structural mutant + GOF, p53 inactive", "location": "DBD exon 5", "phenotype": "LFS classic, aggressive phenotype"},
            {"variant": "p.Arg337His (c.1010G>A)", "protein_effect": "Tetramerisation disruption, Brazilian founder", "location": "Exon 10", "phenotype": "Brazilian LFS — ACC + skin cancer"},
        ],
        "surveillance": [
            "Annual whole-body MRI (WB-MRI) — Toronto Protocol (NO ionising radiation)",
            "Monthly skin self-examination + annual dermatological examination",
            "Informed radiation consent before any diagnostic X-ray",
            "Annual breast MRI from age 20-25yr (women)",
            "Annual abdominopelvic MRI",
        ],
    },
    {
        "gene": "SUFU",
        "protein": (
            "SUFU -- 10q24.32 Autosomal-Dominant-LOF -- 484aa -- "
            "Suppressor-of-Fused-54kDa-SHH-Negative-Regulator-GLI-Sequestration-"
            "Adult-BCC-5-10x-Meningioma-Predisposition-"
            "Less-Severe-than-PTCH1-Vismodegib-Sonidegib-Active-"
            "OMIM-607035"
        ),
        "locus": "10q24.32",
        "protein_size": (
            "484 aa / 54 kDa / 10q24.32 SUFU cutaneous cancer molecular context: "
            "STRUCTURE: "
            "  484 aa / 54 kDa; cytoplasmic SHH pathway suppressor; "
            "  N-terminal domain (aa 1-120): PTCH1-interacting; "
            "  Central SUFU domain (aa 120-390): GLI1/GLI2 sequestration; "
            "  C-terminal (aa 390-484): β-TrCP/proteasome-mediated GLI processing; "
            "SUFU vs PTCH1 IN SKIN CANCER: "
            "  Both suppress SHH signalling downstream of SHH ligand (PTCH1 = receptor; SUFU = nuclear effector); "
            "  SUFU loss → GLI1/GLI2 escape destruction → nuclear entry → BCC; "
            "  PTCH1 acts upstream (ligand recognition); SUFU acts downstream (nuclear effector); "
            "  SUFU germline BCC: typically adult onset (30-50yr); less severe than PTCH1 Gorlin; "
            "  Childhood medulloblastoma (SHH-MB): SUFU germline rare but documented (desmoplastic); "
            "CLINICAL SPECTRUM: "
            "  BCC: 5-10x elevated lifetime (sporadic BCC risk); "
            "  Meningioma: 3-5x elevated (novel SUFU finding 2020-2025); "
            "  OKC: rare (much less common than PTCH1); "
            "  Medulloblastoma in children (SHH type): 5-10% of SHH-MB have SUFU germline; "
            "TREATMENT: "
            "  Vismodegib/sonidegib: active (same SHH pathway); "
            "  Standard surgical excision for sporadic BCC (first-line); "
            "KEY MUTATIONS: "
            "  p.Trp535Stop (c.1605G>A): truncating, most common; "
            "  p.Arg390Cys (c.1168C>T): GLI-sequestration domain; "
            "  p.Leu412Pro: structural, destabilises SUFU fold; "
        ),
        "key_mutations": [
            {"variant": "p.Trp535Stop (c.1605G>A)", "protein_effect": "C-terminal truncation, GLI escape", "location": "Exon 10", "phenotype": "Adult BCC, occasional meningioma"},
            {"variant": "p.Arg390Cys (c.1168C>T)", "protein_effect": "GLI-sequestration domain disruption", "location": "Exon 7", "phenotype": "BCC + rare childhood SHH-MB"},
            {"variant": "p.Leu412Pro", "protein_effect": "Structural destabilisation, partial loss", "location": "Exon 7", "phenotype": "Milder BCC predisposition"},
        ],
        "surveillance": [
            "Annual full-body skin exam from age 20yr",
            "Annual dermatological exam with dermoscopy from age 30yr",
            "Brain MRI in childhood if SHH-MB features",
            "Annual neuroimaging from age 40yr (meningioma surveillance)",
            "Family cascade testing (first-degree relatives)",
        ],
    },
    {
        "gene": "PTCH2",
        "protein": (
            "PTCH2 -- 1p32.3 Autosomal-Dominant-LOF -- 1203aa -- "
            "Patched-2-Receptor-137kDa-12TM-SHH-Inhibitor-Gorlin-Variant-"
            "BCC-5-20x-Less-Severe-than-PTCH1-Medulloblastoma-Rare-OKC-Less-Common-"
            "Annual-Derm-MANDATORY-From-Age-20yr-"
            "OMIM-601309-variant"
        ),
        "locus": "1p32.3",
        "protein_size": (
            "1203 aa / 137 kDa / 1p32.3 PTCH2 cutaneous cancer molecular context: "
            "STRUCTURE: "
            "  1203 aa / 137 kDa; 12-TM receptor; PTCH1 paralogue (~57% identity); "
            "  Sterol-sensing domain (SSD): aa 380-490; less efficient SMO inhibition than PTCH1; "
            "  C-terminal cytoplasmic tail: shorter than PTCH1, less HHIP interaction; "
            "  PTCH2 expressed in skin/testes/cerebellum — more restricted than PTCH1; "
            "PTCH2 vs PTCH1 DIFFERENCES: "
            "  PTCH2 cannot fully compensate for PTCH1 loss → incomplete SHH buffering; "
            "  PTCH2 loss: less severe Gorlin phenotype (fewer BCC, later onset); "
            "  OKC rare in PTCH2 (OKC requires PTCH1 in most families); "
            "  Medulloblastoma rare in PTCH2 (vs 5% PTCH1); "
            "  Skin cancer: BCC 5-20x vs 100s-1000s PTCH1 — a key clinical distinction; "
            "KEY MUTATIONS: "
            "  p.Arg450Stop: C-terminal truncation most common; "
            "  p.Thr1189Met: near C-terminus; "
            "  Splice-site mutations exon 6-8 (30%); "
            "CLINICAL IMPACT: "
            "  PTCH2 carriers at lower absolute risk than PTCH1 — monitoring intensity adjusted; "
            "  Annual dermatological exam from age 20yr MANDATORY (less intensive than PTCH1); "
            "  No OKC surveillance required unless jaw symptoms; "
            "  SHH inhibitors (vismodegib) active if advanced BCC develops; "
        ),
        "key_mutations": [
            {"variant": "p.Arg450Stop (c.1348C>T)", "protein_effect": "C-terminal truncation, partial SSD intact", "location": "Exon 8", "phenotype": "Gorlin variant, BCC 5-20x adult onset"},
            {"variant": "p.Thr1189Met (c.3566C>T)", "protein_effect": "Near C-terminus structural disruption", "location": "Exon 22", "phenotype": "Milder phenotype, BCC-only"},
            {"variant": "Splice exon 6-7", "protein_effect": "Exon skipping, SSD partial loss", "location": "Intron 6-7", "phenotype": "Variable, BCC predisposition"},
        ],
        "surveillance": [
            "Annual full-body skin exam from age 20yr",
            "Dermoscopy annually from age 25yr",
            "Dental OPG if jaw symptoms (OKC rare but possible)",
            "Brain MRI only if neurological symptoms (MB very rare)",
            "Family cascade testing",
        ],
    },
]


def _patients_for_gene(gene_dict: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_dict["gene"]
    pts = []
    for i in range(40):
        age = int(rng.gauss(
            {"PTCH1": 32, "MSH2": 47, "XPC": 15, "ERCC2": 18, "CYLD": 38, "TP53": 35, "SUFU": 42, "PTCH2": 44}[gene],
            {"PTCH1": 8, "MSH2": 10, "XPC": 7, "ERCC2": 8, "CYLD": 9, "TP53": 10, "SUFU": 10, "PTCH2": 10}[gene]
        ))
        age = max(5, min(80, age))

        # Gene-specific cancer types and rates
        bcc_rate = {"PTCH1": 0.90, "MSH2": 0.10, "XPC": 0.70, "ERCC2": 0.65, "CYLD": 0.20, "TP53": 0.25, "SUFU": 0.40, "PTCH2": 0.35}[gene]
        scc_rate = {"PTCH1": 0.30, "MSH2": 0.05, "XPC": 0.80, "ERCC2": 0.75, "CYLD": 0.10, "TP53": 0.20, "SUFU": 0.20, "PTCH2": 0.15}[gene]
        sebaceous_rate = {"PTCH1": 0.02, "MSH2": 0.55, "XPC": 0.01, "ERCC2": 0.01, "CYLD": 0.05, "TP53": 0.05, "SUFU": 0.02, "PTCH2": 0.02}[gene]
        cylindroma_rate = {"PTCH1": 0.02, "MSH2": 0.02, "XPC": 0.01, "ERCC2": 0.01, "CYLD": 0.85, "TP53": 0.02, "SUFU": 0.02, "PTCH2": 0.02}[gene]
        has_bcc = rng.random() < bcc_rate
        has_scc = rng.random() < scc_rate
        has_sebaceous = rng.random() < sebaceous_rate
        has_cylindroma = rng.random() < cylindroma_rate
        has_malignancy = has_bcc or has_scc or has_sebaceous

        # Neurodegeneration: ERCC2 severe only
        neuro_rate = {"PTCH1": 0.0, "MSH2": 0.0, "XPC": 0.0, "ERCC2": 0.35, "CYLD": 0.0, "TP53": 0.0, "SUFU": 0.0, "PTCH2": 0.0}[gene]
        has_neuro = rng.random() < neuro_rate

        # Radiation contraindication flag
        radiation_ci = gene in ("PTCH1", "TP53")
        radiation_used = not radiation_ci and rng.random() < 0.12

        # OKC flag (PTCH1 mainly)
        okc_rate = {"PTCH1": 0.74, "MSH2": 0.0, "XPC": 0.0, "ERCC2": 0.0, "CYLD": 0.0, "TP53": 0.0, "SUFU": 0.05, "PTCH2": 0.08}[gene]
        has_okc = rng.random() < okc_rate

        pts.append({
            "gene": gene,
            "age": age,
            "has_bcc": has_bcc,
            "has_scc": has_scc,
            "has_sebaceous": has_sebaceous,
            "has_cylindroma": has_cylindroma,
            "has_malignancy": has_malignancy,
            "has_neuro": has_neuro,
            "has_okc": has_okc,
            "radiation_ci": radiation_ci,
            "radiation_used": radiation_used,
            "xp_type": gene in ("XPC", "ERCC2"),
        })
    return pts


def _all_patients() -> list:
    all_pts = []
    for idx, gene_dict in enumerate(ATLAS_GENES):
        all_pts.extend(_patients_for_gene(gene_dict, SEED_BASE + idx))
    return all_pts


# ── API generators ────────────────────────────────────────────────────────────
def generate_overview() -> dict:
    pts = _all_patients()
    gene_counts: dict = {}
    for p in pts:
        g = p["gene"]
        if g not in gene_counts:
            gene_counts[g] = {"gene": g, "n": 0, "bcc": 0, "scc": 0, "sebaceous": 0,
                               "cylindroma": 0, "malignancy": 0, "neuro": 0, "okc": 0, "mean_age": 0}
        gene_counts[g]["n"] += 1
        if p["has_bcc"]:         gene_counts[g]["bcc"] += 1
        if p["has_scc"]:         gene_counts[g]["scc"] += 1
        if p["has_sebaceous"]:   gene_counts[g]["sebaceous"] += 1
        if p["has_cylindroma"]:  gene_counts[g]["cylindroma"] += 1
        if p["has_malignancy"]:  gene_counts[g]["malignancy"] += 1
        if p["has_neuro"]:       gene_counts[g]["neuro"] += 1
        if p["has_okc"]:         gene_counts[g]["okc"] += 1
        gene_counts[g]["mean_age"] += p["age"]

    for g in gene_counts:
        n = gene_counts[g]["n"]
        gene_counts[g]["mean_age"] = round(gene_counts[g]["mean_age"] / n, 1)
        gene_counts[g]["malignancy_pct"] = round(100 * gene_counts[g]["malignancy"] / n, 1)
        gene_counts[g]["bcc_pct"] = round(100 * gene_counts[g]["bcc"] / n, 1)
        gene_counts[g]["scc_pct"] = round(100 * gene_counts[g]["scc"] / n, 1)

    bcc_total = sum(1 for p in pts if p["has_bcc"])
    scc_total = sum(1 for p in pts if p["has_scc"])
    seb_total = sum(1 for p in pts if p["has_sebaceous"])
    cyl_total = sum(1 for p in pts if p["has_cylindroma"])
    neuro_total = sum(1 for p in pts if p["has_neuro"])
    mal_total = sum(1 for p in pts if p["has_malignancy"])
    xp_total = sum(1 for p in pts if p["xp_type"])
    rad_ci_total = sum(1 for p in pts if p["radiation_ci"])

    return {
        "atlas":             "Hereditary-Cutaneous-Malignancy-Predisposition-Atlas",
        "atlas_id":          "hereditary-cutaneous-malignancy-predisposition-atlas",
        "subtitle":          "Complete 8-Gene PTCH1-MSH2-XPC-ERCC2-CYLD-TP53-SUFU-PTCH2 Reference",
        "total_patients":    len(pts),
        "gene_cohorts":      len(ATLAS_GENES),
        "seeds":             f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "bcc_cases":         bcc_total,
        "bcc_rate_pct":      round(100 * bcc_total / len(pts), 1),
        "scc_cases":         scc_total,
        "scc_rate_pct":      round(100 * scc_total / len(pts), 1),
        "sebaceous_cases":   seb_total,
        "cylindroma_cases":  cyl_total,
        "malignancy_cases":  mal_total,
        "malignancy_rate_pct": round(100 * mal_total / len(pts), 1),
        "neuro_cases":       neuro_total,
        "xp_patients":       xp_total,
        "radiation_ci_patients": rad_ci_total,
        "gene_summary":      list(gene_counts.values()),
        "key_clinical_rules": [
            "PTCH1 (Gorlin/NBCCS): AVOID ALL RADIATION ABSOLUTELY — even diagnostic RT triggers BCC explosion; vismodegib FDA2012 / sonidegib FDA2015; annual dental OPG from age 8yr (OKC 74%)",
            "MSH2 (Muir-Torre): sebaceous IHC MSH2/MSH6 MANDATORY on ALL sebaceous tumours (no age threshold); EPCAM 3-prime deletion — MSI-H sebaceous triggers Lynch germline workup; MLPA at EPCAM MANDATORY",
            "XPC / ERCC2 (Xeroderma Pigmentosum): SUNLIGHT ABSOLUTE CI from birth; SCC/BCC 10,000x elevated; XPC = NO neurodegeneration (GGR-NER only); ERCC2 = progressive neurodegeneration (TFIIH dual role)",
            "CYLD (Brooke-Spiegler): cylindromas PATHOGNOMONIC scalp ('turban tumor'); annual dermatological survey MANDATORY; malignant transformation 5-10%; mTOR inhibitors emerging",
            "TP53 (LFS): AVOID RADIATION ABSOLUTELY; WB-MRI Toronto Protocol annual (no ionising radiation); cutaneous SCC/BCC 5-10x; Bowen's disease elevated",
            "SUFU: adult BCC 5-10x (distinct from childhood SHH-MB role); meningioma 3-5x; less severe than PTCH1; vismodegib active; annual skin surveillance from age 20yr",
            "PTCH2: Gorlin variant — BCC 5-20x (much less severe than PTCH1); OKC rare; annual dermatological exam from age 20yr",
        ],
        "histology_distinctions": {
            "BCC_genes": "PTCH1 (highest, 1000s lifetime) > PTCH2 (5-20x) > SUFU (5-10x) > TP53 (5-10x LFS)",
            "SCC_genes": "XPC/ERCC2 (10,000x Xeroderma Pigmentosum) > TP53 (5-10x LFS) > MSH2 (Muir-Torre keratoacanthoma)",
            "sebaceous_gene": "MSH2 DOMINANT — Muir-Torre sebaceous carcinoma PATHOGNOMONIC (extremely rare sporadically)",
            "cylindroma_gene": "CYLD DOMINANT — Brooke-Spiegler; no other gene routinely causes cylindromas",
            "radiation_CI": "PTCH1 + TP53 = AVOID RADIATION ABSOLUTELY; XPC/ERCC2 = UV light CI (not ionising CI per se)",
        },
    }


def generate_breakdown() -> dict:
    pts = _all_patients()
    per_gene: dict = {}
    for p in pts:
        g = p["gene"]
        if g not in per_gene:
            per_gene[g] = {"gene": g, "n": 0, "bcc": 0, "scc": 0, "sebaceous": 0,
                            "cylindroma": 0, "malignancy": 0, "neuro": 0, "okc": 0, "ages": []}
        per_gene[g]["n"] += 1
        per_gene[g]["ages"].append(p["age"])
        if p["has_bcc"]:        per_gene[g]["bcc"] += 1
        if p["has_scc"]:        per_gene[g]["scc"] += 1
        if p["has_sebaceous"]:  per_gene[g]["sebaceous"] += 1
        if p["has_cylindroma"]: per_gene[g]["cylindroma"] += 1
        if p["has_malignancy"]: per_gene[g]["malignancy"] += 1
        if p["has_neuro"]:      per_gene[g]["neuro"] += 1
        if p["has_okc"]:        per_gene[g]["okc"] += 1

    syndrome_map = {
        "PTCH1":  "Gorlin/NBCCS",
        "MSH2":   "Muir-Torre/Lynch2",
        "XPC":    "Xeroderma Pigmentosum C",
        "ERCC2":  "Xeroderma Pigmentosum D (+Neuro)",
        "CYLD":   "Brooke-Spiegler",
        "TP53":   "Li-Fraumeni Syndrome",
        "SUFU":   "SHH-BCC/Gorlin-like",
        "PTCH2":  "Gorlin Variant",
    }
    key_avoid_map = {
        "PTCH1":  "RADIATION ABSOLUTELY (BCC explosion)",
        "MSH2":   "Skip sebaceous IHC screening",
        "XPC":    "SUNLIGHT ABSOLUTELY (SCC/BCC 10,000x)",
        "ERCC2":  "SUNLIGHT ABSOLUTELY; neurodegeneration risk",
        "CYLD":   "Delay excision of enlarging lesions",
        "TP53":   "RADIATION ABSOLUTELY; ionising X-ray consent",
        "SUFU":   "Delay annual skin surveillance",
        "PTCH2":  "Underestimate risk vs PTCH1",
    }
    key_rule_map = {
        "PTCH1":  "Vismodegib FDA2012 / annual dental OPG from age 8",
        "MSH2":   "Sebaceous IHC MSH2/MSH6 all sebaceous tumours; MLPA EPCAM",
        "XPC":    "Total UV avoidance from birth; monthly skin exam",
        "ERCC2":  "Annual neurodevelopmental assessment; brain MRI if neuro features",
        "CYLD":   "Annual full-body survey; excise enlarging cylindromas (malignant risk 5-10%)",
        "TP53":   "WB-MRI Toronto Protocol annually (no ionising radiation)",
        "SUFU":   "Annual skin exam from age 20yr; meningioma neuroimaging from 40yr",
        "PTCH2":  "Annual dermatological exam from age 20yr; SHH inhibitors if advanced BCC",
    }

    result = []
    for g, d in per_gene.items():
        n = d["n"]
        mean_age = round(sum(d["ages"]) / n, 1)
        result.append({
            "gene": g,
            "syndrome": syndrome_map.get(g, g),
            "n": n,
            "bcc_n": d["bcc"],
            "bcc_pct": round(100 * d["bcc"] / n, 1),
            "scc_n": d["scc"],
            "scc_pct": round(100 * d["scc"] / n, 1),
            "sebaceous_n": d["sebaceous"],
            "cylindroma_n": d["cylindroma"],
            "malignancy_n": d["malignancy"],
            "malignancy_pct": round(100 * d["malignancy"] / n, 1),
            "neuro_n": d["neuro"],
            "okc_n": d["okc"],
            "mean_age": mean_age,
            "key_avoid": key_avoid_map.get(g, ""),
            "key_rule": key_rule_map.get(g, ""),
        })

    return {
        "per_gene": result,
        "xp_comparison": {
            "XPC_vs_ERCC2": "XPC: GGR-NER only — NO neurodegeneration; ERCC2: TFIIH dual role — neurodegeneration in severe alleles; BOTH: SCC/BCC 10,000x elevated; distinguish by neurological examination",
            "XPC_onset": "Median first skin cancer age 8yr (range 3-15yr)",
            "ERCC2_onset": "Median first skin cancer age 10yr (range 4-18yr)",
            "XP_prevalence": "1 in 250,000 (Japan/N Africa higher — founder effects)",
        },
        "shh_pathway_comparison": {
            "PTCH1_vs_PTCH2": "PTCH1: severe (1000s BCC lifetime); PTCH2: mild (5-20x, later onset, OKC rare)",
            "PTCH1_vs_SUFU": "PTCH1 upstream receptor; SUFU downstream nuclear effector; SUFU milder BCC + meningioma; BOTH: vismodegib/sonidegib active",
            "radiation_CI": "PTCH1: AVOID RADIATION ABSOLUTELY (BCC explosion); SUFU/PTCH2: standard caution",
        },
        "muir_torre_diagnosis": {
            "trigger": "Sebaceous carcinoma at ANY age — cascade MSH2/MSH6 IHC mandatory",
            "epcam_rule": "Standard sequencing misses EPCAM 3' deletion silencing MSH2 — MLPA MANDATORY",
            "pembrolizumab": "MSI-H sebaceous carcinoma → pembrolizumab FDA2017 (tumour-agnostic)",
        },
        "cyld_natural_history": {
            "onset": "Cylindromas typically first decade, trichoepitheliomas second decade",
            "malignant_risk": "5-10% cylindrocarcinoma/spiradenocarcinoma — rapid growth = malignant warning",
            "surgical_principle": "Staged excisions; laser CO2 for multiple trichoepitheliomas",
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas":    "Hereditary-Cutaneous-Malignancy-Predisposition-Atlas",
        "atlas_id": "hereditary-cutaneous-malignancy-predisposition-atlas",
        "genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "full_description": g["protein"],
                "variants": g["key_mutations"],
                "surveillance_protocol": g["surveillance"],
            }
            for g in ATLAS_GENES
        ],
        "key_clinical_concepts": {
            "gorlin_nbccs": (
                "Gorlin Syndrome (Naevoid Basal-Cell Carcinoma Syndrome): PTCH1 LOF → constitutive SMO → GLI1/2 nuclear. "
                "BCC 100s-1000s lifetime (onset 20-30yr vs 60yr sporadic). "
                "OKC (odontogenic keratocyst) jaw 74% PATHOGNOMONIC. Calcified falx PATHOGNOMONIC. "
                "AVOID RADIATION ABSOLUTELY — even diagnostic RT triggers BCC eruption. "
                "Vismodegib 150mg/d FDA2012 / Sonidegib 200mg/d FDA2015 (oral SHH inhibitors). "
                "Annual dental OPG from age 8yr. Monthly skin exam."
            ),
            "muir_torre": (
                "Muir-Torre Syndrome: MSH2 (90%) or MLH1 (10%) LOF → sebaceous tumours + internal malignancy. "
                "Sebaceous carcinoma/adenoma PATHOGNOMONIC (1 in 2M sporadic rate). "
                "Sebaceous IHC MSH2/MSH6 MANDATORY on ALL sebaceous tumours — no age threshold. "
                "EPCAM 3-prime deletion silences MSH2 without coding mutation — MLPA MANDATORY. "
                "Pembrolizumab FDA2017 (tumour-agnostic MSI-H approval active for sebaceous carcinoma)."
            ),
            "xeroderma_pigmentosum": (
                "Xeroderma Pigmentosum (XP): AR NER gene defects → UV-DNA lesion accumulation → SCC/BCC/melanoma 10,000x. "
                "XPC (group C, ~25% XP): GGR-NER only — NO neurodegeneration. "
                "ERCC2/XPD (group D): TFIIH subunit — BOTH NER AND transcription → progressive neurodegeneration (de Sanctis-Cacchione). "
                "SUNLIGHT ABSOLUTE CI from birth — 400nm UV-protective clothing + films. "
                "First skin cancer median age 8yr. Monthly skin exam. Annual ophthalmology."
            ),
            "brooke_spiegler": (
                "Brooke-Spiegler Syndrome: CYLD (K63-deubiquitinase) LOF → NF-kB constitutive → adnexal tumours. "
                "Cylindromas PATHOGNOMONIC scalp ('turban tumor' confluent). "
                "Spiradenomas + trichoepitheliomas (same gene, variable expression). "
                "Malignant transformation 5-10% — rapid growth/ulceration = biopsy urgently. "
                "Annual full-body dermatological survey MANDATORY. mTOR inhibitors emerging (rapamycin)."
            ),
            "shh_pathway_inhibitors": (
                "SMO inhibitors (vismodegib/sonidegib): block SHH pathway downstream of PTCH1/PTCH2/SUFU LOF. "
                "Active in PTCH1/PTCH2 Gorlin BCC, SUFU-associated BCC, sporadic BCC (PTCH1 somatic). "
                "Side effects: muscle spasms, alopecia, dysgeusia, weight loss. "
                "Intermittent cycling protocols used for prolonged prophylaxis in Gorlin."
            ),
            "radiation_ci_in_hereditary_bcc": (
                "RADIATION ABSOLUTELY CONTRAINDICATED in PTCH1 (Gorlin) and TP53 (LFS). "
                "PTCH1: RT induces second somatic hit (9q22 LOH) in every irradiated hair follicle → BCC explosion within 1-2yr. "
                "TP53: RT causes secondary sarcoma/carcinoma in LFS (no intact p53-mediated apoptosis). "
                "XPC/ERCC2: UV light CI (not ionising radiation CI per se — different mechanism). "
                "Clinical impact: BCC in PTCH1/TP53 patients must be treated surgically, never with RT."
            ),
        },
        "abbreviations": {
            "BCC":    "Basal Cell Carcinoma",
            "SCC":    "Squamous Cell Carcinoma",
            "OKC":    "Odontogenic Keratocyst",
            "NER":    "Nucleotide Excision Repair",
            "GGR":    "Global Genome Repair (subpathway of NER)",
            "TC-NER": "Transcription-Coupled NER",
            "TFIIH":  "Transcription Factor IIH (RNA Pol II transcription + NER dual role)",
            "XP":     "Xeroderma Pigmentosum",
            "NBCCS":  "Naevoid Basal Cell Carcinoma Syndrome (Gorlin Syndrome)",
            "SHH":    "Sonic Hedgehog",
            "SMO":    "Smoothened (GPCR SHH pathway transducer)",
            "GLI1/2": "Glioma-Associated Oncogene Homolog 1/2 (SHH transcription factors)",
            "DUB":    "Deubiquitinase",
            "MMR":    "Mismatch Repair",
            "MSI-H":  "Microsatellite Instability High",
            "IHC":    "Immunohistochemistry",
            "MLPA":   "Multiplex Ligation-dependent Probe Amplification",
            "LOH":    "Loss of Heterozygosity",
            "LOF":    "Loss of Function",
            "AD":     "Autosomal Dominant",
            "AR":     "Autosomal Recessive",
            "GOF":    "Gain of Function",
            "WB-MRI": "Whole-Body MRI (Toronto Protocol — no ionising radiation for LFS)",
            "CPD":    "Cyclobutane Pyrimidine Dimer (UV photoproduct)",
            "6-4PP":  "6-4 Photoproduct (UV-DNA lesion)",
        },
    }


if __name__ == "__main__":
    import json, sys
    fn = sys.argv[1] if len(sys.argv) > 1 else "overview"
    if fn == "overview":     print(json.dumps(generate_overview(), indent=2))
    elif fn == "breakdown":  print(json.dumps(generate_breakdown(), indent=2))
    elif fn == "definitions":print(json.dumps(generate_definitions(), indent=2))
