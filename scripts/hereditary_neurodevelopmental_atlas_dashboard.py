#!/usr/bin/env python3
"""Hereditary-Neurodevelopmental-Disorders-Atlas — Complete 8-Gene Neurodevelopmental Genetics Atlas
SHANK3  (SH3 and multiple ankyrin repeat domains 3; 1730 aa; 22q13.33; AD de novo LOF/deletion;
         Phelan-McDermid syndrome; ABSENT/MINIMAL SPEECH PATHOGNOMONIC; autism 94%;
         postsynaptic density scaffold protein; lymphedema 50%; GI dysmotility;
         seed SEED_BASE+0) ·
SYNGAP1 (synaptic RAS GTPase activating protein 1; 1343 aa; 6p21.32; AD de novo LOF;
         MRIDS5 — intellectual disability type 5; epilepsy 50%; PHOTOPAROXYSMAL RESPONSE 70%;
         ASD 50%; hypotonia; RAS-GAP synaptopathy;
         seed SEED_BASE+1) ·
ADNP    (activity-dependent neuroprotective protein; 1102 aa; 20q13.13; AD de novo LOF;
         Helsmoortel-Van der Aa syndrome; autism 90%+; ABSENT/MINIMAL SPEECH;
         p.Tyr719Ter RECURRENT HOTSPOT 25%; chromatin remodelling SWI/SNF;
         seed SEED_BASE+2) ·
ANKRD11 (ankyrin repeat domain 11; 2663 aa; 16q24.3; AD de novo LOF/deletion;
         KBG syndrome; MACRODONTIA UPPER-CENTRAL-INCISORS PATHOGNOMONIC;
         short stature; round face; mild-moderate ID; 16q24.3 microdeletion 30%;
         seed SEED_BASE+3) ·
KAT6A   (lysine acetyltransferase 6A; 1004 aa; 8p11.21; AD de novo LOF;
         KAT6A syndrome; ABSENT SPEECH majority; feeding difficulties neonatal;
         cardiac CHD 30%; microcephaly; ASD features; histone H3K9/K23 acetyltransferase;
         seed SEED_BASE+4) ·
WAC     (WW domain-containing adapter with coiled-coil; 647 aa; 10p12.1; AD de novo LOF;
         DeSanto-Shinawi syndrome; BEHAVIORAL DYSREGULATION PATHOGNOMONIC
         (aggression+self-injury+hyperactivity); epilepsy 30%; mild-moderate ID;
         seed SEED_BASE+5) ·
MED13L  (mediator complex subunit 13-like; 2210 aa; 12q24.21; AD de novo LOF;
         MED13L syndrome; moderate-severe ID; NON-VERBAL 50%; facial hypotonia;
         gait ataxia; feeding tube infancy; wide mouth CHARACTERISTIC;
         seed SEED_BASE+6) ·
FOXP1   (forkhead box P1; 677 aa; 3p13; AD de novo LOF;
         FOXP1 syndrome (MRDFOXP1); intellectual disability; VERBAL DYSPRAXIA;
         language impairment (expressive > receptive); ASD features; macrocephaly;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 3038-3045)
"""
import random

SEED_BASE = 3038

ATLAS_GENES = [
    {
        "gene": "SHANK3",
        "protein": (
            "SHANK3 -- 22q13.33 AD-de-novo-LOF/deletion -- 1730aa -- SH3-And-Multiple-"
            "Ankyrin-Repeat-Domains-3-Phelan-McDermid-Syndrome-ABSENT-SPEECH-PATHOGNOMONIC-"
            "Autism-94pct-Lymphedema-50pct-OMIM-606230"
        ),
        "locus": "22q13.33",
        "protein_size": (
            "1730 aa / 189 kDa (SHANK3; SH3 and multiple ankyrin repeat domains 3; "
            "postsynaptic density (PSD) scaffold protein; "
            "STRUCTURE: ankyrin repeats + SH3 domain + PDZ domain + proline-rich region + SAM domain; "
            "FUNCTION: "
            "  SHANK3 is a MASTER SCAFFOLDING protein of the postsynaptic density at excitatory synapses; "
            "  SHANK3 anchors NMDA receptors, AMPA receptors, mGluR1/5 via Homer-PSD95 scaffold; "
            "  SHANK3 LOF → reduced dendritic spine density + immature synapse morphology; "
            "  SHANK3 interacts: Homer (mGluR tethering), Cortactin (actin), GKAP/SAPAP (PSD-95 bridge); "
            "PHELAN-MCDERMID SYNDROME (PMS) — CLINICAL: "
            "  CAUSE: 22q13.3 deletion (80-85%) OR SHANK3 intragenic pathogenic variant (15-20%); "
            "  DELETION SIZE: 100 kb – 9 Mb; SHANK3 haploinsufficiency drives neurodevelopmental phenotype; "
            "  ABSENT/MINIMAL SPEECH: "
            "    PATHOGNOMONIC: 75-94% non-verbal or severely limited speech; "
            "    Expressive language universally severely impaired; "
            "    Receptive language better than expressive; "
            "  AUTISM SPECTRUM DISORDER: 94% meet DSM-5 criteria; "
            "  HYPOTONIA: neonatal hypotonia universal; delayed motor milestones; "
            "  EPILEPSY: "
            "    Prevalence 17-41% (lower than expected for severe ASD+ID); "
            "    Onset: median age 3-4 years; "
            "    Types: focal + generalised; often treatment-responsive; "
            "  LYMPHEDEMA: 50% — pedal lymphedema, often bilateral; "
            "  GI DYSMOTILITY: chronic diarrhoea; cyclic vomiting; GERD; constipation; "
            "  THERMOREGULATION: impaired thermal pain sensation; decreased perspiration; "
            "    Risk of hyperthermia in warm environments; "
            "  ABSENT DEEP TENDON REFLEXES: near universal; "
            "  REGRESSION: reported in 40%+ especially in adolescence/young adulthood; "
            "    Catatonic-like episodes; loss of skills; "
            "  LARGE HANDS AND EARS; minor dysmorphia; "
            "  RENAL: structural anomalies in ~38%; "
            "  CARDIAC: congenital anomalies occasional; "
            "  SLEEP: severe sleep disturbance; reduced melatonin; "
            "GENOTYPE-PHENOTYPE: "
            "  Larger deletion = more severe but SHANK3 haploinsufficiency is primary driver; "
            "  SHANK3 intragenic variants: similar neurodevelopmental phenotype to small deletions; "
            "  PROSABRI: positive SHANK3 response to insulin-like growth factor 1 (IGF-1) — trials; "
            "DIAGNOSIS: "
            "  22q13.3 deletion: chromosomal microarray (CMA) FIRST; "
            "  SHANK3 intragenic: gene sequencing; "
            "  SNP array > FISH (misses small deletions); "
            "GENE: SHANK3; encoded 22q13.33; OMIM gene 606230; PMS #606232"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — SHANK3 / PHELAN-MCDERMID SYNDROME: "
            "  DE NOVO: >95%; 22q13.3 deletion (80-85%) or intragenic SHANK3 variant (15-20%); "
            "  FAMILIAL: rare; parental mosaicism; ring chromosome 22 in some; "
            "  RING CHR 22: ring chromosome 22 associated; "
            "  RECURRENCE: <1% de novo; 50% if parental mosaicism confirmed; "
            "TESTING ALGORITHM: "
            "  STEP 1: Chromosomal microarray (CMA) — detects 22q13.3 deletion; "
            "  STEP 2: If CMA negative and strong phenotype: SHANK3 sequencing; "
            "  STEP 3: If ring chr 22 suspected: karyotype + FISH; "
            "MANAGEMENT: "
            "  AAC (augmentative/alternative communication) early — non-verbal; "
            "  Intensive ABA/early intervention; "
            "  Lymphedema: compression therapy; "
            "  Gastroenterology: GI dysmotility management; "
            "  Nephrology: renal USS at diagnosis; "
            "  Melatonin: sleep (evidence-based); "
            "  Epilepsy: standard AEDs"
        ),
        "disease_category": (
            "SHANK3-PHELAN-MCDERMID-ABSENT-SPEECH-AUTISM: "
            "  ABSENT/MINIMAL SPEECH PATHOGNOMONIC — 75-94% non-verbal; "
            "  AUTISM 94% — highest ASD prevalence of any single gene; "
            "  LYMPHEDEMA 50% — pedal, bilateral; no other ASD gene does this; "
            "  22q13.3 DELETION: CMA mandatory first; SHANK3 intragenic if CMA normal"
        ),
    },
    {
        "gene": "SYNGAP1",
        "protein": (
            "SYNGAP1 -- 6p21.32 AD-de-novo-LOF -- 1343aa -- Synaptic-RAS-GTPase-Activating-"
            "Protein-1-MRIDS5-Intellectual-Disability-Type5-Epilepsy-50pct-"
            "PHOTOPAROXYSMAL-RESPONSE-70pct-ASD-50pct-OMIM-603384"
        ),
        "locus": "6p21.32",
        "protein_size": (
            "1343 aa / 150 kDa (SYNGAP1; synaptic RAS-GTPase-activating protein 1; "
            "postsynaptic density RAS-GAP; "
            "STRUCTURE: PH domain + C2 domain + RAS-GAP domain + coiled-coil + PDZ-binding motif; "
            "FUNCTION: "
            "  SYNGAP1 is a NEGATIVE REGULATOR of RAS/ERK signalling at excitatory synapses; "
            "  SYNGAP1 is activated by Ca2+/CaM at NMDA receptors: "
            "    Ca2+ influx → CaM activation → SYNGAP1 activation → RAS-GTP hydrolysis → RAS OFF; "
            "  SYNGAP1 LOF → excessive RAS/ERK → immature dendritic spines → "
            "    heightened excitability + impaired plasticity; "
            "  SYNGAP1 regulates AMPA receptor trafficking: "
            "    SYNGAP1 LOF → premature AMPA insertion → excess LTP; "
            "  HAPLOINSUFFICIENCY: one functional allele insufficient; 50% SYNGAP1 protein → phenotype; "
            "MRIDS5 (MENTAL RETARDATION, AUTOSOMAL DOMINANT 5) — CLINICAL: "
            "  INTELLECTUAL DISABILITY: "
            "    MODERATE to SEVERE in majority; "
            "    Language impairment: expressive > receptive; "
            "    Motor: hypotonia; delayed walking (mean 22 months); "
            "  EPILEPSY: "
            "    Present in 50% of individuals; "
            "    ONSET: usually 1-3 years; "
            "    TYPES: myoclonic-atonic ('drop attacks'); absence; generalised tonic-clonic; "
            "    PHOTOPAROXYSMAL RESPONSE (PPR): 70% of those with epilepsy → "
            "      EYELID MYOCLONIA on photic stimulation; pathognomonic EEG finding; "
            "    Doose syndrome (myoclonic-atonic epilepsy): well recognised SYNGAP1 phenotype; "
            "    Dravet-like: febrile seizures + generalised epilepsy; "
            "    REGRESSION: acute worsening of cognition/behaviour at seizure onset; "
            "  ASD: ~50%; social communication impaired; hand stereotypies; "
            "  FACIAL: broad nasal bridge; widely spaced eyes; "
            "  EEG FINDINGS: "
            "    Background abnormalities; paroxysmal activity; "
            "    Photoparoxysmal response CHARACTERISTIC in SYNGAP1; "
            "  BEHAVIOURAL: hyperactivity; attention deficit; sleep problems; anxiety; "
            "  GI: feeding difficulties; constipation; "
            "TREATMENT NOTE: "
            "  VALPROATE: effective for myoclonic component; "
            "  AVOID: carbamazepine/vigabatrin (may worsen generalised epilepsy); "
            "  Ketogenic diet: evidence in refractory cases; "
            "GENE: SYNGAP1; encoded 6p21.32; OMIM gene 603384; MRD5 #612621"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — SYNGAP1 / MRD5: "
            "  DE NOVO: >95%; no parent-of-origin effect; "
            "  FAMILIAL: rare; variable expressivity; "
            "  RECURRENCE: empiric <1% for de novo; 50% if confirmed parental variant; "
            "TESTING ALGORITHM: "
            "  Gene panel (NDD panel) or WES/WGS; "
            "  Chromosomal microarray: SYNGAP1 6p21.32 deletions rare; "
            "MANAGEMENT: "
            "  Epilepsy: valproate + clobazam for myoclonic; avoid CBZ; "
            "  EEG: routine + photic stimulation to detect PPR; "
            "  REGRESSION monitoring at seizure onset; "
            "  ABA/early intervention; "
            "  Occupational therapy; "
            "  Sleep management (melatonin)"
        ),
        "disease_category": (
            "SYNGAP1-MRD5-EPILEPSY-PHOTOPAROXYSMAL: "
            "  PHOTOPAROXYSMAL RESPONSE 70% — EEG with photic stimulation mandatory; "
            "  MYOCLONIC-ATONIC SEIZURES (drop attacks) CHARACTERISTIC — valproate first-line; "
            "  REGRESSION at seizure onset — acute cognitive decline; monitor closely; "
            "  RAS-GAP synaptopathy: SYNGAP1 haploinsufficiency → excess RAS/ERK at synapse"
        ),
    },
    {
        "gene": "ADNP",
        "protein": (
            "ADNP -- 20q13.13 AD-de-novo-LOF -- 1102aa -- Activity-Dependent-Neuroprotective-"
            "Protein-Helsmoortel-Van-Der-Aa-Syndrome-Autism-90pct-ABSENT-SPEECH-"
            "p.Tyr719Ter-RECURRENT-HOTSPOT-25pct-OMIM-611386"
        ),
        "locus": "20q13.13",
        "protein_size": (
            "1102 aa / 124 kDa (ADNP; activity-dependent neuroprotective protein; "
            "SWI/SNF chromatin remodelling complex member; zinc finger protein; "
            "STRUCTURE: 9 zinc finger domains + homeobox domain + nuclear localisation signal; "
            "FUNCTION: "
            "  ADNP is a component of the SWI/SNF (BAF) chromatin remodelling complex; "
            "  ADNP regulates chromatin accessibility → controls transcription of hundreds of genes; "
            "  ADNP protects neurons from oxidative stress (NAPVSIPQ/NAP peptide from ADNP); "
            "  ADNP LOF → disrupted chromatin remodelling → impaired neuronal differentiation; "
            "  ADNP interacts: ARID1A/BAF complex; HP1 (heterochromatin protein 1); "
            "HELSMOORTEL-VAN DER AA SYNDROME (HVDAS) — CLINICAL: "
            "  AUTISM SPECTRUM DISORDER: 90%+ meet diagnostic criteria (one of highest); "
            "  SPEECH/LANGUAGE: "
            "    ABSENT/MINIMAL SPEECH in majority; "
            "    Echo-proximal speech (repeating partial phrases without communicative intent); "
            "  INTELLECTUAL DISABILITY: moderate-severe; "
            "  HYPOTONIA: neonatal + persistent; delayed motor milestones; "
            "  FEEDING DIFFICULTIES: neonatal tube feeding in 30-40%; GERD; constipation; "
            "  SLEEP DISORDERS: severe sleep disturbance in 80%+; "
            "  BEHAVIOURAL: hand stereotypies; tantrum behaviour; self-injurious behaviour; "
            "  FACIAL: bulbous nose; broad nasal bridge; widely spaced teeth; "
            "  CARDIAC: congenital heart defects in 25%; "
            "  SKELETAL: joint hypermobility; scoliosis; "
            "  EPILEPSY: 30%; various types; "
            "RECURRENT p.Tyr719Ter HOTSPOT: "
            "  Present in ~25% of all ADNP cases; "
            "  Located in exon 5 near zinc finger domains; "
            "  Recurrent mutation at CpG site (methylation-induced deamination); "
            "  Clinical phenotype not clearly distinct from other LOF variants; "
            "  Check specifically in all HVDAS-phenotype patients; "
            "DIAGNOSIS NOTE: "
            "  WES/WGS or gene panel (autism/NDD); "
            "  No chromosomal microarray finding (intragenic LOF); "
            "MANAGEMENT: "
            "  AAC (augmentative/alternative communication); "
            "  Early intensive ABA/behavioural intervention; "
            "  GERD: PPI + positioning; "
            "  Sleep: melatonin + behavioural sleep intervention; "
            "  Cardiac: echo at diagnosis; "
            "  NAP peptide (NAPVSIPQ): investigational — derived from ADNP protein; "
            "GENE: ADNP; encoded 20q13.13; OMIM gene 611386; HVDAS #615873"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — ADNP / HELSMOORTEL-VAN DER AA SYNDROME: "
            "  DE NOVO: >98%; extremely rare familial cases; "
            "  RECURRENT p.Tyr719Ter: ~25% — CpG methylation hotspot; "
            "  RECURRENCE: <1% empiric; "
            "TESTING ALGORITHM: "
            "  WES/WGS first (recommended for autism+ID); "
            "  Gene panel (autism/NDD panel); "
            "  p.Tyr719Ter targeted testing if resources limited; "
            "MANAGEMENT: "
            "  AAC early — non-verbal majority; "
            "  ABA/EIBI autism intervention; "
            "  Sleep management; "
            "  Cardiac echo at diagnosis; "
            "  GI (GERD/constipation): PPI + MiraLax; "
            "  Epilepsy: standard AEDs"
        ),
        "disease_category": (
            "ADNP-HVDAS-AUTISM-ABSENT-SPEECH: "
            "  AUTISM 90%+ — near universal; one of highest single-gene ASD rates; "
            "  p.Tyr719Ter RECURRENT HOTSPOT 25% — check specifically in HVDAS phenotype; "
            "  ABSENT/MINIMAL SPEECH in majority — AAC mandatory from early intervention; "
            "  SWI/SNF chromatin remodelling: ADNP LOF → disrupted neuronal gene regulation"
        ),
    },
    {
        "gene": "ANKRD11",
        "protein": (
            "ANKRD11 -- 16q24.3 AD-de-novo-LOF/deletion -- 2663aa -- Ankyrin-Repeat-Domain-11-"
            "KBG-Syndrome-MACRODONTIA-UPPER-INCISORS-PATHOGNOMONIC-Short-Stature-"
            "Round-Face-Mild-Moderate-ID-16q24.3-Microdeletion-30pct-OMIM-611192"
        ),
        "locus": "16q24.3",
        "protein_size": (
            "2663 aa / 294 kDa (ANKRD11; ankyrin repeat domain 11; nuclear receptor corepressor; "
            "STRUCTURE: ankyrin repeats (multiple) + nuclear localisation signal + transcriptional repressor domains; "
            "FUNCTION: "
            "  ANKRD11 is a transcriptional corepressor; "
            "  ANKRD11 interacts with p160 coactivators (SRC1, GRIP1, ACTR) → competes with "
            "  coactivators for nuclear receptor binding → REPRESSES transcription; "
            "  ANKRD11 required for correct craniofacial development; "
            "  ANKRD11 LOF → disrupted gene regulation in neural crest cells → "
            "    craniofacial anomalies + skeletal growth defect; "
            "KBG SYNDROME — CLINICAL: "
            "  MACRODONTIA: "
            "    UPPER CENTRAL INCISORS PATHOGNOMONIC; "
            "    Delayed dental eruption; "
            "    Enamel hypoplasia; "
            "    PANORAMIC DENTAL X-RAY essential at diagnosis; "
            "  SHORT STATURE: "
            "    Postnatal growth failure in ~80%; "
            "    Adult height -2 to -3 SD; "
            "    Bone age DELAYED (vs Sotos where advanced); "
            "  FACIAL FEATURES: "
            "    ROUND FACE; short nose; widely spaced eyes (hypertelorism); "
            "    Thin upper lip; short philtrum; low-set ears; "
            "  BEHAVIOURAL: "
            "    Friendly, social personality; "
            "    Emotional lability; aggression episodes; "
            "    ASD features in ~30%; "
            "    ADHD-like behaviour; "
            "  INTELLECTUAL DISABILITY: mild-moderate; "
            "  HEARING LOSS: ~30%; conductive + sensorineural; "
            "  SKELETAL: brachydactyly; single palmar crease; cervical vertebral anomalies (C1/C2); "
            "    Clinodactyly; rib anomalies; "
            "  CARDIAC: CHD in ~30% (ASD, VSD); "
            "  OPHTHALMOLOGY: strabismus; ptosis; refractive errors; "
            "  SEIZURES: ~30%; "
            "  CRYPTORCHIDISM: males ~40%; "
            "16q24.3 MICRODELETION: "
            "  ~30% of KBG caused by 16q24.3 microdeletion (CMA detectable); "
            "  ~70% ANKRD11 intragenic pathogenic variant; "
            "  Deletion size correlates with additional features (beyond ANKRD11 alone); "
            "GENE: ANKRD11; encoded 16q24.3; OMIM gene 611192; KBG #148050"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — ANKRD11 / KBG SYNDROME: "
            "  DE NOVO: ~85-90%; "
            "  FAMILIAL: 10-15%; variable expressivity → mildly affected parent; "
            "  MOSAICISM: rare; "
            "TESTING ALGORITHM: "
            "  STEP 1: Chromosomal microarray (detects 16q24.3 microdeletion ~30%); "
            "  STEP 2: ANKRD11 sequencing (detects intragenic variants ~70%); "
            "  DENTAL PANORAMIC X-RAY: mandatory at diagnosis (macrodontia); "
            "MANAGEMENT: "
            "  Dental: orthodontist early; macrodontia management; enamel prophylaxis; "
            "  Audiology: hearing assessment annually; hearing aids if needed; "
            "  Growth: GH therapy considered (delayed bone age + short stature); "
            "  Cardiac: echo at diagnosis; "
            "  Ophthalmology; "
            "  Cryptorchidism: orchidopexy; "
            "  Cervical spine: imaging (C1/C2 anomalies risk spinal instability); "
            "  Neurodevelopmental: educational support"
        ),
        "disease_category": (
            "ANKRD11-KBG-MACRODONTIA-SHORT-STATURE: "
            "  MACRODONTIA UPPER CENTRAL INCISORS PATHOGNOMONIC — panoramic X-ray mandatory; "
            "  SHORT STATURE with DELAYED bone age (DDx vs NSD1/EZH2 advanced bone age); "
            "  16q24.3 MICRODELETION 30% — CMA first line; intragenic ANKRD11 sequencing if negative; "
            "  CERVICAL SPINE: C1/C2 anomalies → spinal imaging mandatory before anaesthesia"
        ),
    },
    {
        "gene": "KAT6A",
        "protein": (
            "KAT6A -- 8p11.21 AD-de-novo-LOF -- 1004aa -- Lysine-Acetyltransferase-6A-"
            "KAT6A-Syndrome-ABSENT-SPEECH-Majority-Feeding-Difficulties-"
            "Cardiac-CHD-30pct-Microcephaly-H3K9-K23-Acetyltransferase-OMIM-601408"
        ),
        "locus": "8p11.21",
        "protein_size": (
            "1004 aa / 112 kDa (KAT6A; lysine acetyltransferase 6A; MYST family HAT; "
            "also called MOZ (monocytic leukaemia zinc finger protein) or MYST3; "
            "STRUCTURE: MYST domain (acetyltransferase catalytic) + zinc finger + "
            "Tudor domain + serine-rich region; "
            "FUNCTION: "
            "  KAT6A acetylates H3K9 and H3K23 → active chromatin marks; "
            "  KAT6A is a regulator of HOX gene expression during development; "
            "  KAT6A is required for haematopoiesis and neural crest cell development; "
            "  KAT6A LOF → disrupted histone acetylation → impaired developmental gene programs; "
            "  KAT6A is a fusion partner in AML: t(8;16) KAT6A-CREBBP (somatic, haem malignancy); "
            "KAT6A SYNDROME — CLINICAL: "
            "  SPEECH/LANGUAGE: "
            "    ABSENT/MINIMAL SPEECH in majority: ~50-70% non-verbal or minimally verbal; "
            "    Expressive language severely impaired; "
            "    VERBAL DYSPRAXIA component; "
            "  FEEDING DIFFICULTIES: "
            "    NEONATAL/INFANTILE: hypotonia → poor suck; tube feeding in 40-60%; "
            "    GERD; aspiration risk; swallowing study recommended; "
            "  CARDIAC: "
            "    CHD in ~30%; ASD/VSD most common; AVSD in some; "
            "    Echo at diagnosis mandatory; "
            "  MICROCEPHALY: head circumference ≤ -2 SD in ~40%; "
            "  INTELLECTUAL DISABILITY: moderate-severe; "
            "  AUTISM FEATURES: ASD features in ~40%; "
            "  BEHAVIOURAL: hyperactivity; hand stereotypies; "
            "  OPHTHALMOLOGY: coloboma; optic nerve hypoplasia; "
            "  RENAL: structural anomalies occasional; "
            "  HEARING: sensorineural hearing loss ~20%; "
            "  THROMBOCYTOPENIA: platelet anomalies in some (KAT6A → haematopoiesis); "
            "  RECURRENT INFECTIONS: immune function impaired; "
            "DIAGNOSIS: "
            "  WES/WGS or neurodevelopmental gene panel; "
            "  Chromosomal microarray: 8p11.21 microdeletion rare; "
            "MANAGEMENT: "
            "  Feeding: nasogastric/gastrostomy tube if needed; "
            "  AAC: non-verbal majority; "
            "  ABA early intervention; "
            "  Cardiac: echo + paediatric cardiology; "
            "  Ophthalmology: early vision assessment (coloboma); "
            "  Audiology: newborn hearing screen + follow-up; "
            "  Renal USS at diagnosis; "
            "  FBC for thrombocytopenia; "
            "GENE: KAT6A; encoded 8p11.21; OMIM gene 601408; KAT6A syndrome #616268"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — KAT6A / KAT6A SYNDROME: "
            "  DE NOVO: >97%; "
            "  FAMILIAL: exceptionally rare; "
            "  TRUNCATING VARIANTS: most pathogenic (nonsense, frameshift); "
            "  MISSENSE: limited evidence; "
            "TESTING ALGORITHM: "
            "  WES/WGS or neurodevelopmental/ASD gene panel; "
            "  FISH/CMA: 8p11.21 microdeletion rare but possible; "
            "MANAGEMENT: "
            "  Swallowing study early (feeding difficulties); "
            "  Gastrostomy if prolonged tube dependence; "
            "  Echo at diagnosis (30% CHD); "
            "  Ophthalmology (coloboma); "
            "  AAC and speech therapy early"
        ),
        "disease_category": (
            "KAT6A-ABSENT-SPEECH-CARDIAC-FEEDING: "
            "  ABSENT/MINIMAL SPEECH MAJORITY — AAC mandatory from early intervention; "
            "  FEEDING DIFFICULTIES NEONATAL — tube feeding 40-60%; swallowing study urgently; "
            "  CARDIAC CHD 30% — echo mandatory at diagnosis; "
            "  H3K9/H3K23 acetyltransferase: KAT6A LOF → disrupted HOX gene regulation"
        ),
    },
    {
        "gene": "WAC",
        "protein": (
            "WAC -- 10p12.1 AD-de-novo-LOF -- 647aa -- WW-Domain-Containing-Adapter-"
            "Coiled-Coil-DeSanto-Shinawi-Syndrome-BEHAVIORAL-DYSREGULATION-PATHOGNOMONIC-"
            "Aggression-Self-Injury-Hyperactivity-Epilepsy-30pct-OMIM-615049"
        ),
        "locus": "10p12.1",
        "protein_size": (
            "647 aa / 72 kDa (WAC; WW domain-containing adapter with coiled-coil; "
            "histone H2B ubiquitination regulator; "
            "STRUCTURE: WW domain + coiled-coil domain + nuclear localisation signal; "
            "FUNCTION: "
            "  WAC interacts with RNF20/RNF40 ubiquitin E3 ligase complex; "
            "  WAC bridges RNF20/40 to histone H2B → H2B monoubiquitination (H2Bub1); "
            "  H2Bub1 is required for: "
            "    Transcription elongation (RNA Pol II pausing regulation); "
            "    DNA damage response; "
            "    Chromatin remodelling; "
            "  WAC LOF → reduced H2Bub1 → impaired transcription + chromatin instability; "
            "  WAC is also expressed in postmitotic neurons (role in mature brain function); "
            "DESANTO-SHINAWI SYNDROME — CLINICAL: "
            "  BEHAVIOURAL DYSREGULATION: "
            "    PATHOGNOMONIC COMBINATION: "
            "      AGGRESSION (physical): hitting, biting, kicking; "
            "      SELF-INJURIOUS BEHAVIOUR (SIB): head banging, biting self; "
            "      HYPERACTIVITY: severe, often ADHD-diagnosis before genetic testing; "
            "    Behavioural features present in 90%+; "
            "    Onset: early childhood (2-4 years); "
            "    Variable: some patients have manageable behaviour; others severe; "
            "  INTELLECTUAL DISABILITY: mild-moderate; "
            "    Cognitive range: mild ID to borderline; "
            "  EPILEPSY: "
            "    ~30% of patients; "
            "    Types: focal; generalised; febrile seizures → afebrile; "
            "    Usually well-controlled; "
            "  ASD FEATURES: ~30%; social communication difficulties; "
            "  FACIAL: broad forehead; deep-set eyes; widely spaced teeth; "
            "  SLEEP DIFFICULTIES: common; behavioural and potential melatonin deficiency; "
            "  MACROCEPHALY: relative macrocephaly in some; "
            "  MUSCULOSKELETAL: joint hypermobility; scoliosis; "
            "MANAGEMENT NOTE: "
            "  BEHAVIOURAL: ABA + PBS (positive behaviour support); "
            "  MEDICATION: risperidone/aripiprazole for severe aggression/SIB; "
            "  STIMULANTS for ADHD component (supervised); "
            "  EPILEPSY: standard AEDs; "
            "GENE: WAC; encoded 10p12.1; OMIM gene 615049; DeSanto-Shinawi #615502"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — WAC / DESANTO-SHINAWI SYNDROME: "
            "  DE NOVO: >97%; "
            "  FAMILIAL: rare; mildly affected parent; "
            "  TRUNCATING VARIANTS: most pathogenic; "
            "  RECURRENCE: <1% empiric; "
            "TESTING: "
            "  WES/WGS or neurodevelopmental gene panel; "
            "  CMA: 10p12.1 microdeletion rare; "
            "MANAGEMENT: "
            "  Behavioural: formal PBS plan; applied behaviour analysis; "
            "  Medication: low-dose atypical antipsychotic for severe SIB/aggression; "
            "  ADHD assessment and treatment; "
            "  Sleep: melatonin + behavioural intervention; "
            "  Epilepsy: AED if seizures confirmed"
        ),
        "disease_category": (
            "WAC-DESANTO-SHINAWI-BEHAVIORAL-DYSREGULATION: "
            "  BEHAVIORAL DYSREGULATION PATHOGNOMONIC: AGGRESSION+SELF-INJURY+HYPERACTIVITY; "
            "  Often first diagnosed as ADHD/ODD — WAC testing triggered by ID+severe behaviour; "
            "  EPILEPSY 30% — usually well-controlled; "
            "  H2B ubiquitination: WAC-RNF20/40 complex regulates transcription elongation"
        ),
    },
    {
        "gene": "MED13L",
        "protein": (
            "MED13L -- 12q24.21 AD-de-novo-LOF -- 2210aa -- Mediator-Complex-Subunit-13-Like-"
            "MED13L-Syndrome-NON-VERBAL-50pct-Facial-Hypotonia-Gait-Ataxia-Wide-Mouth-"
            "CHARACTERISTIC-Feeding-Difficulties-OMIM-608771"
        ),
        "locus": "12q24.21",
        "protein_size": (
            "2210 aa / 243 kDa (MED13L; mediator complex subunit 13-like; "
            "subunit of the Mediator transcriptional coactivator complex; "
            "STRUCTURE: MED13-like domain + LXXLL nuclear receptor-interacting motif + "
            "conserved N-terminal domain; "
            "FUNCTION: "
            "  MED13L is a subunit of the large MEDIATOR complex (>30 subunits); "
            "  MEDIATOR bridges sequence-specific transcription factors to RNA Pol II; "
            "  MED13L is part of the 'Kinase module' (CDK8/CDK19, MED12, CYCLIN C, MED13/MED13L); "
            "  Kinase module regulates ENHANCER activity and developmental transcription programs; "
            "  MED13L LOF → disrupted Mediator kinase module → impaired transcriptional activation "
            "    during brain development; "
            "MED13L SYNDROME — CLINICAL: "
            "  INTELLECTUAL DISABILITY: "
            "    Moderate-severe in most; "
            "  SPEECH/LANGUAGE: "
            "    NON-VERBAL or MINIMALLY VERBAL in ~50%; "
            "    Severe expressive language impairment universal; "
            "    Single words only in many; "
            "  FACIAL HYPOTONIA: "
            "    DROOLING: often prominent due to oral hypotonia; "
            "    WIDE MOUTH: characteristic — corners pulled down; "
            "    BULBOUS/WIDE NOSE: broad nasal bridge; bulbous tip; "
            "    PROMINENT EARS; "
            "    DYSPLASTIC EARS; "
            "  GAIT ATAXIA: "
            "    Broad-based gait; truncal ataxia; "
            "    Cerebellum often normal on MRI; functional ataxia; "
            "  FEEDING DIFFICULTIES: "
            "    Neonatal/infantile feeding problems → tube feeding in 30-40%; "
            "    Oral hypotonia + poor coordination; "
            "  BEHAVIOURAL: "
            "    Variable; ASD features ~30%; hyperactivity; stereotypies; "
            "  EPILEPSY: ~30%; various types; usually responsive; "
            "  CARDIAC: CHD occasional; "
            "  BRAIN MRI: often normal; occasional cerebellar hypoplasia; thin corpus callosum; "
            "  CONGENITAL HEART DISEASE: transposition of the great arteries in some; "
            "GENOTYPE-PHENOTYPE: "
            "  MED13L-HEART-OVERGROWTH: specific missense (hotspot) associated with "
            "    congenital heart disease + macrosomia → MED13L syndrome subtype; "
            "  Truncating LOF → classic MED13L syndrome (ID + speech); "
            "GENE: MED13L; encoded 12q24.21; OMIM gene 608771; MED13L syndrome #616536"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — MED13L / MED13L SYNDROME: "
            "  DE NOVO: >95%; "
            "  FAMILIAL: rare; mildly affected parent possible; "
            "  TRUNCATING: most (nonsense, frameshift, splice); "
            "  MISSENSE: variable — some pathogenic (MED13L-HEART subtype), many VUS; "
            "TESTING: "
            "  WES/WGS or neurodevelopmental gene panel; "
            "  CMA: 12q24.21 microdeletion rare; "
            "MANAGEMENT: "
            "  AAC early — non-verbal in 50%; "
            "  Feeding: swallowing study; gastrostomy if needed; "
            "  Oro-motor physiotherapy (facial hypotonia); "
            "  Drooling management (glycopyrrolate or botulinum toxin parotid); "
            "  Physiotherapy for gait ataxia; "
            "  Epilepsy: standard AEDs; "
            "  Cardiac echo at diagnosis"
        ),
        "disease_category": (
            "MED13L-NON-VERBAL-GAIT-ATAXIA-WIDE-MOUTH: "
            "  NON-VERBAL 50% — AAC mandatory from early intervention; "
            "  WIDE MOUTH + FACIAL HYPOTONIA CHARACTERISTIC — drooling management needed; "
            "  GAIT ATAXIA — broad-based gait; cerebellar MRI often normal; "
            "  MEDIATOR KINASE MODULE: MED13L LOF → disrupted developmental transcription"
        ),
    },
    {
        "gene": "FOXP1",
        "protein": (
            "FOXP1 -- 3p13 AD-de-novo-LOF -- 677aa -- Forkhead-Box-P1-FOXP1-Syndrome-"
            "MRDFOXP1-Intellectual-Disability-VERBAL-DYSPRAXIA-Language-Impairment-"
            "ASD-Features-Macrocephaly-OMIM-608106"
        ),
        "locus": "3p13",
        "protein_size": (
            "677 aa / 77 kDa (FOXP1; forkhead box P1; transcription factor; "
            "FOXP subfamily member; "
            "STRUCTURE: glutamine-rich domain + zinc finger + leucine zipper + "
            "forkhead (FOX) DNA-binding domain; "
            "FUNCTION: "
            "  FOXP1 is a transcriptional repressor; "
            "  FOXP1 regulates gene expression in multiple tissues: "
            "    BRAIN: neuronal identity + synaptic gene expression; "
            "    LUNG: pulmonary development; "
            "    HEART: cardiac morphogenesis; "
            "    IMMUNE: B-cell differentiation; "
            "  FOXP1 interacts with FOXP2 (speech/language gene) — heterodimer; "
            "  FOXP1+FOXP2 complex regulates CORTICOSTRIARAL CIRCUITRY → "
            "    speech motor planning (verbal dyspraxia when disrupted); "
            "  FOXP1 LOF → disrupted corticostriatal + hippocampal development; "
            "FOXP1 SYNDROME (MRDFOXP1) — CLINICAL: "
            "  INTELLECTUAL DISABILITY: "
            "    Mild-moderate; IQ typically 50-75 range; "
            "    Memory and attention difficulties; "
            "  LANGUAGE IMPAIRMENT: "
            "    Expressive language impairment > receptive; "
            "    VERBAL DYSPRAXIA: speech motor planning deficit; "
            "      Inconsistent speech errors; groping; "
            "      Difficulties with multisyllabic words; "
            "    Speech often described as 'dysarthric' + dyspraxic; "
            "    Speech intelligibility impaired; "
            "    Often only diagnosed after first attempting FOXP2; "
            "  ASD FEATURES: ~50% ASD criteria; social communication impaired; "
            "  MACROCEPHALY: relative macrocephaly in ~40%; "
            "  BEHAVIOURAL: "
            "    Stereotypies; hand-flapping; "
            "    Anxiety; "
            "    Hyperactivity; "
            "  INTELLECTUAL PROFILE: "
            "    Verbal tasks disproportionately impaired vs performance tasks; "
            "    Relatively preserved visuospatial skills; "
            "  BRAIN MRI: often normal; occasional thin corpus callosum; "
            "  CARDIAC: CHD occasional (~15%); "
            "  SKELETAL: joint hypermobility; "
            "3p13 MICRODELETION: "
            "  Some patients have 3p13 microdeletion (CMA detectable); "
            "  Others have FOXP1 intragenic pathogenic variants (sequencing required); "
            "FOXP1 vs FOXP2: "
            "  FOXP2: pure verbal dyspraxia (CAS) with normal intelligence in many; "
            "  FOXP1: ID + ASD + verbal dyspraxia — more severe neurodevelopmental; "
            "  KEY DDx: FOXP1 causes ID; FOXP2 typically does NOT; "
            "GENE: FOXP1; encoded 3p13; OMIM gene 608106; MRDFOXP1 #613670"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF — FOXP1 / MRDFOXP1: "
            "  DE NOVO: >95%; "
            "  FAMILIAL: rare; variable expressivity; "
            "  3p13 MICRODELETION: some patients — CMA detects; "
            "  INTRAGENIC FOXP1: majority — sequencing required; "
            "TESTING ALGORITHM: "
            "  STEP 1: Chromosomal microarray (detects 3p13 microdeletion); "
            "  STEP 2: FOXP1 sequencing (intragenic pathogenic variants); "
            "  FOXP2 TESTING: consider if verbal dyspraxia without ID (different syndrome); "
            "MANAGEMENT: "
            "  Speech therapy: DIVA therapy, PROMPT, NUFFIELD dyspraxia programme; "
            "  AAC if speech intelligibility severely impaired; "
            "  Educational support: verbal/oral tasks disproportionately harder; "
            "  ASD intervention if ASD criteria met; "
            "  Anxiety management"
        ),
        "disease_category": (
            "FOXP1-MRDFOXP1-VERBAL-DYSPRAXIA-ID: "
            "  VERBAL DYSPRAXIA — speech motor planning deficit; inconsistent errors; groping; "
            "  FOXP1 ID + ASD vs FOXP2 CAS without ID — KEY DIFFERENTIAL; "
            "  EXPRESSIVE > RECEPTIVE language gap — verbal tasks disproportionately impaired; "
            "  3p13 MICRODELETION — CMA first; FOXP1 intragenic sequencing if negative"
        ),
    },
]


def _generate_patients_for_gene(gene: str, seed: int) -> list:
    """Generate 40 synthetic patients for one neurodevelopmental gene."""
    rng = random.Random(seed)

    severity_params = {
        "SHANK3":  {"severe_pct": 0.50, "moderate_pct": 0.40, "mild_pct": 0.10,
                    "age_min": 0,  "age_max": 36, "iq_mean": 38, "iq_sd": 12},
        "SYNGAP1": {"severe_pct": 0.30, "moderate_pct": 0.55, "mild_pct": 0.15,
                    "age_min": 0,  "age_max": 24, "iq_mean": 55, "iq_sd": 12},
        "ADNP":    {"severe_pct": 0.45, "moderate_pct": 0.45, "mild_pct": 0.10,
                    "age_min": 0,  "age_max": 30, "iq_mean": 42, "iq_sd": 12},
        "ANKRD11": {"severe_pct": 0.15, "moderate_pct": 0.60, "mild_pct": 0.25,
                    "age_min": 0,  "age_max": 60, "iq_mean": 68, "iq_sd": 14},
        "KAT6A":   {"severe_pct": 0.40, "moderate_pct": 0.50, "mild_pct": 0.10,
                    "age_min": 0,  "age_max": 24, "iq_mean": 45, "iq_sd": 13},
        "WAC":     {"severe_pct": 0.20, "moderate_pct": 0.55, "mild_pct": 0.25,
                    "age_min": 0,  "age_max": 48, "iq_mean": 65, "iq_sd": 14},
        "MED13L":  {"severe_pct": 0.35, "moderate_pct": 0.50, "mild_pct": 0.15,
                    "age_min": 0,  "age_max": 36, "iq_mean": 48, "iq_sd": 12},
        "FOXP1":   {"severe_pct": 0.20, "moderate_pct": 0.55, "mild_pct": 0.25,
                    "age_min": 0,  "age_max": 48, "iq_mean": 62, "iq_sd": 13},
    }
    p = severity_params.get(gene, {"severe_pct": 0.30, "moderate_pct": 0.50, "mild_pct": 0.20,
                                    "age_min": 0, "age_max": 36, "iq_mean": 55, "iq_sd": 14})

    features_map = {
        "SHANK3":  [["absent-speech-non-verbal", "autism-94pct"],
                    ["lymphedema-50pct-pedal-bilateral", "absent-DTRs"],
                    ["gi-dysmotility-chronic-diarrhoea", "hypotonia-neonatal"],
                    ["thermoregulation-impaired", "regression-adolescent"]],
        "SYNGAP1": [["epilepsy-myoclonic-atonic-drop-attacks", "photoparoxysmal-response"],
                    ["moderate-severe-ID-hypotonia", "expressive-language-impaired"],
                    ["ASD-50pct-hand-stereotypies", "regression-at-seizure-onset"],
                    ["febrile-seizures-afebrile", "eyelid-myoclonia-photic"]],
        "ADNP":    [["autism-90pct-absent-minimal-speech", "pTyr719Ter-hotspot"],
                    ["hypotonia-feeding-tube-40pct", "GERD-constipation"],
                    ["sleep-disorder-80pct", "cardiac-CHD-25pct"],
                    ["hand-stereotypies-SIB", "moderate-severe-ID"]],
        "ANKRD11": [["macrodontia-upper-incisors-pathognomonic", "delayed-dental-eruption"],
                    ["short-stature-delayed-bone-age", "round-face-hypertelorism"],
                    ["hearing-loss-30pct-conductive", "cervical-anomalies-C1-C2"],
                    ["cardiac-CHD-30pct", "cryptorchidism-males-40pct"]],
        "KAT6A":   [["absent-speech-majority-non-verbal", "feeding-difficulties-tube-40pct"],
                    ["cardiac-CHD-30pct-ASD-VSD", "microcephaly-40pct"],
                    ["coloboma-optic-nerve-hypoplasia", "ASD-features-40pct"],
                    ["SNHL-20pct", "thrombocytopenia-platelet-anomalies"]],
        "WAC":     [["aggression-SIB-hyperactivity-pathognomonic", "ADHD-severe"],
                    ["epilepsy-30pct-focal-generalised", "broad-forehead-deep-eyes"],
                    ["ASD-features-30pct", "sleep-difficulties-behavioural"],
                    ["joint-hypermobility", "emotional-lability"]],
        "MED13L":  [["non-verbal-50pct", "wide-mouth-facial-hypotonia"],
                    ["gait-ataxia-broad-based", "drooling-oral-hypotonia"],
                    ["feeding-tube-infancy-30pct", "bulbous-nose-prominent-ears"],
                    ["ASD-features-30pct", "epilepsy-30pct-AED-responsive"]],
        "FOXP1":   [["verbal-dyspraxia-inconsistent-errors", "expressive-greater-receptive"],
                    ["ID-mild-moderate-IQ-50-75", "ASD-features-50pct"],
                    ["macrocephaly-40pct", "hand-stereotypies-anxiety"],
                    ["visuospatial-relatively-preserved", "3p13-microdeletion-or-intragenic"]],
    }
    feature_options = features_map.get(gene, [["neurodevelopmental-delay"]])

    treatment_map = {
        "SHANK3":  ["AAC-augmentative-alternative-communication",
                    "ABA-early-intensive-intervention",
                    "lymphedema-compression-therapy",
                    "gastroenterology-GI-dysmotility",
                    "melatonin-sleep"],
        "SYNGAP1": ["valproate-myoclonic-epilepsy",
                    "clobazam-adjunct-seizures",
                    "avoid-carbamazepine-vigabatrin",
                    "ABA-autism-intervention",
                    "ketogenic-diet-refractory"],
        "ADNP":    ["AAC-non-verbal-majority",
                    "ABA-EIBI-autism",
                    "melatonin-sleep-disorder",
                    "cardiac-echo-surveillance",
                    "GERD-PPI-management"],
        "ANKRD11": ["dental-orthodontist-early-macrodontia",
                    "audiology-hearing-aids",
                    "GH-therapy-short-stature",
                    "cardiac-echo-30pct-CHD",
                    "cervical-spine-imaging-C1-C2"],
        "KAT6A":   ["feeding-NG-tube-gastrostomy",
                    "AAC-non-verbal-majority",
                    "cardiac-echo-30pct-CHD",
                    "ophthalmology-coloboma",
                    "SALT-swallowing-study"],
        "WAC":     ["PBS-positive-behaviour-support-ABA",
                    "risperidone-aripiprazole-severe-aggression",
                    "stimulants-ADHD",
                    "AED-epilepsy-30pct",
                    "melatonin-sleep"],
        "MED13L":  ["AAC-non-verbal-50pct",
                    "oro-motor-physiotherapy-drooling",
                    "gastrostomy-feeding-difficulties",
                    "physiotherapy-gait-ataxia",
                    "AED-epilepsy"],
        "FOXP1":   ["DIVA-PROMPT-dyspraxia-speech-therapy",
                    "AAC-speech-intelligibility-poor",
                    "ABA-ASD-intervention",
                    "anxiety-CBT",
                    "educational-verbal-accommodations"],
    }
    treatments = treatment_map.get(gene, ["neurodevelopmental-support"])

    mutation_map = {
        "SHANK3":  ["22q13.3-deletion-CMA", "p.Arg586Ter", "p.Leu1659Ter-PDZ-domain",
                    "c.2997del-frameshift", "p.Trp1731Ter-SAM-domain",
                    "ring-chromosome-22"],
        "SYNGAP1": ["p.Arg579Ter", "p.Gln503Ter", "c.1701del-frameshift",
                    "p.Arg579Gly-LOF", "c.3187C>T-exon14",
                    "6p21.32-microdeletion"],
        "ADNP":    ["p.Tyr719Ter-hotspot-recurrent", "p.Arg730Ter", "c.2153del-frameshift",
                    "p.Tyr719Ser-hotspot-adjacent", "p.Arg1105Ter",
                    "c.3284C>G-splice"],
        "ANKRD11": ["p.Arg2378Ter", "16q24.3-deletion-CMA", "p.Gln1843Ter",
                    "c.5530del-frameshift", "p.Arg2006Ter",
                    "p.Arg1894Ter-exon9"],
        "KAT6A":   ["p.Arg1022Ter", "p.Arg860Ter", "c.2920del-frameshift",
                    "p.Lys604Ter-MYST-domain", "p.Arg519Ter",
                    "8p11.21-microdeletion"],
        "WAC":     ["p.Gln394Ter", "p.Arg547Ter", "c.1180del-frameshift",
                    "p.Glu263Ter-WW-domain", "10p12.1-microdeletion",
                    "c.871C>T-exon7"],
        "MED13L":  ["p.Arg1560Ter", "p.Gln1234Ter", "c.4680del-frameshift",
                    "p.Ala1984Thr-missense-pathogenic", "p.Arg1038Ter",
                    "12q24.21-microdeletion"],
        "FOXP1":   ["3p13-microdeletion", "p.Arg525Ter", "p.Gln499Ter-forkhead",
                    "c.1573del-frameshift", "p.Arg328His-DBD",
                    "p.Tyr460Ter"],
    }
    mutations = mutation_map.get(gene, ["unknown"])

    patients = []
    for i in range(40):
        rand = rng.random()
        if rand < p["severe_pct"]:
            severity = "severe"
        elif rand < p["severe_pct"] + p["moderate_pct"]:
            severity = "moderate"
        else:
            severity = "mild"
        age_dx = rng.randint(p["age_min"], p["age_max"])
        iq = max(20, int(rng.normalvariate(p["iq_mean"], p["iq_sd"])))
        # Gene-specific complication flags
        non_verbal   = gene in ("SHANK3", "ADNP", "KAT6A", "MED13L") and rng.random() < (
            0.85 if gene == "SHANK3" else 0.70 if gene == "ADNP" else
            0.60 if gene == "KAT6A" else 0.50)
        epilepsy     = gene in ("SYNGAP1", "SHANK3", "WAC", "MED13L", "ADNP") and rng.random() < (
            0.50 if gene == "SYNGAP1" else 0.30 if gene == "SHANK3" else
            0.30 if gene == "WAC" else 0.30 if gene == "MED13L" else 0.30)
        cardiac      = gene in ("ANKRD11", "KAT6A", "ADNP", "FOXP1") and rng.random() < (
            0.30 if gene in ("ANKRD11", "KAT6A") else 0.25 if gene == "ADNP" else 0.15)
        autism_ftr   = gene in ("SHANK3", "ADNP", "SYNGAP1", "KAT6A", "WAC", "MED13L", "FOXP1") and rng.random() < (
            0.94 if gene == "SHANK3" else 0.90 if gene == "ADNP" else
            0.50 if gene == "SYNGAP1" else 0.40 if gene == "KAT6A" else
            0.30 if gene == "WAC" else 0.30 if gene == "MED13L" else 0.50)
        lymphedema   = gene == "SHANK3" and rng.random() < 0.50
        behavioural  = gene == "WAC" and rng.random() < 0.90
        ppr          = gene == "SYNGAP1" and epilepsy and rng.random() < 0.70
        regression   = gene in ("SHANK3", "SYNGAP1") and rng.random() < 0.30
        treatment    = rng.choice(treatments)
        mutation     = rng.choice(mutations)
        features     = rng.choice(feature_options)
        patients.append({
            "id":                  f"{gene}-{seed}-{i+1:03d}",
            "gene":                gene,
            "age_at_diagnosis_mo": age_dx * 12 if age_dx < 5 else age_dx,
            "severity":            severity,
            "iq_estimate":         iq,
            "associated_features": features,
            "non_verbal":          non_verbal,
            "epilepsy":            epilepsy,
            "cardiac_defect":      cardiac,
            "autism_feature":      autism_ftr,
            "lymphedema":          lymphedema,
            "behavioural_dysreg":  behavioural,
            "photoparoxysmal":     ppr,
            "regression":          regression,
            "treatment":           treatment,
            "mutation":            mutation,
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Neurodevelopmental-Disorders-Atlas."""
    return {
        "atlas":          "Hereditary-Neurodevelopmental-Disorders-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Neurodevelopmental Disorders Atlas "
            "(SHANK3-SYNGAP1-ADNP-ANKRD11-KAT6A-WAC-MED13L-FOXP1)"
        ),
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "SHANK3":  "AD LOF/deletion (Phelan-McDermid syndrome; absent speech PATHOGNOMONIC 94%; autism 94%; lymphedema 50%; 22q13.3 deletion CMA first)",
            "SYNGAP1": "AD LOF de novo (MRD5; epilepsy 50%; PHOTOPAROXYSMAL RESPONSE 70%; ASD 50%; myoclonic-atonic; avoid CBZ)",
            "ADNP":    "AD LOF de novo (Helsmoortel-Van der Aa; autism 90%+; absent/minimal speech; p.Tyr719Ter RECURRENT HOTSPOT 25%; SWI/SNF chromatin)",
            "ANKRD11": "AD LOF/deletion (KBG syndrome; MACRODONTIA upper incisors PATHOGNOMONIC; short stature delayed BA; 16q24.3 microdeletion 30%)",
            "KAT6A":   "AD LOF de novo (KAT6A syndrome; absent speech majority; feeding difficulties tube 40-60%; cardiac CHD 30%; microcephaly; H3K9/K23 HAT)",
            "WAC":     "AD LOF de novo (DeSanto-Shinawi; BEHAVIORAL DYSREGULATION PATHOGNOMONIC: aggression+SIB+hyperactivity; epilepsy 30%; H2Bub1 regulator)",
            "MED13L":  "AD LOF de novo (MED13L syndrome; non-verbal 50%; wide mouth+facial hypotonia CHARACTERISTIC; gait ataxia; Mediator kinase module)",
            "FOXP1":   "AD LOF de novo (FOXP1/MRDFOXP1; ID+VERBAL DYSPRAXIA+ASD; expressive>receptive; macrocephaly; DDx FOXP2 = dyspraxia no ID)",
        },
        "key_clinical_rules": [
            "SHANK3 (PMS): ABSENT/MINIMAL SPEECH PATHOGNOMONIC (94%); autism 94%; LYMPHEDEMA 50% (no other ASD gene); CMA first (22q13.3 deletion 80-85%)",
            "SYNGAP1: PHOTOPAROXYSMAL RESPONSE (PPR) 70% — EEG with photic stimulation MANDATORY; myoclonic-atonic epilepsy; avoid carbamazepine/vigabatrin",
            "SYNGAP1: REGRESSION at seizure onset — acute cognitive decline; monitor closely; valproate first-line for myoclonic component",
            "ADNP (HVDAS): p.Tyr719Ter RECURRENT HOTSPOT 25% — check specifically; autism 90%+ with absent/minimal speech; AAC mandatory",
            "ANKRD11 (KBG): MACRODONTIA UPPER CENTRAL INCISORS PATHOGNOMONIC — panoramic X-ray at diagnosis; SHORT STATURE + DELAYED bone age (DDx: NSD1/EZH2 have ADVANCED BA)",
            "ANKRD11 (KBG): CERVICAL SPINE C1/C2 anomalies — spinal imaging mandatory before any anaesthesia; hearing loss 30%",
            "KAT6A: ABSENT SPEECH MAJORITY + FEEDING DIFFICULTIES NEONATAL — tube feeding 40-60%; swallowing study urgently; cardiac echo 30% CHD mandatory",
            "WAC (DeSanto-Shinawi): BEHAVIOURAL DYSREGULATION PATHOGNOMONIC — aggression+SIB+hyperactivity; often first labelled ADHD/ODD before genetic diagnosis",
            "MED13L: NON-VERBAL 50% + WIDE MOUTH + GAIT ATAXIA triad — drooling management; AAC; cerebellar MRI often normal despite clinical ataxia",
            "FOXP1: VERBAL DYSPRAXIA + ID + ASD — FOXP1 causes ID whereas FOXP2 typically does NOT; expressive>receptive gap is key diagnostic clue",
            "NDD PANEL/WES/WGS: all 8 genes → molecular diagnosis; CMA first if deletion suspected (SHANK3 22q13.3; ANKRD11 16q24.3; FOXP1 3p13)",
            "AAC (augmentative/alternative communication): mandatory early in SHANK3, ADNP, KAT6A, MED13L — non-verbal majority; do not wait for spoken language",
        ],
        "gene_panel_note": (
            "Comprehensive neurodevelopmental gene panel (2024): SHANK3, SYNGAP1, ADNP, ANKRD11, KAT6A, WAC, MED13L, FOXP1, "
            "plus FOXP2 (dyspraxia without ID), CHD8 (ASD+macrocephaly), DYRK1A (microcephaly+ID), "
            "DDX3X (X-linked ID females), KAT6B (genitopatellar), MBD5 (2q23.1 deletion), "
            "SETBP1 (Schinzel-Giedion), NRXN1 (ASD+schizophrenia); "
            "CMA + gene sequencing complementary — neither alone sufficient for all NDD genes"
        ),
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for Hereditary-Neurodevelopmental-Disorders-Atlas."""
    genes_data = []
    for idx, gene_info in enumerate(ATLAS_GENES):
        gene = gene_info["gene"]
        seed = SEED_BASE + idx
        patients = _generate_patients_for_gene(gene, seed)
        n = len(patients)
        severe_n      = sum(1 for p in patients if p["severity"] == "severe")
        moderate_n    = sum(1 for p in patients if p["severity"] == "moderate")
        mild_n        = sum(1 for p in patients if p["severity"] == "mild")
        mean_iq       = round(sum(p["iq_estimate"] for p in patients) / n, 1)
        non_verbal_n  = sum(1 for p in patients if p["non_verbal"])
        epilepsy_n    = sum(1 for p in patients if p["epilepsy"])
        cardiac_n     = sum(1 for p in patients if p["cardiac_defect"])
        autism_n      = sum(1 for p in patients if p["autism_feature"])
        lymphedema_n  = sum(1 for p in patients if p["lymphedema"])
        behav_n       = sum(1 for p in patients if p["behavioural_dysreg"])
        regression_n  = sum(1 for p in patients if p["regression"])
        mean_age      = round(sum(p["age_at_diagnosis_mo"] for p in patients) / n, 1)
        mutations_seen = list({p["mutation"] for p in patients})
        genes_data.append({
            "gene":              gene,
            "locus":             gene_info["locus"],
            "n_patients":        n,
            "severe_pct":        round(severe_n / n * 100, 1),
            "moderate_pct":      round(moderate_n / n * 100, 1),
            "mild_pct":          round(mild_n / n * 100, 1),
            "mean_iq":           mean_iq,
            "non_verbal_pct":    round(non_verbal_n / n * 100, 1),
            "epilepsy_pct":      round(epilepsy_n / n * 100, 1),
            "cardiac_pct":       round(cardiac_n / n * 100, 1),
            "autism_pct":        round(autism_n / n * 100, 1),
            "lymphedema_pct":    round(lymphedema_n / n * 100, 1),
            "behavioural_pct":   round(behav_n / n * 100, 1),
            "regression_pct":    round(regression_n / n * 100, 1),
            "mean_age_dx_mo":    mean_age,
            "sample_mutations":  mutations_seen[:4],
            "protein":           gene_info["protein"],
            "inheritance":       gene_info["inheritance"][:200],
            "disease_category":  gene_info["disease_category"],
        })
    return {
        "atlas": "Hereditary-Neurodevelopmental-Disorders-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Clinical definitions for Hereditary-Neurodevelopmental-Disorders-Atlas."""
    definitions = [
        {
            "term": "Neurodevelopmental Disorders — Classification and Diagnostic Approach",
            "genes": ["SHANK3", "SYNGAP1", "ADNP", "ANKRD11", "KAT6A", "WAC", "MED13L", "FOXP1"],
            "definition": (
                "HEREDITARY NEURODEVELOPMENTAL DISORDERS (NDD) — OVERVIEW: "
                "DEFINITION: NDD = disorders affecting brain development → impaired intellectual "
                "function, adaptive behaviour, social communication, motor function, language; "
                "CLASSIFICATION BY MECHANISM: "
                "  POSTSYNAPTIC DENSITY: SHANK3 (PSD scaffold); SYNGAP1 (RAS-GAP); "
                "  CHROMATIN/EPIGENETIC: ADNP (SWI/SNF), ANKRD11 (corepressor), KAT6A (HAT), WAC (H2Bub1); "
                "  TRANSCRIPTION: MED13L (Mediator complex); FOXP1 (FOX TF); "
                "DIAGNOSTIC ALGORITHM: "
                "  STEP 1: Phenotypic characterisation: "
                "    Autism? ID severity? Speech absent/present? Epilepsy? Dysmorphia? Behaviour? "
                "  STEP 2: Chromosomal microarray (CMA) first: "
                "    Detects: SHANK3 22q13.3 del; ANKRD11 16q24.3 del; FOXP1 3p13 del; "
                "  STEP 3: Gene panel or WES/WGS: "
                "    Intragenic SHANK3, SYNGAP1, ADNP, ANKRD11, KAT6A, WAC, MED13L, FOXP1; "
                "  STEP 4: If negative: consider repeat CMA (higher resolution) or WGS; "
                "PHENOTYPE TRIGGERS FOR SPECIFIC GENES: "
                "  ABSENT SPEECH + AUTISM 94% → SHANK3 (22q13.3 CMA); "
                "  MYOCLONIC DROP ATTACKS + PHOTOPAROXYSMAL EEG → SYNGAP1; "
                "  AUTISM 90% + p.Tyr719Ter → ADNP (Helsmoortel-Van der Aa); "
                "  MACRODONTIA + SHORT STATURE + DELAYED BA → ANKRD11 (KBG); "
                "  ABSENT SPEECH + FEEDING TUBE + CARDIAC → KAT6A; "
                "  AGGRESSION + SIB + HYPERACTIVITY → WAC (DeSanto-Shinawi); "
                "  NON-VERBAL + WIDE MOUTH + GAIT ATAXIA → MED13L; "
                "  VERBAL DYSPRAXIA + ID (not pure CAS) → FOXP1 (vs FOXP2 = CAS no ID); "
                "PREVALENCE NOTES: "
                "  SHANK3/PMS: ~1:15,000; SYNGAP1: ~1:8,000-10,000 estimated; "
                "  ADNP: ~1:15,000 estimated; ANKRD11/KBG: ~1:10,000-20,000; "
                "  KAT6A: ~1:20,000 estimated; WAC/DeSanto-Shinawi: very rare; "
                "  MED13L: rare; FOXP1: rare;"
            ),
        },
        {
            "term": "Phelan-McDermid vs ADNP vs KAT6A — Absent Speech Differential Diagnosis",
            "genes": ["SHANK3", "ADNP", "KAT6A", "MED13L"],
            "definition": (
                "ABSENT/MINIMAL SPEECH IN NEURODEVELOPMENTAL DISORDERS — DIFFERENTIAL: "
                "SHANK3 (Phelan-McDermid syndrome): "
                "  ABSENT SPEECH: 75-94%; most severe expressive impairment; "
                "  KEY FEATURES: LYMPHEDEMA 50% (UNIQUE — no other NDD gene); "
                "    absent deep tendon reflexes; chronic diarrhoea; regression 40%; "
                "  TESTING: CMA first (22q13.3 deletion 80-85%); SHANK3 seq if CMA negative; "
                "ADNP (Helsmoortel-Van der Aa syndrome): "
                "  ABSENT/MINIMAL SPEECH: majority; autism 90%+; "
                "  KEY FEATURES: p.Tyr719Ter HOTSPOT 25%; sleep disorder 80%; GERD; "
                "    cardiac anomalies 25%; "
                "  TESTING: WES/WGS or NDD panel (intragenic, CMA negative); "
                "KAT6A syndrome: "
                "  ABSENT SPEECH: ~50-70%; feeding difficulties neonatal PROMINENT; "
                "  KEY FEATURES: cardiac CHD 30% (MANDATORY echo); coloboma; "
                "    thrombocytopenia; microcephaly 40%; "
                "  TESTING: WES/WGS or NDD panel; "
                "MED13L syndrome: "
                "  NON-VERBAL: 50%; WIDE MOUTH + GAIT ATAXIA + DROOLING triad; "
                "  KEY FEATURES: facial hypotonia; feeding difficulties; bulbous nose; "
                "    cerebellar MRI often normal despite clinical ataxia; "
                "  TESTING: WES/WGS or NDD panel; "
                "COMMON PRINCIPLE: "
                "  All four → AAC (augmentative/alternative communication) from earliest intervention; "
                "  Never delay AAC while waiting for spoken language to develop; "
                "  Speech therapy AND AAC simultaneously — they are complementary; "
                "LYMPHEDEMA KEY: only SHANK3/PMS → lymphedema; confirm CMA if lymphedema + absent speech; "
                "MACRODONTIA KEY: ANKRD11/KBG only in this group; panoramic X-ray distinguishes from above"
            ),
        },
        {
            "term": "SYNGAP1 Epilepsy — Photoparoxysmal Response and Treatment Algorithm",
            "genes": ["SYNGAP1"],
            "definition": (
                "SYNGAP1 EPILEPSY — MANAGEMENT GUIDE: "
                "PREVALENCE: epilepsy in ~50% of SYNGAP1 individuals; "
                "ONSET: usually 1-3 years of age; febrile seizures common precursor; "
                "SEIZURE TYPES: "
                "  MYOCLONIC-ATONIC (DROP ATTACKS): characteristic; sudden drop; head nod; "
                "  EYELID MYOCLONIA: with/without absence; triggered by photic stimulation; "
                "  ABSENCE SEIZURES: typical and atypical; "
                "  GENERALISED TONIC-CLONIC; "
                "  FOCAL SEIZURES: variable; "
                "  DRAVET-LIKE: febrile seizures → febrile and afebrile generalised; "
                "PHOTOPAROXYSMAL RESPONSE (PPR): "
                "  DEFINITION: EEG paroxysmal activity triggered by intermittent photic stimulation; "
                "  PREVALENCE: ~70% of SYNGAP1 individuals with epilepsy; "
                "  CLINICAL CORRELATE: eyelid myoclonia on photic stimulation; "
                "  PATHGNOMONIC SIGNIFICANCE: PPR + myoclonic-atonic epilepsy → strongly suggests SYNGAP1; "
                "  CRITICAL: routine EEG MUST include photic stimulation; if PPR found in epilepsy+ID → test SYNGAP1; "
                "REGRESSION: "
                "  Acute cognitive/behavioural decline at seizure onset in many; "
                "  Regression may be partially reversible with seizure control; "
                "  Monitor: cognition, language, motor at each seizure event; "
                "TREATMENT: "
                "  FIRST-LINE: Valproate (sodium valproate): "
                "    Effective for myoclonic and absence components; "
                "    Start low, titrate slowly; "
                "    Monitor LFTs + FBC; "
                "  ADJUNCT: Clobazam (benzodiazepine): useful for drop attacks; "
                "  KETOGENIC DIET: consider if ≥2 AEDs failed; good evidence in myoclonic-atonic; "
                "  LEVETIRACETAM: useful adjunct; some tolerability concerns (behaviour); "
                "  VIGABATRIN: CONTRAINDICATED in generalised epilepsy (worsens); "
                "  CARBAMAZEPINE: CONTRAINDICATED — aggravates myoclonic/absence; "
                "  LAMOTRIGINE: use with caution (may worsen myoclonic); "
                "  FENFLURAMINE: emerging evidence in myoclonic-atonic epilepsy; "
                "EEG PROTOCOL: "
                "  Always include: hyperventilation + intermittent photic stimulation; "
                "  Video-EEG for drop attack characterisation; "
                "  Sleep EEG if daytime EEG non-diagnostic"
            ),
        },
        {
            "term": "KBG Syndrome — ANKRD11 Macrodontia, Dental Protocol, and Skeletal",
            "genes": ["ANKRD11"],
            "definition": (
                "KBG SYNDROME (ANKRD11) — COMPLETE CLINICAL GUIDE: "
                "MACRODONTIA — PATHOGNOMONIC: "
                "  UPPER CENTRAL INCISORS: disproportionately large (macrodontia); "
                "  Usually becomes apparent when upper permanent incisors erupt (~6-8 years); "
                "  ENAMEL HYPOPLASIA: surface pitting + hypomineralisation; "
                "  DELAYED DENTAL ERUPTION: all teeth affected; "
                "  PANORAMIC DENTAL X-RAY: mandatory at diagnosis AND annually; "
                "  MANAGEMENT: "
                "    Orthodontist referral from age 6-7; "
                "    Enamel protection: fluoride sealants; "
                "    Risk of malocclusion; extractions may be needed; "
                "BONE AGE: "
                "  DELAYED (opposite of NSD1/EZH2 overgrowth syndromes); "
                "  Bone age 2-3 years behind chronological age; "
                "  Growth hormone: may be considered for significant short stature (specialist); "
                "CERVICAL SPINE: "
                "  C1/C2 (atlanto-axial) anomalies in ~25%; "
                "  RISK: spinal cord compression if subluxation; "
                "  IMAGING MANDATORY: cervical spine MRI/X-ray before: "
                "    Any general anaesthesia; sports; contact activities; "
                "  Physiotherapy: avoid high-impact activities until cleared; "
                "HEARING: "
                "  Conductive + sensorineural hearing loss in ~30%; "
                "  Annual audiological assessment; "
                "  Hearing aids if needed; "
                "  Recurrent otitis media → grommets; "
                "CARDIAC: "
                "  CHD in ~30% (ASD, VSD, other); "
                "  Echocardiogram at diagnosis; "
                "  Cardiology review if abnormal; "
                "CRYPTORCHIDISM: "
                "  ~40% of males; "
                "  Orchidopexy before 18 months; "
                "  Fertility implications; "
                "OPHTHALMOLOGY: "
                "  Strabismus; ptosis; refractive errors; "
                "  Annual review; "
                "BEHAVIOUR: "
                "  Friendly, social personality (characteristic KBG feature); "
                "  Emotional lability; ASD features 30%; ADHD 30%; "
                "  Behavioural support + educational aide; "
                "KBG vs OVERGROWTH SYNDROMES: "
                "  SHORT STATURE + DELAYED BA: KBG (ANKRD11); "
                "  TALL STATURE + ADVANCED BA: Sotos (NSD1), Weaver (EZH2)"
            ),
        },
        {
            "term": "FOXP1 vs FOXP2 — Verbal Dyspraxia Differential and Language Treatment",
            "genes": ["FOXP1", "MED13L"],
            "definition": (
                "FOXP1 vs FOXP2 — VERBAL DYSPRAXIA DIFFERENTIAL DIAGNOSIS: "
                "FOXP2 SYNDROME: "
                "  CHILDHOOD APRAXIA OF SPEECH (CAS): severe verbal dyspraxia; "
                "  INTELLIGENCE: typically NORMAL or borderline; "
                "  ASD: not a core feature; "
                "  LANGUAGE: oral and written language impaired; "
                "  PREVALENCE: ~1:1,000,000 (very rare); "
                "  KEY: FOXP2 = speech disorder WITHOUT significant ID; "
                "FOXP1 SYNDROME (MRDFOXP1): "
                "  INTELLECTUAL DISABILITY: mild-moderate — KEY DISTINCTION FROM FOXP2; "
                "  VERBAL DYSPRAXIA: inconsistent speech errors; groping; "
                "    multisyllabic word production disproportionately impaired; "
                "  LANGUAGE: expressive > receptive impairment; "
                "    Verbal tasks disproportionately harder than visuospatial; "
                "  ASD FEATURES: ~50%; "
                "  MACROCEPHALY: ~40%; "
                "  TESTING: CMA (3p13 microdeletion) then FOXP1 sequencing; "
                "  KEY: FOXP1 = verbal dyspraxia WITH ID + ASD; "
                "KEY DIFFERENTIAL: "
                "  FOXP1: ID + ASD + dyspraxia → neurodevelopmental panel; "
                "  FOXP2: dyspraxia only + normal intelligence → FOXP2 gene testing; "
                "  Never substitute FOXP1 test for FOXP2 test or vice versa; "
                "VERBAL DYSPRAXIA SPEECH THERAPY: "
                "  SPECIFIC APPROACHES (evidence-based): "
                "    DIVA (Dynamic, Integrated View of Articulatory planning): proprioceptive feedback; "
                "    PROMPT (Prompts for Restructuring Oral Muscular Phonetic Targets): tactile cues; "
                "    NUFFIELD Dyspraxia Programme: hierarchical motor planning; "
                "    Rapid Syllable Transition Treatment (ReST): prosody + accuracy; "
                "  AAC: if speech intelligibility insufficient; "
                "    AAC + speech therapy are complementary — not mutually exclusive; "
                "  FREQUENCY: minimum 3×/week for CAS/dyspraxia (high dose required); "
                "FOXP1 EDUCATIONAL NEEDS: "
                "  Verbal output tasks: extra time; oral alternatives (typing/AAC); "
                "  Verbal IQ underestimates overall ability; "
                "  Performance IQ tasks: relatively preserved; "
                "MED13L NON-VERBAL DDx: "
                "  MED13L: 50% non-verbal; WIDE MOUTH + GAIT ATAXIA; "
                "  NOT dyspraxia-predominant like FOXP1 — different motor pattern; "
                "  MED13L: facial hypotonia + drooling + bulbous nose CHARACTERISTIC"
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Neurodevelopmental-Disorders-Atlas",
        "count":       len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:800])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    for g in bd["genes"]:
        print(f"  {g['gene']}: n={g['n_patients']}, severe={g['severe_pct']}%, "
              f"iq={g['mean_iq']}, non_verbal={g['non_verbal_pct']}%, epilepsy={g['epilepsy_pct']}%")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Definition entries: {df['count']}")
    for d in df["definitions"]:
        print(f"  {d['term'][:60]}")
