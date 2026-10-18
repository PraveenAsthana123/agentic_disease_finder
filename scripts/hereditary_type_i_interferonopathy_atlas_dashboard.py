#!/usr/bin/env python3
"""Hereditary-Type-I-Interferonopathy-Atlas -- Complete 8-Gene Reference
Aicardi-Goutières Spectrum / Nucleic Acid Sensing Defects / TREX1-RNASEH2B-RNASEH2A-RNASEH2C-SAMHD1-ADAR-IFIH1-USP18
(TREX1/AGS1-AR+AD-25-35pct-Most-Common-Chilblains-PATHOGNOMONIC-Lupus-Overlap,
 RNASEH2B/AGS2-AR-Most-Common-Subunit-Mildest-AGS-Ambulatory-Preserved,
 RNASEH2A/AGS2-Catalytic-AR-More-Severe-Early-Calcification,
 RNASEH2C/AGS3-AR-Cree-Founder-p.A177T-French-Canadian-Intermediate,
 SAMHD1/AGS5-AR-Singleton-Merten-Dental-Anomalies-PATHOGNOMONIC-Aortic-Calcification,
 ADAR/AGS6-AR-Bilateral-Striatal-Necrosis-PATHOGNOMONIC-AD-GOF-Overlap,
 IFIH1/AGS7-AD-GOF-Singleton-Merten-Joint-Laxity-Glaucoma-Skin-Calcification,
 USP18/Pseudo-TORCH-AR-Fatal-Neonatal-JAKi-Ruxolitinib-LIFE-SAVING):

TREX1  (Three Prime Repair Exonuclease 1; 314aa; 3p21.31; autosomal recessive (biallelic) + autosomal dominant (monoallelic);
        AGS1 — most common AGS gene (~25-35% of all AGS); TREX1 is the major mammalian 3'→5' DNA exonuclease:
          it degrades ssDNA generated during L1-element retrotransposition and during DNA replication/repair;
          TREX1 LOF → accumulation of self-derived cytosolic ssDNA/dsDNA → cGAS activation → STING → IRF3/7 → IFN-β/α cascade →
          IFN-stimulated gene (ISG) score chronically elevated → chronic neuroinflammation + systemic lupus-like autoimmunity;
        BIALLELIC (AR) TREX1 — classical Aicardi-Goutières syndrome:
          (1) NEONATAL PRESENTATION: thrombocytopenia, hepatosplenomegaly, elevated transaminases (mimics congenital infection);
          (2) BRAIN IMAGING: bilateral basal ganglia calcifications + periventricular white matter abnormality + cerebral atrophy;
          (3) CSF LYMPHOCYTOSIS: 5-30 lymphocytes/µL in CSF of a non-infected infant = PATHOGNOMONIC of AGS;
          (4) CHILBLAINS (PERNIO): violaceous, blister-forming skin lesions on acral sites (fingers, toes, ears) triggered by
              cold exposure — PATHOGNOMONIC in AGS1 context; present in 40-50% TREX1 biallelic; mimics frostnip/pernio but
              in a genetically immunocompromised child; diagnosis clinched when chilblains + elevated IFN score + calcifications;
          (5) IFN score (Crow-Manel): RT-PCR of 6 ISGs (IFI27, IFI44L, IFIT1, RSAD2, SIGLEC1, USP18) in whole blood — score > 2.466
              = interferonopathy; highly sensitive and specific; point-of-care triage test;
        MONOALLELIC (AD-heterozygous) TREX1 — systemic lupus erythematosus (SLE) association: ~2-3% of familial SLE;
          haploinsufficiency → partial TREX1 deficiency → lower-level DNA accumulation → SLE without AGS brain disease;
          TREX1 p.D18N (dominant allele) — most common AD-TREX1 variant; lupus nephritis + anti-dsDNA positive;
        TREATMENT: JAK1/2 inhibitors (baricitinib, ruxolitinib) suppress IFN-I signalling → motor stabilisation in ~50%;
          reverse-transcriptase inhibitors (RTIs: stavudine+lamivudine+abacavir) block LINE-1 retrotransposition → reduce
          substrate accumulation → reduce cGAS activation; trial evidence modest; best early (pre-severe neurological damage);
        seed SEED_BASE+0).
RNASEH2B (Ribonuclease H2 Subunit B; 312aa; 13q14.3; autosomal recessive;
          AGS2 — most common RNASEH2 subunit mutation (~15% of AGS); MILDEST AGS phenotype:
            RNase H2 complex (A+B+C trimer) cleaves RNA in RNA:DNA hybrids (R-loops) and processes ribonucleotides
              misincorporated into DNA during replication; RNASEH2B LOF → residual RNase H2 activity retained (B subunit
              is not catalytic — disrupts complex stability rather than catalysis) → mildest IFN signature of all AGS genes;
            CLINICAL PHENOTYPE (milder than other AGS):
              delayed milestone attainment rather than regression;
              ambulatory in the majority (unlike most other AGS genes where ambulation is lost);
              only mild-to-moderate intellectual disability;
              brain MRI: white matter abnormality (less severe calcification than TREX1/RNASEH2A);
              IFN score elevated but often lower than TREX1 or ADAR;
              RNASEH2B patients may be misdiagnosed as cerebral palsy or non-specific white matter disease;
            DIAGNOSTIC CLUE: elevated IFN score in a patient with mild spastic diplegia + white matter changes
              on MRI + no clear perinatal insult → RNASEH2B sequencing; CSF lymphocytosis present in ~50% (lower than TREX1);
          seed SEED_BASE+1).
RNASEH2A (Ribonuclease H2 Subunit A; 299aa; 19p13.2; autosomal recessive;
          AGS4 — CATALYTIC subunit; loss is more severe than RNASEH2B:
            RNASEH2A encodes the catalytic domain of RNase H2 trimeric complex; biallelic LOF → complete absence of
              RNase H2 enzymatic activity → maximal ribonucleotide-in-DNA accumulation + R-loop persistence →
              stronger cGAS-STING activation → higher IFN score than RNASEH2B;
            CLINICAL: more severe phenotype than RNASEH2B;
              earlier and more extensive basal ganglia calcifications;
              more severe intellectual disability;
              loss of ambulation in majority;
              prominent spasticity + dystonia;
              seizures more frequent;
              IFN score typically higher than RNASEH2B;
            NB: RNASEH2A hypomorphic alleles (p.G37S) associated with Aicardi-Goutières with preserved cognitive function
              (residual catalytic activity) — genotype/phenotype correlation important;
          seed SEED_BASE+2).
RNASEH2C (Ribonuclease H2 Subunit C; 164aa; 11q13.1; autosomal recessive;
          AGS3 — SMALLEST RNase H2 subunit; intermediate phenotype between RNASEH2B and RNASEH2A:
            RNASEH2C encodes the C-terminal scaffolding subunit that anchors the RNase H2 trimer to the
              PCNA sliding clamp (via PIP-box motif) — essential for coupling ribonucleotide removal to DNA replication fork;
              RNASEH2C LOF → loss of replication-coupled ribonucleotide excision → ribonucleotides persist in newly
              synthesised DNA → R-loop accumulation → cGAS pathway activation;
            CREE FOUNDER ALLELE: p.A177T — founder mutation in French-Canadian (Cree) population; AGS3 is enriched in
              French-Canadian and Inuit populations due to p.A177T; all patients with this ancestry should have p.A177T
              screened rapidly before full gene sequencing;
            CLINICAL: intermediate between RNASEH2B (mild) and RNASEH2A (severe);
              calcifications present in majority; IFN score elevated; spasticity; variable ambulatory outcome;
              seizures in ~40%; small subset has chilblains (overlap with TREX1 phenotype);
          seed SEED_BASE+3).
SAMHD1  (SAM Domain and HD Domain-Containing Protein 1; 626aa; 20q11.23; autosomal recessive;
         AGS5 + Singleton-Merten Syndrome (SMS) overlap:
           SAMHD1 is a dNTPase (depletes the intracellular dNTP pool limiting reverse transcription) and
             a 3'→5' DNA exonuclease (cleaves ssDNA); SAMHD1 is HIV-1 restriction factor (virological synapse);
             SAMHD1 LOF → excess dNTP availability → L1 retrotransposition enhanced + elevated cytosolic nucleic acid →
             cGAS-STING → IFN-I cascade; additionally acts as a phosphatase for cyclic nucleotides;
           SINGLETON-MERTEN SYNDROME OVERLAP (SAMHD1-associated SMS):
             (1) DENTAL ANOMALIES: hypoplastic enamel, delayed eruption, root resorption, premature tooth loss —
                 PATHOGNOMONIC in AGS5 context; distinct from other AGS genes; dental panoramic X-ray shows
                 abnormal root morphology and enamel hypoplasia;
             (2) AORTIC CALCIFICATION: calcification of aorta + coronary arteries in a child/young adult = PATHOGNOMONIC
                 of SAMHD1 AGS; vascular imaging (CT angiography, cardiac echo) MANDATORY in SAMHD1 patients;
             (3) GLAUCOMA: open-angle glaucoma in children; annual ophthalmology screening MANDATORY;
             (4) PSORIASIFORM SKIN LESIONS: in some SAMHD1 patients (not IFIH1 where psoriasis is predominant);
           IFN score elevated (similar to TREX1 severity); brain imaging: calcifications + white matter changes;
           SAMHD1 is phospho-regulated: CDK2 phosphorylates Thr592 → dephosphorylated form is the active restriction factor;
           VPHU in HIV degrades SAMHD1 (viral evasion mechanism);
           NB: monoallelic SAMHD1 mutations associated with T-cell prolymphocytic leukaemia (T-PLL) via different mechanism;
         seed SEED_BASE+4).
ADAR    (Adenosine Deaminase, RNA-Specific; 1226aa; 1q21.3; autosomal recessive (AGS6) + autosomal dominant GOF (rare);
         AGS6 / Dyschromatosis Symmetrica Hereditaria overlap:
           ADAR1 (ADAR gene product) edits adenosines to inosines in double-stranded RNA (A-to-I editing);
             editing of endogenous dsRNA (especially inverted Alu repeat elements in mRNA 3'UTRs) is essential to
             prevent cytosolic dsRNA from being recognised as non-self by MDA5 (IFIH1) → IFN-I cascade;
             ADAR biallelic LOF → under-edited dsRNA → MDA5 activation (IFIH1 pathway) → IFN-I cascade; thus
             ADAR and IFIH1 (MDA5) act in opposing directions on the same dsRNA substrate;
           BILATERAL STRIATAL NECROSIS — PATHOGNOMONIC MRI finding in ADAR/AGS6:
             bilateral symmetric T2-hyperintensity (then atrophy/cavitation) in the putamen and caudate nucleus;
             presents as acute paroxysmal dystonia with fever → mimics viral encephalitis or metabolic crisis;
             MRI diffusion restriction in striatum during acute episode = ADAR until proven otherwise;
             recurrent dystonic crises triggered by febrile illness; each episode → further striatal loss;
           DYSCHROMATOSIS SYMMETRICA HEREDITARIA (DSH): heterozygous ADAR mutations → DSH (skin pigmentation disorder);
             inherited separately from AGS6; biallelic ADAR → AGS6; monoallelic → DSH; important genetic counselling point;
           AD-GOF ADAR mutations: constitutively active ADAR2 isoform → over-editing → separate spectrum;
           JAKi (baricitinib) most evidence in ADAR/AGS6 for striatal necrosis prevention;
         seed SEED_BASE+5).
IFIH1   (Interferon-Induced Helicase C domain-containing protein 1 = MDA5; 1025aa; 2q24.2; autosomal dominant GOF;
         AGS7 / Singleton-Merten Syndrome Type 2 / Airway-Related MDA5 Gain-of-Function:
           IFIH1 encodes MDA5 (Melanoma Differentiation-Associated gene 5), a cytosolic dsRNA sensor of the RIG-I family;
             normal MDA5: inactive until long dsRNA binding triggers ATP-dependent filament formation → MAVS-IFN signalling;
             IFIH1 AD-GOF mutations → constitutively active MDA5 → continuous IFN-I signalling even in the absence of
             viral dsRNA; opposite mechanism to ADAR (ADAR LOF → MDA5 constitutively activated by under-edited Alu dsRNA;
             IFIH1 GOF → MDA5 constitutively active by intrinsic mutation);
           SINGLETON-MERTEN SYNDROME TYPE 2 (IFIH1-SMS2):
             (1) DENTAL ANOMALIES: enamel hypoplasia + root resorption + premature dental loss (shared with SAMHD1-SMS1);
             (2) PSORIASIFORM SKIN LESIONS: chronic papulosquamous skin disease (unlike SAMHD1 which has vasculopathy);
             (3) GLAUCOMA: childhood open-angle glaucoma (shared with SAMHD1); annual ophthalmology MANDATORY;
             (4) JOINT LAXITY: generalised joint hypermobility / hyperlaxity (unlike SAMHD1);
             (5) AORTIC CALCIFICATION: milder than SAMHD1 but present; cardiac echo annually;
           NEUROLOGICAL: milder than other AGS genes; brain MRI often shows only mild white matter changes; no striatal necrosis;
           IFN score highly elevated; AGS7 may present primarily as dermatological (psoriasis + dental) with only incidental
             elevated IFN score; misdiagnosis as isolated psoriasis or dental dysplasia common;
           AD-GOF → HETEROZYGOUS MUTATION SUFFICIENT; penetrance variable but high;
           JAKi (baricitinib/ruxolitinib) effective for skin, joints and dental inflammation; evidence class 3 (case series);
         seed SEED_BASE+6).
USP18   (Ubiquitin-Specific Peptidase 18; 371aa; 22q11.21; autosomal recessive;
         Pseudo-TORCH Syndrome / Neonatal Onset Interferonopathy — MOST SEVERE TYPE I INTERFERONOPATHY:
           USP18 is the negative regulator of IFN-I signalling: it de-conjugates ISG15 from Jak1 (USP18 isopeptidase
             activity) and directly inhibits IFN-I receptor signalling by binding IFNAR2 and displacing JAK1; biallelic
             USP18 LOF → absent negative feedback on IFN-I pathway → uncontrolled, self-amplifying IFN-I cascade;
           ISG15 accumulates in USP18-deficient cells (because ISG15 is not cleaved from its targets) — ISGylation
             of Jak1 impairs JAK1 recycling → paradoxically the IFN response is constitutively hyperactivated;
           PSEUDO-TORCH SYNDROME — FATAL NEONATAL INTERFERONOPATHY:
             (1) NEONATAL ONSET: first week/days of life — fever, encephalopathy, seizures, multiorgan involvement;
             (2) TORCH mimic: thrombocytopenia, hepatitis, elevated LFTs, cerebral calcifications, CSF pleocytosis
                 → complete TORCH workup negative (hence "pseudo-TORCH");
             (3) IFN score: massively elevated (highest of all AGS genes) — can be 100x above normal;
             (4) BRAIN IMAGING: severe cerebral calcifications + white matter destruction + lissencephaly-like in severe cases;
             (5) FATAL IF UNTREATED: median survival < 1 year without treatment; renal failure + respiratory failure;
           JAKi (RUXOLITINIB) IS LIFE-SAVING: emergency ruxolitinib in neonatal period → IFN signalling suppressed →
             survival possible; must be started before irreversible organ damage; outcome depends on speed of diagnosis;
             steroid pre-treatment reduces cytokine surge before starting JAKi; dosing: ruxolitinib 0.5-2 mg/kg/day;
           DIAGNOSTIC URGENCY: any neonate with TORCH-negative encephalopathy + elevated IFN score → USP18 sequencing
             EMERGENTLY (do not wait for full panel); simultaneously start ruxolitinib empirically;
           USP18 heterozygous carriers are healthy (haploinsufficiency insufficient to cause disease);
           NB: ISG15 (already in MSMD atlas) and USP18 act in the same pathway: ISG15-free JAK1 is functional;
             ISG15 deficiency (MSMD atlas) → reduced ISGylation → ENHANCED USP18 activity → impaired IFN response
             (opposite to USP18-null); the two genes have opposite effects on the same IFN-axis;
         seed SEED_BASE+7).
"""

import random
from typing import Any

SEED_BASE = 3926

GENES = ["TREX1", "RNASEH2B", "RNASEH2A", "RNASEH2C", "SAMHD1", "ADAR", "IFIH1", "USP18"]

GENE_META: dict[str, dict[str, Any]] = {
    "TREX1": {
        "full_name": "Three Prime Repair Exonuclease 1",
        "aa": 314, "locus": "3p21.31", "kda": 36, "inh": "Autosomal recessive (biallelic) / Autosomal dominant (monoallelic)",
        "syndrome": "AGS1 / Systemic Lupus Erythematosus (monoallelic)",
        "mim_gene": "606609", "mim_disease": "225750",
        "iuis_category": "Type I interferonopathy — cGAS-STING pathway (ssDNA accumulation) / IUIS 2022 Category 9: Autoinflammatory",
    },
    "RNASEH2B": {
        "full_name": "Ribonuclease H2 Subunit B",
        "aa": 312, "locus": "13q14.3", "kda": 33, "inh": "Autosomal recessive",
        "syndrome": "AGS2 (mildest RNase H2 subunit deficiency)",
        "mim_gene": "610326", "mim_disease": "610326",
        "iuis_category": "Type I interferonopathy — RNase H2 (ribonucleotide removal) / IUIS 2022 Category 9",
    },
    "RNASEH2A": {
        "full_name": "Ribonuclease H2 Subunit A (catalytic)",
        "aa": 299, "locus": "19p13.2", "kda": 33, "inh": "Autosomal recessive",
        "syndrome": "AGS4 (catalytic subunit — more severe than RNASEH2B)",
        "mim_gene": "606034", "mim_disease": "610329",
        "iuis_category": "Type I interferonopathy — RNase H2 catalytic (R-loop accumulation) / IUIS 2022 Category 9",
    },
    "RNASEH2C": {
        "full_name": "Ribonuclease H2 Subunit C",
        "aa": 164, "locus": "11q13.1", "kda": 17, "inh": "Autosomal recessive",
        "syndrome": "AGS3 (Cree founder p.A177T; PCNA-coupling subunit)",
        "mim_gene": "610330", "mim_disease": "610330",
        "iuis_category": "Type I interferonopathy — RNase H2 (replication-coupled ribonucleotide excision) / IUIS 2022 Category 9",
    },
    "SAMHD1": {
        "full_name": "Sterile Alpha Motif and HD Domain-Containing Protein 1",
        "aa": 626, "locus": "20q11.23", "kda": 72, "inh": "Autosomal recessive",
        "syndrome": "AGS5 / Singleton-Merten Syndrome Type 1 (dNTPase + exonuclease)",
        "mim_gene": "606754", "mim_disease": "612952",
        "iuis_category": "Type I interferonopathy — dNTPase/ssDNA exonuclease / IUIS 2022 Category 9",
    },
    "ADAR": {
        "full_name": "Adenosine Deaminase RNA-Specific (ADAR1 p150/p110 isoforms)",
        "aa": 1226, "locus": "1q21.3", "kda": 136, "inh": "Autosomal recessive (biallelic AGS6) / Autosomal dominant (monoallelic DSH)",
        "syndrome": "AGS6 / Dyschromatosis Symmetrica Hereditaria (DSH; monoallelic)",
        "mim_gene": "146920", "mim_disease": "615010",
        "iuis_category": "Type I interferonopathy — dsRNA A-to-I editing (MDA5 suppression) / IUIS 2022 Category 9",
    },
    "IFIH1": {
        "full_name": "Interferon-Induced Helicase C Domain-Containing Protein 1 (MDA5)",
        "aa": 1025, "locus": "2q24.2", "kda": 116, "inh": "Autosomal dominant (gain-of-function)",
        "syndrome": "AGS7 / Singleton-Merten Syndrome Type 2 / MDA5 Interferonopathy",
        "mim_gene": "606951", "mim_disease": "615846",
        "iuis_category": "Type I interferonopathy — dsRNA sensor MDA5 GOF / IUIS 2022 Category 9",
    },
    "USP18": {
        "full_name": "Ubiquitin-Specific Peptidase 18",
        "aa": 371, "locus": "22q11.21", "kda": 43, "inh": "Autosomal recessive",
        "syndrome": "Pseudo-TORCH Neonatal Interferonopathy (fatal if untreated; JAKi life-saving)",
        "mim_gene": "607057", "mim_disease": "617300",
        "iuis_category": "Type I interferonopathy — IFN-I negative regulator loss / IUIS 2022 Category 9 / neonatal onset",
    },
}

DEFINITIONS = [
    {"term": "Type I Interferonopathy", "definition": "Disease caused by cell-intrinsic constitutive overproduction of type I interferons (IFN-α/β) due to aberrant nucleic acid sensing, mimicking a sustained antiviral response against self; unifying feature of AGS spectrum; hallmark = elevated IFN-stimulated gene (ISG) score in blood"},
    {"term": "Aicardi-Goutières Syndrome (AGS)", "definition": "Mendelian type I interferonopathy characterized by calcifying encephalopathy, CSF lymphocytosis, white matter abnormality, and elevated IFN score; mimics congenital viral infection (TORCH) but is genetic; caused by mutations in TREX1, RNASEH2A/B/C, SAMHD1, ADAR, IFIH1, or related genes"},
    {"term": "IFN Score (Crow-Manel)", "definition": "Quantitative RT-PCR of 6 interferon-stimulated genes (IFI27, IFI44L, IFIT1, RSAD2, SIGLEC1, USP18) in whole blood RNA; score > 2.466 = positive interferonopathy; highly sensitive for all AGS genes; point-of-care triage before genetic testing; score does NOT distinguish which gene is mutated"},
    {"term": "CSF Lymphocytosis (AGS)", "definition": "5-30 lymphocytes/µL in CSF of an infant with no infectious cause identified = PATHOGNOMONIC of AGS; normal CSF: < 5 lymphocytes/µL; in AGS, T-cell infiltration of CSF reflects CNS neuroinflammation; TORCH workup negative clinches genetic aetiology"},
    {"term": "Chilblains (Pernio) in AGS", "definition": "Violaceous, tender, blister-forming acral skin lesions (fingers, toes, ears, nose) triggered by cold exposure; PATHOGNOMONIC of TREX1 (AGS1) in children; distinguished from true frostbite (no cryoinjury required) and from idiopathic pernio by age of onset, associated neurological features, and elevated IFN score"},
    {"term": "Bilateral Basal Ganglia Calcifications", "definition": "Symmetric calcifications in putamen, caudate, globus pallidus, thalamus on brain CT (hyperdense) or SWI-MRI (hypointense); present in most AGS genes but degree varies: TREX1 and RNASEH2A (severe) > RNASEH2C/SAMHD1 (moderate) > RNASEH2B (mild); calcifications are irreversible and represent neuronal loss from chronic IFN-I-driven neuroinflammation"},
    {"term": "Bilateral Striatal Necrosis (ADAR Pathognomonic)", "definition": "Bilateral symmetric T2-hyperintensity (acute ischaemic-like change) → atrophy/cavitation in caudate + putamen; PATHOGNOMONIC of ADAR/AGS6; presents as acute paroxysmal dystonia during febrile illness; MRI diffusion restriction in acute phase; recurrent crises → cumulative striatal loss; distinguishes ADAR from all other AGS genes"},
    {"term": "Pseudo-TORCH Syndrome (USP18)", "definition": "Neonatal presentation mimicking congenital infection: fever, encephalopathy, hepatitis, thrombocytopenia, cerebral calcifications, CSF pleocytosis — but TORCH serology/PCR negative; caused by USP18 biallelic LOF → uncontrolled IFN-I self-amplifying loop; IFN score massively elevated; fatal without ruxolitinib; diagnosis = URGENT neonatal emergency"},
    {"term": "JAK Inhibitor (JAKi) in Interferonopathy", "definition": "Ruxolitinib (JAK1/2 inhibitor) and baricitinib (JAK1/2 inhibitor) suppress IFN-I signalling downstream of the IFNAR receptor; most evidence in AGS: motor stabilisation (not improvement) in ~50%; best benefit when started early before severe neurological damage; MANDATORY in USP18 (neonatal emergency); emerging evidence in TREX1, ADAR, IFIH1"},
    {"term": "Reverse Transcriptase Inhibitors (RTIs) in AGS", "definition": "Nucleoside analogues (stavudine+lamivudine+abacavir triple, as in ANRS AGS trial) block L1/Alu retrotransposon reverse transcription → reduce cytosolic ssDNA accumulation → reduce cGAS activation; trials in TREX1/SAMHD1 patients showing moderate IFN score reduction; not established standard of care but trials ongoing"},
    {"term": "cGAS-STING Pathway", "definition": "Cyclic GMP-AMP synthase (cGAS) senses cytosolic dsDNA/ssDNA → produces cGAMP → STING activation → TBK1/IRF3 → IFN-I production; activated in TREX1 (ssDNA excess), SAMHD1 (dNTP pool excess → retrotransposition), RNASEH2 (ribonucleotides → R-loops → ssDNA); the upstream driver of IFN-I in DNA-sensing interferonopathies"},
    {"term": "Singleton-Merten Syndrome (SMS)", "definition": "Overlap syndrome between SAMHD1 (SMS1) and IFIH1 (SMS2) interferonopathies: dental anomalies (enamel hypoplasia, root resorption, premature tooth loss) + aortic/vascular calcification + glaucoma; SMS1 (SAMHD1): more severe vascular disease; SMS2 (IFIH1): more prominent psoriasiform skin + joint laxity; both have elevated IFN score"},
    {"term": "RNase H2 Complex", "definition": "Heterotrimeric complex of RNASEH2A (catalytic) + RNASEH2B + RNASEH2C; functions: (1) cleaves RNA strand of RNA:DNA hybrids (R-loops) to prevent genomic instability; (2) removes ribonucleotides misincorporated during DNA replication (ribonucleotide excision repair, RER); RNASEH2A = catalytic (severe LOF); RNASEH2B = stability/scaffold (mild LOF); RNASEH2C = PCNA-coupling (intermediate)"},
    {"term": "ADAR1 / A-to-I RNA Editing", "definition": "ADAR1 converts adenosine to inosine in double-stranded RNA, especially in inverted Alu repeat sequences in mRNA 3'UTRs; editing marks endogenous dsRNA as 'self' to prevent MDA5 (IFIH1) recognition; ADAR1 LOF → under-edited dsRNA → MDA5 constitutively activated → IFN-I loop; biallelic → AGS6; monoallelic → Dyschromatosis Symmetrica Hereditaria (skin pigmentation, no encephalopathy)"},
    {"term": "MDA5 / IFIH1 dsRNA Sensor", "definition": "Cytosolic dsRNA helicase sensor; normally activated by long dsRNA (e.g. reovirus); inhibited by ADAR1 editing of endogenous Alu dsRNA; IFIH1 AD-GOF → constitutively active MDA5 without viral dsRNA → continuous IFN-I signalling; NB: IFIH1 GOF and ADAR LOF are functionally convergent (both → MDA5 constitutively active) via different mechanisms"},
]


def _gene_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    n = 40
    meta = GENE_META[gene]

    if gene == "TREX1":
        mean_onset_months = rng.gauss(2, 1)           # neonatal / first 3 months
        mean_dx_lag = rng.gauss(2.0, 1.0)
        calcifications_n      = rng.randint(34, 40)   # 85-100%
        csf_lymphocytosis_n   = rng.randint(28, 36)   # ~70-90%
        white_matter_n        = rng.randint(32, 40)   # 80-100%
        chilblains_n          = rng.randint(16, 22)   # 40-55%
        seizures_n            = rng.randint(20, 30)   # 50-75%
        regression_n          = rng.randint(30, 38)   # 75-95%
        isg_score_elevated_n  = rng.randint(38, 40)   # near-universal
        jaki_tx_n             = rng.randint(18, 26)   # growing use
        pathognomonic = "CHILBLAINS (PERNIO) PATHOGNOMONIC: violaceous acral skin lesions (fingers/toes/ears) triggered by cold in an infant/child with calcifying encephalopathy + elevated IFN score = TREX1 AGS1; chilblains + negative TORCH + basal ganglia calcifications + IFN score > 2.466 = TREX1 biallelic LOF until proven otherwise; AD monoallelic TREX1 → SLE (no brain disease); JAKi + RTIs as treatment"

    elif gene == "RNASEH2B":
        mean_onset_months = rng.gauss(12, 4)          # often missed early — milder
        mean_dx_lag = rng.gauss(4.0, 1.5)             # long diagnostic delay: misdiagnosed as CP
        calcifications_n      = rng.randint(18, 28)   # only 45-70%
        csf_lymphocytosis_n   = rng.randint(16, 24)   # ~40-60%
        white_matter_n        = rng.randint(28, 36)   # 70-90%
        chilblains_n          = rng.randint(4, 10)    # uncommon
        seizures_n            = rng.randint(12, 20)   # 30-50%
        regression_n          = rng.randint(16, 26)   # milder — often delayed milestones not regression
        isg_score_elevated_n  = rng.randint(34, 40)   # still elevated
        jaki_tx_n             = rng.randint(12, 20)
        pathognomonic = "MILDEST AGS PHENOTYPE — RNASEH2B non-catalytic scaffold subunit: retained residual complex activity; many patients AMBULATORY and reach adulthood (unique among AGS genes); white matter > calcification pattern; diagnosed late (misdiagnosed as cerebral palsy or non-specific white matter disease); RNASEH2B is the ONLY AGS gene where majority may ambulate independently; elevated IFN score in child with spastic diplegia + white matter MRI changes + no perinatal insult → sequence RNASEH2B"

    elif gene == "RNASEH2A":
        mean_onset_months = rng.gauss(3, 1.5)         # earlier than RNASEH2B
        mean_dx_lag = rng.gauss(2.5, 1.0)
        calcifications_n      = rng.randint(32, 40)   # 80-100%
        csf_lymphocytosis_n   = rng.randint(24, 34)   # 60-85%
        white_matter_n        = rng.randint(34, 40)   # 85-100%
        chilblains_n          = rng.randint(6, 12)    # less common than TREX1
        seizures_n            = rng.randint(26, 34)   # 65-85%
        regression_n          = rng.randint(32, 38)   # more severe
        isg_score_elevated_n  = rng.randint(38, 40)
        jaki_tx_n             = rng.randint(16, 24)
        pathognomonic = "CATALYTIC SUBUNIT LOSS — most severe RNASEH2 subunit defect: RNASEH2A encodes the enzymatic active site; biallelic LOF = complete RNase H2 absence → maximal ribonucleotide-in-DNA accumulation + R-loop persistence → higher IFN score than RNASEH2B; earlier and more extensive basal ganglia calcifications; loss of ambulation in majority; p.G37S hypomorph = retained partial activity → milder disease (genotype-phenotype counselling critical)"

    elif gene == "RNASEH2C":
        mean_onset_months = rng.gauss(6, 3)
        mean_dx_lag = rng.gauss(3.0, 1.5)
        calcifications_n      = rng.randint(26, 34)   # intermediate
        csf_lymphocytosis_n   = rng.randint(18, 28)
        white_matter_n        = rng.randint(28, 36)
        chilblains_n          = rng.randint(6, 14)    # some overlap
        seizures_n            = rng.randint(14, 22)   # ~40%
        regression_n          = rng.randint(22, 30)   # intermediate
        isg_score_elevated_n  = rng.randint(36, 40)
        jaki_tx_n             = rng.randint(14, 22)
        pathognomonic = "CREE FOUNDER ALLELE p.A177T — PATHOGNOMONIC IN FRENCH-CANADIAN/INUIT ANCESTRY: any patient of Cree, Inuit, or French-Canadian origin with AGS → RNASEH2C p.A177T must be screened FIRST before full panel (can bypass comprehensive sequencing); RNASEH2C couples RNase H2 to PCNA at replication fork via PIP-box; loss = replication-coupled ribonucleotide removal failure; intermediate phenotype between RNASEH2B (mild) and RNASEH2A (severe)"

    elif gene == "SAMHD1":
        mean_onset_months = rng.gauss(6, 3)
        mean_dx_lag = rng.gauss(4.0, 2.0)             # SMS features may present before neurological
        calcifications_n      = rng.randint(26, 34)
        csf_lymphocytosis_n   = rng.randint(18, 28)
        white_matter_n        = rng.randint(24, 34)
        chilblains_n          = rng.randint(8, 14)
        seizures_n            = rng.randint(14, 22)
        regression_n          = rng.randint(22, 30)
        isg_score_elevated_n  = rng.randint(36, 40)
        jaki_tx_n             = rng.randint(14, 22)
        pathognomonic = "SINGLETON-MERTEN SYNDROME TYPE 1: DENTAL ANOMALIES (enamel hypoplasia + root resorption + premature tooth loss) PATHOGNOMONIC in AGS context — panoramic dental X-ray shows abnormal root morphology; AORTIC CALCIFICATION on cardiac imaging in a child = SAMHD1 PATHOGNOMONIC; dental panoramic X-ray + CT aorta MANDATORY in all SAMHD1 patients; GLAUCOMA annual ophtho screening MANDATORY; SAMHD1 also restricts HIV replication (dNTPase) — VPHU viral protein degrades SAMHD1; NB: heterozygous SAMHD1 associated with T-cell prolymphocytic leukaemia (T-PLL) — different mechanism"

    elif gene == "ADAR":
        mean_onset_months = rng.gauss(4, 2)           # AGS6 onset early
        mean_dx_lag = rng.gauss(2.5, 1.0)
        calcifications_n      = rng.randint(30, 38)
        csf_lymphocytosis_n   = rng.randint(20, 30)
        white_matter_n        = rng.randint(30, 38)
        chilblains_n          = rng.randint(8, 14)
        seizures_n            = rng.randint(24, 32)   # striatal crises → paroxysmal dystonia
        regression_n          = rng.randint(28, 36)
        isg_score_elevated_n  = rng.randint(38, 40)
        jaki_tx_n             = rng.randint(18, 26)   # baricitinib most evidence in ADAR
        pathognomonic = "BILATERAL STRIATAL NECROSIS on MRI PATHOGNOMONIC of ADAR/AGS6: bilateral symmetric T2-hyperintensity (diffusion restriction in acute phase) in putamen + caudate nucleus = ADAR until proven otherwise; presents as PAROXYSMAL ACUTE DYSTONIA triggered by febrile illness — each fever episode → further striatal damage; MRI in acute episode is DIAGNOSTIC; ADAR and IFIH1 are functionally convergent: ADAR LOF → under-edited Alu dsRNA → MDA5 activated (same pathway as IFIH1 GOF); baricitinib most evidence for striatal necrosis prevention in ADAR/AGS6; monoallelic ADAR → Dyschromatosis Symmetrica Hereditaria (skin only, no brain disease)"

    elif gene == "IFIH1":
        mean_onset_months = rng.gauss(8, 4)           # variable — often skin/dental first
        mean_dx_lag = rng.gauss(5.0, 2.5)             # misdiagnosed as isolated psoriasis
        calcifications_n      = rng.randint(14, 22)   # milder than other AGS
        csf_lymphocytosis_n   = rng.randint(8, 16)    # less prominent
        white_matter_n        = rng.randint(14, 22)   # mild
        chilblains_n          = rng.randint(10, 18)
        seizures_n            = rng.randint(8, 14)    # less frequent
        regression_n          = rng.randint(10, 18)   # milder neurological
        isg_score_elevated_n  = rng.randint(38, 40)   # IFN score highly elevated
        jaki_tx_n             = rng.randint(18, 26)
        pathognomonic = "SINGLETON-MERTEN SYNDROME TYPE 2: AD-GOF MDA5 — DENTAL ANOMALIES + PSORIASIFORM SKIN + GLAUCOMA + JOINT LAXITY COMBINATION PATHOGNOMONIC; PSORIASIFORM skin lesions (papulosquamous) are the distinctive SMS2 feature (SMS1/SAMHD1 has vasculopathy not psoriasis); glaucoma annual screening MANDATORY; neurological disease milder than other AGS; primary misdiagnosis = isolated psoriasis or dental dysplasia; AD-GOF → HETEROZYGOUS MUTATION SUFFICIENT; IFIH1 GOF and ADAR LOF are mechanistically convergent (both → constitutively active MDA5 → IFN-I); JAKi effective for skin, joints and dental inflammation"

    else:  # USP18
        mean_onset_months = rng.gauss(0.1, 0.05)      # NEONATAL: days to first week of life
        mean_dx_lag = rng.gauss(0.5, 0.3)             # must diagnose within days
        calcifications_n      = rng.randint(36, 40)   # severe: 90-100%
        csf_lymphocytosis_n   = rng.randint(34, 40)   # very high
        white_matter_n        = rng.randint(36, 40)   # severe destruction
        chilblains_n          = rng.randint(4, 10)
        seizures_n            = rng.randint(36, 40)   # nearly universal in neonates
        regression_n          = rng.randint(36, 40)   # catastrophic neonatal encephalopathy
        isg_score_elevated_n  = rng.randint(40, 40)   # MASSIVELY elevated — 100x normal
        jaki_tx_n             = rng.randint(28, 38)   # JAKi is life-saving — most receive it now
        pathognomonic = "PSEUDO-TORCH NEONATAL INTERFERONOPATHY — FATAL WITHOUT JAKi: neonate (first week) with TORCH-negative encephalopathy + thrombocytopenia + hepatitis + cerebral calcifications + IFN score MASSIVELY elevated (100x normal) = USP18 biallelic LOF DIAGNOSTIC EMERGENCY; ruxolitinib must be started IMMEDIATELY — every day of delay = irreversible organ damage; diagnosis = TORCH complete workup negative + IFN score elevated + NGS emergency panel; USP18 removes ISG15 from Jak1 — without USP18, Jak1 ISGylation accumulates → paradoxical uncontrolled IFN-I self-amplification; steroids pre-JAKi to prevent cytokine surge; NEONATAL URGENCY: empirical ruxolitinib while awaiting confirmatory genetics"

    return {
        "gene": gene,
        "n": n,
        "locus": meta["locus"],
        "aa": meta["aa"],
        "kda": meta["kda"],
        "inheritance": meta["inh"],
        "syndrome": meta["syndrome"],
        "mean_onset_months": round(abs(mean_onset_months), 1),
        "mean_dx_lag_years": round(abs(mean_dx_lag), 2),
        "calcifications_n": calcifications_n,
        "calcifications_pct": round(calcifications_n / n * 100, 1),
        "csf_lymphocytosis_n": csf_lymphocytosis_n,
        "csf_lymphocytosis_pct": round(csf_lymphocytosis_n / n * 100, 1),
        "white_matter_n": white_matter_n,
        "white_matter_pct": round(white_matter_n / n * 100, 1),
        "chilblains_n": chilblains_n,
        "chilblains_pct": round(chilblains_n / n * 100, 1),
        "seizures_n": seizures_n,
        "seizures_pct": round(seizures_n / n * 100, 1),
        "regression_n": regression_n,
        "regression_pct": round(regression_n / n * 100, 1),
        "isg_score_elevated_n": isg_score_elevated_n,
        "isg_score_elevated_pct": round(isg_score_elevated_n / n * 100, 1),
        "jaki_tx_n": jaki_tx_n,
        "jaki_tx_pct": round(jaki_tx_n / n * 100, 1),
        "pathognomonic": pathognomonic,
        "mim_gene": meta["mim_gene"],
        "mim_disease": meta["mim_disease"],
        "seed": seed,
    }


def generate_overview() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total_n             = sum(c["n"] for c in cohorts)
    total_calc          = sum(c["calcifications_n"] for c in cohorts)
    total_csf           = sum(c["csf_lymphocytosis_n"] for c in cohorts)
    total_wm            = sum(c["white_matter_n"] for c in cohorts)
    total_chilblains    = sum(c["chilblains_n"] for c in cohorts)
    total_seizures      = sum(c["seizures_n"] for c in cohorts)
    total_regression    = sum(c["regression_n"] for c in cohorts)
    total_isg           = sum(c["isg_score_elevated_n"] for c in cohorts)
    total_jaki          = sum(c["jaki_tx_n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Type-I-Interferonopathy-Atlas",
        "subtitle": "Aicardi-Goutières Spectrum / Nucleic Acid Sensing Defects — TREX1-RNASEH2B-RNASEH2A-RNASEH2C-SAMHD1-ADAR-IFIH1-USP18 (IUIS 2022 Category 9: Autoinflammatory / Type I Interferonopathy)",
        "total_patients": total_n,
        "genes_covered": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "calcifications_pct":     round(total_calc / total_n * 100, 1),
            "csf_lymphocytosis_pct":  round(total_csf / total_n * 100, 1),
            "white_matter_pct":       round(total_wm / total_n * 100, 1),
            "chilblains_pct":         round(total_chilblains / total_n * 100, 1),
            "seizures_pct":           round(total_seizures / total_n * 100, 1),
            "regression_pct":         round(total_regression / total_n * 100, 1),
            "isg_score_elevated_pct": round(total_isg / total_n * 100, 1),
            "jaki_tx_pct":            round(total_jaki / total_n * 100, 1),
        },
        "cohort_summaries": [
            {k: v for k, v in c.items() if k != "pathognomonic"}
            for c in cohorts
        ],
        "cascade_testing_concepts": [
            "INFANT with TORCH-NEGATIVE encephalopathy + calcifications + CSF lymphocytosis → IFN SCORE FIRST (Crow-Manel 6-gene RT-PCR): score > 2.466 = interferonopathy; do NOT wait for full genetics; start JAKi while awaiting confirmatory sequencing in severe cases",
            "TREX1 biallelic: most common AGS gene; CHILBLAINS (violaceous acral lesions, cold-triggered) in an infant with calcifying encephalopathy = PATHOGNOMONIC TREX1; check IFN score; TREX1 heterozygous = SLE workup (no brain disease); AD SLE association: anti-dsDNA + renal disease",
            "RNASEH2 SEVERITY HIERARCHY: RNASEH2A (catalytic, severe) > RNASEH2C (PCNA-coupling, intermediate) > RNASEH2B (scaffold, mildest); RNASEH2B is the ONLY AGS gene where ambulation is preserved in majority; misdiagnosis as cerebral palsy is common for RNASEH2B",
            "CREE/FRENCH-CANADIAN ANCESTRY + AGS → RNASEH2C p.A177T FIRST: founder allele screen before full gene panel; accelerates diagnosis and saves cost; applicable to Inuit populations too",
            "SAMHD1: DENTAL PANORAMIC X-RAY (enamel hypoplasia, root resorption) + CT AORTA (calcification) MANDATORY; annual glaucoma screen; dental anomalies often present before neurological; misdiagnosed as isolated dental dysplasia; Singleton-Merten Syndrome Type 1",
            "ADAR/AGS6: BILATERAL STRIATAL NECROSIS on MRI = ADAR pathognomonic; acute paroxysmal dystonia with fever = febrile seizure DDx → MRI URGENTLY; striatum T2-hyperintensity → ADAR gene sequencing stat; baricitinib most evidence; monoallelic ADAR = DSH (skin only, no brain disease)",
            "IFIH1 (AGS7, AD-GOF): PSORIASIFORM SKIN + DENTAL ANOMALIES + GLAUCOMA + JOINT LAXITY + elevated IFN score = Singleton-Merten Type 2; neurological disease mild/absent; primary referral often dermatology or dentistry; AD-GOF → single mutant allele sufficient; mechanistic overlap with ADAR (both → MDA5 constitutively active)",
            "USP18: ANY NEONATE with TORCH-negative encephalopathy + IFN score MASSIVELY elevated → EMPIRICAL RUXOLITINIB IMMEDIATELY + emergency sequencing; do NOT wait for genetic confirmation; USP18 is a neonatal emergency; ISGylation accumulates → uncontrolled IFN self-amplification; ruxolitinib reverses this; steroid pre-treatment 1-2 days before JAKi start",
            "BCG SAFETY: NOT contraindicated in most type I interferonopathies (unlike PIDs); however: severe SCID-like presentations (USP18, severe ADAR) → delay BCG until immune status clarified; interferonopathies have intact T and B cells (different from PID); IFN-gamma pathway intact (different from MSMD)",
            "IFN SCORE MONITORING: IFN score useful for treatment response monitoring; score should decrease with effective JAKi therapy; persistent elevation despite JAKi → dose increase or switch; serial IFN scores every 3-6 months guide therapy titration across all AGS genes",
        ],
        "iuis_category": "IUIS 2022 Category 9: Autoinflammatory Disorders — Type I Interferonopathies (TREX1/RNASEH2/SAMHD1/ADAR = nucleic acid sensing; IFIH1 = dsRNA sensor GOF; USP18 = IFN-I feedback LOF)",
    }


def generate_breakdown() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    return {
        "atlas": "Hereditary-Type-I-Interferonopathy-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_cohorts": cohorts,
        "diagnostic_groups": {
            "dna_exonuclease_defects": {
                "genes": ["TREX1"],
                "mechanism": "TREX1 3'→5' ssDNA exonuclease LOF → cytosolic ssDNA accumulation → cGAS-STING → IFN-I; also restricts L1 retrotransposon ssDNA",
                "shared_feature": "Most common AGS; chilblains PATHOGNOMONIC; biallelic → AGS; monoallelic → SLE",
                "key_ddx": "Biallelic (AR) → AGS1 with chilblains + brain disease; monoallelic (AD) → lupus only (no brain disease); anti-dsDNA distinguishes monoallelic SLE",
            },
            "rnase_h2_complex": {
                "genes": ["RNASEH2B", "RNASEH2A", "RNASEH2C"],
                "mechanism": "RNase H2 heterotrimer removes ribonucleotides from DNA and resolves R-loops; subunit LOF → ribonucleotide-in-DNA accumulation + R-loop persistence → cGAS-STING",
                "shared_feature": "All three AR; severity = RNASEH2A (catalytic, severe) > RNASEH2C (intermediate) > RNASEH2B (mildest, ambulatory)",
                "key_ddx": "RNASEH2B = only AGS gene where majority ambulate; RNASEH2A = severe early calcifications; RNASEH2C = Cree founder p.A177T",
            },
            "dntpase_ssdna_exonuclease": {
                "genes": ["SAMHD1"],
                "mechanism": "SAMHD1 dNTPase depletes dNTP pool restricting retrotransposition; also 3'→5' DNA exonuclease; LOF → excess retrotransposition → cytosolic ssDNA → IFN-I",
                "shared_feature": "Singleton-Merten Syndrome Type 1: dental anomalies + aortic calcification + glaucoma TRIAD unique to SAMHD1",
                "key_ddx": "Dental anomalies shared with IFIH1 (SMS2) but SMS1 has vascular disease (aorta); SMS2 has psoriasiform skin + joint laxity instead",
            },
            "dsrna_editing_sensing": {
                "genes": ["ADAR", "IFIH1"],
                "mechanism": "ADAR edits Alu dsRNA to prevent MDA5 activation; IFIH1 (MDA5) senses dsRNA; ADAR LOF → under-edited dsRNA → MDA5 activated; IFIH1 GOF → MDA5 constitutively active; both → IFN-I cascade via same MDA5 node",
                "shared_feature": "Both converge on constitutively active MDA5 → IFN-I; functionally convergent from opposite directions",
                "key_ddx": "ADAR LOF (AR biallelic) → AGS6 + striatal necrosis PATHOGNOMONIC; IFIH1 GOF (AD heterozygous) → AGS7 + psoriasiform + dental (no striatal necrosis); baricitinib in ADAR; JAKi in IFIH1",
            },
            "ifn_feedback_loss": {
                "genes": ["USP18"],
                "mechanism": "USP18 deubiquitylates ISG15 from Jak1 and suppresses IFNAR signalling; USP18 LOF → uncontrolled IFN-I self-amplification; NEONATAL emergency",
                "shared_feature": "Most severe interferonopathy; pseudo-TORCH; massively elevated IFN score; fatal without ruxolitinib",
                "key_ddx": "Other AGS genes: onset months-years, survivable untreated; USP18: onset DAYS, fatal without JAKi; neonatal TORCH-negative encephalopathy + IFN score 100x = USP18 ONLY differential",
            },
        },
        "pathognomonic_features": {
            "TREX1_chilblains": "CHILBLAINS (PERNIO) — violaceous acral lesions (cold-triggered) in infant with calcifying encephalopathy + positive IFN score = TREX1 AGS1 pathognomonic; 40-55% of TREX1 biallelic patients; absent in other AGS genes (except occasional in SAMHD1/RNASEH2C)",
            "RNASEH2B_ambulation": "AMBULATORY MAJORITY — RNASEH2B is the ONLY AGS gene where most patients preserve independent ambulation; milder white matter > calcification pattern; misdiagnosis as cerebral palsy; key DDx: no perinatal insult + IFN score elevated + white matter MRI = RNASEH2B",
            "RNASEH2C_cree": "CREE FOUNDER p.A177T — screen before full panel in French-Canadian/Inuit/Cree ancestry; accelerates diagnosis; RNASEH2C couples RNase H2 to PCNA sliding clamp at replication fork",
            "SAMHD1_dental_aortic": "DENTAL ANOMALIES + AORTIC CALCIFICATION = SAMHD1 PATHOGNOMONIC (Singleton-Merten Type 1); panoramic dental X-ray + CT aorta MANDATORY; glaucoma annual screening",
            "ADAR_striatal_necrosis": "BILATERAL STRIATAL NECROSIS on MRI (putamen + caudate T2-hyperintensity) = ADAR/AGS6 PATHOGNOMONIC; acute paroxysmal dystonia with fever; recurrent crises → cumulative striatal loss; MRI in acute febrile dystonic episode is diagnostic",
            "IFIH1_sms2": "PSORIASIFORM SKIN + DENTAL ANOMALIES + GLAUCOMA + JOINT LAXITY = IFIH1/AGS7 (SMS2) pathognomonic quadrad; milder neurological disease; AD-GOF; distinguished from SMS1 (SAMHD1) by skin type and joint involvement",
            "USP18_pseudo_torch": "PSEUDO-TORCH NEONATAL INTERFERONOPATHY = USP18: TORCH-negative neonate + IFN score 100x normal + multiorgan failure = diagnostic emergency; ruxolitinib LIFE-SAVING; every day without treatment = irreversible damage",
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Type-I-Interferonopathy-Atlas",
        "definitions": DEFINITIONS,
        "gene_metadata": GENE_META,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "key_distinguishers": {
            "TREX1_biallelic_vs_monoallelic": "Biallelic TREX1 (AR) → AGS1: calcifying encephalopathy + chilblains + IFN score elevated; monoallelic TREX1 (AD) → SLE only: anti-dsDNA + renal disease, NO brain calcifications; inheritance determines phenotype completely",
            "RNASEH2A_vs_RNASEH2B_severity": "RNASEH2A: catalytic subunit → complete RNase H2 loss → severe AGS; loss of ambulation common; early calcifications; RNASEH2B: scaffolding subunit → partial RNase H2 activity retained → mildest AGS; ambulation preserved; diagnosed late as CP",
            "ADAR_vs_IFIH1_mechanism": "ADAR LOF and IFIH1 GOF are functionally convergent (both → MDA5 constitutively active) but mechanistically opposite; ADAR LOF → under-edited endogenous dsRNA → MDA5 ligand present; IFIH1 GOF → MDA5 hypersensitive receptor; clinical: ADAR = striatal necrosis; IFIH1 = psoriasiform + dental (no striatal necrosis)",
            "SAMHD1_vs_IFIH1_singleton_merten": "SMS1 (SAMHD1): dental anomalies + aortic/vascular calcification (SEVERE) + glaucoma; SMS2 (IFIH1): dental anomalies + psoriasiform skin (PROMINENT) + glaucoma + joint laxity; SMS1 has more severe vascular disease; SMS2 has more prominent dermatological disease; both have elevated IFN score",
            "USP18_vs_other_AGS": "USP18: neonatal onset (days), fatal without JAKi, IFN score 100x normal; other AGS: onset weeks-months, survivable; USP18 is unique = pseudo-TORCH emergency; no other AGS gene presents at birth with multiorgan failure and 100x IFN elevation",
            "AGS_vs_congenital_infection_TORCH": "TORCH positive: specific IgM/PCR for CMV, toxoplasma, rubella, HSV; AGS: all TORCH tests NEGATIVE; AGS: IFN score ELEVATED (TORCH does not cause elevated IFN score in blood); calcifications distribution similar but AGS: bilateral basal ganglia; CMV: periventricular; toxoplasma: diffuse; CSF lymphocytosis in AGS without intracranial pressure signs",
            "JAKi_choice_by_gene": "ADAR (AGS6): baricitinib most published evidence; TREX1 (AGS1): ruxolitinib/baricitinib (trial data); IFIH1 (AGS7): baricitinib/ruxolitinib effective for skin/joints; USP18: ruxolitinib empirically (neonatal emergency); RNASEH2 variants: baricitinib growing use; dosing: paediatric weight-based; monitor for infections, anaemia, LFTs",
        },
        "diagnostic_pathway": [
            "Step 1: Any infant/child with encephalopathy + brain calcifications + CSF lymphocytosis + TORCH-negative → IFN SCORE (Crow-Manel 6-gene RT-PCR) IMMEDIATELY; score > 2.466 = interferonopathy confirmed; parallel TORCH PCR/serology to rule out congenital infection",
            "Step 2: IFN score positive → NGS gene panel (TREX1, RNASEH2A/B/C, SAMHD1, ADAR, IFIH1, USP18) + brain MRI with calcification protocol (T1/T2/SWI/FLAIR + CT for calcifications); USP18 suspected (neonatal, 100x IFN score) → empirical ruxolitinib BEFORE results return",
            "Step 3: Chilblains (acral cold-triggered violaceous lesions) → TREX1 biallelic first; if TREX1 negative → SAMHD1, RNASEH2C; also check TREX1 monoallelic in family members (SLE risk)",
            "Step 4: Bilateral striatal T2-hyperintensity (putamen/caudate) → ADAR/AGS6 sequencing stat; check for paroxysmal dystonia history; baricitinib started while awaiting results; monoallelic ADAR (DSH) screening in family",
            "Step 5: Dental anomalies (enamel hypoplasia + root resorption) + child with encephalopathy → SAMHD1 AND IFIH1 sequencing; panoramic dental X-ray + CT aorta (SAMHD1) vs skin exam for psoriasiform + joint laxity (IFIH1); annual glaucoma in both",
            "Step 6: French-Canadian/Inuit/Cree ancestry + AGS features → RNASEH2C p.A177T targeted sequencing FIRST (cheaper/faster than panel); if negative proceed to full panel",
            "Step 7: Mildest presentation (spastic diplegia + white matter MRI + no perinatal insult + elevated IFN score) → RNASEH2B primary suspicion; this is the AGS gene most commonly missed (misdiagnosed as CP); RNASEH2B ambulation preserved = key positive finding",
            "Step 8: All AGS patients → JAKi assessment: baricitinib (most studied in paediatric AGS) or ruxolitinib; neurological benefit = stabilisation (not reversal); start before severe damage; monitor IFN score for treatment response every 3-6 months; RTIs (stavudine+lamivudine+abacavir) as adjunct in TREX1/SAMHD1",
        ],
    }
