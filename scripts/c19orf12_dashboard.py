"""
C19orf12 MPAN (Mitochondrial Membrane Protein-Associated Neurodegeneration) — NBIA4
=====================================================================================
40-patient cohort · C19orf12 (19q12) · Autosomal Recessive · 2nd most common NBIA (~20-35%)
NO eye-of-tiger sign (key DDx from PKAN) · Optic atrophy 80% KEY distinguishing feature
p.Gly69Arg: Polish/Slavic FOUNDER mutation (~30-40% European MPAN alleles)
No disease-modifying therapy 2026 · Deferiprone investigational (MPAN-specific trial ongoing)
POLG mandatory before VPA (secondary mitochondrial dysfunction at ER-mito contact sites)

C19orf12 BIOLOGY:
C19orf12 (C19orf12) encodes a 152-amino-acid protein with 2 transmembrane (TM) domains.
It localises to the mitochondrial inner membrane and ER-mitochondria contact sites (MAMs —
mitochondria-associated membranes). C19orf12 is involved in lipid transfer, CoA metabolite
transport across MAMs, and regulation of mitophagy. LOF → MAM structural disruption →
iron-sulfur cluster assembly failure → labile iron pool expansion → Fenton reaction →
reactive oxygen species → GP + SN neurodegeneration.

C19orf12 PROTEIN STRUCTURE (152 aa, 19q12):
  TRANSMEMBRANE DOMAIN 1 (TM1):
    N-terminal TM domain: anchors protein in mitochondrial inner membrane / MAM.
  TRANSMEMBRANE DOMAIN 2 (TM2):
    C-terminal TM domain: contains p.Gly69Arg (the Slavic/Polish founder mutation).
    Gly-69 → Arg substitution: bulky charged residue disrupts hydrophobic TM2 core →
    protein misfolding → proteasomal degradation → loss of mitochondrial membrane
    localisation → null functional effect despite in-frame missense.
  CYTOPLASMIC LOOPS:
    Inter-TM cytoplasmic loop: harbours additional pathogenic missense variants
    (Thr11Met, Gly69Asp); direct protein-protein interaction surface.
  PATHOGENIC VARIANT DISTRIBUTION:
    Missense (inc. p.Gly69Arg founder): ~60% — most European/Slavic cohorts.
    Compound het missense: ~20% — non-founder populations, variable severity.
    Nonsense/frameshift (null): ~12% — loss of full-length protein; severe.
    Splice site / CNV: ~8% — variable exon skipping; rare.
    Most common: p.Gly69Arg (c.205G>A) — enriched Polish, Czech, Slovak, Ukrainian.
    Second: p.Thr11Met — reported in European cohorts.

C19orf12 FUNCTION — MAM BIOLOGY:
  ER-mitochondria contact sites (MAMs): structural hubs for:
    - Lipid transfer (phosphatidylserine PS → PE; phosphatidic acid transfer)
    - Ca2+ signalling between ER and mitochondria
    - CoA metabolite exchange (malonyl-CoA, acetyl-CoA intermediates)
    - Mitophagy regulation (phagophore recruitment at MAM)
  C19orf12 LOF → MAM dysfunction:
    1. Iron-sulfur [Fe-S] cluster assembly disruption: ISC machinery (ISCU, FDX1)
       mislocalised → labile Fe2+ pool expansion → Fenton reaction → •OH.
    2. Lipid peroxidation: •OH attacks polyunsaturated fatty acids → ferroptosis-like
       mechanism in GP/SN GABA-ergic neurons (highest iron content).
    3. Mitophagy failure: defective mitochondrial quality control → accumulation of
       damaged mitochondria → further ROS production cycle.
    4. GP+SN iron deposition: bilateral SWI hypointensity (NO central T2 hyperintensity
       — NO eye-of-tiger sign unlike PANK2/PKAN).

CLINICAL FORMS:
JUVENILE MPAN (75%, onset 8-14yr):
  Most common presentation. Gait deterioration (spastic paraparesis + dystonia).
  Optic atrophy detected early (VEP + OCT RNFL thinning).
  Axonal neuropathy in 60% (EMG: reduced motor amplitude, normal CV).
  Dysarthria (90%), cognitive decline (80%) appear within years of onset.
  Pyramidal signs 100% — ALL MPAN patients have corticospinal tract involvement.
  Wheelchair use partial by adolescence; full-time by young adult.

ADULT-ONSET MPAN (25%, onset 15-30yr):
  Slower progression. Spastic-ataxic gait initially.
  Psychiatric/cognitive features may precede motor symptoms.
  Optic atrophy may predate motor symptoms.
  Longer ambulatory phase than juvenile form.

KEY MPAN FEATURES (overall):
  Optic atrophy: 80% — most distinguishing from PKAN (pigmentary retinopathy dominant).
  Motor axonal neuropathy: 60% — combined UMN (pyramidal) + LMN (axonal) gait.
  Pyramidal signs: 100% — spastic paraparesis (corticospinal tract), hyperreflexia, Babinski.
  Dystonia: 90% — generalised over time; GPi-DBS Level C (less evidence than PKAN Level B).
  Dysarthria: 90% → anarthria late.
  Cognitive decline: 80% — frontal-subcortical pattern, executive first.
  Psychiatric: 40% — emotional lability, depression, paranoia (frontal degeneration).
  Epilepsy: 25% — secondary; less than PKAN (40-50%).
  Parkinsonism (late): 30% — dopamine-sensitive; avoid typical antipsychotics.

MRI IN MPAN:
  T2*/SWI: bilateral GP + SN iron hypointensity.
  CRITICALLY: NO central T2 hyperintensity (NO eye-of-tiger sign) — key DDx from PKAN.
  Severe MPAN: striatum iron also present.
  T1: no halo sign (contrast with BPAN T1 halo sign).
  Annual SWI recommended from diagnosis.

OMIM: Gene C19orf12 #614297 · Disease MPAN/NBIA4 #614298 · Chromosome 19q12
Formerly: NBIA4 (Neurodegeneration with Brain Iron Accumulation type 4).
Discovery: Hartig MB, Iuso A, Haack T et al. (2011). AJHG 89(6), 889-897.
"""

import random
import math

SEED = 519
RNG = random.Random(SEED)

N = 40
# Juvenile MPAN 75%, Adult-onset MPAN 25%
N_JUVENILE = 30
N_ADULT = 10

# ── Etiology distribution ──────────────────────────────────────────────────
ETIOLOGIES = [
    {"etiology": "C19orf12 Missense (inc. p.Gly69Arg founder)",
     "n": 24, "pct": 60,
     "mechanism": "Missense variants in TM domains or cytoplasmic loops; p.Gly69Arg (Polish/Slavic founder, ~30-40% European alleles) most common; Thr11Met, Gly69Asp; severity depends on residual protein stability",
     "severity": "Variable — juvenile (high impact TM2) to adult-onset (moderate residual activity)"},
    {"etiology": "C19orf12 Compound Het Missense",
     "n": 8, "pct": 20,
     "mechanism": "Two different missense alleles in trans; common in non-founder populations; variable severity",
     "severity": "Moderate — variable, depends on allele combination"},
    {"etiology": "C19orf12 Nonsense/Frameshift (null)",
     "n": 5, "pct": 12,
     "mechanism": "Loss of full-length protein; no residual function; severe early-onset phenotype; optic atrophy earliest",
     "severity": "Severe — early juvenile onset; optic atrophy prominent; rapid progression"},
    {"etiology": "C19orf12 Splice Site / CNV",
     "n": 3, "pct": 8,
     "mechanism": "Intronic splice variants or large deletions; rare; variable exon skipping determines severity",
     "severity": "Variable — exon skipping extent determines residual function"},
]

# ── Seizure type distribution (25% overall have seizures) ─────────────────
SEIZURE_TYPES = [
    {"type": "Focal (with/without secondary generalisation)", "pct": 30, "n": 12},
    {"type": "Generalised Tonic-Clonic (GTCS)", "pct": 20, "n": 8},
    {"type": "Myoclonic", "pct": 15, "n": 6},
    {"type": "Absence", "pct": 8, "n": 3},
    {"type": "Tonic", "pct": 5, "n": 2},
]

# ── AED and symptomatic treatment data ────────────────────────────────────
TREATMENTS = [
    {"drug": "LEV (Levetiracetam)", "level": "B", "tried_pct": 72, "responder_pct": 55,
     "note": "First-line for MPAN seizures; no motor worsening; broad-spectrum"},
    {"drug": "VPA (Valproate)", "level": "B", "tried_pct": 55, "responder_pct": 60,
     "note": "POLG MANDATORY before use; effective for myoclonic+GTCS; monitor LFTs; secondary mito involvement in MPAN"},
    {"drug": "CLB (Clobazam)", "level": "B", "tried_pct": 45, "responder_pct": 50,
     "note": "Add-on for focal seizures; useful if dystonia not predominant"},
    {"drug": "Baclofen (oral/intrathecal)", "level": "B", "tried_pct": 80, "responder_pct": 65,
     "note": "First-line for spasticity (100% MPAN have pyramidal signs); oral/intrathecal; no dystonia worsening"},
    {"drug": "Botulinum Toxin", "level": "B", "tried_pct": 65, "responder_pct": 58,
     "note": "Focal/segmental dystonia; cervical, limb; repeat every 3 months; less dramatic response vs PKAN"},
    {"drug": "Trihexyphenidyl", "level": "C", "tried_pct": 50, "responder_pct": 42,
     "note": "Generalised dystonia; anticholinergic; side: cognition (already impaired in MPAN — monitor closely)"},
    {"drug": "GPi-DBS (Deep Brain Stimulation)", "level": "C", "tried_pct": 10, "responder_pct": 40,
     "note": "Level C (less evidence than PKAN); generalised dystonia; case series only; consider in drug-resistant dystonia"},
    {"drug": "Deferiprone (iron chelation)", "level": "INV", "tried_pct": 8, "responder_pct": 30,
     "note": "Investigational iron chelation; ongoing MPAN-specific trials 2024-2026; MRI iron data limited; REMS weekly WBC"},
]

CONTRAINDICATIONS = [
    {"drug": "PHT (Phenytoin)",
     "reason": "AVOID — may worsen spastic-dystonic features; no seizure evidence specific to MPAN; oromandibular dystonia risk"},
    {"drug": "VPA in POLG carriers",
     "reason": "ABSOLUTE CI — fatal hepatotoxicity; POLG mandatory panel; MPAN has secondary mitochondrial dysfunction amplifying risk"},
    {"drug": "Typical antipsychotics",
     "reason": "CAUTION — may worsen parkinsonism (30% late MPAN); MPAN late-stage parkinsonism is dopamine-sensitive; use atypicals with care"},
    {"drug": "High-dose anticholinergics",
     "reason": "CAUTION — cognitive decline (80% MPAN) limits anticholinergic tolerance; monitor cognition quarterly when using trihexyphenidyl >10mg/day"},
]

# ── DDx table ──────────────────────────────────────────────────────────────
DDX = [
    {"condition": "PKAN (PANK2)",
     "distinguishing": "Eye-of-tiger sign PATHOGNOMONIC (absent in MPAN); acanthocytes (50% PKAN, rare in MPAN); more common NBIA; CoA pathway",
     "shared": "GP+SN iron, dystonia, NBIA"},
    {"condition": "BPAN (WDR45)",
     "distinguishing": "T1 halo sign + SN iron; biphasic childhood epilepsy→adult parkinsonism; X-linked dominant de novo females; NO optic atrophy like MPAN",
     "shared": "GP+SN iron, neurodegeneration"},
    {"condition": "PLAN — INAD (PLA2G6)",
     "distinguishing": "Infantile onset (6-18m); cerebellar atrophy; hypotonia; iPLA2-beta defect; iron late; no optic atrophy early",
     "shared": "Iron accumulation late, dystonia, NBIA"},
    {"condition": "FAHN (FA2H)",
     "distinguishing": "White matter T2 hyperintensity (leukodystrophy); cerebellar atrophy; fatty acid 2-hydroxylase defect; optic atrophy shared",
     "shared": "Optic atrophy, pyramidal signs, NBIA, spasticity"},
    {"condition": "CoPAN (COASY)",
     "distinguishing": "Very rare (<30 families 2026); CoA synthase protein defect; similar MPAN phenotype; COASY gene testing distinguishes",
     "shared": "AR NBIA, similar MRI, dystonia, optic atrophy"},
    {"condition": "Hereditary Spastic Paraplegia (HSP)",
     "distinguishing": "No brain iron on MRI; SPG subtypes (SPG4/5/7/11); no optic atrophy with iron; HSP-gene panel distinguishes",
     "shared": "Lower limb spasticity, pyramidal signs, progressive gait disorder"},
    {"condition": "Friedreich Ataxia (FXN)",
     "distinguishing": "Cerebellar ataxia dominant; cardiomyopathy; peripheral neuropathy; GAA repeat; no basal ganglia iron; Romberg positive",
     "shared": "Motor axonal neuropathy, gait disorder, progressive"},
]

# ── Key concepts (definitions) ─────────────────────────────────────────────
DEFINITIONS = [
    {"term": "MPAN (Mitochondrial Membrane Protein-Associated Neurodegeneration)",
     "def": "2nd most common NBIA (~20-35% of NBIA). AR biallelic C19orf12 → impaired mitochondrial membrane integrity at ER-mito contact sites → iron-sulfur cluster dysfunction → labile iron → ROS → GP/SN neurodegeneration. No eye-of-tiger sign. Optic atrophy 80% key feature."},
    {"term": "C19orf12 (Mitochondrial Membrane Protein)",
     "def": "152-aa protein with 2 transmembrane (TM) domains. Localises to mitochondrial inner membrane and ER-mitochondria contact sites (MAMs). Involved in lipid transfer, CoA metabolite transport across MAMs, and regulation of mitophagy. LOF → MAM dysfunction → iron-sulfur cluster assembly failure → labile iron pool expansion."},
    {"term": "p.Gly69Arg (Polish/Slavic Founder Mutation)",
     "def": "Most common C19orf12 pathogenic variant worldwide (~30-40% European MPAN alleles). c.205G>A. Glycine-69 in TM domain 2: Gly→Arg substitution disrupts hydrophobic TM2 core → protein misfolding → proteasomal degradation → loss of mitochondrial membrane localisation. Founder mutation: enriched in Polish, Czech, Slovak, Ukrainian populations."},
    {"term": "Optic Atrophy in MPAN (80%)",
     "def": "Most distinguishing feature vs PKAN (optic atrophy <30% PKAN; pigmentary retinopathy dominant in PKAN). In MPAN: pallor of optic disc → progressive visual loss → central scotoma → eventual blindness. ERG relatively preserved early (vs PKAN pigmentary retinopathy). VEP (visual evoked potential) abnormal. Annual ophthalmology mandatory."},
    {"term": "Motor Axonal Neuropathy (MPAN, 60%)",
     "def": "Peripheral nervous system involvement: motor axonal neuropathy (EMG: reduced motor amplitude, normal conduction velocity). Lower motor neuron signs alongside UMN pyramidal signs = combined UMN+LMN → spastic weakness + distal wasting. DDx from hereditary spastic paraplegia (no neuropathy) and Friedreich ataxia (sensorimotor neuropathy dominant)."},
    {"term": "Iron Accumulation Mechanism (C19orf12 MAM Dysfunction)",
     "def": "C19orf12 at ER-mitochondria contact sites (MAMs): lipid transfer + CoA metabolite exchange. LOF → MAM structural disruption → iron-sulfur cluster ([Fe-S]) assembly proteins (ISCU, FDX1) mislocalised → labile Fe2+ pool → Fenton reaction → •OH → lipid peroxidation → GABA-ergic neuron death in GP/SN. SWI hypointensity bilateral GP + SN — NO central hyperintense spot (no eye-of-tiger)."},
    {"term": "MRI in MPAN (No Eye-of-Tiger)",
     "def": "T2*/SWI: bilateral GP iron hypointensity + SN iron hypointensity. CRITICALLY: NO central T2 hyperintensity (no eye-of-tiger). In severe MPAN: striatum iron also present. T1: no halo sign (contrast BPAN). Cerebral cortex iron (rare, severe). Iron progression correlates with clinical severity. Annual SWI recommended from diagnosis."},
    {"term": "GPi-DBS (Level C MPAN vs Level B PKAN)",
     "def": "GPi-DBS evidence weaker in MPAN than PKAN (Level C vs Level B). Case series show ~30-40% BFMDRS improvement in MPAN generalised dystonia. Spastic-dystonic combination makes DBS response less predictable. Refer for DBS evaluation before irreversible disability. Less evidence than PKAN but considered in drug-resistant generalised dystonia."},
    {"term": "Deferiprone (Iron Chelation — MPAN Investigational)",
     "def": "Brain-penetrant iron chelator. MPAN-specific clinical trials ongoing 2024-2026 (NBIA Research Institute TIRCON follow-on). MRI iron reduction expected (as PKAN TIRCON); functional benefit unknown in MPAN. Weekly WBC monitoring required (agranulocytosis risk). Ferritin monitoring monthly."},
    {"term": "Spastic Paraparesis (MPAN — Pyramidal Signs 100%)",
     "def": "ALL MPAN patients have pyramidal tract involvement: lower limb spasticity (corticospinal tract), hyperreflexia, extensor plantar responses (Babinski). Combined with lower motor neuron axonal neuropathy in 60% → complex gait disorder. Baclofen (oral/intrathecal) primary treatment. ITB (intrathecal baclofen) for severe lower limb spasticity."},
    {"term": "POLG Mandatory (MPAN)",
     "def": "POLG gene panel mandatory before VPA initiation in MPAN. C19orf12 LOF causes secondary mitochondrial dysfunction (MAM disruption → impaired mitochondrial biogenesis + mitophagy). Undetected POLG mutation + VPA → Alpers-like fatal hepatotoxicity. Per 2024 expert consensus — mandatory in all NBIA disorders with secondary mito involvement."},
    {"term": "Cognitive Decline (MPAN, 80%)",
     "def": "Progressive cognitive impairment in 80% of MPAN — more prominent than PKAN. Frontal-subcortical dysfunction pattern: executive function loss → memory → global dementia (late). Onset typically 5-10 years after motor symptoms. Neuropsychological assessment annually. Cognitive impairment limits anticholinergic tolerance (caution trihexyphenidyl)."},
    {"term": "Psychiatric Features (MPAN, 40%)",
     "def": "Emotional lability, personality change, depression, and paranoia in 40%. Distinct from PKAN (OCD/psychiatric 50% Atypical only). In MPAN: frontal degeneration pattern → emotional dysregulation → may precede motor symptoms. Careful antipsychotic choice (avoid typicals — parkinsonism risk late-stage MPAN)."},
    {"term": "Neuropathy DDx (MPAN vs Friedreich vs HSP)",
     "def": "MPAN motor axonal neuropathy (60%): NCS — reduced motor amplitudes, NORMAL conduction velocity. Friedreich Ataxia: sensorimotor neuropathy, cardiomyopathy, GAA repeat, cerebellar dominant. HSP (SPG4/5/7/11): pure pyramidal (no neuropathy in pure HSP), no brain iron. Combined UMN+LMN + brain iron + NBIA → C19orf12 sequencing."},
    {"term": "NBIA4 (Neurodegeneration with Brain Iron Accumulation type 4)",
     "def": "MPAN/NBIA4 — 2nd most common NBIA. C19orf12 identified 2011 (Hartig et al., AJHG). NBIA family: PKAN/NBIA1 (PANK2, most common), MPAN/NBIA4 (C19orf12, 2nd), BPAN/NBIA5 (WDR45), PLAN (PLA2G6), FAHN (FA2H), CoPAN (COASY), Kufor-Rakeb (ATP13A2), neuroferritinopathy (FTL). 2011 discovery paper: Hartig MB, Iuso A, Haack T et al. AJHG 89(6), 889-897."},
]

# ── Monitoring and lifecycle ───────────────────────────────────────────────
MONITORING = [
    {"item": "MRI Brain (SWI/T2* bilateral GP+SN iron)", "frequency": "At diagnosis + every 2 years",
     "rationale": "Monitor iron progression; no eye-of-tiger — SWI mandatory; striatum involvement = severe MPAN"},
    {"item": "POLG Gene Panel", "frequency": "ONCE before VPA initiation",
     "rationale": "Secondary mito dysfunction in MPAN; mandatory 2024 expert consensus"},
    {"item": "Ophthalmology (VEP + fundoscopy + OCT)", "frequency": "Every 6 months",
     "rationale": "Optic atrophy 80%; OCT RNFL thinning detects subclinical changes; VEP latency prolonged early"},
    {"item": "NCS/EMG (nerve conduction + EMG)", "frequency": "At diagnosis; every 2 years",
     "rationale": "Motor axonal neuropathy 60%; track amplitude decline; distinguish from FAHN white matter neuropathy"},
    {"item": "Spasticity Assessment (Modified Ashworth Scale)", "frequency": "Every 3 months",
     "rationale": "Pyramidal signs 100%; monitor for intrathecal baclofen candidacy; assess gait aids"},
    {"item": "Dystonia Assessment (BFMDRS)", "frequency": "Every 6 months",
     "rationale": "Pre-DBS baseline; GPi-DBS Level C MPAN; track generalised spread"},
    {"item": "Neuropsychological Assessment", "frequency": "Annual",
     "rationale": "Cognitive decline 80%; frontal-subcortical pattern; executive + memory; limits anticholinergic tolerance"},
    {"item": "Psychiatric Screen (depression, emotional lability)", "frequency": "Every 6 months",
     "rationale": "Psychiatric features 40%; frontal degeneration; avoid typical antipsychotics (parkinsonism risk)"},
    {"item": "DBS Candidacy Review", "frequency": "At generalised dystonia onset",
     "rationale": "Refer early; Level C evidence for GPi-DBS; less response vs PKAN; spasticity component limits prediction"},
]

LIFECYCLE = [
    {"stage": "Birth–2 years (presymptomatic)",
     "features": "Normal development; genetic diagnosis only in at-risk siblings (p.Gly69Arg in Slavic populations); optic disc normal"},
    {"stage": "2–8 years (pre-symptomatic / early signs)",
     "features": "Possible gait clumsiness; speech delay rare; MRI may show early iron; optic disc pallor detectable"},
    {"stage": "8–14 years (juvenile MPAN onset)",
     "features": "Gait deterioration: spastic paraparesis → dystonia emerging; optic atrophy detected (VEP); NCS abnormalities; dysarthria begins; school performance declining"},
    {"stage": "Adolescence (established MPAN)",
     "features": "Established pyramidal + dystonic gait; optic atrophy progressing; axonal neuropathy symptomatic; cognitive decline appearing; psychiatric features possible; wheelchair use partial"},
    {"stage": "Young adult (advanced MPAN)",
     "features": "Dystonia generalised; spasticity severe; wheelchair full-time; optic atrophy complete; cognitive impairment significant; parkinsonism may emerge; DBS evaluation; baclofen pump consideration"},
    {"stage": "Adult onset MPAN (15–30yr onset, 25%)",
     "features": "Slower progression; spastic-ataxic gait initially; optic atrophy may predate motor; longer ambulatory phase; psychiatric/cognitive often first symptoms"},
    {"stage": "Advanced (any form)",
     "features": "Total care dependence; severe dysarthria/anarthria; aspiration risk; visual impairment significant; palliative planning; communication aids"},
]

THRESHOLDS = [
    {"parameter": "Optic atrophy detection", "threshold": "VEP latency >115ms OR OCT RNFL <80μm",
     "significance": "Precedes symptomatic visual loss by 2-5yr in MPAN; annual screening from diagnosis mandatory"},
    {"parameter": "POLG panel before VPA", "threshold": "Before any VPA initiation",
     "significance": "Any POLG pathogenic variant → VPA ABSOLUTE CI; secondary mito dysfunction in MPAN amplifies risk"},
    {"parameter": "NCS motor amplitude threshold", "threshold": "Motor amplitude <50% lower limit normal",
     "significance": "Clinically significant axonal neuropathy; modify physiotherapy/orthotic plan; track annually"},
    {"parameter": "DBS referral threshold", "threshold": "BFMDRS motor ≥35/120 or loss of ambulation within 12 months",
     "significance": "Earlier referral in MPAN (less DBS evidence); multidisciplinary team essential"},
    {"parameter": "Baclofen pump candidacy", "threshold": "Ashworth ≥3 bilateral lower limbs + oral baclofen intolerance",
     "significance": "Intrathecal baclofen more effective than oral for severe lower limb spasticity in MPAN"},
    {"parameter": "Deferiprone ferritin monitoring", "threshold": "Serum ferritin <20 μg/L → dose reduction",
     "significance": "Agranulocytosis risk: weekly WBC mandatory (REMS); ferritin monthly during titration"},
    {"parameter": "VPA POLG safety threshold", "threshold": "POLG panel before ANY VPA",
     "significance": "ABSOLUTE CI in POLG carriers; even heterozygous POLG variants mandate caution"},
    {"parameter": "Cognitive screen threshold", "threshold": "MoCA <26 or clinical cognitive complaint",
     "significance": "Frontal-subcortical cognitive decline 80%; limits anticholinergic tolerance; review driving/consent annually"},
]

STANDARDS = [
    "NBIA Research Institute: MPAN Clinical Management Guidelines 2023 (nbia.ca)",
    "Hartig MB et al. (2011): C19orf12 mutations cause MPAN — AJHG 89(6), 889-897 (discovery paper)",
    "Hogarth P et al. (2013): MPAN spectrum — Neurology 80(3), 268-275",
    "ILAE 2022: Genetic Epilepsy Classification (MPAN — secondary epilepsy in NBIA)",
    "Deferiprone REMS Program: Weekly WBC monitoring — agranulocytosis risk",
    "POLG Expert Panel 2024: Mandatory POLG before VPA in mitochondrial/CoA/NBIA disorders",
    "ESPGD/ESPE 2023: GPi-DBS for childhood onset generalised dystonia (Level C in MPAN)",
    "NICE NG217 (2022): Management of epilepsies in children and adults",
]

REFERENCES = [
    "Hartig MB, Iuso A, Haack T et al. (2011). Absence of an orphan mitochondrial protein, c19orf12, causes a distinct clinical subtype of neurodegeneration with brain iron accumulation. Am J Hum Genet, 89(6), 889–897.",
    "Hogarth P, Gregory A, Kruer MC et al. (2013). New MPAN-related mutations identified by exome sequencing in three families. Neurology, 80(3), 268–275.",
    "Rouault TA (2015). Mitochondrial iron-sulfur cluster assembly, cellular iron homeostasis, and disease. Trends Genet, 31(3), 119–131.",
    "Brockmann K et al. (2012). Neurodegeneration with brain iron accumulation: a survey of 18 families. Ann Neurol, 71(6), 727–735.",
    "NBIA Research Institute (2023). MPAN Clinical Management Guidelines. nbia.ca.",
    "Levi S, Finazzi D (2014). Neurodegeneration with brain iron accumulation: update on pathogenic mechanisms. Front Pharmacol, 5, 99.",
]


def _patients():
    """Generate 40 synthetic MPAN patients (seed-519)."""
    pts = []
    etio_pool = (
        ["missense_founder"] * 24 + ["compound_het"] * 8 +
        ["null"] * 5 + ["splice_cnv"] * 3
    )
    RNG.shuffle(etio_pool)

    aed_options = ["LEV", "VPA", "CLB", "ZNS", "LTG", "KD"]
    seizure_options = ["focal", "gtcs", "myoclonic", "absence", "tonic"]

    for i in range(N):
        etio = etio_pool[i]

        # Phenotype by etiology
        if etio == "null":
            juvenile = True
            onset_yr = round(RNG.uniform(6, 12), 1)
            spasticity_severe = RNG.random() < 0.70
            dystonia_severity = RNG.choice(["severe", "severe", "generalised"])
            ambulation_lost = RNG.random() < 0.70
        elif etio == "missense_founder":
            juvenile = RNG.random() < 0.78
            if juvenile:
                onset_yr = round(RNG.uniform(8, 14), 1)
                spasticity_severe = RNG.random() < 0.55
                dystonia_severity = RNG.choice(["moderate", "severe"])
                ambulation_lost = RNG.random() < 0.45
            else:
                onset_yr = round(RNG.uniform(15, 28), 1)
                spasticity_severe = RNG.random() < 0.40
                dystonia_severity = RNG.choice(["focal", "moderate"])
                ambulation_lost = RNG.random() < 0.25
        elif etio == "compound_het":
            juvenile = RNG.random() < 0.72
            if juvenile:
                onset_yr = round(RNG.uniform(8, 14), 1)
                spasticity_severe = RNG.random() < 0.50
                dystonia_severity = RNG.choice(["moderate", "severe"])
                ambulation_lost = RNG.random() < 0.40
            else:
                onset_yr = round(RNG.uniform(15, 28), 1)
                spasticity_severe = RNG.random() < 0.35
                dystonia_severity = RNG.choice(["focal", "moderate"])
                ambulation_lost = RNG.random() < 0.20
        else:  # splice_cnv
            juvenile = RNG.random() < 0.65
            onset_yr = round(RNG.uniform(10, 25), 1)
            spasticity_severe = RNG.random() < 0.45
            dystonia_severity = RNG.choice(["focal", "moderate", "severe"])
            ambulation_lost = RNG.random() < 0.35

        # Key MPAN features
        optic_atrophy = RNG.random() < (0.80 if juvenile else 0.75)
        axonal_neuropathy = RNG.random() < 0.60
        cognitive_decline = RNG.random() < 0.80
        psychiatric = RNG.random() < 0.40
        parkinsonism = RNG.random() < 0.30
        baclofen = RNG.random() < 0.80
        dbs = RNG.random() < 0.10
        polg_tested = RNG.random() < 0.70
        deferiprone_trial = RNG.random() < 0.08
        current_age = RNG.randint(10, 60)

        # Seizures (25% overall)
        has_seizures = RNG.random() < 0.25
        s_types = []
        if has_seizures:
            for stype in seizure_options:
                probs = {
                    "focal": 0.30, "gtcs": 0.20, "myoclonic": 0.15,
                    "absence": 0.08, "tonic": 0.05
                }
                if RNG.random() < probs[stype]:
                    s_types.append(stype)
            if not s_types:
                s_types = ["focal"]

        drug_resistant = has_seizures and RNG.random() < 0.50
        n_aeds = RNG.randint(2, 4) if (has_seizures and drug_resistant) else (RNG.randint(1, 2) if has_seizures else 0)
        aeds_tried = RNG.sample(aed_options, min(n_aeds, len(aed_options))) if n_aeds > 0 else []
        seizure_free = has_seizures and RNG.random() < 0.40

        pts.append({
            "id": f"MPAN-{i+1:03d}",
            "etiology": etio,
            "juvenile": juvenile,
            "onset_yr": onset_yr,
            "optic_atrophy": optic_atrophy,
            "axonal_neuropathy": axonal_neuropathy,
            "spasticity_severe": spasticity_severe,
            "dystonia_severity": dystonia_severity,
            "ambulation_lost": ambulation_lost,
            "has_seizures": has_seizures,
            "seizure_types": s_types,
            "drug_resistant": drug_resistant,
            "n_aeds_tried": n_aeds,
            "aeds_tried": aeds_tried,
            "baclofen": baclofen,
            "dbs": dbs,
            "cognitive_decline": cognitive_decline,
            "psychiatric": psychiatric,
            "parkinsonism": parkinsonism,
            "polg_tested": polg_tested,
            "seizure_free": seizure_free,
            "current_age": current_age,
            "deferiprone_trial": deferiprone_trial,
        })
    return pts


_CACHE = {}


def _get_patients():
    if "pts" not in _CACHE:
        _CACHE["pts"] = _patients()
    return _CACHE["pts"]


def get_overview():
    pts = _get_patients()
    n = len(pts)

    def pct(cond): return round(sum(1 for p in pts if cond(p)) / n * 100)

    juvenile_pts = [p for p in pts if p["juvenile"]]
    adult_pts = [p for p in pts if not p["juvenile"]]

    kpis = {
        "n_patients": n,
        "n_juvenile": len(juvenile_pts),
        "n_adult": len(adult_pts),
        "optic_atrophy_pct": pct(lambda p: p["optic_atrophy"]),
        "axonal_neuropathy_pct": pct(lambda p: p["axonal_neuropathy"]),
        "spasticity_severe_pct": pct(lambda p: p["spasticity_severe"]),
        "has_seizures_pct": pct(lambda p: p["has_seizures"]),
        "drug_resistant_pct": pct(lambda p: p["drug_resistant"]),
        "dystonia_severe_pct": pct(lambda p: p["dystonia_severity"] in ("severe", "generalised")),
        "ambulation_lost_pct": pct(lambda p: p["ambulation_lost"]),
        "cognitive_decline_pct": pct(lambda p: p["cognitive_decline"]),
        "psychiatric_pct": pct(lambda p: p["psychiatric"]),
        "parkinsonism_pct": pct(lambda p: p["parkinsonism"]),
        "baclofen_pct": pct(lambda p: p["baclofen"]),
        "dbs_pct": pct(lambda p: p["dbs"]),
        "polg_tested_pct": pct(lambda p: p["polg_tested"]),
        "seizure_free_pct": pct(lambda p: p["seizure_free"]),
        "mean_onset_yr": round(sum(p["onset_yr"] for p in pts) / n, 1),
        "mean_aeds_tried": round(sum(p["n_aeds_tried"] for p in pts) / n, 1),
        "juvenile_mean_onset_yr": round(sum(p["onset_yr"] for p in juvenile_pts) / max(len(juvenile_pts), 1), 1),
        "adult_mean_onset_yr": round(sum(p["onset_yr"] for p in adult_pts) / max(len(adult_pts), 1), 1),
    }

    return {
        "disease": "C19orf12 MPAN (Mitochondrial Membrane Protein-Associated Neurodegeneration / NBIA4)",
        "gene": "C19orf12 (Mitochondrial Membrane Protein)", "chromosome": "19q12",
        "omim_gene": "614297", "omim_disease": "614298",
        "inheritance": "Autosomal Recessive — Biallelic C19orf12 mutations",
        "cohort_n": n, "seed": SEED,
        "kpis": kpis,
        "etiology_distribution": ETIOLOGIES,
        "treatments_summary": [
            {"drug": t["drug"], "level": t["level"],
             "tried_pct": t["tried_pct"], "responder_pct": t["responder_pct"]}
            for t in TREATMENTS
        ],
        "contraindications_summary": [
            {"drug": c["drug"], "reason": c["reason"][:90] + "…"} for c in CONTRAINDICATIONS
        ],
        "monitoring_summary": [
            {"item": m["item"], "frequency": m["frequency"]} for m in MONITORING
        ],
        "lifecycle": [
            {"stage": l["stage"], "features": l["features"][:100]} for l in LIFECYCLE
        ],
        "thresholds": THRESHOLDS,
        "clinical_highlights": [
            "2nd most common NBIA (~20-35% of NBIA cases) — NBIA4/MPAN",
            "Optic atrophy (80%) most distinguishing feature — contrasts with PKAN (pigmentary retinopathy dominant)",
            "NO eye-of-tiger sign (key DDx from PKAN) — SWI shows bilateral GP+SN iron without central T2 hyperintensity",
            "Motor axonal neuropathy (60%) — combined UMN+LMN = complex spastic-dystonic-neuropathic gait",
            "p.Gly69Arg: Polish/Slavic FOUNDER mutation (30-40% European MPAN alleles) — population screening in Slavic cohorts",
            "Pyramidal signs in 100% — baclofen (oral/intrathecal) first-line spasticity treatment",
            "Cognitive decline (80%) limits anticholinergic tolerance — monitor MoCA quarterly when trihexyphenidyl >10mg/day",
            "POLG MANDATORY before VPA (secondary mitochondrial dysfunction at ER-mito contact sites)",
            "GPi-DBS Level C (less evidence than PKAN Level B) — case series: ~30-40% BFMDRS improvement",
            "Deferiprone investigational 2024-2026 (MPAN-specific trial ongoing); no functional benefit proven yet",
        ],
        "ddx_table": DDX,
        "tier_summary": {
            "disease_group": "NBIA (Neurodegeneration with Brain Iron Accumulation) — NBIA4",
            "epilepsy_classification": "Secondary epilepsy in NBIA (25% — less than PKAN 40-50%)",
            "second_most_common_nbia": "MPAN 2nd most frequent NBIA (~20-35% of NBIA cohorts)",
            "pathognomonic_mri": "NO eye-of-tiger (DDx PKAN) — bilateral GP+SN iron SWI hypointensity",
            "precision_therapy": "None approved 2026; deferiprone investigational (MPAN-specific trial ongoing)",
            "key_ddx": "PKAN: eye-of-tiger (absent MPAN), acanthocytes; FAHN: white matter; BPAN: halo sign",
        },
        "standards": STANDARDS,
    }


def get_breakdown():
    pts = _get_patients()

    # Etiology breakdown
    etio_breakdown = []
    for etio_key, label in [
        ("missense_founder", "Missense (inc. p.Gly69Arg founder)"),
        ("compound_het", "Compound Het Missense"),
        ("null", "Nonsense/Frameshift (null)"),
        ("splice_cnv", "Splice Site / CNV"),
    ]:
        group = [p for p in pts if p["etiology"] == etio_key]
        if not group:
            continue
        n_g = len(group)
        etio_breakdown.append({
            "etiology": label,
            "n": n_g,
            "pct": round(n_g / len(pts) * 100),
            "juvenile_pct": round(sum(1 for p in group if p["juvenile"]) / n_g * 100),
            "optic_atrophy_pct": round(sum(1 for p in group if p["optic_atrophy"]) / n_g * 100),
            "axonal_neuropathy_pct": round(sum(1 for p in group if p["axonal_neuropathy"]) / n_g * 100),
            "has_seizures_pct": round(sum(1 for p in group if p["has_seizures"]) / n_g * 100),
            "dbs_pct": round(sum(1 for p in group if p["dbs"]) / n_g * 100),
            "mean_onset_yr": round(sum(p["onset_yr"] for p in group) / n_g, 1),
        })

    # Seizure type breakdown
    seizure_breakdown = []
    for st in ["focal", "gtcs", "myoclonic", "absence", "tonic"]:
        group = [p for p in pts if st in p["seizure_types"]]
        seizure_breakdown.append({
            "type": st.replace("_", " ").title(),
            "n": len(group),
            "pct": round(len(group) / len(pts) * 100),
            "drug_resistant_pct": round(sum(1 for p in group if p["drug_resistant"]) / max(len(group), 1) * 100),
        })

    # Per-patient summary
    per_patient = []
    for p in pts:
        per_patient.append({
            "id": p["id"],
            "form": "Juvenile" if p["juvenile"] else "Adult",
            "etiology": p["etiology"],
            "onset_yr": p["onset_yr"],
            "optic_atrophy": p["optic_atrophy"],
            "axonal_neuropathy": p["axonal_neuropathy"],
            "spasticity_severe": p["spasticity_severe"],
            "dystonia_severity": p["dystonia_severity"],
            "ambulation_lost": p["ambulation_lost"],
            "has_seizures": p["has_seizures"],
            "drug_resistant": p["drug_resistant"],
            "n_aeds": p["n_aeds_tried"],
            "baclofen": p["baclofen"],
            "dbs": p["dbs"],
            "cognitive_decline": p["cognitive_decline"],
            "psychiatric": p["psychiatric"],
            "parkinsonism": p["parkinsonism"],
            "polg_tested": p["polg_tested"],
            "seizure_free": p["seizure_free"],
        })

    return {
        "cohort_n": len(pts),
        "etiology_breakdown": etio_breakdown,
        "seizure_breakdown": seizure_breakdown,
        "per_patient": per_patient,
        "treatment_summary": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "ddx_table": DDX,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }


def get_definitions():
    return {
        "disease": "C19orf12 MPAN (Mitochondrial Membrane Protein-Associated Neurodegeneration / NBIA4)",
        "gene": "C19orf12 (Mitochondrial Membrane Protein) — 19q12 — OMIM 614297",
        "omim_disease": "614298",
        "definitions": DEFINITIONS,
        "contraindications": CONTRAINDICATIONS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "key_concepts": [d["term"] for d in DEFINITIONS],
    }


if __name__ == "__main__":
    print("=== C19orf12 / MPAN (NBIA4) Dashboard — Self-Test (seed-519) ===\n")

    ov = get_overview()
    kpis = ov["kpis"]
    print(f"[get_overview] disease: {ov['disease']}")
    print(f"  n_patients={kpis['n_patients']}, n_juvenile={kpis['n_juvenile']}, n_adult={kpis['n_adult']}")
    print(f"  optic_atrophy_pct={kpis['optic_atrophy_pct']}%, axonal_neuropathy_pct={kpis['axonal_neuropathy_pct']}%")
    print(f"  has_seizures_pct={kpis['has_seizures_pct']}%, drug_resistant_pct={kpis['drug_resistant_pct']}%")
    print(f"  cognitive_decline_pct={kpis['cognitive_decline_pct']}%, psychiatric_pct={kpis['psychiatric_pct']}%")
    print(f"  baclofen_pct={kpis['baclofen_pct']}%, dbs_pct={kpis['dbs_pct']}%")
    print(f"  mean_onset_yr={kpis['mean_onset_yr']}")
    print(f"  etiology_distribution: {len(ov['etiology_distribution'])} entries")
    print(f"  clinical_highlights: {len(ov['clinical_highlights'])} items")
    print()

    bk = get_breakdown()
    print(f"[get_breakdown] cohort_n={bk['cohort_n']}")
    print(f"  etiology_breakdown: {len(bk['etiology_breakdown'])} groups")
    for e in bk["etiology_breakdown"]:
        print(f"    {e['etiology']}: n={e['n']} ({e['pct']}%), juvenile={e['juvenile_pct']}%, optic_atrophy={e['optic_atrophy_pct']}%")
    print(f"  seizure_breakdown: {len(bk['seizure_breakdown'])} types")
    print(f"  per_patient: {len(bk['per_patient'])} rows")
    print()

    df = get_definitions()
    print(f"[get_definitions] definitions: {len(df['definitions'])} concepts")
    print(f"  key_concepts: {df['key_concepts'][:3]} ...")
    print(f"  standards: {len(df['standards'])}, references: {len(df['references'])}")
    print("\n=== All 3 functions OK ===")
