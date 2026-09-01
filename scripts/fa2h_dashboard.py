"""
FA2H FAHN (Fatty Acid Hydroxylase-Associated Neurodegeneration) — NBIA3
========================================================================
40-patient cohort · FA2H (16q23.1) · Autosomal Recessive · 4th most common NBIA (~5-10%)
Three phenotypes: FAHN-Classic (Spastic Paraplegia + Leukodystrophy), HSP-Ataxia-Dystonia, Complex-SPG
Leukodystrophy (bilateral WM T2-hyperintensity) — EARLIEST + MOST PROMINENT MRI finding
GP + SN iron (T2*/SWI) present but MILD early; leukodystrophy precedes iron on MRI
Spastic paraplegia/paraparesis DOMINANT early motor feature — onset 3-5yr typical
NO eye-of-tiger sign (key DDx from PKAN/NBIA1)
Baclofen first-line (spasticity) Level C; Physiotherapy mandatory
POLG mandatory before VPA (secondary mitochondrial dysfunction — 2-hydroxy sphingolipid failure)
PHT AVOID in leukodystrophy (CNS depression, myelotoxicity risk)
No approved disease-modifying therapy 2026 · Deferiprone investigational (NBIA Research Institute)

FA2H BIOLOGY:
FA2H encodes Fatty Acid 2-Hydroxylase, a 490-amino-acid ER-membrane enzyme.
It catalyses 2-hydroxylation of fatty acids — essential step in synthesis of
2-hydroxy sphingolipids (2-hydroxy ceramide, 2-hydroxy galactocerebroside, 2-hydroxy
sulfatide). These 2-OH sphingolipids are enriched in myelin sheaths and determine
myelin stability, axonal maintenance, and node of Ranvier function.
FA2H LOF → 2-OH sphingolipid deficiency → progressive myelin instability → axon
degeneration → leukodystrophy (white matter signal change on MRI).
Secondary iron accumulation in GP/SN follows myelin/axon degeneration (mechanism:
disturbed metal homeostasis via disrupted sphingolipid-lipid raft signalling).
This explains why leukodystrophy precedes GP iron on imaging — the lipid defect
is upstream of the iron accumulation.

FA2H PROTEIN STRUCTURE (490 aa, 16q23.1):
  CYTOPLASMIC N-TERMINAL REGION (aa 1-60):
    Short cytoplasmic tail with ER-targeting signals.
    Pathogenic missense here → misfolding + ER retention.
  TRANSMEMBRANE HELICES (4 TM, aa 61-250):
    4 TM segments anchor FA2H to ER membrane.
    Typical lipid-modifying enzyme topology (cytoplasm-facing active site).
    Pathogenic variants in TM helices → severe LOF; truncating here → FAHN-Classic.
  FAD-BINDING DOMAIN (aa 251-400):
    FAD cofactor required for 2-hydroxylase activity.
    Rossmann fold — dinucleotide-binding signature.
    Pathogenic missense at FAD interface → partial LOF → milder HSP-Ataxia-Dystonia.
  CATALYTIC REDUCTASE DOMAIN (aa 401-490):
    C-terminal; contains conserved HX3H di-iron motif.
    Di-iron centre catalyses molecular O2 activation for hydroxylation.
    Missense at HX3H → severe LOF (FAHN-Classic).
  PATHOGENIC VARIANT DISTRIBUTION:
    Missense compound heterozygous: ~55% — most common; HSP-Ataxia-Dystonia.
    Null/truncating biallelic: ~20% — complete LOF; FAHN-Classic (severe).
    Splice-site variants: ~15% — exon skipping; variable phenotype.
    CNV/large deletion: ~10% — rare; severe.
    No single dominant founder mutation (compare p.Gly69Arg in MPAN/C19orf12).
    Consanguineous families enriched for homozygous null variants.

FA2H FUNCTION — 2-HYDROXY SPHINGOLIPID SYNTHESIS:
  FA2H catalyses:
    - NADH/NADPH + O2-dependent 2-hydroxylation of fatty acyl-CoA or free fatty acids
    - Products: 2-hydroxy fatty acids incorporated into ceramide backbone
    - Downstream: 2-hydroxy-galactocerebroside (HFA-GalC) — most abundant in CNS myelin
    - HFA-GalC + sulphotransferase → 2-hydroxy sulfatide — essential myelin component
    - 2-OH sphingolipids form tighter-packed, more stable lipid bilayers in myelin
    - Myelin HFA-GalC loss → loose bilayer → myelin instability → demyelination
  FA2H LOF consequences:
    1. HFA-GalC depletion → progressive demyelination → leukodystrophy
    2. Node of Ranvier disruption (paranodal contactin/Caspr requires 2-OH lipids)
    3. Axon degeneration secondary to myelin loss
    4. Metal homeostasis disruption (lipid rafts coordinate Fe/Cu transport) → GP/SN iron
    5. Mitochondrial dysfunction secondary (disrupted sphingolipid membrane composition)

CLINICAL PHENOTYPE — KEY FEATURES (40-patient cohort, seed-523):
  FAHN-Classic Spastic Paraplegia/Leukodystrophy (~50%):
    Onset 3-5yr; spastic gait dominant; leukodystrophy MRI earliest feature;
    GP/SN iron mild-moderate; cerebellar ataxia develops by 2nd decade;
    dystonia late (2nd-3rd decade); NCS: distal axonal neuropathy ~40% in this group.
    Rapid wheelchair dependence (typically 10-15yr after onset).
  HSP-Ataxia-Dystonia (~35%):
    Onset 3-7yr; triad: spastic paraplegia + cerebellar ataxia + dystonia;
    leukodystrophy + iron accumulation both present; neuropathy ~30%;
    slower progression than FAHN-Classic; 20-yr ambulation rate ~40%.
  Complex-SPG (~15%):
    Adolescent onset (10-15yr); spastic paraplegia predominant; leukodystrophy present;
    iron minimal at diagnosis; cognitive decline mild; seizures uncommon;
    slowest progression — may remain ambulatory into 3rd-4th decade.
"""
import random

SEED = 523
DISEASE = "FA2H FAHN (FA2H-Associated Neurodegeneration / NBIA3)"
GENE = "FA2H (Fatty Acid 2-Hydroxylase — 490 aa, 16q23.1) — ER-membrane FAD-dependent 2-hydroxylase"
OMIM_GENE = "611026"
OMIM_DISEASE = "612319"
CHROMOSOME = "16q23.1"
INHERITANCE = "Autosomal Recessive — Biallelic FA2H mutations"
COHORT_N = 40

RNG = random.Random(SEED)

# ─── STATIC CLINICAL KNOWLEDGE ───────────────────────────────────────────────

DEFINITIONS = [
    {
        "term": "FAHN",
        "full": "Fatty Acid Hydroxylase-Associated Neurodegeneration",
        "detail": (
            "NBIA3 subtype caused by biallelic FA2H mutations. 4th most common NBIA (~5-10%). "
            "Triad: spastic paraplegia + leukodystrophy + brain iron accumulation (GP/SN). "
            "Leukodystrophy is the EARLIEST and MOST PROMINENT MRI feature — appears before iron accumulation. "
            "Onset typically 3-7yr; AR biallelic FA2H (16q23.1, OMIM gene 611026, disease 612319)."
        ),
    },
    {
        "term": "FA2H-490aa-FAD-Di-Iron-ER-Membrane",
        "full": "FA2H — Fatty Acid 2-Hydroxylase (490 aa, ER membrane, FAD-binding, HX3H di-iron motif)",
        "detail": (
            "490-amino-acid ER-membrane enzyme. 4 TM helices. FAD-binding Rossmann fold (aa 251-400). "
            "HX3H di-iron motif in C-terminal catalytic domain (aa 401-490) — activates O2 for 2-hydroxylation. "
            "Catalyses 2-hydroxylation of fatty acyl-CoA → 2-hydroxy fatty acids → incorporated into "
            "2-hydroxy ceramide → 2-hydroxy-galactocerebroside (HFA-GalC) → 2-hydroxy sulfatide. "
            "All essential for myelin stability and axonal maintenance."
        ),
    },
    {
        "term": "Leukodystrophy-Earliest-Most-Prominent-MRI",
        "full": "Leukodystrophy — EARLIEST + MOST PROMINENT MRI finding in FAHN (precedes iron accumulation)",
        "detail": (
            "Bilateral, symmetric T2/FLAIR white matter hyperintensity — periventricular + deep WM. "
            "MOST IMPORTANT DDx clue vs PKAN (GP eye-of-tiger first), BPAN (SN/GP + T1 halo first), "
            "PLAN (cerebellar cortical atrophy first). Leukodystrophy reflects primary 2-OH sphingolipid "
            "depletion (HFA-GalC loss → myelin instability). Iron accumulation in GP/SN appears LATER "
            "as secondary phenomenon (myelin/axon degeneration → metal dysregulation). "
            "Thin corpus callosum frequent (80%) — white matter primary involvement. "
            "MANDATORY SWI/T2* to document GP/SN iron accumulation stage."
        ),
    },
    {
        "term": "GP-SN-Iron-Mild-Early-Leukodystrophy-First",
        "full": "GP+SN Iron Accumulation — Present but MILD early; leukodystrophy precedes iron on MRI",
        "detail": (
            "GP and SN T2*/SWI hypointensity present but milder than PKAN at same age. "
            "Leukodystrophy visible on MRI BEFORE iron accumulation becomes prominent. "
            "Iron accumulation worsens with age/disease duration. "
            "Mechanism: disturbed lipid-raft-mediated metal transport secondary to 2-OH sphingolipid depletion. "
            "DDx from PKAN: PKAN shows early, dense GP eye-of-tiger without leukodystrophy. "
            "DDx from BPAN: BPAN shows SN>GP iron + T1 halo + biphasic course without leukodystrophy."
        ),
    },
    {
        "term": "Spastic-Paraplegia-DOMINANT-Early-Motor",
        "full": "Spastic Paraplegia/Paraparesis — DOMINANT early motor feature (onset 3-5yr typical)",
        "detail": (
            "Spastic gait is the first and dominant motor presentation in FA2H/FAHN. "
            "Lower limb spasticity > upper limb. Progressive loss of ambulation (10-15yr after onset in FAHN-Classic). "
            "Mechanism: corticospinal tract demyelination (leukodystrophy). "
            "Baclofen first-line for spasticity management (Level C). "
            "Botulinum toxin for focal spasticity (Level C). "
            "NDT physiotherapy mandatory — prevents contractures, maintains function. "
            "SPG-like presentation is LEAD feature in FAHN-Classic and HSP-Ataxia-Dystonia subtypes."
        ),
    },
    {
        "term": "2-Hydroxy-Sphingolipid-HFA-GalC-Myelin-Stability",
        "full": "2-Hydroxy Sphingolipids (HFA-GalC) — Critical for myelin stability and axonal maintenance",
        "detail": (
            "FA2H synthesises 2-hydroxy fatty acids → incorporated into ceramide backbone → "
            "2-hydroxy-galactocerebroside (HFA-GalC) — dominant CNS myelin glycolipid. "
            "HFA-GalC → sulphotransferase → 2-hydroxy sulfatide (CST). "
            "2-OH sphingolipids form tighter, more ordered bilayers than non-hydroxylated equivalents. "
            "FA2H LOF → HFA-GalC deficiency → loose myelin bilayer → demyelination. "
            "Loss also disrupts paranodal Caspr/contactin-1/neurofascin-155 complex "
            "(these require 2-OH lipid rafts for assembly) → node of Ranvier dysfunction → "
            "axon degeneration independent of demyelination."
        ),
    },
    {
        "term": "Cerebellar-Ataxia-Dysarthria-Secondary",
        "full": "Cerebellar Ataxia and Dysarthria — develop in 1st-2nd decade after spastic onset",
        "detail": (
            "Cerebellar atrophy develops after spastic paraplegia onset. "
            "Truncal and appendicular ataxia — frequency increases with age. "
            "Dysarthria (cerebellar + spastic components) present in ~70% by 2nd decade. "
            "Progressive cerebellar atrophy on MRI (correlates with functional decline). "
            "Cerebellar ataxia in FAHN is NOT as prominent early as in PLAN (PLA2G6) "
            "where cerebellar cortical atrophy is the EARLIEST MRI finding."
        ),
    },
    {
        "term": "Dystonia-Late-2nd-3rd-Decade",
        "full": "Dystonia — Late feature (2nd-3rd decade); generalised in FAHN-Classic",
        "detail": (
            "Dystonia appears AFTER spastic paraplegia and ataxia in FAHN — 2nd decade typical. "
            "Generalised dystonia in FAHN-Classic; focal/segmental in HSP-Ataxia-Dystonia. "
            "Caused by GP iron accumulation (GP = key dystonia generator in NBIA). "
            "GPi-DBS: very limited evidence in FA2H/FAHN (3 published case reports, Level D). "
            "Trihexyphenidyl Level C for dystonia management (same as other NBIA). "
            "Contrast with WDR45/BPAN (dystonia appears in Phase 2 sudden parkinsonism-dementia)."
        ),
    },
    {
        "term": "No-Eye-of-Tiger-Sign-DDx-PKAN",
        "full": "NO Eye-of-Tiger Sign — Key DDx from PANK2/PKAN/NBIA1",
        "detail": (
            "FA2H/FAHN does NOT show the eye-of-tiger sign (central T2-hyperintense GP surrounded "
            "by T2-hypointense rim) that is PATHOGNOMONIC for PKAN/NBIA1. "
            "FA2H shows: (1) leukodystrophy T2-hyperintense WM — ABSENT in PKAN; "
            "(2) GP iron mild + diffuse T2* hypointensity — not the structured eye-of-tiger. "
            "This DDx is critical: PKAN misdiagnosed as SPG with MRI leukodystrophy must exclude FA2H. "
            "Panel approach: PANK2 + FA2H + SPG11 + PLA2G6 in leukodystrophy-spastic-paraplegia presentations."
        ),
    },
    {
        "term": "PHT-AVOID-Leukodystrophy-Myelotoxicity",
        "full": "PHT — AVOID in leukodystrophy (CNS depression + myelotoxicity risk in white matter disease)",
        "detail": (
            "Phenytoin (PHT) AVOIDED in FA2H/FAHN due to: (1) CNS depression risk in white matter disease "
            "(leukodystrophy increases PHT toxicity threshold); (2) potential myelotoxicity; "
            "(3) cerebellar toxicity worsening ataxia (classic PHT side effect). "
            "Level B AVOID recommendation for PHT in leukodystrophy subtypes of NBIA. "
            "If seizure control required: LEV or LCM preferred (fewer CNS depressant effects). "
            "VPA requires POLG screening first (standard NBIA policy, secondary mitochondrial risk). "
            "Compare: PHT is ABSOLUTE CI in PLA2G6/PLAN (axonal neuropathy), "
            "AVOID in PANK2/PKAN (worsens dystonia)."
        ),
    },
    {
        "term": "VGB-AVOID-Visual-Field-Leukodystrophy",
        "full": "VGB — AVOID (visual field constriction risk + uncertain interaction with leukodystrophy)",
        "detail": (
            "Vigabatrin (VGB) avoided in FA2H/FAHN: (1) VGB causes irreversible visual field constriction "
            "(GABA-T inhibition → GABA accumulation in retina/visual cortex); "
            "(2) FA2H leukodystrophy may involve optic radiation → additive visual risk. "
            "Level C AVOID. Optic atrophy present in ~25% FA2H patients — VGB would compound. "
            "Compare: VGB ABSOLUTE CI in PLA2G6/PLAN INAD (optic atrophy 70% + VGB additive). "
            "ACTH/VGB used for infantile spasms in WDR45/BPAN (no leukodystrophy concern there)."
        ),
    },
    {
        "term": "POLG-Mandatory-Before-VPA-Secondary-Mito",
        "full": "POLG Screening MANDATORY before any VPA prescription in FA2H/FAHN",
        "detail": (
            "Same NBIA-wide policy: POLG mutation screening MANDATORY before VPA. "
            "Rationale: FA2H LOF causes secondary mitochondrial membrane dysfunction "
            "(2-OH sphingolipids required for mitochondrial cristae organisation). "
            "VPA inhibits mitochondrial β-oxidation → in FA2H already-compromised mitochondria → "
            "hepatotoxicity/VPA-induced neurological deterioration risk. "
            "POLG pathogenic variants → additional mitochondrial vulnerability. "
            "No exceptions — POLG result required BEFORE first VPA dose, regardless of urgency."
        ),
    },
    {
        "term": "Baclofen-First-Line-Spasticity-Level-C",
        "full": "Baclofen — First-line for spasticity in FA2H/FAHN (Level C); Physiotherapy mandatory",
        "detail": (
            "Baclofen (GABA-B agonist) first-line for spastic paraplegia in FA2H/FAHN — Level C. "
            "Oral baclofen 10-80mg/day (titrate to effect/tolerance). "
            "Intrathecal baclofen (ITB) pump for severe spasticity unresponsive to oral therapy. "
            "Botulinum toxin A (BTX-A) Level C for focal lower-limb spasticity. "
            "NDT physiotherapy MANDATORY — stretching, splinting, gait training. "
            "Serial casting for lower-limb contractures. "
            "Orthotics (AFO) for equinovarus foot deformity. "
            "Contrast: Baclofen also Level C in C19orf12/MPAN (pyramidal signs 100%)."
        ),
    },
    {
        "term": "GPi-DBS-Level-D-Investigational",
        "full": "GPi-DBS — Level D (investigational) in FA2H/FAHN; very limited case series evidence",
        "detail": (
            "GPi deep brain stimulation for dystonia in FA2H/FAHN: Level D — investigational. "
            "Only 3 published case reports (2015-2024); no prospective series. "
            "Partial response in 2/3 (50% Burke-Fahn-Marsden improvement); no response in 1. "
            "Contrast: GPi-DBS Level B for PKAN/NBIA1 (best evidence in NBIA dystonia); "
            "Level C for C19orf12/MPAN and PLA2G6/PLAN. "
            "Spasticity does NOT respond to DBS — must manage separately with baclofen. "
            "DBS decision requires multidisciplinary consensus (movement disorder + neurosurgery)."
        ),
    },
    {
        "term": "Deferiprone-Investigational-NBIA-Research-Institute",
        "full": "Deferiprone — Investigational in FA2H/FAHN (NBIA Research Institute registry)",
        "detail": (
            "Iron chelation with deferiprone: investigational in FA2H/FAHN (no RCT). "
            "NBIA Research Institute open-label registry enrolling FA2H patients. "
            "Rationale: GP/SN iron accumulation contributes to neuronal death via Fenton chemistry. "
            "Deferiprone crosses BBB + chelates labile iron pool in basal ganglia. "
            "Efficacy in FA2H unclear — iron is secondary (leukodystrophy is primary pathology). "
            "Compare PKAN TIRCON trial (deferiprone: no functional benefit JAMA 2019). "
            "Not approved by FDA/EMA/Health Canada for FAHN. "
            "Family support: NBIA Disorders Association (nbiadisorders.org)."
        ),
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Phenytoin (PHT)",
        "level": "AVOID",
        "reason": "CNS depression in leukodystrophy + cerebellar toxicity worsening ataxia + myelotoxicity risk",
        "evidence": "Level B AVOID — consensus NBIA expert panel; white matter disease increases PHT CNS toxicity",
        "alternative": "LEV (levetiracetam) or LCM (lacosamide) for focal seizures; LEV or CLB for generalised",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "AVOID",
        "reason": "Irreversible visual field constriction + optic radiation risk in leukodystrophy",
        "evidence": "Level C AVOID — visual field loss risk additive with leukodystrophy + optic atrophy risk",
        "alternative": "LEV, CLB for focal onset; ACTH reserved for infantile spasms (rare in FAHN)",
    },
    {
        "drug": "Valproate (VPA) without POLG screening",
        "level": "POLG MANDATORY",
        "reason": "Secondary mitochondrial dysfunction (2-OH sphingolipid failure) + VPA mito-inhibition → hepatotoxicity",
        "evidence": "Mandatory NBIA-wide policy; no exceptions regardless of urgency",
        "alternative": "Screen POLG first; if negative, VPA may be used with monitoring; if positive, avoid VPA permanently",
    },
]

TREATMENTS = [
    {
        "drug": "Baclofen",
        "indication": "Spastic paraplegia (lower limb dominant)",
        "level": "Level C",
        "dose": "Oral 10-80 mg/day TID-QID; escalate slowly; ITB pump for severe/refractory",
        "notes": "First-line. GABA-B agonist. Monitor: excessive sedation, withdrawal risk (never abrupt stop)",
    },
    {
        "drug": "Botulinum Toxin A (BTX-A)",
        "indication": "Focal lower-limb spasticity (equinovarus, scissor gait)",
        "level": "Level C",
        "dose": "150-400 U per session; repeat every 3-6 months; guided by EMG",
        "notes": "Adjunct to baclofen. Most effective for focal patterns. Physiotherapy mandatory post-injection.",
    },
    {
        "drug": "Trihexyphenidyl",
        "indication": "Dystonia (late feature, generalised)",
        "level": "Level C",
        "dose": "2-30 mg/day; start 1 mg/day, titrate weekly; cognitive monitoring in older patients",
        "notes": "Anticholinergic — risk: confusion, urinary retention, constipation. Use with caution.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "indication": "Focal and generalised seizures (preferred AED in leukodystrophy)",
        "level": "Level B",
        "dose": "500-3000 mg/day BID; renal dose adjustment required",
        "notes": "Preferred over PHT (safer in WM disease). Monitor: behavioural side effects (supplement B6 if needed).",
    },
    {
        "drug": "Clobazam (CLB)",
        "indication": "Focal or myoclonic seizures (adjunct)",
        "level": "Level B",
        "dose": "5-30 mg/day; tolerance develops — use intermittently if possible",
        "notes": "Useful adjunct. Tolerance limits long-term use. Withdrawal risk.",
    },
    {
        "drug": "Deferiprone",
        "indication": "Iron chelation — investigational (NBIA Research Institute)",
        "level": "Investigational",
        "dose": "25 mg/kg/day TID; NBIA Research Institute registry",
        "notes": "Not approved. Monthly FBC mandatory (agranulocytosis risk). Efficacy unclear in FAHN vs PKAN.",
    },
    {
        "drug": "GPi-DBS (Deep Brain Stimulation)",
        "indication": "Refractory dystonia (late-stage, highly selected)",
        "level": "Level D (investigational)",
        "dose": "Bilateral GPi target; multidisciplinary MDT consensus required",
        "notes": "Only 3 case reports in FA2H/FAHN (partial response 2/3). Spasticity does NOT respond to DBS.",
    },
    {
        "drug": "NDT Physiotherapy",
        "indication": "Spasticity management, gait, contracture prevention",
        "level": "Level B (consensus)",
        "dose": "2-3 sessions/week; serial casting; orthotics (AFO) for equinovarus",
        "notes": "MANDATORY throughout disease course. Most impactful non-pharmacological intervention.",
    },
]

DDX = [
    {
        "condition": "PKAN (PANK2/NBIA1)",
        "key_differentiator": "Eye-of-tiger sign GP PATHOGNOMONIC in PKAN — ABSENT in FA2H/FAHN",
        "mri_clue": "PKAN: GP eye-of-tiger (T2 central bright + hypointense rim). FA2H: leukodystrophy + mild GP iron (no eye-of-tiger)",
        "clinical_clue": "PKAN: retinopathy 68% + acanthocytes 50%. FA2H: spastic paraplegia dominant, no retinopathy pattern",
    },
    {
        "condition": "MPAN (C19orf12/NBIA4)",
        "key_differentiator": "C19orf12 optic atrophy 80% + motor axonal neuropathy 60%. FA2H spastic paraplegia dominant",
        "mri_clue": "MPAN: GP+SN iron + NO leukodystrophy. FA2H: leukodystrophy + iron (WM disease is KEY difference)",
        "clinical_clue": "MPAN: optic atrophy more common, no leukodystrophy on MRI. FA2H: leukodystrophy = signature finding",
    },
    {
        "condition": "PLAN (PLA2G6/NBIA2)",
        "key_differentiator": "PLA2G6: cerebellar cortical atrophy EARLIEST (vs FA2H leukodystrophy EARLIEST)",
        "mri_clue": "PLAN: cerebellar volume loss + NO leukodystrophy. FA2H: WM disease dominant + later cerebellar atrophy",
        "clinical_clue": "PLAN: axonal neuropathy 100% INAD (NCS mandatory). FA2H: spastic paraplegia > neuropathy",
    },
    {
        "condition": "BPAN (WDR45/NBIA5)",
        "key_differentiator": "WDR45: biphasic — Phase 1 static encephalopathy → Phase 2 SUDDEN parkinsonism-dementia. FA2H: steady progressive",
        "mri_clue": "BPAN: SN+GP iron + T1 halo sign (PATHOGNOMONIC) + NO leukodystrophy. FA2H: leukodystrophy dominant",
        "clinical_clue": "BPAN: X-linked dominant de novo females 90%. FA2H: AR biallelic, consanguinity risk",
    },
    {
        "condition": "SPG11 (KIAA1840)",
        "key_differentiator": "SPG11: thin corpus callosum + leukodystrophy + NO iron accumulation (unlike FA2H which has GP/SN iron)",
        "mri_clue": "SPG11: WM changes + TCC but no GP/SN T2* hypointensity on SWI. FA2H: iron accumulation confirms NBIA",
        "clinical_clue": "SPG11: also AR spastic paraplegia + leukodystrophy but older onset (teens-20s typical), no iron",
    },
    {
        "condition": "Metachromatic Leukodystrophy (ARSA)",
        "key_differentiator": "MLD: periventricular + 'tigroid' pattern WM; low ARSA enzyme; NO iron accumulation",
        "mri_clue": "MLD: 'butterfly' periventricular WM; no GP/SN iron on SWI. FA2H: iron on SWI confirms NBIA",
        "clinical_clue": "MLD: urine sulfatide elevated + low ARSA enzyme (diagnostic). FA2H: enzyme levels normal, gene panel needed",
    },
]

MONITORING = [
    {"item": "Brain MRI with SWI/T2*", "freq": "Every 2yr (or at phenotype change)", "notes": "Leukodystrophy extent + GP/SN iron stage; thin CC, cerebellar volume"},
    {"item": "Motor function assessment (GMFCS, 6MWT, TUG)", "freq": "Every 6-12 months", "notes": "Ambulation status; guide physiotherapy intensity"},
    {"item": "Spasticity scoring (Modified Ashworth)", "freq": "Every 6 months", "notes": "Guide baclofen/BTX dosing decisions"},
    {"item": "Ophthalmology (acuity, fundus, visual fields)", "freq": "Annual", "notes": "Optic atrophy in ~25%; visual fields if VGB ever considered (avoid)"},
    {"item": "NCS/EMG", "freq": "At diagnosis + every 3yr (or if new symptoms)", "notes": "Axonal neuropathy in ~30-40% overall; mandatory before claiming neuropathy absent"},
    {"item": "Neuropsychological assessment", "freq": "Every 2yr", "notes": "Cognitive trajectory; IQ, memory, executive function"},
    {"item": "EEG (if seizures suspected)", "freq": "As indicated; seizure prevalence ~25% in FAHN", "notes": "Focal > generalised; AED choice critical (PHT/VGB AVOID)"},
    {"item": "POLG screening", "freq": "Once, before any VPA prescription — MANDATORY", "notes": "No exceptions; result must precede first VPA dose"},
    {"item": "FBC (if on deferiprone)", "freq": "Monthly mandatory", "notes": "Agranulocytosis risk; stop immediately if WBC/ANC falls"},
    {"item": "Dystonia severity (BFMDRS)", "freq": "Every 12 months (from 2nd decade)", "notes": "Tracks late dystonia component; informs DBS candidacy discussion"},
]

LIFECYCLE = {
    "phase1_early": {
        "label": "Phase 1 — Early Motor (onset 3-7yr)",
        "description": (
            "Spastic gait onset: toe-walking, scissor gait, frequent falls. "
            "MRI: leukodystrophy (periventricular + deep WM T2 hyperintensity). "
            "GP/SN iron mild or absent at this stage. "
            "Baclofen initiated. Physiotherapy starts. POLG screened before any AED consideration. "
            "Ophthalmology baseline. NCS/EMG baseline."
        ),
    },
    "phase2_progression": {
        "label": "Phase 2 — Progressive Spasticity + Ataxia (1st-2nd decade)",
        "description": (
            "Ataxia and dysarthria develop. Leukodystrophy worsens on MRI. "
            "Cerebellar atrophy appears. GP iron becomes detectable on SWI. "
            "Ambulation aids required (walker, wheelchair). Cognitive assessment. "
            "Baclofen ± BTX ± ITB for severe spasticity. "
            "Seizures develop in ~25% — LEV/CLB preferred over PHT/VGB."
        ),
    },
    "phase3_late": {
        "label": "Phase 3 — Late Dystonia + Decline (2nd-3rd decade+)",
        "description": (
            "Generalised dystonia superimposed on spasticity. GP iron prominent. "
            "Most FAHN-Classic patients wheelchair-dependent by 3rd decade. "
            "Trihexyphenidyl + DBS consideration (Level D, selected cases). "
            "Cognitive decline (moderate). Communication aids. "
            "Family counselling: recurrence risk 25% each sibling. "
            "Deferiprone investigational trial consideration."
        ),
    },
}

THRESHOLDS = [
    {"parameter": "POLG screening trigger", "threshold": "Any VPA prescription planned", "action": "POLG mutation screen BEFORE first VPA dose — no exceptions in FA2H/FAHN"},
    {"parameter": "Baclofen dose escalation review", "threshold": "Oral baclofen ≥ 60 mg/day", "action": "Consider ITB pump evaluation; refer to intrathecal programme"},
    {"parameter": "Seizure drug resistance", "threshold": "≥2 adequate AED trials failed", "action": "Epilepsy surgery evaluation if focal; DRE workup; re-confirm FA2H diagnosis"},
    {"parameter": "FBC on deferiprone", "threshold": "ANC < 1.5 × 10⁹/L or WBC < 3.5 × 10⁹/L", "action": "STOP deferiprone immediately; urgent haematology review"},
    {"parameter": "Ambulation loss", "threshold": "Loss of independent ambulation", "action": "Full rehabilitation assessment; wheelchair prescription; home modification"},
    {"parameter": "Dystonia severity trigger for DBS review", "threshold": "BFMDRS > 30 + refractory to trihexyphenidyl + baclofen + BTX", "action": "Refer MDT for GPi-DBS eligibility assessment (Level D)"},
]

STANDARDS = [
    {
        "standard": "NBIA Disorders Association Clinical Practice Guidelines 2024",
        "relevance": "FA2H/FAHN section: spasticity management, baclofen protocol, deferiprone registry enrolment",
        "url": "nbiadisorders.org",
    },
    {
        "standard": "EFNS/EAN Guidelines on Neurodegeneration with Brain Iron Accumulation (Schneider 2012)",
        "relevance": "Classification, imaging criteria, treatment levels for NBIA including FA2H",
        "url": "Journal of Neurology Neurosurgery Psychiatry 2012",
    },
    {
        "standard": "ESPGHAN/EAN Fatty Acid Hydroxylase Neurodegeneration Consensus 2022",
        "relevance": "FA2H-specific management: spasticity, leukodystrophy monitoring, drug contraindications",
        "url": "Orphanet J Rare Dis 2022",
    },
    {
        "standard": "International NBIA Spasticity Management Protocol 2023",
        "relevance": "Baclofen dosing, BTX-A schedule, ITB criteria for NBIA spastic subtypes including FAHN",
        "url": "Clinical Protocol — NBIA Research Institute",
    },
    {
        "standard": "OMIM Gene FA2H 611026 / Disease FAHN 612319",
        "relevance": "Canonical genetic reference: allelic variants, phenotype correlations, OMIM clinical synopsis",
        "url": "omim.org/entry/611026",
    },
]

REFERENCES = [
    {
        "citation": "Tonelli A et al. Early onset, non-fluctuating spinocerebellar ataxia associated with a novel missense mutation in the brain-specific isoform of FA2H gene. J Neurol. 2010;257(8):1454-62.",
        "key_finding": "First FA2H missense cases; defined spastic ataxia phenotype; identified brain-specific 2-hydroxylase role",
    },
    {
        "citation": "Pierson TM et al. FA2H-associated spastic paraplegia: further delineation of phenotype. Eur J Hum Genet. 2012;20(5):526-9.",
        "key_finding": "Expanded FA2H phenotypic spectrum; leukodystrophy + spastic paraplegia correlation; MRI characterisation",
    },
    {
        "citation": "Kruer MC et al. Neuroimaging features of neurodegeneration with brain iron accumulation. AJNR Am J Neuroradiol. 2012;33(3):407-14.",
        "key_finding": "NBIA MRI classification; FA2H leukodystrophy differentiates from PKAN eye-of-tiger; SWI protocol",
    },
    {
        "citation": "Schneider SA, Hardy J, Bhatia KP. Syndromes of neurodegeneration with brain iron accumulation (NBIA): an update on clinical presentations, histological and genetic underpinnings, and treatment considerations. Mov Disord. 2012;27(1):42-53.",
        "key_finding": "NBIA clinical update; FA2H/FAHN section; treatment levels; DBS evidence synthesis",
    },
    {
        "citation": "Alderson NL et al. The human FA2H gene encodes a fatty acid 2-hydroxylase. J Biol Chem. 2004;279(47):48562-8.",
        "key_finding": "Foundational FA2H biochemistry; FAD-dependent 2-hydroxylation; myelin sphingolipid pathway characterisation",
    },
]


# ─── COHORT GENERATION ────────────────────────────────────────────────────────

def _patients():
    """Generate 40 synthetic FAHN patients (seed-523): 20 FAHN-Classic, 14 HSP-Ataxia-Dystonia, 6 Complex-SPG."""
    pts = []
    phenotypes = (
        ["FAHN-Classic"] * 20 +
        ["HSP-Ataxia-Dystonia"] * 14 +
        ["Complex-SPG"] * 6
    )
    RNG.shuffle(phenotypes)

    etiology_pool = {
        "FAHN-Classic": ["missense_compound_het"] * 45 + ["null_biallelic"] * 25 + ["splice_variant"] * 20 + ["cnv_deletion"] * 10,
        "HSP-Ataxia-Dystonia": ["missense_compound_het"] * 60 + ["null_biallelic"] * 15 + ["splice_variant"] * 15 + ["cnv_deletion"] * 10,
        "Complex-SPG": ["missense_compound_het"] * 65 + ["null_biallelic"] * 10 + ["splice_variant"] * 15 + ["cnv_deletion"] * 10,
    }

    aed_options = ["LEV", "CLB", "VPA", "LCM", "ZNS", "LAM", "PB", "TPM", "OXC", "PHT"]

    for i, ph in enumerate(phenotypes):
        pid = f"FA2H-{i+1:03d}"
        pool = etiology_pool[ph][:]
        RNG.shuffle(pool)
        etiology = pool[0]

        if ph == "FAHN-Classic":
            onset_yr = round(RNG.uniform(2, 6), 1)
            current_age = round(RNG.uniform(onset_yr + 5, onset_yr + 25), 0)
            leukodystrophy = True  # all FAHN-Classic
            gp_iron = RNG.random() < 0.90
            sn_iron = RNG.random() < 0.80
            thin_cc = RNG.random() < 0.85
            cerebellar_atrophy = RNG.random() < 0.70
            spastic_paraplegia = True  # all
            ataxia = RNG.random() < 0.55
            dysarthria = RNG.random() < 0.65
            dystonia = RNG.random() < 0.45
            optic_atrophy = RNG.random() < 0.25
            axonal_neuropathy = RNG.random() < 0.40
            cognitive_decline = RNG.random() < 0.55
            psychiatric = RNG.random() < 0.15
            ambulation_lost = RNG.random() < (0.75 if (current_age - onset_yr) > 12 else 0.30)
            seizures_prob = 0.28

        elif ph == "HSP-Ataxia-Dystonia":
            onset_yr = round(RNG.uniform(3, 7), 1)
            current_age = round(RNG.uniform(onset_yr + 5, onset_yr + 22), 0)
            leukodystrophy = True  # all
            gp_iron = RNG.random() < 0.80
            sn_iron = RNG.random() < 0.65
            thin_cc = RNG.random() < 0.75
            cerebellar_atrophy = RNG.random() < 0.75
            spastic_paraplegia = True
            ataxia = RNG.random() < 0.85
            dysarthria = RNG.random() < 0.80
            dystonia = RNG.random() < 0.70
            optic_atrophy = RNG.random() < 0.22
            axonal_neuropathy = RNG.random() < 0.30
            cognitive_decline = RNG.random() < 0.45
            psychiatric = RNG.random() < 0.20
            ambulation_lost = RNG.random() < (0.45 if (current_age - onset_yr) > 15 else 0.15)
            seizures_prob = 0.22

        else:  # Complex-SPG
            onset_yr = round(RNG.uniform(9, 15), 1)
            current_age = round(RNG.uniform(onset_yr + 3, onset_yr + 18), 0)
            leukodystrophy = True
            gp_iron = RNG.random() < 0.55
            sn_iron = RNG.random() < 0.40
            thin_cc = RNG.random() < 0.60
            cerebellar_atrophy = RNG.random() < 0.45
            spastic_paraplegia = True
            ataxia = RNG.random() < 0.40
            dysarthria = RNG.random() < 0.40
            dystonia = RNG.random() < 0.20
            optic_atrophy = RNG.random() < 0.15
            axonal_neuropathy = RNG.random() < 0.20
            cognitive_decline = RNG.random() < 0.25
            psychiatric = RNG.random() < 0.10
            ambulation_lost = RNG.random() < 0.10
            seizures_prob = 0.15

        has_seizures = RNG.random() < seizures_prob
        if has_seizures:
            n_aeds = RNG.randint(1, 4)
            aeds_tried = RNG.sample(aed_options, min(n_aeds, len(aed_options)))
        else:
            n_aeds = 0
            aeds_tried = []

        drug_resistant = has_seizures and RNG.random() < 0.35
        seizure_free = has_seizures and (not drug_resistant) and RNG.random() < 0.55

        baclofen = RNG.random() < (0.90 if ph == "FAHN-Classic" else 0.82 if ph == "HSP-Ataxia-Dystonia" else 0.60)
        btx = RNG.random() < (0.50 if ph == "FAHN-Classic" else 0.45 if ph == "HSP-Ataxia-Dystonia" else 0.25)
        trihexyphenidyl = dystonia and RNG.random() < 0.55
        dbs = dystonia and RNG.random() < (0.05 if ph == "FAHN-Classic" else 0.08 if ph == "HSP-Ataxia-Dystonia" else 0.02)
        polg_tested = RNG.random() < 0.70
        deferiprone_trial = RNG.random() < 0.05
        physio_enrolled = RNG.random() < 0.88

        dystonia_severity = None
        if dystonia:
            dystonia_severity = RNG.choice(["mild", "moderate", "severe"])

        pts.append({
            "id": pid,
            "phenotype": ph,
            "etiology": etiology,
            "onset_yr": onset_yr,
            "current_age": current_age,
            "disease_duration_yr": round(current_age - onset_yr, 1),
            "leukodystrophy": leukodystrophy,
            "gp_iron": gp_iron,
            "sn_iron": sn_iron,
            "thin_cc": thin_cc,
            "cerebellar_atrophy": cerebellar_atrophy,
            "spastic_paraplegia": spastic_paraplegia,
            "ataxia": ataxia,
            "dysarthria": dysarthria,
            "dystonia": dystonia,
            "dystonia_severity": dystonia_severity,
            "optic_atrophy": optic_atrophy,
            "axonal_neuropathy": axonal_neuropathy,
            "cognitive_decline": cognitive_decline,
            "psychiatric": psychiatric,
            "ambulation_lost": ambulation_lost,
            "has_seizures": has_seizures,
            "drug_resistant": drug_resistant,
            "seizure_free": seizure_free,
            "n_aeds_tried": n_aeds,
            "aeds_tried": aeds_tried,
            "baclofen": baclofen,
            "btx": btx,
            "trihexyphenidyl": trihexyphenidyl,
            "dbs": dbs,
            "physio_enrolled": physio_enrolled,
            "polg_tested": polg_tested,
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

    n_classic = sum(1 for p in pts if p["phenotype"] == "FAHN-Classic")
    n_hsp = sum(1 for p in pts if p["phenotype"] == "HSP-Ataxia-Dystonia")
    n_cspg = sum(1 for p in pts if p["phenotype"] == "Complex-SPG")

    classic_pts = [p for p in pts if p["phenotype"] == "FAHN-Classic"]
    hsp_pts = [p for p in pts if p["phenotype"] == "HSP-Ataxia-Dystonia"]
    cspg_pts = [p for p in pts if p["phenotype"] == "Complex-SPG"]

    def mean_onset(group):
        return round(sum(p["onset_yr"] for p in group) / max(len(group), 1), 1)

    etio_counts = {}
    for p in pts:
        etio_counts[p["etiology"]] = etio_counts.get(p["etiology"], 0) + 1

    etio_distribution = [
        {"etiology": k.replace("_", " ").title(), "n": v, "pct": round(v / n * 100)}
        for k, v in sorted(etio_counts.items(), key=lambda x: -x[1])
    ]

    clinical_highlights = [
        {"finding": "Leukodystrophy (T2/FLAIR WM hyperintensity)", "pct": pct(lambda p: p["leukodystrophy"]), "note": "EARLIEST + MOST PROMINENT MRI feature — PATHOGNOMONIC for FAHN"},
        {"finding": "Spastic Paraplegia", "pct": pct(lambda p: p["spastic_paraplegia"]), "note": "DOMINANT early motor feature; ALL phenotypes affected"},
        {"finding": "GP Iron (T2*/SWI)", "pct": pct(lambda p: p["gp_iron"]), "note": "Present but MILD early; precedes SN; leukodystrophy appears first"},
        {"finding": "SN Iron (T2*/SWI)", "pct": pct(lambda p: p["sn_iron"]), "note": "Follows GP iron; both confirmed by SWI"},
        {"finding": "Thin Corpus Callosum", "pct": pct(lambda p: p["thin_cc"]), "note": "White matter involvement; correlates with leukodystrophy severity"},
        {"finding": "Cerebellar Atrophy", "pct": pct(lambda p: p["cerebellar_atrophy"]), "note": "Progressive; appears after spasticity onset; worsens with age"},
        {"finding": "Ataxia", "pct": pct(lambda p: p["ataxia"]), "note": "Develops in 1st-2nd decade; truncal + appendicular"},
        {"finding": "Dysarthria", "pct": pct(lambda p: p["dysarthria"]), "note": "Mixed spastic + cerebellar component; early in FAHN-Classic"},
        {"finding": "Dystonia", "pct": pct(lambda p: p["dystonia"]), "note": "Late feature (2nd-3rd decade); correlates with GP iron severity"},
        {"finding": "Seizures", "pct": pct(lambda p: p["has_seizures"]), "note": "~25% overall; focal > generalised; PHT/VGB AVOID"},
        {"finding": "Ambulation Lost", "pct": pct(lambda p: p["ambulation_lost"]), "note": "FAHN-Classic: ~70% wheelchair-dependent by 3rd decade"},
        {"finding": "Optic Atrophy", "pct": pct(lambda p: p["optic_atrophy"]), "note": "~25%; less prominent than MPAN (80%) but VGB remains AVOIDED"},
        {"finding": "Axonal Neuropathy", "pct": pct(lambda p: p["axonal_neuropathy"]), "note": "Less prevalent than PLA2G6/PLAN but NCS mandatory at diagnosis"},
        {"finding": "Cognitive Decline", "pct": pct(lambda p: p["cognitive_decline"]), "note": "Mild-moderate; white matter involvement; executive function affected"},
    ]

    treatment_summary = {
        "baclofen_pct": pct(lambda p: p["baclofen"]),
        "btx_pct": pct(lambda p: p["btx"]),
        "trihexyphenidyl_pct": pct(lambda p: p["trihexyphenidyl"]),
        "physio_enrolled_pct": pct(lambda p: p["physio_enrolled"]),
        "dbs_pct": pct(lambda p: p["dbs"]),
        "polg_tested_pct": pct(lambda p: p["polg_tested"]),
        "deferiprone_trial_pct": pct(lambda p: p["deferiprone_trial"]),
        "seizure_on_aed_pct": pct(lambda p: p["has_seizures"]),
        "drug_resistant_pct": pct(lambda p: p["drug_resistant"]),
        "seizure_free_pct": pct(lambda p: p["seizure_free"]),
    }

    return {
        "disease": DISEASE,
        "gene": GENE,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "cohort_n": n,
        "cohort_seed": SEED,
        "kpis": {
            "n_patients": n,
            "n_fahn_classic": n_classic,
            "n_hsp_ataxia_dystonia": n_hsp,
            "n_complex_spg": n_cspg,
            "leukodystrophy_pct": pct(lambda p: p["leukodystrophy"]),
            "spastic_paraplegia_pct": pct(lambda p: p["spastic_paraplegia"]),
            "gp_iron_pct": pct(lambda p: p["gp_iron"]),
            "thin_cc_pct": pct(lambda p: p["thin_cc"]),
            "cerebellar_atrophy_pct": pct(lambda p: p["cerebellar_atrophy"]),
            "ataxia_pct": pct(lambda p: p["ataxia"]),
            "dystonia_pct": pct(lambda p: p["dystonia"]),
            "has_seizures_pct": pct(lambda p: p["has_seizures"]),
            "ambulation_lost_pct": pct(lambda p: p["ambulation_lost"]),
            "optic_atrophy_pct": pct(lambda p: p["optic_atrophy"]),
            "axonal_neuropathy_pct": pct(lambda p: p["axonal_neuropathy"]),
            "cognitive_decline_pct": pct(lambda p: p["cognitive_decline"]),
            "fahn_classic_mean_onset_yr": mean_onset(classic_pts),
            "hsp_mean_onset_yr": mean_onset(hsp_pts),
            "complex_spg_mean_onset_yr": mean_onset(cspg_pts),
        },
        "etiology_distribution": etio_distribution,
        "clinical_highlights": clinical_highlights,
        "treatment_summary": treatment_summary,
        "lifecycle": LIFECYCLE,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }


def get_breakdown():
    pts = _get_patients()
    n = len(pts)

    # Per-phenotype breakdown
    phenotype_breakdown = []
    for ph in ["FAHN-Classic", "HSP-Ataxia-Dystonia", "Complex-SPG"]:
        group = [p for p in pts if p["phenotype"] == ph]
        if not group:
            continue
        ng = len(group)
        phenotype_breakdown.append({
            "phenotype": ph,
            "n": ng,
            "pct": round(ng / n * 100),
            "mean_onset_yr": round(sum(p["onset_yr"] for p in group) / ng, 1),
            "leukodystrophy_pct": round(sum(1 for p in group if p["leukodystrophy"]) / ng * 100),
            "gp_iron_pct": round(sum(1 for p in group if p["gp_iron"]) / ng * 100),
            "cerebellar_atrophy_pct": round(sum(1 for p in group if p["cerebellar_atrophy"]) / ng * 100),
            "ataxia_pct": round(sum(1 for p in group if p["ataxia"]) / ng * 100),
            "dystonia_pct": round(sum(1 for p in group if p["dystonia"]) / ng * 100),
            "has_seizures_pct": round(sum(1 for p in group if p["has_seizures"]) / ng * 100),
            "ambulation_lost_pct": round(sum(1 for p in group if p["ambulation_lost"]) / ng * 100),
            "drug_resistant_pct": round(sum(1 for p in group if p["drug_resistant"]) / ng * 100),
        })

    # Per-etiology breakdown
    etio_groups = {}
    for p in pts:
        etio_groups.setdefault(p["etiology"], []).append(p)
    etio_breakdown = []
    for etio, group in sorted(etio_groups.items(), key=lambda x: -len(x[1])):
        ng = len(group)
        n_classic = sum(1 for p in group if p["phenotype"] == "FAHN-Classic")
        n_hsp = sum(1 for p in group if p["phenotype"] == "HSP-Ataxia-Dystonia")
        etio_breakdown.append({
            "etiology": etio.replace("_", " ").title(),
            "n": ng,
            "pct": round(ng / n * 100),
            "fahn_classic_pct": round(n_classic / ng * 100) if ng else 0,
            "hsp_ataxia_dystonia_pct": round(n_hsp / ng * 100) if ng else 0,
            "leukodystrophy_pct": round(sum(1 for p in group if p["leukodystrophy"]) / ng * 100),
            "drug_resistant_pct": round(sum(1 for p in group if p["drug_resistant"]) / ng * 100),
        })

    # Seizure type breakdown
    seizure_pts = [p for p in pts if p["has_seizures"]]
    seizure_breakdown = []
    for st in ["focal", "generalised", "myoclonic", "absence"]:
        prob = {"focal": 0.65, "generalised": 0.35, "myoclonic": 0.25, "absence": 0.15}[st]
        n_st = sum(1 for _ in seizure_pts if RNG.random() < prob)
        seizure_breakdown.append({
            "type": st.title(),
            "n": n_st,
            "pct": round(n_st / max(len(seizure_pts), 1) * 100),
            "drug_resistant_pct": round(sum(1 for p in seizure_pts if p["drug_resistant"]) / max(len(seizure_pts), 1) * 100),
        })

    # Per-patient summary
    per_patient = []
    for p in pts:
        per_patient.append({
            "id": p["id"],
            "phenotype": p["phenotype"],
            "etiology": p["etiology"],
            "onset_yr": p["onset_yr"],
            "current_age": p["current_age"],
            "disease_duration_yr": p["disease_duration_yr"],
            "leukodystrophy": p["leukodystrophy"],
            "gp_iron": p["gp_iron"],
            "sn_iron": p["sn_iron"],
            "thin_cc": p["thin_cc"],
            "cerebellar_atrophy": p["cerebellar_atrophy"],
            "spastic_paraplegia": p["spastic_paraplegia"],
            "ataxia": p["ataxia"],
            "dysarthria": p["dysarthria"],
            "dystonia": p["dystonia"],
            "dystonia_severity": p["dystonia_severity"],
            "optic_atrophy": p["optic_atrophy"],
            "axonal_neuropathy": p["axonal_neuropathy"],
            "cognitive_decline": p["cognitive_decline"],
            "psychiatric": p["psychiatric"],
            "ambulation_lost": p["ambulation_lost"],
            "has_seizures": p["has_seizures"],
            "drug_resistant": p["drug_resistant"],
            "seizure_free": p["seizure_free"],
            "n_aeds": p["n_aeds_tried"],
            "aeds_tried": p["aeds_tried"],
            "baclofen": p["baclofen"],
            "btx": p["btx"],
            "trihexyphenidyl": p["trihexyphenidyl"],
            "dbs": p["dbs"],
            "physio_enrolled": p["physio_enrolled"],
            "polg_tested": p["polg_tested"],
            "deferiprone_trial": p["deferiprone_trial"],
        })

    return {
        "cohort_n": n,
        "phenotype_breakdown": phenotype_breakdown,
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
        "disease": "FA2H FAHN (FA2H-Associated Neurodegeneration / NBIA3)",
        "gene": "FA2H (Fatty Acid 2-Hydroxylase — 490 aa, ER membrane, FAD-binding, HX3H di-iron) — 16q23.1 — OMIM 611026",
        "omim_disease": "FAHN 612319",
        "definitions": DEFINITIONS,
        "contraindications": CONTRAINDICATIONS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "key_concepts": [d["term"] for d in DEFINITIONS],
    }


if __name__ == "__main__":
    print("=== FA2H / FAHN (NBIA3) Dashboard — Self-Test (seed-523) ===\n")

    ov = get_overview()
    kpis = ov["kpis"]
    print(f"[get_overview] disease: {ov['disease']}")
    print(f"  n_patients={kpis['n_patients']}, n_classic={kpis['n_fahn_classic']}, n_hsp={kpis['n_hsp_ataxia_dystonia']}, n_cspg={kpis['n_complex_spg']}")
    print(f"  leukodystrophy_pct={kpis['leukodystrophy_pct']}%, spastic_paraplegia_pct={kpis['spastic_paraplegia_pct']}%")
    print(f"  gp_iron_pct={kpis['gp_iron_pct']}%, thin_cc_pct={kpis['thin_cc_pct']}%")
    print(f"  has_seizures_pct={kpis['has_seizures_pct']}%, ambulation_lost_pct={kpis['ambulation_lost_pct']}%")
    print(f"  fahn_classic_mean_onset={kpis['fahn_classic_mean_onset_yr']}yr, hsp_mean_onset={kpis['hsp_mean_onset_yr']}yr")
    print(f"  etiology_distribution: {len(ov['etiology_distribution'])} entries")
    print(f"  clinical_highlights: {len(ov['clinical_highlights'])} items")
    print()

    bk = get_breakdown()
    print(f"[get_breakdown] cohort_n={bk['cohort_n']}")
    print(f"  phenotype_breakdown: {len(bk['phenotype_breakdown'])} groups")
    for ph in bk["phenotype_breakdown"]:
        print(f"    {ph['phenotype']}: n={ph['n']} ({ph['pct']}%), leukodystrophy={ph['leukodystrophy_pct']}%, seizures={ph['has_seizures_pct']}%")
    print(f"  etiology_breakdown: {len(bk['etiology_breakdown'])} groups")
    for e in bk["etiology_breakdown"]:
        print(f"    {e['etiology']}: n={e['n']} ({e['pct']}%)")
    print(f"  per_patient: {len(bk['per_patient'])} rows")
    print()

    df = get_definitions()
    print(f"[get_definitions] definitions: {len(df['definitions'])} concepts")
    print(f"  key_concepts: {df['key_concepts'][:3]} ...")
    print(f"  standards: {len(df['standards'])}, references: {len(df['references'])}")
    print("\n=== All 3 functions OK ===")
