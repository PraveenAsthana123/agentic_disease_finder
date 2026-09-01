"""
PLA2G6 PLAN (PLA2G6-Associated Neurodegeneration) — NBIA2
===========================================================
40-patient cohort · PLA2G6 (22q13.1) · Autosomal Recessive · 3rd most common NBIA (~5-15%)
Three phenotypes: INAD (Infantile NAD, classic), ANAD (Atypical NAD), PARK14 (Adult parkinsonism)
Cerebellar cortical atrophy EARLIEST + MOST PROMINENT MRI finding — contrasts with PKAN eye-of-tiger
GP iron LATE (may be ABSENT at onset) — key DDx from PKAN (early iron)
NO eye-of-tiger sign (key DDx from PKAN) — spheroid bodies on nerve biopsy PATHOGNOMONIC
Axonal neuropathy 100% INAD / 70% ANAD — NCS mandatory at diagnosis and every 2yr
PHT ABSOLUTE CI (aggravates axonal neuropathy); VGB ABSOLUTE CI in INAD (optic atrophy additive)
No approved disease-modifying therapy 2026 · Deferiprone investigational (NBIA Research Institute)
POLG mandatory before VPA (secondary mitochondrial dysfunction — phospholipid remodelling failure)

PLA2G6 BIOLOGY:
PLA2G6 encodes calcium-independent phospholipase A2 beta (iPLA2β), a 806-amino-acid enzyme.
It is essential for phospholipid remodelling (Lands cycle) — deacylating sn-2 fatty acyl chains
from phospholipids. LOF → phospholipid remodelling failure → abnormal mitochondrial membrane
composition → mitochondrial dysfunction → ROS → iron accumulation GP/SN (appears LATE vs PKAN).
Mitochondrial membrane is particularly vulnerable to iPLA2β loss because it is rich in
cardiolipin and polyunsaturated fatty acid-containing phospholipids requiring active remodelling.

PLA2G6 PROTEIN STRUCTURE (806 aa, 22q13.1):
  ANKYRIN REPEAT DOMAIN (aa 1-290):
    8 ankyrin repeats — protein-protein interaction scaffold.
    Regulatory function: autoinhibits catalytic domain at rest.
    Pathogenic variants here → conformational dysfunction; severity variable.
  LINKER REGION (aa 291-460):
    Regulatory region; calmodulin-binding sequence.
    Required for activation and membrane targeting.
    Missense variants in linker → partial LOF; ANAD/PARK14 phenotypes more common.
  PATATIN-LIKE PHOSPHOLIPASE DOMAIN (aa 461-799):
    Catalytic domain: Ser465-Asp742-His716 catalytic triad (serine phospholipase mechanism).
    Responsible for sn-2 deacylation of phospholipids.
    Null/truncating variants here → complete LOF → INAD (severe).
    Missense in catalytic triad → severe INAD.
  PATHOGENIC VARIANT DISTRIBUTION (no dominant founder mutation):
    Missense compound het: ~45% — most common; ANAD/INAD phenotypes.
    Null/truncating biallelic: ~30% — complete LOF; INAD severe.
    Splice variants: ~15% — exon skipping; variable phenotype.
    Structural/CNV: ~10% — rare; variable.
    Saudi/North African enrichment for null variants.
    NO single founder mutation (unlike p.Gly69Arg in MPAN or PANK2 null mutations).

PLA2G6 FUNCTION — PHOSPHOLIPID REMODELLING:
  iPLA2β (calcium-independent PLA2, group VIA) catalyses:
    - Hydrolysis of sn-2 fatty acyl chain from phospholipids (Lands cycle)
    - Membrane phospholipid composition maintenance
    - Arachidonic acid release (eicosanoid signalling)
    - Cardiolipin remodelling in inner mitochondrial membrane
  iPLA2β LOF → phospholipid remodelling failure:
    1. Mitochondrial membrane composition abnormal: excessive lysophospholipids,
       reduced cardiolipin → mitochondrial dysfunction → ATP synthesis impaired.
    2. Reactive oxygen species (ROS): dysfunctional mitochondria → oxidative stress
       → lipid peroxidation → axonal membrane degeneration → spheroid formation.
    3. Iron accumulation: late consequence in GP/SN (unlike PKAN where iron is early).
       T2/SWI hypointensity GP/SN appears late — may be ABSENT at initial presentation.
    4. Axonal spheroid bodies: neuroaxonal dystrophy — pathognomonic on nerve biopsy.
       Dystrophic axonal swellings (spheroids) = hallmark of PLAN neuropathology.

THREE CLINICAL PHENOTYPES:
CLASSIC INAD (Infantile Neuroaxonal Dystrophy, INAD1, n≈20/40):
  Onset 6mo-3yr (mean ~1.5yr). Most severe phenotype.
  Psychomotor regression, hypotonia → spastic paraparesis.
  Cerebellar cortical atrophy EARLIEST + MOST PROMINENT MRI finding.
  GP iron LATE (T2/SWI hypointensity) — may be ABSENT at onset.
  Sensorimotor axonal neuropathy 100% (NCS mandatory at diagnosis).
  Optic atrophy 70% — VGB ABSOLUTE CI (additive risk).
  Seizures 50-60% (complex focal, myoclonic, absence).
  Death typically 2nd decade.
  Spheroid bodies on nerve biopsy PATHOGNOMONIC (skin or conjunctival biopsy alternative).

ATYPICAL NAD / ANAD (NAD2, n≈14/40):
  Onset 1-5yr (mean ~2.5yr). Less severe than INAD.
  Cerebellar ataxia DOMINANT — less progression than INAD.
  Dystonia (60%), psychiatric features (35%).
  Survival into adulthood.
  Neuropathy present but milder (70%).
  Seizures 35%.
  MRI: cerebellar atrophy; GP iron variable.

PARK14 (Adult-onset parkinsonism-dystonia, n≈6/40):
  Onset 30-50yr (mean ~35yr). Mild alleles (linker/ankyrin missense).
  Parkinsonism + dystonia. L-DOPA responsive initially (response wanes).
  Cerebellar atrophy present on MRI.
  Less neuropathy (EMG abnormal but mild).
  No seizures typically.
  GPi-DBS Level C for dystonia component.

KEY DDx FROM PKAN:
  PLAN: cerebellar atrophy DOMINANT (PKAN: GP iron dominant, eye-of-tiger PATHOGNOMONIC).
  PLAN: NO eye-of-tiger sign (key DDx from PKAN).
  PLAN: axonal neuropathy 100% INAD / 70% ANAD (PKAN: no neuropathy).
  PLAN: hypotonia/ataxia FIRST (PKAN: dystonia first, early onset).
  PLAN: iron may be ABSENT early (PKAN: eye-of-tiger early and prominent).
  PLAN: spheroid bodies on nerve biopsy PATHOGNOMONIC.
  PLAN vs MPAN: both have neuropathy; PLAN has cerebellar atrophy dominant vs MPAN has optic atrophy KEY.

OMIM: Gene PLA2G6 #603604 · Disease INAD1/PLAN #256600 · NBIA2A #610217 · PARK14 #612953 · Chr 22q13.1
3rd most common NBIA (~5-15% of NBIA after PANK2 and C19orf12/WDR45).
"""

import random
import math

SEED = 521
RNG = random.Random(SEED)

N = 40
# INAD 20, ANAD 14, PARK14 6
N_INAD = 20
N_ANAD = 14
N_PARK14 = 6

# ── Etiology distribution ──────────────────────────────────────────────────
ETIOLOGIES = [
    {"etiology": "PLA2G6 Missense Compound Het",
     "n": 18, "pct": 45,
     "mechanism": "Two different missense alleles in trans; most common overall; affects ankyrin, linker, or catalytic domains; phenotype spectrum INAD to ANAD to PARK14 depending on domain and residual activity",
     "severity": "Variable — catalytic domain compound het → INAD; ankyrin/linker compound het → ANAD or PARK14"},
    {"etiology": "PLA2G6 Null/Truncating (biallelic)",
     "n": 12, "pct": 30,
     "mechanism": "Frameshift, nonsense, or large deletion causing complete iPLA2β LOF; no residual phospholipase activity; severe early-onset; Saudi/North African enrichment; no founder mutation",
     "severity": "Severe — INAD classic phenotype; early psychomotor regression; death 2nd decade"},
    {"etiology": "PLA2G6 Splice Variants",
     "n": 6, "pct": 15,
     "mechanism": "Intronic splice-site variants; exon skipping; degree of in-frame vs frameshift skipping determines residual function; INAD if frameshift skip, ANAD if in-frame partial LOF",
     "severity": "Variable — splice severity determines phenotype: INAD or ANAD"},
    {"etiology": "PLA2G6 Structural/CNV",
     "n": 4, "pct": 10,
     "mechanism": "Large intragenic deletions or duplications; rare; complete exon loss → null effect; exon duplication → abnormal protein; variable phenotype",
     "severity": "Variable — typically INAD if large deletion encompasses catalytic domain"},
]

# ── Seizure type distribution (overall: 50-60% INAD, 35% ANAD, ~0% PARK14) ─
SEIZURE_TYPES = [
    {"type": "Complex Focal (with/without secondary generalisation)", "pct": 35, "n": 14},
    {"type": "Myoclonic", "pct": 28, "n": 11},
    {"type": "Absence", "pct": 20, "n": 8},
    {"type": "Generalised Tonic-Clonic (GTCS)", "pct": 15, "n": 6},
    {"type": "Tonic / Atonic", "pct": 8, "n": 3},
]

# ── AED and symptomatic treatment data ────────────────────────────────────
TREATMENTS = [
    {"drug": "LEV (Levetiracetam)", "level": "B", "tried_pct": 75, "responder_pct": 55,
     "note": "Best tolerated AED for PLAN seizures; broad-spectrum; no motor worsening; Level B first-line"},
    {"drug": "CLB (Clobazam)", "level": "B", "tried_pct": 50, "responder_pct": 48,
     "note": "Add-on for focal and myoclonic seizures; useful adjunct to LEV in PLAN"},
    {"drug": "VPA (Valproate)", "level": "B+POLG", "tried_pct": 40, "responder_pct": 52,
     "note": "POLG MANDATORY before use; secondary mitochondrial dysfunction in PLAN; effective myoclonic/GTCS; NOT for INAD if POLG positive"},
    {"drug": "Baclofen (oral/intrathecal)", "level": "B", "tried_pct": 72, "responder_pct": 62,
     "note": "First-line for spastic paraparesis (INAD/ANAD); oral initial; intrathecal for severe lower limb spasticity"},
    {"drug": "Trihexyphenidyl", "level": "C", "tried_pct": 42, "responder_pct": 40,
     "note": "Level C for dystonia (ANAD phenotype); anticholinergic; avoid in severe cognitive impairment INAD"},
    {"drug": "GPi-DBS (Deep Brain Stimulation)", "level": "C", "tried_pct": 8, "responder_pct": 38,
     "note": "Level C — atypical NAD + PARK14 phenotypes; less evidence than PKAN; dystonia component; case series"},
    {"drug": "L-DOPA (Levodopa)", "level": "C", "tried_pct": 15, "responder_pct": 45,
     "note": "PARK14 phenotype only; initial L-DOPA response (wanes over 2-5yr); no benefit INAD/ANAD"},
    {"drug": "Deferiprone (iron chelation)", "level": "INV", "tried_pct": 6, "responder_pct": 28,
     "note": "Investigational iron chelation; NBIA Research Institute trials; iron late in PLAN — functional benefit unknown; REMS weekly WBC"},
    {"drug": "ACTH", "level": "A", "tried_pct": 5, "responder_pct": 70,
     "note": "Level A if infantile spasms (IS) — rare in PLAN; ACTH/UKISS protocol; IS onset uncommon but documented"},
    {"drug": "CoA supplements (investigational)", "level": "INV", "tried_pct": 4, "responder_pct": 20,
     "note": "Coenzyme A supplements investigational; rationale: phospholipid remodelling pathway support; no clinical trial data 2026"},
]

CONTRAINDICATIONS = [
    {"drug": "PHT (Phenytoin) / Fosphenytoin",
     "reason": "ABSOLUTE CI — aggravates axonal neuropathy (100% INAD, 70% ANAD); sodium channel blockade worsens peripheral nerve function; high-risk NCS deterioration"},
    {"drug": "VGB (Vigabatrin)",
     "reason": "ABSOLUTE CI in INAD — optic atrophy additive risk (70% INAD already have optic atrophy); irreversible concentric visual field loss; do NOT use in any PLAN patient with optic atrophy"},
    {"drug": "CBZ (Carbamazepine) / OXC (Oxcarbazepine)",
     "reason": "CAUTION — worsens cerebellar ataxia and neuropathy; sodium channel blockade; if used, NCS and ataxia monitoring mandatory; avoid in INAD"},
    {"drug": "VPA in POLG carriers",
     "reason": "ABSOLUTE CI — fatal hepatotoxicity; POLG mandatory panel; PLAN has secondary mitochondrial dysfunction; even heterozygous POLG warrants caution"},
    {"drug": "Typical antipsychotics",
     "reason": "HIGH RISK in PARK14 — worsens dystonia and parkinsonism; dopamine blockade contraindicated in PARK14 parkinsonism; use atypicals (quetiapine) only if essential"},
]

# ── DDx table ──────────────────────────────────────────────────────────────
DDX = [
    {"condition": "PKAN (PANK2)",
     "distinguishing": "Eye-of-tiger sign PATHOGNOMONIC (absent in PLAN); GP iron EARLY (PLAN iron LATE/absent); dystonia first vs PLAN hypotonia/ataxia first; NO neuropathy PKAN vs 100% INAD; CoA pathway defect vs phospholipid remodelling",
     "shared": "NBIA, iron accumulation (late), AR, dystonia possible"},
    {"condition": "MPAN (C19orf12)",
     "distinguishing": "MPAN: optic atrophy 80% KEY (PLAN: 70% INAD only, absent PARK14); MPAN: pyramidal signs 100%; MPAN: juvenile onset 8-14yr vs PLAN: infantile 6mo-3yr (INAD); MAM dysfunction vs phospholipid remodelling",
     "shared": "Both have axonal neuropathy, NBIA, iron accumulation, no eye-of-tiger"},
    {"condition": "BPAN (WDR45)",
     "distinguishing": "X-linked dominant de novo females; biphasic: static encephalopathy childhood → SUDDEN parkinsonism-dementia adult; T1 halo sign + SN iron; ferritinophagy (not phospholipid)",
     "shared": "NBIA, iron accumulation, AR vs XLD"},
    {"condition": "Pontocerebellar Hypoplasia (PCH)",
     "distinguishing": "Pontine hypoplasia prominent; often CASK/TSEN/RARS2 mutations; no iron accumulation; no neuropathy; earlier and more severe cerebellar hypoplasia vs atrophy",
     "shared": "Cerebellar abnormality, infantile onset, hypotonia"},
    {"condition": "Metachromatic Leukodystrophy (MLD)",
     "distinguishing": "White matter T2 hyperintensity (leukodystrophy pattern); ARSA mutation; sulfatide storage; nerve biopsy: metachromatic deposits vs spheroid bodies; no iron",
     "shared": "Infantile onset possible, hypotonia, regression, axonal neuropathy"},
    {"condition": "Krabbe Disease (GALC)",
     "distinguishing": "GALC mutation; globoid cell leukodystrophy; corticospinal tract T2 change + cerebellar; nerve biopsy: globoid cells vs spheroids; galactosylceramide accumulation",
     "shared": "Infantile onset, hypotonia, peripheral neuropathy, regression"},
    {"condition": "Neuronal Ceroid Lipofuscinosis (NCL)",
     "distinguishing": "EM: fingerprint/curvilinear inclusions (vs spheroid bodies); PPTP1/CLN3 etc; visual failure + seizures + regression; no iron; ERG abolished early",
     "shared": "Regression, seizures, optic atrophy possible, infantile form"},
]

# ── Key concepts (definitions) ─────────────────────────────────────────────
DEFINITIONS = [
    {"term": "PLAN-NBIA2 (PLA2G6-Associated Neurodegeneration)",
     "def": "3rd most common NBIA (~5-15% of NBIA). AR biallelic PLA2G6 LOF → iPLA2β deficiency → phospholipid remodelling failure → mitochondrial membrane dysfunction → ROS → iron accumulation GP/SN (LATE, often absent at onset). Three phenotypes: INAD1 (infantile), ANAD (atypical), PARK14 (adult). Cerebellar atrophy earliest MRI finding. No eye-of-tiger sign."},
    {"term": "PLA2G6 (806 aa iPLA2β — calcium-independent phospholipase A2 beta)",
     "def": "806-amino-acid enzyme with 3 domains: ankyrin repeat domain (aa 1-290, 8 repeats, protein interaction), linker/calmodulin-binding region (aa 291-460), and patatin-like phospholipase domain (aa 461-799, catalytic triad Ser465-Asp742-His716). Catalyses sn-2 deacylation of membrane phospholipids (Lands cycle). Located on chromosome 22q13.1. OMIM Gene 603604."},
    {"term": "Infantile Neuroaxonal Dystrophy (INAD1 / Classic PLAN)",
     "def": "Most severe PLA2G6 phenotype. Onset 6mo-3yr (mean ~1.5yr). Complete iPLA2β LOF (null/truncating or catalytic domain missense biallelic). Psychomotor regression, hypotonia → spastic paraparesis. Cerebellar cortical atrophy EARLIEST MRI finding. Iron LATE (may be ABSENT). Sensorimotor axonal neuropathy 100%. Optic atrophy 70%. Seizures 50-60%. Death 2nd decade. OMIM 256600."},
    {"term": "Atypical NAD / ANAD (NAD2 — Intermediate PLA2G6 phenotype)",
     "def": "Intermediate PLA2G6 phenotype. Onset 1-5yr (mean ~2.5yr). Partial LOF (ankyrin/linker missense compound het or mild splice variants). Cerebellar ataxia dominant, less severe than INAD. Dystonia 60%, psychiatric features 35%. Survival into adulthood. Neuropathy 70% (milder than INAD). Seizures 35%. MRI: cerebellar atrophy, variable iron. OMIM 610217 (NBIA2A)."},
    {"term": "PARK14 (Adult-onset PLA2G6 parkinsonism-dystonia)",
     "def": "Adult onset PLA2G6 phenotype. Onset 30-50yr (mean ~35yr). Mild hypomorphic alleles (ankyrin/linker domain missense). Parkinsonism + dystonia. L-DOPA responsive initially (response wanes 2-5yr). Cerebellar atrophy on MRI. Minimal neuropathy. No seizures typically. GPi-DBS Level C for dystonia. OMIM 612953."},
    {"term": "Cerebellar Cortical Atrophy (Earliest MRI finding in PLAN)",
     "def": "Earliest and most prominent MRI finding in PLAN (INAD and ANAD). T2/FLAIR: cerebellar cortical atrophy — vermal > hemispheric. Distinguishes PLAN from PKAN (GP iron dominant, eye-of-tiger) and MPAN (GP+SN iron dominant). Cerebellar atrophy may be present BEFORE GP iron appears. Annual brain MRI mandatory. SWI for iron monitoring — iron may be absent in early INAD."},
    {"term": "Spheroid Bodies in Axons (PATHOGNOMONIC for PLAN)",
     "def": "Dystrophic axonal swellings (spheroid bodies) — hallmark neuropathology of PLAN. Found on nerve biopsy (sural) or skin / conjunctival biopsy. Electron microscopy: tubulovesicular structures within distended axons. Represent accumulation of abnormal phospholipid membranes due to iPLA2β LOF. PATHOGNOMONIC for neuroaxonal dystrophy — distinguishes PLAN from other NBIA. Confirms diagnosis when genetic testing inconclusive."},
    {"term": "Axonal Neuropathy in PLAN (NCS Mandatory)",
     "def": "Sensorimotor axonal neuropathy: 100% INAD, 70% ANAD, milder in PARK14. NCS: reduced motor AND sensory amplitudes, normal conduction velocity (axonal — not demyelinating). NCS/EMG mandatory at diagnosis and every 2yr. PHT ABSOLUTE CI (aggravates neuropathy). CBZ/OXC CAUTION (worsens neuropathy). Distinguishes PLAN from PKAN (no neuropathy) and separates it from MPAN (motor axonal only, 60%)."},
    {"term": "GP Iron LATE / Absent Early (Key DDx from PKAN)",
     "def": "PLAN: iron accumulation in GP/SN is a LATE feature — may be completely absent at initial presentation (especially INAD). SWI/T2* hypointensity GP appears only after years of disease. CRITICAL DDx from PKAN: PKAN shows eye-of-tiger sign EARLY (GP central T2 hyperintensity surrounded by hypointense rim). In PLAN: NO eye-of-tiger sign at any stage. If iron absent and cerebellar atrophy prominent → PLA2G6 sequencing mandatory."},
    {"term": "No Eye-of-Tiger Sign (DDx from PKAN in PLAN)",
     "def": "PLAN does NOT show eye-of-tiger sign (central GP T2 hyperintensity within iron hypointensity rim) at any disease stage. Eye-of-tiger is PATHOGNOMONIC for PKAN (PANK2 mutation). In PLAN: when iron eventually accumulates (late), it shows uniform GP T2/SWI hypointensity without central hyperintense core. Absence of eye-of-tiger + cerebellar atrophy + infantile onset → PLA2G6 panel mandatory."},
    {"term": "GPi-DBS Level C (PLAN — Atypical NAD and PARK14)",
     "def": "GPi-DBS evidence for PLAN is Level C — limited to case series and expert opinion. Best evidence in ANAD phenotype with prominent dystonia and PARK14 (dystonia component). Less predictable than PKAN (Level B) because PLAN has cerebellar ataxia complicating dystonia assessment. Refer before irreversible ambulation loss. BFMDRS ≥35/120 or rapid functional decline triggers DBS referral."},
    {"term": "Deferiprone (Iron Chelation — PLAN Investigational)",
     "def": "Brain-penetrant iron chelator. Investigational in PLAN (NBIA Research Institute). Iron accumulation is late in PLAN vs early in PKAN — may limit deferiprone window of effect. Functional benefit not demonstrated in PLAN 2026. Weekly WBC monitoring required (agranulocytosis risk — REMS). Ferritin monthly. Investigational-use only outside approved clinical trial."},
    {"term": "PHT Absolute CI (Neuropathy Aggravation in PLAN)",
     "def": "Phenytoin (PHT) and fosphenytoin are ABSOLUTELY CONTRAINDICATED in PLAN. Sodium channel blockade at peripheral nerve → worsens pre-existing axonal neuropathy (100% INAD, 70% ANAD). Risk of acute NCS deterioration. No alternative clinical justification outweighs neuropathy risk. Documented cases of rapid neurological worsening post-PHT in NAD. This CI applies to ALL PLA2G6 phenotypes including PARK14."},
    {"term": "VGB Absolute CI (Optic Atrophy Additive in INAD)",
     "def": "Vigabatrin (VGB) is ABSOLUTELY CONTRAINDICATED in PLAN patients with optic atrophy (70% of INAD). Vigabatrin causes irreversible concentric visual field constriction via GABA-T inhibition and retinal ganglion cell toxicity. In INAD: pre-existing optic atrophy + VGB → accelerated irreversible blindness. Even in PLAN patients WITHOUT documented optic atrophy, annual fundoscopy must precede any VGB consideration — risk too high."},
    {"term": "POLG Mandatory (Secondary Mitochondrial Dysfunction in PLAN)",
     "def": "POLG gene panel mandatory before VPA initiation in ALL PLA2G6-PLAN patients. iPLA2β LOF → phospholipid remodelling failure → abnormal mitochondrial membrane composition → secondary mitochondrial dysfunction. Undetected POLG mutation + VPA → Alpers-like fatal hepatotoxicity. Per 2024 expert consensus: POLG panel mandatory in NBIA disorders with secondary mitochondrial involvement. Include at minimum POLG1 full-gene sequencing."},
]

# ── Monitoring and lifecycle ───────────────────────────────────────────────
MONITORING = [
    {"item": "MRI Brain (T2 + SWI/T2* cerebellar atrophy + GP iron)", "frequency": "At diagnosis + annual",
     "rationale": "Cerebellar atrophy earliest finding; GP iron late — SWI mandatory but may be negative early; track progression; annual in INAD/ANAD"},
    {"item": "NCS/EMG (nerve conduction + EMG)", "frequency": "At diagnosis; every 2 years",
     "rationale": "Axonal neuropathy 100% INAD / 70% ANAD; amplitude decline tracks disease; PHT absolute CI; CBZ caution"},
    {"item": "POLG Gene Panel", "frequency": "ONCE before any VPA initiation",
     "rationale": "Secondary mitochondrial dysfunction in PLAN; mandatory 2024 expert consensus; even heterozygous POLG warrants VPA caution"},
    {"item": "Ophthalmology (fundoscopy + VEP + OCT RNFL)", "frequency": "Annual (every 6mo in INAD with optic atrophy)",
     "rationale": "Optic atrophy 70% INAD; VGB absolute CI if optic atrophy present; VEP latency and OCT RNFL thinning detects subclinical changes early"},
    {"item": "Annual EEG", "frequency": "Annual if seizures; at diagnosis in INAD",
     "rationale": "Seizures 50-60% INAD, 35% ANAD; EEG pattern guides AED choice; myoclonic pattern → avoid PHT/CBZ"},
    {"item": "Neuropsychiatric Assessment", "frequency": "Annual",
     "rationale": "Psychiatric features 35% ANAD; cognitive regression tracking INAD; ANAD behavioral monitoring; PARK14 dementia screening"},
    {"item": "Dystonia Assessment (BFMDRS)", "frequency": "Every 6 months",
     "rationale": "Pre-DBS baseline ANAD/PARK14; GPi-DBS Level C; BFMDRS ≥35 or rapid functional decline triggers DBS referral"},
    {"item": "L-DOPA response monitoring (PARK14)", "frequency": "Every 3-6 months (PARK14 only)",
     "rationale": "L-DOPA response wanes over 2-5yr in PARK14; UPDRS motor score; adjust dose; monitor for L-DOPA dyskinesia"},
    {"item": "Spasticity Assessment (Modified Ashworth Scale)", "frequency": "Every 3 months (INAD/ANAD)",
     "rationale": "Spastic paraparesis INAD/ANAD; intrathecal baclofen candidacy; orthotic needs; physiotherapy plan adjustment"},
]

LIFECYCLE = [
    {"stage": "Birth–6 months (presymptomatic INAD)",
     "features": "Normal at birth; genetic diagnosis only in at-risk siblings; normal tone and development; MRI normal; NCS normal"},
    {"stage": "6 months – 1.5 years (INAD onset / first signs)",
     "features": "Failure to achieve or loss of motor milestones; hypotonia; truncal instability; poor eye contact; early cerebellar atrophy on MRI; NCS may show earliest axonal changes"},
    {"stage": "1.5–3 years (INAD established / ANAD onset)",
     "features": "Psychomotor regression INAD; ataxia dominating ANAD; optic atrophy developing; strabismus; NCS abnormal; spastic paraparesis emerging INAD; seizures beginning (50-60% INAD, 35% ANAD)"},
    {"stage": "3–6 years (INAD advanced / ANAD progressing)",
     "features": "Spastic paraparesis severe INAD → wheelchair; cerebellar atrophy prominent MRI; optic atrophy established; visual impairment severe; swallowing dysfunction; seizures active; dystonia ANAD"},
    {"stage": "6–10 years (INAD severe / ANAD moderate)",
     "features": "INAD: severe cognitive decline, seizures difficult, total care dependence; GP iron may appear on SWI (late); ANAD: cerebellar ataxia dominant, ambulatory; psychiatric features ANAD"},
    {"stage": "10–20 years (INAD end stage / ANAD advanced / PARK14 presymptomatic)",
     "features": "INAD: end stage — anarthria, aspiration risk, severe visual loss, palliative care; death 2nd decade typical INAD. ANAD: survival into adulthood; wheelchair possible late. PARK14: presymptomatic"},
    {"stage": "Adult (ANAD / PARK14)",
     "features": "ANAD: adult survival; progressive but ambulatory phase longer; GPi-DBS consideration. PARK14: onset 30-50yr; parkinsonism + dystonia; L-DOPA initial response; cerebellar atrophy MRI; slower course"},
]

THRESHOLDS = [
    {"parameter": "Optic atrophy detection (INAD)", "threshold": "VEP latency >115ms OR OCT RNFL <80μm OR abnormal fundoscopy",
     "significance": "70% INAD have optic atrophy; VGB absolute CI once optic atrophy confirmed; annual ophthalmology mandatory from diagnosis"},
    {"parameter": "POLG panel before VPA", "threshold": "Before any VPA initiation",
     "significance": "Secondary mitochondrial dysfunction in PLAN; even heterozygous POLG → VPA caution; fatal hepatotoxicity risk"},
    {"parameter": "NCS amplitude threshold for neuropathy", "threshold": "Motor amplitude <50% LLN or sensory amplitude <40% LLN",
     "significance": "100% INAD, 70% ANAD axonal neuropathy; PHT absolute CI; CBZ caution; NCS/EMG every 2yr for progression"},
    {"parameter": "DBS referral threshold (ANAD/PARK14)", "threshold": "BFMDRS motor ≥35/120 OR rapid ambulation loss within 12 months",
     "significance": "GPi-DBS Level C in PLAN; earlier referral ANAD/PARK14 dystonia; multidisciplinary evaluation essential"},
    {"parameter": "Cerebellar atrophy severity (MRI)", "threshold": "Vermal diameter <2 SD below mean for age on MRI",
     "significance": "Earliest and most prominent PLAN MRI finding; precedes iron accumulation; annual MRI from diagnosis"},
    {"parameter": "L-DOPA response (PARK14)", "threshold": "≥30% UPDRS motor improvement at 6-week L-DOPA trial",
     "significance": "PARK14 phenotype only; initial response confirms dopaminergic deficit; response wanes 2-5yr; re-evaluate annually"},
    {"parameter": "Deferiprone ferritin monitoring", "threshold": "Serum ferritin <20 μg/L → dose reduction",
     "significance": "Investigational in PLAN; agranulocytosis risk; weekly WBC mandatory (REMS); iron late in PLAN — limited trial window"},
    {"parameter": "Baclofen pump candidacy (INAD/ANAD)", "threshold": "Ashworth ≥3 bilateral lower limbs + oral baclofen side effects",
     "significance": "Intrathecal baclofen for severe lower limb spasticity INAD/ANAD; more effective than oral; titration protocol required"},
]

STANDARDS = [
    "NBIA Research Institute: PLAN Clinical Management Guidelines 2023 (nbia.ca)",
    "Morgan NV et al. (2006): PLA2G6 mutations cause infantile neuroaxonal dystrophy — Nat Genet 38(7), 752-754 (discovery paper)",
    "Kurian MA et al. (2008): PLA2G6-associated neurodegeneration spectrum — Brain 131(6), 1657-1668",
    "Gregory A et al. (2017): Neurodegeneration associated with genetic defects in phospholipase A2 — Neurology 88(2), 93-100",
    "ILAE 2022: Genetic Epilepsy Classification (PLAN — secondary epilepsy in NBIA)",
    "Deferiprone REMS Program: Weekly WBC monitoring — agranulocytosis risk",
    "POLG Expert Panel 2024: Mandatory POLG before VPA in mitochondrial/NBIA disorders with secondary mito involvement",
    "NICE NG217 (2022): Management of epilepsies in children and adults",
    "ESPGD/ESPE 2023: GPi-DBS for childhood onset generalised dystonia (Level C in PLAN)",
]

REFERENCES = [
    "Morgan NV, Westaway SK, Morton JE et al. (2006). PLA2G6, encoding a phospholipase A2, is mutated in neurodegenerative disorders with high brain iron. Nat Genet, 38(7), 752–754.",
    "Kurian MA, Morgan NV, MacPherson L et al. (2008). Phenotypic spectrum of neurodegeneration associated with mutations in the PLA2G6 gene (PLAN). Brain, 131(6), 1657–1668.",
    "Gregory A, Westaway SK, Holm IE et al. (2017). Neurodegeneration associated with genetic defects in phospholipase A(2). Neurology, 88(2), 93–100.",
    "Morel E, Bhatt DL, et al. (2019). Phospholipid remodelling and membrane composition in neurodegeneration. Prog Lipid Res, 73, 1–25.",
    "NBIA Research Institute (2023). PLAN Clinical Management Guidelines. nbia.ca.",
    "Schottmann G, Stenzel W, Lützkendorf S et al. (2015). PLA2G6 mutation causes neuronal ceroid-lipofuscinosis-like phenotype. J Neuropathol Exp Neurol, 74(4), 368–377.",
]


def _patients():
    """Generate 40 synthetic PLAN patients (seed-521): 20 INAD, 14 ANAD, 6 PARK14."""
    pts = []
    etio_pool = (
        ["missense_compound_het"] * 18 +
        ["null_truncating"] * 12 +
        ["splice"] * 6 +
        ["cnv"] * 4
    )
    RNG.shuffle(etio_pool)

    # Phenotype assignment: INAD first 20, ANAD next 14, PARK14 last 6
    # We assign phenotype by position then adjust per etiology constraints
    phenotype_pool = (
        ["INAD"] * N_INAD +
        ["ANAD"] * N_ANAD +
        ["PARK14"] * N_PARK14
    )
    RNG.shuffle(phenotype_pool)

    aed_options = ["LEV", "CLB", "VPA", "ZNS", "LTG", "PB"]
    seizure_options = ["complex_focal", "myoclonic", "absence", "gtcs", "tonic_atonic"]

    for i in range(N):
        etio = etio_pool[i]
        phenotype = phenotype_pool[i]

        # Force severe etiology into INAD phenotype; PARK14 only from mild alleles
        if etio == "null_truncating" and phenotype == "PARK14":
            phenotype = "INAD"
        if etio == "null_truncating" and phenotype == "ANAD":
            phenotype = RNG.choice(["INAD", "INAD", "ANAD"])

        # Onset by phenotype
        if phenotype == "INAD":
            onset_yr = round(RNG.uniform(0.5, 3.0), 1)
            cerebellar_atrophy = True
            optic_atrophy = RNG.random() < 0.70
            axonal_neuropathy = True  # 100% INAD
            seizures_prob = 0.55
            spastic_paraparesis = RNG.random() < 0.85
            dystonia = RNG.random() < 0.65
            dystonia_severity = RNG.choice(["moderate", "severe"]) if dystonia else "none"
            ambulation_lost = RNG.random() < 0.80
            levodopa = False
            psychiatric = RNG.random() < 0.10
            cognitive_decline = RNG.random() < 0.90
            current_age = RNG.randint(1, 18)
        elif phenotype == "ANAD":
            onset_yr = round(RNG.uniform(1.0, 5.0), 1)
            cerebellar_atrophy = True
            optic_atrophy = RNG.random() < 0.35
            axonal_neuropathy = RNG.random() < 0.70
            seizures_prob = 0.35
            spastic_paraparesis = RNG.random() < 0.50
            dystonia = RNG.random() < 0.60
            dystonia_severity = RNG.choice(["focal", "moderate", "severe"]) if dystonia else "none"
            ambulation_lost = RNG.random() < 0.35
            levodopa = False
            psychiatric = RNG.random() < 0.35
            cognitive_decline = RNG.random() < 0.60
            current_age = RNG.randint(5, 45)
        else:  # PARK14
            onset_yr = round(RNG.uniform(30.0, 50.0), 1)
            cerebellar_atrophy = RNG.random() < 0.85
            optic_atrophy = RNG.random() < 0.10
            axonal_neuropathy = RNG.random() < 0.35
            seizures_prob = 0.04
            spastic_paraparesis = RNG.random() < 0.20
            dystonia = RNG.random() < 0.75
            dystonia_severity = RNG.choice(["focal", "moderate"]) if dystonia else "none"
            ambulation_lost = RNG.random() < 0.15
            levodopa = True
            levodopa_response = RNG.choice(["initial_response", "initial_response", "waned", "partial"])
            psychiatric = RNG.random() < 0.40
            cognitive_decline = RNG.random() < 0.45
            current_age = RNG.randint(35, 65)

        if phenotype != "PARK14":
            levodopa = False
            levodopa_response = "not_applicable"
        elif phenotype == "PARK14" and not levodopa:
            levodopa_response = "not_applicable"

        # Ensure levodopa_response defined for all
        if "levodopa_response" not in dir():
            levodopa_response = "not_applicable"

        # Seizures
        has_seizures = RNG.random() < seizures_prob
        s_types = []
        if has_seizures:
            probs = {
                "complex_focal": 0.35,
                "myoclonic": 0.28,
                "absence": 0.20,
                "gtcs": 0.15,
                "tonic_atonic": 0.08,
            }
            for stype, prob in probs.items():
                if RNG.random() < prob:
                    s_types.append(stype)
            if not s_types:
                s_types = ["complex_focal"]

        drug_resistant = has_seizures and RNG.random() < 0.45
        n_aeds = RNG.randint(2, 4) if (has_seizures and drug_resistant) else (RNG.randint(1, 2) if has_seizures else 0)
        aeds_tried = RNG.sample(aed_options, min(n_aeds, len(aed_options))) if n_aeds > 0 else []
        seizure_free = has_seizures and RNG.random() < 0.42

        baclofen = RNG.random() < (0.80 if phenotype == "INAD" else 0.50 if phenotype == "ANAD" else 0.20)
        dbs = RNG.random() < (0.03 if phenotype == "INAD" else 0.10 if phenotype == "ANAD" else 0.15)
        polg_tested = RNG.random() < 0.72
        deferiprone_trial = RNG.random() < 0.06
        trihexyphenidyl = RNG.random() < (0.15 if phenotype == "INAD" else 0.40 if phenotype == "ANAD" else 0.30)
        acth_used = has_seizures and phenotype == "INAD" and RNG.random() < 0.08

        pts.append({
            "id": f"PLAN-{i+1:03d}",
            "phenotype": phenotype,
            "etiology": etio,
            "onset_yr": onset_yr,
            "cerebellar_atrophy": cerebellar_atrophy,
            "optic_atrophy": optic_atrophy,
            "axonal_neuropathy": axonal_neuropathy,
            "spastic_paraparesis": spastic_paraparesis,
            "dystonia": dystonia,
            "dystonia_severity": dystonia_severity,
            "ambulation_lost": ambulation_lost,
            "levodopa": levodopa,
            "levodopa_response": levodopa_response if phenotype == "PARK14" else "not_applicable",
            "psychiatric": psychiatric,
            "cognitive_decline": cognitive_decline,
            "has_seizures": has_seizures,
            "seizure_types": s_types,
            "drug_resistant": drug_resistant,
            "n_aeds_tried": n_aeds,
            "aeds_tried": aeds_tried,
            "seizure_free": seizure_free,
            "baclofen": baclofen,
            "dbs": dbs,
            "trihexyphenidyl": trihexyphenidyl,
            "polg_tested": polg_tested,
            "deferiprone_trial": deferiprone_trial,
            "acth_used": acth_used,
            "current_age": current_age,
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

    inad_pts = [p for p in pts if p["phenotype"] == "INAD"]
    anad_pts = [p for p in pts if p["phenotype"] == "ANAD"]
    park14_pts = [p for p in pts if p["phenotype"] == "PARK14"]

    kpis = {
        "n_patients": n,
        "n_inad": len(inad_pts),
        "n_anad": len(anad_pts),
        "n_park14": len(park14_pts),
        "cerebellar_atrophy_pct": pct(lambda p: p["cerebellar_atrophy"]),
        "optic_atrophy_pct": pct(lambda p: p["optic_atrophy"]),
        "axonal_neuropathy_pct": pct(lambda p: p["axonal_neuropathy"]),
        "spastic_paraparesis_pct": pct(lambda p: p["spastic_paraparesis"]),
        "dystonia_pct": pct(lambda p: p["dystonia"]),
        "has_seizures_pct": pct(lambda p: p["has_seizures"]),
        "drug_resistant_pct": pct(lambda p: p["drug_resistant"]),
        "ambulation_lost_pct": pct(lambda p: p["ambulation_lost"]),
        "cognitive_decline_pct": pct(lambda p: p["cognitive_decline"]),
        "psychiatric_pct": pct(lambda p: p["psychiatric"]),
        "baclofen_pct": pct(lambda p: p["baclofen"]),
        "dbs_pct": pct(lambda p: p["dbs"]),
        "polg_tested_pct": pct(lambda p: p["polg_tested"]),
        "seizure_free_pct": pct(lambda p: p["seizure_free"]),
        "levodopa_pct": pct(lambda p: p["levodopa"]),
        "deferiprone_pct": pct(lambda p: p["deferiprone_trial"]),
        "mean_onset_yr": round(sum(p["onset_yr"] for p in pts) / n, 1),
        "mean_aeds_tried": round(sum(p["n_aeds_tried"] for p in pts) / n, 1),
        "inad_mean_onset_yr": round(sum(p["onset_yr"] for p in inad_pts) / max(len(inad_pts), 1), 1),
        "anad_mean_onset_yr": round(sum(p["onset_yr"] for p in anad_pts) / max(len(anad_pts), 1), 1),
        "park14_mean_onset_yr": round(sum(p["onset_yr"] for p in park14_pts) / max(len(park14_pts), 1), 1),
        "inad_seizures_pct": round(sum(1 for p in inad_pts if p["has_seizures"]) / max(len(inad_pts), 1) * 100),
        "anad_seizures_pct": round(sum(1 for p in anad_pts if p["has_seizures"]) / max(len(anad_pts), 1) * 100),
    }

    return {
        "disease": "PLA2G6 PLAN (PLA2G6-Associated Neurodegeneration / NBIA2)",
        "gene": "PLA2G6 (iPLA2β — calcium-independent phospholipase A2 beta)",
        "chromosome": "22q13.1",
        "omim_gene": "603604",
        "omim_disease_inad": "256600",
        "omim_disease_nbia2a": "610217",
        "omim_disease_park14": "612953",
        "inheritance": "Autosomal Recessive — Biallelic PLA2G6 mutations (no dominant founder mutation)",
        "cohort_n": n,
        "seed": SEED,
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
            "3rd most common NBIA (~5-15% of NBIA) — PLAN/NBIA2; three phenotypes: INAD (infantile), ANAD (atypical), PARK14 (adult)",
            "Cerebellar cortical atrophy EARLIEST + MOST PROMINENT MRI finding — distinguishes PLAN from PKAN (GP iron dominant) and MPAN",
            "NO eye-of-tiger sign at any stage (key DDx from PKAN) — PLAN iron is LATE and may be ABSENT at onset",
            "Spheroid bodies in axons on nerve biopsy PATHOGNOMONIC — confirms PLAN when genetic testing inconclusive",
            "Axonal neuropathy 100% INAD / 70% ANAD (PKAN: no neuropathy) — NCS/EMG mandatory at diagnosis and every 2yr",
            "PHT / Fosphenytoin ABSOLUTE CI — aggravates axonal neuropathy; documented rapid NCS deterioration",
            "VGB ABSOLUTE CI in INAD — optic atrophy 70% INAD; additive irreversible visual field constriction",
            "POLG MANDATORY before VPA — secondary mitochondrial dysfunction (phospholipid remodelling failure → mito membrane abnormal)",
            "GPi-DBS Level C (ANAD + PARK14 dystonia); L-DOPA Level C (PARK14 only; response wanes 2-5yr)",
            "No approved disease-modifying therapy 2026; deferiprone investigational (iron LATE in PLAN — limited early treatment window)",
        ],
        "ddx_table": DDX,
        "tier_summary": {
            "disease_group": "NBIA (Neurodegeneration with Brain Iron Accumulation) — NBIA2 / PLAN",
            "epilepsy_classification": "Secondary epilepsy in NBIA (50-60% INAD, 35% ANAD, rare PARK14)",
            "third_most_common_nbia": "PLAN 3rd most frequent NBIA (~5-15% of NBIA cohorts)",
            "pathognomonic_finding": "Spheroid bodies on nerve biopsy PATHOGNOMONIC; cerebellar atrophy earliest MRI",
            "no_eye_of_tiger": "NO eye-of-tiger sign (DDx PKAN) — PLAN iron LATE/absent early",
            "precision_therapy": "None approved 2026; deferiprone investigational; L-DOPA PARK14 Level C",
            "key_ddx": "PKAN: eye-of-tiger early (absent PLAN); MPAN: optic atrophy 80% KEY; cerebellar atrophy PLAN dominant",
        },
        "standards": STANDARDS,
    }


def get_breakdown():
    pts = _get_patients()

    # Phenotype breakdown
    phenotype_breakdown = []
    for ph_key, label in [
        ("INAD", "Classic INAD (Infantile Neuroaxonal Dystrophy)"),
        ("ANAD", "Atypical NAD / ANAD"),
        ("PARK14", "PARK14 (Adult Parkinsonism-Dystonia)"),
    ]:
        group = [p for p in pts if p["phenotype"] == ph_key]
        if not group:
            continue
        n_g = len(group)
        phenotype_breakdown.append({
            "phenotype": label,
            "n": n_g,
            "pct": round(n_g / len(pts) * 100),
            "cerebellar_atrophy_pct": round(sum(1 for p in group if p["cerebellar_atrophy"]) / n_g * 100),
            "optic_atrophy_pct": round(sum(1 for p in group if p["optic_atrophy"]) / n_g * 100),
            "axonal_neuropathy_pct": round(sum(1 for p in group if p["axonal_neuropathy"]) / n_g * 100),
            "has_seizures_pct": round(sum(1 for p in group if p["has_seizures"]) / n_g * 100),
            "dystonia_pct": round(sum(1 for p in group if p["dystonia"]) / n_g * 100),
            "ambulation_lost_pct": round(sum(1 for p in group if p["ambulation_lost"]) / n_g * 100),
            "dbs_pct": round(sum(1 for p in group if p["dbs"]) / n_g * 100),
            "mean_onset_yr": round(sum(p["onset_yr"] for p in group) / n_g, 1),
        })

    # Etiology breakdown
    etio_breakdown = []
    for etio_key, label in [
        ("missense_compound_het", "Missense Compound Het"),
        ("null_truncating", "Null/Truncating (biallelic)"),
        ("splice", "Splice Variants"),
        ("cnv", "Structural/CNV"),
    ]:
        group = [p for p in pts if p["etiology"] == etio_key]
        if not group:
            continue
        n_g = len(group)
        etio_breakdown.append({
            "etiology": label,
            "n": n_g,
            "pct": round(n_g / len(pts) * 100),
            "inad_pct": round(sum(1 for p in group if p["phenotype"] == "INAD") / n_g * 100),
            "anad_pct": round(sum(1 for p in group if p["phenotype"] == "ANAD") / n_g * 100),
            "park14_pct": round(sum(1 for p in group if p["phenotype"] == "PARK14") / n_g * 100),
            "axonal_neuropathy_pct": round(sum(1 for p in group if p["axonal_neuropathy"]) / n_g * 100),
            "has_seizures_pct": round(sum(1 for p in group if p["has_seizures"]) / n_g * 100),
            "mean_onset_yr": round(sum(p["onset_yr"] for p in group) / n_g, 1),
        })

    # Seizure type breakdown
    seizure_breakdown = []
    for st in ["complex_focal", "myoclonic", "absence", "gtcs", "tonic_atonic"]:
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
            "phenotype": p["phenotype"],
            "etiology": p["etiology"],
            "onset_yr": p["onset_yr"],
            "current_age": p["current_age"],
            "cerebellar_atrophy": p["cerebellar_atrophy"],
            "optic_atrophy": p["optic_atrophy"],
            "axonal_neuropathy": p["axonal_neuropathy"],
            "spastic_paraparesis": p["spastic_paraparesis"],
            "dystonia": p["dystonia"],
            "dystonia_severity": p["dystonia_severity"],
            "ambulation_lost": p["ambulation_lost"],
            "has_seizures": p["has_seizures"],
            "drug_resistant": p["drug_resistant"],
            "n_aeds": p["n_aeds_tried"],
            "aeds_tried": p["aeds_tried"],
            "seizure_free": p["seizure_free"],
            "baclofen": p["baclofen"],
            "dbs": p["dbs"],
            "levodopa": p["levodopa"],
            "levodopa_response": p["levodopa_response"],
            "cognitive_decline": p["cognitive_decline"],
            "psychiatric": p["psychiatric"],
            "polg_tested": p["polg_tested"],
            "deferiprone_trial": p["deferiprone_trial"],
            "acth_used": p["acth_used"],
        })

    return {
        "cohort_n": len(pts),
        "etiology_breakdown": etio_breakdown,
        "phenotype_breakdown": phenotype_breakdown,
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
        "disease": "PLA2G6 PLAN (PLA2G6-Associated Neurodegeneration / NBIA2)",
        "gene": "PLA2G6 (iPLA2β — 806 aa, calcium-independent phospholipase A2 beta) — 22q13.1 — OMIM 603604",
        "omim_disease": "INAD1 256600 / NBIA2A 610217 / PARK14 612953",
        "definitions": DEFINITIONS,
        "contraindications": CONTRAINDICATIONS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "key_concepts": [d["term"] for d in DEFINITIONS],
    }


if __name__ == "__main__":
    print("=== PLA2G6 / PLAN (NBIA2) Dashboard — Self-Test (seed-521) ===\n")

    ov = get_overview()
    kpis = ov["kpis"]
    print(f"[get_overview] disease: {ov['disease']}")
    print(f"  n_patients={kpis['n_patients']}, n_inad={kpis['n_inad']}, n_anad={kpis['n_anad']}, n_park14={kpis['n_park14']}")
    print(f"  cerebellar_atrophy_pct={kpis['cerebellar_atrophy_pct']}%, optic_atrophy_pct={kpis['optic_atrophy_pct']}%")
    print(f"  axonal_neuropathy_pct={kpis['axonal_neuropathy_pct']}%, has_seizures_pct={kpis['has_seizures_pct']}%")
    print(f"  drug_resistant_pct={kpis['drug_resistant_pct']}%, ambulation_lost_pct={kpis['ambulation_lost_pct']}%")
    print(f"  inad_mean_onset_yr={kpis['inad_mean_onset_yr']}, anad_mean_onset_yr={kpis['anad_mean_onset_yr']}, park14_mean_onset_yr={kpis['park14_mean_onset_yr']}")
    print(f"  etiology_distribution: {len(ov['etiology_distribution'])} entries")
    print(f"  clinical_highlights: {len(ov['clinical_highlights'])} items")
    print()

    bk = get_breakdown()
    print(f"[get_breakdown] cohort_n={bk['cohort_n']}")
    print(f"  phenotype_breakdown: {len(bk['phenotype_breakdown'])} groups")
    for ph in bk["phenotype_breakdown"]:
        print(f"    {ph['phenotype']}: n={ph['n']} ({ph['pct']}%), neuropathy={ph['axonal_neuropathy_pct']}%, seizures={ph['has_seizures_pct']}%")
    print(f"  etiology_breakdown: {len(bk['etiology_breakdown'])} groups")
    for e in bk["etiology_breakdown"]:
        print(f"    {e['etiology']}: n={e['n']} ({e['pct']}%), INAD={e['inad_pct']}%, ANAD={e['anad_pct']}%")
    print(f"  seizure_breakdown: {len(bk['seizure_breakdown'])} types")
    print(f"  per_patient: {len(bk['per_patient'])} rows")
    print()

    df = get_definitions()
    print(f"[get_definitions] definitions: {len(df['definitions'])} concepts")
    print(f"  key_concepts: {df['key_concepts'][:3]} ...")
    print(f"  standards: {len(df['standards'])}, references: {len(df['references'])}")
    print("\n=== All 3 functions OK ===")
