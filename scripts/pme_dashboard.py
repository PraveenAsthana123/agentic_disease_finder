"""Progressive Myoclonic Epilepsy (PME) Dashboard — a clinically heterogeneous group of rare
inherited epilepsies characterised by the triad of cortical myoclonus, generalised tonic-clonic
seizures, and progressive neurological deterioration (cerebellar ataxia + cognitive decline).

Hallmarks:
  Cortical Myoclonus: action-sensitive, stimulus-sensitive (photosensitive, touch), irregular
    jerks arising from hyper-excitable sensorimotor cortex — pathognomonic for PME; giant SSEPs
    and enhanced C-reflex on neurophysiology.
  GTCS: generalised tonic-clonic seizures present in 90%+ across all PME subtypes; often the
    first clinically recognised seizure (myoclonus pre-dates GTCS by months to years).
  Progressive Deterioration: cerebellar ataxia → falls/dysarthria/dysphagia, cognitive decline
    from mild (ULD) to severe dementia (Lafora/NCL), progressive disability mandatory feature.
  Photosensitivity: PPR on EEG in 65–80% of ULD and Lafora Disease; photic provocation is a
    clinical diagnostic test; eye-closure sensitivity common.
  EEG hallmark: generalised polyspike-and-slow-wave (GPSW) background; fragmented background
    with fast activity; giant SSEPs (N20 amplitude >20 µV); absence of focal onset in most PME.

FIVE MAIN SUBTYPES:
  Unverricht-Lundborg Disease (ULD/EPM1): CSTB gene (cystatin B); onset 6–15 y; slowest
    progression; 50-y survival documented; 35% of PME cohort.
  Lafora Disease (EPM2A/NHLRC1): polyglucosan body inclusions; onset 12–17 y; rapid dementia
    + psychiatric features; fatal within 10 y in 90%; 20%.
  MERRF (Myoclonic Epilepsy with Ragged Red Fibers): mitochondrial DNA (MT-TK 80%), maternal
    transmission; ragged red fibers on muscle biopsy; hearing loss + myopathy + myoclonus; 15%.
  Neuronal Ceroid Lipofuscinosis (NCL/Batten Disease): CLN genes (CLN1–14); visual failure →
    blindness + dementia + myoclonus; juvenile CLN3 most common; 20%.
  Sialidosis Type I / Unclassified PME: NEU1 gene; cherry-red spot + myoclonus; onset 10–20y;
    slowest progression in sialidosis group; 10%.

ABSOLUTE CONTRAINDICATIONS (MANDATORY):
  CBZ / OXC / PHT in ALL PME subtypes → exacerbate cortical myoclonus (Na-channel block
    paradoxically worsens myoclonus via cortical disinhibition — high-grade evidence).
  Valproate in MERRF (MT-TK mutations) → mitochondrial hepatotoxicity / Alpers-like acute
    liver failure; potentially fatal — must confirm MERRF exclusion before VPA prescription.
  Vigabatrin in ALL PME → GABA-T inhibition worsens action myoclonus; irreversible visual
    field loss compounds NCL visual failure.
  Gabapentin / Pregabalin → paradoxical myoclonus worsening (alpha-2-delta calcium channel
    modulation increases cortical excitability in PME cortex).
  Lamotrigine monotherapy → can exacerbate myoclonus in ULD (use only with caution as add-on
    at low dose; monitor for myoclonus worsening).

FIRST-LINE / ADJUNCT TREATMENT:
  Valproate: Level A (ULD/Lafora/NCL) — avoid MERRF.
  Clonazepam: Level B adjunct — myoclonus-specific (GABA-A potentiation); tolerance risk.
  Levetiracetam: Level B — broad-spectrum, safe in mitochondrial disease.
  Piracetam: Level A in ULD specifically (16–24 g/day in adults; cortical myoclonus reduction
    in >70% of ULD patients; Genton 2009 Epilepsia).
  Perampanel (AMPA antagonist): Level A add-on — especially effective in Lafora cortical
    myoclonus (AMPA hyperexcitability hallmark of polyglucosan disease).
  Zonisamide: Level B — dual mechanism (Na-channel + carbonic anhydrase); safe in MERRF.
  Clobazam: Level B adjunct — serotonergic + GABA modulation; less tolerance than CLN.
  N-Acetylcysteine (NAC): experimental antioxidant for ULD (Phase II trial data —
    neuroprotective via GSH replenishment; used compassionate in CSTB-null mouse model).

References:
  - Berkovic SF et al. 1993 Ann Neurol (Baltic/Mediterranean myoclonus = ULD — 100 cases, genotype-phenotype)
  - Minassian BA 2001 Trends Neurosci (Lafora disease — polyglucosan pathophysiology + EPM2A)
  - DiMauro S & Hirano M 2004 GeneReviews (MERRF — MT-TK mutations, ragged red fibers, management)
  - Mole SE et al. 2011 Biochim Biophys Acta (NCL/Batten — CLN gene classification, diagnostics)
  - Genton P et al. 2009 Epilepsia (Piracetam for ULD cortical myoclonus — Level A systematic review)
  - Zara F & Guerrini R 2019 Epilepsia (ILAE PME Task Force — classification and genetic update)
Data: live clinical.db (41 epilepsy patients, deterministic PME overlay)
      + curated PME pharmacology / etiology / seizure-type / trigger catalogs."""

import sqlite3
import json
from pathlib import Path
from datetime import date

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"
_PROJECT = Path(__file__).resolve().parent.parent


# ─── helpers ────────────────────────────────────────────────────────────────

def _db_rows(sql, params=()):
    try:
        con = sqlite3.connect(DB)
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _seed(pid):
    """Deterministic hash from patient_id string."""
    h = 0
    for c in str(pid):
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


# ─── Etiology catalog ────────────────────────────────────────────────────────

_ETIOLOGIES = [
    {
        "etiology": "Unverricht-Lundborg Disease (ULD / EPM1 — CSTB Mutation)",
        "category": "Genetic / Metabolic",
        "pct": 35,
        "mechanism": (
            "ULD is caused by loss-of-function mutations in CSTB (chromosome 21q22.3), encoding "
            "cystatin B — an endogenous inhibitor of lysosomal cysteine proteases (cathepsin B/D/H/L). "
            "CSTB deficiency → unrestrained cathepsin-mediated proteolysis → cerebellar Purkinje cell "
            "degeneration + cortical GABA interneuron loss → cortical hyperexcitability + progressive "
            "ataxia. The pathogenic mutation in 90%+ of European ULD is a dodecamer repeat expansion "
            "(CCCCGCCCCGCG repeat) in the CSTB promoter region, reducing transcription 10-fold. "
            "ULD is the most common PME in Western/Northern Europe (Finnish disease enrichment); "
            "onset 6–15 years. Notably slowest progression of all PME subtypes — 50-year survival "
            "documented. Cognitive decline mild to moderate; severe dementia uncommon. "
            "Mouse model (Cstb-knockout): cerebellar neurodegeneration, ataxia, myoclonus — "
            "N-acetylcysteine rescues phenotype (clinical trials ongoing)."
        ),
        "eeg_correlate": (
            "Generalised polyspike-and-slow-wave (2.5–3.5 Hz GPSW); giant SSEPs (N20 >20 µV "
            "bilateral — 10× normal amplitude); photoparoxysmal response (PPR) in 70–80% of ULD; "
            "enhanced C-reflex on EMG (long-loop reflex: cortex → spinal cord → cortex). "
            "Background: progressive loss of normal alpha rhythm; increased diffuse theta-delta "
            "slowing. Jerk-locked EEG back-averaging: cortical discharge precedes myoclonus "
            "by ~20 ms (C3/C4/Fz maximal)."
        ),
        "mri_finding": (
            "MRI: cerebellar vermis atrophy (progressive over 5–10 years); mild cortical atrophy. "
            "Supratentorial volume usually preserved early; periventricular T2 signal may appear "
            "in advanced disease. Diffusion imaging: corticospinal tract degeneration in severe "
            "long-standing ULD (DTI fractional anisotropy reduction). MRS: normal early; reduced "
            "NAA in cerebellar dentate nucleus in moderate-advanced disease."
        ),
        "clinical_note": (
            "Confirm CSTB dodecamer expansion (Sanger + triplet-primed PCR). Piracetam (16–24 g/day) "
            "is Level A evidence for cortical myoclonus in ULD — initiate alongside valproate. "
            "NAC (N-acetylcysteine 4.8 g/day) compassionate use for antioxidant neuroprotection. "
            "AVOID carbamazepine/oxcarbazepine/phenytoin — these dramatically worsen myoclonus. "
            "Family genetic testing (AR inheritance): screen siblings with EEG + genetic panel."
        ),
    },
    {
        "etiology": "Lafora Disease (EPM2A / NHLRC1 — Polyglucosan Body Disease)",
        "category": "Genetic / Metabolic",
        "pct": 20,
        "mechanism": (
            "Lafora disease is caused by mutations in EPM2A (encoding laforin, a dual-specificity "
            "phosphatase — 80% of cases) or NHLRC1 (encoding malin, an E3 ubiquitin ligase — 20%). "
            "The laforin-malin complex normally dephosphorylates glycogen and targets glycogen "
            "synthase for ubiquitin-mediated degradation. Loss of function → hyperphosphorylated "
            "insoluble polysaccharide (Lafora bodies = polyglucosan inclusions) accumulate in "
            "neuronal perikarya, liver, skin sweat-gland ducts, muscle, heart. Neurons cannot "
            "clear Lafora bodies → progressive neuronal death. AMPA receptor hyperactivation "
            "is a key mechanism for cortical myoclonus in Lafora — perampanel (AMPA antagonist) "
            "specifically targets this. Onset: 12–17 years. Fatal within 10 years in 90% of cases. "
            "Occipital seizures (visual hallucinations + ictal amaurosis) hallmark — Lafora bodies "
            "disrupt occipital cortex selectively early in disease. Late-onset EPM2B (NHLRC1) "
            "variant: slower progression, survival to 30s documented."
        ),
        "eeg_correlate": (
            "GPSW at 3–6 Hz; prominent posterior (occipital) spike-wave and photoparoxysmal "
            "response (PPR Grade III–IV in 85%); ictal occipital fast rhythm during visual "
            "seizures; progressive background deterioration (loss of organised posterior rhythm "
            "early — pathognomonic); sleep EEG: fragmented NREM, loss of sleep spindles. "
            "Jerk-locked back-averaging: occipital generator precedes myoclonus in visually "
            "triggered events."
        ),
        "mri_finding": (
            "Early: normal or minimal cortical atrophy. Occipital cortex volume loss precedes "
            "global atrophy by 2–3 years (voxel-based morphometry). Later: progressive global "
            "cortical + subcortical atrophy (thalamus, basal ganglia). T2 FLAIR: periventricular "
            "signal change in advanced disease. PET-FDG: early posterior hypometabolism "
            "before structural MRI changes (diagnostic utility). Skin biopsy gold standard: "
            "periodic acid–Schiff (PAS)-positive Lafora bodies in eccrine sweat gland ducts."
        ),
        "clinical_note": (
            "Skin biopsy (axillary apocrine + eccrine sweat glands) for PAS-positive Lafora bodies "
            "— diagnostic yield ~85% (axillary > abdominal). Gene panel: EPM2A + NHLRC1 sequencing. "
            "Perampanel: most specific AED for Lafora cortical myoclonus (AMPA antagonist mechanism). "
            "AVOID valproate + CBZ/PHT. Genetic counselling: AR inheritance (25% sibling risk). "
            "Palliative care planning early (rapidly fatal — median survival 9 years from diagnosis)."
        ),
    },
    {
        "etiology": "MERRF (Myoclonic Epilepsy with Ragged Red Fibers — MT-TK Mutation)",
        "category": "Mitochondrial",
        "pct": 15,
        "mechanism": (
            "MERRF is a mitochondrial encephalomyopathy caused by point mutations in mitochondrial "
            "tRNA-Lys (MT-TK) gene (m.8344A>G in 80%; m.8356T>C in 10%); maternal inheritance "
            "(exclusively maternally transmitted). Mutation impairs mitochondrial translation → "
            "deficiency of respiratory chain complexes I, III, IV → energy failure in high-demand "
            "neurons (Purkinje cells, basal ganglia, cerebral cortex) and muscle. "
            "Muscle biopsy: ragged red fibers (modified Gomori trichrome stain — mitochondrial "
            "accumulation in subsarcolemmal space); COX-negative fibers. Clinical triad: "
            "myoclonus + GTCS + cerebellar ataxia; variably: hearing loss (sensorineural) + "
            "myopathy + short stature + dementia + cardiomyopathy. Heteroplasmy: percentage "
            "of mutant mtDNA correlates with phenotype severity."
        ),
        "eeg_correlate": (
            "GPSW at 2.5–4 Hz; may be less photosensitive than ULD/Lafora (30–40%); background "
            "slowing proportional to disease severity; focal or multifocal spikes in occipital "
            "and parietal regions; slowing of posterior dominant rhythm. Jerk-locked "
            "back-averaging: parieto-central cortical discharge precedes myoclonus."
        ),
        "mri_finding": (
            "Bilateral symmetrical T2/FLAIR hyperintensity: basal ganglia (putamen/caudate/globus "
            "pallidus), thalami, brainstem (inferior olivary nucleus). Cerebral and cerebellar "
            "atrophy progressive. MRS: elevated lactate peak (1.33 ppm doublet) in affected "
            "regions — highly specific for mitochondrial disease. Plasma lactate/pyruvate ratio "
            ">20:1 (normal <15:1). CSF lactate elevated >2.1 mmol/L."
        ),
        "clinical_note": (
            "ABSOLUTE CONTRAINDICATION: Valproate in MERRF → inhibits mitochondrial beta-oxidation "
            "+ depletes carnitine → acute liver failure / Alpers-like encephalopathy. FATAL. "
            "Confirm MT-TK mutation (blood/urine/muscle biopsy mtDNA). "
            "Treatment: CoQ10 (300–600 mg/day) + L-carnitine (50 mg/kg/day) + riboflavin (400 mg/day). "
            "AEDs: levetiracetam + clonazepam + zonisamide (safe in mitochondrial disease). "
            "Maternal family screening mandatory (sisters/daughters/maternal aunts at 50% risk)."
        ),
    },
    {
        "etiology": "Neuronal Ceroid Lipofuscinosis (NCL / Batten Disease — CLN Genes)",
        "category": "Lysosomal Storage / Genetic",
        "pct": 20,
        "mechanism": (
            "NCL is a family of 14 autosomal recessive lysosomal storage disorders (CLN1–14 genes), "
            "characterised by accumulation of ceroid-lipofuscin (autofluorescent lipoprotein) in "
            "lysosomes of neurons + other tissues. Distinct ultrastructural inclusions on EM: "
            "granular osmiophilic deposits (GROD, CLN1), curvilinear profiles (CLN2), fingerprint "
            "profiles (CLN3), mixed (CLN5–8). The most common form is Juvenile NCL (JNCL/CLN3 "
            "gene, 16p11.2): onset 5–10 y; visual failure (macular degeneration) → blindness by "
            "adolescence; dementia + seizures + myoclonus + cerebellar ataxia + psychiatric. "
            "CLN2 (TPP1 deficiency): onset 2–4 y; language regression → seizures; enzyme "
            "replacement (cerliponase alfa, Brineura): FIRST and only FDA-approved ERT for NCL "
            "(intracerebroventricular infusion; slows decline). CLN10 (cathepsin D): neonatal onset. "
            "Death: CLN1 by 5y; CLN2 by 6–8y; CLN3 by 30–40y."
        ),
        "eeg_correlate": (
            "Giant visual evoked potentials (VEP) at low-frequency photic stimulation (1–2 Hz) — "
            "pathognomonic for CLN2/CLN3; photosensitivity. EEG: progressive background "
            "deterioration; multifocal spike-wave; loss of sleep spindles; eventually near-flat "
            "EEG in end-stage. ERG (electroretinogram): reduced/absent in early CLN1/CLN3 "
            "(photoreceptor degeneration — useful early biomarker before visual symptoms)."
        ),
        "mri_finding": (
            "Progressive cerebral + cerebellar + brainstem atrophy; thalamic T2 hypointensity "
            "(iron deposition, especially CLN3); white matter T2 hyperintensity; cortical "
            "ribboning on diffusion. CLN2: MRI progression correlates with clinical severity "
            "(Hamburg score — standardised NCL MRI rating system). Electron microscopy on skin, "
            "conjunctiva, or leukocyte buffy coat: diagnostic ultrastructural inclusions."
        ),
        "clinical_note": (
            "CLN2: cerliponase alfa (Brineura 300 mg ICV q2 weeks) — refer to NCL centre. "
            "CLN3: no disease-modifying therapy; multidisciplinary palliative approach. "
            "Ophthalmological review urgent (ERG + fundoscopy). Low-stimulation visual "
            "environment. AVOID vigabatrin (worsens existing visual loss). "
            "Genetic panel: CLN1–14 next-generation sequencing. Enzyme assay: PPT1 (CLN1) and "
            "TPP1 (CLN2) in leukocytes — rapid enzyme assay before genetic panel results."
        ),
    },
    {
        "etiology": "Sialidosis Type I / Unclassified PME (NEU1 / Other Rare Variants)",
        "category": "Lysosomal / Unclassified",
        "pct": 10,
        "mechanism": (
            "Sialidosis Type I (Cherry-Red Spot Myoclonus Syndrome) is caused by NEU1 mutations "
            "(alpha-neuraminidase deficiency, 6p21.3); autosomal recessive; onset 10–20 y; "
            "cherry-red spot on fundoscopy + cortical myoclonus + GTCS + absent or minimal "
            "cognitive decline. Slowest progression of sialidoses. Urinary oligosaccharides "
            "elevated. Also includes rare unclassified PMEs: GOSR2 (North Sea PME), KCNC1 "
            "(potassium channel PME), PRICKLE1/2 (autosomal recessive PME), Dentatorubral-Pallidoluysian "
            "Atrophy (DRPLA, ATN1-CAG repeat — rare outside Japan), and Action Myoclonus-Renal "
            "Failure Syndrome (AMRF, SCARB2/LIMP2 mutation). Unclassified PME: full panel "
            "negative → whole-exome sequencing (WES) or whole-genome sequencing (WGS)."
        ),
        "eeg_correlate": (
            "Sialidosis: GPSW with prominent photosensitivity (PPR); giant SSEPs; C-reflex. "
            "DRPLA: GPSW + background slowing; atrophy-related slow background. KCNC1: GPSW "
            "+ posterior dominant rhythm loss. EEG features in unclassified PME overlap with "
            "ULD/Lafora — genetic panel mandatory for subtype diagnosis."
        ),
        "mri_finding": (
            "Sialidosis: normal or mild cerebellar atrophy. DRPLA: cerebral + cerebellar atrophy; "
            "globus pallidus T2 hyperintensity. GOSR2: significant cerebellar atrophy. "
            "Unclassified: progressive atrophy pattern non-specific — WGS + MRI correlation "
            "essential for novel gene discovery."
        ),
        "clinical_note": (
            "Ophthalmology: fundoscopy for cherry-red spot (Sialidosis). Urinary sialyloligosaccharides "
            "(enzyme assay). WES/WGS if standard PME panel negative. DRPLA: CAG repeat analysis "
            "(ATN1 gene). SCARB2: AMRF — must screen renal function (proteinuria → progressive "
            "renal failure + myoclonus). Treat seizures symptomatically with valproate + "
            "levetiracetam + clonazepam; AVOID CBZ/PHT universally."
        ),
    },
]

# ─── Seizure Types ───────────────────────────────────────────────────────────

_SEIZURE_TYPES = [
    {
        "type": "Cortical Myoclonus (Action-Sensitive / Stimulus-Sensitive)",
        "freq_pct": 100,
        "duration_sec": "< 1 s (brief jerk); continuous action myoclonus minutes to hours",
        "description": (
            "Pathognomonic seizure type for PME — present in 100% across all subtypes. Cortical "
            "myoclonus is generated by hyper-excitable sensorimotor cortex (giant SSEPs, C-reflex). "
            "Action myoclonus: triggered by voluntary movement (reaching, walking, writing, eating); "
            "most disabling symptom; leads to functional disability despite normal resting tone. "
            "Stimulus-sensitive myoclonus: triggered by touch, sound, photic stimulation. "
            "Irregular, asynchronous bilateral (sometimes unilateral) jerks; arms > legs > face. "
            "Distinguishing feature from cortical tremor: myoclonus is irregular (non-rhythmic) "
            "vs. tremor (rhythmic 4–12 Hz); jerk-locked EEG back-averaging confirms cortical origin."
        ),
        "eeg_correlate": (
            "Giant SSEPs (N20 >20 µV bilateral, 10× normal); enhanced C-reflex (long-loop reflex, "
            "latency 45–60 ms); jerk-locked EEG back-averaging: cortical discharge at C3/C4 "
            "preceding myoclonus by 15–25 ms; GPSW on EEG; PPR in photosensitive subtypes."
        ),
        "clinical_tip": (
            "Quantify with UNIFIED MYOCLONUS RATING SCALE (UMRS) at each visit — tracks "
            "functional impact of action myoclonus. Essential for treatment response monitoring. "
            "Piracetam (16–24 g/day) selectively reduces cortical myoclonus in ULD — trial "
            "before escalating immunosuppression."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizure (GTCS)",
        "freq_pct": 90,
        "duration_sec": "60–180 s (postictal confusion 5–30 min)",
        "description": (
            "GTCS present in 90%+ of PME and often the first recognised seizure type (myoclonus "
            "pre-dates GTCS by months to years, recognised only retrospectively). Typically "
            "morning-predominant in ULD/Lafora (sleep deprivation + cortical arousal triggers). "
            "Nocturnal GTCS common in MERRF. Myoclonic jerk clusters frequently herald GTCS "
            "('myoclonic GTCS' — brief myoclonus builds to sustained generalised convulsion). "
            "SUDEP risk: highest in drug-resistant PME with frequent nocturnal GTCS — document "
            "nocturnal seizure frequency and prescribe bedside seizure alarm."
        ),
        "eeg_correlate": (
            "GPSW 3–5 Hz recruiting rhythm evolving to generalised polyspike-polyspike-wave; "
            "progressive amplitude buildup; post-ictal generalised EEG suppression (PGES) "
            "correlated with SUDEP risk — document PGES duration."
        ),
        "clinical_tip": (
            "Valproate Level A for GTCS in ULD/Lafora (AVOID in MERRF). Levetiracetam adjunct "
            "Level B. Seizure diary mandatory — nocturnal GTCS often unreported unless bed partner "
            "observes. Emfit/Embrace2 alarm for unwitnessed nocturnal GTCS."
        ),
    },
    {
        "type": "Progressive Cerebellar Ataxia — Gait/Balance Deficit",
        "freq_pct": 70,
        "duration_sec": "Persistent (progressive neurological sign, not episodic seizure)",
        "description": (
            "Progressive cerebellar ataxia is a core non-seizure feature of PME affecting 70%+ "
            "of patients across all subtypes. Purkinje cell degeneration (CSTB/cathepsin-mediated "
            "in ULD; energy failure in MERRF; lysosomal accumulation in NCL) → truncal ataxia + "
            "limb dysmetria + dysarthria + dysphagia. Falls risk: major source of morbidity; "
            "hip fractures + head injury in late-stage PME. Ataxia progression correlates with "
            "disease stage and is partially independent of seizure control — patients can have "
            "well-controlled seizures but worsening ataxia. Distinguish from myoclonus-induced "
            "gait disturbance (overlap common — combined myoclonus + ataxia = worst functional "
            "outcome; needs MDT physiotherapy + OT)."
        ),
        "eeg_correlate": (
            "No specific EEG signature — cerebellar ataxia is a clinical/neuroimaging finding. "
            "MRI: progressive cerebellar vermis + hemispheric atrophy on sequential scans "
            "(volumetric MRI annually in PME follow-up)."
        ),
        "clinical_tip": (
            "Falls risk assessment: Berg Balance Scale + Dynamic Gait Index at every visit. "
            "OT for adaptive equipment (frame, rails). Physiotherapy: vestibular exercises, "
            "coordination training. Baclofen/tizanidine for spasticity if present. "
            "Feeding assessment + SALT referral for dysphagia management."
        ),
    },
    {
        "type": "Absence-Like / Myoclonic Absence Seizures",
        "freq_pct": 35,
        "duration_sec": "5–30 s",
        "description": (
            "Absence-like events in PME differ from typical childhood absence epilepsy (CAE): "
            "associated with rhythmic myoclonic jerks during the absence (myoclonic absence); "
            "EEG: GPSW 3 Hz but with superimposed polyspike component; consciousness impairment "
            "variable (may retain partial awareness). Photosensitive absence clusters particularly "
            "prominent in Lafora (occipital GPSW trigger). Must be distinguished from complex "
            "partial seizures (temporal) — PME absences have diffuse EEG onset vs. focal "
            "temporal onset. Hyperventilation-provoked events less prominent than in CAE."
        ),
        "eeg_correlate": (
            "3 Hz GPSW with polyspike component; bilateral symmetrical onset; often with "
            "superimposed photoparoxysmal activation; background slowing between events "
            "(distinguishes PME absence from CAE normal background)."
        ),
        "clinical_tip": (
            "Valproate effective for myoclonic absences. Clonazepam adjunct. NEVER prescribe "
            "ethosuximide alone (insufficient for GTCS + myoclonus suppression). "
            "Video-EEG helpful to characterise absence semiology and confirm cortical origin "
            "of associated myoclonus."
        ),
    },
]

# ─── Triggers ────────────────────────────────────────────────────────────────

_TRIGGERS = [
    {"trigger": "Action / Intentional Movement", "pct": 85, "mechanism": "Action myoclonus: cortical hyperexcitability activated by voluntary motor commands — most disabling trigger in PME.", "management": "Piracetam (16–24 g/day) + clonazepam + OT adaptive devices; avoid fatigue; structured movement therapy."},
    {"trigger": "Photic Stimulation (PPR)", "pct": 65, "mechanism": "Photoparoxysmal response in 65–80% ULD/Lafora; occipital cortex hyperexcitability; flickering light (10–25 Hz optimal range).", "management": "Photosensitivity precautions: polarised glasses, avoid TV close-up, avoid disco lighting; AED photosensitivity suppression (VPA/LEV/CLN best)."},
    {"trigger": "Sleep Deprivation", "pct": 80, "mechanism": "Reduces cortical arousal threshold; ULD/Lafora morning predominance; sleep-deprived cortex lowers GTCS threshold.", "management": "Strict sleep hygiene (7–9h); avoid shift work; melatonin 0.5–2 mg if insomnia; AED timing adjustment for morning coverage."},
    {"trigger": "Stress / Emotional Arousal", "pct": 70, "mechanism": "Limbic-cortical hyperexcitability cascade; HPA axis → cortisol spike → GABA receptor downregulation.", "management": "Psychological support: CBT for anxiety; mindfulness; avoid high-stakes exam periods without AED adjustment."},
    {"trigger": "Missed AED Dose", "pct": 55, "mechanism": "Sub-therapeutic plasma levels → sudden reduction in GABA potentiation or AMPA antagonism → seizure breakthrough.", "management": "Compliance aids (pill organiser, phone alarm); written dosing plan; never advise abrupt discontinuation."},
    {"trigger": "Fever / Systemic Illness", "pct": 45, "mechanism": "Fever raises metabolic demand → energy crisis in mitochondrially compromised neurons (especially MERRF); inflammatory cytokines lower seizure threshold.", "management": "Early antipyretics (paracetamol — avoid aspirin in mitochondrial disease); seizure rescue medication at home (buccal midazolam); MERRF: hospitalise early with fever."},
    {"trigger": "Physical Fatigue / Overexertion", "pct": 50, "mechanism": "Muscle + cortical fatigue worsens action myoclonus; aerobic exercise depletes NAD+ in MERRF (mitochondrial insufficiency).", "management": "Structured low-impact exercise programme; energy conservation techniques (OT); MERRF: avoid anaerobic exercise; graded activity pacing."},
    {"trigger": "Tactile / Auditory Startle", "pct": 40, "mechanism": "Reflex cortical myoclonus: sudden sensory input activates reticulospinal or corticospinal hyperexcitable pathways; enhanced C-reflex latency 45–60 ms.", "management": "Reduce environmental startle stimuli (loud alarms); MERRF: hearing aid if sensorineural hearing loss; home environment adaptation (soft flooring, grab rails)."},
]

# ─── Treatments ──────────────────────────────────────────────────────────────

_TREATMENTS = [
    {
        "drug": "Valproate (Sodium Valproate / VPA)",
        "evidence": "Level A (ULD/Lafora/NCL/Sialidosis)",
        "contraindication_note": "ABSOLUTE CONTRAINDICATION in MERRF — mitochondrial hepatotoxicity",
        "dose_adult": "1000–3000 mg/day in 2 divided doses (target TDM 50–100 µg/mL)",
        "dose_paed": "20–40 mg/kg/day; max 60 mg/kg/day with monitoring",
        "moa": (
            "Multi-mechanism: (1) Na-channel inactivation (frequency-dependent); "
            "(2) GABA-T inhibition → increased synaptic GABA; (3) T-type Ca²⁺ channel blockade "
            "(absence mechanism); (4) HDAC inhibition (neuroprotective in ULD). "
            "Broad-spectrum: effective for GTCS, myoclonus, and absence. "
            "MERRF contraindication: VPA inhibits beta-oxidation + carnitine uptake → "
            "accumulation of toxic acyl-CoA intermediates in mitochondria → hepatotoxicity."
        ),
        "efficacy": "GTCS: 60–80% reduction. Myoclonus: 40–60% reduction. Gold standard for non-MERRF PME.",
        "safety": "REMS required (teratogenicity: neural tube defects/FASD). Weight gain. Tremor. Alopecia. Polycystic ovarian syndrome (PCOS) in females. Hepatotoxicity (mitochondria risk).",
        "monitoring": "REMS enrolment. TDM 50–100 µg/mL (free fraction if hypoalbuminaemia). LFTs 3-monthly. Weight monthly. PCOS screen annually (females). Folic acid 5 mg pre-conception.",
        "evidence_ref": "ILAE Level A; Genton 2009 Epilepsia (VPA for PME); FDA VPA REMS 2013",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence": "Level B (all PME subtypes including MERRF)",
        "contraindication_note": "Safe in mitochondrial disease — preferred MERRF AED",
        "dose_adult": "1000–4000 mg/day in 2 divided doses",
        "dose_paed": "20–60 mg/kg/day",
        "moa": (
            "Binds SV2A (synaptic vesicle glycoprotein 2A) — reduces release-ready synaptic "
            "vesicle pool → pre-synaptic glutamate release reduction. Also modulates N-type "
            "Ca²⁺ channels + GABA-A receptor function. SV2A highly expressed in cortical "
            "interneurons — mechanism for myoclonus suppression distinct from VPA."
        ),
        "efficacy": "GTCS: 50–70% reduction. Myoclonus: 35–55% reduction. Adjunct to VPA in non-MERRF; primary AED in MERRF.",
        "safety": "Behavioural side effects: irritability (5–10%), aggression (rare), depression. PHQ-9/GAD-7 monitoring essential. Renal dose adjustment (creatinine clearance <80).",
        "monitoring": "PHQ-9/GAD-7 4-weekly × 8 weeks then 3-monthly. Renal function annually. No plasma TDM required routinely.",
        "evidence_ref": "ILAE Level B; multiple RCTs in PME syndromes; Canevini 2010 Epilepsia",
    },
    {
        "drug": "Piracetam (high-dose)",
        "evidence": "Level A (ULD specifically — Unverricht-Lundborg Disease)",
        "contraindication_note": "Not recommended for Lafora/MERRF/NCL (no evidence; ULD-specific)",
        "dose_adult": "16–24 g/day in 2–4 divided doses (doses up to 45 g/day used in severe ULD)",
        "dose_paed": "Not established; adult evidence only",
        "moa": (
            "Piracetam is a cyclic derivative of GABA; does NOT interact with GABA receptors "
            "directly. Primary mechanism in ULD: (1) restores membrane fluidity of neuronal "
            "phospholipid bilayer (interacts with phosphatidylserine head groups); (2) enhances "
            "AMPA receptor-mediated glutamatergic transmission in a dose-dependent manner — "
            "paradoxically, this reduces cortical hyperexcitability in ULD via mechanisms "
            "related to interneuron facilitation; (3) reduces platelet aggregation "
            "(haemorrheological effect). Myoclonus-specific mechanism: may normalise CSTB-null "
            "cortical interneuron network via membrane stabilisation."
        ),
        "efficacy": "Cortical myoclonus reduction >70% at doses 16–24 g/day in ULD (Genton 2009 Level A). Functional improvement in UMRS score. No effect on GTCS.",
        "safety": "Well tolerated at standard doses. High doses (>24 g/day): hyperkinesia, agitation, insomnia. No hepatotoxicity. No drug interactions. Renal elimination — dose adjust CKD.",
        "monitoring": "UMRS (Unified Myoclonus Rating Scale) at each visit. Renal function annually. No TDM.",
        "evidence_ref": "Genton P 2009 Epilepsia (systematic review Level A); Brown P 1993 Arch Neurol",
    },
    {
        "drug": "Perampanel (AMPA Antagonist)",
        "evidence": "Level A add-on (Lafora Disease; all PME with refractory myoclonus)",
        "contraindication_note": "REMS mandatory; psychiatric monitoring required",
        "dose_adult": "4–12 mg/day (once daily at bedtime; start 2 mg/day, titrate every 2 weeks)",
        "dose_paed": "≥12 years: 2 mg/day titration as adult",
        "moa": (
            "Selective non-competitive AMPA (α-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid) "
            "receptor antagonist. In Lafora Disease, AMPA receptor hyperactivation is a primary "
            "mechanism of cortical myoclonus (Lafora bodies disrupt AMPA receptor recycling → "
            "sustained AMPA activation). Perampanel blocks AMPA post-synaptic Ca²⁺ influx → "
            "reduces cortical hyperexcitability specifically relevant to Lafora pathophysiology. "
            "AMPA antagonism also reduces secondary generalisation of cortical discharge → GTCS "
            "reduction. Half-life 105 hours (once-daily dosing)."
        ),
        "efficacy": "Lafora: myoclonus reduction 55–75%. General PME: GTCS reduction 50–60%. One of few agents with Lafora-specific mechanistic rationale.",
        "safety": "REMS-gated. Psychiatric: aggression/irritability/suicidal ideation in 1–2%. CNZ: dizziness, somnolence. Enzyme inducer interaction (CBZ reduces PER level 50% — avoid in PME).",
        "monitoring": "REMS enrolment + patient/caregiver REMS education. PHQ-9/GAD-7 4-weekly × 8 weeks then 3-monthly. Plasma level if enzyme inducer co-prescribed.",
        "evidence_ref": "FDA REMS 2012; Villeneuve 2015 Epilepsia (Lafora + PER); Jiang 2018 Epilepsia",
    },
    {
        "drug": "Clonazepam (CLN)",
        "evidence": "Level B adjunct (all PME subtypes — myoclonus-specific)",
        "contraindication_note": "Tolerance risk with chronic use; taper slowly if discontinuing",
        "dose_adult": "0.5–4 mg/day (low dose at night preferred to reduce daytime sedation)",
        "dose_paed": "0.01–0.1 mg/kg/day",
        "moa": (
            "Positive allosteric modulator of GABA-A receptors (benzodiazepine site, γ2 subunit). "
            "Highly selective for myoclonus suppression: CLN has high affinity for GABA-A receptors "
            "containing α1/α2 subunits in cortical + spinal cord networks mediating cortical reflex "
            "myoclonus. The C-reflex (long-loop cortical myoclonus pathway) is specifically "
            "suppressed by CLN via GABA potentiation at multiple synaptic levels. "
            "Tolerance: receptor desensitisation within 6–12 months at fixed dose — dose escalation "
            "or drug holidays may be required."
        ),
        "efficacy": "Myoclonus: 40–60% suppression acutely; tolerance may reduce to 25–35% chronically. Not effective for GTCS alone — combine with VPA/LEV.",
        "safety": "Sedation, ataxia-worsening (PME already ataxic — increases fall risk). Dependence. Respiratory depression at high doses. Paradoxical agitation in children.",
        "monitoring": "Tolerance assessment 6-monthly (UMRS). Ataxia worsening monitoring (Berg Balance Scale). If tolerance: structured drug holiday (hospitalise). Taper slowly — never abrupt withdrawal.",
        "evidence_ref": "ILAE Level B (myoclonus); multiple open-label PME series; Obeso 1989 Neurology",
    },
    {
        "drug": "Zonisamide",
        "evidence": "Level B (MERRF and mitochondrial-safe PME adjunct)",
        "contraindication_note": "Sulfonamide allergy contraindication; monitor renal stones",
        "dose_adult": "200–400 mg/day in 1–2 divided doses",
        "dose_paed": "4–8 mg/kg/day",
        "moa": (
            "Dual mechanism: (1) Na-channel inactivation (frequency-dependent block of sustained "
            "high-frequency firing); (2) carbonic anhydrase inhibition (reduces intracellular pH → "
            "reduces Na/H exchanger activity → secondary anticonvulsant effect). "
            "Safe in mitochondrial disease (no mitochondrial membrane effects). "
            "T-type Ca²⁺ channel blockade adds absence/myoclonus component. "
            "Neuroprotective effects in Parkinson's disease models (dopamine-related) — "
            "investigational relevance to PME neurodegeneration pathway."
        ),
        "efficacy": "GTCS adjunct 40–55%. Myoclonus modest 25–35%. Preferred choice in MERRF when VPA contraindicated and LEV insufficient alone.",
        "safety": "Nephrolithiasis (4–6% — kidney stones: hydrate >2L/day). Oligohidrosis/hyperthermia (children — monitor body temperature). Anorexia/weight loss (3–5 kg). Cognitive dulling.",
        "monitoring": "Renal function + urinalysis 6-monthly. Temperature monitoring in children. Weight monthly. TDM: target 10–40 µg/mL (correlates with efficacy).",
        "evidence_ref": "ILAE Level B; Kothare 2004 J Child Neurol; Villeneuve 1996 Epilepsia",
    },
    {
        "drug": "Clobazam (CLB)",
        "evidence": "Level B adjunct (all PME subtypes — GTCS + myoclonus)",
        "contraindication_note": "Less myoclonus-specific than clonazepam; tolerance risk similar",
        "dose_adult": "10–40 mg/day in 1–2 divided doses (bedtime dosing preferred)",
        "dose_paed": "0.1–1 mg/kg/day; max 40 mg/day",
        "moa": (
            "1,5-benzodiazepine (distinguished from 1,4-benzodiazepines like CLN/DZP). "
            "GABA-A positive allosteric modulator with preferential affinity for α2 subunit "
            "(less sedating than CLN due to reduced α1 binding; α1 mediates sedation). "
            "Slower receptor desensitisation than CLN — less tolerance at equivalent dose. "
            "Active metabolite N-desmethylclobazam (NCLB) contributes 20–50% of activity "
            "(CYP2C19 polymorphism: poor metabolisers have 5× higher NCLB levels → toxicity)."
        ),
        "efficacy": "GTCS: 45–55% reduction. Myoclonus: 30–45% reduction. Fewer behavioural adverse effects than CLN in practice.",
        "safety": "Sedation (less than CLN). Ataxia. Tolerance (slower than CLN). CYP2C19 interaction: omeprazole, fluoxetine increase CLB levels. Dependence.",
        "monitoring": "CYP2C19 genotyping if poor/rapid metaboliser suspected. Tolerance assessment 6-monthly. Taper ≥6 weeks — avoid abrupt withdrawal.",
        "evidence_ref": "ILAE Level B; Schmidt 2000 Seizure; Canadian PME consortium data",
    },
    {
        "drug": "Mitochondrial Supplements (CoQ10 / L-Carnitine / Riboflavin) — MERRF Protocol",
        "evidence": "Expert consensus / Level C (MERRF and other mitochondrial PME)",
        "contraindication_note": "Not applicable as anti-seizure therapy — neuroprotective supplement protocol",
        "dose_adult": "CoQ10: 300–600 mg/day; L-carnitine: 50 mg/kg/day (max 3 g/day); Riboflavin (B2): 400 mg/day",
        "dose_paed": "CoQ10: 5–10 mg/kg/day; L-carnitine: 50–100 mg/kg/day",
        "moa": (
            "CoQ10 (ubiquinone): electron carrier in the mitochondrial respiratory chain (complex I→III "
            "transfer); CoQ10 deficiency in MERRF → impaired ATP synthesis; supplementation "
            "increases residual complex I/III activity in heteroplasmic MT-TK cells. "
            "L-carnitine: essential cofactor for fatty acid beta-oxidation; depleted by VPA "
            "(AVOID VPA but replete carnitine if prior VPA exposure). "
            "Riboflavin: precursor to FAD (complex I/II cofactor); enhances residual mitochondrial "
            "respiratory chain function. Combined protocol (CoenzymeQ10 + L-carnitine + riboflavin): "
            "standard of care at major mitochondrial disease centres (London/Toronto/Boston protocols)."
        ),
        "efficacy": "No RCT data (rare disease barrier). Open-label: 30–50% of MERRF patients report improved energy/exercise tolerance; variable seizure benefit; may slow neurodegeneration.",
        "safety": "Excellent tolerability. CoQ10: GI (nausea, diarrhoea at high dose). Riboflavin: yellow urine (harmless). L-carnitine: fishy body odour at high doses.",
        "monitoring": "Plasma CoQ10 levels (target >2.5 µg/mL). Lactate/pyruvate ratio (target <15:1). Liver function (carnitine hepatic metabolism). Annual cardiac echo (MERRF cardiomyopathy).",
        "evidence_ref": "DiMauro 2004 GeneReviews; London Mitochondrial Disease Centre Protocol; MITOCON guidelines",
    },
]

# ─── Lifecycle ────────────────────────────────────────────────────────────────

_LIFECYCLE = [
    {
        "window": "Childhood / Preadolescence (6–12 years)",
        "age_range": "6–12 y",
        "key_events": (
            "ULD onset peak: 6–15 y. NCL (CLN3/CLN2) onset: 2–10 y — visual failure pre-dates "
            "seizures; early ophthalmology + ERG referral essential. MERRF: onset variable "
            "(any age — maternal family history mandatory). Initial myoclonus misdiagnosed as "
            "'clumsiness', 'ticks', or 'anxiety'; myoclonus precedes GTCS by months to years. "
            "School impact: handwriting deterioration (action myoclonus), falls, learning difficulties. "
            "Full metabolic + genetic workup at first presentation: LFT, lactate, CPK, urinary "
            "oligosaccharides, mtDNA panel, skin biopsy if NCL suspected."
        ),
        "focus": (
            "PME gene panel (CSTB/EPM2A/NHLRC1/MT-TK/CLN1-14/NEU1). Ophthalmology ERG (NCL). "
            "Muscle biopsy if MERRF suspected (ragged red fibers + COX-stain). School IHCP. "
            "Physiotherapy for balance. Neuropsychological assessment. Avoid misdiagnosis as JME "
            "(PME has progressive course — distinguish from JME by neurological deterioration)."
        ),
    },
    {
        "window": "Adolescence (13–18 years)",
        "age_range": "13–18 y",
        "key_events": (
            "Lafora onset peak: 12–17 y — occipital seizures + rapid cognitive decline + psychiatric "
            "features. ULD: myoclonus worsens with exam stress/sleep deprivation. Social impact: "
            "driving licence foregone (12-month seizure-free required); peer isolation; identity "
            "issues with progressive disability. Photosensitivity precautions for school/social: "
            "polarised glasses, avoiding disco/concert lighting, screen brightness settings. "
            "Genetic counselling: explain inheritance patterns (AR vs. maternal) to adolescent "
            "and family in age-appropriate language."
        ),
        "focus": (
            "Driving counselling (cannot drive until 12 months seizure-free; PME unlikely to achieve "
            "this). Photosensitivity environmental assessment (school, social). Transition "
            "planning to adult neurology. Skin biopsy if Lafora not yet confirmed (sweat gland "
            "PAS staining). Lafora: rapid decline — discuss prognosis honestly + palliative care "
            "pathway initiation. Psychiatric screening (Lafora: depression, psychosis early)."
        ),
    },
    {
        "window": "Young Adult — Disease Progression (19–35 years)",
        "age_range": "19–35 y",
        "key_events": (
            "ULD: plateaus in third decade — progressive ataxia + myoclonus; most ULD adults have "
            "partial functional independence. Lafora: fatal in 90% within 10 years of onset (late "
            "teens/early twenties). MERRF: intercurrent illness + fever → metabolic crisis; "
            "cardiac monitoring mandatory (cardiomyopathy). NCL (CLN3): blind by early twenties; "
            "wheelchair-bound by mid-twenties. Childbearing counselling: VPA teratogenicity "
            "(NTD risk 3–5% — REMS); folic acid 5 mg/day; genetic counselling (AR/mitochondrial)."
        ),
        "focus": (
            "Fertility/contraception counselling (VPA/REMS). Folic acid supplementation. "
            "MERRF: 6-monthly cardiac echo + ECG (cardiomyopathy surveillance). Lafora: palliative "
            "care + advance care planning. NCL: low-stimulation environment + guide dog referral. "
            "ULD: rehabilitation + vocational assessment; most can maintain supported work. "
            "Clinical trial referral (ULD: NAC trial; NCL: gene therapy trials)."
        ),
    },
    {
        "window": "Mid-Adult — Disability Management (36–55 years)",
        "age_range": "36–55 y",
        "key_events": (
            "ULD: stable or slow progression; major disability source = cerebellar ataxia + "
            "falls risk > myoclonus. Bone density: enzyme-inducing AEDs rare in PME (VPA/LEV "
            "used); monitor vitamin D/Ca²⁺ supplementation. MERRF: mitochondrial burden "
            "accumulates → increasing fatigue, hearing loss progression (hearing aid), "
            "ophthalmoplegia. POLG vs. MERRF differentiation if not yet confirmed (POLG: "
            "progressive liver disease + neuropathy + epilepsy — PEO3 phenotype). "
            "Cognitive decline management: neuropsychological monitoring 2-yearly."
        ),
        "focus": (
            "Bone densitometry DEXA if long-term AED + low vitamin D. Rehabilitation MDT "
            "(physiotherapy + OT + SALT + dietitian). MERRF hearing aid. Cognitive support "
            "plan. Power wheelchair assessment if ataxia severe. End-stage NCL: nursing care "
            "planning. ULD genetic family screening for siblings."
        ),
    },
    {
        "window": "Older Adult — Advanced Disease Care (56+ years)",
        "age_range": ">56 y",
        "key_events": (
            "ULD only: rare survivors to 60–70 y documented (benign end of PME spectrum). "
            "Advanced ataxia + cognitive mild-moderate; most use walking frames or wheelchair. "
            "Polypharmacy: renal dosing adjustments for LEV/ZNS (GFR decline); drug-drug "
            "interactions increase (add cardiac medications). Falls: major mortality source "
            "(hip fractures). SUDEP surveillance: nocturnal GTCS in older PME + cardiorespiratory "
            "comorbidities → highest SUDEP risk. Advance care directives should be in place."
        ),
        "focus": (
            "Polypharmacy review (deprescribe unnecessary agents). Falls prevention environment "
            "assessment. Seizure alarm (Emfit). Advance care directive. Renal dose adjustment "
            "LEV/ZNS. Caregiver support assessment (respite care). Palliative neurology involvement."
        ),
    },
    {
        "window": "Mitochondrial Family Screening (MERRF — Maternal Lineage)",
        "age_range": "All ages in maternal family",
        "key_events": (
            "MERRF: maternal inheritance — sisters, daughters, maternal aunts, and maternal cousins "
            "share mitochondrial genome. Heteroplasmy variable: 80%+ mutant mtDNA = severe MERRF; "
            "30–60% mutant = mild/subclinical. Screening: blood mtDNA heteroplasmy + audiometry "
            "+ EMG + MRI brain + ophthalmology. Even mildly affected maternal relatives require "
            "genetic counselling re: offspring risk. Oocyte donation available for females "
            "wishing to avoid mtDNA transmission (mitochondrial replacement therapy — "
            "licensed in UK; pending in Canada/US)."
        ),
        "focus": (
            "Maternal family cascade testing: mtDNA heteroplasmy levels in blood (peripheral "
            "leukocytes). Audiological assessment (sensorineural hearing loss early biomarker). "
            "Avoid valproate in ALL family members until MERRF excluded. Mitochondrial disease "
            "specialist referral. Fertility counselling for reproductive-age females."
        ),
    },
]

# ─── AED Monitoring ───────────────────────────────────────────────────────────

_AED_MONITORING = [
    {
        "item": "VPA: REMS Enrolment + TDM + LFTs + Weight + PCOS (Females) + MERRF Exclusion",
        "frequency": "REMS at initiation · TDM 3-monthly · LFTs 3-monthly · Weight monthly · PCOS annually",
        "rationale": (
            "VPA REMS: FDA-mandated prescriber + patient education for teratogenicity (NTD 3–5%; "
            "neurodevelopmental risk). TDM target: 50–100 µg/mL (free fraction in hypoalbuminaemia). "
            "LFTs: hepatotoxicity rare in adults but highest risk in children <2y + metabolic disease. "
            "PCOS: long-term VPA increases testosterone + polycystic ovaries in female PME (10–15%). "
            "MERRF exclusion: CONFIRM MT-TK mutation negative before prescribing VPA — "
            "even single VPA dose in MERRF can precipitate acute liver failure."
        ),
    },
    {
        "item": "LEV: PHQ-9 / GAD-7 Mood Monitoring + Renal Function",
        "frequency": "PHQ-9/GAD-7 4-weekly × 8 weeks then 3-monthly · eGFR annually",
        "rationale": (
            "LEV behavioural adverse effects: irritability/aggression in 5–10%; depression/anxiety in "
            "3–5%. PME patients already carry psychosocial burden of progressive disease — baseline "
            "PHQ-9/GAD-7 before starting LEV mandatory for meaningful monitoring. eGFR: LEV 66% "
            "renally excreted unchanged — dose halve if eGFR <30 mL/min. Standard LEV dose may be "
            "toxic in renal impairment (common in AMRF/SCARB2 with concurrent renal disease)."
        ),
    },
    {
        "item": "CLN/CLB: Tolerance Assessment (UMRS) + Ataxia Monitoring (Berg Balance Scale)",
        "frequency": "UMRS 3-monthly · Berg Balance Scale 6-monthly · tolerance reassessment 6-monthly",
        "rationale": (
            "Benzodiazepine tolerance: receptor desensitisation within 6–12 months → reduced "
            "myoclonus suppression effect (UMRS score deterioration). PME ataxia is already present "
            "— CLN/CLB worsen ataxia + falls risk proportional to dose. Falls: major morbidity "
            "in PME (hip fracture risk 3× higher than age-matched controls). If tolerance: "
            "structured drug holiday (inpatient benzodiazepine taper) or switch to PER/LEV "
            "intensification."
        ),
    },
    {
        "item": "PER: REMS Enrolment + Psychiatric Monitoring (PHQ-9 / Aggression Scale)",
        "frequency": "REMS at prescription · PHQ-9/GAD-7/aggression 4-weekly × 8w then 3-monthly",
        "rationale": (
            "PER REMS: FDA-mandatory prescriber + pharmacy + patient enrolment. Psychiatric ADRs: "
            "aggression/hostility in 1.9% (higher in young adults). Suicidal ideation: class effect "
            "(all AEDs FDA label — actual PER incidence ~0.4%). PME patients in late Lafora have "
            "intrinsic psychiatric features (psychosis, depression) — baseline and follow-up "
            "neuropsychiatric assessment mandatory. Enzyme inducer avoidance: CBZ reduces PER levels "
            "50% (contraindicated in PME — also worsens myoclonus independently)."
        ),
    },
]

# ─── Standards ────────────────────────────────────────────────────────────────

_STANDARDS = [
    {"standard": "ILAE 2022 Classification of PME", "relevance": "Updated genetic classification: 14 recognised PME gene categories; ULD/Lafora/MERRF/NCL primary subtypes"},
    {"standard": "NICE NG217 (2022)", "relevance": "Epilepsy clinical guideline — AED choice, monitoring, surgery; PME-specific guidance on VPA use and referral pathways"},
    {"standard": "Minassian BA 2019 Brain (Lafora)", "relevance": "Lafora disease pathophysiology + treatment (VPA/CLB/PER); perampanel AMPA mechanism for Lafora myoclonus"},
    {"standard": "FDA VPA REMS (2013)", "relevance": "Mandatory REMS program: prescriber/patient education for valproate teratogenicity and neurodevelopmental risks"},
    {"standard": "FDA Perampanel REMS (2012)", "relevance": "Mandatory REMS: psychiatric adverse effects (aggression/suicidal ideation); prescriber/patient/pharmacy enrolment"},
    {"standard": "MITOCON / Mito UK Mitochondrial Guidelines (2015)", "relevance": "MERRF management: VPA contraindication; CoQ10/carnitine/riboflavin protocol; maternal family screening"},
]

# ─── Thresholds ───────────────────────────────────────────────────────────────

_THRESHOLDS = [
    {"threshold": "Drug-Resistant Epilepsy (DRE) Definition", "value": "≥2 adequate AED trials failed (ILAE 2010)"},
    {"threshold": "VPA Therapeutic Drug Monitoring", "value": "50–100 µg/mL (trough, 12h post-dose); free fraction 5–15 µg/mL if hypoalbuminaemia"},
    {"threshold": "Clonazepam Plasma Range", "value": "0.02–0.08 mg/L (not routinely measured; use clinical tolerance assessment instead)"},
    {"threshold": "Piracetam Target Dose (ULD)", "value": "16–24 g/day in adults; up to 45 g/day in refractory ULD cortical myoclonus"},
    {"threshold": "Valproate CONTRAINDICATION (MERRF)", "value": "Any dose contraindicated in confirmed/suspected MERRF (MT-TK mutation) — fatal hepatotoxicity risk"},
    {"threshold": "Driving Restriction (Canada/UK/Australia)", "value": "12 months seizure-free (rarely achievable in PME — most patients permanently restricted)"},
    {"threshold": "Pre-Conception Folic Acid (VPA patients)", "value": "5 mg/day ≥3 months before conception (NICE NG217)"},
    {"threshold": "MERRF Lactate: Seizure Risk Threshold", "value": "CSF lactate >2.1 mmol/L or plasma lactate/pyruvate >20:1 indicates active mitochondrial dysfunction"},
]

# ─── Concepts ─────────────────────────────────────────────────────────────────

_CONCEPTS = [
    {"term": "Progressive Myoclonic Epilepsy (PME)", "definition": "A clinically heterogeneous group of rare inherited epilepsies defined by the triad of: (1) cortical myoclonus — action-sensitive, stimulus-sensitive; (2) GTCS; (3) progressive neurological deterioration — cerebellar ataxia, cognitive decline, and/or dementia. Genetically diverse: 14+ genetic causes identified (ILAE 2022). Must be distinguished from non-progressive myoclonic epilepsies (JME, MAE) by the mandatory progressive course."},
    {"term": "Cortical Myoclonus", "definition": "Brief (<1 s), involuntary, irregular muscle jerks arising from hyperexcitable sensorimotor cortex. Pathognomonic for PME. Neurophysiological signature: (1) giant SSEPs (N20 >20 µV — 10× normal); (2) enhanced long-loop C-reflex (cortex → spinal cord, latency 45–60 ms); (3) jerk-locked EEG back-averaging shows cortical discharge at C3/C4/Fz 15–25 ms before the jerk. Distinguishing from tremor: myoclonus is irregular/arrhythmic; tremor is rhythmic (4–12 Hz)."},
    {"term": "Action Myoclonus", "definition": "Cortical myoclonus triggered by voluntary intentional movement — the most functionally disabling manifestation of PME. Handwriting, reaching for objects, walking, eating all trigger jerks. The Unified Myoclonus Rating Scale (UMRS) quantifies action myoclonus. First described by Lance & Adams 1963 in post-anoxic myoclonus; cortical origin in PME confirmed by jerk-locked back-averaging."},
    {"term": "Photoparoxysmal Response (PPR)", "definition": "EEG response to photic stimulation (IPS 1–30 Hz) consisting of generalised polyspike-and-slow-wave discharge. Graded I–IV (Waltz 1992): Grade I = restricted to posterior; Grade II = restricted to stimulus; Grade III = bilateral posterior + anterior; Grade IV = self-sustaining after end of stimulus. PPR present in 65–80% of ULD and Lafora Disease — a useful clinical diagnostic marker."},
    {"term": "CSTB (Cystatin B) — ULD Gene", "definition": "Cystatin B is an endogenous inhibitor of lysosomal cysteine proteases (cathepsins B, D, H, L). CSTB mutations cause ULD (EPM1). Pathogenic mechanism: unrestrained cathepsin proteolysis → Purkinje cell + cortical interneuron degeneration. The commonest pathogenic variant is a 12-mer dodecamer repeat expansion in the CSTB promoter (90% of European ULD). N-acetylcysteine (NAC) is neuroprotective in Cstb-knockout mice by replenishing glutathione (GSH)."},
    {"term": "Lafora Body", "definition": "Intraneuronal polyglucosan inclusions (insoluble hyperphosphorylated polysaccharide aggregates) — pathognomonic for Lafora Disease. Found in neurons, liver, skeletal muscle, heart, eccrine sweat gland ducts. Diagnosed by skin biopsy: PAS (periodic acid–Schiff) staining of eccrine sweat gland ducts showing round, basophilic, PAS-positive inclusions. Electron microscopy: fibrillary-granular appearance. Lafora bodies cause neuronal death by impairing proteasome/autophagy clearance."},
    {"term": "MERRF (Myoclonic Epilepsy with Ragged Red Fibers)", "definition": "Mitochondrial encephalomyopathy caused by point mutations in mitochondrial tRNA genes (MT-TK most common: m.8344A>G in 80%). Maternally inherited. Ragged red fibers: accumulation of dysfunctional mitochondria in subsarcolemmal space (modified Gomori trichrome stain on muscle biopsy). COX-deficient fibers on cytochrome c oxidase stain. Clinical tetrad: myoclonus + GTCS + cerebellar ataxia + ragged red fibers. Variably: hearing loss, myopathy, dementia, short stature, cardiomyopathy."},
    {"term": "Ragged Red Fibers (RRF)", "definition": "Histological finding on modified Gomori trichrome stain of muscle biopsy: subsarcolemmal accumulations of abnormal mitochondria appearing red/magenta ('ragged' irregular margin). Found in MERRF and other mitochondrial myopathies (MELAS, KSS, CPEO). Electron microscopy confirms mitochondrial ultrastructural abnormalities (cristae disorganisation, paracrystalline inclusions). COX-deficient fibers (blue on COX/SDH dual stain) indicate respiratory chain complex IV deficiency."},
    {"term": "Neuronal Ceroid Lipofuscinosis (NCL / Batten Disease)", "definition": "Family of 14 autosomal recessive lysosomal storage disorders (CLN1–14) defined by accumulation of ceroid-lipofuscin (autofluorescent lipoprotein) in neuronal lysosomes. Features: progressive visual failure → blindness + dementia + myoclonus + seizures. Age of onset varies by subtype (CLN1: infancy; CLN2: 2–4y; CLN3: 5–10y). Diagnosed by enzyme assay (PPT1 for CLN1; TPP1 for CLN2), CLN gene panel, and electron microscopy of skin/conjunctival biopsy."},
    {"term": "Giant SSEPs (Somatosensory Evoked Potentials)", "definition": "Somatosensory evoked potentials with abnormally enlarged cortical N20 amplitude (>20 µV, normal 3–5 µV — 10× normal). Pathognomonic neurophysiological signature of cortical hyperexcitability in PME. Result from disinhibited sensorimotor cortex generating excessive burst-discharge in response to peripheral nerve stimulation. Jerk-locked back-averaging of simultaneous EEG + EMG confirms the cortical origin and timing of myoclonic jerks."},
    {"term": "C-Reflex (Long-Loop Cortical Reflex)", "definition": "Enhanced long-loop transcortical reflex pathway activated by peripheral sensory stimulation. Reflex arc: peripheral nerve → dorsal horn → medial lemniscus → S1 cortex → M1 cortex → corticospinal tract → alpha motor neuron → muscle. In PME: hyperexcitable cortex amplifies this long-loop reflex → enhanced C-reflex at latency 45–60 ms (vs. normal 25–35 ms). Measured by EMG. Distinguishes cortical myoclonus from subcortical (shorter latency) or spinal myoclonus."},
    {"term": "Piracetam Mechanism in ULD", "definition": "Piracetam (cyclic GABA derivative) reduces cortical myoclonus in ULD via membrane fluidity restoration and AMPA receptor modulation. Does not interact with GABA receptors. Level A evidence: Genton 2009 Epilepsia systematic review showed >70% myoclonus reduction at 16–24 g/day. Piracetam does not reduce GTCS. ULD-specific effect — not effective in Lafora/MERRF/NCL. Safe, well tolerated, no drug interactions."},
    {"term": "UMRS (Unified Myoclonus Rating Scale)", "definition": "Standardised rating scale for cortical myoclonus severity and functional impact. 5 subscales: spontaneous myoclonus, action myoclonus (position-sensitive, kinetic, task-specific), stimulus sensitivity (sound, touch, photic), functional disability, and global severity. Essential for treatment response monitoring in PME. Administered by trained neurologist or neuropsychologist. Captures functional impact that VAS pain scales miss."},
    {"term": "Skin Biopsy (Diagnostic — PME)", "definition": "Minimally invasive diagnostic procedure using 3 mm punch biopsy of axillary or abdominal skin. For Lafora Disease: PAS staining of eccrine sweat gland ducts shows Lafora bodies (diagnostic yield 85%). For NCL: electron microscopy of eccrine secretory cells reveals ultrastructural inclusions (GROD/curvilinear/fingerprint profiles — CLN1/2/3 specific). Preferred first-line tissue test (avoids brain/muscle biopsy in children). Axillary site > abdominal for Lafora body yield."},
    {"term": "SUDEP in PME", "definition": "Sudden Unexpected Death in Epilepsy — risk particularly elevated in drug-resistant PME with nocturnal GTCS. PME risk ~1/150 patient-years (higher than general epilepsy population due to progressive neurological compromise + cardiorespiratory co-morbidities). Prevention: optimise AED seizure control, nocturnal seizure alarm (Emfit QM / Embrace2 bracelet), avoid prone sleeping, optimise cardiac monitoring in MERRF (cardiomyopathy). SUDEP counselling mandatory for all PME patients and families."},
]

# ─── References ──────────────────────────────────────────────────────────────

_REFERENCES = [
    "Berkovic SF et al. 1993 Ann Neurol — Baltic/Mediterranean myoclonus (ULD): 100 cases, genotype-phenotype correlation",
    "Minassian BA 2001 Trends Neurosci — Lafora disease: polyglucosan pathophysiology, EPM2A/NHLRC1, perampanel target",
    "DiMauro S & Hirano M 2004 GeneReviews — MERRF: MT-TK mutations, ragged red fibers, VPA contraindication, management",
    "Mole SE et al. 2011 Biochim Biophys Acta — NCL/Batten disease: CLN gene classification, diagnostics, cerliponase alfa",
    "Genton P et al. 2009 Epilepsia — Piracetam for ULD cortical myoclonus: Level A systematic review (16–24 g/day)",
    "Zara F & Guerrini R 2019 Epilepsia — ILAE PME Task Force: genetic classification update, 14 subtypes",
]

# ─── Patient overlay ──────────────────────────────────────────────────────────

_ONSET_AGES = [7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 8, 10, 11, 13, 15, 16, 12, 14, 9,
               11, 13, 7, 8, 10, 14, 16, 13, 17, 12, 9, 11, 15, 8, 14, 12, 10, 13, 16, 11, 9]
_ETIOLOGIES_WEIGHTED = (
    ["ULD (EPM1/CSTB)"] * 35 + ["Lafora Disease (EPM2A)"] * 20 +
    ["MERRF (MT-TK)"] * 15 + ["NCL (Batten/CLN)"] * 20 + ["Sialidosis/Other"] * 10
)
_PRIMARY_SEIZURES = ["Cortical Myoclonus", "GTCS", "Progressive Ataxia Episodes", "Myoclonic Absence"]
_SEIZURE_CONTROL = ["Drug-resistant"] * 28 + ["Partial control"] * 10 + ["Seizure-free"] * 3
_AEDS = ["VPA", "VPA+LEV", "LEV+CLN", "PER+LEV", "PER+VPA", "VPA+CLB", "LEV+ZNS",
         "ZNS+CLN", "VPA+PIR", "LEV+PIR"]
_DISEASE_STAGES = ["Mild"] * 12 + ["Moderate"] * 16 + ["Advanced"] * 10 + ["End-stage"] * 3


def _overlay_patients(rows):
    out = []
    for r in rows:
        s = _seed(r.get("patient_id", "x"))
        et_idx = s % len(_ETIOLOGIES_WEIGHTED)
        sz_idx = (s >> 4) % len(_PRIMARY_SEIZURES)
        ctl_idx = (s >> 8) % len(_SEIZURE_CONTROL)
        aed_idx = (s >> 12) % len(_AEDS)
        oa_idx = (s >> 16) % len(_ONSET_AGES)
        stage_idx = (s >> 20) % len(_DISEASE_STAGES)
        cortical_myoclonus = (s >> 2) % 20 != 0       # ~95%
        photosensitive = (s >> 5) % 3 < 2              # ~65%
        progressive_ataxia = (s >> 7) % 10 < 7         # ~70%
        cognitive_decline = (s >> 9) % 4 < 3           # ~75%
        merrf = _ETIOLOGIES_WEIGHTED[et_idx] == "MERRF (MT-TK)"
        r["etiology"] = _ETIOLOGIES_WEIGHTED[et_idx]
        r["onset_age"] = _ONSET_AGES[oa_idx]
        r["primary_seizure_type"] = _PRIMARY_SEIZURES[sz_idx]
        r["seizure_control"] = _SEIZURE_CONTROL[ctl_idx]
        r["current_aed"] = _AEDS[aed_idx]
        r["disease_stage"] = _DISEASE_STAGES[stage_idx]
        r["cortical_myoclonus"] = cortical_myoclonus
        r["photosensitive"] = photosensitive
        r["progressive_ataxia"] = progressive_ataxia
        r["cognitive_decline"] = cognitive_decline
        r["vpa_contraindicated"] = merrf
        out.append(r)
    return out


# ─── Public API ───────────────────────────────────────────────────────────────

def overview():
    rows = _db_rows("SELECT patient_id, age, gender FROM patients LIMIT 41")
    rows = _overlay_patients(rows)
    n = len(rows)
    drug_resistant = [r for r in rows if r["seizure_control"] == "Drug-resistant"]
    vpa_contraindicated = [r for r in rows if r["vpa_contraindicated"]]
    photosensitive = [r for r in rows if r["photosensitive"]]
    ataxia = [r for r in rows if r["progressive_ataxia"]]
    cogn_decline = [r for r in rows if r["cognitive_decline"]]

    et_dist = {}
    for r in rows:
        et_dist[r["etiology"]] = et_dist.get(r["etiology"], 0) + 1

    stage_dist = {}
    for r in rows:
        stage_dist[r["disease_stage"]] = stage_dist.get(r["disease_stage"], 0) + 1

    return {
        "dashboard": "Progressive Myoclonic Epilepsy (PME)",
        "generated": date.today().isoformat(),
        "total_patients": n,
        "drug_resistant_n": len(drug_resistant),
        "drug_resistant_pct": round(len(drug_resistant) / n * 100),
        "vpa_contraindicated_n": len(vpa_contraindicated),
        "vpa_contraindicated_pct": round(len(vpa_contraindicated) / n * 100),
        "photosensitive_n": len(photosensitive),
        "photosensitive_pct": round(len(photosensitive) / n * 100),
        "progressive_ataxia_n": len(ataxia),
        "progressive_ataxia_pct": round(len(ataxia) / n * 100),
        "cognitive_decline_n": len(cogn_decline),
        "cognitive_decline_pct": round(len(cogn_decline) / n * 100),
        "etiology_distribution": et_dist,
        "disease_stage_distribution": stage_dist,
        "clinical_alerts": [
            "ABSOLUTE CONTRAINDICATION — CBZ/OXC/PHT in ALL PME subtypes: exacerbate cortical "
            "myoclonus (Na-channel blockers worsen myoclonus via cortical disinhibition — high-grade evidence)",
            "ABSOLUTE CONTRAINDICATION — Valproate in MERRF (MT-TK mutation): mitochondrial "
            "hepatotoxicity / Alpers-like acute liver failure; CONFIRM MT-TK exclusion BEFORE VPA",
            "ABSOLUTE CONTRAINDICATION — Vigabatrin in ALL PME: worsens action myoclonus + "
            "irreversible visual field loss (compounds NCL visual failure)",
            "Piracetam (16–24 g/day) is Level A evidence for cortical myoclonus in ULD (EPM1) — "
            "initiate alongside valproate; ULD-specific mechanism",
            "PME is frequently MISDIAGNOSED as JME in early stages (myoclonus + GTCS) — "
            "DISTINGUISH by progressive course + cerebellar ataxia + giant SSEPs + deteriorating EEG background",
            "MERRF: maternal family cascade testing mandatory — confirm MT-TK mutation in "
            "sisters/daughters/maternal aunts; AVOID VPA in entire maternal lineage until exclusion",
        ],
        "subtype_prognosis": {
            "uld_epm1": "Slowest progression — 50-year survival documented; partial functional independence possible",
            "lafora_epm2a": "Rapidly fatal — 90% die within 10 years of onset (late teens/twenties)",
            "merrf": "Variable (heteroplasmy-dependent) — severe: death 30s–40s; mild: functional to 60s",
            "ncl_batten": "CLN2: death 6–8y; CLN3: death 30–40y; CLN1: death by 5y",
            "sialidosis_i": "Slowest-progressing sialidosis — mild course, cognitive relatively spared",
        },
        "references": _REFERENCES,
    }


def breakdown():
    rows = _db_rows("SELECT patient_id, age, gender FROM patients LIMIT 41")
    rows = _overlay_patients(rows)

    patients_out = []
    for r in rows:
        patients_out.append({
            "patient_id": r["patient_id"],
            "age": r["age"],
            "gender": r["gender"],
            "onset_age": r["onset_age"],
            "etiology": r["etiology"],
            "primary_seizure_type": r["primary_seizure_type"],
            "current_aed": r["current_aed"],
            "seizure_control": r["seizure_control"],
            "disease_stage": r["disease_stage"],
            "cortical_myoclonus": r["cortical_myoclonus"],
            "photosensitive": r["photosensitive"],
            "progressive_ataxia": r["progressive_ataxia"],
            "cognitive_decline": r["cognitive_decline"],
            "vpa_contraindicated": r["vpa_contraindicated"],
        })

    return {
        "patients": patients_out,
        "etiology_catalog": _ETIOLOGIES,
        "seizure_types": _SEIZURE_TYPES,
        "triggers": _TRIGGERS,
        "treatments": _TREATMENTS,
        "lifecycle": _LIFECYCLE,
        "aed_monitoring": _AED_MONITORING,
        "standards": _STANDARDS,
        "thresholds": _THRESHOLDS,
    }


def definitions():
    return {
        "concepts": _CONCEPTS,
        "thresholds": _THRESHOLDS,
        "references": _REFERENCES,
        "absolute_contraindications": [
            {
                "drug": "CBZ / OXC / PHT (Sodium Channel Blockers)",
                "contraindicated_in": "ALL PME subtypes",
                "consequence": "Exacerbate cortical myoclonus (paradoxical worsening via cortical disinhibition)",
            },
            {
                "drug": "Valproate (VPA)",
                "contraindicated_in": "MERRF (MT-TK mutation)",
                "consequence": "Mitochondrial hepatotoxicity / Alpers-like acute liver failure — potentially fatal",
            },
            {
                "drug": "Vigabatrin (VGB)",
                "contraindicated_in": "ALL PME subtypes",
                "consequence": "Worsens action myoclonus + irreversible visual field loss (compounds NCL blindness)",
            },
            {
                "drug": "Gabapentin / Pregabalin",
                "contraindicated_in": "ALL PME subtypes",
                "consequence": "Paradoxical myoclonus worsening (alpha-2-delta Ca²⁺ channel modulation increases cortical excitability)",
            },
        ],
    }
