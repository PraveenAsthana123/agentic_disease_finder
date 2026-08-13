"""
DEPDC5 Focal Epilepsy — GATOR1 Complex (DEPDC5-FE / FFEVF / ADNFLE-DEPDC5)
============================================================================
41-patient cohort · DEPDC5 (22q12.3) · GATOR1 complex · mTOR pathway
DEPDC5-related focal epilepsy: heterozygous pathogenic loss-of-function (LOF) variants
in DEPDC5 (Dishevelled, Egl-10, and Pleckstrin domain-containing protein 5, 22q12.3)
are the most common identified genetic cause of familial focal epilepsy, accounting for
~10% of all familial focal epilepsy families and ~1% of population-level focal epilepsy.

DEPDC5 BIOLOGY: DEPDC5 is the scaffold/regulatory subunit of the GATOR1 complex
(GAP activity toward Rags complex 1). GATOR1 = DEPDC5 + NPRL2 + NPRL3 — a trimeric
negative regulator of the mTORC1 pathway. The pathway: amino acid sensing → GATOR2
activates Rag GTPases → GATOR1 (via GAP activity toward RagA/B GTPase) inactivates
Rags when amino acids are low → mTORC1 inhibition. LOF in any GATOR1 subunit
(DEPDC5/NPRL2/NPRL3) → constitutive Rag GTPase activation → unrestricted mTORC1
signalling → hyperactivated cellular growth, protein synthesis, inhibition of
autophagy → focal cortical dysplasia type II (FCD-II) in some patients → focal
hyperexcitability. The mTOR pathway is the same one targeted by Everolimus/Sirolimus
in TSC — making DEPDC5 an mTOR-precision-therapy candidate.

EPILEPSY PHENOTYPE — DEPDC5-FE: Strongly familial (autosomal dominant, ~60-70%
penetrance). Variable foci across and within families — a defining clinical hallmark:
① FFEVF (Familial Focal Epilepsy with Variable Foci) — seizure focus varies between
  family members (frontal/temporal/parietal/occipital); MRI often negative
② ADNFLE (Autosomal Dominant Nocturnal Frontal Lobe Epilepsy) — nocturnal hypermotor
  seizures from frontal focus; can mimic parasomnias; DEPDC5 accounts for ~12% of ADNFLE
③ FMTLE (Familial Mesial Temporal Lobe Epilepsy) — DEPDC5 identified in ~10% of FMTLE
④ Lesional subgroup: FCD type IIa/IIb on MRI (~30% of DEPDC5 cohorts) — important
  for surgical workup
SUDEP RISK: DEPDC5 is associated with higher-than-average SUDEP risk — 3-fold
increased risk vs. general epilepsy population (Bagnall 2016, Klassen 2014). Seizures
often nocturnal with prone sleeping posture — DEPDC5-specific SUDEP prevention counselling.

KEY PRECISION MEDICINE OPPORTUNITY:
mTOR inhibitors (Everolimus/Sirolimus) — mechanistically targeted therapy (same pathway
as TSC → Everolimus). Small case series (Scheffer 2019, Baldassari 2019) show seizure
reduction in DEPDC5-LOF + FCD patients. Ongoing trials: GMTD (GAP mTOR Trial DEPDC5).
mTOR inhibitor response may predict surgical outcome — patients with FCD responding to
Everolimus: FCD resection after medical lead-in may improve outcomes.

SAFETY PEARLS:
• NOCTURNAL SEIZURE PROTOCOL: Bed rails, seizure mattress, NO prone sleeping.
• SUDEP COUNSELLING: mandatory at diagnosis — DEPDC5 has ~3× SUDEP excess risk.
• AED INTERACTIONS: Everolimus is CYP3A4 substrate — CBZ/OXC induce CYP3A4 → markedly
  reduce Everolimus levels; VPA weakly inhibits CYP3A4 → mild level increase; monitor TDM.
• FAMILY CASCADE SCREENING: AD inheritance — first-degree relatives need testing.
• FCD WORKUP: if drug-resistant, high-resolution 3T MRI with 1mm isotropic acquisition +
  FDG-PET + SEEG may reveal occult FCD missed on routine MRI (lesion in 30-50% of cases).
"""

import random
from datetime import datetime

SEED = 9182  # dashboard 182
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "Pathogenic DEPDC5 truncating / frameshift — FFEVF / AD focal epilepsy (de novo or familial)",
        "n": 18, "pct": 44,
        "category": "DEPDC5-truncating-frameshift-FFEVF-AD",
        "mechanism": (
            "Most prevalent class (~44%): de novo or familial (AD) nonsense, frameshift, or "
            "canonical splice-site variants causing premature stop codon (PTC) + NMD → "
            "DEPDC5 haploinsufficiency. Loss of one functional DEPDC5 allele reduces GATOR1 "
            "complex functional capacity by ~50% — GATOR1 normally limits mTORC1 activity "
            "in response to amino acid shortage. DEPDC5-LOF → constitutive RagA/B GTP-loading "
            "→ mTORC1 hyperactivation → S6K1/4E-BP1 phosphorylation → enhanced protein "
            "synthesis + cell growth + autophagy suppression. In cortical development: "
            "mTOR hyperactivation in neural progenitors → FCD type IIa (dysmorphic neurons "
            "without balloon cells) or IIb (with balloon cells). Variable foci across family "
            "members — a defining DEPDC5 hallmark (FFEVF). Penetrance ~60-70% (incomplete). "
            "Truncating variants are pathogenic by PVS1; de novo status confirmed by trio "
            "exome or familial segregation (autosomal dominant). ClinGen DEPDC5 dosage "
            "sensitivity: Haploinsufficiency Score 3 (Sufficient Evidence). Most severe "
            "phenotype in truncating class: drug-resistant focal epilepsy, FCD on MRI in "
            "~40% of truncating carriers."
        ),
        "eeg_signature": (
            "Focal epileptiform discharges from variable location (frontal, temporal, parietal, "
            "occipital) — often matches clinical seizure semiology (frontal → nocturnal hypermotor; "
            "temporal → aura + automatisms; occipital → visual aura). Interictal: focal spike-wave "
            "or sharp-slow discharge at seizure-onset zone; may be sparse/absent between seizures "
            "(up to 40% of DEPDC5 patients have normal routine EEG). Ictal: focal onset with "
            "evolving beta/gamma discharge; SEEG more sensitive than scalp EEG for deep/basal "
            "frontal foci. FCD-associated DEPDC5: continuous or near-continuous focal IED at "
            "cortical dysplasia site; high-amplitude fast activity at seizure onset ('low-voltage "
            "fast activity' pattern if FCD IIb). Background: normal in most; no generalised IED "
            "(distinguishes from generalised DEE)."
        ),
        "mri": (
            "FCD type IIa or IIb on MRI in ~40% of truncating class (bottom-of-sulcus FCD "
            "IIb most common — may require 3T 1mm isotropic T2/FLAIR with MPR reconstruction). "
            "Subtle blurring of grey-white junction at bottom of sulcus = FCD IIb signature. "
            "Normal MRI in 60% — occult FCD cannot be excluded; FDG-PET/MR fusion may reveal "
            "focal hypometabolism. No hippocampal sclerosis on routine MRI (unless coincidental)."
        ),
        "clinical_note": (
            "FAMILY CASCADE SCREENING mandatory for all first-degree relatives. AD with "
            "~60-70% penetrance — asymptomatic carriers exist. SUDEP COUNSELLING at "
            "diagnosis: DEPDC5-LOF has ~3× excess SUDEP risk (Bagnall 2016). Prescribe "
            "nocturnal seizure safety: bed rails, seizure monitor, AVOID prone sleeping. "
            "If drug-resistant: refer for presurgical evaluation — FCD surgery highly "
            "effective (Seizure Freedom 60-80% if FCD resected). mTOR inhibitor "
            "(Everolimus) trial eligibility: discuss GMTD trial with patient."
        ),
    },
    {
        "etiology": "Pathogenic DEPDC5 missense (LOF) — FFEVF / focal epilepsy (de novo or familial)",
        "n": 11, "pct": 27,
        "category": "DEPDC5-missense-LOF-FFEVF",
        "mechanism": (
            "De novo or familial missense LOF variants in DEPDC5 (~27%). Pathogenic missense "
            "cluster at: SHEN domain (N-terminal, residues 1-450 — RagA/B GAP domain binding); "
            "DEP domain (pleckstrin homology-like, C-terminal) — membrane targeting and GATOR1 "
            "complex formation; DEPDC-like domain (residues 900-1250). Functional assays: "
            "pathogenic missense reduce DEPDC5 GAP activity toward RagA (measured by GTP-hydrolysis "
            "assay or S6K1 phosphorylation in HEK293 cells). Partial GAP-activity reduction → "
            "intermediate mTOR hyperactivation → milder phenotype than truncating; FCD less "
            "frequent (~15-20%). Penetrance may be lower for missense vs. truncating variants "
            "(family-specific penetrance curves). ACMG classification: missense + functional "
            "assay data (PS3) + de novo status (PS2) + segregation in family (PP1-Strong) → "
            "likely pathogenic or pathogenic. ClinVar: many DEPDC5 missense VUS — request "
            "functional studies for reclassification. Phenotype: FFEVF (variable foci), ADNFLE, "
            "or single-family focal epilepsy; onset early adulthood (mean 12-18y)."
        ),
        "eeg_signature": (
            "Focal IED matching seizure focus — often frontal or temporal; less frequently "
            "occipital. Routine EEG frequently normal (up to 50% of missense class). "
            "Long-term ambulatory or video-EEG required to capture ictal events. Seizure "
            "onset zone: focal fast-activity in frontal patients; rhythmic 5-7 Hz in temporal "
            "patients. Sleep: IED may be more apparent in N2 sleep (ADNFLE-DEPDC5 — "
            "nocturnal hypermotor seizures arise from N2 sleep and are captured on PSG-EEG)."
        ),
        "mri": (
            "Normal in ~80% of missense class. FCD type IIa in ~15% (less severe FCD vs "
            "truncating class). Bottom-of-sulcus sign (T2 hyperintensity in sulcal depth) "
            "on 3T with thin sections. Consider FDG-PET in drug-resistant missense DEPDC5 "
            "with normal MRI to guide SEEG placement."
        ),
        "clinical_note": (
            "ADNFLE-DEPDC5 diagnostic trap: nocturnal hypermotor seizures from sleep may "
            "be misdiagnosed as nightmares or parasomnias. Key distinguishing features: "
            "stereotyped recurrence, brief duration (<2 min), sudden awakening, preserved "
            "memory, EEG scalp recording during event (may be normal on scalp EEG — "
            "SEEG confirms frontal origin). DEPDC5 missense VUS: if strong phenotype + "
            "AD family + frontal onset, request functional GAP assay before classifying VUS."
        ),
    },
    {
        "etiology": "DEPDC5 splice-site variant — focal epilepsy with FCD (de novo or familial)",
        "n": 5, "pct": 12,
        "category": "DEPDC5-splice-site-FCD-focal",
        "mechanism": (
            "Canonical or near-canonical splice-site variants in DEPDC5 (~12%). Aberrant "
            "splicing → exon skipping or intron retention → premature stop codon or missense "
            "amino acid insertion disrupting SHEN or DEP domains. Canonical ±1/2 splice variants "
            "treated as PVS1_Strong by ACMG. Deep-intronic DEPDC5 variants may require WGS + "
            "RNA sequencing from leukocytes or fibroblasts for confirmation. Two-hit somatic "
            "model in FCD tissue: heterozygous germline DEPDC5 splice variant + somatic second-"
            "hit in FCD-IIb lesion (D'Gama 2015, Ribierre 2018) — explains focal nature of "
            "dysplasia despite systemic haploinsufficiency. Somatic second-hit: detectable by "
            "deep WGS of brain biopsy tissue (>100× coverage at FCD lesion) — available via "
            "research sequencing of surgical specimens. Phenotype: drug-resistant focal epilepsy "
            "with FCD in majority of splice-site class; surgical resection highly effective."
        ),
        "eeg_signature": (
            "Consistent focal IED — most often frontal (hypomotor or hypermotor seizures) "
            "or temporal (aura + automatisms). High-amplitude focal slow with embedded "
            "spikes at FCD location. Ictal: low-voltage fast activity evolving to rhythmic "
            "theta at FCD-IIb onset; sharp contraction artifact on face EMG (hypermotor). "
            "SEEG: 'DC shift' pattern at FCD-IIb ictal onset (negative DC potential 2-10s "
            "before visible EEG fast activity) — pathognomonic of FCD type IIb."
        ),
        "mri": (
            "FCD type IIa or IIb in ~65% of splice-site class. Bottom-of-sulcus FCD IIb: "
            "T2/FLAIR cortical thickening + blurred grey-white junction + 'transmantle sign' "
            "(T2 hyperintensity extending from FCD to ventricle). High-resolution 3T with "
            "1mm isotropic voxels + MPR in 3 planes mandatory. FDG-PET: focal hypometabolism "
            "at FCD even when MRI subtle. SISCOM (subtraction ictal SPECT co-registered to MRI) "
            "for surgical planning in MRI-negative cases."
        ),
        "clinical_note": (
            "FCD resection highly effective in DEPDC5 splice variant: ~70-80% seizure freedom "
            "at 2 years if complete FCD resection (Seizure Freedom Engel Class I). Somatic "
            "second-hit in FCD tissue confirms two-hit pathogenesis — request deep WGS on "
            "surgical specimen (research protocol). Post-resection: continue AED 2 years "
            "post-surgery before taper attempt; monitor for recurrence at resection margin."
        ),
    },
    {
        "etiology": "NPRL2 / NPRL3 (other GATOR1 subunit) pathogenic variant — FFEVF phenocopy",
        "n": 4, "pct": 10,
        "category": "NPRL2-NPRL3-GATOR1-FFEVF-phenocopy",
        "mechanism": (
            "Pathogenic LOF variants in the other two GATOR1 subunits — NPRL2 (2p23.3) and "
            "NPRL3 (16p13.3) — produce a clinically identical FFEVF phenotype (~10% of this "
            "cohort). NPRL2 (Nitrogen Permease Regulator-Like 2): directly interacts with "
            "DEPDC5 C-terminal DEP domain; NPRL2 LOF prevents GATOR1 complex formation → "
            "identical mTOR hyperactivation. NPRL3 (Nitrogen Permease Regulator-Like 3): "
            "third GATOR1 subunit, interacts with NPRL2; NPRL3 LOF → complex destabilisation "
            "→ mTOR hyperactivation. Clinical equivalence: DEPDC5/NPRL2/NPRL3 LOF produce "
            "overlapping FFEVF + FCD phenotype — clinical management is identical. Genetic "
            "diagnosis: standard epilepsy panels should include all 3 GATOR1 genes. NPRL2 "
            "and NPRL3 are less frequently tested — underascertainment in many centres. "
            "In this cohort: 2 NPRL2 truncating; 1 NPRL3 frameshift; 1 NPRL2 missense (LOF "
            "confirmed by S6K1 assay)."
        ),
        "eeg_signature": (
            "Identical to DEPDC5 focal epilepsy phenotype: focal IED matching seizure "
            "semiology; variable foci across family members. NPRL2 cases in this cohort: "
            "2 frontal, 1 temporal, 1 occipital focus. FCD-associated interictal changes "
            "as for DEPDC5 (focal slow + embedded spikes at FCD location)."
        ),
        "mri": (
            "FCD type IIa or IIb in 2/4 NPRL2/NPRL3 patients (similar frequency to DEPDC5). "
            "Normal MRI in 2/4. FDG-PET positive in both MRI-negative cases."
        ),
        "clinical_note": (
            "GATOR1 complex epilepsy — treat as DEPDC5 epilepsy regardless of subunit. "
            "Everolimus (mTOR inhibitor) mechanistically appropriate for all 3 GATOR1 genes. "
            "Panel testing: ensure NPRL2 and NPRL3 are included in epilepsy panels — some "
            "older panels omit these genes. Family cascade testing: same AD + incomplete "
            "penetrance pattern as DEPDC5. SUDEP counselling mandatory."
        ),
    },
    {
        "etiology": "Clinical DEPDC5-negative FFEVF phenocopy — unexplained familial focal epilepsy",
        "n": 3, "pct": 7,
        "category": "Clinical-DEPDC5-negative-FFEVF-phenocopy",
        "mechanism": (
            "Patients with FFEVF/ADNFLE clinical phenotype but no GATOR1 gene (DEPDC5/NPRL2/NPRL3) "
            "pathogenic variant (~7%). Differential: (1) DEPDC5 deep-intronic variant missed on "
            "standard exome — WGS required (detects 5-15% additional causal variants); "
            "(2) Somatic DEPDC5 mosaicism in brain — not detectable from blood; (3) Other focal "
            "epilepsy genes: KCNT1-ADNFLE (gain-of-function), CHRNA4/CHRNB2 (ADNFLE — nicotinic "
            "receptor mutations); LGI1 (ADLTE — lateral temporal); RELN, ADGRV1 (FMTLE); "
            "(4) Non-genetic focal epilepsy mimicking FFEVF in family (phenocopy); (5) Rare "
            "DEPDC5 structural variants (CNV) at 22q12.3 — check CMA. Management: treat seizure "
            "phenotype (AED + SUDEP counselling). WGS analysis to expand diagnostic yield. "
            "Enrol in GATOR1 registry for research-grade WGS if clinically available."
        ),
        "eeg_signature": "Focal IED matching semiology; EEG indistinguishable from DEPDC5-positive FFEVF.",
        "mri": "Normal in all 3. Consider repeat MRI at 3T if initial 1.5T was normal in drug-resistant cases.",
        "clinical_note": (
            "FFEVF without genetic diagnosis: (1) Order WGS (not exome) — covers deep intronic, "
            "UTR, structural variants; (2) Check CMA for 22q12.3 deletion; (3) Panel expand to "
            "CHRNA4/CHRNB2/KCNT1/LGI1/RELN/ADGRV1; (4) Request brain organoid or iPSC model "
            "if research available for somatic mosaicism detection. SUDEP counselling: maintain "
            "nocturnal safety regardless of genetic result."
        ),
    },
]

# ── Seizure Types (4 types) ──────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Nocturnal hypermotor seizures (ADNFLE-type / frontal)",
        "pct": 68,
        "eeg_correlate": (
            "Scalp EEG: focal frontal onset — low-voltage fast activity (beta/gamma) or rhythmic "
            "theta from supplementary motor area (SMA) or lateral frontal cortex. Ictal scalp EEG "
            "often obscured by movement artifact in hypermotor phase — video review essential. "
            "SEEG: clear frontal ictal onset with propagation to cingulate → motor cortex. "
            "Sleep stage: emerges from N2 (NREM stage 2) predominantly; may cluster 2-4× per "
            "night. Duration 20-90 seconds; partial responsiveness or preserved consciousness "
            "in SMA-onset seizures."
        ),
        "clinical_tip": (
            "ADNFLE-DEPDC5 diagnostic pitfall: easily misdiagnosed as nightmares, sleep terrors, "
            "REM behaviour disorder, or parasomnia. Key clues: (1) Stereotyped, highly repetitive "
            "events; (2) Brief duration (<2 min); (3) Occur from established sleep (not sleep-wake "
            "transition); (4) Patient may have partial amnesia but partial awareness during event; "
            "(5) Family history of same events (AD pattern). PSG-EEG required for diagnosis — "
            "routine overnight EEG may miss seizures. SUDEP risk highest in this group (nocturnal "
            "prone position + unwitnessed seizures). Bed alarm or seizure monitor (Empatica E4, "
            "SmartWatch) + safety mattress mandatory."
        ),
    },
    {
        "type": "Focal aware seizures (aura / focal impaired awareness)",
        "pct": 58,
        "eeg_correlate": (
            "Focal IED at seizure-onset zone during interictal state — temporal, parietal, or "
            "occipital location common for aware focal seizures. Ictal: rhythmic 5-7 Hz theta "
            "discharge at onset zone; awareness preserved if onset in non-eloquent cortex. "
            "Aura types (location-dependent): frontal — déjà vu/fear; temporal — epigastric "
            "rising/déjà vu/automatisms; parietal — somatosensory tingling; occipital — visual "
            "phenomena (coloured flickering, scotoma). EEG may be falsely normal during brief "
            "focal aware seizures (<30s) — ambulatory or video-EEG to capture."
        ),
        "clinical_tip": (
            "Aura characterisation is critical for focus localisation in DEPDC5 (variable "
            "foci — must define this patient's seizure-onset zone). Use detailed aura interview: "
            "earliest sensation, direction of head turning, hand posturing, timing of altered "
            "awareness. Aura consistency predicts surgical outcome (consistent aura = focal "
            "onset = higher seizure-free rate post-surgery). Do not dismiss aura as anxiety "
            "or psychological — DEPDC5 auras can be intense and frightening."
        ),
    },
    {
        "type": "Focal to bilateral tonic-clonic seizures (FBTCS)",
        "pct": 49,
        "eeg_correlate": (
            "Ictal: focal onset (frontal/temporal/parietal) → rapid generalisation with "
            "bifrontal rhythmic delta → bilateral tonic-clonic phase. Postictal: generalised "
            "delta slowing; postictal Todd's paresis in motor-onset DEPDC5. Ictal scalp EEG "
            "during FBTCS usually shows clear bilateral discharge despite focal onset — focal "
            "onset may be captured only at very beginning before spread. SEEG provides definitive "
            "onset localisation. FBTCS are highest-SUDEP-risk seizure type in DEPDC5 — "
            "emphasise SUDEP counselling when FBTCS reported."
        ),
        "clinical_tip": (
            "FBTCS + nocturnal occurrence + family history = DEPDC5 FFEVF signature triad. "
            "SUDEP ACTION PLAN mandatory: (1) Avoid sleep deprivation; (2) Nocturnal monitoring "
            "(bed alarm, pulse oximeter); (3) No swimming alone, bathing without supervision, "
            "heights; (4) Driving prohibition (jurisdiction-dependent seizure-free period); "
            "(5) SUDEP counselling document in notes. AED optimisation to achieve FBTCS freedom "
            "reduces SUDEP risk."
        ),
    },
    {
        "type": "Focal clonic / tonic seizures (frontal or parietal onset)",
        "pct": 32,
        "eeg_correlate": (
            "Focal clonic: rhythmic 3-5 Hz contralateral clonic jerking from primary motor "
            "or SMA cortex; ictal EEG shows contralateral centro-parietal rhythmic 5-7 Hz "
            "theta. Focal tonic: sustained tonic posturing from SMA; ictal: low-voltage fast "
            "activity from SMA (often missed on scalp EEG — 'flat' scalp EEG during SMA-onset "
            "tonic seizure is a known false-negative). Todd's paresis post-clonic in primary "
            "motor onset. FCD at motor strip is high-stakes for surgical planning (must map "
            "eloquent cortex with functional MRI + direct cortical stimulation)."
        ),
        "clinical_tip": (
            "Motor cortex FCD-DEPDC5: if FCD at primary motor cortex, surgery requires "
            "eloquent cortex mapping (fMRI language/motor + intraoperative DCS). Partial "
            "resection may be offered if FCD is partially in eloquent cortex (vs no surgery). "
            "Discuss neurological deficit risk vs seizure improvement with patient/family — "
            "shared decision-making mandatory. Robotic SEEG placement may offer more accurate "
            "targeting of peri-central FCD than grid electrode arrays."
        ),
    },
]

# ── Triggers (8 triggers) ─────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Sleep deprivation / altered sleep architecture", "pct": 82,
     "mechanism": "Reduced sleep homeostasis pressure + K-complex suppression releases nocturnal "
                  "frontal foci; sleep deprivation lowers seizure threshold across all focal "
                  "epilepsies but disproportionately activates nocturnal DEPDC5 frontal seizures."},
    {"trigger": "Missed AED dose", "pct": 74,
     "mechanism": "Subtherapeutic AED levels — particularly relevant for carbamazepine (narrow "
                  "therapeutic window) and lacosamide (short half-life) commonly used in DEPDC5 "
                  "focal epilepsy. Breakthrough seizures after missed dose = highest SUDEP risk window."},
    {"trigger": "Stress / psychological stressors", "pct": 61,
     "mechanism": "HPA-axis activation → cortisol → reduced GABAergic inhibition + increased "
                  "glutamatergic tone → lowers focal seizure threshold. Particularly potent "
                  "trigger for temporal-onset DEPDC5 seizures."},
    {"trigger": "Alcohol consumption / hangover", "pct": 52,
     "mechanism": "Acute alcohol: GABAergic enhancement; alcohol withdrawal (hangover rebound): "
                  "GABAergic suppression + glutamatergic rebound → seizure precipitation. "
                  "Alcohol interactions with AEDs: CBZ levels increased by enzyme-inhibition effect; "
                  "PHT levels unpredictably altered; all AEDs + alcohol = increased CNS depression."},
    {"trigger": "Illness / fever / systemic infection", "pct": 44,
     "mechanism": "Fever lowers seizure threshold (temperature-dependent Na-channel kinetics). "
                  "Metabolic stress during illness may alter AED pharmacokinetics (reduced "
                  "absorption, altered renal/hepatic clearance) → breakthrough levels."},
    {"trigger": "Hormonal fluctuations (catamenial in women)", "pct": 35,
     "mechanism": "Perimenstrual oestrogen-progesterone shift: oestrogen is proconvulsant "
                  "(glutamatergic enhancement); progesterone withdrawal reduces allopregnanolone "
                  "(GABAergic neurosteroid) → perimenstrual seizure cluster. DEPDC5 women with "
                  "catamenial pattern: consider progesterone supplement or clobazam perimenstrual."},
    {"trigger": "AED interactions (enzyme inducer co-prescription)", "pct": 28,
     "mechanism": "CBZ/OXC/PHT induce hepatic CYP enzymes → reduce co-AED (LEV is renal — "
                  "not affected; LTG is glucuronidated — 50% level reduction with CBZ; CLB "
                  "is CYP3A4 substrate — levels reduced by CBZ). Everolimus (mTOR inhibitor): "
                  "CYP3A4 substrate — CBZ/OXC cause >80% Everolimus level reduction → "
                  "avoid CBZ/OXC co-prescription with Everolimus in DEPDC5."},
    {"trigger": "Sleep position (prone sleeping — nocturnal DEPDC5)", "pct": 22,
     "mechanism": "Prone sleep position is independent SUDEP risk factor in nocturnal seizures "
                  "— impairs arousal + airway protection during postictal period. DEPDC5 "
                  "nocturnal hypermotor seizures + prone sleeping = highest SUDEP risk dyad. "
                  "Prescribe supine sleep + bed alarm as safety mandatories."},
]

# ── Treatments (8 treatments) ────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "level": "Level B — focal epilepsy first-line (NICE NG217 / ILAE 2022)",
        "dose": "CBZ: 400-1200 mg/day (TDM 4-12 µg/mL); OXC: 600-2400 mg/day (MHD 12-24 µg/mL)",
        "moa": "Voltage-gated Na-channel blocker (Nav1.1/1.2/1.6) — reduces high-frequency "
               "neuronal firing at seizure-onset zone; particularly effective for frontal and "
               "temporal focal epilepsies.",
        "efficacy": "~50-60% seizure freedom in DEPDC5 focal epilepsy; high efficacy for "
                    "ADNFLE-type nocturnal hypermotor seizures. First-line for focal epilepsy.",
        "safety": "SJS/TEN risk: HLA-B*15:02 (SE Asian) — mandatory CPIC-Level A genetic "
                  "test before prescribing. SIADH/hyponatraemia (OXC > CBZ). Diplopia, "
                  "dizziness, ataxia at high doses. CYP3A4/2C8/2C9 induction: reduces "
                  "levels of co-administered drugs (LTG, CLB, OCP). CRITICAL: CBZ/OXC "
                  "markedly reduce Everolimus levels (mTOR inhibitor) — AVOID co-prescription "
                  "if Everolimus is planned.",
        "monitoring": "TDM: CBZ 4-12 µg/mL; OXC-MHD 12-24 µg/mL. Na at baseline and q3M "
                      "(OXC SIADH). HLA-B*15:02 before starting. LFT (CBZ induction).",
    },
    {
        "drug": "Lacosamide (LCM)",
        "level": "Level B — focal epilepsy first/second-line (NICE NG217 / FDA 2008)",
        "dose": "100-400 mg/day (twice daily; half-life 13h)",
        "moa": "Selective slow-inactivation enhancer of voltage-gated Na-channels — distinct "
               "MOA from CBZ/OXC (slow vs fast inactivation); no hepatic enzyme induction; "
               "no significant drug interactions.",
        "efficacy": "~40-50% ≥50% seizure reduction as adjunct; monotherapy effective for "
                    "focal epilepsy (non-inferior to CBZ in DEPDC5 focal; FDA approved "
                    "monotherapy). Advantage over CBZ: no CYP induction → does NOT reduce "
                    "Everolimus levels (safe co-prescription with mTOR inhibitor).",
        "safety": "PR interval prolongation (caution in AV block / cardiac disease); "
                  "dizziness, diplopia, nausea. No SJS/TEN risk. No hyponatraemia. "
                  "CYP2C19 inhibitors (OMP) may increase LCM levels slightly (minor). "
                  "Safe in CYP2C19/HLA-B*15:02 carriers.",
        "monitoring": "ECG at baseline (AV conduction); no TDM routinely required "
                      "(clinical titration). Renal dose adjustment (CrCl <30).",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — focal epilepsy adjunct (NICE NG217 / ILAE 2022)",
        "dose": "1000-3000 mg/day (twice daily; renal elimination)",
        "moa": "SV2A synaptic vesicle protein modulator — reduces exocytosis of "
               "neurotransmitter vesicles; also GABA-A receptor modulation. No enzyme "
               "induction or inhibition. Renally eliminated — no hepatic drug interactions.",
        "efficacy": "~30-40% ≥50% seizure reduction as adjunct in focal epilepsy. "
                    "Advantage: no CYP interactions → safe with Everolimus. Used in "
                    "DEPDC5 as add-on after CBZ/LCM monotherapy failure.",
        "safety": "Neuropsychiatric adverse effects (irritability, aggression, depression) "
                  "in ~10-15% — significant in patients with pre-existing anxiety/depression. "
                  "Pyridoxine (B6 100 mg/day) may mitigate mood effects. No hepatic or "
                  "haematological toxicity. Safe in pregnancy (limited data).",
        "monitoring": "No TDM routinely required. Renal dose adjustment mandatory "
                      "(CrCl <50: reduce dose 50%). Mood/behaviour monitoring q3M.",
    },
    {
        "drug": "Lamotrigine (LTG)",
        "level": "Level B — focal epilepsy first/second-line (NICE NG217 / ILAE 2022)",
        "dose": "100-500 mg/day (slow titration over 8-12 weeks to reduce SJS risk; "
                "halve dose if co-administered with VPA; double dose with CBZ)",
        "moa": "Voltage-gated Na-channel (fast inactivation) + N/P-type Ca-channel blockade. "
               "Glutamate release reduction. CYP induction by CBZ halves LTG levels; "
               "VPA inhibits glucuronidation → doubles LTG levels (SJS risk).",
        "efficacy": "Broad-spectrum; effective for focal epilepsy. Useful in DEPDC5 women "
                    "of childbearing age (safest AED in pregnancy — relatively). Low-dose "
                    "LTG + VPA combination carries SJS risk.",
        "safety": "SJS/TEN risk (especially with rapid titration or VPA co-prescription). "
                  "Dizziness, diplopia, headache. Safe haematologically. May worsen myoclonic "
                  "epilepsies (not relevant in focal DEPDC5).",
        "monitoring": "Titration schedule: start 25 mg/day × 2 weeks, then increase "
                      "25-50 mg q2 weeks. LTG TDM: 3-15 µg/mL (wide range). "
                      "Rash surveillance (any rash in first 8 weeks → withdraw and reassess).",
    },
    {
        "drug": "Everolimus (EVR) — mTOR inhibitor [investigational in DEPDC5]",
        "level": "Level C — investigational (GMTD trial / Scheffer 2019 case series)",
        "dose": "Investigational: 4.5-7.5 mg/m² BSA twice daily (titrate to trough "
                "level 5-10 ng/mL); paediatric <3y: very limited data",
        "moa": "mTORC1 inhibitor (same pathway as in TSC treatment) — directly reverses "
               "the mTOR hyperactivation caused by DEPDC5-LOF. FKBP12-Everolimus complex "
               "binds mTOR kinase → prevents mTORC1 phosphorylation of S6K1/4E-BP1 → "
               "reduces protein synthesis, cell growth, autophagy suppression in FCD tissue.",
        "efficacy": "Small DEPDC5 case series (Scheffer 2019): 3/5 DEPDC5-LOF + FCD patients "
                    "showed ≥50% seizure reduction with Everolimus 4.5-7.5 mg/m². Ongoing "
                    "GMTD trial (NCT04203940) — first RCT in DEPDC5/GATOR1. Mechanistically "
                    "rational but not yet Level A evidence.",
        "safety": "Immunosuppression → infection risk (prophylactic TMP-SMX for PCP if "
                  "long-term). Mucositis, impaired wound healing. Hyperlipidaemia. Nephrotoxicity "
                  "at high doses. Drug interaction CRITICAL: CBZ/OXC/PHT (CYP3A4 inducers) "
                  "reduce Everolimus AUC by >80% — AVOID. Use LCM or LEV as co-AED if Everolimus "
                  "planned. Voriconazole/ketoconazole (CYP3A4 inhibitors) markedly increase levels.",
        "monitoring": "Everolimus trough TDM: 5-10 ng/mL (whole blood LCMS-MS). "
                      "CBC, CMP, lipids at baseline q3M. Urinalysis. Oral mucositis inspection "
                      "q visit. Infection surveillance. HBV reactivation screening before start.",
    },
    {
        "drug": "Clobazam (CLB) — adjunct / rescue",
        "level": "Level C — focal epilepsy adjunct (NICE NG217)",
        "dose": "10-40 mg/day (twice daily or nocturnal dosing for ADNFLE-DEPDC5)",
        "moa": "Benzodiazepine — positive allosteric GABA-A modulator (1,5-benzodiazepine; "
               "less tolerance than 1,4-BZDs); active metabolite norclobazam. Nocturnal "
               "dosing exploits peak plasma concentration at sleep onset → reduces nocturnal "
               "frontal seizure frequency in ADNFLE-DEPDC5.",
        "efficacy": "Add-on CLB reduces nocturnal hypermotor seizure frequency in ADNFLE-DEPDC5 "
                    "by ~50-60% in open-label series. Catamenial: CLB 10-20 mg × 7-10 days "
                    "perimenstrual reduces catamenial breakthrough seizures.",
        "safety": "Sedation, cognitive slowing. Tolerance with chronic use (less than 1,4-BZDs). "
                  "CYP2C19 polymorphism: poor metabolisers → high norclobazam levels → excess "
                  "sedation (CPIC guidance available). CBZ co-administration reduces CLB levels "
                  "(CYP3A4 induction — may need dose increase).",
        "monitoring": "CLB TDM: 30-300 ng/mL (norclobazam 300-3000 ng/mL). "
                      "CYP2C19 genotype (CPIC Level A). Sedation VAS at each visit.",
    },
    {
        "drug": "Epilepsy surgery (FCD resection / laser ablation)",
        "level": "Level A — Drug-resistant focal epilepsy (ILAE Class I evidence / Wiebe 2001 RCT)",
        "dose": "N/A — procedure; presurgical evaluation protocol: MRI 3T + SEEG + FDG-PET + "
                "SISCOM + neuropsychology + fMRI language/memory + intraoperative ECoG",
        "moa": "Complete resection of FCD-IIa/IIb lesion removes hyperexcitable cortical tissue "
               "and primary epileptogenic zone → seizure freedom. In MRI-negative DEPDC5: SEEG "
               "localisation + radiofrequency thermocoagulation of SEEG contacts at seizure-"
               "onset zone (RFTC-SEEG) as alternative to open surgery.",
        "efficacy": "FCD resection in DEPDC5: ~65-75% seizure freedom (Engel I) at 2 years "
                    "when FCD completely resected. FDG-PET guided resection in MRI-negative: "
                    "~50-60% Engel I. RFTC-SEEG: ~35-45% seizure freedom (lower but less "
                    "invasive option for small lesions).",
        "safety": "Craniotomy risks: infection (1-3%), haemorrhage (<1%), neurological deficit "
                  "(location-dependent: motor cortex 10-20% transient, 2-5% permanent; language "
                  "cortex). SEEG risks: haemorrhage <1%, infection <1%, electrode breakage <1%.",
        "monitoring": "Post-resection EEG at 3M, 6M, 12M, 24M. Neuropsychological follow-up "
                      "at 6M post-surgery. AED weaning: typically wait 2 years seizure-free "
                      "before AED taper attempt (DEPDC5 recurrence risk 20-30% after wean).",
    },
    {
        "drug": "Ketogenic Diet (KD) — adjunct for drug-resistant DEPDC5",
        "level": "Level B — drug-resistant focal epilepsy (ILAE Dietary Therapies 2018)",
        "dose": "4:1 or 3:1 lipid:carbohydrate+protein ratio; dietitian-supervised; "
                "ketone target: BHB 2-4 mmol/L",
        "moa": "Ketone bodies (beta-hydroxybutyrate) inhibit mTORC1 signalling — mechanistically "
               "relevant in DEPDC5-LOF (additive mTOR suppression alongside Everolimus or "
               "independently). Also: GABAergic modulation, HCN channel effects, mitochondrial "
               "biogenesis enhancement.",
        "efficacy": "~40-50% ≥50% seizure reduction in drug-resistant focal epilepsy (ILAE 2018 "
                    "recommendation). In DEPDC5: small series suggest KD + mTOR rationale; KD "
                    "as bridge to surgery or standalone for surgery-ineligible patients.",
        "safety": "Dyslipidaemia, nephrolithiasis (K-citrate supplementation), growth impairment "
                  "(children), constipation, acidosis risk. POLG mitochondrial disease exclusion "
                  "before KD (contraindicated in POLG). Metabolic screening pre-KD.",
        "monitoring": "BHB ketones (Freestyle Libre Ketone or blood ketometer) target 2-4 mmol/L. "
                      "Lipid profile q6M. Renal USS annually (nephrolithiasis). "
                      "Micronutrient monitoring: Se, Zn, carnitine.",
    },
]

# ── Contraindications (4) ────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "item": "Everolimus + CBZ/OXC/PHT co-prescription — AVOID",
        "reason": "CYP3A4 inducers (CBZ/OXC/PHT) reduce Everolimus AUC by >80% — "
                  "renders mTOR inhibitor therapy ineffective. If Everolimus planned for "
                  "DEPDC5 precision therapy, switch to non-inducing AED (LCM, LEV, LTG "
                  "without VPA) as co-medication before starting Everolimus.",
    },
    {
        "item": "Prone sleeping position — mandatory avoidance in nocturnal DEPDC5",
        "reason": "Prone sleep position is the strongest modifiable SUDEP risk factor in "
                  "nocturnal epilepsy — impairs airway reflexes during postictal unconsciousness. "
                  "DEPDC5 nocturnal hypermotor seizures + prone sleeping = highest-risk SUDEP "
                  "dyad. Prescribe supine/lateral sleep position + nocturnal monitoring at "
                  "every clinical encounter.",
    },
    {
        "item": "HLA-B*15:02 carriers — CBZ/OXC CONTRAINDICATED (SJS/TEN)",
        "reason": "HLA-B*15:02 allele (Southeast Asian ancestry) carriers have >15× risk of "
                  "CBZ/OXC-induced Stevens-Johnson Syndrome and Toxic Epidermal Necrolysis "
                  "(potentially fatal). CPIC Level A: CBZ/OXC contraindicated in HLA-B*15:02 "
                  "carriers. Mandatory HLA-B*15:02 testing before prescribing CBZ/OXC in "
                  "patients of SE Asian, Han Chinese, Thai, Vietnamese, Korean ancestry.",
    },
    {
        "item": "Delay in SUDEP counselling — NEVER defer",
        "reason": "DEPDC5-LOF carries ~3× excess SUDEP risk vs general focal epilepsy. "
                  "SUDEP counselling (Morrell 2018 JAMA Neurol / NICE NG217) must occur "
                  "at or within 1 visit of diagnosis. Document counselling in notes. "
                  "Failure to counsel constitutes a patient safety gap — high medicolegal "
                  "significance. Provide SUDEP Action Plan (SUDEP Action website).",
    },
]

# ── Monitoring Items (8) ─────────────────────────────────────────────────────
MONITORING = [
    {"item": "CBZ TDM 4-12 µg/mL", "schedule": "At steady-state, q6M, after dose changes",
     "detail": "Trough sample (pre-morning dose). GC-MS or HPLC. Active metabolite "
               "CBZ-10,11-epoxide also measured (epileptogenic). Levels altered by CYP induction "
               "(auto-induction) and interaction with co-AEDs."},
    {"item": "LCM TDM (clinical titration — not routine)", "schedule": "If toxicity or seizure recurrence",
     "detail": "LCM therapeutic range: 2-10 µg/mL (not routinely monitored). ECG at baseline "
               "for PR interval (AV block caution). Renal function q6M (LCM renally eliminated)."},
    {"item": "Everolimus trough (mTOR inhibitor TDM)", "schedule": "q2 weeks during titration, q3M at stable dose",
     "detail": "Target trough: 5-10 ng/mL (whole blood LCMS-MS). Below 5 ng/mL: insufficient mTOR "
               "inhibition. Above 10 ng/mL: immunosuppression + nephrotoxicity risk. CBC, CMP, "
               "lipids q3M during Everolimus therapy."},
    {"item": "HLA-B*15:02 genotype (CPIC Level A)", "schedule": "Once before CBZ/OXC prescription",
     "detail": "Mandatory in patients of SE Asian ancestry before CBZ/OXC. Turnaround 2-5 days "
               "for commercial genetic test. Alternative: use LCM or LTG if HLA result pending "
               "and urgent AED needed."},
    {"item": "SUDEP risk assessment + counselling documentation", "schedule": "At diagnosis and annually",
     "detail": "SUDEP-7 instrument (Harden 2014) — quantifies SUDEP risk factors: GTCS frequency, "
               "nocturnal seizures, non-adherence, solo living, suboptimal AED. Document counselling "
               "in medical notes. Prescribe: bed alarm (Empatica/SmartWatch), supine sleep position, "
               "nocturnal seizure monitor, no unsupervised bathing."},
    {"item": "Presurgical evaluation (if drug-resistant)", "schedule": "After 2 AED failures",
     "detail": "Referral criteria: failure of 2 appropriately trialled AEDs → drug-resistant focal "
               "epilepsy (ILAE 2010 definition). Presurgical workup: 3T MRI (1mm isotropic) + "
               "video-EEG (scalp) + FDG-PET + SISCOM. SEEG if non-concordant data or deep/"
               "mesial/multifocal onset. Neuropsychology + fMRI language/memory before surgery."},
    {"item": "Family cascade genetic testing", "schedule": "At diagnosis — first-degree relatives",
     "detail": "DEPDC5: autosomal dominant, ~60-70% penetrance. First-degree relatives "
               "(parents, siblings, children) require targeted DEPDC5 testing. Asymptomatic "
               "carriers: EEG + clinical assessment. Positive carrier: counsel re SUDEP risk "
               "and AED prophylaxis discussion (shared decision)."},
    {"item": "Neuropsychological assessment", "schedule": "Baseline, post-seizure-freedom or post-surgery",
     "detail": "WAIS-IV (adults), WISC-V (children) — cognitive profile in DEPDC5 is usually "
               "normal/near-normal (focal epilepsy, not DEE). Memory assessment (WMS-IV) "
               "if temporal onset (hippocampal involvement risk). Post-surgery neuropsych: "
               "6M post-resection to assess language/memory impacts."},
]

# ── Lifecycle Windows (6) ─────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Genetic discovery / pre-symptomatic (any age — familial testing)",
        "age": "Variable — detected via cascade screening in AD family",
        "focus": (
            "Family member tested after index case identified — DEPDC5 carrier status known. "
            "If pre-symptomatic: EEG (may show focal IED before clinical seizures in ~20%); "
            "counsel re ~60-70% penetrance and variable onset age (range 0.5-66y, median ~15y). "
            "No prophylactic AED in asymptomatic carriers — seizure onset variable and uncertain. "
            "SUDEP counselling at genetic result disclosure. Lifestyle advice: avoid precipitants "
            "(sleep deprivation, alcohol, missed doses when seizures begin)."
        ),
    },
    {
        "window": "Seizure onset — childhood / adolescence",
        "age": "Median onset 10-18y (range 6M-40y)",
        "focus": (
            "First seizure: most commonly nocturnal hypermotor or focal to bilateral tonic-clonic "
            "in adolescence. EEG: focal IED at onset zone (may be normal on routine EEG — request "
            "prolonged or ambulatory EEG if high clinical suspicion). MRI 3T: baseline to identify "
            "FCD. First-line AED: CBZ (NICE NG217 Level A for focal epilepsy) or LCM (if HLA-"
            "B*15:02 risk or Everolimus planned). SUDEP counselling at first seizure — do not delay. "
            "School/driving implications to address. Family cascade testing for siblings/parents."
        ),
    },
    {
        "window": "AED optimisation — young adult",
        "age": "18-30 years",
        "focus": (
            "AED titration to seizure freedom — FBTCS freedom is the minimum target (SUDEP risk "
            "reduction). Lifestyle counselling: no alcohol, no sleep deprivation, AED adherence "
            "app. Driving: inform of jurisdiction-specific seizure-free requirements (typically "
            "6-12 months seizure-free for car; 5+ years for commercial). Contraception counselling: "
            "CBZ/OXC are enzyme inducers — reduce OCP efficacy; LTG levels reduced by OCP → "
            "breakthrough seizures. Women planning pregnancy: LTG or LEV preferred over CBZ "
            "(teratogenicity profile). Everolimus + OCP interaction: no significant interaction."
        ),
    },
    {
        "window": "Drug-resistant focal epilepsy — presurgical evaluation",
        "age": "Any age after 2 AED failures (ILAE drug-resistance definition)",
        "focus": (
            "ILAE drug-resistance: failure of 2 appropriately trialled AEDs at adequate doses "
            "and duration → refer for presurgical evaluation without delay. Do NOT add third "
            "AED without presurgical referral (multiple AED trials without surgery = missed "
            "surgical window). Presurgical workup: 3T MRI + video-EEG + FDG-PET + SISCOM + "
            "SEEG (if non-concordant). FCD identified → surgical resection (65-75% Engel I). "
            "MRI-negative → SEEG-guided radiofrequency ablation or guided resection. "
            "Everolimus trial: discuss GMTD trial enrolment for DEPDC5-LOF + FCD."
        ),
    },
    {
        "window": "Post-surgical / Everolimus treated — monitoring phase",
        "age": "Post-procedure (typically 20-50y)",
        "focus": (
            "FCD resection: postoperative EEG at 3M, 6M, 12M, 24M. AED weaning at 2 years "
            "if Engel I (seizure-free). Recurrence risk ~25% after AED wean — discuss with "
            "patient before taper. Neuropsychological follow-up 6M post-surgery. Everolimus: "
            "TDM trough 5-10 ng/mL q3M; CBC/CMP/lipids q3M; infectious disease monitoring. "
            "Everolimus response assessment at 3M: ≥50% seizure reduction = responder; continue "
            "at least 6M before discontinuation decision."
        ),
    },
    {
        "window": "Long-term remission / adult chronic management",
        "age": "30+ years",
        "focus": (
            "DEPDC5 penetrance/severity may improve with age in some family members (natural "
            "history variable). Long-term AED: annual medication review (is AED still needed? "
            "especially if seizure-free >2y post-surgery). SUDEP risk persists even in "
            "remission if nocturnal seizures resume — maintain nocturnal monitoring. "
            "Family planning: preconception genetic counselling (50% of children inherit "
            "DEPDC5-LOF allele; 60-70% of inheriting children will develop epilepsy). "
            "Preimplantation genetic testing (PGT-M) available for DEPDC5 families."
        ),
    },
]

# ── Definitions / Concepts (14) ──────────────────────────────────────────────
DEFINITIONS = [
    {"term": "DEPDC5 (Dishevelled, Egl-10, Pleckstrin domain-containing protein 5)",
     "definition": "Gene at 22q12.3 encoding the scaffold subunit of the GATOR1 complex. "
                   "DEPDC5 contains: N-terminal SHEN domain (RagA/B binding + GAP activity), "
                   "central region, C-terminal DEP domain (membrane targeting). The most "
                   "commonly mutated gene in familial focal epilepsy — LOF variants account "
                   "for ~10% of all familial focal epilepsy families worldwide (Ishida 2013, "
                   "Dibbens 2013). The name reflects its protein domain composition (DEP + "
                   "Dishevelled/EGL-10/Pleckstrin domains identified in Drosophila/C. elegans)."},
    {"term": "GATOR1 Complex (GTPase-activating protein toward Rags complex 1)",
     "definition": "Trimeric complex: DEPDC5 + NPRL2 + NPRL3. Negatively regulates mTORC1 "
                   "by accelerating GTP hydrolysis of RagA/B GTPases (GAP activity). "
                   "Activated when amino acids are scarce → inhibits mTORC1 via lysosomal "
                   "amino acid sensing mechanism. LOF in any GATOR1 subunit → constitutive "
                   "mTORC1 hyperactivation → FCD formation and epilepsy. GATOR1 directly "
                   "opposes GATOR2 (which activates Rag GTPases). Structural biology: "
                   "cryo-EM structure resolved 2019 (Saxton 2019 Nature)."},
    {"term": "mTORC1 (mechanistic Target of Rapamycin Complex 1)",
     "definition": "Master regulatory kinase complex integrating nutrient, growth factor, "
                   "and energy signals → phosphorylates S6K1 and 4E-BP1 → promotes protein "
                   "synthesis, ribosome biogenesis, lipid synthesis; inhibits autophagy. "
                   "Normally inhibited by GATOR1 complex (via Rag GTPase inactivation). "
                   "Dysregulated (hyperactivated) in DEPDC5-LOF → excess protein synthesis "
                   "→ cortical dysplasia and seizures. Inhibited therapeutically by "
                   "Everolimus (mTOR inhibitor = rapamycin analogue = rapalog)."},
    {"term": "FFEVF (Familial Focal Epilepsy with Variable Foci)",
     "definition": "Syndrome where seizure focus varies between family members (AD inheritance). "
                   "Defining characteristic of DEPDC5-related epilepsy: one family member may "
                   "have frontal lobe seizures, another temporal, another occipital — all with "
                   "the same DEPDC5 variant. FFEVF was the first clinical syndrome linked to "
                   "DEPDC5 (Dibbens 2013 NatGenet). ILAE 2022 recognises FFEVF as a specific "
                   "genetic epilepsy syndrome. Penetrance ~60-70%; variable foci explained by "
                   "stochastic somatic second-hit in different cortical regions during development."},
    {"term": "ADNFLE (Autosomal Dominant Nocturnal Frontal Lobe Epilepsy)",
     "definition": "Syndrome of nocturnal hypermotor seizures from frontal lobe, AD inheritance. "
                   "DEPDC5 accounts for ~12% of ADNFLE families (other genes: CHRNA4/CHRNB2 "
                   "= nicotinic receptor subunits, historically the first ADNFLE genes). "
                   "Seizures: brief (20-90s), stereotyped, arise from N2 sleep, hypermotor "
                   "(thrashing, pedalling, arm posturing), ± vocalisation. High diagnostic "
                   "challenge — misdiagnosed as parasomnias for years in many patients. "
                   "Video-PSG-EEG is gold standard for diagnosis."},
    {"term": "FCD (Focal Cortical Dysplasia) — Type IIa/IIb",
     "definition": "Malformation of cortical development characterised by: FCD IIa = dysmorphic "
                   "neurons + disrupted laminar architecture; FCD IIb = IIa features + balloon "
                   "cells (giant astrocyte-like cells expressing both glial and neuronal markers). "
                   "FCD-IIb in DEPDC5: caused by somatic second-hit mTOR pathway mutation "
                   "in cortical progenitor → mTOR hyperactivation → abnormal cell growth. "
                   "MRI: T2/FLAIR cortical thickening, blurred grey-white junction, transmantle "
                   "sign. Highly epileptogenic — complete resection → seizure freedom ~70%."},
    {"term": "Somatic Two-Hit Model (DEPDC5 + somatic mTOR activation)",
     "definition": "Hypothesis explaining focal nature of FCD in systemic DEPDC5-LOF: "
                   "germline DEPDC5-LOF (first hit, in all cells) + somatic activating "
                   "mutation in mTOR pathway (second hit, in single cortical progenitor during "
                   "development) → focal mTOR hyperactivation → localised FCD. Second-hit "
                   "genes: mTOR (Ser2215, Thr2173), PIK3CA, RHEB, TSC1/2, MTOR. Detectable "
                   "by deep WGS (>100×) of surgical FCD tissue. Confirms why FCD is spatially "
                   "restricted despite systemic DEPDC5 haploinsufficiency (D'Gama 2015, "
                   "Ribierre 2018 Science)."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy) — DEPDC5 excess risk",
     "definition": "DEPDC5-LOF epilepsy has ~3× higher SUDEP rate than general focal epilepsy "
                   "(Bagnall 2016 Neurology, Klassen 2014 Neurology). Mechanism: nocturnal "
                   "FBTCS in prone position → postictal hypoventilation/apnea → hypoxia → "
                   "cardiac arrhythmia. DEPDC5-specific SUDEP risk factors: nocturnal hypermotor "
                   "seizures, FBTCS, poor AED adherence, solo nocturnal sleeping. Prevention: "
                   "maximise seizure freedom (AED optimisation + surgery), nocturnal monitoring, "
                   "avoid prone sleeping, bed alarm system."},
    {"term": "Everolimus (Afinitor) — mTOR inhibitor precision therapy",
     "definition": "Rapamycin analogue (rapalog) that forms FKBP12-Everolimus complex → allosteric "
                   "mTOR inhibitor → mTORC1 inactivation. FDA-approved for TSC-associated subependymal "
                   "giant cell astrocytoma (SEGA) and renal angiomyolipoma. Off-label/investigational "
                   "for DEPDC5-LOF (same mTOR pathway mechanistic basis). GMTD Trial (NCT04203940): "
                   "randomised crossover design, DEPDC5/NPRL2/NPRL3 LOF carriers, Everolimus "
                   "4.5 mg/m² twice daily. Monitoring: TDM trough 5-10 ng/mL."},
    {"term": "SEEG (Stereo-EEG) — invasive presurgical evaluation",
     "definition": "Depth electrode implantation for 3D seizure-onset zone mapping before "
                   "FCD resection. SEEG is preferred over grid/strip electrodes (ECoG) for: "
                   "deep/mesial/multifocal foci, bilateral/multifocal hypotheses, small FCD, "
                   "MRI-negative cases. Robotically guided SEEG implantation (Rosa, ROSA One): "
                   "<1% haemorrhage, <1% infection. SEEG provides: ictal onset zone, propagation "
                   "network, eloquent cortex mapping via stimulation (language, motor, memory). "
                   "RFTC-SEEG: radiofrequency thermocoagulation of SEEG contacts at ictal onset "
                   "zone — option for small/deep FCD not amenable to open resection."},
    {"term": "GATOR1 Alliance / DEPDC5-GATOR1 Research Network",
     "definition": "International research consortium coordinating genotype-phenotype databases, "
                   "natural history studies, and clinical trial readiness for GATOR1-related "
                   "epilepsies. Enrolment via EPIGAD registry (European) and GMTD trial "
                   "(NCT04203940). Patient advocacy: CURE Epilepsy, Epilepsy Foundation "
                   "DEPDC5 resources. Genotype-phenotype data submission: ClinVar + DECIPHER "
                   "+ LOVD-DEPDC5 databases."},
    {"term": "Penetrance (DEPDC5) — incomplete penetrance",
     "definition": "~60-70% of DEPDC5-LOF carriers develop epilepsy. Penetrance is variant-"
                   "dependent (truncating > missense), family-dependent, and possibly modified "
                   "by: sex (slight male predominance in some families), genetic background "
                   "(modifier genes — mTOR pathway variants), somatic second-hit probability "
                   "during development. ~30-40% of DEPDC5-LOF carriers remain seizure-free "
                   "lifelong — prognostic uncertainty at individual level. Clinical implication: "
                   "asymptomatic DEPDC5 carrier = monitor, no prophylactic AED (seizure risk "
                   "uncertain for individual), counselling re signs of seizure onset."},
    {"term": "mTOR inhibitor drug interactions — CYP3A4",
     "definition": "Everolimus is a sensitive CYP3A4 and P-glycoprotein substrate. "
                   "CYP3A4 INDUCERS (reduce Everolimus levels ≥80%): CBZ, OXC, PHT, PB, "
                   "rifampicin, St John's Wort — AVOID co-prescription. "
                   "CYP3A4 INHIBITORS (increase Everolimus levels 2-15×): ketoconazole, "
                   "voriconazole, clarithromycin — use with extreme caution + dose reduction. "
                   "Safe co-AEDs with Everolimus: LCM (no CYP3A4), LEV (renal), CLB (minor "
                   "CYP3A4 substrate), LTG (glucuronidated). AED selection in DEPDC5 patients "
                   "with planned Everolimus therapy must account for this interaction."},
    {"term": "SUDEP Action Plan — DEPDC5-specific safety protocol",
     "definition": "Structured safety counselling document (SUDEP Action 2018 / Morrell 2018 "
                   "JAMA Neurol) mandatory for all DEPDC5 patients: (1) Nocturnal monitoring "
                   "(EMFIT bed sensor, Empatica E4, Embrace2 seizure watch); (2) Avoid prone "
                   "sleeping — lateral or supine only; (3) Safety mattress / rail; "
                   "(4) No unsupervised bathing or swimming; (5) Driving restriction (inform "
                   "of jurisdiction regulations); (6) AED adherence app (Medisafe); "
                   "(7) Emergency rescue plan (seizure rescue medication at home — CLB buccal "
                   "or diazepam rectal PRN for prolonged/clustered events); "
                   "(8) SUDEP Action website (sudepaction.org) — patient/caregiver resources."},
]

# ── Clinical Alerts (6) ──────────────────────────────────────────────────────
ALERTS = [
    {"text": "⚠️ SUDEP RISK 3× ELEVATED — DEPDC5-LOF: mandatory SUDEP counselling at diagnosis. "
             "Document in notes. Prescribe nocturnal monitoring + supine sleep protocol.", "variant": "danger"},
    {"text": "🧬 FAMILY CASCADE TESTING MANDATORY — AD inheritance (60-70% penetrance). "
             "Test all first-degree relatives. Asymptomatic carriers: EEG + annual review.", "variant": "warning"},
    {"text": "💊 EVEROLIMUS + CBZ/OXC: CONTRAINDICATED — CYP3A4 induction reduces Everolimus "
             "AUC >80%. Use LCM/LEV as co-AED if mTOR inhibitor therapy planned.", "variant": "danger"},
    {"text": "🔬 HLA-B*15:02: MANDATORY before CBZ/OXC in SE Asian patients — SJS/TEN risk. "
             "CPIC Level A recommendation. Use LCM or LTG while awaiting result.", "variant": "warning"},
    {"text": "⚕️ DRUG-RESISTANT: REFER EARLY for presurgical evaluation after 2 AED failures — "
             "FCD resection 65-75% Engel I. Do NOT add third AED without surgical referral.", "variant": "info"},
    {"text": "🛌 NOCTURNAL HYPERMOTOR: Bed alarm + lateral/supine sleep + no unsupervised bathing "
             "mandatory. Prone sleeping is the primary modifiable SUDEP risk factor.", "variant": "warning"},
]

# ── Standards (8) ────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE-2022 (Fisher — Epilepsy Classification + DEPDC5-GATOR1 genetic focal epilepsy recognition)",
    "NICE-NG217-2022 (Epilepsies in Adults — focal epilepsy AED Level A-B recommendations)",
    "CPIC-HLA-B-CBZ-2023 (CBZ/OXC pharmacogenomics — HLA-B*15:02 CPIC Level A mandate)",
    "ILAE-Dietary-Therapies-2018 (Ketogenic Diet for drug-resistant focal epilepsy, Level B)",
    "ACMG-AMP-2015 (Pathogenicity classification — DEPDC5 LOF PVS1 + PS3 functional criteria)",
    "ILAE-Presurgical-Evaluation-2019 (SEEG + FDG-PET + SISCOM presurgical protocol)",
    "SUDEP-Action-Counselling-2018-Morrell-JAMA-Neurol (mandatory SUDEP counselling guidelines)",
    "ACNS-EEG-Standards-2021 (EEG acquisition, SEEG-VEEG protocol, ambulatory EEG)",
]

# ── Thresholds (10) ──────────────────────────────────────────────────────────
THRESHOLDS = [
    "AED-drug-resistance: 2 appropriately-trialled AEDs failure → presurgical referral MANDATORY",
    "Everolimus-trough: 5-10 ng/mL (whole-blood LCMS-MS) — below = ineffective, above = toxic",
    "CBZ-TDM: 4-12 µg/mL (trough pre-morning dose); epoxide monitored if toxicity",
    "OXC-MHD: 12-24 µg/mL (trough); Na+ q3M (SIADH threshold: Na<130 mEq/L → withdraw)",
    "HLA-B*15:02: ANY carrier → CBZ/OXC contraindicated (CPIC Level A — absolute)",
    "SUDEP-counselling: within first clinical visit of diagnosis — never defer",
    "FCD-resection-outcome: Engel I goal (≥65% DEPDC5-FCD); <50% = review SEEG localisation",
    "Family-cascade-testing: first-degree relatives within 3 months of index diagnosis",
    "Everolimus-CYP3A4-inducer-AVOID: CBZ/OXC/PHT reduce AUC >80% — switch AED before starting EVR",
    "SUDEP-nocturnal-prevention: prone sleep → immediate lateral/supine correction + bed alarm",
]

# ── References (6) ────────────────────────────────────────────────────────────
REFERENCES = [
    "Dibbens-2013-NatGenet: DEPDC5 mutations in familial focal epilepsy with variable foci",
    "Ishida-2013-NatGenet: DEPDC5 mutations in ADNFLE / FMTLE / FFEVF",
    "Bagnall-2016-Neurology: Sudden unexpected death in DEPDC5 epilepsy — 3× excess risk",
    "Scheffer-2019-Neurology: Everolimus for DEPDC5-LOF with FCD — proof of concept case series",
    "Ribierre-2018-Science: Somatic DEPDC5 second-hit in FCD tissue confirms two-hit model",
    "Saxton-2019-Nature: Cryo-EM structure of GATOR1 complex bound to Rag-Ragulator",
]


def get_overview():
    """DEPDC5 Focal Epilepsy — overview endpoint."""
    total = sum(e["n"] for e in ETIOLOGY_CATALOG)
    return {
        "syndrome": "DEPDC5 Focal Epilepsy (GATOR1 Complex / FFEVF / ADNFLE-DEPDC5)",
        "gene": "DEPDC5 — 22q12.3 — Dishevelled/EGL-10/Pleckstrin domain protein 5",
        "complex": "GATOR1 = DEPDC5 + NPRL2 + NPRL3 — mTORC1 negative regulator",
        "pathway": "GATOR1 → Rag GTPase inactivation → mTORC1 inhibition (amino acid sensing)",
        "lof_consequence": "GATOR1 loss → constitutive mTORC1 hyperactivation → FCD + focal epilepsy",
        "inheritance": "Autosomal dominant — incomplete penetrance (~60-70%)",
        "prevalence_focal_epilepsy": "~10% of familial focal epilepsy; ~1% population-level focal epilepsy",
        "cohort": total,
        "etiology_classes": len(ETIOLOGY_CATALOG),
        "seizure_types": len(SEIZURE_TYPES),
        "triggers": len(TRIGGERS),
        "treatments": len(TREATMENTS),
        "contraindications": len(CONTRAINDICATIONS),
        "monitoring_items": len(MONITORING),
        "lifecycle_windows": len(LIFECYCLE),
        "concepts": len(DEFINITIONS),
        "standards": len(STANDARDS),
        "thresholds": len(THRESHOLDS),
        "references": len(REFERENCES),
        "sudep_risk": "~3× elevated vs general focal epilepsy (Bagnall 2016)",
        "precision_medicine": "Everolimus (mTOR inhibitor) — investigational (GMTD trial NCT04203940)",
        "key_surgery_option": "FCD resection → 65-75% Engel I seizure freedom",
        "top_alerts": ALERTS[:3],
        "generated_at": datetime.now().isoformat(),
        "dashboard_id": 182,
    }


def get_breakdown():
    """DEPDC5 Focal Epilepsy — breakdown endpoint (full clinical detail)."""
    return {
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "alerts": ALERTS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "generated_at": datetime.now().isoformat(),
    }


def get_definitions():
    """DEPDC5 Focal Epilepsy — definitions endpoint (14 key concepts)."""
    return {
        "syndrome": "DEPDC5 Focal Epilepsy (GATOR1 Complex / FFEVF / ADNFLE-DEPDC5)",
        "definitions": DEFINITIONS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "generated_at": datetime.now().isoformat(),
    }
