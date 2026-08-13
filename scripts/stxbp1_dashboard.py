"""
STXBP1 Encephalopathy (STXBP1-DEE) Dashboard
==============================================
41-patient cohort · STXBP1 (9q34.11) · Munc18-1 / Syntaxin Binding Protein 1
STXBP1 encephalopathy: pathogenic de novo variant in STXBP1 (syntaxin-binding
protein 1, 9q34.11) causing early-infantile developmental epileptic encephalopathy
(DEE). STXBP1 (Munc18-1) is essential for synaptic vesicle fusion and
neurotransmitter release; haploinsufficiency reduces inhibitory > excitatory
synaptic tone, precipitating severe neonatal/early-infantile epilepsy.
KEY EEG: Burst-suppression (Ohtahara-like) in neonates → suppression-burst with
hypsarrhythmia (IS/West) at 3-6 months → multifocal independent spike-wave in
childhood. Theta hypersynchrony in wakefulness is highly characteristic.
AED NOTE: No STXBP1-specific AED approved; phenobarbital (acute neonatal first-line);
ACTH + vigabatrin for infantile spasms phase (UKISS protocol); ketogenic diet
highly effective (level B); levetiracetam adjunct. Phenytoin may worsen.
DISEASE-MODIFYING: No approved therapy; Munc18-1 chaperone rescue (OP-01, Phase 1)
and STXBP1 ASO/gene therapy in early research. Refer to STXBP1 Foundation registry.
"""

import random
from datetime import datetime

SEED = 9175  # dashboard 175
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ─────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "De novo STXBP1 missense (haploinsufficiency / LOF)",
        "n": 18, "pct": 44,
        "category": "De-novo-STXBP1-missense-LOF",
        "mechanism": (
            "The most common STXBP1-DEE variant class (~44%): de novo missense variants "
            "causing loss-of-function (LOF) through protein misfolding, instability, or "
            "impaired binding to syntaxin-1A (STX1A). STXBP1 (Munc18-1) belongs to the "
            "Sec1/Munc18 (SM) protein family and is an obligate chaperone for STX1A — "
            "required for correct folding, plasma membrane targeting, and regulated exocytosis "
            "of synaptic vesicles via the SNARE complex (STX1A/SNAP25/VAMP2). "
            "Missense variants cluster in three functional domains: "
            "(1) Domain 1 (aa 1-100): syntaxin-binding groove — variants disrupt STX1A-closed "
            "conformation stabilisation; e.g., G544D (most frequent globally, ~8% of all cases), "
            "R406C/H, P480L. "
            "(2) Hydrophobic core: protein folding stability — M443R, V84D; protein is degraded "
            "before reaching the active zone. "
            "(3) C-terminal regulatory: vesicle priming — I328T, Y473H. "
            "Haploinsufficiency model: one wild-type STXBP1 allele is insufficient to supply "
            "adequate Munc18-1 for the axon terminal exocytic machinery, particularly in "
            "GABAergic interneurons which fire at high frequency and have the highest Munc18-1 "
            "demand. Consequence: preferential loss of inhibitory synaptic transmission → "
            "excitation/inhibition (E/I) imbalance → neonatal seizures. "
            "Mouse model (Stxbp1+/-): reproduces neonatal seizures, tremor, and hyperactivity; "
            "epilepsy-of-infancy-with-migrating-focal-seizures (EIMFS) pattern. "
            "Genotype-phenotype: G544D shows classic Ohtahara-to-West evolution with profound "
            "global DD; P480L may have relatively better motor outcome; M443R/V84D correlate "
            "with early lethality in severe cases."
        ),
        "eeg_correlate": (
            "Neonatal period (0-1 month): Burst-suppression (BS) pattern — high-amplitude "
            "poly-spike-wave bursts (200-600 µV, duration 2-8 s) alternating with near-isoelectric "
            "suppression intervals (< 5 µV). The BS in STXBP1 is typically ASYNCHRONOUS between "
            "hemispheres (unlike the hemisynchronous BS of KCNQ2-GOF) — a useful discriminating "
            "feature. Seizures: diffuse low-voltage fast activity (LVFA, 15-25 Hz) or tonic "
            "stiffening with EEG flattening (electrodecrement). "
            "Infantile spasms phase (2-8 months): chaotic high-amplitude disorganised "
            "background — modified hypsarrhythmia (the typical synchronous hypsarrhythmia of "
            "cryptogenic West is often replaced by fragmented multifocal HVS with asymmetric "
            "inter-burst suppression in STXBP1-West). Ictal: electrodecrement + slow wave "
            "following each spasm cluster. "
            "Childhood (> 1 year): THETA HYPERSYNCHRONY in wakefulness (4-6 Hz generalised "
            "rhythmic theta) — highly characteristic of STXBP1-DEE and rarely seen in other "
            "genetic epilepsies. Multifocal independent spike-wave complexes (frontal, central, "
            "occipital); generalised spike-wave (2-2.5 Hz) in LGS-like evolution; paroxysmal "
            "fast activity in NREM sleep. Sleep EEG: continuous spike-wave in slow-wave sleep "
            "(CSWS) in ~20% — consider electrical status epilepticus in sleep (ESES) monitoring."
        ),
        "mri_finding": (
            "Brain MRI in de novo STXBP1 missense (LOF): "
            "(1) Often NORMAL on structural MRI in 40-50% at initial scan — do NOT use normal MRI "
            "to exclude STXBP1-DEE; genetic testing is mandatory. "
            "(2) Delayed / reduced myelination: T2 hyperintensity in posterior periventricular "
            "white matter and centrum semiovale — progressive lag vs. age-matched norms; "
            "seen in ~35% at 6-18 months. "
            "(3) Thin corpus callosum: hypoplasia of body and genu — ~30% of cases, correlates "
            "with severity of cognitive disability. "
            "(4) Diffuse cerebral atrophy: widened sulci + dilated ventricles on serial MRI "
            "at 2-5 years in severe cases (~25%). "
            "(5) Cerebellar atrophy: vermis > hemispheres, particularly in STXBP1-DEE with "
            "truncating variants — ~15%. Reflects high STXBP1 expression in Purkinje cells. "
            "Spectroscopy: normal NAA/Cr in most; mild lactate elevation in severe acute "
            "episodes not expected (helps exclude mitochondrial). "
            "MRI evolution: serial imaging every 12-24 months recommended to quantify "
            "white matter maturation lag and atrophy progression."
        ),
        "clinical_note": (
            "G544D is the most frequent single pathogenic variant (~8% of all STXBP1-DEE); "
            "screen all unsolved early-onset DEE with BS pattern. "
            "Missense LOF: start rapid genomic workup (WES with trio if possible) within 48h "
            "of neonatal seizure presentation — targeted gene panel (early DEE panel including "
            "STXBP1/KCNQ2/SCN2A/CDKL5/ARX/MECP2) is faster in resource-limited settings. "
            "Phenobarbital as bridging AED; transition to ACTH when spasms emerge."
        ),
    },
    {
        "etiology": "De novo STXBP1 truncating (nonsense / frameshift)",
        "n": 14, "pct": 34,
        "category": "De-novo-STXBP1-truncating-nonsense-frameshift",
        "mechanism": (
            "Second most common class (~34%): de novo nonsense and frameshift variants producing "
            "premature stop codons and loss of the C-terminal regulatory domain of Munc18-1. "
            "These variants trigger nonsense-mediated mRNA decay (NMD) — the truncated transcript "
            "is degraded, resulting in TRUE haploinsufficiency (50% reduction in total STXBP1 "
            "protein). Common truncating variants: R388X (exon 11), Q374X (exon 10), c.1217delC "
            "(frameshift, exon 9), c.1690delA (frameshift, exon 14). "
            "Phenotype: typically SEVERE — neonatal Ohtahara-like presentation with burst-suppression, "
            "rapid progression to infantile spasms, profound global developmental disability (GDD), "
            "non-verbal, non-ambulatory. Mortality ~5% in first year from seizure-related "
            "respiratory failure or sudden unexpected death in epilepsy (SUDEP). "
            "NMD rate: ~80-90% of truncating variants undergo NMD — confirms true "
            "haploinsufficiency; allele-specific expression analysis available in research setting. "
            "Therapeutic implication: protein restoration strategy (upregulation of wild-type "
            "allele via chaperone rescue — OP-01; or antisense oligonucleotide exon-skipping "
            "to restore reading frame) being studied; currently no approved therapy."
        ),
        "eeg_correlate": (
            "Truncating variants typically produce the MOST SEVERE EEG phenotype: "
            "Neonatal: near-continuous suppression-burst with minimal inter-burst activity; "
            "periodic epileptiform discharges (PLEDs-like) on burst components. "
            "Clinically: 'seizure at birth' or within the first 6 hours of life — earlier than "
            "the 48-72h typical of KCNQ2-GOF. "
            "Evolution to hypsarrhythmia: faster than missense LOF — frequently by 6-8 weeks. "
            "Modified hypsarrhythmia: asynchronous, fragmented, with interposed focal spikes. "
            "LGS-like pattern (slow spike-wave < 2.5 Hz + paroxysmal fast in sleep) by 3-5 years "
            "in ~40% of truncating cases. "
            "EEG prognostic biomarker: absence of sleep spindles by 12 months correlates with "
            "severe global DD; spindle presence (even rudimentary) suggests better thalamocortical "
            "connectivity and relatively better prognosis."
        ),
        "mri_finding": (
            "Truncating STXBP1 variants: "
            "(1) Higher frequency of structural abnormalities than missense: thin corpus callosum "
            "in ~45%, reduced white matter volume in ~40%. "
            "(2) Pontine hypoplasia in ~10% — reflects STXBP1 expression in brainstem nuclei. "
            "(3) Progressive cerebral atrophy on serial MRI — most prominent in the frontal lobes "
            "and corpus callosum by 3-5 years. "
            "(4) Cavum septum pellucidum / vergae in ~15% (non-specific congenital variant). "
            "(5) Rarely (< 5%): periventricular nodular heterotopia has been reported — if present, "
            "consider FLNA / ARFGEF2 in parallel testing."
        ),
        "clinical_note": (
            "Truncating variants: onset may be < 6 hours of life. "
            "Neonatal resuscitation team readiness; NICU admission standard. "
            "NG tube feeding essential from day 1 — sucking/swallowing impaired. "
            "Genetic counselling: de novo in ~95% truncating; recurrence risk < 1% for parents "
            "(somatic/germline mosaicism counselling — mosaic parent rate ~2%)."
        ),
    },
    {
        "etiology": "De novo STXBP1 large deletion / duplication (9q34 CNV)",
        "n": 5, "pct": 12,
        "category": "De-novo-STXBP1-large-deletion-CNV-9q34",
        "mechanism": (
            "Third class (~12%): submicroscopic chromosomal deletions or duplications at 9q34.11 "
            "encompassing the entire STXBP1 gene, often including flanking genes. "
            "Detected by chromosomal microarray (CMA) / array-CGH — NOT by standard gene sequencing. "
            "Key size categories: "
            "(a) Isolated STXBP1 deletion (< 200 kb): pure haploinsufficiency phenotype — "
            "same as truncating variant; CDD/Ohtahara-to-West pattern. "
            "(b) 9q34.11 intermediate deletion (200 kb - 2 Mb): encompasses STXBP1 + SPTAN1 "
            "(αII-spectrin) + other genes; co-deletion of SPTAN1 (which causes early-onset "
            "epileptic encephalopathy independently) produces the MOST SEVERE phenotype — "
            "hypomyelination + profound GDD + dysmorphic facial features (wide nasal bridge, "
            "epicanthal folds, large ears). "
            "(c) 9q34.11-q34.3 large deletion (> 2 Mb): multiple genes deleted; severe GDD, "
            "dysmorphic features, congenital heart disease (CHD) in some — resembles 9q34 "
            "deletion syndrome (Kleefstra-like without EHMT1 involvement). "
            "Duplication of 9q34.11: gain of STXBP1 function — paradoxical LOF via dominant- "
            "negative effect on SM-protein stoichiometry; rare, limited case series. "
            "CMA first-line investigation in all STXBP1-negative DEE with BS — before expensive "
            "WES; cost-effective in resource-limited settings."
        ),
        "eeg_correlate": (
            "Large CNV deletions (co-deletion of SPTAN1): most severe EEG — "
            "Burst-suppression with minimal burst complexity (nearly flat between bursts); "
            "STXBP1+SPTAN1 co-deletion: hypomyelination-associated EEG — very low background "
            "amplitude with superimposed multi-focal very-fast spikes. "
            "Isolated STXBP1 deletion: EEG indistinguishable from truncating variant sequence. "
            "9q34.11 deletion syndrome with additional genes: LGS-like by 2-3 years; "
            "paroxysmal fast activity in NREM more prominent than in isolated STXBP1."
        ),
        "mri_finding": (
            "Large CNV 9q34.11: "
            "(1) SPTAN1 co-deletion: HYPOMYELINATION (severe global) — classic T2 hyperintensity "
            "across all white matter at 6-12 months; pontine T2 signal abnormality; "
            "vermian cerebellar hypoplasia. Distinct from the partial/patchy delayed myelination "
            "of isolated STXBP1 deletion. "
            "(2) Isolated STXBP1 deletion: delayed myelination + thin CC (same as truncating). "
            "(3) Large 9q34.11-q34.3 deletion: structural abnormalities more common — "
            "simplified gyral pattern, cortical thinning, rare periventricular lissencephaly "
            "if additional lissencephaly-pathway genes included in deletion."
        ),
        "clinical_note": (
            "Mandatory: CMA (chromosomal microarray) in all STXBP1-sequencing-negative DEE; "
            "also in dysmorphic neonates with seizures + delayed myelination on MRI. "
            "SPTAN1 co-deletion: refer to paediatric neurology + genetics + ophthalmology "
            "(cortical visual impairment common). Prognosis counselling: isolated STXBP1 del "
            "vs. large 9q34 syndrome diverge significantly."
        ),
    },
    {
        "etiology": "De novo STXBP1 splice-site variant",
        "n": 3, "pct": 7,
        "category": "De-novo-STXBP1-splice-site",
        "mechanism": (
            "Fourth class (~7%): splice-site variants (canonical ±1/±2 positions or deep intronic "
            "with cryptic splice activation) disrupting pre-mRNA splicing of STXBP1. "
            "Canonical splice-site variants (IVS-1, IVS-8 5' donor; IVS-11 3' acceptor): "
            "trigger exon skipping or intron retention — produce out-of-frame transcripts, "
            "NMD, and functionally equivalent to truncating LOF. "
            "Deep intronic variants (e.g., c.1217+43C>T, c.516+89G>A): activate cryptic "
            "donor/acceptor sites, insert pseudo-exons into the mRNA with premature stop codons. "
            "These variants are MISSED by standard clinical exome sequencing (which reads exons "
            "± 10-20 bp flanking); require RNA-seq or long-read whole-genome sequencing (WGS) "
            "for detection. Important diagnostic implication: ~5-10% of unsolved STXBP1-DEE "
            "with a convincing clinical/EEG phenotype harbour deep intronic variants detectable "
            "only by RNA-seq. RT-PCR on patient-derived fibroblasts/lymphoblastoid cells "
            "confirms aberrant splicing. "
            "Phenotype: variable — depends on degree of functional transcript loss; "
            "partial splice variants with residual correct splicing show MILDER phenotype "
            "(later onset, lower seizure burden, better verbal acquisition) vs. complete "
            "exon skipping (equivalent to truncating/severe)."
        ),
        "eeg_correlate": (
            "Splice-site variants with residual splicing: attenuated BS pattern at birth — "
            "may show only suppression-burst in sleep (not continuous waking BS); "
            "onset at 2-6 weeks rather than day 1. "
            "Canonical splice loss (complete exon skip / NMD): EEG indistinguishable from "
            "truncating variant — continuous BS, then classic modified hypsarrhythmia. "
            "Mild splice variants: may first present as FOCAL EPILEPSY (temporal-onset spikes "
            "with secondary generalisation) without a clear BS phase — can mimic other focal "
            "genetic epilepsies; RNA-seq diagnosis in unsolved DRE."
        ),
        "mri_finding": (
            "Splice-site variants: "
            "Mild/partial: near-normal MRI or only subtle delayed myelination at 12 months. "
            "Severe (complete NMD): same as truncating — thin CC, reduced WM volume, atrophy. "
            "Deep intronic (RNA-seq-diagnosed): MRI pattern correlates with phenotypic severity; "
            "typically less severe than canonical LOF on structural neuroimaging."
        ),
        "clinical_note": (
            "Order RNA-seq / long-read WGS when: clinical phenotype strongly suggests STXBP1-DEE "
            "but standard panel/exome is negative. "
            "Functional splicing assay in research lab can confirm pathogenicity of VUS at "
            "splice consensus before reclassifying. "
            "Genetic counselling: splice-site variants with residual function may have "
            "better reproductive outcomes than typical severe DEE — nuanced counselling."
        ),
    },
    {
        "etiology": "Clinical STXBP1-DEE — STXBP1-negative (expanded panel pending)",
        "n": 1, "pct": 3,
        "category": "Clinical-STXBP1-DEE-STXBP1-negative",
        "mechanism": (
            "Residual clinical category (~3%): patients meeting clinical / EEG / MRI criteria "
            "for STXBP1 encephalopathy (neonatal BS → IS → GDD + theta hypersynchrony) but "
            "with a non-diagnostic STXBP1 gene panel / WES. "
            "Differential diagnoses to exclude: "
            "(a) STXBP1 deep intronic variant — RNA-seq required. "
            "(b) STXBP1 mosaic variant — ultra-deep sequencing (> 500×) of blood + possibly "
            "saliva/fibroblast. "
            "(c) Other DEE genes mimicking STXBP1-DEE: SCN2A (neonatal BS / Ohtahara), "
            "ARX (males, neonatal onset), KCNQ2-LOF (BS, but CBZ-responsive), SCN8A (BS, "
            "early-infantile), KCNT1 (MMPSI phenotype, migratory focal), "
            "ATP1A3 (CAPOS/AHC overlap), GRIN2A/B (later onset), DNM1 (spasms/GDD). "
            "(d) Metabolic mimics: pyridoxine-dependent epilepsy (ALDH7A1) — IV pyridoxine "
            "trial mandatory; pyridoxal-5'-phosphate responsive (PNPO); "
            "biotinidase deficiency; GLUT1 deficiency (CSF glucose ratio). "
            "(e) Structural: focal cortical dysplasia type IIb (spike-burst EEG early in life). "
            "Management: parallel metabolic screen + expanded panel / WGS; "
            "IV pyridoxine trial in NICU; epilepsy MRI protocol with 3T + thin-cut FLAIR; "
            "refer to specialist DEE multidisciplinary team (MDT)."
        ),
        "eeg_correlate": (
            "STXBP1-negative clinical DEE: EEG-driven differential — "
            "Continuous BS + tonic seizures: STXBP1-DEE > SCN2A > ARX > KCNQ2-LOF > structural FCD. "
            "Migratory focal seizures (EIMFS pattern): KCNT1 first; STXBP1 second. "
            "Pyridoxine trial (empiric, NICU): 100 mg IV — EEG recording mandatory during infusion; "
            "EEG normalisation within 30 min confirms PDX-DE (pyridoxine-dependent epilepsy). "
            "Pyridoxal-5'-phosphate (PLP) trial if pyridoxine negative: 30 mg/kg/day PO in 3 doses. "
            "Theta hypersynchrony in wakefulness — if present in STXBP1-negative, expand STXBP1 "
            "testing (RNA-seq/WGS) before pivoting to alternative diagnoses."
        ),
        "mri_finding": (
            "STXBP1-negative clinical DEE with Ohtahara-like EEG: "
            "MRI is KEY to differential: FCD (cortical thickening + T2 signal + blurring of GM-WM junction); "
            "structural cause (heterotopia, lissencephaly — ARX/LIS1 panel); "
            "metabolic (T2 signal in basal ganglia + diffuse WM — mitochondrial, organic aciduria). "
            "Normal MRI with Ohtahara + STXBP1-negative: priority to RNA-seq + deep intronic WGS. "
            "Defer STXBP1 gene therapy trial enrolment until genetic diagnosis confirmed."
        ),
        "clinical_note": (
            "Mandatory NICU workup when STXBP1-negative: IV pyridoxine (100 mg) trial with EEG; "
            "CSF glucose + pyruvate (GLUT1/PDH); plasma amino acids + urine organic acids; "
            "biotinidase + plasma carnitine; karyotype + CMA. "
            "If all negative: WGS with RNA-seq co-analysis at specialist DEE centre."
        ),
    },
]

# ── Seizure Types ────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Tonic / Tonic-Clonic Neonatal Seizures",
        "prevalence_pct": 95,
        "eeg_correlate": (
            "Electrodecrement (sudden attenuation of background) followed by diffuse low-voltage "
            "fast activity (LVFA, 15-25 Hz) evolving over 5-20 seconds; may have ictal "
            "tachycardia (autonomic). Clinically: tonic extension / stiffening ± subtle clonic "
            "jerking. Electroclinical dissociation (ECD) common after phenobarbital loading — "
            "EEG seizures continue without clinical manifestation; do NOT equate clinical "
            "seizure cessation with EEG seizure cessation in STXBP1-DEE neonates."
        ),
        "clinical_tip": (
            "EEG monitoring (aEEG or full 10-20 cEEG) mandatory in NICU for STXBP1-DEE — "
            "electroclinical dissociation occurs in > 60% after PB loading. "
            "Treat to EEG seizure suppression, not just clinical cessation. "
            "Phenobarbital 20 mg/kg IV loading dose; repeat bolus 10 mg/kg if EEG seizures persist "
            "after 20 min (total max 40 mg/kg in 24h). "
            "Second-line: levetiracetam 40-60 mg/kg IV or phenytoin 20 mg/kg IV (PHT: limited "
            "STXBP1 evidence; use if LEV unavailable). "
            "Benzodiazepine (midazolam 0.1 mg/kg IV) as third-line bridge; clonazepam "
            "0.05 mg/kg IV for refractory."
        ),
    },
    {
        "type": "Infantile Spasms (West Syndrome / IS Phase)",
        "prevalence_pct": 70,
        "eeg_correlate": (
            "Modified hypsarrhythmia in STXBP1-IS: high-amplitude (> 200 µV) chaotic "
            "multifocal spike-wave activity with fragmented, asynchronous inter-burst intervals "
            "(less synchronous than classical cryptogenic hypsarrhythmia). "
            "Ictal: generalised electrodecrement (amplitude attenuation) coinciding with "
            "each spasm; electrodecrement preceded by high-amplitude slow transient. "
            "Spasm series: 5-30 spasms in a cluster, typically on awakening or at "
            "sleep-wake transition. "
            "Hypsarrhythmia variant: 'modified' hypsarrhythmia — may have sustained focal "
            "discharge superimposed; high-amplitude interictal discharge in inter-spasm intervals."
        ),
        "clinical_tip": (
            "ACTH (Tetracosactide / Synacthen): first-line for STXBP1-associated IS — UKISS "
            "protocol: 0.5 mg/kg/day IM for 2 weeks then taper. Target: hypsarrhythmia "
            "resolution within 14 days (EEG criterion). "
            "Vigabatrin (VGB) co-administration in STXBP1-IS: UKISS protocol (VGB + ACTH) "
            "recommended over ACTH alone — higher rate of hypsarrhythmia resolution. "
            "VGB mandatory SHARE REMS enrolment (FDA); Goldman visual field perimetry q3 months. "
            "Ketogenic diet (KD) early: consider KD initiation if ACTH + VGB insufficient by "
            "4 weeks — KD + VGB combination has level B evidence in refractory IS. "
            "Clinical alert: Prednisolone can be used as alternative to ACTH if IM injection "
            "is not tolerated; 40 mg/day PO (UK Child's UKISS Prednisolone arm)."
        ),
    },
    {
        "type": "Focal Seizures (multifocal / migratory in neonates)",
        "prevalence_pct": 65,
        "eeg_correlate": (
            "Focal onset in neonates: rhythmic discharge (delta or alpha-band, 2-8 Hz) starting "
            "unilaterally (frontal or temporal), evolving to spread across homologous "
            "contralateral regions within 5-30 seconds — 'migrating' pattern in neonates "
            "mimics KCNT1-EIMFS but less stereotyped and shifts are less frequent. "
            "In infancy: multifocal independent spike-wave complexes on interictal EEG; "
            "focal seizures from any region. "
            "In childhood (> 1 year): theta hypersynchrony background + multifocal focal "
            "seizures that secondarily generalise; frontal-onset most common."
        ),
        "clinical_tip": (
            "Focal seizures in STXBP1-DEE: levetiracetam (LEV) 20-60 mg/kg/day adjunct — "
            "SV2A mechanism; some evidence for GABAergic synapse restoration in Stxbp1+/- mice. "
            "Clobazam (CLB) as add-on for focal clusters. "
            "Avoid high-dose sodium channel blockers (CBZ/OXC) as monotherapy — limited "
            "evidence in STXBP1; may worsen myoclonic component."
        ),
    },
    {
        "type": "Myoclonic Seizures (axial / massive)",
        "prevalence_pct": 40,
        "eeg_correlate": (
            "Generalised poly-spike-wave (GSW) at 3-4 Hz, bilaterally synchronous, "
            "discharge duration 0.5-3 seconds per myoclonic jerk. "
            "Negative myoclonus (atonia interrupting sustained posture) in ~10% — "
            "associated with centroparietal spike-wave. "
            "Myoclonic-atonic seizures in ~15% of STXBP1-DEE (Doose-like evolution) — "
            "sudden drop attacks; high injury risk."
        ),
        "clinical_tip": (
            "Myoclonic seizures in STXBP1-DEE: valproate (VPA) AVOID — POLG exclusion mandatory "
            "first; if POLG excluded, VPA 20-30 mg/kg/day may reduce myoclonus but "
            "risk-benefit vs. KD. "
            "Clobazam (CLB) 0.1-0.5 mg/kg/day — effective for myoclonic attenuation. "
            "Clonazepam (CLN) 0.05-0.2 mg/kg/day — specifically for myoclonic type. "
            "Ketogenic diet: highly effective for myoclonic-atonic type (Doose-like STXBP1)."
        ),
    },
]

# ── Seizure Triggers ─────────────────────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Fever / febrile illness", "prevalence_pct": 85,
     "note": "Most common — even low-grade fever (37.5°C) sufficient; "
             "aggressive antipyretic protocol (paracetamol at 37.5°C); rescue plan with "
             "diazepam rectal gel 0.5 mg/kg or buccal midazolam 0.3 mg/kg at onset."},
    {"trigger": "Intercurrent illness (viral / bacterial)", "prevalence_pct": 75,
     "note": "Illness-related breakthrough in 75%; AED compliance is most important factor; "
             "oral AED suspension formulations mandatory for sick children who cannot swallow."},
    {"trigger": "Missed / late AED dose", "prevalence_pct": 70,
     "note": "Strict twice-daily dosing schedule essential; written emergency protocol; "
             "hospital NPO policy MUST include IV/NG AED continuation — STXBP1-DEE "
             "patients MUST NOT fast without AED route established."},
    {"trigger": "Sleep deprivation / disrupted sleep-wake cycle", "prevalence_pct": 60,
     "note": "Sleep EEG changes (CSWS) in ~20%; overnight seizures in NREM common. "
             "Melatonin for sleep consolidation (0.5-5 mg nocte); consistent sleep schedule."},
    {"trigger": "Hyperthermia (bath / environment)", "prevalence_pct": 50,
     "note": "Warm baths trigger tonic/clonic seizures; lukewarm bath only (< 37°C); "
             "air-conditioned environment in summer; avoid prolonged heat exposure."},
    {"trigger": "Physical / procedural stimulation", "prevalence_pct": 35,
     "note": "Venepuncture, NG insertion, physiotherapy can trigger seizures in neonates; "
             "pre-procedural benzodiazepine cover if high seizure burden."},
    {"trigger": "Rapid AED dose reduction / change", "prevalence_pct": 30,
     "note": "Never abrupt — taper any AED change over minimum 4-6 weeks; "
             "ACTH taper must be gradual (> 6 weeks after hypsarrhythmia resolution)."},
    {"trigger": "Puberty / hormonal change", "prevalence_pct": 20,
     "note": "Catamenial exacerbation reported in adolescent females with STXBP1-DEE; "
             "clobazam pulsed therapy may attenuate perimenstrual clusters."},
]

# ── Treatment Catalog ────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Phenobarbital (PB)",
        "evidence": "Level B — First-line neonatal acute seizure control",
        "dose": "Loading: 20 mg/kg IV over 20 min; Maintenance: 3-5 mg/kg/day (neonates 4-6 mg/kg)",
        "moa": "GABA-A positive allosteric modulator (barbiturate); prolongs Cl- channel opening; "
               "reduces neuronal firing frequency. Na-channel block at high concentrations.",
        "efficacy": (
            "First-line for acute neonatal seizures in STXBP1-DEE — achieves EEG seizure "
            "suppression in ~50-60% of cases (note: clinical suppression overestimates due "
            "to electroclinical dissociation). PB alone is RARELY sufficient long-term — "
            "adjunct AED required in > 90%."
        ),
        "safety": "Sedation (reduced arousal, feeding difficulty); respiratory depression at "
                  "high doses (cEEG to detect over-suppression); hypotension IV bolus; "
                  "long-term: dupuytren contracture, reduced bone density, enzyme induction "
                  "(CYP2C19/3A4 — reduces levels of co-administered AEDs).",
        "monitoring": "TDM: target 20-40 µg/mL (neonates); 10-40 µg/mL (older); "
                      "respiratory rate monitoring during IV loading; AST/ALT at 6 months.",
    },
    {
        "drug": "ACTH (Tetracosactide / Synacthen) — Infantile Spasms Phase",
        "evidence": "Level B — First-line for STXBP1-associated IS (UKISS protocol)",
        "dose": "UK: 0.5 mg (40 IU) IM on alternate days × 2 weeks, then taper; "
                "USA: Natural ACTH 150 IU/m²/day IM × 2 weeks; "
                "UK prednisolone alternative: 40 mg/day PO in 3 divided doses × 14 days",
        "moa": "Adrenocorticotropic hormone: steroidogenesis stimulation (cortisol); "
               "direct direct melanocortin receptor (MC4R) action in brain — reduces "
               "CRH-induced hippocampal hyperexcitability; reduces ACTH/CRH-mediated "
               "spasm circuits; upregulates GABA-A expression.",
        "efficacy": (
            "UKISS trial (Lux 2004, Lancet): ACTH + VGB superior to VGB alone at 14 days "
            "for hypsarrhythmia resolution (~73% vs ~54%). "
            "STXBP1-specific: hypsarrhythmia resolves in ~60% with ACTH; spasm freedom "
            "at 3 months in ~40-50% — lower than cryptogenic IS (~75%)."
        ),
        "safety": "Hypertension (monitor BP q day during treatment); Cushingoid features "
                  "(striae, moon face — transient, resolves after taper); hyperglycaemia; "
                  "immunosuppression (live vaccines contraindicated for 3 months); "
                  "irritability / insomnia; rare: cerebral haemorrhage (HTN-related).",
        "monitoring": "BP daily during treatment; blood glucose 6-hourly first week; "
                      "electrolytes weekly; infection screen monthly; "
                      "EEG at day 14 — criterion for treatment success.",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "evidence": "Level B — STXBP1-IS (combined UKISS protocol with ACTH); mandatory SHARE REMS",
        "dose": "IS: 50-150 mg/kg/day PO in 2 divided doses; "
                "max 3 g/day. Titrate up over 1-2 weeks.",
        "moa": "Irreversible GABA transaminase (GABA-T) inhibitor — prevents GABA catabolism; "
               "elevates brain GABA; reduces thalamocortical excitability and spasm circuits.",
        "efficacy": (
            "UKISS: ACTH + VGB combination superior to ACTH alone for hypsarrhythmia resolution "
            "at 14 days. VGB alone less effective than ACTH alone in IS. "
            "In STXBP1-DEE: VGB reduces spasm frequency in ~40-50%; full hypsarrhythmia "
            "resolution in ~30% on VGB alone (lower than TSC-IS ~65%)."
        ),
        "safety": "VISUAL FIELD DEFECT (VFD) — irreversible bilateral peripheral constriction "
                  "(nasal > temporal); cumulative dose-related; risk ~30-40% with prolonged use. "
                  "SHARE REMS MANDATORY (FDA) — Goldman perimetry every 3 months (from baseline); "
                  "OCT (optical coherence tomography) every 6 months (retinal nerve fibre layer). "
                  "Sedation; weight gain; MRI signal changes in basal ganglia / thalamus "
                  "(transient, reversible — vigabatrin-associated MRI abnormalities, VAMA — "
                  "do not misdiagnose as acute injury).",
        "monitoring": "Goldman VF q3 months (mandatory SHARE REMS); OCT q6 months; "
                      "VGB MRI abnormalities: T2 signal in globus pallidus / dorsal brainstem / "
                      "dentate nucleus — benign; monitor clinically.",
    },
    {
        "drug": "Ketogenic Diet (KD, 4:1 classic)",
        "evidence": "Level B — Highly effective for drug-resistant STXBP1-DEE (myoclonic / IS / multi-type)",
        "dose": "4:1 fat:protein+carbohydrate ratio; target BHB 2-4 mmol/L; "
                "hospitalised initiation recommended; MCT diet alternative for older children.",
        "moa": "Ketone bodies (BHB, AcAc) — multiple mechanisms: direct inhibition of "
               "voltage-gated Na+ channels (BHB); GABA-B receptor upregulation; "
               "purinergic (adenosine A1 receptor) anti-seizure effect; reduces glycolysis "
               "(decreases glucose availability for seizure propagation); "
               "Munc18-1 interaction: BHB may stabilise SNARE complex stoichiometry "
               "in Stxbp1+/- mouse model (mechanistic hypothesis, not yet clinical).",
        "efficacy": (
            "In STXBP1-DEE (observational cohort): 50-day response (> 50% seizure reduction) "
            "in ~55-65% of patients; seizure freedom in ~15-20%. "
            "Particularly effective for myoclonic-atonic type (Doose-like) and residual IS "
            "after failed ACTH/VGB. Level B (ILAE Dietary Therapies Commission 2018)."
        ),
        "safety": "GI side effects (nausea, constipation, reflux); growth deceleration on "
                  "prolonged KD (selenium / zinc / carnitine supplementation essential); "
                  "renal calculi risk (citrate supplementation 2-3 mEq/kg/day, liberal fluid); "
                  "hyperlipidaemia; rare: cardiomyopathy (selenium deficiency); "
                  "anticoagulation interaction (warfarin effect enhanced).",
        "monitoring": "BHB twice weekly (target 2-4 mmol/L); lipid panel q6 months; "
                      "selenium, zinc, carnitine, bone density (DEXA) annually; "
                      "renal USS annually; growth (weight/height) monthly.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "evidence": "Level C — Adjunct (focal seizures, tonic-clonic, broad-spectrum)",
        "dose": "20-60 mg/kg/day PO/IV in 2 divided doses; max 3 g/day",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) ligand — modulates vesicle fusion "
               "and neurotransmitter exocytosis; reduces presynaptic glutamate release. "
               "STXBP1 relevance: STXBP1 (Munc18-1) and SV2A both regulate the SNARE-mediated "
               "exocytosis machinery — LEV may partially compensate for Munc18-1 deficiency "
               "via complementary presynaptic modulation (pre-clinical only).",
        "efficacy": "Broad-spectrum adjunct; 20-40% responder rate in STXBP1-DEE focal "
                    "and generalised seizures. Rapid IV availability (critical in NICU); "
                    "well-tolerated in neonates (fewer CNS depression effects than PB).",
        "safety": "Behavioural side effects (irritability, agitation, aggression) in 10-20% "
                  "— use CBCL-6 questionnaire at 4 weeks; switch to brivaracetam if "
                  "LEV behavioural problems intolerable. Hyponatraemia rare. No enzyme "
                  "induction. Renal dose adjustment (eGFR < 30).",
        "monitoring": "Serum LEV level (if available): 12-46 mg/L; renal function 6-monthly; "
                      "behavioural assessment at 4 weeks post-initiation.",
    },
    {
        "drug": "Clobazam (CLB)",
        "evidence": "Level C — Adjunct (myoclonic / focal, tolerance management required)",
        "dose": "0.1-0.5 mg/kg/day PO in 2 divided doses; max 1 mg/kg/day (children)",
        "moa": "Benzodiazepine (1,5-BZD) — selective GABA-A α2/α3 subunit modulator (less "
               "sedating than 1,4-BZD); positive allosteric modulator increasing Cl- "
               "conductance. Less tolerance-prone than clonazepam.",
        "efficacy": "Effective adjunct for myoclonic and focal seizure clusters in STXBP1-DEE; "
                    "tolerance develops in ~30-40% within 6 months (drug holiday strategy: "
                    "3 months on / 1 month dose reduction may restore efficacy).",
        "safety": "Sedation; drooling; ataxia at high doses; paradoxical agitation (rare); "
                  "tolerance (less than clonazepam). Avoid abrupt withdrawal.",
        "monitoring": "Sedation scale (RASS or parental report); quarterly tolerance assessment; "
                      "drug holiday if tolerance detected.",
    },
    {
        "drug": "OP-01 (Munc18-1 chaperone) — Investigational",
        "evidence": "Phase 1 — First STXBP1-specific Munc18-1 protein restoration therapy",
        "dose": "Investigational — Phase 1 dose-escalation; eligibility: confirmed STXBP1 "
                "pathogenic variant, age 1-12y, ongoing seizures; refer to STXBP1 Foundation "
                "/ ClinicalTrials.gov for active trials.",
        "moa": "Small-molecule pharmacological chaperone — binds misfolded Munc18-1 protein "
               "(products of missense LOF variants), stabilising its tertiary structure, "
               "preventing proteasomal degradation, and restoring correct trafficking to "
               "the active zone. Increases total functional Munc18-1 protein towards "
               "haploinsufficiency threshold. Pre-clinical (Stxbp1+/- mouse + iPSC-neuron): "
               "OP-01 significantly reduces seizure frequency and partially restores "
               "inhibitory synaptic transmission. Does NOT apply to truncating / NMD variants "
               "(no protein produced to fold).",
        "efficacy": "Phase 1 (safety/dose-finding) — no efficacy data available as of 2026. "
                    "Refer eligible patients (missense LOF variants only) to STXBP1 Foundation "
                    "patient registry for trial access.",
        "safety": "Phase 1 safety profile pending — no known toxicology signal from "
                  "pre-clinical programme.",
        "monitoring": "Clinical trial protocol monitoring (trial-specific).",
    },
    {
        "drug": "Phenytoin (PHT) — CAUTION / Relative Contraindication",
        "evidence": "Level C — Limited STXBP1 evidence; may WORSEN myoclonic component",
        "dose": "If used: 20 mg/kg IV loading (max 1 g); 5-8 mg/kg/day PO maintenance; "
                "TDM target 10-20 µg/mL. USE ONLY as third-line bridge in NICU if "
                "PB + LEV insufficient and other agents unavailable.",
        "moa": "Na-channel blocker (voltage-dependent inactivation); reduces high-frequency "
               "burst firing; no GABA/glutamate effect.",
        "efficacy": "May reduce tonic/tonic-clonic seizures in NICU but UNRELIABLE in STXBP1; "
                    "no data for long-term maintenance in STXBP1-DEE; generally avoided after "
                    "NICU stabilisation.",
        "safety": "CARDIAC: bradyarrhythmia / AV block during IV bolus (rate limit: "
                  "< 1 mg/kg/min); gingival hypertrophy (long-term); "
                  "ENZYME INDUCER (reduces levels of CBZ, CLB, KD, other AEDs); "
                  "cerebellar atrophy with chronic use; SJS/TEN (rare).",
        "monitoring": "Cardiac monitor mandatory during IV loading; TDM (10-20 µg/mL); "
                      "ECG if bradycardia.",
    },
]

# ── Absolute Contraindications ───────────────────────────────────────────────
ABSOLUTE_CONTRAINDICATIONS = [
    {
        "drug": "Valproate (VPA) — WITHOUT POLG EXCLUSION",
        "severity": "ABSOLUTE CI (until POLG excluded)",
        "reason": (
            "Valproate in POLG-positive patients causes fatal hepatotoxicity (Alpers-Huttenlocher "
            "syndrome). POLG gene mutations can present with epilepsy indistinguishable from "
            "STXBP1-DEE. POLG exclusion (molecular testing — POLG panel or WES) is MANDATORY "
            "before administering VPA to any child with unexplained DEE / Ohtahara / IS. "
            "Once POLG excluded: VPA may be used cautiously for myoclonic seizures (20-30 mg/kg/day) "
            "but KD is preferred in STXBP1-DEE."
        ),
    },
    {
        "drug": "Vigabatrin (VGB) without SHARE REMS enrolment",
        "severity": "REGULATORY CI (USA — FDA mandate)",
        "reason": (
            "VGB carries FDA Black Box Warning for permanent bilateral concentric visual field "
            "defect in up to 30-40% of patients. SHARE REMS (Risk Evaluation and Mitigation "
            "Strategy) enrolment is MANDATORY in the USA before prescribing VGB — "
            "without enrolment, prescription is illegal. Goldman perimetry q3 months and "
            "OCT q6 months mandatory throughout treatment course."
        ),
    },
    {
        "drug": "Hospital NPO (nil per os) without IV/NG AED continuation",
        "severity": "ABSOLUTE OPERATIONAL CI",
        "reason": (
            "STXBP1-DEE patients on oral AED regimen who are made nil-by-mouth for surgery / "
            "procedures without IV or NG-tube AED administration are at very high risk of "
            "status epilepticus and SUDEP. Hospital anaesthesia/surgery teams MUST have a "
            "written AED substitution protocol: PB IV substitution for oral PB; LEV IV for "
            "oral LEV; CLB crushed via NG tube; KD formula via NG tube (do NOT switch off KD "
            "without neurologist review — seizure breakthrough risk within 24-48h)."
        ),
    },
    {
        "drug": "Live vaccines during ACTH / steroid course",
        "severity": "ABSOLUTE CI during immunosuppression",
        "reason": (
            "ACTH (tetracosactide) and corticosteroids cause iatrogenic immunosuppression. "
            "Live attenuated vaccines (MMR, rotavirus, varicella, BCG, yellow fever) are "
            "ABSOLUTELY contraindicated during ACTH course and for 3 months after completion. "
            "Inactivated vaccines (pneumococcal, meningococcal, influenza) may proceed — "
            "but efficacy may be reduced. Catch-up vaccination schedule required after "
            "immunosuppression window."
        ),
    },
]

# ── AED Monitoring ───────────────────────────────────────────────────────────
AED_MONITORING = [
    {"item": "Phenobarbital TDM", "target": "20-40 µg/mL (neonates); 10-40 µg/mL (children)",
     "frequency": "At 3-5 days; then every 3-6 months; after dose change",
     "rationale": "Narrow TI neonates; monitor for toxicity (sedation) and enzyme induction"},
    {"item": "VGB Goldman Perimetry (SHARE REMS)", "target": "No progressive VFD",
     "frequency": "Baseline before VGB; then every 3 months throughout",
     "rationale": "FDA-mandated for irreversible peripheral VFD — cumulative dose risk"},
    {"item": "VGB OCT (optical coherence tomography)", "target": "RNFL thickness stable",
     "frequency": "Baseline + every 6 months", "rationale": "RNFL thinning = subclinical VFD marker"},
    {"item": "KD Ketone monitoring (BHB)", "target": "2-4 mmol/L",
     "frequency": "Twice weekly (home urine ketone strips or blood BHB meter)",
     "rationale": "Confirm ketosis — < 2 mmol/L often not therapeutic; > 5 mmol/L excess risk"},
    {"item": "KD Micronutrients (selenium, zinc, carnitine)", "target": "Within normal reference",
     "frequency": "Annually", "rationale": "KD restriction causes deficiencies → cardiomyopathy (Se); immune (Zn)"},
    {"item": "EEG (sleep + wake, 2h)", "target": "Seizure burden reduction; hypsarrhythmia resolution",
     "frequency": "At 14 days of ACTH (UKISS criterion); then every 6-12 months",
     "rationale": "EEG criterion for ACTH success: hypsarrhythmia resolution by day 14"},
    {"item": "ACTH Blood pressure", "target": "< age-95th centile",
     "frequency": "Daily during ACTH course", "rationale": "ACTH-induced hypertension — risk of cerebral haemorrhage"},
    {"item": "Developmental assessment (Bayley-III / Griffiths)", "target": "Track trajectory",
     "frequency": "6-monthly (age 0-3y); annually thereafter",
     "rationale": "STXBP1-DEE: all domains severely affected; trajectory informs KD/ACTH response"},
]

# ── Lifecycle Windows ─────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {"window": "Neonatal / NICU", "age_range": "0–28 days",
     "focus": "Burst-suppression; PB loading; EEG monitoring (cEEG mandatory); feeding support; "
              "rapid genetic testing (panel or WES trio); POLG exclusion; metabolic screen",
     "key_action": "Initiate PB + LEV; cEEG to electroclinical dissociation; rapid WES trio"},
    {"window": "Early Infantile (IS phase)", "age_range": "1–6 months",
     "focus": "Infantile spasms onset; ACTH + VGB UKISS protocol; EEG day 14 assessment; "
              "early KD consideration if ACTH/VGB insufficient; VGB SHARE REMS enrolment",
     "key_action": "ACTH + VGB; EEG day 14; SHARE REMS enrolment; KD if ACTH/VGB fail"},
    {"window": "Late Infantile", "age_range": "6–18 months",
     "focus": "Post-IS transition; multifocal epilepsy; KD optimisation; "
              "theta hypersynchrony emergence; physiotherapy + OT input; "
              "ACTH taper complete; VGB duration decision (VFD monitoring)",
     "key_action": "KD continuation; OT/PT/SLP referral; VGB VFD monitoring q3 months"},
    {"window": "Early Childhood", "age_range": "18 months–5 years",
     "focus": "Drug-resistant multifocal epilepsy; LGS-like evolution in some; "
              "CLB / LEV / KD combination; movement disorder (tremor, ataxia); "
              "feeding assessment (G-tube decision); epilepsy surgery evaluation (VNS)",
     "key_action": "VNS referral if 2+ AED failures + KD; G-tube consideration; LGS protocol"},
    {"window": "School Age", "age_range": "5–12 years",
     "focus": "Severe GDD — STXBP1-DEE non-verbal in 70%; non-ambulatory in 40%; "
              "specialist special needs education; respite care; SUDEP counselling; "
              "annual EEG + developmental review; catamenial exacerbation monitoring (girls)",
     "key_action": "SUDEP counselling; nocturnal alarm; rescue BZD plan; special needs education"},
    {"window": "Adolescence / Adult", "age_range": "12+ years",
     "focus": "Transition to adult neurology; reproductive counselling (< 5% recurrence risk); "
              "seizure management stabilisation; social care planning; "
              "consider OP-01 / STXBP1 gene therapy trials (if available); "
              "driving exclusion (STXBP1-DEE — permanent exclusion in most jurisdictions)",
     "key_action": "Adult transition; driving exclusion; STXBP1 Foundation registry enrolment"},
]

# ── Clinical Standards ────────────────────────────────────────────────────────
STANDARDS = [
    {"std": "ILAE 2022", "title": "Classification of Seizures and Epilepsy Syndromes",
     "note": "STXBP1-DEE: classified as DEE (developmental + epileptic); EIEE-4 (MIM #612164)"},
    {"std": "NICE NG217 2022", "title": "Epilepsies: Diagnosis and Management (UK)",
     "note": "Genetic DEE pathway; ACTH for IS; SUDEP §1.15; specialist referral within 4 weeks of IS onset"},
    {"std": "UKISS Trial 2004", "title": "Lux AL, Lancet 2004 — ACTH vs VGB for IS",
     "note": "Combination ACTH + VGB recommended (UK) — superior hypsarrhythmia resolution vs. monotherapy"},
    {"std": "FDA SHARE REMS (Vigabatrin)", "title": "Sabril REMS Program",
     "note": "Mandatory enrolment before VGB prescription in USA; Goldman VF q3M; OCT q6M"},
    {"std": "ILAE Dietary Therapies Commission 2018", "title": "KD Practice Guideline",
     "note": "KD level B for Dravet / DEE / Doose; hospitalised initiation; nutrition monitoring"},
    {"std": "ACMG-AMP 2015", "title": "Variant Classification Standards",
     "note": "Pathogenicity classification for STXBP1 variants; P/LP for confirmed DEE association"},
    {"std": "ACNS Neonatal EEG Guidelines 2011/2021", "title": "Continuous EEG in Neonates",
     "note": "cEEG mandatory in STXBP1-DEE NICU; electroclinical dissociation monitoring"},
    {"std": "EAN Neonatal SE Guidelines 2019", "title": "Treatment of Neonatal SE",
     "note": "PB first-line; LEV second; PHT third-line bridge; address electroclinical dissociation"},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "Seizure onset < 5 days of life", "action": "Trigger rapid WES trio / neonatal DEE panel (STXBP1/KCNQ2/SCN2A/ARX)"},
    {"threshold": "PB TDM 20-40 µg/mL (neonates)", "action": "Target for EEG seizure suppression; > 40 = toxicity (sedation, respiratory)"},
    {"threshold": "LEV TDM 12-46 mg/L", "action": "Reference range (adjust if poor response or behavioural toxicity)"},
    {"threshold": "ACTH day 14 EEG (UKISS)", "action": "Criterion: hypsarrhythmia resolution = treatment success; continue 4-week course then taper"},
    {"threshold": "VGB duration decision (VFD risk)", "action": "If IS in remission × 12-24 months: discuss VGB taper to reduce cumulative VFD risk"},
    {"threshold": "KD BHB 2-4 mmol/L", "action": "Therapeutic ketone range; < 2 = insufficient; > 5 = excessive (acidosis risk)"},
    {"threshold": "2 AED failures → KD discussion", "action": "After 2 AED failure (ILAE DRE definition): formally offer KD trial"},
    {"threshold": "VNS after 3 AED failures", "action": "VNS referral: consider after 3+ AED + KD failures in STXBP1-DEE with persistent focal/tonic seizures"},
    {"threshold": "POLG exclusion before VPA", "action": "MANDATORY — do not administer VPA until POLG panel negative"},
    {"threshold": "ACTH BP > 95th centile", "action": "Hypertension during ACTH: reduce dose or pause; antihypertensive if persistent"},
]

# ── Concepts (Definitions) ────────────────────────────────────────────────────
CONCEPTS = [
    {
        "term": "STXBP1 (Syntaxin Binding Protein 1)",
        "definition": (
            "Gene at 9q34.11 encoding Munc18-1 (mammalian uncoordinated-18 protein 1), a member "
            "of the Sec1/Munc18 (SM) family. Munc18-1 is essential for synaptic vesicle exocytosis: "
            "it chaperones STX1A (syntaxin-1A) from the ER to the plasma membrane, templates the "
            "SNARE complex (STX1A/SNAP25/VAMP2), and gates the final Ca2+-triggered fusion step. "
            "STXBP1 haploinsufficiency impairs both excitatory and inhibitory transmission, "
            "with preferential loss of inhibitory (GABAergic) tone at high firing frequencies — "
            "net excitation/inhibition imbalance causes seizures."
        ),
    },
    {
        "term": "Munc18-1",
        "definition": (
            "Protein product of STXBP1; the essential gatekeeper of synaptic vesicle fusion. "
            "Crystal structure: arc-shaped SM protein that embraces STX1A in its closed "
            "conformation (domain 1 groove), releasing STX1A only when the SNARE motif is "
            "available for complex assembly. 50% reduction (haploinsufficiency) reduces "
            "vesicle priming rate — preferentially impacts fast-spiking interneurons "
            "which depend on high-capacity vesicle docking."
        ),
    },
    {
        "term": "Burst-Suppression (BS) Pattern",
        "definition": (
            "Neonatal EEG hallmark of severe epileptic encephalopathy: alternating high-amplitude "
            "bursts (100-600 µV, poly-spike-wave, 2-8 s) with near-isoelectric suppression "
            "intervals (< 5 µV, 2-10 s). Ohtahara syndrome / EIEE: defined by BS pattern "
            "± structural/genetic cause. STXBP1-BS: typically ASYNCHRONOUS between hemispheres "
            "(contrast: KCNQ2-BS is hemisynchronous). Continuous BS in waking AND sleep = "
            "Ohtahara; BS only in sleep (suppression-burst) = less severe phenotype."
        ),
    },
    {
        "term": "Ohtahara Syndrome (EIEE)",
        "definition": (
            "Early Infantile Epileptic Encephalopathy (EIEE) / Ohtahara syndrome: neonatal-onset "
            "DEE with continuous burst-suppression, tonic spasms, and severe GDD. "
            "Multiple genetic causes: STXBP1 (EIEE4), SCN2A (EIEE11), ARX (EIEE1, males), "
            "KCNQ2 (EIEE7), SCN8A (EIEE13). STXBP1 is the most frequently identified gene "
            "in clinically-diagnosed Ohtahara syndrome (~30% of genetically-solved cases). "
            "Evolves to West syndrome (IS/hypsarrhythmia) in ~50-70% by 3-6 months — "
            "'Ohtahara-to-West continuum.'"
        ),
    },
    {
        "term": "Hypsarrhythmia (Modified STXBP1)",
        "definition": (
            "Classical hypsarrhythmia (Gibbs, 1952): high-amplitude (> 200 µV) chaotic, "
            "disorganised, nearly continuous spike-wave activity with no normal background — "
            "characteristic of cryptogenic infantile spasms. "
            "STXBP1-modified hypsarrhythmia: fragmented, asymmetric, asynchronous HVS with "
            "intermittent suppression intervals — less 'chaotic' than classical; focal "
            "independent spike clusters; inter-spasm EEG intervals preserved (modified variant "
            "by Hrachovy criteria). Hypsarrhythmia resolution at day 14 (UKISS criterion) "
            "is the primary EEG endpoint for ACTH response."
        ),
    },
    {
        "term": "Theta Hypersynchrony",
        "definition": (
            "Highly characteristic EEG feature of STXBP1-DEE in wakefulness (> 6 months of age): "
            "generalised rhythmic 4-6 Hz theta activity with high amplitude (50-150 µV), "
            "bilaterally synchronous, present in waking background — UNUSUAL in other genetic "
            "epilepsies and essentially pathognomonic when combined with the DEE history. "
            "Clinically: associated with non-convulsive bursts (mild head drops or eye deviation "
            "during theta runs). Not to be confused with drowsiness theta (lower amplitude, "
            "disappears with arousal)."
        ),
    },
    {
        "term": "Electroclinical Dissociation (ECD)",
        "definition": (
            "Phenomenon in which clinical seizure manifestations cease (e.g., no visible tonic "
            "stiffening) while EEG seizure activity continues — common in STXBP1-DEE neonates "
            "after phenobarbital loading (> 60% rate). ECD is a major pitfall: clinicians may "
            "conclude seizures are controlled when EEG demonstrates persistent ictal activity. "
            "Continuous EEG (cEEG) monitoring is the only reliable method to detect ECD. "
            "Treatment should target EEG seizure suppression, not just clinical cessation."
        ),
    },
    {
        "term": "SNARE Complex",
        "definition": (
            "Soluble N-ethylmaleimide-sensitive factor Attachment protein REceptors (SNAREs): "
            "the core molecular machinery for synaptic vesicle fusion. "
            "Components: VAMP2 (vesicle-associated, v-SNARE) + STX1A + SNAP25 (plasma membrane, "
            "t-SNAREs). SNARE complex formation draws vesicle and plasma membrane together, "
            "providing energy for lipid bilayer fusion. STXBP1 (Munc18-1) is REQUIRED for "
            "correct SNARE complex assembly — without Munc18-1, STX1A cannot bind VAMP2 in "
            "the primed state, blocking Ca2+-triggered exocytosis."
        ),
    },
    {
        "term": "EIEE4 (MIM #612164)",
        "definition": (
            "OMIM designation for STXBP1-related Early Infantile Epileptic Encephalopathy, "
            "type 4. First described by Saitsu et al. (2008, Nature Genetics): 6/10 Ohtahara "
            "syndrome patients had de novo STXBP1 mutations. Current ILAE preferred terminology: "
            "'STXBP1 encephalopathy' or 'STXBP1-DEE' — more accurately reflects the combined "
            "developmental (non-epilepsy) and epileptic components. Synonyms: "
            "STXBP1-related disorder, MUNC18-1 encephalopathy."
        ),
    },
    {
        "term": "West Syndrome",
        "definition": (
            "Epileptic encephalopathy triad: (1) infantile spasms (flexor/extensor/mixed); "
            "(2) hypsarrhythmia on EEG; (3) developmental arrest / regression. "
            "Onset 3-12 months (peak 4-8 months). STXBP1-DEE commonly transitions from "
            "Ohtahara to West syndrome by 3-6 months ('Ohtahara-to-West continuum'). "
            "STXBP1-West: modified (asymmetric/fragmented) hypsarrhythmia; poor prognosis "
            "for spasm freedom vs. cryptogenic IS (40-50% vs. 75-80%). Treatment: ACTH + VGB "
            "(UKISS protocol); KD for ACTH/VGB-resistant cases."
        ),
    },
    {
        "term": "SHARE REMS (Vigabatrin FDA Mandate)",
        "definition": (
            "Support, Help and Resources for Epilepsy (SHARE) Risk Evaluation and Mitigation "
            "Strategy: FDA-mandated programme for vigabatrin (Sabril). Enrolled prescribers, "
            "pharmacies, and patients/caregivers. Required ophthalmological monitoring: "
            "Goldman perimetry q3 months (to detect peripheral VFD); OCT q6 months "
            "(RNFL thickness). Prescribers MUST confirm monitoring is in place before "
            "each refill. VFD risk: ~30-40% with prolonged use; typically irreversible. "
            "Benefit-risk: in infantile spasms the short-term (< 6-12 months) benefit of "
            "spasm control outweighs VFD risk (which is cumulative and dose-dependent)."
        ),
    },
    {
        "term": "POLG (Polymerase Gamma)",
        "definition": (
            "POLG (mitochondrial DNA polymerase gamma): mutations cause Alpers-Huttenlocher "
            "syndrome (AHS), which presents with epilepsy, liver disease, and neurodegeneration. "
            "POLG-positive patients exposed to valproate develop fatal acute liver failure — "
            "Black Box Warning (FDA). POLG mutations can mimic STXBP1-DEE clinically "
            "(neonatal/early infantile seizures, regression). POLG exclusion is MANDATORY "
            "before administering VPA in any child with unexplained DEE. POLG panel (WES or "
            "targeted) required; cannot be excluded by clinical assessment alone."
        ),
    },
    {
        "term": "VNS (Vagus Nerve Stimulation)",
        "definition": (
            "VNS: implantable device delivering intermittent electrical stimulation to the left "
            "vagus nerve, activating ascending noradrenergic/serotonergic pathways to diffusely "
            "suppress cortical excitability. FDA-approved for drug-resistant epilepsy (> 12 years, "
            "then off-label expansion to < 12 years). In STXBP1-DEE: VNS considered after "
            "3+ AED failures + KD failure; response rate ~30-40% (> 50% seizure reduction); "
            "particularly effective for tonic and generalised seizures. Advantage over resective "
            "surgery: no hemisphere involvement needed; can be used in diffuse DEE."
        ),
    },
    {
        "term": "SUDEP (Sudden Unexpected Death in Epilepsy)",
        "definition": (
            "SUDEP: sudden death in a person with epilepsy, without identified structural or "
            "toxicological cause, presumed epilepsy-related. Mechanism: post-ictal brainstem "
            "depression (PGES — post-ictal generalised EEG suppression) → respiratory and/or "
            "cardiac arrest. STXBP1-DEE SUDEP risk: HIGH — uncontrolled convulsive seizures "
            "are the strongest risk factor; nocturnal supervision/alarms recommended. "
            "NICE NG217 §1.15: all families with epilepsy + uncontrolled seizures should receive "
            "SUDEP counselling; written information; supervised sleeping environment (baby monitor "
            "with movement sensor); rescue midazolam plan."
        ),
    },
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    {
        "ref": "Saitsu 2008 Nat Genet",
        "title": "De novo mutations in the gene encoding SYNTAXIN BINDING PROTEIN 1 (STXBP1) cause early infantile epileptic encephalopathy",
        "note": "Landmark discovery paper — 6/10 Ohtahara syndrome patients; established EIEE4 as STXBP1-DEE"
    },
    {
        "ref": "Lux 2004 Lancet (UKISS)",
        "title": "The United Kingdom Infantile Spasms Study comparing vigabatrin with prednisolone or tetracosactide",
        "note": "RCT: ACTH + VGB vs. VGB alone — combination superior; basis for UKISS IS protocol"
    },
    {
        "ref": "Stamberger 2016 Neurology",
        "title": "STXBP1 encephalopathy: a neurodevelopmental disorder including epilepsy, autism, ADHD and movement disorder",
        "note": "256-patient cohort; phenotypic spectrum; theta hypersynchrony as EEG biomarker; natural history"
    },
    {
        "ref": "Kovacevic 2018 Brain",
        "title": "Heterozygous loss of Munc18-1 causes cognitive deficits and seizures by haploinsufficiency",
        "note": "Stxbp1+/- mouse model; excitatory/inhibitory imbalance mechanism; GABAergic interneuron preference"
        },
    {
        "ref": "Deprez 2010 Ann Neurol",
        "title": "Clinical spectrum of early-onset epileptic encephalopathies associated with STXBP1 mutations",
        "note": "33 patients; Ohtahara → West transition in 70%; severe GDD; mortality in first year"
    },
    {
        "ref": "Pappas 2021 Brain Commun",
        "title": "STXBP1 encephalopathy: from genotype to phenotype to targeted therapeutics",
        "note": "Comprehensive review; therapeutic pipeline including OP-01 chaperone; trial landscape"
    },
]

# ── Patient Generator ─────────────────────────────────────────────────────────
def _generate_patients():
    random.seed(SEED)
    categories = [
        ("De-novo-STXBP1-missense-LOF", 18),
        ("De-novo-STXBP1-truncating-nonsense-frameshift", 14),
        ("De-novo-STXBP1-large-deletion-CNV-9q34", 5),
        ("De-novo-STXBP1-splice-site", 3),
        ("Clinical-STXBP1-DEE-STXBP1-negative", 1),
    ]
    sexes = ["M", "F"]
    disease_phases = ["Neonatal-NICU", "IS-phase", "Post-IS-multifocal-DRE",
                      "School-age-LGS-like", "Adult-stable"]
    aed_options = ["PB+LEV", "PB+LEV+CLB", "LEV+CLB+KD", "LEV+KD", "VGB+ACTH-completed+LEV+KD",
                   "CLB+LEV+KD", "PB+LEV+VPA(POLG-excl)", "LEV+CLB", "KD+CLB", "CLB+LEV+VGB"]
    seizure_controls = ["drug-resistant", "partial-control", "seizure-free"]
    control_weights = [0.65, 0.25, 0.10]

    patients = []
    pid = 1
    for cat, n in categories:
        for _ in range(n):
            sex = random.choice(sexes)
            age_months = random.randint(1, 240)
            # onset in first 5 days of life (hours)
            onset_hours = random.randint(2, 120)
            phase = random.choice(disease_phases)
            treatment = random.choice(aed_options)
            control = random.choices(seizure_controls, weights=control_weights)[0]
            kd_on = "Y" if "KD" in treatment else "N"
            vgb_on = "Y" if "VGB" in treatment else "N"
            bhb = round(random.uniform(1.5, 4.5), 1) if kd_on == "Y" else None
            pb_level = round(random.uniform(12, 38), 1) if "PB" in treatment else None
            share_enrolled = "Y" if vgb_on == "Y" else "N"
            polg_tested = random.choice(["Y", "Y", "Y", "N"])
            eeg_theta = random.choice(["Y", "Y", "N"]) if age_months > 6 else "N"
            patients.append({
                "id": f"STXBP{pid:03d}",
                "age_months": age_months,
                "sex": sex,
                "onset_age_hours": onset_hours,
                "category": cat,
                "disease_phase": phase,
                "current_treatment": treatment,
                "seizure_control": control,
                "kd_on": kd_on,
                "bhb_mmoll": bhb,
                "vgb_on": vgb_on,
                "share_rems_enrolled": share_enrolled,
                "pb_level_ugml": pb_level,
                "polg_tested": polg_tested,
                "eeg_theta_hypersynchrony": eeg_theta,
            })
            pid += 1
    return patients


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    pts = _generate_patients()
    n = len(pts)
    seizure_free = sum(1 for p in pts if p["seizure_control"] == "seizure-free")
    drug_resistant = sum(1 for p in pts if p["seizure_control"] == "drug-resistant")
    kd_on = sum(1 for p in pts if p["kd_on"] == "Y")
    vgb_on = sum(1 for p in pts if p["vgb_on"] == "Y")
    share_enrolled = sum(1 for p in pts if p["share_rems_enrolled"] == "Y")
    polg_tested = sum(1 for p in pts if p["polg_tested"] == "Y")
    avg_onset_hrs = round(sum(p["onset_age_hours"] for p in pts) / n, 1)
    theta_positive = sum(1 for p in pts if p["eeg_theta_hypersynchrony"] == "Y")
    missense_n = sum(1 for p in pts if p["category"] == "De-novo-STXBP1-missense-LOF")

    return {
        "syndrome": "STXBP1 Encephalopathy (STXBP1-DEE / EIEE4)",
        "gene": "STXBP1 (9q34.11)",
        "inheritance": "De novo in > 95%; < 5% familial (somatic/germline mosaic parent)",
        "n_patients": n,
        "key_gene": "STXBP1 (9q34.11) — Munc18-1 synaptic vesicle fusion regulator; SNARE chaperone; SM-protein family",
        "eeg_hallmark": "Asynchronous burst-suppression (neonatal Ohtahara-like) → modified hypsarrhythmia (IS/West phase 3-6M) → theta hypersynchrony + multifocal spikes (childhood). Theta hypersynchrony in wakefulness = pathognomonic.",
        "key_biomarker": "STXBP1 sequencing (WES trio / neonatal DEE panel); POLG exclusion before VPA; PB TDM 20-40 µg/mL; VGB SHARE REMS (Goldman VF q3M + OCT q6M); BHB 2-4 mmol/L on KD",
        "key_aha": (
            "STXBP1 is the MOST FREQUENT gene in Ohtahara syndrome (~30% of solved cases). "
            "Neonatal EEG: ASYNCHRONOUS burst-suppression (contrast: KCNQ2 = hemisynchronous). "
            "THETA HYPERSYNCHRONY in wakefulness (> 6 months) = near-pathognomonic EEG marker. "
            "POLG EXCLUSION MANDATORY before ANY VPA administration. "
            "ACTH + VGB (UKISS) for infantile spasms phase; KD highly effective as add-on. "
            "VGB = SHARE REMS mandatory (USA) — Goldman VF q3 months throughout treatment. "
            "ELECTROCLINICAL DISSOCIATION (> 60% after PB): treat to EEG, not clinical, seizure cessation. "
            "OP-01 chaperone trial (Phase 1) — refer eligible missense LOF patients."
        ),
        "etiologies": [
            {"etiology": "De novo STXBP1 missense LOF", "category": "De-novo-STXBP1-missense-LOF", "pct": 44},
            {"etiology": "De novo STXBP1 truncating (nonsense / frameshift)", "category": "De-novo-STXBP1-truncating-nonsense-frameshift", "pct": 34},
            {"etiology": "De novo STXBP1 large deletion (9q34 CNV)", "category": "De-novo-STXBP1-large-deletion-CNV-9q34", "pct": 12},
            {"etiology": "De novo STXBP1 splice-site variant", "category": "De-novo-STXBP1-splice-site", "pct": 7},
            {"etiology": "Clinical STXBP1-DEE — STXBP1-negative", "category": "Clinical-STXBP1-DEE-STXBP1-negative", "pct": 3},
        ],
        "seizure_type_prevalence": {
            "Tonic / Tonic-Clonic Neonatal Seizures": 95,
            "Infantile Spasms (West Syndrome phase)": 70,
            "Focal Seizures (multifocal / migratory neonatal)": 65,
            "Myoclonic Seizures (axial / massive)": 40,
        },
        "trigger_seizure_rates": {
            "Fever / febrile illness": 85,
            "Intercurrent illness (viral / bacterial)": 75,
            "Missed / late AED dose": 70,
            "Sleep deprivation / disrupted sleep-wake cycle": 60,
            "Hyperthermia (bath / environment)": 50,
            "Physical / procedural stimulation": 35,
            "Rapid AED dose reduction / change": 30,
            "Puberty / hormonal change": 20,
        },
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "kpis": {
            "seizure_free_pct": round(seizure_free / n * 100, 1),
            "drug_resistant_pct": round(drug_resistant / n * 100, 1),
            "kd_on_pct": round(kd_on / n * 100, 1),
            "vgb_on_pct": round(vgb_on / n * 100, 1),
            "share_rems_enrolled_pct": round(share_enrolled / n * 100, 1),
            "polg_tested_pct": round(polg_tested / n * 100, 1),
            "avg_onset_age_hours": avg_onset_hrs,
            "theta_hypersynchrony_pct": round(theta_positive / n * 100, 1),
        },
        "clinical_alerts": [
            "🧬 STXBP1 = MOST COMMON gene in Ohtahara syndrome (~30% of genetically-solved cases) — test FIRST in neonatal BS.",
            "⚡ EEG: ASYNCHRONOUS burst-suppression (neonatal) → theta hypersynchrony (> 6M) — pathognomonic combination.",
            "🔬 ELECTROCLINICAL DISSOCIATION in > 60% after PB — treat to EEG seizure cessation, NOT clinical cessation.",
            "🚫 POLG EXCLUSION MANDATORY before VPA — fatal hepatotoxicity (Alpers syndrome); POLG panel or WES required.",
            "💊 ACTH + VGB (UKISS protocol) for infantile spasms — hypsarrhythmia resolution criterion at day 14.",
            "👁️ VGB = SHARE REMS MANDATORY (USA) — Goldman VF q3 months + OCT q6 months throughout VGB course.",
            "🥑 Ketogenic Diet — Level B for STXBP1-DEE; particularly effective for myoclonic-atonic and post-IS-residual seizures.",
            "🛡️ SUDEP HIGH RISK — nocturnal alarm + supervised sleeping + rescue midazolam plan (NICE NG217 §1.15).",
        ],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_breakdown():
    pts = _generate_patients()
    return {
        "patients": pts,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "aed_monitoring": AED_MONITORING,
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "standards": STANDARDS,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "absolute_contraindications": ABSOLUTE_CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "standards": STANDARDS,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
