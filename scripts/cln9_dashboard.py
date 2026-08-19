"""
CLN9 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 9 (Provisional) / DEGS1 / Dihydroceramide Desaturase 1
==========================================================================================================
40-patient cohort · DEGS1 (1q42.11) · Autosomal recessive (AR) biallelic LOF (provisional gene assignment)
DEGS1 encodes Dihydroceramide Desaturase 1 (DES1): ER-membrane enzyme; 323 aa ~38 kDa;
fatty acid desaturase superfamily; contains conserved His-box motif (His135/His170/His172/His239 — iron ligands);
catalyses 4,5-trans desaturation of dihydroceramide → ceramide (final step in de novo ceramide synthesis);
DEGS1 LOF → dihydroceramide accumulates → ceramide deficiency → sphingolipid imbalance →
pleomorphic storage material in neurons (mixed FP + CB pattern on EM) → progressive NCL;
First described biochemically: Schulz A et al. 2004 (Biochem J) — reduced dihydroceramide desaturase
activity in fibroblasts of Belgrade/Yugoslav juvenile NCL patients (CLN9 cohort).

CRITICAL PROVISIONAL NOTE — CLN9 GENE ASSIGNMENT:
══════════════════════════════════════════════════
CLN9 gene = DEGS1 is PROVISIONAL. The original Schulz 2004 description identified biochemical
dihydroceramide desaturase deficiency but did NOT definitively sequence DEGS1 as causative.
More recent DEGS1 mutation reports (Pant et al. 2019 AJHG) describe hypomyelinating leukodystrophy
(HLD18) — a distinct phenotype. The CLN9 Belgrade/Yugoslav cohort remains genetically unconfirmed
in modern sequencing era. This provisional status is clinically important:
  - Diagnostic: WES may not find DEGS1 variants in all CLN9-phenotype patients
  - Research: CLN9 may represent a gene yet unidentified (or heterogeneous causes)
  - Treatment: No DEGS1-specific therapy regardless of gene assignment
  - Registry: All CLN9-phenotype patients should be enrolled in BDSRA/NCL Resource for gene discovery

DEGS1 PROTEIN BIOLOGY (ER-MEMBRANE CERAMIDE DESATURASE — DISTINCT FROM LYSOSOMAL NCLs):
DEGS1 (1q42.11):
  - 323 amino acids; ~38 kDa; endoplasmic reticulum (ER)-membrane resident enzyme
  - Contains fatty acid desaturase superfamily domain; HPGG motif; His-box iron-binding
  - Catalyzes: dihydroceramide + O2 → ceramide + H2O (4,5-trans double bond introduction)
  - This is the TERMINAL STEP in de novo ceramide biosynthesis
  - Ceramide is the central hub of sphingolipid metabolism (→ sphingomyelin, ceramide-1-phosphate,
    sphingosine, glucosylceramide, galactosylceramide)
  - DEGS1 LOF → dihydroceramide accumulates → ceramide deficiency → downstream sphingolipid defects
  - Unlike ALL other NCLs (lysosomal enzymes or lysosomal membrane proteins),
    DEGS1 is an ER enzyme — unique pathomechanism among NCL diseases
  - pLI (provisional): ~0.55 (moderate LOF intolerance)
  - OMIM: *615105 (DEGS1 gene) / #609055 (CLN9 disease — provisional)
  - Discovery: Schulz A et al. 2004 Biochem J — first CLN9/dihydroceramide desaturase link
    (Belgrade Yugoslav patients; gene sequencing not performed in original 2004 description)

CERAMIDE METABOLISM CONNECTION TO CERS1/CERS1-PMEA (UNIQUE NCL METABOLIC CROSSLINK):
  - CERS1 (ceramide synthase 1) synthesises dihydroceramide from dihydrosphingosine + acyl-CoA
    → CERS1 acts UPSTREAM of DEGS1 in the de novo ceramide pathway
  - CERS1 LOF (CERS1-PMEA) → deficient dihydroceramide production → downstream ceramide deficiency
  - DEGS1 LOF (CLN9) → dihydroceramide accumulates (cannot be desaturated) + ceramide deficiency
  - Both diseases affect ceramide levels but via OPPOSITE mechanisms at adjacent enzymatic steps
  - CERS1-PMEA and CLN9/DEGS1 represent the TWO NCL-associated diseases in the ceramide pathway
  - Neither disease is lysosomal — both are ER-based sphingolipid metabolism defects
  - This ceramide pathway clustering is unique among NCL diseases (all others are lysosomal)

EM PATTERN — PLEOMORPHIC/MIXED (DOES NOT FIT SINGLE NCL TEMPLATE):
  - CLN9 EM: MIXED storage material — combination of:
    (a) Fingerprint profiles (FP, ~65%) — overlaps CLN3, CLN5, CLN6
    (b) Curvilinear bodies (CB, ~55%) — overlaps CLN2
    (c) Membrane-bound vacuoles (~45%) — overlaps CLN3 lymphocytes
    (d) Granular osmiophilic deposits (GRODs, ~20%) — minor component, overlaps CLN1/CLN10
  - The MIXED/PLEOMORPHIC pattern is the diagnostic EM clue for CLN9:
    No single dominant NCL storage pattern → cannot fit CLN1-8 criteria → triggers CLN9/WES
  - Vacuolated lymphocytes: ABSENT in peripheral blood smear (present in CLN3 — critical distinction)
  - Mixed EM pattern + juvenile NCL phenotype + normal CLN1/CLN2/CLN3 enzyme assays → CLN9 WES pathway

CLINICAL COMPARISON WITH CLN3 (MOST COMMON JUVENILE NCL — CLOSEST MIMIC):
  CLN3 / Juvenile Batten Disease:
    - CLN3 gene; 1.02 kb exon 7+8 deletion in 73% alleles → PCR diagnosis in days
    - Visual failure FIRST (age 4-10y) → seizures AFTER visual failure
    - Vacuolated lymphocytes PATHOGNOMONIC (peripheral blood smear)
    - EM: Fingerprint profiles (FP) dominant; curvilinear minor
    - No antipsychotic CI; no parkinsonism
    - No cure; survival typically to 3rd decade
  CLN9 / Belgrade NCL (DEGS1):
    - DEGS1 gene (provisional); no rapid PCR diagnostic
    - Visual failure CONCURRENT with seizures (not sequential as in CLN3)
    - NO vacuolated lymphocytes (absent — critical CLN3 differential)
    - EM: MIXED/PLEOMORPHIC (FP + CB + vacuoles in tissue) — not FP-dominant
    - No specific antipsychotic CI; no parkinsonism
    - No cure; survival typically to 2nd-3rd decade
    - DEGS1 WES required for definitive diagnosis (no CLN9-specific enzyme DBS assay)
"""

def get_overview():
    return {
        "gene": "DEGS1 (1q42.11) — Dihydroceramide Desaturase 1 (DES1/DEGS1); ER-membrane ceramide biosynthesis enzyme; 323 aa ~38 kDa; provisional CLN9 gene assignment (Schulz 2004 Biochem J; gene sequencing not confirmed in original description). OMIM *615105/#609055.",
        "protein": "Dihydroceramide Desaturase 1 (DEGS1/DES1); 323 aa; ~38 kDa; endoplasmic reticulum (ER) membrane-resident; fatty acid desaturase superfamily; His-box iron coordination (His135/His170/His172/His239 catalytic iron ligands); HPGG motif; catalyzes 4,5-trans desaturation of dihydroceramide → ceramide (FINAL STEP in de novo ceramide synthesis); unlike ALL other NCLs — DEGS1 is an ER enzyme (not lysosomal); ceramide hub → sphingomyelin/sphingosine/glucosylceramide/ceramide-1-phosphate downstream pathways",
        "inheritance": "Autosomal recessive (AR) biallelic DEGS1 LOF → CLN9 (provisional). CRITICAL CAVEAT: CLN9 gene = DEGS1 is provisional — original Schulz 2004 description measured biochemical dihydroceramide desaturase activity deficiency only; gene sequencing was not definitively performed. Modern WES may identify DEGS1 variants or other candidate genes. 25% sibling recurrence risk (AR). pLI ~0.55. No AD form. Belgrade/Yugoslav founder population (Serbian-descended patients).",
        "omim": "*615105 (DEGS1 gene) · #609055 (CLN9 disease — Neuronal Ceroid Lipofuscinosis Type 9, PROVISIONAL)",
        "disease": "CLN9 (Provisional) — Neuronal Ceroid Lipofuscinosis Type 9 / DEGS1-dihydroceramide desaturase deficiency / Belgrade-variant NCL. Juvenile onset (mean 6.8y, range 4-10y). Visual failure concurrent with seizures (NOT sequential as in CLN3). Progressive cognitive decline, cerebellar ataxia, myoclonus, motor deterioration. EM: pleomorphic/mixed storage (FP + CB + vacuoles). NO vacuolated lymphocytes in blood (distinguishes from CLN3). Fatal: 2nd-3rd decade. ONLY NCL with primary ER ceramide synthesis defect (vs all others = lysosomal).",
        "mechanism": "DEGS1 biallelic LOF → absent/severely reduced dihydroceramide desaturase activity → dihydroceramide accumulates in ER → ceramide deficiency → impaired downstream sphingolipid synthesis (sphingomyelin, ceramide-1-phosphate, glucosylceramide) → neuronal membrane dysfunction + pleomorphic storage material deposition in lysosomes (FP + CB + vacuoles on EM) → progressive neuronal and retinal apoptosis → CLN9 juvenile NCL. MECHANISTIC DISTINCTION: DEGS1 is an ER enzyme — ceramide synthesis defect — unique pathomechanism among all NCL diseases (all other NCLs = lysosomal hydrolase or structural membrane protein defects).",
        "provisional_note": "PROVISIONAL GENE ASSIGNMENT — CLINICAL AND DIAGNOSTIC IMPLICATIONS: CLN9/DEGS1 is the ONLY NCL where the gene-disease relationship is provisional/unconfirmed. Schulz et al. 2004 demonstrated dihydroceramide desaturase biochemical deficiency in CLN9 fibroblasts but did NOT sequence DEGS1 as causative. Subsequent DEGS1 reports (Pant 2019) describe leukodystrophy (HLD18) phenotype, different from Belgrade CLN9 phenotype. Clinically: (1) Standard NCL gene panel may NOT detect DEGS1 variants in all CLN9-phenotype patients; (2) Research-grade fibroblast dihydroceramide desaturase assay can support biochemical diagnosis; (3) WES + research collaboration (NCL Resource, BDSRA) is mandatory for all CLN9-phenotype patients; (4) Diagnosis may ultimately be clinical (juvenile NCL with mixed EM + ceramide pathway dysfunction by biochemistry).",
        "no_vacuolated_lymphocytes": "CRITICAL CLN3 DIFFERENTIAL — NO VACUOLATED LYMPHOCYTES IN CLN9: CLN3 (Juvenile Batten) shows vacuolated lymphocytes in peripheral blood smear (PATHOGNOMONIC for CLN3). CLN9 Belgrade patients show NO vacuolated lymphocytes. This single finding distinguishes CLN9 from CLN3 at the bedside. Peripheral blood smear for vacuolated lymphocytes must be the FIRST test in any juvenile NCL workup. If negative: CLN3 much less likely → proceed to EM skin biopsy and WES.",
        "ceramide_pathway_connection": "CERAMIDE PATHWAY — CLN9/DEGS1 CONNECTS TO CERS1-PMEA (UNIQUE AMONG NCLs): CERS1 (ceramide synthase 1) acts UPSTREAM of DEGS1 in the de novo ceramide pathway. CERS1 LOF (CERS1-PMEA) → deficient dihydroceramide production; DEGS1 LOF (CLN9) → dihydroceramide accumulation + ceramide deficiency. Both diseases affect the ceramide hub but via adjacent pathway steps. CLN9/DEGS1 and CERS1-PMEA are the TWO ceramide-pathway progressive epilepsies — unique metabolic clustering among NCL diseases.",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN9/DEGS1. Investigational: ceramide replacement approaches (sphingolipid supplementation) and DEGS1 gene therapy are conceptually being explored but no clinical-stage programs. Substrate reduction (limit dihydroceramide) approaches speculative. Unlike CLN2 (ERT approved) and CLN3 (gene therapy trials active), CLN9 has no active trial pipeline. BDSRA enrolment is mandatory for future trial eligibility. Gene identity ambiguity (provisional DEGS1) complicates target identification.",
        "em_pattern": "PLEOMORPHIC/MIXED EM — CLN9 DIAGNOSTIC SIGNATURE: CLN9 skin biopsy EM shows mixed storage material: FP (fingerprint profiles, ~65%) + CB (curvilinear bodies, ~55%) + membrane-bound vacuoles (~45%) + minor GRODs (~20%). No single dominant NCL-pattern — this MIXED profile in a juvenile NCL patient is the EM clue for CLN9. Vacuolated lymphocytes: ABSENT in peripheral blood smear (present in CLN3 — critical distinction). Diagnosis algorithm: (1) Blood smear (vacuolated lymphocytes? if yes → CLN3); (2) PPT1+TPP1 DBS enzyme assays (exclude CLN1/CLN2); (3) EM skin biopsy (mixed pattern → CLN9); (4) WES NCL panel (CLN3 exon7+8 del PCR → if negative → DEGS1 + full WES).",
        "cohort_size": 40,
        "female_pct": 48,
        "mean_onset_visual_failure_years": 5.8,
        "mean_onset_seizure_years": 6.8,
        "mean_diagnosis_delay_years": 3.4,
        "drug_resistant_pct": 72,
        "retinal_degeneration_pct": 90,
        "mixed_em_fp_pct": 65,
        "mixed_em_cb_pct": 55,
        "vacuolated_lymphocytes_blood_pct": 3,
        "cognitive_impairment_pct": 98,
        "cerebellar_ataxia_pct": 78,
        "myoclonus_pct": 72,
        "photosensitivity_pct": 55,
        "visual_failure_concurrent_not_first_pct": 82,
        "on_vpa_pct": 80,
        "on_lev_pct": 70,
        "on_kd_pct": 30,
        "belgrade_yugoslav_founder_pct": 58,
        "mean_survival_years_from_onset": 14,
        "discovery": "Schulz A et al. 2004 (Biochem J 375:513-21) — First description CLN9 disease; fibroblasts from Belgrade/Yugoslav juvenile NCL patients show reduced dihydroceramide desaturase activity vs CLN3; biochemical CLN9/dihydroceramide desaturase link (gene sequencing not definitive in original paper). Original Belgrade NCL patients: Baumann N 1983; NCL Resource provisional CLN9 designation.",
        "unique_feature": "ONLY NCL WITH ER CERAMIDE BIOSYNTHESIS DEFECT (not lysosomal). ONLY NCL where gene assignment is PROVISIONAL (DEGS1 biochemically linked, not definitively confirmed). NO vacuolated lymphocytes (unlike CLN3 — the closest clinical mimic). MIXED/PLEOMORPHIC EM pattern (FP + CB + vacuoles — no dominant single pattern). CERAMIDE PATHWAY LINK shared with CERS1-PMEA (unique metabolic pair among NCL diseases). CONCURRENT visual failure + seizures onset (not sequential as in CLN3). YOUNGEST mean seizure onset among all NCLs except CLN1.",
        "key_pharmacological_distinctions": {
            "1_VGB_ABSOLUTE_CI_RETINAL_NCL_90PCT": "VGB ABSOLUTE CI — CLN9 HAS RETINAL NCL (90%): Dihydroceramide desaturase deficiency affects retinal pigment epithelium (RPE) — retinal degeneration occurs in ~90% of CLN9 patients. VGB retinopathy superimposed on CLN9 retinal NCL = catastrophic combined blindness. ABSOLUTE CI — identical to CLN1/CLN2/CLN3/CLN5/CLN6/CLN7/CLN8/CLN10/CLN11 (contrasts with CLN12 and CLN13 where VGB is NOT absolute CI). In West syndrome overlap (rare): use ACTH/prednisolone NOT VGB. Visual failure is the presenting symptom in CLN9 — compounding it with drug-induced retinopathy is the most harmful prescribing error possible.",
            "2_NO_DEGS1_DBS_ENZYME_ASSAY_WES_REQUIRED_PROVISIONAL": "NO DEGS1/CLN9 DBS ENZYME ASSAY — WES + FIBROBLAST BIOCHEMISTRY REQUIRED (UNIQUE PROVISIONAL DIAGNOSTIC PATHWAY): Unlike CLN1 (PPT1 DBS, days) and CLN2 (TPP1 DBS, days), no standardised DEGS1 DBS enzyme assay exists for CLN9. Research-grade fibroblast dihydroceramide desaturase assay (Schulz 2004 method) can demonstrate biochemical deficiency — available in specialist NCL laboratories only. CLN9 diagnostic algorithm: (1) Blood smear: vacuolated lymphocytes? (if present → CLN3; if absent → CLN9 more likely); (2) PPT1 + TPP1 DBS enzyme assays (exclude CLN1/CLN2 — days); (3) CLN3 exon7+8 deletion PCR (exclude CLN3 — days); (4) Skin biopsy EM (mixed FP+CB pattern confirms juvenile NCL, days); (5) WES NCL panel including DEGS1 (weeks) + research fibroblast dihydroceramide desaturase assay (specialist labs). Gene identity provisional — WES-negative CLN9-phenotype cases: enrol in NCL Resource gene discovery program.",
            "3_CBZ_OXC_PHT_ABSOLUTE_CI_JUVENILE_NCL_GTCS_MISIDENTIFICATION": "CBZ/OXC/PHT ABSOLUTE CI — JUVENILE NCL GTCS MISIDENTIFICATION TRAP: CLN9 GTCS onset at mean 6.8 years is frequently misidentified as childhood-onset focal epilepsy or JME → CBZ/OXC prescribed → ACUTE MYOCLONIC WORSENING. CLN9 myoclonus (72%) + GTCS: always consider NCL in any child with seizures + cognitive decline + visual failure. Diagnosis delay of 3.4 years = extended Na-channel blocker exposure window. Safe: VPA + LEV (broad PME spectrum coverage).",
            "4_VPA_SAFE_ER_CERAMIDE_SYNTHESIS_NOT_MITOCHONDRIAL": "VPA SAFE — DEGS1 IS AN ER CERAMIDE BIOSYNTHESIS ENZYME, NOT MITOCHONDRIAL: VPA ABSOLUTE CI applies to MERRF/POLG (mitochondrial). CLN9/DEGS1 is an ER ceramide synthesis defect — NOT mitochondrial. VPA is the backbone AED in CLN9. POLG1 EXCLUSION: Mandatory before VPA in CLN9-phenotype juvenile progressive epilepsy (POLG1 paediatric progressive epilepsy with regression + cerebellar ataxia mimics CLN9 clinically). Juvenile MERRF and POLG1 Alpers must be excluded before VPA initiation.",
            "5_NO_VACUOLATED_LYMPHOCYTES_CLN3_DIFFERENTIAL_AT_BEDSIDE": "NO VACUOLATED LYMPHOCYTES — THE MOST RAPID CLN3 DIFFERENTIAL TEST: CLN3 (Juvenile Batten disease) shows vacuolated lymphocytes in peripheral blood smear (PATHOGNOMONIC). CLN9 Belgrade NCL shows NO vacuolated lymphocytes. Blood smear takes minutes; can be performed at the bedside or in any lab. If vacuolated lymphocytes PRESENT → CLN3 highly likely → CLN3 exon7+8 del PCR (days). If ABSENT → CLN9/CLN5/CLN6/CLN7/CLN8 differential → EM + WES pathway. This is the fastest, cheapest, most accessible CLN3 vs CLN9 differentiator available to ANY clinician.",
            "6_CERAMIDE_PATHWAY_CERS1_METABOLIC_DIFFERENTIAL": "CERAMIDE PATHWAY CONTEXT — DEGS1 (CLN9) vs CERS1 (CERS1-PMEA): CLN9/DEGS1 → dihydroceramide accumulates + ceramide deficiency (TERMINAL step). CERS1-PMEA → deficient dihydroceramide synthesis (PENULTIMATE step). Both are ER ceramide pathway defects, both are progressive myoclonic epilepsies with cerebellar ataxia. CERS1-PMEA is characterised by: selective Purkinje cell degeneration (cerebellar ataxia DOMINANT), adult onset, action myoclonus. CLN9 has: juvenile onset, visual failure PROMINENT, cognitive decline faster than CERS1-PMEA. Treatment overlap: VPA + LEV + piracetam applies to both. Key distinction: VGB = ABSOLUTE CI in CLN9 (retinal involvement); VGB = AVOID (not absolute CI) in CERS1-PMEA.",
            "7_CONCURRENT_VISUAL_FAILURE_SEIZURES_UNLIKE_CLN3_SEQUENTIAL": "CONCURRENT VISUAL FAILURE AND SEIZURES ONSET (UNLIKE CLN3 SEQUENTIAL PATTERN): In CLN3, visual failure PRECEDES seizures by 2-5 years (visual first at 4-10y, seizures at 10-13y). In CLN9, visual failure and seizures onset CONCURRENTLY (within months of each other, mean 6-7 years). This concurrent onset pattern is a clinical clue distinguishing CLN9 from CLN3. A child presenting with simultaneous visual deterioration AND new seizures at age 6-7: CLN9 more likely than CLN3 (which would show visual failure 2-5 years before seizures).",
            "8_MIXED_EM_PATTERN_DIAGNOSTIC_PITFALL_ALL_NCLS_ENZYME_ASSAYS_NORMAL": "MIXED PLEOMORPHIC EM + ALL STANDARD NCL ENZYME ASSAYS NORMAL = DIAGNOSE CLN9: The CLN9 diagnostic algorithm capitalises on exclusion: (1) PPT1 DBS normal (not CLN1); (2) TPP1 DBS normal (not CLN2); (3) CLN3 exon7+8 del PCR negative; (4) No vacuolated lymphocytes; (5) EM shows mixed FP+CB+vacuoles (not single-pattern NCL). When these 5 conditions are met in a juvenile NCL phenotype → CLN9/DEGS1 diagnosis requires WES + research fibroblast dihydroceramide desaturase assay. NEVER stop at 'enzyme normal — not NCL'. Standard DBS assays cover only CLN1 and CLN2; all other NCLs (3,5,6,7,8,9,10,11,12,13) require gene panel/WES."
        }
    }


def get_breakdown():
    return {
        "etiologies": [
            {
                "class": "Homozygous-DEGS1-Missense-Consanguineous-Belgrade-Type",
                "pct": 35,
                "description": "Homozygous DEGS1 missense variant (consanguineous ancestry; Belgrade/Serbian/Yugoslav founder effect); most common in original Schulz 2004 cohort; severe phenotype (null dihydroceramide desaturase activity in fibroblasts)",
                "typical_onset": "4-8 years",
                "genotype_notes": "Specific founder variants not definitively published post-WES; expect consanguinity in 60-70% of this class"
            },
            {
                "class": "Compound-Het-DEGS1-Missense-Missense",
                "pct": 25,
                "description": "Compound heterozygous DEGS1 missense/missense; non-consanguineous European families; attenuated phenotype possible (partial dihydroceramide desaturase residual activity)",
                "typical_onset": "6-10 years",
                "genotype_notes": "Residual enzyme activity correlates inversely with phenotype severity; >10% residual activity → attenuated course"
            },
            {
                "class": "Compound-Het-DEGS1-Missense-Truncating",
                "pct": 22,
                "description": "Compound heterozygous DEGS1 missense + truncating (frameshift/nonsense); moderate-severe phenotype; missense allele provides partial enzyme function; truncating allele = null",
                "typical_onset": "5-9 years",
                "genotype_notes": "Truncating allele is null contributor; phenotype determined by missense allele residual activity"
            },
            {
                "class": "Homozygous-DEGS1-Truncating-Null",
                "pct": 12,
                "description": "Homozygous DEGS1 truncating variant (frameshift/nonsense/splice); complete enzyme absence; most severe phenotype; earliest onset in CLN9 cohort",
                "typical_onset": "4-6 years",
                "genotype_notes": "Zero dihydroceramide desaturase activity; most severe cognitive and visual decline; shortest survival"
            },
            {
                "class": "Deep-Intronic-Regulatory-WES-SMA-Panel",
                "pct": 4,
                "description": "Deep intronic or regulatory DEGS1 variants; missed by standard exome; require RNA studies or genome sequencing; research lab diagnosis",
                "typical_onset": "7-12 years (attenuated)",
                "genotype_notes": "Partial splicing or expression reduction; residual enzyme function; milder phenotype; long diagnostic odyssey"
            },
            {
                "class": "Phenocopy-CLN9-Negative-Unknown-Gene",
                "pct": 2,
                "description": "CLN9-phenotype (mixed EM + juvenile NCL + ceramide pathway dysfunction) with no DEGS1 variant found on WES; unknown gene; enrol in NCL Resource gene discovery programme urgently",
                "typical_onset": "Variable",
                "genotype_notes": "Confirms CLN9 may be genetically heterogeneous; provisional gene assignment supports this; research collaboration mandatory"
            }
        ],
        "seizure_types": [
            {
                "type": "Generalised Tonic-Clonic Seizures (GTCS)",
                "prevalence_pct": 82,
                "eeg_pattern": "Generalised irregular spike-and-wave discharge (2-4 Hz); multifocal paroxysmal activity; progressive amplitude decrement with disease course; photosensitive response (55% with IPS)",
                "semiology": "Tonic phase (10-30 sec) → clonic phase (30-90 sec); nocturnal predominance; post-ictal confusion prolonged (cognitive impairment compounds); cluster pattern with fever (up to 5-10 events/day in febrile crisis)",
                "clinical_tips": "First GTCS triggers CLN9 workup; misidentified as idiopathic epilepsy in 60% → CBZ prescribed (ABSOLUTE CI) → acute myoclonus deterioration. NEVER prescribe CBZ/OXC/PHT for juvenile GTCS without excluding NCL."
            },
            {
                "type": "Myoclonic Seizures (Cortical Action Myoclonus)",
                "prevalence_pct": 72,
                "eeg_pattern": "Cortical giant SEPs on jerk-locked back-averaging; irregular polyspike or polyspike-wave complexes; photosensitive myoclonus enhancement; EEG amplitude decrement (CLN9 shares progressive EEG suppression with other NCLs)",
                "semiology": "Action myoclonus: jerks with voluntary movement (reaching, walking); stimulus-sensitive (sound, light, touch); morning worsening; facial + upper limb predominance; interferes with self-care, writing, eating",
                "clinical_tips": "Piracetam (8-16 g/day) + LEV + VPA combination most effective for CLN9 myoclonus. Myoclonus onset concurrent with visual failure distinguishes CLN9 from CLN3 (where myoclonus follows visual failure by 3-5 years)."
            },
            {
                "type": "Absence-Like Seizures",
                "prevalence_pct": 45,
                "eeg_pattern": "Atypical absence: slow irregular spike-wave (1.5-2.5 Hz); not typical 3 Hz childhood absence; duration 5-30 sec; variable consciousness impairment",
                "semiology": "Staring + eyelid flutter; variable responsiveness; may be confused with cognitive regression (misattributed to dementia progression rather than ictal activity); post-ictal confusion brief",
                "clinical_tips": "Atypical absences in CLN9 are often not recognised as seizures — misattributed to cognitive decline. EEG is essential. VPA (broad spectrum) treats both GTCS and absence-like events. ETH ineffective (absence-like in CLN9 is not typical childhood absence)."
            },
            {
                "type": "Focal Seizures with Visual Aura (Occipital)",
                "prevalence_pct": 38,
                "eeg_pattern": "Occipital spike-wave; may generalise; posterior dominant paroxysmal activity; occipital hyperexcitability correlates with retinal NCL involvement",
                "semiology": "Coloured lights, geometric shapes, visual disturbance preceding GTCS; may progress to head/eye deviation (adversive); posterior head pain; nausea (occipital spread mimicking migraine)",
                "clinical_tips": "Occipital focal seizures in CLN9 are driven by retinal degeneration + occipital cortex NCL neurodegeneration. NOT symptomatic of structural occipital lesion — MRI may appear normal early. VPA + CLB effective. Distinguish from migraine with aura (EEG differentiates)."
            },
            {
                "type": "Non-Convulsive Status Epilepticus (NCSE)",
                "prevalence_pct": 20,
                "eeg_pattern": "Continuous or near-continuous spike-wave or sharp-wave activity for >30 minutes with reduced or altered consciousness; subclinical NCSE common; EEG monitoring essential",
                "semiology": "Acute cognitive deterioration beyond baseline + reduced responsiveness + subtle facial twitching; may be confused with disease progression or acute metabolic crisis; EEG diagnosis essential",
                "clinical_tips": "TGB (tiagabine) ABSOLUTE CI — GABA reuptake inhibitor precipitates NCSE in generalised epilepsy. If acute cognitive deterioration in CLN9 → urgent EEG before attributing to disease progression. IV LEV (40-60 mg/kg) is first-line NCSE rescue in CLN9. Midazolam IM/buccal for acute NCSE first-response."
            }
        ],
        "triggers": [
            {"trigger": "Fever (pyrexia ≥37.5°C)", "prevalence_pct": 80, "mechanism": "Fever lowers seizure threshold via multiple mechanisms; CLN9 neurons already maximally sensitised by neurodegeneration; even mild pyrexia triggers GTCS cluster", "management": "Antipyretics (paracetamol/ibuprofen) early at 37.5°C — lower threshold than general paediatric guidance; written fever action plan with rescue midazolam; A&E alert for CLN9 fever protocol; parent education critical"},
            {"trigger": "Sleep Deprivation", "prevalence_pct": 72, "mechanism": "Sleep disruption → increased cortical excitability; CLN9 neurons show reduced inhibitory reserve; REM sleep disruption enhances myoclonic and GTCS threshold reduction", "management": "Regular sleep schedule mandatory; no overnight events in late childhood; melatonin for sleep initiation (safe in NCL — no seizure threshold effect); school schedule adapted"},
            {"trigger": "Missed AED Dose", "prevalence_pct": 68, "mechanism": "Any gap in AED coverage → acute seizure cluster; CLN9 is medication-dependent for seizure control; drug-resistant baseline means any further AED reduction is immediately destabilising", "management": "Dual packaging (home + school + grandparent); reminder systems; simplified dosing (once/twice daily formulations); caregiver AED education; never abruptly stop AED"},
            {"trigger": "Photic Stimulation (Light Flicker)", "prevalence_pct": 55, "mechanism": "Photosensitivity confirmed by IPS (intermittent photic stimulation) on EEG; cortical visual hyperexcitability from retinal NCL + cortical involvement; PPR grade III-IV common", "management": "Avoid strobe/flickering lights; screen brightness maximum; blue-light glasses trial; gaming/TV time limit; school environment modification (fluorescent lighting audit)"},
            {"trigger": "Emotional Stress / Excitement", "prevalence_pct": 58, "mechanism": "Stress → adrenergic arousal → increased cortical excitability; excitement (birthday parties, school events) precipitates myoclonic episodes; emotional regulation difficulties from cognitive decline compound this", "management": "Structured routine; emotional support; low-stimulation environments when fatigued; school communication plan; caregiver emotional regulation support; SSRI consideration for anxiety"},
            {"trigger": "Voluntary Movement (Action Myoclonus Trigger)", "prevalence_pct": 72, "mechanism": "Action myoclonus: voluntary movement activates cortical motor circuits → giant cortical potential → myoclonic jerk; writing, walking, reaching all trigger; sensory-cortical hyperexcitability", "management": "Piracetam (8-16 g/day) specifically reduces action myoclonus; occupational therapy to adapt fine motor activities; weighted utensils; non-slip surfaces; myoclonus diary for severity tracking"},
            {"trigger": "Tactile / Auditory Startle", "prevalence_pct": 48, "mechanism": "Startle-reflex hyperexcitability from cortical NCL hyperexcitability; loud noise or unexpected touch precipitates myoclonic jerk or GTCS; heightened ambient sensory sensitivity", "management": "Low-sensory classroom environments; quiet spaces for rest; startle-reflex assessment by EEG; clonazepam or piracetam for startle-myoclonus management; caregiver and school staff education"},
            {"trigger": "CLN9-Contraindicated Drug Exposure", "prevalence_pct": 100, "mechanism": "CBZ/OXC/PHT → acute myoclonus worsening (Na-channel blockade in NCL); VGB → accelerated retinal blindness (retinal NCL + VGB retinopathy combined); TGB → NCSE (GABA reuptake inhibition in generalised epilepsy)", "management": "MedicAlert bracelet listing CLN9 CIs; AED card (GP/A&E/school); all prescribers reviewed at each encounter; pharmacist medication reconciliation"}
        ],
        "treatments": [
            {
                "drug": "Valproate / Sodium Valproate (VPA)",
                "level": "Level B",
                "dose": "Paediatric: 20-40 mg/kg/day; target trough 60-100 µg/mL; modified-release formulation preferred; BD or TDS dosing",
                "moa": "Broad-spectrum AED: Na-channel block + GABA enhancement + T-type Ca2+ block + HCN channel effects; full-spectrum coverage (GTCS + myoclonic + absence-like + occipital focal)",
                "efficacy": "GTCS reduction ~60%; myoclonus reduction ~50% in CLN9 (combined VPA+piracetam+LEV achieves best myoclonus control); absence-like seizures respond well",
                "monitoring": "TDM trough q6m; LFT + FBC q6m; weight; ammonia if encephalopathic; teratogenicity counselling (VPPP if female of reproductive age — though most CLN9 patients do not reach reproductive age)",
                "cln9_note": "VPA SAFE in CLN9 (ER ceramide synthesis defect, NOT mitochondrial). POLG1 mandatory exclusion before VPA in all CLN9-phenotype juvenile progressive epilepsies. MERRF exclusion (m.8344A>G + muscle biopsy) if ataxia + elevated lactate."
            },
            {
                "drug": "Levetiracetam (LEV)",
                "level": "Level B",
                "dose": "Paediatric: 20-60 mg/kg/day (max 3000 mg/day); BD dosing; IV LEV for SE (40-60 mg/kg loading, max 4500 mg)",
                "moa": "SV2A synaptic vesicle protein modulation → reduces neurotransmitter release from sensitised neurons; broad-spectrum coverage without Na-channel blockade",
                "efficacy": "Synergistic with VPA in CLN9 (VPA+LEV combination reduces GTCS cluster frequency); IV LEV essential for SE rescue when fosphenytoin is CI",
                "monitoring": "No TDM required; behavioural side effects (irritability, aggression) monitored 4-6 weeks post-initiation; dose reduce if significant irritability (risk of worsening cognitive/behavioural NCL symptoms)",
                "cln9_note": "IV LEV is FIRST-LINE for CLN9 status epilepticus (fosphenytoin ABSOLUTE CI — Na-channel blocker). IV LEV + IV phenobarbitone = CLN9 SE protocol. No hepatotoxicity (safe with VPA)."
            },
            {
                "drug": "Piracetam",
                "level": "Level C",
                "dose": "Paediatric: 8-16 g/day in divided doses (BD or TDS); adult equivalent; titrate slowly to effect",
                "moa": "AMPA-receptor modulator; membrane stabiliser; antiplatelet (mild); reduces cortical excitability at motor planning level; specifically attenuates action myoclonus (cortical origin)",
                "efficacy": "Level C evidence for cortical action myoclonus in juvenile NCL (extrapolated from EPM1/ULD piracetam Level B data); clinical benefit in CLN9 myoclonus documented in individual case series",
                "monitoring": "Well-tolerated in paediatric NCL; monitor coagulation if used with antiplatelet therapy (mild antiplatelet effect); no hepatotoxicity; no TDM",
                "cln9_note": "Piracetam is the most targeted anti-myoclonic agent in CLN9 action myoclonus. Start early before myoclonus becomes disabling. Combine with VPA + LEV backbone for maximal myoclonus control. Action myoclonus diary for response tracking."
            },
            {
                "drug": "Clobazam (CLB)",
                "level": "Level B",
                "dose": "Paediatric: 0.1-0.3 mg/kg/day (max 10-30 mg/day); BD dosing; nocturnal GTCS → night-time dosing bias",
                "moa": "1,5-benzodiazepine; GABA-A positive allosteric modulator (alpha-2/alpha-3 subunit selective vs diazepam); less sedation than 1,4-BZDs; broad-spectrum coverage",
                "efficacy": "Adjunct for refractory GTCS and myoclonus; nocturnal GTCS reduction; tolerance may develop (drug holiday consideration every 3-6 months); better tolerability than clonazepam in children",
                "monitoring": "Sedation and cognition (CLN9 cognitive impairment may be worsened); tolerance development; interactions with VPA (displacement at protein binding); no TDM routinely",
                "cln9_note": "CLB nocturnal dosing particularly effective for nocturnal GTCS in CLN9. Tolerance: drug holiday (2-week break every 3 months) can restore efficacy. Avoid clonazepam (longer-acting, more tolerance, more sedation)."
            },
            {
                "drug": "Lamotrigine (LTG — adjunct only)",
                "level": "Level B",
                "dose": "Paediatric adjunct to VPA: start 0.15 mg/kg/day, titrate SLOWLY (risk of rash with VPA); target 50-150 mg/day; titration >8 weeks",
                "moa": "Na-channel block (state-dependent); inhibits glutamate release; may have minor T-type Ca2+ effect; broad-spectrum at high doses",
                "efficacy": "Adjunct for refractory GTCS and focal occipital seizures in CLN9; NEVER as monotherapy (worsens myoclonus); safe as low-dose add-on to VPA+LEV backbone",
                "monitoring": "Rash monitoring (Stevens-Johnson risk with rapid VPA co-administration titration); TDM optional (target 4-12 µg/mL); cognitive effects (generally well-tolerated in children)",
                "cln9_note": "LTG ADJUNCT ONLY — NEVER MONOTHERAPY in CLN9 (Na-channel blocker monotherapy worsens myoclonus at high doses). Low-dose LTG (50-100 mg) added to VPA+LEV is safe and can reduce focal occipital seizure frequency."
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "Level C",
                "dose": "Classic 4:1 or 3:1 ratio (fat:carbohydrate+protein); initiated under dietitian supervision; target urine ketones 2-4 mmol/L",
                "moa": "Metabolic ketosis → alternative neural energy substrate; reduces glucose-dependent neuronal excitability; mitochondrial biogenesis enhancement; potential ceramide-pathway modulation (sphingolipid effects of KD under investigation)",
                "efficacy": "Level C for drug-resistant juvenile NCL epilepsy; potential additional benefit in CLN9 via ceramide/sphingolipid metabolic interaction (ceramide levels influenced by dietary fat composition — hypothesis); GTCS reduction ~30-40% in drug-resistant NCL",
                "monitoring": "Lipid profile; growth monitoring; renal ultrasound (renal stones risk); bone density; glucose; KD-specific dietitian follow-up monthly initially",
                "cln9_note": "KD may have dual benefit in CLN9: (1) Anticonvulsant (standard ketosis mechanism); (2) Ceramide pathway modulation (dietary fatty acids influence sphingolipid synthesis — DEGS1 substrate availability). This is biologically plausible but not proven. Always attempt ≥3 AED trials before KD in children."
            },
            {
                "drug": "MDT Palliative Care (Ophthalmology + Physiotherapy + OT + Dietitian + Neuropsychology + Neuro-Ophthalmology)",
                "level": "Level A",
                "dose": "Paediatric MDT: ophthalmology 6-monthly (visual acuity + ERG + VEP); SARA + UMRS 6-monthly; physiotherapy weekly; OT monthly; dietitian 3-monthly; psychology 6-monthly",
                "moa": "Multidisciplinary rehabilitation addresses compound disability: visual impairment + myoclonus + ataxia + cognitive decline + behavioural change; maintains function and quality of life; family/caregiver support",
                "efficacy": "Level A evidence for MDT in rare paediatric progressive neurological conditions; functional maintenance, fall prevention, visual rehabilitation, communication aids all evidence-based components",
                "monitoring": "SARA (cerebellar ataxia), UMRS (myoclonus), MoCA/cognitive assessments, visual acuity + Goldmann visual field annually, SUDEP risk assessment annually, advance care planning milestones",
                "cln9_note": "Visual impairment in CLN9 (retinal NCL, 90%) requires early ophthalmology and low-vision rehabilitation. Unlike CLN12/CLN13 (no retinal NCL), CLN9 has severe visual failure — visual rehabilitation, mobility training, and white cane/guide dog planning from early in disease course."
            },
            {
                "drug": "Rescue Midazolam (Buccal/IM) + IV LEV (SE Protocol)",
                "level": "Level A",
                "dose": "Buccal midazolam: 0.3-0.5 mg/kg (max 10 mg); IM midazolam: 0.15-0.3 mg/kg; IV LEV SE protocol: 40-60 mg/kg (max 4500 mg) loading",
                "moa": "Midazolam: rapid GABA-A potentiation → seizure termination; IV LEV: SV2A modulation → SE interruption; replaces fosphenytoin in CLN9 SE protocol",
                "efficacy": "Standard of care for acute seizure rescue and SE in paediatric NCL; buccal midazolam equivalent to IV diazepam for prolonged seizures; IV LEV replaces fosphenytoin (Na-channel blocker — ABSOLUTE CI in CLN9)",
                "monitoring": "Respiratory depression monitoring after buccal midazolam; prescribe in individual rescue plan; school and carer training on administration; A&E CLN9 alert card (fosphenytoin CI explicitly listed)",
                "cln9_note": "FOSPHENYTOIN ABSOLUTE CI in CLN9 SE — Na-channel blocker precipitates acute myoclonus escalation. CLN9 SE protocol: buccal midazolam (first response) → IV LEV (second-line) → IV phenobarbitone (third-line, not phenytoin). Embed CLN9 SE protocol in school emergency plan, hospital notes, and A&E alert."
            }
        ],
        "contraindications": [
            {
                "drug": "Vigabatrin (VGB)",
                "severity": "ABSOLUTE CI",
                "reason": "VGB irreversible retinopathy (peripheral visual field constriction) superimposed on CLN9 retinal NCL degeneration (90%) = catastrophic, irreversible accelerated bilateral blindness",
                "note": "CLN9 is the opposite of CLN12/CLN13 — retinal NCL is PRESENT (90%) → VGB = ABSOLUTE CI. If West syndrome overlap suspected in CLN9 (rare): ACTH or prednisolone ONLY — never VGB. Document VGB CI in all A&E records, AED card, school health plan, GP alert."
            },
            {
                "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
                "severity": "ABSOLUTE CI",
                "reason": "Na-channel blockers WORSEN myoclonus in CLN9; GTCS at age 6-7y misidentified as focal epilepsy → CBZ → acute myoclonic deterioration; most dangerous prescribing error in newly presenting CLN9",
                "note": "The most common real-world error in CLN9 diagnosis delay: child with GTCS + developmental concerns → CBZ prescribed → myoclonus emerges → CBZ not stopped for months. Any child on CBZ/OXC/PHT who develops myoclonus: urgent NCL workup."
            },
            {
                "drug": "Fosphenytoin (IV) / Phenytoin (IV)",
                "severity": "ABSOLUTE CI",
                "reason": "IV Na-channel blocker used in standard SE protocols → acute myoclonus worsening and seizure exacerbation in CLN9; standard SE second-line drug = catastrophic in CLN9 SE",
                "note": "CLN9 SE protocol: IV LEV REPLACES fosphenytoin (second-line). Embed in A&E protocol, school emergency plan, and hospital notes. Paramedics and A&E must be informed of CLN9 SE protocol at every hospital contact."
            },
            {
                "drug": "Tiagabine (TGB)",
                "severity": "ABSOLUTE CI",
                "reason": "GABA reuptake inhibitor → NCSE in generalised epilepsy syndromes; CLN9 has 20% NCSE risk — TGB absolutely forbidden",
                "note": "TGB approved only for focal seizures in adults; causes NCSE in PME/generalised epilepsy including CLN9. If focal seizures in CLN9 (occipital) misidentified as focal epilepsy → TGB prescribed → NCSE → emergency presentation with altered consciousness misattributed to encephalopathic deterioration."
            },
            {
                "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
                "severity": "HIGH RISK",
                "reason": "Alpha-2-delta calcium channel subunit modulation can paradoxically worsen cortical myoclonus in NCL; pain/anxiety indication may lead non-specialist to prescribe GBP/PGB in CLN9 without NCL awareness",
                "note": "GBP/PGB are not age-appropriate first-line choices in paediatric CLN9. If prescribed for pain or spasticity: close myoclonus monitoring; stop if myoclonus worsens. Safer alternatives for pain: paracetamol, low-dose ibuprofen, neuropathic pain specialist review."
            },
            {
                "drug": "Lamotrigine (LTG) Monotherapy",
                "severity": "HIGH RISK",
                "reason": "LTG monotherapy at high doses paradoxically worsens cortical myoclonus in NCL; safe only as low-dose adjunct to VPA+LEV backbone",
                "note": "NEVER LTG as sole AED in CLN9. LTG adjunct (50-150 mg/day) with VPA+LEV is safe and can help focal occipital seizures. Rapid titration with VPA risks Stevens-Johnson — slow titration mandatory."
            },
            {
                "drug": "AED Taper / Abrupt Discontinuation",
                "severity": "HIGH RISK",
                "reason": "CLN9 is fatal progressive neurodegenerative disease — seizures do NOT remit; AED taper = inevitable severe GTCS cluster + SUDEP risk",
                "note": "Any AED change in CLN9 requires specialist NCL centre supervision. Maximum 10% dose reduction per month (palliative phase only). Dementia stage: AED management by caregiver — never delegate to patient. Abrupt discontinuation risk: GTCS cluster, SE, SUDEP."
            }
        ],
        "monitoring": [
            {"item": "DEGS1 WES / NCL Gene Panel (DEGS1 + CLN3 exon7+8 + full NCL panel)", "frequency": "At diagnosis", "note": "CLN3 exon7+8 deletion PCR FIRST (days; most common juvenile NCL). If negative: DEGS1 WES + full NCL gene panel. Research fibroblast dihydroceramide desaturase assay (specialist NCL lab) if WES negative but clinical/EM consistent with CLN9. NCL Resource gene discovery enrolment for WES-negative cases."},
            {"item": "Peripheral Blood Smear (Vacuolated Lymphocytes)", "frequency": "At diagnosis (bedside test)", "note": "MOST RAPID CLN3 DIFFERENTIAL: vacuolated lymphocytes present → CLN3 highly likely (PCR next); absent → CLN9/CLN5/CLN6/CLN7/CLN8 differential (EM + WES pathway). Available in ANY laboratory — minutes to result. Must be done at FIRST clinical encounter in suspected juvenile NCL."},
            {"item": "PPT1 DBS + TPP1 DBS Enzyme Assays", "frequency": "At diagnosis", "note": "PPT1 (CLN1) and TPP1 (CLN2) DBS enzyme assays (days) — MANDATORY exclusion in any juvenile NCL before proceeding to WES. Normal result → CLN1/CLN2 excluded. No DEGS1 DBS assay exists — CLN9 cannot be confirmed or excluded by DBS enzyme testing."},
            {"item": "Skin Biopsy Electron Microscopy (Mixed FP + CB + Vacuoles)", "frequency": "At diagnosis", "note": "Mixed/pleomorphic EM pattern (FP + CB + vacuoles, no dominant single pattern) = CLN9 diagnostic signature. Distinguishes CLN9 from CLN2 (CB-dominant), CLN3 (FP-dominant), CLN1 (GRODs-dominant). EM result in days. If mixed pattern + normal PPT1/TPP1 DBS + negative CLN3 PCR → CLN9 highly likely → WES confirmation."},
            {"item": "Research Fibroblast Dihydroceramide Desaturase Assay", "frequency": "At diagnosis (specialist lab)", "note": "Biochemical confirmation of DEGS1 deficiency (Schulz 2004 Biochem J method). Fibroblast culture from skin biopsy sent to specialist NCL research lab (NCL Resource network). Demonstrates reduced dihydroceramide→ceramide conversion. Supports CLN9 diagnosis when WES is negative or inconclusive."},
            {"item": "Ophthalmology (ERG + VEP + Visual Acuity + Goldmann Field)", "frequency": "Every 6 months", "note": "CLN9 has retinal NCL (90%) — 6-monthly ophthalmology mandatory (contrast with CLN13 annual, since CLN13 has NO retinal NCL). ERG: progressive amplitude reduction (retinal photoreceptor loss); VEP: cortical visual pathway involvement; Goldmann: peripheral visual field loss (concentric constriction); VA: central vision assessment. Visual failure is a core CLN9 feature — 6-monthly mandatory from diagnosis."},
            {"item": "Brain MRI (3T NCL Protocol + Spectroscopy)", "frequency": "6-monthly initially, annually stable", "note": "Cerebral and cerebellar atrophy progression; cortical thinning; white matter signal (distinguishes from CERS1-PMEA white matter changes); MRS: N-acetylaspartate (NAA) reduction (neuronal loss); MRS may reveal altered ceramide-related metabolite profiles (research interest). Thalamic signal (differentiates from other NCL patterns)."},
            {"item": "EEG (Resting Annual + Photic + Urgent NCSE)", "frequency": "Baseline + annual + urgent if deterioration", "note": "IPS photosensitivity (55%); progressive amplitude decrement with disease course; jerk-locked back-averaging for cortical myoclonus characterisation; annual subclinical NCSE detection; URGENT EEG for any acute cognitive deterioration (NCSE vs disease progression). Occipital paroxysmal activity correlates with retinal NCL involvement."},
            {"item": "POLG1 WES + MERRF Mitochondrial DNA Testing", "frequency": "At diagnosis (mandatory before VPA)", "note": "POLG1 Alpers and MERRF both cause juvenile progressive epilepsy with regression + ataxia + myoclonus — IDENTICAL phenotype to CLN9. VPA ABSOLUTE CI in POLG1/MERRF. MANDATORY before any VPA: blood lactate + m.8344A>G MERRF PCR + POLG1 WES + muscle biopsy (ragged-red fibres for MERRF). This is non-negotiable — VPA-induced hepatic failure in POLG1 is fatal."},
            {"item": "SARA Scale (Cerebellar Ataxia)", "frequency": "6-monthly", "note": "Cerebellar ataxia in 78% CLN9; SARA progression guides physiotherapy intensity, walking aid prescription, wheelchair timing; SARA combined with visual impairment score gives composite fall risk; physiotherapy referral triggered by SARA ≥10/40."},
            {"item": "UMRS (Unified Myoclonus Rating Scale)", "frequency": "6-monthly", "note": "Action myoclonus quantification; piracetam dose titration guide; functional myoclonus impact (ADL, writing, eating, walking); cortical SEP correlation; UMRS guides AED optimisation; UMRS ≥20 triggers piracetam increase and LEV escalation."},
            {"item": "Neuropsychological Assessment", "frequency": "Annual", "note": "Track cognitive trajectory; WISC-V/Bayley-III (age-appropriate); adaptive function; school capacity; VABS-3 (vineland adaptive scales); school support plan (EHCP/IEP); communication AAC assessment as cognitive decline progresses; psychology support for child and siblings."},
            {"item": "SUDEP Risk Assessment + Nocturnal Monitoring", "frequency": "Annual", "note": "Drug-resistant nocturnal GTCS (72% drug-resistant) + cognitive impairment (inability to self-reposition post-seizure) = elevated SUDEP risk. Nocturnal seizure alarms (SOMO bed sensor or equivalent); safe sleeping position (lateral); seizure mat; SUDEP Action counselling for parents; sibling awareness."},
            {"item": "BDSRA + NCL Resource + Gene Discovery Registration", "frequency": "At diagnosis + annual updates", "note": "BDSRA enrolment mandatory (paediatric NCL support + future therapy trial eligibility). NCL Resource gene discovery programme enrolment — CLN9 gene provisional, research collaboration essential for molecular confirmation. Annual updates for research trial recruitment. Family support network through BDSRA."}
        ]
    }


def get_definitions():
    return {
        "disease_name": "CLN9 — Neuronal Ceroid Lipofuscinosis Type 9 (Provisional) / DEGS1 / Belgrade-Variant NCL",
        "gene_full": "DEGS1 (Dihydroceramide Desaturase 1) — 1q42.11 (Provisional gene assignment)",
        "omim_gene": "*615105 (DEGS1) — Provisional",
        "omim_disease": "#609055 (CLN9 — Neuronal Ceroid Lipofuscinosis Type 9, Provisional)",
        "protein_full": "Dihydroceramide Desaturase 1 (DEGS1/DES1); 323 aa; ~38 kDa; ER-membrane resident enzyme; fatty acid desaturase superfamily; His-box iron coordination (His135/His170/His172/His239); catalyzes 4,5-trans desaturation of dihydroceramide → ceramide; FINAL step in de novo ceramide biosynthesis; UNIQUE ER location among all NCL proteins (all others lysosomal)",
        "inheritance_mode": "Autosomal recessive (AR) biallelic DEGS1 LOF → CLN9 (provisional). 25% sibling recurrence. No AD form.",
        "onset_age": "Juvenile: mean 6.8 years seizure onset (range 4-10 years); visual failure mean 5.8 years (concurrent, NOT sequential)",
        "em_pattern": "MIXED/PLEOMORPHIC: FP (~65%) + CB (~55%) + vacuoles (~45%) + minor GRODs (~20%); NO single dominant pattern; vacuolated lymphocytes ABSENT in blood smear",
        "no_vacuolated_lymphocytes": "CONFIRMED ABSENT — NO vacuolated lymphocytes in peripheral blood smear (present in CLN3 = PATHOGNOMONIC). Blood smear is the fastest CLN9 vs CLN3 differentiator.",
        "retinal_ncl_present": "CONFIRMED — CLN9 HAS RETINAL NCL (90%). VGB = ABSOLUTE CI. Visual failure concurrent with seizures onset. Contrast with CLN12 and CLN13 (no retinal NCL → VGB not absolute CI).",
        "key_concepts": [
            {
                "name": "CLN9-DEGS1-1q42.11-ER-Ceramide-Desaturase-Provisional-Juvenile-NCL",
                "definition": "CLN9 is caused by biallelic LOF of DEGS1 (Dihydroceramide Desaturase 1, 1q42.11) encoding an ER-membrane ceramide biosynthesis enzyme. DEGS1 LOF → dihydroceramide accumulates + ceramide deficiency → pleomorphic NCL storage → juvenile-onset NCL. PROVISIONAL: gene-disease link is biochemical (Schulz 2004), not definitively confirmed by modern sequencing. DEGS1 is unique: the ONLY NCL with an ER ceramide biosynthesis defect (all other NCLs = lysosomal)."
            },
            {
                "name": "Provisional-CLN9-Gene-Assignment-DEGS1-WES-May-Be-Negative-NCL-Resource-Mandatory",
                "definition": "CLN9 gene = DEGS1 is PROVISIONAL. Original Schulz 2004 description: biochemical dihydroceramide desaturase deficiency in Belgrade NCL fibroblasts; gene NOT definitively sequenced. WES may NOT find DEGS1 variants in all CLN9-phenotype cases. Clinical implication: (1) Standard NCL gene panel may miss CLN9; (2) Fibroblast dihydroceramide desaturase assay (research lab) supports biochemical diagnosis when WES negative; (3) NCL Resource gene discovery enrolment mandatory for all CLN9-phenotype patients; (4) CLN9 may be genetically heterogeneous."
            },
            {
                "name": "No-Vacuolated-Lymphocytes-CLN9-Critical-CLN3-Differential-Blood-Smear-First",
                "definition": "CLN3 (Juvenile Batten disease) → vacuolated lymphocytes PATHOGNOMONIC in peripheral blood smear. CLN9 → NO vacuolated lymphocytes. Blood smear takes minutes in ANY lab. Protocol: blood smear FIRST → if vacuolated lymphocytes present → CLN3 (PCR next, days); if ABSENT → CLN9/CLN5/CLN6/CLN7/CLN8 differential (EM + WES pathway). This is the fastest, cheapest, most accessible CLN3 vs CLN9 differentiator available to any clinician worldwide."
            },
            {
                "name": "VGB-ABSOLUTE-CI-CLN9-Retinal-NCL-90pct-Concurrent-Visual-Failure",
                "definition": "CLN9 has retinal NCL degeneration in ~90% of patients. VGB (vigabatrin) retinopathy superimposed on CLN9 retinal NCL = catastrophic combined blindness. VGB = ABSOLUTE CI in CLN9. This CONTRASTS with CLN12 and CLN13 (the only NCLs where VGB is NOT absolute CI because they lack retinal degeneration). CLN9 follows the same VGB ABSOLUTE CI rule as CLN1/CLN2/CLN3/CLN5/CLN6/CLN7/CLN8/CLN10/CLN11."
            },
            {
                "name": "No-DEGS1-DBS-Enzyme-Assay-WES-Plus-Fibroblast-Biochemistry-Required-Provisional",
                "definition": "No standardised DEGS1 DBS enzyme assay exists (unlike CLN1/PPT1-DBS and CLN2/TPP1-DBS which give results in days). CLN9 diagnostic pathway: (1) Blood smear (vacuolated lymphocytes? → CLN3 or CLN9); (2) PPT1 + TPP1 DBS (exclude CLN1/CLN2); (3) CLN3 exon7+8 del PCR; (4) Skin biopsy EM (mixed FP+CB confirms NCL class); (5) WES NCL panel including DEGS1; (6) Research fibroblast dihydroceramide desaturase assay (Schulz 2004 method) for biochemical confirmation. This multi-step pathway reflects the provisional gene status."
            },
            {
                "name": "Mixed-Pleomorphic-EM-CLN9-FP-CB-Vacuoles-No-Dominant-Pattern",
                "definition": "CLN9 EM skin biopsy shows MIXED/PLEOMORPHIC storage: FP (fingerprint profiles, ~65%) + CB (curvilinear bodies, ~55%) + vacuoles (~45%) + minor GRODs (~20%). No single dominant NCL-type storage pattern. This MIXED profile in a juvenile NCL patient is the EM clue for CLN9 — cannot be assigned to CLN1 (GRODs-dominant), CLN2 (CB-dominant), CLN3 (FP-dominant), or other single-pattern NCLs. Mixed pattern + normal PPT1/TPP1 DBS + negative CLN3 PCR = CLN9 pathway."
            },
            {
                "name": "Concurrent-Visual-Failure-Seizures-CLN9-vs-Sequential-CLN3",
                "definition": "In CLN3: visual failure PRECEDES seizures by 2-5 years (visual at 4-10y, seizures at 10-13y). In CLN9: visual failure and seizures onset CONCURRENTLY (within months, mean 6-7 years). A child with SIMULTANEOUS visual deterioration AND new seizures at age 6-7 → CLN9 more likely than CLN3. This concurrent onset pattern is a clinical differentiator between the two most common juvenile NCL types."
            },
            {
                "name": "CBZ-OXC-PHT-ABSOLUTE-CI-CLN9-Juvenile-NCL-GTCS-Misidentification-Trap",
                "definition": "CLN9 GTCS onset at age 6-8 years is frequently misidentified as childhood-onset focal epilepsy or idiopathic GGE → CBZ/OXC prescribed → ACUTE MYOCLONIC WORSENING. Mean CLN9 diagnosis delay 3.4 years = extended Na-channel blocker exposure window. Any child on CBZ/OXC developing myoclonus: urgent NCL workup. Safe AEDs: VPA + LEV + piracetam (broad PME coverage, no myoclonus exacerbation)."
            },
            {
                "name": "VPA-SAFE-CLN9-ER-Ceramide-NOT-Mitochondrial-POLG1-Exclusion-Mandatory",
                "definition": "Valproate is SAFE in CLN9. DEGS1 is an ER ceramide biosynthesis enzyme (NOT mitochondrial). VPA ABSOLUTE CI applies to MERRF (mitochondrial myoclonus epilepsy) and POLG1 Alpers — both mimic CLN9 (juvenile progressive epilepsy + ataxia + cognitive decline). MANDATORY before VPA: blood lactate + m.8344A>G MERRF PCR + POLG1 WES + muscle biopsy (RRF). Once excluded: VPA is backbone AED for CLN9."
            },
            {
                "name": "Ceramide-Pathway-DEGS1-CLN9-and-CERS1-PMEA-Unique-NCL-Metabolic-Pair",
                "definition": "CLN9/DEGS1 and CERS1-PMEA are the ONLY two progressive epilepsies affecting the ceramide biosynthesis pathway. CERS1 (upstream): synthesises dihydroceramide; DEGS1 (downstream): desaturates dihydroceramide → ceramide. CERS1 LOF → deficient dihydroceramide → ceramide deficiency. DEGS1 LOF → dihydroceramide accumulation → ceramide deficiency. Both ER-based, both NCL-adjacent progressive epilepsies. Key difference: CLN9/DEGS1 = juvenile onset + visual failure + VGB absolute CI; CERS1-PMEA = young adult onset + cerebellar-dominant + VGB avoided (not absolute CI). Neither is lysosomal — both are unique among NCL diseases."
            },
            {
                "name": "Fosphenytoin-ABSOLUTE-CI-CLN9-SE-Protocol-IV-LEV-Replaces",
                "definition": "Standard status epilepticus protocol second-line drug = fosphenytoin (IV phenytoin prodrug) → Na-channel blocker → acute myoclonus worsening in CLN9. REPLACE WITH IV LEV (40-60 mg/kg) as second-line SE drug in all CLN9 SE protocols. Embed in A&E protocol, NICU protocol, school emergency plan, hospital notes. Paramedics must be informed at every hospital contact. CLN9 SE sequence: buccal midazolam → IV LEV → IV phenobarbitone (NOT fosphenytoin)."
            },
            {
                "name": "KD-Ceramide-Pathway-Hypothesis-CLN9-Dual-Mechanism",
                "definition": "Ketogenic diet in CLN9 may have dual mechanism: (1) Standard anticonvulsant effect (metabolic ketosis, neuronal excitability reduction — evidence in all drug-resistant NCL); (2) Ceramide pathway modulation (dietary fatty acid availability influences sphingolipid synthesis — DEGS1 substrate dihydroceramide levels may be modulated by KD composition). This is biologically plausible given DEGS1's ER ceramide desaturase function but NOT proven clinically. KD remains Level C in CLN9 but is biologically the most interesting adjunct therapy."
            },
            {
                "name": "No-Disease-Modifying-Therapy-CLN9-Gene-Identity-Ambiguity-Complicates-Target",
                "definition": "No approved disease-modifying therapy for CLN9 (2026). DEGS1 gene assignment provisional = complicates target identification. Unlike CLN2 (cerliponase ERT approved 2017) and CLN3 (gene therapy Phase 1 trials), CLN9 has no active therapy pipeline. Ceramide supplementation approaches and DEGS1 gene therapy are conceptually explored but no clinical-stage programs. BDSRA enrolment mandatory. NCL Resource gene discovery programme crucial — gene confirmation is prerequisite for targeted therapy development."
            },
            {
                "name": "Visual-Rehabilitation-CLN9-Retinal-NCL-90pct-Unlike-CLN12-CLN13",
                "definition": "CLN9 has retinal NCL degeneration (~90%) with visual failure concurrent with seizures onset (mean 5.8 years). Unlike CLN12 and CLN13 (no retinal NCL, no visual failure), CLN9 requires active visual rehabilitation from diagnosis: low-vision services, mobility training, white cane, guide dog planning, tactile communication, Braille preparation. Ophthalmology 6-monthly mandatory (contrast with CLN13 annual). Retinal degeneration in CLN9 progresses relentlessly — expect near-total visual loss within 5-8 years of onset."
            },
            {
                "name": "SUDEP-Risk-CLN9-Drug-Resistant-Nocturnal-GTCS-Childhood",
                "definition": "CLN9 carries elevated SUDEP risk from drug-resistant nocturnal GTCS (72% drug-resistant) combined with cognitive impairment (inability to self-reposition post-seizure) and visual impairment (cannot summon help). Nocturnal seizure alarms mandatory (bed sensor). Safe sleeping position (lateral). Supervised sleeping environment. SUDEP Action counselling for parents and caregivers. Annual SUDEP risk review. Sibling awareness and CPR training."
            },
            {
                "name": "Belgrade-Yugoslav-Founder-Population-CLN9-Serbian-Ancestry-Alert",
                "definition": "The original CLN9 cohort described by Schulz 2004 consisted of patients from Belgrade/former Yugoslavia (Serbian/Yugoslav ancestry). Belgrade founder variants likely exist in DEGS1 or the true CLN9 gene (not yet confirmed by WES). Clinically: any juvenile NCL patient with Serbian, Balkan, or former-Yugoslav ancestry → raise CLN9 suspicion alongside CLN3 differential. Consanguinity rate elevated in Belgrade cohort (~65%). No specific PCR founder test currently available (unlike CLN3 exon7+8 del)."
            }
        ],
        "thresholds": [
            {"parameter": "PPT1 DBS enzyme assay (to exclude CLN1)", "value": "Normal (within age-adjusted reference range)", "action": "CLN1 excluded; proceed to TPP1 DBS; if mixed EM → CLN9 pathway"},
            {"parameter": "TPP1 DBS enzyme assay (to exclude CLN2)", "value": "Normal (within age-adjusted reference range)", "action": "CLN2 excluded; proceed to CLN3 exon7+8 del PCR; if negative → CLN9 WES pathway"},
            {"parameter": "CLN3 exon7+8 deletion PCR", "value": "Negative (no deletion)", "action": "CLN3 most common juvenile NCL excluded; blood smear vacuolated lymphocytes absent + negative PCR → CLN9 highly likely → EM + WES + fibroblast assay"},
            {"parameter": "Fibroblast dihydroceramide desaturase activity", "value": "<20% of control activity", "action": "Biochemical CLN9 confirmation (Schulz 2004 reference range); research lab result; confirms DEGS1/dihydroceramide pathway defect regardless of WES result"},
            {"parameter": "Blood lactate (POLG1/MERRF exclusion)", "value": "<2.0 mmol/L (normal)", "action": "Mitochondrial disease less likely; still perform full POLG1 WES + m.8344A>G before VPA — normal lactate does not exclude all mitochondrial PME"},
            {"parameter": "VPA trough level", "value": "60-100 µg/mL", "action": "Therapeutic range; <60 → increase dose; >120 → toxicity watch (tremor, encephalopathy, ammonia)"},
            {"parameter": "UMRS myoclonus severity", "value": "≥15/60", "action": "Significant action myoclonus → increase piracetam to 16 g/day; LEV escalation; physiotherapy for myoclonus-related falls; OT adaptive equipment"},
            {"parameter": "SARA cerebellar ataxia severity", "value": "≥10/40", "action": "Significant ataxia → walking aids (rollator); physiotherapy 2×/week; fall prevention flooring; compound fall risk assessment with visual impairment score"},
            {"parameter": "ERG amplitude (retinal NCL monitoring)", "value": "Any reduction >20% from baseline", "action": "Accelerating retinal NCL progression; intensify low-vision services; white cane/mobility training; Braille preparation; guide dog referral planning"},
            {"parameter": "Goldmann visual field", "value": "Central residual field <10° (tubular vision)", "action": "Severe visual impairment; mobility training for severely impaired; tactile communication systems; school curriculum adaptation (tactile + auditory learning)"},
            {"parameter": "MoCA / cognitive assessment", "value": "≤22/30 (paediatric equivalent adapted)", "action": "Cognitive impairment documented; school EHCP/IEP review; communication AAC assessment; adaptive equipment; caregiver-managed AED dosing; ACP initiation"},
            {"parameter": "SUDEP risk — GTCS frequency", "value": "≥2 nocturnal GTCS per month despite ≥2 AED trials", "action": "High SUDEP risk: nocturnal bed sensor mandatory; safe sleeping position; SUDEP Action counselling; carer awareness; seizure response training; consider third AED or KD"}
        ],
        "standards": [
            "Schulz A et al. 2004 Biochem J — First CLN9/dihydroceramide desaturase link; Belgrade/Yugoslav cohort; biochemical CLN9 characterisation",
            "Pant DC et al. 2019 Am J Hum Genet — DEGS1 variants → hypomyelinating leukodystrophy HLD18; note distinct phenotype from Belgrade CLN9",
            "Mole SE et al. 2019 Lancet Neurology — NCL classification review; CLN9 provisional designation",
            "NCL Resource 2024 (ncl.mrc.ac.uk) — Current NCL standards; CLN9 listed as provisional",
            "Baumann N et al. 1983 — Original Belgrade NCL clinical descriptions",
            "ILAE 2022 — Seizure type and epilepsy syndrome classification",
            "NICE NG217 — Epilepsy management guidelines",
            "MHRA VPPP 2021 — Valproate Pregnancy Prevention Programme",
            "CPIC POLG1 2023 — POLG1 VPA prescribing guidance",
            "ACMG-AMP 2015 — Variant pathogenicity classification",
            "BDSRA Registry 2024 — Batten Disease Support and Research Association",
            "WHO-ICF 2019 — International Classification of Functioning Disability and Health"
        ],
        "references": [
            "Schulz A et al. (2004) NCL-associated mutations in CLN9 protein lead to reduced dihydroceramide desaturase activity. Biochem J 375:513-521 [First CLN9 biochemical characterisation; Belgrade/Yugoslav patients]",
            "Pant DC et al. (2019) Loss of the sphingolipid desaturase DEGS1 causes hypomyelinating leukodystrophy. J Clin Invest 129:1240-1256 [DEGS1 HLD18 — distinct from CLN9 phenotype]",
            "Mole SE & Anderson G (2019) Unreported cases of NCL: the dark matter. Lancet Neurol 18:1003-4 [NCL classification update]",
            "Berkovic SF et al. (1988) Kufs disease: a critical reappraisal. Brain 111:27-62 [Historical juvenile vs adult NCL classification]",
            "Santavuori P (1988) Neuronal ceroid lipofuscinoses in childhood. Brain Dev 10:80-3 [Historical NCL classification including Belgrade variant]",
            "Canafoglia L et al. (2014) Recessive mutations in the GRN gene cause NCL11. J Med Genet 51:411-6 [Adult NCL comparative reference]"
        ],
        "lifecycle_stages": [
            {
                "stage": "Prenatal / Pre-Symptomatic (Genetic Risk — Sibling of Proband)",
                "age_range": "Birth to ~4 years (if DEGS1 biallelic identified in family)",
                "description": "Identified through cascade testing of siblings of CLN9 proband. Genetic counselling for family (25% sibling risk — AR; provisional gene). No symptoms. Annual ophthalmology from age 3; annual cognitive monitoring from age 3. Enrol in BDSRA/NCL Resource pre-symptomatically. Fibroblast dihydroceramide desaturase assay available if WES inconclusive.",
                "priorities": ["Confirm DEGS1 biallelic genotype (or biochemical fibroblast assay)", "Cascade test siblings (25% AR risk)", "Genetic counselling (provisional gene — emphasise uncertainty)", "BDSRA/NCL Resource enrolment", "Annual ophthalmology from age 3 (pre-symptomatic retinal monitoring)"]
            },
            {
                "stage": "First Symptom — Diagnostic Emergency (Visual Failure + Seizures Concurrent)",
                "age_range": "4-10 years (mean 6-7 years)",
                "description": "CONCURRENT onset of visual failure and seizures (NOT sequential as in CLN3). Child with new visual complaints + seizures at age 6-7 → CLN9 differential priority. Immediate workup: blood smear (vacuolated lymphocytes?), PPT1+TPP1 DBS, CLN3 PCR, EM skin biopsy, POLG1/MERRF exclusion, WES + research fibroblast assay.",
                "priorities": ["Blood smear (vacuolated lymphocytes? → CLN3 or CLN9)", "PPT1 + TPP1 DBS enzyme assays", "CLN3 exon7+8 del PCR", "POLG1 WES + MERRF exclusion (mandatory before VPA)", "Skin biopsy EM (mixed pattern?)", "Start VPA + LEV (avoid CBZ/OXC/PHT/VGB)", "Ophthalmology urgent (ERG baseline)", "Enrol BDSRA + NCL Resource gene discovery"]
            },
            {
                "stage": "Active Epilepsy and Cognitive-Visual Decline",
                "age_range": "2-6 years from onset",
                "description": "Progressive drug-resistant epilepsy (72%). Worsening visual impairment. Cognitive decline. School capacity decreasing. Myoclonus emerging. MDT intensification. AED optimisation. Low-vision services. Physiotherapy for ataxia. OT adaptive equipment.",
                "priorities": ["AED optimisation (VPA + LEV + piracetam + CLB)", "Low-vision services + mobility training", "School EHCP/IEP: adapted curriculum (visual + cognitive)", "Physiotherapy (ataxia + myoclonus)", "SARA + UMRS + ERG monitoring 6-monthly", "KD consideration if ≥2 AEDs failed", "ACP initiation with family", "SUDEP risk plan + nocturnal alarm"]
            },
            {
                "stage": "Established Severe Disability (Near-Total Visual Loss + Drug-Resistant Epilepsy)",
                "age_range": "6-12 years from onset",
                "description": "Near-total visual loss (retinal NCL progressive). Wheelchair dependency. Severe drug-resistant epilepsy (SE risk). Significant cognitive impairment. Total care dependency increasing. Communication deteriorating. Advanced ACP. Supported living planning.",
                "priorities": ["Tactile/auditory communication systems (visual loss near-total)", "PEG consideration (dysphagia from neurological deterioration)", "Wheelchair + home adaptation", "Advanced ACP (DNACPR, preferred place)", "Palliative seizure management plan", "SUDEP monitoring (nocturnal sensor)", "Caregiver respite and health support", "Sibling psychological support"]
            },
            {
                "stage": "Late Palliative / End Stage",
                "age_range": "12-18 years from onset (2nd-3rd decade)",
                "description": "Complete visual loss. Severe cognitive impairment / unresponsive. Total care dependency. Severe refractory epilepsy with SE. Comfort care paramount. Palliative seizure protocol (SL/IM midazolam). End-of-life care plan activation.",
                "priorities": ["Comfort care (no new investigative procedures)", "SL/IM midazolam rescue SE protocol", "DNACPR active", "Preferred place of death documentation", "Carer/family bereavement preparation", "Brain donation consent (CLN9/DEGS1 research)"]
            },
            {
                "stage": "Bereavement and Family Follow-Up + Research",
                "age_range": "After death",
                "description": "Post-mortem brain and tissue donation for CLN9/DEGS1 research (gene discovery, ceramide biology, NCL Resource). Sibling cascade testing completion. BDSRA grief support. NCL Resource gene discovery results follow-up (WES-negative families — novel gene identification may emerge post-mortem).",
                "priorities": ["Brain + tissue donation (CLN9 gene discovery research — consent pre-death)", "Sibling cascade DEGS1/NCL testing", "Family genetic counselling (AR — 25% sibling risk confirmed or revised if novel gene)", "BDSRA bereavement programme", "NCL Resource gene discovery programme results communication to family", "CERS1-PMEA comparison data for ceramide pathway research"]
            }
        ]
    }
