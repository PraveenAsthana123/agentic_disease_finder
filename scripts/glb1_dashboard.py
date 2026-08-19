"""
GLB1 Epilepsy — GM1 Gangliosidosis / β-Galactosidase-1 Deficiency
===================================================================
40-patient cohort · GLB1 (3p22.3) · Autosomal recessive (AR) biallelic LOF
GLB1 encodes β-Galactosidase 1 (lysosomal acid β-galactosidase); 677 aa precursor ~88 kDa
(signal peptide 1-23; propeptide 24-69; mature ~64 kDa); GH family 35 glycoside hydrolase;
catalytic Glu268-Glu185 general acid-base pair; forms 1.3 MDa multienzyme complex with
CTSA (Cathepsin A / Protective Protein) and NEU1 (α-Neuraminidase 1) in the lysosome.

GLB1 FUNCTIONS:
  (1) Cleaves terminal β-galactose from GM1 ganglioside → GM2 ganglioside (key step)
  (2) Cleaves β-galactose from galactose-containing glycoproteins and keratan sulphate
  (3) Part of multienzyme complex: requires CTSA to protect GLB1 from premature
      intralysosomal cathepsin-mediated degradation

GLB1 LOF → TWO DISTINCT DISEASES (allele-dependent; phenotype-genotype correlation):
  (1) GM1 GANGLIOSIDOSIS: GLB1 LOF → GM1 ganglioside accumulates in lysosomes of CNS
      neurons → neurodegeneration (predominant storage = GM1 ganglioside + asialo-GM1 + lactosylceramide)
      Subtypes: Type 1 (Infantile; severe; null variants), Type 2 (Late-infantile/Juvenile;
      onset 7m-3y), Type 3 (Adult/Chronic; milder; predominant dystonia-parkinsonism)
  (2) MPS IVB (MORQUIO B): GLB1 LOF → keratan sulphate accumulates in non-CNS tissues
      (bones, cartilage, cornea); minimal CNS involvement; skeletal dysplasia dominant
      THIS DASHBOARD FOCUS: GM1 GANGLIOSIDOSIS TYPE 2 (Late-Infantile/Juvenile) +
      TYPE 3 (Adult/Chronic) — where epilepsy and seizure management are clinically central.

GM1 GANGLIOSIDOSIS — THREE PHENOTYPIC FORMS:
══════════════════════════════════════════════
(1) TYPE 1 (INFANTILE; SEVERE): Onset 0-6 months; hypotonia; rapid neurodegeneration;
    cherry-red macular spot (50-90%); coarse facies; hepatosplenomegaly; seizures by 6m-1y
    (infantile spasms → myoclonus → tonic); death by age 2-4y (respiratory failure).
    Null alleles (frameshift/nonsense/large deletion). Macular cherry-red spot + coarse facies
    + hepatosplenomegaly = classic infantile presentation.
(2) TYPE 2 (LATE-INFANTILE/JUVENILE; INTERMEDIATE): Onset 7 months to 3 years (mean 18m);
    psychomotor regression; progressive spasticity; ataxia; dystonia; seizures prominent (75%);
    myoclonus (55%); cherry-red spot ~25-40%; minimal hepatosplenomegaly; coarse facies mild;
    survival to childhood/adolescence (mean death ~10y). EPILEPSY IS A MAJOR MANAGEMENT CHALLENGE.
    Missense alleles with partial residual enzyme activity.
(3) TYPE 3 (ADULT/CHRONIC; MILD): Onset 3-30 years (mean 10y); predominantly extrapyramidal
    (dystonia + parkinsonism + cerebellar ataxia); seizures in ~25-35%; NO cherry-red spot;
    NO hepatosplenomegaly; NO coarse facies; normal cognitive baseline initially;
    survival to adulthood; predominantly Japanese (founder p.Ile51Thr in Japanese Type 3).
    Dystonia-predominant: bilateral putaminal lesions on MRI (T2 hyperintensity).
    OMIM: *611458 (GLB1 gene) / #230500 (GM1 Gangliosidosis) / #253010 (MPS IVB Morquio B)

GLB1 PROTEIN BIOLOGY:
═════════════════════
GLB1 (β-Galactosidase 1; 3p22.3):
  - 677 aa precursor (~88 kDa); signal peptide 1-23 (ER targeting); propeptide 24-69
    (removed in lysosome by CTSA); mature form ~64 kDa; homodimer (2 × 64 kDa active form)
  - GH family 35 (glycoside hydrolase 35); TIM barrel catalytic domain
  - Catalytic mechanism: Glu268 (acid/base) — Glu185 (nucleophile); two-step retaining mechanism
  - Specificity: hydrolyses β-1,4 galactosidic linkages (GM1 → GM2; Galβ1-3GalNAc-oligosaccharides)
  - Lysosomal localisation; requires CTSA (Protective Protein) for stability
  - CTSA LOF → GLB1 rapidly degraded → COMBINED NEU1+GLB1 deficiency = galactosialidosis
    GLB1 mutations → GLB1 only deficient (NEU1 preserved) = GM1 gangliosidosis or MPS IVB
  - pLI ~0.76 (moderate LOF intolerance)
  - Discovery: Okada & O'Brien (1968) Science — first GM1 gangliosidosis enzymatic diagnosis;
    β-galactosidase deficiency confirmed in patient leucocytes.

MULTIENZYME COMPLEX DIFFERENTIAL — CRITICAL DIAGNOSTIC RULE:
═════════════════════════════════════════════════════════════
Leukocyte enzyme pattern:
  NEU1 only deficient (GLB1 normal) → Sialidosis Type I/II (NEU1 mutation)
  GLB1 only deficient (NEU1 normal) → GM1 Gangliosidosis / MPS IVB (GLB1 mutation)
  BOTH NEU1 + GLB1 deficient      → Galactosialidosis (CTSA mutation — CTSA is SHARED protective protein)
WES panel MUST include GLB1 + NEU1 + CTSA simultaneously to avoid misclassification.

KEY PHARMACOLOGICAL DISTINCTIONS — GM1 GANGLIOSIDOSIS TYPE 2/3:
══════════════════════════════════════════════════════════════
(1) CBZ/OXC/PHT ABSOLUTE CI in Type 2 (myoclonic/infantile spasm overlap)
(2) VPA SAFE (lysosomal enzyme); POLG1/MERRF mandatory exclusion before VPA
(3) ACTH Level A for infantile spasms (West syndrome onset in Type 1/early Type 2)
(4) VGB HIGH RISK in Type 1/early Type 2 (cherry-red spot; potential macular + VGB retinopathy)
(5) Type 3 — dystonia phenotype: L-DOPA partial response; trihexyphenidyl for dystonia
(6) No approved disease-modifying therapy; gene therapy (AAV-GLB1) in clinical trials
(7) Leukocyte GLB1 enzyme assay (DBS possible for newborn screening); MRI bilateral putaminal T2
(8) Miglustat (substrate reduction) — off-label, limited evidence
"""


def get_overview():
    return {
        "gene": "GLB1 (3p22.3) — β-Galactosidase 1; 677 aa ~88 kDa precursor → mature ~64 kDa homodimer; GH family 35 glycoside hydrolase; Glu268-Glu185 catalytic dyad (retaining mechanism); hydrolyses β-1,4 galactosidic linkages (GM1→GM2 ganglioside key step); forms 1.3 MDa multienzyme complex with CTSA (protective protein, prevents intralysosomal GLB1 degradation) and NEU1. OMIM *611458/#230500 (GM1)/#253010 (MPS IVB Morquio B).",
        "protein": "β-Galactosidase 1 (GLB1); 677 aa; ~88 kDa precursor; signal peptide aa 1-23 (ER targeting); propeptide 24-69 (CTSA-mediated lysosomal processing); mature ~64 kDa (homodimer in lysosomes); TIM barrel catalytic domain; GH35 family; Glu268 general acid/base + Glu185 nucleophile (two-step retaining mechanism); specificity for β-1,4 galactosidic bonds; hydrolyses GM1 ganglioside → GM2 + galactose; also keratan sulphate + lactosylceramide; requires CTSA (protective protein) for lysosomal stability (CTSA LOF → GLB1 rapidly degraded by intralysosomal cathepsins); pLI ~0.76.",
        "inheritance": "Autosomal recessive (AR) biallelic GLB1 LOF. 25% sibling recurrence. Type 1 (infantile): null alleles (nonsense/frameshift/large deletion); Type 2 (late-infantile/juvenile): missense alleles with partial residual activity; Type 3 (adult/chronic): missense alleles with highest residual activity (~5-10% normal). Japanese founder: p.Ile51Thr (exon 2, c.152T>C) — >80% of Japanese Type 3 adult/chronic GM1 gangliosidosis. No AD form. Genotype-phenotype correlation: residual GLB1 activity determines phenotypic form.",
        "omim": "*611458 (GLB1 gene) · #230500 (GM1 Gangliosidosis — types 1/2/3) · #253010 (MPS IVB / Morquio B — skeletal phenotype of same GLB1 gene)",
        "disease": "GM1 Gangliosidosis — β-Galactosidase-1 Deficiency. GLB1 biallelic LOF → GLB1 enzyme deficient → GM1 ganglioside accumulates in lysosomes of CNS neurons → progressive neurodegeneration. Three forms: Type 1 (infantile; null; onset 0-6m; death 2-4y), Type 2 (late-infantile/juvenile; partial residual; onset 7m-3y; THIS COHORT FOCUS — epilepsy most prominent), Type 3 (adult/chronic; highest residual; onset 3-30y; dystonia-parkinsonism). DISTINGUISHING from galactosialidosis (CTSA): GLB1 ONLY deficient in GM1 gangliosidosis (NEU1 normal); galactosialidosis has BOTH NEU1+GLB1 deficient. No cherry-red spot in Type 3; present in 25-40% Type 2. SEIZURE MANAGEMENT IS DOMINANT CHALLENGE IN TYPE 2. No approved disease-modifying therapy; AAV-GLB1 gene therapy in Phase I/II trials (2024).",
        "mechanism": "GLB1 biallelic LOF → β-galactosidase enzyme absent/severely reduced → GM1 ganglioside cannot be hydrolysed (GM1 → GM2 + galactose step blocked) → GM1 ganglioside + asialo-GM1 + lactosylceramide accumulate in lysosomal compartment of neurons, astrocytes, retinal cells → lysosomal distension → cellular dysfunction → neuronal apoptosis → progressive neurodegeneration. Type 2 (juvenile): selective accumulation in neocortex, basal ganglia, thalamus, cerebellum → myoclonus + seizures + spasticity + ataxia + dystonia. Type 3 (adult): preferential accumulation in basal ganglia (putamen, globus pallidus, thalamus) → bilateral putaminal lesions on MRI → extrapyramidal syndrome (dystonia + parkinsonism). GLB1 is lysosomal enzyme (NOT mitochondrial) → VPA is SAFE. POLG1/MERRF mandatory exclusion before VPA (both mimic juvenile-onset neurodegeneration + seizures).",
        "glb1_deficiency_only_note": "GLB1 ONLY DEFICIENT IN GM1 GANGLIOSIDOSIS — CRITICAL MULTIENZYME COMPLEX DIFFERENTIAL: In GM1 gangliosidosis, GLB1 is intrinsically defective (GLB1 biallelic mutations); CTSA (protective protein) is intact; therefore NEU1 is unaffected. DIAGNOSTIC RULE: (1) Leukocyte β-galactosidase (GLB1) LOW + α-neuraminidase (NEU1) NORMAL → GM1 Gangliosidosis (GLB1 WES); (2) GLB1 LOW + NEU1 ALSO LOW → Galactosialidosis (CTSA WES — both low because CTSA protective protein absent); (3) NEU1 only low, GLB1 normal → Sialidosis (NEU1 WES). ALWAYS measure BOTH GLB1 and NEU1 simultaneously in leukocytes. Urine oligosaccharides and uronic acid: GM1 has excess sialyloligosaccharides (GM1-derived fragments) + urinary keratan sulphate. DBS β-galactosidase activity possible for neonatal screening (unlike NEU1 which requires fresh leukocytes).",
        "type2_type3_differential_note": "GM1 TYPE 2 vs TYPE 3 — EPILEPSY MANAGEMENT DIFFERS FUNDAMENTALLY: Type 2 (Late-Infantile/Juvenile): seizure-dominated onset; infantile spasms or myoclonic-atonic seizures by 18m; ACTH/VPA first-line; progressive neurodegeneration; death ~5-15y. Type 3 (Adult/Chronic): DYSTONIA-DOMINANT; extrapyramidal syndrome preceding seizures; bilateral putaminal T2 hyperintensity on MRI; seizures in ~25-35% (secondary, not primary); L-DOPA for dystonia-parkinsonism (unlike Type 2 where L-DOPA not useful); much slower progression; survival to adulthood; Japanese population enriched (p.Ile51Thr founder). KEY TREATMENT DISTINCTION: Type 2 → AED-first (VPA, LEV, ACTH for spasms); Type 3 → Dystonia-first (trihexyphenidyl + L-DOPA) then AED if seizures. Mistake: prescribing L-DOPA for a Type 2 patient thinking they have Type 3.",
        "mps_ivb_distinction_note": "GM1 GANGLIOSIDOSIS vs MPS IVB (MORQUIO B) — SAME GENE, DIFFERENT TISSUES: Both caused by GLB1 biallelic LOF, but phenotype depends on substrate accumulation pattern: GM1 GANGLIOSIDOSIS = predominantly CNS (neuronal ganglioside accumulation → neurodegeneration + seizures); MPS IVB (MORQUIO B) = predominantly skeletal (keratan sulphate accumulation in bones + cartilage + cornea → short stature, odontoid hypoplasia, corneal clouding — MINIMAL CNS). MECHANISM: missense alleles with different substrate selectivity affect CNS vs skeletal presentation; most alleles cause GM1 gangliosidosis; only specific alleles (particularly p.Trp273Leu) cause MPS IVB without CNS involvement. DIAGNOSTIC: urine keratan sulphate elevated in BOTH; CNS involvement on MRI distinguishes. Radiological: odontoid hypoplasia in MPS IVB → C1-C2 instability risk (MUST screen before anaesthesia/intubation).",
        "cohort_size": 40,
        "female_pct": 48,
        "mean_onset_years": 2.3,
        "mean_diagnosis_delay_years": 2.8,
        "drug_resistant_pct": 52,
        "cherry_red_spot_pct": 32,
        "seizure_pct": 75,
        "myoclonus_pct": 55,
        "infantile_spasms_pct": 38,
        "dystonia_pct": 62,
        "type3_adult_pct": 22,
        "on_vpa_pct": 78,
        "on_lev_pct": 60,
        "on_acth_pct": 35,
        "on_trihexyphenidyl_pct": 28,
        "putaminal_mri_pct": 45,
        "japanese_founder_pct": 18,
        "discovery": "Okada S & O'Brien JS (1968) Science 160:1002-4 — first enzymatic diagnosis of GM1 gangliosidosis; β-galactosidase deficiency confirmed in patient leucocytes, establishing biochemical basis of GM1 storage. Suzuki K (1968) — ganglioside nomenclature; GM1/GM2/GM3 classification. Norden AG et al. (1974) — GM1 gangliosidosis Type 3 (adult/chronic) first description. O'Brien JS (1975) — phenotype-genotype correlation; three-form classification. Yoshida K et al. (1991) — Japanese founder mutation p.Ile51Thr in Type 3 adult/chronic GM1. Brunetti-Pierri N & Scaglia F (2008) — comprehensive clinical review GM1 gangliosidosis all types.",
        "unique_feature": "GLB1 ONLY DEFICIENT — DISTINGUISHES GM1 GANGLIOSIDOSIS FROM GALACTOSIALIDOSIS (BOTH NEU1+GLB1) AND SIALIDOSIS (NEU1 ONLY). DUAL-DISEASE GENE: same GLB1 gene causes GM1 Gangliosidosis (CNS neurodegeneration) OR MPS IVB/Morquio B (skeletal dysplasia) depending on allele type and residual enzyme substrate selectivity. TYPE 3 BILATERAL PUTAMINAL MRI SIGNATURE — T2 hyperintensity in putamen + globus pallidus is PATHOGNOMONIC for adult GM1 gangliosidosis (distinguishes from Wilson disease, pantothenate kinase deficiency, other putaminal diseases). JAPANESE FOUNDER p.Ile51Thr in Type 3. ACTH LEVEL A for infantile spasms (West syndrome overlap in Type 1/early Type 2). AAV-GLB1 GENE THERAPY IN PHASE I/II TRIALS (2024) — unlike other lysosomal epilepsies where gene therapy is preclinical.",
        "key_pharmacological_distinctions": {
            "1_GLB1_ONLY_DEFICIENT_MULTIENZYME_COMPLEX_DIFFERENTIAL": "GLB1 ONLY DEFICIENT IN GM1 GANGLIOSIDOSIS (NEU1 NORMAL) — CRITICAL DIAGNOSTIC RULE vs GALACTOSIALIDOSIS AND SIALIDOSIS: Leukocyte pattern: (1) GLB1 only low (NEU1 normal) → GM1 Gangliosidosis (GLB1 WES); (2) BOTH GLB1+NEU1 low → Galactosialidosis (CTSA WES — CTSA is the shared protective protein for both NEU1 and GLB1); (3) NEU1 only low (GLB1 normal) → Sialidosis (NEU1 WES). DBS β-galactosidase activity is available for neonatal screening (unlike NEU1 which requires fresh leukocytes — 4-hour stability). Urine sialyloligosaccharides positive in GM1 (ganglioside-derived fragments). Urine keratan sulphate elevated in GM1 type 3 and MPS IVB. ALWAYS measure BOTH GLB1 and NEU1 simultaneously to avoid missing galactosialidosis (CTSA).",
            "2_ACTH_LEVEL_A_INFANTILE_SPASMS_TYPE1_EARLY_TYPE2": "ACTH LEVEL A FOR INFANTILE SPASMS IN GM1 GANGLIOSIDOSIS TYPE 1 AND EARLY TYPE 2 — UNIQUE TREATMENT NOT IN OTHER LYSOSOMAL EPILEPSIES: GM1 Type 1 (infantile) and early Type 2 (onset 7m-18m) frequently present with West syndrome phenotype (infantile spasms + hypsarrhythmia). ACTH (tetracosactide) is Level A evidence for infantile spasms across all aetiologies including GM1 gangliosidosis. Response rate ~50-70% for spasm cessation (lower than in unknown-aetiology West syndrome due to structural-metabolic substrate). VGB is alternative Level A for infantile spasms BUT HIGH RISK in GM1 Type 1 (cherry-red spot 50-90% in Type 1 → retinal risk from VGB retinopathy). Therefore ACTH preferred over VGB in GM1 gangliosidosis with cherry-red spot. Ketogenic diet Level B adjunct for refractory spasms. Vigabatrin: if cherry-red spot absent (Type 3, some Type 2) → VGB can be used; if cherry-red spot present (Type 1, early Type 2) → AVOID VGB.",
            "3_CBZ_OXC_PHT_ABSOLUTE_CI_MYOCLONIC_SEIZURES_TYPE2": "CBZ/OXC/PHT ABSOLUTE CI — GM1 TYPE 2 MYOCLONIC SEIZURES AND INFANTILE SPASM EVOLUTION TRAP: GM1 Type 2 evolves from infantile spasms → myoclonic-atonic seizures → generalised tonic-clonic as disease progresses. If CBZ/OXC/PHT prescribed (sodium channel blockers) for the GTCS component → ACUTE MYOCLONIC WORSENING (same mechanism as all progressive myoclonic encephalopathies). GM1 Type 2 with myoclonus is frequently misidentified as progressive encephalopathy without lysosomal aetiology → CBZ prescribed. SAFE backbone: VPA + LEV + piracetam (myoclonus) + clonazepam (nocturnal myoclonus).",
            "4_VPA_SAFE_LYSOSOMAL_POLG1_MERRF_MANDATORY": "VPA SAFE IN GM1 GANGLIOSIDOSIS — GLB1 IS LYSOSOMAL GLYCOSIDE HYDROLASE, NOT MITOCHONDRIAL; POLG1/MERRF EXCLUSION MANDATORY: GLB1 is a lysosomal glycoside hydrolase (GH35 family). VPA ABSOLUTE CI applies to MERRF (m.8344A>G mtDNA mutation) and POLG1 Alpers syndrome (mitochondrial DNA polymerase deficiency). GLB1 enzyme is lysosomal — VPA is SAFE as backbone AED once POLG1/MERRF excluded. MANDATORY PROTOCOL before VPA in any infant/child with progressive neurodegeneration + seizures: blood lactate + m.8344A>G PCR + POLG1 WES + muscle biopsy (if clinical suspicion). GM1 Type 2 milder cases can resemble POLG1 Alpers at presentation (progressive encephalopathy + seizures) — POLG1 is the most dangerous phenocopy (VPA → fatal hepatotoxicity).",
            "5_VGB_HIGH_RISK_CHERRY_RED_TYPE1_AVOID_USE_TYPE3_CAUTIOUS": "VGB HIGH RISK IN GM1 TYPE 1 (cherry-red 50-90%) — ALTERNATIVE ACTH FIRST-LINE FOR INFANTILE SPASMS; VGB USABLE IN TYPE 3 (NO CHERRY-RED): VGB is Level A for infantile spasms BUT causes irreversible peripheral visual field constriction (Müller cell toxicity). In GM1 Type 1 (cherry-red spot 50-90%) → additional retinal macular storage risk from VGB retinopathy → PREFER ACTH over VGB. In GM1 Type 2 (cherry-red 25-40%) → if cherry-red present → ACTH preferred; if cherry-red absent → VGB acceptable under ERG monitoring. In GM1 Type 3 (no cherry-red spot, no retinal involvement) → VGB NOT an absolute concern (similar to CLN12/CLN13 rule — no retinal degeneration). Document cherry-red spot status BEFORE choosing infantile spasm treatment.",
            "6_TYPE3_DYSTONIA_L_DOPA_TRIHEXYPHENIDYL_SEIZURE_TREATMENT_DIFFERS": "GM1 TYPE 3 DYSTONIA-PARKINSONISM — L-DOPA + TRIHEXYPHENIDYL FIRST-LINE; AED ONLY IF SEIZURES: GM1 Type 3 (adult/chronic) is predominantly extrapyramidal (dystonia in 85%, parkinsonism in 60%) with seizures in only 25-35%. Treatment priority is DYSTONIA MANAGEMENT: (1) Trihexyphenidyl (anticholinergic) for dystonia — initial 1 mg/day, titrate to 6-12 mg/day (Level C); (2) L-DOPA/carbidopa for parkinsonism component — partial response (dopaminergic neurons partially preserved in Type 3); (3) Baclofen for spasticity. AED only if seizures documented. CRITICAL ERROR: prescribing L-DOPA for a Type 2 patient mistaken for Type 3 — Type 2 has NEGLIGIBLE dopaminergic benefit and L-DOPA adds pharmacological complexity. MRI bilateral putaminal T2 hyperintensity + striatal atrophy = Type 3 radiological signature.",
            "7_AAV_GLB1_GENE_THERAPY_PHASE_I_II_2024_UNIQUE_GM1": "AAV-GLB1 GENE THERAPY IN PHASE I/II CLINICAL TRIALS (2024) — MOST ADVANCED LYSOSOMAL EPILEPSY GENE THERAPY IN THIS SERIES: Unlike NEU1, CTSA, and most NCL diseases where gene therapy is preclinical, AAV9-GLB1 (intrathecal/intracerebroventricular) is in active Phase I/II clinical trials for GM1 gangliosidosis (2024). Enrolment: infants/young children with GM1 Type 1 and Type 2 (pre-symptomatic or early symptomatic). Cornell-Weill/NCT04273269 and additional trials ongoing. IMPLICATIONS: (1) Newly diagnosed GM1 → urgent referral to gene therapy trial centre; (2) Pre-symptomatic newborn screening positive → immediate enrolment; (3) Gene therapy most effective before significant neuronal loss → early diagnosis critical. This distinguishes GM1 from other diseases in this lysosomal series where gene therapy is far from clinical application.",
            "8_MIGLUSTAT_SUBSTRATE_REDUCTION_OFF_LABEL_LIMITED_EVIDENCE": "MIGLUSTAT (SUBSTRATE REDUCTION THERAPY) — OFF-LABEL IN GM1; LIMITED EVIDENCE; DO NOT REPLACE AED BACKBONE: Miglustat (N-butyldeoxynojirimycin; iminosugar; glucosylceramide synthase inhibitor) reduces substrate upstream of the enzyme block in GM1 gangliosidosis. Approved for Gaucher Type 1 and Niemann-Pick C; used off-label in GM1 Type 3. Limited published case series (Inui et al. 2012; Japanese Type 3): some stabilisation of neurological progression. NOT approved for GM1 gangliosidosis. NOT first-line epilepsy treatment. Role: adjunct substrate reduction in specialist metabolic centre — does NOT replace AED seizure management. Eliglustat (more selective; approved Gaucher) has no published data in GM1.",
            "9_FOSPHENYTOIN_ABSOLUTE_CI_SE_PROTOCOL_IV_LEV_REPLACES": "FOSPHENYTOIN ABSOLUTE CI IN GM1 GANGLIOSIDOSIS SE PROTOCOL (TYPE 2 MYOCLONIC STATUS) — IV LEV REPLACES: Myoclonic status epilepticus is a recognised emergency in GM1 Type 2. Standard SE protocol second-line drug is fosphenytoin (IV phenytoin prodrug) — ABSOLUTE CI in GM1 gangliosidosis because IV PHT (Na-channel blocker) WORSENS myoclonic seizures. REPLACE with IV LEV 60 mg/kg (max 4500 mg) as second-line in all GM1 SE protocols. Pre-inform A&E and paediatric neurology: 'fosphenytoin ABSOLUTE CI in GM1 gangliosidosis — use IV LEV.' Provide family with written medical alert card. Rescue buccal midazolam 0.5 mg/kg for home use (cluster or prolonged seizures)."
        }
    }


def get_breakdown():
    return {
        "etiologies": [
            {
                "class": "Homozygous-Missense-Type2-Juvenile-Most-Common-Non-Japanese",
                "pct": 32,
                "description": "Homozygous GLB1 missense in non-Japanese consanguineous families; partial residual enzyme activity (1-5% normal); late-infantile/juvenile Type 2 phenotype; onset 7m-3y; myoclonic seizures + infantile spasms prominent; progressive psychomotor regression + spasticity + dystonia; minimal hepatosplenomegaly; cherry-red spot ~25-40%; death childhood/adolescence",
                "typical_onset": "7 months – 3 years (mean 18 months)",
                "genotype_notes": "Residual GLB1 activity 1-5% (above null threshold) → Type 2 not Type 1. Common European missense: p.Arg457Gln, p.Thr500Ala, p.Trp509Cys. Consanguineous families: Turkish, Arabic, Italian, Gypsy Roma. Phase confirmation needed; functional enzyme assay (leukocyte DBS or fibroblast) confirms."
            },
            {
                "class": "Compound-Het-Missense-Truncating-Type2-Type1-Overlap",
                "pct": 25,
                "description": "Compound heterozygous GLB1 missense + truncating (frameshift/nonsense); one null allele + one partial allele; intermediate phenotype (Type 2 or Type 1/2 overlap depending on missense allele residual activity); seizures onset 7-18m; heterogeneous severity; coarse facies moderate; cherry-red spot in ~35%; hepatosplenomegaly variable",
                "typical_onset": "7-24 months",
                "genotype_notes": "One allele null (truncating → complete LOF) + one missense (partial activity) → phenotype driven by missense allele residual activity. Higher residual → Type 2 (later onset, slower); near-null residual → Type 1/2 overlap (earlier, more severe). Western European non-consanguineous families predominantly."
            },
            {
                "class": "Homozygous-Missense-Type3-Adult-Chronic-Japanese-p.Ile51Thr",
                "pct": 18,
                "description": "Homozygous or compound het GLB1 missense with high residual activity (~5-10%); predominantly adult/chronic Type 3 phenotype; dystonia-parkinsonism dominant; seizures in ~30%; NO cherry-red spot; NO hepatosplenomegaly; bilateral putaminal MRI lesions; Japanese population enriched (founder p.Ile51Thr); survival to adulthood; slower progression",
                "typical_onset": "3-30 years (mean 10y; Type 3)",
                "genotype_notes": "Japanese founder: p.Ile51Thr (c.152T>C, exon 2) — >80% of Japanese Type 3 carry this allele. In Japanese patients with adult-onset dystonia + bilateral putaminal lesions: targeted p.Ile51Thr PCR first (days) before full WES. Non-Japanese Type 3: p.Asp448His, p.Tyr316Cys, p.Cys457Arg (European/non-Japanese)."
            },
            {
                "class": "Compound-Het-Missense-Missense-Type2-European-Attenuated",
                "pct": 15,
                "description": "Compound heterozygous GLB1 missense/missense; both alleles partial activity; attenuated Type 2 phenotype (milder, later onset within Type 2 range); myoclonus + GTCS without fulminant infantile spasm onset; clinical overlap with early Type 3 in some; mildest cognitive decline within Type 2; longest survival in juvenile category",
                "typical_onset": "18 months – 5 years (late-Type 2 spectrum)",
                "genotype_notes": "Both alleles retain partial GLB1 activity → aggregate residual ~2-8% of normal → attenuated Type 2. Clinical resembles Type 2 but with later onset, milder course. Distinguish from Type 3 (which has bilateral putaminal MRI vs Type 2 cortical-predominant MRI). Fibroblast enzyme assay confirms partial activity."
            },
            {
                "class": "Homozygous-Truncating-Null-Type1-Infantile-Severe",
                "pct": 7,
                "description": "Homozygous or biallelic null GLB1 variants (frameshift/nonsense/large deletion); complete enzyme absence; Type 1 infantile severe phenotype; onset 0-6m; hypotonia + developmental arrest; cherry-red spot 50-90%; coarse facies; hepatosplenomegaly; infantile spasms by 6-12m; death before 4y. Included in cohort for paediatric SE management context.",
                "typical_onset": "0-6 months (Type 1 infantile)",
                "genotype_notes": "Null alleles: p.Tyr83*, p.Arg482*, large exon deletions. Zero residual GLB1 activity in DBS/leukocyte. Type 1 phenotype. Represents most severe end of GM1 spectrum. Palliative pathway early; AED management focuses on comfort and seizure burden reduction."
            },
            {
                "class": "MPS-IVB-Morquio-B-Skeletal-Phenotype-Minimal-CNS",
                "pct": 3,
                "description": "GLB1 variants with specific allele selectivity (particularly p.Trp273Leu) causing predominantly keratan sulphate accumulation in skeletal tissues → MPS IVB (Morquio B); minimal neurological involvement; seizures rare (<10%); short stature + odontoid hypoplasia + corneal clouding dominant; included as phenotypic extreme of GLB1 gene.",
                "typical_onset": "1-5 years (skeletal presentation)",
                "genotype_notes": "Specific alleles alter substrate specificity toward keratan sulphate over ganglioside. Seizures not prominent in MPS IVB. Critical: odontoid hypoplasia → C1-C2 instability → pre-anaesthesia cervical spine MRI MANDATORY before any procedure including EEG lead placement under GA. Alert anaesthetist: CTSA/GLB1 Morquio B → odontoid risk."
            }
        ],
        "seizure_types": [
            {
                "type": "Infantile Spasms / West Syndrome (Type 1 + Early Type 2)",
                "prevalence_pct": 38,
                "eeg_pattern": "Hypsarrhythmia (modified or classic); chaotic high-amplitude mixed-frequency discharge; may be asymmetric (metabolic storage asymmetric deposition); modified hypsarrhythmia common in GM1 (high amplitude background activity + multifocal spikes without fully classic pattern); post-spasm EEG attenuation; electrodecremental response during spasms",
                "semiology": "Axial or generalised spasms (flexion/extension/mixed); clusters of spasms (5-50 per cluster); maximal on waking; head nods; cry post-spasm; truncal flexion; eye deviation; PRECEDE overt developmental regression in some GM1 Type 1/early Type 2 cases",
                "clinical_tips": "ACTH Level A first-line (prefer over VGB in GM1 — cherry-red spot risk with VGB). Tetracosactide (synthetic ACTH) 0.5-0.75 mg/m² IM/IV per protocol. If no cherry-red spot (confirmed fundoscopy) → VGB is acceptable alternative. KD Level B adjunct for ACTH-refractory spasms. DO NOT prescribe CBZ/OXC/PHT — worsens co-occurring myoclonus as disease evolves."
            },
            {
                "type": "Myoclonic-Atonic Seizures (Type 2 Dominant Seizure Type)",
                "prevalence_pct": 55,
                "eeg_pattern": "High-amplitude generalised polyspike-wave bursts (2-4 Hz); myoclonic component precedes atonic drop; enhanced cortical excitability on jerk-locked back-averaging; photosensitivity in ~30%; progressive background slowing correlating with neurodegeneration; giant SEPs on somatosensory stimulation in some Type 2 patients",
                "semiology": "Myoclonic jerk followed by atonic drop (head-nod, fall); action-sensitive component (voluntary movement provoked); morning predominance; atonic component causes FALLS (helmet essential); stimulus-sensitive jerks (light, sound, unexpected touch); severe myoclonus interferes with feeding, mobility, communication",
                "clinical_tips": "VPA (backbone) + LEV combination for myoclonic-atonic seizures. Piracetam adjunct for cortical action myoclonus (Level B PME myoclonus). Clonazepam for nocturnal myoclonus. Falls risk: helmet + padded environment mandatory. CBZ/OXC/PHT ABSOLUTE CI — worsens myoclonic-atonic component acutely."
            },
            {
                "type": "Generalised Tonic-Clonic Seizures (GTCS)",
                "prevalence_pct": 62,
                "eeg_pattern": "Generalised polyspike-wave (3-4 Hz); tonic recruiting rhythm evolving to clonic discharges; post-ictal generalised EEG suppression; background progressively slows in Type 2 (correlates with neurodegeneration); generalised slowing + multifocal spikes as disease advances",
                "semiology": "Generalised tonic-clonic evolution from myoclonic onset; often preceded by myoclonic jerk cascade; post-ictal confusion prolonged (correlates with degree of underlying neurodegeneration); nocturnal GTCS common in Type 2; high SUDEP risk with nocturnal unwitnessed GTCS",
                "clinical_tips": "VPA backbone (500-2000 mg/day) + LEV adjunct for GTCS control. NEVER CBZ/OXC/PHT in GM1 (ABSOLUTE CI — myoclonus worsening even when GTCS is primary concern). If GTCS misidentified as idiopathic generalised epilepsy in older Type 2/early Type 3 → CBZ prescribed → acute myoclonic worsening. POLG1/MERRF exclusion mandatory before VPA."
            },
            {
                "type": "Focal Seizures (Type 2 — Cortical Storage; Type 3 — Frontal-Occipital)",
                "prevalence_pct": 32,
                "eeg_pattern": "Focal temporal or occipital epileptiform discharges; may evolve to bilateral tonic-clonic; regional slowing correlating with MRI lesions; in Type 3 — perilesional focal discharges around putaminal storage areas (frontal-parietal); focal discharges may be asymmetric in early Type 2 (asymmetric storage deposition)",
                "semiology": "Focal impaired-awareness seizures (temporal); visual phenomena (occipital storage); focal motor seizures; complex automatisms; secondary bilateral tonic-clonic common",
                "clinical_tips": "LEV or LCM as focal adjunct onto VPA backbone. Do NOT use CBZ/OXC/PHT as focal AED in GM1 (myoclonus CI). In Type 3 with focal seizures around putaminal lesion — DWI/PWI MRI at seizure onset (exclude ischaemic component vs storage seizure)."
            },
            {
                "type": "Non-Convulsive Status Epilepticus (NCSE) — Myoclonic SE",
                "prevalence_pct": 20,
                "eeg_pattern": "Continuous or near-continuous polyspike-wave activity (1.5-3 Hz); myoclonic SE without overt clonic jerks; EEG essential for diagnosis (clinical confusion may resemble post-ictal or encephalopathic state); progressive buildup correlating with febrile illness, missed AEDs, or disease progression",
                "semiology": "Sustained myoclonic activity + reduced responsiveness without major convulsive movements; eyelid myoclonia; perioral jerks; impaired consciousness (background myoclonic SE); triggered by fever, AED taper, illness, or missed doses",
                "clinical_tips": "HIGH SUSPICION for myoclonic SE in GM1 Type 2 with acute deterioration. EEG urgently. IV LEV (preferred second-line; fosphenytoin ABSOLUTE CI) + IV midazolam. Avoid fosphenytoin (standard SE protocol second-line drug) — worsens myoclonic SE. TGB ABSOLUTE CI (provokes NCSE). Aggressive temperature management during febrile illness."
            }
        ],
        "triggers": [
            {
                "trigger": "Febrile Illness / Intercurrent Infection",
                "prevalence_pct": 82,
                "mechanism": "Fever increases cortical excitability; GM1 ganglioside-loaded neurons have reduced metabolic reserve (impaired lysosomal clearance) → lower seizure threshold; febrile illness commonly triggers infantile spasm flurries, myoclonic bursts, or GTCS clusters in GM1 Type 2",
                "management": "Aggressive fever management (paracetamol/ibuprofen); sick-day AED protocol (if vomiting → IV/buccal alternative); rescue midazolam home protocol training for carers; emergency letter for A&E (fosphenytoin ABSOLUTE CI — use IV LEV)"
            },
            {
                "trigger": "Sleep Deprivation / Sleep Disruption",
                "prevalence_pct": 72,
                "mechanism": "Reduced inhibitory tone during sleep-wake transitions; GM1 Type 2 children often have disrupted sleep (neurological discomfort, spasticity, feeding difficulties); sleep deprivation reduces infantile spasm threshold; nocturnal myoclonus disturbs sleep creating cycle",
                "management": "Structured sleep schedule; melatonin for sleep initiation (0.5-5 mg nocte); nocturnal monitoring for undetected GTCS (bed alarm/seizure mat); overnight carer for severe cases"
            },
            {
                "trigger": "Missed AED Doses",
                "prevalence_pct": 70,
                "mechanism": "VPA + LEV level drop below therapeutic; rebound excitability; VPA half-life 9-16h → missed doses more impactful than longer half-life drugs; myoclonic bursts or infantile spasm flurries on VPA withdrawal",
                "management": "Carer-administered blister pack; phone reminder system; NG/PEG tube administration if oral route compromised (GM1 Type 2 patients often have dysphagia → NG/PEG early); sick-day IV VPA protocol (inpatient if prolonged vomiting)"
            },
            {
                "trigger": "Photosensitivity (Light Flicker / Visual Stimuli)",
                "prevalence_pct": 30,
                "mechanism": "Enhanced cortical visual excitability in ~30% of GM1 Type 2; cortical storage in visual areas amplifies photosensitive response; photoparoxysmal response on EEG",
                "management": "Polarised lenses; screen brightness reduction; TV ≥2m distance; avoid strobe/disco lights; VPA + LEV reduce photosensitivity threshold"
            },
            {
                "trigger": "Voluntary Movement (Action Myoclonus — Type 2 Progressive)",
                "prevalence_pct": 55,
                "mechanism": "Action myoclonus in progressive GM1 Type 2 — voluntary movement triggers cortical re-entrant excitation → myoclonic jerks; similar mechanism to other progressive myoclonic encephalopathies",
                "management": "Piracetam (Level B for action myoclonus); VPA; OT/PT for adaptive strategies; helmet for fall prevention; UMRS (Unified Myoclonus Rating Scale) monitoring"
            },
            {
                "trigger": "Startle / Acoustic / Tactile Stimuli",
                "prevalence_pct": 40,
                "mechanism": "Stimulus-sensitive myoclonus (hyperekplexia-like component); unexpected sound, touch, or visual stimuli trigger myoclonic jerks; enhanced cortical-subcortical excitability",
                "management": "Predictable sensory environment; warn before examination; LEV reduces stimulus-sensitive myoclonus; structured routine at home and care setting"
            },
            {
                "trigger": "Excitement / Emotional Arousal",
                "prevalence_pct": 35,
                "mechanism": "Emotional arousal increases cortical excitability; startle myoclonus with emotional trigger; Type 2 children may exhibit 'excitability-triggered' spasm clusters",
                "management": "Calm structured environment; reduce unexpected emotional triggers; CLB (low dose nocte) for general excitability reduction"
            },
            {
                "trigger": "Contraindicated Drug Exposure (CBZ/OXC/PHT/TGB/Fosphenytoin)",
                "prevalence_pct": 100,
                "mechanism": "Na-channel blockers (CBZ/OXC/PHT/fosphenytoin) → paradoxical myoclonic worsening. TGB → provokes NCSE/myoclonic SE in GM1 Type 2. Risk from inadvertent prescribing by non-specialist (general paediatrician, GP, A&E prescribing fosphenytoin in SE)",
                "management": "Document ABSOLUTE CI in ALL records; medical alert card (wallet + bracelet); A&E liaison letter; GP letter; pharmacist alert; NEVER use standard SE protocol fosphenytoin — use IV LEV instead"
            }
        ],
        "treatments": [
            {
                "drug": "ACTH / Tetracosactide (for Infantile Spasms — Type 1/early Type 2)",
                "level": "Level A",
                "dose": "Tetracosactide (synthetic ACTH1-24): 0.5 mg/m² IM daily for 2 weeks → 0.5 mg/m² on alternate days for 2 weeks → taper over 4 weeks. Or natural ACTH 40-80 units IM daily. Duration 6 weeks total. UK ISS protocol or modified.",
                "moa": "ACTH agonism at melanocortin receptors (MC2R in adrenal + brain MCR); corticosteroid production; direct CNS effect on hypothalamic-pituitary axis; reduces hypsarrhythmia; may modulate synaptic transmission independent of steroid effect",
                "efficacy": "Spasm cessation 50-70% at 2 weeks (lower than non-metabolic West syndrome due to structural-metabolic substrate); hypsarrhythmia resolution; response predicts long-term neurodevelopmental outcome; less effective in structural-metabolic causes vs unknown-aetiology IS",
                "monitoring": "BP (hypertension — ACTH adrenal effect); electrolytes (hyponatraemia, hypokalaemia); blood glucose (hyperglycaemia); weight (rapid weight gain — corticosteroid effect); infection risk (immunosuppression); adrenal axis monitoring on taper; ophthalmology review (cataracts with prolonged steroid)",
                "glb1_note": "ACTH preferred over VGB as first-line infantile spasm treatment in GM1 gangliosidosis due to cherry-red spot risk with VGB (Type 1: 50-90%; early Type 2: 25-40%). If cherry-red spot ABSENT on fundoscopy → VGB is acceptable alternative. Urgent fundoscopy BEFORE choosing IS treatment. Refer urgently to gene therapy trial centre post-IS diagnosis (AAV-GLB1 trial eligibility screening)."
            },
            {
                "drug": "Valproate (VPA)",
                "level": "Level B",
                "dose": "20-40 mg/kg/day (paediatric); adults 500-2000 mg/day (target level 50-100 mg/L); IV sodium valproate 20 mg/kg loading for SE; modified release preferred; liquid formulation for NG/PEG administration (GM1 dysphagia common)",
                "moa": "GABA potentiation (inhibits GABA-transaminase → GABA increase); Na-channel modulation (voltage-independent — does NOT worsen PME unlike CBZ); T-type calcium channel inhibition; broad-spectrum anti-epileptic across generalised and focal seizure types",
                "efficacy": "GTCS reduction 60-70%; myoclonic-atonic reduction 50-60%; infantile spasm adjunct (add to ACTH for refractory spasms). Best combined with LEV + piracetam for myoclonic-atonic dominant Type 2",
                "monitoring": "TDM (target 50-100 mg/L); LFTs (hepatotoxicity — POLG1 ABSOLUTELY EXCLUDED before starting — Type 2 mimics POLG1 Alpers); FBC (thrombocytopenia); weight; teratogenicity (VPA most teratogenic common AED — contraception mandatory in females of reproductive potential); pancreatitis (rare); carbapenems reduce VPA levels",
                "glb1_note": "VPA SAFE in GM1 gangliosidosis — GLB1 is lysosomal glycoside hydrolase, NOT mitochondrial. HOWEVER: POLG1 Alpers is the most dangerous phenocopy of GM1 Type 2 juvenile encephalopathy + seizures → VPA in POLG1 = fatal hepatotoxicity. MANDATORY: blood lactate + POLG1 WES BEFORE VPA in ANY progressive encephalopathy + seizures. Liquid formulation essential for NG/PEG-dependent GM1 Type 2 patients (dysphagia at disease progression)."
            },
            {
                "drug": "Levetiracetam (LEV)",
                "level": "Level B",
                "dose": "Paediatric: 10-60 mg/kg/day (max 3000 mg/day); adults 500-3000 mg/day; IV LEV 60 mg/kg for SE (REPLACES FOSPHENYTOIN which is ABSOLUTE CI); available as liquid for NG/PEG administration",
                "moa": "SV2A (synaptic vesicle glycoprotein 2A) binding → reduces neurotransmitter vesicle release probability; reduces N-type calcium current; modulates GABA-A receptor function; antimyoclonic action across multiple PME/myoclonic encephalopathy aetiologies",
                "efficacy": "Myoclonic seizure reduction 40-55%; GTCS reduction 45-60% combined with VPA; IV LEV preferred in GM1 SE (replaces fosphenytoin CI); well-tolerated combination with VPA",
                "monitoring": "Behavioural side-effects (irritability, aggression — amplified in GM1 Type 2 children with limited verbal communication; monitor via carer observations); no TDM required; renal dosing (eGFR <80)",
                "glb1_note": "IV LEV is the preferred second-line SE drug in GM1 (replaces fosphenytoin which is ABSOLUTE CI). Pre-inform all A&E, ward staff, and GP: 'GM1 gangliosidosis — FOSPHENYTOIN ABSOLUTE CI — use IV LEV in SE.' Liquid formulation for NG/PEG. Behavioural monitoring especially important in GM1 Type 2 non-verbal children who cannot report LEV-related dysphoria verbally."
            },
            {
                "drug": "Piracetam (for Action Myoclonus — Type 2 Progressive)",
                "level": "Level B",
                "dose": "Age 5+: 8-16 g/day in divided doses (start 4 g/day, increase by 2 g/week to effect); weight-based paediatric dosing available; liquid suspension for NG/PEG in dysphasic patients",
                "moa": "Racetam class; AMPA glutamate receptor modulation + neuronal membrane fluidity improvement; Level B evidence for cortical PME action myoclonus across multiple aetiologies (CSTB/Unverricht-Lundborg, NEU1, CTSA, GM1 Type 2 late-stage cortical myoclonus)",
                "efficacy": "Action myoclonus reduced 35-50% in Type 2 progressive phase; enables improved feeding, communication, ADL function; piracetam + VPA + LEV combination best for myoclonic-atonic dominant Type 2",
                "monitoring": "Generally well tolerated; mild CNS effects (restlessness, insomnia at high doses); renal dosing; note: in GM1 Type 2 with progressive cognitive decline, baseline neuropsychological state documented before piracetam to attribute changes to disease vs drug",
                "glb1_note": "Piracetam most appropriate in Type 2 late stage when cortical action myoclonus dominates (similar to other lysosomal PME diseases). Less relevant in Type 1 (short survival, spasm-dominant) or Type 3 (action myoclonus uncommon; dystonia dominant). NG/PEG liquid available for dysphasic Type 2 patients."
            },
            {
                "drug": "Clobazam (CLB)",
                "level": "Level B",
                "dose": "Paediatric: 0.1-1 mg/kg/day in 1-2 divided doses; adults 10-40 mg/day; nocturnal loading dose for nocturnal myoclonus and GTCS control",
                "moa": "1,5-benzodiazepine; positive allosteric modulator of GABA-A; less sedating than clonazepam (1,4-BDZ); efficacy for myoclonic-atonic seizures in refractory epilepsy syndromes",
                "efficacy": "Myoclonic-atonic seizure reduction 50-60% adjunct; nocturnal GTCS control; BDZ tolerance may develop in some patients over months (monitor for loss of efficacy)",
                "monitoring": "Sedation; behavioural dis-inhibition (paradoxical agitation in children with cognitive impairment); tolerance monitoring; avoid abrupt withdrawal; interaction with CYP2C19 inhibitors (increases CLB active metabolite N-CLB)",
                "glb1_note": "Nocturnal dosing strategy preferred in GM1 Type 2 to target nocturnal myoclonus + GTCS without excessive daytime sedation (which impairs already limited neurological function). Clonazepam (stronger 1,4-BDZ) alternative for more severe nocturnal myoclonus but higher sedation burden."
            },
            {
                "drug": "Trihexyphenidyl + L-DOPA/Carbidopa (Type 3 Dystonia — NOT for Type 2)",
                "level": "Level C",
                "dose": "Trihexyphenidyl: start 1 mg/day, titrate to 6-12 mg/day over 4-6 weeks. L-DOPA/carbidopa: start 50/12.5 mg TDS, titrate to 200/50 mg TDS over weeks; L-DOPA partial response expected (not dramatic); trial 3 months before assessing response",
                "moa": "Trihexyphenidyl: muscarinic ACh antagonist → reduces striatal cholinergic excess in dystonia (dopamine-acetylcholine imbalance). L-DOPA: dopamine precursor → partially restores dopaminergic nigrostriatal function (partially preserved in Type 3 vs Type 2)",
                "efficacy": "Type 3 dystonia: trihexyphenidyl 40-60% improvement in functional dystonia score; L-DOPA partial (~50% of Parkinson's disease response due to mixed neurodegeneration); NOT effective for Type 1/2 phenotype",
                "monitoring": "Trihexyphenidyl: anticholinergic side-effects (dry mouth, urinary retention, constipation, blurred vision, cognitive confusion in elderly); L-DOPA: dyskinesias (lower risk than PD due to partial nigrostriatal degeneration pattern); psychiatric effects",
                "glb1_note": "CRITICAL: Trihexyphenidyl + L-DOPA are for GM1 TYPE 3 DYSTONIA-PARKINSONISM ONLY. DO NOT use in Type 1 or Type 2 (no dopaminergic component in infantile/juvenile forms; adds pharmacological burden without benefit). Type 3 with seizures: AED (VPA + LEV) ADDED to the dystonia regimen — not replaced by it. Type 2 occasionally misclassified as Type 3 → prescribe L-DOPA → no benefit + side-effects."
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "Level B",
                "dose": "4:1 or 3:1 fat:carbohydrate+protein ratio; dietitian-led initiation; consider modified Atkins for older Type 2 children (>4y); NG/PEG liquid KD formula for dysphasic patients",
                "moa": "Ketosis → metabolic shift → GABA enhancement + glutamate reduction; reduces cortical excitability; evidence across infantile spasms, myoclonic-atonic, and GTCS in metabolic-structural epilepsies",
                "efficacy": "Infantile spasms refractory to ACTH: Level B adjunct (~30-50% additional spasm reduction); myoclonic-atonic: 40-50% reduction in half of patients; most effective in early disease before severe neurodegeneration",
                "monitoring": "Lipid profile; growth parameters (critical in GM1 Type 2 — already nutritionally compromised); renal stones; acidosis; NG/PEG formula management; seizure diary for KD response assessment",
                "glb1_note": "KD particularly valuable in GM1 Type 2 refractory to ACTH+VPA+LEV. NG/PEG ketogenic formula essential for Type 2 children with dysphagia (swallowing significantly impaired). Nutritional status in GM1 Type 2 often poor (increased metabolic demand + feeding difficulty) — dietitian input from diagnosis essential."
            },
            {
                "drug": "Rescue Midazolam (Buccal/Nasal) + IV LEV (SE Protocol)",
                "level": "Level A",
                "dose": "Rescue: buccal midazolam 0.5 mg/kg (max 10 mg) after 5 min sustained seizure; nasal midazolam 0.2 mg/kg alternative. SE protocol: IV midazolam 0.2 mg/kg → IV LEV 60 mg/kg (max 4500 mg) — NEVER fosphenytoin (ABSOLUTE CI)",
                "moa": "Midazolam: rapid GABA enhancement → aborts seizure. IV LEV: SV2A → reduces synaptic vesicle release. Replaces fosphenytoin (standard protocol) which is ABSOLUTE CI in GM1 gangliosidosis myoclonic SE.",
                "efficacy": "Rescue midazolam terminates 75-85% of prolonged GM1 seizures at home; IV LEV aborts myoclonic SE in 65-75% without myoclonus-worsening risk of fosphenytoin",
                "monitoring": "Emergency action plan laminated + displayed; carer training annual renewal; A&E alert card; out-of-date medications replaced",
                "glb1_note": "CRITICAL: provide WRITTEN CARD: 'I have GM1 Gangliosidosis (GLB1). DO NOT give phenytoin, fosphenytoin, carbamazepine, oxcarbazepine, tiagabine. For status epilepticus: IV LEVETIRACETAM + IV MIDAZOLAM.' Medical alert bracelet. Ensure A&E fosphenytoin ABSOLUTE CI documented in hospital system — fosphenytoin is reflexly prescribed in paediatric SE protocols."
            }
        ],
        "contraindications": [
            {
                "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
                "severity": "ABSOLUTE CI",
                "reason": "Na-channel blockers worsen cortical PME myoclonus and myoclonic-atonic seizures in GM1 Type 2. Mechanism: blockade of high-frequency Na channels in inhibitory interneurons → paradoxical myoclonic worsening. GM1 Type 2 progression from infantile spasms → myoclonic-atonic → GTCS creates repeated risk of CBZ prescription as seizure type evolves. Mean diagnosis delay 2.8y = extended CBZ exposure window.",
                "note": "ABSOLUTE CI in GM1 gangliosidosis Type 2 (myoclonic component). In Type 3 with purely focal seizures and NO myoclonus — debate exists but avoid if possible (LCM preferred). Document in ALL records."
            },
            {
                "drug": "Fosphenytoin (IV Phenytoin Prodrug)",
                "severity": "ABSOLUTE CI",
                "reason": "Fosphenytoin is standard second-line SE protocol drug (IV PHT prodrug). ABSOLUTE CI in GM1 gangliosidosis because IV PHT worsens myoclonic SE. A&E/paediatric emergency teams unfamiliar with GM1 will instinctively use fosphenytoin per SE protocol.",
                "note": "REPLACE FOSPHENYTOIN WITH IV LEVETIRACETAM in all GM1 SE protocols. Document in hospital records, SE protocol exceptions, A&E alert. Provide family with emergency card and medical alert jewellery. 'GM1 gangliosidosis — FOSPHENYTOIN ABSOLUTE CI — USE IV LEV 60 mg/kg.'"
            },
            {
                "drug": "Tiagabine (TGB)",
                "severity": "ABSOLUTE CI",
                "reason": "TGB (GABA reuptake inhibitor) provokes non-convulsive status epilepticus (myoclonic SE) in generalised/myoclonic epilepsy syndromes including GM1 Type 2. Mechanism: excess GABA accumulation at synaptic terminals → paradoxical NCSE.",
                "note": "ABSOLUTE CI in GM1 gangliosidosis. No role in any myoclonic encephalopathy syndrome."
            },
            {
                "drug": "Vigabatrin (VGB) — HIGH RISK / Context-Dependent",
                "severity": "HIGH RISK (Type 1 + cherry-red Type 2) / CAUTIOUS USE (Type 3 + cherry-red negative Type 2)",
                "reason": "VGB causes irreversible peripheral visual field constriction (Müller cell retinal toxicity). Cherry-red macular spot in GM1 Type 1 (50-90%) indicates retinal ganglion cell storage — VGB retinopathy adds to macular vulnerability. VGB acceptable as infantile spasm treatment only if cherry-red spot confirmed ABSENT. Type 3: no cherry-red spot, no retinal NCL → VGB not categorically prohibited.",
                "note": "ACTH preferred over VGB for infantile spasms in GM1 if cherry-red spot present. If VGB used (Type 3, or cherry-red negative Type 2): ERG baseline + 3-monthly peripheral visual field monitoring mandatory. Document fundoscopy result and VGB decision basis."
            },
            {
                "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
                "severity": "HIGH RISK",
                "reason": "Alpha-2-delta calcium channel modulators — may worsen myoclonic seizures; no evidence of benefit in GM1 myoclonic epilepsy; sedation may mask seizure/neurodegeneration assessment; no PME/myoclonic encephalopathy efficacy data.",
                "note": "Avoid. Safe alternatives for neuropathic pain (if comorbid): low-dose amitriptyline."
            },
            {
                "drug": "Lamotrigine Monotherapy",
                "severity": "HIGH RISK",
                "reason": "LTG monotherapy at GTCS-control doses may produce Na-channel-like myoclonic worsening. LTG only acceptable as low-dose ADJUNCT on VPA backbone. As monotherapy in GM1 Type 2 — HIGH RISK of myoclonic worsening.",
                "note": "If used: only as adjunct (25-100 mg/day on VPA backbone). Slow titration (VPA doubles LTG levels). Never as monotherapy in GM1 Type 2."
            },
            {
                "drug": "L-DOPA in Type 1 or Type 2 GM1 Gangliosidosis",
                "severity": "HIGH RISK (inappropriate use)",
                "reason": "L-DOPA has no therapeutic role in GM1 Type 1 or Type 2 (no significant dopaminergic nigrostriatal degeneration in these forms). Misclassification of Type 2 as Type 3 → L-DOPA prescribed → no benefit + potential psychiatric side-effects + delays appropriate AED optimisation.",
                "note": "L-DOPA ONLY for confirmed GM1 TYPE 3 with extrapyramidal features on MRI (bilateral putaminal T2 change + clinical dystonia/parkinsonism). Never prescribe L-DOPA for GM1 Type 1/2 seizure management — no evidence base."
            }
        ],
        "monitoring": [
            {"item": "GLB1 + NEU1 leukocyte enzyme dual assay", "frequency": "At diagnosis", "note": "GLB1 only low → GM1 (not galactosialidosis); BOTH GLB1+NEU1 low → galactosialidosis (CTSA WES). DBS β-galactosidase acceptable for neonatal screening."},
            {"item": "GLB1 WES (+ NEU1 + CTSA + POLG1 on same panel)", "frequency": "At diagnosis", "note": "POLG1 exclusion mandatory before VPA; include NEU1+CTSA to exclude galactosialidosis phenocopy."},
            {"item": "Urine oligosaccharides + urinary keratan sulphate", "frequency": "At diagnosis", "note": "GM1 oligosaccharides positive; keratan sulphate elevated (GM1 + MPS IVB); screen for phenotypic subtype."},
            {"item": "Blood lactate + m.8344A>G (MERRF) exclusion", "frequency": "Before VPA", "note": "MANDATORY before VPA — POLG1 Alpers and MERRF are fatal phenocopies with VPA ABSOLUTE CI."},
            {"item": "Fundoscopy (cherry-red macular spot assessment)", "frequency": "At diagnosis; annual", "note": "CRITICAL before choosing IS treatment (ACTH vs VGB). Cherry-red: ACTH first-line (not VGB). Absent: VGB acceptable."},
            {"item": "Ophthalmology + ERG + VEP", "frequency": "Annual (6-monthly if VGB used)", "note": "Cherry-red macular monitoring; ERG if VGB used under any circumstance."},
            {"item": "Brain MRI 3T (T1, T2, FLAIR, DWI)", "frequency": "At diagnosis; 12-monthly (Type 2); 2-yearly (Type 3)", "note": "Type 2: cerebral + thalamic + cerebellar atrophy progression; Type 3: bilateral putaminal T2 hyperintensity (PATHOGNOMONIC); basal ganglia/thalamic signal changes."},
            {"item": "EEG (baseline; hypsarrhythmia screen; routine + video)", "frequency": "At diagnosis + every 6 months; urgent if change", "note": "Infantile spasms: VPSG. Type 2 progressive: annual EEG for evolving seizure type. Urgent if acute deterioration/NCSE suspected."},
            {"item": "VPA TDM + LFTs + FBC (if on VPA)", "frequency": "4 weeks post-start; 12 weeks; then 6-monthly", "note": "VPA hepatotoxicity monitoring (POLG1 excluded, but ongoing LFT standard); target 50-100 mg/L."},
            {"item": "GMFCS / motor function (Gross Motor Function Classification System)", "frequency": "Every 6 months", "note": "Motor deterioration tracking in Type 2; gait aid progression; wheelchair assessment timing."},
            {"item": "Nutritional status / dysphagia screen", "frequency": "Every 3 months", "note": "GM1 Type 2: progressive dysphagia early; PEG/NG timing decision; dietitian input for KD if refractory."},
            {"item": "Developmental assessment + neuropsychology", "frequency": "Annual", "note": "Cognitive trajectory tracking; Type 2 progressive regression (distinguish disease from medication effect); Type 3 slower cognitive decline."},
            {"item": "Cervical spine MRI (odontoid hypoplasia screen)", "frequency": "If MPS IVB features / before any GA anaesthesia", "note": "MPS IVB/Morquio B: odontoid hypoplasia → C1-C2 instability risk under GA. Cervical spine MRI BEFORE intubation if any GLB1 patient has skeletal phenotype features."},
            {"item": "SUDEP risk + nocturnal seizure assessment", "frequency": "Annual", "note": "GM1 Type 2 drug-resistant (52%) + nocturnal GTCS → elevated SUDEP risk. Nocturnal monitoring device (SAMi or equivalent); carer training; SUDEP Action resources."},
            {"item": "Gene therapy trial eligibility screening", "frequency": "At diagnosis (urgent referral)", "note": "AAV-GLB1 Phase I/II trials recruiting (2024). Newly diagnosed Type 1 and Type 2 → urgent referral to trial centre (Cornell-Weill NCT04273269 or equivalent). Pre-symptomatic = highest likelihood of benefit."}
        ],
        "lifecycle_stages": [
            {
                "stage": "Prenatal / Pre-Symptomatic (Newborn Screen / Genetic Risk)",
                "age_range": "Preconception – 6 months (Type 1) / 7 months (Type 2)",
                "description": "Newborn screening (DBS β-galactosidase expanding globally). Pre-symptomatic diagnosis enables gene therapy trial eligibility (highest benefit before neuronal loss). Sibling cascade testing (25% AR risk). Antenatal diagnosis if biallelic GLB1 variants known in family.",
                "priorities": ["DBS β-galactosidase newborn screening (if available)", "GLB1 WES sibling testing", "Gene therapy trial referral (NCT04273269) — pre-symptomatic", "Genetic counselling (25% sibling risk; prenatal diagnosis options)", "Fundoscopy at birth if Type 1 suspected"]
            },
            {
                "stage": "Infantile Spasm Phase (Type 1 / Early Type 2)",
                "age_range": "6-24 months",
                "description": "West syndrome presentation: infantile spasms + hypsarrhythmia. Critical diagnostic phase — GLB1 diagnosis often delayed (mean 2.8y). Fundoscopy + leukocyte GLB1 assay + MRI urgently. ACTH first-line (AVOID VGB if cherry-red present). Gene therapy trial referral.",
                "priorities": ["Fundoscopy — cherry-red determines ACTH vs VGB", "Urgent leukocyte GLB1+NEU1 dual enzyme panel", "ACTH (preferred) or VGB (if no cherry-red) for IS", "POLG1/MERRF exclusion before VPA", "Gene therapy trial eligibility assessment", "MRI baseline (thalamic involvement, basal ganglia changes)"]
            },
            {
                "stage": "Active Epilepsy Phase (Type 2 Dominant — Myoclonic-Atonic + GTCS)",
                "age_range": "18 months – 8 years",
                "description": "Myoclonic-atonic seizures + GTCS dominate after infantile spasm evolution. VPA + LEV backbone. Piracetam for progressive action myoclonus. KD if refractory. Nutritional support (dysphagia emerging). AED optimisation. Risk of CBZ prescription if seizure type misidentified.",
                "priorities": ["VPA+LEV+piracetam combination", "CBZ/OXC/PHT ABSOLUTE CI documentation", "Fosphenytoin ABSOLUTE CI — SE protocol updated with IV LEV", "Helmet + fall prevention (atonic drops)", "NG/PEG nutritional support (dysphagia management)", "KD initiation if drug-resistant (52%)"]
            },
            {
                "stage": "Type 3 Adult Onset — Dystonia-First / Epilepsy-Second",
                "age_range": "3-30 years",
                "description": "Predominantly extrapyramidal: dystonia + parkinsonism + cerebellar ataxia. Seizures in 25-35% (secondary). Bilateral putaminal MRI signature. Japanese founder p.Ile51Thr enriched. Trihexyphenidyl + L-DOPA for dystonia. AED only if seizures present. Very slow progression; independence maintained longer than Type 2.",
                "priorities": ["Trihexyphenidyl first-line for dystonia", "L-DOPA for parkinsonism (partial response)", "AED (VPA/LEV) if seizures documented", "Bilateral putaminal MRI confirmation (Type 3 signature)", "Japanese patient: p.Ile51Thr PCR first", "Driving assessment (extrapyramidal + possible seizures)"]
            },
            {
                "stage": "Advanced Neurodegeneration (Type 2 — Progressive Loss)",
                "age_range": "5-15 years (Type 2)",
                "description": "Progressive spasticity + loss of ambulation + dysphagia + severe cognitive impairment + drug-resistant seizures. Palliative care goals increasingly central. Continued AED regimen for seizure burden reduction. PEG/gastrostomy for nutrition. Nocturnal seizure surveillance. SUDEP risk counselling.",
                "priorities": ["PEG/gastrostomy placement (dysphagia severe)", "Baclofen for spasticity (intrathecal if severe)", "Continue AED (VPA+LEV) — seizure burden reduction in palliative context", "Nocturnal monitoring (SUDEP risk)", "Palliative care team integration", "Family respite + psychological support"]
            },
            {
                "stage": "End of Life / Bereavement / Research",
                "age_range": "Post-death (Type 2 typically childhood/adolescence; Type 3 adulthood)",
                "description": "Research autopsy for GLB1 brain tissue (advances gene therapy and pathomechanism research). Sibling/family cascade testing. Bereavement support. Link to gene therapy trial centres for future siblings (pre-symptomatic screening). International GM1 patient registries (GLIA-B registry, Lysosomal Disease Network).",
                "priorities": ["Research autopsy consent (brain tissue for gene therapy research)", "Sibling GLB1 WES + DBS newborn screening", "Family genetic counselling (25% AR recurrence)", "Gene therapy trial linkage for future affected siblings (pre-symptomatic eligibility)", "GLIA-B Registry / Lysosomal Disease Network registration", "Bereavement psychological support"]
            }
        ]
    }


def get_definitions():
    return {
        "disease_name": "GM1 Gangliosidosis — β-Galactosidase-1 Deficiency (Types 1, 2, 3) / MPS IVB (Morquio B — skeletal variant of same gene)",
        "gene_full": "GLB1 (β-Galactosidase 1) — 3p22.3; 677 aa precursor ~88 kDa; GH family 35 glycoside hydrolase; Glu268-Glu185 catalytic dyad; lysosomal homodimer ~64 kDa mature form; hydrolyses GM1 ganglioside β-1,4 galactosidic bond; forms multienzyme complex with CTSA + NEU1; requires CTSA (protective protein) for stability",
        "omim_gene": "OMIM *611458 (GLB1 gene)",
        "omim_disease": "OMIM #230500 (GM1 Gangliosidosis types 1/2/3) · #253010 (MPS IVB / Morquio B — skeletal GLB1 phenotype)",
        "protein_full": "β-Galactosidase 1 (GLB1); 677 aa; ~88 kDa precursor; signal peptide aa 1-23; propeptide 24-69 (CTSA-processed in lysosome); mature ~64 kDa homodimer; GH35 family TIM barrel catalytic domain; Glu268 (acid-base general catalyst) + Glu185 (nucleophile); two-step retaining mechanism; hydrolyses β-1,4 galactosidic linkages (GM1→GM2 ganglioside; galactose-terminated glycoproteins; keratan sulphate); lysosomal localisation; requires CTSA protective protein for stability; pLI ~0.76",
        "inheritance_mode": "Autosomal recessive (AR) biallelic GLB1 LOF; 25% sibling recurrence; Type 1: null alleles (no residual activity); Type 2: missense (1-5% residual); Type 3: missense (5-10% residual); Japanese founder p.Ile51Thr (c.152T>C exon 2) in >80% of Japanese Type 3",
        "onset_age": "Type 1 (Infantile): 0-6 months; Type 2 (Late-infantile/Juvenile): 7 months – 3 years (mean 2.3y cohort); Type 3 (Adult/Chronic): 3-30 years",
        "multienzyme_complex_role": "GLB1 IN NEU1-CTSA-GLB1 MULTIENZYME COMPLEX: GLB1 (β-galactosidase) is part of the 1.3 MDa lysosomal multienzyme complex with CTSA (protective protein) and NEU1 (α-neuraminidase). CTSA is required to PROTECT GLB1 from premature intralysosomal cathepsin degradation — CTSA LOF → GLB1 rapidly degraded → combined NEU1+GLB1 deficiency (galactosialidosis). In GM1 gangliosidosis: GLB1 is intrinsically defective (GLB1 biallelic mutations); CTSA is intact; NEU1 is unaffected. DIAGNOSTIC RULE: (1) GLB1 only low (NEU1 normal) → GM1 Gangliosidosis/MPS IVB (GLB1 WES); (2) BOTH GLB1+NEU1 low → Galactosialidosis (CTSA WES); (3) NEU1 only low (GLB1 normal) → Sialidosis (NEU1 WES). Always measure BOTH GLB1 and NEU1 simultaneously in leukocytes.",
        "cherry_red_differentials": [
            {"disease": "GM1 Gangliosidosis Type 1/2 (GLB1) — this disease", "distinguishing": "GLB1 only deficient (NEU1 normal); infantile/juvenile onset; coarse facies (Type 1/2); cherry-red in Type 1 (50-90%), Type 2 (25-40%), Type 3 (absent); AR GLB1 biallelic"},
            {"disease": "Galactosialidosis (CTSA/PPCA)", "distinguishing": "BOTH NEU1+GLB1 deficient (CTSA protective protein absent); coarse facies + angiokeratoma + mild ID; Japanese founder p.Ser23Leu+p.Gly411Ser; AR CTSA biallelic"},
            {"disease": "Sialidosis Type I (NEU1)", "distinguishing": "NEU1 only deficient (GLB1 normal); NORMAL cognition; NO coarse facies; adolescent PME onset; cherry-red spot 90-95%"},
            {"disease": "GM2 Gangliosidosis / Tay-Sachs (HEXA) or Sandhoff (HEXB)", "distinguishing": "Hex A or Hex B deficient; infantile onset; rapid neurodegeneration; no β-galactosidase deficiency; cherry-red 90%+ in infantile form"},
            {"disease": "MERRF (m.8344A>G mtDNA)", "distinguishing": "Mitochondrial; VPA ABSOLUTE CI; rare cherry-red; maternal inheritance; deafness; ragged-red fibres; cherry-red uncommon"},
            {"disease": "Niemann-Pick Type C (NPC1/NPC2)", "distinguishing": "Cholesterol trafficking; vertical supranuclear gaze palsy (PATHOGNOMONIC NPC); cataplexy; no β-galactosidase deficiency; dementia-dominant"},
            {"disease": "Normal variant (pseudo-cherry-red spot)", "distinguishing": "No lysosomal enzyme deficiency; ERG normal; re-examine by experienced ophthalmologist"}
        ],
        "concepts": [
            {"name": "GLB1-3p22.3-Lysosomal-GH35-Beta-Galactosidase-GM1-Ganglioside-Hydrolysis", "definition": "GLB1 (β-Galactosidase 1) is a lysosomal GH family 35 glycoside hydrolase at 3p22.3 that cleaves terminal β-galactose from GM1 ganglioside (GM1→GM2 + galactose). GLB1 LOF → GM1 ganglioside accumulates in CNS lysosomes → progressive neurodegeneration (Types 1-3) or keratan sulphate in skeleton (MPS IVB/Morquio B)."},
            {"name": "GLB1-Only-Deficient-NEU1-Normal-Critical-Galactosialidosis-Differential", "definition": "In GM1 Gangliosidosis, GLB1 is intrinsically defective; CTSA (protective protein) is intact; NEU1 is NORMAL. This distinguishes from galactosialidosis (CTSA LOF → BOTH NEU1+GLB1 deficient). ALWAYS measure both GLB1 and NEU1 in leukocytes simultaneously."},
            {"name": "Three-Types-GM1-Genotype-Phenotype-Residual-GLB1-Activity", "definition": "GM1 has 3 forms determined by residual GLB1 activity: Type 1 (infantile, null alleles, 0% activity; onset 0-6m; death ~3y), Type 2 (late-infantile/juvenile, missense, 1-5% activity; onset 7m-3y; epilepsy dominant), Type 3 (adult/chronic, missense, 5-10% activity; onset 3-30y; dystonia dominant; longest survival)."},
            {"name": "ACTH-Level-A-Infantile-Spasms-Type1-Early-Type2-Prefer-Over-VGB-Cherry-Red", "definition": "ACTH (tetracosactide) is Level A first-line for infantile spasms in GM1 Types 1 and early 2. VGB is alternative Level A for infantile spasms BUT cherry-red macular spot (Type 1: 50-90%; Type 2: 25-40%) creates VGB retinopathy risk. ACTH preferred when cherry-red present. Fundoscopy BEFORE choosing between ACTH and VGB."},
            {"name": "CBZ-OXC-PHT-ABSOLUTE-CI-GM1-Type2-Myoclonic-Atonic-Seizures", "definition": "CBZ/OXC/PHT ABSOLUTE CI in GM1 Type 2 (myoclonic component). Na-channel blockers paradoxically worsen cortical myoclonic seizures. GM1 Type 2 GTCS misidentified as idiopathic epilepsy → CBZ → acute myoclonic worsening. Mean diagnosis delay 2.8y = extended CBZ exposure window."},
            {"name": "VPA-SAFE-GLB1-Lysosomal-NOT-Mitochondrial-POLG1-MERRF-Mandatory-Before-VPA", "definition": "VPA SAFE in GM1 — GLB1 is lysosomal glycoside hydrolase, not mitochondrial. POLG1 Alpers and MERRF are the most dangerous GM1 phenocopies (progressive encephalopathy + seizures + VPA ABSOLUTE CI). MANDATORY: blood lactate + POLG1 WES + MERRF PCR before VPA in any progressive neurodegeneration + seizures."},
            {"name": "VGB-Cherry-Red-Status-Determines-IS-Treatment-High-Risk-Type1-Usable-Type3", "definition": "VGB HIGH RISK when cherry-red spot present (Type 1: 50-90%; Type 2: 25-40%) — VGB retinopathy adds to macular vulnerability. VGB NOT absolute CI in Type 3 (no cherry-red, no retinal involvement). Fundoscopy result determines ACTH vs VGB choice for infantile spasms."},
            {"name": "Fosphenytoin-ABSOLUTE-CI-IV-LEV-Replaces-GM1-SE-Protocol", "definition": "Fosphenytoin (standard SE second-line) is ABSOLUTE CI in GM1 gangliosidosis (worsens myoclonic SE). Replace with IV LEV 60 mg/kg in ALL GM1 SE protocols. Pre-inform A&E: 'GM1 gangliosidosis — FOSPHENYTOIN ABSOLUTE CI — IV LEV INSTEAD.'"},
            {"name": "Type3-Bilateral-Putaminal-MRI-T2-Hyperintensity-PATHOGNOMONIC-Adult-GM1", "definition": "Bilateral putaminal T2 signal hyperintensity on MRI is PATHOGNOMONIC for adult/chronic GM1 gangliosidosis (Type 3). Distinguishes from Wilson disease (T1 signal, Kayser-Fleischer rings, ceruloplasmin), pantothenate kinase deficiency (PANK2 'eye-of-the-tiger' sign), and other putaminal diseases. Japanese founder p.Ile51Thr enriched."},
            {"name": "L-DOPA-Trihexyphenidyl-Type3-Dystonia-ONLY-Not-Type1-2", "definition": "L-DOPA and trihexyphenidyl are for GM1 TYPE 3 dystonia-parkinsonism ONLY. NOT effective in Type 1 or Type 2 (different pathomechanism). Misclassifying Type 2 as Type 3 → inappropriate L-DOPA → no benefit + side-effects + delays AED optimisation."},
            {"name": "AAV-GLB1-Gene-Therapy-Phase-I-II-2024-Most-Advanced-Lysosomal-Epilepsy-Trial", "definition": "AAV9-GLB1 intrathecal gene therapy is in Phase I/II clinical trials (2024; NCT04273269 Cornell-Weill). Most clinically advanced lysosomal disease gene therapy in this epilepsy series. Pre-symptomatic eligibility highest likelihood of benefit. All newly diagnosed GM1 Type 1/2 → urgent referral to trial centre."},
            {"name": "MPS-IVB-Morquio-B-Odontoid-Hypoplasia-Anaesthesia-Risk-Same-GLB1-Gene", "definition": "MPS IVB (Morquio B) is caused by same GLB1 gene but with keratan sulphate accumulation in skeleton. Odontoid hypoplasia → C1-C2 instability. MANDATORY cervical spine MRI before any GA in GLB1 patients with skeletal features. Alert anaesthetist."},
            {"name": "Miglustat-Substrate-Reduction-Off-Label-Limited-Evidence-Not-Primary-AED", "definition": "Miglustat (iminosugar SRT; glucosylceramide synthase inhibitor) is used off-label in GM1 Type 3. Limited evidence for neurological stabilisation. Does NOT replace AED backbone for seizure management. Adjunct in specialist metabolic centre only."},
            {"name": "Dual-Disease-Single-Gene-GM1-vs-MPS-IVB-Phenotype-Allele-Dependent", "definition": "GLB1 is a dual-disease gene: same gene causes GM1 Gangliosidosis (CNS ganglioside accumulation → neurodegeneration + seizures) OR MPS IVB/Morquio B (skeletal keratan sulphate accumulation → skeletal dysplasia, minimal CNS). Phenotype depends on allele type and substrate selectivity of residual enzyme."},
            {"name": "SUDEP-Risk-GM1-Type2-Drug-Resistant-Nocturnal-GTCS-Progressive", "definition": "GM1 Type 2 has elevated SUDEP risk: drug-resistant (52%) + nocturnal GTCS + progressive neurodegeneration (reduced arousal response). Nocturnal monitoring device (SAMi) + supervised sleeping environment + SUDEP counselling mandatory from diagnosis."}
        ],
        "thresholds": [
            {"parameter": "Leukocyte β-galactosidase (GLB1)", "value": "<10% of mean control activity", "action": "GM1 Gangliosidosis diagnosis confirmed. Simultaneously measure NEU1 — if ALSO low → galactosialidosis (CTSA WES). GLB1 only low → GM1 (GLB1 WES)."},
            {"parameter": "Leukocyte α-neuraminidase (NEU1) — measured simultaneously", "value": "Within normal range (NEU1 normal)", "action": "Normal NEU1 confirms GM1 (not galactosialidosis). If NEU1 also low → galactosialidosis. Document NEU1 result alongside GLB1."},
            {"parameter": "DBS β-galactosidase (neonatal screening)", "value": "Below 2nd percentile of population reference", "action": "Recall for leukocyte confirmation + GLB1 WES. Pre-symptomatic identification enables gene therapy trial referral before neuronal loss."},
            {"parameter": "VPA serum level", "value": "Target 50-100 mg/L", "action": "TDM monthly x3; then 6-monthly. Below 50 mg/L → dose increase. Above 100 mg/L → hepatotoxicity risk; reduce dose."},
            {"parameter": "ACTH response assessment (infantile spasms)", "value": "Spasm-free at 2 weeks", "action": "Spasm cessation → continue ACTH taper per protocol. Incomplete response → add KD; reassess gene therapy eligibility."},
            {"parameter": "GMFCS progression", "value": "Change of ≥1 GMFCS level in 6 months", "action": "Accelerated motor deterioration: MDT conference; physiotherapy intensification; powered wheelchair assessment; PEG timing assessment; gene therapy trial eligibility reassessment."},
            {"parameter": "MRI putaminal signal (Type 3 monitoring)", "value": "New bilateral T2 hyperintensity or progression", "action": "Confirms Type 3 diagnosis. Baseline for dystonia progression. Note: acute deterioration on DWI → exclude ischaemic stroke vs GM1 storage progression."},
            {"parameter": "LFT (on VPA)", "value": "ALT/AST >2× ULN", "action": "Pause VPA dose increase; recheck 2 weeks; if >3× ULN → reduce/stop VPA; POLG1 confirmed excluded; consult metabolic hepatology."},
            {"parameter": "Drug-resistant epilepsy threshold", "value": "≥2 appropriate AEDs tried and failed at adequate doses", "action": "Consider KD; gene therapy trial eligibility; referral to paediatric epilepsy centre; review diagnosis (confirm GM1 WES, exclude POLG1 phenocopy)."},
            {"parameter": "Leukocyte enzyme assay stability (GLB1 DBS)", "value": "DBS stable at ambient temperature for 14 days (unlike NEU1 fresh leukocytes)", "action": "GLB1 DBS can be posted. NEU1 leukocyte requires fresh cells (4h). Neonatal screening uses DBS. Diagnostic confirmation: leukocyte GLB1+NEU1 (must measure NEU1 to exclude galactosialidosis)."},
            {"parameter": "SUDEP risk — nocturnal GTCS frequency", "value": ">1 nocturnal GTCS/month", "action": "High SUDEP risk: nocturnal monitoring device (SAMi); supervised sleeping; optimise AED; SUDEP Action resources; carer training."},
            {"parameter": "Cervical spine stability (MPS IVB screening)", "value": "Odontoid-dens distance >3 mm on lateral flexion/extension MRI", "action": "C1-C2 instability: neurosurgical referral; posterior cervical fusion consideration; hard cervical collar; anaesthesia alert for difficult airway + cord injury risk."}
        ],
        "standards": [
            "Okada S & O'Brien JS (1968) Science 160:1002 — first enzymatic diagnosis of GM1 gangliosidosis; β-galactosidase deficiency in leukocytes",
            "Suzuki K (1968) J Neurochem 15:285 — GM1 ganglioside classification and nomenclature",
            "Yoshida K et al. (1991) Biochem Biophys Res Commun 181:831 — Japanese founder mutation p.Ile51Thr in Type 3 adult/chronic GM1",
            "Brunetti-Pierri N & Scaglia F (2008) Mol Genet Metab 94:391 — comprehensive clinical review GM1 gangliosidosis",
            "Mole SE & Cotman SL (2015) Biochim Biophys Acta 1852:2262 — lysosomal storage disease classification and epilepsy review",
            "ILAE Task Force (2022) Epilepsia 63:1663 — classification and management recommendations for genetic epilepsies",
            "NICE NG217 (2021) — Epilepsies: diagnosis and management; AED prescribing framework including metabolic epilepsies",
            "Mhra VPPP (2021) — Valproate Pregnancy Prevention Programme; teratogenicity counselling mandatory",
            "CPIC Valproate/POLG Guidelines (2023) — mandatory POLG1 testing before valproate in progressive neurodegeneration",
            "ACMG/AMP Variant Classification Standards (2015) — pathogenicity criteria for GLB1 variants",
            "NCT04273269 — AAV9-GLB1 Gene Therapy Phase I/II (Cornell-Weill); eligibility: GM1 Types 1/2, age <2y, pre-symptomatic preferred",
            "Lysosomal Disease Network (LDN-RDCRN) (2024) — GM1 natural history registry and trial coordination"
        ],
        "references": [
            "Okada S & O'Brien JS (1968) Generalized gangliosidosis: beta-galactosidase deficiency. Science 160:1002.",
            "Yoshida K et al. (1991) Mutation in the beta-galactosidase gene (GLB1) of a Japanese patient with adult GM1 gangliosidosis. Biochem Biophys Res Commun 181:831.",
            "Brunetti-Pierri N & Scaglia F (2008) GM1 gangliosidosis: review of clinical, molecular, and therapeutic aspects. Mol Genet Metab 94:391.",
            "Regier DS & Tifft CJ (2013) GLB1-Related Disorders. GeneReviews [Internet]. PMID: 23409120.",
            "Roze E et al. (2005) Dystonia and parkinsonism in GM1 type 3 gangliosidosis. Mov Disord 20:1366.",
            "Gray-Edwards HL et al. (2017) AAV gene therapy for GM1 gangliosidosis: large animal proof-of-concept. Mol Ther 25:1462."
        ]
    }
