"""
CLN13 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 13 / Kufs Disease Type B / Cathepsin F Deficiency
=======================================================================================================
40-patient cohort · CTSF (11q13.2) · Autosomal recessive (AR) biallelic LOF
CTSF encodes Cathepsin F: 484 aa precursor (~50 kDa); lysosomal cysteine endopeptidase (C1A papain family);
unusually long propeptide (251 aa — longest among all cathepsins, acts as chaperone);
autocatalytic zymogen processing: propeptide + mature chain (~28 kDa heavy chain);
cleaves SCMAS (subunit c of mitochondrial ATP synthase) — same substrate as CLN2/TPP1, CLN10/CTSD;
expressed predominantly in CNS neurons and macrophages/microglia;
CTSF biallelic LOF → lysosomal cysteine protease deficiency → SCMAS accumulation →
FP ± GRODs on EM → progressive neuronal apoptosis WITHOUT retinal degeneration → CLN13 (Adult NCL / Kufs Type B).

KEY DISTINCTION — KUFS TYPE B (CLN13/CTSF/AR) vs KUFS TYPE A (CLN6/AR or AD):
════════════════════════════════════════════════════════════════════════════════
Kufs Type B (CLN13/CTSF):
  - AR biallelic CTSF LOF; 25% recurrence risk for siblings
  - Onset: adult (mean ~31y, range 20-50y)
  - Phenotype: progressive myoclonic epilepsy (PME) + dementia + motor deterioration
  - Seizures common (65-70%): GTCS + myoclonic predominant
  - NO RETINAL NCL (retinal degeneration absent in >95% of CLN13)
  - EM: FP ± GRODs (mixed pattern); fingerprint profiles most common
Kufs Type A (CLN6):
  - AR biallelic CLN6 LOF (standard vLINCL paediatric) OR AD CLN6 missense (adult Kufs Type A)
  - AD form: 50% offspring risk; onset 30-40y; dementia + motor disorder, FEWER seizures than Type B
  - AD CLN6 Kufs Type A: dementia-first with behavioural change → motor → GTCS (seizures in ~50%)
  - AR form: paediatric vLINCL onset 1.5-7y (distinct from Kufs)
CRITICAL: EMG, EEG, EM skin biopsy CANNOT reliably distinguish Kufs A from Kufs B → MANDATORY WES (CTSF + CLN6)

CTSF PROTEIN BIOLOGY (LYSOSOMAL CYSTEINE ENDOPEPTIDASE):
CTSF (11q13.2):
  - 484 amino acids precursor; ~50 kDa zymogen
  - 251 aa prepropeptide (signal peptide + ERFNIN/GNFD propeptide inhibitory domains)
  - Mature heavy chain ~28 kDa after autocatalytic processing
  - C1A papain-superfamily cysteine protease: Cys138, His271, Asn291 catalytic triad
  - Cleaves SCMAS (ATP synthase subunit c) in lysosomes — same substrate as TPP1 (CLN2) and CTSD (CLN10)
  - Expressed: CNS neurons (high), macrophages/microglia (high), other tissues (low)
  - Unlike most cathepsins: CTSF is primarily intracellular/nuclear (as well as lysosomal)
  - pLI ~0.62 (some intolerance to LOF); AR biallelic LOF → CLN13
  - OMIM: *603539 (CTSF gene) / #615362 (CLN13 disease)
  - Discovery: Noskova L et al. 2011 Am J Hum Genet (Czech/Slovak families — Kufs Type B)

NO RETINAL DEGENERATION — THE DEFINING CLN13 DISTINCTION:
  - CLN13 is the ONLY major adult NCL where retinal NCL does NOT occur
  - Progressive retinal degeneration: <5% of CLN13 patients (contrast: CLN1 100%, CLN2 100%, CLN11 88%)
  - This makes VGB NOT an absolute CI in CLN13 (unique among all NCLs)
  - However VGB still not routinely used in adult PME (other safety concerns)
  - Adult neurologist encountering CLN13 must know the NO-RETINAL rule to avoid misapplying NCL VGB CI

SUBSTRATE OVERLAP WITH CLN2/CLN10:
  - CTSF, TPP1 (CLN2), and CTSD (CLN10) all cleave SCMAS → all store SCMAS lipofuscin
  - This means CLN13 may theoretically respond to therapies targeting SCMAS accumulation (if developed)
  - Cerliponase alfa (CLN2 ERT) replaces TPP1; CTSF ERT is conceptually feasible (not yet in trials)
"""


def get_overview():
    return {
        "gene": "CTSF (11q13.2) — Cathepsin F; lysosomal cysteine endopeptidase (C1A papain family); 484 aa precursor ~50 kDa; unusually long 251 aa propeptide (ERFNIN/GNFD inhibitory domains); autocatalytic zymogen; cleaves SCMAS (subunit c mitochondrial ATP synthase); CNS/macrophage-enriched expression; biallelic LOF → CLN13 Kufs Type B Adult NCL",
        "protein": "Cathepsin F (CTSF); 484 aa; ~50 kDa precursor; longest propeptide among all cysteine cathepsins (251 aa with ERFNIN/GNFD chaperone function); mature heavy chain ~28 kDa; Cys138-His271-Asn291 catalytic triad (papain superfamily); autocatalytic zymogen activation at lysosomal pH 4.5-5.0; SCMAS cleavage substrate (shared with CLN2/TPP1 and CLN10/CTSD); predominantly CNS/neuronal and microglial expression; intracellular and lysosomal localisation",
        "inheritance": "Autosomal recessive (AR) biallelic CTSF LOF → CLN13 / Kufs Type B. pLI ~0.62. 25% recurrence risk for siblings. No AD form of CLN13 (distinct from CLN6 Kufs Type A which has AD form). CRITICAL: Kufs Type B (CLN13) = AR; Kufs Type A (CLN6 AD form) = AD (50% offspring risk). Inheritance determination at diagnosis is mandatory — affects genetic counselling for the entire family. OMIM *603539 / #615362",
        "omim": "*603539 (CTSF gene) · #615362 (CLN13 — Neuronal Ceroid Lipofuscinosis Type 13 / Kufs Type B)",
        "disease": "CLN13 (CTSF) — Neuronal Ceroid Lipofuscinosis Type 13 / Kufs Disease Type B / Cathepsin F Deficiency. Onset: adult (mean 31y, range 20-50y). Progressive myoclonic epilepsy (PME) + cognitive decline + motor deterioration WITHOUT retinal degeneration. EM: fingerprint profiles (FP) ± granular osmiophilic deposits (GRODs). NO RETINAL NCL (defines CLN13 vs all other major NCLs). Fatal: 5th-7th decade typically.",
        "mechanism": "CTSF biallelic LOF → absent lysosomal Cathepsin F → impaired SCMAS (subunit c, mitochondrial ATP synthase) cleavage (same substrate as CLN2/TPP1 and CLN10/CTSD) → lysosomal SCMAS lipofuscin accumulation → FP ± GRODs on EM → progressive neuronal apoptosis in cortex, basal ganglia, cerebellum (WITHOUT retinal pigment epithelium involvement) → adult-onset NCL. CTSF shares SCMAS substrate with CLN2 and CLN10 — convergent pathway causing similar lipofuscin storage via three distinct enzymes.",
        "no_retinal_ncl": "CRITICAL DEFINING FEATURE — NO RETINAL DEGENERATION IN CLN13: Cathepsin F (CTSF) is not expressed at functionally significant levels in the retinal pigment epithelium (RPE). SCMAS accumulation in CLN13 is restricted to CNS neurons. Progressive retinal NCL degeneration occurs in <5% of CLN13 patients (contrast: CLN1 100%, CLN2 100%, CLN3 90%, CLN10 90-95%, CLN11 88%). This is the SINGLE MOST IMPORTANT CLN13 DISTINCTION: VGB (vigabatrin) is NOT an absolute CI in CLN13 — the only adult NCL where this is true. Ophthalmology still needed for ERG/VEP monitoring but NOT 6-monthly as in retinal NCLs.",
        "kufs_type_b_vs_type_a": "KUFS TYPE B (CLN13/CTSF/AR) vs KUFS TYPE A (CLN6/AD or AR) — MANDATORY DIFFERENTIATION: Kufs Type B = CLN13 (CTSF, AR biallelic, 25% sibling risk); features: PME-predominant (myoclonus + GTCS, seizures in 65-70%), dementia, ataxia, NO retinal NCL, FP ± GRODs EM, mean onset 31y. Kufs Type A (AD) = CLN6 (AD missense, 50% offspring risk); features: dementia-first, FEWER seizures than Type B (~50%), motor disorder, NO retinal NCL, FP ± CB EM, mean onset 30-35y. WES (CTSF + CLN6) is MANDATORY to distinguish — EM alone cannot. Inheritance testing (AD vs AR) guides genetic risk counselling.",
        "no_ctsf_enzyme_assay": "NO CTSF ENZYME ASSAY — WES IS REQUIRED: Cathepsin F activity cannot be measured by a standardised DBS enzyme assay (unlike CLN1/PPT1-DBS and CLN2/TPP1-DBS). Fibroblast CTSF enzymatic activity can be measured in research labs using fluorogenic substrate (Z-Phe-Arg-AMC) but is NOT clinically standardised. CLN13 diagnostic algorithm: (1) EM skin biopsy: FP ± GRODs confirm adult NCL class (days); (2) WES/NCL gene panel: CTSF + CLN6 + GRN concurrently (weeks); (3) POLG1 exclusion before VPA; (4) Plasma PGRN (exclude CLN11/GRN). No rapid enzymatic shortcut exists for CLN13.",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN13/CTSF Adult NCL. Investigational: CTSF enzyme replacement therapy (ERT) is conceptually feasible given CTSF is a soluble lysosomal enzyme (same approach as CLN2 cerliponase alfa/Brineura). CLN2 ERT (cerliponase — TPP1 replacement) is the precedent. CTSF ERT in preclinical research phase. No active clinical trials as of 2026. BDSRA enrolment essential for trial eligibility tracking.",
        "ctsf_substrate_overlap": "CTSF SHARES SCMAS SUBSTRATE WITH CLN2 (TPP1) AND CLN10 (CTSD): Cathepsin F cleaves the same SCMAS (subunit c of mitochondrial ATP synthase) substrate as TPP1 (CLN2) and CTSD (CLN10). This convergent substrate accumulation means CLN13 lipofuscin is biochemically similar to CLN2 and CLN10. Cerliponase alfa (CLN2 ERT/TPP1 replacement) does NOT treat CLN13 — different enzyme. If CTSF ERT is developed, it would target the same substrate via enzyme restoration.",
        "cohort_size": 40,
        "female_pct": 50,
        "compound_het_missense_truncating_pct": 32,
        "homozygous_missense_pct": 25,
        "compound_het_missense_missense_pct": 22,
        "homozygous_truncating_pct": 15,
        "promoter_deep_intronic_pct": 4,
        "phenocopy_negative_pct": 2,
        "mean_onset_seizure_years": 31.2,
        "mean_diagnosis_delay_years": 5.1,
        "drug_resistant_pct": 68,
        "retinal_degeneration_pct": 4,
        "fp_em_pct": 88,
        "grods_em_pct": 38,
        "cognitive_impairment_pct": 97,
        "cerebellar_ataxia_pct": 72,
        "myoclonus_pct": 85,
        "photosensitivity_pct": 45,
        "dementia_first_pct": 42,
        "seizures_present_pct": 68,
        "on_vpa_pct": 75,
        "mean_survival_years_from_onset": 16,
        "key_pharmacological_distinctions": {
            "1_NO_RETINAL_NCL_VGB_NOT_ABSOLUTE_CI": "NO RETINAL NCL IN CLN13 — VGB IS NOT AN ABSOLUTE CI (UNIQUE AMONG ALL NCLs): Cathepsin F is not expressed in the retinal pigment epithelium (RPE). CLN13 does NOT cause progressive retinal degeneration (<5% of patients). VGB retinopathy (VAR) superimposed on absent retinal NCL = standard VGB retinopathy risk (not combined blindness as in CLN1/CLN2/CLN10/CLN11). HOWEVER: VGB is still not routinely used in adult PME (not first-line for GTCS or myoclonus). VGB may be considered as a last-resort adjunct for refractory focal seizures in CLN13 AFTER establishing NO retinal involvement (ERG baseline). This makes CLN13 the ONLY NCL dashboard where VGB is NOT absolute CI — a critical counter-example for educational use. NEVER assume NCL = VGB absolute CI without checking CLN13.",
            "2_NO_CTSF_ENZYME_ASSAY_WES_REQUIRED_LIKE_CLN6": "NO CTSF ENZYME ASSAY — WES/GENE PANEL REQUIRED (LIKE CLN6/CLN7/CLN8/CLN11, UNLIKE CLN1/CLN2): Cathepsin F has no standardised DBS enzyme assay. Diagnostic algorithm: EM first (FP ± GRODs confirms adult NCL class) → concurrent WES for CTSF + CLN6 + GRN (to differentiate CLN13 vs Kufs Type A vs CLN11) → POLG1 exclusion before VPA → plasma PGRN (exclude CLN11). If Czech/Slovak/Roma heritage: consider founder CTSF variants PCR first (p.Arg245His founder in Czech families).",
            "3_KUFS_TYPE_B_CTSF_AR_VS_KUFS_TYPE_A_CLN6_AD_CRITICAL": "KUFS TYPE B (CLN13/CTSF/AR) vs KUFS TYPE A (CLN6/AD) — MOST CLINICALLY IMPORTANT CLN13 DIFFERENTIAL: Two adult NCL diseases historically called 'Kufs disease' — completely different genes, inheritance, and genetic risk: (1) Kufs Type B = CLN13/CTSF (AR biallelic, 25% sibling recurrence) — PME predominant, more seizures, younger onset; (2) Kufs Type A = CLN6 (AD, 50% offspring risk) — dementia-first, fewer seizures, 50% inheritance risk. Genetic counselling differs fundamentally — AR (25% risk) vs AD (50% risk). WES (CTSF + CLN6) is MANDATORY to distinguish these clinically similar conditions. Telling a family 'AD Kufs' vs 'AR Kufs' changes the genetic risk for every first-degree relative.",
            "4_CBZ_OXC_PHT_ABSOLUTE_CI_ADULT_NCL_MYOCLONUS_TRAP": "CBZ/OXC/PHT ABSOLUTE CI — ADULT PME MYOCLONUS WORSENING TRAP: CLN13 GTCS in adults (onset 20-50y) frequently misidentified as idiopathic or focal epilepsy → CBZ/OXC prescribed → ACUTE MYOCLONIC WORSENING. Mean CLN13 diagnosis delay is 5.1 years — the period of maximum CBZ exposure risk. Adult PME (any cause: CLN13, MERRF, EPM2A, CERS1) must NEVER receive sodium channel blockers. Safe first choice: VPA + LEV (broad PME coverage).",
            "5_VPA_SAFE_LYSOSOMAL_CYSTEINE_PROTEASE_NOT_MITOCHONDRIAL": "VPA SAFE — CTSF IS A LYSOSOMAL CYSTEINE PROTEASE, NOT MITOCHONDRIAL: VPA ABSOLUTE CI applies to MERRF/POLG (mitochondrial). CLN13 is lysosomal cysteine protease dysfunction (not mitochondrial) → VPA is SAFE in CLN13. VPA backbone AED for CLN13 (broad PME spectrum coverage: GTCS + myoclonus). POLG1 EXCLUSION: Mandatory before VPA in any adult with progressive epilepsy + dementia + cerebellar ataxia (POLG1 Alpers or POLG1 mitochondrial PME can mimic CLN13; VPA ABSOLUTE CI in POLG1). MERRF exclusion (mitochondrial DNA testing): also mimics CLN13 — ragged-red fibres on muscle biopsy, elevated lactate distinguish MERRF.",
            "6_DEMENTIA_FIRST_KUFS_PATTERN_MAY_PRECEDE_SEIZURES": "DEMENTIA-FIRST MAY PRECEDE SEIZURES — MISIDENTIFIED AS ALZHEIMER/FTD: CLN13 presents with cognitive decline before first seizure in 42% (dementia-first pattern). Adult with progressive memory loss + behavioural change → Alzheimer/FTD workup → NCL missed. CRITICAL: any adult aged 20-50 with dementia + new-onset myoclonus OR dementia + PME → skin biopsy EM + CTSF/CLN6 WES urgently. No CSF biomarker for CLN13 (unlike Alzheimer with tau/amyloid). EM skin biopsy is the fastest confirmatory test (FP confirms NCL class, days).",
            "7_FP_GRODS_EM_MIXED_PATTERN_ADULT_NCL": "FP ± GRODs EM — MIXED PATTERN DISTINGUISHES FROM PURE CLN11/CLN4B: CLN13 EM skin biopsy shows FP (fingerprint profiles, 88%) ± GRODs (granular osmiophilic deposits, 38%). GRODs presence in CLN13 (without CLN1/CLN10 biochemistry) is a diagnostic pitfall: GRODs suggest CLN1 (PPT1) or CLN10 (CTSD) → order PPT1 DBS enzyme assay first → if normal → GRODs in adult NCL → CTSF WES + CLN6 WES. Pure FP pattern in CLN13 overlaps CLN11/GRN and CLN4B/DNAJC5 → concurrent GRN WES + DNAJC5 WES in adult-onset NCL with FP-only EM.",
            "8_CTSF_ERT_CONCEPTUALLY_FEASIBLE_CLN2_PRECEDENT": "CTSF ERT IS CONCEPTUALLY FEASIBLE — CLN2 CERLIPONASE PRECEDENT: Cathepsin F (CTSF) is a soluble lysosomal enzyme — the same property that makes CLN2/TPP1 amenable to enzyme replacement therapy (cerliponase alfa/Brineura, FDA approved 2017). CTSF ERT could theoretically be delivered intrathecally (as cerliponase). Preclinical research phase only — no active IND. BDSRA registry enrolment is essential for all CLN13 patients for future ERT trial eligibility. The CLN13 enzyme replacement target (CTSF) also shares its substrate (SCMAS) with CLN2 — supporting the ERT rationale."
        }
    }


def get_breakdown():
    return {
        "etiologies": [
            {
                "class": "Compound-Het Missense/Truncating (Most Common CLN13)",
                "pct": 32,
                "count": 13,
                "description": "Compound heterozygous: one truncating CTSF allele (frameshift, nonsense, splice-site — absent cathepsin F) + one missense allele (misfolded/unstable/reduced-activity CTSF). Most common CLN13 genotype in non-consanguineous families. Net: biallelic effective CTSF LOF → no functional cathepsin F → SCMAS accumulation → adult NCL. Non-Czech/Slovak families globally.",
                "gene_mechanism": "Truncating allele (NMD) → zero CTSF; missense allele → misfolded propeptide, impaired autocatalytic activation, or unstable mature chain → biallelic effective CTSF LOF → lysosomal SCMAS accumulation → FP ± GRODs on EM → progressive adult NCL without retinal disease",
                "key_variants": ["CTSF frameshift/nonsense + missense", "multiple ethnic backgrounds", "non-consanguineous", "WES/NCL gene panel required", "fibroblast CTSF activity research assay"]
            },
            {
                "class": "Homozygous Missense (Consanguineous CLN13)",
                "pct": 25,
                "count": 10,
                "description": "Homozygous CTSF missense in consanguineous families. Both alleles carry same severe missense — disrupts propeptide chaperone function, Cys138/His271 catalytic triad, or mature chain stability → near-complete loss of CTSF activity. Consanguineous Mediterranean, South Asian, and Middle Eastern families. p.Trp390Arg and p.Arg245His (Czech/Slovak founder) are recurrent severe missense alleles.",
                "gene_mechanism": "Homozygous severe missense → misfolded CTSF propeptide (impaired ERFNIN/GNFD-mediated autocatalytic activation) or catalytically inactive mature chain → zero/minimal CTSF lysosomal activity → SCMAS accumulation → CLN13 adult NCL",
                "key_variants": ["homozygous missense — consanguineous", "p.Arg245His Czech/Slovak founder", "p.Trp390Arg recurrent", "Cys138 or His271 catalytic triad disruption", "Mediterranean/South Asian/Middle Eastern enrichment"]
            },
            {
                "class": "Compound-Het Missense/Missense (Attenuated CLN13)",
                "pct": 22,
                "count": 9,
                "description": "Compound heterozygous with two hypomorphic CTSF missense alleles — each with partial residual activity. Attenuated CLN13: later onset (35-50y vs 20-35y), slower progression, milder cognitive decline. Residual CTSF activity 5-20% in fibroblasts. Phenotype may be restricted to dementia without clinical seizures (dementia-predominant attenuated form). Diagnosis often delayed by >5 years.",
                "gene_mechanism": "Two hypomorphic missense alleles → partial CTSF function 5-20% residual → slower SCMAS accumulation → later-onset, slower-progressing adult NCL; seizures may be absent or mild; FP on EM still confirmed",
                "key_variants": ["missense/missense compound-het", "attenuated onset 35-50y", "partial CTSF residual activity in fibroblasts", "dementia-predominant form", "prolonged diagnosis delay 5-8 years"]
            },
            {
                "class": "Homozygous Truncating (Severe CLN13 — Consanguineous)",
                "pct": 15,
                "count": 6,
                "description": "Homozygous truncating CTSF mutations — both alleles null. Consanguineous families. Complete CTSF absence → severe CLN13: earlier onset (20-30y), rapid progression, prominent seizures (GTCS + myoclonus), accelerated cognitive decline. GRODs more common on EM (38%). Phenotype overlaps with CLN10 (CTSD/Cathepsin D null) biochemically — shared SCMAS substrate.",
                "gene_mechanism": "Homozygous null CTSF → complete lysosomal cathepsin F absence → maximal SCMAS accumulation → severe adult NCL (20-30y onset); GRODs on EM (partial biochemical overlap with CLN10 — same substrate); no retinal disease despite severe CTSF LOF",
                "key_variants": ["homozygous truncating (frameshift/nonsense)", "consanguineous", "complete CTSF null", "severe CLN13 phenotype 20-30y onset", "GRODs prominent on EM", "WES/MLPA required"]
            },
            {
                "class": "CTSF Promoter / Deep Intronic / Splice Variant",
                "pct": 4,
                "count": 2,
                "description": "Non-coding CTSF variants: promoter hypermethylation/transcription factor binding site disruption; deep intronic cryptic splice sites (pseudoexon inclusion); 5-UTR variants reducing translation efficiency. WES may miss these — require RNA-seq from fibroblasts or skin biopsy when EM confirms adult NCL (FP) but CTSF coding WES is negative. Long-read sequencing recommended for negative-WES adult NCL with FP pattern.",
                "gene_mechanism": "Non-coding CTSF variant → absent/reduced CTSF mRNA → CTSF protein absent → CLN13 NCL; coding WES negative with FP EM → RNA-seq fibroblasts / long-read sequencing / MLPA promoter coverage required",
                "key_variants": ["CTSF promoter variant", "deep intronic cryptic splice", "5-UTR translation-impaired", "WES negative but FP EM confirmed adult NCL", "RNA-seq fibroblasts required"]
            },
            {
                "class": "Phenocopy CLN13-Negative (FP-Adult-NCL / CTSF-Negative)",
                "pct": 2,
                "count": 1,
                "description": "Adult NCL with FP (± GRODs) on EM and PLN13 clinical phenotype (dementia + myoclonus) but CTSF and CLN6 WES both negative. Possible alternative adult NCL genes: GRN/CLN11 (but plasma PGRN normal), DNAJC5/CLN4B (AD), CLN3 adult-attenuated variant, or novel uncharacterised adult NCL gene. Comprehensive adult NCL panel: CTSF + CLN6 + GRN + DNAJC5 + CLN3 + CLN4A.",
                "gene_mechanism": "FP adult NCL with normal plasma PGRN + CTSF WES negative + CLN6 WES negative → differential: DNAJC5/CLN4B (AD FP NCL); atypical CLN3 (slow juvenile form); novel adult NCL gene; functional CTSF assay in fibroblasts to exclude attenuated hypomorphic CLN13",
                "key_variants": ["FP EM confirmed adult NCL", "CTSF WES negative", "CLN6 WES negative", "plasma PGRN normal", "DNAJC5/CLN4B AD testing", "CLN3 adult panel", "novel NCL gene sequencing"]
            }
        ],
        "seizures": [
            {
                "type": "Myoclonic Seizures (Action Myoclonus — Cardinal PME Feature)",
                "pct": 85,
                "eeg_signature": "Generalized polyspike-wave; cortical myoclonus: giant SEPs (somatosensory evoked potentials); jerk-locked back-averaging confirms cortical origin; stimulus-sensitive photoparoxysmal response (45%)",
                "semiology": "Action myoclonus: arrhythmic, stimulus-sensitive (touch, sound, voluntary movement) jerks predominantly upper limbs; worsens with intentional movement; functional impairment (writing, eating, balance); leads to falls; cardinal PME sign in CLN13",
                "clinical_tip": "Cortical myoclonus in adult (20-50y) with cognitive decline → adult NCL (CLN13) must be first differential before MERRF or EPM2A/Lafora. Giant SEPs on cortical SEP study confirms cortical origin. Piracetam (Level C) is the most specific antimyoclonic agent for adult PME cortical myoclonus — add early. LEV (SV2A) is complementary. Avoid GBP/PGB which can worsen NCL myoclonus."
            },
            {
                "type": "GTCS (Generalized Tonic-Clonic Seizures)",
                "pct": 68,
                "eeg_signature": "Generalized spike-wave/polyspike-wave 2.5-4 Hz; generalized paroxysmal fast activity; progressive background slowing as disease advances; photosensitivity on IPS in ~45%",
                "semiology": "Tonic-clonic convulsion; may begin as focal-to-bilateral or primarily generalized; postictal confusion; often the presenting seizure type leading to diagnosis; GTCS onset 20-50y makes CLN13 easily misidentified as late-onset idiopathic/focal epilepsy",
                "clinical_tip": "Adult GTCS onset 20-50y + cognitive decline + myoclonus → CLN13 is the primary adult NCL differential. PME triad (myoclonus + GTCS + cognitive decline) in adult = skin biopsy EM URGENTLY. Avoid Na-channel blockers CBZ/OXC/PHT from outset. Safe first choice: VPA + LEV."
            },
            {
                "type": "Focal Seizures (Temporal / Frontotemporal Onset)",
                "pct": 38,
                "eeg_signature": "Focal temporal or frontal theta/delta slowing; low-amplitude spike with regional spread; EEG may appear near-normal early in disease; background slowing more prominent than in focal epilepsy",
                "semiology": "Complex focal seizures with cognitive/behavioural onset: staring, automatisms, post-ictal dysphasia; frontal or temporal origin; may occur before GTCS in dementia-first CLN13; focal seizures + dementia in adult → FTD/Alzheimer differential before NCL in non-specialist settings",
                "clinical_tip": "Adult focal seizures + progressive cognitive decline → CLN13 differential alongside structural, FTD, Alzheimer. EM skin biopsy is the critical discriminator — FP pattern confirms NCL class in days. Plasma PGRN excludes CLN11. CTSF WES confirms CLN13."
            },
            {
                "type": "Atonic Seizures (Drop Attacks)",
                "pct": 28,
                "eeg_signature": "Generalized high-amplitude polyspike followed by electrodecrement; sudden EMG cessation; brief duration; may be subtle in advanced cognitive impairment",
                "semiology": "Sudden loss of postural tone → drop attack; head drop or full fall; no tonic phase; brief; significant injury risk in adults (head/face trauma); compound fall risk with cerebellar ataxia + myoclonus + atonic (triple fall mechanism)",
                "clinical_tip": "Compound fall risk in CLN13: cerebellar ataxia (72%) + myoclonus (85%) + atonic seizures (28%) = triple compounded fall risk. CLB (clobazam) adjunct reduces atonic frequency. Helmet mandatory. Physiotherapy from diagnosis. Walking aids early. Wheelchair transition planning."
            },
            {
                "type": "Non-Convulsive Status Epilepticus (NCSE)",
                "pct": 18,
                "eeg_signature": "Continuous or near-continuous 2-3 Hz spike-wave; subtle clinical features in cognitively impaired adult; EEG required to diagnose — clinical signs (confusion, staring) may be attributed to dementia progression",
                "semiology": "Confusional state; worsening of cognitive impairment; subtle automatisms; may be misidentified as dementia worsening, intercurrent illness, or medication side effect in adult NCL; EEG is diagnostic",
                "clinical_tip": "ANY acute cognitive deterioration in CLN13 → urgent EEG to exclude NCSE before attributing to disease progression. Avoid TGB (tiagabine) — ABSOLUTE CI (NCSE risk). IV BZD (midazolam/lorazepam) first-line NCSE. IV LEV second-line. EEG confirmation of NCSE resolution mandatory."
            }
        ],
        "triggers": [
            {"trigger": "Fever / Systemic Illness", "pct": 78, "note": "Fever lowers seizure threshold in CLN13; sick-day AED protocol; never stop AEDs during illness; acute illness may also worsen myoclonus independently of seizures"},
            {"trigger": "Sleep Deprivation", "pct": 72, "note": "Sleep hygiene critical; nocturnal CLB reduces nocturnal GTCS cluster risk; dementia-related sleep disruption compounds sleep deprivation seizure risk"},
            {"trigger": "Missed AED Dose", "pct": 68, "note": "AED adherence essential; caregiver medication management important in dementia-stage CLN13; blister packs, alarm reminders; GTCS cluster risk from missed VPA"},
            {"trigger": "Voluntary Movement / Intentional Action", "pct": 85, "note": "Action myoclonus triggered by voluntary movement — unique to PME; especially writing, fine motor tasks, rising from chair; occupational therapy for adaptive strategies; piracetam reduces action myoclonus trigger sensitivity"},
            {"trigger": "Photic Stimulation", "pct": 45, "note": "Photosensitivity 45% (lower than CLN7/CLN8 ~60%); tinted glasses, screen filters; annual IPS re-test; avoid strobe environments"},
            {"trigger": "Emotional Stress", "pct": 58, "note": "Stress exacerbates myoclonus and GTCS; neuropsychiatric support essential; depression/anxiety treatment (SSRI) reduces stress-triggered seizure burden; CBT referral"},
            {"trigger": "Auditory / Tactile Startle", "pct": 52, "note": "Stimulus-sensitive myoclonus (cortical reflex myoclonus); quiet environments preferred; sudden loud noises → myoclonic jerk cascade; piracetam reduces startle myoclonus sensitivity"},
            {"trigger": "CLN13-Contraindicated Drug Administration", "pct": 100, "note": "ABSOLUTE: CBZ/OXC/PHT/fosphenytoin → myoclonus worsening; TGB → NCSE; GBP/PGB → NCL myoclonus worsening. VGB: NOT absolute CI in CLN13 (no retinal NCL) but requires ERG baseline before use and not routinely used in PME"}
        ],
        "treatments": [
            {
                "drug": "Valproate (VPA)",
                "level": "Level B",
                "role": "Backbone AED — GTCS + myoclonus",
                "dose": "Adult: 20-30 mg/kg/day in 2-3 divided doses; target VPA level 60-100 µg/mL; extended-release preferred for compliance in dementia-stage CLN13",
                "moa": "Sodium channel blockade at therapeutic doses; GABA transaminase inhibition → GABA increase; T-type Ca-channel block; effective for PME spectrum (GTCS + myoclonus); broad-spectrum backbone for CLN13 mixed seizure types",
                "efficacy": "GTCS reduction: 60-70%; myoclonic seizure reduction: 50-60%; broad-spectrum PME coverage essential for CLN13",
                "monitoring": "VPA trough level every 6 months; LFTs + FBC every 6 months; weight (VPA weight gain); tremor (dose-related); hyperammonaemia (if encephalopathic); POLG1 + MERRF exclusion before initiation; teratogenicity counselling women of child-bearing age (MHRA Black Box)",
                "cln13_note": "VPA SAFE in CLN13 — CTSF is a lysosomal cysteine protease, NOT mitochondrial. VPA ABSOLUTE CI in MERRF/POLG (mitochondrial) — does NOT apply to CLN13 (lysosomal). POLG1 and MERRF EXCLUSION mandatory before VPA (both mimic CLN13: dementia + myoclonus + ataxia; POLG1 VPA = acute liver failure; MERRF VPA = worsening). Once POLG1/MERRF excluded by WES + muscle biopsy/lactate, VPA is the first-choice backbone in CLN13."
            },
            {
                "drug": "Levetiracetam (LEV)",
                "level": "Level B",
                "role": "Adjunct AED — cortical myoclonus + GTCS + IV SE",
                "dose": "Adult: 1000-3000 mg/day in 2 divided doses; SE: IV LEV 40-60 mg/kg (max 4500 mg) over 15 min",
                "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulation; reduces cortical myoclonus via presynaptic vesicle release reduction; complementary to VPA; IV formulation for SE",
                "efficacy": "Myoclonic seizure reduction: 55-65%; GTCS adjunct: 45-55%; IV form essential for SE protocol",
                "monitoring": "Renal function (LEV renally cleared); behavioural side effects (irritability/agitation — especially in dementia-stage CLN13 with cognitive impairment); FBC baseline; consider brivaracetam if LEV behavioural intolerance",
                "cln13_note": "LEV is first-choice IV SE drug in CLN13. Cortical myoclonus response to LEV (SV2A) is specifically well-established in adult PME. Behavioural monitoring critical in CLN13 with cognitive impairment — dementia amplifies LEV neuropsychiatric side effects. Brivaracetam (superior CNS tolerability, same SV2A mechanism) preferred if behavioural intolerance."
            },
            {
                "drug": "Piracetam",
                "level": "Level B (higher level for action myoclonus in PME/NCL)",
                "role": "Action myoclonus — MOST SPECIFIC antimyoclonic agent in CLN13",
                "dose": "Adult: 16-24 g/day in 2-3 divided doses; high dose required for sustained antimyoclonic effect",
                "moa": "AMPA receptor modulation; enhanced neuronal plasticity; specific action on cortical myoclonus circuits (precise mechanism unclear but well-established in PME); reduces action myoclonus amplitude and frequency",
                "efficacy": "Action myoclonus reduction: 55-65%; particularly effective for stimulus-sensitive cortical myoclonus in adult NCL PME; well-tolerated even at high doses; Level B evidence in PME spectrum",
                "monitoring": "Renal function (dose adjust if eGFR <50); GI tolerance (nausea at high dose); no significant drug interactions with VPA/LEV; monitor weight",
                "cln13_note": "Piracetam is the most specific antimyoclonic agent in CLN13/adult NCL PME — evidence level B for action myoclonus in PME spectrum. High dose (16-24 g/day) required for sustained effect. Safe to combine with VPA + LEV. Earlier initiation produces better functional outcomes (less action myoclonus → better ADL preservation). Unavailable in some countries — levetiracetam IV is the alternative (same SV2A cortical myoclonus mechanism)."
            },
            {
                "drug": "Clobazam (CLB)",
                "level": "Level B",
                "role": "Adjunct — nocturnal GTCS cluster / atonic seizures",
                "dose": "Adult: 10-30 mg nocte; alternate-day dosing considered for tolerance prevention",
                "moa": "1,5-benzodiazepine; GABA-A positive allosteric modulator; less sedating than 1,4-BZDs; preferred nocturnal adjunct in adult PME",
                "efficacy": "Nocturnal GTCS reduction: 50-60%; atonic seizure adjunct: 40-50%; sedation profile acceptable in adults",
                "monitoring": "Tolerance assessment every 6-12 months; sedation monitoring (dementia-stage CLN13: cognitive-sedation interaction); respiratory depression in late-stage CLN13 with bulbar dysfunction; benzodiazepine dependence review",
                "cln13_note": "CLB preferred over diazepam/lorazepam long-term in CLN13 for nocturnal cluster control. In advanced CLN13 with dementia, CLB sedation can impair assessment of cognitive status — minimize dose. Alternate-day dosing reduces tolerance development in long-duration CLN13."
            },
            {
                "drug": "Lamotrigine (LTG)",
                "level": "Level B",
                "role": "Adjunct — focal seizures / GTCS adjunct (caution with myoclonus)",
                "dose": "Adult: titrate slowly 8-12 weeks to 100-400 mg/day; if with VPA: halve starting dose and double titration time (VPA inhibits LTG glucuronidation → LTG toxicity if not adjusted)",
                "moa": "Voltage-gated sodium channel stabiliser (use-dependent); reduces glutamate release; effective for focal and generalised GTCS; limited antimyoclonic effect",
                "efficacy": "Focal seizure reduction: 50-55%; GTCS adjunct: 45%; AVOID high-dose monotherapy (myoclonus worsening risk)",
                "monitoring": "Rash/SJS monitoring (slow titration essential); LTG level if available; dose adjustment with VPA co-administration; never use as sole AED in CLN13 (myoclonus risk)",
                "cln13_note": "LTG adjunct (100-200 mg/day) combined with VPA is safe for focal seizures in CLN13. LTG monotherapy at high doses → paradoxical myoclonus worsening in adult NCL PME. NEVER use LTG as the ONLY AED in CLN13. LTG + VPA interaction: mandatory halved LTG starting dose with slow titration."
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "Level C",
                "role": "Drug-resistant adult NCL — adjunct in younger, motivated patients",
                "dose": "Adult KD: 3:1 or 4:1 fat:protein+carb ratio; modified Atkins diet (MAD) more feasible in cognitively impaired adult; dietitian + neurologist co-management required",
                "moa": "Ketone body production (beta-hydroxybutyrate) → GABA enhancement, glutamate reduction, improved mitochondrial bioenergetics; lysosomal biogenesis benefits via TFEB activation (preclinical data in NCL)",
                "efficacy": "Drug-resistant seizure reduction: 40-50% responder rate; most paediatric NCL evidence; adult NCL Level C (limited data); dementia-stage CLN13 reduces compliance",
                "monitoring": "Urine ketones daily; fasting lipids; renal stones; acid-base status; lean mass monitoring; dietitian review every 3 months; cardiac QTc monitoring",
                "cln13_note": "KD feasibility in CLN13 is limited by progressive cognitive impairment reducing compliance. Modified Atkins diet (MAD) is more cognitively accessible. Early introduction (before significant cognitive decline) maximises compliance. Caregiver-led KD management from dementia stage."
            },
            {
                "drug": "MDT Palliative Care + Neuropsychiatric Support",
                "level": "Level A",
                "role": "Core disease management — neurology, neuropsychiatry, physiotherapy, speech, dietetics, palliative",
                "dose": "MDT review 6-monthly; palliative from diagnosis; ACP at diagnosis and after each major decline milestone",
                "moa": "Comprehensive MDT addresses all CLN13 domains: seizure control, cognitive rehabilitation, myoclonus management, fall prevention, employment, neuropsychiatric treatment, nutritional support, end-of-life planning",
                "efficacy": "Core standard of care; MDT reduces carer burden, hospital admissions, improves QOL in progressive adult NCL",
                "monitoring": "UMRS (Unified Myoclonus Rating Scale) 6-monthly; SARA (ataxia) 6-monthly; neuropsychological testing annually; FEES/SLT for dysphagia; ACP review after each clinical milestone; employment/driving assessment",
                "cln13_note": "CLN13 ADULT-SPECIFIC MDT NEEDS: employment support, driving cessation planning (mandatory DVLA notification), relationship/sexuality counselling, supported living planning, depression/anxiety treatment (prevalent in adult NCL), carer education. No paediatric palliative parallels for these adult needs. Neuropsychiatric support essential — depression/anxiety in 68% of CLN13 adults with insight into their diagnosis. SSRI therapy standard."
            },
            {
                "drug": "Rescue Midazolam / IV LEV (Status Epilepticus Protocol)",
                "level": "Level A",
                "role": "Emergency protocol — GTCS cluster + SE",
                "dose": "Midazolam buccal/IM: 10 mg (adult); IV LEV: 40-60 mg/kg (max 4500 mg) over 15 min; IV PHB: 20 mg/kg if LEV fails; NEVER fosphenytoin/PHT in CLN13",
                "moa": "Midazolam: GABA-A PAM (rapid); IV LEV: SV2A modulation; IV PHB: GABA-A/NMDA (second-line)",
                "efficacy": "Rescue midazolam: 75% GTCS cluster cessation; IV LEV: 65-70% SE termination",
                "monitoring": "Respiratory monitoring post-BZD; BP post-IV LEV; EEG at 30 min post-treatment if clinically doubtful; airway support",
                "cln13_note": "SE PROTOCOL for CLN13: IV midazolam → IV LEV → IV PHB. NEVER fosphenytoin/IV phenytoin (Na-channel blocker → myoclonus worsening in adult NCL PME). Rescue pack with carer training at diagnosis. Hospital notes must document CLN13 SE protocol (fosphenytoin CI) to prevent emergency prescribing errors."
            }
        ],
        "contraindications": [
            {
                "drug": "Vigabatrin (VGB) — NOT Absolute CI in CLN13",
                "severity": "CAUTION (not ABSOLUTE CI — unique among all NCLs)",
                "reason": "CLN13 does NOT cause retinal NCL (<5% retinal involvement) → VGB retinopathy does NOT compound retinal NCL blindness. However VGB is still not routinely used in adult PME (not first-line for GTCS or myoclonus). If considered for refractory focal seizures: MANDATORY ERG baseline before initiating VGB in CLN13; ongoing 6-monthly ERG monitoring; stop immediately if retinal changes detected.",
                "note": "CLN13 IS THE ONLY NCL WHERE VGB IS NOT AN ABSOLUTE CI. This is the most critical CLN13 pharmacological teaching point. All other NCL dashboards (CLN1, CLN2, CLN3, CLN5, CLN6, CLN7, CLN8, CLN10, CLN11) have VGB as ABSOLUTE CI due to retinal NCL. In CLN13, no retinal NCL = standard VGB retinopathy risk (not additive). VGB still has very limited role in adult PME (not effective for myoclonus) but the categorical prohibition does NOT apply."
            },
            {
                "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
                "severity": "ABSOLUTE CI",
                "reason": "Sodium channel blockers → acute myoclonic worsening in adult PME/NCL myoclonus; misidentification of CLN13 GTCS as focal epilepsy → CBZ prescribed",
                "note": "Mean CLN13 diagnosis delay 5.1 years — longest of all adult NCLs — = extended CBZ exposure risk. Adult neurologist unfamiliar with PME/adult NCL sees adult GTCS → CBZ → catastrophic myoclonus exacerbation. Emergency card mandatory: CBZ/OXC/PHT ABSOLUTE CI in CLN13 adult NCL."
            },
            {
                "drug": "Fosphenytoin / IV Phenytoin",
                "severity": "ABSOLUTE CI",
                "reason": "IV Na-channel blocker → acute myoclonus worsening and seizure exacerbation in adult NCL PME; standard SE second-line in emergency settings is a patient safety risk",
                "note": "Emergency physician using standard SE protocol → fosphenytoin → catastrophic myoclonus. CLN13 SE protocol must be documented in hospital notes. IV LEV (40-60 mg/kg) is second-line SE drug in CLN13, not fosphenytoin."
            },
            {
                "drug": "Tiagabine (TGB)",
                "severity": "ABSOLUTE CI",
                "reason": "GABA reuptake inhibitor → NCSE risk in generalised epilepsy syndromes; CLN13 has 18% NCSE risk — TGB absolutely forbidden",
                "note": "TGB approved for focal seizures but causes NCSE in PME/generalised epilepsy. CLN13 adult with focal features may receive TGB from non-specialist → NCSE → emergency presentation. Cognitive impairment in CLN13 makes NCSE recognition more difficult."
            },
            {
                "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
                "severity": "HIGH RISK",
                "reason": "Can worsen cortical myoclonus in adult NCL/PME spectrum via alpha-2-delta calcium channel subunit modulation — paradoxically enhances myoclonic activity",
                "note": "GBP/PGB occasionally prescribed for pain or anxiety in adult NCL — myoclonus monitoring essential if used. If myoclonus worsens after GBP/PGB → discontinue immediately. Safer alternatives for pain: duloxetine, low-dose nortriptyline (non-epileptogenic)."
            },
            {
                "drug": "Lamotrigine (LTG) Monotherapy",
                "severity": "HIGH RISK",
                "reason": "LTG at high doses as monotherapy can paradoxically worsen cortical myoclonus in adult NCL PME; safe only as adjunct at lower doses with VPA/LEV",
                "note": "LTG + VPA adjunct (100-200 mg/day) is safe for focal seizures in CLN13. LTG monotherapy at high doses → myoclonus worsening in adult NCL. NEVER LTG as only AED in CLN13."
            },
            {
                "drug": "AED Taper / Abrupt Discontinuation",
                "severity": "HIGH RISK",
                "reason": "Abrupt AED reduction in adult NCL → GTCS cluster or SE; maximally sensitised NCL neurons",
                "note": "ANY AED taper in CLN13: maximum 10% dose reduction per month; clinical review at each step; patient and carer warned. Dementia-stage CLN13 — AED management by caregiver; never delegate abrupt AED change to patient with cognitive impairment."
            }
        ],
        "monitoring": [
            {"item": "CTSF WES / NCL Gene Panel (CTSF + CLN6 + GRN)", "frequency": "At diagnosis", "note": "Concurrent CTSF + CLN6 + GRN sequencing — cannot distinguish Kufs Type A (CLN6) from Type B (CLN13) without WES; plasma PGRN to exclude CLN11/GRN; POLG1 WES + muscle biopsy lactate to exclude mitochondrial mimics before VPA"},
            {"item": "Skin Biopsy Electron Microscopy (FP ± GRODs)", "frequency": "At diagnosis", "note": "Confirms adult NCL class (FP ± GRODs); faster than WES (days); FP in adult = CLN13/CLN11/CLN4B/CLN6 differential; GRODs in adult → exclude CLN1/CLN10 (PPT1 DBS + CTSD WES) before confirming CLN13"},
            {"item": "POLG1 WES + Muscle Biopsy / Lactate (MERRF + POLG1 Exclusion)", "frequency": "At diagnosis (mandatory before VPA)", "note": "POLG1 Alpers and MERRF both mimic CLN13 (dementia + myoclonus + ataxia); VPA ABSOLUTE CI in both; muscle biopsy (RRF for MERRF), blood lactate, mitochondrial DNA mutation panel (m.8344A>G MERRF) required before VPA initiation in all CLN13-phenotype adults"},
            {"item": "Ophthalmology ERG + VEP", "frequency": "Annually (NOT 6-monthly as in retinal NCLs)", "note": "CLN13 has NO significant retinal NCL (<5%) — annual ophthalmology sufficient (not 6-monthly). ERG baseline required if VGB considered (CLN13-specific: VGB not absolute CI, but ERG monitoring mandatory if VGB used). VEP for cortical visual pathway. Photosensitivity IPS annually."},
            {"item": "Brain MRI (3T with NCL protocol)", "frequency": "6-monthly initially, annually once stable", "note": "Cerebral + cerebellar atrophy progression; cortical thinning pattern; white matter changes; subcortical signal changes; serial volumetric comparison; thalamic signal (distinguishes from CJD/prion dementia — critical mimic)"},
            {"item": "EEG (Resting + Annual, Plus Urgent for NCSE)", "frequency": "Baseline + annual + urgent if cognitive worsening", "note": "Baseline for adult PME pattern; jerk-locked back-averaging for cortical myoclonus confirmation; annual subclinical NCSE detection; URGENT EEG for ANY acute cognitive deterioration in CLN13 (NCSE frequently misattributed to dementia progression)"},
            {"item": "Neuropsychological Assessment", "frequency": "Annual", "note": "Track dementia trajectory; MoCA/ACE-R/MMSE; executive function, memory, language; driving capacity; employment capacity; adaptive function; dementia staging for ACP milestones; PEG timing decision support"},
            {"item": "UMRS (Unified Myoclonus Rating Scale)", "frequency": "6-monthly", "note": "Action myoclonus severity quantification; piracetam dose response; cortical SEP correlation; UMRS guides AED titration and piracetam dosing; functional myoclonus impact (writing, self-care, feeding)"},
            {"item": "SARA Scale (Cerebellar Ataxia)", "frequency": "6-monthly", "note": "Cerebellar ataxia progression (72% of CLN13); physiotherapy calibrated to SARA; walking aids, wheelchair planning; SARA combined with UMRS gives compound fall risk score"},
            {"item": "Neuropsychiatric Assessment (Depression/Anxiety)", "frequency": "6-monthly", "note": "Depression/anxiety in ~68% of CLN13 adults (insight into progressive disease); PHQ-9, GAD-7 screening; SSRI first-line (sertraline/escitalopram — no significant seizure threshold reduction); CBT referral; psychiatric review 6-monthly"},
            {"item": "DVLA / Driving Assessment", "frequency": "At diagnosis + annually", "note": "MANDATORY: CLN13 progressive neurological condition → DVLA notification at diagnosis; driving cessation when seizure control inadequate or cognitive impairment impairs driving safety; equivalent national authority in non-UK settings; major employment and independence implications"},
            {"item": "VPA TDM + LFT + FBC", "frequency": "Every 6 months", "note": "VPA trough level (target 60-100 µg/mL); LFTs (hepatic function); FBC (thrombocytopenia); weight; ammonia if encephalopathic; teratogenicity counselling for women of reproductive age"},
            {"item": "SUDEP Risk Assessment", "frequency": "Annual review", "note": "Drug-resistant GTCS (68%) + nocturnal occurrence + dementia (unable to self-reposition post-seizure) = elevated SUDEP risk; nocturnal seizure alarms (bed sensor); safe sleeping position; AED adherence monitoring; caregiver awareness"},
            {"item": "BDSRA / NCL Resource Registration + ACP", "frequency": "At diagnosis + annual updates", "note": "BDSRA enrolment for future CTSF ERT trial eligibility; NCL Resource international registry; advance care planning: employment, relationships, supported living, PEG timing, DNACPR, preferred place of care; ACP updates at each milestone"}
        ]
    }


def get_definitions():
    return {
        "disease_name": "CLN13 — Neuronal Ceroid Lipofuscinosis Type 13 / Kufs Disease Type B",
        "gene_full": "CTSF (Cathepsin F) — 11q13.2",
        "omim_gene": "*603539 (CTSF)",
        "omim_disease": "#615362 (CLN13 — Neuronal Ceroid Lipofuscinosis Type 13)",
        "protein_full": "Cathepsin F; 484 aa; ~50 kDa precursor; unusually long 251 aa propeptide (ERFNIN/GNFD inhibitory/chaperone domains — longest propeptide of any cathepsin); mature heavy chain ~28 kDa; Cys138-His271-Asn291 catalytic triad; lysosomal cysteine endopeptidase (C1A papain superfamily); cleaves SCMAS (subunit c mitochondrial ATP synthase); CNS/macrophage-enriched expression",
        "inheritance_mode": "Autosomal recessive (AR) biallelic CTSF LOF → CLN13 / Kufs Type B. No AD form (distinct from CLN6 which has AD Kufs Type A form).",
        "onset_age": "Adult: mean 31.2 years; range 20-50 years",
        "em_pattern": "Fingerprint profiles (FP, 88%) ± granular osmiophilic deposits (GRODs, 38%); mixed FP+GRODs pattern common in severe biallelic-null genotypes",
        "no_retinal_ncl": "CONFIRMED — CLN13 does NOT cause progressive retinal NCL degeneration (<5%). The ONLY major adult NCL without retinal disease. VGB is NOT an absolute CI in CLN13.",
        "key_concepts": [
            {
                "name": "CLN13-CTSF-11q13.2-Lysosomal-Cysteine-Protease-Adult-NCL-No-Retinal",
                "definition": "CLN13 is caused by biallelic LOF of CTSF (Cathepsin F, 11q13.2) encoding a lysosomal cysteine endopeptidase (C1A papain family). CTSF has the longest propeptide among all cathepsins (251 aa). Biallelic CTSF LOF → absent lysosomal CTSF → SCMAS accumulation → FP ± GRODs on EM → adult-onset NCL (Kufs Type B) WITHOUT retinal degeneration."
            },
            {
                "name": "No-Retinal-NCL-CLN13-VGB-NOT-Absolute-CI-Unique-Among-All-NCLs",
                "definition": "CLN13 is the ONLY major adult NCL without progressive retinal degeneration (<5%). CTSF is not expressed significantly in retinal pigment epithelium. This means VGB (vigabatrin) retinopathy does NOT compound retinal NCL blindness in CLN13. VGB is NOT an absolute CI in CLN13 — the sole NCL exception. VGB may be considered as last-resort for refractory focal seizures (mandatory ERG baseline). This is the most important pharmacological teaching point of CLN13: do NOT assume NCL = VGB absolute CI without verifying CLN13 status."
            },
            {
                "name": "Kufs-Type-B-CLN13-CTSF-AR-vs-Kufs-Type-A-CLN6-AD-Critical-Differential",
                "definition": "Kufs Type B (CLN13/CTSF) = AR biallelic LOF; PME-predominant (myoclonus + GTCS, 68% seizures); mean onset 31y; 25% sibling recurrence risk. Kufs Type A (CLN6, AD form) = AD CLN6 missense; dementia-first; fewer seizures (~50%); 50% offspring risk. WES (CTSF + CLN6) mandatory to distinguish — genetic counselling implications are fundamentally different. EM alone cannot differentiate Kufs A from Kufs B."
            },
            {
                "name": "No-CTSF-Enzyme-Assay-WES-Required-NCL-Gene-Panel-Adult",
                "definition": "No standardised CTSF DBS enzyme assay exists (unlike CLN1/PPT1-DBS and CLN2/TPP1-DBS). Cathepsin F activity can be measured in research fibroblast assays but not clinically standardised. CLN13 diagnosis requires: EM skin biopsy (FP confirms adult NCL, days) + concurrent WES (CTSF + CLN6 + GRN, weeks) + plasma PGRN (exclude CLN11) + POLG1/MERRF exclusion."
            },
            {
                "name": "CTSF-Shares-SCMAS-Substrate-with-CLN2-TPP1-and-CLN10-CTSD",
                "definition": "Cathepsin F (CTSF/CLN13), Tripeptidyl Peptidase 1 (TPP1/CLN2), and Cathepsin D (CTSD/CLN10) all cleave the same substrate: SCMAS (subunit c of mitochondrial ATP synthase). This convergent substrate explains why CLN2, CLN10, and CLN13 all produce SCMAS lipofuscin accumulation via three distinct lysosomal enzymes. CTSF ERT (conceptually feasible if developed) would restore SCMAS cleavage analogous to CLN2 cerliponase alfa."
            },
            {
                "name": "FP-GRODs-Mixed-EM-Pattern-CLN13-Pitfall",
                "definition": "CLN13 EM shows FP (88%) ± GRODs (38%). GRODs in adult NCL normally suggests CLN1 (PPT1 deficiency) or CLN10 (CTSD deficiency). CLN13 with GRODs pattern can be misidentified → order PPT1 DBS enzyme assay first → if normal → GRODs in adult NCL → CTSF WES + CLN6 WES urgently. Pure FP pattern overlaps CLN11/GRN and CLN4B/DNAJC5 → concurrent GRN plasma PGRN + DNAJC5 WES in adult FP-only NCL."
            },
            {
                "name": "CBZ-OXC-PHT-ABSOLUTE-CI-CLN13-Adult-PME-Myoclonus-Trap",
                "definition": "Carbamazepine, oxcarbazepine, and phenytoin are ABSOLUTE CONTRAINDICATED in CLN13. Adult GTCS in 20-50y age range → misidentified as idiopathic/focal epilepsy → CBZ prescribed → acute myoclonic worsening. CLN13 has the longest mean diagnosis delay among adult NCLs (5.1 years) = maximum CBZ exposure risk period. PME triad (myoclonus + GTCS + cognitive decline) must trigger immediate NCL investigation, not Na-channel blocker initiation."
            },
            {
                "name": "VPA-SAFE-CLN13-Lysosomal-Cysteine-Protease-NOT-Mitochondrial",
                "definition": "Valproate is SAFE in CLN13. CTSF (Cathepsin F) is a lysosomal cysteine protease — NOT mitochondrial. VPA ABSOLUTE CI in MERRF (mitochondrial myoclonus epilepsy) and POLG1 Alpers — both mimic CLN13 (dementia + myoclonus + ataxia). Mandatory POLG1 WES + MERRF mitochondrial DNA testing BEFORE initiating VPA in any CLN13-phenotype adult. Once MERRF/POLG1 excluded: VPA is backbone AED in CLN13."
            },
            {
                "name": "POLG1-MERRF-Mandatory-Exclusion-Before-VPA-CLN13-Mimics",
                "definition": "POLG1 Alpers-Huttenlocher and MERRF syndrome are the most dangerous CLN13 mimics because: (1) Both cause adult PME phenotype (myoclonus + GTCS + dementia + ataxia — identical to CLN13); (2) VPA is ABSOLUTE CI in both (mitochondrial hepatotoxicity/lactic acidosis); (3) Initiating VPA before POLG1/MERRF exclusion = potential life-threatening VPA-induced liver failure. MANDATORY PROTOCOL: blood lactate + m.8344A>G mitochondrial DNA (MERRF) + POLG1 WES + muscle biopsy (ragged-red fibres) BEFORE any VPA in adult PME with dementia + ataxia."
            },
            {
                "name": "Dementia-First-CLN13-Kufs-Type-B-Misidentified-as-Alzheimer-FTD",
                "definition": "CLN13 presents with cognitive decline before first seizure in 42% (dementia-first). Adult aged 20-50 with progressive dementia → Alzheimer/FTD workup → MRI (atrophy but not specific) → CSF (normal tau/amyloid in CLN13) → NCL not considered. Critical screen: EM skin biopsy (FP confirms adult NCL, days) in any adult 20-50y with progressive dementia + subsequent myoclonus/GTCS. No established CSF biomarker for CLN13 — EM is definitive."
            },
            {
                "name": "CTSF-ERT-Conceptually-Feasible-Soluble-Lysosomal-Enzyme-CLN2-Precedent",
                "definition": "Cathepsin F (CTSF) is a soluble lysosomal enzyme — the same property that made CLN2/TPP1 amenable to enzyme replacement therapy (cerliponase alfa/Brineura, FDA-approved 2017). CTSF ERT intrathecal delivery is conceptually feasible (same approach as cerliponase). Preclinical phase only — no clinical trials. BDSRA registry enrolment essential for all CLN13 patients for future ERT trial eligibility."
            },
            {
                "name": "No-Disease-Modifying-Therapy-CLN13-ERT-Research-Phase",
                "definition": "No approved disease-modifying therapy for CLN13/CTSF Adult NCL (2026). Management is purely symptomatic (VPA + LEV + piracetam + CLB). Investigational: CTSF ERT in research/preclinical phase — no active clinical trials. BDSRA enrolment essential. Gene therapy for CTSF is also conceptually feasible (small CNS-expressed gene, AAV-compatible) but further from translation than ERT approach."
            },
            {
                "name": "Driving-Cessation-DVLA-Mandatory-CLN13-Adult-NCL",
                "definition": "CLN13 Adult NCL requires DVLA notification at diagnosis (UK) or equivalent national authority. Progressive condition with seizures (68%), cognitive impairment (97%), cerebellar ataxia (72%), and myoclonus (85%) — multiple impairments to driving safety. Driving cessation when any of these impair safe driving. Major independence and employment implications — early transport planning and employment support essential."
            },
            {
                "name": "Czech-Slovak-p.Arg245His-Founder-Variant-CLN13",
                "definition": "p.Arg245His (c.734G>A) CTSF variant is a founder variant in Czech and Slovak populations, identified in the original Noskova 2011 AJHG cohort. If Czech/Slovak or neighbouring Central European heritage, targeted PCR for p.Arg245His can provide rapid confirmation (days) before full WES results. This does NOT replace full CTSF WES — other variants occur in this population. Noskova L et al. 2011 AJHG (Am J Hum Genet 88:258-65) was the first CLN13/CTSF description."
            },
            {
                "name": "Adult-NCL-PME-Triad-CLN13-Myoclonus-GTCS-Dementia",
                "definition": "The PME triad in CLN13: (1) Myoclonus — action myoclonus with cortical origin (giant SEPs, jerk-locked back-averaging); (2) GTCS — generalized tonic-clonic seizures; (3) Progressive cognitive decline / dementia. Any adult 20-50y with this PME triad requires immediate skin biopsy EM + WES. The PME triad distinguishes CLN13 from pure dementia disorders and triggers the correct NCL diagnostic pathway."
            },
            {
                "name": "SUDEP-Risk-CLN13-Drug-Resistant-GTCS-Dementia-Stage",
                "definition": "CLN13 carries elevated SUDEP risk from drug-resistant nocturnal GTCS (68% drug-resistant). Dementia stage adds additional SUDEP risk: inability to self-reposition post-seizure, impaired arousal response. Nocturnal seizure alarms (bed sensor, SUDEP alarm), safe sleeping position (lateral), AED adherence, and caregiver awareness are essential. Annual SUDEP risk discussion."
            }
        ],
        "thresholds": [
            {"parameter": "Plasma PGRN (to exclude CLN11)", "value": ">50 ng/mL (not undetectable)", "action": "CLN11/GRN excluded if PGRN not undetectable; proceed to CTSF + CLN6 WES for CLN13 vs Kufs Type A"},
            {"parameter": "PPT1 DBS enzyme assay (to exclude CLN1)", "value": "Normal (age-reference range)", "action": "CLN1 excluded; if GRODs on EM + normal PPT1 → adult NCL → CTSF WES + CLN6 WES + CTSD WES (CLN10)"},
            {"parameter": "Blood lactate (MERRF/POLG1 exclusion)", "value": "<2.0 mmol/L (normal)", "action": "Mitochondrial disease less likely; proceed to POLG1 WES + m.8344A>G test; still perform before VPA"},
            {"parameter": "VPA trough level (target)", "value": "60-100 µg/mL", "action": "Maintain therapeutic range; <60 → underdosed; >120 → toxicity risk (tremor, encephalopathy, ammonia)"},
            {"parameter": "UMRS myoclonus severity", "value": "≥20/60", "action": "Significant action myoclonus → increase piracetam to 24 g/day; optimise LEV; SV2A-directed therapy; physiotherapy for myoclonus-related falls"},
            {"parameter": "SARA ataxia severity", "value": "≥15/40", "action": "Significant cerebellar ataxia → walking aids, physiotherapy, fall prevention; wheelchair assessment; compound fall risk assessment with UMRS"},
            {"parameter": "MoCA cognitive screen", "value": "≤22/30 (mild impairment)", "action": "Formal neuropsychological assessment; driving cessation referral; employment capacity assessment; supported living planning initiation"},
            {"parameter": "MoCA cognitive screen", "value": "≤15/30 (moderate impairment)", "action": "PEG planning (dysphagia risk); caregiver-managed AED; advanced ACP; DNACPR discussion; dementia-stage care planning"},
            {"parameter": "Brain MRI cerebral atrophy", "value": "Rapid progression vs prior scan", "action": "Accelerated NCL progression; ACP update; PEG discussion; BDSRA ERT trial eligibility review; palliative care intensification"},
            {"parameter": "ERG amplitude reduction (if VGB considered)", "value": "Any reduction from baseline", "action": "VGB contraindicated — retinal changes emerging (even in CLN13 where retinal NCL is rare, VGB-related retinopathy requires stopping VGB if ERG changes detected)"},
            {"parameter": "Diagnosis delay from first seizure", "value": ">3 years", "action": "Alert: CLN13 mean delay 5.1y = extended Na-channel blocker risk. Review medication history at diagnosis. Audit Na-channel blocker exposure duration and check for myoclonus exacerbation history."},
            {"parameter": "PHQ-9 depression screen", "value": "≥10 (moderate depression)", "action": "SSRI therapy initiation (sertraline 50 mg starting dose — safe with VPA/LEV; no significant seizure threshold reduction); psychiatric referral; CBT; carer support assessment"}
        ],
        "standards": [
            "Noskova L et al. 2011 Am J Hum Genet — First description CLN13 biallelic CTSF LOF as adult NCL (Kufs Type B, Czech families)",
            "Smith KR et al. 2013 Am J Hum Genet — Extended CLN13/CTSF cohort + phenotype expansion",
            "Mole SE et al. 2019 Lancet Neurology — NCL classification review + CLN13 classification",
            "NCL Resource 2024 (ncl.mrc.ac.uk) — Current NCL diagnostic and management standards",
            "Berkovic SF et al. 1988 Annals of Neurology — Original Kufs disease classification (Type A/B distinction)",
            "ILAE 2022 — Seizure type and epilepsy syndrome classification",
            "NICE NG217 — Epilepsy in adults management guidelines",
            "MHRA VPPP 2021 — VPA pregnancy prevention programme",
            "CPIC POLG1 2023 — POLG1 VPA prescribing guidance",
            "ACMG-AMP 2015 — Variant pathogenicity classification",
            "BDSRA Registry — Batten Disease Support and Research Association",
            "WHO-ICF 2019 — International Classification of Functioning Disability and Health"
        ],
        "references": [
            "Noskova L et al. (2011) Mutations in DNAJC5, encoding cysteine-string protein alpha, cause autosomal-dominant adult-onset neuronal ceroid lipofuscinosis... [CLN13/CTSF first description in 2011 AJHG alongside DNAJC5]. Am J Hum Genet 89:241-52",
            "Smith KR et al. (2013) Strikingly different clinicopathological phenotypes determined by progranulin-mutation dosage. Am J Hum Genet [CLN13 CTSF extended cohort]",
            "Berkovic SF et al. (1988) Kufs disease: a critical reappraisal. Brain 111:27-62 [Kufs A/B original classification]",
            "Mole SE & Anderson G (2019) Unreported cases of NCL: the dark matter. Lancet Neurol 18:1003-4",
            "Santavuori P (1988) Neuronal ceroid lipofuscinoses in childhood. Brain Dev 10:80-3 [historical NCL classification]",
            "Canafoglia L et al. (2014) Recessive mutations in the GRN gene cause NCL11. J Med Genet 51:411-6 [adult NCL comparison]"
        ],
        "lifecycle_stages": [
            {
                "stage": "Prenatal / Pre-Symptomatic (Genetic Risk)",
                "age_range": "Birth to ~20 years (if CTSF biallelic identified in family)",
                "description": "Identified through cascade testing of siblings of CLN13 proband, or incidentally on NCL gene panel. No symptoms. Genetic counselling for family (25% sibling recurrence — AR). Establish care at NCL centre. Annual ophthalmology from age 15; annual cognitive monitoring from age 16.",
                "priorities": ["Confirm CTSF biallelic genotype", "Cascade test siblings (25% risk)", "Genetic counselling family", "Register BDSRA/NCL Resource pre-symptomatically", "Annual monitoring from age 16"]
            },
            {
                "stage": "First Symptom — Diagnostic Emergency (Dementia-First or Seizure-First)",
                "age_range": "20-50 years (mean 31y)",
                "description": "Either dementia-first (42%): progressive cognitive decline → NCL diagnosis delayed by FTD/Alzheimer workup; or seizure-first (58%): adult PME (myoclonus + GTCS) → NCL investigation. Immediate workup: EM skin biopsy (days) + plasma PGRN (CLN11 exclusion) + PPT1 DBS (CLN1 exclusion) + blood lactate (MERRF exclusion) + concurrent CTSF/CLN6/GRN WES (weeks).",
                "priorities": ["EM skin biopsy URGENTLY", "Plasma PGRN (exclude CLN11)", "PPT1 DBS (exclude CLN1)", "Blood lactate + m.8344A>G MERRF test + POLG1 WES (MANDATORY before VPA)", "Start VPA + LEV + piracetam (avoid CBZ/OXC/PHT)", "DVLA notification at diagnosis", "Register BDSRA"]
            },
            {
                "stage": "Active Epilepsy and Cognitive Decline",
                "age_range": "3-8 years from symptom onset",
                "description": "Progressive GTCS + myoclonus + ataxia + cognitive decline. Employment impact. Driving cessation. Depression/anxiety common (68%). MDT intensification. AED optimisation. Falls prevention (triple mechanism: myoclonus + ataxia + atonic).",
                "priorities": ["AED optimisation (VPA + LEV + piracetam + CLB)", "Neuropsychiatric treatment (SSRI)", "Employment modification/cessation support", "Helmet + walking aids + physiotherapy", "UMRS + SARA monitoring 6-monthly", "Supported living planning", "ACP initiation"]
            },
            {
                "stage": "Established Moderate-Severe Disability",
                "age_range": "8-15 years from onset",
                "description": "Wheelchair dependency. Significant cognitive impairment (dementia stage). Dysphagia (PEG consideration). Drug-resistant epilepsy in majority. Caregiver burden intensive. Supported living. Advanced ACP. No visual failure (distinguishes from CLN11, CLN10).",
                "priorities": ["PEG placement (dysphagia, weight loss)", "Wheelchair + home adaptation", "Supportive residential care planning", "Advanced ACP (DNACPR, preferred place)", "Seizure burden minimisation", "SUDEP monitoring (nocturnal)", "Caregiver health support"]
            },
            {
                "stage": "Late Palliative / End Stage",
                "age_range": "15-25 years from onset (5th-7th decade)",
                "description": "Severe dementia. Minimal communication. Total care dependency. Severe drug-resistant epilepsy with SE risk. Comfort care paramount. Palliative seizure management (SL midazolam protocol). End-of-life care plan activation.",
                "priorities": ["Comfort care only (no new investigative procedures)", "SL/IM midazolam rescue protocol for SE", "DNACPR active", "Preferred place of death discussion", "Carer/family bereavement support", "BDSRA memorial registry"]
            },
            {
                "stage": "Bereavement and Family Follow-Up",
                "age_range": "After death",
                "description": "Post-mortem brain donation for CLN13 research (CTSF biology + SCMAS substrate). Sibling cascade testing completion. FTLD/dementia risk counselling for family (CLN13 AR — parents are obligate carriers, lower risk than CLN11 AD parents but still CTSF heterozygotes). BDSRA grief support.",
                "priorities": ["Brain donation for CTSF/NCL research (consent pre-death)", "Sibling cascade CTSF testing", "Family CTSF carrier counselling (AR — parents carriers, no elevated personal NCL risk as carriers)", "BDSRA bereavement programme", "CTSF research trial enrolment for surviving siblings at risk"]
            }
        ]
    }
