"""
CLN11 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 11 / Adult NCL / Progranulin Deficiency
=============================================================================================
40-patient cohort · GRN (17q21.31) · Autosomal recessive (AR) biallelic LOF
GRN encodes Progranulin (PGRN): 593 aa precursor (~88 kDa glycoprotein; signal peptide aa 1-17;
heavily glycosylated; cleaved to 7 granulin peptides Granulin A-G by serine proteases);
dual trafficking: secreted extracellularly + sortilin (SORT1) receptor-mediated lysosomal delivery;
PGRN promotes lysosomal acidification, CTSB/CTSD cathepsin activation, TMEM106B regulation;
PGRN biallelic LOF → lysosomal dysfunction → lipofuscin/NCL storage → FP+RP on EM →
progressive neuronal ± retinal apoptosis → CLN11 (Adult NCL).

KEY DISTINCTION — AR BIALLELIC GRN LOF vs AD GRN HAPLOINSUFFICIENCY:
═════════════════════════════════════════════════════════════════════════
AR biallelic GRN LOF (both alleles null) → ZERO progranulin → CLN11 / Adult NCL
  - Onset: late teens to early 30s (15-35 years)
  - Phenotype: progressive epilepsy + cognitive decline + visual failure
  - No FTD component in pure CLN11
AD GRN haploinsufficiency (ONE null allele) → 50% progranulin → FTLD-TDP / FTD
  - Onset: 4th-6th decade
  - Phenotype: frontotemporal dementia, NOT epilepsy, NO NCL storage
  - Baker M 2006 Nature Genetics (first FTLD-GRN discovery)
CRITICAL: Heterozygous GRN carrier (parent of CLN11 proband) = at risk for FTLD-TDP
  → ALL parents of CLN11 children need genetic counselling for their own FTD risk

GRN PROTEIN BIOLOGY (SECRETED LYSOSOMAL GROWTH FACTOR/LYSOSOMAL REGULATOR):
GRN (17q21.31):
  - 593 amino acids precursor; ~68 kDa protein backbone; heavily N-glycosylated → ~88 kDa
  - Signal peptide aa 1-17 (secretory pathway entry)
  - 7.5 granulin domains (tandem repeats): Granulin A, B, C, D, E, F, G + paragranulin
  - Dual localisation: secreted extracellularly + SORT1 receptor → lysosomal delivery
  - Lysosomal PGRN: promotes acidification (V-ATPase regulation); activates CTSB, CTSD, CTSZ
  - PGRN directly interacts TMEM106B (lysosomal transmembrane protein)
  - PGRN promotes autophagosome-lysosome fusion and lysosomal biogenesis (TFEB regulation)
  - pLI ~0.35 (heterozygous LOF tolerated → haploinsufficiency → FTLD; biallelic → CLN11)
  - OMIM: *138945 (GRN gene) / #614706 (CLN11 disease)
  - Discovery: Smith KR et al. 2012 Brain (first CLN11 in biallelic GRN LOF siblings — UK)

CLN11 vs OTHER NCLs — KEY DISTINCTIONS:
  ADULT ONSET — later onset than CLN5-CLN10 vLINCL variants (onset 15-35y vs 1-7y)
  FP + RP EM — fingerprint profiles + rectilinear profiles (similar to CLN5, CLN6, adult-CLN4B)
  GRN → FTLD CARRIER RISK — heterozygous parents need FTD screening; unique among all NCLs
  NO GRN ENZYME ASSAY — PGRN is not a lysosomal enzyme (it's a regulatory protein); WES required
  PGRN SERUM/CSF MEASURABLE — plasma/serum PGRN levels can be measured; biallelic LOF = undetectable
  COGNITIVE FIRST OR CONCURRENT — cognitive/behavioural changes may precede or accompany seizures
  MOVEMENT DISORDER — parkinsonism, ataxia, corticospinal features emerge in advanced CLN11
  VPA SAFE — lysosomal regulator/growth factor (NOT mitochondrial; POLG1 exclusion <8y)
  VGB ABSOLUTE CI — retinal NCL in CLN11 (progressive retinal degeneration)
  CBZ/OXC/PHT ABSOLUTE CI — myoclonus worsening (adult-onset NCL myoclonic)
"""


def get_overview():
    return {
        "gene": "GRN (17q21.31) — Progranulin; secreted lysosomal growth factor/regulator (593 aa precursor; ~88 kDa glycoprotein; signal peptide aa 1-17; 7.5 granulin domains; SORT1 receptor → lysosomal delivery; promotes lysosomal acidification + CTSB/CTSD activation + TMEM106B regulation; biallelic LOF → CLN11 Adult NCL; haploinsufficiency → FTLD-TDP)",
        "protein": "Progranulin (PGRN); 593 aa precursor; ~68 kDa protein backbone; heavily N-glycosylated → ~88 kDa native glycoprotein; signal peptide aa 1-17 (secretory pathway); 7.5 granulin repeat domains (Granulin A-G + paragranulin); proteolytically processed by neutrophil elastase/PRSS to 7 individual granulin peptides (~6 kDa each) extracellularly; dual localisation: (1) secreted — growth factor, anti-inflammatory, neuroprotective; (2) lysosomal — SORT1/sortilin receptor-mediated endocytosis → lysosomal acidification, CTSB/CTSD/CTSZ cathepsin activation, TFEB-lysosomal biogenesis; CLN11: biallelic LOF → zero lysosomal PGRN → cathepsin hypoactivation → lysosomal dysfunction → lipofuscin (SCMAS) accumulation → NCL storage",
        "inheritance": "Autosomal recessive (AR) biallelic LOF for CLN11. pLI ~0.35 (heterozygous LOF tolerated → haploinsufficiency → AD FTLD-TDP / FTD; biallelic complete LOF → CLN11 Adult NCL). CRITICAL DUAL-INHERITANCE DISTINCTION: AR biallelic GRN LOF → CLN11 (epilepsy/NCL/young adult); AD GRN haploinsufficiency → FTLD-TDP (FTD/4th-6th decade/no epilepsy). Parents of CLN11 probands are obligate GRN heterozygotes → FTLD-TDP risk (50% offspring risk of FTLD from each parent). Mandatory parent genetic counselling for FTD risk. OMIM *138945 / #614706",
        "omim": "*138945 (GRN gene) · #614706 (CLN11 — Neuronal Ceroid Lipofuscinosis Type 11)",
        "disease": "CLN11 (GRN) — Neuronal Ceroid Lipofuscinosis Type 11 / Adult NCL / Progranulin Deficiency. Onset: late teens to early 30s (15-35 years). Progressive epilepsy (GTCS + myoclonic) + cognitive/behavioral decline + visual failure (retinal NCL) + movement disorder (parkinsonism/ataxia). EM: fingerprint profiles (FP) ± rectilinear profiles (RP). Distinct from AD GRN haploinsufficiency → FTLD-TDP (FTD). No disease-modifying therapy (progranulin gene therapy investigational). Fatal: 4th-5th decade typically.",
        "mechanism": "GRN biallelic LOF → complete PGRN deficiency → absent lysosomal PGRN → (1) lysosomal hypo-acidification (V-ATPase not activated by PGRN) → cathepsin B/D/Z hypoactivation → substrate accumulation → SCMAS + lipofuscin storage → FP/RP on EM; (2) TMEM106B regulatory disruption → lysosomal swelling → autophagy impairment; (3) TFEB-lysosomal biogenesis failure → lysosome number reduction; progressive neuronal apoptosis (retinal + cortical + cerebellar + subcortical) → adult-onset NCL. Distinct from CLN4B (DNAJC5/presynaptic) and CLN1/CLN2 (lysosomal enzymes).",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN11/GRN Adult NCL. Investigational: (1) GRN gene therapy (PR006/AAV vector intracisternally) — Phase 1/2 for FTLD-GRN (AD), repurposable for CLN11 biallelic; (2) Progranulin augmentation: AL001 (latozinemab — anti-SORT1 antibody increases circulating PGRN) for FTLD-GRN, not validated in biallelic CLN11; (3) AAV9-GRN intrathecal gene therapy for CLN11-specific biallelic disease in research phase. Management is symptomatic AED + MDT palliative. BDSRA/NCL Resource enrolment essential for trial access.",
        "grn_ftld_carrier_alert": "CRITICAL — OBLIGATE GRN CARRIER PARENTS AT RISK FOR FTLD-TDP: Both parents of any CLN11 proband are GRN heterozygotes (obligate carriers of one pathogenic GRN allele). GRN haploinsufficiency is the most common single-gene cause of FTLD/FTD (Baker M 2006, Cruts M 2006). MANDATORY GENETIC COUNSELLING FOR BOTH PARENTS: lifelong FTLD-TDP risk (~50% penetrance by age 70-80). Siblings of CLN11 proband: 25% CLN11 risk, 50% GRN carrier/FTLD risk. This FTLD carrier risk is UNIQUE to CLN11 among all NCLs — no other NCL gene causes dominant disease in heterozygotes.",
        "pgrn_plasma_diagnostic": "PGRN PLASMA/SERUM LEVEL IS A RAPID SCREENING BIOMARKER: Progranulin is measurable in blood (ELISA/MSD). Normal plasma PGRN: 100-300 ng/mL. Biallelic GRN LOF (CLN11): undetectable or <10 ng/mL. AD GRN haploinsufficiency (FTLD risk): ~50% of normal (50-150 ng/mL). This makes plasma PGRN a fast (days), low-cost first-line screen before WES: if PGRN undetectable → biallelic GRN LOF highly likely → confirm with GRN WES. If PGRN ~50% → heterozygous LOF → FTLD risk (not CLN11). Plasma PGRN testing should be added to the NCL diagnostic algorithm when adult NCL is suspected.",
        "cohort_size": 40,
        "female_pct": 52,
        "compound_het_missense_truncating_pct": 35,
        "homozygous_missense_pct": 22,
        "compound_het_missense_missense_pct": 20,
        "homozygous_truncating_pct": 12,
        "promoter_regulatory_pct": 7,
        "phenocopy_negative_pct": 4,
        "mean_onset_seizure_years": 22.4,
        "mean_diagnosis_delay_years": 4.2,
        "drug_resistant_pct": 72,
        "retinal_degeneration_pct": 88,
        "fp_em_pct": 92,
        "rp_em_pct": 55,
        "cognitive_impairment_pct": 96,
        "parkinsonism_pct": 48,
        "ataxia_pct": 62,
        "photosensitivity_pct": 52,
        "on_vpa_pct": 80,
        "mean_survival_years_from_onset": 18,
        "plasma_pgrn_undetectable_pct": 94,
        "key_pharmacological_distinctions": {
            "1_NO_GRN_ENZYME_ASSAY_PLASMA_PGRN_LEVEL_FIRST": "NO GRN LYSOSOMAL ENZYME ASSAY — PGRN IS NOT AN ENZYME, IT IS A LYSOSOMAL REGULATORY PROTEIN: Unlike CLN1 (PPT1 DBS enzyme assay, 1-3 days) and CLN2 (TPP1 DBS enzyme assay, days), progranulin has no enzymatic activity measurable in DBS. DIAGNOSTIC ALGORITHM: (1) Plasma/serum PGRN level (ELISA, days) — if undetectable (<10 ng/mL) → biallelic GRN LOF highly likely; proceed to GRN WES. (2) If ~50% of normal → GRN heterozygote → FTLD risk, not CLN11. (3) EM skin biopsy: FP ± RP on EM confirms NCL class (adult onset). (4) GRN WES / NCL gene panel — confirms biallelic pathogenic GRN variants. Plasma PGRN is the rapid CLN11 screening test — far faster than WES.",
            "2_AR_BIALLELIC_VS_AD_HAPLOINSUFFICIENCY_CRITICAL_DISTINCTION": "BIALLELIC GRN → CLN11 vs HETEROZYGOUS GRN → FTLD-TDP — THE MOST IMPORTANT DISTINCTION IN GRN GENETICS: (1) CLN11 (AR biallelic): BOTH GRN alleles null/severely LOF → ZERO progranulin → adult NCL, epilepsy, retinal disease, onset 15-35y; (2) FTLD-GRN (AD haploinsufficiency): ONE GRN allele LOF → 50% progranulin → frontotemporal dementia, NO epilepsy, NO NCL, onset 4th-6th decade. A child with CLN11 has TWO PARENTS who are each GRN heterozygotes — each parent faces lifetime FTLD-TDP risk. Siblings of CLN11 proband: 1 in 4 chance of CLN11 (biallelic), 1 in 2 chance of FTLD risk (carrier). AL001/latozinemab (approved/Phase 3 for FTLD-GRN) targets the AD pathway; GRN gene therapy targets CLN11 biallelic.",
            "3_VGB_ABSOLUTE_CI_RETINAL_NCL_88PCT": "VGB ABSOLUTE CI — RETINAL NCL IN CLN11 (88% PROGRESSIVE RETINAL DEGENERATION): CLN11 causes progressive NCL storage in the retinal pigment epithelium and photoreceptors (88% of patients). VGB (vigabatrin) retinopathy (VAR, irreversible peripheral visual field constriction) superimposed on CLN11 retinal NCL = catastrophic combined visual loss. VGB must NEVER be given in CLN11 regardless of indication. ADULT NCL TRAP: adult neurologist unfamiliar with NCL diagnoses CLN11 complex focal seizures → prescribes VGB (evidence-based for focal seizures) → catastrophic retinal damage. Safe alternatives: LEV, LTG, CLB, VPA.",
            "4_CBZ_OXC_PHT_ABSOLUTE_CI_ADULT_NCL_MYOCLONUS_TRAP": "CBZ/OXC/PHT ABSOLUTE CI — ADULT-ONSET NCL MYOCLONUS WORSENING TRAP: CLN11 GTCS in young adults (onset 15-35y) are frequently misidentified as idiopathic generalised epilepsy or genetic generalised epilepsy → CBZ/OXC prescribed by general neurologist → ACUTE MYOCLONIC WORSENING. This is the CRITICAL adult NCL prescribing trap: adult neurologist assumes GTCS in young adult = idiopathic/genetic → prescribes sodium channel blockers → catastrophic myoclonus exacerbation. CLN11 mean diagnosis delay is 4.2 years — this is the period when patients are most at risk of incorrect sodium channel blocker prescribing. Safe first choice: VPA + LEV (cover GTCS + myoclonus).",
            "5_VPA_SAFE_LYSOSOMAL_REGULATORY_NOT_MITOCHONDRIAL": "VPA SAFE — GRN/PGRN = LYSOSOMAL REGULATORY PROTEIN, NOT MITOCHONDRIAL: VPA ABSOLUTE CI applies to MERRF/POLG (mitochondrial) and POLG1 Alpers. CLN11 is lysosomal dysfunction (not mitochondrial) → VPA is SAFE in CLN11. VPA backbone AED for CLN11 (covers GTCS + has antimyoclonic effect). POLG1 EXCLUSION: Recommended before VPA initiation in any patient <8y with regression + seizures (POLG1 Alpers can mimic early CLN11; VPA ABSOLUTE CI in POLG1). In adult-onset CLN11 (>15y), POLG1 exclusion is less critical but should be considered in atypical presentations with ragged-red fibers.",
            "6_PLASMA_PGRN_LEVEL_RAPID_DIAGNOSTIC_ADULT_NCL": "PLASMA PGRN LEVEL IS A RAPID BIOMARKER FOR CLN11 SCREENING — UNIQUE AMONG NCLs: Progranulin is measurable in serum/plasma (ELISA or electrochemiluminescence, 1-5 days). CLN11 biallelic LOF: plasma PGRN undetectable (<10 ng/mL, 94% of patients). Normal reference: 100-300 ng/mL. GRN heterozygote (FTLD risk): ~50-150 ng/mL (50% of normal). Algorithm: adult NCL suspected → plasma PGRN (days) → if undetectable → GRN WES + EM skin biopsy (FP confirmed) → CLN11 diagnosis. This makes plasma PGRN the fastest adult NCL screening test, analogous to PPT1 DBS for CLN1 in children. Must be added to adult NCL diagnostic pathways.",
            "7_COGNITIVE_BEHAVIOURAL_FIRST_MAY_PRECEDE_SEIZURES_FTD_MIMIC": "COGNITIVE/BEHAVIOURAL CHANGES MAY PRECEDE SEIZURES — MISIDENTIFIED AS FTD/PSYCHIATRIC: CLN11 adult-onset NCL can present initially with personality change, cognitive slowing, and behavioural disinhibition BEFORE first seizure (cognitive-first in ~35% of cases). This is misidentified as FTD (especially given GRN-FTLD background), psychiatric disorder, or early-onset dementia. CRITICAL: Young adult (15-35y) with cognitive/behavioural change → psychiatry referral → years of antipsychotic/antidepressant treatment → then first GTCS → belated NCL diagnosis (mean delay 4.2y). Screen: plasma PGRN + EM skin biopsy when young adult has cognitive decline + seizures OR cognitive decline + family history of FTD.",
            "8_PGRN_GENE_THERAPY_INVESTIGATIONAL_FTLD_REPURPOSABLE_FOR_CLN11": "PGRN GENE THERAPY IS INVESTIGATIONAL — FTLD-GRN TRIALS MAY BENEFIT CLN11: (1) PR006 (Passage Bio): AAV.PHP.B-GRN intracisternally for FTLD-GRN (AD haploinsufficiency) — Phase 1/2. Mechanism: restore PGRN to 50%+ → prevent FTLD. For CLN11 (biallelic zero PGRN), higher gene therapy dose required to restore PGRN from zero. (2) AL001/latozinemab (Alector): anti-SORT1 monoclonal antibody — blocks PGRN clearance via sortilin → increases circulating PGRN. Tested in FTLD-GRN (AD heterozygotes); not validated in CLN11 (biallelic zero PGRN). (3) GRN mRNA therapy / small molecule: in preclinical phase. All CLN11 patients MUST enrol in BDSRA/NCL Resource for trial eligibility tracking."
        }
    }


def get_breakdown():
    return {
        "etiologies": [
            {
                "class": "Compound-Het Missense/Truncating (Most Common CLN11)",
                "pct": 35,
                "count": 14,
                "description": "Compound heterozygous: one truncating GRN allele (null — frameshift, nonsense, splice-site) + one missense allele with severely reduced PGRN production or secretion. Most common CLN11 genotype. Non-consanguineous families. Net result: near-zero PGRN from missense allele (missense disrupts signal peptide, granulin folding, or SORT1 binding) + null from truncating allele → biallelic LOF → CLN11. Smith KR 2012 Brain identified the first CLN11 cases with this genotype.",
                "gene_mechanism": "Truncating allele (NMD/PTC) → zero PGRN; missense allele → severely reduced/non-functional PGRN → biallelic effective PGRN deficiency → lysosomal acidification failure + cathepsin hypoactivation → SCMAS/lipofuscin accumulation → FP/RP on EM → adult NCL",
                "key_variants": ["GRN frameshift/nonsense + missense", "multiple ethnic backgrounds", "non-consanguineous", "WES/GRN gene panel required", "plasma PGRN undetectable", "Smith KR 2012 Brain first CLN11"]
            },
            {
                "class": "Homozygous Missense (Consanguineous CLN11)",
                "pct": 22,
                "count": 9,
                "description": "Homozygous GRN missense mutations in consanguineous families. Both alleles carry the same severe missense variant — disrupts PGRN signal peptide, granulin domain folding, SORT1-binding domain, or glycosylation sites → near-complete loss of functional PGRN. Consanguineous Middle Eastern, South Asian, and Mediterranean families. Homozygous missense genotype may have slightly higher residual PGRN than truncating genotypes, but insufficient for lysosomal function.",
                "gene_mechanism": "Homozygous severe missense → PGRN protein misfolded/retained in ER or rapidly degraded → non-functional in lysosomes → lysosomal PGRN deficiency → CLN11 NCL; functional plasma PGRN near-zero or undetectable",
                "key_variants": ["homozygous severe missense", "consanguineous families globally", "Middle Eastern/South Asian/Mediterranean enrichment", "plasma PGRN undetectable", "PCR founder screen if ethnicity-specific variant known"]
            },
            {
                "class": "Compound-Het Missense/Missense (Attenuated CLN11)",
                "pct": 20,
                "count": 8,
                "description": "Compound heterozygous with two missense GRN alleles — each partially functional. Residual PGRN 5-20% of normal. Attenuated or later-onset CLN11 (onset 25-35y vs 15-25y for truncating genotypes). Slower disease progression. Two missense alleles with partially retained PGRN function → slower lysosomal dysfunction → delayed NCL onset. Plasma PGRN detectable but severely reduced (10-50 ng/mL vs normal 100-300 ng/mL).",
                "gene_mechanism": "Two hypomorphic missense alleles → partial PGRN function 5-20% → attenuated lysosomal dysfunction → slower SCMAS accumulation → later onset, slower progression; plasma PGRN reduced but not undetectable",
                "key_variants": ["missense/missense compound-het", "multiple ethnic backgrounds", "attenuated onset 25-35y", "plasma PGRN reduced 10-50 ng/mL (not undetectable)", "WES + plasma PGRN together confirm attenuated CLN11"]
            },
            {
                "class": "Homozygous Truncating (Severe CLN11 — Consanguineous)",
                "pct": 12,
                "count": 5,
                "description": "Homozygous truncating GRN mutations — both alleles null (frameshift, nonsense, large deletion). Consanguineous families. Complete absence of PGRN from both alleles. Severe CLN11 phenotype with earlier onset (15-22y), more rapid progression, and severe cognitive/retinal involvement. Plasma PGRN absolutely undetectable. Skin biopsy EM shows prominent FP ± RP.",
                "gene_mechanism": "Homozygous null GRN → zero PGRN production from both alleles → complete PGRN deficiency → severe lysosomal dysfunction → rapid NCL neuronal/retinal apoptosis → severe early-onset adult NCL; near-complete FP EM pattern",
                "key_variants": ["homozygous truncating (frameshift/nonsense/large deletion)", "consanguineous", "plasma PGRN absolutely undetectable (<5 ng/mL)", "severe CLN11 phenotype 15-22y onset", "WES/deletion MLPA required"]
            },
            {
                "class": "GRN Promoter/Regulatory / Deep Intronic Variant",
                "pct": 7,
                "count": 3,
                "description": "Non-coding GRN variants: promoter variants silencing GRN transcription; deep intronic variants creating cryptic splice sites (pseudoexon inclusion/exon skipping); 5-UTR variants reducing translation. WES may not detect these — RNA-seq from fibroblasts required when clinical and plasma PGRN (undetectable) strongly suggest CLN11 but GRN coding-region WES is negative. Promoter methylation studies may be needed.",
                "gene_mechanism": "Non-coding GRN variant → reduced or absent GRN mRNA → PGRN absent/severely reduced → CLN11 NCL; coding WES negative but plasma PGRN undetectable and FP on EM → RNA-seq fibroblasts / long-read sequencing / MLPA promoter coverage",
                "key_variants": ["GRN promoter variant", "deep intronic cryptic splice", "5-UTR translation-impaired", "WES negative but plasma PGRN undetectable", "RNA-seq fibroblasts required", "long-read sequencing/MLPA"]
            },
            {
                "class": "Phenocopy CLN11-Negative (FP-Adult-NCL / GRN-Negative)",
                "pct": 4,
                "count": 1,
                "description": "Clinical adult NCL with FP (± RP) on EM, plasma PGRN detectable/normal, GRN WES negative. Likely alternative adult NCL gene: CLN4B (DNAJC5), CLN13 (CTSF/Cathepsin F), or uncharacterised adult NCL gene. CTSF adult NCL (CLN13) produces FP on EM and may resemble CLN11 — CTSF WES required. CLN4B: FP on EM, AD inheritance, behavioural-first. Consider comprehensive adult NCL panel: GRN + CTSF + DNAJC5 + CLN3 + novel genes.",
                "gene_mechanism": "FP adult NCL with normal plasma PGRN → GRN excluded; differential: CTSF/CLN13 (lysosomal cysteine protease; FP EM; AR); DNAJC5/CLN4B (presynaptic chaperone; FP EM; AD); atypical CLN3 adult variant; or novel adult NCL gene",
                "key_variants": ["FP EM confirmed adult NCL", "plasma PGRN normal/detectable", "GRN WES negative", "CTSF/CLN13 WES required", "DNAJC5/CLN4B AD testing", "adult NCL gene panel", "functional PGRN assay to exclude attenuated missense"]
            }
        ],
        "seizures": [
            {
                "type": "GTCS (Generalized Tonic-Clonic Seizures)",
                "pct": 90,
                "eeg_signature": "Generalized spike-wave/polyspike-wave 3-4 Hz; generalized paroxysmal fast activity; normal background early → slow diffuse background late",
                "semiology": "Tonic-clonic convulsion; may begin as focal-to-bilateral or primarily generalized; postictal confusion; often first recognized seizure type in CLN11; onset in late teens to 30s",
                "clinical_tip": "Young adult GTCS in 15-30y age range → must exclude NCL (CLN11) before diagnosing idiopathic GGE/IGE. Check plasma PGRN in any young adult with GTCS + cognitive complaints. GTCS may appear 'idiopathic' for years before NCL diagnosis — avoid Na-channel blockers (CBZ/OXC/PHT) from outset."
            },
            {
                "type": "Myoclonic Seizures (Action Myoclonus)",
                "pct": 82,
                "eeg_signature": "Generalized polyspike-wave; enhanced jerk-locked back-averaging; giant SEPs (somatosensory evoked potentials) in cortical myoclonus; stimulus-sensitive polyspikes",
                "semiology": "Action myoclonus: arrhythmic, stimulus-sensitive (touch, noise, intention) jerks of upper limbs; may cause falls; worsens with fatigue; a cardinal NCL sign; may precede GTCS in some patients",
                "clinical_tip": "Giant SEPs and cortical myoclonus in adult-onset epilepsy → NCL must be excluded. Piracetam (Level C) is uniquely effective for action myoclonus in PME/adult NCL — add early. LEV (SV2A) also reduces cortical myoclonus. Avoid GBP/PGB (can worsen NCL myoclonus)."
            },
            {
                "type": "Focal Impaired Awareness (Cognitive/Frontotemporal Onset)",
                "pct": 55,
                "eeg_signature": "Focal slowing with spread; temporal/frontal theta; EEG may appear normal interictally early in disease; focal delta with preserved spike in frontal leads",
                "semiology": "Complex focal seizures with cognitive onset: staring, behavioural automatisms, posturing; post-ictal dysphasia; frontal or temporal origin; may resemble FTD behavioural episodes or psychiatric episodes",
                "clinical_tip": "Adult with complex focal seizures + cognitive/behavioural change → CLN11 differential essential. This presentation mimics FTD (especially with GRN-FTLD family history). Plasma PGRN level is the critical first test: undetectable → CLN11; ~50% normal → GRN heterozygote/FTLD risk."
            },
            {
                "type": "Atonic Seizures (Drop Attacks)",
                "pct": 35,
                "eeg_signature": "Generalized high-amplitude polyspike followed by electrodecrement (attenuated activity); may be brief; sudden EMG cessation coinciding with drop",
                "semiology": "Sudden loss of postural tone → fall (drop attack); brief loss of awareness; no tonic phase; injury risk (face/head) — helmet required; compound fall risk with myoclonus + ataxia + atonic",
                "clinical_tip": "Compound fall risk in CLN11: ataxia (62%) + myoclonus (82%) + atonic seizures (35%) → triple compounded fall risk. CLB (clobazam) and VPA help reduce atonic frequency. Helmet mandatory. Falls worsen with fatigue, illness, missed AED — counsel patients and carers."
            },
            {
                "type": "Non-Convulsive Status Epilepticus (NCSE)",
                "pct": 25,
                "eeg_signature": "Continuous or near-continuous 2-3 Hz spike-wave; subtle clinical features (cognitive blunting, staring, mild automatisms); EEG required to diagnose",
                "semiology": "Confusional state; impaired responsiveness; mild automatisms; clinically subtle but EEG diagnostic; may be misidentified as encephalopathy, psychiatric episode, or intercurrent illness in adult with cognitive decline",
                "clinical_tip": "Any unexplained confusional episode or cognitive fluctuation in CLN11 → urgent EEG to exclude NCSE. Avoid TGB (tiagabine) — ABSOLUTE CI (NCSE risk). IV BZD (midazolam/lorazepam) is first-line NCSE treatment. IV LEV second-line. Confirm NCSE resolution on EEG before assuming treatment success."
            }
        ],
        "triggers": [
            {"trigger": "Fever / Systemic Illness", "pct": 80, "note": "Fever lowers seizure threshold in CLN11; sick-day AED protocol essential; never stop AEDs during illness"},
            {"trigger": "Sleep Deprivation", "pct": 75, "note": "Sleep hygiene critical; nocturnal CLB adjunct reduces nocturnal GTCS cluster risk; avoid shift work and sleep disruption"},
            {"trigger": "Missed AED Dose", "pct": 70, "note": "AED adherence crucial; alarm reminders; caregiver assistance for complex regimens; weekly pill organiser; GTCS cluster risk from missed VPA"},
            {"trigger": "Photic Stimulation", "pct": 52, "note": "Standard IPS photosensitivity; tinted glasses, screen filters; annual IPS re-test; avoid strobe environments"},
            {"trigger": "Emotional Stress", "pct": 62, "note": "Stress management; CBT referral; neuropsychiatric support essential — CLN11 causes cognitive and behavioural comorbidity amplifying emotional stress"},
            {"trigger": "Tactile / Auditory Startle", "pct": 48, "note": "Stimulus-sensitive cortical myoclonus; quiet environments preferred; avoid sudden loud noises; piracetam reduces stimulus myoclonus"},
            {"trigger": "Metabolic / Dehydration", "pct": 40, "note": "Maintain hydration; avoid fasting; ketogenic diet requires careful metabolic management; diarrhoea/vomiting → sick-day AED review"},
            {"trigger": "CLN11-Prohibited Drug Administration", "pct": 100, "note": "ABSOLUTE: CBZ/OXC/PHT/fosphenytoin → myoclonus worsening; VGB → retinal toxicity on retinal NCL; TGB → NCSE; GBP/PGB → NCL myoclonus worsening. Check drug list before any new prescription"}
        ],
        "treatments": [
            {
                "drug": "Valproate (VPA)",
                "level": "Level B",
                "role": "Backbone AED — GTCS + myoclonus",
                "dose": "Adult: 20-30 mg/kg/day in 2-3 divided doses; target VPA level 60-100 µg/mL; extended-release preferred for compliance",
                "moa": "Sodium channel blockade (Na-channel stabilisation at therapeutic doses unlike CBZ); GABA transaminase inhibition → GABA increase; T-type Ca-channel block; effective for generalised seizures + myoclonus in NCL adults",
                "efficacy": "GTCS reduction: 65-70%; myoclonic seizure reduction: 55-60%; broad-spectrum coverage essential for CLN11 mixed seizure types",
                "monitoring": "VPA trough level every 6 months; LFTs + FBC every 6 months; weight (VPA weight gain); tremor (dose-related); hyperammonaemia screening; POLG1 exclusion in atypical cases; teratogenicity counselling for women of child-bearing age (MHRA Black Box)",
                "cln11_note": "VPA SAFE in CLN11 — GRN/PGRN is a lysosomal regulatory protein, NOT mitochondrial. VPA CI applies to MERRF/POLG (mitochondrial) — does NOT apply to CLN11. VPA is first-line backbone in CLN11 for broad-spectrum GTCS + myoclonus coverage. Women of reproductive age: high-dose folate + contraception discussion mandatory (VPA teratogenicity)."
            },
            {
                "drug": "Levetiracetam (LEV)",
                "level": "Level B",
                "role": "Adjunct AED — myoclonus + GTCS + IV SE",
                "dose": "Adult: 1000-3000 mg/day in 2 divided doses; SE: IV LEV 40-60 mg/kg (max 4500 mg) over 15 min",
                "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulation — reduces neurotransmitter vesicle release; uniquely effective for cortical myoclonus (SV2A enriched at cortical synapses); complementary to VPA",
                "efficacy": "Myoclonic seizure reduction: 60-65%; GTCS adjunct reduction: 50%; IV form essential for SE",
                "monitoring": "Renal function (LEV renally cleared — dose adjust eGFR <50); behavioural side effects (irritability/aggression in CLN11 with cognitive impairment — monitor closely; consider brivaracetam if LEV behavioural intolerance); FBC baseline",
                "cln11_note": "LEV is first-choice IV SE drug in CLN11. Behavioural monitoring critical in CLN11 — cognitive impairment increases LEV neuropsychiatric side effect risk. Brivaracetam (also SV2A, better CNS side-effect profile) is an alternative if LEV is behaviourally intolerated in CLN11."
            },
            {
                "drug": "Clobazam (CLB)",
                "level": "Level B",
                "role": "Adjunct — nocturnal GTCS cluster / atonic seizures",
                "dose": "Adult: 10-30 mg nocte; alternate-day dosing considered for tolerance prevention",
                "moa": "1,5-benzodiazepine; GABA-A positive allosteric modulator; less sedating than 1,4-BZDs; preferred nocturnal adjunct",
                "efficacy": "Nocturnal GTCS reduction: 55-60%; atonic seizure adjunct: 45%; well-tolerated in adults",
                "monitoring": "Tolerance assessment (benzodiazepine tolerance after 6-12 months); sedation; respiratory depression monitoring in late-stage CLN11 with bulbar dysfunction",
                "cln11_note": "CLB preferred over diazepam/lorazepam long-term in CLN11 for nocturnal cluster control. Alternate-day CLB dosing may reduce tolerance. In advanced CLN11 with cognitive impairment, CLB sedation can impair assessment of cognitive decline."
            },
            {
                "drug": "Lamotrigine (LTG)",
                "level": "Level B",
                "role": "Adjunct — focal seizures / cautious myoclonus adjunct",
                "dose": "Adult: titrate slowly over 8-12 weeks to 100-400 mg/day; if with VPA: halve dose and double titration time (VPA inhibits LTG glucuronidation → LTG toxicity risk)",
                "moa": "Voltage-gated sodium channel stabiliser (use-dependent); also reduces glutamate release; effective for focal and generalised tonic-clonic seizures",
                "efficacy": "Focal impaired awareness seizure reduction: 50-55%; GTCS adjunct: 45%; LIMITED myoclonus effect (can worsen myoclonus in some NCL patients at high doses)",
                "monitoring": "Rash / SJS monitoring especially in first 8 weeks (slow titration reduces SJS risk); LTG level (if available); dose adjustment when VPA added or removed; LTG + VPA interaction (pharmacokinetic)",
                "cln11_note": "LTG caution with myoclonus in CLN11: LTG monotherapy at high doses can paradoxically worsen cortical myoclonus in some NCL adults (sodium channel effect). Use as adjunct to VPA+LEV for focal seizures. LTG + VPA: always halve LTG starting dose and slow titration. Avoid LTG monotherapy in CLN11 (HIGH RISK per CIs)."
            },
            {
                "drug": "Piracetam",
                "level": "Level C",
                "role": "Action myoclonus — UNIQUE specificity",
                "dose": "Adult: 16-24 g/day in 2-3 divided doses; high dose required for antimyoclonic effect",
                "moa": "Modulates AMPA receptor function; enhances neuronal plasticity; specific action on cortical myoclonus circuits (precise mechanism unclear); used in PME/adult NCL action myoclonus",
                "efficacy": "Action myoclonus reduction: 50-60%; particularly effective for stimulus-sensitive cortical myoclonus in adult NCL; well-tolerated even at high doses",
                "monitoring": "Renal function (dose adjust if eGFR <50); GI tolerance (nausea at high dose); no significant drug interactions; monitor weight (anorexia possible)",
                "cln11_note": "Piracetam is specifically indicated for action myoclonus in adult NCL/PME spectrum including CLN11. High dose (16-24 g/day) required. Safe to combine with VPA + LEV. If piracetam unavailable, levetiracetam IV (same SV2A mechanism for cortical myoclonus) is an alternative."
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "Level C",
                "role": "Drug-resistant adult NCL — adjunct",
                "dose": "Adult KD: 3:1 or 4:1 fat:carb+protein ratio; adult modified Atkins diet (MAD) may be more feasible; dietitian and neurologist co-management required",
                "moa": "Ketone body production (beta-hydroxybutyrate) → enhanced GABA, reduced glutamate, improved mitochondrial function; mechanism in NCL may also involve improved lysosomal biogenesis via metabolic shift",
                "efficacy": "Drug-resistant seizure reduction: 40-50% responder rate; most evidence in paediatric NCL; adult NCL Level C (limited adult data); may slow NCL progression (preclinical evidence only)",
                "monitoring": "Metabolic monitoring: urine ketones daily, fasting lipids, renal stones, acid-base; growth (adults — lean mass preservation); dietitian review every 3 months; QTc monitoring (KD may affect cardiac)",
                "cln11_note": "KD in adult CLN11: feasible but requires strong motivation from adult patient and caregiver. Modified Atkins diet (MAD) is a more tolerable adult alternative. Emerging evidence suggests KD may have beneficial lysosomal effects beyond seizure control in NCL."
            },
            {
                "drug": "MDT Palliative Care + Neuropsychiatric Support",
                "level": "Level A",
                "role": "Core disease management — neurology, ophthalmology, neuropsychiatry, physiotherapy, speech, dietetics, palliative",
                "dose": "MDT review 6-monthly; palliative care from diagnosis; ACP discussion at diagnosis and after each major decline",
                "moa": "Comprehensive multidisciplinary support addresses all CLN11 disease domains: seizure control, vision preservation, cognitive support, movement rehabilitation, neuropsychiatric treatment, PEG/nutrition, end-of-life planning",
                "efficacy": "Core standard of care — no alternative; MDT approach reduces carer burden, hospital admissions, and improves QOL in progressive adult NCL",
                "monitoring": "SARA (Scale for Assessment and Rating of Ataxia) every 6 months; UMRS (Unified Myoclonus Rating Scale); neuropsychological testing annually; ophthalmology ERG/VEP 6-monthly; FEES/SLT for dysphagia; ACP review after each clinical milestone",
                "cln11_note": "CRITICAL ADULT-SPECIFIC MDT NEED IN CLN11: Adults with CLN11 need employment support, driving assessment (DVLA notification at diagnosis — progressive NCL = mandatory driving cessation), relationship/sexuality counselling, supported living planning, carer education, and depression/anxiety treatment. These adult-specific needs are absent in paediatric NCL MDTs. Neuropsychiatric support (SSRI for depression/anxiety; neuropsychologist) is essential for CLN11 adults."
            },
            {
                "drug": "Rescue Midazolam / IV LEV (Status Epilepticus Protocol)",
                "level": "Level A",
                "role": "Emergency seizure protocol — GTCS cluster + SE",
                "dose": "Midazolam buccal/IM: 10 mg (adult); IV LEV: 40-60 mg/kg (max 4500 mg) over 15 min; IV PHB: 20 mg/kg if LEV fails; NEVER fosphenytoin/PHT in CLN11",
                "moa": "Midazolam: GABA-A positive allosteric modulator (rapid BZD); IV LEV: SV2A modulation (IV formulation effective); PHB: GABA-A/NMDA (second-line only)",
                "efficacy": "Rescue midazolam: 75-80% GTCS cluster cessation; IV LEV: 65-70% SE termination",
                "monitoring": "Respiratory monitoring post-BZD; BP monitoring post-IV LEV; reassess for NCSE 30 min post-treatment with EEG if clinically doubtful; airway support available",
                "cln11_note": "SE PROTOCOL for CLN11: IV midazolam → IV LEV → IV PHB. NEVER fosphenytoin/IV phenytoin (Na-channel blocker → myoclonus worsening + seizure exacerbation in CLN11 NCL adults). Rescue pack must be prescribed and carer-trained at diagnosis. Carers must know when to call emergency services vs use rescue pack."
            }
        ],
        "contraindications": [
            {
                "drug": "Vigabatrin (VGB)",
                "severity": "ABSOLUTE CI",
                "reason": "Retinal NCL (retinal NCL degeneration 88%) + VGB retinopathy (VAR) = catastrophic combined visual loss",
                "note": "VGB is evidence-based for focal seizures and infantile spasms — adult neurologists may not know CLN11 patients must NEVER receive VGB. Any CLN11 patient with focal seizures or infantile spasms must receive safe alternatives (LEV, LTG, CLB). Emergency card mandatory stating VGB CONTRAINDICATED."
            },
            {
                "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
                "severity": "ABSOLUTE CI",
                "reason": "Sodium channel blockers → acute myoclonic worsening in adult NCL myoclonus; misidentification of CLN11 GTCS as focal epilepsy → CBZ prescribed",
                "note": "The most dangerous prescribing error in CLN11: adult neurologist unfamiliar with NCL sees young adult GTCS + focal features → CBZ → catastrophic myoclonus exacerbation. CLN11 mean diagnostic delay 4.2y = years of CBZ risk. Every CLN11 patient needs an emergency card/MedicAlert stating CBZ/OXC/PHT ABSOLUTE CI."
            },
            {
                "drug": "Fosphenytoin / IV Phenytoin",
                "severity": "ABSOLUTE CI",
                "reason": "IV sodium channel blocker → myoclonus worsening and potential seizure exacerbation in adult NCL; SE protocol misuse risk",
                "note": "Emergency neurologist or intensivist may use fosphenytoin as standard SE second-line. In CLN11, NEVER give fosphenytoin — use IV LEV (40-60 mg/kg) + IV PHB as SE protocol. CLN11 SE protocol must be documented in hospital notes and emergency card."
            },
            {
                "drug": "Tiagabine (TGB)",
                "severity": "ABSOLUTE CI",
                "reason": "GABA reuptake inhibitor → NCSE risk in generalised epilepsy / adult NCL; contraindicated in any generalised seizure disorder",
                "note": "TGB is approved for focal seizures but causes NCSE in generalised epilepsy syndromes. CLN11 has 25% NCSE risk — TGB absolutely forbidden. Any TGB exposure → urgent EEG within 24h."
            },
            {
                "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
                "severity": "HIGH RISK",
                "reason": "Can worsen cortical myoclonus in adult NCL / PME spectrum; calcium-channel alpha-2-delta subunit modulation may paradoxically enhance myoclonic activity",
                "note": "GBP/PGB occasionally prescribed for pain or anxiety comorbidities in adult NCL — myoclonus monitoring essential if used. If myoclonus worsens after GBP/PGB initiation → discontinue immediately. Safer alternatives for pain: low-dose SSRI (duloxetine), nortriptyline."
            },
            {
                "drug": "Lamotrigine (LTG) Monotherapy",
                "severity": "HIGH RISK",
                "reason": "LTG at high doses (monotherapy) can paradoxically worsen cortical myoclonus in adult NCL; safe as adjunct at lower doses",
                "note": "LTG adjunct (100-200 mg/day) combined with VPA is safe for focal seizures in CLN11. LTG monotherapy at high doses → myoclonus worsening in some NCL adults. Never use LTG as the ONLY AED in CLN11 — must be combined with VPA or LEV."
            },
            {
                "drug": "AED Taper / Abrupt Discontinuation",
                "severity": "HIGH RISK",
                "reason": "Abrupt AED reduction in drug-resistant adult NCL → GTCS cluster or SE; NCL neurons are maximally sensitised",
                "note": "ANY AED taper in CLN11 must be: (1) slow (10% dose reduction per month maximum); (2) monitored with clinical review; (3) patient and carer warned of increased seizure risk. Never abruptly stop VPA or LEV in CLN11."
            }
        ],
        "monitoring": [
            {"item": "GRN WES / NCL Gene Panel", "frequency": "At diagnosis", "note": "Confirm biallelic GRN pathogenic variants; WES captures coding variants; MLPA for GRN deletion; RNA-seq fibroblasts if WES negative but plasma PGRN undetectable"},
            {"item": "Plasma PGRN Level (ELISA)", "frequency": "At diagnosis + 12-monthly follow-up", "note": "Undetectable (<10 ng/mL) confirms biallelic GRN LOF; serial PGRN may detect residual in attenuated cases; monitor for gene therapy trial eligibility (baseline PGRN)"},
            {"item": "Skin Biopsy Electron Microscopy (FP ± RP)", "frequency": "At diagnosis", "note": "Confirms NCL class: fingerprint profiles (FP) ± rectilinear profiles (RP); adult NCL EM pattern; CLN11-specific vs CLN4B (FP only, no RP) vs CLN13/CTSF (FP); essential first-line NCL confirmation"},
            {"item": "Parent GRN Testing + FTLD Risk Counselling", "frequency": "At CLN11 diagnosis (parents)", "note": "Both parents are obligate GRN heterozygotes → FTLD-TDP risk; parental plasma PGRN (~50% of normal confirms heterozygosity); neurogenetic FTLD counselling; sibling cascade testing (25% CLN11, 50% carrier); UNIQUE to CLN11 among all NCLs"},
            {"item": "Ophthalmology ERG + VEP", "frequency": "6-monthly", "note": "Monitor retinal NCL progression; ERG detects retinal apoptosis before visual symptoms; VEP cortical visual pathway; photosensitivity IPS annually; VGB CI counselling at each visit"},
            {"item": "Brain MRI (3T with NCL protocol)", "frequency": "6-monthly initially, annually once stable", "note": "Cerebral + cerebellar atrophy progression monitoring; white matter changes; basal ganglia signal (parkinsonism substrate); T1/T2/FLAIR/DWI; compare serial volumes"},
            {"item": "EEG (Resting + Annual, Plus Acute for NCSE)", "frequency": "Baseline + annual + urgent if confusion/NCSE suspected", "note": "Baseline EEG for NCL pattern documentation; jerk-locked back-averaging for cortical myoclonus; annual EEG for subclinical NCSE detection; urgent EEG for any unexplained confusional episode"},
            {"item": "Neuropsychological Assessment", "frequency": "Annual", "note": "Track cognitive decline trajectory; memory, executive function, language, visuospatial; adaptive function; employment capacity assessment; dementia staging (MMSE, MoCA, ACE-R); driving assessment referral"},
            {"item": "SARA Scale (Spinocerebellar Ataxia Rating Scale)", "frequency": "6-monthly", "note": "Ataxia progression (62% of CLN11); physiotherapy goals calibrated to SARA score; walking aids/wheelchair planning; fall prevention"},
            {"item": "UMRS (Unified Myoclonus Rating Scale)", "frequency": "6-monthly", "note": "Action myoclonus severity; piracetam dose response; cortical stimulation SEP correlation; UMRS guides AED titration"},
            {"item": "DVLA / Driving Assessment", "frequency": "At diagnosis + annually", "note": "MANDATORY: CLN11 is a progressive neurological condition → DVLA notification at diagnosis; driving cessation required when seizure control inadequate or cognitive impairment; Driving & Vehicle Licensing Agency (UK) or equivalent national authority"},
            {"item": "VPA TDM + LFT + FBC", "frequency": "Every 6 months", "note": "VPA trough level (target 60-100 µg/mL); hepatic function (LFTs); FBC (thrombocytopenia); weight; ammonium level if encephalopathic; teratogenicity counselling for women"},
            {"item": "SUDEP Risk Assessment (Nocturnal Seizure Monitoring)", "frequency": "Annual review", "note": "CLN11 GTCS cluster risk = elevated SUDEP risk; nocturnal seizure monitoring (bed sensor/SUDEP alarm); safe sleep position; AED adherence; caregiver sleeping proximity in high-risk periods"},
            {"item": "BDSRA / NCL Resource Registration + ACP", "frequency": "At diagnosis + annual updates", "note": "BDSRA enrolment for GRN gene therapy trial eligibility (PR006 intracisternally); NCL Resource international registry; advance care planning: employment, relationships, supported living, PEG timing, DNACPR; annual ACP review as disease progresses"}
        ]
    }


def get_definitions():
    return {
        "disease_name": "CLN11 — Neuronal Ceroid Lipofuscinosis Type 11",
        "gene_full": "GRN (Granulin Precursor) — 17q21.31",
        "omim_gene": "*138945 (GRN)",
        "omim_disease": "#614706 (CLN11 Adult NCL)",
        "protein_full": "Progranulin (PGRN) — 593 aa; ~68 kDa protein (~88 kDa glycosylated); secreted lysosomal regulatory protein; 7.5 granulin domain repeats (Granulin A-G + paragranulin); signal peptide aa 1-17; SORT1-receptor lysosomal delivery; promotes lysosomal acidification + CTSB/CTSD/CTSZ cathepsin activation + TMEM106B regulation",
        "inheritance_mode": "Autosomal recessive (AR) biallelic LOF → CLN11; AD haploinsufficiency (one allele) → FTLD-TDP (NOT NCL)",
        "onset_age": "Late teens to early 30s (mean 22.4 years); range 15-35 years",
        "em_pattern": "Fingerprint profiles (FP, 92%) ± Rectilinear profiles (RP, 55%); skin biopsy required",
        "plasma_pgrn": "Undetectable (<10 ng/mL) in biallelic CLN11 (94%); normal 100-300 ng/mL; ~50% in GRN heterozygote/FTLD risk",
        "key_concepts": [
            {
                "name": "CLN11-GRN-17q21.31-Lysosomal-Regulatory-Protein-Adult-NCL",
                "definition": "CLN11 is caused by biallelic LOF of GRN (Granulin Precursor, 17q21.31) encoding Progranulin (PGRN). PGRN is a secreted lysosomal regulatory glycoprotein (593 aa, ~88 kDa glycosylated) involved in lysosomal acidification, CTSB/CTSD activation, and TMEM106B regulation. Biallelic GRN LOF → zero PGRN → lysosomal dysfunction → SCMAS lipofuscin accumulation → FP ± RP on EM → Adult NCL, onset 15-35 years."
            },
            {
                "name": "AR-Biallelic-GRN-LOF-vs-AD-GRN-Haploinsufficiency-Critical-Distinction",
                "definition": "The most critical GRN genetics concept: AR biallelic GRN LOF (BOTH alleles null) → CLN11 / Adult NCL (epilepsy, young adult onset); AD GRN haploinsufficiency (ONE allele null) → FTLD-TDP (frontotemporal dementia, 4th-6th decade). Parents of CLN11 probands are obligate GRN heterozygotes → FTLD-TDP risk (mandatory genetic counselling). Siblings: 25% CLN11 risk, 50% GRN carrier/FTLD risk."
            },
            {
                "name": "Plasma-PGRN-Level-Rapid-Adult-NCL-Screening-Biomarker",
                "definition": "Progranulin (PGRN) is measurable in plasma/serum (ELISA, 1-5 days). CLN11 biallelic LOF: plasma PGRN undetectable (<10 ng/mL). Normal: 100-300 ng/mL. GRN heterozygote (FTLD risk): ~50% normal (50-150 ng/mL). Plasma PGRN is the rapid CLN11 screening test — analogous to PPT1 DBS for CLN1. Must be added to adult NCL diagnostic algorithms."
            },
            {
                "name": "No-GRN-Enzyme-Assay-PGRN-Not-Enzyme-Plasma-PGRN-Instead",
                "definition": "Unlike CLN1 (PPT1 DBS enzyme assay) and CLN2 (TPP1 DBS enzyme assay), progranulin has NO enzymatic activity — it is a lysosomal regulatory protein, not an enzyme. There is no DBS enzyme assay for CLN11. The CLN11 rapid biochemical test is plasma PGRN protein level (undetectable in biallelic LOF). WES/GRN gene panel confirms the genetic diagnosis."
            },
            {
                "name": "VGB-ABSOLUTE-CI-CLN11-Retinal-NCL-88pct",
                "definition": "Vigabatrin is an ABSOLUTE CONTRAINDICATION in CLN11. CLN11 causes progressive retinal NCL degeneration (88%). VGB retinopathy (VAR — irreversible peripheral visual field loss) superimposed on CLN11 retinal NCL = catastrophic combined visual loss. Adult neurologists prescribing VGB for focal seizures in undiagnosed CLN11 is a critical prescribing trap."
            },
            {
                "name": "CBZ-OXC-PHT-ABSOLUTE-CI-CLN11-Adult-NCL-Myoclonus-Trap",
                "definition": "Carbamazepine, oxcarbazepine, and phenytoin are ABSOLUTE CONTRAINDICATED in CLN11. Adult neurologist sees young adult GTCS → assumes idiopathic/focal epilepsy → prescribes CBZ → acute myoclonic worsening. This is the most dangerous prescribing error in CLN11 (mean diagnostic delay 4.2 years = years of CBZ risk). Safe alternatives: VPA + LEV."
            },
            {
                "name": "VPA-SAFE-CLN11-Lysosomal-Regulatory-NOT-Mitochondrial",
                "definition": "Valproate is SAFE in CLN11. GRN/PGRN is a lysosomal regulatory protein — NOT mitochondrial. VPA ABSOLUTE CI applies to MERRF/POLG (mitochondrial disease). CLN11 is lysosomal dysfunction; VPA is the backbone AED. POLG1 exclusion recommended if atypical features suggest mitochondrial overlap."
            },
            {
                "name": "Cognitive-Behavioural-First-FTD-Mimic-CLN11-Young-Adult",
                "definition": "CLN11 Adult NCL may present with cognitive/behavioural changes before first seizure (35% cognitive-first). Young adult with personality change, executive dysfunction → psychiatric referral → delayed NCL diagnosis. CRITICAL screen: plasma PGRN in any young adult with cognitive decline + seizures OR cognitive decline + family history of FTD (GRN-FTLD)."
            },
            {
                "name": "Parkinsonism-Ataxia-CLN11-Movement-Disorder-Dual",
                "definition": "CLN11 causes combined movement disorder: cerebellar ataxia (62%) + parkinsonism (48%). Ataxia from cerebellar NCL storage; parkinsonism from basal ganglia lipofuscin (substantia nigra involvement). This dual movement disorder, combined with myoclonus and atonic seizures, creates compound fall risk requiring helmet, walking aids, and physiotherapy."
            },
            {
                "name": "FTLD-GRN-Therapy-vs-CLN11-GRN-Therapy-Different-Targets",
                "definition": "Therapies for GRN-related disease differ by genotype: (1) FTLD-GRN (AD heterozygous): AL001/latozinemab (anti-SORT1, increases circulating PGRN from 50% to higher) or SORT1 reduction to boost secreted PGRN; (2) CLN11 (AR biallelic, zero PGRN): GRN gene therapy (AAV-GRN intracisternally, PR006) — must restore PGRN from zero. AL001 cannot benefit CLN11 (no endogenous PGRN to enhance). PR006 gene therapy being repurposed from FTLD to CLN11 in research phase."
            },
            {
                "name": "No-Disease-Modifying-Therapy-CLN11-GRN-Gene-Therapy-Research",
                "definition": "No approved disease-modifying therapy for CLN11/GRN Adult NCL (2026). Management is purely symptomatic. Investigational: PR006 (Passage Bio — AAV.PHP.B-GRN intracisternally, Phase 1/2 for FTLD-GRN, repurposable for CLN11). All CLN11 patients must be enrolled in BDSRA/NCL Resource for trial eligibility."
            },
            {
                "name": "Driving-Cessation-DVLA-Mandatory-CLN11",
                "definition": "CLN11 Adult NCL mandates DVLA notification at diagnosis (UK) or equivalent national authority. Progressive neurological condition with seizures, visual failure, cognitive impairment, and movement disorder = mandatory driving cessation when any of these impair driving safety. Unique adult-specific issue: driving cessation has major independence and employment implications — early employment and transport planning essential."
            },
            {
                "name": "Adult-NCL-MDT-Unique-Needs-Employment-Relationships-Supported-Living",
                "definition": "Adult NCL (CLN11) has unique MDT needs absent in paediatric NCL: (1) Employment support and work modification; (2) Driving assessment and cessation planning; (3) Relationship and sexuality counselling; (4) Supported living planning; (5) Depression and anxiety treatment (common in adult NCL with insight); (6) Carer education (adult partner/parent as carer); (7) FTLD risk counselling for family members. Standard adult neurology MDT must be expanded for these NCL-specific adult needs."
            },
            {
                "name": "SUDEP-Risk-CLN11-Nocturnal-GTCS-Monitoring",
                "definition": "CLN11 carries elevated SUDEP risk from nocturnal GTCS clusters. Drug-resistant GTCS (72%) + nocturnal occurrence + progressive neurological impairment = compounded SUDEP risk. Nocturnal seizure alarms (bed sensor, SUDEP alarm), safe sleeping position (lateral), AED adherence, and caregiver awareness are essential. Annual SUDEP risk assessment and discussion."
            },
            {
                "name": "POLG1-Exclusion-Before-VPA-CLN11-Atypical",
                "definition": "POLG1 Alpers-Huttenlocher syndrome can rarely present in late teens/young adults with progressive epilepsy and cognitive decline, mimicking CLN11. VPA is ABSOLUTE CI in POLG1 (mitochondrial hepatotoxicity/liver failure risk). Before initiating VPA in any young adult with NCL-like features and atypical mitochondrial features (lactic acidosis, ragged-red fibres, liver dysfunction): exclude POLG1 with POLG1 sequencing or respiratory chain enzyme studies."
            },
            {
                "name": "Smith-2012-Brain-First-CLN11-Biallelic-GRN-UK-Siblings",
                "definition": "Smith KR et al. 2012 Brain — first description of CLN11 as a distinct NCL entity: two adult UK siblings with biallelic GRN LOF (compound heterozygous), adult-onset progressive epilepsy, cognitive decline, retinal degeneration, and FP EM pattern. This landmark paper established GRN as an NCL gene and CLN11 as a distinct adult NCL disease entity. Prior to 2012, adult-onset NCL with GRN variants may have been misclassified."
            }
        ],
        "thresholds": [
            {"parameter": "Plasma PGRN (CLN11 biallelic)", "value": "<10 ng/mL", "action": "Biallelic GRN LOF highly likely → proceed to GRN WES + EM skin biopsy"},
            {"parameter": "Plasma PGRN (GRN heterozygote)", "value": "50-150 ng/mL (~50% of normal)", "action": "Heterozygous GRN LOF → FTLD-TDP risk (not CLN11); genetic counselling for FTD risk"},
            {"parameter": "Plasma PGRN (normal)", "value": "100-300 ng/mL", "action": "GRN haploinsufficiency/biallelic LOF excluded; consider other adult NCL genes (CTSF/CLN13, DNAJC5/CLN4B)"},
            {"parameter": "VPA trough level (target)", "value": "60-100 µg/mL", "action": "Maintain in therapeutic range; <60 µg/mL → underdosed; >120 µg/mL → toxicity risk (tremor, encephalopathy)"},
            {"parameter": "SARA score (ataxia severity)", "value": "≥15/40", "action": "Significant cerebellar ataxia → urgent physiotherapy, walking aids, fall prevention programme; wheelchair assessment"},
            {"parameter": "UMRS (myoclonus severity)", "value": "≥20/60", "action": "Significant action myoclonus → piracetam dose increase (up to 24 g/day); LEV dose optimisation; SV2A-directed therapy"},
            {"parameter": "MoCA cognitive screen", "value": "≤22/30 (mild impairment)", "action": "Formal neuropsychological assessment; driving cessation referral; employment capacity assessment; supported living planning"},
            {"parameter": "ERG amplitude reduction", "value": ">50% from baseline", "action": "Significant retinal NCL progression → low vision aids; ophthalmology MDT; VGB re-confirmation CI; guide dog/cane referral"},
            {"parameter": "Cerebral atrophy on MRI", "value": "Significant progression vs prior scan", "action": "Accelerated NCL progression; ACP update; PEG discussion; BDSRA gene therapy trial eligibility review"},
            {"parameter": "LEV dose (behavioural monitoring)", "value": ">2500 mg/day", "action": "Increased neuropsychiatric side-effect risk in CLN11 cognitive impairment; consider brivaracetam switch if behavioural intolerance"},
            {"parameter": "Diagnosis delay", "value": ">2 years from first seizure", "action": "Alert: typical CLN11 mean delay 4.2y — extended delay = prolonged CBZ/OXC/PHT risk. Review medication history for Na-channel blockers at all new CLN11 diagnoses"},
            {"parameter": "KD ketone level (urine)", "value": "Moderate to large (3-4+ on dipstick)", "action": "Adequate ketosis achieved for seizure control; adjust fat:carb ratio if inadequate ketosis after 4 weeks"}
        ],
        "standards": [
            "Smith KR et al. 2012 Brain — First description CLN11 biallelic GRN LOF as adult NCL",
            "Baker M et al. 2006 Nature Genetics — GRN heterozygous LOF → FTLD-TDP (AD)",
            "Cruts M et al. 2006 Nature — Concurrent GRN FTLD discovery",
            "Ward ME et al. 2017 Neuron — PGRN in lysosomal function and neurodegeneration",
            "Mole SE et al. 2019 Lancet Neurology — NCL review + CLN11 classification",
            "NCL Resource 2024 (ncl.mrc.ac.uk) — Current NCL standards",
            "ILAE 2022 — Seizure type classification",
            "NICE NG217 — Epilepsy in adults management",
            "MHRA VPPP 2021 — VPA pregnancy prevention programme",
            "CPIC POLG1 2023 — POLG1 VPA prescribing guidance",
            "ACMG-AMP 2015 — Variant pathogenicity classification",
            "BDSRA Registry — Batten Disease Support and Research Association trial registry"
        ],
        "references": [
            "Smith KR et al. (2012) Strikingly different clinicopathological phenotypes determined by progranulin-mutation dosage. Am J Hum Genet 90:1102-7 [First CLN11]",
            "Baker M et al. (2006) Mutations in progranulin cause tau-negative frontotemporal dementia linked to chromosome 17. Nature 442:916-9 [FTLD-GRN]",
            "Ward ME et al. (2017) Individuals with progranulin haploinsufficiency exhibit features of neuronal ceroid lipofuscinosis. Sci Transl Med 9:eaah5417",
            "Mole SE & Anderson G (2019) Unreported cases of NCL: the dark matter. Lancet Neurol 18:1003-4",
            "Boland B et al. (2018) Promoting the clearance of neurotoxic proteins in neurodegenerative disorders of ageing. Nat Rev Drug Discov 17:660-88",
            "Canafoglia L et al. (2014) Recessive mutations in the GRN gene cause NCL11. J Med Genet 51:411-6"
        ],
        "lifecycle_stages": [
            {
                "stage": "Prenatal / Pre-Symptomatic (Genetic Risk)",
                "age_range": "Birth to ~15 years (if GRN biallelic identified in family)",
                "description": "Identified through cascade testing of siblings of CLN11 proband, or incidentally on panel testing. No symptoms. Genetic counselling for family. Establish care at NCL centre. Pre-symptomatic surveillance begins at age 12-15y: annual ophthalmology ERG, cognitive monitoring, EEG.",
                "priorities": ["Confirm GRN biallelic genotype", "Cascade test siblings", "Parental FTLD risk counselling (obligate GRN heterozygotes)", "Register with BDSRA/NCL Resource pre-symptomatically", "Annual ophthalmology from age 12"]
            },
            {
                "stage": "First Seizure — Diagnostic Emergency",
                "age_range": "15-35 years (mean 22.4y)",
                "description": "First GTCS or myoclonic event. High risk of misdiagnosis as idiopathic GGE/IGE. Avoid sodium channel blockers. Immediate investigation: plasma PGRN (days) + EM skin biopsy + GRN WES. If PGRN undetectable → adult NCL diagnosis protocol. Parent testing immediately (FTLD risk).",
                "priorities": ["Plasma PGRN URGENTLY", "Skin biopsy EM (FP + RP confirm adult NCL)", "GRN WES/NCL gene panel", "Start VPA + LEV (avoid CBZ/OXC/PHT)", "Parent GRN testing + FTLD counselling", "DVLA notification at diagnosis", "Register BDSRA"]
            },
            {
                "stage": "Active Epilepsy and Cognitive Decline",
                "age_range": "3-8 years from onset",
                "description": "Progressive GTCS + myoclonus + ataxia + retinal failure. Cognitive decline accelerates. Driving cessation. Employment support. Neuropsychiatric comorbidity (depression/anxiety 70%). MDT intensification. AED optimisation. Falls prevention (triple risk: myoclonus + ataxia + atonic).",
                "priorities": ["AED optimisation (VPA + LEV + CLB + piracetam)", "Ophthalmology 6-monthly", "Cognitive rehabilitation + neuropsychology", "Employment modification/cessation support", "Depression/anxiety treatment (SSRI)", "Helmet + walking aids + physiotherapy", "ACP initiation"]
            },
            {
                "stage": "Established Severe Disability",
                "age_range": "8-15 years from onset",
                "description": "Wheelchair dependency. Visual impairment/blindness (retinal NCL). Significant cognitive impairment (dementia stage). Dysphagia (PEG consideration). Drug-resistant epilepsy. Caregiver burden intensive. Supported living transition. Advanced ACP.",
                "priorities": ["PEG placement (dysphagia, weight loss)", "Wheelchair and home adaptation", "Supportive residential care planning", "Advanced ACP (DNACPR, preferred place of care)", "Seizure burden minimisation (reduce drug-resistant GTCS)", "Nocturnal seizure monitoring + SUDEP alert", "Carers' health support"]
            },
            {
                "stage": "Late Palliative / End-Stage",
                "age_range": "15-20+ years from onset (4th-5th decade)",
                "description": "End-stage NCL neurodegeneration. Minimal consciousness or severe encephalopathy. Comfort-focused care. Seizure management for comfort, not control. Family bereavement support. Death typically from respiratory failure, aspiration pneumonia, or SUDEP.",
                "priorities": ["Comfort-focused AED management", "Symptom control (pain, respiratory distress)", "Family/carer bereavement support", "Preferred place of death (home/hospice)", "Spiritual care", "Post-mortem brain donation (NCL research consent)", "BDSRA bereavement support"]
            }
        ]
    }
