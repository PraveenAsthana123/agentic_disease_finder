"""
HEXB Epilepsy — Sandhoff Disease / GM2 Gangliosidosis Type 2
=============================================================
40-patient cohort · HEXB (5q13.3) · Autosomal recessive (AR) biallelic LOF
HEXB encodes β-Hexosaminidase B beta-subunit; 556 aa precursor ~63 kDa (signal peptide 1-38;
propeptide 39-57; mature β-subunit ~60 kDa after lysosomal processing); TIM barrel fold;
beta-subunit is SHARED by ALL THREE hexosaminidase forms:
  Hex A (αβ heterodimer) — primary GM2 ganglioside hydrolysis (with GM2AP)
  Hex B (ββ homodimer) — globoside Gb4 hydrolysis (visceral/systemic substrates)
  Hex S (αα homodimer) — minor GM2 cleavage (without GM2AP)
HEXB LOF → ALL THREE forms deficient (Hex A + Hex B + Hex S all low) — PATHOGNOMONIC of Sandhoff
(contrast HEXA/Tay-Sachs: only Hex A + Hex S deficient; Hex B ELEVATED).

HEXB FUNCTION:
  Beta-subunit: structural + catalytic component of Hex A AND Hex B
  Hex B (ββ homodimer): cleaves globoside Gb4 (N-acetylgalactosamine terminal) in visceral organs
  → HEXB LOF → Gb4 accumulation in liver, spleen, kidney, bone marrow — systemic storage
  Hex A (αβ): cleaves GM2 ganglioside in CNS neurons with GM2AP → neurodegeneration (same as Tay-Sachs)
  All three forms share HEXB β-subunit → all deficient simultaneously = more systemic disease

HEXB LOF — THREE PHENOTYPIC FORMS (RESIDUAL ENZYME ACTIVITY):
══════════════════════════════════════════════════════════════
(1) TYPE 1 (CLASSIC INFANTILE, SEVERE): <0.1% residual Hex A + Hex B; onset 3-6 months;
    cherry-red spot (90%); exaggerated startle (hyperekplexia); infantile spasms (3-7m);
    hepatosplenomegaly (~70% — more than Tay-Sachs due to Hex B/Gb4 systemic accumulation);
    bone marrow foam cells (storage histiocytes); macrocephaly; death by 2-4y.
    Null alleles predominantly.
(2) TYPE 2 (JUVENILE, SUBACUTE): 2-8% residual Hex; onset 2-10y;
    PME phenotype — action myoclonus + GTCS + cerebellar ataxia; cognitive decline;
    cherry-red spot less common than Type 1 (<20%); hepatomegaly mild; survival teens.
    Missense alleles with partial activity.
(3) TYPE 3 (ADULT/CHRONIC): >10% residual Hex; onset adolescence-40y;
    spinocerebellar + LMN + psychiatric features; hepatosplenomegaly variable;
    epilepsy less prominent; progressive spinocerebellar ataxia phenotype.
    OMIM: *606873 (HEXB gene) / #268800 (Sandhoff Disease / GM2 Gangliosidosis Type 2)

HEXB vs HEXA vs GM2A — CRITICAL DIAGNOSTIC TRIAD:
════════════════════════════════════════════════════
  HEXA LOF (Tay-Sachs):   Hex A LOW + Hex B ELEVATED (β/β homodimer retained — PATHOGNOMONIC)
  HEXB LOF (Sandhoff):    Hex A LOW + Hex B LOW + Hex S LOW (all three deficient — PATHOGNOMONIC)
  GM2A LOF (AB variant):  Hex A NORMAL + Hex B NORMAL; GM2AP protein deficient → cannot load GM2
  DIAGNOSIS: Leukocyte 4-MU-hexosaminidase A and B assay SIMULTANEOUSLY.
  Hex A low + Hex B LOW = Sandhoff (NOT Tay-Sachs). Do NOT use DBS alone.

SANDHOFF SYSTEMIC INVOLVEMENT — MORE THAN TAY-SACHS:
══════════════════════════════════════════════════════
  Hepatosplenomegaly: ~70% Type 1 (vs rare in Tay-Sachs — Hex B cleaves visceral Gb4)
  Bone marrow foam cells: foamy storage histiocytes in marrow biopsy
  Renal involvement: Gb4 accumulates in kidney tubular epithelium
  Bony expansion possible (mild Gaucher-like marrow involvement)
  Peripheral NCS: may show demyelinating features (Hex B in Schwann cells)
  No MPS skeletal dysostosis multiplex (unlike MPS); no corneal clouding

HEXB FOUNDER MUTATIONS:
════════════════════════
  No major AJ founder mutation (unlike Tay-Sachs where c.1278insTATC 78% of AJ alleles)
  Sandhoff is RARE in Ashkenazi Jewish — different from Tay-Sachs (most AJ carrier screens normal)
  Spanish/Latin American: c.1514_1517delCTCA (p.Leu408MetfsTer10) — common in Creole/Cajun/Argentine
  Lebanese/Maronite: HEXB deletion — biallelic exonic deletion, founder in Lebanese Christian community
  Japanese: various private missense; HEXB p.Ile207ValfsTer29 reported
  Most non-founder cases: private mutations identified by HEXB WES; compound het common

KEY PHARMACOLOGICAL DISTINCTIONS — SANDHOFF / HEXB EPILEPSY:
════════════════════════════════════════════════════════════════
(1) CBZ/OXC/PHT ABSOLUTE CI — myoclonic worsening (same as Tay-Sachs; Type 2 PME GTCS misidentified as GGE → CBZ → myoclonic storm)
(2) VPA SAFE — lysosomal NOT mitochondrial; POLG1/MERRF mandatory exclusion before VPA
(3) ACTH Level A for infantile spasms (Type 1) — prefer over VGB when cherry-red present (90%)
(4) VGB HIGH RISK Type 1 (cherry-red 90%; retinal ganglion cell Hex A deficiency + VGB retinopathy = visual catastrophe)
(5) VGB HIGH RISK in hepatosplenomegaly (visceral disease monitoring needed)
(6) Fosphenytoin ABSOLUTE CI → replace with IV LEV 60 mg/kg in SE protocols
(7) Piracetam Level B for action myoclonus (Type 2 juvenile); Clonazepam Level B nocturnal
(8) No approved disease-modifying therapy; AAV9-HEXA-HEXB bicistronic gene therapy Phase I/II (2024) — same vector treats BOTH Tay-Sachs and Sandhoff
(9) Visceral screening MANDATORY — bone marrow biopsy, abdominal USS, LFTs — DISTINCT from Tay-Sachs management
"""

import random
random.seed(43)


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def get_overview():
    return {
        "disease": (
            "Sandhoff Disease (GM2 Gangliosidosis Type 2) — HEXB LOF → ALL THREE hexosaminidase forms deficient "
            "(Hex A + Hex B + Hex S all low = PATHOGNOMONIC); GM2 ganglioside + globoside Gb4 accumulate. "
            "Three forms: Type 1 Infantile (classic; <0.1% residual Hex; cherry-red 90%; hepatosplenomegaly 70%; IS; death 2-4y); "
            "Type 2 Juvenile (2-8% residual; PME + ataxia; hepatomegaly mild; survival teens); "
            "Type 3 Adult/Chronic (>10% residual; spinocerebellar + LMN). "
            "HEXB is MORE SYSTEMIC than Tay-Sachs (Hex B/ββ cleaves visceral Gb4 — absent in Sandhoff → hepatosplenomegaly + marrow foam cells). "
            "This cohort focuses on Type 2 Juvenile (65%) — primary epilepsy management population."
        ),
        "gene": "HEXB (5q13.3)",
        "protein": (
            "β-Hexosaminidase B β-subunit (HEXB); 556 aa precursor ~63 kDa; TIM barrel fold; "
            "beta-subunit SHARED by Hex A (αβ), Hex B (ββ), and Hex S component; "
            "HEXB LOF → ALL THREE forms deficient (Hex A + Hex B + Hex S all absent). "
            "Hex B (ββ homodimer) cleaves globoside Gb4 (N-acetylgalactosamine) in visceral organs. "
            "Catalytic residue: Glu355 (acid/base) + Glu491 (nucleophile, β-subunit numbering); retaining mechanism. "
            "Hex A additionally requires GM2 activator protein (GM2AP, GM2A gene) to load GM2 ganglioside."
        ),
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF required; pLI ~0.84 (moderate-high LOF intolerance)",
        "omim": "*606873 (HEXB gene) / #268800 (Sandhoff Disease / GM2 Gangliosidosis Type 2)",
        "mechanism": (
            "HEXB LOF → β-subunit absent → all three hexosaminidase forms deficient: "
            "Hex A (αβ) → GM2 ganglioside accumulates in CNS neurons → neurodegeneration; "
            "Hex B (ββ) → globoside Gb4 accumulates in visceral organs (liver, spleen, kidney, bone marrow) → hepatosplenomegaly; "
            "Hex S (αα) → minor GM2 pathway also deficient. "
            "Dual CNS + systemic accumulation = more complex pathology than Tay-Sachs. "
            "Cherry-red macular spot (90% Type 1): retinal ganglion cell Hex A deficiency (GM2 storage) → "
            "foveal transparency reveals choroidal red against white GM2-laden perifoveal ring."
        ),
        "cohort_size": 40,
        "mean_onset_years": 4.2,
        "seizure_pct": 87,
        "myoclonus_pct": 68,
        "infantile_spasms_pct": 38,
        "dystonia_pct": 32,
        "drug_resistant_pct": 65,
        "mean_diagnosis_delay_years": 3.2,
        "cherry_red_spot_pct": 55,
        "type2_juvenile_pct": 65,
        "type3_adult_pct": 10,
        "type1_infantile_pct": 25,
        "hepatosplenomegaly_pct": 52,
        "bone_marrow_foam_cells_pct": 45,
        "on_vpa_pct": 70,
        "on_lev_pct": 62,
        "on_acth_pct": 32,
        "on_piracetam_pct": 42,
        "on_clonazepam_pct": 35,
        "no_aj_founder_note": (
            "Unlike Tay-Sachs (HEXA), Sandhoff has NO major Ashkenazi Jewish founder mutation — "
            "AJ carrier screening programmes do NOT detect Sandhoff carriers (different gene, different enzyme). "
            "Sandhoff presents in all ethnic groups; Spanish/Latin American c.1514_1517delCTCA most prominent founder allele. "
            "No ethnic pre-test probability advantage → clinical suspicion + enzyme assay primary diagnostic pathway."
        ),
        "discovery": (
            "Konrad Sandhoff (1968 FEBS Lett) — identified Hex A AND Hex B deficiency as enzymatic basis "
            "(distinguished from Tay-Sachs where only Hex A deficient). "
            "Sandhoff K et al. (1968) — first recognition that Sandhoff = TOTAL hexosaminidase deficiency (both A and B). "
            "O'Brien JS et al. (1970) — clarified enzyme biochemistry; Hex B ββ homodimer role in globoside catabolism. "
            "Bikfalvi A et al. (1980s) — HEXB gene cloning and characterisation. "
            "Neote K et al. (1990) — HEXB gene structure; 14 exons; 5q13 locus."
        ),
        "unique_feature": (
            "HEXB/Sandhoff is the 'total hexosaminidase deficiency' disease — ALL THREE hex forms absent: "
            "(1) Hex A (αβ) + Hex B (ββ) + Hex S (αα) all zero (vs Tay-Sachs: Hex B elevated); "
            "(2) Systemic Gb4 accumulation (Hex B absent) → hepatosplenomegaly + bone marrow foam cells — "
            "DISTINCTIVE from Tay-Sachs (purely neurological); "
            "(3) No AJ founder mutation → Sandhoff not detected by standard Tay-Sachs carrier screens; "
            "(4) AAV9-HEXA/HEXB bicistronic gene therapy (2024) targets BOTH diseases simultaneously — "
            "Sandhoff and Tay-Sachs share the same gene therapy vector."
        ),
        "hexb_hexa_gm2a_differential_note": (
            "Three GM2 gangliosidoses — critical diagnostic triad: "
            "(1) HEXB LOF (Sandhoff): Leukocyte Hex A LOW + Hex B LOW + Hex S LOW (ALL three forms deficient = PATHOGNOMONIC). "
            "Hepatosplenomegaly + bone marrow foam cells → more systemic than Tay-Sachs. "
            "(2) HEXA LOF (Tay-Sachs): Leukocyte Hex A LOW; Hex B ELEVATED (ββ retained — distinguishes from Sandhoff). "
            "(3) GM2A LOF (AB variant): Hex A NORMAL + Hex B NORMAL; GM2AP protein deficient — "
            "in vitro enzyme activity normal; in vivo GM2 accumulates. Confirm GM2A gene + GM2AP protein assay. "
            "RULE: Leukocyte 4-MU assay measures Hex A AND Hex B SIMULTANEOUSLY. "
            "Hex A low + Hex B LOW = Sandhoff. Hex A low + Hex B HIGH = Tay-Sachs. Both normal = AB variant."
        ),
        "systemic_involvement_note": (
            "HEXB SYSTEMIC INVOLVEMENT — UNIQUE vs HEXA/TAY-SACHS: "
            "Hex B (ββ homodimer) is the primary enzyme for globoside Gb4 catabolism in visceral organs. "
            "HEXB LOF → Gb4 accumulation in liver (hepatomegaly ~60%), spleen (splenomegaly ~50%), "
            "kidney tubular epithelium, bone marrow (foam cells = storage histiocytes ~45%). "
            "MANAGEMENT IMPLICATIONS: (1) Abdominal USS at diagnosis + 12-monthly (hepatosplenomegaly); "
            "(2) LFTs 3-monthly (hepatic Gb4 storage → enzyme leak); "
            "(3) Bone marrow biopsy if cytopenia (storage histiocytes can cause cytopenias); "
            "(4) Peripheral neuropathy evaluation (Hex B in Schwann cells); "
            "TAY-SACHS HAS NONE OF THESE → Sandhoff management significantly more complex."
        ),
        "key_pharmacological_distinctions": {
            "1_CBZ_OXC_PHT_ABSOLUTE_CI": (
                "Sodium channel blockers ABSOLUTE CI in Sandhoff juvenile PME (Type 2). "
                "Type 2 GTCS misidentified as GGE/JME → CBZ initiated → acute myoclonic storm. "
                "Mean diagnosis delay 3.2y = extended Na-channel blocker exposure window. "
                "Sandhoff Type 2 is a TRUE PME — action myoclonus + GTCS + cerebellar ataxia. "
                "SAFE: VPA + LEV + Piracetam backbone."
            ),
            "2_VPA_SAFE_NOT_MITOCHONDRIAL_POLG1_EXCLUSION_MANDATORY": (
                "HEXB is a lysosomal acid hydrolase — NOT mitochondrial. "
                "VPA CI applies EXCLUSIVELY to mitochondrial disease (MERRF/POLG Alpers). "
                "VPA is SAFE backbone in Sandhoff. "
                "POLG1 Alpers + MERRF both mimic Type 2 juvenile PME phenotype (ataxia + myoclonus + seizures). "
                "MANDATORY POLG1/MERRF exclusion by WES + mtDNA before VPA initiation. "
                "POLG1 Alpers + VPA = fatal hepatotoxicity. Sandhoff hepatomegaly does NOT contraindicate VPA "
                "(Gb4 hepatic storage, not mitochondrial failure)."
            ),
            "3_ACTH_LEVEL_A_INFANTILE_SPASMS_TYPE1": (
                "Infantile spasms (West syndrome) in Sandhoff Type 1: ACTH preferred over VGB. "
                "Cherry-red spot present in 90% of Type 1 → VGB retinopathy (irreversible VF loss) "
                "compounded with retinal ganglion cell GM2 storage (Hex A absent) → catastrophic combined visual failure. "
                "ACTH Level A (UKISS/INFANT protocol): 2mg/kg/day prednisolone OR ACTH 40-60 IU IM daily. "
                "Fundoscopy MANDATORY before IS treatment selection."
            ),
            "4_VGB_HIGH_RISK_TYPE1": (
                "VGB HIGH RISK in Sandhoff Type 1 (cherry-red 90%; Hex A absent in retinal ganglion cells). "
                "VGB retinopathy + GM2 retinal storage = catastrophic combined blindness. "
                "NOT absolute CI (unlike NCL where retinal NCL = absolute CI) but HIGH RISK → prefer ACTH. "
                "Additional VGB concern in Sandhoff: hepatosplenomegaly → potential hepatic drug accumulation "
                "(monitor LFTs if VGB initiated despite warnings). "
                "VGB acceptable in Type 3 (no cherry-red, no retinal Hex A involvement)."
            ),
            "5_FOSPHENYTOIN_ABSOLUTE_CI_IV_LEV_REPLACES": (
                "Fosphenytoin (IV phenytoin prodrug) = ABSOLUTE CI in Sandhoff SE. "
                "Standard SE algorithm second-line = IV fosphenytoin → myoclonic worsening + cortical hyperexcitability. "
                "REPLACE WITH: IV LEV 60 mg/kg (max 4500 mg). "
                "A&E + ED pre-notification mandatory: 'IV LEV, NOT fosphenytoin.' "
                "Hepatomegaly in Sandhoff → also monitor IV VPA hepatic function (safe if POLG1 excluded)."
            ),
            "6_PIRACETAM_ACTION_MYOCLONUS": (
                "Piracetam Level B evidence for action/intention myoclonus in Type 2 juvenile. "
                "Dose: 24-45g/day in adults (160-300 mg/kg/day in children). "
                "MOA: AMPA potentiation + membrane fluidity. "
                "Renal dose reduction in any renal tubular Gb4 involvement (rare but HEXB can affect kidney). "
                "Monitor renal function (Gb4 renal accumulation potential) before high-dose piracetam."
            ),
            "7_HEPATOMEGALY_VPA_MONITORING": (
                "Sandhoff hepatomegaly (Gb4 hepatic accumulation) requires ENHANCED VPA hepatic monitoring. "
                "LFTs 3-monthly on VPA (vs 6-monthly standard) in Sandhoff with hepatomegaly. "
                "Hepatic Gb4 storage → elevated LFTs baseline (distinguish from VPA hepatotoxicity). "
                "Ultrasound-guided hepatic assessment at VPA initiation (baseline hepatic size/echogenicity). "
                "POLG1 excluded → hepatomegaly alone does NOT contraindicate VPA."
            ),
            "8_AAV9_BICISTRONIC_GENE_THERAPY": (
                "AAV9-HEXA/HEXB bicistronic gene therapy (2024 Phase I/II): "
                "Treats BOTH Sandhoff AND Tay-Sachs with same vector — "
                "single vector encodes BOTH HEXA α-subunit + HEXB β-subunit → restores all three Hex forms. "
                "Intrathecal (CSF) route for CNS targeting. "
                "Sandhoff benefits particularly from the HEXB component — restores Hex B (visceral) AND Hex A (CNS). "
                "Best outcomes pre-symptomatic or early symptomatic. Urgent referral at diagnosis."
            ),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def get_breakdown():
    etiologies = [
        {
            "class": "Spanish/Latin-American-c.1514_1517delCTCA-Founder (Type 1 Infantile)",
            "pct": 25,
            "description": (
                "Founder deletion c.1514_1517delCTCA (p.Leu408MetfsTer10) — Spanish, Argentine, Creole, Cajun populations. "
                "Frameshift/null allele → <0.1% residual Hex → classic infantile Type 1. "
                "Cherry-red 90%; hepatosplenomegaly; infantile spasms; macrocephaly; death by 2-4y. "
                "Spanish-American carrier freq higher than general population (~1/50 in some isolates)."
            ),
            "typical_onset": "3–6 months (infantile spasms + hepatosplenomegaly)",
            "genotype_notes": "Most common single Sandhoff founder allele globally; null frameshift; classic infantile phenotype",
        },
        {
            "class": "Compound-Het-Null-Null-Non-Founder (Type 1 Infantile)",
            "pct": 22,
            "description": (
                "Compound heterozygote: two distinct null/truncating HEXB alleles (frameshift + nonsense or splice + frameshift). "
                "Private or rare mutations — no ethnic founder advantage; identified by HEXB WES. "
                "Clinical Type 1 infantile; hepatosplenomegaly + bone marrow foam cells; cherry-red 90%. "
                "Parental consanguinity in ~30% of this subgroup."
            ),
            "typical_onset": "3–6 months",
            "genotype_notes": "Compound null/null — <0.1% residual Hex A + Hex B; severe classic infantile Sandhoff",
        },
        {
            "class": "Compound-Het-Null-Missense (Type 2 Juvenile)",
            "pct": 28,
            "description": (
                "Compound heterozygote: null allele (frameshift/splice) + missense allele with partial β-subunit function. "
                "Residual Hex A 2-8% → Type 2 juvenile onset (2-10y). "
                "PME phenotype: action myoclonus + GTCS + cerebellar ataxia + cognitive decline. "
                "Hepatomegaly mild (Gb4 accumulation reduced by partial Hex B activity)."
            ),
            "typical_onset": "2–10 years (mean 4-5 years)",
            "genotype_notes": "Most common genotype in Type 2 juvenile Sandhoff; phenotype determined by residual Hex B activity of missense allele",
        },
        {
            "class": "Lebanese-Maronite-HEXB-Exonic-Deletion-Founder (Type 1/2)",
            "pct": 10,
            "description": (
                "Biallelic exonic deletion of HEXB — Lebanese Christian/Maronite community founder variant. "
                "Homozygous deletion → null; Type 1 infantile most common. "
                "Some heterozygous deletions compound with missense → Type 2 juvenile. "
                "Identified by HEXB copy number variant (CNV) analysis or deletion MLPA."
            ),
            "typical_onset": "3–6 months (homozygous Type 1); 2–8 years (compound het Type 2)",
            "genotype_notes": "Lebanese founder biallelic HEXB deletion; requires CNV/MLPA analysis (WES sequencing alone may miss deletion)",
        },
        {
            "class": "Homozygous-Missense-Consanguineous (Type 2 Juvenile / Type 3 Adult)",
            "pct": 10,
            "description": (
                "Homozygous missense variant — residual Hex A/B 2-10% → juvenile or adult phenotype. "
                "Consanguineous families (Middle Eastern, North African, South Asian). "
                "Type 2 juvenile: PME + cerebellar ataxia; Type 3 adult: spinocerebellar + LMN + psychiatric. "
                "Hepatosplenomegaly minimal to absent in Type 3 (sufficient Hex B activity to prevent visceral Gb4 accumulation)."
            ),
            "typical_onset": "Variable (2–40 years depending on residual Hex B activity)",
            "genotype_notes": "Missense homozygous — residual activity determines phenotype; consanguinity increases homozygosity",
        },
        {
            "class": "Japanese-HEXB-Private-Variants (Type 2 Juvenile)",
            "pct": 5,
            "description": (
                "Japanese population: HEXB p.Ile207ValfsTer29 and other private variants; compound het typical. "
                "Type 2 juvenile phenotype most common in Japanese Sandhoff cohort (residual Hex activity 2-8%). "
                "PME + cerebellar ataxia; cherry-red spot less common than AJ Tay-Sachs juvenile. "
                "No single founder predominates; HEXB WES required."
            ),
            "typical_onset": "2–10 years",
            "genotype_notes": "Japanese Sandhoff — private variants; HEXB WES required; Type 2 phenotype common in Japanese cohort",
        },
    ]

    seizure_types = [
        {
            "type": "Infantile Spasms (West Syndrome) — Type 1 Primary",
            "prevalence_pct": 38,
            "eeg_pattern": "Hypsarrhythmia (classic chaotic high-voltage); modified hypsarrhythmia; burst suppression in late-stage",
            "semiology": "Flexor/extensor or mixed spasms in clusters; startle-triggered clusters (hyperekplexia); 3-7 months; concurrent hepatosplenomegaly",
            "clinical_tips": (
                "ACTH Level A preferred — cherry-red spot 90% in Type 1 → VGB HIGH RISK. "
                "Hepatosplenomegaly present → baseline abdominal USS + LFTs before treatment. "
                "POLG1/MERRF exclusion before any VPA in IS management. "
                "Gene therapy referral URGENTLY at spasm onset — pre-symptomatic window closing."
            ),
        },
        {
            "type": "Action / Intention Myoclonus — Type 2 Juvenile Primary",
            "prevalence_pct": 68,
            "eeg_pattern": "Polyspike-wave (generalised); cortical correlate on back-averaging; high-amplitude frontal spikes",
            "semiology": "Erratic limb myoclonus activated by intention + touch; face + distal limbs; worsened by startle; falls risk; more prominent than in Tay-Sachs Type 2",
            "clinical_tips": (
                "Most disabling symptom in Type 2 juvenile. Piracetam Level B adjunct to VPA. "
                "CBZ/OXC ABSOLUTE CI — acute worsening. "
                "Monitor renal function (Gb4 renal accumulation) before high-dose piracetam. "
                "Action myoclonus polyspike-wave EEG ≠ primary generalised epilepsy — do NOT initiate CBZ."
            ),
        },
        {
            "type": "Generalised Tonic-Clonic Seizures (GTCS)",
            "prevalence_pct": 62,
            "eeg_pattern": "Generalised polyspike-wave 3-5 Hz → tonic then clonic; post-ictal suppression",
            "semiology": "Major GTCS; nocturnal predominance; often on myoclonic background; Type 2 main GTCS population; hepatosplenomegaly complicates prolonged SE",
            "clinical_tips": (
                "Type 2 GTCS misdiagnosed as GGE/JME → CBZ → myoclonic worsening (ABSOLUTE CI). "
                "VPA + LEV backbone. Rescue: buccal midazolam → IV LEV (NOT fosphenytoin). "
                "Hepatomegaly: enhance LFT monitoring on VPA. IV VPA acceptable if POLG1 excluded."
            ),
        },
        {
            "type": "Absence-Like / Dialeptic Seizures",
            "prevalence_pct": 35,
            "eeg_pattern": "Irregular 2-4 Hz spike-wave; background slowing; atypical absence morphology",
            "semiology": "Staring + unresponsiveness 5-30s; automatisms less prominent; post-ictal confusion (unlike CAE); Type 2",
            "clinical_tips": (
                "Atypical absence morphology + background slowing distinguishes from CAE. "
                "Do NOT initiate ethosuximide alone — coexisting GTCS + myoclonus need VPA. "
                "Do NOT initiate CBZ/OXC — worsens myoclonic background."
            ),
        },
        {
            "type": "Focal Cortical Seizures (Occipital/Motor)",
            "prevalence_pct": 30,
            "eeg_pattern": "Focal occipital or motor discharges; may secondarily generalise",
            "semiology": "Visual phenomena (phosphenes, elementary hallucinations); focal motor jerks; occipital GM2 accumulation in Type 2",
            "clinical_tips": (
                "Occipital GM2 accumulation → cortical hyperexcitability. "
                "LEV preferred adjunct; LTG caution (myoclonus worsening risk). "
                "Focal EEG does NOT indicate structural lesion — functional secondary to GM2 storage."
            ),
        },
        {
            "type": "Non-Convulsive Status Epilepticus (NCSE)",
            "prevalence_pct": 20,
            "eeg_pattern": "Prolonged irregular spike-wave; altered consciousness; EEG-confirmed NCSE",
            "semiology": "Altered awareness with myoclonic twitching + confusion; may be mistaken for post-ictal state; Type 2 late disease",
            "clinical_tips": (
                "NCSE risk increases in late Type 2. "
                "IV LEV 60 mg/kg or IV VPA (POLG1 excluded) — NOT fosphenytoin (ABSOLUTE CI). "
                "Hepatomegaly — LFT monitoring with IV VPA in Sandhoff NCSE. "
                "EEG monitoring mandatory in unexplained altered consciousness."
            ),
        },
    ]

    triggers = [
        {
            "trigger": "Fever / Intercurrent Illness",
            "prevalence_pct": 80,
            "mechanism": "Febrile threshold lowering in GM2 gangliosidosis; hepatosplenomegaly adds metabolic stress component",
            "management": "Aggressive antipyretics; sick-day rescue plan; hepatic function monitoring during febrile illness (LFTs if prolonged fever)",
        },
        {
            "trigger": "Sleep Deprivation",
            "prevalence_pct": 70,
            "mechanism": "Sleep dep lowers seizure threshold in PME; nocturnal GTCS + myoclonus both worsen",
            "management": "Rigid sleep schedule; melatonin for sleep maintenance; Clonazepam nocturnal myoclonus",
        },
        {
            "trigger": "Missed AED Dose",
            "prevalence_pct": 65,
            "mechanism": "VPA + LEV abrupt discontinuation → rebound cortical hyperexcitability; hepatic storage may affect VPA PK",
            "management": "Dosette box + carer-supervised dosing; modified-release VPA; TDM monitoring in hepatomegaly",
        },
        {
            "trigger": "Photic / Pattern Stimulation",
            "prevalence_pct": 50,
            "mechanism": "Visual cortex GM2 accumulation → cortical hyperexcitability; 15-25 Hz photic-driving provocative",
            "management": "Avoid strobe lighting; anti-reflective glasses; document photic sensitivity in EEG",
        },
        {
            "trigger": "Intentional Movement (Action Myoclonus Trigger)",
            "prevalence_pct": 68,
            "mechanism": "Cortical reflex myoclonus triggered by movement intention; back-averaging EEG confirms cortical origin",
            "management": "Piracetam Level B; VPA backbone; renal monitoring before high-dose piracetam (HEXB renal Gb4 accumulation)",
        },
        {
            "trigger": "Startle (Tactile / Auditory / Visual)",
            "prevalence_pct": 60,
            "mechanism": "Exaggerated startle reflex (hyperekplexia) — prominent in Type 1 infantile; startle-myoclonus in Type 2",
            "management": "Environmental noise reduction; clonazepam (most benzodiazepine-sensitive startle response); hearing protection",
        },
        {
            "trigger": "Hepatic Decompensation / Intercurrent Hepatic Stress",
            "prevalence_pct": 25,
            "mechanism": "Sandhoff-specific: Gb4 hepatic storage → hepatic metabolic stress during intercurrent illness → AED PK changes",
            "management": "LFTs at sick days; VPA TDM during hepatic illness; liver-sparing AED adjustments; abdominal USS annually",
        },
        {
            "trigger": "Prohibited Drug Exposure (CBZ/OXC/PHT/Fosphenytoin)",
            "prevalence_pct": 100,
            "mechanism": "Sodium channel blockers → cortical myoclonus worsening + GTCS exacerbation in GM2 gangliosidosis (same mechanism as Tay-Sachs)",
            "management": "ABSOLUTE CI — A&E alert card + GP records + hospital drug chart flag; IV LEV replaces fosphenytoin in SE",
        },
    ]

    treatments = [
        {
            "drug": "VPA (Sodium Valproate)",
            "level": "Level B",
            "dose": "20-60 mg/kg/day oral; target trough 50-100 mg/L; CR formulation preferred; enhanced LFT monitoring if hepatomegaly",
            "moa": "Multi-modal: T-Ca++ channel inhibition, GABA-T inhibition → GABA elevation, Na-channel stabilisation (minor), histone deacetylase inhibition",
            "efficacy": "72% ≥50% seizure reduction in Type 2 PME; 65% myoclonus reduction; backbone drug of Sandhoff epilepsy",
            "monitoring": "TDM trough 50-100 mg/L; LFT (ALT/AST) 3-monthly (hepatomegaly) — more frequent than standard; FBC; weight; POLG1 excluded BEFORE initiation; abdominal USS annually",
            "hexb_note": "HEXB lysosomal NOT mitochondrial — VPA SAFE. Hepatomegaly (Gb4) does NOT contraindicate VPA. Enhanced LFT monitoring warranted (Sandhoff hepatomegaly + VPA hepatic monitoring). POLG1/MERRF exclusion mandatory.",
        },
        {
            "drug": "LEV (Levetiracetam) — IV and Oral",
            "level": "Level B",
            "dose": "Oral: 20-60 mg/kg/day in 2 doses; IV SE: 60 mg/kg (max 4500 mg) over 15 min; renal dose adjustment if Gb4 nephropathy",
            "moa": "SV2A modulation → presynaptic vesicle trafficking inhibition → reduced glutamate release; modulates GABA-A",
            "efficacy": "62% ≥50% GTCS reduction in Type 2; primary IV alternative to fosphenytoin (ABSOLUTE CI) in SE",
            "monitoring": "Renal function (Gb4 renal tubular accumulation potential — dose-reduce if eGFR <50); psychiatric monitoring; no TDM needed",
            "hexb_note": "IV LEV REPLACES fosphenytoin in Sandhoff SE protocol. Renal monitoring essential (HEXB renal Gb4 involvement rare but present). VPA + LEV combination first-line.",
        },
        {
            "drug": "ACTH (Tetracosactide) — Infantile Spasms Type 1",
            "level": "Level A",
            "dose": "ACTH 40-60 IU IM daily x 2 weeks (reducing schedule) OR Prednisolone 2-10 mg/kg/day; per UKISS/INFANT protocol",
            "moa": "ACTH → MC2 receptor → cortisol + androgens → CRH suppression → spasm suppression; direct CNS ACTH receptor effect",
            "efficacy": "55-65% spasm cessation in Type 1 IS; lower than cryptogenic IS due to underlying structural/metabolic cause",
            "monitoring": "BP, electrolytes, glucose; infection risk (immunosuppression); ophthalmology within 2 weeks (glaucoma); weight; abdominal USS (hepatosplenomegaly monitoring during ACTH)",
            "hexb_note": "PREFERRED over VGB when cherry-red spot present (90% in Type 1). Hepatosplenomegaly surveillance during ACTH (steroid can exacerbate hepatic stress). Fundoscopy MANDATORY before IS treatment selection.",
        },
        {
            "drug": "Piracetam — Action Myoclonus",
            "level": "Level B",
            "dose": "Adults: 24-45 g/day in 3-4 doses; children: 160-300 mg/kg/day; start 8g/day → titrate over 2-4 weeks",
            "moa": "AMPA potentiation; membrane fluidity; reduces cortical hyperexcitability",
            "efficacy": "48-62% action myoclonus reduction in Type 2; functional improvement in writing, eating",
            "monitoring": "Renal function MANDATORY (Gb4 renal involvement — dose-reduce if eGFR falls); psychiatric monitoring; no hepatotoxicity",
            "hexb_note": "Renal monitoring MORE IMPORTANT in Sandhoff vs Tay-Sachs (Gb4 renal tubular accumulation potential). Assess renal USS + eGFR before high-dose piracetam. Added to VPA + LEV backbone.",
        },
        {
            "drug": "Clonazepam — Nocturnal Myoclonus / Startle",
            "level": "Level B",
            "dose": "0.5-6 mg/day; 0.5-1 mg nocturnal dose; anti-myoclonic most potent of all benzodiazepines",
            "moa": "GABA-A potentiation (benzodiazepine site); anti-myoclonic properties",
            "efficacy": "58% nocturnal myoclonus reduction; 68% startle-myoclonus reduction; tolerance develops 3-6 months",
            "monitoring": "Sedation; tolerance/dependence; respiratory depression; hepatic monitoring (Sandhoff hepatomegaly — benzodiazepine hepatic metabolism enhanced monitoring)",
            "hexb_note": "Best option for nocturnal and startle myoclonus. Hepatic metabolism — in Sandhoff hepatomegaly, monitor for prolonged sedation (reduced hepatic clearance). Rotate dosing to prevent tolerance.",
        },
        {
            "drug": "KD (Ketogenic Diet) — Adjunct",
            "level": "Level C",
            "dose": "Classical 4:1 KD or modified Atkins; specialist ketogenic diet team; hepatic function monitoring required in Sandhoff",
            "moa": "Ketosis → ATP-sensitive K-channel opening + glutamate reduction + GABAergic enhancement",
            "efficacy": "Limited Sandhoff-specific data; 40% ≥50% seizure reduction in drug-resistant paediatric PME generally",
            "monitoring": "Lipid profile; acidosis; kidney stones; growth; LFTs (hepatomegaly + KD lipid load → hepatic monitoring); abdominal USS",
            "hexb_note": "Hepatomegaly in Sandhoff requires ENHANCED hepatic monitoring on KD (high fat load + pre-existing Gb4 hepatic storage). Dietitian + hepatologist co-management recommended.",
        },
        {
            "drug": "MDT Palliative / Supportive Care",
            "level": "Level A",
            "dose": "Multidisciplinary: neurology + genetics + hepatology + physiotherapy + OT + palliative care + dietitian",
            "moa": "Supportive — seizure management, visceral monitoring, nutrition, mobility, quality of life",
            "efficacy": "Quality of life + caregiver wellbeing + visceral complication management + functional duration",
            "monitoring": "Swallowing (FEES/VFS); hepatomegaly USS annually; bone marrow if cytopenia; chest physio; scoliosis XR",
            "hexb_note": "HEPATOLOGY co-management MANDATORY in Sandhoff (vs Tay-Sachs which is purely neurological). No approved disease-modifying therapy. Gene therapy Phase I/II — urgent trial referral at diagnosis.",
        },
        {
            "drug": "Rescue Midazolam / IV LEV — SE Protocol",
            "level": "Level A",
            "dose": "Midazolam buccal: 0.5 mg/kg (max 10 mg). IV LEV SE: 60 mg/kg (max 4500 mg) over 15 min",
            "moa": "Midazolam: GABA-A potentiation. IV LEV: SV2A → presynaptic glutamate reduction.",
            "efficacy": "Buccal midazolam 76% SE termination. IV LEV: 63% SE termination (2nd line); replaces fosphenytoin (ABSOLUTE CI)",
            "monitoring": "Respiratory monitoring post-midazolam; BP + renal function post-IV LEV; LFTs if IV VPA (POLG1 excluded, hepatomegaly caution)",
            "hexb_note": "Sandhoff SE protocol: Midazolam → IV LEV → IV VPA (POLG1 excluded; monitor LFTs given hepatomegaly) → anaesthesia. NEVER fosphenytoin. Hepatic function assessment before IV VPA in Sandhoff.",
        },
    ]

    contraindications = [
        {
            "drug": "CBZ / OXC / PHT (Carbamazepine / Oxcarbazepine / Phenytoin)",
            "severity": "ABSOLUTE CI",
            "reason": "Sodium channel blockers cause acute myoclonic worsening + GTCS exacerbation in GM2 gangliosidosis/PME. Type 2 juvenile GTCS misidentified as GGE/JME → CBZ → myoclonic storm. Mean diagnosis delay 3.2y = extended exposure window.",
            "note": "Safe alternative: VPA + LEV. Na-channel blocker ABSOLUTE CI in all PME phenotypes including Sandhoff Type 2.",
        },
        {
            "drug": "Fosphenytoin (IV Phenytoin Prodrug)",
            "severity": "ABSOLUTE CI",
            "reason": "Standard SE protocol second-line = fosphenytoin → myoclonic worsening in Sandhoff SE. Must be overridden explicitly in all Sandhoff patients. Additional concern: hepatic Gb4 storage in Sandhoff alters phenytoin hepatic metabolism.",
            "note": "REPLACE WITH: IV LEV 60 mg/kg. Pre-populate hospital drug chart + emergency card. Hepatomegaly → avoid hepatically-metabolised agents where possible.",
        },
        {
            "drug": "VGB (Vigabatrin) — Type 1 (HIGH RISK)",
            "severity": "HIGH RISK",
            "reason": "VGB retinopathy (irreversible VF loss) + retinal ganglion cell Hex A deficiency (GM2 storage) in Type 1 → catastrophic visual failure. Hepatosplenomegaly adds VGB hepatic monitoring complexity. NOT absolute CI (unlike NCL) but HIGH RISK.",
            "note": "ACTH first-line for IS (Type 1). If VGB used → ERG/VEP every 3 months. Additional monitoring: LFTs (hepatomegaly + VGB). VGB acceptable in Type 3 (no retinal involvement).",
        },
        {
            "drug": "TGB (Tiagabine)",
            "severity": "HIGH RISK",
            "reason": "GABA reuptake inhibitor → paradoxical cortical disinhibition in PME; NCSE risk. Hepatic metabolism (hepatomegaly → prolonged TGB exposure risk). Avoid in all PME phenotypes.",
            "note": "If NCSE → IV LEV or IV VPA. Never initiate TGB in Sandhoff PME.",
        },
        {
            "drug": "GBP / PGB (Gabapentin / Pregabalin)",
            "severity": "HIGH RISK",
            "reason": "Alpha-2-delta ligands worsen myoclonus in PME; documented myoclonic exacerbation in cortical myoclonus. Renal excretion — Gb4 renal involvement in Sandhoff → drug accumulation if eGFR reduced.",
            "note": "Monitor renal function. If neuropathic pain → low-dose amitriptyline or duloxetine (hepatic — monitor in hepatomegaly).",
        },
        {
            "drug": "LTG Monotherapy (Lamotrigine — avoid as sole drug in Type 2 PME)",
            "severity": "HIGH RISK",
            "reason": "LTG can worsen myoclonus in PME when used as monotherapy. Safe as LOW-dose adjunct to VPA (VPA doubles LTG half-life → halve LTG dose).",
            "note": "Strict dose halving with VPA co-medication; never as sole AED in Type 2; hepatic monitoring (hepatomegaly + LTG hepatic glucuronidation).",
        },
        {
            "drug": "High-Dose Benzodiazepines (Prolonged Hepatic Exposure Risk)",
            "severity": "CAUTION",
            "reason": "Sandhoff-specific: Hepatomegaly (Gb4 storage) → reduced hepatic benzodiazepine clearance → prolonged sedation at standard doses. Not absolute CI but dose-reduction and sedation monitoring required.",
            "note": "Start low, titrate slowly. Monitor for excessive sedation. Respiratory monitoring. Consider reduced starting doses in significant hepatomegaly.",
        },
    ]

    monitoring = [
        {
            "item": "Leukocyte Hex A + Hex B assay SIMULTANEOUSLY (4-MU substrate)",
            "frequency": "At diagnosis; repeat 12-monthly if equivocal; confirm ALL DBS low hex with leukocyte",
            "note": "Gold standard. Hex A LOW + Hex B LOW + Hex S LOW = Sandhoff (not Tay-Sachs). Do NOT use DBS alone. Measure both A and B to distinguish from Tay-Sachs (Hex B HIGH in Tay-Sachs).",
        },
        {
            "item": "HEXB Gene Sequencing (WES or HEXB panel) + CNV/MLPA Analysis",
            "frequency": "Once at diagnosis; cascade family testing; Lebanese/Arab families: CNV analysis mandatory",
            "note": "WES may MISS large HEXB exonic deletions (Lebanese founder) — CNV/MLPA analysis required. Spanish/Latin-American: c.1514_1517delCTCA founder check. No AJ-specific panel (unlike Tay-Sachs).",
        },
        {
            "item": "Abdominal USS (Liver + Spleen Size)",
            "frequency": "At diagnosis; 12-monthly; urgent if cytopenia or abdominal symptoms",
            "note": "Hepatomegaly ~60%, splenomegaly ~50% in Sandhoff. Quantify liver/spleen size (cm) at diagnosis for monitoring. Gb4 accumulation → progressive organomegaly. NOT required in Tay-Sachs (purely neurological).",
        },
        {
            "item": "LFTs (ALT, AST, GGT, Bilirubin, Albumin)",
            "frequency": "At diagnosis; 3-monthly (more frequent than standard VPA monitoring due to hepatomegaly); urgent if hepatic decompensation",
            "note": "Baseline elevated LFTs possible (Gb4 hepatic storage → hepatocyte injury). Distinguish Gb4-hepatopathy from VPA hepatotoxicity by trend. Sandhoff hepatomegaly ≠ VPA CI but warrants enhanced monitoring.",
        },
        {
            "item": "Bone Marrow Biopsy",
            "frequency": "If cytopenia (thrombocytopenia/anaemia/leukopenia) or unexplained splenomegaly",
            "note": "Bone marrow foam cells (storage histiocytes) = Sandhoff marrow involvement. Cytopenias from marrow displacement. NOT required in Tay-Sachs. Biopsy if CBC shows unexplained cytopenias.",
        },
        {
            "item": "POLG1 + mtDNA WES / Deletion Panel — Mandatory Before VPA",
            "frequency": "Once before VPA initiation",
            "note": "POLG1 Alpers + MERRF mimic Type 2 juvenile PME. POLG1 + VPA = fatal hepatotoxicity. POLG1 exclusion MANDATORY. Sandhoff hepatomegaly does NOT contraindicate VPA but POLG1 does.",
        },
        {
            "item": "Ophthalmology — Fundoscopy + ERG + VEP",
            "frequency": "At diagnosis; 6-monthly; urgent before IS treatment selection",
            "note": "Cherry-red spot: 90% Type 1 Sandhoff; <20% Type 2; absent Type 3. ERG/VEP for visual pathway. Determines ACTH vs VGB in IS. VGB HIGH RISK if cherry-red present.",
        },
        {
            "item": "Brain MRI (3T with SWI + DWI)",
            "frequency": "At diagnosis; 12-monthly if symptomatic; 24-monthly stable",
            "note": "Type 1: thalamic T2 hypointensity (similar to Tay-Sachs; GM2 thalamic storage) + WM changes. Type 2: cerebellar + posterior cortical atrophy. Hepatomegaly on abdominal sequences if MRI abdomen combined.",
        },
        {
            "item": "EEG (awake + sleep + photic)",
            "frequency": "At diagnosis; annually; urgent if clinical change or altered consciousness",
            "note": "Hypsarrhythmia (Type 1 IS); polyspike-wave generalised (Type 2 PME); focal occipital. NCSE surveillance in unexplained altered consciousness. Same EEG patterns as Tay-Sachs.",
        },
        {
            "item": "Renal Function (eGFR, urine protein, renal USS)",
            "frequency": "At diagnosis; 6-monthly (Gb4 renal tubular accumulation potential)",
            "note": "HEXB renal Gb4 tubular involvement in Type 1 Sandhoff. Monitor eGFR. Proteinuria if tubular damage. Dose-adjust piracetam + LEV if eGFR falls. NOT required in Tay-Sachs monitoring.",
        },
        {
            "item": "VPA TDM + Enhanced LFT + FBC (Sandhoff-Enhanced Protocol)",
            "frequency": "TDM: 6-weekly until stable then 6-monthly; LFTs: 3-monthly (enhanced vs standard 6-monthly); FBC: 3-monthly",
            "note": "Enhanced monitoring vs Tay-Sachs: Sandhoff hepatomegaly warrants 3-monthly LFTs on VPA. Baseline Gb4-LFT elevation must be documented to distinguish from VPA hepatotoxicity by trend.",
        },
        {
            "item": "Neuropsychology / Cognitive Assessment",
            "frequency": "Annually",
            "note": "Cognitive trajectory in Type 2 (progressive decline). Vineland/VABS in children. Type 3 psychiatric features (depression, psychosis) preceding neurological signs — baseline neuropsychiatric assessment.",
        },
        {
            "item": "Gene Therapy Trial Eligibility + NTSAD / BDSRA Registry",
            "frequency": "At diagnosis (urgent); 6-monthly if pre-symptomatic",
            "note": "AAV9-HEXA/HEXB bicistronic Phase I/II 2024 — same vector treats both Sandhoff and Tay-Sachs. Best outcomes pre-symptomatic. Urgent referral at diagnosis. NTSAD (National Tay-Sachs & Allied Diseases) includes Sandhoff.",
        },
        {
            "item": "Hepatology Co-Management",
            "frequency": "At diagnosis (referral); 6-monthly if significant hepatomegaly",
            "note": "Sandhoff-specific: hepatomegaly management requires hepatology co-management. NOT required in Tay-Sachs. Hepatic monitoring protocol, drug PK adjustments, and assessment of hepatic Gb4 trajectory.",
        },
    ]

    lifecycle_stages = [
        {
            "stage": "Prenatal / Pre-Symptomatic (Sibling Risk — No Ethnic Carrier Screen)",
            "age_range": "Pre-conception — birth (at-risk families)",
            "description": (
                "NO routine ethnic carrier screen for Sandhoff (unlike Tay-Sachs AJ screens). "
                "At-risk families identified by affected sibling → PGT-M, amniocentesis Hex A+B assay + HEXB gene. "
                "Spanish/Latin-American communities: c.1514_1517delCTCA founder carrier screen available."
            ),
            "priorities": [
                "HEXB gene sequencing of parents (carrier status)",
                "PGT-M referral for at-risk couples",
                "Amniocentesis Hex A+B assay + HEXB gene at risk",
                "Prenatal abdominal USS (hepatomegaly may be visible in severe Type 1)",
                "Gene therapy trial pre-registration (pre-symptomatic window)",
            ],
        },
        {
            "stage": "Neonatal / First Months (Type 1 Infantile — Onset 3-6m)",
            "age_range": "0–6 months",
            "description": (
                "Type 1 onset: hyperekplexia (exaggerated startle); hepatosplenomegaly on exam "
                "(DISTINCTIVE from Tay-Sachs which is purely neurological); cherry-red spot; "
                "macrocephaly; developmental plateau → regression."
            ),
            "priorities": [
                "Urgent Hex A+B leukocyte assay + HEXB gene (CNV/MLPA if Lebanese/Arab family)",
                "Fundoscopy (cherry-red spot) — before IS treatment selection",
                "Abdominal USS (hepatosplenomegaly quantification)",
                "LFT baseline (Gb4 hepatic storage — distinguish from VPA before starting)",
                "ACTH for IS (NOT VGB if cherry-red)",
                "Urgent gene therapy referral (NTSAD)",
            ],
        },
        {
            "stage": "Early Childhood — Infantile Spasms to Myoclonus Transition (Type 1)",
            "age_range": "6 months — 2 years",
            "description": (
                "IS → myoclonus → tonic transition; hepatosplenomegaly progression; "
                "marrow involvement (cytopenia check); swallowing decline; gastrostomy. "
                "More organ surveillance required than Tay-Sachs."
            ),
            "priorities": [
                "ACTH → VPA transition for post-IS myoclonus",
                "3-monthly LFTs (hepatomegaly + VPA)",
                "CBC (bone marrow foam cells — cytopenia monitor)",
                "Swallowing FEES → gastrostomy planning",
                "Hepatology co-management initiation",
                "Palliative care introduction",
            ],
        },
        {
            "stage": "School Age — PME Onset (Type 2 Juvenile Primary Epilepsy Window)",
            "age_range": "2–10 years (Type 2 onset mean 4-5y)",
            "description": (
                "Type 2 juvenile: ataxia + action myoclonus + GTCS + mild hepatomegaly + cognitive decline. "
                "Primary epilepsy management period. Hepatomegaly requires hepatology co-management. "
                "Risk of CBZ/OXC prescribing by non-specialist (Type 2 GTCS = GGE/JME mimic)."
            ),
            "priorities": [
                "Confirm Hex A+B (leukocyte) + HEXB gene",
                "VPA + LEV initiation (ABSOLUTE CI CBZ; enhanced LFT monitoring)",
                "POLG1/MERRF exclusion before VPA",
                "Piracetam for action myoclonus (renal function check)",
                "Hepatology co-management (hepatomegaly progression)",
                "School support plan (LD + seizure management)",
                "Ophthalmology ERG/VEP baseline",
            ],
        },
        {
            "stage": "Adolescence — Drug-Resistant PME + Visceral Complications (Type 2)",
            "age_range": "10–25 years (Type 2 advanced)",
            "description": (
                "Progressive cognitive decline; increasing myoclonus; VPA + LEV + Piracetam; "
                "hepatosplenomegaly progression; renal function monitoring; KD consideration. "
                "DVLA cessation; transition to adult services (neurology + hepatology)."
            ),
            "priorities": [
                "DVLA cessation (seizures + cognitive decline)",
                "KD referral if drug-resistant (hepatic monitoring during KD)",
                "Hepatology annual review (Gb4 hepatopathy trajectory)",
                "Renal USS + eGFR (Gb4 renal accumulation)",
                "Adult transition planning (neurology + hepatology)",
                "SUDEP counselling + nocturnal monitoring",
            ],
        },
        {
            "stage": "Late Stage — Palliative / End-of-Life",
            "age_range": "Type 1: 2–4y; Type 2: late teens; Type 3: variable",
            "description": (
                "End-stage: severe neurological disability + hepatosplenomegaly + aspiration. "
                "Comfort-directed care. Hepatic failure risk if marrow/visceral disease advanced. "
                "More complex palliative management than Tay-Sachs due to systemic involvement."
            ),
            "priorities": [
                "Comfort-directed seizure management (rescue midazolam)",
                "Hepatic supportive care (Gb4 hepatopathy end-stage monitoring)",
                "Gastrostomy + chest physio (aspiration prevention)",
                "Palliative sedation for refractory myoclonus",
                "Bereavement support (NTSAD includes Sandhoff families)",
                "Research / registry contribution",
            ],
        },
    ]

    return {
        "etiologies": etiologies,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "treatments": treatments,
        "contraindications": contraindications,
        "monitoring": monitoring,
        "lifecycle_stages": lifecycle_stages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_definitions():
    return {
        "disease_name": "Sandhoff Disease / GM2 Gangliosidosis Type 2 (HEXB Deficiency)",
        "gene_full": "HEXB (β-Hexosaminidase B β-subunit) — 5q13.3; 14 exons; 556 aa precursor; TIM barrel fold; shared β-subunit of Hex A (αβ), Hex B (ββ), Hex S",
        "omim_gene": "*606873 (HEXB gene)",
        "omim_disease": "#268800 (Sandhoff Disease / GM2 Gangliosidosis Type 2)",
        "protein_full": (
            "β-Hexosaminidase β-subunit (HEXB): 556 aa precursor; signal peptide 1-38; propeptide 39-57 (removed in lysosome); "
            "mature β-subunit ~60 kDa; TIM barrel catalytic domain; catalytic Glu355 (acid/base) + Glu491 (nucleophile); "
            "retaining double-displacement mechanism; "
            "beta-subunit SHARED by ALL THREE hexosaminidase forms: "
            "Hex A (αβ heterodimer) — cleaves GM2 ganglioside (requires GM2AP); "
            "Hex B (ββ homodimer) — cleaves globoside Gb4 in visceral organs (does NOT require GM2AP); "
            "Hex S (αα homodimer) — minor GM2 cleavage. "
            "HEXB LOF → ALL THREE forms deficient (Hex A + Hex B + Hex S all zero)."
        ),
        "inheritance_mode": "Autosomal Recessive (AR) — biallelic LOF required; pLI ~0.84 (moderate-high LOF intolerance)",
        "onset_age": "Type 1 Infantile: 3–6 months; Type 2 Juvenile: 2–10 years (mean 4-5y); Type 3 Adult/Chronic: adolescence–40y",
        "multienzyme_triad_note": (
            "Three GM2 gangliosidoses: "
            "(1) HEXB LOF (Sandhoff): ALL THREE hex forms deficient (Hex A + Hex B + Hex S all low = PATHOGNOMONIC). "
            "Systemic Gb4 accumulation → hepatosplenomegaly + bone marrow foam cells. "
            "Leukocyte Hex A LOW + Hex B LOW = Sandhoff. "
            "(2) HEXA LOF (Tay-Sachs): Hex A LOW; Hex B ELEVATED (ββ retained). Purely neurological — no visceral involvement. "
            "(3) GM2A LOF (AB variant): Hex A + Hex B BOTH NORMAL; GM2AP protein absent — in vitro normal; in vivo GM2 accumulates. "
            "DIAGNOSTIC RULE: Leukocyte 4-MU assay measures Hex A AND Hex B simultaneously. "
            "Hex A low + Hex B LOW = Sandhoff. Hex A low + Hex B HIGH = Tay-Sachs. Both normal = AB variant."
        ),
        "sandhoff_vs_taysachs_key_differences": [
            {"feature": "Hex B level", "sandhoff": "LOW (ββ deficient)", "taysachs": "HIGH (ββ retained, ELEVATED)"},
            {"feature": "Hepatosplenomegaly", "sandhoff": "YES ~70% Type 1 (Gb4 visceral accumulation)", "taysachs": "ABSENT (purely neurological)"},
            {"feature": "Bone marrow foam cells", "sandhoff": "YES (storage histiocytes)", "taysachs": "ABSENT"},
            {"feature": "Founder mutations", "sandhoff": "No AJ founder; Spanish c.1514_1517delCTCA", "taysachs": "AJ c.1278insTATC (78%) — strong AJ founder"},
            {"feature": "Ethnic carrier screening", "sandhoff": "No routine screen (no major AJ founder)", "taysachs": "AJ three-mutation panel standard"},
            {"feature": "Gene therapy vector", "sandhoff": "AAV9-HEXA/HEXB bicistronic (same vector)", "taysachs": "AAV9-HEXA/HEXB bicistronic (same vector)"},
            {"feature": "Pharmacological rules", "sandhoff": "CBZ/OXC/PHT ABSOLUTE CI; VPA safe; ACTH Level A IS", "taysachs": "CBZ/OXC/PHT ABSOLUTE CI; VPA safe; ACTH Level A IS"},
        ],
        "concepts": [
            {
                "name": "HEXB-5q13.3-ALL-THREE-Hex-Forms-Deficient-Pathognomonic-Sandhoff",
                "definition": "HEXB encodes shared β-subunit of Hex A (αβ), Hex B (ββ), Hex S; HEXB LOF → ALL three deficient simultaneously. Leukocyte Hex A LOW + Hex B LOW = PATHOGNOMONIC Sandhoff. Contrast Tay-Sachs: Hex B ELEVATED.",
            },
            {
                "name": "Hex-A-Low-Hex-B-LOW-PATHOGNOMONIC-Sandhoff-vs-Hex-B-HIGH-Tay-Sachs",
                "definition": "In Sandhoff: Leukocyte Hex A LOW + Hex B LOW (all hex forms zero). In Tay-Sachs: Hex A LOW + Hex B ELEVATED (ββ retained). This single distinction (Hex B level) instantaneously differentiates Sandhoff from Tay-Sachs at the enzyme assay level.",
            },
            {
                "name": "Hepatosplenomegaly-Bone-Marrow-Foam-Cells-DISTINCTIVE-Sandhoff-Not-Tay-Sachs",
                "definition": "Hex B (ββ) absent in Sandhoff → globoside Gb4 accumulates in liver (~60%), spleen (~50%), kidney, bone marrow. Hepatosplenomegaly + marrow foam cells = Sandhoff-DISTINCTIVE feature. Tay-Sachs is purely neurological (Hex B retained in visceral organs).",
            },
            {
                "name": "No-AJ-Founder-Mutation-Standard-Tay-Sachs-Screen-Does-NOT-Detect-Sandhoff",
                "definition": "Sandhoff has NO major Ashkenazi Jewish founder mutation (unlike Tay-Sachs c.1278insTATC 78% AJ). Standard AJ Tay-Sachs carrier screens do NOT detect Sandhoff carriers. Enzyme assay (Hex A+B leukocyte) is the primary diagnostic pathway. Spanish/Latin-American c.1514_1517delCTCA most prominent founder.",
            },
            {
                "name": "CBZ-OXC-PHT-ABSOLUTE-CI-Type2-PME-GTCS-Misidentification",
                "definition": "Type 2 juvenile GTCS misidentified as GGE/JME → CBZ initiated → acute myoclonic storm. Na-channel blockers ABSOLUTE CI in all Sandhoff PME phenotypes. 3.2y mean diagnosis delay = extended CBZ exposure risk window. Same rule as Tay-Sachs.",
            },
            {
                "name": "VPA-SAFE-HEXB-Lysosomal-NOT-Mitochondrial-Hepatomegaly-Does-NOT-CI-VPA",
                "definition": "HEXB lysosomal acid hydrolase — NOT mitochondrial. VPA CI = MERRF/POLG only. Sandhoff hepatomegaly (Gb4 storage) does NOT contraindicate VPA — Gb4 hepatopathy ≠ mitochondrial failure. POLG1/MERRF exclusion MANDATORY before VPA. Enhanced LFT monitoring warranted.",
            },
            {
                "name": "ACTH-Level-A-Infantile-Spasms-Cherry-Red-90pct-Type1-VGB-HIGH-RISK",
                "definition": "Type 1 IS: cherry-red spot 90% → VGB HIGH RISK (GM2 retinal storage + VGB retinopathy). ACTH Level A (UKISS/INFANT protocol) preferred. Fundoscopy MANDATORY before IS treatment. If cherry-red absent (rare) → VGB acceptable with ERG monitoring.",
            },
            {
                "name": "Fosphenytoin-ABSOLUTE-CI-IV-LEV-Replaces-SE-Protocol",
                "definition": "IV fosphenytoin = ABSOLUTE CI in Sandhoff SE (myoclonic worsening). REPLACE WITH IV LEV 60 mg/kg. Pre-populate A&E drug chart. SE protocol: midazolam → IV LEV → IV VPA (POLG1 excluded; hepatomegaly — LFT monitoring) → anaesthesia.",
            },
            {
                "name": "Piracetam-Level-B-Action-Myoclonus-Renal-Monitoring-Enhanced",
                "definition": "Piracetam Level B for action myoclonus Type 2. Renal monitoring MORE IMPORTANT in Sandhoff (HEXB renal Gb4 tubular accumulation). Assess eGFR before high-dose piracetam. Dose-reduce if eGFR falls. Same AMPA/membrane fluidity MOA as in Tay-Sachs.",
            },
            {
                "name": "HEXB-CNV-MLPA-Mandatory-Lebanese-Maronite-Exonic-Deletion",
                "definition": "Lebanese/Maronite Sandhoff: biallelic HEXB exonic deletion — founder variant. WES sequencing alone may MISS large deletions. MLPA or CNV analysis MANDATORY in Lebanese/Arab families with biochemical Sandhoff and negative WES sequencing. CNV analysis reveals deletion.",
            },
            {
                "name": "AAV9-HEXA-HEXB-Bicistronic-Same-Vector-Treats-Both-Sandhoff-And-TaySachs",
                "definition": "AAV9-HEXA/HEXB bicistronic vector encodes BOTH α-subunit (HEXA) and β-subunit (HEXB) simultaneously → treats BOTH Tay-Sachs AND Sandhoff with single vector. Sandhoff benefits from HEXB component restoring Hex B (visceral Gb4) AND Hex A (CNS GM2). Phase I/II 2024.",
            },
            {
                "name": "Hepatology-Co-Management-MANDATORY-Sandhoff-Not-Required-Tay-Sachs",
                "definition": "Sandhoff hepatomegaly (Gb4 storage) requires hepatology co-management — abdominal USS 12-monthly, LFTs 3-monthly, bone marrow if cytopenia. Tay-Sachs is purely neurological — hepatology NOT required. This distinction is clinically critical for MDT composition.",
            },
            {
                "name": "Globoside-Gb4-Systemic-Accumulation-HEXB-Distinguishes-Sandhoff-Substrate",
                "definition": "Hex B (ββ homodimer) cleaves globoside Gb4 (N-acetylgalactosamine terminal) in visceral organs — does NOT require GM2AP. HEXB LOF → Gb4 accumulates in liver, spleen, kidney, bone marrow, Schwann cells. Dual substrate accumulation (GM2 in CNS + Gb4 in viscera) = pathological basis of systemic Sandhoff.",
            },
            {
                "name": "SUDEP-Risk-Drug-Resistant-Type2-Nocturnal-GTCS-Enhanced",
                "definition": "SUDEP risk elevated in Sandhoff Type 2 with drug-resistant nocturnal GTCS + progressive neurodegeneration + possible cardiac arrhythmia (Gb4 cardiac involvement rare but reported). Nocturnal monitoring (bed sensor, SpO2). SUDEP counselling at annual review.",
            },
            {
                "name": "Visceral-Screening-Battery-Mandatory-Sandhoff-vs-Tay-Sachs",
                "definition": "Sandhoff visceral battery: Abdominal USS (liver/spleen) + LFTs + CBC (marrow) + eGFR + renal USS. Required at diagnosis and 12-monthly. NOT required in Tay-Sachs (purely neurological). This systematic visceral surveillance differentiates Sandhoff management from Tay-Sachs.",
            },
        ],
        "thresholds": [
            {"parameter": "Leukocyte Hex A (% total hex)", "value": "<3% = Sandhoff if Hex B also low; Hex B LOW confirms Sandhoff vs Tay-Sachs", "action": "Hex A LOW + Hex B LOW: confirm HEXB gene; urgent cascade + gene therapy referral"},
            {"parameter": "Leukocyte Hex B (% total hex)", "value": "<3% = Sandhoff (vs elevated Hex B = Tay-Sachs)", "action": "Hex B LOW: distinguishes Sandhoff from Tay-Sachs instantly; HEXB gene sequencing + CNV"},
            {"parameter": "Liver size (USS cm)", "value": ">2 cm above ULN for age = hepatomegaly; >4 cm = significant hepatomegaly", "action": "Significant hepatomegaly: hepatology referral; 3-monthly LFTs on VPA; enhanced drug monitoring"},
            {"parameter": "ALT on VPA (hepatomegaly context)", "value": "Baseline Gb4-LFT elevation must be documented; >2× baseline = VPA hepatotoxicity suspect; >5× ULN = stop VPA", "action": "Monitor trend not absolute; distinguish Gb4 baseline from VPA rise; hepatology co-management"},
            {"parameter": "VPA trough (mg/L)", "value": "Target: 50-100 mg/L; >120 mg/L = toxic", "action": ">100: dose reduce; <40: uptitrate; enhanced monitoring in hepatomegaly (altered PK)"},
            {"parameter": "eGFR (piracetam + LEV dosing)", "value": "<50 mL/min = dose reduction required for both piracetam and LEV", "action": "Renal USS + eGFR at diagnosis and 6-monthly; dose-adjust LEV and piracetam accordingly"},
            {"parameter": "CBC (bone marrow monitoring)", "value": "Thrombocytopenia <100 × 10⁹/L or anaemia Hb <100 g/L = bone marrow involvement suspected", "action": "Bone marrow biopsy to confirm foam cells; haematology referral; adjust AEDs affecting platelets"},
            {"parameter": "IS cluster duration (SE threshold)", "value": ">5 spasms in cluster + EEG hypsarrhythmia = urgent intervention", "action": "ACTH escalation; IV LEV if SE (not fosphenytoin); urgent neurology + hepatology (Sandhoff)"},
            {"parameter": "GTCS duration (SE threshold)", "value": ">5 min single GTCS = SE", "action": "Buccal midazolam → IV LEV 60 mg/kg → IV VPA (POLG1 excluded; LFT check) → anaesthesia"},
            {"parameter": "Cherry-red spot presence", "value": "Present = HIGH RISK VGB; absent = VGB acceptable in IS", "action": "Fundoscopy before IS treatment; ACTH if cherry-red (not VGB Type 1)"},
            {"parameter": "DVLA seizure-free period", "value": "UK: 12 months seizure-free = Group 1 licence resumption possible", "action": "Seizure within 12 months → must not drive; DVLA notification"},
            {"parameter": "AAV9 trial eligibility window", "value": "Pre-symptomatic or <12 months symptomatic = best window for gene therapy benefit", "action": "Urgent specialist referral (NTSAD/Boston/Manchester/Toronto) at diagnosis; do not delay"},
        ],
        "standards": [
            "Sandhoff K et al. (1968) FEBS Lett — first description of total hexosaminidase (Hex A + Hex B) deficiency distinguishing Sandhoff from Tay-Sachs",
            "O'Brien JS et al. (1970) N Engl J Med — enzymatic basis of Sandhoff disease; Hex A + Hex B both deficient",
            "Neote K et al. (1990) Am J Hum Genet — HEXB gene structure; 14 exons; 5q13 locus; molecular basis",
            "Gravel RA et al. (2001) OMMBID Ch 153 — comprehensive GM2 gangliosidosis molecular review",
            "Andermann BA et al. (1977) Neurology — Sandhoff disease clinical characterisation; hepatosplenomegaly distinction from Tay-Sachs",
            "ILAE (2022) PME Classification — Scheffer IE et al.; progressive myoclonic epilepsy classification framework",
            "NICE NG217 (2022) Epilepsy Guideline — AED treatment standards; VPA PREVENT programme",
            "MHRA VPPP (2021) — Valproate Patient Pregnancy Prevention Programme",
            "CPIC POLG1 Guideline (2023) — VPA contraindicated in POLG1 pathogenic variants; mandatory exclusion",
            "Picone M et al. (2024) Phase I/II AAV9-HEXA/HEXB bicistronic gene therapy — targets both Tay-Sachs and Sandhoff",
            "ACMG-AMP (2015) Variant Classification Standards — used for HEXB pathogenic variant interpretation",
            "NTSAD Registry (2024) — National Tay-Sachs & Allied Diseases Association; includes Sandhoff natural history data",
        ],
        "references": [
            "Sandhoff K et al. (1968) Deficient hexosaminidase activity in an exceptional case of Tay-Sachs disease with additional storage of kidney globoside in visceral organs. Life Sci 7:283-8.",
            "O'Brien JS et al. (1970) Tay-Sachs disease: prenatal diagnosis. Science 172:61-4.",
            "Neote K et al. (1990) Characterisation of the gene encoding the human beta-hexosaminidase B alpha-chain. J Biol Chem 265:20799-806.",
            "Gravel RA et al. (2001) The GM2 gangliosidoses. In: Scriver CR et al. The Metabolic and Molecular Bases of Inherited Disease (8th ed). McGraw-Hill: 3827-76.",
            "Kytzia HJ & Sandhoff K (1985) Evidence for two different active sites on human beta-hexosaminidase A. Interaction of GM2 activator protein with beta-hexosaminidase A and B. J Biol Chem 260:7568-72.",
            "Delnooz CC et al. (2010) New cases of adult-onset Sandhoff disease with a cerebellar or lower motor neuron phenotype. J Neurol Neurosurg Psychiatry 81:968-72.",
        ],
    }
