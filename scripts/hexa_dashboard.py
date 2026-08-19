"""
HEXA Epilepsy — Tay-Sachs Disease / GM2 Gangliosidosis Type 1
==============================================================
40-patient cohort · HEXA (15q23) · Autosomal recessive (AR) biallelic LOF
HEXA encodes β-Hexosaminidase A α-subunit; 529 aa precursor ~60 kDa (signal peptide 1-22;
propeptide 23-42; mature α-subunit ~56 kDa after lysosomal processing); TIM barrel fold;
pairs with β-subunit (HEXB gene product) to form heterodimer Hex A (αβ), the main
GM2-cleaving enzyme. HEXA LOF → Hex A (αβ) and Hex S (αα) deficient; Hex B (ββ) retained
and elevated (discriminates Tay-Sachs from Sandhoff where ALL three forms lost).

HEXA FUNCTION:
  Hex A (αβ heterodimer) = primary enzyme for GM2 ganglioside hydrolysis
  Requires GM2 activator protein (GM2AP, gene GM2A) to load GM2 into active site
  Hex A cleaves N-acetylgalactosamine (β-1,4) from GM2 ganglioside → GM3 ganglioside
  HEXA LOF → GM2 ganglioside accumulates in CNS neurons → neurodegeneration
  Hex B (ββ homodimer): cleaves sulfated substrates (globoside Gb4) — preserved in Tay-Sachs
  Elevated Hex B (total hex HIGH; Hex A fraction LOW) = PATHOGNOMONIC of Tay-Sachs

HEXA LOF — THREE PHENOTYPIC FORMS (RESIDUAL ENZYME ACTIVITY DETERMINES SEVERITY):
════════════════════════════════════════════════════════════════════════════════════
(1) TYPE 1 (CLASSIC INFANTILE, SEVERE): <0.1% residual Hex A; onset 3-6 months;
    hyperekplexia (exaggerated startle); cherry-red macular spot (90-95%);
    infantile spasms (3-7m) → myoclonus → tonic → polyspike-wave;
    progressive macrocephaly (3-18m); rapid neurodegeneration; death by 2-5y.
    Null alleles (frameshift/splice/nonsense). Cherry-red + exaggerated startle = CLASSIC.
(2) TYPE 2 (JUVENILE, SUBACUTE): 2-8% residual Hex A; onset 2-10y (mean 5y);
    progressive cerebellar ataxia + myoclonic epilepsy (PME phenotype) + cognitive decline;
    cherry-red spot ABSENT or rare (<10%); survival to teens/young adult;
    EPILEPSY IS MAJOR MANAGEMENT CHALLENGE — myoclonus + GTCS + absence-like.
    Missense alleles with partial residual Hex A activity.
(3) TYPE 3 (ADULT/CHRONIC): >10% residual Hex A; onset adolescence-40y;
    spinocerebellar syndrome + lower motor neuron signs + psychiatric features;
    seizures less prominent; PME features in some; survival to adulthood.
    p.Gly269Ser when compound heterozygous with null allele = late-onset adult.
    OMIM: *606869 (HEXA gene) / #272800 (Tay-Sachs / GM2 gangliosidosis Type 1)

PSEUDODEFICIENCY — CRITICAL TRAP:
════════════════════════════════
p.Arg247Trp or p.Arg249Trp: DBS Hex A assay shows LOW Hex A (<3%),
but LEUKOCYTE assay is NORMAL — these are PSEUDODEFICIENCY variants.
Carriers are NOT at risk; their children are NOT at risk for Tay-Sachs.
Misclassification leads to unnecessary genetic counselling + terminations.
RULE: Always confirm DBS low Hex A with LEUKOCYTE Hex A before counselling.
DBS Hex A falsely low in females on oral contraceptive pills (OCP) — use leukocyte assay.

HEXA vs HEXB vs GM2A (GM2AP) — CRITICAL DIAGNOSTIC TRIAD:
══════════════════════════════════════════════════════════
  HEXA LOF (Tay-Sachs): Hex A LOW + Hex B ELEVATED (ββ retained)
  HEXB LOF (Sandhoff):  Hex A LOW + Hex B LOW + Hex S LOW (all forms deficient)
  GM2A LOF (AB variant): Hex A NORMAL; Hex B NORMAL; GM2AP protein deficient → can't load GM2
  DIAGNOSIS: Leukocyte 4-MU-hexosaminidase A and B assay SIMULTANEOUSLY.
  Do NOT use DBS alone — pseudodeficiency and OCP interference invalidate DBS.

FOUNDER MUTATIONS — MAJOR CLINICAL SIGNIFICANCE:
════════════════════════════════════════════════
Ashkenazi Jewish (carrier freq 1/27 AJ population; general pop 1/300):
  c.1278insTATC (p.Tyr427IlefsTer5): FRAMESHIFT; 78% of AJ alleles → null; classic infantile
  c.1421+1G>C (IVS12+1G>C): SPLICE SITE; 18% of AJ alleles → null; classic infantile
  p.Gly269Ser: MISSENSE; 4% AJ; partial activity → juvenile/adult if compound with null
  Three-mutation AJ panel covers ~98% of AJ carriers.

French-Canadian (carrier freq 1/30; Quebec origin):
  c.1278+1G>A (IVS7+1G>A): SPLICE SITE; 100% of French-Canadian alleles → null
  Note: different mutation from AJ despite similar carrier frequency.

Irish/British: c.IVS9+1G>A (different splice from FC); enriched in Irish-American families.
Japanese: p.Gly269Ser (juvenile/adult when compound; classic when homozygous)

KEY PHARMACOLOGICAL DISTINCTIONS — GM2 GANGLIOSIDOSIS / HEXA EPILEPSY:
══════════════════════════════════════════════════════════════════════
(1) CBZ/OXC/PHT ABSOLUTE CI — myoclonic worsening (Type 2 juvenile PME misidentified as GGE/JME → CBZ → acute myoclonic storm)
(2) VPA SAFE — lysosomal NOT mitochondrial; POLG1/MERRF mandatory exclusion before VPA
(3) ACTH Level A for infantile spasms (Type 1, West syndrome) — prefer over VGB when cherry-red present
(4) VGB HIGH RISK Type 1 (cherry-red spot; retinal ganglion cell storage + VGB retinopathy = catastrophic bilateral blindness)
(5) VGB acceptable in Type 3 (no cherry-red spot, no retinal involvement)
(6) Fosphenytoin ABSOLUTE CI → replace with IV LEV 60 mg/kg in SE protocols
(7) Piracetam Level B for action myoclonus (Type 2 juvenile); Clonazepam Level B nocturnal myoclonus
(8) No approved disease-modifying therapy; AAV9-HEXA gene therapy Phase I/II (2024)
(9) Pseudodeficiency variants (p.Arg247Trp/p.Arg249Trp) — confirm ALL DBS low with leukocyte assay
"""

import random
random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def get_overview():
    return {
        "disease": (
            "Tay-Sachs Disease (GM2 Gangliosidosis Type 1) — HEXA LOF → Hex A (αβ heterodimer) deficient; "
            "GM2 ganglioside accumulates in CNS neurons → neurodegeneration. "
            "Three forms: Type 1 Infantile (classic; <0.1% residual Hex A; cherry-red + hyperekplexia + IS; fatal by 2-5y); "
            "Type 2 Juvenile (2-8% residual; PME + ataxia; survival teens); "
            "Type 3 Adult/Chronic (>10% residual; spinocerebellar + LMN + psychiatric). "
            "This cohort focuses on Type 2 Juvenile (70%) — the primary epilepsy management population."
        ),
        "gene": "HEXA (15q23)",
        "protein": (
            "β-Hexosaminidase A α-subunit (HEXA); 529 aa precursor ~60 kDa; TIM barrel fold; "
            "pairs with HEXB β-subunit to form Hex A (αβ) heterodimer. "
            "HEXA LOF → Hex A (αβ) + Hex S (αα) both deficient; Hex B (ββ) elevated = PATHOGNOMONIC Tay-Sachs. "
            "Catalytic residue: Glu323 (acid/base) + Glu325 (nucleophile); retaining mechanism; "
            "requires GM2 activator protein (GM2AP) to load GM2 substrate into active site."
        ),
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF; pLI ~0.81 moderate intolerance",
        "omim": "*606869 (HEXA gene) / #272800 (Tay-Sachs Disease / GM2 Gangliosidosis Type 1)",
        "mechanism": (
            "HEXA LOF → Hex A (αβ heterodimer) unable to cleave N-acetylgalactosamine from GM2 ganglioside "
            "→ GM2 accumulates in lysosomes of CNS neurons (especially thalamus, basal ganglia, cerebellum, cortex) "
            "→ neuronal swelling (Ballon cells) + axonal torpedoes + dendritic ectopia "
            "→ progressive neurodegeneration; cherry-red macular spot (retinal ganglion cell GM2 storage + "
            "foveal transparency reveals underlying choroidal red against white GM2-laden perifoveal ring)."
        ),
        "cohort_size": 40,
        "mean_onset_years": 4.8,
        "seizure_pct": 88,
        "myoclonus_pct": 72,
        "infantile_spasms_pct": 35,
        "dystonia_pct": 28,
        "drug_resistant_pct": 62,
        "mean_diagnosis_delay_years": 2.9,
        "cherry_red_spot_pct": 42,
        "type2_juvenile_pct": 70,
        "type3_adult_pct": 15,
        "type1_infantile_pct": 15,
        "ashkenazi_jewish_pct": 52,
        "french_canadian_pct": 18,
        "on_vpa_pct": 72,
        "on_lev_pct": 65,
        "on_acth_pct": 28,
        "on_piracetam_pct": 45,
        "on_clonazepam_pct": 38,
        "discovery": (
            "Waren Tay (1881 Lancet) — described cherry-red spot in infant with progressive neurodegeneration. "
            "Bernard Sachs (1887 J Nerv Ment Dis) — described 'amaurotic family idiocy' as distinct entity. "
            "Okada & O'Brien (1969 Science) — identified β-hexosaminidase A deficiency as enzymatic basis. "
            "Kaback et al. (1974 Prog Med Genet) — established AJ carrier screening programme (landmark public health success). "
            "Myerowitz R (1988 PNAS) — identified frameshift c.1278insTATC as major AJ allele. "
            "Triggs-Raine et al. (1990 NEJM) — IVS12+1G>C as second major AJ allele. "
            "AJ carrier screening (1970s–present) has reduced classic Tay-Sachs incidence by >90% in AJ population."
        ),
        "unique_feature": (
            "HEXA/Tay-Sachs occupies a unique intersection: (1) the most successful inherited disease prevention programme "
            "in history (AJ carrier screening reducing incidence >90%); (2) pseudodeficiency trap "
            "(p.Arg247Trp/p.Arg249Trp → DBS falsely low; leukocyte assay mandatory before counselling); "
            "(3) Hex B ELEVATED in Tay-Sachs (ββ retained) vs Hex B LOW in Sandhoff (HEXB LOF) = "
            "rapid bedside discrimination; (4) AAV9-HEXA/HEXB bicistronic gene therapy Phase I/II 2024 "
            "— most biologically rational approach restoring both subunits simultaneously."
        ),
        "hexa_hexb_gm2a_differential_note": (
            "Three GM2 gangliosidoses — critical diagnostic triad: "
            "(1) HEXA LOF (Tay-Sachs): Leukocyte Hex A LOW (<1-3% of total); Hex B ELEVATED (ββ retained — PATHOGNOMONIC). "
            "(2) HEXB LOF (Sandhoff): Leukocyte Hex A LOW + Hex B LOW + Hex S LOW (all three forms deficient; "
            "more severe systemic involvement: hepatosplenomegaly + bone marrow foam cells). "
            "(3) GM2A LOF (AB variant): Leukocyte Hex A NORMAL + Hex B NORMAL; GM2 activator protein (GM2AP) deficient — "
            "cannot load GM2 into Hex A active site; enzyme activity in vitro NORMAL (artificial substrate); "
            "in vivo GM2 accumulates as if enzyme absent; rare; confirm with GM2AP protein assay or GM2A gene. "
            "RULE: Measure Hex A AND Hex B simultaneously in leukocytes. If Hex A low + Hex B elevated → Tay-Sachs. "
            "Do NOT use DBS alone (pseudodeficiency + OCP interference in females)."
        ),
        "pseudodeficiency_trap_note": (
            "PSEUDODEFICIENCY VARIANTS — p.Arg247Trp (c.739C>T) and p.Arg249Trp (c.745C>T): "
            "These missense variants reduce Hex A activity toward the ARTIFICIAL fluorescent substrate 4-MU-GlcNAc "
            "used in DBS assays — DBS Hex A appears LOW (<3% of total). "
            "HOWEVER: these variants do NOT impair Hex A catalytic activity toward the NATURAL substrate GM2 "
            "(requires GM2 activator protein for loading). "
            "Leukocyte Hex A assay with natural GM2 substrate = NORMAL. "
            "p.Arg247Trp alone = pseudodeficiency carrier; p.Arg247Trp + null allele = pseudodeficiency compound het → NORMAL neurological function. "
            "Clinical disaster: misclassification → false carrier counselling → unnecessary reproductive decisions. "
            "RULE: CONFIRM ALL DBS LOW HEX A WITH LEUKOCYTE ASSAY BEFORE GENETIC COUNSELLING. "
            "Additional trap: females on OCP → serum Hex A falsely low; use leukocyte assay (not serum) in women of reproductive age."
        ),
        "type2_type3_differential_note": (
            "Type 2 Juvenile vs Type 3 Adult — epilepsy management pivot: "
            "Type 2 (2-8% residual Hex A; onset 2-10y): PME phenotype dominant — action myoclonus + GTCS + cerebellar ataxia; "
            "VPA backbone + LEV + Piracetam (action myoclonus) + Clonazepam (nocturnal); cherry-red rare (<10%); "
            "VGB NOT absolute CI but HIGH RISK caution. "
            "Type 3 Adult (>10% residual; onset adolescence-40y): spinocerebellar + LMN + psychiatric phenotype; "
            "seizures less prominent; PME features in minority; p.Gly269Ser compound het typical AJ adult; "
            "Pyrimethamine pharmacological chaperone (very limited evidence — p.Gly269Ser only); "
            "seizure treatment same pharmacological rules; psychiatric features managed with atypical antipsychotics "
            "(AVOID typical D2-blockers — motor exacerbation risk in spinocerebellar syndrome)."
        ),
        "key_pharmacological_distinctions": {
            "1_CBZ_OXC_PHT_ABSOLUTE_CI": (
                "Sodium channel blockers ABSOLUTE CI in Tay-Sachs juvenile (Type 2 PME). "
                "Type 2 GTCS misidentified as GGE/JME → CBZ initiated → acute myoclonic storm + worsening action myoclonus. "
                "Mean diagnosis delay 2.9y = extended Na-channel blocker exposure window. "
                "Atonic components additionally worsened. SAFE: VPA + LEV + Piracetam backbone."
            ),
            "2_VPA_SAFE_NOT_MITOCHONDRIAL_POLG1_EXCLUSION_MANDATORY": (
                "HEXA is a lysosomal acid hydrolase — NOT mitochondrial. "
                "VPA CI applies EXCLUSIVELY to mitochondrial disease (MERRF/POLG Alpers). "
                "VPA is SAFE backbone in Tay-Sachs. "
                "HOWEVER: POLG1 Alpers syndrome and MERRF both mimic Type 2 juvenile PME phenotype "
                "(ataxia + myoclonus + seizures). MANDATORY POLG1/MERRF exclusion by WES + mtDNA "
                "before VPA initiation in ANY juvenile PME. "
                "POLG1 Alpers + VPA = fatal hepatotoxicity (>50% mortality)."
            ),
            "3_ACTH_LEVEL_A_INFANTILE_SPASMS_TYPE1": (
                "Infantile spasms (West syndrome) in Tay-Sachs Type 1: ACTH preferred over VGB. "
                "Cherry-red spot present in 90-95% of Type 1 → VGB retinopathy (irreversible peripheral VF loss) "
                "compounded with retinal ganglion cell GM2 storage → catastrophic combined visual failure. "
                "ACTH Level A (UKISS/INFANT trials): 2mg/kg/day prednisolone equivalent OR ACTH 40-60 IU IM daily. "
                "If cherry-red absent (rare Type 1 or early before visible) → VGB acceptable with urgent ERG/VEP monitoring."
            ),
            "4_VGB_HIGH_RISK_TYPE1_NOT_ABSOLUTE_CI_TYPE3": (
                "VGB is HIGH RISK (not absolute CI) in Tay-Sachs. "
                "Type 1 (cherry-red 90-95%): retinal ganglion cell GM2 storage + VGB retinopathy risk = HIGH RISK → use ACTH first-line. "
                "Type 2 (cherry-red <10%): VGB HIGH RISK caution; monitor ERG/VEP 6-monthly if used. "
                "Type 3 (no cherry-red; no retinal involvement): VGB acceptable alternative if indicated. "
                "Contrast: CLN1/CLN2/CLN3 NCL = VGB ABSOLUTE CI (retinal NCL guarantees catastrophe). "
                "GLB1 Type 1/2 and HEXA Type 1 = VGB HIGH RISK (cherry-red macular, not retinal NCL)."
            ),
            "5_FOSPHENYTOIN_ABSOLUTE_CI_IV_LEV_REPLACES": (
                "Fosphenytoin (IV phenytoin prodrug) = ABSOLUTE CI in Tay-Sachs status epilepticus. "
                "Standard SE algorithm second-line = IV fosphenytoin → myoclonic worsening + cortical hyperexcitability. "
                "REPLACE WITH: IV LEV 60 mg/kg (max 4500 mg) — primary second-line for SE. "
                "A&E + ED must be pre-notified: 'IV LEV, NOT fosphenytoin' documented in drug chart + emergency card. "
                "ALSO: IV valproate acceptable if POLG1 excluded."
            ),
            "6_PIRACETAM_ACTION_MYOCLONUS_TYPE2": (
                "Piracetam Level B evidence for action/intention myoclonus in Type 2 juvenile. "
                "Dose: 24-45g/day in adults (weight-adjusted in children: 160-300 mg/kg/day). "
                "MOA: AMPA potentiation + membrane fluidity + anti-cortical hyperexcitability. "
                "Added as adjunct to VPA backbone when action myoclonus disabling. "
                "Monitor: renal function (dose-reduce in CKD); psychiatric side effects at high doses."
            ),
            "7_PSEUDODEFICIENCY_LEUKOCYTE_ASSAY_MANDATORY": (
                "p.Arg247Trp and p.Arg249Trp = pseudodeficiency variants; DBS Hex A appears LOW (<3%). "
                "MANDATORY: Confirm all DBS low Hex A with leukocyte Hex A assay (natural GM2 substrate). "
                "Pseudodeficiency → leukocyte Hex A NORMAL; true Tay-Sachs → leukocyte Hex A LOW. "
                "OCP interference: females on oral contraceptives → serum Hex A falsely low → use leukocyte assay. "
                "This is the single most important quality control step in Tay-Sachs carrier screening."
            ),
            "8_AAV9_BICISTRONIC_GENE_THERAPY_PHASE_I_II": (
                "AAV9-HEXA/HEXB bicistronic gene therapy (2024 Phase I/II): "
                "Most biologically rational approach — single vector encodes BOTH HEXA α-subunit and HEXB β-subunit "
                "→ restores Hex A (αβ heterodimer) simultaneously; avoids problem of α-subunit alone misfolding. "
                "Route: intrathecal (CSF) for CNS targeting. Eligible: pre-symptomatic or early symptomatic. "
                "Urgent referral at diagnosis — pre-symptomatic window closing rapidly after onset. "
                "Substrate reduction (miglustat, N-butyldeoxynojirimycin) — limited CNS penetration; not primary therapy."
            ),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def get_breakdown():
    etiologies = [
        {
            "class": "AJ-Compound-Het-c.1278insTATC-plus-IVS12+1G>C (Type 1 Infantile Classic)",
            "pct": 28,
            "description": (
                "Ashkenazi Jewish compound heterozygote: c.1278insTATC (frameshift, null) + c.1421+1G>C (splice, null). "
                "Both null alleles → <0.1% residual Hex A → classic infantile Type 1. "
                "Cherry-red spot 95%; hyperekplexia; infantile spasms 3-6m; macrocephaly; death by 2-5y."
            ),
            "typical_onset": "3–6 months (infantile spasms → progressive neurodegeneration)",
            "genotype_notes": "Two most common AJ null alleles; ~14% of all AJ Tay-Sachs compound hets",
        },
        {
            "class": "AJ-Homozygous-c.1278insTATC (Type 1 Infantile Classic)",
            "pct": 18,
            "description": (
                "Ashkenazi Jewish homozygous frameshift c.1278insTATC (p.Tyr427IlefsTer5). "
                "Null/null → <0.1% residual Hex A → classic infantile. "
                "Most severe; cherry-red spot 95%; macrocephaly; rapid fatal progression."
            ),
            "typical_onset": "3–5 months",
            "genotype_notes": "Homozygous most common single AJ genotype; 78% of AJ alleles individually",
        },
        {
            "class": "AJ-Compound-Het-Null-plus-p.Gly269Ser (Type 2 Juvenile)",
            "pct": 22,
            "description": (
                "Ashkenazi Jewish compound heterozygote: null allele (c.1278insTATC or IVS12+1G>C) + p.Gly269Ser (missense). "
                "p.Gly269Ser retains 2-8% residual Hex A → juvenile phenotype (onset 2-10y). "
                "Progressive myoclonic epilepsy + cerebellar ataxia + cognitive decline; cherry-red rare (<10%)."
            ),
            "typical_onset": "2–10 years (mean 5 years)",
            "genotype_notes": "p.Gly269Ser = 4% of AJ alleles; juvenile when compound with null; adult when homozygous",
        },
        {
            "class": "French-Canadian-IVS7+1G>A-Homozygous-or-Compound (Type 1 Infantile)",
            "pct": 15,
            "description": (
                "French-Canadian splice variant c.1278+1G>A (IVS7+1G>A) — 100% of FC alleles; null allele. "
                "Homozygous or compound with other null → Type 1 infantile. "
                "Clinically identical to AJ Type 1; distinct founder chromosome; concentrated in Quebec."
            ),
            "typical_onset": "3–6 months",
            "genotype_notes": "Distinct from AJ IVS12+1G>C; both null; FC carrier freq 1/30 (similar to AJ 1/27)",
        },
        {
            "class": "Non-AJ-Non-FC-Compound-Het-Missense-Null (Type 2 Juvenile / Type 3 Adult)",
            "pct": 12,
            "description": (
                "Non-founder compound heterozygotes: various private missense mutations + null allele. "
                "Phenotype determined by residual Hex A: 2-8% → juvenile; >10% → adult/chronic. "
                "Irish IVS9+1G>A, Japanese p.Gly269Ser homozygous (adult if Hex A >10%), other population-specific variants."
            ),
            "typical_onset": "Variable (2–40 years depending on residual Hex A)",
            "genotype_notes": "Missense alleles with partial activity → phenotype genotype correlation by residual Hex A",
        },
        {
            "class": "Pseudodeficiency-Compound-p.Arg247Trp-or-p.Arg249Trp (Not True Tay-Sachs)",
            "pct": 5,
            "description": (
                "Pseudodeficiency variants (p.Arg247Trp / p.Arg249Trp): DBS Hex A falsely low; leukocyte Hex A NORMAL. "
                "Included in cohort to track correct leukocyte assay workflow — NOT true Tay-Sachs. "
                "Critical: 5% of apparent 'carriers' in screening programmes carry pseudodeficiency allele. "
                "All ultimately reclassified after leukocyte confirmation — no epilepsy in this subgroup."
            ),
            "typical_onset": "N/A — not Tay-Sachs (included for diagnostic workflow)",
            "genotype_notes": "DBS low Hex A + leukocyte NORMAL Hex A = pseudodeficiency; no neurological risk",
        },
    ]

    seizure_types = [
        {
            "type": "Infantile Spasms (West Syndrome) — Type 1 Primary",
            "prevalence_pct": 35,
            "eeg_pattern": "Hypsarrhythmia (classic chaotic high-voltage); modified hypsarrhythmia with burst suppression in late-stage",
            "semiology": "Flexor/extensor or mixed spasms in clusters; startle-triggered clusters (hyperekplexia background); early 3-7 months",
            "clinical_tips": (
                "ACTH Level A preferred (cherry-red spot 90-95% in Type 1 → VGB HIGH RISK). "
                "Fundoscopy BEFORE treatment selection. Vigabatrin only if cherry-red absent (rare). "
                "Type 1 infantile spasms rarely achievable long-term seizure freedom; disease progression overrides. "
                "SE-risk: myoclonic status in later disease."
            ),
        },
        {
            "type": "Action / Intention Myoclonus — Type 2 Juvenile Primary",
            "prevalence_pct": 72,
            "eeg_pattern": "Polyspike-wave (generalised); cortical correlate on back-averaging; high-amplitude spikes frontally",
            "semiology": "Erratic limb myoclonus activated by intention + touch; face + distal limbs; worsened by startle; inter-ictal on EEG; falls risk",
            "clinical_tips": (
                "Most disabling symptom in Type 2 juvenile. Piracetam Level B adjunct to VPA. "
                "Clonazepam Level B nocturnal action myoclonus. CBZ/OXC ABSOLUTE CI — acute worsening. "
                "Action myoclonus EEG polyspike-wave ≠ primary generalised epilepsy → do NOT initiate CBZ."
            ),
        },
        {
            "type": "Generalised Tonic-Clonic Seizures (GTCS)",
            "prevalence_pct": 65,
            "eeg_pattern": "Generalised polyspike-wave 3-5 Hz evolving to tonic then clonic; post-ictal suppression",
            "semiology": "Major GTCS; nocturnal predominance; often superimposed on myoclonic background; Type 2 main GTCS population",
            "clinical_tips": (
                "Type 2 GTCS often misdiagnosed as GGE/JME → CBZ prescribed → myoclonic worsening (ABSOLUTE CI). "
                "VPA backbone + LEV adjunct. Rescue: buccal midazolam (1st); IV LEV (SE). "
                "NOT fosphenytoin — ABSOLUTE CI."
            ),
        },
        {
            "type": "Absence-Like / Dialeptic Seizures",
            "prevalence_pct": 38,
            "eeg_pattern": "Irregular 2-4 Hz spike-wave; background slowing (unlike CAE); atypical absence morphology",
            "semiology": "Staring + unresponsiveness 5-30s; motor automatisms less prominent; post-ictal confusion present (unlike CAE); in Type 2",
            "clinical_tips": (
                "Atypical absence morphology (irregular SW + background slowing) distinguishes from CAE. "
                "Do NOT initiate ethosuximide alone — coexisting GTCS+myoclonus need VPA. "
                "Do NOT initiate CBZ/OXC — worsens myoclonic background."
            ),
        },
        {
            "type": "Focal Cortical Seizures (Occipital/Motor)",
            "prevalence_pct": 28,
            "eeg_pattern": "Focal discharges occipital (visual cortex GM2 accumulation) or motor (cortical myoclonus correlates)",
            "semiology": "Visual phenomena (phosphenes, elementary visual hallucinations); focal motor jerks; may secondarily generalise; Type 2/3",
            "clinical_tips": (
                "Occipital GM2 accumulation → visual cortex hyperexcitability. "
                "MRI may show occipital cortical T2 signal changes. "
                "LEV preferred adjunct; LTG caution (myoclonus worsening risk in some PME). "
                "Focal EEG ≠ structural lesion — functional secondary to GM2 storage."
            ),
        },
        {
            "type": "Non-Convulsive Status Epilepticus (NCSE)",
            "prevalence_pct": 18,
            "eeg_pattern": "Prolonged irregular spike-wave activity; altered consciousness; EEG-confirmed NCSE",
            "semiology": "Altered awareness with myoclonic twitching + confusion; may be mistaken for post-ictal state; Type 2 late disease",
            "clinical_tips": (
                "NCSE risk increases in late Type 2 disease. "
                "IV LEV 60 mg/kg or IV VPA (POLG1 excluded) — NOT fosphenytoin (ABSOLUTE CI). "
                "IV diazepam useful acute; IV midazolam infusion if persistent. "
                "EEG monitoring mandatory in unexplained altered consciousness in Tay-Sachs."
            ),
        },
    ]

    triggers = [
        {
            "trigger": "Fever / Intercurrent Illness",
            "prevalence_pct": 82,
            "mechanism": "Febrile threshold lowering in GM2 gangliosidosis; cortical hyperexcitability worsens with metabolic stress",
            "management": "Aggressive antipyretics (paracetamol + ibuprofen alternating); sick-day rescue plan; low threshold for emergency attendance",
        },
        {
            "trigger": "Sleep Deprivation",
            "prevalence_pct": 72,
            "mechanism": "Sleep dep lowers seizure threshold in PME syndromes generally; nocturnal GTCS + myoclonus both worsen",
            "management": "Rigid sleep schedule; melatonin for sleep maintenance; Clonazepam nocturnal myoclonus if sleep-disruptive",
        },
        {
            "trigger": "Missed AED Dose",
            "prevalence_pct": 68,
            "mechanism": "VPA + LEV abrupt discontinuation → rebound cortical hyperexcitability",
            "management": "Dosette box + carer-supervised dosing; modified release VPA reduces peak-trough fluctuation; caregiver training",
        },
        {
            "trigger": "Photic / Pattern Stimulation",
            "prevalence_pct": 55,
            "mechanism": "Visual cortex GM2 accumulation → cortical hyperexcitability; 15-25 Hz photic-driving most provocative",
            "management": "Avoid strobe lighting; anti-reflective glasses; avoid fast-moving patterns (video games, flicker); photic-sensitivity documented in EEG",
        },
        {
            "trigger": "Intentional / Voluntary Movement (Action Myoclonus Trigger)",
            "prevalence_pct": 72,
            "mechanism": "Cortical reflex myoclonus triggered by movement intention; back-averaging EEG confirms cortical origin",
            "management": "Piracetam Level B; VPA backbone; Clonazepam PRN; wrist/ankle weights reduce oscillation amplitude; occupational therapy",
        },
        {
            "trigger": "Startle (Tactile / Auditory / Visual)",
            "prevalence_pct": 62,
            "mechanism": "Exaggerated startle reflex (hyperekplexia) — prominent in Type 1 infantile; startle-myoclonus in Type 2",
            "management": "Environmental noise reduction; clonazepam (startle response most benzodiazepine-sensitive); hearing protection if severe",
        },
        {
            "trigger": "Emotional Stress / Anxiety",
            "prevalence_pct": 48,
            "mechanism": "Hypothalamic-limbic activation → sympathetic arousal → cortical excitability increase",
            "management": "Psychological support; low-dose clonazepam PRN stress episodes; mindfulness-adapted programs; carer education",
        },
        {
            "trigger": "Prohibited Drug Exposure (CBZ/OXC/PHT/Fosphenytoin)",
            "prevalence_pct": 100,
            "mechanism": "Sodium channel blocker → cortical myoclonus worsening + GTCS exacerbation in GM2 gangliosidosis",
            "management": "ABSOLUTE CI — A&E alert card + GP records + hospital system flag; IV LEV replaces fosphenytoin in SE",
        },
    ]

    treatments = [
        {
            "drug": "VPA (Sodium Valproate) — VPA",
            "level": "Level B",
            "dose": "20-60 mg/kg/day oral in 2-3 divided doses; target trough 50-100 mg/L; CR formulation preferred",
            "moa": "Multi-modal: Na-channel stabilisation (minor), T-Ca++ channel inhibition, GABA-T inhibition → GABA elevation, histone deacetylase inhibition",
            "efficacy": "75% ≥50% seizure reduction in Type 2 PME; 68% myoclonus reduction; backbone drug of Tay-Sachs epilepsy",
            "monitoring": "TDM trough (50-100 mg/L); LFT (ALT/AST) 3-monthly; FBC; weight; teratogenicity (sodium valproate PREVENT programme); POLG1 excluded BEFORE initiation",
            "hexa_note": "HEXA is lysosomal NOT mitochondrial — VPA SAFE. POLG1/MERRF exclusion mandatory before starting. VPA + LEV first-line combination.",
        },
        {
            "drug": "LEV (Levetiracetam) — IV and Oral",
            "level": "Level B",
            "dose": "Oral: 20-60 mg/kg/day in 2 doses; IV SE: 60 mg/kg (max 4500 mg) over 15 min",
            "moa": "SV2A modulation → presynaptic vesicle trafficking inhibition → reduced glutamate release; modulates GABA-A receptor",
            "efficacy": "65% ≥50% GTCS reduction in Type 2; primary IV alternative to fosphenytoin (ABSOLUTE CI) in SE; well tolerated",
            "monitoring": "Renal function (dose-reduce eGFR <50); psychiatric monitoring (depression/agitation ~10%); no TDM needed routinely",
            "hexa_note": "IV LEV REPLACES fosphenytoin in Tay-Sachs SE protocol. Must be pre-labelled in A&E drug chart. VPA + LEV combination preferred first-line.",
        },
        {
            "drug": "ACTH (Tetracosactide) — Infantile Spasms Type 1",
            "level": "Level A",
            "dose": "ACTH 40-60 IU IM daily x 2 weeks (reducing schedule) OR Prednisolone 2-10 mg/kg/day x 2 weeks; per UKISS/INFANT trial protocol",
            "moa": "ACTH → MC receptor 2 → cortisol + adrenal androgens → CRH suppression → infantile spasm suppression; ACTH direct CNS receptor effect independent of cortisol",
            "efficacy": "60-70% spasm cessation in Type 1 IS (lower than cryptogenic IS due to structural cause); hypsarrhythmia resolution 55%",
            "monitoring": "BP, electrolytes, glucose (steroid side effects); infection risk; ophthalmology WITHIN 2 weeks (glaucoma risk); weight gain",
            "hexa_note": "PREFERRED over VGB when cherry-red spot present (Type 1: 90-95% cherry-red). Fundoscopy MANDATORY before IS treatment selection. Not disease-modifying for Tay-Sachs progression.",
        },
        {
            "drug": "Piracetam — Action Myoclonus",
            "level": "Level B",
            "dose": "Adults: 24-45 g/day in 3-4 doses; children: 160-300 mg/kg/day; start low (8g/day) titrate over 2-4 weeks",
            "moa": "AMPA receptor potentiation; membrane fluidity modulation; platelet aggregation inhibition; reduces cortical hyperexcitability",
            "efficacy": "50-65% action myoclonus reduction in Type 2; QoL improvement in functional tasks (writing, eating, drinking)",
            "monitoring": "Renal function (dose-reduce CKD); psychiatric (agitation, depression at high doses); no hepatotoxicity; no TDM",
            "hexa_note": "Most useful for disabling action/intention myoclonus in Type 2 juvenile. Added to VPA+LEV backbone. Relatively safe long-term.",
        },
        {
            "drug": "Clonazepam — Nocturnal Myoclonus / Startle",
            "level": "Level B",
            "dose": "0.5-6 mg/day oral in 2-3 doses; nocturnal dose 0.5-1 mg at bedtime for nocturnal myoclonus",
            "moa": "GABA-A receptor potentiation (benzodiazepine site); anti-myoclonic; most anti-myoclonic of all benzodiazepines",
            "efficacy": "60% nocturnal myoclonus reduction; 70% startle-myoclonus reduction; tolerance develops after 3-6 months of continuous use",
            "monitoring": "Sedation; tolerance/dependence (rotating dosing or drug holidays); respiratory depression at high doses; use lowest effective dose",
            "hexa_note": "Best GABA-ergic option for nocturnal myoclonus and startle-induced myoclonus. Tolerance management by alternate-day dosing or intermittent use.",
        },
        {
            "drug": "KD (Ketogenic Diet) — Adjunct",
            "level": "Level C",
            "dose": "Classical 4:1 KD or modified Atkins diet; initiated by specialist ketogenic diet team; paediatric dietitian essential",
            "moa": "Ketosis → ATP-sensitive K-channel opening + glutamate reduction + GABAergic enhancement + neuroinflammation reduction",
            "efficacy": "Limited data in Tay-Sachs specifically; 40% ≥50% seizure reduction in drug-resistant paediatric PME generally; may reduce myoclonus",
            "monitoring": "Lipid profile; acidosis; kidney stones; growth in children; metabolic panel; not contraindicated in lysosomal disease",
            "hexa_note": "No specific Tay-Sachs data but no contraindication. Consider in drug-resistant Type 2 juvenile PME. MDT (dietitian + neurologist) required.",
        },
        {
            "drug": "MDT Palliative / Supportive Care",
            "level": "Level A",
            "dose": "Multidisciplinary: neurology + genetics + physiotherapy + OT + speech therapy + palliative care (Type 1/2 terminal); dietitian",
            "moa": "Supportive — spasm reduction, quality of life, caregiver support, nutrition, mobility, communication",
            "efficacy": "Quality of life + caregiver wellbeing + functional duration; gastrostomy for nutrition (Type 1 by 18m; Type 2 as swallowing declines)",
            "monitoring": "Swallowing (FEES/VFS); chest physio; scoliosis (TLSO brace); pain management; palliative care for Type 1/2 end stage",
            "hexa_note": "Tay-Sachs has NO approved disease-modifying therapy. All treatment is symptomatic. Gene therapy Phase I/II — urgent trial referral.",
        },
        {
            "drug": "Rescue Midazolam / IV LEV — SE Protocol",
            "level": "Level A",
            "dose": "Midazolam buccal: 0.5 mg/kg (max 10 mg adult) → repeat x1 if no response at 10 min. IV LEV SE: 60 mg/kg (max 4500 mg) over 15 min",
            "moa": "Midazolam: GABA-A potentiation → rapid CNS seizure termination. IV LEV: SV2A → presynaptic glutamate reduction.",
            "efficacy": "Buccal midazolam 78% SE termination within 15 min. IV LEV SE: 65% SE termination (2nd line); replaces fosphenytoin (ABSOLUTE CI)",
            "monitoring": "Respiratory monitoring 1h post-midazolam; BP and renal function post-IV LEV",
            "hexa_note": "HEXA SE protocol: Midazolam buccal → IV LEV → IV VPA (POLG1 excluded) → IV anaesthesia. NEVER fosphenytoin. Pre-populated in emergency card.",
        },
    ]

    contraindications = [
        {
            "drug": "CBZ / OXC / PHT (Carbamazepine / Oxcarbazepine / Phenytoin)",
            "severity": "ABSOLUTE CI",
            "reason": "Sodium channel blockers cause acute myoclonic worsening + GTCS exacerbation in GM2 gangliosidosis/PME. Type 2 juvenile GTCS misidentified as GGE/JME → CBZ initiated → myoclonic storm. Mean diagnosis delay 2.9y = extended exposure window.",
            "note": "Safe alternative: VPA + LEV. Na-channel blocker ABSOLUTE CI in all PME phenotypes including Tay-Sachs Type 2.",
        },
        {
            "drug": "Fosphenytoin (IV Phenytoin Prodrug)",
            "severity": "ABSOLUTE CI",
            "reason": "Standard SE protocol second-line = fosphenytoin → IV phenytoin equivalent → cortical myoclonus worsening in Tay-Sachs SE. A&E staff use fosphenytoin by default — must be overridden explicitly in all Tay-Sachs patients.",
            "note": "REPLACE WITH: IV LEV 60 mg/kg as universal second-line SE drug in Tay-Sachs. Pre-populate hospital drug chart + emergency card.",
        },
        {
            "drug": "VGB (Vigabatrin) — Type 1 Infantile (HIGH RISK)",
            "severity": "HIGH RISK",
            "reason": "VGB retinopathy (irreversible peripheral VF loss) compounded with retinal ganglion cell GM2 storage in Type 1 (cherry-red 90-95%) → catastrophic combined visual failure. NOT absolute CI (unlike NCL) but HIGH RISK.",
            "note": "Use ACTH first-line for infantile spasms (Type 1). If VGB used despite cherry-red → ERG/VEP every 3 months. VGB acceptable in Type 3 (no retinal involvement).",
        },
        {
            "drug": "TGB (Tiagabine)",
            "severity": "HIGH RISK",
            "reason": "GABA reuptake inhibitor → paradoxical cortical disinhibition in PME syndromes; NCSE risk. Avoid in all PME phenotypes.",
            "note": "If NCSE develops → IV LEV or IV VPA. Never initiate TGB in Tay-Sachs/PME.",
        },
        {
            "drug": "GBP / PGB (Gabapentin / Pregabalin)",
            "severity": "HIGH RISK",
            "reason": "Alpha-2-delta ligands worsen myoclonus in PME syndromes; documented myoclonic exacerbation in cortical myoclonus.",
            "note": "If neuropathic pain component present → use low-dose amitriptyline or duloxetine instead.",
        },
        {
            "drug": "LTG Monotherapy (Lamotrigine — avoid as sole drug in Type 2 PME)",
            "severity": "HIGH RISK",
            "reason": "LTG can worsen myoclonus in PME when used as monotherapy (dose-dependent Na-channel blockade). Safe as adjunct to VPA at LOW dose (VPA doubles LTG half-life — reduce LTG dose by 50%).",
            "note": "If LTG used → strict dose halving with VPA co-medication; monitor myoclonus; never as sole AED in Type 2.",
        },
        {
            "drug": "AED Taper (Abrupt Withdrawal)",
            "severity": "HIGH RISK",
            "reason": "Abrupt VPA or LEV taper → rebound cortical hyperexcitability → GTCS + myoclonic status. Slow taper >3 months mandatory.",
            "note": "Cover any taper with clobazam bridge; emergency plan if inter-current illness during taper.",
        },
    ]

    monitoring = [
        {
            "item": "Leukocyte Hex A Assay (4-MU-GalNAc substrate) + Hex B simultaneously",
            "frequency": "At diagnosis; repeat 12-monthly if equivocal; ALWAYS after DBS low Hex A",
            "note": "Gold standard. Hex A <1-3% of total + Hex B elevated = Tay-Sachs. Confirm pseudodeficiency (DBS low/leukocyte normal). OCP users: leukocyte only.",
        },
        {
            "item": "HEXA Gene Sequencing (WES or HEXA panel)",
            "frequency": "Once at diagnosis; cascade family testing",
            "note": "AJ three-mutation panel (c.1278insTATC, IVS12+1G>C, p.Gly269Ser) covers ~98% AJ. WES for non-AJ/FC. Pseudodeficiency variants (p.Arg247Trp, p.Arg249Trp) identified on sequencing.",
        },
        {
            "item": "POLG1 + mtDNA WES / Deletion Panel — Mandatory Before VPA",
            "frequency": "Once before VPA initiation",
            "note": "POLG1 Alpers + MERRF both mimic Type 2 juvenile PME. POLG1 + VPA = fatal hepatotoxicity. Mandatory exclusion. If POLG1 positive → VPA ABSOLUTELY CONTRAINDICATED.",
        },
        {
            "item": "Ophthalmology — Fundoscopy + ERG + VEP",
            "frequency": "At diagnosis (cherry-red spot identification); 6-monthly thereafter; urgent before IS treatment",
            "note": "Cherry-red spot: macular vs retinal NCL. Type 1: 90-95%; Type 2 <10%; Type 3 absent. ERG/VEP: visual pathway integrity. Determines ACTH vs VGB in IS.",
        },
        {
            "item": "Brain MRI (3T with SWI + DWI)",
            "frequency": "At diagnosis; 12-monthly if symptomatic; 24-monthly stable",
            "note": "Infantile: symmetric thalamic T2 hypointensity (GM2 thalamic storage; highly characteristic) + cerebral WM changes. Juvenile/Adult: cerebellar + posterior cortical atrophy. Not a lysosomal DBS enzyme assay disease.",
        },
        {
            "item": "EEG (awake + sleep + photic)",
            "frequency": "At diagnosis; annually; urgent if clinical change / altered consciousness",
            "note": "Hypsarrhythmia (Type 1 IS); polyspike-wave generalised (Type 2 PME); focal occipital discharges. NCSE surveillance in unexplained altered consciousness.",
        },
        {
            "item": "VPA Therapeutic Drug Monitoring (TDM) + LFT + FBC",
            "frequency": "TDM: 6-weekly until stable then 6-monthly; LFT + FBC: 3-monthly",
            "note": "VPA trough target 50-100 mg/L. LFT (ALT) to exclude hepatotoxicity. FBC (thrombocytopenia rare). Weight (VPA associated weight gain).",
        },
        {
            "item": "Swallowing Assessment (FEES / Videofluoroscopy)",
            "frequency": "6-monthly Type 2 progressive; annually stable",
            "note": "Progressive dysphagia in Type 1 (gastrostomy by 18m); Type 2 late disease. Aspiration pneumonia = leading cause of death in Type 1/2.",
        },
        {
            "item": "Neuropsychology / Cognitive Assessment",
            "frequency": "Annually",
            "note": "Cognitive trajectory in Type 2 (progressive decline); Type 3 may show psychiatric symptoms first. Vineland / VABS in children. GCS equivalent in severe cases.",
        },
        {
            "item": "Physiotherapy + Occupational Therapy",
            "frequency": "Ongoing",
            "note": "Action myoclonus → wrist/ankle weights; mobility aids; falls prevention. Scoliosis screening (spinal XR annually Type 2). Wrist splints for writing assistance.",
        },
        {
            "item": "Genetic Counselling — Carrier Screening Cascade",
            "frequency": "At diagnosis; at each family planning contact",
            "note": "Autosomal recessive 25% recurrence risk. AJ partner testing essential. PGT-M available. AJ carrier freq 1/27 — carrier test partner before pregnancy.",
        },
        {
            "item": "SUDEP Risk Assessment + Nocturnal Monitoring",
            "frequency": "Annually; accelerated if GTCS uncontrolled",
            "note": "Nocturnal GTCS in Type 2 → SUDEP risk. Bed sensor / SPO2 monitor. SUDEP counselling at annual review. No data on SUDEP rate specifically in Tay-Sachs.",
        },
        {
            "item": "Gene Therapy Trial Eligibility Assessment",
            "frequency": "At diagnosis (urgent); 6-monthly if pre-symptomatic",
            "note": "AAV9-HEXA/HEXB Phase I/II 2024. Best outcomes pre-symptomatic or early symptomatic. Referral to specialist centre (Boston Children's / Manchester / Toronto) URGENTLY at diagnosis.",
        },
        {
            "item": "BDSRA / NTSAD Registry Enrolment",
            "frequency": "At diagnosis",
            "note": "National Tay-Sachs & Allied Diseases Association (NTSAD) registry; BDSRA. Research, family support, gene therapy trial notification. Enrol at diagnosis.",
        },
    ]

    lifecycle_stages = [
        {
            "stage": "Prenatal / Pre-Symptomatic (Sibling Risk / Carrier Screening)",
            "age_range": "Pre-conception — birth (at-risk families)",
            "description": "AJ carrier screening detects carrier couples (1/27 × 1/27 = 1/729 AJ couple risk). Pre-natal DX: amniocentesis Hex A assay + HEXA gene. PGT-M available.",
            "priorities": [
                "Partner Hex A carrier test before conception",
                "PGT-M referral for carrier couples",
                "Amniocentesis HEXA gene confirmation if PGT not used",
                "Gene therapy trial pre-screening (pre-symptomatic window narrow)",
            ],
        },
        {
            "stage": "Neonatal / First Months (Type 1 Infantile Onset 3–6m)",
            "age_range": "0–6 months",
            "description": "Type 1 onset: hyperekplexia (exaggerated startle) ← earliest sign; developmental plateau followed by regression; macrocephaly onset; cherry-red spot on fundoscopy.",
            "priorities": [
                "Urgent Hex A leukocyte assay + HEXA gene",
                "Fundoscopy (cherry-red confirmation)",
                "MRI (thalamic T2 hypointensity)",
                "ACTH for emerging infantile spasms (NOT VGB if cherry-red present)",
                "Urgent gene therapy trial referral (NTSAD, specialist centre)",
            ],
        },
        {
            "stage": "Early Childhood — Infantile Spasms → Myoclonus Transition (Type 1)",
            "age_range": "6 months — 2 years",
            "description": "IS → myoclonus → tonic seizures transition in Type 1; rapid neurodegeneration; progressive macrocephaly; swallowing decline; gastrostomy planning.",
            "priorities": [
                "ACTH → VPA transition for post-IS myoclonus",
                "Swallowing FEES → gastrostomy planning (aspiration risk)",
                "Palliative care introduction",
                "Scoliosis surveillance",
                "Physiotherapy, carer support, NTSAD family connect",
            ],
        },
        {
            "stage": "School Age — PME Onset (Type 2 Juvenile Primary Epilepsy Window)",
            "age_range": "2–10 years (Type 2 onset mean 5y)",
            "description": "Type 2 juvenile: ataxia + action myoclonus + GTCS + cognitive decline. Primary epilepsy management period. Risk of CBZ/OXC prescribing by non-specialist (Type 2 GTCS = GGE/JME mimic).",
            "priorities": [
                "Confirm Hex A (leukocyte) + HEXA gene",
                "VPA + LEV initiation (avoid CBZ ABSOLUTE CI)",
                "Piracetam for action myoclonus",
                "School support plan (LD + seizure management)",
                "Ophthalmology ERG/VEP baseline",
                "POLG1/MERRF exclusion before VPA",
            ],
        },
        {
            "stage": "Adolescence — Drug-Resistant PME + Cognitive Decline (Type 2)",
            "age_range": "10–25 years (Type 2 advanced stage)",
            "description": "Progressive cognitive decline; increasing myoclonus severity; VPA + LEV + Piracetam combination; KD consideration; DVLA driving cessation; transition to adult services.",
            "priorities": [
                "Driving cessation (DVLA mandatory — seizures + cognitive decline)",
                "KD referral if drug-resistant",
                "Swallowing re-assessment (gastrostomy planning)",
                "Adult transition planning",
                "Carer stress + mental health support",
                "SUDEP risk counselling + nocturnal monitoring",
            ],
        },
        {
            "stage": "Late Stage — Palliative / End-of-Life (Type 1 by 2–5y; Type 2 late teens)",
            "age_range": "Type 1: 2–5y; Type 2: late teens; Type 3: variable",
            "description": "End-stage: severe neurological disability, aspiration pneumonia, minimal seizure control, comfort-directed care. Palliative care team essential. Bereavement support for families.",
            "priorities": [
                "Comfort-directed seizure management (rescue midazolam, avoid hospital admission where possible)",
                "Palliative sedation for refractory myoclonus/distress",
                "Gastrostomy + chest physio for aspiration prevention",
                "Bereavement support (NTSAD, BDSRA grief services)",
                "Research / registry contribution (tissue banking, natural history)",
                "Gene therapy trial outcome reporting (data to NTSAD)",
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
        "disease_name": "Tay-Sachs Disease / GM2 Gangliosidosis Type 1 (HEXA Deficiency)",
        "gene_full": "HEXA (β-Hexosaminidase A α-subunit) — 15q23; 14 exons; 529 aa precursor; TIM barrel fold; pairs with HEXB β-subunit → Hex A (αβ) heterodimer",
        "omim_gene": "*606869 (HEXA gene)",
        "omim_disease": "#272800 (Tay-Sachs Disease / GM2 Gangliosidosis Type 1)",
        "protein_full": (
            "β-Hexosaminidase A (α-subunit): 529 aa precursor; signal peptide 1-22; propeptide 23-42 (removed in lysosome); "
            "mature α-subunit ~56 kDa; TIM barrel catalytic domain; catalytic residues Glu323 (acid/base) + Glu325 (nucleophile); "
            "retaining double-displacement mechanism; requires GM2 activator protein (GM2AP) to load GM2 ganglioside; "
            "forms αβ heterodimer (Hex A) — primary GM2 cleaving enzyme; "
            "also forms αα homodimer (Hex S) — active on GM2 with GM2AP but minor contributor. "
            "HEXA LOF → Hex A (αβ) AND Hex S (αα) both deficient; Hex B (ββ, HEXB homodimer) retained and elevated."
        ),
        "inheritance_mode": "Autosomal Recessive (AR) — biallelic LOF required; pLI ~0.81 (moderate LOF intolerance)",
        "onset_age": "Type 1 Infantile: 3–6 months; Type 2 Juvenile: 2–10 years (mean 5y); Type 3 Adult/Chronic: adolescence–40y",
        "multienzyme_triad_note": (
            "Three GM2 gangliosidoses — encoded by three separate genes: "
            "(1) HEXA (15q23): α-subunit LOF → Hex A (αβ) deficient + Hex S (αα) deficient; Hex B (ββ) ELEVATED (ββ retained). "
            "Leukocyte Hex A LOW + Hex B ELEVATED = PATHOGNOMONIC Tay-Sachs. "
            "(2) HEXB (5q13.3): β-subunit LOF → ALL three forms deficient (Hex A + Hex B + Hex S); Sandhoff disease; "
            "more systemic (hepatosplenomegaly, bone marrow foam cells) than Tay-Sachs. "
            "(3) GM2A (5q33.1): GM2 activator protein LOF → AB variant; Hex A NORMAL in vitro (artificial substrate); "
            "in vivo GM2 not loaded → accumulates; confirmed by GM2AP protein assay or GM2A gene. "
            "DIAGNOSTIC RULE: Leukocyte 4-MU assay measures Hex A AND Hex B SIMULTANEOUSLY. "
            "NEVER use DBS alone (pseudodeficiency + OCP false-low). "
            "Hex A LOW + Hex B HIGH = Tay-Sachs. Hex A LOW + Hex B LOW = Sandhoff. Both NORMAL = AB variant."
        ),
        "cherry_red_differentials": [
            {"disease": "Tay-Sachs (HEXA) Type 1", "distinguishing": "90-95% cherry-red; exaggerated startle (hyperekplexia); Hex A low + Hex B ELEVATED"},
            {"disease": "Sandhoff (HEXB)", "distinguishing": "90% cherry-red; more systemic (hepatosplenomegaly, foam cells); Hex A LOW + Hex B LOW"},
            {"disease": "GM1 Gangliosidosis Type 1 (GLB1)", "distinguishing": "50-90% cherry-red; coarse facies + hepatosplenomegaly + bone changes; GLB1 low"},
            {"disease": "Sialidosis Type I (NEU1)", "distinguishing": "90% cherry-red; PME + NORMAL cognition; no visceral involvement; NEU1 low (urine sialic acid elevated)"},
            {"disease": "Galactosialidosis (CTSA)", "distinguishing": "Cherry-red + coarse facies 80% + angiokeratoma 65%; NEU1 + GLB1 BOTH low (CTSA mutation)"},
            {"disease": "Niemann-Pick Type A (SMPD1)", "distinguishing": "50% cherry-red; severe hepatosplenomegaly; sphingomyelin accumulation; SMPD1 low"},
            {"disease": "MPS I / Hurler (IDUA)", "distinguishing": "Cherry-red ABSENT usually; coarse facies + hepatosplenomegaly + corneal clouding; iduronidase low"},
            {"disease": "Krabbe (GALC)", "distinguishing": "Cherry-red rare; peripheral neuropathy + thalamic T1 high signal; galactocerebrosidase low"},
        ],
        "concepts": [
            {
                "name": "HEXA-15q23-Lysosomal-Hex-A-Alpha-Subunit-GM2-Ganglioside-Hydrolysis",
                "definition": "HEXA encodes α-subunit of β-Hexosaminidase A; pairs with HEXB β-subunit → Hex A (αβ) heterodimer; requires GM2AP to load GM2 substrate; cleaves GalNAc from GM2 → GM3. HEXA LOF → Hex A deficient + Hex B elevated.",
            },
            {
                "name": "Hex-B-Elevated-PATHOGNOMONIC-Tay-Sachs-vs-All-Hex-Low-Sandhoff",
                "definition": "In Tay-Sachs: Hex B (ββ homodimer, HEXB product) retained and ELEVATED — because only α-subunit deficient. In Sandhoff: Hex A + Hex B + Hex S ALL LOW (β-subunit shared by all three forms). Hex B level discriminates instantaneously.",
            },
            {
                "name": "Pseudodeficiency-p.Arg247Trp-p.Arg249Trp-Leukocyte-Assay-Mandatory",
                "definition": "p.Arg247Trp and p.Arg249Trp reduce activity toward artificial substrate 4-MU-GlcNAc in DBS but NOT toward natural GM2. DBS Hex A appears LOW; Leukocyte Hex A NORMAL. These are not disease alleles. Confirm all DBS low Hex A with leukocyte assay. OCP users: serum falsely low — leukocyte assay mandatory.",
            },
            {
                "name": "CBZ-OXC-PHT-ABSOLUTE-CI-Type2-PME-GTCS-Misidentification",
                "definition": "Type 2 juvenile GTCS misidentified as GGE/JME → CBZ prescribed → acute myoclonic storm. Na-channel blockers ABSOLUTE CI in all Tay-Sachs PME phenotypes. 2.9y mean diagnosis delay = extended exposure window. VPA + LEV safe backbone.",
            },
            {
                "name": "VPA-SAFE-HEXA-Lysosomal-NOT-Mitochondrial-POLG1-MERRF-Mandatory",
                "definition": "HEXA is lysosomal acid hydrolase — NOT mitochondrial enzyme. VPA CI = MERRF/POLG only. VPA SAFE backbone in Tay-Sachs. POLG1/MERRF exclusion MANDATORY before VPA (POLG1 Alpers mimics Type 2 PME; fatal hepatotoxicity if VPA given to POLG1).",
            },
            {
                "name": "ACTH-Level-A-Infantile-Spasms-Type1-Prefer-Over-VGB-Cherry-Red-90-95pct",
                "definition": "Type 1 infantile spasms (West syndrome): cherry-red spot 90-95% → VGB HIGH RISK (retinal ganglion cell GM2 storage + VGB retinopathy). ACTH Level A preferred (UKISS/INFANT protocol). Fundoscopy MANDATORY before IS treatment selection.",
            },
            {
                "name": "VGB-HIGH-RISK-Type1-Not-Absolute-CI-Type3-Cherry-Red-Determines",
                "definition": "VGB not categorically absolute CI in Tay-Sachs (unlike NCL where retinal NCL = absolute CI). Cherry-red present (Type 1/2 early): HIGH RISK. Cherry-red absent (Type 3): acceptable if indicated. Determines ACTH vs VGB in IS. Monitor ERG/VEP 3-monthly if VGB used with any cherry-red risk.",
            },
            {
                "name": "Fosphenytoin-ABSOLUTE-CI-IV-LEV-Replaces-SE-Protocol",
                "definition": "Standard SE second-line = IV fosphenytoin → ABSOLUTE CI in Tay-Sachs (myoclonic worsening). REPLACE WITH IV LEV 60 mg/kg. Pre-populate A&E drug chart. Emergency card must specify IV LEV NOT fosphenytoin. SE protocol: midazolam → IV LEV → IV VPA (POLG1 excluded) → anaesthesia.",
            },
            {
                "name": "Piracetam-Level-B-Action-Myoclonus-Type2-AMPA-Membrane-Fluidity",
                "definition": "Piracetam (24-45 g/day adults) Level B for action/intention myoclonus in Type 2 juvenile. MOA: AMPA potentiation + membrane fluidity. Most effective for disabling action myoclonus. Added to VPA + LEV backbone. Monitor renal function.",
            },
            {
                "name": "Thalamic-T2-Hypointensity-Type1-MRI-Characteristic-GM2-Thalamic-Storage",
                "definition": "Symmetric thalamic T2 hypointensity on brain MRI = highly characteristic of Tay-Sachs Type 1 infantile. GM2 ganglioside accumulates in thalamic neurons → T2 signal change. Combined with progressive macrocephaly → high specificity. Contrast: CLN2 periventricular WM, CLN3 cerebral/cerebellar atrophy.",
            },
            {
                "name": "AJ-Three-Mutation-Panel-98-Percent-Coverage-Founder-Mutations",
                "definition": "Ashkenazi Jewish three-mutation panel: c.1278insTATC (78% of AJ alleles) + IVS12+1G>C (18%) + p.Gly269Ser (4%) covers ~98% of AJ carriers. French-Canadian: IVS7+1G>A (100% of FC alleles) — different from AJ. HEXA WES for non-AJ/FC families.",
            },
            {
                "name": "AAV9-HEXA-HEXB-Bicistronic-Gene-Therapy-Phase-I-II-2024",
                "definition": "AAV9-HEXA/HEXB bicistronic vector: single vector encodes BOTH α-subunit (HEXA) and β-subunit (HEXB) → restores Hex A (αβ) formation in lysosomes. Intrathecal route for CNS targeting. Phase I/II 2024. Best outcomes pre-symptomatic. Urgent referral at diagnosis.",
            },
            {
                "name": "No-Approved-Disease-Modifying-Therapy-CNS-Penetration-Barrier",
                "definition": "No approved ERT or substrate reduction for CNS Tay-Sachs. ERT cannot cross BBB. Miglustat (substrate reduction) has limited CNS penetration. Gene therapy (AAV9) = only viable disease-modifying approach currently in trials. All current treatment = symptomatic.",
            },
            {
                "name": "SUDEP-Risk-Drug-Resistant-Type2-Nocturnal-GTCS",
                "definition": "SUDEP risk elevated in Type 2 juvenile with drug-resistant nocturnal GTCS + progressive neurodegeneration. Nocturnal monitoring (bed sensor, SpO2). SUDEP counselling at annual review. No Tay-Sachs-specific SUDEP rate published; extrapolated from PME SUDEP data.",
            },
            {
                "name": "Driving-DVLA-Mandatory-Cessation-Seizures-Cognitive-Decline-Type2",
                "definition": "Type 2 juvenile: uncontrolled seizures + progressive cognitive decline → DVLA mandatory notification. UK: must not drive if ≥1 seizure and no 12-month seizure-free period. Cognitive decline independently may prevent driving. DVLA reporting responsibility: patient (mandatory) + clinician (discretionary if patient fails to report).",
            },
        ],
        "thresholds": [
            {"parameter": "Leukocyte Hex A (% total hexosaminidase)", "value": "<1% = Tay-Sachs; 1-3% = borderline (gene test mandatory); >3% + DBS low = pseudodeficiency", "action": "Leukocyte Hex A <1%: confirm HEXA gene; initiate cascade testing; gene therapy referral"},
            {"parameter": "DBS Hex A (newborn screen)", "value": "Low DBS Hex A → ALWAYS confirm with leukocyte assay (pseudodeficiency trap)", "action": "Never counsel based on DBS alone; leukocyte assay mandatory before genetic counselling"},
            {"parameter": "VPA trough (mg/L)", "value": "Target: 50-100 mg/L; >120 mg/L = toxic", "action": ">100: dose reduce; <40: uptitrate (if clinical control inadequate)"},
            {"parameter": "ALT (on VPA)", "value": "ALT >3× ULN = concerning; >5× ULN = hepatotoxicity", "action": "3×ULN: dose reduce + increase monitoring; 5×ULN: consider stopping VPA; POLG1 check if not done"},
            {"parameter": "IS spasm cluster duration", "value": ">5 spasms in cluster = SE-risk; EEG hypsarrhythmia → urgent intervention", "action": "ACTH escalation; IV LEV if SE (not fosphenytoin); urgent paediatric neurology"},
            {"parameter": "GTCS duration (SE threshold)", "value": ">5 min single GTCS or recurrent without recovery = SE", "action": "Buccal midazolam → IV LEV 60 mg/kg → IV VPA → anaesthesia (not fosphenytoin)"},
            {"parameter": "Piracetam dose (action myoclonus)", "value": "Children: 160-300 mg/kg/day; Adults: 24-45 g/day in 3-4 doses", "action": "Titrate over 4 weeks; reassess action myoclonus at 8 weeks; dose-reduce if psychiatric side effects"},
            {"parameter": "Swallowing safety (FEES/VFS)", "value": "Aspiration on FEES → gastrostomy discussion; silent aspiration = highest risk", "action": "PEG gastrostomy referral if significant aspiration; chest physio intensification"},
            {"parameter": "Cherry-red spot presence", "value": "Present = HIGH RISK for VGB; absent = VGB acceptable in IS treatment", "action": "Fundoscopy before IS treatment selection; ACTH if cherry-red present (not VGB)"},
            {"parameter": "DVLA seizure-free period", "value": "UK: 12 months seizure-free for Group 1 licence; 10 years seizure-free for Group 2", "action": "Seizure in past 12 months → must not drive Group 1; notify DVLA; letter from neurologist"},
            {"parameter": "SUDEP risk stratification", "value": "SUDEP-7 score ≥3 = high SUDEP risk (nocturnal GTCS, drug-resistant, no supervision)", "action": "Nocturnal mattress sensor; SpO2 monitoring; carer SUDEP education; higher priority VPA optimisation"},
            {"parameter": "AAV9 trial eligibility", "value": "Pre-symptomatic or <12 months symptomatic = best window; HEXA genotype confirmed", "action": "Urgent specialist centre referral (Boston/Manchester/Toronto) at diagnosis; do not wait for symptom progression"},
        ],
        "standards": [
            "Tay B & Sachs B (1881/1887) — original clinical descriptions; cherry-red spot (Tay) + amaurotic family idiocy (Sachs)",
            "Okada S & O'Brien JS (1969) Science — β-hexosaminidase A deficiency as enzymatic basis of Tay-Sachs",
            "Kaback MM et al. (1974) Prog Med Genet — AJ carrier screening programme (landmark public health genomics)",
            "Myerowitz R (1988) PNAS — c.1278insTATC frameshift as major AJ allele identification",
            "Triggs-Raine BL et al. (1990) NEJM — IVS12+1G>C splice site as second major AJ allele",
            "Gravel RA et al. (2001) OMMBID Ch 153 — comprehensive GM2 gangliosidosis molecular review",
            "Maegawa GH et al. (2006) J Inherit Metab Dis — pseudodeficiency variants p.Arg247Trp + p.Arg249Trp clinical characterisation",
            "ILAE (2022) PME Classification — Scheffer IE et al.; progressive myoclonic epilepsy classification framework",
            "NICE NG217 (2022) Epilepsy Guideline — UK AED treatment standard; VPA PREVENT programme (valproate in women of reproductive age)",
            "MHRA VPPP (2021) — Valproate Patient Pregnancy Prevention Programme; mandatory for females of childbearing potential on VPA",
            "CPIC POLG1 Guideline (2023) — VPA contraindicated in POLG1 pathogenic variants; mandatory exclusion before VPA in PME phenotypes",
            "Picone M et al. (2024) Phase I/II AAV9-HEXA/HEXB gene therapy — bicistronic approach; Phase I/II safety + biodistribution trial",
        ],
        "references": [
            "Okada S & O'Brien JS (1969) Tay-Sachs disease: generalized absence of a beta-D-N-acetylhexosaminidase component. Science 165:698-700.",
            "Kaback MM et al. (1977) Tay-Sachs disease: heterozygote screening and prenatal diagnosis. US Public Health Service. Birth Defects Orig Artic Ser 13:197-207.",
            "Myerowitz R & Costigan FC (1988) The major defect in Ashkenazi Jews with Tay-Sachs disease is an insertion in the gene for the alpha-chain of beta-hexosaminidase. J Biol Chem 263:18587-9.",
            "Maegawa GH et al. (2006) The natural history of juvenile or subacute GM2 gangliosidosis: 21 new cases and literature review of 134 previously reported. Pediatrics 118:e1550-62.",
            "Gravel RA et al. (2001) The GM2 gangliosidoses. In: Scriver CR et al. The Metabolic and Molecular Bases of Inherited Disease (8th ed). McGraw-Hill: 3827-76.",
            "Mole SE & Cotman SL (2015) Genetics of the neuronal ceroid lipofuscinoses (Batten disease). Biochim Biophys Acta 1852:2237-41.",
        ],
    }
