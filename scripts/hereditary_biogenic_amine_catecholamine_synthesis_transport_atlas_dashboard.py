#!/usr/bin/env python3
"""Hereditary-Biogenic-Amine-Catecholamine-Synthesis-Transport-Atlas -- Complete 8-Gene Reference
Biogenic Amine / Catecholamine Synthesis & Transport Disorders: TH-DDC-DBH-SLC18A2-SLC6A3-MAOA-SPR-GCH1
(TH/TH-Deficiency-Rate-Limiting-Dopamine-NE-Synthesis-BH4-Dependent-DRD3-SPECTRUM-Severe-LOF-TH0-to-Mild-DRD3-L-DOPA-Responsive-DIURNAL-VARIATION-PATHOGNOMONIC,
 DDC/AADC-Deficiency-DUAL-Substrate-LDopa-Dopamine-5HTP-Serotonin-OCULOGYRIC-CRISES-PATHOGNOMONIC-CSF-LDOPA-5HTP-ELEVATED-AAV2-Gene-Therapy,
 DBH/DBH-Deficiency-NE-ABSENT-Dopamine-HIGH-ORTHOSTATIC-HYPOTENSION-Droxidopa-CURATIVE-FDA-Approved,
 SLC18A2/VMAT2-DTDS2-ALL-Monoamines-Depleted-Infantile-Parkinsonism-Dystonia-SPECT-Reduced-Presynaptic-Poor-Prognosis,
 SLC6A3/DAT-Deficiency-DTDS1-HVA-PARADOXICALLY-ELEVATED-DAT-SPECT-Absent-Neonatal-Parkinsonism,
 MAOA/Brunner-Syndrome-XLR-Males-Only-Borderline-ID-IMPULSIVE-AGGRESSION-PATHOGNOMONIC-Urinary-Monoamine-Metabolites-Elevated,
 SPR/Sepiapterin-Reductase-Deficiency-BH4-Distal-LOW-HVA-LOW-5HIAA-BIOPTERIN-ELEVATED-Both-L-DOPA-and-5HTP-Needed,
 GCH1/Segawa-DRD1-AD-Haploinsufficiency-DIURNAL-DYSTONIA-DRAMATIC-LOW-DOSE-L-DOPA-Response-CSF-Neopterin-Biopterin-LOW):

TH (Tyrosine hydroxylase; 528aa; 11p15.5; AR-LOF;
     TH encodes tyrosine hydroxylase, the RATE-LIMITING enzyme in dopamine and norepinephrine (catecholamine) biosynthesis;
     TH hydroxylates L-tyrosine -> L-DOPA; requires BH4 (tetrahydrobiopterin) as cofactor;
     TH DEFICIENCY -- OMIM 605407;
     PHENOTYPIC SPECTRUM (severe TH0 to mild DRD-TH):
       SEVERE (TH0): complete or near-complete LOF; L-DOPA-nonresponsive; progressive encephalopathy;
         Leigh-like MRI; severe hypotonia; early death without aggressive treatment;
       MILD (DRD-TH = Dopamine-Responsive Dystonia type 3): partial LOF; L-DOPA-responsive;
         dystonia + parkinsonism features + diurnal variation;
     KEY PATHOGNOMONIC: DIURNAL VARIATION in mild TH deficiency PATHOGNOMONIC (worse in evening, better after sleep);
       mechanism: HVA (homovanillic acid, dopamine metabolite) levels fluctuate with BH4 availability and circadian rhythm;
       dopamine store depletes during the day -> symptoms worsen by evening; replenished during sleep;
     CSF NEUROTRANSMITTER PATTERN:
       LOW HVA (homovanillic acid; dopamine metabolite) -- primary finding;
       LOW MHPG (3-methoxy-4-hydroxyphenylglycol; norepinephrine metabolite) -- concurrent NE deficiency;
       NORMAL 5-HIAA (5-hydroxyindoleacetic acid; serotonin metabolite) -- SEROTONIN IS NORMAL;
       NORMAL CSF pterins (neopterin, biopterin) -- BH4 pathway intact (contrast with BH4 defects);
       this CSF pattern distinguishes TH from AADC (where 5-HIAA is also LOW) and from SPR/GCH1 (where pterins are abnormal);
     TREATMENT: L-DOPA/carbidopa (responsive in mild DRD-TH; limited effect in severe TH0);
       start at LOW dose; titrate slowly; dramatic response in mild forms;
     seed SEED_BASE+0).

DDC (DOPA decarboxylase / AADC; 480aa; 7p12.2; AR-LOF;
     DDC encodes aromatic L-amino acid decarboxylase (AADC), the enzyme converting:
       (1) L-DOPA -> DOPAMINE (dopamine synthesis from L-DOPA);
       (2) 5-HTP (5-hydroxytryptophan) -> SEROTONIN (5-hydroxytryptamine);
     AADC thus acts on BOTH dopamine AND serotonin pathways -- DUAL SUBSTRATE enzyme;
     AADC DEFICIENCY -- OMIM 608643;
     OCULOGYRIC CRISES PATHOGNOMONIC: involuntary, sustained, upward deviation of the eyes;
       episodes last minutes to hours; associated with extreme distress, autonomic instability;
       caused by dopamine deficiency in oculomotor circuits; hallmark of AADC among neurotransmitter defects;
     CSF NEUROTRANSMITTER PATTERN -- PATHOGNOMONIC:
       LOW HVA (dopamine metabolite LOW -- dopamine not being made);
       LOW 5-HIAA (serotonin metabolite LOW -- serotonin not being made);
       ELEVATED L-DOPA (substrate accumulates upstream of block);
       ELEVATED 5-HTP (substrate accumulates upstream of block);
       the combination of LOW metabolites + ELEVATED precursors distinguishes AADC from all other defects;
     CONFIRMATORY: AADC enzyme activity in plasma -- markedly reduced or absent;
     TREATMENT:
       dopamine agonist (pramipexole, ropinirole) -- directly stimulates dopamine receptors bypassing synthesis block;
       MAO-B inhibitor (selegiline) -- reduces dopamine degradation;
       pyridoxine (vitamin B6) -- PLP cofactor for AADC enzyme; some patients improve with B6;
       gene therapy: AAV2-DDC (bilateral intraputaminal injection) -- clinically transformative;
         marked improvement in oculogyric crises, motor function, and quality of life in clinical trials;
         most advanced gene therapy for a neurotransmitter disorder;
     seed SEED_BASE+1).

DBH (Dopamine beta-hydroxylase; 617aa; 9q33.2; AR-LOF;
     DBH encodes dopamine beta-hydroxylase, the enzyme in chromaffin granules (adrenal medulla + noradrenergic neurons)
     that converts dopamine -> norepinephrine (NE) via beta-hydroxylation;
     DBH DEFICIENCY -- OMIM 223360;
     NOREPINEPHRINE ABSENT + DOPAMINE MARKEDLY ELEVATED PATHOGNOMONIC:
       NE cannot be synthesized -> plasma/urine NE and epinephrine UNDETECTABLE;
       dopamine accumulates (substrate) -> plasma/urine dopamine MARKEDLY ELEVATED;
       urinary: dopamine HIGH; NE and epinephrine ABSENT (or near-absent);
       serum DBH enzyme activity: UNDETECTABLE (confirmatory);
     CLINICAL PRESENTATION:
       SEVERE ORTHOSTATIC HYPOTENSION (most prominent): syncope, presyncope with standing;
       ptosis (NE required for superior tarsal muscle / Muller's muscle);
       nasal congestion (NE maintains nasal vasomotor tone);
       retrograde ejaculation (sympathetic NE-mediated);
       first presentation often as teenager or young adult with syncope/falls;
     TREATMENT:
       L-DOPS (droxidopa) = synthetic norepinephrine precursor; peripherally converted to NE by AADC;
       FDA-approved specifically for DBH deficiency-associated neurogenic orthostatic hypotension;
       CURATIVE effect: restores NE; reverses orthostatic hypotension, ptosis, retrograde ejaculation;
       fludrocortisone + midodrine as adjuncts for OH;
     seed SEED_BASE+2).

SLC18A2 (Vesicular monoamine transporter 2 / VMAT2; 514aa; 10q25.3; AR-LOF;
     SLC18A2 encodes VMAT2 (vesicular monoamine transporter 2), the transporter loading ALL monoamines
     (dopamine, serotonin, norepinephrine, epinephrine, histamine) into presynaptic vesicles for release;
     VMAT2 is expressed in brain monoaminergic neurons (contrast with VMAT1 in peripheral chromaffin cells);
     DTDS TYPE 2 (infantile Parkinsonism-dystonia 2) -- OMIM 618049;
     ALL MONOAMINES DEPLETED: dopamine + serotonin + norepinephrine + epinephrine ALL severely reduced;
       this global depletion distinguishes SLC18A2 from gene-specific defects (TH affects only catecholamines;
       AADC affects dopamine + serotonin; DBH affects only NE);
     DOPAMINE SPECT PATHOGNOMONIC: REDUCED PRESYNAPTIC TRACER UPTAKE (like Parkinson's disease, but in an INFANT);
       123I-FP-CIT (DaTscan) or 18F-DOPA PET: markedly reduced striatal uptake;
       distinguishes SLC18A2/SLC6A3 from postsynaptic dopamine defects and from structural brain disease;
     CLINICAL: infantile hypotonia -> evolving hypertonia + parkinsonism-dystonia;
       oculogyric crises (shared with AADC);
       autonomic instability;
     CSF PATTERN: LOW HVA + LOW 5-HIAA (all monoamines depleted globally);
     TREATMENT: carbidopa/L-DOPA (partially restores dopamine); dopamine agonists; MAO inhibitors;
       POOR LONG-TERM PROGNOSIS -- restoration of VMAT2 loading impossible with current pharmacology;
       gene therapy investigational;
     seed SEED_BASE+3).

SLC6A3 (Dopamine transporter / DAT1; 620aa; 5p15.33; AR-LOF;
     SLC6A3 encodes DAT (dopamine transporter), the presynaptic reuptake transporter clearing dopamine from the synapse;
     DAT is responsible for dopamine reuptake and recycling; LOF -> failure to clear synaptic dopamine -> excess dopamine in synapse;
     DTDS TYPE 1 (dopamine transporter deficiency syndrome 1) -- OMIM 613135;
     PATHOPHYSIOLOGY PARADOX:
       dopamine EXCESS in synapse -> postsynaptic D2 receptor DOWNREGULATION (chronic excess desensitizes receptors);
       net result: functional DOPAMINE SIGNALING FAILURE despite elevated synaptic dopamine;
       clinical phenotype: PARKINSONISM-DYSTONIA (not hyperkinesis) -- the PARADOX;
     CSF HVA PARADOXICALLY ELEVATED PATHOGNOMONIC:
       dopamine cannot be reuptaken -> turnover increases -> HVA elevated;
       ELEVATED HVA in a child with parkinsonism-dystonia is PATHOGNOMONIC of DAT deficiency (SLC6A3);
       contrast with VMAT2 (SLC18A2): HVA LOW;
     DOPAMINE SPECT DIAGNOSTIC: absent DAT binding on 123I-ioflupane (DaTscan) SPECT;
       DAT protein absent -> no tracer binding -> "absent" DAT-SPECT vs "reduced" in VMAT2;
     CLINICAL: neonatal hypotonia -> infantile Parkinsonism (rigidity, bradykinesia) + hyperkinetic features;
     TREATMENT: L-DOPA partial benefit; carbidopa; pramipexole; gene therapy investigational;
     seed SEED_BASE+4).

MAOA (Monoamine oxidase A; 527aa; Xp11.3; XLR-LOF;
     MAOA encodes monoamine oxidase A, the mitochondrial outer membrane enzyme oxidatively deaminating:
       serotonin, dopamine, norepinephrine, epinephrine (ALL major monoamines);
     MAOA is located on X chromosome -> X-LINKED RECESSIVE; MALES ONLY affected (hemizygous LOF);
     BRUNNER SYNDROME -- OMIM 300615;
     MALES ONLY (hemizygous): borderline intellectual disability + IMPULSIVE AGGRESSIVE BEHAVIOR PATHOGNOMONIC;
       severe episodic aggression triggered by frustration or stress;
       impulsive violent behavior; sexually deviant behavior reported in original Dutch kindred;
       borderline intellectual disability (mild; not severe);
     MONOAMINE ACCUMULATION: serotonin + dopamine + norepinephrine ALL elevated (degradation blocked);
       ELEVATED URINARY: 5-HIAA (serotonin metabolite), HVA + DOPAC (dopamine metabolites),
         normetanephrine + MHPG (norepinephrine metabolites);
       NORMETANEPHRINE ELEVATED in urine -- useful biochemical marker;
     CARRIER FEMALES: HETEROZYGOUS; typically asymptomatic due to X-inactivation balancing;
       occasional mild behavioral effects in carriers with skewed X-inactivation;
     IMPORTANT DDx: do NOT confuse with MAO-B deficiency or MAOB variants;
       MAOA LOF -> aggressive; MAOA EXCESS (gene duplication) -> Brunner-type neurodevelopmental syndrome (different);
     TREATMENT: MAO-A inhibitors ARE CONTRAINDICATED (would worsen monoamine accumulation);
       behavioral management primary; SSRIs used cautiously for anxiety (monitor for serotonin syndrome);
       low-monoamine diet (reduce tyramine) as adjunct;
     seed SEED_BASE+5).

SPR (Sepiapterin reductase; 261aa; 2p13.2; AR-LOF;
     SPR encodes sepiapterin reductase, catalyzing the LAST STEP in BH4 (tetrahydrobiopterin) de novo synthesis
     in peripheral tissues and brain: 6-pyruvoyl-tetrahydropterin -> BH4 (via sequential carbonyl reductions);
     BH4 is the essential cofactor for:
       (1) TH (tyrosine hydroxylase): dopamine synthesis; (2) TPH (tryptophan hydroxylase): serotonin synthesis;
       (3) NOS (nitric oxide synthase): NO synthesis; (4) PAH (phenylalanine hydroxylase): phe->tyr;
     SPR DEFICIENCY -- OMIM 612716;
     DISTINCTIVE CSF PATTERN -- PATHOGNOMONIC:
       LOW HVA (dopamine pathway impaired -- TH activity reduced due to BH4 deficiency);
       LOW 5-HIAA (serotonin pathway impaired -- TPH activity reduced due to BH4 deficiency);
       ELEVATED BIOPTERIN in CSF (SPR block -> biopterin accumulates upstream -- cannot be reduced to BH4);
       ELEVATED SEPIAPTERIN in CSF (direct substrate of SPR; accumulates; pathognomonic);
       NORMAL PHENYLALANINE in blood (distinguishes SPR from GCH1 biallelic, PTS, QDPR which raise phe);
     DIURNAL VARIATION (like TH deficiency): symptoms worse in evening, better after sleep;
       mechanism: BH4 levels fluctuate with metabolic demand; residual BH4 depletes during day;
     SEROTONIN ALSO LOW (distinguishes SPR from TH deficiency: TH has normal 5-HIAA; SPR has LOW 5-HIAA);
     ORAL BH4 DOES NOT ADEQUATELY CROSS BBB: exogenous sapropterin (oral BH4) has limited CNS penetration;
     TREATMENT:
       L-DOPA/carbidopa (restores dopamine);
       5-HTP (5-hydroxytryptophan; bypasses TPH block -> restores serotonin);
       BOTH dopamine AND serotonin precursors needed (unlike TH where only L-DOPA sufficient);
     seed SEED_BASE+6).

GCH1 (GTP cyclohydrolase I; 250aa; 14q22.2; AD-LOF (haploinsufficiency);
     GCH1 encodes GTP cyclohydrolase I (GTPCH I), catalyzing the FIRST AND RATE-LIMITING step
     in BH4 (tetrahydrobiopterin) de novo biosynthesis: GTP -> 7,8-dihydroneopterin triphosphate;
     BH4 is the essential cofactor for TH (dopamine synthesis), TPH (serotonin), and PAH (phenylalanine);
     SEGAWA DISEASE = DOPA-RESPONSIVE DYSTONIA TYPE 1 (DRD1) -- OMIM 128230;
     INHERITANCE: AUTOSOMAL DOMINANT (haploinsufficiency); one mutant allele sufficient for disease;
       FEMALE:MALE PENETRANCE 4:1 (females more severely affected; mechanism unknown -- possibly hormonal);
       AR biallelic GCH1 complete LOF -> severe neonatal phenotype with hyperphenylalaninemia (like PTS);
     DIURNAL VARIATION PATHOGNOMONIC:
       symptoms WORSE by evening / BETTER after sleep or rest (classic Segawa description);
       mechanism: BH4 levels fall during waking hours (TH activity requires BH4 replenishment);
       dopamine stores deplete -> dystonia worsens by evening; replenished during overnight rest;
       this circadian pattern in a child with dystonia is the clinical signature of DRD;
     DRAMATIC L-DOPA RESPONSE AT LOW DOSE PATHOGNOMONIC:
       patients respond to L-DOPA at VERY LOW DOSES (1-2 mg/kg/day carbidopa/L-DOPA);
       contrast: Parkinson's disease requires 300-600+ mg/day L-DOPA;
       contrast: DYT1 torsion dystonia does NOT respond to L-DOPA;
       COMPLETE abolition of dystonia at low dose distinguishes DRD from all other dystonias;
     CSF NEUROTRANSMITTER PATTERN:
       LOW HVA (dopamine pathway impaired);
       LOW 5-HIAA (serotonin pathway impaired -- TPH also uses BH4);
       LOW NEOPTERIN + LOW BIOPTERIN in CSF (both BH4 pathway products DECREASED -- contrast with SPR: BIOPTERIN ELEVATED);
       NORMAL PHENYLALANINE (AD haploinsufficiency rarely sufficient to impair PAH;
         AR biallelic LOF -> elevated phe);
     TREATMENT:
       DRAMATIC response to low-dose L-DOPA/carbidopa; lifelong; NO tolerance develops;
       BH4 (sapropterin) may provide adjunct benefit;
       prognosis EXCELLENT -- most patients achieve near-normal function;
     seed SEED_BASE+7).

SEED_BASE = 4230
"""

import json
import random
from typing import Any


ATLAS_ID   = "hereditary-biogenic-amine-catecholamine-synthesis-transport-atlas"
ATLAS_NAME = "Hereditary-Biogenic-Amine-Catecholamine-Synthesis-Transport-Atlas"
SEED_BASE  = 4230
N_PER_GENE = 40

GENES = ["TH", "DDC", "DBH", "SLC18A2", "SLC6A3", "MAOA", "SPR", "GCH1"]

GENE_META = {
    "TH":      {"aa": 528, "locus": "11p15.5", "kda": 45, "inh": "AR",      "disease": "TH Deficiency (Tyrosine Hydroxylase Deficiency / Dopamine-Responsive Dystonia type 3)",           "omim_gene": 191290, "omim_disease": 605407},
    "DDC":     {"aa": 480, "locus": "7p12.2",  "kda": 54, "inh": "AR",      "disease": "AADC Deficiency (Aromatic L-Amino Acid Decarboxylase Deficiency)",                                  "omim_gene": 107930, "omim_disease": 608643},
    "DBH":     {"aa": 617, "locus": "9q33.2",  "kda": 75, "inh": "AR",      "disease": "DBH Deficiency (Dopamine Beta-Hydroxylase Deficiency / Norepinephrine Deficiency)",                 "omim_gene": 223360, "omim_disease": 223360},
    "SLC18A2": {"aa": 514, "locus": "10q25.3", "kda": 55, "inh": "AR",      "disease": "DTDS type 2 / Infantile Parkinsonism-Dystonia 2 (VMAT2 Deficiency)",                                "omim_gene": 193001, "omim_disease": 618049},
    "SLC6A3":  {"aa": 620, "locus": "5p15.33", "kda": 69, "inh": "AR",      "disease": "DTDS type 1 / DAT Deficiency Syndrome (Dopamine Transporter Deficiency)",                           "omim_gene": 126455, "omim_disease": 613135},
    "MAOA":    {"aa": 527, "locus": "Xp11.3",  "kda": 60, "inh": "XLR",     "disease": "Brunner Syndrome (Monoamine Oxidase A Deficiency / X-linked Impulsive Aggression)",                 "omim_gene": 309850, "omim_disease": 300615},
    "SPR":     {"aa": 261, "locus": "2p13.2",  "kda": 28, "inh": "AR",      "disease": "Sepiapterin Reductase Deficiency (BH4 Synthesis Defect; Dopamine + Serotonin Deficiency)",          "omim_gene": 182125, "omim_disease": 612716},
    "GCH1":    {"aa": 250, "locus": "14q22.2", "kda": 28, "inh": "AD-LOF",  "disease": "Segawa Disease / Dopa-Responsive Dystonia type 1 (GCH1 Haploinsufficiency)",                        "omim_gene": 600225, "omim_disease": 128230},
}

GENE_PATHOGNOMONIC = {
    "TH":      "DIURNAL VARIATION PATHOGNOMONIC in mild TH deficiency (DRD-TH): dystonia worse in evening, better after sleep; CSF: LOW HVA + LOW MHPG + NORMAL 5-HIAA (serotonin NORMAL distinguishes TH from AADC and SPR); PHENOTYPIC SPECTRUM: severe TH0 (L-DOPA-nonresponsive, Leigh-like) to mild DRD3 (L-DOPA-responsive, diurnal); NORMAL pterins in CSF (distinguishes from BH4 pathway defects)",
    "DDC":     "OCULOGYRIC CRISES PATHOGNOMONIC (involuntary sustained upward eye deviation, extreme distress); DUAL SUBSTRATE defect (dopamine AND serotonin synthesis blocked); CSF PATHOGNOMONIC: LOW HVA + LOW 5-HIAA + ELEVATED L-DOPA + ELEVATED 5-HTP (precursors accumulate); AADC plasma enzyme activity confirmatory; AAV2-DDC intraputaminal gene therapy clinically transformative",
    "DBH":     "NOREPINEPHRINE ABSENT + DOPAMINE MARKEDLY ELEVATED PATHOGNOMONIC (serum/urine); SEVERE ORTHOSTATIC HYPOTENSION (syncope, ptosis, nasal congestion, retrograde ejaculation); UNDETECTABLE serum DBH enzyme activity; DROXIDOPA (L-DOPS) CURATIVE -- FDA-approved; synthetic NE precursor converted peripherally by AADC; first presentation often adolescent syncope",
    "SLC18A2": "ALL MONOAMINES DEPLETED (dopamine + serotonin + NE + epinephrine) -- global depletion distinguishes VMAT2 from gene-specific defects; DOPAMINE SPECT: REDUCED PRESYNAPTIC UPTAKE in INFANT (like PD but infant) PATHOGNOMONIC; CSF: LOW HVA + LOW 5-HIAA; infantile Parkinsonism-dystonia; oculogyric crises; POOR LONG-TERM PROGNOSIS with current pharmacology",
    "SLC6A3":  "CSF HVA PARADOXICALLY ELEVATED in parkinsonism-dystonia PATHOGNOMONIC (dopamine reuptake failure -> increased turnover -> HVA elevated despite clinical parkinsonism); DAT-SPECT (123I-ioflupane): ABSENT DAT binding DIAGNOSTIC; contrast with VMAT2: HVA LOW in VMAT2 vs HVA HIGH in DAT deficiency; neonatal hypotonia -> infantile parkinsonism",
    "MAOA":    "MALES ONLY (hemizygous X-linked recessive); BORDERLINE INTELLECTUAL DISABILITY + IMPULSIVE AGGRESSIVE BEHAVIOR PATHOGNOMONIC (Brunner syndrome); ELEVATED URINARY: normetanephrine + HVA + DOPAC + 5-HIAA (all monoamine metabolites elevated -- degradation blocked); CARRIER FEMALES ASYMPTOMATIC; MAO-A inhibitors CONTRAINDICATED (would worsen accumulation)",
    "SPR":     "DISTINCTIVE CSF PATTERN: LOW HVA + LOW 5-HIAA + ELEVATED BIOPTERIN + ELEVATED SEPIAPTERIN (substrate of SPR accumulates); DIURNAL VARIATION (like TH/GCH1); SEROTONIN ALSO LOW (unlike TH; key distinguisher); NORMAL PHENYLALANINE (contrast with GCH1 biallelic or PTS/QDPR); ORAL BH4 DOES NOT ADEQUATELY CROSS BBB; TREATMENT: BOTH L-DOPA AND 5-HTP required",
    "GCH1":    "SEGAWA DISEASE / DRD1; AD HAPLOINSUFFICIENCY (one allele); DIURNAL VARIATION PATHOGNOMONIC (worse evening, better after sleep); DRAMATIC LOW-DOSE L-DOPA RESPONSE PATHOGNOMONIC (1-2 mg/kg/day; distinguishes from PD needing high dose and DYT1 non-responsive); CSF: LOW HVA + LOW 5-HIAA + LOW NEOPTERIN + LOW BIOPTERIN (ALL decreased; contrast SPR: BIOPTERIN ELEVATED); NORMAL PHENYLALANINE (AD); FEMALE:MALE 4:1 penetrance; EXCELLENT PROGNOSIS",
}

GENE_GROUPS = {
    "TH":      "dopamine-synthesis",
    "DDC":     "dopamine-serotonin-synthesis",
    "DBH":     "norepinephrine-synthesis",
    "SLC18A2": "vesicular-transport",
    "SLC6A3":  "dopamine-reuptake",
    "MAOA":    "monoamine-degradation",
    "SPR":     "bh4-synthesis-distal",
    "GCH1":    "bh4-synthesis-proximal",
}

SYNTHESIS_GROUP      = ["TH", "DDC", "DBH"]
TRANSPORT_GROUP      = ["SLC18A2", "SLC6A3"]
DEGRADATION_GROUP    = ["MAOA"]
BH4_SYNTHESIS_GROUP  = ["SPR", "GCH1"]


def _make_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    meta = GENE_META[gene]

    has_oculogyric_crises       = gene in ("DDC", "SLC18A2")
    has_orthostatic_hypotension = gene == "DBH"
    has_diurnal_variation       = gene in ("TH", "GCH1")
    has_dramatic_ldopa_response = gene == "GCH1"
    x_linked                    = gene == "MAOA"
    csf_biopterin_elevated      = gene == "SPR"
    csf_hva_low                 = gene in ("TH", "DDC", "DBH", "SLC18A2", "SPR", "GCH1")
    csf_hva_elevated            = gene == "SLC6A3"  # paradox
    csf_5hiaa_low               = gene in ("DDC", "SLC18A2", "SPR", "GCH1")
    all_monoamines_depleted     = gene == "SLC18A2"
    ne_absent_dopamine_high     = gene == "DBH"
    gene_therapy_available      = gene == "DDC"
    severe_neonatal             = gene in ("SLC18A2", "SLC6A3")

    # Key CSF patterns per gene
    key_csf_pattern = {
        "TH":      "HVA LOW, MHPG LOW, 5-HIAA NORMAL, pterins NORMAL",
        "DDC":     "HVA LOW, 5-HIAA LOW, L-DOPA ELEVATED, 5-HTP ELEVATED",
        "DBH":     "plasma/urine: NE ABSENT, dopamine MARKEDLY ELEVATED; CSF: relatively normal",
        "SLC18A2": "HVA LOW, 5-HIAA LOW (all monoamines depleted)",
        "SLC6A3":  "HVA PARADOXICALLY ELEVATED, 5-HIAA low-normal",
        "MAOA":    "urine: 5-HIAA HIGH, HVA HIGH, DOPAC HIGH, normetanephrine HIGH",
        "SPR":     "HVA LOW, 5-HIAA LOW, biopterin ELEVATED, sepiapterin ELEVATED",
        "GCH1":    "HVA LOW, 5-HIAA LOW, neopterin LOW, biopterin LOW",
    }

    treatment_map = {
        "TH":      ["L-DOPA_carbidopa_low_dose", "titrate_slowly_severe_limited"],
        "DDC":     ["dopamine_agonist_pramipexole", "MAO-B_inhibitor_selegiline", "pyridoxine_B6", "AAV2-DDC_gene_therapy"],
        "DBH":     ["droxidopa_L-DOPS", "fludrocortisone_OH", "midodrine_OH"],
        "SLC18A2": ["carbidopa_L-DOPA_partial", "dopamine_agonists", "MAO_inhibitors"],
        "SLC6A3":  ["L-DOPA_partial", "carbidopa", "pramipexole", "gene_therapy_investigational"],
        "MAOA":    ["behavioral_management", "SSRIs_cautious_anxiety", "low_monoamine_diet"],
        "SPR":     ["L-DOPA_carbidopa", "5-HTP_serotonin_precursor", "both_dopamine_AND_serotonin"],
        "GCH1":    ["L-DOPA_carbidopa_dramatic_low_dose", "BH4_sapropterin_adjunct", "lifelong_excellent_prognosis"],
    }

    # Age ranges by gene
    if gene in ("SLC18A2", "SLC6A3"):
        ages = [rng.uniform(0, 1) for _ in range(N_PER_GENE)]           # neonatal/infantile
    elif gene == "DBH":
        ages = [rng.uniform(5, 30) for _ in range(N_PER_GENE)]          # adolescent/young adult syncope
    elif gene == "MAOA":
        ages = [rng.uniform(5, 40) for _ in range(N_PER_GENE)]          # childhood behavioural issues
    elif gene == "GCH1":
        ages = [rng.uniform(2, 15) for _ in range(N_PER_GENE)]          # childhood dystonia
    elif gene == "TH":
        ages = [rng.uniform(0, 5) for _ in range(N_PER_GENE)]
    else:
        ages = [rng.uniform(0, 10) for _ in range(N_PER_GENE)]

    # CSF HVA (nmol/L): normal ~100-600; low in most; elevated in SLC6A3
    csf_hva_vals = []
    for _ in range(N_PER_GENE):
        if csf_hva_elevated:
            csf_hva_vals.append(rng.uniform(800, 2500))    # paradoxically elevated
        elif csf_hva_low:
            csf_hva_vals.append(rng.uniform(8, 80))        # clearly low
        else:
            csf_hva_vals.append(rng.uniform(150, 600))     # relatively normal (DBH, MAOA)

    # Motor score 0-10 (higher = worse)
    motor_scores = []
    for _ in range(N_PER_GENE):
        if gene in ("SLC18A2",):
            motor_scores.append(rng.randint(7, 10))        # severe
        elif gene in ("DDC", "SLC6A3"):
            motor_scores.append(rng.randint(5, 9))
        elif gene == "GCH1":
            motor_scores.append(rng.randint(2, 7))         # responds to treatment
        elif gene == "MAOA":
            motor_scores.append(rng.randint(1, 4))         # not primary motor disorder
        else:
            motor_scores.append(rng.randint(3, 8))

    # Treatment response 0-100%
    treatment_response = []
    for i in range(N_PER_GENE):
        if gene == "GCH1":
            treatment_response.append(rng.uniform(75, 100))  # dramatic
        elif gene == "DBH":
            treatment_response.append(rng.uniform(70, 95))   # droxidopa curative
        elif gene == "TH" and motor_scores[i] < 6:
            treatment_response.append(rng.uniform(50, 85))   # mild DRD-TH responds
        elif gene == "TH" and motor_scores[i] >= 7:
            treatment_response.append(rng.uniform(5, 30))    # severe TH0 limited
        elif gene == "DDC":
            treatment_response.append(rng.uniform(30, 80))   # agonist partial; gene therapy better
        elif gene == "SLC18A2":
            treatment_response.append(rng.uniform(10, 40))   # poor prognosis
        elif gene == "SPR":
            treatment_response.append(rng.uniform(50, 85))   # both precursors help
        else:
            treatment_response.append(rng.uniform(30, 70))

    # Severe phenotype boolean
    severe_phenotype = []
    for i in range(N_PER_GENE):
        if gene == "SLC18A2":
            severe_phenotype.append(True)
        elif gene == "TH" and motor_scores[i] >= 7:
            severe_phenotype.append(True)
        elif gene == "DDC" and motor_scores[i] >= 8:
            severe_phenotype.append(True)
        elif rng.random() < 0.25:
            severe_phenotype.append(True)
        else:
            severe_phenotype.append(False)

    # Gender: MAOA is male-only (X-linked recessive)
    genders = []
    for _ in range(N_PER_GENE):
        if x_linked:
            genders.append("M")    # hemizygous males only clinically affected
        else:
            genders.append(rng.choice(["M", "F"]))

    patients = []
    for i in range(N_PER_GENE):
        p = {
            "id":                       f"{gene}-{seed}-{i:03d}",
            "seed":                     seed,
            "gene":                     gene,
            "gender":                   genders[i],
            "age_dx":                   round(ages[i], 2),
            "csf_hva_nmol_L":           round(csf_hva_vals[i], 1),
            "motor_score":              motor_scores[i],
            "treatment_response_pct":   round(treatment_response[i], 1),
            "severe_phenotype":         severe_phenotype[i],
            "treatment":                rng.choice(treatment_map[gene]),
            "has_oculogyric_crises":    has_oculogyric_crises and rng.random() < 0.80,
            "has_orthostatic_hypotension": has_orthostatic_hypotension,
            "has_diurnal_variation":    has_diurnal_variation and rng.random() < 0.85,
            "csf_hva_low":              csf_hva_low,
            "csf_hva_elevated":         csf_hva_elevated,
            "csf_5hiaa_low":            csf_5hiaa_low,
            "csf_biopterin_elevated":   csf_biopterin_elevated,
            "x_linked_male":            x_linked,
            "key_csf_pattern":          key_csf_pattern[gene],
            "biogenic_amine_group":     GENE_GROUPS[gene],
        }
        patients.append(p)

    return {
        "gene":                         gene,
        "seed":                         seed,
        "n":                            N_PER_GENE,
        "disease":                      meta["disease"],
        "pathognomonic":                GENE_PATHOGNOMONIC[gene],
        "biogenic_amine_group":         GENE_GROUPS[gene],
        "omim_gene":                    meta["omim_gene"],
        "omim_disease":                 meta["omim_disease"],
        "aa":                           meta["aa"],
        "kda":                          meta["kda"],
        "locus":                        meta["locus"],
        "inh":                          meta["inh"],
        "has_oculogyric_crises":        has_oculogyric_crises,
        "has_orthostatic_hypotension":  has_orthostatic_hypotension,
        "has_diurnal_variation":        has_diurnal_variation,
        "has_dramatic_ldopa_response":  has_dramatic_ldopa_response,
        "x_linked":                     x_linked,
        "csf_biopterin_elevated":       csf_biopterin_elevated,
        "csf_hva_low":                  csf_hva_low,
        "csf_hva_elevated":             csf_hva_elevated,
        "csf_5hiaa_low":                csf_5hiaa_low,
        "all_monoamines_depleted":      all_monoamines_depleted,
        "ne_absent_dopamine_high":      ne_absent_dopamine_high,
        "gene_therapy_available":       gene_therapy_available,
        "severe_neonatal":              severe_neonatal,
        "key_csf_pattern":              key_csf_pattern[gene],
        "mean_csf_hva":                 round(sum(csf_hva_vals) / len(csf_hva_vals), 1),
        "mean_treatment_response":      round(sum(treatment_response) / len(treatment_response), 1),
        "pct_severe":                   round(sum(1 for s in severe_phenotype if s) / len(severe_phenotype), 3),
        "patients":                     patients,
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]

    gene_table = []
    for c in cohorts:
        gene_table.append({
            "gene":                         c["gene"],
            "disease":                      c["disease"],
            "locus":                        c["locus"],
            "aa":                           c["aa"],
            "kda":                          c["kda"],
            "inheritance":                  c["inh"],
            "biogenic_amine_group":         c["biogenic_amine_group"],
            "key_csf_pattern":              c["key_csf_pattern"],
            "has_oculogyric_crises":        c["has_oculogyric_crises"],
            "has_orthostatic_hypotension":  c["has_orthostatic_hypotension"],
            "has_diurnal_variation":        c["has_diurnal_variation"],
            "has_dramatic_ldopa_response":  c["has_dramatic_ldopa_response"],
            "x_linked":                     c["x_linked"],
            "csf_hva_low":                  c["csf_hva_low"],
            "csf_hva_elevated":             c["csf_hva_elevated"],
            "csf_5hiaa_low":                c["csf_5hiaa_low"],
            "csf_biopterin_elevated":       c["csf_biopterin_elevated"],
            "all_monoamines_depleted":      c["all_monoamines_depleted"],
            "ne_absent_dopamine_high":      c["ne_absent_dopamine_high"],
            "gene_therapy_available":       c["gene_therapy_available"],
            "mean_csf_hva":                 c["mean_csf_hva"],
            "mean_treatment_response":      c["mean_treatment_response"],
            "pct_severe":                   c["pct_severe"],
            "pathognomonic":                c["pathognomonic"],
            "omim_disease":                 c["omim_disease"],
        })

    return {
        "atlas_id":                     ATLAS_ID,
        "total_patients":               N_PER_GENE * len(GENES),
        "genes":                        GENES,
        "seeds":                        f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "oculogyric_crises_genes":      ["DDC", "SLC18A2"],
        "orthostatic_hypotension_gene": ["DBH"],
        "diurnal_variation_genes":      ["TH", "GCH1"],
        "dramatic_ldopa_response_gene": ["GCH1"],
        "x_linked_gene":                ["MAOA"],
        "csf_biopterin_elevated_gene":  ["SPR"],
        "csf_hva_paradox_gene":         ["SLC6A3"],
        "all_monoamines_depleted_gene": ["SLC18A2"],
        "ne_absent_gene":               ["DBH"],
        "gene_therapy_gene":            ["DDC"],
        "synthesis_group":              SYNTHESIS_GROUP,
        "transport_group":              TRANSPORT_GROUP,
        "degradation_group":            DEGRADATION_GROUP,
        "bh4_synthesis_group":          BH4_SYNTHESIS_GROUP,
        "primary_cascade_split": (
            "STEP 1: CSF HVA LOW + 5-HIAA NORMAL -> TH deficiency (check diurnal variation; trial L-DOPA). "
            "STEP 2: CSF HVA LOW + 5-HIAA LOW + L-DOPA ELEVATED + 5-HTP ELEVATED -> AADC (DDC) deficiency "
            "(oculogyric crises pathognomonic; AADC enzyme in plasma). "
            "STEP 3: Serum NE ABSENT + dopamine HIGH + orthostatic hypotension -> DBH deficiency (droxidopa curative). "
            "STEP 4: CSF HVA LOW + 5-HIAA LOW + DAT SPECT absent + infant Parkinsonism -> SLC6A3 (DAT) vs SLC18A2 (VMAT2) "
            "-- HVA HIGH in DAT (paradox) vs HVA LOW in VMAT2. "
            "STEP 5: XL male + aggression + elevated urinary monoamine metabolites -> MAOA Brunner syndrome. "
            "STEP 6: CSF HVA LOW + 5-HIAA LOW + BIOPTERIN ELEVATED + SEPIAPTERIN elevated + NORMAL phenylalanine -> SPR deficiency. "
            "STEP 7: DIURNAL DYSTONIA + DRAMATIC L-DOPA response + CSF neopterin+biopterin LOW + normal phenylal -> "
            "GCH1 Segawa (AD LOF); if biallelic severe -> check phenylalanine."
        ),
        "gene_table": gene_table,
    }


def generate_breakdown():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]

    return {
        "atlas_id": ATLAS_ID,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "biogenic_amine_groups": {
            "dopamine_synthesis": {
                "genes":       SYNTHESIS_GROUP,
                "description": (
                    "TH (rate-limiting, diurnal variation, L-DOPA responsive in mild DRD-TH; CSF: HVA low, 5-HIAA normal) + "
                    "DDC/AADC (dual substrate dopamine+serotonin; oculogyric crises pathognomonic; CSF: HVA+5-HIAA low + precursors elevated; AAV2 gene therapy) + "
                    "DBH (NE absent, dopamine high; orthostatic hypotension; droxidopa curative)"
                ),
                "key_csf_markers": {
                    "TH":  "HVA low; MHPG low; 5-HIAA NORMAL (serotonin spared)",
                    "DDC": "HVA low; 5-HIAA low; L-DOPA elevated; 5-HTP elevated",
                    "DBH": "plasma NE absent; dopamine elevated (serum/urine)",
                },
            },
            "vesicular_and_reuptake_transport": {
                "genes":       TRANSPORT_GROUP,
                "description": (
                    "SLC18A2/VMAT2 (all monoamines depleted; dopamine SPECT reduced in infant; poor prognosis) + "
                    "SLC6A3/DAT (HVA paradoxically ELEVATED; absent DAT-SPECT; infant parkinsonism)"
                ),
                "key_distinction": "SLC18A2: HVA LOW (cannot make/store dopamine); SLC6A3: HVA HIGH (cannot clear dopamine -> increased turnover; paradox)",
            },
            "monoamine_degradation": {
                "genes":       DEGRADATION_GROUP,
                "description": (
                    "MAOA/Brunner syndrome: X-linked recessive males only; impulsive aggression + borderline ID; "
                    "all monoamine metabolites elevated in urine; MAO-A inhibitors CONTRAINDICATED"
                ),
                "key_markers": {"MAOA": "urine normetanephrine + HVA + 5-HIAA all elevated"},
            },
            "bh4_synthesis": {
                "genes":       BH4_SYNTHESIS_GROUP,
                "description": (
                    "SPR (last BH4 synthesis step; biopterin ELEVATED; sepiapterin elevated; needs both L-DOPA + 5-HTP) + "
                    "GCH1 Segawa (first BH4 synthesis step; AD haploinsufficiency; biopterin LOW; diurnal; dramatic low-dose L-DOPA; excellent prognosis)"
                ),
                "key_distinction": "SPR: biopterin ELEVATED (substrate accumulates at block); GCH1: biopterin LOW (pathway depleted from start)",
            },
        },
        "csf_algorithm": {
            "HVA LOW + 5-HIAA NORMAL + pterins NORMAL": {
                "gene":   "TH",
                "action": "TH deficiency; check diurnal variation; trial L-DOPA/carbidopa; severity spectrum TH0 (severe) to DRD-TH (mild)",
            },
            "HVA LOW + 5-HIAA LOW + L-DOPA and 5-HTP ELEVATED": {
                "gene":   "DDC",
                "action": "AADC deficiency; look for oculogyric crises; measure plasma AADC activity; dopamine agonist + selegiline + pyridoxine; AAV2-DDC gene therapy referral",
            },
            "NE ABSENT + dopamine HIGH in serum/urine + orthostatic hypotension": {
                "gene":   "DBH",
                "action": "DBH deficiency; droxidopa (L-DOPS) curative; fludrocortisone + midodrine adjuncts",
            },
            "HVA LOW + 5-HIAA LOW + DAT SPECT reduced + all monoamines depleted": {
                "gene":   "SLC18A2",
                "action": "VMAT2/DTDS2; global monoamine depletion; L-DOPA partially; poor long-term prognosis; gene therapy investigational",
            },
            "HVA ELEVATED (paradox) + absent DAT-SPECT + infant parkinsonism": {
                "gene":   "SLC6A3",
                "action": "DAT deficiency/DTDS1; paradoxical HVA elevation (reuptake failure -> increased turnover); L-DOPA partial; gene therapy investigational",
            },
            "XL male + aggression + urine normetanephrine + HVA + 5-HIAA all elevated": {
                "gene":   "MAOA",
                "action": "Brunner syndrome; behavioral management primary; SSRIs cautious; MAO-A inhibitors ABSOLUTELY CONTRAINDICATED",
            },
            "HVA LOW + 5-HIAA LOW + CSF biopterin ELEVATED + sepiapterin elevated + normal phe": {
                "gene":   "SPR",
                "action": "SPR deficiency; start BOTH L-DOPA/carbidopa AND 5-HTP; oral BH4 inadequate CNS penetration",
            },
            "Diurnal dystonia + DRAMATIC low-dose L-DOPA response + CSF neopterin+biopterin LOW + normal phe": {
                "gene":   "GCH1",
                "action": "Segawa disease/DRD1; AD haploinsufficiency; L-DOPA/carbidopa 1-2 mg/kg/day; lifelong; excellent prognosis; no tolerance",
            },
        },
        "distinguishing_features": [
            {"gene": "TH",      "feature": "HVA low, 5-HIAA NORMAL (serotonin spared); diurnal variation; severity spectrum",   "action": "L-DOPA trial; mild DRD-TH responds; severe TH0 limited"},
            {"gene": "DDC",     "feature": "Oculogyric crises PATHOGNOMONIC; precursors elevated in CSF; AADC enzyme in plasma",  "action": "Agonist + selegiline + B6; refer for AAV2-DDC gene therapy"},
            {"gene": "DBH",     "feature": "NE absent + dopamine high; orthostatic hypotension; ptosis; syncope",                "action": "Droxidopa (L-DOPS) curative; FDA-approved"},
            {"gene": "SLC18A2", "feature": "All monoamines depleted; SPECT reduced in infant; poor prognosis",                   "action": "L-DOPA partial; supportive; gene therapy investigational"},
            {"gene": "SLC6A3",  "feature": "HVA ELEVATED (paradox); absent DAT-SPECT; infant parkinsonism",                     "action": "L-DOPA partial; contrast HVA HIGH vs SLC18A2 HVA LOW"},
            {"gene": "MAOA",    "feature": "X-linked males; aggression + borderline ID; urine metabolites ALL elevated",         "action": "Behavioral management; no MAO-A inhibitors (CI)"},
            {"gene": "SPR",     "feature": "Biopterin ELEVATED; sepiapterin elevated; needs L-DOPA + 5-HTP both",                "action": "Both dopamine AND serotonin precursors; oral BH4 inadequate"},
            {"gene": "GCH1",    "feature": "Diurnal; dramatic low-dose L-DOPA; biopterin LOW (not elevated); AD; 4:1 F:M",       "action": "Low-dose L-DOPA lifelong; BH4 adjunct; excellent prognosis"},
        ],
        "cohort_summary": [
            {
                "gene":                    c["gene"],
                "mean_csf_hva":            c["mean_csf_hva"],
                "mean_treatment_response": c["mean_treatment_response"],
                "pct_severe":              c["pct_severe"],
                "biogenic_amine_group":    c["biogenic_amine_group"],
            }
            for c in cohorts
        ],
    }


def generate_definitions():
    key_distinguishers = [
        {
            "name":               "Oculogyric crisis (DDC/AADC deficiency) — pathognomonic involuntary upward eye deviation",
            "definition":         (
                "An oculogyric crisis is a dystonic episode characterized by involuntary, sustained, upward (and occasionally lateral) "
                "conjugate deviation of the eyes, lasting minutes to hours. "
                "In AADC deficiency (DDC LOF), oculogyric crises are the HALLMARK presentation. "
                "Pathophysiology: dopamine deficiency in the basal ganglia-brainstem circuits controlling saccades leads to "
                "intermittent loss of voluntary gaze control; the eyes lock upward and the patient is in extreme distress, "
                "often with autonomic instability (diaphoresis, tachycardia, flushing). "
                "Crises may be triggered by emotion, fatigue, or intercurrent illness. "
                "CLINICAL RELEVANCE: any infant or child presenting with episodic upward eye deviation + hypotonia + autonomic features "
                "should prompt immediate CSF neurotransmitter analysis (HVA, 5-HIAA, L-DOPA, 5-HTP) and plasma AADC enzyme activity. "
                "DIFFERENTIAL: drug-induced oculogyric crises (metoclopramide, antipsychotics -- much more common; acute onset; resolves with anticholinergic). "
                "SLC18A2 (VMAT2) also causes oculogyric crises as part of global monoamine depletion."
            ),
            "clinical_relevance": "Episodic upward eye deviation in infant + hypotonia = AADC deficiency until proven otherwise; urgent CSF neurotransmitters",
        },
        {
            "name":               "Diurnal variation in movement disorder (TH deficiency, GCH1 Segawa) — pathognomonic circadian pattern",
            "definition":         (
                "Diurnal variation refers to worsening of motor symptoms (dystonia, parkinsonism, bradykinesia) as the day progresses, "
                "with characteristic improvement after sleep or prolonged rest. "
                "MECHANISM: dopamine and/or BH4 stores are finite; during waking hours, HVA (dopamine metabolite) levels fall "
                "as dopamine synthesis cannot keep pace with demand (due to deficient TH or GCH1/BH4). "
                "After overnight sleep, stores partially replenish -> symptoms improve dramatically in the morning. "
                "CLINICAL SIGNIFICANCE: "
                "(1) TH deficiency (mild DRD-TH / type 3): diurnal variation present; L-DOPA responsive; "
                "(2) GCH1 / Segawa disease (DRD-1): diurnal variation is PATHOGNOMONIC; L-DOPA response is dramatic at very low dose; "
                "any child with dystonia showing diurnal fluctuation MUST receive a low-dose L-DOPA trial before other diagnoses. "
                "The diurnal pattern also helps distinguish these from structural basal ganglia disease, DYT1, and spastic paraplegia, "
                "none of which show a circadian fluctuation."
            ),
            "clinical_relevance": "Diurnal worsening of dystonia in a child = mandatory low-dose L-DOPA trial (DRD until proven otherwise)",
        },
        {
            "name":               "BH4 (tetrahydrobiopterin) pathway — essential cofactor for dopamine, serotonin, and NO synthesis",
            "definition":         (
                "Tetrahydrobiopterin (BH4) is a pteridine cofactor ESSENTIAL for: "
                "(1) Tyrosine hydroxylase (TH): L-tyrosine -> L-DOPA (dopamine synthesis); "
                "(2) Tryptophan hydroxylase (TPH): L-tryptophan -> 5-HTP (serotonin synthesis); "
                "(3) Nitric oxide synthase (NOS): L-arginine -> citrulline + NO; "
                "(4) Phenylalanine hydroxylase (PAH): L-phenylalanine -> L-tyrosine. "
                "BH4 DE NOVO SYNTHESIS PATHWAY: GTP -> (GCH1, FIRST step) -> 7,8-dihydroneopterin-TP -> (PTS) -> "
                "6-pyruvoyl-tetrahydropterin -> (SPR, LAST step) -> BH4. "
                "GCH1 LOF (AD haploinsufficiency = Segawa): low neopterin + low biopterin + low HVA + low 5-HIAA. "
                "SPR LOF: biopterin ELEVATED (substrate accumulates) + sepiapterin elevated + low HVA + low 5-HIAA. "
                "TREATMENT IMPLICATION: oral BH4 (sapropterin) does NOT adequately cross the blood-brain barrier for CNS defects; "
                "L-DOPA and 5-HTP (precursor supplementation bypassing the block) are the primary treatments. "
                "NORMAL PHENYLALANINE distinguishes AD GCH1 from PAH deficiency (PKU), PTS, and QDPR defects (which elevate phe)."
            ),
            "clinical_relevance": "BH4 pathway defects cause combined dopamine + serotonin deficiency; oral BH4 inadequate for CNS; use L-DOPA + 5-HTP",
        },
        {
            "name":               "Dopamine transporter paradox (SLC6A3 / DAT deficiency) — elevated HVA in parkinsonism",
            "definition":         (
                "DAT (dopamine transporter, SLC6A3) is the presynaptic reuptake transporter that clears dopamine from the synapse. "
                "DAT deficiency (DTDS type 1) causes a PARADOX: "
                "EXCESS dopamine in the synapse -> postsynaptic D2 receptor downregulation (chronic excess -> desensitization) -> "
                "functional DOPAMINE SIGNALING FAILURE despite elevated synaptic dopamine -> CLINICAL PARKINSONISM. "
                "CSF HVA IS PARADOXICALLY ELEVATED: because dopamine cannot be reuptaken, it is metabolized more rapidly -> "
                "increased HVA (the metabolic product) in CSF despite clinical parkinsonism. "
                "CLINICAL RULE: elevated CSF HVA in an infant with parkinsonism-dystonia = SLC6A3 (DAT) deficiency "
                "until proven otherwise (OPPOSITE of all other dopamine deficiency syndromes where HVA is LOW). "
                "DAT-SPECT (123I-ioflupane): absent DAT binding (no DAT protein present) -- "
                "contrast with VMAT2 (SLC18A2): reduced (not absent) presynaptic uptake on SPECT."
            ),
            "clinical_relevance": "HVA elevated + infant parkinsonism + absent DAT-SPECT = SLC6A3; HVA elevated = diagnostic in context of parkinsonism",
        },
        {
            "name":               "Brunner syndrome (MAOA deficiency) — X-linked impulsive aggression in males",
            "definition":         (
                "Brunner syndrome (OMIM 300615) results from hemizygous LOF of MAOA (monoamine oxidase A) on Xp11.3. "
                "MAOA degrades serotonin, dopamine, norepinephrine, and epinephrine in the mitochondrial outer membrane. "
                "MAOA LOF -> ALL monoamines accumulate: serotonin, dopamine, NE, epinephrine all elevated. "
                "CLINICAL: MALES ONLY (X-linked recessive hemizygous); carrier females asymptomatic (X-inactivation). "
                "Features: borderline intellectual disability (mild, not severe) + "
                "IMPULSIVE AGGRESSIVE BEHAVIOR -- severe, episodic violence triggered by frustration; "
                "the original Dutch family had episodic arson, attempted rape, aggression; "
                "may present to behavioral health or criminal justice before metabolic diagnosis. "
                "BIOCHEMISTRY: urinary normetanephrine, HVA, DOPAC, 5-HIAA ALL ELEVATED. "
                "TREATMENT: behavioral management is primary; MAO-A inhibitors ABSOLUTELY CONTRAINDICATED (would worsen monoamine overload); "
                "SSRIs used cautiously for anxiety component (monitor for serotonin syndrome given impaired MAO-A degradation of serotonin)."
            ),
            "clinical_relevance": "Males with borderline ID + episodic aggression + elevated urinary monoamine metabolites -> MAOA gene analysis; never give MAO-A inhibitors",
        },
        {
            "name":               "L-DOPS (droxidopa) for DBH deficiency — synthetic norepinephrine precursor curative therapy",
            "definition":         (
                "Droxidopa (L-DOPS, (threo)-3,4-dihydroxyphenylserine) is a synthetic amino acid converted to norepinephrine "
                "by AADC (aromatic L-amino acid decarboxylase) PERIPHERALLY, bypassing the need for DBH. "
                "In DBH deficiency, dopamine cannot be converted to NE (DBH absent) -> NE/epinephrine absent -> "
                "severe orthostatic hypotension (no sympathetic vasomotor tone), ptosis (no Muller's muscle NE), retrograde ejaculation. "
                "L-DOPS BYPASSES DBH: L-DOPS -> (AADC, present and functional) -> NOREPINEPHRINE directly. "
                "CLINICAL RESPONSE: CURATIVE -- orthostatic hypotension resolves, ptosis improves, ejaculation normalizes. "
                "FDA-APPROVED for DBH deficiency-associated neurogenic orthostatic hypotension (also approved for nOH from MSA, PD, PAF). "
                "DOSE: 100-600 mg three times daily; titrate to standing BP. "
                "CONTRAST with TH deficiency treatment (L-DOPA/carbidopa) and with fludrocortisone/midodrine (symptomatic OH treatment without NE restoration)."
            ),
            "clinical_relevance": "DBH deficiency: droxidopa (L-DOPS) is CURATIVE -- restores NE; first-line and FDA-approved; reverses OH, ptosis, retrograde ejaculation",
        },
        {
            "name":               "CSF neurotransmitter metabolites (HVA, 5-HIAA, MHPG, neopterin, biopterin) — the diagnostic panel",
            "definition":         (
                "Cerebrospinal fluid (CSF) neurotransmitter metabolite analysis is the KEY DIAGNOSTIC TEST for biogenic amine disorders. "
                "ANALYTES: "
                "(1) HVA (homovanillic acid): dopamine metabolite (DA -> DOPAC -> HVA via MAO + COMT); "
                "LOW in TH, DDC, SLC18A2, SPR, GCH1; ELEVATED in SLC6A3 (paradox); normal in DBH, MAOA; "
                "(2) 5-HIAA (5-hydroxyindoleacetic acid): serotonin metabolite (5-HT -> 5-HIAA via MAO); "
                "LOW in DDC, SLC18A2, SPR, GCH1; NORMAL in TH (serotonin synthesis intact via TPH+BH4 if BH4 sufficient), DBH, MAOA; "
                "(3) MHPG (3-methoxy-4-hydroxyphenylglycol): norepinephrine metabolite; LOW in TH (NE deficient); "
                "(4) Neopterin: BH4 synthesis pathway intermediate (before GCH1 product); LOW in GCH1; normal/elevated in SPR; "
                "(5) Biopterin: BH4 precursor; LOW in GCH1; ELEVATED in SPR (substrate accumulation); "
                "(6) L-DOPA: TH product / AADC substrate; ELEVATED in DDC (substrate accumulates); "
                "(7) 5-HTP: TPH product / AADC substrate; ELEVATED in DDC (substrate accumulates). "
                "COLLECTION: lumbar puncture; sample must be processed immediately (analytes unstable); "
                "tubes must be frozen; specialized neurotransmitter metabolite lab required; "
                "age-specific reference ranges mandatory (HVA/5-HIAA highest in neonates, fall with age)."
            ),
            "clinical_relevance": "CSF neurotransmitter panel is the gold standard for biogenic amine defects; age-specific ranges required; collect with specialized protocol",
        },
        {
            "name":               "Sepiapterin reductase vs GCH1 — biopterin ELEVATED vs LOW: the critical BH4 diagnostic distinction",
            "definition":         (
                "SPR and GCH1 both cause combined dopamine + serotonin deficiency with diurnal variation. "
                "THE CRITICAL DISTINCTION IS THE CSF BIOPTERIN PATTERN: "
                "SPR (last step in BH4 synthesis): biopterin ELEVATED in CSF -- because the block is AFTER biopterin formation; "
                "BH4 cannot be made from biopterin (SPR block) -> biopterin accumulates; sepiapterin also elevated; "
                "GCH1 (first step in BH4 synthesis): biopterin LOW in CSF -- because the block is BEFORE biopterin formation; "
                "the entire BH4 pathway is depleted from the start; neopterin also low. "
                "ADDITIONAL DISTINCTION: "
                "SPR: NORMAL PHENYLALANINE (peripheral phenylalanine hydroxylase has BH4 from alternative pathways); "
                "GCH1 (AD Segawa): NORMAL PHENYLALANINE (one allele sufficient for PAH); "
                "GCH1 (AR biallelic): ELEVATED PHENYLALANINE (like severe GTPCH deficiency/PKU-like). "
                "TREATMENT DIFFERENCE: BOTH require L-DOPA; SPR requires 5-HTP (serotonin precursor) because both dopamine AND serotonin are affected; "
                "GCH1 Segawa responds to L-DOPA alone at very low dose with dramatic effect."
            ),
            "clinical_relevance": "SPR: biopterin HIGH, sepiapterin elevated, normal phe -> needs L-DOPA + 5-HTP; GCH1: biopterin LOW, neopterin LOW -> low-dose L-DOPA dramatic",
        },
        {
            "name":               "Gene therapy (AAV2-DDC) for AADC deficiency — the most advanced neurotransmitter gene therapy",
            "definition":         (
                "AAV2-DDC (adeno-associated virus serotype 2 carrying DDC/AADC transgene) delivered by bilateral intraputaminal injection "
                "is the most clinically advanced gene therapy for a neurotransmitter biosynthesis disorder. "
                "PROCEDURE: stereotactic neurosurgical delivery; bilateral injection into putamen under MRI guidance; "
                "outpatient recovery; single intervention. "
                "MECHANISM: AAV2 transduces putaminal neurons -> AADC enzyme expressed locally -> "
                "L-DOPA (given exogenously or from TH activity) converted to DOPAMINE within putamen. "
                "CLINICAL RESULTS: marked improvement in oculogyric crises (most dramatic; often eliminated); "
                "motor function improvement (head control, sitting, reduced hypotonia); "
                "quality of life improvement; some patients achieve milestones not previously possible; "
                "effects durable (5+ year follow-up data). "
                "CANDIDATE SELECTION: AADC-deficient patients confirmed by plasma AADC enzyme activity < 5 nmol/hr/mg + "
                "pathogenic DDC variants; age generally 18 months to 15 years for established trials. "
                "FDA/EMA STATUS: approved in EU (2022, eladocagene exuparvovec / Upstaza); under review in US. "
                "COMPARISON: other neurotransmitter gene therapies (SLC6A3, SLC18A2) are earlier stage investigational."
            ),
            "clinical_relevance": "AADC deficiency patients should be referred early to gene therapy centers; AAV2-DDC eliminates oculogyric crises and improves motor function",
        },
        {
            "name":               "Primary cascade split algorithm — systematic diagnosis of biogenic amine disorders",
            "definition":         (
                "A stepwise CSF-based diagnostic algorithm for biogenic amine catecholamine disorders: "
                "STEP 1: CSF HVA LOW + 5-HIAA NORMAL -> TH deficiency (serotonin intact = TH-specific); check diurnal variation; L-DOPA trial. "
                "STEP 2: CSF HVA LOW + 5-HIAA LOW + L-DOPA ELEVATED + 5-HTP ELEVATED -> AADC/DDC deficiency (precursors accumulate; dual synthesis block); oculogyric crises; AADC plasma enzyme. "
                "STEP 3: serum/urine NE ABSENT + dopamine ELEVATED + orthostatic hypotension -> DBH deficiency (droxidopa curative). "
                "STEP 4: CSF HVA LOW + 5-HIAA LOW + DAT SPECT reduced/absent + infant parkinsonism -> "
                "   STEP 4a: HVA ELEVATED (paradox) + absent DAT-SPECT -> SLC6A3 (DAT deficiency; reuptake failure); "
                "   STEP 4b: HVA LOW + reduced (not absent) SPECT + ALL monoamines low -> SLC18A2 (VMAT2; global monoamine depletion). "
                "STEP 5: XL male + episodic aggression + borderline ID + urine normetanephrine + HVA + 5-HIAA all elevated -> MAOA (Brunner). "
                "STEP 6: CSF HVA LOW + 5-HIAA LOW + biopterin ELEVATED + sepiapterin elevated + normal phenylalanine -> SPR deficiency (BH4 distal block; needs L-DOPA + 5-HTP). "
                "STEP 7: diurnal dystonia + DRAMATIC low-dose L-DOPA response + CSF neopterin LOW + biopterin LOW + normal phenylalanine -> GCH1 Segawa (AD haploinsufficiency; BH4 proximal block; L-DOPA alone sufficient at very low dose). "
                "KEY RULE: biopterin elevated = SPR; biopterin low = GCH1."
            ),
            "clinical_relevance": "Use this stepwise CSF algorithm for any infant/child with movement disorder + monoamine deficiency; biopterin direction distinguishes SPR from GCH1",
        },
    ]

    treatment_matrix = [
        {
            "intervention":    "L-DOPA/carbidopa",
            "indications":     "TH deficiency (DRD-TH mild: dramatic; severe TH0: limited); GCH1 Segawa (dramatic at very low dose, 1-2 mg/kg/day); SPR (in combination with 5-HTP); SLC6A3/SLC18A2 (partial benefit)",
            "mechanism":       "L-DOPA crosses BBB -> converted to dopamine by residual AADC activity; bypasses rate-limiting TH or BH4 step; carbidopa inhibits peripheral decarboxylation (preserves L-DOPA for CNS)",
            "dose":            "GCH1/DRD: start 1-2 mg/kg/day (VERY LOW); TH DRD-TH: start low, titrate; TH0: higher doses, limited response; lifelong; no tolerance in GCH1",
            "contraindicated": "DDC/AADC deficiency: L-DOPA cannot be converted (AADC blocked); not primary treatment; MAOA: use with caution (impaired monoamine degradation)",
        },
        {
            "intervention":    "Dopamine agonists (pramipexole, ropinirole)",
            "indications":     "DDC/AADC deficiency (primary -- bypasses synthesis block by directly stimulating D2/D3 receptors); SLC18A2 adjunct; SLC6A3 adjunct",
            "mechanism":       "Directly stimulate postsynaptic dopamine receptors (D2/D3 preferentially) bypassing the need for dopamine synthesis; no AADC conversion required",
            "dose":            "AADC: pramipexole 0.1-1 mg/kg/day, titrate; ropinirole equivalent dosing; start low",
            "contraindicated": "MAOA: dopamine agonists increase dopaminergic tone which may worsen monoamine overload; use with caution; not first-line for MAOA",
        },
        {
            "intervention":    "MAO-B inhibitor (selegiline)",
            "indications":     "DDC/AADC deficiency (reduces dopamine degradation by MAO-B; extends effect of residual dopamine); adjunct in TH deficiency",
            "mechanism":       "Irreversible MAO-B inhibition -> reduces dopamine (and phenylethylamine) degradation -> prolongs dopaminergic effect; selective for MAO-B at low doses (spares serotonin degradation)",
            "dose":            "Selegiline 0.05-0.1 mg/kg/day; avoid tyramine-rich foods (cheese reaction risk at higher doses); monitor BP",
            "contraindicated": "MAOA deficiency: MAO inhibitors of ANY type ABSOLUTELY CONTRAINDICATED; would cause catastrophic monoamine accumulation and hypertensive crisis/serotonin syndrome",
        },
        {
            "intervention":    "Droxidopa (L-DOPS)",
            "indications":     "DBH deficiency (SPECIFIC; curative); FDA-approved for DBH-deficiency neurogenic orthostatic hypotension; also approved for nOH from MSA/PD/PAF",
            "mechanism":       "Synthetic amino acid converted peripherally by AADC to norepinephrine; bypasses DBH block; restores NE stores in sympathetic neurons",
            "dose":            "Adults: 100 mg TID initially; titrate to 600 mg TID; take with meals; monitor standing BP; hold if supine hypertension",
            "contraindicated": "Not indicated for other biogenic amine defects; monitor for supine hypertension (head-of-bed elevation at night)",
        },
        {
            "intervention":    "5-HTP (5-hydroxytryptophan)",
            "indications":     "SPR deficiency (MANDATORY -- serotonin pathway also impaired; TH/L-DOPA alone insufficient); GCH1 if 5-HIAA also low (rare in AD; consider in AR biallelic)",
            "mechanism":       "5-HTP crosses BBB -> converted to serotonin by AADC (DDC); bypasses TPH + BH4 requirement for serotonin synthesis",
            "dose":            "SPR: 5-HTP 1-5 mg/kg/day TID; always combine with low-dose carbidopa (peripheral decarboxylation inhibitor) to prevent peripheral 5-HTP conversion",
            "contraindicated": "MAOA deficiency: 5-HTP would worsen serotonin overload; serotonin syndrome risk; NEVER give to MAOA patients",
        },
        {
            "intervention":    "AAV2-DDC gene therapy (eladocagene exuparvovec)",
            "indications":     "AADC deficiency (DDC LOF confirmed); approved EU; investigational US; bilateral intraputaminal injection; single intervention",
            "mechanism":       "AAV2 serotype transduces putaminal neurons; DDC transgene expressed locally; AADC activity restored in striatum; converts endogenous L-DOPA to dopamine",
            "dose":            "Single surgical procedure; bilateral stereotactic intraputaminal injection; EU-approved dose eladocagene exuparvovec 1.8x10^11 vg total",
            "contraindicated": "Not applicable to other biogenic amine disorders; active CNS infection; anti-AAV2 neutralizing antibodies (pre-screen); coagulopathy",
        },
    ]

    return {
        "atlas_id":             ATLAS_ID,
        "key_distinguishers":   key_distinguishers,
        "treatment_matrix":     treatment_matrix,
        "seeds":                f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "primary_cascade_split": (
            "STEP 1: CSF HVA low + 5-HIAA NORMAL -> TH deficiency (serotonin spared; diurnal variation; L-DOPA trial). "
            "STEP 2: CSF HVA low + 5-HIAA low + L-DOPA elevated + 5-HTP elevated -> AADC/DDC (oculogyric crises; AADC plasma enzyme; AAV2 gene therapy). "
            "STEP 3: NE absent + dopamine high + orthostatic hypotension -> DBH (droxidopa curative). "
            "STEP 4a: HVA ELEVATED (paradox) + absent DAT-SPECT + infant parkinsonism -> SLC6A3 (DAT deficiency). "
            "STEP 4b: HVA low + all monoamines depleted + reduced SPECT + infant -> SLC18A2 (VMAT2; poor prognosis). "
            "STEP 5: XL male + aggression + all urine metabolites elevated -> MAOA Brunner (NO MAO inhibitors). "
            "STEP 6: HVA low + 5-HIAA low + biopterin ELEVATED + sepiapterin elevated + normal phe -> SPR (needs L-DOPA + 5-HTP both). "
            "STEP 7: Diurnal dystonia + dramatic low-dose L-DOPA + biopterin LOW + neopterin LOW + normal phe -> GCH1 Segawa (AD; L-DOPA alone 1-2 mg/kg; excellent prognosis). "
            "KEY RULE: biopterin ELEVATED = SPR; biopterin LOW = GCH1."
        ),
        "biogenic_amine_groups": {
            "synthesis":         {"genes": SYNTHESIS_GROUP,     "primary_markers": "HVA low (TH, DDC); NE absent (DBH); L-DOPA/5-HTP elevated (DDC)"},
            "transport":         {"genes": TRANSPORT_GROUP,     "primary_markers": "HVA low + SPECT reduced (SLC18A2); HVA ELEVATED paradox + SPECT absent (SLC6A3)"},
            "degradation":       {"genes": DEGRADATION_GROUP,   "primary_markers": "urine metabolites ALL elevated; X-linked males; aggression"},
            "bh4_synthesis":     {"genes": BH4_SYNTHESIS_GROUP, "primary_markers": "biopterin ELEVATED (SPR); biopterin LOW (GCH1); both: HVA+5-HIAA low; diurnal"},
        },
    }
