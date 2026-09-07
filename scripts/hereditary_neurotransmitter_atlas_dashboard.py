#!/usr/bin/env python3
"""Hereditary-Neurotransmitter-Synthesis-Disorders-Atlas — Complete 8-Gene Atlas
DDC     (aromatic L-amino acid decarboxylase; 480 aa; 7p12.2; AR;
         AADC deficiency — dopamine + serotonin both depleted;
         oculogyric crises PATHOGNOMONIC; HVA LOW + 5-HIAA LOW CSF;
         eladocagene exuparvovec (Upstaza) FDA2023 gene therapy;
         seed SEED_BASE+0) .
GCH1    (GTP cyclohydrolase 1; 250 aa; 14q22.2; AD DRD / AR;
         Dopa-responsive dystonia (Segawa disease) — AD GOF masquerades as CP;
         diurnal variation PATHOGNOMONIC — worse evening better morning;
         L-Dopa ultra-low dose CURATIVE — diagnostic trial FIRST;
         seed SEED_BASE+1) .
TH      (tyrosine hydroxylase; 498 aa; 11p15.5; AR;
         TH deficiency — infantile Parkinsonism-dystonia;
         L-Dopa DRAMATIC response (TH-1) vs complex phenotype (TH-2);
         CSF HVA low + BH4 normal distinguishes from GCH1 AR;
         seed SEED_BASE+2) .
SPR     (sepiapterin reductase; 263 aa; 2p14; AR;
         Sepiapterin reductase deficiency — BH4 NOT elevated in urine (unlike PAH);
         CSF sepiapterin PATHOGNOMONIC; L-Dopa + 5-HTP combination MANDATORY;
         L-Dopa alone → serotonin depletion worsens;
         seed SEED_BASE+3) .
ALDH7A1 (aldehyde dehydrogenase 7A1 / antiquitin; 539 aa; 5q31.2; AR;
         Pyridoxine-dependent epilepsy (PDE) — antiquitin deficiency;
         alpha-aminoadipic semialdehyde (alpha-AASA) urine PATHOGNOMONIC;
         pyridoxine (NOT PLP) first + lysine restriction second;
         seed SEED_BASE+4) .
PNPO    (pyridox(am)ine 5-phosphate oxidase; 261 aa; 17q21.32; AR;
         PNPO deficiency — PLP (pyridoxal-5-phosphate) MANDATORY NOT pyridoxine;
         giving pyridoxine (B6) INSTEAD OF PLP = potentially fatal mistake;
         neonatal burst-suppression EEG + seizures unresponsive to pyridoxine;
         seed SEED_BASE+5) .
SLC6A3  (solute carrier family 6 member 3 / DAT1; 620 aa; 5p15.33; AR;
         DAT deficiency syndrome (DTDS) — dopamine transporter deficiency;
         dopamine accumulates in synapse — L-Dopa and dopamine agonists WORSEN;
         DAT-SPECT absent/markedly reduced binding PATHOGNOMONIC;
         seed SEED_BASE+6) .
GATM    (glycine amidinotransferase / AGAT; 423 aa; 15q21.1; AR;
         AGAT deficiency — creatine biosynthesis step 1 defect;
         guanidinoacetate (GAA) LOW (not HIGH as in GAMT) + creatine LOW;
         intellectual disability + absent speech — creatine 400 mg/kg/day CURATIVE;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1878–1885)
"""

import random

SEED_BASE = 1878

NT_GENES = [
    # -- DDC -- AADC Deficiency -------------------------------------------------------
    {
        "gene": "DDC",
        "protein": (
            "DDC -- 7p12.2 AR -- Aromatic-L-Amino-Acid-Decarboxylase-480aa -- "
            "AADC-Deficiency-Dopamine-Serotonin-Both-Depleted -- "
            "Oculogyric-Crises-PATHOGNOMONIC -- "
            "HVA-LOW-5HIAA-LOW-CSF-DUAL-Depletion -- "
            "Eladocagene-Exuparvovec-Upstaza-FDA2023-Gene-Therapy"
        ),
        "alias": (
            "DDC (aromatic L-amino acid decarboxylase, AADC); OMIM gene 107930; "
            "AADC deficiency OMIM 608643. "
            "7p12.2; 480 aa; ~53 kDa; cytoplasmic; PLP-dependent; autosomal recessive. "
            "FUNCTION: DDC encodes AADC, a PLP-dependent enzyme that catalyses: "
            "(1) L-DOPA → dopamine (in dopaminergic neurons); "
            "(2) 5-hydroxytryptophan (5-HTP) → serotonin (in serotonergic neurons); "
            "(3) DOPA → dopamine in peripheral tissues. "
            "AADC is the FINAL committed step for BOTH dopamine and serotonin synthesis. "
            "In AADC deficiency: L-DOPA accumulates (cannot be converted to dopamine); "
            "5-HTP accumulates (cannot be converted to serotonin); "
            "dopamine, serotonin, epinephrine, norepinephrine all severely depleted. "
            "CSF metabolites: HVA (homovanillic acid, dopamine metabolite) markedly LOW; "
            "5-HIAA (5-hydroxyindoleacetic acid, serotonin metabolite) markedly LOW; "
            "L-DOPA and 5-HTP ELEVATED in CSF (accumulated substrates). "
            "CLINICAL PRESENTATION: "
            "Severe hypotonia from neonatal/early-infantile period; "
            "oculogyric crises (OGC): involuntary sustained upward/lateral eye deviations "
            "with opisthotonus → HALLMARK of AADC deficiency and monoamine disorders; "
            "episodes last minutes to hours; precipitated by illness, excitement, feeding; "
            "developmental delay: profound (cannot sit, limited eye contact); "
            "autonomic dysfunction: hyperhidrosis, temperature instability, ptosis, "
            "miosis (small pupils from norepinephrine depletion), nasal congestion; "
            "movement disorder: dystonia, choreiform movements, hypotonia; "
            "sleep disturbance: excessive or inverted sleep-wake cycle; "
            "feeding difficulties requiring nasogastric/gastrostomy feeding. "
            "DIAGNOSIS: "
            "CSF neurotransmitter profile: HVA low, 5-HIAA low, L-DOPA elevated, "
            "5-HTP elevated — this DUAL depletion (dopamine+serotonin) distinguishes AADC "
            "from conditions depleting only one monoamine; "
            "plasma AADC enzyme activity: markedly reduced (most centres); "
            "urine: 3-O-methyldopa elevated (L-DOPA shunted to COMT pathway); "
            "DDC gene sequencing confirms diagnosis. "
            "TREATMENT: "
            "Pyridoxine (B6): AADC requires PLP cofactor — high-dose B6 (10-30 mg/kg/day) "
            "partially activates residual AADC enzyme; variable response; "
            "MAO-B inhibitors (selegiline/tranylcypromine): inhibit dopamine/serotonin "
            "breakdown — preserves what little monoamine is made; "
            "Dopamine agonists (pramipexole, ropinirole): bypass AADC at dopamine receptors; "
            "AADC GENE THERAPY: "
            "Eladocagene exuparvovec (Upstaza, PTC Therapeutics/BioMarin): "
            "AAV2 vector delivering DDC gene into bilateral putamen via stereotactic injection; "
            "FDA approved 2023 (first CNS gene therapy via direct brain injection for IEM); "
            "EMA approved 2022; "
            "restores local dopamine production in striatum; "
            "dramatic motor improvement post-gene therapy (first 3-6 months); "
            "best outcomes in younger patients (<4 years); "
            "Does NOT correct systemic/peripheral AADC deficiency. "
            "KEY CLINICAL FACTS: "
            "Oculogyric crises are PATHOGNOMONIC — seen in AADC deficiency, "
            "SLC6A3 (DTDS, but paradoxical), glutaric aciduria type 1 (crisis only), "
            "tardive dyskinesia (drug-induced) — AADC is top differential for OGC in infants; "
            "CSF studies MANDATORY — blood tests do not diagnose this condition; "
            "Pyridoxine ALONE rarely sufficient — dopamine agonist combination essential; "
            "Gene therapy is now first-line for eligible patients — refer early."
        ),
        "age_of_onset": "Neonatal/early infantile (birth to 6 months)",
        "inheritance": "AR",
        "locus": "7p12.2",
        "protein_size": "480 aa",
        "key_biomarker": "CSF HVA low + 5-HIAA low + L-DOPA elevated + 5-HTP elevated",
        "pathognomonic": "Oculogyric crises + hypotonia + CSF dual monoamine depletion",
        "treatment": "Pyridoxine B6; MAO-B inhibitor selegiline; dopamine agonists; Upstaza gene therapy FDA2023",
        "critical_flags": [
            "OCULOGYRIC-CRISES-PATHOGNOMONIC — sustained upward eye deviation + opisthotonus; AADC until proven otherwise in infants",
            "CSF-HVA-5HIAA-BOTH-LOW — DUAL depletion (dopamine + serotonin) distinguishes from single-pathway defects",
            "UPSTAZA-FDA2023-GENE-THERAPY — bilateral putamen AAV2; best outcome <4 years; refer to specialist early",
            "BLOOD-TESTS-INSUFFICIENT — CSF neurotransmitter profile mandatory; diagnosis cannot be made from blood alone",
            "AUTONOMIC-DYSFUNCTION — hyperhidrosis, ptosis, miosis, temperature instability; autonomic crisis risk",
            "PYRIDOXINE-B6-TRIAL-MANDATORY — cofactor supplementation; partial response in ~50%; combine with DA agonists",
            "L-DOPA-NOT-FIRST-LINE — substrate accumulates; L-DOPA bypassed by gene therapy or dopamine agonists",
        ],
    },

    # -- GCH1 -- Dopa-Responsive Dystonia (Segawa) ------------------------------------
    {
        "gene": "GCH1",
        "protein": (
            "GCH1 -- 14q22.2 AD-DRD-Segawa / AR -- GTP-Cyclohydrolase-1-250aa -- "
            "Dopa-Responsive-Dystonia-Segawa-Disease-OMIM-128230 -- "
            "Diurnal-Variation-PATHOGNOMONIC-Worse-Evening-Better-Morning -- "
            "L-Dopa-Ultra-Low-Dose-1-2mgkgday-CURATIVE-Diagnostic-Trial-FIRST -- "
            "Masquerades-as-Cerebral-Palsy-Female-Predominance-3:1"
        ),
        "alias": (
            "GCH1 (GTP cyclohydrolase 1); OMIM gene 600225; "
            "Dopa-responsive dystonia (DRD/Segawa disease) OMIM 128230 (AD); "
            "Hyperphenylalaninaemia BH4 type (HPABH4C) OMIM 233910 (AR). "
            "14q22.2; 250 aa; ~28 kDa; cytoplasmic; forms homodecamer; autosomal dominant (heterozygous LOF) or AR. "
            "FUNCTION: GCH1 encodes GTP cyclohydrolase 1, the RATE-LIMITING enzyme in "
            "tetrahydrobiopterin (BH4) synthesis. "
            "BH4 is the essential cofactor for: "
            "(1) tyrosine hydroxylase (TH) — dopamine synthesis rate-limiting step; "
            "(2) phenylalanine hydroxylase (PAH) — phenylalanine metabolism; "
            "(3) tryptophan hydroxylase (TPH) — serotonin synthesis rate-limiting step; "
            "(4) nitric oxide synthase (NOS) — vascular tone. "
            "GCH1 pathway: GTP → BH4 (via PTPS, SPR). "
            "In AD DRD (Segawa): one GCH1 allele lost → BH4 synthesis halved → "
            "TH activity insufficient (especially in nigro-striatal neurons with high metabolic demand); "
            "dopamine preferentially depleted in striatum; serotonin less affected initially. "
            "CLINICAL PRESENTATION (AD DRD - Segawa disease): "
            "onset typically age 6-12 years (range 1-40); "
            "lower limb dystonia first — gait abnormality, equinovarus posturing; "
            "DIURNAL VARIATION: symptoms worst in evening after sustained activity, "
            "markedly improved after sleep/morning rest — THIS IS PATHOGNOMONIC; "
            "female-to-male ratio ~3:1 (more penetrant in females); "
            "Parkinsonism features in adults (bradykinesia, rigidity) without diurnal variation; "
            "commonly MISDIAGNOSED as cerebral palsy (especially ataxic or spastic CP); "
            "cognition NORMAL (critical distinction from dopa-responsive dystonias with ID). "
            "CLINICAL PRESENTATION (AR GCH1 - severe): "
            "severe BH4 deficiency → hyperphenylalaninaemia + progressive neurological disease; "
            "CSF HVA and 5-HIAA both markedly low (similar to AADC but BH4 deficiency pattern). "
            "DIAGNOSIS: "
            "L-DOPA TRIAL IS DIAGNOSTIC — ultra-low dose (0.5-1 mg/kg/day) → dramatic sustained response; "
            "CSF: HVA low, 5-HIAA low (mild in AD), BH4/biopterin ratio may guide BH4 type; "
            "urine: neopterin normal/low, biopterin normal/low (BH4 production reduced); "
            "Phenylalanine loading test: delayed phenylalanine clearance (BH4-dependent); "
            "GCH1 gene sequencing: many unique family mutations; "
            "BRAIN MRI: NORMAL (critical differential from structural causes of dystonia). "
            "TREATMENT: "
            "L-DOPA + carbidopa (peripheral decarboxylase inhibitor): "
            "ultra-low dose: 1-2 mg/kg/day L-DOPA equivalent; "
            "COMPLETE, SUSTAINED response in virtually ALL AD DRD patients; "
            "response maintained life-long without dose escalation (no wearing off); "
            "if response is incomplete — consider AR GCH1 or other BH4 deficiency; "
            "BH4 supplementation (sapropterin) for AR severe forms. "
            "KEY CLINICAL FACTS: "
            "DIURNAL VARIATION — if a dystonia has diurnal variation, try L-DOPA before anything else; "
            "L-DOPA TRIAL MUST BE GIVEN — misdiagnosed CP patients given botulinum, surgery, "
            "orthopaedic interventions when simple L-DOPA would cure; "
            "L-DOPA RESPONSE IS DRAMATIC AND COMPLETE — partial response → question diagnosis; "
            "NORMAL BRAIN MRI — structural abnormalities should prompt alternative diagnoses; "
            "FEMALE PREDOMINANCE in AD DRD — penetrance ~87% in females, ~38% in males (sex-modifying factors)."
        ),
        "age_of_onset": "Childhood (6-12 years typical; range 1 year to adulthood)",
        "inheritance": "AD (DRD/Segawa); AR (severe BH4 deficiency)",
        "locus": "14q22.2",
        "protein_size": "250 aa",
        "key_biomarker": "CSF HVA low; urine neopterin/biopterin low; phenylalanine loading test",
        "pathognomonic": "Diurnal variation of dystonia (worse evening, better morning after sleep) + dramatic L-Dopa response",
        "treatment": "L-Dopa + carbidopa ultra-low dose (1-2 mg/kg/day); dramatic sustained lifelong response",
        "critical_flags": [
            "DIURNAL-VARIATION-PATHOGNOMONIC — symptoms worst evening, dramatically better after sleep; try L-Dopa before any procedure",
            "L-DOPA-TRIAL-MANDATORY — do NOT diagnose CP or refer for botulinum/surgery without L-Dopa trial first",
            "COMPLETE-SUSTAINED-RESPONSE — virtually 100% AD DRD patients respond; partial response → question diagnosis",
            "NORMAL-BRAIN-MRI — structural abnormalities exclude GCH1 DRD; refer for MRI first",
            "FEMALE-PREDOMINANCE-3:1 — penetrance ~87% females vs ~38% males; paternal inheritance often clinically silent",
            "AR-GCH1-SEVERE — biallelic mutations → hyperphenylalaninaemia + profound neurological disease; treat as BH4 deficiency",
            "MISDIAGNOSIS-CEREBRAL-PALSY — most common misdiagnosis; DRD = treatable cause of childhood dystonia",
        ],
    },

    # -- TH -- Tyrosine Hydroxylase Deficiency ----------------------------------------
    {
        "gene": "TH",
        "protein": (
            "TH -- 11p15.5 AR -- Tyrosine-Hydroxylase-498aa -- "
            "TH-Deficiency-Infantile-Parkinsonism-Dystonia -- "
            "TH1-DRD-Like-L-Dopa-DRAMATIC-Response -- "
            "TH2-Complex-Encephalopathy-Autonomic-Dysfunction -- "
            "CSF-HVA-LOW-BH4-NORMAL-Distinguishes-from-GCH1-AR"
        ),
        "alias": (
            "TH (tyrosine hydroxylase); OMIM gene 191290; "
            "TH deficiency OMIM 605407 (TH-1 and TH-2). "
            "11p15.5; 498 aa; ~56 kDa; cytoplasmic tetramer; BH4-dependent; iron-containing; autosomal recessive. "
            "FUNCTION: TH encodes tyrosine hydroxylase, the RATE-LIMITING enzyme for catecholamine synthesis: "
            "Tyrosine → L-DOPA (TH, requires BH4 + O2 + Fe2+). "
            "L-DOPA → dopamine (AADC/DDC). "
            "Dopamine → norepinephrine → epinephrine (sequential). "
            "TH acts at the committed step for ALL catecholamines: "
            "dopamine, norepinephrine (noradrenaline), epinephrine (adrenaline). "
            "In TH deficiency: catecholamines severely reduced; BH4 consumption reduced; "
            "CSF HVA (dopamine metabolite) markedly low; "
            "BH4 levels NORMAL (distinguishes from GCH1 AR); "
            "5-HIAA (serotonin) may be mildly reduced (BH4 indirectly needed for TPH). "
            "TWO CLINICAL PHENOTYPES: "
            "TH-1 (DRD-like/mild): "
            "onset in childhood; lower limb dystonia; diurnal variation similar to GCH1 DRD; "
            "L-DOPA response DRAMATIC and complete; often clinically indistinguishable from GCH1 DRD; "
            "residual TH activity maintained; "
            "TH-2 (severe/complex): "
            "onset in infancy; profound hypotonia; infantile Parkinsonism-dystonia; "
            "encephalopathy; autonomic dysfunction (ptosis, miosis, sweating, temperature); "
            "truncal hypotonia + limb hypertonia paradox; "
            "L-DOPA response: present but often incomplete; dose titration needed; "
            "developmental delay common but improved with treatment. "
            "DIAGNOSIS: "
            "CSF: HVA markedly low (dopamine depleted); "
            "5-HIAA: mildly low or normal (contrast AADC: both very low); "
            "BH4 (biopterin): NORMAL — this is the KEY distinguishing finding from GCH1 AR; "
            "PTPS activity normal; "
            "L-DOPA response trial: TH-1 responds dramatically; TH-2 partial; "
            "TH gene sequencing: confirms diagnosis. "
            "TREATMENT: "
            "L-DOPA + carbidopa: "
            "TH-1: ultra-low dose as in GCH1 DRD; dramatic, sustained; "
            "TH-2: higher doses often needed; titrate slowly; dyskinesias possible; "
            "Dopamine agonists: adjunct in TH-2; "
            "No gene therapy currently approved. "
            "KEY CLINICAL FACTS: "
            "BH4 NORMAL differentiates TH from GCH1 AR (where BH4 is low); "
            "TH-1 vs GCH1 AD-DRD: both respond to L-DOPA; distinction by gene testing; "
            "TH-2 phenotype overlaps with AADC deficiency — CSF metabolites distinguish: "
            "TH-2: HVA low, 5-HIAA near-normal, L-DOPA low; "
            "AADC: HVA low, 5-HIAA low, L-DOPA HIGH (accumulated); "
            "Truncal hypotonia + limb hypertonia in TH-2 is a distinctive clinical clue; "
            "ALL patients with infantile dystonia-Parkinsonism should have CSF neurotransmitter profile."
        ),
        "age_of_onset": "TH-1: childhood (DRD-like); TH-2: infancy (neonatal to 6 months)",
        "inheritance": "AR",
        "locus": "11p15.5",
        "protein_size": "498 aa",
        "key_biomarker": "CSF HVA markedly low; BH4 NORMAL (not low — distinguishes from GCH1 AR)",
        "pathognomonic": "Infantile Parkinsonism-dystonia + CSF HVA low + BH4 NORMAL + L-Dopa dramatic response",
        "treatment": "L-Dopa + carbidopa; TH-1: ultra-low dose complete response; TH-2: higher titrated dose",
        "critical_flags": [
            "BH4-NORMAL-KEY-DISTINGUISHER — BH4 normal in TH deficiency vs low in GCH1 AR; critical for differential",
            "TH1-DRAMATIC-L-DOPA-RESPONSE — indistinguishable from GCH1 DRD clinically; gene testing required",
            "TH2-SEVERE-INFANTILE — truncal hypotonia + limb hypertonia paradox; encephalopathy + autonomic dysfunction",
            "CSF-MANDATORY-ALL-INFANTILE-PARKINSONISM — HVA + 5-HIAA + L-DOPA + BH4 profile distinguishes AADC/GCH1/TH/SPR",
            "L-DOPA-ELEVATED-IN-AADC-LOW-IN-TH — substrate accumulates in AADC (DDC cannot use); depleted in TH (cannot make)",
            "DYSKINESIAS-TH2 — higher L-DOPA doses in TH-2 may cause dyskinesias; slow titration essential",
            "NO-GENE-THERAPY-APPROVED — unlike AADC; treatment remains pharmacological L-Dopa/agonists",
        ],
    },

    # -- SPR -- Sepiapterin Reductase Deficiency --------------------------------------
    {
        "gene": "SPR",
        "protein": (
            "SPR -- 2p14 AR -- Sepiapterin-Reductase-263aa -- "
            "Sepiapterin-Reductase-Deficiency -- "
            "BH4-NOT-Elevated-in-Urine-Unlike-PAH-DHPR -- "
            "CSF-Sepiapterin-PATHOGNOMONIC -- "
            "L-Dopa-PLUS-5-HTP-Combination-MANDATORY-L-Dopa-Alone-Worsens-Serotonin"
        ),
        "alias": (
            "SPR (sepiapterin reductase); OMIM gene 182125; "
            "Sepiapterin reductase deficiency OMIM 612716. "
            "2p14; 263 aa; ~28 kDa; cytoplasmic homodimer; NADPH-dependent; autosomal recessive. "
            "FUNCTION: SPR encodes sepiapterin reductase, which catalyses the FINAL STEP "
            "in the de novo BH4 synthesis pathway: "
            "6-pyruvoyl-tetrahydropterin (PTPS product) → BH4 (via two sequential SPR reactions, "
            "through sepiapterin and 7,8-dihydrobiopterin intermediates). "
            "WITHOUT SPR: BH4 severely deficient in CNS; "
            "HOWEVER: alternative salvage pathway exists — aromatic tissues can regenerate some BH4; "
            "sepiapterin accumulates (CSF sepiapterin is DIAGNOSTIC); "
            "BH4 deficiency → TH impaired → HVA low; TPH impaired → 5-HIAA low; "
            "CRITICAL: urine biopterin NOT elevated (BH4 not overproduced peripherally) — "
            "THIS DISTINGUISHES SPR FROM PAH AND DHPR DEFICIENCIES where urine biopterin IS elevated. "
            "CLINICAL PRESENTATION: "
            "Onset in infancy; "
            "motor delay, hypotonia; "
            "dystonia with diurnal variation (similar to DRD); "
            "oculomotor abnormalities; "
            "intellectual disability (mild to severe); "
            "sleep disturbance (serotonin depletion); "
            "autonomic features; "
            "does NOT cause hyperphenylalaninaemia (PAH retained — peripheral salvage pathway sufficient); "
            "so routine NBS for PKU/HPA is NORMAL — misses SPR deficiency. "
            "DIAGNOSIS: "
            "NBS: NORMAL phenylalanine (does not cause HPA — unlike GCH1 AR); "
            "CSF neurotransmitter profile: "
            "HVA: low; 5-HIAA: low (both depleted — but less severely than AADC); "
            "CSF sepiapterin: ELEVATED — PATHOGNOMONIC (direct evidence of block before SPR); "
            "urine biopterin: normal or low (NOT elevated — key differential from DHPR/PAH); "
            "Phenylalanine loading test: NORMAL (SPR deficiency does NOT impair hepatic PAH); "
            "SPR gene sequencing confirms. "
            "TREATMENT — COMBINATION IS MANDATORY: "
            "L-DOPA + carbidopa: restores dopamine; "
            "5-HTP (5-hydroxytryptophan): MANDATORY alongside L-DOPA — "
            "L-DOPA alone → dopamine restored but serotonin remains depleted → "
            "AADC preferentially diverts to dopamine → serotonin synthesis worsens; "
            "5-HTP bypasses TPH step → serotonin restored; "
            "BH4 (sapropterin): supplemental BH4 to bypass SPR step; improves enzyme cofactor availability; "
            "Folinic acid: some centres add for CSF folate support. "
            "KEY CLINICAL FACTS: "
            "L-DOPA ALONE WORSENS SEROTONIN DEPLETION — always combine with 5-HTP; "
            "CSF SEPIAPTERIN IS PATHOGNOMONIC — cannot be detected in blood or urine (only CSF); "
            "NORMAL NBS (no HPA) means this condition is NOT caught by standard newborn screening; "
            "urine biopterin NORMAL/LOW — this is opposite of DHPR where biopterin very high; "
            "Diurnal variation present (like GCH1 DRD) — misleads clinicians into GCH1 diagnosis."
        ),
        "age_of_onset": "Infancy (3-6 months) to early childhood",
        "inheritance": "AR",
        "locus": "2p14",
        "protein_size": "263 aa",
        "key_biomarker": "CSF sepiapterin elevated (PATHOGNOMONIC); HVA low; 5-HIAA low; urine biopterin NOT elevated",
        "pathognomonic": "CSF sepiapterin elevated + HVA low + 5-HIAA low + normal urine biopterin + normal NBS phenylalanine",
        "treatment": "L-Dopa + carbidopa PLUS 5-HTP (MANDATORY combination) + BH4 sapropterin",
        "critical_flags": [
            "L-DOPA-ALONE-WORSENS-SEROTONIN — L-Dopa competes with 5-HTP for AADC; always combine L-Dopa + 5-HTP",
            "CSF-SEPIAPTERIN-PATHOGNOMONIC — only found in CSF; cannot be diagnosed without lumbar puncture",
            "NORMAL-NBS-PHENYLALANINE — SPR does NOT cause HPA; routine NBS misses this diagnosis entirely",
            "URINE-BIOPTERIN-NOT-ELEVATED — critical differential from DHPR (very high biopterin) and PAH (high biopterin)",
            "DIURNAL-VARIATION — misleads to GCH1 DRD diagnosis; CSF metabolites distinguish",
            "BH4-SUPPLEMENTATION-ADJUNCT — sapropterin improves residual TH/TPH activity; use alongside L-Dopa + 5-HTP",
            "COMBINATION-MANDATORY — monotherapy with any single agent is insufficient; three-drug approach",
        ],
    },

    # -- ALDH7A1 -- Pyridoxine-Dependent Epilepsy ------------------------------------
    {
        "gene": "ALDH7A1",
        "protein": (
            "ALDH7A1 -- 5q31.2 AR -- Aldehyde-Dehydrogenase-7A1-Antiquitin-539aa -- "
            "Pyridoxine-Dependent-Epilepsy-PDE -- "
            "Alpha-AASA-Urine-PATHOGNOMONIC -- "
            "Pyridoxine-NOT-PLP-First-Line -- "
            "Lysine-Restriction-Pipecolic-Acid-Adjunct -- "
            "L-Pipecolic-Acid-CSF-Plasma-Elevated"
        ),
        "alias": (
            "ALDH7A1 (aldehyde dehydrogenase 7A1, antiquitin); OMIM gene 107323; "
            "Pyridoxine-dependent epilepsy (PDE) OMIM 266100. "
            "5q31.2; 539 aa; ~58 kDa; mitochondrial; NAD+-dependent; forms trimer; autosomal recessive. "
            "FUNCTION: ALDH7A1 (antiquitin) encodes an aldehyde dehydrogenase involved in "
            "lysine catabolism. "
            "In lysine degradation via the saccharopine pathway: "
            "...pipecoline acid → 1-piperidine-6-carboxylate (P6C) → antiquitin → "
            "alpha-aminoadipic semialdehyde (alpha-AASA) → 2-aminoadipic acid. "
            "In ALDH7A1 deficiency: P6C and alpha-AASA accumulate; "
            "P6C INACTIVATES PYRIDOXAL-5-PHOSPHATE (PLP) — forms a Knoevenagel condensation product; "
            "PLP depletion → multiple PLP-dependent enzymes impaired, including AADC (DDC); "
            "PLP-dependent enzymes: GABA synthesis (GAD), glycine cleavage, serine synthesis, etc.; "
            "net result: PLP-dependent seizure threshold lowered → refractory neonatal seizures. "
            "CLINICAL PRESENTATION: "
            "Neonatal onset MOST COMMON (80%): "
            "seizures begin in first hours to days of life; "
            "seizures REFRACTORY to standard AEDs (phenobarbitone, phenytoin, benzodiazepines); "
            "EEG: may show burst-suppression or multifocal seizures; "
            "atypical forms: late-onset (to 3 years), initial response to AEDs then breakthrough; "
            "seizure types: myoclonic, tonic, clonic, or generalised. "
            "DIAGNOSIS: "
            "alpha-AASA: PATHOGNOMONIC when elevated in urine (most reliable test); "
            "L-pipecolic acid: elevated in plasma AND CSF (less specific — elevated in peroxisomal disorders too); "
            "CSF/plasma P6C: elevated (functional assay in research centres); "
            "MRI: often shows basal ganglia signal changes, cerebral atrophy, delayed myelination; "
            "Pyridoxine trial: 100 mg IV → seizure cessation within minutes/hours — DIAGNOSTIC if response seen; "
            "ALDH7A1 gene sequencing: confirms; p.Glu399Gln common mutation. "
            "TREATMENT: "
            "PYRIDOXINE (B6, pyridoxine hydrochloride): NOT PLP — "
            "pyridoxine is converted to PLP peripherally; effective because it replenishes "
            "the PLP pool being consumed by P6C condensation; "
            "dose: 15-30 mg/kg/day; life-long; "
            "L-LYSINE RESTRICTION: reduces lysine flux through antiquitin pathway → "
            "less P6C/alpha-AASA accumulation; adjunct therapy; "
            "LNP-LDAER (L-arginine supplementation): competes with lysine for transport "
            "(experimental in some centres); "
            "Folinic acid: some centres add empirically. "
            "KEY CLINICAL FACTS: "
            "PYRIDOXINE, NOT PLP, IS FIRST LINE — pharmacological doses of pyridoxine (not PLP) work; "
            "PLP works in PNPO deficiency (different condition); giving PLP in PDE vs pyridoxine in PNPO "
            "is the CRITICAL DIFFERENTIAL; "
            "ALPHA-AASA IS PATHOGNOMONIC — but requires fresh urine stored correctly; "
            "RESPONSE TO PYRIDOXINE IV IS DIAGNOSTIC — EEG control within 10-60 minutes; "
            "LYSINE RESTRICTION IMPROVES OUTCOMES — reduces accumulation independent of pyridoxine; "
            "LATE-ONSET forms exist — PDE should be considered up to 3 years of age with refractory epilepsy."
        ),
        "age_of_onset": "Neonatal (80%) or late-onset to 3 years",
        "inheritance": "AR",
        "locus": "5q31.2",
        "protein_size": "539 aa",
        "key_biomarker": "Alpha-AASA (alpha-aminoadipic semialdehyde) in urine; L-pipecolic acid plasma/CSF elevated",
        "pathognomonic": "Alpha-AASA urine elevated + seizure cessation with IV pyridoxine within 60 minutes",
        "treatment": "Pyridoxine (B6) 15-30 mg/kg/day lifelong (NOT PLP); lysine restriction adjunct",
        "critical_flags": [
            "PYRIDOXINE-NOT-PLP — ALDH7A1/PDE is treated with pyridoxine (B6); PNPO deficiency needs PLP; DO NOT CONFUSE",
            "ALPHA-AASA-PATHOGNOMONIC — fresh urine test; most reliable biomarker for PDE",
            "IV-PYRIDOXINE-100mg-DIAGNOSTIC — seizure cessation within 10-60 min; life-saving diagnostic test",
            "REFRACTORY-NEONATAL-SEIZURES — standard AEDs fail; try pyridoxine before escalation",
            "LYSINE-RESTRICTION-ADJUNCT — reduces substrate flux; improves long-term neurodevelopmental outcome",
            "LATE-ONSET-UP-TO-3-YEARS — do not exclude PDE if not neonatal; up to 3 years first seizure possible",
            "BRAIN-MRI-ABNORMAL — basal ganglia changes, delayed myelination; structural brain abnormalities common",
        ],
    },

    # -- PNPO -- Pyridox(am)ine 5-Phosphate Oxidase Deficiency -----------------------
    {
        "gene": "PNPO",
        "protein": (
            "PNPO -- 17q21.32 AR -- Pyridoxamine-5-Phosphate-Oxidase-261aa -- "
            "PNPO-Deficiency-Neonatal-Epileptic-Encephalopathy -- "
            "PLP-Pyridoxal-5-Phosphate-MANDATORY-NOT-Pyridoxine-B6 -- "
            "Pyridoxine-Instead-of-PLP-FATAL-MISTAKE -- "
            "Burst-Suppression-EEG-Neonatal-Seizures-Unresponsive-to-Pyridoxine"
        ),
        "alias": (
            "PNPO (pyridox(am)ine 5'-phosphate oxidase); OMIM gene 610090; "
            "PNPO deficiency OMIM 610090. "
            "17q21.32; 261 aa; ~30 kDa; mitochondrial homodimer; FMN-dependent; autosomal recessive. "
            "FUNCTION: PNPO encodes pyridox(am)ine 5'-phosphate oxidase, which converts: "
            "(1) Pyridoxamine-5-phosphate (PMP) → Pyridoxal-5-phosphate (PLP); "
            "(2) Pyridoxine-5-phosphate (PNP) → PLP. "
            "PLP is the ACTIVE FORM of vitamin B6 — the essential cofactor for >140 enzyme reactions including: "
            "GABA synthesis (GAD), serine/glycine metabolism, amino acid transamination, "
            "PLP-dependent seizure threshold maintenance, AADC activity. "
            "In PNPO deficiency: conversion of PN and PM to PLP is blocked; "
            "PLP deficient in brain → GABA synthesis impaired → seizures; "
            "HOWEVER: administering pyridoxine (which is PN → PNP → PNPO → PLP) does NOT help — "
            "the conversion step (PNPO) is missing; "
            "giving PLP DIRECTLY bypasses PNPO → PLP enters cells and restores enzyme function. "
            "CLINICAL PRESENTATION: "
            "Neonatal onset: seizures often begin within hours of birth; "
            "EEG: burst-suppression pattern in neonates; "
            "Prematurity association (many reported cases born premature); "
            "seizures refractory to ALL standard AEDs; "
            "CRUCIAL: seizures also UNRESPONSIVE TO PYRIDOXINE (B6) — "
            "this distinguishes PNPO deficiency from PDE (ALDH7A1) where pyridoxine works; "
            "metabolic acidosis may be present; "
            "threonine, glycine, taurine elevated in CSF (PLP-dependent enzymes impaired); "
            "without PLP treatment: severe encephalopathy, death or profound disability. "
            "DIAGNOSIS: "
            "Clinical: neonatal seizures unresponsive to pyridoxine (B6) but RESPONSIVE TO PLP; "
            "CSF: elevated glycine, threonine, taurine (PLP-dependent enzymes impaired); "
            "plasma: PLP LOW; pyridoxamine HIGH (upstream metabolite accumulates); "
            "PNPO gene sequencing: p.Arg229Trp most common in Middle Eastern populations; "
            "NOTE: diagnosis sometimes missed because PDE (pyridoxine-responsive) is tried first. "
            "TREATMENT: "
            "PLP (pyridoxal-5-phosphate): "
            "oral PLP 30-60 mg/kg/day — divided doses; "
            "IV PLP available in some centres for acute control; "
            "PLP is the ONLY effective form — pyridoxine (B6) is INEFFECTIVE; "
            "pyridoxamine also ineffective (same problem — cannot be converted to PLP); "
            "Life-long treatment; doses may need adjustment with age. "
            "KEY CLINICAL FACTS: "
            "PLP NOT PYRIDOXINE — THE MOST CRITICAL POINT: "
            "In PNPO: PLP is the active vitamin B6 form that is needed; pyridoxine cannot be converted to PLP; "
            "In PDE (ALDH7A1): pyridoxine WORKS because the conversion is intact; "
            "Giving pyridoxine to PNPO patient = zero benefit, continuing seizures, encephalopathy; "
            "UNRESPONSIVE TO PYRIDOXINE IS THE CLINICAL CLUE — if seizures continue after pyridoxine trial, "
            "try PLP IMMEDIATELY; "
            "BURST-SUPPRESSION EEG + NEONATAL SEIZURES + PYRIDOXINE FAILURE = PNPO DEFICIENCY UNTIL PROVEN OTHERWISE; "
            "p.Arg229Trp: most common in Middle Eastern/North African ancestry — targeted sequencing if available."
        ),
        "age_of_onset": "Neonatal (hours to days after birth)",
        "inheritance": "AR",
        "locus": "17q21.32",
        "protein_size": "261 aa",
        "key_biomarker": "Plasma PLP low; pyridoxamine elevated; CSF glycine/threonine/taurine elevated",
        "pathognomonic": "Neonatal seizures + burst-suppression EEG + unresponsive to pyridoxine + responds to PLP",
        "treatment": "PLP (pyridoxal-5-phosphate) 30-60 mg/kg/day ONLY; pyridoxine is INEFFECTIVE",
        "critical_flags": [
            "PLP-MANDATORY-NOT-PYRIDOXINE — PNPO cannot convert pyridoxine to PLP; giving B6 is futile and delays PLP treatment",
            "PYRIDOXINE-FAILURE-IS-THE-CLUE — if pyridoxine trial fails in neonatal seizures, try PLP IMMEDIATELY",
            "BURST-SUPPRESSION-EEG-NEONATAL — pattern seen; all standard AEDs fail; diagnostic of severe PLP deficiency",
            "PNPO-vs-PDE-ALDH7A1-CRITICAL-DISTINCTION — PDE responds to pyridoxine; PNPO does NOT; choose treatment correctly",
            "p.Arg229Trp-MIDDLE-EASTERN — most common variant in MENA populations; targeted sequencing available",
            "CSF-GLYCINE-THREONINE-ELEVATED — PLP-dependent enzymes impaired; CSF amino acids guide diagnosis",
            "LIFELONG-PLP — cannot stop; seizure recurrence common on dose reduction or interruption",
        ],
    },

    # -- SLC6A3 -- DAT Deficiency Syndrome -------------------------------------------
    {
        "gene": "SLC6A3",
        "protein": (
            "SLC6A3 -- 5p15.33 AR -- Dopamine-Transporter-DAT1-620aa -- "
            "DAT-Deficiency-Syndrome-DTDS -- "
            "Dopamine-Transporter-Absent-DAT-SPECT-PATHOGNOMONIC -- "
            "L-Dopa-Dopamine-Agonists-WORSEN -- "
            "Infantile-Parkinsonism-Hyperkinesis-Paradox"
        ),
        "alias": (
            "SLC6A3 (solute carrier family 6 member 3, dopamine transporter, DAT1); OMIM gene 126455; "
            "DAT deficiency syndrome (DTDS) OMIM 613135. "
            "5p15.33; 620 aa; ~69 kDa; plasma membrane transporter; 12 transmembrane domains; Na+/Cl- dependent; autosomal recessive. "
            "FUNCTION: SLC6A3 encodes the dopamine transporter (DAT, DAT1), which reuptakes "
            "dopamine from the synapse back into the presynaptic dopaminergic neuron. "
            "DAT is essential for: "
            "(1) terminating dopamine signalling after release; "
            "(2) recycling dopamine for re-use; "
            "(3) maintaining appropriate synaptic dopamine concentration. "
            "In DTDS: DAT absent or non-functional → "
            "dopamine released into synapse CANNOT be cleared → "
            "synaptic dopamine accumulates → dopamine receptors become downregulated/desensitised; "
            "dopamine synthesis NORMAL (TH, DDC, GCH1 all intact); "
            "dopamine does NOT circulate in blood (remains trapped in synapse); "
            "intraneuronal dopamine depleted (cannot be recycled into neuron); "
            "dopamine degradation increases (MAO, COMT working maximally on synaptic dopamine); "
            "HVA (dopamine metabolite) may be ELEVATED in CSF (from synaptic overflow degradation). "
            "CLINICAL PRESENTATION: "
            "Infantile Parkinsonism-hyperkinesis PARADOX: "
            "Parkinsonism features: bradykinesia, rigidity, tremor, masked facies; "
            "SIMULTANEOUSLY: hyperkinesia, dystonia, choreiform movements; "
            "this combination of hypokinesia + hyperkinesia is highly characteristic; "
            "oculogyric crises: can occur (dopamine receptor signalling abnormal); "
            "hypotonia → hypertonia transition; "
            "onset: typically first months to first year of life; "
            "severe developmental delay and regression. "
            "DIAGNOSIS: "
            "DAT-SPECT (FP-CIT SPECT/DaTSCAN): "
            "absent or markedly reduced DAT binding in striatum — PATHOGNOMONIC; "
            "(normal or increased binding would exclude DTDS); "
            "CSF: HVA may be HIGH (excess synaptic dopamine metabolised by MAO/COMT); "
            "(contrast: TH, GCH1, AADC deficiencies where HVA is LOW); "
            "plasma: dopamine normal or near-normal; "
            "SLC6A3 gene sequencing confirms. "
            "TREATMENT — AVOID DOPAMINERGIC AGENTS: "
            "L-DOPA IS CONTRAINDICATED: "
            "more dopamine synthesised → more accumulates in synapse → worsens excitotoxicity; "
            "dopamine agonists similarly WORSEN symptoms; "
            "MAO inhibitors similarly CONTRAINDICATED (block the only remaining clearance pathway); "
            "There is NO established effective treatment; "
            "Supportive care, dystonia management (baclofen, benzodiazepines for spasticity/dystonia); "
            "Pramipexole: paradoxically reported to help in a FEW cases (receptor-level effects); "
            "research: DAT gene therapy under investigation. "
            "KEY CLINICAL FACTS: "
            "L-DOPA WORSENS DTDS — this is the MOST CRITICAL clinical fact; "
            "clinicians may try L-DOPA thinking this is another Parkinsonism — it will worsen the child; "
            "DAT-SPECT IS PATHOGNOMONIC — absent striatal DAT in a child; "
            "HVA HIGH (not low) in DTDS — the opposite of all other dopamine synthesis disorders; "
            "INFANTILE PARKINSONISM + HYPERKINESIS COMBINATION is the signature phenotype — "
            "simultaneous hypo and hyperkinesia in an infant should trigger DAT-SPECT."
        ),
        "age_of_onset": "Infantile (first months to first year)",
        "inheritance": "AR",
        "locus": "5p15.33",
        "protein_size": "620 aa",
        "key_biomarker": "DAT-SPECT absent striatal binding (PATHOGNOMONIC); CSF HVA may be HIGH",
        "pathognomonic": "Infantile Parkinsonism-hyperkinesis + absent DAT-SPECT binding + HVA NOT low (may be high)",
        "treatment": "Supportive only; L-Dopa and dopamine agonists CONTRAINDICATED; no established pharmacotherapy",
        "critical_flags": [
            "L-DOPA-ABSOLUTELY-CONTRAINDICATED — worsens symptoms; more dopamine cannot be cleared from synapse",
            "DOPAMINE-AGONISTS-CONTRAINDICATED — same mechanism; worsening guaranteed",
            "MAO-INHIBITORS-CONTRAINDICATED — removes only remaining clearance pathway; catastrophic worsening",
            "DAT-SPECT-PATHOGNOMONIC — absent striatal DAT binding; must be performed in all infantile movement disorders",
            "HVA-HIGH-NOT-LOW — opposite to all other dopamine synthesis disorders; critical for differential",
            "PARKINSONISM-HYPERKINESIS-PARADOX — simultaneous hypo and hyperkinesia; pathognomonic combination",
            "NO-APPROVED-TREATMENT — research ongoing; gene therapy investigational; refer to specialist centre",
        ],
    },

    # -- GATM (AGAT) -- AGAT Deficiency -----------------------------------------------
    {
        "gene": "GATM",
        "protein": (
            "GATM -- 15q21.1 AR -- Glycine-Amidinotransferase-AGAT-423aa -- "
            "AGAT-Deficiency-Creatine-Biosynthesis-Step1 -- "
            "Guanidinoacetate-GAA-LOW-Not-HIGH-Unlike-GAMT -- "
            "Intellectual-Disability-Absent-Speech-Autism -- "
            "Creatine-400mgkgday-CURATIVE-Dramatic-Response"
        ),
        "alias": (
            "GATM (glycine amidinotransferase, AGAT); OMIM gene 602360; "
            "AGAT deficiency (creatine deficiency type 1) OMIM 612736. "
            "15q21.1; 423 aa; ~48 kDa; mitochondrial matrix; homodimer; autosomal recessive. "
            "FUNCTION: GATM encodes AGAT (arginine:glycine amidinotransferase), which catalyses "
            "the FIRST STEP in creatine biosynthesis: "
            "Arginine + Glycine → Guanidinoacetate (GAA) + Ornithine. "
            "(AGAT primarily in kidney and pancreas → GAA exported to liver.) "
            "SECOND STEP: GAA → Creatine (by GAMT, guanidinoacetate N-methyltransferase, in liver). "
            "CREATINE → phosphocreatine (CK reaction) → ATP buffer in brain, muscle, heart. "
            "In AGAT deficiency: "
            "AGAT step is BLOCKED → GAA is NOT PRODUCED; "
            "therefore GAA = LOW/ABSENT (distinguish from GAMT where GAA is very HIGH); "
            "creatine cannot be synthesised (no substrate GAA available); "
            "creatine = LOW/ABSENT; "
            "phosphocreatine in brain = ABSENT → energy buffer missing in neurons; "
            "brain creatine peak on MRS (magnetic resonance spectroscopy) ABSENT. "
            "CLINICAL PRESENTATION: "
            "Intellectual disability: moderate to severe; "
            "language delay: severely delayed or absent speech; "
            "autistic features: repetitive behaviours, social withdrawal; "
            "hypotonia in early infancy; "
            "hyperkinesia: non-specific movement disorder; "
            "NO specific dysmorphic features; "
            "seizures: occur in some patients; "
            "PRESENTATION IS SUBTLE — may be misdiagnosed as autism, non-specific ID, CP. "
            "DIAGNOSIS: "
            "Urine: guanidinoacetate (GAA) ABSENT or very LOW "
            "(KEY DISTINCTION from GAMT where GAA is VERY HIGH due to GAMT block downstream); "
            "urine creatine: low; "
            "plasma GAA: low; "
            "plasma creatine: low; "
            "Brain MRS: absent creatine peak (Cr peak at 3.03 ppm absent on 1H-MRS) — DIAGNOSTIC; "
            "GATM gene sequencing confirms. "
            "TREATMENT: "
            "Creatine monohydrate supplementation: "
            "400 mg/kg/day (divided doses); "
            "oral creatine bypasses the AGAT + GAMT steps — creatine is transported across the BBB; "
            "DRAMATIC RESPONSE: motor, cognitive, seizure improvement; "
            "CURATIVE if treated early (before significant neuronal damage); "
            "treatment later in childhood still beneficial but outcome better with early start; "
            "Protein restriction NOT required (unlike GAMT where arginine restriction helps); "
            "ornithine supplementation: some centres use (mild adjunct). "
            "KEY CLINICAL FACTS: "
            "GAA LOW (not high) — CRITICAL DISTINCTION from GAMT deficiency where GAA is very HIGH; "
            "CREATINE IS CURATIVE — simple, inexpensive, oral supplementation reverses neurological disease; "
            "NBS NEEDED — this condition is treatable but not routinely screened; urine GAA + creatine should be checked in all unexplained ID; "
            "AUTISM + ABSENT SPEECH + ID → check urine GAA and creatine before extensive genetic workup; "
            "Brain MRS absent Cr peak: rapidly confirms diagnosis non-invasively; "
            "NO GUANIDINOACETATE TOXICITY — unlike GAMT where GAA accumulates and is neurotoxic; in AGAT, GAA is low so no toxicity."
        ),
        "age_of_onset": "Infancy to early childhood (presentation often 2-4 years with speech delay)",
        "inheritance": "AR",
        "locus": "15q21.1",
        "protein_size": "423 aa",
        "key_biomarker": "Urine/plasma guanidinoacetate (GAA) LOW; creatine LOW; brain MRS absent creatine peak",
        "pathognomonic": "GAA LOW + creatine LOW + absent brain MRS creatine peak + intellectual disability + absent speech",
        "treatment": "Creatine monohydrate 400 mg/kg/day oral; CURATIVE if treated early; protein restriction NOT needed",
        "critical_flags": [
            "GAA-LOW-NOT-HIGH — AGAT deficiency: no GAA made (LOW); GAMT deficiency: GAA cannot be converted (VERY HIGH); CRITICAL distinction",
            "CREATINE-CURATIVE-400mgkgday — oral creatine crosses BBB; dramatic reversal of neurological symptoms if treated early",
            "AUTISM-ABSENT-SPEECH-WORKUP — check urine GAA + creatine in ALL unexplained ID/autism before complex genetic testing",
            "BRAIN-MRS-ABSENT-CR-PEAK — non-invasive rapid diagnosis; Cr peak at 3.03 ppm absent; send for MRS in suspected creatine disorder",
            "NO-PROTEIN-RESTRICTION-NEEDED — unlike GAMT where arginine restriction used; AGAT: creatine alone sufficient",
            "EARLY-TREATMENT-CRITICAL — pretreatment period correlates with residual ID; treat as early as possible",
            "NBS-NOT-ROUTINE — not on standard newborn screen in most countries; high clinical suspicion needed in unexplained ID/autism",
        ],
    },
]


def _make_cohort(gene_entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    ages = [round(rng.gauss(6, 3), 1) for _ in range(n)]
    ages = [max(0.1, min(18.0, a)) for a in ages]
    sexes = [rng.choice(["M", "F"]) for _ in range(n)]
    if gene_entry["gene"] == "GCH1":
        sexes = [rng.choices(["M", "F"], weights=[1, 3])[0] for _ in range(n)]
    severities = [rng.choice(["mild", "moderate", "severe"]) for _ in range(n)]
    return [
        {
            "patient_id": f"{gene_entry['gene']}-{i+1:03d}",
            "gene": gene_entry["gene"],
            "age_at_presentation_yr": ages[i],
            "sex": sexes[i],
            "severity": severities[i],
            "inheritance": gene_entry["inheritance"],
            "locus": gene_entry["locus"],
        }
        for i in range(n)
    ]


def overview() -> dict:
    all_patients = []
    for idx, g in enumerate(NT_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        all_patients.extend(cohort)

    total = len(all_patients)
    gene_counts = {}
    for p in all_patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1

    age_vals = [p["age_at_presentation_yr"] for p in all_patients]
    avg_age = round(sum(age_vals) / len(age_vals), 1)
    severe_count = sum(1 for p in all_patients if p["severity"] == "severe")

    gene_summary = []
    for g in NT_GENES:
        gene_summary.append({
            "gene": g["gene"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "n_patients": gene_counts.get(g["gene"], 0),
            "critical_flags": g["critical_flags"],
        })

    return {
        "atlas": "Hereditary-Neurotransmitter-Synthesis-Disorders-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Neurotransmitter Synthesis Disorders Atlas — "
            "DDC-480aa-7p12.2-AR-AADC-Oculogyric-Crises-PATHOGNOMONIC-Upstaza-FDA2023 | "
            "GCH1-250aa-14q22.2-AD-DRD-Segawa-Diurnal-Variation-PATHOGNOMONIC-L-Dopa-CURATIVE | "
            "TH-498aa-11p15.5-AR-TH-Deficiency-BH4-NORMAL-Infantile-Parkinsonism-L-Dopa-Response | "
            "SPR-263aa-2p14-AR-CSF-Sepiapterin-PATHOGNOMONIC-L-Dopa-PLUS-5HTP-MANDATORY | "
            "ALDH7A1-539aa-5q31.2-AR-PDE-alpha-AASA-PATHOGNOMONIC-Pyridoxine-NOT-PLP | "
            "PNPO-261aa-17q21.32-AR-PLP-MANDATORY-NOT-Pyridoxine-Burst-Suppression-EEG | "
            "SLC6A3-620aa-5p15.33-AR-DTDS-DAT-SPECT-PATHOGNOMONIC-L-Dopa-ABSOLUTELY-CI | "
            "GATM-423aa-15q21.1-AR-AGAT-GAA-LOW-Creatine-400mgkgday-CURATIVE | "
            "320-Patient-Aggregate-8x40-seeds-1878-1885"
        ),
        "aggregate_stats": {
            "total_patients": total,
            "genes_covered": len(NT_GENES),
            "avg_age_at_presentation_yr": avg_age,
            "severe_cases_pct": round(100 * severe_count / total, 1),
            "seed_range": f"{SEED_BASE}–{SEED_BASE + len(NT_GENES) - 1}",
        },
        "gene_summary": gene_summary,
        "key_clinical_distinctions": [
            "DDC-vs-GCH1: both have OGC + hypotonia — CSF distinguishes: HVA+5HIAA both very low (DDC) vs HVA low 5HIAA mild (GCH1)",
            "SPR-vs-ALDH7A1-PNPO: SPR has CSF sepiapterin; ALDH7A1 has urine alpha-AASA; PNPO has plasma PLP low + pyridoxamine high",
            "PNPO-vs-ALDH7A1: PNPO fails pyridoxine (PLP needed); ALDH7A1 responds to pyridoxine (PLP pathway intact)",
            "SLC6A3-DTDS: HVA HIGH (not low) + DAT-SPECT absent + L-Dopa WORSENS — opposite to all other dopamine deficiency disorders",
            "GATM-vs-GAMT: GAA LOW in AGAT deficiency; GAA VERY HIGH in GAMT deficiency — same lab test, opposite result",
            "TH-vs-GCH1-AR: both have BH4 pathway involvement — BH4 LOW in GCH1 AR; BH4 NORMAL in TH deficiency",
            "GCH1-AD-DRD: diurnal variation + L-Dopa COMPLETE response — L-Dopa trial BEFORE any diagnostic procedure or surgery",
        ],
    }


def breakdown() -> dict:
    result = []
    for idx, g in enumerate(NT_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        severities = {}
        for p in cohort:
            severities[p["severity"]] = severities.get(p["severity"], 0) + 1
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "severity_distribution": severities,
            "n_patients": len(cohort),
            "patients": cohort[:5],
        })
    return {"genes": result, "total_genes": len(NT_GENES)}


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Neurotransmitter-Synthesis-Disorders-Atlas",
        "genes": [
            {
                "gene": g["gene"],
                "definition": g["alias"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "age_of_onset": g["age_of_onset"],
                "critical_flags": g["critical_flags"],
            }
            for g in NT_GENES
        ],
        "glossary": {
            "AADC": "Aromatic L-amino acid decarboxylase (DDC gene) — converts L-DOPA to dopamine and 5-HTP to serotonin; PLP-dependent",
            "BH4": "Tetrahydrobiopterin — essential cofactor for TH (dopamine), TPH (serotonin), PAH (phenylalanine); made by GCH1→PTPS→SPR",
            "DAT": "Dopamine transporter (SLC6A3) — reuptakes dopamine from synapse into neuron; absent in DTDS",
            "DAT-SPECT": "FP-CIT (DaTSCAN) SPECT imaging — radiolabelled DAT ligand; absent binding in DTDS; reduced in Parkinson's",
            "DRD": "Dopa-responsive dystonia (Segawa disease) — GCH1 AD mutation causing BH4 deficiency; L-Dopa curative",
            "DTDS": "DAT deficiency syndrome — SLC6A3 AR mutations; dopamine transporter absent; L-Dopa worsens",
            "GAMT": "Guanidinoacetate methyltransferase — converts GAA to creatine (step 2); deficiency causes GAA accumulation",
            "GAA": "Guanidinoacetate — product of AGAT step 1; LOW in AGAT deficiency; VERY HIGH in GAMT deficiency",
            "GCH1": "GTP cyclohydrolase 1 — rate-limiting BH4 synthesis step; AD heterozygous LOF causes DRD; AR biallelic causes severe BH4 deficiency",
            "HVA": "Homovanillic acid — major dopamine metabolite in CSF; LOW in DDC/GCH1/TH/SPR deficiencies; HIGH in DTDS (SLC6A3)",
            "L-pipecolic acid": "Elevated in PDE (ALDH7A1) AND peroxisomal disorders — not specific; use alpha-AASA for PDE confirmation",
            "MRS": "Magnetic resonance spectroscopy — non-invasive brain metabolite measurement; absent Cr peak diagnoses creatine deficiency",
            "OGC": "Oculogyric crises — involuntary sustained upward eye deviation; PATHOGNOMONIC for AADC deficiency in infants",
            "P6C": "1-Piperidine-6-carboxylate — intermediate in lysine catabolism; inactivates PLP (Knoevenagel condensation) in PDE",
            "PDE": "Pyridoxine-dependent epilepsy — ALDH7A1/antiquitin deficiency; neonatal seizures responsive to pyridoxine",
            "PLP": "Pyridoxal-5-phosphate — active form of vitamin B6; cofactor for AADC, GAD, glycine cleavage, and >140 enzymes",
            "PNPO": "Pyridox(am)ine 5-phosphate oxidase — converts pyridoxamine-P and pyridoxine-P to PLP; deficiency requires direct PLP (not pyridoxine)",
            "alpha-AASA": "Alpha-aminoadipic semialdehyde — PATHOGNOMONIC biomarker for PDE/ALDH7A1 deficiency in urine",
            "5-HIAA": "5-hydroxyindoleacetic acid — major serotonin metabolite in CSF; LOW in DDC/SPR deficiencies",
            "5-HTP": "5-hydroxytryptophan — serotonin precursor (bypasses TPH step); used in SPR deficiency COMBINATION with L-Dopa",
            "sepiapterin": "CSF accumulation in SPR deficiency — PATHOGNOMONIC; accumulated precursor because SPR is blocked",
            "TH": "Tyrosine hydroxylase — rate-limiting catecholamine synthesis; TH deficiency causes infantile Parkinsonism; BH4-dependent",
            "Upstaza": "Eladocagene exuparvovec — AAV2 DDC gene therapy; bilateral putamen injection; FDA/EMA approved 2022-2023",
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2))
