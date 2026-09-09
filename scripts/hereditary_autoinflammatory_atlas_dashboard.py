#!/usr/bin/env python3
"""Hereditary-Autoinflammatory-Syndrome-Atlas — Complete 8-Gene Autoinflammatory Disease Atlas
(MEFV · TNFRSF1A · NLRP3 · MVK · PSTPIP1 · NOD2 · IL1RN · IL36RN).

MEFV     (Pyrin/Marenostrin; 781 aa; ~95 kDa; 16p13.3; AR with incomplete penetrance;
          Familial Mediterranean Fever (FMF);
          MOST COMMON autoinflammatory disease worldwide; Mediterranean/Middle Eastern enriched;
          SHORT SELF-LIMITING FEVER ATTACKS 12-72h + SEROSITIS — PATHOGNOMONIC;
          AA AMYLOIDOSIS risk highest in M694V homozygotes;
          Colchicine 1-2 mg/day FIRST-LINE LIFELONG — prevents attacks AND amyloid;
          seed SEED_BASE+0).
TNFRSF1A (TNF Receptor 1/TNFR1; 455 aa; ~50 kDa; 12p13.31; AD;
          TRAPS — TNF Receptor-Associated Periodic Syndrome;
          LONGEST FEVER ATTACKS among hereditary periodic fevers — >7 days typical;
          MIGRATORY CENTRIFUGAL RASH + PERIORBITAL EDEMA + MYALGIA PATHOGNOMONIC;
          AA amyloidosis risk HIGHEST of all hereditary fevers;
          IL-1 blockade (anakinra/canakinumab) SUPERIOR to etanercept — first-line;
          seed SEED_BASE+1).
NLRP3    (Cryopyrin/PYPAF1; 1036 aa; ~118 kDa; 1q44; AD GOF;
          CAPS — Cryopyrin-Associated Periodic Syndrome; spectrum: FCAS < MWS < NOMID;
          COLD-TRIGGERED urticaria-like RASH + fever + sensorineural hearing loss;
          NOMID: neonatal onset, chronic meningitis, arthropathy, deafness;
          CANAKINUMAB (FDA 2009) FIRST-LINE — DRAMATIC RESPONSE pathognomonic;
          seed SEED_BASE+2).
MVK      (Mevalonate Kinase; 396 aa; ~45 kDa; 12q24.11; AR;
          MKD/HIDS — Mevalonate Kinase Deficiency / Hyperimmunoglobulinemia D Syndrome;
          VACCINATION-TRIGGERED ATTACKS PATHOGNOMONIC;
          Elevated URINARY MEVALONATE during attacks — biochemical PATHOGNOMONIC;
          Anakinra/canakinumab effective; Dutch/European enriched;
          seed SEED_BASE+3).
PSTPIP1  (Proline-Serine-Threonine Phosphatase-Interacting Protein 1; 416 aa; ~47 kDa; 15q24.3; AD;
          PAPA Syndrome — Pyogenic Arthritis, Pyoderma Gangrenosum, Acne;
          STERILE DESTRUCTIVE ARTHRITIS + PYODERMA GANGRENOSUM + CYSTIC ACNE TRIAD PATHOGNOMONIC;
          p.A230T and p.E250K founder variants — most PAPA cases;
          TNF blockade for skin/arthritis; IL-1 blockade for refractory;
          seed SEED_BASE+4).
NOD2     (CARD15; 1040 aa; ~115 kDa; 16q12.1; AD;
          Blau Syndrome — Early-Onset Sarcoidosis;
          GRANULOMATOUS UVEITIS + POLYARTHRITIS + SKIN RASH TRIAD PATHOGNOMONIC;
          NOD2 GOF → NF-kB activation → granuloma formation; childhood onset <4 years;
          Uveitis MOST SEVERE complication — can cause blindness;
          TNF blockade (adalimumab/infliximab) most effective;
          seed SEED_BASE+5).
IL1RN    (IL-1 Receptor Antagonist/IL-1Ra; 177 aa; ~25 kDa; 2q14.1; AR;
          DIRA — Deficiency of IL-1 Receptor Antagonist;
          NEONATAL ONSET within first weeks PATHOGNOMONIC;
          MULTIFOCAL OSTEOMYELITIS + PERIOSTITIS + STERILE PUSTULOSIS TRIAD PATHOGNOMONIC;
          ANAKINRA CURATIVE — dramatic response within 24-48h; WITHOUT TREATMENT: LETHAL;
          seed SEED_BASE+6).
IL36RN   (IL-36 Receptor Antagonist/IL-36Ra; 155 aa; ~17 kDa; 2q14.1; AR;
          DITRA — Deficiency of IL-36 Receptor Antagonist;
          GENERALIZED PUSTULAR PSORIASIS (GPP) — recurrent pustular eruptions PATHOGNOMONIC;
          SPESOLIMAB (anti-IL-36R, FDA 2022) FIRST APPROVED TARGETED THERAPY for GPP;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2342-2349).
"""

import random

SEED_BASE = 2342

AUTOINFLAMMATORY_GENES = [
    # -- MEFV -- FMF (Pyrin/Marenostrin) -------------------------------------------------------
    {
        "gene": "MEFV",
        "alt_name": (
            "MEFV (MEFV-781aa-16p13.3 / AR-Incomplete-Penetrance -- FMF -- "
            "MOST-COMMON-AUTOINFLAMMATORY-DISEASE-WORLDWIDE -- "
            "SHORT-SELF-LIMITING-FEVER-12-72h+SEROSITIS-PATHOGNOMONIC -- "
            "AA-AMYLOIDOSIS-M694V-HOMOZYGOTES-HIGHEST-RISK -- "
            "COLCHICINE-1-2mg-LIFELONG-FIRST-LINE-Prevents-Attacks-AND-Amyloid)"
        ),
        "protein": (
            "MEFV -- 16p13.3 AR -- MEFV-781aa -- "
            "Pyrin-Marenostrin-95kDa-B30.2-SPRY-Domain-C-Terminal-PYD-Domain-N-Terminal -- "
            "Expressed-In-Neutrophils-Monocytes-Dendritic-Cells-Peritoneal-Fibroblasts -- "
            "Pyrin-Regulates-Neutrophil-Apoptosis-And-Suppresses-NLRP3-Inflammasome -- "
            "MEFV-Variants-Gain-Of-Function-Pyrin-Cannot-Suppress-Caspase-1-Activation -- "
            "IL-18-IL-1B-Via-Pyrin-Inflammasome-Pathway-Uncontrolled-Activation -- "
            "M694V-Homozygotes-Highest-Amyloid-Risk-M694V-Most-Pathogenic-Variant -- "
            "SEROSITIS-Peritonitis-Pleuritis-Pericarditis-Synovitis-Self-Limiting-Attacks -- "
            "OMIM-Gene-608107-Disease-FMF-249100"
        ),
        "locus": "16p13.3",
        "protein_size": "781 aa / 95 kDa",
        "inheritance": (
            "AR with incomplete penetrance; biallelic variants most severe; "
            "heterozygous carriers can be symptomatic (incomplete penetrance); "
            "Sephardic Jewish, Armenian, Turkish, Arab -- Mediterranean/Middle Eastern enriched; "
            "p.M694V most common and most pathogenic"
        ),
        "disease_category": "FMF -- Familial Mediterranean Fever (Most common autoinflammatory disease worldwide)",
        "disease_pathway": (
            "Pyrin normally suppresses NLRP3 inflammasome and caspase-1 activation in neutrophils. "
            "MEFV gain-of-function variants -> pyrin loses suppressive function -> unchecked pyrin inflammasome -> "
            "excessive IL-1beta/IL-18 cleavage and release -> recurrent self-limiting systemic inflammation. "
            "Persistent low-grade IL-1beta/SAA -> AA amyloid deposition in kidneys (end-stage complication)."
        ),
        "pathognomonic": (
            "SHORT SELF-LIMITING FEVER ATTACKS 12-72h + SEROSITIS (peritonitis most common, pleuritis, pericarditis, synovitis) = FMF PATHOGNOMONIC. "
            "APRs (CRP, SAA, fibrinogen) strikingly elevated DURING attacks, normalise BETWEEN. "
            "AA AMYLOIDOSIS: proteinuria -> nephrotic syndrome -> renal failure in untreated M694V homozygotes. "
            "Colchicine trial: dramatic attack reduction confirms diagnosis (therapeutic diagnosis)."
        ),
        "treatment": (
            "COLCHICINE 1-2 mg/day LIFELONG -- FIRST-LINE, mandatory: prevents attacks AND prevents AA amyloidosis. "
            "Colchicine-resistant (approx 5-10%): anakinra (anti-IL-1Ra) or canakinumab (anti-IL-1beta) -- FDA-approved. "
            "IL-18 pathway inhibitors under investigation. GI side effects: reduce dose, split dosing. "
            "Monitoring: urine albumin/creatinine ratio annually (amyloid screen); SAA level target <10 mg/L."
        ),
        "key_features": [
            "Most common autoinflammatory disease worldwide",
            "Short self-limiting attacks 12-72h (PATHOGNOMONIC)",
            "Serositis: peritonitis most common, then pleuritis",
            "AA amyloidosis risk -- highest in M694V homozygotes",
            "Colchicine 1-2mg/day lifelong FIRST-LINE",
            "Sephardic Jewish/Armenian/Turkish/Arab enriched",
        ],
        "key_ddx": (
            "TRAPS (TNFRSF1A): longer attacks >7 days; migratory rash; periorbital edema; no colchicine response. "
            "PFAPA: mainly children; predictable periodicity; good response to corticosteroids. "
            "FMF vs appendicitis: peritonitis of FMF self-resolves in 24-72h -- observe before surgery. "
            "HIDS (MVK): vaccination-triggered; high IgD; urinary mevalonate elevated."
        ),
        "amyloid_risk": "HIGH -- AA amyloidosis (M694V homozygotes highest); prevented by colchicine",
        "colchicine_response": "EXCELLENT -- first-line; 90-95% attack reduction in compliant patients",
        "il1_response": "Yes -- anakinra/canakinumab for colchicine-resistant FMF",
        "hsct_required": "No",
        "attack_trigger_common": "Stress, minor trauma, infection, cold, fatigue",
        "onset_age": "Childhood or adolescence; 80% onset before age 20; rarely adult-onset",
    },
    # -- TNFRSF1A -- TRAPS -------------------------------------------------------
    {
        "gene": "TNFRSF1A",
        "alt_name": (
            "TNFRSF1A (TNFRSF1A-455aa-12p13.31 / AD -- TRAPS -- "
            "LONGEST-FEVER-ATTACKS->7-DAYS-HEREDITARY-PERIODIC-FEVERS -- "
            "MIGRATORY-CENTRIFUGAL-RASH+PERIORBITAL-EDEMA+MYALGIA-PATHOGNOMONIC -- "
            "HIGHEST-AA-AMYLOID-RISK-All-Hereditary-Fevers -- "
            "IL-1-BLOCKADE-SUPERIOR-Etanercept-Partial-Only)"
        ),
        "protein": (
            "TNFRSF1A -- 12p13.31 AD -- TNFRSF1A-455aa -- "
            "TNF-Receptor-Superfamily-Member-1A-TNFR1-p55-50kDa-Type-I-Transmembrane -- "
            "4-Cysteine-Rich-Domains-CRD1-CRD4-Extracellular-PLAD-Domain -- "
            "Missense-Variants-CRD1-CRD2-Most-Pathogenic-Impair-Receptor-Shedding -- "
            "Defective-TNFR1-Shedding-Sustained-TNF-Signalling-Mechanism -- "
            "Misfolded-TNFR1-ER-Retention-Unfolded-Protein-Response-Alternative -- "
            "Low-Penetrance-Variants-p.R92Q-p.P46L-Fever-Susceptibility-Not-Classic-TRAPS -- "
            "OMIM-Gene-191190-Disease-TRAPS-142680"
        ),
        "locus": "12p13.31",
        "protein_size": "455 aa / 50 kDa",
        "inheritance": (
            "AD (autosomal dominant); high-penetrance structural variants (CRD1/CRD2 Cys residues) cause classic TRAPS; "
            "low-penetrance variants (p.R92Q, p.P46L) cause fever susceptibility not classic TRAPS; "
            "Irish/Scottish enriched but all ethnicities"
        ),
        "disease_category": "TRAPS -- TNF Receptor-Associated Periodic Syndrome",
        "disease_pathway": (
            "Structural TNFR1 variants impair receptor shedding from cell surface -> "
            "sustained membrane-bound TNFR1 signalling -> prolonged TNF-driven inflammation. "
            "Alternative: misfolded TNFR1 retained in ER -> UPR + intracellular ROS -> NF-kB activation. "
            "Result: prolonged fever attacks driven by IL-1beta and TNF -- responds poorly to etanercept, "
            "well to IL-1 blockade (suppresses downstream cytokine)."
        ),
        "pathognomonic": (
            "FEVER ATTACKS >7 DAYS (longest among hereditary periodic fevers) -- may last weeks. "
            "MIGRATORY CENTRIFUGAL RASH (erysipelas-like, migrates from trunk to periphery) -- PATHOGNOMONIC. "
            "PERIORBITAL EDEMA (unilateral or bilateral) -- PATHOGNOMONIC unique feature. "
            "MYALGIA (severe, migratory) accompanying fever and rash. "
            "AA AMYLOIDOSIS: highest risk of all hereditary fever syndromes -- mandatory monitoring."
        ),
        "treatment": (
            "IL-1 BLOCKADE FIRST-LINE: anakinra (daily SC) or canakinumab (q8w SC) -- dramatic response. "
            "Etanercept: partial response only (not recommended as primary monotherapy for classic TRAPS). "
            "Corticosteroids: short-term attack rescue; do NOT prevent amyloid. "
            "Amyloid monitoring: urine albumin/creatinine ratio annually; SAA target <10 mg/L. "
            "Low-penetrance variants (p.R92Q): milder course, individual treatment decisions."
        ),
        "key_features": [
            "Longest fever attacks >7 days (PATHOGNOMONIC) among periodic fevers",
            "Migratory centrifugal rash PATHOGNOMONIC",
            "Periorbital edema PATHOGNOMONIC unique feature",
            "Highest AA amyloid risk of all hereditary fevers",
            "IL-1 blockade superior to etanercept",
            "Low-penetrance p.R92Q/p.P46L -- fever susceptibility only",
        ],
        "key_ddx": (
            "FMF (MEFV): shorter attacks 12-72h; serositis dominant; colchicine responsive; no rash/periorbital edema. "
            "CAPS (NLRP3): cold-triggered urticaria-like rash; sensorineural deafness; neonatal onset possible. "
            "Adult-onset Still's: quotidian fever, salmon rash, arthritis -- no family history, no serositis periodicity. "
            "TRAPS p.R92Q DDx: low-penetrance; does not justify anti-IL-1 monotherapy without classic phenotype."
        ),
        "amyloid_risk": "VERY HIGH -- highest AA amyloidosis risk of all hereditary periodic fever syndromes",
        "colchicine_response": "Poor -- colchicine not effective for TRAPS",
        "il1_response": "Excellent -- anakinra/canakinumab first-line",
        "hsct_required": "No",
        "attack_trigger_common": "Stress, minor infection, physical exertion, menstruation, spontaneous",
        "onset_age": "Any age including adult onset; mean onset childhood/adolescence; family history critical",
    },
    # -- NLRP3 -- CAPS -------------------------------------------------------
    {
        "gene": "NLRP3",
        "alt_name": (
            "NLRP3 (NLRP3-1036aa-1q44 / AD-GOF -- CAPS -- "
            "CRYOPYRIN-ASSOCIATED-PERIODIC-SYNDROME-FCAS-MWS-NOMID-Spectrum -- "
            "COLD-TRIGGERED-URTICARIA-RASH+FEVER+SENSORINEURAL-DEAFNESS-PATHOGNOMONIC -- "
            "NOMID-NEONATAL-ONSET-CHRONIC-MENINGITIS-ARTHROPATHY-DEAFNESS -- "
            "CANAKINUMAB-FDA2009-FIRST-LINE-DRAMATIC-RESPONSE-PATHOGNOMONIC)"
        ),
        "protein": (
            "NLRP3 -- 1q44 AD-GOF -- NLRP3-1036aa -- "
            "Cryopyrin-PYPAF1-NALP3-118kDa-NLR-Family-Pyrin-Domain-Containing-3 -- "
            "PYD-Domain-NACHT-ATPase-Domain-LRR-Domain-Architecture -- "
            "NLRP3-Inflammasome-Sensor-Activates-On-Danger-Signals-ATP-Urate-Cholesterol -- "
            "NLRP3-Recruits-ASC-PYCARD-Caspase-1-Activation-IL-1B-IL-18-Cleavage-Maturation -- "
            "GOF-Variants-Lower-Activation-Threshold-Excessive-IL-1B-Cleavage-Constitutive -- "
            "Spectrum-FCAS-p.R260W-Mildest-Cold-Urticaria-MWS-p.D303N-Hearing-Loss-NOMID-Severe -- "
            "OMIM-Gene-606416-Disease-CAPS-120100-191900-607115"
        ),
        "locus": "1q44",
        "protein_size": "1036 aa / 118 kDa",
        "inheritance": (
            "AD (autosomal dominant) GOF; de novo mutations common especially in NOMID (~50%); "
            "somatic mosaicism accounts for some NOMID without family history; "
            "phenotypic spectrum: FCAS (mildest) < MWS (intermediate) < NOMID (most severe)"
        ),
        "disease_category": "CAPS -- Cryopyrin-Associated Periodic Syndrome (FCAS/MWS/NOMID spectrum)",
        "disease_pathway": (
            "NLRP3 GOF variants lower the activation threshold of the NLRP3 inflammasome -> "
            "excessive, sustained IL-1beta cleavage and release -> systemic sterile inflammation. "
            "Cold exposure is the canonical trigger (especially FCAS). "
            "CNS: NLRP3 activation in meningeal macrophages -> chronic meningitis -> brain atrophy (NOMID). "
            "Inner ear: IL-1beta-driven cochlear inflammation -> sensorineural hearing loss (MWS/NOMID)."
        ),
        "pathognomonic": (
            "COLD-TRIGGERED URTICARIA-LIKE RASH (neutrophilic, not mast-cell urticaria) + FEVER = FCAS/CAPS PATHOGNOMONIC. "
            "SENSORINEURAL HEARING LOSS + recurrent fever + urticaria rash = MWS PATHOGNOMONIC. "
            "NEONATAL ONSET + CHRONIC ASEPTIC MENINGITIS + DESTRUCTIVE ARTHROPATHY + DEAFNESS = NOMID PATHOGNOMONIC. "
            "DRAMATIC RESPONSE TO CANAKINUMAB within 24-48h -- confirms diagnosis therapeutically."
        ),
        "treatment": (
            "CANAKINUMAB (anti-IL-1beta, FDA 2009): 150 mg SC q8w (standard dose) -- FIRST-LINE, dramatic response. "
            "Rilonacept (IL-1 trap, FDA 2008): alternate for FCAS/MWS. "
            "Anakinra (daily): effective but less convenient than canakinumab. "
            "NOMID: higher canakinumab doses often required; early treatment prevents hearing loss and cognitive impairment. "
            "MRI brain annually in NOMID; audiology every 6 months; ophthalmology (disc edema in NOMID)."
        ),
        "key_features": [
            "Cold-triggered urticaria-like rash PATHOGNOMONIC",
            "Spectrum: FCAS (mildest) -> MWS -> NOMID (most severe)",
            "Sensorineural hearing loss in MWS/NOMID",
            "NOMID: neonatal, chronic meningitis, joint destruction",
            "Canakinumab (FDA 2009) -- first-line, dramatic response",
            "De novo mutations common in NOMID (~50%)",
        ],
        "key_ddx": (
            "Mast-cell urticaria: triggered by allergens, not cold exclusively; no fever pattern; antihistamine response. "
            "FMF (MEFV): not cold-triggered; serositis not urticaria; colchicine responsive. "
            "TRAPS: longer attacks; no cold trigger; migratory rash different from urticaria. "
            "Chronic meningitis (NOMID DDx): exclude infection (CSF culture/PCR) before diagnosing NOMID."
        ),
        "amyloid_risk": "Moderate -- AA amyloidosis possible in untreated MWS/NOMID; prevented by IL-1 blockade",
        "colchicine_response": "Poor -- colchicine not effective for CAPS",
        "il1_response": "DRAMATIC -- canakinumab/anakinra first-line; response within 24-48h pathognomonic",
        "hsct_required": "No",
        "attack_trigger_common": "Cold exposure (FCAS/MWS), spontaneous/continuous (NOMID)",
        "onset_age": "FCAS/MWS: early childhood; NOMID: neonatal (within first weeks of life)",
    },
    # -- MVK -- MKD/HIDS -------------------------------------------------------
    {
        "gene": "MVK",
        "alt_name": (
            "MVK (MVK-396aa-12q24.11 / AR -- MKD-HIDS -- "
            "MEVALONATE-KINASE-DEFICIENCY-Hyperimmunoglobulinemia-D-Periodic-Fever-Syndrome -- "
            "VACCINATION-TRIGGERED-ATTACKS-PATHOGNOMONIC-No-Other-Periodic-Fever-Reliably-Triggered -- "
            "ELEVATED-URINARY-MEVALONATE-During-Attacks-Biochemical-PATHOGNOMONIC -- "
            "HIGH-IgD->100-IU-mL-HIDS-Anakinra-Canakinumab-Effective)"
        ),
        "protein": (
            "MVK -- 12q24.11 AR -- MVK-396aa -- "
            "Mevalonate-Kinase-45kDa-ATP-Binding-Enzyme-Mevalonate-Pathway -- "
            "Phosphorylates-Mevalonate-To-5-Phosphomevalonate-Cholesterol-Isoprenoid-Synthesis -- "
            "MVK-Deficiency-Mevalonate-Accumulates-Isoprenoid-Deficiency-Downstream -- "
            "HIDS-p.Val377Ile-Founder-Dutch-Northern-European-Hypomorphic-5pct-Residual-Activity -- "
            "Mevalonic-Aciduria-Severe-End-Complete-Loss-Ataxia-Dysmorphism-Intellectual-Disability -- "
            "High-IgD->100-IU-mL-HIDS-Not-Mevalonic-Aciduria-Severe-End -- "
            "Urinary-Mevalonate-Elevated-During-Attacks-Biochemical-Confirmation -- "
            "OMIM-Gene-251170-Disease-HIDS-260920-MVA-610377"
        ),
        "locus": "12q24.11",
        "protein_size": "396 aa / 45 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic; "
            "p.Val377Ile Dutch/Northern European founder (~80% of HIDS alleles in Netherlands); "
            "spectrum: HIDS (hypomorphic, ~5% residual MVK activity) -> mevalonic aciduria (complete loss)"
        ),
        "disease_category": "MKD/HIDS -- Mevalonate Kinase Deficiency / Hyperimmunoglobulinemia D with Periodic Fever Syndrome",
        "disease_pathway": (
            "MVK deficiency -> mevalonate accumulates -> downstream isoprenoid/cholesterol synthesis impaired. "
            "Isoprenoid deficiency -> failure to geranylgeranylate Rac1/other GTPases -> caspase-1 activation -> IL-1beta release. "
            "Vaccination (immune activation) sharply lowers residual MVK activity -> predictable attack trigger. "
            "Severe MVK loss (mevalonic aciduria): dysmorphism, cerebellar ataxia, "
            "intellectual disability from isoprenoid deficiency."
        ),
        "pathognomonic": (
            "VACCINATION-TRIGGERED ATTACKS -- no other hereditary periodic fever is so reliably triggered by vaccination. "
            "ELEVATED URINARY MEVALONATE during attacks -- biochemical PATHOGNOMONIC (send spot urine during attack). "
            "High IgD (>100 IU/mL) in HIDS -- not always present, not pathognomonic alone. "
            "Fever + lymphadenopathy + abdominal pain + diarrhoea/vomiting + rash -- attack phenotype. "
            "Mevalonate plasma levels elevated in mevalonic aciduria -- severe end."
        ),
        "treatment": (
            "ANAKINRA (anti-IL-1Ra) or CANAKINUMAB (anti-IL-1beta): effective for attack prevention and reduction. "
            "Simvastatin: some evidence for attack frequency reduction in HIDS. "
            "Etanercept: reported benefit in older series. "
            "Mevalonic aciduria (severe): simvastatin + supportive; HSCT reported in severe cases. "
            "Vaccinations: use pre-medication (ibuprofen/anakinra) to blunt post-vaccination attacks -- do NOT withhold vaccines."
        ),
        "key_features": [
            "Vaccination-triggered attacks PATHOGNOMONIC",
            "Elevated urinary mevalonate during attacks -- biochemical PATHOGNOMONIC",
            "High IgD (>100 IU/mL) in HIDS (not in severe MKD/mevalonic aciduria)",
            "Dutch/Northern European p.Val377Ile founder",
            "Spectrum: HIDS (mild) -> mevalonic aciduria (severe, neurological)",
            "Anakinra/canakinumab effective",
        ],
        "key_ddx": (
            "FMF (MEFV): shorter attacks; serositis; no vaccination trigger; no IgD elevation; colchicine responsive. "
            "TRAPS: longer attacks; migratory rash; periorbital edema; Irish/Scottish enriched. "
            "PFAPA: tonsillitis + aphthous ulcers + lymphadenopathy; responds to single prednisolone dose. "
            "Mevalonic aciduria DDx from HIDS: severity, ataxia, dysmorphism, near-zero residual MVK activity."
        ),
        "amyloid_risk": "Low -- AA amyloidosis rare in HIDS; more risk in severe MKD",
        "colchicine_response": "Poor -- not effective for MKD/HIDS",
        "il1_response": "Yes -- anakinra/canakinumab effective",
        "hsct_required": "Rarely -- mevalonic aciduria severe cases only",
        "attack_trigger_common": "Vaccination, infection, minor stress, spontaneous",
        "onset_age": "First year of life; attacks from infancy; IgD elevation may take years to develop",
    },
    # -- PSTPIP1 -- PAPA Syndrome -------------------------------------------------------
    {
        "gene": "PSTPIP1",
        "alt_name": (
            "PSTPIP1 (PSTPIP1-416aa-15q24.3 / AD -- PAPA-Syndrome -- "
            "PYOGENIC-ARTHRITIS+PYODERMA-GANGRENOSUM+CYSTIC-ACNE-TRIAD-PATHOGNOMONIC -- "
            "p.A230T-AND-p.E250K-FOUNDER-Most-PAPA-Cases -- "
            "PSTPIP1-BINDS-PYRIN-LOF-Pyrin-Cannot-Suppress-IL-1B-NLRP3 -- "
            "TNF-Blockade-Skin-Arthritis-IL-1-Blockade-Refractory)"
        ),
        "protein": (
            "PSTPIP1 -- 15q24.3 AD -- PSTPIP1-416aa -- "
            "Proline-Serine-Threonine-Phosphatase-Interacting-Protein-1-47kDa-FCH-SH3 -- "
            "FCH-Domain-F-BAR-Coiled-Coil-SH3-Architecture-Cytoskeletal-Scaffolding -- "
            "PSTPIP1-Binds-Pyrin-In-Neutrophils-Monocytes-Stabilises-Pyrin-Suppressive-Function -- "
            "PSTPIP1-Missense-Variants-Hyperphosphorylated-Cannot-Bind-Pyrin-Properly -- "
            "Pyrin-Loses-NLRP3-IL-1B-Suppression-Sterile-Inflammation-Neutrophilic -- "
            "p.A230T-Ala230Thr-Most-Common-PAPA-Variant-Coiled-Coil-Domain -- "
            "p.E250K-Glu250Lys-Second-Founder-Variant-PAPA-SH3-Adjacent -- "
            "OMIM-Gene-606347-Disease-PAPA-604416"
        ),
        "locus": "15q24.3",
        "protein_size": "416 aa / 47 kDa",
        "inheritance": (
            "AD (autosomal dominant); missense gain-of-toxic-function; "
            "p.A230T and p.E250K account for most PAPA families; "
            "variable expressivity -- not all carriers develop all three features of the triad; "
            "rare syndrome -- fewer than 200 families reported"
        ),
        "disease_category": "PAPA Syndrome -- Pyogenic Arthritis, Pyoderma Gangrenosum, Acne",
        "disease_pathway": (
            "PSTPIP1 binds pyrin to maintain its suppressive function on NLRP3 and IL-1beta. "
            "Missense PSTPIP1 variants -> hyperphosphorylation -> cannot bind pyrin -> "
            "pyrin loses suppressive role -> neutrophil-driven sterile inflammation in joints and skin. "
            "Joint: sterile destructive pyogenic arthritis with culture-negative synovial fluid. "
            "Skin: IL-1beta/TNF-driven neutrophilic dermatosis -> pyoderma gangrenosum + nodulocystic acne."
        ),
        "pathognomonic": (
            "STERILE DESTRUCTIVE ARTHRITIS (culture-negative synovial fluid) + PYODERMA GANGRENOSUM + CYSTIC ACNE = PAPA TRIAD PATHOGNOMONIC. "
            "Arthritis: typically monoarticular/oligoarticular; large joints (knee, ankle); no infection on culture. "
            "Pyoderma gangrenosum: rapidly enlarging ulcers with undermined violaceous borders -- pathergy positive. "
            "Triad need not be simultaneous; acne dominates in adults, arthritis in children."
        ),
        "treatment": (
            "ARTHRITIS: corticosteroids (intra-articular or systemic); TNF blockade (infliximab/adalimumab). "
            "PYODERMA GANGRENOSUM: wound care; corticosteroids; cyclosporin; TNF blockade. "
            "IL-1 BLOCKADE (anakinra): for refractory arthritis and skin disease -- most effective single agent. "
            "ACNE: isotretinoin; TNF blockade for severe cystic acne. "
            "Avoid surgery on pyoderma gangrenosum wounds (pathergy risk worsens lesion). Biologics often lifelong."
        ),
        "key_features": [
            "Pyogenic arthritis + pyoderma gangrenosum + cystic acne TRIAD PATHOGNOMONIC",
            "Arthritis: sterile (culture-negative synovial fluid)",
            "Pathergy-positive pyoderma gangrenosum",
            "p.A230T and p.E250K founder variants",
            "PSTPIP1 binds pyrin -- LOF -> IL-1beta/NLRP3 dysregulation",
            "IL-1 blockade most effective single agent",
        ],
        "key_ddx": (
            "Septic arthritis: culture-positive synovial fluid; no pyoderma/acne family history. "
            "Crohn's-associated arthritis and pyoderma: IBD present; NOD2 variants; no PSTPIP1 variants. "
            "SAPHO syndrome: synovitis/acne/pustulosis/hyperostosis/osteitis -- no destructive arthritis. "
            "Behcet's: oral/genital ulcers; pathergy; no PSTPIP1 gene variant."
        ),
        "amyloid_risk": "Low -- AA amyloidosis rarely reported in PAPA",
        "colchicine_response": "Variable -- limited evidence; not first-line",
        "il1_response": "Yes -- anakinra first-line for refractory disease",
        "hsct_required": "No",
        "attack_trigger_common": "Trauma, stress, spontaneous, pathergy-provoking procedures",
        "onset_age": "Childhood onset (arthritis early, pyoderma/acne in adolescence/adulthood); variable expressivity",
    },
    # -- NOD2 -- Blau Syndrome -------------------------------------------------------
    {
        "gene": "NOD2",
        "alt_name": (
            "NOD2 (NOD2-1040aa-16q12.1 / AD -- Blau-Syndrome-Early-Onset-Sarcoidosis -- "
            "GRANULOMATOUS-UVEITIS+POLYARTHRITIS+SKIN-RASH-TRIAD-PATHOGNOMONIC -- "
            "CHILDHOOD-ONSET-<4-YEARS-NOD2-GOF-NF-kB-Granuloma-Formation -- "
            "UVEITIS-MOST-SEVERE-Blindness-Risk-SAME-LOCUS-Crohns-Different-Variants -- "
            "TNF-BLOCKADE-ADALIMUMAB-Most-Effective-Uveitis-Arthritis)"
        ),
        "protein": (
            "NOD2 -- 16q12.1 AD -- NOD2-1040aa -- "
            "CARD15-Nucleotide-Binding-Oligomerisation-Domain-2-115kDa-NLR-Family -- "
            "2-CARD-Domains-NACHT-ATPase-LRR-Architecture-Cytoplasmic-PRR -- "
            "Senses-Muramyl-Dipeptide-MDP-Bacterial-Cell-Wall-Fragment -- "
            "NOD2-GOF-Variants-Blau-Lower-Activation-Threshold-Spontaneous-NF-kB-Signalling -- "
            "Granuloma-Formation-Non-Caseating-Epithelioid-Macrophage-Giant-Cell-Clusters -- "
            "SAME-LOCUS-AS-CROHNS-RISK-BUT-DIFFERENT-VARIANTS-Blau-GOF-Crohns-LOF -- "
            "Uveitis-Panuveitis-Granulomatous-Most-Severe-Complication-Blindness-Risk -- "
            "OMIM-Gene-605956-Disease-Blau-186580"
        ),
        "locus": "16q12.1",
        "protein_size": "1040 aa / 115 kDa",
        "inheritance": (
            "AD (autosomal dominant) GOF; de novo mutations in simplex cases (~50%); "
            "same chromosomal locus 16q12.1 as Crohn's disease susceptibility variants -- "
            "KEY DISTINCTION: Blau = GOF variants (NACHT domain), Crohn's = LOF variants (LRR domain)"
        ),
        "disease_category": "Blau Syndrome -- Early-Onset Sarcoidosis (granulomatous uveitis/arthritis/dermatitis)",
        "disease_pathway": (
            "NOD2 GOF variants lower the threshold for MDP sensing -> "
            "constitutive NF-kB activation -> TNF, IL-12, IL-23 production -> "
            "macrophage polarisation into epithelioid granuloma-forming phenotype. "
            "Granulomas form without infection trigger -> chronic sterile granulomatous inflammation in "
            "synovium (arthritis), uveal tract (uveitis), skin (ichthyosis-like rash). "
            "Crohn's LOF variants: impaired bacterial sensing -> defective mucosal immunity -> gut inflammation."
        ),
        "pathognomonic": (
            "GRANULOMATOUS UVEITIS + POLYARTHRITIS + ICHTHYOSIS-LIKE SKIN RASH = BLAU TRIAD PATHOGNOMONIC. "
            "Onset <4 years (earlier onset than adult sarcoidosis). "
            "Synovium biopsy: non-caseating epithelioid granulomas. Skin biopsy: non-caseating granulomas. "
            "Uveitis: panuveitis, granulomatous -- most severe; can lead to blindness without treatment. "
            "ACE levels may be elevated. NOD2 sequencing confirms."
        ),
        "treatment": (
            "UVEITIS: methotrexate + adalimumab (TNF blockade) -- most effective biologic for ocular sarcoidosis. "
            "ARTHRITIS: methotrexate + TNF blockade (adalimumab/infliximab). "
            "IL-1 blockade (anakinra/canakinumab): some efficacy reported. "
            "Corticosteroids: topical (eye drops) + systemic for attacks -- limited long-term use. "
            "OPHTHALMIC MONITORING MANDATORY: every 3-6 months; OCT + slit-lamp; aggressive treatment prevents blindness."
        ),
        "key_features": [
            "Granulomatous uveitis + polyarthritis + skin rash TRIAD PATHOGNOMONIC",
            "Childhood onset <4 years",
            "NOD2 GOF (NOT same as Crohn's LOF variants) -- KEY DDx",
            "Uveitis: most severe complication -- blindness risk",
            "Non-caseating granulomas on biopsy",
            "Adalimumab/infliximab most effective (TNF blockade)",
        ],
        "key_ddx": (
            "Juvenile idiopathic arthritis (JIA): no granulomatous uveitis pattern; no skin ichthyosis; no NOD2 GOF. "
            "Paediatric sarcoidosis: sporadic; later onset; ACE elevated; no NOD2 variant usually. "
            "Crohn's disease: same locus but LOF variants; gut inflammation; no granulomatous uveitis triad. "
            "Blau vs Crohn's DDx: NOD2 variant type (GOF vs LOF) resolves ambiguity."
        ),
        "amyloid_risk": "Low -- AA amyloidosis rarely reported in Blau syndrome",
        "colchicine_response": "Poor -- not indicated for Blau syndrome",
        "il1_response": "Variable -- some cases respond; TNF blockade preferred",
        "hsct_required": "No",
        "attack_trigger_common": "Spontaneous (continuous inflammation, not episodic attacks)",
        "onset_age": "Early childhood, typically before age 4; skin rash often first manifestation",
    },
    # -- IL1RN -- DIRA -------------------------------------------------------
    {
        "gene": "IL1RN",
        "alt_name": (
            "IL1RN (IL1RN-177aa-2q14.1 / AR -- DIRA -- "
            "DEFICIENCY-OF-IL-1-RECEPTOR-ANTAGONIST -- "
            "NEONATAL-ONSET-WITHIN-FIRST-WEEKS-PATHOGNOMONIC -- "
            "MULTIFOCAL-OSTEOMYELITIS+PERIOSTITIS+STERILE-PUSTULOSIS-TRIAD-PATHOGNOMONIC -- "
            "ANAKINRA-CURATIVE-24-48h-DRAMATIC-RESPONSE-WITHOUT-TREATMENT-LETHAL)"
        ),
        "protein": (
            "IL1RN -- 2q14.1 AR -- IL1RN-177aa -- "
            "IL-1-Receptor-Antagonist-IL-1Ra-25kDa-Cytokine-Anti-Inflammatory -- "
            "Competitive-Inhibitor-Binds-IL-1R1-Without-Signalling-Blocks-IL-1A-IL-1B -- "
            "IL-1Ra-Absent-Unopposed-IL-1A-IL-1B-Systemic-Sterile-Inflammation -- "
            "Bone-Periosteum-Skin-Major-Targets-In-DIRA-Osteomyelitis-Periostitis-Pustulosis -- "
            "Puerto-Rico-Founder-72kb-Deletion-Encompassing-IL1RN -- "
            "Dutch-Founder-Deletion-European-DIRA-Cases -- "
            "Anakinra-Recombinant-IL-1Ra-Replacement-Therapy-CURATIVE-Lifelong -- "
            "OMIM-Gene-147679-Disease-DIRA-612852"
        ),
        "locus": "2q14.1",
        "protein_size": "177 aa / 25 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic deletions common (Puerto Rican 72kb founder deletion, Dutch deletion); "
            "rare but often founder-mutation-enriched in Puerto Rico and Netherlands; "
            "all ethnicities reported; consanguinity in some cases"
        ),
        "disease_category": "DIRA -- Deficiency of IL-1 Receptor Antagonist (neonatal autoinflammatory disease)",
        "disease_pathway": (
            "IL-1Ra is the physiological antagonist of IL-1alpha and IL-1beta -- binds IL-1R1 without triggering signalling. "
            "IL-1Ra absent -> unopposed IL-1alpha/IL-1beta signalling at all tissues -> "
            "sterile neutrophilic inflammation in bone (osteomyelitis, periostitis) and skin (pustulosis). "
            "Without IL-1Ra, even basal IL-1 is sufficient to drive continuous tissue destruction. "
            "Anakinra (recombinant IL-1Ra) replaces the missing protein -> dramatic resolution."
        ),
        "pathognomonic": (
            "NEONATAL ONSET (within first 2-4 weeks of life) PATHOGNOMONIC for DIRA. "
            "MULTIFOCAL OSTEOMYELITIS + PERIOSTITIS (rib, long bone) + STERILE SKIN PUSTULOSIS = DIRA TRIAD. "
            "Imaging: multifocal periosteal reaction/new bone formation -- distinctive radiographic pattern. "
            "Culture-negative pus from bone and skin. Without treatment: multiorgan failure and death. "
            "ANAKINRA response within 24-48h: pustules resolve, bone inflammation subsides -- confirms diagnosis."
        ),
        "treatment": (
            "ANAKINRA (recombinant IL-1Ra) CURATIVE: 1-2 mg/kg/day SC -- LIFELONG replacement therapy. "
            "Response within 24-48h: dramatic resolution of pustules and systemic inflammation. "
            "MUST NOT STOP anakinra -- disease recurs immediately; multiorgan failure risk. "
            "Canakinumab: reported as alternative in some cases. "
            "Supportive: nutritional support, pain management, wound care. "
            "WITHOUT ANAKINRA: lethal from multiorgan failure in untreated neonates."
        ),
        "key_features": [
            "Neonatal onset within first weeks PATHOGNOMONIC",
            "Multifocal osteomyelitis + periostitis + sterile pustulosis TRIAD",
            "Culture-negative bone/skin lesions",
            "IL-1Ra absent -> unopposed IL-1alpha/IL-1beta",
            "Anakinra curative -- response within 24-48h PATHOGNOMONIC",
            "Puerto Rican and Dutch founder deletions",
        ],
        "key_ddx": (
            "Neonatal septic osteomyelitis: culture-positive; responds to antibiotics; no pustulosis. "
            "DITRA (IL36RN): skin pustulosis but not bone; older onset; IL-36 not IL-1 pathway. "
            "NOMID (NLRP3): neonatal onset with skin rash + CNS but urticaria not pustulosis; cold trigger; GOF not AR. "
            "Neonatal infections: culture/PCR positive; no multifocal bone periostitis pattern."
        ),
        "amyloid_risk": "Low -- prevented by anakinra treatment",
        "colchicine_response": "Not effective",
        "il1_response": "CURATIVE -- anakinra (IL-1Ra replacement) pathognomonic dramatic response",
        "hsct_required": "No",
        "attack_trigger_common": "Spontaneous (continuous from neonatal period -- not episodic)",
        "onset_age": "Neonatal -- within first 2-4 weeks of life; no later presentations reported",
    },
    # -- IL36RN -- DITRA -------------------------------------------------------
    {
        "gene": "IL36RN",
        "alt_name": (
            "IL36RN (IL36RN-155aa-2q14.1 / AR -- DITRA -- "
            "DEFICIENCY-OF-IL-36-RECEPTOR-ANTAGONIST -- "
            "GENERALIZED-PUSTULAR-PSORIASIS-GPP-Recurrent-Episodic-Pustular-Eruptions-PATHOGNOMONIC -- "
            "IL-36Ra-ABSENT-IL-36A-B-G-Unopposed-Skin-Keratinocyte-IL-8-IL-6-Neutrophil-Influx -- "
            "SPESOLIMAB-Anti-IL-36R-FDA2022-FIRST-APPROVED-TARGETED-THERAPY-GPP)"
        ),
        "protein": (
            "IL36RN -- 2q14.1 AR -- IL36RN-155aa -- "
            "IL-36-Receptor-Antagonist-IL-36Ra-17kDa-IL-1-Superfamily-Member -- "
            "Competitive-Inhibitor-Binds-IL-36R-IL1RL2-Without-Signalling -- "
            "Blocks-IL-36A-IL-36B-IL-36G-All-Three-IL-36-Agonists -- "
            "IL-36-Agonists-Signal-Through-IL-36R-IL-1RAcP-Heterodimer-In-Keratinocytes -- "
            "IL-36Ra-Absent-Unopposed-IL-36-Signalling-Skin-Keratinocytes -- "
            "Keratinocyte-IL-8-CXCL8-IL-6-Release-Neutrophil-Influx-Subcorneal-Pustules -- "
            "p.Ser113Leu-Founder-Variant-Mediterranean-Asian-Enriched -- "
            "OMIM-Gene-605507-Disease-DITRA-614204"
        ),
        "locus": "2q14.1",
        "protein_size": "155 aa / 17 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic; "
            "p.Ser113Leu founder variant Mediterranean (especially Tunisia) and East Asian enriched; "
            "heterozygous variants may modify psoriasis severity but biallelic required for DITRA/GPP"
        ),
        "disease_category": "DITRA -- Deficiency of IL-36 Receptor Antagonist (Generalized Pustular Psoriasis)",
        "disease_pathway": (
            "IL-36Ra is the physiological antagonist of IL-36alpha, IL-36beta, and IL-36gamma in the skin. "
            "IL-36Ra absent -> unopposed IL-36 signalling in keratinocytes -> "
            "NF-kB activation -> IL-8 (CXCL8), IL-6, TNF release from keratinocytes -> "
            "massive neutrophil influx into epidermis -> subcorneal pustule formation. "
            "Recurrent/episodic pattern driven by superimposed triggers (infection, drugs, pregnancy, withdrawal)."
        ),
        "pathognomonic": (
            "GENERALISED PUSTULAR PSORIASIS (GPP): widespread coalescent sterile pustules on erythematous skin -- PATHOGNOMONIC. "
            "Systemic: high fever, leukocytosis, elevated CRP during flares. "
            "Triggers: infections, menstruation, pregnancy (impetigo herpetiformis), drug withdrawal (steroids, retinoids). "
            "Biopsy: Kogoj spongiform pustules in epidermis (neutrophil accumulation). "
            "SPESOLIMAB response confirms IL-36 pathway involvement."
        ),
        "treatment": (
            "SPESOLIMAB (anti-IL-36R, FDA August 2022): 900 mg IV single dose for GPP flare -- first approved targeted therapy. "
            "Maintenance: spesolimab 300 mg SC every 4 weeks. "
            "Acitretin (retinoid): maintenance therapy. "
            "Cyclosporin: rapid flare control (bridging). "
            "Methotrexate: maintenance option. "
            "Avoid systemic corticosteroids (rebound flare on withdrawal). "
            "Infection triggers: identify and treat proactively."
        ),
        "key_features": [
            "Generalized pustular psoriasis (GPP) PATHOGNOMONIC",
            "IL-36Ra absent -> IL-36alpha/beta/gamma unopposed in keratinocytes",
            "Triggers: infection, pregnancy, menstruation, drug withdrawal",
            "p.Ser113Leu founder variant Mediterranean/Asian",
            "Spesolimab (anti-IL-36R, FDA 2022) -- first approved targeted therapy",
            "Avoid systemic corticosteroids (withdrawal rebound)",
        ],
        "key_ddx": (
            "Plaque psoriasis: no pustules; no systemic fever; chronic stable; HLA-Cw6; different treatment. "
            "Pustular drug reactions: drug history; resolves with drug withdrawal; no IL36RN variant. "
            "DIRA (IL1RN): neonatal; bone + periostitis + skin pustulosis; IL-1 not IL-36 pathway. "
            "Acute generalised exanthematous pustulosis (AGEP): drug-induced; spontaneously resolves; patch test positive."
        ),
        "amyloid_risk": "Low -- not a major complication of DITRA",
        "colchicine_response": "Not effective",
        "il1_response": "Partial -- IL-1 pathway less relevant; IL-36 blockade (spesolimab) preferred",
        "hsct_required": "No",
        "attack_trigger_common": "Infection, menstruation, pregnancy, systemic corticosteroid withdrawal, stress",
        "onset_age": "Adolescence or adulthood; paediatric cases reported; pregnancy can trigger first episode",
    },
]

PATIENTS_PER_GENE = 40


def _make_cohort(gene_entry, seed):
    rng = random.Random(seed)
    gene = gene_entry["gene"]
    patients = []
    for i in range(PATIENTS_PER_GENE):
        age_at_dx = round(rng.uniform(0.1, 40.0), 1)
        attack_duration = round(rng.uniform(0.5, 21.0), 1)
        fever_peak = round(rng.uniform(38.2, 41.5), 1)

        # gene-specific amyloid risk
        if gene == "MEFV":
            amyloid_risk = rng.random() < 0.18
        elif gene == "TNFRSF1A":
            amyloid_risk = rng.random() < 0.25
        elif gene in ("NLRP3", "MVK"):
            amyloid_risk = rng.random() < 0.06
        else:
            amyloid_risk = rng.random() < 0.02

        # attack trigger
        if gene == "NLRP3":
            attack_trigger = rng.choice(["cold", "cold", "cold", "spontaneous", "infection"])
        elif gene == "MVK":
            attack_trigger = rng.choice(["vaccination", "vaccination", "vaccination", "infection", "stress"])
        elif gene == "MEFV":
            attack_trigger = rng.choice(["stress", "infection", "cold", "spontaneous", "stress"])
        elif gene in ("IL1RN", "NOD2"):
            attack_trigger = "spontaneous"
        elif gene == "IL36RN":
            attack_trigger = rng.choice(["infection", "stress", "spontaneous", "vaccination"])
        elif gene == "PSTPIP1":
            attack_trigger = rng.choice(["stress", "spontaneous", "stress", "infection"])
        else:
            attack_trigger = rng.choice(["infection", "stress", "spontaneous", "cold"])

        # colchicine response (meaningful for MEFV only)
        if gene == "MEFV":
            colchicine_response = rng.choice(["full", "full", "full", "full", "partial", "none"])
        else:
            colchicine_response = "not-applicable"

        # IL-1 response
        if gene in ("MEFV", "TNFRSF1A", "NLRP3", "MVK", "PSTPIP1", "IL1RN"):
            il1_response = rng.random() < 0.85
        elif gene == "NOD2":
            il1_response = rng.random() < 0.50
        elif gene == "IL36RN":
            il1_response = rng.random() < 0.30
        else:
            il1_response = False

        # HSCT required (only severe mevalonic aciduria end of MVK spectrum)
        hsct_required = gene == "MVK" and rng.random() < 0.03

        # outcome
        if gene == "MEFV":
            outcome = rng.choice([
                "colchicine controlled", "colchicine controlled", "colchicine controlled",
                "partial response", "amyloid complication", "remission on biologics",
            ])
        elif gene == "NLRP3":
            outcome = rng.choice([
                "remission on biologics", "remission on biologics", "remission on biologics",
                "partial response", "remission on biologics",
            ])
        elif gene == "IL1RN":
            outcome = rng.choice([
                "remission on biologics", "remission on biologics", "remission on biologics",
                "remission on biologics", "partial response",
            ])
        else:
            outcome = rng.choice([
                "remission on biologics", "remission on biologics",
                "partial response", "colchicine controlled", "amyloid complication", "refractory",
            ])

        patients.append({
            "patient_id": f"{gene}-{seed}-{i + 1:03d}",
            "age_at_diagnosis_years": age_at_dx,
            "attack_duration_days": attack_duration,
            "fever_peak_celsius": fever_peak,
            "amyloid_risk": amyloid_risk,
            "attack_trigger": attack_trigger,
            "colchicine_response": colchicine_response,
            "il1_response": il1_response,
            "hsct_required": hsct_required,
            "outcome": outcome,
            "gene": gene,
        })
    return patients


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(AUTOINFLAMMATORY_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))

    total = len(all_patients)
    amyloid_count = sum(1 for p in all_patients if p["amyloid_risk"])
    il1_response_count = sum(1 for p in all_patients if p["il1_response"])
    avg_attack_duration = sum(p["attack_duration_days"] for p in all_patients) / total
    vaccination_trigger = sum(1 for p in all_patients if p["attack_trigger"] == "vaccination")
    cold_trigger = sum(1 for p in all_patients if p["attack_trigger"] == "cold")
    remission_biologics = sum(1 for p in all_patients if "remission" in p["outcome"])

    gene_summary = {}
    for idx, entry in enumerate(AUTOINFLAMMATORY_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "n_patients": len(cohort),
            "avg_attack_duration_days": round(
                sum(p["attack_duration_days"] for p in cohort) / len(cohort), 1
            ),
            "amyloid_risk_pct": round(100 * sum(1 for p in cohort if p["amyloid_risk"]) / len(cohort), 1),
            "il1_response_pct": round(100 * sum(1 for p in cohort if p["il1_response"]) / len(cohort), 1),
            "vaccination_trigger_pct": round(
                100 * sum(1 for p in cohort if p["attack_trigger"] == "vaccination") / len(cohort), 1
            ),
            "cold_trigger_pct": round(
                100 * sum(1 for p in cohort if p["attack_trigger"] == "cold") / len(cohort), 1
            ),
            "remission_biologics_pct": round(
                100 * sum(1 for p in cohort if "remission" in p["outcome"]) / len(cohort), 1
            ),
        }

    return {
        "atlas": "Hereditary-Autoinflammatory-Syndrome-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Autoinflammatory Disease Reference -- FMF/TRAPS/CAPS/MKD/PAPA/Blau/DIRA/DITRA",
        "genes_covered": [e["gene"] for e in AUTOINFLAMMATORY_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "avg_attack_duration_days": round(avg_attack_duration, 1),
            "amyloid_risk_pct": round(100 * amyloid_count / total, 1),
            "il1_response_pct": round(100 * il1_response_count / total, 1),
            "vaccination_trigger_pct": round(100 * vaccination_trigger / total, 1),
            "cold_trigger_pct": round(100 * cold_trigger / total, 1),
            "remission_on_biologics_pct": round(100 * remission_biologics / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(AUTOINFLAMMATORY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "amyloid_risk": entry["amyloid_risk"],
            "colchicine_response": entry["colchicine_response"],
            "il1_response": entry["il1_response"],
            "hsct_required": entry["hsct_required"],
            "attack_trigger_common": entry["attack_trigger_common"],
            "onset_age": entry["onset_age"],
            "n_patients": len(cohort),
            "avg_attack_duration_days": round(
                sum(p["attack_duration_days"] for p in cohort) / len(cohort), 1
            ),
            "avg_fever_peak_celsius": round(
                sum(p["fever_peak_celsius"] for p in cohort) / len(cohort), 1
            ),
            "amyloid_risk_pct": round(
                100 * sum(1 for p in cohort if p["amyloid_risk"]) / len(cohort), 1
            ),
            "il1_response_pct": round(
                100 * sum(1 for p in cohort if p["il1_response"]) / len(cohort), 1
            ),
            "vaccination_trigger_pct": round(
                100 * sum(1 for p in cohort if p["attack_trigger"] == "vaccination") / len(cohort), 1
            ),
            "cold_trigger_pct": round(
                100 * sum(1 for p in cohort if p["attack_trigger"] == "cold") / len(cohort), 1
            ),
            "remission_biologics_pct": round(
                100 * sum(1 for p in cohort if "remission" in p["outcome"]) / len(cohort), 1
            ),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:400],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "amyloid_risk": entry["amyloid_risk"],
                "colchicine_response": entry["colchicine_response"],
                "il1_response": entry["il1_response"],
                "hsct_required": entry["hsct_required"],
                "attack_trigger_common": entry["attack_trigger_common"],
                "onset_age": entry["onset_age"],
            }
            for entry in AUTOINFLAMMATORY_GENES
        },
        "autoinflammatory_glossary": {
            "Hereditary Autoinflammatory Diseases (AID)": (
                "A group of monogenic disorders of innate immunity characterised by recurrent or chronic sterile inflammation "
                "without autoantibodies or antigen-specific T cells. Pathomechanism: dysregulated IL-1beta (most common), "
                "TNF, IL-36, or NF-kB signalling from mutations in inflammasome components, cytokine receptors, or "
                "cytokine antagonists. All are distinguished from autoimmune disease by the absence of adaptive immune hallmarks."
            ),
            "MEFV/FMF and Colchicine": (
                "Colchicine (1-2 mg/day) is the cornerstone of FMF management -- both attack prophylaxis and AA amyloidosis prevention. "
                "Mechanism: inhibits microtubule polymerisation -> impairs neutrophil degranulation and NLRP3 pyrin inflammasome assembly. "
                "90-95% attack reduction in adherent patients. Colchicine resistance (5-10%): biallelic M694V, poor adherence -- "
                "switch to IL-1 blockade (anakinra/canakinumab). Lifelong treatment -- stopping colchicine restores amyloid risk."
            ),
            "CAPS Disease Spectrum (FCAS to MWS to NOMID)": (
                "FCAS (Familial Cold Autoinflammatory Syndrome): cold-triggered urticaria + fever + arthralgia -- mildest. "
                "MWS (Muckle-Wells Syndrome): urticaria + fever + sensorineural hearing loss + amyloid risk -- intermediate. "
                "NOMID (Neonatal-Onset Multisystem Inflammatory Disease / CINCA): neonatal onset, chronic aseptic meningitis, "
                "destructive arthropathy, optic disc oedema, intellectual disability -- most severe. "
                "All three caused by NLRP3 GOF; genotype-phenotype correlation exists but not absolute."
            ),
            "IL-1 Blockade in Autoinflammatory Diseases": (
                "Anakinra (IL-1Ra): recombinant IL-1 receptor antagonist; daily SC injection; rapid onset (hours-days); "
                "first-line for DIRA (curative), FMF-resistant, TRAPS, CAPS, MKD, PAPA. "
                "Canakinumab (anti-IL-1beta): q8w SC; long-acting; FDA-approved for CAPS, FMF, TRAPS, MKD. "
                "Rilonacept (IL-1 trap): q1w SC; FDA-approved for CAPS. "
                "IL-1 blockade dramatically effective in IL-1-driven AIDs -- a dramatic response supports the diagnosis."
            ),
            "AA Amyloidosis in Autoinflammatory Disease": (
                "Complication of chronic/recurrent serum amyloid A (SAA) elevation -> "
                "SAA misfolded -> amyloid A fibrils deposit in kidneys (dominant), spleen, liver, adrenals. "
                "Clinical: proteinuria -> nephrotic syndrome -> end-stage renal disease. "
                "Risk ranking: TRAPS > FMF (M694V homozygotes) > MWS > others. "
                "Prevention: suppress inflammation (colchicine in FMF; IL-1 blockade in TRAPS/CAPS); "
                "target SAA <10 mg/L."
            ),
            "DITRA and Spesolimab (FDA 2022)": (
                "Generalised pustular psoriasis (GPP) from IL36RN LOF was the rationale for developing IL-36 pathway blockade. "
                "Spesolimab (anti-IL-36 receptor): 900 mg IV for acute GPP flare -- FDA approved August 2022 (first GPP-targeted drug). "
                "Maintenance: 300 mg SC q4w. Mechanism: blocks all three IL-36 agonists (IL-36alpha/beta/gamma) from activating keratinocytes. "
                "Transformed GPP management -- previously cyclosporin/acitretin only, high hospitalisation burden."
            ),
            "DIRA (IL1RN) -- Neonatal Emergency": (
                "DIRA is a medical emergency in neonates: multifocal osteomyelitis + periostitis + pustulosis without sepsis organisms. "
                "Key diagnostic step: exclude infection (blood/CSF/bone culture) then start anakinra empirically in a neonate with "
                "culture-negative multifocal bone disease + pustulosis. IL1RN sequencing/deletion FISH confirms. "
                "Puerto Rican and Dutch founders: test in neonates from these populations with compatible phenotype. "
                "Anakinra withdrawal -> immediate relapse -> mandatory lifelong continuation."
            ),
            "Blau Syndrome NOD2 GOF vs Crohn's Disease NOD2 LOF": (
                "NOD2 is located at 16q12.1; variants in the same gene cause opposite predispositions. "
                "Blau Syndrome: gain-of-function (GOF) variants in the NACHT domain -> constitutive NF-kB -> sterile granulomatous disease. "
                "Crohn's disease susceptibility: loss-of-function (LOF) variants in the LRR domain -> impaired MDP sensing -> "
                "defective mucosal immunity -> gut inflammation. "
                "Clinically opposite: Blau = no gut, yes granulomatous uveitis/arthritis; Crohn's = gut, no uveitis triad. "
                "Variant type (GOF vs LOF + domain) distinguishes them definitively."
            ),
            "PAPA Syndrome PSTPIP1 Pathomechanism": (
                "PSTPIP1 acts as a molecular scaffold linking actin cytoskeletal dynamics to pyrin regulation in neutrophils. "
                "PAPA variants: pathological hyperphosphorylation of PSTPIP1 -> cannot bind pyrin -> pyrin loses "
                "NLRP3 suppressive function -> sterile neutrophilic inflammation in joints (pyogenic arthritis) and skin. "
                "Key clinical point: joint fluid is sterile (culture-negative) but looks like septic arthritis -- "
                "must send for culture before starting antibiotics to avoid misdiagnosis. "
                "IL-1 blockade (anakinra) addresses root pathomechanism most directly."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (gene 0 only) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first key) ===")
    defn = generate_definitions()
    first_key = next(iter(defn["gene_entries"]))
    print(json.dumps(defn["gene_entries"][first_key], indent=2)[:1500])
