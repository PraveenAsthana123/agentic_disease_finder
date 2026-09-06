#!/usr/bin/env python3
"""Hereditary-Amyloidosis-Atlas — Complete 8-Gene Hereditary Systemic Amyloidosis Atlas
TTR      (Transthyretin; 147 aa; 18q12.1; AD;
          ATTRv — hereditary transthyretin amyloidosis; FAP/FAC; Val30Met most common;
          patisiran/inotersen/vutrisiran/tafamidis approved;
          seed SEED_BASE+0) ·
APOA1    (Apolipoprotein A-I; 267 aa; 11q23.3; AD missense GOF;
          Hereditary APOA1 amyloidosis — renal/hepatic/neuropathic; Leu75Pro, Arg173Pro;
          seed SEED_BASE+1) ·
APOA2    (Apolipoprotein A-II; 100 aa; 1q23.3; AD stop-gain → C-terminal extension;
          Hereditary APOA2 amyloidosis — renal dominant; Stop78Arg/Cys/Tyr late-onset;
          seed SEED_BASE+2) ·
FGA      (Fibrinogen Aα-chain; 866 aa; 4q31.3; AD missense exon-5;
          Hereditary Fibrinogen Aα-chain amyloidosis (Ostertag type) — renal dominant; E526V;
          liver transplant CURATIVE;
          seed SEED_BASE+3) ·
LYZ      (Lysozyme C; 148 aa; 12q15; AD destabilising missense;
          Hereditary Lysozyme amyloidosis — hepatic rupture risk; renal; I56T/W64R;
          seed SEED_BASE+4) ·
CST3     (Cystatin C; 146 aa; 20p11.21; AD;
          Hereditary Cystatin C Amyloid Angiopathy (HCCAA Iceland) — L68Q; cerebral amyloid
          angiopathy; young-onset stroke/haemorrhage (20s–30s); PATHOGNOMONIC low CSF cystatin C;
          seed SEED_BASE+5) ·
GSN      (Gelsolin; 782 aa; 9q33.2; AD;
          Hereditary Gelsolin amyloidosis (Finnish type / Meretoja disease) — G654A;
          corneal lattice dystrophy type II PATHOGNOMONIC + facial palsy + cutis laxa;
          seed SEED_BASE+6) ·
B2M      (Beta-2 microglobulin; 119 aa; 15q21.1; AD rare hereditary form;
          Hereditary B2M amyloidosis (Asp76Asn/D76N) — systemic; distinct from dialysis-related;
          cardiomyopathy + hepatic + renal; mass spectrometry mandatory for typing;
          seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1622–1629)
"""

import random

SEED_BASE = 1622

AMYLOIDOSIS_GENES = [
    # ── TTR — ATTRv hereditary transthyretin amyloidosis ─────────────────────
    {
        "gene": "TTR",
        "protein": "TTR — ATTRv AD — Transthyretin Tetramer Misfolding — Val30Met Most Common — Patisiran/Vutrisiran/Tafamidis Approved",
        "alias": (
            "TTR; OMIM gene 176300; Familial Amyloid Polyneuropathy type I OMIM 105210 (Val30Met); "
            "Familial Amyloid Cardiomyopathy OMIM 105210 (Val122Ile); "
            "18q12.1; 147 aa mature (preprotein 176 aa with 29-aa signal peptide); "
            "~55 kDa homo-tetramer (4 × 13.8 kDa subunits); AD — gain of amyloidogenic function. "
            "TTR encodes transthyretin (formerly prealbumin), a retinol-binding protein carrier and "
            "thyroxine (T4) transporter in plasma and CSF. SYNTHESIS: 90% liver; 10% choroid plexus "
            "(for CSF) + retinal pigment epithelium. TETRAMER STRUCTURE: homo-tetramer with two T4 "
            "binding sites at the dimer interface; thermodynamic stability of the native tetramer "
            "prevents misfolding — pathogenic variants destabilise tetramer → monomer → misfolded "
            "monomer → amyloid fibril. "
            "KEY PATHOGENIC VARIANTS (>140 known): "
            "Val30Met (p.Val50Met new nomenclature) — most common worldwide; Portuguese (2,000/100k), "
            "Japanese, Swedish (late-onset phenotype); ATTR-FAP type I — length-dependent "
            "polyneuropathy + autonomic + cardiac. "
            "Val122Ile — 3.5% of African Americans carry heterozygous Val122Ile; predominantly "
            "CARDIAC (ATTR-FAC); late onset (60–70yo); most common hereditary cause of ATTR "
            "cardiomyopathy in African Americans. "
            "Ile68Leu — African Americans; cardiac. "
            "Thr60Ala — Irish descent; cardiac dominant. "
            "His90Asn — cardiac dominant, widely distributed. "
            "Phe64Leu — late onset, cardiac. "
            "CLINICAL SYNDROMES: "
            "ATTRv-FAP (Familial Amyloid Polyneuropathy): length-dependent peripheral sensorimotor "
            "neuropathy (feet first → hands); AUTONOMIC NEUROPATHY — erectile dysfunction (first sign "
            "in men), orthostatic hypotension, constipation/diarrhea alternating, bladder dysfunction; "
            "CARPAL TUNNEL SYNDROME — often the first manifestation (5–10 years before neuropathy); "
            "vitreous opacities (amyloid in vitreous); cardiac involvement (restrictive cardiomyopathy). "
            "ATTRv-FAC (Familial Amyloid Cardiomyopathy): HCM-like appearance on echo; "
            "restrictive physiology; HFpEF; conduction disease (AV block, bundle branch block); "
            "SPARKLING/granular myocardium on echo; significantly elevated NT-proBNP/BNP; "
            "thickened interventricular septum (IVS); preserved LV ejection fraction initially. "
            "INVESTIGATION: 99mTc-DPD or 99mTc-PYP bone scintigraphy — PATHOGNOMONIC for ATTR "
            "cardiomyopathy: Grade 2 (equal to bone) or Grade 3 (greater than bone) cardiac uptake "
            "with NEGATIVE serum/urine protein electrophoresis + immunofixation = ATTR-CM without "
            "biopsy; Grade 1 or 0 does not exclude ATTR-CM. "
            "EMG/NCS: reduced CMAP/SNAP amplitudes in dying-back polyneuropathy pattern. "
            "Genetic testing: TTR gene sequencing (and large deletion panel if negative). "
            "TREATMENT: "
            "PATISIRAN (Onpattro) — siRNA, IV every 3 weeks; reduces hepatic TTR synthesis by 80%; "
            "approved FAP stages 1-2 (APOLLO trial 2018); premedication required. "
            "INOTERSEN (Tegsedi) — ASO, SC weekly; reduces hepatic TTR by 70%; "
            "approved FAP 2018; REMS required for thrombocytopenia + glomerulonephritis monitoring "
            "(platelet monitoring mandatory every 2 weeks). "
            "VUTRISIRAN (Amvuttra) — siRNA, SC quarterly; reduces TTR by 87%; approved 2022 "
            "(HELIOS-A trial); superior safety/convenience to patisiran; no premedication needed. "
            "EPLONTERSEN (Wainua) — ASO, SC monthly; ~80% TTR reduction; approved 2023. "
            "TAFAMIDIS (Vyndaqel/Vyndamax) — TTR tetramer stabiliser; binds T4 site → "
            "prevents tetramer dissociation; approved ATTR-CM (ATTR-ACT trial 2019); "
            "4.2 g (Vyndamax) once daily oral; reduces cardiovascular mortality/hospitalisation. "
            "DIFLUNISAL — NSAID with off-label TTR stabiliser activity; 250 mg BD; "
            "gastroprotection mandatory; avoid in CKD/heart failure; less potent than tafamidis. "
            "LIVER TRANSPLANT: historically curative for FAP (replaces mutant TTR-producing liver); "
            "cardiac progression can continue (choroid plexus still produces mutant TTR; "
            "WT-ATTR deposits on mutant fibril seeds in heart); now largely superseded by "
            "pharmacotherapy but still used in selected young patients with severe neuropathy."
        ),
        "aa": "147 aa",
        "kDa": "~55 kDa (homo-tetramer)",
        "locus": "18q12.1",
        "omim_gene": 176300,
        "omim_disease": 105210,
        "inheritance": "AD; >140 pathogenic variants; Val30Met most common; Val122Ile 3.5% African American population",
        "gene_class": (
            "TTR encodes transthyretin, a small homo-tetrameric transport protein. "
            "STRUCTURE: each 127-aa monomer folds into a β-sandwich (8 antiparallel β-strands); "
            "two monomers form a dimer via β-strand hydrogen bonding; "
            "two dimers assemble head-to-tail to form the functional tetramer. "
            "FUNCTION: THYROXINE (T4) TRANSPORT: TTR is one of three T4-binding proteins in plasma "
            "(with TBG and albumin); binds 2 T4 molecules at the central channel of the tetramer; "
            "in CSF, TTR is the primary T4 carrier. "
            "RETINOL TRANSPORT: TTR binds retinol-binding protein (RBP4) → retinol delivery to tissues. "
            "AMYLOIDOGENIC MECHANISM: rate-limiting step is TETRAMER DISSOCIATION → monomers → "
            "partially unfolded amyloidogenic intermediate → oligomers → amyloid fibrils. "
            "Pathogenic missense variants (Val30Met etc.) shift the equilibrium toward the monomeric "
            "intermediate by reducing tetramer thermodynamic stability. "
            "TETRAMER STABILISERS (tafamidis, diflunisal, acoramidis) work by binding to the T4 "
            "binding sites and locking the tetramer in the native conformation → kinetic stabilisation. "
            "LIVER IS THE PRIMARY SOURCE of circulating TTR (90%); liver transplant removes the "
            "main amyloidogenic TTR production; choroid plexus production (10%) can allow cardiac "
            "deposition to continue using seeded fibril templates."
        ),
        "n_patients": 40,
        "key_alerts": [
            "TTR-DPD-PYP-SCAN-PATHOGNOMONIC-CARDIAC: Grade 2 (cardiac uptake equal to ribs) or Grade 3 (greater than ribs) on 99mTc-DPD or 99mTc-PYP bone scintigraphy combined with NEGATIVE serum and urine protein electrophoresis and immunofixation = ATTR cardiomyopathy diagnosis WITHOUT biopsy; this algorithm is 98% specific; do NOT diagnose by scan alone without excluding AL amyloidosis — light chains must be excluded first",
            "TTR-CARPAL-TUNNEL-EARLY-SIGN: Bilateral carpal tunnel syndrome (CTS) is the earliest and most common manifestation of ATTRv — it often precedes the polyneuropathy or cardiomyopathy by 5–10 years; any patient with bilateral CTS + family history of neuropathy/cardiomyopathy should have TTR genetic testing; surgical CTS release does not prevent systemic progression",
            "TTR-INOTERSEN-REMS-THROMBOCYTOPENIA: Inotersen (Tegsedi) carries a REMS program for thrombocytopenia (can be severe, including fatal) and glomerulonephritis; platelet count must be checked every 2 weeks; if platelets <100,000 → dose reduction; if <75,000 → discontinue; patients must be enrolled in the REMS before treatment initiation — prescribers must be certified",
            "TTR-VAL122ILE-AFRICAN-AMERICAN: Val122Ile is carried by ~3.5% of African Americans (1 in 29 individuals) — the most common hereditary ATTR variant worldwide by carrier frequency; causes predominantly cardiac ATTR-CM presenting in 60s–70s; massively underdiagnosed; any African American with HFpEF + thickened LV wall + elevated NT-proBNP should have ATTR evaluation including genetic testing",
            "TTR-AUTONOMIC-NEUROPATHY-FIRST: Autonomic neuropathy — erectile dysfunction in men, orthostatic hypotension, alternating constipation/diarrhea, sweating abnormalities, bladder dysfunction — often precedes or coexists with sensorimotor neuropathy in ATTRv-FAP; autonomic symptoms in a younger patient with a family history of neuropathy demand TTR evaluation; patients are often treated for IBS or diabetic autonomic neuropathy for years before ATTRv is diagnosed",
            "TTR-VITREOUS-OPACITIES-PATHOGNOMONIC: Amyloid deposits in the vitreous body of the eye produce vitreous opacities (floaters, blurred vision) that are PATHOGNOMONIC for TTR amyloidosis; they occur because TTR is produced locally by the retinal pigment epithelium; vitreoretinal surgery may be needed but does not stop systemic progression; ophthalmological evaluation is part of standard ATTRv workup",
            "TTR-LIVER-TRANSPLANT-CARDIAC-PROGRESSION: Liver transplant eliminates 90% of circulating mutant TTR and arrests/reverses FAP in many patients; however, cardiac amyloid can CONTINUE TO PROGRESS post-transplant because: (1) choroid plexus still produces mutant TTR; (2) WT-TTR from the new liver deposits on pre-existing mutant fibril seeds in the heart; combined liver-heart transplant or liver transplant + tafamidis is used in selected patients with cardiac involvement",
            "TTR-TAFAMIDIS-NOT-FOR-FAP: Tafamidis (Vyndaqel) is approved specifically for ATTR CARDIOMYOPATHY — it slowed cardiovascular mortality in the ATTR-ACT trial but has NOT been proven to improve polyneuropathy; do NOT use tafamidis as primary treatment for ATTRv-FAP; use vutrisiran, patisiran, or inotersen for neuropathy; tafamidis is adjunctive if cardiac involvement coexists with FAP",
        ],
        "etiologies": {
            "Val30Met (p.Val50Met) — FAP type I; Portuguese/Japanese/Swedish": 14,
            "Val122Ile — ATTR cardiomyopathy; African American predominance": 9,
            "Thr60Ala — cardiac dominant; Irish descent": 5,
            "His90Asn — cardiac dominant; worldwide": 4,
            "Ile68Leu — cardiac; African American": 4,
            "Other rare variants (>140 known)": 4,
        },
        "stats": {
            "mean_dx_age_y": 52.4,
            "mean_dx_delay_months": 48.0,
            "pct_carpal_tunnel_first_sign": 68,
            "pct_autonomic_neuropathy": 72,
            "pct_cardiac_involvement": 55,
            "pct_vitreous_opacities": 20,
            "pct_misdiagnosed_cmt_or_cidp": 42,
            "pct_treated_before_diagnosis": 38,
        },
        "dx_delay_distribution": {"<1 y": 4, "1–5 y": 16, "5–15 y": 14, ">15 y": 6},
    },
    # ── APOA1 — Hereditary APOA1 Amyloidosis ────────────────────────────────
    {
        "gene": "APOA1",
        "protein": "APOA1 — Hereditary APOA1 Amyloidosis AD — Low HDL PATHOGNOMONIC — Renal/Hepatic/Neuropathic — Liver Transplant",
        "alias": (
            "APOA1; OMIM gene 107680; Hereditary APOA1 amyloidosis OMIM 107680 (allelic); "
            "11q23.3; 267 aa mature (preprotein 243 aa + 24-aa propeptide + 18-aa signal peptide = "
            "267 aa total prepropeptide); ~29 kDa monomer; AD missense gain-of-amyloidogenic-function. "
            "APOA1 encodes Apolipoprotein A-I, the major structural protein of HDL (high-density "
            "lipoprotein). FUNCTION: reverse cholesterol transport — accepts cholesterol from "
            "peripheral tissues and delivers to liver for excretion; activates LCAT "
            "(lecithin-cholesterol acyltransferase) which esterifies free cholesterol on HDL. "
            "STRUCTURE: mostly α-helical; 10 amphipathic helices (helix 1-10) wrap around lipid "
            "core of HDL particle; when lipid-free (apoA-I in plasma), loosely ordered. "
            "AMYLOIDOGENIC VARIANTS: pathogenic missense mutations cluster in helix 1-2 (N-terminal "
            "region, residues 26–75 segment) and helix 5-7 (C-terminal helices, residues 165–180): "
            "N-terminal variants (helix 1-2): Leu75Pro, Gly26Arg, Trp50Arg, Leu90Pro — "
            "deposits as the N-terminal fragment (residues 1–75 or 1–83 depending on variant) "
            "predominantly in KIDNEY > liver; "
            "C-terminal variants (helix 5-7): Arg173Pro, Ala175Pro, Leu178His, Lys107del — "
            "deposits as C-terminal fragment predominantly in LIVER > kidney + neuropathy. "
            "LABORATORY: HDL CHOLESTEROL VERY LOW (hallmark); total cholesterol low-normal; "
            "LDL normal; apoA-I protein level low (amyloidogenic variant has reduced HDL-binding). "
            "RENAL INVOLVEMENT: proteinuria → nephrotic syndrome → progressive CKD → ESRD; "
            "N-terminal variants more kidney-dominant. "
            "HEPATIC INVOLVEMENT: hepatomegaly; liver dysfunction; C-terminal variants more liver-dominant. "
            "NEUROPATHY: C-terminal variants — peripheral neuropathy (less severe than TTR-FAP); "
            "some skin and adrenal involvement. "
            "DIAGNOSIS: Serum Congo red-positive deposits (fat pad biopsy); renal biopsy; "
            "MASS SPECTROMETRY — essential for typing (anti-apoA-I IHC on tissue; proteomics confirms apoA-I); "
            "apoA-I gene sequencing; low HDL is the biochemical clue. "
            "TREATMENT: Liver transplant removes the mutant apoA-I source (liver produces ~70% of apoA-I); "
            "renal transplant for ESRD; no approved pharmacotherapy; high-dose statins or niacin do not "
            "correct the amyloidogenic protein; recombinant apoA-I or gene therapy investigational."
        ),
        "aa": "267 aa",
        "kDa": "~29 kDa",
        "locus": "11q23.3",
        "omim_gene": 107680,
        "omim_disease": 107680,
        "inheritance": "AD missense; multiple variants (Leu75Pro, Gly26Arg, Arg173Pro most common); strong genotype-phenotype (N-terminal vs C-terminal variants differ in organ tropism)",
        "gene_class": (
            "APOA1 encodes Apolipoprotein A-I, the dominant scaffolding protein of HDL particles. "
            "STRUCTURAL BIOLOGY: lipid-free apoA-I exists in a molten-globule state with 10 "
            "amphipathic α-helices; when it binds lipid (on HDL surface), it adopts a belt/horseshoe "
            "conformation wrapping around the HDL disc. "
            "AMYLOIDOGENESIS MECHANISM: pathogenic missense mutations in helix 1-2 or helix 5-7 "
            "reduce thermal stability of the corresponding domain → partial unfolding → "
            "aggregation-prone intermediate → amyloid fibrils. Specific fragmentation by proteases "
            "releases a defined fragment (e.g. residues 1–75 for N-terminal variants) that deposits "
            "as amyloid. Different fragments have different organ tropisms (kidney for N-terminal, "
            "liver/nerves for C-terminal). "
            "HDL CONSEQUENCE: mutant apoA-I cannot efficiently form functional HDL → "
            "very low HDL cholesterol (often <0.5 mmol/L) → LCAT activity low → "
            "impaired reverse cholesterol transport (cardiovascular risk, though atherosclerosis "
            "is often not the primary clinical problem in hereditary amyloidosis). "
            "NOTE: inherited APOA1 mutations causing amyloidosis are distinct from APOA1 mutations "
            "causing isolated low HDL (e.g. Tangier disease involves ABCA1, not APOA1); "
            "the hereditary amyloidosis mutations create amyloidogenic variants, not simply "
            "unstable proteins that are degraded."
        ),
        "n_patients": 40,
        "key_alerts": [
            "APOA1-LOW-HDL-PATHOGNOMONIC-CLUE: Very low HDL cholesterol (<0.5 mmol/L) is the key biochemical pointer to hereditary APOA1 amyloidosis; in a patient with progressive proteinuria/renal failure or hepatomegaly and very low HDL with normal LDL, APOA1 amyloidosis should be in the differential; test apoA-I level and genotype; standard amyloid typing (SAA, AL, TTR) will be negative",
            "APOA1-ORGAN-TROPISM-GENOTYPE: The pattern of organ involvement (kidney vs liver vs neuropathy) is predicted by the variant location; N-terminal variants (residues 1–90, Leu75Pro, Gly26Arg) cause predominantly renal amyloidosis; C-terminal variants (residues 165–180, Arg173Pro, Ala175Pro) cause predominantly hepatic + neuropathic amyloidosis; genotyping guides surveillance priorities",
            "APOA1-MASS-SPECTROMETRY-MANDATORY: Standard immunohistochemistry panels (SAA, TTR, AL kappa/lambda) will NOT identify APOA1 amyloidosis; mass spectrometry-based proteomics of the amyloid extract is required to type APOA1 amyloid — this is available at specialist amyloid centres (Mayo, UCL/UCLH, Heidelberg); referring to a specialist centre for amyloid typing before treatment planning is mandatory",
            "APOA1-LIVER-TRANSPLANT-SOURCE-REMOVAL: Liver produces approximately 70% of circulating apoA-I; liver transplant replaces the mutant apoA-I source → serum apoA-I normalises → amyloid deposition halts → renal function may stabilise or partially recover; timing is critical — transplant before ESRD if possible; combined liver-kidney transplant for established ESRD",
        ],
        "etiologies": {
            "Leu75Pro — helix 1-2; renal dominant; N-terminal fragment": 12,
            "Gly26Arg — helix 1; renal/hepatic; N-terminal fragment": 9,
            "Arg173Pro — helix 7; hepatic/neuropathic; C-terminal fragment": 8,
            "Ala175Pro — helix 7; hepatic + skin; C-terminal fragment": 6,
            "Trp50Arg — helix 2; renal dominant": 3,
            "Other rare APOA1 missense variants": 2,
        },
        "stats": {
            "mean_dx_age_y": 48.6,
            "mean_dx_delay_months": 72.0,
            "pct_renal_presentation": 72,
            "pct_hepatic_involvement": 45,
            "pct_peripheral_neuropathy": 30,
            "pct_very_low_hdl": 95,
            "pct_misdiagnosed_iga_nephropathy": 38,
        },
        "dx_delay_distribution": {"<1 y": 3, "1–5 y": 12, "5–15 y": 18, ">15 y": 7},
    },
    # ── APOA2 — Hereditary APOA2 Amyloidosis ────────────────────────────────
    {
        "gene": "APOA2",
        "protein": "APOA2 — Hereditary APOA2 Amyloidosis AD Stop-Gain — C-Terminal Extension — Renal Dominant — Late-Onset — Renal Transplant",
        "alias": (
            "APOA2; OMIM gene 107670; Hereditary APOA2 amyloidosis OMIM 614357; "
            "1q23.3; 100 aa mature (after signal peptide and propeptide cleavage); "
            "~8.7 kDa monomer (~17 kDa as disulfide-linked homodimer); AD. "
            "APOA2 encodes Apolipoprotein A-II, the second most abundant apolipoprotein of HDL after "
            "APOA1. STRUCTURE: 77-aa mature protein forms a disulfide-linked homodimer via Cys6; "
            "associated with HDL and chylomicrons; modulates LCAT, HL (hepatic lipase), CETP. "
            "UNIQUE AMYLOIDOGENIC MECHANISM: STOP-GAIN (nonsense-to-missense, stop codon extension) — "
            "unlike other hereditary amyloidoses caused by amino acid substitutions within the protein, "
            "APOA2 amyloidosis is caused by variants that REPLACE the normal stop codon with a "
            "sense codon → ribosome continues translation → EXTENDED C-TERMINAL TAIL "
            "(5–21 additional amino acids added beyond normal stop) → extended C-terminus is "
            "amyloidogenic → deposits in kidney. "
            "APOA2 VARIANTS CAUSING AMYLOIDOSIS: "
            "p.Stop78Arg (c.235T>C) — most common; French, Belgian families; "
            "p.Stop78Cys (c.235T>G); "
            "p.Stop78Tyr (c.235T>A); "
            "p.Stop78Leu (c.235T>C / different position); "
            "All variants extend the C-terminus by 5–21 residues that include a Trp residue which "
            "is critical for fibril formation. "
            "PHENOTYPE: Predominantly RENAL AMYLOIDOSIS "
            "(tubulo-interstitial and glomerular deposits) → progressive proteinuria → "
            "nephrotic syndrome → CKD → ESRD. "
            "ONSET: LATE (median age of ESRD ~55–70 years); "
            "slower progression than FGA or LYZ amyloidosis. "
            "HEPATIC: minimal involvement. "
            "CARDIAC: not a major feature. "
            "NEUROPATHY: not a feature. "
            "HDL: low but not as dramatically low as APOA1 amyloidosis. "
            "DIAGNOSIS: Renal biopsy — Congo red positive deposits in glomeruli and interstitium; "
            "MASS SPECTROMETRY MANDATORY for typing (distinguishes from AA, AL, TTR); "
            "anti-apoA-II immunohistochemistry; APOA2 gene sequencing. "
            "TREATMENT: Renal transplant for ESRD; no pharmacotherapy; liver transplant (replaces "
            "mutant apoA-II source) is theoretically curative but less established than for FGA or APOA1; "
            "monitoring: annual urinalysis + creatinine + GFR estimation in all carriers."
        ),
        "aa": "100 aa",
        "kDa": "~8.7 kDa monomer (~17 kDa dimer)",
        "locus": "1q23.3",
        "omim_gene": 107670,
        "omim_disease": 614357,
        "inheritance": "AD; stop-gain (stop codon → sense codon) extending C-terminus; Stop78Arg most common; incomplete penetrance in some families; late onset reduces apparent penetrance",
        "gene_class": (
            "APOA2 encodes Apolipoprotein A-II, a small apolipoprotein with a disulfide-bridged "
            "homodimeric structure. "
            "STRUCTURAL FEATURES: signal peptide (23 aa) + propeptide (5 aa) → cleaved to yield "
            "77-aa mature protein; Cys6 forms a disulfide bridge between the two identical chains "
            "of the homodimer; no tryptophan in normal mature APOA2. "
            "AMYLOIDOGENIC EXTENSION: stop-gain mutations replace the TGA stop codon at position 78 "
            "with a sense codon → ribosome reads through into the 3'UTR → an additional 5–21 residues "
            "are appended; this extension contains a Trp residue critical for fibril formation. "
            "UNIQUE ASPECT: hereditary APOA2 amyloidosis represents one of the few examples where "
            "amyloidogenicity arises from GAIN OF SEQUENCE (C-terminal extension) rather than "
            "loss of function or amino acid substitution; the extended C-terminus is highly "
            "prone to aggregation. "
            "DISTINCTION FROM DIALYSIS-RELATED AMYLOIDOSIS: B2M amyloidosis in dialysis patients "
            "also causes renal amyloidosis but through WT-B2M accumulation due to impaired clearance; "
            "APOA2 amyloidosis occurs in patients with NORMAL or near-normal renal function initially; "
            "proteomics clearly distinguishes the two."
        ),
        "n_patients": 40,
        "key_alerts": [
            "APOA2-STOP-GAIN-UNIQUE-MECHANISM: APOA2 amyloidosis is caused by stop-gain (readthrough) mutations that add extra amino acids at the C-terminus — NOT by substitutions within the coding sequence; genetic testing must include analysis of the stop codon region (codon 78); standard coding sequence panels that stop at the last sense codon will MISS the pathogenic variant; request full APOA2 sequencing including stop codon and flanking 3'UTR",
            "APOA2-LATE-ONSET-LOW-PENETRANCE-APPARENT: The late onset of ESRD (50s–70s) combined with the moderate penetrance means carriers may be unaware of their status; family history may show 'kidney failure in old age' attributed to hypertension or diabetes; genetic testing in the proband + cascade testing of all first-degree relatives is mandatory; carriers need annual urinalysis starting in the 4th decade",
            "APOA2-PROTEOMICS-MANDATORY: Standard amyloid subtyping panels do not include apoA-II; if Congo red-positive renal amyloid is present and SAA, AL, and TTR are excluded, request mass spectrometry proteomics — specialist amyloid centres can type apoA-II amyloid; do not diagnose as 'idiopathic' or 'unclassified' amyloidosis without proteomics",
            "APOA2-RENAL-TRANSPLANT-RECURRENCE-RISK: Renal transplant is the treatment for ESRD but mutant apoA-II continues to circulate from liver → amyloid may recur in the transplanted kidney over years; liver transplant in addition to or instead of renal transplant may be considered to eliminate the amyloidogenic protein source",
        ],
        "etiologies": {
            "Stop78Arg (c.235T>C) — most common; extra Arg-Asn-Trp-Lys-Ala C-terminal tail": 22,
            "Stop78Cys (c.235T>G) — extra Cys-Asn-Trp-Lys-Ala C-terminal tail": 9,
            "Stop78Tyr (c.235T>A) — extra Tyr-Asn-Trp-Lys-Ala C-terminal tail": 6,
            "Stop78Leu — variant at same site": 3,
        },
        "stats": {
            "mean_dx_age_y": 58.2,
            "mean_dx_delay_months": 84.0,
            "pct_renal_presentation": 95,
            "pct_esrd_at_diagnosis": 45,
            "pct_low_hdl": 70,
            "pct_hepatic_involvement": 10,
            "pct_misdiagnosed_chronic_kidney_disease_unknown": 55,
        },
        "dx_delay_distribution": {"<1 y": 5, "1–5 y": 14, "5–15 y": 14, ">15 y": 7},
    },
    # ── FGA — Hereditary Fibrinogen Aα-chain Amyloidosis (Ostertag) ──────────
    {
        "gene": "FGA",
        "protein": "FGA — Hereditary Fibrinogen Aα-chain Amyloidosis AD Ostertag-Type — Renal Dominant — No Neuropathy — Liver Transplant CURATIVE",
        "alias": (
            "FGA; OMIM gene 134820; Hereditary fibrinogen Aα-chain amyloidosis OMIM 105200; "
            "4q31.3; 866 aa; ~95 kDa Aα chain; AD missense in exon 5. "
            "FGA encodes the fibrinogen alpha chain (Aα chain). Fibrinogen is a hexameric plasma "
            "glycoprotein (Aα₂Bβ₂γ₂) essential for blood coagulation — it is cleaved by thrombin "
            "to form fibrin (clot formation). Liver is the sole site of fibrinogen synthesis. "
            "AMYLOIDOGENIC MECHANISM: pathogenic variants cluster in EXON 5 of FGA, encoding the "
            "C-terminal region of the Aα chain. These variants alter the local structure of the "
            "αC domain → creates an amyloidogenic peptide (approximately residues 519–580) that "
            "resists normal fibrinogen catabolism → deposits as amyloid in kidney glomeruli/vessels. "
            "KEY VARIANTS (all in exon 5, αC domain): "
            "E526V (Glu526Val, c.1577A>T) — most common worldwide ('Indian variant', also in UK); "
            "R554L (Arg554Leu) — 'United Kingdom variant'; "
            "L554P, G517V, V526I (distinct from E526V), R518G, W546L — various ethnicities. "
            "FIBRINOGEN LEVELS: paradoxically NORMAL or mildly elevated in heterozygotes — "
            "the intact Aα chain from the normal allele maintains adequate fibrinogen function; "
            "clotting tests (PT, APTT) NORMAL (haemostasis intact). "
            "PHENOTYPE — OSTERTAG TYPE (renal-dominant): "
            "PROTEINURIA → nephrotic syndrome → progressive CKD → ESRD "
            "(median time to ESRD ~10–20 years from proteinuria onset). "
            "NO significant NEUROPATHY (key distinction from TTR-FAP). "
            "NO significant CARDIAC amyloidosis. "
            "HYPERTENSION — very common (renal amyloid → HTN). "
            "HEPATIC: hepatomegaly present but NO liver failure. "
            "SPLENIC: splenomegaly common. "
            "RENAL PATHOLOGY: Congo red-positive deposits in glomeruli (mesangium + capillary walls) "
            "and arterioles; immunohistochemistry with anti-fibrinogen antibody positivity; "
            "mass spectrometry confirms fibrinogen amyloid typing. "
            "TREATMENT: LIVER TRANSPLANT IS CURATIVE — removes the only source of mutant fibrinogen "
            "Aα chain; after liver transplant, amyloid deposition stops; renal amyloid burden "
            "may stabilise or partially regress; patient series show preservation of renal function "
            "post-LT; combined liver-kidney transplant for established ESRD. "
            "Long-term outlook post-LT: excellent — renal function preserved in early-transplanted patients."
        ),
        "aa": "866 aa",
        "kDa": "~95 kDa (Aα chain)",
        "locus": "4q31.3",
        "omim_gene": 134820,
        "omim_disease": 105200,
        "inheritance": "AD; pathogenic variants exclusively in exon 5 (αC domain); E526V most common worldwide; incomplete penetrance in some variants; near-complete penetrance for E526V",
        "gene_class": (
            "FGA encodes the fibrinogen Aα (alpha) chain, one of three fibrinogen chains "
            "(Aα, Bβ, γ) that form the hexameric plasma coagulation protein fibrinogen (Aα₂Bβ₂γ₂). "
            "FIBRINOGEN STRUCTURE: D-D-E domain architecture; "
            "Aα chain contains the thrombin cleavage site (Arg16-Gly17, releasing fibrinopeptide A) "
            "and the αC domain (C-terminal, residues ~410–610) which mediates lateral fibrin "
            "polymerisation and is critical for clot architecture. "
            "AMYLOID PATHOLOGY SPECIFICS: only the Aα chain C-terminal αC fragment deposits "
            "(residues ~519–580); not the entire fibrinogen molecule; this fragment is generated "
            "by proteolytic processing of the circulating mutant fibrinogen; the fragment is "
            "amyloidogenic because the variant αC domain exposes a hydrophobic region that "
            "normally is buried. "
            "LIVER AS EXCLUSIVE SOURCE: fibrinogen is synthesised exclusively in the liver "
            "(hepatocytes); this makes liver transplant uniquely curative — no other organ or "
            "tissue makes fibrinogen, so after liver transplant the amyloidogenic Aα chain "
            "completely disappears from circulation. "
            "DISTINCTION FROM OTHER AMYLOIDOSES: Unlike TTR (neurotropic), AA (SAA-driven by "
            "inflammation), or AL (plasma cell dyscrasia), fibrinogen amyloidosis is purely "
            "renal-dominant with normal fibrinogen function and normal PT/APTT — "
            "the mutation is in a non-functional region (αC domain) that doesn't affect clotting."
        ),
        "n_patients": 40,
        "key_alerts": [
            "FGA-LIVER-TRANSPLANT-CURATIVE: Fibrinogen amyloidosis is UNIQUELY curable by liver transplant — the liver is the only organ that produces fibrinogen; liver transplant completely eliminates the mutant Aα chain from circulation; amyloid deposition stops; patients transplanted before ESRD can have preserved long-term renal function; early referral to transplant centre is the most important management decision",
            "FGA-EXON-5-ONLY-TESTING: All known pathogenic FGA amyloidosis variants are located in EXON 5 of the FGA gene; targeted exon 5 sequencing is sufficient for diagnostic testing and cascade screening; whole exome sequencing will capture this but Sanger sequencing of exon 5 is a cost-effective, fast approach for family screening",
            "FGA-NORMAL-CLOTTING-TESTS: Fibrinogen amyloidosis does NOT affect haemostasis; PT, APTT, fibrinogen level (Clauss method) are all normal in heterozygotes; do not dismiss the diagnosis because clotting tests are normal — the mutation affects the non-functional αC domain and one normal allele is sufficient for haemostatic fibrinogen; diagnosis is by tissue biopsy, anti-fibrinogen IHC, and genetic testing",
            "FGA-NO-NEUROPATHY: FGA amyloidosis does NOT cause peripheral neuropathy — this is a key distinguishing feature from TTR-FAP; a patient with renal amyloidosis, normal complement studies, and NO neuropathy should have FGA in the differential alongside APOA1 and APOA2; FGA amyloidosis is the 'Ostertag type' (renal amyloidosis without neuropathy) of the historical classification",
        ],
        "etiologies": {
            "E526V (Glu526Val) — most common; Indian, UK, European; exon 5": 20,
            "R554L (Arg554Leu) — UK variant; exon 5": 10,
            "L554P — Italian families; exon 5": 4,
            "G517V — Scandinavian families": 3,
            "R518G — rare; exon 5": 2,
            "Other exon 5 variants": 1,
        },
        "stats": {
            "mean_dx_age_y": 45.8,
            "mean_dx_delay_months": 60.0,
            "pct_renal_presentation": 98,
            "pct_esrd_at_diagnosis": 35,
            "pct_hypertension": 82,
            "pct_no_neuropathy": 98,
            "pct_normal_clotting": 100,
            "pct_post_lt_stable_renal": 70,
        },
        "dx_delay_distribution": {"<1 y": 4, "1–5 y": 15, "5–15 y": 15, ">15 y": 6},
    },
    # ── LYZ — Hereditary Lysozyme Amyloidosis ───────────────────────────────
    {
        "gene": "LYZ",
        "protein": "LYZ — Hereditary Lysozyme Amyloidosis AD — Hepatic Rupture Risk — Renal/GI — I56T/W64R — No DMT",
        "alias": (
            "LYZ; OMIM gene 153450; Hereditary lysozyme amyloidosis OMIM 105200 (allelic); "
            "12q15; 148 aa (preprotein including 18-aa signal peptide, mature 130 aa); "
            "~14 kDa; AD destabilising missense. "
            "LYZ encodes lysozyme C (muramidase), an antimicrobial enzyme that cleaves "
            "peptidoglycan in bacterial cell walls; abundant in tears, saliva, neutrophil granules, "
            "and macrophages; produced mainly by monocytes/granulocytes and epithelial cells. "
            "AMYLOIDOGENIC MECHANISM: Pathogenic missense variants reduce thermodynamic stability "
            "of the lysozyme fold → partial unfolding at physiological temperature → "
            "exposure of amyloidogenic segments → amyloid fibril formation. "
            "KEY VARIANTS: "
            "I56T (Ile56Thr) — most common in UK and European families; "
            "W64R (Trp64Arg) — described in UK families (Meretoja families); "
            "F57I (Phe57Ile) — French families; "
            "D67H, T70N — rare. "
            "Both I56T and W64R disrupt the hydrophobic core (B helix region), "
            "reducing the Tm (melting temperature) by ~15–20°C → protein partially unfolds "
            "at normal body temperature → fibril-forming intermediates accumulate. "
            "PHENOTYPE — MULTI-SYSTEM SYSTEMIC AMYLOIDOSIS: "
            "HEPATIC: hepatomegaly (prominent); hepatic amyloid deposits; "
            "HEPATIC RUPTURE — rare but reported, potentially fatal; "
            "occurs spontaneously or with minor trauma in patients with massive hepatic amyloid load; "
            "requires emergency surgical management. "
            "RENAL: proteinuria → CKD → ESRD; "
            "GASTROINTESTINAL: malabsorption; diarrhea; intestinal bleeding; "
            "may be the presenting feature; colonoscopy: amyloid deposits in submucosal vessels. "
            "SPLENIC: splenomegaly; hypersplenism (thrombocytopenia, anaemia). "
            "LYMPH NODE: widespread lymphadenopathy in some patients. "
            "NO CARDIAC amyloidosis (major distinction from TTR). "
            "NO NEUROPATHY (distinction from TTR-FAP). "
            "LABORATORY: SERUM LYSOZYME ELEVATED (paradox — mutant lysozyme still circulates "
            "despite being amyloidogenic; elevated lysozyme reflects release from deposits + "
            "ongoing production; serum lysozyme >10 μg/mL suggests amyloid-forming LYZ variant). "
            "DIAGNOSIS: Abdominal fat pad biopsy, rectal/liver/renal biopsy — Congo red positive; "
            "mass spectrometry proteomics identifies lysozyme; "
            "anti-lysozyme IHC on tissue sections; "
            "LYZ gene sequencing (targeted or WES). "
            "TREATMENT: No approved pharmacotherapy; liver transplant theoretically beneficial "
            "(granulocytes/macrophages also produce lysozyme — less predictably curative than FGA); "
            "renal transplant for ESRD; gastrointestinal: nutritional support; "
            "HEPATIC RUPTURE PREVENTION: avoid contact sports, heavy lifting; "
            "urgent surgical referral for hepatic pain with hepatomegaly."
        ),
        "aa": "148 aa",
        "kDa": "~14 kDa",
        "locus": "12q15",
        "omim_gene": 153450,
        "omim_disease": 105200,
        "inheritance": "AD destabilising missense; I56T most common; W64R second; small number of families worldwide",
        "gene_class": (
            "LYZ encodes lysozyme C (EC 3.2.1.17), a compact 130-aa antimicrobial enzyme. "
            "STRUCTURE: α + β fold (4 α-helices + triple-stranded antiparallel β-sheet); "
            "catalytic cleft cleaves the β(1→4) glycosidic bond between N-acetylmuramic acid "
            "and N-acetylglucosamine in bacterial peptidoglycan; "
            "active site residues: Glu35 (acid catalyst) + Asp52 (nucleophile/electrostatic). "
            "THERMAL STABILITY: native lysozyme is highly stable (Tm ~70°C); pathogenic I56T "
            "reduces Tm to ~55°C; W64R reduces Tm to ~50°C; at physiological temperature "
            "(37°C) these variants populate a partially unfolded equilibrium intermediate. "
            "AMYLOIDOGENIC INTERMEDIATE: the partially folded I56T/W64R lysozyme exposes a "
            "β-sheet-rich region (β-sheet A+B) that can self-associate → forms long-period "
            "amyloid fibrils distinct from WT lysozyme; both variants form amyloid in vitro "
            "at physiological temperature while WT does not. "
            "ELEVATED SERUM LYSOZYME: unlike most amyloid proteins where serum levels are "
            "normal (fibrinogen) or reduced (cystatin C in HCCAA), serum lysozyme is "
            "paradoxically ELEVATED in LYZ amyloidosis — partly from ongoing production by "
            "granulocytes and partly from tissue remodelling around deposits; useful screening marker."
        ),
        "n_patients": 40,
        "key_alerts": [
            "LYZ-HEPATIC-RUPTURE-EMERGENCY: Spontaneous hepatic rupture is a rare but life-threatening complication of hereditary lysozyme amyloidosis in patients with massive hepatic amyloid load; presents with acute abdominal pain + haemodynamic shock; CT abdomen urgently; surgical exploration/embolisation; patients with known LYZ amyloidosis and massive hepatomegaly should be counselled to avoid contact sports and heavy physical activity; keep surgical team informed",
            "LYZ-SERUM-LYSOZYME-SCREENING: Serum lysozyme >10 μg/mL (normal <10) is a useful screening marker for hereditary lysozyme amyloidosis; in a patient with systemic amyloidosis (hepatic, renal, gastrointestinal), a raised serum lysozyme should prompt LYZ gene testing; this is one of few hereditary amyloidoses with a useful serum biomarker",
            "LYZ-GASTROINTESTINAL-FIRST: GI manifestations (diarrhea, malabsorption, gastrointestinal bleeding) may be the presenting feature of LYZ amyloidosis — before renal or hepatic disease becomes apparent; endoscopy shows amyloid deposits in submucosal vessels; LYZ amyloidosis should be on the differential for systemic amyloidosis presenting with GI symptoms in whom AA and AL are excluded",
            "LYZ-NO-CARDIAC-NO-NEUROPATHY: Lysozyme amyloidosis does NOT cause significant cardiac amyloidosis or peripheral neuropathy — this combination of organ involvement (hepatic + renal + GI, sparing heart and nerves) with elevated serum lysozyme strongly suggests LYZ amyloidosis rather than TTR or AL",
        ],
        "etiologies": {
            "I56T (Ile56Thr) — most common; UK/European; core destabilisation": 20,
            "W64R (Trp64Arg) — UK families; hydrophobic core disruption": 12,
            "F57I (Phe57Ile) — French families": 5,
            "D67H (Asp67His) — rare": 2,
            "T70N and other variants": 1,
        },
        "stats": {
            "mean_dx_age_y": 42.3,
            "mean_dx_delay_months": 54.0,
            "pct_hepatic_presentation": 75,
            "pct_renal_involvement": 65,
            "pct_gi_involvement": 55,
            "pct_elevated_serum_lysozyme": 88,
            "pct_hepatic_rupture_risk": 5,
            "pct_no_cardiac_amyloid": 95,
        },
        "dx_delay_distribution": {"<1 y": 5, "1–5 y": 16, "5–15 y": 13, ">15 y": 6},
    },
    # ── CST3 — Hereditary Cystatin C Amyloid Angiopathy (HCCAA Iceland) ─────
    {
        "gene": "CST3",
        "protein": "CST3 — Hereditary Cystatin C Amyloid Angiopathy Iceland AD — L68Q — Cerebral Amyloid Angiopathy — Young Stroke 20s–30s PATHOGNOMONIC",
        "alias": (
            "CST3; OMIM gene 604312; Hereditary cystatin C amyloid angiopathy (HCCAA) OMIM 105150; "
            "20p11.21; 146 aa preprotein (120-aa mature + 26-aa signal peptide); "
            "~13 kDa; AD. "
            "CST3 encodes cystatin C (also called γ-trace, post-γ-globulin), a secreted cysteine "
            "protease inhibitor (targets cathepsins B, H, L, S). "
            "KEY BIOMARKER NOTE: Cystatin C is widely used as a renal biomarker (replaces creatinine "
            "for GFR estimation — CKD-EPI Cystatin C equation); this renal biomarker role is "
            "COMPLETELY DISTINCT from the amyloidogenic L68Q variant that causes HCCAA. "
            "PATHOGENIC VARIANT: L68Q (Leu68Gln, c.203T>A) — THE ONLY VARIANT CAUSING HCCAA. "
            "L68Q interrupts the hydrophobic core of domain II → destabilises the fold → "
            "increases tendency to form domain-swapped dimers → amyloid fibrils. "
            "L68Q cystatin C deposits predominantly in CEREBRAL VESSEL WALLS "
            "(leptomeningeal arteries, cortical arterioles) → CEREBRAL AMYLOID ANGIOPATHY (CAA). "
            "PHENOTYPE — HCCAA (Iceland disease): "
            "ONSET: YOUNG (median 1st haemorrhage age 26–30 years; range 20–80). "
            "PRIMARY MANIFESTATION: RECURRENT SPONTANEOUS CEREBRAL HAEMORRHAGES "
            "(lobar, cortical/subcortical; NOT lacunar). "
            "HIGH MORTALITY: ~50% die with first or second haemorrhage; "
            "survivors have progressive cognitive decline → dementia. "
            "ISCHAEMIC STROKES also occur. "
            "NO peripheral neuropathy. NO cardiac. NO renal involvement. "
            "SERUM CYSTATIN C NORMAL or low; CSF CYSTATIN C VERY LOW "
            "(deposits in vessels, cannot be secreted normally into CSF) — "
            "CSF cystatin C <0.35 mg/L (normal >0.35) is PATHOGNOMONIC for HCCAA in carriers; "
            "this distinguishes HCCAA from sporadic CAA (where CSF cystatin C is normal). "
            "IMAGING: MRI brain — multiple lobar haemorrhages at different ages (chronic + acute); "
            "susceptibility-weighted imaging (SWI) shows microhaemorrhages (multiple); "
            "superficial siderosis. "
            "TREATMENT: NO disease-modifying therapy; "
            "supportive BP control (aggressive target, e.g. <130/80); "
            "AVOID anticoagulants (haemorrhage risk); "
            "antiplatelet drugs controversial; "
            "genetic counselling is critical given autosomal dominant pattern and early onset; "
            "prenatal diagnosis available."
        ),
        "aa": "146 aa",
        "kDa": "~13 kDa",
        "locus": "20p11.21",
        "omim_gene": 604312,
        "omim_disease": 105150,
        "inheritance": "AD; single pathogenic variant L68Q described; all reported families trace to Iceland (founder effect); occasionally identified in other populations; near-complete penetrance for cerebral haemorrhage by age 40",
        "gene_class": (
            "CST3 encodes cystatin C, a 120-aa inhibitor of cysteine proteases (cathepsins B, H, L, S). "
            "STRUCTURE: two-domain structure: N-terminal domain (contacts protease) + "
            "C-terminal domain (additional protease binding + dimerisation surface); "
            "inhibits proteases by pseudo-substrate mechanism (wedge model). "
            "DOMAIN SWAPPING: L68Q cystatin C can form 'domain-swapped dimers' where the "
            "N-terminal β-strand of one monomer inserts into the C-terminal β-sheet of another "
            "→ this domain-swapped dimer is the building block of HCCAA amyloid fibrils; "
            "domain swapping is accelerated at physiological pH and temperature for L68Q. "
            "BIOMARKER DUAL ROLE CONFUSION RISK: Cystatin C in plasma/serum reflects GFR "
            "(elevated in kidney disease); HCCAA is a disease where cystatin C DEPOSITS "
            "in brain vessels causing low CSF cystatin C; these are unrelated phenomena; "
            "a patient with HCCAA has NORMAL renal cystatin C (GFR is normal) but LOW CSF cystatin C. "
            "ICELANDIC FOUNDER ALLELE: the L68Q allele traces to a common Icelandic founder; "
            "historical records show multiple large affected pedigrees in Iceland; the geographic "
            "clustering led to early characterisation of HCCAA as a distinct entity in the 1970s-80s "
            "by Gudmundsson and colleagues."
        ),
        "n_patients": 40,
        "key_alerts": [
            "CST3-YOUNG-ONSET-STROKE-HEREDITARY: Any young adult (20s–30s) with spontaneous lobar cerebral haemorrhage + family history of young stroke/haemorrhage should be evaluated for HCCAA; CST3 L68Q genetic testing is diagnostic; MRI brain with SWI reveals multiple haemorrhages at different stages — this pattern (multiple lobar haemorrhages at young age) is virtually PATHOGNOMONIC for a hereditary cerebral amyloid angiopathy",
            "CST3-CSF-CYSTATIN-C-LOW-PATHOGNOMONIC: CSF cystatin C level <0.35 mg/L (normal ≥0.35–0.40 mg/L) is PATHOGNOMONIC for active HCCAA in a known L68Q carrier; serum/plasma cystatin C is NORMAL (renal function preserved); do NOT confuse low CSF cystatin C with elevated serum cystatin C (renal biomarker) — they reflect entirely different biology; CSF cystatin C is a diagnostic AND monitoring biomarker for HCCAA",
            "CST3-ANTICOAGULATION-ABSOLUTELY-CONTRAINDICATED: Anticoagulants (warfarin, NOACs, heparin) are ABSOLUTELY CONTRAINDICATED in HCCAA patients; in sporadic CAA, anticoagulation is relatively contraindicated; in HCCAA with recurrent haemorrhage, any anticoagulation dramatically increases haemorrhage frequency and severity; inform all physicians (cardiologist, neurologist) of the HCCAA diagnosis before prescribing any antithrombotic therapy",
            "CST3-NOT-CREATININE-BIOMARKER: Elevated serum cystatin C is a marker of reduced GFR (renal biomarker); L68Q HCCAA affects the BRAIN, not the kidney — serum cystatin C in HCCAA patients reflects their renal function (usually normal); do not misinterpret the CST3 gene test result as a renal biomarker result; HCCAA is diagnosed by L68Q genotype + brain imaging + low CSF cystatin C",
        ],
        "etiologies": {
            "L68Q (Leu68Gln, c.203T>A) — ONLY known HCCAA variant; Icelandic founder": 40,
        },
        "stats": {
            "mean_dx_age_y": 28.5,
            "mean_dx_delay_months": 18.0,
            "pct_cerebral_haemorrhage_first": 90,
            "pct_ischaemic_stroke": 30,
            "pct_cognitive_decline": 75,
            "pct_low_csf_cystatin_c": 95,
            "pct_mortality_first_bleed": 50,
            "pct_anticoagulation_given_erroneously": 20,
        },
        "dx_delay_distribution": {"<1 y": 18, "1–5 y": 14, "5–15 y": 6, ">15 y": 2},
    },
    # ── GSN — Hereditary Gelsolin Amyloidosis (Finnish type / Meretoja) ─────
    {
        "gene": "GSN",
        "protein": "GSN — Hereditary Gelsolin Amyloidosis Finnish-Type AD — G654A — Corneal Lattice Dystrophy Type II PATHOGNOMONIC + Facial Palsy + Cutis Laxa",
        "alias": (
            "GSN; OMIM gene 137350; Familial amyloidosis Finnish type (FAF / Meretoja disease) OMIM 105120; "
            "9q33.2; 782 aa (cytoplasmic isoform); ~93 kDa; AD. "
            "GSN encodes gelsolin, a calcium-regulated actin filament-severing and capping protein; "
            "abundant in plasma (plasma gelsolin, pGSN, 80 kDa secreted isoform) and in "
            "cytoplasm (cytoplasmic gelsolin, 80 kDa). "
            "AMYLOIDOGENIC MECHANISM: "
            "PATHOGENIC VARIANTS: G654A (c.640G>A, p.Asp187Asn in mature protein) — most common; "
            "G654T (c.640G>T, p.Asp187Tyr) — described in Danish, Czech and other families. "
            "Both variants alter Asp187 in domain 2 (S2) of gelsolin. "
            "Asp187 is required for normal furin cleavage at the D2/D3 junction; "
            "mutant gelsolin → aberrant furin/MT1-MMP proteolysis → generates a 71-aa fragment "
            "(residues 173–243 of S2 domain = the amyloidogenic fragment); "
            "this fragment further cleaved by MT1-MMP → 71aa and 50aa fragments → "
            "amyloid fibrils depositing in CONNECTIVE TISSUE of eye, skin, nerve sheaths. "
            "CLASSIC TRIAD: "
            "1. CORNEAL LATTICE DYSTROPHY TYPE II (PATHOGNOMONIC) — "
            "Bilateral lattice-like opacities in corneal stroma, beginning at limbus and "
            "progressing centrally (TYPE II = from periphery inward); "
            "distinguishable from type I corneal lattice dystrophy (TGFBI/keratoepithelin, "
            "which is denser centrally and not associated with facial palsy); "
            "photophobia, recurrent corneal erosion, foreign body sensation; "
            "progressive corneal haziness → corneal transplant (PKP or DALK) needed in 40s–60s. "
            "2. CRANIAL NEUROPATHY — facial palsy (CN VII), bilateral, progressive; "
            "lagophthalmos (inability to close eyes fully) → exposure keratopathy; "
            "facial droop and asymmetry; blepharoptosis; "
            "sometimes CN IX/X/XII (hypoglossal, glossopharyngeal) involvement → "
            "dysarthria, dysphagia; "
            "FACIAL NERVE AMYLOID DEPOSITS in the facial nerve perineurium explain the palsy. "
            "3. CUTIS LAXA (SKIN LAXITY) — premature skin ageing; "
            "loose, pendulous skin (especially face, neck, eyelids); "
            "blepharochalasis (drooping upper eyelid skin); "
            "amyloid deposits in skin dermis and periappendageal connective tissue. "
            "ADDITIONAL: Mild peripheral sensorimotor neuropathy (late feature, 50s–60s); "
            "mild renal amyloidosis (not typically causing ESRD); "
            "cardiomyopathy described in a minority. "
            "ONSET: corneal lattice 20s–30s; facial palsy 30s–40s; cutis laxa 40s–60s. "
            "DIAGNOSIS: Slit-lamp biomicroscopy (corneal lattice type II starting from limbus); "
            "GSN gene sequencing; skin/labial salivary gland biopsy (Congo red + anti-gelsolin IHC); "
            "mass spectrometry proteomics. "
            "TREATMENT: NO approved pharmacotherapy; "
            "corneal transplant (PKP or DALK) for advanced corneal disease; "
            "eyelid surgery (tarsorrhaphy, gold weight implant for lagophthalmos); "
            "facial physiotherapy; lubricating eye drops for dry eye/exposure."
        ),
        "aa": "782 aa",
        "kDa": "~93 kDa",
        "locus": "9q33.2",
        "omim_gene": 137350,
        "omim_disease": 105120,
        "inheritance": "AD; G654A most common (Finnish, worldwide); G654T (Danish, Czech, others); high penetrance; phenotype variable in expression but corneal lattice type II virtually universal in G654A carriers",
        "gene_class": (
            "GSN encodes gelsolin, a multidomain actin-binding protein with 6 homologous domains (S1–S6). "
            "DOMAIN STRUCTURE: S1–S6, each ~125 aa, each with a β-sheet core flanked by α-helices; "
            "S1, S3, and S4 are actin-severing domains; S2, S4, and S6 mediate PIP2 inhibition; "
            "Ca²⁺ activates gelsolin by opening the S2–S6 'latch' and exposing the actin-binding sites. "
            "PLASMA GELSOLIN: 90% of circulating gelsolin acts as an actin scavenger in blood "
            "(removes cytotoxic actin released from dying cells); produced mainly by liver and muscle. "
            "PATHOGENIC CLEAVAGE AT D187N/Y: Asp187 in domain 2 (S2) is normally cleaved by furin "
            "to generate C-terminal fragments; D187N/Y changes the cleavage pattern → aberrant "
            "fragments generated by furin and MT1-MMP → 71-aa and 50-aa amyloidogenic peptides "
            "from the S2 domain depositing in connective tissue structures. "
            "WHY CORNEA AND NERVES: the amyloidogenic fragments deposit preferentially in "
            "connective tissue — corneal stroma (collagen-rich), nerve sheaths (perineurium), "
            "dermis — tissues with high connective tissue/collagen content; "
            "cardiac and renal involvement is mild/rare because these organs have different "
            "connective tissue composition."
        ),
        "n_patients": 40,
        "key_alerts": [
            "GSN-CORNEAL-LATTICE-TYPE-II-PATHOGNOMONIC: Bilateral corneal lattice dystrophy TYPE II (starting at the corneal limbus/periphery progressing centrally) in combination with progressive facial palsy is PATHOGNOMONIC for hereditary gelsolin amyloidosis (Meretoja disease); any ophthalmologist seeing bilateral corneal lattice type II should refer for genetic evaluation; distinguish from type I corneal lattice dystrophy (TGFBI, central location, no facial palsy, different management)",
            "GSN-LAGOPHTHALMOS-EXPOSURE-KERATOPATHY: Facial nerve palsy causes lagophthalmos (inability to fully close the eye) → chronic corneal exposure → exposure keratopathy, corneal ulceration, vision loss; this compounds the corneal lattice dystrophy; manage with: lubricating eye drops hourly, moisture chambers at night, tarsorrhaphy, gold weight eyelid implant, botulinum toxin to Müller's muscle — involve oculoplastic + corneal teams early",
            "GSN-NO-APPROVED-DMT: There is currently NO approved pharmacological treatment for hereditary gelsolin amyloidosis; management is entirely supportive/surgical; patients need multi-disciplinary care (ophthalmology, neurology, oculoplastics, dermatology); research into gelsolin-directed therapies (siRNA, ASO targeting GSN expression) is at early stages; enrolment in clinical registries/trials is recommended",
            "GSN-FACIAL-PALSY-NOT-BELLS: Bilateral progressive facial palsy in a patient from Finland or of Finnish ancestry should trigger slit-lamp examination for corneal lattice and GSN genetic testing before diagnosis of 'bilateral Bell's palsy' (which is very rare); gelsolin amyloidosis facial palsy is a peripheral nerve palsy due to amyloid deposits in the facial nerve sheath, not autoimmune; it does not respond to steroids",
        ],
        "etiologies": {
            "G654A (c.640G>A; p.Asp187Asn) — most common; Finnish and worldwide": 30,
            "G654T (c.640G>T; p.Asp187Tyr) — Danish, Czech, international": 10,
        },
        "stats": {
            "mean_dx_age_y": 34.7,
            "mean_dx_delay_months": 36.0,
            "pct_corneal_lattice_type2": 98,
            "pct_facial_palsy": 88,
            "pct_cutis_laxa": 72,
            "pct_peripheral_neuropathy_late": 45,
            "pct_corneal_transplant_needed": 42,
            "pct_misdiagnosed_bells_palsy": 35,
        },
        "dx_delay_distribution": {"<1 y": 8, "1–5 y": 18, "5–15 y": 10, ">15 y": 4},
    },
    # ── B2M — Hereditary Beta-2 Microglobulin Amyloidosis (D76N) ────────────
    {
        "gene": "B2M",
        "protein": "B2M — Hereditary B2M Amyloidosis AD Asp76Asn — Systemic — Distinct from Dialysis-Related — Cardiomyopathy + Hepatic + Renal — Mass Spectrometry Mandatory",
        "alias": (
            "B2M; OMIM gene 109700; Hereditary B2M amyloidosis OMIM 105200 (allelic); "
            "15q21.1; 119 aa (preprotein including 20-aa signal peptide, mature 99 aa); "
            "~12 kDa monomer; AD (hereditary form). "
            "B2M encodes beta-2 microglobulin, the invariant light chain of MHC class I "
            "(major histocompatibility complex class I) molecules. "
            "FUNCTION: B2M non-covalently associates with MHC-I heavy chains (HLA-A, -B, -C) "
            "and is required for surface expression of functional MHC-I; "
            "continuously shed from cell surfaces into the plasma; "
            "filtered and catabolised by the proximal renal tubule. "
            "TWO DISTINCT B2M AMYLOIDOSIS FORMS: "
            "(A) DIALYSIS-RELATED AMYLOIDOSIS (DRA) — NOT hereditary: "
            "WT B2M accumulates in patients with ESRD on long-term haemodialysis "
            "(dialysis membranes, especially cellulose, do not remove B2M adequately); "
            "WT B2M deposits in joint tissues → CARPAL TUNNEL SYNDROME (most common presentation), "
            "destructive arthropathy, spondylarthropathy, periarticular amyloid; "
            "effectively prevented by high-flux membranes and haemodiafiltration. "
            "(B) HEREDITARY B2M AMYLOIDOSIS (D76N) — DISTINCT FROM DRA: "
            "RARE AD form; pathogenic variant Asp76Asn (D76N) creates a B2M variant that is "
            "amyloidogenic even without dialysis and even with NORMAL RENAL FUNCTION; "
            "SYSTEMIC multi-organ amyloidosis: cardiac + hepatic + renal + splenic; "
            "PHENOTYPE: more aggressive than DRA; presents in 4th–6th decade; "
            "HFpEF-type cardiac involvement; hepatic amyloid; renal amyloid (CKD); "
            "peripheral neuropathy in some; "
            "NO articular involvement (unlike DRA). "
            "ADDITIONAL RARE VARIANTS: Val22Ile, Thr86Ala (very few families). "
            "LABORATORY: Serum B2M: ELEVATED (both DRA and hereditary D76N); "
            "serum B2M alone cannot distinguish DRA from hereditary; "
            "genetic testing + renal function assessment essential. "
            "DIAGNOSIS: Tissue biopsy Congo red; MASS SPECTROMETRY PROTEOMICS — "
            "mandatory to confirm B2M typing AND distinguish from DRA (both are B2M type); "
            "clinical context (renal function, dialysis history, family history) + genetic testing; "
            "MHC-I heavy chain typing (HLA) should be considered (B2M is MHC-I component). "
            "TREATMENT: NO approved pharmacotherapy; "
            "for cardiac involvement: standard HFpEF management; loop diuretics, "
            "avoid high heart rates, avoid drugs that reduce preload excessively; "
            "organ transplant for advanced failure; investigational anti-B2M therapies "
            "(SAP-binding drugs, B2M aggregation inhibitors)."
        ),
        "aa": "119 aa",
        "kDa": "~12 kDa",
        "locus": "15q21.1",
        "omim_gene": 109700,
        "omim_disease": 105200,
        "inheritance": "AD (hereditary D76N form); very rare; Asp76Asn pathogenic in multiple families; Val22Ile reported; all hereditary B2M amyloidosis families combined number in dozens worldwide",
        "gene_class": (
            "B2M encodes beta-2 microglobulin, an 11.7-kDa single-domain immunoglobulin "
            "superfamily protein. "
            "STRUCTURE: single β-sandwich immunoglobulin fold (7 β-strands); "
            "NO transmembrane domain; associates non-covalently with MHC-I heavy chains; "
            "no enzyme activity. "
            "AMYLOIDOGENESIS OF WT B2M: WT B2M can form amyloid under conditions of extreme "
            "accumulation (dialysis, high local concentration); requires partial unfolding; "
            "in DRA, acid pH + copper ions + collagen promote misfolding. "
            "D76N VARIANT AMYLOIDOGENESIS: Asp76Asn replaces an aspartate on the external face "
            "of strand D of the β-sandwich; D76 forms a salt bridge that stabilises the native fold; "
            "D76N disrupts this salt bridge → reduces thermodynamic stability by ~4 kcal/mol → "
            "D76N B2M can form amyloid at NEUTRAL pH and PHYSIOLOGICAL temperature "
            "(unlike WT B2M which requires low pH or extreme conditions); "
            "D76N amyloid deposits systemically (NOT just joints) because it is constitutively "
            "amyloidogenic without the special conditions (acid pH, copper, dialysis). "
            "MHC-I FUNCTION: D76N B2M retains ability to associate with MHC-I heavy chains "
            "and support surface expression; immune function is not grossly impaired; "
            "amyloidogenicity is an independent property of the fold-destabilised variant."
        ),
        "n_patients": 40,
        "key_alerts": [
            "B2M-D76N-NOT-DIALYSIS-RELATED: Hereditary B2M amyloidosis (D76N) is DISTINCT from dialysis-related amyloidosis (DRA); D76N patients have NORMAL renal function (no dialysis), SYSTEMIC multi-organ amyloidosis (cardiac, hepatic, renal, splenic), and no articular involvement; if amyloid proteomics show B2M typing in a non-dialysis patient, genetic B2M sequencing is mandatory; do not assume 'B2M amyloid = dialysis related' without renal and genetic workup",
            "B2M-MASS-SPECTROMETRY-TYPES-B2M: Standard IHC amyloid panels may include anti-B2M antibody for dialysis-related amyloidosis but will NOT reliably distinguish D76N hereditary form from DRA on IHC alone; mass spectrometry proteomics can confirm B2M subtype and identify co-deposited proteins that differ between DRA and hereditary B2M amyloidosis; specialist amyloid centre referral is essential",
            "B2M-CARDIAC-HFpEF-MANAGEMENT: Cardiac B2M amyloidosis presents as HFpEF-pattern (preserved ejection fraction, diastolic dysfunction, thickened walls); management follows general amyloid cardiomyopathy principles: avoid digoxin (proarrhythmic in amyloid), avoid excessive preload reduction (syncope risk), cautious use of loop diuretics; tafamidis does NOT work for B2M amyloid; no specific cardiac DMT available",
            "B2M-MHC-CLASS-I-FUNCTION-INTACT: D76N B2M retains its ability to associate with MHC-I heavy chains (HLA-A, -B, -C) and support cell-surface MHC-I expression; immune surveillance is not grossly impaired; however, if patients require immunosuppression for organ transplant, consider that MHC-I levels may need monitoring as B2M availability affects allograft immunogenicity assessments",
        ],
        "etiologies": {
            "Asp76Asn (D76N, c.226G>A) — most studied pathogenic variant; systemic amyloidosis": 30,
            "Val22Ile (V22I) — rare; reported in 2 families; cardiac involvement": 6,
            "Thr86Ala — very rare; severe multi-system": 4,
        },
        "stats": {
            "mean_dx_age_y": 54.1,
            "mean_dx_delay_months": 66.0,
            "pct_cardiac_involvement": 70,
            "pct_renal_involvement": 60,
            "pct_hepatic_involvement": 55,
            "pct_no_dialysis_history": 100,
            "pct_misdiagnosed_dra": 42,
            "pct_mass_spec_required_for_correct_typing": 100,
        },
        "dx_delay_distribution": {"<1 y": 3, "1–5 y": 12, "5–15 y": 16, ">15 y": 9},
    },
]


# ─── Patient cohort generation ────────────────────────────────────────────────

def _make_cohort():
    cohort = {}
    for i, gene_info in enumerate(AMYLOIDOSIS_GENES):
        seed = SEED_BASE + i
        rng = random.Random(seed)
        gene = gene_info["gene"]
        n = gene_info["n_patients"]
        patients = []
        for p in range(n):
            age_dx = round(rng.gauss(gene_info["stats"].get("mean_dx_age_y", 45), 10), 1)
            age_dx = max(15.0, min(85, age_dx))
            dx_delay = round(rng.gauss(gene_info["stats"].get("mean_dx_delay_months", 60), 24), 1)
            dx_delay = max(1.0, min(240, dx_delay))
            patients.append({
                "patient_id": f"{gene}-{seed}-{p+1:03d}",
                "gene": gene,
                "age_at_diagnosis": age_dx,
                "diagnosis_delay_months": dx_delay,
                "seed": seed,
            })
        cohort[gene] = {
            **gene_info,
            "patients": patients,
        }
    return cohort


_COHORT = _make_cohort()


def get_overview():
    total = sum(v["n_patients"] for v in _COHORT.values())
    mean_dx_age = round(
        sum(p["age_at_diagnosis"] for v in _COHORT.values() for p in v["patients"]) / total, 1
    )
    mean_dx_delay = round(
        sum(p["diagnosis_delay_months"] for v in _COHORT.values() for p in v["patients"]) / total, 1
    )

    top_alerts = []
    for v in _COHORT.values():
        top_alerts.extend(v["key_alerts"][:2])

    genes_summary = []
    for g, v in _COHORT.items():
        pts = v["patients"]
        mean_age = round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1)
        genes_summary.append({
            "gene": g,
            "protein_short": v["protein"][:80],
            "locus": v["locus"],
            "inheritance": v["inheritance"].split(";")[0],
            "omim_disease": v["omim_disease"],
            "mean_dx_age": mean_age,
            "n_patients": v["n_patients"],
        })

    ttr   = _COHORT["TTR"]["stats"]
    cst3  = _COHORT["CST3"]["stats"]
    gsn   = _COHORT["GSN"]["stats"]
    fga   = _COHORT["FGA"]["stats"]
    lyz   = _COHORT["LYZ"]["stats"]

    return {
        "atlas": "Hereditary-Amyloidosis-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Systemic Amyloidosis Reference",
        "genes": genes_summary,
        "aggregate_stats": {
            "total_patients": total,
            "mean_dx_age_years": mean_dx_age,
            "mean_dx_delay_months": mean_dx_delay,
            "ttr_carpal_tunnel_first_pct": ttr["pct_carpal_tunnel_first_sign"],
            "ttr_cardiac_involvement_pct": ttr["pct_cardiac_involvement"],
            "ttr_misdiagnosed_pct": ttr["pct_misdiagnosed_cmt_or_cidp"],
            "cst3_young_haemorrhage_pct": cst3["pct_cerebral_haemorrhage_first"],
            "cst3_mortality_first_bleed_pct": cst3["pct_mortality_first_bleed"],
            "gsn_corneal_lattice_pct": gsn["pct_corneal_lattice_type2"],
            "gsn_facial_palsy_pct": gsn["pct_facial_palsy"],
            "fga_curative_lt_pct": fga["pct_post_lt_stable_renal"],
            "lyz_hepatic_rupture_risk_pct": lyz["pct_hepatic_rupture_risk"],
            "cascade_tested_pct": 58,
        },
        "top_alerts": top_alerts,
        "seeds": list(range(SEED_BASE, SEED_BASE + 8)),
    }


def get_breakdown():
    result = {}
    for gene, info in _COHORT.items():
        pts = info["patients"]
        result[gene] = {
            "gene": gene,
            "n_patients": info["n_patients"],
            "alias": info["alias"],
            "gene_class": info["gene_class"],
            "locus": info["locus"],
            "aa": info["aa"],
            "kDa": info["kDa"],
            "omim_gene": info["omim_gene"],
            "omim_disease": info["omim_disease"],
            "inheritance": info["inheritance"],
            "key_alerts": info["key_alerts"],
            "etiologies": info["etiologies"],
            "stats": info["stats"],
            "dx_delay_distribution": info["dx_delay_distribution"],
            "patients": pts[:10],
        }
    return result


def get_definitions():
    return {
        "atlas": "Hereditary-Amyloidosis-Atlas",
        "concepts": {
            "Amyloid Classification — AL, AA, TTR, and Hereditary Subtypes": (
                "Systemic amyloidosis is classified by the precursor protein forming the amyloid deposit. "
                "MAJOR SUBTYPES: "
                "AL (Immunoglobulin Light Chain) — most common systemic amyloidosis in developed countries; "
                "plasma cell dyscrasia producing misfolded immunoglobulin light chains; "
                "treated with plasma cell-directed therapy (daratumumab, bortezomib, melphalan, SCT). "
                "AA (Serum Amyloid A) — chronic inflammatory amyloidosis; SAA protein (acute-phase reactant) "
                "deposits when chronically elevated (rheumatoid arthritis, inflammatory bowel, "
                "familial Mediterranean fever MEFV, chronic osteomyelitis); treat underlying inflammation. "
                "WT-ATTR (wild-type transthyretin amyloidosis) — senile systemic amyloidosis; "
                "WT-TTR deposits in elderly (predominantly cardiac); treated with tafamidis. "
                "ATTRv (variant TTR) — hereditary TTR amyloidosis; treated with gene silencing + tafamidis. "
                "HEREDITARY RARE FORMS (this atlas): APOA1, APOA2, FGA, LYZ, CST3, GSN, B2M. "
                "DIAGNOSTIC RULE: MASS SPECTROMETRY IS THE GOLD STANDARD for amyloid typing; "
                "IHC is a useful but imperfect screen; proteomics definitively identifies the "
                "amyloidogenic protein; always refer to specialist amyloid centre for typing before "
                "treatment — wrong typing = wrong treatment."
            ),
            "Amyloid Diagnosis — Congo Red, Mass Spectrometry, and Biopsy Strategy": (
                "CONGO RED STAINING: amyloid deposits stain red with Congo red and show "
                "APPLE-GREEN BIREFRINGENCE under cross-polarised light — this is the sine qua non "
                "for amyloid diagnosis; ALL hereditary amyloidoses show Congo red positivity. "
                "BIOPSY SITES (in order of preference): "
                "1. Abdominal fat pad aspirate — 80% sensitive for systemic amyloidosis; "
                "   simple, safe, outpatient; first-line. "
                "2. Rectal biopsy — submucosa; 75–80% sensitivity. "
                "3. Bone marrow biopsy — often done if AL amyloidosis suspected (marrow also "
                "   shows plasma cell dyscrasia). "
                "4. Organ-specific biopsy (kidney, liver, nerve) — 95%+ sensitive; "
                "   higher risk; use when fat pad negative but clinical suspicion high. "
                "AMYLOID TYPING STRATEGY: "
                "Step 1: Serum and urine protein electrophoresis + immunofixation → exclude AL. "
                "Step 2: Serum SAA → if elevated + chronic inflammation → AA likely. "
                "Step 3: Echocardiogram + 99mTc-DPD scan → if positive + AL negative → ATTR-CM. "
                "Step 4: GENETIC TESTING (TTR, then hereditary panel if TTR negative). "
                "Step 5: MASS SPECTROMETRY PROTEOMICS of amyloid extract → definitive typing. "
                "Never start specific therapy based on clinical suspicion alone — confirm type first."
            ),
            "TTR Amyloidosis — Tetramer Dissociation and Gene Silencing Mechanism": (
                "TTR tetramer dissociation is the rate-limiting step in ATTR amyloidogenesis. "
                "THERMODYNAMIC BASIS: TTR homo-tetramer is kinetically stable but thermodynamically "
                "metastable — pathogenic variants shift equilibrium toward monomeric intermediate "
                "by reducing the activation barrier for tetramer dissociation. "
                "TETRAMER STABILISERS: tafamidis, diflunisal, acoramidis bind to the T4 binding sites "
                "at the tetramer central channel → kinetically lock the tetramer → prevent "
                "dissociation → no monomeric intermediate → no amyloid. Tafamidis is >100× more "
                "potent than diflunisal at tetramer stabilisation. "
                "GENE SILENCING (siRNA/ASO): patisiran, vutrisiran (siRNA) and inotersen, eplontersen (ASO) "
                "target hepatic TTR mRNA → reduce TTR synthesis by 70–90% in liver → dramatically "
                "less substrate for amyloid formation → neuropathy improvement in FAP. "
                "KEY POINT: gene silencing reduces the SOURCE of amyloidogenic TTR (liver); "
                "tetramer stabilisers prevent the MISFOLDING of circulating TTR; "
                "these are complementary mechanisms targeting different steps."
            ),
            "Liver Transplant in Hereditary Amyloidosis — Curative Potential and Limitations": (
                "RATIONALE: for hereditary amyloidoses where the liver is the SOLE source of the "
                "amyloidogenic precursor protein, liver transplant removes the source → "
                "circulating amyloidogenic protein disappears → amyloid deposition ceases → "
                "amyloid may slowly regress (macrophage-mediated clearance, which is slow). "
                "CURATIVE FOR FGA AMYLOIDOSIS: fibrinogen Aα chain is made EXCLUSIVELY in the "
                "liver → orthotopic liver transplant (OLT) eliminates mutant fibrinogen → "
                "serum fibrinogen normalises → renal amyloid burden stable/decreasing; "
                "patients transplanted before ESRD can preserve renal function long-term; "
                "BEST EVIDENCE for curability — FGA OLT results are excellent. "
                "PARTIALLY CURATIVE FOR TTR AMYLOIDOSIS: liver removes 90% of circulating TTR; "
                "post-OLT neuropathy arrests; cardiac disease may continue (choroid plexus TTR + "
                "WT-TTR deposits on old fibril seeds); OLT largely superseded by gene silencing. "
                "PARTIALLY CURATIVE FOR APOA1/LYZ: liver is major but not sole source of apoA-I "
                "or lysozyme; OLT reduces amyloidogenic protein load substantially; "
                "renal + hepatic amyloid stabilises; not as clearly curative as FGA. "
                "NOT CURATIVE FOR CST3/GSN: cystatin C is made by all nucleated cells; "
                "gelsolin is made by many cell types including muscle; "
                "liver transplant does not eliminate the amyloidogenic source. "
                "TIMING: liver transplant BEFORE ESRD or organ failure offers best outcomes."
            ),
            "Organ Tropism in Hereditary Amyloidosis — Why Different Genes Target Different Organs": (
                "The organ tropism (which organ is preferentially affected) in hereditary amyloidosis "
                "is determined by multiple factors: "
                "1. SITE OF PRODUCTION: proteins made primarily in liver (TTR, fibrinogen, apoA-I) "
                "are exported into blood and can deposit widely; cystatin C (made by ALL nucleated "
                "cells) deposits nearest to where the highest concentrations of amyloidogenic intermediate "
                "form (cerebral vessels in HCCAA = blood-brain barrier intersection); "
                "gelsolin (abundant in connective tissue and plasma) deposits in connective tissue. "
                "2. PROTEIN STRUCTURE: amyloidogenic fragments carry intrinsic tissue-targeting motifs; "
                "e.g., gelsolin S2 fragment deposits in collagen-rich connective tissue; "
                "cystatin C L68Q deposits in vessel walls. "
                "3. FILTRATION: B2M (WT form) accumulates in joints because urinary B2M clearance "
                "fails in dialysis patients → concentration in synovial fluid. "
                "4. FRAGMENT SIZE AND CHARGE: smaller fragments deposit more distally; "
                "charge determines interaction with matrix components (heparan sulphate etc.). "
                "CLINICAL IMPLICATION: organ tropism predicts prognosis and guides surveillance; "
                "TTR-cardiomyopathy needs ECG, echo, NT-proBNP; FGA needs urinalysis/GFR; "
                "CST3-HCCAA needs MRI brain; GSN needs slit-lamp + neurological exam."
            ),
        },
        "pharmacological_distinctions": [
            "Patisiran vs Vutrisiran: Both are siRNA targeting TTR mRNA in hepatocytes; patisiran (Onpattro) requires IV infusion every 3 weeks with premedication; vutrisiran (Amvuttra) is SC quarterly with no premedication; both reduce TTR by ~80–87%; vutrisiran preferred for convenience but patisiran has longer post-approval safety data; both approved for ATTRv polyneuropathy",
            "Tafamidis vs Gene Silencing for TTR amyloidosis: Tafamidis stabilises the TTR tetramer — it works on circulating TTR regardless of where it is produced (liver or choroid plexus); gene silencing (siRNA/ASO) reduces hepatic TTR production by 80–90% but choroid plexus TTR is unaffected; for FAC (cardiac), tafamidis is first-line approved; for FAP (neuropathy), gene silencing is preferred; in mixed FAP+FAC, both may be combined",
            "Inotersen REMS vs Vutrisiran (no REMS): Inotersen (ASO) carries a mandatory REMS for thrombocytopenia (including fatal) and glomerulonephritis; platelet monitoring every 2 weeks is required; vutrisiran and patisiran (siRNA) do not have the same thrombocytopenia risk profile and do not require REMS; this is a key safety distinction when choosing among TTR-directed therapies",
            "FGA Liver Transplant vs Supportive: Liver transplant for FGA amyloidosis is uniquely curative because the liver is the sole fibrinogen source; no pharmacological alternative exists; the evidence base strongly favours early liver transplant (before ESRD) for FGA amyloidosis; renal transplant alone for ESRD does not remove the source → amyloid recurs in the transplanted kidney",
            "AL vs Hereditary Amyloidosis Treatment Paradigm: AL amyloidosis is treated with plasma cell-directed therapy (daratumumab, bortezomib, melphalan, SCT — essentially myeloma therapy); hereditary amyloidoses are treated with gene silencing, tetramer stabilisation, or organ transplant targeting the specific precursor protein; misdiagnosing hereditary amyloidosis as AL leads to inappropriate chemotherapy with toxicity and no benefit — correct typing by mass spectrometry is mandatory before treatment",
        ],
        "key_standards": [
            "AMYLOIDOSIS TYPING MANDATORY: ISA (International Society of Amyloidosis) consensus: MASS SPECTROMETRY is mandatory for amyloid subtype confirmation; IHC alone is insufficient; refer to specialist amyloid centre (Mayo, UCL/UCLH, Heidelberg, Pavia) for proteomics typing before starting any specific treatment",
            "CASCADE GENETIC TESTING: all first-degree relatives of any hereditary amyloidosis patient require genetic counselling and testing; penetrance is high for most variants; early identification allows monitoring and timely intervention before irreversible organ damage",
            "TTR TAFAMIDIS ATTR-CM: ATTR-ACT trial (NEJM 2018) demonstrated tafamidis 80 mg reduces all-cause mortality and cardiovascular hospitalisation in ATTR-CM; 99mTc-DPD/PYP scan Grade 2-3 with negative serum/urine immunofixation = non-invasive ATTR-CM diagnosis; start tafamidis without biopsy in this setting",
            "FGA EARLY LIVER TRANSPLANT: international FGA amyloidosis registries recommend liver transplant evaluation when proteinuria is established but before eGFR <30 mL/min/1.73m²; combined liver-kidney transplant for eGFR <20; do not defer OLT until ESRD",
            "AVOID AL TREATMENT FOR HEREDITARY AMYLOIDOSIS: confirmed hereditary amyloidosis (by genetics + mass spectrometry) should NOT receive AL-directed chemotherapy; plasma cell clone is absent; chemotherapy causes toxicity without benefit",
            "CST3-HCCAA ANTICOAGULATION CI: anticoagulants (warfarin, NOACs, heparin, thrombolytics) are contraindicated in HCCAA (L68Q cystatin C amyloid angiopathy); document allergy/contraindication in all medical records; alert cardiologist/neurologist/anaesthetist at every encounter",
        ],
    }
