#!/usr/bin/env python3
"""Hereditary-Sarcoglycanopathy-LGMD-Atlas — Complete 8-Gene Sarcoglycanopathy & LGMD Spectrum Atlas
(LGMD-R3-SGCA · LGMD-R4-SGCB · LGMD-R5-SGCG · LGMD-R6-SGCD ·
 LGMD-R9-FKRP · LGMD-R12-ANO5 · LGMD-R8-TRIM32 · LGMD-R7-TCAP).

SGCA    (α-Sarcoglycan; 387 aa; 17q21.33; AR;
         LGMD-R3 — most common sarcoglycanopathy in European cohorts;
         CK 10-70× ULN PATHOGNOMONIC; childhood onset 5-15 yr;
         IHC: absent α-SG + secondary reduction β/γ/δ-SG;
         seed SEED_BASE+0).
SGCB    (β-Sarcoglycan; 318 aa; 4q12; AR;
         LGMD-R4 — C283Y founder mutation North Africa / Middle East;
         DMD-like phenotype possible in severe cases;
         secondary IHC reduction pattern localises primary gene;
         seed SEED_BASE+1).
SGCG    (γ-Sarcoglycan; 291 aa; 13q12.12; AR;
         LGMD-R5 — most severe sarcoglycanopathy; del521T founder N.Africa;
         early childhood onset, rapid progression, wheelchair by teens;
         CARDIAC + RESPIRATORY involvement mandatory screening;
         seed SEED_BASE+2).
SGCD    (δ-Sarcoglycan; 290 aa; 5q33.3; AR;
         LGMD-R6 — rarest sarcoglycanopathy;
         DILATED CARDIOMYOPATHY up to 100% PATHOGNOMONIC — DCM may precede weakness;
         present as isolated DCM — muscle biopsy/gene panel in unexplained DCM;
         seed SEED_BASE+3).
FKRP    (Fukutin-Related Protein; 495 aa; 19q13.32; AR;
         LGMD-R9 — most common AR LGMD in UK/Scandinavia;
         p.Leu276Ile (L276I) founder — mild LGMD spectrum;
         CARDIAC DCM 30-40% MANDATORY ECHO SCREENING;
         calf hypertrophy prominent; spectrum L276I-mild → MDC1C-congenital;
         seed SEED_BASE+4).
ANO5    (Anoctamin-5; 913 aa; 11p14.3; AR;
         LGMD-R12 — Miyoshi-like; p.Arg758Cys Dutch/Belgian founder;
         QUADRICEPS SPARED early PATHOGNOMONIC DDx DYSF;
         posterior calf atrophy; CK markedly elevated;
         NO cardiac involvement — critical DDx from FKRP/SGCD;
         seed SEED_BASE+5).
TRIM32  (Tripartite Motif-Containing 32; 653 aa; 9q33.1; AR;
         LGMD-R8 — sarcotubular myopathy (STM) on biopsy;
         SARCOTUBULAR AGGREGATES on EM PATHOGNOMONIC;
         facial weakness + psychiatric features clue;
         very rare (<100 cases worldwide);
         seed SEED_BASE+6).
TCAP    (Titin-Cap/Telethonin; 167 aa; 17q12; AR;
         LGMD-R7 — rarest LGMD; Z-disc structural protein;
         DILATED CARDIOMYOPATHY significant cardiac involvement;
         rimmed vacuoles on biopsy;
         Uruguay/Brazil/China founder mutations;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2182-2189).
"""

import random

SEED_BASE = 2182

LGMD_GENES = [
    # -- SGCA — α-Sarcoglycan, LGMD-R3 -----------------------------------------------
    {
        "gene": "SGCA",
        "alt_name": (
            "SGCA (SGCA-387aa-17q21.33 / AR — LGMD-R3-alpha-Sarcoglycanopathy — "
            "CK-10-70x-ULN-PATHOGNOMONIC-Childhood-Onset-5-15yr — "
            "IHC-Absent-alphaSG-Secondary-Reduction-betaGammaDelta-SG — "
            "Gene-Therapy-rAAVrh74-MCK-SGCA-Phase2-Trials)"
        ),
        "protein": (
            "SGCA -- 17q21.33 AR -- SGCA-387aa -- "
            "alpha-Sarcoglycan-Adhalin-50kDa-Type-I-Transmembrane-Dystrophin-Associated-Protein-Complex -- "
            "LGMD-R3-Formerly-LGMD2D-OMIM-600119-Disease-OMIM-608099 -- "
            "Sarcoglycan-Complex-SGCA-SGCB-SGCG-SGCD-Heterotetrameric-Membrane-Stabilisation -- "
            "CK-10-70x-ULN-Mean-30x-PATHOGNOMONIC-Level-Myonecrosis -- "
            "Childhood-Onset-5-15yr-Pelvifemoral-Weakness-Gowers-Sign-Calf-Pseudohypertrophy -- "
            "Wheelchair-Age-15-25yr-Severe-Cases-Later-Mild-Cases -- "
            "IHC-Absent-alphaSG-Secondary-REDUCTION-betaSG-gammaSG-deltaSG -- "
            "IHC-Panel-Mandatory-All-4-SG-Localise-Primary-Gene -- "
            "Western-Blot-Absent-Band-50kDa-SGCA -- "
            "Most-Common-Sarcoglycanopathy-European-Cohorts -- "
            "Founder-Mutations-T228C-R284C-V247M-Population-Specific -- "
            "Gene-Therapy-rAAVrh74-MCK-SGCA-Phase1-2-Trials-Underway -- "
            "Cardiac-Rare-Unlike-SGCD-Respiratory-Late-FVC-Annual-After-Wheelchair -- "
            "Corticosteroids-Deflazacort-Prednisone-Slows-Progression-Level-C -- "
            "No-STEROIDS-Contraindication-Unlike-DYSF-But-Evidence-Weaker-Than-DMD -- "
            "17q21.33"
        ),
        "locus": "17q21.33",
        "protein_size": "387 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function or missense variants in SGCA; "
            "no carrier phenotype; de novo rare; "
            "IHC guides gene selection; WES/panel mandatory for gene confirmation; "
            "population-specific founder alleles (T228C, R284C)."
        ),
        "key_features": [
            "Pelvifemoral weakness: hip flexors, knee extensors, glutei — proximal lower limb first",
            "CK 10-70× ULN PATHOGNOMONIC — highest CK among sarcoglycanopathies",
            "Childhood onset 5-15 yr; Gowers sign; calf pseudohypertrophy early",
            "IHC: absent α-SG + secondary reduction β/γ/δ-SG — IHC panel directs gene testing",
            "Cardiac: RARE — contrast with SGCD (DCM common) and FKRP (DCM 30-40%)",
            "Gene therapy trials: rAAVrh74.MCK.SGCA Phase 1-2 — first sarcoglycanopathy GTx",
        ],
        "treatment": [
            "Deflazacort 0.9 mg/kg/day or prednisone 0.75 mg/kg/day — Level C evidence; slows decline",
            "Physiotherapy + stretching — ankle-foot orthoses when toe-walking; spine bracing if scoliosis",
            "Annual ECG + echo — cardiac surveillance even if rare; baseline at diagnosis",
            "Annual spirometry — NIV when FVC <50% (late, usually after wheelchair)",
            "Gene therapy: trial eligibility assessment — rAAVrh74.MCK.SGCA early-phase trials",
            "Genetic counselling mandatory — sibling testing; carrier testing in consanguineous families",
        ],
        "contraindications": [
            "STEROIDS CAUTION — less evidence than in DMD; monitor bone density with long-term use",
            "DEHYDRATION — myoglobinuria risk with vigorous exercise; hydration mandatory",
            "AVOID high-impact exercise — eccentric contractions worsen membrane fragility",
            "GENERAL ANAESTHESIA — CK monitoring post-op; rhabdomyolysis risk with volatile agents",
        ],
        "critical_pearls": [
            "IHC shows SECONDARY reduction of all 4 SGs when primary gene is SGCA — must test all 4",
            "Normal IHC does NOT exclude sarcoglycanopathy — normal in up to 15%; WES needed",
            "CK >10,000 IU/L in a child — sarcoglycanopathy must be on differential (alongside DMD if male)",
            "Gene therapy trials are open: check ClinicalTrials.gov with SGCA/LGMD2D search",
            "Cardiac rare but not zero — annual echo still required",
        ],
    },
    # -- SGCB — β-Sarcoglycan, LGMD-R4 -----------------------------------------------
    {
        "gene": "SGCB",
        "alt_name": (
            "SGCB (SGCB-318aa-4q12 / AR — LGMD-R4-beta-Sarcoglycanopathy — "
            "C283Y-Founder-Mutation-North-Africa-Middle-East — "
            "DMD-Like-Phenotype-Severe-Cases — "
            "IHC-Absent-betaSG-Secondary-Reduction-alphaSG)"
        ),
        "protein": (
            "SGCB -- 4q12 AR -- SGCB-318aa -- "
            "beta-Sarcoglycan-43kDa-Type-II-Transmembrane-Glycoprotein-Dystrophin-Associated -- "
            "LGMD-R4-Formerly-LGMD2E-OMIM-600900-Disease-OMIM-604286 -- "
            "Sarcoglycan-Complex-Nucleation-betaSG-First-Incorporated-Into-Complex -- "
            "CK-10-50x-ULN-Similar-SGCA -- "
            "Childhood-Onset-5-15yr-Similar-SGCA-Phenotype -- "
            "C283Y-Founder-North-Africa-Middle-East-Saudi-Arabia-Common -- "
            "IHC-Absent-betaSG-Secondary-Reduction-alphaSG-gammaSG-deltaSG -- "
            "DMD-Like-Phenotype-Possible-Severe-Alleles-Genetic-Test-Mandatory-Males -- "
            "Cardiac-Involvement-Less-Common-Than-SGCD-Annual-Echo-Recommended -- "
            "Scoliosis-Lumbar-Lordosis-Common-Physiotherapy-Bracing -- "
            "4q12"
        ),
        "locus": "4q12",
        "protein_size": "318 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic SGCB variants; "
            "c.848G>A (p.Cys283Tyr) — North African/Middle Eastern founder mutation; "
            "DMD-like severity in null alleles; mild in missense alleles; "
            "IHC guides initial gene selection (absent β-SG); WES panel confirms."
        ),
        "key_features": [
            "Phenotype overlaps SGCA: pelvifemoral weakness, CK 10-50× ULN, childhood onset",
            "p.Cys283Tyr (C283Y) — North African/Middle Eastern founder: most common SGCB allele worldwide",
            "IHC: absent β-SG + secondary reduction α/γ/δ-SG — IHC pattern similar across sarcoglycanopathies",
            "DMD-like severe phenotype in null/null or null/missense — genetic test mandatory in males",
            "Scoliosis + lumbar lordosis prominent in ambulant phase — physiotherapy + bracing",
            "Cardiac: less frequent than SGCD — annual echo still recommended",
        ],
        "treatment": [
            "Deflazacort/prednisone — same Level C evidence as SGCA; same protocol",
            "Scoliosis surveillance — Cobb angle annually; surgical fixation if >45°",
            "Annual ECG + echo — cardiac less common but still mandatory surveillance",
            "Annual spirometry — NIV when FVC <50%",
            "Physiotherapy, orthoses, adaptive equipment — mobility prolongation",
            "Genetic counselling — founder mutation C283Y allows targeted carrier screening in N.African/M.East populations",
        ],
        "contraindications": [
            "STEROIDS CAUTION — same as SGCA; bone density monitoring with prolonged use",
            "VIGOROUS ECCENTRIC EXERCISE — membrane fragility; myoglobinuria risk",
            "ANAESTHETIC MONITORING — CK/rhabdomyolysis post-op; volatile anaesthetic caution",
        ],
        "critical_pearls": [
            "In any North African/Middle Eastern child with LGMD phenotype — SGCB C283Y is top differential",
            "IHC: β-SG is the nucleating subunit — absent β-SG causes secondary loss of others",
            "Males with LGMD + absent IHC: test DMD (dystrophin deletion) before SGCB — DMD far more common",
            "Genotype-phenotype correlation: null/null = severe (DMD-like); missense/null = intermediate",
        ],
    },
    # -- SGCG — γ-Sarcoglycan, LGMD-R5 -----------------------------------------------
    {
        "gene": "SGCG",
        "alt_name": (
            "SGCG (SGCG-291aa-13q12.12 / AR — LGMD-R5-gamma-Sarcoglycanopathy-MOST-SEVERE — "
            "del521T-c525delT-Founder-North-Africa — "
            "CARDIAC-Dilated-Cardiomyopathy-Respiratory-Mandatory-Screening — "
            "Childhood-Rapid-Progression-Wheelchair-Teens)"
        ),
        "protein": (
            "SGCG -- 13q12.12 AR -- SGCG-291aa -- "
            "gamma-Sarcoglycan-35kDa-Type-II-Transmembrane-Dystrophin-Associated-Protein-Complex -- "
            "LGMD-R5-Formerly-LGMD2C-OMIM-253700-Disease-OMIM-608896 -- "
            "Most-Severe-Sarcoglycanopathy-Earliest-Onset-Fastest-Progression -- "
            "CK-20-100x-ULN-HIGHEST-CK-Sarcoglycanopathy -- "
            "del521T-c525delT-North-African-Founder-Algeria-Tunisia-Morocco -- "
            "Childhood-Onset-3-10yr-Earlier-Than-SGCA-SGCB -- "
            "Wheelchair-Dependence-Teens-Early-Teens-Faster-Than-Other-SG -- "
            "CARDIAC-DCM-Dilated-Cardiomyopathy-Significant-6-Month-Echo-Mandatory -- "
            "RESPIRATORY-FVC-Annual-NIV-Often-Required-Pre-Wheelchair -- "
            "Scoliosis-Severe-Surgical-Fixation-Common -- "
            "IHC-Absent-gammaSG-Primary-Secondary-All-SG -- "
            "13q12.12"
        ),
        "locus": "13q12.12",
        "protein_size": "291 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic SGCG loss-of-function; "
            "c.525delT (del521T/p.Cys175Serfs*18) — North African founder, allele frequency up to 1/50 in Algeria; "
            "most severe sarcoglycanopathy phenotype; "
            "onset 3-10 yr; rapid loss of ambulation; "
            "IHC: absent γ-SG + secondary reduction all SGs."
        ),
        "key_features": [
            "MOST SEVERE sarcoglycanopathy — onset 3-10 yr, wheelchair dependence teens",
            "CK 20-100× ULN — HIGHEST among sarcoglycanopathies at presentation",
            "del521T (c.525delT) — North African founder; allele frequency 1/50 in Algeria",
            "CARDIAC DCM: significant, earlier than other sarcoglycanopathies — echo every 6 months",
            "RESPIRATORY: FVC annual; NIV may be needed before wheelchair stage",
            "Scoliosis severe — surgical fixation commonly required",
        ],
        "treatment": [
            "Deflazacort/prednisone — Level C; may briefly prolong ambulation; start early",
            "ECHO every 6 months — ACE-I/beta-blocker early if DCM develops",
            "Annual spirometry / sleep study — NIV when FVC <50% or nocturnal hypoventilation",
            "Scoliosis — annual Cobb angle; surgical fixation if >45° (before severe respiratory compromise)",
            "Physiotherapy intensive — joint range, posture, respiratory physiotherapy",
            "Genetic counselling + founder mutation screening in North African populations",
        ],
        "contraindications": [
            "VIGOROUS EXERCISE ABSOLUTELY AVOID — most severe membrane fragility; rhabdomyolysis risk",
            "DELAY CARDIAC ASSESSMENT — every 6-month echo is mandatory; do not skip",
            "ANAESTHESIA RISK — rhabdomyolysis post-op; volatile anaesthetic monitoring",
        ],
        "critical_pearls": [
            "In North African child with rapid-onset LGMD + cardiac — SGCG del521T: first test",
            "Cardiac failure can be first presentation before overt weakness is noted",
            "Most severe sarcoglycanopathy — realistic prognosis counselling essential at diagnosis",
            "NIV may be needed before full wheelchair dependence — proactive spirometry",
        ],
    },
    # -- SGCD — δ-Sarcoglycan, LGMD-R6 -----------------------------------------------
    {
        "gene": "SGCD",
        "alt_name": (
            "SGCD (SGCD-290aa-5q33.3 / AR — LGMD-R6-delta-Sarcoglycanopathy-RAREST — "
            "DILATED-CARDIOMYOPATHY-100pct-PATHOGNOMONIC-DCM-May-Precede-Weakness — "
            "Isolated-DCM-Gene-Panel-Mandatory — "
            "Rhabdomyolysis-Episodes-CK-50-200x)"
        ),
        "protein": (
            "SGCD -- 5q33.3 AR -- SGCD-290aa -- "
            "delta-Sarcoglycan-35kDa-Type-II-Transmembrane-Dystrophin-Associated-Protein-Complex -- "
            "LGMD-R6-Formerly-LGMD2F-OMIM-601287-Disease-OMIM-601287 -- "
            "Rarest-Sarcoglycanopathy-Fewer-Than-50-Families-Worldwide -- "
            "DILATED-CARDIOMYOPATHY-Most-Severe-Cardiac-Involvement-All-Sarcoglycanopathies -- "
            "DCM-May-PRECEDE-Muscle-Weakness-By-Years-Present-As-Isolated-DCM -- "
            "CK-50-200x-ULN-HIGHEST-CK-Any-Sarcoglycanopathy-Rhabdomyolysis-Episodes -- "
            "Early-Childhood-Onset-3-8yr-Rapid-Progression -- "
            "IHC-Absent-deltaSG-Secondary-Reduction-All-SG -- "
            "Gene-Panel-In-ALL-Unexplained-Childhood-DCM-Include-SGCD -- "
            "SGCD-Knockout-Hamster-Model-Classic-DCM-Genetics-Research -- "
            "5q33.3"
        ),
        "locus": "5q33.3",
        "protein_size": "290 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic SGCD loss-of-function or missense; "
            "very rare — fewer than 50 families; no common founder; "
            "mutations scattered across gene; "
            "include SGCD in gene panel for unexplained paediatric DCM."
        ),
        "key_features": [
            "RAREST sarcoglycanopathy — fewer than 50 families worldwide",
            "DILATED CARDIOMYOPATHY — most severe cardiac involvement of all sarcoglycanopathies",
            "DCM may PRECEDE muscle weakness by years — PATHOGNOMONIC: present as isolated DCM",
            "CK 50-200× ULN — rhabdomyolysis episodes; highest CK of all sarcoglycanopathies",
            "Include SGCD in gene panel for all unexplained childhood-onset DCM",
            "IHC: absent δ-SG + secondary reduction of all SGs",
        ],
        "treatment": [
            "AGGRESSIVE CARDIAC MANAGEMENT — ACE-I (enalapril) + beta-blocker at first DCM sign",
            "ECHO every 3-6 months — cardiac trajectory determines prognosis",
            "Heart transplant evaluation early — DCM may be refractory",
            "Deflazacort/prednisone — muscle progression; cardiac effect uncertain",
            "Annual spirometry — NIV when indicated",
            "Defibrillator (ICD) consideration — sudden death risk with severe DCM",
        ],
        "contraindications": [
            "DELAY CARDIAC ASSESSMENT — DCM can be fatal if missed; 3-6 monthly echo mandatory",
            "CLASS I/III ANTIARRHYTHMICS — monitor QTc with DCM; amiodarone caution",
            "VIGOROUS EXERCISE — rhabdomyolysis risk; CK monitoring post-exercise",
        ],
        "critical_pearls": [
            "Any unexplained child or young adult DCM — add SGCD to the gene panel",
            "DCM before weakness: SGCD mimics isolated cardiomyopathy — miss the diagnosis without muscle panel",
            "Rarest sarcoglycanopathy but MOST dangerous cardiac phenotype",
            "Hamster model (BIO TO-2 cardiomyopathic hamster) — classic SGCD DCM model",
        ],
    },
    # -- FKRP — Fukutin-Related Protein, LGMD-R9 --------------------------------------
    {
        "gene": "FKRP",
        "alt_name": (
            "FKRP (FKRP-495aa-19q13.32 / AR — LGMD-R9-MDC1C-Most-Common-AR-LGMD-UK-Scandinavia — "
            "p.Leu276Ile-L276I-Founder-Mild-LGMD-Spectrum — "
            "CARDIAC-DCM-30-40pct-MANDATORY-ECHO — "
            "Calf-Hypertrophy-Prominent-RESPIRATORY-Common)"
        ),
        "protein": (
            "FKRP -- 19q13.32 AR -- FKRP-495aa -- "
            "Fukutin-Related-Protein-Golgi-Enzyme-O-Mannosylation-alpha-Dystroglycan -- "
            "LGMD-R9-Formerly-LGMD2I-OMIM-606596-Disease-OMIM-607155 -- "
            "Most-Common-AR-LGMD-UK-Norway-Scandinavia-Northern-Europe -- "
            "p.Leu276Ile-L276I-c.826C>A-Founder-Northern-European -- "
            "L276I-Homozygous-Mild-LGMD-Adult-Onset-Slower-Progression -- "
            "L276I-Heterozygous-Severe-Null-Allele-MDC1C-Congenital-Never-Walks -- "
            "Spectrum-Mild-LGMD-R9-To-Severe-MDC1C-Based-On-Allele-Severity -- "
            "CARDIAC-DCM-30-40pct-ALL-FKRP-Patients-MANDATORY-6-Month-ECHO -- "
            "RESPIRATORY-FVC-Annual-Common-More-Than-SGCA -- "
            "Calf-Hypertrophy-Prominent-Feature-85pct-Patients -- "
            "CK-10-80x-ULN-High-Even-In-Mild-Cases -- "
            "Glycosylation-Defect-Reduced-Glycosylated-Alpha-DG-IHC-VIA4-1-Antibody -- "
            "Ataluren-NOT-Applicable-Non-Sense-Suppression-Not-FKRP-Mechanism -- "
            "19q13.32"
        ),
        "locus": "19q13.32",
        "protein_size": "495 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic FKRP variants; "
            "p.Leu276Ile (L276I) — Northern European founder, most common LGMD-R9 allele; "
            "L276I/L276I homozygous = mild LGMD, adult-onset; "
            "L276I/null or null/null = severe MDC1C (congenital, never walks, intellectual disability); "
            "carrier frequency 1/400 in UK."
        ),
        "key_features": [
            "Most common AR LGMD in UK and Scandinavia (L276I founder allele)",
            "CARDIAC DCM 30-40% — ALL patients require 6-monthly cardiac echo",
            "RESPIRATORY: FVC annually — more common than SGCA/B; NIV often required",
            "Calf hypertrophy PROMINENT — 85% of patients; strong phenotypic clue",
            "L276I/L276I = mild LGMD (adult-onset); L276I/null = severe MDC1C",
            "IHC: reduced glycosylated α-DG (VIA4-1 antibody) — glycosylation defect",
        ],
        "treatment": [
            "ECHO every 6 months — ACE-I + beta-blocker for DCM; ICD if arrhythmia",
            "Annual spirometry + sleep study — NIV when FVC <50%",
            "Deflazacort/prednisone — Level C; may slow progression; same protocol as SGCA",
            "Physiotherapy — ankle orthoses, spine bracing if scoliosis develops",
            "Genetic counselling — L276I carrier testing in Northern European families",
            "Clinical trial eligibility — gene therapy and ribitol supplementation in development",
        ],
        "contraindications": [
            "MISS CARDIAC SURVEILLANCE — 6-monthly echo is mandatory; DCM is treatable if caught early",
            "CLASSIFY AS ISOLATED CARDIOMYOPATHY — without muscle panel; FKRP must be in DCM gene panel",
            "MDC1C PROGNOSIS SAME AS MILD FKRP — they are very different; allele-based counselling mandatory",
        ],
        "critical_pearls": [
            "FKRP is the most common AR LGMD gene in the UK — test in any adult AR LGMD before others",
            "Cardiac: 30-40% DCM makes FKRP the highest cardiac risk LGMD gene outside SGCD",
            "Calf hypertrophy + high CK in adult + AR LGMD = FKRP until proven otherwise",
            "MDC1C vs LGMD-R9 are the SAME gene — different allele severity; never counsel as one disease",
            "VIA4-1 IHC antibody screens for glycosylation defect — reduced staining supports FKRP dx",
        ],
    },
    # -- ANO5 — Anoctamin-5, LGMD-R12 ------------------------------------------------
    {
        "gene": "ANO5",
        "alt_name": (
            "ANO5 (ANO5-913aa-11p14.3 / AR — LGMD-R12-Miyoshi-Like-Anoctaminopathy — "
            "QUADRICEPS-SPARED-Early-PATHOGNOMONIC-DDx-DYSF — "
            "p.Arg758Cys-Dutch-Belgian-Founder — "
            "NO-Cardiac-Involvement-Critical-DDx-FKRP-SGCD)"
        ),
        "protein": (
            "ANO5 -- 11p14.3 AR -- ANO5-913aa -- "
            "Anoctamin-5-TMEM16E-Ca2+-Activated-Cl-Channel-Phospholipid-Scramblase -- "
            "LGMD-R12-Formerly-LGMD2L-OMIM-608662-Disease-OMIM-611307 -- "
            "Miyoshi-Myopathy-3-MM3-Distal-Onset-Posterior-Calf-Atrophy -- "
            "p.Arg758Cys-c.2272C>T-Dutch-Belgian-Founder-Most-Common-ANO5-Allele -- "
            "c.191dupA-c.155delA-Common-Across-Populations -- "
            "QUADRICEPS-SPARED-Early-PATHOGNOMONIC-DDx-DYSF-Tibialis-Anterior-Mild -- "
            "Posterior-Calf-Medial-Gastrocnemius-Atrophy-Selective-Posterior-Compartment -- "
            "CK-5-100x-ULN-Mean-15-30x -- "
            "NO-CARDIAC-INVOLVEMENT-CRITICAL-DDx-From-FKRP-SGCD-SGCG -- "
            "Asymmetric-Onset-Common-One-Calf-First -- "
            "Muscle-MRI-Selective-Posterior-Compartment-Lower-Leg-Medial-Gastrocnemius -- "
            "Often-Misdiagnosed-As-DYSF-LGMD-R2-Molecular-Confirmation-Required -- "
            "11p14.3"
        ),
        "locus": "11p14.3",
        "protein_size": "913 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic ANO5 variants; "
            "p.Arg758Cys (c.2272C>T) — Dutch/Belgian founder, most common ANO5 allele; "
            "c.191dupA — second most common; "
            "asymmetric muscle involvement is clinically characteristic; "
            "muscle MRI distinctive: medial gastrocnemius-predominant posterior calf."
        ),
        "key_features": [
            "QUADRICEPS SPARED early PATHOGNOMONIC — clear DDx from DYSF (quads early involved)",
            "Posterior calf (medial gastrocnemius) selective atrophy — asymmetric onset frequent",
            "p.Arg758Cys — Dutch/Belgian founder; first allele to test in Northern European LGMD",
            "CK 5-100× ULN (mean 15-30×) — markedly elevated like DYSF",
            "NO CARDIAC INVOLVEMENT — critical DDx from FKRP/SGCD/SGCG",
            "Muscle MRI: medial gastrocnemius + posterior compartment selective — guides diagnosis",
        ],
        "treatment": [
            "No disease-modifying therapy approved — physiotherapy central",
            "Posterior calf physiotherapy + ankle-foot orthoses — foot drop management",
            "Annual review — no cardiac surveillance required (no cardiac involvement)",
            "Muscle MRI guided physiotherapy — selective posterior compartment weakness targeted",
            "Genetic counselling — R758C carrier screening in Dutch/Belgian families",
            "Clinical trial watchlist — ANO5 gene therapy/exon skipping in development",
        ],
        "contraindications": [
            "CARDIAC SURVEILLANCE UNNECESSARY — unlike FKRP/SGCD, no cardiac involvement",
            "DO NOT CONFUSE WITH DYSF — treatment implications differ; quads spared = ANO5",
            "STEROIDS — not evidence-based in ANO5; avoid unless in trial context",
        ],
        "critical_pearls": [
            "ANO5 is DDx #1 for DYSF — both have distal onset, posterior calf, high CK",
            "KEY distinguisher: quads spared (ANO5) vs quads involved (DYSF)",
            "Asymmetric calf onset is ANO5 clue — DYSF tends to be more symmetric",
            "No cardiac: reassure patient; unlike FKRP where cardiac risk is major management issue",
            "Muscle MRI of calf: medial gastrocnemius selective atrophy is ANO5 pattern",
        ],
    },
    # -- TRIM32 — Tripartite Motif-32, LGMD-R8 ----------------------------------------
    {
        "gene": "TRIM32",
        "alt_name": (
            "TRIM32 (TRIM32-653aa-9q33.1 / AR — LGMD-R8-Sarcotubular-Myopathy — "
            "SARCOTUBULAR-AGGREGATES-EM-PATHOGNOMONIC — "
            "Facial-Weakness-Psychiatric-Features-Clue — "
            "Very-Rare-Fewer-100-Cases)"
        ),
        "protein": (
            "TRIM32 -- 9q33.1 AR -- TRIM32-653aa -- "
            "Tripartite-Motif-Containing-32-E3-Ubiquitin-Ligase-RING-B-Box-Coiled-Coil-NHL -- "
            "LGMD-R8-Formerly-LGMD2H-OMIM-602290-Disease-OMIM-254110 -- "
            "Sarcotubular-Myopathy-STM-Original-Name-EM-Diagnostic -- "
            "SARCOTUBULAR-AGGREGATES-Subsarcolemmal-Vacuoles-EM-PATHOGNOMONIC-Feature -- "
            "Ubiquitin-Substrates-Nebulin-Actin-Myosin-HC-Dysbindin-PIMT -- "
            "Very-Rare-Fewer-100-Families-Worldwide -- "
            "Founder-Mutations-D487N-p.Asp487Asn-Manitoba-Hutterite-Population -- "
            "Mild-Proximal-Weakness-Slow-Progression-Upper-Lower-Limb -- "
            "Facial-Weakness-Present-Unlike-Other-LGMD -- "
            "Psychiatric-Features-Schizophrenia-Like-Behaviour-Some-Patients -- "
            "CK-Mildly-Elevated-2-10x-ULN-Lower-Than-Other-LGMD -- "
            "Bardet-Biedl-Syndrome-BBS11-Allelic-Different-Mutation-Spectrum -- "
            "9q33.1"
        ),
        "locus": "9q33.1",
        "protein_size": "653 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic TRIM32 loss-of-function; "
            "p.Asp487Asn (D487N) — Manitoba Hutterite founder mutation; "
            "BBS11 (Bardet-Biedl syndrome-11) is allelic — different mutations; "
            "very rare: fewer than 100 families; CK mildly elevated (contrast with other LGMD)."
        ),
        "key_features": [
            "SARCOTUBULAR AGGREGATES on EM PATHOGNOMONIC — unique diagnostic hallmark of STM",
            "FACIAL WEAKNESS present — clinically distinguishes from most other LGMD subtypes",
            "Psychiatric features (schizophrenia-like behaviour) — some patients; important clue",
            "CK 2-10× ULN — LOWER than other LGMD subtypes; mild elevation can lead to under-investigation",
            "Manitoba Hutterite founder mutation (D487N) — test first in this population",
            "Very rare — muscle biopsy with EM is essential for diagnosis",
        ],
        "treatment": [
            "No disease-modifying therapy — physiotherapy and supportive care",
            "Psychiatric co-management if behaviour/psychiatric features present",
            "Annual review — no cardiac surveillance specifically required",
            "Genetic counselling — rare; Hutterite community population screening available",
            "Assistive devices — slow progression; ambulant into 5th/6th decade typical",
        ],
        "contraindications": [
            "PSYCHIATRIC DRUGS WITHOUT MUSCLE AWARENESS — drug interactions and NMS risk; inform treating psychiatrist",
            "MISS FACIAL WEAKNESS — rarer in LGMD; if present, consider TRIM32, FSHD, EDMD",
        ],
        "critical_pearls": [
            "EM is the key — sarcotubular aggregates are not seen on light microscopy alone",
            "Facial weakness in LGMD: think TRIM32, FSHD (asymmetric), Oculopharyngeal (ptosis)",
            "Psychiatric features reported — may precede motor symptoms; psychiatry referral + muscle panel",
            "Mildly elevated CK does not exclude LGMD — TRIM32 has lower CK than other subtypes",
        ],
    },
    # -- TCAP — Titin-Cap/Telethonin, LGMD-R7 -----------------------------------------
    {
        "gene": "TCAP",
        "alt_name": (
            "TCAP (TCAP-167aa-17q12 / AR — LGMD-R7-Telethonin-Z-Disc-LGMD2G — "
            "DILATED-CARDIOMYOPATHY-Cardiac-Involvement-Significant — "
            "RIMMED-VACUOLES-Biopsy-Clue — "
            "Uruguay-Brazil-China-Founder-Mutations-Extremely-Rare)"
        ),
        "protein": (
            "TCAP -- 17q12 AR -- TCAP-167aa -- "
            "Titin-Cap-Telethonin-19kDa-Z-Disc-Structural-Protein-Binds-Titin-N-Terminus -- "
            "LGMD-R7-Formerly-LGMD2G-OMIM-604488-Disease-OMIM-601954 -- "
            "Z-Disc-Component-Sarcomere-Structural-Integrity-Mechano-Sensing -- "
            "Binds-Titin-N-Terminal-Z1-Z2-Domains-Anti-Parallel-Beta-Sheet -- "
            "Very-Rare-Fewer-50-Patients-Worldwide -- "
            "Founder-Mutations-Brazil-Uruguay-China-Specific-Variants -- "
            "DILATED-CARDIOMYOPATHY-Cardiac-Involvement-Major-Feature -- "
            "RIMMED-VACUOLES-Muscle-Biopsy-Light-Microscopy-Clue -- "
            "Variable-Phenotype-Mild-LGMD-To-Severe-DCM-Predominant -- "
            "Proximal-Lower-Limb-Weakness-Hip-Thigh-Involved -- "
            "CK-5-30x-ULN-Moderate-Elevation -- "
            "Titin-Interaction-Mechano-Sensing-Pathway-Cardiac-Skeletal -- "
            "17q12"
        ),
        "locus": "17q12",
        "protein_size": "167 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic TCAP variants; "
            "population-specific founders in Uruguay, Brazil, China; "
            "very rare globally — fewer than 50 patients; "
            "variable phenotype; dilated cardiomyopathy may be the dominant feature; "
            "titin interaction disrupted → mechano-sensing failure → cardiomyopathy."
        ),
        "key_features": [
            "RAREST LGMD — fewer than 50 patients worldwide; predominantly case reports",
            "DILATED CARDIOMYOPATHY — significant cardiac feature; may dominate the phenotype",
            "RIMMED VACUOLES on muscle biopsy — diagnostic clue on light microscopy",
            "Uruguay/Brazil/China founder mutations — population-specific testing first",
            "Z-disc protein: disrupts titin interaction → mechano-sensing failure",
            "Variable severity: mild proximal LGMD to DCM-predominant presentation",
        ],
        "treatment": [
            "CARDIAC MANAGEMENT FIRST — ACE-I + beta-blocker for DCM; ICD if arrhythmia",
            "ECHO every 6 months — cardiac trajectory determines prognosis",
            "Physiotherapy — proximal lower limb weakness support",
            "No disease-modifying muscle therapy available",
            "Genetic counselling — population-specific; carrier testing in affected families",
            "Clinical watchlist — titin pathway therapies under research",
        ],
        "contraindications": [
            "MISS CARDIAC ASSESSMENT — DCM may be dominant; cardiac evaluation at diagnosis mandatory",
            "ASSUME BENIGN MUSCLE DISEASE — cardiac risk is real; full cardiac workup required",
        ],
        "critical_pearls": [
            "Rimmed vacuoles + proximal LGMD + DCM = consider TCAP",
            "Extremely rare: diagnosis usually through WES panel; IHC not routinely available",
            "Cardiac management is the highest priority — DCM prognosis drives mortality",
            "Uruguay/Brazil/China patients with LGMD + DCM — TCAP is top differential",
        ],
    },
]


def _make_patients(gene_data, seed, n=40):
    """Generate deterministic synthetic patient records for one LGMD gene."""
    rng = random.Random(seed)
    gene = gene_data["gene"]
    locus = gene_data["locus"]
    inh = gene_data["inheritance"]
    ad = "AR" not in inh.split(";")[0].upper()[:10]

    # Age at onset varies by condition
    if gene == "SGCG":
        ages = [rng.randint(3, 12) for _ in range(n)]   # most severe; earliest onset
    elif gene == "SGCD":
        ages = [rng.randint(3, 15) for _ in range(n)]   # early, cardiac may precede
    elif gene in ("SGCA", "SGCB"):
        ages = [rng.randint(5, 20) for _ in range(n)]
    elif gene == "FKRP":
        ages = [rng.randint(5, 40) for _ in range(n)]   # L276I adult-onset spectrum
    elif gene == "ANO5":
        ages = [rng.randint(20, 55) for _ in range(n)]  # typically adult onset
    elif gene == "TRIM32":
        ages = [rng.randint(10, 45) for _ in range(n)]
    else:  # TCAP
        ages = [rng.randint(5, 40) for _ in range(n)]

    # Treatment response
    if gene == "SGCA":
        treated_fraction = 0.65  # corticosteroids Level C
    elif gene == "SGCB":
        treated_fraction = 0.60
    elif gene == "SGCG":
        treated_fraction = 0.55  # severe; treatment less effective
    elif gene == "SGCD":
        treated_fraction = 0.70  # cardiac tx high compliance
    elif gene == "FKRP":
        treated_fraction = 0.72  # cardiac + steroids
    elif gene == "ANO5":
        treated_fraction = 0.40  # no disease-modifying Tx
    elif gene == "TRIM32":
        treated_fraction = 0.35  # no disease-modifying Tx
    else:  # TCAP
        treated_fraction = 0.65  # cardiac tx

    treated = [rng.random() < treated_fraction for _ in range(n)]

    # Alive fraction
    if gene == "SGCG":
        alive_fraction = 0.88  # most severe; cardiac + respiratory
    elif gene == "SGCD":
        alive_fraction = 0.85  # severe DCM
    elif gene == "FKRP":
        alive_fraction = 0.90  # DCM mortality
    elif gene == "TCAP":
        alive_fraction = 0.88  # DCM mortality
    else:
        alive_fraction = 0.95

    alive = [rng.random() < alive_fraction for _ in range(n)]

    # Sex distribution — AR, roughly equal
    if gene == "SGCA":
        female_fraction = 0.50
    elif gene == "ANO5":
        female_fraction = 0.45  # slightly male predominant in reports
    else:
        female_fraction = 0.50

    # Attack frequency (muscle crisis/rhabdomyolysis episodes per year)
    if gene in ("SGCG", "SGCD"):
        attack_freqs = [rng.randint(0, 6) for _ in range(n)]
    elif gene in ("SGCA", "SGCB", "FKRP"):
        attack_freqs = [rng.randint(0, 3) for _ in range(n)]
    else:
        attack_freqs = [rng.randint(0, 2) for _ in range(n)]

    patients = []
    for i in range(n):
        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "seed": seed,
            "age": ages[i],
            "sex": "F" if rng.random() < female_fraction else "M",
            "alive": alive[i],
            "treated": treated[i],
            "attacks_per_year": attack_freqs[i],
            "inheritance": "AR",
            "locus": locus,
        })
    return patients


def overview():
    all_patients = []
    for i, gd in enumerate(LGMD_GENES):
        all_patients += _make_patients(gd, SEED_BASE + i)

    n = len(all_patients)
    alive_n = sum(1 for p in all_patients if p["alive"])
    treated_n = sum(1 for p in all_patients if p["treated"])

    gene_summaries = {}
    for i, gd in enumerate(LGMD_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        gene_summaries[gd["gene"]] = {
            "n_patients": len(pts),
            "alive_pct": round(100 * sum(p["alive"] for p in pts) / len(pts)),
            "treated_pct": round(100 * sum(p["treated"] for p in pts) / len(pts)),
            "mean_onset_age": round(sum(p["age"] for p in pts) / len(pts), 1),
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "lgmd_class": {
                "SGCA": "LGMD-R3",
                "SGCB": "LGMD-R4",
                "SGCG": "LGMD-R5",
                "SGCD": "LGMD-R6",
                "FKRP": "LGMD-R9",
                "ANO5": "LGMD-R12",
                "TRIM32": "LGMD-R8",
                "TCAP": "LGMD-R7",
            }.get(gd["gene"], "?"),
            "cardiac_risk": {
                "SGCA": "low",
                "SGCB": "low",
                "SGCG": "moderate",
                "SGCD": "high-DCM up to 100%",
                "FKRP": "high-DCM 30-40%",
                "ANO5": "none",
                "TRIM32": "low",
                "TCAP": "high-DCM significant",
            }.get(gd["gene"], "?"),
        }

    return {
        "atlas": "Hereditary Sarcoglycanopathy & LGMD Atlas",
        "subtitle": "Complete 8-Gene LGMD-R3/R4/R5/R6/R7/R8/R9/R12 Reference",
        "total_patients": n,
        "alive_pct": round(100 * alive_n / n),
        "treated_pct": round(100 * treated_n / n),
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "n_genes": len(LGMD_GENES),
        "genes": [g["gene"] for g in LGMD_GENES],
        "lgmd_classes": ["LGMD-R3", "LGMD-R4", "LGMD-R5", "LGMD-R6", "LGMD-R9", "LGMD-R12", "LGMD-R8", "LGMD-R7"],
        "gene_summaries": gene_summaries,
        "key_clinical_facts": {
            "most_severe": "SGCG — earliest onset (3-10yr), fastest progression, highest CK (20-100x)",
            "most_common_ar_lgmd_uk": "FKRP (L276I founder) — most common AR LGMD in UK/Scandinavia",
            "highest_cardiac_risk": "SGCD — DCM up to 100%; FKRP — DCM 30-40%; TCAP — DCM significant",
            "no_cardiac_risk": "ANO5 — no cardiac involvement (critical DDx from FKRP/SGCD)",
            "ihc_guides_gene": "IHC panel (all 4 SGs + glyco-αDG) — primary gene absent, others reduced",
            "key_ddx": "ANO5 vs DYSF: quads spared = ANO5; quads involved = DYSF",
            "founder_populations": "SGCG del521T (N.Africa), SGCB C283Y (N.Africa/M.East), FKRP L276I (N.Europe), ANO5 R758C (Dutch/Belgian), TRIM32 D487N (Hutterite)",
            "gene_therapy_pipeline": "SGCA rAAVrh74.MCK.SGCA Phase 1-2 trials underway",
        },
    }


def breakdown():
    result = {}
    for i, gd in enumerate(LGMD_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        gene = gd["gene"]
        result[gene] = {
            "n_patients": len(pts),
            "alive_pct": round(100 * sum(p["alive"] for p in pts) / len(pts)),
            "treated_pct": round(100 * sum(p["treated"] for p in pts) / len(pts)),
            "mean_age": round(sum(p["age"] for p in pts) / len(pts), 1),
            "female_pct": round(100 * sum(1 for p in pts if p["sex"] == "F") / len(pts)),
            "mean_attacks_per_year": round(sum(p["attacks_per_year"] for p in pts) / len(pts), 1),
            "key_features": gd["key_features"],
            "treatment": gd["treatment"],
            "contraindications": gd["contraindications"],
            "critical_pearls": gd["critical_pearls"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance": gd["inheritance"][:160],
        }
    return result


def definitions():
    gene_defs = {}
    for gd in LGMD_GENES:
        gene_defs[gd["gene"]] = {
            "full_name": gd["protein"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance_detail": gd["inheritance"],
            "alt_name": gd["alt_name"],
        }

    return {
        "gene_definitions": gene_defs,
        "lgmd_classification_2017": {
            "LGMD-R3": "SGCA — α-sarcoglycanopathy (formerly LGMD2D)",
            "LGMD-R4": "SGCB — β-sarcoglycanopathy (formerly LGMD2E)",
            "LGMD-R5": "SGCG — γ-sarcoglycanopathy (formerly LGMD2C) — most severe",
            "LGMD-R6": "SGCD — δ-sarcoglycanopathy (formerly LGMD2F) — highest cardiac risk",
            "LGMD-R7": "TCAP — telethonin/titin-cap (formerly LGMD2G)",
            "LGMD-R8": "TRIM32 — sarcotubular myopathy (formerly LGMD2H)",
            "LGMD-R9": "FKRP — fukutin-related protein (formerly LGMD2I) — most common AR LGMD UK",
            "LGMD-R12": "ANO5 — anoctamin-5 (formerly LGMD2L) — quads spared, no cardiac",
        },
        "ihc_pattern_table": {
            "primary_SGCA_absent": "α-SG absent; β/γ/δ-SG reduced (secondary)",
            "primary_SGCB_absent": "β-SG absent; α/γ/δ-SG reduced (secondary)",
            "primary_SGCG_absent": "γ-SG absent; α/β/δ-SG reduced (secondary)",
            "primary_SGCD_absent": "δ-SG absent; α/β/γ-SG reduced (secondary)",
            "FKRP_IHC": "Reduced glycosylated α-DG (VIA4-1 antibody); SG normal",
            "ANO5_IHC": "Usually normal — diagnosis by genetic panel",
            "TRIM32_IHC": "Normal light microscopy; EM shows sarcotubular aggregates",
            "TCAP_IHC": "Rimmed vacuoles on light microscopy; IHC not routine",
        },
        "cardiac_surveillance_table": {
            "SGCA": "Annual ECG + echo (cardiac rare but baseline required)",
            "SGCB": "Annual ECG + echo (cardiac less frequent)",
            "SGCG": "ECG + echo every 6 months (DCM significant risk)",
            "SGCD": "ECG + echo every 3-6 months (DCM up to 100% — highest urgency)",
            "FKRP": "ECG + echo every 6 months (DCM 30-40%)",
            "ANO5": "No routine cardiac surveillance (no cardiac involvement)",
            "TRIM32": "Annual ECG (no major cardiac risk defined)",
            "TCAP": "ECG + echo every 6 months (DCM significant)",
        },
        "ddx_table": {
            "ANO5_vs_DYSF": "ANO5: quads spared, posterior calf, no cardiac. DYSF: quads involved, CK very high, no cardiac.",
            "SGCA_vs_DMD": "SGCA: AR biallelic, IHC absent α-SG. DMD: X-linked, absent dystrophin on IHC/WB.",
            "SGCD_vs_cardiomyopathy": "SGCD: DCM may precede weakness; CK elevated; muscle biopsy/gene panel in unexplained DCM.",
            "FKRP_mild_vs_MDC1C": "FKRP L276I/L276I = mild LGMD-R9; FKRP null/null = congenital MDC1C — same gene, allele severity determines phenotype.",
            "TRIM32_vs_FSHD": "TRIM32: AR, facial weakness, sarcotubular aggregates EM. FSHD: AD, asymmetric, D4Z4 contraction.",
            "SGCG_vs_DMD": "SGCG: AR biallelic, LGMD-R5, del521T N.Africa. DMD: XLR, boys, dystrophin absent.",
        },
        "founder_mutations": {
            "SGCG_del521T": "SGCG c.525delT — North Africa (Algeria/Tunisia/Morocco); allele frequency up to 1/50",
            "SGCB_C283Y": "SGCB p.Cys283Tyr — North Africa/Middle East",
            "FKRP_L276I": "FKRP p.Leu276Ile — Northern Europe/UK; carrier frequency ~1/400",
            "ANO5_R758C": "ANO5 p.Arg758Cys — Dutch/Belgian",
            "TRIM32_D487N": "TRIM32 p.Asp487Asn — Manitoba Hutterite community",
            "TCAP_Brazil_Uruguay": "TCAP various mutations — Brazil/Uruguay/China (very rare globally)",
        },
        "glossary": {
            "Sarcoglycan_complex": "Heterotetrameric (α+β+γ+δ) transmembrane glycoprotein complex; part of dystrophin-associated protein complex; stabilises sarcolemma during contraction",
            "IHC_secondary_reduction": "When one SG absent → others secondarily reduced; absent SG = primary gene; guides gene panel selection",
            "DAPC": "Dystrophin-Associated Protein Complex — includes dystrophin, dystroglycans, sarcoglycans, syntrophins; mutations cause muscular dystrophies",
            "alpha_dystroglycan_glycosylation": "α-DG requires O-mannosylation; FKRP is a glycosyltransferase; reduced glyco-αDG = FKRP/fukutin/POMT1 group",
            "MDC1C": "Merosin-Deficient Congenital Muscular Dystrophy type 1C — severe FKRP null phenotype; never ambulant; intellectual disability",
            "Sarcotubular_myopathy": "Histopathological pattern = subsarcolemmal vacuoles on EM; specific to TRIM32 (LGMD-R8)",
            "Rimmed_vacuoles": "Vacuoles on muscle biopsy with basophilic rims; seen in TCAP, GNE myopathy, IBM",
            "CK_sarcoglycanopathy": "CK: SGCA 10-70x, SGCB 10-50x, SGCG 20-100x (highest), SGCD 50-200x (highest), FKRP 10-80x, ANO5 5-100x, TRIM32 2-10x (lowest), TCAP 5-30x",
            "Pseudohypertrophy": "Calf enlargement due to fat+connective tissue replacing muscle; seen in DMD, sarcoglycanopathies, FKRP; FKRP: prominent calf hypertrophy 85%",
            "LGMD_R_classification": "2017 LGMD Workshop reclassification: R = recessive (AR); D = dominant (AD); numbered by gene discovery order",
            "Titin_Z-disc": "Titin (TTN) spans from Z-disc to M-line; TCAP (telethonin) binds titin N-terminus at Z-disc; mechano-sensing complex",
            "Gene_therapy_rAAVrh74": "Recombinant AAV serotype rh74 — targets muscle; MCK promoter = muscle-specific; SGCA rAAVrh74.MCK.SGCA in Phase 1-2 trials",
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = overview()
    print(json.dumps({k: v for k, v in ov.items() if k != "gene_summaries"}, indent=2))
    print("\n=== BREAKDOWN keys ===")
    br = breakdown()
    for gene, data in br.items():
        print(f"  {gene}: {data['n_patients']} patients, alive={data['alive_pct']}%, treated={data['treated_pct']}%")
    print("\n=== DEFINITIONS keys ===")
    defs = definitions()
    print(list(defs.keys()))
