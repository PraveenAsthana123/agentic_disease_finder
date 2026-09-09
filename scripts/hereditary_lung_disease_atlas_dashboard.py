#!/usr/bin/env python3
"""Hereditary-Lung-Disease-Atlas — Complete 8-Gene Hereditary Lung Disease Atlas
CFTR    (cystic fibrosis transmembrane conductance regulator; 1480 aa; 7q31.2; AR;
         Cystic Fibrosis [CF] — F508del 70% Europeans; Trikafta FDA 2019 ≥12y;
         sweat chloride ≥60 mmol/L DIAGNOSTIC; Burkholderia cepacia complex = transplant risk;
         seed SEED_BASE+0) ·
SERPINA1 (alpha-1 antitrypsin; 418 aa; 14q32.13; AR/codominant;
         Alpha-1 Antitrypsin Deficiency [AATD] — PiZZ 1:3500 Northern European;
         lower lobe emphysema (reverse of smoking!); augmentation therapy LUNG ONLY (NOT liver);
         Pi phenotyping by IEF mandatory; smoking ABSOLUTE PROHIBITION;
         seed SEED_BASE+1) ·
SFTPC   (surfactant protein C; 197 aa; 8p21.3; AD dominant-negative;
         Familial ILD / SP-C deficiency — I73T most common mutation;
         neonatal RDS to adult ILD; hydroxychloroquine + azithromycin first-line;
         seed SEED_BASE+2) ·
TERT    (telomerase reverse transcriptase; 1132 aa; 5p15.33; AD;
         Telomere Biology Disorders / Familial IPF / Dyskeratosis Congenita;
         telomere length <1st percentile DIAGNOSTIC; IPF + aplastic anemia + liver cirrhosis triad;
         immunosuppression CONTRAINDICATED; transplant preferred;
         seed SEED_BASE+3) ·
FLCN    (folliculin; 579 aa; 17p11.2; AD;
         Birt-Hogg-Dubé Syndrome — fibrofolliculomas PATHOGNOMONIC 70-90%;
         spontaneous pneumothorax 25-33% lifetime; chromophobe/oncocytoma RCC;
         NO aviation/scuba/high-altitude ABSOLUTE PROHIBITION;
         seed SEED_BASE+4) ·
TSC1    (tuberous sclerosis complex 1 / hamartin; 1164 aa; 9q34.13; AD;
         Tuberous Sclerosis Complex — LAM exclusively females;
         sirolimus FDA 2015 for LAM; VEGF-D >800 pg/mL PATHOGNOMONIC;
         FEV1 decline 75-100 mL/y without therapy;
         seed SEED_BASE+5) ·
ABCA3   (ATP-binding cassette sub-family A member 3; 1704 aa; 16p13.3; AR;
         Surfactant Metabolism Dysfunction Type 3 — neonatal RDS (null/null) to adult ILD;
         lamellar bodies ABSENT/abnormal on EM DIAGNOSTIC;
         seed SEED_BASE+6) ·
NKX2-1  (NK2 homeobox 1 / TTF-1; 371 aa; 14q13.3; AD;
         Brain-Lung-Thyroid Syndrome — TRIAD: BHC + congenital hypothyroidism + ILD PATHOGNOMONIC;
         haploinsufficiency; annual pulmonary function tests;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1710–1717)
"""

import random

SEED_BASE = 1710

LUNG_GENES = [
    # ── CFTR — Cystic Fibrosis ───────────────────────────────────────────────
    {
        "gene": "CFTR",
        "protein": "CFTR — 7q31.2 AR — CFTR-Regulator-1480aa — Cystic-Fibrosis-F508del-70pct — Trikafta-FDA2019-Elexacaftor-Tezacaftor-Ivacaftor — Sweat-Cl-60mmolL-DIAGNOSTIC — Burkholderia-Transplant-Risk — CFRD-40pct-Adults",
        "alias": (
            "CFTR (cystic fibrosis transmembrane conductance regulator); OMIM gene 602421; "
            "Cystic Fibrosis (CF) OMIM 219700. "
            "7q31.2; 1480 aa; ~168 kDa; autosomal recessive. "
            "FUNCTION: CFTR is an ATP-gated chloride channel expressed at the apical surface of epithelial cells "
            "in airways, pancreas, intestine, sweat glands, and reproductive tract. "
            "It mediates Cl- and HCO3- transport, regulating hydration and pH of secretions. "
            "CFTR LOF → thickened, dehydrated mucus → impaired mucociliary clearance → "
            "chronic bacterial colonisation → bronchiectasis → progressive lung destruction. "
            "MUTATION CLASSES: "
            "Class I: no protein (nonsense, frameshift) — most severe; "
            "Class II: misfolding/ER retention — F508del (Phe508del) is the archetypal class II, "
            "present in ~70% of European CF chromosomes (~85% of CF patients have at least one F508del); "
            "Class III: gating defects — G551D (~4% CF), residual protein at membrane but channel does not open; "
            "Class IV: reduced conductance; Class V: reduced synthesis; Class VI: reduced stability; "
            "CLINICAL FEATURES: "
            "Pulmonary: chronic productive cough, bronchiectasis, recurrent exacerbations; "
            "bacterial progression: S. aureus (early) → H. influenzae → P. aeruginosa (chronic) → "
            "Burkholderia cepacia complex (BCC, late/end-stage — MAJOR TRANSPLANT CONCERN); "
            "DIAGNOSIS: sweat chloride ≥60 mmol/L by Gibson-Cooke pilocarpine iontophoresis = DIAGNOSTIC; "
            "40-59 mmol/L = intermediate; <40 mmol/L = normal (but genetic variants can still cause CF); "
            "CFTR genetic testing for 2 pathogenic variants confirms diagnosis; "
            "Newborn screening by IRT (immunoreactive trypsinogen) + DNA in most countries; "
            "EXTRAPULMONARY: "
            "Pancreatic exocrine insufficiency (PI) in 85% — requires pancreatic enzyme replacement therapy (PERT); "
            "CFRD (CF-related diabetes) in 40-50% of adults — insulin required; not type 1 or 2; "
            "Male infertility (CBAVD — congenital bilateral absence of vas deferens) in ~98% males; "
            "hepatobiliary: CF liver disease/cirrhosis in ~5-10%; "
            "nasal polyps, chronic sinusitis; "
            "CFTR MODULATOR THERAPY — TRANSFORMATIVE: "
            "Trikafta (elexacaftor/tezacaftor/ivacaftor) — FDA approved October 2019 for F508del ≥12y; "
            "subsequently extended to ≥6y and ≥2y (2023); "
            "mechanism: elexacaftor+tezacaftor = correctors (chaperone F508del protein to membrane); "
            "ivacaftor = potentiator (opens CFTR channel gate); "
            "clinical impact: ppFEV1 improvement ~14 percentage points; sweat chloride normalisation; "
            "BMI gain; dramatic reduction in exacerbations; "
            "earlier agents: ivacaftor (Kalydeco) alone for G551D and other gating mutations; "
            "lumacaftor/ivacaftor (Orkambi) for F508del homozygous (modest benefit); "
            "BURKHOLDERIA CEPACIA COMPLEX (BCC): "
            "BCC colonisation = RELATIVE CONTRAINDICATION to lung transplant at many centres; "
            "B. cenocepacia (genomovar III) = worst prognosis; cepacia syndrome (necrotising pneumonia) fatal; "
            "strict infection control — BCC patients segregated from non-BCC; "
            "ANNUAL MONITORING: spirometry (FEV1, FVC), sputum culture, CFRD screen, liver USS, bone density."
        ),
        "locus": "7q31.2",
        "aa": 1480,
        "kDa": 168,
        "omim_gene": "602421",
        "omim_disease": "219700",
        "inheritance": "AR — Autosomal Recessive — biallelic LOF; F508del most common (~70% European chromosomes); 2000+ pathogenic variants; compound heterozygous very common; newborn screening via IRT + DNA",
        "gene_class": "ATP-gated Cl-/HCO3- channel (ABC transporter superfamily, ABCC7); apical epithelial surface; mucus hydration and airway surface liquid regulation; 12 TM domains + 2 NBDs + regulatory R domain",
        "key_alerts": [
            "CFTR-TRIKAFTA-FDA2019-F508DEL-TRANSFORMATIVE: elexacaftor/tezacaftor/ivacaftor approved 2019 for F508del ≥12y (now ≥2y); ppFEV1 improves ~14pp; sweat Cl normalises; exacerbations fall dramatically; eligible ~90% CF patients; check CFTR genotype before prescribing — non-F508del mutations may require alternative modulators",
            "CFTR-SWEAT-CHLORIDE-60-DIAGNOSTIC: sweat chloride ≥60 mmol/L = DIAGNOSTIC for CF (Gibson-Cooke pilocarpine iontophoresis); 40-59 = intermediate (further evaluation needed); false negatives in newborns <48h; confirmatory genetic CFTR testing for 2 pathogenic variants; do NOT rely on sweat chloride alone if clinical suspicion high",
            "CFTR-BURKHOLDERIA-CEPACIA-TRANSPLANT-RISK: BCC colonisation, especially B. cenocepacia (genomovar III), is a RELATIVE CONTRAINDICATION to lung transplant at most centres; cepacia syndrome (bilateral necrotising pneumonia + septicaemia) is rapidly fatal; strict patient segregation mandatory (separate clinic days, rooms, equipment); NEVER mix BCC and non-BCC patients",
            "CFTR-CFRD-INSULIN-NOT-TYPE1-OR-2: CF-related diabetes in 40-50% adults; pathophysiology = beta-cell destruction from pancreatic fibrosis + exocrine failure (NOT autoimmune like T1DM, NOT insulin-resistant like T2DM); screening: annual OGTT from age 10; insulin is treatment of choice; sulfonylureas and metformin generally NOT recommended in CF",
            "CFTR-PANCREATIC-ENZYME-REPLACEMENT-85PCT-PI: 85% of CF patients are pancreatic insufficient; PERT (pancrelipase) with every meal and snack; fat-soluble vitamins A/D/E/K supplementation; caloric targets 120-150% normal; malnutrition accelerates lung decline; BMI tracking essential",
            "CFTR-ANNUAL-SPIROMETRY-SPUTUM-CULTURE: FEV1/FVC monitoring every 3-6 months; sputum culture for bacterial/fungal pathogens quarterly; exacerbation defined as ≥2 of: increased cough, sputum, dyspnoea, fatigue, fever, FEV1 decline; IV antibiotics for severe exacerbations targeting P. aeruginosa",
            "CFTR-MALE-INFERTILITY-CBAVD-98PCT: congenital bilateral absence of vas deferens in ~98% males with CF; sperm production intact (testicular biopsy viable); ICSI/IVF achievable; counsel adolescents/young adults early; CFTR sequencing of female partner mandatory before IVF",
            "CFTR-NEWBORN-SCREENING-IRT-DNA: IRT (immunoreactive trypsinogen) elevated at birth in CF; followed by CFTR DNA analysis; early diagnosis + early PERT + early physiotherapy improves nutritional and lung outcomes; false-positive IRT in prematurity and other neonatal illness",
        ],
        "etiologies": {
            "F508del_homozygous": {"pct": 45, "phenotype": "classic_severe_PI", "notes": "most common; PI; bronchiectasis; Trikafta eligible; sweat Cl >80"},
            "F508del_compound_het_class2": {"pct": 25, "phenotype": "classic_moderate_severe", "notes": "F508del + second class I/II variant; PI common; Trikafta eligible most"},
            "Gating_G551D_class3": {"pct": 4, "phenotype": "variable_PS_possible", "notes": "ivacaftor monotherapy (Kalydeco) highly effective; sweat Cl normalises"},
            "Mild_class4_5_compound": {"pct": 12, "phenotype": "mild_PS_CBAVD", "notes": "pancreatic sufficient; lung disease mild-moderate; CBAVD main manifestation"},
            "Severe_class1_nonsense": {"pct": 14, "phenotype": "severe_classic", "notes": "W1282X, G542X common; no CFTR protein; PTB ataluren trials; Trikafta partial benefit some"},
        },
        "stats": {
            "incidence_Northern_European": "1 in 2,500 live births",
            "carrier_frequency_Northern_European": "1 in 25",
            "mean_dx_age_newborn_screen_y": 0.1,
            "mean_dx_age_clinical_y": 2.5,
            "pancreatic_insufficiency_pct": 85,
            "CFRD_adults_pct": 45,
            "median_survival_current_y": 53,
        },
        "dx_delay_distribution": {"mean_months": 8, "median_months": 2, "range": "0-60", "notes": "newborn screening reduces delay; adult-onset CF (mild variants) still delayed 5-10y; atypical CF (single organ) delayed longest"},
    },

    # ── SERPINA1 — Alpha-1 Antitrypsin Deficiency ────────────────────────────
    {
        "gene": "SERPINA1",
        "protein": "SERPINA1 — 14q32.13 AR-Codominant — Alpha1-Antitrypsin-418aa — AATD-PiZZ-1:3500 — Lower-Lobe-Emphysema-REVERSED — Augmentation-Therapy-LUNG-ONLY-NOT-Liver — Pi-IEF-Phenotyping-MANDATORY — Smoking-ABSOLUTE-PROHIBITION",
        "alias": (
            "SERPINA1 (serpin peptidase inhibitor, clade A, member 1); OMIM gene 107400; "
            "Alpha-1 Antitrypsin Deficiency (AATD) OMIM 613490. "
            "14q32.13; 418 aa; ~52 kDa (mature); autosomal recessive / codominant. "
            "FUNCTION: Alpha-1 antitrypsin (AAT) is the primary serine protease inhibitor (serpin) in plasma, "
            "produced predominantly by hepatocytes (~80%) and alveolar macrophages. "
            "AAT's main target is neutrophil elastase (NE): AAT irreversibly inhibits NE → "
            "prevents elastin destruction in alveolar walls. "
            "AATD pathomechanism — TWO DISTINCT MECHANISMS: "
            "(1) LUNG: reduced serum AAT → uninhibited NE → elastin destruction → emphysema; "
            "(2) LIVER: misfolded Z-AAT polymer accumulates in ER of hepatocytes → ER stress → hepatocyte death → cirrhosis. "
            "NOMENCLATURE: Pi (protease inhibitor) system: "
            "PiMM = normal (AAT level 100%); "
            "PiMS = carrier (~80%); PiMZ = carrier (~60%); "
            "PiZZ = severe deficiency (AAT ~10-15% of normal) — most common severe genotype; "
            "PiSZ = intermediate (~40%); PiSS = mild deficiency (~60%); "
            "PiNULL = no protein produced — severe lung disease, NO liver disease; "
            "CLINICAL — LUNG DISEASE: "
            "Lower lobe predominant emphysema — OPPOSITE of smoking emphysema (upper lobe); "
            "this basal predominance is PATHOGNOMONIC for AATD; "
            "panacinar (panlobular) emphysema on HRCT; "
            "bronchiectasis in ~30%; "
            "onset: non-smokers 45-55y; smokers 35-45y; "
            "CLINICAL — LIVER DISEASE: "
            "Neonatal cholestasis (~10% neonates with PiZZ); "
            "childhood liver disease in ~15% PiZZ; "
            "adult cirrhosis in ~15% PiZZ adults (NOT all); "
            "HCC risk in cirrhotic PiZZ; "
            "liver transplant cures liver disease and restores normal AAT phenotype; "
            "DIAGNOSIS: "
            "Serum AAT level (nephelometry): <11 µmol/L (80 mg/dL) suggests deficiency; "
            "Pi PHENOTYPING by isoelectric focusing (IEF) gel = MANDATORY — identifies alleles directly; "
            "SERPINA1 genotyping (Z/S common allele assay ± full sequencing); "
            "AUGMENTATION THERAPY: "
            "IV infusion of pooled human plasma AAT (Prolastin-C, Zemaira, Glassia, Aralast NP); "
            "weekly IV; maintains serum AAT >11 µmol/L; "
            "LUNG BENEFIT: slows CT emphysema progression (RAPID trial); "
            "NO LIVER BENEFIT — augmentation replaces secreted AAT, not intracellular polymerised Z-AAT; "
            "liver disease requires separate management; "
            "SMOKING: smoking accelerates emphysema 15-20 years earlier; "
            "ABSOLUTE SMOKING PROHIBITION — the single most important intervention; "
            "passive smoke exposure also harmful; "
            "NOVEL THERAPIES: "
            "Fazirsiran (ARO-AAT) — RNA interference targeting liver AAT production (reduces Z-polymer load); "
            "gene therapy trials (AAV-SERPINA1) — ongoing."
        ),
        "locus": "14q32.13",
        "aa": 418,
        "kDa": 52,
        "omim_gene": "107400",
        "omim_disease": "613490",
        "inheritance": "AR / Codominant — PiZZ = severe; PiSZ = intermediate; PiMZ = carrier elevated liver risk; each parent of PiZZ child is obligate PiMZ; codominant = both alleles expressed (phenotyping by IEF identifies each)",
        "gene_class": "Serine protease inhibitor (serpin superfamily); reactive centre loop (RCL) mechanism; irreversibly inhibits neutrophil elastase, proteinase 3, cathepsin G; acute-phase reactant (CRP-like); hepatocyte-secreted",
        "key_alerts": [
            "SERPINA1-SMOKING-ABSOLUTE-PROHIBITION: smoking in PiZZ accelerates emphysema onset by 15-20 years (non-smoker PiZZ emphysema at 45-55y vs smoker at 30-40y); ABSOLUTE PROHIBITION — single most impactful intervention; passive smoke exposure also harmful; document smoking status every visit",
            "SERPINA1-LOWER-LOBE-EMPHYSEMA-PATHOGNOMONIC: basal/lower lobe predominant panacinar emphysema on HRCT is PATHOGNOMONIC for AATD; smoking emphysema is upper lobe; young patient with lower lobe emphysema or panacinar emphysema without smoking history → AATD testing mandatory",
            "SERPINA1-AUGMENTATION-LUNG-ONLY-NOT-LIVER: IV AAT augmentation therapy slows CT emphysema progression (RAPID trial, ERS guideline); it does NOT benefit liver disease — augmented AAT is secreted normally but does NOT reduce intrahepatic Z-polymer accumulation; liver disease managed separately",
            "SERPINA1-Pi-IEF-PHENOTYPING-MANDATORY: serum AAT level alone is INSUFFICIENT — AAT is an acute-phase reactant and may be falsely normal in inflammation even in PiZZ; Pi isoelectric focusing gel phenotyping identifies Z/S/M/null alleles directly; phenotyping + genotyping together = gold standard",
            "SERPINA1-PIZZ-THRESHOLD-11-MICROMOL: serum AAT <11 µmol/L (<80 mg/dL) = protective threshold for lung; augmentation therapy targets trough >11 µmol/L; PiZZ baseline typically 3-7 µmol/L; PiMZ (~60%) rarely needs augmentation for lung but has increased liver fibrosis risk",
            "SERPINA1-LIVER-DISEASE-SEPARATE-MANAGEMENT: ~15% PiZZ adults develop cirrhosis; annual LFTs + liver USS + AFP in PiZZ with liver disease; liver biopsy if transaminases persistently elevated; liver transplant = CURE (graft converts patient to donor's Pi phenotype); fazirsiran (RNAi) reduces liver Z-polymer in trials",
            "SERPINA1-CASCADE-TESTING-FAMILY: identification of PiZZ index case → test all first-degree relatives; PiMZ parents need counselling (liver risk, smoking prohibition); genetic counselling for reproductive planning; genetic testing preferred over phenotyping for prenatal diagnosis",
            "SERPINA1-BRONCHIECTASIS-30PCT-SPIROMETRY: ~30% AATD patients develop bronchiectasis; airflow obstruction (obstructive pattern) may coexist with emphysema; annual spirometry; pulmonary rehab for FEV1 <80%; lung transplant evaluation when FEV1 <25-30% or rapid decline",
        ],
        "etiologies": {
            "PiZZ_homozygous": {"pct": 55, "phenotype": "severe_lung_liver", "notes": "AAT ~10-15%; panacinar lower-lobe emphysema; liver disease ~15%; augmentation eligible"},
            "PiSZ_compound": {"pct": 20, "phenotype": "intermediate_lung", "notes": "AAT ~40%; lung disease develops with smoking; liver disease rare; augmentation not standard"},
            "PiZnull_compound": {"pct": 8, "phenotype": "severe_lung_no_liver", "notes": "null allele → no protein produced from that chromosome; no polymer → no liver disease; very low total AAT"},
            "PiSS_homozygous": {"pct": 7, "phenotype": "mild_rarely_symptomatic", "notes": "AAT ~60%; rarely causes lung disease alone; augmentation rarely needed"},
            "PiMZ_heterozygous": {"pct": 10, "phenotype": "carrier_liver_risk", "notes": "AAT ~60%; mild lung risk with smoking; liver fibrosis risk 2-3x above PiMM; counsel re smoking"},
        },
        "stats": {
            "prevalence_PiZZ_Northern_European": "1 in 3,500",
            "carrier_PiMZ_European": "1 in 25",
            "mean_dx_age_lung_y": 42,
            "misdiagnosed_COPD_prior_pct": 70,
            "augmentation_eligible_pct": 55,
            "liver_cirrhosis_PiZZ_pct": 15,
        },
        "dx_delay_distribution": {"mean_months": 84, "median_months": 72, "range": "12-360", "notes": "most common hereditary disease of adults that is routinely misdiagnosed as COPD/asthma; average 8y diagnostic delay; lower lobe emphysema clue often missed"},
    },

    # ── SFTPC — Surfactant Protein C / Familial ILD ──────────────────────────
    {
        "gene": "SFTPC",
        "protein": "SFTPC — 8p21.3 AD — Surfactant-Protein-C-197aa — SP-C-Deficiency-Familial-ILD — I73T-Most-Common-40pct — Neonatal-RDS-to-Adult-ILD — HCQ-Azithromycin-First-Line — Nintedanib-Pirfenidone-Progression — Transplant-Curative",
        "alias": (
            "SFTPC (surfactant protein C); OMIM gene 178620; "
            "Surfactant Protein C Deficiency / Interstitial Lung Disease due to SP-C dysfunction OMIM 265120. "
            "8p21.3; 197 aa (proprotein); ~5 kDa (mature peptide); autosomal dominant (dominant-negative). "
            "FUNCTION: Surfactant protein C (SP-C) is a hydrophobic, lipid-binding protein "
            "essential for surfactant function. SP-C reduces alveolar surface tension, "
            "preventing alveolar collapse at end-expiration. "
            "Mature SP-C is a 35-aa peptide derived from the 197-aa proprotein (proSP-C) "
            "through multi-step processing in type II alveolar epithelial cells. "
            "PATHOMECHANISM: DOMINANT-NEGATIVE — "
            "Mutant proSP-C (esp. I73T) misfolds in the ER → accumulates as toxic aggregates → "
            "triggers ER stress → type II pneumocyte apoptosis → ILD. "
            "The misfolded protein also poisons processing of wild-type proSP-C from the normal allele "
            "(dominant-negative effect). "
            "CLINICAL SPECTRUM — EXTREMELY VARIABLE: "
            "Neonatal presentation: severe RDS at birth (term or near-term) — in null/severe variants; "
            "Childhood ILD (chILD): recurrent respiratory infections, failure to thrive, hypoxia; "
            "Adult-onset ILD: usual interstitial pneumonia (UIP) or non-specific interstitial pneumonia (NSIP) "
            "pattern; familial IPF; "
            "I73T MUTATION (~40% of SFTPC ILD): typically adult-onset familial ILD; "
            "incomplete penetrance — family members with same mutation may be asymptomatic; "
            "HRCT patterns: ground-glass opacity, reticulation, traction bronchiectasis; UIP or NSIP; "
            "BAL: foamy macrophages, elevated phospholipids; "
            "biopsy: DIP (desquamative interstitial pneumonia) or NSIP in children; UIP in adults; "
            "TREATMENT: "
            "No randomised controlled trial data specific to SFTPC ILD; "
            "Hydroxychloroquine (HCQ) + azithromycin: first-line empirical combination in children and adults; "
            "HCQ mechanism: reduces lysosomal dysfunction and autophagy disruption; "
            "Nintedanib or pirfenidone: for progressive fibrosis (UIP pattern) — used off-label; "
            "Lung transplantation: curative for end-stage ILD; "
            "PROGNOSIS: extremely variable; some patients stable for decades; others rapid progression; "
            "paediatric onset generally more severe than adult-onset I73T."
        ),
        "locus": "8p21.3",
        "aa": 197,
        "kDa": 5,
        "omim_gene": "178620",
        "omim_disease": "265120",
        "inheritance": "AD — Autosomal Dominant — dominant-negative mechanism; 50% transmission risk; incomplete penetrance (especially I73T); de novo mutations documented; family members with same variant may be unaffected",
        "gene_class": "Surfactant-associated protein C (hydrophobic transmembrane helical); type II pneumocyte; surface-tension reduction; proprotein processing in lamellar bodies; BRICHOS domain mutations cause ER misfolding",
        "key_alerts": [
            "SFTPC-I73T-MOST-COMMON-INCOMPLETE-PENETRANCE: I73T (c.218T>C) accounts for ~40% of SFTPC ILD; adult-onset familial ILD; incomplete penetrance means asymptomatic family members may carry same variant — phenotype prediction unreliable; genetic counselling essential; annual surveillance for carriers",
            "SFTPC-DOMINANT-NEGATIVE-NOT-HAPLOINSUFFICIENCY: mutant proSP-C poisons wild-type proSP-C processing (dominant-negative); not simple LOF/haploinsufficiency; explains why heterozygotes are affected despite one normal allele; frameshift/null mutations tend to be less severe than missense dominant-negative",
            "SFTPC-HCQ-AZITHROMYCIN-FIRST-LINE-EMPIRICAL: hydroxychloroquine + azithromycin combination is first-line empirical treatment (children and adults with SFTPC ILD); evidence base = case series and retrospective cohorts only (no RCT); HCQ dose 5 mg/kg/day; monitor QTc with azithromycin; ophthalmology surveillance for HCQ toxicity",
            "SFTPC-NINTEDANIB-PIRFENIDONE-PROGRESSIVE-UIP: nintedanib (150 mg BID) or pirfenidone for progressive fibrosis with UIP pattern; used off-label for monogenic ILD; approved for IPF/progressive fibrotic ILD; monitor LFTs with nintedanib; GI side effects common; combine with HCQ in some centres",
            "SFTPC-NEONATAL-RDS-TERM-INFANT-DDx: term or near-term infant with unexplained RDS not responding to surfactant → suspect SFTPC (or ABCA3/NKX2-1) mutation; surfactant replacement partially effective but does not correct underlying ER stress; lung biopsy shows DIP/PAP pattern",
            "SFTPC-LUNG-TRANSPLANT-CURATIVE: lung transplant is curative for end-stage SFTPC ILD; disease does not recur in graft (donor alveoli produce normal SP-C); refer for transplant evaluation when FVC <50% or rapid functional decline; outcomes comparable to IPF transplant",
            "SFTPC-FAMILY-CASCADE-GENETIC-TESTING: index case → test first-degree relatives; annual spirometry + DLCO for variant carriers; low threshold for HRCT if symptoms develop; genetic counselling for reproductive decisions",
            "SFTPC-BAL-FOAMY-MACROPHAGES: bronchoalveolar lavage in SFTPC ILD shows foamy/lipid-laden macrophages and increased phospholipids — non-specific but supportive of surfactant dysfunction aetiology; PAS-positive granules in type II cells on biopsy",
        ],
        "etiologies": {
            "SFTPC_I73T_adult_ILD": {"pct": 40, "phenotype": "adult_onset_familial_ILD", "notes": "most common; UIP/NSIP HRCT; incomplete penetrance; HCQ first-line"},
            "SFTPC_BRICHOS_domain_adult": {"pct": 25, "phenotype": "moderate_adult", "notes": "BRICHOS domain mutations; ER stress; adult ILD; variable severity"},
            "SFTPC_neonatal_severe": {"pct": 20, "phenotype": "neonatal_RDS_severe", "notes": "null/severe ER-retention variants; surfactant replacement partially helps; transplant may be required"},
            "SFTPC_childhood_chILD": {"pct": 15, "phenotype": "childhood_ILD_DIP", "notes": "DIP/NSIP pattern on biopsy; failure to thrive; HCQ + steroid; transplant if refractory"},
        },
        "stats": {
            "prevalence": "rare — <1 in 100,000",
            "most_common_mutation_pct": 40,
            "mean_dx_age_adult_y": 38,
            "lung_transplant_rate_pct": 20,
            "5yr_survival_post_transplant_pct": 65,
        },
        "dx_delay_distribution": {"mean_months": 48, "median_months": 36, "range": "0-240", "notes": "neonatal onset diagnosed quickly; adult-onset ILD labelled IPF for years; family history of ILD is the key clue; genetic ILD panel should include SFTPC"},
    },

    # ── TERT — Telomere Biology Disorders / Familial IPF ─────────────────────
    {
        "gene": "TERT",
        "protein": "TERT — 5p15.33 AD — Telomerase-Reverse-Transcriptase-1132aa — Telomere-Biology-Disorder-Familial-IPF — Telomere-Length-1st-Percentile-DIAGNOSTIC — IPF+AplasticAnemia+Cirrhosis-TRIAD — Danazol-Partial — IS-CONTRAINDICATED — Transplant-Preferred",
        "alias": (
            "TERT (telomerase reverse transcriptase); OMIM gene 187270; "
            "Telomere Biology Disorders (TBD) / Dyskeratosis Congenita (DC) OMIM 127550; "
            "Familial Pulmonary Fibrosis (FPF) OMIM 614742. "
            "5p15.33; 1132 aa; ~127 kDa; autosomal dominant (haploinsufficiency). "
            "FUNCTION: TERT is the catalytic reverse transcriptase subunit of telomerase. "
            "Telomerase elongates telomeres (TTAGGG repeats) at chromosome ends, counteracting "
            "progressive shortening with each cell division. "
            "Components: TERT (catalytic) + TERC (RNA template) + TCAB1/dyskerin (assembly). "
            "TERT LOF → haploinsufficiency → telomerase activity reduced ~50% → "
            "accelerated telomere shortening → critically short telomeres → replicative senescence "
            "in high-turnover tissues (bone marrow, epithelium, liver, lung). "
            "ANTICIPATION: telomere length shorter in successive generations → disease onset earlier "
            "and more severe in offspring. "
            "CLINICAL SPECTRUM — THREE MAIN PHENOTYPES: "
            "(1) PULMONARY FIBROSIS (most common adult TERT phenotype): "
            "UIP pattern on HRCT; rapidly progressive; mean survival from diagnosis 2-4 years; "
            "identical to sporadic IPF clinically but younger age at onset (~50-60y vs 70y); "
            "familial clustering — at least two first-degree relatives with IPF; "
            "(2) BONE MARROW FAILURE (aplastic anemia, MDS, AML): "
            "cytopenias; pure red cell aplasia; MDS transformation risk; "
            "(3) LIVER DISEASE: "
            "cryptogenic cirrhosis; non-alcoholic liver disease pattern; "
            "MUCOCUTANEOUS TRIAD (classic DC, more severe/younger): "
            "reticulated skin pigmentation + oral leukoplakia + nail dystrophy; "
            "TELOMERE LENGTH — DIAGNOSTIC ANCHOR: "
            "Telomere length <1st percentile (by flow-FISH) = DIAGNOSTIC for TBD; "
            "not all TERT carriers have short telomeres early — length is a dynamic biomarker; "
            "TREATMENT: "
            "Danazol (synthetic androgen): upregulates TERT and TERC expression via androgen response element; "
            "slows telomere attrition partially; haematological benefit in bone marrow failure; "
            "limited pulmonary benefit; hepatotoxic; "
            "IMMUNOSUPPRESSION CONTRAINDICATED FOR PULMONARY FIBROSIS: "
            "Classic IPF management with cyclophosphamide/azathioprine/prednisone WORSENS outcomes in TBD; "
            "IS suppresses residual haematopoiesis and may precipitate bone marrow failure; "
            "nintedanib for lung fibrosis anti-fibrotic benefit; "
            "LUNG TRANSPLANT: preferred over IS; "
            "transplant centre must be aware of TBD — increased post-transplant complications "
            "(bone marrow failure, skin cancers, hepatic veno-occlusive disease); "
            "matched sibling donor haematopoietic stem cell transplant for bone marrow failure "
            "(use reduced-intensity conditioning — full myeloablation catastrophic in TBD)."
        ),
        "locus": "5p15.33",
        "aa": 1132,
        "kDa": 127,
        "omim_gene": "187270",
        "omim_disease": "127550",
        "inheritance": "AD — Autosomal Dominant — haploinsufficiency; ANTICIPATION (telomeres shorter each generation → earlier more severe disease in offspring); de novo mutations in severe early-onset cases; AR biallelic = very severe Hoyeraal-Hreidarsson",
        "gene_class": "Reverse transcriptase (telomerase catalytic subunit); elongates TTAGGG repeats using TERC RNA template; expressed in stem cells, germ cells, proliferative epithelia; haploinsufficiency reduces telomerase activity ~50%",
        "key_alerts": [
            "TERT-TELOMERE-LENGTH-1ST-PERCENTILE-DIAGNOSTIC: telomere length <1st percentile by flow-FISH (or qPCR with caution) = DIAGNOSTIC for telomere biology disorder; test multiple cell types (lymphocytes, granulocytes); short telomeres = biomarker of disease AND risk stratification; repeat annually in confirmed carriers",
            "TERT-IPF-APLASTIC-ANEMIA-CIRRHOSIS-TRIAD: presence of IPF PLUS aplastic anemia AND/OR unexplained liver cirrhosis in same patient or family = TERT (or TERC) mutation until proven otherwise; screen for all three organs in every TERT patient; bone marrow biopsy + LFTs + liver imaging at diagnosis",
            "TERT-IMMUNOSUPPRESSION-CONTRAINDICATED-IPF: azathioprine/cyclophosphamide/high-dose prednisone are CONTRAINDICATED for TERT-associated pulmonary fibrosis — PANTHER-IPF trial showed harm in IPF generally, but harm is even greater in TBD (precipitates bone marrow failure); nintedanib anti-fibrotic is acceptable",
            "TERT-LUNG-TRANSPLANT-PREFERRED-OVER-IS: lung transplant is preferred management for end-stage TERT pulmonary fibrosis; transplant centre MUST know TBD diagnosis — risk of post-transplant cytopenias, skin SCC, hepatic complications; reduced-intensity conditioning if HSCT needed for bone marrow failure",
            "TERT-DANAZOL-ANDROGEN-PARTIAL-BENEFIT: danazol (600 mg/day) upregulates telomerase via androgen response element in TERT/TERC promoter; phase 2 trial showed telomere attrition slowed and haematological improvement; hepatotoxic (monitor LFTs monthly); limited pulmonary benefit; not approved but used off-label in TBD",
            "TERT-ANTICIPATION-GENETIC-COUNSELLING: offspring of TERT heterozygotes inherit shorter telomeres → earlier disease onset AND potentially more severe phenotype; genetic counselling mandatory; children of affected parent may be symptomatic in adulthood (or earlier if very short telomeres); prenatal testing available",
            "TERT-FAMILIAL-IPF-PANEL-MANDATORY: all patients with apparent sporadic IPF under age 60 OR familial IPF (≥2 affected first-degree relatives) should have comprehensive telomere gene panel (TERT, TERC, DKC1, RTEL1, PARN, NAF1, TINF2) — up to 25% familial IPF has identifiable telomere gene mutation",
            "TERT-BONE-MARROW-BIOPSY-AT-DIAGNOSIS: all TERT patients need bone marrow biopsy at diagnosis to assess baseline haematopoietic reserve; cytopenias may be clinically silent; MDS surveillance annually; avoid unnecessary myelosuppressive drugs (methotrexate, mycophenolate) in TBD",
        ],
        "etiologies": {
            "TERT_adult_IPF_dominant": {"pct": 50, "phenotype": "adult_pulmonary_fibrosis_UIP", "notes": "most common adult presentation; familial IPF pattern; telomeres <1st percentile; nintedanib"},
            "TERT_aplastic_anemia_dominant": {"pct": 20, "phenotype": "bone_marrow_failure", "notes": "pure red cell aplasia or AA; danazol first-line; HSCT if refractory; avoid myeloablation"},
            "TERT_multi_organ_adult": {"pct": 15, "phenotype": "IPF_plus_liver_or_BM", "notes": "two or three organ triad; worst prognosis; transplant evaluation"},
            "TERT_mucocutaneous_triad_younger": {"pct": 10, "phenotype": "classic_DC_younger", "notes": "skin pigmentation + leukoplakia + nail dystrophy; severe; de novo common"},
            "TERT_biallelic_severe": {"pct": 5, "phenotype": "Hoyeraal_Hreidarsson_neonatal", "notes": "AR biallelic; cerebellum hypoplasia; IUGR; very short telomeres; rarely survive to adulthood"},
        },
        "stats": {
            "prevalence_familial_IPF_TERT_pct": 15,
            "mean_dx_age_IPF_y": 54,
            "mean_survival_post_IPF_dx_y": 3,
            "bone_marrow_failure_lifetime_pct": 30,
            "liver_cirrhosis_pct": 20,
        },
        "dx_delay_distribution": {"mean_months": 36, "median_months": 24, "range": "6-120", "notes": "TERT IPF indistinguishable from sporadic IPF without genetic testing; family history is key; telomere length testing not widely available — delays diagnosis"},
    },

    # ── FLCN — Birt-Hogg-Dubé Syndrome ──────────────────────────────────────
    {
        "gene": "FLCN",
        "protein": "FLCN — 17p11.2 AD — Folliculin-579aa — Birt-Hogg-Dube-Syndrome — Fibrofolliculomas-PATHOGNOMONIC-70-90pct — Pneumothorax-25-33pct-Lifetime — Chromophobe-Oncocytoma-Hybrid-RCC-20-34pct — NO-Aviation-Scuba-Diving-High-Altitude-ABSOLUTE-PROHIBITION — Pleurodesis-Recurrent",
        "alias": (
            "FLCN (folliculin); OMIM gene 607273; "
            "Birt-Hogg-Dubé (BHD) Syndrome OMIM 135150. "
            "17p11.2; 579 aa; ~64 kDa; autosomal dominant (tumour suppressor, LOF). "
            "FUNCTION: Folliculin is a tumour suppressor protein that interacts with "
            "FNIP1 and FNIP2 (folliculin-interacting proteins) forming a complex that "
            "regulates the mTOR pathway (via AMPK), lysosomal biogenesis (via TFEB nuclear exclusion), "
            "and mitochondrial biogenesis. "
            "FLCN LOF → mTOR hyperactivation → cell growth and proliferation; "
            "also impairs ciliogenesis in kidney tubular cells → renal tumourigenesis. "
            "CLINICAL TRIAD: "
            "(1) FIBROFOLLICULOMAS: benign hamartomas of hair follicles; "
            "multiple skin-coloured dome-shaped papules on face (nose, cheeks), neck, upper trunk; "
            "70-90% BHD patients by age 25-35y; "
            "PATHOGNOMONIC — no other condition causes multiple fibrofolliculomas; "
            "biopsy shows: concentric fibrous stroma around distorted hair follicle epithelium; "
            "(2) LUNG CYSTS AND PNEUMOTHORAX: "
            "bilateral, basilar, subpleural thin-walled cysts on CT in >80% of BHD patients; "
            "cyst size: 0.1-7 cm; usually no respiratory symptoms from cysts alone; "
            "SPONTANEOUS PNEUMOTHORAX in 25-33% lifetime; mean age first pneumothorax ~38y; "
            "recurrence rate: 75% after first SP; "
            "ABSOLUTE PROHIBITION: aviation (including commercial air travel without pressure warning), "
            "scuba diving, high-altitude mountaineering — all dramatically increase SP risk; "
            "PLEURODESIS for recurrent pneumothorax (≥2 ipsilateral episodes); "
            "(3) RENAL TUMOURS: "
            "Bilateral, multifocal, chromophobe RCC (most common in BHD), "
            "hybrid oncocytic tumour (PATHOGNOMONIC for BHD), oncocytoma, clear cell RCC; "
            "20-34% BHD patients develop renal tumours; "
            "BHD renal tumours are SLOW GROWING — surveillance approach: "
            "3 cm threshold for intervention (nephron-sparing surgery); "
            "annual renal MRI surveillance from diagnosis; "
            "DIAGNOSIS: "
            "Clinical (fibrofolliculomas = sufficient for clinical diagnosis); "
            "FLCN sequencing (pathogenic variant confirms); "
            "most common mutations: c.1285dupC (exon 11 hotspot, 40% BHD families); "
            "MANAGEMENT: "
            "Skin: ablative laser (CO2/erbium) for cosmetic improvement; "
            "Lung: avoid SP triggers; pleurodesis for recurrent SP; "
            "Kidney: annual MRI; nephron-sparing surgery at 3 cm; mTOR inhibitors investigated."
        ),
        "locus": "17p11.2",
        "aa": 579,
        "kDa": 64,
        "omim_gene": "607273",
        "omim_disease": "135150",
        "inheritance": "AD — Autosomal Dominant — tumour suppressor LOF (two-hit Knudson in renal tumours); 50% transmission; de novo in ~10%; c.1285dupC hotspot exon 11 accounts for ~40% of all BHD variants worldwide",
        "gene_class": "Tumour suppressor (FLCN-FNIP1/2 complex); mTOR pathway regulator via AMPK; lysosomal biogenesis via TFEB nuclear exclusion; mitochondrial biogenesis; expressed in skin, kidney, lung; LOF = mTOR hyperactivation",
        "key_alerts": [
            "FLCN-FIBROFOLLICULOMAS-PATHOGNOMONIC: multiple fibrofolliculomas (dome-shaped skin-coloured papules on face/neck/trunk) are PATHOGNOMONIC for BHD — no other hereditary syndrome causes this lesion; biopsy confirms if clinical diagnosis uncertain; presence = immediate FLCN genetic testing + full BHD surveillance",
            "FLCN-NO-AVIATION-SCUBA-HIGH-ALTITUDE-ABSOLUTE: aviation, scuba diving, and high-altitude exposure are ABSOLUTELY PROHIBITED in BHD (lung cysts + pneumothorax risk); even pressurised commercial flights carry risk of cabin depressurisation; FLCN patients should carry medical alert card; provide written activity restriction letter",
            "FLCN-PNEUMOTHORAX-25-33PCT-PLEURODESIS: lifetime SP risk 25-33%; recurrence 75% after first episode; PLEURODESIS (chemical or mechanical/VATS) recommended after FIRST SP in BHD (given high recurrence); lobectomy of apical blebs inadequate alone without pleurodesis; contralateral prophylactic pleurodesis considered at some centres",
            "FLCN-RENAL-MRI-ANNUAL-3CM-THRESHOLD: annual renal MRI from age 20 (or 10 years before youngest affected family member); 3 cm = intervention threshold (nephron-sparing surgery, partial nephrectomy or ablation); tumours <3 cm — active surveillance; avoid total nephrectomy; bilateral multifocal tumours common",
            "FLCN-HYBRID-ONCOCYTOMA-BHD-PATHOGNOMONIC-RENAL: hybrid chromophobe/oncocytoma renal tumour is the PATHOGNOMONIC renal tumour histology in BHD (not seen in VHL, SDHB, MET); presence of bilateral hybrid oncocytic tumours → BHD genetic testing even without skin/lung manifestations",
            "FLCN-EXON11-C1285DUPC-HOTSPOT: c.1285dupC in exon 11 accounts for ~40% of BHD-associated FLCN variants worldwide (frameshift, insertion in C8 homopolymer tract); targeted testing for this variant first before full gene sequencing in appropriate clinical context",
            "FLCN-MTOR-PATHWAY-RESEARCH: everolimus/sirolimus (mTOR inhibitors) shown to reduce cyst size and renal tumour growth in preclinical BHD models; clinical trials ongoing; not standard of care; may be considered for inoperable/multifocal renal disease",
            "FLCN-LUNG-CYSTS-NO-TREATMENT-NEEDED: bilateral basilar lung cysts in BHD are asymptomatic in most patients; no treatment required for cysts alone; baseline CT chest at diagnosis; repeat only if symptoms or before intervention; FEV1/DLCO typically normal despite cysts",
        ],
        "etiologies": {
            "FLCN_exon11_C1285dupC_frameshift": {"pct": 40, "phenotype": "classic_BHD_all_three", "notes": "hotspot; frameshift; classic fibrofolliculoma + cysts + RCC risk"},
            "FLCN_other_truncating": {"pct": 35, "phenotype": "BHD_variable", "notes": "various exons; truncating; full BHD phenotype; SP and RCC variable"},
            "FLCN_missense": {"pct": 15, "phenotype": "variable_often_lung_skin", "notes": "some missense reduce function partially; skin and lung prominent; RCC variable"},
            "FLCN_large_deletion": {"pct": 10, "phenotype": "severe_RCC_predominant", "notes": "whole gene or exon deletion; MLPA required; RCC may be early and multiple"},
        },
        "stats": {
            "prevalence": "1 in 200,000 (estimated)",
            "fibrofolliculoma_pct_by_35y": 85,
            "pneumothorax_lifetime_pct": 29,
            "renal_tumour_pct": 27,
            "mean_age_first_SP_y": 38,
        },
        "dx_delay_distribution": {"mean_months": 72, "median_months": 60, "range": "12-240", "notes": "fibrofolliculomas often attributed to acne/normal skin; lung cysts found incidentally; diagnosis often prompted by renal tumour workup or recurrent pneumothorax"},
    },

    # ── TSC1 — Tuberous Sclerosis Complex / LAM ──────────────────────────────
    {
        "gene": "TSC1",
        "protein": "TSC1 — 9q34.13 AD — Hamartin-1164aa — Tuberous-Sclerosis-Complex-1 — LAM-Exclusively-Females — Sirolimus-FDA2015-LAM — VEGF-D-800-PATHOGNOMONIC — FEV1-Decline-75-100mL-yr-without-Therapy — Chylothorax-Pneumothorax",
        "alias": (
            "TSC1 (tuberous sclerosis complex 1 / hamartin); OMIM gene 605284; "
            "Tuberous Sclerosis Complex (TSC) OMIM 191100; "
            "Lymphangioleiomyomatosis (LAM) OMIM 606690. "
            "9q34.13; 1164 aa; ~130 kDa (hamartin); autosomal dominant. "
            "FUNCTION: Hamartin (TSC1) forms a heterodimer with tuberin (TSC2) — the TSC1/TSC2 complex "
            "is a major negative regulator of mTORC1 via its GAP (GTPase activating protein) activity "
            "on the small GTPase Rheb. "
            "TSC1/TSC2 → Rheb-GDP (inactive) → mTORC1 OFF → controlled cell growth/proliferation/metabolism. "
            "TSC1 or TSC2 LOF → Rheb-GTP (active) → mTORC1 hyperactivation → "
            "hamartoma formation in brain, kidney, skin, lung, heart. "
            "LAM PATHOMECHANISM: "
            "LAM cells (smooth muscle-like cells with TSC2 mutations) metastasise from uterus/lymphatics "
            "to lung → progressive cystic lung destruction. "
            "LAM occurs in TWO SETTINGS: "
            "(1) TSC-LAM (tuberous sclerosis associated LAM): 30-40% of females with TSC develop LAM; "
            "(2) Sporadic LAM (S-LAM): somatic TSC2 mutations in LAM cells; no germline TSC mutation. "
            "TSC1 mutations cause LAM LESS COMMONLY than TSC2 (TSC2 ~80% TSC-LAM). "
            "LAM CLINICAL FEATURES — EXCLUSIVELY FEMALES: "
            "Exertional dyspnoea (insidious onset); "
            "recurrent spontaneous pneumothorax (30-40%); "
            "chylous pleural effusion (chylothorax — thoracic duct involvement); "
            "FEV1 decline: 75-100 mL/year without treatment (rapid vs 25-30 mL/y normal); "
            "HRCT: bilateral, diffuse, thin-walled cysts (uniform, round, distributed throughout lung); "
            "DIAGNOSIS: "
            "VEGF-D >800 pg/mL = PATHOGNOMONIC for LAM (sensitivity 73%, specificity >97%); "
            "can avoid lung biopsy if VEGF-D elevated + characteristic HRCT + clinical context; "
            "Lung biopsy: LAM cells (HMB-45+, smooth muscle actin+, ER/PR+); "
            "TREATMENT: "
            "Sirolimus (rapamycin): FDA-approved 2015 for LAM — mTOR inhibitor; "
            "MILES trial: FEV1 decline stabilised (+1 mL/year on sirolimus vs -12 mL/y placebo); "
            "symptoms improved; QoL improved; "
            "everolimus: alternative mTOR inhibitor; "
            "Bronchodilators: some LAM patients have airflow obstruction → salbutamol helpful; "
            "OTHER TSC MANIFESTATIONS: "
            "SEGA (subependymal giant cell astrocytoma): everolimus shrinks SEGA; "
            "renal angiomyolipomata (AML): sirolimus/everolimus; embolisation if >3 cm; "
            "skin: angiofibromas, hypomelanotic macules, shagreen patches; "
            "cardiac rhabdomyoma (neonatal); epilepsy (>80% TSC)."
        ),
        "locus": "9q34.13",
        "aa": 1164,
        "kDa": 130,
        "omim_gene": "605284",
        "omim_disease": "191100",
        "inheritance": "AD — Autosomal Dominant — haploinsufficiency (tumour suppressor); 50% transmission; de novo mutations account for ~65% of TSC (no family history); mosaic TSC in ~15%; TSC2 mutations more severe/common than TSC1",
        "gene_class": "Tumour suppressor (TSC1/TSC2 heterodimer = GAP for Rheb); mTORC1 inhibitor; hamartin stabilises tuberin; LOF → mTORC1 hyperactivation → hamartoma formation across multiple organs",
        "key_alerts": [
            "TSC1-LAM-EXCLUSIVELY-FEMALES-SIROLIMUS-FDA2015: LAM (lymphangioleiomyomatosis) occurs EXCLUSIVELY in females (oestrogen-dependent TSC2 LAM cell proliferation); sirolimus (rapamycin) FDA-approved 2015 for LAM (MILES trial — stabilises FEV1, improves QoL); start sirolimus when FEV1 <70% predicted or rapid decline; do NOT wait for severe impairment",
            "TSC1-VEGF-D-800-PATHOGNOMONIC: serum VEGF-D >800 pg/mL = PATHOGNOMONIC for LAM (high specificity >97%); enables LAM diagnosis WITHOUT lung biopsy when HRCT is characteristic and clinical context appropriate; VEGF-D also useful for monitoring sirolimus response (levels fall on therapy)",
            "TSC1-FEV1-DECLINE-75-100mL-YEAR: untreated LAM shows FEV1 decline 75-100 mL/year (vs 25-30 mL/year in normal ageing); sirolimus reduces decline to near-normal; spirometry every 3-6 months; DLCO (gas transfer) also declines; 6MWT for functional assessment",
            "TSC1-CHYLOTHORAX-CHYLOUS-EFFUSION: chylous pleural effusion (chylothorax) from thoracic duct involvement by LAM cells; milky appearance; triglycerides >110 mg/dL confirms chyle; management: sirolimus (often resolves chylothorax), low-fat diet with MCT supplement, pleurodesis, thoracic duct ligation if refractory",
            "TSC1-SEGA-EVEROLIMUS-MONITORING: subependymal giant cell astrocytoma (SEGA) in ~10% TSC patients; everolimus shrinks SEGA (FDA-approved); annual brain MRI for SEGA surveillance; seizure control with mTOR inhibitor as adjunct; neurodevelopmental assessment",
            "TSC1-RENAL-AML-EMBOLISATION-3CM: renal angiomyolipomata (AML) in ~80% TSC; bleed risk when >3 cm; embolisation or nephron-sparing surgery at 3 cm; sirolimus/everolimus reduce AML volume; avoid total nephrectomy; annual renal imaging",
            "TSC1-PNEUMOTHORAX-PLEURODESIS-LAM: spontaneous pneumothorax in 30-40% LAM; high recurrence (>70%); pleurodesis recommended after first LAM-associated SP (high recurrence unlike non-LAM SP); VATS pleurodesis preferred; does NOT preclude lung transplant in most centres",
            "TSC1-TSC2-MORE-SEVERE-THAN-TSC1: TSC2 mutations cause more severe neurological, renal, and LAM phenotype than TSC1; TSC1 mutations more commonly present with milder phenotype; pulmonary LAM still requires sirolimus regardless of which gene; genetic testing identifies TSC1 vs TSC2 for prognostication",
        ],
        "etiologies": {
            "TSC1_LOF_truncating_LAM": {"pct": 25, "phenotype": "LAM_TSC_associated", "notes": "females; LAM + other TSC features; VEGF-D elevated; sirolimus"},
            "TSC1_missense_milder_TSC": {"pct": 35, "phenotype": "TSC_multi_organ_mild", "notes": "neurological + renal + skin; LAM less common in TSC1 than TSC2; everolimus for SEGA/AML"},
            "TSC1_de_novo_classic": {"pct": 25, "phenotype": "classic_TSC_full", "notes": "de novo (65% TSC is de novo); epilepsy prominent; AML; skin; SEGA risk"},
            "TSC1_mosaic_TSC": {"pct": 15, "phenotype": "mosaic_variable_mild", "notes": "somatic mosaicism; milder; may escape diagnosis until adulthood; blood DNA may miss mosaic"},
        },
        "stats": {
            "prevalence_TSC": "1 in 6,000",
            "LAM_in_females_with_TSC_pct": 35,
            "FEV1_decline_untreated_mL_per_y": 88,
            "FEV1_decline_sirolimus_mL_per_y": 1,
            "de_novo_mutation_pct": 65,
            "SEGA_pct": 10,
        },
        "dx_delay_distribution": {"mean_months": 60, "median_months": 48, "range": "6-180", "notes": "LAM often diagnosed after recurrent pneumothorax or incidental CT; misdiagnosed as asthma/emphysema; VEGF-D and HRCT pattern reduce biopsy need and delay"},
    },

    # ── ABCA3 — Surfactant Metabolism Dysfunction Type 3 ─────────────────────
    {
        "gene": "ABCA3",
        "protein": "ABCA3 — 16p13.3 AR — ABCA3-Transporter-1704aa — Surfactant-Metabolism-Dysfunction-Type3 — Neonatal-RDS-Null/Null-to-Childhood-Adult-ILD — Lamellar-Bodies-ABSENT-EM-DIAGNOSTIC — No-Specific-Therapy — HCQ-Modest — Transplant-Severe-Neonatal",
        "alias": (
            "ABCA3 (ATP-binding cassette sub-family A member 3); OMIM gene 601615; "
            "Surfactant Metabolism Dysfunction, Pulmonary, Type 3 (SMDP3) OMIM 610921. "
            "16p13.3; 1704 aa; ~190 kDa; autosomal recessive. "
            "FUNCTION: ABCA3 is a lipid transporter located on the limiting membrane of lamellar bodies "
            "(secretory organelles of type II alveolar epithelial cells). "
            "Lamellar bodies store and release pulmonary surfactant. "
            "ABCA3 transports phospholipids (primarily phosphatidylcholine and phosphatidylglycerol) "
            "into lamellar bodies for incorporation into surfactant. "
            "ABCA3 LOF → impaired phospholipid transport → "
            "dysfunctional or absent lamellar bodies → surfactant deficiency/dysfunction → "
            "respiratory failure (neonatal) or progressive ILD (childhood/adult). "
            "MUTATION-PHENOTYPE CORRELATION: "
            "NULL/NULL (two truncating mutations): severe neonatal RDS; "
            "fatal within days-weeks without respiratory support; lung transplant required; "
            "MISSENSE/MISSENSE or NULL/MISSENSE: childhood or adult-onset ILD; "
            "slower progression; HCQ may modestly help; "
            "W292C is the most common pathogenic missense (type I trafficking mutation); "
            "CLINICAL FEATURES — NEONATAL (null/null): "
            "Term or near-term infant with respiratory failure not responding to surfactant replacement; "
            "diffuse ground-glass opacity on CXR; surfactant replacement transiently helps but disease recurs; "
            "death without transplant in majority; "
            "CLINICAL FEATURES — CHILDHOOD/ADULT (missense): "
            "Insidious hypoxia, dyspnoea, failure to thrive; "
            "HRCT: ground-glass opacity, reticulation, cystic change; DIP/NSIP/UIP pattern on biopsy; "
            "DIAGNOSIS: "
            "ELECTRON MICROSCOPY (EM) of lung biopsy — DIAGNOSTIC: "
            "lamellar bodies absent or grossly abnormal (small, dense, concentric whorled structure); "
            "this EM finding is PATHOGNOMONIC for ABCA3 deficiency (not seen in SFTPC, NKX2-1); "
            "ABCA3 sequencing confirms diagnosis; "
            "TREATMENT: "
            "No approved specific therapy; "
            "Hydroxychloroquine: modest benefit in some missense cases; "
            "Azithromycin: anti-inflammatory adjunct; "
            "Corticosteroids: pulse methylprednisolone may slow progression in some; "
            "Nintedanib: for progressive fibrosis; "
            "Lung transplant: sole curative option for severe neonatal or rapidly progressive childhood ILD."
        ),
        "locus": "16p13.3",
        "aa": 1704,
        "kDa": 190,
        "omim_gene": "601615",
        "omim_disease": "610921",
        "inheritance": "AR — Autosomal Recessive — biallelic LOF; null/null = severe neonatal; missense compound heterozygous = childhood/adult ILD; parents obligate heterozygotes; carrier frequency ~1 in 65 (European)",
        "gene_class": "ABC lipid transporter (ABCA subfamily, type 1 half-transporter); lamellar body membrane; phosphatidylcholine/PG transport into lamellar bodies; essential for surfactant biogenesis in type II alveolar epithelial cells",
        "key_alerts": [
            "ABCA3-LAMELLAR-BODIES-ABSENT-EM-DIAGNOSTIC: electron microscopy of lung biopsy shows absent or grossly abnormal lamellar bodies (small, dense, concentric whorled inclusions) = PATHOGNOMONIC for ABCA3 deficiency; this EM finding is the gold standard diagnostic feature — not seen in SFTPC or NKX2-1 ILD; request EM specifically when ordering lung biopsy",
            "ABCA3-NULL-NULL-FATAL-NEONATAL-TRANSPLANT: biallelic truncating (null/null) mutations cause lethal neonatal RDS; exogenous surfactant provides only transient benefit; lung transplant is the only curative option; refer to paediatric transplant centre urgently for null/null neonates; prognosis without transplant = weeks",
            "ABCA3-TERM-INFANT-UNEXPLAINED-RDS-SURFACTANT-GENETIC: term infant with unexplained RDS not fully responding to surfactant → genetic surfactant disorder (ABCA3, SFTPC, NKX2-1) must be excluded; order ABCA3 sequencing in parallel with surfactant administration; do NOT delay genetic testing",
            "ABCA3-W292C-MOST-COMMON-MISSENSE-CHILDHOOD-ILD: W292C (c.875G>T) is the most frequent pathogenic missense ABCA3 variant (type I trafficking defect — ABCA3 does not reach lamellar body membrane); compound heterozygous W292C/truncating causes childhood ILD with slower progression than null/null; HCQ modestly helpful",
            "ABCA3-HCQ-MODEST-BENEFIT-MISSENSE: hydroxychloroquine (5 mg/kg/day) may slow progression in missense ABCA3 ILD; evidence from retrospective cohorts only; not curative; monitor ophthalmology toxicity (cumulative dose); azithromycin added for anti-inflammatory; no RCT data",
            "ABCA3-NINTEDANIB-PROGRESSIVE-FIBROSIS: nintedanib used off-label for progressive fibrosis pattern in ABCA3 ILD (UIP on HRCT or biopsy); mechanism: anti-fibrotic via PDGFR/VEGFR/FGFR inhibition; monitor LFTs; GI side effects; combine with HCQ in some protocols",
            "ABCA3-CARRIER-TESTING-RELATIVES: carrier frequency ~1 in 65 in Europeans; autosomal recessive — parents of affected child are obligate carriers; sibling risk 25%; prenatal diagnosis available; preconception carrier testing offered to relatives of affected",
            "ABCA3-BAL-FOAMY-MACROPHAGES-SURFACTANT: BAL shows foamy macrophages and increased phospholipid content (surfactant dysfunction); PAS-positive material; non-specific but supports surfactant disorder diagnosis alongside EM and sequencing",
        ],
        "etiologies": {
            "ABCA3_null_null_neonatal": {"pct": 20, "phenotype": "fatal_neonatal_RDS", "notes": "two truncating mutations; term infant; death within weeks without transplant"},
            "ABCA3_W292C_compound_het": {"pct": 30, "phenotype": "childhood_adult_ILD", "notes": "most common missense; slower progression; HCQ ± nintedanib"},
            "ABCA3_missense_missense_mild": {"pct": 25, "phenotype": "childhood_ILD_variable", "notes": "two missense; intermediate severity; HRCT ground-glass; some reach adulthood"},
            "ABCA3_null_missense_intermediate": {"pct": 25, "phenotype": "severe_childhood_ILD", "notes": "one null + one missense; intermediate severity; transplant in 30-40%"},
        },
        "stats": {
            "prevalence": "rare — estimated 1 in 500,000 (clinical)",
            "carrier_frequency_European": "1 in 65",
            "null_null_mortality_without_transplant_pct": 90,
            "mean_dx_age_childhood_y": 3,
            "lung_transplant_rate_pct": 35,
        },
        "dx_delay_distribution": {"mean_months": 18, "median_months": 12, "range": "0-120", "notes": "neonatal null/null diagnosed quickly by clinical context; childhood/adult missense misdiagnosed as hypersensitivity pneumonitis or idiopathic ILD; EM biopsy and genetic testing reduce delay when ordered"},
    },

    # ── NKX2-1 — Brain-Lung-Thyroid Syndrome ────────────────────────────────
    {
        "gene": "NKX2-1",
        "protein": "NKX2-1 — 14q13.3 AD — TTF1-Homeodomain-TF-371aa — Brain-Lung-Thyroid-Syndrome — TRIAD-BHC+Hypothyroidism+ILD-PATHOGNOMONIC — Haploinsufficiency — L-Thyroxine-Mandatory — Chorea-Non-Progressive-Benign — ILD-Variable-Mild-to-Fatal — Annual-PFTs",
        "alias": (
            "NKX2-1 (NK2 homeobox 1; also known as TTF-1, TITF1, T/EBP); OMIM gene 600635; "
            "Brain-Lung-Thyroid Syndrome (BLTS) OMIM 610978. "
            "14q13.3; 371 aa; ~42 kDa; autosomal dominant (haploinsufficiency). "
            "FUNCTION: NKX2-1 (thyroid transcription factor 1, TTF-1) is a homeodomain transcription factor "
            "critical for development of three organs: thyroid, lung, and brain (basal ganglia). "
            "In THYROID: NKX2-1 drives thyroglobulin, TPO, and TSH-receptor expression → "
            "thyroid hormone biosynthesis. "
            "In LUNG: NKX2-1 activates SP-B, SP-C, and ABCA3 expression → surfactant production; "
            "also regulates airway branching morphogenesis. "
            "In BRAIN: NKX2-1 specifies GABAergic interneurons of the striatum/basal ganglia; "
            "NKX2-1 LOF → reduced GABA interneurons → chorea (hyperkinetic movement disorder). "
            "PATHOMECHANISM: haploinsufficiency — one functional allele insufficient for full transcriptional "
            "output in three organs. "
            "CLINICAL TRIAD (PATHOGNOMONIC when all three present): "
            "(1) BENIGN HEREDITARY CHOREA (BHC): "
            "choreiform movements (involuntary, rapid, irregular limb movements); "
            "onset in infancy/childhood; non-progressive (benign — does NOT worsen like HD); "
            "most patients walk and have normal intelligence (some mild cognitive delay); "
            "may improve in adulthood; may cause school difficulties (fine motor, handwriting); "
            "(2) CONGENITAL HYPOTHYROIDISM: "
            "neonatal thyroid dysfunction (absent/hypoplastic thyroid or functional failure); "
            "detected on neonatal screening (elevated TSH); "
            "L-thyroxine (levothyroxine) replacement — MANDATORY lifelong; "
            "without treatment: intellectual disability, growth failure; "
            "(3) INTERSTITIAL LUNG DISEASE: "
            "variable severity — from mild asymptomatic changes to fatal neonatal RDS; "
            "neonatal: RDS in term infant (surfactant dysfunction due to NKX2-1 haploinsufficiency "
            "→ reduced SP-B/SP-C/ABCA3); "
            "childhood/adult: ILD with ground-glass, reticulation; "
            "ILD may be the least penetrant feature (absent in some NKX2-1 mutation carriers); "
            "DIAGNOSIS: "
            "TRIAD is pathognomonic; genetic testing (NKX2-1 sequencing); "
            "MLPA for deletions (14q13.3 microdeletion detectable); "
            "MANAGEMENT: "
            "Thyroid: levothyroxine titrated to normal TSH + fT4; "
            "Chorea: tetrabenazine (dopamine depleter) if functionally limiting; "
            "clonazepam; most patients manage without treatment; "
            "Lung: annual spirometry + DLCO from diagnosis; "
            "HRCT if dyspnoea; HCQ/nintedanib if progressive ILD; "
            "lung transplant for severe progressive ILD."
        ),
        "locus": "14q13.3",
        "aa": 371,
        "kDa": 42,
        "omim_gene": "600635",
        "omim_disease": "610978",
        "inheritance": "AD — Autosomal Dominant — haploinsufficiency; 50% transmission; de novo mutations common; 14q13.3 microdeletion (detectable by array CGH/MLPA) accounts for subset; variable expressivity — all three features may not be equally penetrant",
        "gene_class": "Homeodomain transcription factor (NKX family, TTF-1); master regulator of thyroid, lung, and basal ganglia development; activates SP-B/SP-C/ABCA3 in lung; thyroglobulin/TPO/TSHR in thyroid; GABAergic neuron specification in striatum",
        "key_alerts": [
            "NKX2-1-BRAIN-LUNG-THYROID-TRIAD-PATHOGNOMONIC: combination of benign hereditary chorea + congenital hypothyroidism + ILD is PATHOGNOMONIC for NKX2-1 haploinsufficiency (Brain-Lung-Thyroid Syndrome); any two of three features in the same patient or family → NKX2-1 sequencing + 14q13.3 MLPA deletion testing",
            "NKX2-1-HYPOTHYROIDISM-LEVOTHYROXINE-MANDATORY: congenital hypothyroidism from NKX2-1 LOF requires lifelong levothyroxine replacement; detected by neonatal TSH screen; untreated congenital hypothyroidism → irreversible intellectual disability and growth failure; dose titrated by TSH and fT4; thyroid USS (hypoplastic/ectopic thyroid)",
            "NKX2-1-CHOREA-NON-PROGRESSIVE-BENIGN: NKX2-1 chorea is BENIGN HEREDITARY CHOREA (BHC) — does NOT progress like Huntington disease; involuntary movements prominent in childhood; improve in many adults; no cognitive decline; distinguish from HD (NKX2-1 chorea onset childhood, HD onset adulthood with cognitive decline)",
            "NKX2-1-ILD-VARIABLE-ANNUAL-PFTs: ILD is the most variable feature — absent in some carriers, fatal in others; annual spirometry + DLCO mandatory from diagnosis; low threshold for HRCT if dyspnoea or declining PFTs; monitor for neonatal RDS risk in offspring of affected parents",
            "NKX2-1-NEONATAL-RDS-TERM-INFANT-SURFACTANT-PANEL: term infant with unexplained RDS → NKX2-1 mutation in differential (alongside ABCA3, SFTPC); NKX2-1 reduces SP-B/SP-C/ABCA3 transcription → functional surfactant deficiency; check neonatal TSH screen + thyroid USS in same infant to identify BLTS triad",
            "NKX2-1-14Q13-MICRODELETION-MLPA: ~20-30% NKX2-1 cases are caused by 14q13.3 microdeletions (not detectable by standard sequencing); MLPA or chromosomal microarray required; larger deletions may include adjacent genes → more severe/complex phenotype",
            "NKX2-1-TTF1-IHC-LUNG-ADENOCARCINOMA: NOTE — TTF-1 (NKX2-1) IHC is a widely used lung adenocarcinoma marker; TTF-1 positivity on lung tumour biopsy indicates lung primary or thyroid metastasis; do NOT confuse clinical TTF-1 staining in tumour with NKX2-1 germline syndrome",
            "NKX2-1-TETRABENAZINE-CHOREA-IF-DISABLING: tetrabenazine (vesicular monoamine transporter 2 inhibitor) or clonazepam for functionally disabling chorea in NKX2-1; most patients do not require pharmacological treatment; monitor for depression/parkinsonism with tetrabenazine; valbenazine/deutetrabenazine as alternatives",
        ],
        "etiologies": {
            "NKX2_1_full_triad_dominant": {"pct": 55, "phenotype": "BLTS_all_three_features", "notes": "BHC + hypothyroidism + ILD; classic triad; levothyroxine + lung surveillance"},
            "NKX2_1_thyroid_chorea_no_ILD": {"pct": 25, "phenotype": "BHC_hypothyroidism_only", "notes": "ILD absent or subclinical; least penetrant organ = lung; annual PFTs"},
            "NKX2_1_severe_neonatal_lung": {"pct": 10, "phenotype": "neonatal_RDS_triad", "notes": "surfactant dysfunction at birth; NKX2-1 LOF reduces SP-B/C/ABCA3; lung transplant risk"},
            "NKX2_1_microdeletion_14q13": {"pct": 10, "phenotype": "BLTS_variable_severity", "notes": "14q13.3 microdeletion; MLPA required; adjacent gene effects possible; variable expressivity"},
        },
        "stats": {
            "prevalence": "rare — estimated 1 in 300,000",
            "triad_complete_pct": 55,
            "hypothyroidism_penetrance_pct": 95,
            "chorea_penetrance_pct": 80,
            "ILD_penetrance_pct": 60,
            "mean_dx_age_y": 3,
        },
        "dx_delay_distribution": {"mean_months": 36, "median_months": 24, "range": "0-120", "notes": "hypothyroidism detected by neonatal screen (fast); chorea attributed to other causes; ILD may be silent for years; diagnosis often prompted by assembling the triad"},
    },
]


# ── Patient simulation ────────────────────────────────────────────────────────

def _simulate_patients(gene_dict, gene_index):
    rng = random.Random(SEED_BASE + gene_index)
    gene = gene_dict["gene"]
    patients = []

    # Gene-specific parameter tables
    params = {
        "CFTR": {
            "dx_age_range": (0.1, 30), "dx_age_mean": 4,
            "delay_mean": 8, "delay_sd": 12,
            "treatments": ["Trikafta (elexacaftor/tezacaftor/ivacaftor)", "Ivacaftor (G551D)", "Lumacaftor/ivacaftor",
                           "Supportive (PERT + airway clearance)", "Lung transplant + Trikafta"],
            "fev1_mean": 68, "dlco_mean": 72,
            "transplant_pct": 8, "smoking_ever_pct": 5,
            "exac_mean": 1.8,
            "subtypes": ["F508del/F508del", "F508del/other class II", "G551D compound het", "Mild class IV/V", "Severe nonsense"],
        },
        "SERPINA1": {
            "dx_age_range": (25, 65), "dx_age_mean": 42,
            "delay_mean": 84, "delay_sd": 60,
            "treatments": ["Augmentation therapy (Prolastin-C weekly)", "Augmentation (Zemaira)", "Bronchodilators + smoking cessation",
                           "Lung transplant", "Nintedanib (bronchiectasis progression)"],
            "fev1_mean": 52, "dlco_mean": 55,
            "transplant_pct": 12, "smoking_ever_pct": 45,
            "exac_mean": 1.2,
            "subtypes": ["PiZZ homozygous", "PiSZ compound", "PiZnull compound", "PiSS homozygous", "PiMZ heterozygous"],
        },
        "SFTPC": {
            "dx_age_range": (0, 65), "dx_age_mean": 32,
            "delay_mean": 48, "delay_sd": 36,
            "treatments": ["HCQ + azithromycin", "Nintedanib (progressive UIP)", "Pirfenidone (UIP)",
                           "Lung transplant", "Supportive O2 + pulmonary rehab"],
            "fev1_mean": 60, "dlco_mean": 55,
            "transplant_pct": 18, "smoking_ever_pct": 10,
            "exac_mean": 0.6,
            "subtypes": ["I73T adult ILD", "BRICHOS domain adult", "Neonatal severe RDS", "Childhood chILD DIP"],
        },
        "TERT": {
            "dx_age_range": (30, 70), "dx_age_mean": 54,
            "delay_mean": 36, "delay_sd": 24,
            "treatments": ["Nintedanib (IPF/progressive fibrosis)", "Danazol (bone marrow failure component)",
                           "Lung transplant", "HSCT (aplastic anemia)", "Supportive O2 + pulmonary rehab"],
            "fev1_mean": 55, "dlco_mean": 48,
            "transplant_pct": 22, "smoking_ever_pct": 20,
            "exac_mean": 0.4,
            "subtypes": ["Adult IPF dominant", "Aplastic anemia dominant", "Multi-organ triad", "Mucocutaneous DC", "Biallelic severe"],
        },
        "FLCN": {
            "dx_age_range": (20, 60), "dx_age_mean": 40,
            "delay_mean": 72, "delay_sd": 48,
            "treatments": ["Active surveillance (renal MRI annual)", "Pleurodesis (recurrent pneumothorax)",
                           "Nephron-sparing surgery (RCC 3cm+)", "CO2 laser skin (fibrofolliculomas)", "mTOR inhibitor trial"],
            "fev1_mean": 88, "dlco_mean": 82,
            "transplant_pct": 2, "smoking_ever_pct": 15,
            "exac_mean": 0.2,
            "subtypes": ["Exon11 c.1285dupC classic", "Other truncating", "Missense variable", "Large deletion RCC"],
        },
        "TSC1": {
            "dx_age_range": (20, 55), "dx_age_mean": 35,
            "delay_mean": 60, "delay_sd": 48,
            "treatments": ["Sirolimus (LAM FDA 2015)", "Everolimus (SEGA/AML)", "Pleurodesis (pneumothorax)",
                           "Lung transplant", "Bronchodilator (airflow obstruction)"],
            "fev1_mean": 62, "dlco_mean": 58,
            "transplant_pct": 8, "smoking_ever_pct": 5,
            "exac_mean": 0.5,
            "subtypes": ["TSC1 LOF truncating LAM", "TSC1 missense mild TSC", "TSC1 de novo classic", "TSC1 mosaic mild"],
        },
        "ABCA3": {
            "dx_age_range": (0, 40), "dx_age_mean": 3,
            "delay_mean": 18, "delay_sd": 18,
            "treatments": ["Supportive NICU surfactant (neonatal)", "HCQ + azithromycin (missense ILD)",
                           "Nintedanib (progressive fibrosis)", "Lung transplant (null/null neonatal)",
                           "Pulse methylprednisolone"],
            "fev1_mean": 55, "dlco_mean": 50,
            "transplant_pct": 32, "smoking_ever_pct": 2,
            "exac_mean": 1.1,
            "subtypes": ["Null/null neonatal fatal", "W292C compound het childhood", "Missense/missense mild", "Null/missense intermediate"],
        },
        "NKX2-1": {
            "dx_age_range": (0, 20), "dx_age_mean": 3,
            "delay_mean": 36, "delay_sd": 30,
            "treatments": ["Levothyroxine (hypothyroidism)", "HCQ ± azithromycin (ILD)", "Tetrabenazine (disabling chorea)",
                           "Nintedanib (progressive ILD)", "Lung transplant (severe ILD)"],
            "fev1_mean": 72, "dlco_mean": 68,
            "transplant_pct": 6, "smoking_ever_pct": 2,
            "exac_mean": 0.4,
            "subtypes": ["Full BLTS triad", "BHC + hypothyroidism no ILD", "Severe neonatal RDS", "14q13 microdeletion"],
        },
    }

    p = params[gene]
    lo, hi = p["dx_age_range"]

    for i in range(40):
        pid = f"{gene}-{SEED_BASE + gene_index:04d}-{i+1:03d}"
        age_dx = round(max(lo, min(hi, rng.gauss(p["dx_age_mean"], p["dx_age_mean"] * 0.4))), 1)
        delay = max(0, round(rng.gauss(p["delay_mean"], p["delay_sd"])))
        treatment = rng.choice(p["treatments"])
        fev1 = max(20, round(rng.gauss(p["fev1_mean"], 15)))
        dlco = max(15, round(rng.gauss(p["dlco_mean"], 14)))
        on_modulator = (gene == "CFTR") and rng.random() < 0.72
        transplanted = rng.random() < (p["transplant_pct"] / 100)
        smoking = rng.random() < (p["smoking_ever_pct"] / 100)
        exac = round(max(0, rng.gauss(p["exac_mean"], p["exac_mean"] * 0.5)), 1)
        subtype = rng.choice(p["subtypes"])

        patients.append({
            "patient_id": pid,
            "age_at_dx": age_dx,
            "dx_delay_months": delay,
            "treatment": treatment,
            "fev1_pct_predicted": fev1,
            "dlco_pct_predicted": dlco,
            "on_modulator_therapy": on_modulator,
            "ever_transplanted": transplanted,
            "smoking_status": "ever" if smoking else "never",
            "exacerbation_count_per_year": exac,
            "genetic_subtype": subtype,
        })

    return patients


# Attach patients to genes
for _idx, _g in enumerate(LUNG_GENES):
    _g["patients"] = _simulate_patients(_g, _idx)


# ── Public API functions ──────────────────────────────────────────────────────

def overview():
    """Return atlas overview dict."""
    all_ages = [p["age_at_dx"] for g in LUNG_GENES for p in g["patients"]]
    all_delays = [p["dx_delay_months"] for g in LUNG_GENES for p in g["patients"]]

    genes = []
    for idx, g in enumerate(LUNG_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        genes.append({
            "gene": g["gene"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
            "n_patients": len(g["patients"]),
        })

    return {
        "atlas": "Hereditary Lung Disease Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Lung Disease Atlas — "
            "CFTR / SERPINA1 / SFTPC / TERT / FLCN / TSC1 / ABCA3 / NKX2-1 — "
            "320 Patients (8×40, Seeds 1710–1717)"
        ),
        "total_patients": sum(len(g["patients"]) for g in LUNG_GENES),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "aggregate_stats": {
            "genes_covered": len(LUNG_GENES),
            "patients_per_gene": 40,
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
        },
        "genes": genes,
        "top_alerts": [
            "CFTR-TRIKAFTA-FDA2019-TRANSFORMATIVE: elexacaftor/tezacaftor/ivacaftor approved 2019 for F508del ≥12y (now ≥2y); ppFEV1 +14pp; sweat Cl normalises; eligible ~90% CF patients; confirm genotype before prescribing",
            "CFTR-BURKHOLDERIA-CEPACIA-TRANSPLANT-RISK: BCC colonisation (especially B. cenocepacia) = relative contraindication to lung transplant; cepacia syndrome rapidly fatal; strict patient segregation mandatory",
            "SERPINA1-SMOKING-ABSOLUTE-PROHIBITION: PiZZ smoking accelerates emphysema 15-20 years; single most impactful intervention; lower lobe emphysema in young non-smoker = AATD until proven otherwise",
            "TERT-IMMUNOSUPPRESSION-CONTRAINDICATED-IPF: azathioprine/cyclophosphamide CONTRAINDICATED in TERT-associated pulmonary fibrosis — precipitates bone marrow failure; nintedanib acceptable; lung transplant preferred",
            "FLCN-NO-AVIATION-SCUBA-ABSOLUTE: aviation, scuba diving, high-altitude ABSOLUTELY PROHIBITED in BHD — bilateral lung cysts dramatically increase spontaneous pneumothorax risk; provide written medical alert",
            "TSC1-VEGF-D-800-LAM-PATHOGNOMONIC: serum VEGF-D >800 pg/mL = PATHOGNOMONIC for LAM; avoids lung biopsy; sirolimus FDA 2015 for LAM; start when FEV1 <70% or rapid decline",
            "ABCA3-LAMELLAR-BODIES-ABSENT-EM-DIAGNOSTIC: EM of lung biopsy showing absent/abnormal lamellar bodies = PATHOGNOMONIC for ABCA3 deficiency; null/null biallelic → fatal neonatal RDS; lung transplant only curative option",
            "NKX2-1-BRAIN-LUNG-THYROID-TRIAD-PATHOGNOMONIC: benign hereditary chorea + congenital hypothyroidism + ILD = PATHOGNOMONIC for NKX2-1; levothyroxine mandatory; chorea non-progressive; annual PFTs",
        ],
    }


def breakdown():
    """Return list of per-gene dicts with computed stats and sample patients."""
    result = []
    for idx, g in enumerate(LUNG_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "computed": {
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def definitions():
    """Return dict with concepts, pharmacological_distinctions, key_standards."""
    return {
        "concepts": {
            "Hereditary Lung Disease — Classification Framework": (
                "Hereditary lung diseases are monogenic disorders causing primary pulmonary pathology "
                "through surfactant dysfunction, protease-antiprotease imbalance, cystic airway disease, "
                "telomere biology failure, hamartoma formation, or transcription factor haploinsufficiency. "
                "MAIN CATEGORIES: "
                "1. CFTR-related disorders: CF (biallelic) and CFTR-related disorders (CRD, single allele + modifier); "
                "2. Surfactant dysfunction disorders: SFTPC (dominant-negative), ABCA3 (AR), NKX2-1 (haploinsufficiency), "
                "   SP-B (SFTPB, AR severe neonatal); "
                "3. Protease-antiprotease imbalance: SERPINA1 AATD; "
                "4. Telomere biology disorders: TERT, TERC, DKC1, RTEL1, PARN — familial IPF; "
                "5. Cystic lung / hamartoma syndromes: FLCN (BHD — lung cysts + RCC), TSC1/TSC2 (LAM + hamartomata); "
                "DIAGNOSTIC ALGORITHM FOR HEREDITARY ILD/EMPHYSEMA: "
                "Young patient + lower-lobe emphysema + panacinar: SERPINA1 (AATD) first; "
                "Familial IPF <60y or ≥2 affected relatives: telomere gene panel (TERT, TERC, RTEL1, PARN); "
                "Bilateral lung cysts + spontaneous pneumothorax ± fibrofolliculomas ± RCC: FLCN; "
                "Bilateral diffuse lung cysts exclusively in female: TSC1/2 (LAM) + VEGF-D; "
                "Term infant unexplained RDS: ABCA3/SFTPC/NKX2-1 panel; "
                "Familial ILD in children: SFTPC (dominant) or ABCA3 (AR); "
                "Chorea + hypothyroidism + ILD: NKX2-1."
            ),
            "CFTR Biology and the Modulator Revolution": (
                "CFTR is an ATP-gated chloride channel (ABCC7) at the apical surface of airway epithelia. "
                "The channel has 12 transmembrane domains, 2 nucleotide-binding domains (NBD1, NBD2), "
                "and a regulatory R domain. Channel opening requires: ATP binding at NBDs + R domain phosphorylation by PKA. "
                "MUTATION CLASSES and MODULATOR STRATEGY: "
                "Class I (nonsense/frameshift): no mRNA/protein — read-through agents (ataluren), or splice correction; "
                "Class II (misfolding/ER retention): F508del is the prototype — "
                "CORRECTORS (VX-661 tezacaftor, VX-445 elexacaftor) act as pharmacological chaperones → "
                "rescue F508del from ER → allow it to traffic to membrane; "
                "Class III (gating): G551D reaches membrane but gate stuck closed — "
                "POTENTIATORS (ivacaftor) pry channel gate open → restore Cl- flow; "
                "TRIKAFTA (triple therapy) = two correctors + one potentiator → "
                "corrects trafficking AND opens gate → most effective for F508del; "
                "Clinical impact: ppFEV1 +14 percentage points; sweat Cl falls from >90 to <40 mmol/L; "
                "exacerbations reduced 63%; BMI improved; quality of life transformed; "
                "median survival projected to exceed 60 years for F508del homozygous patients now on Trikafta."
            ),
            "Alpha-1 Antitrypsin Deficiency — Protease-Antiprotease Imbalance": (
                "The protease-antiprotease hypothesis explains AATD lung disease: "
                "Normal: AAT (serine protease inhibitor) in alveolar lining fluid inhibits neutrophil elastase (NE) "
                "released during normal neutrophil transit through lung capillaries. "
                "AATD: reduced AAT → uninhibited NE → progressive elastin destruction → "
                "panacinar emphysema preferentially in lower lobes (highest neutrophil traffic). "
                "LIVER DISEASE — DIFFERENT MECHANISM: "
                "Z-AAT (Glu342Lys) misfolds in ER → forms polymers → accumulates in hepatocyte ER → "
                "ER stress and hepatocyte apoptosis → cirrhosis. "
                "Null alleles produce NO protein → lung disease (no AAT) but NO liver disease (no polymer). "
                "This mechanistic distinction explains why: "
                "augmentation therapy (replacing secreted AAT) benefits lung only; "
                "liver disease requires separate management (RNAi fazirsiran reduces intrahepatic polymer); "
                "Pi phenotyping is critical — PiZZ and PiZnull have different liver risk despite both having low serum AAT. "
                "AUGMENTATION THERAPY — SLOW EMPHYSEMA PROGRESSION: "
                "RAPID trial (2015): IV augmentation weekly → CT emphysema density loss slower vs placebo; "
                "ERS/ATS guidelines: augmentation for PiZZ + FEV1 35-70% predicted + non-smoker; "
                "home infusion programs improve quality of life and adherence."
            ),
            "Telomere Biology Disorders — Why Immunosuppression Kills": (
                "Telomere biology disorders (TBD) arise when telomerase complex function is reduced ≥50% "
                "(TERT haploinsufficiency being most common adult cause). "
                "Consequence: accelerated telomere shortening in all rapidly dividing tissues. "
                "PULMONARY: alveolar type II pneumocytes have high turnover; short telomeres → "
                "senescence → impaired alveolar repair → progressive fibrosis (UIP pattern indistinguishable from IPF). "
                "BONE MARROW: haematopoietic stem cells most sensitive to short telomeres → "
                "cytopenias, aplastic anemia, MDS. "
                "THE IMMUNOSUPPRESSION TRAP: "
                "Clinicians treating TERT-associated pulmonary fibrosis may reach for azathioprine/CYC "
                "(as was used in IPF before PANTHER-IPF trial proved harm) — in TBD this is DOUBLY dangerous: "
                "1. Myelosuppression in already-compromised bone marrow → precipitate aplastic anemia; "
                "2. No immune target to suppress (fibrosis is not autoimmune in TBD); "
                "3. PANTHER-IPF 2012 showed AZA/NAC/prednisone increased IPF mortality; "
                "CORRECT APPROACH: identify TBD (telomere length + gene panel) before prescribing IS; "
                "nintedanib for fibrosis; danazol for bone marrow failure; lung transplant for end-stage IPF. "
                "TRANSPLANT CONSIDERATIONS: "
                "Increased post-transplant complications (cytopenias, SCC, hepatic VOD); "
                "haematology co-management mandatory; avoid full myeloablation if HSCT needed."
            ),
            "LAM — mTOR Addiction and the Sirolimus Revolution": (
                "Lymphangioleiomyomatosis (LAM) is caused by biallelic somatic inactivation of TSC2 "
                "(in LAM cells) in the context of germline TSC1 or TSC2 mutation (TSC-LAM) "
                "or purely somatic mutation (sporadic LAM). "
                "LAM cells are smooth-muscle-like cells with TSC2 biallelic inactivation → mTORC1 hyperactivation → "
                "uncontrolled proliferation + proteolytic cyst formation via matrix metalloproteinases. "
                "LAM cells spread haematogenously from uterus/lymphatics → implant in lung → "
                "progressive cyst formation → air trapping → obstructive physiology + pneumothorax. "
                "OESTROGEN DEPENDENCE: LAM occurs exclusively in females; oestrogen promotes TSC2 LOF cell survival; "
                "pregnancy and exogenous oestrogen may accelerate disease. "
                "SIROLIMUS (RAPAMYCIN) — DIRECT mTOR INHIBITOR: "
                "MILES trial (NEJM 2011): sirolimus vs placebo in LAM; "
                "FEV1 stable on sirolimus (+1 mL/y) vs declining on placebo (-12 mL/y); "
                "VEGF-D fell (LAM cell suppression biomarker); chylous effusions resolved; "
                "FDA approved 2015 for LAM; "
                "VEGF-D as diagnostic and monitoring biomarker: "
                ">800 pg/mL PATHOGNOMONIC for LAM (avoids biopsy); "
                "falls on sirolimus → confirms LAM cell suppression; rises on stopping → LAM recurrence."
            ),
        },
        "pharmacological_distinctions": [
            "CFTR — Trikafta (elexacaftor 200 mg / tezacaftor 100 mg / ivacaftor 150 mg): morning triple tablet + ivacaftor 150 mg evening; with fat-containing food; approved ≥2y for F508del or responsive mutation; most transformative CF therapy; liver toxicity monitoring; interactions (CYP3A4 inhibitors raise ivacaftor levels — dose adjust with azole antifungals)",
            "CFTR — Ivacaftor (Kalydeco) monotherapy: for Class III gating mutations (G551D, R117H, others on label); 150 mg BID; normalises sweat chloride; ppFEV1 +10pp; not for F508del alone (no trafficking benefit); available for infants ≥4 months (specific mutations)",
            "SERPINA1 — AAT Augmentation (Prolastin-C 60 mg/kg IV weekly): weekly IV infusion; targets serum trough >11 µmol/L; plasma-derived pooled human AAT; benefits lung only (NOT liver); home infusion programs available; monitor for infusion reactions; check IgA level (IgA-deficient patients may react to trace IgA in product)",
            "SFTPC/ABCA3 — Hydroxychloroquine (5 mg/kg/day): first-line empirical for surfactant dysfunction ILD; ophthalmology baseline + annual surveillance (cumulative retinal toxicity); QTc monitoring when combined with azithromycin; may take 3-6 months to show spirometric benefit; continue if tolerated and stabilising",
            "TERT — Danazol (600 mg/day in divided doses): synthetic androgen; upregulates telomerase expression via androgen-response element; Phase 2 trial showed telomere attrition slowed + haematological improvement; hepatotoxic (monthly LFTs); virilisation in females; not approved for this indication; use in bone marrow failure component",
            "TSC1-LAM — Sirolimus (rapamycin): 2 mg/day (titrate to trough 5-15 ng/mL); FDA-approved 2015 for LAM; stabilises FEV1; reduces VEGF-D; resolves chylothorax; monitor for stomatitis, infections, hyperlipidaemia, poor wound healing; hold peri-operatively; everolimus (1-10 mg/day) is alternative",
            "FLCN-BHD — No disease-modifying therapy approved: management is surveillance-based; renal MRI annually; pleurodesis for recurrent pneumothorax; nephron-sparing surgery at 3 cm renal tumour; mTOR inhibitors under investigation for renal disease; CO2 laser for skin fibrofolliculomas (cosmesis only)",
            "NKX2-1 — Levothyroxine: mandatory for congenital hypothyroidism (detected neonatal screen); titrate to normal TSH + fT4; neonates require close monitoring of levels; euthyroidism essential for neurodevelopment; lifelong; tetrabenazine or clonazepam for disabling chorea",
            "TERT-IPF — Nintedanib (150 mg BID with food): anti-fibrotic for TERT/familial IPF; reduces FVC decline ~50% vs placebo (INPULSIS data, extended to progressive fibrotic ILD); monitor LFTs; diarrhoea (manage with loperamide, dose reduction); NOT immunosuppressive — safe in TBD unlike IS",
            "ABCA3 null/null — Lung transplant (only curative): bilateral sequential lung transplant; performed in specialist paediatric centres; timing: early before multi-organ compromise; outcomes: 5-year survival ~50% in paediatric lung transplant; post-transplant graft produces normal type II pneumocytes",
        ],
        "key_standards": [
            "Ramsey et al. 2011 (NEJM) — Ivacaftor (VX-770) in CF with G551D: landmark Phase 3 trial demonstrating first CFTR potentiator benefit; ppFEV1 +10pp; sweat Cl -47 mmol/L; established proof-of-concept for modulator therapy; led to Trikafta development",
            "Heijerman et al. 2019 (Lancet) — Elexacaftor/Tezacaftor/Ivacaftor (ELX/TEZ/IVA) Phase 3: n=403 F508del heterozygous; ppFEV1 +14pp vs tezacaftor/ivacaftor; sweat Cl normalised; defined Trikafta as standard of care for eligible CF patients",
            "Dirksen et al. 2009 (Lancet) — RAPID Trial AATD Augmentation: first RCT showing CT lung density attenuation decline significantly slower with augmentation vs placebo in PiZZ AATD; established augmentation as evidence-based therapy for AATD lung disease",
            "McCormack et al. 2011 (NEJM) — MILES Trial Sirolimus for LAM: n=89; FEV1 stable (+1 mL/y sirolimus vs -12 mL/y placebo); VEGF-D and LAM-related symptoms improved; established sirolimus as FDA-approved standard of care for LAM 2015",
            "Armanios et al. 2016 (NEJM) — Danazol for Telomere Disease: Phase 2 trial; danazol 600 mg/day for 24 months in TBD; telomere attrition halted in 79%; haematological improvement; liver toxicity occurred — monthly LFT monitoring required; not approved but widely used off-label",
            "PANTHER-IPF Investigators 2012 (NEJM) — Three-Drug Regimen Harmful in IPF: prednisone + AZA + NAC increased mortality and hospitalisations vs placebo in IPF; established that immunosuppression is harmful in IPF (and by extension TBD-associated fibrosis); pivotal for TBD management guidance",
            "Gupta et al. 2015 (AJRCCM) — ABCA3 Mutations and ILD: landmark study defining mutation-phenotype correlation in ABCA3 (null/null = neonatal fatal; missense compound = childhood ILD); established EM lamellar body assessment as diagnostic gold standard",
            "Courcier et al. 2021 (ERJ) — BHD Syndrome Management: multi-centre BHD outcomes; fibrofolliculoma prevalence, pneumothorax recurrence (75% after first SP), renal tumour surveillance outcomes; pleurodesis recommendation after first SP; 3 cm renal threshold validated",
        ],
    }
