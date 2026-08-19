"""
NEU1 Epilepsy — Sialidosis Type I / Cherry-Red-Spot-Myoclonus Syndrome / α-Neuraminidase 1 Deficiency
========================================================================================================
40-patient cohort · NEU1 (6p21.33) · Autosomal recessive (AR) biallelic LOF
NEU1 encodes α-Neuraminidase 1 (sialidase 1; 415 aa ~46 kDa; lysosomal/plasma-membrane enzyme;
cleaves terminal α-2,3 and α-2,6 sialic acid residues from sialylated glycoproteins, glycolipids,
and oligosaccharides). NEU1 LOF → sialyloligosaccharide and sialoglycoprotein accumulation
→ lysosomal storage → progressive neurological disease.

SIALIDOSIS TYPE I = CHERRY-RED-SPOT-MYOCLONUS SYNDROME:
═══════════════════════════════════════════════════════
Onset 8-25y (mean 14y); AR biallelic NEU1 LOF; Progressive Myoclonic Epilepsy (PME) phenotype:
  - Action myoclonus (dominant feature, 98%) + GTCS + progressive cerebellar ataxia
  - NORMAL or near-normal cognition (KEY DISTINCTION from ALL NCL types and Sialidosis Type II)
  - PATHOGNOMONIC: cherry-red macular spot (90-95% of patients)
  - Visual acuity generally PRESERVED (unlike retinal NCL) — cherry-red spot ≠ visual failure
  - Progressive course: myoclonus → severe disability over 10-20 years
  - Survival to 5th-6th decade possible (much longer than NCLs)
  - OMIM: *608272 (NEU1) / #256550 (Sialidosis Type I)
  - Discovery: Cantz M et al. 1977 Eur J Biochem — first lysosomal sialidase deficiency;
    Federico A et al. 1980 — first Sialidosis Type I clinical series

SIALIDOSIS TYPE II (DYSMORPHIC/SEVERE — NOT THIS DASHBOARD):
  - Earlier onset (fetal/infantile/juvenile); biallelic NEU1 null
  - Coarse facies, hepatosplenomegaly, hydrops fetalis (severe)
  - Significant cognitive impairment; cherry-red spot present
  - Not the focus of this dashboard (Type I only, PME phenotype)

NEU1 PROTEIN BIOLOGY (LYSOSOMAL SIALIDASE — MULTIENZYME COMPLEX):
═══════════════════════════════════════════════════════════════════
NEU1 (6p21.33):
  - 415 amino acids; ~46 kDa; lysosomal and plasma-membrane sialidase
  - Contains Asp-box motifs (SXDXGXTW): Asp-box 1/2/3/4/5/6 — sialidase superfamily
  - Catalytic Tyr370 (active site nucleophile); Arg-Arg-Arg (catalytic Arg residues)
  - Cleaves α-2,3 and α-2,6 sialic acid linkages from:
    (a) Sialoglycoconjugates (glycoproteins, glycolipids)
    (b) Sialyloligosaccharides
    (c) Gangliosides (minor substrate)
  - NEU1 LOF → accumulation of sialyloligosaccharides + sialoglycoproteins → lysosomal storage
    → progressive neuronal dysfunction → PME phenotype
  - pLI ~0.91 (high LOF intolerance)
  - OMIM: *608272 (NEU1 gene) / #256550 (Sialidosis Type I)
  - Discovery: Cantz M & Gehler J 1977 Eur J Biochem — first lysosomal neuraminidase deficiency

NEU1-CTSA-GLB1 MULTIENZYME COMPLEX (CRITICAL FOR DIAGNOSIS AND UNDERSTANDING):
═══════════════════════════════════════════════════════════════════════════════
NEU1 operates as a multienzyme complex in the lysosome with:
  (1) CTSA / Cathepsin A / PPCA (Protective Protein Cathepsin A) — encoded by CTSA gene (20q13.12)
      → CTSA is REQUIRED for NEU1 stability, activation, and lysosomal targeting
      → Without CTSA: NEU1 remains inactive apo-enzyme (not folded/activated)
      → CTSA biallelic LOF → GALACTOSIALIDOSIS (combined NEU1 + GLB1 deficiency)
  (2) GLB1 / β-Galactosidase — encoded by GLB1 gene (3p22.3)
      → GLB1 cleaves terminal galactose from glycoconjugates
      → GLB1 biallelic LOF alone → GM1 Gangliosidosis / Mucopolysaccharidosis IVB
  - GALACTOSIALIDOSIS: CTSA LOF → BOTH NEU1 and GLB1 activity deficient
    (because CTSA is required to protect GLB1 from premature intralysosomal degradation)
    → Combined NEU1 + GLB1 enzyme deficiency = PATHOGNOMONIC for galactosialidosis
    → Sialidosis Type I = NEU1 only deficient (GLB1 normal)
  - DIAGNOSTIC IMPLICATION: When neuraminidase assay is deficient:
    → Test GLB1 simultaneously → if both NEU1+GLB1 deficient → GALACTOSIALIDOSIS (CTSA mutation)
    → If NEU1 only deficient → SIALIDOSIS (NEU1 mutation)

CHERRY-RED MACULAR SPOT — MECHANISM AND DIFFERENTIAL DIAGNOSIS:
══════════════════════════════════════════════════════════════
Cherry-red spot = cherry-red appearance of fovea against pale/grey surrounding macula:
  - Mechanism: storage material accumulates in macular ganglion cells (LARGE retinal ganglion cells)
    → pale/grey opaque macula (storage in large cells around fovea)
    → normal foveal vascular supply remains visible as cherry-red spot (fovea has NO large ganglion cells)
  - Sialidosis Type I: cherry-red spot (90-95%) + visual ACUITY PRESERVED
    (unlike retinal NCL where peripheral + central vision progressively lost)
  - Cherry-red spot differentials:
    (a) GM1 Gangliosidosis — infantile onset, coarse facies, no PME
    (b) GM2 Gangliosidosis (Tay-Sachs) — infantile onset, progressive neurodegeneration
    (c) Niemann-Pick Type A — infantile hepatosplenomegaly
    (d) Niemann-Pick Type C — cholesterol trafficking, vertical gaze palsy, cataplexy
    (e) MERRF (rare) — mitochondrial; cherry-red spot + PME + cognitive decline (VPA CI)
    (f) Normal variant (pseudo-cherry-red spot) — if uncertain → ERG + storage enzyme screen

CLINICAL COMPARISON WITH MERRF (MOST DANGEROUS PHENOCOPY — VPA CI IMPLICATIONS):
  MERRF (Myoclonic Epilepsy with Ragged-Red Fibres):
    - mtDNA m.8344A>G (80% of MERRF); mitochondrial inheritance
    - PME + myoclonus + ataxia + cognitive decline + cherry-red spot (rare) + deafness
    - VPA = ABSOLUTE CI (mitochondrial hepatotoxicity)
    - Muscle: ragged-red fibres (RRF) on Gomori trichrome
  Sialidosis Type I:
    - NEU1 biallelic LOF; AR inheritance
    - PME + action myoclonus + ataxia + NORMAL COGNITION + cherry-red spot (90-95%)
    - VPA = SAFE (lysosomal neuraminidase, NOT mitochondrial)
    - Urine: sialyloligosaccharides (TLC/tandem MS) = RAPID SCREEN
    - MANDATORY: exclude MERRF before VPA in any PME with cherry-red spot

DIAGNOSTIC PATHWAY — SIALIDOSIS TYPE I:
  (1) Fundoscopy → cherry-red spot? → if present + PME → Sialidosis Type I first differential
  (2) Urine sialic acid oligosaccharides (TLC/tandem MS) — 1-5 days, any metabolic lab
      ABNORMAL = rapid sialidosis screen (faster than enzyme assay)
  (3) Leukocyte α-neuraminidase assay + simultaneous GLB1 assay (4-hour stability)
      NEU1 only deficient → Sialidosis; both NEU1+GLB1 deficient → Galactosialidosis (CTSA mutation)
  (4) NEU1 WES / gene panel (SIMULTANEOUSLY with enzyme assay)
  (5) POLG1 + MERRF exclusion MANDATORY before VPA
"""

def get_overview():
    return {
        "gene": "NEU1 (6p21.33) — α-Neuraminidase 1 (Sialidase 1); 415 aa ~46 kDa; lysosomal/plasma-membrane sialidase; Asp-box sialidase superfamily; cleaves α-2,3 and α-2,6 sialic acid from sialoglycoproteins/glycolipids/oligosaccharides. OMIM *608272/#256550. Multienzyme complex with CTSA (required for NEU1 activation) and GLB1.",
        "protein": "α-Neuraminidase 1 (NEU1/Sialidase-1); 415 aa; ~46 kDa; lysosomal and plasma-membrane sialidase; 6 Asp-box motifs (SXDXGXTW); catalytic Tyr370 nucleophile; Arg-Arg-Arg catalytic triad; cleaves terminal α-2,3 and α-2,6 sialic acid from glycoconjugates and oligosaccharides; requires CTSA (cathepsin A/PPCA) for activation and lysosomal stability; NEU1 without CTSA = inactive apo-enzyme; forms multienzyme complex NEU1-CTSA-GLB1 in lysosome; pLI ~0.91 (high LOF intolerance)",
        "inheritance": "Autosomal recessive (AR) biallelic NEU1 LOF → Sialidosis Type I. 25% sibling recurrence risk. No AD form. Italian/North-African/Japanese founder variants most prevalent. Consanguinity in ~35% of severe cases.",
        "omim": "*608272 (NEU1 gene) · #256550 (Sialidosis Type I — Cherry-Red-Spot-Myoclonus Syndrome)",
        "disease": "Sialidosis Type I — Cherry-Red-Spot-Myoclonus Syndrome. NEU1 biallelic LOF → lysosomal α-neuraminidase deficiency → sialyloligosaccharide/sialoglycoprotein accumulation → progressive PME. Onset mean 14y (range 8-25y). PATHOGNOMONIC: cherry-red macular spot (90-95%) + PME with NORMAL COGNITION + adolescent/young adult onset. Action myoclonus (98%) + GTCS + cerebellar ataxia. Survival to 5th-6th decade (much longer than NCLs). Urine sialic acid oligosaccharides = rapid screening test. NO disease-modifying therapy (ERT research phase).",
        "mechanism": "NEU1 biallelic LOF → absent/severely reduced α-neuraminidase 1 activity in lysosomes → sialyloligosaccharides and sialoglycoproteins accumulate → lysosomal storage in neurons and other cells → progressive neuronal dysfunction → PME phenotype. Retinal ganglion cell storage → cherry-red macular spot (visual ACUITY generally preserved — unlike retinal NCL where peripheral+central vision lost). NEU1 is lysosomal — unlike DEGS1 (ER) or mitochondrial PME causes (MERRF/POLG) — VPA is SAFE. Multienzyme complex: NEU1 requires CTSA activation; CTSA LOF phenocopies combined NEU1+GLB1 deficiency (galactosialidosis) — critical diagnostic distinction.",
        "cherry_red_spot_note": "CHERRY-RED MACULAR SPOT + PME + NORMAL COGNITION + ADOLESCENT ONSET = SIALIDOSIS TYPE I UNTIL PROVEN OTHERWISE. Mechanism: sialyloligosaccharide storage in large macular ganglion cells → pale/grey opaque macula → normal foveal vasculature appears cherry-red (fovea has no large ganglion cells). Cherry-red spot present in 90-95% of Sialidosis Type I patients. VISUAL ACUITY GENERALLY PRESERVED (unlike retinal NCL where peripheral + central vision progressively lost). Cherry-red spot in PME: ONLY sialidosis combines cherry-red + PME + normal cognition + adolescent onset. Other cherry-red spot causes (GM1, GM2, Niemann-Pick A/C, MERRF) have different age/cognition/phenotype profiles.",
        "multienzyme_complex_note": "NEU1-CTSA-GLB1 MULTIENZYME COMPLEX: NEU1 requires CTSA (Cathepsin A/PPCA, 20q13.12) for lysosomal stability and activation. Without CTSA: NEU1 is inactive apo-enzyme. CTSA biallelic LOF (galactosialidosis) → BOTH NEU1 and GLB1 (β-galactosidase) deficient. DIAGNOSTIC RULE: when leukocyte neuraminidase is deficient, SIMULTANEOUSLY test GLB1: if both NEU1+GLB1 deficient → galactosialidosis (CTSA mutation); if NEU1 only deficient → sialidosis (NEU1 mutation). WES panel MUST include NEU1 + CTSA + GLB1 to avoid diagnostic error.",
        "normal_cognition_note": "NORMAL COGNITION IS THE DEFINING FEATURE OF SIALIDOSIS TYPE I — distinguishes from ALL NCL types (CLN1-13, all with universal cognitive decline) and Sialidosis Type II (significant intellectual disability). Adolescent/young adult with action myoclonus + GTCS + NORMAL IQ → Sialidosis Type I until excluded. Cognitive preservation also distinguishes from MERRF (variable cognitive impairment) and Niemann-Pick Type C (dementia). School performance, employment, and independent living are maintained in early Sialidosis Type I — major prognostic difference from NCL.",
        "cohort_size": 40,
        "female_pct": 52,
        "mean_onset_years": 14.2,
        "mean_diagnosis_delay_years": 4.8,
        "drug_resistant_pct": 58,
        "cherry_red_spot_pct": 93,
        "visual_acuity_preserved_pct": 78,
        "myoclonus_pct": 98,
        "cerebellar_ataxia_pct": 85,
        "gtcs_pct": 75,
        "on_vpa_pct": 85,
        "on_lev_pct": 72,
        "on_piracetam_pct": 60,
        "stimulus_sensitive_pct": 88,
        "photosensitivity_pct": 72,
        "cognitive_impairment_pct": 8,
        "discovery": "Cantz M & Gehler J (1977) Eur J Biochem — first identification of lysosomal sialidase deficiency in sialidosis; biochemical characterisation. Federico A et al. (1980) Neurol Sci — first Sialidosis Type I (cherry-red-spot-myoclonus syndrome) clinical series; established the PME + cherry-red spot + normal cognition triad. Bonten EJ et al. (1996) Genes Dev — CTSA/PPCA required for NEU1 activation; NEU1-CTSA-GLB1 multienzyme complex characterisation.",
        "unique_feature": "CHERRY-RED MACULAR SPOT + PME + NORMAL COGNITION TRIAD (unique among all PME syndromes). URINE SIALIC ACID OLIGOSACCHARIDES RAPID SCREEN (1-5 days) — fastest first-line test (unlike NCLs where DBS enzyme assay is primary). NEU1-CTSA-GLB1 MULTIENZYME COMPLEX — NEU1 requires CTSA for activation; CTSA LOF = galactosialidosis (NEU1+GLB1 both deficient). LEUKOCYTE NEURAMINIDASE ASSAY NOT A DBS TEST — requires fresh cells (4-hour stability). VISUAL ACUITY PRESERVED — cherry-red spot is NOT retinal NCL (no progressive visual failure as in NCL). LONGEST SURVIVAL AMONG LYSOSOMAL PME SYNDROMES (to 5th-6th decade). POLG1/MERRF EXCLUSION MANDATORY before VPA (MERRF = cherry-red + PME + VPA CI — most dangerous phenocopy).",
        "key_pharmacological_distinctions": {
            "1_URINE_SIALIC_ACID_OLIGOSACCHARIDES_RAPID_SCREEN": "URINE SIALIC ACID OLIGOSACCHARIDES = FASTEST SIALIDOSIS SCREEN (1-5 DAYS, ANY METABOLIC LAB): Urine thin-layer chromatography (TLC) or tandem mass spectrometry (LC-MS/MS) detects abnormal sialyloligosaccharide pattern — elevated/abnormal bands visible on TLC within 1-5 days in any metabolic genetics laboratory. This is FASTER and MORE ACCESSIBLE than the leukocyte neuraminidase enzyme assay (which requires fresh viable cells and specialist lysosomal biochemistry laboratory). ORDER SIMULTANEOUSLY with NEU1 WES/gene panel and leukocyte enzyme assay. A normal urine sialyloligosaccharide TLC does NOT exclude Sialidosis Type I (sensitivity ~85%) — enzyme assay and WES required regardless. This contrasts with CLN1 where PPT1 DBS enzyme assay is the primary test (not urine screen).",
            "2_LEUKOCYTE_NEURAMINIDASE_ASSAY_NOT_DBS": "LEUKOCYTE α-NEURAMINIDASE ASSAY REQUIRES FRESH CELLS — NOT A STANDARD DBS ASSAY (UNIQUE LOGISTICAL CHALLENGE): Unlike CLN1 (PPT1 DBS — posted to reference lab) and CLN2 (TPP1 DBS — posted to reference lab), the NEU1 enzyme assay requires FRESH LEUKOCYTES (4-hour stability post-phlebotomy) or CULTURED FIBROBLASTS. Logistical requirements: (1) Pre-arranged referral to specialist lysosomal biochemistry laboratory (same-day contact); (2) Blood drawn in EDTA at specialist centre or with coordinated same-day courier to lab; (3) Leukocyte separation and enzyme assay must begin within 4 hours of venesection; (4) Cannot be 'posted' to reference lab as a dried blood spot. SIMULTANEOUSLY test GLB1 (β-galactosidase) on the same specimen: if both NEU1+GLB1 deficient → galactosialidosis (CTSA gene); if NEU1 only deficient → Sialidosis (NEU1 gene). Document in ALL referral letters: 'NEU1 enzyme assay requires fresh leukocytes — coordinate with specialist lab on day of blood draw.'",
            "3_CHERRY_RED_SPOT_PME_PATHOGNOMONIC_COMBINATION": "CHERRY-RED SPOT + PME + NORMAL COGNITION + ADOLESCENT/YOUNG ADULT ONSET = SIALIDOSIS TYPE I UNTIL PROVEN OTHERWISE: The combination of (1) cherry-red macular spot, (2) PME phenotype (myoclonus + GTCS + ataxia), (3) NORMAL intelligence, and (4) onset in adolescence/young adulthood (8-25y) is PATHOGNOMONIC for Sialidosis Type I. No other PME syndrome combines all four features. Cherry-red spot DIFFERENTIALS in PME context: GM1 Gangliosidosis (infantile onset, coarse facies, NOT PME); GM2 Gangliosidosis (Tay-Sachs/Sandhoff — infantile, rapid degeneration); Niemann-Pick Type A (infantile, hepatosplenomegaly); Niemann-Pick Type C (any age, vertical gaze palsy, cataplexy, dementia, NOT normal cognition); MERRF (cherry-red rare, cognitive decline, VPA ABSOLUTE CI, mitochondrial). ONLY Sialidosis Type I = cherry-red + PME + NORMAL COGNITION + adolescent. Ophthalmology fundus exam (dilated fundoscopy) is the FASTEST clinical diagnostic clue — available in any eye department within hours.",
            "4_NORMAL_COGNITION_NCL_DIFFERENTIAL": "NORMAL COGNITION IN SIALIDOSIS TYPE I — KEY DISTINCTION FROM ALL NCL TYPES AND SIALIDOSIS TYPE II: All NCL types (CLN1-13) cause universal, progressive cognitive decline — this is a defining feature of NCL. Sialidosis Type I PRESERVES COGNITION (near-normal or normal IQ throughout early-mid disease course). This is the single most important clinical feature distinguishing Sialidosis Type I from NCL: (1) Adolescent with GTCS + action myoclonus + NORMAL school performance → Sialidosis Type I (NOT NCL); (2) Adolescent with GTCS + action myoclonus + COGNITIVE DECLINE → NCL, Sialidosis Type II, MERRF, or other PME with dementia. This distinction has major therapeutic implications: Sialidosis Type I patients maintain employment, relationships, and independent living far longer than NCL patients. It also prevents premature educational and social withdrawal. Document cognitive baseline at diagnosis; serial neuropsychological testing should demonstrate PRESERVATION, not decline.",
            "5_VGB_HIGH_RISK_NOT_ABSOLUTE_CI_CHERRY_RED_MACULAR": "VGB HIGH RISK (CAUTION) IN SIALIDOSIS TYPE I — NOT ABSOLUTE CI LIKE NCL, BUT AVOID IF POSSIBLE (CHERRY-RED MACULAR DISTINCTION): Sialidosis cherry-red spot = macular ganglion cell storage/loss. Visual acuity is GENERALLY PRESERVED in Sialidosis Type I (unlike retinal NCL where peripheral and central vision are progressively and catastrophically lost). VGB retinopathy predominantly affects peripheral visual fields (nasal retina, periphery) — NOT primarily the macula/central vision. Therefore VGB retinopathy risk in Sialidosis ≠ the same catastrophic bilateral blindness risk as VGB in retinal NCL (CLN1/CLN2/CLN3/CLN5/CLN6/CLN7/CLN8/CLN10/CLN11 — all ABSOLUTE CI). RULE: VGB is HIGH RISK in Sialidosis Type I (caution; avoid if possible), NOT an absolute CI. If VGB is considered (absolute last resort for focal status or infantile spasms overlap): ERG baseline + 3-monthly peripheral field monitoring is MANDATORY. HOWEVER: CBZ/OXC/PHT remain ABSOLUTE CI regardless (myoclonus worsening). VGB should not be routinely used in Sialidosis Type I PME.",
            "6_CTSA_MULTIENZYME_COMPLEX_DIAGNOSTIC_RELEVANCE": "NEU1-CTSA-GLB1 MULTIENZYME COMPLEX — GALACTOSIALIDOSIS PHENOCOPY EXCLUSION (CRITICAL DIAGNOSTIC STEP): CTSA (cathepsin A/PPCA, 20q13.12) is required for NEU1 lysosomal activation. CTSA biallelic LOF (GALACTOSIALIDOSIS) → BOTH NEU1 AND GLB1 (β-galactosidase) are deficient — because CTSA protects GLB1 from intralysosomal degradation AND activates NEU1. Galactosialidosis phenotype: late infantile-juvenile onset; coarse facies + macular cherry-red spot + PME + angiokeratoma. DISTINGUISHING RULE: when neuraminidase is deficient, SIMULTANEOUSLY measure GLB1: (1) NEU1 deficient + GLB1 NORMAL → SIALIDOSIS (NEU1 mutation); (2) NEU1 deficient + GLB1 ALSO DEFICIENT → GALACTOSIALIDOSIS (CTSA mutation). WES gene panel MUST include NEU1 + CTSA + GLB1. Diagnosing galactosialidosis incorrectly as sialidosis = different gene, different phenotype, different family counselling (both AR, 25% recurrence).",
            "7_CBZ_OXC_PHT_ABSOLUTE_CI_PME_MYOCLONUS_TRAP": "CBZ/OXC/PHT ABSOLUTE CI — SIALIDOSIS PME MYOCLONUS WORSENING TRAP (IDENTICAL TO ALL PME SYNDROMES): Sialidosis Type I PME onset in adolescence (14y mean) with GTCS and action myoclonus is FREQUENTLY MISIDENTIFIED as Juvenile Myoclonic Epilepsy (JME) or focal temporal/frontal epilepsy → CBZ/OXC/PHT prescribed → ACUTE MYOCLONIC DETERIORATION (Na-channel block worsens cortical PME myoclonus). Diagnosis delay mean 4.8 years = extended Na-channel blocker exposure window. MOST CRITICAL TEACHING POINT: ANY adolescent with myoclonus + GTCS who deteriorates on CBZ → URGENT cherry-red spot fundoscopy (most rapid Sialidosis clue). Cherry-red spot SHOULD be checked in ALL adolescent GTCS presentations with myoclonus — takes <5 minutes. Safe AEDs in Sialidosis: VPA + LEV + piracetam (PME-specific backbone).",
            "8_VPA_SAFE_LYSOSOMAL_NEURAMINIDASE_NOT_MITOCHONDRIAL_POLG1_MANDATORY_EXCLUSION": "VPA SAFE IN SIALIDOSIS TYPE I — NEU1 IS LYSOSOMAL α-NEURAMINIDASE (NOT MITOCHONDRIAL); POLG1/MERRF EXCLUSION MANDATORY BEFORE VPA: NEU1 = lysosomal enzyme (sialic acid cleavage). VPA ABSOLUTE CI applies to MERRF (mitochondrial myoclonus epilepsy, m.8344A>G) and POLG1 Alpers (mitochondrial DNA polymerase). VPA is the backbone PME AED in Sialidosis Type I — NOT contraindicated. HOWEVER: MERRF is the MOST DANGEROUS SIALIDOSIS PHENOCOPY because MERRF can also cause cherry-red spot (rare) + PME + progressive neurological disease. MANDATORY PROTOCOL before VPA in ANY adolescent PME with cherry-red spot: (1) Blood lactate; (2) m.8344A>G MERRF PCR (MERRF); (3) POLG1 WES; (4) Muscle biopsy (ragged-red fibres for MERRF if clinical suspicion). Once MERRF and POLG1 Alpers excluded → VPA is safe and first-line backbone AED for Sialidosis Type I. VPA hepatotoxicity in POLG1 = fatal — this exclusion is non-negotiable."
        }
    }


def get_breakdown():
    return {
        "etiologies": [
            {
                "class": "Homozygous-NEU1-Missense-Consanguineous-Italian-North-African-Founder",
                "pct": 32,
                "description": "Homozygous NEU1 missense variant (consanguineous ancestry; Italian/North African founder effect); specific founder variants: p.Trp249Leu (Italian), p.Arg294His (Japanese/Italian); severe α-neuraminidase deficiency in leukocytes; classic Sialidosis Type I PME phenotype with cherry-red spot",
                "typical_onset": "8-18 years (mean 14y)",
                "genotype_notes": "Founder variants identifiable by targeted PCR before full WES in appropriate ethnic background (Italian/North African consanguineous families); homozygous missense often retains partial residual enzyme activity (<5-10% control)"
            },
            {
                "class": "Compound-Het-Missense-Missense-Non-Consanguineous-European",
                "pct": 28,
                "description": "Compound heterozygous NEU1 missense/missense; non-consanguineous European ancestry; residual enzyme activity variable (5-30% control); attenuated Sialidosis Type I phenotype possible; slower progression; cherry-red spot present in 85-90%",
                "typical_onset": "12-25 years (later onset)",
                "genotype_notes": "Residual enzyme activity inversely correlates with phenotype severity; >20% residual activity → milder phenotype, slower progression; non-consanguineous requires full WES (no founder variant screening)"
            },
            {
                "class": "Compound-Het-Missense-Truncating-Moderate-Severe",
                "pct": 22,
                "description": "Compound heterozygous NEU1 missense + truncating (frameshift/nonsense/splice); moderate-severe phenotype; missense allele provides partial enzyme function; truncating allele = null contribution; earlier onset than missense/missense",
                "typical_onset": "8-16 years",
                "genotype_notes": "Phenotype determined by missense allele residual activity; truncating allele = complete null on one chromosome; compound het requires phase confirmation (trans configuration); ACMG classification: missense pathogenic if enzyme assay confirms deficiency"
            },
            {
                "class": "Homozygous-Truncating-Null-Severe-Type-II-Overlap",
                "pct": 10,
                "description": "Homozygous NEU1 truncating variant (frameshift/nonsense); complete enzyme absence; most severe phenotype in Type I cohort; may overlap with Type II features (mild dysmorphism, early onset); onset at lower end of Type I age range; consanguineous in 80%",
                "typical_onset": "6-12 years (earliest Type I onset)",
                "genotype_notes": "Zero neuraminidase activity; most severe Type I disease; may have mild hepatosplenomegaly not meeting full Type II criteria; consanguinity essential family history"
            },
            {
                "class": "Deep-Intronic-Splicing-Variant-RNA-Studies-Required",
                "pct": 5,
                "description": "Deep intronic or splice-site NEU1 variants; missed by standard exome; require mRNA/cDNA studies or genome sequencing; partial splicing defect → partial residual enzyme activity; attenuated phenotype; long diagnostic odyssey common",
                "typical_onset": "16-25 years (attenuated, late)",
                "genotype_notes": "Partial splicing = residual enzyme activity; milder presentation; RNA studies (RT-PCR from fibroblast/blood) needed to confirm pathogenicity of deep intronic variant; ACMG PS3 functional evidence via enzyme assay"
            },
            {
                "class": "Phenocopy-NEU1-Negative-Galactosialidosis-CTSA-Exclusion-Required",
                "pct": 3,
                "description": "Sialidosis-phenotype (cherry-red spot + PME) with NEU1 enzyme deficiency but NO NEU1 mutation on WES; enzyme deficiency caused by CTSA mutation (galactosialidosis — both NEU1+GLB1 deficient); or NEU1-negative cherry-red PME requiring alternative diagnosis",
                "typical_onset": "Variable (phenocopy-dependent)",
                "genotype_notes": "CRITICAL: if NEU1 enzyme deficient + GLB1 also deficient → galactosialidosis (CTSA WES); if NEU1 only deficient → confirm NEU1 WES; if WES-negative → re-test enzyme + consider genome sequencing + consider alternative diagnosis (NPC, MERRF)"
            }
        ],
        "seizure_types": [
            {
                "type": "Action Myoclonus (Cortical PME)",
                "prevalence_pct": 98,
                "eeg_pattern": "Cortical giant SEPs on jerk-locked back-averaging; irregular polyspike or polyspike-wave complexes (high-amplitude); photosensitive myoclonus enhancement (PPR grade III-IV in 72%); enhanced cortical excitability; jerk-locked averaging confirms cortical origin; EEG relatively preserved amplitude (unlike NCL where progressive decrement)",
                "semiology": "Action myoclonus: jerks triggered by voluntary movement (reaching, walking, writing); stimulus-sensitive (light, sound, touch); morning worsening; facial + upper limb + trunk predominance; intention tremor-like in fine motor tasks; interferes with writing, eating, walking; gait severely impacted by myoclonus + ataxia combination",
                "clinical_tips": "PIRACETAM IS MOST EFFECTIVE DRUG FOR SIALIDOSIS ACTION MYOCLONUS (Level B; 16-24 g/day). Add early before myoclonus disables ADLs. Piracetam + VPA + LEV combination achieves best myoclonus control. Myoclonus action diary (UMRS) essential for titration. COGNITION PRESERVED — patient can self-report myoclonus severity (unlike NCL)."
            },
            {
                "type": "GTCS (Generalised Tonic-Clonic Seizures)",
                "prevalence_pct": 75,
                "eeg_pattern": "Generalised irregular spike-and-wave or polyspike-wave (2-4 Hz); generalised onset; nocturnal predominance in ~60%; photosensitive GTCS enhancement; relatively preserved EEG background amplitude (normal cognition correlate)",
                "semiology": "Tonic phase (10-30 sec) → clonic phase (30-90 sec); generalised onset; nocturnal clustering; post-ictal confusion relatively brief (preserved cognition); may cluster with sleep deprivation or missed AED; onset GTCS triggers diagnostic workup in most patients (cherry-red spot not yet identified)",
                "clinical_tips": "First GTCS at age 14 (mean) triggers workup → ALWAYS check fundus for cherry-red spot in adolescent GTCS with myoclonus. GTCS at 14y misidentified as JME in ~65% → CBZ prescribed (ABSOLUTE CI) → myoclonus worsening → delayed Sialidosis diagnosis. VPA is FIRST-LINE for Sialidosis GTCS."
            },
            {
                "type": "Stimulus-Sensitive Myoclonus",
                "prevalence_pct": 88,
                "eeg_pattern": "Time-locked polyspike response to photic, auditory, or tactile stimulus; PPR type II-IV on EEG with IPS; cortical excitability hyperresponsiveness; reflex myoclonic jerk correlates with giant cortical SEP",
                "semiology": "Myoclonic jerks triggered by flashing lights, loud sounds, unexpected touch, sudden movement; reflex myoclonus (non-voluntary); distinguishable from action myoclonus (triggered by active movement); both types co-exist in Sialidosis Type I; stimulus-sensitive myoclonus is a cardinal diagnostic feature",
                "clinical_tips": "AVOID STROBE/FLICKER ENVIRONMENTS (photosensitivity 72%). Dark glasses in bright environments. Auditory startle management: ear protection in noisy settings. Stimulus-sensitive myoclonus assessment by EEG with IPS + auditory + tactile probes. VPA + LEV most effective for stimulus-sensitive myoclonus. Piracetam primarily targets ACTION myoclonus."
            },
            {
                "type": "Focal Impaired Awareness Seizures (Temporal/Occipital)",
                "prevalence_pct": 35,
                "eeg_pattern": "Temporal or occipital spike-wave focus; may secondarily generalise; temporal: theta slowing + sharp waves; occipital: posterior paroxysmal activity (correlating with macular cherry-red spot involvement)",
                "semiology": "Déjà vu, olfactory or visual aura; impaired awareness; automatisms (lip-smacking, hand fumbling); may secondarily generalise; visual aura from occipital foci correlates with cherry-red macular involvement; brief post-ictal confusion (cognition preserved)",
                "clinical_tips": "Focal features in Sialidosis Type I misidentified as temporal lobe epilepsy → CBZ prescribed (ABSOLUTE CI). NEVER initiate CBZ for adolescent focal epilepsy without cherry-red spot fundoscopy. VPA + LEV effective for focal seizures in Sialidosis. LTG adjunct for refractory focal features (never monotherapy — myoclonus risk)."
            },
            {
                "type": "Non-Convulsive Status Epilepticus (NCSE)",
                "prevalence_pct": 18,
                "eeg_pattern": "Continuous or near-continuous spike-wave for >30 minutes; altered consciousness; EEG diagnosis essential; subclinical NCSE common in advanced disease; cognitive preservation means NCSE-related behavioural change may be more readily noticed (patient can report 'feeling different')",
                "semiology": "Reduced responsiveness + confusion + subtle myoclonus + blank staring; may be missed in early disease (cognition preserved → subtle change noticed); acute cognitive deterioration beyond baseline → urgent EEG; more common in drug-resistant disease phase",
                "clinical_tips": "TGB (tiagabine) ABSOLUTE CI — GABA reuptake inhibitor precipitates NCSE in PME/generalised epilepsy. If acute confusional state in Sialidosis Type I → urgent EEG before attributing to disease progression. IV LEV (40-60 mg/kg) first-line NCSE rescue. Midazolam IM/buccal for acute NCSE emergency. NCSE diagnosis may be particularly important in Sialidosis Type I because preserved cognition means ANY confusional episode is ABNORMAL."
            }
        ],
        "triggers": [
            {"trigger": "Voluntary Movement (Action Myoclonus Trigger)", "prevalence_pct": 98, "mechanism": "Action myoclonus: voluntary motor cortex activation → cortical giant potential → myoclonic jerk; reaching, walking, writing, eating all trigger; cortical hyperexcitability from sialyloligosaccharide storage → abnormal motor cortex excitability", "management": "Piracetam (16-24 g/day) specifically targets action myoclonus; occupational therapy to adapt fine motor activities; weighted utensils; myoclonus action diary (UMRS); VPA + LEV + piracetam combination; physiotherapy gait adaptation; non-slip surfaces"},
            {"trigger": "Photic Stimulation (Photosensitivity)", "prevalence_pct": 72, "mechanism": "PPR grade III-IV on EEG IPS; abnormal cortical visual hyperexcitability; macular cherry-red spot correlates with photic stimulus-sensitive myoclonus via visual cortex hyperexcitability pathway", "management": "Avoid strobes/flickering lights; dark glasses in bright environments; screen brightness maximum; blue-light filter glasses; TV/gaming limits; photosensitivity documented in all medical records; school environment audit (fluorescent lighting)"},
            {"trigger": "Sleep Deprivation", "prevalence_pct": 68, "mechanism": "Reduced cortical inhibitory reserve with sleep deprivation → lowered seizure threshold; action myoclonus severity increases significantly with fatigue; nocturnal GTCS frequency increases with sleep disruption", "management": "Regular sleep schedule mandatory (target 9 hours in adolescence); melatonin for sleep onset (safe in PME); school schedule adapted to later morning start; avoid overnight events; GTCS nocturnal alarm (bed sensor) for safety monitoring"},
            {"trigger": "Emotional Stress / Excitement", "prevalence_pct": 62, "mechanism": "Adrenergic arousal + HPA axis activation → increased cortical excitability → myoclonus precipitation; preserved cognition means full emotional reactivity (unlike NCL patients); excitement (social events, sports) and anxiety both trigger", "management": "Structured routine; CBT/psychological support (preserved cognition enables therapy); SSRI for anxiety comorbidity; school exam schedule modification; advance planning for anticipated high-stress events; patient self-monitoring (preserved cognition)"},
            {"trigger": "Auditory Startle", "prevalence_pct": 58, "mechanism": "Cortical hyperexcitability → startle-reflex hyperresponsiveness; loud sudden noise → cortical giant SEP → myoclonic jerk; auditory startle myoclonus assessed by EEG (time-locked cortical response)", "management": "Low-sensory environments at school and home; ear protection in noisy settings; EEG auditory startle assessment; CLB PRN for startle episodes; caregiver and employer education (noise in workplace); hearing protection during sports"},
            {"trigger": "Missed AED Dose", "prevalence_pct": 65, "mechanism": "Any gap in AED coverage → acute seizure cluster (action myoclonus surge + GTCS risk); VPA dose-dependent effect; Sialidosis Type I drug-resistant baseline (58%) means any further AED reduction is destabilising; preserved cognition means patient can self-monitor", "management": "Dual packaging (home + work/school); mobile phone reminders; simplified dosing (once-daily modified-release formulations where possible); VPA modified-release preferred; patient education (cognition preserved allows self-management)"},
            {"trigger": "Fever (pyrexia ≥37.5°C)", "prevalence_pct": 45, "mechanism": "Fever lowers seizure threshold; metabolic stress + ion channel temperature sensitivity → cortical hyperexcitability; Sialidosis neurons have reduced inhibitory reserve; febrile myoclonus clusters less common than fever-triggered GTCS", "management": "Antipyretics early (paracetamol/ibuprofen at 37.5°C — lower threshold than general advice); written fever action plan with rescue midazolam; employer/school awareness; paracetamol stock at workplace (adult patients with preserved employment)"},
            {"trigger": "Contraindicated Drug Exposure", "prevalence_pct": 100, "mechanism": "CBZ/OXC/PHT → acute Na-channel block → myoclonus worsening in PME (cortical hyperexcitability increased by Na-channel state-dependent blocking); TGB → GABA reuptake inhibition → NCSE in generalised epilepsy; VGB → peripheral field restriction (cherry-red macular involvement adds risk)", "management": "MedicAlert bracelet listing Sialidosis CIs; AED card (GP/hospital/employer); all prescribers reviewed at each encounter; pharmacist medication reconciliation; patient education (preserved cognition enables active participation in medication safety)"}
        ],
        "treatments": [
            {
                "drug": "Valproate / Sodium Valproate (VPA)",
                "level": "Level B",
                "dose": "Adult/adolescent: 20-40 mg/kg/day (target trough 60-100 µg/mL); modified-release formulation preferred (Epilim Chrono, Depakote ER); BD dosing; initiate 200-400 mg BD, titrate monthly",
                "moa": "Broad-spectrum AED: Na-channel block + GABA enhancement + T-type Ca2+ block + HCN channel effects; full-spectrum PME coverage (GTCS + myoclonic + absence + focal); most effective single agent for Sialidosis PME GTCS component",
                "efficacy": "GTCS reduction ~70-80% in Sialidosis PME; myoclonus reduction ~50% (requires piracetam combination for optimal myoclonus control); backbone AED for all Sialidosis PME; maintained efficacy over years",
                "monitoring": "TDM trough q6m; LFT + FBC q6m; weight; ammonia if encephalopathic; VPPP counselling for females of reproductive age (pregnancy prevention programme mandatory — teratogenic); platelet count (thrombocytopenia risk)",
                "neu1_note": "VPA SAFE in Sialidosis Type I (NEU1 = lysosomal enzyme, NOT mitochondrial). MANDATORY before VPA: blood lactate + m.8344A>G MERRF PCR + POLG1 WES (MERRF/POLG are the most dangerous phenocopies — VPA causes fatal hepatotoxicity in POLG1 Alpers). Once MERRF/POLG excluded: VPA is backbone AED."
            },
            {
                "drug": "Levetiracetam (LEV)",
                "level": "Level B",
                "dose": "Adult/adolescent: 250 mg BD increasing to 500-1500 mg BD (target 1000-3000 mg/day); weight-based paediatric: 20-60 mg/kg/day; IV LEV for SE: 60 mg/kg loading (max 4500 mg); extended-release formulation for once-daily dosing",
                "moa": "SV2A synaptic vesicle protein modulation → reduces neurotransmitter release from hyperexcitable neurons; broad-spectrum PME coverage without Na-channel blockade; IV formulation available for SE",
                "efficacy": "Synergistic with VPA in Sialidosis PME (VPA+LEV+piracetam combination best myoclonus control); IV LEV essential for SE rescue when fosphenytoin/phenytoin are CI; adjunct GTCS reduction additional 20-30% on VPA monotherapy",
                "monitoring": "No routine TDM; behavioural side effects (irritability, mood changes) monitored — Sialidosis patients have preserved cognition and can self-report; dose reduce if significant irritability; no hepatotoxicity; renal dose adjustment if eGFR <50",
                "neu1_note": "IV LEV replaces fosphenytoin in all Sialidosis Type I SE protocols (fosphenytoin = ABSOLUTE CI — Na-channel blocker worsens PME myoclonus). Patient-reported outcome monitoring for behavioural effects is feasible in Sialidosis Type I (preserved cognition) — use validated PRO tools."
            },
            {
                "drug": "Piracetam",
                "level": "Level B",
                "dose": "Adult: 16-24 g/day in divided doses (BD or TDS); initiate at 4 g/day, increase weekly by 4 g until response or max 24 g/day; BID or TID divided dosing; paediatric: 8-16 g/day",
                "moa": "AMPA-receptor modulation + membrane fluidity effects; reduces cortical hyperexcitability at motor planning level; specifically attenuates cortical action myoclonus (cortical origin confirmed by jerk-locked SEP); antiplatelet effect (mild)",
                "efficacy": "Level B evidence for action myoclonus in PME (strongest evidence base in MERRF/Sialidosis/EPM1 — extrapolated from Marseille school and Italian PME trials); 60-70% reduction in action myoclonus severity at 16-24 g/day; MOST TARGETED anti-myoclonic agent available for Sialidosis Type I",
                "monitoring": "Well-tolerated at PME doses; coagulation if used with antiplatelet therapy (mild antiplatelet effect); no hepatotoxicity; no renal dose requirement at standard doses; cognitive monitoring (preserved cognition in Sialidosis = can use self-reported UMRS); no TDM",
                "neu1_note": "Piracetam is the MOST EFFECTIVE drug for Sialidosis Type I action myoclonus — this is the defining treatment distinction from NCL (where piracetam evidence is extrapolated/Level C). Level B evidence from Italian PME/cherry-red-spot myoclonus trials. Start EARLY before action myoclonus disables ADLs. The triad VPA+LEV+piracetam is the Sialidosis Type I AED backbone."
            },
            {
                "drug": "Perampanel (PER)",
                "level": "Level B",
                "dose": "Adult: 2 mg nocturnal, titrate fortnightly by 2 mg (max 12 mg/day); nocturnal administration reduces dizziness/drowsiness side effects; target 4-8 mg/day for Sialidosis PME GTCS",
                "moa": "AMPA receptor antagonist (post-synaptic glutamate); reduces neuronal hyperexcitability via glutamate antagonism; different MOA from VPA/LEV → true add-on without pharmacodynamic redundancy; approved for PME (generalised myoclonic seizures) as adjunct",
                "efficacy": "ILAE Level B for adjunctive PME treatment (GTCS and myoclonic seizures); perampanel add-on to VPA+LEV achieves additional 30-40% GTCS reduction in drug-resistant Sialidosis PME; specific AMPA mechanism beneficial in PME where AMPA receptor upregulation is part of cortical hyperexcitability",
                "monitoring": "Dizziness, somnolence, aggressive behaviour (dose-dependent); NSAID interaction (ibuprofen increases perampanel levels); psychiatric history review before initiation; ECG not required; nocturnal administration preferred (peak sedation overnight); MHRA black triangle — PRO mood monitoring",
                "neu1_note": "Perampanel AMPA antagonism particularly relevant in Sialidosis: cortical PME myoclonus partly driven by AMPA receptor hyperexcitability. Perampanel + piracetam (AMPA modulator/antagonist dual approach) + VPA + LEV = quadruple PME backbone for drug-resistant Sialidosis Type I. Perampanel APPROVED for PME (generalised myoclonic seizures) — not off-label in this context."
            },
            {
                "drug": "Clobazam (CLB)",
                "level": "Level B",
                "dose": "Adolescent/adult: 10-30 mg/day (max 40 mg in adults); BD dosing; nocturnal GTCS → night-time dose bias (15-20 mg nocturnal); PRN use for predictable high-trigger periods (travel, examinations)",
                "moa": "1,5-benzodiazepine; GABA-A positive allosteric modulator (alpha-2/alpha-3 subunit selective vs diazepam); less sedation than 1,4-BZDs; broad-spectrum PME coverage",
                "efficacy": "Adjunct for refractory GTCS and myoclonus; nocturnal GTCS reduction in Sialidosis; tolerance may develop (drug holiday every 3 months); preserved cognition allows rational drug holiday management by patient; better tolerability than clonazepam",
                "monitoring": "Sedation (reduced with nocturnal dosing); tolerance; interactions with VPA (displacement at protein binding); no TDM; PRN use for predictable triggers (rational use possible in preserved-cognition Sialidosis patients); VPPP if female",
                "neu1_note": "CLB nocturnal dosing effective for nocturnal GTCS in Sialidosis Type I. Preserved cognition enables patient-directed drug holiday (unlike NCL where carer-managed). PRN CLB for high-trigger periods (travel, exams, illness) is a rational strategy in Sialidosis — not feasible in NCL/PME with cognitive impairment."
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "Level C",
                "dose": "Classic 3:1 or 4:1 ratio (fat:carbohydrate+protein); MAD or LGIT alternatives in adults; initiated under specialist dietitian supervision; target serum BHB 2-4 mmol/L; dietitian-supervised at KD centre",
                "moa": "Metabolic ketosis → alternative neural energy substrate; reduces glucose-dependent neuronal excitability; possible effects on sialyloligosaccharide metabolism (theoretical — not confirmed); mitochondrial biogenesis enhancement; reduces cortical excitability via multiple mechanisms",
                "efficacy": "Level C for drug-resistant PME; 30-50% reduction in GTCS in drug-resistant Sialidosis Type I; adult KD (MAD/LGIT) more feasible than classic KD given preserved cognition and adult independent living; maintained long-term adherence better than in cognitively impaired NCL patients",
                "monitoring": "Lipid profile (3-monthly initially); renal stone risk (ultrasound annually); bone density (DEXA annually); weight; glucose; HbA1c; KD-specific micronutrient monitoring (Se, Zn, carnitine); monthly dietitian review initially; KD holiday planning (weddings, travel — feasible with preserved cognition)",
                "neu1_note": "Adult KD (MAD — modified Atkins diet) is particularly suitable for Sialidosis Type I given preserved cognition: patient can manage diet independently, count ratios, plan meals. This is a major advantage over NCL (where KD requires carer administration). KD dietitian centre with adult PME experience essential — different from paediatric KD teams."
            },
            {
                "drug": "MDT Care (Ophthalmology + Physiotherapy + OT + Neuropsychology + Genetic Counselling)",
                "level": "Level A",
                "dose": "Ophthalmology 6-monthly (fundus + ERG + visual acuity + peripheral fields); physiotherapy weekly (ataxia + myoclonus gait); SARA + UMRS 6-monthly; neuropsychology annual (cognitive preservation monitoring); genetic counselling at diagnosis + family cascade; employment support",
                "moa": "Multidisciplinary rehabilitation: preserved cognition = richer MDT engagement than NCL; employment rehabilitation, driving assessment, psychological support, relationship/fertility counselling — all feasible in Sialidosis Type I (not appropriate in NCL with dementia)",
                "efficacy": "Level A for MDT in rare progressive neurological conditions; Sialidosis Type I MDT distinct from NCL MDT: employment + driving + relationships + independent living are realistic goals (preserved cognition); fall prevention (ataxia + myoclonus); cherry-red monitoring",
                "monitoring": "SARA (cerebellar ataxia), UMRS (myoclonus), ophthalmology (cherry-red + ERG + visual acuity 6-monthly), cognitive assessments (annual — verify preservation), employment + driving + social function (annual), genetic counselling (family cascade), SUDEP risk (annual)",
                "neu1_note": "SIALIDOSIS TYPE I MDT IS FUNDAMENTALLY DIFFERENT FROM NCL MDT: preserved cognition allows EMPLOYMENT REHABILITATION, DRIVING ASSESSMENT, RELATIONSHIP AND FERTILITY COUNSELLING — all major quality-of-life components impossible in NCL. Neuropsychologist role: verify cognitive preservation (not track decline). Employment advisor involvement from diagnosis. Driving — seizure-freedom criteria apply (standard DVLA/DVSA rules); preserved cognition = driving licence may be maintained if seizure-controlled."
            },
            {
                "drug": "Rescue Midazolam (Buccal/IM) + IV LEV (SE Protocol)",
                "level": "Level A",
                "dose": "Buccal midazolam: 5-10 mg (adult); IM midazolam: 0.15-0.3 mg/kg; IV LEV SE loading: 60 mg/kg (max 4500 mg); sequence: buccal midazolam → IV LEV → IV phenobarbitone (NOT fosphenytoin)",
                "moa": "Midazolam: rapid GABA-A potentiation → seizure termination; IV LEV: SV2A modulation → SE interruption; replaces fosphenytoin (ABSOLUTE CI) in Sialidosis SE protocol",
                "efficacy": "Standard of care for acute seizure rescue and SE in PME; buccal midazolam equivalent to IV diazepam for prolonged seizures; IV LEV replaces fosphenytoin (Na-channel blocker — ABSOLUTE CI in PME); adult patient (preserved cognition) can be trained in midazolam self-administration protocol",
                "monitoring": "Respiratory depression monitoring after buccal midazolam; individualised rescue plan; employer/partner training in midazolam administration; A&E Sialidosis alert card (fosphenytoin CI explicitly listed); patient can carry own rescue pack (preserved cognition)",
                "neu1_note": "FOSPHENYTOIN ABSOLUTE CI in Sialidosis Type I SE. Adult Sialidosis patients with preserved cognition can be taught to self-administer buccal midazolam or train partners/colleagues — this is a major safety advantage vs NCL. SE protocol must be in all A&E records, employer first-aid instructions, and medical alert documentation."
            }
        ],
        "contraindications": [
            {
                "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
                "severity": "ABSOLUTE CI",
                "reason": "Na-channel blockers WORSEN cortical PME myoclonus in Sialidosis Type I; GTCS at age 14y misidentified as JME or focal epilepsy → CBZ → acute myoclonic deterioration; diagnosis delay mean 4.8y = extended Na-channel exposure window",
                "note": "THE MOST COMMON REAL-WORLD ERROR IN SIALIDOSIS TYPE I: adolescent GTCS + myoclonus → JME presumed → CBZ → myoclonus worsening. ANY adolescent on CBZ/OXC who develops action myoclonus → urgent cherry-red spot fundoscopy → Sialidosis workup. Safe alternatives: VPA + LEV + piracetam."
            },
            {
                "drug": "Fosphenytoin (IV) / Phenytoin (IV)",
                "severity": "ABSOLUTE CI",
                "reason": "IV Na-channel blocker used in standard SE protocols → acute myoclonus worsening and GTCS exacerbation in Sialidosis PME; standard SE second-line drug = catastrophic in Sialidosis SE",
                "note": "Sialidosis SE protocol: IV LEV (second-line) REPLACES fosphenytoin. Embed in A&E protocol, hospital notes, employer first-aid. Adult Sialidosis patients can carry written SE protocol card (preserved cognition). Paramedics and A&E must be informed at every hospital contact."
            },
            {
                "drug": "Tiagabine (TGB)",
                "severity": "ABSOLUTE CI",
                "reason": "GABA reuptake inhibitor → NCSE in generalised epilepsy syndromes; Sialidosis Type I has 18% NCSE risk; TGB absolutely forbidden in PME/generalised epilepsy",
                "note": "TGB approved only for focal seizures in adults; causes NCSE in PME including Sialidosis. Sialidosis focal features (temporal/occipital seizures) may lead non-specialist to prescribe TGB → NCSE → acute confusional state. A&E: urgent EEG if Sialidosis patient presents with confusion (NCSE vs disease)."
            },
            {
                "drug": "Vigabatrin (VGB)",
                "severity": "HIGH RISK",
                "reason": "Cherry-red macular spot = macular ganglion cell storage/loss; VGB retinopathy affects peripheral visual fields; visual acuity PRESERVED in Sialidosis Type I (unlike retinal NCL where VGB = ABSOLUTE CI); VGB is HIGH RISK (avoid if possible) but NOT absolute CI since Sialidosis does not cause progressive peripheral retinal NCL degeneration",
                "note": "VGB is HIGH RISK in Sialidosis Type I — not absolute CI. If VGB considered (last resort focal SE): mandatory ERG baseline + 3-monthly peripheral field monitoring. Contrast with NCL (CLN1/CLN2/CLN3/CLN5/CLN6/CLN7/CLN8/CLN10/CLN11) where VGB = ABSOLUTE CI due to peripheral+central retinal NCL degeneration."
            },
            {
                "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
                "severity": "HIGH RISK",
                "reason": "Alpha-2-delta calcium channel modulation can paradoxically worsen cortical myoclonus in PME; GBP/PGB may be prescribed by pain specialists or GPs for myoclonus-related pain/discomfort without awareness of Sialidosis PME",
                "note": "Preserved cognition in Sialidosis = adult healthcare access across multiple specialties; pain team, orthopaedics, GP may prescribe GBP/PGB for musculoskeletal pain from ataxia without knowing PME diagnosis. Document Sialidosis CI in ALL specialty correspondence. If GBP/PGB prescribed: close myoclonus monitoring; stop if myoclonus worsens."
            },
            {
                "drug": "Lamotrigine (LTG) Monotherapy",
                "severity": "HIGH RISK",
                "reason": "LTG monotherapy at high doses paradoxically worsens cortical myoclonus in PME; safe only as low-dose adjunct to VPA+LEV for focal features; Sialidosis focal seizures (35%) may prompt LTG monotherapy attempt → myoclonus worsening",
                "note": "NEVER LTG monotherapy in Sialidosis Type I PME. LTG adjunct (25-100 mg/day) added to VPA+LEV backbone can help focal temporal/occipital seizures. Slow titration mandatory with VPA co-prescription (Stevens-Johnson risk — VPA inhibits LTG glucuronidation → higher LTG levels)."
            },
            {
                "drug": "AED Taper / Abrupt Discontinuation",
                "severity": "HIGH RISK",
                "reason": "Sialidosis Type I is a progressive disease — seizures do NOT remit; AED taper = inevitable severe myoclonus cluster + GTCS + SUDEP risk; 58% drug-resistant baseline means any AED reduction is high-risk",
                "note": "Preserved cognition: patient may self-discontinue AEDs (feeling well, employment concerns, fertility planning). Patient education critical: Sialidosis Type I is NOT self-limited — AEDs are lifelong. Pregnancy planning: VPPP mandatory for VPA; LTG preferred in pregnancy (low teratogenicity); specialist PME pregnancy clinic from reproductive age."
            }
        ],
        "monitoring": [
            {"item": "NEU1 WES / Lysosomal Gene Panel (NEU1 + CTSA + GLB1)", "frequency": "At diagnosis", "note": "SIMULTANEOUS: NEU1 WES/gene panel + CTSA WES + GLB1 gene panel. CTSA exclusion mandatory (galactosialidosis phenocopy). Panel: NEU1 + CTSA + GLB1 + NPC1 + HEXA + HEXB (GM2) + GLB1 (GM1). Founder variant PCR first if Italian/North African consanguineous (p.Trp249Leu, p.Arg294His)."},
            {"item": "Urine Sialic Acid Oligosaccharides (TLC / Tandem MS)", "frequency": "At diagnosis (rapid screen, 1-5 days)", "note": "FASTEST SIALIDOSIS SCREEN: urine TLC detects abnormal sialyloligosaccharide bands within 1-5 days in any metabolic genetics laboratory. Order simultaneously with leukocyte enzyme assay and WES. Normal urine screen does NOT exclude Sialidosis Type I (sensitivity ~85%); abnormal screen strongly supports Sialidosis."},
            {"item": "Leukocyte α-Neuraminidase Assay + Simultaneous GLB1 Assay", "frequency": "At diagnosis (fresh cells — coordinate same day)", "note": "FRESH LEUKOCYTES REQUIRED (4-hour stability). Pre-arrange with specialist lysosomal biochemistry lab on day of blood draw. Both NEU1 and GLB1 assayed simultaneously: NEU1 only deficient → Sialidosis; both deficient → galactosialidosis (CTSA mutation). Send to specialist lab immediately on ice."},
            {"item": "CTSA Galactosialidosis Exclusion (GLB1 enzyme + CTSA WES)", "frequency": "At diagnosis", "note": "MANDATORY: if leukocyte neuraminidase deficient, measure GLB1 simultaneously. Combined NEU1+GLB1 deficiency = galactosialidosis (CTSA WES next). NEU1-only deficiency confirms Sialidosis pathway. CTSA WES should be included in any sialidosis gene panel. Prevents diagnostic error with different phenotype/counselling."},
            {"item": "Ophthalmology — Fundus Exam (Cherry-Red Spot) + ERG + Visual Acuity", "frequency": "Every 6 months (dilated fundoscopy)", "note": "Cherry-red spot monitoring (90-95% present). ERG: rod and cone function (monitor for visual acuity preservation — Sialidosis preserves VA unlike NCL). Visual acuity: Snellen chart 6-monthly. Peripheral fields: Goldmann or perimetry annually. ERG amplitude reduction warrants VGB caution upgrade. Ophthalmology report in all Sialidosis medical records."},
            {"item": "POLG1 WES + MERRF Mitochondrial DNA Testing (MANDATORY before VPA)", "frequency": "At diagnosis — mandatory before VPA initiation", "note": "POLG1 Alpers and MERRF (m.8344A>G) both cause progressive PME — IDENTICAL phenotype territory to Sialidosis Type I. MERRF: cherry-red spot (rare) + PME → can directly mimic Sialidosis Type I. VPA ABSOLUTE CI in POLG1/MERRF → fatal hepatotoxicity. MANDATORY: blood lactate + m.8344A>G PCR + POLG1 WES + muscle biopsy (RRF in MERRF). Non-negotiable before VPA initiation."},
            {"item": "Brain MRI (3T — Annual)", "frequency": "Annual", "note": "Cerebral and cerebellar atrophy progression (cerebellar ataxia is progressive in Sialidosis); white matter changes; MRS: NAA reduction (neuronal loss quantification); compare to baseline; generally less dramatic changes than NCL — brain atrophy is slower and later; useful for differential diagnosis at presentation."},
            {"item": "EEG (Baseline + Annual + Urgent if Deterioration)", "frequency": "Baseline + annual + urgent", "note": "IPS photosensitivity (72%); jerk-locked back-averaging for cortical myoclonus characterisation (giant SEP confirms cortical origin); annual subclinical NCSE detection; URGENT EEG for any acute confusional state (NCSE vs non-ictal in Sialidosis — preserved cognition makes confusional state more apparent); TGB CI risk."},
            {"item": "SARA Scale (Cerebellar Ataxia)", "frequency": "6-monthly", "note": "Cerebellar ataxia in 85% of Sialidosis Type I; SARA progression guides physiotherapy intensity, walking aid prescription, wheelchair planning (typically much later than NCL — decades not years); SARA combined with UMRS gives composite disability picture; SARA >10 → walking aids; >20 → rollator/wheelchair preparation."},
            {"item": "UMRS (Unified Myoclonus Rating Scale)", "frequency": "6-monthly", "note": "Action myoclonus quantification (98% of patients); piracetam dose titration guide; UMRS tracks functional myoclonus impact on ADLs (writing, eating, walking, working); cortical SEP correlation; UMRS ≥20 triggers piracetam dose increase; UMRS also tracks response to VPA/LEV/perampanel add-on."},
            {"item": "Neuropsychological Assessment (Cognitive Preservation Monitoring)", "frequency": "Annual", "note": "VERIFY COGNITIVE PRESERVATION (Sialidosis Type I hallmark — should NOT show significant decline). WAIS-IV (adults); MOCA; educational/vocational capacity; employment assessment; if cognitive decline detected → re-evaluate diagnosis (exclude Sialidosis Type II, galactosialidosis, NPC, MERRF); cognitive preservation = major quality-of-life outcome measure in Sialidosis."},
            {"item": "VPA TDM + LFT + FBC", "frequency": "6-monthly", "note": "VPA trough 60-100 µg/mL (therapeutic range); LFT (hepatotoxicity monitoring — especially POLG1 exclusion must precede VPA); FBC (thrombocytopenia risk); ammonia if encephalopathic; VPPP compliance (females of reproductive age — mandatory UK/EU); weight (VPA-induced weight gain affects mobility in ataxia context)."},
            {"item": "SUDEP Risk Assessment + Nocturnal Safety", "frequency": "Annual", "note": "Drug-resistant GTCS (58% drug-resistant) + adults living independently (preserved cognition) = SUDEP risk from unsupervised nocturnal GTCS. Nocturnal seizure alarm (SOMO or equivalent). Safe sleeping position. Adult independence planning — overnight alone risk assessment. SUDEP Action counselling for patient AND partner/family. Annual SUDEP risk review at every PME clinic visit."},
            {"item": "BDSRA / LSN / NEU1-Sialidosis Registry + ACP", "frequency": "At diagnosis + annual updates", "note": "Register with lysosomal storage disorder (LSD) national registry; European Sialidosis Registry (rare disease); MPS Society (UK) or NORD (USA) for patient support; ERT research trial eligibility (NEU1 ERT in research phase — BDSRA/LSD registry for future trial recruitment); ACP initiation at diagnosis (long survival but progressive disability)."}
        ],
        "lifecycle_stages": [
            {
                "stage": "Prenatal / Pre-Symptomatic (Genetic Risk — Sibling of Proband)",
                "age_range": "Birth to ~8 years (if biallelic NEU1 identified in family)",
                "description": "Identified through cascade testing of siblings of Sialidosis Type I proband. Genetic counselling for family (25% sibling risk — AR). No symptoms. Annual ophthalmology from age 5 (pre-symptomatic cherry-red monitoring). Urine sialic acid oligosaccharides screening annually from age 5. Leukocyte neuraminidase assay at diagnosis (cascade testing).",
                "priorities": ["Confirm biallelic NEU1 genotype in proband", "Cascade test siblings (25% AR recurrence risk)", "Genetic counselling for parents (carrier status, reproductive options)", "Annual urine sialic acid TLC from age 5", "Annual fundoscopy from age 5 (pre-symptomatic cherry-red monitoring)", "Register in LSD/sialidosis registry", "ERT trial eligibility pre-registration"]
            },
            {
                "stage": "Pre-Symptomatic Cherry-Red Incidental Discovery",
                "age_range": "8-14 years (incidental fundoscopy, no seizures yet)",
                "description": "Cherry-red spot discovered incidentally (ophthalmic screening, other ophthalmology referral) BEFORE first seizure. May have subtle action myoclonus not yet clinically recognised. Diagnostic workup: urine sialic acid + leukocyte neuraminidase + NEU1/CTSA WES. POLG1/MERRF exclusion. Initiation of surveillance. May delay AED initiation if no seizures.",
                "priorities": ["Urine sialic acid oligosaccharides (immediate)", "Leukocyte neuraminidase + GLB1 assay (same-day coordination)", "NEU1 + CTSA WES panel", "POLG1 + MERRF exclusion before any VPA consideration", "EEG baseline (subclinical myoclonus?)", "Ophthalmology baseline ERG", "MRI brain baseline", "Family cascade testing"]
            },
            {
                "stage": "First Seizure — PME Diagnostic Emergency",
                "age_range": "8-25 years (mean 14y) — first GTCS or myoclonus",
                "description": "First GTCS or action myoclonus presents. Frequently misidentified as JME → CBZ → myoclonus worsening. Cherry-red spot MUST BE CHECKED at first seizure presentation in all adolescent/young adult cases with myoclonus. Immediate full workup: urine sialic acid + leukocyte neuraminidase + NEU1/CTSA WES + POLG1/MERRF exclusion.",
                "priorities": ["URGENT FUNDOSCOPY — cherry-red spot at first seizure presentation", "Urine sialic acid TLC (same-day request)", "Leukocyte neuraminidase + GLB1 (same-day specialist lab coordination)", "NEU1 + CTSA WES gene panel (immediate referral)", "POLG1 WES + m.8344A>G MERRF PCR (mandatory before VPA)", "Start VPA + LEV (AVOID CBZ/OXC/PHT/TGB)", "EEG + IPS (photosensitivity?)", "Ophthalmology ERG + VA baseline", "Sialidosis registry enrolment"]
            },
            {
                "stage": "Active PME — Myoclonus-Dominant Phase",
                "age_range": "1-10 years from onset",
                "description": "Progressive action myoclonus + GTCS (drug-resistant 58%). Cerebellar ataxia emerging. NORMAL COGNITION maintained. Employment and independent living continue with adaptation. MDT intensification: piracetam titration, VPA+LEV optimisation, perampanel add-on, physiotherapy for ataxia, OT for myoclonus. Driving assessment (preserved cognition = driving potentially maintained if seizure-controlled).",
                "priorities": ["AED optimisation (VPA+LEV+piracetam+perampanel backbone)", "Piracetam titration to 16-24 g/day (action myoclonus target)", "Physiotherapy for progressive ataxia (gait aids)", "OT adaptive equipment (weighted utensils, myoclonus adaptation)", "Driving assessment (DVLA/DVSA — seizure-freedom criteria)", "Employment support + reasonable adjustments", "Relationship counselling + fertility planning (VPA VPPP mandatory)", "SARA + UMRS 6-monthly", "SUDEP nocturnal safety plan"]
            },
            {
                "stage": "Established Disability — Ataxia-Dependent Phase",
                "age_range": "10-20 years from onset (3rd-4th decade)",
                "description": "Progressive cerebellar ataxia → walking aids → wheelchair. Drug-resistant myoclonus. COGNITION STILL LARGELY PRESERVED (major quality-of-life advantage over NCL). Supported living transition. KD consideration (adult MAD feasible with preserved cognition). Perampanel + VNS consideration for drug-resistant GTCS.",
                "priorities": ["Wheelchair provision + home adaptation", "VNS/RNS consideration (drug-resistant PME GTCS)", "KD-MAD (adult modified Atkins — patient-managed with preserved cognition)", "Speech therapy if dysarthria from cerebellar ataxia", "Supported living transition (preserved cognition = more independence than NCL)", "ACP initiation (advance care planning)", "SUDEP monitoring (nocturnal sensor)", "Carer training in buccal midazolam (patient can also self-administer)", "ERT trial eligibility monitoring (NEU1 ERT in research phase)"]
            },
            {
                "stage": "Late Adult / Palliative Phase (5th-6th Decade)",
                "age_range": "20-40 years from onset",
                "description": "Severe ataxia + refractory myoclonus + GTCS. Progressive disability. Cognitive function relatively preserved into late stages. Quality of life maintained longer than NCL. Comfort care + palliative AED management. ACP active. Survival to 5th-6th decade possible (much longer than NCLs — managed PME without dementia).",
                "priorities": ["Palliative seizure management (comfort-focused)", "SL/IM midazolam rescue SE protocol", "Maintained cognitive engagement (preserved cognition) — reading, communication, relationships", "ACP active (DNACPR, preferred place)", "Brain + tissue donation consent (NEU1/sialidosis ERT research)", "Carer/family ACP counselling", "LSD/sialidosis family support network", "ERT trial legacy — data contribution even in palliative phase"]
            }
        ]
    }


def get_definitions():
    return {
        "disease_name": "Sialidosis Type I — Cherry-Red-Spot-Myoclonus Syndrome / α-Neuraminidase 1 Deficiency",
        "gene_full": "NEU1 (α-Neuraminidase 1 / Sialidase 1) — 6p21.33",
        "omim_gene": "*608272 (NEU1)",
        "omim_disease": "#256550 (Sialidosis Type I — Cherry-Red-Spot-Myoclonus Syndrome)",
        "protein_full": "α-Neuraminidase 1 (NEU1/Sialidase-1); 415 aa; ~46 kDa; lysosomal and plasma-membrane sialidase; 6 Asp-box motifs (SXDXGXTW); catalytic Tyr370 nucleophile; Arg-Arg-Arg catalytic triad; cleaves terminal α-2,3 and α-2,6 sialic acid from glycoconjugates; requires CTSA (cathepsin A/PPCA) for lysosomal activation; forms NEU1-CTSA-GLB1 multienzyme complex",
        "inheritance_mode": "Autosomal recessive (AR) biallelic NEU1 LOF → Sialidosis Type I. 25% sibling recurrence. No AD form. Consanguinity in ~35%. Italian/North-African/Japanese founder variants most prevalent.",
        "onset_age": "Adolescent/young adult: mean 14.2 years (range 8-25 years); PME onset",
        "cherry_red_differentials": [
            {"disease": "GM1 Gangliosidosis", "distinguishing": "Infantile onset (6M-2y); coarse facies; skeletal dysostosis; cognitive impairment prominent; GLB1 enzyme deficiency; NOT a PME of adolescence"},
            {"disease": "GM2 Gangliosidosis (Tay-Sachs/Sandhoff)", "distinguishing": "Infantile onset (6-18M); rapid progressive degeneration; hypotonia; acoustic startle; HEXA (Tay-Sachs) or HEXA+HEXB (Sandhoff) enzyme deficiency; not PME of adolescence"},
            {"disease": "Niemann-Pick Type A", "distinguishing": "Infantile onset (3-6M); hepatosplenomegaly; severe; SMPD1 sphingomyelinase deficiency; not a PME syndrome; fatal in infancy/early childhood"},
            {"disease": "Niemann-Pick Type C", "distinguishing": "Any age (fetal-adult); vertical supranuclear gaze palsy (PATHOGNOMONIC); cataplexy; progressive dementia; NPC1/NPC2 cholesterol trafficking; NOT lysosomal enzyme deficiency; filipin staining; cognitive decline prominent"},
            {"disease": "MERRF (Myoclonic Epilepsy with Ragged-Red Fibres)", "distinguishing": "Mitochondrial m.8344A>G (80%); cherry-red spot RARE (uncommon); PME + cognitive decline + deafness + lactic acidosis; VPA ABSOLUTE CI (mitochondrial hepatotoxicity); muscle biopsy RRF; cognitive decline distinguishes from Sialidosis Type I (normal cognition)"},
            {"disease": "Normal Variant / Pseudo-Cherry-Red Spot", "distinguishing": "No storage material; fundoscopy appearance due to macular pigmentation or thinning; ERG normal; no enzyme deficiency; urine sialic acid normal; no PME; screen with enzyme assays to exclude storage disease"}
        ],
        "multienzyme_complex": "NEU1-CTSA-GLB1 MULTIENZYME COMPLEX: NEU1 (α-neuraminidase 1, 6p21.33) operates in the lysosome as part of a multienzyme complex with CTSA (Cathepsin A/PPCA, 20q13.12) and GLB1 (β-galactosidase, 3p22.3). CTSA is required for NEU1 lysosomal stability and activation — without CTSA, NEU1 remains an inactive apo-enzyme. CTSA also protects GLB1 from premature intralysosomal degradation. DIAGNOSTIC CONSEQUENCE: CTSA biallelic LOF (GALACTOSIALIDOSIS) → BOTH NEU1 AND GLB1 deficient (CTSA required for both). Sialidosis Type I = NEU1 ONLY deficient (GLB1 normal). WHEN NEURAMINIDASE IS DEFICIENT: test GLB1 simultaneously — combined NEU1+GLB1 deficiency = GALACTOSIALIDOSIS (CTSA gene); NEU1-only deficiency = SIALIDOSIS (NEU1 gene). WES panel must include NEU1 + CTSA + GLB1.",
        "concepts": [
            {
                "id": "NEU1-6p21.33-Lysosomal-Alpha-Neuraminidase-PME-Cherry-Red",
                "name": "NEU1 — 6p21.33 Lysosomal α-Neuraminidase / Sialidosis Type I PME + Cherry-Red Spot",
                "definition": "NEU1 (6p21.33) encodes α-neuraminidase 1 (sialidase 1; 415 aa ~46 kDa). NEU1 biallelic LOF → sialyloligosaccharide/sialoglycoprotein accumulation → Sialidosis Type I. PME onset 8-25y (mean 14y): action myoclonus (98%) + GTCS + cerebellar ataxia + NORMAL COGNITION. Cherry-red macular spot: 90-95% (macular ganglion cell storage; visual acuity PRESERVED). AR inheritance. Survival to 5th-6th decade."
            },
            {
                "id": "Cherry-Red-Macular-Spot-PME-Normal-Cognition-Adolescent-PATHOGNOMONIC",
                "name": "Cherry-Red Macular Spot + PME + Normal Cognition + Adolescent Onset = Sialidosis Type I PATHOGNOMONIC",
                "definition": "The combination of cherry-red macular spot + PME + NORMAL intelligence + adolescent/young adult onset (8-25y) is PATHOGNOMONIC for Sialidosis Type I. No other PME syndrome combines all four features. Differential: cherry-red spot without PME = GM1/GM2 gangliosidosis, Niemann-Pick Type A (infantile); PME without cherry-red = NCL, MERRF, Lafora; PME with cognitive decline = NCL/MERRF (not Sialidosis Type I). Fundoscopy is the FASTEST diagnostic step (minutes in any eye department)."
            },
            {
                "id": "Urine-Sialic-Acid-Oligosaccharides-Rapid-Screen-Not-DBS",
                "name": "Urine Sialic Acid Oligosaccharides Rapid Screen — 1-5 Days, Any Metabolic Lab",
                "definition": "Urine TLC or tandem MS for sialyloligosaccharides = FASTEST Sialidosis screen (1-5 days in any metabolic genetics laboratory). Abnormal sialyloligosaccharide bands detected before enzyme assay. Order SIMULTANEOUSLY with NEU1 WES panel and leukocyte neuraminidase assay. Sensitivity ~85% — normal result does NOT exclude Sialidosis. This differs from NCL where DBS enzyme assay (CLN1/CLN2) is the primary diagnostic test. Sialidosis = urine screen FIRST."
            },
            {
                "id": "Leukocyte-Neuraminidase-Assay-Fresh-Cells-Required",
                "name": "Leukocyte α-Neuraminidase Assay — Fresh Cells Required (NOT DBS); 4-Hour Stability",
                "definition": "NEU1 enzyme assay requires FRESH LEUKOCYTES (4-hour stability post-phlebotomy) or cultured fibroblasts — NOT a standard dried blood spot (DBS) test. Logistical requirement: pre-arrange same-day specialist lysosomal biochemistry lab coordination. Cannot be posted. Simultaneously measure GLB1: combined NEU1+GLB1 deficiency = galactosialidosis (CTSA); NEU1-only = sialidosis. This is the MAJOR LOGISTICAL DIFFERENCE between Sialidosis diagnosis (fresh cells) and NCL diagnosis (DBS for CLN1/CLN2)."
            },
            {
                "id": "NEU1-CTSA-GLB1-Multienzyme-Complex-Galactosialidosis-Differential",
                "name": "NEU1-CTSA-GLB1 Multienzyme Complex — Galactosialidosis vs Sialidosis Differential",
                "definition": "NEU1 requires CTSA (cathepsin A/PPCA) for lysosomal activation. CTSA LOF (galactosialidosis) → BOTH NEU1 and GLB1 deficient. Sialidosis Type I = NEU1 ONLY deficient. RULE: when neuraminidase deficient, test GLB1 simultaneously. Combined deficiency → galactosialidosis (CTSA WES). NEU1-only → Sialidosis Type I (NEU1 WES). WES panel MUST include NEU1 + CTSA + GLB1. Prevents misclassification with different phenotype, gene, and family counselling."
            },
            {
                "id": "Normal-Cognition-Type-I-vs-NCL-Key-Distinction",
                "name": "Normal Cognition in Sialidosis Type I — Key Distinction from All NCL Types",
                "definition": "Sialidosis Type I PRESERVES COGNITION (near-normal/normal IQ throughout early-mid course). All NCL types (CLN1-13) cause universal progressive cognitive decline. This is the most important differential feature: adolescent PME + NORMAL SCHOOL PERFORMANCE = Sialidosis Type I first. PME + cognitive decline = NCL, MERRF, Sialidosis Type II, Lafora. Preserved cognition enables employment, driving, relationships, independent living — major quality-of-life distinction. Serial neuropsychology should VERIFY PRESERVATION (not track decline)."
            },
            {
                "id": "CBZ-OXC-PHT-ABSOLUTE-CI-PME-Myoclonus-Trap",
                "name": "CBZ/OXC/PHT ABSOLUTE CI — PME Myoclonus Worsening Trap",
                "definition": "Na-channel blockers (CBZ/OXC/PHT) WORSEN cortical PME myoclonus in Sialidosis Type I. Adolescent GTCS frequently misidentified as JME → CBZ → acute myoclonic deterioration. Diagnosis delay 4.8y = extended Na-channel blocker exposure. ANY adolescent on CBZ/OXC who develops action myoclonus → urgent cherry-red spot fundoscopy → Sialidosis workup. Fosphenytoin ABSOLUTE CI in SE (same Na-channel mechanism)."
            },
            {
                "id": "VPA-SAFE-Lysosomal-NOT-Mitochondrial",
                "name": "VPA SAFE in Sialidosis Type I — Lysosomal Neuraminidase NOT Mitochondrial",
                "definition": "NEU1 = lysosomal enzyme. VPA ABSOLUTE CI applies only to mitochondrial diseases (MERRF/POLG1 Alpers). VPA is SAFE and is the backbone PME AED in Sialidosis Type I. HOWEVER: POLG1/MERRF mandatory exclusion BEFORE VPA initiation (MERRF = most dangerous phenocopy with potential cherry-red + PME + VPA CI). Once MERRF/POLG excluded → VPA first-line."
            },
            {
                "id": "POLG1-MERRF-Mandatory-Exclusion-PME-Cherry-Red",
                "name": "POLG1/MERRF Mandatory Exclusion Before VPA — Cherry-Red PME Phenocopy Danger",
                "definition": "MERRF (m.8344A>G) = most dangerous Sialidosis Type I phenocopy: cherry-red spot (rare) + PME + cognitive decline + VPA ABSOLUTE CI. POLG1 Alpers: PME + liver disease + VPA ABSOLUTE CI. MANDATORY before ANY VPA in cherry-red + PME: blood lactate + m.8344A>G PCR + POLG1 WES + muscle biopsy (RRF for MERRF). VPA-induced hepatic failure in POLG1 = fatal and irreversible. This exclusion is non-negotiable regardless of cherry-red spot finding."
            },
            {
                "id": "VGB-HIGH-RISK-Cherry-Red-Macular-Not-Retinal-NCL",
                "name": "VGB HIGH RISK (Not Absolute CI) in Sialidosis — Cherry-Red Macular Not Retinal NCL",
                "definition": "Sialidosis cherry-red = macular ganglion cell storage (NOT progressive peripheral retinal NCL as in CLN1-CLN11). Visual acuity PRESERVED in Sialidosis Type I. VGB retinopathy affects peripheral fields. Therefore VGB = HIGH RISK (caution; avoid if possible) but NOT ABSOLUTE CI as in NCLs. Contrast: NCL (CLN1/CLN2/CLN3/CLN5/CLN6/CLN7/CLN8/CLN10/CLN11) = VGB ABSOLUTE CI (progressive peripheral+central retinal NCL degeneration). If VGB unavoidable (last resort): ERG baseline + 3-monthly field monitoring mandatory."
            },
            {
                "id": "Piracetam-Level-B-Action-Myoclonus-PME-Backbone",
                "name": "Piracetam Level B — Most Effective for Action Myoclonus in Sialidosis Type I",
                "definition": "Piracetam (16-24 g/day) is Level B evidence for cortical action myoclonus in Sialidosis Type I / cherry-red-spot-myoclonus syndrome. Strongest evidence base among all PME syndromes (Italian/Marseille PME trials including sialidosis/MERRF/EPM1). AMPA receptor modulation + membrane fluidity effects specifically attenuate cortical action myoclonus. Start early before myoclonus disables ADLs. VPA + LEV + piracetam = Sialidosis Type I PME backbone triad."
            },
            {
                "id": "No-Disease-Modifying-Therapy-NEU1-ERT-Research-Phase",
                "name": "No Disease-Modifying Therapy — NEU1 ERT in Research Phase",
                "definition": "No approved disease-modifying therapy for Sialidosis Type I (2026). NEU1 is a soluble lysosomal enzyme — enzyme replacement therapy (ERT) is CONCEPTUALLY FEASIBLE (same principle as CLN2/cerliponase alfa, Gaucher ERT). NEU1 ERT research phase: proof-of-concept studies ongoing; no clinical-stage programme approved. BDSRA/LSD registry enrolment mandatory for future trial eligibility. Unlike CLN2 (approved ERT) and CLN3 (gene therapy Phase 1), Sialidosis Type I has no active clinical therapy pipeline as of 2026."
            },
            {
                "id": "Galactosialidosis-CTSA-Phenocopy-Combined-NEU1-GLB1-Deficiency",
                "name": "Galactosialidosis (CTSA) — NEU1+GLB1 Combined Deficiency Phenocopy",
                "definition": "Galactosialidosis: CTSA (20q13.12) biallelic LOF → both NEU1 and GLB1 (β-galactosidase) deficient (CTSA required for both). Phenotype: late infantile-juvenile onset; coarse facies + macular cherry-red spot + PME + angiokeratoma. Distinguishing test: measure both NEU1 + GLB1 when neuraminidase deficient — combined deficiency = GALACTOSIALIDOSIS (CTSA); NEU1-only = SIALIDOSIS (NEU1). Different gene, different phenotype, different genetic counselling — but both AR, 25% recurrence."
            },
            {
                "id": "Cherry-Red-Differentials-GMI-GMII-NPC-MERRF-Sialidosis",
                "name": "Cherry-Red Spot Differential Diagnosis — GM1/GM2/NPC/MERRF vs Sialidosis",
                "definition": "Cherry-red macular spot differentials: (1) GM1 Gangliosidosis — infantile, coarse facies, GLB1 deficiency; (2) GM2/Tay-Sachs — infantile, HEXA deficiency; (3) Niemann-Pick A — infantile hepatosplenomegaly, SMPD1; (4) Niemann-Pick C — vertical gaze palsy, cataplexy, NPC1/NPC2; (5) MERRF — rare cherry-red, PME+cognitive decline+deafness, m.8344A>G, VPA CI; (6) Sialidosis Type I — PME+NORMAL COGNITION+adolescent, NEU1 deficiency. KEY: only Sialidosis Type I = cherry-red + PME + NORMAL cognition + adolescent onset."
            },
            {
                "id": "SUDEP-Risk-Drug-Resistant-PME-Nocturnal-GTCS",
                "name": "SUDEP Risk — Drug-Resistant PME + Nocturnal GTCS + Adult Independent Living",
                "definition": "Sialidosis Type I SUDEP risk: drug-resistant GTCS (58% drug-resistant) + adult independent living (preserved cognition = unsupervised nocturnal sleep). Unlike NCL (cognitively impaired, supervised), Sialidosis Type I adults live independently → unsupervised nocturnal GTCS = SUDEP risk without immediate carer response. Nocturnal bed sensor MANDATORY. Safe sleeping position (lateral). SUDEP Action counselling for patient AND partner. Annual SUDEP risk review."
            }
        ],
        "thresholds": [
            {"parameter": "Leukocyte α-neuraminidase activity (NEU1)", "value": "<10% of age-matched control activity", "action": "NEU1 deficiency confirmed; simultaneously check GLB1; if GLB1 also deficient → galactosialidosis (CTSA WES); if NEU1 only → Sialidosis pathway (NEU1 WES confirmation)"},
            {"parameter": "GLB1 (β-galactosidase) enzyme assay (simultaneous)", "value": "Normal (within reference range)", "action": "NEU1 deficient + GLB1 normal = Sialidosis Type I (NEU1 gene); if GLB1 also deficient = Galactosialidosis (CTSA gene) — different diagnosis, different management"},
            {"parameter": "Blood lactate (POLG1/MERRF exclusion before VPA)", "value": "<2.0 mmol/L (normal)", "action": "Mitochondrial disease less likely; however, still perform full POLG1 WES + m.8344A>G PCR before VPA — normal lactate does NOT exclude MERRF/POLG in all cases"},
            {"parameter": "VPA trough level", "value": "60-100 µg/mL", "action": "Therapeutic range; <60 → increase dose; >120 → toxicity monitoring (tremor, encephalopathy, ammonia); adjust if LFT abnormal"},
            {"parameter": "UMRS (myoclonus severity)", "value": "≥15/60", "action": "Significant action myoclonus → increase piracetam to 20-24 g/day; review VPA+LEV combination; perampanel add-on; OT adaptive equipment for ADLs; physiotherapy myoclonus gait assessment"},
            {"parameter": "SARA (cerebellar ataxia severity)", "value": "≥10/40", "action": "Significant ataxia → walking stick/walking aids; physiotherapy 2×/week; fall prevention; SARA ≥18 → rollator; SARA ≥25 → wheelchair assessment; compound fall risk (ataxia + myoclonus)"},
            {"parameter": "Piracetam dose (action myoclonus control)", "value": "16-24 g/day (target clinical response)", "action": "UMRS response: if <30% UMRS reduction at 16 g/day → increase to 20 g/day → 24 g/day; above 24 g/day: diminishing returns; add perampanel or LEV escalation if insufficient response"},
            {"parameter": "Ophthalmology ERG amplitude", "value": "Any reduction >20% from baseline", "action": "Progressive macular involvement; intensify ophthalmology monitoring to 3-monthly; visual acuity assessment; low-vision assessment; VGB use contraindicated (any peripheral ERG change = VGB CI upgrade)"},
            {"parameter": "Visual acuity (preserved — Sialidosis Type I goal)", "value": "Snellen ≥6/18 (driving standard)", "action": "Visual acuity MAINTAINED in Sialidosis Type I (unlike NCL). If VA deteriorates below 6/12 → low-vision referral; driving reassessment; if VA falls below 6/60 → severe visual impairment (unexpected in Type I → re-evaluate diagnosis)"},
            {"parameter": "Cognitive assessment (MOCA/WAIS — preserved)", "value": "MOCA ≥26/30 (normal)", "action": "COGNITIVE PRESERVATION IS THE EXPECTED OUTCOME in Sialidosis Type I. If MOCA <24 → re-evaluate diagnosis (consider NPC, galactosialidosis, MERRF, Sialidosis Type II); consider NCSE (EEG urgently); significant cognitive decline = diagnostic red flag in Sialidosis Type I"},
            {"parameter": "Drug resistance criterion", "value": "≥2 AED trials (adequate dose/duration) — seizure frequency unchanged", "action": "Drug-resistant PME: consider perampanel add-on → KD (adult MAD) → VNS assessment → neurology MDT review; 58% drug-resistant at 5-year follow-up in Sialidosis Type I cohort"},
            {"parameter": "SUDEP risk — nocturnal GTCS frequency", "value": "≥1 nocturnal GTCS per month despite ≥2 AED trials", "action": "High SUDEP risk: nocturnal bed sensor mandatory; safe sleeping position; adult independent living review (supervised sleep may be needed); SUDEP Action counselling; consider VNS for drug-resistant nocturnal GTCS; partner/carer training in emergency response"}
        ],
        "standards": [
            "Federico A et al. 1980 Neurol Sci — First Sialidosis Type I (cherry-red-spot-myoclonus syndrome) clinical series; established PME + cherry-red spot + normal cognition triad",
            "Cantz M & Gehler J 1977 Eur J Biochem — First lysosomal sialidase deficiency identification in sialidosis; biochemical foundation",
            "Bonten EJ et al. 1996 Genes Dev — CTSA/PPCA required for NEU1 activation; NEU1-CTSA-GLB1 multienzyme complex characterisation; galactosialidosis/sialidosis molecular distinction",
            "Pshezhetsky AV & Ashmarina M 2001 Nature Genetics perspective — NEU1 molecular biology and pathomechanism review",
            "Mole SE & Anderson G 2019 Lancet Neurology — Rare PME classification update including sialidosis in differential context",
            "ILAE 2022 — Seizure type and epilepsy syndrome classification; PME criteria",
            "NICE NG217 — Epilepsy management guidelines; PME-specific recommendations",
            "MHRA VPPP 2021 — Valproate Pregnancy Prevention Programme (mandatory for Sialidosis Type I females of reproductive age on VPA)",
            "CPIC POLG1 2023 — POLG1 VPA prescribing guidance; mandatory exclusion before VPA in PME",
            "ACMG-AMP 2015 — Variant pathogenicity classification (NEU1/CTSA variant interpretation)",
            "BDSRA Registry 2024 — Batten Disease Support and Research Association (Sialidosis ERT trial eligibility tracking)",
            "WHO-ICF 2019 — International Classification of Functioning Disability and Health (Sialidosis disability framework — preserved cognition enables higher functioning classification)"
        ],
        "references": [
            "Federico A et al. (1980) Cherry red spot myoclonus syndrome and alpha-neuraminidase deficiency: juvenile form of sialidosis. J Neurol Sci 48:157-169 [First Sialidosis Type I clinical series; PME + cherry-red + normal cognition triad]",
            "Cantz M & Gehler J (1977) The neuraminidase-deficient mucopolysaccharide storage disorders. Eur J Biochem 74:453-461 [First lysosomal sialidase deficiency characterisation]",
            "Bonten EJ et al. (1996) Defining the role of the protective protein in the lysosomal multienzyme complex. Genes Dev 10:3163-3175 [NEU1-CTSA-GLB1 multienzyme complex; CTSA required for NEU1 activation; galactosialidosis mechanism]",
            "Pshezhetsky AV et al. (1997) Multienzyme lysosomal complex including sphingolipid-activating proteins and the common molecular chaperone cathepsin A. Nat Med 3:1023-1031 [Multienzyme complex structural basis]",
            "Mole SE & Anderson G (2019) Unreported cases of neuronal ceroid lipofuscinosis. Lancet Neurol 18:1003-4 [PME classification context; rare lysosomal storage PME]",
            "Rubboli G et al. (2017) The neurophysiology of the perisylvian epileptic network. Epilepsia 58 Suppl 1:28-44 [PME electrophysiology; piracetam evidence base in PME syndromes including sialidosis]"
        ]
    }
