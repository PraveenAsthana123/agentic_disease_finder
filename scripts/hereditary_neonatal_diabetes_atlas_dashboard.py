#!/usr/bin/env python3
"""Hereditary-Neonatal-Diabetes-Atlas — Complete 8-Gene Monogenic Neonatal/Infancy-Onset Diabetes Atlas
KCNJ11  (Kir6.2; 390 aa; 11p15.1; AD de novo / inherited;
         ATP-sensitive K-channel pore subunit; GOF → channel stays open → β-cell never depolarises
         → no insulin secretion; PNDM or TNDM; SULFONYLUREA (glibenclamide) CURATIVE — 90% transfer
         from insulin to oral SU even after years on insulin; DEND syndrome (severe: Dev delay+Epilepsy+ND)
         if K-ATP remains open in neurons; seed SEED_BASE+0) ·
ABCC8   (SUR1; 1581 aa; 11p15.1; AD de novo / AR;
         K-ATP channel regulatory subunit; GOF → K-ATP always open → no depolarisation → no insulin;
         TNDM > PNDM; sulfonylurea response good but slightly less reliable than KCNJ11;
         AD GOF same chromosome arm as KCNJ11; AR LOF = congenital hyperinsulinism (OPPOSITE phenotype);
         seed SEED_BASE+1) ·
INS     (preproinsulin; 110 aa; 11p15.5; AD de novo;
         missense → ER stress / misfolded proinsulin aggregates → β-cell apoptosis → permanent ND;
         DIFFERENT from MODY10 (AR LOF INS → severe PNDM from birth; AD missense → ER stress PNDM);
         INSULIN ONLY — sulfonylurea does NOT work (mechanism is β-cell loss, not K-ATP);
         seed SEED_BASE+2) ·
EIF2AK3 (PERK; 1116 aa; 2p11.2; AR;
         eIF2α kinase — Wolcott-Rallison syndrome (WRS); ER stress → unfolded-protein response impaired →
         β-cell apoptosis; multiple epiphyseal dysplasia (skeletal); recurrent hepatic failure; hypothyroidism;
         most common AR neonatal diabetes in consanguineous families (Middle East / North Africa);
         no specific diabetes treatment beyond insulin; bone + liver management paramount; seed SEED_BASE+3) ·
FOXP3   (forkhead box P3; 431 aa; Xp11.23; XLR;
         IPEX syndrome — Immune dysregulation, Polyendocrinopathy, Enteropathy, X-linked;
         Treg master regulator; LOF → uncontrolled T-effector cells → autoimmune type-1-like diabetes
         + severe enteropathy (life-threatening watery diarrhoea) + eczema + additional endocrinopathies;
         born healthy, deteriorate in weeks; HSCT only cure; before HSCT: tacrolimus/sirolimus as bridge;
         neonatal T1-like but NOT classic T1D autoantibodies early; seed SEED_BASE+4) ·
RFX6    (regulatory factor X6; 890 aa; 6q22.31; AR;
         Mitchell-Riley syndrome (MRS); transcription factor required for endocrine pancreas development
         (β, α, δ, ε cell agenesis) and enteroendocrine cell differentiation;
         ND + neonatal hypothyroidism + intestinal atresia (duodenal/jejunal) + gallbladder atresia;
         pancreatic hypoplasia (very small exocrine + absent endocrine);
         PERT mandatory (exocrine insufficiency); seed SEED_BASE+5) ·
GLIS3   (GLI-similar zinc finger 3; 829 aa; 9p24.2; AR;
         multi-organ TF; β-cell development + thyroid folliculogenesis + renal tubulogenesis;
         ND + congenital hypothyroidism + congenital heart disease (CHD) + polycystic kidneys + liver fibrosis;
         neonatal presentation: hypothyroxinaemia + hyperglycaemia simultaneously;
         neonatal screening will show congenital hypothyroidism — ND may be missed initially;
         seed SEED_BASE+6) ·
PDX1    (pancreatic and duodenal homeobox 1; 283 aa; 13q12.2; AR (homozygous/compound het);
         master TF for pancreas development; AR homozygous → pancreatic agenesis (complete absence
         of exocrine + endocrine pancreas); ND from birth + severe exocrine insufficiency + malabsorption;
         PERT mandatory; AD heterozygous = MODY4 (adult-onset, different disease);
         pancreatic agenesis on imaging PATHOGNOMONIC; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2894–2901)
"""
import random

ATLAS_GENES = [
    {
        "gene": "KCNJ11",
        "protein": (
            "KCNJ11 -- 11p15.1 AD-GOF -- 390aa -- Kir6.2-"
            "ATP-Sensitive-Inward-Rectifier-K-Channel-43kDa-"
            "Pore-Forming-Subunit-K-ATP-Channel-"
            "PNDM-TNDM-DEND-Sulfonylurea-Curative-"
            "OMIM-Gene-600937-Disease-PNDM-606176"
        ),
        "locus": "11p15.1",
        "protein_size": (
            "390 aa / 43 kDa (Kir6.2 — ATP-sensitive inward-rectifier K⁺ channel pore subunit; "
            "forms hetero-octamer with SUR1 (ABCC8): (Kir6.2)₄(SUR1)₄; "
            "transmembrane topology: 2 TM segments (TM1+TM2) flanking pore loop; "
            "ATP-binding pocket: N-terminal K185 and other residues bind ATP → channel closure; "
            "GOF MECHANISM: KCNJ11 missense → reduced ATP sensitivity → channel stays OPEN → "
            "  K⁺ efflux sustained → membrane hyperpolarised → no Ca²⁺ influx → "
            "  no depolarisation-triggered exocytosis → β-cell secretes NO insulin; "
            "GOF mutations also expressed in brain neurons → DEND syndrome if severe GOF; "
            "encoded adjacent to ABCC8 on 11p15.1 — both test together; "
            "prevalence: ~1:100,000–300,000 neonates; most de novo"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — GAIN-OF-FUNCTION — KCNJ11: "
            "  Most variants: de novo (no family history); some dominant inherited from mildly affected parent; "
            "  HETEROZYGOUS GOF: sufficient to cause PNDM (one hypersensitive copy drives K-ATP open); "
            "  HOMOZYGOUS or compound het: rare; more severe; "
            "CLINICAL SUBTYPES by GOF severity: "
            "  PNDM (Permanent Neonatal Diabetes Mellitus): "
            "    Onset <6 months; diabetes persists for life if untreated; "
            "    Hb A1c: >10% untreated; severe hyperglycaemia; "
            "  TNDM (Transient Neonatal Diabetes Mellitus): "
            "    Onset <6 months; apparent remission by weeks-months; relapse in adolescence/adult; "
            "  DEND syndrome (Developmental delay + Epilepsy + ND): "
            "    Most severe KCNJ11 GOF variants (e.g. C42R, I296L); "
            "    K-ATP open in neurons → neuron hyperpolarised → action potential failure; "
            "    Seizures, severe DD, muscle weakness; "
            "    Sulfonylurea improves neurology in ~50% if started early enough; "
            "  iDEND (intermediate DEND): DD + muscle hypotonia WITHOUT epilepsy; "
            "FAMILY SCREENING: "
            "  Dominant: test first-degree relatives; parent may have mild undiagnosed T2D-like history"
        ),
        "disease_category": (
            "MONOGENIC NEONATAL DIABETES — K-ATP CHANNEL GOF — PNDM/TNDM/DEND: "
            "  ONSET: < 6 months (by definition neonatal DM); usually 1-12 weeks of life; "
            "  CLINICAL FEATURES: "
            "    Hyperglycaemia + glycosuria; "
            "    Dehydration; failure to thrive; "
            "    Diabetic ketoacidosis (DKA) at presentation in ~30%; "
            "    Insulin therapy started (mistaken for T1D); "
            "  LABS: "
            "    Glucose: >11 mmol/L; "
            "    C-peptide: LOW (insulin not released); "
            "    Islet autoantibodies (GAD/IA-2/ZnT8): NEGATIVE — KEY DDx from T1D autoimmune; "
            "    Hb A1c: markedly elevated; "
            "    pH: may be acidotic; "
            "  NEUROLOGICAL (DEND): "
            "    Developmental delay, muscle hypotonia; epilepsy (severe GOF); "
            "    MRI brain: usually normal; "
            "    EEG: may show epileptiform activity; "
            "GENETIC TESTING: "
            "  Any diabetes onset <6 months → KCNJ11 + ABCC8 first (most common, treatable); "
            "  Chromosomal abnormality (6q24 methylation) excluded first for TNDM; "
            "  If KCNJ11/ABCC8 negative → INS, EIF2AK3, FOXP3, RFX6, GLIS3, PDX1"
        ),
        "disease_pathway": (
            "K-ATP CHANNEL — BETA-CELL INSULIN SECRETION MECHANISM: "
            "NORMAL GLUCOSE-STIMULATED INSULIN SECRETION (GSIS): "
            "  Glucose enters β-cell (GLUT2) → glycolysis → ↑ ATP/ADP ratio → "
            "  K-ATP channel (Kir6.2/SUR1) CLOSES → membrane depolarises → "
            "  VDCC (voltage-dependent Ca²⁺ channel) opens → Ca²⁺ influx → "
            "  insulin granule exocytosis → insulin secretion; "
            "KCNJ11 GOF: "
            "  Kir6.2 missense → reduced ATP binding/sensitivity → K-ATP stays OPEN despite high glucose; "
            "  Membrane remains hyperpolarised → VDCC does NOT open → no Ca²⁺ influx → "
            "  NO insulin secretion despite hyperglycaemia; "
            "SULFONYLUREA (glibenclamide/glyburide) ACTION: "
            "  Sulfonylurea binds SUR1 (ABCC8) directly → K-ATP forced CLOSED → membrane depolarises → "
            "  Ca²⁺ influx → insulin secretion — BYPASSES the ATP sensing defect; "
            "  TREATMENT IS CURATIVE: insulin can be discontinued in ~90% (even after years on insulin); "
            "  Dose: glibenclamide 0.05-0.8 mg/kg/day (much higher than T2D doses); "
            "DEND MECHANISM: "
            "  Kir6.2 expressed in neurons (especially substantia nigra, hippocampus, cerebellum); "
            "  GOF → neurons chronically hyperpolarised → reduced excitability → "
            "  seizures (paradoxical hyperexcitability from circuit imbalance); "
            "  Sulfonylurea crosses BBB → closes neuronal K-ATP → normalises excitability in some"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — KCNJ11-PNDM: "
            "  1. DIABETES ONSET < 6 MONTHS OF AGE: "
            "     Any diabetes in first 6 months → genetic testing MANDATORY; "
            "     T1D autoimmune is extremely rare <6 months; "
            "  2. NEGATIVE ISLET AUTOANTIBODIES (GAD, IA-2, ZnT8): "
            "     KEY DDx from T1D; positive = T1D not KCNJ11; "
            "  3. DETECTABLE C-PEPTIDE (residual) OR LOW C-PEPTIDE: "
            "     β-cells present (not destroyed like T1D) but cannot secrete; "
            "  4. DRAMATIC RESPONSE TO SULFONYLUREA: "
            "     Glibenclamide → insulin discontinued in ~90%; "
            "     'Switchover' can be done even after 30+ years on insulin; "
            "  5. DEND SYNDROME (severe GOF): "
            "     DD + epilepsy + neonatal diabetes TRIAD pathognomonic; "
            "  6. FAMILY HISTORY (may be absent if de novo): "
            "     Parent with 'T2D'-like history at young age — actually unrecognised KCNJ11"
        ),
        "treatment": (
            "TREATMENT — KCNJ11-PNDM: SULFONYLUREA IS CURATIVE: "
            "STEP 1 — SWITCH FROM INSULIN TO GLIBENCLAMIDE: "
            "  Overlap insulin + sulfonylurea for transition; "
            "  Glibenclamide (glyburide): 0.05 mg/kg/day → titrate to 0.8 mg/kg/day; "
            "  Children may need adult-equivalent mg/kg doses; "
            "  Monitor glucose closely during transition (hypoglycaemia risk); "
            "  Most successful transitions: 1-2 weeks; "
            "  Once stable glucose on SU → insulin stopped; "
            "STEP 2 — MAINTENANCE: "
            "  Glibenclamide (preferred) or glipizide; "
            "  Monitor Hb A1c; HbA1c target < 7% (53 mmol/mol); "
            "  Dose adjustments as child grows; "
            "NEUROLOGICAL (DEND): "
            "  Early SU → potential neurological improvement; "
            "  Antiepileptic if needed (standard); "
            "  Educational support; "
            "MONITORING: "
            "  Hb A1c every 3 months; "
            "  Renal + retinal surveillance (same as T1D if poor control); "
            "  Neurodevelopment assessment (DEND); "
            "DO NOT: "
            "  Miss the switchover opportunity — delay worsens neurodevelopmental outcomes (DEND); "
            "  Assume it is T1D without genetic testing if onset <6 months; "
            "  Use metformin alone (mechanism-inappropriate)"
        ),
        "seed": 2894,
    },
    {
        "gene": "ABCC8",
        "protein": (
            "ABCC8 -- 11p15.1 AD-GOF -- 1581aa -- SUR1-"
            "Sulfonylurea-Receptor-1-177kDa-ABC-Transporter-"
            "K-ATP-Channel-Regulatory-Subunit-NBD1-NBD2-"
            "PNDM-TNDM-Sulfonylurea-Response-"
            "OMIM-Gene-600509-Disease-PNDM-606176"
        ),
        "locus": "11p15.1",
        "protein_size": (
            "1581 aa / 177 kDa (SUR1 — sulfonylurea receptor 1; ATP-binding cassette transporter superfamily; "
            "regulatory subunit of K-ATP channel hetero-octamer (SUR1)₄(Kir6.2)₄; "
            "structure: TMD0 + L0 linker + [TMD1+NBD1] + [TMD2+NBD2]; "
            "function: K-ATP channel regulatory subunit — MgADP stimulates channel OPENING (metabolic sensor); "
            "sulfonylurea binding site: transmembrane helices TM14-17; "
            "GOF mechanism: ABCC8 missense → MgADP-gating enhanced → K-ATP stays OPEN → no insulin; "
            "LOF mechanism (AR): SUR1 absent → K-ATP cannot open → persistent insulin secretion → CONGENITAL HI; "
            "OPPOSITE PHENOTYPE: GOF = neonatal diabetes; LOF = hyperinsulinism; "
            "adjacent to KCNJ11 on 11p15.1 — test together (same gene panel)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — GAIN-OF-FUNCTION (GOF): "
            "  De novo or dominantly inherited; "
            "  Heterozygous GOF → PNDM or TNDM; "
            "  TNDM phenotype more common with ABCC8 (vs KCNJ11 more often PNDM); "
            "  Some ABCC8 GOF: onset neonatal then remission (TNDM) → relapse young adult T2D-like; "
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION (LOF): "
            "  Biallelic LOF → SUR1 absent → K-ATP CANNOT open → K⁺ trapped inside → "
            "  membrane never hyperpolarised → VDCC always open → persistent Ca²⁺ → "
            "  excess insulin secretion → CONGENITAL HYPERINSULINISM (CHI) — OPPOSITE DISEASE; "
            "  Diazoxide therapy (K-ATP agonist) → CHI from focal ABCC8 lesion → surgery; "
            "GENETIC TESTING NOTE: "
            "  ABCC8 GOF and ABCC8 LOF require phenotype-based interpretation; "
            "  Glucose: GOF = HIGH; LOF = LOW (hypoglycaemia); "
            "  Same gene, same position, different direction = opposite syndrome"
        ),
        "disease_category": (
            "MONOGENIC NEONATAL DIABETES — K-ATP SUR1 GOF — PNDM/TNDM: "
            "  ONSET: < 6 months; "
            "  PHENOTYPE: similar to KCNJ11 GOF; "
            "  TNDM: apparent remission weeks-to-months → relapse puberty/adulthood; "
            "  PNDM: persistent; "
            "  DKA: ~25-30% at presentation; "
            "  C-PEPTIDE: low but detectable; "
            "  AUTOANTIBODIES: negative; "
            "  SULFONYLUREA RESPONSE: good but slightly less reliable than KCNJ11 "
            "    (~75-80% complete insulin discontinuation vs ~90% for KCNJ11); "
            "  NEUROLOGY: DEND possible but less common than KCNJ11 (SUR1 less expressed in neurons); "
            "  DIFFERENTIAL: "
            "    ABCC8 LOF (CHI): LOW glucose, high insulin — opposite presentation; "
            "    KCNJ11 GOF: clinically identical — genetic test distinguishes"
        ),
        "disease_pathway": (
            "K-ATP SUR1 REGULATION — MgADP GATING: "
            "NORMAL: "
            "  High glucose → glycolysis → ↑ATP, ↓ADP → ATP binds Kir6.2 → K-ATP CLOSES; "
            "  Low glucose → ↑MgADP → MgADP binds SUR1 NBDs → K-ATP opens (protection from hypoglycaemia); "
            "ABCC8 GOF: "
            "  SUR1 missense → NBD conformational change → MgADP binding enhanced or ATP inhibition reduced; "
            "  K-ATP stays OPEN even at high glucose → membrane hyperpolarised → no insulin; "
            "ABCC8 LOF (CHI — opposite): "
            "  SUR1 absent or non-functional → K-ATP cannot respond to MgADP → K-ATP LOCKED CLOSED; "
            "  Membrane always depolarised → Ca²⁺ always flowing in → insulin always secreted → hypoglycaemia; "
            "SULFONYLUREA (ABCC8 GOF): "
            "  Glibenclamide binds SUR1 TM14-17 → K-ATP forced CLOSED despite GOF conformation → "
            "  depolarisation → Ca²⁺ → insulin secretion; "
            "DIAZOXIDE (ABCC8 LOF CHI): "
            "  K-ATP agonist → opens remaining K-ATP channels → hyperpolarises → reduces insulin → "
            "  works for focal CHI NOT diffuse biallelic ABCC8 LOF (need surgery)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — ABCC8-PNDM: "
            "  1. DIABETES ONSET < 6 MONTHS: genetic testing mandatory; "
            "  2. NEGATIVE ISLET AUTOANTIBODIES: ABCC8 GOF not autoimmune; "
            "  3. TNDM PATTERN (especially ABCC8): "
            "     Neonatal DM → remission weeks-months → relapse puberty/adulthood; "
            "     Relapse may look like T2D — ABCC8 genetic test retrospectively clarifies; "
            "  4. SULFONYLUREA RESPONSE (~75-80%): "
            "     Dramatic glucose improvement on glibenclamide; "
            "  5. SAME LOCUS AS KCNJ11 (11p15.1): "
            "     Test both genes on same panel; "
            "  6. OPPOSITE TO CHI: "
            "     CHI sibling or relative of ABCC8 LOF family — understand biallelic vs heterozygous status"
        ),
        "treatment": (
            "TREATMENT — ABCC8-GOF NEONATAL DIABETES: "
            "SULFONYLUREA (GLIBENCLAMIDE): "
            "  Mechanism: closes K-ATP via SUR1 binding → insulin secretion; "
            "  Dose: 0.1-0.8 mg/kg/day (similar to KCNJ11); "
            "  Success rate: ~75-80% complete insulin transfer; "
            "  Partial responders: reduce insulin dose, continue glibenclamide; "
            "  Non-responders: continue insulin; glipizide or repaglinide alternatives; "
            "TNDM MANAGEMENT: "
            "  During remission: monitor Hb A1c; SU prophylactic may delay relapse; "
            "  Relapse: restart SU (often still responsive); "
            "MONITORING: "
            "  Same as KCNJ11: Hb A1c, renal, retinal; "
            "  If TNDM: annual glucose/HbA1c even during remission; "
            "DO NOT: "
            "  Diagnose ABCC8 LOF CHI sibling as T2D if they develop diabetes later — test first; "
            "  Confuse GOF (neonatal DM) with LOF (CHI) clinically — opposite presentations"
        ),
        "seed": 2895,
    },
    {
        "gene": "INS",
        "protein": (
            "INS -- 11p15.5 AD-ER-Stress -- 110aa -- Preproinsulin-"
            "6kDa-Mature-Insulin-ER-Stress-Beta-Cell-Apoptosis-"
            "PNDM-Insulin-ONLY-No-Sulfonylurea-Effect-"
            "OMIM-Gene-176730-Disease-PNDM-606176"
        ),
        "locus": "11p15.5",
        "protein_size": (
            "110 aa (preproinsulin) → 86 aa (proinsulin after signal peptide cleavage) → "
            "51 aa mature insulin (A+B chains after C-peptide removal); "
            "molecular weight: ~6 kDa mature insulin; "
            "PNDM MECHANISM (AD missense): "
            "  Missense → misfolded proinsulin → ER retention → ER stress → UPR → β-cell apoptosis; "
            "  C96Y (Ins^Akita in mice), C43G, H29D, R55C, etc. — disrupt disulphide bonds or folding; "
            "  Consequence: progressive β-cell destruction → permanent, absolute insulin deficiency; "
            "  Mechanism distinct from K-ATP channel defects: β-cell absent (lost), not non-secreting; "
            "AR LOF INS: "
            "  Biallelic LOF → no insulin synthesised → severe PNDM from birth (less common); "
            "  C-peptide: absent or very low; "
            "AD MODY10: "
            "  Heterozygous promoter variants reducing INS expression → mild late-onset MODY (different disease)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — ER STRESS GAIN-OF-TOXIC-FUNCTION: "
            "  Heterozygous missense in insulin gene → misfolded proinsulin; "
            "  One missense allele sufficient (dominant negative or haploinsufficiency of normal insulin; "
            "  misfolded protein stresses ER → activates UPR → β-cell programmed cell death); "
            "  Most: de novo; some dominantly inherited; "
            "  KCNJ11/ABCC8 GOF: β-cells present, not secreting; "
            "  INS AD missense: β-cells progressively lost; "
            "AUTOSOMAL RECESSIVE (AR) — BIALLELIC LOF: "
            "  Both INS alleles non-functional → no insulin made → PNDM; "
            "  Less common; less pronounced ER stress (no misfolded protein to stress ER); "
            "CLINICAL DISCRIMINATION: "
            "  INS AD: onset neonatal; ER stress markers (not routinely tested); "
            "  Functional test: no sulfonylurea response (confirms non-K-ATP mechanism); "
            "  Genetic: INS sequencing"
        ),
        "disease_category": (
            "MONOGENIC NEONATAL DIABETES — INS ER STRESS — PNDM — INSULIN ONLY: "
            "  ONSET: < 6 months; often < 2 weeks of life; "
            "  SEVERITY: complete insulin dependence from birth; "
            "  C-PEPTIDE: very low or absent (β-cells lost or non-functional); "
            "  AUTOANTIBODIES: NEGATIVE (not autoimmune); "
            "  SULFONYLUREA TEST: NO RESPONSE — key discriminator from KCNJ11/ABCC8 GOF; "
            "  DKA at presentation: ~40% (severe absolute insulin deficiency); "
            "  PANCREAS: normal imaging (β-cells lost microscopically; exocrine preserved); "
            "  UNLIKE EIF2AK3/PDX1: no exocrine insufficiency; normal pancreas architecture; "
            "  PROGRESSION: absolute insulin dependence for life; worsens with age; "
            "  COMPLICATION RISK: high (very tight HbA1c targeting needed lifelong)"
        ),
        "disease_pathway": (
            "INS ER STRESS PATHWAY — UPR-MEDIATED BETA-CELL APOPTOSIS: "
            "NORMAL PROINSULIN FOLDING: "
            "  Preproinsulin synthesised → signal peptide cleaved → proinsulin in ER lumen → "
            "  three disulphide bonds (A6-A11, A7-B7, B19-A20) form → correctly folded proinsulin → "
            "  packaged into secretory granules → C-peptide cleaved → mature insulin + C-peptide stored; "
            "INS AD MISSENSE (ER STRESS): "
            "  Missense disrupts disulphide bond or tertiary fold (e.g. C96Y destroys C-peptide-A-chain junction); "
            "  Misfolded proinsulin retained in ER → activates UPR sensors: IRE1, PERK, ATF6; "
            "  Chronic ER stress → CHOP (DDIT3) upregulation → mitochondrial pathway apoptosis; "
            "  β-cells progressively die → irreversible insulin deficiency; "
            "  Misfolded protein also sequesters normal proinsulin → haploinsufficiency contribution; "
            "INS AR LOF: "
            "  No proinsulin made → no ER stress but no insulin either → PNDM from lack of hormone; "
            "K-ATP CHANNEL: "
            "  INTACT in INS PNDM → sulfonylurea closes K-ATP → β-cells depolarise → Ca²⁺ influx → "
            "  no insulin (none to secrete); SU response absent (confirms INS not K-ATP disease)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — INS-PNDM: "
            "  1. NEONATAL DIABETES + NO SULFONYLUREA RESPONSE: "
            "     K-ATP genes excluded (KCNJ11/ABCC8 negative or no SU response); "
            "     INS is next most common; "
            "  2. ABSENT ISLET AUTOANTIBODIES: not autoimmune T1D; "
            "  3. VERY LOW/ABSENT C-PEPTIDE: β-cell loss (contrast KCNJ11 where C-peptide may be detectable); "
            "  4. AD FAMILY HISTORY (if not de novo): parent with young-onset diabetes not responding to SU; "
            "  5. NORMAL PANCREAS ON IMAGING: exocrine preserved (contrast PDX1/pancreatic agenesis); "
            "  6. ABSOLUTE INSULIN REQUIREMENT FROM BIRTH: "
            "     Insulin pump strongly recommended for neonates/infants with PNDM"
        ),
        "treatment": (
            "TREATMENT — INS-PNDM: INSULIN ONLY (LIFELONG): "
            "  INSULIN PUMP (CSII): strongly recommended; "
            "    Continuous subcutaneous insulin infusion; basal + bolus; "
            "    Neonatal/infant pumps with diluted insulin available; "
            "  CONTINUOUS GLUCOSE MONITORING (CGM): "
            "    Mandatory for all neonatal DM — prevents severe hypoglycaemia + monitors hyperglycaemia; "
            "  TARGETS: "
            "    HbA1c < 7% (53 mmol/mol) to prevent microvascular complications; "
            "    Time-in-range > 70%; "
            "  DO NOT TRIAL SULFONYLUREA (no benefit, confirms diagnosis if no response); "
            "  PANCREATIC TRANSPLANT / BETA-CELL REPLACEMENT: "
            "    Research stage; not standard care; "
            "  MONITORING: "
            "    Thyroid function (coexisting autoimmune diseases — NEGATIVE Ab but check anyway); "
            "    Annual renal + retinal surveillance from adolescence; "
            "    Neurodevelopment: normal (INS not expressed in CNS)"
        ),
        "seed": 2896,
    },
    {
        "gene": "EIF2AK3",
        "protein": (
            "EIF2AK3 -- 2p11.2 AR -- 1116aa -- PERK-"
            "EIF2-Alpha-Kinase-3-126kDa-ER-Stress-Sensor-"
            "Wolcott-Rallison-Syndrome-Neonatal-Diabetes-"
            "Multiple-Epiphyseal-Dysplasia-Liver-Failure-"
            "OMIM-Gene-604032-Disease-WRS-226980"
        ),
        "locus": "2p11.2",
        "protein_size": (
            "1116 aa / 126 kDa (PERK — PKR-like endoplasmic reticulum kinase; type I ER transmembrane protein; "
            "luminal domain: ER stress sensor (analogous to IRE1 — dimerises on stress); "
            "cytoplasmic kinase domain: phosphorylates eIF2α-Ser51 → attenuates translation → "
            "prevents further ER protein overload; "
            "PERK is part of UPR (unfolded protein response) — one of three ER stress sensors (IRE1, PERK, ATF6); "
            "PERK is particularly critical for β-cells because proinsulin synthesis is the highest ER load "
            "  of any secretory cell → β-cells uniquely sensitive to PERK absence; "
            "ROLE: PERK LOF → uncontrolled ER stress → UPR fails to protect β-cells → apoptosis; "
            "Also critical in: chondrocytes (epiphyseal dysplasia), hepatocytes (liver failure), osteoblasts"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION: "
            "  Biallelic EIF2AK3 pathogenic variants → PERK absent/non-functional; "
            "  High frequency in consanguineous families (Middle East, North Africa, Turkey); "
            "  Most common AR cause of neonatal diabetes overall (especially in those populations); "
            "  Heterozygous carriers: unaffected (no known heterozygous phenotype); "
            "CONSANGUINITY: "
            "  Homozygous founder variants common (e.g. p.Asp543His); "
            "  Always consider in consanguineous pedigrees with neonatal DM; "
            "PRENATAL DIAGNOSIS: possible with family history"
        ),
        "disease_category": (
            "WOLCOTT-RALLISON SYNDROME (WRS) — MULTI-ORGAN ER STRESS DISEASE: "
            "  TRIAD: Neonatal Diabetes + Multiple Epiphyseal Dysplasia + Recurrent Liver Failure; "
            "  DIABETES: "
            "    Onset < 6 months (usually 1-6 months); "
            "    Permanent; absolute insulin dependence; "
            "    C-peptide: absent; autoantibodies: negative; "
            "  SKELETAL (EPIPHYSEAL DYSPLASIA): "
            "    Multiple epiphyseal dysplasia — short stature, joint deformity, waddling gait; "
            "    Evident radiologically by 1-3 years; "
            "    PATHOGNOMONIC for WRS in context of neonatal DM; "
            "  LIVER (ACUTE HEPATIC CRISES): "
            "    Recurrent episodes of acute hepatic failure (potentially lethal); "
            "    Triggers: intercurrent illness, fever; "
            "    Elevated AST/ALT/bilirubin; PT prolonged; "
            "    Can progress to cirrhosis; "
            "  OTHER FEATURES: "
            "    Hypothyroidism (30-50%); "
            "    Intellectual disability (mild to moderate); "
            "    Recurrent infections; "
            "    Exocrine pancreatic insufficiency (some patients); "
            "  PROGNOSIS: "
            "    Poor if hepatic crises unmanaged; "
            "    Some survive to adulthood with close monitoring"
        ),
        "disease_pathway": (
            "PERK — UPR — BETA-CELL AND MULTI-ORGAN ER STRESS: "
            "NORMAL PERK FUNCTION: "
            "  ER stress detected (misfolded proteins accumulate) → PERK dimerises + trans-autophosphorylates → "
            "  activated PERK kinase phosphorylates eIF2α-Ser51 → "
            "  global translation attenuation (reduces new protein load on ER) + "
            "  selective translation of stress-response mRNAs (ATF4 → CHOP/GADD34); "
            "  Net effect: reduces ER burden → β-cell survival; "
            "WRS (PERK LOF): "
            "  ER stress NOT buffered → proinsulin load overwhelms ER → UPR activation without PERK rescue; "
            "  β-cell apoptosis (identical mechanism to INS AD missense but PERK is intrinsic sensor); "
            "SKELETAL: "
            "  Chondrocytes require PERK for cartilage matrix synthesis; "
            "  PERK LOF → growth plate chondrocyte ER stress → apoptosis → epiphyseal dysplasia; "
            "LIVER: "
            "  Hepatocytes require PERK for VLDL secretory pathway; "
            "  Fever/illness → ER stress surge → hepatocyte death → acute liver failure; "
            "THYROID: "
            "  Thyroglobulin is a major secretory protein; PERK LOF → thyrocyte stress → hypothyroidism"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — EIF2AK3/WRS: "
            "  1. NEONATAL DIABETES + MULTIPLE EPIPHYSEAL DYSPLASIA: "
            "     TRIAD is pathognomonic for WRS; "
            "  2. ACUTE HEPATIC CRISES (recurrent): "
            "     Any febrile illness → acute liver failure episode; "
            "  3. CONSANGUINITY + AR INHERITANCE: "
            "     Common in Middle Eastern / North African families; "
            "  4. RADIOLOGICAL: "
            "     Multiple epiphyseal dysplasia on X-ray; "
            "  5. HYPOTHYROIDISM (30-50%): "
            "     TSH elevated; T4 low; coexists with ND; "
            "  6. ABSENT ISLET AUTOANTIBODIES: not T1D autoimmune"
        ),
        "treatment": (
            "TREATMENT — WRS/EIF2AK3: SUPPORTIVE MULTI-ORGAN: "
            "DIABETES: "
            "  Insulin pump + CGM (same as INS PNDM — absolute insulin dependence); "
            "  No sulfonylurea (not K-ATP); "
            "HEPATIC CRISES: "
            "  URGENT on fever: "
            "    IV glucose (prevent hypoglycaemia from liver failure); "
            "    N-acetylcysteine (antioxidant); "
            "    Avoid hepatotoxic drugs; "
            "  Chronic: regular LFTs, LFT-based triggers for admission; "
            "  Liver transplant: considered if repeated severe failure; "
            "SKELETAL: "
            "  Physiotherapy, orthotics; "
            "  Orthopaedic consultation for joint deformity; "
            "  No disease-modifying treatment available; "
            "THYROID: "
            "  Levothyroxine if hypothyroid (common); "
            "MONITORING: "
            "  LFTs every 3 months; "
            "  TSH every 6 months; "
            "  Annual skeletal survey; "
            "  Neurodevelopment support; "
            "PROGNOSIS: "
            "  Variable; hepatic crises are main cause of early death; "
            "  Genetic counselling and carrier testing for family"
        ),
        "seed": 2897,
    },
    {
        "gene": "FOXP3",
        "protein": (
            "FOXP3 -- Xp11.23 XLR -- 431aa -- Forkhead-Box-P3-"
            "47kDa-Treg-Master-Regulator-Transcription-Factor-"
            "IPEX-Syndrome-Immune-Dysregulation-Polyendocrinopathy-"
            "Enteropathy-X-Linked-HSCT-Only-Cure-"
            "OMIM-Gene-300292-Disease-IPEX-304790"
        ),
        "locus": "Xp11.23",
        "protein_size": (
            "431 aa / 47 kDa (FOXP3 — forkhead box P3; transcription factor; "
            "domains: N-terminal repressor domain + zinc finger + leucine zipper + C-terminal forkhead domain; "
            "CRITICAL FUNCTION: master regulator of regulatory T-cell (Treg) development and function; "
            "expressed in CD4+CD25+ Treg cells; "
            "FOXP3 activates Treg gene programme → Tregs suppress autoreactive T-effector cells; "
            "LOF: Tregs absent/dysfunctional → uncontrolled T-effector cells target self-antigens; "
            "multi-organ autoimmune destruction: pancreatic β-cells (T1D-like), gut epithelium (enteropathy), "
            "  thyroid, adrenal, skin (eczema), blood cells (haemolytic anaemia, thrombocytopenia); "
            "X-LINKED: males affected (hemizygous); females carriers (usually healthy unless skewed X-inactivation)"
        ),
        "inheritance": (
            "X-LINKED RECESSIVE (XLR) — LOSS-OF-FUNCTION: "
            "  Males affected (hemizygous); "
            "  Carrier females: usually unaffected; rare symptomatic female carriers (skewed X-inactivation); "
            "  New mutations: ~20-30% de novo; "
            "  SPECTRUM: "
            "    Classic IPEX: severe onset first days-weeks of life; lethal without HSCT; "
            "    Partial FOXP3 function: later onset, milder; may survive childhood; "
            "    C-terminal forkhead domain mutations: generally more severe; "
            "    N-terminal domain: milder phenotypes described; "
            "FAMILY HISTORY: "
            "  Maternal uncles may have died in infancy from unknown cause; "
            "  Carrier testing of maternal female relatives"
        ),
        "disease_category": (
            "IPEX SYNDROME — IMMUNE DYSREGULATION + NEONATAL T1D-LIKE DIABETES: "
            "  CLASSIC TRIAD: Immune dysregulation + Polyendocrinopathy + Enteropathy + X-linked; "
            "  ENTEROPATHY (life-threatening): "
            "    Profuse watery diarrhoea from first weeks; "
            "    Villous atrophy; "
            "    Malabsorption + failure to thrive; "
            "    Serum IgE very high; "
            "  DIABETES (T1D-like): "
            "    Onset neonatal; T-cell–mediated β-cell destruction; "
            "    Positive islet autoantibodies (GAD, IA-2, ZnT8) — PRESENT (unlike KCNJ11/INS — KEY DDx); "
            "    Absolute insulin dependence; "
            "  ECZEMA: severe atopic dermatitis; "
            "  OTHER ENDOCRINOPATHY: "
            "    Thyroiditis (hypothyroidism); "
            "    Adrenal insufficiency; "
            "  HAEMATOLOGICAL: "
            "    Haemolytic anaemia (Coombs +); "
            "    Thrombocytopenia; "
            "  INFECTIONS: susceptibility from immune dysregulation; "
            "  WITHOUT HSCT: fatal usually < 2 years"
        ),
        "disease_pathway": (
            "FOXP3 — TREG DEVELOPMENT — SELF-TOLERANCE BREAKDOWN: "
            "NORMAL: "
            "  Thymic Treg development: TCR stimulation + TGFβ + IL-2 → FOXP3 upregulated → "
            "  CD4+CD25+FOXP3+ Treg → suppresses self-reactive T-effectors via: "
            "    IL-10, TGFβ secretion; CTLA-4 competition for CD28 co-stimulation; "
            "    IL-2 consumption (starves T-effectors); "
            "    Contact-dependent suppression; "
            "  Result: self-reactive T-cells deleted or suppressed → no autoimmunity; "
            "IPEX (FOXP3 LOF): "
            "  Tregs absent or non-functional → T-effector cells attack self-antigens; "
            "  Pancreas: β-cell antigens targeted (GAD, IA-2, ZnT8 expressed as autoantibodies); "
            "  Gut: enterocytes targeted → villous atrophy → severe diarrhoea; "
            "  Thyroid: thyrocytes → autoimmune thyroiditis; "
            "  Skin: eczema; "
            "  Blood: RBC + platelet antigens → haemolytic anaemia + thrombocytopenia; "
            "TREATMENT RATIONALE: "
            "  Immunosuppression: tacrolimus/sirolimus → suppress T-effectors; "
            "  HSCT: reconstitutes FOXP3+ Treg pool from normal donor; "
            "    Diabetes usually NOT reversed post-HSCT (β-cells already destroyed); "
            "    Enteropathy + eczema respond best to HSCT"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — FOXP3/IPEX: "
            "  1. NEONATAL DIABETES + SEVERE DIARRHOEA (enteropathy): "
            "     Combination in neonatal male is PATHOGNOMONIC for IPEX; "
            "  2. POSITIVE ISLET AUTOANTIBODIES (GAD, IA-2, ZnT8): "
            "     KEY DDx from all other monogenic neonatal DM (all autoantibody-NEGATIVE); "
            "     IPEX is the ONLY monogenic neonatal DM with positive autoantibodies; "
            "  3. ECZEMA + HIGH IgE: "
            "     Atopic-like presentation; "
            "  4. MALE ONLY (XLR): "
            "     Maternal uncles with neonatal death — suspect IPEX; "
            "  5. MULTIPLE AUTOIMMUNE ENDOCRINOPATHIES: "
            "     Thyroiditis + adrenal involvement alongside diabetes; "
            "  6. LYMPHOCYTE STUDIES: "
            "     CD4+CD25+FOXP3+ Tregs absent on flow cytometry (diagnostic)"
        ),
        "treatment": (
            "TREATMENT — IPEX/FOXP3: HSCT IS ONLY CURE: "
            "BRIDGING BEFORE HSCT: "
            "  TACROLIMUS (calcineurin inhibitor): "
            "    Suppresses T-effector activation; partial control; "
            "    Dose: 0.05-0.2 mg/kg/day; monitor levels + nephrotoxicity; "
            "  SIROLIMUS (mTOR inhibitor): "
            "    Alternatively or in combination; spares Treg function better than cyclosporin; "
            "  ABATACEPT (CTLA-4-Ig): experimental; "
            "  STEROIDS: partial effect on enteropathy; "
            "  NUTRITIONAL SUPPORT: parenteral nutrition for severe enteropathy; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANT (HSCT): "
            "  Only potentially curative treatment; "
            "  Reconstitutes FOXP3+ Treg pool from donor; "
            "  Best results: early transplant (before multi-organ damage); "
            "  Conditioning: reduced-intensity preferred; "
            "  Outcomes: enteropathy + eczema resolve; diabetes usually persists (β-cells gone); "
            "DIABETES MANAGEMENT: "
            "  Insulin pump + CGM (absolute insulin dependence); "
            "  Diabetes persists even after HSCT; "
            "SCREENING: "
            "  Thyroid function, adrenal function, CBP every 3 months"
        ),
        "seed": 2898,
    },
    {
        "gene": "RFX6",
        "protein": (
            "RFX6 -- 6q22.31 AR -- 890aa -- Regulatory-Factor-X-6-"
            "99kDa-Winged-Helix-Transcription-Factor-"
            "Mitchell-Riley-Syndrome-MRS-Neonatal-Diabetes-"
            "Hypothyroidism-Intestinal-Atresia-Gallbladder-Agenesis-"
            "OMIM-Gene-612409-Disease-MRS-615710"
        ),
        "locus": "6q22.31",
        "protein_size": (
            "890 aa / 99 kDa (RFX6 — regulatory factor X-box binding protein 6; "
            "transcription factor family: RFX family; "
            "DNA-binding domain: winged-helix domain; dimerisation domain; "
            "CRITICAL DEVELOPMENTAL FUNCTIONS: "
            "  Pancreatic endocrine cell differentiation: "
            "    RFX6 required for β, α, δ, ε (ghrelin) cell specification from pancreatic progenitors; "
            "    LOF → all endocrine cells absent → 'endocrine agenesis' (exocrine preserved or hypoplastic); "
            "  Enteroendocrine cell differentiation: intestinal enteroendocrine cells absent; "
            "  Thyroid development: RFX6 expressed in thyroid → hypothyroidism; "
            "  Gallbladder: RFX6 expressed → gallbladder agenesis/atresia"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION: "
            "  Biallelic pathogenic variants → RFX6 absent; "
            "  Rare: < 30 cases reported worldwide; "
            "  Consanguineous families; "
            "  No heterozygous phenotype reported; "
            "  PRENATAL: polyhydramnios (from intestinal atresia); abnormal glucose in foetal blood"
        ),
        "disease_category": (
            "MITCHELL-RILEY SYNDROME (MRS) — MULTI-ORGAN DEVELOPMENTAL DEFECT: "
            "  NEONATAL DIABETES: "
            "    Onset first days to weeks; "
            "    Complete absence of endocrine pancreas → absolute insulin deficiency; "
            "    No insulin secretion; autoantibodies: negative; "
            "  CONGENITAL HYPOTHYROIDISM: "
            "    TSH elevated at birth; T4 low; "
            "    Detected on neonatal screen; "
            "  INTESTINAL ATRESIA: "
            "    Duodenal/jejunal/ileal atresia; "
            "    Bilious vomiting from birth; "
            "    Surgical correction required urgently; "
            "  GALLBLADDER ATRESIA/AGENESIS: "
            "    Absent or rudimentary gallbladder on imaging; "
            "  PANCREATIC HYPOPLASIA: "
            "    Pancreas very small on imaging; "
            "    PERT mandatory (exocrine insufficiency); "
            "  PROGNOSIS: "
            "    Depends on intestinal atresia repair success; "
            "    Chronic insulin + exocrine support long-term"
        ),
        "disease_pathway": (
            "RFX6 TRANSCRIPTIONAL REGULATION OF PANCREAS AND GUT DEVELOPMENT: "
            "PANCREATIC ENDOCRINE DIFFERENTIATION: "
            "  Pancreatic progenitors (Ptf1a+ / Pdx1+) → common progenitor → "
            "  Ngn3+ endocrine progenitor → RFX6 activates transcriptional programme → "
            "  β-cells (Nkx6.1+/Pdx1+/MafA+/Ins+), α-cells (Arx+/Gcg+), δ, ε cells; "
            "RFX6 LOF: Ngn3+ progenitors form but cannot differentiate → endocrine agenesis; "
            "  Insulin absent; glucagon absent; somatostatin absent; "
            "ENTEROENDOCRINE CELLS: "
            "  RFX6 required for EEC lineage in gut; "
            "  LOF → absent EECs → impaired GLP-1, GIP, CCK, secretin secretion; "
            "  Contributes to malabsorption + gut dysmotility; "
            "INTESTINAL ATRESIA: "
            "  Mechanism: RFX6 role in intestinal mesenchymal/endodermal patterning; "
            "  Atresia at varying levels; "
            "THYROID: "
            "  RFX6 expressed in thyroid follicular cells → hypothyroidism"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — RFX6/MRS: "
            "  1. NEONATAL DIABETES + INTESTINAL ATRESIA: "
            "     COMBINATION is pathognomonic for Mitchell-Riley syndrome; "
            "  2. CONGENITAL HYPOTHYROIDISM on neonatal screen: "
            "     Three simultaneous neonatal problems: DM + CH + gut atresia; "
            "  3. GALLBLADDER ABSENT on imaging: "
            "     Agenesis/atresia; "
            "  4. PANCREATIC HYPOPLASIA: "
            "     Very small pancreas on ultrasound/MRI; "
            "  5. NEGATIVE ISLET AUTOANTIBODIES: not autoimmune; "
            "  6. EXOCRINE INSUFFICIENCY: "
            "     Low faecal elastase; malabsorption"
        ),
        "treatment": (
            "TREATMENT — MRS/RFX6: MULTI-SYSTEM SUPPORT: "
            "INTESTINAL ATRESIA: "
            "  Urgent surgical correction (duodeno-jejunoplasty); "
            "  Parenteral nutrition until gut functional; "
            "  Long-term: short bowel syndrome management if extensive resection; "
            "DIABETES: "
            "  Insulin pump + CGM (absolute insulin dependence); "
            "  Lifelong insulin (no endogenous insulin production); "
            "HYPOTHYROIDISM: "
            "  Levothyroxine replacement from diagnosis; "
            "  TSH target: normal for age; "
            "EXOCRINE INSUFFICIENCY: "
            "  Pancreatic enzyme replacement therapy (PERT): creon with each meal; "
            "  Fat-soluble vitamin supplementation (ADEK); "
            "  Monitor growth and nutritional status; "
            "MONITORING: "
            "  TSH every 6 months; "
            "  Faecal elastase + fat-soluble vitamins annually; "
            "  HbA1c every 3 months; "
            "  Nutritional review"
        ),
        "seed": 2899,
    },
    {
        "gene": "GLIS3",
        "protein": (
            "GLIS3 -- 9p24.2 AR -- 829aa -- GLI-Similar-Zinc-Finger-3-"
            "95kDa-Transcription-Factor-Zinc-Finger-"
            "Neonatal-Diabetes-Congenital-Hypothyroidism-"
            "Congenital-Heart-Disease-Renal-Cysts-Liver-Fibrosis-"
            "OMIM-Gene-610192-Disease-NDH-610199"
        ),
        "locus": "9p24.2",
        "protein_size": (
            "829 aa / 95 kDa (GLIS3 — GLI-similar zinc finger 3; "
            "C2H2 zinc finger transcription factor (6 zinc fingers); "
            "CRITICAL ROLES: "
            "  Pancreas: β-cell development + survival; GLIS3 activates INS gene transcription directly; "
            "    LOF → impaired β-cell differentiation + impaired insulin gene expression; "
            "  Thyroid: follicular cell development; GLIS3 activates thyroid-specific gene programme; "
            "    LOF → congenital hypothyroidism; "
            "  Kidney: ureteric bud branching + collecting duct differentiation; "
            "    LOF → polycystic kidney disease; "
            "  Liver: biliary epithelial development; "
            "    LOF → hepatic fibrosis + biliary tree anomalies; "
            "  Heart: cardiac development (mechanism not fully defined); "
            "    LOF → congenital heart disease (ASD, VSD)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION: "
            "  Biallelic pathogenic variants → GLIS3 absent; "
            "  Rare: ~25-30 cases reported; "
            "  Consanguineous families common; "
            "  Founder variants in some communities (Pakistani, Saudi, French Canadian); "
            "  No haploinsufficiency phenotype in humans; "
            "PRENATAL FEATURES: "
            "  Hydrops/macrosomia: rare; "
            "  Neonatal screen: congenital hypothyroidism detected at birth → "
            "    triggers investigation → ND + other features identified"
        ),
        "disease_category": (
            "GLIS3 SYNDROME — NEONATAL DIABETES + CONGENITAL HYPOTHYROIDISM + CHD + RENAL CYSTS: "
            "  NEONATAL DIABETES: "
            "    Onset first days to weeks; permanent; absolute insulin deficiency; "
            "    Autoantibodies: negative; C-peptide: absent; "
            "  CONGENITAL HYPOTHYROIDISM: "
            "    TSH very high; T4 very low; thyroid hypoplasia/dysplasia; "
            "    Neonatal screen positive (CH detected); "
            "    Levothyroxine from day 1 of life; "
            "  CONGENITAL HEART DISEASE: "
            "    ASD, VSD, coarctation, other structural heart defects; "
            "    Echocardiogram at birth mandatory; "
            "  POLYCYSTIC KIDNEYS: "
            "    Renal cysts → progressive renal dysfunction; "
            "    CKD in adolescence-early adulthood; "
            "    Renal transplant may eventually be required; "
            "  LIVER FIBROSIS: "
            "    Progressive; portal hypertension in older patients; "
            "  ADDITIONAL: "
            "    Intellectual disability (mild-moderate); "
            "    Sensorineural hearing loss (some patients); "
            "  PROGNOSIS: "
            "    Depends on CHD severity and renal progression"
        ),
        "disease_pathway": (
            "GLIS3 TRANSCRIPTIONAL REGULATION — MULTI-ORGAN DEVELOPMENTAL PROGRAMME: "
            "PANCREAS: "
            "  GLIS3 activated in β-cell precursors → binds INS gene promoter → activates insulin expression; "
            "  Also activates Nkx6.1, MafA (β-cell identity genes); "
            "  LOF → β-cells form but cannot express insulin + survival impaired → diabetes; "
            "THYROID: "
            "  GLIS3 binds TTF1/NKX2-1 and PAX8 target sites in thyroid → follicular cell maturation; "
            "  LOF → thyroid dysgenesis → congenital hypothyroidism; "
            "KIDNEY: "
            "  GLIS3 expressed in collecting duct cells → regulates cilia length + tubulogenesis; "
            "  LOF → abnormal cilia → polycystic kidney (similar to other ciliopathy genes); "
            "HEART: "
            "  Mechanism: GLIS3 in second heart field / cardiac neural crest; "
            "  LOF → structural heart defects; "
            "LIVER: "
            "  Biliary cholangiocytes express GLIS3; "
            "  LOF → bile duct development abnormal → fibrosis"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — GLIS3: "
            "  1. NEONATAL DIABETES + CONGENITAL HYPOTHYROIDISM (SIMULTANEOUS): "
            "     Two neonatal endocrine defects together → GLIS3 (or RFX6) most likely; "
            "  2. RENAL CYSTS (polycystic kidneys): "
            "     Ultrasonographic cysts in neonatal period with ND + CH; "
            "  3. CONGENITAL HEART DISEASE: "
            "     Echocardiographic finding + ND + CH + renal cysts = GLIS3 until proven otherwise; "
            "  4. LIVER FIBROSIS: "
            "     Elevated GGT + fibrosis markers in older patients; "
            "  5. NEGATIVE ISLET AUTOANTIBODIES: not autoimmune; "
            "  6. AR INHERITANCE: consanguinity often present"
        ),
        "treatment": (
            "TREATMENT — GLIS3 SYNDROME: MULTI-ORGAN LIFELONG MANAGEMENT: "
            "DIABETES: "
            "  Insulin pump + CGM (absolute insulin dependence); "
            "  No sulfonylurea (not K-ATP); "
            "HYPOTHYROIDISM: "
            "  Levothyroxine from day 1 (critical for neurodevelopment); "
            "  TSH target: normal for age; "
            "  Higher doses needed in infancy (fast metabolism); "
            "CONGENITAL HEART DISEASE: "
            "  Cardiology referral at birth; echocardiogram; "
            "  Surgical correction if indicated; "
            "RENAL: "
            "  Annual renal ultrasound + eGFR + urine PCR; "
            "  ACE inhibitor if proteinuria; "
            "  Renal transplant planning if CKD stage 4-5; "
            "  Avoid nephrotoxic drugs; "
            "LIVER: "
            "  Annual liver function + ultrasound; "
            "  Monitor for portal hypertension (Doppler); "
            "  Hepatology referral; "
            "HEARING: "
            "  Annual audiogram if SNHL; hearing aids/implant; "
            "NEURODEVELOPMENT: "
            "  Educational support"
        ),
        "seed": 2900,
    },
    {
        "gene": "PDX1",
        "protein": (
            "PDX1 -- 13q12.2 AR-Homozygous -- 283aa -- Pancreatic-"
            "Duodenal-Homeobox-1-32kDa-Master-Pancreas-TF-"
            "Homeobox-Protein-IDX-1-IPF-1-STF-1-"
            "Pancreatic-Agenesis-Complete-Exocrine-Endocrine-Failure-"
            "OMIM-Gene-600733-Disease-PACRM-260370"
        ),
        "locus": "13q12.2",
        "protein_size": (
            "283 aa / 32 kDa (PDX1 — pancreatic and duodenal homeobox 1; homeodomain TF; "
            "also known as IDX-1, IPF-1, STF-1, IUF-1; "
            "CRITICAL ROLE: master transcription factor for ENTIRE pancreas development; "
            "  Stage 1 (early foetal): PDX1 marks all pancreatic progenitors (pancreatic bud); "
            "  Stage 2 (later): PDX1 expression maintained in β-cells → activates INS gene + MafA + Nkx6.1; "
            "AR HOMOZYGOUS LOF: both pancreatic exocrine AND endocrine tissue absent → "
            "  pancreatic agenesis (complete); "
            "  No ductal, acinar, islet cells; pancreas replaced by fatty/fibrous tissue; "
            "AD HETEROZYGOUS (MODY4): one functional copy → adult-onset mild diabetes; "
            "  COMPLETELY DIFFERENT DISEASE from AR homozygous; "
            "  Haploinsufficiency reduces β-cell mass → MODY; pancreatic anatomy normal"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION (pancreatic agenesis): "
            "  Homozygous or compound heterozygous PDX1 null/severe LOF variants; "
            "  Rare: ~10 known cases; "
            "  Consanguineous families; "
            "  Parents (heterozygous): MODY4-like mild diabetes if penetrance high; "
            "AUTOSOMAL DOMINANT (AD) — HAPLOINSUFFICIENCY (MODY4): "
            "  Heterozygous PDX1 → reduced β-cell mass → adult-onset diabetes (MODY); "
            "  CLINICALLY DISTINCT from AR homozygous (normal pancreas anatomy; no exocrine failure); "
            "COMPOUND HETEROZYGOUS: "
            "  One severe LOF + one milder → intermediate phenotype"
        ),
        "disease_category": (
            "PANCREATIC AGENESIS — COMPLETE EXOCRINE + ENDOCRINE PANCREATIC FAILURE: "
            "  NEONATAL DIABETES: "
            "    Onset first days of life; complete absence of insulin secretory capacity; "
            "    Absolute insulin dependence; C-peptide: absent; autoantibodies: negative; "
            "  EXOCRINE PANCREATIC INSUFFICIENCY (COMPLETE): "
            "    No acinar cells → no pancreatic lipase/amylase/protease; "
            "    Malabsorption from birth: steatorrhoea, fat-soluble vitamin deficiency; "
            "    PERT mandatory (high dose); "
            "  PANCREAS ABSENT on imaging: "
            "    PATHOGNOMONIC: absent or severely hypoplastic pancreas on ultrasound/MRI; "
            "    Fatty replacement; no identifiable body/tail/head; "
            "  GROWTH: "
            "    Failure to thrive (both diabetes + malabsorption); "
            "  PROGNOSIS: "
            "    Chronic but manageable; limited by metabolic control + nutritional status"
        ),
        "disease_pathway": (
            "PDX1 — PANCREATIC ORGANOGENESIS — MASTER DEVELOPMENTAL REGULATOR: "
            "NORMAL PANCREAS DEVELOPMENT: "
            "  E8.5 (mouse) / 26-28 days (human): foregut endoderm → PDX1+ pancreatic bud; "
            "  PDX1 binds target genes: Ptf1a, Mist1, Sox9 → pancreatic identity; "
            "  Proliferation/branching → ductal tree + acinar clusters; "
            "  Ngn3+ endocrine progenitors → RFX6 + GLIS3 → β, α, δ cells; "
            "  Later: PDX1 expression maintained in β-cells → directly activates INS gene; "
            "AR PDX1 AGENESIS: "
            "  Pancreatic bud fails to develop → no pancreatic progenitors → no pancreas; "
            "  Complete absence of exocrine + endocrine tissue; "
            "  Duodenum develops (PDX1 also expressed briefly in duodenal tip → name); "
            "AD PDX1 (MODY4): "
            "  Pancreas forms but β-cell mass reduced ~50%; "
            "  Insulin production adequate until stress/age → adult-onset DM; "
            "SHARED PATHWAY WITH MODY10 (INS): "
            "  PDX1 (β-cell master TF) activates INS gene → PDX1 LOF reduces INS transcription; "
            "  But pancreatic agenesis is due to early developmental loss, not just INS reduction"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — PDX1/PANCREATIC AGENESIS: "
            "  1. NEONATAL DIABETES + ABSENT PANCREAS ON IMAGING: "
            "     PANCREATIC AGENESIS IS PATHOGNOMONIC; "
            "     Absent pancreatic body/tail/head on neonatal ultrasound; "
            "  2. SEVERE EXOCRINE INSUFFICIENCY FROM BIRTH: "
            "     Fat-soluble vitamin deficiency (ADEK); "
            "     Steatorrhoea; very low/absent faecal elastase; "
            "  3. NEGATIVE ISLET AUTOANTIBODIES: not autoimmune; "
            "  4. PARENTS MODY4-like: heterozygous PDX1 → young-onset diabetes in parent; "
            "  5. CONSANGUINITY: AR homozygous more likely in consanguineous families; "
            "  6. C-PEPTIDE ABSENT: no endogenous insulin production"
        ),
        "treatment": (
            "TREATMENT — PDX1/PANCREATIC AGENESIS: INSULIN + PERT LIFELONG: "
            "DIABETES (ABSOLUTE INSULIN DEFICIENCY): "
            "  Insulin pump + CGM (neonatal from diagnosis); "
            "  No sulfonylurea (no K-ATP channel pathway issue); "
            "  Sensitive to hypoglycaemia (no glucagon counterregulation — α-cells also absent); "
            "    IMPORTANT: glucagon pen may NOT work (no α-cells → no glucagon response); "
            "    Glucose gel / IV dextrose for hypoglycaemia; "
            "EXOCRINE INSUFFICIENCY (PERT): "
            "  High-dose PERT (creon) with every meal and snack; "
            "  Infant formulas: semi-elemental or elemental if malabsorption severe; "
            "  Fat-soluble vitamins ADEK daily (monitor levels); "
            "  Monitor growth centiles; "
            "NUTRITIONAL: "
            "  High-calorie diet; dietitian from birth; "
            "  Monitor vitamin D, A, E, K levels every 6 months; "
            "MONITORING: "
            "  HbA1c every 3 months; "
            "  Annual pancreatic enzyme replacement adequacy (faecal elastase meaningless — no pancreas); "
            "  Bone density from adolescence (fat-soluble vitamin malabsorption → metabolic bone disease)"
        ),
        "seed": 2901,
    },
]

SEED_BASE = 2894

def _rng(seed):
    return random.Random(seed)

def _generate_patients(gene_entry):
    rng = _rng(gene_entry["seed"])
    gene = gene_entry["gene"]
    n = 40
    patients = []
    # Neonatal diabetes: onset age in weeks (1-24 weeks = <6 months)
    for i in range(n):
        age_onset_wks = rng.randint(1, 24)  # weeks of age at diagnosis
        age_current_yrs = rng.randint(1, 20)
        sex = rng.choice(["M", "F"])
        if gene == "FOXP3":
            sex = "M"  # X-linked, males only
        glucose_mmol = round(rng.uniform(14, 45), 1)
        hba1c_pct = round(rng.uniform(8.5, 14.0), 1)
        su_response = gene in ("KCNJ11", "ABCC8")
        c_peptide_low = gene in ("INS", "EIF2AK3", "RFX6", "GLIS3", "PDX1")
        autoab_pos = gene in ("FOXP3",)  # Only IPEX has positive autoantibodies
        patients.append({
            "id": f"{gene}-{i+1:02d}",
            "gene": gene,
            "age_onset_weeks": age_onset_wks,
            "age_current_years": age_current_yrs,
            "sex": sex,
            "glucose_mmol_l": glucose_mmol,
            "hba1c_pct": hba1c_pct,
            "su_response": su_response,
            "c_peptide_low": c_peptide_low,
            "islet_autoantibodies_positive": autoab_pos,
        })
    return patients

def generate_overview():
    genes = [g["gene"] for g in ATLAS_GENES]
    total_patients = 0
    gene_rows = []
    for entry in ATLAS_GENES:
        pts = _generate_patients(entry)
        total_patients += len(pts)
        avg_onset = round(sum(p["age_onset_weeks"] for p in pts) / len(pts), 1)
        avg_hba1c = round(sum(p["hba1c_pct"] for p in pts) / len(pts), 1)
        su_resp_n = sum(1 for p in pts if p["su_response"])
        gene_rows.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_summary": entry["protein"],
            "patients": len(pts),
            "avg_onset_weeks": avg_onset,
            "avg_hba1c_pct": avg_hba1c,
            "sulfonylurea_response": su_resp_n > 0,
            "su_response_pct": round(100 * su_resp_n / len(pts), 0) if su_resp_n > 0 else 0,
            "autoantibodies": "POSITIVE" if entry["gene"] == "FOXP3" else "NEGATIVE",
        })
    return {
        "atlas": "Hereditary Neonatal Diabetes Atlas",
        "subtitle": "8-Gene Reference: KCNJ11-ABCC8-INS-EIF2AK3-FOXP3-RFX6-GLIS3-PDX1",
        "description": (
            "Comprehensive atlas of hereditary monogenic neonatal and infancy-onset diabetes mellitus, "
            "covering all eight major genetic causes of diabetes presenting before 6 months of age. "
            "Includes K-ATP channel GOF (KCNJ11/ABCC8 — sulfonylurea curative), ER stress (INS/EIF2AK3), "
            "immune dysregulation (FOXP3/IPEX), developmental transcription factors (RFX6/GLIS3/PDX1). "
            "320 patients (8 × 40), seeds 2894-2901."
        ),
        "total_patients": total_patients,
        "total_genes": len(genes),
        "genes": genes,
        "gene_rows": gene_rows,
        "categories": {
            "K-ATP Channel GOF (sulfonylurea curative)": ["KCNJ11", "ABCC8"],
            "ER Stress / Beta-Cell Apoptosis": ["INS", "EIF2AK3"],
            "Immune Dysregulation (IPEX)": ["FOXP3"],
            "Developmental Transcription Factors": ["RFX6", "GLIS3", "PDX1"],
        },
        "key_facts": [
            "Any diabetes onset <6 months: genetic testing MANDATORY — not T1D autoimmune at this age",
            "KCNJ11/ABCC8 K-ATP GOF: sulfonylurea (glibenclamide) CURATIVE — 90%/80% transfer from insulin",
            "INS AD missense: ER stress mechanism — insulin ONLY, no sulfonylurea benefit",
            "EIF2AK3 (WRS): triad of ND + epiphyseal dysplasia + recurrent liver failure — most common AR ND",
            "FOXP3 (IPEX): ONLY neonatal DM with POSITIVE islet autoantibodies — HSCT only cure",
            "RFX6 (Mitchell-Riley): ND + hypothyroidism + intestinal atresia + gallbladder agenesis",
            "GLIS3: ND + congenital hypothyroidism + CHD + renal cysts — simultaneous dual endocrinopathy",
            "PDX1 homozygous: pancreatic agenesis — absent pancreas on imaging PATHOGNOMONIC; glucagon pen FAILS",
        ],
        "diagnostic_algorithm": (
            "Neonatal DM workup (onset <6 months): "
            "1. Confirm hyperglycaemia; DKA screen; islet autoantibodies (GAD/IA-2/ZnT8); C-peptide; "
            "2. Chromosome 6q24 methylation (most common TNDM cause — not gene mutation); "
            "3. KCNJ11 + ABCC8 first (most common PNDM, sulfonylurea treatable); "
            "   If positive → trial glibenclamide (can transfer even after years of insulin); "
            "4. If negative: INS sequencing (ER stress, insulin only); "
            "5. If AR features (consanguinity): EIF2AK3 (WRS triad), RFX6 (MRS), GLIS3, PDX1; "
            "6. If autoantibodies POSITIVE + severe enteropathy + male: FOXP3 (IPEX — HSCT); "
            "7. Imaging: pancreas ultrasound (PDX1/RFX6 — hypoplasia/agenesis); renal USS (GLIS3); echo (GLIS3)"
        ),
    }

def generate_breakdown():
    result = []
    for entry in ATLAS_GENES:
        pts = _generate_patients(entry)
        result.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein": entry["protein"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "patient_count": len(pts),
            "seed": entry["seed"],
        })
    return {"genes": result, "count": len(result)}

def generate_definitions():
    return {
        "definitions": [
            {
                "term": "Neonatal Diabetes Mellitus (NDM)",
                "definition": (
                    "Diabetes onset <6 months of age. Distinguished from Type 1 autoimmune diabetes, "
                    "which is extremely rare <6 months. Genetic testing MANDATORY in all cases. "
                    "Permanent (PNDM): persists lifelong. Transient (TNDM): remits weeks-months, may relapse. "
                    "Most TNDM: 6q24 methylation. Most PNDM: KCNJ11 GOF (K-ATP channel)."
                ),
            },
            {
                "term": "K-ATP Channel (Kir6.2/SUR1)",
                "definition": (
                    "ATP-sensitive potassium channel in β-cell plasma membrane: (Kir6.2)₄(SUR1)₄ octamer. "
                    "Function: closes when glucose rises (↑ATP) → depolarisation → Ca²⁺ influx → insulin. "
                    "GOF (KCNJ11/ABCC8): K-ATP stays open → no insulin → neonatal DM. "
                    "LOF (ABCC8/KCNJ11): K-ATP locked closed → excess insulin → congenital hyperinsulinism. "
                    "OPPOSITE PHENOTYPES from same genes depending on LOF vs GOF."
                ),
            },
            {
                "term": "Sulfonylurea (Glibenclamide) in Neonatal DM",
                "definition": (
                    "Sulfonylureas (glibenclamide/glyburide, glipizide) bind SUR1 → force K-ATP channel closed → "
                    "depolarisation → Ca²⁺ → insulin secretion — BYPASSES the ATP-sensing defect. "
                    "Indication: KCNJ11 GOF or ABCC8 GOF. "
                    "SUCCESS: ~90% KCNJ11, ~75-80% ABCC8 — insulin discontinued. "
                    "Dose: 0.05-0.8 mg/kg/day (much higher than T2D doses). "
                    "Switchover possible even after years of insulin therapy. "
                    "DEND benefit: partial neurological improvement if started early."
                ),
            },
            {
                "term": "DEND Syndrome",
                "definition": (
                    "Developmental delay + Epilepsy + Neonatal Diabetes — caused by severe KCNJ11 GOF. "
                    "Mechanism: Kir6.2 expressed in neurons → GOF → neurons chronically hyperpolarised → "
                    "impaired action potentials + seizures. "
                    "iDEND: intermediate — DD + muscle hypotonia WITHOUT epilepsy. "
                    "Sulfonylurea: neurological improvement in ~50% if started early (crosses BBB). "
                    "Do not wait: delay worsens irreversible neurodevelopmental outcomes."
                ),
            },
            {
                "term": "IPEX Syndrome (FOXP3 LOF)",
                "definition": (
                    "Immune dysregulation, Polyendocrinopathy, Enteropathy, X-linked. "
                    "FOXP3 loss → Treg absence → uncontrolled T-effectors attack self antigens. "
                    "KEY DISCRIMINATOR: islet autoantibodies POSITIVE (only monogenic ND with positive Ab). "
                    "Diarrhoea + eczema + diabetes in neonatal male = IPEX until proven otherwise. "
                    "HSCT is only cure; tacrolimus/sirolimus as bridge therapy."
                ),
            },
            {
                "term": "Wolcott-Rallison Syndrome (EIF2AK3/PERK LOF)",
                "definition": (
                    "PERK (eIF2α kinase) absent → ER stress unresolved → β-cell + chondrocyte + hepatocyte death. "
                    "Triad: Neonatal Diabetes + Multiple Epiphyseal Dysplasia + Recurrent Liver Failure. "
                    "Most common AR neonatal DM in consanguineous Middle Eastern/North African families. "
                    "Liver crises: triggered by fever/illness — urgent IV glucose + N-acetylcysteine. "
                    "No disease-modifying treatment; insulin lifelong; liver transplant if severe."
                ),
            },
            {
                "term": "Mitchell-Riley Syndrome (RFX6 LOF)",
                "definition": (
                    "RFX6 TF required for pancreatic endocrine + enteroendocrine + thyroid + biliary development. "
                    "Neonatal Diabetes + Hypothyroidism + Intestinal Atresia + Gallbladder Agenesis. "
                    "Pancreatic hypoplasia on imaging (endocrine cell agenesis + exocrine hypoplasia). "
                    "PERT mandatory. Surgical correction of intestinal atresia urgent at birth."
                ),
            },
            {
                "term": "Pancreatic Agenesis (PDX1 AR LOF)",
                "definition": (
                    "PDX1 is the master TF for entire pancreas development (exocrine + endocrine). "
                    "AR homozygous LOF → absent pancreas → absolute insulin deficiency + complete exocrine failure. "
                    "PATHOGNOMONIC: absent pancreas on neonatal imaging. "
                    "No glucagon (α-cells absent) → glucagon pen FAILS for hypoglycaemia — use glucose gel/IV. "
                    "PERT essential from birth. Contrast MODY4 (AD het) — normal pancreas, adult-onset DM."
                ),
            },
            {
                "term": "GLIS3 Syndrome",
                "definition": (
                    "GLIS3 zinc-finger TF required for β-cell, thyroid follicular cell, kidney collecting duct, "
                    "cardiac, and biliary development. "
                    "Neonatal Diabetes + Congenital Hypothyroidism + CHD + Polycystic Kidneys + Liver Fibrosis. "
                    "Simultaneous ND + CH on neonatal screen: suspect GLIS3 or RFX6. "
                    "Renal cysts → CKD in adolescence-adulthood; renal transplant may be needed."
                ),
            },
            {
                "term": "Islet Autoantibodies in Neonatal DM",
                "definition": (
                    "In monogenic neonatal DM: islet autoantibodies (GAD65, IA-2/ICA512, ZnT8) are NEGATIVE. "
                    "EXCEPTION: FOXP3 (IPEX) — autoantibodies POSITIVE (T-cell mediated β-cell destruction). "
                    "Negative autoantibodies in neonatal DM (<6 months) = strongly suggests monogenic cause. "
                    "Positive autoantibodies: IPEX first; true T1D onset <6 months is vanishingly rare."
                ),
            },
        ]
    }
