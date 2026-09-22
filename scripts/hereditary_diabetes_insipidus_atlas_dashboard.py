#!/usr/bin/env python3
"""Hereditary-Diabetes-Insipidus-Atlas — Complete 8-Gene DI Atlas
AVP     (arginine vasopressin; 164 aa; 20p13; AD/AR LOF;
         familial neurohypophyseal DI (FNDI/central DI);
         AD: progressive magnocellular neuronal loss (pro-AVP misfolding ER stress → neuronal death);
         AR biallelic: severe congenital non-progressive; MRI posterior pituitary bright spot ABSENT;
         desmopressin EFFECTIVE — distinguishes from NDI; seed SEED_BASE+0) ·
AVPR2   (vasopressin V2 receptor; 371 aa; Xq28; XLR;
         nephrogenic DI type 1 (NDI1); XLR males severely affected; females variable (X-inactivation);
         desmopressin COMPLETELY UNRESPONSIVE (V2R absent/dysfunctional); >200 mutations;
         treatment: hydrochlorothiazide + amiloride + low-solute diet;
         GOF → NSIAD (nephrogenic syndrome of inappropriate antidiuresis); seed SEED_BASE+1) ·
AQP2    (aquaporin-2 water channel; 271 aa; 12q13.12; AR/AD;
         nephrogenic DI type 2 (NDI2); AR biallelic classic; AD C-terminal truncations dominant-negative;
         desmopressin UNRESPONSIVE (V2R intact but AQP2 absent/non-trafficked);
         urine AQP2 absent diagnostic; emerging: sildenafil/statins improve AQP2 trafficking; seed SEED_BASE+2) ·
WFS1    (wolframin; 890 aa; 4p16.1; AR;
         Wolfram syndrome 1 (DIDMOAD); ER stress UPR → DI+DM+optic atrophy+deafness;
         SEQUENCE: optic atrophy age 6 → DM age 6 → DI age 14 → deafness age 16;
         DI central → desmopressin EFFECTIVE; DM non-immune (no autoantibodies); seed SEED_BASE+3) ·
CISD2   (CDGSH iron-sulfur domain 2; 135 aa; 4q24; AR;
         Wolfram syndrome 2 (WFS2); Israeli Arab founder p.Trp45Ser;
         KEY: NO DI in most patients; ADD bleeding tendency (peptic ulcers + GI bleeding);
         DM + optic atrophy + peripheral neuropathy + bleeding; seed SEED_BASE+4) ·
PCSK1   (proprotein convertase subtilisin/kexin type 1; 753 aa; 5q15; AR;
         PC1/3 deficiency; processes pro-AVP → AVP, pro-POMC → ACTH, pro-insulin → insulin;
         neonatal malabsorption (secretory diarrhea) PATHOGNOMONIC and FIRST;
         morbid obesity + central DI 50-60% + ACTH deficiency + hypogonadotropic hypogonadism; seed SEED_BASE+5) ·
KCNJ1   (ROMK Kir1.1; 391 aa; 11q24.3; AR;
         Bartter syndrome type 2 (antenatal/neonatal); polyhydramnios → premature birth;
         PARADOXICAL TRANSIENT NEONATAL HYPERKALEMIA (distinguishes from Bartter type 1/3);
         hypokalemia + metabolic alkalosis + hypercalciuria + TAL dysfunction polyuria; seed SEED_BASE+6) ·
SLC12A1 (NKCC2 Na-K-2Cl cotransporter; 1099 aa; 15q21.1; AR;
         Bartter syndrome type 1 (antenatal) — MOST SEVERE; polyhydramnios + premature;
         FUROSEMIDE-LIKE PHENOTYPE; nephrocalcinosis 80%; no neonatal hyperkalemia (unlike KCNJ1);
         hypokalemia + metabolic alkalosis + hypercalciuria; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2990-2997)
"""
import random

SEED_BASE = 2990

ATLAS_GENES = [
    {
        "gene": "AVP",
        "protein": (
            "AVP -- 20p13 AD/AR LOF -- 164aa -- Arginine-Vasopressin-Prepropeptide-"
            "Neurohypophyseal-Neuropeptide-ER-Stress-Neuronal-Death-"
            "FNDI-Central-DI-Desmopressin-EFFECTIVE-Posterior-Pituitary-Bright-Spot-ABSENT-OMIM-192340"
        ),
        "locus": "20p13",
        "protein_size": (
            "164 aa (AVP — arginine vasopressin prepropeptide; the precursor encodes three peptides: "
            "  (1) signal peptide; (2) AVP nonapeptide (antidiuretic hormone); (3) neurophysin II; (4) copeptin; "
            "FUNCTION: AVP (ADH) acts on V2 receptors in renal collecting duct principal cells: "
            "  AVP → V2R (Gs-coupled GPCR) → cAMP → PKA → AQP2 phosphorylation → AQP2 insertion into apical membrane; "
            "  Net effect: water reabsorption from urine → urine concentration; "
            "  High plasma osmolality / low volume → magnocellular neurons (hypothalamic supraoptic + paraventricular nuclei) → "
            "    AVP released from posterior pituitary → V2R → water reabsorption → urine concentration; "
            "AD LOF MECHANISM (FNDI — familial neurohypophyseal DI): "
            "  Missense mutations in pro-AVP → misfolded protein accumulates in ER; "
            "  ER stress → unfolded protein response (UPR) → progressive magnocellular neuron death; "
            "  DI onset: usually childhood (age 2-10 years); progressive polyuria/polydipsia; "
            "  MRI hallmark: POSTERIOR PITUITARY BRIGHT SPOT ABSENT (normal bright spot = stored AVP); "
            "  Penetrance: incomplete early, near-complete by adulthood; "
            "AR MECHANISM (biallelic LOF): "
            "  Pure AVP synthesis failure; severe congenital DI; non-progressive (neurons intact); "
            "  Less neurodegeneration than AD; complete AVP deficiency from birth; "
            "encoded 20p13; OMIM gene 192340, disease FNDI #125700"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT/RECESSIVE LOF — AVP / FAMILIAL NEUROHYPOPHYSEAL DI (FNDI): "
            "  AD (most common): "
            "    Progressive onset: age 2-10 years (not congenital); "
            "    Progressive worsening as more magnocellular neurons die (ER stress accumulation); "
            "    Polyuria (10-20 L/day untreated) + compensatory polydipsia; "
            "    Plasma osmolality elevated; urine osmolality low (50-150 mOsm/kg); "
            "    MRI POSTERIOR PITUITARY BRIGHT SPOT: ABSENT — hallmark finding; "
            "    DESMOPRESSIN RESPONSIVE — response distinguishes AVP LOF from NDI; "
            "  AR (biallelic): "
            "    Congenital severe DI from birth; non-progressive; "
            "    Parents unaffected (carriers); "
            "  DIAGNOSIS: "
            "    Water deprivation test → plasma osmolality rises, urine osmolality fails to concentrate; "
            "    Desmopressin administration → urine osmolality RISES >50% (central DI response); "
            "    MRI pituitary: posterior pituitary bright spot absent; "
            "    Copeptin measurement: low baseline copeptin confirms central DI; "
            "  TREATMENT: "
            "    Desmopressin (DDAVP): intranasal or oral; EFFECTIVE; titrate to prevent hyponatraemia; "
            "    Avoid overtreatment: hyponatraemia risk; "
            "  DIFFERENTIAL: "
            "    NDI (AVPR2/AQP2): desmopressin UNRESPONSIVE; "
            "    Wolfram syndrome (WFS1): DI + DM + optic atrophy + deafness; "
            "    Gestational DI: AVP degraded by placental vasopressinase; self-limited"
        ),
        "disease_category": (
            "FNDI-CENTRAL-DI-DESMOPRESSIN-EFFECTIVE: "
            "  KEY RULE: desmopressin response DISTINGUISHES central DI (AVP/WFS1/PCSK1) from NDI (AVPR2/AQP2); "
            "  MRI POSTERIOR PITUITARY BRIGHT SPOT ABSENT: pathognomonic for FNDI; "
            "  COPEPTIN LOW (<4.9 pmol/L after osmotic stimulation) confirms central DI; "
            "  AD ER STRESS MECHANISM: unique among endocrine diseases — protein toxicity causes neuron death; "
            "  GENETIC TESTING: AVP sequencing; cascade testing AD family; biallelic for AR cases; "
            "  TREATMENT: desmopressin (intranasal/oral/sublingual); lifelong; titrate carefully"
        ),
    },
    {
        "gene": "AVPR2",
        "protein": (
            "AVPR2 -- Xq28 XLR -- 371aa -- V2-Vasopressin-Receptor-"
            "GPCR-Gs-Coupled-cAMP-PKA-AQP2-Trafficking-"
            "NDI1-Desmopressin-COMPLETELY-UNRESPONSIVE-Thiazide-Amiloride-NSIAD-GOF-OMIM-300538"
        ),
        "locus": "Xq28",
        "protein_size": (
            "371 aa (AVPR2 — V2 vasopressin receptor; "
            "FUNCTION: Gs-coupled GPCR in renal collecting duct principal cells; "
            "  AVP binding → Gs → adenylyl cyclase → cAMP ↑ → PKA activation; "
            "  PKA phosphorylates AQP2 at Ser256 → AQP2 vesicles fuse with apical membrane; "
            "  Water channel insertion → osmotic water reabsorption from tubular lumen; "
            "  V2R also mediates: von Willebrand factor release from endothelium (DDAVP haemostatic use); "
            "XLR MECHANISM (NDI1): "
            "  >200 mutations described: missense (50%), frameshift, nonsense, splice-site; "
            "  Missense: often ER-retained (mistrafficking) or non-functional at membrane; "
            "  Desmopressin COMPLETELY UNRESPONSIVE: V2R absent or non-functional → no cAMP → no AQP2 insertion; "
            "  MALES: severely affected (hemizygous); severe neonatal polyuria + hypernatraemia; "
            "  FEMALES: variable (X-inactivation); usually mild or unaffected; occasional symptomatic (skewed inactivation); "
            "GOF VARIANTS → NSIAD (Nephrogenic Syndrome of Inappropriate Antidiuresis): "
            "  Constitutive V2R activation → constitutive AQP2 → water retention → dilutional hyponatraemia; "
            "  Opposite phenotype to NDI1; "
            "encoded Xq28; OMIM gene 300538, disease NDI1 #304800"
        ),
        "inheritance": (
            "X-LINKED RECESSIVE — AVPR2 / NEPHROGENIC DI TYPE 1 (NDI1): "
            "  MALES (hemizygous): "
            "    Severe neonatal onset: polyuria (up to 10-15 mL/kg/hour), hypernatraemia, fever, failure to thrive; "
            "    Risk of brain damage from recurrent hypernatraemic dehydration if unrecognised; "
            "    Persistent polyuria throughout life; "
            "  FEMALES (heterozygous): "
            "    Variable: usually mild; some completely asymptomatic; occasional severe (skewed X-inactivation); "
            "    Carrier females: 50% chance of affected sons; "
            "  DESMOPRESSIN COMPLETELY UNRESPONSIVE: "
            "    Desmopressin gives NO urine concentration (distinguishes from central DI); "
            "    Desmopressin haemostatic response (vWF) may also be absent; "
            "  DIAGNOSIS: "
            "    Clinical: infant male + polyuria + hypernatraemia; "
            "    Water deprivation: no urine concentration; desmopressin: no response; "
            "    Copeptin: elevated (AVP high as appropriate response); "
            "    Genetic: AVPR2 sequencing; females: carrier testing; "
            "  TREATMENT: "
            "    Hydrochlorothiazide (25 mg/day in infants): paradoxical — thiazide-induced mild volume depletion → "
            "      reduced GFR → more proximal reabsorption → less fluid to collecting duct; "
            "    Amiloride: blocks ENaC; reduces urine output further; "
            "    Low-solute diet: reduces obligatory osmolar excretion → less obligatory water; "
            "    Desmopressin: INEFFECTIVE — do not use for DI (confusion with haemostasis use); "
            "    Adequate water intake: prevent hypernatraemia; "
            "  PROGNOSIS: brain damage risk if early hypernatraemia unrecognised; normal IQ with treatment"
        ),
        "disease_category": (
            "NDI1-XLR-DESMOPRESSIN-COMPLETELY-UNRESPONSIVE-THIAZIDE-AMILORIDE: "
            "  KEY RULE: infant male + polyuria + hypernatraemia → exclude NDI1 (AVPR2) first; "
            "  DESMOPRESSIN INEFFECTIVE: do NOT prescribe DDAVP for water balance (confuses with haemostatic use); "
            "  THIAZIDE PARADOX: thiazides REDUCE polyuria in NDI (opposite of expected diuretic effect); "
            "  NSIAD GOF: constitutive V2R → hyponatraemia + suppressed urine output; opposite phenotype; "
            "  GENETIC COUNSELLING: X-linked; 50% of sons of carrier mothers affected; "
            "  NEONATAL RECOGNITION: critical to prevent hypernatraemic brain damage"
        ),
    },
    {
        "gene": "AQP2",
        "protein": (
            "AQP2 -- 12q13.12 AR/AD -- 271aa -- Aquaporin-2-"
            "Water-Channel-26kDa-Collecting-Duct-Principal-Cell-Apical-Membrane-"
            "NDI2-Desmopressin-UNRESPONSIVE-Urine-AQP2-Absent-Trafficking-Defect-OMIM-107777"
        ),
        "locus": "12q13.12",
        "protein_size": (
            "271 aa / 26 kDa (AQP2 — aquaporin-2 water channel; "
            "FUNCTION: water channel expressed in collecting duct principal cells; "
            "  Apical membrane: AQP2 mediates water entry from tubular lumen into cell; "
            "  Basolateral: AQP3/AQP4 mediate water exit into peritubular capillary; "
            "  TRAFFICKING REGULATION: "
            "    Unstimulated: AQP2 stored in intracellular vesicles; "
            "    AVP → V2R → cAMP → PKA → phospho-Ser256-AQP2 → vesicle fusion with apical membrane; "
            "    AQP2 inserted → water reabsorption; "
            "    Dephosphorylation → AQP2 endocytosis back to vesicles; "
            "AR MUTATIONS (biallelic — NDI2 classic): "
            "  Missense in transmembrane domains or AQP2 pore → protein misfolded → ER retention; "
            "  Or trafficking to apical membrane impaired; "
            "  V2R intact and functional; cAMP response normal; but NO AQP2 at apical membrane; "
            "AD MUTATIONS (dominant-negative — less common): "
            "  C-terminal truncation mutations; "
            "  Mutant AQP2 forms heterotetramer with wild-type → dominant-negative → misrouting; "
            "  Heterozygous → disease (dominant-negative mechanism); "
            "URINE AQP2: normally excreted in urine (vesicle exocytosis); absent in NDI2 = diagnostic marker; "
            "encoded 12q13.12; OMIM gene 107777, disease NDI2 #125800 (AR), #107777 (AD)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic) / AUTOSOMAL DOMINANT (C-terminal) — AQP2 / NDI TYPE 2: "
            "  AR (biallelic): "
            "    Both alleles LOF; severe NDI; neonatal/infantile onset; "
            "    V2R functional (cAMP rises after DDAVP) but no AQP2 response; "
            "    More common AR form; "
            "  AD (dominant-negative C-terminal): "
            "    Single allele → dominant-negative effect on tetrameric AQP2; "
            "    May have milder phenotype than biallelic AR; "
            "    De novo mutations possible; "
            "  DESMOPRESSIN UNRESPONSIVE: "
            "    V2R intact → copeptin response normal; cAMP rises → but AQP2 cannot traffic → no water reabsorption; "
            "    Clinically: desmopressin water deprivation test: no urine concentration despite normal V2R; "
            "    Distinguishing from AVPR2: cAMP (or blood pressure rise) after desmopressin present in AQP2 mutations; "
            "  URINE AQP2 MEASUREMENT: "
            "    Absent urine AQP2 (by ELISA) = diagnostic for AQP2 NDI; "
            "    Present urine AQP2 = AQP2 is trafficked = post-receptor defect elsewhere; "
            "  TREATMENT: "
            "    Thiazide + amiloride (same as NDI1); low-solute diet; "
            "    Emerging therapies: sildenafil (PDE5 inhibitor → cGMP → alternative AQP2 trafficking); "
            "      Statins (HMG-CoA reductase inhibitors): improve AQP2 membrane insertion (preclinical/early clinical); "
            "    Desmopressin: INEFFECTIVE for water balance (V2R present but AQP2 absent)"
        ),
        "disease_category": (
            "NDI2-AR-AD-DESMOPRESSIN-UNRESPONSIVE-URINE-AQP2-ABSENT: "
            "  KEY DISTINCTION from AVPR2 NDI1: V2R intact → cAMP rises after DDAVP; defect post-receptor (AQP2); "
            "  URINE AQP2 ABSENT: diagnostic marker; orders of magnitude lower than normal; "
            "  SILDENAFIL/STATINS: emerging AQP2 trafficking rescue (clinical trial evidence growing); "
            "  AD (C-terminal): dominant-negative; single allele sufficient; de novo mutations possible; "
            "  TREATMENT: thiazide + amiloride; low-solute diet; emerging: sildenafil; statins; "
            "  GENETIC TESTING: AQP2 sequencing; biallelic for AR; heterozygous C-terminal for AD"
        ),
    },
    {
        "gene": "WFS1",
        "protein": (
            "WFS1 -- 4p16.1 AR -- 890aa -- Wolframin-"
            "ER-Transmembrane-Protein-9-TM-Helices-100kDa-UPR-ER-Calcium-"
            "Wolfram-Syndrome-1-DIDMOAD-DI-DM-OpticAtrophy-Deafness-NonImmune-DM-Desmopressin-EFFECTIVE-OMIM-606201"
        ),
        "locus": "4p16.1",
        "protein_size": (
            "890 aa / 100 kDa (WFS1 — wolframin; ER transmembrane protein; "
            "FUNCTION: ER calcium homeostasis + UPR modulation; "
            "  9 transmembrane helices; ER-resident; "
            "  Regulates ER calcium release (IP3R channels); maintains ER Ca2+ homeostasis; "
            "  Involved in unfolded protein response (UPR) modulation: reduces ER stress; "
            "  Critical for beta cell survival (high secretory load → ER stress susceptibility); "
            "  Also expressed in: neurons (hypothalamic magnocellular → AVP secretion), cochlear hair cells, retinal ganglion cells; "
            "WFS1 LOF → ER STRESS IN MULTIPLE CELL TYPES: "
            "  Beta cells: progressive ER stress → UPR → apoptosis → insulin-dependent DM (non-immune); "
            "  Magnocellular neurons: ER stress → cell death → central DI; "
            "  Retinal ganglion cells: optic atrophy (bilateral); "
            "  Cochlear hair cells: sensorineural deafness; "
            "WOLFRAM SYNDROME 1 (DIDMOAD) SEQUENCE: "
            "  Optic atrophy: mean age 6 years (bilateral, progressive); "
            "  Diabetes mellitus: mean age 6 years (insulin-dependent; NO autoantibodies → non-immune); "
            "  Diabetes insipidus: mean age 14 years (CENTRAL; desmopressin EFFECTIVE); "
            "  Deafness (sensorineural): mean age 16 years (bilateral); "
            "HET WFS1 common variants: "
            "  p.Arg456His and others: adult-onset DM risk ONLY (NOT Wolfram syndrome); "
            "encoded 4p16.1; OMIM gene 606201, disease Wolfram 1 #222300"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — WFS1 / WOLFRAM SYNDROME 1 (DIDMOAD): "
            "  CLINICAL SEQUENCE: "
            "    1. Optic atrophy (age 6): bilateral; loss of central vision; no drusen/pigment; "
            "    2. Diabetes mellitus (age 6): insulin-dependent; NO autoantibodies (GAD-65, IA-2, ZnT8); "
            "       Distinguishes from type 1 DM (autoimmune); non-immune beta cell loss from ER stress; "
            "    3. Diabetes insipidus (age 14): CENTRAL DI; desmopressin EFFECTIVE; polyuria/polydipsia; "
            "       Posterior pituitary bright spot may be absent on MRI; "
            "    4. Sensorineural deafness (age 16): bilateral; high-frequency first; "
            "  ADDITIONAL FEATURES (variable): "
            "    Neurological: ataxia, peripheral neuropathy, psychiatric illness, cognitive decline; "
            "    Urological: neurogenic bladder (atonic), hydronephrosis; "
            "    Endocrine: hypogonadism; "
            "    GI: gastroparesis; "
            "  DM NON-IMMUNE: "
            "    CRITICAL DISTINCTION from T1DM: no autoantibodies; non-immune mechanism; "
            "    HbA1c may be spuriously low (haemolysis); "
            "    Treatment: insulin (same as T1DM in practice); "
            "  PROGNOSIS: multisystem; mean survival ~35 years; brainstem degeneration terminal; "
            "  HET WFS1 CARRIERS: "
            "    DM risk ~3-5x increased; psychiatric disease risk; "
            "    NOT Wolfram syndrome (full DIDMOAD requires biallelic WFS1 LOF)"
        ),
        "disease_category": (
            "WOLFRAM-SYNDROME-1-DIDMOAD-CENTRAL-DI-NONIMMUNE-DM: "
            "  KEY RULE: young patient + insulin-dependent DM + NO autoantibodies → suspect Wolfram (WFS1); "
            "  DM NON-IMMUNE: GAD-65 / IA-2 / ZnT8 antibodies ABSENT — critical distinguishing feature; "
            "  OPTIC ATROPHY: bilateral; no drusen → not AMD; MRI needed (brainstem atrophy late); "
            "  DI CENTRAL: desmopressin EFFECTIVE (unlike NDI); "
            "  MULTIDISCIPLINARY: endocrinology + ophthalmology + audiology + neurology + urology; "
            "  TREATMENT: desmopressin for DI; insulin for DM; no disease-modifying therapy available; "
            "  GENETIC TESTING: WFS1 sequencing + MLPA; biallelic mutations required for diagnosis"
        ),
    },
    {
        "gene": "CISD2",
        "protein": (
            "CISD2 -- 4q24 AR -- 135aa -- CDGSH-Iron-Sulfur-Domain-Containing-Protein-2-"
            "ERIS-NAF1-Mitochondrial-Outer-Membrane-ER-Calcium-Homeostasis-"
            "Wolfram-Syndrome-2-WFS2-NO-DI-Bleeding-Tendency-IsraeliArab-Founder-OMIM-611507"
        ),
        "locus": "4q24",
        "protein_size": (
            "135 aa (CISD2 — CDGSH iron-sulfur domain 2; also known as ERIS or NAF-1; "
            "FUNCTION: iron-sulfur protein on mitochondrial outer membrane and ER; "
            "  Contains CDGSH iron-sulfur domain (1 [2Fe-2S] cluster); "
            "  Localisation: mitochondrial outer membrane + MAM (mitochondria-associated ER membranes); "
            "  Role: mitochondrial calcium homeostasis; ER calcium regulation; mitophagy; "
            "  Interacts with BCL-2 at mitochondria → anti-apoptotic; "
            "  Without CISD2: mitochondrial and ER calcium dysequilibrium → cell death in multiple tissues; "
            "WOLFRAM SYNDROME 2 (WFS2): "
            "  COMPARED TO WFS1: "
            "    SAME: DM + optic atrophy; "
            "    KEY DIFFERENCE 1: NO DI in most WFS2 patients (unlike WFS1 where DI is ~70-75%); "
            "    KEY DIFFERENCE 2: ADD bleeding tendency — peptic ulcers + upper GI bleeding + high platelet reactivity; "
            "    Peripheral neuropathy: present in WFS2; "
            "  FOUNDER MUTATION: p.Trp45Ser (c.134G>C) in Israeli Arab families; "
            "    First described in Bedouin Arab families from Israel; "
            "    Most WFS2 families reported from Middle East; "
            "  OPTIC ATROPHY: earlier onset than WFS1 (mean age 3-4 years); severe; "
            "encoded 4q24; OMIM gene 611507, disease WFS2 #604928"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — CISD2 / WOLFRAM SYNDROME 2 (WFS2): "
            "  PHENOTYPE: "
            "    DM: insulin-dependent; non-immune (no autoantibodies); "
            "    Optic atrophy: bilateral; early onset (mean age 3-4 years); often severe; "
            "    Peripheral neuropathy: sensorimotor; present in most WFS2 patients; "
            "    Bleeding tendency: "
            "      Upper GI bleeding: peptic ulcers + gastritis; "
            "      High platelet reactivity (hyperaggregable platelets); "
            "      NOT present in WFS1 — WFS2 DISTINGUISHING FEATURE; "
            "  DI STATUS: "
            "    Most WFS2 patients do NOT have clinically significant DI; "
            "    Subclinical AVP deficiency in some; "
            "    DI prevalence: ~15% (vs ~70-75% in WFS1); "
            "  KEY DIFFERENCES WFS1 vs WFS2: "
            "    WFS1: DI common (~75%); deafness common (~65%); no bleeding tendency; "
            "    WFS2: DI rare (~15%); bleeding tendency PATHOGNOMONIC for WFS2; no deafness (rare); "
            "  FOUNDER POPULATION: "
            "    p.Trp45Ser: Israeli Arab + Bedouin families; "
            "    Other mutations: rare case reports worldwide; "
            "  TREATMENT: "
            "    Insulin (DM); ophthalmology follow-up (optic atrophy progression); "
            "    GI protection (PPI for peptic ulcers); monitor platelet function; "
            "    Desmopressin: only if DI confirmed (uncommon)"
        ),
        "disease_category": (
            "WFS2-NO-DI-MOSTLY-BLEEDING-TENDENCY-ISRAELI-ARAB-FOUNDER: "
            "  KEY DISTINGUISHING RULE: WFS2 has BLEEDING TENDENCY (peptic ulcers, platelet hyperaggregability) — NOT in WFS1; "
            "  DI ABSENT in ~85%: WFS2 mainly DM + optic atrophy + neuropathy + bleeding; "
            "  FOUNDER: p.Trp45Ser in Israeli Arab / Bedouin families → test this variant first; "
            "  GENETIC TESTING: CISD2 sequencing; biallelic mutations required; "
            "  TREATMENT: insulin; PPI; ophthalmology; monitor platelet function; desmopressin only if DI confirmed; "
            "  DIFFERENTIAL: WFS1 — DI present; deafness; no bleeding; WFS2 — DI absent; bleeding"
        ),
    },
    {
        "gene": "PCSK1",
        "protein": (
            "PCSK1 -- 5q15 AR -- 753aa -- Proprotein-Convertase-Subtilisin-Kexin-Type-1-"
            "PC1-3-83kDa-Serine-Protease-Pro-Hormone-Processor-"
            "Neonatal-Malabsorption-PATHOGNOMONIC-FIRST-Morbid-Obesity-Central-DI-55pct-ACTH-Deficiency-OMIM-600955"
        ),
        "locus": "5q15",
        "protein_size": (
            "753 aa / 83 kDa (PCSK1 — proprotein convertase subtilisin/kexin type 1; also called PC1/3; "
            "FUNCTION: serine protease; processes prohormones in regulated secretory pathway (neuroendocrine cells); "
            "  Substrates: "
            "    Pro-AVP → AVP (in magnocellular neurons) → DI if PCSK1 deficient; "
            "    Pro-POMC → ACTH → adrenal insufficiency if PCSK1 deficient; "
            "    Pro-insulin → insulin (in beta cells) → hyperproinsulinaemia; "
            "    Pro-GLP-1: GLP-1 processing impaired → glucose regulation affected; "
            "    Pro-NPY, pro-GnRH, pro-TRH: hypogonadism + hypothyroidism; "
            "    Pro-enteroglucagon in intestinal L cells → malabsorption; "
            "PCSK1 DEFICIENCY (biallelic LOF): "
            "  NEONATAL MALABSORPTION (FIRST AND PATHOGNOMONIC): "
            "    Secretory diarrhea in neonates/infancy; severe; life-threatening; "
            "    Intestinal PCSK1 processes prohormones essential for enterocyte function; "
            "    Malabsorption → failure to thrive; "
            "    FIRST symptom before obesity or DI develop; "
            "  MORBID OBESITY: profound hyperphagia; severe early-onset obesity; "
            "    Mechanism: impaired POMC/NPY processing → leptin resistance; MC4R pathway disrupted; "
            "  CENTRAL DI: 50-60% of patients; PCSK1 cannot process pro-AVP → AVP deficiency; "
            "  ACTH DEFICIENCY: adrenal insufficiency; pro-POMC cannot be cleaved to ACTH; "
            "  HYPOGONADOTROPIC HYPOGONADISM: pro-GnRH processing impaired; "
            "  HYPOTHYROIDISM: pro-TRH processing impaired; "
            "HET COMMON VARIANTS (p.Asn221Asp, p.Gln665Glu): "
            "  Common obesity variants (~3x increased obesity risk); "
            "  NOT causing DI or full PCSK1 deficiency syndrome; "
            "encoded 5q15; OMIM gene 162150, disease PCSK1 deficiency #600955"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — PCSK1 / PC1/3 DEFICIENCY: "
            "  CLINICAL SEQUENCE (diagnosis order): "
            "    1. NEONATAL MALABSORPTION: secretory diarrhea; neonatal/infant; PATHOGNOMONIC and FIRST; "
            "       Often attributed to other causes before diagnosis; "
            "    2. MORBID OBESITY: profound early-onset (age 2-5 years); BMI >> 40; hyperphagia; "
            "    3. CENTRAL DI: present in ~50-60%; polyuria/polydipsia; desmopressin EFFECTIVE; "
            "    4. ACTH DEFICIENCY: hypoglycaemia, lethargy; adrenal crisis risk; cortisol low; ACTH low; "
            "       Hyperproinsulinaemia (pro-insulin not cleaved); "
            "    5. HYPOGONADOTROPIC HYPOGONADISM: pubertal delay; amenorrhoea; "
            "    6. HYPOTHYROIDISM: central (TSH low/inappropriately normal + low fT4); "
            "  DIAGNOSIS: "
            "    Elevated pro-insulin (disproportionate to insulin) = PCSK1 deficiency signature; "
            "    Pro-ACTH elevated; "
            "    Genetic: PCSK1 sequencing; biallelic mutations; "
            "  TREATMENT: "
            "    Desmopressin for DI (EFFECTIVE); "
            "    Hydrocortisone for ACTH deficiency; "
            "    Levothyroxine for hypothyroidism; "
            "    Sex steroids for hypogonadism; "
            "    Very low fat diet for malabsorption; "
            "    Obesity: no specific therapy; orlistat limited by malabsorption; bariatric surgery reported; "
            "  HET VARIANTS: obesity risk only (not full syndrome)"
        ),
        "disease_category": (
            "PC1-3-DEFICIENCY-MALABSORPTION-FIRST-DI-55pct-DESMOPRESSIN-EFFECTIVE: "
            "  PATHOGNOMONIC CLUE: neonatal secretory diarrhea + morbid obesity + multiple hormone deficiencies; "
            "  MALABSORPTION FIRST: always before obesity and DI develop — key diagnostic clue; "
            "  PRO-INSULIN ELEVATED: signature biomarker (10-100x normal pro-insulin:insulin ratio); "
            "  DI CENTRAL: ~55%; desmopressin EFFECTIVE (same as WFS1, AVP); "
            "  MULTIHORMONE DEFICIENCY: replace each axis (cortisol, thyroid, sex steroids, vasopressin); "
            "  GENETIC TESTING: PCSK1 sequencing; biallelic pathogenic variants confirm diagnosis"
        ),
    },
    {
        "gene": "KCNJ1",
        "protein": (
            "KCNJ1 -- 11q24.3 AR -- 391aa -- ROMK-Renal-Outer-Medullary-Potassium-Channel-"
            "Kir1.1-Inwardly-Rectifying-K-Channel-45kDa-TAL-DCT-"
            "Bartter-Syndrome-Type-2-Antenatal-Neonatal-PARADOXICAL-HYPERKALEMIA-Polyhydramnios-OMIM-600359"
        ),
        "locus": "11q24.3",
        "protein_size": (
            "391 aa / 45 kDa (KCNJ1 — ROMK; renal outer medullary potassium channel; Kir1.1; "
            "FUNCTION: inwardly-rectifying K+ channel; APICAL membrane of thick ascending limb (TAL) + connecting tubule; "
            "  In TAL: ROMK recycles K+ into tubular lumen (essential for continued NKCC2 activity); "
            "    NKCC2 transports Na+/K+/2Cl- from lumen into cell; but K+ is limiting → ROMK recycles K+ back; "
            "    Without ROMK: NKCC2 stops → TAL reabsorption abolished → furosemide-like biochemistry; "
            "  In collecting duct: ROMK forms principal cell K+ secretion channel (regulated by aldosterone); "
            "BARTTER SYNDROME TYPE 2 (KCNJ1 LOF): "
            "  ANTENATAL: polyhydramnios (fetal polyuria from TAL dysfunction); premature birth common (32-36 weeks); "
            "  NEONATAL: severe life-threatening electrolyte crisis; "
            "  PARADOXICAL TRANSIENT HYPERKALEMIA: "
            "    ROMK is also main K+ secretory channel in collecting duct; "
            "    Without collecting duct ROMK: K+ cannot be secreted → INITIAL HYPERKALEMIA; "
            "    Resolved over weeks as aldosterone-driven upregulation of other K+ channels (BK channel); "
            "    DISTINGUISHES Bartter type 2 from types 1 and 3 (no neonatal hyperkalemia); "
            "  CHRONIC: hypokalaemic metabolic alkalosis + hypercalciuria + normal/low BP; "
            "  Polyuria: TAL dysfunction → reduced concentrating ability; urine not maximally concentrated; "
            "    Urine osmolality: typically 80-200 mOsm/kg (hypotonic but not zero — can partially concentrate); "
            "  PGE2 ELEVATED: prostaglandin-mediated; explains fever and constitutional symptoms; "
            "encoded 11q24.3; OMIM gene 600359, disease Bartter 2 #241200"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — KCNJ1 / BARTTER SYNDROME TYPE 2: "
            "  ANTENATAL PRESENTATION: "
            "    Polyhydramnios: detected on antenatal ultrasound; fetal urine production massively elevated; "
            "    Premature birth (32-36 weeks): obstetric intervention often required; "
            "  NEONATAL: "
            "    Life-threatening electrolyte disturbance; "
            "    PARADOXICAL TRANSIENT HYPERKALEMIA: first days-weeks of life; ROMK absent → K+ not secreted in CD; "
            "      Resolves spontaneously over weeks as BK channels upregulate; "
            "      Must recognise: Bartter + hyperkalemia = TYPE 2 (KCNJ1); all other Bartter types = hypokalaemic; "
            "    Hyponatraemia + metabolic alkalosis + elevated PGE2; "
            "    Polyuria + polydipsia + failure to thrive; "
            "  CHRONIC BIOCHEMISTRY: "
            "    Hypokalaemia (after neonatal hyperkalemia resolves); "
            "    Metabolic alkalosis (raised HCO3-); "
            "    Hypercalciuria + nephrocalcinosis (may develop, less severe than Bartter 1); "
            "    Normal or low BP (unlike Gitelman — normal/low BP; vs hypertension in AME); "
            "    Elevated renin + aldosterone (secondary hyperaldosteronism); "
            "    PGE2 markedly elevated in urine/serum; "
            "  DIAGNOSIS: "
            "    Clinical biochemistry; exclude diuretic abuse; "
            "    Genetic: KCNJ1 sequencing; biallelic pathogenic variants; "
            "  TREATMENT: "
            "    Acute: IV KCl + NaCl rehydration; correct alkalosis; "
            "    Chronic: KCl + NaCl supplements; indomethacin (COX inhibitor → reduces PGE2 → reduces polyuria); "
            "    Indomethacin most effective treatment; start after first weeks (renal maturity); "
            "    Monitor growth + renal function; nephrocalcinosis surveillance"
        ),
        "disease_category": (
            "BARTTER-TYPE-2-KCNJ1-NEONATAL-PARADOXICAL-HYPERKALEMIA-POLYHYDRAMNIOS: "
            "  PATHOGNOMONIC: neonatal hyperkalemia + Bartter phenotype = TYPE 2 (KCNJ1) specifically; "
            "  POLYHYDRAMNIOS: antenatal marker; premature birth; "
            "  HYPOKALAEMIC ALKALOSIS (after neonatal period): typical Bartter; hypercalciuria; "
            "  PGE2 ELEVATED: indomethacin (COX inhibition) = most effective chronic treatment; "
            "  POLYURIA MECHANISM: TAL dysfunction → impaired urine concentration (not zero — can partially concentrate); "
            "  GENETIC TESTING: KCNJ1 sequencing; biallelic mutations; ROMK type is type 2"
        ),
    },
    {
        "gene": "SLC12A1",
        "protein": (
            "SLC12A1 -- 15q21.1 AR -- 1099aa -- NKCC2-Na-K-2Cl-Cotransporter-Type-2-"
            "Apical-TAL-120kDa-SLC12-Family-Furosemide-Target-"
            "Bartter-Syndrome-Type-1-Antenatal-MOST-SEVERE-Furosemide-Like-Nephrocalcinosis-80pct-OMIM-600839"
        ),
        "locus": "15q21.1",
        "protein_size": (
            "1099 aa / 120 kDa (SLC12A1 — NKCC2; Na-K-2Cl cotransporter type 2; "
            "FUNCTION: apical membrane cotransporter of TAL (thick ascending limb of Henle); "
            "  Electroneutral cotransport: 1 Na+ + 1 K+ + 2 Cl- from lumen into TAL cell; "
            "  Driven by Na+ gradient (maintained by basolateral Na-K-ATPase); "
            "  NKCC2 is responsible for ~25% of total renal NaCl reabsorption; "
            "  FUROSEMIDE BINDING SITE: furosemide (loop diuretic) specifically inhibits NKCC2; "
            "    LOF mutations phenotype = chronic furosemide effect; "
            "  URINE CONCENTRATION: TAL reabsorption generates medullary hypertonicity; "
            "    Without TAL function (NKCC2 LOF): medullary gradient abolished → cannot concentrate urine; "
            "    Urine osmolality severely low (50-180 mOsm/kg); "
            "  CALCIUM: TAL calcium reabsorption (paracellular, driven by lumen-positive voltage); "
            "    Without lumen-positive voltage (NKCC2 LOF): hypercalciuria → nephrocalcinosis (80%); "
            "BARTTER SYNDROME TYPE 1 (SLC12A1 LOF): "
            "  MOST SEVERE Bartter type: "
            "    Polyhydramnios + premature birth (often < 32 weeks); "
            "    Life-threatening neonatal electrolyte crisis (more severe than type 2); "
            "    NO neonatal hyperkalemia (unlike KCNJ1 Bartter type 2); "
            "    Nephrocalcinosis: ~80% of patients (more than Bartter 2); "
            "encoded 15q21.1; OMIM gene 600839, disease Bartter 1 #601678"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — SLC12A1 / BARTTER SYNDROME TYPE 1 (MOST SEVERE): "
            "  ANTENATAL: "
            "    Severe polyhydramnios (earlier onset than Bartter 2): often detected 20-24 weeks; "
            "    Premature birth < 32 weeks common (vs 32-36 weeks in Bartter 2); "
            "    Fetal growth restriction in some; "
            "  NEONATAL: "
            "    Severe electrolyte crisis: hypokalaemia + hyponatraemia + metabolic alkalosis; "
            "    Fever + vomiting + failure to thrive; "
            "    NO neonatal hyperkalemia (KCNJ1/collecting duct channel intact); "
            "      ROMK (KCNJ1) functional → K+ secretion intact → hypokalaemia from birth; "
            "    Life-threatening if not treated; "
            "  CHRONIC BIOCHEMISTRY: "
            "    Hypokalaemia + metabolic alkalosis (persistent); "
            "    Hypercalciuria: severe; "
            "    Nephrocalcinosis: 80% (medullary; bilateral); MOST SEVERE AND FREQUENT of all Bartter types; "
            "    Normal/low BP; elevated renin + aldosterone; PGE2 elevated; "
            "    Polyuria: severe; urine osmolality 50-180 mOsm/kg; "
            "  FUROSEMIDE-LIKE PHENOTYPE: "
            "    Biochemistry identical to chronic furosemide overdose; "
            "    Useful for pathophysiology understanding and patient explanation; "
            "  DIAGNOSIS: "
            "    Clinical biochemistry + antenatal history; "
            "    Genetic: SLC12A1 sequencing; biallelic pathogenic variants; "
            "  TREATMENT: "
            "    Acute: aggressive IV KCl + NaCl + bicarbonate correction; "
            "    Chronic: KCl + NaCl supplements; indomethacin; "
            "    Nephrocalcinosis surveillance: renal USS annually; GFR monitoring; "
            "    Thiazide: not helpful (affects DCT, not TAL); "
            "    Prognosis: GFR usually preserved if treated; nephrocalcinosis can progress to CKD"
        ),
        "disease_category": (
            "BARTTER-TYPE-1-SLC12A1-FUROSEMIDE-LIKE-MOST-SEVERE-NEPHROCALCINOSIS-80pct: "
            "  MOST SEVERE Bartter: earliest polyhydramnios; most premature; most nephrocalcinosis; "
            "  FUROSEMIDE-LIKE: phenotype identical to chronic furosemide — NKCC2 target; "
            "  NO NEONATAL HYPERKALEMIA: distinguishes from Bartter 2 (KCNJ1); "
            "  NEPHROCALCINOSIS 80%: bilateral medullary; long-term renal function monitoring critical; "
            "  TREATMENT: indomethacin + KCl/NaCl supplements; aggressive acute resuscitation; "
            "  GENETIC TESTING: SLC12A1 sequencing; biallelic mutations; type 1 = most severe Bartter"
        ),
    },
]


def _make_patients(seed: int, gene: str) -> list:
    """Generate 40 synthetic DI-spectrum patients per gene."""
    rng = random.Random(seed)

    gene_profiles = {
        "AVP":     dict(polyuria_severe_pct=0.70, desmopresin_responsive_pct=0.95,
                        neonatal_onset_pct=0.05, urine_osmol_min=50, urine_osmol_max=150),
        "AVPR2":   dict(polyuria_severe_pct=0.90, desmopresin_responsive_pct=0.00,
                        neonatal_onset_pct=0.85, urine_osmol_min=50, urine_osmol_max=100),
        "AQP2":    dict(polyuria_severe_pct=0.80, desmopresin_responsive_pct=0.00,
                        neonatal_onset_pct=0.70, urine_osmol_min=80, urine_osmol_max=150),
        "WFS1":    dict(polyuria_severe_pct=0.45, desmopresin_responsive_pct=0.70,
                        neonatal_onset_pct=0.00, urine_osmol_min=80, urine_osmol_max=200),
        "CISD2":   dict(polyuria_severe_pct=0.10, desmopresin_responsive_pct=0.15,
                        neonatal_onset_pct=0.00, urine_osmol_min=120, urine_osmol_max=250),
        "PCSK1":   dict(polyuria_severe_pct=0.35, desmopresin_responsive_pct=0.55,
                        neonatal_onset_pct=0.10, urine_osmol_min=100, urine_osmol_max=200),
        "KCNJ1":   dict(polyuria_severe_pct=0.60, desmopresin_responsive_pct=0.00,
                        neonatal_onset_pct=0.95, urine_osmol_min=80, urine_osmol_max=200),
        "SLC12A1": dict(polyuria_severe_pct=0.75, desmopresin_responsive_pct=0.00,
                        neonatal_onset_pct=0.98, urine_osmol_min=50, urine_osmol_max=180),
    }
    p = gene_profiles.get(gene, gene_profiles["AVP"])

    associated_features_map = {
        "AVP":     [["progressive-polyuria-polydipsia"], ["posterior-pituitary-bright-spot-absent"],
                    ["progressive-polyuria-polydipsia", "posterior-pituitary-bright-spot-absent"],
                    ["childhood-onset-DI"]],
        "AVPR2":   [["hypernatraemia-neonatal"], ["failure-to-thrive"],
                    ["hypernatraemia-neonatal", "failure-to-thrive"],
                    ["polyuria-polydipsia-severe", "hypernatraemia"]],
        "AQP2":    [["urine-AQP2-absent"], ["NDI2-AR"],
                    ["urine-AQP2-absent", "thiazide-partial-response"],
                    ["NDI2-dominant-negative-C-terminal"]],
        "WFS1":    [["DM-non-immune-insulin-dependent", "optic-atrophy"],
                    ["optic-atrophy", "sensorineural-deafness"],
                    ["DIDMOAD-complete"], ["DM-non-immune", "optic-atrophy", "central-DI"]],
        "CISD2":   [["DM-non-immune", "optic-atrophy", "bleeding-tendency"],
                    ["peptic-ulcer-GI-bleeding"],
                    ["peripheral-neuropathy", "DM", "bleeding-tendency"],
                    ["optic-atrophy-early-onset", "bleeding"]],
        "PCSK1":   [["neonatal-malabsorption-secretory-diarrhea", "morbid-obesity"],
                    ["ACTH-deficiency", "hypogonadotropic-hypogonadism"],
                    ["morbid-obesity", "hyperproinsulinaemia", "central-DI"],
                    ["multihormone-deficiency", "neonatal-malabsorption"]],
        "KCNJ1":   [["polyhydramnios", "neonatal-hyperkalemia-transient"],
                    ["hypokalemia-alkalosis-hypercalciuria"],
                    ["neonatal-hyperkalemia-transient", "hypokalemia-alkalosis"],
                    ["Bartter-type-2-ROMK", "elevated-PGE2"]],
        "SLC12A1": [["polyhydramnios", "nephrocalcinosis"],
                    ["hypokalemia-alkalosis-hypercalciuria", "nephrocalcinosis-80pct"],
                    ["furosemide-like-phenotype", "premature-birth"],
                    ["Bartter-type-1-most-severe", "nephrocalcinosis"]],
    }
    features_options = associated_features_map.get(gene, [["polyuria-polydipsia"]])

    treatment_map = {
        "AVP":     ["desmopressin-intranasal", "desmopressin-oral", "desmopressin-sublingual"],
        "AVPR2":   ["hydrochlorothiazide+amiloride+low-solute-diet", "thiazide+amiloride", "low-solute-diet+thiazide"],
        "AQP2":    ["thiazide+amiloride", "thiazide+amiloride+low-solute-diet", "sildenafil-trial"],
        "WFS1":    ["desmopressin+insulin+multidisciplinary", "insulin+desmopressin+levothyroxine",
                    "multidisciplinary-endocrine-ophthal-audiol"],
        "CISD2":   ["insulin+PPI+ophthalmology", "insulin+ophthalmology+neuropathy-management",
                    "insulin+PPI+platelet-monitoring"],
        "PCSK1":   ["desmopressin+hydrocortisone+levothyroxine", "multihormone-replacement",
                    "desmopressin+insulin+cortisol+very-low-fat-diet"],
        "KCNJ1":   ["indomethacin+KCl-supplements", "IV-electrolytes-acute+indomethacin", "KCl+NaCl+indomethacin"],
        "SLC12A1": ["aggressive-IV-KCl-NaCl+indomethacin", "indomethacin+KCl+NaCl-supplements",
                    "IV-resuscitation+indomethacin+chronic-supplements"],
    }
    treatments = treatment_map.get(gene, ["desmopressin"])

    mutation_map = {
        "AVP":     ["p.Gly17Val", "p.Pro7Leu", "p.Ala19Thr", "p.Cys58Ser", "p.Gly65Arg"],
        "AVPR2":   ["p.Arg137His", "p.Arg137Cys", "p.Val88Met", "p.Phe105Ser", "p.Pro322His"],
        "AQP2":    ["p.Arg187Cys", "p.Ala70Thr", "p.Gly175Arg", "p.Pro262Leu", "p.Leu22Val"],
        "WFS1":    ["p.Ser769Leu", "p.Arg456His", "p.Pro724Leu", "p.Arg558Cys", "p.Gln672Ter"],
        "CISD2":   ["p.Trp45Ser", "p.Arg44Gln", "p.Trp45Arg", "p.Ala57Pro", "p.Gly93Ser"],
        "PCSK1":   ["p.Asn221Asp", "p.Gln665Glu", "p.Arg74Ter", "p.Gly209Asp", "p.Trp462Ter"],
        "KCNJ1":   ["p.Gln316Arg", "p.Lys80Asn", "p.Ala198Thr", "p.Arg324Gln", "p.Ser140Arg"],
        "SLC12A1": ["p.Arg302Gln", "p.Ala555Val", "p.Ala555Thr", "del_exon7-8", "p.Gly463Glu"],
    }
    mutations = mutation_map.get(gene, ["unknown"])

    polyuria_severity_map = ["mild", "moderate", "severe"]

    patients = []
    for i in range(40):
        desmo_resp = rng.random() < p["desmopresin_responsive_pct"]
        neonatal = rng.random() < p["neonatal_onset_pct"]
        urine_osmol = rng.randint(p["urine_osmol_min"], p["urine_osmol_max"])
        plasma_osmol = rng.randint(295, 320) if not desmo_resp or not neonatal else rng.randint(290, 320)
        # polyuria severity
        sev_rand = rng.random()
        if sev_rand < p["polyuria_severe_pct"]:
            polyuria_sev = "severe"
        elif sev_rand < p["polyuria_severe_pct"] + 0.20:
            polyuria_sev = "moderate"
        else:
            polyuria_sev = "mild"
        if gene == "CISD2":
            # DI rare in WFS2; many patients have no DI
            has_di = rng.random() < 0.15
            if not has_di:
                polyuria_sev = "none"
                urine_osmol = rng.randint(300, 600)
                plasma_osmol = rng.randint(285, 295)
        age_dx = rng.randint(0, 3) if neonatal else rng.randint(5, 45)
        treatment = rng.choice(treatments)
        mutation = rng.choice(mutations)
        features = rng.choice(features_options)
        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis": age_dx,
            "polyuria_severity": polyuria_sev,
            "urine_osmol_nadir_mOsm": urine_osmol,
            "plasma_osmol_mOsm": plasma_osmol,
            "desmopressin_responsive": desmo_resp,
            "neonatal_onset": neonatal,
            "associated_features": features,
            "treatment": treatment,
            "mutation": mutation,
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Diabetes-Insipidus-Atlas."""
    return {
        "atlas":          "Hereditary-Diabetes-Insipidus-Atlas",
        "subtitle":       "Complete 8-Gene DI Reference Atlas (AVP-AVPR2-AQP2-WFS1-CISD2-PCSK1-KCNJ1-SLC12A1)",
        "total_genes":    len(ATLAS_GENES),
        "seed_range":     f"{SEED_BASE}–{SEED_BASE + 7}",
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "gene_loci":      {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "AVP":     "AD LOF (ER-stress-neuronal-death) / AR LOF (congenital pure AVP synthesis failure)",
            "AVPR2":   "XLR (V2R LOF; males severely affected; females variable by X-inactivation)",
            "AQP2":    "AR LOF (biallelic, classic NDI2) / AD dominant-negative (C-terminal truncation)",
            "WFS1":    "AR LOF (Wolfram syndrome 1 DIDMOAD; DI+DM+optic atrophy+deafness)",
            "CISD2":   "AR LOF (Wolfram syndrome 2; NO DI mostly; bleeding tendency + DM + optic atrophy)",
            "PCSK1":   "AR LOF (PC1/3 deficiency; neonatal malabsorption FIRST; multi-hormone deficiency)",
            "KCNJ1":   "AR LOF (Bartter type 2; neonatal PARADOXICAL HYPERKALEMIA; polyhydramnios)",
            "SLC12A1": "AR LOF (Bartter type 1 MOST SEVERE; furosemide-like; nephrocalcinosis 80%)",
        },
        "key_clinical_rules": [
            "DESMOPRESSIN-RESPONSE-TEST: central DI (AVP, WFS1, PCSK1) = desmopressin RESPONSIVE (urine osmolality rises >50%); NDI (AVPR2, AQP2, KCNJ1, SLC12A1) = desmopressin UNRESPONSIVE",
            "AVP-MRI-BRIGHT-SPOT-ABSENT: posterior pituitary T1 bright spot absent = FNDI hallmark; normal bright spot reflects stored AVP peptide",
            "AVPR2-NDI1-XLR: infant male + polyuria + hypernatraemia = AVPR2 NDI1 until proven otherwise; desmopressin COMPLETELY UNRESPONSIVE; treat with thiazide + amiloride + low-solute diet",
            "WFS1-NONIMMUNE-DM: Wolfram DM = insulin-dependent but NO autoantibodies (GAD-65, IA-2, ZnT8); key distinguishing feature from type 1 DM; DIDMOAD sequence: optic atrophy → DM → DI → deafness",
            "CISD2-NO-DI-BLEEDING: WFS2 (CISD2) differs from WFS1 — NO DI in ~85%; ADD bleeding tendency (peptic ulcers + platelet hyperaggregability) NOT in WFS1; Israeli Arab founder p.Trp45Ser",
            "PCSK1-MALABSORPTION-FIRST: neonatal secretory diarrhea PATHOGNOMONIC and FIRST presentation of PCSK1 deficiency before obesity or DI develop; pro-insulin elevated (signature biomarker)",
            "KCNJ1-NEONATAL-HYPERKALEMIA: Bartter type 2 ONLY = paradoxical transient neonatal hyperkalemia; ROMK absent in collecting duct → K+ not secreted → initial hyperkalemia → resolves in weeks",
            "SLC12A1-FUROSEMIDE-LIKE: Bartter type 1 = NKCC2 LOF = chronic furosemide effect; MOST SEVERE; nephrocalcinosis 80%; NO neonatal hyperkalemia (unlike KCNJ1 Bartter 2)",
            "AQP2-URINE-AQP2-ABSENT: NDI2 diagnostic marker; absent urine AQP2 by ELISA; V2R intact (cAMP rises after DDAVP); defect post-receptor; emerging: sildenafil + statins improve AQP2 trafficking",
            "8-GENE-DIFFERENTIAL: central DI (desmopressin responsive) = AVP/WFS1/PCSK1; NDI (desmopressin unresponsive) = AVPR2/AQP2; Bartter polyuria (TAL dysfunction) = KCNJ1/SLC12A1; WFS2 (CISD2) mostly no DI",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for all 8 hereditary DI-spectrum genes."""
    genes_data = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        treatments = {}
        for pt in pts:
            treatments[pt["treatment"]] = treatments.get(pt["treatment"], 0) + 1
        mutations_seen = {}
        for pt in pts:
            mutations_seen[pt["mutation"]] = mutations_seen.get(pt["mutation"], 0) + 1
        di_pts = [pt for pt in pts if pt["polyuria_severity"] != "none"]
        mean_urine = (
            round(sum(pt["urine_osmol_nadir_mOsm"] for pt in di_pts) / len(di_pts), 1)
            if di_pts else None
        )
        desmo_resp_pct = round(100 * sum(1 for pt in pts if pt["desmopressin_responsive"]) / len(pts), 1)
        neonatal_pct = round(100 * sum(1 for pt in pts if pt["neonatal_onset"]) / len(pts), 1)
        mean_age_dx = round(sum(pt["age_at_diagnosis"] for pt in pts) / len(pts), 1)
        genes_data.append({
            "gene":                       g["gene"],
            "locus":                      g["locus"],
            "protein":                    g["protein"],
            "protein_size":               g["protein_size"],
            "inheritance":                g["inheritance"],
            "disease_category":           g["disease_category"],
            "n_patients":                 len(pts),
            "mean_urine_osmol_nadir":     mean_urine,
            "desmopressin_responsive_pct": desmo_resp_pct,
            "neonatal_onset_pct":         neonatal_pct,
            "mean_age_dx":                mean_age_dx,
            "treatment_breakdown":        treatments,
            "mutation_breakdown":         mutations_seen,
            "patients":                   pts,
        })
    return {
        "atlas": "Hereditary-Diabetes-Insipidus-Atlas",
        "count": len(genes_data),
        "genes": genes_data,
    }


def generate_definitions() -> dict:
    """Key clinical definitions for Hereditary-Diabetes-Insipidus-Atlas."""
    definitions = [
        {
            "term": "Desmopressin Response Test — Distinguishing Central DI from Nephrogenic DI",
            "genes": ["AVP", "WFS1", "PCSK1", "AVPR2", "AQP2"],
            "definition": (
                "DESMOPRESSIN RESPONSE TEST — GOLD STANDARD DISTINCTION: "
                "CENTRAL DI (AVP deficiency): "
                "  After water deprivation (plasma osmolality >295 mOsm/kg): "
                "    Desmopressin (2 μg IV or 20 μg intranasal) → urine osmolality RISES >50% above baseline; "
                "    Often >750 mOsm/kg (above plasma osmolality) — confirms intact renal V2R/AQP2 axis; "
                "  AVP LOF (FNDI): highly responsive; "
                "  WFS1: DI present in 70-75%; central; desmopressin effective; "
                "  PCSK1: DI in 50-60%; central (pro-AVP not cleaved); desmopressin effective; "
                "NEPHROGENIC DI: "
                "  AVPR2 (NDI1): V2R LOF → no cAMP → no AQP2 trafficking → urine osmolality DOES NOT RISE; "
                "    Desmopressin COMPLETELY UNRESPONSIVE; plasma AVP/copeptin elevated (appropriate response); "
                "  AQP2 (NDI2): V2R intact (cAMP rises normally after DDAVP) → AQP2 absent/cannot traffic; "
                "    Urine osmolality DOES NOT RISE despite normal cAMP; "
                "    Distinguishing AQP2 from AVPR2: cAMP rise + vWF rise after DDAVP = AVPR2 (V2R absent); "
                "    AQP2: cAMP rises but no urine concentration (post-receptor defect); "
                "COPEPTIN: "
                "  Copeptin (C-terminal pro-AVP) = surrogate marker for AVP; stable; easily measured; "
                "  Hypertonic saline stimulation test + copeptin: <4.9 pmol/L → central DI; "
                "  High copeptin (>21 pmol/L) → nephrogenic DI (appropriate high AVP response); "
                "BARTTER (KCNJ1/SLC12A1): not true DI; TAL dysfunction → impaired concentration; "
                "  Desmopressin NOT helpful (concentrating defect structural, not hormone); "
                "WATER DEPRIVATION TEST: standard but unsafe in severe unrecognised DI; "
                "  Copeptin-based tests now preferred in many centres."
            ),
        },
        {
            "term": "Wolfram Syndrome Sequence (DIDMOAD) — WFS1 vs WFS2 Key Differences",
            "genes": ["WFS1", "CISD2"],
            "definition": (
                "WOLFRAM SYNDROME 1 (WFS1) — DIDMOAD SEQUENCE: "
                "  D: Diabetes Insipidus (central; desmopressin effective) — mean onset age 14 years; "
                "  I: (Insulin-dependent) Diabetes Mellitus — mean onset age 6 years; "
                "  D: Optic Atrophy (bilateral progressive) — mean onset age 6 years (FIRST symptom); "
                "  A: (Sensorineural) deafness — mean onset age 16 years; "
                "  SEQUENCE: optic atrophy ≈ DM (age 6) → DI (age 14) → deafness (age 16); "
                "  DM NON-IMMUNE: NO GAD-65 / IA-2 / ZnT8 autoantibodies; insulin-dependent from ER-stress beta cell loss; "
                "    Critical: always check antibodies in 'T1DM' with optic atrophy — if absent → WFS1; "
                "  DI: central; posterior pituitary degeneration; desmopressin effective; "
                "  Additional WFS1 features: ataxia, neuropathy, neurogenic bladder, psychiatric illness, brainstem atrophy (late); "
                "  HET WFS1 variants: increased DM risk + psychiatric risk only (NOT Wolfram syndrome); "
                "WOLFRAM SYNDROME 2 (CISD2) — KEY DIFFERENCES: "
                "  SAME as WFS1: DM (non-immune) + optic atrophy (earlier, mean age 3-4 years); "
                "  DIFFERENT: "
                "    NO DI in ~85% of WFS2 patients; "
                "    ADD BLEEDING TENDENCY: peptic ulcers + upper GI bleeding + platelet hyperaggregability; "
                "    Peripheral neuropathy (more prominent than WFS1); "
                "    NO deafness (rare in WFS2); "
                "  WFS2 FOUNDER: p.Trp45Ser (c.134G>C) in Israeli Arab/Bedouin families; "
                "  GENETIC CONFIRMATION: biallelic WFS1 vs biallelic CISD2; "
                "    WFS1 panel test first (more common); CISD2 if Middle Eastern ancestry + bleeding + no DI."
            ),
        },
        {
            "term": "FNDI MRI Posterior Pituitary Bright Spot — Pathognomonic for AVP LOF",
            "genes": ["AVP"],
            "definition": (
                "POSTERIOR PITUITARY T1 BRIGHT SPOT — MRI HALLMARK: "
                "NORMAL: on T1-weighted MRI, posterior pituitary (neurohypophysis) is HYPERINTENSE (bright spot); "
                "  Signal source: phospholipid vesicles containing vasopressin (AVP) and neurophysin II; "
                "  Normally stores AVP before release; lipid-protein content creates T1 shortening; "
                "FNDI (AVP LOF): "
                "  Progressive magnocellular neuron death → no AVP stored → bright spot ABSENT; "
                "  Absent bright spot on T1 MRI = pathognomonic for central/hypothalamic DI; "
                "  May be absent before full clinical DI in early FNDI; "
                "  Coronal + sagittal T1 without contrast: protocol of choice; "
                "OTHER CAUSES OF ABSENT BRIGHT SPOT: "
                "  Craniopharyngioma, Langerhans cell histiocytosis, sarcoidosis, post-surgical, trauma; "
                "  Wolfram syndrome (WFS1): also absent bright spot (hypothalamic degeneration); "
                "  Primary polydipsia: bright spot PRESENT (AVP synthesis intact); "
                "ECTOPIC BRIGHT SPOT: "
                "  After posterior pituitary surgery/trauma: bright spot may appear along pituitary stalk; "
                "  Indicates AVP neurosecretory terminals relocated; "
                "BRIGHT SPOT ABSENT + NO MASS LESION: "
                "  Strong support for hereditary central DI (AVP, WFS1) or idiopathic; "
                "  Send AVP/WFS1/PCSK1 genetic testing; "
                "NOTE: in young patients with absent bright spot + T1DM-like DM without antibodies → WFS1 first."
            ),
        },
        {
            "term": "NDI vs Central DI — Diagnostic Algorithm: Copeptin, Desmopressin, Urine AQP2, Genetics",
            "genes": ["AVP", "AVPR2", "AQP2"],
            "definition": (
                "NDI vs CENTRAL DI — DIAGNOSTIC ALGORITHM: "
                "STEP 1 — CONFIRM DI: "
                "  Plasma osmolality >295 mOsm/kg + urine osmolality <300 mOsm/kg (or <600 with partial DI); "
                "  Exclude primary polydipsia (plasma osmolality <280 usually; copeptin normal); "
                "STEP 2 — COPEPTIN (or plasma AVP): "
                "  Low copeptin (<4.9 pmol/L after osmotic stimulation) → CENTRAL DI (AVP low/absent); "
                "    Causes: AVP LOF (FNDI), WFS1, PCSK1, post-surgical, idiopathic, Langerhans; "
                "  High copeptin (>21 pmol/L) → NEPHROGENIC DI (AVP high = appropriate response; kidney not responding); "
                "    Causes: AVPR2 (NDI1), AQP2 (NDI2), lithium nephropathy, hypercalcaemia; "
                "STEP 3 — DESMOPRESSIN TRIAL (after water deprivation or hypertonic saline): "
                "  Urine osmolality rises >50%: central DI confirmed (V2R + AQP2 intact); "
                "  No response: nephrogenic DI (V2R or AQP2 absent/non-functional); "
                "STEP 4 — DISTINGUISH AVPR2 vs AQP2 in NDI: "
                "  cAMP in urine after DDAVP: "
                "    No rise: AVPR2 (V2R absent → no cAMP); "
                "    Rise: AQP2 (V2R intact → cAMP rises → but AQP2 absent → no urine concentration); "
                "  von Willebrand factor rise after DDAVP: present in AQP2, absent in AVPR2; "
                "  Urine AQP2 (ELISA): absent → AQP2 mutation confirmed; "
                "STEP 5 — GENETICS: "
                "  Central DI: AVP sequencing; WFS1 if DM + optic atrophy; PCSK1 if malabsorption + obesity; "
                "  NDI: AVPR2 first (XLR; male patients); AQP2 (AR/AD); panel if complex; "
                "MRI PITUITARY: central DI → absent posterior bright spot; NDI → bright spot PRESENT."
            ),
        },
        {
            "term": "Bartter Syndrome Types 1 and 2 — TAL Dysfunction Polyuria, Neonatal Hyperkalemia, Nephrocalcinosis",
            "genes": ["KCNJ1", "SLC12A1"],
            "definition": (
                "BARTTER SYNDROME — TAL DYSFUNCTION POLYURIA: "
                "TYPE 1 (SLC12A1 / NKCC2 LOF) — MOST SEVERE: "
                "  NKCC2 = furosemide target; LOF = permanent furosemide effect; "
                "  Antenatal: severe polyhydramnios (often 20-24 weeks); premature birth <32 weeks; "
                "  Neonatal: severe electrolyte crisis; hyponatraemia + hypokalaemia + metabolic alkalosis; "
                "  NO neonatal hyperkalemia (ROMK/KCNJ1 intact in collecting duct → K+ secretion intact); "
                "  Nephrocalcinosis: 80% (bilateral medullary; calcium hypercalciuria + absent TAL Ca reabsorption); "
                "  Urine osmolality: 50-180 mOsm/kg (severely impaired concentration); "
                "TYPE 2 (KCNJ1 / ROMK LOF): "
                "  ROMK: K+ recycling channel in TAL apical membrane + K+ secretion in collecting duct; "
                "  Antenatal: polyhydramnios; premature birth 32-36 weeks (less severe than type 1); "
                "  PARADOXICAL TRANSIENT NEONATAL HYPERKALEMIA: "
                "    ROMK absent in collecting duct → K+ CANNOT be secreted → initial hyperkalemia; "
                "    Resolves over weeks as BK (big conductance K+) channels compensate; "
                "    PATHOGNOMONIC for Bartter type 2; ALL other Bartter types = neonatal HYPOKALEMIA; "
                "  Chronic: hypokalaemia + metabolic alkalosis + hypercalciuria (same as type 1); "
                "  Urine osmolality: 80-200 mOsm/kg; "
                "  Nephrocalcinosis: less frequent than type 1; "
                "TREATMENT (both types): "
                "  Acute: aggressive IV KCl + NaCl; "
                "  Chronic: indomethacin (COX inhibitor → PGE2 reduction → reduced polyuria); "
                "  KCl + NaCl supplementation; "
                "  Monitor renal function + nephrocalcinosis (annual USS); "
                "DIFFERENTIAL: "
                "  Furosemide abuse: biochemically identical to type 1 (urine Cl- high confirms ongoing); "
                "  Gitelman (SLC12A3): hypocalciuria + hypomagnesaemia (vs hypercalciuria in Bartter); "
                "  Primary aldosteronism: hypertension (vs low/normal BP in Bartter)."
            ),
        },
        {
            "term": "PCSK1 Deficiency — Neonatal Malabsorption First, Multi-Hormone Processing Defect",
            "genes": ["PCSK1"],
            "definition": (
                "PCSK1 (PC1/3) DEFICIENCY — MULTI-HORMONE PROCESSING DEFECT: "
                "MECHANISM: PC1/3 cleaves prohormones at paired basic residues in regulated secretory pathway; "
                "SUBSTRATES AFFECTED (all deficient in PCSK1 LOF): "
                "  Pro-AVP → AVP: central DI in ~55%; desmopressin effective; "
                "  Pro-POMC → ACTH: secondary adrenal insufficiency; pro-ACTH elevated; "
                "    Risk of adrenal crisis; mandatory hydrocortisone; "
                "  Pro-insulin → insulin: hyperproinsulinaemia (10-100x normal pro-insulin:insulin ratio); "
                "    SIGNATURE BIOMARKER: disproportionately elevated pro-insulin; "
                "  Pro-GLP-1 → GLP-1: glucose regulation impaired (incretin defect); "
                "  Pro-GnRH → GnRH: hypogonadotropic hypogonadism; pubertal delay; "
                "  Pro-TRH → TRH: central hypothyroidism; low TSH + low fT4; "
                "  Intestinal prohormones: pro-enteroglucagon, pro-neurotensin → malabsorption; "
                "NEONATAL MALABSORPTION — FIRST AND PATHOGNOMONIC: "
                "  Secretory diarrhea: osmotically resistant (continues with fasting); "
                "  Neonatal/infantile onset; often severe; misdiagnosed as IPEX or microvillus inclusion disease; "
                "  FIRST symptom before obesity or DI develop — key diagnostic clue; "
                "  If neonatal malabsorption + later obesity + multiple hormone deficiencies → PCSK1; "
                "MORBID OBESITY: "
                "  Profound hyperphagia; BMI often >40; early onset age 2-5 years; "
                "  Leptin resistance via impaired pro-POMC/NPY processing; "
                "  Setmelanotide (MC4R agonist): used for PCSK1-related obesity (rare case reports); "
                "TREATMENT: "
                "  Each deficient axis replaced: desmopressin + hydrocortisone + levothyroxine + sex steroids; "
                "  Very low fat diet (malabsorption); "
                "  Pro-insulin biomarker confirms treatment adequacy (should not need to fall — marker of processing)."
            ),
        },
        {
            "term": "8-Gene Hereditary DI Differential Guide — Central vs NDI vs Bartter vs Wolfram",
            "genes": ["AVP", "AVPR2", "AQP2", "WFS1", "CISD2", "PCSK1", "KCNJ1", "SLC12A1"],
            "definition": (
                "8-GENE HEREDITARY DI DIFFERENTIAL: "
                "BY DESMOPRESSIN RESPONSE: "
                "  RESPONSIVE (central DI — AVP deficiency): AVP, WFS1, PCSK1; "
                "  UNRESPONSIVE (nephrogenic DI — renal resistance): AVPR2, AQP2; "
                "  NOT APPLICABLE (Bartter polyuria — structural TAL defect): KCNJ1, SLC12A1; "
                "  NO DI in most (Wolfram 2): CISD2; "
                "BY INHERITANCE: "
                "  AD: AVP (AD LOF ER stress); "
                "  AR: AVP (AR biallelic), AQP2 (AR), WFS1, CISD2, PCSK1, KCNJ1, SLC12A1; "
                "  XLR: AVPR2 (males severely affected); "
                "  AD dominant-negative: AQP2 (C-terminal); "
                "BY ONSET: "
                "  Neonatal/antenatal: AVPR2 (hypernatraemia neonatal), AQP2 (neonatal/infantile), "
                "    KCNJ1 (Bartter 2, polyhydramnios), SLC12A1 (Bartter 1 most severe); "
                "  Childhood (age 2-15): AVP AD LOF (progressive); WFS1 (optic atrophy + DM age 6, DI age 14); "
                "  Any age: PCSK1 (neonatal malabsorption + obesity + hormone deficiency); CISD2; "
                "BY ASSOCIATED FEATURES: "
                "  Optic atrophy + non-immune DM → WFS1 or CISD2; "
                "  Bleeding tendency + DM + NO DI → CISD2 (WFS2); "
                "  Neonatal malabsorption + morbid obesity → PCSK1; "
                "  Polyhydramnios + neonatal hyperkalemia → KCNJ1 (Bartter 2); "
                "  Polyhydramnios + NO neonatal hyperkalemia + nephrocalcinosis 80% → SLC12A1 (Bartter 1); "
                "  Posterior pituitary bright spot absent (MRI) + progressive DI → AVP LOF (FNDI); "
                "  Infant male + hypernatraemia + desmopressin unresponsive → AVPR2; "
                "  V2R intact (cAMP rises) + desmopressin unresponsive + urine AQP2 absent → AQP2; "
                "COPEPTIN: "
                "  Low → central DI (AVP/WFS1/PCSK1); "
                "  High → NDI (AVPR2/AQP2) or Bartter (KCNJ1/SLC12A1)."
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Diabetes-Insipidus-Atlas",
        "count":       len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:800])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    for g in bd["genes"]:
        print(f"  {g['gene']}: n={g['n_patients']}, mean_urine_osmol={g['mean_urine_osmol_nadir']}, desmo_resp={g['desmopressin_responsive_pct']}%, neonatal={g['neonatal_onset_pct']}%, mean_age_dx={g['mean_age_dx']}")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Terms: {df['count']}")
