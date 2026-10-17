#!/usr/bin/env python3
"""Hereditary-Metabolic-Cardiomyopathy-Atlas -- Complete 8-Gene Reference
Hereditary metabolic/storage diseases causing cardiomyopathy
(Fabry/GLA, Danon/LAMP2, Pompe/GAA, PRKAG2, Barth/TAZ,
 Primary-Carnitine-Deficiency/SLC22A5, LCHAD/HADHA, Riboflavin-Responsive/ACAD9):

GLA    (Alpha-galactosidase A; 429aa; Xq22.1; X-linked;
         Fabry disease; lysosomal sphingolipid storage;
         Cardiomyopathy (HCM-like) + neuropathic pain + renal + angiokeratoma + corneal;
         Agalsidase alfa/beta; Migalastat (amenable mutations); enzyme replacement FDA-approved;
         seed SEED_BASE+0).
LAMP2  (Lysosome-associated membrane protein 2; 410aa; Xq24; X-linked;
         Danon disease; TRIAD: cardiomyopathy + skeletal myopathy + intellectual disability PATHOGNOMONIC;
         LAMP2-B cardiac isoform dominant; males severe/early; females later-onset;
         Cardiac transplant only definitive option; seed SEED_BASE+1).
GAA    (Acid alpha-glucosidase; 952aa; 17q25.3; AR;
         Pompe disease / GSD-II; glycogen storage in lysosomes;
         Infantile-onset: massive cardiomyopathy + hypotonia PATHOGNOMONIC;
         Late-onset: proximal myopathy dominant, cardiomyopathy mild;
         Enzyme replacement alglucosidase alfa / avalglucosidase alfa FDA-approved;
         seed SEED_BASE+2).
PRKAG2 (AMP-kinase gamma-2; 569aa; 7q36.1; AD;
         PRKAG2-cardiomyopathy; WPW + HCM + progressive-conduction-disease TRIAD PATHOGNOMONIC;
         Glycogen-filled vacuoles on biopsy; ICD insufficient if WPW (anti-tachycardia pacing triggers AF);
         Catheter ablation WPW + ICD separately; seed SEED_BASE+3).
TAZ    (Tafazzin; 292aa; Xq28; X-linked;
         Barth syndrome; cardiolipin remodeling deficiency;
         DCM + neutropenia + 3-methylglutaconic-aciduria type II TRIAD PATHOGNOMONIC;
         Males only (X-linked); LVNC variant; seed SEED_BASE+4).
SLC22A5 (Organic cation/carnitine transporter 2; 557aa; 5q31.1; AR;
         Primary carnitine deficiency (CDSP/OCTN2-deficiency);
         DCM + hypoketotic-hypoglycaemic-encephalopathy PATHOGNOMONIC combination;
         L-carnitine supplementation CURATIVE for cardiac disease;
         NBS detects low free carnitine; seed SEED_BASE+5).
HADHA  (Mitochondrial trifunctional protein alpha; 763aa; 2p23.3; AR;
         LCHAD/MTP deficiency; LCFA beta-oxidation;
         DCM + peripheral-neuropathy + pigmentary-retinopathy TRIAD;
         Heterozygous mother of affected child: AFLP (acute fatty liver of pregnancy) PATHOGNOMONIC;
         Low-fat MCT-enriched diet + avoid fasting;
         seed SEED_BASE+6).
ACAD9  (Acyl-CoA dehydrogenase family member 9; 621aa; 3q21.3; AR;
         Complex I assembly factor; mitochondrial cardiomyopathy;
         Riboflavin-responsive: high-dose riboflavin trial FIRST-LINE ALWAYS;
         Lactic acidosis + cardiomyopathy; seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3790-3797)
"""
import random

SEED_BASE = 3790

ATLAS_GENES = [
    {
        "gene": "GLA",
        "protein": (
            "GLA -- Xq22.1 X-linked lysosomal GOF/LOF -- 429aa -- "
            "Alpha-galactosidase-A-lysosomal-glycohydrolase-sphingolipid-Gb3-GL3-"
            "Fabry-disease-OMIM-301500-angiokeratoma-neuropathy-renal-cardiac-corneal-"
            "Agalsidase-alfa-beta-ERT-FDA-Migalastat-chaperon-FDA2018"
        ),
        "locus": "Xq22.1",
        "protein_size": (
            "429 aa / ~46 kDa / Xq22.1 GLA alpha-galactosidase A molecular context: "
            "STRUCTURE: "
            "  429 aa; homodimer lysosomal hydrolase; cleaves terminal alpha-galactosyl residues; "
            "  Deficiency -> globotriaosylceramide (Gb3/GL3) accumulation in lysosomes; "
            "  Lyso-Gb3 (deacylated form) measurable in plasma -- biomarker and disease monitor; "
            "  >900 variants identified; missense most common; amenable-mutation list for migalastat; "
            "FABRY DISEASE -- MULTISYSTEM: "
            "  Males (hemizygous): classic full phenotype; "
            "  Females (heterozygous): variable, often mild, can be as severe as males; "
            "  CARDIAC (most common death cause in treated patients): "
            "    HCM-like concentric hypertrophy; LVH asymmetric septum; "
            "    T1 mapping CMR: LOW T1 signal (Gb3 glycolipid storage) PATHOGNOMONIC -- "
            "      opposite direction from fibrosis (high T1) and amyloidosis (low T1 similar to Fabry); "
            "    Characteristic late enhancement BASAL-INFEROLATERAL + MIDWALL = PATHOGNOMONIC Fabry; "
            "    Short PR interval (enhanced AV conduction early); "
            "    Progressive fibrosis -> HFpEF, AF, sudden death; "
            "  RENAL: "
            "    Podocytopathy -> proteinuria -> ESRD (males, median 40yr without ERT); "
            "    Gb3 inclusions in renal biopsy (electron microscopy 'zebra bodies' PATHOGNOMONIC); "
            "  NEUROPATHIC PAIN: "
            "    Small-fiber neuropathy; acroparesthesiae (burning hands/feet) in childhood; "
            "    Fabry crises (febrile pain attacks) in childhood; "
            "    Reduced cold/heat/pain threshold; "
            "  CORNEA VERTICILLATA: "
            "    Whorl-like corneal opacities slit-lamp PATHOGNOMONIC (seen in ~80% heterozygotes); "
            "    Does not affect vision; does NOT respond to ERT; "
            "  ANGIOKERATOMA: "
            "    Dark red papules bathing-trunk distribution PATHOGNOMONIC; "
            "    Onset childhood-adolescence; "
            "  CEREBROVASCULAR: "
            "    Early stroke; posterior circulation white matter; dolichoectasia basilar; "
            "DIAGNOSIS: "
            "  Alpha-galactosidase A enzyme activity (males: low diagnostic; females: may be normal!); "
            "  GLA gene sequencing -- MANDATORY in females (enzyme assay unreliable); "
            "  Plasma Lyso-Gb3: elevated in classic, correlates with disease burden; "
            "  Urine sediment/renal biopsy: zebra-body inclusions EM; "
            "TREATMENT: "
            "  Agalsidase alfa (0.2mg/kg biweekly, EMA); agalsidase beta (1.0mg/kg biweekly, FDA/EMA); "
            "  Migalastat (123mg EOD oral): pharmacological chaperone; ONLY amenable variants; "
            "  Pegunigalsidase alfa (1mg/kg monthly): stabilized PEGylated ERT; FDA 2023; "
            "  NEPHROPROTECTION: ACEi/ARB for proteinuria; "
            "  CARDIAC: anticoagulation for AF; ICD for VT; septal myectomy rare"
        ),
        "syndrome": (
            "Fabry-disease / X-linked-lysosomal-storage / "
            "HCM-like-cardiomyopathy / angiokeratoma-bathing-trunk-PATHOGNOMONIC / "
            "cornea-verticillata-PATHOGNOMONIC / acroparesthesiae-childhood"
        ),
        "inheritance": "X-linked (hemizygous males / heterozygous females variable)",
        "key_therapy": "Agalsidase alfa/beta ERT + Migalastat (amenable mutations) + Pegunigalsidase alfa FDA2023",
        "pathognomonic": (
            "CMR: low T1 signal (Gb3 storage) + basal-inferolateral-midwall LGE = PATHOGNOMONIC Fabry cardiac; "
            "Cornea verticillata on slit-lamp PATHOGNOMONIC; "
            "Zebra bodies (EM) in renal biopsy PATHOGNOMONIC; "
            "Bathing-trunk angiokeratoma PATHOGNOMONIC"
        ),
        "key_drugs_ci": [
            "Agalsidase-alfa-ERT-0.2mgkg-biweekly-FDA2003",
            "Agalsidase-beta-ERT-1.0mgkg-biweekly-FDA2003",
            "Migalastat-amenable-mutations-only-FDA2018",
            "Pegunigalsidase-alfa-FDA2023",
            "ACEi-ARB-nephroprotection-mandatory",
            "CHLOROQUINE-HYDROXYCHLOROQUINE-AVOID-inhibit-lysosomal-enzyme-activity",
        ],
    },
    {
        "gene": "LAMP2",
        "protein": (
            "LAMP2 -- Xq24 X-linked lysosomal structural -- 410aa -- "
            "Lysosome-associated-membrane-protein-2-glycoprotein-lysosomal-membrane-"
            "Danon-disease-OMIM-300257-HCM-DCM-skeletal-myopathy-intellectual-disability-"
            "LAMP2-A-autophagy-LAMP2-B-cardiac-isoform-heart-transplant-only-option"
        ),
        "locus": "Xq24",
        "protein_size": (
            "410 aa / ~45 kDa (core) / ~110 kDa glycosylated / Xq24 LAMP2 molecular context: "
            "STRUCTURE: "
            "  410 aa; type I transmembrane glycoprotein; heavily glycosylated (~75% carbohydrate); "
            "  Three isoforms (LAMP2-A, LAMP2-B, LAMP2-C) by alternative C-terminal exon splicing; "
            "  LAMP2-B: cardiac and skeletal muscle dominant isoform; "
            "  Function: lysosomal membrane stability + macroautophagy + chaperone-mediated autophagy; "
            "  Deficiency -> autophagosome accumulation + glycogen accumulation in lysosomes; "
            "DANON DISEASE -- CLASSIC TRIAD: "
            "  1. CARDIOMYOPATHY (HCM most common, DCM, LVNC): SEVERE, rapidly progressive; "
            "     Males: onset adolescence-early adult; massive HCM (mean wall thickness 30-40mm); "
            "     Females: onset 2nd-3rd decade; later than males, often DCM form; "
            "  2. SKELETAL MYOPATHY: proximal muscle weakness; CPK elevated; "
            "  3. INTELLECTUAL DISABILITY: present in ~70% males; often absent in females; "
            "  ALL THREE TOGETHER = PATHOGNOMONIC DANON; "
            "  WPW (pre-excitation) in ~30-50% = additional clue; "
            "PROGNOSIS: "
            "  Males: severe; death/transplant mean age 19-25yr without transplant; "
            "  Females: milder; cardiac events mean 40s; "
            "DIAGNOSIS: "
            "  Cardiac MRI: massive LVH + LGE + autophagosome-pattern; "
            "  Muscle biopsy: vacuolar myopathy + autophagic vacuoles + PAS+ glycogen (electron microscopy); "
            "  LAMP2 gene sequencing: MANDATORY (enzyme assay not available); "
            "  LAMP2 protein immunostaining: absent on skeletal muscle biopsy; "
            "TREATMENT: "
            "  Cardiac transplant: ONLY definitive therapy; listed early; "
            "  Enzyme replacement/gene therapy: in clinical trials; "
            "  Antiarrhythmic + ICD: bridge to transplant; "
            "  WPW: catheter ablation or antiarrhythmic"
        ),
        "syndrome": (
            "Danon-disease / X-linked-lysosomal-cardiomyopathy / "
            "HCM-DCM-massive / skeletal-myopathy / intellectual-disability / "
            "TRIAD-PATHOGNOMONIC / WPW-30-50pct / heart-transplant-only"
        ),
        "inheritance": "X-linked (males severe early / females later milder)",
        "key_therapy": "Cardiac transplant (only definitive) — ICD/ablation bridge — gene therapy trials",
        "pathognomonic": (
            "DANON TRIAD: massive cardiomyopathy + skeletal myopathy + intellectual disability = PATHOGNOMONIC; "
            "Absent LAMP2 protein on muscle immunostaining PATHOGNOMONIC; "
            "Autophagic vacuoles on muscle EM PATHOGNOMONIC"
        ),
        "key_drugs_ci": [
            "Cardiac-transplant-ONLY-definitive-option",
            "ICD-bridge-to-transplant-males",
            "WPW-catheter-ablation-MANDATORY-before-antiarrhythmics",
            "AMIODARONE-avoid-if-WPW-pre-excitation-risk-AF-conduction",
        ],
    },
    {
        "gene": "GAA",
        "protein": (
            "GAA -- 17q25.3 AR lysosomal acid-glucosidase LOF -- 952aa -- "
            "Acid-alpha-glucosidase-lysosomal-glycogen-hydrolysis-"
            "Pompe-disease-GSD-II-OMIM-232300-infantile-cardiomyopathy-hypotonia-"
            "late-onset-proximal-myopathy-alglucosidase-alfa-avalglucosidase-alfa-ERT-FDA"
        ),
        "locus": "17q25.3",
        "protein_size": (
            "952 aa / ~110 kDa / 17q25.3 GAA acid alpha-glucosidase molecular context: "
            "STRUCTURE: "
            "  952 aa; lysosomal acid alpha-1,4-glucosidase; hydrolyzes lysosomal glycogen; "
            "  Deficiency -> glycogen accumulates in lysosomes of all tissues (muscle-dominant); "
            "  AR inheritance; two GAA mutations required; "
            "  Pseudodeficiency alleles (c.-32-13T>G splice, c.1726G>A) -- reduce enzyme but NO disease; "
            "INFANTILE-ONSET POMPE (IOPD): "
            "  Onset: first weeks-months of life; "
            "  MASSIVE cardiomyopathy (cardiomegaly on CXR) + profound hypotonia PATHOGNOMONIC combination; "
            "  'Floppy baby' with macroglossia; "
            "  ECG: SHORT PR + HIGH VOLTAGE QRS + LEFT AXIS DEVIATION PATHOGNOMONIC triad; "
            "  Death from cardiorespiratory failure <1yr without ERT; "
            "  NBS (newborn screening): now standard in many countries -- dried blood spot; "
            "LATE-ONSET POMPE (LOPD): "
            "  Onset: from 2yr to >70yr; "
            "  Proximal myopathy dominant (limb-girdle pattern + respiratory); "
            "  Cardiomyopathy RARE in late-onset (distinguishes from IOPD); "
            "  Respiratory failure (diaphragm weakness): primary mortality cause; "
            "  c.-32-13T>G (IVS1) + severe allele = most common LOPD compound heterozygote; "
            "TREATMENT: "
            "  Alglucosidase alfa (Myozyme/Lumizyme): standard ERT; FDA 2006; "
            "  Avalglucosidase alfa (Nexviazyme): enhanced uptake (M6P receptor); FDA 2021; "
            "  Cipaglucosidase alfa + miglustat (Pombiliti + Opfolda): combination ERT+inhibitor; FDA 2023; "
            "  MIGLUSTAT ADJUNCT: reduces glycogen synthesis; improves ERT efficacy; "
            "  Respiratory support: NIV crucial; ventilator planning early; "
            "  Physiotherapy: essential for motor maintenance; "
            "  Alglucosidase-responsive: cross-reactive immunological material (CRIM) status important for IOPD"
        ),
        "syndrome": (
            "Pompe-disease / GSD-II / acid-maltase-deficiency / "
            "IOPD: massive-cardiomegaly+hypotonia-PATHOGNOMONIC / "
            "LOPD: proximal-myopathy+respiratory-dominant / "
            "ECG-short-PR-high-voltage-left-axis-PATHOGNOMONIC"
        ),
        "inheritance": "AR (biallelic GAA mutations)",
        "key_therapy": "Alglucosidase alfa / Avalglucosidase alfa ERT + respiratory support + miglustat adjunct",
        "pathognomonic": (
            "IOPD: massive cardiomegaly + hypotonia PATHOGNOMONIC; "
            "ECG triad: short PR + high voltage + left axis deviation = PATHOGNOMONIC IOPD; "
            "PAS+ glycogen in lysosomes on biopsy PATHOGNOMONIC"
        ),
        "key_drugs_ci": [
            "Alglucosidase-alfa-ERT-FDA2006",
            "Avalglucosidase-alfa-ERT-FDA2021-enhanced-M6P",
            "Cipaglucosidase-alfa-miglustat-FDA2023",
            "Respiratory-NIV-early-MANDATORY-LOPD",
            "CRIM-status-check-IOPD-before-ERT-start",
        ],
    },
    {
        "gene": "PRKAG2",
        "protein": (
            "PRKAG2 -- 7q36.1 AD glycogen-storage GOF -- 569aa -- "
            "AMP-activated-protein-kinase-gamma-2-regulatory-subunit-glycogen-sensor-"
            "PRKAG2-cardiomyopathy-syndrome-OMIM-194200-WPW-HCM-conduction-disease-"
            "glycogen-filled-vacuoles-ICD-WPW-catheter-ablation-mandatory-separately"
        ),
        "locus": "7q36.1",
        "protein_size": (
            "569 aa / ~63 kDa / 7q36.1 PRKAG2 AMP kinase gamma-2 molecular context: "
            "STRUCTURE: "
            "  569 aa; gamma (regulatory) subunit of AMP-activated protein kinase (AMPK); "
            "  Contains four cystathionine-beta-synthase (CBS) domains = adenine nucleotide binding; "
            "  Missense variants mimic AMP-bound state -> AMPK constitutively active; "
            "  -> excessive glycogen synthesis + storage in cardiomyocytes (lysosomal accumulation); "
            "  AD inheritance; variable penetrance; "
            "  Variants: R302Q, N488I, T400N most-studied; "
            "PRKAG2-CARDIOMYOPATHY TRIAD (ALL THREE PATHOGNOMONIC): "
            "  1. WPW (Wolff-Parkinson-White): accessory pathway delta wave; "
            "     RIGHT-SIDED or left-sided accessory pathway; "
            "  2. HCM (hypertrophic cardiomyopathy): concentric or septal; "
            "  3. PROGRESSIVE CONDUCTION DISEASE: sinus node dysfunction -> complete heart block; "
            "  KEY TRAP: ICD alone is INSUFFICIENT and DANGEROUS in WPW: "
            "     Anti-tachycardia pacing can trigger rapid conduction over AP -> VF; "
            "     CATHETER ABLATION of accessory pathway FIRST, THEN reassess ICD need; "
            "BIOPSY: "
            "  Vacuolated cardiomyocytes; glycogen-filled vacuoles PAS+; "
            "  Vacuoles contain lysosomal membrane markers (LAMP2 positive); "
            "  Electron microscopy: glycogen granules within membrane-bound structures; "
            "MANAGEMENT: "
            "  Catheter ablation: WPW -- BEFORE antiarrhythmics/ICD; "
            "  Pacemaker: sinus node dysfunction / heart block; "
            "  ICD: after WPW ablation if residual VT risk; "
            "  Avoid: digoxin (enhances AV nodal conduction paradoxically dangerous with AP); "
            "  Avoid: verapamil/diltiazem (block AV node, rapid AP conduction -> VF); "
            "  Heart transplant: end-stage cardiomyopathy"
        ),
        "syndrome": (
            "PRKAG2-cardiomyopathy / AMPK-glycogen-storage / "
            "WPW+HCM+conduction-disease-TRIAD-PATHOGNOMONIC / "
            "ICD-insufficient-alone-WPW / catheter-ablation-FIRST"
        ),
        "inheritance": "AD (variable penetrance)",
        "key_therapy": "Catheter ablation WPW (FIRST before ICD) + pacemaker (conduction) + ICD after ablation",
        "pathognomonic": (
            "PRKAG2 TRIAD: WPW + HCM + progressive conduction disease = PATHOGNOMONIC; "
            "Glycogen-filled vacuoles (PAS+) in cardiomyocytes PATHOGNOMONIC; "
            "ICD anti-tachycardia pacing with intact WPW = VF RISK"
        ),
        "key_drugs_ci": [
            "Catheter-ablation-WPW-BEFORE-ICD-MANDATORY",
            "DIGOXIN-AVOID-WPW-rapid-accessory-conduction",
            "VERAPAMIL-DILTIAZEM-AVOID-WPW-rapid-AP-conduction-risk",
            "ICD-alone-INSUFFICIENT-active-WPW-antitachycardia-pacing-risk",
            "Pacemaker-sinus-node-dysfunction-PRKAG2",
        ],
    },
    {
        "gene": "TAZ",
        "protein": (
            "TAZ -- Xq28 X-linked mitochondrial LOF -- 292aa -- "
            "Tafazzin-phospholipid-transacylase-cardiolipin-remodeling-"
            "Barth-syndrome-OMIM-302060-DCM-neutropenia-3-methylglutaconic-aciduria-type-II-"
            "LVNC-males-only-cardiolipin-deficiency-mitochondrial-dysfunction"
        ),
        "locus": "Xq28",
        "protein_size": (
            "292 aa / ~33 kDa / Xq28 TAZ tafazzin molecular context: "
            "STRUCTURE: "
            "  292 aa; mitochondrial inner membrane phospholipid transacylase; "
            "  Remodels cardiolipin (CL): tetra-linoleoyl-CL (L4-CL) -- heart-specific form; "
            "  Deficiency -> abnormal CL -> mitochondrial dysfunction -> energy failure; "
            "  X-linked; virtually exclusively males; females asymptomatic carriers; "
            "  >200 variants; frameshift/nonsense most severe; missense variable; "
            "BARTH SYNDROME TRIAD (PATHOGNOMONIC): "
            "  1. DILATED CARDIOMYOPATHY (DCM) or LVNC (left ventricular non-compaction): "
            "     Onset neonatal / first years; severe; arrhythmias; "
            "     Can fluctuate -- CMR shows LVNC-trabeculations; "
            "  2. SKELETAL MYOPATHY: proximal weakness; fatigue; exercise intolerance; "
            "  3. NEUTROPENIA (cyclic or persistent): recurrent bacterial infections; "
            "     ANC typically 500-1500 cells/uL with cycling; "
            "  BONUS: 3-METHYLGLUTACONIC ACIDURIA TYPE II -- urine organic acids PATHOGNOMONIC marker; "
            "     3-methylglutaric acid + 3-methylglutaconic acid + 2-ethylhydracrylic acid; "
            "  Growth retardation; hyperlipidaemia (triacylglycerols); "
            "DIAGNOSIS: "
            "  Urine 3-methylglutaconic acid: elevated PATHOGNOMONIC in Barth; "
            "  Cardiolipin analysis (blood/muscle/fibroblasts): monolyso-CL elevated, L4-CL reduced; "
            "  TAZ gene sequencing (MANDATORY -- phenotype does not diagnose gene); "
            "  Echocardiogram/CMR: DCM or LVNC; "
            "  CBC: cyclic neutropenia; "
            "TREATMENT: "
            "  Cardiac: ACEi, beta-blocker, diuretics; ICD; transplant if refractory; "
            "  G-CSF (filgrastim): neutropenia if recurrent infections; "
            "  Elamipretide (SS-31): mitochondrial cardiolipin peptide; clinical trials; "
            "  Bezafibrate: experimental; "
            "  GENE THERAPY: early trials"
        ),
        "syndrome": (
            "Barth-syndrome / X-linked-mitochondrial / "
            "DCM-LVNC + neutropenia + 3-methylglutaconic-aciduria-TRIAD-PATHOGNOMONIC / "
            "males-only / cardiolipin-deficiency"
        ),
        "inheritance": "X-linked (males affected, females asymptomatic carriers)",
        "key_therapy": "Cardiac ACEi/BB/diuretics + G-CSF (neutropenia) + ICD + transplant if refractory",
        "pathognomonic": (
            "BARTH TRIAD: DCM/LVNC + neutropenia + 3-methylglutaconic aciduria type II = PATHOGNOMONIC; "
            "Urine 3-methylglutaconic acid elevated PATHOGNOMONIC Barth; "
            "Monolyso-cardiolipin elevated on CL analysis PATHOGNOMONIC"
        ),
        "key_drugs_ci": [
            "G-CSF-filgrastim-neutropenia-recurrent-infections",
            "Elamipretide-SS-31-clinical-trials",
            "ICD-arrhythmia-DCM",
            "Cardiac-transplant-refractory-DCM",
        ],
    },
    {
        "gene": "SLC22A5",
        "protein": (
            "SLC22A5 -- 5q31.1 AR carnitine-transporter LOF -- 557aa -- "
            "OCTN2-organic-cation-carnitine-transporter-2-plasma-membrane-"
            "Primary-carnitine-deficiency-CDSP-OMIM-212140-DCM-myopathy-hypoketotic-hypoglycemia-"
            "L-carnitine-supplementation-CURATIVE-NBS-free-carnitine-low"
        ),
        "locus": "5q31.1",
        "protein_size": (
            "557 aa / ~63 kDa / 5q31.1 SLC22A5 OCTN2 molecular context: "
            "STRUCTURE: "
            "  557 aa; organic cation/carnitine transporter 2 (OCTN2); "
            "  High-affinity sodium-dependent carnitine transporter; "
            "  Expressed: intestine (absorption), kidney (reabsorption), heart, muscle; "
            "  Deficiency -> carnitine urinary wasting -> low plasma carnitine -> "
            "  mitochondrial fatty-acid beta-oxidation failure -> energy crisis; "
            "  AR; biallelic loss-of-function variants; "
            "PRIMARY CARNITINE DEFICIENCY (CDSP): "
            "  CARDIAC PRESENTATION (most important): "
            "    Dilated cardiomyopathy: onset 1-7yr; severe; rapidly progressive; "
            "    Skeletal myopathy: proximal weakness; hypotonia; "
            "    HYPOKETOTIC HYPOGLYCAEMIC ENCEPHALOPATHY: "
            "      Fasting-induced; low glucose + absent ketones PATHOGNOMONIC combination; "
            "      Mimics Reye syndrome; "
            "  METABOLIC PRESENTATION: "
            "    Fasting intolerance; hypoketotic hypoglycemia; metabolic crises; "
            "  MATERNAL CDSP: "
            "    Mother homozygous/compound-het: may be asymptomatic; "
            "    NBS of infant finds low carnitine -- index case may be the MOTHER; "
            "NBS: "
            "  Free carnitine C0: severely reduced (<5 uM, normal >20); PATHOGNOMONIC; "
            "  Acylcarnitine profile: C0 low; C16/C18 acylcarnitines may also be elevated; "
            "TREATMENT: "
            "  L-carnitine supplementation: 50-100mg/kg/day oral; "
            "  DCM REVERSES with treatment -- CURATIVE for cardiac disease if started early; "
            "  Avoid prolonged fasting; high-carb diet during illness; "
            "  Lifelong supplementation required (renal wasting continues)"
        ),
        "syndrome": (
            "Primary-carnitine-deficiency / CDSP / OCTN2-deficiency / "
            "DCM-myopathy + hypoketotic-hypoglycemia-PATHOGNOMONIC / "
            "L-carnitine-CURATIVE-cardiac / NBS-low-free-carnitine"
        ),
        "inheritance": "AR (biallelic SLC22A5 mutations)",
        "key_therapy": "L-carnitine 50-100mg/kg/day oral — CURATIVE for cardiac disease — lifelong",
        "pathognomonic": (
            "NBS: plasma free carnitine C0 severely reduced (<5 uM) PATHOGNOMONIC; "
            "Hypoketotic hypoglycemia (low glucose + absent ketones) PATHOGNOMONIC; "
            "DCM reversal with L-carnitine PATHOGNOMONIC confirmation"
        ),
        "key_drugs_ci": [
            "L-carnitine-50-100mgkg-day-CURATIVE-cardiac",
            "PIVALIC-ACID-ANTIBIOTICS-AVOID-pivmecillinam-pivampicillin-deplete-carnitine",
            "VALPROATE-AVOID-carnitine-depletion-exacerbates",
            "Fasting-AVOID-strict-especially-illness",
        ],
    },
    {
        "gene": "HADHA",
        "protein": (
            "HADHA -- 2p23.3 AR mitochondrial-MTP LOF -- 763aa -- "
            "Mitochondrial-trifunctional-protein-alpha-subunit-LCHAD-enoyl-CoA-hydratase-"
            "LCHAD-deficiency-MTP-deficiency-OMIM-609015-DCM-neuropathy-retinopathy-"
            "AFLP-heterozygous-mother-PATHOGNOMONIC-MCT-diet-avoid-fasting"
        ),
        "locus": "2p23.3",
        "protein_size": (
            "763 aa / ~83 kDa / 2p23.3 HADHA MTP alpha molecular context: "
            "STRUCTURE: "
            "  763 aa; alpha subunit of mitochondrial trifunctional protein (MTP); "
            "  Contains: long-chain enoyl-CoA hydratase + long-chain 3-hydroxyacyl-CoA "
            "    dehydrogenase (LCHAD) activities; "
            "  MTP = heterooctamer (4 alpha HADHA + 4 beta HADHB); "
            "  HADHB: long-chain thiolase activity; "
            "  Catalyses three steps of mitochondrial long-chain fatty acid beta-oxidation; "
            "  c.1528G>C (G510A, E474Q) = common Caucasian LCHAD-specific mutation (~87% alleles); "
            "LCHAD/MTP DEFICIENCY: "
            "  CARDIAC: "
            "    Dilated or hypertrophic cardiomyopathy; arrhythmias; "
            "    DCM/HCM more prominent in pure LCHAD than MTP-deficiency; "
            "  NEUROPATHIC COMPLICATIONS: "
            "    Peripheral motor-sensory neuropathy: progressive; "
            "    CMT-like pattern; loss of deep tendon reflexes; "
            "  PIGMENTARY RETINOPATHY: "
            "    LCHAD-specific: chorioretinal pigment epithelial changes; "
            "    Progressive visual loss; annual ophthalmology MANDATORY; "
            "  METABOLIC CRISES: "
            "    Fasting-induced hypoketotic hypoglycemia + rhabdomyolysis + hepatitis; "
            "MATERNAL COMPLICATION (PATHOGNOMONIC ASSOCIATION): "
            "  Heterozygous mother carrying LCHAD-affected fetus: "
            "  AFLP (acute fatty liver of pregnancy): 3rd trimester; nausea/vomiting/jaundice; "
            "  HELLP syndrome (haemolysis + elevated liver enzymes + low platelets); "
            "  THIS ASSOCIATION IS PATHOGNOMONIC FOR LCHAD: "
            "    A mother who had AFLP/HELLP = TEST THE BABY FOR LCHAD IMMEDIATELY; "
            "    A baby with LCHAD = SCREEN THE MOTHER retrospectively; "
            "TREATMENT: "
            "  Low-fat diet (10-15% total calories from fat, LCT <10%); "
            "  MCT (medium-chain triglyceride) supplementation: bypasses LCHAD; "
            "  Avoid fasting; cornstarch / frequent feeds; "
            "  DHA (docosahexaenoic acid) supplementation: retinopathy may slow; "
            "  Annual eye exam; EMG monitoring for neuropathy"
        ),
        "syndrome": (
            "LCHAD-deficiency / MTP-deficiency / HADHA-biallelic / "
            "DCM + peripheral-neuropathy + pigmentary-retinopathy-TRIAD / "
            "AFLP-HELLP-in-heterozygous-mother-PATHOGNOMONIC / MCT-diet"
        ),
        "inheritance": "AR (biallelic HADHA mutations)",
        "key_therapy": "Low-fat MCT-supplemented diet + avoid fasting + DHA + annual ophthalmology",
        "pathognomonic": (
            "LCHAD TRIAD: cardiomyopathy + peripheral neuropathy + pigmentary retinopathy PATHOGNOMONIC; "
            "Mother with AFLP/HELLP in pregnancy = TEST BABY FOR LCHAD immediately PATHOGNOMONIC association; "
            "Elevated C16-OH/C18-OH acylcarnitines on NBS PATHOGNOMONIC"
        ),
        "key_drugs_ci": [
            "MCT-diet-MANDATORY-bypass-LCHAD",
            "VALPROATE-AVOID-impairs-fatty-acid-oxidation",
            "PROPOFOL-AVOID-impairs-mitochondrial-function",
            "DHA-supplementation-retinopathy-prevention",
            "Fasting-AVOID-especially-illness-intercurrent",
        ],
    },
    {
        "gene": "ACAD9",
        "protein": (
            "ACAD9 -- 3q21.3 AR mitochondrial-CI-assembly LOF -- 621aa -- "
            "Acyl-CoA-dehydrogenase-family-member-9-complex-I-assembly-factor-"
            "mitochondrial-respiratory-chain-complex-I-deficiency-OMIM-611126-"
            "riboflavin-responsive-cardiomyopathy-lactic-acidosis-riboflavin-first-line-ALWAYS"
        ),
        "locus": "3q21.3",
        "protein_size": (
            "621 aa / ~68 kDa / 3q21.3 ACAD9 complex I assembly molecular context: "
            "STRUCTURE: "
            "  621 aa; member of acyl-CoA dehydrogenase (ACAD) family; "
            "  Despite family membership: primary function is Complex I (NADH:ubiquinone oxidoreductase) "
            "    assembly, NOT fatty-acid oxidation (unlike other ACADs); "
            "  Essential for early steps of mitochondrial Complex I assembly; "
            "  Deficiency -> reduced Complex I activity -> mitochondrial energy failure; "
            "  AR; typically missense variants; "
            "ACAD9-COMPLEX I DEFICIENCY: "
            "  CARDIAC: "
            "    Hypertrophic or dilated cardiomyopathy: prominent, often presenting feature; "
            "    Early onset (neonatal to infantile); "
            "    Rapidly progressive without riboflavin treatment; "
            "  METABOLIC: "
            "    Lactic acidosis (elevated plasma lactate): hallmark; "
            "    Exercise intolerance; "
            "    Encephalopathy in severe cases; "
            "  MUSCLE: "
            "    Skeletal myopathy; weakness; hypotonia; "
            "RIBOFLAVIN RESPONSIVENESS (CRITICAL THERAPEUTIC INSIGHT): "
            "  ACAD9 is the most riboflavin-responsive Complex I gene; "
            "  Riboflavin (vitamin B2) acts as FAD cofactor -> stabilizes ACAD9 protein; "
            "  Clinical response: "
            "    Cardiomyopathy IMPROVES or resolves; lactate normalizes; "
            "    Some patients have near-complete cardiac recovery; "
            "  HIGH-DOSE RIBOFLAVIN (100-300mg/day) MUST BE TRIED IN EVERY ACAD9 PATIENT; "
            "  RIBOFLAVIN TRIAL IS DIAGNOSTIC: "
            "    Response confirms ACAD9 (or other riboflavin-responsive CI gene like TIMMDC1); "
            "  Timing: START before genetic confirmation in any unexplained Complex I cardiomyopathy; "
            "DIAGNOSIS: "
            "  Plasma lactate/pyruvate: elevated; "
            "  Muscle/cardiac biopsy: Complex I activity reduced; "
            "  WES/NGS: ACAD9 biallelic variants; "
            "  Functional: ACAD9 protein reduced on Western blot; "
            "  Riboflavin response: clinical improvement clinches diagnosis; "
            "TREATMENT: "
            "  Riboflavin 100-300mg/day: FIRST-LINE ALWAYS before other interventions; "
            "  CoQ10: supportive for complex I dysfunction; "
            "  Cardiac management: ACEi/BB/diuretics standard HF; "
            "  Transplant: if refractory and riboflavin non-responder"
        ),
        "syndrome": (
            "ACAD9-complex-I-deficiency / riboflavin-responsive-cardiomyopathy / "
            "HCM-DCM + lactic-acidosis / riboflavin-100-300mg-FIRST-LINE-ALWAYS / "
            "mitochondrial-cardiomyopathy"
        ),
        "inheritance": "AR (biallelic ACAD9 mutations)",
        "key_therapy": "High-dose riboflavin 100-300mg/day FIRST-LINE — CoQ10 supportive — cardiac HF management",
        "pathognomonic": (
            "Riboflavin-responsive cardiomyopathy + lactic acidosis = ACAD9 until proven otherwise; "
            "Complex I activity reduced on muscle/cardiac biopsy; "
            "Cardiomyopathy improvement with riboflavin = diagnostic"
        ),
        "key_drugs_ci": [
            "Riboflavin-100-300mg-day-FIRST-LINE-MANDATORY",
            "CoQ10-adjunct-complex-I-support",
            "ACEi-BB-diuretics-cardiac-HF",
            "Cardiac-transplant-riboflavin-non-responders-refractory",
            "METFORMIN-AVOID-lactic-acidosis-complex-I-inhibits",
        ],
    },
]


def _rng(seed):
    rng = random.Random(seed)
    return rng


def _gene_cohort(gene_def, seed):
    rng = _rng(seed)
    gene = gene_def["gene"]

    # Gene-specific clinical profiles
    profiles = {
        "GLA": dict(
            age_range=(20, 55), cardiac_prob=0.80, neuro_prob=0.75,
            renal_prob=0.70, myopathy_prob=0.15, metabolic_crisis_prob=0.25,
            enzyme_treated_prob=0.60, transplant_prob=0.05,
            cardiac_type="HCM-like + LVH", special="cornea_verticillata",
        ),
        "LAMP2": dict(
            age_range=(10, 35), cardiac_prob=0.98, neuro_prob=0.00,
            renal_prob=0.00, myopathy_prob=0.90, metabolic_crisis_prob=0.00,
            enzyme_treated_prob=0.00, transplant_prob=0.35,
            cardiac_type="Massive-HCM / DCM", special="intellectual_disability",
        ),
        "GAA": dict(
            age_range=(1, 60), cardiac_prob=0.55, neuro_prob=0.10,
            renal_prob=0.00, myopathy_prob=0.90, metabolic_crisis_prob=0.30,
            enzyme_treated_prob=0.80, transplant_prob=0.08,
            cardiac_type="IOPD-HCM / LOPD-mild", special="respiratory",
        ),
        "PRKAG2": dict(
            age_range=(15, 50), cardiac_prob=0.98, neuro_prob=0.00,
            renal_prob=0.00, myopathy_prob=0.10, metabolic_crisis_prob=0.00,
            enzyme_treated_prob=0.00, transplant_prob=0.10,
            cardiac_type="HCM + WPW + conduction-disease", special="wpw",
        ),
        "TAZ": dict(
            age_range=(0, 20), cardiac_prob=0.98, neuro_prob=0.00,
            renal_prob=0.00, myopathy_prob=0.80, metabolic_crisis_prob=0.00,
            enzyme_treated_prob=0.00, transplant_prob=0.25,
            cardiac_type="DCM / LVNC", special="neutropenia",
        ),
        "SLC22A5": dict(
            age_range=(1, 30), cardiac_prob=0.70, neuro_prob=0.10,
            renal_prob=0.00, myopathy_prob=0.60, metabolic_crisis_prob=0.55,
            enzyme_treated_prob=0.00, transplant_prob=0.05,
            cardiac_type="DCM", special="hypoketotic_hypoglycemia",
        ),
        "HADHA": dict(
            age_range=(0, 25), cardiac_prob=0.65, neuro_prob=0.65,
            renal_prob=0.00, myopathy_prob=0.55, metabolic_crisis_prob=0.50,
            enzyme_treated_prob=0.00, transplant_prob=0.05,
            cardiac_type="DCM / HCM", special="retinopathy",
        ),
        "ACAD9": dict(
            age_range=(0, 30), cardiac_prob=0.90, neuro_prob=0.25,
            renal_prob=0.00, myopathy_prob=0.70, metabolic_crisis_prob=0.60,
            enzyme_treated_prob=0.00, transplant_prob=0.05,
            cardiac_type="HCM / DCM", special="lactic_acidosis",
        ),
    }
    p = profiles[gene]

    ages = sorted([
        rng.uniform(p["age_range"][0], p["age_range"][1])
        for _ in range(40)
    ])
    mean_age = round(sum(ages) / 40, 1)

    cardiac_n = sum(1 for _ in range(40) if rng.random() < p["cardiac_prob"])
    neuro_n = sum(1 for _ in range(40) if rng.random() < p["neuro_prob"])
    renal_n = sum(1 for _ in range(40) if rng.random() < p["renal_prob"])
    myopathy_n = sum(1 for _ in range(40) if rng.random() < p["myopathy_prob"])
    metabolic_crisis_n = sum(1 for _ in range(40) if rng.random() < p["metabolic_crisis_prob"])
    enzyme_treated_n = sum(1 for _ in range(40) if rng.random() < p["enzyme_treated_prob"])
    transplant_n = sum(1 for _ in range(40) if rng.random() < p["transplant_prob"])

    # Special phenotypes
    cornea_verticillata_n = 0
    wpw_n = 0
    neutropenia_n = 0
    retinopathy_n = 0
    hypoketotic_hypoglycemia_n = 0
    lactic_acidosis_n = 0
    intellectual_disability_n = 0
    respiratory_n = 0

    if gene == "GLA":
        cornea_verticillata_n = sum(1 for _ in range(40) if rng.random() < 0.75)
    elif gene == "LAMP2":
        intellectual_disability_n = sum(1 for _ in range(40) if rng.random() < 0.65)
        wpw_n = sum(1 for _ in range(40) if rng.random() < 0.40)
    elif gene == "GAA":
        respiratory_n = sum(1 for _ in range(40) if rng.random() < 0.75)
    elif gene == "PRKAG2":
        wpw_n = sum(1 for _ in range(40) if rng.random() < 0.75)
    elif gene == "TAZ":
        neutropenia_n = sum(1 for _ in range(40) if rng.random() < 0.90)
    elif gene == "SLC22A5":
        hypoketotic_hypoglycemia_n = sum(1 for _ in range(40) if rng.random() < 0.60)
    elif gene == "HADHA":
        retinopathy_n = sum(1 for _ in range(40) if rng.random() < 0.55)
    elif gene == "ACAD9":
        lactic_acidosis_n = sum(1 for _ in range(40) if rng.random() < 0.85)

    return {
        "gene": gene,
        "n": 40,
        "mean_age_dx": mean_age,
        "cardiac_n": cardiac_n,
        "neuropathy_n": neuro_n,
        "renal_n": renal_n,
        "myopathy_n": myopathy_n,
        "metabolic_crisis_n": metabolic_crisis_n,
        "enzyme_treated_n": enzyme_treated_n,
        "transplant_n": transplant_n,
        "cornea_verticillata_n": cornea_verticillata_n,
        "wpw_n": wpw_n,
        "neutropenia_n": neutropenia_n,
        "retinopathy_n": retinopathy_n,
        "hypoketotic_hypoglycemia_n": hypoketotic_hypoglycemia_n,
        "lactic_acidosis_n": lactic_acidosis_n,
        "intellectual_disability_n": intellectual_disability_n,
        "respiratory_n": respiratory_n,
        "cardiac_type": p["cardiac_type"],
    }


def generate_overview():
    genes = [g["gene"] for g in ATLAS_GENES]
    per_gene = {}
    for i, gd in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        cohort = _gene_cohort(gd, seed)
        per_gene[gd["gene"]] = cohort

    total_patients = sum(c["n"] for c in per_gene.values())

    return {
        "atlas": "Hereditary-Metabolic-Cardiomyopathy-Atlas",
        "subtitle": (
            "Complete 8-Gene Reference: GLA/Fabry · LAMP2/Danon · GAA/Pompe · PRKAG2 · "
            "TAZ/Barth · SLC22A5/Primary-Carnitine-Deficiency · HADHA/LCHAD · ACAD9/Complex-I"
        ),
        "genes": genes,
        "total_patients": total_patients,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "per_gene_summary": per_gene,
        "key_clinical_themes": {
            "enzyme_replacement_available": ["GLA", "GAA"],
            "small_molecule_therapy": ["GLA (Migalastat)", "SLC22A5 (L-Carnitine)", "ACAD9 (Riboflavin)"],
            "cardiac_transplant_often_needed": ["LAMP2", "TAZ"],
            "wpw_present": ["PRKAG2 (~75%)", "LAMP2 (~40%)"],
            "NBS_detected": ["GAA", "SLC22A5 (low C0)", "HADHA (C16-OH elevated)"],
            "maternal_complication": ["HADHA (AFLP/HELLP in heterozygous mother)"],
            "riboflavin_responsive": ["ACAD9 (first-line ALWAYS)"],
            "X_linked_diseases": ["GLA", "LAMP2", "TAZ"],
        },
    }


def generate_breakdown():
    gene_data = {}
    for i, gd in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        gene_data[gd["gene"]] = _gene_cohort(gd, seed)

    return {
        "atlas": "Hereditary-Metabolic-Cardiomyopathy-Atlas",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "cardiac_type_map": {g["gene"]: _gene_cohort(g, SEED_BASE + i)["cardiac_type"]
                             for i, g in enumerate(ATLAS_GENES)},
        "inheritance_map": {g["gene"]: g["inheritance"] for g in ATLAS_GENES},
        "therapy_curative": {
            "GLA": "ERT (agalsidase alfa/beta) slows progression; Migalastat (amenable mutations)",
            "LAMP2": "Cardiac transplant only definitive option",
            "GAA": "ERT (alglucosidase/avalglucosidase) + respiratory support; curative if NBS early",
            "PRKAG2": "Catheter ablation WPW FIRST; pacemaker + ICD after ablation",
            "TAZ": "Cardiac transplant if refractory; G-CSF neutropenia",
            "SLC22A5": "L-carnitine supplementation CURATIVE for cardiac disease",
            "HADHA": "Low-fat MCT diet; avoid fasting; DHA supplementation",
            "ACAD9": "Riboflavin 100-300mg/day FIRST-LINE (responsive in majority)",
        },
    }


def generate_definitions():
    return {
        "atlas": "Hereditary-Metabolic-Cardiomyopathy-Atlas",
        "definitions": {
            "Fabry disease": (
                "X-linked lysosomal storage disorder due to GLA mutations causing alpha-galactosidase A "
                "deficiency; Gb3/GL3 accumulation in lysosomes of cardiac, renal, neural, endothelial cells; "
                "HCM-like cardiomyopathy with characteristic CMR: low T1 + basal-inferolateral LGE; "
                "Cornea verticillata (whorl corneal opacities) PATHOGNOMONIC on slit-lamp"
            ),
            "Danon disease": (
                "X-linked LAMP2 deficiency; TRIAD: cardiomyopathy (massive HCM/DCM) + skeletal myopathy "
                "+ intellectual disability PATHOGNOMONIC; males severe onset adolescence; females milder "
                "later; WPW in 30-50%; absent LAMP2 on muscle immunostaining diagnostic; "
                "cardiac transplant only definitive therapy"
            ),
            "Pompe disease": (
                "AR GAA deficiency; lysosomal glycogen accumulation; IOPD: massive cardiomegaly + "
                "hypotonia PATHOGNOMONIC + ECG triad (short PR + high voltage + left axis); LOPD: "
                "proximal myopathy + respiratory failure; enzyme replacement (alglucosidase/avalglucosidase) "
                "FDA-approved; NBS mandatory; CRIM status important for IOPD"
            ),
            "PRKAG2-cardiomyopathy": (
                "AD AMPK-gamma-2 constitutive activation; glycogen storage in cardiomyocytes; "
                "PATHOGNOMONIC TRIAD: WPW + HCM + progressive conduction disease; "
                "ICD alone DANGEROUS if WPW active (anti-tachycardia pacing -> AF -> rapid AP -> VF); "
                "catheter ablation WPW FIRST, THEN ICD; avoid digoxin/verapamil/diltiazem"
            ),
            "Barth syndrome": (
                "X-linked TAZ deficiency; cardiolipin remodeling failure; PATHOGNOMONIC TRIAD: "
                "DCM/LVNC + neutropenia + 3-methylglutaconic aciduria type II; males only; "
                "urine organic acids (elevated 3-methylglutaconic acid) PATHOGNOMONIC; "
                "monolyso-CL elevated on cardiolipin analysis PATHOGNOMONIC"
            ),
            "Primary carnitine deficiency": (
                "AR SLC22A5/OCTN2 deficiency; carnitine urinary wasting; plasma free carnitine C0 "
                "severely low (<5 uM) on NBS PATHOGNOMONIC; DCM + hypoketotic hypoglycemia PATHOGNOMONIC; "
                "L-carnitine supplementation CURATIVE for cardiac disease; lifelong therapy"
            ),
            "LCHAD deficiency": (
                "AR HADHA deficiency; long-chain fatty acid beta-oxidation failure; "
                "TRIAD: cardiomyopathy + peripheral neuropathy + pigmentary retinopathy; "
                "PATHOGNOMONIC: heterozygous mother develops AFLP/HELLP in pregnancy; "
                "C16-OH/C18-OH acylcarnitines elevated on NBS; MCT diet mandatory"
            ),
            "ACAD9-Complex I deficiency": (
                "AR ACAD9 deficiency; Complex I assembly factor; mitochondrial cardiomyopathy + "
                "lactic acidosis; most riboflavin-responsive Complex I gene; "
                "riboflavin 100-300mg/day FIRST-LINE in ALL patients; "
                "cardiomyopathy may resolve with riboflavin; CoQ10 adjunct"
            ),
            "HCM (metabolic)": (
                "Hypertrophic cardiomyopathy caused by metabolic gene mutations (GLA/Fabry, LAMP2/Danon, "
                "GAA/Pompe, PRKAG2) -- distinguished from sarcomeric HCM (MYH7, MYBPC3) by: "
                "LGE pattern (metabolic), WPW, organomegaly, enzyme deficiency, lysosomal vacuoles; "
                "Metabolic HCM may respond to disease-specific therapy (ERT, riboflavin, migalastat)"
            ),
            "LVNC": (
                "Left ventricular non-compaction: deep intertrabecular recesses + thin compacted layer; "
                "Associated with TAZ/Barth syndrome, LAMP2/Danon (some), Pompe (IOPD); "
                "Arrhythmias + HF + thromboembolism; "
                "CMR: non-compacted/compacted ratio >2.3 (Petersen) diagnostic"
            ),
            "ERT (enzyme replacement therapy)": (
                "Intravenous infusion of recombinant enzyme to replace deficient lysosomal enzyme; "
                "FDA-approved for: GLA (Fabry: agalsidase alfa/beta/pegunigalsidase alfa), "
                "GAA (Pompe: alglucosidase alfa / avalglucosidase alfa); "
                "Infusion-associated reactions; immune response monitoring required; "
                "Does NOT cross blood-brain barrier"
            ),
            "Riboflavin responsiveness": (
                "FAD (flavin adenine dinucleotide) cofactor supplementation stabilizes ACAD9 protein; "
                "Riboflavin 100-300mg/day oral; clinical response within weeks-months; "
                "Cardiomyopathy improvement/resolution + lactate normalization; "
                "ACAD9 is most riboflavin-responsive of all Complex I assembly genes; "
                "Other riboflavin-responsive CI genes: TIMMDC1, NDUFAF6"
            ),
            "AFLP": (
                "Acute fatty liver of pregnancy; PATHOGNOMONIC association with LCHAD/HADHA; "
                "Mechanism: heterozygous mother accumulates LCHA intermediates from affected fetus; "
                "Onset 3rd trimester; nausea/jaundice/coagulopathy; "
                "HELLP syndrome (haemolysis, elevated liver enzymes, low platelets) also occurs; "
                "A mother with AFLP/HELLP must have her baby TESTED FOR LCHAD IMMEDIATELY; "
                "A baby with LCHAD = screen mother retrospectively"
            ),
        },
        "critical_clinical_rules": {
            "GLA-CMR-LOW-T1-PATHOGNOMONIC": (
                "Fabry cardiomyopathy (GLA): CMR T1 mapping shows LOW native T1 (below normal range); "
                "This is OPPOSITE to fibrosis (high T1) and similar direction to amyloidosis; "
                "Characteristic LGE basal-inferolateral + midwall pattern distinguishes Fabry from HCM; "
                "Cornea verticillata = slit-lamp screen all suspected metabolic cardiomyopathy patients; "
                "Enzyme assay: diagnostic in males; may be NORMAL in females -- gene sequencing MANDATORY"
            ),
            "LAMP2-TRIAD-PATHOGNOMONIC": (
                "Danon disease (LAMP2): massive cardiomyopathy + skeletal myopathy + intellectual "
                "disability = PATHOGNOMONIC TRIAD; WPW in 30-50%; "
                "Males: fatal by age 19-25yr without transplant; "
                "Absent LAMP2 protein on muscle immunostaining = DIAGNOSTIC; "
                "Cardiac transplant is ONLY definitive therapy -- list early in males"
            ),
            "GAA-ECG-TRIAD-PATHOGNOMONIC": (
                "IOPD (infantile Pompe/GAA): ECG TRIAD = short PR + HIGH voltage QRS + left axis deviation "
                "in a floppy infant with massive cardiomegaly = PATHOGNOMONIC; "
                "CRIM status MANDATORY before starting ERT in IOPD "
                "(CRIM-negative -> immune tolerance induction with rituximab+methotrexate before ERT); "
                "LOPD: cardiomyopathy rare -- dominant is proximal myopathy + respiratory failure"
            ),
            "PRKAG2-ICD-ALONE-DANGEROUS-WPW": (
                "PRKAG2-cardiomyopathy: ICD alone is INSUFFICIENT and potentially DANGEROUS when WPW active; "
                "Anti-tachycardia pacing (ATP) sequences can trigger AF -- if accessory pathway has short ERP "
                "-> rapid conduction -> VF; "
                "RULE: catheter ablation WPW FIRST, then reassess need for ICD separately; "
                "AVOID digoxin (accessory conduction enhancement); "
                "AVOID verapamil/diltiazem (AV block + rapid AP conduction = VF risk)"
            ),
            "TAZ-3MGA-URINE-PATHOGNOMONIC": (
                "Barth syndrome (TAZ): 3-methylglutaconic acid + 3-methylglutaric acid elevated in urine "
                "= PATHOGNOMONIC TYPE II 3-MGA-uria (Barth type); "
                "Monolyso-CL elevated / L4-CL reduced on cardiolipin analysis = DIAGNOSTIC; "
                "Males ONLY (X-linked); LVNC or DCM; neutropenia with cycling; "
                "Screen infants: DCM + neutropenia + 3-MGA in urine = BARTH until proven otherwise"
            ),
            "SLC22A5-LCARNITINE-CURATIVE": (
                "Primary carnitine deficiency (SLC22A5/OCTN2): L-carnitine 50-100mg/kg/day oral; "
                "DCM REVERSES with treatment if started before irreversible fibrosis; "
                "NBS: free carnitine C0 < 5 uM (normal >20) PATHOGNOMONIC; "
                "PIVAMPICILLIN/PIVALATE-CONTAINING ANTIBIOTICS: deplete carnitine -- ABSOLUTELY AVOID; "
                "VALPROATE: depletes carnitine -- AVOID in CDSP; "
                "Mother may be asymptomatic carrier: index case in baby prompts maternal testing"
            ),
            "HADHA-AFLP-MOTHER-PATHOGNOMONIC": (
                "LCHAD deficiency (HADHA): A mother who develops AFLP (acute fatty liver of pregnancy) "
                "or HELLP syndrome = TEST THE BABY FOR LCHAD IMMEDIATELY; "
                "Mechanism: heterozygous mother carrying LCHAD-affected fetus accumulates long-chain "
                "3-OH-acylcarnitine intermediates from the placenta -> maternal hepatotoxicity; "
                "This association is PATHOGNOMONIC for LCHAD; "
                "Pigmentary retinopathy: annual ophthalmology MANDATORY from diagnosis; "
                "PROPOFOL INFUSION SYNDROME: propofol inhibits mitochondria -> AVOID in HADHA"
            ),
            "ACAD9-RIBOFLAVIN-FIRST-LINE-ALWAYS": (
                "ACAD9 (Complex I deficiency): HIGH-DOSE RIBOFLAVIN (vitamin B2) 100-300mg/day "
                "MUST BE TRIED IN EVERY ACAD9 PATIENT before other interventions; "
                "ACAD9 is THE MOST riboflavin-responsive Complex I assembly gene; "
                "Riboflavin stabilizes FAD cofactor -> ACAD9 protein stability -> Complex I assembly; "
                "Clinical response: cardiomyopathy improvement/resolution + lactate normalization; "
                "METFORMIN ABSOLUTELY AVOID: Complex I inhibitor -> severe lactic acidosis in ACAD9"
            ),
            "METABOLIC-HCM-SCREEN-PROTOCOL": (
                "For any HCM or unexplained cardiomyopathy, METABOLIC SCREEN MANDATORY: "
                "1. Alpha-galactosidase A enzyme (males) + GLA sequencing (females) -- Fabry; "
                "2. LAMP2 gene sequencing if massive HCM + myopathy +/- intellectual disability; "
                "3. GAA enzyme assay -- Pompe (short PR + high voltage ECG clues); "
                "4. Plasma lyso-Gb3 -- Fabry; "
                "5. Urine 3-methylglutaconic acid -- Barth/TAZ; "
                "6. Plasma free carnitine + acylcarnitine profile -- CDSP, LCHAD, ACAD9; "
                "7. Lactate/pyruvate -- mitochondrial (ACAD9, HADHA); "
                "8. CMR with T1 mapping -- Fabry (low T1 + basal-inferolateral LGE)"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Metabolic Cardiomyopathy Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(
            f"  {gene}: n={d['n']}, age_dx={d['mean_age_dx']}yr, "
            f"cardiac={d['cardiac_n']}/40, neuro={d['neuropathy_n']}/40, "
            f"myopathy={d['myopathy_n']}/40, transplant={d['transplant_n']}/40"
        )
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
