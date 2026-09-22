#!/usr/bin/env python3
"""Hereditary-Hypophosphatemic-Rickets-Atlas — Complete 8-Gene FGF23/Phosphate-Wasting Atlas
PHEX    (phosphate-regulating endopeptidase homolog X-linked; 749 aa; Xp22.11; XL LOF;
         X-linked hypophosphatemia (XLH) — most common hereditary rickets 1:20,000;
         iFGF23 HIGH + TRP LOW + 1,25D inappropriately normal; burosumab CURATIVE; seed SEED_BASE+0) ·
FGF23   (fibroblast growth factor 23; 251 aa; 12p13.32; AD GOF = ADHR / AR LOF = tumoral calcinosis;
         ADHR: GOF mutation resists FGF23 cleavage → same phosphaturia as XLH; burosumab also effective;
         AR LOF: HYPERPHOSPHATEMIA + periarticular calcifications PATHOGNOMONIC; seed SEED_BASE+1) ·
DMP1    (dentin matrix protein 1; 473aa; 4q22.1; AR LOF;
         ARHR1 — FGF23 overproduction; ENTHESOPATHY distinctive in adults; seed SEED_BASE+2) ·
ENPP1   (ectonucleotide pyrophosphatase/phosphodiesterase 1; 925aa; 6q23.2; AR LOF;
         ARHR2 + GACI — pyrophosphate deficiency dual phenotype; neonatal arterial calcification;
         etidronate dissolves GACI calcification; survivors → hypophosphatemia; seed SEED_BASE+3) ·
CLCN5   (chloride voltage-gated channel 5; 746aa; Xp11.23; XL LOF;
         Dent disease type 1 — LMW proteinuria + hypercalciuria + nephrocalcinosis + variable rickets;
         β2-microglobulin in urine PATHOGNOMONIC for tubular proteinuria; seed SEED_BASE+4) ·
OCRL    (oculocerebrorenal syndrome protein; 901aa; Xq26.1; XL LOF;
         Lowe syndrome — cataracts + intellectual disability + Fanconi renal syndrome;
         Dent disease type 2 = males without eye/brain features; seed SEED_BASE+5) ·
SLC34A3 (sodium-phosphate cotransporter IIc; 599aa; 9q34.3; AR LOF;
         HHRH — suppressed PTH + HIGH 1,25D + hypercalciuria PATHOGNOMONIC; seed SEED_BASE+6) ·
CYP27B1 (25-hydroxyvitamin D 1-alpha-hydroxylase; 508aa; 12q14.1; AR LOF;
         VDDR1 — low 1,25D despite normal 25D; calcitriol supplementation CURATIVE; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2974-2981)
"""
import random

SEED_BASE = 2974

ATLAS_GENES = [
    {
        "gene": "PHEX",
        "protein": (
            "PHEX -- Xp22.11 XL LOF -- 749aa -- Phosphate-Regulating-Endopeptidase-"
            "Homolog-X-Linked-86kDa-Osteoblast-Osteocyte-Metalloendopeptidase-"
            "XLH-iFGF23-HIGH-TRP-LOW-1,25D-Inappropriately-Normal-Burosumab-CURATIVE-OMIM-307800"
        ),
        "locus": "Xp22.11",
        "protein_size": (
            "749 aa / 86 kDa (PHEX — phosphate-regulating endopeptidase homolog, X-linked; "
            "FUNCTION: zinc metalloendopeptidase expressed in osteoblasts and osteocytes; "
            "  Normally cleaves and inactivates FGF23 (fibroblast growth factor 23); "
            "  FGF23 acts on kidney proximal tubule: "
            "    1. Downregulates NaPi-IIa/IIc (SLC34A1/SLC34A3) → phosphate wasting; "
            "    2. Inhibits CYP27B1 (1α-hydroxylase) → low 1,25(OH)2D production; "
            "  Without PHEX-mediated cleavage → FGF23 accumulates → chronic phosphaturia; "
            "  PHEX does NOT directly cleave FGF23 — actual mechanism: PHEX cleaves ASARM peptides "
            "  (acidic serine/aspartate-rich MEPE-associated motif) which inhibit PHEX substrate; "
            "  Net effect of PHEX LOF: FGF23 excess → phosphate wasting + low active vitamin D; "
            "LOF CONSEQUENCE (XLH = X-linked hypophosphatemia): "
            "  Most common hereditary rickets: 1 in 20,000; "
            "  Lifelong phosphate wasting; short stature; bone pain; dental abscesses; "
            "  encoded Xp22.11"
        ),
        "inheritance": (
            "X-LINKED (XL) LOF — PHEX / X-LINKED HYPOPHOSPHATEMIA (XLH): "
            "  PHENOTYPE (all X-linked: males fully affected; females variably affected): "
            "    Skeletal: "
            "      Rickets in childhood — bowed legs, short stature, waddling gait; "
            "      Bone pain; muscle weakness; reduced exercise capacity; "
            "      Short stature: adult height 2-3 SDS below mean; "
            "    Dental: "
            "      SPONTANEOUS DENTAL ABSCESSES without dental caries — PATHOGNOMONIC: "
            "      Hypomineralised dentine → bacteria enter via dentinal tubules → periapical abscess; "
            "      Occurs even in teeth without cavities; regular dental follow-up MANDATORY; "
            "    Adult complications: "
            "      Enthesopathy (tendons/ligaments calcify at insertions); "
            "      Hearing loss; osteoarthritis; Chiari malformation (rare); "
            "  BIOCHEMISTRY (all HPR share some of these): "
            "    Serum phosphate: LOW (hypophosphataemia); "
            "    Tubular reabsorption of phosphate (TRP): LOW (<85%); "
            "    FGF23 (intact): MARKEDLY ELEVATED; "
            "    1,25(OH)2D (calcitriol): NORMAL or LOW — INAPPROPRIATELY NORMAL for degree of hypophosphataemia; "
            "    PTH: normal (distinguish from secondary hyperparathyroidism); "
            "    ALP: ELEVATED (active bone disease marker); "
            "    Serum calcium: normal; "
            "  KEY CLINICAL RULE — BUROSUMAB CURATIVE: "
            "    Burosumab (KRN23) = anti-FGF23 monoclonal antibody; "
            "    Directly neutralises excess FGF23 → phosphate reabsorption restored; "
            "    FDA approved 2018 (children ≥1 year); "
            "    Dramatically improves rickets, normalises phosphate, improves growth; "
            "    REPLACE conventional therapy (oral phosphate + calcitriol) with burosumab; "
            "    Oral phosphate + calcitriol: still used if burosumab unavailable; "
            "      Risk with oral Rx: secondary hyperparathyroidism + nephrocalcinosis; "
            "  X-LINKED PATTERN: "
            "    Males: full phenotype (no normal X to compensate); "
            "    Females: variable (X-inactivation → range from near-normal to affected); "
            "    50% sons of affected females = affected; 100% daughters of affected males = affected carriers"
        ),
        "disease_category": (
            "PHEX-XLH — MOST-COMMON-HEREDITARY-RICKETS-1:20000 — BUROSUMAB-CURATIVE: "
            "  DIAGNOSIS: serum phosphate LOW + TRP LOW + iFGF23 HIGH + 1,25D inappropriately normal; "
            "    X-linked family history; spontaneous dental abscesses without caries; "
            "  TREATMENT: "
            "    First-line: burosumab (anti-FGF23 mAb) sc every 2 weeks (paediatric); "
            "    Alternatively: oral phosphate 4-6×/day + calcitriol (risk: nephrocalcinosis); "
            "    Dental surveillance every 6 months; orthopedic review for deformity; "
            "  GENETIC TESTING: "
            "    PHEX sequencing + MLPA (deletions); "
            "    If PHEX negative → FGF23/DMP1/ENPP1/SLC34A3 panel"
        ),
        "disease_pathway": (
            "PHEX LOF → FGF23 EXCESS → PHOSPHATURIA + LOW 1,25D → RICKETS: "
            "  Normal PHEX-FGF23 axis: "
            "    Osteocytes produce FGF23 → secreted into circulation; "
            "    PHEX in osteoblasts/osteocytes cleaves and inactivates FGF23; "
            "    FGF23 at kidney proximal tubule + FGF receptor + α-klotho co-receptor: "
            "      Downregulates SLC34A1 (NaPi-IIa) + SLC34A3 (NaPi-IIc) → phosphate wasting; "
            "      Inhibits CYP27B1 → reduced 1,25(OH)2D synthesis; "
            "  PHEX LOF: "
            "    ASARM peptides not cleaved → FGF23 cleavage impaired → FGF23 accumulates; "
            "    Chronic phosphaturia → hypophosphataemia; "
            "    1,25(OH)2D synthetically suppressed → cannot compensate for low phosphate; "
            "    Osteoblasts cannot mineralise matrix without adequate phosphate → osteomalacia/rickets; "
            "    Dental tubules hypomineralised → bacterial ingress → abscess without caries"
        ),
    },
    {
        "gene": "FGF23",
        "protein": (
            "FGF23 -- 12p13.32 AD GOF = ADHR / AR LOF = Tumoral-Calcinosis -- 251aa -- "
            "Fibroblast-Growth-Factor-23-26kDa-Phosphatonin-RXXR-Cleavage-Site-"
            "ADHR-Resistant-to-Proteolysis-GOF / Tumoral-Calcinosis-HYPERPHOSPHATEMIA-Calcification-LOF-OMIM-605380"
        ),
        "locus": "12p13.32",
        "protein_size": (
            "251 aa / 26 kDa (FGF23 — fibroblast growth factor 23; phosphatonin; "
            "FUNCTION: phosphate-regulating hormone made in osteocytes; "
            "  Acts via FGFR1/3/4 + α-klotho co-receptor on kidney proximal tubule; "
            "  Downregulates NaPi-IIa/IIc → phosphate wasting; "
            "  Inhibits CYP27B1 → reduces 1,25(OH)2D synthesis; "
            "  Normally cleaved at RXXR motif (R176-Q177 cleavage site) by subtilisin-like proteases; "
            "  Cleavage inactivates FGF23; intact FGF23 (iFGF23) = active; "
            "AD GOF CONSEQUENCE (ADHR = autosomal dominant hypophosphatemic rickets): "
            "  R176Q or R179W/Q mutations destroy RXXR cleavage site → FGF23 resistant to proteolysis; "
            "  iFGF23 accumulates → same phosphaturia as XLH; "
            "  ADHR is INTERMITTENT — disease can fluctuate; iron deficiency triggers flares; "
            "AR LOF CONSEQUENCE (Tumoral Calcinosis / Hyperostosis-Hyperphosphatemia): "
            "  FGF23 absent → HYPERPHOSPHATEMIA + HIGH 1,25(OH)2D; "
            "  Massive periarticular calcium-phosphate deposits (tumoral calcinosis); "
            "  OPPOSITE of all other HPR genes; "
            "  encoded 12p13.32"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GOF (ADHR) / AUTOSOMAL RECESSIVE LOF (TUMORAL CALCINOSIS): "
            "  ADHR PHENOTYPE (AD GOF): "
            "    Phosphate-wasting rickets: similar to XLH but AD; "
            "    KEY DIFFERENCE: INTERMITTENT disease — can remit then relapse (vs XLH lifelong); "
            "    Flares triggered by: iron deficiency, pregnancy, puberty; "
            "    Iron supplementation may stabilise disease (iron needed for FGF23 cleavage); "
            "    Serum iFGF23 HIGH; intact FGF23 resistant to proteolysis; "
            "    Treatment: burosumab (anti-FGF23 mAb) effective same as XLH; "
            "    Or phosphate + calcitriol; iron supplementation for flares; "
            "  TUMORAL CALCINOSIS (AR LOF) PHENOTYPE: "
            "    BIOCHEMISTRY OPPOSITE TO ALL OTHER HPR: "
            "      Serum phosphate: ELEVATED (hyperphosphataemia); "
            "      TRP: HIGH (too much phosphate reabsorption); "
            "      1,25(OH)2D: ELEVATED; "
            "      PTH: normal or suppressed; "
            "    CALCIUM-PHOSPHATE DEPOSITS: "
            "      Massive periarticular calcifications (hips, elbows, shoulders); "
            "      Dense calcium deposits on imaging — PATHOGNOMONIC APPEARANCE; "
            "      NOT true tumours — calcium phosphate crystal deposits; "
            "    TREATMENT: low-phosphate diet; phosphate binders; acetazolamide; "
            "      Surgery for refractory/disabling deposits; "
            "  KEY CLINICAL RULE — FGF23 LOF vs GOF OPPOSITE PHENOTYPES: "
            "    GOF (ADHR): FGF23 high → phosphate LOW (same as XLH); "
            "    LOF (Tumoral Calcinosis): FGF23 absent → phosphate HIGH + calcifications; "
            "    Same gene, opposite phenotypes — distinguish by biochemistry"
        ),
        "disease_category": (
            "FGF23-ADHR-GOF-HYPOPHOSPHATAEMIA / FGF23-LOF-TUMORAL-CALCINOSIS-HYPERPHOSPHATAEMIA: "
            "  ADHR: intermittent rickets + iFGF23 HIGH + iron deficiency triggers; burosumab + iron; "
            "  TUMORAL CALCINOSIS: massive periarticular Ca-P deposits + HYPERPHOSPHATAEMIA + HIGH 1,25D; "
            "  DIAGNOSIS CLUE: phosphate HIGH with calcifications = FGF23 LOF until proven otherwise; "
            "  GENETIC TESTING: "
            "    ADHR: R176Q or R179W/Q hotspot mutations in FGF23; "
            "    Tumoral calcinosis: GALNT3 (O-glycosylation) and KL (α-klotho) also cause TC"
        ),
        "disease_pathway": (
            "FGF23 GOF → CLEAVAGE-RESISTANT FGF23 → SAME AS XLH PATHWAY: "
            "  ADHR mechanism: "
            "    FGF23 mutations destroy R-X-X-R cleavage motif; "
            "    Furin/subtilisin proteases cannot cleave → intact FGF23 persists; "
            "    Elevated iFGF23 → phosphaturia + low 1,25D → rickets/osteomalacia; "
            "    Iron status modulates: low iron → increased FGF23 transcription → worsens; "
            "  FGF23 LOF (Tumoral Calcinosis) mechanism: "
            "    No FGF23 signal → FGFR/klotho unactivated → NaPi transporters maximally active; "
            "    Phosphate freely reabsorbed → hyperphosphataemia; "
            "    CYP27B1 not inhibited → 1,25D high; "
            "    Supersaturation of Ca×P product → periarticular precipitation; "
            "    Soft tissue calcifications grow progressively"
        ),
    },
    {
        "gene": "DMP1",
        "protein": (
            "DMP1 -- 4q22.1 AR LOF -- 473aa -- Dentin-Matrix-Protein-1-"
            "54kDa-SIBLING-Family-RGD-Motif-Osteocyte-FGF23-Regulator-"
            "ARHR1-Enthesopathy-Spinal-Ligament-Calcification-Distinctive-OMIM-600980"
        ),
        "locus": "4q22.1",
        "protein_size": (
            "473 aa / 54 kDa (DMP1 — dentin matrix acidic phosphoprotein 1; SIBLING family; "
            "FUNCTION: expressed in osteocytes and odontoblasts; "
            "  Normally suppresses FGF23 overproduction in osteocytes; "
            "  DMP1 in bone matrix provides structural support to the canalicular network; "
            "  Promotes osteocyte maturation; "
            "  Directly binds and sequesters FGF23 → reduces FGF23 secretion; "
            "  Also has direct mineralisation role via RGD motif (cell adhesion); "
            "LOF CONSEQUENCE (ARHR1 = autosomal recessive hypophosphatemic rickets type 1): "
            "  DMP1 absent → osteocytes immature → FGF23 overproduction → phosphaturia; "
            "  Same biochemical profile as XLH (phosphate low, TRP low, FGF23 high, 1,25D low); "
            "  AR biallelic required (vs XLH X-linked); "
            "  DISTINCTIVE ADULT FEATURE: "
            "    ENTHESOPATHY: calcification of tendons, ligaments at bony insertions; "
            "    Particularly spinal ligament calcification (resembles ankylosing spondylitis on imaging); "
            "    Periosteal reactions in diaphysis; "
            "  encoded 4q22.1"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — DMP1 / ARHR1: "
            "  PHENOTYPE (onset childhood to early adulthood): "
            "    Rickets / osteomalacia: similar to XLH; bowing; short stature; "
            "    Dentinogenesis imperfecta: abnormal dentin structure; "
            "    ENTHESOPATHY — DISTINCTIVE FEATURE OF DMP1/ARHR1: "
            "      Tendon and ligament insertions calcify (entheses); "
            "      Spinal ligament calcification → restricted spinal mobility; "
            "      Mimics ankylosing spondylitis on imaging; "
            "      Periosteal reactions along long bone diaphyses; "
            "      More prominent in adults; "
            "  BIOCHEMISTRY: "
            "    Same profile as XLH: low phosphate, low TRP, high iFGF23, low/normal 1,25D; "
            "  KEY DISTINGUISHER FROM XLH: "
            "    AR inheritance (not X-linked); "
            "    Enthesopathy more prominent; "
            "    Periosteal reactions more common; "
            "    Dental features: opalescent teeth, dentinogenesis imperfecta (more than XLH); "
            "  TREATMENT: "
            "    Burosumab (anti-FGF23): same mechanism, same benefit as for XLH; "
            "    Alternatively: oral phosphate + calcitriol; "
            "  GENETIC TESTING: "
            "    PHEX negative XLH-like → DMP1 sequencing (AR; family history crucial)"
        ),
        "disease_category": (
            "DMP1-ARHR1 — AR-INHERITED-XLH-LIKE — ENTHESOPATHY-DISTINCTIVE: "
            "  DIAGNOSIS CLUE: XLH-like biochemistry + AR family history + spinal enthesopathy; "
            "    Periosteal reactions in adults; opalescent teeth; "
            "  TREATMENT: burosumab or phosphate + calcitriol; spinal physiotherapy; "
            "  GENETIC TESTING: "
            "    Biallelic DMP1 mutations confirm ARHR1; "
            "    ARHR1 founder mutation: c.1484delT (del of T in exon 6) in some populations"
        ),
        "disease_pathway": (
            "DMP1 LOF → FGF23 OVERPRODUCTION → PHOSPHATURIA + LOW 1,25D → RICKETS/ENTHESOPATHY: "
            "  Normal DMP1-FGF23 regulation: "
            "    Osteocytes produce DMP1 → matrix glycoprotein in lacunocanalicular network; "
            "    DMP1 directly suppresses FGF23 gene transcription in osteocytes; "
            "    DMP1 may also promote osteocyte maturation (immature osteocytes = FGF23 high); "
            "  DMP1 LOF: "
            "    Immature osteocytes + absent FGF23 suppression → chronic FGF23 overproduction; "
            "    FGF23 acts on kidney → phosphaturia + suppressed 1,25D → rickets; "
            "    In bone matrix: DMP1 absent → abnormal canalicular network; "
            "    Abnormal mineralisation at entheses → calcium-phosphate deposition at insertions; "
            "    Spinal ligaments particularly affected in adults"
        ),
    },
    {
        "gene": "ENPP1",
        "protein": (
            "ENPP1 -- 6q23.2 AR LOF -- 925aa -- Ectonucleotide-Pyrophosphatase-Phosphodiesterase-1-"
            "100kDa-Type-II-Transmembrane-Glycoprotein-PPi-Generator-"
            "ARHR2-Hypophosphataemia-PLUS-GACI-Neonatal-Arterial-Calcification-Pyrophosphate-Deficiency-OMIM-173335"
        ),
        "locus": "6q23.2",
        "protein_size": (
            "925 aa / 100 kDa (ENPP1 — ectonucleotide pyrophosphatase/phosphodiesterase 1; "
            "FUNCTION: type II transmembrane glycoprotein; "
            "  Converts extracellular ATP → AMP + inorganic pyrophosphate (PPi); "
            "  PPi is a KEY inhibitor of calcification: "
            "    PPi inhibits hydroxyapatite crystal growth → prevents ectopic mineralisation; "
            "    Low PPi → abnormal calcification can occur; "
            "  Also regulates FGF23 via unclear mechanism (ENPP1 LOF → FGF23 elevated); "
            "  High PPi is normally maintained to prevent vascular calcification; "
            "LOF CONSEQUENCE (dual phenotype — age-dependent): "
            "  NEONATAL/INFANTILE: GACI (generalised arterial calcification of infancy): "
            "    PPi absent → massive hydroxyapatite deposits in arterial walls; "
            "    Coronary, aortic, pulmonary arterial calcification → stenosis; "
            "    Heart failure, hydrops fetalis, respiratory failure → HIGH MORTALITY without treatment; "
            "  CHILDHOOD/LATER: ARHR2 (if survive GACI): "
            "    FGF23 elevated → hypophosphataemia + rickets; "
            "    PPi deficiency → paradoxically LESS calcification than expected (PPi partly clears); "
            "  encoded 6q23.2"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — ENPP1 / GACI + ARHR2 DUAL PHENOTYPE: "
            "  GACI PHENOTYPE (neonatal/infantile): "
            "    Generalised arterial calcification: "
            "      Multifocal: coronary arteries, aorta, pulmonary arteries, renal arteries; "
            "      Calcification of intima + media → stenosis → ischaemia; "
            "    Clinical: "
            "      Cardiomegaly, heart failure, respiratory distress; "
            "      Hydrops fetalis (severe cases); "
            "      Hypertension (renal artery stenosis); "
            "    MORTALITY: untreated ~50% before 6 months; "
            "    TREATMENT FOR GACI: "
            "      ETIDRONATE (bisphosphonate): replenishes PPi analogue → dissolves calcification; "
            "        Etidronate is FIRST-LINE for GACI — can REVERSE arterial calcification; "
            "        Dramatic improvement possible; "
            "      IV sodium pyrophosphate: investigational; "
            "  ARHR2 PHENOTYPE (survivors of GACI or mild ENPP1 mutations): "
            "    Hypophosphataemic rickets: same as XLH biochemically; "
            "    FGF23 elevated (mechanism: ENPP1 LOF → unclear FGF23 elevation pathway); "
            "    Onset: childhood; bowing, short stature, bone pain; "
            "    Treatment: burosumab or phosphate + calcitriol; "
            "  KEY CLINICAL RULE — CALCIFICATION PARADOX: "
            "    ENPP1 LOF → PPi absent → expect hypocalcification; "
            "    BUT arteries calcify massively (calcium-phosphate product too high); "
            "    Bone: rickets (not enough calcium for bone); "
            "    Vessels: calcification (calcium-phosphate deposits in smooth muscle); "
            "    Same mutation = calcification in vessels + de-calcification in bone"
        ),
        "disease_category": (
            "ENPP1-GACI-ARHR2 — NEONATAL-ARTERIAL-CALCIFICATION — ETIDRONATE-FIRST-LINE: "
            "  DIAGNOSIS: neonate with heart failure + vascular calcification on echo/X-ray → ENPP1; "
            "    ENPP1 activity assay in plasma; ENPP1 sequencing; "
            "  TREATMENT: "
            "    GACI: etidronate URGENTLY; IV PPi investigational; "
            "    ARHR2: burosumab; phosphate + calcitriol; "
            "  GENETIC TESTING: "
            "    ENPP1 biallelic — any severe GACI in neonate should include ENPP1 + ABCC6 testing"
        ),
        "disease_pathway": (
            "ENPP1 LOF → PPi DEFICIENCY → ARTERIAL CALCIFICATION (GACI) + FGF23 EXCESS → ARHR2: "
            "  Normal PPi homeostasis: "
            "    ENPP1 cleaves extracellular ATP → AMP + PPi; "
            "    PPi inhibits hydroxyapatite growth on collagen in vessel walls → no ectopic calcification; "
            "    Tissue non-specific alkaline phosphatase (TNSALP) cleaves PPi → Pi (releases brake); "
            "    Balance: ENPP1 (generates PPi brake) vs TNSALP (removes PPi brake); "
            "  ENPP1 LOF: "
            "    PPi absent → no inhibition of calcification → hydroxyapatite precipitates in arteries; "
            "    Massive coronary/aortic calcification; "
            "    FGF23 elevated via unknown mechanism → phosphate wasting → ARHR2 in survivors; "
            "  Etidronate treatment: "
            "    Etidronate is a PPi analogue → inhibits hydroxyapatite growth → reverses GACI"
        ),
    },
    {
        "gene": "CLCN5",
        "protein": (
            "CLCN5 -- Xp11.23 XL LOF -- 746aa -- Chloride-Voltage-Gated-Channel-5-"
            "83kDa-CLC-5-Endosomal-H-Cl-Exchanger-Proximal-Tubule-Endocytosis-"
            "Dent-Disease-Type-1-LMW-Proteinuria-Hypercalciuria-Nephrocalcinosis-CKD-OMIM-300008"
        ),
        "locus": "Xp11.23",
        "protein_size": (
            "746 aa / 83 kDa (CLCN5 — chloride voltage-gated channel 5; CLC-5; "
            "FUNCTION: endosomal H+/Cl- exchanger expressed in proximal tubule cells; "
            "  Acidifies endosomes → required for normal endocytosis of filtered proteins; "
            "  Proximal tubule reabsorbs filtered low-molecular-weight (LMW) proteins by megalin-cubilin; "
            "  Without CLC-5: endosomes not acidified → receptor recycling impaired → LMW proteins lost in urine; "
            "  Also affects NHE3 (sodium-hydrogen exchanger) trafficking → aminoaciduria + glucosuria; "
            "LOF CONSEQUENCE (Dent disease type 1): "
            "  LMW proteinuria (β2-microglobulin, retinol-binding protein, α1-microglobulin): "
            "    CARDINAL FEATURE — present in >95% of males; "
            "    β2-microglobulin in urine PATHOGNOMONIC for tubular (not glomerular) proteinuria; "
            "  Hypercalciuria: calcium reabsorption impaired; "
            "  Nephrocalcinosis + nephrolithiasis; "
            "  Progressive CKD (30-80% reach ESRD by age 30-50); "
            "  Variable rickets/osteomalacia (20-40% of males); "
            "  encoded Xp11.23"
        ),
        "inheritance": (
            "X-LINKED (XL) LOF — CLCN5 / DENT DISEASE TYPE 1: "
            "  PHENOTYPE (males predominantly; females variably): "
            "    CARDINAL TRIAD (must-have): "
            "      1. Low-molecular-weight proteinuria (β2-microglobulin, retinol-binding protein): "
            "         MANDATORY for diagnosis; ALL affected males have this; "
            "         β2-MG >1 mg/L in urine (or >0.3 mg/mmol creatinine) = abnormal; "
            "      2. Hypercalciuria (>4 mg/kg/day; spot urine Ca:Cr ratio >0.25); "
            "      3. At least one of: nephrocalcinosis, nephrolithiasis, CKD, rickets, hypophosphataemia; "
            "    RENAL DISEASE: "
            "      Progressive chronic kidney disease (30-80% of males reach ESRD by 30-50 years); "
            "      Mechanism: chronic tubular dysfunction + nephrocalcinosis + recurrent stones; "
            "    BONE DISEASE (variable): "
            "      Rickets in childhood (20-40% of males); "
            "      Fractures; osteopenia/osteomalacia; "
            "      Hypophosphataemia (due to tubular phosphate wasting); "
            "    KEY DISTINCTION FROM OTHER HPR: "
            "      Dent 1 = LMW PROTEINURIA is defining feature (not pure rickets); "
            "      Other HPR genes (PHEX/DMP1/SLC34A3): no LMW proteinuria; "
            "      Dent 1 vs Dent 2 (OCRL): identical biochemistry — need gene testing to distinguish; "
            "  FEMALES: "
            "    Usually mild (heterozygous); may have LMW proteinuria; rarely full disease; "
            "  TREATMENT: "
            "    No disease-modifying therapy; "
            "    Thiazide diuretics: reduce hypercalciuria + nephrolithiasis risk; "
            "    Phosphate + calcitriol for rickets; "
            "    ACE inhibitor for proteinuria/CKD progression; "
            "    Avoid calcium supplements; high fluid intake"
        ),
        "disease_category": (
            "CLCN5-DENT-1 — LMW-PROTEINURIA-PATHOGNOMONIC — PROGRESSIVE-CKD: "
            "  DIAGNOSIS: male with LMW proteinuria + hypercalciuria + nephrocalcinosis; "
            "    β2-microglobulin in urine marks tubular (not glomerular) origin; "
            "  TREATMENT: thiazide + fluid; phosphate + calcitriol if rickets; ACE inhibitor for CKD; "
            "  GENETIC TESTING: "
            "    CLCN5 sequencing; if negative → OCRL (Dent 2 / Lowe) testing; "
            "    Dent 1 and Dent 2 biochemically identical — gene testing mandatory to distinguish"
        ),
        "disease_pathway": (
            "CLCN5 LOF → ENDOSOMAL ACIDIFICATION DEFECT → LMW PROTEINURIA + TUBULAR DYSFUNCTION: "
            "  Normal CLCN5/CLC-5 in proximal tubule: "
            "    Filtered LMW proteins (β2-MG, RBP, albumin) bind megalin-cubilin at brush border; "
            "    Endocytosed into clathrin-coated vesicles → early endosomes; "
            "    CLC-5 on endosomal membrane: H+/Cl- exchanger → acidifies endosome (pH 5.0-5.5); "
            "    Acid pH: megalin-cubilin releases cargo → cargo transferred to lysosomes; "
            "    Megalin-cubilin recycled to brush border → repeat cycle; "
            "  CLCN5 LOF: "
            "    Endosomes not acidified → megalin-cubilin cannot release cargo; "
            "    Receptors not recycled → tubular endocytosis capacity reduced; "
            "    LMW proteins not reabsorbed → appear in urine (LMW proteinuria); "
            "    NHE3 trafficking also impaired → aminoaciduria, glucosuria; "
            "    Calcium: CLC-5 affects calcium transport indirectly → hypercalciuria; "
            "    Progressive tubular injury → nephron loss → CKD"
        ),
    },
    {
        "gene": "OCRL",
        "protein": (
            "OCRL -- Xq26.1 XL LOF -- 901aa -- Oculocerebrorenal-Syndrome-Protein-"
            "105kDa-Phosphatidylinositol-5-Phosphatase-INPP5F-PIP2-PI(4,5)P2-Hydrolysis-"
            "Lowe-Syndrome-Cataracts-ID-Fanconi-Rickets-Dent-Disease-Type-2-Incomplete-OMIM-300535"
        ),
        "locus": "Xq26.1",
        "protein_size": (
            "901 aa / 105 kDa (OCRL — oculocerebrorenal syndrome protein of Lowe; "
            "FUNCTION: phosphatidylinositol-4,5-bisphosphate 5-phosphatase; "
            "  Hydrolyses PI(4,5)P2 → PI(4)P at the Golgi and endosomes; "
            "  Critical for vesicular trafficking (clathrin-coated pit recycling); "
            "  Expressed in: kidney proximal tubule + lens + brain; "
            "  In proximal tubule: same role as CLCN5 — megalin-cubilin recycling; "
            "  OCRL LOF → endosomal PI(4,5)P2 accumulates → clathrin-coated vesicle recycling impaired; "
            "LOF CONSEQUENCE (Lowe syndrome / OCRL syndrome): "
            "  OCULOCEREBRORENAL TRIAD (when all three features present = Lowe syndrome): "
            "    1. Congenital cataracts (bilateral, dense — present at birth); "
            "    2. Intellectual disability (moderate-severe — cerebral hypotonia); "
            "    3. Renal Fanconi syndrome (aminoaciduria, LMW proteinuria, phosphaturia, glucosuria); "
            "  DENT DISEASE TYPE 2 (incomplete Lowe): "
            "    Males with OCRL mutations but WITHOUT cataracts or significant ID; "
            "    Only renal Dent phenotype: LMW proteinuria + hypercalciuria; "
            "  encoded Xq26.1"
        ),
        "inheritance": (
            "X-LINKED (XL) LOF — OCRL / LOWE SYNDROME (complete) or DENT DISEASE TYPE 2 (incomplete): "
            "  LOWE SYNDROME PHENOTYPE (full triad in males): "
            "    CATARACTS: "
            "      Bilateral dense cataracts at birth (congenital); "
            "      Immediate surgical intervention required to prevent amblyopia; "
            "      May also have glaucoma and corneal keloids; "
            "    INTELLECTUAL DISABILITY: "
            "      Moderate-severe ID; hypotonia at birth; "
            "      Behavioural problems: stereotypies, tantrums, OCD-like; "
            "      Seizures in ~50%; "
            "    RENAL FANCONI SYNDROME: "
            "      LMW proteinuria + aminoaciduria + phosphaturia + glucosuria + bicarbonate wasting; "
            "      Rickets due to phosphaturia; hypophosphataemia; "
            "      Progressive CKD (similar to Dent 1 but often more severe); "
            "      Nephrocalcinosis + nephrolithiasis; "
            "  DENT DISEASE TYPE 2 PHENOTYPE (males without eye/brain features): "
            "    LMW proteinuria + hypercalciuria — identical to Dent 1; "
            "    Mild ID in some (but not full Lowe syndrome); "
            "    Must be distinguished from Dent 1 by OCRL sequencing; "
            "  FEMALES: "
            "    Slit-lamp exam: punctate lens opacities (carrier sign) in >90%; "
            "    Usually mild renal disease; rarely intellectual disability; "
            "  KEY CLINICAL RULE: "
            "    Dent 1 (CLCN5) vs Dent 2 (OCRL): biochemically identical — gene testing mandatory; "
            "    Lowe syndrome: cataracts at birth = check OCRL; "
            "  TREATMENT: "
            "    Cataracts: surgery in first weeks of life; glasses/patching; "
            "    Renal Fanconi: phosphate + calcitriol + alkali (bicarbonate); "
            "    ID: special educational support; behavioural therapy; "
            "    No disease-modifying therapy for OCRL"
        ),
        "disease_category": (
            "OCRL-LOWE — CATARACTS-ID-FANCONI-TRIAD / DENT-2-INCOMPLETE: "
            "  DIAGNOSIS: congenital cataracts + hypotonia + LMW proteinuria = Lowe until proven otherwise; "
            "    Female carriers: slit-lamp lens opacities (>90%) = useful carrier screening; "
            "  TREATMENT: cataract surgery immediately; phosphate + calcitriol; alkali; edu support; "
            "  GENETIC TESTING: "
            "    OCRL sequencing; if negative → CLCN5 (Dent 1) or check Lowe-gene panel; "
            "    Dent 2 males: OCRL testing when CLCN5 negative"
        ),
        "disease_pathway": (
            "OCRL LOF → PI(4,5)P2 ACCUMULATION → ENDOSOMAL TRAFFICKING DEFECT → FANCONI + CATARACTS: "
            "  Normal OCRL phosphoinositide regulation: "
            "    Clathrin-coated vesicles require PI(4,5)P2 for budding at plasma membrane; "
            "    After vesicle formation, OCRL hydrolyses PI(4,5)P2 → PI(4)P; "
            "    PI(4,5)P2 removal allows vesicle to mature into early endosome; "
            "    Allows megalin-cubilin recycling (same as CLCN5 pathway but different step); "
            "  OCRL LOF: "
            "    PI(4,5)P2 not cleared → abnormal clathrin recycling; "
            "    Endosomal trafficking defect → LMW protein reabsorption impaired → LMW proteinuria; "
            "    Proximal tubule: generalised Fanconi (all transporters affected); "
            "    Lens: PI(4,5)P2 accumulates in lens epithelium → abnormal lens differentiation → cataracts; "
            "    Brain: vesicular trafficking in neurons impaired → ID/hypotonia"
        ),
    },
    {
        "gene": "SLC34A3",
        "protein": (
            "SLC34A3 -- 9q34.3 AR LOF -- 599aa -- Sodium-Phosphate-Cotransporter-IIc-"
            "NaPi-IIc-68kDa-Renal-Proximal-Tubule-SLC34-Family-"
            "HHRH-Hereditary-Hypophosphatemia-Hypercalciuria-Suppressed-PTH-HIGH-1,25D-PATHOGNOMONIC-OMIM-609826"
        ),
        "locus": "9q34.3",
        "protein_size": (
            "599 aa / 68 kDa (SLC34A3 — solute carrier family 34, member 3; NaPi-IIc; "
            "FUNCTION: sodium-phosphate cotransporter type IIc; "
            "  Apical membrane of proximal tubule S1/S2 segments; "
            "  Reabsorbs phosphate from glomerular filtrate (Na+-dependent cotransport); "
            "  Works in parallel with SLC34A1 (NaPi-IIa) — together account for >80% of tubular Pi reabsorption; "
            "  SLC34A3 expression regulated by PTH (downregulated) and FGF23 (downregulated); "
            "  SLC34A3 also exists in intestine (minor contribution to dietary Pi absorption); "
            "LOF CONSEQUENCE (HHRH = hereditary hypophosphatemic rickets with hypercalciuria): "
            "  PRIMARY TUBULAR PHOSPHATE WASTING (FGF23 INDEPENDENT): "
            "  Phosphate not reabsorbed → hypophosphataemia; "
            "  LOW phosphate → PTH SUPPRESSED (low phosphate reduces PTH release); "
            "  LOW phosphate → CYP27B1 STIMULATED → HIGH 1,25(OH)2D; "
            "  HIGH 1,25D → increased intestinal Ca absorption → hypercalciuria; "
            "  This profile is UNIQUE and PATHOGNOMONIC for HHRH; "
            "  encoded 9q34.3"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — SLC34A3 / HHRH (HEREDITARY HYPOPHOSPHATEMIC RICKETS WITH HYPERCALCIURIA): "
            "  BIOCHEMISTRY — PATHOGNOMONIC COMBINATION: "
            "    Serum phosphate: LOW; "
            "    TRP: LOW (tubular phosphate wasting); "
            "    FGF23: normal or LOW (phosphate wasting is FGF23-INDEPENDENT here); "
            "    1,25(OH)2D: HIGH (calcitriol elevated — opposite to XLH/DMP1/ENPP1); "
            "    PTH: SUPPRESSED (opposite to XLH); "
            "    Serum calcium: normal or slightly elevated; "
            "    Urinary calcium: HIGH (hypercalciuria) — 1,25D increases intestinal Ca absorption; "
            "  THIS PATTERN (LOW Pi + SUPPRESSED PTH + HIGH 1,25D + HYPERCALCIURIA) = "
            "    PATHOGNOMONIC for HHRH (SLC34A3 deficiency); "
            "    No other hereditary rickets gene gives this combination; "
            "  PHENOTYPE: "
            "    Rickets in childhood: bowing, short stature, bone pain (similar severity to XLH); "
            "    Nephrolithiasis + nephrocalcinosis (from hypercalciuria); "
            "    Risk of progressive CKD if calcium stones untreated; "
            "  HETEROZYGOUS CARRIERS: "
            "    May have isolated hypercalciuria + nephrolithiasis (without rickets); "
            "    Especially on high-calcium/phosphate diet; "
            "  KEY TREATMENT DIFFERENCE FROM XLH/DMP1: "
            "    XLH/DMP1: calcitriol needed (1,25D low → supplement); "
            "    HHRH: 1,25D already HIGH → DO NOT give calcitriol (worsens hypercalciuria); "
            "    HHRH treatment: PHOSPHATE SUPPLEMENTATION ALONE → normalises phosphate → "
            "      PTH rises, 1,25D normalises, hypercalciuria resolves; "
            "    Correct phosphate → 1,25D comes down → Ca absorption normalised → "
            "      nephrolithiasis risk decreases"
        ),
        "disease_category": (
            "SLC34A3-HHRH — SUPPRESSED-PTH-HIGH-1,25D-HYPERCALCIURIA-PATHOGNOMONIC: "
            "  DIAGNOSIS CLUE: rickets + low Pi + SUPPRESSED PTH + HIGH calcitriol + hypercalciuria; "
            "    This combination occurs in NO other hereditary rickets gene; "
            "  TREATMENT: "
            "    Phosphate supplementation ALONE (NOT calcitriol — already high); "
            "    Correct phosphate → PTH/1,25D/Ca normalise; nephrolithiasis resolves; "
            "    Neutral phosphate preferred (avoid Na-phosphate → hypernatraemia); "
            "  GENETIC TESTING: "
            "    SLC34A3 sequencing; "
            "    Heterozygous carriers: check for hypercalciuria + nephrolithiasis"
        ),
        "disease_pathway": (
            "SLC34A3 LOF → PRIMARY PHOSPHATE WASTING → LOW Pi DRIVES HIGH 1,25D + HYPERCALCIURIA: "
            "  Normal SLC34A3 function: "
            "    Phosphate filtered at glomerulus; NaPi-IIc (SLC34A3) + NaPi-IIa (SLC34A1) "
            "    co-transport phosphate back across apical membrane of proximal tubule; "
            "    80-90% of filtered phosphate normally reclaimed; "
            "  SLC34A3 LOF: "
            "    Tubular phosphate reabsorption reduced → hypophosphataemia; "
            "    LOW Pi → parathyroid gland senses low phosphate → PTH SUPPRESSED; "
            "    LOW Pi also stimulates CYP27B1 (1α-hydroxylase): "
            "      1,25(OH)2D markedly elevated → intestinal Ca absorption increased; "
            "    Elevated 1,25D + hypercalciuria → nephrolithiasis/nephrocalcinosis; "
            "    FGF23 low/normal (not driving the phosphate wasting); "
            "    Phosphate supplementation corrects the primary defect: "
            "      Pi normalised → CYP27B1 no longer stimulated → 1,25D falls → Ca absorption normalised"
        ),
    },
    {
        "gene": "CYP27B1",
        "protein": (
            "CYP27B1 -- 12q14.1 AR LOF -- 508aa -- 25-Hydroxyvitamin-D-1-Alpha-Hydroxylase-"
            "P450c1alpha-55kDa-Mitochondrial-CYP450-Renal-Proximal-Tubule-"
            "VDDR1-Vitamin-D-Dependent-Rickets-Type-1-Low-1,25D-Calcitriol-CURATIVE-OMIM-609506"
        ),
        "locus": "12q14.1",
        "protein_size": (
            "508 aa / 55 kDa (CYP27B1 — 25-hydroxyvitamin D 1-alpha-hydroxylase; P450c1α; "
            "FUNCTION: mitochondrial cytochrome P450 enzyme in renal proximal tubule cells; "
            "  Converts 25(OH)D (calcidiol) → 1,25(OH)2D (calcitriol = active vitamin D); "
            "  1,25(OH)2D acts on VDR in gut (Ca absorption), kidney (Ca/P reabsorption), bone; "
            "  Regulated by: "
            "    PTH: STIMULATES CYP27B1 → more 1,25D; "
            "    FGF23: INHIBITS CYP27B1 → less 1,25D; "
            "    Low phosphate: STIMULATES CYP27B1; "
            "    1,25D itself: NEGATIVE feedback (induces CYP24A1 which inactivates 1,25D); "
            "LOF CONSEQUENCE (VDDR1 = vitamin D-dependent rickets type 1): "
            "  Cannot convert 25(OH)D to 1,25(OH)2D; "
            "  Despite abundant 25(OH)D (sun exposure / vitamin D supplementation) → 1,25D ABSENT; "
            "  Serum: 25(OH)D normal/high + 1,25(OH)2D very low (THE diagnostic combination); "
            "  Clinical: severe rickets + hypocalcaemia + secondary hyperparathyroidism; "
            "  IMPORTANT: responds completely to calcitriol supplement (NOT vitamin D3 alone); "
            "  encoded 12q14.1"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — CYP27B1 / VDDR TYPE 1 (PSEUDOVITAMIN D DEFICIENCY RICKETS): "
            "  PHENOTYPE (onset first 2 years of life): "
            "    SEVERE RICKETS: "
            "      Rachitic changes: bowing, widened metaphyses, rachitic rosary, craniotabes; "
            "      Severity similar to or worse than nutritional vitamin D deficiency; "
            "    HYPOCALCAEMIA (often severe): "
            "      Can present with neonatal/infantile seizures (hypocalcaemic tetany); "
            "      Carpopedal spasm; muscle cramps; "
            "    SECONDARY HYPERPARATHYROIDISM: "
            "      PTH markedly elevated (response to hypocalcaemia + low 1,25D); "
            "    DENTAL: "
            "      Enamel hypoplasia; delayed dental eruption; caries; "
            "    BIOCHEMISTRY — DISTINCTIVE: "
            "      25(OH)D: NORMAL or HIGH (stored in liver normally); "
            "      1,25(OH)2D: VERY LOW or UNDETECTABLE (cannot be made without CYP27B1); "
            "      PTH: HIGH (secondary HPT); "
            "      Serum phosphate: low (secondary to low 1,25D impairing intestinal absorption); "
            "      ALP: markedly elevated; "
            "  KEY CLINICAL RULE — CALCITRIOL CURATIVE (NOT VITAMIN D3 ALONE): "
            "    The defect is in the ACTIVATION step (25D → 1,25D); "
            "    Vitamin D3 (cholecalciferol): provides 25(OH)D but cannot be activated → useless alone; "
            "    CALCITRIOL (1,25-dihydroxycholecalciferol) = already activated → bypasses the defect; "
            "    Response to calcitriol: DRAMATIC and COMPLETE within weeks; "
            "    Dosing: 0.25-2 mcg/day calcitriol; lifelong therapy usually required; "
            "  DISTINGUISH FROM NUTRITIONAL VITAMIN D DEFICIENCY: "
            "    Nutritional: 25(OH)D LOW; "
            "    VDDR1: 25(OH)D NORMAL/HIGH (paradox — cannot be activated); "
            "  DISTINGUISH FROM VDDR2 (VDR mutations): "
            "    VDDR2: 1,25D VERY HIGH (made but receptor absent → end-organ resistance); "
            "    VDDR2: alopecia totalis in severe cases (PATHOGNOMONIC for VDR LOF); "
            "    VDDR1: 1,25D VERY LOW (enzyme absent)"
        ),
        "disease_category": (
            "CYP27B1-VDDR1 — 25D-NORMAL-1,25D-LOW — CALCITRIOL-CURATIVE: "
            "  DIAGNOSIS CLUE: severe rickets + hypocalcaemia + 25(OH)D NORMAL + 1,25D UNDETECTABLE; "
            "    PTH high; phosphate low; ALP very high; responds dramatically to calcitriol; "
            "  TREATMENT: "
            "    Calcitriol (1,25-dihydroxyvitamin D): 0.25-2 mcg/day PO + calcium supplements; "
            "    Monitor: Ca, phosphate, ALP, PTH, urine Ca:Cr ratio every 3 months; "
            "    Calcitriol ALONE can cure; vitamin D3 alone is INEFFECTIVE; "
            "  GENETIC TESTING: "
            "    CYP27B1 sequencing; "
            "    Diagnose before empirical high-dose vitamin D (ineffective → toxicity risk)"
        ),
        "disease_pathway": (
            "CYP27B1 LOF → ABSENT 1,25D → IMPAIRED Ca/P ABSORPTION → SECONDARY HPT + RICKETS: "
            "  Normal CYP27B1 activation step: "
            "    Solar UV → skin → vitamin D3 (cholecalciferol); "
            "    Liver CYP2R1: vitamin D3 → 25(OH)D (calcidiol) — stored form; "
            "    Kidney CYP27B1: 25(OH)D → 1,25(OH)2D (calcitriol) — ACTIVE form; "
            "    1,25D binds VDR in gut → upregulates TRPV6 + calbindin → Ca absorption; "
            "    1,25D binds VDR in kidney → Ca reabsorption; "
            "    1,25D acts on bone + parathyroid gland (negative feedback on PTH); "
            "  CYP27B1 LOF: "
            "    25(OH)D cannot be converted → 1,25D absent; "
            "    Gut: no VDR activation → Ca/P not absorbed → Ca LOW; "
            "    Parathyroid: hypocalcaemia + absent 1,25D → PTH elevated (secondary HPT); "
            "    PTH stimulates osteoclasts → bone resorption → rickets worsens; "
            "    Calcitriol supplementation: bypasses CYP27B1 → directly activates VDR → cures disease"
        ),
    },
]


def _make_patients(seed: int, gene: str) -> list:
    rng = random.Random(seed)
    gene_profiles = {
        "PHEX":    dict(phosphate_low_pct=0.98, fgf23_high_pct=0.96, nephrocalc_pct=0.20,
                        lmw_prot_pct=0.02, low_1_25d_pct=0.90, suppressed_pth_pct=0.05),
        "FGF23":   dict(phosphate_low_pct=0.75, fgf23_high_pct=0.70, nephrocalc_pct=0.35,
                        lmw_prot_pct=0.03, low_1_25d_pct=0.72, suppressed_pth_pct=0.10),
        "DMP1":    dict(phosphate_low_pct=0.97, fgf23_high_pct=0.93, nephrocalc_pct=0.18,
                        lmw_prot_pct=0.02, low_1_25d_pct=0.88, suppressed_pth_pct=0.05),
        "ENPP1":   dict(phosphate_low_pct=0.82, fgf23_high_pct=0.78, nephrocalc_pct=0.52,
                        lmw_prot_pct=0.04, low_1_25d_pct=0.78, suppressed_pth_pct=0.08),
        "CLCN5":   dict(phosphate_low_pct=0.35, fgf23_high_pct=0.08, nephrocalc_pct=0.75,
                        lmw_prot_pct=0.98, low_1_25d_pct=0.30, suppressed_pth_pct=0.12),
        "OCRL":    dict(phosphate_low_pct=0.70, fgf23_high_pct=0.06, nephrocalc_pct=0.65,
                        lmw_prot_pct=0.97, low_1_25d_pct=0.65, suppressed_pth_pct=0.08),
        "SLC34A3": dict(phosphate_low_pct=0.99, fgf23_high_pct=0.04, nephrocalc_pct=0.82,
                        lmw_prot_pct=0.02, low_1_25d_pct=0.00, suppressed_pth_pct=0.95),
        "CYP27B1": dict(phosphate_low_pct=0.85, fgf23_high_pct=0.05, nephrocalc_pct=0.10,
                        lmw_prot_pct=0.02, low_1_25d_pct=0.99, suppressed_pth_pct=0.00),
    }
    p = gene_profiles.get(gene, gene_profiles["PHEX"])

    treatment_map = {
        "PHEX":    ["burosumab-sc", "phosphate+calcitriol", "burosumab+calcitriol"],
        "FGF23":   ["burosumab-sc", "phosphate+calcitriol+iron", "etidronate(GACI)"],
        "DMP1":    ["burosumab-sc", "phosphate+calcitriol", "phosphate+calcitriol+physio"],
        "ENPP1":   ["etidronate(GACI)+burosumab", "phosphate+calcitriol", "burosumab-sc"],
        "CLCN5":   ["thiazide+fluid", "thiazide+phosphate+calcitriol", "ACEi+thiazide+fluid"],
        "OCRL":    ["phosphate+calcitriol+alkali+cataract-surgery", "calcitriol+alkali", "cataract-surgery+phosphate"],
        "SLC34A3": ["neutral-phosphate-alone", "neutral-phosphate+fluid", "neutral-phosphate+low-Ca-diet"],
        "CYP27B1": ["calcitriol+Ca-supplements", "calcitriol-0.5mcg/day", "calcitriol-1mcg/day"],
    }
    treatments = treatment_map.get(gene, ["phosphate+calcitriol"])

    patients = []
    for i in range(40):
        phosphate_low = rng.random() < p["phosphate_low_pct"]
        fgf23_high = rng.random() < p["fgf23_high_pct"]
        nephrocalc = rng.random() < p["nephrocalc_pct"]
        lmw_prot = rng.random() < p["lmw_prot_pct"]
        low_1_25d = rng.random() < p["low_1_25d_pct"]
        suppressed_pth = rng.random() < p["suppressed_pth_pct"]
        serum_phosphate = round(rng.uniform(0.35, 0.75) if phosphate_low else rng.uniform(0.78, 1.2), 2)
        age_dx = rng.randint(0, 15) if gene not in ("CLCN5", "OCRL") else rng.randint(0, 25)
        treatment = rng.choice(treatments)
        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis": age_dx,
            "serum_phosphate_mmol_L": serum_phosphate,
            "phosphate_low": phosphate_low,
            "fgf23_high": fgf23_high,
            "nephrocalcinosis": nephrocalc,
            "lmw_proteinuria": lmw_prot,
            "low_1_25d": low_1_25d,
            "suppressed_pth": suppressed_pth,
            "treatment": treatment,
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Hypophosphatemic-Rickets-Atlas."""
    return {
        "atlas":       "Hereditary-Hypophosphatemic-Rickets-Atlas",
        "subtitle":    "Complete 8-Gene FGF23 / Phosphate-Wasting Rickets Reference Atlas",
        "total_genes": len(ATLAS_GENES),
        "seed_range":  f"{SEED_BASE}–{SEED_BASE+7}",
        "total_patients": 320,
        "genes": [g["gene"] for g in ATLAS_GENES],
        "gene_loci": {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "PHEX":    "XL LOF",
            "FGF23":   "AD GOF (ADHR) / AR LOF (Tumoral Calcinosis)",
            "DMP1":    "AR LOF",
            "ENPP1":   "AR LOF (ARHR2 + GACI)",
            "CLCN5":   "XL LOF",
            "OCRL":    "XL LOF",
            "SLC34A3": "AR LOF",
            "CYP27B1": "AR LOF",
        },
        "key_clinical_rules": [
            "PHEX-XLH-BUROSUMAB: iFGF23 HIGH + TRP LOW + 1,25D inappropriately normal — burosumab (anti-FGF23 mAb) CURATIVE, FDA 2018",
            "SLC34A3-HHRH-PATHOGNOMONIC: suppressed PTH + HIGH 1,25D + hypercalciuria = HHRH (no other HPR gene gives this) — phosphate ALONE treats it",
            "CYP27B1-VDDR1: 25(OH)D normal/HIGH + 1,25D very LOW — calcitriol CURES; vitamin D3 alone INEFFECTIVE (cannot be activated)",
            "FGF23-LOF-OPPOSITE: FGF23 AR LOF = HYPERPHOSPHATEMIA + periarticular calcifications (opposite of all other HPR genes)",
            "ENPP1-GACI-ETIDRONATE: neonatal arterial calcification — etidronate FIRST-LINE (dissolves calcification); ARHR2 in survivors",
            "CLCN5-LMW-PROTEINURIA-PATHOGNOMONIC: beta-2-microglobulin in urine marks tubular proteinuria — Dent 1 defining feature",
            "OCRL-LOWE-TRIAD: congenital cataracts + intellectual disability + Fanconi renal syndrome; Dent 2 = males without eye/brain",
            "DMP1-ENTHESOPATHY-ADULTS: spinal ligament calcification + periosteal reactions distinguish ARHR1 from XLH in adults",
            "HHRH-TREATMENT-PHOSPHATE-ONLY: do NOT give calcitriol in HHRH (1,25D already high) — phosphate alone normalises everything",
            "DENT-1-vs-DENT-2: CLCN5 vs OCRL biochemically identical — gene testing MANDATORY to distinguish",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for all 8 HPR genes."""
    genes_data = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        genes_data.append({
            "gene":              g["gene"],
            "locus":             g["locus"],
            "protein":           g["protein"],
            "protein_size":      g["protein_size"],
            "inheritance":       g["inheritance"],
            "disease_category":  g["disease_category"],
            "disease_pathway":   g["disease_pathway"],
            "n_patients":        len(pts),
            "mean_age_dx":       round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1),
            "mean_phosphate":    round(sum(p["serum_phosphate_mmol_L"] for p in pts) / len(pts), 2),
            "phosphate_low_pct": round(100 * sum(1 for p in pts if p["phosphate_low"]) / len(pts), 1),
            "fgf23_high_pct":    round(100 * sum(1 for p in pts if p["fgf23_high"]) / len(pts), 1),
            "nephrocalc_pct":    round(100 * sum(1 for p in pts if p["nephrocalcinosis"]) / len(pts), 1),
            "lmw_prot_pct":      round(100 * sum(1 for p in pts if p["lmw_proteinuria"]) / len(pts), 1),
            "low_1_25d_pct":     round(100 * sum(1 for p in pts if p["low_1_25d"]) / len(pts), 1),
            "suppressed_pth_pct": round(100 * sum(1 for p in pts if p["suppressed_pth"]) / len(pts), 1),
            "treatment_breakdown": treatments,
            "patients":          pts,
        })
    return {
        "atlas":  "Hereditary-Hypophosphatemic-Rickets-Atlas",
        "count":  len(genes_data),
        "genes":  genes_data,
    }


def generate_definitions() -> dict:
    """Key clinical terms for Hereditary-Hypophosphatemic-Rickets-Atlas."""
    definitions = [
        {
            "term": "PHEX (XLH) — Most Common Hereditary Rickets — Burosumab CURATIVE",
            "genes": ["PHEX"],
            "definition": (
                "X-LINKED HYPOPHOSPHATEMIA (XLH): PHEX LOF → FGF23 not cleaved → chronic phosphaturia. "
                "Prevalence 1:20,000 — most common hereditary form of rickets. "
                "BIOCHEMISTRY: serum phosphate LOW; TRP LOW (<85%); iFGF23 MARKEDLY HIGH; "
                "1,25(OH)2D normal/low (INAPPROPRIATELY NORMAL for degree of hypophosphataemia — "
                "FGF23 inhibits CYP27B1 so calcitriol cannot compensate). "
                "PTH: normal. "
                "CLINICAL: childhood rickets + short stature + spontaneous dental abscesses "
                "(hypomineralised dentine → bacterial access without caries — PATHOGNOMONIC). "
                "Adult: bone pain, enthesopathy, fatigue, hearing loss. "
                "X-LINKED: males fully affected; females variably affected (X-inactivation). "
                "TREATMENT: "
                "BUROSUMAB (anti-FGF23 mAb, KRN23): FDA approved 2018 for children ≥1 year; "
                "Direct FGF23 neutralisation → restores phosphate reabsorption; "
                "Dramatically improves rickets radiographically within 6 months; "
                "Normalises growth velocity; "
                "REPLACE old therapy (oral phosphate + calcitriol) with burosumab when available. "
                "Old therapy: oral phosphate 4-6×/day + calcitriol — risk: nephrocalcinosis + secondary HPT."
            ),
        },
        {
            "term": "SLC34A3 (HHRH) — Pathognomonic Biochemistry: Suppressed PTH + HIGH 1,25D + Hypercalciuria",
            "genes": ["SLC34A3"],
            "definition": (
                "HEREDITARY HYPOPHOSPHATEMIC RICKETS WITH HYPERCALCIURIA (HHRH): "
                "PATHOGNOMONIC COMBINATION (no other HPR gene produces this): "
                "  Low serum phosphate + LOW TRP + "
                "  SUPPRESSED PTH (low phosphate suppresses PTH) + "
                "  HIGH 1,25(OH)2D (low phosphate stimulates CYP27B1) + "
                "  HYPERCALCIURIA (high 1,25D increases intestinal Ca absorption → urinary Ca high). "
                "FGF23: normal or LOW (phosphate wasting is FGF23-INDEPENDENT). "
                "MECHANISM: SLC34A3 (NaPi-IIc) absent → phosphate not reabsorbed → "
                "  LOW Pi → CYP27B1 stimulated → HIGH 1,25D; "
                "  LOW Pi → PTH suppressed; HIGH 1,25D → hypercalciuria → nephrolithiasis risk. "
                "CRITICAL TREATMENT RULE: "
                "  DO NOT give calcitriol (already high — worsens hypercalciuria + nephrolithiasis); "
                "  Give PHOSPHATE SUPPLEMENTATION ALONE: "
                "    Phosphate normalised → CYP27B1 no longer stimulated → 1,25D falls → "
                "    Ca absorption normalised → nephrolithiasis risk decreases. "
                "HETEROZYGOUS CARRIERS: isolated hypercalciuria + nephrolithiasis (no rickets). "
                "Genetic testing: SLC34A3 sequencing."
            ),
        },
        {
            "term": "CYP27B1 (VDDR1) — Low 1,25D with Normal 25D — Calcitriol Curative",
            "genes": ["CYP27B1"],
            "definition": (
                "VITAMIN D-DEPENDENT RICKETS TYPE 1 (VDDR1 / pseudovitamin D deficiency): "
                "CYP27B1 (1α-hydroxylase) LOF → cannot convert 25(OH)D → 1,25(OH)2D. "
                "BIOCHEMISTRY (DISTINCTIVE): "
                "  25(OH)D: NORMAL or HIGH (liver hydroxylation intact; sun/diet provides 25-OH-D); "
                "  1,25(OH)2D: VERY LOW or UNDETECTABLE (enzyme absent). "
                "  PTH: HIGH (secondary HPT from hypocalcaemia + absent VDR activation). "
                "  Serum phosphate: LOW (secondary). "
                "CLINICAL: severe rickets in first 2 years; hypocalcaemic seizures/tetany; "
                "enamel hypoplasia; secondary HPT. "
                "KEY TREATMENT RULE: "
                "  Vitamin D3 (cholecalciferol): INEFFECTIVE (cannot be activated without CYP27B1); "
                "  CALCITRIOL (1,25-dihydroxyvitamin D): CURATIVE — bypasses defective enzyme; "
                "  Dose: 0.25-2 mcg/day + calcium supplements; "
                "  Response: DRAMATIC within weeks — rickets heals, PTH normalises, Ca normalises. "
                "DISTINGUISH FROM VDDR2 (VDR LOF): "
                "  VDDR1: 1,25D LOW; VDDR2: 1,25D VERY HIGH; "
                "  VDDR2: alopecia totalis (PATHOGNOMONIC when severe — not seen in VDDR1)."
            ),
        },
        {
            "term": "FGF23 — GOF (ADHR): Intermittent Hypophosphataemia vs LOF (Tumoral Calcinosis): Hyperphosphataemia",
            "genes": ["FGF23"],
            "definition": (
                "FGF23 GOF (ADHR — AUTOSOMAL DOMINANT HYPOPHOSPHATEMIC RICKETS): "
                "  Mutations in RXXR cleavage site (R176Q, R179W/Q) → FGF23 resistant to proteolysis; "
                "  iFGF23 accumulates → same phosphaturia as XLH; AD inheritance. "
                "  KEY DIFFERENCE FROM XLH: INTERMITTENT disease — can remit then relapse; "
                "  Flares triggered by iron deficiency, pregnancy, puberty; "
                "  Iron supplementation: iron required for FGF23 cleavage → treat iron deficiency; "
                "  Treatment: burosumab + iron supplementation. "
                "FGF23 LOF (TUMORAL CALCINOSIS / FAMILIAL HYPERPHOSPHATAEMIC TUMORAL CALCINOSIS): "
                "  FGF23 absent → OPPOSITE biochemistry: HYPERPHOSPHATAEMIA + HIGH 1,25D; "
                "  PATHOGNOMONIC: massive periarticular calcium-phosphate crystal deposits; "
                "  Hips, elbows, shoulders: dense calcified masses on imaging; "
                "  NOT true tumours: calcium-phosphate crystal aggregates; "
                "  Treatment: low-phosphate diet + phosphate binders + acetazolamide; "
                "  Surgery for refractory/disabling deposits. "
                "KEY RULE: FGF23 GOF vs LOF = opposite phenotypes on same gene: "
                "  GOF: phosphate LOW (phosphaturia); "
                "  LOF: phosphate HIGH (hyperphosphataemia)."
            ),
        },
        {
            "term": "ENPP1 — GACI (Neonatal Arterial Calcification) + ARHR2 — Etidronate FIRST-LINE",
            "genes": ["ENPP1"],
            "definition": (
                "ENPP1 LOF: pyrophosphate (PPi) deficiency → dual age-dependent phenotype. "
                "GACI (GENERALISED ARTERIAL CALCIFICATION OF INFANCY): "
                "  Presents at birth or in utero: massive calcification of coronary, aortic, "
                "  pulmonary arteries → stenosis → heart failure → 50% mortality if untreated. "
                "  Imaging: chalky-white periarticular arterial calcification on X-ray/echo. "
                "  TREATMENT: ETIDRONATE (bisphosphonate = PPi analogue): "
                "    FIRST-LINE URGENT — can dissolve existing arterial calcification; "
                "    Dramatic reversal possible; start immediately; "
                "    IV sodium PPi: investigational adjunct. "
                "ARHR2 (survivors of GACI or milder ENPP1 mutations): "
                "  FGF23 elevated → hypophosphataemic rickets (same biochemistry as XLH); "
                "  Treatment: burosumab or phosphate + calcitriol. "
                "CALCIFICATION PARADOX: "
                "  Same ENPP1 mutation → vessels calcify + bones decalcify (rickets); "
                "  VESSELS: calcium-phosphate supersaturation → deposits; "
                "  BONE: insufficient calcium for mineralisation → rickets/osteomalacia. "
                "GENETIC TESTING: biallelic ENPP1; also check ABCC6 (GACI type 2)."
            ),
        },
        {
            "term": "CLCN5 (Dent 1) and OCRL (Dent 2 / Lowe) — LMW Proteinuria as Cardinal Feature",
            "genes": ["CLCN5", "OCRL"],
            "definition": (
                "DENT DISEASE SPECTRUM (CLCN5 and OCRL): "
                "CARDINAL BIOCHEMICAL TRIAD (both Dent 1 and 2): "
                "  1. LOW MOLECULAR WEIGHT PROTEINURIA: "
                "     β2-microglobulin in urine — PATHOGNOMONIC for tubular (not glomerular) origin; "
                "     β2-MG >1 mg/L OR >0.3 mg/mmol Cr = tubular proteinuria confirmed; "
                "  2. Hypercalciuria (>4 mg/kg/day); "
                "  3. At least one: nephrocalcinosis / nephrolithiasis / CKD / rickets. "
                "DENT 1 (CLCN5, Xp11.23): X-linked; "
                "  CLC-5 (H+/Cl- exchanger) absent → endosomes not acidified → "
                "  megalin-cubilin recycling impaired → LMW proteins not reabsorbed. "
                "  Males fully affected; progressive CKD (30-80% reach ESRD by 30-50 years). "
                "DENT 2 = LOWE SYNDROME (OCRL, Xq26.1): X-linked; "
                "  COMPLETE LOWE (full triad): congenital cataracts + intellectual disability + Fanconi; "
                "  INCOMPLETE LOWE (Dent 2): males with only renal Dent phenotype, no eye/brain. "
                "  Female OCRL carriers: slit-lamp lens opacities (punctate) in >90% — CARRIER SIGN. "
                "KEY RULE: Dent 1 and Dent 2 biochemically IDENTICAL — gene testing MANDATORY; "
                "TREATMENT (both): thiazide diuretics (hypercalciuria); "
                "phosphate + calcitriol if rickets; high fluid intake; ACEi for CKD."
            ),
        },
        {
            "term": "DMP1 (ARHR1) — Enthesopathy Distinguishes from XLH in Adults",
            "genes": ["DMP1"],
            "definition": (
                "ARHR TYPE 1 (DMP1 LOF): "
                "BIOCHEMISTRY: identical to XLH (PHEX LOF): "
                "  Low phosphate; low TRP; high iFGF23; low/normal 1,25D; normal PTH. "
                "DISTINGUISHES FROM XLH: "
                "  INHERITANCE: autosomal recessive (not X-linked); "
                "  ENTHESOPATHY — DISTINCTIVE ADULT FEATURE: "
                "    Calcification of tendons and ligaments at bony insertions (entheses); "
                "    Spinal ligament calcification: restricted spinal mobility; "
                "    Resembles ankylosing spondylitis on imaging (but seronegative); "
                "    Periosteal reactions along diaphyses; "
                "  Dentinogenesis imperfecta: opalescent teeth; "
                "  Onset often in childhood (similar to XLH). "
                "PATHOMECHANISM: DMP1 absent → osteocytes immature → FGF23 overproduction → "
                "  phosphaturia + low 1,25D → same downstream pathway as PHEX. "
                "TREATMENT: burosumab (anti-FGF23) equally effective as in XLH; "
                "  alternatively phosphate + calcitriol. "
                "FOUNDER MUTATION: c.1484delT in some populations; "
                "  Screen by sequencing if PHEX-negative XLH-like phenotype."
            ),
        },
        {
            "term": "Hereditary Rickets Differential — 8-Gene Biochemical Guide",
            "genes": ["PHEX", "FGF23", "DMP1", "ENPP1", "CLCN5", "OCRL", "SLC34A3", "CYP27B1"],
            "definition": (
                "PHOSPHATE WASTING RICKETS — DIFFERENTIAL BIOCHEMISTRY: "
                "PHEX (XLH): Pi LOW + TRP LOW + iFGF23 HIGH + 1,25D inappropriately NORMAL + PTH normal; "
                "FGF23 GOF (ADHR): same as XLH but AD + intermittent; iron deficiency triggers; "
                "DMP1 (ARHR1): same as XLH but AR + enthesopathy; "
                "ENPP1 (ARHR2): same as XLH but AR + GACI in neonate; PPi absent; etidronate; "
                "CLCN5 (Dent 1): LMW proteinuria + hypercalciuria + nephrocalcinosis + variable Pi LOW; "
                "OCRL (Dent 2/Lowe): LMW proteinuria + cataracts + ID + Fanconi; "
                "SLC34A3 (HHRH): Pi LOW + TRP LOW + FGF23 NORMAL/LOW + 1,25D HIGH + PTH SUPPRESSED + HYPERCALCIURIA; "
                "CYP27B1 (VDDR1): 25D normal/HIGH + 1,25D VERY LOW + PTH HIGH; calcitriol CURATIVE; "
                "FGF23 AR LOF (Tumoral Calcinosis): Pi HIGH + 1,25D HIGH + periarticular calcifications; "
                "TARGETED THERAPY SUMMARY: "
                "Burosumab: PHEX + FGF23 GOF + DMP1 + ENPP1 (anti-FGF23, FDA 2018); "
                "Calcitriol: CYP27B1 VDDR1 (replaces deficient 1,25D); "
                "Etidronate: ENPP1 GACI (PPi analogue, dissolves arterial calcification); "
                "Phosphate alone: SLC34A3 HHRH (calcitriol worsens hypercalciuria); "
                "Thiazide: CLCN5/OCRL Dent (reduces hypercalciuria + nephrolithiasis)."
            ),
        },
    ]
    return {
        "atlas":  "Hereditary-Hypophosphatemic-Rickets-Atlas",
        "count":  len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:600])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Terms: {df['count']}")
