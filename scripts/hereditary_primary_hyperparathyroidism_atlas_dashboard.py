#!/usr/bin/env python3
"""Hereditary-Primary-Hyperparathyroidism-Atlas — Complete 8-Gene Hereditary PHPT / FHH / HPT-JT / MEN Atlas
MEN1    (menin; 610 aa; 11q13.1; AD LOF;
         Most common hereditary PHPT; MEN1 = PHPT+pituitary+pancreatic NET; multiglandular;
         parathyroid carcinoma RARE (<1%); subtotal parathyroidectomy; annual Ca2+ from age 8;
         seed SEED_BASE+0) ·
CASR    (CaSR; 1078 aa; 3q13.3; AD LOF = FHH1 / biallelic LOF = NSHPT;
         Urine Ca:Cr clearance ratio (CCCR) <0.01 PATHOGNOMONIC FHH1; BENIGN — NEVER operate FHH;
         NSHPT = neonatal life-threatening hypercalcemia → emergency total parathyroidectomy;
         seed SEED_BASE+1) ·
CDC73   (parafibromin; 531 aa; 1q31.2; AD LOF = HPT-JT;
         Parathyroid CARCINOMA 15-20% risk PATHOGNOMONIC HPT-JT; ossifying jaw fibroma 50%;
         parafibromin IHC loss = carcinoma marker; uterine fibroids 75% females;
         seed SEED_BASE+2) ·
GNA11   (Galpha11; 359 aa; 19p13.3; AD LOF = FHH2;
         Phenotypically identical to FHH1; low CCCR; benign; GNA11 GOF = ADH2 (opposite);
         seed SEED_BASE+3) ·
AP2S1   (AP2 sigma-1; 142 aa; 19q13.32; AD LOF = FHH3;
         Highest serum calcium of all FHH types; cognitive impairment subset; benign — observe;
         seed SEED_BASE+4) ·
RET     (RET RTK; 1114 aa; 10q11.21; AD GOF = MEN2A;
         MEN2A PHPT risk 10-20%; milder than MEN1; PHEO excluded BEFORE any surgery;
         C634 codons highest PHPT risk; MEN2B virtually absent PHPT (key distinction);
         seed SEED_BASE+5) ·
CDKN1B  (p27Kip1; 196 aa; 12p13.1; AD LOF = MEN4;
         MEN1-like without menin; PHPT + pituitary (often ACTH); ~2-3% MEN1-like cases;
         seed SEED_BASE+6) ·
GCM2    (GCM2; 495 aa; 6p24.2; AD GOF = FIHPT;
         SAME GENE opposite direction: GCM2 LOF = hypoparathyroidism; GCM2 GOF = FIHPT;
         Master parathyroid TF; GOF → gland hyperplasia; isolated PHPT; multiglandular common;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2950–2957)
"""
import random

SEED_BASE = 2950

ATLAS_GENES = [
    {
        "gene": "MEN1",
        "protein": (
            "MEN1 -- 11q13.1 AD LOF -- 610aa -- Menin-"
            "70kDa-Transcriptional-Regulator-Histone-H3K4-Methyltransferase-"
            "MEN1-Triad-PHPT+Pituitary+Pancreatic-NET-OMIM-131100"
        ),
        "locus": "11q13.1",
        "protein_size": (
            "610 aa / 70 kDa (MEN1 — menin; tumour suppressor; "
            "FUNCTION: scaffold for histone H3K4 methyltransferase complex (MLL1/MLL2); "
            "  transcriptional co-repressor; cell-cycle regulation via CDKIs (p21, p27); "
            "  regulates JunD, NF-κB, TGF-β signalling; "
            "LOF CONSEQUENCE (haploinsufficiency + second somatic hit Knudson model): "
            "  Parathyroid glands: chief-cell hyperplasia or adenoma (multiglandular); "
            "  Anterior pituitary: prolactinoma (most common), ACTH-oma, GHoma, NFoma; "
            "  Pancreatic/duodenal NET: gastrinoma (ZES), insulinoma, glucagonoma, VIPoma; "
            "  Adrenal cortical tumours: 20-40% benign non-functional; "
            "GENE ARCHITECTURE: "
            "  10 exons; heterozygous germline LOF in all MEN1 families; "
            "  Founder mutations: Italian (c.784-9del16, p.Leu220ValfsX1), French, Finnish; "
            "  >1500 germline variants known (missense, nonsense, frameshift, splice); "
            "  encoded 11q13.1"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) LOF — MEN1 (Multiple Endocrine Neoplasia type 1): "
            "  PHPT phenotype: "
            "    >90% lifetime risk (most penetrant MEN1 component); "
            "    Usually presents 3rd-4th decade (earlier than sporadic PHPT); "
            "    Multiglandular (3-4 glands involved in ~85%); rarely single adenoma; "
            "    Serum calcium elevated (usually 2.6-3.2 mmol/L); PTH elevated; "
            "    Urine calcium high (unlike FHH); CCCR typically >0.02; "
            "    Parathyroid carcinoma: extremely rare in MEN1 (<1%); "
            "  MEN1 surveillance: "
            "    Annual serum calcium + PTH from age 8; "
            "    Annual serum prolactin + IGF-1 from age 8; "
            "    MRI pituitary every 3-5 years from age 8; "
            "    Annual fasting gastrin + glucagon + VIP + PP from age 8; "
            "    CT/MRI pancreas + abdomen every 1-2 years from age 20; "
            "  TREATMENT: "
            "    Subtotal parathyroidectomy (3.5 glands) or total + autograft; "
            "    Early surgery recommended (persistent hypercalcaemia); "
            "    High recurrence rate (15-20% at 10 years) — multiglandular; "
            "    Cinacalcet: temporising bridge, not definitive; "
            "PHENOTYPIC VARIABILITY: "
            "    Even within same family, manifestations differ; "
            "    Incomplete penetrance in very young (<20 years)"
        ),
        "disease_category": (
            "MEN1 HEREDITARY PHPT — MULTIGLANDULAR PARATHYROID HYPERPLASIA/ADENOMA: "
            "  Most common hereditary PHPT syndrome; "
            "  Autosomal dominant; 50% offspring risk; "
            "  PHPT is the most penetrant component: >90% affected by age 50; "
            "  KEY DISTINGUISHER FROM SPORADIC: "
            "    Young age (<45 years); multiglandular disease; family history; "
            "    Concurrent pituitary or pancreatic NET; "
            "  SURGICAL PLANNING: "
            "    4D-CT / sestamibi scan limited value (multiglandular → all glands affected); "
            "    Bilateral exploration mandatory; "
            "    Retain ≥0.5 gland (autograft to forearm for re-exploration access); "
            "  RECURRENCE RISK: "
            "    15-20% at 10 years (monomorphic cell proliferation continues); "
            "    Renin-angiotensin axis intact; eucalcaemic window often 5-10 years; "
            "  GENETIC TESTING: "
            "    Offer to all first-degree relatives; "
            "    MEN1-panel: sequencing + MLPA for exon deletions"
        ),
        "disease_pathway": (
            "MEN1 LOF → MENIN LOSS → DISINHIBITION OF PARATHYROID CELL PROLIFERATION: "
            "  Normal: menin assembles MLL-WDR5-RBBP5 complex → H3K4me3 at CDKI loci (p21, p27); "
            "    p21 (CDKN1A) + p27 (CDKN1B) → inhibit CDK4/6 → suppress G1→S cell cycle; "
            "  MEN1 LOF: "
            "    H3K4me3 falls at CDKI promoters → p21/p27 expression drops; "
            "    CDK4/6 released → RB hyperphosphorylation → E2F activation; "
            "    Parathyroid chief cells enter uncontrolled proliferation; "
            "    Somatic second hit (LOH 11q13) inactivates remaining allele → clonal expansion; "
            "    Result: chief-cell hyperplasia → adenoma → excess PTH synthesis/secretion; "
            "  PTH effect: "
            "    Excess PTH → ↑ osteoclast activity (bone resorption → Ca²⁺ + PO₄³⁻ released); "
            "    ↑ renal Ca²⁺ reabsorption (DCT); ↑ 1α-hydroxylase (calcitriol); "
            "    ↑ intestinal Ca²⁺ absorption → hypercalcaemia; "
            "    Hypercalcaemia → polyuria/polydipsia, nephrolithiasis, pancreatitis, neuropsychiatric"
        ),
    },
    {
        "gene": "CASR",
        "protein": (
            "CASR -- 3q13.3 AD LOF FHH1 / biallelic LOF NSHPT -- 1078aa -- CaSR-"
            "120kDa-GPCR-Class-C-Ca2+-Sensing-Receptor-FHH1-Benign-CCCR-lt0.01-PATHOGNOMONIC-"
            "NSHPT-Life-Threatening-Neonatal-Emergency-OMIM-145980"
        ),
        "locus": "3q13.3",
        "protein_size": (
            "1078 aa / 120 kDa (CASR — calcium-sensing receptor; GPCR class C; "
            "FUNCTION: detects extracellular Ca²⁺ → Gi/Gq coupling → inhibits PTH secretion; "
            "  expressed on parathyroid chief cells + thick ascending limb of Henle (TAL); "
            "  TAL: Ca²⁺ sensing → inhibits ROMK + NKCC2 → reduces paracellular Ca²⁺ reabsorption; "
            "HETEROZYGOUS LOF = FHH1: "
            "  Right-shift of Ca²⁺ set-point → parathyroids 'see' normocalcemia as hypocalcemia; "
            "  Mildly elevated Ca²⁺ (2.6-2.9 mmol/L); normal/mildly elevated PTH; "
            "  KEY: urine Ca²⁺ low (TAL reabsorbs Ca²⁺ inappropriately); "
            "  CCCR = (urine Ca/serum Ca) ÷ (urine Cr/serum Cr) <0.01 PATHOGNOMONIC FHH; "
            "BIALLELIC LOF = NSHPT: "
            "  Neonatal severe hyperparathyroidism — life-threatening; "
            "  Serum Ca²⁺ often >3.5-4.0 mmol/L; PTH massively elevated; "
            "  Hypotonia, respiratory failure, skeletal demineralisation; "
            "  Emergency total parathyroidectomy; "
            "encoded 3q13.3"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) LOF — FHH1 (Familial Hypocalciuric Hypercalcemia type 1): "
            "  Incidence: ~1/78,000; accounts for ~2-3% of all hypercalcaemic patients; "
            "  Biochemistry: Ca²⁺ mildly elevated (2.55-2.90 mmol/L typical); PTH normal or mildly elevated; "
            "    Magnesium also mildly elevated (subtle but consistent clue); "
            "    Phosphate: normal or mildly low; "
            "  CCCR formula: (uCa × sCr) ÷ (uCr × sCa) <0.01 = FHH (strong evidence); "
            "    0.01-0.02 = borderline — consider genetic testing; >0.02 = primary HPT; "
            "  CLINICAL IMPORTANCE: "
            "    FHH1 ALMOST NEVER requires surgery — misdiagnosed as sporadic PHPT; "
            "    Parathyroidectomy does NOT normalise calcium in FHH (set-point remains shifted); "
            "    Cases of unnecessary parathyroidectomies without resolution are well documented; "
            "  BIALLELIC LOF (NSHPT): "
            "    Both parents typically FHH1 carriers; "
            "    Neonate: markedly elevated Ca²⁺, hypotonia, respiratory distress, skeletal fractures; "
            "    EMERGENCY: calcimimetics (cinacalcet) may temporise but total parathyroidectomy usually required; "
            "  GENOTYPE-PHENOTYPE: "
            "    ECD mutations: milder (less efficient Ca²⁺ binding impairment); "
            "    TMD/ICD mutations: more severe NSHPT when biallelic; "
            "AUTOSOMAL DOMINANT (AD) GOF — ADH1 (Autosomal Dominant Hypocalcaemia type 1): "
            "    Covered in Hereditary-Hypoparathyroidism-Atlas (opposite phenotype of FHH1)"
        ),
        "disease_category": (
            "FHH1 — FAMILIAL HYPOCALCIURIC HYPERCALCEMIA TYPE 1 (BENIGN): "
            "  CRITICAL CLINICAL PITFALL: DO NOT OPERATE FHH — surgery ineffective and causes permanent hypoparathyroidism; "
            "  Differentiation from PHPT: "
            "    CCCR <0.01: FHH (specific); CCCR >0.02: PHPT; "
            "    Family history of hypercalcaemia (often found in multiple generations); "
            "    Normal or mildly elevated PTH (not massively elevated like symptomatic PHPT); "
            "    No nephrolithiasis, no bone disease typically (asymptomatic for decades); "
            "  NSHPT EMERGENCY RECOGNITION: "
            "    Neonate + severe hypercalcaemia + PTH markedly elevated + parents with mild hypercalcaemia; "
            "    Do not wait for genetic confirmation — act clinically; "
            "    IV fluids + furosemide + calcitonin/bisphosphonate temporise → surgical planning; "
            "  MONITORING FHH1: "
            "    Reassure patient; annual calcium + renal function; "
            "    Screen first-degree relatives for hypercalcaemia; "
            "    Avoid thiazide diuretics (worsen hypercalcaemia in FHH)"
        ),
        "disease_pathway": (
            "CASR LOF → Ca²⁺ SET-POINT RIGHT-SHIFT → INAPPROPRIATE PTH SECRETION + RENAL Ca²⁺ RETENTION: "
            "  Normal CaSR function: "
            "    Normal serum Ca²⁺ (2.1-2.6 mmol/L) activates CaSR → Gi coupling → ↓ cAMP → ↓ PTH secretion; "
            "    TAL CaSR activation → ↓ ROMK/NKCC2 → ↓ paracellular Ca²⁺ reabsorption → calciuria; "
            "  CASR LOF: "
            "    CaSR requires higher Ca²⁺ to activate (right-shifted set-point); "
            "    At serum Ca²⁺ 2.7 mmol/L, parathyroid 'senses' this as normal → PTH secretion continues; "
            "    TAL CaSR also LOF → ROMK + NKCC2 active → Ca²⁺ reabsorbed → URINE Ca²⁺ LOW; "
            "    Consequence: hypercalcaemia + hypocalciuria (the hallmark combination); "
            "  NSHPT (biallelic): "
            "    No functional CaSR anywhere → PTH secretion completely unsuppressed at any Ca²⁺; "
            "    Massive PTH → severe hypercalcaemia → multiorgan failure without treatment"
        ),
    },
    {
        "gene": "CDC73",
        "protein": (
            "CDC73 -- 1q31.2 AD LOF -- 531aa -- Parafibromin-"
            "60kDa-PAF1-Complex-Subunit-HPT-JT-PARATHYROID-CARCINOMA-15pct-"
            "Ossifying-Jaw-Fibroma-50pct-PATHOGNOMONIC-IHC-Loss-Carcinoma-Marker-OMIM-145001"
        ),
        "locus": "1q31.2",
        "protein_size": (
            "531 aa / 60 kDa (CDC73 — cell division cycle 73; parafibromin protein; "
            "FUNCTION: subunit of RNA polymerase II-associated PAF1 complex; "
            "  regulates histone H3K4 and H3K36 methylation; transcriptional elongation; "
            "  inhibits Wnt/β-catenin; promotes apoptosis; negative regulator of cell cycle; "
            "LOF CONSEQUENCE: "
            "  Loss of parafibromin → uncontrolled parathyroid cell proliferation; "
            "  CARCINOMA RISK: 15-20% in HPT-JT (vs <1% in MEN1, ~0.5% sporadic PHPT); "
            "  Jaw ossifying fibromas (benign but locally aggressive); "
            "  Uterine tumours (fibroids/fibromas, occasionally leiomyoma); "
            "  Renal abnormalities: hamartomas, cysts, Wilms-precursor lesions; "
            "PARAFIBROMIN IHC: "
            "  Retained (nuclear staining) in normal parathyroid + adenoma; "
            "  LOST in CDC73-mutant carcinoma and most CDC73-related adenomas; "
            "  IHC loss: highly specific marker for CDC73 defect / malignancy; "
            "encoded 1q31.2 (tumour suppressor)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) LOF — HPT-JT (Hyperparathyroidism-Jaw Tumor Syndrome): "
            "  PHPT phenotype: "
            "    Near-universal (>90% lifetime PHPT risk); often presents 3rd-5th decade; "
            "    Usually single large adenoma (unlike MEN1 multiglandular pattern); "
            "    SEVERE HYPERCALCAEMIA: Ca²⁺ often 3.0-3.8 mmol/L (higher than FHH, MEN1); "
            "    PTH markedly elevated (often >5-10× ULN); "
            "    High urine calcium; nephrolithiasis common; osteitis fibrosa cystica in severe cases; "
            "  CARCINOMA: "
            "    15-20% develop parathyroid carcinoma (key distinction from all other hereditary PHPT); "
            "    Carcinoma features: palpable neck mass, very high Ca²⁺ (>3.5 mmol/L), very high PTH; "
            "    En-bloc resection mandatory for carcinoma (inadvertent capsule breach → seeding); "
            "    Carcinoma recurs in 50-60% (local + distant); denosumab/cinacalcet for recurrence; "
            "  JAW FIBROMAS: "
            "    Ossifying fibromas of mandible/maxilla (50% of HPT-JT); "
            "    Orthopantomogram (OPG) mandatory in all CDC73 mutation carriers; "
            "    Radiologic: radio-opaque (ossifying) lesions — contrast to radiolucent giant cell lesions; "
            "  UTERINE TUMOURS: "
            "    Fibroids: ~75% of female CDC73 carriers; "
            "  RENAL SURVEILLANCE: "
            "    Annual renal ultrasound (hamartomas, Wilms precursor)"
        ),
        "disease_category": (
            "HPT-JT — HYPERPARATHYROIDISM-JAW TUMOR SYNDROME — PARATHYROID CARCINOMA RISK: "
            "  MOST IMPORTANT HEREDITARY PHPT GENE FOR MALIGNANCY RISK; "
            "  Do not miss CDC73: parathyroid carcinoma = poor prognosis without early identification; "
            "  SURGICAL APPROACH: "
            "    En-bloc resection (thyroid lobe + soft tissue + parathyroid) for suspected carcinoma; "
            "    NEVER rupture capsule — capsule breach seeds neck → recurrence; "
            "    Frozen section unreliable to diagnose carcinoma → treat suspicious cases as carcinoma; "
            "  POST-OPERATIVE: "
            "    Ca²⁺ + PTH every 3-6 months; if rising PTH = recurrence; "
            "    18F-choline PET: best imaging for recurrent/metastatic CDC73 carcinoma; "
            "    Cinacalcet: reduces Ca²⁺ in inoperable recurrence; "
            "    Denosumab: bone protection in skeletal disease; "
            "  GENETIC TESTING: "
            "    All first-degree relatives of HPT-JT patients; "
            "    Young PHPT + OPG ossifying fibroma + high Ca²⁺/PTH = CDC73 until proven otherwise"
        ),
        "disease_pathway": (
            "CDC73 LOF → PARAFIBROMIN LOSS → UNRESTRAINED PARATHYROID CELL GROWTH: "
            "  Normal parafibromin function: "
            "    Assembled in PAF1 complex (PAF1, LEO1, CTR9, WDR61, RTF1, CDC73); "
            "    Elongates RNA Pol II; promotes H3K4me3 at pro-apoptotic genes; "
            "    Directly binds β-catenin → sequesters → inhibits Wnt target genes (cyclin D1, c-Myc); "
            "    Pro-apoptotic role via JunB induction; "
            "  CDC73 LOF: "
            "    PAF1 complex loses CDC73 subunit → transcriptional elongation dysregulated; "
            "    β-catenin free → Wnt targets activated (cyclin D1 ↑, c-Myc ↑) → G1/S entry; "
            "    H3K4me3 at pro-apoptotic loci lost → resistance to apoptosis; "
            "    Clonal expansion → adenoma or carcinoma; "
            "  CARCINOMA vs ADENOMA in CDC73: "
            "    Both show LOH at 1q31 (second somatic hit); "
            "    Additional hits (TP53, RB1, PRUNE2) → malignant transformation; "
            "    IHC parafibromin loss present in both but complete absence more common in carcinoma"
        ),
    },
    {
        "gene": "GNA11",
        "protein": (
            "GNA11 -- 19p13.3 AD LOF FHH2 -- 359aa -- Galpha11-"
            "42kDa-G-Protein-Alpha11-FHH2-Phenotypically-Identical-FHH1-Benign-"
            "GNA11-GOF=ADH2-Opposite-Phenotype-OMIM-145981"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "359 aa / 42 kDa (GNA11 — guanine nucleotide-binding protein subunit alpha-11; Gα11; "
            "FUNCTION: Gq-family alpha subunit; couples CaSR (Gq) → PLCβ → IP3 + DAG → Ca²⁺ mobilisation; "
            "  downstream of CaSR on the same signalling cascade; "
            "LOF = FHH2: same phenotype as CASR LOF (FHH1) — Ca²⁺ set-point right-shifted; "
            "  Mechanism: Gα11 LOF → even when CaSR activated, downstream IP3/Ca²⁺ signalling impaired; "
            "  Result: CaSR activation fails to suppress PTH; renal Ca²⁺ reabsorption persists; "
            "  Biochemistry: identical to FHH1 (mild hypercalcaemia + low CCCR + normal/mildly elevated PTH); "
            "GOF = ADH2 (Autosomal Dominant Hypocalcaemia type 2): "
            "  Covered in Hereditary-Hypoparathyroidism-Atlas; "
            "  GOF → CaSR pathway constitutively active → Ca²⁺ set-point left-shifted → hypocalcaemia; "
            "encoded 19p13.3"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) LOF — FHH2 (Familial Hypocalciuric Hypercalcemia type 2): "
            "  Phenotypically IDENTICAL to FHH1 (CASR LOF): "
            "    Mild hypercalcaemia (2.55-2.90 mmol/L); PTH normal or mildly elevated; "
            "    CCCR <0.01; magnesium mildly elevated; "
            "    Asymptomatic in most; benign lifelong course; "
            "  PREVALENCE: less common than FHH1; ~10-15% of FHH families; "
            "  KEY CLINICAL RULE: FHH2 = FHH1 in ALL clinical respects → DO NOT OPERATE; "
            "  Differentiation from FHH1: only by genetic testing (GNA11 vs CASR panel); "
            "  MONITORING: same as FHH1 (annual Ca²⁺ + renal function; reassure patient); "
            "  Cinacalcet: theoretically ineffective (acts on CaSR, downstream of GNA11); "
            "  IMPORTANT EDUCATIONAL POINT: "
            "    GNA11 LOF = FHH2 = hypercalcaemia (benign); "
            "    GNA11 GOF = ADH2 = hypocalcaemia (hypoparathyroidism); "
            "    SAME GENE, OPPOSITE DIRECTIONS, OPPOSITE DISEASES"
        ),
        "disease_category": (
            "FHH2 — FAMILIAL HYPOCALCIURIC HYPERCALCEMIA TYPE 2 (BENIGN): "
            "  NEVER operate FHH2; clinically identical to FHH1; "
            "  Diagnose: CCCR <0.01 + family history + GNA11 mutation; "
            "  Rarer than FHH1 but same management: reassure + monitor; "
            "  Key teaching: CaSR signalling axis disruption at different points gives same FHH phenotype; "
            "    CaSR LOF (FHH1) → GNA11 LOF (FHH2) → AP2S1 LOF (FHH3); "
            "  Cinacalcet works on CaSR (positive allosteric modulator) → may be ineffective in FHH2/FHH3; "
            "  Genetic testing differentiates: CASR (exons 1-7) negative → test GNA11 → test AP2S1"
        ),
        "disease_pathway": (
            "GNA11 LOF → IMPAIRED CaSR-Gq SIGNALLING → Ca²⁺ SET-POINT RIGHT-SHIFT: "
            "  Normal CaSR-GNA11 axis: "
            "    High Ca²⁺ → CaSR (GPCR) → Gαq (GNA11) activated → PLCβ → IP3 → ER Ca²⁺ release; "
            "    IP3 + Ca²⁺ → CaM-kinase → reduces PTH gene expression + granule exocytosis; "
            "    Simultaneously: MAPK pathway → reduced CYP27B1 (less calcitriol); "
            "  GNA11 LOF: "
            "    CaSR activates normally but Gα11 signal transduction impaired; "
            "    IP3 generation reduced → ER Ca²⁺ release blunted → PTH suppression incomplete; "
            "    Parathyroid set-point shifts right (requires higher Ca²⁺ to suppress PTH); "
            "  Renal effect: "
            "    TAL GNA11 LOF → ROMK/NKCC2 activity not inhibited → increased Ca²⁺ reabsorption; "
            "    Urine Ca²⁺ inappropriately low for serum Ca²⁺ level → CCCR <0.01"
        ),
    },
    {
        "gene": "AP2S1",
        "protein": (
            "AP2S1 -- 19q13.32 AD LOF FHH3 -- 142aa -- AP2-sigma1-"
            "17kDa-Clathrin-Adaptor-Protein-FHH3-Highest-Ca2+-All-FHH-"
            "Cognitive-Impairment-Subset-Benign-CaSR-Internalisation-Defect-OMIM-600456"
        ),
        "locus": "19q13.32",
        "protein_size": (
            "142 aa / 17 kDa (AP2S1 — adaptor-related protein complex 2 sigma-1 subunit; "
            "FUNCTION: component of AP-2 clathrin-adaptor complex; recognises YxxΦ and di-leucine motifs; "
            "  AP-2 complex drives clathrin-coated vesicle formation for receptor endocytosis; "
            "  CaSR-specific: AP2S1 mutations disrupt CaSR-AP2 interaction → CaSR not internalised; "
            "LOF CONSEQUENCE: "
            "  CaSR fails to internalise after activation → less recycling → reduced cell-surface expression; "
            "  PARADOX: less CaSR on surface → less Ca²⁺ signalling despite elevated extracellular Ca²⁺; "
            "  Phenotype: FHH3 — biochemically similar to FHH1/FHH2 but HIGHEST Ca²⁺; "
            "  Unique: some patients have cognitive impairment (mechanism unclear — possible CNS CaSR role); "
            "  Biochemistry: Ca²⁺ typically 2.7-3.1 mmol/L (higher than FHH1/FHH2); CCCR <0.01; "
            "encoded 19q13.32 (same chromosome arm as GNA11 at 19p13.3)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) LOF — FHH3 (Familial Hypocalciuric Hypercalcemia type 3): "
            "  Phenotype similar to FHH1/FHH2 but with important differences: "
            "    HIGHEST SERUM CALCIUM OF ALL FHH TYPES (can be 2.8-3.2 mmol/L); "
            "    PTH normal or mildly elevated (usually not as high as PHPT); "
            "    CCCR <0.01 (same as FHH1/FHH2); "
            "    Magnesium elevated; "
            "  COGNITIVE IMPAIRMENT: "
            "    Subset of AP2S1 R15L/R15C carriers; intellectual disability described in some families; "
            "    Mechanism: CNS CaSR involved in synaptogenesis/GABA signalling?; "
            "  CLINICAL RULE: FHH3 = BENIGN; DO NOT OPERATE; "
            "  Higher Ca²⁺ than FHH1/FHH2 can cause misdiagnosis as symptomatic PHPT → genetic testing critical; "
            "  PREVALENCE: rarest of the three FHH types; "
            "  R15L mutation: most common AP2S1 variant in FHH3 families; "
            "  Cinacalcet: may work partially (CaSR still present, just less internalised); "
            "  GENOTYPING STRATEGY: "
            "    If CCCR <0.01 + CASR negative + GNA11 negative → sequence AP2S1 exon 1 (most mutations R15)"
        ),
        "disease_category": (
            "FHH3 — FAMILIAL HYPOCALCIURIC HYPERCALCEMIA TYPE 3 (BENIGN, HIGHEST Ca²⁺ OF FHH): "
            "  Rarest FHH; diagnose by CCCR + family history + AP2S1 sequencing; "
            "  Higher Ca²⁺ causes more anxiety/misdiagnosis than FHH1/FHH2; "
            "  Management: same as FHH1/FHH2 — reassure + monitor; "
            "  Cognitive impairment: not all carriers; screen; refer neurodevelopment if concerned; "
            "  Teaching point: three FHH genes disrupt the same CaSR-Gα11-AP2 axis at different steps; "
            "  FHH vs PHPT differentiation flowchart: "
            "    Hypercalcaemia → measure CCCR; "
            "    CCCR <0.01 → CASR panel (CASR + GNA11 + AP2S1) → FHH confirmed; "
            "    CCCR >0.02 → PHPT likely (MEN1, CDC73, RET, CDKN1B panel)"
        ),
        "disease_pathway": (
            "AP2S1 LOF → CaSR INTERNALISATION DEFECT → REDUCED SURFACE CaSR SIGNALLING: "
            "  Normal CaSR trafficking: "
            "    Activated CaSR → β-arrestin recruitment → AP-2/clathrin recognition of CaSR C-tail; "
            "    AP2S1 (sigma-1) binds YxxΦ motif on CaSR intracellular C-terminal domain; "
            "    Clathrin-coated pit → vesicle formation → CaSR internalised → endosome; "
            "    Recycled back to surface (or degraded if ubiquitinated); "
            "    Internalisation terminates signalling + allows resensitisation; "
            "  AP2S1 LOF: "
            "    AP-2 complex cannot bind CaSR C-tail → CaSR not internalised after activation; "
            "    PARADOX: CaSR accumulates on cell surface but signalling desensitised (β-arr coupled); "
            "    Net effect: less TOTAL Ca²⁺ signal per unit extracellular Ca²⁺; "
            "    Parathyroid + TAL cells respond as if less Ca²⁺ present → PTH not fully suppressed; "
            "    CCCR falls (renal TAL fails to detect hypercalcaemia → Ca²⁺ reabsorbed)"
        ),
    },
    {
        "gene": "RET",
        "protein": (
            "RET -- 10q11.21 AD GOF MEN2A -- 1114aa -- RET-RTK-"
            "120kDa-GDNF-Receptor-MEN2A-PHPT-10-20pct-C634-Codon-Highest-Risk-"
            "PHEO-EXCLUDED-BEFORE-SURGERY-MEN2B-VIRTUALLY-NO-PHPT-OMIM-171400"
        ),
        "locus": "10q11.21",
        "protein_size": (
            "1114 aa / 120 kDa (RET — rearranged during transfection; receptor tyrosine kinase; "
            "FUNCTION: receptor for GDNF-family ligands (GDNF, neurturin, artemin, persephin); "
            "  co-receptor complex: RET + GFRα1-4 → RAS/MAPK/PI3K/PLC-γ; "
            "  essential for enteric nervous system development + kidney organogenesis; "
            "GOF mutations → constitutive dimerisation/phosphorylation: "
            "  Extracellular cysteine mutations (C609, C611, C618, C620, C630, C634): "
            "    Unpaired cysteine → intermolecular disulphide bond → constitutive dimer → GOF; "
            "    C634: highest risk for MTC + pheochromocytoma + PHPT; "
            "    C634R/Y: highest PHPT penetrance (~20-30%); "
            "  Kinase domain mutation (M918T): MEN2B — aggressive MTC; NO PHPT in MEN2B (key!); "
            "PHPT in MEN2A: "
            "  10-20% lifetime risk (lower than MEN1); single gland involvement more common; "
            "  Mild-moderate hypercalcaemia; resectable at same operation as thyroidectomy; "
            "encoded 10q11.21"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) GOF — MEN2A (Multiple Endocrine Neoplasia type 2A): "
            "  Triad: MTC + pheochromocytoma + PHPT (in descending frequency): "
            "  MTC: near 100% lifetime risk; aggressive; prophylactic thyroidectomy age-stratified by codon; "
            "  Phaeochromocytoma: 50% lifetime risk; bilateral; adrenal; BENIGN in most; "
            "  PHPT: 10-20% lifetime risk; milder than MEN1; often single adenoma; "
            "  CRITICAL PRE-OPERATIVE RULE: "
            "    EXCLUDE PHAEOCHROMOCYTOMA BEFORE ANY SURGERY (including parathyroid or thyroid); "
            "    Undiagnosed pheo → hypertensive crisis under anaesthesia → life-threatening; "
            "    Screen: 24h urine catecholamines/metanephrines OR plasma metanephrines; "
            "  MEN2A vs MEN2B: "
            "    MEN2B (M918T): most aggressive MTC + pheo + mucosal neuromas + marfanoid; "
            "    MEN2B: VIRTUALLY NO PHPT — key clinical distinguisher; "
            "  TREATMENT OF MEN2A PHPT: "
            "    Parathyroidectomy at time of thyroid surgery (if hypercalcaemic); "
            "    Unilateral exploration acceptable if single gland confirmed imaging; "
            "    Surveillance: annual serum calcium + PTH; "
            "  CODON-RISK TABLE: "
            "    C634: high MTC+Pheo+PHPT risk; thyroidectomy recommended <5 years old; "
            "    C609/611/618/620: moderate risk; thyroidectomy age 5-10; "
            "    Others: lower risk counselling"
        ),
        "disease_category": (
            "MEN2A PHPT — MILD SINGLE-GLAND HYPERPARATHYROIDISM (PHEO EXCLUSION MANDATORY): "
            "  Milder than MEN1 PHPT — single adenoma; moderate Ca²⁺ elevation; "
            "  MANAGEMENT PRIORITY ORDER in MEN2A: "
            "    1. Diagnose/treat phaeochromocytoma first; "
            "    2. Thyroidectomy (MTC); "
            "    3. Parathyroidectomy (at same or separate operation); "
            "  POST-OP SURVEILLANCE: "
            "    Annual Ca²⁺ + PTH; annual plasma metanephrines; calcitonin every 6 months; "
            "  VANDETANIB/CABOZANTINIB: for metastatic MTC (RET inhibitors); no effect on PHPT component; "
            "  MEN2B: NEVER expect PHPT — if found, suspect concurrent MEN1 or sporadic PHPT; "
            "  GENETIC COUNSELLING: "
            "    RET mutation in child → prophylactic thyroidectomy timing per codon risk category; "
            "    50% offspring risk (AD); codon determines urgency"
        ),
        "disease_pathway": (
            "RET GOF → CONSTITUTIVE RTK SIGNALLING → PARATHYROID CELL PROLIFERATION + EXCESS PTH: "
            "  Normal RET: requires ligand (GDNF + GFRα1) → dimerisation → trans-autophosphorylation; "
            "    Activation is ligand-dependent, transient; "
            "  C634R/Y GOF: "
            "    Unpaired extracellular cysteine → intermolecular disulphide bridge with partner RET; "
            "    Constitutive dimer → constant kinase activation WITHOUT ligand; "
            "    RAS/MAPK → cyclin D1 upregulation → G1 bypass → proliferation; "
            "    PI3K/AKT → BCL-2 family pro-survival → anti-apoptosis; "
            "  PARATHYROID CELL specific: "
            "    RET expressed in parathyroid chief cells; "
            "    Constitutive RET signalling → chief cell proliferation → adenoma; "
            "    Parallel: chief cells increase PTH synthesis (transcriptional upregulation via MAPK); "
            "  WHY SINGLE GLAND in MEN2A (vs multiglandular MEN1): "
            "    RET drives clonal expansion via single somatic second-hit (LOH 10q); "
            "    One gland gets second hit first → single adenoma more common than MEN1"
        ),
    },
    {
        "gene": "CDKN1B",
        "protein": (
            "CDKN1B -- 12p13.1 AD LOF -- 196aa -- p27Kip1-"
            "22kDa-CDK2-CDK4-Inhibitor-MEN4-MEN1-Like-WITHOUT-Menin-"
            "PHPT+Pituitary-ACTH-More-Common-2-3pct-MEN1-Like-Cases-OMIM-610755"
        ),
        "locus": "12p13.1",
        "protein_size": (
            "196 aa / 22 kDa (CDKN1B — cyclin-dependent kinase inhibitor 1B; p27Kip1; "
            "FUNCTION: CDK inhibitor; binds CDK2/cyclin E + CDK4/cyclin D → blocks cell cycle G1→S; "
            "  major tumour suppressor in pituitary + parathyroid + pancreatic islets; "
            "  controlled by ubiquitin-mediated proteasomal degradation (SKP2 pathway); "
            "  nuclear p27 = growth suppressor; cytoplasmic p27 = pro-oncogenic (in some contexts); "
            "LOF CONSEQUENCE — MEN4: "
            "  Reduced CDK inhibition → G1→S unrestricted → cell proliferation; "
            "  PHPT: similar to MEN1 (multiglandular hyperplasia; sometimes single adenoma); "
            "  Pituitary: ACTH-secreting adenomas more common (vs prolactinoma in MEN1); "
            "  Pancreatic NET, adrenal cortical tumours: less frequent than MEN1; "
            "  GENOTYPE: germline CDKN1B LOF; somatic second hit (LOH 12p); "
            "encoded 12p13.1 (CDKN1 locus cluster)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) LOF — MEN4 (Multiple Endocrine Neoplasia type 4): "
            "  Prevalence: ~2-3% of MEN1-phenotype patients who test negative for MEN1; "
            "  PHPT phenotype: "
            "    Similar to MEN1: multiglandular, moderate-severe hypercalcaemia; "
            "    Penetrance slightly lower than MEN1; "
            "  Pituitary: ACTH-secreting adenoma (Cushing) more common than in MEN1 (prolactinoma); "
            "  Pancreatic NET: less common than MEN1; gastrinoma, insulinoma reported; "
            "  DIAGNOSIS: "
            "    MEN1-phenotype (PHPT + pituitary ± pancreatic) + MEN1 sequencing NEGATIVE; "
            "    → Test CDKN1B; "
            "    MEN4 accounts for most menin-mutation-negative MEN1-like families; "
            "  TREATMENT: same as MEN1 (subtotal parathyroidectomy; annual surveillance); "
            "  SURVEILLANCE PROTOCOL: "
            "    Annual Ca²⁺ + PTH; annual cortisol/ACTH/prolactin/IGF-1; "
            "    MRI pituitary every 3 years; "
            "    Pancreatic screening every 2 years (CT/MRI); "
            "  CDKN1B in rats: multiple endocrine neoplasia rat (MENX) model → validated path"
        ),
        "disease_category": (
            "MEN4 — MEN1-LIKE SYNDROME WITHOUT MENIN MUTATION (CDK INHIBITOR p27 LOF): "
            "  Key clinical message: 'MEN1-phenotype, MEN1-negative → sequence CDKN1B'; "
            "  PHPT management identical to MEN1 (subtotal 3.5-gland parathyroidectomy); "
            "  Pituitary ACTH (Cushing) more common than MEN1 → screen with 24h UFC + cortisol; "
            "  Prognosis: generally similar to MEN1 (less data due to rarity); "
            "  Molecular biology insight: MEN1 and CDKN1B converge on CDK inhibition: "
            "    Menin maintains p21/p27 expression (via H3K4me3 at promoters); "
            "    CDKN1B IS p27 → both LOF pathways reduce CDK inhibition → same tumour spectrum; "
            "  RESEARCH: p27 cytoplasmic overexpression in other cancers (breast, prostate) — "
            "    germline CDKN1B LOF → nuclear p27 loss → CDK release → endocrine proliferation"
        ),
        "disease_pathway": (
            "CDKN1B LOF → p27 LOSS → CDK2/CDK4 RELEASED → UNRESTRICTED PARATHYROID CELL CYCLE: "
            "  Normal p27 function: "
            "    p27 binds CDK2/cyclin E (G1→S inhibition) + CDK4/cyclin D (mid-G1 inhibition); "
            "    p27 nuclear import: JAK/STAT pathway; nuclear export blocked by PI3K/AKT; "
            "    p27 degradation: phospho-T157 (AKT) → CRM1 export → phospho-T187 → SCF-Skp2 ubiquitination; "
            "  CDKN1B LOF: "
            "    p27 protein absent/reduced → CDK2/CDK4 constitutively active; "
            "    RB hyperphosphorylated → E2F1 released → S-phase entry; "
            "    Parathyroid chief cells cannot pause in G1 → proliferation; "
            "    Somatic second hit at 12p → clonal expansion → adenoma or hyperplasia; "
            "  CONVERGENCE WITH MEN1: "
            "    MEN1 LOF → menin gone → H3K4me3 at CDKN1B locus falls → p27 expression drops; "
            "    CDKN1B LOF → p27 directly eliminated; "
            "    Both ultimately reduce CDK inhibition → same endocrine tumour phenotype"
        ),
    },
    {
        "gene": "GCM2",
        "protein": (
            "GCM2 -- 6p24.2 AD GOF FIHPT -- 495aa -- GCM2-"
            "47kDa-Master-Parathyroid-TF-FIHPT-GOF-Opposite-Hypoparathyroidism-"
            "Same-Gene-LOF=HP-GOF=PHPT-Multiglandular-Hyperplasia-Common-OMIM-603716"
        ),
        "locus": "6p24.2",
        "protein_size": (
            "495 aa / 47 kDa (GCM2 — glial cells missing 2; GCM domain zinc-finger transcription factor; "
            "FUNCTION: master regulator of parathyroid gland development and chief cell identity; "
            "  regulates PTH gene expression; maintains CaSR expression; "
            "  required for parathyroid gland survival post-formation; "
            "  GCM2 ortholog in Drosophila: 'gcm' = glial cell fate determinant; "
            "BIDIRECTIONAL CLINICAL ROLE: "
            "  LOF → HYPOPARATHYROIDISM (Hereditary-Hypoparathyroidism-Atlas): "
            "    Reduced parathyroid gland mass/function → ↓ PTH → hypocalcaemia; "
            "    Most common gene for familial isolated hypoparathyroidism; "
            "  GOF → HYPERPARATHYROIDISM (this atlas): "
            "    Increased GCM2 activity → excess parathyroid cell proliferation; "
            "    Elevated PTH + hypercalcaemia → Familial Isolated Hyperparathyroidism (FIHPT); "
            "    Multiglandular hyperplasia common (similar to MEN1 morphology but MEN1-negative); "
            "KEY TEACHING: GCM2 = only endocrine gene where LOF and GOF cause opposite calcium disorders; "
            "encoded 6p24.2"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) GOF — FIHPT (Familial Isolated Hyperparathyroidism): "
            "  Phenotype: "
            "    Isolated PHPT without features of MEN1 (no pituitary, no pancreatic NET) or MEN2 or HPT-JT; "
            "    Multiglandular parathyroid hyperplasia in majority; "
            "    Moderate hypercalcaemia (Ca²⁺ 2.6-3.2 mmol/L); PTH elevated; "
            "    High urine calcium; nephrolithiasis; "
            "    Onset typically 3rd-5th decade; "
            "  GCM2 GOF variants: "
            "    Missense mutations that enhance GCM2 transcriptional activity; "
            "    Incomplete penetrance observed in some families; "
            "    Multiglandular disease pattern suggests a cell-autonomous proliferative drive; "
            "  TREATMENT: "
            "    Subtotal parathyroidectomy (3.5 gland) for multiglandular; "
            "    Annual Ca²⁺ + PTH surveillance post-operatively; "
            "  GENETIC TESTING STRATEGY: "
            "    FIHPT work-up: MEN1 → CDKN1B → CDC73 → GCM2 → CaSR (LOF) panel; "
            "  CLINICAL TEACHING POINT: "
            "    GCM2 LOF = isolated hypoparathyroidism (covered Hypo-PTH Atlas); "
            "    GCM2 GOF = isolated hyperparathyroidism (this atlas); "
            "    Same gene, two calcium disorders at opposite ends of the spectrum"
        ),
        "disease_category": (
            "GCM2 GOF — FAMILIAL ISOLATED HYPERPARATHYROIDISM (FIHPT) — MULTIGLANDULAR HYPERPLASIA: "
            "  Diagnosis: FIHPT phenotype + MEN1/CDC73/RET/CDKN1B negative + GCM2 GOF variant; "
            "  Multiglandular hyperplasia → subtotal 3.5-gland parathyroidectomy (same as MEN1); "
            "  Single gland resection insufficient in multiglandular FIHPT; "
            "  Recurrence: ~15-20% at 10 years (similar to MEN1); "
            "  KEY EDUCATIONAL POINT: "
            "    GCM2 GOF = PHPT; GCM2 LOF = hypoparathyroidism (both are AD); "
            "    Bidirectional gene with dosage-sensitive parathyroid gland maintenance; "
            "  Intraoperative PTH monitoring: helpful to confirm adequate resection; "
            "  Calcium monitoring: 24h post-op hypocalcaemia = successful debulking (hungry bone); "
            "  Long-term: annual Ca²⁺ + PTH; 4D-CT if recurrence suspected"
        ),
        "disease_pathway": (
            "GCM2 GOF → ENHANCED TF ACTIVITY → EXCESS PARATHYROID CELL MASS → PTH OVERPRODUCTION: "
            "  Normal GCM2 function: "
            "    GCM2 binds GCM motif (TGCGGGT) in promoters of PTH gene + CaSR gene; "
            "    Activates chief cell differentiation and maintenance programme; "
            "    Tightly regulated — dosage-sensitive (LOF reduces parathyroid → hypo-Ca; GOF expands → hyper-Ca); "
            "  GCM2 GOF: "
            "    Enhanced GCM2 transcriptional activity → upregulation of pro-proliferative targets; "
            "    Chief cell number increases → multiglandular hyperplasia; "
            "    Each chief cell overexpresses PTH (GCM2 drives PTH transcription); "
            "    PTH excess → hypercalcaemia; "
            "  BIDIRECTIONAL DOSAGE: "
            "    0 copies functional GCM2 → no parathyroids → no PTH → fatal hypocalcaemia (mouse); "
            "    0.5 copies (LOF heterozygous) → reduced parathyroid function → hypoparathyroidism (human); "
            "    1.0 copies (normal) → normal parathyroid; "
            "    >1.0 effective copies (GOF) → parathyroid hyperplasia → hyperparathyroidism"
        ),
    },
]


def _make_patients(seed: int, gene: str, n: int = 40) -> list:
    """Generate 40 synthetic PHPT/FHH patients for one gene. All values are plausible
    educational simulations, not real patient data."""
    rng = random.Random(seed)

    gene_params = {
        "MEN1": {
            "ca_range": (2.65, 3.30), "pth_range": (9, 55), "cccr_range": (0.020, 0.055),
            "mg_range": (0.75, 1.05), "carcinoma_pct": 0.01, "multigung_pct": 0.85,
            "jaw_fibroma_pct": 0.0, "nephrolithiasis_pct": 0.45, "bone_loss_pct": 0.35,
        },
        "CASR": {
            "ca_range": (2.58, 2.92), "pth_range": (4, 11), "cccr_range": (0.003, 0.009),
            "mg_range": (0.90, 1.15), "carcinoma_pct": 0.0, "multigung_pct": 0.05,
            "jaw_fibroma_pct": 0.0, "nephrolithiasis_pct": 0.05, "bone_loss_pct": 0.05,
        },
        "CDC73": {
            "ca_range": (2.80, 3.85), "pth_range": (20, 120), "cccr_range": (0.025, 0.080),
            "mg_range": (0.70, 1.00), "carcinoma_pct": 0.18, "multigung_pct": 0.30,
            "jaw_fibroma_pct": 0.50, "nephrolithiasis_pct": 0.60, "bone_loss_pct": 0.55,
        },
        "GNA11": {
            "ca_range": (2.55, 2.90), "pth_range": (4, 10), "cccr_range": (0.003, 0.009),
            "mg_range": (0.88, 1.12), "carcinoma_pct": 0.0, "multigung_pct": 0.05,
            "jaw_fibroma_pct": 0.0, "nephrolithiasis_pct": 0.04, "bone_loss_pct": 0.04,
        },
        "AP2S1": {
            "ca_range": (2.70, 3.12), "pth_range": (5, 13), "cccr_range": (0.002, 0.008),
            "mg_range": (0.92, 1.20), "carcinoma_pct": 0.0, "multigung_pct": 0.05,
            "jaw_fibroma_pct": 0.0, "nephrolithiasis_pct": 0.08, "bone_loss_pct": 0.06,
        },
        "RET": {
            "ca_range": (2.60, 3.10), "pth_range": (7, 40), "cccr_range": (0.015, 0.050),
            "mg_range": (0.72, 1.00), "carcinoma_pct": 0.02, "multigung_pct": 0.25,
            "jaw_fibroma_pct": 0.0, "nephrolithiasis_pct": 0.30, "bone_loss_pct": 0.25,
        },
        "CDKN1B": {
            "ca_range": (2.62, 3.20), "pth_range": (9, 50), "cccr_range": (0.018, 0.055),
            "mg_range": (0.74, 1.02), "carcinoma_pct": 0.01, "multigung_pct": 0.70,
            "jaw_fibroma_pct": 0.0, "nephrolithiasis_pct": 0.40, "bone_loss_pct": 0.30,
        },
        "GCM2": {
            "ca_range": (2.60, 3.25), "pth_range": (8, 48), "cccr_range": (0.018, 0.055),
            "mg_range": (0.73, 1.02), "carcinoma_pct": 0.02, "multigung_pct": 0.75,
            "jaw_fibroma_pct": 0.0, "nephrolithiasis_pct": 0.42, "bone_loss_pct": 0.32,
        },
    }
    p = gene_params.get(gene, gene_params["MEN1"])

    def treatment_choice(carcinoma, multigung, fhh_type):
        if fhh_type:
            r = rng.random()
            if r < 0.80: return "Observation + annual Ca/PTH monitoring"
            if r < 0.92: return "Reassurance + dietary counselling (avoid dehydration)"
            return "Cinacalcet (symptomatic FHH — rare)"
        r = rng.random()
        if gene == "MEN1":
            if r < 0.60: return "Subtotal parathyroidectomy (3.5-gland)"
            if r < 0.82: return "Total parathyroidectomy + forearm autograft"
            if r < 0.92: return "Cinacalcet (bridge to surgery)"
            return "Subtotal PTX + bilateral neck exploration"
        elif gene == "CDC73":
            if carcinoma:
                if r < 0.70: return "En-bloc resection (carcinoma) + unilateral thyroid lobe"
                return "En-bloc resection + bilateral neck dissection"
            if r < 0.60: return "Focused parathyroidectomy (single adenoma)"
            if r < 0.80: return "Bilateral exploration + subtotal PTX"
            return "Denosumab + cinacalcet (inoperable/recurrent)"
        elif gene == "RET":
            if r < 0.55: return "Parathyroidectomy at time of thyroidectomy (MTC)"
            if r < 0.75: return "Focused parathyroidectomy (single adenoma)"
            if r < 0.90: return "Total PTX + thyroidectomy (simultaneous)"
            return "Surveillance (mild hypercalcaemia, pheo excluded)"
        elif gene in ("CDKN1B", "GCM2"):
            if multigung:
                if r < 0.60: return "Subtotal parathyroidectomy (3.5-gland)"
                if r < 0.82: return "Total parathyroidectomy + forearm autograft"
                return "Cinacalcet bridge → surgery"
            if r < 0.70: return "Focused parathyroidectomy (single adenoma)"
            return "Subtotal parathyroidectomy"
        else:
            if r < 0.60: return "Focused parathyroidectomy"
            if r < 0.82: return "Subtotal parathyroidectomy"
            return "Total parathyroidectomy + autograft"

    patients = []
    is_fhh = gene in ("CASR", "GNA11", "AP2S1")
    for i in range(n):
        ca       = round(rng.uniform(*p["ca_range"]), 2)
        pth      = round(rng.uniform(*p["pth_range"]), 1)
        cccr     = round(rng.uniform(*p["cccr_range"]), 4)
        mg       = round(rng.uniform(*p["mg_range"]), 2)
        carcinoma = rng.random() < p["carcinoma_pct"]
        multigung = rng.random() < p["multigung_pct"]
        jaw_fib   = rng.random() < p["jaw_fibroma_pct"]
        nephro    = rng.random() < p["nephrolithiasis_pct"]
        bone_loss = rng.random() < p["bone_loss_pct"]
        t_score   = round(rng.uniform(-2.5, 0.5) if bone_loss else rng.uniform(-1.0, 1.5), 1)
        tx        = treatment_choice(carcinoma, multigung, is_fhh)
        age_dx    = rng.randint(20, 65) if not is_fhh else rng.randint(25, 75)

        patients.append({
            "id":                   f"{gene}-{i+1:02d}",
            "gene":                 gene,
            "age_at_dx":            age_dx,
            "serum_ca_mmol":        ca,
            "serum_pth_pmol":       pth,
            "cccr":                 cccr,
            "serum_mg_mmol":        mg,
            "parathyroid_carcinoma": carcinoma,
            "multiglandular":       multigung,
            "jaw_fibroma":          jaw_fib,
            "nephrolithiasis":      nephro,
            "bone_loss_osteoporosis": bone_loss,
            "dexa_t_score":         t_score,
            "fhh_type":             is_fhh,
            "treatment":            tx,
        })
    return patients


# ── API surface ───────────────────────────────────────────────────────────────

def generate_overview() -> dict:
    """Atlas overview — aggregate stats across all 8 hereditary PHPT / FHH genes."""
    all_patients = []
    for idx, g in enumerate(ATLAS_GENES):
        all_patients.extend(_make_patients(SEED_BASE + idx, g["gene"]))

    n                = len(all_patients)
    n_high_ca        = sum(1 for p in all_patients if p["serum_ca_mmol"] > 2.75)
    n_carcinoma      = sum(1 for p in all_patients if p["parathyroid_carcinoma"])
    n_multigung      = sum(1 for p in all_patients if p["multiglandular"])
    n_jaw            = sum(1 for p in all_patients if p["jaw_fibroma"])
    n_nephro         = sum(1 for p in all_patients if p["nephrolithiasis"])
    n_fhh            = sum(1 for p in all_patients if p["fhh_type"])
    mean_ca          = round(sum(p["serum_ca_mmol"] for p in all_patients) / n, 3)
    mean_pth         = round(sum(p["serum_pth_pmol"] for p in all_patients) / n, 1)
    mean_cccr        = round(sum(p["cccr"] for p in all_patients) / n, 4)

    gene_summary = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        gene_summary.append({
            "gene":           g["gene"],
            "locus":          g["locus"],
            "n_patients":     len(pts),
            "mean_ca":        round(sum(p["serum_ca_mmol"] for p in pts) / len(pts), 3),
            "mean_pth":       round(sum(p["serum_pth_pmol"] for p in pts) / len(pts), 1),
            "mean_cccr":      round(sum(p["cccr"] for p in pts) / len(pts), 4),
            "carcinoma_pct":  round(100 * sum(1 for p in pts if p["parathyroid_carcinoma"]) / len(pts), 1),
            "multigung_pct":  round(100 * sum(1 for p in pts if p["multiglandular"]) / len(pts), 1),
            "jaw_fibroma_pct":round(100 * sum(1 for p in pts if p["jaw_fibroma"]) / len(pts), 1),
            "nephro_pct":     round(100 * sum(1 for p in pts if p["nephrolithiasis"]) / len(pts), 1),
            "fhh_type":       g["gene"] in ("CASR", "GNA11", "AP2S1"),
            "syndrome": (
                "MEN1-Multiglandular"  if g["gene"] == "MEN1" else
                "FHH1-Benign"          if g["gene"] == "CASR" else
                "HPT-JT-Carcinoma"     if g["gene"] == "CDC73" else
                "FHH2-Benign"          if g["gene"] == "GNA11" else
                "FHH3-Benign-HighCa"   if g["gene"] == "AP2S1" else
                "MEN2A-Mild-PHPT"      if g["gene"] == "RET" else
                "MEN4-MEN1-Like"       if g["gene"] == "CDKN1B" else
                "FIHPT-Multiglandular"
            ),
        })

    return {
        "atlas":         "Hereditary-Primary-Hyperparathyroidism-Atlas",
        "genes":         [g["gene"] for g in ATLAS_GENES],
        "n_genes":       len(ATLAS_GENES),
        "n_patients":    n,
        "seeds":         f"{SEED_BASE}–{SEED_BASE + len(ATLAS_GENES) - 1}",
        "syndromes": [
            "MEN1 multiglandular PHPT (MEN1 LOF)",
            "FHH1 benign hypocalciuric hypercalcaemia (CASR LOF)",
            "HPT-JT parathyroid carcinoma 15% (CDC73 LOF)",
            "FHH2 benign identical-to-FHH1 (GNA11 LOF)",
            "FHH3 benign highest Ca²⁺ of FHH (AP2S1 LOF)",
            "MEN2A mild PHPT pheo-exclusion mandatory (RET GOF)",
            "MEN4 MEN1-like p27 CDK inhibitor (CDKN1B LOF)",
            "FIHPT multiglandular GCM2 GOF (opposite of hypoparathyroidism)",
        ],
        "aggregate_metrics": {
            "mean_serum_ca_mmol":       mean_ca,
            "mean_serum_pth_pmol":      mean_pth,
            "mean_cccr":                mean_cccr,
            "high_ca_gt275_pct":        round(100 * n_high_ca   / n, 1),
            "parathyroid_carcinoma_pct":round(100 * n_carcinoma / n, 1),
            "multiglandular_pct":       round(100 * n_multigung / n, 1),
            "jaw_fibroma_pct":          round(100 * n_jaw       / n, 1),
            "nephrolithiasis_pct":      round(100 * n_nephro    / n, 1),
            "fhh_benign_pct":           round(100 * n_fhh       / n, 1),
        },
        "gene_summary": gene_summary,
        "key_clinical_rules": [
            "FHH RULE: CCCR <0.01 → FHH (CASR/GNA11/AP2S1) → DO NOT OPERATE — surgery does NOT cure FHH and causes permanent hypoparathyroidism",
            "CDC73 CARCINOMA: 15-20% parathyroid carcinoma in HPT-JT; en-bloc resection; parafibromin IHC loss = carcinoma marker; OPG jaw X-ray for ossifying fibromas",
            "MEN1 RULE: PHPT + any pituitary + any pancreatic NET = MEN1 until proven otherwise; annual Ca²⁺ from age 8; MLPA for large deletions",
            "RET RULE: exclude phaeochromocytoma BEFORE ANY SURGERY in MEN2A — hypertensive crisis under anaesthesia; MEN2B has virtually NO PHPT",
            "GCM2 OPPOSITE DIRECTIONS: GCM2 LOF = hypoparathyroidism (Hypo-PTH Atlas); GCM2 GOF = FIHPT (this atlas) — same gene, opposite calcium phenotypes",
            "AP2S1 FHH3: highest Ca²⁺ of all FHH types (2.7-3.1 mmol/L) — easily mistaken for PHPT; check CCCR + family history before recommending surgery",
            "MEN4 STRATEGY: MEN1-phenotype + MEN1 sequencing NEGATIVE → sequence CDKN1B (MEN4 accounts for ~2-3% of MEN1-like cases)",
            "NSHPT EMERGENCY: neonate + severe hypercalcaemia + CASR biallelic LOF → emergency total parathyroidectomy; parents often have FHH1",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for all 8 hereditary PHPT/FHH genes."""
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
            "mean_ca":           round(sum(p["serum_ca_mmol"] for p in pts) / len(pts), 3),
            "mean_pth":          round(sum(p["serum_pth_pmol"] for p in pts) / len(pts), 1),
            "mean_cccr":         round(sum(p["cccr"] for p in pts) / len(pts), 4),
            "carcinoma_pct":     round(100 * sum(1 for p in pts if p["parathyroid_carcinoma"]) / len(pts), 1),
            "multigung_pct":     round(100 * sum(1 for p in pts if p["multiglandular"]) / len(pts), 1),
            "jaw_fibroma_pct":   round(100 * sum(1 for p in pts if p["jaw_fibroma"]) / len(pts), 1),
            "nephro_pct":        round(100 * sum(1 for p in pts if p["nephrolithiasis"]) / len(pts), 1),
            "bone_loss_pct":     round(100 * sum(1 for p in pts if p["bone_loss_osteoporosis"]) / len(pts), 1),
            "fhh_type":          g["gene"] in ("CASR", "GNA11", "AP2S1"),
            "top_treatments":    sorted(treatments.items(), key=lambda x: -x[1])[:4],
        })
    return {"atlas": "Hereditary-Primary-Hyperparathyroidism-Atlas", "count": len(genes_data), "genes": genes_data}


def generate_definitions() -> dict:
    """Clinical definitions and teaching points for hereditary PHPT / FHH."""
    definitions = [
        {
            "term": "Familial Hypocalciuric Hypercalcaemia (FHH1/2/3)",
            "genes": ["CASR", "GNA11", "AP2S1"],
            "definition": (
                "BENIGN autosomal dominant hypercalcaemia due to CaSR-Gα11-AP2 pathway LOF. "
                "THREE TYPES: FHH1 = CASR LOF; FHH2 = GNA11 LOF; FHH3 = AP2S1 LOF. "
                "ALL THREE: mildly elevated Ca²⁺ (2.55-3.1 mmol/L); normal/mildly elevated PTH; "
                "  CCCR (urine Ca/Cr clearance ratio) <0.01 PATHOGNOMONIC; "
                "  magnesium mildly elevated; lifelong asymptomatic; DO NOT OPERATE. "
                "FHH3 has the highest Ca²⁺ of the three (AP2S1 LOF → CaSR internalisation defect). "
                "SURGERY IN FHH: ineffective (set-point remains shifted) + causes permanent hypoparathyroidism; "
                "  well-documented cases of unnecessary parathyroidectomies in FHH without resolution. "
                "CCCR FORMULA: (uCa × sCr) ÷ (uCr × sCa); "
                "  <0.01 = strong evidence FHH; 0.01-0.02 = borderline (genetic test); >0.02 = PHPT likely. "
                "NSHPT = biallelic CASR LOF; neonatal life-threatening emergency; parents both FHH1 carriers."
            ),
        },
        {
            "term": "Multiple Endocrine Neoplasia type 1 (MEN1)",
            "genes": ["MEN1"],
            "definition": (
                "AUTOSOMAL DOMINANT LOF of MEN1 (menin tumour suppressor). "
                "MEN1 TRIAD: PHPT (>90% lifetime) + pituitary adenoma (40-70%) + pancreatic NET (30-80%). "
                "PHPT features: multiglandular (85%); moderate-severe Ca²⁺ (2.65-3.3 mmol/L); "
                "  young onset (3rd-4th decade); high urine Ca²⁺; CCCR >0.02; carcinoma rare (<1%); "
                "SURVEILLANCE: annual Ca²⁺ + PTH + prolactin + IGF-1 from age 8; "
                "  annual gastrin + glucagon + VIP; MRI pituitary every 3-5 years; "
                "  CT/MRI pancreas every 1-2 years from age 20. "
                "SURGERY: subtotal 3.5-gland parathyroidectomy (high recurrence 15-20% at 10 years); "
                "  intraoperative PTH drop >50% confirms resection; "
                "  bilateral exploration mandatory (multiglandular); "
                "MOLECULAR: >1500 germline MEN1 variants; Italian/French/Finnish founders; "
                "  MLPA for large exonic deletions (missed by sequencing alone)."
            ),
        },
        {
            "term": "HPT-JT (Hyperparathyroidism-Jaw Tumor Syndrome)",
            "genes": ["CDC73"],
            "definition": (
                "AUTOSOMAL DOMINANT LOF of CDC73 (parafibromin). "
                "MOST IMPORTANT HEREDITARY PHPT FOR MALIGNANCY: parathyroid carcinoma 15-20%. "
                "HPT-JT TRIAD: PHPT + ossifying jaw fibromas (50%) + uterine tumours (75% females). "
                "PHPT features: usually single large adenoma (not multiglandular like MEN1); "
                "  SEVERE hypercalcaemia (Ca²⁺ 2.80-3.85 mmol/L); PTH often >5-10× ULN; "
                "  high urine Ca²⁺; nephrolithiasis; osteitis fibrosa cystica in advanced cases. "
                "CARCINOMA DIAGNOSIS: palpable neck mass + very high Ca²⁺/PTH; parafibromin IHC LOSS; "
                "  molecular: CDC73 somatic second hit (LOH 1q31); additional TP53/RB1 hits in carcinoma. "
                "SURGERY FOR CARCINOMA: en-bloc resection mandatory; capsule breach = seeding → recurrence; "
                "  frozen section unreliable → treat suspicious cases as carcinoma; "
                "  recurrence in 50-60%; denosumab + cinacalcet for palliation. "
                "JAW FIBROMAS: OPG (orthopantomogram) mandatory in all CDC73 carriers; "
                "  radio-opaque (ossifying) on X-ray = CDC73; radiolucent = giant cell reparative granuloma (sporadic). "
                "18F-CHOLINE PET: best imaging for recurrent/metastatic CDC73 carcinoma."
            ),
        },
        {
            "term": "MEN2A Parathyroid Disease",
            "genes": ["RET"],
            "definition": (
                "AUTOSOMAL DOMINANT GOF of RET (MEN2A). "
                "MEN2A = MTC (near 100%) + phaeochromocytoma (50%) + PHPT (10-20%). "
                "PHPT FEATURES: milder than MEN1; single adenoma more common; moderate Ca²⁺; "
                "  Ca²⁺ typically 2.60-3.10 mmol/L; moderate PTH elevation. "
                "CRITICAL RULE — PHEO FIRST: exclude phaeochromocytoma BEFORE ANY surgery; "
                "  plasma metanephrines OR 24h urine metanephrines; "
                "  missed pheo → hypertensive crisis under anaesthesia → death. "
                "MEN2B DISTINCTION: M918T mutation → MEN2B = aggressive MTC + pheo + mucosal neuromas; "
                "  MEN2B has VIRTUALLY NO PHPT — key exam question; "
                "  if PHPT found in alleged MEN2B: suspect concurrent sporadic PHPT or re-check diagnosis. "
                "CODON-RISK: C634R/Y = highest risk for MTC + pheo + PHPT; "
                "  prophylactic thyroidectomy timing: C634 = before age 5; C609/611/618/620 = age 5-10. "
                "TREATMENT: parathyroidectomy at time of thyroid surgery (if hypercalcaemic); "
                "  unilateral exploration acceptable if single gland confirmed pre-op."
            ),
        },
        {
            "term": "MEN4 (CDKN1B LOF) and FIHPT (GCM2 GOF)",
            "genes": ["CDKN1B", "GCM2"],
            "definition": (
                "MEN4 (CDKN1B LOF — p27Kip1): "
                "  MEN1-phenotype WITHOUT menin mutation; ~2-3% of MEN1-like cases; "
                "  PHPT + pituitary (ACTH-secreting adenomas more common than MEN1 prolactinoma); "
                "  Pancreatic NET less common than MEN1; "
                "  Diagnosis: MEN1-like phenotype + MEN1 sequencing negative → CDKN1B; "
                "  Molecular: p27 (CDK2/CDK4 inhibitor) lost → G1→S unrestricted; "
                "    converges with MEN1 path (menin maintains p27 expression); "
                "  Treatment: same as MEN1 (subtotal 3.5-gland parathyroidectomy). "
                "FIHPT (GCM2 GOF — familial isolated hyperparathyroidism): "
                "  Isolated PHPT without MEN1/MEN2/HPT-JT features; "
                "  GCM2 GOF = parathyroid hyperplasia (same gene as GCM2 LOF = hypoparathyroidism); "
                "  KEY TEACHING: GCM2 LOF → hypoparathyroidism; GCM2 GOF → hyperparathyroidism; "
                "    SAME GENE, OPPOSITE CALCIUM PHENOTYPES (unique bidirectional endocrine gene); "
                "  Multiglandular hyperplasia common → subtotal 3.5-gland parathyroidectomy; "
                "  Recurrence ~15-20% at 10 years (similar to MEN1). "
                "GENETIC TESTING LADDER FOR FIHPT: "
                "  MEN1 (most common) → CDKN1B (MEN4) → CDC73 (HPT-JT) → GCM2 → CASR LOF; "
                "  Panel sequencing preferred over sequential testing."
            ),
        },
        {
            "term": "Parathyroid Carcinoma vs Adenoma",
            "genes": ["CDC73", "MEN1", "RET"],
            "definition": (
                "MALIGNANT parathyroid carcinoma: ~1-2% of all PHPT; "
                "  HEREDITARY RISK BY GENE: CDC73/HPT-JT = 15-20% (highest); MEN1 <1%; sporadic <0.5%. "
                "CLINICAL CLUES FOR CARCINOMA (vs adenoma): "
                "  Palpable neck mass (rare in adenoma); "
                "  Very high Ca²⁺ (>3.5 mmol/L), very high PTH (often >10× ULN); "
                "  Gross invasion of surrounding structures on imaging; "
                "  Severe symptomatic hypercalcaemia (renal failure, pancreatitis, coma); "
                "  CDC73 germline mutation carrier. "
                "PATHOLOGY: parafibromin IHC LOSS (nuclear staining absent) = high specificity for CDC73 defect; "
                "  fibrous bands, thick capsule, vascular invasion = carcinoma histology; "
                "  mitoses NOT reliable discriminator. "
                "INTRAOPERATIVE: if carcinoma suspected → en-bloc resection; "
                "  NEVER inadvertently rupture capsule (capsule breach = locoregional seeding → incurable). "
                "RECURRENCE: 50-60% local ± distant (lung, bone, liver); "
                "  18F-choline PET best for restaging; cinacalcet for hypercalcaemia palliation; "
                "  denosumab for skeletal disease. "
                "SURVEILLANCE POST-RESECTION: PTH every 3-6 months; rising PTH = recurrence."
            ),
        },
    ]
    return {
        "atlas": "Hereditary-Primary-Hyperparathyroidism-Atlas",
        "count": len(definitions),
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
