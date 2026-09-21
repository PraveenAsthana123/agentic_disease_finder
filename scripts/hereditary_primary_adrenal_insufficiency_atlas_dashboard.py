#!/usr/bin/env python3
"""Hereditary-Primary-Adrenal-Insufficiency-Atlas — Complete 8-Gene Non-CAH PAI Atlas
AIRE    (autoimmune regulator; 545 aa; 21q22.3; AR;
         APS-1/APECED — autoimmune polyendocrinopathy-candidiasis-ectodermal dystrophy;
         clinical triad: mucocutaneous candidiasis + hypoparathyroidism + primary adrenal insufficiency;
         central tolerance failure → anti-interferon-ω POSITIVE (>98% sensitivity) PATHOGNOMONIC;
         Finnish founder c.769C>T (p.Arg257*); seed SEED_BASE+0) ·
NR0B1   (nuclear receptor subfamily 0, group B, member 1 / DAX1; 470 aa; Xp21.2; XLR;
         X-linked adrenal hypoplasia congenita (XL-AHC) + hypogonadotropic hypogonadism (IHH);
         males only: neonatal/infantile salt-wasting + absent puberty in adolescence;
         fludrocortisone + hydrocortisone; GnRH pulsatile for fertility; seed SEED_BASE+1) ·
AAAS    (aladin / triple A syndrome protein; 546 aa; 12q13.13; AR;
         Allgrove syndrome — triple A (3A): adrenal insufficiency + alacrima + achalasia;
         ACTH-resistant PAI (mineralocorticoid PRESERVED initially); progressive neurodegeneration;
         autonomic dysfunction; CYP21A2 EXCLUDED before diagnosis; seed SEED_BASE+2) ·
MC2R    (melanocortin 2 receptor / ACTH receptor; 297 aa; 18p11.21; AR;
         familial glucocorticoid deficiency type 1 (FGD1);
         PURE glucocorticoid deficiency — mineralocorticoid (aldosterone) PRESERVED;
         tall stature + hyperpigmentation + recurrent hypoglycaemia;
         adrenal crisis without salt-wasting distinguishes from 21-OHD; seed SEED_BASE+3) ·
MRAP    (melanocortin 2 receptor accessory protein; 227 aa; 21q22.11; AR;
         familial glucocorticoid deficiency type 2 (FGD2);
         MRAP chaperones MC2R to cell surface — LOF → ACTH unresponsive despite intact MC2R;
         same clinical phenotype as FGD1 but younger onset; seed SEED_BASE+4) ·
NNT     (nicotinamide nucleotide transhydrogenase; 1086 aa; 5p12; AR;
         familial glucocorticoid deficiency type 5 (FGD5) — mitochondrial oxidative stress model;
         NNT recycles NADPH in mitochondria → LOF → excess mitochondrial ROS → adrenocortical cell death;
         glucocorticoid only; tall stature; seed SEED_BASE+5) ·
TXNRD2  (thioredoxin reductase 2; 524 aa; 22q11.21; AR;
         familial glucocorticoid deficiency type 4 (FGD4) — antioxidant pathway;
         TXNRD2 reduces thioredoxin (TXN2) → LOF → mitochondrial thioredoxin oxidised →
         adrenocortical oxidative death; same phenotype as NNT/FGD5; seed SEED_BASE+6) ·
ABCD1   (ATP-binding cassette subfamily D member 1 / ALDP; 745 aa; Xq28; XLR;
         X-linked adrenoleukodystrophy (X-ALD) — adrenal insufficiency + cerebral demyelination;
         VLCFA accumulate (C26:0 elevated) PATHOGNOMONIC; adrenal crisis before neurological symptoms;
         HSCT curative for cerebral ALD if LOES ≤ 9; Lorenzo's oil slows VLCFA accumulation;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2902–2909)
"""
import random

SEED_BASE = 2902

ATLAS_GENES = [
    {
        "gene": "AIRE",
        "protein": (
            "AIRE -- 21q22.3 AR -- 545aa -- Autoimmune-Regulator-"
            "60kDa-CARD-SAND-PHD-AIRE-TF-"
            "APS1-APECED-Triad-Anti-IFN-omega-Pathognomonic-"
            "OMIM-Gene-607358-Disease-APS1-240300"
        ),
        "locus": "21q22.3",
        "protein_size": (
            "545 aa / 60 kDa (AIRE — autoimmune regulator; "
            "CARD domain + SAND domain (nuclear localisation) + 2 PHD zinc-finger domains; "
            "expressed in medullary thymic epithelial cells (mTECs); "
            "FUNCTION: drives ectopic expression of peripheral tissue antigens in thymus → "
            "  T-cells encountering self antigens undergo negative selection (central tolerance); "
            "AIRE LOF → antigens NOT presented → autoreactive T-cells escape → "
            "  T-effectors attack: adrenals, parathyroids, thyroid, pancreatic islets, gonads; "
            "encoded 21q22.3; Finnish founder: c.769C>T (p.Arg257*) in 1:25,000 Finns; "
            "Sardinian founder: p.Arg139* (15-fold enriched); "
            "dominant-negative heterozygous alleles described in isolated cases"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — AIRE APS-1: "
            "  Both alleles required; 25% recurrence risk in siblings; "
            "  CONSANGUINITY: not required but increases probability; "
            "  Finnish: founder homozygous (p.Arg257*); compound het in others; "
            "  DOMINANT-NEGATIVE: rare p.Cys322* and p.Gly228Trpfs cause AD-like haploinsufficiency "
            "    in some families — reduced penetrance; "
            "  ANTI-IFN-ω ANTIBODIES: 98% sensitivity for APS-1 (screen before gene result); "
            "  PHENOTYPE VARIABILITY: component order varies between patients and families"
        ),
        "disease_category": (
            "APS-1 / APECED — AUTOIMMUNE POLYENDOCRINOPATHY: "
            "  CLASSIC TRIAD (2 of 3 required for diagnosis): "
            "    1. Mucocutaneous candidiasis (CMC): EARLIEST component (infancy/childhood); "
            "       refractory oral/oesophageal/vaginal Candida; fluconazole prophylaxis; "
            "    2. Hypoparathyroidism (HPT): hypocalcaemia + hyperphosphataemia; "
            "       calcification risk; Ca²⁺ + calcitriol lifelong; "
            "    3. Primary adrenal insufficiency (PAI/Addison): "
            "       late component (mean onset 13 years); glucocorticoid + fludrocortisone; "
            "  ADDITIONAL COMPONENTS (~30–70%): "
            "    Autoimmune hypothyroidism, T1D, premature ovarian failure (females), "
            "    autoimmune hepatitis (10–20%; hepatic failure risk), "
            "    ectodermal dystrophy (enamel hypoplasia, nail dystrophy, alopecia), "
            "    pernicious anaemia, vitiligo, intestinal malabsorption; "
            "  ANTI-IFN-ω antibodies: >98% sensitivity, 100% specificity for APS-1; "
            "    screen ALL suspected APS-1 before genetic testing — faster and cheaper"
        ),
        "disease_pathway": (
            "AIRE LOF → CENTRAL TOLERANCE FAILURE → MULTI-ORGAN AUTOIMMUNITY: "
            "  Normal: AIRE in mTECs drives expression of peripheral antigens "
            "    (insulin, CYP11A1, CYP17A1, NALP5, IL-17A/F) → T-cells recognising these → clonal deletion; "
            "  AIRE LOF → peripheral antigens not expressed in thymus → "
            "    autoreactive T-cells survive negative selection → enter periphery; "
            "  Target organs determined by which AIRE-regulated antigens are most affected; "
            "  Anti-IFN-ω: autoreactive B-cells produce anti-interferon antibodies → "
            "    IFN-ω neutralised → paradoxically explains CMC susceptibility (IFN needed for Candida clearance); "
            "  CANDIDA SUSCEPTIBILITY: "
            "    Anti-IL-17A/IL-17F and anti-IL-22 antibodies also produced → "
            "    Th17 cytokines blocked → mucocutaneous immunity impaired → CMC"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC: Anti-interferon-omega (anti-IFN-ω) antibodies POSITIVE (>98% sensitivity) "
            "in patient with mucocutaneous candidiasis + hypoparathyroidism ± PAI. "
            "Finnish founder: p.Arg257* homozygous. "
            "ANTI-IFN-ω is the initial diagnostic test — result available before genetic panel."
        ),
        "treatment": (
            "TREATMENT: "
            "Mucocutaneous candidiasis: fluconazole prophylaxis (lifelong); "
            "  monitor for azole resistance; echinocandin for refractory; "
            "Hypoparathyroidism: Ca²⁺ supplements + calcitriol (or alfacalcidol); "
            "  target Ca²⁺ 2.0–2.2 mmol/L; "
            "  PTH-replacement (recombinant PTH 1-84) if difficult to control; "
            "PAI: hydrocortisone (8–12 mg/m²/day divided) + fludrocortisone; "
            "  sick-day rules + emergency injection kit (IM hydrocortisone); "
            "Autoimmune hepatitis: prednisolone + azathioprine; "
            "Annual surveillance: thyroid function, fasting glucose, haematinics (pernicious anaemia), "
            "  liver enzymes, ovarian function (females) — components accumulate over decades. "
            "IMMUNE CHECKPOINT INHIBITORS: ABSOLUTE CI — risk of catastrophic polyendocrinopathy flare."
        ),
        "seed": 2902,
    },
    {
        "gene": "NR0B1",
        "protein": (
            "NR0B1 -- Xp21.2 XLR -- 470aa -- Nuclear-Receptor-Subfamily-0-B1-DAX1-"
            "51kDa-Orphan-Nuclear-Receptor-"
            "X-Linked-Adrenal-Hypoplasia-Congenita-IHH-"
            "OMIM-Gene-300473-Disease-AHC-300200"
        ),
        "locus": "Xp21.2",
        "protein_size": (
            "470 aa / 51 kDa (DAX1 — dosage-sensitive sex reversal-adrenal hypoplasia critical region "
            "on the X chromosome, gene 1; "
            "N-terminal repeat domain (instead of conventional DBD) + C-terminal ligand-binding-like domain; "
            "atypical orphan nuclear receptor — no confirmed ligand; acts as transcriptional repressor; "
            "expressed in: adrenal cortex, gonads, pituitary, hypothalamus; "
            "FUNCTION: repressor of SF1/NR5A1-driven steroidogenesis; "
            "developmental role in adrenal progenitor zone maintenance; "
            "LOF: adrenal cells fail to mature into functional zona fasciculata/reticularis/glomerulosa → "
            "  adrenal hypoplasia congenita (AHC); "
            "encoded Xp21.2; adjacent to DMD/Duchenne — contiguous gene deletions possible"
        ),
        "inheritance": (
            "X-LINKED RECESSIVE (XLR) — LOSS-OF-FUNCTION — NR0B1/DAX1 AHC: "
            "  MALES affected; females carriers (usually asymptomatic); "
            "  Carrier females: rare partial adrenal insufficiency reported; "
            "  CONTIGUOUS GENE DELETIONS: "
            "    NR0B1 + DMD: adrenal hypoplasia + Duchenne muscular dystrophy; "
            "    NR0B1 + GK: adrenal hypoplasia + glycerol kinase deficiency (hypoglycaemia); "
            "  POINT MUTATIONS: truncating/missense throughout NR0B1; "
            "  X-LINKED: carrier females → X-inactivation usually protects; "
            "    rarely skewed X-inactivation → mild PAI in females"
        ),
        "disease_category": (
            "X-LINKED ADRENAL HYPOPLASIA CONGENITA (XL-AHC) + HYPOGONADOTROPIC HYPOGONADISM (IHH): "
            "  NEONATAL/INFANTILE PRESENTATION: "
            "    Salt-wasting adrenal crisis in first days-weeks of life (75%); "
            "    Cortisol + aldosterone both deficient (cortex agenesis — BOTH glucocorticoid + mineralocorticoid); "
            "    Vomiting, poor feeding, hyponatraemia, hyperkalaemia, hypoglycaemia; "
            "  CHILDHOOD: "
            "    Adrenal insufficiency without crisis if detected early; "
            "    Pigmentation (ACTH elevated); "
            "  ADOLESCENCE: "
            "    ABSENT PUBERTY — hypogonadotropic hypogonadism (IHH): "
            "    LH/FSH low despite absent puberty; testes present (undescended/descended); "
            "    Testosterone low; GnRH pulsatile therapy for pubertal induction; "
            "    Fertility possible with pulsatile GnRH/gonadotropin therapy; "
            "  KEY DISTINCTION FROM 21-OHD: "
            "    21-OHD → adrenal hyperplasia (enlarged); "
            "    AHC → adrenal aplasia/hypoplasia (absent/tiny on imaging)"
        ),
        "disease_pathway": (
            "NR0B1/DAX1 LOF → ADRENAL CORTEX DEVELOPMENT FAILURE + IHH: "
            "  DAX1 represses SF1/NR5A1 target genes; "
            "  Paradox: LOF → reduced SF1 repression → dysregulated steroidogenesis progenitors → "
            "    adrenal cortex fails to organise properly → aplasia/hypoplasia; "
            "  HYPOTHALAMIC-PITUITARY: DAX1 also expressed in GnRH neurons and pituitary → "
            "    LOF → GnRH pulse generator fails → LH/FSH low → IHH; "
            "    Intrinsic pituitary LH response to GnRH: may be normal initially (hypothalamic origin)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC: Neonatal adrenal crisis in MALE + ABSENT PUBERTY in adolescence "
            "(salt-wasting in infancy → IHH in teen years = AHC until proven otherwise). "
            "Adrenal imaging: absent/hypoplastic adrenals (vs. enlarged in CAH). "
            "EXCLUDE CAH (17-OHP normal — steroidogenesis enzymes intact). "
            "Test maternal carrier status (X-linked)."
        ),
        "treatment": (
            "TREATMENT: "
            "PAI: hydrocortisone (8–10 mg/m²/day) + fludrocortisone (0.05–0.15 mg/day); "
            "  LIFELONG; sick-day rules; emergency IM hydrocortisone kit; "
            "IHH: "
            "  Pubertal induction: testosterone (males) — 50mg IM monthly, escalating; "
            "  Fertility: pulsatile GnRH pump or gonadotropins (hCG + FSH); "
            "  DO NOT expect spontaneous puberty — ABSENT without treatment; "
            "CONTIGUOUS GENE DELETION: "
            "  Check for DMD (creatine kinase) and glycerol kinase deficiency (triglycerides). "
            "GENETIC COUNSELLING: X-linked; carrier females generally unaffected; "
            "  each carrier female passes to 50% sons (affected), 50% daughters (carriers)."
        ),
        "seed": 2903,
    },
    {
        "gene": "AAAS",
        "protein": (
            "AAAS -- 12q13.13 AR -- 546aa -- Aladin-WD-Repeat-Nucleoporin-"
            "60kDa-Nuclear-Pore-Complex-WD40-Repeat-"
            "Triple-A-Allgrove-Syndrome-ACTH-Resistant-"
            "OMIM-Gene-605378-Disease-TripleA-231550"
        ),
        "locus": "12q13.13",
        "protein_size": (
            "546 aa / 60 kDa (aladin — named for alacrima in Triple A syndrome; "
            "WD40 repeat domain (β-propeller fold); "
            "NUCLEAR PORE COMPLEX component — localises to cytoplasmic face of NPC; "
            "FUNCTION: nuclear import of DNA repair proteins (PARP1, PCNA, RAD51) and antioxidant enzymes "
            "(ferritin heavy chain, thioredoxin); "
            "LOF: impaired nuclear import → DNA repair deficiency → oxidative stress in adrenal, "
            "  tear duct, oesophageal smooth muscle — selective vulnerability; "
            "encoded 12q13.13; Mediterranean and Middle Eastern founder variants; "
            "WD40 domain: 7-bladed β-propeller; most pathogenic variants cluster in WD repeats"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — AAAS TRIPLE-A SYNDROME: "
            "  Both alleles needed; 25% sibling recurrence; "
            "  FOUNDER VARIANTS: "
            "    p.Ser263Pro: Middle Eastern (Palestinian/Jordanian) founder; "
            "    p.Gly61Arg: Iberian; "
            "    c.1331+1G>A: Ashkenazi Jewish; "
            "  PENETRANCE: nearly complete for alacrima; adrenal insufficiency 80–90%; "
            "    neurodegeneration: variable, progressive; "
            "  DIFFERENTIAL: "
            "    Achalasia in isolation: do not diagnose Triple-A; "
            "    Require at least 2 of 3 triple-A features OR AAAS sequencing"
        ),
        "disease_category": (
            "ALLGROVE / TRIPLE-A SYNDROME — AAAS: "
            "  TRIAD (A-A-A): "
            "    1. Alacrima (EARLIEST — absent/reduced tear secretion; Schirmer test abnormal from infancy); "
            "    2. Achalasia (failure of lower oesophageal sphincter relaxation; dysphagia + vomiting); "
            "    3. Adrenal insufficiency (ACTH-resistant PAI): "
            "       GLUCOCORTICOID DEFICIENCY ONLY initially (mineralocorticoid preserved); "
            "       ACTH markedly elevated; cortisol low; aldosterone initially normal; "
            "       Late: mineralocorticoid may also fail; "
            "  NEUROLOGICAL (late, progressive): "
            "    Peripheral neuropathy (progressive); cerebellar ataxia; "
            "    Bulbar palsy; pyramidal signs; autonomic neuropathy; "
            "    Severe disability in adulthood (neurodegenerative); "
            "  KEY: CYP21A2 sequencing MUST BE NEGATIVE before AAAS diagnosed"
        ),
        "disease_pathway": (
            "AAAS LOF → NUCLEAR PORE DEFECT → ADRENAL/LACRIMAL/OESOPHAGEAL VULNERABILITY: "
            "  Aladin LOF → reduced nuclear import of antioxidant enzymes (ferritin, TXN): "
            "    Adrenocortical cells highly vulnerable to oxidative stress (steroidogenesis generates ROS); "
            "    Lacrimal acini: similarly oxidative stress-sensitive; "
            "    Oesophageal inhibitory myenteric neurons: vulnerable to DNA damage → apoptosis; "
            "  DNA repair proteins (PCNA, PARP1, APE1) impaired nuclear import → "
            "    unrepairedDNA damage → apoptosis of vulnerable cell types; "
            "  ACTH resistance: adrenocortical cells apoptose → ACTH receptor present but no cells to respond"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC: Alacrima (Schirmer test: <5 mm at 5 min) PRESENT FROM INFANCY + "
            "ACTH-resistant PAI (ACTH markedly elevated, cortisol flat on synacthen) + "
            "achalasia on barium swallow or manometry. "
            "Alacrima is the earliest and most consistent feature — ask about dry eyes since infancy. "
            "CYP21A2 excluded (17-OHP normal)."
        ),
        "treatment": (
            "TREATMENT: "
            "Alacrima: artificial tears (frequent; preservative-free); punctal plugs; "
            "Achalasia: pneumatic dilatation or Heller myotomy; PPI; "
            "PAI: hydrocortisone (mineralocorticoid may be normal initially — check annually); "
            "  sick-day rules + emergency IM hydrocortisone; "
            "  fludrocortisone added when aldosterone fails (monitor renin); "
            "Neurological: physiotherapy; OT; speech therapy (bulbar); wheelchair as progresses; "
            "  No disease-modifying treatment for neurodegeneration. "
            "Annual surveillance: glucose (PAI), renin/aldosterone, neurological assessment, "
            "  spirometry (diaphragm weakness), swallow assessment."
        ),
        "seed": 2904,
    },
    {
        "gene": "MC2R",
        "protein": (
            "MC2R -- 18p11.21 AR -- 297aa -- Melanocortin-2-Receptor-ACTH-Receptor-"
            "33kDa-7TM-GPCR-Gs-cAMP-FGD1-"
            "Pure-Glucocorticoid-Deficiency-Mineralocorticoid-Preserved-"
            "OMIM-Gene-607397-Disease-FGD1-202200"
        ),
        "locus": "18p11.21",
        "protein_size": (
            "297 aa / 33 kDa (MC2R — melanocortin 2 receptor / ACTH receptor; "
            "7 transmembrane GPCR (Gs-coupled); "
            "UNIQUE: MC2R only binds ACTH (not other melanocortins MSH); "
            "REQUIRES MRAP (MRAP1) co-chaperone for membrane trafficking; "
            "ACTH → MC2R → Gs → adenylyl cyclase → ↑cAMP → PKA → "
            "  StAR phosphorylation → cholesterol import → cortisol synthesis; "
            "MC2R LOF: ACTH binds but no Gs activation → cortisol absent; "
            "ALDOSTERONE: zona glomerulosa regulated by RAAS (angiotensin II/K⁺), NOT ACTH → "
            "  MINERALOCORTICOID PRESERVED in FGD1; "
            "encoded 18p11.21; p.Ser74Ile: common FGD1 variant in Irish/Northern European"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — MC2R FGD1: "
            "  Both alleles required; 25% sibling recurrence; "
            "  FOUNDER: p.Ser74Ile enriched in Irish descent; "
            "  COMPOUND HET: common in non-consanguineous populations; "
            "  PENETRANCE: complete — MC2R LOF → absent cortisol response to ACTH; "
            "  ALDOSTERONE: NORMAL (zona glomerulosa not affected — no salt-wasting); "
            "  Clinical clue: PAI WITHOUT salt-wasting (hyponatraemia from glucocorticoid deficiency, "
            "    not from mineralocorticoid deficiency)"
        ),
        "disease_category": (
            "FGD1 — FAMILIAL GLUCOCORTICOID DEFICIENCY TYPE 1 — PURE GLUCOCORTICOID DEFICIENCY: "
            "  ONSET: neonatal to early childhood (hypoglycaemia + hyperpigmentation); "
            "  CLINICAL FEATURES: "
            "    Hypoglycaemia (recurrent; morning hypoglycaemia from absent cortisol counter-regulation); "
            "    Hyperpigmentation (ACTH markedly elevated → MCR1 stimulation → melanin); "
            "    TALL STATURE (ACTH → sex hormone precursors → bone age advancement; adrenal androgens); "
            "    Seizures (from hypoglycaemia); "
            "  ABSENCE OF SALT-WASTING: "
            "    Aldosterone NORMAL → no hyponatraemia + hyperkalaemia; "
            "    This DISTINGUISHES FGD from 21-hydroxylase deficiency (CAH) and AHC; "
            "  ADRENAL SIZE: "
            "    Initially LARGE (ACTH drives adrenal growth without cortisol feedback); "
            "    Later adrenal atrophy as adrenocortical cells apoptose"
        ),
        "disease_pathway": (
            "MC2R LOF → ACTH UNRESPONSIVE → ABSENT CORTISOL → GLUCOCORTICOID DEFICIENCY: "
            "  ACTH secreted normally (ACTH is NOT the problem); "
            "  MC2R LOF → ACTH cannot bind productively → Gs not activated → cAMP not raised → "
            "    PKA inactive → StAR not phosphorylated → cholesterol not transported → "
            "    steroidogenesis stalled at FIRST STEP; "
            "  ACTH feedback absent → pituitary CRH/ACTH released unchecked → "
            "    very high ACTH levels → MCR1 stimulated → hyperpigmentation; "
            "  ACTH-driven adrenal growth (via another pathway) → adrenomegaly early; "
            "  MINERALOCORTICOID: zona glomerulosa uses angiotensin II + K⁺ sensors → intact"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC: Markedly elevated ACTH + absent cortisol response on synacthen + "
            "NORMAL aldosterone + NORMAL renin (no salt-wasting) + "
            "tall stature + hyperpigmentation + recurrent hypoglycaemia. "
            "Distinguishes from 21-OHD (elevated 17-OHP + salt-wasting in CAH)."
        ),
        "treatment": (
            "TREATMENT: "
            "Hydrocortisone (8–10 mg/m²/day; physiological replacement); "
            "  fludrocortisone NOT required (mineralocorticoid normal); "
            "Hypoglycaemia: glucose gel/IV as immediate treatment; "
            "  hydrocortisone corrects long-term; "
            "Sick-day rules: double/triple dose during illness; IM hydrocortisone kit; "
            "Growth monitoring: tall stature due to adrenal androgens; "
            "  bone age + growth charts annually; "
            "  avoid excess hydrocortisone (suppresses growth); "
            "DHEAS monitoring (adrenal androgen insufficiency in some)."
        ),
        "seed": 2905,
    },
    {
        "gene": "MRAP",
        "protein": (
            "MRAP -- 21q22.11 AR -- 227aa -- Melanocortin-2-Receptor-Accessory-Protein-"
            "25kDa-Single-Pass-TM-Chaperone-"
            "FGD2-ACTH-Receptor-Trafficking-Failure-"
            "OMIM-Gene-609196-Disease-FGD2-607398"
        ),
        "locus": "21q22.11",
        "protein_size": (
            "227 aa / 25 kDa (MRAP — melanocortin 2 receptor accessory protein; "
            "type I single-pass transmembrane protein; "
            "forms antiparallel homodimers in ER membrane; "
            "FUNCTION: essential chaperone for MC2R: "
            "  binds newly synthesised MC2R in ER → assists folding and glycosylation → "
            "  escorts MC2R to plasma membrane (trafficking); "
            "MRAP LOF: MC2R trapped in ER → not expressed at cell surface → "
            "  ACTH cannot bind → identical phenotype to MC2R LOF (FGD1); "
            "encoded 21q22.11; functionally distinguished from MC2R only by gene sequencing; "
            "MRAP2 (closely related gene): expressed in brain; LOF linked to obesity/hyperphagia (not PAI)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — MRAP FGD2: "
            "  Both alleles required; 25% recurrence; "
            "  Often compound heterozygous; "
            "  CLINICAL: identical to FGD1 (MC2R LOF) — distinguished ONLY by gene sequencing; "
            "  FGD2 onset: typically EARLIER and more SEVERE than FGD1 (average ~3 months vs. ~2 years); "
            "  NEONATAL PRESENTATION common (hypoglycaemia + hyperpigmentation in first months); "
            "  ALDOSTERONE: NORMAL (same as FGD1)"
        ),
        "disease_category": (
            "FGD2 — FAMILIAL GLUCOCORTICOID DEFICIENCY TYPE 2: "
            "  NEONATAL/EARLY INFANTILE ONSET: "
            "    Hypoglycaemia + hyperpigmentation in first 3 months of life; "
            "    More severe than FGD1 on average; neonatal adrenal crisis; "
            "  CLINICAL FEATURES: same as FGD1: "
            "    Pure glucocorticoid deficiency — NO salt-wasting; "
            "    ACTH markedly elevated; cortisol absent on synacthen; "
            "    Aldosterone NORMAL (renin NORMAL); "
            "    Tall stature + hyperpigmentation; "
            "    Recurrent hypoglycaemia ± hypoglycaemic seizures; "
            "  DIFFERENTIATION FROM FGD1: "
            "    Earlier onset; no genotype-phenotype clue from clinical features alone; "
            "    Gene panel needed: sequence both MC2R and MRAP together"
        ),
        "disease_pathway": (
            "MRAP LOF → MC2R MISFOLDED/TRAPPED IN ER → ACTH UNRESPONSIVE: "
            "  MRAP essential for MC2R ER exit: forms antiparallel dimer (one N-in/C-out, one N-out/C-in); "
            "  MRAP LOF: MC2R synthesised but cannot fold correctly in ER → "
            "    ER-associated degradation (ERAD) → MC2R absent from adrenocortical cell surface; "
            "  ACTH: CANNOT signal (no surface receptor to bind); "
            "  cAMP pathway downstream: intact — recapitulated by cAMP analogues experimentally; "
            "  ZONA GLOMERULOSA: MRAP expressed but aldosterone pathway uses AT1R/Ca²⁺, not MC2R → "
            "    aldosterone unaffected"
        ),
        "pathognomonic": (
            "NO clinical feature distinguishes FGD2 (MRAP) from FGD1 (MC2R). "
            "BIOCHEMICAL PATTERN: elevated ACTH + absent cortisol + normal aldosterone + normal renin. "
            "Earlier onset than FGD1 (median 3 months) is a clue but not diagnostic. "
            "PANEL SEQUENCING of MC2R + MRAP together is standard — phenotype identical."
        ),
        "treatment": (
            "TREATMENT: Identical to FGD1 (MC2R). "
            "Hydrocortisone replacement (physiological; no fludrocortisone needed); "
            "Urgent IV glucose + emergency hydrocortisone for hypoglycaemic crisis; "
            "Sick-day rules; IM hydrocortisone kit; "
            "Bone age/growth surveillance (tall stature from adrenal androgen excess); "
            "Neonatal screening protocol: "
            "  FGD should be on differential for neonatal hypoglycaemia + hyperpigmentation — "
            "  serum cortisol + ACTH + 17-OHP (exclude CAH); "
            "  confirm with synacthen test if initial cortisol equivocal."
        ),
        "seed": 2906,
    },
    {
        "gene": "NNT",
        "protein": (
            "NNT -- 5p12 AR -- 1086aa -- Nicotinamide-Nucleotide-Transhydrogenase-"
            "114kDa-Inner-Mitochondrial-Membrane-"
            "FGD5-Mitochondrial-NADPH-Antioxidant-Defence-"
            "OMIM-Gene-607878-Disease-FGD5-617825"
        ),
        "locus": "5p12",
        "protein_size": (
            "1086 aa / 114 kDa (NNT — nicotinamide nucleotide transhydrogenase; "
            "inner mitochondrial membrane protein; "
            "catalyses: NADH + NADP⁺ → NAD⁺ + NADPH (coupled to H⁺ transport across IMM); "
            "uses mitochondrial membrane potential to regenerate NADPH; "
            "NADPH is essential electron donor for: glutathione reductase (recycling GSH), "
            "  thioredoxin reductase 2 (TXNRD2), catalase; "
            "LOF: mitochondrial NADPH depleted → GSH not recycled → excess H₂O₂ accumulates → "
            "  adrenocortical cell oxidative death (zona fasciculata highly vulnerable — "
            "  highest cytochrome P450 activity of any cell type = highest mitochondrial ROS); "
            "INBRED C57BL/6J mice: spontaneous NNT frameshift → hypocortisolaemia (model organism proof)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — NNT FGD5: "
            "  Both alleles needed; 25% recurrence; "
            "  Rare — few families reported worldwide; "
            "  CONSANGUINITY common in reported cases; "
            "  PENETRANCE: adrenal insufficiency penetrant; "
            "  PHENOTYPE: pure glucocorticoid deficiency (same as FGD1/FGD2); "
            "  FGD5 may be systematically underdiagnosed — NNT often not on standard MODY/FGD panels"
        ),
        "disease_category": (
            "FGD5 — FAMILIAL GLUCOCORTICOID DEFICIENCY TYPE 5 — NNT: "
            "  ONSET: infancy to childhood; "
            "  CLINICAL: "
            "    Pure glucocorticoid deficiency (same as FGD1/2); "
            "    ACTH elevated, cortisol absent; aldosterone NORMAL; renin NORMAL; "
            "    Hypoglycaemia + hyperpigmentation + tall stature; "
            "    NO salt-wasting; "
            "  DISTINCTION FROM FGD1/2: "
            "    No reliable clinical distinction; gene panel needed; "
            "    ACTH response (synacthen): flat cortisol; "
            "    NNT identified in FGD patients with negative MC2R + MRAP sequencing"
        ),
        "disease_pathway": (
            "NNT LOF → MITOCHONDRIAL NADPH DEPLETION → ADRENOCORTICAL OXIDATIVE CELL DEATH: "
            "  Adrenocortical cells (zona fasciculata): highest density of cytochrome P450 (CYP11A1, "
            "    CYP11B1, CYP17A1) → massive ROS generation during steroidogenesis; "
            "  NNT normally: uses IMM proton gradient → drives NADH → NADPH (reverse thermodynamic); "
            "  NNT LOF: NADPH not regenerated → GSH/GS-system depleted → H₂O₂ accumulates → "
            "    lipid peroxidation + protein oxidation → adrenocortical apoptosis; "
            "  Zona glomerulosa less severely affected initially (lower steroidogenic rate); "
            "  NNT POLYMORPHISM in C57BL/6J mice: 5-exon deletion → hypocortisolaemia → "
            "    widely studied HPA-axis phenotype model (confirmed NNT as adrenal gene)"
        ),
        "pathognomonic": (
            "NO clinical distinction from other FGD subtypes. "
            "Pattern: elevated ACTH + absent cortisol + normal aldosterone + negative CYP21A2, MC2R, MRAP. "
            "Identified by extended gene panel including NNT. "
            "Mitochondrial dysfunction screening (lactate, pyruvate) may be mildly abnormal in some."
        ),
        "treatment": (
            "TREATMENT: Identical to FGD1/FGD2. "
            "Hydrocortisone (physiological replacement — 8–10 mg/m²/day); "
            "No fludrocortisone required; "
            "Hypoglycaemia management; sick-day rules; IM hydrocortisone kit; "
            "ANTIOXIDANT supplementation: theoretical benefit (vitamin C/E, CoQ10) — no RCT evidence; "
            "NNT panel: order NNT sequencing when FGD phenotype but MC2R + MRAP negative. "
            "Family cascade: AR — test siblings."
        ),
        "seed": 2907,
    },
    {
        "gene": "TXNRD2",
        "protein": (
            "TXNRD2 -- 22q11.21 AR -- 524aa -- Thioredoxin-Reductase-2-"
            "57kDa-Selenoprotein-Mitochondrial-FAD-Oxidoreductase-"
            "FGD4-Thioredoxin-Antioxidant-Defence-"
            "OMIM-Gene-606448-Disease-FGD4-614736"
        ),
        "locus": "22q11.21",
        "protein_size": (
            "524 aa / 57 kDa (TXNRD2 — mitochondrial thioredoxin reductase 2; "
            "SELENOPROTEIN: contains selenocysteine (Sec) at C-terminus (TGA codon decoded as Sec); "
            "FAD-containing oxidoreductase homodimer; "
            "FUNCTION: reduces oxidised mitochondrial thioredoxin (TXN2-S₂) back to TXN2-(SH)₂ "
            "  using NADPH as electron donor; "
            "TXN2 then reduces: peroxiredoxin 3 (PRDX3), ribonucleotide reductase (DNA synthesis); "
            "TXNRD2 LOF: TXN2 trapped oxidised → PRDX3 inactivated → mitochondrial H₂O₂ → "
            "  adrenocortical oxidative apoptosis (same mechanism as NNT but different step); "
            "located in chromosome 22q11.21 — the DiGeorge/22q11DS locus"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — TXNRD2 FGD4: "
            "  Both alleles required; 25% recurrence; "
            "  RARE — reported in isolated consanguineous families; "
            "  PHENOTYPE: identical to FGD1/2/5 — pure glucocorticoid deficiency; "
            "  22q11.21 LOCATION: TXNRD2 gene at DiGeorge region — "
            "    haploinsufficiency (22q11DS) does NOT cause FGD4 (biallelic required); "
            "  SELENIUM DEPENDENCY: TXNRD2 selenoprotein — selenium deficiency → reduced activity; "
            "    selenium deficiency can mimic partial FGD4"
        ),
        "disease_category": (
            "FGD4 — FAMILIAL GLUCOCORTICOID DEFICIENCY TYPE 4: "
            "  ONSET: infancy/early childhood; "
            "  CLINICAL: "
            "    Pure glucocorticoid deficiency — same as FGD1/2/5; "
            "    ACTH elevated; cortisol absent on synacthen; "
            "    Aldosterone NORMAL; renin NORMAL (no salt-wasting); "
            "    Hypoglycaemia + hyperpigmentation + tall stature; "
            "  CARDIAC: "
            "    Dilated cardiomyopathy reported in some TXNRD2-deficient individuals "
            "    (cardiac mitochondria vulnerable to oxidative stress); "
            "  DISTINCTION: FGD4 may present with cardiac manifestations alongside PAI — "
            "    prompts TXNRD2 testing in FGD + cardiomyopathy combination"
        ),
        "disease_pathway": (
            "TXNRD2 LOF → TXN2 TRAPPED OXIDISED → ADRENOCORTICAL MITOCHONDRIAL STRESS: "
            "  TXNRD2 → TXN2(SH)₂ → PRDX3(2SH) (active peroxidase) → H₂O₂ eliminated; "
            "  TXNRD2 LOF: PRDX3 remains oxidised → H₂O₂ accumulates in mitochondria → "
            "    mtDNA damage + protein oxidation → adrenocortical cell apoptosis; "
            "  CARDIAC: "
            "    Cardiomyocytes: high mitochondrial activity → TXNRD2 critical; "
            "    LOF → oxidative cardiomyopathy (dilated); "
            "  SELENIUM: TXNRD2 Sec residue (Sec498) essential for catalysis → "
            "    selenium deficiency → reduced TXNRD2 function → partial FGD-like phenotype"
        ),
        "pathognomonic": (
            "GLUCOCORTICOID DEFICIENCY: same biochemical pattern as all FGD subtypes. "
            "CARDIOMYOPATHY + FGD: combination raises TXNRD2 suspicion specifically. "
            "FGD4 identified by extended panel when MC2R + MRAP + NNT negative. "
            "Check selenium levels (low selenium can reduce TXNRD2 activity)."
        ),
        "treatment": (
            "TREATMENT: "
            "Hydrocortisone (physiological replacement); no fludrocortisone required; "
            "Hypoglycaemia prevention + sick-day rules; IM hydrocortisone kit; "
            "CARDIOMYOPATHY: "
            "  Echo at diagnosis + annual surveillance; "
            "  ACE inhibitor/ARB + beta-blocker for dilated CM; "
            "  Paediatric cardiology co-management; "
            "SELENIUM supplementation: selenium-yeast 50–100 μg/day if selenium low; "
            "Family cascade: AR — test siblings + parents (carrier testing)."
        ),
        "seed": 2908,
    },
    {
        "gene": "ABCD1",
        "protein": (
            "ABCD1 -- Xq28 XLR -- 745aa -- ATP-Binding-Cassette-Subfamily-D-Member-1-"
            "84kDa-Peroxisomal-VLCFA-Transporter-"
            "X-Linked-Adrenoleukodystrophy-XL-ALD-"
            "VLCFA-C26-0-Pathognomonic-HSCT-Cerebral-ALD-LOES-le-9-"
            "OMIM-Gene-300371-Disease-XALD-300100"
        ),
        "locus": "Xq28",
        "protein_size": (
            "745 aa / 84 kDa (ABCD1 — adrenoleukodystrophy protein / ALDP; "
            "peroxisomal half-transporter (forms homodimer with itself, or heterodimers with ABCD2/3); "
            "NBD (nucleotide-binding domain) + TMD (6 transmembrane helices); "
            "localised to peroxisomal membrane; "
            "FUNCTION: transports very long-chain acyl-CoA (VLCFA-CoA, ≥ C22) into peroxisome "
            "  for β-oxidation; "
            "LOF: VLCFA accumulate in plasma + adrenocortical cells + CNS white matter → "
            "  adrenocortical lipid storage → adrenocortical insufficiency; "
            "  CNS: demyelination (cerebral ALD in boys, adrenomyeloneuropathy in adults); "
            "C26:0 (hexacosanoic acid) elevated in plasma/erythrocytes — PATHOGNOMONIC; "
            "encoded Xq28; >800 pathogenic variants described; no founder variant"
        ),
        "inheritance": (
            "X-LINKED RECESSIVE (XLR) — LOSS-OF-FUNCTION — ABCD1 X-ALD: "
            "  MALES: nearly universal X-ALD manifestations by ~50 years; "
            "  CLINICAL PHENOTYPES IN MALES: "
            "    Cerebral ALD (cALD): 35–40% boys (peak 4–12 years) — fatal without HSCT; "
            "    Adrenomyeloneuropathy (AMN): spinal cord + peripheral neuropathy in adults (age 20–40); "
            "    PAI only (10%): no neurological disease; "
            "    Asymptomatic (10%): VLCFA elevated, no clinical disease yet; "
            "  FEMALES (carrier, heterozygous): "
            "    80% develop AMN-like myelopathy by ~60 years; "
            "    PAI: 1–2%; cALD: <1%; "
            "  GENOTYPE-PHENOTYPE: NONE — same variant can cause cALD in one brother, AMN in another"
        ),
        "disease_category": (
            "X-LINKED ADRENOLEUKODYSTROPHY (X-ALD): "
            "  PRIMARY ADRENAL INSUFFICIENCY: "
            "    Adrenal crisis may be the FIRST presentation (before neurological symptoms); "
            "    ACTH elevated; cortisol absent; aldosterone ± affected; "
            "    VLCFA accumulate in adrenal cortex → lipid inclusions → adrenocortical failure; "
            "    PLASMA C26:0 elevated in all males with ABCD1 mutations — SCREEN ALL PAI MALES; "
            "  CEREBRAL ALD (cALD — boys 4–12 years): "
            "    Inflammatory cerebral demyelination (posterior white matter, PATHOGNOMONIC on MRI); "
            "    Loes MRI score (0–34): severity of white matter lesions; "
            "    HSCT CURATIVE if LOES ≤ 9 and neurological deficit mild; "
            "    If LOES > 9: HSCT still may slow — but poor prognosis; "
            "    FATAL untreated within 2–4 years; "
            "  AMN (adrenomyeloneuropathy — adults): "
            "    Spastic paraparesis (pyramidal tract); "
            "    Peripheral neuropathy; bladder dysfunction; "
            "    NOT inflammatory — does not respond to HSCT; "
            "    Lorenzo's oil: slows VLCFA accumulation in pre-symptomatic; "
            "  NEW TREATMENT: elivaldogene autotemcel (Lenti-D/ALD-101 gene therapy) — FDA 2022 approval "
            "    for cerebral ALD in boys"
        ),
        "disease_pathway": (
            "ABCD1 LOF → VLCFA ACCUMULATE IN PEROXISOME-DEFICIENT CELLS → ADRENAL + CNS DAMAGE: "
            "  ABCD1 LOF: VLCFA-CoA not imported into peroxisomes → cytoplasmic accumulation → "
            "    VLCFA esterified into lipids → lipid droplets in adrenocortical cells + CNS; "
            "  ADRENAL: "
            "    VLCFA-laden lipid droplets disrupt StAR/CYP11A1 steroidogenesis → adrenocortical failure; "
            "    Can cause both glucocorticoid AND mineralocorticoid deficiency (unlike pure FGDs); "
            "  CNS — CEREBRAL ALD: "
            "    Mechanism unclear but inflammatory: VLCFA → microglial activation → "
            "    MCP-1, TNF-α, IL-1β → BBB breakdown → T-cell infiltration → demyelination cascade; "
            "    Posterior fossa + corticospinal tract: earliest affected; "
            "  BLOOD BIOMARKER: C26:0-lyso-PC (lysophosphatidylcholine with C26:0): "
            "    sensitive and specific for X-ALD; used in neonatal screening (ENSP/USA newborn panel)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC: Plasma VLCFA C26:0 elevated (and C26:0/C22:0 ratio elevated) in all ABCD1 males. "
            "C26:0-lyso-PC: neonatal screening marker. "
            "MRI: symmetric posterior parieto-occipital white matter signal (gadolinium enhancement = active cALD). "
            "LOES score ≤ 9: window for HSCT before irreversible damage. "
            "SCREEN: ALL male PAI patients for VLCFA — adrenal failure may precede neurological disease."
        ),
        "treatment": (
            "TREATMENT: "
            "PAI: hydrocortisone ± fludrocortisone (both zones affected — check aldosterone/renin); "
            "  sick-day rules + emergency IM hydrocortisone kit; "
            "Cerebral ALD (BOYS, LOES ≤ 9, early neurological: "
            "  HSCT (allogeneic stem cell transplant): CURATIVE if performed early; "
            "  Lenti-D/elivaldogene autotemcel (gene therapy): FDA 2022 — autologous; "
            "  MONITOR: annual brain MRI from diagnosis (all boys) — catch cALD EARLY; "
            "  LOES > 9: poor HSCT outcome — consider gene therapy or palliative; "
            "AMN: no disease-modifying treatment; "
            "  physiotherapy; bladder management; baclofen for spasticity; "
            "Lorenzo's oil (GTO:GTE 4:1 mixture): "
            "  Normalises VLCFA in blood; "
            "  Pre-symptomatic: slows cerebral MRI changes; "
            "  After neurological onset: no benefit; "
            "NEONATAL SCREENING: C26:0-lyso-PC (several US states + Europe); "
            "  allows pre-symptomatic HSCT in boys before cALD onset."
        ),
        "seed": 2909,
    },
]


def _rng(seed):
    return random.Random(seed)


def _generate_patients(entry):
    rng = _rng(entry["seed"])
    gene = entry["gene"]
    n = 40
    patients = []
    for i in range(n):
        # Age of adrenal crisis/diagnosis in months for neonatal/infantile presentations
        if gene in ("NR0B1", "MRAP"):
            age_mos = rng.randint(0, 6)  # neonatal
        elif gene in ("MC2R", "NNT", "TXNRD2"):
            age_mos = rng.randint(1, 36)  # infant to toddler
        elif gene == "AAAS":
            age_mos = rng.randint(12, 120)  # childhood
        elif gene == "AIRE":
            age_mos = rng.randint(12, 144)  # childhood/adolescence
        else:  # ABCD1
            age_mos = rng.randint(12, 180)  # childhood/adult
        sex = rng.choice(["M", "F"])
        if gene in ("NR0B1", "ABCD1"):
            sex = "M"  # X-linked, males primarily
        cortisol_nm = round(rng.uniform(20, 150), 1)  # low (normal >450 on stimulation)
        acth_pmol = round(rng.uniform(50, 400), 1)  # high
        aldosterone_deficient = gene in ("NR0B1",) or (gene == "ABCD1" and rng.random() < 0.5)
        glucocorticoid_only = gene in ("MC2R", "MRAP", "NNT", "TXNRD2", "AAAS")
        hypoglycaemia = gene in ("MC2R", "MRAP", "NNT", "TXNRD2") and rng.random() < 0.7
        tall_stature = gene in ("MC2R", "MRAP", "NNT", "TXNRD2") and rng.random() < 0.6
        patients.append({
            "id": f"{gene}-{i+1:02d}",
            "gene": gene,
            "age_diagnosis_months": age_mos,
            "sex": sex,
            "cortisol_basal_nmol_l": cortisol_nm,
            "acth_pmol_l": acth_pmol,
            "aldosterone_deficient": aldosterone_deficient,
            "glucocorticoid_only": glucocorticoid_only,
            "hypoglycaemia": hypoglycaemia,
            "tall_stature": tall_stature,
        })
    return patients


def generate_overview():
    genes = [g["gene"] for g in ATLAS_GENES]
    total_patients = 0
    gene_rows = []
    for entry in ATLAS_GENES:
        pts = _generate_patients(entry)
        total_patients += len(pts)
        avg_age = round(sum(p["age_diagnosis_months"] for p in pts) / len(pts), 1)
        avg_acth = round(sum(p["acth_pmol_l"] for p in pts) / len(pts), 1)
        gc_only_n = sum(1 for p in pts if p["glucocorticoid_only"])
        gene_rows.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_summary": entry["protein"],
            "patients": len(pts),
            "avg_age_diagnosis_months": avg_age,
            "avg_acth_pmol_l": avg_acth,
            "glucocorticoid_only_pct": round(100 * gc_only_n / len(pts)),
            "mineralocorticoid_affected": entry["gene"] in ("AIRE", "NR0B1", "ABCD1"),
        })
    return {
        "atlas": "Hereditary Primary Adrenal Insufficiency Atlas",
        "subtitle": "8-Gene Reference: AIRE-NR0B1-AAAS-MC2R-MRAP-NNT-TXNRD2-ABCD1",
        "description": (
            "Comprehensive atlas of hereditary non-CAH primary adrenal insufficiency, "
            "covering the eight major non-congenital adrenal hyperplasia genetic causes: "
            "AIRE (APS-1/APECED — autoimmune, anti-IFN-ω pathognomonic), "
            "NR0B1/DAX1 (X-linked AHC + IHH), "
            "AAAS (Triple A syndrome — alacrima + achalasia + PAI), "
            "MC2R + MRAP (FGD1/FGD2 — pure glucocorticoid deficiency), "
            "NNT + TXNRD2 (FGD5/FGD4 — mitochondrial oxidative stress), "
            "ABCD1 (X-ALD — adrenal + cerebral demyelination, VLCFA pathognomonic). "
            "320 patients (8 × 40), seeds 2902-2909."
        ),
        "total_patients": total_patients,
        "total_genes": len(genes),
        "genes": genes,
        "gene_rows": gene_rows,
        "categories": {
            "Autoimmune (central tolerance failure)": ["AIRE"],
            "X-linked developmental (adrenal aplasia)": ["NR0B1"],
            "Nuclear pore / neurodegeneration": ["AAAS"],
            "ACTH receptor / chaperone (FGD1/2)": ["MC2R", "MRAP"],
            "Mitochondrial antioxidant (FGD4/5)": ["TXNRD2", "NNT"],
            "Peroxisomal VLCFA (X-ALD)": ["ABCD1"],
        },
        "key_facts": [
            "AIRE (APS-1): anti-IFN-ω antibodies >98% sensitive — test BEFORE genetic panel",
            "NR0B1/DAX1 (X-AHC): neonatal salt-wasting in males + absent puberty in adolescence = AHC until proven otherwise",
            "AAAS (Triple A): alacrima (Schirmer test) is EARLIEST feature — present from infancy",
            "MC2R/MRAP (FGD1/2): pure glucocorticoid deficiency — aldosterone PRESERVED, NO salt-wasting",
            "NNT/TXNRD2 (FGD5/4): mitochondrial oxidative stress; FGD4 + cardiomyopathy = TXNRD2",
            "ABCD1 (X-ALD): plasma VLCFA C26:0 elevated — screen ALL male PAI patients",
            "X-ALD: adrenal crisis may PRECEDE cerebral demyelination — HSCT curative if LOES ≤ 9",
            "FGD1/2/4/5: NO salt-wasting — distinguishes from 21-hydroxylase deficiency (CAH)",
        ],
        "diagnostic_algorithm": (
            "PAI workup (exclude CAH first with 17-OHP, then proceed): "
            "1. 17-OHP: if elevated → CYP21A2 (classical CAH) — not in this atlas; "
            "2. Anti-IFN-ω antibodies: if positive → AIRE/APS-1 (test before genetic panel); "
            "3. Male infant with salt-wasting + absent gonads imaging: NR0B1 (X-AHC); "
            "4. Plasma VLCFA (C26:0): if elevated → ABCD1 (X-ALD) — screen ALL males with PAI; "
            "5. Alacrima + achalasia: AAAS (Triple A) — Schirmer test + barium swallow; "
            "6. Pure glucocorticoid deficiency (no salt-wasting) + synacthen flat: "
            "   FGD panel: MC2R → MRAP → NNT → TXNRD2 (in order of frequency); "
            "7. FGD + cardiomyopathy: TXNRD2 first; "
            "8. All negative + PAI: consider STAR, CYP11A1 (in Hereditary-Adrenal-Steroidogenesis-Atlas)"
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
                "term": "Primary Adrenal Insufficiency (PAI) — Addison's Disease Spectrum",
                "definition": (
                    "PAI: adrenal cortex fails to produce cortisol ± aldosterone despite normal/high ACTH. "
                    "Autoimmune Addison's: most common in adults (anti-21-hydroxylase antibodies). "
                    "HEREDITARY NON-CAH PAI: "
                    "  Exclude CAH (17-OHP screen, CYP21A2 sequencing) FIRST; "
                    "  Then systematic genetic workup by phenotype (this atlas). "
                    "EMERGENCY: adrenal crisis — IV saline 0.9% + IV hydrocortisone 100 mg bolus; "
                    "  carry IM hydrocortisone kit (buccal prednisolone/rectal hydrocortisone as backup). "
                    "SICK-DAY RULE: double/triple hydrocortisone dose during fever/illness."
                ),
            },
            {
                "term": "APS-1 / APECED — Anti-IFN-ω Screening",
                "definition": (
                    "Autoimmune Polyendocrinopathy Candidiasis Ectodermal Dystrophy. "
                    "DIAGNOSTIC CLUE: anti-interferon-omega (anti-IFN-ω) antibodies >98% sensitive. "
                    "Test anti-IFN-ω BEFORE waiting for AIRE gene sequencing result. "
                    "Anti-IL-17A and anti-IL-22 also present → impaired Th17 → Candida susceptibility. "
                    "CLASSIC TRIAD: mucocutaneous candidiasis + hypoparathyroidism + PAI. "
                    "COMPONENT ORDER: candidiasis first (infancy) → HPT → PAI (teen years) typically. "
                    "IMMUNE CHECKPOINT INHIBITORS ABSOLUTE CI: risk of catastrophic polyendocrinopathy."
                ),
            },
            {
                "term": "Familial Glucocorticoid Deficiency (FGD) — Types 1-5",
                "definition": (
                    "FGD: ACTH-resistant cortisol deficiency with NORMAL mineralocorticoid (aldosterone preserved). "
                    "KEY DISCRIMINATOR FROM CAH: no salt-wasting — aldosterone and renin NORMAL. "
                    "TYPES: "
                    "  FGD1: MC2R (ACTH receptor) — Irish founder p.Ser74Ile; "
                    "  FGD2: MRAP (MC2R chaperone) — earlier onset, neonatal; "
                    "  FGD3: MCM4 (DNA helicase); "
                    "  FGD4: TXNRD2 (thioredoxin reductase) — + cardiomyopathy clue; "
                    "  FGD5: NNT (NADPH regeneration) — C57BL/6J mouse model. "
                    "TALL STATURE + HYPERPIGMENTATION: ACTH-driven adrenal androgens → growth acceleration. "
                    "TREATMENT: hydrocortisone only — NO fludrocortisone needed."
                ),
            },
            {
                "term": "X-Linked Adrenoleukodystrophy (X-ALD) — VLCFA Screening",
                "definition": (
                    "ABCD1 LOF → VLCFA (C26:0) accumulate → adrenal + CNS damage. "
                    "PLASMA C26:0 ELEVATED: pathognomonic — screen ALL males with unexplained PAI. "
                    "PHENOTYPE SPECTRUM (same variant → different outcomes): "
                    "  Cerebral ALD (cALD): 4–12-year-old boys — aggressive demyelination; "
                    "  Adrenomyeloneuropathy (AMN): adult males — slowly progressive myelopathy; "
                    "  PAI only; asymptomatic. "
                    "MRI BRAIN: annual from diagnosis in boys — posterior parieto-occipital WM signal. "
                    "LOES SCORE ≤ 9: HSCT window (or elivaldogene/Lenti-D gene therapy). "
                    "NEONATAL SCREENING: C26:0-lyso-PC detects pre-symptomatic boys → treat before cALD."
                ),
            },
            {
                "term": "Triple A Syndrome (AAAS / Allgrove)",
                "definition": (
                    "3A = Adrenal insufficiency + Alacrima + Achalasia. "
                    "ALACRIMA: absent/severely reduced tearing — present from BIRTH. "
                    "  Schirmer test <5 mm at 5 min. "
                    "  Earliest feature — ask about dry eyes in infancy. "
                    "ACHALASIA: failure of lower oesophageal sphincter relaxation → dysphagia/vomiting. "
                    "ACTH RESISTANCE: ACTH elevated; cortisol absent; "
                    "  MINERALOCORTICOID INITIALLY PRESERVED (can fail later). "
                    "NEURODEGENERATION: progressive — peripheral neuropathy, ataxia, bulbar palsy. "
                    "  No treatment for neurological component. "
                    "DIFFERENTIAL: CYP21A2 negative before AAAS labelled."
                ),
            },
            {
                "term": "X-Linked Adrenal Hypoplasia Congenita (NR0B1/DAX1)",
                "definition": (
                    "NR0B1/DAX1 LOF → adrenal cortex aplasia + IHH (hypogonadotropic hypogonadism). "
                    "NEONATAL: salt-wasting crisis in first weeks of life in males (75%). "
                    "  Cortisol + aldosterone BOTH low (full cortex aplasia — unlike FGD). "
                    "  DISTINGUISH FROM CAH: 17-OHP NORMAL (steroidogenic enzymes intact). "
                    "ADOLESCENCE: absent puberty (LH/FSH low) in males who survived infancy. "
                    "  CONTIGUOUS GENE: check for DMD (CK) + glycerol kinase deficiency (TG). "
                    "TREATMENT: HC + fludrocortisone lifelong; pulsatile GnRH for puberty/fertility. "
                    "IMAGING: absent/hypoplastic adrenals on CT (vs. enlarged adrenals in CAH)."
                ),
            },
            {
                "term": "Mitochondrial Antioxidant FGD (NNT and TXNRD2) — Oxidative Stress Mechanism",
                "definition": (
                    "NNT (FGD5) and TXNRD2 (FGD4): mitochondrial NADPH/thioredoxin antioxidant pathway. "
                    "NNT: regenerates NADPH from NADH using IMM proton gradient. "
                    "TXNRD2: uses NADPH to recycle mitochondrial thioredoxin (TXN2). "
                    "TXN2 → PRDX3 (H₂O₂ scavenger): LOF of either → excess H₂O₂ → adrenocortical apoptosis. "
                    "ADRENAL VULNERABILITY: zona fasciculata = highest cytochrome P450 density → "
                    "  highest mitochondrial ROS production → exquisitely oxidative-stress-sensitive. "
                    "TXNRD2 clue: + dilated cardiomyopathy (cardiac mitochondria also vulnerable). "
                    "NNT MODEL: spontaneous NNT deletion in C57BL/6J mice → confirmed role in adrenal function."
                ),
            },
            {
                "term": "Adrenal Crisis — Emergency Management",
                "definition": (
                    "LIFE-THREATENING: salt-wasting shock + hypoglycaemia + hyperpigmentation (if PAI). "
                    "IMMEDIATE TREATMENT (do not delay for confirmatory tests): "
                    "  1. IV 0.9% saline bolus (20 ml/kg) for hypotension; "
                    "  2. IV hydrocortisone 100 mg (adult) / 50 mg (child) bolus → infusion; "
                    "  3. IV glucose (10% dextrose) for hypoglycaemia; "
                    "  4. Monitor K⁺ — hyperkalaemia (mineralocorticoid deficiency) may need treatment. "
                    "PREVENTION: "
                    "  Sick-day rules: 2-3× hydrocortisone during fever/vomiting; "
                    "  Injectable IM hydrocortisone kit: train patient + family + school; "
                    "  Medical alert ID (bracelet/card). "
                    "TRIGGERS: gastroenteritis, surgery, trauma, fever — NEVER OMIT steroids."
                ),
            },
        ]
    }
