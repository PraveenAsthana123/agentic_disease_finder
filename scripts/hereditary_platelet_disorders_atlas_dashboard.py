"""Hereditary Platelet Disorders Atlas — 8-Gene Reference
ITGA2B-ITGB3-GP1BA-GP9-MYH9-ANKRD26-RUNX1-GFI1B
Glanzmann Thrombasthenia / Bernard-Soulier / MYH9-RD / FPD-AML / GFI1B-Thrombocytopenia
320 patients (8 x 40), seeds 2830-2837.
Endpoints: /api/hereditary-platelet-disorders-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "ITGA2B",
        "protein": (
            "ITGA2B -- 17q21.31 AR -- 1039aa -- Integrin-Alpha-IIb-114kDa-"
            "Platelet-Fibrinogen-Receptor-GPIIb-Non-Covalent-Heterodimer-With-ITGB3-"
            "OMIM-Gene-607759-Disease-Glanzmann-Thrombasthenia-GT1-273800"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1039 aa / 114 kDa (integrin alpha-IIb chain; non-covalently complexes with ITGB3/beta-3 "
            "to form alphaIIbbeta3 / GPIIb-IIIa heterodimer on platelet surface; "
            "~80,000 copies per platelet surface; binds fibrinogen, fibronectin, vitronectin, vWF; "
            "mediates platelet aggregation; biallelic mutations → absent or non-functional alphaIIbbeta3 "
            "→ Glanzmann thrombasthenia; AR; most common GT gene)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function) — Glanzmann Thrombasthenia Type 1 (GT1); "
            "EPIDEMIOLOGY: "
            "  GT1 prevalence ~1:1,000,000 general population; "
            "  Higher frequency: Iraeli Arabs (ITGA2B p.Cys674Stop), French Gypsies (ITGA2B/ITGB3 founder mutations), "
            "  South Asians, Iraqi Jews (ITGB3 p.Ser752Pro); "
            "MECHANISM: "
            "  ITGA2B and ITGB3 must co-assemble in ER → correctly folded heterodimer → Golgi processing → "
            "  platelet surface expression as alphaIIbbeta3; "
            "  ITGA2B LOF → absent or misfolded alphaIIb → ITGB3 also not surface expressed "
            "    (requires ITGA2B for stable surface expression = co-dependence); "
            "CLASSIFICATION (by protein expression): "
            "  Type I GT: <5% surface alphaIIbbeta3 — severe (most ITGA2B mutations); "
            "  Type II GT: 5-25% — moderate bleeding; "
            "  Type III GT (variant): normal quantity but non-functional alphaIIbbeta3; "
            "GENETIC HETEROGENEITY: >200 mutations described; missense, frameshift, splice-site, large deletions; "
            "ALLOIMMUNISATION RISK: "
            "  Multiple platelet transfusions → anti-alphaIIbbeta3 alloantibodies develop in 15-20% → "
            "  refractoriness to platelet transfusion (antibody-mediated platelet destruction); "
            "  CRITICAL MANAGEMENT: minimise platelet transfusions; use only for life-threatening bleeding"
        ),
        "disease_category": (
            "GLANZMANN THROMBASTHENIA TYPE 1 (GT1) — OMIM 273800; "
            "HAEMATOLOGICAL PROFILE: "
            "  Platelet count: NORMAL (150-400 × 10⁹/L) — CRITICAL DDx from thrombocytopenias; "
            "  Platelet morphology: NORMAL size (no giant platelets) — DDx from MYH9/BSS; "
            "  Bleeding time / PFA-100 / PFA-200: markedly prolonged; "
            "  PT / APTT: NORMAL; "
            "  PLATELET AGGREGATION (light transmission aggregometry — LTA): "
            "    ABSENT to ADP, collagen, arachidonic acid, epinephrine, PAR-1 thrombin, PAR-4 — ALL agonists; "
            "    NORMAL (preserved) to ristocetin (ristocetin uses GPIb-vWF axis, NOT alphaIIbbeta3); "
            "    ABSENT ristocetin + ABSENT to other agonists = BSS; "
            "    NORMAL ristocetin + ABSENT to all others = GLANZMANN (ITGA2B/ITGB3); "
            "CLINICAL PRESENTATION: "
            "  Onset: usually in childhood (umbilical bleeding, gingival bleeding, heavy menorrhagia); "
            "  Mucocutaneous bleeding: "
            "    Epistaxis (most common, recurrent); "
            "    Gingival bleeding (tooth eruption, brushing); "
            "    Menorrhagia (leading cause of iron-deficiency anaemia in GT females); "
            "    GI bleeding; haematuria; post-circumcision/surgery bleeding; "
            "  Spontaneous haemarthrosis / intracranial haemorrhage: rare but can occur; "
            "  Variable severity: same genotype → variable phenotype; "
            "  Life-threatening bleeding: 1-2% per decade; "
            "  Female carriers: no bleeding (haploinsufficiency tolerated)"
        ),
        "disease_pathway": (
            "ALPHAIIBBETA3 FIBRINOGEN RECEPTOR / PLATELET AGGREGATION PATHWAY: "
            "NORMAL PLATELET ACTIVATION SIGNALLING: "
            "  Agonist (ADP, collagen, thrombin) binds platelet receptor → inside-out signalling; "
            "  Inside-out: talin-1 + kindlin-3 bind alphaIIbbeta3 cytoplasmic tail → "
            "    → conformational change (extension + head piece opening) → high-affinity fibrinogen binding; "
            "  Fibrinogen cross-links adjacent platelets (bridges alphaIIbbeta3 on two platelets) → "
            "  → platelet aggregation → thrombus formation; "
            "  Outside-in signalling: fibrinogen-bound alphaIIbbeta3 → Src/Syk kinase activation → "
            "    → cytoskeletal remodelling → platelet spreading + clot retraction; "
            "GLANZMANN (ITGA2B LOF): "
            "  No functional alphaIIbbeta3 → no fibrinogen bridge → "
            "  → NO platelet aggregation to any agonist; "
            "  Platelet activation/secretion intact (dense granules release ADP, alpha granules release PF4/vWF); "
            "  Clot retraction: markedly impaired or absent (requires alphaIIbbeta3-outside-in); "
            "  BM megakaryocytes: NORMAL (platelet production unaffected); "
            "SECONDARY HAEMOSTASIS INTACT: "
            "  Thrombin generation → fibrin clot forms normally; "
            "  Clot retraction poor (alphaIIbbeta3 required for clot retraction) → loose friable clot; "
            "VWF RISTOCETIN AXIS PRESERVED: "
            "  GPIb-IX-V complex intact → ristocetin-induced platelet agglutination preserved; "
            "  Explains NORMAL ristocetin LTA in GT (diagnostic distinguisher from BSS)"
        ),
        "pathognomonic": (
            "PLATELET COUNT NORMAL + ABSENT AGGREGATION TO ALL AGONISTS EXCEPT RISTOCETIN — "
            "PATHOGNOMONIC FOR GLANZMANN THROMBASTHENIA: "
            "  Normal platelet count: differentiates GT from thrombocytopenias (ITP, MYH9, BSS, ANKRD26, RUNX1, GFI1B); "
            "  Absent aggregation to ALL agonists (ADP, collagen, AA, epinephrine, thrombin, PAR1, PAR4); "
            "  Normal ristocetin agglutination: GPIb-vWF axis intact → distinguishes from BSS (absent ristocetin in BSS); "
            "  This pattern is unique to alphaIIbbeta3 deficiency (ITGA2B or ITGB3); "
            "NORMAL PLATELET MORPHOLOGY + SIZE: "
            "  No giant platelets (unlike MYH9, BSS); no Döhle-body inclusions; "
            "  MPV: NORMAL (unlike MYH9-RD with high MPV); "
            "ABSENT CLOT RETRACTION: "
            "  Whole blood clotting test: clot forms but fails to retract — historical diagnostic test; "
            "  Modern: flow cytometry for surface alphaIIbbeta3 expression; "
            "FAMILY HISTORY OF CONSANGUINITY with NORMAL COUNT BLEEDING DISORDER: "
            "  AR mucocutaneous bleeding disorder, normal platelet count, normal CBC → GT first; "
            "ALLOIMMUNISATION HISTORY: "
            "  GT patient with declining platelet transfusion response → anti-alphaIIbbeta3 antibodies → "
            "    recombinant FVIIa (rFVIIa) or antifibrinolytic therapy instead of platelets"
        ),
        "treatment": (
            "PLATELET TRANSFUSION — FOR MAJOR BLEEDING/SURGERY: "
            "  HLA-matched / single donor / leukodepleted platelets preferred; "
            "  RISK: alloimmunisation to alphaIIbbeta3 (HPA antigens) in 15-20% → refractoriness; "
            "  MINIMISE: avoid prophylactic transfusions; use only for active major bleeding or surgery; "
            "  RULE: no more than 3-4 lifetime transfusions if possible before HSCT or rFVIIa era; "
            "RECOMBINANT FACTOR VIIA (rFVIIa, NovoSeven): "
            "  MECHANISM: tissue-factor-independent thrombin burst → fibrin clot despite absent alphaIIbbeta3; "
            "  INDICATION: alloimmunised GT patients refractory to platelets; also first-line some centres; "
            "  DOSE: 90-120 mcg/kg IV every 2h until haemostasis; "
            "  Approved for GT in EU; used off-label in other jurisdictions; "
            "ANTIFIBRINOLYTICS (tranexamic acid, epsilon-aminocaproic acid): "
            "  FIRST-LINE for mucosal bleeding (epistaxis, gingival, menorrhagia); "
            "  Tranexamic acid 25 mg/kg TDS PO; topical for epistaxis/dental; "
            "  SAFE and effective adjunct — use BEFORE platelet transfusion for minor bleeding; "
            "DESMOPRESSIN (DDAVP): "
            "  LIMITED ROLE in GT (no vWF/GP1b defect → mild benefit only); "
            "  May help some Type III GT (variant form); generally NOT first-line for GT; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): "
            "  CURATIVE — cures alphaIIbbeta3 deficiency; "
            "  INDICATIONS: severe phenotype, recurrent life-threatening bleeding, alloimmunisation, "
            "    intracranial haemorrhage, poor quality of life; "
            "  Outcomes: OS >90% with matched sibling donor; "
            "MENORRHAGIA MANAGEMENT: "
            "  Combined oral contraceptive pill (COC) — highly effective for menorrhagia in GT; "
            "  Levonorgestrel-releasing IUD; tranexamic acid at menses; rFVIIa for breakthrough; "
            "GENE THERAPY: "
            "  Preclinical and early phase trials; lentiviral ITGA2B delivery to HSCs"
        ),
        "seed": 2830,
        "pt_vars": {
            "platelet_count": (150, 380),
            "bleeding_score": (3, 14),
            "transfusions_lifetime": (0, 8),
            "alloimmunised_pct": 18,
            "hsct_pct": 12,
        }
    },
    {
        "gene": "ITGB3",
        "protein": (
            "ITGB3 -- 17q21.32 AR -- 788aa -- Integrin-Beta-3-87kDa-"
            "GPIIIa-Platelet-Endothelial-Cell-Adhesion-Molecule-"
            "AlphaIIbBeta3-Vitronectin-Receptor-AlphaVBeta3-"
            "OMIM-Gene-173470-Disease-Glanzmann-GT2-273800"
        ),
        "locus": "17q21.32",
        "protein_size": (
            "788 aa / 87 kDa (integrin beta-3 chain = GPIIIa; "
            "heterodimerises with ITGA2B → alphaIIbbeta3 on platelets; "
            "also heterodimerises with alphaV → alphaVbeta3 on endothelial/osteoclasts; "
            "ITGB3 mutations primarily affect platelet alphaIIbbeta3 = GT2; "
            "South Asian/Ashkenazi founder: p.Ser752Pro reduces outside-in signalling; "
            "ITGB3 gene at 17q21.32 — same chromosome region as ITGA2B at 17q21.31; "
            "biallelic AR — carrier parents normal; alloimmunisation generates anti-HPA-1a/3b)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function) — Glanzmann Thrombasthenia Type 2 (GT2); "
            "MECHANISM SAME AS ITGA2B: "
            "  ITGB3 mutations → absent or non-functional beta-3 → "
            "  alphaIIbbeta3 complex fails assembly in ER → no surface expression; "
            "  ITGA2B also not expressed without ITGB3 partner (co-degradation in ER); "
            "DISTINCTION FROM GT1 (ITGA2B): "
            "  Clinically identical to GT1 (same disease, different gene); "
            "  ITGB3 mutations slightly less common than ITGA2B overall but higher frequency in South Asians; "
            "  ALLOIMMUNISATION: anti-HPA-1a (PlA1) — most common platelet-specific alloantibody; "
            "    HPA-1a epitope on ITGB3 (Leu33) — anti-HPA-1a in GT2 with Arg33 allele → neonatal alloimmune thrombocytopenia (NAIT) risk in carrier mothers; "
            "NEONATAL ALLOIMMUNE THROMBOCYTOPENIA (NAIT) CONNECTION: "
            "  Heterozygous ITGB3-Arg33 (HPA-1b) mothers with HPA-1a foetus → anti-HPA-1a → NAIT; "
            "  GT2 homozygous Arg33 female → anti-HPA-1a after first platelet transfusion → refractoriness; "
            "  Most important HPA antigen system clinically; "
            "KNOWN FOUNDER MUTATIONS: "
            "  p.Ser752Pro (South Asian: India, Pakistan) — reduced outside-in signalling variant; "
            "  c.1544+1G>A Iraqi/Ashkenazi; p.Tyr490Stop North African; "
            "AML/MDS RISK: NOT ELEVATED (unlike ANKRD26 / RUNX1)"
        ),
        "disease_category": (
            "GLANZMANN THROMBASTHENIA TYPE 2 (GT2) — OMIM 273800; "
            "CLINICALLY IDENTICAL TO GT1 except for HPA system: "
            "HAEMATOLOGICAL PROFILE: "
            "  Platelet count: NORMAL (150-400 × 10⁹/L); "
            "  Platelet morphology: NORMAL (no giant platelets, no inclusions); "
            "  Platelet aggregation LTA: "
            "    ABSENT to ADP, collagen, AA, epinephrine, thrombin, PAR1/PAR4; "
            "    PRESERVED (normal) to ristocetin; "
            "  alphaIIbbeta3 surface expression: <5% (Type I GT) or 5-25% (Type II GT); "
            "CLINICAL PRESENTATION: "
            "  Identical to GT1: mucocutaneous bleeding from infancy; "
            "  Epistaxis, gingival bleeding, menorrhagia, GI bleeding; "
            "  Perioperative and post-traumatic bleeding; "
            "SPECIAL CONCERN — HPA-1a ALLOIMMUNISATION: "
            "  GT2 patients with HPA-1b/1b genotype (homozygous for Arg33): "
            "  → Platelet transfusion → anti-HPA-1a antibodies in 20-30% → refractoriness; "
            "  → rFVIIa or antifibrinolytics become primary management; "
            "FEMALE GT2 PATIENTS: "
            "  Pregnancy: foetal HPA-1a (from HPA-1a father) → maternal alloimmunisation; "
            "  Foetal/neonatal thrombocytopenia possible; "
            "  Obstetric team must be aware; peripartum planning essential"
        ),
        "disease_pathway": (
            "ALPHAIIBBETA3 FIBRINOGEN RECEPTOR (BETA-3 CHAIN DEFECT) — SAME PATHWAY AS ITGA2B: "
            "NORMAL ITGB3 FUNCTION: "
            "  ITGB3 synthesised in megakaryocyte ER → "
            "    → co-translational folding with ITGA2B → "
            "    → disulphide bond formation (beta-3 has 8 cysteine-rich EGF-like domains); "
            "    → correct folding required for ITGA2B-ITGB3 heterodimer assembly; "
            "  alphaIIbbeta3 on platelet: "
            "    headpiece = ligand binding (fibrinogen, vWF, fibronectin); "
            "    transmembrane domains: inside-out activation site; "
            "    cytoplasmic tails: bind talin-1/kindlin-3 (inside-out); Src/Syk (outside-in); "
            "ITGB3 LOF: "
            "  Beta-3 absent/misfolded → no alphaIIbbeta3 complex → "
            "    → no fibrinogen-receptor → NO platelet aggregation; "
            "  Outside-in signalling absent → no clot retraction, no platelet spreading; "
            "ALPHAIIBBETA3 IN OTHER CONTEXTS: "
            "  alphaVbeta3 (vitronectin receptor on endothelial cells, osteoclasts) — also uses ITGB3; "
            "  GT2 mutations mostly selective for platelet alphaIIbbeta3 (less severe on endothelial alphaVbeta3); "
            "  Some ITGB3 variants affect both complexes → multiorgan bleeding + bone density changes; "
            "HPA-1 EPITOPE BIOLOGY: "
            "  Pro33 (HPA-1a, common) vs Arg33 (HPA-1b, rare) = Leu33Pro polymorphism in ITGB3; "
            "  GT2 with homozygous Arg33/Arg33 → HPA-1a epitope absent → vulnerable to anti-HPA-1a"
        ),
        "pathognomonic": (
            "SAME CLINICAL PATTERN AS GT1 — NORMAL COUNT + ABSENT AGGREGATION ALL AGONISTS EXCEPT RISTOCETIN: "
            "  Differentiates GT from BSS (giant platelets, absent ristocetin in BSS); "
            "  Differentiates GT from thrombocytopenias (NORMAL platelet count in GT); "
            "  Gene panel required to distinguish ITGA2B (GT1) from ITGB3 (GT2) — clinically identical; "
            "HPA-1b/1b GENOTYPE WITH GT DIAGNOSIS: "
            "  GT2 patient who is HPA-1b homozygous → highest alloimmunisation risk; "
            "  Anti-HPA-1a develops → refractoriness to random donor platelets; "
            "  Management shift: rFVIIa + antifibrinolytics first-line; HPA-1b platelets (rare donors); "
            "SOUTH ASIAN ANCESTRY + GT: "
            "  p.Ser752Pro ITGB3 variant common in South Asians; "
            "  Molecular diagnosis critical — affects HPA genotyping and transfusion planning; "
            "LOW FLOW CYTOMETRY alphaIIbbeta3 (<5%): "
            "  Platelet surface alphaIIbbeta3 by flow cytometry (anti-CD41/CD61): <5% = Type I GT; "
            "  Distinguishes GT (absent) from VWD type 1 or platelet-type VWD (normal); "
            "ABSENT CLOT RETRACTION IN WHOLE BLOOD CLOTTING TEST: "
            "  Formed clot fails to retract → absence of outside-in signalling; "
            "  Classic historical finding; replaced by flow cytometry in modern labs"
        ),
        "treatment": (
            "SAME AS GT1/ITGA2B — IDENTICAL MANAGEMENT APPROACH: "
            "ANTIFIBRINOLYTICS (FIRST-LINE MUCOSAL BLEEDING): "
            "  Tranexamic acid 25 mg/kg TDS; topical for epistaxis; before dental procedures; "
            "PLATELET TRANSFUSION (MAJOR BLEEDING/SURGERY): "
            "  HLA/HPA-matched when possible; single donor; leukodepleted; "
            "  RISK: anti-HPA-1a alloimmunisation (especially HPA-1b/1b patients); "
            "  MINIMISE transfusions to preserve future options; "
            "RECOMBINANT FACTOR VIIA (rFVIIa): "
            "  First-line for alloimmunised patients or those requiring surgery; "
            "  90-120 mcg/kg IV every 2h; highly effective; "
            "DESMOPRESSIN (DDAVP): minimal role in GT2 (absent alphaIIbbeta3 → DDAVP cannot help); "
            "HSCT: "
            "  Curative for severe/refractory GT2; "
            "  Indications: life-threatening bleeding, alloimmunisation refractory to all other treatment; "
            "  Matched sibling donor preferred; OS >90%; "
            "OBSTETRIC MANAGEMENT: "
            "  Pre-conception counselling for GT2 females; "
            "  Foetal HPA genotyping; "
            "  Peripartum rFVIIa plan; neonatology team for potential NAIT in infant; "
            "ORAL HEALTH: "
            "  Chlorhexidine mouthwash; local antifibrinolytic before dental procedures; "
            "  Tranexamic acid-soaked gauze for post-extraction haemostasis"
        ),
        "seed": 2831,
        "pt_vars": {
            "platelet_count": (150, 380),
            "bleeding_score": (3, 13),
            "transfusions_lifetime": (0, 7),
            "alloimmunised_pct": 20,
            "hsct_pct": 10,
        }
    },
    {
        "gene": "GP1BA",
        "protein": (
            "GP1BA -- 17p13.2 AR/AD -- 626aa -- Glycoprotein-Ib-Alpha-75kDa-"
            "Leucine-Rich-Repeat-Platelet-VWF-Binding-Subunit-"
            "GPIb-IX-V-Complex-BSS-Bernard-Soulier-Syndrome-"
            "OMIM-Gene-606672-Disease-BSS-A-231200"
        ),
        "locus": "17p13.2",
        "protein_size": (
            "626 aa / 75 kDa (GPIbα chain; leucine-rich repeat domain binds vWF at high shear; "
            "cytoplasmic tail anchors to cytoskeleton via filamin-1; "
            "forms GPIb-IX-V complex with GP1BB (GPIbβ), GP9 (GPIX), GP5 (GPV); "
            "~25,000 complexes per platelet; "
            "biallelic AR mutations → classic BSS type A (most common); "
            "AD heterozygous mutations → rare Bolzano-type or macrothrombocytopenia without full BSS)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic) — classic BSS Type A (most common BSS); "
            "AUTOSOMAL DOMINANT (some heterozygous gain-of-partial-function / expression variants) — rare; "
            "EPIDEMIOLOGY: "
            "  BSS prevalence ~1:1,000,000; increased in consanguineous populations; "
            "  Mediterranean, Middle Eastern, Japanese populations; "
            "MECHANISM: "
            "  GP1BA encodes the primary vWF-binding subunit of GPIb-IX-V complex; "
            "  GP1BA LOF → no GPIb-IX-V surface complex (GP9 and GP1BB also fail to express "
            "    without GP1BA — co-dependent surface expression); "
            "  Absent GPIb-IX-V → no platelet adhesion to injured vessel wall (primary haemostasis fails); "
            "  No vWF-platelet bridge at high shear → bleeding even with normal platelet count; "
            "GIANT PLATELET MECHANISM: "
            "  GPIb-IX-V is critical for proplatelet formation by megakaryocytes; "
            "  Absent GPIb-IX-V → abnormal megakaryocyte maturation → "
            "    irregular platelet budding → release of giant platelets (macro-platelets); "
            "  Some platelets as large as lymphocytes (15-20 μm) — diagnostic on blood film; "
            "THROMBOCYTOPENIA: "
            "  GPIb-IX-V also regulates platelet lifespan; absent complex → reduced survival; "
            "  Platelet count typically 20-100 × 10⁹/L (thrombocytopenia + GIANT platelets); "
            "PLATELET-TYPE VWD: different disease — GOF GP1BA mutation → enhanced spontaneous vWF binding → "
            "  depletes plasma vWF → low vWF mimicking VWD type 2B; distinguish by molecular testing"
        ),
        "disease_category": (
            "BERNARD-SOULIER SYNDROME TYPE A (BSS-A) — OMIM 231200; "
            "CARDINAL TRIAD — PATHOGNOMONIC: "
            "  1. GIANT PLATELETS: "
            "     MPV typically >12 fL (normal 7-11 fL); some platelets lymphocyte-sized on blood film; "
            "     May be misread as lymphocytes → thrombocytopenia falsely underdiagnosed; "
            "     PATHOGNOMONIC FEATURE — differentiates BSS from all other hereditary thrombocytopenias except MYH9; "
            "  2. THROMBOCYTOPENIA: "
            "     Platelet count 20-100 × 10⁹/L; moderate-severe; "
            "     ITP mimicry — frequently misdiagnosed as ITP → inappropriate IVIG/steroids; "
            "  3. PROLONGED BLEEDING TIME / PFA: "
            "     Out of proportion to platelet count (GPIb-vWF adhesion absent); "
            "LABORATORY: "
            "  LTA: "
            "    ABSENT to ristocetin (ristocetin requires GPIbα-vWF interaction — absent in BSS); "
            "    PRESERVED to ADP, collagen, arachidonic acid, epinephrine (alphaIIbbeta3 intact); "
            "  CRITICAL DDx: "
            "    BSS: absent ristocetin + present other agonists + giant platelets + thrombocytopenia; "
            "    GT: present ristocetin + absent other agonists + NORMAL count + normal platelet size; "
            "  GPIb flow cytometry (CD42b): markedly reduced / absent in BSS; "
            "CLINICAL PRESENTATION: "
            "  Onset: infancy / childhood; "
            "  Mucocutaneous bleeding (same organs as GT but with thrombocytopenia component); "
            "  Epistaxis, gingival bleeding, bruising, menorrhagia; "
            "  GI bleeding; post-surgical bleeding; "
            "  Intracranial haemorrhage: rare but described"
        ),
        "disease_pathway": (
            "GPIb-IX-V COMPLEX / VWF-PLATELET ADHESION PATHWAY: "
            "NORMAL GPIb-IX-V FUNCTION: "
            "  GPIb-IX-V complex = heterodimer of GP1BA (GPIbα) + GP1BB (GPIbβ) + GP9 (GPIX) + GP5 (GPV); "
            "  GPIbα leucine-rich repeat domain: "
            "    binds vWF-A1 domain at HIGH SHEAR (arteries, arterioles); "
            "    binds thrombin (PAR-independent low-level thrombin activation); "
            "  Filamin-A anchors GPIbα cytoplasmic tail to actin cytoskeleton; "
            "  14-3-3ζ + PI3K signalling via cytoplasmic tail → platelet activation; "
            "VWF BRIDGE AT HIGH SHEAR: "
            "  Endothelial injury → vWF released (ultra-large VWF multimers); "
            "  High shear unfolds vWF → A1 domain exposed → binds GPIbα → "
            "    → platelet tethering and rolling → activation → platelet plug; "
            "  This is PRIMARY HAEMOSTASIS step 1 (platelet adhesion to subendothelium); "
            "ABSENT GPIb-IX-V (BSS): "
            "  No vWF tethering at high shear → "
            "    → no platelet adhesion to injured vessel wall → "
            "    → delayed or absent platelet plug → prolonged bleeding; "
            "GIANT PLATELET MECHANISM: "
            "  Normal GPIb-IX-V signals for proplatelet formation in megakaryocytes; "
            "  Absent → proplatelet fragmentation defect → larger, irregular platelet release; "
            "  Large platelets also more reactive (more metabolically active); "
            "ALLOIMMUNISATION IN BSS: "
            "  Anti-GPIb (anti-CD42) alloantibodies develop after platelet transfusion; "
            "  HPA-2 (KoBa antigen on GPIbα, Thr145Met) — alloimmunisation target; "
            "  Less common than anti-HPA-1a (GT2) but relevant in BSS transfusion management"
        ),
        "pathognomonic": (
            "GIANT PLATELETS + THROMBOCYTOPENIA + ABSENT RISTOCETIN AGGREGATION — "
            "PATHOGNOMONIC TRIAD FOR BERNARD-SOULIER SYNDROME: "
            "  Giant platelets on blood film (lymphocyte-sized): automated analysers often miscount → "
            "    manual review of film MANDATORY in any patient with apparent mild thrombocytopenia + big MPV; "
            "  Absent ristocetin: UNIQUE to BSS among common platelet disorders "
            "    (GT has NORMAL ristocetin; ITP has NORMAL aggregation); "
            "  Preserved ADP/collagen aggregation: confirms alphaIIbbeta3 intact (vs GT where absent); "
            "FREQUENT ITP MISDIAGNOSIS: "
            "  Giant platelets counted as lymphocytes → thrombocytopenia appears less severe → "
            "  → IVIG/steroid treated as ITP → no response → BSS should be considered; "
            "  RULE: any 'ITP' with giant platelets + family history + no steroid/IVIG response → molecular panel; "
            "LOW CD42b FLOW CYTOMETRY: "
            "  <5% GPIbα surface expression → BSS confirmed; "
            "  Normal GPIbα flow = excludes classic BSS (point to platelet-type VWD or VWD2B); "
            "ABSENT PFA-100 CLOSURE TIME EVEN WITH RELATIVELY MAINTAINED COUNT: "
            "  PFA-100 collagen-ADP cartridge closure time: very prolonged (>300s); "
            "  Reflects GPIb-vWF adhesion defect; "
            "NEONATAL ALLOIMMUNE THROMBOCYTOPENIA (NAIT) DUE TO ANTI-GPIbα: "
            "  BSS mother (absent GPIbα) → anti-GPIbα after transfusion/pregnancy → NAIT in next pregnancy"
        ),
        "treatment": (
            "PLATELET TRANSFUSION — MAINSTAY FOR MAJOR BLEEDING/SURGERY: "
            "  HLA-matched and/or GPIb-matched donors preferred; "
            "  ALLOIMMUNISATION RISK: anti-GPIb (anti-CD42) after multiple transfusions; "
            "  MINIMISE transfusions; use only for active major bleeding or peri-operative cover; "
            "ANTIFIBRINOLYTICS (FIRST-LINE MUCOSAL BLEEDING): "
            "  Tranexamic acid 25 mg/kg TDS; topical for epistaxis, gingival bleeding; "
            "  Before dental/minor procedures; highly effective adjunct; "
            "DESMOPRESSIN (DDAVP): "
            "  SOME BENEFIT in mild BSS: temporarily increases plasma vWF → helps residual GPIb-vWF axis; "
            "  0.3 mcg/kg IV/SC; tachyphylaxis after 2-3 doses; "
            "  MORE EFFECTIVE IN BSS THAN IN GT (because vWF increase can partially compensate); "
            "RECOMBINANT FACTOR VIIA (rFVIIa): "
            "  For alloimmunised patients refractory to platelet transfusion; "
            "  90-120 mcg/kg IV every 2-3h; "
            "HSCT: "
            "  CURATIVE for severe BSS; corrects giant platelet defect + bleeding phenotype; "
            "  Less commonly needed than in GT but indicated for severe/refractory cases; "
            "MENORRHAGIA: "
            "  COC pill, LNG-IUD, tranexamic acid; rFVIIa for breakthrough; "
            "AVOID: "
            "  NSAIDs / aspirin / P2Y12 inhibitors — platelet function further impaired; "
            "  Iron supplementation for anaemia from chronic mucosal blood loss"
        ),
        "seed": 2832,
        "pt_vars": {
            "platelet_count": (20, 100),
            "bleeding_score": (4, 16),
            "transfusions_lifetime": (0, 10),
            "alloimmunised_pct": 22,
            "hsct_pct": 8,
        }
    },
    {
        "gene": "GP9",
        "protein": (
            "GP9 -- 3q21.3 AR -- 160aa -- Glycoprotein-IX-17kDa-"
            "GPIb-IX-V-Complex-GPIX-Subunit-Leucine-Rich-Repeat-"
            "Bernard-Soulier-Syndrome-Type-C-"
            "OMIM-Gene-173515-Disease-BSS-C-231200"
        ),
        "locus": "3q21.3",
        "protein_size": (
            "160 aa / 17 kDa (glycoprotein IX = GPIX; smallest subunit of GPIb-IX-V complex; "
            "single transmembrane leucine-rich repeat protein; "
            "GPIX stabilises GP1BA surface expression — without GPIX, GPIbα-GPIbβ complex poorly trafficked to surface; "
            "GPIX is the limiting component for GPIb complex surface expression; "
            "BSS type C — clinically indistinguishable from GP1BA/GP1BB BSS; "
            "AR biallelic; less common than GP1BA mutations)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function) — Bernard-Soulier Syndrome Type C (BSS-C); "
            "EPIDEMIOLOGY: "
            "  Less common than BSS-A (GP1BA) and BSS-B (GP1BB); "
            "  Consanguineous families enriched; founder mutations in some populations; "
            "MECHANISM: "
            "  GPIX is required for efficient surface trafficking of the GPIb-IX-V complex; "
            "  Without GPIX: GPIbα (GP1BA) and GPIbβ (GP1BB) are synthesised but poorly trafficked to platelet surface; "
            "  GPIX LOF → severe reduction in surface GPIb-IX-V → same consequence as GP1BA LOF; "
            "  Result: absent GPIb-vWF interaction at high shear → giant platelets + thrombocytopenia + absent ristocetin; "
            "MILDER VARIANT FORMS: "
            "  Some GPIX missense mutations → partial expression → variable phenotype (BSS 'Type C variant'); "
            "  Incomplete thrombocytopenia with near-normal giant platelet count occasionally; "
            "  p.Asn45Ser: partial function; reduced GP complex expression; milder phenotype; "
            "PSEUDO-VWD RISK: "
            "  Absent GPIb-IX-V → vWF not cleared → mild elevation in plasma vWF multimers sometimes; "
            "  Does NOT cause the spontaneous binding of VWD type 2B or platelet-type VWD; "
            "GENETIC COUNSELLING: "
            "  AR biallelic; sibling risk 25%; carrier testing of parents; "
            "  Compound heterozygous common in non-consanguineous families; "
            "CLINICAL SEVERITY: "
            "  Generally same as BSS-A (GP1BA); variable by mutation type; "
            "  GPIb complex is a single functional unit — losing any subunit equally disabling"
        ),
        "disease_category": (
            "BERNARD-SOULIER SYNDROME TYPE C (BSS-C) — OMIM 231200; "
            "CLINICALLY IDENTICAL TO BSS-A (GP1BA) AND BSS-B (GP1BB): "
            "LABORATORY PROFILE: "
            "  Platelet count: 20-100 × 10⁹/L (moderate-severe thrombocytopenia); "
            "  Giant platelets: PATHOGNOMONIC — MPV >12 fL; "
            "  LTA: absent to ristocetin; preserved to ADP/collagen/AA; "
            "  CD42b (GPIbα) flow cytometry: markedly reduced; "
            "CLINICAL PRESENTATION: "
            "  Same as BSS-A: mucocutaneous bleeding from infancy; "
            "  Epistaxis, gingival bleeding, menorrhagia, bruising; "
            "  Perioperative bleeding risk; "
            "DIAGNOSIS: "
            "  Gene panel for BSS: GP1BA, GP1BB, GP9 — all three sequenced; "
            "  GPIX-specific flow cytometry (anti-CD42a) may be reduced in BSS-C; "
            "  Gene panel required to distinguish BSS-A/B/C — clinically identical; "
            "DISTINGUISHING BSS FROM PLATELET-TYPE VWD: "
            "  Platelet-type VWD (GP1BA-GOF): spontaneous vWF binding → reduced plasma vWF; "
            "    GP1BA GOF → enhanced affinity for vWF; "
            "  BSS (all types): absent vWF binding; low surface GPIb; "
            "  Test: platelet cryoprecipitate mixing study; "
            "    BSS: no response to platelet-poor plasma; "
            "    Platelet-type VWD: adds cryo normalises aggregation"
        ),
        "disease_pathway": (
            "GPIX / GPIb-IX-V COMPLEX STABILISATION AND TRAFFICKING PATHWAY: "
            "GPIX MOLECULAR FUNCTION: "
            "  GPIX single LRR domain → contacts GP1BB (GPIbβ) in the ER; "
            "  GPIX-GP1BB interaction stabilises the dimer; "
            "  GP1BA (GPIbα) heterodimerises with GP1BB-GPIX → trimeric core; "
            "  GP5 (GPV) loosely associates (non-covalently) on platelet surface; "
            "  GPIX = rate-limiting subunit for GPIb complex surface expression; "
            "  Without GPIX: GP1BA-GP1BB dimer formed but INEFFICIENTLY TRAFFICKED → ER retention → degradation; "
            "CONSEQUENCE OF GPIX LOF: "
            "  Reduced surface GPIb-IX-V → no vWF high-shear binding → no platelet adhesion; "
            "  Same functional deficit as GP1BA LOF; "
            "  Giant platelet mechanism same (proplatelet formation requires GPIb complex signalling); "
            "RISTOCETIN MECHANISM: "
            "  Ristocetin is a polycationic antibiotic that FORCES vWF-GPIbα interaction "
            "    even in resting platelets; "
            "  Absent GPIb-IX-V (any BSS type) → no ristocetin-induced agglutination; "
            "  CRITICAL DIAGNOSTIC: tests GPIb-vWF axis specifically; "
            "GPIX p.Asn45Ser VARIANT: "
            "  Partial loss of GPIX folding → ~40-60% surface GPIb-IX-V; "
            "  Milder thrombocytopenia; less severe bleeding; "
            "  May be missed as 'variant of uncertain significance' in sequencing — functional assays needed"
        ),
        "pathognomonic": (
            "SAME TRIAD AS ALL BSS — GIANT PLATELETS + THROMBOCYTOPENIA + ABSENT RISTOCETIN: "
            "  Clinically identical to BSS-A (GP1BA) and BSS-B (GP1BB); "
            "  Gene panel with GP1BA, GP1BB, GP9 sequencing required to assign BSS type; "
            "  GPIX flow cytometry (anti-CD42a antibody) reduced in BSS-C; "
            "  GP1BA flow (anti-CD42b) also reduced (co-expression dependence); "
            "LOW/ABSENT CD42a (ANTI-GPIX) FLOW CYTOMETRY: "
            "  CD42a absent → BSS-C confirmed; "
            "  Both CD42a AND CD42b absent → GPIb-IX-V complex absent → BSS any type; "
            "GENE PANEL DIAGNOSIS (MANDATORY for BSS subtyping): "
            "  GP9 mutation + absent GPIX flow + absent ristocetin LTA + giant platelets = BSS-C; "
            "  Treatment same regardless of subtype; "
            "  Subtyping matters for: genetic counselling, prenatal diagnosis, registry, "
            "    and future gene therapy targeting (gene-specific); "
            "BSS vs MYH9 DISTINCTION: "
            "  Both: giant platelets + thrombocytopenia; "
            "  MYH9: NORMAL ristocetin; Döhle-body-like inclusions in neutrophils; AUTOSOMAL DOMINANT; "
            "  BSS: ABSENT ristocetin; no inclusions; AUTOSOMAL RECESSIVE; "
            "  Deafness/nephritis/cataracts absent in BSS"
        ),
        "treatment": (
            "SAME MANAGEMENT AS BSS-A (GP1BA) — ALL BSS TYPES TREATED IDENTICALLY: "
            "ANTIFIBRINOLYTICS (FIRST-LINE): "
            "  Tranexamic acid 25 mg/kg TDS; topical for mucosal bleeds; "
            "DESMOPRESSIN (DDAVP): "
            "  Partial benefit possible (increases plasma vWF); "
            "  0.3 mcg/kg IV/SC; used before minor procedures in mild-moderate BSS; "
            "PLATELET TRANSFUSION: "
            "  Matched donors preferred (minimise anti-GPIb alloimmunisation); "
            "  For major bleeding/surgery when antifibrinolytics insufficient; "
            "rFVIIa: "
            "  For alloimmunised patients; "
            "  Alternative to platelets in surgical cover; "
            "HSCT: "
            "  Curative for severe BSS — corrects all haematological and functional abnormalities; "
            "  Indicated for life-threatening bleeding refractory to other measures; "
            "AVOID NSAIDs/ASPIRIN: "
            "  Any antiplatelet agent worsens platelet dysfunction in BSS; "
            "  Surgical team must be notified of BSS diagnosis before any procedure"
        ),
        "seed": 2833,
        "pt_vars": {
            "platelet_count": (25, 110),
            "bleeding_score": (3, 15),
            "transfusions_lifetime": (0, 8),
            "alloimmunised_pct": 18,
            "hsct_pct": 7,
        }
    },
    {
        "gene": "MYH9",
        "protein": (
            "MYH9 -- 22q13.1 AD -- 1960aa -- Non-Muscle-Myosin-IIA-Heavy-Chain-"
            "227kDa-NMHC-IIA-Cytoskeletal-Motor-Protein-"
            "May-Hegglin-Fechtner-Sebastian-Epstein-Anomalies-"
            "OMIM-Gene-160775-Disease-MYH9-RD-155100"
        ),
        "locus": "22q13.1",
        "protein_size": (
            "1960 aa / 227 kDa (non-muscle myosin heavy chain IIA = NMHC-IIA; "
            "cytoskeletal motor protein; forms bipolar filaments with NMHC-IIA head-tail interactions; "
            "role in cell division, cell migration, exocytosis, proplatelet formation; "
            "expressed in platelets, leukocytes, kidney, cochlea, eye lens; "
            "AD heterozygous mutations → MYH9-related disease (MYH9-RD); "
            "most common hereditary macrothrombocytopenia with extrasystemic features)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (heterozygous haploinsufficiency / dominant-negative) — MYH9-Related Disease; "
            "EPIDEMIOLOGY: "
            "  MYH9-RD = most common hereditary macrothrombocytopenia (estimated ~1:50,000-100,000); "
            "  May-Hegglin anomaly: historically first described (1945); "
            "  Fechtner, Sebastian, Epstein syndromes: all allelic MYH9 mutations — same disease spectrum; "
            "  ALL FOUR eponymous syndromes unified as MYH9-RD (2003 Seri/Kunishima); "
            "MECHANISM: "
            "  NMHC-IIA haploinsufficiency → "
            "    1. Megakaryocyte proplatelet defect → giant, fewer platelets; "
            "    2. Döhle-body inclusions in neutrophils (aggregated NMHC-IIA) — PATHOGNOMONIC; "
            "    3. Extrasystemic: cochlear hair cell dysfunction → SNHL; "
            "       glomerular podocyte dysfunction → nephritis → ESRD; "
            "       lens cells → pre-senile cataracts; "
            "  DOMINANT NEGATIVE in some mutations: mutant NMHC-IIA misassembles → "
            "    aggregates with normal NMHC-IIA → loss of both copies' function; "
            "GENOTYPE-PHENOTYPE CORRELATIONS (MYH9-RD): "
            "  p.Arg702 mutations: highest risk for SNHL, nephritis, cataracts; "
            "  p.Asp1424, Glu1841: milder; lower extrasystemic risk; "
            "  Head domain mutations (motor domain): generally more severe; "
            "  Tail domain mutations (rod): milder; mostly pure haematological; "
            "  CRITICAL: same mutation can have variable expressivity between family members"
        ),
        "disease_category": (
            "MYH9-RELATED DISEASE (MYH9-RD) — OMIM 155100; "
            "UNIFIED EPONYMS: May-Hegglin Anomaly / Fechtner Syndrome / Sebastian Syndrome / Epstein Syndrome; "
            "HAEMATOLOGICAL PROFILE — DIAGNOSTIC TRIAD: "
            "  1. GIANT PLATELETS (MACROTHROMBOCYTOPENIA): "
            "     MPV >12 fL (often 15-25 fL — amongst largest in any platelet disorder); "
            "     Platelet count: 20-100 × 10⁹/L; "
            "     Platelet function: RELATIVELY PRESERVED (aggregation studies: near-normal to most agonists); "
            "  2. DÖHLE-BODY-LIKE NEUTROPHIL INCLUSIONS — PATHOGNOMONIC: "
            "     Light blue cytoplasmic inclusions in neutrophils on Romanowsky-stained blood film; "
            "     Immunofluorescence: NMHC-IIA aggregates; "
            "     PRESENT IN ALL MYH9-RD — diagnostic before molecular testing; "
            "  3. RELATIVELY MILD BLEEDING: "
            "     Mucocutaneous bleeding less severe than BSS/GT despite giant platelets; "
            "     Platelet function partially preserved (NMHC-IIA affects formation, not receptor function); "
            "     PFA-100: mildly prolonged or near-normal; "
            "EXTRASYSTEMIC COMPLICATIONS (variable, genotype-dependent): "
            "  SENSORINEURAL HEARING LOSS (SNHL): 30-70% (Fechtner, Epstein); progressive; cochlea; "
            "  NEPHRITIS: 25-50%; proteinuria → nephrotic syndrome → ESRD (young adult); "
            "  CATARACTS: 20-35% (pre-senile, lens MYH9); "
            "  SCREENING: audiogram + renal function + ophthalmology at diagnosis, then annual"
        ),
        "disease_pathway": (
            "NON-MUSCLE MYOSIN IIA (NMHC-IIA) CYTOSKELETAL MOTOR PATHWAY: "
            "NMHC-IIA STRUCTURE AND FUNCTION: "
            "  Head domain: ATPase motor activity; actin-binding; force generation; "
            "  Neck domain: light chain regulatory binding; "
            "  Rod/tail domain: coiled-coil → bipolar thick filament assembly; "
            "  NMHC-IIA in platelets: "
            "    Required for proplatelet formation from megakaryocytes; "
            "    Contractile function (ATPase) needed for final platelet release from proplatelet tips; "
            "    Without NMHC-IIA: megakaryocytes make large proplatelet buds → giant, irregular platelets; "
            "  NMHC-IIA in neutrophils: "
            "    Normal NMHC-IIA: uniformly distributed in cytoplasm; "
            "    Mutant NMHC-IIA: forms aggregates → Döhle-body-like inclusions visible on film; "
            "    Inclusions = diagnostic; neutrophil function relatively intact; "
            "EXTRASYSTEMIC MECHANISMS: "
            "  COCHLEA: outer hair cell stereocilia require NMHC-IIA for electromotility → "
            "    NMHC-IIA LOF → hair cell dysfunction → progressive SNHL; "
            "  GLOMERULUS: podocyte foot processes require NMHC-IIA for filtration slit integrity → "
            "    NMHC-IIA LOF → podocytopathy → glomerulonephritis → proteinuria → ESRD; "
            "  LENS: lens epithelial cells require NMHC-IIA for fibre cell differentiation → "
            "    NMHC-IIA LOF → premature cataract formation; "
            "DOMINANT NEGATIVE vs HAPLOINSUFFICIENCY: "
            "  Tail domain mutations (rod segment): haploinsufficiency; fewer NMHC-IIA filaments; "
            "  Head domain mutations: dominant negative — mutant binds normal NMHC-IIA → misassembly → "
            "    aggregates (more severe Döhle inclusions) → more organ dysfunction"
        ),
        "pathognomonic": (
            "DÖHLE-BODY-LIKE INCLUSIONS IN NEUTROPHILS — PATHOGNOMONIC FOR MYH9-RD: "
            "  Light blue cytoplasmic inclusions in neutrophils on Romanowsky stain (May-Grünwald-Giemsa); "
            "  Present in ALL MYH9-RD patients; absent in BSS, GT, ITP, ANKRD26, RUNX1; "
            "  Immunofluorescence: anti-NMHC-IIA antibody stains inclusions brightly (vs. cytoplasm); "
            "  BSS DIFFERENTIAL: both have giant platelets + thrombocytopenia; "
            "    MYH9: normal ristocetin, Döhle inclusions, AD; "
            "    BSS: absent ristocetin, no inclusions, AR; "
            "  ITP DIFFERENTIAL: ITP = normal/small platelets, no inclusions, acquired; "
            "GIANT PLATELET + MACROTHROMBOCYTOPENIA + NORMAL RISTOCETIN LTA: "
            "  MYH9 platelet receptors (GPIb, alphaIIbbeta3) INTACT → aggregation to ADP/collagen/AA/ristocetin: near-normal; "
            "  This pattern (thrombocytopenia + giant platelets + normal aggregation) + Döhle inclusions = MYH9; "
            "EARLY-ONSET HEARING LOSS IN YOUNG ADULT WITH THROMBOCYTOPENIA: "
            "  SNHL developing in 20s-30s + macrothrombocytopenia → MYH9-RD; "
            "HAEMATURIA + PROTEINURIA IN YOUNG ADULT WITH MACROTHROMBOCYTOPENIA: "
            "  Glomerulonephritis + macrothrombocytopenia → MYH9 nephritis; "
            "GENOTYPE-PHENOTYPE SCREENING MANDATORY: "
            "  All MYH9-RD patients: audiometry + urine ACR + renal function + ophthalmology at diagnosis; "
            "  Annual review for extrasystemic progression"
        ),
        "treatment": (
            "BLEEDING MANAGEMENT (RELATIVELY MILD IN MOST MYH9-RD): "
            "ANTIFIBRINOLYTICS: "
            "  Tranexamic acid for epistaxis, dental, menorrhagia; "
            "DESMOPRESSIN (DDAVP): "
            "  May be beneficial — increases plasma vWF level, mild platelet activating effect; "
            "  0.3 mcg/kg IV/SC; option before procedures; "
            "PLATELET TRANSFUSION: "
            "  Rarely needed; reserved for major surgery or severe bleeding; "
            "  Platelet function relatively intact → platelet count correction usually sufficient; "
            "THROMBOPOIETIN RECEPTOR AGONISTS (TPO-RA): "
            "  Eltrombopag, romiplostim: EMERGING ROLE in MYH9-RD with severe thrombocytopenia; "
            "  Evidence limited but case reports show platelet count rise; "
            "  Does NOT improve platelet function (NMHC-IIA defect persists); "
            "EXTRASYSTEMIC MONITORING + INTERVENTION: "
            "  SNHL: audiometry annually; hearing aids; cochlear implants if profound SNHL; "
            "  NEPHRITIS: "
            "    ACE inhibitor or ARB when proteinuria develops (proteinuria >0.5g/day); "
            "    BP control; nephrology co-management; "
            "    Renal biopsy if nephrotic/uncertain; "
            "    ESRD: dialysis/renal transplant (transplanted kidney has donor's normal MYH9 — function OK); "
            "  CATARACTS: ophthalmology follow-up; surgical extraction when indicated; "
            "AVOID: "
            "  NSAIDs / aspirin; antiplatelet agents worsen already-elevated bleeding time; "
            "GENETIC COUNSELLING: AD; 50% offspring risk; variable expressivity counselling essential"
        ),
        "seed": 2834,
        "pt_vars": {
            "platelet_count": (20, 100),
            "bleeding_score": (1, 10),
            "transfusions_lifetime": (0, 3),
            "alloimmunised_pct": 5,
            "hsct_pct": 3,
        }
    },
    {
        "gene": "ANKRD26",
        "protein": (
            "ANKRD26 -- 10p12.1 AD -- 1181aa -- Ankyrin-Repeat-Domain-26-"
            "134kDa-Megakaryocyte-Thrombopoietin-TPO-Signalling-Regulator-"
            "THC2-Thrombocytopenia-2-AML-MDS-Predisposition-"
            "OMIM-Gene-610855-Disease-THC2-188000"
        ),
        "locus": "10p12.1",
        "protein_size": (
            "1181 aa / 134 kDa (ankyrin-repeat domain protein 26; "
            "expressed in haematopoietic stem cells, megakaryocytes; "
            "regulates THPO/TPO-MPL signalling and megakaryocyte differentiation; "
            "mutations in 5' UTR disrupt transcription factor binding → loss of normal silencing → "
            "ANKRD26 overexpression during megakaryocyte differentiation; "
            "LOF mutations → impaired megakaryocyte differentiation; "
            "AD heterozygous)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (heterozygous loss-of-function OR 5'UTR gain-of-expression mutations) — THC2; "
            "MECHANISM: "
            "  5'UTR mutations (most common): disrupt RUNX1/FLI1 repressor binding site → "
            "    ANKRD26 not silenced during terminal megakaryocyte differentiation → "
            "    ANKRD26 overexpressed → impairs proplatelet formation; "
            "  Coding LOF mutations: less common; "
            "  Both lead to impaired TPO-MPL signalling in megakaryocytes → "
            "    reduced platelet production → moderate thrombocytopenia; "
            "THROMBOCYTOPENIA SEVERITY: "
            "  Platelet count 50-150 × 10⁹/L (moderate; non-profound); "
            "  MPV: NORMAL or mildly elevated (not the giant platelets of BSS/MYH9); "
            "AML/MDS PREDISPOSITION — CRITICAL FEATURE: "
            "  ~5% lifetime risk of AML or MDS; "
            "  Mechanism: ANKRD26 mutations → aberrant megakaryocyte/HSC signalling → "
            "    clonal evolution in ANKRD26 background; "
            "  RUNX1/ANKRD26 epistasis: both affect MK differentiation axis → shared pathway; "
            "ELEVATED PLASMA TPO LEVELS: "
            "  Low platelet count → reduced TPO clearance by platelets → elevated plasma TPO; "
            "  (Normal platelet count disorders [GT] have normal TPO); "
            "  Elevated TPO in ANKRD26-THC2 = consistent finding; "
            "FAMILY HISTORY: AD with high penetrance for thrombocytopenia; variable AML expressivity"
        ),
        "disease_category": (
            "THROMBOCYTOPENIA-2 (THC2) — OMIM 188000; "
            "HAEMATOLOGICAL PROFILE: "
            "  Platelet count: 50-150 × 10⁹/L (mild-moderate thrombocytopenia); "
            "  MPV: normal or mildly elevated; "
            "  Platelet morphology: no giant platelets (differentiates from BSS/MYH9); "
            "  Platelet function: mildly impaired (reduced dense granule release in some); "
            "  Bleeding time: mildly prolonged or near-normal; "
            "CLINICAL PRESENTATION: "
            "  Bleeding: mild mucocutaneous (often minimal); "
            "  Epistaxis, bruising, menorrhagia (mild); "
            "  Often discovered incidentally (family history / routine CBC); "
            "  Perioperative bleeding risk (inform surgical team); "
            "MALIGNANCY SURVEILLANCE (AML/MDS): "
            "  ~5% lifetime AML/MDS risk: "
            "    Annual CBC with differential; "
            "    Low threshold for BM aspirate if unexplained cytopenias; "
            "    BM biopsy if blast count rises; "
            "    Molecular: BM cytogenetics at baseline and if worsening; "
            "DIFFERENTIATION FROM RUNX1-FPD/AML: "
            "  ANKRD26: ~5% AML risk; platelet function mild impairment; usually no platelet dense granule defect; "
            "  RUNX1-FPD/AML: 35-40% AML risk; dense granule defect (absent second-wave aggregation); "
            "  Both: AD thrombocytopenia with AML predisposition → molecular panel needed; "
            "ITP DIFFERENTIAL: "
            "  ANKRD26-THC2 frequently misdiagnosed as ITP; "
            "  ITP: absent family history; normal-sized platelets; responds to ITP treatment; "
            "  ANKRD26: positive family history; refractory to ITP treatment → should prompt gene panel"
        ),
        "disease_pathway": (
            "ANKRD26 / THROMBOPOIETIN-MPL MEGAKARYOCYTE DIFFERENTIATION PATHWAY: "
            "NORMAL ANKRD26 FUNCTION IN MEGAKARYOCYTES: "
            "  ANKRD26 expressed in HSCs and immature MKs; "
            "  As MKs differentiate: RUNX1 + FLI1 → bind ANKRD26 5'UTR → transcriptional silencing → "
            "    ANKRD26 protein levels fall during terminal MK differentiation; "
            "  Silencing required for normal proplatelet formation; "
            "  ANKRD26 modulates TPO/THPO → MPL (MPL = TPO receptor, CD110) → "
            "    JAK2/STAT5 + PI3K/Akt + MAPK/ERK → MK proliferation + differentiation; "
            "ANKRD26 5'UTR MUTATION (MOST COMMON): "
            "  RUNX1/FLI1 binding disrupted → ANKRD26 NOT silenced during MK terminal differentiation → "
            "  → ANKRD26 overexpressed → impairs proplatelet formation → fewer/larger platelets released; "
            "  TPO-MPL signalling dysregulated → reduced platelet production; "
            "AML/MDS MECHANISM: "
            "  Aberrant ANKRD26 expression → dysregulated MK/HSC signalling → "
            "    background for clonal haematopoiesis; "
            "  ANKRD26 overexpression may augment ERK/MAPK survival signals → "
            "    clonal expansion of mutant HSCs → AML/MDS over time; "
            "  RUNX1 co-mutation in some AML transformations from ANKRD26; "
            "ELEVATED TPO: "
            "  Low platelet count → less platelet MPL to clear circulating TPO → "
            "  → elevated plasma TPO levels (consistent finding in ANKRD26-THC2); "
            "  Distinguishes from ITP where TPO also elevated but pathology different"
        ),
        "pathognomonic": (
            "MODERATE THROMBOCYTOPENIA + ELEVATED PLASMA TPO + POSITIVE FAMILY HISTORY + "
            "NORMAL PLATELET SIZE + REFRACTORY TO ITP TREATMENT: "
            "  'ITP' that does not respond to IVIG or steroids + family history → gene panel; "
            "  ANKRD26 5'UTR mutation diagnostic; "
            "  Platelet count 50-150 × 10⁹/L with normal MPV differentiates from MYH9 (giant platelets) and BSS; "
            "ELEVATED PLASMA TPO LEVEL: "
            "  TPO elevated (>200 pg/mL in some series) in ANKRD26-THC2; "
            "  Reflects platelet consumption of TPO absent → free plasma TPO accumulates; "
            "  Not specific but consistent finding; "
            "AML/MDS DEVELOPMENT IN FAMILY MEMBER WITH THROMBOCYTOPENIA: "
            "  AML in a relative with thrombocytosis or thrombocytopenia → "
            "    consider FPD/AML (RUNX1) or THC2 (ANKRD26); "
            "  MOLECULAR PANEL MANDATORY in any familial thrombocytopenia + AML; "
            "5'UTR MUTATIONS NOT DETECTED BY STANDARD WES/WGS: "
            "  CRITICAL: most ANKRD26 pathogenic mutations are in the 5'UTR — "
            "    NOT in coding exons; standard exome sequencing MISSES THEM; "
            "  Requires: targeted ANKRD26 5'UTR Sanger sequencing or "
            "    specialised platelet gene panel that covers 5'UTR; "
            "  DIAGNOSTIC TRAP: negative WES does NOT exclude ANKRD26-THC2"
        ),
        "treatment": (
            "PRIMARILY OBSERVATION FOR MILD-MODERATE THROMBOCYTOPENIA: "
            "BLEEDING MANAGEMENT: "
            "  Antifibrinolytics (tranexamic acid) for mucosal bleeding; "
            "  Platelet transfusion for major bleeding/surgery; "
            "  DESMOPRESSIN: limited evidence but some use peri-procedurally; "
            "THROMBOPOIETIN RECEPTOR AGONISTS (TPO-RAs): "
            "  Eltrombopag, romiplostim: increase platelet count in ANKRD26-THC2; "
            "  SHORT-TERM use for surgery/procedures with low platelet count; "
            "  RISK: theoretical concern that TPO-RA may accelerate clonal evolution → use cautiously; "
            "  Evidence: limited case series; not standard of care; "
            "AVOID: "
            "  NSAIDs / aspirin (worsens bleeding); "
            "  Steroid courses or IVIG for ANKRD26 (ineffective, not ITP); "
            "  Long-term TPO-RA without monitoring for clonal evolution; "
            "AML/MDS SURVEILLANCE — MANDATORY: "
            "  Annual CBC + differential; "
            "  BM aspirate + cytogenetics if unexplained worsening cytopenia or blast increase; "
            "  Haematology follow-up every 12 months minimum; "
            "BONE MARROW TRANSPLANTATION (HSCT): "
            "  For AML/MDS transformation: standard intensive chemotherapy + HSCT; "
            "  Pre-emptive HSCT for high-risk ANKRD26 before AML: no established protocol; "
            "  Individual risk assessment; "
            "GENETIC COUNSELLING: "
            "  AD 50% offspring risk; all first-degree relatives offered testing; "
            "  Carriers asymptomatic rarely (near-complete penetrance for thrombocytopenia); "
            "  Inform of AML/MDS risk with appropriate psychosocial support"
        ),
        "seed": 2835,
        "pt_vars": {
            "platelet_count": (50, 150),
            "bleeding_score": (0, 8),
            "transfusions_lifetime": (0, 2),
            "alloimmunised_pct": 4,
            "hsct_pct": 5,
        }
    },
    {
        "gene": "RUNX1",
        "protein": (
            "RUNX1 -- 21q22.12 AD -- 453aa -- Runt-Related-Transcription-Factor-1-"
            "48kDa-RUNX1-CBFbeta-Heterodimer-AML1-CBFA2-"
            "Familial-Platelet-Disorder-AML-FPD-AML-"
            "OMIM-Gene-151385-Disease-FPD-AML-601399"
        ),
        "locus": "21q22.12",
        "protein_size": (
            "453 aa / 48 kDa (RUNX1 = AML1 = CBFA2; runt homology domain (RHD) binds DNA; "
            "transactivation domain recruits co-activators; "
            "heterodimerises with CBFβ (core-binding factor beta) → core binding factor complex; "
            "master haematopoietic transcription factor: regulates HSC self-renewal, "
            "  megakaryocyte differentiation, myeloid/lymphoid lineage commitment; "
            "RUNX1 germline LOF mutations → FPD/AML (familial platelet disorder + AML predisposition); "
            "RUNX1 somatic mutations: second most common in de novo AML (~13%); "
            "AD haploinsufficiency)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (heterozygous haploinsufficiency) — Familial Platelet Disorder / AML (FPD/AML); "
            "EPIDEMIOLOGY: "
            "  Rare germline; increasingly diagnosed with expanding gene panel use; "
            "  AML frequency: 35-40% lifetime risk; "
            "MECHANISM: "
            "  RUNX1 LOF → haploinsufficiency; "
            "  Megakaryocyte effects: "
            "    RUNX1 required for: "
            "      • platelet dense granule biogenesis (regulates granule genes); "
            "      • proplatelet formation; "
            "      • MK polyploidisation; "
            "    RUNX1 LOF → reduced dense granules → impaired second-wave ADP/ATP release → "
            "      absent second-wave platelet aggregation → platelet function defect (aspirin-like); "
            "  PLATELET SECRETION DEFECT: "
            "    Primary aggregation wave preserved (alphaIIbbeta3 intact); "
            "    Second wave absent (ADP release from dense granules impaired); "
            "    ADP, serotonin, ATP dense granule content reduced; "
            "AML/MDS TRANSFORMATION: "
            "  RUNX1 germline heterozygous → vulnerable to second-hit somatic mutation: "
            "    Loss of heterozygosity (LOH) at 21q; RUNX1 somatic mutation on second allele; "
            "    Co-operating mutations: CDC25C, FPD-associated mutations; "
            "  AML-M0 / AML-M2 most common transformation subtypes; "
            "  Median age of AML onset: 33 years (range: childhood to elderly); "
            "GENOTYPE-PHENOTYPE: "
            "  Haploinsufficiency: LOF → FPD; "
            "  Dominant-negative mutations (RHD point mutations): higher AML risk than LOF; "
            "  Missense in RHD (DNA-binding domain): dominant-negative → worse phenotype"
        ),
        "disease_category": (
            "FAMILIAL PLATELET DISORDER WITH PREDISPOSITION TO AML (FPD/AML) — OMIM 601399; "
            "DUAL PHENOTYPE: BLEEDING DISORDER + CANCER PREDISPOSITION: "
            "HAEMATOLOGICAL PROFILE: "
            "  Platelet count: 50-150 × 10⁹/L (mild-moderate thrombocytopenia); "
            "  MPV: normal or mildly elevated; "
            "  PLATELET FUNCTION: CHARACTERISTIC DEFECT: "
            "    LTA: "
            "      Primary aggregation: PRESENT (first wave to ADP, collagen, ristocetin normal); "
            "      SECOND WAVE ABSENT: reduced dense granule ADP release → no secondary aggregation; "
            "      PATTERN: first wave normal + absent second wave = dense granule secretion defect; "
            "    Electron microscopy: reduced/absent platelet dense granules; "
            "    Luminometry (ATP release): markedly reduced dense granule ATP secretion; "
            "CLINICAL PRESENTATION: "
            "  Bleeding: mild-moderate mucocutaneous; "
            "  Epistaxis, bruising, menorrhagia, post-surgical bleeding (disproportionate to platelet count); "
            "  'Easy bruising' in the family — family history critical; "
            "AML/MDS (35-40% LIFETIME RISK): "
            "  Often presents in 3rd-5th decade; "
            "  AML type: AML-M0, AML-M2, MDS/AML transition; "
            "  Median age ~33yr; "
            "  Family clusters: multiple affected members with AML decades apart; "
            "BONE MARROW SURVEILLANCE MANDATORY: "
            "  Annual CBC + differential; "
            "  Baseline BM biopsy + cytogenetics at diagnosis; "
            "  Annual BM if unexplained worsening; pre-emptive HSCT discussions in high-risk"
        ),
        "disease_pathway": (
            "RUNX1 TRANSCRIPTION FACTOR / MEGAKARYOCYTE DIFFERENTIATION / DENSE GRANULE PATHWAY: "
            "NORMAL RUNX1 FUNCTION: "
            "  RUNX1 (AML1): master haematopoietic TF; "
            "  RUNX1 + CBFβ heterodimer → binds RUNT-domain consensus sequence (TGT/cGGT) in promoters; "
            "  Key RUNX1 targets in megakaryocytes: "
            "    • RAB27B (Rab GTPase — dense granule biogenesis/trafficking); "
            "    • PF4 (platelet factor 4 — alpha granule marker); "
            "    • GP1BA, GP9 (GPIb complex genes); "
            "    • MPL (TPO receptor — MK self-renewal); "
            "    • MYL9 (myosin light chain — MK contraction); "
            "  Regulates ploidy increase in MKs (polyploidisation required for proplatelet formation); "
            "RUNX1 LOF IN MEGAKARYOCYTES: "
            "  Haploinsufficiency → 50% normal RUNX1 levels → "
            "  → RAB27B/other granule genes underexpressed → "
            "  → dense granule biogenesis impaired → fewer dense granules per platelet → "
            "  → secondary ADP/ATP release reduced → second wave aggregation absent; "
            "  → MK polyploidisation partially impaired → fewer platelets + mild thrombocytopenia; "
            "AML TRANSFORMATION PATHWAY: "
            "  Germline RUNX1 LOF (1 allele) → haploinsufficiency in HSC → "
            "  + Somatic second-hit (RUNX1 mutation on second allele, LOH, RUNX1 fusion): "
            "    → biallelic RUNX1 loss → complete CBF transcription loss → "
            "    → HSC proliferation without differentiation → AML-M0 / AML with RUNX1 mutation; "
            "  COOPERATING MUTATIONS: CDC25C, FLT3, NRAS — clonal evolution in FPD/AML background"
        ),
        "pathognomonic": (
            "FAMILIAL THROMBOCYTOPENIA + ABSENT SECOND-WAVE AGGREGATION (DENSE GRANULE DEFECT) + "
            "FAMILY HISTORY OF AML — PATHOGNOMONIC FOR FPD/AML (RUNX1): "
            "  Absent second wave in LTA: first-wave aggregation present but REVERSIBLE or absent sustained; "
            "  Dense granule ATP release (luminometry): markedly reduced; "
            "  Electron microscopy: reduced platelet dense granule numbers; "
            "  Family pedigree: thrombocytopenia in parent + AML in sibling/grandparent → FPD/AML panel; "
            "RUNX1 DOMINANT-NEGATIVE vs HAPLOINSUFFICIENCY RISK STRATIFICATION: "
            "  RHD missense dominant-negative mutations: HIGHER AML risk (~50%) than LOF truncations (~30%); "
            "  All RUNX1 germline mutations stratified by mutation class for AML surveillance intensity; "
            "AML PRECEDING DIAGNOSIS OF THROMBOCYTOPENIA: "
            "  AML without prior thrombocytopenia diagnosis is common presentation of FPD/AML; "
            "  AML with RUNX1 somatic mutation in young adult → GERMLINE RUNX1 TESTING MANDATORY; "
            "  ALL AML patients with RUNX1 somatic mutation AND positive family history → germline testing; "
            "CRITICAL WES/NGS CAVEAT: "
            "  Standard AML somatic NGS (tumour only) may miss germline RUNX1 origin; "
            "  Germline testing (blood, buccal, cultured skin fibroblast) required if AML with thrombocytopenia family history; "
            "ASPIRIN-LIKE PLATELET FUNCTION PATTERN WITHOUT ASPIRIN USE: "
            "  Reduced dense granule secondary release → aspirin-like absent second wave; "
            "  Patient history of aspirin-like result without aspirin exposure → RUNX1 investigation"
        ),
        "treatment": (
            "BLEEDING MANAGEMENT: "
            "ANTIFIBRINOLYTICS (FIRST-LINE): "
            "  Tranexamic acid for mucosal bleeding, dental procedures; "
            "PLATELET TRANSFUSION: "
            "  For major bleeding or peri-surgical cover when platelet count <50; "
            "  Platelet function defect (dense granule) — transfusion provides functional donor platelets; "
            "DESMOPRESSIN (DDAVP): "
            "  SOME BENEFIT: DDAVP increases vWF → mild augmentation of platelet adhesion; "
            "  0.3 mcg/kg before dental/minor surgery; "
            "AVOID: "
            "  NSAIDs/aspirin (dense granule defect → aspirin-like effect already present); "
            "  Drugs that inhibit platelet function (P2Y12 inhibitors, GPs IIb/IIIa inhibitors); "
            "AML/MDS SURVEILLANCE (MANDATORY): "
            "  Annual haematology review + CBC; "
            "  Baseline BM biopsy + cytogenetics at FPD/AML diagnosis; "
            "  BM repeat if unexplained worsening of thrombocytopenia or anaemia; "
            "  Molecular: next-generation sequencing of BM for clonal haematopoiesis markers (DNMT3A, TET2, etc.); "
            "PRE-EMPTIVE HSCT COUNSELLING: "
            "  High-risk RUNX1 (dominant-negative RHD mutation, prior RUNX1-somatic clones in BM): "
            "    pre-emptive HSCT consideration in 20s-30s; "
            "  Sibling donor with FPD/AML excluded before use as HSCT donor; "
            "AML TREATMENT: "
            "  Induction chemotherapy + HSCT; "
            "  Sibling donor evaluation: all first-degree relatives tested for FPD/AML before use as donor; "
            "  Allogeneic HSCT curative for AML + eliminates thrombocytopenia; "
            "GENETIC COUNSELLING: "
            "  AD, 50% offspring risk; family cascade testing; psychosocial support for AML risk disclosure; "
            "  Reproductive planning: preimplantation genetic testing (PGT) available"
        ),
        "seed": 2836,
        "pt_vars": {
            "platelet_count": (50, 150),
            "bleeding_score": (1, 9),
            "transfusions_lifetime": (0, 3),
            "alloimmunised_pct": 3,
            "hsct_pct": 35,
        }
    },
    {
        "gene": "GFI1B",
        "protein": (
            "GFI1B -- 9q34.13 AD -- 330aa -- Growth-Factor-Independence-1B-"
            "37kDa-Zinc-Finger-Transcriptional-Repressor-"
            "SNAG-Domain-ZF1-ZF6-Megakaryocyte-Erythroid-Regulator-"
            "OMIM-Gene-604383-Disease-GFI1B-Thrombocytopenia-187900"
        ),
        "locus": "9q34.13",
        "protein_size": (
            "330 aa / 37 kDa (Growth Factor Independence 1B = GFI1B; "
            "zinc-finger transcriptional repressor; "
            "SNAG domain (SNAIL/Gfi repressor domain) at N-terminus recruits LSD1/CoREST complex; "
            "6 zinc finger motifs (ZF1-6): ZF3-5 bind DNA; "
            "paralog of GFI1 (myeloid regulator); "
            "expressed in haematopoietic cells: megakaryocytes, erythroid cells, HSCs; "
            "regulates megakaryocyte terminal differentiation AND erythropoiesis; "
            "heterozygous dominant-negative mutations → "
            "  macrothrombocytopenia + red cell membrane changes; "
            "AR biallelic: severe haematopoietic failure"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (heterozygous dominant-negative) — GFI1B-related macrothrombocytopenia; "
            "MECHANISM: "
            "  GFI1B dominant-negative mutations (typically ZF4-5 DNA-binding domain): "
            "    Mutant GFI1B protein retains SNAG domain (recruits LSD1/CoREST); "
            "    Mutant CANNOT bind DNA (ZF4-5 disrupted) → "
            "    → accumulates at LSD1 without repression → "
            "    → sequesters LSD1 away from wildtype GFI1B → "
            "    → dominant-negative effect on wildtype GFI1B function → haploinsufficiency AND sequestration; "
            "  MEGAKARYOCYTE EFFECTS: "
            "    GFI1B normally represses genes inhibiting MK differentiation (e.g., GATA1-opposing); "
            "    GFI1B LOF → terminal MK differentiation impaired → fewer, larger, hypogranular platelets; "
            "    Platelet alpha-granule content reduced (grey platelet-like appearance); "
            "  ERYTHROID EFFECTS: "
            "    GFI1B also regulates erythroid differentiation; "
            "    Some patients: erythrocyte membrane abnormalities (CD42b on RBCs, ovalocytosis); "
            "    Rare: red cell aplasia (severe GFI1B mutations); "
            "PLATELET PHENOTYPE: "
            "  Macrothrombocytopenia (giant platelets + reduced count); "
            "  Platelet alpha-granules reduced → 'grey platelet-like' on electron microscopy; "
            "  CD42b aberrantly expressed on red blood cells (RBCs) in some GFI1B patients — DIAGNOSTIC; "
            "SEVERITY SPECTRUM: "
            "  Mild: incidental macrothrombocytopenia, minimal bleeding; "
            "  Moderate: recurrent mucosal bleeding; "
            "  Severe: red cell aplasia, bone marrow failure (rare AR biallelic form)"
        ),
        "disease_category": (
            "GFI1B-RELATED MACROTHROMBOCYTOPENIA — OMIM 187900 (autosomal dominant form); "
            "HAEMATOLOGICAL PROFILE: "
            "  Platelet count: 30-120 × 10⁹/L (moderate thrombocytopenia); "
            "  MPV: elevated (macrothrombocytes); "
            "  Platelet alpha-granules: reduced (grey platelet appearance, EM); "
            "  Platelet function: alpha-granule secretion impaired (PF4, vWF release reduced); "
            "  FLOW CYTOMETRY DIAGNOSTIC CLUE: "
            "    CD42b (GPIbα) ABERRANTLY EXPRESSED ON RED BLOOD CELLS — "
            "    PATHOGNOMONIC finding in GFI1B-related thrombocytopenia; "
            "    Normal: CD42b on platelets only; "
            "    GFI1B-affected: CD42b+ erythrocytes detectable by flow; "
            "  CD34+ progenitors: increased in some patients; "
            "CLINICAL PRESENTATION: "
            "  Mild-moderate mucocutaneous bleeding; "
            "  Bruising, epistaxis, menorrhagia; "
            "  Generally milder than GT or BSS; "
            "  Perioperative bleeding risk; "
            "ERYTHROID MANIFESTATIONS (subset): "
            "  Mild ovalocytosis or elliptocytosis on blood film; "
            "  Red cell aplasia: rare, in very severe mutations or AR biallelic; "
            "  Anaemia: usually mild from erythroid changes"
        ),
        "disease_pathway": (
            "GFI1B ZINC-FINGER REPRESSOR / MEGAKARYOCYTE-ERYTHROID DIFFERENTIATION PATHWAY: "
            "NORMAL GFI1B FUNCTION: "
            "  GFI1B SNAG domain recruits LSD1 (KDM1A, histone H3K4 demethylase) + CoREST complex; "
            "  GFI1B DNA-binding (ZF3-5): recognises TAAATCAC(T/A)GCA consensus; "
            "  Transcriptional repression mechanism: GFI1B → LSD1/CoREST → H3K4me2 → H3K4me0 → gene silencing; "
            "  Target genes repressed by GFI1B in MKs: "
            "    • GATA2 (MK progenitor gene — must be silenced for terminal differentiation); "
            "    • HSC self-renewal genes; "
            "    • Proteins inhibiting proplatelet formation; "
            "  GFI1B in erythroid cells: represses GATA2 + HIF1α → promotes GATA1-driven erythropoiesis; "
            "GFI1B DOMINANT-NEGATIVE MECHANISM: "
            "  ZF4-5 deletion/mutation → cannot bind DNA → "
            "  → SNAG domain intact → still recruits LSD1 (sequestration); "
            "  → LSD1 unavailable for wildtype GFI1B complex → "
            "  → wildtype GFI1B cannot repress targets → "
            "  → GATA2 and other inhibitory genes not silenced → "
            "  → terminal MK differentiation arrested → fewer, hypogranular, giant platelets; "
            "ALPHA-GRANULE DEFECT: "
            "  GFI1B regulates genes required for alpha-granule biogenesis (VPS33B target genes); "
            "  GFI1B LOF → grey platelet-like alpha-granule hypoplasia; "
            "ABERRANT CD42b ON RBCS: "
            "  GFI1B LOF → GP1BA (CD42b) not silenced in erythroid precursors → "
            "  → CD42b expressed on mature RBCs → detectable by flow cytometry → DIAGNOSTIC"
        ),
        "pathognomonic": (
            "CD42b (GPIbα) EXPRESSION ON RED BLOOD CELLS BY FLOW CYTOMETRY — "
            "PATHOGNOMONIC FOR GFI1B-RELATED MACROTHROMBOCYTOPENIA: "
            "  CD42b+ erythrocytes detectable by flow cytometry (normal = 0% RBCs are CD42b+); "
            "  GFI1B normally silences GP1BA in erythroid lineage; "
            "  GFI1B LOF → GP1BA expressed in erythroid cells → CD42b+ RBCs; "
            "  This finding is NOT seen in BSS, MYH9, GT, ANKRD26, RUNX1, ITP; "
            "  DIAGNOSTIC EVEN BEFORE GENETIC TESTING; "
            "GREY PLATELET APPEARANCE ON ELECTRON MICROSCOPY: "
            "  Absent/reduced platelet alpha-granules (grey platelet); "
            "  Distinguishes from MYH9 (normal granules but Döhle inclusions) and BSS (normal granules); "
            "  Grey platelet syndrome DDx: GFI1B + GPS (NBEAL2) + GATA1 mutations; "
            "  GFI1B: CD42b on RBCs + giant platelets + grey appearance = unique triad; "
            "MACROTHROMBOCYTOPENIA WITH POSITIVE FAMILY HISTORY + AUTOSOMAL DOMINANT: "
            "  Both MYH9 and GFI1B are AD macrothrombocytopenia; "
            "  MYH9: Döhle inclusions + normal granules + normal CD42b on RBCs; "
            "  GFI1B: grey platelets + CD42b on RBCs + no Döhle inclusions; "
            "OVALOCYTOSIS / ELLIPTOCYTOSIS ON BLOOD FILM: "
            "  Mild erythrocyte membrane changes in subset → supports GFI1B (not MYH9/BSS)"
        ),
        "treatment": (
            "PRIMARILY OBSERVATION FOR MILD-MODERATE MACROTHROMBOCYTOPENIA: "
            "ANTIFIBRINOLYTICS (FIRST-LINE FOR MUCOSAL BLEEDING): "
            "  Tranexamic acid 25 mg/kg TDS; topical; before dental procedures; "
            "PLATELET TRANSFUSION: "
            "  For major bleeding or surgical cover; "
            "  Standard donor platelets provide normally functional platelets; "
            "  Alloimmunisation risk lower than GT/BSS; "
            "DESMOPRESSIN (DDAVP): "
            "  Limited evidence; may help (vWF increase + platelet activation boost); "
            "  0.3 mcg/kg before minor procedures; "
            "RECOMBINANT FACTOR VIIA (rFVIIa): "
            "  For major bleeding refractory to platelet transfusion; "
            "  Off-label but reported effective; "
            "ALPHA-GRANULE DEFICIENCY — no specific therapy: "
            "  Platelet transfusion provides exogenous functional alpha-granules; "
            "  PF4, vWF, fibrinogen provided by transfused donor platelets; "
            "ERYTHROID COMPLICATIONS: "
            "  Anaemia (if present): folic acid supplementation; "
            "  Red cell aplasia (severe/biallelic): HSCT consideration; "
            "  Ovalocytosis: usually asymptomatic; splenomegaly monitoring; "
            "AVOID: "
            "  NSAIDs / aspirin (alpha-granule defect + thrombocytopenia); "
            "  Excessive platelet transfusions (minimise alloimmunisation risk); "
            "GENETIC COUNSELLING: "
            "  AD, 50% offspring risk; variable expressivity (carrier parent may have mild thrombocytopenia); "
            "  Biallelic (AR) GFI1B: severe phenotype → HSCT counselling from diagnosis"
        ),
        "seed": 2837,
        "pt_vars": {
            "platelet_count": (30, 120),
            "bleeding_score": (1, 10),
            "transfusions_lifetime": (0, 4),
            "alloimmunised_pct": 6,
            "hsct_pct": 7,
        }
    },
]

DEFINITIONS = {
    "definitions": [
        {
            "term": "Glanzmann Thrombasthenia (GT)",
            "definition": (
                "Autosomal recessive platelet disorder; absent or dysfunctional alphaIIbbeta3 (GPIIb-IIIa); "
                "ITGA2B (GT1) or ITGB3 (GT2); "
                "Platelet count: NORMAL; Platelet aggregation: ABSENT to all agonists except ristocetin; "
                "Ristocetin PRESERVED (GPIb-vWF axis intact); Clot retraction absent; "
                "Treatment: antifibrinolytics, platelet transfusion (minimise), rFVIIa, HSCT (curative); "
                "Alloimmunisation (anti-alphaIIbbeta3 / anti-HPA-1a) limits transfusion options"
            )
        },
        {
            "term": "Bernard-Soulier Syndrome (BSS)",
            "definition": (
                "Autosomal recessive; absent or dysfunctional GPIb-IX-V complex; "
                "GP1BA (BSS-A), GP1BB (BSS-B), GP9 (BSS-C); "
                "TRIAD: giant platelets (PATHOGNOMONIC) + thrombocytopenia + prolonged bleeding time; "
                "LTA: ABSENT to ristocetin; PRESERVED to ADP/collagen/AA; "
                "CD42b flow cytometry markedly reduced; "
                "Frequently misdiagnosed as ITP (giant platelets counted as lymphocytes); "
                "Treatment: antifibrinolytics, DDAVP, platelet transfusion, rFVIIa, HSCT"
            )
        },
        {
            "term": "MYH9-Related Disease (MYH9-RD)",
            "definition": (
                "Autosomal dominant; MYH9 heterozygous mutations; most common hereditary macrothrombocytopenia; "
                "Historically named: May-Hegglin anomaly, Fechtner, Sebastian, Epstein syndromes — all unified as MYH9-RD; "
                "DIAGNOSTIC TRIAD: macrothrombocytopenia + Döhle-body-like neutrophil inclusions (PATHOGNOMONIC) + relatively mild bleeding; "
                "LTA: near-normal (platelet receptors intact); "
                "Extrasystemic: SNHL (30-70%), nephritis (25-50%), cataracts (20-35%); "
                "Treatment: antifibrinolytics, DDAVP, monitoring for organ complications; nephropathy: ACEi/ARB; HSCT rarely needed"
            )
        },
        {
            "term": "Familial Platelet Disorder / AML (FPD/AML — RUNX1)",
            "definition": (
                "Autosomal dominant; RUNX1 germline haploinsufficiency; "
                "DUAL PHENOTYPE: thrombocytopenia + platelet dense granule defect + 35-40% lifetime AML/MDS risk; "
                "LTA: absent second-wave aggregation (dense granule secretion defect); "
                "Electron microscopy: reduced dense granules; "
                "CRITICAL: ALL AML patients with RUNX1 somatic mutation + family history → germline testing; "
                "Standard WES may miss coding RUNX1 variants; dominant-negative RHD mutations → highest AML risk; "
                "Annual BM surveillance mandatory; HSCT for AML transformation"
            )
        },
        {
            "term": "ANKRD26-Thrombocytopenia-2 (THC2)",
            "definition": (
                "Autosomal dominant; ANKRD26 mutations (5'UTR most common); "
                "Moderate thrombocytopenia (50-150 × 10⁹/L) with normal platelet size; "
                "AML/MDS risk ~5% lifetime; elevated plasma TPO; "
                "DIAGNOSTIC TRAP: 5'UTR mutations NOT detected by standard WES/WGS — requires targeted 5'UTR Sanger or specialised panel; "
                "Frequently misdiagnosed as ITP (no response to IVIG/steroids); "
                "Annual CBC surveillance mandatory; BM evaluation for unexplained worsening"
            )
        },
        {
            "term": "GFI1B-Related Macrothrombocytopenia",
            "definition": (
                "Autosomal dominant; GFI1B dominant-negative mutations (ZF4-5); "
                "Macrothrombocytopenia + reduced alpha-granules (grey platelet appearance); "
                "PATHOGNOMONIC: CD42b (GPIbα) aberrantly expressed on RED BLOOD CELLS by flow cytometry; "
                "CD42b+ RBCs = unique to GFI1B; absent in all other hereditary platelet disorders; "
                "Treatment: antifibrinolytics, platelet transfusion; HSCT for severe/biallelic cases; "
                "Mild erythrocyte membrane changes (ovalocytosis) in subset"
            )
        },
        {
            "term": "Dense Granule Defect (δ-storage pool disease)",
            "definition": (
                "Reduced platelet dense granule number or ADP/ATP/serotonin content; "
                "Causes in hereditary thrombocytopenias: RUNX1 (FPD/AML), HPS (Hermansky-Pudlak), Chediak-Higashi; "
                "LTA pattern: primary aggregation wave present + ABSENT second wave (second wave requires ADP release); "
                "Luminometry: markedly reduced ATP release; "
                "RUNX1 vs aspirin effect: both give absent second wave; aspirin use history must be excluded; "
                "EM: electron-dense bodies absent/reduced on dense granule staining"
            )
        },
        {
            "term": "Ristocetin-Induced Platelet Agglutination (RIPA) in Platelet Disorders",
            "definition": (
                "Ristocetin is a polycationic antibiotic that forces vWF to bind GPIbα → platelet agglutination (NOT aggregation); "
                "NORMAL: agglutination seen at 1.5 mg/mL ristocetin (requires GPIbα + vWF); "
                "ABSENT RIPA: BSS (any type GP1BA/GP1BB/GP9) — no GPIbα; vWD type 3 — no vWF; "
                "PRESERVED RIPA: Glanzmann (GT) — GPIbα intact; MYH9; ANKRD26; RUNX1; GFI1B; ITP; "
                "LOW-DOSE RIPA (0.5 mg/mL): ENHANCED in platelet-type VWD (GOF GP1BA) and VWD type 2B; "
                "RIPA result distinguishes BSS from GT and ITP in one test"
            )
        },
    ],
    "standards": [
        "European Haematology Association (EHA) Guidelines — Inherited Platelet Disorders (2019)",
        "International Society on Thrombosis and Haemostasis (ISTH) — Platelet Physiology Subcommittee",
        "OMIM: Glanzmann GT1 273800 (ITGA2B), GT2 273800 (ITGB3), BSS 231200 (GP1BA/GP9), MYH9-RD 155100 (MYH9), FPD/AML 601399 (RUNX1), THC2 188000 (ANKRD26), GFI1B-RD 187900 (GFI1B)",
        "Nurden AT et al. Rare inherited bleeding disorders dependent on how platelets make contacts. Blood. 2015",
        "Pecci A et al. MYH9-Related Disease: Genotype-Phenotype Correlations. J Thromb Haemost. 2014",
        "Pippucci T et al. Mutations in the 5'UTR of ANKRD26 Gene Lead to Thrombocytopenia-2. Am J Hum Genet. 2011",
        "Bluteau D et al. Mutations in GFI1B result in a new form of familial thrombocytopenia. Nat Genet. 2014",
        "Song WJ et al. Haploinsufficiency of CBFA2 causes familial thrombocytopenia with propensity to develop AML. Nat Genet. 1999",
        "British Society for Haematology (BSH) — Guidelines on the Diagnosis of Inherited Platelet Function Disorders (2020)",
        "Hayward CP et al. Diagnostic Approach to Platelet Function Disorders. J Thromb Haemost. 2010",
    ]
}


def _make_patients(gene_data):
    rng = random.Random(gene_data["seed"])
    pts = []
    count_lo, count_hi = gene_data["pt_vars"]["platelet_count"]
    score_lo, score_hi = gene_data["pt_vars"]["bleeding_score"]
    tx_lo, tx_hi = gene_data["pt_vars"]["transfusions_lifetime"]
    alloimmunised_pct = gene_data["pt_vars"]["alloimmunised_pct"]
    hsct_pct = gene_data["pt_vars"]["hsct_pct"]

    for i in range(40):
        count = round(rng.uniform(count_lo, count_hi), 0)
        score = round(rng.uniform(score_lo, score_hi), 1)
        tx = int(rng.uniform(tx_lo, tx_hi))
        age_diag = round(rng.uniform(0.0, 25.0), 1)
        sex = rng.choice(["M", "F"])
        alloimmunised = rng.random() < alloimmunised_pct / 100
        hsct = rng.random() < hsct_pct / 100

        # gene-specific features
        giant_platelets = gene_data["gene"] in ("GP1BA", "GP9", "MYH9", "GFI1B")
        absent_ristocetin = gene_data["gene"] in ("GP1BA", "GP9")
        normal_ristocetin = gene_data["gene"] in ("ITGA2B", "ITGB3", "MYH9", "ANKRD26", "RUNX1", "GFI1B")
        absent_all_agonists = gene_data["gene"] in ("ITGA2B", "ITGB3")
        dohle_inclusions = gene_data["gene"] == "MYH9" and rng.random() < 0.95
        aml_event = gene_data["gene"] in ("RUNX1", "ANKRD26") and rng.random() < (
            0.38 if gene_data["gene"] == "RUNX1" else 0.05
        )
        cd42b_on_rbc = gene_data["gene"] == "GFI1B" and rng.random() < 0.88
        dense_granule_defect = gene_data["gene"] == "RUNX1"
        snhl = gene_data["gene"] == "MYH9" and rng.random() < 0.45
        nephritis = gene_data["gene"] == "MYH9" and rng.random() < 0.32

        pts.append({
            "patient_id": f"{gene_data['gene']}-{i+1:03d}",
            "gene": gene_data["gene"],
            "sex": sex,
            "age_at_diagnosis_years": age_diag,
            "platelet_count": count,
            "bleeding_score_isth": score,
            "transfusions_lifetime": tx,
            "giant_platelets": giant_platelets,
            "absent_ristocetin": absent_ristocetin,
            "normal_ristocetin": normal_ristocetin,
            "absent_all_agonists": absent_all_agonists,
            "dohle_inclusions": dohle_inclusions,
            "cd42b_on_rbc": cd42b_on_rbc,
            "dense_granule_defect": dense_granule_defect,
            "aml_mds_event": aml_event,
            "snhl": snhl,
            "nephritis": nephritis,
            "alloimmunised": alloimmunised,
            "hsct_performed": hsct,
            "seed": gene_data["seed"],
        })
    return pts


def generate_overview():
    total_patients = 0
    all_genes = []

    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        total_patients += len(patients)
        aml_ev = sum(1 for p in patients if p["aml_mds_event"])
        hsct = sum(1 for p in patients if p["hsct_performed"])
        alloimmunised = sum(1 for p in patients if p["alloimmunised"])
        dohle = sum(1 for p in patients if p["dohle_inclusions"])
        cd42b = sum(1 for p in patients if p["cd42b_on_rbc"])
        snhl = sum(1 for p in patients if p["snhl"])
        nephritis = sum(1 for p in patients if p["nephritis"])

        all_genes.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "disease_category": gene_data["disease_category"],
            "inheritance": gene_data["inheritance"],
            "n_patients": len(patients),
            "median_platelet_count": round(
                sorted(p["platelet_count"] for p in patients)[len(patients) // 2], 0
            ),
            "mean_bleeding_score": round(
                sum(p["bleeding_score_isth"] for p in patients) / len(patients), 1
            ),
            "pct_aml_mds": round(aml_ev / len(patients) * 100, 1),
            "pct_hsct": round(hsct / len(patients) * 100, 1),
            "pct_alloimmunised": round(alloimmunised / len(patients) * 100, 1),
            "pct_dohle_inclusions": round(dohle / len(patients) * 100, 1),
            "pct_cd42b_on_rbc": round(cd42b / len(patients) * 100, 1),
            "pct_snhl": round(snhl / len(patients) * 100, 1),
            "pct_nephritis": round(nephritis / len(patients) * 100, 1),
            "seed": gene_data["seed"],
        })

    return {
        "atlas": "Hereditary Platelet Disorders Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Platelet Disorders Reference — "
            "ITGA2B·ITGB3·GP1BA·GP9·MYH9·ANKRD26·RUNX1·GFI1B"
        ),
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "gene_summaries": all_genes,
        "seeds": "2830-2837",
        "pathway_categories": [
            {
                "pathway": "Platelet Fibrinogen Receptor / alphaIIbbeta3 Deficiency (Glanzmann Thrombasthenia)",
                "genes": ["ITGA2B", "ITGB3"],
                "note": (
                    "ITGA2B (GT1) + ITGB3 (GT2): absent alphaIIbbeta3 → absent aggregation to ALL agonists except ristocetin; "
                    "NORMAL platelet count + NORMAL platelet size = GT fingerprint; "
                    "Ristocetin PRESERVED (GPIb-vWF intact); alloimmunisation limits transfusions; "
                    "rFVIIa for alloimmunised; antifibrinolytics first-line mucosal; HSCT curative"
                ),
            },
            {
                "pathway": "GPIb-IX-V Complex / vWF-Platelet Adhesion Deficiency (Bernard-Soulier Syndrome)",
                "genes": ["GP1BA", "GP9"],
                "note": (
                    "GP1BA (BSS-A) + GP9 (BSS-C): absent GPIb-IX-V → absent ristocetin agglutination + giant platelets; "
                    "Giant platelets PATHOGNOMONIC; thrombocytopenia present; "
                    "aggregation to ADP/collagen/AA preserved (alphaIIbbeta3 intact); "
                    "Frequently misdiagnosed as ITP — giant platelets counted as lymphocytes; "
                    "Gene panel required to subtype BSS"
                ),
            },
            {
                "pathway": "Non-Muscle Myosin IIA Cytoskeletal Motor (MYH9-RD)",
                "genes": ["MYH9"],
                "note": (
                    "NMHC-IIA haploinsufficiency → giant platelets + Döhle-body-like neutrophil inclusions (PATHOGNOMONIC); "
                    "Most common hereditary macrothrombocytopenia; May-Hegglin/Fechtner/Sebastian/Epstein = same gene; "
                    "Platelet aggregation near-normal (receptors intact); "
                    "Extrasystemic: SNHL (30-70%), nephritis (25-50%), cataracts; annual organ screening mandatory"
                ),
            },
            {
                "pathway": "Megakaryocyte Differentiation / AML Predisposition Axis",
                "genes": ["ANKRD26", "RUNX1"],
                "note": (
                    "ANKRD26 (THC2): 5'UTR mutations missed by WES; ~5% AML risk; normal platelet size; "
                    "elevated plasma TPO; frequent ITP misdiagnosis; "
                    "RUNX1 (FPD/AML): dense granule secretion defect + absent second-wave; 35-40% AML risk; "
                    "ALL AML with RUNX1 somatic + thrombocytopenia family history → germline testing; "
                    "Both: AD, annual BM surveillance mandatory"
                ),
            },
            {
                "pathway": "GFI1B Zinc-Finger Repressor / Megakaryocyte-Erythroid Differentiation",
                "genes": ["GFI1B"],
                "note": (
                    "GFI1B dominant-negative → impaired MK terminal differentiation → macrothrombocytopenia + grey platelets; "
                    "PATHOGNOMONIC: CD42b (GPIbα) aberrantly expressed on RED BLOOD CELLS — unique to GFI1B; "
                    "Not seen in any other hereditary platelet disorder; erythroid membrane abnormalities in subset"
                ),
            },
        ],
        "critical_distinctions": [
            "GLANZMANN (GT) vs BERNARD-SOULIER (BSS): GT = NORMAL COUNT + NORMAL SIZE + absent ALL agonists + NORMAL ristocetin; BSS = GIANT PLATELETS + THROMBOCYTOPENIA + absent RISTOCETIN + normal other agonists; opposite patterns",
            "BSS vs MYH9 (both giant platelets + thrombocytopenia): BSS = absent ristocetin + AR + no inclusions; MYH9 = NORMAL ristocetin + AD + Döhle-body-like inclusions; inclusions are pathognomonic for MYH9",
            "GT vs ITP (both: normal platelet count or near-normal): GT = absent aggregation; ITP = normal aggregation; PFA-100 prolonged in GT, normal in ITP when count corrected",
            "BSS vs ITP (both thrombocytopenia): BSS = giant platelets; ITP = small or normal-sized platelets; manual film review MANDATORY when 'ITP' does not respond to steroids/IVIG",
            "RUNX1 (FPD/AML) vs ANKRD26 (THC2): both AD thrombocytopenia + AML risk; RUNX1: dense granule defect + absent second wave + 35-40% AML; ANKRD26: mild function defect + ~5% AML; both miss on WES (ANKRD26 5'UTR, RUNX1 germline vs somatic)",
            "GFI1B vs GREY PLATELET SYNDROME (NBEAL2): both grey platelets on EM + macrothrombocytopenia; GFI1B: CD42b on RBCs; NBEAL2 (GPS): no CD42b on RBCs + different inheritance; CD42b flow on RBCs distinguishes them",
            "ANKRD26 5'UTR DIAGNOSTIC TRAP: mutations in 5'UTR of ANKRD26 NOT found by WES/WGS; requires Sanger of 5'UTR or specific platelet panel; negative WES does NOT exclude ANKRD26-THC2",
            "ALLOIMMUNISATION RANKING: GT2/ITGB3 HPA-1a (~20%) > GT1/ITGA2B HPA (~18%) > BSS anti-GPIb (~18-22%) > GFI1B (~6%) > MYH9/ANKRD26/RUNX1 (<5%); GT/BSS patients at highest alloimmunisation risk from platelet transfusion",
            "AML RISK RANKING: RUNX1-FPD (35-40%) >> ANKRD26-THC2 (~5%) > others (GT, BSS, MYH9, GFI1B: NOT elevated AML risk); FPD/AML and THC2 are the only platelet disorders with significant inherited AML predisposition",
            "RECOMBINANT FACTOR VIIA USE: GT and BSS alloimmunised patients → rFVIIa replaces platelet transfusion; MYH9/ANKRD26/RUNX1/GFI1B: rFVIIa rarely needed; antifibrinolytics usually sufficient",
        ],
    }


def generate_breakdown():
    result = []
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        result.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "n_patients": len(patients),
            "patients": patients[:5],
        })
    return {"genes": result, "total": len(ATLAS_GENES), "seeds": "2830-2837"}


def generate_definitions():
    return {
        "atlas": "Hereditary Platelet Disorders Atlas",
        "definitions": DEFINITIONS["definitions"],
        "standards": DEFINITIONS["standards"],
        "gene_count": len(ATLAS_GENES),
        "seeds": "2830-2837",
    }
