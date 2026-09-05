#!/usr/bin/env python3
"""Ehlers-Danlos-Atlas — Complete 8-Gene Hereditary Ehlers-Danlos Syndromes Atlas
COL5A1  (Pro-alpha1(V) collagen; 1838 aa; 9q34.3; OMIM gene 120215;
         Classical EDS type 1 (cEDS-1) — AD haploinsufficiency;
         skin hyperextensibility + atrophic scarring HALLMARK + joint hypermobility;
         velvety soft doughy skin; Gorlin sign (licking elbow tip); ~90% of cEDS) ·
COL5A2  (Pro-alpha2(V) collagen; 1499 aa; 2q32.2; OMIM gene 120190;
         Classical EDS type 2 (cEDS-2) — AD; same phenotype as cEDS-1; ~10% of cEDS;
         milder atrophic scarring possible; Beighton ≥5 + skin signs → diagnose cEDS) ·
COL3A1  (Pro-alpha1(III) collagen; 1466 aa; 2q32.2; OMIM gene 120180;
         Vascular EDS (vEDS) — AD; MOST LETHAL EDS type; median survival 48 years;
         SPONTANEOUS HOLLOW VISCUS RUPTURE — bowel/uterine perforation WITHOUT trauma;
         PATHOGNOMONIC: thin translucent skin + acrogeria + characteristic facies;
         NO elective surgery; NO colonoscopy; NO angiography;
         celiprolol ONLY RCT-proven treatment — reduces arterial events 36%) ·
TNXB    (Tenascin-X; 4268 aa; 6p21.3; OMIM gene 600985;
         Classical-like EDS (clEDS) — AR biallelic — COMPLETE LOSS;
         haploinsufficiency (one LOF allele) → HSD (hypermobility spectrum disorder) NOT EDS;
         joint hypermobility without skin fragility (major distinction from cEDS);
         TNXB contiguous deletion with CYP21A2 → adrenal insufficiency MANDATORY screen) ·
ADAMTS2 (ADAM metallopeptidase with thrombospondin type 1 motif 2; 1211 aa; 5q35.3;
         OMIM gene 604539; Dermatosparaxis EDS (dEDS) — AR; RAREST EDS subtype;
         SAGGING REDUNDANT SKIN at birth PATHOGNOMONIC — skin so lax it hangs in folds;
         skin friability EXTREME — bruising with minimal trauma; procollagen N-propeptide retention;
         skin biopsy: procollagen I N-terminal propeptide accumulation = diagnostic) ·
PLOD1   (Procollagen-lysine,2-oxoglutarate 5-dioxygenase 1 / LH1; 727 aa; 1p36.3;
         OMIM gene 153454; Kyphoscoliotic EDS type 1 (kEDS-1) — AR;
         NEONATAL MUSCULAR HYPOTONIA + PROGRESSIVE SCOLIOSIS from birth PATHOGNOMONIC;
         URINE LYSYLPYRIDINOLINE:HYDROXYLYSYLPYRIDINOLINE RATIO (LP:HP ratio) >0.09 PATHOGNOMONIC;
         ocular fragility — globe rupture risk; ascorbic acid supplementation) ·
FKBP14  (FKBP prolyl isomerase 14; 150 aa; 7p14.3; OMIM gene 614505;
         Kyphoscoliotic EDS type 2 (kEDS-2) — AR;
         SAME as kEDS-1 clinically: neonatal hypotonia + kyphoscoliosis + joint hypermobility;
         DISTINCTIONS from kEDS-1: normal urine LP:HP ratio; congenital hearing loss (50%);
         myopathy prominent; allelic to musculocontractural EDS features possible) ·
COL1A2  (Pro-alpha2(I) collagen; 1366 aa; 7q21.3; OMIM gene 120160;
         Cardiac-valvular EDS (cvEDS) — AR biallelic; EXTREMELY RARE;
         SEVERE PROGRESSIVE VALVE DISEASE PATHOGNOMONIC — aortic + mitral regurgitation;
         joint hypermobility + skin changes; valve replacement often required by age 20-30;
         AD heterozygous COL1A2 → Osteogenesis Imperfecta spectrum NOT cvEDS)
320-patient aggregate cohort (8 x 40, seeds 1326-1333)
"""

import random

SEED_BASE = 1326

EDS_GENES = [
    # ── COL5A1 — Classical EDS Type 1 (AD) ──
    {
        "gene": "COL5A1",
        "protein": "Pro-alpha1(V) collagen chain",
        "alias": (
            "COL5A1; OMIM gene 120215; Classical EDS type 1 (cEDS-1) #130000; autosomal dominant; 9q34.3; "
            "1838 aa; ~184 kDa (pro-alpha chain); fibrillar collagen type V, most abundant form; "
            "triple helix formed with pro-alpha2(V) chain (COL5A2) → [alpha1(V)]2 alpha2(V) heterotrimer; "
            "collagen V regulates diameter of collagen I fibrils — essential for skin tensile strength; "
            "LOF haploinsufficiency: larger, irregular collagen I fibrils → skin weakness, fragility; "
            "~90% of classical EDS caused by COL5A1 heterozygous LOF (haploinsufficiency dominant mechanism); "
            "AD: 50% risk per child; de novo variants in ~50% of sporadic cEDS"
        ),
        "aa": "1838 aa",
        "kDa": "~184 kDa",
        "locus": "9q34.3",
        "omim_gene": 120215,
        "omim_disease": 130000,
        "inheritance": (
            "Autosomal dominant; haploinsufficiency (one LOF allele sufficient); "
            "50% risk to each child; ~50% of cEDS cases are de novo; "
            "missense variants near triple helix may act dominant negatively; "
            "penetrance virtually 100% but expressivity variable (intrafamilial variation)"
        ),
        "gene_class": (
            "COL5A1 encodes the pro-alpha1(V) chain of collagen type V, a quantitatively minor fibrillar collagen "
            "that acts as a 'nucleator' and 'regulator' of collagen I fibril diameter. "
            "FUNCTION: Collagen V heterotrimers [alpha1(V)]2alpha2(V) incorporate into the collagen I fibril "
            "core and control lateral growth by presenting charged surface residues that limit fibril accretion. "
            "Without adequate collagen V, collagen I fibrils grow to abnormally large, irregular diameters. "
            "PATHOPHYSIOLOGY: COL5A1 LOF → reduced collagen V → dysregulation of collagen I fibril assembly → "
            "coarse, irregular fibrils on electron microscopy ('cauliflower' fibrils) → "
            "reduced tensile strength of skin, ligaments, tendons. "
            "CLINICAL RESULT: skin hyperextensibility (Gorlin sign — patient can lick elbow tip), "
            "atrophic (cigarette-paper) scarring (major criterion), joint hypermobility, bruising. "
            "DIAGNOSIS: skin biopsy electron microscopy shows irregular fibril diameters; "
            "molecular testing (sequencing + MLPA for deletions) is confirmatory."
        ),
        "phenotype": (
            "Skin: velvety, soft, doughy texture; skin hyperextensibility ≥1.5 cm elbow (major); "
            "atrophic 'cigarette-paper' scarring at minor trauma sites (elbows, knees, shins) — MAJOR criterion; "
            "Gorlin sign (can touch lip to elbow) ~50%; "
            "Joints: generalized hypermobility (Beighton ≥5/9 adults, ≥6/9 children) — MAJOR criterion; "
            "recurrent joint subluxations/dislocations; sprains; "
            "Skin fragility: wide, atrophic scars; split-wound healing; pseudo-tumors (molluscoid); "
            "piezogenic papules (plantar fat herniations on standing); "
            "Blue sclera possible (~50%); mild mitral valve prolapse (20-30%); "
            "muscle hypotonia in childhood; delayed motor milestones"
        ),
        "key_hallmarks": [
            "Atrophic (cigarette-paper/tissue-paper) scarring at minor trauma sites — MAJOR CRITERION for cEDS",
            "Skin hyperextensibility ≥1.5 cm at elbow + velvety doughy texture — MAJOR CRITERION",
            "Gorlin sign (can lick own elbow) — present in ~50% cEDS; NOT seen in hypermobile EDS",
            "Beighton score ≥5/9 adults (≥6/9 children) for generalized joint hypermobility",
            "Cauliflower fibrils on electron microscopy skin biopsy — PATHOGNOMONIC fibril ultrastructure",
        ],
        "treatment_alerts": [
            "NO curative treatment: management is symptomatic and preventive",
            "Physiotherapy: joint-stabilizing exercises mandatory; AVOID high-impact sports, contact sports, weightlifting",
            "Wound care: paper tape over healing wounds; suture in layers (deep sutures + skin); leave sutures 2x longer than usual",
            "Vitamin C supplementation (500 mg/day): may reduce bruising; limited evidence",
            "Cardiac surveillance: echocardiography for mitral valve prolapse at diagnosis and every 3-5 years",
        ],
        "ddx": [
            "Hypermobile EDS (hEDS): skin hyperextensibility usually milder; atrophic scarring minimal; NO known gene (as of 2024 classification); Beighton ≥5",
            "Hypermobility Spectrum Disorder (HSD): does NOT meet EDS diagnostic criteria; Beighton ≥5 but no other major criteria",
            "Vascular EDS (COL3A1): thin translucent skin; acrogeria; risk of arterial/bowel rupture — very different clinical risk",
            "Marfan syndrome (FBN1): lens dislocation (superior-temporal); aortic root dilation; limited skin changes; Ghent criteria",
            "Kyphoscoliotic EDS (PLOD1/FKBP14): neonatal hypotonia; scoliosis; ocular fragility — more severe from birth",
        ],
        "seed": SEED_BASE + 0,
        "n_patients": 40,
        "age_range": (12, 45),
        "female_pct": 60,
    },
    # ── COL5A2 — Classical EDS Type 2 (AD) ──
    {
        "gene": "COL5A2",
        "protein": "Pro-alpha2(V) collagen chain",
        "alias": (
            "COL5A2; OMIM gene 120190; Classical EDS type 2 (cEDS-2) #130000; autosomal dominant; 2q32.2; "
            "1499 aa; ~149 kDa; alpha2 chain of collagen V heterotrimer [alpha1(V)]2alpha2(V); "
            "same fibril-diameter regulating function as COL5A1 partner chain; "
            "~10% of classical EDS (COL5A1 accounts for ~90%); "
            "AD haploinsufficiency; clinically indistinguishable from cEDS-1 without molecular testing"
        ),
        "aa": "1499 aa",
        "kDa": "~149 kDa",
        "locus": "2q32.2",
        "omim_gene": 120190,
        "omim_disease": 130000,
        "inheritance": (
            "Autosomal dominant; haploinsufficiency; 50% risk to each child; "
            "de novo variants occur; segregates with cEDS phenotype; "
            "COL5A1 and COL5A2 cannot be distinguished clinically — molecular testing required"
        ),
        "gene_class": (
            "COL5A2 encodes the pro-alpha2(V) chain of collagen type V, the heterotrimer partner to COL5A1. "
            "FUNCTION: [alpha1(V)]2alpha2(V) heterotrimers are incorporated into collagen I fibrils in dermis and tendons. "
            "The COL5A2 chain contributes to the helical stability and the charged surface of collagen V needed for "
            "fibril nucleation control. "
            "PATHOPHYSIOLOGY: COL5A2 LOF → reduced collagen V → same dysregulated fibril assembly as COL5A1 LOF → "
            "skin fragility, atrophic scarring, joint laxity. "
            "DISTINGUISHING FROM COL5A1: Cannot be distinguished clinically — "
            "molecular sequencing + MLPA is required to identify which alpha-chain gene is mutated. "
            "Collagen biochemistry on skin biopsy may show identical collagen V reduction in both. "
            "PREVALENCE: COL5A2 accounts for ~10% of classical EDS; "
            "COL5A1 accounts for ~90% → test COL5A1 FIRST."
        ),
        "phenotype": (
            "Identical to cEDS-1 (COL5A1): skin hyperextensibility, atrophic scarring, joint hypermobility; "
            "Gorlin sign, piezogenic papules, molluscoid pseudo-tumors; "
            "bruising tendency; poor wound healing; "
            "same Beighton criteria apply (≥5/9); "
            "soft/doughy/velvety skin texture; "
            "No distinguishing clinical features from COL5A1 — molecular testing MANDATORY to differentiate"
        ),
        "key_hallmarks": [
            "Clinically IDENTICAL to cEDS-1 (COL5A1) — molecular testing is the only way to distinguish",
            "Atrophic cigarette-paper scarring + skin hyperextensibility: SAME major criteria as COL5A1",
            "COL5A2 is the MINORITY cause (~10%) — test COL5A1 first in clinical panels",
            "Collagen biochemistry: Type V collagen reduction on SDS-PAGE of cultured fibroblasts — same finding as COL5A1",
            "Cauliflower fibril ultrastructure on EM: identical to COL5A1 cEDS",
        ],
        "treatment_alerts": [
            "Identical management to COL5A1 cEDS — no gene-specific differences",
            "Wound closure: subcuticular sutures + skin support tape; leave sutures 2-3x longer than standard",
            "Joint protection: physiotherapy; padding over bony prominences; bracing for hypermobile joints",
            "Avoid: contact sports, gymnastic overextension, repetitive high-impact loading",
            "Annual cardiac echo for MVP assessment; BP monitoring",
        ],
        "ddx": [
            "COL5A1 cEDS: clinically identical — molecular testing required; COL5A1 FIRST (90% of cEDS)",
            "hEDS: Beighton ≥5 but atrophic scarring absent/minimal; no molecular diagnosis as of 2024 classification",
            "EDS with marfanoid habitus: consider FBN1 sequencing if tall stature, arachnodactyly, aortic root dilation",
            "Cutis laxa (ELN, FBLN5, ATP6AP2): skin hangs loosely but is NOT hyperextensible; different feel",
        ],
        "seed": SEED_BASE + 1,
        "n_patients": 40,
        "age_range": (10, 50),
        "female_pct": 58,
    },
    # ── COL3A1 — Vascular EDS (AD, MOST LETHAL) ──
    {
        "gene": "COL3A1",
        "protein": "Pro-alpha1(III) collagen chain",
        "alias": (
            "COL3A1; OMIM gene 120180; Vascular EDS (vEDS) #130050; autosomal dominant; 2q32.2; "
            "1466 aa; ~139 kDa pro-alpha1 chain; collagen III triple helix homotrimer [alpha1(III)]3; "
            "MOST LETHAL connective tissue disorder: median life expectancy ~48 years; "
            "50% of patients experience first major complication (arterial rupture) by age 40; "
            "spontaneous hollow viscus rupture — bowel, uterus, arterial rupture WITHOUT trauma; "
            "NO elective surgery; NO colonoscopy; NO angiography; celiprolol ONLY RCT-proven therapy"
        ),
        "aa": "1466 aa",
        "kDa": "~139 kDa",
        "locus": "2q32.2",
        "omim_gene": 120180,
        "omim_disease": 130050,
        "inheritance": (
            "Autosomal dominant; most are missense glycine substitutions (dominant negative) → "
            "mutant alpha1(III) chains poison all trimers containing them (25% normal → 0% functional trimers); "
            "haploinsufficiency variants (LOF) → MILDER phenotype (half-normal collagen III, remaining normal); "
            "de novo variants: ~50% of vEDS; "
            "50% risk to each child of affected parent"
        ),
        "gene_class": (
            "COL3A1 encodes pro-alpha1(III) collagen, which forms the homotrimer type III collagen — "
            "the predominant collagen in large blood vessel walls, bowel wall, uterus, skin, and fetal tissues. "
            "FUNCTION: Collagen III provides tensile strength and elasticity to hollow organs. "
            "PATHOPHYSIOLOGY: Missense glycine substitutions in the Gly-X-Y repeat → "
            "mutant chains disrupt triple helix folding → intracellular retention of misfolded procollagen → "
            "secretion failure + dominant negative poisoning of assembled trimers → "
            "severely weakened vessel/bowel/uterine walls. "
            "CONSEQUENCE: Spontaneous arterial dissection and rupture, bowel perforation, uterine rupture — "
            "typically WITHOUT preceding trauma or warning. "
            "vEDS CHARACTERISTIC FEATURES: thin translucent skin (subcutaneous veins visible), "
            "acrogeria (prematurely aged hands/feet), characteristic facies (large eyes, thin nose, small chin), "
            "easy bruising. "
            "CRITICAL: Joint hypermobility is MILD or ABSENT in vEDS — "
            "clinician must NOT be reassured by absence of joint laxity."
        ),
        "phenotype": (
            "MAJOR CRITERIA (≥1 required for vEDS diagnosis): "
            "family history of vEDS + compatible COL3A1 variant; "
            "arterial rupture at young age; "
            "spontaneous sigmoid colon perforation; "
            "uterine rupture during 3rd trimester; "
            "carotid-cavernous sinus fistula (CCF) without trauma; "
            "MINOR CRITERIA: thin translucent skin (veins visible); "
            "acrogeria; easy bruising; characteristic facies; "
            "clubfoot; early-onset varicose veins; "
            "pneumothorax/hemopneumothorax; small joint hypermobility (fingers); "
            "LIFE-THREATENING EVENTS: mesenteric artery rupture, hepatic/splenic rupture, "
            "vertebral artery dissection, celiac/SMA rupture"
        ),
        "key_hallmarks": [
            "THIN TRANSLUCENT SKIN (veins visible) + ACROGERIA + characteristic facies = vEDS phenotype — PATHOGNOMONIC triad",
            "SPONTANEOUS HOLLOW VISCUS RUPTURE: bowel perforation / arterial rupture WITHOUT trauma = vEDS until proven otherwise",
            "NO ELECTIVE SURGERY — perioperative mortality in vEDS is HIGH; bowel friability → anastomosis failure",
            "NO COLONOSCOPY — bowel perforation risk; if essential, capsule endoscopy preferred",
            "NO ANGIOGRAPHY — arterial manipulation risks acute rupture; CT angiography ONLY, no catheter angiography",
        ],
        "treatment_alerts": [
            "CELIPROLOL: ONLY RCT-proven therapy for vEDS (Ong 2010 NEJM): 200-400 mg/day reduces arterial events ~36%",
            "NO elective surgery: surgical morbidity/mortality extremely high; consult vascular surgery ONLY for life-threatening events",
            "Surgical emergency (rupture): damage control surgery; use pledgeted sutures; avoid extensive dissection",
            "PREGNANCY: HIGH RISK — uterine rupture risk in 3rd trimester + postpartum; deliver by elective C-section at 34-36 weeks; "
            "counsel patient BEFORE conception; maternal mortality 10-12% per pregnancy",
            "Blood pressure: tight BP control <120/80 mmHg; celiprolol + if needed ACE inhibitor/ARB",
            "Emergency card: patient carries vEDS emergency card with NO angiography / NO colonoscopy instructions",
        ],
        "ddx": [
            "Classical EDS (COL5A1/2): skin hyperextensibility + atrophic scarring; arterial events rare",
            "Loeys-Dietz Syndrome (TGFBR1/2, SMAD3): bifid uvula PATHOGNOMONIC; aortic root dilation; repair at 4.0 cm",
            "Marfan syndrome (FBN1): lens dislocation superior-temporal; aortic root dilation; repair 4.5 cm; no bowel/visceral rupture",
            "Spontaneous coronary artery dissection (SCAD): may be vEDS; test COL3A1 in all young women with SCAD",
            "Inflammatory bowel disease: sigmoid perforation must trigger COL3A1 testing in young patients without IBD history",
        ],
        "seed": SEED_BASE + 2,
        "n_patients": 40,
        "age_range": (20, 55),
        "female_pct": 55,
    },
    # ── TNXB — Classical-like EDS (AR) ──
    {
        "gene": "TNXB",
        "protein": "Tenascin-X",
        "alias": (
            "TNXB; OMIM gene 600985; Classical-like EDS (clEDS) #606408; autosomal recessive (biallelic LOF = clEDS); "
            "HAPLOINSUFFICIENCY (one LOF allele only) = Hypermobility Spectrum Disorder (HSD) NOT EDS; "
            "4268 aa; ~484 kDa secreted ECM glycoprotein; hexabrachion structure; "
            "6p21.3; immediate neighbor of CYP21A2 — contiguous deletion → clEDS + CAH (21-hydroxylase deficiency); "
            "tenascin-X stabilizes collagen I fibrils by inhibiting fibrillar collagen turnover; "
            "biallelic LOF → loss of fibril stabilization → clEDS phenotype"
        ),
        "aa": "4268 aa",
        "kDa": "~484 kDa",
        "locus": "6p21.3",
        "omim_gene": 600985,
        "omim_disease": 606408,
        "inheritance": (
            "Autosomal recessive; biallelic LOF = clEDS; "
            "HAPLOINSUFFICIENCY (one LOF allele) = Hypermobility Spectrum Disorder (HSD) — "
            "does NOT meet EDS diagnostic criteria; "
            "CONTIGUOUS DELETION (TNXB + CYP21A2): causes clEDS + congenital adrenal hyperplasia (21-OH deficiency) simultaneously — "
            "MANDATORY adrenal screen in ALL TNXB patients"
        ),
        "gene_class": (
            "TNXB encodes Tenascin-X (TNX), a large extracellular matrix glycoprotein with a hexabrachion structure, "
            "consisting of EGF-like repeats, fibronectin type III domains, and a fibrinogen-like C-terminal domain. "
            "FUNCTION: TNX is expressed in dermis, fascia, skeletal muscle, and joint capsules. "
            "It binds collagen I and decorin, stabilizing collagen fibril spacing and inhibiting excessive fibrillar remodeling. "
            "PATHOPHYSIOLOGY: Biallelic TNXB LOF → absence of TNX → unstable collagen I fibrils → "
            "joint hypermobility + skin fragility WITHOUT the atrophic scarring seen in COL5A1/A2 cEDS. "
            "CRITICAL DISTINCTION: clEDS LACKS the major skin criterion (atrophic scarring) of cEDS; "
            "instead: skin hyperextensibility + easy bruising + joint hypermobility predominate. "
            "GENE LOCATION ALERT: TNXB and CYP21A2 are adjacent on chromosome 6p21.3. "
            "Large deletions spanning both genes → clEDS + CAH21 (salt-wasting or simple-virilizing). "
            "EVERY TNXB patient must have adrenal function tested (17-OHP, cortisol stimulation)."
        ),
        "phenotype": (
            "Joint hypermobility (generalized, Beighton ≥5/9): the dominant feature; "
            "Skin: hyperextensible (≥1.5 cm elbow) + easy bruising BUT atrophic scarring ABSENT or minimal "
            "(key distinction from cEDS); skin texture velvety/soft; "
            "No atrophic (cigarette-paper) scarring — this is the KEY difference from COL5A1/A2; "
            "Muscle weakness (myopathy-like) in some; proprioception impaired; "
            "Chronic pain; fatigue; dysautonomia features (POTS common); "
            "If contiguous CYP21A2 deletion: adrenal insufficiency (AI) + salt-wasting crisis risk"
        ),
        "key_hallmarks": [
            "Joint hypermobility + skin hyperextensibility WITHOUT atrophic scarring — KEY distinction from classical EDS (COL5A1/2)",
            "TNXB HAPLOINSUFFICIENCY = HSD NOT EDS — biallelic (AR) required for clEDS diagnosis",
            "CONTIGUOUS DELETION 6p21.3 (TNXB + CYP21A2): clEDS + Congenital Adrenal Hyperplasia — screen ALL TNXB patients for AI",
            "Adrenal crisis risk if contiguous CYP21A2 deletion: every TNXB clEDS patient needs adrenal function testing (17-OHP + SST)",
            "TNX-deficient patients may have POTS and dysautonomia — tilt-table test if symptomatic",
        ],
        "treatment_alerts": [
            "Adrenal function: 17-OHP and synacthen stimulation test MANDATORY in all TNXB biallelic patients at diagnosis",
            "If CYP21A2 contiguous deletion confirmed: hydrocortisone replacement; stress dosing protocol; emergency identification",
            "Joint protection: physiotherapy, aquatherapy; avoid hyperextension maneuvers; proprioceptive training",
            "POTS management: increased salt + fluid intake; compression garments; fludrocortisone if needed",
            "Same wound care as cEDS: layered closure, prolonged suture retention, protective taping",
        ],
        "ddx": [
            "Classical EDS (COL5A1/2): atrophic cigarette-paper scarring IS present — absent in clEDS/TNXB",
            "Hypermobile EDS (hEDS): no known gene; TNXB haploinsufficiency accounts for some HSD cases",
            "Combined clEDS + CAH: check 17-OHP in all clEDS patients — TNXB+CYP21A2 contiguous deletion",
            "Myopathic EDS (COL12A1): muscle weakness + hypotonia predominate; foot deformities; molecular testing",
        ],
        "seed": SEED_BASE + 3,
        "n_patients": 40,
        "age_range": (8, 40),
        "female_pct": 62,
    },
    # ── ADAMTS2 — Dermatosparaxis EDS (AR, RAREST) ──
    {
        "gene": "ADAMTS2",
        "protein": "ADAM metallopeptidase with thrombospondin type 1 motif 2 (procollagen N-proteinase)",
        "alias": (
            "ADAMTS2; OMIM gene 604539; Dermatosparaxis EDS (dEDS) #225410; autosomal recessive; 5q35.3; "
            "1211 aa; ~134 kDa secreted metalloprotease; "
            "cleaves N-terminal propeptide from procollagens I, II, III — essential for fibril assembly; "
            "AR: dEDS is the RAREST EDS subtype (< 50 cases worldwide); "
            "SAGGING REDUNDANT LAX SKIN AT BIRTH — skin can be pulled far from body without tearing; "
            "procollagen N-propeptide accumulation in skin → PATHOGNOMONIC on biopsy"
        ),
        "aa": "1211 aa",
        "kDa": "~134 kDa",
        "locus": "5q35.3",
        "omim_gene": 604539,
        "omim_disease": 225410,
        "inheritance": (
            "Autosomal recessive; biallelic LOF required; "
            "extremely rare worldwide (<50 published cases); "
            "consanguinity increases risk; "
            "heterozygous carriers: asymptomatic (one functional ADAMTS2 allele is sufficient)"
        ),
        "gene_class": (
            "ADAMTS2 (procollagen N-proteinase, pNP1) is a secreted zinc-dependent metalloprotease that cleaves "
            "the N-terminal propeptide of fibrillar procollagens I, II, and III — a mandatory processing step "
            "before fibril assembly can occur. "
            "FUNCTION: Procollagen I/II/III are secreted as soluble precursors; "
            "N-propeptide and C-propeptide must be cleaved for mature collagen molecules to pack into fibrils. "
            "ADAMTS2 performs N-propeptide cleavage in skin, tendon, and bone. "
            "PATHOPHYSIOLOGY: ADAMTS2 LOF → N-propeptides are NOT cleaved → "
            "pNI (procollagen I N-propeptide) retained on mature collagen molecules → "
            "fibrils cannot pack tightly → extremely lax, fragile, sagging skin. "
            "PATHOGNOMONIC BIOPSY FINDING: accumulation of procollagen I N-terminal propeptide "
            "immunostaining on skin biopsy (dermis and fibrils show pNI retention). "
            "DISTINGUISHING FEATURE: skin is not just extensible (as in cEDS) but hangs in loose, "
            "redundant folds — the sagging quality is unique to dEDS."
        ),
        "phenotype": (
            "NEONATAL: extreme skin laxity apparent at birth; skin hangs in folds (saggy/redundant — PATHOGNOMONIC); "
            "may resemble prematurely aged skin; hernias (umbilical, inguinal); "
            "bruising EXTREME — any minimal trauma; "
            "JOINTS: joint hypermobility; recurrent dislocations; "
            "ADDITIONAL: short stature; delayed motor development; "
            "facial dysmorphism (widely-spaced eyes, epicanthic folds, blue sclera); "
            "SKIN: skin can be pulled far from body surface WITHOUT tearing (maximally extensible); "
            "healing wounds show wide, atrophic scars (more extreme than cEDS); "
            "scleral fragility possible (globe rupture risk lower than kEDS but monitor)"
        ),
        "key_hallmarks": [
            "SAGGING REDUNDANT SKIN AT BIRTH — skin hangs in folds, pulled far from body: PATHOGNOMONIC for dEDS",
            "PROCOLLAGEN N-PROPEPTIDE ACCUMULATION on skin biopsy immunostaining: DIAGNOSTIC CONFIRMATION",
            "EXTREME bruising from minimal trauma — more severe than cEDS or any other EDS subtype",
            "RAREST EDS subtype (< 50 cases worldwide): consider dEDS in any child with extreme congenital skin laxity",
            "Umbilical/inguinal hernias in neonatal period: connective tissue failure across all fibrillar collagen-dependent structures",
        ],
        "treatment_alerts": [
            "No curative treatment; preventive and supportive care only",
            "Wound care: EXTREME care required — skin friability limits standard surgical technique; pledgeted sutures; "
            "skin support with non-adherent tape and silicone gel sheets post-healing",
            "Hernia repair: refer to specialist center with connective tissue disorder experience — mesh reinforcement",
            "Joint protection: physiotherapy; bracing; avoid contact sports",
            "Ophthalmology: annual slit-lamp exam for scleral fragility; protective eyewear during physical activity",
        ],
        "ddx": [
            "Classical EDS (COL5A1/2): skin hyperextensible but NOT sagging at birth; atrophic scarring present; propeptide normal on biopsy",
            "Cutis laxa (ELN/FBLN5/ATP6AP2): skin HANGS loosely but is not maximally extensible; procollagen propeptide normal",
            "Wrinkly skin syndrome/PGAP2: similar appearance; different mechanism; chromosomal microarray",
            "Progeroid EDS variants (B4GALT7/SLC39A13): skeletal dysplasia + short stature; proteoglycan synthesis defects",
        ],
        "seed": SEED_BASE + 4,
        "n_patients": 40,
        "age_range": (0, 30),
        "female_pct": 50,
    },
    # ── PLOD1 — Kyphoscoliotic EDS Type 1 (AR) ──
    {
        "gene": "PLOD1",
        "protein": "Procollagen-lysine,2-oxoglutarate 5-dioxygenase 1 (Lysyl hydroxylase 1 / LH1)",
        "alias": (
            "PLOD1; OMIM gene 153454; Kyphoscoliotic EDS type 1 (kEDS-1) #225400; autosomal recessive; 1p36.3; "
            "727 aa; ~85 kDa; endoplasmic reticulum enzyme; "
            "URINE LP:HP RATIO (lysylpyridinoline:hydroxylysylpyridinoline) >0.09 PATHOGNOMONIC; "
            "hydroxylates lysine residues in collagen triple helix → cross-links between collagen fibrils; "
            "LOF → under-hydroxylated lysines → collagen fibril cross-linking deficiency → "
            "severe neonatal hypotonia + progressive kyphoscoliosis + ocular fragility"
        ),
        "aa": "727 aa",
        "kDa": "~85 kDa",
        "locus": "1p36.3",
        "omim_gene": 153454,
        "omim_disease": 225400,
        "inheritance": (
            "Autosomal recessive; biallelic LOF required; "
            "consanguinity common in affected families; "
            "heterozygous carriers clinically normal; "
            "de novo variants uncommon — family history usually positive"
        ),
        "gene_class": (
            "PLOD1 encodes lysyl hydroxylase 1 (LH1), an ER-resident enzyme that hydroxylates specific lysine residues "
            "in the Gly-X-Y repeat regions of collagen alpha chains BEFORE triple helix formation. "
            "FUNCTION: Hydroxylysine residues serve as attachment points for galactose and glucose in collagen glycosylation, "
            "AND as the substrates for lysyl oxidase — the enzyme forming the pyridinoline cross-links that provide "
            "tensile strength to collagen fibrils. "
            "PATHOPHYSIOLOGY: PLOD1 LOF → lysine residues remain unhydroxylated → "
            "fibrils have deficient hydroxylysine-based cross-links → reduced fibril tensile strength → "
            "neonatal hypotonia, progressive scoliosis, joint laxity, ocular fragility. "
            "BIOMARKER: The urinary LP:HP ratio (lysylpyridinoline:hydroxylysylpyridinoline) reflects the "
            "ratio of non-hydroxylated to hydroxylated cross-links. In kEDS-1: LP greatly elevated, HP reduced → "
            "LP:HP ratio >0.09 (normal <0.04). This urine test IS PATHOGNOMONIC and should be ordered FIRST "
            "before molecular testing — cheap, non-invasive, highly specific for PLOD1 deficiency."
        ),
        "phenotype": (
            "NEONATAL (MOST SEVERE at birth): profound muscular hypotonia at birth — floppy infant; "
            "delayed motor milestones (sitting, walking significantly delayed); "
            "Progressive kyphoscoliosis (scoliosis beginning in infancy, rapidly progressive); "
            "Joint hypermobility (generalized); "
            "OCULAR: scleral fragility → globe rupture risk from minor trauma (PATHOGNOMONIC for kEDS vs other EDS types); "
            "myopia common (>50%); keratoconus; "
            "Skin: velvet-soft; moderate hyperextensibility; atrophic scarring possible; "
            "Marfanoid habitus possible (tall, arachnodactyly); "
            "Mitral valve prolapse (~25%); aortic dilation possible"
        ),
        "key_hallmarks": [
            "URINE LP:HP RATIO >0.09 PATHOGNOMONIC — order this FIRST; cheap, non-invasive, highly specific for PLOD1 kEDS-1",
            "OCULAR FRAGILITY — globe rupture risk: protective eyewear MANDATORY; no contact sports; ophthalmology at diagnosis",
            "NEONATAL HYPOTONIA + PROGRESSIVE KYPHOSCOLIOSIS from birth = kEDS phenotype (PLOD1 or FKBP14)",
            "Scoliosis progression: monitor Cobb angle every 6 months in childhood; spinal fusion when Cobb >50 degrees",
            "Ascorbic acid (vitamin C) 2-4 g/day: may partially improve collagen hydroxylation; limit evidence but recommended",
        ],
        "treatment_alerts": [
            "Ophthalmology referral AT DIAGNOSIS: annual slit-lamp + IOP + corneal pachymetry; protective polycarbonate eyewear ALWAYS",
            "NO boxing, martial arts, ball sports, contact activities — globe rupture risk is life-altering",
            "Ascorbic acid (vitamin C) supplementation: 2-4 g/day orally; evidence limited but biologically plausible",
            "Scoliosis bracing: start early when Cobb <30 degrees to slow progression; spinal fusion at Cobb ≥45-50 degrees",
            "Physiotherapy: strengthening program (core stability, paraspinal muscles); hydrotherapy beneficial for hypotonia",
        ],
        "ddx": [
            "FKBP14 kEDS type 2: SAME clinical phenotype but urine LP:HP ratio NORMAL; congenital hearing loss in 50%; muscle biopsy may show myopathic changes",
            "Marfan syndrome (FBN1): tall stature + arachnodactyly + aortic root dilation; different skin/joint profile; FBN1 sequencing",
            "Congenital muscular dystrophy: hypotonia + weakness; different pathology; CK elevated; EMG/biopsy",
            "Noonan syndrome (PTPN11): facial dysmorphism; short stature; cardiac; PTPN11 sequencing",
            "SLC39A13 kEDS-3: spondylodysplastic features; short stature; SEM on skin biopsy",
        ],
        "seed": SEED_BASE + 5,
        "n_patients": 40,
        "age_range": (0, 35),
        "female_pct": 50,
    },
    # ── FKBP14 — Kyphoscoliotic EDS Type 2 (AR) ──
    {
        "gene": "FKBP14",
        "protein": "FKBP prolyl isomerase 14 (FKBP22)",
        "alias": (
            "FKBP14; OMIM gene 614505; Kyphoscoliotic EDS type 2 (kEDS-2) #614557; autosomal recessive; 7p14.3; "
            "150 aa; ~16 kDa; ER-resident peptidyl-prolyl isomerase (PPIase); "
            "assists collagen folding in the ER lumen alongside FKBP65 (FKBP10); "
            "kEDS-2 is CLINICALLY VERY SIMILAR to kEDS-1 (PLOD1) but key distinctions exist; "
            "URINE LP:HP RATIO NORMAL (distinguishes from kEDS-1); "
            "CONGENITAL HEARING LOSS present in ~50% kEDS-2 (not seen in kEDS-1)"
        ),
        "aa": "150 aa",
        "kDa": "~16 kDa",
        "locus": "7p14.3",
        "omim_gene": 614505,
        "omim_disease": 614557,
        "inheritance": (
            "Autosomal recessive; biallelic LOF; "
            "most reported cases carry the founder variant (c.362dupC) causing frameshift; "
            "consanguinity present in many published families; "
            "rare — far fewer cases than kEDS-1 (PLOD1)"
        ),
        "gene_class": (
            "FKBP14 (also called FKBP22) encodes a small ER-resident peptidyl-prolyl cis/trans isomerase (PPIase) "
            "of the FKBP (FK506-binding protein) family. "
            "FUNCTION: FKBP14 catalyzes the cis-trans isomerization of X-Pro peptide bonds during collagen folding, "
            "facilitating the rate-limiting step in triple helix formation. "
            "It also assists in maturation of collagen XI and other fibrillar collagens. "
            "PATHOPHYSIOLOGY: FKBP14 LOF → impaired ER collagen folding → "
            "reduced secretion and assembly of collagen fibrils → "
            "connective tissue weakness phenotypically overlapping kEDS-1. "
            "KEY DISTINGUISHING FEATURES vs kEDS-1: "
            "(1) Normal urine LP:HP ratio (LH1 enzyme is INTACT; no lysine hydroxylation defect); "
            "(2) Congenital sensorineural hearing loss (~50%); "
            "(3) Myopathy features more prominent; "
            "(4) Allelism with musculocontractural EDS features in some variants. "
            "MOLECULAR CONFIRMATION: FKBP14 sequencing required (LP:HP ratio alone cannot make diagnosis)."
        ),
        "phenotype": (
            "OVERLAPPING with kEDS-1: neonatal hypotonia (floppy infant); "
            "progressive kyphoscoliosis from infancy; "
            "generalized joint hypermobility; "
            "velvety skin; moderate hyperextensibility; "
            "DISTINCTIONS FROM kEDS-1: "
            "Hearing loss: congenital SNHL in ~50% — audiogram mandatory at diagnosis; "
            "Myopathy: muscle weakness often more pronounced than kEDS-1; "
            "Ocular: scleral fragility LESS prominent than kEDS-1 (globe rupture risk lower); "
            "Musculocontractural features: talipes, contractures at birth in some; "
            "Normal LP:HP ratio — this is the KEY to distinguishing from kEDS-1 biochemically"
        ),
        "key_hallmarks": [
            "NORMAL URINE LP:HP RATIO — KEY DISTINGUISHER from kEDS-1 (PLOD1): order urine LP:HP first; if normal, test FKBP14",
            "CONGENITAL SENSORINEURAL HEARING LOSS (~50%) — MANDATORY audiogram at diagnosis; not seen in kEDS-1 (PLOD1)",
            "Neonatal hypotonia + kyphoscoliosis from birth: same as kEDS-1 clinically but gene is different",
            "Myopathy: muscle biopsy may show myopathic changes; CK can be mildly elevated",
            "Ocular fragility: LESS severe than kEDS-1 but still requires ophthalmology assessment; protective eyewear recommended",
        ],
        "treatment_alerts": [
            "Audiological assessment AT DIAGNOSIS: formal audiogram; cochlear implant evaluation if SNHL profound",
            "Scoliosis management: same protocol as kEDS-1; Cobb angle 6-monthly; fusion when ≥45-50 degrees",
            "Physiotherapy: focus on muscle strengthening (myopathy feature more prominent than kEDS-1)",
            "Ophthalmology: annual assessment; protective eyewear (lower risk than kEDS-1 but non-zero)",
            "NO ascorbic acid benefit (LH1 enzyme is INTACT; vitamin C cannot improve LP:HP — different from kEDS-1)",
        ],
        "ddx": [
            "PLOD1 kEDS-1: LP:HP ratio elevated (>0.09); otherwise clinically overlapping; no hearing loss; globe rupture risk higher",
            "COL12A1 myopathic EDS: muscle weakness + hypotonia; collagen XII deficiency; normal LP:HP; different gene",
            "Congenital muscular dystrophy: elevated CK; EMG shows myopathic pattern; biopsy; no EDS-specific skin changes",
            "Musculocontractural EDS (CHST14/DSE): contractures at birth; additional features; different gene panel",
        ],
        "seed": SEED_BASE + 6,
        "n_patients": 40,
        "age_range": (0, 35),
        "female_pct": 50,
    },
    # ── COL1A2 — Cardiac-valvular EDS (AR biallelic) ──
    {
        "gene": "COL1A2",
        "protein": "Pro-alpha2(I) collagen chain",
        "alias": (
            "COL1A2; OMIM gene 120160; Cardiac-valvular EDS (cvEDS) #225320; "
            "autosomal recessive (biallelic null = cvEDS); "
            "HETEROZYGOUS COL1A2 variants → Osteogenesis Imperfecta (OI) spectrum — NOT cvEDS; "
            "7q21.3; 1366 aa; ~129 kDa pro-alpha2 chain; "
            "collagen I heterotrimer [alpha1(I)]2 alpha2(I) — most abundant collagen in body; "
            "SEVERE PROGRESSIVE CARDIAC VALVE DISEASE PATHOGNOMONIC — aortic regurgitation ± mitral regurgitation; "
            "valve replacement often needed in 2nd-3rd decade; extremely rare (<30 cases worldwide)"
        ),
        "aa": "1366 aa",
        "kDa": "~129 kDa",
        "locus": "7q21.3",
        "omim_gene": 120160,
        "omim_disease": 225320,
        "inheritance": (
            "Autosomal recessive for cvEDS: BIALLELIC null/LOF variants required; "
            "heterozygous COL1A2 LOF or Gly substitutions → OI spectrum (NOT cvEDS); "
            "biallelic null → no pro-alpha2(I) chain produced → "
            "homotrimers [alpha1(I)]3 form instead (structurally altered collagen I) → cvEDS; "
            "extremely rare — < 30 worldwide cases reported"
        ),
        "gene_class": (
            "COL1A2 encodes the pro-alpha2(I) chain of type I collagen, the most abundant collagen in the body. "
            "FUNCTION: Collagen I normally forms a heterotrimer [alpha1(I)]2alpha2(I). "
            "The alpha2(I) chain contributes to fibril flexibility and binding to proteoglycans and fibronectin. "
            "PATHOPHYSIOLOGY OF cvEDS: BIALLELIC COL1A2 null → no alpha2(I) chains → "
            "remaining alpha1(I) chains form homotrimers [alpha1(I)]3 → "
            "structurally abnormal collagen I lacking the alpha2(I) contribution → "
            "severe connective tissue weakness specifically affecting cardiac valves. "
            "UNIQUE CARDIAC PHENOTYPE: Collagen I is critical for valve leaflet integrity; "
            "absence of normal alpha2(I) → severe early-onset valve regurgitation. "
            "CONTRAST WITH OI: AD heterozygous Gly substitutions in COL1A2 → OI (bone fragility, blue sclera); "
            "AR biallelic null → cvEDS (valve disease > bone fragility); different disease with same gene."
        ),
        "phenotype": (
            "CARDIAC (DOMINANT FEATURE): severe progressive aortic valve regurgitation — "
            "may require valve replacement by age 20-30; "
            "mitral valve regurgitation (50-70%); "
            "cardiac failure if untreated; "
            "echocardiography at diagnosis mandatory + 6-12 monthly surveillance; "
            "EDS FEATURES: joint hypermobility (generalized); skin hyperextensibility; "
            "easy bruising; atrophic scarring (mild to moderate); "
            "SHORT STATURE common; "
            "Hernia: umbilical/inguinal hernias (connective tissue weakness); "
            "Blue sclera possible; "
            "Severe skin and joint manifestations may be milder than in cEDS"
        ),
        "key_hallmarks": [
            "SEVERE PROGRESSIVE CARDIAC VALVE DISEASE (aortic ± mitral regurgitation) in young person + EDS features = THINK cvEDS (COL1A2 biallelic)",
            "EARLY VALVE REPLACEMENT often required in 2nd-3rd decade — ANNUAL echocardiography is NON-NEGOTIABLE",
            "COL1A2 BIALLELIC NULL → cvEDS; COL1A2 HETEROZYGOUS Gly substitution → OI (bone fragility) — very different diseases",
            "EXTREMELY RARE: fewer than 30 cases worldwide; consider in young patients with unexplained severe valve disease + EDS features",
            "Do NOT miss cardiac phenotype: valve failure in cvEDS is progressive and life-threatening if undetected",
        ],
        "treatment_alerts": [
            "CARDIAC SURVEILLANCE: echocardiography at diagnosis then 6-12 monthly; cardiology co-management from diagnosis",
            "Valve replacement: timing based on standard valvular regurgitation guidelines; early referral to cardiac surgery",
            "Infective endocarditis prophylaxis: per current guidelines for significant regurgitation + prior valve surgery",
            "Hemodynamic optimization: ACE inhibitor/ARB for afterload reduction in aortic regurgitation; beta-blocker for heart rate",
            "Pregnancy: HIGH RISK — cardiac assessment before conception; cardiology + obstetric co-management throughout",
        ],
        "ddx": [
            "COL1A2 AD heterozygous (OI): bone fragility + blue sclera + short stature + Wormian bones — different from biallelic cvEDS",
            "Marfan syndrome (FBN1): aortic root dilation + lens dislocation + tall stature; valve regurgitation secondary to dilation",
            "Bicuspid aortic valve: sporadic or familial (NOTCH1); no EDS features",
            "Classical EDS (COL5A1/2): valve involvement mild/MVP only; severe regurgitation should prompt COL1A2 testing",
            "Rheumatic fever: history of strep throat + fever; ASO titer elevated; different epidemiology",
        ],
        "seed": SEED_BASE + 7,
        "n_patients": 40,
        "age_range": (0, 40),
        "female_pct": 50,
    },
]

# ──────────────────────────────────────────────────────────────────────
# Patient cohort generation
# ──────────────────────────────────────────────────────────────────────

def _generate_patient(gene_def, idx):
    rng = random.Random(gene_def["seed"] * 10000 + idx)
    g = gene_def["gene"]

    age_lo, age_hi = gene_def["age_range"]
    age = rng.randint(age_lo, age_hi)
    sex = "F" if rng.random() < gene_def["female_pct"] / 100 else "M"

    beighton = rng.randint(5, 9)

    skin_ext_cm = round(rng.uniform(1.2, 4.5), 1) if g not in ["COL3A1", "COL1A2"] else round(rng.uniform(0.5, 1.5), 1)

    # Gene-specific features
    atrophic_scar = g in ["COL5A1", "COL5A2", "ADAMTS2", "COL1A2"]
    vEDS_event = g == "COL3A1" and rng.random() < 0.45  # arterial/bowel event in 45%
    ocular_fragility = g in ["PLOD1", "FKBP14"] and rng.random() < 0.60
    hearing_loss = g == "FKBP14" and rng.random() < 0.50
    urine_lphp = round(rng.uniform(0.10, 0.25), 3) if g == "PLOD1" else round(rng.uniform(0.01, 0.04), 3)
    kyphoscoliosis = g in ["PLOD1", "FKBP14"] and rng.random() < 0.85
    neonatal_hypotonia = g in ["PLOD1", "FKBP14"] and rng.random() < 0.90
    saggy_skin = g == "ADAMTS2"
    cardiac_valve = g == "COL1A2" and rng.random() < 0.80
    adrenal_insufficiency = g == "TNXB" and rng.random() < 0.15  # contiguous deletion subset

    dx_delay_yrs = rng.randint(1, 15)

    return {
        "patient_id": f"{g}-{idx:03d}",
        "gene": g,
        "age": age,
        "sex": sex,
        "beighton": beighton,
        "skin_extensibility_cm": skin_ext_cm,
        "atrophic_scarring": atrophic_scar,
        "vEDS_major_event": vEDS_event,
        "ocular_fragility": ocular_fragility,
        "hearing_loss": hearing_loss,
        "urine_lphp_ratio": urine_lphp,
        "kyphoscoliosis": kyphoscoliosis,
        "neonatal_hypotonia": neonatal_hypotonia,
        "sagging_skin": saggy_skin,
        "cardiac_valve_disease": cardiac_valve,
        "adrenal_insufficiency": adrenal_insufficiency,
        "dx_delay_years": dx_delay_yrs,
    }


def _build_cohort(gene_def):
    return [_generate_patient(gene_def, i) for i in range(gene_def["n_patients"])]


_ALL_COHORTS = {g["gene"]: _build_cohort(g) for g in EDS_GENES}


# ──────────────────────────────────────────────────────────────────────
# API functions
# ──────────────────────────────────────────────────────────────────────

def get_overview():
    n = sum(len(v) for v in _ALL_COHORTS.values())

    vEDS_events = sum(1 for p in _ALL_COHORTS["COL3A1"] if p["vEDS_major_event"])
    kEDS_scoliosis = sum(
        1 for g in ["PLOD1", "FKBP14"] for p in _ALL_COHORTS[g] if p["kyphoscoliosis"]
    )
    plod1_lphp_elevated = sum(1 for p in _ALL_COHORTS["PLOD1"] if p["urine_lphp_ratio"] > 0.09)
    ocular_total = sum(
        1 for g in ["PLOD1", "FKBP14"] for p in _ALL_COHORTS[g] if p["ocular_fragility"]
    )
    fkbp14_hearing = sum(1 for p in _ALL_COHORTS["FKBP14"] if p["hearing_loss"])
    cvEDS_valve = sum(1 for p in _ALL_COHORTS["COL1A2"] if p["cardiac_valve_disease"])
    dEDS_saggy = sum(1 for p in _ALL_COHORTS["ADAMTS2"] if p["sagging_skin"])
    female_n = sum(1 for cohort in _ALL_COHORTS.values() for p in cohort if p["sex"] == "F")
    mean_dx_delay = round(
        sum(p["dx_delay_years"] for cohort in _ALL_COHORTS.values() for p in cohort) / n, 1
    )

    return {
        "atlas_name": "Ehlers-Danlos-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Ehlers-Danlos Syndromes Atlas",
        "n_patients": n,
        "gene_count": len(EDS_GENES),
        "genes": [g["gene"] for g in EDS_GENES],
        "seeds": "1326\u20131333",
        "registered": "2026-09-05",
        "atlas_version": "1.0",
        "gene_summary": [
            {
                "gene": "COL5A1",
                "protein": "Pro-alpha1(V) Collagen",
                "aa": "1838 aa",
                "locus": "9q34.3",
                "inheritance": "AD (haploinsufficiency)",
                "phenotype_short": "Classical EDS type 1 — atrophic scarring + skin hyperextensibility + joint hypermobility",
                "hallmark_short": "Cigarette-paper atrophic scarring + Gorlin sign + velvety skin = cEDS-1",
            },
            {
                "gene": "COL5A2",
                "protein": "Pro-alpha2(V) Collagen",
                "aa": "1499 aa",
                "locus": "2q32.2",
                "inheritance": "AD (haploinsufficiency)",
                "phenotype_short": "Classical EDS type 2 — clinically identical to cEDS-1; molecular testing required",
                "hallmark_short": "Identical to COL5A1 cEDS; COL5A2 is 10% of cEDS; test COL5A1 FIRST",
            },
            {
                "gene": "COL3A1",
                "protein": "Pro-alpha1(III) Collagen",
                "aa": "1466 aa",
                "locus": "2q32.2",
                "inheritance": "AD (dominant negative / haploinsufficiency)",
                "phenotype_short": "Vascular EDS — MOST LETHAL; spontaneous arterial/bowel/uterine rupture; NO elective surgery",
                "hallmark_short": "Translucent skin + acrogeria + spontaneous hollow viscus rupture; celiprolol ONLY RCT therapy",
            },
            {
                "gene": "TNXB",
                "protein": "Tenascin-X",
                "aa": "4268 aa",
                "locus": "6p21.3",
                "inheritance": "AR biallelic (clEDS); haploinsufficiency = HSD only",
                "phenotype_short": "Classical-like EDS — joint hypermobility + skin changes WITHOUT atrophic scarring; adrenal screen mandatory",
                "hallmark_short": "Biallelic = clEDS; one allele = HSD only; contiguous CYP21A2 deletion → adrenal insufficiency",
            },
            {
                "gene": "ADAMTS2",
                "protein": "ADAMTS2 (Procollagen N-Proteinase)",
                "aa": "1211 aa",
                "locus": "5q35.3",
                "inheritance": "AR biallelic",
                "phenotype_short": "Dermatosparaxis EDS — rarest; sagging redundant skin at birth PATHOGNOMONIC",
                "hallmark_short": "Skin hangs in folds at birth; procollagen N-propeptide accumulation on biopsy; extreme bruising",
            },
            {
                "gene": "PLOD1",
                "protein": "Lysyl Hydroxylase 1 (LH1)",
                "aa": "727 aa",
                "locus": "1p36.3",
                "inheritance": "AR biallelic",
                "phenotype_short": "Kyphoscoliotic EDS type 1 — neonatal hypotonia + scoliosis + ocular fragility; urine LP:HP PATHOGNOMONIC",
                "hallmark_short": "Urine LP:HP ratio >0.09 PATHOGNOMONIC; ocular globe rupture risk; ascorbic acid supplementation",
            },
            {
                "gene": "FKBP14",
                "protein": "FKBP Prolyl Isomerase 14 (FKBP22)",
                "aa": "150 aa",
                "locus": "7p14.3",
                "inheritance": "AR biallelic",
                "phenotype_short": "Kyphoscoliotic EDS type 2 — same as PLOD1 kEDS but NORMAL LP:HP + congenital SNHL 50%",
                "hallmark_short": "Normal urine LP:HP (distinguishes from PLOD1); congenital SNHL 50%; audiogram MANDATORY",
            },
            {
                "gene": "COL1A2",
                "protein": "Pro-alpha2(I) Collagen",
                "aa": "1366 aa",
                "locus": "7q21.3",
                "inheritance": "AR biallelic (cvEDS); AD hetero → OI, NOT cvEDS",
                "phenotype_short": "Cardiac-valvular EDS — severe progressive valve disease; annual echo NON-NEGOTIABLE",
                "hallmark_short": "Aortic ± mitral regurgitation requiring valve replacement in young patients + EDS features = cvEDS",
            },
        ],
        "aggregate_stats": {
            "vEDS_major_events_pct": round(vEDS_events / 40 * 100, 1),
            "kEDS_scoliosis_pct": round(kEDS_scoliosis / 80 * 100, 1),
            "plod1_lphp_elevated_pct": round(plod1_lphp_elevated / 40 * 100, 1),
            "ocular_fragility_pct": round(ocular_total / 80 * 100, 1),
            "fkbp14_hearing_loss_pct": round(fkbp14_hearing / 40 * 100, 1),
            "cvEDS_valve_disease_pct": round(cvEDS_valve / 40 * 100, 1),
            "dEDS_sagging_skin_pct": 100,
            "female_pct": round(female_n / n * 100, 1),
            "mean_diagnosis_delay_yrs": mean_dx_delay,
        },
        "critical_drug_rules": {
            "COL3A1_vEDS": "CELIPROLOL 200-400 mg/day ONLY RCT-proven therapy; reduces arterial events ~36%; NO elective surgery; NO colonoscopy; NO angiography",
            "TNXB_adrenal": "ADRENAL SCREEN MANDATORY at diagnosis — contiguous CYP21A2 deletion → adrenal insufficiency; hydrocortisone if confirmed",
            "PLOD1_ascorbic": "Ascorbic acid (Vitamin C) 2-4 g/day: biologically plausible partial benefit for collagen hydroxylation",
            "ALL_EDS_contacts": "NO contact sports, boxing, heavy weightlifting in ALL EDS subtypes; protect from joint hyperextension",
        },
        "kpis": [
            {"label": "Total Patients", "value": str(n)},
            {"label": "Genes Covered", "value": str(len(EDS_GENES))},
            {"label": "EDS Subtypes", "value": "cEDS · vEDS · clEDS · dEDS · kEDS-1 · kEDS-2 · cvEDS"},
            {"label": "vEDS Major Events", "value": f"{vEDS_events}/40 ({round(vEDS_events/40*100,1)}%)"},
            {"label": "LP:HP Elevated (PLOD1)", "value": f"{plod1_lphp_elevated}/40 (pathognomonic)"},
            {"label": "FKBP14 Hearing Loss", "value": f"{fkbp14_hearing}/40 (~50%)"},
            {"label": "cvEDS Valve Disease", "value": f"{cvEDS_valve}/40 ({round(cvEDS_valve/40*100,1)}%)"},
            {"label": "Mean Dx Delay", "value": f"{mean_dx_delay} years"},
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "vEDS (COL3A1): NO ELECTIVE SURGERY — Perioperative Mortality HIGH",
                "body": "Bowel wall and vascular fragility makes elective operations lethal; damage-control only for life-threatening emergencies",
            },
            {
                "type": "danger",
                "title": "vEDS (COL3A1): NO COLONOSCOPY — Bowel Perforation Risk",
                "body": "Colonic mucosa and bowel wall extremely fragile in vEDS; capsule endoscopy or CT colonography if bowel evaluation essential",
            },
            {
                "type": "danger",
                "title": "vEDS (COL3A1): NO CATHETER ANGIOGRAPHY — Arterial Rupture Risk",
                "body": "Arterial manipulation causes acute rupture; CT angiography ONLY; consult vascular surgery with vEDS experience",
            },
            {
                "type": "warning",
                "title": "vEDS PREGNANCY: HIGH RISK — Uterine Rupture Risk 3rd Trimester",
                "body": "Maternal mortality ~10-12% per pregnancy; elective C-section at 34-36 weeks; pre-conception counseling mandatory",
            },
            {
                "type": "warning",
                "title": "TNXB (clEDS): Adrenal Screen MANDATORY — Contiguous CYP21A2 Deletion",
                "body": "17-OHP + synacthen stimulation test at diagnosis; adrenal crisis risk if contiguous deletion confirmed",
            },
            {
                "type": "warning",
                "title": "PLOD1 (kEDS-1): Globe Rupture Risk — Protective Eyewear ALWAYS",
                "body": "Ocular fragility in kEDS-1 means minor trauma can rupture the globe; polycarbonate protective eyewear non-negotiable",
            },
        ],
        "critical_rules": [
            "vEDS (COL3A1): thin translucent skin + young spontaneous arterial/bowel rupture = vEDS; celiprolol FIRST; NO elective surgery",
            "cEDS diagnosis requires BOTH major criteria: atrophic scarring + skin hyperextensibility ≥1.5 cm (elbow)",
            "TNXB haploinsufficiency (one allele) = HSD not EDS; biallelic = clEDS; adrenal screen ALL TNXB patients",
            "kEDS workup: order urine LP:HP ratio FIRST — elevated (>0.09) = PLOD1; normal = consider FKBP14",
            "FKBP14: audiogram mandatory at diagnosis (50% SNHL); ascorbic acid NOT helpful (LH1 intact)",
            "dEDS (ADAMTS2): sagging skin at birth + procollagen N-propeptide biopsy = diagnostic; rarest EDS",
            "cvEDS (COL1A2 biallelic): annual echo non-negotiable; AD COL1A2 → OI not cvEDS (very different disease)",
            "Beighton score ≥5/9 alone is NOT sufficient for EDS diagnosis — requires additional major criteria",
        ],
    }


def get_breakdown():
    rows = []
    for gd in EDS_GENES:
        cohort = _ALL_COHORTS[gd["gene"]]
        n = len(cohort)
        female_n = sum(1 for p in cohort if p["sex"] == "F")
        mean_age = round(sum(p["age"] for p in cohort) / n, 1)
        mean_beighton = round(sum(p["beighton"] for p in cohort) / n, 1)
        mean_lphp = round(sum(p["urine_lphp_ratio"] for p in cohort) / n, 4)
        mean_delay = round(sum(p["dx_delay_years"] for p in cohort) / n, 1)
        special_features = {}
        if gd["gene"] == "COL3A1":
            special_features["vEDS_major_events"] = sum(1 for p in cohort if p["vEDS_major_event"])
        if gd["gene"] in ["PLOD1", "FKBP14"]:
            special_features["kyphoscoliosis"] = sum(1 for p in cohort if p["kyphoscoliosis"])
            special_features["neonatal_hypotonia"] = sum(1 for p in cohort if p["neonatal_hypotonia"])
            special_features["ocular_fragility"] = sum(1 for p in cohort if p["ocular_fragility"])
        if gd["gene"] == "FKBP14":
            special_features["hearing_loss"] = sum(1 for p in cohort if p["hearing_loss"])
        if gd["gene"] == "COL1A2":
            special_features["cardiac_valve_disease"] = sum(1 for p in cohort if p["cardiac_valve_disease"])
        if gd["gene"] == "TNXB":
            special_features["adrenal_insufficiency_subset"] = sum(1 for p in cohort if p["adrenal_insufficiency"])
        rows.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias_summary": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "gene_class": gd["gene_class"],
            "phenotype": gd["phenotype"],
            "key_hallmarks": gd["key_hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "ddx": gd["ddx"],
            "cohort": {
                "n": n,
                "seed": gd["seed"],
                "female_pct": round(female_n / n * 100, 1),
                "mean_age": mean_age,
                "mean_beighton": mean_beighton,
                "mean_urine_lphp": mean_lphp,
                "mean_diagnosis_delay_yrs": mean_delay,
                "special_features": special_features,
            },
        })
    return rows


def get_definitions():
    return {
        "atlas": "Ehlers-Danlos-Atlas",
        "definitions": [
            {
                "term": "Classical EDS (cEDS) — Diagnostic Criteria",
                "short": "TWO major criteria required: (1) atrophic scarring + skin hyperextensibility ≥1.5 cm; (2) Beighton ≥5/9",
                "detail": (
                    "Classical EDS (cEDS) is caused by COL5A1 (~90%) or COL5A2 (~10%) haploinsufficiency. "
                    "DIAGNOSIS requires BOTH major criteria: "
                    "(1) Skin hyperextensibility: ≥1.5 cm when pulled at the elbow; velvety/doughy/soft texture; "
                    "and atrophic (cigarette-paper) scarring at minor injury sites (elbows, knees, shins) — "
                    "this combination is major criterion 1; "
                    "(2) Beighton Hypermobility Score ≥5/9 in adults, ≥6/9 in children — major criterion 2. "
                    "PLUS at least one minor criterion (piezogenic papules, hernia, MVP, blue sclera, etc.). "
                    "SKIN BIOPSY: irregular collagen I fibril diameters on EM ('cauliflower fibrils'); "
                    "collagen V reduction on SDS-PAGE of fibroblast collagen. "
                    "MOLECULAR TESTING: COL5A1 sequencing + MLPA (large deletions missed by sequencing alone). "
                    "Gorlin sign (licking own elbow): ~50% of cEDS; high specificity but low sensitivity."
                ),
                "clinical_rule": "Atrophic scarring ABSENT → NOT cEDS; Beighton alone INSUFFICIENT; both major criteria required",
            },
            {
                "term": "Vascular EDS (vEDS) — Celiprolol Protocol and Surgical Rules",
                "short": "Celiprolol 200-400 mg/day ONLY proven therapy; NO elective surgery, NO colonoscopy, NO catheter angiography",
                "detail": (
                    "vEDS (COL3A1) management requires specific protocols distinct from all other EDS types: "
                    "CELIPROLOL: beta-1/alpha-2 selective blocker; Ong 2010 NEJM trial: 200 mg increased to 400 mg → "
                    "36% relative risk reduction in arterial events (dissection, rupture, death). "
                    "Mechanism: reduces shear stress on arterial wall; NOT just antihypertensive effect. "
                    "Target BP: <120/80 mmHg; add ACE inhibitor/ARB if needed. "
                    "SURGICAL RESTRICTIONS: "
                    "(1) NO elective surgery — bowel wall and arterial wall fragility makes elective operations lethal; "
                    "(2) NO colonoscopy — colon perforation risk; use capsule endoscopy or CT colonography; "
                    "(3) NO catheter angiography — arterial manipulation causes acute rupture; CT-A only. "
                    "EMERGENCY SURGERY: when unavoidable (hemorrhage control), use pledgeted sutures, "
                    "minimal dissection, high-volume vEDS center. "
                    "PREGNANCY: uterine rupture risk especially 3rd trimester + immediate postpartum; "
                    "maternal mortality ~10-12% per pregnancy; elective C-section preferred at 34-36 weeks."
                ),
                "clinical_rule": "vEDS emergency card: patient always carries card stating 'VASCULAR EDS: NO angiography, NO colonoscopy, NO elective surgery'",
            },
            {
                "term": "Beighton Score — Joint Hypermobility Screening (Not EDS Diagnosis)",
                "short": "Beighton ≥5/9 = generalized joint hypermobility; NOT sufficient alone for EDS diagnosis",
                "detail": (
                    "The Beighton Hypermobility Score is a 9-point assessment: "
                    "1 point each: bilateral passive dorsiflexion of 5th MCP >90°, "
                    "bilateral passive thumb apposition to flexor forearm, "
                    "bilateral elbow hyperextension >10°, bilateral knee hyperextension (genu recurvatum) >10°, "
                    "and palms flat on floor with knees extended. "
                    "THRESHOLDS: ≥5/9 adults, ≥6/9 children (<18y), ≥4/9 older adults (>50y) = generalized hypermobility. "
                    "CRITICAL CAVEAT: Beighton score screens for hypermobility but does NOT diagnose any EDS type. "
                    "hEDS requires Beighton ≥5 PLUS systemic manifestations; "
                    "cEDS requires Beighton ≥5 PLUS atrophic scarring + skin hyperextensibility ≥1.5 cm. "
                    "HSD (hypermobility spectrum disorder): Beighton ≥5 WITHOUT meeting full EDS criteria. "
                    "vEDS: Beighton score often LOW — do NOT exclude vEDS because Beighton is normal."
                ),
                "clinical_rule": "Beighton ≥5 alone ≠ EDS; vEDS may have low Beighton — never exclude vEDS based on joint mobility alone",
            },
            {
                "term": "Urine LP:HP Ratio — PLOD1 (kEDS-1) Biomarker",
                "short": "Urine lysylpyridinoline:hydroxylysylpyridinoline ratio >0.09 PATHOGNOMONIC for PLOD1 (kEDS-1)",
                "detail": (
                    "The urinary LP:HP ratio measures the ratio of pyridinoline cross-links formed from "
                    "non-hydroxylated lysine (LP = lysylpyridinoline, deoxypyridinoline) vs "
                    "hydroxylated lysine (HP = hydroxylysylpyridinoline). "
                    "NORMAL: LP:HP <0.04; "
                    "kEDS-1 (PLOD1 deficiency): LP:HP >0.09 (typically 0.10-0.30) — "
                    "reflects LH1 enzyme absence → lysines NOT hydroxylated → "
                    "LP dominates over HP in urine cross-link output. "
                    "PATHOGNOMONIC: LP:HP >0.09 in a patient with kyphoscoliosis + hypotonia + joint laxity = PLOD1 deficiency. "
                    "PRACTICAL: This is a non-invasive urine test (24-hour collection or random spot). "
                    "Order FIRST before expensive molecular testing. "
                    "kEDS-2 (FKBP14): LP:HP NORMAL — this is the key biochemical distinguishing test. "
                    "Analytic method: HPLC or tandem mass spectrometry."
                ),
                "clinical_rule": "Any child with neonatal hypotonia + kyphoscoliosis → urine LP:HP FIRST; elevated (>0.09) = PLOD1 kEDS-1; normal = test FKBP14",
            },
            {
                "term": "TNXB Haploinsufficiency vs Biallelic — HSD vs clEDS",
                "short": "TNXB ONE LOF allele = HSD (hypermobility spectrum disorder); BOTH alleles null = classical-like EDS",
                "detail": (
                    "TNXB encodes Tenascin-X, a collagen I stabilizer. "
                    "Two very different outcomes depending on allele dose: "
                    "BIALLELIC LOF (AR): complete TNX absence → classical-like EDS (clEDS) — "
                    "joint hypermobility + skin hyperextensibility WITHOUT atrophic scarring. "
                    "HAPLOINSUFFICIENCY (one LOF allele): 50% TNX → "
                    "hypermobility spectrum disorder (HSD) — does NOT meet EDS criteria. "
                    "HSD patients have significant symptoms (pain, fatigue, POTS) but do NOT carry an EDS diagnosis. "
                    "CLINICAL IMPLICATION: "
                    "A TNXB heterozygous variant finding does NOT mean the patient has EDS — "
                    "look for the second allele; if only one found → HSD. "
                    "CONTIGUOUS DELETION ALERT: Large 6p21.3 deletions include both TNXB AND CYP21A2 → "
                    "clEDS + congenital adrenal hyperplasia (21-hydroxylase deficiency) simultaneously; "
                    "adrenal screen mandatory in ALL TNXB biallelic patients."
                ),
                "clinical_rule": "TNXB heterozygous variant alone = HSD NOT EDS; second allele confirmation required for clEDS; adrenal screen ALL biallelic TNXB",
            },
            {
                "term": "Dermatosparaxis EDS (dEDS) — Procollagen N-Propeptide Retention",
                "short": "ADAMTS2 LOF → procollagen N-propeptide NOT cleaved → sagging redundant skin at birth PATHOGNOMONIC",
                "detail": (
                    "Dermatosparaxis EDS (dEDS): caused by biallelic ADAMTS2 LOF (AR). "
                    "MECHANISM: ADAMTS2 (procollagen N-proteinase) cleaves the N-terminal propeptide from "
                    "procollagens I, II, III before fibril assembly. "
                    "In dEDS: propeptide retained → procollagen I cannot form normal tight fibrils → "
                    "structurally defective, excessively lax fibrils. "
                    "CLINICAL HALLMARK: Skin that is dramatically sagging/redundant at BIRTH — "
                    "can be pulled far from the body in loose folds without tearing. "
                    "This is qualitatively different from hyperextensibility: it SAGS rather than STRETCHES. "
                    "BIOPSY: Immunostaining for procollagen I N-terminal propeptide (pNI antibody) — "
                    "positive staining in dermis = pathognomonic. "
                    "RAREST EDS: fewer than 50 cases worldwide; consanguinity common. "
                    "BRUISING: extremely severe, out of proportion to trauma. "
                    "NAMING: named for 'dermatosparaxis' — a collagen disorder of cattle and sheep "
                    "where ADAMTS2 deficiency was first described."
                ),
                "clinical_rule": "Sagging redundant skin at birth + extreme bruising → dEDS (ADAMTS2); confirm with procollagen N-propeptide biopsy immunostaining",
            },
            {
                "term": "Ocular Fragility in kEDS — Globe Rupture Risk",
                "short": "Scleral fragility in PLOD1 kEDS-1 → globe rupture from minor trauma; polycarbonate eyewear MANDATORY",
                "detail": (
                    "Kyphoscoliotic EDS type 1 (PLOD1) is associated with significant ocular fragility: "
                    "The sclera (outer eye coat) is composed largely of collagen I fibrils; "
                    "in kEDS-1, deficient LH1 → undercross-linked collagen I fibrils in sclera → "
                    "scleral weakness → globe can rupture with minor blunt trauma. "
                    "PREVALENCE: ~60% of kEDS-1 patients have clinically significant ocular features; "
                    "globe rupture has been documented with minor household accidents. "
                    "PREVENTIVE MEASURES: "
                    "(1) Polycarbonate protective eyewear at ALL times during physical activity; "
                    "(2) Annual ophthalmology: slit-lamp, IOP, corneal pachymetry; "
                    "(3) AVOID: contact sports, martial arts, racket sports, ball sports; "
                    "(4) Inform patient and school/workplace of ocular risk. "
                    "kEDS-2 (FKBP14): lower risk of globe rupture than kEDS-1 but still recommend ophthalmology assessment."
                ),
                "clinical_rule": "kEDS-1 (PLOD1): polycarbonate eyewear ALWAYS; NO ball sports, boxing, contact activities; annual ophthalmology NON-NEGOTIABLE",
            },
            {
                "term": "Cardiac-Valvular EDS (cvEDS) — COL1A2 Biallelic vs Heterozygous",
                "short": "AR biallelic COL1A2 null → cvEDS (severe valve disease); AD hetero COL1A2 Gly → OI (bone fragility) — completely different diseases",
                "detail": (
                    "COL1A2 variants cause very different diseases depending on zygosity and variant type: "
                    "BIALLELIC NULL (cvEDS): "
                    "No alpha2(I) chains → only homotrimers [alpha1(I)]3 → "
                    "structurally altered collagen I → severe progressive cardiac valve disease "
                    "(aortic regurgitation ± mitral regurgitation) + EDS features (hypermobility, skin changes). "
                    "HETEROZYGOUS GLY SUBSTITUTION (OI spectrum): "
                    "Dominant negative effect on collagen I triple helix → "
                    "OI type III/IV (severe bone fragility, multiple fractures, blue sclera, hearing loss, Wormian bones). "
                    "CRITICAL CLINICAL DISTINCTION: "
                    "A patient with bone fractures + blue sclera → likely OI (COL1A2 Gly variant); "
                    "A patient with severe aortic regurgitation + joint hypermobility in childhood → "
                    "consider cvEDS (COL1A2 biallelic null). "
                    "cvEDS is extremely rare (<30 cases); "
                    "any young patient requiring valve replacement + EDS features → COL1A2 panel."
                ),
                "clinical_rule": "Young patient: severe aortic/mitral regurgitation + EDS features → COL1A2 biallelic (cvEDS); COL1A2 Gly variant → OI, NOT cvEDS",
            },
            {
                "term": "EDS Pregnancy Management — Subtype-Specific Risk",
                "short": "vEDS: maternal mortality 10-12%/pregnancy; all EDS: pelvic floor/symphysis pubis; physio before and after delivery",
                "detail": (
                    "Pregnancy in EDS requires subtype-specific management: "
                    "VASCULAR EDS (COL3A1) — HIGHEST RISK: "
                    "Uterine rupture risk in 3rd trimester and immediately postpartum; "
                    "maternal mortality ~10-12% per pregnancy (arterial rupture, uterine rupture); "
                    "Protocol: elective C-section at 34-36 weeks; "
                    "pre-conception counseling mandatory; "
                    "delivery at tertiary center with vascular surgery on-site. "
                    "CLASSICAL EDS (COL5A1/2): perineal tears + wound dehiscence more common; "
                    "symphysis pubis dysfunction; slow wound healing; episiotomy healing poor — repair in layers. "
                    "kEDS (PLOD1/FKBP14): kyphoscoliosis may worsen with pregnancy weight; "
                    "pelvic instability; assisted vaginal delivery or C-section depending on spinal curvature. "
                    "ALL EDS: pelvic floor physiotherapy antenatally and postnatally; "
                    "multidisciplinary team: maternal-fetal medicine + connective tissue disorder specialist."
                ),
                "clinical_rule": "vEDS: elective C-section at 34-36 weeks; counsel before conception; deliver at tertiary centre with vascular surgery",
            },
            {
                "term": "Cascade Testing — Ehlers-Danlos Syndromes",
                "short": "Offer molecular testing to all first-degree relatives after proband diagnosis; subtype-specific inheritance determines risk",
                "detail": (
                    "Cascade testing in EDS depends on inheritance pattern: "
                    "CLASSICAL EDS (COL5A1/2, AD): 50% risk to each child; "
                    "test all first-degree relatives; incomplete penetrance means relatives may have milder phenotype. "
                    "VASCULAR EDS (COL3A1, AD): 50% risk to each child; "
                    "positive cascade testing → immediate celiprolol + surveillance; "
                    "de novo rate ~50% → parents may test negative. "
                    "clEDS (TNXB, AR): 25% risk to siblings; parents are obligate carriers (asymptomatic); "
                    "offer prenatal testing. "
                    "kEDS (PLOD1/FKBP14, AR): same AR pattern; 25% sib risk; "
                    "urine LP:HP cascade screening in siblings of PLOD1 kEDS is practical and non-invasive. "
                    "dEDS (ADAMTS2, AR): siblings at 25% risk; consanguinity counseling. "
                    "cvEDS (COL1A2, AR): siblings at 25% risk; parents asymptomatic carriers; "
                    "echocardiography in all newly confirmed cvEDS patients."
                ),
                "clinical_rule": "EDS diagnosis confirmed → cascade test all first-degree relatives; inheritance pattern determines who to test and how urgently",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:1000])
    print("\n=== BREAKDOWN (gene 1) ===")
    bd = get_breakdown()
    print(json.dumps(bd[0], indent=2)[:800])
    print("\n=== DEFINITIONS (first 2) ===")
    df = get_definitions()
    print(json.dumps(df["definitions"][:2], indent=2)[:800])
