#!/usr/bin/env python3
"""Skeletal Dysplasia Atlas — Complete 8-Gene Hereditary Skeletal Dysplasia Atlas
COL1A1  (Osteogenesis Imperfecta Type I — most common OI; AD; blue sclerae PATHOGNOMONIC;
         bisphosphonates first-line; 17q21.33; 1464 aa) ·
COL1A2  (Osteogenesis Imperfecta Type III/IV — most severe survivable OI; AD dominant;
         AR rare; 7q21.3; 1366 aa) ·
FGFR3   (Achondroplasia/Hypochondroplasia/TDII — most common short-limb dwarfism;
         AD GOF; G380R 95%; vosoritide FDA 2021; 4p16.3; 806 aa) ·
EXT1    (Hereditary Multiple Exostoses Type 1 — most common; AD LOF; heparan sulphate;
         HME1; malignant transformation 2-5%; 8q24.11; 746 aa) ·
EXT2    (Hereditary Multiple Exostoses Type 2 — AD LOF; HME2; EXT1 more severe;
         11p12-p11; 718 aa) ·
SLC26A2 (Diastrophic Dysplasia / Achondrogenesis IB — sulphate transporter;
         AR; DTD survivable vs ACG1B lethal by genotype; 5q32; 739 aa) ·
RMRP    (Cartilage-Hair Hypoplasia — ribonuclease MRP RNA; AR; CHH;
         Finnish/Amish founder; immune deficiency; lymphoma risk; 9p13.3) ·
COMP    (Pseudoachondroplasia/Multiple Epiphyseal Dysplasia — COMP; AD;
         19p13.11; 757 aa; PSACH severe; MED mild)
320-patient aggregate cohort (8 × 40, seeds 1182–1189)
"""

import random

SEED_BASE = 1182

SKELETAL_DYSPLASIA_GENES = [
    # ── COL1A1 — Osteogenesis Imperfecta Type I ─────────────────────────────
    {
        "gene": "COL1A1",
        "protein": "Procollagen alpha-1(I) chain",
        "alias": (
            "COL1A1; OMIM gene 120150; 17q21.33; 1464 aa; OI Type I OMIM #166200; "
            "AD; prevalence ~1 in 15,000–20,000; most common form of OI (~60% of all OI); "
            "quantitative defect — haploinsufficiency → 50% normal collagen"
        ),
        "aa": "1464 aa",
        "kDa": "~139 kDa (preprotein)",
        "gene_class": (
            "Fibrillar collagen — type I collagen alpha-1 chain; "
            "triple helix with COL1A2 (two alpha-1 + one alpha-2 chains → collagen I heterotrimer); "
            "forms bone matrix (90% of organic bone matrix is collagen I); "
            "also in tendons, ligaments, dermis, cornea, sclera; "
            "COL1A1 null allele (stop codon, splice, large deletion) → HALF-NORMAL collagen I output "
            "(haploinsufficiency) → Type I OI — mild, blue sclerae, hearing loss by 30-40y; "
            "COL1A1 glycine substitutions → structurally abnormal collagen → Type II (perinatal lethal) "
            "or Type III (progressively deforming) OI"
        ),
        "locus": "17q21.33",
        "omim_gene": 120150,
        "omim_disease": 166200,
        "phenotype": (
            "OI Type I (quantitative defect): Blue sclerae (PATHOGNOMONIC — blue-grey hue from thin sclera "
            "allowing choroidal pigment to show through); fractures with minimal trauma from infancy; "
            "typically 5-20 fractures/year in childhood declining after puberty; "
            "short stature (mild — within normal range or low-normal); "
            "dentinogenesis imperfecta (DI) in ~25% Type I (Type IB); "
            "sensorineural and/or conductive hearing loss (40-60% by age 40); "
            "normal at birth, fractures begin when child starts weight-bearing; "
            "joint hypermobility; easy bruising; "
            "MOBILE between fractures — normal function except during acute episodes; "
            "OI Type II/III (structural defect): perinatal lethal / severe progressive deformity"
        ),
        "hallmark": (
            "BLUE SCLERAE — pathognomonic for OI (Type I especially); "
            "thin sclera allows underlying uvea to be seen; "
            "FRACTURES with minimal trauma (sneeze, roll over in bed, blood pressure cuff); "
            "WORMIAN BONES on skull X-ray (intrasutural ossicles — highly specific for OI); "
            "NORMAL INTELLIGENCE — critical to distinguish from NAI (non-accidental injury)"
        ),
        "treatment_alert": (
            "BISPHOSPHONATES (pamidronate IV, alendronate PO) — first-line in moderate/severe OI; "
            "reduces fracture rate 40-50%; increases BMD; given cyclically (pamidronate q4m); "
            "ZOLEDRONATE annual infusion — simpler regimen; "
            "DENOSUMAB for bisphosphonate-refractory; "
            "ANTI-SCLEROSTIN (romosozumab) — under study; "
            "HORMONE: GH in growing children with some response; "
            "SURGICAL: telescoping intramedullary rods (Fassier-Duval) for severe/recurrent fractures; "
            "CALCIUM + VITAMIN D — adjunct (do NOT substitute for bisphosphonates); "
            "STEROID EXCESS (exogenous or Cushing) — WORSENS bone fragility — avoid; "
            "BISPHOSPHONATE HOLIDAY after growth plate closure — atypical femoral fracture risk if continued"
        ),
        "key_ddx": (
            "Non-accidental injury (NAI/child abuse): OI blue sclerae, wormian bones, family history; "
            "NAI: fractures in non-ambulatory infant, posterior rib, metaphyseal corner fractures; "
            "Rickets: wide growth plates, low vitamin D/phosphate, cupping; NOT wormian bones; "
            "Hypophosphatasia: low ALP (PATHOGNOMONIC), phosphoethanolaminuria; "
            "Ehlers-Danlos: hypermobility + skin extensibility + normal bone density; "
            "Juvenile osteoporosis: normal sclerae; no wormian bones; DXA essential"
        ),
        "gfr_pattern": (
            "Normal renal function in OI; "
            "hypercalciuria possible with immobility (fracture/post-op) → nephrolithiasis risk; "
            "bisphosphonate nephrotoxicity rare at standard doses; "
            "eGFR monitoring recommended annually in patients on long-term bisphosphonates"
        ),
        "proteinuria_pattern": (
            "Not primary; "
            "NSAID overuse for pain → analgesic nephropathy risk in older patients; "
            "no direct collagen I-related nephropathy"
        ),
        "primary_complication": (
            "Recurrent fractures → progressive deformity; sensorineural hearing loss; "
            "basilar invagination (brainstem compression — rare, severe OI); "
            "scoliosis; dental fragility (dentinogenesis imperfecta)"
        ),
        "disease_detail": (
            "Type I collagen is the principal structural protein of bone. The COL1A1 gene encodes "
            "procollagen alpha-1(I). Two alpha-1(I) chains plus one alpha-2(I) chain (COL1A2) "
            "form the triple-helical procollagen I molecule. "
            "In OI Type I (the most common and mildest), a COL1A1 null allele (frameshift, "
            "nonsense, splice-site, large deletion) causes haploinsufficiency — the cell "
            "produces only 50% of the normal amount of structurally normal collagen I. "
            "This reduces bone matrix quantity but preserves quality, producing a mild phenotype.\n\n"
            "The triple helix requires an uninterrupted Gly-X-Y repeat. When a glycine "
            "(the only amino acid small enough to occupy the helix axis) is substituted "
            "by any other residue (COL1A1 Gly→ missense), structurally abnormal collagen is "
            "produced. These structural (qualitative) defects → OI Type II (perinatally lethal) "
            "or Type III (progressively deforming, never ambulatory).\n\n"
            "Diagnosis: clinical triad (fractures + blue sclerae + family history); "
            "X-ray (osteopenia, wormian bones, bowing); biochemical (collagen electrophoresis "
            "on skin fibroblasts); molecular (COL1A1/COL1A2 sequencing + deletion/duplication).\n\n"
            "Blue sclerae arise because the thinned scleral collagen is translucent, "
            "permitting the dark blue-grey choroidal pigment to show through — "
            "PATHOGNOMONIC for OI Type I but may fade in adults.\n\n"
            "Wormian bones (intrasutural ossicles, >10 in number, >6×4mm each) are "
            "pathognomonic — found in 78% of OI patients; "
            "also in hypothyroidism, Down syndrome, cleidocranial dysostosis but rare."
        ),
        "inheritance": "Autosomal dominant (haploinsufficiency — null allele); rarely AR for structural defects",
        "variants": [
            {"variant": "COL1A1 null alleles (stop/frameshift/splice)", "effect": "Type I OI (haploinsufficiency)", "frequency": "60%"},
            {"variant": "c.934C>T p.Arg312* (CGA→TGA)", "effect": "Recurrent null mutation", "frequency": "4%"},
            {"variant": "Gly→Ser/Cys/Arg (helix)", "effect": "Type II/III OI (structural)", "frequency": "30%"},
            {"variant": "Exon skipping (splice)", "effect": "Variable — exon position-dependent", "frequency": "10%"},
        ],
        "drug_ci": [
            "Prolonged corticosteroids (Cushing → osteoporosis — WORSENS OI)",
            "Bisphosphonate holiday omission post-growth (atypical femur fracture risk)",
        ],
    },
    # ── COL1A2 — Osteogenesis Imperfecta Type III / IV ──────────────────────
    {
        "gene": "COL1A2",
        "protein": "Procollagen alpha-2(I) chain",
        "alias": (
            "COL1A2; OMIM gene 120160; 7q21.3; 1366 aa; OI Type III OMIM #259420; "
            "OI Type IV OMIM #166220; AD (structural glycine substitutions); "
            "AR (null — rare, recessive OI); prevalence: OI III ~1 in 60,000"
        ),
        "aa": "1366 aa",
        "kDa": "~129 kDa (preprotein)",
        "gene_class": (
            "Fibrillar collagen — type I collagen alpha-2 chain; "
            "forms [α1(I)]₂α2(I) heterotrimer with two COL1A1 chains; "
            "COL1A2 null (AR) → homotrimeric [α1(I)]₃ collagen — overmodified, less stable; "
            "glycine substitutions in COL1A2 → structurally abnormal heterotrimer → "
            "OI Type III (progressively deforming) or Type IV (moderately severe); "
            "C-terminal glycine substitutions in COL1A2 tend to be more severe "
            "than N-terminal (gradient rule — proximity to C-terminal propeptide cleavage site)"
        ),
        "locus": "7q21.3",
        "omim_gene": 120160,
        "omim_disease": 259420,
        "phenotype": (
            "OI Type III (COL1A2 Gly substitutions — progressively deforming): "
            "Most severe non-lethal OI; multiple fractures at birth (often diagnosed in utero); "
            "severe short stature (adult height 85-107 cm); "
            "progressive scoliosis (80%+); "
            "triangular face (frontal bossing, small mandible); "
            "progressive limb bowing from recurrent fractures; "
            "wheelchair-bound in majority by adolescence; "
            "white or pale blue sclerae (less blue than Type I); "
            "dentinogenesis imperfecta (50%); "
            "basilar invagination (life-threatening brainstem compression — 25%); "
            "OI Type IV (moderate): intermediate; normal/pale white sclerae; ambulatory"
        ),
        "hallmark": (
            "PROGRESSIVELY DEFORMING OI — multiple intrauterine fractures; "
            "TRIANGULAR FACE (frontal bossing + small jaw) PATHOGNOMONIC for Type III; "
            "BASILAR INVAGINATION (brainstem compression from cranial settling) — "
            "requires annual MRI surveillance from age 5; "
            "ADULT HEIGHT <110 cm in most Type III; "
            "POPCORN CALCIFICATIONS at metaphyses on X-ray — pathognomonic for severe OI"
        ),
        "treatment_alert": (
            "BISPHOSPHONATES (pamidronate IV) — start in infancy (3-6 months); "
            "reduces fracture rate 30-40%; does NOT cure deformity; "
            "INTRAMEDULLARY RODS mandatory for ambulatory potential (Fassier-Duval telescoping); "
            "BASILAR INVAGINATION SURVEILLANCE — annual MRI brain/cervical spine from age 5; "
            "surgical decompression if brainstem compressed (C1-C2 compression → respiratory failure); "
            "SCOLIOSIS SURGERY — mandatory if Cobb angle >50°; "
            "GROWTH HORMONE — modest benefit in Type IV, minimal in Type III; "
            "ANTIRESORPTIVE HOLIDAY — risk of atypical femoral fracture if continued post-growth plate fusion; "
            "TERIPARATIDE — CONTRAINDICATED in active bone growth (risk of osteosarcoma in rodents)"
        ),
        "key_ddx": (
            "OI Type II (perinatal lethal): undulations of ribs, dark bones on X-ray, "
            "still/perinatal death — COL1A1/COL1A2 Gly→Cys; "
            "Hypophosphatasia (perinatal lethal): LOW ALP (alkaline phosphatase) — "
            "CRITICAL — do NOT give bisphosphonates; worsens HPP; "
            "Achondrogenesis: lethal, no ossification of spine; "
            "Campomelic dysplasia: bowed femora + tracheobronchomalacia + SRY sex reversal; "
            "Thanatophoric dysplasia: FGFR3 GOF; telephone receiver femora; "
            "Severe OI vs NAI: OI multiple ages/stages of fractures; family history"
        ),
        "gfr_pattern": (
            "Normal GFR; "
            "hypercalciuria with prolonged immobility → nephrolithiasis; "
            "bisphosphonate monitoring (creatinine pre each infusion); "
            "no primary collagen nephropathy from COL1A2"
        ),
        "proteinuria_pattern": (
            "Not a feature; "
            "monitor for analgesic nephropathy with chronic pain patients; "
            "no Alport-type nephropathy (collagen IV — different genes COL4A3/4/5)"
        ),
        "primary_complication": (
            "Progressive skeletal deformity; scoliosis; basilar invagination; "
            "pulmonary compromise from thoracic deformity; "
            "dentinogenesis imperfecta; early deafness"
        ),
        "disease_detail": (
            "COL1A2 encodes procollagen alpha-2(I), which combines with two COL1A1 "
            "alpha-1(I) chains to form the type I collagen heterotrimer [α1(I)]₂α2(I). "
            "Glycine substitutions in the triple helical domain of COL1A2 produce "
            "qualitatively abnormal collagen I. The gradient rule applies: "
            "more C-terminal glycine substitutions (closer to propeptide cleavage site) "
            "produce more severe phenotypes, because folding of the triple helix proceeds "
            "from C-to-N terminus and C-terminal defects disrupt a greater proportion of the molecule.\n\n"
            "OI Type III (progressively deforming) is the most severe non-lethal form. "
            "Patients sustain dozens to hundreds of fractures over a lifetime. "
            "Progressive bowing and angulation of long bones occurs with each healed fracture. "
            "The triangular face (prominent forehead, small mandible) is characteristic.\n\n"
            "Basilar invagination develops as the skull base softens and the odontoid "
            "process of C2 migrates superiorly to compress the brainstem. "
            "Early symptoms: headache, tinnitus, dysphagia, limb weakness. "
            "Untreated: respiratory failure and death. Annual MRI from age 5 is mandatory.\n\n"
            "Rare recessive COL1A2 null mutations (AR) produce homotrimeric [α1(I)]₃ "
            "collagen — overmodified (excess hydroxylysine/glycosylation) and mechanically "
            "weaker than normal heterotrimer, causing a severe OI phenotype."
        ),
        "inheritance": "Autosomal dominant (structural Gly substitutions); rarely AR (null → homotrimeric collagen)",
        "variants": [
            {"variant": "Gly→Ser/Ala/Cys/Arg/Asp/Val (C-term helix)", "effect": "Type III OI — severe", "frequency": "40%"},
            {"variant": "Gly→Ser/Ala (N-term helix)", "effect": "Type IV OI — moderate", "frequency": "35%"},
            {"variant": "Biallelic null (AR)", "effect": "Recessive OI — severe", "frequency": "5%"},
            {"variant": "Splice/exon skip", "effect": "Variable — position dependent", "frequency": "20%"},
        ],
        "drug_ci": [
            "Teriparatide (PTH1-34) — CONTRAINDICATED during active bone growth (rodent osteosarcoma risk; also futile in severe OI)",
            "Hypophosphatasia misdiagnosed as OI — bisphosphonates WORSEN HPP (CRITICAL DDx: check ALP)",
        ],
    },
    # ── FGFR3 — Achondroplasia / Hypochondroplasia / TDII ───────────────────
    {
        "gene": "FGFR3",
        "protein": "Fibroblast growth factor receptor 3",
        "alias": (
            "FGFR3; OMIM gene 134934; 4p16.3; 806 aa; Achondroplasia OMIM #100800; "
            "AD GOF; prevalence ~1 in 15,000–25,000; MOST COMMON form of short-limb dwarfism; "
            "~80% de novo (new mutation in paternal germ cell, increases with paternal age)"
        ),
        "aa": "806 aa",
        "kDa": "~87 kDa",
        "gene_class": (
            "Receptor tyrosine kinase — fibroblast growth factor receptor family; "
            "three extracellular Ig-like domains + transmembrane domain + intracellular kinase domain; "
            "normally expressed in cartilage growth plate chondrocytes — inhibits chondrocyte "
            "proliferation and differentiation (negative regulator of endochondral ossification); "
            "GOF variants (gain-of-function) → constitutive FGFR3 signalling → "
            "EXCESS INHIBITION of endochondral bone growth → rhizomelic short stature; "
            "activating pathway: RAS-MAPK, STAT1/3 → antiproliferative signal to chondrocytes; "
            "ACH: G380R (transmembrane domain — 95% of ACH); "
            "HCH: N540K (TK1 domain — milder); "
            "TDII: K650E/M (TK2 domain — most severe, perinatally lethal)"
        ),
        "locus": "4p16.3",
        "omim_gene": 134934,
        "omim_disease": 100800,
        "phenotype": (
            "Achondroplasia (ACH): Rhizomelic short stature (proximal limbs shortened more than distal); "
            "macrocephaly (large head) with frontal bossing; midface hypoplasia (flat nasal bridge); "
            "adult height 118-145 cm (mean 131 cm M, 124 cm F); "
            "trident hand (fingers splayed in characteristic tripod pattern); "
            "thoracolumbar kyphosis (gibbus) in infancy → lordosis in childhood; "
            "lumbar spinal stenosis (MOST COMMON ADULT COMPLICATION — pain, neurogenic claudication); "
            "foramen magnum stenosis (infancy — apnoea, sudden death risk); "
            "otitis media (very common — Eustachian tube small); "
            "NORMAL INTELLIGENCE"
        ),
        "hallmark": (
            "RHIZOMELIC SHORT STATURE + MACROCEPHALY + TRIDENT HAND PATHOGNOMONIC; "
            "FORAMEN MAGNUM STENOSIS — LIFE-THREATENING in infancy; "
            "screen ALL ACH infants: brain MRI at 6-12 months; "
            "apnoea monitor until foramen magnum assessed; "
            "LUMBAR SPINAL STENOSIS — #1 adult complication; "
            "VOSORITIDE (anti-FGFR3 CNP analogue) FDA 2021 — first disease-modifying therapy for ACH"
        ),
        "treatment_alert": (
            "VOSORITIDE (BMN 111) — daily SC injection; CNP analogue bypasses FGFR3 signalling; "
            "FDA approved August 2021 (age 5 to growth plate closure); "
            "increases annual growth velocity +1.6 cm/year; "
            "START BEFORE growth plate closure; "
            "FORAMEN MAGNUM DECOMPRESSION (C1 laminectomy) — URGENT if central apnoea, "
            "hyperreflexia, clonus, or severe hydrocephalus; "
            "LUMBAR DECOMPRESSION surgery for spinal stenosis (most adults eventually need); "
            "LIMB LENGTHENING (Ilizarov/STRYDE nail) — controversial; patient-choice; "
            "GROWTH HORMONE — does NOT significantly increase final height in ACH (FGFR3 pathway blocks); "
            "DO NOT use GH as primary therapy — discuss early; "
            "OBESITY PREVENTION — critical (worsens back/joint pain and apnoea)"
        ),
        "key_ddx": (
            "Hypochondroplasia (HCH): FGFR3 N540K; milder; NO macrocephaly; NO trident hand; "
            "may not be diagnosed until childhood growth failure; "
            "Thanatophoric Dysplasia (TDII): K650E — perinatally lethal; telephone receiver femora; "
            "severe platyspondyly; brain heterotopia; "
            "Pseudoachondroplasia: COMP gene (NOT FGFR3); NORMAL HEAD CIRCUMFERENCE; normal face; "
            "diagnosed later (2-3y when walking begins); "
            "SADDAN (Severe achondroplasia with developmental delay and acanthosis nigricans): K650M; "
            "Homozygous ACH: fatal perinatally (two copies G380R → TDII-equivalent); "
            "parents both ACH → 25% chance homozygous lethal child"
        ),
        "gfr_pattern": (
            "Normal GFR; "
            "obesity-associated CKD if BMI uncontrolled; "
            "foramen magnum stenosis → brainstem → autonomic → no direct renal involvement; "
            "NSAIDs chronic use for back pain → analgesic nephropathy risk"
        ),
        "proteinuria_pattern": (
            "Not a primary feature; "
            "obesity/hypertension → microalbuminuria in adults; "
            "no intrinsic nephropathy from FGFR3"
        ),
        "primary_complication": (
            "Foramen magnum stenosis (infancy — apnoea, sudden death); "
            "lumbar spinal stenosis (adults — pain, neurogenic claudication); "
            "recurrent otitis media → conductive hearing loss; "
            "obesity (metabolic complications); "
            "psychosocial impact"
        ),
        "disease_detail": (
            "FGFR3 is a negative regulator of chondrocyte proliferation in the growth plate. "
            "The G380R variant (c.1138G>A or c.1138G>C) in the transmembrane domain causes "
            "constitutive activation by promoting receptor dimerisation and signalling "
            "without ligand binding — a gain-of-function mechanism. "
            "This leads to sustained STAT1 and MAPK signalling in growth plate chondrocytes, "
            "suppressing their proliferation and hypertrophy. "
            "Endochondral ossification (the mechanism by which long bones grow in length) "
            "is therefore impaired, producing rhizomelic shortening.\n\n"
            "G380R arises predominantly de novo in paternal spermatogenesis — "
            "advanced paternal age (>35 years) increases risk. "
            "The same base pair mutation is responsible for >95% of all achondroplasia cases worldwide.\n\n"
            "Vosoritide is a modified C-type natriuretic peptide (CNP) analogue. "
            "CNP normally activates NPR-B on chondrocytes, counteracting FGFR3 signalling "
            "downstream of the receptor. Daily SC vosoritide restores partial endochondral "
            "ossification, increasing growth velocity by ~1.6 cm/year in children aged 5-18.\n\n"
            "Foramen magnum stenosis occurs because the posterior fossa is small in ACH "
            "(membranous ossification of cranial base is FGFR3-independent but "
            "the foramen magnum is small due to impaired endochondral skull base growth). "
            "Compression of the medulla → central apnoea → sudden infant death. "
            "Brain MRI with foramen magnum measurements (normal >3.5 mm) is mandatory."
        ),
        "inheritance": "Autosomal dominant GOF; ~80% de novo (paternal germline); 20% inherited from affected parent",
        "variants": [
            {"variant": "c.1138G>A p.Gly380Arg", "effect": "Achondroplasia — 95%", "frequency": "95%"},
            {"variant": "c.1138G>C p.Gly380Arg", "effect": "Achondroplasia — same AA change", "frequency": "1%"},
            {"variant": "c.1620C>A/G p.Asn540Lys", "effect": "Hypochondroplasia", "frequency": "60% of HCH"},
            {"variant": "c.1948A>G p.Lys650Glu", "effect": "Thanatophoric Dysplasia II (lethal)", "frequency": "common TDII"},
        ],
        "drug_ci": [
            "Growth hormone (not effective in ACH — FGFR3 signalling blocks response; do not use as primary therapy)",
            "Homozygous ACH parents — 25% lethal homozygote risk — prenatal counselling MANDATORY",
        ],
    },
    # ── EXT1 — Hereditary Multiple Exostoses Type 1 ─────────────────────────
    {
        "gene": "EXT1",
        "protein": "Exostosin glycosyltransferase 1",
        "alias": (
            "EXT1; OMIM gene 608177; 8q24.11; 746 aa; HME Type 1 OMIM #133700; "
            "AD LOF; prevalence ~1 in 50,000; MOST COMMON hereditary bone tumour syndrome; "
            "EXT1 mutations cause more severe phenotype than EXT2"
        ),
        "aa": "746 aa",
        "kDa": "~77 kDa",
        "gene_class": (
            "Heparan sulphate (HS) glycosyltransferase — EXT family; "
            "forms obligate heterodimeric complex with EXT2 in ER (EXT1/EXT2 complex); "
            "elongates heparan sulphate chains on proteoglycans (syndecan, glypican); "
            "HS chains regulate Hedgehog (HH), FGF, BMP, WNT signalling gradients in growth plate; "
            "LOF → reduced HS → disorganised growth plate signalling → "
            "osteochondroma (exostosis) formation at physeal cartilage; "
            "two-hit model (Knudson): germline EXT1 LOF + somatic LOF of second allele → exostosis; "
            "EXT1 loss → HS reduction → disrupted Hh gradient → ectopic cartilaginous growth"
        ),
        "locus": "8q24.11",
        "omim_gene": 608177,
        "omim_disease": 133700,
        "phenotype": (
            "Multiple osteochondromas (exostoses) — bony outgrowths capped by cartilage; "
            "arise from growth plates of long bones (distal femur, proximal tibia, proximal humerus most common); "
            "also ribs, vertebrae, scapula, pelvis; "
            "typically begin appearing 2-3 years, usually diagnosed by 12 years; "
            "number: 3-100+ lesions; "
            "SHORT STATURE (mean -1.5 SD); bowing of forearm (radius/ulna); "
            "coxa valga; valgus deformity of knee; ankle deformity; "
            "nerve/vessel compression (peroneal nerve palsy, popliteal artery compression); "
            "MALIGNANT TRANSFORMATION → chondrosarcoma: 2-5% EXT1 (higher than EXT2 1-2%); "
            "transformation signs: lesion growth after skeletal maturity, pain in adult"
        ),
        "hallmark": (
            "MULTIPLE BONY OUTGROWTHS capped by cartilage on X-ray (PATHOGNOMONIC); "
            "FOREARM DEFORMITY (Madelung-like) from differential ulna/radius growth — "
            "most functionally significant deformity; "
            "MALIGNANT TRANSFORMATION WARNING — any lesion growing after skeletal maturity "
            "or becoming painful requires urgent MRI + biopsy; "
            "cartilage cap >2 cm on MRI = malignant transformation suspected"
        ),
        "treatment_alert": (
            "SURGICAL RESECTION of symptomatic lesions (nerve/vessel compression, deformity); "
            "CORRECTIVE OSTEOTOMY for forearm/knee/ankle deformities; "
            "SURVEILLANCE: annual clinical review of all exostoses; "
            "MRI (NOT CT for radiation) for lesions with suspicion of change; "
            "CARTILAGE CAP >2 cm on MRI → biopsy to exclude chondrosarcoma; "
            "NEW/GROWING LESION AFTER SKELETAL MATURITY → urgent imaging + surgical evaluation; "
            "LESION GROWING + PAIN + SIZE >5 cm → chondrosarcoma until proven otherwise; "
            "EXT1 MORE SEVERE than EXT2 — more lesions, more deformity, higher malignancy risk; "
            "GENETIC COUNSELLING: 50% transmission; each child 50% risk"
        ),
        "key_ddx": (
            "EXT2 HME: clinically identical but milder; 11p12; malignancy risk 1-2%; "
            "Solitary osteochondroma: NOT hereditary; only 1 lesion; no family history; "
            "Trevor disease (dysplasia epiphysealis hemimelica): epiphyseal; unilateral; not hereditary; "
            "Ollier disease/Maffucci (enchondromatosis): IDH1/2 mutations; INTRAMEDULLARY lesions; "
            "NOT surface/exostosis; higher malignancy risk 25-50%; "
            "Secondary chondrosarcoma: history of pre-existing osteochondroma; "
            "PRIMARY chondrosarcoma: no previous exostosis; adults; central or peripheral"
        ),
        "gfr_pattern": (
            "Normal renal function; "
            "rarely pelvis/spine exostoses → ureteric compression → hydronephrosis; "
            "chondrosarcoma with renal metastases in advanced malignant transformation (rare)"
        ),
        "proteinuria_pattern": (
            "Not a feature of HME; "
            "NSAIDs for chronic pain → analgesic nephropathy risk; "
            "no intrinsic nephropathy"
        ),
        "primary_complication": (
            "Forearm deformity (most common functional problem); "
            "nerve compression (peroneal palsy, cord compression from vertebral lesions); "
            "malignant transformation to chondrosarcoma (2-5%); "
            "joint deformity; short stature"
        ),
        "disease_detail": (
            "EXT1 and EXT2 encode the two subunits of an obligate heterotetrameric "
            "(2+2) heparan sulphate polymerase complex in the ER. "
            "This complex is the principal enzyme responsible for elongation of "
            "heparan sulphate (HS) chains on cell-surface proteoglycans.\n\n"
            "HS chains form concentration gradients for morphogens (Hedgehog, "
            "FGF, BMP, Wnt) in growth plate cartilage. When EXT1 is inactivated, "
            "HS biosynthesis falls, disrupting these gradients. "
            "A second somatic EXT1 mutation (Knudson two-hit) abolishes "
            "all HS synthesis in a growth plate chondrocyte clone, "
            "resulting in unregulated, ectopic bone+cartilage growth — an osteochondroma.\n\n"
            "EXT1 mutations cause ~60% of HME (EXT1 on 8q24.11) "
            "and produce more severe phenotypes than EXT2 mutations: "
            "more lesions, greater deformity, higher chondrosarcoma risk.\n\n"
            "Malignant transformation to chondrosarcoma occurs in 2-5% of EXT1 HME. "
            "Risk factors: numerous lesions, axial (pelvis/spine) location, adult age, "
            "cartilage cap >2 cm on MRI (normal adult cap ≤2 cm). "
            "Chondrosarcoma in HME is typically low-grade; wide excision is standard of care; "
            "chondrosarcoma does NOT respond to chemotherapy."
        ),
        "inheritance": "Autosomal dominant LOF; penetrance near-complete; expressivity variable",
        "variants": [
            {"variant": "Frameshift/nonsense (truncating)", "effect": "Most HME1 mutations", "frequency": "65%"},
            {"variant": "Missense (glycosyltransferase domain)", "effect": "HME1 — variable severity", "frequency": "15%"},
            {"variant": "Large deletions (8q24.11)", "effect": "Severe — contiguous gene", "frequency": "10%"},
            {"variant": "Splice site", "effect": "LOF", "frequency": "10%"},
        ],
        "drug_ci": [
            "CT surveillance (cumulative radiation) — use MRI preferentially for cartilage cap measurement",
            "Biopsy of lesion without imaging — always MRI first to characterise before biopsy",
        ],
    },
    # ── EXT2 — Hereditary Multiple Exostoses Type 2 ─────────────────────────
    {
        "gene": "EXT2",
        "protein": "Exostosin glycosyltransferase 2",
        "alias": (
            "EXT2; OMIM gene 608210; 11p12-p11; 718 aa; HME Type 2 OMIM #133701; "
            "AD LOF; prevalence: ~40% of HME cases; milder than EXT1 HME"
        ),
        "aa": "718 aa",
        "kDa": "~74 kDa",
        "gene_class": (
            "Heparan sulphate (HS) glycosyltransferase — obligate EXT1/EXT2 heterocomplex; "
            "EXT2 catalyses alternating addition of GlcA and GlcNAc to HS chains; "
            "LOF → EXT1/EXT2 complex non-functional → impaired HS elongation; "
            "same growth plate signalling disruption as EXT1 but generally milder "
            "because EXT2 variants may partially retain activity in some contexts; "
            "two-hit model applies as for EXT1"
        ),
        "locus": "11p12-p11",
        "omim_gene": 608210,
        "omim_disease": 133701,
        "phenotype": (
            "Multiple osteochondromas — clinically similar to EXT1 HME but MILDER: "
            "fewer exostoses; less deformity; shorter stature loss; "
            "forearm involvement less frequent/less severe; "
            "malignancy risk 1-2% (vs 2-5% EXT1); "
            "Fibula/tibia, femur, humerus, ribs; "
            "complications: same as EXT1 but less common; "
            "GENOTYPE-PHENOTYPE: EXT1 variants uniformly more severe than EXT2 — "
            "useful clinically (EXT2 family has milder prognosis)"
        ),
        "hallmark": (
            "Multiple bony exostoses — same as EXT1 but fewer; "
            "GENOTYPE-PHENOTYPE RULE: EXT1 more severe than EXT2 — "
            "useful for prognosis at genetic diagnosis; "
            "MALIGNANT TRANSFORMATION RATE LOWER than EXT1 (1-2% vs 2-5%); "
            "still requires same surveillance; "
            "FOREARM DEFORMITY less frequent in EXT2"
        ),
        "treatment_alert": (
            "SAME SURVEILLANCE as EXT1 — do NOT reduce surveillance based on EXT2 diagnosis; "
            "malignant transformation still occurs (1-2%); "
            "SURGICAL RESECTION of symptomatic or deformity-causing lesions; "
            "CORRECTIVE OSTEOTOMY for forearm/valgus; "
            "MRI for cartilage cap measurement (>2 cm = malignancy concern); "
            "GENETIC COUNSELLING: same 50% transmission; "
            "EXT1/EXT2 mosaic: milder in some; somatic mosaicism detectable by high-depth NGS"
        ),
        "key_ddx": (
            "EXT1 HME: same clinical picture; more lesions; more deformity; higher malignancy; "
            "Solitary osteochondroma: one lesion; no family history; not hereditary; "
            "EXT3 (rare, 19p): very rare; not fully characterised; "
            "EXTL3: rare; EXT family; akin to EXT2 severity"
        ),
        "gfr_pattern": (
            "Normal; same rare ureteric compression as EXT1; "
            "no primary nephropathy"
        ),
        "proteinuria_pattern": (
            "Not a feature; "
            "same considerations as EXT1 for pain management nephropathy"
        ),
        "primary_complication": (
            "Same as EXT1 but milder; malignant transformation (1-2%); "
            "nerve compression; deformity; short stature"
        ),
        "disease_detail": (
            "EXT2 encodes the second subunit of the obligate EXT1/EXT2 "
            "heparan sulphate polymerase complex. EXT2 alone has minimal "
            "enzymatic activity; it requires EXT1 for full HS chain elongation capability. "
            "The complex localises to the ER Golgi where it processively adds "
            "GlcNAc-GlcA disaccharide repeats to the growing HS chain on core proteoglycans.\n\n"
            "EXT2 mutations account for ~40% of hereditary multiple exostoses. "
            "Genotype-phenotype correlations consistently show EXT1 cases have: "
            "more exostoses, greater skeletal deformity, more forearm involvement, "
            "and higher chondrosarcoma risk (2-5%) compared with EXT2 (1-2%).\n\n"
            "The molecular basis of the EXT1 vs EXT2 severity difference is debated. "
            "One hypothesis: EXT1 may have additional HS-independent functions; "
            "alternatively, EXT2 truncating variants may allow some residual EXT2 "
            "protein that partially scaffolds a hypomorphic complex with EXT1.\n\n"
            "Management is identical to EXT1 HME: regular surveillance, "
            "surgical resection of problematic lesions, and MRI-based cartilage cap monitoring."
        ),
        "inheritance": "Autosomal dominant LOF; near-complete penetrance; variable expressivity",
        "variants": [
            {"variant": "Frameshift/nonsense (truncating)", "effect": "Most EXT2 mutations", "frequency": "60%"},
            {"variant": "Missense (glycosyltransferase domain)", "effect": "Variable — often hypomorphic", "frequency": "20%"},
            {"variant": "Large deletions (11p)", "effect": "Contiguous gene syndrome", "frequency": "8%"},
            {"variant": "Splice site", "effect": "LOF", "frequency": "12%"},
        ],
        "drug_ci": [
            "CT preferentially over MRI — avoid cumulative radiation; use MRI for cap measurement",
        ],
    },
    # ── SLC26A2 — Diastrophic Dysplasia / Achondrogenesis IB ────────────────
    {
        "gene": "SLC26A2",
        "protein": "Sulfate transporter SLC26A2 (DTDST)",
        "alias": (
            "SLC26A2 (DTD sulfate transporter); OMIM gene 606718; 5q32; 739 aa; "
            "Diastrophic Dysplasia OMIM #222600; AR; prevalence: DTD ~1 in 100,000 "
            "(enriched in Finland — Finnish founder mutation); "
            "spectrum: Achondrogenesis IB (lethal) → DTD (survivable) → MED (mild) by residual sulphate transport"
        ),
        "aa": "739 aa",
        "kDa": "~82 kDa",
        "gene_class": (
            "Anion exchanger — SLC26 family (sulphate/chloride/bicarbonate transporter); "
            "12 transmembrane domains + STAS domain (C-term regulatory); "
            "expressed in cartilage/chondrocytes: imports sulphate from ECF into cell; "
            "sulphate required for sulphation of proteoglycans (aggrecan, decorin, biglycan) "
            "and glycosaminoglycans (chondroitin sulphate, keratan sulphate) in cartilage matrix; "
            "SLC26A2 LOF → undersulphated proteoglycans → disorganised cartilage matrix → "
            "impaired endochondral ossification → short limbs, skeletal dysplasia"
        ),
        "locus": "5q32",
        "omim_gene": 606718,
        "omim_disease": 222600,
        "phenotype": (
            "Diastrophic Dysplasia (DTD — most common survivable form): "
            "Severe rhizomelic short stature; club feet (talipes equinovarus — PATHOGNOMONIC, "
            "very severe, resist serial casting); "
            "'hitchhiker thumb' (abducted short thumb — PATHOGNOMONIC for DTD); "
            "cauliflower ear (pinnae calcification after birth — 30-40% — PATHOGNOMONIC for DTD); "
            "scoliosis (progressive, severe, 40-60%); "
            "cervical kyphosis (risk of cord compression — fatal if severe); "
            "joint contractures; cleft palate (25%); "
            "normal intelligence; lifespan near-normal with intervention; "
            "adult height 110-130 cm; "
            "Achondrogenesis IB (null alleles): perinatal lethal — no endochondral ossification"
        ),
        "hallmark": (
            "HITCHHIKER THUMB (proximally placed, short, wide, abducted) — PATHOGNOMONIC for DTD; "
            "CAULIFLOWER EAR (pinnal swelling → haematoma → calcification at birth/neonatal) — "
            "PATHOGNOMONIC; DO NOT ASPIRATE the pinnal swelling — high infection risk; "
            "CLUB FOOT MOST SEVERE of any skeletal dysplasia — resist standard casting; "
            "CERVICAL KYPHOSIS — risk of cord compression → death; MRI mandatory"
        ),
        "treatment_alert": (
            "CLUB FEET: early Ponseti method (modified) + serial casting + tendon Achilles release; "
            "MOST SEVERE club feet in any skeletal dysplasia — surgical correction usually required; "
            "CERVICAL KYPHOSIS: MRI spine in all DTD infants; "
            "surgical fusion if kyphosis >60° or cord signal change; "
            "CAULIFLOWER EAR: DO NOT ASPIRATE (high infection/chondritis risk); "
            "treat conservatively with ice compression ONLY; "
            "SCOLIOSIS: early bracing, spinal fusion for curves >50°; "
            "HEARING AIDS: conductive hearing loss from middle ear malformation + cauliflower ear; "
            "CLEFT PALATE: multidisciplinary team; speech therapy; "
            "PRENATAL DIAGNOSIS: ultrasound shows hitchhiker thumb + club feet from 14-16 weeks"
        ),
        "key_ddx": (
            "DTD vs Achondroplasia: ACH macrocephaly, G380R, trident hand (NOT hitchhiker); "
            "DTD vs Metatropic Dysplasia: TRPV4 GOF; severe scoliosis like DTD but NO hitchhiker; "
            "DTD vs Larsen syndrome: multiple dislocations; normal thumb; FLNB gene; "
            "ACG IB (null SLC26A2): lethal — similar genes but different outcome based on residual activity; "
            "Camptomelic dysplasia: SOX9 gene; bowed tibiae; sex reversal; "
            "Atelosteogenesis: FLNB gene; pulmonary hypoplasia"
        ),
        "gfr_pattern": (
            "Normal; "
            "urinary sulphate wasting not clinically significant for kidneys; "
            "chronic NSAID use for pain → analgesic nephropathy risk"
        ),
        "proteinuria_pattern": (
            "Not a feature; "
            "no collagen IV defect; no nephropathy from SLC26A2"
        ),
        "primary_complication": (
            "Club feet (most limiting early); cervical cord compression; "
            "progressive scoliosis; conductive hearing loss; "
            "respiratory compromise (severe thoracic/cervical deformity)"
        ),
        "disease_detail": (
            "SLC26A2 (DTDST) is the principal sulphate transporter in cartilage. "
            "It operates as an electroneutral anion exchanger, importing sulphate (SO4²⁻) "
            "in exchange for chloride across the chondrocyte plasma membrane. "
            "The imported sulphate is then activated to 3'-phosphoadenosine-5'-phosphosulphate (PAPS), "
            "the universal sulphate donor for proteoglycan sulphation.\n\n"
            "Proteoglycans in cartilage (aggrecan with its chondroitin sulphate side chains) "
            "require sulphation to maintain their highly charged, water-retaining structure. "
            "Undersulphated aggrecan → cartilage matrix disorganisation → impaired growth plate "
            "architecture → failed endochondral ossification.\n\n"
            "The DTD spectrum is governed by residual sulphate transport activity:\n"
            "- Achondrogenesis IB (ACG1B): two null alleles → 0% residual activity → lethal\n"
            "- Atelosteogenesis II (AOII): severe, may die at/after birth\n"
            "- Diastrophic Dysplasia: compound het (null + hypomorphic) or two hypomorphic → survivable\n"
            "- Multiple Epiphyseal Dysplasia (recessive DTDST-type): mildest\n\n"
            "The Finnish founder allele (c.835-2A>G, deep intronic IVS1+2T→G) "
            "is the most common DTD-causing variant in Finland (carrier frequency 1 in 70).\n\n"
            "The hitchhiker thumb in DTD results from hypoplasia of the first metacarpal "
            "and abnormal proximal placement of the thumb — clinically distinctive."
        ),
        "inheritance": "Autosomal recessive; Finnish/European enriched (founder allele c.835-2A>G)",
        "variants": [
            {"variant": "c.835-2A>G (IVS1+2T→G) Finnish founder", "effect": "DTD — most Finnish cases", "frequency": "common in Finland"},
            {"variant": "c.1957T>A p.Cys653Ser", "effect": "ACG1B (lethal)", "frequency": "severe allele"},
            {"variant": "p.Arg178X", "effect": "Null — ACG1B", "frequency": "null allele"},
            {"variant": "p.Gly255Glu", "effect": "DTD moderate", "frequency": "European"},
        ],
        "drug_ci": [
            "Cauliflower ear aspiration — CONTRAINDICATED (high chondritis infection risk)",
            "Standard Ponseti casting alone insufficient for DTD club feet — always plan surgical backup",
        ],
    },
    # ── RMRP — Cartilage-Hair Hypoplasia ────────────────────────────────────
    {
        "gene": "RMRP",
        "protein": "RNA component of mitochondrial RNA processing endoribonuclease (RNase MRP)",
        "alias": (
            "RMRP; OMIM gene 157660; 9p13.3; Cartilage-Hair Hypoplasia OMIM #250250; "
            "AR; prevalence: CHH ~1 in 20,000 (Old Order Amish); "
            "Finnish: ~1 in 23,000; enriched in Amish/Finnish; "
            "RNA gene — NOT protein-coding; variants in non-coding RNA gene"
        ),
        "aa": "N/A — RNA gene (267 nucleotides)",
        "kDa": "N/A (RNA component)",
        "gene_class": (
            "Non-coding RNA gene — RMRP encodes the RNA subunit of RNase MRP (ribonucleoprotein complex); "
            "RNase MRP functions: "
            "(1) processes 5.8S rRNA in nucleolus (ribosome biogenesis); "
            "(2) cleaves mitochondrial RNA primer for mtDNA replication; "
            "(3) cell-cycle regulation (Cdc13 mRNA cleavage); "
            "LOF → impaired ribosome biogenesis → reduced cell proliferation; "
            "cartilage is particularly sensitive (rapidly proliferating chondrocytes); "
            "also affects lymphocyte proliferation (immune deficiency) and skin proliferation (thin hair); "
            "ALL CHH variants are in regulatory regions of RMRP (promoter, 5'UTR, internal loops) — "
            "NOT in the structural RNA sequence"
        ),
        "locus": "9p13.3",
        "omim_gene": 157660,
        "omim_disease": 250250,
        "phenotype": (
            "Cartilage-Hair Hypoplasia (CHH): "
            "SHORT-LIMB DWARFISM (rhizomelic) — milder than achondroplasia (adult height 102-145 cm); "
            "FINE LIGHT HAIR (hypoplastic — reduced diameter, less pigmented) — PATHOGNOMONIC; "
            "immune deficiency: T-cell lymphopenia (CD4/CD8 cells reduced); "
            "NK cell dysfunction; combined T+B cell deficiency possible (SCID-like in most severe); "
            "Hirschsprung disease (10%) — aganglionosis colon; "
            "LYMPHOMA RISK (10x general population) — non-Hodgkin lymphoma in 3rd-5th decade; "
            "anaemia (macrocytic) from marrow progenitor failure; "
            "NORMAL INTELLIGENCE"
        ),
        "hallmark": (
            "FINE HYPOPLASTIC LIGHT HAIR + SHORT-LIMB DWARFISM — PATHOGNOMONIC combination for CHH; "
            "IMMUNE DEFICIENCY — vaccinations: avoid live vaccines when immune-deficient; "
            "LYMPHOMA (10% lifetime) — annual lymphoma surveillance mandatory from age 20; "
            "HIRSCHSPRUNG (10%) — exclude in any CHH neonate with delayed meconium; "
            "ALL VARIANTS IN RMRP REGULATORY REGIONS — structural RNA region variants are benign"
        ),
        "treatment_alert": (
            "LIVE VACCINES — CONTRAINDICATED in CHH with T-cell lymphopenia (check immune status first); "
            "measure lymphocyte subsets (CD4, CD8, NK, B-cells) before ANY vaccination; "
            "BONE MARROW TRANSPLANT (HSCT) — indicated for SCID-like severe immune deficiency; "
            "corrects immune defect, does NOT correct skeletal manifestations; "
            "IVIG — for hypogammaglobulinaemia; "
            "PROPHYLACTIC ANTIBIOTIC + ANTIFUNGAL — for severe combined immunodeficiency; "
            "LYMPHOMA SURVEILLANCE: annual clinical exam + LDH from age 20; CT/PET if clinical suspicion; "
            "HIRSCHSPRUNG: early diagnosis and surgical pull-through; "
            "VARICELLA ZOSTER: acyclovir prophylaxis if CD4 < 200; "
            "DO NOT IGNORE HAIR FINDING — it is clinically diagnostic (hair shaft microscopy)"
        ),
        "key_ddx": (
            "McKusick-Kaufman (MKKS gene): short stature + heart defect + polydactyly; no hair finding; "
            "Achondroplasia: macrocephaly; G380R FGFR3; normal hair; NO immune deficiency; "
            "Metaphyseal chondrodysplasia Schmid type: COL10A1; milder; no hair/immune findings; "
            "Shwachman-Diamond syndrome: SBDS gene; exocrine pancreatic failure + neutropenia; "
            "Adenosine deaminase deficiency (ADA-SCID): SCID without skeletal features; "
            "Reticular dysgenesis: most severe SCID; RAG1/2 ADA; no skeletal features"
        ),
        "gfr_pattern": (
            "Normal GFR; "
            "immunosuppression (post-HSCT) → calcineurin inhibitor nephrotoxicity; "
            "recurrent infections → chronic pyelonephritis risk in immune-deficient patients"
        ),
        "proteinuria_pattern": (
            "Not a primary feature; "
            "proteinuria possible from frequent infections; "
            "monitor post-HSCT (cyclosporine/tacrolimus nephrotoxicity)"
        ),
        "primary_complication": (
            "Opportunistic infections (T-cell deficiency); lymphoma (3rd-5th decade); "
            "Hirschsprung disease (10%); anaemia; short stature"
        ),
        "disease_detail": (
            "RMRP is unique among skeletal dysplasia genes: it encodes an RNA, "
            "not a protein. The RNA component of RNase MRP (a ribonucleoprotein) "
            "partners with protein subunits (RNGTT, POP1, POP4, etc.) to form the "
            "active endoribonuclease complex.\n\n"
            "RNase MRP cleaves the internal transcribed spacer 1 (ITS1) of 45S pre-rRNA, "
            "a required step in processing the 5.8S ribosomal RNA. "
            "Impaired rRNA processing → reduced ribosome biogenesis → "
            "reduced proliferative capacity in cells that divide rapidly. "
            "Cartilage growth plate chondrocytes are among the most rapidly dividing cells, "
            "explaining why CHH predominantly affects skeletal growth.\n\n"
            "The immune deficiency arises because T-cell and NK-cell progenitors "
            "also require rapid proliferation. The result is a combined "
            "immunodeficiency with T-cell lymphopenia, NK dysfunction, and in severe cases, "
            "a SCID-like phenotype requiring HSCT.\n\n"
            "The hair is affected because hair follicle matrix cells — the most rapidly "
            "dividing cells in the body — also depend on ribosome biogenesis. "
            "Hair shafts are thin, brittle, and hypopigmented.\n\n"
            "ALL known pathogenic RMRP variants affect regulatory regions "
            "(promoter, 5'UTR, internal RNA loops), NOT the structural regions "
            "of the RNA. This is consistent with hypomorphic reduction of RNase MRP "
            "activity rather than complete loss.\n\n"
            "The 70A>G variant in the 5'UTR is the predominant Amish founder variant. "
            "Finnish cases predominantly carry the g.71A>G variant."
        ),
        "inheritance": "Autosomal recessive; Finnish founder (g.71A>G); Amish founder (70A>G)",
        "variants": [
            {"variant": "g.70A>G (5'UTR)", "effect": "CHH — Amish founder most common", "frequency": "most Amish cases"},
            {"variant": "g.71A>G (5'UTR)", "effect": "CHH — Finnish founder", "frequency": "most Finnish cases"},
            {"variant": "Promoter variants (various)", "effect": "CHH — reduced RMRP transcription", "frequency": "European non-Finnish"},
            {"variant": "Internal loop variants (P3, P4 domain)", "effect": "Severe CHH/immune deficiency", "frequency": "rare"},
        ],
        "drug_ci": [
            "Live vaccines (MMR, varicella, yellow fever, rotavirus) — CONTRAINDICATED in CHH with T-cell lymphopenia; check CD4 count first",
            "Post-HSCT calcineurin inhibitors — monitor creatinine monthly (nephrotoxicity)",
        ],
    },
    # ── COMP — Pseudoachondroplasia / Multiple Epiphyseal Dysplasia ──────────
    {
        "gene": "COMP",
        "protein": "Cartilage oligomeric matrix protein",
        "alias": (
            "COMP; OMIM gene 600310; 19p13.11; 757 aa; "
            "Pseudoachondroplasia OMIM #177170; MED OMIM #132400; "
            "AD; prevalence: PSACH ~1 in 30,000; MED ~1 in 10,000; "
            "~90% PSACH cases are de novo"
        ),
        "aa": "757 aa",
        "kDa": "~83 kDa (monomer); 524 kDa (pentamer)",
        "gene_class": (
            "Thrombospondin family member — non-collagenous extracellular matrix protein; "
            "forms disulphide-linked PENTAMER in RER; "
            "5 monomers → star-shaped pentameric complex; "
            "binds collagens I, II, IX, XII; matrilins; fibronectin; "
            "key bridging molecule in cartilage ECM (tethers collagen fibres); "
            "PSACH mutations — primarily in the type III calcium-binding repeat (T3) domain → "
            "ER retention of COMP pentamer → ER stress in chondrocytes → apoptosis; "
            "MED mutations — primarily in EGF-like or C-terminal domain → less ER retention; "
            "NORMAL at birth — diagnosis typically delayed until age 2-3 when walking begins"
        ),
        "locus": "19p13.11",
        "omim_gene": 600310,
        "omim_disease": 177170,
        "phenotype": (
            "Pseudoachondroplasia (PSACH — most severe COMP phenotype): "
            "NORMAL AT BIRTH — diagnosed at age 2-3 when walking delayed/abnormal; "
            "NORMAL FACE AND HEAD — distinguishes from achondroplasia; "
            "Severe short stature (adult height 82-130 cm); "
            "ligamentous laxity + joint instability (knees, hips, cervical spine); "
            "C1-C2 instability (odontoid hypoplasia) → cord compression risk; "
            "early-onset arthritis; scoliosis; lordosis; "
            "Multiple Epiphyseal Dysplasia (MED — milder): "
            "NORMAL STATURE or mild shortening; joint pain from childhood; "
            "delayed epiphyseal ossification on X-ray; early OA hip/knee"
        ),
        "hallmark": (
            "PSACH: NORMAL FACE + NORMAL HEAD CIRCUMFERENCE + SHORT LIMBS — "
            "DISTINGUISHES from achondroplasia (which has macrocephaly); "
            "NORMAL AT BIRTH — LATE DIAGNOSIS (age 2-3); "
            "ODONTOID HYPOPLASIA → C1-C2 INSTABILITY → cord compression; "
            "lateral cervical spine X-ray in ALL PSACH patients (flexion-extension views); "
            "LIGAMENTOUS LAXITY with instability — contradicts short stature; "
            "ER RETENTION of mutant COMP pentamer = pathomechanism"
        ),
        "treatment_alert": (
            "CERVICAL SPINE INSTABILITY SURVEILLANCE: "
            "flexion-extension lateral X-ray of cervical spine annually in PSACH; "
            "MRI cervical spine if cord symptoms (weakness, gait change, handwriting); "
            "C1-C2 FUSION if instability or cord compression — URGENT; "
            "ANAESTHESIA PRECAUTIONS: cervical instability → extreme care with intubation; "
            "inform anaesthetist of diagnosis before any surgery; "
            "EARLY PHYSIOTHERAPY: joint stability exercises; "
            "AVOID CONTACT SPORTS: risk of cervical cord injury; "
            "JOINT REPLACEMENT (hip/knee): often early (30s-40s) due to severe OA; "
            "SCOLIOSIS: bracing/surgery as standard; "
            "PAMIDRONATE: not established benefit in PSACH; "
            "GROWTH HORMONE: no significant benefit (ER stress pathway, not FGFR3)"
        ),
        "key_ddx": (
            "Achondroplasia (FGFR3 G380R): macrocephaly at birth; trident hand; diagnosed at birth; "
            "MED vs PSACH: PSACH = short stature; MED = normal stature; both COMP; "
            "SEDC (Spondyloepiphyseal Dysplasia Congenita): COL2A1; spine involved at birth; "
            "Morquio A (MPS IVA): GALNS enzyme; "
            "dental involvement + corneal clouding in severe; "
            "Diastrophic Dysplasia: hitchhiker thumb; club feet; SLC26A2; "
            "JIA (juvenile idiopathic arthritis): normal stature; no radiographic epiphyseal change"
        ),
        "gfr_pattern": (
            "Normal; "
            "chronic NSAID use for early arthritis → analgesic nephropathy risk; "
            "no primary nephropathy from COMP"
        ),
        "proteinuria_pattern": (
            "Not a feature; "
            "chronic pain management → monitor renal function annually"
        ),
        "primary_complication": (
            "C1-C2 instability (cervical cord compression — can be fatal); "
            "early-onset osteoarthritis; short stature; scoliosis; "
            "anaesthesia risk from cervical instability"
        ),
        "disease_detail": (
            "COMP (cartilage oligomeric matrix protein) is a pentameric glycoprotein "
            "that bridges collagen fibrils in the cartilage extracellular matrix. "
            "Each pentamer consists of five identical subunits linked by N-terminal "
            "coiled-coil disulphide bonds, forming a star-shaped bouquet structure.\n\n"
            "Pathogenic COMP variants (primarily missense in the type III "
            "calcium-binding repeat domain) produce misfolded monomers that cannot "
            "form normal pentamers. These are retained in the rough ER, "
            "activating the unfolded protein response (UPR). "
            "Chronically activated UPR → ER stress → chondrocyte apoptosis → "
            "growth plate disorganisation → progressive skeletal dysplasia.\n\n"
            "The key clinical teaching point: PSACH is NORMAL AT BIRTH. "
            "The diagnosis is typically delayed to age 2-3 when gait abnormality "
            "prompts evaluation. This contrasts with achondroplasia (diagnosed at birth "
            "by macrocephaly and rhizomelia) and OI (fractures at birth).\n\n"
            "The normal face and head circumference in PSACH (vs macrocephaly in ACH) "
            "is the critical distinguishing examination finding. "
            "Many families experience a 1-2 year diagnostic odyssey before PSACH is considered.\n\n"
            "MED (multiple epiphyseal dysplasia), the milder COMP phenotype, "
            "presents with joint pain and early osteoarthritis in childhood/adolescence, "
            "often misdiagnosed as JIA. The epiphyseal ossification delay on X-ray "
            "('delayed, small, irregular epiphyses') is the radiological clue."
        ),
        "inheritance": "Autosomal dominant; ~90% PSACH de novo (new mutation); MED can be inherited",
        "variants": [
            {"variant": "p.Asp469 (T3 repeat mutations)", "effect": "PSACH — severe (ER retention)", "frequency": "most PSACH"},
            {"variant": "p.Thr508 (T3 repeat)", "effect": "PSACH — ER stress pathway", "frequency": "common PSACH"},
            {"variant": "EGF domain missense", "effect": "MED — milder (less ER retention)", "frequency": "MED cases"},
            {"variant": "C-terminal domain missense", "effect": "MED — minimal ER retention", "frequency": "MED cases"},
        ],
        "drug_ci": [
            "Contact sports (PSACH) — ABSOLUTELY CONTRAINDICATED due to C1-C2 instability risk of cord injury",
            "Anaesthesia without cervical precautions — alert anaesthetist of PSACH/COMP diagnosis before any procedure",
        ],
    },
]


def _make_cohort(gene: dict, seed: int, n: int = 40) -> list:
    """Generate a deterministic synthetic patient cohort for a skeletal dysplasia gene."""
    rng = random.Random(seed)
    gene_id = gene["gene"]
    patients = []

    for i in range(n):
        # Base demographics
        age = rng.randint(4, 65)
        sex = rng.choice(["M", "F"])

        # Gene-specific clinical profiles
        if gene_id == "COL1A1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[60, 30, 10])[0]
            blue_sclerae = rng.random() < 0.92
            fractures_lifetime = rng.randint(3, 40) if severity != "Severe" else rng.randint(20, 80)
            hearing_loss = rng.random() < (0.5 if age > 30 else 0.15)
            di = rng.random() < 0.25
            wormian_bones = rng.random() < 0.78
            bisphosphonate_tx = severity in ("Moderate", "Severe") and rng.random() < 0.80
            drug_error = rng.random() < 0.10
            dx_delayed = rng.random() < 0.25
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.15
            transplant = False
        elif gene_id == "COL1A2":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[15, 35, 50])[0]
            blue_sclerae = rng.random() < 0.60
            fractures_lifetime = rng.randint(15, 100)
            hearing_loss = rng.random() < 0.55
            di = rng.random() < 0.50
            wormian_bones = rng.random() < 0.78
            bisphosphonate_tx = rng.random() < 0.90
            drug_error = rng.random() < 0.12
            dx_delayed = rng.random() < 0.20
            esrd = rng.random() < 0.03
            htn = rng.random() < 0.20
            transplant = False
        elif gene_id == "FGFR3":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[5, 60, 35])[0]
            blue_sclerae = False
            fractures_lifetime = rng.randint(0, 2)
            hearing_loss = rng.random() < 0.50
            di = False
            wormian_bones = False
            bisphosphonate_tx = False
            drug_error = rng.random() < 0.15  # GH use in ACH
            dx_delayed = rng.random() < 0.10
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.20
            transplant = False
        elif gene_id == "EXT1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[20, 55, 25])[0]
            blue_sclerae = False
            fractures_lifetime = rng.randint(0, 5)
            hearing_loss = False
            di = False
            wormian_bones = False
            bisphosphonate_tx = False
            malignant_transform = rng.random() < 0.04
            forearm_deformity = rng.random() < 0.55
            drug_error = rng.random() < 0.08
            dx_delayed = rng.random() < 0.20
            esrd = rng.random() < 0.01
            htn = rng.random() < 0.10
            transplant = False
        elif gene_id == "EXT2":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[35, 50, 15])[0]
            blue_sclerae = False
            fractures_lifetime = rng.randint(0, 3)
            hearing_loss = False
            di = False
            wormian_bones = False
            bisphosphonate_tx = False
            malignant_transform = rng.random() < 0.015
            forearm_deformity = rng.random() < 0.35
            drug_error = rng.random() < 0.07
            dx_delayed = rng.random() < 0.22
            esrd = rng.random() < 0.01
            htn = rng.random() < 0.10
            transplant = False
        elif gene_id == "SLC26A2":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[10, 45, 45])[0]
            blue_sclerae = False
            fractures_lifetime = rng.randint(0, 3)
            hearing_loss = rng.random() < 0.40
            di = False
            wormian_bones = False
            bisphosphonate_tx = False
            hitchhiker_thumb = rng.random() < 0.95
            cauliflower_ear = rng.random() < 0.35
            club_foot = rng.random() < 0.95
            cervical_kyphosis = rng.random() < 0.45
            drug_error = rng.random() < 0.18  # cauliflower ear aspiration
            dx_delayed = rng.random() < 0.30
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.12
            transplant = False
        elif gene_id == "RMRP":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[20, 50, 30])[0]
            blue_sclerae = False
            fractures_lifetime = rng.randint(0, 2)
            hearing_loss = rng.random() < 0.35
            di = False
            wormian_bones = False
            bisphosphonate_tx = False
            immune_deficiency = rng.random() < 0.85
            hirschsprung = rng.random() < 0.10
            lymphoma = rng.random() < (0.10 if age > 30 else 0.02)
            live_vaccine_error = rng.random() < 0.25  # given live vaccine when immune-deficient
            drug_error = live_vaccine_error
            dx_delayed = rng.random() < 0.40
            esrd = rng.random() < 0.02
            htn = rng.random() < 0.10
            transplant = rng.random() < (0.15 if severity == "Severe" else 0.03)
        elif gene_id == "COMP":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[15, 45, 40])[0]
            blue_sclerae = False
            fractures_lifetime = rng.randint(0, 3)
            hearing_loss = False
            di = False
            wormian_bones = False
            bisphosphonate_tx = False
            cervical_instability = rng.random() < 0.70
            early_oa = rng.random() < 0.75
            anaes_error = rng.random() < 0.20
            drug_error = anaes_error
            dx_delayed = rng.random() < 0.55  # commonly misdiagnosed for years
            esrd = rng.random() < 0.01
            htn = rng.random() < 0.15
            transplant = False

        # Common computed fields
        surveillance_adherent = rng.random() < 0.65
        adult_height_cm = {
            "COL1A1": rng.randint(145, 175),
            "COL1A2": rng.randint(85, 130),
            "FGFR3":  rng.randint(118, 145),
            "EXT1":   rng.randint(152, 175),
            "EXT2":   rng.randint(155, 178),
            "SLC26A2": rng.randint(110, 130),
            "RMRP":   rng.randint(102, 145),
            "COMP":   rng.randint(82, 145),
        }.get(gene_id, 160)

        p = {
            "id": f"{gene_id}-{i+1:03d}",
            "gene": gene_id,
            "age": age,
            "sex": sex,
            "severity": severity,
            "adult_height_cm": adult_height_cm,
            "fractures_lifetime": fractures_lifetime,
            "blue_sclerae": blue_sclerae,
            "hearing_loss": hearing_loss,
            "dentinogenesis_imperfecta": di,
            "wormian_bones": wormian_bones,
            "esrd": esrd,
            "hypertension": htn,
            "transplant": transplant,
            "drug_error": drug_error,
            "dx_delayed": dx_delayed,
            "surveillance_adherent": surveillance_adherent,
        }
        patients.append(p)

    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    def pct(key):
        return round(sum(1 for p in patients if p.get(key)) / n * 100, 1)
    sev = {s: round(sum(1 for p in patients if p["severity"] == s) / n * 100, 1)
           for s in ["Mild", "Moderate", "Severe"]}
    return {
        "n": n,
        "esrd_pct": pct("esrd"),
        "htn_pct": pct("hypertension"),
        "transplant_pct": pct("transplant"),
        "drug_error_pct": pct("drug_error"),
        "dx_delayed_pct": pct("dx_delayed"),
        "surveillance_adherent_pct": pct("surveillance_adherent"),
        "severity": sev,
    }


def _build_all_patients():
    all_patients = []
    for idx, gene in enumerate(SKELETAL_DYSPLASIA_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(gene, seed=seed, n=40)
        all_patients.extend(cohort)
    return all_patients


ALL_PATIENTS = _build_all_patients()


def get_overview() -> dict:
    n = len(ALL_PATIENTS)
    agg = _cohort_stats(ALL_PATIENTS)
    return {
        "atlas_name": "Skeletal Dysplasia Atlas",
        "atlas_subtitle": (
            "Complete 8-Gene Hereditary Skeletal Dysplasia Atlas — "
            "COL1A1 · COL1A2 · FGFR3 · EXT1 · EXT2 · SLC26A2 · RMRP · COMP"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1182–1189",
        "description": (
            "Comprehensive hereditary skeletal dysplasia reference covering the 8 most clinically "
            "significant monogenic skeletal disorders: "
            "OI Type I (COL1A1 — haploinsufficiency; blue sclerae PATHOGNOMONIC; bisphosphonates); "
            "OI Type III/IV (COL1A2 — structural Gly substitution; basilar invagination surveillance mandatory); "
            "Achondroplasia (FGFR3 GOF — G380R 95%; vosoritide FDA 2021; GH NOT effective; "
            "foramen magnum stenosis life-threatening); "
            "HME Type 1 (EXT1 LOF — heparan sulphate; most lesions; 2-5% chondrosarcoma; "
            "cap >2 cm = malignancy); "
            "HME Type 2 (EXT2 LOF — milder; 1-2% malignancy; same surveillance as EXT1); "
            "Diastrophic Dysplasia (SLC26A2 AR — hitchhiker thumb + cauliflower ear PATHOGNOMONIC; "
            "do NOT aspirate pinnae); "
            "CHH (RMRP AR RNA gene — fine hair + immune deficiency + lymphoma 10%; "
            "live vaccines CI when T-cell depleted); "
            "PSACH/MED (COMP AD — normal at birth; normal face distinguishes from ACH; "
            "cervical instability mandatory surveillance; contact sports CI)"
        ),
        "aggregate_clinical": {
            "esrd_pct": agg.get("esrd_pct", 0),
            "hypertension_pct": agg.get("htn_pct", 0),
            "transplant_rate_pct": agg.get("transplant_pct", 0),
            "drug_error_pct": agg.get("drug_error_pct", 0),
            "diagnosis_delayed_pct": agg.get("dx_delayed_pct", 0),
            "surveillance_adherent_pct": agg.get("surveillance_adherent_pct", 0),
            "severity_mild_pct": agg.get("severity", {}).get("Mild", 0),
            "severity_moderate_pct": agg.get("severity", {}).get("Moderate", 0),
            "severity_severe_pct": agg.get("severity", {}).get("Severe", 0),
        },
        "drug_alerts": [
            {
                "type": "danger",
                "title": "FGFR3-ACH: Growth Hormone NOT Effective — Do Not Use as Primary Therapy",
                "body": (
                    "In achondroplasia (FGFR3 G380R GOF), growth hormone does not significantly "
                    "increase final adult height. FGFR3 constitutive signalling in the growth plate "
                    "is downstream of GH/IGF-1 axis. Vosoritide (BMN111, FDA 2021) targets the "
                    "FGFR3 pathway directly and is the appropriate disease-modifying therapy. "
                    "GH should not be started without specialist discussion."
                ),
            },
            {
                "type": "danger",
                "title": "COMP-PSACH: Contact Sports ABSOLUTELY CONTRAINDICATED — C1-C2 Instability",
                "body": (
                    "Pseudoachondroplasia patients have odontoid hypoplasia and C1-C2 ligamentous laxity. "
                    "Axial loading or cervical hyperflexion during contact sports (rugby, football, gymnastics) "
                    "risks catastrophic cervical cord injury and tetraplegia. Annual lateral cervical "
                    "X-ray (flexion-extension views) and anaesthesia alert mandatory for all PSACH patients."
                ),
            },
            {
                "type": "danger",
                "title": "SLC26A2-DTD: Cauliflower Ear — DO NOT ASPIRATE (Chondritis Risk)",
                "body": (
                    "The neonatal/infantile pinnal swelling (haematoma-like) in Diastrophic Dysplasia "
                    "must NOT be aspirated. Aspiration carries a very high risk of introducing infection "
                    "leading to suppurative chondritis and permanent ear deformity. Management is "
                    "conservative — ice compression only. The swelling calcifies and creates the "
                    "pathognomonic 'cauliflower ear' appearance."
                ),
            },
            {
                "type": "danger",
                "title": "RMRP-CHH: Live Vaccines CONTRAINDICATED with T-Cell Lymphopenia",
                "body": (
                    "Cartilage-Hair Hypoplasia causes T-cell lymphopenia with NK dysfunction. "
                    "Live attenuated vaccines (MMR, varicella, BCG, yellow fever, rotavirus) "
                    "can cause disseminated vaccine-strain infection and death. "
                    "Always measure lymphocyte subsets (CD4, CD8, NK, B-cells) before "
                    "ANY vaccination in CHH. HSCT corrects the immune defect when severe."
                ),
            },
            {
                "type": "warning",
                "title": "COL1A2-OI: Distinguish from Hypophosphatasia — Bisphosphonates WORSEN HPP",
                "body": (
                    "Severe OI (multiple fractures, blue/white sclerae) can mimic hypophosphatasia (HPP). "
                    "CRITICAL DDx: measure serum alkaline phosphatase (ALP). "
                    "In HPP, ALP is LOW (pathognomonic). If OI is mistaken for HPP and "
                    "bisphosphonates are given, this WORSENS HPP by blocking osteoclast activity "
                    "which already cannot remineralise bones properly. ALP MUST be checked before "
                    "starting bisphosphonates in any child with multiple fractures."
                ),
            },
            {
                "type": "warning",
                "title": "FGFR3-ACH: Foramen Magnum Stenosis — Screen ALL Infants (Brain MRI 6-12 months)",
                "body": (
                    "All achondroplasia infants must have brain MRI with foramen magnum measurements "
                    "at 6-12 months. Foramen magnum AP diameter <3.5 mm = stenosis. "
                    "Untreated: medullary compression → central apnoea → sudden infant death. "
                    "Signs: hyperreflexia, clonus, arm/leg weakness, apnoea, failure to thrive. "
                    "Surgical decompression (C1 laminectomy) is indicated for significant stenosis."
                ),
            },
        ],
        "critical_rules": [
            "COL1A1/COL1A2-OI: Check ALP before bisphosphonates — LOW ALP = hypophosphatasia (DO NOT give bisphosphonates in HPP)",
            "FGFR3-ACH: Growth hormone is NOT effective — do not start without specialist review; use vosoritide FDA 2021",
            "FGFR3-ACH: Brain MRI at 6-12 months MANDATORY — foramen magnum stenosis → central apnoea → sudden death",
            "COMP-PSACH: Contact sports ABSOLUTELY CI — cervical C1-C2 instability; anaesthesia alert mandatory",
            "COMP-PSACH: NORMAL AT BIRTH — diagnosis delayed to age 2-3; normal face distinguishes from ACH",
            "SLC26A2-DTD: Cauliflower ear — DO NOT ASPIRATE (chondritis); conservative ice compression only",
            "RMRP-CHH: Live vaccines CI with T-cell lymphopenia — check CD4 count BEFORE vaccination",
            "RMRP-CHH: Lymphoma surveillance mandatory annually from age 20 (10% lifetime risk)",
            "EXT1/EXT2: Cartilage cap >2 cm on MRI = malignant transformation suspected — urgent biopsy",
            "EXT1/EXT2: Lesion growing after skeletal maturity + pain = chondrosarcoma until proven otherwise",
        ],
        "pathway_targets": {
            "COL1A1": "Type I collagen (haploinsufficiency → bisphosphonates + telescoping rods)",
            "COL1A2": "Type I collagen structural (Gly sub → ER misfolding → bisphosphonates + rods)",
            "FGFR3":  "FGFR3 GOF (constitutive kinase → CNP analogue vosoritide bypasses FGFR3)",
            "EXT1":   "Heparan sulphate synthesis (LOF → osteochondroma; surveillance + surgical resection)",
            "EXT2":   "Heparan sulphate synthesis (LOF → osteochondroma milder; same surveillance as EXT1)",
            "SLC26A2": "Sulphate transporter (undersulphated proteoglycan → supportive; corrective surgery)",
            "RMRP":   "RNase MRP RNA (ribosome biogenesis impaired → HSCT for SCID; lymphoma surveillance)",
            "COMP":   "Cartilage ECM bridging (ER retention → cervical surveillance; early joint replacement)",
        },
        "kpis": [
            {"label": "Total Patients", "value": str(n)},
            {"label": "Genes Covered", "value": "8"},
            {"label": "Cohort Seeds", "value": "1182–1189"},
            {"label": "Drug Errors (Agg)", "value": f"{agg.get('drug_error_pct', 0)}%"},
            {"label": "Dx Delayed (Agg)", "value": f"{agg.get('dx_delayed_pct', 0)}%"},
            {"label": "Surveillance Adherent", "value": f"{agg.get('surveillance_adherent_pct', 0)}%"},
        ],
        "disease_category_breakdown": {
            "Collagenopathy (OI)": 25.0,
            "FGFR3 GOF (Achondroplasia)": 12.5,
            "HS Biosynthesis (HME)": 25.0,
            "Sulphate Transport (DTD)": 12.5,
            "RNase MRP (CHH)": 12.5,
            "ECM Bridging (PSACH/MED)": 12.5,
        },
    }


def get_breakdown() -> dict:
    genes_out = []
    for idx, gene_def in enumerate(SKELETAL_DYSPLASIA_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(gene_def, seed=seed, n=40)
        stats = _cohort_stats(cohort)
        genes_out.append({
            "gene": gene_def["gene"],
            "protein": gene_def["protein"],
            "alias": gene_def["alias"],
            "aa": gene_def["aa"],
            "kDa": gene_def["kDa"],
            "gene_class": gene_def["gene_class"],
            "locus": gene_def["locus"],
            "omim_gene": gene_def["omim_gene"],
            "omim_disease": gene_def["omim_disease"],
            "phenotype": gene_def["phenotype"],
            "hallmark": gene_def["hallmark"],
            "treatment_alert": gene_def["treatment_alert"],
            "key_ddx": gene_def["key_ddx"],
            "gfr_pattern": gene_def["gfr_pattern"],
            "proteinuria_pattern": gene_def["proteinuria_pattern"],
            "primary_complication": gene_def["primary_complication"],
            "disease_detail": gene_def["disease_detail"],
            "inheritance": gene_def["inheritance"],
            "variants": gene_def.get("variants", []),
            "drug_ci": gene_def.get("drug_ci", []),
            "cohort_n": len(cohort),
            "cohort_stats": {
                "esrd_pct": stats.get("esrd_pct", 0),
                "htn_pct": stats.get("htn_pct", 0),
                "transplant_pct": stats.get("transplant_pct", 0),
                "drug_error_pct": stats.get("drug_error_pct", 0),
                "dx_delayed_pct": stats.get("dx_delayed_pct", 0),
                "surveillance_adherent_pct": stats.get("surveillance_adherent_pct", 0),
                "severity": stats.get("severity", {}),
            },
        })
    return {"genes": genes_out}


def get_definitions() -> list:
    return [
        {
            "term": "Wormian Bones (Intrasutural Ossicles)",
            "full": "Extra ossicles within skull sutures — hallmark of OI",
            "explanation": (
                "Wormian bones are small, irregular ossicles found within the cranial sutures. "
                "They are named after 17th-century anatomist Ole Worm. "
                "Diagnostic criteria for significance: >10 bones, each >6×4 mm, on AP skull X-ray. "
                "Found in 78% of OI patients — highly sensitive. "
                "Also present (less commonly) in: hypothyroidism, Down syndrome, "
                "cleidocranial dysostosis, Menkes disease, pyknodysostosis. "
                "In OI: result from abnormal collagen I → impaired membranous ossification "
                "of calvaria → 'patchwork' ossification pattern. "
                "NOT seen in achondroplasia, HME, or COMP dysplasias (different mechanism)."
            ),
        },
        {
            "term": "Blue Sclerae (OI)",
            "full": "Blue-grey scleral hue — pathognomonic for OI Type I; fades in adults",
            "explanation": (
                "The sclera (white of the eye) contains collagen I. "
                "In OI Type I (COL1A1 haploinsufficiency), the sclera is thin "
                "because only 50% of normal collagen I is produced. "
                "The thin, translucent sclera allows the underlying dark blue-grey "
                "choroidal pigment to show through, producing the characteristic blue hue. "
                "Blue sclerae are PATHOGNOMONIC for OI Type I in combination with fragility fractures. "
                "They may fade to pale blue or white in adults. "
                "OI Type II also has blue sclerae (Gly substitutions). "
                "OI Type III has pale blue-white sclerae. "
                "Important: NOT seen in hypophosphatasia (which can mimic OI) — "
                "check ALP before bisphosphonates."
            ),
        },
        {
            "term": "Vosoritide (BMN 111) — Achondroplasia Disease-Modifying Therapy",
            "full": "C-type natriuretic peptide analogue — FDA 2021; first FGFR3 pathway modifier",
            "explanation": (
                "Vosoritide (formerly BMN 111) is a modified C-type natriuretic peptide (CNP) analogue. "
                "CNP normally binds NPR-B (natriuretic peptide receptor B) on chondrocytes, "
                "activating cGMP signalling that counteracts constitutive FGFR3/MAPK signalling. "
                "In achondroplasia, FGFR3 is constitutively active (G380R), suppressing chondrocyte "
                "proliferation in the growth plate. Vosoritide provides an opposing signal, "
                "partially restoring endochondral ossification. "
                "FDA approved August 2021 for children aged 5 years to growth plate closure. "
                "Daily SC injection (15 μg/kg); increases annualised growth velocity by +1.6 cm/year. "
                "Does NOT cure achondroplasia or fully normalise height; "
                "contraindicated after growth plate fusion (ineffective)."
            ),
        },
        {
            "term": "Hitchhiker Thumb (Diastrophic Dysplasia)",
            "full": "Short proximally-set abducted thumb — pathognomonic for Diastrophic Dysplasia",
            "explanation": (
                "In Diastrophic Dysplasia (SLC26A2/DTDST), the thumb is characteristically "
                "short, wide, proximally placed (closer to the wrist), and held in radial abduction "
                "at an angle resembling a person hitching a ride — 'hitchhiker thumb.' "
                "This results from hypoplasia of the first metacarpal and metacarpophalangeal "
                "joint abnormalities secondary to undersulphated proteoglycans in cartilage. "
                "PATHOGNOMONIC for DTD — present in >95% of cases. "
                "Distinguishes DTD from achondroplasia (trident hand) and OI (no specific hand sign). "
                "Visible on fetal ultrasound from ~14-16 weeks gestation, enabling prenatal diagnosis."
            ),
        },
        {
            "term": "Cauliflower Ear (Diastrophic Dysplasia) — DO NOT ASPIRATE",
            "full": "Neonatal pinnal swelling → calcification — pathognomonic DTD; aspiration CONTRAINDICATED",
            "explanation": (
                "In DTD, neonates and infants develop swelling of the ear pinnae (external ears) "
                "resembling haematomas. These are NOT true haematomas. "
                "The mechanism: undersulphated cartilage proteoglycans → inflammatory reaction in "
                "auricular cartilage → cartilage inflammation and swelling at birth → "
                "over weeks to months, the swelling calcifies, creating the rigid, "
                "'cauliflower' deformity that gives the ear its irregular, lumpy appearance. "
                "CRITICAL: aspiration of the pinnal swelling carries a HIGH RISK of introducing "
                "bacteria into the cartilaginous ear pinna, leading to suppurative chondritis "
                "(infection of cartilage) which can destroy the pinna and cause permanent deformity. "
                "Management = conservative ice compression only. "
                "Present in 30-40% of DTD cases. PATHOGNOMONIC when combined with hitchhiker thumb."
            ),
        },
        {
            "term": "Foramen Magnum Stenosis (Achondroplasia)",
            "full": "Narrowed foramen magnum → medullary compression → central apnoea; screen all ACH infants",
            "explanation": (
                "The foramen magnum is formed by the skull base, which grows by endochondral "
                "ossification. In achondroplasia (FGFR3 GOF), endochondral ossification is impaired, "
                "producing a smaller-than-normal foramen magnum. "
                "The posterior fossa (cerebellum and brainstem) is therefore compressed. "
                "The medulla oblongata at the foramen can be compressed → central apnoea. "
                "In infants: may present as central apnoea, snoring, or sudden infant death. "
                "In older children: myelopathy (hyperreflexia, clonus, hand weakness). "
                "ALL ACH infants: brain MRI at 6-12 months with foramen magnum AP measurement. "
                "Normal: >3.5 mm AP. Stenosis = ≤3.5 mm AND/OR cord signal change. "
                "Treatment: C1 laminectomy (posterior decompression). "
                "This complication accounts for the elevated sudden death rate in ACH infants <2 years."
            ),
        },
        {
            "term": "Chondrosarcoma (HME Malignant Transformation)",
            "full": "Malignant transformation of osteochondroma → chondrosarcoma; cap >2 cm MRI = high risk",
            "explanation": (
                "Osteochondromas in EXT1/EXT2 HME have a 2-5% (EXT1) and 1-2% (EXT2) lifetime "
                "risk of malignant transformation to secondary chondrosarcoma. "
                "Risk factors: numerous lesions, axial location (pelvis, shoulder girdle, spine), "
                "adult age (post-skeletal maturity), cartilage cap thickness >2 cm on MRI. "
                "After skeletal maturity, a normal osteochondroma DOES NOT GROW and its "
                "cartilage cap thins to <2 cm. Any growing lesion or cap >2 cm in an adult = "
                "malignant transformation until proven otherwise. "
                "Symptoms: pain in a previously painless lesion in an adult. "
                "Investigation: MRI (NOT CT — avoid radiation in young patients). "
                "Treatment: wide surgical excision. "
                "Chondrosarcoma DOES NOT respond to chemotherapy or radiotherapy."
            ),
        },
        {
            "term": "Cartilage-Hair Hypoplasia (CHH) — Immune Deficiency and Lymphoma",
            "full": "RMRP AR RNA gene; T-cell lymphopenia + 10% lifetime lymphoma risk; live vaccines CI",
            "explanation": (
                "CHH (RMRP gene — RNA gene, not protein-coding) causes metaphyseal dysplasia "
                "with fine hypoplastic hair AND a cellular immune deficiency. "
                "The immune deficiency ranges from mild T-cell lymphopenia to SCID-like presentations "
                "requiring haematopoietic stem cell transplantation (HSCT). "
                "The immune defect arises because the RNase MRP complex is required for rRNA "
                "processing and ribosome biogenesis — T-cell and NK-cell progenitors "
                "are highly proliferative and sensitive to ribosomal insufficiency. "
                "LYMPHOMA RISK: 10-fold increased compared with general population; "
                "predominantly non-Hodgkin B-cell lymphoma in 3rd-5th decade. "
                "Annual clinical lymphoma surveillance from age 20. "
                "LIVE VACCINES (MMR, varicella, BCG, yellow fever): absolutely contraindicated "
                "in CHH with T-cell lymphopenia — disseminated vaccine-strain infection and death reported."
            ),
        },
        {
            "term": "Pseudoachondroplasia (PSACH) — Normal at Birth, Normal Face",
            "full": "COMP AD; diagnosed age 2-3; NORMAL face+head (distinguishes from ACH); C1-C2 instability",
            "explanation": (
                "PSACH (COMP gene, 19p13.11) is a critical DDx for achondroplasia "
                "but differs in several essential ways: "
                "(1) NORMAL AT BIRTH: limb shortening and gait abnormality not apparent until "
                "child begins walking at age 2-3 years; "
                "(2) NORMAL FACE AND HEAD: no frontal bossing, no macrocephaly, no midface hypoplasia "
                "— this is THE key distinguishing examination finding; "
                "(3) LIGAMENTOUS LAXITY: hyperextensible joints (especially knees and cervical spine) "
                "— paradoxically present despite short stature; "
                "(4) ODONTOID HYPOPLASIA and C1-C2 instability: lateral cervical spine X-ray "
                "(flexion-extension views) mandatory annually; contact sports absolutely CI. "
                "Mechanism: COMP missense variants in type III repeat domain → ER retention "
                "of misfolded COMP pentamer → chondrocyte ER stress → apoptosis → "
                "growth plate disorganisation."
            ),
        },
        {
            "term": "Diastrophic Dysplasia — Club Feet Most Severe of Any Skeletal Dysplasia",
            "full": "SLC26A2 AR; talipes equinovarus most severe; requires surgical correction beyond Ponseti",
            "explanation": (
                "Club feet (talipes equinovarus) are present in virtually all DTD patients (~95%). "
                "They are characteristically more rigid and severe than idiopathic club feet "
                "or those in other skeletal dysplasias. "
                "The rigidity results from undersulphated cartilage and joint contractures "
                "caused by impaired proteoglycan sulphation (SLC26A2/DTDST LOF). "
                "Standard Ponseti serial casting produces partial correction but "
                "almost always requires operative Achilles tendon tenotomy and often "
                "more extensive posterior-medial release or subtalar fusion. "
                "Recurrence rate is high — long-term orthopaedic follow-up mandatory. "
                "The severity of club feet in DTD is diagnostically useful: "
                "idiopathic clubfoot responds well to Ponseti; DTD clubfoot resists. "
                "Combined with hitchhiker thumb and cauliflower ear: "
                "PATHOGNOMONIC triad for Diastrophic Dysplasia."
            ),
        },
    ]
