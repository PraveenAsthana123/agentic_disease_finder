#!/usr/bin/env python3
"""Hereditary-Skeletal-Dysplasia-Atlas — Complete 8-Gene Hereditary Skeletal Dysplasia Atlas.

FGFR3    (fibroblast growth factor receptor 3; 806 aa; 4p16.3; AD;
          Achondroplasia (G380R — >98% de novo) / Hypochondroplasia (N540K) /
          Thanatophoric Dysplasia (K650E/M — lethal perinatally);
          vosoritide FDA 2021 for ACH; foramen magnum decompression <1 yr if severe;
          seed SEED_BASE+0).
COL2A1   (type II collagen alpha-1; 1487 aa; 12q13.11; AD;
          Stickler Syndrome Type 1 — vitreous anomaly PATHOGNOMONIC + progressive myopia + SNHL + EARLYOA;
          Spondyloepiphyseal Dysplasia Congenita (SEDC) — cleft palate + odontoid hypoplasia C-SPINE CI;
          seed SEED_BASE+1).
EXT1     (exostosin glycosyltransferase 1; 746 aa; 8q24.11; AD;
          Multiple Hereditary Exostoses (MHE) Type 1 — cartilage-capped bony outgrowths;
          EXT1 > EXT2 risk of malignant transformation (chondrosarcoma 1-2%);
          SARCOMA SURVEILLANCE mandatory — rapid growth = biopsy urgently;
          seed SEED_BASE+2).
EXT2     (exostosin glycosyltransferase 2; 718 aa; 11p12-p11; AD;
          Multiple Hereditary Exostoses (MHE) Type 2;
          milder phenotype than EXT1 but malignant transformation still possible;
          seed SEED_BASE+3).
COMP     (cartilage oligomeric matrix protein; 757 aa; 19p13.11; AD;
          Pseudoachondroplasia (PSACH) — NORMAL head + NORMAL face = CRITICAL DDx achondroplasia;
          Multiple Epiphyseal Dysplasia (MED) — early osteoarthritis + joint pain childhood;
          seed SEED_BASE+4).
SLC26A2  (sulfate transporter SLC26A2/DTDST; 739 aa; 5q32; AR;
          Diastrophic Dysplasia (DTD) — CAULIFLOWER EAR PATHOGNOMONIC + hitchhiker thumb + club foot;
          Achondrogenesis Type 1B (ACG1B) — lethal; Atelosteogenesis Type 2;
          seed SEED_BASE+5).
TRPV4    (transient receptor potential cation channel V4; 871 aa; 12q24.11; AD;
          Metatropic Dysplasia — SEVERE short-limb + progressive kyphoscoliosis lethal;
          Brachyolmia — short trunk, platyspondyly, mild;
          SMAD / SEDM spectrum;
          seed SEED_BASE+6).
ACAN     (aggrecan; 2153 aa; 15q26.1; AD/AR;
          Familial Short Stature (ACAN-FSS) — AD LOF — advanced bone age PATHOGNOMONIC + short stature;
          Spondyloepiphyseal Dysplasia (SED) Kimberley type;
          GH therapy may improve height but early bone age closure limits response;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2038-2045).
"""

import random

SEED_BASE = 2038

SD_GENES = [
    # -- FGFR3 — Achondroplasia / Hypochondroplasia / Thanatophoric Dysplasia (AD) -----
    {
        "gene": "FGFR3",
        "alt_name": "FGFR3 (Fibroblast-Growth-Factor-Receptor-3 / AD — Achondroplasia-G380R-De-Novo-98pct — Hypochondroplasia-N540K — Thanatophoric-Dysplasia-K650E-Lethal — Vosoritide-FDA-2021-ACH)",
        "protein": (
            "FGFR3 -- 4p16.3 AD -- FGFR3-806aa -- "
            "Achondroplasia-Most-Common-Lethal-Skeletal-Dysplasia-Vosoritide-FDA-2021 -- "
            "G380R-De-Novo->98pct-New-Mutation-Risk-Advanced-Paternal-Age -- "
            "Foramen-Magnum-Stenosis-MANDATORY-Screening-Year-1 -- "
            "Thanatophoric-Dysplasia-K650E-Lethal-Perinatally-Cloverleaf-Skull"
        ),
        "locus": "4p16.3",
        "protein_size": "806 aa",
        "inheritance": "AD (autosomal dominant) — gain-of-function constitutive FGFR3 signalling; >98% de novo for ACH G380R",
        "age_of_onset": (
            "Achondroplasia: diagnosed at birth (radiological) or prenatally from 22 wk ultrasound; "
            "Limb shortening: rhizomelic (proximal > distal); trident hand; macrocephaly; frontal bossing; midface hypoplasia; "
            "Foramen magnum stenosis: present in >50% infants — risk of sudden death, sleep apnoea, central apnoea; MRI mandatory within 1st year; "
            "Obstructive sleep apnoea: >80% by age 5 — polysomnography; ENT review; "
            "Spinal stenosis: progressive — onset late childhood/adult; claudication; "
            "Intelligence: NORMAL; "
            "Hypochondroplasia (N540K): milder — shorter stature but near-normal proportions; often diagnosed age 2-5 yr; "
            "Thanatophoric Dysplasia (K650E/K650M): severe — lethal perinatally; cloverleaf skull (type 2 TD); telephonic femur bowing; "
            "Paternal age effect: G380R de novo risk increases with advanced paternal age (spermatogonial advantage)"
        ),
        "key_biomarker": (
            "Skeletal survey (X-ray): 'telephone-receiver' femur (bowing) pathognomonic for TD; "
            "ACH: narrowing of interpediculate distance from L1→L5 (opposite of normal); bullet-shaped lumbar vertebrae (neonatal); "
            "FGFR3 molecular testing: G380R >98% of ACH (point mutation, PCR-based); "
            "Prenatal: increased BPD + shortened femur/humerus on 18-22 wk anomaly scan; "
            "CSF flow MRI (CINE-MRI): quantify foramen magnum stenosis; "
            "Polysomnography: respiratory events (obstructive + central); "
            "Spine MRI: cord compression; lumbar canal diameter; "
            "Brain MRI: ventriculomegaly (communicating hydrocephalus in ACH); "
            "Molecular: FGFR3 panel — G380R, N540K, K650 codons; NGS panel for atypical cases"
        ),
        "pathognomonic": (
            "RHIZOMELIC SHORT STATURE + TRIDENT HAND + MACROCEPHALY + NORMAL INTELLIGENCE = Achondroplasia until proven otherwise; "
            "Narrowing of L1→L5 interpediculate distance on AP spine X-ray = ACH (OPPOSITE of normal widening); "
            "TELEPHONE-RECEIVER femur on prenatal/neonatal X-ray = Thanatophoric Dysplasia; "
            "DISTINGUISH Hypochondroplasia: milder, near-normal proportions, normal head — N540K variant most common; "
            "DISTINGUISH PSACH (COMP): normal face + normal head but short stature — CON FGFR3 has normal face too but distinct X-ray pattern; "
            "FORAMEN MAGNUM STENOSIS: sudden infant death, apnoea, sleep disordered breathing — SCREEN ALL ACH INFANTS"
        ),
        "treatment": (
            "Vosoritide (C-type natriuretic peptide analogue; FDA 2021, EMA 2021): 15 µg/kg SC daily — "
            "increases annualised height velocity ~1.6 cm/yr; start >2 yr, open-ended; "
            "Foramen magnum decompression: if severe stenosis (CSF flow impairment) — ideally <6-12 months; "
            "Adenotonsillectomy / CPAP: for OSA — polysomnography first; "
            "Spinal cord decompression: for symptomatic stenosis; "
            "Limb lengthening (Ilizarov/PRECICE nail): elective cosmetic — patient/family decision; typically deferred to adulthood; "
            "AVOID contact sports if foramen magnum stenosis persists; "
            "Annual neurodevelopmental review; audiometry (middle ear dysfunction common); ophthalmology; "
            "AVOID Thanatophoric Dysplasia newborn resuscitation: generally redirected to comfort care — parental decision with palliative team"
        ),
        "critical_flags": [
            "FGFR3-ACH-G380R-DE-NOVO-98pct",
            "FGFR3-FORAMEN-MAGNUM-STENOSIS-SCREEN-YEAR-1",
            "FGFR3-VOSORITIDE-FDA-2021",
            "FGFR3-INTERPEDICULATE-NARROWING-L1-L5-PATHOGNOMONIC",
            "FGFR3-TD-TELEPHONE-RECEIVER-FEMUR-LETHAL",
            "FGFR3-NORMAL-INTELLIGENCE-ACH",
            "FGFR3-ADVANCED-PATERNAL-AGE-DE-NOVO-RISK",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- COL2A1 — Stickler Syndrome / SEDC (AD) ----------------------------------------
    {
        "gene": "COL2A1",
        "alt_name": "COL2A1 (Type-II-Collagen-Alpha-1 / AD — Stickler-Syndrome-Type-1-Vitreous-Anomaly-PATHOGNOMONIC — SEDC-Odontoid-Hypoplasia-C-SPINE-CI — Early-OA-Mandatory-Surveillance)",
        "protein": (
            "COL2A1 -- 12q13.11 AD -- COL2A1-1487aa -- "
            "Stickler-Syndrome-Type-1-Congenital-Vitreous-Anomaly-PATHOGNOMONIC -- "
            "SEDC-Spondyloepiphyseal-Dysplasia-Congenita-Odontoid-Hypoplasia-C1-C2-CI -- "
            "Early-Onset-Osteoarthritis-Joint-Replacement-30s-40s-Inevitable -- "
            "Retinal-Detachment-Prophylactic-Laser-MANDATORY"
        ),
        "locus": "12q13.11",
        "protein_size": "1487 aa",
        "inheritance": "AD (autosomal dominant) — COL2A1 encodes type II collagen alpha-1 chain; haploinsufficiency (premature stop) vs dominant negative (glycine substitutions)",
        "age_of_onset": (
            "Stickler Syndrome Type 1 (COL2A1): neonatal — Pierre Robin sequence (cleft palate + micrognathia + glossoptosis) in ~50%; "
            "High myopia: typically >−10 D; present in infancy; "
            "Vitreous anomaly: congenital membranous vitreous (TYPE 1 Stickler — COL2A1) vs beaded vitreous (Type 2 — COL11A1); "
            "Retinal detachment: 40-60% lifetime risk; prophylactic laser to lattice degeneration; "
            "Progressive SNHL: mild-moderate sensorineural — audiological follow-up; "
            "Early OA: hips, knees — onset 20-30s; joint replacement often by 40s; "
            "SEDC: more severe — disproportionate short stature (short trunk > short limb) at birth; coxa vara; "
            "Odontoid hypoplasia (SEDC): C1-C2 instability — cervical spine X-ray + MRI mandatory; "
            "Cleft palate: 50% of Stickler; Robin sequence neonatal airway emergency"
        ),
        "key_biomarker": (
            "Ophthalmology: congenital membranous vitreous anomaly on slit-lamp (Type 1 COL2A1 — PATHOGNOMONIC, distinct from Type 2 COL11A1 beaded); "
            "Refraction: high myopia >-3 D (>-10 D typical); "
            "Retinal exam: lattice degeneration — laser prophylaxis decision; "
            "Skeletal survey: platyspondyly (SEDC); epiphyseal dysplasia; coxa vara; "
            "Cervical spine X-ray + flexion-extension MRI: odontoid hypoplasia (SEDC); C1-C2 instability; "
            "Audiometry: SNHL characterisation; "
            "COL2A1 sequencing: premature truncation (Stickler) vs glycine substitution (more severe SEDC phenotype); "
            "Echocardiography: mitral valve prolapse 50% (connective tissue laxity); "
            "Joint imaging: early OA features on plain X-ray hips/knees"
        ),
        "pathognomonic": (
            "CONGENITAL MEMBRANOUS VITREOUS ANOMALY on slit-lamp = Stickler Type 1 (COL2A1) — distinguishes from Stickler Type 2 (COL11A1 beaded vitreous) and Marshall Syndrome; "
            "Pierre Robin sequence + high myopia + family history of early OA = COL2A1 Stickler until proven otherwise; "
            "ODONTOID HYPOPLASIA in SEDC patient = CERVICAL SPINE INSTABILITY — contact sports ABSOLUTELY CONTRAINDICATED; "
            "DISTINGUISH Marshall Syndrome: similar but more severe SNHL + cataract + flatter midface (COL11A1); "
            "DISTINGUISH Kniest Dysplasia: more severe COL2A1 — Swiss-cheese cartilage on histology; "
            "Early OA in 30s with personal/family history of myopia + hearing loss = Stickler screening"
        ),
        "treatment": (
            "Retinal detachment prophylaxis: 360° prophylactic laser (cerclage) or sector laser to lattice degeneration — HIGHLY RECOMMENDED; "
            "Myopia correction: glasses/contact lenses; refraction annually; "
            "Cleft palate repair: neonatal — Robin sequence may need nasopharyngeal airway/intubation; "
            "SNHL: hearing aids; cochlear implant if severe-profound; "
            "SEDC cervical spine: hard collar for activities; surgical stabilisation if myelopathy; "
            "Early OA management: low-impact exercise; physiotherapy; early joint replacement planning; "
            "Mitral valve prolapse: annual echocardiography; endocarditis prophylaxis if significant regurgitation; "
            "Genetic counselling: AD — 50% risk; prenatal diagnosis available"
        ),
        "critical_flags": [
            "COL2A1-STICKLER-MEMBRANOUS-VITREOUS-PATHOGNOMONIC",
            "COL2A1-RETINAL-DETACHMENT-LASER-MANDATORY",
            "COL2A1-SEDC-ODONTOID-HYPOPLASIA-C-SPINE-CI",
            "COL2A1-PIERRE-ROBIN-NEONATAL-AIRWAY",
            "COL2A1-EARLY-OA-30s-40s",
            "COL2A1-CONTACT-SPORTS-SEDC-ABSOLUTELY-CI",
            "COL2A1-HIGH-MYOPIA->10D",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- EXT1 — Multiple Hereditary Exostoses Type 1 (AD) --------------------------------
    {
        "gene": "EXT1",
        "alt_name": "EXT1 (Exostosin-Glycosyltransferase-1 / AD — Multiple-Hereditary-Exostoses-MHE1 — Chondrosarcoma-1-2pct-HIGHEST-RISK-EXT1 — Rapid-Growth-BIOPSY-URGENTLY — Heparan-Sulfate-Deficiency)",
        "protein": (
            "EXT1 -- 8q24.11 AD -- EXT1-746aa -- "
            "Multiple-Hereditary-Exostoses-MHE-Type-1-Highest-Malignant-Transformation-Risk -- "
            "Cartilage-Capped-Bony-Osteochondroma-Growth-Plate-Origin -- "
            "SARCOMA-SURVEILLANCE-Rapid-Growth-After-Skeletal-Maturity-BIOPSY-URGENTLY -- "
            "Heparan-Sulphate-Proteoglycan-HSPG-Biosynthesis-Defect"
        ),
        "locus": "8q24.11",
        "protein_size": "746 aa",
        "inheritance": "AD (autosomal dominant) — EXT1 tumour suppressor; two-hit model for exostosis formation (somatic second hit)",
        "age_of_onset": (
            "Osteochondromas: appear in early childhood, grow with skeleton — usually evident by age 5; "
            "Growth plates at risk: metaphyses of long bones (knee, ankle, wrist most common); "
            "Skeletal deformity: leg length discrepancy, forearm shortening (radius > ulna — Madelung-like deformity), ankle valgus; "
            "Skeletal growth: ceases with epiphyseal fusion at skeletal maturity (18-20 yr males, 16-18 yr females); "
            "New growth after skeletal maturity = MALIGNANT TRANSFORMATION until excluded; "
            "Malignant transformation: 1-2% EXT1 (higher than EXT2 ~0.5%); secondary chondrosarcoma; "
            "Pain: bursal formation, nerve compression, tendon impingement; "
            "Spinal exostoses: cervical > thoracic (cord compression — MRI if neurological symptoms); "
            "Lung exostoses (rare): pulmonary exostosis"
        ),
        "key_biomarker": (
            "Skeletal survey: multiple osteochondromas — cartilage-capped exostoses on plain X-ray; "
            "MRI: cartilage cap thickness — >2 cm in skeletally mature patient = HIGH RISK for sarcoma; "
            "CT: mineralisation pattern within lesion (ring-and-arc = chondrosarcoma); "
            "EXT1 molecular testing: sequencing + MLPA (large deletions 30%); "
            "Whole-body MRI: for surveillance in high-risk patients; "
            "PET-CT: if sarcomatous transformation suspected; "
            "Alkaline phosphatase: non-specific; "
            "Biopsy: core needle under image guidance if cap >2 cm or new growth post-skeletal maturity"
        ),
        "pathognomonic": (
            "MULTIPLE OSTEOCHONDROMAS with family history (AD) = Multiple Hereditary Exostoses — EXT1 or EXT2 testing; "
            "CARTILAGE CAP >2 cm in skeletally mature adult = high risk secondary chondrosarcoma — URGENT biopsy; "
            "NEW GROWTH or PAIN in existing exostosis after skeletal maturity = sarcoma until excluded; "
            "EXT1 CARRIES HIGHER MALIGNANT RISK than EXT2 — distinguish by genotyping; "
            "DISTINGUISH solitary osteochondroma (sporadic — no germline EXT mutation); "
            "MADELUNG-LIKE FOREARM DEFORMITY (radius > ulna length discrepancy) common in MHE — wrist X-ray"
        ),
        "treatment": (
            "Surgical excision: symptomatic exostoses (pain, nerve compression, deformity, cosmesis); "
            "Malignant transformation: wide surgical excision ± radiotherapy — chemotherapy for dedifferentiated component; "
            "Deformity correction: osteotomy for leg length discrepancy / forearm deformity if functionally significant; "
            "Surveillance: annual clinical review; MRI of concerning lesions; "
            "Patient education: self-examination; report any rapid growth or new pain immediately; "
            "AVOID watchful waiting for cap >2 cm — biopsy required; "
            "Genetic counselling: AD 50% risk; prenatal diagnosis; "
            "No approved disease-modifying medical therapy; mTOR pathway (rapamycin) investigational"
        ),
        "critical_flags": [
            "EXT1-MALIGNANT-TRANSFORMATION-1-2pct-HIGHEST-RISK",
            "EXT1-CARTILAGE-CAP->2cm-BIOPSY-URGENT",
            "EXT1-NEW-GROWTH-POST-SKELETAL-MATURITY-SARCOMA",
            "EXT1-MRI-CAP-THICKNESS-SURVEILLANCE",
            "EXT1-HIGHER-RISK-THAN-EXT2",
            "EXT1-MADELUNG-FOREARM-DEFORMITY",
            "EXT1-TWO-HIT-TUMOUR-SUPPRESSOR",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- EXT2 — Multiple Hereditary Exostoses Type 2 (AD) ---------------------------------
    {
        "gene": "EXT2",
        "alt_name": "EXT2 (Exostosin-Glycosyltransferase-2 / AD — Multiple-Hereditary-Exostoses-MHE2 — Milder-Than-EXT1 — Malignant-Transformation-0.5pct — Same-Surveillance-Protocol)",
        "protein": (
            "EXT2 -- 11p12-p11 AD -- EXT2-718aa -- "
            "Multiple-Hereditary-Exostoses-MHE-Type-2-Milder-Phenotype-Than-EXT1 -- "
            "Malignant-Transformation-0.5pct-Lower-Risk-Than-EXT1 -- "
            "Same-Sarcoma-Surveillance-Cap->2cm-BIOPSY -- "
            "EXT1-EXT2-Heterodimer-Golgi-HSPG-Heparan-Sulphate"
        ),
        "locus": "11p12-p11",
        "protein_size": "718 aa",
        "inheritance": "AD (autosomal dominant) — EXT2 tumour suppressor; forms heterodimer with EXT1 in Golgi (both required for heparan sulphate biosynthesis)",
        "age_of_onset": (
            "Osteochondromas: childhood onset — same distribution as EXT1; "
            "Phenotype: generally milder than EXT1 — fewer exostoses, less deformity on average (significant overlap); "
            "Malignant transformation: 0.3-0.5% (lower than EXT1 1-2%) — secondary chondrosarcoma still possible; "
            "Skeletal deformities: forearm, knee, ankle — similar to EXT1 but less severe on average; "
            "Penetrance: >90% — most mutation carriers have visible exostoses by adulthood; "
            "New growth post-skeletal maturity: SAME surveillance protocol as EXT1 regardless of lower overall risk; "
            "Pain: nerve impingement, bursitis — same clinical management as EXT1"
        ),
        "key_biomarker": (
            "Skeletal survey: multiple osteochondromas — indistinguishable radiologically from EXT1; "
            "Molecular: EXT2 sequencing + MLPA; cannot distinguish EXT1 vs EXT2 clinically — molecular testing required; "
            "MRI cap thickness: same threshold as EXT1 (>2 cm = high risk); "
            "EXT1/EXT2 panel: sent together — test EXT1 first (more common, higher risk); "
            "Alkaline phosphatase: non-specific; "
            "Family segregation: AD 50% risk — test first-degree relatives with skeletal survey"
        ),
        "pathognomonic": (
            "Multiple osteochondromas with family history — if EXT1 negative, test EXT2; "
            "EXT2 phenotype is generally MILDER than EXT1 but individual variation is large — "
            "NEVER reassure a patient with EXT2 that malignant risk is zero; "
            "EXT1-EXT2 HETERODIMER: both subunits required for heparan sulphate chain elongation — either mutant disrupts function; "
            "DISTINGUISH: solitary sporadic osteochondroma (no family history, single lesion — very low malignant risk); "
            "Sarcoma surveillance identical for EXT1 and EXT2 patients"
        ),
        "treatment": (
            "Surgical excision: symptomatic exostoses — same indications as EXT1; "
            "Sarcoma: wide excision — same oncological approach as EXT1; "
            "Surveillance: identical protocol to EXT1 — annual clinical; MRI cap thickness; "
            "Patient education: identical to EXT1 — NEVER delay reporting rapid growth; "
            "Genetic counselling: AD 50%; family testing; "
            "No differential medical therapy between EXT1 and EXT2 currently"
        ),
        "critical_flags": [
            "EXT2-MALIGNANT-TRANSFORMATION-0.5pct-LOWER-THAN-EXT1",
            "EXT2-SAME-SURVEILLANCE-AS-EXT1",
            "EXT2-MILDER-PHENOTYPE-ON-AVERAGE",
            "EXT2-EXT1-HETERODIMER-HSPG",
            "EXT2-NEVER-REASSURE-ZERO-SARCOMA-RISK",
            "EXT2-TEST-EXT1-FIRST",
            "EXT2-CAP->2cm-BIOPSY-URGENTLY",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- COMP — Pseudoachondroplasia / MED (AD) -------------------------------------------
    {
        "gene": "COMP",
        "alt_name": "COMP (Cartilage-Oligomeric-Matrix-Protein / AD — Pseudoachondroplasia-PSACH-NORMAL-Face-NORMAL-Head-CRITICAL-DDx-ACH — Multiple-Epiphyseal-Dysplasia-MED-Early-OA — C1-C2-Instability-PSACH)",
        "protein": (
            "COMP -- 19p13.11 AD -- COMP-757aa -- "
            "Pseudoachondroplasia-PSACH-Normal-Face-Normal-Head-CRITICAL-DDx-Achondroplasia -- "
            "Multiple-Epiphyseal-Dysplasia-MED-Childhood-Joint-Pain-Early-OA -- "
            "C1-C2-Instability-Odontoid-Hypoplasia-PSACH-MANDATORY-Surveillance -- "
            "Short-Limb-Short-Trunk-COMBINED-Dysplasia"
        ),
        "locus": "19p13.11",
        "protein_size": "757 aa",
        "inheritance": "AD (autosomal dominant) — dominant negative effect; COMP misfolding causes ER retention and chondrocyte death",
        "age_of_onset": (
            "Pseudoachondroplasia: normal at birth — SHORT STATURE NOT APPARENT UNTIL AMBULATION (18 months); "
            "CRITICAL POINT: normal face + normal head circumference AT BIRTH — DISTINGUISH from ACH (macrocephaly) and FGFR3; "
            "Joint laxity: extreme ligamentous laxity — waddling gait; "
            "C1-C2 instability: odontoid hypoplasia/ligamentous laxity — cervical spine MRI mandatory; "
            "Scoliosis + kyphosis: progressive; "
            "MED (milder): childhood joint pain (hips, knees, ankles) → early OA; mild short stature; "
            "Adult PSACH height: 82-130 cm; "
            "Pain: significant from childhood joint laxity + cartilage erosion; "
            "Adult complications: severe multi-joint OA requiring replacement"
        ),
        "key_biomarker": (
            "Skeletal survey: normal skull/face (DISTINGUISH from ACH) + epiphyseal irregularity + platyspondyly (bullet vertebrae) + rhizomelic > mesomelic shortening; "
            "Vertebral bodies: anterior notching ('bullet shape') in early childhood — same as ACH but face/head NORMAL in PSACH; "
            "Cervical spine X-ray + MRI: odontoid hypoplasia, C1-C2 instability (flexion-extension views); "
            "Epiphyses: irregular, delayed ossification (hips, knees) — MED pattern; "
            "COMP molecular testing: >97% PSACH/MED due to COL2A1 or COMP — distinguish by mutation; "
            "ER-retained COMP: immunohistochemistry on biopsy (ER inclusions); "
            "Calcium + phosphate: normal (DISTINGUISH rickets — same short stature presentation)"
        ),
        "pathognomonic": (
            "SHORT STATURE + NORMAL HEAD + NORMAL FACE = PSACH (COMP) not ACH (FGFR3); "
            "This DDx is LIFE-CRITICAL: ACH needs foramen magnum screening; PSACH needs C1-C2 screening; "
            "NOT APPARENT AT BIRTH — parents often alarmed when child's proportional growth diverges at ambulation; "
            "EXTREME JOINT LAXITY distinguishes PSACH from ACH; "
            "MED: early hip/knee OA with MILD short stature — COMP or MATN3 testing; "
            "C1-C2 INSTABILITY: contact sports ABSOLUTELY CONTRAINDICATED in PSACH; "
            "DISTINGUISH Schmid metaphyseal chondrodysplasia (COL10A1) — cup-shaped metaphyses, normal vertebrae"
        ),
        "treatment": (
            "Cervical spine: soft collar; activity restriction; surgical fusion if instability or myelopathy; "
            "CONTACT SPORTS ABSOLUTELY CONTRAINDICATED — C1-C2 instability risk; "
            "Joint management: low-impact activity; hydrotherapy; physiotherapy; "
            "Early OA: joint replacement — hips/knees often by 4th-5th decade; "
            "Scoliosis: bracing; surgical correction for severe curves; "
            "Pain management: NSAIDs; joint injections; "
            "Growth: GH therapy generally ineffective in PSACH; "
            "Limb lengthening: feasible but complex — specialised centre; "
            "Genetic counselling: AD 50% risk; prenatal molecular testing"
        ),
        "critical_flags": [
            "COMP-PSACH-NORMAL-FACE-NORMAL-HEAD-CRITICAL-DDx-ACH",
            "COMP-NOT-APPARENT-AT-BIRTH-AMBULATION-18M",
            "COMP-C1-C2-INSTABILITY-SCREEN-MANDATORY",
            "COMP-CONTACT-SPORTS-ABSOLUTELY-CI",
            "COMP-MED-EARLY-OA-CHILDHOOD",
            "COMP-EXTREME-JOINT-LAXITY",
            "COMP-ER-RETENTION-DOMINANT-NEGATIVE",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- SLC26A2 — Diastrophic Dysplasia / Achondrogenesis 1B (AR) ------------------------
    {
        "gene": "SLC26A2",
        "alt_name": "SLC26A2 (DTDST-Sulphate-Transporter / AR — Diastrophic-Dysplasia-CAULIFLOWER-EAR-PATHOGNOMONIC — Hitchhiker-Thumb — Achondrogenesis-1B-Lethal — Spectrum-Severity-Biallelic)",
        "protein": (
            "SLC26A2 -- 5q32 AR -- SLC26A2-739aa -- "
            "Diastrophic-Dysplasia-DTD-Cauliflower-Ear-PATHOGNOMONIC-Ear-Pinna-Cystic-Inflammation -- "
            "Hitchhiker-Thumb-Abducted-PATHOGNOMONIC-Bilateral -- "
            "Club-Foot-Talipes-Equinovarus-Bilateral-INVARIANT -- "
            "Achondrogenesis-1B-Lethal-Most-Severe-Spectrum"
        ),
        "locus": "5q32",
        "protein_size": "739 aa",
        "inheritance": "AR (autosomal recessive) — biallelic SLC26A2 loss-of-function; sulfate transporter deficiency → undersulfated proteoglycans → defective cartilage matrix",
        "age_of_onset": (
            "Diastrophic Dysplasia (DTD): prenatal/neonatal; "
            "Cauliflower ear: neonatal swelling and inflammation of ear pinna → calcification → CAULIFLOWER DEFORMITY by 3-6 months; pathognomonic; "
            "Hitchhiker thumb: bilateral abduction of first metacarpophalangeal joint — pathognomonic; "
            "Club foot: bilateral talipes equinovarus — invariant; "
            "Cleft palate: 25-35%; "
            "Cervical kyphosis: neonatal — risk of cord compression; may resolve spontaneously or require intervention; "
            "Progressive scoliosis: significant — main cause of morbidity; "
            "Intelligence: NORMAL; "
            "Adult survival: possible with supportive care — significant musculoskeletal disability; "
            "ACG1B (null mutations): lethal perinatally — severe micromelia, absent ossification; "
            "Spectrum: ACG1B (lethal) > Atelosteogenesis Type 2 > DTD (survivable)"
        ),
        "key_biomarker": (
            "Clinical triad: cauliflower ear + hitchhiker thumb + club foot = DTD until proven otherwise; "
            "Skeletal survey: severe micromelia; 'hitchhiker thumb' on hand X-ray (abducted); cervical spine kyphosis; platyspondyly; "
            "Ear ultrasonography: pinna calcification; "
            "Cervical spine MRI: cord compression risk — MANDATORY neonatal; "
            "SLC26A2 molecular testing: biallelic variants; Finnish founder variant c.835-2A>G (IVS1-2A>G) — very common in Finland; "
            "Urine/plasma sulphate: low in sulphate transporter disorders (not routinely done); "
            "Histology: undersulfated proteoglycans in cartilage (special staining)"
        ),
        "pathognomonic": (
            "CAULIFLOWER EAR (cystic pinna inflammation → calcification) + HITCHHIKER THUMB (bilateral) + BILATERAL CLUB FOOT = DTD pathognomonic triad; "
            "Cauliflower ear timing: pinna swelling at 1-2 weeks of age → inflammatory → hardens by 3-6 months; "
            "DISTINGUISH: ACG1B (allelic — null mutations — lethal; more severe radiological); "
            "DISTINGUISH PSACH (COMP): normal head/face, no ear/thumb/foot features; "
            "Finnish founder variant IVS1-2A>G: prevalent in Finland — point mutation carrier screening available; "
            "CERVICAL KYPHOSIS: may appear life-threatening neonately but often resolves — MONITOR CLOSELY, not always surgically treated"
        ),
        "treatment": (
            "Club foot: serial casting (Ponseti or French method) from birth; surgical correction if refractory; "
            "Cauliflower ear: no proven treatment to prevent calcification — early compression debated; cosmetic approaches; "
            "Cleft palate: surgical repair; "
            "Cervical kyphosis: HALO-VEST or custom cervical orthosis; surgical fusion if progressive/myelopathic; "
            "Scoliosis: bracing (limited effectiveness); posterior spinal fusion when curve >40-50°; "
            "Joint management: low-impact; hydrotherapy; physiotherapy; "
            "AVOID contact sports — spinal instability; "
            "Genetic counselling: AR — 25% recurrence risk; prenatal diagnosis; carrier testing for siblings; "
            "ACG1B: redirect to comfort care neonatally in most cases"
        ),
        "critical_flags": [
            "SLC26A2-CAULIFLOWER-EAR-PATHOGNOMONIC",
            "SLC26A2-HITCHHIKER-THUMB-BILATERAL-PATHOGNOMONIC",
            "SLC26A2-CLUB-FOOT-BILATERAL-INVARIANT",
            "SLC26A2-CERVICAL-KYPHOSIS-CORD-COMPRESSION-NEONATAL",
            "SLC26A2-ACG1B-LETHAL-NULL-MUTATIONS",
            "SLC26A2-FINNISH-FOUNDER-IVS1-2AG",
            "SLC26A2-NORMAL-INTELLIGENCE-DTD",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- TRPV4 — Metatropic Dysplasia / Brachyolmia (AD) ---------------------------------
    {
        "gene": "TRPV4",
        "alt_name": "TRPV4 (TRP-Cation-Channel-V4 / AD — Metatropic-Dysplasia-SEVERE-Kyphoscoliosis-Lethal — Brachyolmia-Mild-Short-Trunk — SEDM-Spondylo-Epi-Meta-Dysplasia — Gain-of-Function-Spectrum)",
        "protein": (
            "TRPV4 -- 12q24.11 AD -- TRPV4-871aa -- "
            "Metatropic-Dysplasia-Severe-Progressive-Kyphoscoliosis-Respiratory-Failure -- "
            "Brachyolmia-Mild-End-Short-Trunk-Platyspondyly -- "
            "SEDM-Spondyloepimetaphyseal-Dysplasia-Intermediate-Severity -- "
            "Gain-of-Function-GOF-Spectrum-Severity-Correlates-Channel-Overactivation"
        ),
        "locus": "12q24.11",
        "protein_size": "871 aa",
        "inheritance": "AD (autosomal dominant) — TRPV4 gain-of-function; excessive Ca2+ influx via activated TRP channel disrupts chondrocyte and osteoblast function",
        "age_of_onset": (
            "Metatropic Dysplasia (severe TRPV4 GOF): prenatal — short-limbed dwarfism; "
            "Metaphyseal widening: PATHOGNOMONIC — 'dumbbell' or 'halberd' shape femur/humerus on X-ray; "
            "Progressive kyphoscoliosis: rapidly progressive from infancy — severe restrictive lung disease; "
            "Respiratory failure: main cause of death (infancy-childhood in severe cases); "
            "'Metatropic' name: body proportions CHANGE with age (short limb at birth → more trunk shortening with progressive kyphoscoliosis); "
            "Joint hyperextensibility: large joints; "
            "Brachyolmia (mild GOF): mild short stature, platyspondyly, near-normal limb length; diagnosis often in adult; "
            "SEDM (intermediate): moderately severe; "
            "Intelligence: normal"
        ),
        "key_biomarker": (
            "Skeletal survey: DUMBBELL-shaped femur/humerus (expanded metaphyses, narrow diaphysis) = PATHOGNOMONIC for Metatropic; "
            "Platyspondyly: flat vertebral bodies — universal across spectrum; "
            "Pelvis: 'champagne-glass' pelvic inlet; "
            "Respiratory function: FVC, FEV1 — restrictive pattern from thoracic kyphoscoliosis; "
            "TRPV4 molecular testing: kinase domain variants (most severe), N-terminal/ankyrin domain (milder); "
            "Genotype-phenotype: GOF severity correlates with channel calcium flux increase; "
            "Spinal MRI: cord compression risk in severe kyphoscoliosis; "
            "CT chest: lung volume estimation"
        ),
        "pathognomonic": (
            "DUMBBELL (HALBERD) METAPHYSEAL APPEARANCE of long bones = Metatropic Dysplasia (TRPV4 GOF) pathognomonic; "
            "PROGRESSIVELY CHANGING PROPORTIONS ('metatropic' = changing shape): short-limb dwarfism at birth → trunk shortening dominates with age; "
            "PLATYSPONDYLY across ALL TRPV4 spectrum severity levels — hallmark finding; "
            "DISTINGUISH Kniest Dysplasia (COL2A1): dumbbell metaphyses also present but Swiss-cheese vitreous + hearing loss distinguish; "
            "Brachyolmia: platyspondyly alone without limb shortening — TRPV4 or PAPSS2 testing; "
            "RESPIRATORY SURVEILLANCE mandatory in Metatropic — main mortality determinant"
        ),
        "treatment": (
            "Respiratory: non-invasive ventilation (NIV/CPAP) — early if OSA or hypoventilation; "
            "Tracheostomy: for severe respiratory failure (some severe Metatropic cases); "
            "Kyphoscoliosis: posterior spinal fusion — complex surgery; significant bleeding/respiratory risk; "
            "Spinal cord decompression: if myelopathy; "
            "Joint management: hydrotherapy; physiotherapy; "
            "TRPV4 inhibitors (HC-067047 etc): investigational — not approved; animal studies promising; "
            "Prenatal counselling: de novo severe Metatropic — often lethal — palliative care decision; "
            "Genetic counselling: AD 50% risk if parent affected; recurrence risk for de novo = parental mosaicism risk (~1%)"
        ),
        "critical_flags": [
            "TRPV4-METATROPIC-DUMBBELL-METAPHYSES-PATHOGNOMONIC",
            "TRPV4-CHANGING-PROPORTIONS-METATROPIC-NAME",
            "TRPV4-PLATYSPONDYLY-ALL-SEVERITY-SPECTRUM",
            "TRPV4-RESPIRATORY-FAILURE-MAIN-MORTALITY",
            "TRPV4-GOF-SPECTRUM-BRACHYOLMIA-TO-METATROPIC",
            "TRPV4-INHIBITORS-INVESTIGATIONAL",
            "TRPV4-NIV-EARLY-RESPIRATORY-SURVEILLANCE",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- ACAN — ACAN-FSS / SED Kimberley (AD/AR) -----------------------------------------
    {
        "gene": "ACAN",
        "alt_name": "ACAN (Aggrecan / AD-LOF-Familial-Short-Stature-Advanced-Bone-Age-PATHOGNOMONIC — GH-Therapy-Partial-Response — AR-SED-Kimberley-Severe — Intervertebral-Disc-Disease-Adult)",
        "protein": (
            "ACAN -- 15q26.1 AD-AR -- ACAN-2153aa -- "
            "ACAN-Familial-Short-Stature-AD-LOF-Advanced-Bone-Age-PATHOGNOMONIC -- "
            "GH-Therapy-Height-Benefit-Limited-By-Bone-Age-Advancement -- "
            "Osteochondritis-Dissecans-ACAN-Association -- "
            "AR-SED-Kimberley-Severe-Short-Trunk-Deafness"
        ),
        "locus": "15q26.1",
        "protein_size": "2153 aa",
        "inheritance": "AD (ACAN-FSS — haploinsufficiency) or AR (SED Kimberley — biallelic; more severe)",
        "age_of_onset": (
            "ACAN-FSS (AD LOF): recognised at 2-4 yr with progressive short stature; "
            "Advanced bone age: PATHOGNOMONIC — bone age AHEAD of chronological age (opposite of GH deficiency); "
            "Height SDS: typically −2 to −4 SDS; mid-parental height often not achieved; "
            "Proportions: mildly short trunk + short limbs — dysmorphic but not severe; "
            "Osteochondritis dissecans (OCD): adolescent/adult — knee, elbow — ACAN association; "
            "Intervertebral disc disease: adult — back pain; disc herniation; "
            "Intelligence: normal; "
            "SED Kimberley (AR biallelic): more severe short stature; short trunk; platyspondyly; SNHL; early OA; "
            "GH therapy (ACAN-FSS): may increase height velocity but advanced bone age limits final height gain"
        ),
        "key_biomarker": (
            "Bone age X-ray (left hand): ADVANCED bone age (ahead of chronological) — distinguishes ACAN-FSS from GH deficiency (delayed bone age); "
            "Growth chart: short stature with high growth velocity for chronological age but appropriate for advanced bone age; "
            "IGF-1 / GH stimulation: NORMAL (DISTINGUISH GH deficiency); "
            "ACAN molecular testing: heterozygous LOF variants (frameshift, splice, nonsense) or missense; "
            "Skeletal survey: mild platyspondyly; mild epiphyseal irregularity; "
            "Knee MRI: osteochondritis dissecans — cartilage defect on weight-bearing surface; "
            "Spine MRI (adult): disc degeneration, disc herniation — earlier than population; "
            "SED Kimberley: biallelic ACAN — audiometry (SNHL)"
        ),
        "pathognomonic": (
            "SHORT STATURE + ADVANCED BONE AGE + FAMILY HISTORY (AD) + NORMAL GH AXIS = ACAN-FSS until proven otherwise; "
            "The ADVANCED bone age (not delayed as in GH deficiency) is the key radiological clue; "
            "OSTEOCHONDRITIS DISSECANS in adolescent with family history of short stature = ACAN testing; "
            "DISTINGUISH GH deficiency: delayed bone age, low IGF-1, low GH peak — ACAN-FSS has NORMAL GH; "
            "DISTINGUISH Turner syndrome (45,X): female short stature + delayed bone age + stigmata — karyotype; "
            "SED Kimberley (AR): more severe — SNHL + platyspondyly — BIALLELIC ACAN"
        ),
        "treatment": (
            "GH therapy (ACAN-FSS): recombinant GH — modest height improvement (∼0.5-1 SD); "
            "ADVANCED BONE AGE limits response: GnRH analogues to delay puberty may extend GH window (investigational in ACAN); "
            "Monitor bone age 6-monthly during GH therapy; "
            "OCD management: conservative (activity restriction) or arthroscopic drilling/fixation; "
            "Disc disease: physiotherapy; weight management; surgical decompression if refractory; "
            "SED Kimberley: hearing aids / cochlear implant (SNHL); OA joint replacement; "
            "Genetic counselling: AD — 50% risk; diagnose tall parents with advanced bone age in family"
        ),
        "critical_flags": [
            "ACAN-FSS-ADVANCED-BONE-AGE-PATHOGNOMONIC",
            "ACAN-NORMAL-GH-DISTINGUISH-GH-DEFICIENCY",
            "ACAN-GH-THERAPY-LIMITED-BY-BONE-AGE",
            "ACAN-OCD-OSTEOCHONDRITIS-DISSECANS-ASSOCIATION",
            "ACAN-DISC-DISEASE-ADULT",
            "ACAN-SED-KIMBERLEY-AR-BIALLELIC-SNHL",
            "ACAN-AD-HAPLOINSUFFICIENCY-LOF",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _generate_cohort(gene_entry: dict) -> list:
    gene = gene_entry["gene"]
    rng = random.Random(gene_entry["seed"])
    cohort = []

    for i in range(40):
        age = rng.randint(2, 70)
        sex = rng.choice(["M", "F"])

        if gene == "FGFR3":
            rhizomelic_short_stature = rng.random() < 0.98
            macrocephaly             = rng.random() < 0.92
            foramen_magnum_stenosis  = rng.random() < 0.60
            sleep_apnoea             = rng.random() < 0.82
            spinal_stenosis          = rng.random() < 0.55
            scoliosis                = rng.random() < 0.30
            early_oa                 = rng.random() < 0.40
            retinal_detachment       = rng.random() < 0.05
            osteochondroma           = rng.random() < 0.02
            joint_laxity             = rng.random() < 0.20
            cauliflower_ear          = rng.random() < 0.02
            advanced_bone_age        = rng.random() < 0.10
            malignant_transformation = rng.random() < 0.01
            club_foot                = rng.random() < 0.15
            cleft_palate             = rng.random() < 0.10

        elif gene == "COL2A1":
            rhizomelic_short_stature = rng.random() < 0.70
            macrocephaly             = rng.random() < 0.10
            foramen_magnum_stenosis  = rng.random() < 0.30
            sleep_apnoea             = rng.random() < 0.20
            spinal_stenosis          = rng.random() < 0.25
            scoliosis                = rng.random() < 0.30
            early_oa                 = rng.random() < 0.85
            retinal_detachment       = rng.random() < 0.50
            osteochondroma           = rng.random() < 0.02
            joint_laxity             = rng.random() < 0.55
            cauliflower_ear          = rng.random() < 0.02
            advanced_bone_age        = rng.random() < 0.05
            malignant_transformation = rng.random() < 0.01
            club_foot                = rng.random() < 0.05
            cleft_palate             = rng.random() < 0.50

        elif gene == "EXT1":
            rhizomelic_short_stature = rng.random() < 0.30
            macrocephaly             = rng.random() < 0.05
            foramen_magnum_stenosis  = rng.random() < 0.03
            sleep_apnoea             = rng.random() < 0.05
            spinal_stenosis          = rng.random() < 0.15
            scoliosis                = rng.random() < 0.10
            early_oa                 = rng.random() < 0.30
            retinal_detachment       = rng.random() < 0.02
            osteochondroma           = rng.random() < 0.99
            joint_laxity             = rng.random() < 0.20
            cauliflower_ear          = rng.random() < 0.03
            advanced_bone_age        = rng.random() < 0.05
            malignant_transformation = rng.random() < 0.015  # 1.5% lifetime
            club_foot                = rng.random() < 0.05
            cleft_palate             = rng.random() < 0.02

        elif gene == "EXT2":
            rhizomelic_short_stature = rng.random() < 0.20
            macrocephaly             = rng.random() < 0.05
            foramen_magnum_stenosis  = rng.random() < 0.02
            sleep_apnoea             = rng.random() < 0.05
            spinal_stenosis          = rng.random() < 0.10
            scoliosis                = rng.random() < 0.08
            early_oa                 = rng.random() < 0.25
            retinal_detachment       = rng.random() < 0.02
            osteochondroma           = rng.random() < 0.97
            joint_laxity             = rng.random() < 0.18
            cauliflower_ear          = rng.random() < 0.02
            advanced_bone_age        = rng.random() < 0.04
            malignant_transformation = rng.random() < 0.005  # 0.5% lifetime
            club_foot                = rng.random() < 0.04
            cleft_palate             = rng.random() < 0.02

        elif gene == "COMP":
            rhizomelic_short_stature = rng.random() < 0.88
            macrocephaly             = rng.random() < 0.05  # NORMAL head
            foramen_magnum_stenosis  = rng.random() < 0.05
            sleep_apnoea             = rng.random() < 0.25
            spinal_stenosis          = rng.random() < 0.35
            scoliosis                = rng.random() < 0.50
            early_oa                 = rng.random() < 0.90
            retinal_detachment       = rng.random() < 0.03
            osteochondroma           = rng.random() < 0.03
            joint_laxity             = rng.random() < 0.92
            cauliflower_ear          = rng.random() < 0.02
            advanced_bone_age        = rng.random() < 0.08
            malignant_transformation = rng.random() < 0.01
            club_foot                = rng.random() < 0.08
            cleft_palate             = rng.random() < 0.05

        elif gene == "SLC26A2":
            rhizomelic_short_stature = rng.random() < 0.92
            macrocephaly             = rng.random() < 0.08
            foramen_magnum_stenosis  = rng.random() < 0.08
            sleep_apnoea             = rng.random() < 0.20
            spinal_stenosis          = rng.random() < 0.30
            scoliosis                = rng.random() < 0.80
            early_oa                 = rng.random() < 0.65
            retinal_detachment       = rng.random() < 0.02
            osteochondroma           = rng.random() < 0.02
            joint_laxity             = rng.random() < 0.30
            cauliflower_ear          = rng.random() < 0.90
            advanced_bone_age        = rng.random() < 0.05
            malignant_transformation = rng.random() < 0.01
            club_foot                = rng.random() < 0.95
            cleft_palate             = rng.random() < 0.30

        elif gene == "TRPV4":
            rhizomelic_short_stature = rng.random() < 0.82
            macrocephaly             = rng.random() < 0.08
            foramen_magnum_stenosis  = rng.random() < 0.20
            sleep_apnoea             = rng.random() < 0.55
            spinal_stenosis          = rng.random() < 0.70
            scoliosis                = rng.random() < 0.88
            early_oa                 = rng.random() < 0.50
            retinal_detachment       = rng.random() < 0.02
            osteochondroma           = rng.random() < 0.02
            joint_laxity             = rng.random() < 0.60
            cauliflower_ear          = rng.random() < 0.03
            advanced_bone_age        = rng.random() < 0.05
            malignant_transformation = rng.random() < 0.01
            club_foot                = rng.random() < 0.15
            cleft_palate             = rng.random() < 0.05

        elif gene == "ACAN":
            rhizomelic_short_stature = rng.random() < 0.75
            macrocephaly             = rng.random() < 0.05
            foramen_magnum_stenosis  = rng.random() < 0.05
            sleep_apnoea             = rng.random() < 0.15
            spinal_stenosis          = rng.random() < 0.40
            scoliosis                = rng.random() < 0.20
            early_oa                 = rng.random() < 0.70
            retinal_detachment       = rng.random() < 0.05
            osteochondroma           = rng.random() < 0.03
            joint_laxity             = rng.random() < 0.30
            cauliflower_ear          = rng.random() < 0.02
            advanced_bone_age        = rng.random() < 0.95
            malignant_transformation = rng.random() < 0.01
            club_foot                = rng.random() < 0.05
            cleft_palate             = rng.random() < 0.05

        else:
            rhizomelic_short_stature = foramen_magnum_stenosis = macrocephaly = False
            sleep_apnoea = spinal_stenosis = scoliosis = early_oa = retinal_detachment = False
            osteochondroma = joint_laxity = cauliflower_ear = advanced_bone_age = False
            malignant_transformation = club_foot = cleft_palate = False

        cohort.append({
            "patient_id":             f"{gene}-{i+1:03d}",
            "age":                    age,
            "sex":                    sex,
            "gene":                   gene,
            "rhizomelic_short_stature": rhizomelic_short_stature,
            "macrocephaly":           macrocephaly,
            "foramen_magnum_stenosis": foramen_magnum_stenosis,
            "sleep_apnoea":           sleep_apnoea,
            "spinal_stenosis":        spinal_stenosis,
            "scoliosis":              scoliosis,
            "early_oa":               early_oa,
            "retinal_detachment":     retinal_detachment,
            "osteochondroma":         osteochondroma,
            "joint_laxity":           joint_laxity,
            "cauliflower_ear":        cauliflower_ear,
            "advanced_bone_age":      advanced_bone_age,
            "malignant_transformation": malignant_transformation,
            "club_foot":              club_foot,
            "cleft_palate":           cleft_palate,
        })

    return cohort


# ---------------------------------------------------------------------------
# API functions
# ---------------------------------------------------------------------------

def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in SD_GENES]
    all_pts = [p for c in all_cohorts for p in c]
    total = len(all_pts)

    def N(key): return sum(1 for p in all_pts if p[key])

    return {
        "atlas": "Hereditary-Skeletal-Dysplasia-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Skeletal Dysplasia Atlas: "
            "FGFR3 (achondroplasia G380R — foramen magnum screen yr1 — vosoritide FDA 2021) + "
            "COL2A1 (Stickler membranous vitreous PATHOGNOMONIC — retinal detachment laser mandatory — SEDC C-spine CI) + "
            "EXT1 (MHE1 — chondrosarcoma 1-2% highest risk — cap >2 cm biopsy urgently) + "
            "EXT2 (MHE2 — milder — same surveillance as EXT1) + "
            "COMP (PSACH normal face CRITICAL DDx ACH — C1-C2 instability — MED early OA) + "
            "SLC26A2 (DTD cauliflower ear PATHOGNOMONIC — hitchhiker thumb — bilateral club foot) + "
            "TRPV4 (metatropic dumbbell metaphyses PATHOGNOMONIC — kyphoscoliosis respiratory failure) + "
            "ACAN (FSS advanced bone age PATHOGNOMONIC — normal GH axis — GH therapy)"
        ),
        "genes": [g["gene"] for g in SD_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}–{SEED_BASE + len(SD_GENES) - 1}",
        "short_stature_patients":           N("rhizomelic_short_stature"),
        "macrocephaly_patients":            N("macrocephaly"),
        "foramen_magnum_stenosis_patients": N("foramen_magnum_stenosis"),
        "sleep_apnoea_patients":            N("sleep_apnoea"),
        "spinal_stenosis_patients":         N("spinal_stenosis"),
        "scoliosis_patients":               N("scoliosis"),
        "early_oa_patients":                N("early_oa"),
        "retinal_detachment_patients":      N("retinal_detachment"),
        "osteochondroma_patients":          N("osteochondroma"),
        "joint_laxity_patients":            N("joint_laxity"),
        "cauliflower_ear_patients":         N("cauliflower_ear"),
        "advanced_bone_age_patients":       N("advanced_bone_age"),
        "malignant_transformation_patients":N("malignant_transformation"),
        "club_foot_patients":               N("club_foot"),
        "cleft_palate_patients":            N("cleft_palate"),
        "gene_patient_counts": {g["gene"]: 40 for g in SD_GENES},
        "pathway": (
            "Hereditary skeletal dysplasias — shared mechanism: "
            "germline variants disrupt cartilage/bone extracellular matrix or signalling pathways: "
            "FGFR3 GOF → excess tyrosine kinase signalling → inhibited chondrocyte proliferation (ACH/TD); "
            "COL2A1 LOF/dominant-negative → defective type II collagen fibril formation (vitreous, cartilage, disc); "
            "EXT1/EXT2 LOF → impaired heparan sulphate proteoglycan biosynthesis → growth plate exostoses; "
            "COMP dominant-negative → ER retention of COMP pentamer → chondrocyte apoptosis; "
            "SLC26A2 LOF → sulphate transporter dysfunction → undersulfated proteoglycans → defective cartilage; "
            "TRPV4 GOF → excess Ca2+ influx → disrupted chondrocyte/osteoblast function; "
            "ACAN LOF → reduced aggrecan → defective cartilage matrix → advanced bone age + short stature."
        ),
        "key_clinical_insight": (
            "Critical DDx in the clinic: FGFR3-ACH (macrocephaly + normal face) vs COMP-PSACH (NORMAL face + NORMAL head — NOT apparent at birth); "
            "BOTH require urgent workup but for DIFFERENT complications: ACH → foramen magnum stenosis screen; PSACH → C1-C2 instability screen. "
            "EXT1 carries HIGHER sarcoma risk than EXT2 — molecular diagnosis essential for surveillance intensity. "
            "ACAN-FSS hallmark: advanced bone age (opposite of GH deficiency) with normal GH axis. "
            "SLC26A2-DTD triad (cauliflower ear + hitchhiker thumb + club foot) is pathognomonic."
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in SD_GENES:
        gene = gene_entry["gene"]
        cohort = _generate_cohort(gene_entry)

        def pct(key): return round(100 * sum(1 for p in cohort if p[key]) / len(cohort))

        result[gene] = {
            "gene":                  gene,
            "locus":                 gene_entry["locus"],
            "protein_size":          gene_entry["protein_size"],
            "inheritance":           gene_entry["inheritance"],
            "n_patients":            len(cohort),
            "rhizomelic_short_stature_pct": pct("rhizomelic_short_stature"),
            "macrocephaly_pct":      pct("macrocephaly"),
            "foramen_magnum_stenosis_pct": pct("foramen_magnum_stenosis"),
            "sleep_apnoea_pct":      pct("sleep_apnoea"),
            "spinal_stenosis_pct":   pct("spinal_stenosis"),
            "scoliosis_pct":         pct("scoliosis"),
            "early_oa_pct":          pct("early_oa"),
            "retinal_detachment_pct":pct("retinal_detachment"),
            "osteochondroma_pct":    pct("osteochondroma"),
            "joint_laxity_pct":      pct("joint_laxity"),
            "cauliflower_ear_pct":   pct("cauliflower_ear"),
            "advanced_bone_age_pct": pct("advanced_bone_age"),
            "malignant_transformation_pct": pct("malignant_transformation"),
            "club_foot_pct":         pct("club_foot"),
            "cleft_palate_pct":      pct("cleft_palate"),
            "age_of_onset":   gene_entry["age_of_onset"],
            "key_biomarker":  gene_entry["key_biomarker"],
            "pathognomonic":  gene_entry["pathognomonic"],
            "treatment":      gene_entry["treatment"],
            "critical_flags": gene_entry["critical_flags"],
            "seed":           gene_entry["seed"],
            "cohort_preview": cohort[:5],
        }
    return result


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Skeletal-Dysplasia-Atlas",
        "pathway": "FGFR3-Signalling / Type-II-Collagen / Heparan-Sulphate-HSPG / COMP-ECM / Sulphate-Transport / TRPV4-Ca2+ / Aggrecan-Matrix",
        "shared_mechanism": (
            "Hereditary skeletal dysplasias arise from germline variants disrupting cartilage and bone development. "
            "FGFR3 gain-of-function (G380R → ACH) constitutively activates STAT1/MAPK signalling, inhibiting chondrocyte proliferation in growth plates — "
            "the most common lethal skeletal dysplasia. "
            "COL2A1 type II collagen defects cause multisystem disease: vitreous anomaly, early OA, platyspondyly — "
            "severity ranges from Stickler (mild) to SEDC to Kniest (severe). "
            "EXT1/EXT2 heparan sulphate proteoglycan synthesis defects impair Indian Hedgehog and FGF signalling gradients → growth plate osteochondromas. "
            "COMP dominant-negative misfolding causes ER stress and chondrocyte apoptosis → PSACH/MED. "
            "SLC26A2 sulphate transporter loss → undersulfated aggrecan and other PGs → defective cartilage ECM → DTD spectrum. "
            "TRPV4 gain-of-function → excess Ca2+ influx in chondrocytes → severe metatropic dysplasia to mild brachyolmia. "
            "ACAN (aggrecan gene) haploinsufficiency → reduced aggrecan quantity → impaired growth plate → advanced bone age + short stature."
        ),
        "genes": {
            g["gene"]: {
                "full_name": g["alt_name"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "critical_flags": g["critical_flags"],
                "pathognomonic": g["pathognomonic"],
                "treatment_summary": g["treatment"],
            }
            for g in SD_GENES
        },
        "glossary": {
            "Achondroplasia (ACH)": "FGFR3 G380R GOF — most common non-lethal skeletal dysplasia; rhizomelic short stature + macrocephaly + trident hand + normal intelligence; vosoritide FDA 2021; foramen magnum screen year 1",
            "Thanatophoric Dysplasia (TD)": "FGFR3 K650E/M GOF — lethal perinatally; cloverleaf skull (TD2); telephone-receiver femur; severely narrow thorax → respiratory failure at birth",
            "Hypochondroplasia": "FGFR3 N540K — milder ACH allele; near-normal proportions; mild short stature; often diagnosed at 2-5 yr; head circumference normal or mildly increased",
            "Vosoritide": "Recombinant C-type natriuretic peptide (CNP) analogue; 15 µg/kg SC daily; FDA/EMA approved 2021 for ACH (>2 yr); increases annualised height velocity ~1.6 cm/yr; natriuretic peptide pathway opposes FGFR3 signalling",
            "Foramen Magnum Stenosis": "ACH — posterior fossa compression of brainstem at C0-C1 level; risk: sudden infant death, central apnoea, myelopathy; CINE-MRI quantifies CSF flow impairment; decompression surgery if severe (<6-12 months)",
            "Stickler Syndrome Type 1 (COL2A1)": "Congenital MEMBRANOUS vitreous anomaly PATHOGNOMONIC; high myopia >-10D; Pierre Robin sequence 50%; retinal detachment 40-60% lifetime; progressive SNHL; early OA; distinguish Type 2 (COL11A1 beaded vitreous)",
            "SEDC (Spondyloepiphyseal Dysplasia Congenita)": "COL2A1 — severe end; short trunk + platyspondyly + epiphyseal dysplasia from birth; ODONTOID HYPOPLASIA = C1-C2 instability; cleft palate 50%; retinal detachment; contact sports ABSOLUTELY CI",
            "Multiple Hereditary Exostoses (MHE)": "EXT1 (chr8) or EXT2 (chr11) AD LOF; multiple cartilage-capped osteochondromas at metaphyses; skeletal deformity (forearm/ankle/knee); sarcoma risk: EXT1 1-2%, EXT2 0.5%; cap >2 cm in skeletally mature = urgent biopsy",
            "Osteochondroma (Exostosis)": "Cartilage-capped bony projection from metaphysis; grows parallel to bone during childhood; ceases at skeletal maturity; new growth post-maturity = sarcomatous transformation until excluded",
            "Pseudoachondroplasia (PSACH)": "COMP — NORMAL FACE + NORMAL HEAD CIRCUMFERENCE (CRITICAL DDx from ACH) — not apparent at birth — seen at ambulation 18 months; extreme joint laxity; C1-C2 instability; severe early OA; contact sports ABSOLUTELY CI",
            "Multiple Epiphyseal Dysplasia (MED)": "COMP (or MATN3, COL9A1/A2/A3) AD — milder than PSACH; epiphyseal irregularity; childhood joint pain; early OA of hips/knees; mild short stature; platyspondyly absent or mild",
            "Diastrophic Dysplasia (DTD)": "SLC26A2 biallelic — CAULIFLOWER EAR (pinna inflammation → calcification) + HITCHHIKER THUMB (bilateral first MCP abduction) + BILATERAL CLUB FOOT = pathognomonic triad; cervical kyphosis; scoliosis; cleft palate 25-35%; AR inheritance",
            "Achondrogenesis Type 1B (ACG1B)": "SLC26A2 biallelic null mutations — most severe end of spectrum; lethal perinatally; severe micromelia; absent ossification; distinguish from ACG1A (TRIP11)",
            "Metatropic Dysplasia": "TRPV4 GOF — dumbbell/halberd metaphyseal appearance PATHOGNOMONIC; progressive kyphoscoliosis → respiratory failure; proportions change with age (metatropic = changing shape); requires respiratory surveillance",
            "Brachyolmia": "TRPV4 mild GOF (or PAPSS2) — short trunk only; platyspondyly; near-normal limb length; diagnosed often in adulthood with back pain and short stature review",
            "ACAN-FSS (Familial Short Stature)": "ACAN haploinsufficiency AD — ADVANCED bone age (ahead of chronological) PATHOGNOMONIC; normal GH axis; GH therapy partial response limited by bone age; osteochondritis dissecans association; early disc disease",
            "Advanced Bone Age": "Bone age ahead of chronological age on left hand X-ray (Greulich-Pyle or TW3); distinguishes ACAN-FSS from GH deficiency (delayed), Turner (delayed), or normal short stature (concordant)",
            "Rhizomelic shortening": "Shortening of proximal limb segments (humerus, femur) greater than distal; characteristic of ACH (FGFR3) and PSACH (COMP)",
            "Platyspondyly": "Flat vertebral bodies on lateral spine X-ray; feature of COL2A1, TRPV4, SLC26A2, COMP, and ACAN disorders",
            "Odontoid Hypoplasia": "Underdeveloped odontoid process (dens) of C2; risk of atlantoaxial instability and spinal cord compression; found in COL2A1 SEDC and COMP PSACH; flexion-extension MRI mandatory",
        },
        "surveillance_protocols": {
            "FGFR3": "Brain/spine MRI year 1 (foramen magnum); polysomnography 6-12 months; adenotonsillectomy/CPAP if OSA; audiometry annually; spine MRI if neurological symptoms; annual neurodevelopment review; vosoritide: bone age and growth velocity 6-monthly",
            "COL2A1": "Annual ophthalmology (retinal detachment surveillance + myopia); 360° prophylactic laser to lattice at diagnosis; audiometry annually; cervical spine X-ray SEDC (odontoid); echocardiogram MVP; joint review (early OA planning); Pierre Robin — NICU airway team at delivery",
            "EXT1": "Annual clinical examination of known exostoses; MRI of growing/painful lesions; cap thickness ≤2 cm annual review; cap >2 cm → urgent biopsy; skeletal survey at diagnosis; patient self-examination education; forearm/ankle X-ray for deformity progression",
            "EXT2": "Identical surveillance protocol to EXT1 despite lower average risk; cap >2 cm → urgent biopsy regardless of EXT2 genotype",
            "COMP": "C1-C2 stability (cervical spine X-ray + MRI) — contact sports ABSOLUTELY CI if instability; annual orthopaedic review; scoliosis monitoring (Cobb angle 6-monthly if progressive); pain management review; joint replacement planning from 4th decade",
            "SLC26A2": "Neonatal: cervical spine MRI (kyphosis); serial casting club foot from birth; ENT (cleft palate); physiotherapy; scoliosis monitoring; spinal MRI if neurological symptoms; orthopaedic review annually",
            "TRPV4": "Respiratory function tests 6-monthly (metatropic); polysomnography; kyphoscoliosis Cobb angle 6-monthly; chest CT (lung volumes); spinal cord MRI if neurological change; NIV titration if respiratory impairment",
            "ACAN": "Bone age X-ray 6-monthly during GH therapy; growth velocity 3-monthly; knee/elbow MRI if OCD suspected; spine MRI (disc disease) in adults; GH axis testing at diagnosis to exclude GH deficiency; audiometry (SED Kimberley biallelic — SNHL)",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Short stature: {ov['short_stature_patients']}")
    print(f"Scoliosis: {ov['scoliosis_patients']}")
    print(f"Early OA: {ov['early_oa_patients']}")
    print(f"Osteochondroma: {ov['osteochondroma_patients']}")
    print(f"Malignant transformation: {ov['malignant_transformation_patients']}")
    print(f"Cauliflower ear: {ov['cauliflower_ear_patients']}")
    print(f"Advanced bone age: {ov['advanced_bone_age_patients']}")
