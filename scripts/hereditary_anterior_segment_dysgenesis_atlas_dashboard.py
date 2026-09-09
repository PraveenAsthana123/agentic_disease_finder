#!/usr/bin/env python3
"""Hereditary-Anterior-Segment-Dysgenesis-Atlas — Complete 8-Gene Atlas
(PITX2 · FOXC1 · PAX6 · FOXE3 · B3GLCT · PXDN · HCCS · CYP1B1).

PITX2   (Paired-Like Homeodomain 2; 317 aa; ~35 kDa; 4q25; AD;
          Axenfeld-Rieger Syndrome Type 1 (ARS1) — IRIDOCORNEAL ADHESION STRANDS +
          POSTERIOR EMBRYOTOXON + DYSPLASTIC IRIS STROMA TRIAD PATHOGNOMONIC;
          GLAUCOMA 50% lifetime; DENTAL ANOMALIES (hypodontia/oligodontia) + UMBILICAL HERNIA;
          seed SEED_BASE+0).
FOXC1   (Forkhead Box C1; 553 aa; ~60 kDa; 6p25.3; AD;
          Axenfeld-Rieger Syndrome Type 3 (ARS3) / Iridogoniodysgenesis Type 1;
          POSTERIOR EMBRYOTOXON + IRIS HYPOPLASIA + ELEVATED IOP PATHOGNOMONIC;
          Cardiac septal defects 8%; FOXC1 duplication → iridogoniodysgenesis;
          seed SEED_BASE+1).
PAX6    (Paired Box 6; 422 aa; ~46 kDa; 11p13; AD;
          Aniridia type II — BILATERAL NEAR-TOTAL IRIS ABSENCE + FOVEAL HYPOPLASIA +
          PENDULAR NYSTAGMUS PATHOGNOMONIC TRIAD; WAGR deletion: Wilms + Aniridia + GU anomalies + ID;
          Progressive keratopathy; glaucoma 30-50%; seed SEED_BASE+2).
FOXE3   (Forkhead Box E3; 338 aa; ~38 kDa; 1p33; AD/AR;
          Anterior Segment Dysgenesis type 2 — Peters anomaly / sclerocornea / primary aphakia;
          ANTERIOR LENS ADHESION TO CORNEA (LENS TOUCH) PATHOGNOMONIC;
          Congenital cataract + microphthalmia + coloboma in AR; seed SEED_BASE+3).
B3GLCT  (Beta-3-Glucosyltransferase; 498 aa; ~57 kDa; 13q12.3; AR;
          Peters Plus Syndrome — CORNEAL CLOUDING + SHORT STATURE + INTELLECTUAL DISABILITY +
          CLEFT LIP/PALATE = DIAGNOSTIC TETRAD; glycosylation of thrombospondin-1 repeats;
          Peters anomaly is the universal ocular hallmark; seed SEED_BASE+4).
PXDN    (Peroxidasin; 1479 aa; ~165 kDa; 2p25.3; AR;
          Anterior Segment Dysgenesis type 7 — BILATERAL CONGENITAL CORNEAL OPACIFICATION
          AT BIRTH + SCLEROCORNEA PATHOGNOMONIC; peroxidasin crosslinks collagen IV in anterior
          basement membranes; absent crosslinks → abnormal corneal development; seed SEED_BASE+5).
HCCS    (Holocytochrome C-Type Synthase; 295 aa; ~33 kDa; Xp22.2; XLD;
          MIDAS / MLS Syndrome — PERIOCULAR/FACIAL LINEAR SKIN DEFECTS + MICROPHTHALMIA
          IN FEMALES PATHOGNOMONIC; LETHAL IN HEMIZYGOUS MALES (X-linked dominant lethal);
          cardiac defects 20%; CNS structural anomalies 30%; seed SEED_BASE+6).
CYP1B1  (Cytochrome P450 Family 1 Subfamily B Member 1; 543 aa; ~60 kDa; 2p22.2; AR;
          Peters Anomaly type 2 / Primary Congenital Glaucoma (PCG) — CENTRAL CORNEAL OPACITY
          AT BIRTH + HIGH IOP + BUPHTHALMOS PATHOGNOMONIC; metabolizes steroids in trabecular
          meshwork development; G61E, R368H, R390H common; seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2390-2397).
"""

import random

SEED_BASE = 2390

ASD_GENES = [
    # -- PITX2 -- Axenfeld-Rieger Syndrome Type 1 -------------------------------------------
    {
        "gene": "PITX2",
        "alt_name": (
            "PITX2 (PITX2-317aa-4q25 / AD -- "
            "ARS1-AXENFELD-RIEGER-SYNDROME-TYPE-1-MOST-COMMON-ARS-GENE -- "
            "IRIDOCORNEAL-ADHESION-STRANDS+POSTERIOR-EMBRYOTOXON+DYSPLASTIC-IRIS-TRIAD-PATHOGNOMONIC -- "
            "GLAUCOMA-50pct-LIFETIME-GONIOTOMY-TRABECULOTOMY -- "
            "DENTAL-ANOMALIES-HYPODONTIA-OLIGODONTIA+UMBILICAL-HERNIA-EXTRAOCULAR-HALLMARKS)"
        ),
        "protein": (
            "PITX2 -- 4q25 AD -- PITX2-317aa -- "
            "Paired-Like-Homeodomain-Transcription-Factor-2-RIEG1-35kDa-Bicoid-Subclass -- "
            "N-Terminal-Paired-Like-Homeodomain-OAR-Domain-C-Terminal -- "
            "Regulates-NCC-Neural-Crest-Cell-Migration-Into-Anterior-Segment -- "
            "Expressed-Eye-Teeth-Umbilicus-Pituitary-Heart-Developing-Embryo -- "
            "OMIM-Gene-601542-Disease-ARS1-180500"
        ),
        "locus": "4q25",
        "protein_size": "317 aa / 35 kDa",
        "inheritance": (
            "AD — haploinsufficiency is the primary mechanism; PITX2 mutations account for ~50-70% of all ARS cases; "
            "RIEG1 locus; autosomal dominant with incomplete penetrance and variable expressivity; "
            "de novo mutations common (40%); familial cases show variable expressivity; "
            "no clear genotype-phenotype correlation; truncating = frameshift = missense — all cause similar ARS phenotype; "
            "isoforms: PITX2A (longest), PITX2B, PITX2C — isoform C most important for cardiac laterality (situs inversus if biallelic)"
        ),
        "disease_category": "Axenfeld-Rieger Syndrome Type 1 (ARS1) — anterior segment dysgenesis + glaucoma + dental + umbilical anomalies",
        "disease_pathway": (
            "PITX2 encodes a paired-like homeodomain transcription factor critical for neural crest cell (NCC) migration and differentiation "
            "into structures of the anterior segment of the eye, teeth, and umbilicus. "
            "DEVELOPMENTAL ROLE: PITX2 drives NCC to form trabecular meshwork, corneal stroma, iris stroma, and ciliary body. "
            "PATHOMECHANISM: PITX2 haploinsufficiency → insufficient NCC-derived anterior segment mesenchyme → "
            "incomplete maturation of trabecular meshwork + iris stroma + corneoscleral angle. "
            "OCULAR CONSEQUENCES: "
            "1. POSTERIOR EMBRYOTOXON: anteriorly displaced Schwalbe line (junction of Descemet membrane + trabecular meshwork) — "
            "visible as prominent white ring at limbus by slit lamp or gonoscopy. "
            "2. IRIDOCORNEAL ADHESION STRANDS (Axenfeld anomaly): NCC bridges connect iris surface to posterior embryotoxon — "
            "these strands impede aqueous outflow. "
            "3. IRIS STROMA HYPOPLASIA: dysplastic iris with thinned stroma, correctopia (displaced pupil), polycoria (false). "
            "4. GLAUCOMA (50% lifetime risk): trabecular meshwork hypoplasia → elevated IOP → "
            "requires early goniotomy/trabeculotomy; angle surgery preferred over trabeculectomy in children. "
            "SYSTEMIC: DENTAL (hypodontia — peg-shaped teeth, oligodontia, maxillary incisor reduction); "
            "UMBILICAL (umbilical hernia — redundant periumbilical skin); "
            "PITUITARY (growth hormone deficiency in some); CARDIAC (less common than FOXC1)."
        ),
        "pathognomonic": (
            "IRIDOCORNEAL ADHESION STRANDS (iris bridges to posterior embryotoxon) + POSTERIOR EMBRYOTOXON "
            "(anteriorly displaced Schwalbe line visible at slit lamp as prominent limbal ring) + "
            "DYSPLASTIC IRIS STROMA (hypoplastic stroma, correctopia, pseudopolycoria) = ARS TRIAD PATHOGNOMONIC. "
            "DENTAL: HYPODONTIA (reduced number of teeth, especially maxillary incisors reduced/peg-shaped). "
            "UMBILICAL: REDUNDANT PERIUMBILICAL SKIN (soft, non-reducible 'hernial' appearance). "
            "KEY DDx FROM FOXC1 (ARS3): PITX2 has DENTAL + UMBILICAL anomalies more frequently; "
            "FOXC1 has cardiac defects more frequently; both share anterior segment findings; "
            "Iris findings: PITX2 often more prominent correctopia. "
            "Isolated POSTERIOR EMBRYOTOXON (Axenfeld anomaly alone) without iris strands = normal variant (15% of population). "
            "GLAUCOMA RISK: 50% lifetime — commences in childhood; progressive despite IOP control."
        ),
        "treatment": (
            "GLAUCOMA MANAGEMENT (50% lifetime risk): "
            "First-line: goniotomy (NdYAG or surgical) or trabeculotomy — angle surgery preferred in children; "
            "Medical: topical prostaglandins, beta-blockers, CAIs — may delay surgery; "
            "Trabeculectomy + MMC: if angle surgery fails; "
            "Glaucoma drainage implant (Ahmed/Baerveldt): for refractory cases. "
            "OCULAR REFRACTION: regular monitoring for myopia + astigmatism due to iris/angle anomalies. "
            "DENTAL: orthodontic evaluation — dental implants for oligodontia at skeletal maturity. "
            "PITUITARY SCREEN: height velocity monitoring — endocrinology if short stature. "
            "GENETIC COUNSELLING: 50% AD recurrence risk; ophthalmology for all first-degree relatives. "
            "ANNUAL: IOP + optic disc + visual field (from school age). "
            "AVOID: corticosteroid eye drops (steroid-responder risk elevated in ARS)."
        ),
        "key_features": [
            "Posterior embryotoxon (anteriorly displaced Schwalbe line) — slit lamp/gonioscopy",
            "Iridocorneal adhesion strands (iris bridges) — pathognomonic",
            "Iris stromal hypoplasia + correctopia + pseudopolycoria",
            "Glaucoma 50% lifetime — trabecular meshwork hypoplasia",
            "Dental: hypodontia, peg teeth, reduced maxillary incisors",
            "Umbilical: redundant periumbilical skin",
            "PITX2 haploinsufficiency — NCC migration defect",
            "AD with incomplete penetrance, variable expressivity",
        ],
        "key_ddx": (
            "FOXC1 (ARS3): identical anterior segment — FOXC1 has more cardiac defects; PITX2 has more dental/umbilical; "
            "CYP1B1: primary congenital glaucoma with corneal clouding — no iris strands, no dental; "
            "ICE syndrome (iridocorneal endothelial): unilateral, older onset, acquired iris atrophy — "
            "corneal endothelial cells migrate across angle; no gene; "
            "Isolated posterior embryotoxon: present in 15% general population — normal variant IF NO iris strands; "
            "Peters anomaly: central corneal opacity with absent posterior stroma — different from ARS anterior strands."
        ),
        "systemic_involvement": True,
        "onset_age": "Congenital/neonatal anterior segment; glaucoma childhood to adult",
        "surgical_urgency": "Moderate — glaucoma surgery when IOP uncontrolled; dental management at school age",
        "gene_family": "Paired-like homeodomain transcription factor (RIEG1 subclass)",
        "morphology": "Iridocorneal strands + posterior embryotoxon + iris hypoplasia",
    },

    # -- FOXC1 -- Axenfeld-Rieger Syndrome Type 3 / Iridogoniodysgenesis ----------------------
    {
        "gene": "FOXC1",
        "alt_name": (
            "FOXC1 (FOXC1-553aa-6p25.3 / AD -- "
            "ARS3-AXENFELD-RIEGER-SYNDROME-TYPE-3-IRIDOGONIODYSGENESIS-TYPE-1 -- "
            "POSTERIOR-EMBRYOTOXON+IRIS-HYPOPLASIA+ELEVATED-IOP-PATHOGNOMONIC -- "
            "CARDIAC-SEPTAL-DEFECTS-8pct-KEY-DDx-PITX2 -- "
            "FOXC1-DUPLICATION-IRIDOGONIODYSGENESIS-DISTINCT-FROM-LOF)"
        ),
        "protein": (
            "FOXC1 -- 6p25.3 AD -- FOXC1-553aa -- "
            "Forkhead-Box-C1-FREAC3-IRID1-60kDa-Forkhead-Transcription-Factor -- "
            "Forkhead-DNA-Binding-Domain-Nuclear-Localization-Signal-Transactivation-Domain -- "
            "Regulates-NCC-Differentiation-Anterior-Segment-Mesodermal-Derivatives -- "
            "Expressed-Eye-Heart-Kidney-Lung-Vasculature-Developing-Embryo -- "
            "OMIM-Gene-601090-Disease-ARS3-602482-IRID1-601656"
        ),
        "locus": "6p25.3",
        "protein_size": "553 aa / 60 kDa",
        "inheritance": (
            "AD — both haploinsufficiency (LOF: deletions, truncations, missense) and triplosensitivity (duplication) cause disease; "
            "FOXC1 LOF: Axenfeld-Rieger Syndrome type 3 (ARS3) / iridogoniodysgenesis anomaly type 1 (IGDA1); "
            "FOXC1 duplication (6p25 copy number gain): iridogoniodysgenesis anomaly type 2 (IGDA2) — glaucoma without dental/umbilical; "
            "de novo mutations: ~40% of LOF cases; familial AD cases common; "
            "Digenic disease: FOXC1 + PITX2 double heterozygotes → more severe phenotype; "
            "No homozygous LOF reported in humans (likely lethal — Foxc1-/- mice die perinatally with cardiovascular defects)"
        ),
        "disease_category": "Axenfeld-Rieger Syndrome Type 3 (ARS3) — anterior segment + glaucoma + cardiac anomalies",
        "disease_pathway": (
            "FOXC1 encodes Forkhead Box C1, a forkhead transcription factor essential for neural crest cell (NCC) differentiation "
            "into anterior segment mesenchyme AND for cardiovascular development (heart septation, valve formation, "
            "great vessel patterning from cardiac NCCs). "
            "PATHOMECHANISM (LOF): FOXC1 haploinsufficiency → impaired NCC differentiation → "
            "trabecular meshwork hypoplasia + iris stromal hypoplasia + posterior embryotoxon; "
            "simultaneously → cardiac NCC defect → septal defects (ASD, VSD) in ~8% of patients. "
            "PATHOMECHANISM (DUPLICATION): FOXC1 overdose → iridogoniodysgenesis without dental/umbilical — "
            "excess FOXC1 disrupts the precise transcriptional balance needed for proper anterior segment NCC differentiation; "
            "mechanism is distinct from haploinsufficiency — gain-of-function at the transcriptional level. "
            "GLAUCOMA: trabecular meshwork hypoplasia → chronic open-angle or mixed mechanism; "
            "FOXC1 also regulates VEGF signalling in trabecular meshwork drainage — "
            "pathogenic variants may dysregulate aqueous outflow even with some NCC present. "
            "HEARING LOSS: FOXC1 expressed in developing inner ear — sensorineural hearing loss reported in some families. "
            "KEY DISTINCTION FROM PITX2 (ARS1): FOXC1 has cardiac defects more frequently (8% vs <3% for PITX2); "
            "dental anomalies LESS common in FOXC1 ARS3; FOXC1 duplication = iridogoniodysgenesis only (no systemic)."
        ),
        "pathognomonic": (
            "POSTERIOR EMBRYOTOXON + IRIS STROMAL HYPOPLASIA + IRIDOCORNEAL STRANDS = same ARS triad as PITX2 "
            "(clinically indistinguishable without genetic testing). "
            "CARDIAC DEFECT (8%): ASD, VSD, or conotruncal — FOXC1-specific discriminator from PITX2 (cardiac in PITX2 <3%). "
            "FOXC1 DUPLICATION: IRIDOGONIODYSGENESIS (iris hypoplasia + raised IOP) WITHOUT dental or umbilical anomalies — "
            "diagnosis missed if only coding sequencing done (MLPA or CNV array required). "
            "HEARING LOSS: sensorineural in subset — audiological screening mandatory. "
            "GLAUCOMA (50% lifetime): same risk as PITX2 — trabecular meshwork hypoplasia. "
            "KEY: genetic panel testing (PITX2 + FOXC1) required to distinguish ARS1 from ARS3 — "
            "clinical phenotype alone cannot reliably differentiate."
        ),
        "treatment": (
            "GLAUCOMA MANAGEMENT: same as PITX2-ARS1 — goniotomy/trabeculotomy first-line in children; "
            "medical IOP lowering: prostaglandins, beta-blockers, CAIs; "
            "trabeculectomy + MMC or GDD for refractory cases. "
            "CARDIAC SCREEN: echo + ECG at diagnosis — refer to paediatric cardiology if structural defect; "
            "cardiology follow-up before any general anaesthesia (IOP surgery). "
            "HEARING SCREEN: audiological assessment at diagnosis and every 2-3 years; "
            "hearing aids if SNHL confirmed. "
            "FOXC1 DUPLICATION CASES: MLPA/CNV array — targeted glaucoma surveillance; "
            "no dental/systemic workup required for duplication phenotype. "
            "GENETIC COUNSELLING: 50% AD recurrence; molecular testing of parents mandatory (de novo vs inherited)."
        ),
        "key_features": [
            "Posterior embryotoxon + iris hypoplasia + iridocorneal strands (ARS triad)",
            "Glaucoma 50% lifetime — same as PITX2",
            "Cardiac septal defects 8% — KEY DDx from PITX2",
            "FOXC1 duplication → iridogoniodysgenesis (no systemic) — MLPA required",
            "Sensorineural hearing loss subset — audiological screening mandatory",
            "Dental anomalies LESS common than PITX2",
            "Digenic FOXC1 + PITX2: more severe phenotype",
            "No homozygous LOF in humans (lethal)",
        ],
        "key_ddx": (
            "PITX2 (ARS1): same anterior segment — PITX2 has more dental/umbilical; FOXC1 has more cardiac; "
            "CYP1B1 (PCG): corneal clouding + high IOP at birth — no iris strands, no posterior embryotoxon; "
            "PAX6 (aniridia): complete iris absence, foveal hypoplasia, nystagmus — no iridocorneal strands; "
            "FOXC1 duplication (IGDA2) vs FOXC1 LOF (ARS3): duplication = no dental/umbilical — "
            "CNV array essential in iridogoniodysgenesis without systemic features."
        ),
        "systemic_involvement": True,
        "onset_age": "Congenital anterior segment; glaucoma childhood to adult",
        "surgical_urgency": "Moderate — glaucoma surgery when IOP uncontrolled; cardiac pre-op workup mandatory",
        "gene_family": "Forkhead Box transcription factor (FOXC subclass)",
        "morphology": "Iridocorneal strands + posterior embryotoxon + iris hypoplasia",
    },

    # -- PAX6 -- Aniridia / WAGR syndrome -----------------------------------------------------
    {
        "gene": "PAX6",
        "alt_name": (
            "PAX6 (PAX6-422aa-11p13 / AD -- "
            "ANIRIDIA-TYPE-II-BILATERAL-NEAR-TOTAL-IRIS-ABSENCE-PATHOGNOMONIC -- "
            "FOVEAL-HYPOPLASIA+PENDULAR-NYSTAGMUS-PATHOGNOMONIC-TRIAD -- "
            "WAGR-DELETION-11p13-WILMS-TUMOR-ANIRIDIA-GU-ANOMALIES-ID -- "
            "PROGRESSIVE-KERATOPATHY-CORNEAL-STEM-CELL-DEFICIENCY)"
        ),
        "protein": (
            "PAX6 -- 11p13 AD -- PAX6-422aa -- "
            "Paired-Box-Protein-PAX6-ANIRIDIA-AN2-46kDa-Master-Eye-Regulator -- "
            "N-Terminal-Paired-Domain-Linker-Homeodomain-Transactivation-Domain -- "
            "Master-Transcription-Factor-Eye-Pancreas-Nose-Brain-Development -- "
            "Regulates-Iris-Retina-Corneal-Epithelium-Lens-Development -- "
            "OMIM-Gene-607108-Disease-Aniridia-106210-WAGR-194072"
        ),
        "locus": "11p13",
        "protein_size": "422 aa / 46 kDa",
        "inheritance": (
            "AD — haploinsufficiency; PAX6 is a dosage-sensitive master eye transcription factor; "
            "point mutations (missense, truncating, splice): aniridia type II (AN2); "
            "11p13 DELETION encompassing PAX6 + WT1: WAGR syndrome (Wilms tumour + Aniridia + GU anomalies + Range of ID); "
            "CHROMOSOME ARRAY mandatory in all new aniridia diagnoses to exclude 11p13 deletion (WAGR); "
            "homozygous PAX6 LOF: bilateral anophthalmia + absent nasal/olfactory structures + brain anomalies (lethal); "
            "AR hypomorphic PAX6: foveal hypoplasia without aniridia (FHIPA); "
            "de novo: 30-40% of cases"
        ),
        "disease_category": "Aniridia (AN2/PAX6) — complete iris absence + foveal hypoplasia + nystagmus + progressive keratopathy + glaucoma",
        "disease_pathway": (
            "PAX6 is the master transcription factor for eye development, acting as a 'master control gene' — "
            "ectopic PAX6 expression in Drosophila produces ectopic eyes anywhere on the body. "
            "In humans, PAX6 haploinsufficiency produces a complex pan-ocular phenotype: "
            "IRIS: PAX6 is required for iris progenitor cell specification from optic cup marginal zone — "
            "haploinsufficiency → near-total iris absence (bilateral, with only rudimentary iris remnant). "
            "FOVEA: PAX6 required for cone photoreceptor development and foveal pit formation — "
            "haploinsufficiency → foveal hypoplasia (no avascular zone, no pit) → PENDULAR NYSTAGMUS (visual defect drives nystagmus). "
            "CORNEA: PAX6 is essential for limbal stem cells (LSC) — LSC depletion in aniridia → "
            "PROGRESSIVE KERATOPATHY (conjunctival goblet cell ingrowth → corneal vascularisation + opacification) "
            "— begins as superior pannus in 1st decade, can progress to total corneal opacification. "
            "LENS: PAX6 required for crystallin expression → cataract (posterior subcapsular or nuclear) in 50-85%. "
            "GLAUCOMA: trabecular meshwork hypoplasia → IOP elevation in 30-50% (angle closure mechanism as iris remnant migrates). "
            "WAGR DELETION: 11p13 deletion includes WT1 (Wilms tumour suppressor) → "
            "Wilms tumour risk 45-60% (bilateral in 10%); annual renal ultrasound mandatory until age 8."
        ),
        "pathognomonic": (
            "BILATERAL NEAR-TOTAL IRIS ABSENCE (only rudimentary iris stumps visible on slit lamp gonioscopy) + "
            "FOVEAL HYPOPLASIA (OCT: absent foveal pit, absent avascular zone, continuity of inner nuclear layer through fovea) + "
            "PENDULAR HORIZONTAL NYSTAGMUS = ANIRIDIA TRIAD PATHOGNOMONIC. "
            "WAGR SYNDROME (11p13 DELETION): aniridia + Wilms tumour risk + GU anomalies (cryptorchidism, hypospadias) + "
            "intellectual disability — MANDATORY chromosome microarray in ALL aniridia to exclude. "
            "PROGRESSIVE KERATOPATHY: superior pannus → vascular ingrowth → corneal opacification — "
            "PAX6 aniridia is a LIMBAL STEM CELL DEFICIENCY DISEASE. "
            "CATARACT (50-85%): posterior subcapsular most common — slit-lamp annual monitoring. "
            "GLAUCOMA (30-50%): late-onset, open or closed-angle. "
            "PHOTOPHOBIA: severe — iris absence → uncontrolled light entry."
        ),
        "treatment": (
            "KERATOPATHY (limbal stem cell deficiency): "
            "Preservative-free lubricants — corneal epithelial protection; "
            "Scleral contact lens — irregular cornea correction; "
            "Cultivated limbal epithelial transplantation (CLET) / simple limbal epithelial transplantation (SLET) "
            "for advanced keratopathy — donor site from healthy eye or allogeneic; "
            "Corneal graft (DALK/PKP) as last resort — high rejection risk without LSC therapy first. "
            "NYSTAGMUS: contact lenses (reduce nystagmus intensity vs spectacles); "
            "prism adaptation; low vision aids. "
            "GLAUCOMA: same escalation as ARS — medical → goniotomy → GDD. "
            "WAGR DELETION: renal ultrasound every 3-6 months until age 8 (Wilms surveillance); "
            "urology for GU anomalies; educational support for ID range. "
            "CATARACT: paediatric cataract surgery when dense — amblyopia management. "
            "ANIRIDIA SPECTACLES / TINTED LENSES: improve photophobia and cosmesis. "
            "GENE THERAPY: ongoing clinical trials (AAV-PAX6 to limbal cells — phase 1/2)."
        ),
        "key_features": [
            "Bilateral near-total iris absence (only rudimentary stumps)",
            "Foveal hypoplasia on OCT — no pit, no avascular zone",
            "Pendular horizontal nystagmus",
            "Progressive limbal stem cell deficiency keratopathy",
            "Cataract 50-85% (posterior subcapsular)",
            "Glaucoma 30-50% (late onset)",
            "WAGR: 11p13 deletion → Wilms tumour risk 45-60%",
            "Chromosome microarray MANDATORY in all aniridia",
        ],
        "key_ddx": (
            "PITX2/FOXC1 (ARS): iris strands NOT absent iris — iridocorneal bridges visible; no foveal hypoplasia; no nystagmus; "
            "Traumatic aniridia: unilateral; history of trauma; "
            "Congenital glaucoma (buphthalmos): iris present but stretched; corneal oedema; "
            "Albinism: iris present but translucent; foveal hypoplasia without iris absence; "
            "Iris coloboma (PAX2 or sporadic): segmental iris defect (keyhole pupil) vs near-total absence; "
            "FRMD7-nystagmus: nystagmus WITHOUT iris absence or foveal hypoplasia."
        ),
        "systemic_involvement": True,
        "onset_age": "Congenital (nystagmus, iris absence); keratopathy progressive from 1st decade",
        "surgical_urgency": "Urgent WAGR screen (Wilms surveillance); progressive keratopathy management",
        "gene_family": "Paired-box (PAX) family transcription factor — master eye regulator",
        "morphology": "Near-total iris absence + foveal hypoplasia + progressive corneal opacification",
    },

    # -- FOXE3 -- Anterior Segment Mesenchymal Dysgenesis / Peters anomaly type 2 -------------
    {
        "gene": "FOXE3",
        "alt_name": (
            "FOXE3 (FOXE3-338aa-1p33 / AD/AR -- "
            "ASD2-ANTERIOR-SEGMENT-MESENCHYMAL-DYSGENESIS-TYPE-2 -- "
            "PETERS-ANOMALY-WITH-LENS-TOUCH-PATHOGNOMONIC-CENTRAL-CORNEAL-OPACITY -- "
            "AR-BIALLELIC-PRIMARY-APHAKIA-SCLEROCORNEA-MICROPHTHALMIA-COLOBOMA -- "
            "AD-HYPOMORPHIC-CATARACT-MICROCORNEA)"
        ),
        "protein": (
            "FOXE3 -- 1p33 AD/AR -- FOXE3-338aa -- "
            "Forkhead-Box-E3-DYSGNATHIA1-38kDa-Forkhead-Transcription-Factor -- "
            "Forkhead-DNA-Binding-Domain-Minimal-C-Terminal -- "
            "Expressed-Lens-Vesicle-Corneal-Epithelium-Anterior-Segment -- "
            "Essential-For-Lens-Vesicle-Separation-From-Surface-Ectoderm -- "
            "OMIM-Gene-601094-Disease-ASD2-610256"
        ),
        "locus": "1p33",
        "protein_size": "338 aa / 38 kDa",
        "inheritance": (
            "Bimodal: AD (hypomorphic) + AR (severe LOF); "
            "AR biallelic LOF: primary aphakia (absent lens) + sclerocornea + microphthalmia + iris coloboma — SEVERE; "
            "AD missense (hypomorphic): congenital cataract + microcornea + iris coloboma — MILDER; "
            "Peters anomaly (central corneal opacity with posterior defect ± lens adhesion): "
            "both AD (milder, lens touch) and AR (severe, sclerocornea, microphthalmia) reported; "
            "Incomplete penetrance and variable expressivity in AD families; "
            "FOXE3 mutations account for ~5-10% of Peters anomaly/anterior segment dysgenesis"
        ),
        "disease_category": "Anterior Segment Dysgenesis type 2 (ASD2) — Peters anomaly / primary aphakia / sclerocornea (FOXE3-associated)",
        "disease_pathway": (
            "FOXE3 encodes a forkhead transcription factor expressed specifically in the lens vesicle and anterior segment ectoderm. "
            "DEVELOPMENTAL ROLE: FOXE3 is required for lens vesicle SEPARATION from the overlying surface ectoderm (corneal progenitor). "
            "During normal eye development, the lens placode invaginates, separates from ectoderm, and closes as lens vesicle; "
            "FOXE3 drives the epithelial-mesenchymal transition needed for clean separation and lens vesicle closure. "
            "PATHOMECHANISM (AR LOF): FOXE3 complete loss → lens vesicle FAILS to separate from surface ectoderm → "
            "PRIMARY APHAKIA (lens is never formed as a separate structure) + "
            "SCLEROCORNEA (ectoderm covering anterior is not properly specified → opaque sclera-like tissue) + "
            "MICROPHTHALMIA (absent lens during development → smaller eye) + IRIS COLOBOMA. "
            "PATHOMECHANISM (AD hypomorphic): partial FOXE3 function → lens separates but is dysplastic → "
            "CONGENITAL CATARACT; cornea opacifies centrally due to incomplete separation (PETERS ANOMALY with lens touch — "
            "posterior corneal surface adheres to lens anterior capsule due to incomplete separation). "
            "SECONDARY: absent lens → no spatial signal for iris/ciliary body → coloboma; "
            "absent AQP0 (MIP) expression (FOXE3-regulated) contributes to lens opacity."
        ),
        "pathognomonic": (
            "CENTRAL CORNEAL OPACITY WITH ABSENT POSTERIOR STROMA + DESCEMET MEMBRANE + ENDOTHELIUM "
            "(Peters anomaly morphology on anterior segment OCT/ultrasound biomicroscopy) PATHOGNOMONIC. "
            "LENS TOUCH: anterior lens capsule adheres to posterior corneal surface in the opacity zone — "
            "visible on UBM as direct contact between lens and posterior cornea. "
            "AR BIALLELIC SEVERE: PRIMARY APHAKIA (no identifiable lens on ultrasound/OCT) + "
            "SCLEROCORNEA (diffuse corneal opacification resembling sclera, no limbal landmarks) + "
            "MICROPHTHALMIA (axial length <16mm newborn). "
            "AD MILDER: congenital cataract + microcornea + iris coloboma (without full Peters anomaly). "
            "KEY CLINICAL TEST: B-scan ultrasound — primary aphakia = no lens echo; "
            "Peters anomaly = lens present but anteriorly displaced with posterior corneal adhesion."
        ),
        "treatment": (
            "PETERS ANOMALY (central corneal opacity ± lens touch): "
            "URGENT SURGERY if dense bilateral: penetrating keratoplasty (PK/PKP) within first few weeks of life; "
            "optical iridectomy if peripheral cornea clear; "
            "glaucoma surgery concurrent or staged (trabecular meshwork hypoplasia in many); "
            "SCLEROCORNEA: PK only if peripheral cornea available; prognosis poor without limbal tissue; "
            "artificial cornea (Boston keratoprosthesis) in severe bilateral cases; "
            "PRIMARY APHAKIA: contact lens correction (aphakic silicone) immediately post-PK; "
            "IOL not feasible in primary aphakia; patching for amblyopia; "
            "CONGENITAL CATARACT (AD milder): standard paediatric cataract protocol; "
            "genetic testing: FOXE3 panel + microarray; ophthalmology surveillance 6-monthly. "
            "GLAUCOMA: high risk post-PK (40-60%) — IOP monitoring + goniotomy/medical."
        ),
        "key_features": [
            "Peters anomaly: central corneal opacity + posterior stromal/Descemet defect",
            "Lens touch (anterior lens-to-cornea adhesion) — UBM diagnostic",
            "AR biallelic: primary aphakia + sclerocornea + microphthalmia + coloboma",
            "AD hypomorphic: congenital cataract + microcornea + iris coloboma",
            "FOXE3: lens vesicle separation defect — embryological mechanism",
            "Glaucoma post-PK 40-60% — trabecular meshwork hypoplasia",
            "Absent lens echo on B-scan = primary aphakia (AR severe)",
            "5-10% of Peters anomaly cases",
        ],
        "key_ddx": (
            "B3GLCT (Peters Plus): Peters anomaly + SHORT STATURE + ID + CLEFT LIP — systemic features absent in FOXE3; "
            "CYP1B1 (Peters type 2): Peters anomaly + high IOP at birth — CYP1B1 metabolises steroids; "
            "Sclerocornea (non-genetic): sporadic, unilateral common; "
            "Peters anomaly from rubella: maternal infection, cataract, deafness triad; "
            "PITX2/FOXC1: iridocorneal STRANDS without central corneal opacity; "
            "PAX6 AR (sclerocornea): rare compound heterozygous — full eye exam + genetics."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital",
        "surgical_urgency": "Urgent — PK within weeks if bilateral dense opacity; concurrent glaucoma surgery",
        "gene_family": "Forkhead Box transcription factor (FOXE subclass)",
        "morphology": "Central corneal opacity (Peters anomaly) + lens touch OR primary aphakia + sclerocornea",
    },

    # -- B3GLCT -- Peters Plus Syndrome -------------------------------------------------------
    {
        "gene": "B3GLCT",
        "alt_name": (
            "B3GLCT (B3GLCT-498aa-13q12.3 / AR -- "
            "PETERS-PLUS-SYNDROME-CORNEAL-CLOUDING+SHORT-STATURE+ID+CLEFT-LIP-DIAGNOSTIC-TETRAD -- "
            "PETERS-ANOMALY-UNIVERSAL-OCULAR-HALLMARK-100pct -- "
            "GLYCOSYLATION-DEFECT-THROMBOSPONDIN-REPEATS -- "
            "IVS8DS-GA-SPLICE-SITE-COMMON-EUROPEAN-FOUNDER)"
        ),
        "protein": (
            "B3GLCT -- 13q12.3 AR -- B3GLCT-498aa -- "
            "Beta-3-Glucosyltransferase-B3GALTL-57kDa-Golgi-Glycosyltransferase -- "
            "Glucosylates-O-Fucosylated-Thrombospondin-Type-1-Repeats-TSR -- "
            "Substrates-Thrombospondin-1-ADAMTS-Properdin-CSF1-COMP -- "
            "Essential-For-TSR-Glucosylation-Extracellular-Matrix-Signalling -- "
            "OMIM-Gene-610308-Disease-Peters-Plus-261540"
        ),
        "locus": "13q12.3",
        "protein_size": "498 aa / 57 kDa",
        "inheritance": (
            "AR — biallelic LOF only; "
            "most common variant: c.660+1G>A (IVS8+1G>A splice site) — European founder mutation present in ~60% of European alleles; "
            "compound heterozygous: c.660+1G>A / missense in non-European populations; "
            "carrier frequency ~1:100 Northern Europe; "
            "Peters Plus is the only known human disease caused by B3GLCT mutation; "
            "phenotype fairly consistent — corneal findings universal; "
            "no homozygous patients described outside consanguineous families; "
            "prenatal diagnosis available by molecular testing"
        ),
        "disease_category": "Peters Plus Syndrome — Peters anomaly + short stature + intellectual disability + cleft lip/palate (AR glycosylation disorder)",
        "disease_pathway": (
            "B3GLCT (Beta-3-Glucosyltransferase) catalyses the addition of glucose to O-fucosylated thrombospondin type-1 repeats (TSR). "
            "TSR-containing proteins include: thrombospondin-1 (ECM), ADAMTS proteases, properdin (complement), CSF1, COMP. "
            "PATHOMECHANISM: B3GLCT LOF → absent TSR glucosylation → "
            "misfolded TSR-containing proteins that cannot be properly secreted or fold correctly → "
            "disrupted extracellular matrix signalling in anterior segment, limbs, brain, and palate. "
            "OCULAR (UNIVERSAL — 100%): Peters anomaly — central corneal opacity with posterior stromal/Descemet defect; "
            "mechanism: impaired TSR-containing ECM proteins disrupt normal corneal endothelial-to-mesenchymal signalling "
            "→ failure of neural crest cells to properly form posterior corneal layers. "
            "SYSTEMIC CONSEQUENCES OF TSR GLUCOSYLATION DEFECT: "
            "SHORT STATURE (80-90%): COMP (cartilage oligomeric matrix protein) contains TSR — "
            "COMP misfolding → growth plate cartilage defect → disproportionate short stature. "
            "INTELLECTUAL DISABILITY (60-80%): brain ECM TSR proteins disrupted — "
            "neuronal migration/connectivity impaired; structural brain anomalies in some. "
            "CLEFT LIP ± PALATE (60-70%): palatal shelf elevation requires TSR-mediated ECM signalling. "
            "LIMB: brachydactyly (short, broad distal phalanges) in 50-60%; broad thumbs."
        ),
        "pathognomonic": (
            "DIAGNOSTIC TETRAD: CORNEAL CLOUDING (Peters anomaly — central opacity) + "
            "SHORT STATURE (disproportionate, below 3rd centile) + "
            "INTELLECTUAL DISABILITY (mild-moderate, 60-80%) + "
            "CLEFT LIP ± PALATE (60-70%) = PETERS PLUS SYNDROME PATHOGNOMONIC. "
            "PETERS ANOMALY: universal (100%) — central corneal opacity present from birth; "
            "anterior segment OCT: absent posterior stroma + Descemet membrane in central zone; "
            "lens touch less common than in FOXE3 Peters. "
            "BRACHYDACTYLY: short broad distal phalanges + broad thumbs — hand X-ray shows. "
            "IVS8+1G>A SPLICE SITE: founder mutation — targeted screening in European patients. "
            "B3GLCT Peters Plus is NOT a classic CDG (transferrin isoelectric focusing normal) — "
            "it is a disorder of O-fucose modification specifically (TSR-specific), not N-glycosylation. "
            "GLAUCOMA: 30-50% — trabecular meshwork developmental defect concurrent with Peters."
        ),
        "treatment": (
            "CORNEAL SURGERY: PK (penetrating keratoplasty) for dense Peters anomaly; "
            "prognosis guarded — concurrent trabecular meshwork dysgenesis → frequent glaucoma post-PK; "
            "optical iridectomy if peripheral cornea clear. "
            "GLAUCOMA: 30-50% — goniotomy + medical IOP lowering; GDD for refractory. "
            "INTELLECTUAL DISABILITY: early intervention — speech, occupational, physiotherapy; "
            "special education; cognitive assessment at school entry. "
            "SHORT STATURE: auxology 6-monthly; growth hormone assessment if growth velocity low; "
            "COMP-related short stature responds poorly to growth hormone vs GHD. "
            "CLEFT LIP/PALATE: craniofacial team — surgical repair at 3-6 months (lip), 9-12 months (palate). "
            "BRACHYDACTYLY: hand function assessment; orthopaedic referral if functional impairment. "
            "GENETIC: molecular confirmation IVS8+1G>A targeted first in European patients; "
            "full B3GLCT sequencing + MLPA for non-European. "
            "PRENATAL: ultrasound — corneal clouding sometimes visible on fetal echo."
        ),
        "key_features": [
            "Peters anomaly (universal — 100%) — central corneal opacity at birth",
            "Short stature (disproportionate, 80-90%)",
            "Intellectual disability (mild-moderate, 60-80%)",
            "Cleft lip ± palate (60-70%)",
            "Brachydactyly — broad distal phalanges + broad thumbs (50-60%)",
            "IVS8+1G>A founder mutation — 60% European alleles",
            "TSR glucosylation defect — distinct from classic CDG (normal transferrin IEF)",
            "Glaucoma 30-50% — trabecular meshwork hypoplasia",
        ],
        "key_ddx": (
            "FOXE3 (Peters anomaly): NO systemic features — isolated ocular Peters; "
            "CYP1B1 (Peters type 2): high IOP at birth, no cleft, no short stature; "
            "STRA6 (Matthew-Wood): Peters + CHD + pulmonary agenesis — STRA6 is RBP4 receptor; "
            "COL4A1 (Peters-like): Peters + porencephaly + muscle cramps — different systemic; "
            "Classic CDG (PMM2, MPI): transferrin IEF abnormal — normal in Peters Plus; "
            "Rubella embryopathy: maternal infection, cataract + deafness + cardiac triad; "
            "Warburg micro syndrome (RAB18, TRAPPC): Peters + polymicrogyria + hypotonia — severe."
        ),
        "systemic_involvement": True,
        "onset_age": "Congenital",
        "surgical_urgency": "Urgent — PK for bilateral Peters; multidisciplinary neonatal assessment",
        "gene_family": "Beta-glycosyltransferase (O-fucose TSR-modification)",
        "morphology": "Peters anomaly (central corneal opacity) — universal; associated systemic tetrad",
    },

    # -- PXDN -- Anterior Segment Dysgenesis type 7 / Sclerocornea ---------------------------
    {
        "gene": "PXDN",
        "alt_name": (
            "PXDN (PXDN-1479aa-2p25.3 / AR -- "
            "ASD7-ANTERIOR-SEGMENT-DYSGENESIS-TYPE-7 -- "
            "BILATERAL-CONGENITAL-CORNEAL-OPACIFICATION+SCLEROCORNEA-AT-BIRTH-PATHOGNOMONIC -- "
            "PEROXIDASIN-COLLAGEN-IV-SULFILIMINE-CROSSLINKS-ANTERIOR-BM -- "
            "ABSENT-CROSSLINKS-ABNORMAL-CORNEAL-DEVELOPMENT)"
        ),
        "protein": (
            "PXDN -- 2p25.3 AR -- PXDN-1479aa -- "
            "Peroxidasin-Vascular-Peroxidase-1-VPO1-165kDa-Heme-Peroxidase-Extracellular -- "
            "N-Terminal-Leucine-Rich-Repeats-Immunoglobulin-Domains-Peroxidase-Domain-C-Terminal -- "
            "Uses-H2O2-Hypohalous-Acid-To-Crosslink-Collagen-IV-NC1-Domains-Sulfilimine-Bond -- "
            "Expressed-Corneal-Epithelium-Basement-Membrane-Lens-Vitreous-Secreted-Extracellular -- "
            "OMIM-Gene-605158-Disease-ASD7-269400"
        ),
        "locus": "2p25.3",
        "protein_size": "1479 aa / 165 kDa",
        "inheritance": (
            "AR — biallelic LOF only; "
            "gene discovered as ASD cause in 2012 (Khan et al.); "
            "multiple consanguineous families (Turkish, Pakistani, North African); "
            "mutations: frameshift, nonsense, splice, missense in peroxidase domain; "
            "no clear founder mutation — diverse alleles; "
            "phenotype: bilateral anterior segment dysgenesis (corneal opacification from birth) is the universal finding; "
            "some patients have additional microphthalmia, coloboma, or glaucoma; "
            "heterozygous carriers: clinically unaffected (haploinsufficiency insufficient)"
        ),
        "disease_category": "Anterior Segment Dysgenesis type 7 (ASD7) — bilateral corneal opacification / sclerocornea (PXDN-associated peroxidasin deficiency)",
        "disease_pathway": (
            "PXDN (Peroxidasin) is the only known enzyme that forms SULFILIMINE BONDS (S=N crosslinks) in collagen IV NC1 domains. "
            "Sulfilimine bonds are unique to metazoan basement membranes — they provide extraordinary mechanical strength "
            "to basement membranes (resistance to proteolytic degradation, structural rigidity). "
            "MECHANISM: PXDN secreted into extracellular space → uses H2O2 and hypohalous acid (HOBr from bromide) → "
            "oxidizes methionine sulfur + lysine nitrogen → forms sulfilimine bond crosslinking adjacent collagen IV NC1 domains. "
            "IN THE ANTERIOR SEGMENT: corneal basement membranes (epithelial BM, Bowman layer progenitor, Descemet membrane) "
            "contain collagen IV that REQUIRES sulfilimine crosslinks for proper mechanical integrity and developmental signalling. "
            "PATHOMECHANISM: PXDN LOF → absent sulfilimine crosslinks in anterior segment basement membranes → "
            "structurally weak, dysplastic collagen IV network → "
            "SCLEROCORNEA (anterior segment cornea becomes sclera-like — white, opaque) or "
            "CORNEAL OPACIFICATION (anterior stroma fails to develop normally) from birth. "
            "GLAUCOMA: trabecular meshwork collagen IV also affected — angle developmental defect possible. "
            "UNIQUE PATHOMECHANISM: not a transcription factor defect but an ENZYMATIC EXTRACELLULAR CROSSLINKING defect."
        ),
        "pathognomonic": (
            "BILATERAL CORNEAL OPACIFICATION AT BIRTH (variable severity: from central haze to total sclerocornea). "
            "SCLEROCORNEA: entire cornea appears white/sclera-like with no visible limbal boundary — "
            "most severe presentation in PXDN AR; "
            "anterior segment OCT: thick, hyperreflective corneal stroma; absent normal corneal architecture. "
            "CONSANGUINEOUS FAMILY HISTORY (most reported patients): Turkish, Pakistani, North African. "
            "PXDN unique pathomechanism: collagen IV sulfilimine crosslink deficiency — "
            "key distinguishing biochemical mechanism from other ASD genes. "
            "GLAUCOMA: elevated IOP from birth in subset — concurrent IOP management mandatory. "
            "MICROPHTHALMIA ± COLOBOMA: additional features in some severe cases. "
            "IMAGING: ultrasound biomicroscopy confirms ASD; B-scan for posterior segment assessment when cornea opaque."
        ),
        "treatment": (
            "SCLEROCORNEA / CORNEAL OPACIFICATION: "
            "Penetrating keratoplasty (PK) — only option for visual rehabilitation; "
            "timing: bilateral cases within weeks of birth if severe (deprivation amblyopia); "
            "prognosis guarded in total sclerocornea (no limbal tissue for PK anchor); "
            "Boston keratoprosthesis (KPro) type 2 in total bilateral sclerocornea; "
            "GLAUCOMA: concurrent assessment essential — combined PK + glaucoma surgery when IOP elevated; "
            "AMBLYOPIA: optical correction + patching post-PK — critical for visual outcome; "
            "LOW VISION: in patients not amenable to keratoplasty — low vision aids, magnification; "
            "GENETIC: PXDN sequencing confirmation; consanguinity counselling; "
            "ANIMAL MODELS: PXDN-null Drosophila shows basement membrane defects — "
            "potential therapeutic: no current targeted therapy."
        ),
        "key_features": [
            "Bilateral corneal opacification at birth — range central haze to total sclerocornea",
            "Unique mechanism: collagen IV sulfilimine crosslink deficiency (PXDN enzyme absent)",
            "Consanguineous families — Turkish, Pakistani, North African",
            "Sclerocornea: no visible limbal landmarks, sclera-like cornea",
            "Anterior segment OCT: thick hyperreflective dysplastic stroma",
            "Glaucoma in subset — trabecular meshwork collagen IV affected",
            "Microphthalmia ± coloboma in severe cases",
            "PK only treatment; Boston KPro for total sclerocornea",
        ],
        "key_ddx": (
            "B3GLCT (Peters Plus): Peters anomaly + SHORT STATURE + ID + CLEFT — systemic features absent in PXDN; "
            "FOXE3 (ASD2): Peters anomaly + primary aphakia — lens absent; PXDN → lens usually present; "
            "Congenital hereditary endothelial dystrophy (SLC4A11-CHED2): bilateral diffuse ground-glass haze, "
            "thickened cornea — different mechanism (endothelial pump), normal anterior stroma architecture; "
            "COL8A2 (FECD1): early Fuchs — adult onset, guttae, NOT congenital opacification; "
            "LCAT deficiency: arcus lipoides + corneal opacity + HDL deficiency — systemic lipid involvement; "
            "Sclerocornea non-genetic: sporadic, unilateral common; no family history."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital (corneal opacification at birth)",
        "surgical_urgency": "Urgent — PK within weeks for bilateral dense opacity; Boston KPro for total sclerocornea",
        "gene_family": "Heme peroxidase (extracellular peroxidasin — collagen IV crosslinking enzyme)",
        "morphology": "Bilateral corneal opacification to total sclerocornea — congenital",
    },

    # -- HCCS -- MIDAS / MLS Syndrome ---------------------------------------------------------
    {
        "gene": "HCCS",
        "alt_name": (
            "HCCS (HCCS-295aa-Xp22.2 / XLD -- "
            "MIDAS-MLS-SYNDROME-MICRO-OPHTHALMIA-LINEAR-SKIN-DEFECTS -- "
            "PERIOCULAR-FACIAL-LINEAR-SKIN-DEFECTS+MICROPHTHALMIA-IN-FEMALES-PATHOGNOMONIC -- "
            "X-LINKED-DOMINANT-LETHAL-IN-HEMIZYGOUS-MALES -- "
            "CARDIAC-DEFECTS-20pct-CNS-ANOMALIES-30pct)"
        ),
        "protein": (
            "HCCS -- Xp22.2 XLD -- HCCS-295aa -- "
            "Holocytochrome-C-Type-Synthase-33kDa-Mitochondrial-Heme-Lyase -- "
            "Attaches-Heme-To-Apocytochrome-C-And-C1-In-Mitochondrial-Intermembrane-Space -- "
            "Essential-For-Cytochrome-C-Electron-Transport-Chain-Complex-III -- "
            "LOF-Impairs-OXPHOS-Especially-In-Neural-Cells -- "
            "OMIM-Gene-300056-Disease-MIDAS-309801"
        ),
        "locus": "Xp22.2",
        "protein_size": "295 aa / 33 kDa",
        "inheritance": (
            "X-linked dominant (XLD) — LETHAL IN HEMIZYGOUS MALES (only affected females reported in MIDAS); "
            "heterozygous females: mosaic due to X-inactivation → variable phenotype (skewed X-inactivation modifies severity); "
            "de novo XLD mutations account for most cases; "
            "affected females survive because mosaic X-inactivation allows some cells to use normal HCCS allele; "
            "males with HCCS deletion: in utero lethality (rarely survive as 47,XXY or with somatic mosaicism); "
            "diagnosis: females with MLS phenotype + Xp22.2 deletion/point mutation; "
            "prenatal: ultrasonographic microphthalmia/anophthalmia in female fetus raises suspicion"
        ),
        "disease_category": "MIDAS Syndrome (MLS — Microphthalmia with Linear Skin defects) — XLD mitochondrial cytochrome c assembly defect",
        "disease_pathway": (
            "HCCS (Holocytochrome C-Type Synthase) catalyses attachment of heme to apocytochrome c and c1 "
            "in the mitochondrial intermembrane space. "
            "Holocytochrome c is an essential component of Complex III of the mitochondrial electron transport chain. "
            "PATHOMECHANISM: HCCS LOF → absent holocytochrome c in mitochondria → "
            "Complex III (cytochrome bc1 complex) dysfunction → impaired OXPHOS. "
            "X-INACTIVATION MOSAICISM explains the phenotype: "
            "In HCCS heterozygous females, cells with the mutant X active undergo apoptosis "
            "(particularly in neural/ectodermal lineages that depend on OXPHOS) → "
            "LINEAR SKIN DEFECTS along Blaschko lines (boundaries of X-inactivation clone territories in skin) — "
            "HALLMARK: periocular and facial linear skin streaks/erosions following Blaschko lines. "
            "OCULAR: HCCS is required in optic vesicle/cup progenitors → HCCS-null cells apoptose → "
            "MICROPHTHALMIA (small eye) or ANOPHTHALMIA (absent eye); bilateral or unilateral; "
            "anterior segment dysgenesis in some (sclerocornea, iris anomalies). "
            "CARDIAC (20%): HCCS required in cardiomyocytes — septal defects, cardiomyopathy. "
            "CNS (30%): agenesis of corpus callosum, lissencephaly, arachnoid cyst — "
            "OXPHOS requirement during neuronal migration and synaptogenesis."
        ),
        "pathognomonic": (
            "PERIOCULAR / FACIAL LINEAR SKIN DEFECTS (erosions/streaks following Blaschko lines around eyes and face) "
            "IN A FEMALE PATIENT = MIDAS/MLS PATHOGNOMONIC. "
            "MICROPHTHALMIA / ANOPHTHALMIA (bilateral in 70%, unilateral in 30%) — "
            "axial length severely reduced or eye clinically absent. "
            "X-LINKED DOMINANT LETHAL MALES: affected family members are EXCLUSIVELY FEMALE; "
            "history of miscarriages (male lethal). "
            "CARDIAC DEFECTS (20%): ASD, VSD, cardiomyopathy — echocardiography mandatory. "
            "CNS ANOMALIES (30%): corpus callosum agenesis on brain MRI — MRI mandatory. "
            "SCLEROCORNEA / ANTERIOR SEGMENT DYSGENESIS: in subset with residual microphthalmic eye. "
            "LINEAR SKIN DEFECTS: NOT random scarring — strictly follow Blaschko lines (clone boundaries of X-inactivation). "
            "Skin biopsy: thin epidermis with inflammatory infiltrate at active lesions; heal to hyperpigmented lines."
        ),
        "treatment": (
            "LINEAR SKIN DEFECTS: emollients + wound care during active phase; "
            "hyperpigmented Blaschko lines persist — camouflage cosmetics; "
            "active erosions: silver sulfadiazine or sterile dressings. "
            "MICROPHTHALMIA/ANOPHTHALMIA: conformers (graded prosthetic shells) to stimulate socket growth; "
            "ocular prosthesis fitting at 6 weeks; referral to ocularist; "
            "orbital expansion with graded conformers — critical to prevent socket contraction; "
            "low vision assessment for any residual vision. "
            "SCLEROCORNEA/ASD: PK if cornea amenable; "
            "CARDIAC: paediatric cardiology — echocardiogram at birth; surgical repair if indicated. "
            "CNS: brain MRI in all patients; neurodevelopmental assessment; "
            "epilepsy management if seizures (AEDs appropriate for structural epilepsy). "
            "GENETIC: Xp22.2 microarray + HCCS sequencing; "
            "family counselling: XLD inheritance pattern; male embryo lethality counselling."
        ),
        "key_features": [
            "Linear skin defects following Blaschko lines — periocular/facial (pathognomonic)",
            "Microphthalmia/anophthalmia — bilateral 70%, unilateral 30%",
            "XLD lethal in males — affected patients exclusively female",
            "Cardiac defects 20% — echo mandatory",
            "CNS anomalies 30% — corpus callosum agenesis, lissencephaly",
            "HCCS = holocytochrome c synthase — mitochondrial Complex III component",
            "X-inactivation mosaicism drives Blaschko-line phenotype",
            "Sclerocornea/ASD in subset with residual microphthalmia",
        ],
        "key_ddx": (
            "CHARGE syndrome (CHD7): COLOBOMA + heart + choanal atresia + ears — no Blaschko skin; CHD7 AD; "
            "Focal dermal hypoplasia (PORCN XLD): Blaschko skin + limb defects — no microphthalmia as primary; "
            "Incontinentia pigmenti (IKBKG XLD): Blaschko skin in females, lethal males — vesicular → verrucous → "
            "hyperpigmented Blaschko stages (vs erosive in MIDAS); neurologic, dental, retinal anomalies; "
            "PAX6: microphthalmia rare; iris absence predominates; no Blaschko skin; "
            "SOX2 (anophthalmia-oesophageal-genital): bilateral severe anophthalmia + oesophageal atresia; "
            "OTX2: bilateral anophthalmia/microphthalmia — pituitary, craniofacial anomalies."
        ),
        "systemic_involvement": True,
        "onset_age": "Congenital (skin lesions + microphthalmia present at birth)",
        "surgical_urgency": "Urgent — conformer fitting within days; cardiac + CNS assessment at birth",
        "gene_family": "Mitochondrial heme lyase (holocytochrome c-type synthase) — OXPHOS Complex III",
        "morphology": "Microphthalmia/anophthalmia + linear Blaschko skin defects — females only",
    },

    # -- CYP1B1 -- Peters Anomaly type 2 / Primary Congenital Glaucoma -----------------------
    {
        "gene": "CYP1B1",
        "alt_name": (
            "CYP1B1 (CYP1B1-543aa-2p22.2 / AR -- "
            "PETERS-ANOMALY-TYPE-2-PRIMARY-CONGENITAL-GLAUCOMA-PCG -- "
            "CENTRAL-CORNEAL-OPACITY+HIGH-IOP+BUPHTHALMOS-AT-BIRTH-PATHOGNOMONIC -- "
            "G61E-R368H-R390H-MOST-COMMON-PATHOGENIC-VARIANTS -- "
            "STEROID-METABOLISM-TRABECULAR-MESHWORK-DEVELOPMENTAL-FAILURE)"
        ),
        "protein": (
            "CYP1B1 -- 2p22.2 AR -- CYP1B1-543aa -- "
            "Cytochrome-P450-Family-1-Subfamily-B-Member-1-60kDa-Extrahepatic-CYP -- "
            "Expressed-Eye-Trabecular-Meshwork-Iris-Ciliary-Body-Lens -- "
            "Metabolises-Steroids-Retinoic-Acid-17beta-Estradiol-Testosterone-Arachidonic-Acid -- "
            "Absent-CYP1B1-Accumulation-Unmetabolised-Steroids-Toxic-To-NCC -- "
            "OMIM-Gene-601771-Disease-PCG-231300-Peters2-604229"
        ),
        "locus": "2p22.2",
        "protein_size": "543 aa / 60 kDa",
        "inheritance": (
            "AR — biallelic LOF (most cases) or compound heterozygous; "
            "CYP1B1 is the most common gene for primary congenital glaucoma (PCG) worldwide — "
            "accounts for 20-40% of PCG in Western populations, up to 95% in some consanguineous populations (Saudi, Turkish, Roma); "
            "common pathogenic variants: G61E (European), R368H (Middle Eastern/Indian), R390H (European/Pakistani); "
            "PCG phenotype (no Peters): biallelic severe LOF; "
            "Peters anomaly type 2: different variant spectrum — some biallelic, some heterozygous; "
            "Digenic: CYP1B1 + TEK, CYP1B1 + LTBP2 double heterozygotes show synergistic effect; "
            "heterozygous carriers: mild IOP elevation risk (incomplete penetrance)"
        ),
        "disease_category": "Primary Congenital Glaucoma (PCG / GLC3A) + Peters Anomaly type 2 (central corneal opacity + high IOP)",
        "disease_pathway": (
            "CYP1B1 belongs to the cytochrome P450 superfamily and is the primary extrahepatic CYP expressed in the eye. "
            "CYP1B1 metabolises: 17β-estradiol → 4-hydroxyestradiol; testosterone → 16α-hydroxytestosterone; "
            "retinoic acid to less-active forms; arachidonic acid → epoxyeicosatrienoic acids (EETs). "
            "PATHOMECHANISM IN GLAUCOMA: CYP1B1 LOF → accumulation of unmetabolised steroid substrates (especially estradiol metabolites) "
            "and retinoic acid → TOXIC TO NEURAL CREST CELLS (NCC) that form trabecular meshwork → "
            "NCC apoptosis or arrested differentiation → TRABECULAR MESHWORK AGENESIS or DYSGENESIS → "
            "absent/dysfunctional aqueous outflow pathway → "
            "IOP elevation from birth → BUPHTHALMOS (enlargement of globe under elevated IOP in infant) → "
            "HAAB STRIAE (horizontal corneal breaks from Descemet membrane stretch at high IOP). "
            "PATHOMECHANISM IN PETERS ANOMALY TYPE 2: CYP1B1-dependent metabolite abnormalities also affect "
            "corneal endothelial NCC → failure of posterior corneal stroma/Descemet formation → "
            "CENTRAL CORNEAL OPACITY (Peters anomaly morphology: absent posterior stroma, absent Descemet centrally). "
            "RETINOIC ACID ACCUMULATION: excess RA in developing eye → NCC migration arrest → "
            "trabecular meshwork + corneal mesenchyme defects. "
            "SYNERGISM: CYP1B1 + LTBP2 double heterozygotes → worse PCG (LTBP2 regulates TGF-beta → ECM in TM)."
        ),
        "pathognomonic": (
            "PRIMARY CONGENITAL GLAUCOMA (PCG): BUPHTHALMOS (large globe — corneal diameter >12mm in newborn) + "
            "HAAB STRIAE (horizontal corneal breaks, white parallel lines on posterior cornea from stretch) + "
            "EXCESSIVE TEARING + PHOTOPHOBIA at birth or first months = PCG TRIAD PATHOGNOMONIC. "
            "PETERS ANOMALY TYPE 2: CENTRAL CORNEAL OPACITY (present at birth, variable density) + "
            "HIGH IOP (>21 mmHg) — the combination of Peters + elevated IOP distinguishes CYP1B1 Peters from other Peters genes. "
            "G61E VARIANT: most common European CYP1B1 variant — targeted sequencing first in European PCG. "
            "R368H VARIANT: most common Middle Eastern/Indian — Arg368His. "
            "CONSANGUINEOUS FAMILIES: CYP1B1 accounts for up to 95% of PCG in consanguineous Turkish/Saudi populations. "
            "DIGENIC: CYP1B1 + LTBP2 or TEK compound heterozygotes — family history may appear irregular. "
            "PCG is a surgical emergency: goniotomy or trabeculotomy within days of diagnosis."
        ),
        "treatment": (
            "PRIMARY CONGENITAL GLAUCOMA — SURGICAL EMERGENCY: "
            "GONIOTOMY: first-line angle surgery — goniotomy lens + surgical opening of trabecular meshwork; "
            "SUCCESS RATE: 80-90% first surgery if cornea clear; "
            "TRABECULOTOMY: alternative if cornea opaque (PCG-specific probes, TRAB360 for circumferential); "
            "MEDICAL: topical beta-blockers (timolol 0.1% paediatric) + CAI (dorzolamide/brinzolamide) + "
            "alpha-2 agonists (brimonidine AVOID <2 years — apnea risk); "
            "Medical = BRIDGE to surgery, NOT definitive; "
            "GLAUCOMA DRAINAGE DEVICE: Ahmed/Baerveldt if goniotomy fails; "
            "PETERS ANOMALY TYPE 2: PK for corneal opacity + concurrent glaucoma surgery; "
            "AMBLYOPIA: optical correction + patching post-surgery — critical; "
            "LIFELONG FOLLOW-UP: IOP + optic disc + visual field — glaucoma relapse common in teens; "
            "CYP1B1 molecular confirmation + family screening; "
            "consanguineous population screening programs for CYP1B1 carrier status."
        ),
        "key_features": [
            "Buphthalmos (large globe >12mm cornea in newborn)",
            "Haab striae (horizontal corneal breaks — Descemet stretch)",
            "Excessive tearing + photophobia at birth",
            "Most common PCG gene worldwide — up to 95% in consanguineous populations",
            "G61E (European), R368H (Middle Eastern/Indian), R390H (European) — common variants",
            "Peters anomaly type 2: central corneal opacity + HIGH IOP (distinguishes from FOXE3/B3GLCT Peters)",
            "Steroid/RA metabolism defect → NCC toxic accumulation",
            "Goniotomy/trabeculotomy: surgical emergency — 80-90% success",
        ],
        "key_ddx": (
            "FOXE3 (Peters type 1/ASD2): Peters anomaly WITHOUT elevated IOP usually — FOXE3 primary; "
            "B3GLCT (Peters Plus): Peters + SHORT STATURE + ID + CLEFT — CYP1B1 Peters has no systemic; "
            "PITX2/FOXC1 (ARS): elevated IOP YES but iris strands + posterior embryotoxon, NOT corneal opacity; "
            "TEK (GLC3E): PCG without Peters — TEK mutations less common; "
            "LTBP2 (GLC3F): PCG + microspherophakia — lens smaller and spherical; "
            "Infantile glaucoma from uveitis: inflammatory cells + KPs; "
            "Traumatic glaucoma: birth trauma — Haab striae but IOP may be normal; "
            "Axenfeld-Rieger (not true PCG): iris strands, not trabecular mesh agenesis."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital/neonatal (symptoms from birth or first weeks)",
        "surgical_urgency": "URGENT — goniotomy/trabeculotomy within days of diagnosis; surgical emergency",
        "gene_family": "Cytochrome P450 (CYP1B subfamily) — extrahepatic steroid/RA metaboliser",
        "morphology": "Buphthalmos + Haab striae (PCG) OR central corneal opacity + high IOP (Peters type 2)",
    },
]


def _make_cohort(entry, seed):
    rng = random.Random(seed)
    gene = entry["gene"]
    patients = []
    for i in range(40):
        age_dx = 0  # all congenital or neonatal

        # Glaucoma presence
        if gene == "PITX2":
            glaucoma = rng.random() < 0.50
        elif gene == "FOXC1":
            glaucoma = rng.random() < 0.50
        elif gene == "PAX6":
            glaucoma = rng.random() < 0.40
        elif gene == "FOXE3":
            glaucoma = rng.random() < 0.45
        elif gene == "B3GLCT":
            glaucoma = rng.random() < 0.40
        elif gene == "PXDN":
            glaucoma = rng.random() < 0.35
        elif gene == "HCCS":
            glaucoma = rng.random() < 0.20
        else:  # CYP1B1
            glaucoma = rng.random() < 0.92  # PCG — near-universal

        # Corneal opacity
        if gene in ("FOXE3", "B3GLCT", "PXDN", "CYP1B1"):
            corneal_opacity = rng.random() < 0.90
        elif gene == "HCCS":
            corneal_opacity = rng.random() < 0.30  # only in subset with residual eye
        elif gene in ("PITX2", "FOXC1"):
            corneal_opacity = rng.random() < 0.10  # secondary/rare
        else:  # PAX6
            corneal_opacity = rng.random() < 0.55  # progressive keratopathy

        # Surgery performed
        if gene in ("FOXE3", "B3GLCT", "PXDN"):
            surgery = corneal_opacity and rng.random() < 0.85
        elif gene == "CYP1B1":
            surgery = rng.random() < 0.95  # surgical emergency
        elif gene in ("PITX2", "FOXC1"):
            surgery = glaucoma and rng.random() < 0.70
        elif gene == "PAX6":
            surgery = (corneal_opacity and rng.random() < 0.40) or (rng.random() < 0.15)
        else:  # HCCS
            surgery = rng.random() < 0.50

        # Systemic involvement
        if gene == "PITX2":
            systemic = rng.random() < 0.75  # dental + umbilical
        elif gene == "FOXC1":
            systemic = rng.random() < 0.55  # cardiac + hearing
        elif gene == "PAX6":
            systemic = rng.random() < 0.20  # WAGR subset
        elif gene == "B3GLCT":
            systemic = True  # universal tetrad
        elif gene == "HCCS":
            systemic = rng.random() < 0.80  # cardiac + CNS
        else:
            systemic = False

        # Visual acuity poor (worse than 6/18)
        if gene == "HCCS":
            va_poor = rng.random() < 0.90  # microphthalmia/anophthalmia
        elif gene in ("PXDN",):
            va_poor = rng.random() < 0.75
        elif gene in ("FOXE3", "B3GLCT") and corneal_opacity:
            va_poor = rng.random() < 0.65
        elif gene == "CYP1B1" and not surgery:
            va_poor = rng.random() < 0.80
        elif gene == "PAX6":
            va_poor = rng.random() < 0.50  # foveal hypoplasia + keratopathy
        elif gene in ("PITX2", "FOXC1") and glaucoma:
            va_poor = rng.random() < 0.25
        else:
            va_poor = rng.random() < 0.12

        # Consanguinity
        if gene == "PXDN":
            consanguineous = rng.random() < 0.75  # Turkish/Pakistani/North African
        elif gene == "CYP1B1":
            consanguineous = rng.random() < 0.50
        elif gene in ("FOXE3", "B3GLCT", "HCCS"):
            consanguineous = rng.random() < 0.30
        else:
            consanguineous = rng.random() < 0.08

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis_years": age_dx,
            "glaucoma": glaucoma,
            "corneal_opacity": corneal_opacity,
            "surgery_performed": surgery,
            "va_poor": va_poor,
            "systemic_involvement": systemic,
            "consanguineous": consanguineous,
            "inheritance": entry["inheritance"].split(";")[0].strip(),
        })
    return patients


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(ASD_GENES):
        all_patients.extend(_make_cohort(entry, SEED_BASE + idx))

    total = len(all_patients)
    glaucoma_count = sum(1 for p in all_patients if p["glaucoma"])
    corneal_count = sum(1 for p in all_patients if p["corneal_opacity"])
    surgery_count = sum(1 for p in all_patients if p["surgery_performed"])
    systemic_count = sum(1 for p in all_patients if p["systemic_involvement"])
    va_poor_count = sum(1 for p in all_patients if p["va_poor"])
    consanguineous_count = sum(1 for p in all_patients if p["consanguineous"])

    gene_summary = {}
    for idx, entry in enumerate(ASD_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "morphology": entry["morphology"],
            "systemic_involvement": entry["systemic_involvement"],
            "onset_age": entry["onset_age"],
            "surgical_urgency": entry["surgical_urgency"],
            "gene_family": entry["gene_family"],
            "n_patients": len(cohort),
            "glaucoma_pct": round(100 * sum(1 for p in cohort if p["glaucoma"]) / len(cohort), 1),
            "corneal_opacity_pct": round(100 * sum(1 for p in cohort if p["corneal_opacity"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["surgery_performed"]) / len(cohort), 1),
            "va_poor_pct": round(100 * sum(1 for p in cohort if p["va_poor"]) / len(cohort), 1),
            "consanguineous_pct": round(100 * sum(1 for p in cohort if p["consanguineous"]) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Anterior-Segment-Dysgenesis-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Anterior Segment Dysgenesis Reference -- PITX2/FOXC1/PAX6/FOXE3/B3GLCT/PXDN/HCCS/CYP1B1",
        "genes_covered": [e["gene"] for e in ASD_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "glaucoma_pct": round(100 * glaucoma_count / total, 1),
            "corneal_opacity_pct": round(100 * corneal_count / total, 1),
            "surgery_performed_pct": round(100 * surgery_count / total, 1),
            "systemic_involvement_pct": round(100 * systemic_count / total, 1),
            "va_worse_than_6_18_pct": round(100 * va_poor_count / total, 1),
            "consanguineous_family_pct": round(100 * consanguineous_count / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(ASD_GENES):
        cohort = _make_cohort(entry, SEED_BASE + idx)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "morphology": entry["morphology"],
            "systemic_involvement": entry["systemic_involvement"],
            "onset_age": entry["onset_age"],
            "surgical_urgency": entry["surgical_urgency"],
            "gene_family": entry["gene_family"],
            "n_patients": len(cohort),
            "glaucoma_pct": round(100 * sum(1 for p in cohort if p["glaucoma"]) / len(cohort), 1),
            "corneal_opacity_pct": round(100 * sum(1 for p in cohort if p["corneal_opacity"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["surgery_performed"]) / len(cohort), 1),
            "va_poor_pct": round(100 * sum(1 for p in cohort if p["va_poor"]) / len(cohort), 1),
            "consanguineous_pct": round(100 * sum(1 for p in cohort if p["consanguineous"]) / len(cohort), 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:500],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "morphology": entry["morphology"],
                "systemic_involvement": entry["systemic_involvement"],
                "onset_age": entry["onset_age"],
                "surgical_urgency": entry["surgical_urgency"],
                "gene_family": entry["gene_family"],
            }
            for entry in ASD_GENES
        },
        "asd_glossary": {
            "Peters Anomaly — Types 1, 2, and Plus": (
                "Peters anomaly describes a specific anterior segment malformation: "
                "CENTRAL CORNEAL OPACITY with ABSENCE OF POSTERIOR CORNEAL STRUCTURES (posterior stroma + Descemet membrane + endothelium) "
                "in the opacity zone — diagnosed by anterior segment OCT or UBM. "
                "TYPE 1 (no lens touch): corneal opacity WITHOUT lens-cornea adhesion; "
                "genes: FOXE3 (ASD2), B3GLCT (Peters Plus), CYP1B1, PAX6, PITX2/FOXC1 (rare). "
                "TYPE 2 (with lens touch): lens ADHERES to posterior corneal surface in opacity zone — UBM shows direct contact; "
                "genes: FOXE3 primary; CYP1B1 secondary. "
                "PETERS PLUS SYNDROME (B3GLCT): Peters anomaly + SHORT STATURE + ID + CLEFT = diagnostic tetrad — "
                "the 'Plus' denotes the systemic features; "
                "SURGICAL IMPLICATION: bilateral dense Peters = ophthalmic emergency — PK within weeks; "
                "unilateral Peters with clear fellow eye: amblyopia management; "
                "concurrent glaucoma surgery when IOP elevated."
            ),
            "Axenfeld-Rieger Spectrum — ARS Triad and Glaucoma Risk": (
                "Axenfeld-Rieger Syndrome (ARS) is a spectrum of anterior segment dysgenesis with: "
                "POSTERIOR EMBRYOTOXON (anteriorly displaced Schwalbe line — visible at slit lamp as prominent white limbal ring): "
                "present in ARS + also as normal variant (15% population) — NOT pathological alone; "
                "IRIDOCORNEAL STRANDS (Axenfeld anomaly): iris bridges from pupil margin to posterior embryotoxon — "
                "PATHOLOGICAL; requires slit lamp gonioscopy for detection; "
                "IRIS STROMAL DYSGENESIS: iris stroma hypoplastic, correctopia (displaced pupil), pseudopolycoria; "
                "ARS TRIAD = posterior embryotoxon + iris strands + iris hypoplasia. "
                "GLAUCOMA RISK: 50% lifetime — trabecular meshwork hypoplasia from NCC defect; "
                "childhood-onset most common but can be adult; "
                "PITX2 (ARS1) vs FOXC1 (ARS3): clinically indistinguishable — genetic testing mandatory; "
                "FOXC1 DUPLICATION: iridogoniodysgenesis (iris hypoplasia + glaucoma without dental/umbilical) — "
                "MLPA/CNV array required to diagnose."
            ),
            "Primary Congenital Glaucoma — PCG Surgical Emergency Protocol": (
                "PCG (glaucoma from birth to age 3) from trabecular meshwork agenesis/dysgenesis. "
                "SYMPTOMS: excessive tearing (epiphora) + photophobia + blepharospasm — CLASSIC TRIAD in infant. "
                "SIGNS: buphthalmos (cornea >12mm newborn, >13mm by 1yr); "
                "Haab striae (horizontal white breaks in posterior cornea — Descemet tears from IOP stretch); "
                "corneal oedema (diffuse haze from raised IOP). "
                "CYP1B1 is the most common gene — accounts for 20-40% Western PCG, up to 95% consanguineous. "
                "SURGICAL EMERGENCY — goniotomy within days: "
                "goniotomy (trans-corneal incision through trabecular meshwork under gonioscopy) — "
                "success 80-90% first attempt if cornea clear; "
                "trabeculotomy (external approach) if cornea cloudy; "
                "MEDICAL PRE-OP: topical timolol + dorzolamide + oral acetazolamide — bridge only; "
                "AVOID brimonidine <2yr (respiratory depression, apnea risk in neonates); "
                "LONG-TERM: lifelong IOP monitoring — re-surgery common in teens; "
                "optic disc imaging + visual field from school age."
            ),
            "Aniridia Keratopathy — PAX6 Limbal Stem Cell Deficiency Management": (
                "ANIRIDIA is NOT just an iris disorder — it is a PAN-OCULAR disease with PROGRESSIVE CORNEAL FAILURE "
                "as a major cause of visual loss (beyond glaucoma + cataract + nystagmus). "
                "MECHANISM: PAX6 haploinsufficiency → limbal stem cell (LSC) depletion at the corneoscleral limbus → "
                "loss of the corneal epithelial progenitor reservoir → "
                "CONJUNCTIVALISATION (goblet cells + vessels invade cornea from limbus) → "
                "PANNUS FORMATION (superior vascularised opacity, then circumferential extension) → "
                "total corneal opacification (end-stage). "
                "STAGING: Stage 1 — superior pannus <1mm, no visual impact; "
                "Stage 2 — pannus >1mm, reduced best-corrected VA; Stage 3 — total clouding. "
                "TREATMENT: scleral contact lens — reduces friction + traps moisture + provides regular optical surface; "
                "CULTIVATED LIMBAL EPITHELIAL TRANSPLANTATION (CLET): culture biopsy of 1-2mm limbus → "
                "grow on amniotic membrane → transplant — restores corneal epithelium; "
                "SLET (simple limbal epithelial transplant): faster, less infrastructure; "
                "CORNEAL GRAFT (DALK/PKP): only after LSC transplant succeeded — "
                "graft without LSC: 100% failure from conjunctivalization."
            ),
            "WAGR Syndrome — Wilms Tumour Surveillance Protocol in PAX6 Deletion": (
                "WAGR syndrome = WILMS TUMOUR + ANIRIDIA + GU ANOMALIES + INTELLECTUAL DISABILITY RANGE — "
                "caused by 11p13 DELETION encompassing both PAX6 (aniridia) and WT1 (Wilms tumour suppressor). "
                "CHROMOSOME MICROARRAY MANDATORY in ALL new aniridia patients — cannot be excluded clinically. "
                "WILMS RISK: 45-60% of WAGR patients develop Wilms tumour (nephroblastoma); "
                "bilateral Wilms in 10% (vs 5% sporadic Wilms); "
                "SURVEILLANCE PROTOCOL: renal ultrasound every 3 months until age 8 years; "
                "screening stops at 8 (Wilms rare after age 8 in WT1 heterozygotes); "
                "URGENCY: a Wilms tumour can grow rapidly — surveillance must be consistent; "
                "GU ANOMALIES: cryptorchidism (males), hypospadias, renal structural variants — "
                "urology assessment mandatory at diagnosis; "
                "ID RANGE: wide spectrum from normal to moderate ID — "
                "neuropsychological assessment to guide educational support."
            ),
        },
    }
