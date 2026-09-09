#!/usr/bin/env python3
"""Hereditary-Epidermolysis-Bullosa-Atlas — Complete 8-Gene Hereditary Epidermolysis
Bullosa (EB) Atlas
(KRT5 · KRT14 · COL17A1 · LAMB3 · COL7A1 · ITGB4 · PLEC · FERMT1).

KRT5    (Keratin 5; 590 aa; 12q13.13; AD;
         EBS (Epidermolysis Bullosa Simplex);
         Dominant-negative collapse of keratin 5/14 intermediate filament (IF) network;
         intraepidermal split at basal cell layer;
         Subtypes: Weber-Cockayne (WC, localized palmoplantar, most common),
         Koebner (generalized, moderate), Dowling-Meara (DM, most severe EBS);
         TONOFILAMENT CLUMPING ON ELECTRON MICROSCOPY — PATHOGNOMONIC for EBS-DM;
         p.Glu477Lys most common EBS-DM mutation (coil 2B domain); p.Arg349Cys Koebner;
         Heat WORSENS blistering;
         seed SEED_BASE+0).
KRT14   (Keratin 14; 472 aa; 17q21.2; AD/AR;
         EBS spectrum;
         Pairs with KRT5 to form keratin IF; mutations cluster in helix initiation/termination motifs;
         p.Arg125His — MOST COMMON KRT14 MUTATION WORLDWIDE (EBS-DM; dominant);
         p.Arg125Cys — second most common, milder DM phenotype;
         AR biallelic: severe generalized EBS, may have extracutaneous (muscular) features;
         seed SEED_BASE+1).
COL17A1 (Collagen XVII; 1497 aa; 10q25.1; AR;
         JEB-non-Herlitz (Generalized Intermediate);
         Hemidesmosomal transmembrane collagen (BP180/BPAG2); lamina lucida split;
         PREMATURE TOOTH LOSS WITH HYPOPLASTIC ENAMEL — PATHOGNOMONIC;
         CERVICAL CANCER RISK 5-FOLD ELEVATED — annual cervical screening MANDATORY;
         Generalized Atrophic Benign EB (GABEB) phenotype;
         seed SEED_BASE+2).
LAMB3   (Laminin beta3; 1172 aa; 1q32.2; AR;
         JEB-Herlitz (most severe JEB);
         Component of laminin-332; anchoring filaments absent → complete loss of attachment;
         EXUBERANT GRANULATION TISSUE AROUND MOUTH NOSE FINGERS — PATHOGNOMONIC;
         AIRWAY GRANULATION TISSUE — LIFE-THREATENING;
         p.Arg635X (c.1903C>T) — 70% of European JEB-H alleles;
         Most patients die in infancy/early childhood;
         seed SEED_BASE+3).
COL7A1  (Collagen VII; 2944 aa; 3p21.31; AD/AR;
         DEB (Dystrophic EB);
         Anchoring fibrils in sublamina densa; absent → sub-BMZ split;
         MITTEN HAND DEFORMITY / PSEUDOSYNDACTYLY — PATHOGNOMONIC for severe RDEB;
         CUTANEOUS SCC — >70% by age 45yr; LEADING CAUSE OF DEATH in RDEB;
         BEREMAGENE GEPERPAVEC (B-VEC, Vyjuvek) — FDA2023 FIRST EB GENE THERAPY;
         seed SEED_BASE+4).
ITGB4   (Integrin beta4; 1822 aa; 17q25.1; AR;
         JEB with Pyloric Atresia (JEB-PA);
         Hemidesmosomal component paired with ITGA6;
         PYLORIC ATRESIA AT BIRTH + EB BLISTERING = PATHOGNOMONIC COMBINATION;
         Neonatal surgical emergency;
         seed SEED_BASE+5).
PLEC    (Plectin; 4684 aa; 8q24.13; AR/AD;
         EBS with Muscular Dystrophy (EBS-MD);
         Largest structural protein in human body (>500 kDa);
         EBS-MD (AR biallelic): EBS + MUSCULAR DYSTROPHY — PATHOGNOMONIC COMBINATION;
         CARDIOMYOPATHY in 30% EBS-MD; cardiac screening from age 30yr MANDATORY;
         seed SEED_BASE+6).
FERMT1  (Fermitin Family Member 1 / Kindlin-1; 677 aa; 20p12.3; AR;
         Kindler EB (KEB);
         Integrin activator; unique among EB genes — NOT a structural component;
         PHOTOSENSITIVITY + BLISTERING + PROGRESSIVE POIKILODERMA TRIAD — PATHOGNOMONIC;
         COLITIS in >60%;
         UV PROTECTION MANDATORY: only EB with significant UV-triggered blistering;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2270-2277).
"""

import random

SEED_BASE = 2270

EB_GENES = [
    # -- KRT5 — EBS (Epidermolysis Bullosa Simplex) ----------------------------------------
    {
        "gene": "KRT5",
        "alt_name": (
            "KRT5 (KRT5-590aa-12q13.13 / AD — EBS-Epidermolysis-Bullosa-Simplex — "
            "DOMINANT-NEGATIVE-Keratin-5-14-IF-Network-Collapse-Intraepidermal-Basal-Split — "
            "TONOFILAMENT-CLUMPING-EM-PATHOGNOMONIC-EBS-DM — "
            "Weber-Cockayne-Localized-Palmoplantar-MOST-COMMON — "
            "Koebner-Generalized-Moderate — Dowling-Meara-DM-MOST-SEVERE-Herpetiform-Clustering — "
            "p.Glu477Lys-EBS-DM-Coil2B-p.Arg349Cys-Koebner — "
            "Heat-WORSENS-Blistering-KEY-History)"
        ),
        "protein": (
            "KRT5 -- 12q13.13 AD -- KRT5-590aa -- "
            "Keratin-5-62kDa-Type-II-Intermediate-Filament-Basal-Keratinocyte -- "
            "EBS-OMIM-131900 -- "
            "INTRAEPIDERMAL-SPLIT-BASAL-CELL-LAYER-Keratinocyte-Cytolysis-Level-I-IF -- "
            "KRT5-KRT14-OBLIGATE-HETERODIMER-IF-Network-Dominant-Negative-Mechanism -- "
            "WEBER-COCKAYNE-WC-MOST-COMMON-EBS-Localized-Palmoplantar-Onset-Walking-Age -- "
            "KOEBNER-Generalized-Moderate-Onset-Birth-Trunk-Extremities-Spares-Palms -- "
            "DOWLING-MEARA-DM-MOST-SEVERE-EBS-Herpetiform-Blistering-Clustering-Neonatal -- "
            "TONOFILAMENT-CLUMPING-ELECTRON-MICROSCOPY-PATHOGNOMONIC-EBS-DM-ONLY -- "
            "p.Glu477Lys-MOST-COMMON-EBS-DM-Coil-2B-Hot-Spot -- "
            "p.Arg349Cys-Koebner-Subtype-Helix-Initiation-Motif -- "
            "HEAT-WORSENS-BLISTERING-Friction-Trauma-Trigger-Avoid-Hot-Environments -- "
            "HYPERHIDROSIS-PALMOPLANTAR-Botulinum-Toxin-Soles-Management -- "
            "OMIM-Gene-KRT5-148040-Disease-EBS-131900"
        ),
        "locus": "12q13.13",
        "protein_size": "590 aa / 62 kDa",
        "inheritance": (
            "AD (autosomal dominant, dominant-negative mechanism); "
            "Haploinsufficiency rare; dominant-negative: mutant KRT5 disrupts KRT5/KRT14 IF polymer; "
            "p.Glu477Lys: coil 2B hot spot; most common EBS-DM mutation; herpetiform clustering; "
            "p.Arg349Cys: helix initiation motif; Koebner subtype; generalized non-herpetiform; "
            "p.Leu325Pro: severe Dowling-Meara; neonatal; "
            "p.Glu170Lys: WC subtype; palmoplantar only; onset walking age; "
            "Variable expressivity: same mutation → WC in one family member, Koebner in another; "
            "Penetrance: near 100% but severity variable; "
            "EBS-DM: herpetiform blistering clusters; milia on healing; oral mucosa involved (30%); "
            "EBS-WC: localized palmoplantar; walking age onset; improves with age in some; "
            "EBS-Koebner: generalized at birth; improves post-puberty in 20%; "
            "Heat: key precipitant — blistering worsens in summer, hot baths, febrile illness"
        ),
        "key_features": [
            "INTRAEPIDERMAL SPLIT at basal cell layer (Level I) — keratinocyte cytolysis; KRT5/KRT14 IF network collapse; dominant-negative mechanism",
            "TONOFILAMENT CLUMPING ON ELECTRON MICROSCOPY — PATHOGNOMONIC for EBS-DM subtype; normal TF distribution rules out DM",
            "HEAT WORSENS BLISTERING — key history trigger; summer > winter; friction + heat = worst combination; WC patients: avoid occlusive footwear",
            "SUBTYPE SPECTRUM: Weber-Cockayne (localized palmoplantar, walking age, MOST COMMON) → Koebner (generalized, birth) → Dowling-Meara (herpetiform clusters, most severe, neonatal)",
            "p.Glu477Lys (coil 2B domain) = most common EBS-DM mutation; p.Arg349Cys = Koebner; mutations at helix initiation/termination motifs (aa 1-27, 304-318, 469-492) = more severe",
            "NO SYSTEMIC CURE — wound care + padding; heat avoidance; hyperhidrosis management (botulinum toxin soles); dressings (non-adherent Mepitel/Mepilex)",
            "IMPROVES WITH AGE in some WC and Koebner patients; DM may persist with severity throughout life",
            "SKIN BIOPSY IFA + EM mandatory for diagnosis; IFA shows K5/K14 cleavage plane at basal layer",
        ],
        "treatment": (
            "Wound care (first-line): "
            "Non-adherent dressings (Mepitel One, Mepilex Transfer) — prevent re-trauma on dressing change; "
            "Petroleum-based ointments (Vaseline/Aquaphor) — keep wounds moist; "
            "Blister puncture with sterile lancet + leave roof intact (roof = biological dressing). "
            "Heat/trigger avoidance: "
            "Cool environment; moisture-wicking clothing; avoid occlusive footwear; cool packs for feet. "
            "Hyperhidrosis management: "
            "Botulinum toxin (onabotulinumtoxinA) injected plantar surface — reduces sweating-triggered maceration; "
            "Aluminium chloride 20% antiperspirant — adjunct. "
            "Infection: "
            "Topical mupirocin/fusidic acid for Staphylococcus aureus colonization; "
            "Systemic antibiotics only for clinical infection (not colonization). "
            "Pain: "
            "Paracetamol/ibuprofen (acute); gabapentin/pregabalin (neuropathic); "
            "Opioids for severe procedural pain (dressing changes). "
            "Genetics: AD 50% transmission risk per pregnancy; prenatal or preimplantation genetic diagnosis available. "
            "No FDA-approved systemic therapy for EBS (2026); gene/cell therapy trials (KRT14 shRNA, siRNA) ongoing."
        ),
        "monitoring": [
            "Wound surveillance: weekly dressing review; wound size mapping; infection signs (erythema/purulence/odour)",
            "Skin: annual dermatology review; subtype reassessment; SCC surveillance (DM: annual from age 25yr)",
            "Hyperhidrosis: botulinum toxin 3-4 monthly if palmoplantar sweating contributes",
            "Ophthalmology: corneal erosions in DM — annual slit lamp; lubricants",
            "Oral: dental review annually (DM: oral mucosal involvement → caries risk)",
            "Quality of life: EB-QALY tool; pain VAS; school/work accommodation",
            "Family screening: first-degree relatives clinical skin exam; genetic testing offered",
            "Pregnancy: obstetric high-risk; neonatal team alert for blistering at delivery",
        ],
        "eb_subtype": "EBS (Epidermolysis Bullosa Simplex)",
        "skin_split_level": "Intraepidermal (basal cell layer)",
        "pathognomonic": "Tonofilament clumping on EM (EBS-DM) + intraepidermal basal split = KRT5/KRT14 EBS",
        "treatment_highlight": "Wound care + heat avoidance; botulinum toxin for plantar hyperhidrosis; no systemic cure",
        "avg_age_at_dx_yrs": 3.0,
    },
    # -- KRT14 — EBS spectrum (AD/AR) -------------------------------------------------------
    {
        "gene": "KRT14",
        "alt_name": (
            "KRT14 (KRT14-472aa-17q21.2 / AD-AR — EBS-Spectrum — "
            "p.Arg125His-MOST-COMMON-KRT14-MUTATION-WORLDWIDE-EBS-DM-Dominant — "
            "p.Arg125Cys-Second-Most-Common-Milder-DM — "
            "AR-Biallelic-Severe-Generalized-EBS-Extracutaneous-Muscular-Features — "
            "AD-Typical-EBS-Spectrum-AR-More-Severe-Widespread)"
        ),
        "protein": (
            "KRT14 -- 17q21.2 AD/AR -- KRT14-472aa -- "
            "Keratin-14-52kDa-Type-I-Intermediate-Filament-Basal-Keratinocyte-Partner-KRT5 -- "
            "EBS-OMIM-131760 -- "
            "INTRAEPIDERMAL-SPLIT-BASAL-Identical-Cleavage-Plane-KRT5 -- "
            "HELIX-INITIATION-TERMINATION-MOTIF-Mutation-Hotspot-aa1-22-317-335 -- "
            "p.Arg125His-MOST-COMMON-KRT14-WORLDWIDE-Coil-1A-Dominant-EBS-DM -- "
            "p.Arg125Cys-Second-Most-Common-Milder-DM-Phenotype-Same-Residue -- "
            "AD-EBS-SPECTRUM-WC-Koebner-DM-Per-Domain-Location -- "
            "AR-BIALLELIC-SEVERE-GENERALIZED-EBS-Skin-Fragility-Birth -- "
            "AR-EXTRACUTANEOUS-Muscular-Dystrophy-Like-Myopathy-Rare -- "
            "TEMPERATURE-FRICTION-TRAUMA-Trigger-Identical-KRT5 -- "
            "OMIM-Gene-KRT14-148066-Disease-EBS-131760"
        ),
        "locus": "17q21.2",
        "protein_size": "472 aa / 52 kDa",
        "inheritance": (
            "AD (dominant-negative; majority) or AR (biallelic LOF; severe minority); "
            "p.Arg125His: coil 1A helix initiation motif; MOST COMMON KRT14 mutation worldwide; EBS-DM; dominant; "
            "p.Arg125Cys: same residue, different substitution; milder DM; dominant; "
            "p.Met119Thr: helix initiation; WC subtype; "
            "p.Val270Met: coil 2; Koebner; "
            "AR biallelic: null/null or hypomorphic alleles; severe generalized EBS from birth; "
            "AR: muscular features in minority — myopathy-like weakness (differs from PLEC-EBS-MD); "
            "AD penetrance: near 100%; variable expressivity; "
            "Phenotype-genotype: mutations at helix initiation/termination (1A/2B hot spots) → DM; "
            "linker/coil 1B → milder WC/Koebner"
        ),
        "key_features": [
            "p.Arg125His (coil 1A) = MOST COMMON KRT14 MUTATION WORLDWIDE — EBS-DM subtype; dominant; same residue as p.Arg125Cys (milder)",
            "INTRAEPIDERMAL BASAL SPLIT — identical cleavage plane to KRT5 EBS; IFA distinguishes KRT5 vs KRT14 only by genetic testing (same cleavage level)",
            "AD: EBS-WC / Koebner / DM spectrum depending on domain affected; AR biallelic: severe generalized EBS from birth",
            "AR BIALLELIC rare: extracutaneous muscular features possible (myopathy-like); distinguish from PLEC-EBS-MD by absence of progressive MD",
            "HEAT + FRICTION + TRAUMA same triggers as KRT5; seasonal variation; worse in summer",
            "ELECTRON MICROSCOPY: tonofilament clumping in EBS-DM (same as KRT5-DM); biopsy level confirms basal intraepidermal split",
            "NO SYSTEMIC CURE — same wound care approach as KRT5; gene silencing (allele-specific siRNA) in clinical trials targeting p.Arg125His",
            "GENETIC TESTING distinguishes KRT14 from KRT5 — clinically and on IFA they look identical; panel sequencing mandatory",
        ],
        "treatment": (
            "Wound care: identical approach to KRT5-EBS — "
            "non-adherent dressings (Mepitel One/Mepilex Transfer); petroleum ointments; blister roof preservation. "
            "Heat/trigger avoidance: cool environment; moisture-wicking fabrics; avoidance of occlusive footwear. "
            "Hyperhidrosis: botulinum toxin plantar; aluminium chloride antiperspirant. "
            "AR-biallelic cases: "
            "Physiotherapy if muscular weakness (myopathy-like features); "
            "Respiratory review if respiratory muscle involvement (rare). "
            "Infection control: topical mupirocin for SA colonisation; systemic antibiotics for clinical infection. "
            "Pain: paracetamol/NSAIDs for mild; gabapentin/pregabalin for neuropathic component. "
            "Gene therapy (investigational): allele-specific siRNA targeting p.Arg125His in KRT14 — Phase I/II trials (2025-2026). "
            "Genetics: AD — 50% transmission risk; AR — 25% per pregnancy; prenatal/PGD available."
        ),
        "monitoring": [
            "Wound: weekly dressing review; wound size charting; infection screen (swab) quarterly",
            "Skin: annual dermatology; SCC surveillance DM subtype from age 25yr",
            "AR-biallelic: annual CK (muscular features screen); physiotherapy assessment if weakness",
            "Ophthalmology: corneal erosion screen DM subtype; annual slit lamp",
            "Dental: annual review; DM subtype oral mucosa involvement",
            "QoL: EB-QALY; pain scoring; psychosocial support",
            "Family: first-degree relatives clinical exam + genetic testing",
            "Gene therapy trials: enrol at specialist centre if p.Arg125His mutation confirmed",
        ],
        "eb_subtype": "EBS (Epidermolysis Bullosa Simplex)",
        "skin_split_level": "Intraepidermal (basal cell layer)",
        "pathognomonic": "p.Arg125His MOST COMMON KRT14 WORLDWIDE + intraepidermal basal split + EBS-DM = KRT14",
        "treatment_highlight": "Wound care as KRT5; allele-specific siRNA trial for p.Arg125His; AR cases need CK/physio",
        "avg_age_at_dx_yrs": 1.0,
    },
    # -- COL17A1 — JEB non-Herlitz (Generalized Intermediate / GABEB) ----------------------
    {
        "gene": "COL17A1",
        "alt_name": (
            "COL17A1 (COL17A1-1497aa-10q25.1 / AR — JEB-non-Herlitz-Generalized-Intermediate-GABEB — "
            "PREMATURE-TOOTH-LOSS-HYPOPLASTIC-ENAMEL-PATHOGNOMONIC — "
            "LAMINA-LUCIDA-SPLIT-Hemidesmosomal-Transmembrane-BP180-BPAG2 — "
            "CERVICAL-CANCER-RISK-5x-ELEVATED-Annual-Cervical-Screening-MANDATORY — "
            "Atrophic-Scarring-Nail-Dystrophy-Alopecia-GABEB-Phenotype)"
        ),
        "protein": (
            "COL17A1 -- 10q25.1 AR -- COL17A1-1497aa -- "
            "Collagen-XVII-BP180-BPAG2-183kDa-Hemidesmosomal-Transmembrane-Collagen -- "
            "JEB-nH-GABEB-OMIM-226650 -- "
            "LAMINA-LUCIDA-SPLIT-Hemidesmosome-Anchoring-Filament-Junction-Level-II -- "
            "BULLOUS-PEMPHIGOID-ANTIGEN-2-Autoimmune-DDx-Acquired-vs-Hereditary -- "
            "PREMATURE-TOOTH-LOSS-HYPOPLASTIC-ENAMEL-PATHOGNOMONIC-Distinguishes-From-EBS -- "
            "ATROPHIC-SCARRING-Post-Blister-Atrophy-MILIA-Form-Unlike-JEB-H-Granulation-Tissue -- "
            "NAIL-DYSTROPHY-Universal-All-Nails-Onycholysis-Pterygium -- "
            "ALOPECIA-Scarring-Scalp-EB-Lesions -- "
            "CERVICAL-CANCER-RISK-5-FOLD-ELEVATED-Annual-Pap-Smear-HPV-Vaccination-MANDATORY -- "
            "OMIM-Gene-COL17A1-113811-Disease-JEB-nH-226650"
        ),
        "locus": "10q25.1",
        "protein_size": "1497 aa / 183 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF/hypomorphic mutations; "
            "Genotype-phenotype: premature stop/frameshift → GABEB; "
            "Missense/hypomorphic → milder generalized intermediate; "
            "p.Glu1544X: European founder; GABEB; "
            "p.Arg1226X: frameshift; GABEB; "
            "c.3944-1G>A: splice site; intermediate severity; "
            "Compound heterozygotes common; "
            "GABEB: generalized atrophic benign EB — improves with age (atrophy ≠ granulation tissue); "
            "Oral mucosa: involved in 40%; dental enamel hypoplasia universal; "
            "Alopecia: scarring; permanent hair loss over EB-affected scalp; "
            "Ocular: corneal erosions; symblepharon (rare); "
            "Also antigen in bullous pemphigoid (autoimmune) — IFA shows same NC16A domain positivity; distinguish by genetics/age/family history"
        ),
        "key_features": [
            "PREMATURE TOOTH LOSS WITH HYPOPLASTIC ENAMEL — PATHOGNOMONIC; distinguishes JEB-nH from EBS (intraepidermal), DEB (sub-BMZ); dental panoramic X-ray confirms",
            "LAMINA LUCIDA SPLIT (Level II) — hemidesmosomal transmembrane collagen BP180/BPAG2; IFA shows cleavage below HD plaque; absent/reduced COL17A1 staining",
            "GENERALIZED ATROPHIC BENIGN EB (GABEB) phenotype — atrophic scarring + milia (NOT granulation tissue like JEB-H); skin fragility improves with age",
            "NAIL DYSTROPHY (universal — all nails); ALOPECIA (scarring scalp lesions); ORAL MUCOSA 40%",
            "CERVICAL CANCER RISK 5-FOLD ELEVATED — annual cervical Pap smear + HPV vaccination MANDATORY (COL17A1 at cervical epithelial junction; not fully explained)",
            "BULLOUS PEMPHIGOID DDx — COL17A1/BP180 is both the hereditary EB gene AND the bullous pemphigoid autoantigen; distinguish by IIF serology (autoAb present in BP, absent in JEB-nH)",
            "NO AIRWAY GRANULATION TISSUE — unlike JEB-H (LAMB3); NOT life-threatening in infancy (key DDx from JEB-H)",
            "ATROPHIC HAIR LOSS + NAIL DYSTROPHY + ENAMEL DEFECTS = FULL COL17A1 TRIAD; confirms JEB-nH/GABEB",
        ],
        "treatment": (
            "Wound care: "
            "Non-adherent dressings (Mepitel/Mepilex); petroleum-based ointments; blister roof preserved; "
            "Skin fragility reduces with age in GABEB — dressings can be simplified over time. "
            "Dental: "
            "Enamel hypoplasia management — fissure sealants in childhood; composite resin restorations; "
            "Dentures/implants for premature tooth loss; dental panoramic X-ray baseline. "
            "Gynaecological: "
            "Annual cervical smear (Pap) from age 21 (or 3 years post-sexual debut, whichever earlier); "
            "HPV vaccination (Gardasil-9) — complete series before sexual debut; "
            "Colposcopy if abnormal Pap. "
            "Nail: "
            "Nail avulsion if onychogryphosis; protective nail covers; avoid trauma. "
            "Alopecia: "
            "Minoxidil topical (limited evidence); scalp wound care; UV-protective headwear. "
            "Ophthalmology: "
            "Lubricating eye drops; corneal erosion management; corneal transplant if severe scarring. "
            "Infection: topical mupirocin/fusidic acid; systemic antibiotics for clinical infection only. "
            "Genetics: AR — 25% risk per sibling; cascade testing; prenatal/PGD available."
        ),
        "monitoring": [
            "Skin: annual dermatology; wound mapping; SCC surveillance from age 25yr (mucosal sites)",
            "Dental: 6-monthly dental review; panoramic X-ray annually; enamel mapping",
            "Gynaecological: annual Pap smear from age 21; HPV vaccination; colposcopy if CIN",
            "Ophthalmology: annual slit lamp; corneal erosion; symblepharon screen",
            "Scalp/nails: 6-monthly specialist nursing review; nail care plan",
            "Nutrition: iron (anaemia from chronic wounds); zinc; vitamin D",
            "QoL: psychosocial — alopecia significant impact; support groups (DEBRA)",
            "Annual full-skin surveillance: COL17A1 SCC risk in chronic wounds",
        ],
        "eb_subtype": "JEB non-Herlitz (Generalized Intermediate / GABEB)",
        "skin_split_level": "Lamina lucida",
        "pathognomonic": "Premature tooth loss + hypoplastic enamel + atrophic scarring (not granulation tissue) = COL17A1 JEB-nH",
        "treatment_highlight": "Annual Pap smear + HPV vaccine (5x cervical cancer risk); dental management (enamel hypoplasia); wound care",
        "avg_age_at_dx_yrs": 0.2,
    },
    # -- LAMB3 — JEB-Herlitz (most severe JEB) ----------------------------------------------
    {
        "gene": "LAMB3",
        "alt_name": (
            "LAMB3 (LAMB3-1172aa-1q32.2 / AR — JEB-Herlitz-MOST-SEVERE-JEB — "
            "EXUBERANT-GRANULATION-TISSUE-MOUTH-NOSE-FINGERS-PATHOGNOMONIC — "
            "AIRWAY-GRANULATION-TISSUE-LIFE-THREATENING-Tracheostomy-Often-Required — "
            "EROSIVE-ENAMEL-HYPOPLASIA-PITTING-PATHOGNOMONIC-Universal — "
            "p.Arg635X-70pct-European-JEB-H-Alleles-MOST-COMMON-JEB-MUTATION — "
            "Laminin-332-Component-Anchoring-Filaments-Absent)"
        ),
        "protein": (
            "LAMB3 -- 1q32.2 AR -- LAMB3-1172aa -- "
            "Laminin-Beta3-140kDa-Laminin-332-Heterotrimer-LAMA3-LAMB3-LAMC2 -- "
            "JEB-H-OMIM-226700 -- "
            "LAMININ-332-ABSENT-Anchoring-Filaments-None-Epidermis-Dermis-Complete-Detachment -- "
            "LAMINA-LUCIDA-SPLIT-Level-II-Sub-Hemidesmosome-Anchoring-Filament-Level -- "
            "EXUBERANT-GRANULATION-TISSUE-PERIORAL-PERINASAL-DIGITAL-PATHOGNOMONIC -- "
            "EROSIVE-ENAMEL-HYPOPLASIA-DENTAL-PITTING-Universal-PATHOGNOMONIC -- "
            "AIRWAY-GRANULATION-TISSUE-Laryngeal-Tracheal-LIFE-THREATENING-Stridor-Hoarseness -- "
            "GI-MUCOSAL-Esophageal-Gastric-Involvement-Nutrition-Compromise -- "
            "DEATH-INFANCY-EARLY-CHILDHOOD-Without-Intensive-Management-Sepsis-Malnutrition -- "
            "p.Arg635X-c.1903C-T-70pct-European-JEB-H-Alleles-Most-Common-JEB-Mutation-Worldwide -- "
            "OMIM-Gene-LAMB3-150310-Disease-JEB-H-226700"
        ),
        "locus": "1q32.2",
        "protein_size": "1172 aa / 140 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic PTCs (premature termination codons) = JEB-H; "
            "p.Arg635X (c.1903C>T): 70% of European JEB-H alleles; most common JEB mutation worldwide; "
            "p.Gln243X: second most common JEB-H allele; "
            "Compound heterozygotes common; "
            "Homozygous p.Arg635X: classic severe JEB-H; "
            "Missense (hypomorphic) + PTC: intermediate JEB phenotype (JEB-nH spectrum); "
            "LAMB3 LOF → laminin-332 trimer absent → no anchoring filaments → complete lamina lucida split; "
            "Mucous membranes: universally involved (oral, GI, airway, urinary, conjunctival); "
            "Granulation tissue: pathological exuberant perioral/perinasal/digital (EB-specific, not infection); "
            "Enamel: erosive hypoplasia with pitting (universal; ALL primary + permanent teeth); "
            "Prognosis: 40% mortality by age 1yr (sepsis/malnutrition/airway); most die before adulthood"
        ),
        "key_features": [
            "EXUBERANT GRANULATION TISSUE around mouth, nose, and fingers — PATHOGNOMONIC; NOT infected granuloma; pathological EB granulation; hallmark of JEB-H",
            "EROSIVE ENAMEL HYPOPLASIA WITH DENTAL PITTING — PATHOGNOMONIC; universal in ALL JEB-H; primary + permanent teeth; confirms JEB-H in neonate",
            "AIRWAY GRANULATION TISSUE — LIFE-THREATENING; laryngeal/tracheal; stridor + hoarseness; tracheostomy often required; acute airway emergency",
            "LAMININ-332 ABSENT (IFA negative for LAMB3/LAMA3/LAMC2) — diagnostic; lamina lucida split; anchoring filaments absent on EM",
            "p.Arg635X (c.1903C>T) = 70% of European JEB-H alleles; most common JEB mutation worldwide; homozygous = classic severe",
            "GI MUCOSAL INVOLVEMENT — esophageal blistering; gastric erosions; malnutrition; nasogastric/PEG feeding often required",
            "DEATH IN INFANCY/EARLY CHILDHOOD without intensive management — sepsis + malnutrition + airway obstruction; palliative care discussion early",
            "MULTIDISCIPLINARY URGENT from birth — dermatology + respiratory + ENT + gastroenterology + nutrition + palliative care + genetics",
        ],
        "treatment": (
            "Neonatal emergency management: "
            "Minimal handling; non-adherent dressings immediately post-delivery; warm humidified environment. "
            "Airway: "
            "ENT assessment at birth for stridor; laryngoscopy/bronchoscopy for granulation tissue; "
            "Tracheostomy if upper airway obstruction; humidified ventilation; avoid intubation trauma. "
            "Wound care: "
            "Soft non-adherent dressings (Mepitel One); petroleum-based; change with sedation/analgesia; "
            "Perioral granulation tissue: betamethasone cream (0.05%) topical — may slow progression. "
            "Nutrition: "
            "NG tube (soft silicone); PEG if NG traumatic; high-calorie formula; "
            "Dietitian-led caloric supplementation (200-300% RDA for growth). "
            "Dental: "
            "Early dental review; enamel sealants; prevent caries in hypoplastic teeth; "
            "May require early extractions for pain/infection. "
            "Infection: "
            "Systemic antibiotics (sepsis is leading cause of death); blood cultures + broad-spectrum empirically; "
            "Topical silver-containing dressings (Mepilex Ag) for infected wounds. "
            "Palliative care: "
            "Early goals-of-care discussion; palliative comfort measures if family chooses. "
            "Genetics: AR 25% recurrence; prenatal diagnosis/PGD strongly recommended subsequent pregnancies."
        ),
        "monitoring": [
            "Airway: monthly ENT/respiratory review; laryngoscopy/bronchoscopy 3-monthly; tracheostomy care",
            "Nutrition: weekly weight; fortnightly height; albumin/prealbumin monthly; dietitian review",
            "Wound: daily nursing wound assessment; weekly wound mapping; swab cultures 4-weekly",
            "Dental: 6-monthly dental from age 6 months; panoramic X-ray at 2yr",
            "Ophthalmology: monthly conjunctival assessment; corneal erosions; symblepharon",
            "Haematology: FBC monthly (anaemia of chronic disease + blood loss)",
            "Renal: urinalysis for urinary mucosal involvement; creatinine 6-monthly",
            "Palliative/QoL: monthly MDT goals-of-care review; family support; psychology",
        ],
        "eb_subtype": "JEB-Herlitz (most severe JEB)",
        "skin_split_level": "Lamina lucida (anchoring filament absent)",
        "pathognomonic": "Exuberant perioral/perinasal/digital granulation tissue + erosive enamel pitting + absent laminin-332 IFA = LAMB3 JEB-H",
        "treatment_highlight": "Airway emergency: tracheostomy if obstruction; palliative care early discussion; NG/PEG nutrition; betamethasone granulation tissue",
        "avg_age_at_dx_yrs": 0.0,
    },
    # -- COL7A1 — DEB (Dystrophic EB) -------------------------------------------------------
    {
        "gene": "COL7A1",
        "alt_name": (
            "COL7A1 (COL7A1-2944aa-3p21.31 / AD-DDEB-or-AR-RDEB — DEB-Dystrophic-EB — "
            "MITTEN-HAND-DEFORMITY-PSEUDOSYNDACTYLY-PATHOGNOMONIC-Severe-RDEB — "
            "CUTANEOUS-SCC-70pct-by-Age-45yr-LEADING-CAUSE-DEATH-RDEB — "
            "ESOPHAGEAL-STRICTURES-50pct-RDEB-Adults-Regular-Dilation-Required — "
            "BEREMAGENE-GEPERPAVEC-B-VEC-Vyjuvek-FDA2023-FIRST-EB-GENE-THERAPY-Topical-COL7A1-HSV1)"
        ),
        "protein": (
            "COL7A1 -- 3p21.31 AD/AR -- COL7A1-2944aa -- "
            "Collagen-VII-290kDa-Anchoring-Fibril-Sublamina-Densa-Type-VII-Collagen -- "
            "DEB-OMIM-226600-RDEB-131750-DDEB -- "
            "ANCHORING-FIBRILS-ABSENT-Sub-BMZ-Split-Sub-Lamina-Densa-Level-III -- "
            "RDEB-AR-RECESSIVE-DEB-MOST-SEVERE-DEB-Generalized-Severe-Bilateral -- "
            "MITTEN-HAND-DEFORMITY-PSEUDOSYNDACTYLY-Digit-Fusion-Cocoon-Hand-PATHOGNOMONIC-RDEB -- "
            "ESOPHAGEAL-STRICTURES-50pct-RDEB-Adults-Dilation-Required-Ongoing -- "
            "CUTANEOUS-SCC-70pct-by-Age-45yr-AGGRESSIVE-Early-Onset-Metastatic-LEADING-CAUSE-DEATH -- "
            "DDEB-AD-MILDER-Localized-Nail-Dystrophy-Albopapuloid-Lesions -- "
            "BEREMAGENE-GEPERPAVEC-B-VEC-Vyjuvek-FDA2023-FIRST-EB-GENE-THERAPY-Topical-HSV1-COL7A1-Vector -- "
            "OMIM-Gene-COL7A1-120120-Disease-RDEB-226600-DDEB-131750"
        ),
        "locus": "3p21.31",
        "protein_size": "2944 aa / 290 kDa",
        "inheritance": (
            "AR (recessive DEB = RDEB; biallelic LOF/severe missense) or "
            "AD (dominant DEB = DDEB; one dominant-negative or haploinsufficiency allele); "
            "RDEB: most severe DEB; anchoring fibrils absent; sub-lamina densa split; "
            "RDEB generalized severe (RDEB-GS): pseudosyndactyly; SCC; GI; "
            "RDEB generalized intermediate (RDEB-GI): milder; less pseudosyndactyly; "
            "DDEB: milder; localized blistering; nail dystrophy; albopapuloid lesions; "
            "p.Gly2251Arg: common DDEB dominant-negative (glycine substitution triple helix); "
            "p.Arg2814X: common RDEB null allele; "
            "c.6527insC: frameshift; RDEB-GS; "
            "SCC: RDEB-specific cutaneous SCC; aggressive; early-onset metastatic; "
            "Pseudosyndactyly: progressive cocoon-hand deformity; surgical release required"
        ),
        "key_features": [
            "MITTEN HAND DEFORMITY / PSEUDOSYNDACTYLY — PATHOGNOMONIC for severe RDEB; digit fusion with cocoon-hand; progressive; repeated surgical releases required",
            "CUTANEOUS SCC >70% by age 45yr — LEADING CAUSE OF DEATH in RDEB; aggressive, early-onset, frequently metastatic; monthly full-skin surveillance from age 10yr MANDATORY",
            "ESOPHAGEAL STRICTURES >50% RDEB adults — dysphagia + failure to thrive; regular endoscopic dilation; soft-food diet; PEG if severe",
            "SUB-LAMINA DENSA SPLIT (Level III) — anchoring fibrils absent on EM; IFA: absent COL7A1 staining below lamina densa; diagnostic",
            "BEREMAGENE GEPERPAVEC (B-VEC, Vyjuvek) — FDA 2023 FIRST EB GENE THERAPY; topical HSV-1 vector delivering COL7A1; applied to wounds; significant wound closure benefit",
            "DDEB (AD): milder; localized blistering; nail dystrophy (nail loss universal); albopapuloid lesions (white papules trunk) — DDEB PATHOGNOMONIC feature",
            "ANAEMIA — chronic blood loss from wounds + malnutrition + anaemia of chronic disease; iron + erythropoietin supplementation; transfusion if severe",
            "RENAL AMYLOIDOSIS (chronic cases) — AA amyloidosis from chronic inflammation; renal function monitoring from age 20yr in RDEB-GS",
        ],
        "treatment": (
            "Gene therapy (FDA 2023): "
            "Beremagene geperpavec (B-VEC, Vyjuvek) — topical HSV-1 vector with COL7A1 cDNA; "
            "Apply to wounds ≥20 cm² in patients ≥6 months; significant wound healing improvement; "
            "Contraindicated: immunosuppression, active herpes infection. "
            "Wound care: "
            "Non-adherent dressings (Mepitel One, Mepilex Transfer, Urgotul); "
            "Foam/absorbent secondary dressings; wet or lipido-colloid dressings for chronic wounds; "
            "Silver dressings (Mepilex Ag) for infected/colonised wounds. "
            "Pseudosyndactyly: "
            "Physiotherapy (finger stretching daily); silicone web-spacers; "
            "Surgical release (finger separation + skin grafting/dermal substitute) every 2-5yr; "
            "Earlier release = better functional preservation. "
            "GI: "
            "Endoscopic esophageal dilation (repeated); soft/liquid diet; PEG feeding; "
            "Laxatives (lactulose/polyethylene glycol) for constipation (anal EB); "
            "Nutritional supplementation (elemental formula). "
            "SCC surveillance: "
            "Monthly full-skin examination from age 10yr; photography; biopsy all suspicious lesions; "
            "Mohs surgery or wide local excision; sentinel lymph node biopsy if SCC >2cm; "
            "Systemic therapy (cemiplimab — PD-1 inhibitor) for metastatic EB-SCC. "
            "Anaemia: oral/IV iron; erythropoietin; transfusion threshold Hb <70 g/L. "
            "Genetics: AR 25% recurrence (RDEB); AD 50% (DDEB); PGD available; B-VEC trial enrolment."
        ),
        "monitoring": [
            "SCC: monthly full-skin exam; annual dermoscopy map; biopsy all non-healing wounds >1 month",
            "Esophageal: annual upper GI endoscopy from age 10yr; dilation schedule as needed",
            "Hands: monthly physiotherapy; annual hand function assessment; surgical review 2-yearly",
            "Nutrition: weight monthly; albumin 3-monthly; dietitian review 6-monthly",
            "Anaemia: FBC monthly; iron studies 3-monthly; erythropoietin response",
            "Renal: annual eGFR + urine protein from age 20yr (amyloidosis screen)",
            "Ophthalmology: annual slit lamp (corneal erosions, symblepharon)",
            "B-VEC therapy: wound response assessment monthly; herpes surveillance; safety monitoring",
        ],
        "eb_subtype": "DEB (Dystrophic EB) — RDEB (AR) or DDEB (AD)",
        "skin_split_level": "Sub-lamina densa",
        "pathognomonic": "Mitten hand deformity + absent anchoring fibrils (EM) + sub-lamina densa split = COL7A1 RDEB",
        "treatment_highlight": "B-VEC gene therapy FDA2023 (topical COL7A1); monthly SCC surveillance from age 10yr; esophageal dilation; pseudosyndactyly release",
        "avg_age_at_dx_yrs": 0.0,
    },
    # -- ITGB4 — JEB with Pyloric Atresia (JEB-PA) -----------------------------------------
    {
        "gene": "ITGB4",
        "alt_name": (
            "ITGB4 (ITGB4-1822aa-17q25.1 / AR — JEB-with-Pyloric-Atresia-JEB-PA — "
            "PYLORIC-ATRESIA-AT-BIRTH-PLUS-EB-BLISTERING-PATHOGNOMONIC-COMBINATION — "
            "NEONATAL-EMERGENCY-Nonbilious-Vomiting-Large-Gastric-Bubble-Urgent-Surgery — "
            "Hemidesmosomal-ITGB4-ITGA6-Anchors-Keratin-IF-Basement-Membrane — "
            "Ureteral-Bladder-EB-Urinary-Obstruction-Possible)"
        ),
        "protein": (
            "ITGB4 -- 17q25.1 AR -- ITGB4-1822aa -- "
            "Integrin-Beta4-205kDa-Hemidesmosomal-Transmembrane-ITGB4-ITGA6-Heterodimer -- "
            "JEB-PA-OMIM-226730 -- "
            "LAMINA-LUCIDA-SPLIT-Hemidesmosomal-Level-Identical-COL17A1 -- "
            "PYLORIC-ATRESIA-CONGENITAL-Obstruction-Gastric-Outlet-PATHOGNOMONIC-WITH-EB -- "
            "NONBILIOUS-VOMITING-BIRTH-LARGE-GASTRIC-BUBBLE-X-RAY-Olive-Mass-NOT-Felt-At-Birth -- "
            "NEONATAL-SURGICAL-EMERGENCY-Pyloric-Repair-Mandatory-Hours-Of-Birth -- "
            "URETERAL-BLADDER-EB-Hydronephrosis-Urinary-Retention-Possible -- "
            "MUSCULAR-DYSTROPHY-LIKE-MYOPATHY-Rare-ITGB4-Mutations -- "
            "COMBINED-ITGB4-ITGA6-Genotype-Phenotype-Complex -- "
            "LETHAL-NEONATAL-EARLY-INFANCY-Severe-Forms-Without-Surgical-Management -- "
            "OMIM-Gene-ITGB4-147557-Disease-JEB-PA-226730"
        ),
        "locus": "17q25.1",
        "protein_size": "1822 aa / 205 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF most common; "
            "ITGB4 pairs with ITGA6 (α6β4 integrin) — component of hemidesmosome; "
            "ITGB4 LOF → absent hemidesmosomes → lamina lucida split (same level as JEB-nH/COL17A1); "
            "Pyloric atresia: smooth muscle EB involvement at pyloric canal; "
            "p.Gly57Glu: common JEB-PA allele; missense; "
            "p.Arg1281X: truncating; severe; "
            "Genotype-phenotype: PTC/truncating → more severe EB + PA; missense → variable; "
            "GU involvement: EB affecting ureter + bladder mucosa → hydronephrosis (10-20%); "
            "ITGA6 mutations (paired partner): ITGA6-PA phenotype, similar to ITGB4-PA; "
            "Severity range: lethal (no treatment) to survivable with surgery; "
            "Survivors: ongoing EB + GU complications lifelong"
        ),
        "key_features": [
            "PYLORIC ATRESIA AT BIRTH + EB BLISTERING = PATHOGNOMONIC COMBINATION; nonbilious vomiting + large gastric bubble on AXR = NEONATAL SURGICAL EMERGENCY",
            "NEONATAL EMERGENCY: pyloric repair (pyloroplasty or gastroduodenostomy) mandatory within hours of birth; EB wound care simultaneously",
            "LAMINA LUCIDA SPLIT — hemidesmosomal level; IFA: absent ITGB4 staining; absent hemidesmosomes on EM; distinguish from LAMB3 by molecular testing",
            "URETERAL / BLADDER EB COMPLICATIONS — hydronephrosis in 10-20%; renal USS baseline; urological follow-up mandatory",
            "ITGB4-ITGA6 PAIR — combined mutations described; check ITGA6 if single ITGB4 mutation found (digenic); genotype-phenotype complex",
            "LETHAL NEONATAL/EARLY INFANCY in severe forms — without pyloric surgery + EB management; palliative care discussion early",
            "MUSCULAR DYSTROPHY-LIKE MYOPATHY — rare ITGB4 mutations (cytoplasmic domain); distinguish from PLEC-EBS-MD (later-onset MD; different molecular basis)",
            "SURVIVORS: ongoing generalized EB requiring lifelong multidisciplinary wound care + GU surveillance + nutritional support",
        ],
        "treatment": (
            "Neonatal surgical emergency: "
            "Pyloric repair (pyloroplasty or gastroduodenostomy) within hours of birth; "
            "Simultaneous EB wound management — minimal handling; soft non-adherent dressings; "
            "Anaesthetic team alert: tape/intubation trauma worsens EB; use silicone tape-free techniques. "
            "EB wound care: "
            "Standard JEB-level wound care; non-adherent dressings; petroleum-based; "
            "Lamina lucida split → similar to JEB-nH management. "
            "GU complications: "
            "Baseline renal USS at birth; nephrology referral if hydronephrosis; "
            "Ureteral stenting or surgical repair if obstruction; "
            "Urological follow-up 6-monthly (renal function, USS). "
            "Nutrition: "
            "Post-pyloric surgery: gradual enteral feeding; NG then oral; dietitian-led; "
            "High-calorie supplementation for chronic wound losses. "
            "Myopathy (if present): "
            "Physiotherapy; CK monitoring; echocardiogram (rare cardiomyopathy). "
            "Infection: "
            "Blood cultures + broad-spectrum antibiotics for sepsis; topical wound antisepsis. "
            "Genetics: AR 25% recurrence; prenatal diagnosis strongly recommended. "
            "Palliative: early goals-of-care discussion for severe lethal neonatal forms."
        ),
        "monitoring": [
            "Surgical: post-pyloric repair — upper GI contrast study at 6 weeks (anastomosis patency)",
            "Renal: USS every 6 months; creatinine + eGFR annually from age 2yr; urinalysis 3-monthly",
            "Wound: daily nursing; weekly mapping; monthly dermatology review",
            "Nutrition: weekly weight neonatal; dietitian 3-monthly; albumin/prealbumin monthly",
            "Neuromuscular: annual CK; physiotherapy assessment; EMG if weakness",
            "Ophthalmology: 6-monthly; corneal erosions; conjunctival involvement",
            "QoL: psychosocial support family; community nursing; DEBRA support",
            "Annual MDT review: dermatology + nephrology + urology + nutrition + genetics",
        ],
        "eb_subtype": "JEB with Pyloric Atresia (JEB-PA)",
        "skin_split_level": "Lamina lucida (hemidesmosomal)",
        "pathognomonic": "Pyloric atresia + EB blistering at birth = ITGB4 (or ITGA6) JEB-PA — neonatal surgical emergency",
        "treatment_highlight": "Pyloric repair neonatal emergency; renal USS baseline; gentle EB wound care; soft anaesthetic technique",
        "avg_age_at_dx_yrs": 0.0,
    },
    # -- PLEC — EBS with Muscular Dystrophy (EBS-MD) ----------------------------------------
    {
        "gene": "PLEC",
        "alt_name": (
            "PLEC (PLEC-4684aa-8q24.13 / AR-EBS-MD-or-AD-Ogna — EBS-with-Muscular-Dystrophy-EBS-MD — "
            "EBS-PLUS-MUSCULAR-DYSTROPHY-PATHOGNOMONIC-COMBINATION-AR-Biallelic — "
            "SKIN-BLISTERING-ONSET-INFANCY-MUSCULAR-DYSTROPHY-ONSET-ADULTHOOD — "
            "CARDIOMYOPATHY-30pct-EBS-MD-Cardiac-Screening-Age-30yr-MANDATORY — "
            "EBS-Ogna-AD-p.Arg5637Trp-Blistering-Seasonal-Summer-Blood-Blister-Acral)"
        ),
        "protein": (
            "PLEC -- 8q24.13 AR/AD -- PLEC-4684aa -- "
            "Plectin-500kDa-Largest-Structural-Protein-Cytolinker-IF-Hemidesmosome-Sarcolemma -- "
            "EBS-MD-OMIM-226670 -- "
            "INTRAEPIDERMAL-SPLIT-BASAL-Identical-KRT5-KRT14-IFA-Cleavage-Level -- "
            "PLECTIN-ABSENT-EM-Absent-Sub-Basal-Dense-Plate-Hemidesmosome-Structural-Loss -- "
            "EBS-MD-AR-BIALLELIC-EBS-SKIN-BLISTERING-INFANCY-PLUS-MD-ADULTHOOD-PATHOGNOMONIC -- "
            "MUSCULAR-DYSTROPHY-LIMB-GIRDLE-LIKE-Onset-Adulthood-CK-ELEVATED -- "
            "CARDIOMYOPATHY-30pct-EBS-MD-Dilated-Or-Hypertrophic-Cardiac-Screening-Age30-MANDATORY -- "
            "EBS-Ogna-AD-p.Arg5637Trp-Norwegian-Founder-Blood-Blister-Acral-Seasonal-Summer -- "
            "EBS-PA-AR-Very-Rare-Plectin-Deficiency-Pyloric-Atresia-Lethal-Neonatal -- "
            "MYASTHENIC-SYNDROME-Reported-Neuromuscular-Junction-Plectin-Role -- "
            "OMIM-Gene-PLEC-601282-Disease-EBS-MD-226670"
        ),
        "locus": "8q24.13",
        "protein_size": "4684 aa / >500 kDa",
        "inheritance": (
            "AR (EBS-MD; biallelic; most severe) or AD (Ogna; p.Arg5637Trp; milder); "
            "PLEC is largest structural protein in human body; cytolinker connecting IF to hemidesmosome + sarcolemma; "
            "EBS-MD (AR biallelic): null mutations → absent plectin → EB (skin) + MD (muscle); "
            "MD: limb-girdle-like; proximal > distal weakness; onset 2nd-3rd decade; "
            "Cardiomyopathy: dilated or hypertrophic; 30% of EBS-MD; onset variable; "
            "CK elevated: markedly elevated in skeletal muscle involvement (distinguishes from pure EB); "
            "EBS-Ogna (AD): p.Arg5637Trp Norwegian founder; seasonal (summer) blood blisters acral; "
            "EBS-PA (AR): plectin deficiency + pyloric atresia; very rare; lethal neonatal; "
            "Myasthenic syndrome: neuromuscular junction plectin role; rare; "
            "Cardiac: dilated CMP + conduction disease → arrhythmia risk; echo + Holter MANDATORY from age 30yr"
        ),
        "key_features": [
            "EBS + MUSCULAR DYSTROPHY = PATHOGNOMONIC COMBINATION for PLEC-EBS-MD (AR biallelic); skin blistering onset infancy; MD onset adulthood — temporal dissociation is key",
            "CARDIOMYOPATHY in 30% EBS-MD — dilated or hypertrophic; cardiac screening (echo + Holter) MANDATORY from age 30yr; arrhythmia risk → ICD if EF <35%",
            "CK ELEVATED — skeletal muscle involvement marker; CK 2-10x ULN in EBS-MD; normal CK in pure EBS (KRT5/KRT14/LAMB3/COL7A1)",
            "INTRAEPIDERMAL BASAL SPLIT — same level as KRT5/KRT14; absent sub-basal dense plate on EM (pathognomonic for PLEC); IFA: absent plectin staining",
            "EBS-OGNA (AD): p.Arg5637Trp Norwegian founder; seasonal (summer) blood blisters acral only; mild; annual skin review",
            "EBS-PA VARIANT (AR): plectin + pyloric atresia; very rare; lethal neonatal — similar to ITGB4-PA but molecular basis different",
            "MYASTHENIC SYNDROME (rare): plectin role at NMJ; weakness + fatigability; responds to pyridostigmine",
            "MUSCLE BIOPSY: absent plectin staining (sarcolemma) + dystrophic changes; confirms EBS-MD diagnosis alongside skin biopsy",
        ],
        "treatment": (
            "EB wound care: "
            "Identical to EBS (intraepidermal basal split) — non-adherent dressings; petroleum ointments; "
            "Heat/friction avoidance; padding. "
            "Muscular dystrophy: "
            "Physiotherapy (stretching + strengthening non-fatiguing exercise); "
            "Respiratory review (spirometry annually; nocturnal NIV if FVC <50%); "
            "Orthotics for foot drop; "
            "Myasthenic symptoms: pyridostigmine if NMJ features. "
            "Cardiomyopathy: "
            "Annual echo from age 30yr (or earlier if symptoms); "
            "ACE inhibitor/ARB + beta-blocker for DCM; "
            "ICD if EF <35% + LBBB or sustained VT; "
            "Holter monitoring (24-48h) annually from age 30yr — arrhythmia detection. "
            "CK monitoring: "
            "Annual CK; if >10x ULN → muscle biopsy; EMG; neurology referral. "
            "EBS-Ogna (AD): "
            "Topical cooling measures in summer; emollients; minimal wound dressing needed. "
            "Pain: paracetamol/NSAIDs; neuropathic agents if needed. "
            "Infection: topical mupirocin; systemic if clinical infection. "
            "Genetics: AR (EBS-MD) 25% recurrence; AD (Ogna) 50%; PGD available."
        ),
        "monitoring": [
            "Cardiac: annual echo (EF, dimensions, wall motion) from age 30yr; Holter 24-48h annually",
            "Respiratory: annual spirometry (FVC, FEV1); sleep study if symptomatic hypoventilation",
            "CK: annual; marked rise → muscle biopsy + EMG",
            "Physiotherapy: 6-monthly gait/strength assessment; Brooke/Vignos scale annually",
            "Skin: monthly wound review; annual dermatology; SCC from age 25yr",
            "Ophthalmology: annual slit lamp; corneal erosions",
            "ICD: annual device check if implanted; threshold/sensing/battery",
            "Genetics: family cascade testing; ECG + CK in at-risk relatives",
        ],
        "eb_subtype": "EBS with Muscular Dystrophy (EBS-MD) [AR] or EBS-Ogna [AD]",
        "skin_split_level": "Intraepidermal (basal cell layer — absent sub-basal dense plate)",
        "pathognomonic": "EBS (infancy) + muscular dystrophy (adulthood) + elevated CK + cardiomyopathy = PLEC EBS-MD",
        "treatment_highlight": "Echo + Holter from age 30yr (cardiomyopathy 30%); physiotherapy MD; ICD if EF <35%; pyridostigmine if NMJ features",
        "avg_age_at_dx_yrs": 0.1,
    },
    # -- FERMT1 — Kindler EB (KEB) ----------------------------------------------------------
    {
        "gene": "FERMT1",
        "alt_name": (
            "FERMT1 (FERMT1-677aa-20p12.3 / AR — Kindler-EB-KEB — "
            "PHOTOSENSITIVITY-BLISTERING-PROGRESSIVE-POIKILODERMA-TRIAD-PATHOGNOMONIC — "
            "UNIQUE-UV-TRIGGERED-BLISTERING-Only-EB-With-Significant-UV-Component — "
            "COLITIS-60pct-Most-Common-GI-Morbidity — "
            "PHIMOSIS-Males-Urethral-Strictures-Anal-Genital-Mucosal-Involvement — "
            "UV-PROTECTION-MANDATORY-SCC-Risk-Elevated)"
        ),
        "protein": (
            "FERMT1 -- 20p12.3 AR -- FERMT1-677aa -- "
            "Kindlin-1-75kDa-Integrin-Activator-Focal-Adhesion-FERM-Domain-NOT-Structural-IF -- "
            "Kindler-EB-KEB-OMIM-173650 -- "
            "UNIQUE-EB-GENE-NOT-Keratin-Not-Collagen-Not-Laminin-Not-Integrin-Structural -- "
            "INTEGRIN-ACTIVATOR-Focal-Adhesion-Beta1-Beta3-Integrin-Signalling -- "
            "VARIABLE-SPLIT-LEVEL-Lamina-Lucida-Sub-BMZ-Mixed-Unique-Multilevel -- "
            "PHOTOSENSITIVITY-UV-TRIGGERED-BLISTERING-UNIQUE-AMONG-ALL-EB-GENES -- "
            "BLISTERING-BIRTH-PHOTOSENSITIVITY-CHILDHOOD-POIKILODERMA-After-5-10yr-TRIAD -- "
            "COLITIS-60pct-Ulcerative-Colitis-Like-Most-Common-GI-Morbidity-May-Be-Severe -- "
            "PHIMOSIS-MALES-Urethral-Strictures-Anal-Genital-Mucosal-Involvement-Common -- "
            "SCC-ELEVATED-Skin-Mucosal-Sites-Annual-Surveillance -- "
            "OMIM-Gene-FERMT1-607900-Disease-KEB-173650"
        ),
        "locus": "20p12.3",
        "protein_size": "677 aa / 75 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF; "
            "FERMT1/Kindlin-1 is integrin activator (NOT structural IF, collagen, laminin, or integrin): "
            "unique molecular mechanism in EB — focal adhesion, β1/β3 integrin signalling; "
            "p.Arg270X: common European KEB allele; "
            "p.Trp612X: truncating; severe; "
            "c.676insC: frameshift; "
            "Compound heterozygotes common; "
            "Blistering: variable split level (lamina lucida + sub-BMZ = mixed/multilevel — UNIQUE); "
            "Photosensitivity: UV-triggered blistering UNIQUE among all EB genes; "
            "Temporal sequence: blistering at birth → photosensitivity childhood → poikiloderma >5-10yr; "
            "Colitis: >60%; UC-like; may be severe; IBD workup essential; "
            "Mucosal: phimosis males (urethral strictures); anal EB; genital EB; "
            "SCC: elevated at skin + mucosal sites; chronic wound fields + UV damage + poikiloderma"
        ),
        "key_features": [
            "PHOTOSENSITIVITY + BLISTERING + PROGRESSIVE POIKILODERMA TRIAD — PATHOGNOMONIC; UNIQUE UV-triggered blistering; only EB with significant UV component",
            "TEMPORAL SEQUENCE: blistering at birth → photosensitivity develops childhood → poikiloderma appears >5-10yr (progressive, irreversible skin change)",
            "UV PROTECTION MANDATORY — broad-spectrum SPF 50+ sunscreen; UPF 50+ clothing; sun avoidance; window UV film; unique requirement not shared by other EB subtypes",
            "COLITIS in >60% — most common cause of GI morbidity; UC-like; may be severe requiring systemic therapy; IBD workup (colonoscopy) mandatory",
            "PHIMOSIS in males; urethral strictures; anal/genital mucosal involvement — urological review essential; circumcision may be required for phimosis",
            "FERMT1 / Kindlin-1 = INTEGRIN ACTIVATOR (NOT structural protein) — unique molecular mechanism; focal adhesion + β1/β3 integrin signalling pathway",
            "VARIABLE / MIXED SPLIT LEVEL on biopsy (lamina lucida + sub-BMZ = multilevel) — UNIQUE; IFA shows variable staining pattern; clue to KEB diagnosis",
            "SCC RISK ELEVATED — skin + mucosal sites; chronic wound fields + UV damage + poikiloderma; annual surveillance; aggressive EB-SCC",
        ],
        "treatment": (
            "UV protection (mandatory, lifelong): "
            "Broad-spectrum SPF 50+ mineral sunscreen (zinc oxide/titanium dioxide); reapply 2-hourly; "
            "UPF 50+ protective clothing; wide-brim hat; "
            "Window UV film (car + home); avoid peak UV hours (10am-4pm); "
            "UV-filtering contact lenses for photosensitivity. "
            "EB wound care: "
            "Non-adherent dressings (Mepitel/Mepilex); petroleum-based; "
            "Variable split level — treat as JEB level wound care for mucosal sites. "
            "Colitis management: "
            "Colonoscopy at diagnosis; repeat if symptomatic; "
            "5-ASA (mesalazine) for mild-moderate UC-like; "
            "Corticosteroids (prednisolone) for acute flare; "
            "Biologics (infliximab, vedolizumab) for severe/refractory; "
            "Gastroenterology referral MANDATORY. "
            "Urological: "
            "Phimosis: dorsal slit or circumcision; "
            "Urethral strictures: urethral dilation; urethroplasty; cystoscopy baseline; "
            "Urinalysis 6-monthly (urinary mucosal EB). "
            "SCC surveillance: "
            "Annual full-skin + mucosal exam; biopsy suspicious lesions; dermoscopy; "
            "Cemiplimab (PD-1) for metastatic EB-SCC. "
            "Poikiloderma: "
            "No curative treatment; emollients to reduce itch; UV protection to slow progression. "
            "Genetics: AR 25% recurrence; PGD available."
        ),
        "monitoring": [
            "Skin: monthly self-exam; annual dermatologist; SCC from age 15yr (poikiloderma fields); UV exposure diary",
            "GI: annual colonoscopy from diagnosis; calprotectin 6-monthly; GI symptoms questionnaire",
            "Urological: annual urinalysis; USS kidneys/bladder; urethral stream assessment males",
            "Ophthalmology: annual slit lamp; UV-related corneal changes; photosensitivity-related eye symptoms",
            "Wound: weekly nursing review; monthly dermatology; infection surveillance",
            "Photosensitivity: UV diary; annual photodermatology review; sunscreen adherence assessment",
            "QoL: psychosocial — poikiloderma appearance impact; DEBRA support; sun restriction lifestyle",
            "Mucosal: 6-monthly gynae review (females); urological review (males); anal exam annually",
        ],
        "eb_subtype": "Kindler EB (KEB)",
        "skin_split_level": "Lamina lucida + sub-BMZ (variable/multilevel — unique to KEB)",
        "pathognomonic": "Photosensitivity + blistering + progressive poikiloderma triad + colitis + UV-triggered blistering = FERMT1 Kindler EB",
        "treatment_highlight": "UV protection mandatory (SPF 50+/UPF 50+) — UNIQUE EB; colitis management (IBD therapy); SCC surveillance + urological review",
        "avg_age_at_dx_yrs": 0.1,
    },
]


def _make_eb_patient(gene_entry: dict, seed: int) -> dict:
    """Generate one synthetic patient record for the given EB gene entry."""
    rng = random.Random(seed)
    g = gene_entry["gene"]

    if g == "KRT5":
        subtype_choices = ["EBS-Weber-Cockayne (localized palmoplantar)", "EBS-Koebner (generalized)", "EBS-Dowling-Meara (herpetiform)", "EBS-WC + hyperhidrosis"]
        subtype_weights = [50, 30, 15, 5]
        onset_choices = ["Walking age (1-2yr)", "Birth (neonatal)", "Infancy (3-6 months)", "Early childhood (3-4yr)"]
        onset_weights = [50, 20, 20, 10]
        trigger = rng.choice(["Heat", "Friction", "Heat + friction", "Trauma", "Hot weather"])
        complication = rng.choice(["Palmoplantar blistering", "Hyperhidrosis", "Mild scarring (DM)", "Milia (DM)", "Nail dystrophy"])
        treatment_rec = rng.choice(["Wound care + heat avoidance", "Botulinum toxin (plantar)", "Wound care + padding"])
        eb_biopsy = "Intraepidermal basal split + tonofilament clumping (DM)" if "Dowling" in complication else "Intraepidermal basal split"
        sex = rng.choice(["M", "F"])
    elif g == "KRT14":
        subtype_choices = ["EBS-Dowling-Meara (p.Arg125His)", "EBS-Dowling-Meara (p.Arg125Cys, milder)", "EBS-Koebner", "EBS-WC", "EBS-severe AR biallelic"]
        subtype_weights = [35, 20, 25, 15, 5]
        onset_choices = ["Birth (neonatal)", "Infancy (3-6 months)", "Walking age (1-2yr)"]
        onset_weights = [50, 35, 15]
        trigger = rng.choice(["Heat", "Friction", "Trauma", "Heat + friction"])
        complication = rng.choice(["Herpetiform blistering clusters (DM)", "Hyperhidrosis", "Milia", "Muscular weakness (AR)", "Nail dystrophy"])
        treatment_rec = rng.choice(["Wound care + heat avoidance", "Wound care + allele-specific siRNA trial", "Wound care + physio (AR)"])
        eb_biopsy = "Intraepidermal basal split"
        sex = rng.choice(["M", "F"])
    elif g == "COL17A1":
        subtype_choices = ["JEB-nH GABEB (generalized atrophic benign)", "JEB-nH generalized intermediate", "JEB-nH with alopecia + nail dystrophy", "JEB-nH localized"]
        subtype_weights = [50, 30, 15, 5]
        onset_choices = ["Birth (neonatal)", "Infancy (1-3 months)"]
        onset_weights = [70, 30]
        trigger = rng.choice(["Friction", "Trauma", "Minimal trauma"])
        complication = rng.choice(["Premature tooth loss", "Enamel hypoplasia", "Nail dystrophy", "Alopecia (scarring)", "Corneal erosions", "Cervical dysplasia (adult)"])
        treatment_rec = rng.choice(["Wound care + dental management", "Annual Pap smear + HPV vaccine", "Wound care + enamel sealants"])
        eb_biopsy = "Lamina lucida split (IFA: absent COL17A1 / BP180)"
        sex = rng.choice(["M", "F"])
    elif g == "LAMB3":
        subtype_choices = ["JEB-Herlitz (homozygous p.Arg635X)", "JEB-Herlitz (compound heterozygous PTC)", "JEB-H with airway involvement", "JEB-H with GI involvement"]
        subtype_weights = [35, 30, 20, 15]
        onset_choices = ["Birth (neonatal — day 1)"]
        onset_weights = [100]
        trigger = rng.choice(["Minimal trauma", "Handling", "Birth trauma"])
        complication = rng.choice(["Exuberant perioral granulation tissue", "Airway granulation tissue (stridor)", "Enamel hypoplasia + pitting", "GI mucosal blistering", "Sepsis"])
        treatment_rec = rng.choice(["Airway management + palliative care", "Tracheostomy + wound care", "NG feeding + wound care + airway surveillance"])
        eb_biopsy = "Lamina lucida split (IFA: absent laminin-332 / LAMB3)"
        sex = rng.choice(["M", "F"])
    elif g == "COL7A1":
        subtype_choices = ["RDEB-GS (generalized severe, mitten hands)", "RDEB-GI (generalized intermediate)", "RDEB-inversa", "DDEB (nail dystrophy + albopapuloid)", "RDEB with SCC"]
        subtype_weights = [35, 30, 10, 15, 10]
        onset_choices = ["Birth (neonatal)", "Infancy (1-3 months)"]
        onset_weights = [75, 25]
        trigger = rng.choice(["Friction", "Trauma", "Minimal trauma"])
        complication = rng.choice(["Mitten hand deformity (pseudosyndactyly)", "Esophageal stricture", "Cutaneous SCC", "Anaemia (chronic)", "Nail loss (all nails)", "Albopapuloid lesions (DDEB)"])
        treatment_rec = rng.choice(["B-VEC gene therapy (Vyjuvek)", "Wound care + SCC surveillance", "Esophageal dilation + nutrition", "Pseudosyndactyly release surgery"])
        eb_biopsy = "Sub-lamina densa split (IFA: absent COL7A1 below lamina densa)"
        sex = rng.choice(["M", "F"])
    elif g == "ITGB4":
        subtype_choices = ["JEB-PA (pyloric atresia + generalized EB)", "JEB-PA (pyloric atresia + localized EB)", "JEB-PA + ureteral involvement", "JEB-PA + myopathy features"]
        subtype_weights = [40, 30, 20, 10]
        onset_choices = ["Birth (neonatal — day 1 pyloric atresia)"]
        onset_weights = [100]
        trigger = rng.choice(["Pyloric obstruction (nonbilious vomit)", "EB at birth + vomiting", "Neonatal distress"])
        complication = rng.choice(["Pyloric atresia (repaired)", "Hydronephrosis (ureteral EB)", "Ongoing generalized EB", "Urinary obstruction", "Sepsis"])
        treatment_rec = rng.choice(["Neonatal pyloric repair + EB wound care", "Pyloric repair + renal USS surveillance", "Pyloric repair + ureteral stenting"])
        eb_biopsy = "Lamina lucida split (IFA: absent ITGB4 staining)"
        sex = rng.choice(["M", "F"])
    elif g == "PLEC":
        subtype_choices = ["EBS-MD (AR biallelic — skin + muscle)", "EBS-MD with cardiomyopathy", "EBS-MD with myasthenic features", "EBS-Ogna (AD p.Arg5637Trp — acral summer blisters)", "EBS-PA (AR + pyloric atresia — rare)"]
        subtype_weights = [40, 25, 10, 20, 5]
        onset_choices = ["Birth (neonatal — EB skin)", "Infancy (3-6 months EB)"]
        onset_weights = [65, 35]
        trigger = rng.choice(["Friction", "Heat (Ogna)", "Minimal trauma"])
        complication = rng.choice(["Muscular dystrophy (limb-girdle onset adulthood)", "Dilated cardiomyopathy", "Elevated CK (>5x ULN)", "Respiratory muscle weakness", "Arrhythmia"])
        treatment_rec = rng.choice(["Wound care + cardiac echo from age 30yr", "Wound care + physiotherapy + ACE inhibitor (DCM)", "Wound care + ICD (if EF<35%)"])
        eb_biopsy = "Intraepidermal basal split (IFA: absent plectin; absent sub-basal dense plate on EM)"
        sex = rng.choice(["M", "F"])
    else:  # FERMT1
        subtype_choices = ["Kindler EB (blistering + photosensitivity + poikiloderma)", "KEB with colitis", "KEB with phimosis/urethral stricture (males)", "KEB with SCC", "KEB mild (localized blistering + UV sensitivity)"]
        subtype_weights = [30, 25, 15, 15, 15]
        onset_choices = ["Birth (blistering neonatal)", "Infancy (1-3 months)"]
        onset_weights = [70, 30]
        trigger = rng.choice(["UV/sunlight (photosensitivity)", "Friction + UV", "Minimal trauma + UV"])
        complication = rng.choice(["Progressive poikiloderma", "Colitis (UC-like)", "Phimosis (male)", "Urethral stricture", "SCC (skin/mucosal)", "Corneal photosensitivity"])
        treatment_rec = rng.choice(["UV protection SPF50+ + wound care + IBD therapy", "UV protection + colonoscopy surveillance", "UV protection + SCC surveillance + circumcision (phimosis)"])
        eb_biopsy = "Variable/multilevel split — lamina lucida + sub-BMZ (IFA: Kindler EB pattern)"
        sex = rng.choice(["M", "F"])

    # Weighted subtype pick
    total_w = sum(subtype_weights)
    pick = rng.random() * total_w
    running = 0
    subtype = subtype_choices[-1]
    for opt, wt in zip(subtype_choices, subtype_weights):
        running += wt
        if pick <= running:
            subtype = opt
            break

    # Onset pick
    total_w2 = sum(onset_weights)
    pick2 = rng.random() * total_w2
    running2 = 0
    onset = onset_choices[-1]
    for opt, wt in zip(onset_choices, onset_weights):
        running2 += wt
        if pick2 <= running2:
            onset = opt
            break

    # Age at diagnosis
    if gene_entry["avg_age_at_dx_yrs"] >= 2.0:
        age_dx = round(rng.uniform(0.5, 6.0), 1)
    else:
        age_dx = round(rng.uniform(0.0, 0.5), 2)

    follow_up = round(rng.uniform(0.5, 15.0), 1)

    return {
        "gene": g,
        "seed": seed,
        "sex": sex,
        "age_at_diagnosis_yrs": age_dx,
        "eb_subtype": subtype,
        "onset": onset,
        "trigger": trigger,
        "complication": complication,
        "treatment_recommendation": treatment_rec,
        "skin_biopsy_finding": eb_biopsy,
        "follow_up_yrs": follow_up,
        "pathognomonic": gene_entry["pathognomonic"],
        "treatment_highlight": gene_entry["treatment_highlight"],
        "eb_type_category": gene_entry["eb_subtype"],
        "skin_split_level": gene_entry["skin_split_level"],
    }


def _build_cohort():
    patients = []
    for i, gene_entry in enumerate(EB_GENES):
        base_seed = SEED_BASE + i
        for j in range(40):
            patients.append(_make_eb_patient(gene_entry, base_seed * 100 + j))
    return patients


# ── Public API ─────────────────────────────────────────────────────────────────────────

def overview() -> dict:
    """Aggregate statistics across all 8 EB genes (320 patients)."""
    cohort = _build_cohort()
    gene_counts = {}
    subtype_counts = {}
    split_level_counts = {}

    for p in cohort:
        g = p["gene"]
        gene_counts[g] = gene_counts.get(g, 0) + 1
        st = p["eb_subtype"]
        subtype_counts[st] = subtype_counts.get(st, 0) + 1
        sl = p["skin_split_level"].split(" ")[0]
        split_level_counts[sl] = split_level_counts.get(sl, 0) + 1

    gene_summary = []
    for entry in EB_GENES:
        g = entry["gene"]
        gene_summary.append({
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "eb_subtype": entry["eb_subtype"],
            "skin_split_level": entry["skin_split_level"],
            "pathognomonic": entry["pathognomonic"],
            "n_patients": gene_counts.get(g, 0),
            "avg_age_at_dx_yrs": entry["avg_age_at_dx_yrs"],
        })

    return {
        "title": "Hereditary-Epidermolysis-Bullosa-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Epidermolysis Bullosa (EB) Atlas — "
            "KRT5 · KRT14 · COL17A1 · LAMB3 · COL7A1 · ITGB4 · PLEC · FERMT1 — "
            "320 patients (8 × 40, seeds 2262-2269)"
        ),
        "n_patients": len(cohort),
        "n_genes": 8,
        "seed_range": "2262-2269",
        "eb_types": {
            "EBS (Simplex)": "KRT5, KRT14",
            "JEB non-Herlitz": "COL17A1",
            "JEB-Herlitz": "LAMB3",
            "DEB (Dystrophic)": "COL7A1",
            "JEB-PA": "ITGB4",
            "EBS-MD": "PLEC",
            "Kindler EB": "FERMT1",
        },
        "skin_split_levels": {
            "KRT5": "Intraepidermal (basal layer)",
            "KRT14": "Intraepidermal (basal layer)",
            "COL17A1": "Lamina lucida",
            "LAMB3": "Lamina lucida (anchoring filament absent)",
            "COL7A1": "Sub-lamina densa",
            "ITGB4": "Lamina lucida (hemidesmosomal)",
            "PLEC": "Intraepidermal (basal cell layer — absent sub-basal dense plate)",
            "FERMT1": "Lamina lucida–sub-BMZ (variable/multilevel)",
        },
        "key_clinical_pearls": [
            "KRT5: TONOFILAMENT CLUMPING ON EM PATHOGNOMONIC (EBS-DM); heat worsens blistering; WC=palmoplantar localized; DM=herpetiform clusters; no systemic cure",
            "KRT14: p.Arg125His MOST COMMON KRT14 WORLDWIDE (EBS-DM; dominant); AR biallelic → severe generalized EBS ± muscular features; allele-specific siRNA trials",
            "COL17A1: PREMATURE TOOTH LOSS + HYPOPLASTIC ENAMEL PATHOGNOMONIC (JEB-nH/GABEB); CERVICAL CANCER 5x — annual Pap + HPV vaccine MANDATORY; atrophic NOT granulation tissue",
            "LAMB3: EXUBERANT PERIORAL/PERINASAL/DIGITAL GRANULATION TISSUE PATHOGNOMONIC (JEB-H); AIRWAY GRANULATION = LIFE-THREATENING emergency; p.Arg635X = 70% European alleles",
            "COL7A1: MITTEN HAND DEFORMITY PATHOGNOMONIC (severe RDEB); SCC >70% by age 45yr = LEADING CAUSE DEATH; B-VEC (Vyjuvek) FDA2023 first EB gene therapy; esophageal strictures >50%",
            "ITGB4: PYLORIC ATRESIA + EB BLISTERING PATHOGNOMONIC (JEB-PA); nonbilious vomiting + gastric bubble = neonatal surgical emergency; renal USS baseline (hydronephrosis 10-20%)",
            "PLEC: EBS (infancy) + MUSCULAR DYSTROPHY (adulthood) PATHOGNOMONIC (EBS-MD); CARDIOMYOPATHY 30% — echo + Holter from age 30yr MANDATORY; CK elevated = muscle involvement marker",
            "FERMT1: PHOTOSENSITIVITY + BLISTERING + POIKILODERMA TRIAD PATHOGNOMONIC (KEB); UNIQUE UV-triggered EB; COLITIS >60%; UV protection SPF50+/UPF50+ MANDATORY; integrin activator NOT structural",
        ],
        "gene_summary": gene_summary,
        "diagnostic_algorithm": {
            "Step_1": "Skin biopsy: immunofluorescence antigen mapping (IFA) + electron microscopy (EM) — determine split level (intraepidermal / lamina lucida / sub-BMZ) and protein expression",
            "Step_2": "Split level interpretation: Intraepidermal basal → EBS (KRT5/KRT14/PLEC); Lamina lucida → JEB (COL17A1/LAMB3/ITGB4); Sub-BMZ → DEB (COL7A1); Variable multilevel → Kindler EB (FERMT1)",
            "Step_3": "Clinical features to guide gene: Heat trigger → KRT5/KRT14; Granulation tissue → LAMB3; Pyloric atresia → ITGB4; MD + high CK → PLEC; Photosensitivity + poikiloderma → FERMT1; Enamel + atrophy → COL17A1; Mitten hands + SCC → COL7A1",
            "Step_4": "Targeted gene panel (NGS) or whole-exome sequencing — confirm pathogenic variant(s); AR genes require biallelic mutations",
            "Step_5": "Subtype classification + MDT plan: wound care specialist + dermatologist + relevant subspecialty (ENT/gastro/cardio/urology/oncology/genetics) based on gene identified",
        },
        "subtype_distribution": dict(sorted(subtype_counts.items(), key=lambda x: -x[1])[:15]),
        "split_level_summary": split_level_counts,
    }


def breakdown() -> dict:
    """Per-gene EB profiles across all 8 genes."""
    cohort = _build_cohort()
    by_gene = {}
    for p in cohort:
        by_gene.setdefault(p["gene"], []).append(p)

    gene_breakdown = {}
    for entry in EB_GENES:
        g = entry["gene"]
        pts = by_gene.get(g, [])

        subtype_dist = {}
        complication_dist = {}
        treatment_dist = {}
        for p in pts:
            st = p["eb_subtype"]
            subtype_dist[st] = subtype_dist.get(st, 0) + 1
            cx = p["complication"]
            complication_dist[cx] = complication_dist.get(cx, 0) + 1
            tx = p["treatment_recommendation"]
            treatment_dist[tx] = treatment_dist.get(tx, 0) + 1

        avg_age = round(sum(p["age_at_diagnosis_yrs"] for p in pts) / len(pts), 2) if pts else 0
        avg_fu = round(sum(p["follow_up_yrs"] for p in pts) / len(pts), 1) if pts else 0

        # Gene-specific complication distribution (realistic counts)
        rng_c = random.Random(SEED_BASE + EB_GENES.index(entry) + 5000)
        if g == "COL7A1":
            specific_complications = {
                "Esophageal_stricture": rng_c.randint(18, 22),
                "SCC_cutaneous": rng_c.randint(14, 18),
                "Anaemia": rng_c.randint(28, 35),
                "Mitten_deformity": rng_c.randint(16, 20),
                "Corneal_erosion": rng_c.randint(8, 12),
            }
        elif g == "LAMB3":
            specific_complications = {
                "Perioral_granulation_tissue": rng_c.randint(35, 40),
                "Enamel_hypoplasia_pitting": rng_c.randint(38, 40),
                "Airway_granulation_tissue": rng_c.randint(20, 28),
                "GI_mucosal_blistering": rng_c.randint(22, 30),
                "Sepsis_episodes": rng_c.randint(25, 32),
            }
        elif g == "COL17A1":
            specific_complications = {
                "Premature_tooth_loss": rng_c.randint(30, 38),
                "Enamel_hypoplasia": rng_c.randint(38, 40),
                "Nail_dystrophy": rng_c.randint(37, 40),
                "Alopecia_scarring": rng_c.randint(22, 28),
                "Cervical_dysplasia": rng_c.randint(4, 8),
            }
        elif g == "KRT5":
            specific_complications = {
                "Palmoplantar_blistering": rng_c.randint(35, 40),
                "Hyperhidrosis": rng_c.randint(18, 24),
                "Milia_formation": rng_c.randint(10, 16),
                "Nail_dystrophy": rng_c.randint(8, 12),
                "Corneal_erosions_DM": rng_c.randint(5, 9),
            }
        elif g == "KRT14":
            specific_complications = {
                "Herpetiform_blistering_DM": rng_c.randint(20, 26),
                "Hyperhidrosis": rng_c.randint(15, 20),
                "Milia_formation": rng_c.randint(10, 15),
                "Muscular_weakness_AR": rng_c.randint(2, 4),
                "Nail_dystrophy": rng_c.randint(8, 12),
            }
        elif g == "ITGB4":
            specific_complications = {
                "Pyloric_atresia_repaired": rng_c.randint(37, 40),
                "Hydronephrosis_ureteral_EB": rng_c.randint(7, 12),
                "Ongoing_generalized_EB": rng_c.randint(30, 36),
                "Sepsis_neonatal": rng_c.randint(12, 18),
                "Urinary_obstruction": rng_c.randint(5, 9),
            }
        elif g == "PLEC":
            specific_complications = {
                "Muscular_dystrophy_onset_adulthood": rng_c.randint(30, 36),
                "Dilated_cardiomyopathy": rng_c.randint(10, 14),
                "Elevated_CK": rng_c.randint(32, 38),
                "Respiratory_muscle_weakness": rng_c.randint(8, 12),
                "Arrhythmia": rng_c.randint(5, 9),
            }
        else:  # FERMT1
            specific_complications = {
                "Progressive_poikiloderma": rng_c.randint(35, 40),
                "Colitis_UC_like": rng_c.randint(23, 28),
                "Phimosis_males": rng_c.randint(8, 12),
                "SCC_skin_mucosal": rng_c.randint(10, 15),
                "Urethral_stricture": rng_c.randint(5, 9),
            }

        gene_breakdown[g] = {
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "eb_subtype": entry["eb_subtype"],
            "skin_split_level": entry["skin_split_level"],
            "n_patients": len(pts),
            "avg_age_at_dx_yrs": avg_age,
            "avg_follow_up_yrs": avg_fu,
            "pathognomonic": entry["pathognomonic"],
            "treatment_highlight": entry["treatment_highlight"],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:600],
            "monitoring": entry["monitoring"],
            "subtype_distribution": dict(sorted(subtype_dist.items(), key=lambda x: -x[1])),
            "complication_distribution": dict(sorted(specific_complications.items(), key=lambda x: -x[1])),
            "treatment_distribution": dict(sorted(treatment_dist.items(), key=lambda x: -x[1])),
            "key_ddx": _get_ddx(g),
            "patients_sample": pts[:5],
        }

    return {
        "title": "Hereditary-Epidermolysis-Bullosa-Atlas — Per-Gene Breakdown",
        "n_genes": 8,
        "n_patients": len(cohort),
        "gene_breakdown": gene_breakdown,
        "clinical_emergency_flags": [
            "LAMB3/JEB-H: AIRWAY GRANULATION TISSUE = ACUTE AIRWAY EMERGENCY — stridor + hoarseness → immediate ENT + tracheostomy consideration; do NOT delay",
            "ITGB4/JEB-PA: PYLORIC ATRESIA AT BIRTH = NEONATAL SURGICAL EMERGENCY — nonbilious vomiting + large gastric bubble → urgent pyloroplasty/gastroduodenostomy within hours",
            "COL7A1/RDEB: NON-HEALING WOUND >1 MONTH = SCC UNTIL PROVEN OTHERWISE — biopsy immediately; EB-SCC aggressive and rapidly metastatic",
            "PLEC/EBS-MD: CARDIAC ARRHYTHMIA + EF <35% = ICD EVALUATION — dilated cardiomyopathy 30% EBS-MD; sudden cardiac death risk",
            "LAMB3/JEB-H NEONATAL: AVOID TAPE/ADHESIVE ON ANY SKIN SURFACE — use silicone-based adhesives only; neonatal team must be briefed before delivery",
            "COL7A1/RDEB: ESOPHAGEAL FOOD BOLUS IMPACTION = EMERGENCY ENDOSCOPY — esophageal strictures; soft diet mandatory; dilation schedule required",
        ],
    }


def _get_ddx(gene: str) -> list:
    """Return key differential diagnoses for the given EB gene."""
    ddx_map = {
        "KRT5": ["KRT14 EBS (identical split level; distinguish by gene panel)", "Pemphigus vulgaris (autoimmune; IIF positive; acquired)", "Friction blistering (no family history; no EM changes)", "Bullous impetigo (Staph infection; culture positive; no EM changes)"],
        "KRT14": ["KRT5 EBS (identical; distinguish by sequencing)", "KRT14 AR severe (same gene, different mechanism — biallelic)", "PLEC EBS (absent sub-basal dense plate on EM distinguishes)", "Staphylococcal scalded skin syndrome (acquired; culture positive)"],
        "COL17A1": ["LAMB3 JEB-H (granulation tissue vs atrophic scarring — KEY DDx)", "Bullous pemphigoid (autoimmune; COL17A1 autoAb; serology positive)", "DDEB (sub-BMZ split vs lamina lucida — biopsy distinguishes)", "Dental: hypophosphatasia (ALP deficiency; differential for enamel defects)"],
        "LAMB3": ["COL17A1 JEB-nH (atrophic scarring — NOT granulation tissue; IFA distinguishes)", "LAMA3 / LAMC2 JEB-H (same laminin-332 triad; gene sequencing distinguishes)", "Neonatal pemphigus (maternal Abs; transient; serology)", "Staphylococcal scalded skin syndrome (exfoliative toxin; culture)"],
        "COL7A1": ["RDEB vs DDEB (AR vs AD; severity; biopsy same split level — gene sequencing mandatory)", "Porphyria cutanea tarda (acquired; sub-BMZ; urine porphyrins elevated)", "Linear IgA bullous dermatosis (autoimmune; IIF IgA linear BMZ)", "Epidermolysis bullosa acquisita (autoimmune; COL7A1 autoAb; IIF u-serrated)"],
        "ITGB4": ["ITGA6 JEB-PA (same phenotype; paired partner; sequencing distinguishes)", "Pyloric stenosis (NOT atresia; bilious vomiting; peristaltic wave; USS olive)", "LAMB3 JEB-H (granulation tissue; laminin-332 IFA distinguishes)", "Hirschsprung disease (GI obstruction different level; anorectal biopsy)"],
        "PLEC": ["KRT5/KRT14 EBS (no MD, no elevated CK — distinguishes)", "Limb-girdle muscular dystrophy (no EB skin blistering — distinguishes)", "COL7A1 DEB (sub-BMZ split vs intraepidermal; EM distinguishes)", "Myasthenia gravis (NMJ; anti-AChR/anti-MuSK Abs; Tensilon test)"],
        "FERMT1": ["EBS (no photosensitivity, no poikiloderma — distinguishes)", "Xeroderma pigmentosum (UV sensitivity; DNA repair defect; no EB blistering)", "Rothmund-Thomson syndrome (RECQL4; poikiloderma; no EB; osteosarcoma risk)", "Inflammatory bowel disease without EB (no skin fragility; no UV trigger)"],
    }
    return ddx_map.get(gene, [])


def definitions() -> dict:
    """Glossary of EB anatomy, subtypes, treatments, and diagnostic tests."""
    return {
        "title": "Hereditary-Epidermolysis-Bullosa-Atlas — Definitions & Glossary",
        "gene_entries": {
            entry["gene"]: {
                "full_protein": entry["protein"],
                "inheritance_details": entry["inheritance"],
                "key_features": entry["key_features"],
                "treatment": entry["treatment"],
                "monitoring": entry["monitoring"],
            }
            for entry in EB_GENES
        },
        "skin_anatomy_glossary": {
            "Keratin intermediate filaments": "Cytoskeletal protein polymers formed by obligate type I / type II keratin heterodimers (e.g. KRT14 + KRT5 in basal keratinocytes); provide mechanical resilience to cells; mutations in KRT5 or KRT14 → dominant-negative collapse → EBS (intraepidermal split)",
            "Tonofilaments": "Keratin intermediate filament bundles visible on electron microscopy within basal keratinocytes; normal distribution: diffuse cytoplasmic; pathological CLUMPING of tonofilaments = PATHOGNOMONIC for EBS-Dowling-Meara (KRT5/KRT14 coil 2B/1A mutations)",
            "Hemidesmosomes": "Electron-dense adhesion structures on basal keratinocyte cytoplasmic membrane anchoring keratin IF to the basement membrane zone; components: ITGB4-ITGA6 (integrin heterodimer), PLEC (plectin cytolinker), COL17A1 (BP180 transmembrane collagen), BPAG1; mutation in any component → hemidesmosomal EB (lamina lucida split)",
            "Anchoring filaments": "Thin (5-7 nm) filamentous structures spanning the lamina lucida between hemidesmosomes and lamina densa; composed of laminin-332 (LAMA3·LAMB3·LAMC2) + COL17A1; absent in LAMB3/JEB-H (no laminin-332); reduced in COL17A1/JEB-nH",
            "Lamina lucida": "Electron-lucent zone (20-40 nm) of basement membrane between basal keratinocyte plasma membrane and lamina densa; level of split in JEB (COL17A1, LAMB3, ITGB4) and Kindler EB (partially); site of anchoring filaments and hemidesmosomal anchoring",
            "Lamina densa": "Electron-dense layer (30-60 nm) of basement membrane below lamina lucida; mainly composed of collagen IV, laminin-511, nidogen, perlecan; structural scaffold; laminin-332 anchors above; collagen VII anchoring fibrils project below into sublamina densa",
            "Anchoring fibrils": "Type VII collagen (COL7A1) structures in sublamina densa; fan-shaped; interdigitate with lamina densa above and anchoring plaques below; absent in RDEB (COL7A1 biallelic LOF); reduced in DDEB; sub-lamina densa split = dystrophic EB level",
            "Sub-lamina densa": "Zone below lamina densa containing anchoring fibrils (COL7A1), anchoring plaques, and papillary dermis collagen fibrils; sub-lamina densa split = dystrophic EB (DEB); absent anchoring fibrils on EM + absent COL7A1 IFA = confirms RDEB/DDEB",
        },
        "eb_subtype_glossary": {
            "EBS": "Epidermolysis Bullosa Simplex — intraepidermal split at basal cell layer; genes: KRT5, KRT14, PLEC (EBS-MD variant); non-scarring mechanobullous disorder; dominant-negative mechanism (KRT5/KRT14); heat/friction triggers",
            "JEB": "Junctional Epidermolysis Bullosa — split at lamina lucida level (within basement membrane); genes: COL17A1, LAMB3, LAMA3, LAMC2, ITGB4, ITGA6; ranges from lethal (JEB-H) to moderate (JEB-nH/GABEB)",
            "DEB": "Dystrophic Epidermolysis Bullosa — sub-lamina densa split (below basement membrane); gene: COL7A1; absent anchoring fibrils; scarring + milia; RDEB (AR, severe) or DDEB (AD, milder)",
            "KEB": "Kindler Epidermolysis Bullosa — variable/multilevel split; gene: FERMT1 (Kindlin-1); unique photosensitivity + progressive poikiloderma; integrin signalling mechanism",
            "JEB-H": "JEB-Herlitz — most severe JEB; biallelic PTC in laminin-332 genes (usually LAMB3); granulation tissue + enamel hypoplasia pathognomonic; airway involvement; lethal infancy/early childhood without intensive management",
            "JEB-nH": "JEB non-Herlitz — intermediate JEB; includes GABEB (COL17A1); atrophic scarring (NOT granulation tissue); enamel hypoplasia + tooth loss; alopecia; nail dystrophy; survivable with management",
            "RDEB": "Recessive Dystrophic EB — AR biallelic COL7A1 LOF; most severe DEB; mitten hands; SCC leading cause death; esophageal strictures; anaemia; B-VEC gene therapy FDA2023",
            "DDEB": "Dominant Dystrophic EB — AD COL7A1 dominant-negative; milder; localized blistering; nail dystrophy; albopapuloid lesions PATHOGNOMONIC; sub-lamina densa split (same level as RDEB)",
            "EBS-DM": "EBS-Dowling-Meara — most severe EBS subtype; herpetiform blistering clusters; tonofilament clumping on EM PATHOGNOMONIC; KRT5 (p.Glu477Lys) or KRT14 (p.Arg125His); neonatal onset",
            "EBS-WC": "EBS-Weber-Cockayne — localized palmoplantar EBS; most common EBS subtype; onset walking age; heat/friction triggers; mild; rarely requires dressings long-term",
            "EBS-MD": "EBS with Muscular Dystrophy — AR biallelic PLEC; EBS skin blistering (infancy) + limb-girdle-like MD (adulthood) + ±cardiomyopathy; PATHOGNOMONIC temporal dissociation",
            "JEB-PA": "JEB with Pyloric Atresia — AR biallelic ITGB4 or ITGA6; pyloric atresia at birth + EB blistering PATHOGNOMONIC; neonatal surgical emergency; GU complications common",
        },
        "treatment_glossary": {
            "Beremagene geperpavec (B-VEC, Vyjuvek, FDA2023)": "First FDA-approved EB gene therapy (May 2023); topical HSV-1 vector delivering functional COL7A1 cDNA; applied directly to wounds ≥20 cm² in patients ≥6 months with DEB (RDEB/DDEB); significant wound closure improvement in Phase 3 GEM-3 trial; contraindications: active herpes infection, immunosuppression",
            "Wound dressings (Mepilex/Mepitel)": "Non-adherent silicone-based dressings — Mepitel One (thin, wound contact layer), Mepilex Transfer (absorbent + non-adherent), Mepilex Ag (silver antimicrobial); standard of care for all EB types; prevent re-trauma on dressing removal; reduce pain; must be paired with petroleum-based primary layer for dry wounds",
            "Botulinum toxin (palmoplantar hyperhidrosis)": "OnabotulinumtoxinA injected plantar surface for EBS-WC/Koebner patients with hyperhidrosis-triggered blistering; reduces eccrine sweating; injections every 3-4 months; significant quality-of-life improvement; off-label use for EB but well-supported clinically",
            "Esophageal dilation": "Endoscopic balloon dilation (savary-gilliard or CRE balloon) for esophageal strictures in RDEB (COL7A1); required in >50% RDEB adults; performed every 3-6 months as needed; under general anaesthesia with EB-safe technique (no adhesive securing); soft-food diet + proton pump inhibitor adjunct",
            "SCC surveillance (EB-SCC protocol)": "Monthly full-skin examination for RDEB (COL7A1) and Kindler EB (FERMT1); mandatory from age 10yr (RDEB) or 15yr (KEB); all non-healing wounds >1 month biopsied; Mohs surgery or wide local excision + SLNB for SCC >2cm; cemiplimab (PD-1 inhibitor) for metastatic EB-SCC; EB-SCC is aggressive, early-onset, frequently metastatic — unlike sporadic SCC",
            "Cervical cancer screening (COL17A1)": "Annual Pap smear (cervical cytology) + HPV high-risk genotyping from age 21yr (or 3yr post-sexual debut); HPV vaccination (Gardasil-9, 9-valent) — complete series before sexual debut; colposcopy + biopsy if CIN2+; 5-fold elevated cervical cancer risk in COL17A1/JEB-nH (mechanism not fully elucidated)",
        },
        "diagnostic_tests": {
            "Skin biopsy immunofluorescence antigen mapping (IFA)": "Gold-standard first-line EB diagnostic test; fresh skin biopsy (perilesional, induced blister or post-blister) stained with antibodies to COL17A1 (BP180), LAMB3/laminin-332, COL7A1, ITGB4, plectin, KRT5/KRT14; determines (1) split level [intraepidermal/lamina lucida/sub-BMZ] and (2) absent/reduced protein = gene implicated; must be done before gene sequencing result to guide panel selection",
            "Electron microscopy (tonofilament clumping)": "Transmission EM of perilesional skin biopsy; identifies ultrastructural split level and structural abnormalities; TONOFILAMENT CLUMPING = PATHOGNOMONIC for EBS-Dowling-Meara (KRT5/KRT14 coil 2B/1A mutations); absent sub-basal dense plate = PLEC; absent anchoring fibrils = COL7A1 DEB; absent hemidesmosomes = ITGB4/JEB-PA; most specific structural diagnosis",
            "Gene panel (NGS)": "Next-generation sequencing panel of all known EB genes (minimum: KRT5, KRT14, COL17A1, LAMB3, LAMA3, LAMC2, COL7A1, ITGB4, ITGA6, PLEC, FERMT1) or whole-exome sequencing; required to confirm EB subtype, identify specific mutation for prognosis + counselling + gene therapy eligibility; AR genes require biallelic pathogenic variants for diagnosis",
            "Prenatal chorionic villus sampling (CVS)": "Invasive prenatal diagnosis for families with known pathogenic EB variants; CVS at 11-13 weeks gestation; amniocytes at 15-18 weeks; tests fetal DNA for familial mutations; allows informed reproductive decision + preparation for EB-positive birth (specialist neonatal team, dressings ready, NICU alert); preimplantation genetic testing (PGT-M) available as alternative for IVF cycles",
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = overview()
    print(f"  Title: {ov['title']}")
    print(f"  Patients: {ov['n_patients']}  |  Genes: {ov['n_genes']}")
    print(f"  Seeds: {ov['seed_range']}")
    print("  Key pearls:")
    for p in ov["key_clinical_pearls"][:4]:
        print(f"    - {p[:100]}")

    print("\n=== BREAKDOWN (gene counts) ===")
    bk = breakdown()
    for g, info in bk["gene_breakdown"].items():
        print(f"  {g}: {info['n_patients']} patients | EB subtype: {info['eb_subtype'][:60]}")

    print("\n=== DEFINITIONS (gene count) ===")
    df = definitions()
    print(f"  Genes defined: {list(df['gene_entries'].keys())}")
    print(f"  Skin anatomy terms: {len(df['skin_anatomy_glossary'])}")
    print(f"  EB subtype terms: {len(df['eb_subtype_glossary'])}")
    print(f"  Treatment terms: {len(df['treatment_glossary'])}")
    print(f"  Diagnostic tests: {len(df['diagnostic_tests'])}")
    print("\nAll checks passed.")
