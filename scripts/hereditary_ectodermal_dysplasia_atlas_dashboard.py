#!/usr/bin/env python3
"""Hereditary-Ectodermal-Dysplasia-Atlas — Complete 8-Gene Hereditary Ectodermal Dysplasia Atlas
(EDA · EDAR · WNT10A · TP63 · IKBKG · GJB6 · IRF6 · PVRL1).

EDA      (Ectodysplasin-A; 391 aa; Xq12; XLR;
          X-Linked Hypohidrotic Ectodermal Dysplasia (XLHED);
          ANHIDROSIS + SPARSE/ABSENT HAIR + HYPODONTIA TRIAD PATHOGNOMONIC;
          Anhidrosis = HEAT STROKE LIFE-THREATENING — cooling vest mandatory;
          Conical/absent teeth (avg 2-4 permanent teeth); saddle-nose; frontal bossing;
          EDX111 (intra-amniotic EDA1 protein injection 26-30wks) = first prenatal DMT;
          Sweat test (starch-iodine) confirms anhidrosis at diagnosis;
          seed SEED_BASE+0).
EDAR     (EDA Receptor; 448 aa; 2q13; AD/AR;
          Autosomal Hypohidrotic Ectodermal Dysplasia (HED);
          Same ANHIDROSIS+HYPOTRICHOSIS+HYPODONTIA TRIAD as XLHED;
          EDAR-D374A — hyperfunctional variant in >85% Han Chinese with straight, thick hair;
          AD (heterozygous hypomorphic) or AR (severe biallelic null);
          EDA/EDAR/EDARADD/NF-kB converging pathway;
          seed SEED_BASE+1).
WNT10A   (Wingless-Type MMTV Integration Site Family Member 10A; 417 aa; 2q35; AR;
          Odonto-Onycho-Dermal Dysplasia (OODD) / Schöpf-Schulz-Passarge syndrome (SSPS);
          SEVERE SELECTIVE OLIGODONTIA — 20-28 missing permanent teeth PATHOGNOMONIC;
          Most common identified cause of severe isolated oligodontia in Europeans (20%);
          Nail dystrophy + palmoplantar hyperkeratosis + dry hair; teeth MOST SEVERELY AFFECTED;
          SSPS adds eyelid cysts (hidrocystomas) + hypotrichosis;
          seed SEED_BASE+2).
TP63     (Tumor Protein p63; 680 aa; 3q28; AD;
          EEC syndrome: Ectrodactyly-Ectodermal Dysplasia-Clefting;
          EEC: SPLIT HAND/FOOT (ectrodactyly) + CLEFT LIP/PALATE + EDA TRIAD PATHOGNOMONIC;
          AEC/Hay-Wells syndrome (TP63 allelic): ANKYLOBLEPHARON FILIFORM ADNATUM AT BIRTH PATHOGNOMONIC;
          ADULT syndrome, Limb-Mammary syndrome, SHFM4 also allelic;
          p63 master regulator of ectoderm/stratified epithelium development;
          seed SEED_BASE+3).
IKBKG    (Inhibitor of NF-kB Kinase Regulatory Subunit Gamma / NEMO; 419 aa; Xq28; XLD;
          Incontinentia Pigmenti (IP) in females;
          4-STAGE SKIN: VESICULAR→VERRUCOUS→HYPERPIGMENTED→ATROPHIC WHORLED PATHOGNOMONIC;
          Females: IP (seizures + retinal detachment + dental/hair abnormalities);
          Males: usually LETHAL in utero (hemizygous null);
          Rare surviving males: EDA-ID (Ectodermal Dysplasia + Immunodeficiency — mycobacterial);
          RETINAL EMERGENCY: traction detachment → blind if not treated;
          seed SEED_BASE+4).
GJB6     (Gap Junction Protein Beta-6 / Connexin-30; 261 aa; 13q12.11; AD;
          Clouston syndrome / Hidrotic Ectodermal Dysplasia (HED);
          DIFFUSE PROGRESSIVE ALOPECIA + NAIL DYSTROPHY + PPK TRIAD PATHOGNOMONIC;
          NO ANHIDROSIS = KEY DDx from EDA/EDAR (which have anhidrosis as cardinal feature);
          Quebec French-Canadian founder: p.Gly11Arg (GJB6 del(GJB6-D13S1830) also DFNB1b);
          Sweating NORMAL — heat stroke not a risk; teeth mostly spared;
          seed SEED_BASE+5).
IRF6     (Interferon Regulatory Factor 6; 467 aa; 1q32.3; AD;
          Van der Woude syndrome (VWS) — most common syndromic cleft worldwide;
          LOWER LIP PITS (paramedian commissural pits) + CLEFT LIP/PALATE PATHOGNOMONIC;
          Lip pits UNIQUE — not seen in other cleft syndromes;
          Popliteal Pterygium syndrome (PPS) allelic: adds popliteal webbing + syndactyly;
          IRF6 accounts for 2-5% ALL cleft lip/palate; autosomal dominant;
          seed SEED_BASE+6).
PVRL1    (Poliovirus Receptor-Related 1 / Nectin-1; 517 aa; 11q23.3; AR;
          CLPED1 — Cleft Lip/Palate-Ectodermal Dysplasia type 1 / Zlotogora-Ogur syndrome;
          CLEFT LIP/PALATE + HYPODONTIA + NAIL DYSPLASIA + SPARSE HAIR PATHOGNOMONIC;
          Impaired corneal innervation → recurrent herpetic keratitis PATHOGNOMONIC;
          Mediterranean (Middle East/North Africa) founder populations;
          Nectin-1 cell adhesion molecule; herpes simplex virus receptor;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2294-2301).
"""

import random

SEED_BASE = 2294

ED_GENES = [
    # -- EDA — XLHED (X-Linked Hypohidrotic Ectodermal Dysplasia) -------------------------
    {
        "gene": "EDA",
        "alt_name": (
            "EDA (EDA-391aa-Xq12 / XLR — X-Linked-Hypohidrotic-Ectodermal-Dysplasia-XLHED — "
            "ANHIDROSIS+HYPOTRICHOSIS+HYPODONTIA-TRIAD-PATHOGNOMONIC — "
            "HEAT-STROKE-LIFE-THREATENING-Cooling-Vest-Mandatory — "
            "EDX111-Prenatal-Treatment-First-DMT)"
        ),
        "protein": (
            "EDA -- Xq12 XLR -- EDA-391aa -- "
            "Ectodysplasin-A-43kDa-TNF-Family-Ligand-Transmembrane-Cleaved-Secreted -- "
            "XLHED-OMIM-305100 -- "
            "EDA→EDAR→EDARADD→NF-kB-Pathway-Ectoderm-Appendage-Development -- "
            "ANHIDROSIS-Complete-No-Sweat-Glands-HEAT-STROKE-RISK -- "
            "HYPOTRICHOSIS-Sparse-Fine-Brittle-Hair-Eyebrows-Eyelashes-Absent -- "
            "HYPODONTIA-Conical-Widely-Spaced-2-4-Permanent-Teeth-Mean-PATHOGNOMONIC -- "
            "SADDLE-NOSE-Frontal-Bossing-Periorbital-Wrinkling-Facies -- "
            "SWEAT-TEST-Starch-Iodine-Pilocarpine-Iontophoresis-Confirms-Anhidrosis -- "
            "EDX111-EDA1-Protein-Intra-Amniotic-26-30wks-First-Prenatal-DMT-Approved -- "
            "OMIM-Gene-EDA-300451-Disease-XLHED-305100"
        ),
        "locus": "Xq12",
        "protein_size": "391 aa / 43 kDa",
        "inheritance": (
            "XLR (X-linked recessive); hemizygous males fully affected; heterozygous females: "
            "variable (usually mild: patchy sweating, some dental anomalies); "
            "de novo mutations ~30%; gene panel + carrier testing for female relatives; "
            "prenatal diagnosis: amniocentesis/CVS for EDA mutation + fetal sex"
        ),
        "ed_category": "Anhidrotic/Hypohidrotic ED — Classic XLHED (EDA pathway, complete triad)",
        "pathognomonic": (
            "ANHIDROSIS (zero sweat output) + SPARSE/ABSENT HAIR (hypotrichosis) + "
            "HYPODONTIA (conical/absent teeth, avg 2-4 permanent teeth) TRIAD = "
            "PATHOGNOMONIC for XLHED; saddle nose + frontal bossing + periorbital hyperpigmentation facies; "
            "starch-iodine sweat test confirms complete anhidrosis"
        ),
        "treatment": (
            "COOLING VEST MANDATORY from diagnosis — anhidrosis means ANY fever or hot environment "
            "= heat stroke risk; temperature monitoring app + fever management protocol at school; "
            "Dentures from age 2-3yr (primary teeth) → implant-supported prosthetics from adulthood; "
            "Saline nasal spray for mucous membrane dryness; lubricating eye drops; "
            "EDX111 (recombinant EDA1 protein) via intra-amniotic injection 26-30wks = "
            "first approved prenatal treatment — dramatically improves sweating in affected males if given in utero; "
            "Wig/hair prosthesis + eyebrow/eyelash microblading for cosmesis; "
            "Multidisciplinary team: genetics + dental + dermatology + ophthalmology"
        ),
        "key_features": [
            "ANHIDROSIS — complete absence of sweat glands — HEAT STROKE LIFE-THREATENING without cooling",
            "HYPOTRICHOSIS — sparse, fine, brittle hair; eyebrows/eyelashes absent or sparse",
            "HYPODONTIA — conical (peg-shaped), widely-spaced teeth; avg 2-4 permanent teeth retained; primary teeth often absent",
            "CHARACTERISTIC FACIES — saddle nose, frontal bossing, periorbital wrinkling, prominent supraorbital ridges",
            "MUCOUS MEMBRANE DRYNESS — eyes (recurrent conjunctivitis), nose, respiratory tract",
            "EDX111 prenatal treatment — intra-amniotic EDA1 protein injection at 26-30 weeks restores sweat glands in affected fetuses",
        ],
        "monitoring": [
            "Temperature monitoring continuous — cooling vest + fever protocol mandatory",
            "Dental panoramic X-ray from age 3yr — document missing/conical teeth for prosthetic planning",
            "Ophthalmology: lubricating drops + annual review for corneal complications",
            "Hearing test: sensorineural hearing loss reported in some XLHED patients",
            "Psychosocial support: appearance + social impact (school, peers) significant",
            "Carrier testing for maternal female relatives (heterozygous females may have patchy hypohidrosis)",
        ],
        "key_ddx": [
            "EDAR-autosomal HED (same triad, autosomal — check family history for male-to-male transmission/AR pattern)",
            "GJB6-Clouston syndrome (alopecia + nail dystrophy but NORMAL sweating — no heat stroke risk)",
            "CHARGE syndrome (coloboma+choanal atresia+growth retardation+genital/ear anomalies — broader spectrum)",
            "Hypodontia isolated (WNT10A most common gene — no anhidrosis or hair anomaly)",
        ],
        "cardiac_risk": False,
        "dental_risk": True,
        "heat_risk": True,
        "immunodeficiency_risk": False,
        "cleft_risk": False,
        "retinal_risk": False,
        "severity_options": ["Mild (some sweating, few dental anomalies)", "Moderate (partial anhidrosis, multiple missing teeth)", "Severe (complete anhidrosis, profound hypodontia, heat stroke history)"],
        "systemic_options": ["Anhidrosis", "Heat intolerance", "Hypodontia", "Hypotrichosis", "Mucous membrane dryness", "Recurrent respiratory infections"],
        "treatments_used": ["Cooling vest", "Saline nasal spray", "Dentures/implants", "Lubricating eye drops", "EDX111 prenatal", "Wig prosthesis"],
    },
    # -- EDAR — Autosomal HED ---------------------------------------------------------------
    {
        "gene": "EDAR",
        "alt_name": (
            "EDAR (EDAR-448aa-2q13 / AD-AR — Autosomal-Hypohidrotic-Ectodermal-Dysplasia-HED — "
            "Same-Triad-as-XLHED-Anhidrosis+Hypotrichosis+Hypodontia — "
            "EDAR-D374A-East-Asian-Thick-Straight-Hair-Variant — "
            "NF-kB-Pathway-EDA-EDAR-EDARADD-Converging)"
        ),
        "protein": (
            "EDAR -- 2q13 AD-AR -- EDAR-448aa -- "
            "EDA-Receptor-50kDa-TNFR-Family-Death-Domain-Transmembrane -- "
            "Autosomal-HED-OMIM-129490-AD-614941-AR -- "
            "EDAR→EDARADD→NF-kB-Hair-Sweat-Tooth-Gland-Development -- "
            "AD-Missense-Dominant-Negative-Hypomorphic-Mild-HED -- "
            "AR-Biallelic-Null-Severe-HED-Phenocopy-XLHED -- "
            "EDAR-D374A-Derived-30kya-East-Asia-Thick-Straight-Hair-85pct-Han-Chinese -- "
            "EDAR-D374A-Increased-Hair-Follicle-Density-Mammary-Gland-Branching -- "
            "SWEAT-GLANDS-Reduced-but-NOT-Absent-in-most-EDAR-AD -- "
            "OMIM-Gene-EDAR-604095-Disease-HED2-129490"
        ),
        "locus": "2q13",
        "protein_size": "448 aa / 50 kDa",
        "inheritance": (
            "AD (heterozygous hypomorphic) or AR (biallelic null — severe, phenocopy XLHED); "
            "EDAR-D374A: common hyperfunctional variant in East Asian populations (positive selection); "
            "de novo AD mutations possible; "
            "AR form: both parents carriers (25% recurrence); no sex bias unlike EDA"
        ),
        "ed_category": "Anhidrotic/Hypohidrotic ED — Autosomal HED (EDA pathway, same clinical triad as XLHED)",
        "pathognomonic": (
            "Same ANHIDROSIS + HYPOTRICHOSIS + HYPODONTIA TRIAD as XLHED but AUTOSOMAL pattern; "
            "AD form often milder (partial hypohidrosis, some teeth present); "
            "AR form = severe HED phenocopy of XLHED; "
            "EDAR-D374A in Han Chinese: thick straight scalp hair + increased mammary branching — population variant, not disease"
        ),
        "treatment": (
            "Same as XLHED: cooling measures + dental prosthetics + mucous membrane care; "
            "AD form often milder — partial sweating means lower but still present heat stroke risk; "
            "AR form: identical management to XLHED including cooling vest; "
            "EDX111 prenatal treatment active trials for EDAR-AR (same EDA1 pathway); "
            "No approved DMT currently beyond EDX111 trials; "
            "Genetic counselling: AD (50% risk) vs AR (25% risk) has major impact on family planning"
        ),
        "key_features": [
            "Autosomal inheritance — male-to-male transmission (AD) or consanguinity/carrier parents (AR)",
            "AD form typically milder than XLHED — partial hypohidrosis, fewer missing teeth",
            "AR biallelic: identical severity to XLHED — complete anhidrosis, profound hypodontia",
            "EDAR-D374A — East Asian population variant: thicker straight hair, no disease phenotype",
            "EDA/EDAR/EDARADD pathway — all three genes → same HED phenotype (molecular diagnosis needed)",
            "Sebaceous glands and mammary glands also reduced in severe EDAR deficiency",
        ],
        "monitoring": [
            "Same monitoring as XLHED: temperature, dental, ophthalmology",
            "Family cascade: test siblings (AR — 25% recurrence); test children (AD — 50% recurrence)",
            "Sweat testing (iontophoresis/starch-iodine): quantify residual sweating (AD often partial)",
            "Genetic panel covers EDA+EDAR+EDARADD — molecular distinction essential for recurrence risk",
        ],
        "key_ddx": [
            "EDA-XLHED (X-linked — hemizygous males severely affected; females carriers/mild; no male-to-male transmission)",
            "EDARADD-HED3 (same autosomal HED — molecular panel distinguishes; same management)",
            "WNT10A-OODD (dental most severely affected; sweating normal; no anhidrosis)",
            "Hypodontia isolated (WNT10A — absent sweat/hair anomalies distinguish from HED)",
        ],
        "cardiac_risk": False,
        "dental_risk": True,
        "heat_risk": True,
        "immunodeficiency_risk": False,
        "cleft_risk": False,
        "retinal_risk": False,
        "severity_options": ["Mild AD (partial hypohidrosis, oligodontia)", "Moderate AD (significant sweating reduction, multiple dental)", "Severe AR (complete anhidrosis — XLHED phenocopy)"],
        "systemic_options": ["Partial anhidrosis", "Hypodontia", "Hypotrichosis", "Reduced sebaceous glands", "Heat intolerance", "Mucous membrane dryness"],
        "treatments_used": ["Cooling measures", "Dental prosthetics", "EDX111 trial", "Lubricating drops", "Saline nasal spray"],
    },
    # -- WNT10A — OODD / severe isolated oligodontia ---------------------------------------
    {
        "gene": "WNT10A",
        "alt_name": (
            "WNT10A (WNT10A-417aa-2q35 / AR — Odonto-Onycho-Dermal-Dysplasia-OODD — "
            "SEVERE-SELECTIVE-OLIGODONTIA-20-28-Missing-Permanent-Teeth-PATHOGNOMONIC — "
            "Most-Common-Isolated-Oligodontia-Gene-Europe-20pct — "
            "Nail-Dystrophy+Palmoplantar-Hyperkeratosis+Dry-Hair)"
        ),
        "protein": (
            "WNT10A -- 2q35 AR -- WNT10A-417aa -- "
            "Wingless-Type-10A-Wnt-Ligand-46kDa-Frizzled-Pathway-Ectoderm -- "
            "OODD-OMIM-257980-Schoepf-Schulz-Passarge-SSPS-OMIM-224750 -- "
            "WNT10A-WNT-Pathway-LRP5-6-AXIN-GSK3-β-Catenin-Ectoderm-Appendage -- "
            "SELECTIVE-TOOTH-AGENESIS-All-Permanent-Teeth-Severely-Affected -- "
            "NAIL-DYSTROPHY-Thin-Koilonychia-Anonychia-Severe -- "
            "PALMOPLANTAR-HYPERKERATOSIS-Diffuse-PPK-Without-EDA-Triad -- "
            "DRY-STRAIGHT-HAIR-Not-Absent-Distinguishes-from-HED -- "
            "SWEATING-NORMAL-No-Heat-Stroke-Risk-KEY-DDx-HED -- "
            "SSPS-Adds-Eyelid-Hidrocystomas-Hypotrichosis-to-OODD -- "
            "OMIM-Gene-WNT10A-606268-Disease-OODD-257980"
        ),
        "locus": "2q35",
        "protein_size": "417 aa / 46 kDa",
        "inheritance": (
            "AR (biallelic loss-of-function); p.Phe228Ile most common European pathogenic variant; "
            "heterozygous carriers: mild oligodontia (1-2 missing teeth) possible; "
            "WNT10A accounts for ~20% of severe isolated oligodontia (hypodontia >6 missing teeth) in Europeans; "
            "consanguineous families: founders in some populations"
        ),
        "ed_category": "Hypodontia-predominant ED — WNT-pathway (teeth + nails + PPK; sweating NORMAL)",
        "pathognomonic": (
            "SEVERE SELECTIVE OLIGODONTIA (20-28 missing permanent teeth, ALL teeth types affected) PATHOGNOMONIC; "
            "nail dystrophy (thin, koilonychia, anonychia) + palmoplantar hyperkeratosis + dry hair; "
            "SWEATING COMPLETELY NORMAL = KEY DDx from EDA/EDAR-HED; "
            "panoramic X-ray shows near-complete absence of tooth germs"
        ),
        "treatment": (
            "Comprehensive dental rehabilitation mandatory from early childhood; "
            "Removable partial dentures → implant-supported fixed prosthetics at skeletal maturity (>16yr); "
            "Alveolar bone augmentation if multiple missing teeth cause atrophy; "
            "Nail care: emollients + protective footwear; "
            "Palmoplantar hyperkeratosis: urea 40% + keratolytic emollients; acitretin if severe; "
            "NORMAL SWEATING — cooling vest NOT required; heat stroke NOT a risk; "
            "Multidisciplinary: maxillofacial surgery + orthodontics + genetics + dermatology"
        ),
        "key_features": [
            "SEVERE SELECTIVE OLIGODONTIA — 20-28 permanent teeth missing; deciduous teeth often also affected",
            "NAIL DYSTROPHY — thin, brittle, koilonychia or complete anonychia (absent nails)",
            "PALMOPLANTAR HYPERKERATOSIS — diffuse PPK; no pseudoainhum or transgredient extension",
            "DRY, STRAIGHT HAIR — not absent (unlike HED); no hypotrichosis",
            "NORMAL SWEATING — heat stroke NOT a risk; distinguishes from EDA/EDAR-HED",
            "SSPS allelic: adds hidrocystomas of eyelids + palms + soles to OODD features",
        ],
        "monitoring": [
            "Dental panoramic X-ray from age 3yr — document tooth agenesis pattern",
            "Orthodontic + maxillofacial assessment from age 5yr for prosthetic planning",
            "Annual nail assessment — protective footwear + emollients",
            "Dermatology review for PPK: urea 40% cream compliance",
            "Genetic cascade: test siblings (25% recurrence AR); parents often carriers with mild oligodontia",
        ],
        "key_ddx": [
            "EDA/EDAR-HED (anhidrosis + hair anomaly — SWEATING IS NORMAL in WNT10A — key DDx)",
            "IRF6-Van der Woude (lip pits + cleft — NO lip pits in WNT10A-OODD)",
            "PAX9/MSX1 isolated oligodontia (molecular panel distinguishes — fewer teeth missing, specific pattern)",
            "Hypodontia isolated PAX9 (molars predominantly; less severe; no nail/skin changes)",
        ],
        "cardiac_risk": False,
        "dental_risk": True,
        "heat_risk": False,
        "immunodeficiency_risk": False,
        "cleft_risk": False,
        "retinal_risk": False,
        "severity_options": ["Moderate (6-12 missing permanent teeth, mild nail)", "Severe (13-20 missing, nail dystrophy + PPK)", "Profound OODD (20-28 missing, anonychia, diffuse PPK)"],
        "systemic_options": ["Severe oligodontia", "Nail dystrophy", "Palmoplantar hyperkeratosis", "Dry hair", "Eyelid hidrocystomas (SSPS)", "Sparse hair (SSPS)"],
        "treatments_used": ["Removable dentures", "Dental implants", "Bone augmentation", "Urea 40% PPK", "Acitretin severe PPK", "Nail emollients"],
    },
    # -- TP63 — EEC/AEC syndrome ------------------------------------------------------------
    {
        "gene": "TP63",
        "alt_name": (
            "TP63 (TP63-680aa-3q28 / AD — EEC-Syndrome-Ectrodactyly-Ectodermal-Dysplasia-Clefting — "
            "SPLIT-HAND-FOOT+CLEFT-LIP-PALATE+EDA-TRIAD-PATHOGNOMONIC — "
            "AEC-Hay-Wells-ANKYLOBLEPHARON-AT-BIRTH-PATHOGNOMONIC — "
            "p63-Master-Regulator-Stratified-Epithelium)"
        ),
        "protein": (
            "TP63 -- 3q28 AD -- TP63-680aa -- "
            "Tumor-Protein-p63-72kDa-p53-Family-TF-Stratified-Epithelium-Master-Regulator -- "
            "EEC3-OMIM-604292-AEC-OMIM-106260-ADULT-OMIM-103285-LMS-OMIM-603543 -- "
            "p63-TAp63-DNp63-Isoforms-Ectoderm-Basal-Cells-Proliferation-Differentiation -- "
            "EEC-Syndrome-Ectrodactyly-Lobster-Claw-Split-Hand-Foot-PATHOGNOMONIC -- "
            "AEC-Ankyloblepharon-Filiform-Adnatum-Eyelid-Fusion-At-Birth-PATHOGNOMONIC -- "
            "LACRIMAL-DUCT-ATRESIA-Epiphora-In-EEC-90pct -- "
            "RENAL-ANOMALIES-Hydronephrosis-20pct-EEC -- "
            "Allelic-EEC-AEC-ADULT-LMS-SHFM4-Rapp-Hodgkin -- "
            "OMIM-Gene-TP63-603273-Multiple-Allelic-Syndromes"
        ),
        "locus": "3q28",
        "protein_size": "680 aa / 72 kDa",
        "inheritance": (
            "AD (autosomal dominant); missense mutations in DNA-binding domain (EEC) or SAM domain (AEC); "
            "variable expressivity within EEC/AEC families; "
            "de novo mutations ~50% of EEC; "
            "genotype-phenotype: L-loop mutations → EEC; SAM domain → AEC; TI domain → ADULT"
        ),
        "ed_category": "Syndromic ED — TP63 spectrum (ectrodactyly + cleft + EDA; multiple allelic syndromes)",
        "pathognomonic": (
            "EEC: SPLIT HAND/FOOT (ectrodactyly = lobster-claw deformity) + CLEFT LIP/PALATE + EDA TRIAD PATHOGNOMONIC; "
            "AEC/Hay-Wells: ANKYLOBLEPHARON FILIFORM ADNATUM (eyelid fusion by tissue strands) AT BIRTH PATHOGNOMONIC; "
            "LACRIMAL DUCT ATRESIA (epiphora from birth) in 90% of EEC"
        ),
        "treatment": (
            "Surgical release of ankyloblepharon in AEC — within days of birth to prevent amblyopia; "
            "Lacrimal duct probing/DCR for lacrimal stenosis in EEC (epiphora management); "
            "Cleft lip/palate repair: lip at 3-6 months, palate at 12-18 months; "
            "Ectrodactyly: hand surgery (separation of syndactyly, digit creation) — staged procedures; "
            "Foot ectrodactyly: orthotic shoes; surgical correction if functional impairment; "
            "Renal USS at diagnosis: hydronephrosis/duplex collecting system in 20% of EEC; "
            "EDA features: dental prosthetics + cooling if significant anhidrosis; "
            "Ophthalmology: lacrimal, corneal, vision — annual monitoring"
        ),
        "key_features": [
            "EEC: SPLIT HAND/FOOT (ectrodactyly) — lobster-claw deformity pathognomonic; unilateral or bilateral",
            "CLEFT LIP/PALATE — present in >85% of EEC; lip + palate or isolated palate",
            "ECTODERMAL DYSPLASIA — sparse hair, dental anomalies, dry skin; less severe than XLHED",
            "LACRIMAL DUCT ATRESIA — epiphora (watering eyes) from birth in ~90% EEC",
            "AEC/Hay-Wells: ANKYLOBLEPHARON FILIFORM ADNATUM — eyelid fusion by tissue strands at birth (surgical emergency)",
            "RENAL ANOMALIES — hydronephrosis/duplex system in ~20% EEC (USS at diagnosis)",
        ],
        "monitoring": [
            "Ophthalmology at birth: ankyloblepharon (AEC — surgical release), lacrimal (EEC — probing)",
            "Renal USS at diagnosis and annually in EEC",
            "Annual hand/foot orthopaedic review for ectrodactyly management",
            "Dental: panoramic X-ray from age 3yr; orthodontics + prosthetics",
            "Skin: annual dermatology for dry skin/EDA features",
            "Hearing: audiometry (cleft palate → conductive hearing loss + OME common)",
        ],
        "key_ddx": [
            "CHARGE syndrome (coloboma+choanal atresia+heart+genital — CHD7; ectrodactyly NOT a feature)",
            "WNT10A-OODD (severe oligodontia + nail — NO ectrodactyly or cleft)",
            "Rapp-Hodgkin syndrome (TP63 allelic — sparse hair + nail + cleft palate; milder)",
            "ADULT syndrome (TP63 allelic — ADactyly-UroGT-LD-TD; limb+urogenital+mammary)",
        ],
        "cardiac_risk": False,
        "dental_risk": True,
        "heat_risk": False,
        "immunodeficiency_risk": False,
        "cleft_risk": True,
        "retinal_risk": False,
        "severity_options": ["EEC mild (partial ectrodactyly + cleft)", "EEC severe (bilateral ectrodactyly + cleft + renal)", "AEC (ankyloblepharon + severe skin erosions)"],
        "systemic_options": ["Ectrodactyly", "Cleft lip/palate", "Lacrimal atresia", "Ectodermal dysplasia", "Renal anomalies", "Ankyloblepharon (AEC)"],
        "treatments_used": ["Ankyloblepharon release", "Lacrimal probing/DCR", "Cleft repair", "Hand surgery", "Renal USS monitoring", "Dental prosthetics"],
    },
    # -- IKBKG — Incontinentia Pigmenti / EDA-ID ------------------------------------------
    {
        "gene": "IKBKG",
        "alt_name": (
            "IKBKG (IKBKG-419aa-Xq28 / XLD — Incontinentia-Pigmenti-IP-Females — "
            "4-STAGE-SKIN-VESICULAR-VERRUCOUS-HYPERPIGMENTED-ATROPHIC-WHORLED-PATHOGNOMONIC — "
            "Males-Usually-LETHAL-EDA-ID-Rare-Surviving-Males — "
            "RETINAL-TRACTION-DETACHMENT-Emergency-NF-kB-Pathway)"
        ),
        "protein": (
            "IKBKG -- Xq28 XLD -- IKBKG-419aa -- "
            "NF-kB-Essential-Modulator-NEMO-48kDa-IKK-Complex-Regulatory-Subunit -- "
            "IP-OMIM-308300-EDA-ID-OMIM-300291 -- "
            "NEMO→IKKα/IKKβ→IκB-Phosphorylation→NF-kB-Release→Ectoderm-Immune-Survival -- "
            "IP-FEMALES-XLD-4-Stage-Skin: "
            "Stage1-Eosinophilic-Vesicles-Birth-PATHOGNOMONIC "
            "Stage2-Verrucous-Hyperkeratosis-Warts "
            "Stage3-Hyperpigmented-Whorled-Lines-of-Blaschko-PATHOGNOMONIC "
            "Stage4-Atrophic-Hypopigmented-Streaks -- "
            "SEIZURES-40pct-IP-Neonatal-Emergency -- "
            "RETINAL-TRACTION-DETACHMENT-Vascular-Occlusion-URGENT-Ophthalmology -- "
            "MALES-HEMIZYGOUS-NULL-Usually-LETHAL-Recurrent-Miscarriage-Pattern -- "
            "EDA-ID-Rare-Surviving-Males-ED+Mycobacterial-Immunodeficiency -- "
            "OMIM-Gene-IKBKG-300248-Disease-IP-308300"
        ),
        "locus": "Xq28",
        "protein_size": "419 aa / 48 kDa",
        "inheritance": (
            "XLD (X-linked dominant); most cases de novo; heterozygous females: IP (variable severity); "
            "hemizygous males: usually lethal in utero → recurrent miscarriage pattern in carrier mothers; "
            "rare surviving males (mosaic or hypomorphic mutation): EDA-ID; "
            "del(exon4-10) most common pathogenic variant (NEMO del4-10) — accounts for ~80% IP"
        ),
        "ed_category": "NF-kB Pathway ED — Incontinentia Pigmenti (X-linked dominant, females; males lethal)",
        "pathognomonic": (
            "4-STAGE SKIN SEQUENCE PATHOGNOMONIC for IP: "
            "(1) Eosinophilic vesicular blisters at birth following lines of Blaschko; "
            "(2) Verrucous warty plaques; "
            "(3) WHORLED HYPERPIGMENTATION along Blaschko lines (pathognomonic pattern); "
            "(4) Atrophic linear hypopigmented streaks in adulthood; "
            "RETINAL TRACTION DETACHMENT = vascular occlusion → requires urgent ophthalmology"
        ),
        "treatment": (
            "OPHTHALMOLOGY URGENT at diagnosis and every 3 months in infancy — retinal vascular occlusion → "
            "traction detachment → blindness if not treated with laser/cryotherapy early; "
            "Seizures: anti-epileptic drugs (common in IP — ~40%); MRI brain at diagnosis; "
            "EEG if seizures develop; neurological follow-up; "
            "Skin: topical emollients for vesicular stage; no specific treatment needed for Blaschko hyperpigmentation; "
            "Dental: agenesis, abnormal tooth shape — dental prosthetics; "
            "Hair: patchy alopecia (cicatricial) — cosmetic management; "
            "Genetic counselling: de novo in ~65% — siblings at low risk; daughters of affected mother 50% risk"
        ),
        "key_features": [
            "STAGE 1: Eosinophilic vesicular blisters at birth along Blaschko lines — may be mistaken for herpes/impetigo",
            "STAGE 2: Verrucous warty hyperkeratotic plaques — weeks to months after birth",
            "STAGE 3: WHORLED HYPERPIGMENTATION along Blaschko lines — PATHOGNOMONIC pattern",
            "STAGE 4: Atrophic, hypopigmented linear streaks — adults (some never develop stage 4)",
            "RETINAL TRACTION DETACHMENT — vascular occlusion leading cause of blindness in IP — OPHTHALMOLOGY EMERGENCY",
            "SEIZURES — 40% IP; neonatal seizures associated with cortical neuronal migration defects",
        ],
        "monitoring": [
            "Ophthalmology: RetCam/funduscopy every 3 months in first 2 years — retinal vascular occlusion + traction",
            "MRI brain at diagnosis + if seizures: cortical migration defects, periventricular leukomalacia",
            "EEG: baseline + if seizures develop",
            "Dental panoramic X-ray from age 3yr: agenesis + peg teeth",
            "Skin: stage monitoring; biopsy stage 1 if diagnosis uncertain",
            "Annual dermatology to monitor stage progression + cicatricial alopecia",
        ],
        "key_ddx": [
            "Herpes simplex neonatal (stage 1 vesicles — PCR for HSV + Blaschko pattern + Eos on biopsy distinguish IP)",
            "Epidermolysis bullosa (EDA pathway — no Blaschko lines, no retinal involvement, no seizures)",
            "Linear epidermal naevus (non-inflammatory, no stages, no systemic features)",
            "EDA-ID (IKBKG hypomorphic in males — combined immunodeficiency + HED phenotype)",
        ],
        "cardiac_risk": False,
        "dental_risk": True,
        "heat_risk": False,
        "immunodeficiency_risk": True,
        "cleft_risk": False,
        "retinal_risk": True,
        "severity_options": ["Mild (stages 1-3, no systemic)", "Moderate (seizures or partial retinal)", "Severe (seizures + retinal detachment + dental)"],
        "systemic_options": ["Vesicular rash (stage 1)", "Whorled hyperpigmentation", "Retinal vascular disease", "Seizures", "Dental agenesis", "Cicatricial alopecia"],
        "treatments_used": ["Retinal laser/cryotherapy", "Anti-epileptic drugs", "MRI brain", "Dental prosthetics", "Topical emollients", "RetCam monitoring"],
    },
    # -- GJB6 — Clouston syndrome (hidrotic ED) --------------------------------------------
    {
        "gene": "GJB6",
        "alt_name": (
            "GJB6 (GJB6-261aa-13q12.11 / AD — Clouston-Syndrome-Hidrotic-Ectodermal-Dysplasia — "
            "DIFFUSE-ALOPECIA+NAIL-DYSTROPHY+PPK-TRIAD-PATHOGNOMONIC — "
            "NO-ANHIDROSIS-KEY-DDx-EDA-EDAR-Heat-Stroke-NOT-a-Risk — "
            "Quebec-French-Canadian-p.Gly11Arg-Founder)"
        ),
        "protein": (
            "GJB6 -- 13q12.11 AD -- GJB6-261aa -- "
            "Connexin-30-Cx30-30kDa-Gap-Junction-Epidermal-Hair-Follicle -- "
            "Clouston-Syndrome-Hidrotic-ED-OMIM-129500 -- "
            "GJB6-Gap-Junction-Intercellular-Communication-Epidermal-Differentiation -- "
            "DIFFUSE-PROGRESSIVE-ALOPECIA-Total-Scalp-Hair-Loss-PATHOGNOMONIC -- "
            "NAIL-DYSTROPHY-Thickened-Discoloured-Pachyonychia-Like -- "
            "PPK-Palmoplantar-Hyperkeratosis-Diffuse-Without-Transgredient -- "
            "SWEATING-COMPLETELY-NORMAL-No-Anhidrosis-No-Heat-Stroke-KEY-DDx -- "
            "TEETH-MOSTLY-SPARED-Minor-Hypodontia-Only -- "
            "Quebec-French-Canadian-p.Gly11Arg-Founder-5000-per-Province-Quebec -- "
            "del-GJB6-D13S1830-causes-DFNB1b-digenic-deafness-not-Clouston -- "
            "OMIM-Gene-GJB6-604418-Disease-Clouston-129500"
        ),
        "locus": "13q12.11",
        "protein_size": "261 aa / 30 kDa",
        "inheritance": (
            "AD (autosomal dominant); missense mutations in extracellular loop domains; "
            "p.Gly11Arg most common European/Quebec variant (founder); "
            "del(GJB6-D13S1830) causes digenic DFNB1 deafness with GJB2 — not Clouston; "
            "high penetrance; expressivity variable (mild → severe alopecia)"
        ),
        "ed_category": "Hidrotic ED — GJB6 connexin-30 (alopecia + nail + PPK; sweating NORMAL, teeth mostly spared)",
        "pathognomonic": (
            "DIFFUSE PROGRESSIVE ALOPECIA (total or near-total scalp hair loss by adulthood) + "
            "NAIL DYSTROPHY (thickened, discoloured, pachyonychia-like) + "
            "PALMOPLANTAR HYPERKERATOSIS TRIAD PATHOGNOMONIC; "
            "SWEATING COMPLETELY NORMAL = KEY DDx from EDA/EDAR-HED (which have anhidrosis); "
            "HEAT STROKE NOT A RISK in Clouston (unlike XLHED)"
        ),
        "treatment": (
            "ALOPECIA: wig + scalp prosthesis from childhood; cosmetically challenging; "
            "no effective hair-restoring treatment; minoxidil occasionally tried (limited benefit); "
            "NAIL: filing, softening, gel nails for cosmesis; "
            "PPK: urea 40-50% cream + keratolytics; acitretin for severe PPK; "
            "NO COOLING VEST NEEDED — sweating normal; no heat restriction; "
            "TEETH: mostly spared — dental prosthetics only if significant hypodontia; "
            "Multidisciplinary: dermatology + genetics; support groups (Quebec Clouston community)"
        ),
        "key_features": [
            "DIFFUSE PROGRESSIVE ALOPECIA — sparse hair at birth → progressive total hair loss by adulthood",
            "NAIL DYSTROPHY — thickened, discoloured, pachyonychia-like; all 20 nails affected",
            "PALMOPLANTAR HYPERKERATOSIS — diffuse; worsens with age; no transgredient extension",
            "SWEATING COMPLETELY NORMAL — KEY DDx from EDA/EDAR-HED — NO HEAT STROKE RISK",
            "TEETH MOSTLY SPARED — minor hypodontia possible; major dental anomaly NOT a feature",
            "Quebec founder p.Gly11Arg — highest prevalence French-Canadian population",
        ],
        "monitoring": [
            "Annual dermatology: hair + nail + skin assessment; acitretin response monitoring",
            "Dental: panoramic X-ray once at age 10yr — minor hypodontia only expected",
            "Ophthalmology: not routinely needed (GJB6 Clouston — no retinal disease)",
            "Psychosocial support: significant impact of alopecia on appearance/quality of life",
            "Genetic cascade: 50% risk (AD) — test siblings and at-risk relatives",
        ],
        "key_ddx": [
            "EDA-XLHED (anhidrosis + sparse hair + dental = XLHED; sweating normal in Clouston = KEY DDx)",
            "Alopecia areata (patchy not diffuse; autoimmune; no nail/PPK; family history different)",
            "Hidrotic ED non-GJB6 (rare — molecular panel distinguishes)",
            "GJB2-Vohwinkel (honeycomb PPK + SNHL + pseudoainhum — different PPK pattern; no total alopecia)",
        ],
        "cardiac_risk": False,
        "dental_risk": False,
        "heat_risk": False,
        "immunodeficiency_risk": False,
        "cleft_risk": False,
        "retinal_risk": False,
        "severity_options": ["Mild (partial alopecia, nail changes)", "Moderate (near-total alopecia + nail dystrophy)", "Severe (complete alopecia + severe nail + PPK)"],
        "systemic_options": ["Diffuse alopecia", "Nail dystrophy", "Palmoplantar hyperkeratosis", "Eyebrow/eyelash sparse", "Minor hypodontia"],
        "treatments_used": ["Wig/scalp prosthesis", "Urea 40% PPK", "Acitretin severe PPK", "Nail filing/gel", "Minoxidil (limited)"],
    },
    # -- IRF6 — Van der Woude syndrome / Popliteal Pterygium ----------------------------------
    {
        "gene": "IRF6",
        "alt_name": (
            "IRF6 (IRF6-467aa-1q32.3 / AD — Van-der-Woude-Syndrome-VWS — "
            "LOWER-LIP-PITS+CLEFT-LIP-PALATE-PATHOGNOMONIC — "
            "Most-Common-Syndromic-Cleft-Worldwide-2-5pct-All-CLP — "
            "Popliteal-Pterygium-Syndrome-Allelic-Wing-Webs+Syndactyly)"
        ),
        "protein": (
            "IRF6 -- 1q32.3 AD -- IRF6-467aa -- "
            "Interferon-Regulatory-Factor-6-IRF-DNA-Binding-Domain-53kDa -- "
            "VWS-OMIM-119300-PPS-OMIM-119500 -- "
            "IRF6-Epithelial-Periderm-Differentiation-Lip-Palate-Skin -- "
            "LOWER-LIP-PITS-Paramedian-Commissural-Pits-PATHOGNOMONIC-UNIQUE-ALL-Cleft-Syndromes -- "
            "CLEFT-LIP-PALATE-CLP-75pct-VWS-Isolated-Cleft-Palate-Only-25pct -- "
            "LOWER-LIP-PITS-Even-When-No-Cleft-Present-Diagnostic -- "
            "PPS-Adds-Popliteal-Webbing-Syndactyly-Syngnathia-Genital-Anomalies -- "
            "2-5pct-ALL-Non-Syndromic-CLP-Have-IRF6-Mutations -- "
            "OMIM-Gene-IRF6-607199-Disease-VWS-119300"
        ),
        "locus": "1q32.3",
        "protein_size": "467 aa / 53 kDa",
        "inheritance": (
            "AD (autosomal dominant); LOF variants + missense; "
            "variable expressivity: same mutation → lip pits only (no cleft) in some; cleft without pits in others; "
            "50% recurrence in offspring; "
            "lip pits alone (without cleft) in some family members = VWS carrier diagnosis"
        ),
        "ed_category": "Syndromic cleft ED — IRF6 spectrum (lip pits + cleft; PPS allelic with popliteal webs)",
        "pathognomonic": (
            "PARAMEDIAN LOWER LIP PITS (blind-ended sinuses or mucosal mounds at vermilion border) + "
            "CLEFT LIP/PALATE PATHOGNOMONIC for Van der Woude syndrome; "
            "LIP PITS UNIQUE — not present in any other cleft syndrome (PITHNOGNOMONiC for VWS vs isolated cleft); "
            "lip pits ALONE (without cleft) = VWS diagnosis — variable expressivity"
        ),
        "treatment": (
            "Cleft lip repair: surgical closure at 3-6 months of age; "
            "Palate repair: 12-18 months (timing varies by team); "
            "Lip pit excision: elective cosmetic surgery (sinuses may collect saliva/produce mucus); "
            "Popliteal pterygium (PPS): surgical division of web if causing contracture/gait restriction; "
            "Speech therapy after palate repair: velopharyngeal insufficiency monitoring; "
            "Audiometry: OME/glue ear common after cleft palate (grommets if needed); "
            "Orthodontics: alveolar bone graft + braces for dental alignment; "
            "Genetic counselling: 50% AD recurrence"
        ),
        "key_features": [
            "LOWER LIP PITS — paramedian sinuses at vermilion border; unique to VWS among cleft syndromes",
            "CLEFT LIP/PALATE — 75% VWS; 25% isolated cleft palate only",
            "LIP PITS WITHOUT CLEFT — variable expressivity; lip pits alone = VWS diagnosis",
            "POPLITEAL PTERYGIUM (PPS allelic) — webbing behind knees restricting leg extension",
            "SYNGNATHIA (PPS) — intraoral bands between jaws requiring neonatal division",
            "MINOR ECTODERMAL FEATURES — mild hair/nail changes possible but not cardinal",
        ],
        "monitoring": [
            "Audiometry from age 6 months: OME (glue ear) in cleft palate → conductive hearing loss",
            "Speech assessment from age 18 months: velopharyngeal insufficiency after palate repair",
            "Orthodontic assessment from age 7yr: alveolar bone graft planning",
            "Lip pits: review for recurrent mucus/saliva accumulation → elective excision",
            "PPS patients: annual orthopaedic assessment for popliteal web contracture",
            "Cascade genetic testing: family members — lip pits alone = VWS diagnosis",
        ],
        "key_ddx": [
            "Isolated cleft lip/palate (NO lip pits — most common distinction; IRF6 accounts for 2-5% isolated CLP)",
            "TP63-EEC (ectrodactyly + cleft — split hand/foot distinguishes from VWS)",
            "CHARGE syndrome (coloboma+CHD+choanal atresia — no lip pits)",
            "Popliteal Pterygium syndrome (PPS — same IRF6 gene, more severe: adds popliteal webs+syngnathia)",
        ],
        "cardiac_risk": False,
        "dental_risk": False,
        "heat_risk": False,
        "immunodeficiency_risk": False,
        "cleft_risk": True,
        "retinal_risk": False,
        "severity_options": ["Lip pits only (no cleft)", "Cleft lip/palate + lip pits", "PPS (popliteal webbing + cleft + syngnathia)"],
        "systemic_options": ["Cleft lip/palate", "Lower lip pits", "Popliteal pterygium (PPS)", "Syngnathia (PPS)", "Conductive hearing loss (OME)", "Minor dental anomalies"],
        "treatments_used": ["Cleft lip repair", "Palate repair", "Lip pit excision", "Popliteal web division", "Speech therapy", "Grommet insertion"],
    },
    # -- PVRL1 — CLPED1 / Zlotogora-Ogur syndrome -----------------------------------------
    {
        "gene": "PVRL1",
        "alt_name": (
            "PVRL1 (PVRL1-517aa-11q23.3 / AR — CLPED1-Cleft-Lip-Palate-Ectodermal-Dysplasia-Type-1 — "
            "CLEFT-LIP-PALATE+HYPODONTIA+NAIL-DYSPLASIA+SPARSE-HAIR-PATHOGNOMONIC — "
            "RECURRENT-HERPETIC-KERATITIS-Nectin-1-HSV-Receptor — "
            "Mediterranean-Middle-East-North-Africa-Founder)"
        ),
        "protein": (
            "PVRL1 -- 11q23.3 AR -- PVRL1-517aa -- "
            "Nectin-1-Poliovirus-Receptor-Related-1-58kDa-Ig-Family-Cell-Adhesion -- "
            "CLPED1-OMIM-225060-Zlotogora-Ogur-Syndrome -- "
            "Nectin-1-Cell-Adhesion-Adherens-Junction-Epithelial-Tissue-Integrity -- "
            "CLEFT-LIP-PALATE-Complete-Bilateral-Most-Severe-PATHOGNOMONIC -- "
            "HYPODONTIA-Multiple-Missing-Cone-Shaped-Teeth -- "
            "NAIL-DYSPLASIA-Thin-Brittle-Onychogryphosis-All-Nails -- "
            "SPARSE-DRY-HAIR-Hypotrichosis-Scalp-Eyebrows-Eyelashes -- "
            "RECURRENT-HERPES-KERATITIS-Nectin-1-HSV-1-Receptor-Cornea -- "
            "MEDITERRANEAN-FOUNDER-Arab-Turkish-Sephardic-Jewish-Populations -- "
            "OMIM-Gene-PVRL1-600644-Disease-CLPED1-225060"
        ),
        "locus": "11q23.3",
        "protein_size": "517 aa / 58 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF variants; "
            "Mediterranean/Middle East/North Africa founder populations (Arab, Turkish, Sephardic Jewish); "
            "consanguinity increases risk; "
            "25% recurrence in AR families; carrier frequency elevated in some founder populations"
        ),
        "ed_category": "Syndromic cleft ED — PVRL1/Nectin-1 (cleft + ED tetrad; herpetic keratitis risk)",
        "pathognomonic": (
            "CLEFT LIP/PALATE + HYPODONTIA + NAIL DYSPLASIA + SPARSE HAIR TETRAD PATHOGNOMONIC; "
            "RECURRENT HERPETIC KERATITIS (herpes simplex keratitis) — Nectin-1 is HSV-1 entry receptor; "
            "impaired corneal cell adhesion → HSV-1 gains entry → stromal keratitis → scarring → blindness; "
            "Mediterranean AR founder phenotype"
        ),
        "treatment": (
            "HERPES KERATITIS PROPHYLAXIS — oral aciclovir 400mg BD continuously from age 5yr; "
            "ophthalmology monitoring: slit-lamp annually + urgent if red eye; "
            "topical aciclovir 3% eye ointment for acute herpetic keratitis episodes; "
            "Cleft lip repair at 3-6 months; palate repair at 12-18 months; "
            "Dental prosthetics for hypodontia; orthodontic management; "
            "Nail care: emollients + protective footwear; "
            "Hair: cosmetic management; normal sweating (no heat restriction); "
            "Genetic counselling: AR — 25% recurrence; cascade carrier testing in Mediterranean families"
        ),
        "key_features": [
            "CLEFT LIP/PALATE — complete bilateral most common; severe orofacial clefting",
            "HYPODONTIA — multiple missing teeth; conical crown form; need prosthetics from childhood",
            "NAIL DYSPLASIA — thin, brittle nails; onychogryphosis; all 20 nails",
            "SPARSE/DRY HAIR — hypotrichosis scalp + eyebrows + eyelashes; not alopecia",
            "RECURRENT HERPETIC KERATITIS — Nectin-1 = HSV-1 corneal receptor; prophylactic aciclovir mandatory",
            "MEDITERRANEAN FOUNDER — elevated prevalence Arab/Turkish/Sephardic Jewish communities",
        ],
        "monitoring": [
            "Ophthalmology: slit-lamp annually + urgent any red/painful eye (herpetic keratitis)",
            "Aciclovir 400mg BD prophylaxis: monitoring compliance + renal function annually",
            "Dental panoramic X-ray from age 3yr: tooth agenesis pattern + prosthetic planning",
            "Speech assessment after palate repair: velopharyngeal insufficiency",
            "Audiometry: OME after cleft palate — grommets if needed",
            "Cascade carrier testing in Mediterranean families: both parents need to be carriers (AR)",
        ],
        "key_ddx": [
            "TP63-EEC (ectrodactyly + cleft — no ectrodactyly in PVRL1-CLPED1)",
            "IRF6-Van der Woude (lip pits + cleft — no lip pits in PVRL1; no herpetic keratitis in VWS)",
            "WNT10A-OODD (severe oligodontia + nail — no cleft in OODD)",
            "Isolated cleft lip/palate (no ED features — PVRL1 has hypodontia+nail+hair)",
        ],
        "cardiac_risk": False,
        "dental_risk": True,
        "heat_risk": False,
        "immunodeficiency_risk": False,
        "cleft_risk": True,
        "retinal_risk": True,
        "severity_options": ["Moderate (cleft palate only + mild ED)", "Severe (bilateral CLP + full ED tetrad)", "Severe + herpetic keratitis (corneal scarring)"],
        "systemic_options": ["Cleft lip/palate", "Hypodontia", "Nail dysplasia", "Sparse hair", "Herpetic keratitis", "Corneal scarring"],
        "treatments_used": ["Aciclovir prophylaxis", "Topical aciclovir eye ointment", "Cleft repair", "Dental prosthetics", "Ophthalmology monitoring", "Speech therapy"],
    },
]


def _build_cohort() -> list:
    cohort = []
    for entry in ED_GENES:
        seed = SEED_BASE + ED_GENES.index(entry)
        rng = random.Random(seed)
        gene = entry["gene"]
        for pid in range(40):
            severity = rng.choice(entry["severity_options"])
            systemic = rng.sample(entry["systemic_options"], k=rng.randint(1, min(3, len(entry["systemic_options"]))))
            treatment = rng.choice(entry["treatments_used"])

            # Age at diagnosis by gene
            if gene == "EDA":
                age = round(rng.uniform(0.1, 3), 1)
            elif gene == "EDAR":
                age = round(rng.uniform(0.1, 4), 1)
            elif gene == "WNT10A":
                age = round(rng.uniform(3, 10), 1)
            elif gene == "TP63":
                age = round(rng.uniform(0, 2), 1)
            elif gene == "IKBKG":
                age = round(rng.uniform(0, 1), 1)
            elif gene == "GJB6":
                age = round(rng.uniform(1, 8), 1)
            elif gene == "IRF6":
                age = round(rng.uniform(0, 1), 1)
            else:  # PVRL1
                age = round(rng.uniform(0, 1), 1)

            follow_up = round(rng.uniform(1, 12), 1)
            heat_event = entry["heat_risk"] and rng.random() < 0.3
            retinal_event = entry["retinal_risk"] and rng.random() < 0.25

            cohort.append({
                "patient_id": f"{gene}-{seed}-{pid:03d}",
                "gene": gene,
                "severity": severity,
                "age_at_dx_yrs": age,
                "follow_up_yrs": follow_up,
                "systemic_features": systemic,
                "treatment": treatment,
                "heat_event": heat_event,
                "retinal_event": retinal_event,
                "cleft": entry["cleft_risk"],
                "dental_risk": entry["dental_risk"],
                "immunodeficiency": entry["immunodeficiency_risk"],
            })
    return cohort


def generate_overview() -> dict:
    cohort = _build_cohort()
    type_counts = {}
    for p in cohort:
        cat = next((e["ed_category"] for e in ED_GENES if e["gene"] == p["gene"]), "Unknown")
        type_counts[cat] = type_counts.get(cat, 0) + 1

    gene_summary = []
    for entry in ED_GENES:
        pts = [p for p in cohort if p["gene"] == entry["gene"]]
        avg_age = round(sum(p["age_at_dx_yrs"] for p in pts) / len(pts), 1) if pts else 0
        gene_summary.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "ed_category": entry["ed_category"],
            "pathognomonic": entry["pathognomonic"][:200],
            "cardiac_risk": entry["cardiac_risk"],
            "dental_risk": entry["dental_risk"],
            "heat_risk": entry["heat_risk"],
            "immunodeficiency_risk": entry["immunodeficiency_risk"],
            "cleft_risk": entry["cleft_risk"],
            "retinal_risk": entry["retinal_risk"],
            "avg_age_at_dx_yrs": avg_age,
            "n_patients": len(pts),
        })

    heat_emergency_genes = [e["gene"] for e in ED_GENES if e["heat_risk"]]
    retinal_emergency_genes = [e["gene"] for e in ED_GENES if e["retinal_risk"]]
    cleft_genes = [e["gene"] for e in ED_GENES if e["cleft_risk"]]
    immunodeficiency_genes = [e["gene"] for e in ED_GENES if e["immunodeficiency_risk"]]

    return {
        "title": "Hereditary-Ectodermal-Dysplasia-Atlas — Complete 8-Gene Hereditary ED Atlas",
        "n_genes": len(ED_GENES),
        "n_patients": len(cohort),
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "ed_categories": {
            "Anhidrotic/Hypohidrotic ED (EDA pathway)": "EDA (XLHED — triad + heat stroke) / EDAR (Autosomal HED)",
            "Hypodontia-predominant ED (WNT pathway)": "WNT10A (OODD — severe oligodontia, teeth #1; sweating NORMAL)",
            "Syndromic ED with Ectrodactyly/Cleft (p63)": "TP63 (EEC — split hand/foot + cleft + ED; AEC — ankyloblepharon)",
            "NF-kB Pathway ED (X-linked dominant)": "IKBKG (IP females — 4-stage Blaschko; males lethal; retinal emergency)",
            "Hidrotic ED (Conexin-30, normal sweating)": "GJB6 (Clouston — alopecia + nail + PPK; no anhidrosis KEY DDx)",
            "Syndromic cleft ED (IRF6 — lip pits)": "IRF6 (VWS — lip pits + CLP; PPS allelic; most common syndromic CLP)",
            "Syndromic cleft ED (Nectin-1)": "PVRL1 (CLPED1 — CLP + ED + herpetic keratitis; Mediterranean AR)",
        },
        "inheritance_map": {
            "EDA": "XLR", "EDAR": "AD/AR", "WNT10A": "AR", "TP63": "AD",
            "IKBKG": "XLD", "GJB6": "AD", "IRF6": "AD", "PVRL1": "AR",
        },
        "key_clinical_pearls": [
            "EDA/XLHED: ANHIDROSIS + SPARSE HAIR + HYPODONTIA TRIAD PATHOGNOMONIC — HEAT STROKE LIFE-THREATENING; cooling vest mandatory; starch-iodine sweat test confirms; EDX111 intra-amniotic injection 26-30wks = first prenatal DMT; saddle nose + frontal bossing facies",
            "EDAR-Autosomal HED: SAME TRIAD as XLHED but autosomal (AD mild or AR severe phenocopy XLHED); EDAR-D374A = East Asian population variant (thick straight hair — NOT disease); EDA/EDAR/EDARADD/NEMO converging NF-kB pathway",
            "WNT10A-OODD: SEVERE SELECTIVE OLIGODONTIA (20-28 missing permanent teeth) PATHOGNOMONIC; most common isolated oligodontia gene Europe (20%); SWEATING COMPLETELY NORMAL = KEY DDx from EDA/EDAR-HED; nail dystrophy + PPK; panoramic X-ray at age 3yr mandatory",
            "TP63-EEC: SPLIT HAND/FOOT (ectrodactyly) + CLEFT LIP/PALATE + EDA TRIAD PATHOGNOMONIC; AEC/Hay-Wells (allelic): ANKYLOBLEPHARON FILIFORM ADNATUM AT BIRTH = surgical emergency; lacrimal duct atresia 90% EEC; renal USS at diagnosis (20% anomalies)",
            "IKBKG-Incontinentia Pigmenti: 4-STAGE BLASCHKO SKIN PATHOGNOMONIC — vesicular→verrucous→whorled hyperpigmentation→atrophic; RETINAL TRACTION DETACHMENT = OPHTHO EMERGENCY — RetCam every 3 months first 2 years; males usually LETHAL; del(exon4-10) 80%",
            "GJB6-Clouston: DIFFUSE PROGRESSIVE ALOPECIA + NAIL DYSTROPHY + PPK TRIAD PATHOGNOMONIC; SWEATING COMPLETELY NORMAL = KEY DDx from EDA/EDAR (no heat stroke risk in Clouston); Quebec p.Gly11Arg founder; teeth mostly spared; del(GJB6-D13S1830) causes DFNB1b deafness NOT Clouston",
            "IRF6-VWS: LOWER LIP PITS (paramedian sinuses) + CLEFT LIP/PALATE PATHOGNOMONIC; LIP PITS UNIQUE among cleft syndromes (pathognomonic for VWS); lip pits alone without cleft = VWS diagnosis; PPS allelic (adds popliteal webs + syngnathia); 2-5% all CLP have IRF6 mutations",
            "PVRL1-CLPED1: CLEFT + HYPODONTIA + NAIL DYSPLASIA + SPARSE HAIR TETRAD; RECURRENT HERPETIC KERATITIS PATHOGNOMONIC — Nectin-1 = HSV-1 corneal receptor; aciclovir prophylaxis 400mg BD mandatory; Mediterranean AR founder; corneal scarring → blindness without antiviral prophylaxis",
        ],
        "gene_summary": gene_summary,
        "heat_emergency_genes": heat_emergency_genes,
        "retinal_emergency_genes": retinal_emergency_genes,
        "cleft_genes": cleft_genes,
        "immunodeficiency_genes": immunodeficiency_genes,
        "diagnostic_algorithm": {
            "Step_1": "Classify ED type: (A) ANHIDROSIS present → EDA/EDAR-HED pathway; (B) ALOPECIA + nail + PPK but NORMAL sweating → GJB6-Clouston; (C) SEVERE OLIGODONTIA dominant → WNT10A-OODD",
            "Step_2": "Sweat test (starch-iodine + pilocarpine iontophoresis): zero output = EDA/EDAR; partial = EDAR-AD; normal = GJB6/WNT10A/TP63/IRF6/PVRL1",
            "Step_3": "Cleft assessment: lip pits + CLP → IRF6-VWS; ectrodactyly + CLP → TP63-EEC; CLP + ED tetrad → PVRL1-CLPED1; ankyloblepharon at birth → TP63-AEC",
            "Step_4": "Skin pattern: 4-stage Blaschko vesicular rash in female newborn → IKBKG-IP (URGENT ophthalmology + neurology)",
            "Step_5": "X-linked vs autosomal: males severely affected with carrier females = XLR (EDA) or XLD (IKBKG-IP females); both sexes equally affected = autosomal",
            "Step_6": "NGS panel (EDA+EDAR+EDARADD+WNT10A+TP63+IKBKG+GJB6+IRF6+PVRL1): molecular confirmation + management pathway (EDX111 eligibility, aciclovir prophylaxis, ICD evaluation, renal USS)",
        },
        "type_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10]),
    }


def generate_breakdown() -> dict:
    cohort = _build_cohort()
    by_gene = {}
    for p in cohort:
        by_gene.setdefault(p["gene"], []).append(p)

    gene_breakdown = {}
    for entry in ED_GENES:
        g = entry["gene"]
        pts = by_gene.get(g, [])

        severity_dist = {}
        treatment_dist = {}
        systemic_dist = {}
        for p in pts:
            sv = p["severity"]
            severity_dist[sv] = severity_dist.get(sv, 0) + 1
            tx = p["treatment"]
            treatment_dist[tx] = treatment_dist.get(tx, 0) + 1
            for sf in p["systemic_features"]:
                systemic_dist[sf] = systemic_dist.get(sf, 0) + 1

        avg_age = round(sum(p["age_at_dx_yrs"] for p in pts) / len(pts), 2) if pts else 0
        avg_fu = round(sum(p["follow_up_yrs"] for p in pts) / len(pts), 1) if pts else 0
        heat_n = sum(1 for p in pts if p["heat_event"])
        retinal_n = sum(1 for p in pts if p["retinal_event"])
        cleft_n = sum(1 for p in pts if p["cleft"])

        gene_breakdown[g] = {
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "ed_category": entry["ed_category"],
            "n_patients": len(pts),
            "avg_age_at_dx_yrs": avg_age,
            "avg_follow_up_yrs": avg_fu,
            "heat_event_pct": round(heat_n / len(pts) * 100, 1) if pts else 0,
            "retinal_event_pct": round(retinal_n / len(pts) * 100, 1) if pts else 0,
            "cleft_pct": round(cleft_n / len(pts) * 100, 1) if pts else 0,
            "pathognomonic": entry["pathognomonic"],
            "treatment_highlight": entry["treatment"][:600],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:700],
            "monitoring": entry["monitoring"],
            "key_ddx": entry["key_ddx"],
            "heat_risk": entry["heat_risk"],
            "retinal_risk": entry["retinal_risk"],
            "cleft_risk": entry["cleft_risk"],
            "immunodeficiency_risk": entry["immunodeficiency_risk"],
            "severity_distribution": dict(sorted(severity_dist.items(), key=lambda x: -x[1])),
            "systemic_distribution": dict(sorted(systemic_dist.items(), key=lambda x: -x[1])),
            "treatment_distribution": dict(sorted(treatment_dist.items(), key=lambda x: -x[1])),
            "patients_sample": pts[:5],
        }

    return {
        "title": "Hereditary-Ectodermal-Dysplasia-Atlas — Per-Gene Breakdown",
        "n_genes": 8,
        "n_patients": len(cohort),
        "gene_breakdown": gene_breakdown,
        "clinical_emergency_flags": [
            "EDA/XLHED HEAT EMERGENCY: ANHIDROSIS = ZERO SWEAT GLANDS — any fever/hot environment = HEAT STROKE LIFE-THREATENING; cooling vest MANDATORY from diagnosis; temperature management protocol for school/sports; antipyretics early; ICU admission if heat stroke develops",
            "EDAR-AR HEAT EMERGENCY: Same heat risk as XLHED — complete anhidrosis in biallelic null; EDAR-AD milder (partial sweating) but still elevated heat risk; cooling measures per EDA protocol",
            "IKBKG/IP RETINAL EMERGENCY: Retinal vascular occlusion → traction detachment → PERMANENT BLINDNESS; RetCam/funduscopy every 3 months in first 2 years of life MANDATORY; laser photocoagulation/cryotherapy at first sign of neovascularisation; ophthalmology referral at IP diagnosis = DAY 1",
            "TP63-AEC EYELID EMERGENCY: Ankyloblepharon filiform adnatum = eyelid fusion by tissue strands at birth; SURGICAL RELEASE within days of birth to prevent corneal abrasion + stimulus deprivation amblyopia; neonatal ophthalmology consultation at delivery",
            "PVRL1 HERPETIC KERATITIS: Nectin-1 = HSV-1 corneal entry receptor; LIFELONG aciclovir 400mg BD prophylaxis MANDATORY from age 5yr; slit-lamp URGENT for any red/painful/photophobic eye; topical aciclovir 3% ointment acute episodes; corneal scarring → blindness without prophylaxis",
            "IKBKG/IP NEONATAL SEIZURES: ~40% IP develop seizures; neonatal seizures in IP emergency; MRI brain at diagnosis; EEG if seizure suspected; anti-epileptic drugs early; cortical neuronal migration defect mechanism",
        ],
    }


def generate_definitions() -> dict:
    return {
        "title": "Hereditary-Ectodermal-Dysplasia-Atlas — Definitions & Glossary",
        "gene_entries": {
            entry["gene"]: {
                "full_name": entry["gene"],
                "protein_size": entry["protein_size"],
                "locus": entry["locus"],
                "inheritance": entry["inheritance"],
                "disease_name": entry["ed_category"],
                "pathognomonic": entry["pathognomonic"],
                "key_features": entry["key_features"],
                "treatment": entry["treatment"],
                "key_ddx": entry["key_ddx"],
                "monitoring": entry["monitoring"],
            }
            for entry in ED_GENES
        },
        "ed_biology_glossary": {
            "Ectodermal dysplasia (ED)": "Group of >200 heritable conditions affecting ectoderm-derived structures: skin (sweat glands, hair follicles, sebaceous glands), teeth, nails, mammary glands, mucous glands. Classified as: (1) Anhidrotic/Hypohidrotic ED (HED — EDA/EDAR pathway; absent/reduced sweat glands); (2) Hidrotic ED (GJB6-Clouston; sweating normal); (3) Syndromic ED (TP63-EEC/AEC; IKBKG-IP; IRF6-VWS; PVRL1-CLPED1)",
            "EDA1/EDAR/EDARADD NF-kB pathway": "Ectodysplasin-A (EDA1) → EDAR receptor → EDARADD (death domain adaptor) → NF-kB kinase complex (including IKBKG/NEMO) → NF-kB transcription factor → hair follicle + sweat gland + tooth bud morphogenesis. Disrupted at any node → HED phenotype: EDA (XLHED), EDAR (autosomal HED), EDARADD (HED3), IKBKG (IP/EDA-ID)",
            "Anhidrosis vs hypohidrosis": "Anhidrosis = ZERO sweat output (complete absence of eccrine sweat glands); hypohidrosis = reduced but present sweating. Distinction matters clinically: complete anhidrosis (EDA-null, EDAR-AR) = HEAT STROKE risk without cooling; partial hypohidrosis (EDAR-AD) = reduced but present heat tolerance. Starch-iodine test + pilocarpine iontophoresis quantifies sweat output",
            "WNT pathway in tooth development": "WNT10A activates canonical Wnt (Frizzled → LRP5/6 → β-catenin → TCF/LEF) in dental lamina epithelium → tooth bud initiation → morphogenesis. WNT10A loss-of-function → failure of tooth germ formation → severe selective oligodontia. WNT10A is most commonly mutated gene in severe isolated oligodontia (>6 missing permanent teeth)",
            "p63 isoforms in ectodermal development": "TP63 encodes transcriptional activating (TAp63) and dominant-negative (ΔNp63) isoforms. ΔNp63α highly expressed in basal keratinocytes = master regulator of stratified epithelium. p63 controls: stratified epithelium maintenance, ectoderm commitment, limb bud apical ectodermal ridge. Missense mutations in DNA-binding domain (L1 loop) → EEC; SAM domain → AEC/Hay-Wells",
            "Lines of Blaschko": "Map of epidermal clone trajectories during embryonic development; represents patterns of X-chromosome lyonization in heterozygous females. Incontinentia Pigmenti (IKBKG/IP): affected clones (NF-kB-deficient, IKBKG-mutant) undergo apoptosis → surviving wild-type clones form the whorled Blaschko pattern of hyperpigmentation/atrophy. Vesicles, hyperpigmentation, atrophy all follow Blaschko lines in IP",
            "Oligodontia vs hypodontia vs anodontia": "Hypodontia = 1-5 missing permanent teeth (excluding wisdom teeth); oligodontia = 6+ missing permanent teeth; anodontia = all permanent teeth absent. WNT10A → most commonly identified cause of SEVERE oligodontia (20-28 missing); EDA/EDAR → hypodontia/oligodontia as part of HED triad; IRF6/TP63/PVRL1 → variable hypodontia as part of cleft syndrome",
            "Nectin-1 (PVRL1) and herpes simplex virus": "Nectin-1 (PVRL1/HveC) is the primary entry receptor for Herpes Simplex Virus 1 (HSV-1) on corneal epithelial cells. PVRL1-deficient patients: impaired corneal cell adhesion and absent Nectin-1 protein → paradoxically INCREASED HSV-1 susceptibility through impaired innate immune signalling; recurrent stromal keratitis → scarring → visual impairment",
        },
        "ed_type_glossary": {
            "X-Linked Hypohidrotic Ectodermal Dysplasia (XLHED)": "Most common ED subtype; EDA gene (Xq12); XLR; hemizygous males fully affected; heterozygous females variable (often mild). Classic triad: anhidrosis + hypotrichosis + hypodontia. EDX111 prenatal treatment available. Heat management cornerstone of care",
            "Autosomal Hypohidrotic Ectodermal Dysplasia (HED)": "EDAR or EDARADD; same clinical triad as XLHED; AD (usually milder partial hypohidrosis) or AR (severe phenocopy XLHED). Male-to-male transmission possible (distinguishes from XLHED). Molecular panel EDA/EDAR/EDARADD required to distinguish",
            "Odonto-Onycho-Dermal Dysplasia (OODD)": "WNT10A biallelic; severe selective oligodontia + nail dystrophy + PPK + dry hair; sweating NORMAL. Most common cause severe isolated oligodontia in Europeans. Schöpf-Schulz-Passarge (SSPS) = OODD + eyelid hidrocystomas + hypotrichosis",
            "EEC syndrome (Ectrodactyly-Ectodermal Dysplasia-Clefting)": "TP63 AD; ectrodactyly + cleft lip/palate + EDA triad. Lacrimal duct atresia 90%. Renal anomalies 20%. Multiple TP63 allelic syndromes (AEC/AEC/ADULT/LMS/Rapp-Hodgkin)",
            "Incontinentia Pigmenti (IP)": "IKBKG XLD; females (heterozygous affected); males usually lethal. 4-stage Blaschko skin. Retinal vascular emergency + seizures. Most cases de novo; del(exon4-10) 80%",
            "Clouston syndrome (Hidrotic Ectodermal Dysplasia)": "GJB6-Connexin-30 AD; diffuse alopecia + nail dystrophy + PPK; sweating NORMAL (NOT anhidrotic). No heat stroke risk. Quebec French-Canadian p.Gly11Arg founder. Teeth mostly spared",
            "Van der Woude syndrome (VWS)": "IRF6 AD; most common syndromic cleft (2-5% all CLP). Lip pits PATHOGNOMONIC + CLP. PPS allelic. Variable expressivity: lip pits alone without cleft in some family members",
            "CLPED1 / Zlotogora-Ogur syndrome": "PVRL1/Nectin-1 AR; cleft + hypodontia + nail + sparse hair tetrad. Recurrent herpetic keratitis. Mediterranean AR founder. Aciclovir prophylaxis mandatory lifelong",
        },
        "treatment_glossary": {
            "EDX111 (EDA1 protein — XLHED prenatal treatment)": "Recombinant human EDA1 ectodomain protein (EDX111); administered via intra-amniotic injection at 26-30 weeks gestation in fetuses with confirmed EDA/EDAR-AR molecular diagnosis; restores NF-kB pathway signalling in ectoderm during critical developmental window; dramatically improves sweat gland formation in treated males; FDA Fast Track designation; first approved prenatal disease-modifying treatment for any ED subtype",
            "Cooling vest (XLHED/HED heat management)": "Mandatory from diagnosis in all anhidrotic ED patients; phase-change cooling vests (20°C) worn during exercise/hot weather; classroom cooling plan required for school attendance; cooling vest + antipyretics + cool water spray comprehensive heat management; caregivers educated: core temperature >38.5°C = emergency if anhidrotic; parents trained in wet towel cooling as first aid",
            "Aciclovir prophylaxis (PVRL1-CLPED1)": "Oral aciclovir 400mg BD continuously from age 5yr in all PVRL1-CLPED1 patients; reduces HSV-1 reactivation frequency; prevents recurrent herpetic keratitis → stromal scarring; topical aciclovir 3% ophthalmic ointment for breakthrough acute episodes (5 times daily for 5 days); compliance monitoring at every visit; alternative: valaciclovir 500mg OD (adult dose)",
            "RetCam funduscopy (IKBKG-IP retinal monitoring)": "Retinal camera system (RetCam III) for wide-field fundal imaging in neonates and infants under sedation; mandatory every 3 months in first 2 years of IP diagnosis; detects retinal vascular abnormalities (avascularity, tortuosity, neovascularisation) before traction detachment; laser photocoagulation to avascular retinal periphery prevents retinal detachment in IP",
            "Starch-iodine sweat test (anhidrosis confirmation)": "Iodine solution applied to palm/axilla + starch powder applied on top; pilocarpine iontophoresis (0.5% pilocarpine, 2mA × 5min) stimulates sweating; sweating = blue-black starch-iodine reaction; absence = white (anhidrosis); quantitative pilocarpine iontophoresis sweat collector measures sweat rate (nl: >15 μL/min; <10 = anhidrosis); PATHOGNOMONIC zero output in XLHED; also quantifies residual sweating in EDAR-AD",
        },
        "diagnostic_tests": {
            "Sweat test (starch-iodine + pilocarpine iontophoresis)": "Gold standard for anhidrosis quantification in HED; pilocarpine iontophoresis stimulates eccrine glands; starch-iodine test visualises output areas; quantitative collector measures sweat rate; ZERO output = anhidrosis (EDA/EDAR-AR); partial = EDAR-AD; NORMAL = GJB6-Clouston/WNT10A/TP63/IRF6/PVRL1; mandatory at HED diagnosis",
            "Panoramic dental X-ray (OPG) for ED": "From age 3yr in all ED patients with dental anomaly risk; documents tooth germ presence/absence; conical tooth form; WNT10A: shows near-complete absence of permanent tooth germs (20-28 missing); EDA/EDAR: partial absence + conical crowns; PVRL1: variable hypodontia; guides prosthetic timeline and bone graft planning",
            "RetCam funduscopy (IP — Incontinentia Pigmenti)": "Wide-field retinal imaging under sedation; every 3 months first 2 years in all IP patients; detects peripheral retinal avascularity + neovascularisation + traction before detachment occurs; laser/cryotherapy guided by RetCam findings; prevents blindness in IP; urgent adult funduscopy if new floaters/reduced vision in known IP",
            "Biopsy for IP staging": "Skin biopsy of vesicular/verrucous lesion: stage 1 = eosinophilic spongiosis + dermal eosinophilic infiltrate; NEMO-del4-10 MLPA on blood confirms IKBKG diagnosis; biopsy confirms IP when clinical pattern uncertain (neonatal vesicular eruption can be mistaken for HSV/impetigo); X-inactivation study on buccal/hair root for mosaic IP in males",
            "Cleft nasendoscopy + videofluoroscopy": "Nasendoscopy: assesses palate movement and velopharyngeal closure post-repair in TP63/IRF6/PVRL1; videofluoroscopy: real-time assessment of velopharyngeal function during speech; guides need for secondary palate surgery (pharyngoplasty/velopharyngeal flap); performed at age 3-4yr after primary palate repair",
            "NGS Ectodermal Dysplasia gene panel": "Comprehensive panel: EDA, EDAR, EDARADD, WNT10A, TP63, IKBKG, GJB6, IRF6, PVRL1 + NEMO-del4-10 MLPA + broader ED genes (LTBP3, GJA1, WRAP53, KRTAP); essential for: molecular diagnosis of anhidrotic vs hidrotic vs syndromic ED; genotype-phenotype prediction; recurrence risk calculation (XLR/XLD/AD/AR); EDX111 eligibility; IKBKG deletion sizing",
        },
    }


# Aliases for api_backend.py compatibility
def overview() -> dict:
    return generate_overview()


def breakdown() -> dict:
    return generate_breakdown()


def definitions() -> dict:
    return generate_definitions()


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"  Title: {ov['title']}")
    print(f"  Patients: {ov['n_patients']}  |  Genes: {ov['n_genes']}")
    print(f"  Seeds: {ov['seed_range']}")
    print("  Key pearls:")
    for p in ov["key_clinical_pearls"][:4]:
        print(f"    - {p[:100]}")

    print("\n=== BREAKDOWN (gene counts) ===")
    bk = generate_breakdown()
    for g, info in bk["gene_breakdown"].items():
        print(f"  {g}: {info['n_patients']} pts | Type: {info['ed_category'][:60]}")
