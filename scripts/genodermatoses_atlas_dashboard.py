#!/usr/bin/env python3
"""Genodermatoses Atlas — Complete 8-Gene Hereditary Skin Disorder Reference
ABCA12  (Harlequin Ichthyosis HI; 2595 aa; 2q34; AR; ABCA12 lipid transporter lamellar granules;
         most severe ichthyosis — Collodion membrane at birth; skin barrier failure; temperature
         dysregulation; retinoid acitretin mandatory; emollient saturation protocol; life expectancy
         improving with modern care but neonatal mortality remains high) ·
KRT1    (Epidermolytic Ichthyosis EI / Bullous Congenital Ichthyosiform Erythroderma BCIE; 644 aa;
         12q13.13; AD; suprabasal keratin 1 defects → tonofilament aggregation → blistering then
         hyperkeratosis; secondary Staphylococcus infection mandatory monitoring; retinoids reduce
         hyperkeratosis; dilute bleach baths antiseptic) ·
STS     (X-linked Recessive Ichthyosis XLRI; 583 aa; Xp22.31; XLR; steroid sulfatase deficiency
         → cholesterol sulfate accumulation → barrier defect; CONTIGUOUS XDEL = KAL1 (Kallmann) +
         STS + NLGN4X → MUST check; corneal opacities 50% ASYMPTOMATIC; undescended testes 15-25%;
         scaling improves in summer; emollients lifelong) ·
COL7A1  (Dystrophic Epidermolysis Bullosa DEB; 2944 aa; 3p21.31; AD = DDEB / AR = RDEB; collagen VII
         anchoring fibrils sublaminadensa; RDEB most severe — esophageal strictures, pseudosyndactyly,
         SCC risk 40-80% lifetime — leading cause of death in RDEB; wound dressings NON-ADHERENT
         mandatory; Oleogel-S10 FDA 2023 first approved topical EB therapy) ·
LAMA3   (Junctional Epidermolysis Bullosa Herlitz type JEB-H; 3936 aa; 18q11.2; AR; laminin-332 alpha3
         chain; JEB-H is LETHAL — most infants die by age 2 from sepsis/respiratory failure/
         malnutrition; granulation tissue PATHOGNOMONIC — larynx/trachea/esophagus; no specific therapy;
         gene therapy trials ongoing; LAMA3/LAMB3/LAMC2 all cause JEB — LAMB3 most common mutation) ·
KRT5    (Epidermolysis Bullosa Simplex EBS; 590 aa; 12q13.13; AD; basal keratin 5; Weber-Cockayne
         subtype (hands/feet, blisters with heat); Dowling-Meara DM-EBS (herpetiform clusters, most
         severe EBS, generalized); cooling strategy for DM-EBS; diacerein (off-label) reduces blistering;
         KRT14 allelic — same clinical phenotype) ·
ATP2C1  (Hailey-Hailey Disease HHD / Familial Benign Pemphigus; 919 aa; 3q22.1; AD; SERCA2C Ca2+
         pump Golgi/ER → desmosome assembly failure → suprabasal acantholysis; INTERTRIGINOUS
         EROSIONS with foul odour; triggers: heat, sweating, friction, secondary infection (Candida,
         HSV, Staphylococcus); tacrolimus 0.1% first-line topical; botulinum toxin injections reduce
         sweating = excellent for flexural disease; systemic retinoids if refractory) ·
EDA     (X-linked Hypohidrotic Ectodermal Dysplasia XLHED; 391 aa; Xq12-q13.1; XLR; ectodysplasin-A
         NF-κB pathway → anhidrosis (HEAT STROKE RISK — life-threatening), hypotrichosis, hypodontia
         conical teeth; ANHIDROSIS = cooling vest mandatory; dental rehabilitation from age 3; prenatal
         EDA1 intraamniotic injection EDX111 (Xolair-class biologics) — first disease-modifying Rx
         if given in utero; female carriers mosaic — check for segmental hypohidrosis)
320-patient aggregate cohort (8 × 40, seeds 1278–1285)
"""

import random

SEED_BASE = 1278

GENO_GENES = [
    # ── ABCA12 — Harlequin Ichthyosis ────────────────────────────────────────
    {
        "gene": "ABCA12",
        "protein": "ATP-Binding Cassette Subfamily A Member 12",
        "alias": (
            "ABCA12; OMIM gene 607800; Harlequin Ichthyosis #242500; "
            "2q34; 2595 aa; ~290 kDa; AR (biallelic LOF); ABC lipid transporter in lamellar granules of "
            "keratinocytes; transports glucosylceramide → extracellular lipid lamellae in stratum corneum; "
            "ABCA12 LOF → failure of lamellar body lipid secretion → absent intercellular lipid lamellae → "
            "catastrophic skin barrier failure; Harlequin Ichthyosis (HI) is the most severe ichthyosis; "
            "previously universally fatal in infancy; modern retinoid therapy dramatically improves survival; "
            "heterozygous carriers are clinically normal (pure AR);"
        ),
        "aa": "2595 aa",
        "kDa": "~290 kDa",
        "locus": "2q34",
        "omim_gene": 607800,
        "omim_disease": 242500,
        "inheritance": "AR (biallelic LOF); de novo very rare; consanguinity increases risk; carriers unaffected",
        "gene_class": (
            "ABCA12 is an ATP-binding cassette (ABC) transporter expressed in the granular layer of keratinocytes. "
            "It is essential for secretion of lipid-laden lamellar granules (Odland bodies) into the intercellular "
            "space of the stratum corneum. These lipid lamellae form the primary permeability barrier of skin. "
            "ABCA12 LOF → empty lamellar granules → absent intercellular lipid lamellae → Harlequin phenotype: "
            "(1) Collodion membrane at birth (hard, armour-like cracked scales restricting movement); "
            "(2) Ectropion (eyelid eversion — corneal ulceration risk from exposure); "
            "(3) Eclabium (lip eversion — feeding failure); "
            "(4) Ear canal occlusion → sensorineural hearing loss risk; "
            "(5) Limb contractures (skin armour restricts extension); "
            "(6) Thermoregulatory failure (no functional barrier = uncontrolled water and heat loss); "
            "Biallelic null variants (nonsense, frameshift, large deletions) = severe HI; "
            "Missense variants in ABC domain = milder congenital ichthyosis (LI/CIE spectrum); "
            "Prenatal diagnosis: fetal skin biopsy (36 wks) or exome sequencing of fetal DNA; "
            "CRITICAL: Retinoid acitretin (0.5–1 mg/kg/day) must be started within first days of life in HI — "
            "accelerates shedding of collodion plates and normalises scaling; "
            "EMOLLIENT saturation protocol: 6+ applications/day, urea-based products after cradle phase"
        ),
        "phenotype": (
            "AT BIRTH: Harlequin Ichthyosis — encasement in hard, fissured collodion-like armour; "
            "deep red fissures (rhagades) penetrate to dermis; ECTROPION + ECLABIUM PATHOGNOMONIC; "
            "flat/absent ears (occluded); joint contractures; digits shortened. "
            "NEONATAL CRISIS: Thermoregulatory failure (heat loss + heat stroke risk); "
            "sepsis from skin fissures (Staphylococcus, GNR); respiratory compromise; feeding failure. "
            "LONG-TERM (with treatment): Persistent generalised scaling (lamellar/plate-like) lifelong; "
            "ectropion persists — ophthalmology annually; hearing loss if ear canals remain occluded; "
            "improved life expectancy with modern NICU + retinoids (adults now surviving). "
            "ABSENT: brain, visceral, immune involvement (pure skin phenotype in survivors). "
            "GENETICS: biallelic ABCA12 null variants (WES essential — heterogeneous mutations); "
            "Carrier testing + prenatal diagnosis mandatory in subsequent pregnancies."
        ),
        "hallmark": (
            "HARLEQUIN INFANT: hard, cracked, armour-like collodion membrane at birth + "
            "ECTROPION + ECLABIUM = ABCA12 until proven otherwise; "
            "START ACITRETIN WITHIN 48H OF BIRTH (do NOT wait for genetic confirmation); "
            "EMOLLIENT PROTOCOL 6x/day minimum; "
            "Refer to specialist genodermatosis center immediately."
        ),
        "treatment_alerts": [
            "ACITRETIN MANDATORY: Start 0.5-1 mg/kg/day within 48h of birth — do not wait for genetics; accelerates collodion shedding",
            "EMOLLIENT SATURATION: 6+ applications/day; urea 5-10% cream; paraffin baths; avoid desiccation",
            "OPHTHALMOLOGY URGENT: Ectropion → corneal exposure → ulceration → blindness; lubricating drops + tape eyelids at night",
            "EAR CANAL CLEARANCE: ENT review for occluded canals; hearing testing at 6 months",
            "TEMPERATURE CONTROL: Air-conditioned environment 24h; cooling vest; no heated rooms; heat stroke = medical emergency",
            "SEPSIS PROPHYLAXIS: Regular skin cultures; low threshold for antibiotics; Staphylococcus aureus colonisation ubiquitous",
        ],
        "key_ddx": [
            "Lamellar Ichthyosis LI (TGM1 most common AR) — similar but NOT present as full harlequin armour at birth; less severe; TGM1 WES",
            "Congenital Ichthyosiform Erythroderma CIE (ALOX12B/ALOXE3) — erythroderma but NO harlequin plate armour",
            "Netherton syndrome (SPINK5) — ichthyosiform erythroderma + ichthyosis linearis circumflexa + atopy; SPINK5 WES",
            "Epidermolytic Ichthyosis (KRT1/KRT10) — blistering THEN hyperkeratosis; not collodion at birth",
        ],
        "seed": SEED_BASE,
    },
    # ── KRT1 — Epidermolytic Ichthyosis ──────────────────────────────────────
    {
        "gene": "KRT1",
        "protein": "Keratin 1 (Type II Cytoskeletal 1)",
        "alias": (
            "KRT1; OMIM gene 139350; Epidermolytic Ichthyosis EI #113800 (formerly BCIE); "
            "12q13.13; 644 aa; ~66 kDa; AD (haploinsufficiency or dominant-negative); "
            "Type II suprabasal keratin; KRT1/KRT10 obligate heterodimer in suprabasal epidermis; "
            "KRT1 LOF or missense → tonofilament aggregation → cytolysis → blistering; "
            "KRT10 allelic (same phenotype — EI); de novo mutations common (~50%); "
            "Epidermolytic hyperkeratosis (EHK) is the histological hallmark"
        ),
        "aa": "644 aa",
        "kDa": "~66 kDa",
        "locus": "12q13.13",
        "omim_gene": 139350,
        "omim_disease": 113800,
        "inheritance": "AD (dominant-negative or haploinsufficiency); de novo ~50%; familial AD with full penetrance",
        "gene_class": (
            "KRT1 encodes Keratin 1, the Type II suprabasal keratin partner of Keratin 10 (Type I). "
            "The KRT1/KRT10 obligate heterodimer forms intermediate filaments in suprabasal keratinocytes. "
            "These tonofilaments anchor to desmosomes and maintain suprabasal keratinocyte cohesion. "
            "KRT1 mutations (especially in helix initiation/termination motifs — L1 and L2 segments) → "
            "dominant-negative interference with filament assembly → intracellular tonofilament clumping → "
            "cytolysis → epidermolytic hyperkeratosis (EHK): "
            "Phase 1 (blistering): heat/friction → suprabasal cytolysis → bullae (± haemorrhagic); "
            "Phase 2 (hyperkeratosis): compensatory hyperproliferation → thick, warty hyperkeratosis; "
            "FLEXURAL PREDOMINANCE (axillae, groin, popliteal fossae, antecubital fossae) in most severe; "
            "ODOUR: retained shed cells + secondary bacterial colonisation → offensive smell (significant QoL impact); "
            "KRT1-specific feature: PALMOPLANTAR INVOLVEMENT (keratoderma) — KRT10 mutations spare palms; "
            "SECONDARY INFECTION: Staphylococcus aureus, Streptococcus pyogenes — regular cultures; "
            "RETINOIDS: acitretin 0.3-0.5 mg/kg/day reduces hyperkeratosis but increases fragility (balance); "
            "BLEACH BATHS: sodium hypochlorite 0.005% dilute 3x/week — Staphylococcus decolonisation"
        ),
        "phenotype": (
            "AT BIRTH: Blistering + erythroderma (often mistaken for EB); generalised bullae from friction; "
            "Collodion-like presentation in some. "
            "INFANCY: Generalised blistering with evolution to hyperkeratosis (warty/verrucous); "
            "flexural worst; palmoplantar keratoderma (KRT1-specific). "
            "ADOLESCENCE/ADULT: Blistering diminishes; hyperkeratosis predominates; characteristic malodour. "
            "HISTOLOGY: Epidermolytic hyperkeratosis (EHK) = tonofilament aggregation + perinuclear vacuolation + "
            "thickened stratum corneum — PATHOGNOMONIC for KRT1/KRT10 mutations. "
            "COMPLICATIONS: Recurrent Staphylococcus superinfection (eczematous flares); "
            "psychological morbidity from odour/appearance."
        ),
        "hallmark": (
            "BLISTERING AT BIRTH → EVOLUTION TO WARTY HYPERKERATOSIS with MALODOUR = EI (KRT1/KRT10); "
            "EPIDERMOLYTIC HYPERKERATOSIS on biopsy = pathognomonic; "
            "PALMOPLANTAR KERATODERMA in KRT1 (distinguishes from KRT10); "
            "DILUTE BLEACH BATHS 3x/week for decolonisation; retinoids reduce hyperkeratosis."
        ),
        "treatment_alerts": [
            "RETINOIDS: Acitretin 0.3-0.5 mg/kg/day reduces hyperkeratosis — but increases blistering risk at higher doses; balance carefully",
            "BLEACH BATHS: Sodium hypochlorite 0.005% 3x/week — essential for Staphylococcus decolonisation",
            "ANTIBIOTICS: Flucloxacillin prophylaxis during flares; culture-guided systemic therapy for infections",
            "EMOLLIENTS: Urea 10-20% creams for scale removal; apply to damp skin post-bath",
            "AVOID HEAT + FRICTION: air conditioning; loose cotton clothing; non-irritating laundry detergents",
            "KRT10 ALLELIC: Same management; KRT1 additionally has palmoplantar keratoderma — manage with keratolytics",
        ],
        "key_ddx": [
            "EB Simplex KRT5/KRT14 — blistering WITHOUT hyperkeratosis evolution; basal layer affected (not suprabasal)",
            "Lamellar Ichthyosis TGM1 — scaling WITHOUT blistering phase; no epidermolytic hyperkeratosis on biopsy",
            "Bullous Pemphigoid (acquired, elderly) — no genetic basis; immunofluorescence positive; ELISA for BP180/BP230",
            "Palmoplantar keratoderma Vorner type (KRT9) — PPK only without generalised blistering; KRT9 WES",
        ],
        "seed": SEED_BASE + 1,
    },
    # ── STS — X-linked Recessive Ichthyosis ──────────────────────────────────
    {
        "gene": "STS",
        "protein": "Steroid Sulfatase (Arylsulfatase C, Microsomal)",
        "alias": (
            "STS; OMIM gene 300747; X-linked Recessive Ichthyosis XLRI #308100; "
            "Xp22.31; 583 aa; ~63 kDa; XLR (hemizygous males affected; heterozygous females mosaic or normal); "
            "steroid sulfatase cleaves steroid sulfate esters; STS LOF → cholesterol sulfate accumulation → "
            "abnormal desquamation → scaling; most common hereditary ichthyosis in males (1:2000-1:6000 males); "
            "contiguous gene deletion Xp22.31 = CRITICAL: KAL1 (Kallmann syndrome) + STS + NLGN4X (autism); "
            "DELETION MUST BE EXCLUDED by array CGH/MLPA — point mutation vs deletion has critical phenotypic implications"
        ),
        "aa": "583 aa",
        "kDa": "~63 kDa",
        "locus": "Xp22.31",
        "omim_gene": 300747,
        "omim_disease": 308100,
        "inheritance": "XLR; hemizygous males fully affected; carrier females usually unaffected (occasional mild scaling)",
        "gene_class": (
            "STS (Steroid Sulfatase) is a microsomal enzyme expressed in keratinocytes, liver, adrenal, and placenta. "
            "It cleaves sulfate groups from: cholesterol sulfate, dehydroepiandrosterone sulfate (DHEA-S), "
            "estrone sulfate, and pregnenolone sulfate. "
            "STS LOF in keratinocytes → cholesterol sulfate accumulation → abnormal corneocyte retention → "
            "desquamation failure → visible fine-to-lamellar brown scaling; "
            "XDEL (contiguous Xp22.31 deletion) — affects multiple genes: "
            "KAL1 = Kallmann syndrome (anosmia + hypogonadotropic hypogonadism); "
            "STS = ichthyosis; NLGN4X = autism/intellectual disability; "
            "SEMPRE check MLPA/array CGH in XLRI males — do not assume point mutation without deletion excluded. "
            "OBSTETRICS: STS is essential for placental estrogen synthesis (dehydroepiandrosterone → estrogen). "
            "STS deficiency in pregnancy → PROLONGED LABOUR (estrogen deficiency) → "
            "high Caesarean section rate in STS-deficient pregnancies; "
            "amniotic fluid cholesterol sulfate elevated → first clue to diagnosis. "
            "EYE: Corneal opacity (diffuse, stromal) in 50% — ASYMPTOMATIC (does not impair vision); "
            "slit-lamp exam recommended but corneal opacities are a feature NOT a complication; "
            "TESTES: Undescended testes (cryptorchidism) 15-25% — orchidopexy before age 2; "
            "SCALP: seborrhoeic dermatitis-like scalp scaling common"
        ),
        "phenotype": (
            "SKIN: Fine, dark, adherent scaling; predominantly neck, extensor surfaces (elbows, knees, shins); "
            "IMPROVES IN SUMMER (humidity-dependent desquamation); worsens winter/dry climate. "
            "FACE: Usually spared (important DDx from other ichthyoses). "
            "EYE: Corneal opacities (diffuse stromal) 50% — asymptomatic; slit-lamp exam; "
            "GENITALIA: Undescended testes 15-25%; "
            "CONTIGUOUS DELETION (XDEL): Kallmann syndrome (anosmia, absent puberty, cryptorchidism) + "
            "ichthyosis + autism/ID; "
            "OBSTETRIC HISTORY: Maternal low estriol in pregnancy (prenatal STS deficiency = placental; "
            "suggests fetal STS LOF); prolonged labour. "
            "CARRIER FEMALES: Usually unaffected; occasional mild scaling; normal fertility."
        ),
        "hallmark": (
            "XLRI IN A MALE: dark, fine, adherent scaling (not blistering, not erythroderma) + "
            "ASYMPTOMATIC CORNEAL OPACITIES + UNDESCENDED TESTES + improves in summer; "
            "CRITICAL: MLPA/array CGH to exclude Xp22.31 contiguous deletion (KAL1 Kallmann syndrome); "
            "Cholesterol sulfate ELEVATED in serum/urine = confirmatory biochemical test."
        ),
        "treatment_alerts": [
            "DELETION EXCLUSION MANDATORY: MLPA or array CGH before assuming point mutation — KAL1/NLGN4X co-deletion = different management",
            "EMOLLIENTS: Urea 5-10% + ammonium lactate creams — improve desquamation; apply after bathing",
            "KERATOLYTICS: Propylene glycol 40-60% under occlusion for thick scale; urea 20-40% cream",
            "OPHTHALMOLOGY: Annual slit-lamp exam; corneal opacities asymptomatic but document progression",
            "CRYPTORCHIDISM: Orchidopexy before age 2 — fertility preservation; testicular tumour risk if untreated",
            "OBSTETRIC ALERT: Carrier mothers of STS-hemizygous fetuses — monitor for labour induction need (estrogen deficiency = prolonged labour)",
        ],
        "key_ddx": [
            "Lamellar Ichthyosis LI (TGM1) — AR; affects both sexes; more severe; flexural involvement; no corneal opacities",
            "Vulgaris Ichthyosis (FLG) — fine scaling; no corneal opacities; atopy association; AD; not sex-linked",
            "Kallmann + ichthyosis without STS mutation — rare; check ANOS1 (KAL1) separately",
            "Refsum disease (PHYH) — scaling + peripheral neuropathy + retinitis pigmentosa + ataxia; phytanic acid elevated",
        ],
        "seed": SEED_BASE + 2,
    },
    # ── COL7A1 — Dystrophic Epidermolysis Bullosa ─────────────────────────────
    {
        "gene": "COL7A1",
        "protein": "Collagen Type VII Alpha 1 Chain",
        "alias": (
            "COL7A1; OMIM gene 120120; DEB Dominant DDEB #131750 / DEB Recessive RDEB #226600; "
            "3p21.31; 2944 aa; ~290 kDa (monomer); AD = DDEB (haploinsufficiency/dominant-negative); "
            "AR = RDEB (biallelic null = Hallopeau-Siemens, most severe; biallelic missense = non-HS RDEB); "
            "Collagen VII is the major structural component of anchoring fibrils at the dermo-epidermal junction "
            "sublaminadensa; COL7A1 LOF → absent/reduced anchoring fibrils → sublaminadensa cleavage → blistering"
        ),
        "aa": "2944 aa",
        "kDa": "~290 kDa monomer (~700 kDa homotrimer)",
        "locus": "3p21.31",
        "omim_gene": 120120,
        "omim_disease": 226600,
        "inheritance": "AD = DDEB (one pathogenic variant, dominant-negative or haploinsufficiency); AR = RDEB (biallelic, most severe)",
        "gene_class": (
            "COL7A1 encodes Collagen Type VII, the unique collagen of anchoring fibrils. "
            "Anchoring fibrils are cruciform structures in the sublaminadensa zone of the basement membrane. "
            "They loop from the lamina densa around dermal collagen fibers and back — securing epidermis to dermis. "
            "COL7A1 LOF → reduced/absent anchoring fibrils → mechanical trauma → sublaminadensa blister → "
            "Dystrophic Epidermolysis Bullosa (DEB) spectrum: "
            "DDEB (AD): Usually milder; nail dystrophy + albopapuloid lesions; blistering predominantly acral; "
            "RDEB Hallopeau-Siemens (AR biallelic null): Most severe EB — generalised blistering from birth; "
            "RDEB COMPLICATIONS: (1) PSEUDOSYNDACTYLY (repeated blistering → fibrosis → digit fusion → "
            "mitten deformity — hand surgery required); "
            "(2) ESOPHAGEAL STRICTURES (blistering in mouth + esophagus → dysphagia → malnutrition → growth failure; "
            "regular esophageal dilation under anaesthesia); "
            "(3) SQUAMOUS CELL CARCINOMA (SCC) — LEADING CAUSE OF DEATH in RDEB: "
            "chronic wound environment → SCC risk 40-80% lifetime; first SCC often by age 25-35; "
            "SURVEILLANCE: monthly skin examination; annual skin biopsies of chronic wounds; "
            "(4) ANAEMIA: chronic inflammation + iron loss from wounds; regular transfusions; "
            "OLEOGEL-S10 (Filsuvez®, FDA April 2023): first approved topical therapy for DEB — "
            "birch bark extract (betulin); accelerates wound closure; reduces blister formation; "
            "GENE THERAPY: KB103 (beremagene geperpavec / B-VEC, FDA 2023) — herpes simplex viral vector topical"
        ),
        "phenotype": (
            "DDEB: Nail dystrophy (may be only feature in mildest); acral blistering; albopapuloid lesions. "
            "RDEB-HS (severe): "
            "SKIN: Generalised blistering from birth; healing with milia and atrophic scarring; "
            "HANDS: Pseudosyndactyly (digit fusion) by childhood → mitten hands; multiple surgeries. "
            "MOUTH/GI: Oral blistering → microstomia; esophageal strictures → dysphagia; malabsorption. "
            "MALIGNANCY: SCC risk 40-80% lifetime — most lethal complication; median SCC onset 34 years. "
            "ANAEMIA: Chronic; multifactorial (iron loss, inflammation, malabsorption). "
            "RENAL: Amyloidosis in advanced chronic disease. "
            "PSYCHOLOGICAL: Severe QoL impact; chronic pain; caregiver burden enormous."
        ),
        "hallmark": (
            "RDEB: BLISTERING + PSEUDOSYNDACTYLY (mitten hands) + ESOPHAGEAL STRICTURES = COL7A1 biallelic; "
            "DDEB: NAIL DYSTROPHY + ACRAL BLISTERING = COL7A1 dominant; "
            "SCC SURVEILLANCE MANDATORY: monthly skin exam + annual biopsy of chronic wounds; "
            "NON-ADHERENT DRESSINGS ONLY (Mepitel One, Mepilex, Urgotul); "
            "OLEOGEL-S10 (Filsuvez) FDA-approved topical therapy 2023."
        ),
        "treatment_alerts": [
            "NON-ADHERENT DRESSINGS MANDATORY: Mepitel One / Mepilex / Urgotul — adhesive dressings cause catastrophic secondary blistering",
            "ESOPHAGEAL DILATION: Regular endoscopic dilation under GA for strictures; dietitian + PEG tube if severe dysphagia",
            "SCC SURVEILLANCE: Monthly skin inspection + annual biopsy chronic wounds; PET-CT if SCC confirmed",
            "OLEOGEL-S10 (Filsuvez): FDA April 2023 — first approved topical EB therapy; apply to open wounds",
            "ANAEMIA MANAGEMENT: IV iron infusions; EPO if renal amyloidosis; transfusions; ferritin target >30",
            "GENE THERAPY: B-VEC (beremagene geperpavec, Vyjuvek) — FDA May 2023 topical gene therapy; COL7A1 HSV vector",
        ],
        "key_ddx": [
            "EB Simplex KRT5 — basal cleavage (not sublaminadensa); NO pseudosyndactyly; electron microscopy diagnostic",
            "Junctional EB LAMA3 — intralamina lucida cleavage; absent hemidesmosomes; NOT sublaminadensa; fatal neonatal",
            "Kindler syndrome FERMT1 — mixed cleavage planes + poikiloderma + photosensitivity; FERMT1 WES",
            "Bullous Pemphigoid (acquired) — elderly; no genetic basis; anti-BP180/BP230 autoantibodies positive",
        ],
        "seed": SEED_BASE + 3,
    },
    # ── LAMA3 — Junctional Epidermolysis Bullosa Herlitz ─────────────────────
    {
        "gene": "LAMA3",
        "protein": "Laminin Subunit Alpha 3",
        "alias": (
            "LAMA3; OMIM gene 600805; Junctional EB Herlitz JEB-H #226700; "
            "18q11.2; 3936 aa; ~340 kDa; AR (biallelic null = lethal JEB-H); "
            "laminin-332 (alpha3/beta3/gamma2 heterotrimer) = primary adhesion molecule of keratinocyte hemidesmosomes; "
            "LAMA3 LOF → absent laminin-332 → absent hemidesmosomes → intralamina lucida cleavage → JEB; "
            "JEB-H is the most severe EB subtype — LETHAL: most infants die by age 2 from sepsis/respiratory failure; "
            "LAMB3 most common JEB mutation (European); LAMC2 also causes JEB (allelic); "
            "no specific therapy exists; gene therapy trials ongoing (Phase I/II)"
        ),
        "aa": "3936 aa",
        "kDa": "~340 kDa",
        "locus": "18q11.2",
        "omim_gene": 600805,
        "omim_disease": 226700,
        "inheritance": "AR (biallelic null = JEB-H lethal; missense compound heterozygous = intermediate JEB non-Herlitz)",
        "gene_class": (
            "LAMA3 encodes the alpha3 chain of Laminin-332 (formerly Laminin-5), the primary extracellular matrix "
            "protein of keratinocyte hemidesmosomes. Laminin-332 heterotrimer (α3β3γ2) anchors basal "
            "keratinocytes to the basement membrane through interactions with: "
            "integrin α6β4 (on keratinocyte surface) → inner hemidesmosome plate → keratin 5/14 cytoskeleton. "
            "LAMA3 biallelic null → absent laminin-332 → absent hemidesmosomes → "
            "blister plane WITHIN lamina lucida (intralamina lucida) — diagnostic on electron microscopy. "
            "CLINICAL HALLMARKS OF JEB-H: "
            "(1) GENERALISED BLISTERING from birth — extensive, haemorrhagic, slow-healing; "
            "(2) GRANULATION TISSUE PATHOGNOMONIC — exuberant perioral, perinasal, periocular, tracheal, laryngeal; "
            "laryngeal granulation → hoarse cry → stridor → respiratory failure; "
            "(3) NAIL DYSTROPHY + NAIL LOSS — universal; "
            "(4) APLASIA CUTIS CONGENITA — areas of skin absent at birth (especially scalp); "
            "(5) URINARY TRACT: ureteric strictures → hydronephrosis; "
            "(6) PROGNOSIS: 80-90% die by age 2; surviving infants have continuous blistering; "
            "PALLIATIVE APPROACH: Many centres transition to palliative/comfort care in severe JEB-H; "
            "GENE THERAPY: LAMA3 ex vivo autologous epidermis (Holoclar-type approach) — Phase I/II trials"
        ),
        "phenotype": (
            "AT BIRTH: Extensive haemorrhagic blistering; aplasia cutis congenita (absent skin patches); "
            "GRANULATION TISSUE: perioral/perinasal/periocular (PATHOGNOMONIC for JEB); "
            "AIRWAY: hoarse cry → stridor → respiratory failure (tracheostomy may be needed); "
            "FEEDING: oral/esophageal involvement → severe feeding failure; NG tube; "
            "EYES: corneal erosions + conjunctival blistering; "
            "NAILS: dystrophic from birth → nail loss; "
            "TEETH: Enamel hypoplasia (laminin-332 required for enamel matrix); "
            "RENAL: Ureteric strictures; recurrent UTIs; "
            "PROGNOSIS: JEB-H — 80-90% mortality by age 2; surviving infants have continued blistering; "
            "JEB non-Herlitz (missense variants): milder, survival into adulthood with complications."
        ),
        "hallmark": (
            "JEB-H: GENERALISED BLISTERING FROM BIRTH + GRANULATION TISSUE (perioral/perinasal) PATHOGNOMONIC + "
            "STRIDOR (laryngeal granulation) + APLASIA CUTIS CONGENITA = LAMA3/LAMB3/LAMC2; "
            "LETHAL PROGNOSIS — initiate goals-of-care discussion immediately; "
            "Electron microscopy: INTRALAMINA LUCIDA cleavage = JEB; "
            "LAMB3 most common European mutation — test LAMB3 first if consanguinity absent."
        ),
        "treatment_alerts": [
            "GOALS OF CARE DISCUSSION: JEB-H has 80-90% mortality by age 2; immediate palliative care consultation",
            "AIRWAY EMERGENCY: Laryngeal granulation tissue → stridor → respiratory failure; early ENT + ICU",
            "NON-ADHERENT DRESSINGS: Mepitel One / Mepilex — adhesive dressings cause further skin loss",
            "FEEDING SUPPORT: NG tube early; PEG only if survival expected beyond 6-12 months",
            "GENE THERAPY REFERRAL: Phase I/II trials for LAMA3/LAMB3 ex vivo keratinocyte gene therapy — specialist referral",
            "LAMB3 FIRST: In European patients — LAMB3 is most common JEB gene; sequence LAMB3 before LAMA3 if WES not available",
        ],
        "key_ddx": [
            "Dystrophic EB COL7A1 — sublaminadensa cleavage (NOT intralamina lucida); no granulation tissue pattern; EM diagnostic",
            "EB Simplex KRT5 — basal cleavage within basal cells; no granulation tissue; NOT lethal; EM diagnostic",
            "Aplasia Cutis Congenita isolated — focal (not generalised blistering); no EB features; check JAM3/IKBKG",
            "Pemphigus neonatorum (maternal autoimmune) — maternal desmoglein antibodies; transient; immunofluorescence positive",
        ],
        "seed": SEED_BASE + 4,
    },
    # ── KRT5 — EB Simplex ─────────────────────────────────────────────────────
    {
        "gene": "KRT5",
        "protein": "Keratin 5 (Type II Cytoskeletal 5)",
        "alias": (
            "KRT5; OMIM gene 148040; EB Simplex EBS-Dowling-Meara DM-EBS #131760 / EBS-Generalized Severe; "
            "12q13.13; 590 aa; ~58 kDa; AD (dominant-negative); Type II basal keratin; "
            "KRT5/KRT14 obligate heterodimer in basal keratinocytes; "
            "KRT5 LOF → basal cytolysis within basal cells → intraepidermal blister → EBS; "
            "KRT14 allelic (same phenotype — same management); "
            "DM-EBS (Dowling-Meara) is most severe EBS subtype — herpetiform blisters, generalised; "
            "Weber-Cockayne WC-EBS is mildest — localised to hands/feet"
        ),
        "aa": "590 aa",
        "kDa": "~58 kDa",
        "locus": "12q13.13",
        "omim_gene": 148040,
        "omim_disease": 131760,
        "inheritance": "AD (dominant-negative); de novo common in DM-EBS; familial AD in WC-EBS",
        "gene_class": (
            "KRT5 encodes Keratin 5, the Type II basal keratin partner of Keratin 14 (Type I). "
            "The KRT5/KRT14 obligate heterodimer forms intermediate filaments in basal keratinocytes, "
            "anchoring to hemidesmosomes through plectin and BPAG1. "
            "KRT5 mutations in the helix initiation motif (L12 segment, coil 1A) → dominant-negative → "
            "tonofilament collapse → basal keratinocyte cytolysis → intraepidermal blister (not subepidermal). "
            "EB SIMPLEX SPECTRUM: "
            "DOWLING-MEARA (DM-EBS, most severe): generalised blistering; herpetiform cluster pattern; "
            "can be life-threatening in infancy; hyperkeratosis palms+soles in adolescence; "
            "GENERALISED SEVERE (formerly Koebner): generalised blistering but not herpetiform; "
            "LOCALIZED (formerly Weber-Cockayne): blistering confined to hands/feet; triggered by friction/heat; "
            "MOTTLED PIGMENTATION (KRT5 p.Pro25Leu): reticulate hyperpigmentation pattern — distinctive; "
            "TREATMENT: DIACEREIN (off-label): anti-IL-1α mechanism reduces blistering in DM-EBS by ~50%; "
            "COOLING: Cold water compresses/fans for DM-EBS — heat is a major trigger; "
            "WOUND CARE: Non-adherent dressings; lancing blisters with sterile needle (prevents expansion)"
        ),
        "phenotype": (
            "DM-EBS (most severe): Generalised herpetiform blisters from birth; haemorrhagic; "
            "palmoplantar hyperkeratosis develops in adolescence; nails dystrophic; mucous membranes mild. "
            "WC-EBS (mild): Blistering confined to hands/feet after friction/heat; heals without scarring; "
            "summer/childhood worse; often improves with age. "
            "GENERALISED SEVERE: Intermediate — generalised but not herpetiform. "
            "COMMON: All EBS subtypes — NO SCARRING (intraepidermal cleavage heals without fibrosis); "
            "NO PSEUDOSYNDACTYLY (distinguishes from DEB). "
            "EM: Cytolysis within basal cells (intraepidermal, above BMZ) = EBS hallmark."
        ),
        "hallmark": (
            "EBS: BLISTERING WITHOUT SCARRING (intraepidermal) = KRT5/KRT14; "
            "DM-EBS: herpetiform clusters, generalised, severe in infancy; "
            "WC-EBS: hands/feet only, friction-triggered, improves with age; "
            "NO PSEUDOSYNDACTYLY (distinguishes from DEB/COL7A1); "
            "COOLING is therapeutic (heat = major trigger); diacerein reduces blistering by 50%."
        ),
        "treatment_alerts": [
            "DIACEREIN 1% cream (off-label): Reduces blistering ~50% in DM-EBS by anti-IL-1α mechanism; apply daily",
            "COOLING THERAPY: Cool water, fans, air conditioning for DM-EBS — heat is the primary trigger",
            "LANCE BLISTERS STERILE: Prevents expansion and secondary infection; drain but preserve roof as biological dressing",
            "NON-ADHERENT DRESSINGS: Mepitel / Mepilex — adhesive dressings cause secondary blistering",
            "FOOTWEAR: Custom orthotics + padded socks for WC-EBS; avoid plastic/occlusive materials",
            "KRT14 ALLELIC: Identical management; p.Arg125Cys in KRT14 = severe EBS; p.Arg125His = moderate EBS",
        ],
        "key_ddx": [
            "Dystrophic EB COL7A1 — sublaminadensa cleavage → SCARRING + pseudosyndactyly; EM diagnostic",
            "Junctional EB LAMA3 — intralamina lucida; granulation tissue; lethal; EM diagnostic",
            "Pemphigus Vulgaris (acquired) — suprabasal acantholysis; anti-desmoglein-3 autoantibodies; no genetics",
            "Bullous Impetigo (Staphylococcal) — S. aureus exfoliatin toxin; culture positive; resolves with antibiotics",
        ],
        "seed": SEED_BASE + 5,
    },
    # ── ATP2C1 — Hailey-Hailey Disease ───────────────────────────────────────
    {
        "gene": "ATP2C1",
        "protein": "ATPase Secretory Pathway Ca2+ Transporting 1 (SERCA2C)",
        "alias": (
            "ATP2C1; OMIM gene 604384; Hailey-Hailey Disease HHD / Familial Benign Pemphigus #169600; "
            "3q22.1; 919 aa; ~97 kDa; AD (haploinsufficiency); SERCA2C = secretory pathway Ca2+/Mn2+ ATPase; "
            "ATP2C1 LOF → reduced Ca2+ uptake into Golgi/ER → desmosome assembly failure → suprabasal acantholysis; "
            "adult-onset blistering in flexures; chronic relapsing-remitting; "
            "NOT life-threatening but severe QoL impact from odour + chronicity; "
            "DARIER DISEASE (ATP2A2/SERCA2B) allelic disease — different gene, overlapping Ca2+ pathway"
        ),
        "aa": "919 aa",
        "kDa": "~97 kDa",
        "locus": "3q22.1",
        "omim_gene": 604384,
        "omim_disease": 169600,
        "inheritance": "AD (haploinsufficiency); familial with high penetrance; de novo rare; variable expressivity",
        "gene_class": (
            "ATP2C1 encodes SERCA2C (Secretory Pathway Ca2+ ATPase 2C), expressed in the trans-Golgi network. "
            "SERCA2C pumps Ca2+ and Mn2+ into the Golgi lumen and ER. "
            "Ca2+ in the Golgi is essential for: (1) Desmoglein-3 post-translational processing; "
            "(2) Corneodesmosin cleavage by Kallikrein serine proteases; (3) Profilaggrin processing. "
            "ATP2C1 LOF → reduced Golgi Ca2+ → impaired desmosome assembly → suprabasal acantholysis → "
            "HHD clinical features: "
            "INTERTRIGINOUS PREDILECTION: axillae, groin, submammary, perianal, neck — areas of friction + sweat; "
            "TRIGGERS: Heat, sweating, friction, secondary infection (Candida albicans, HSV, Staphylococcus); "
            "HISTOLOGY: Suprabasal acantholysis with 'dilapidated brick wall' appearance; "
            "SECONDARY INFECTION: Candida superinfection common → extends/worsens flares; "
            "MANAGEMENT STRATEGY: "
            "(1) TACROLIMUS 0.1% ointment → first-line topical (calcineurin inhibitor reduces inflammation); "
            "(2) BOTULINUM TOXIN A injections → reduces sweating → removes primary trigger → "
            "excellent response 6-12 months for axillary/inguinal disease; "
            "(3) RETINOIDS (acitretin): useful for refractory/widespread disease; "
            "(4) CO2 LASER: ablation of affected intertriginous skin — durable remission in some patients; "
            "(5) ANTI-HSV PROPHYLAXIS: aciclovir 400 mg BD if HSV superinfection is a recurrent trigger"
        ),
        "phenotype": (
            "ADULT ONSET: Typically 2nd-3rd decade (puberty rarely); "
            "LESIONS: Vesicles/bullae → erosions → crusted plaques → malodorous (foul smell from bacterial maceration); "
            "DISTRIBUTION: Axillae, groin, neck, submammary, perianal — INTERTRIGINOUS EXCLUSIVELY; "
            "COURSE: Chronic relapsing-remitting; summer/heat = worst; winter = spontaneous improvement; "
            "SECONDARY INFECTION: Candida (white discharge + satellite papules); HSV (herpetiform clusters — "
            "KAPOSI VARICELLIFORM ERUPTION if HSV on HHD = emergency → IV aciclovir); "
            "HISTOLOGY: Suprabasal acantholysis 'dilapidated brick wall' (tombstoning keratinocytes); "
            "NOT LIFE-THREATENING: but severe QoL impact — chronic pain, odour, social withdrawal."
        ),
        "hallmark": (
            "INTERTRIGINOUS EROSIONS WITH FOUL ODOUR + CHRONIC RELAPSING COURSE IN ADULT = HHD (ATP2C1); "
            "TRIGGERS: Heat, sweat, friction, Candida, HSV; "
            "BOTULINUM TOXIN A INJECTIONS: excellent response for axillary/inguinal disease (reduces sweating); "
            "HSV + HHD = KAPOSI VARICELLIFORM ERUPTION EMERGENCY → IV aciclovir immediately; "
            "DARIER DISEASE (ATP2A2) is allelic — check for follicular keratosis on trunk."
        ),
        "treatment_alerts": [
            "TACROLIMUS 0.1% OINTMENT FIRST-LINE: Apply to flexures BD; steroid-sparing; reduces inflammation without skin atrophy",
            "BOTULINUM TOXIN A INJECTION: Excellent for axillary/inguinal HHD — reduces sweating (primary trigger); repeat 6-12 monthly",
            "ANTIFUNGAL COVER: Candida superinfection ubiquitous; clotrimazole/fluconazole for Candida flares",
            "HSV EMERGENCY: Kaposi varicelliform eruption (KVE) = HHD + HSV = emergency; IV aciclovir 5mg/kg TDS; not topical alone",
            "ANTI-HSV PROPHYLAXIS: If HSV triggers recurrent HHD flares — aciclovir 400 mg BD prophylaxis",
            "CO2 LASER: Ablative laser to intertriginous skin — durable remission; refer to specialist laser unit",
        ],
        "key_ddx": [
            "Darier Disease ATP2A2 — follicular keratosis TRUNK + warty papules (NOT intertriginous erosions primarily); V and W lesion nails; ATP2A2 WES",
            "Pemphigus Vulgaris (acquired) — mucous membrane involvement prominent; anti-desmoglein-3/1 ELISA positive; older onset",
            "Grover disease (Transient Acantholytic Dermatosis) — trunk (NOT flexures); transient; elderly males; acquired",
            "Intertrigo (non-genetic) — no acantholysis on biopsy; resolves with antifungal/antiseptic; no family history",
        ],
        "seed": SEED_BASE + 6,
    },
    # ── EDA — X-linked Hypohidrotic Ectodermal Dysplasia ─────────────────────
    {
        "gene": "EDA",
        "protein": "Ectodysplasin-A (EDA-A1/A2 Isoforms)",
        "alias": (
            "EDA; OMIM gene 300451; X-linked Hypohidrotic Ectodermal Dysplasia XLHED #305100; "
            "Xq12-q13.1; 391 aa (EDA-A1 isoform); ~39 kDa; XLR (hemizygous males fully affected; "
            "heterozygous females may have mosaic hypohidrosis); "
            "EDA encodes EDA1 (ectodysplasin-A), a TNF family ligand; "
            "EDA1 binds EDAR receptor → NF-κB signaling → ectodermal placode development; "
            "XLHED triad: ANHIDROSIS (heat stroke risk = most dangerous) + HYPOTRICHOSIS + HYPODONTIA (conical); "
            "EDAR/EDARADD allelic (AD/AR forms of HED — same clinical phenotype); "
            "PRENATAL TREATMENT: intraamniotic EDA1 protein (EDX111) given in utero = FIRST disease-modifying therapy"
        ),
        "aa": "391 aa (EDA-A1); 389 aa (EDA-A2)",
        "kDa": "~39 kDa",
        "locus": "Xq12-q13.1",
        "omim_gene": 300451,
        "omim_disease": 305100,
        "inheritance": "XLR; hemizygous males fully affected; carrier females mosaic (segmental hypohidrosis in Blaschko lines)",
        "gene_class": (
            "EDA (ectodysplasin-A) encodes a TNF-family transmembrane ligand expressed on epithelial cells during "
            "embryonic development. EDA-A1 isoform binds EDAR (ectodysplasin receptor); EDA-A2 binds XEDAR. "
            "EDA/EDAR signaling via NF-κB pathway is essential for ECTODERMAL PLACODE induction — the embryonic "
            "primordium of: hair follicles, sweat glands, sebaceous glands, teeth, and mammary glands. "
            "EDA LOF → absent ectodermal placode development → XLHED triad: "
            "(1) ANHIDROSIS: absent eccrine sweat glands → HEAT STROKE RISK = MOST DANGEROUS FEATURE; "
            "core temperature cannot be regulated; exercise/fever/hot weather = life-threatening; "
            "COOLING VEST mandatory; avoid hot baths; outdoor heat monitoring; "
            "(2) HYPOTRICHOSIS: sparse/absent scalp hair (lanugo-like), absent eyebrows/lashes, absent body hair; "
            "(3) HYPODONTIA: absent teeth (oligodontia) or small conical teeth (characteristic shape); "
            "ADDITIONAL FEATURES: periorbital hyperpigmentation + wrinkled skin ('senile' appearance in infant); "
            "prominent supraorbital ridges; saddle nose; protuberant lips; dry mucous membranes; "
            "RESPIRATORY: absent/sparse mucous glands → recurrent respiratory infections; bronchiectasis; "
            "PRENATAL EDA1 PROTEIN THERAPY (EDX111): intraamniotic injection restores sweat gland development "
            "if given at 26-30 weeks gestation → partially reverses anhidrosis; "
            "CARRIER FEMALES: segmental hypohidrosis along Blaschko lines; mosaic hypodontia; careful examination"
        ),
        "phenotype": (
            "NEONATAL: Sparse/absent hair at birth; pyrexia without clear infection (anhidrosis); "
            "hyperthermia episodes → febrile seizures in infancy (anhidrosis misdiagnosed as febrile illness). "
            "INFANCY/CHILDHOOD: Characteristic facies (periorbital pigmentation, frontal bossing, saddle nose, "
            "protuberant lips, large ears); alopecia or sparse hair; missing/conical teeth; "
            "HEAT INTOLERANCE: The most dangerous feature — heat stroke in summer, exercise, fever; "
            "RESPIRATORY: Recurrent sinusitis/otitis/bronchitis (dry mucous membranes); "
            "DENTAL: Oligodontia (most teeth absent); conical incisors; dentures from age 2; "
            "implants only after jaw growth complete (adulthood); "
            "FEMALE CARRIERS: Segmental sweating defect (Blaschko lines); mild hypodontia; "
            "XLHED males may have failure-to-thrive infancy (hyperthermia + poor feeding)."
        ),
        "hallmark": (
            "XLHED TRIAD: ANHIDROSIS + HYPOTRICHOSIS + CONICAL HYPODONTIA = EDA until proven otherwise; "
            "HEAT STROKE IS THE MOST DANGEROUS FEATURE — cooling vest mandatory from infancy; "
            "PERIORBITAL PIGMENTATION + SADDLE NOSE + FRONTAL BOSSING = characteristic facies; "
            "PRENATAL TREATMENT (EDX111 intraamniotic): first disease-modifying therapy if fetal XLHED diagnosed; "
            "EDAR/EDARADD allelic — same phenotype, autosomal forms."
        ),
        "treatment_alerts": [
            "HEAT STROKE PREVENTION: Cooling vest mandatory; air-conditioned home/car; outdoor temperature monitoring; NO hot baths",
            "DENTAL REHABILITATION: Dentures from age 2-3 years; dental implants after jaw growth complete (adulthood); orthodontic referral",
            "PRENATAL EDA1 THERAPY (EDX111): If fetal XLHED diagnosed by prenatal genetics — intraamniotic injection at 26-30 weeks gestation",
            "RESPIRATORY PROPHYLAXIS: N-acetylcysteine mucolytics; aggressive treatment of respiratory infections; pulmonary function monitoring",
            "SKIN MOISTURISATION: Emollients for dry skin; humidifiers in home; avoid drying soaps",
            "CARRIER FEMALES: Mosaic features — check segmental hypohidrosis (starch-iodine test); dental X-ray for oligodontia; IODP referral",
        ],
        "key_ddx": [
            "EDAR-related HED (AD/AR) — same phenotype; EDAR sequencing; autosomal inheritance; same management",
            "Hidrotic Ectodermal Dysplasia Clouston (GJB6) — hair/nail dystrophy WITHOUT significant anhidrosis; palmoplantar keratoderma; GJB6 WES",
            "Ankyloblepharon-Ectodermal defects-Cleft lip/palate AEC (TP63) — cleft palate + ankyloblepharon + ectodermal features; TP63 dominant",
            "Rapp-Hodgkin syndrome (TP63) — HED features + cleft lip/palate + TP63 mutation; allelic to AEC",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _make_patient(gene_info: dict, idx: int) -> dict:
    """Generate a realistic synthetic patient for genodermatoses analysis."""
    rng = random.Random(gene_info["seed"] * 1000 + idx)
    gene = gene_info["gene"]

    # Age at diagnosis (years): varies by condition
    age_dx_map = {
        "ABCA12": (0.0, 0.08),   # birth to 1 month
        "KRT1":   (0.0, 0.5),    # birth to 6 months
        "STS":    (0.5, 5.0),    # infancy to early childhood
        "COL7A1": (0.0, 1.0),    # birth to 1 year
        "LAMA3":  (0.0, 0.5),    # birth to 6 months
        "KRT5":   (0.0, 1.0),    # birth to 1 year
        "ATP2C1": (18, 45),      # adult onset
        "EDA":    (0.0, 2.0),    # birth to 2 years
    }
    age_range = age_dx_map.get(gene, (1, 10))
    age_dx = round(rng.uniform(*age_range), 2)

    # Sex: EDA/STS X-linked → mostly males
    if gene in ("EDA", "STS"):
        sex = "M"
    else:
        sex = rng.choice(["M", "F"])

    # Gene-specific clinical features
    if gene == "ABCA12":
        severity = rng.choice(["Harlequin (severe)", "Harlequin (severe)", "Harlequin (severe)", "Lamellar (moderate)"])
        collodion_membrane = True
        ectropion = rng.random() < 0.85
        eclabium = rng.random() < 0.80
        hyperthermia_episode = rng.random() < 0.70
        retinoid_started = rng.random() < 0.90
        alive_12m = rng.random() < 0.72
        return {
            "gene": gene, "patient_id": f"{gene}-{idx+1:03d}",
            "age_dx_yrs": age_dx, "sex": sex,
            "severity": severity,
            "collodion_membrane": collodion_membrane,
            "ectropion": ectropion,
            "eclabium": eclabium,
            "hyperthermia_episode": hyperthermia_episode,
            "retinoid_started": retinoid_started,
            "alive_12m": alive_12m,
            "variant_class": rng.choice(["Biallelic null", "Biallelic null", "Missense/null compound het"]),
            "treatment": "Acitretin + emollient protocol",
        }

    elif gene == "KRT1":
        severity = rng.choice(["Mild (PPK only)", "Moderate (generalised)", "Severe (blistering + HK)"])
        ppk = rng.random() < 0.85
        flexural_hk = rng.random() < 0.70
        staph_infection = rng.random() < 0.60
        retinoid_use = rng.random() < 0.55
        bleach_baths = rng.random() < 0.65
        return {
            "gene": gene, "patient_id": f"{gene}-{idx+1:03d}",
            "age_dx_yrs": age_dx, "sex": sex,
            "severity": severity,
            "ppk": ppk,
            "flexural_hyperkeratosis": flexural_hk,
            "staphylococcus_infection": staph_infection,
            "retinoid_use": retinoid_use,
            "bleach_baths": bleach_baths,
            "variant_class": rng.choice(["Helix initiation motif", "Helix termination motif", "Other domain"]),
            "treatment": "Retinoids + bleach baths + emollients",
        }

    elif gene == "STS":
        xdel = rng.random() < 0.15          # ~15% have contiguous deletion
        kallmann = xdel and rng.random() < 0.80
        corneal_opacity = rng.random() < 0.50
        cryptorchidism = sex == "M" and rng.random() < 0.22
        scaling_summer_improvement = rng.random() < 0.85
        return {
            "gene": gene, "patient_id": f"{gene}-{idx+1:03d}",
            "age_dx_yrs": age_dx, "sex": sex,
            "contiguous_xdel": xdel,
            "kallmann_syndrome": kallmann,
            "corneal_opacity": corneal_opacity,
            "cryptorchidism": cryptorchidism,
            "scaling_summer_improvement": scaling_summer_improvement,
            "variant_class": rng.choice(["Xp22.31 deletion (MLPA)", "Point mutation/small indel", "Xp22.31 deletion (MLPA)"]),
            "treatment": "Emollients + urea 10% cream",
        }

    elif gene == "COL7A1":
        subtype = rng.choice(["RDEB-HS (AR biallelic null)", "RDEB-HS (AR biallelic null)", "RDEB non-HS (AR missense)", "DDEB (AD)"])
        pseudosyndactyly = "RDEB" in subtype and rng.random() < 0.65
        esophageal_stricture = "RDEB" in subtype and rng.random() < 0.55
        scc_dx = "RDEB-HS" in subtype and rng.random() < 0.35
        oleogel_use = rng.random() < 0.45
        non_adherent_dressings = rng.random() < 0.95
        return {
            "gene": gene, "patient_id": f"{gene}-{idx+1:03d}",
            "age_dx_yrs": age_dx, "sex": sex,
            "subtype": subtype,
            "pseudosyndactyly": pseudosyndactyly,
            "esophageal_stricture": esophageal_stricture,
            "scc_diagnosed": scc_dx,
            "oleogel_s10_use": oleogel_use,
            "non_adherent_dressings": non_adherent_dressings,
            "variant_class": rng.choice(["Biallelic null (RDEB-HS)", "Compound het missense/null (RDEB)", "AD dominant-negative (DDEB)"]),
            "treatment": "Wound dressings + Filsuvez (Oleogel-S10) + esophageal dilation",
        }

    elif gene == "LAMA3":
        subtype = rng.choice(["JEB-Herlitz (null/null)", "JEB-Herlitz (null/null)", "JEB non-Herlitz (missense/null)"])
        granulation_tissue = "Herlitz" in subtype and rng.random() < 0.90
        respiratory_failure = "Herlitz" in subtype and rng.random() < 0.60
        survived_2yr = "non-Herlitz" in subtype or rng.random() < 0.20
        gene_mutated = rng.choice(["LAMA3", "LAMB3", "LAMB3", "LAMC2"])  # LAMB3 most common
        return {
            "gene": gene, "patient_id": f"{gene}-{idx+1:03d}",
            "age_dx_yrs": age_dx, "sex": sex,
            "jeb_subtype": subtype,
            "granulation_tissue": granulation_tissue,
            "respiratory_failure": respiratory_failure,
            "survived_2yr": survived_2yr,
            "actual_gene_mutated": gene_mutated,
            "palliative_care_initiated": "Herlitz" in subtype,
            "variant_class": rng.choice(["Biallelic null (JEB-H)", "Null/missense compound het (JEB non-H)"]),
            "treatment": "Palliative/supportive care + non-adherent dressings",
        }

    elif gene == "KRT5":
        subtype = rng.choice(["EBS-Dowling-Meara (severe)", "EBS-Generalized Severe", "EBS-Localized (Weber-Cockayne)"])
        herpetiform = "Dowling-Meara" in subtype
        cooling_therapy = "Dowling-Meara" in subtype and rng.random() < 0.80
        diacerein_use = rng.random() < 0.40
        ppk_adolescent = rng.random() < 0.50
        krt_gene = rng.choice(["KRT5", "KRT5", "KRT14"])
        return {
            "gene": gene, "patient_id": f"{gene}-{idx+1:03d}",
            "age_dx_yrs": age_dx, "sex": sex,
            "ebs_subtype": subtype,
            "herpetiform_clusters": herpetiform,
            "cooling_therapy": cooling_therapy,
            "diacerein_cream_use": diacerein_use,
            "ppk_adolescent": ppk_adolescent,
            "actual_gene": krt_gene,
            "variant_class": rng.choice(["Helix initiation 1A (severe)", "Helix termination 2B", "Other rod domain"]),
            "treatment": "Cooling + non-adherent dressings + diacerein 1%",
        }

    elif gene == "ATP2C1":
        flexural_sites = rng.sample(["axillae", "groin", "submammary", "perianal", "neck"], k=rng.randint(2, 5))
        candida_superinfection = rng.random() < 0.65
        hsv_superinfection = rng.random() < 0.30
        kaposi_varicelliform = hsv_superinfection and rng.random() < 0.35
        botox_use = rng.random() < 0.55
        tacrolimus_use = rng.random() < 0.70
        summer_flare = rng.random() < 0.85
        return {
            "gene": gene, "patient_id": f"{gene}-{idx+1:03d}",
            "age_dx_yrs": age_dx, "sex": sex,
            "flexural_sites": flexural_sites,
            "candida_superinfection": candida_superinfection,
            "hsv_superinfection": hsv_superinfection,
            "kaposi_varicelliform_eruption": kaposi_varicelliform,
            "botulinum_toxin_use": botox_use,
            "tacrolimus_topical_use": tacrolimus_use,
            "summer_flare": summer_flare,
            "variant_class": rng.choice(["Haploinsufficiency (PTC/frameshift)", "Missense (SERCA2C domain)", "Splice site"]),
            "treatment": "Tacrolimus 0.1% + botulinum toxin A injections + antifungals",
        }

    elif gene == "EDA":
        anhidrosis_severe = rng.random() < 0.90
        heat_stroke_episode = rng.random() < 0.45
        n_teeth_absent = rng.randint(4, 28)
        conical_teeth = rng.random() < 0.80
        respiratory_infections = rng.random() < 0.70
        cooling_vest = rng.random() < 0.65
        prenatal_treatment = rng.random() < 0.10  # rare — new therapy
        return {
            "gene": gene, "patient_id": f"{gene}-{idx+1:03d}",
            "age_dx_yrs": age_dx, "sex": sex,
            "anhidrosis_severe": anhidrosis_severe,
            "heat_stroke_episode": heat_stroke_episode,
            "n_teeth_absent": n_teeth_absent,
            "conical_teeth": conical_teeth,
            "respiratory_infections_per_year": rng.randint(2, 8),
            "cooling_vest_prescribed": cooling_vest,
            "prenatal_eda1_treatment": prenatal_treatment,
            "variant_class": rng.choice(["Hemizygous null (XLR)", "Hemizygous missense TNF-domain", "Xq12 deletion"]),
            "treatment": "Cooling vest + dental rehabilitation + respiratory prophylaxis",
        }

    return {"gene": gene, "patient_id": f"{gene}-{idx+1:03d}", "age_dx_yrs": age_dx, "sex": sex}


# Build all cohorts once at import time
_ALL_COHORTS: dict = {}
for _ginfo in GENO_GENES:
    _ALL_COHORTS[_ginfo["gene"]] = [_make_patient(_ginfo, i) for i in range(40)]


def _pct(pts: list, key: str) -> float:
    if not pts:
        return 0.0
    return round(sum(1 for p in pts if p.get(key)) / len(pts) * 100, 1)


def _avg(pts: list, key: str) -> float:
    vals = [p[key] for p in pts if key in p and isinstance(p[key], (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def get_overview() -> dict:
    all_pts = [p for pts in _ALL_COHORTS.values() for p in pts]
    genes = [g["gene"] for g in GENO_GENES]

    # Aggregate clinical stats across conditions
    return {
        "atlas_name": "Genodermatoses Atlas",
        "atlas_subtitle": (
            "Complete 8-Gene Hereditary Skin Disorder Reference — "
            "ABCA12 · KRT1 · STS · COL7A1 · LAMA3 · KRT5 · ATP2C1 · EDA"
        ),
        "n_genes": 8,
        "n_patients": len(all_pts),
        "seeds": "1278–1285",
        "genes": genes,
        "description": (
            "This atlas covers eight primary hereditary skin disorders in clinical genetics. "
            "Ichthyosis spectrum: ABCA12 (Harlequin Ichthyosis — most severe; collodion membrane at birth; "
            "ectropion + eclabium PATHOGNOMONIC; acitretin mandatory within 48h of birth), "
            "KRT1 (Epidermolytic Ichthyosis — AD; blistering at birth → warty hyperkeratosis with malodour; "
            "epidermolytic hyperkeratosis on biopsy PATHOGNOMONIC; bleach baths + retinoids), and "
            "STS (X-linked Recessive Ichthyosis — most common hereditary ichthyosis in males; "
            "cholesterol sulfate accumulation; MLPA mandatory to exclude Xp22.31 contiguous deletion "
            "= Kallmann syndrome co-deletion; corneal opacities 50% asymptomatic). "
            "Epidermolysis Bullosa spectrum: COL7A1 (Dystrophic EB — RDEB most severe; "
            "pseudosyndactyly + esophageal strictures + SCC 40-80% lifetime = leading cause death; "
            "Oleogel-S10 FDA 2023 first approved topical therapy), "
            "LAMA3 (Junctional EB Herlitz — lethal; granulation tissue PATHOGNOMONIC; "
            "80-90% mortality by age 2; LAMB3 most common JEB mutation), and "
            "KRT5 (EB Simplex — blistering without scarring; Dowling-Meara = most severe EBS; "
            "cooling therapy + diacerein 1% cream reduce blistering). "
            "Other hereditary skin disorders: ATP2C1 (Hailey-Hailey Disease — AD; "
            "intertriginous erosions with malodour; botulinum toxin A injections = excellent for "
            "axillary/inguinal disease; HSV + HHD = Kaposi varicelliform eruption emergency), "
            "EDA (X-linked Hypohidrotic Ectodermal Dysplasia — XLR; anhidrosis + hypotrichosis + "
            "conical hypodontia; heat stroke = most dangerous feature; cooling vest mandatory; "
            "prenatal EDX111 intraamniotic injection = first disease-modifying therapy)."
        ),
        "aggregate_clinical": {
            "collodion_membrane_pct":      round(_pct(all_pts, "collodion_membrane"), 1),
            "ectropion_pct":               round(_pct(all_pts, "ectropion"), 1),
            "blistering_pct":              round(sum(1 for p in all_pts if p.get("collodion_membrane") or p.get("herpetiform_clusters") is not None or p.get("jeb_subtype") or p.get("subtype") and "EB" in str(p.get("subtype", ""))) / len(all_pts) * 100, 1),
            "hyperkeratosis_pct":          round(_pct(all_pts, "flexural_hyperkeratosis"), 1),
            "pseudosyndactyly_pct":        round(_pct(all_pts, "pseudosyndactyly"), 1),
            "esophageal_stricture_pct":    round(_pct(all_pts, "esophageal_stricture"), 1),
            "scc_pct":                     round(_pct(all_pts, "scc_diagnosed"), 1),
            "anhidrosis_pct":              round(_pct(all_pts, "anhidrosis_severe"), 1),
            "heat_stroke_pct":             round(_pct(all_pts, "heat_stroke_episode") + _pct(all_pts, "hyperthermia_episode"), 1),
            "candida_infection_pct":       round(_pct(all_pts, "candida_superinfection"), 1),
            "granulation_tissue_pct":      round(_pct(all_pts, "granulation_tissue"), 1),
            "corneal_opacity_pct":         round(_pct(all_pts, "corneal_opacity"), 1),
            "retinoid_use_pct":            round(_pct(all_pts, "retinoid_started") + _pct(all_pts, "retinoid_use"), 1),
            "botox_pct":                   round(_pct(all_pts, "botulinum_toxin_use"), 1),
        },
        "drug_alerts": [
            {
                "title": "ABCA12 (Harlequin Ichthyosis) — ACITRETIN MANDATORY WITHIN 48H OF BIRTH",
                "body": (
                    "Acitretin must be started within 48 hours of birth in Harlequin Ichthyosis — "
                    "do NOT wait for genetic confirmation. Accelerates shedding of collodion plates. "
                    "Dose: 0.5-1 mg/kg/day. Emollient protocol 6+ applications/day simultaneously. "
                    "Temperature control 24h in air-conditioned environment. "
                    "Ophthalmology urgent — ectropion → corneal ulceration → blindness if untreated."
                ),
            },
            {
                "title": "COL7A1 (RDEB) — NON-ADHERENT DRESSINGS MANDATORY; ADHESIVE = CATASTROPHIC HARM",
                "body": (
                    "In Epidermolysis Bullosa (all types), ADHESIVE DRESSINGS cause secondary skin loss — "
                    "NEVER use standard adhesive wound dressings. Use exclusively non-adherent: "
                    "Mepitel One, Mepilex Transfer, Urgotul. "
                    "SCC surveillance mandatory in RDEB: monthly skin examination + annual biopsy of chronic wounds. "
                    "Oleogel-S10 (Filsuvez, FDA April 2023) approved for DEB — apply to open wounds."
                ),
            },
            {
                "title": "LAMA3 (JEB-Herlitz) — LETHAL PROGNOSIS; GOALS-OF-CARE DISCUSSION IMMEDIATELY",
                "body": (
                    "Junctional EB Herlitz (biallelic null LAMA3/LAMB3/LAMC2) has 80-90% mortality by age 2. "
                    "Initiate immediate goals-of-care and palliative care discussion with family. "
                    "Granulation tissue (perioral, perinasal, laryngeal) is PATHOGNOMONIC for JEB. "
                    "Laryngeal granulation → stridor → respiratory failure = emergency. "
                    "LAMB3 is most common JEB gene in European patients — test first if WES unavailable."
                ),
            },
            {
                "title": "ATP2C1 (Hailey-Hailey) + HSV = KAPOSI VARICELLIFORM ERUPTION EMERGENCY",
                "body": (
                    "When HSV superinfects Hailey-Hailey Disease (or other barrier-disrupted genodermatosis), "
                    "Kaposi Varicelliform Eruption (KVE) can occur — widespread haemorrhagic HSV-laden erosions. "
                    "KVE requires IMMEDIATE IV aciclovir 5 mg/kg TDS (NOT topical alone — systemic mandatory). "
                    "Patients with recurrent HSV-triggered HHD flares → aciclovir 400 mg BD prophylaxis. "
                    "Botulinum toxin A injections reduce sweating → remove primary HHD trigger → 6-12 month remission."
                ),
            },
            {
                "title": "EDA (XLHED) — ANHIDROSIS = HEAT STROKE RISK; COOLING VEST MANDATORY FROM INFANCY",
                "body": (
                    "X-linked Hypohidrotic Ectodermal Dysplasia (EDA): the most dangerous feature is ANHIDROSIS. "
                    "Absent eccrine sweat glands → cannot regulate core body temperature → heat stroke. "
                    "COOLING VEST mandatory from infancy. Air-conditioned home and car essential. "
                    "Avoid hot baths, outdoor heat, vigorous exercise in warm weather. "
                    "Febrile illness → emergency cooling measures (wet towels, fans, cold packs to neck/groin). "
                    "Prenatal EDX111 (intraamniotic EDA1 protein) = first disease-modifying therapy if fetal XLHED known."
                ),
            },
            {
                "title": "STS (XLRI) — MLPA/ARRAY CGH MANDATORY TO EXCLUDE Xp22.31 CONTIGUOUS DELETION (KALLMANN)",
                "body": (
                    "X-linked Recessive Ichthyosis (STS): do NOT assume point mutation without excluding Xp22.31 deletion. "
                    "Contiguous deletion of Xp22.31 co-deletes KAL1 (Kallmann syndrome: anosmia + hypogonadotropic hypogonadism) "
                    "+ NLGN4X (autism spectrum disorder/intellectual disability) alongside STS. "
                    "Management and prognosis differ fundamentally from isolated STS point mutation. "
                    "MLPA or chromosomal microarray (array CGH) is mandatory in all XLRI males — "
                    "sequencing alone does NOT detect deletions."
                ),
            },
        ],
        "clinical_pearls": [
            "ABCA12: Harlequin armour at birth + ectropion + eclabium = ABCA12 HI; acitretin within 48h mandatory.",
            "KRT1: Blistering at birth → evolution to warty hyperkeratosis with malodour = EI; PPK distinguishes KRT1 from KRT10.",
            "STS: Fine dark scaling in males (improves in summer) + asymptomatic corneal opacities + undescended testes = XLRI; MLPA mandatory.",
            "COL7A1: Pseudosyndactyly + esophageal strictures + SCC risk 40-80% = RDEB; non-adherent dressings only.",
            "LAMA3: Generalised blistering + GRANULATION TISSUE (perioral/perinasal) = JEB; LETHAL — LAMB3 most common mutation.",
            "KRT5: Blistering WITHOUT scarring = EBS; DM-EBS herpetiform clusters (most severe); cooling therapy + diacerein 1% cream.",
            "ATP2C1: Intertriginous erosions with malodour in adult = HHD; botulinum toxin A injections for axillary/inguinal disease.",
            "EDA: Anhidrosis + hypotrichosis + conical hypodontia = XLHED; heat stroke = most dangerous; cooling vest mandatory from infancy.",
            "ALL EB TYPES: Non-adherent dressings (Mepitel One/Mepilex) — adhesive dressings ABSOLUTELY contraindicated.",
            "CASCADE TESTING: First-degree relatives of all genodermatosis genes; carrier females EDA/STS require mosaic examination.",
        ],
    }


def get_breakdown() -> dict:
    out: dict = {}
    for ginfo in GENO_GENES:
        gene = ginfo["gene"]
        pts = _ALL_COHORTS[gene]
        # Build per-gene stats dict
        stats: dict = {
            "n": len(pts),
            "sex_m_pct": round(sum(1 for p in pts if p.get("sex") == "M") / len(pts) * 100, 1),
        }
        # Gene-specific computed stats
        if gene == "ABCA12":
            stats.update({
                "collodion_membrane_pct": _pct(pts, "collodion_membrane"),
                "ectropion_pct": _pct(pts, "ectropion"),
                "eclabium_pct": _pct(pts, "eclabium"),
                "hyperthermia_pct": _pct(pts, "hyperthermia_episode"),
                "retinoid_started_pct": _pct(pts, "retinoid_started"),
                "alive_12m_pct": _pct(pts, "alive_12m"),
            })
        elif gene == "KRT1":
            stats.update({
                "ppk_pct": _pct(pts, "ppk"),
                "flexural_hk_pct": _pct(pts, "flexural_hyperkeratosis"),
                "staph_infection_pct": _pct(pts, "staphylococcus_infection"),
                "retinoid_pct": _pct(pts, "retinoid_use"),
                "bleach_baths_pct": _pct(pts, "bleach_baths"),
            })
        elif gene == "STS":
            stats.update({
                "xdel_pct": _pct(pts, "contiguous_xdel"),
                "kallmann_pct": _pct(pts, "kallmann_syndrome"),
                "corneal_opacity_pct": _pct(pts, "corneal_opacity"),
                "cryptorchidism_pct": _pct(pts, "cryptorchidism"),
                "summer_improvement_pct": _pct(pts, "scaling_summer_improvement"),
            })
        elif gene == "COL7A1":
            stats.update({
                "pseudosyndactyly_pct": _pct(pts, "pseudosyndactyly"),
                "esophageal_stricture_pct": _pct(pts, "esophageal_stricture"),
                "scc_pct": _pct(pts, "scc_diagnosed"),
                "oleogel_use_pct": _pct(pts, "oleogel_s10_use"),
                "non_adherent_dressings_pct": _pct(pts, "non_adherent_dressings"),
            })
        elif gene == "LAMA3":
            stats.update({
                "granulation_tissue_pct": _pct(pts, "granulation_tissue"),
                "respiratory_failure_pct": _pct(pts, "respiratory_failure"),
                "survived_2yr_pct": _pct(pts, "survived_2yr"),
                "palliative_care_pct": _pct(pts, "palliative_care_initiated"),
            })
        elif gene == "KRT5":
            stats.update({
                "herpetiform_pct": _pct(pts, "herpetiform_clusters"),
                "cooling_therapy_pct": _pct(pts, "cooling_therapy"),
                "diacerein_pct": _pct(pts, "diacerein_cream_use"),
                "ppk_adolescent_pct": _pct(pts, "ppk_adolescent"),
            })
        elif gene == "ATP2C1":
            stats.update({
                "candida_pct": _pct(pts, "candida_superinfection"),
                "hsv_pct": _pct(pts, "hsv_superinfection"),
                "kve_pct": _pct(pts, "kaposi_varicelliform_eruption"),
                "botox_pct": _pct(pts, "botulinum_toxin_use"),
                "tacrolimus_pct": _pct(pts, "tacrolimus_topical_use"),
                "summer_flare_pct": _pct(pts, "summer_flare"),
            })
        elif gene == "EDA":
            stats.update({
                "anhidrosis_pct": _pct(pts, "anhidrosis_severe"),
                "heat_stroke_pct": _pct(pts, "heat_stroke_episode"),
                "conical_teeth_pct": _pct(pts, "conical_teeth"),
                "avg_teeth_absent": _avg(pts, "n_teeth_absent"),
                "respiratory_infection_pct": _pct(pts, "respiratory_infections"),
                "cooling_vest_pct": _pct(pts, "cooling_vest_prescribed"),
                "prenatal_rx_pct": _pct(pts, "prenatal_eda1_treatment"),
            })

        out[gene] = {
            "gene": gene,
            "protein": ginfo["protein"],
            "aa": ginfo["aa"],
            "kDa": ginfo["kDa"],
            "locus": ginfo["locus"],
            "omim_gene": ginfo["omim_gene"],
            "omim_disease": ginfo["omim_disease"],
            "inheritance": ginfo["inheritance"],
            "gene_class": ginfo["gene_class"],
            "phenotype": ginfo["phenotype"],
            "hallmark": ginfo["hallmark"],
            "treatment_alerts": ginfo["treatment_alerts"],
            "key_ddx": ginfo["key_ddx"],
            "cohort_stats": stats,
        }
    return {"breakdown": out}


def get_definitions() -> dict:
    return {
        "definitions": {
            "Genodermatoses": (
                "Genodermatoses are hereditary skin disorders caused by single-gene mutations. "
                "They encompass ichthyoses (scaling disorders), epidermolysis bullosa (EB, blistering disorders), "
                "ectodermal dysplasias, and other hereditary skin conditions. "
                "Classification by cleavage level (for EB): intraepidermal (EBS), intralamina lucida (JEB), "
                "sublaminadensa (DEB) — determined by electron microscopy and immunofluorescence antigen mapping."
            ),
            "Ichthyosis": (
                "Ichthyosis refers to a group of disorders of skin desquamation/cornification causing scaling. "
                "ABCA12 (Harlequin Ichthyosis — most severe, AR); KRT1/KRT10 (Epidermolytic Ichthyosis — AD, blistering); "
                "STS (X-linked Recessive Ichthyosis — most common in males, XLR); "
                "TGM1 (Lamellar Ichthyosis type 1 — AR, non-bullous); ALOX12B/ALOXE3 (CIE — AR)."
            ),
            "Epidermolysis Bullosa": (
                "EB = mechanobullous disorders caused by mutations in structural proteins of skin adherence. "
                "EB Simplex (EBS): KRT5/KRT14 mutations → intraepidermal blister (basal cytolysis) → no scarring. "
                "Junctional EB (JEB): LAMA3/LAMB3/LAMC2/COL17A1 → intralamina lucida blister → granulation tissue. "
                "Dystrophic EB (DEB): COL7A1 → sublaminadensa blister → scarring, pseudosyndactyly, SCC. "
                "Kindler EB: FERMT1 → mixed cleavage + poikiloderma + photosensitivity. "
                "Electron microscopy + immunofluorescence antigen mapping (IFM) = definitive cleavage plane diagnosis."
            ),
            "Harlequin Ichthyosis ABCA12": (
                "Harlequin Ichthyosis (HI) caused by biallelic ABCA12 null variants. "
                "ABCA12 lipid transporter in lamellar granules → absent intercellular lipid lamellae → "
                "catastrophic barrier failure → hard, fissured, armour-like Collodion membrane at birth. "
                "KEY RULES: (1) Acitretin WITHIN 48h — do not wait for genetics; "
                "(2) Ectropion urgent ophthalmology — corneal ulceration risk; "
                "(3) Temperature-controlled environment 24h — heat stroke risk; "
                "(4) Emollient saturation 6+ applications/day; (5) Sepsis prophylaxis from fissures."
            ),
            "STS Xp22.31 Contiguous Deletion": (
                "STS gene at Xp22.31 lies within a deletion-prone region containing KAL1 (ANOS1, Kallmann syndrome), "
                "STS (steroid sulfatase, XLRI), and NLGN4X (autism). "
                "Contiguous Xp22.31 deletion co-deletes all three → XLRI + Kallmann syndrome (anosmia + "
                "hypogonadotropic hypogonadism + cryptorchidism) + autism/intellectual disability. "
                "MLPA or chromosomal microarray MANDATORY in all XLRI males — sequencing alone does NOT detect deletions. "
                "15% of XLRI males have contiguous deletion vs point mutation."
            ),
            "COL7A1 RDEB SCC Risk": (
                "Recessive Dystrophic EB Hallopeau-Siemens (RDEB-HS) has 40-80% lifetime risk of cutaneous SCC. "
                "SCC is the LEADING CAUSE OF DEATH in RDEB. Median age first SCC: 34 years. "
                "Mechanism: chronic wound environment → chronic inflammation → pro-oncogenic keratinocyte selection. "
                "MANDATORY SURVEILLANCE: Monthly skin inspection + annual skin biopsy of all chronic wounds. "
                "PET-CT if SCC confirmed (high risk of regional/distant metastasis from RDEB-SCC). "
                "Oleogel-S10 (Filsuvez, FDA 2023) reduces wound burden = may reduce SCC risk."
            ),
            "JEB Granulation Tissue Sign": (
                "Exuberant granulation tissue is PATHOGNOMONIC for Junctional Epidermolysis Bullosa (JEB). "
                "Sites: perioral, perinasal, periocular, laryngeal, tracheal, esophageal, urogenital. "
                "Mechanism: absent laminin-332 → impaired re-epithelialisation of wounds → chronic wound → "
                "granulation tissue overgrowth. "
                "Laryngeal granulation tissue → hoarse cry → stridor → respiratory failure → EMERGENCY. "
                "Granulation tissue differentiates JEB from EBS (no granulation) and DEB (rare granulation)."
            ),
            "Hailey-Hailey Kaposi Varicelliform Eruption": (
                "Kaposi Varicelliform Eruption (KVE) = widespread HSV infection of skin with disrupted barrier. "
                "In Hailey-Hailey Disease: HSV infects intertriginous erosions → rapid spread → "
                "widespread haemorrhagic vesicles/pustules + systemic illness. "
                "TREATMENT: IV aciclovir 5 mg/kg TDS immediately (NOT topical) → hospitalisation. "
                "PROPHYLAXIS: In recurrent HSV-triggered HHD → aciclovir 400 mg BD continuous prophylaxis. "
                "KVE also occurs in atopic dermatitis (eczema herpeticum) — same emergency treatment."
            ),
            "EDA Anhidrosis Heat Emergency": (
                "XLHED (EDA): Absent eccrine sweat glands = anhidrosis = inability to thermoregulate. "
                "Heat stroke (hyperthermia >40°C) = life-threatening emergency. "
                "PREVENTION: Cooling vest (personal cooling system) worn outdoors above 25°C; "
                "air-conditioned home and car; no hot baths; cold packs during fever. "
                "EMERGENCY: Immersion in cold water; wet towels + fans; cold IV fluids; ICU. "
                "PRENATAL THERAPY (EDX111): Intraamniotic EDA1 protein at 26-30 weeks gestation → "
                "partially restores sweat gland development — first disease-modifying therapy for XLHED."
            ),
            "Botulinum Toxin Hailey-Hailey": (
                "Botulinum Toxin Type A (onabotulinumtoxinA, 50-100U per axilla) injected intradermally "
                "into affected intertriginous skin of Hailey-Hailey Disease. "
                "Mechanism: blocks acetylcholine → eccrine sweat gland inhibition → removes sweating trigger → "
                "dramatic improvement in axillary and inguinal HHD within 2-4 weeks. "
                "Duration of effect: 6-12 months; repeat injections PRN. "
                "Strong evidence: multiple case series and small RCTs; considered standard of care for flexural HHD."
            ),
        }
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(f"Atlas: {ov['atlas_name']}")
    print(f"N patients: {ov['n_patients']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Aggregate clinical stats: {json.dumps(ov['aggregate_clinical'], indent=2)}")
    print("\n=== BREAKDOWN (gene list) ===")
    bd = get_breakdown()
    for g, info in bd["breakdown"].items():
        print(f"  {g}: {info['cohort_stats']}")
    print("\n=== DEFINITIONS (keys) ===")
    df = get_definitions()
    for k in df["definitions"]:
        print(f"  - {k}")
