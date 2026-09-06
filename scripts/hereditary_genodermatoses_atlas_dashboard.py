#!/usr/bin/env python3
"""Hereditary-Genodermatoses-Atlas — Complete 8-Gene Hereditary Skin Disease Atlas
COL7A1  (Collagen type VII alpha-1; 2944 aa; 3p21.31; AR/AD;
         Epidermolysis Bullosa Dystrophica — RDEB most severe, SCC risk;
         no type VII collagen in skin; beremagene (gene therapy FDA 2023);
         seed SEED_BASE+0) ·
LAMB3   (Laminin subunit beta-3; 1172 aa; 1q32.2; AR;
         Junctional Epidermolysis Bullosa — Herlitz lethal (null/null);
         laminin-332 trimer (LAMA3/LAMB3/LAMC2); skin biopsy + IF essential;
         seed SEED_BASE+1) ·
KRT5    (Keratin 5; 590 aa; 12q13.13; AD;
         Epidermolysis Bullosa Simplex — most common EB, Dowling-Meara severe;
         tonofilament aggregates; heat triggers; no disease-modifying therapy;
         seed SEED_BASE+2) ·
ABCA12  (ATP-binding cassette transporter A12; 2595 aa; 2q34; AR;
         Harlequin Ichthyosis — neonatal emergency; acitretin ASAP life-saving;
         lamellar body lipid transport; collodion membrane; corneal ulceration;
         seed SEED_BASE+3) ·
SPINK5  (Serine protease inhibitor Kazal-type 5 / LEKTI; 1094 aa; 5q32; AR;
         Netherton Syndrome — trichorrhexis invaginata PATHOGNOMONIC;
         IgE >2000 IU/mL; KLK5/KLK7 hyperactivity; dupilumab effective;
         seed SEED_BASE+4) ·
ATP2A2  (Sarcoplasmic/ER Ca2+-ATPase isoform 2 / SERCA2; 1042 aa; 12q24.11; AD;
         Darier Disease — warty papules seborrhoeic distribution; keratosis follicularis;
         lithium triggers/worsens; isotretinoin/acitretin first-line;
         seed SEED_BASE+5) ·
STS     (Steroid sulfatase; 583 aa; Xp22.31; XLR;
         X-Linked Ichthyosis — corneal opacities (deep stromal); contiguous Kallmann/ANOS1;
         cryptorchidism 25%; placental sulfatase deficiency → impaired labour;
         seed SEED_BASE+6) ·
IKBKG   (NF-kB essential modulator / NEMO; 419 aa; Xq28; XLD;
         Incontinentia Pigmenti — males usually lethal; 4-stage skin;
         dental/eye (retinal vasculopathy)/CNS; females heterozygous;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1590–1597)
"""

import random

SEED_BASE = 1590

GENO_GENES = [
    # ── COL7A1 — Epidermolysis Bullosa Dystrophica (RDEB) ─────────────────
    {
        "gene": "COL7A1",
        "protein": "COL7A1 — Epidermolysis Bullosa Dystrophica (RDEB), SCC Risk, Beremagene Gene Therapy FDA 2023",
        "alias": (
            "COL7A1; OMIM gene 120120; Epidermolysis Bullosa Dystrophica (EBD/RDEB) OMIM 226600 (AR) / 131750 (AD); "
            "3p21.31; 2944 aa (pro-collagen); ~290 kDa; AR (RDEB, most severe) / AD (DDEB, milder); "
            "prevalence ~1:500,000 (RDEB), ~1:30,000 EB overall. "
            "COL7A1 encodes type VII collagen, the major structural component of anchoring fibrils "
            "that tether the lamina densa of the basement membrane to the upper papillary dermis. "
            "Without functional type VII collagen: anchoring fibrils are absent or deficient → "
            "the epidermis detaches from the dermis at the sub-lamina densa zone → "
            "SUBEPIDERMAL BLISTERING at the slightest friction or trauma. "
            "RDEB (RECESSIVE DYSTROPHIC EB — most severe): biallelic null or truncating COL7A1 mutations → "
            "complete absence of type VII collagen → generalised severe blistering from birth → "
            "chronic wounds → MILIA (keratin-filled cysts at healing blister sites, PATHOGNOMONIC for DEB) → "
            "PSEUDOSYNDACTYLY (mitten deformity — digits fused from repeated blistering/scarring → "
            "progressive loss of hand function); "
            "OESOPHAGEAL STRICTURES (mucous membrane blistering → stricture → dysphagia → malnutrition); "
            "ANAEMIA (chronic blood loss through wounds + iron deficiency); "
            "SQUAMOUS CELL CARCINOMA (SCC) — THE LEADING CAUSE OF DEATH IN RDEB: "
            "chronic non-healing wounds → aggressive, multi-focal SCC; "
            "cumulative SCC risk 90.1% by age 45 in generalised severe RDEB; "
            "metastasis common; standard surgical excision + early surveillance mandatory. "
            "DDEB (DOMINANT DYSTROPHIC EB — AD, milder): heterozygous missense COL7A1 → "
            "reduced/abnormal anchoring fibrils → localised blistering (hands, feet, elbows, knees); "
            "nail dystrophy; SCC risk present but substantially lower. "
            "DIAGNOSIS: skin biopsy + transmission electron microscopy (TEM) → absent/reduced anchoring fibrils "
            "in sub-lamina densa zone; immunofluorescence (IF) mapping with anti-type VII collagen antibody "
            "→ absent or reduced staining; COL7A1 gene sequencing. "
            "WOUND MANAGEMENT: non-adherent dressings (silicone, foam); "
            "lance and drain blisters (burst small end, leave roof); "
            "infection control (Staphylococcus aureus colonisation universal in RDEB — "
            "bleach baths, mupirocin for acute infection; avoid systemic antibiotics for colonisation alone); "
            "nutritional support (PEG tube for severe oesophageal strictures). "
            "GENE THERAPY: beremagene geperpavec (B-VEC, Krystal Biotech, FDA May 2023) — "
            "first approved gene therapy for EB: topical HSV-1 vector delivering COL7A1 directly to wounds; "
            "shown to increase complete wound closure vs placebo in Phase 3 (GEM trial); "
            "applied directly to blistered wounds; COL7A1 gene expression in keratinocytes → "
            "de novo type VII collagen production → anchoring fibril formation → wound healing. "
            "Bone marrow / fibroblast transplantation strategies under investigation."
        ),
        "aa": "2944 aa",
        "kDa": "~290 kDa",
        "locus": "3p21.31",
        "omim_gene": 120120,
        "omim_disease": 226600,
        "inheritance": "AR for RDEB (biallelic, most severe); AD for DDEB (heterozygous, milder); >800 pathogenic variants",
        "gene_class": (
            "COL7A1 encodes the alpha1 chain of type VII collagen (homotrimer: [α1(VII)]3). "
            "Domain structure: N-terminal non-collagenous domain 1 (NC1, 1253 aa — contains fibronectin type III-like repeats; "
            "mediates binding to type IV collagen, laminin-332, fibronectin) → "
            "central triple-helical collagenous domain (Gly-X-Y repeats, ~145 kDa) → "
            "C-terminal non-collagenous domain 2 (NC2, 161 aa — mediates antiparallel dimer formation). "
            "Assembly: two COL7A1 homotrimers align antiparallel → NC2-NC2 disulfide dimer → "
            "lateral aggregation → anchoring fibril (cross-links lamina densa to papillary dermis). "
            "Mutation spectrum: null/truncating mutations (nonsense, frameshift, splice site) → absent type VII collagen → RDEB; "
            "glycine substitutions in triple helix → disrupted helix → RDEB or DDEB depending on severity; "
            "NC1 domain missense → DDEB (dominant-negative interference with anchoring fibril assembly). "
            "Over 800 unique COL7A1 variants reported (LOVD database). "
            "Beremagene geperpavec targets COL7A1 delivery to skin keratinocytes via HSV-1 vector."
        ),
        "n_patients": 40,
        "key_alerts": [
            "COL7A1-SCC-LEADING-CAUSE-OF-DEATH: Squamous cell carcinoma (SCC) is THE LEADING CAUSE OF DEATH in generalised severe RDEB — cumulative risk >90% by age 45; multifocal, aggressive, metastasising SCC arises in chronic non-healing wounds; monthly skin surveillance with biopsy of suspicious areas is mandatory from adolescence",
            "COL7A1-BEREMAGENE-FDA-2023: Beremagene geperpavec (B-VEC, Krystal Biotech) is the FIRST FDA-APPROVED gene therapy for EB (May 2023) — topical HSV-1 vector delivering COL7A1 directly to wounds; shown to improve complete wound closure vs placebo in Phase 3 GEM trial; apply to accessible blistered areas",
            "COL7A1-NO-TYPE-VII-COLLAGEN-IF: Immunofluorescence (IF) mapping of skin biopsy with anti-type VII collagen antibody shows ABSENT or markedly reduced staining in RDEB — this guides genotype confirmation; TEM shows absent anchoring fibrils in sub-lamina densa zone; essential for EB subtype diagnosis",
            "COL7A1-MILIA-PSEUDOSYNDACTYLY: Milia (keratin-filled cysts, PATHOGNOMONIC for DEB) and pseudosyndactyly (mitten deformity of hands) are clinical hallmarks of RDEB — progressive digit fusion from repeated blistering and scarring; hand surgery (syndactyly release) may temporarily improve function but recurs",
            "COL7A1-WOUND-MANAGEMENT-LANCE: Blisters in RDEB should be LANCED AND DRAINED (pierce at small end, leave the blister roof as a biological dressing) — do NOT leave intact as tension causes larger area blistering; non-adherent silicone-based dressings are standard; avoid tape and adhesives directly on skin",
            "COL7A1-OESOPHAGEAL-STRICTURE: Oesophageal blistering → strictures → dysphagia → severe malnutrition and growth failure; endoscopic dilation may be needed; gastrostomy (PEG) for severe cases; maintain soft/pureed diet; adequate calories are critical for wound healing in RDEB",
            "COL7A1-ANAEMIA-CHRONIC: Chronic iron-deficiency anaemia is universal in severe RDEB (chronic wound blood loss + poor oral intake + malabsorption); regular FBC, ferritin, transferrin saturation; IV iron infusions often required; erythropoietin in refractory anaemia",
            "COL7A1-DDEB-MILDER-AD: Dominant DEB (DDEB) is caused by HETEROZYGOUS COL7A1 mutations (often glycine substitutions in triple helix) — localised blistering on hands/feet/bony prominences, nail dystrophy, SCC risk present but substantially lower than RDEB; prognosis much better than biallelic disease",
        ],
        "etiologies": {
            "Biallelic null mutations (nonsense/frameshift) — RDEB severe": 15,
            "Biallelic missense (glycine substitutions, triple helix)": 10,
            "Compound heterozygous (null + missense)": 9,
            "Heterozygous missense DDEB (dominant negative)": 4,
            "Splice site biallelic": 2,
        },
        "stats": {
            "rdeb_generalised_severe_pct": 55,
            "ddeb_pct": 15,
            "scc_by_45_pct": 90,
            "pseudosyndactyly_pct": 70,
            "oesophageal_stricture_pct": 50,
            "chronic_anaemia_pct": 85,
            "milia_present_pct": 80,
            "beremagene_eligible_pct": 40,
            "mean_dx_age_months": 1,
            "mean_dx_delay_months": 2,
        },
        "dx_delay_distribution": {"<1m": 22, "1-6m": 12, "6-24m": 4, ">24m": 2},
    },

    # ── LAMB3 — Junctional Epidermolysis Bullosa ───────────────────────────
    {
        "gene": "LAMB3",
        "protein": "LAMB3 — Junctional EB, Herlitz Lethal (Null/Null), Laminin-332 Trimer, IF/TEM Diagnosis",
        "alias": (
            "LAMB3; OMIM gene 150310; Junctional Epidermolysis Bullosa (JEB) — Herlitz type OMIM 226700 / "
            "non-Herlitz JEB (LAMB3) OMIM 226650; 1q32.2; 1172 aa; ~130 kDa; AR; "
            "prevalence JEB overall ~1:450,000. "
            "LAMB3 encodes laminin subunit beta-3, a critical structural component of laminin-332 "
            "(formerly laminin-5 / kalinin / nicein / epiligrin), the principal laminin in the "
            "lamina lucida of the cutaneous basement membrane zone (BMZ). "
            "Laminin-332 is a heterotrimer of three chains: LAMA3 (alpha-3) + LAMB3 (beta-3) + LAMC2 (gamma-2); "
            "mutations in any of the three genes (LAMA3, LAMB3, or LAMC2) can cause JEB. "
            "Laminin-332 anchors hemidesmosomes to the lamina lucida → secures the basal keratinocyte "
            "layer to the BMZ; without it, cleavage occurs through the LAMINA LUCIDA → "
            "intralamina-lucida blistering. "
            "HERLITZ JEB (LETHAL): biallelic NULL/NULL mutations (nonsense/frameshift in both alleles) → "
            "COMPLETE ABSENCE of laminin-332 → SEVERE generalised blistering from birth → "
            "large denuded areas → fatal sepsis/respiratory failure in infancy (months to 2 years); "
            "exuberant granulation tissue on face/neck/trunk (PATHOGNOMONIC for Herlitz JEB); "
            "involvement of airways (laryngotracheal blistering → stridor, respiratory distress); "
            "no effective long-term treatment; palliative care often appropriate; "
            "LAMB3 biallelic null is the most common genotype for Herlitz JEB. "
            "NON-HERLITZ JEB (LAMB3): one or two non-null mutations (missense + one null OR biallelic missense) → "
            "some residual laminin-332 → milder phenotype → SURVIVAL; "
            "generalised blistering (improves with age in some); "
            "severe nail dystrophy / nail loss; dental enamel hypoplasia (PATHOGNOMONIC for JEB — "
            "laminin-332 expressed in enamel organ); alopecia (patchy or total); "
            "respiratory, gastrointestinal, urinary tract mucosal involvement; "
            "eye: symblepharon (conjunctival adhesion). "
            "DIAGNOSIS: skin biopsy + IF mapping (anti-laminin-332 antibody — absent in Herlitz, reduced in non-Herlitz); "
            "TEM (cleavage plane = lamina lucida level → confirms JEB subtype); "
            "LAMA3, LAMB3, LAMC2 gene sequencing panel. "
            "TREATMENT: wound management as for DEB (non-adherent dressings); "
            "probiotic feeds; tracheostomy for airway involvement; "
            "systemic antibiotic treatment for sepsis; "
            "gene therapy trials for LAMA3/LAMB3 JEB under development; "
            "ex vivo gene-corrected keratinocyte grafting under investigation (HOLOCLAR precedent)."
        ),
        "aa": "1172 aa",
        "kDa": "~130 kDa",
        "locus": "1q32.2",
        "omim_gene": 150310,
        "omim_disease": 226700,
        "inheritance": "AR; biallelic null/null = Herlitz (lethal); one null + one missense or biallelic missense = non-Herlitz (survival)",
        "gene_class": (
            "LAMB3 encodes the beta-3 chain of laminin-332 (LN-332, heterotrimer α3β3γ2). "
            "Domain structure: N-terminal LN domain (laminin N-terminal globule) → "
            "L4 domain → EGF-like domains (tandem repeats) → coiled-coil domain (α-helical, "
            "mediates heterotrimer assembly with LAMA3 and LAMC2). "
            "Laminin-332 assembly: LAMC2 + LAMB3 form beta-gamma dimer first → "
            "LAMA3 associates → secreted as heterotrimer → deposited in lamina lucida. "
            "Function: LAMB3 short arm binds nidogen-1 and type IV collagen in lamina densa; "
            "LAMA3 G-domain binds alpha-6/beta-4 integrin (hemidesmosome anchor) → "
            "cell-matrix adhesion. "
            "Key null mutations: R635X (c.1903C>T) and Q243X (c.727C>T) — "
            "both common in European JEB-Herlitz patients; detected by targeted genotyping. "
            "Genotype-phenotype: biallelic truncating = Herlitz (no laminin-332) → lethal; "
            "compound heterozygous (truncating + missense) or biallelic missense = non-Herlitz (residual laminin-332). "
            "Anti-laminin-332 antibodies can also cause acquired JEB (pemphigoid variant) — "
            "distinguish by checking autoantibodies in adult-onset cases."
        ),
        "n_patients": 40,
        "key_alerts": [
            "LAMB3-HERLITZ-NULL-NULL-LETHAL: Biallelic LAMB3 null mutations (null/null genotype) = Herlitz JEB — LETHAL in infancy; complete absence of laminin-332; exuberant perioral/perigenital granulation tissue is PATHOGNOMONIC for Herlitz JEB; genotype is critical for prognosis and palliative planning",
            "LAMB3-NON-HERLITZ-SURVIVAL: One null + one missense (or biallelic missense) = non-Herlitz JEB — residual laminin-332 → SURVIVAL; severity variable; dental enamel hypoplasia (pitting, enamel defects) is a PATHOGNOMONIC non-cutaneous feature of JEB across subtypes",
            "LAMB3-IF-MAPPING-ESSENTIAL: Immunofluorescence (IF) mapping with anti-laminin-332 antibody is ESSENTIAL for JEB diagnosis — completely absent staining = Herlitz; reduced staining = non-Herlitz; guides prognosis before genetic results; anti-type XVII collagen (BP180) also absent in JEB (hemidesmosome)",
            "LAMB3-LAMININ-332-TRIMER: Mutations in ANY of the three laminin-332 chains (LAMA3, LAMB3, LAMC2) cause JEB — sequence ALL THREE GENES when clinical and IF findings are consistent with JEB; LAMB3 null mutations are the most common cause of Herlitz JEB",
            "LAMB3-AIRWAY-INVOLVEMENT: Laryngotracheal mucosal blistering in Herlitz JEB → stridor, respiratory failure; tracheostomy may be required acutely; airway involvement is a major cause of early death; assess stridor in all Herlitz JEB infants as an emergency",
            "LAMB3-DENTAL-ENAMEL-HYPOPLASIA: Dental enamel hypoplasia (pitting, irregular enamel, enamel loss) is PATHOGNOMONIC for JEB (all subtypes) — laminin-332 is expressed in the enamel organ; absent/reduced laminin-332 → enamel matrix defect; dental involvement distinguishes JEB from DEB (COL7A1) clinically",
            "LAMB3-EXUBERANT-GRANULATION-PERIORAL: Exuberant granulation tissue around mouth, nose, and genitalia is PATHOGNOMONIC for Herlitz JEB — a bright red granulating tissue collar around perioral skin; contrasts with DEB which heals with scarring; distinctive on clinical examination from birth",
            "LAMB3-TEM-CLEAVAGE-LAMINA-LUCIDA: Transmission electron microscopy (TEM) of fresh skin biopsy shows cleavage plane within the LAMINA LUCIDA (between lamina lucida and hemidesmosomes) → confirms JEB; DEB cleaves BELOW lamina densa (sub-lamina densa); EBS cleaves WITHIN basal keratinocytes (intraepidermal)",
        ],
        "etiologies": {
            "Biallelic null/null (nonsense/frameshift) — Herlitz JEB": 16,
            "Compound heterozygous null + missense (non-Herlitz)": 14,
            "Biallelic missense (non-Herlitz, milder)": 6,
            "LAMA3 or LAMC2 mutation (phenotypically JEB)": 3,
            "Splice site biallelic": 1,
        },
        "stats": {
            "herlitz_lethal_pct": 40,
            "non_herlitz_survival_pct": 55,
            "enamel_hypoplasia_pct": 90,
            "exuberant_granulation_pct": 40,
            "airway_involvement_pct": 30,
            "alopecia_pct": 60,
            "mean_dx_age_months": 0,
            "mean_dx_delay_months": 1,
        },
        "dx_delay_distribution": {"<1m": 28, "1-3m": 8, "3-12m": 3, ">12m": 1},
    },

    # ── KRT5 — Epidermolysis Bullosa Simplex ──────────────────────────────
    {
        "gene": "KRT5",
        "protein": "KRT5 — Epidermolysis Bullosa Simplex, Dowling-Meara Severe, Heat Triggers, Kobner Phenomenon",
        "alias": (
            "KRT5; OMIM gene 148040; Epidermolysis Bullosa Simplex (EBS) OMIM 131900 (generalized severe/Dowling-Meara); "
            "12q13.13; 590 aa; ~62 kDa; AD (dominant negative); "
            "EBS is the MOST COMMON subtype of EB overall; prevalence ~1:30,000–1:50,000. "
            "KRT5 encodes keratin 5, a type II intermediate filament protein expressed in basal keratinocytes. "
            "Keratin 5 (KRT5) obligatorily pairs with keratin 14 (KRT14) to form 10-nm heteropolymer "
            "intermediate filaments (IFs) that form the structural cytoskeleton of basal epidermal cells. "
            "The IF network distributes mechanical stress across the epidermis; without intact KRT5-KRT14 filaments: "
            "basal keratinocytes rupture under mechanical stress → INTRAEPIDERMAL BLISTERING "
            "(within the basal cell layer → supra-basal splitting). "
            "CLINICAL FEATURES: blistering from birth or early childhood at friction sites "
            "(palms, soles, elbows, knees); KOBNER PHENOMENON (blistering at sites of rubbing/trauma); "
            "HEAT TRIGGERS — sweating and warmth make blistering DRAMATICALLY WORSE; "
            "summer exacerbations are typical; many EBS patients improve in cool environments. "
            "DOWLING-MEARA SUBTYPE (EBS-DM, SEVERE): "
            "caused by glycine substitution mutations within the helix initiation/termination motifs "
            "(1A or 2B rod domain) of KRT5 or KRT14 → DOMINANT NEGATIVE disruption → "
            "CLUSTERED PERINUCLEAR TONOFILAMENT AGGREGATES on TEM (PATHOGNOMONIC for EBS-DM); "
            "herpetiform blister grouping (clusters, hence 'herpetiform'); "
            "most severe EBS subtype — widespread generalised blistering from birth; "
            "may improve slightly with age; milia less common than DEB; "
            "palmoplantar keratoderma (PPK) develops in some. "
            "EBS-OTHER / LOCALISED (Webber-Cockayne): milder mutations → localised to hands/feet; "
            "worst in summer; often misdiagnosed as pompholyx or contact dermatitis. "
            "EBS + MUSCULAR DYSTROPHY: rare, associated with PLEC1 (plectin) mutations — KRT5/14 normal. "
            "DIAGNOSIS: skin biopsy + IF mapping (type IV collagen above blister = intraepidermal) → "
            "confirms EBS; TEM for tonofilament aggregates in EBS-DM; "
            "KRT5 or KRT14 gene sequencing. "
            "TREATMENT: symptomatic only — COOLING (cool water, cool environments), "
            "non-adherent dressings, lance and drain blisters; "
            "lancing early (before enlargement) reduces pain and skin loss; "
            "topical antiinfective for secondary infection; "
            "no FDA-approved disease-modifying therapy for EBS (unlike RDEB beremagene); "
            "diacerein (IL-1β inhibitor, topical) showed signal in a Phase 3 trial; "
            "clinical trials of KB105 (gene therapy) and SD-101 (oligonucleotide) ongoing."
        ),
        "aa": "590 aa",
        "kDa": "~62 kDa",
        "locus": "12q13.13",
        "omim_gene": 148040,
        "omim_disease": 131900,
        "inheritance": "AD dominant negative; severe (EBS-DM) from glycine substitutions in helix initiation/termination motifs; milder localised variants from distal rod/head domain mutations",
        "gene_class": (
            "KRT5 encodes keratin 5, a type II (basic) intermediate filament protein. "
            "Domain structure: N-terminal head domain (V1 subdomain) → "
            "central rod domain (1A–L12–1B–L2–2A–L2–2B coils, ~310 aa) → "
            "C-terminal tail domain (V2 subdomain). "
            "Keratin obligate pairing: KRT5 (type II) + KRT14 (type I) → antiparallel heterodimer → "
            "protofilament → protofibril → 10-nm intermediate filament (tonofilament). "
            "Rod domain 1A start ('helix initiation motif', TYRKLLEGEE) and 2B end "
            "('helix termination motif', YRKLLEGEE) are HOT SPOTS for dominant-negative mutations: "
            "glycine substitutions here (e.g., R125C/H, E170K in KRT5; R125C in KRT14) → "
            "dominant-negative disruption of IF assembly → tonofilament aggregation → "
            "cell fragility → Dowling-Meara. "
            "Distal mutations (tail/head domains, L12, 2B distal) → localised EBS (Webber-Cockayne). "
            "KRT14 (17q21.2) mutations cause EBS phenotypically identical to KRT5 mutations — "
            "sequence both genes when KRT5 is negative."
        ),
        "n_patients": 40,
        "key_alerts": [
            "KRT5-DOWLING-MEARA-TONOFILAMENT-AGGREGATES: Dowling-Meara EBS (EBS-DM) is identified by CLUSTERED PERINUCLEAR TONOFILAMENT AGGREGATES on TEM — PATHOGNOMONIC for EBS-DM; caused by glycine substitutions in helix initiation/termination motifs; most severe EBS subtype with herpetiform blister clusters",
            "KRT5-HEAT-TRIGGERS: HEAT dramatically worsens EBS blistering — sweating, friction, and warmth are the main triggers; advise COOLING strategies (cool baths, air conditioning, avoiding hot climates); many patients note summer exacerbations and relief in winter/cold environments",
            "KRT5-KOBNER-PHENOMENON: Kobner (isomorphic) phenomenon — blistering at sites of trauma/rubbing — is universal in EBS; advise against friction, tight footwear, contact sports; blister lancing early (before enlargement) reduces collateral damage; non-adherent dressings for all wound sites",
            "KRT5-NO-DISEASE-MODIFYING-THERAPY: Unlike RDEB (beremagene, FDA 2023), there is NO FDA-APPROVED disease-modifying therapy for EBS; management is purely SYMPTOMATIC (cooling, wound care, lancing); clinical trials of diacerein (topical IL-1β inhibitor), KB105 (gene therapy), SD-101 ongoing",
            "KRT5-INTRAEPIDERMAL-BLISTERING-IF: IF mapping of skin biopsy shows type IV collagen ABOVE the split (blister roof contains dermis-side lamina densa) — confirms INTRAEPIDERMAL blistering; contrasts with JEB (split in lamina lucida) and DEB (split below lamina densa); EBS diagnosis confirmed",
            "KRT5-LOCALISED-EBS-MISDIAGNOSIS: Mild EBS (Webber-Cockayne) localised to palms/soles is frequently misdiagnosed as pompholyx, dyshidrotic eczema, or contact dermatitis — friction-induced plantar blisters in summer that resolve in winter; family history of 'blisters on feet' is the clue; biopsy confirms EBS",
            "KRT5-PALMOPLANTAR-KERATODERMA: Palmoplantar keratoderma (PPK) develops in a subset of EBS patients (particularly EBS-DM and those with KRT5 p.Glu170Lys) — diffuse thickening of palms and soles; distinct from blistering but coexistent; reduces blister frequency but can be painful/disabling",
            "KRT5-KRT14-SAME-PHENOTYPE: KRT14 mutations (17q21.2) cause EBS phenotypically identical to KRT5 mutations — sequence BOTH KRT5 AND KRT14 when clinical diagnosis is EBS; KRT14 p.R125C/H are common EBS-DM mutations; compound heterozygous or homozygous KRT14 mutations cause EBS-AR (autosomal recessive, rare, severe)",
        ],
        "etiologies": {
            "KRT5 glycine substitution helix motif (EBS-DM, severe)": 16,
            "KRT5 missense distal rod/head (EBS localised, Webber-Cockayne)": 12,
            "KRT14 mutation (same EBS phenotype)": 8,
            "KRT5 novel/unique missense (intermediate severity)": 3,
            "KRT5 + PPK-associated variant (Glu170Lys)": 1,
        },
        "stats": {
            "ebs_dm_dowling_meara_pct": 40,
            "ebs_localised_pct": 35,
            "heat_trigger_pct": 95,
            "kobner_phenomenon_pct": 100,
            "ppk_pct": 20,
            "summer_exacerbation_pct": 85,
            "misdiagnosis_before_dx_pct": 45,
            "mean_dx_age_months": 1,
            "mean_dx_delay_months": 18,
        },
        "dx_delay_distribution": {"<3m": 18, "3-12m": 10, "12-60m": 8, ">60m": 4},
    },

    # ── ABCA12 — Harlequin Ichthyosis ─────────────────────────────────────
    {
        "gene": "ABCA12",
        "protein": "ABCA12 — Harlequin Ichthyosis, Neonatal Emergency, Acitretin ASAP Life-Saving, Lamellar Body",
        "alias": (
            "ABCA12; OMIM gene 607800; Harlequin Ichthyosis OMIM 242500; 2q34; 2595 aa; ~290 kDa; "
            "AR; prevalence ~1:300,000–1:500,000 (rare, severe). "
            "ABCA12 encodes an ATP-binding cassette (ABC) transporter of the A subfamily, expressed "
            "in differentiating epidermal keratinocytes (stratum granulosum). "
            "ABCA12 is the LAMELLAR BODY LIPID TRANSPORTER: normally transports glucosylceramides, "
            "ceramides, and other lipids into lamellar body granules (Odland bodies) within "
            "stratum granulosum keratinocytes; lamellar bodies fuse with the plasma membrane → "
            "extrude lipid into the extracellular space → form the water-impermeant lipid lamellar "
            "membrane of the stratum corneum (epidermal permeability barrier). "
            "Without functional ABCA12: lamellar bodies are EMPTY or malformed → "
            "lipid is not delivered to the extracellular space → "
            "SEVERE STRATUM CORNEUM BARRIER FAILURE. "
            "HARLEQUIN ICHTHYOSIS — NEONATAL EMERGENCY: "
            "affected neonates are born encased in a thick, plate-like COLLODION MEMBRANE "
            "(hardened, cracked, fissured stratum corneum covering entire body); "
            "the collodion membrane causes: ECTROPION (eyelids turned out → corneal exposure → "
            "corneal ulceration and blindness risk); ECLABIUM (lips turned out → feeding difficulty); "
            "LIMB CONSTRICTION (digital ischemia → necrosis); RESPIRATORY RESTRICTION "
            "(rigid chest → respiratory failure); TEMPERATURE INSTABILITY (barrier failure → "
            "evaporative water and heat loss → hypothermia/hyperthermia); "
            "SEPSIS RISK (skin cracking → portal of entry for bacteria). "
            "ACITRETIN (oral retinoid): ACITRETIN IS LIFE-SAVING in Harlequin Ichthyosis — "
            "promotes keratinocyte differentiation and collodion membrane shedding; "
            "should be commenced AS SOON AS POSSIBLE after birth (1 mg/kg/day); "
            "initiates shedding of the collodion membrane within days to weeks; "
            "survivors require long-term acitretin/retinoid therapy; "
            "without retinoids, mortality in first weeks is very high. "
            "LONG-TERM (SURVIVORS): generalised severe lamellar ichthyosis pattern "
            "(thick plate-like scales, ectropion, eclabium, alopecia, hypohidrosis → heat intolerance); "
            "repeated infections; joint contractures from scaling; severe QoL impairment; "
            "emollients (multiple applications daily) are a cornerstone of management; "
            "keratolytics (urea, salicylic acid for body, AVOID SALICYLATE TOXICITY IN INFANTS); "
            "ophthalmology surveillance for corneal complications; "
            "barrier nursing in neonatal ICU (humidified incubator, saline soaks, petrolatum)."
        ),
        "aa": "2595 aa",
        "kDa": "~290 kDa",
        "locus": "2q34",
        "omim_gene": 607800,
        "omim_disease": 242500,
        "inheritance": "AR; biallelic loss-of-function mutations; ABCA12 missense mutations in some cases of congenital ichthyosiform erythroderma (lamellar variant, milder)",
        "gene_class": (
            "ABCA12 encodes a full transporter ABC protein (ABCA subfamily). "
            "Domain structure: TMD1 (6 TM helices) → NBD1 (Walker A/B motifs, LSGGQ) → "
            "TMD2 (6 TM helices) → NBD2 → C-terminal regulatory domain. "
            "Full transporter (not half-transporter) — functions as monomer. "
            "ABCA12 is expressed in granular layer keratinocytes (stratum granulosum). "
            "Function: loads glucosylceramides and other long-chain ceramides into lamellar bodies "
            "(Odland bodies); lamellar body content → extruded into stratum corneum intercellular space → "
            "enzymatically processed to ceramides → lipid lamellar membrane = water barrier. "
            "Complete loss (biallelic null): EMPTY LAMELLAR BODIES (key ultrastructural finding on TEM); "
            "no lipid lamellar membrane → Harlequin. "
            "Partial loss (missense, some residual function): lamellar ichthyosiform erythroderma (LIE) "
            "or non-bullous congenital ichthyosiform erythroderma (NBCIE) — milder than Harlequin. "
            "ABCA12 also shows structural similarity to ABCA1/ABCA4 (cholesterol and retinoid transporters). "
            "Skin biopsy: TEM shows abnormal/empty lamellar bodies; lipid extrusion fails on freeze-fracture."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ABCA12-ACITRETIN-ASAP-LIFE-SAVING: Acitretin (1 mg/kg/day oral retinoid) MUST be commenced AS SOON AS POSSIBLE after birth in Harlequin Ichthyosis — promotes collodion membrane shedding and keratinization normalisation; without retinoids, neonatal mortality is very high; do not wait for genetic confirmation before starting",
            "ABCA12-NEONATAL-EMERGENCY-COLLODION: Harlequin Ichthyosis neonate presents with rigid collodion membrane encasing the entire body — NEONATAL EMERGENCY; urgent issues are respiratory restriction (rigid chest), corneal exposure (ectropion → ulceration), digital vascular compromise (constriction), temperature instability, and sepsis risk",
            "ABCA12-CORNEAL-ULCERATION-ECTROPION: Ectropion (eyelids everted outward) from collodion membrane tension → exposed cornea → CORNEAL ULCERATION AND BLINDNESS; urgent ophthalmology involvement; lubricating eye drops every 1–2 hours; may require temporary tarsorrhaphy; long-term ectropion repair surgery",
            "ABCA12-LAMELLAR-BODY-LIPID-TRANSPORT: ABCA12 is the lamellar body lipid transporter — loads glucosylceramides into Odland bodies in the stratum granulosum; without ABCA12, lamellar bodies are EMPTY → no lipid extruded into stratum corneum → complete barrier failure; TEM of skin biopsy shows empty lamellar bodies",
            "ABCA12-HYPOHIDROSIS-HEAT-INTOLERANCE: Survivors of Harlequin Ichthyosis have HYPOHIDROSIS (reduced sweating) due to plugged sweat ducts beneath scales → HEAT INTOLERANCE and risk of heat stroke; avoid hot environments; cooling strategies essential; emollient-first approach to keep scales manageable",
            "ABCA12-JOINT-CONTRACTURES: Thick scaling over joints → contractures (elbows, knees, fingers) → loss of function; physiotherapy and stretching essential; keratolytics (urea, lactic acid) soften scale over joints; orthotic devices may be needed; surgery rarely required but considered in severe cases",
            "ABCA12-LONG-TERM-RETINOIDS: Long-term oral retinoids (acitretin or isotretinoin) are required for survivors of Harlequin Ichthyosis — dramatically reduce scale thickness and improve quality of life; monitor for retinoid toxicity (hepatotoxicity, dyslipidaemia, teratogenicity — mandatory contraception in females)",
            "ABCA12-MILDER-ABCA12-MISSENSE: Biallelic ABCA12 missense mutations with partial residual function → milder lamellar ichthyosis phenotype (NBCIE/LIE) rather than Harlequin — neonatal collodion baby that sheds to milder generalized ichthyosis; genotype-phenotype exists but full genotyping essential for prognosis",
        ],
        "etiologies": {
            "Biallelic null/frameshift/nonsense (Harlequin, severe)": 20,
            "Compound heterozygous null + missense (intermediate)": 10,
            "Biallelic missense with partial activity (NBCIE/LIE phenotype)": 7,
            "Large deletion biallelic": 2,
            "Splice site biallelic": 1,
        },
        "stats": {
            "neonatal_emergency_collodion_pct": 100,
            "acitretin_response_pct": 80,
            "corneal_involvement_pct": 70,
            "joint_contracture_pct": 55,
            "hypohidrosis_pct": 90,
            "long_term_retinoid_pct": 75,
            "mean_dx_age_months": 0,
            "mean_dx_delay_months": 1,
        },
        "dx_delay_distribution": {"<1m": 35, "1-3m": 3, "3-12m": 1, ">12m": 1},
    },

    # ── SPINK5 — Netherton Syndrome ────────────────────────────────────────
    {
        "gene": "SPINK5",
        "protein": "SPINK5 — Netherton Syndrome, Trichorrhexis Invaginata PATHOGNOMONIC, IgE >2000, Dupilumab",
        "alias": (
            "SPINK5; OMIM gene 605010; Netherton Syndrome OMIM 256500; 5q32; 1094 aa (LEKTI); "
            "~120 kDa; AR; prevalence ~1:200,000. "
            "SPINK5 encodes LEKTI (Lympho-Epithelial Kazal-Type Serine Protease Inhibitor), "
            "a multi-domain serine protease inhibitor expressed in stratified squamous epithelia "
            "(epidermis, thymic epithelium, oesophagus, tonsil). "
            "LEKTI is the principal inhibitor of kallikrein serine proteases KLK5 and KLK7 "
            "in the stratum corneum. "
            "Without LEKTI: KLK5 and KLK7 are UNINHIBITED → hyperactive → "
            "excessive cleavage of corneodesmosomal proteins (CDSN, DSG1) → "
            "premature desquamation of stratum corneum → severe barrier dysfunction; "
            "KLK5 also activates PAR-2 (protease-activated receptor 2) → "
            "pro-inflammatory cytokines → Th2 polarization → atopic disease. "
            "NETHERTON SYNDROME — CLINICAL TRIAD: "
            "(1) TRICHORRHEXIS INVAGINATA ('bamboo hair' or 'ball-and-socket' hair) — "
            "PATHOGNOMONIC for Netherton syndrome; requires polarising microscope or electron microscopy "
            "of hair shaft; the distal hair shaft invaginates ('intussuscepts') into the proximal shaft "
            "→ ball-and-socket appearance; hair is brittle, sparse, and breaks easily; "
            "scalp, eyebrows, eyelashes affected. "
            "(2) ICHTHYOSIS LINEARIS CIRCUMFLEXA (ILC) — migratory serpiginous erythematous plaques "
            "with double-edged scaling (scales at the border of the plaque, not within — "
            "characteristic 'iceberg' periphery); distributed over trunk/extremities. "
            "(3) SEVERE ATOPIC DIATHESIS — anaphylaxis (food-triggered), asthma, allergic rhinitis, "
            "severe atopic dermatitis-like skin; "
            "IgE is DRAMATICALLY ELEVATED: typically >2000 IU/mL, often 5,000–100,000 IU/mL "
            "(values that high are HIGHLY SUSPICIOUS for Netherton if not seen in pure atopic dermatitis). "
            "NEONATAL/INFANT PERIOD: life-threatening in early infancy — "
            "generalised erythroderma (red baby); severe barrier failure → hypothermia, hypernatraemia "
            "(excessive transepidermal water loss), sepsis; failure to thrive; "
            "hyperkalaemia from topical potassium permanganate (avoid); "
            "neonatal Netherton can mimic SSSS (staphylococcal scalded skin syndrome). "
            "IMMUNOLOGICAL: Th2 hyperpolarisation → extremely high IgE; "
            "eosinophilia; multiple food allergies (IgE-mediated anaphylaxis); "
            "recurrent skin infections (S. aureus, HSV). "
            "DIAGNOSIS: hair shaft examination under polarising microscope or SEM (trichorrhexis invaginata); "
            "SPINK5 gene sequencing; IgE level + atopy screen; "
            "LEKTI immunostaining absent in epidermis (diagnostic IF skin biopsy). "
            "TREATMENT: "
            "DUPILUMAB (IL-4Rα monoclonal antibody, FDA 2022 for Netherton) — "
            "blocks IL-4/IL-13 signalling → reduces Th2 inflammation → "
            "improves skin, reduces IgE, reduces atopic symptoms; first targeted therapy; "
            "topical corticosteroids (with caution — barrier failure increases absorption → "
            "HPA axis suppression risk); topical calcineurin inhibitors; "
            "emollients (multiple daily, essential for barrier support); "
            "anaphylaxis preparedness (epinephrine autoinjector); "
            "allergen avoidance."
        ),
        "aa": "1094 aa",
        "kDa": "~120 kDa",
        "locus": "5q32",
        "omim_gene": 605010,
        "omim_disease": 256500,
        "inheritance": "AR; SPINK5 null mutations most common; missense variants rare; no clear genotype-phenotype correlation in severity",
        "gene_class": (
            "SPINK5 encodes LEKTI, a multi-domain serine protease inhibitor (Kazal-type). "
            "Domain structure: 15 Kazal-type serine protease inhibitory domains (domains 1–15); "
            "signal peptide → prodomain → 15 inhibitory units connected by linker peptides. "
            "After cleavage, individual LEKTI domains 6–9 are the principal inhibitors of KLK5; "
            "domains 2, 6–8, 11, 15 inhibit KLK7. "
            "Processing: full-length LEKTI is processed by furin in the trans-Golgi → "
            "individual inhibitory fragments secreted into extracellular space of stratum corneum. "
            "KLK5 and KLK7: kallikrein serine proteases that degrade corneodesmosin (CDSN) and "
            "desmoglein-1 (DSG1) → corneodesmosome dissolution → desquamation. "
            "Without LEKTI: KLK5/KLK7 uncontrolled → premature CDSN/DSG1 cleavage → "
            "premature shedding of stratum corneum → barrier failure + inflammation. "
            "KLK5 also activates PAR-2 on keratinocytes → TSLP, IL-33, LARC release → "
            "Th2 polarisation → IgE synthesis → atopic disease. "
            "Mutation spectrum: truncating/null mutations most common; "
            "no genotype-phenotype correlation; even mild missense mutations → severe phenotype in some patients."
        ),
        "n_patients": 40,
        "key_alerts": [
            "SPINK5-TRICHORRHEXIS-INVAGINATA-PATHOGNOMONIC: Trichorrhexis invaginata ('bamboo hair', ball-and-socket intussusception of hair shaft) is PATHOGNOMONIC for Netherton syndrome — examine hair under POLARISING MICROSCOPE; may need to examine multiple hair sites (scalp, eyebrows, eyelashes) as distribution is patchy; electron microscopy confirms",
            "SPINK5-IGE-OVER-2000: Total IgE >2000 IU/mL (often 5,000–100,000) is characteristic of Netherton syndrome — in the context of ichthyosis + atopic features, markedly elevated IgE (higher than typical atopic dermatitis) is a key diagnostic clue; SPINK5 sequencing confirms",
            "SPINK5-DUPILUMAB-EFFECTIVE: Dupilumab (IL-4Rα monoclonal antibody) is EFFECTIVE in Netherton syndrome and was FDA-approved for this indication (2022 expanded label) — blocks IL-4/IL-13 signalling → reduces Th2-driven inflammation; improves skin, reduces itch, lowers IgE; first targeted therapy for Netherton",
            "SPINK5-NEONATAL-ERYTHRODERMA-EMERGENCY: Neonatal Netherton presents as ERYTHRODERMIC BABY — red, scaling, blistered skin in a sick neonate; can mimic SSSS or neonatal EBS; thermoregulation failure, hypernatraemia from water loss, and sepsis are immediate risks; NICU management with humidified incubator essential",
            "SPINK5-LEKTI-KLK5-KLK7: LEKTI normally inhibits KLK5 and KLK7 (kallikrein serine proteases) in stratum corneum; without LEKTI, these proteases hyperactively degrade corneodesmosin and desmoglein-1 → premature desquamation; KLK5 also activates PAR-2 → TSLP/IL-33 → Th2 skewing → extremely high IgE",
            "SPINK5-ANAPHYLAXIS-RISK: Multiple food-triggered IgE-mediated anaphylaxis is common in Netherton syndrome (IgE-sensitised, severe atopic diathesis) — prescribe EPINEPHRINE AUTOINJECTOR for all patients; identify trigger foods via prick testing/specific IgE; major anaphylaxis risk at food introduction in infants",
            "SPINK5-TOPICAL-STEROID-ABSORPTION: Topical corticosteroids must be used with GREAT CAUTION in Netherton syndrome — severe barrier failure → greatly increased systemic absorption → HPA axis suppression; prefer emollient-first strategies; use lowest-potency steroid for shortest time; calcineurin inhibitors preferred on face",
            "SPINK5-ICHTHYOSIS-LINEARIS-CIRCUMFLEXA: Ichthyosis linearis circumflexa (ILC) — migratory, serpiginous erythematous patches with double-edged scaling at the periphery of plaques — is the SPECIFIC SKIN FINDING of Netherton (along with bamboo hair); may not always be present simultaneously with active atopic flares",
        ],
        "etiologies": {
            "Biallelic null/frameshift/nonsense (most common)": 28,
            "Compound heterozygous null + missense": 8,
            "Biallelic missense (rare)": 3,
            "Splice site biallelic": 1,
        },
        "stats": {
            "trichorrhexis_invaginata_pct": 95,
            "ige_over_2000_pct": 88,
            "neonatal_erythroderma_pct": 70,
            "food_anaphylaxis_pct": 55,
            "ilc_pct": 75,
            "dupilumab_response_pct": 72,
            "s_aureus_colonisation_pct": 80,
            "mean_dx_age_months": 3,
            "mean_dx_delay_months": 24,
        },
        "dx_delay_distribution": {"<6m": 10, "6-24m": 14, "24-60m": 10, ">60m": 6},
    },

    # ── ATP2A2 — Darier Disease ────────────────────────────────────────────
    {
        "gene": "ATP2A2",
        "protein": "ATP2A2 — Darier Disease (Keratosis Follicularis), SERCA2 ER Calcium Pump, Lithium Triggers, Isotretinoin",
        "alias": (
            "ATP2A2; OMIM gene 108740; Darier Disease (Keratosis Follicularis) OMIM 124200; "
            "12q24.11; 1042 aa (SERCA2a isoform) / 1042/1000 aa (isoforms 2a/2b); ~110 kDa; "
            "AD (haploinsufficiency); prevalence ~1:30,000–1:100,000. "
            "ATP2A2 encodes SERCA2 (Sarcoplasmic/Endoplasmic Reticulum Ca2+-ATPase type 2), "
            "the principal calcium pump in the endoplasmic reticulum of keratinocytes. "
            "SERCA2 maintains low cytoplasmic Ca2+ and high ER luminal Ca2+ by actively "
            "pumping Ca2+ from the cytoplasm into the ER lumen using ATP hydrolysis. "
            "Calcium signalling is critical for keratinocyte differentiation and "
            "desmosome assembly/stability. "
            "Without adequate SERCA2 (haploinsufficiency): calcium gradient disrupted → "
            "impaired desmosome assembly → loss of cell-cell adhesion (ACANTHOLYSIS) + "
            "abnormal keratinocyte differentiation → dyskeratosis. "
            "DARIER DISEASE — CLINICAL FEATURES: "
            "DISTRIBUTION: seborrhoeic distribution (oily/hair-bearing areas — "
            "chest/back/scalp/neck/nasolabial folds/retroauricular sulci) — "
            "this distribution is characteristic of Darier's; "
            "LESION TYPE: hyperkeratotic warty/crusted papules ('pitted rough papules') "
            "at follicular and sebaceous gland sites → coalesce → malodorous plaques "
            "('keratosis follicularis' — follicular plugging + hyperkeratosis); "
            "SECONDARY ODOUR: characteristic unpleasant smell from secondary bacterial "
            "(Staphylococcus aureus, Gram-negatives) and yeast (Candida) overgrowth in "
            "fissures between crusted papules. "
            "NAIL SIGNS: V-shaped notching of distal nail edge + longitudinal red/white "
            "streaking (subtungual keratosis) — PATHOGNOMONIC for Darier disease. "
            "PALMAR PITS: punctate palmar pits. "
            "FLEXURAL INVOLVEMENT: malodorous macerated fissures in groins/axillae. "
            "MUCOSAL INVOLVEMENT: white cobblestone papules on palate/buccal mucosa. "
            "TRIGGERS: UV sunlight, HEAT (sweat), friction, LITHIUM (MAJOR TRIGGER — "
            "lithium salt therapy, commonly used for comorbid bipolar disorder, "
            "dramatically worsens Darier disease; an important drug-disease interaction; "
            "lithium competes with Ca2+ and further disrupts SERCA function; "
            "consider mood stabiliser SWITCH if using lithium in Darier patient — "
            "valproate/lamotrigine do not worsen Darier's); "
            "secondary herpes simplex infection (Kaposi varicelliform eruption, KVE): "
            "HSV superinfection of Darier plaques → "
            "widespread vesiculopustular eruption, fever, systemic upset — "
            "TREAT WITH ACICLOVIR IMMEDIATELY. "
            "PSYCHIATRIC COMORBIDITY: depression, bipolar disorder occur at elevated "
            "frequency in Darier disease (possibly SERCA2 expressed in brain). "
            "TREATMENT: "
            "ISOTRETINOIN (systemic retinoid, 0.3–1 mg/kg/day) = FIRST-LINE for severe disease → "
            "reduces dyskeratosis, clears lesions dramatically; "
            "ACITRETIN alternative (better for chronic use, no teratogenic window issue but "
            "teratogenic in pregnancy); "
            "topical retinoids (tretinoin, adapalene) for mild-moderate; "
            "keratolytics (urea 10%, AHA); "
            "antimicrobials (fusidic acid, doxycycline for bacterial superinfection); "
            "sunscreen + sun avoidance; "
            "laser/dermabrasion for localised severe disease."
        ),
        "aa": "1042 aa",
        "kDa": "~110 kDa",
        "locus": "12q24.11",
        "omim_gene": 108740,
        "omim_disease": 124200,
        "inheritance": "AD haploinsufficiency; ~50% de novo mutations; variable expressivity; 1000+ variants described",
        "gene_class": (
            "ATP2A2 encodes SERCA2 (Sarco/Endoplasmic Reticulum Ca2+-ATPase 2). "
            "Domain structure: transmembrane domain (TM1–10, 10 helices; Ca2+ binding sites in TM4/5/6/8) → "
            "actuator domain A (gateway helix, Glu309, dephosphorylation) → "
            "phosphorylation domain P (Asp351, autophosphorylation from ATP) → "
            "nucleotide-binding domain N (ATP binding). "
            "ATP2A2 isoforms: SERCA2a (cardiac/slow-twitch muscle, 1042 aa) and "
            "SERCA2b (ubiquitous including keratinocytes, 1042 aa + 4 aa C-terminal extension → "
            "higher Ca2+ affinity). "
            "Reaction cycle: (1) Ca2+ binds TM sites from cytoplasm → (2) ATP binds N-domain → "
            "(3) phosphorylation of Asp351 → (4) Ca2+ translocated to ER lumen → "
            "(5) dephosphorylation + Ca2+ release into ER → (6) reset. "
            "In keratinocytes: SERCA2b is the relevant isoform; haploinsufficiency → "
            "~50% SERCA2 activity → elevated cytoplasmic Ca2+ → impaired Ca2+-dependent "
            "desmosome assembly (desmoplakin, plakophilin targeting fails) → acantholysis. "
            "Mutation spectrum: missense in P-domain, TM helices, A-domain; "
            "nonsense/frameshift; splice site. No dominant-negative — pure haploinsufficiency. "
            "Heterologous SERCA2b expression experiments confirm loss of function."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ATP2A2-LITHIUM-TRIGGERS-WORSENS: LITHIUM therapy (commonly used for comorbid bipolar disorder, which occurs at increased frequency in Darier disease) DRAMATICALLY WORSENS Darier disease — lithium competes with Ca2+ and further impairs SERCA2 function; consider switching to valproate or lamotrigine in Darier patients needing mood stabilisation",
            "ATP2A2-ISOTRETINOIN-FIRST-LINE: Isotretinoin (0.3–1 mg/kg/day) is FIRST-LINE systemic therapy for moderate-severe Darier disease — reduces dyskeratosis and clears lesions dramatically; acitretin is an alternative for long-term use in non-reproductive-aged patients; systemic retinoids are the most effective available treatment",
            "ATP2A2-KAPOSI-VARICELLIFORM-ERUPTION: HSV superinfection of Darier plaques causes Kaposi varicelliform eruption (KVE) — widespread, rapidly spreading vesiculopustular eruption, fever, systemic upset; TREAT IMMEDIATELY WITH SYSTEMIC ACICLOVIR; KVE in Darier can be life-threatening if not treated promptly",
            "ATP2A2-NAIL-SIGNS-PATHOGNOMONIC: V-shaped notching of the distal nail edge plus longitudinal red and white striping (subungual keratosis) are PATHOGNOMONIC for Darier disease — examine nails in all suspected cases; nail signs can be the first clinical clue before prominent skin involvement",
            "ATP2A2-SEBORRHOEIC-DISTRIBUTION: Darier disease lesions follow a SEBORRHOEIC DISTRIBUTION (chest, back, scalp, nasolabial folds, retroauricular, neck flexures) — this distribution distinguishes Darier from acne (face/back without flexure involvement) and Hailey-Hailey (axillae/groins, ATP2C1 mutation)",
            "ATP2A2-SECONDARY-ODOUR-INFECTION: The characteristic MALODOROUS SMELL of Darier disease plaques results from bacterial (Staphylococcus aureus, Gram-negative) and yeast (Candida) overgrowth in fissures between crusted papules; antiseptic washes (chlorhexidine), topical antibiotics, and regular washing of lesions are essential",
            "ATP2A2-PSYCHIATRIC-COMORBIDITY: Depression and bipolar disorder occur at ELEVATED FREQUENCY in Darier disease (SERCA2 is also expressed in brain neurons — neurological haploinsufficiency effect); screen all Darier patients for mood disorders; note that lithium (common mood stabiliser) MUST BE AVOIDED in Darier",
            "ATP2A2-HAILEY-HAILEY-DDX: Hailey-Hailey disease (benign familial pemphigus, ATP2C1 mutation, Golgi Ca2+ pump) closely mimics Darier — BOTH cause acantholysis and are AD; DDx: Darier = seborrhoeic distribution + nail signs + follicular papules; Hailey-Hailey = axillary/groin macerated plaques, no nail V-notching, no mucosal lesions",
        ],
        "etiologies": {
            "Missense in P-domain or TM helix (haploinsufficiency)": 20,
            "Frameshift/nonsense (truncating, haploinsufficiency)": 12,
            "Splice site mutations": 5,
            "De novo novel missense": 2,
            "A-domain or N-domain missense": 1,
        },
        "stats": {
            "seborrhoeic_distribution_pct": 98,
            "nail_signs_pct": 80,
            "lithium_worsening_pct": 60,
            "kvs_herpetic_superinfection_pct": 25,
            "psychiatric_comorbidity_pct": 30,
            "isotretinoin_response_pct": 85,
            "flexural_involvement_pct": 65,
            "mean_dx_age": 15,
            "mean_dx_delay_months": 36,
        },
        "dx_delay_distribution": {"<12m": 8, "12-36m": 12, "36-60m": 10, ">60m": 10},
    },

    # ── STS — X-Linked Ichthyosis ──────────────────────────────────────────
    {
        "gene": "STS",
        "protein": "STS — X-Linked Ichthyosis, Corneal Opacities, Contiguous Kallmann/ANOS1, Placental Sulfatase Deficiency",
        "alias": (
            "STS; OMIM gene 300747; X-Linked Ichthyosis (XLI) OMIM 308100; Xp22.31; "
            "583 aa; ~65 kDa; XLR (X-linked recessive); "
            "prevalence 1:2,000–1:6,000 males. "
            "STS encodes steroid sulfatase (arylsulfatase C / ARSC1), a microsomal enzyme expressed "
            "in many tissues including skin (keratinocytes), liver, placenta, brain, testis. "
            "STS hydrolyses sulfate esters of steroids: "
            "DHEA-S → DHEA (androgen precursor); cholesterol sulfate → cholesterol; "
            "oestrone sulfate → oestrone; "
            "in the skin, the primary substrate is CHOLESTEROL SULFATE (CS): "
            "normal STS: CS accumulation in stratum corneum → STS degrades CS → "
            "releases cholesterol → part of lipid lamellar membrane assembly; "
            "without STS: CS ACCUMULATES in stratum corneum → "
            "CS at elevated concentrations inhibits serine proteases (KLK5, KLK7) → "
            "impaired corneodesmosome dissolution → FAILURE OF DESQUAMATION → "
            "scaling (retained stratum corneum layers). "
            "CLINICAL FEATURES: "
            "ONSET: males affected; symptoms appear WEEKS AFTER BIRTH (INITIAL NEONATAL SHEDDING "
            "— collodion-like desquamation in first weeks of life is actually physiological neonatal "
            "shedding; XLI ichthyosis typically appears and becomes apparent in the first 3–6 months); "
            "SCALE: dark, brown-grey polygonal scales, predominantly on flanks, neck, trunk, "
            "forearms, lower legs; sparing of face/central chest/palms/soles (KEY FEATURE — "
            "distinguishes XLI from autosomal recessive ichthyoses); "
            "scaling improves in summer and worsens in winter; "
            "NO ERYTHRODERMA (unlike ARCI — non-inflammatory scale); "
            "CORNEAL OPACITIES: deep stromal (Bowman membrane level) corneal opacities → "
            "occur in >50% of males (and some female carriers) → "
            "do NOT affect vision (punctate opacities, not axial) but ARE A DIAGNOSTIC CLUE; "
            "TESTICULAR MALDESCENT: cryptorchidism in 25% — likely due to role of STS "
            "in placental androgen synthesis. "
            "PLACENTAL SULFATASE DEFICIENCY: STS is HIGHLY expressed in placenta → "
            "affects placental oestrogen biosynthesis → "
            "IMPAIRED ONSET OF LABOUR (failure of labour to initiate spontaneously, "
            "prolonged pregnancy, need for induction/emergency CS) — "
            "a clue to XLI diagnosis is a mother who required labour induction with a male infant; "
            "low maternal urinary oestriol in pregnancy (reflects placental STS deficiency). "
            "CONTIGUOUS GENE DELETION SYNDROME (Xp22.31): "
            "large deletions at Xp22.31 can delete STS + adjacent genes: "
            "ANOS1 (KAL1) → Kallmann syndrome (anosmia + hypogonadotrophic hypogonadism); "
            "CDKL5 → atypical Rett syndrome; "
            "SHOX → short stature; "
            "NLGN4X → autism spectrum disorder; "
            "COMPLEX PHENOTYPE (XLI + Kallmann + other features) = SUSPECT CONTIGUOUS DELETION "
            "→ MULTIPLEX LIGATION-DEPENDENT PROBE AMPLIFICATION (MLPA) or chromosomal microarray. "
            "DIAGNOSIS: STS enzyme activity in leucocytes/fibroblasts (low/absent); "
            "plasma cholesterol sulfate elevated; MLPA/array-CGH (most XLI = deletion); "
            "STS gene sequencing for point mutations. "
            "TREATMENT: emollients (urea 5–10%, lactic acid); keratolytics "
            "(ammonium lactate 12%, alpha-hydroxy acids); "
            "topical retinoids for resistant areas; "
            "systemic retinoids (acitretin) for severe cases; "
            "manage cryptorchidism (orchidopexy by 18 months); "
            "ophthalmology referral for corneal opacity documentation."
        ),
        "aa": "583 aa",
        "kDa": "~65 kDa",
        "locus": "Xp22.31",
        "omim_gene": 300747,
        "omim_disease": 308100,
        "inheritance": "XLR; ~90% of cases = complete STS gene DELETION (detected by MLPA/array-CGH); female carriers typically unaffected or mild; contiguous deletion → Kallmann/ANOS1",
        "gene_class": (
            "STS encodes steroid sulfatase (arylsulfatase C / microsomal sulfatase). "
            "Domain structure: N-terminal signal peptide → type II transmembrane domain (single TM helix, "
            "Arg-basic stretch, ER/microsomal retention) → "
            "large luminal catalytic domain (sulfatase family fold: central β-sheet + flanking helices). "
            "Active site: Cys99 (formylglycine — post-translational modification by SUMF1 "
            "generates the catalytic Cα-formylglycine, Cfα-FGly, essential for sulfatase activity; "
            "SUMF1 = Sulfatase Modifying Factor 1; SUMF1 mutations → multiple sulfatase deficiency). "
            "Reaction: steroid sulfate → steroid + SO4^2- (sulfate ester hydrolysis). "
            "Substrates in skin: cholesterol sulfate (primary), DHEAS, pregnenolone sulfate. "
            "Excess CS in stratum corneum: at high CS concentrations, CS inhibits kallikrein "
            "serine proteases (KLK5/KLK7) → corneodesmosome not dissolved → scale retention. "
            "STS gene at Xp22.31: 10 exons spanning ~147 kb; "
            "~90% of XLI = complete/partial gene deletion (not detectable by sequencing alone) → "
            "MLPA or array-CGH mandatory when STS sequencing negative. "
            "Gene escapes X-inactivation (like SHOX) in some tissues — explains mild symptoms in some female carriers."
        ),
        "n_patients": 40,
        "key_alerts": [
            "STS-CORNEAL-OPACITIES-DEEP-STROMAL: Deep stromal corneal opacities (punctate, at Bowman membrane level) occur in >50% of XLI males and some female carriers — DO NOT affect vision but are an important DIAGNOSTIC CLUE; slit-lamp examination by ophthalmologist is essential in suspected XLI; useful in equivocal cases",
            "STS-CONTIGUOUS-KALLMANN-DELETION: Large Xp22.31 deletions encompassing STS + ANOS1 (KAL1) → XLI + KALLMANN SYNDROME (anosmia + hypogonadotrophic hypogonadism + absent puberty) — ALWAYS perform MLPA or chromosomal microarray in XLI patients to exclude contiguous gene deletion; assess olfaction and pubertal development",
            "STS-PLACENTAL-SULFATASE-DEFICIENCY: STS deficiency in the placenta → impaired placental oestrogen synthesis → FAILURE TO INITIATE LABOUR (prolonged pregnancy, need for induction or emergency caesarean section); low maternal urinary oestriol in pregnancy is a biochemical clue to XLI in the fetus; obstetric history of labour induction",
            "STS-CRYPTORCHIDISM-25PCT: Cryptorchidism (undescended testes) occurs in ~25% of XLI males — refer for orchidopexy before 18 months to reduce risk of testicular cancer and infertility; likely due to STS role in placental androgen biosynthesis affecting fetal testicular descent",
            "STS-DELETION-NOT-SEQUENCING: ~90% of XLI cases are caused by COMPLETE STS GENE DELETION — STS sequencing will be NEGATIVE in these patients; MLPA or chromosomal microarray (array-CGH) is MANDATORY when clinical features are consistent with XLI; do not stop at a negative sequencing result",
            "STS-FACE-PALMS-SPARED: XLI scales are predominantly on flanks, neck, lower extremities with SPARING OF FACE, PALMS, SOLES, and central chest — this pattern distinguishes XLI from autosomal recessive congenital ichthyoses (ARCI) which involve palms/soles and face more; non-inflammatory dark polygonal scales are characteristic",
            "STS-CHOLESTEROL-SULFATE-MECHANISM: STS deficiency → CHOLESTEROL SULFATE ACCUMULATES in stratum corneum → at high concentrations CS inhibits KLK5/KLK7 serine proteases → corneodesmosome proteins not cleaved → desquamation fails → scale retention; opposite to Netherton (SPINK5) where KLK5/7 are OVERACTIVE",
            "STS-FEMALE-CARRIERS-MILD: Female carriers of STS mutations are usually CLINICALLY UNAFFECTED (X-linked recessive) but may have mild scaling on lower legs, and approximately 25% develop deep stromal corneal opacities; carrier females had placentas deficient in STS (labour induction history in carrier mothers)",
        ],
        "etiologies": {
            "Complete STS gene deletion (Xp22.31, ~90% of XLI)": 34,
            "Partial STS deletion (intragenic)": 3,
            "STS point mutation/missense (rare)": 2,
            "Contiguous deletion XLI + Kallmann (STS + ANOS1)": 1,
        },
        "stats": {
            "complete_deletion_pct": 88,
            "corneal_opacity_pct": 55,
            "cryptorchidism_pct": 25,
            "labour_induction_history_pct": 60,
            "kallmann_contiguous_pct": 5,
            "face_sparing_pct": 92,
            "emollient_controlled_pct": 70,
            "mean_dx_age_months": 6,
            "mean_dx_delay_months": 48,
        },
        "dx_delay_distribution": {"<12m": 10, "12-36m": 12, "36-60m": 10, ">60m": 8},
    },

    # ── IKBKG — Incontinentia Pigmenti ────────────────────────────────────
    {
        "gene": "IKBKG",
        "protein": "IKBKG — Incontinentia Pigmenti (NEMO), Males Usually Lethal, 4-Stage Skin, Retinal Vasculopathy, Dental/CNS",
        "alias": (
            "IKBKG (also NEMO — NF-κB Essential MOdulator); OMIM gene 300248; "
            "Incontinentia Pigmenti (IP) OMIM 308300; Xq28; 419 aa; ~48 kDa; "
            "X-linked dominant (XLD); prevalence ~1:50,000. "
            "IKBKG encodes NEMO (NF-κB Essential Modulator, also known as IKKγ), "
            "the regulatory subunit of the IκB kinase (IKK) complex. "
            "The IKK complex (IKKα + IKKβ + IKKγ/NEMO) is the master activator of NF-κB: "
            "IKK phosphorylates IκB → IκB ubiquitinated and degraded → "
            "NF-κB released → translocates to nucleus → activates survival/inflammatory genes. "
            "Without NEMO: NF-κB CANNOT be activated → cells hypersensitive to TNF-α-induced apoptosis. "
            "X-LINKED DOMINANT LETHALITY IN MALES: "
            "hemizygous IKBKG loss in males → NF-κB signalling completely abolished in all cells → "
            "massive TNF-α-induced apoptosis → LETHAL IN UTERO or neonatally "
            "(explains ~80% pregnancy loss of affected male fetuses); "
            "surviving males are MOSAIC (KLINEFELTER 47,XXY — one normal X protected; "
            "or somatic mosaicism with mosaic IKBKG mutation). "
            "FEMALES (heterozygous): NF-κB signalling from normal X allele protects most cells → "
            "X-inactivation selects AGAINST cells expressing the mutant IKBKG allele → "
            "skewed X-inactivation in peripheral blood → "
            "mosaicism produces the CHARACTERISTIC 4-STAGE SKIN PATTERN. "
            "FOUR STAGES OF INCONTINENTIA PIGMENTI SKIN: "
            "STAGE 1 — VESICULAR/BULLOUS (birth–weeks): vesiculobullous eruption "
            "following Blaschko's lines on trunk/extremities; eosinophilia (up to 50%); "
            "eosinophilic intracellular spongiosis on skin biopsy. "
            "STAGE 2 — VERRUCOUS (weeks–months): hyperkeratotic verrucous plaques "
            "along Blaschko's lines. "
            "STAGE 3 — HYPERPIGMENTED (months–years): SWIRLING BROWN HYPERPIGMENTATION "
            "along Blaschko's lines — this is the MOST RECOGNISABLE stage and gives the disease "
            "its name ('incontinentia' = melanin from upper epidermis 'incontinent' = dropped "
            "into dermis — melanin incontinence); PATHOGNOMONIC distribution. "
            "STAGE 4 — HYPOPIGMENTED/ATROPHIC (adult): pale, hairless, atrophic streaks "
            "following Blaschko's lines; not always present; stages do not always occur sequentially. "
            "EXTRACUTANEOUS FEATURES (critical): "
            "EYES: RETINAL VASCULOPATHY — most important complication: "
            "retinal vascular abnormalities (neovascularisation, retinal detachment) → "
            "BLINDNESS; LEADING CAUSE OF BLINDNESS in IP; "
            "RetCam screening mandatory from birth through childhood; "
            "laser/anti-VEGF for neovascularisation. "
            "CNS: seizures (up to 30%), strokes (NF-κB-dependent endothelial function), "
            "intellectual disability, microcephaly, white matter lesions on MRI. "
            "DENTAL: ANODONTIA or HYPODONTIA (missing teeth), PEGGED/CONICAL TEETH "
            "(dental abnormalities in 70–80%); dental X-rays essential. "
            "HAIR: alopecia (patchy or cicatricial). "
            "NAILS: nail dystrophy. "
            "MUTATIONS: exon 4–10 deletion of IKBKG (in 80% of IP cases) → "
            "detected by exon-specific PCR; remaining 20% = point mutations. "
            "MOLECULAR DIAGNOSIS: PCR for exon 4–10 deletion (standard); full IKBKG sequencing; "
            "skewed X-inactivation analysis on peripheral blood (supports diagnosis). "
            "MANAGEMENT: ophthalmology RetCam every 3 months in first 3 years; "
            "laser/anti-VEGF for retinal vasculopathy; neurology for seizures (AED); "
            "dental specialist; dermatology; genetics."
        ),
        "aa": "419 aa",
        "kDa": "~48 kDa",
        "locus": "Xq28",
        "omim_gene": 300248,
        "omim_disease": 308300,
        "inheritance": "XLD; hemizygous males usually lethal (>80% miscarriage); surviving males = somatic mosaic or Klinefelter (47,XXY); heterozygous females = classic IP; exon 4–10 deletion in 80%",
        "gene_class": (
            "IKBKG encodes NEMO (IKKγ), the non-catalytic regulatory subunit of the IκB kinase (IKK) complex. "
            "Domain structure: N-terminal dimerization domain (DD) → "
            "coiled-coil 1 (CC1) → leucine zipper (LZ) → "
            "zinc finger (ZF, UBAN domain — ubiquitin binding at Asp311-Lys321) → "
            "coiled-coil 2 (CC2) → "
            "C-terminal NEMO ubiquitin-binding domain (NUB). "
            "Function: NEMO bridges upstream signalling (RIP1, TRAF2/6 ubiquitin chains) to "
            "catalytic IKKα/IKKβ → allows IKKβ to phosphorylate IκBα (Ser32/36) → "
            "IκBα ubiquitinated by SCF-βTrCP E3 ligase → proteasomal degradation → "
            "NF-κB (p65/p50) freed → nuclear translocation → NF-κB target genes "
            "(anti-apoptotic: BCL-xL, cIAP, c-FLIP; inflammatory: cytokines). "
            "Without NEMO: cells hypersensitive to TNF-α-induced apoptosis "
            "(normally NF-κB-activated anti-apoptotic genes protect cells from TNF-α); "
            "in males (hemizygous): all cells susceptible → massive apoptosis → lethal. "
            "In females (heterozygous): cells expressing mutant IKBKG are eliminated by "
            "negative selection (X-inactivation skewed) → surviving cells form mosaic streaks "
            "along Blaschko's lines (clonal patches of cells sharing same X-inactivation). "
            "Exon 4–10 deletion (80%): removes coiled-coil + ZF domains → complete loss of function. "
            "Mutation spectrum: deletion (most common, exon 4–10); "
            "missense in UBAN domain → ectodermal dysplasia with immunodeficiency (EDA-ID) phenotype."
        ),
        "n_patients": 40,
        "key_alerts": [
            "IKBKG-MALES-USUALLY-LETHAL: Hemizygous IKBKG loss in males is LETHAL IN UTERO in ~80% (complete NF-κB failure → massive TNF-α-induced apoptosis); surviving males have SOMATIC MOSAICISM or KLINEFELTER (47,XXY protecting one normal X); UNEXPLAINED RECURRENT MISCARRIAGES of male fetuses in maternal IP pedigrees",
            "IKBKG-RETINAL-VASCULOPATHY-LEADING-BLINDNESS: Retinal vasculopathy (neovascularisation, retinal detachment) is THE LEADING CAUSE OF BLINDNESS in IP — RETCAM screening from birth (every 3 months for first 3 years, then 6-monthly) is MANDATORY; laser photocoagulation and/or anti-VEGF for neovascularisation URGENTLY to preserve vision",
            "IKBKG-4-STAGE-BLASCHKO: Incontinentia Pigmenti follows Blaschko's lines in 4 stages — vesicular → verrucous → hyperpigmented (swirling brown = PATHOGNOMONIC stage 3, 'melanin incontinence') → hypopigmented/atrophic; not all stages present simultaneously; Blaschko distribution is critical for diagnosis",
            "IKBKG-DENTAL-ANOMALIES: Dental abnormalities occur in 70–80% of IP — PEGGED or CONICAL TEETH, hypodontia (missing teeth), anodontia, delayed eruption; dental X-ray (orthopantomogram) is mandatory in all IP cases; early orthodontic/prosthodontic planning essential for permanent dentition",
            "IKBKG-CNS-SEIZURES-STROKE: CNS involvement (seizures, stroke-like episodes, white matter lesions, intellectual disability) occurs in 30% of IP patients — NF-κB-dependent endothelial survival is impaired; brain MRI + EEG in all IP patients; seizures require prompt AED therapy; stroke in infancy/childhood needs urgent imaging",
            "IKBKG-EXON-4-10-DELETION-PCR: 80% of IP cases are caused by EXON 4–10 DELETION of IKBKG — detected by exon-specific PCR (not standard sequencing); PCR for this deletion must be the FIRST-LINE molecular test in suspected IP; if PCR negative, proceed to full IKBKG sequencing for point mutations",
            "IKBKG-SKEWED-X-INACTIVATION: Highly skewed X-inactivation (>90:10) on peripheral blood MKBKG allele analysis is a SUPPORTIVE DIAGNOSTIC FINDING in female IP carriers — reflects negative selection against cells expressing the mutant IKBKG allele; useful when clinical features are equivocal",
            "IKBKG-NEMO-NF-KB-ECTODERMAL: Hypomorphic IKBKG missense mutations in the UBAN domain cause ECTODERMAL DYSPLASIA WITH IMMUNODEFICIENCY (EDA-ID) rather than IP — associated with severe mycobacterial/bacterial infections (NF-κB-dependent immune signalling partially preserved); distinct from classic IP deletion phenotype; distinguish by molecular testing",
        ],
        "etiologies": {
            "Exon 4–10 deletion IKBKG (most common, ~80%)": 32,
            "IKBKG point mutation/missense (non-deletion, ~20%)": 5,
            "Somatic mosaic male (Klinefelter or mosaic)": 2,
            "Novel deletion variant (atypical breakpoints)": 1,
        },
        "stats": {
            "female_heterozygous_classic_ip_pct": 95,
            "male_lethal_in_utero_pct": 80,
            "retinal_vasculopathy_pct": 35,
            "cns_seizures_stroke_pct": 30,
            "dental_anomalies_pct": 78,
            "blaschko_hyperpigmentation_pct": 95,
            "exon_4_10_deletion_pct": 80,
            "visual_impairment_pct": 20,
            "mean_dx_age_months": 1,
            "mean_dx_delay_months": 6,
        },
        "dx_delay_distribution": {"<3m": 22, "3-12m": 10, "12-36m": 5, ">36m": 3},
    },
]


def _make_patients(gene_entry, rng):
    """Generate synthetic patient records for one gene."""
    gene = gene_entry["gene"]
    n = gene_entry["n_patients"]
    ages = [rng.randint(0, 50) for _ in range(n)]
    delays = [rng.choice([1, 2, 3, 6, 12, 24, 36, 48, 60]) for _ in range(n)]
    etiol_keys = list(gene_entry["etiologies"].keys())
    etiol_weights = list(gene_entry["etiologies"].values())
    etiols = rng.choices(etiol_keys, weights=etiol_weights, k=n)
    patients = []
    for i in range(n):
        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "dx_age": ages[i],
            "dx_delay_months": delays[i],
            "variant_class": etiols[i],
        })
    return patients


def _build_cohort():
    all_data = {}
    for idx, ge in enumerate(GENO_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        ge_copy = dict(ge)
        ge_copy["seed"] = seed
        ge_copy["patients"] = _make_patients(ge, rng)
        all_data[ge["gene"]] = ge_copy
    return all_data


_COHORT = _build_cohort()


def get_overview():
    genes_summary = []
    total = 0
    all_dx_ages = []
    all_delays = []
    top_alerts = []
    for gene, info in _COHORT.items():
        n = info["n_patients"]
        total += n
        pts = info["patients"]
        ages = [p["dx_age"] for p in pts]
        delays = [p["dx_delay_months"] for p in pts]
        all_dx_ages.extend(ages)
        all_delays.extend(delays)
        genes_summary.append({
            "gene": gene,
            "protein_short": info["protein"].split(" — ")[0],
            "n_patients": n,
            "locus": info["locus"],
            "inheritance": info["inheritance"].split(";")[0],
            "omim_disease": info["omim_disease"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
        })
        top_alerts.extend(info["key_alerts"][:2])
    aggregate_stats = {
        "total_patients": total,
        "mean_dx_age_years": round(sum(all_dx_ages) / len(all_dx_ages), 1),
        "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
        "col7a1_scc_risk_pct": 90,
        "lamb3_herlitz_pct": 40,
        "krt5_dowling_meara_pct": 40,
        "abca12_acitretin_response_pct": 80,
        "spink5_ige_over_2000_pct": 88,
        "atp2a2_isotretinoin_response_pct": 85,
        "sts_corneal_opacity_pct": 55,
        "ikbkg_retinal_vasculopathy_pct": 35,
        "cascade_tested_pct": 62,
    }
    return {
        "atlas": "Hereditary-Genodermatoses-Atlas",
        "genes": genes_summary,
        "aggregate_stats": aggregate_stats,
        "top_alerts": top_alerts,
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
    }


def get_breakdown():
    result = {}
    for gene, info in _COHORT.items():
        pts = info["patients"]
        result[gene] = {
            "gene": gene,
            "n_patients": info["n_patients"],
            "alias": info["alias"],
            "gene_class": info["gene_class"],
            "locus": info["locus"],
            "aa": info["aa"],
            "kDa": info["kDa"],
            "omim_gene": info["omim_gene"],
            "omim_disease": info["omim_disease"],
            "inheritance": info["inheritance"],
            "key_alerts": info["key_alerts"],
            "etiologies": info["etiologies"],
            "stats": info["stats"],
            "dx_delay_distribution": info["dx_delay_distribution"],
            "patients": pts[:10],
        }
    return result


def get_definitions():
    return {
        "atlas": "Hereditary-Genodermatoses-Atlas",
        "concepts": {
            "Epidermolysis Bullosa Classification (COL7A1/LAMB3/KRT5)": (
                "Epidermolysis Bullosa (EB) is classified by cleavage plane on transmission electron microscopy (TEM) "
                "and immunofluorescence (IF) mapping, then confirmed by molecular genetics. "
                "EBS (Simplex, KRT5/KRT14): INTRAEPIDERMAL cleavage (within basal keratinocytes) → "
                "type IV collagen ABOVE the blister; keratin IF network disrupted; heat-triggered; Kobner. "
                "JEB (Junctional, LAMB3/LAMA3/LAMC2): cleavage WITHIN LAMINA LUCIDA → "
                "type XVII collagen (BP180) and laminin-332 absent on IF; enamel hypoplasia PATHOGNOMONIC; "
                "Herlitz = null/null LAMB3 (lethal); non-Herlitz = missense/compound → survival. "
                "DEB (Dystrophic, COL7A1): cleavage SUB-LAMINA DENSA → type VII collagen absent/reduced on IF; "
                "anchoring fibrils absent on TEM; RDEB (AR) most severe; SCC risk leading cause of death; "
                "beremagene gene therapy FDA 2023 for RDEB. "
                "Diagnosis algorithm: skin biopsy (fresh, transport in Michel's medium) → "
                "IF panel (anti-type IV collagen, anti-BP180, anti-laminin-332, anti-type VII collagen) → "
                "TEM → gene panel (COL7A1, LAMB3, LAMA3, LAMC2, KRT5, KRT14, others). "
                "Mutation → severity: null/null = maximum severity (Herlitz, RDEB-severe); "
                "missense with partial function = intermediate; AD mutations = milder."
            ),
            "Ichthyosis and Lipid Barrier Disorders (ABCA12/STS)": (
                "Ichthyoses are disorders of epidermal differentiation characterised by scaling. "
                "ABCA12 (Harlequin Ichthyosis, AR): ABCA12 = lamellar body lipid transporter; "
                "loads glucosylceramides into lamellar bodies (Odland bodies) for extrusion into "
                "stratum corneum intercellular space → lipid lamellar membrane = permeability barrier. "
                "Without ABCA12: EMPTY LAMELLAR BODIES (TEM diagnostic) → no lipid barrier → "
                "neonatal emergency (collodion membrane, ectropion, respiratory restriction); "
                "acitretin (1 mg/kg/day) MUST be started ASAP — promotes shedding; life-saving. "
                "STS (X-Linked Ichthyosis, XLR): steroid sulfatase hydrolyses cholesterol sulfate (CS) → cholesterol "
                "in stratum corneum; without STS: CS ACCUMULATES → inhibits KLK5/KLK7 (serine proteases) → "
                "impaired corneodesmosome dissolution → desquamation failure → scaling. "
                "OPPOSITE mechanism to Netherton (SPINK5): in XLI, KLK5/7 are INHIBITED by excess CS; "
                "in Netherton, KLK5/7 are OVERACTIVE (no LEKTI to inhibit them). "
                "XLI presents in males: dark polygonal scales; face/palms SPARED; "
                "corneal opacities (>50%, diagnostic clue); cryptorchidism 25%; "
                "contiguous Xp22.31 deletion → Kallmann syndrome (ANOS1) + XLI. "
                "90% of XLI = complete STS gene deletion → MLPA or array-CGH essential "
                "(sequencing alone will miss most cases). "
                "Treatment: emollients + keratolytics for both; systemic retinoids for severe ABCA12."
            ),
            "Immune Dysregulation Genodermatoses (SPINK5)": (
                "SPINK5 encodes LEKTI (Lympho-Epithelial Kazal-Type Serine Protease Inhibitor), "
                "the principal inhibitor of epidermal KLK5 and KLK7 in stratum corneum. "
                "Without LEKTI: KLK5/KLK7 hyperactivity → excess corneodesmosin (CDSN) and DSG1 cleavage → "
                "premature desquamation + barrier failure; "
                "KLK5 activates PAR-2 on keratinocytes → TSLP, IL-33 → Th2 polarisation → "
                "IgE synthesis (typically >2000 IU/mL, often >10,000 IU/mL) + atopic disease. "
                "Netherton Triad: (1) trichorrhexis invaginata ('bamboo hair', ball-and-socket — PATHOGNOMONIC); "
                "(2) ichthyosis linearis circumflexa (ILC — migratory erythematous plaques, double-edged scale); "
                "(3) severe atopic diathesis (IgE-mediated anaphylaxis, asthma, urticaria). "
                "Neonatal period: life-threatening erythroderma, hypernatraemia, hypothermia, sepsis. "
                "Dupilumab (IL-4Rα mAb) FDA-approved for Netherton — blocks IL-4/IL-13 → reduces Th2 → "
                "improves skin, reduces IgE; first targeted therapy. "
                "Hair examination under POLARISING MICROSCOPE essential for diagnosis — "
                "trichorrhexis invaginata may be patchy; examine multiple sites including eyebrows. "
                "LEKTI absent on skin biopsy IF confirms diagnosis before genetic results."
            ),
            "ER Calcium Pump Disorders (ATP2A2)": (
                "ATP2A2 encodes SERCA2 (Sarcoplasmic/Endoplasmic Reticulum Ca2+-ATPase 2), "
                "the ER calcium pump in keratinocytes. "
                "Normal SERCA2: maintains high ER luminal Ca2+ and low cytoplasmic Ca2+; "
                "Ca2+ gradient drives desmosome assembly (Ca2+-dependent desmoplakin/plakophilin targeting) and "
                "keratinocyte differentiation. "
                "Haploinsufficiency (Darier disease, AD): ~50% SERCA2 → elevated cytoplasmic Ca2+ → "
                "impaired desmosome assembly → ACANTHOLYSIS + dyskeratosis. "
                "Clinical: warty hyperkeratotic papules in seborrhoeic distribution (chest/back/scalp/neck); "
                "characteristic malodour (S. aureus/Candida overgrowth); V-shaped nail notching PATHOGNOMONIC; "
                "mucosal cobblestone papules. "
                "Triggers: UV light, heat, LITHIUM (major drug trigger — impairs Ca2+ homeostasis further; "
                "must switch mood stabiliser to valproate/lamotrigine in Darier patients). "
                "Herpetic superinfection (Kaposi varicelliform eruption, KVE): treat IMMEDIATELY with systemic aciclovir. "
                "Treatment: isotretinoin/acitretin first-line for moderate-severe; topical retinoids for mild. "
                "Hailey-Hailey (ATP2C1, Golgi Ca2+ pump, AD): similar acantholysis but "
                "axillary/groin predominance, no nail V-notching, no mucosal involvement — key DDx."
            ),
            "X-Linked Genodermatoses (STS/IKBKG)": (
                "Both STS and IKBKG map to the X chromosome but have fundamentally different genetics. "
                "STS (Xp22.31, XLR): X-linked recessive; males affected (hemizygous loss); "
                "females carriers (usually unaffected or mild); "
                "~90% = complete gene deletion → MLPA/array-CGH mandatory; "
                "contiguous Xp22.31 deletion → XLI + Kallmann syndrome (ANOS1/KAL1) + potentially CDKL5/SHOX. "
                "Corneal opacities (deep stromal, non-visual) + cryptorchidism + "
                "impaired labour onset (placental STS deficiency) = characteristic triad in affected males. "
                "IKBKG (Xq28, XLD): X-linked dominant; "
                "hemizygous males = LETHAL (>80% intrauterine demise — no NF-κB → TNF-α apoptosis); "
                "heterozygous females = Incontinentia Pigmenti (IP); "
                "surviving males = mosaic (somatic or Klinefelter 47,XXY). "
                "IP key features: 4-stage skin along Blaschko's lines; "
                "retinal vasculopathy (blindness risk — RetCam mandatory); "
                "dental anomalies (pegged/conical teeth, hypodontia); CNS (seizures, stroke). "
                "Exon 4–10 IKBKG deletion (~80% of IP) detected by PCR, not sequencing. "
                "The XLD lethality pattern (excess female births, unexplained male miscarriages) "
                "is the key diagnostic clue in IKBKG family pedigrees."
            ),
        },
        "pharmacological_distinctions": [
            "Acitretin vs Isotretinoin in genodermatoses: acitretin (aromatic retinoid) preferred for chronic ichthyosis/Darier (no teratogenic window after stopping — remains teratogenic 3 years post-cessation, unlike isotretinoin's 1-month window); isotretinoin preferred for Darier when future pregnancy possible (1-month wash-out); both equally effective for scaling and dyskeratosis; acitretin is first choice for Harlequin Ichthyosis (ABCA12) life-saving neonatal use at 1 mg/kg/day",
            "Dupilumab (IL-4Rα mAb) in Netherton/SPINK5: FDA approved for Netherton syndrome (2022) — blocks IL-4/IL-13 signalling to reduce Th2-driven KLK5/KLK7 cascade; reduces IgE, improves skin barrier, reduces pruritus; first targeted disease-modifying therapy for Netherton; does not restore LEKTI function but reduces downstream inflammation",
            "Beremagene geperpavec (B-VEC, topical HSV-1 gene therapy, FDA 2023) for RDEB/COL7A1: delivers COL7A1 directly to wound surface; shown to improve complete wound closure in Phase 3 GEM trial; applied topically to accessible blistered wounds; requires intact skin delivery mechanism; no systemic delivery needed; first approved gene therapy for EB",
            "Wound care hierarchy in RDEB: (1) lance and drain blisters early at small end, leave blister roof as biological cover; (2) non-adherent silicone foam or lipido-colloid dressings; (3) no tape or adhesive directly on skin (Mefix/tape → new blister); (4) bleach baths (sodium hypochlorite 0.005%) for S. aureus colonisation reduction; (5) mupirocin/fusidic acid topically for acute infection only; (6) systemic antibiotics only for cellulitis/bacteraemia — not routine colonisation",
            "Lithium avoidance in Darier disease (ATP2A2): lithium carbonate/citrate dramatically worsens Darier disease — competes with Ca2+ signalling and further impairs SERCA2 function; substitute with valproate (sodium valproate) or lamotrigine for bipolar disorder/mood stabilisation in Darier patients; this is a MANDATORY drug-disease interaction to screen",
            "Retinoid teratogenicity warnings (all genodermatoses requiring retinoids): acitretin — teratogenic for 3 years post-cessation (etretinate metabolite persists); isotretinoin — teratogenic for 1 month post-cessation; both require mandatory contraception in females of childbearing potential; iPLEDGE (US) or REMS programme enrolment required; avoid in first trimester; monitor liver function + lipids on all systemic retinoids",
            "Aciclovir for Kaposi varicelliform eruption (KVE) in Darier: HSV superinfection of Darier plaques → widespread vesiculopustular KVE — SYSTEMIC aciclovir (IV for severe; oral for mild-moderate, 400 mg 5×/day, 5–10 days) must be started immediately; KVE in barrier-deficient genodermatoses can be life-threatening; do NOT use topical aciclovir alone for KVE",
        ],
        "key_standards": [
            "DEBRA International Clinical Management Guidelines for EB — J Invest Dermatol 2014/2020 — EB classification by IF/TEM/genetics; wound care hierarchy; SCC surveillance in RDEB (monthly skin checks from adolescence, biopsy suspicious wounds); nutritional support; beremagene geperpavec gene therapy (FDA 2023) for accessible RDEB wounds",
            "Netherton Syndrome / SPINK5 Management Guidelines — Br J Dermatol 2018 / ESID consensus — trichorrhexis invaginata diagnosis (polarising microscope mandatory); IgE measurement; dupilumab use (FDA 2022 Netherton indication); neonatal erythroderma management; anaphylaxis preparedness (epinephrine autoinjector mandatory); LEKTI IF skin biopsy",
            "Darier Disease Management Guidelines — Br J Dermatol 2016 / EDF — isotretinoin/acitretin dosing; lithium contraindication (switch to valproate/lamotrigine); KVE (systemic aciclovir immediately); trigger avoidance (UV, heat, friction); nail signs diagnosis; Hailey-Hailey differential diagnosis (ATP2C1); psychiatric co-morbidity screening",
            "X-Linked Ichthyosis Consensus — J Eur Acad Dermatol Venereol 2020 — STS deletion testing (MLPA/array-CGH mandatory, not sequencing alone); corneal opacity documentation (ophthalmology referral); cryptorchidism management (orchidopexy by 18 months); contiguous Xp22.31 deletion screening (ANOS1 Kallmann); emollient + keratolytic regimen",
            "Incontinentia Pigmenti Clinical Guidelines — Orphanet J Rare Dis 2014 / IP Foundation — exon 4–10 IKBKG deletion PCR (first-line); RetCam ophthalmology from birth (every 3 months for 3 years, then 6-monthly); dental OPG from age 3; brain MRI + EEG (all IP patients); anti-VEGF/laser for retinal vasculopathy; male lethality pedigree analysis for family counselling",
            "Ichthyosis International Consensus Guidelines (ARCI/Harlequin) — Br J Dermatol 2019 — ABCA12 mutation confirmation; neonatal ICU management (humidified incubator, petrolatum, saline soaks); acitretin 1 mg/kg/day ASAP; ectropion ophthalmology (corneal lubrication; consider tarsorrhaphy); long-term retinoid monitoring (liver, lipids, DEXA for bone density)",
            "EB Consensus — DEBRA EB Registry / International Rare Disease Network — EB clinical phenotyping checklist; genetic confirmation requirements; anaemia management in RDEB (IV iron, erythropoietin); oesophageal dilation protocol; SCC surveillance biopsy protocol; beremagene eligibility criteria (FDA label: accessible chronic wounds, RDEB diagnosis confirmed by gene and IF)",
        ],
    }
