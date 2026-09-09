#!/usr/bin/env python3
"""Hereditary-Ichthyosis-Atlas — Complete 8-Gene Hereditary Ichthyosis Atlas
(TGM1 · ABCA12 · CYP4F22 · NIPAL4 · STS · KRT1 · GJB3 · ALOX12B).

TGM1    (Transglutaminase 1; 817 aa; 14q12; AR;
         ARCI type 1 / Lamellar Ichthyosis LI1;
         Most common ARCI worldwide (30-35%);
         TGase-1 cross-links cornified envelope proteins;
         COLLODION BABY — tight shiny membrane at birth + ectropion + eclabium — PATHOGNOMONIC;
         Plate-like dark-brown lamellar scale trunk/extensors after shedding collodion;
         Erythroderma variable; anhidrosis → heat intolerance;
         seed SEED_BASE+0).
ABCA12  (ATP-Binding Cassette A12; 2595 aa; 2q35; AR;
         Harlequin Ichthyosis (HI) — most severe ARCI;
         Lipid transporter for lamellar granule exocytosis;
         ARMOR-PLATE DIAMOND-SHAPED SCALE PLATES SEPARATING AT BIRTH — PATHOGNOMONIC;
         Ectropion + eclabium + flattened nose/ears = neonatal emergency;
         Acitretin (retinoid) life-saving from birth;
         seed SEED_BASE+1).
CYP4F22 (Cytochrome P450 4F22; 524 aa; 19p13.12; AR;
         ARCI type 6 / LI6;
         Omega-hydroxylase — ultra-long chain fatty acid (ULCFA) barrier synthesis;
         FINE LAMELLAR BROWN SCALE + PALMOPLANTAR KERATODERMA — PATHOGNOMONIC;
         No blistering; minimal erythema (non-erythrodermic LI);
         seed SEED_BASE+2).
NIPAL4  (Ichthyin; 399 aa; 5q33.3; AR;
         ARCI type 4 / LI4;
         Membrane transporter; epidermal lipid metabolism;
         PRURITUS + PLATE-LIKE LAMELLAR SCALE + ANHIDROSIS — PATHOGNOMONIC;
         Heat intolerance; may improve with age;
         seed SEED_BASE+3).
STS     (Steroid Sulfatase; 583 aa; Xp22.31; XLR;
         X-linked Ichthyosis (XLI); males affected; females carriers (may have mild scaling);
         Steroid sulfatase deficiency → cholesterol sulfate accumulation in SC;
         LARGE DARK BROWN POLYGONAL SCALE NECK/EXTENSOR + POSTERIOR CORNEAL OPACITY — PATHOGNOMONIC;
         CRYPTORCHIDISM 20% males;
         No collodion baby — postnatal onset 1-3 months;
         Contiguous gene deletion: Kallmann + XLI (Xp22.3 deletion);
         seed SEED_BASE+4).
KRT1    (Keratin 1; 644 aa; 12q13.13; AD;
         Epidermolytic Ichthyosis (EI) / Bullous CRIE;
         Pairs with KRT10 to form suprabasal keratin IF;
         EPIDERMOLYTIC HYPERKERATOSIS ON BIOPSY — PATHOGNOMONIC (suprabasal vacuolation + granular epidermolysis + compact hyperkeratosis);
         Blistering at birth → dark verrucous scale by childhood;
         SECONDARY SUPERINFECTION (S. aureus, malodor) = MOST COMMON COMPLICATION;
         seed SEED_BASE+5).
GJB3    (Connexin 31; 270 aa; 1p34.3; AD;
         Erythrokeratoderma Variabilis (EKV) type 1;
         Gap junction defect — connexin-31 epidermal communication;
         TRANSIENT MIGRATORY FIGURATE ERYTHEMATOUS PATCHES + FIXED HYPERKERATOTIC PLAQUES — PATHOGNOMONIC;
         Erythema fluctuates with emotional stress/temperature;
         Palmoplantar keratoderma; limited to skin — no systemic involvement;
         seed SEED_BASE+6).
ALOX12B (12R-Lipoxygenase; 701 aa; 17p13.1; AR;
         ARCI type 8 / LI8;
         12R-LOX epidermal ceramide/sphingolipid barrier synthesis;
         BROWN PLATE-LIKE SCALE + PALMAR FISSURING — PATHOGNOMONIC;
         No blistering; heat intolerance; anhidrosis;
         Indistinguishable from TGM1-LI clinically — gene panel mandatory;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2278-2285).
"""

import random

SEED_BASE = 2278

ICHTHYOSIS_GENES = [
    # -- TGM1 — ARCI type 1 / Lamellar Ichthyosis (most common ARCI) ----------------------
    {
        "gene": "TGM1",
        "alt_name": (
            "TGM1 (TGM1-817aa-14q12 / AR — ARCI1-Lamellar-Ichthyosis-LI1 — "
            "MOST-COMMON-ARCI-WORLDWIDE-30-35pct — "
            "COLLODION-BABY-ECTROPION-ECLABIUM-SHINY-MEMBRANE-BIRTH-PATHOGNOMONIC — "
            "PLATE-LIKE-DARK-BROWN-LAMELLAR-SCALE-TRUNK-EXTENSORS — "
            "ANHIDROSIS-HEAT-INTOLERANCE-MANDATORY-COOL-ENVIRONMENT — "
            "p.Arg143His-Most-Common-European — TGase-1-Cornified-Envelope-Cross-Linking)"
        ),
        "protein": (
            "TGM1 -- 14q12 AR -- TGM1-817aa -- "
            "Transglutaminase-1-TGase-1-90kDa-Membrane-Bound-Calcium-Dependent -- "
            "ARCI1-OMIM-242300 -- "
            "CROSS-LINKS-CORNIFIED-ENVELOPE-PROTEINS-Loricrin-Involucrin-SPRRs-Terminal-Differentiation -- "
            "TGM1-LOF-Defective-CE-Assembly-Deficient-Barrier-Function -- "
            "COLLODION-BABY-Ectropion-Eclabium-Tight-Shiny-Collodion-Membrane-Birth-PATHOGNOMONIC -- "
            "PLATE-LIKE-DARK-BROWN-LAMELLAR-SCALE-Trunk-Extensors-After-Shedding -- "
            "ERYTHRODERMA-Variable-Neonatal-Resolves-Scale-Dominant-Feature-Adulthood -- "
            "ANHIDROSIS-Eccrine-Duct-Occlusion-by-Scale-HEAT-INTOLERANCE-Mandatory-Cool -- "
            "p.Arg143His-p.Arg307X-Most-Common-European-Founder-Alleles -- "
            "ENZYME-ACTIVITY-ASSAY-TGase-1-activity-reduced-zero-diagnostic-tool -- "
            "RETINOID-ACITRETIN-Most-Effective-Systemic-Therapy-Reduces-Scale-Ectropion -- "
            "OMIM-Gene-TGM1-190195-Disease-ARCI1-242300"
        ),
        "locus": "14q12",
        "protein_size": "817 aa / 90 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF mutations (nonsense/missense/splice); "
            "p.Arg143His: most common European founder allele (ARCI1); "
            "p.Arg307X: second common European allele; "
            "Compound heterozygotes frequent; "
            "Genotype-phenotype: null/null → collodion baby + classic LI; "
            "Hypomorphic/missense → milder; some activity retained; "
            "Variable expressivity within families; "
            "Erythroderma: ~20% persistent erythrodermic LI (Erythrodermic form); "
            "Collodion baby: pathognomonic — 85% of TGM1 present as collodion baby; "
            "Anhidrosis universal → heat stroke risk; "
            "Enzyme assay: TGase-1 activity in epidermis/granulocytes — if absent/severely reduced, confirms TGM1 functionally; "
            "Most common ARCI gene (30-35% of ARCI)"
        ),
        "key_features": [
            "COLLODION BABY — tight shiny membrane at birth with ectropion + eclabium + restricted facial movement — PATHOGNOMONIC for ARCI; 85% of TGM1 present this way",
            "PLATE-LIKE DARK-BROWN LAMELLAR SCALE — trunk, extensors, scalp; emerges after collodion shedding (neonatal); persists lifelong; worsens with dryness/cold",
            "ANHIDROSIS — eccrine ducts blocked by scale; HEAT INTOLERANCE mandatory history; heat stroke risk; cool environment + wet clothing thermoregulation strategy",
            "ERYTHRODERMA — variable (20% persistent); neonatal period always erythrodermic under collodion; in adults: scale dominant, erythema subsides in most",
            "TGase-1 ENZYME ACTIVITY ASSAY — gold-standard functional test; activity absent/severely reduced in TGM1-ARCI1; available at specialist centres (granulocytes/keratinocytes)",
            "p.Arg143His — most common European founder allele; functional null; classic LI phenotype",
            "RETINOID (acitretin) — most effective systemic therapy; reduces scale burden; improves ectropion; long-term maintenance; teratogenic (females: contraception mandatory)",
            "NEONATAL ICU — collodion baby requires temperature regulation, ophthalmology (ectropion, corneal exposure), feeding support, high TEWL fluid management",
        ],
        "treatment": (
            "Neonatal (collodion baby): "
            "High-humidity incubator (70-80% RH) — prevents cracking/fissuring of collodion membrane; "
            "Emollient application every 2-4h (petroleum jelly, 50:50 white soft paraffin/liquid paraffin); "
            "IV fluid support — transepidermal water loss (TEWL) greatly elevated; "
            "Ophthalmology: lubricant eye drops hourly; eye protective chamber if ectropion severe; "
            "Nasogastric feeding if eclabium impairs suckling; "
            "Temperature regulation — servo-controlled incubator; "
            "Infection: topical mupirocin for fissure-associated infection; systemic antibiotics for sepsis. "
            "Long-term (child/adult): "
            "Emollients (first-line): urea 10-40% creams (keratolytic); lactic acid 5-12% lotions; "
            "Retinoids (most effective systemic): acitretin 0.3-0.5 mg/kg/day — reduces scale significantly; "
            "Alitretinoin (9-cis-retinoic acid) alternative; "
            "Isotretinoin — used off-label; "
            "Thermoregulation: cool environment; wet clothing; cooling vests; avoid exercise in heat; "
            "Ophthalmology: lubricant drops; surgical ectropion correction if vision threatened; "
            "Scalp: descaling shampoos (salicylic acid); mechanical scale removal after bath. "
            "Genetics: AR — 25% recurrence per pregnancy; prenatal diagnosis (CVS/amnio) available."
        ),
        "monitoring": [
            "Annual dermatology: scale burden assessment; ectropion; erythroderma; infection signs (fissures/cellulitis)",
            "Ophthalmology: 6-monthly (ectropion → corneal exposure → keratopathy risk); lubricant compliance check",
            "Thermoregulation counselling: heat avoidance; school/work environment plan; cooling strategies documented",
            "Retinoid monitoring: LFTs + lipids (triglycerides) before + 1-2 monthly during therapy; DEXA scan if long-term (vertebral changes); teratogenicity counselling",
            "TGase-1 enzyme activity: confirm at diagnosis; useful for variant interpretation",
            "QoL: DLQI (Dermatology Life Quality Index); itch/pain VAS; psychosocial support",
            "Family: first-degree relative screening if biallelic confirmed; prenatal/PGD offered",
            "Neonatal: weekly weight/growth; TEWL monitoring; electrolytes",
        ],
        "ichthyosis_type": "ARCI1 — Lamellar Ichthyosis type 1",
        "scale_morphology": "Plate-like dark-brown lamellar scale, trunk/extensors",
        "pathognomonic": "Collodion baby (ectropion + eclabium + tight shiny membrane) + plate-like lamellar scale = TGM1-ARCI",
        "treatment_highlight": "Acitretin most effective systemic; collodion baby = neonatal ICU; TGase-1 enzyme assay confirms",
        "avg_age_at_dx_yrs": 0.0,
    },
    # -- ABCA12 — Harlequin Ichthyosis (most severe ARCI) ----------------------------------
    {
        "gene": "ABCA12",
        "alt_name": (
            "ABCA12 (ABCA12-2595aa-2q35 / AR — Harlequin-Ichthyosis-HI-MOST-SEVERE-ARCI — "
            "ARMOR-PLATE-DIAMOND-SHAPED-SCALE-PLATES-BIRTH-PATHOGNOMONIC — "
            "ECTROPION-ECLABIUM-FLATTENED-NOSE-EARS-NEONATAL-EMERGENCY — "
            "ACITRETIN-RETINOID-LIFE-SAVING-FROM-BIRTH — "
            "ABCA12-Lipid-Transporter-Lamellar-Granule-Exocytosis)"
        ),
        "protein": (
            "ABCA12 -- 2q35 AR -- ABCA12-2595aa -- "
            "ATP-Binding-Cassette-Subfamily-A-Member-12-260kDa-Lamellar-Granule-Lipid-Transporter -- "
            "HI-OMIM-242500 -- "
            "TRANSPORTS-GLUCOCEREBROSIDES-INTO-LAMELLAR-GRANULES-Keratinocyte-Lipid-Secretion -- "
            "ABCA12-LOF-Absent-Lamellar-Granules-EM-PATHOGNOMONIC-NO-Lipid-Bilayers-in-SC -- "
            "ARMOR-PLATE-SCALE-DIAMOND-SHAPED-FISSURED-HYPERKERATOTIC-PLATES-AT-BIRTH-PATHOGNOMONIC -- "
            "ECTROPION-SEVERE-Corneal-Exposure-ECLABIUM-Feeding-Impossibility-FLATTENED-NOSE-EARS -- "
            "AIRWAY-HYPERKERATOSIS-Respiratory-Compromise-NICU-Mandatory -- "
            "ACITRETIN-RETINOID-Life-Saving-Start-Within-24h-Birth -- "
            "SURVIVAL-IMPROVED-Modern-NICU-40pct-Pre-NICU-Era-Now-80pct-5yr -- "
            "OMIM-Gene-ABCA12-607800-Disease-HI-242500"
        ),
        "locus": "2q35",
        "protein_size": "2595 aa / 260 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF mutations (large deletions, nonsense, frameshift); "
            "No founder mutation — distributed worldwide; "
            "Prenatal: fetal skin biopsy (absent lamellar granules on EM) historically; now NGS; "
            "HI: always severe (homozygous/compound heterozygous null); no mild HI alleles; "
            "Partial ABCA12 function (missense/hypomorph) → lamellar ichthyosis (not HI); "
            "Lamellar granules: absent on EM in HI — ultrastructural hallmark; "
            "ABCA12 also transports glucocerebrosides → ceramide pathway; "
            "Lipid extrusion absent → stratum corneum compacted protein-dominant; "
            "EM: keratosome/lamellar granule absent in upper spinous/granular layers (diagnostic)"
        ),
        "key_features": [
            "ARMOR-PLATE DIAMOND-SHAPED SCALE PLATES SEPARATING AT BIRTH — PATHOGNOMONIC for Harlequin Ichthyosis; plates have deep red fissures between",
            "SEVERE ECTROPION (eyelids everted) + ECLABIUM (lips everted) + FLATTENED NOSE/EARS (hypoplastic auricles) — neonatal emergency at delivery",
            "AIRWAY HYPERKERATOSIS — stridor + respiratory compromise; intubation may be needed; ICU mandatory",
            "ABSENT LAMELLAR GRANULES ON ELECTRON MICROSCOPY — ultrastructural pathognomonic finding",
            "ACITRETIN (oral retinoid) LIFE-SAVING — start within 24 hours of birth (0.5 mg/kg/day); accelerates collodion/armor-plate shedding; improves survival dramatically",
            "SURVIVAL: historically lethal (~40% survival pre-NICU era); modern NICU + acitretin → 80% 5-year survival; long-term: classic LI-like phenotype with scale burden",
            "FEEDING: eclabium + restricted jaw movement → nasogastric/gastrostomy feeding; speech therapy + orthodontic follow-up",
            "NEONATAL EMERGENCY PROTOCOL: deliver at tertiary center with pediatric dermatology + NICU + ophthalmology + ENT pre-briefed",
        ],
        "treatment": (
            "Neonatal EMERGENCY (first 48-72 hours): "
            "NICU admission mandatory — temperature regulation + respiratory monitoring; "
            "High-humidity incubator (≥80% RH); "
            "Acitretin 0.5 mg/kg/day PO or via NG — start WITHIN 24h of birth (most critical intervention); "
            "Ophthalmology URGENT: corneal exposure → lubricant drops every 30 min; moisture chambers; lateral tarsorrhaphy if corneal abrasion; "
            "ENT: stridor assessment; flexible laryngoscopy; intubation if airway compromise; "
            "IV fluids: TEWL massively elevated; maintenance + replacement; "
            "Emollient immersion bath twice daily (coconut oil/petroleum); "
            "Feeding: NG tube if eclabium prevents feeding; dietitian from day 1. "
            "Long-term: "
            "Acitretin maintenance 0.3-0.5 mg/kg/day — lifelong or near-lifelong; reduces scale burden; "
            "Emollients: high-lipid keratolytic (urea 20-40%; lactic acid 10%); "
            "Ophthalmology: 3-monthly; artificial tears + gel; "
            "Physiotherapy: joint contracture prevention; "
            "Thermoregulation: cool environment; cooling strategies; "
            "Dental: orthodontic assessment (eclabium-related malocclusion); "
            "Psychosocial: multidisciplinary support; parent/patient networks (FIRST / Debra)."
        ),
        "monitoring": [
            "NICU: daily weight/TEWL/electrolytes; airway; ophthalmology daily (first week); infection surveillance",
            "Acitretin: LFTs + triglycerides monthly (first 3 months), then 3-monthly; teratogenicity (females: contraception 3yr post-cessation)",
            "Ophthalmology: every 3 months — corneal exposure; ectropion; vision; lubricant efficacy",
            "ENT: 6-monthly — ear canal scaling; otitis externa; hearing screen; laryngeal assessment if stridor",
            "Rheumatology/physio: joint mobility — finger + wrist contractures; annual goniometry",
            "QoL: DLQI; CDLQI (child); anxiety/depression screen; school integration plan",
            "Dermatology: monthly (first year), then 3-6 monthly — scale pattern; fissure infection; SCC surveillance (from teenage years)",
            "Genetics: family cascade; prenatal/PGD counselling offered",
        ],
        "ichthyosis_type": "Harlequin Ichthyosis (HI) — most severe ARCI",
        "scale_morphology": "Armor-plate diamond-shaped hyperkeratotic plates with deep red fissures at birth",
        "pathognomonic": "Armor-plate scale + ectropion + eclabium at birth + absent lamellar granules on EM = ABCA12 Harlequin Ichthyosis",
        "treatment_highlight": "Acitretin within 24h of birth = life-saving; NICU + ophthalmology + ENT mandatory; lamellar granules absent on EM",
        "avg_age_at_dx_yrs": 0.0,
    },
    # -- CYP4F22 — ARCI type 6 / Lamellar Ichthyosis 6 -------------------------------------
    {
        "gene": "CYP4F22",
        "alt_name": (
            "CYP4F22 (CYP4F22-524aa-19p13.12 / AR — ARCI6-Lamellar-Ichthyosis-LI6 — "
            "ULTRA-LONG-CHAIN-FATTY-ACID-OMEGA-HYDROXYLASE-BARRIER-DEFECT — "
            "FINE-BROWN-LAMELLAR-SCALE-PALMOPLANTAR-KERATODERMA-PATHOGNOMONIC — "
            "NON-ERYTHRODERMIC-LI-Minimal-Erythema-KEY-DDx — "
            "p.Gly230Asp-Most-Common-CYP4F22)"
        ),
        "protein": (
            "CYP4F22 -- 19p13.12 AR -- CYP4F22-524aa -- "
            "Cytochrome-P450-4F22-60kDa-Microsomal-Omega-Hydroxylase -- "
            "ARCI6-OMIM-612300 -- "
            "OMEGA-HYDROXYLATES-ULTRA-LONG-CHAIN-FATTY-ACIDS-ULCFA-Ceramide-Precursor -- "
            "CYP4F22-LOF-Deficient-ULCFA-Hydroxylation-Impaired-Corneocyte-Lipid-Envelope -- "
            "FINE-LAMELLAR-BROWN-SCALE-Non-Adherent-Trunk-Limbs-Scalp -- "
            "PALMOPLANTAR-KERATODERMA-Diffuse-Thickened-Fissured-Prominent -- "
            "NON-ERYTHRODERMIC-Minimal-Erythema-Distinguishes-From-TGM1-Erythrodermic -- "
            "NO-COLLODION-BABY-or-Mild-Collodion -- "
            "p.Gly230Asp-Most-Common-CYP4F22-European-Founder -- "
            "EMOLLIENT-PLUS-RETINOID-Standard-Therapy -- "
            "OMIM-Gene-CYP4F22-611609-Disease-ARCI6-612300"
        ),
        "locus": "19p13.12",
        "protein_size": "524 aa / 60 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic missense/nonsense/splice; "
            "p.Gly230Asp: most common CYP4F22 allele (European); "
            "No collodion baby (or mild transient collodion — 30%); "
            "Non-erythrodermic LI: scale dominant, erythema minimal/absent — key DDx from TGM1 erythrodermic form; "
            "Palmoplantar keratoderma (PPK) prominent — more pronounced than TGM1-LI; "
            "Anhidrosis: common; heat intolerance; "
            "Clinically overlaps with TGM1-LI, NIPAL4-LI, ALOX12B-LI — gene panel mandatory; "
            "Rare worldwide; "
            "CYP4F22 specifically hydroxylates fatty acids to form esterified ceramides in corneocyte envelope"
        ),
        "key_features": [
            "FINE BROWN LAMELLAR SCALE — non-erythrodermic; scale on trunk, limbs, scalp; fine plate-like to powdery morphology",
            "PALMOPLANTAR KERATODERMA — diffuse thickened fissured PPK; more prominent than TGM1-LI; walking difficulty",
            "NON-ERYTHRODERMIC — minimal or absent erythema; key DDx from erythrodermic ARCI types (NPS1, ALOX3); "
            "NO (OR MILD) COLLODION BABY — ~70% CYP4F22 lack collodion; mild if present",
            "ANHIDROSIS — heat intolerance; eccrine duct occlusion by scale; cooling strategies mandatory",
            "p.Gly230Asp — most common CYP4F22 European allele; functional null; classic non-erythrodermic LI",
            "GENE PANEL MANDATORY — clinically indistinguishable from TGM1-LI, NIPAL4-LI, ALOX12B-LI; TGase-1 enzyme assay (negative) helps exclude TGM1",
            "RETINOID (acitretin) + emollients — effective; PPK responds to urea 40% + salicylic acid footcare",
        ],
        "treatment": (
            "Emollients (first-line): "
            "Urea 20-40% cream (keratolytic for scale body + feet); "
            "Lactic acid 10-12% lotion; "
            "Salicylic acid 5-10% ointment (scalp + PPK); "
            "Petroleum jelly base for dry fissures. "
            "Palmoplantar keratoderma: "
            "Urea 40-50% cream feet daily + occlusion overnight; "
            "Salicylic acid 15-20% paste; "
            "Mechanical debridement after soaking bath; "
            "Orthopaedic footwear for callosities. "
            "Systemic: "
            "Acitretin 0.3-0.5 mg/kg/day — reduces scale and PPK; effective for CYP4F22-LI; "
            "Monitor LFTs + lipids; teratogenic (contraception mandatory females). "
            "Thermoregulation: "
            "Cool environment; cooling vest; water mist spray; "
            "Avoid strenuous outdoor exercise in heat. "
            "Scalp: "
            "Descaling shampoos (salicylic acid, zinc pyrithione); "
            "Topical steroid short-term for scalp pruritus. "
            "Genetics: AR — 25% recurrence; prenatal/PGD available."
        ),
        "monitoring": [
            "Annual dermatology: scale burden; PPK severity; fissure infection; ectropion screen",
            "Thermoregulation counselling: heat events documented; cooling strategy review annually",
            "Retinoid: LFTs + triglycerides 1-2 monthly (first 3 months), then 3-monthly; DEXA if long-term",
            "Ophthalmology: annual — ectropion rare but PPK fissures near eyelids possible",
            "QoL: DLQI; foot pain VAS; school/work accommodation",
            "TGase-1 enzyme assay: if not done at diagnosis — helps exclude TGM1",
            "Gene panel: confirm biallelic CYP4F22 pathogenic variants",
            "Family: cascade; prenatal/PGD offered",
        ],
        "ichthyosis_type": "ARCI6 — Lamellar Ichthyosis type 6",
        "scale_morphology": "Fine brown lamellar scale; palmoplantar keratoderma; non-erythrodermic",
        "pathognomonic": "Fine non-erythrodermic lamellar scale + prominent PPK + no/mild collodion = CYP4F22-LI (gene panel confirms)",
        "treatment_highlight": "Acitretin + urea 40% PPK care; non-erythrodermic; gene panel mandatory to distinguish from TGM1/NIPAL4/ALOX12B",
        "avg_age_at_dx_yrs": 0.1,
    },
    # -- NIPAL4 — ARCI type 4 / Lamellar Ichthyosis 4 (Ichthyin) --------------------------
    {
        "gene": "NIPAL4",
        "alt_name": (
            "NIPAL4 (NIPAL4-399aa-5q33.3 / AR — ARCI4-Lamellar-Ichthyosis-LI4-Ichthyin — "
            "PRURITUS-PROMINENT-PLATE-LIKE-LAMELLAR-SCALE-PATHOGNOMONIC — "
            "ANHIDROSIS-HEAT-INTOLERANCE-ALL-PATIENTS — "
            "COLLODION-BABY-Variable-Some-Patients — "
            "Ichthyin-Membrane-Transporter-Epidermal-Lipid-Metabolism)"
        ),
        "protein": (
            "NIPAL4 -- 5q33.3 AR -- NIPAL4-399aa -- "
            "Non-Imprinted-Prader-Willi-Angelman-Locus-4-Ichthyin-44kDa-Membrane-Transporter -- "
            "ARCI4-OMIM-615024 -- "
            "NIPA-Domain-Transports-Fatty-Acids-and-Sphingolipids-Epidermal-Lipid-Barrier -- "
            "NIPAL4-LOF-Impaired-Lipid-Extrusion-Defective-Lamellar-Granule-Function -- "
            "PRURITUS-SIGNIFICANT-All-Patients-KEY-Clinical-Feature-vs-TGM1 -- "
            "PLATE-LIKE-LAMELLAR-SCALE-Trunk-Extremities-Moderate-Severity -- "
            "ANHIDROSIS-ALL-PATIENTS-Heat-Intolerance-Cooling-Mandatory -- "
            "COLLODION-BABY-Variable-30-50pct -- "
            "MAY-IMPROVE-With-Age-in-Some -- "
            "p.Glu273Lys-Most-Common-European-NIPAL4 -- "
            "OMIM-Gene-NIPAL4-609383-Disease-ARCI4-615024"
        ),
        "locus": "5q33.3",
        "protein_size": "399 aa / 44 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic missense/nonsense; "
            "p.Glu273Lys: most common NIPAL4 European allele; "
            "Collodion baby: ~30-50% (variable penetrance for neonatal presentation); "
            "PRURITUS: significant feature in most patients — distinguishes NIPAL4-LI from TGM1-LI (less itch) and HI (no itch focus); "
            "Anhidrosis: universal; eccrine duct occlusion; heat intolerance; "
            "May improve in some patients after adolescence; "
            "Clinically overlaps other ARCI types — gene panel mandatory; "
            "NIPAL4 is a magnesium transporter homologue expressed in upper epidermis; "
            "Ichthyin = historical synonym (based on Greek 'ichthys' = fish)"
        ),
        "key_features": [
            "PRURITUS — significant in most NIPAL4-LI patients; distinguishes from TGM1-LI (less pruritic) and CYP4F22 (minimal pruritus); histamine-independent itch mechanism",
            "PLATE-LIKE LAMELLAR SCALE — moderate severity; trunk/extremities/scalp; plate morphology; dark brown colour",
            "ANHIDROSIS — universal; heat intolerance; heat stroke risk without thermoregulation plan",
            "COLLODION BABY — variable (30-50%); may be milder than TGM1-ARCI1",
            "MAY IMPROVE WITH AGE — some patients note reduced scale in adulthood; not universal",
            "p.Glu273Lys — most common European NIPAL4 allele; confirms diagnosis with biallelic pathogenic variants",
            "GENE PANEL MANDATORY — clinically indistinguishable from TGM1/CYP4F22/ALOX12B; TGase-1 assay (negative for NIPAL4 — TGM1 excluded)",
            "RETINOID + EMOLLIENT + ANTIPRURITIC — triple therapy; antihistamines limited efficacy for ichthyosis itch; dupilumab under investigation",
        ],
        "treatment": (
            "Pruritus management (NIPAL4 priority): "
            "Emollients containing ceramides + lipid replacement — first-line; reduce barrier-derived itch signal; "
            "Antihistamines (non-sedating): cetirizine/loratadine — limited efficacy for ichthyosis itch (non-histaminergic); "
            "Sedating antihistamines (hydroxyzine) — nocturnal itch management; "
            "Dupilumab (IL-4/IL-13 blockade) — growing evidence for pruritus in ichthyosis; off-label (2026); "
            "Cooling topicals: menthol 0.5-2% in emollient base; "
            "Short wet wraps: emollient under damp tube bandage overnight — reduces itch + scale. "
            "Scale management: "
            "Emollients: urea 20-40% cream; lactic acid 10%; daily bath + immediate emollient application; "
            "Acitretin 0.3-0.5 mg/kg/day — effective for scale + itch in NIPAL4-LI; "
            "Scalp: salicylic acid shampoo. "
            "Thermoregulation: cool environment; cooling vest; water spray; "
            "Avoid strenuous activity in heat. "
            "Genetics: AR — 25%; prenatal/PGD available."
        ),
        "monitoring": [
            "Dermatology: 3-6 monthly — scale score; itch VAS (NRS); fissure infection; retinoid side effects",
            "Thermoregulation plan: annual review; heat events documented; cooling strategy updated",
            "Pruritus: NRS 0-10 monthly diary; dupilumab response tracking if started",
            "Retinoid: LFTs + lipids 1-2 monthly (first 3 months), then 3-monthly",
            "QoL: DLQI; itch-specific QoL (ItchyQoL); sleep disruption assessment",
            "TGase-1 enzyme assay: exclude TGM1 if not done at diagnosis",
            "Gene panel: confirm biallelic NIPAL4 variants",
            "Ophthalmology: annual — ectropion screen; lubricant use",
        ],
        "ichthyosis_type": "ARCI4 — Lamellar Ichthyosis type 4 (Ichthyin)",
        "scale_morphology": "Plate-like lamellar scale; moderate severity; significant pruritus",
        "pathognomonic": "Plate-like lamellar scale + PROMINENT PRURITUS + anhidrosis = NIPAL4-LI (distinguishes from TGM1-LI)",
        "treatment_highlight": "Pruritus management priority (dupilumab under investigation); acitretin + emollients; TGase-1 assay excludes TGM1",
        "avg_age_at_dx_yrs": 0.1,
    },
    # -- STS — X-linked Ichthyosis (XLI) ---------------------------------------------------
    {
        "gene": "STS",
        "alt_name": (
            "STS (STS-583aa-Xp22.31 / XLR — X-Linked-Ichthyosis-XLI — "
            "LARGE-DARK-BROWN-POLYGONAL-SCALE-NECK-EXTENSOR-PATHOGNOMONIC — "
            "POSTERIOR-CORNEAL-OPACITY-ASYMPTOMATIC-PATHOGNOMONIC-Slit-Lamp-Mandatory — "
            "CRYPTORCHIDISM-20pct-Males-Orchidopexy-Mandatory-Malignancy-Risk — "
            "NO-COLLODION-BABY-Postnatal-Onset-1-3-Months — "
            "CHOLESTEROL-SULFATE-ACCUMULATION-Steroid-Sulfatase-Deficiency)"
        ),
        "protein": (
            "STS -- Xp22.31 XLR -- STS-583aa -- "
            "Steroid-Sulfatase-STS-62kDa-Microsomal-Membrane-Bound-Lysosomal -- "
            "XLI-OMIM-308100 -- "
            "HYDROLYSES-CHOLESTEROL-SULFATE-To-Cholesterol-In-Stratum-Corneum -- "
            "STS-LOF-Cholesterol-Sulfate-Accumulates-SC-Prevents-Desquamation-Corneocyte-Retention -- "
            "LARGE-DARK-BROWN-POLYGONAL-SCALE-Neck-Trunk-Extensors-PATHOGNOMONIC -- "
            "POSTERIOR-CORNEAL-OPACITY-Comma-Shaped-Asymptomatic-Slit-Lamp-PATHOGNOMONIC -- "
            "CRYPTORCHIDISM-20pct-Bilateral-or-Unilateral-Orchidopexy-Malignancy-Surveillance -- "
            "NO-COLLODION-BABY-Onset-1-3-Months-After-Birth -- "
            "Xp22.3-CONTIGUOUS-DELETION-Kallmann-KAL1-PLUS-XLI -- "
            "STS-SULFATASE-ACTIVITY-Fibroblast-Leukocyte-Diagnostic-Gold-Standard -- "
            "OMIM-Gene-STS-300747-Disease-XLI-308100"
        ),
        "locus": "Xp22.31",
        "protein_size": "583 aa / 62 kDa",
        "inheritance": (
            "XLR (X-linked recessive); males affected; female carriers usually asymptomatic (may have mild scaling, corneal opacities); "
            "Point mutations (~20%) or large deletions/duplications (~80%); "
            "Large Xp22.3 contiguous gene deletion: STS + KAL1 → XLI + Kallmann syndrome (hypogonadism + anosmia); "
            "Female carriers: steroid sulfatase activity ~50% — most asymptomatic; "
            "Prenatal: postnatal onset; cholesterol sulfate elevated in maternal urine in STS-affected pregnancy; "
            "Cryptorchidism: 20% males — bilateral or unilateral; early orchidopexy mandatory (malignancy risk 3-5× if uncorrected); "
            "Carrier females: Corneal opacities in 25% carrier females (slit-lamp); "
            "Steroid sulfatase activity assay: gold-standard functional test (fibroblasts/leukocytes)"
        ),
        "key_features": [
            "LARGE DARK BROWN POLYGONAL SCALE — neck, trunk, extensors; adherent; 'dirty skin' appearance; axillae + antecubital fossae involved; face and palms/soles SPARED",
            "POSTERIOR CORNEAL OPACITY — comma-shaped or punctate opacities in Descemet's membrane/deep stroma; ASYMPTOMATIC (no visual impact) but PATHOGNOMONIC; slit-lamp mandatory at diagnosis",
            "CRYPTORCHIDISM — 20% males; bilateral or unilateral; orchidopexy mandatory (malignancy risk 3-5× if uncorrected by age 2yr); testes examination at diagnosis",
            "NO COLLODION BABY — postnatal onset 1-3 months after birth; normal neonatal skin; important DDx from ARCI (TGM1/ABCA12)",
            "PALMS AND SOLES SPARED — distinguishes XLI from EI (KRT1) and ARCI types with PPK",
            "KALLMANN SYNDROME — XLI + anosmia + hypogonadism if contiguous Xp22.3 deletion (STS + KAL1); olfaction testing + LH/FSH at diagnosis",
            "STEROID SULFATASE ACTIVITY ASSAY — absent/severely reduced in fibroblasts or leukocytes; gold-standard functional test; distinguishes XLI from non-XLI ichthyosis",
            "PLACENTAL INSUFFICIENCY — affected male fetuses: low maternal urine estriol (sulfatase deficiency → impaired placental estrogen synthesis); prolonged labor historically",
        ],
        "treatment": (
            "Emollients (first-line): "
            "Cholesterol-containing creams — theoretical benefit (replace SC cholesterol); "
            "Urea 10-25% cream (keratolytic); lactic acid 5-12%; "
            "Salicylic acid 5-10% for thick neck/extensor scale; "
            "Daily bath + immediate emollient (soak-and-smear technique). "
            "Keratolytics: "
            "Salicylic acid lotion/shampoo for scalp; "
            "Urea 40% for localized thick areas. "
            "Systemic: "
            "Retinoids (acitretin) — effective but rarely needed for mild-moderate XLI; used for severe cases; "
            "Long-term: most managed with emollients alone. "
            "Cryptorchidism: "
            "Orchidopexy — recommended by age 1-2yr; reduces malignancy risk; "
            "Annual testicular self-examination from puberty. "
            "Ophthalmology: "
            "Slit-lamp at diagnosis; annual review (asymptomatic corneal opacities — no treatment); "
            "Lubricant drops if any corneal dryness. "
            "Contiguous deletion (Kallmann): "
            "Endocrinology: LH/FSH/testosterone; GnRH therapy if hypogonadism; "
            "ENT: olfactory testing; "
            "Fertility counselling."
        ),
        "monitoring": [
            "Annual dermatology: scale distribution; neck/extensor burden; emollient compliance; infection screen",
            "Ophthalmology: slit-lamp baseline + annually — posterior corneal opacity; asymptomatic but document",
            "Urology: cryptorchidism — confirm orchidopexy done; post-orchidopexy testicular USS annually (malignancy); self-exam from puberty",
            "Endocrinology: if contiguous Xp22.3 deletion — annual LH/FSH/testosterone; anosmia (Kallmann); growth",
            "STS enzyme assay: confirm at diagnosis (fibroblast/leukocyte); useful for variant interpretation",
            "ARRAY CGH: Xp22.3 deletion size — determine if KAL1 involved (Kallmann);",
            "QoL: DLQI; psychosocial (scale on neck/visible areas); ADHD/learning (STS role in neurosteroid metabolism — reported association)",
            "Female carriers: slit-lamp (25% have corneal opacities); mild scaling check",
        ],
        "ichthyosis_type": "X-linked Ichthyosis (XLI)",
        "scale_morphology": "Large dark brown polygonal scale, neck/extensors; palms/soles spared",
        "pathognomonic": "Dark polygonal neck/extensor scale + POSTERIOR CORNEAL OPACITY (slit-lamp) + cryptorchidism (males) + no collodion baby = XLI/STS",
        "treatment_highlight": "Emollients + slit-lamp for corneal opacities; orchidopexy mandatory 20% males; STS enzyme assay confirms; Kallmann overlap if Xp22.3 deletion",
        "avg_age_at_dx_yrs": 0.2,
    },
    # -- KRT1 — Epidermolytic Ichthyosis (EI) / Bullous CRIE --------------------------------
    {
        "gene": "KRT1",
        "alt_name": (
            "KRT1 (KRT1-644aa-12q13.13 / AD — Epidermolytic-Ichthyosis-EI-Bullous-CRIE — "
            "EPIDERMOLYTIC-HYPERKERATOSIS-EHK-ON-BIOPSY-PATHOGNOMONIC — "
            "BLISTERING-AT-BIRTH-VERRUCOUS-SCALE-CHILDHOOD — "
            "SECONDARY-SUPERINFECTION-S-Aureus-Malodor-MOST-COMMON-COMPLICATION — "
            "p.Asn171Ser-Most-Common-KRT1-Mutation — "
            "KRT1-KRT10-Suprabasal-Keratin-IF-Pair)"
        ),
        "protein": (
            "KRT1 -- 12q13.13 AD -- KRT1-644aa -- "
            "Keratin-1-67kDa-Type-II-Intermediate-Filament-Suprabasal-Keratinocyte -- "
            "EI-OMIM-113800 -- "
            "PAIRS-WITH-KRT10-Suprabasal-Spinous-Granular-Layers -- "
            "KRT1-MUTATION-Dominant-Negative-Collapse-Suprabasal-IF-Cytolysis -- "
            "EPIDERMOLYTIC-HYPERKERATOSIS-EHK-ON-BIOPSY-Suprabasal-Vacuolation-Granular-Epidermolysis-Compact-HK-PATHOGNOMONIC -- "
            "BLISTERING-AT-BIRTH-Resolves-Years-1-2-Then-VERRUCOUS-SCALE-Dominant -- "
            "SECONDARY-SUPERINFECTION-S-Aureus-Malodor-MOST-COMMON-COMPLICATION -- "
            "PALMOPLANTAR-KERATODERMA-KRT1-Prominent-More-Than-KRT10 -- "
            "p.Asn171Ser-Most-Common-KRT1-Mutation-Helix-Initiation-Motif -- "
            "OMIM-Gene-KRT1-139350-Disease-EI-113800"
        ),
        "locus": "12q13.13",
        "protein_size": "644 aa / 67 kDa",
        "inheritance": (
            "AD (autosomal dominant; dominant-negative mechanism); "
            "De novo mutations in 30-50% of cases; "
            "Pairs with KRT10 (Type I); mutations cluster in helix initiation/termination motifs (most severe) or coil regions; "
            "p.Asn171Ser: most common KRT1 mutation (helix initiation motif 1A); classic EI; blistering + PPK; "
            "p.Val166Met: coil 1B; moderate; "
            "KRT1 vs KRT10: KRT1 mutations more likely to cause PPK (KRT1 expressed in palmoplantar suprabasal layers — unlike KRT10); "
            "Genotype-phenotype: helix initiation/termination mutations → more severe EHK + more blistering; "
            "Prenatal skin biopsy (historical): EHK on EM; now: NGS prenatal"
        ),
        "key_features": [
            "EPIDERMOLYTIC HYPERKERATOSIS (EHK) ON BIOPSY — PATHOGNOMONIC: suprabasal vacuolation (clear halos around keratinocyte nuclei) + granular layer epidermolysis (tonofilament collapse) + compact hyperkeratosis above; distinguishes EI from all other ichthyoses",
            "BLISTERING AT BIRTH — neonatal blistering resembles staphylococcal scalded skin (DDx); resolves by age 1-2yr; replaced by scale",
            "DARK VERRUCOUS SCALE — particularly over joints (elbows, knees, ankles), scalp; offensive odour (bacterial colonisation + keratin debris)",
            "PALMOPLANTAR KERATODERMA — prominent in KRT1 (more than KRT10 mutations, since KRT1 expressed in palms/soles suprabasal layers); fissuring + walking difficulty",
            "SECONDARY SUPERINFECTION (S. aureus) — MOST COMMON COMPLICATION; malodor = key complaint; bacterial scale colonization; antibiotics/antiseptic washes",
            "p.Asn171Ser — most common KRT1 mutation; helix initiation motif 1A; classic EI with PPK + scale + occasional blistering",
            "NO SYSTEMIC INVOLVEMENT — pure skin disorder; no internal organs; normal intelligence + lifespan",
            "MOSAIC FORMS — if somatic mutation only: ichthyosis hystrix / linear EHK (Blaschko lines); germline risk for offspring is 50%",
        ],
        "treatment": (
            "Infection management (priority for EI): "
            "Daily antiseptic washes (triclosan 1%, chlorhexidine 4%) — reduce S. aureus colonization + malodor; "
            "Topical mupirocin to crusted/infected lesions; "
            "Systemic antibiotics (flucloxacillin/trimethoprim) for clinical flares; "
            "Dilute bleach baths (sodium hypochlorite 0.005% = 1 tsp/gallon water) twice weekly — proven decolonization. "
            "Scale management: "
            "Emollients: urea 10-40% cream; lactic acid 10-12%; salicylic acid 5-10% for thick areas; "
            "Bathing: daily soaking + mechanical scale removal (loofah/wet glove); "
            "Retinoids: acitretin 0.3-0.5 mg/kg/day — reduces scale; blistering recurrence rare but monitor; "
            "Topical retinoids (tretinoin 0.05% cream) — adjunct for localized areas. "
            "Palmoplantar keratoderma: "
            "Urea 40-50% cream + occlusion; salicylic acid paste; orthopaedic footwear. "
            "Odour management: "
            "Antiseptic washing; clothing change daily; social/psychological support. "
            "Neonatal: "
            "NICU if blistering extensive; wound care as for EB (Mepitel/petroleum); distinguish from SSSS (culture + EHK biopsy). "
            "Genetics: AD — 50% transmission risk; preimplantation genetic testing (PGT-M) available."
        ),
        "monitoring": [
            "Dermatology: every 3-6 months — scale score; PPK severity; blistering flares; infection surveillance; swab culture quarterly",
            "Microbiological: quarterly skin swabs for S. aureus sensitivities; bleach bath frequency adjustment",
            "Retinoid: LFTs + lipids 1-2 monthly (first 3 months), then 3-monthly; DEXA if long-term",
            "Orthopaedic/physio: PPK walking difficulty; annual goniometry; footwear assessment",
            "QoL: DLQI; odour-specific QoL; social/school assessment; psychosocial support",
            "Ophthalmology: annual — ectropion rare but scale near eyelids possible",
            "Genetic counselling: 50% AD risk; mosaic forms — germline testing offered; PGT-M discussion",
            "Biopsy: EHK confirmation at diagnosis; repeat if phenotype changes unexpectedly",
        ],
        "ichthyosis_type": "Epidermolytic Ichthyosis (EI) — Bullous CRIE",
        "scale_morphology": "Dark verrucous scale joints/scalp; blistering neonatal → scale dominant childhood; PPK prominent",
        "pathognomonic": "EHK on biopsy (suprabasal vacuolation + granular epidermolysis + compact HK) + neonatal blistering → verrucous scale = KRT1-EI",
        "treatment_highlight": "Antiseptic washes daily (S. aureus decolonization); acitretin for scale; EHK on biopsy is pathognomonic",
        "avg_age_at_dx_yrs": 0.0,
    },
    # -- GJB3 — Erythrokeratoderma Variabilis (EKV) type 1 ---------------------------------
    {
        "gene": "GJB3",
        "alt_name": (
            "GJB3 (GJB3-270aa-1p34.3 / AD — Erythrokeratoderma-Variabilis-EKV-Type1 — "
            "TRANSIENT-MIGRATORY-FIGURATE-ERYTHEMA-FIXED-HYPERKERATOTIC-PLAQUES-PATHOGNOMONIC — "
            "ERYTHEMA-FLUCTUATES-Emotional-Stress-Temperature-Changes — "
            "PALMOPLANTAR-KERATODERMA-Limited-to-Skin-No-Systemic — "
            "Connexin-31-Gap-Junction-Epidermal-Communication)"
        ),
        "protein": (
            "GJB3 -- 1p34.3 AD -- GJB3-270aa -- "
            "Gap-Junction-Protein-Beta-3-Connexin-31-Cx31-31kDa-Hexameric-Connexon -- "
            "EKV1-OMIM-133200 -- "
            "FORMS-CONNEXONS-In-Upper-Epidermis-Cell-Cell-Communication-Differentiation -- "
            "GJB3-MUTATION-Dominant-Negative-Connexon-Dysfunction-Epidermal-Signalling -- "
            "TRANSIENT-MIGRATORY-FIGURATE-ERYTHEMATOUS-PATCHES-Vary-Hours-Days-PATHOGNOMONIC -- "
            "FIXED-HYPERKERATOTIC-PLAQUES-Knees-Elbows-Trunk-Persistent -- "
            "ERYTHEMA-TRIGGERS-Emotional-Stress-Temperature-Change-Friction -- "
            "PALMOPLANTAR-KERATODERMA-Variable-Severity -- "
            "PURE-SKIN-DISORDER-No-Systemic-Involvement -- "
            "p.Arg42Pro-Most-Common-European-GJB3 -- "
            "OMIM-Gene-GJB3-603324-Disease-EKV1-133200"
        ),
        "locus": "1p34.3",
        "protein_size": "270 aa / 31 kDa",
        "inheritance": (
            "AD (autosomal dominant; dominant-negative or GOF connexon); "
            "Also AR forms reported (rare, more severe); "
            "p.Arg42Pro: most common European EKV1 allele; "
            "de novo mutations in 10-20%; "
            "GJB4 (Cx30.3) mutations → EKV type 2 — clinically similar; "
            "Variable expressivity within families — same mutation → mild figurate erythema or prominent PPK; "
            "Erythema component may decrease with age in some patients; "
            "Hyperkeratotic plaques persist lifelong; "
            "No internal organ involvement; normal lifespan; "
            "Unique among ichthyoses: erythema is transient and migratory (not fixed)"
        ),
        "key_features": [
            "TRANSIENT MIGRATORY FIGURATE ERYTHEMATOUS PATCHES — change shape/location over hours/days; disappear and reappear; PATHOGNOMONIC for EKV1 (unique among ichthyoses)",
            "FIXED HYPERKERATOTIC PLAQUES — knees, elbows, dorsal feet, trunk; persistent; dark/yellowish; symmetric",
            "ERYTHEMA TRIGGERED BY EMOTIONAL STRESS, TEMPERATURE CHANGES, FRICTION — key history; erythema may precede hyperkeratosis by years",
            "PALMOPLANTAR KERATODERMA — variable severity (mild diffuse thickening to prominent PPK); present in most",
            "PURE SKIN DISORDER — no systemic involvement; normal intelligence, lifespan; no internal organs",
            "p.Arg42Pro — most common European GJB3 allele; EKV1 with figurate erythema + plaques + PPK",
            "GJB4 DDx — EKV type 2 (GJB4/Cx30.3); clinically nearly identical; gene panel distinguishes",
            "FIGURATE ERYTHEMA DECREASES WITH AGE in some — hyperkeratosis may persist; adult phenotype dominated by plaques",
        ],
        "treatment": (
            "Hyperkeratotic plaques: "
            "Emollients: urea 10-25% cream; lactic acid 10%; petroleum-based; "
            "Topical keratolytics: salicylic acid 5-10% cream/ointment; "
            "Topical retinoids: tretinoin 0.025-0.05% cream for plaques; "
            "Topical calcipotriol (vitamin D3 analogue) — beneficial for plaques; "
            "Occlusion overnight + keratolytic. "
            "Erythema management: "
            "Trigger avoidance: temperature regulation (cool environment, warm layers in cold); "
            "Stress management: psychological support; mindfulness; "
            "Topical low-moderate potency corticosteroids (short-term for active erythema); "
            "Topical calcineurin inhibitors (tacrolimus 0.03-0.1%) — anti-inflammatory, off-label. "
            "Palmoplantar keratoderma: "
            "Urea 40% cream + occlusion; salicylic acid paste; orthopaedic footwear. "
            "Systemic: "
            "Acitretin 0.3-0.5 mg/kg/day — effective for both plaques and PPK; less effect on transient erythema; "
            "Intermittent courses often preferred (mild disease); continuous if severe. "
            "Genetics: AD — 50% risk; PGT-M available."
        ),
        "monitoring": [
            "Annual dermatology: figurate erythema frequency/severity; plaque burden; PPK; trigger diary",
            "Retinoid: LFTs + lipids (if on systemic); teratogenicity counselling",
            "QoL: DLQI; social impact of erythema (visible, fluctuating — distressing); psychological screen",
            "Trigger diary: document emotional stress, temperature, friction events — correlate with flares",
            "Ophthalmology: annual — corneal involvement rare but GJB3 expressed in inner ear + peripheral nerves (hearing screen annually)",
            "Audiology: GJB3 mutations occasionally associated with sensorineural hearing loss (especially peripheral neuropathy overlap); annual hearing screen",
            "Genetic counselling: 50% AD risk; de novo exclusion by parental testing",
            "GJB4 gene: if GJB3 negative on panel — check GJB4 (EKV type 2)",
        ],
        "ichthyosis_type": "Erythrokeratoderma Variabilis (EKV) type 1",
        "scale_morphology": "Fixed hyperkeratotic plaques (knees/elbows) + transient migratory figurate erythema",
        "pathognomonic": "Transient migratory figurate erythematous patches (change daily) + fixed hyperkeratotic plaques = EKV1/GJB3",
        "treatment_highlight": "Trigger avoidance (stress/temperature); acitretin for plaques; GJB4 DDx; hearing screen annually",
        "avg_age_at_dx_yrs": 1.0,
    },
    # -- ALOX12B — ARCI type 8 / Lamellar Ichthyosis 8 ------------------------------------
    {
        "gene": "ALOX12B",
        "alt_name": (
            "ALOX12B (ALOX12B-701aa-17p13.1 / AR — ARCI8-Lamellar-Ichthyosis-LI8 — "
            "12R-LIPOXYGENASE-EPIDERMAL-CERAMIDE-BARRIER-DEFECT — "
            "BROWN-PLATE-LIKE-SCALE-PALMAR-FISSURING-PATHOGNOMONIC — "
            "NO-BLISTERING-HEAT-INTOLERANCE-ANHIDROSIS — "
            "GENE-PANEL-MANDATORY-Clinically-Indistinguishable-TGM1-NIPAL4-CYP4F22)"
        ),
        "protein": (
            "ALOX12B -- 17p13.1 AR -- ALOX12B-701aa -- "
            "Arachidonate-12-Lipoxygenase-Type-12R-LOX-77kDa -- "
            "ARCI8-OMIM-604769 -- "
            "OXYGENATES-LINOLEIC-ACID-Esterified-To-Ceramide-Corneocyte-Lipid-Envelope -- "
            "ALOX12B-LOF-Deficient-12R-Hydroxy-Ceramide-Impaired-Corneocyte-Lipid-Envelope -- "
            "BROWN-PLATE-LIKE-LAMELLAR-SCALE-Trunk-Extremities-Scalp -- "
            "PALMAR-FISSURING-PROMINENT-KEY-Feature -- "
            "NO-BLISTERING-Distinguishes-From-EI-KRT1 -- "
            "HEAT-INTOLERANCE-ANHIDROSIS-Eccrine-Duct-Occlusion -- "
            "COLLODION-BABY-Variable-40-60pct -- "
            "CLINICALLY-IDENTICAL-TGM1-NIPAL4-CYP4F22-Gene-Panel-Mandatory -- "
            "p.Gly386Ser-Most-Common-ALOX12B-European -- "
            "OMIM-Gene-ALOX12B-603741-Disease-ARCI8-604769"
        ),
        "locus": "17p13.1",
        "protein_size": "701 aa / 77 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic missense/nonsense/splice; "
            "p.Gly386Ser: most common European ALOX12B allele; "
            "Collodion baby: 40-60% (variable); "
            "Phenotype: lamellar ichthyosis — plate-like brown scale; palmar fissuring prominent; "
            "Clinically indistinguishable from TGM1-LI, NIPAL4-LI, CYP4F22-LI — gene panel is only way to distinguish; "
            "ALOX12B and ALOXE3 work in tandem — eLOX-3 (ALOXE3) converts 12R-HPETE to epoxy-alcohol ceramide; "
            "Both ALOX12B and ALOXE3 mutations → identical ARCI phenotype; "
            "TGase-1 assay: negative for ALOX12B (helps exclude TGM1); "
            "Anhidrosis: common; heat intolerance"
        ),
        "key_features": [
            "BROWN PLATE-LIKE LAMELLAR SCALE — trunk, extremities, scalp; similar morphology to TGM1-LI; persistent lifelong",
            "PALMAR FISSURING — prominent palmar fissures/hyperlinearity; useful clinical pointer within ARCI group (combined with genetic testing)",
            "NO BLISTERING — distinguishes ALOX12B from EI (KRT1) and HI (ABCA12 acute neonatal) — blistering never occurs in ALOX12B",
            "HEAT INTOLERANCE + ANHIDROSIS — eccrine duct occlusion; heat stroke risk; cooling strategies mandatory",
            "COLLODION BABY — 40-60% present as collodion baby at birth; shedding reveals lamellar scale",
            "p.Gly386Ser — most common European ALOX12B allele; confirms diagnosis with biallelic variants",
            "GENE PANEL MANDATORY — clinically identical to TGM1/NIPAL4/CYP4F22; TGase-1 assay (negative) excludes TGM1; ALOX12B + ALOXE3 tandem pathway (check both if one negative)",
            "RETINOID (acitretin) + emollients — standard effective therapy; response similar to TGM1-LI",
        ],
        "treatment": (
            "Emollients (first-line): "
            "Urea 20-40% cream; lactic acid 10-12%; petroleum jelly; "
            "Daily bathing + immediate application (soak-and-smear). "
            "Palmar fissuring: "
            "Urea 40-50% cream with occlusion overnight; "
            "Salicylic acid 10-15% paste for fissures; "
            "Petroleum-based barrier cream for mechanical protection. "
            "Systemic: "
            "Acitretin 0.3-0.5 mg/kg/day — effective; monitor LFTs + lipids + teratogenicity. "
            "Thermoregulation: "
            "Cool environment; cooling vest; water spray; "
            "Avoid strenuous exercise in heat; heat stroke emergency plan. "
            "Collodion baby: "
            "High-humidity incubator; emollient 2-4 hourly; IV fluid support (TEWL); "
            "Temperature regulation; ophthalmology (ectropion screen). "
            "Scalp: "
            "Salicylic acid descaling shampoo; "
            "Mineral oil pre-treatment (apply, wrap, bath, remove scale). "
            "Genetics: AR — 25% recurrence; prenatal/PGD available. "
            "Note: check ALOXE3 if ALOX12B negative on sequencing — tandem pathway partner."
        ),
        "monitoring": [
            "Annual dermatology: scale burden; PPK/palmar fissuring; heat intolerance events; infection signs",
            "Thermoregulation: heat events documented; cooling strategy updated; school/work environment plan",
            "Retinoid: LFTs + triglycerides monthly (first 3 months), then 3-monthly; DEXA if long-term; teratogenicity",
            "Ophthalmology: annual — ectropion screen (collodion baby history)",
            "TGase-1 enzyme assay: at diagnosis — negative confirms non-TGM1; helps narrow gene panel",
            "ALOXE3 gene: if ALOX12B result inconclusive — check ALOXE3 (tandem pathway; same ARCI phenotype)",
            "QoL: DLQI; palmar fissure pain VAS; occupational impact assessment",
            "Family: cascade; prenatal/PGD counselling",
        ],
        "ichthyosis_type": "ARCI8 — Lamellar Ichthyosis type 8",
        "scale_morphology": "Brown plate-like lamellar scale; prominent palmar fissuring; no blistering",
        "pathognomonic": "Brown plate-like lamellar scale + palmar fissuring + no blistering + anhidrosis = ALOX12B-LI (gene panel mandatory; TGase-1 assay excludes TGM1)",
        "treatment_highlight": "Acitretin + emollients; palmar fissuring responds to urea 40%; check ALOXE3 if ALOX12B negative; TGase-1 assay excludes TGM1",
        "avg_age_at_dx_yrs": 0.1,
    },
]


def _make_ichthyosis_patient(gene_entry: dict, seed: int) -> dict:
    rng = random.Random(seed)
    gene = gene_entry["gene"]

    # Age at diagnosis (years)
    base_age = gene_entry["avg_age_at_dx_yrs"]
    age_dx = round(max(0.0, base_age + rng.gauss(0, 0.3)), 1)

    follow_up = round(rng.uniform(0.5, 12.0), 1)

    # Scale severity
    if gene in ("ABCA12",):
        scale_severity = rng.choice(["Extreme", "Severe", "Severe"])
    elif gene in ("TGM1", "KRT1"):
        scale_severity = rng.choice(["Severe", "Moderate-Severe", "Severe", "Moderate"])
    elif gene == "STS":
        scale_severity = rng.choice(["Mild", "Mild-Moderate", "Moderate"])
    elif gene == "GJB3":
        scale_severity = rng.choice(["Mild", "Moderate", "Mild-Moderate"])
    else:
        scale_severity = rng.choice(["Moderate", "Moderate-Severe", "Severe"])

    # Collodion baby
    if gene in ("TGM1",):
        collodion = rng.random() < 0.85
    elif gene == "ABCA12":
        collodion = True
    elif gene in ("CYP4F22", "ALOX12B"):
        collodion = rng.random() < 0.45
    elif gene == "NIPAL4":
        collodion = rng.random() < 0.40
    elif gene in ("STS", "GJB3", "KRT1"):
        collodion = False
    else:
        collodion = rng.random() < 0.30

    # Erythroderma
    if gene == "TGM1":
        erythroderma = rng.random() < 0.22
    elif gene == "GJB3":
        erythroderma = True  # figurate erythema
    elif gene == "ABCA12":
        erythroderma = True
    else:
        erythroderma = rng.random() < 0.15

    # Main complication
    complication_map = {
        "TGM1": ["Heat_intolerance", "Ectropion", "Scale_infection", "Anhidrosis", "Scalp_alopecia"],
        "ABCA12": ["Corneal_exposure_ectropion", "Feeding_difficulty", "Respiratory_compromise", "Joint_contracture", "Sepsis"],
        "CYP4F22": ["Heat_intolerance", "PPK_fissuring", "Scale_infection", "Anhidrosis", "Scalp_scale"],
        "NIPAL4": ["Pruritus_severe", "Heat_intolerance", "Scale_infection", "Anhidrosis", "PPK_fissuring"],
        "STS": ["Cryptorchidism", "Posterior_corneal_opacity", "Scale_neck_extensor", "Kallmann_overlap", "Testicular_malignancy_surveillance"],
        "KRT1": ["S_aureus_superinfection", "Malodor", "PPK_fissuring", "Blistering_neonatal", "Joint_scale_contracture"],
        "GJB3": ["Figurate_erythema_flares", "PPK_fissuring", "Plaque_burden", "Sensorineural_hearing_loss_screen", "Psychological_impact_erythema"],
        "ALOX12B": ["Heat_intolerance", "Palmar_fissuring", "Scale_infection", "Anhidrosis", "Ectropion_collodion"],
    }
    complication = rng.choice(complication_map.get(gene, ["Scale_burden"]))

    # Treatment
    treatment_map = {
        "TGM1": ["Acitretin+Emollient", "Emollient_only", "Acitretin+Urea40pct"],
        "ABCA12": ["Acitretin+NICU_support", "Retinoid_ongoing", "Intensive_emollient+Retinoid"],
        "CYP4F22": ["Acitretin+Urea40pct", "Emollient_only", "Keratolytic+Emollient"],
        "NIPAL4": ["Acitretin+Antipruritic", "Dupilumab_trial", "Emollient+Antihistamine"],
        "STS": ["Emollient_urea20pct", "Orchidopexy_done", "Retinoid_severe_case"],
        "KRT1": ["Antiseptic_wash+Acitretin", "Antiseptic_only", "Retinoid+Bleach_bath"],
        "GJB3": ["Acitretin+Trigger_avoidance", "Topical_retinoid+Emollient", "Emollient+Calcipotriol"],
        "ALOX12B": ["Acitretin+Emollient", "Emollient_only", "Urea40pct+Retinoid"],
    }
    treatment_recommendation = rng.choice(treatment_map.get(gene, ["Emollient"]))

    return {
        "patient_id": f"{gene}-{seed % 10000:04d}",
        "gene": gene,
        "ichthyosis_type": gene_entry["ichthyosis_type"],
        "scale_morphology": gene_entry["scale_morphology"],
        "age_at_diagnosis_yrs": age_dx,
        "follow_up_yrs": follow_up,
        "scale_severity": scale_severity,
        "collodion_baby": collodion,
        "erythroderma": erythroderma,
        "complication": complication,
        "treatment_recommendation": treatment_recommendation,
        "pathognomonic": gene_entry["pathognomonic"],
        "treatment_highlight": gene_entry["treatment_highlight"],
    }


def _build_cohort():
    patients = []
    for i, gene_entry in enumerate(ICHTHYOSIS_GENES):
        base_seed = SEED_BASE + i
        for j in range(40):
            patients.append(_make_ichthyosis_patient(gene_entry, base_seed * 100 + j))
    return patients


# ── Public API ──────────────────────────────────────────────────────────────────────────

def overview() -> dict:
    """Aggregate statistics across all 8 ichthyosis genes (320 patients)."""
    cohort = _build_cohort()
    gene_counts = {}
    type_counts = {}
    inheritance_counts = {}

    for p in cohort:
        g = p["gene"]
        gene_counts[g] = gene_counts.get(g, 0) + 1
        t = p["ichthyosis_type"].split(" (")[0].split(" — ")[0][:30]
        type_counts[t] = type_counts.get(t, 0) + 1

    gene_summary = []
    for entry in ICHTHYOSIS_GENES:
        g = entry["gene"]
        gene_summary.append({
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "ichthyosis_type": entry["ichthyosis_type"],
            "scale_morphology": entry["scale_morphology"],
            "pathognomonic": entry["pathognomonic"],
            "n_patients": gene_counts.get(g, 0),
            "avg_age_at_dx_yrs": entry["avg_age_at_dx_yrs"],
        })

    return {
        "title": "Hereditary-Ichthyosis-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Ichthyosis Atlas — "
            "TGM1 · ABCA12 · CYP4F22 · NIPAL4 · STS · KRT1 · GJB3 · ALOX12B — "
            "320 patients (8 × 40, seeds 2278-2285)"
        ),
        "n_patients": len(cohort),
        "n_genes": 8,
        "seed_range": "2278-2285",
        "ichthyosis_categories": {
            "ARCI (Autosomal Recessive Congenital Ichthyosis)": "TGM1, ABCA12, CYP4F22, NIPAL4, ALOX12B",
            "X-linked Ichthyosis": "STS",
            "Epidermolytic Ichthyosis": "KRT1",
            "Erythrokeratoderma Variabilis": "GJB3",
        },
        "inheritance_map": {
            "TGM1": "AR", "ABCA12": "AR", "CYP4F22": "AR",
            "NIPAL4": "AR", "STS": "XLR", "KRT1": "AD", "GJB3": "AD", "ALOX12B": "AR",
        },
        "key_clinical_pearls": [
            "TGM1: COLLODION BABY (ectropion + eclabium + shiny membrane at birth) PATHOGNOMONIC for ARCI; most common ARCI (30-35%); plate-like lamellar scale post-shedding; acitretin most effective; TGase-1 assay confirms",
            "ABCA12: ARMOR-PLATE DIAMOND-SHAPED SCALE AT BIRTH PATHOGNOMONIC (Harlequin Ichthyosis); most severe ARCI; acitretin WITHIN 24h = LIFE-SAVING; absent lamellar granules on EM; NICU mandatory",
            "CYP4F22: Non-erythrodermic LI; fine brown lamellar scale + prominent PPK; no/mild collodion; TGase-1 assay negative (excludes TGM1); gene panel mandatory",
            "NIPAL4: Plate-like lamellar scale + PROMINENT PRURITUS (KEY DDx from TGM1-LI); anhidrosis; dupilumab under investigation for itch",
            "STS: LARGE DARK BROWN POLYGONAL SCALE neck/extensors + POSTERIOR CORNEAL OPACITY (slit-lamp, asymptomatic) + CRYPTORCHIDISM 20% males (orchidopexy mandatory) — X-linked; no collodion baby",
            "KRT1: EPIDERMOLYTIC HYPERKERATOSIS (EHK) ON BIOPSY PATHOGNOMONIC; neonatal blistering → verrucous scale; S. AUREUS SUPERINFECTION = most common complication; antiseptic washes daily",
            "GJB3: TRANSIENT MIGRATORY FIGURATE ERYTHEMA + FIXED HYPERKERATOTIC PLAQUES PATHOGNOMONIC (EKV1); erythema changes shape daily — unique among ichthyoses; trigger avoidance (stress/temperature)",
            "ALOX12B: Brown plate-like lamellar scale + PALMAR FISSURING + no blistering; gene panel mandatory (identical to TGM1/NIPAL4/CYP4F22); TGase-1 negative; check ALOXE3 (tandem partner)",
        ],
        "gene_summary": gene_summary,
        "diagnostic_algorithm": {
            "Step_1": "Clinical classification: (A) neonatal blistering → EI/KRT1 or HI/ABCA12; (B) collodion baby at birth → ARCI (TGM1/ABCA12/CYP4F22/NIPAL4/ALOX12B); (C) postnatal onset + neck/extensor scale in male → XLI/STS; (D) figurate erythema + plaques → EKV/GJB3",
            "Step_2": "TGase-1 enzyme activity (granulocytes/fibroblasts): absent → TGM1; normal → not TGM1 (points to CYP4F22/NIPAL4/ALOX12B/NIPAL4)",
            "Step_3": "STS sulfatase activity (leukocytes/fibroblasts): absent in males → XLI/STS confirmed; array CGH for Xp22.3 deletion size (Kallmann overlap?)",
            "Step_4": "Skin biopsy: EHK (suprabasal vacuolation + granular epidermolysis + compact HK) → EI/KRT1; absent lamellar granules on EM → HI/ABCA12; normal histology → ARCI types",
            "Step_5": "NGS gene panel (ARCI panel: TGM1/ABCA12/CYP4F22/NIPAL4/ALOX12B/ALOXE3/CERS3/PNPLA1/ST14/CASP14 + EI: KRT1/KRT10 + EKV: GJB3/GJB4 + XLI: STS): confirms gene + variant; determines management (retinoid type/dose, ophthalmology intensity, orchidopexy, NICU)",
        },
        "collodion_baby_genes": ["TGM1 (85%)", "ABCA12 (100% — armor plate)", "CYP4F22 (30%)", "NIPAL4 (30-50%)", "ALOX12B (40-60%)"],
        "no_collodion_genes": ["STS", "KRT1", "GJB3"],
        "type_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10]),
    }


def breakdown() -> dict:
    """Per-gene ichthyosis profiles across all 8 genes."""
    cohort = _build_cohort()
    by_gene = {}
    for p in cohort:
        by_gene.setdefault(p["gene"], []).append(p)

    gene_breakdown = {}
    for entry in ICHTHYOSIS_GENES:
        g = entry["gene"]
        pts = by_gene.get(g, [])

        severity_dist = {}
        complication_dist = {}
        treatment_dist = {}
        for p in pts:
            sv = p["scale_severity"]
            severity_dist[sv] = severity_dist.get(sv, 0) + 1
            cx = p["complication"]
            complication_dist[cx] = complication_dist.get(cx, 0) + 1
            tx = p["treatment_recommendation"]
            treatment_dist[tx] = treatment_dist.get(tx, 0) + 1

        avg_age = round(sum(p["age_at_diagnosis_yrs"] for p in pts) / len(pts), 2) if pts else 0
        avg_fu = round(sum(p["follow_up_yrs"] for p in pts) / len(pts), 1) if pts else 0
        collodion_n = sum(1 for p in pts if p["collodion_baby"])
        erythroderma_n = sum(1 for p in pts if p["erythroderma"])

        rng_c = random.Random(SEED_BASE + ICHTHYOSIS_GENES.index(entry) + 5000)
        if g == "TGM1":
            specific = {"Anhidrosis": rng_c.randint(37, 40), "Ectropion_managed": rng_c.randint(28, 35), "Scale_infection": rng_c.randint(18, 24), "Heat_stroke_episode": rng_c.randint(8, 14), "Scalp_alopecia": rng_c.randint(15, 20)}
        elif g == "ABCA12":
            specific = {"Ectropion_corneal_exposure": rng_c.randint(37, 40), "Respiratory_compromise": rng_c.randint(18, 26), "Joint_contracture": rng_c.randint(20, 28), "Sepsis_neonatal": rng_c.randint(14, 20), "Feeding_difficulty": rng_c.randint(35, 40)}
        elif g == "CYP4F22":
            specific = {"Anhidrosis": rng_c.randint(35, 40), "PPK_fissuring": rng_c.randint(28, 35), "Scale_infection": rng_c.randint(10, 16), "Heat_tolerance_impaired": rng_c.randint(30, 38), "Ectropion_mild": rng_c.randint(8, 14)}
        elif g == "NIPAL4":
            specific = {"Pruritus_significant": rng_c.randint(35, 40), "Anhidrosis": rng_c.randint(35, 40), "Scale_infection": rng_c.randint(12, 18), "PPK_fissuring": rng_c.randint(16, 22), "Heat_intolerance": rng_c.randint(30, 38)}
        elif g == "STS":
            specific = {"Posterior_corneal_opacity": rng_c.randint(37, 40), "Cryptorchidism_males": rng_c.randint(7, 10), "Neck_extensor_scale": rng_c.randint(38, 40), "Kallmann_overlap": rng_c.randint(2, 5), "Testicular_malignancy_screen": rng_c.randint(7, 10)}
        elif g == "KRT1":
            specific = {"S_aureus_infection": rng_c.randint(28, 35), "Malodor_significant": rng_c.randint(30, 38), "PPK_fissuring": rng_c.randint(25, 32), "Neonatal_blistering": rng_c.randint(35, 40), "Joint_scale": rng_c.randint(20, 28)}
        elif g == "GJB3":
            specific = {"Figurate_erythema_flares": rng_c.randint(37, 40), "PPK_moderate": rng_c.randint(28, 35), "Stress_triggered_erythema": rng_c.randint(35, 40), "Hearing_loss_screen_flag": rng_c.randint(5, 10), "Plaque_burden_elbows_knees": rng_c.randint(35, 40)}
        else:  # ALOX12B
            specific = {"Palmar_fissuring": rng_c.randint(30, 38), "Anhidrosis": rng_c.randint(35, 40), "Scale_infection": rng_c.randint(10, 16), "Heat_intolerance": rng_c.randint(30, 38), "Ectropion_mild": rng_c.randint(6, 12)}

        def _get_ddx_ich(gene: str) -> list:
            ddx_map = {
                "TGM1": ["ABCA12 (more severe; armor plate; EM no lamellar granules)", "NIPAL4 (more pruritic; TGase-1 normal)", "CYP4F22 (non-erythrodermic; PPK more prominent)", "ALOX12B (palmar fissuring; TGase-1 negative)"],
                "ABCA12": ["TGM1 (collodion only — not armor plate; TGase-1 absent)", "Restrictive dermopathy (fetal LMNA — arthrogryposis + thin skin; different)", "Peeling skin syndrome (CAST/FLG — acral; different morphology)", "Ichthyosis congenita (old term — now genetically defined by panel)"],
                "CYP4F22": ["TGM1 (TGase-1 absent; erythrodermic in 20%)", "NIPAL4 (prominent pruritus — KEY DDx)", "ALOX12B (palmar fissuring — prominent)", "KRT1 (blistering + EHK on biopsy — distinguishes)"],
                "NIPAL4": ["TGM1 (less pruritus; TGase-1 absent)", "CYP4F22 (minimal pruritus; PPK dominant)", "ALOX12B (palmar fissuring; less itch)", "Prurigo nodularis (acquired; no scale/family history; biopsy)"],
                "STS": ["TGM1 (collodion baby; lamellar scale not polygonal neck scale)", "Ichthyosis vulgaris (FLG; flexural sparing; palmar hyperlinearity; atopic background)", "KRT1 (blistering + EHK; different distribution)", "Refsum disease (phytanic acid; retinitis pigmentosa; neuropathy; systemic)"],
                "KRT1": ["KRT10 EI (identical EHK; no PPK in KRT10 — PPK distinguishes KRT1)", "ABCA12 HI (armor plate — not blistering; different)", "SSSS Staphylococcal scalded skin (neonatal; culture positive; no EHK)", "Pemphigus vulgaris (autoimmune; IIF; no EHK on biopsy)"],
                "GJB3": ["GJB4 EKV type 2 (identical clinically; gene sequencing distinguishes)", "Psoriasis (fixed plaques; no figurate erythema; pitting nails; family history)", "Erythema annulare centrifugum (acquired; no hyperkeratosis; serpiginous border)", "Darier disease (ATP2A2; follicular keratoses; V-shaped nails; different biopsy)"],
                "ALOX12B": ["TGM1 (TGase-1 absent; collodion dominant)", "NIPAL4 (prominent pruritus)", "CYP4F22 (PPK more prominent)", "ALOXE3 (tandem partner; identical phenotype; gene panel distinguishes)"],
            }
            return ddx_map.get(gene, [])

        gene_breakdown[g] = {
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "ichthyosis_type": entry["ichthyosis_type"],
            "scale_morphology": entry["scale_morphology"],
            "n_patients": len(pts),
            "avg_age_at_dx_yrs": avg_age,
            "avg_follow_up_yrs": avg_fu,
            "collodion_baby_n": collodion_n,
            "erythroderma_n": erythroderma_n,
            "pathognomonic": entry["pathognomonic"],
            "treatment_highlight": entry["treatment_highlight"],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:600],
            "monitoring": entry["monitoring"],
            "severity_distribution": dict(sorted(severity_dist.items(), key=lambda x: -x[1])),
            "complication_distribution": dict(sorted(specific.items(), key=lambda x: -x[1])),
            "treatment_distribution": dict(sorted(treatment_dist.items(), key=lambda x: -x[1])),
            "key_ddx": _get_ddx_ich(g),
            "patients_sample": pts[:5],
        }

    return {
        "title": "Hereditary-Ichthyosis-Atlas — Per-Gene Breakdown",
        "n_genes": 8,
        "n_patients": len(cohort),
        "gene_breakdown": gene_breakdown,
        "clinical_emergency_flags": [
            "ABCA12/HI NEONATAL: ARMOR-PLATE SCALE + ECTROPION + ECLABIUM AT BIRTH = NEONATAL EMERGENCY — tertiary NICU; acitretin within 24h; ophthalmology; airway assessment; ENT immediate",
            "TGM1/ARCI1 COLLODION BABY: ECTROPION + ECLABIUM + HIGH TEWL — high-humidity incubator; IV fluids; emollient 2-4h; ophthalmology urgent; NG feed if eclabium impairs suckling",
            "KRT1/EI NEONATAL: BLISTERING resembles SSSS — BIOPSY IMMEDIATELY for EHK; do NOT treat as SSSS without culture/biopsy; skin fragility wound care",
            "STS/XLI CRYPTORCHIDISM: ORCHIDOPEXY BY AGE 2yr — malignancy risk 3-5× if uncorrected; check testes in ALL male XLI patients at diagnosis",
            "ABCA12/HI ACITRETIN: START WITHIN 24h — delay worsens outcome; dose 0.5 mg/kg/day; formulate syrup if needed for neonates; do NOT wait for gene confirmation in classic HI phenotype",
            "GJB3/EKV HEARING: SENSORINEURAL HEARING LOSS SCREEN ANNUALLY — GJB3/Cx31 expressed in cochlea; rare overlap; audiometry required",
        ],
    }


def definitions() -> dict:
    """Glossary of ichthyosis biology, subtypes, treatments, and diagnostic tests."""
    return {
        "title": "Hereditary-Ichthyosis-Atlas — Definitions & Glossary",
        "gene_entries": {
            entry["gene"]: {
                "full_protein": entry["protein"],
                "inheritance_details": entry["inheritance"],
                "key_features": entry["key_features"],
                "treatment": entry["treatment"],
                "monitoring": entry["monitoring"],
            }
            for entry in ICHTHYOSIS_GENES
        },
        "skin_biology_glossary": {
            "Stratum corneum (SC)": "Outermost 10-20 cell layers of epidermis; anucleate corneocytes embedded in lipid bilayers (ceramide/cholesterol/fatty acids); forms the permeability barrier; 'brick-and-mortar' model — corneocytes = bricks; lipid matrix = mortar; ichthyosis disrupts SC assembly",
            "Cornified envelope (CE)": "Insoluble 15-nm-thick protein shell replacing plasma membrane in corneocytes; cross-linked by transglutaminases (TGase-1/TGM1) using isopeptide bonds; components: loricrin (70%), involucrin, SPRRs, elafin; defective CE = ARCI",
            "Lamellar granules (keratinosomes)": "Secretory organelles in upper spinous/granular keratinocytes; contain lipid precursors (glycosphingolipids/phospholipids) + hydrolytic enzymes; transported to SC extracellular space by ABCA12 (lipid transporter); absent in HI/ABCA12",
            "Transglutaminase-1 (TGase-1 / TGM1)": "Membrane-bound calcium-dependent enzyme; cross-links CE proteins (loricrin + involucrin + SPRRs) via ε-(γ-glutamyl)lysine isopeptide bonds; gene: TGM1 (14q12); absent/reduced activity in ARCI1; assayed in granulocytes or keratinocytes",
            "Cholesterol sulfate": "Abundant SC lipid (~5% in normal SC); normal function: desquamation inhibition (binds LEKTI-KLK axis); excess = X-linked ichthyosis (STS LOF); accumulation prevents corneocyte shedding → scale retention",
            "Collodion baby": "Neonatal presentation of ARCI: tight shiny transparent membrane (resembles collodion/cellophane) covering entire body at birth; restricts facial movement (ectropion, eclabium); associated with TGM1 (85%), ALOX12B, NIPAL4, CYP4F22; PATHOGNOMONIC for ARCI; sheds over 2-6 weeks",
            "Ectropion": "Eyelid eversion (turning outward) due to tight collodion membrane or hyperkeratotic scale near eyelid; prevents complete eye closure; corneal exposure → keratopathy; ophthalmology emergency; tarsorrhaphy if severe",
            "Eclabium": "Lip eversion due to collodion membrane; prevents normal lip seal; impairs feeding (suckling impossible); nasogastric tube mandatory; improves after collodion sheds",
            "Epidermolytic hyperkeratosis (EHK)": "Histopathological pattern: suprabasal vacuolation (perinuclear halos) + granular layer epidermolysis (tonofilament collapse, inclusion bodies) + compact hyperkeratosis; PATHOGNOMONIC for EI/KRT1 and KRT10; confirms KRT1/KRT10 diagnosis without sequencing",
            "Steroid sulfatase (STS)": "Microsomal membrane enzyme; hydrolyses cholesterol sulfate to cholesterol; gene: STS (Xp22.31); absent in XLI; assayed in leukocytes/fibroblasts; deficiency → cholesterol sulfate accumulation → scale retention → XLI",
        },
        "ichthyosis_type_glossary": {
            "ARCI": "Autosomal Recessive Congenital Ichthyosis — umbrella term for AR forms of congenital ichthyosis; genes include TGM1 (most common), ABCA12, CYP4F22, NIPAL4, ALOX12B, ALOXE3, CERS3, PNPLA1, ST14; spectrum: Harlequin (most severe) → Lamellar → Non-bullous CRIE → Self-healing collodion; erythroderma variable",
            "Harlequin Ichthyosis (HI)": "Most severe ARCI; gene: ABCA12; armor-plate scale at birth (plates separated by deep red fissures); absent lamellar granules on EM; acitretin life-saving within 24h of birth; historically lethal; modern NICU + retinoid → 80% 5yr survival; long-term: classic LI phenotype",
            "Lamellar Ichthyosis (LI)": "Non-erythrodermic ARCI subtype; genes: TGM1, CYP4F22, NIPAL4, ALOX12B (and others); plate-like dark scale; erythema absent or minimal; collodion baby in most; anhidrosis; acitretin + emollient standard; gene panel required to confirm specific gene",
            "Non-bullous CRIE": "Non-bullous Congenital Ichthyosiform Erythroderma — erythrodermic ARCI; diffuse erythema + scale; TGM1 most common cause; 'CRIE' = congenital ichthyosiform erythroderma; collodion baby → erythroderma; acitretin reduces both scale and erythema",
            "Epidermolytic Ichthyosis (EI)": "AD ichthyosis; genes: KRT1 or KRT10; blistering at birth → verrucous scale; EHK on biopsy PATHOGNOMONIC; dominant-negative collapse of suprabasal IF; KRT1 = PPK common; KRT10 = no PPK; S. aureus superinfection most common complication; antiseptic washes mandatory",
            "X-linked Ichthyosis (XLI)": "XLR; gene: STS; males affected; large dark brown polygonal scale (neck/extensors/trunk); palms/soles SPARED; posterior corneal opacities (asymptomatic; slit-lamp mandatory); cryptorchidism 20% males; steroid sulfatase absent in leukocytes; postnatal onset (1-3 months); emollient-managed; orchidopexy mandatory if cryptorchid",
            "Erythrokeratoderma Variabilis (EKV)": "AD connexin ichthyosis; genes: GJB3 (EKV1) or GJB4 (EKV2); two components: (1) transient migratory figurate erythema (changes shape/location daily — UNIQUE among ichthyoses) + (2) fixed hyperkeratotic plaques; triggered by stress/temperature; PPK variable; acitretin for plaques; trigger avoidance for erythema",
        },
        "treatment_glossary": {
            "Acitretin (systemic retinoid)": "Oral aromatic retinoid (0.3-0.5 mg/kg/day); first-line systemic therapy for ARCI/EI/EKV; reduces scale burden, ectropion, palmoplantar keratoderma; teratogenic (Category X) — contraception during + 3yr after in females; monitor: LFTs, triglycerides, DEXA (long-term); most effective systemic agent for lamellar ichthyosis spectrum",
            "TGase-1 enzyme activity assay": "Functional test for TGM1 mutations; measures transglutaminase-1 cross-linking activity in granulocytes (FACS-based assay) or keratinocytes (fluorescence assay); absent/severely reduced → TGM1-ARCI confirmed functionally; normal activity → not TGM1 → narrows to CYP4F22/NIPAL4/ALOX12B; available at specialist dermatogenetics centres",
            "STS sulfatase activity assay": "Functional test for XLI; measures steroid sulfatase activity in leukocytes or fibroblasts; absent = XLI confirmed; gold standard alongside gene sequencing; also available as cholesterol sulfate level in SC (elevated in XLI)",
            "Keratolytic emollients (urea/lactic acid)": "Urea 10-40%: humectant + keratolytic; disrupts H-bonds between corneocytes; reduces scale; urea 40-50% for PPK; lactic acid 5-12%: alpha-hydroxy acid; reduces SC thickness; ceramide-containing emollients especially beneficial (barrier replacement); apply immediately after bath (soak-and-smear technique)",
            "Dupilumab (anti-IL-4Rα)": "Biologic agent (IL-4/IL-13 blockade); FDA-approved for atopic dermatitis; growing evidence for ARCI pruritus — especially NIPAL4-LI where itch is prominent feature; off-label (2026); mechanism: reduces Th2 itch-related cytokines in ichthyosis skin; trials ongoing (Phase 2 NIPAL4, CYP4F22)",
            "Bleach baths (sodium hypochlorite)": "0.005% NaOCl (1 teaspoon household bleach per gallon water) twice weekly; reduces S. aureus skin colonization in KRT1-EI; proven in atopic dermatitis decolonization; malodor management; safe for long-term use; combine with antiseptic washes (chlorhexidine/triclosan)",
        },
        "diagnostic_tests": {
            "TGase-1 enzyme activity (granulocyte assay)": "FACS-based functional assay using patient peripheral blood granulocytes or keratinocytes; tests cross-linking activity of TGM1 enzyme; absent/severely reduced = TGM1-ARCI1 confirmed functionally; test distinguishes TGM1 from other ARCI types without gene sequencing; available at specialist dermatogenetics centres; request early in ARCI workup to guide gene panel prioritisation",
            "Steroid sulfatase activity (leukocyte/fibroblast)": "Biochemical assay measuring STS enzyme activity; absent in XLI males; carrier females ~50% activity; gold standard for XLI confirmation alongside STS deletion/mutation testing; cholesterol sulfate levels also elevated in XLI SC (supplemental test); array CGH for Xp22.3 deletion sizing (Kallmann overlap)",
            "Skin biopsy (EHK / lamellar granule EM)": "H&E: EHK (suprabasal vacuolation + granular epidermolysis + compact hyperkeratosis) = PATHOGNOMONIC for KRT1/KRT10-EI; electron microscopy (EM): absent lamellar granules = HI/ABCA12; tonofilament clumping in EI (suprabasal level — different from EBS basal level); normal histology in ARCI types — biopsy shows hyperkeratosis only",
            "NGS ichthyosis gene panel": "Next-generation sequencing panel covering: ARCI (TGM1, ABCA12, CYP4F22, NIPAL4, ALOX12B, ALOXE3, CERS3, PNPLA1, ST14, CASP14) + EI (KRT1, KRT10) + EKV (GJB3, GJB4) + XLI (STS); whole-exome as alternative; identifies biallelic AR variants or heterozygous AD mutations; essential for prognosis, counselling, gene therapy eligibility (ABCA12 trials upcoming), prenatal diagnosis",
            "Slit-lamp examination (posterior corneal opacity)": "Mandatory in all male patients presenting with ichthyosis; posterior corneal opacities (comma-shaped/punctate at Descemet's/deep stroma) present in >95% XLI males and ~25% carrier females; asymptomatic (no visual consequence); pathognomonic for XLI/STS; slit-lamp finding + neck/extensor dark scale = XLI clinical diagnosis (confirm with STS assay + gene testing)",
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
        print(f"  {g}: {info['n_patients']} pts | Type: {info['ichthyosis_type'][:50]}")

    print("\n=== DEFINITIONS (gene count) ===")
    df = definitions()
    print(f"  Genes: {list(df['gene_entries'].keys())}")
    print(f"  Skin biology terms: {len(df['skin_biology_glossary'])}")
    print(f"  Ichthyosis types: {len(df['ichthyosis_type_glossary'])}")
    print(f"  Treatments: {len(df['treatment_glossary'])}")
    print(f"  Diagnostic tests: {len(df['diagnostic_tests'])}")
