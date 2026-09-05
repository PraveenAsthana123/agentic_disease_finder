#!/usr/bin/env python3
"""Bone and Mineral Metabolism Disorders Atlas — Complete 8-Gene Hereditary Bone/Mineral Reference
PHEX    (X-linked Hypophosphatemia XLH; 749 aa; Xp22.11; XLD; most common hereditary rickets 1:20,000;
         FGF23 NOT cleaved → phosphaturia → hypophosphatemia + low 1,25(OH)2D3; burosumab FDA/EMA standard-of-care;
         phosphate alone CI without calcitriol — secondary HPT + nephrocalcinosis; craniosynostosis risk) ·
FGF23   (Autosomal Dominant Hypophosphatemia ADHR; 251 aa; 12p13.32; AD GOF; FGF23 resistant to cleavage;
         fluctuating phenotype — may remit when iron replete; burosumab effective; DDx from XLH: ADHR fluctuates) ·
SLC34A3 (Hereditary Hypophosphatemic Rickets with Hypercalciuria HHRH; 599 aa; 9q34.3; AR;
         primary renal phosphate wasting; FGF23 NORMAL/LOW (DDx from XLH/ADHR); 1,25(OH)2D3 ELEVATED;
         HYPERCALCIURIA → nephrolithiasis/nephrocalcinosis; NO calcitriol — already high; phosphate + low-Ca diet) ·
CASR    (FHH1 LOF: Familial Hypocalciuric Hypercalcemia — DO NOT parathyroidectomy; UCaR<0.01 DIAGNOSTIC;
         NSHPT: homozygous LOF — neonatal severe HPT — URGENT parathyroidectomy;
         ADH1 GOF: autosomal dominant hypocalcemia — calcitriol worsens nephrocalcinosis; thiazide + low-Ca diet;
         1078 aa; 3q21.1; calcium-sensing receptor; setpoint disorder not structural HPT) ·
MEN1    (Multiple Endocrine Neoplasia Type 1; Menin tumor suppressor; 610 aa; 11q13.1; AD; 1:10,000-30,000;
         TRIAD: pHPT 90-95% multiglandular → subtotal PTX; pNET 60-70% gastrinoma/insulinoma; pituitary 30-40% prolactinoma;
         cinacalcet — bridge; annual EUS; MEN1 screening panel; no phenotype-genotype correlation; thymic carcinoid lethal) ·
RET     (MEN2A/MEN2B; proto-oncogene tyrosine kinase; 1114 aa; 10q11.21; AD GOF;
         MTC 95-100% MANDATORY prophylactic thyroidectomy — codon-risk category A/B/C/D drives timing;
         PHEO BEFORE THYROID SURGERY — alpha-blockade then thyroidectomy; MEN2B thyroidectomy <6 months;
         codon C634F/Y + M918T = highest risk D; vandetanib/cabozantinib for metastatic MTC) ·
TCIRG1  (Autosomal Recessive Osteopetrosis ARO / Malignant Infantile Osteopetrosis; 830 aa; 11q13.2; AR;
         most common ARO 50-60%; osteoclast V-ATPase subunit; pancytopenia + hepatosplenomegaly + cranial nerve palsies;
         HSCT CURATIVE — must do before age 3 months for best optic nerve outcome; Ca-restricted diet peri-HSCT;
         dense bones + Erlenmeyer flask + bone-within-bone on X-ray; gamma-IFN bridge to HSCT) ·
CLCN7   (Autosomal Dominant Osteopetrosis Type 2 ADO2 / Albers-Schönberg Disease; 803 aa; 16p13.3; AD;
         most common osteopetrosis; dominant-negative CLC-7 Cl-/H+ exchanger; HSCT NOT EFFECTIVE for AD form;
         cranial nerve palsies; osteomyelitis mandible; avoid bisphosphonates — worsen impaired resorption;
         AR CLCN7 → severe infantile form — HSCT may benefit AR form; DDx from TCIRG1: AD vs AR)
320-patient aggregate cohort (8 × 40, seeds 1262–1269)
"""

import random

SEED_BASE = 1262

BONE_MINERAL_GENES = [
    # ── PHEX — X-linked Hypophosphatemia (XLH) ──────────────────────────────
    {
        "gene": "PHEX",
        "protein": "Phosphate-Regulating Endopeptidase Homolog X-Linked (PHEX)",
        "alias": (
            "PHEX; OMIM gene 300550; X-linked Hypophosphatemia XLH #307800; "
            "Xp22.11; 749 aa; ~85 kDa; XLD (X-linked dominant); most common hereditary rickets; "
            "prevalence 1:20,000; affects males and heterozygous females (variable severity); "
            "PHEX is a zinc metallopeptidase expressed on osteoblast/osteocyte surface; "
            "normally inactivates FGF23 (intact FGF23); PHEX LOF → FGF23 not cleared → elevated circulating intact FGF23; "
            "formerly called HYP (hypophosphatemia) gene; >600 pathogenic variants; no hotspot"
        ),
        "aa": "749 aa",
        "kDa": "~85 kDa",
        "locus": "Xp22.11",
        "omim_gene": 300550,
        "omim_disease": 307800,
        "inheritance": "XLD (X-linked dominant); males hemizygous typically more severe; females variable (X-inactivation)",
        "gene_class": (
            "PHEX (phosphate-regulating endopeptidase homolog, X-linked) is a zinc metalloendopeptidase "
            "in the neprilysin family; expressed on osteoblast/osteocyte cell surface; "
            "PHEX maintains low circulating intact FGF23 by promoting FGF23 cleavage (in concert with FAM20C); "
            "LOF → intact FGF23 accumulates → FGF23 binds FGFR1c/αKlotho on proximal tubule → "
            "downregulates NaPi-IIa/IIc sodium-phosphate cotransporters → phosphaturia → HYPOPHOSPHATEMIA; "
            "FGF23 also suppresses CYP27B1 (1α-hydroxylase) → 1,25(OH)2D3 inappropriately LOW despite hypophosphatemia; "
            "(1,25-D should be HIGH to compensate for hypophosphatemia — it fails to rise = inappropriate); "
            "Net result: chronic hypophosphatemia + relative 1,25-D deficiency → impaired osteoid mineralisation → rickets/osteomalacia; "
            "FGF23 also suppresses renal 1α-hydroxylase in collecting duct → further 1,25-D suppression; "
            "Serum calcium NORMAL; PTH NORMAL or mildly elevated; ALP elevated (bone turnover); "
            "Burosumab: anti-FGF23 monoclonal antibody (IgG1) → neutralises excess FGF23 → normalises phosphate + 1,25-D; "
            "FDA 2018 (paediatric), EMA 2018; dosing 0.8 mg/kg Q2W s.c. (paediatric); 1 mg/kg Q4W (adult); "
            "Phosphate salts + calcitriol: older regimen, replaced by burosumab for children; still used if burosumab unavailable"
        ),
        "phenotype": (
            "PAEDIATRIC (most typical): "
            "Short stature (GHD-like, but GH normal); genu varum (bowing of legs) — CARDINAL SIGN; "
            "waddling gait; dental abscesses (periodontal disease, odontogenic infections PATHOGNOMONIC); "
            "craniosynostosis (25–35%); enthesopathy (Achilles/plantar fascia calcification — adults); "
            "rachitic changes on X-ray: widened irregular growth plates, coarsening + fraying of metaphyses; "
            "Pseudofractures (Looser zones) in adults; coxa vara; "
            "BIOCHEMISTRY: hypophosphatemia + elevated 24h urine phosphate + TmP/GFR low + "
            "1,25(OH)2D3 inappropriately NORMAL or LOW + FGF23 elevated + PTH NORMAL + calcium NORMAL + ALP elevated; "
            "FEMALES: milder (X-inactivation); may have only hypophosphatemia without overt rickets; "
            "MALES: uniformly affected; "
            "Hearing loss: 25–30% (sensorineural or conductive); "
            "NBS: NOT routinely performed; diagnosis often delayed 2–3 years (mistaken for nutritional rickets)"
        ),
        "hallmark": (
            "FGF23 ELEVATED — PATHOGNOMONIC for phosphatonin-driven rickets (vs SLC34A3 where FGF23 is LOW); "
            "1,25(OH)2D3 INAPPROPRIATELY LOW (should rise with hypophosphatemia — failure to rise = diagnostic); "
            "DENTAL ABSCESSES WITHOUT CARIES — spontaneous periapical abscesses in otherwise intact teeth = XLH signature; "
            "PHOSPHATE SALTS WITHOUT CALCITRIOL ABSOLUTELY CONTRAINDICATED: "
            "  phosphate alone suppresses PTH secretion → rebound PTH elevation → secondary then tertiary HPT → "
            "  nephrocalcinosis; always co-prescribe calcitriol with phosphate (or switch to burosumab); "
            "BUROSUMAB MONITORING: serum phosphate (target low-normal, not normal — to avoid nephrocalcinosis); "
            "  do NOT combine burosumab with oral phosphate/calcitriol (hypercalciuria/nephrocalcinosis risk); "
            "CRANIOSYNOSTOSIS: screen skull X-ray at diagnosis; surgical release if symptomatic/progressive; "
            "NSAID CI: block prostaglandin-mediated FGF23 compensation; avoid in XLH"
        ),
        "treatment_alert": (
            "BUROSUMAB (KRN23): anti-FGF23 IgG1 MAb; 0.8 mg/kg Q2W s.c. (children ≥1yr); 1 mg/kg Q4W (adults); "
            "discontinue oral phosphate + calcitriol 1 week BEFORE starting burosumab; "
            "PHOSPHATE + CALCITRIOL (old standard): only if burosumab unavailable; always use BOTH together; "
            "PHOSPHATE ALONE NEVER: secondary HPT → tertiary HPT → nephrocalcinosis — severe long-term harm; "
            "DENTAL: regular dental review + prophylactic antibiotics pre-procedure; "
            "ORTHOPEDIC: guided growth (8-plate) for genu varum; corrective osteotomy in severe cases; "
            "CRANIOSYNOSTOSIS: neurosurgery if symptomatic; "
            "HEARING: audiometry annually; hearing aids if needed; "
            "MONITORING on burosumab: serum phosphate, ALP, PTH, renal ultrasound (nephrocalcinosis check); "
            "PREGNANCY: phosphate + calcitriol (burosumab not approved in pregnancy — switch back); "
            "FAMILY SCREENING: sequence all at-risk children; X-ray + phosphate + FGF23 to confirm carrier females"
        ),
        "key_ddx": (
            "FGF23 (ADHR): AD; FGF23 fluctuating (may remit when iron replete); same biochemistry as XLH; "
            "SLC34A3 (HHRH): AR; FGF23 NORMAL/LOW; 1,25-D ELEVATED; HYPERCALCIURIA + nephrolithiasis; NO calcitriol; "
            "Nutritional rickets: 25-OHD LOW; Ca LOW; FGF23 NORMAL; ALP high; responds to vitamin D + Ca; "
            "Hypophosphatasia: ALP VERY LOW (not elevated); urinary phosphoethanolamine elevated; ALPL gene; "
            "Oncogenic osteomalacia (TIO): FGF23 elevated (same biochem as XLH) but ACQUIRED; FGF-23-secreting mesenchymal tumor; "
            "PHEX in females vs males: females have variable expressivity; diagnose via molecular testing + family history"
        ),
        "bone_disease": "Hypophosphatemic rickets/osteomalacia; pseudofractures; enthesopathy",
        "mineral_disturbance": "Hypophosphatemia + phosphaturia; 1,25-D inappropriately low; Ca NORMAL; PTH NORMAL",
        "fgf23_status": "ELEVATED (phosphatonin-driven)",
        "severity_weights": [0.30, 0.40, 0.30],  # mild(F carriers)/moderate/severe(M)
    },

    # ── FGF23 — Autosomal Dominant Hypophosphatemia (ADHR) ──────────────────
    {
        "gene": "FGF23",
        "protein": "Fibroblast Growth Factor 23 (FGF23)",
        "alias": (
            "FGF23; OMIM gene 605380; Autosomal Dominant Hypophosphatemic Rickets ADHR #193100; "
            "12p13.32; 251 aa; ~32 kDa; AD GOF; rare; "
            "FGF23 encodes the phosphatonin FGF23; cleavage site at Arg176-Xxx-Xxx-Arg179 by furin/PHEX/FAM20C; "
            "GOF mutations at Arg176/Arg179 → resistant to proteolytic cleavage → intact FGF23 persists; "
            "iron deficiency amplifies phenotype (iron regulates FGF23 cleavage efficiency); "
            "fluctuating disease course — distinguishes from XLH"
        ),
        "aa": "251 aa",
        "kDa": "~32 kDa",
        "locus": "12p13.32",
        "omim_gene": 605380,
        "omim_disease": 193100,
        "inheritance": "AD (GOF; de novo or familial); variable expressivity; partial penetrance possible",
        "gene_class": (
            "FGF23 is the principal circulating phosphatonin; "
            "produced by osteocytes; binds FGFR1c + αKlotho co-receptor on proximal tubule and pituitary; "
            "downstream: inhibits NaPi-IIa/IIc → phosphaturia; inhibits CYP27B1 → low 1,25(OH)2D3; "
            "cleavage site: Arg176-X-X-Arg179 — cleavage separates N-terminal (active) from C-terminal (inactive) fragments; "
            "pathogenic GOF variants (Arg176Gln/Trp, Arg179Gln/Trp) resist cleavage → intact FGF23 accumulates; "
            "IRON REGULATION: iron deficiency → suppresses FGF23 cleavage → more intact FGF23; "
            "iron replete → restores cleavage → FGF23 cleaved → phenotype can REMIT; "
            "this iron-phenotype coupling is UNIQUE to ADHR (not seen in XLH where PHEX is gone) — "
            "explains the fluctuating course; "
            "Intact FGF23 assay (Kainos): measures biologically active intact form; "
            "C-terminal FGF23 assay: measures total (intact + fragments); "
            "In ADHR: elevated intact FGF23 even when phenotype mild/remitted"
        ),
        "phenotype": (
            "FLUCTUATING DISEASE COURSE — hallmark of ADHR: "
            "May present in childhood (rickets, bowing, growth failure — similar to XLH) OR "
            "in adulthood (osteomalacia, bone pain, fatigue, weakness); "
            "Some patients have spontaneous remissions (phosphate normalises temporarily — often correlates with iron replenishment); "
            "IRON DEFICIENCY TRIGGERS EXACERBATION: puberty, pregnancy, menstruation (iron loss) → phenotype worsens; "
            "Biochemistry during active phase: identical to XLH — hypophosphatemia + low TmP/GFR + "
            "  1,25-D inappropriately low/normal + intact FGF23 elevated; "
            "Biochemistry during remission: phosphate and FGF23 normalise; "
            "Dental abscesses (as XLH but may be less severe); "
            "Adults: muscle weakness, bone pain, pseudofractures (Looser zones), fractures"
        ),
        "hallmark": (
            "FLUCTUATING COURSE IS PATHOGNOMONIC — distinguishes ADHR from XLH (XLH does not remit); "
            "IRON DEFICIENCY EXACERBATES — iron repletion may cause phenotype remission; "
            "CHECK SERUM FERRITIN + TRANSFERRIN SAT in all ADHR patients; replete iron before escalating phosphate/burosumab; "
            "GOF AT CLEAVAGE SITE: only Arg176 and Arg179 are hotspot residues — almost all ADHR variants at these codons; "
            "INTACT FGF23 ASSAY: use intact assay (not C-terminal) for accurate activity assessment; "
            "BUROSUMAB WORKS: anti-FGF23 MAb neutralises intact FGF23 regardless of mutation mechanism; "
            "DDx FROM XLH: ADHR fluctuates + may have adult onset + AD pattern + iron connection; XLH is stable from birth"
        ),
        "treatment_alert": (
            "IRON REPLETION FIRST: if ferritin low, replete iron → may normalise phenotype before starting drugs; "
            "BUROSUMAB: effective (same mechanism as XLH — neutralises intact FGF23); "
            "PHOSPHATE + CALCITRIOL: second-line if burosumab unavailable; same co-prescription rule as XLH; "
            "PHOSPHATE ALONE NEVER (same secondary HPT risk as XLH); "
            "MONITOR: serum phosphate, FGF23, ferritin, ALP, renal ultrasound; "
            "PREGNANCY: iron supplementation especially important; burosumab not approved — switch to phosphate+calcitriol; "
            "GENETIC COUNSELLING: AD; 50% risk per child; sequencing confirms R176/R179 variant"
        ),
        "key_ddx": (
            "PHEX (XLH): X-linked (not AD); non-fluctuating from birth; no iron connection; FGF23 chronically elevated; "
            "SLC34A3 (HHRH): AR; FGF23 LOW; 1,25-D HIGH; hypercalciuria; no fluctuation; "
            "Oncogenic osteomalacia (TIO): acquired not hereditary; FGF23-secreting tumor; imaging needed; "
            "Nutritional rickets: FGF23 NORMAL; 25-OHD LOW; responds to Vit D alone; "
            "Fanconi syndrome: generalised proximal tubular wasting (amino acids + glucose + phosphate + urate + bicarb); "
            "Hypophosphatasia: ALP VERY LOW (ADHR: ALP elevated)"
        ),
        "bone_disease": "Hypophosphatemic rickets (childhood) / osteomalacia (adult); pseudofractures; Looser zones",
        "mineral_disturbance": "Fluctuating hypophosphatemia; FGF23 elevated (intact); 1,25-D inappropriately low; iron-dependent",
        "fgf23_status": "ELEVATED (GOF — resistant to cleavage); fluctuates with iron status",
        "severity_weights": [0.50, 0.30, 0.20],  # mild(remission phases)/moderate/severe(active)
    },

    # ── SLC34A3 — Hereditary Hypophosphatemic Rickets with Hypercalciuria (HHRH) ──
    {
        "gene": "SLC34A3",
        "protein": "Sodium-Phosphate Cotransporter IIc (NaPi-IIc / SLC34A3)",
        "alias": (
            "SLC34A3; OMIM gene 609826; Hereditary Hypophosphatemic Rickets with Hypercalciuria HHRH #241530; "
            "9q34.3; 599 aa; ~67 kDa; AR (biallelic); rare; "
            "NaPi-IIc is the primary high-affinity phosphate cotransporter in proximal tubule (brush border); "
            "LOF → phosphaturia → hypophosphatemia → APPROPRIATELY HIGH 1,25-D → hypercalciuria; "
            "FGF23 is LOW or NORMAL (not elevated) — KEY DDx from XLH/ADHR"
        ),
        "aa": "599 aa",
        "kDa": "~67 kDa",
        "locus": "9q34.3",
        "omim_gene": 609826,
        "omim_disease": 241530,
        "inheritance": "AR (biallelic; homozygous or compound heterozygous); heterozygotes may have subclinical hypercalciuria",
        "gene_class": (
            "SLC34A3 (NaPi-IIc) is a sodium-phosphate IIc cotransporter expressed on proximal tubule brush border; "
            "transports 2 Na+ + 1 HPO4²⁻ (divalent preference, electroneutral); "
            "NaPi-IIa (SLC34A1) transports electrogenic 3Na+ + HPO4²⁻; together they mediate >70% of renal phosphate reabsorption; "
            "HHRH pathway: NaPi-IIc LOF → renal phosphate wasting → hypophosphatemia → "
            "  1,25(OH)2D3 synthesis APPROPRIATELY ELEVATED (CYP27B1 upregulated by hypophosphatemia); "
            "  elevated 1,25-D → increased intestinal Ca absorption + bone resorption → HYPERCALCEMIA risk + "
            "  HYPERCALCIURIA → nephrolithiasis and nephrocalcinosis; "
            "FGF23 NORMAL or LOW: because 1,25-D is elevated (would normally suppress FGF23 production by osteocytes); "
            "  this is the opposite of XLH where FGF23 is elevated and suppresses 1,25-D; "
            "NaPi-IIa (SLC34A1) LOF → similar but different phenotype (Fanconi renal tubular syndrome often); "
            "Regulation: PTH inhibits NaPi-IIa/IIc (causes phosphaturia); FGF23 also inhibits; Vit D upregulates"
        ),
        "phenotype": (
            "Childhood: rickets (genu varum, growth failure, widened metaphyses on X-ray); "
            "  similar severity to XLH in childhood — rickets, bone pain, growth impairment; "
            "Adolescent/adult: bone pain, muscle weakness, stress fractures, osteomalacia; "
            "HYPERCALCIURIA: most consistent feature; 24h urine calcium >4 mg/kg/day; "
            "NEPHROLITHIASIS: calcium oxalate/phosphate stones (15–30%); "
            "NEPHROCALCINOSIS: medullary; can cause CKD long-term; "
            "BIOCHEMISTRY: hypophosphatemia + phosphaturia (TmP/GFR LOW) + 1,25(OH)2D3 ELEVATED + "
            "  FGF23 NORMAL or LOW + PTH suppressed/low-normal + calcium NORMAL or mildly elevated + "
            "  HYPERCALCIURIA + ALP elevated; "
            "HETEROZYGOTES: may have subclinical hypercalciuria ± hypophosphatemia (incomplete penetrance)"
        ),
        "hallmark": (
            "FGF23 NORMAL/LOW — DISTINGUISHES FROM XLH/ADHR (where FGF23 is elevated); "
            "1,25(OH)2D3 ELEVATED (appropriate response to hypophosphatemia — working PTH/CYP27B1 axis); "
            "HYPERCALCIURIA = CARDINAL FEATURE — not seen in XLH (which has normocalciuria); "
            "CALCITRIOL ABSOLUTELY CONTRAINDICATED: 1,25-D already high — adding more → severe hypercalciuria + nephrocalcinosis; "
            "BUROSUMAB NOT INDICATED: FGF23 is not elevated — the drug has nothing to neutralise; "
            "TREATMENT IS ORAL PHOSPHATE ALONE (+ calcium-restricted diet to limit hypercalciuria); "
            "HIGH-CALCIUM DIET AND CALCIUM SUPPLEMENTS CONTRAINDICATED: worsen hypercalciuria and stone risk; "
            "RENAL MONITORING MANDATORY: regular renal ultrasound for nephrocalcinosis; eGFR annually"
        ),
        "treatment_alert": (
            "PHOSPHATE SALTS ONLY: oral phosphate 20-40 mg/kg/day in 4-5 divided doses; "
            "NO CALCITRIOL/VITAMIN D ANALOGS: 1,25-D already elevated — CI (worsen hypercalciuria); "
            "NO HIGH-CALCIUM DIET: further increases urinary calcium → stone risk; "
            "THIAZIDE DIURETIC (HCTZ): reduces urinary calcium excretion — consider if nephrolithiasis; "
            "ADEQUATE HYDRATION: ≥2L/day to dilute urinary calcium; "
            "LOW OXALATE DIET: if calcium-oxalate stones; "
            "STONE MANAGEMENT: lithotripsy/ureteroscopy as needed; "
            "RENAL ULTRASOUND: 6-monthly in children; annually in adults; "
            "BUROSUMAB: NOT indicated (FGF23 normal — would not help and has no mechanism); "
            "MONITORING: serum phosphate, Ca, 1,25-D, PTH, 24h urine Ca and phosphate, eGFR, renal US; "
            "GENETIC COUNSELLING: AR — 25% risk in siblings; test parents"
        ),
        "key_ddx": (
            "PHEX (XLH): FGF23 ELEVATED; 1,25-D LOW; normocalciuria; XLD not AR; burosumab appropriate; "
            "FGF23 (ADHR): FGF23 ELEVATED; 1,25-D LOW; normocalciuria; AD; iron-fluctuating; "
            "OCRL (Lowe syndrome/Dent-2): phosphaturia + proteinuria + cataracts + aminoaciduria; FGF23 normal; "
            "Dent disease (CLCN5): X-linked; phosphaturia + proteinuria + nephrocalcinosis; FGF23 normal; "
            "Primary hyperparathyroidism: phosphaturia BUT PTH elevated + 1,25-D may be elevated; Ca ELEVATED not normal; "
            "Tumor-induced osteomalacia: FGF23 elevated (as XLH biochemistry) — acquired; imaging shows tumor"
        ),
        "bone_disease": "Hypophosphatemic rickets/osteomalacia; stress fractures",
        "mineral_disturbance": "Hypophosphatemia + hypercalciuria; 1,25-D ELEVATED; FGF23 NORMAL/LOW; nephrolithiasis/nephrocalcinosis risk",
        "fgf23_status": "NORMAL or LOW (distinguishing from PHEX/FGF23-driven disorders)",
        "severity_weights": [0.40, 0.40, 0.20],
    },

    # ── CASR — Familial Hypocalciuric Hypercalcemia / ADH / NSHPT ─────────────
    {
        "gene": "CASR",
        "protein": "Calcium-Sensing Receptor (CaSR)",
        "alias": (
            "CASR; OMIM gene 601199; "
            "FHH1 (LOF) #145980; ADH1 (GOF) #601198; NSHPT #239200; "
            "3q21.1; 1078 aa; ~130 kDa; "
            "CaSR is a class C GPCR sensing extracellular calcium → suppresses PTH; "
            "LOF → calcium setpoint HIGH → hypercalcemia but LOW urine calcium (renal retention); "
            "GOF → calcium setpoint LOW → hypocalcemia; biallelic LOF → NSHPT (severe neonatal)"
        ),
        "aa": "1078 aa",
        "kDa": "~130 kDa",
        "locus": "3q21.1",
        "omim_gene": 601199,
        "omim_disease": 145980,
        "inheritance": (
            "FHH1: AD LOF (heterozygous); "
            "NSHPT: AR (homozygous or compound het LOF) OR de novo dominant LOF (few reported); "
            "ADH1: AD GOF (heterozygous)"
        ),
        "gene_class": (
            "CaSR (calcium-sensing receptor) is a 1078 aa class C GPCR; "
            "expressed in parathyroid chief cells, renal tubules (thick ascending limb + collecting duct), "
            "osteoblasts/osteoclasts, gut, thyroid C-cells; "
            "high Ca²⁺ → CaSR activated → Gαq/Gi signalling → PTH SUPPRESSED + calcitonin RELEASED + "
            "renal calcium excretion INCREASED (NKCC2 regulated); "
            "CaSR sets the 'calcium setpoint' — the Ca²⁺ level at which PTH is half-maximally suppressed; "
            "LOF (FHH1): setpoint shifted RIGHT → PTH suppressed at HIGHER Ca²⁺ → "
            "  hypercalcemia with normal/mildly elevated PTH → renal calcium RETAINED (low urine Ca); "
            "  benign: no end-organ damage (no nephrocalcinosis, no bone loss); "
            "  hypercalcemia is lifelong but does not progress; "
            "GOF (ADH1): setpoint shifted LEFT → PTH suppressed at LOWER Ca²⁺ → "
            "  hypocalcemia (± hypomagnesemia) with suppressed PTH → may cause Chvostek/Trousseau signs; "
            "Biallelic LOF (NSHPT): no functional CaSR → severe uncontrolled hypercalcemia from birth; "
            "  PTH massively elevated; bone demineralisation; respiratory failure; "
            "Activating antibodies: autoimmune anti-CaSR Ab → acquired ADH (not CASR mutation); "
            "Calcimimetics (cinacalcet, evocalcet): positive allosteric modulators — shift setpoint left → "
            "  reduce PTH + Ca²⁺ in LOF (FHH1) but WORSEN hypocalcemia in GOF (ADH1)"
        ),
        "phenotype": (
            "FHH1 (AD LOF — most common): "
            "Asymptomatic hypercalcemia (usually discovered incidentally); "
            "Serum Ca 2.7–3.2 mmol/L; PTH normal or mildly elevated; "
            "24h urine Ca/Cr ratio (UCaR) < 0.01 (< 100 mg/24h) — DIAGNOSTIC; "
            "Urine Ca LOW (kidneys retain calcium — wrong direction for PHPT); "
            "No nephrocalcinosis, no nephrolithiasis, no bone loss, no neuromuscular symptoms; "
            "Benign — no treatment required; parathyroidectomy NOT indicated; "
            "NSHPT (biallelic LOF — rare): "
            "Neonatal: severe hypercalcemia (Ca > 4 mmol/L); respiratory failure; hypotonia; fractures; "
            "PTH massively elevated; bones demineralised; polyuria; dehydration; "
            "URGENT total parathyroidectomy if severe; cinacalcet bridge in mild NSHPT; "
            "ADH1 (AD GOF): "
            "Hypocalcemia (Ca 1.8–2.1 mmol/L); hypomagnesemia; PTH SUPPRESSED (paradoxically); "
            "May cause tetany, seizures (especially neonatal); "
            "Calcitriol treatment → urinary Ca rises → nephrocalcinosis risk; "
            "Must treat symptoms (Ca, Mg) without over-correcting (target low-normal Ca)"
        ),
        "hallmark": (
            "FHH1 — DO NOT PARATHYROIDECTOMY: "
            "  UCaR (24h urinary Ca to creatinine ratio) < 0.01 is DIAGNOSTIC of FHH1; "
            "  vs primary HPT: UCaR >0.01–0.02; "
            "  unnecessary parathyroidectomy in FHH = permanent hypoparathyroidism — devastating lifelong harm; "
            "  CaSR genetic testing BEFORE any parathyroid surgery in hypercalcaemia; "
            "ADH1 (GOF) — CALCITRIOL CAUTION: "
            "  calcitriol supplements → urinary Ca rises → nephrocalcinosis → CKD; "
            "  target low-normal serum Ca (not normal); consider thiazide diuretic to reduce urine Ca; "
            "  recombinant PTH (1-34 teriparatide) or PTH(1-84) treats hypocalcemia without worsening urine Ca; "
            "CINACALCET IN FHH1: may reduce Ca but NOT approved for FHH (benign — why treat?); "
            "NSHPT EMERGENCY: neonatal hypercalcemia + fractures + respiratory failure → URGENT PTX or IV bisphosphonate bridge; "
            "FAMILIAL CLUSTERING: FHH1 often multi-generation; test parents before labelling as sporadic PHPT"
        ),
        "treatment_alert": (
            "FHH1: OBSERVATION ONLY — no specific treatment (benign, asymptomatic); "
            "  annual serum Ca, PTH, 25-OHD (ensure vitamin D sufficiency); "
            "  AVOID unnecessary parathyroidectomy; "
            "NSHPT: cinacalcet bridge (limited evidence in infants); total parathyroidectomy if severe; "
            "ADH1 treatment: "
            "  calcium carbonate (elemental Ca) to maintain low-normal serum Ca; "
            "  magnesium supplements if hypomagnesemia; "
            "  AVOID CALCITRIOL or use at very low doses with URINE CALCIUM MONITORING; "
            "  THIAZIDE diuretic (HCTZ) reduces urine calcium → allows safer calcitriol use; "
            "  RECOMBINANT PTH (rhPTH 1-34 / teriparatide): corrects hypocalcemia without raising urine Ca — experimental; "
            "  RENAL ULTRASOUND: 6-monthly in ADH1 (nephrocalcinosis surveillance); "
            "CaSR MUTATION TESTING: essential before any PTX decision in normocalciuric hypercalcemia"
        ),
        "key_ddx": (
            "Primary HPT (PHPT): Ca elevated + PTH elevated + UCaR >0.02 (hypercalciuria); parathyroidectomy curative; "
            "FHH1 vs PHPT: UCaR <0.01 = FHH; PHPT: imaging shows adenoma; "
            "ADH1 vs hypoparathyroidism: ADH1 has GOF CASR mutation; hypoparathyroidism: absent/damaged gland + PTH low; "
            "ADH1 vs pseudohypoparathyroidism (PHP1A): PHP → PTH HIGH (resistance); ADH1 → PTH SUPPRESSED; "
            "FHH2 (GNA11 LOF): identical phenotype to FHH1 but different gene; "
            "FHH3 (AP2S1 LOF): same phenotype; clathrin adaptor protein"
        ),
        "bone_disease": "FHH1: NONE (benign); NSHPT: severe demineralisation + fractures; ADH1: mild osteopenia",
        "mineral_disturbance": "FHH1: hypercalcemia + LOW urine Ca; ADH1: hypocalcemia + raised urine Ca; NSHPT: severe hypercalcemia",
        "fgf23_status": "NORMAL (mineral setpoint disorder, not phosphatonin pathway)",
        "severity_weights": [0.60, 0.25, 0.15],  # mild(FHH1 benign)/moderate(ADH1)/severe(NSHPT)
    },

    # ── MEN1 — Multiple Endocrine Neoplasia Type 1 ──────────────────────────
    {
        "gene": "MEN1",
        "protein": "Menin (MEN1 Tumor Suppressor)",
        "alias": (
            "MEN1; OMIM gene 613733; Multiple Endocrine Neoplasia Type 1 #131100; "
            "11q13.1; 610 aa; ~67 kDa; AD; prevalence 1:10,000–30,000; "
            "Menin is a nuclear scaffold protein; tumor suppressor (2-hit); "
            "TRIAD: primary HPT (90-95%) + pNETs (60-70%) + pituitary adenoma (30-40%); "
            "no phenotype-genotype correlation; >600 germline variants"
        ),
        "aa": "610 aa",
        "kDa": "~67 kDa",
        "locus": "11q13.1",
        "omim_gene": 613733,
        "omim_disease": 131100,
        "inheritance": "AD; high penetrance (>95% by age 50); 10% de novo; 2-hit tumor suppressor",
        "gene_class": (
            "MEN1 encodes Menin — a scaffold/adaptor protein in the nucleus; "
            "Menin interacts with: LEDGF/p75 (epigenetic regulator), "
            "SET1/MLL histone methyltransferase complex (H3K4me3 activating mark → tumour suppressor genes), "
            "FANCD2 (DNA repair), JunD (transcriptional repressor), Smad3 (TGF-β signalling); "
            "LOF → loss of epigenetic tumour suppressor activity → NET proliferation in multiple glands; "
            "2-hit: germline LOF allele + somatic LOH (loss of heterozygosity) at 11q13 → full LOF in tumour; "
            "Tissues: parathyroid (chief cells), pancreatic islets (alpha/beta/delta/PP), anterior pituitary (lacto/somato/cortico/thyrotroph); "
            "Adrenal cortical adenomas (20-25%); thymic/bronchial carcinoids (5-10%, often lethal); "
            "No phenotype-genotype correlation: same variant → HPT in one family member, insulinoma in another; "
            "MEN1 gene sequence: 10 exons; no hotspot — must sequence full gene + MLPA for deletions"
        ),
        "phenotype": (
            "PRIMARY HYPERPARATHYROIDISM (pHPT) — 90-95% by age 50: "
            "Multiglandular disease (4-gland hyperplasia ≠ single adenoma as in sporadic PHPT); "
            "Earliest and most common manifestation; presents 20-30 years typically; "
            "Hypercalcemia; elevated PTH; UCaR >0.02 (vs FHH1 <0.01); "
            "Nephrolithiasis in 25-35%; bone demineralisation; neuromuscular symptoms; "
            "PANCREATIC NETs (pNETs) — 60-70%: "
            "Gastrinoma (most common functional, 40%): Zollinger-Ellison syndrome — peptic ulcers, diarrhoea, elevated gastrin; "
            "Insulinoma (10%): hypoglycaemia episodes; "
            "Non-functional pNET (50%): most >2 cm → resect (malignant potential); "
            "VIPoma/glucagonoma: rare; "
            "PITUITARY ADENOMA — 30-40%: "
            "Prolactinoma (most common, 60% of pituitary): amenorrhoea, galactorrhoea, impotence; "
            "GH-secreting: acromegaly; ACTH-secreting: Cushing's disease; "
            "THYMIC CARCINOID (5-10%): often non-functional; lethal if metastatic; annual CT chest"
        ),
        "hallmark": (
            "MULTIGLANDULAR PARATHYROID DISEASE: 4-gland hyperplasia in MEN1 vs single adenoma in sporadic PHPT; "
            "SUBTOTAL (3.5-gland) PARATHYROIDECTOMY preferred: preserves some parathyroid function (vs total → permanent hypoparathyroidism); "
            "ANNUAL BIOCHEMICAL SURVEILLANCE PANEL: "
            "  Ca²⁺, PTH, gastrin (fasting), glucose, insulin (fasting, C-peptide), glucagon, chromogranin A, "
            "  prolactin, IGF-1, cortisol (overnight DST), annual EUS from age 20; "
            "GASTRINOMA MANAGEMENT: PPI acid suppression first → then surgical debulking; "
            "  serum gastrin >10× ULN + secretin stimulation test (gastrin rise >120 pg/mL) = DIAGNOSTIC; "
            "INSULINOMA: surgical resection if localised (EUS or 68Ga-DOTATATE PET-CT); "
            "  medical: diazoxide + somatostatin analogue (octreotide) pre-op; "
            "PROLACTINOMA: cabergoline/bromocriptine first-line (usually shrinks); neurosurgery if visual field compromise; "
            "NO PHENOTYPE-GENOTYPE CORRELATION: cannot predict manifestations from variant type; "
            "CINACALCET: temporary Ca²⁺ control pre-surgery or in patients declining surgery (not curative); "
            "THYMIC CARCINOID: annual chest CT; smoking cessation essential (smoking → thymic carcinoid risk)"
        ),
        "treatment_alert": (
            "PARATHYROIDECTOMY: subtotal (3.5-gland) OR total with autotransplantation; "
            "  NOT single-gland removal (MEN1 has occult multi-gland disease → rapid recurrence); "
            "POSTOP MONITORING: Ca²⁺ + PTH hourly perioperatively (hungry bone — IV Ca drip ready); "
            "GASTRINOMA: PPIs (omeprazole 40–80 mg/day) → then surgical debulking; "
            "  somatostatin analogue (lanreotide/octreotide) for unresectable gastrinoma; "
            "IMAGING: 68Ga-DOTATATE PET-CT (superior to octreoscan for pNETs); "
            "EVEROLIMUS + SUNITINIB: approved for progressive pNETs; "
            "GENETIC COUNSELLING: AD; test all first-degree relatives; annual biochemistry from age 5-10 in gene-positive relatives; "
            "PREGNANCY: MEN1 may worsen in pregnancy (HPT + prolactinoma — dopamine agonists safe in T1-T2); "
            "PASIREOTIDE: for Cushing's disease in ACTH-secreting pituitary adenoma"
        ),
        "key_ddx": (
            "MEN2A (RET): MTC is dominant + pheo + parathyroid (only 15-20% HPT; single adenoma unlike MEN1 multiglandular); "
            "MEN4 (CDKN1B): identical phenotype to MEN1 but RET gene; "
            "Sporadic PHPT: single adenoma; UCaR >0.02; no family history; no pituitary/pNET; "
            "MEN1 vs FIPA (AIP): familial isolated pituitary adenoma (AIP LOF); no HPT or pNET; "
            "VHL syndrome: pheo + RCC + CNS hemangioblastoma; HPT absent; "
            "Sporadic gastrinoma: no germline MEN1 mutation; older age; single tumor often"
        ),
        "bone_disease": "Osteoporosis/osteopenia from pHPT (bone resorption); nephrolithiasis; osteitis fibrosa cystica (severe/untreated)",
        "mineral_disturbance": "Hypercalcemia (pHPT); elevated PTH; UCaR >0.02; hypophosphatemia; elevated ALP (bone turnover)",
        "fgf23_status": "NORMAL (HPT drives hypercalcemia, not phosphatonin pathway)",
        "severity_weights": [0.20, 0.40, 0.40],
    },

    # ── RET — Multiple Endocrine Neoplasia Type 2A/2B ────────────────────────
    {
        "gene": "RET",
        "protein": "RET Proto-Oncogene (Tyrosine Kinase Receptor)",
        "alias": (
            "RET; OMIM gene 164761; MEN2A #171400; MEN2B #162300; FMTC (familial MTC) #155240; "
            "10q11.21; 1114 aa; ~124 kDa; AD GOF; "
            "RET is a receptor tyrosine kinase for GDNF-family ligands; "
            "GOF mutations → constitutive kinase activation → MTC + pheo + HPT (MEN2A) or "
            "MTC + pheo + marfanoid + mucosal neuromas (MEN2B)"
        ),
        "aa": "1114 aa",
        "kDa": "~124 kDa",
        "locus": "10q11.21",
        "omim_gene": 164761,
        "omim_disease": 171400,
        "inheritance": "AD GOF; penetrance near 100% for MTC; de novo in ~6% MEN2A, ~50% MEN2B",
        "gene_class": (
            "RET (rearranged during transfection) encodes a receptor tyrosine kinase; "
            "extracellular: cadherin-like + cysteine-rich domains; transmembrane; intracellular: kinase domain; "
            "ligands: GDNF, NRTN, ARTN, PSPN (glial-cell-derived neurotrophic factor family) + co-receptor GFRα1-4; "
            "normally expressed in: thyroid C-cells, adrenal chromaffin cells, parathyroid, enteric nervous system; "
            "GOF cysteine-rich domain mutations (Cys609/618/620/630/634): "
            "  unpaired Cys → intermolecular disulfide bond → CONSTITUTIVE DIMERISATION → ligand-independent kinase activation; "
            "  risk categories: C634R/Y/W (highest among cysteine codons); "
            "GOF kinase domain mutations: "
            "  M918T (codon 918): RET kinase constitutively active monomerically → MOST AGGRESSIVE; "
            "  A883F: MEN2B-like; E768D, V804M/L: lower-risk; "
            "CODON-BASED RISK STRATIFICATION (ATA 2015 guidelines): "
            "  Category D (highest): C634F/Y, A883F, M918T → thyroidectomy ≤6 months of age; "
            "  Category C (high): C618R/S/G, C620R/G/F, C630R/Y, C634G/R/S/W → thyroidectomy ≤5 years; "
            "  Category B (intermediate): E768D, L790F, V804M/L, S891A, Y791F etc. → thyroidectomy ≤5yr or if calcitonin elevated; "
            "Vandetanib/cabozantinib: RET kinase inhibitors; pralsetinib/selpercatinib: highly selective RET inhibitors"
        ),
        "phenotype": (
            "MEN2A (most common, 85%): "
            "MEDULLARY THYROID CANCER (MTC) 95-100%: calcitonin-secreting C-cell tumour; "
            "  may be bilateral/multifocal; flushing + diarrhoea if metastatic (calcitonin excess); "
            "PHAEOCHROMOCYTOMA (40-50%): usually bilateral; chromaffin cell tumour; "
            "  LIFE-THREATENING HYPERTENSIVE CRISIS during surgery if untreated; "
            "PRIMARY HPT (15-20%): usually single parathyroid adenoma (unlike MEN1 multiglandular); "
            "MEN2B (rare, 5%): "
            "MTC (100%; most aggressive — earlier onset, 1st year of life); "
            "PHEO (50%): bilateral; "
            "MARFANOID HABITUS: tall, long limbs, joint laxity; "
            "MUCOSAL NEUROMAS: tongue, lips, eyelids — EARLIEST FEATURE; ganglioneuromatosis of bowel → constipation; "
            "FMTC: MTC only (no pheo or HPT); low-risk codons; "
            "PROPHYLACTIC THYROIDECTOMY: mainstay of management; timing = codon risk category; "
            "BASAL CALCITONIN: most sensitive MTC biochemical marker; "
            "CARCINOEMBRYONIC ANTIGEN (CEA): second marker for MTC"
        ),
        "hallmark": (
            "PHEO BEFORE THYROID SURGERY — MANDATORY: "
            "  alpha-blockade (phenoxybenzamine/doxazosin) for 10-14 days → then beta-blockade if tachycardia → "
            "  THEN safe thyroidectomy; unblocked pheo during surgery → hypertensive crisis → stroke/death; "
            "  24h urine metanephrines/catecholamines OR plasma fractionated metanephrines ANNUALLY in all MEN2; "
            "PROPHYLACTIC THYROIDECTOMY TIMING BY CODON: "
            "  D (highest — M918T, C634F/Y, A883F): within first 6 months of life; "
            "  C (high — C634 others, C618/620/630): by age 5; "
            "  B (intermediate): by age 5 OR when calcitonin rises (if deferring); "
            "RET GENE TESTING IN INDEX CASE: sequence full exons 10, 11, 13, 14, 15, 16 (codon hotspots); "
            "PREDICTIVE TESTING IN RELATIVES: from birth (DNA — not phenotype-based); "
            "SELECTIVE RET INHIBITORS (selpercatinib/pralsetinib): FDA approved for metastatic MTC; "
            "CALCITONIN: sensitive biomarker for MTC recurrence/persistence; "
            "MEN2B — MUCOSAL NEUROMAS ARE EARLIEST SIGN: tongue examination at birth in all children of MEN2B parents"
        ),
        "treatment_alert": (
            "PRE-SURGICAL PHEO SCREENING: 24h urine metanephrines or plasma metanephrines — BEFORE any neck surgery; "
            "ALPHA-BLOCKADE: phenoxybenzamine 10-40 mg/day × 10-14 days + adequate hydration pre-pheo adrenalectomy; "
            "TOTAL THYROIDECTOMY: central neck dissection (levels VI) if calcitonin elevated; "
            "  lateral neck dissection if lymph node metastases confirmed; "
            "POSTOP: levothyroxine replacement; calcium monitoring (transient hypocalcemia post-PTX); "
            "RET KINASE INHIBITORS: selpercatinib (LOXO-292) or pralsetinib for advanced/metastatic MTC; "
            "  vandetanib/cabozantinib: approved for progressive/metastatic MTC; "
            "ANNUAL SURVEILLANCE (gene-positive relatives): "
            "  calcitonin + CEA + 24h urine metanephrines + serum Ca + PTH; "
            "  neck ultrasound annually; "
            "PARATHYROID HPT (MEN2A): usually single adenoma (unlike MEN1) → single adenomectomy ± exploration; "
            "CASCADE FAMILY TESTING: genetic testing all first-degree relatives from birth"
        ),
        "key_ddx": (
            "MEN1: HPT multiglandular (MEN1) vs single adenoma (MEN2A); MEN1 has pNETs/pituitary — MEN2 does NOT; "
            "Sporadic MTC: no germline RET (15% of apparent sporadic MTC have germline RET — always test); "
            "VHL: pheo (usually without MTC); RCC + CNS hemangioblastoma — no MTC; "
            "SDHx (pheo/PGL): pheo + paraganglioma; no MTC; check SDHB/C/D; "
            "Sporadic pheo: 40% have hereditary cause (RET, VHL, SDHx, NF1, TMEM127, MAX); "
            "Calcitonin elevation WITHOUT MTC: Hashimoto's, renal failure, proton pump inhibitors (false positives)"
        ),
        "bone_disease": "pHPT in MEN2A (15-20%) → mild hypercalcemia; MTC skeletal metastases (late disease)",
        "mineral_disturbance": "HPT in MEN2A (hypercalcemia, elevated PTH); otherwise normal mineral metabolism; calcitonin elevated",
        "fgf23_status": "NORMAL (endocrine tumor syndrome, not phosphatonin pathway)",
        "severity_weights": [0.20, 0.30, 0.50],  # high MTC risk drives severity
    },

    # ── TCIRG1 — Autosomal Recessive Osteopetrosis (ARO / Malignant Infantile) ──
    {
        "gene": "TCIRG1",
        "protein": "V-ATPase Subunit a3 (ATP6i / TCIRG1 — T-Cell Immune Regulator 1)",
        "alias": (
            "TCIRG1; OMIM gene 604592; Autosomal Recessive Osteopetrosis ARO #259700; "
            "also OPTB1 (osteopetrosis type B1); "
            "11q13.2; 830 aa; ~97 kDa; AR; most common severe osteopetrosis (50-60% of ARO); "
            "TCIRG1 is the V-ATPase a3 subunit; expressed on osteoclast ruffled border membrane; "
            "pumps H+ into resorption lacuna → acidification required for bone mineral dissolution"
        ),
        "aa": "830 aa",
        "kDa": "~97 kDa",
        "locus": "11q13.2",
        "omim_gene": 604592,
        "omim_disease": 259700,
        "inheritance": "AR (biallelic); consanguinity major risk factor; 50-60% of infantile malignant osteopetrosis",
        "gene_class": (
            "TCIRG1 encodes the a3 isoform of the V0 domain of vacuolar H+-ATPase (V-ATPase); "
            "V-ATPase structure: V1 domain (cytoplasmic, ATP hydrolysis: subunits A-H) + V0 domain (membrane, proton translocation: a, c, c', d, e); "
            "TCIRG1/a3 (Atp6i): specifically expressed in osteoclasts on the ruffled border membrane (apical membrane facing bone); "
            "V-ATPase pumps H+ across ruffled border into sealed resorption lacuna → pH drops to ~4.5 → "
            "  mineral dissolution (hydroxyapatite dissolves at low pH); "
            "  cathepsin K then degrades organic matrix (collagen I); "
            "TCIRG1 LOF → osteoclasts CANNOT acidify lacuna → bone resorption FAILS; "
            "bone accumulates but is structurally WEAK (unmineralised/improperly remodelled matrix); "
            "Result: skeleton dense on X-ray but fractures easily (brittle-hard paradox); "
            "Marrow space obliterated → extramedullary haematopoiesis (liver/spleen) → hepatosplenomegaly; "
            "Cranial foramina compressed → optic canal stenosis → optic atrophy; "
            "  facial canal → facial palsy; auditory canal → sensorineural hearing loss; "
            "Other TCIRG1 roles: also expressed on dendritic cell phagosomes, synaptic vesicles (hence name T-cell immune regulator)"
        ),
        "phenotype": (
            "MALIGNANT INFANTILE OSTEOPETROSIS (ARO) — presents birth to 3 months: "
            "PANCYTOPENIA (anaemia, thrombocytopenia, leukopenia): marrow obliteration → hepatosplenomegaly (EMH); "
            "  anaemia from birth; transfusion-dependent; failure to thrive; "
            "CRANIAL NERVE PALSIES: optic atrophy (25-50%); facial nerve palsy; hearing loss; "
            "FRACTURES: despite radiologically dense bones (metaphyseal); "
            "DENTAL: failure of primary tooth eruption (foramina obstructed); "
            "HYPOCALCEMIA: 'hungry bone' — mineral trapped in unresorbed bone → serum Ca falls; "
            "  Ca supplementation required; "
            "HYPOPHOSPHATEMIA: secondary to impaired bone resorption releasing phosphate; "
            "RADIOLOGICAL: "
            "  'bone-within-bone' (endobone) appearance; "
            "  Erlenmeyer flask deformity of distal femur/proximal tibia; "
            "  'sandwich' vertebrae (dense endplates); "
            "  skull base sclerosis; "
            "METABOLIC: compensatory secondary HPT (elevated PTH); elevated ALP (bone formation continuing despite no resorption)"
        ),
        "hallmark": (
            "HSCT IS THE ONLY CURE: "
            "  restores monocyte-osteoclast lineage from donor → functional V-ATPase → bone resorption resumes; "
            "  TIMING CRITICAL: HSCT before age 3 months → best optic nerve + auditory outcomes; "
            "    delay → irreversible cranial nerve compression → permanent blindness/deafness; "
            "  HSCT does NOT reverse established neurological damage; "
            "CALCIUM-RESTRICTED DIET PERI-HSCT MANDATORY: "
            "  post-HSCT bone resorption suddenly resumes → massive calcium release → HYPERCALCEMIA; "
            "  restrict dietary Ca + hydration before and after HSCT; monitor Ca Q4h post-HSCT; "
            "GAMMA-INTERFERON (IFN-γ): increases superoxide production by residual osteoclasts → partial improvement; "
            "  used as bridge to HSCT or in patients ineligible for HSCT; "
            "CALCIUM SUPPLEMENTATION: for perioperative hypocalcemia (hungry bone); "
            "DISTINGUISHING FROM CLCN7 (ADO2): TCIRG1=AR infantile (pancytopenia) vs CLCN7=AD adult/childhood (no pancytopenia); "
            "CARBONIC ANHYDRASE II (CA2) DEFICIENCY: different gene; AR; renal tubular acidosis + osteopetrosis — milder; HSCT not indicated"
        ),
        "treatment_alert": (
            "HSCT: MUD (matched unrelated donor) acceptable if no HLA-matched sibling available; "
            "  pre-HSCT conditioning: myeloablative (RIC regimens studied); "
            "  timing: ideally < 3 months of age (before irreversible optic/auditory damage); "
            "PERI-HSCT CALCIUM: restrict Ca in diet + hydrate; IV bisphosphonate if severe post-HSCT hypercalcemia; "
            "  monitor Ca Q4h for first 72h post-HSCT; "
            "GAMMA-IFN: 1.5 µg/kg SC 3× per week; reduces infections + partial bone resorption improvement; "
            "RED CELL TRANSFUSION: packed RBCs (phenotypically matched); "
            "G-CSF: for severe neutropenia (leukocyte transfusions as bridge); "
            "OPHTHALMOLOGY: optic nerve decompression (surgical) for acute optic canal stenosis — emergency; "
            "DENTAL: antibiotic prophylaxis perioperatively; "
            "GENE THERAPY: in development (ex vivo lentiviral TCIRG1); trials ongoing; "
            "NO BISPHOSPHONATES: worsen bone resorption failure — absolutely contraindicated"
        ),
        "key_ddx": (
            "CLCN7 (ADO2): AD (autosomal dominant) not AR; childhood/adult onset; NO pancytopenia; HSCT NOT effective; "
            "CLCN7 AR (severe infantile): AR CLCN7 exists — phenotype intermediate; "
            "CA2 (carbonic anhydrase II) deficiency: AR; renal tubular acidosis + osteopetrosis + cerebral calcification; milder; "
            "TNFSF11 (RANKL) deficiency: AR; osteoclast differentiation failure (no RANKL signal); "
            "  DENOSUMAB exposure in mother during pregnancy reported as cause; "
            "OSTM1, SNX10, PLEKHM1: other rare AR osteopetrosis genes; "
            "Paediatric leukaemia: pancytopenia; bone marrow biopsy distinguishes (infiltration vs obliteration)"
        ),
        "bone_disease": "Severe marble bone disease; dense brittle bones; fractures; Erlenmeyer flask; bone-within-bone",
        "mineral_disturbance": "Hypocalcemia (hungry bone — trapped mineral); hypophosphatemia; elevated PTH (secondary HPT); elevated ALP",
        "fgf23_status": "NORMAL (osteoclast function disorder, not phosphatonin pathway)",
        "severity_weights": [0.10, 0.30, 0.60],  # predominantly severe infantile form
    },

    # ── CLCN7 — Autosomal Dominant Osteopetrosis Type 2 (ADO2 / Albers-Schönberg) ──
    {
        "gene": "CLCN7",
        "protein": "Chloride Voltage-Gated Channel 7 (CLC-7 / CLCN7)",
        "alias": (
            "CLCN7; OMIM gene 602727; Autosomal Dominant Osteopetrosis Type 2 ADO2 #166600 (Albers-Schönberg disease); "
            "also AR-CLCN7 osteopetrosis #259710 (intermediate severity); "
            "16p13.3; 803 aa; ~90 kDa; AD (dominant negative) for ADO2; AR for severe form; "
            "most common osteopetrosis overall (ADO2 ~1:20,000); "
            "CLC-7: Cl-/H+ exchanger on osteoclast ruffled border and late endosomes/lysosomes"
        ),
        "aa": "803 aa",
        "kDa": "~90 kDa",
        "locus": "16p13.3",
        "omim_gene": 602727,
        "omim_disease": 166600,
        "inheritance": "AD (dominant negative LOF) for ADO2; AR for intermediate/severe form; de novo common in ADO2",
        "gene_class": (
            "CLCN7 (CLC-7) is a member of the CLC (chloride channel) family; "
            "CLC-7 functions as a 2Cl-/H+ antiporter (exchanger, not a channel — unlike CLC name): "
            "  moves 2 Cl- outward + 1 H+ inward → acidifies late endosomes and lysosomes; "
            "On osteoclast ruffled border: CLC-7 + V-ATPase both required for lacunar acidification; "
            "  V-ATPase pumps H+ out → CLC-7 provides electrical shunting (Cl- entry) to maintain electroneutrality; "
            "  without CLC-7, V-ATPase voltage builds up and stalls → acidification incomplete; "
            "CLC-7 partners with OSTM1 (Ostm1 protein) as a β-subunit — OSTM1 mutations cause AR osteopetrosis with brain malformations; "
            "ADO2 DOMINANT NEGATIVE MECHANISM: "
            "  CLC-7 forms homodimers; one mutant subunit inactivates the dimer → "
            "  even 1 mutant allele causes significant impairment; "
            "  this explains AD inheritance; simple haploinsufficiency would give milder disease; "
            "AR form (biallelic CLCN7): more severe than ADO2 but usually less than TCIRG1 (intermediate); "
            "  some AR-CLCN7 cases have neuronal ceroid lipofuscinosis-like brain changes (CLC-7 also in neurons); "
            "OSTEOPETROSIS MECHANISM: CLC-7 LOF → lacunar acidification insufficient → bone resorption fails → "
            "  dense, brittle bones + cranial foramina narrowing (same endpoint as TCIRG1 but MILDER in ADO2)"
        ),
        "phenotype": (
            "ADO2 (Albers-Schönberg disease — most common form): "
            "Often diagnosed in childhood–adulthood (later than TCIRG1); "
            "Variable penetrance/expression — some asymptomatic; "
            "FRACTURES: 40-75% lifetime; femur + forearm most common (dense but brittle bone paradox); "
            "CRANIAL NERVE PALSIES: optic atrophy (5-10%); facial nerve palsy; hearing loss; "
            "  typically milder than TCIRG1 (no marrow failure to drive rapid progression); "
            "OSTEOMYELITIS: mandibular osteomyelitis (10-30%) — CHARACTERISTIC; "
            "  dental extractions or root canal → avascular necrosis + osteomyelitis (similar to bisphosphonate-related ONJ); "
            "SCOLIOSIS + KYPHOSIS: spinal involvement; "
            "ANAEMIA: mild (marrow partially maintained — NOT complete obliteration as in TCIRG1); "
            "RADIOLOGICAL: diffuse osteosclerosis; Erlenmeyer flask; 'sandwich' vertebrae; bone-within-bone; "
            "AR-CLCN7 FORM: intermediate severity; pancytopenia possible; earlier onset"
        ),
        "hallmark": (
            "HSCT NOT EFFECTIVE FOR ADO2 (AD FORM): "
            "  dominant-negative mechanism → donor osteoclasts carry 1 normal CLCN7 allele but will form dimers with "
            "  the patient's mutant protein (expressed in stroma?) — theoretical argument; "
            "  in practice, HSCT corrects marrow but bone density improvement is limited; "
            "  CONTRAST with TCIRG1 (AR) where HSCT provides fully normal monocytes → curative; "
            "  AR-CLCN7: HSCT may be beneficial (as for TCIRG1 AR); "
            "BISPHOSPHONATES ABSOLUTELY CONTRAINDICATED: "
            "  worsen already-impaired bone resorption → increased fracture risk and osteomyelitis; "
            "MANDIBULAR OSTEOMYELITIS: avoid dental extractions if possible; "
            "  if unavoidable: prophylactic antibiotics + antiseptic rinses; "
            "  treat osteomyelitis aggressively (long-course IV then oral antibiotics); "
            "OPHTHALMOLOGY: regular fundoscopy; optic nerve decompression if acute visual loss; "
            "CALCIUM + VITAMIN D: supplementation to optimise mineralisation of new bone; "
            "  (DO NOT give excess — worsen hypercalcemia if any resorption returns); "
            "INTERFERON-γ: limited benefit for ADO2 (works better for TCIRG1 AR)"
        ),
        "treatment_alert": (
            "NO BISPHOSPHONATES — CONTRAINDICATED (anti-resorptive → worsens impaired resorption already); "
            "NO HSCT for ADO2: generally not effective; AR-CLCN7 may be candidate (consult centre); "
            "FRACTURE MANAGEMENT: intramedullary nailing preferred (avoid plate fixation in sclerotic bone); "
            "DENTAL CARE: avoid extractions; panoramic X-ray annually (mandibular osteomyelitis surveillance); "
            "  antibiotic prophylaxis for any invasive dental procedure; "
            "OSTEOMYELITIS: prolonged antibiotics (6–12 weeks); surgical debridement if abscess; "
            "PHYSIOTHERAPY: weight-bearing exercises to maintain muscle strength; fall prevention; "
            "OPHTHALMOLOGY: annual review; emergency optic canal decompression if acute loss; "
            "AUDIOMETRY: biennial; hearing aids if needed; "
            "GENETIC COUNSELLING: AD; 50% risk per offspring; de novo mutations common; "
            "  test parents and siblings of index case"
        ),
        "key_ddx": (
            "TCIRG1 (ARO): AR (not AD); infantile onset; pancytopenia + hepatosplenomegaly; HSCT curative; "
            "CA2 (carbonic anhydrase II) deficiency: AR; renal tubular acidosis + cerebral calcification; "
            "  CA2 acts similarly to CLC-7 (supports lacunar acidification); "
            "RANKL (TNFSF11) deficiency: AR; osteoclast maturation failure; no osteoclasts at all; "
            "RANK (TNFRSF11A): AD GOF → dense bone but different mechanism; "
            "LRP5 GOF (high bone mass): very dense bones but NOT osteopetrosis — osteoblast activity increased, osteoclasts functional; "
            "Sclerostin deficiency (SOST): van Buchem disease / sclerosteosis — WNT pathway; "
            "  cortical thickening + entrapment neuropathy; NOT lacunar acidification defect"
        ),
        "bone_disease": "Dense brittle bones (osteopetrosis); Erlenmeyer flask; sandwich vertebrae; mandibular osteomyelitis; fractures",
        "mineral_disturbance": "Serum calcium NORMAL (ADO2 — no marrow failure); mild anaemia; ALP elevated",
        "fgf23_status": "NORMAL (osteoclast chloride channel disorder)",
        "severity_weights": [0.30, 0.40, 0.30],
    },
]


def _make_cohort(gene_data, seed):
    """Generate a 40-patient synthetic cohort for one gene."""
    rng = random.Random(seed)
    g = gene_data["gene"]

    pts = []
    sev_labels = ["mild", "moderate", "severe"]
    for i in range(40):
        sev = rng.choices(sev_labels, weights=gene_data["severity_weights"])[0]

        pt = {
            "patient_id": f"{g}-{seed:04d}-{i+1:02d}",
            "gene": g,
            "seed": seed,
            "severity": sev,
            "age_at_diagnosis_years": round(
                rng.uniform(0, 2) if g in ("TCIRG1",)
                else rng.uniform(0, 5) if g in ("CLCN7", "SLC34A3")
                else rng.uniform(0, 45), 1
            ),
            "sex": rng.choice(["M", "F"]),
            # Mineral disturbances
            "hypophosphatemia": rng.random() < (
                0.95 if g in ("PHEX", "FGF23", "SLC34A3") else
                0.70 if g == "TCIRG1" else  # hungry bone hypophos
                0.05
            ),
            "hypercalcemia": rng.random() < (
                0.90 if g in ("CASR", "MEN1") else
                0.05 if g == "TCIRG1" else  # post-HSCT
                0.02
            ),
            "hypocalcemia": rng.random() < (
                0.15 if g == "CASR" else  # ADH1 GOF subset
                0.75 if g == "TCIRG1" else  # hungry bone
                0.05
            ),
            # Bone disease
            "rickets_or_osteomalacia": rng.random() < (
                0.90 if g in ("PHEX", "FGF23", "SLC34A3") else 0.05
            ),
            "fractures": rng.random() < (
                0.55 if sev == "severe" else 0.30 if sev == "moderate" else 0.10
                if g in ("PHEX", "SLC34A3") else
                0.60 if g in ("TCIRG1", "CLCN7") else
                0.20
            ),
            # Renal complications
            "nephrocalcinosis_or_stones": rng.random() < (
                0.35 if g == "SLC34A3" else
                0.15 if g == "CASR" else  # ADH1 GOF
                0.20 if g == "MEN1" else
                0.05
            ),
            "hypercalciuria": rng.random() < (
                0.95 if g == "SLC34A3" else
                0.10 if g == "CASR" else
                0.05
            ),
            # Endocrine tumors
            "parathyroid_disease": rng.random() < (
                0.90 if g == "MEN1" else
                0.15 if g == "RET" else  # MEN2A HPT
                0.90 if g == "CASR" else  # FHH1 is parathyroid-setpoint disorder
                0.03
            ),
            "pancreatic_net": rng.random() < (0.65 if g == "MEN1" else 0.01),
            "pituitary_adenoma": rng.random() < (0.38 if g == "MEN1" else 0.01),
            "medullary_thyroid_cancer": rng.random() < (0.95 if g == "RET" else 0.01),
            "pheochromocytoma": rng.random() < (0.45 if g == "RET" else 0.01),
            # Haematologic
            "pancytopenia": rng.random() < (0.85 if g == "TCIRG1" else 0.08 if g == "CLCN7" else 0.01),
            "hepatosplenomegaly": rng.random() < (0.80 if g == "TCIRG1" else 0.05),
            # Cranial nerve
            "cranial_nerve_palsy": rng.random() < (
                0.40 if g == "TCIRG1" else
                0.15 if g == "CLCN7" else
                0.03
            ),
            "optic_atrophy": rng.random() < (0.35 if g == "TCIRG1" else 0.08 if g == "CLCN7" else 0.01),
            "hearing_loss": rng.random() < (
                0.28 if g == "PHEX" else
                0.30 if g == "TCIRG1" else
                0.20 if g == "CLCN7" else
                0.05
            ),
            # Treatment
            "on_burosumab": rng.random() < (0.70 if g == "PHEX" else 0.45 if g == "FGF23" else 0.01),
            "on_phosphate_calcitriol": rng.random() < (
                0.25 if g == "PHEX" else  # older patients before burosumab era
                0.50 if g == "FGF23" else
                0.80 if g == "SLC34A3" else
                0.02
            ),
            "prophylactic_thyroidectomy_done": rng.random() < (0.75 if g == "RET" else 0.01),
            "hsct_performed": rng.random() < (0.60 if g == "TCIRG1" else 0.05 if g == "CLCN7" else 0.0),
            "parathyroidectomy_done": rng.random() < (
                0.55 if g == "MEN1" else  # pHPT surgery
                0.10 if g == "CASR" else  # NSHPT emergency
                0.0
            ),
        }
        pts.append(pt)
    return pts


# Pre-build cohorts at import time
_ALL_COHORTS = {}
for _idx, _gd in enumerate(BONE_MINERAL_GENES):
    _seed = SEED_BASE + _idx
    _ALL_COHORTS[_gd["gene"]] = _make_cohort(_gd, _seed)


def _pct(pts, key):
    return round(100 * sum(1 for p in pts if p.get(key)) / max(len(pts), 1))


def get_overview():
    all_pts = [p for pts in _ALL_COHORTS.values() for p in pts]
    genes = [g["gene"] for g in BONE_MINERAL_GENES]
    return {
        "atlas_name": "Bone and Mineral Metabolism Disorders Atlas",
        "atlas_subtitle": (
            "Complete 8-Gene Hereditary Bone & Mineral Metabolism Disorders Reference — "
            "PHEX · FGF23 · SLC34A3 · CASR · MEN1 · RET · TCIRG1 · CLCN7"
        ),
        "n_genes": 8,
        "n_patients": len(all_pts),
        "seeds": "1262–1269",
        "genes": genes,
        "description": (
            "This atlas covers eight primary hereditary bone and mineral metabolism disorders in clinical genetics. "
            "Hypophosphatemic rickets: PHEX (XLH; X-linked dominant; burosumab standard-of-care — phosphate alone CI), "
            "FGF23 (ADHR; autosomal dominant GOF; fluctuating — iron-sensitive), and "
            "SLC34A3 (HHRH; autosomal recessive; FGF23 NORMAL/LOW; 1,25-D ELEVATED; calcitriol CI; hypercalciuria). "
            "Calcium setpoint disorder: CASR (FHH1 LOF — benign hypercalcemia — DO NOT parathyroidectomy; "
            "ADH1 GOF — hypocalcemia — calcitriol causes nephrocalcinosis; NSHPT biallelic — urgent PTX). "
            "Multiple endocrine neoplasia: MEN1 (menin; pHPT 90-95% multiglandular → subtotal PTX; "
            "pNET 60-70% gastrinoma/insulinoma; pituitary 30-40%) and "
            "RET (MEN2A/2B; MTC prophylactic thyroidectomy timing by codon risk category; "
            "PHEO before thyroid surgery mandatory — alpha-blockade first). "
            "Osteoclast defects: TCIRG1 (ARO malignant infantile osteopetrosis; "
            "HSCT curative — timing before 3 months critical; BISPHOSPHONATES ABSOLUTELY CI) and "
            "CLCN7 (ADO2 Albers-Schönberg; dominant negative; HSCT NOT effective for AD form; "
            "mandibular osteomyelitis; BISPHOSPHONATES ABSOLUTELY CI)."
        ),
        "aggregate_clinical": {
            "hypophosphatemia_pct": _pct(all_pts, "hypophosphatemia"),
            "hypercalcemia_pct": _pct(all_pts, "hypercalcemia"),
            "hypocalcemia_pct": _pct(all_pts, "hypocalcemia"),
            "rickets_pct": _pct(all_pts, "rickets_or_osteomalacia"),
            "fractures_pct": _pct(all_pts, "fractures"),
            "nephrocalcinosis_pct": _pct(all_pts, "nephrocalcinosis_or_stones"),
            "cranial_nerve_palsy_pct": _pct(all_pts, "cranial_nerve_palsy"),
            "pancytopenia_pct": _pct(all_pts, "pancytopenia"),
            "medullary_thyroid_cancer_pct": _pct(all_pts, "medullary_thyroid_cancer"),
            "pheochromocytoma_pct": _pct(all_pts, "pheochromocytoma"),
            "pancreatic_net_pct": _pct(all_pts, "pancreatic_net"),
            "on_burosumab_pct": _pct(all_pts, "on_burosumab"),
            "hsct_performed_pct": _pct(all_pts, "hsct_performed"),
            "prophylactic_thyroidectomy_pct": _pct(all_pts, "prophylactic_thyroidectomy_done"),
        },
        "drug_alerts": [
            {
                "title": "PHEX (XLH) — PHOSPHATE WITHOUT CALCITRIOL ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "Oral phosphate alone in XLH stimulates PTH secretion and drives secondary → tertiary "
                    "hyperparathyroidism → nephrocalcinosis and renal failure. Calcitriol must ALWAYS be "
                    "co-prescribed with phosphate. Preferred: switch to burosumab (anti-FGF23 MAb) which "
                    "avoids this complication entirely. Do NOT combine burosumab with oral phosphate/calcitriol."
                ),
            },
            {
                "title": "CASR (FHH1) — DO NOT PARATHYROIDECTOMY (benign hypercalcemia — UCaR <0.01 DIAGNOSTIC)",
                "body": (
                    "FHH1 (CASR LOF) causes lifelong hypercalcemia that is entirely benign — no end-organ damage. "
                    "Performing parathyroidectomy causes permanent hypoparathyroidism — a far worse condition. "
                    "The 24h urinary calcium-to-creatinine ratio (UCaR) <0.01 diagnoses FHH1 and rules out primary HPT. "
                    "CASR gene testing is MANDATORY before any parathyroidectomy decision in normo-calciuric hypercalcemia."
                ),
            },
            {
                "title": "RET (MEN2) — PHEO BEFORE THYROID SURGERY MANDATORY (alpha-blockade first)",
                "body": (
                    "All MEN2 patients must be screened for phaeochromocytoma before thyroidectomy. "
                    "Unblocked pheo during thyroid surgery → catecholamine storm → hypertensive crisis → stroke/death. "
                    "Protocol: 24h urine metanephrines or plasma fractionated metanephrines → if positive, "
                    "alpha-blockade (phenoxybenzamine) for 10-14 days → adrenalectomy → THEN thyroidectomy."
                ),
            },
            {
                "title": "TCIRG1/CLCN7 (Osteopetrosis) — BISPHOSPHONATES ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "Both TCIRG1 (ARO) and CLCN7 (ADO2) patients cannot resorb bone normally. "
                    "Bisphosphonates further suppress osteoclast activity → catastrophic worsening of bone accumulation, "
                    "fractures, and osteomyelitis risk. Bisphosphonates are ABSOLUTELY CONTRAINDICATED in all "
                    "osteopetrosis forms. This is a common and potentially lethal prescribing error."
                ),
            },
            {
                "title": "SLC34A3 (HHRH) — CALCITRIOL CONTRAINDICATED (1,25-D already elevated)",
                "body": (
                    "In HHRH, 1,25(OH)2D3 is already appropriately elevated (renal hypophosphatemia → CYP27B1 upregulated). "
                    "Adding calcitriol or vitamin D analogs → severe hypercalciuria → nephrocalcinosis and renal failure. "
                    "Burosumab is NOT indicated (FGF23 not elevated). Treat with phosphate alone + low-calcium diet + "
                    "thiazide diuretic if stones are a problem."
                ),
            },
            {
                "title": "TCIRG1 (ARO) — HSCT MUST BE DONE BEFORE AGE 3 MONTHS (optic nerve irreversible damage)",
                "body": (
                    "In malignant infantile osteopetrosis (TCIRG1 ARO), optic canal compression causes progressive optic "
                    "atrophy beginning in the first weeks of life. HSCT performed before 3 months restores osteoclast "
                    "function and decompresses foramina before irreversible nerve damage. Each month of delay increases "
                    "risk of permanent blindness. Do NOT wait for growth failure or severe anaemia — act on molecular "
                    "diagnosis as soon as confirmed."
                ),
            },
        ],
        "clinical_pearls": [
            "PHEX (XLH): FGF23 ELEVATED + 1,25-D inappropriately LOW + hypophosphatemia = XLH; dental abscesses pathognomonic.",
            "FGF23 (ADHR): FLUCTUATING hypophosphatemia + iron-sensitive = ADHR; replete iron first before escalating treatment.",
            "SLC34A3 (HHRH): FGF23 NORMAL/LOW + 1,25-D ELEVATED + HYPERCALCIURIA = HHRH; calcitriol CI; phosphate only.",
            "CASR (FHH1): hypercalcemia + UCaR <0.01 + family history = FHH1; benign — DO NOT parathyroidectomy.",
            "CASR (ADH1): GOF hypocalcemia + suppressed PTH + calcitriol → nephrocalcinosis; target low-normal Ca only.",
            "MEN1: pHPT multiglandular (4-gland) + pNETs + pituitary = MEN1; annual EUS from age 20; subtotal PTX (3.5-gland).",
            "RET: MTC + pheo + (±HPT) = MEN2; always test pheo BEFORE thyroid surgery; codon drives prophylactic thyroidectomy timing.",
            "TCIRG1: pancytopenia + hepatosplenomegaly + dense bones in infant = ARO; HSCT before 3 months; Ca-restricted peri-HSCT.",
            "CLCN7: dominant osteopetrosis + fractures + mandibular osteomyelitis = ADO2; HSCT NOT effective for AD form.",
            "ALL OSTEOPETROSIS: BISPHOSPHONATES ABSOLUTELY CONTRAINDICATED — worsen bone resorption failure.",
        ],
    }


def get_breakdown():
    out = {}
    for gd in BONE_MINERAL_GENES:
        pts = _ALL_COHORTS[gd["gene"]]
        out[gd["gene"]] = {
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "gene_class": gd["gene_class"],
            "phenotype": gd["phenotype"],
            "hallmark": gd["hallmark"],
            "treatment_alert": gd["treatment_alert"],
            "key_ddx": gd["key_ddx"],
            "bone_disease": gd["bone_disease"],
            "mineral_disturbance": gd["mineral_disturbance"],
            "fgf23_status": gd["fgf23_status"],
            "cohort_n": len(pts),
            "stats": {
                "hypophosphatemia_pct": _pct(pts, "hypophosphatemia"),
                "hypercalcemia_pct": _pct(pts, "hypercalcemia"),
                "hypocalcemia_pct": _pct(pts, "hypocalcemia"),
                "rickets_pct": _pct(pts, "rickets_or_osteomalacia"),
                "fractures_pct": _pct(pts, "fractures"),
                "nephrocalcinosis_pct": _pct(pts, "nephrocalcinosis_or_stones"),
                "hypercalciuria_pct": _pct(pts, "hypercalciuria"),
                "cranial_nerve_palsy_pct": _pct(pts, "cranial_nerve_palsy"),
                "optic_atrophy_pct": _pct(pts, "optic_atrophy"),
                "hearing_loss_pct": _pct(pts, "hearing_loss"),
                "pancytopenia_pct": _pct(pts, "pancytopenia"),
                "hepatosplenomegaly_pct": _pct(pts, "hepatosplenomegaly"),
                "parathyroid_disease_pct": _pct(pts, "parathyroid_disease"),
                "pancreatic_net_pct": _pct(pts, "pancreatic_net"),
                "pituitary_adenoma_pct": _pct(pts, "pituitary_adenoma"),
                "medullary_thyroid_cancer_pct": _pct(pts, "medullary_thyroid_cancer"),
                "pheochromocytoma_pct": _pct(pts, "pheochromocytoma"),
                "on_burosumab_pct": _pct(pts, "on_burosumab"),
                "on_phosphate_calcitriol_pct": _pct(pts, "on_phosphate_calcitriol"),
                "prophylactic_thyroidectomy_pct": _pct(pts, "prophylactic_thyroidectomy_done"),
                "hsct_performed_pct": _pct(pts, "hsct_performed"),
                "parathyroidectomy_pct": _pct(pts, "parathyroidectomy_done"),
                "severity_severe_pct": round(100 * sum(1 for p in pts if p["severity"] == "severe") / 40),
                "severity_moderate_pct": round(100 * sum(1 for p in pts if p["severity"] == "moderate") / 40),
            },
        }
    return out


def get_definitions():
    return {
        "terms": [
            {
                "term": "X-linked Hypophosphatemia (PHEX/XLH) — FGF23 Elevated + Phosphate WITHOUT Calcitriol CI",
                "definition": (
                    "XLH is the most common hereditary rickets (1:20,000). PHEX LOF → elevated circulating intact FGF23 → "
                    "proximal tubule NaPi-IIa/IIc downregulation → phosphaturia → hypophosphatemia. Simultaneously, "
                    "FGF23 suppresses 1α-hydroxylase → 1,25(OH)2D3 inappropriately LOW (fails to rise despite hypophosphatemia). "
                    "Serum Ca and PTH NORMAL. Key: dental abscesses (spontaneous periapical abscesses — no caries) are pathognomonic. "
                    "Treatment: burosumab (anti-FGF23 MAb, 0.8 mg/kg Q2W s.c.) now standard for children. "
                    "CRITICAL: phosphate alone CI (secondary HPT → nephrocalcinosis); always combine with calcitriol if burosumab unavailable. "
                    "DO NOT combine burosumab with phosphate/calcitriol."
                ),
            },
            {
                "term": "Autosomal Dominant Hypophosphatemia (FGF23/ADHR) — Fluctuating, Iron-Sensitive",
                "definition": (
                    "ADHR: GOF mutations at FGF23 cleavage site (Arg176/179) → FGF23 resistant to proteolysis → intact FGF23 elevated. "
                    "Biochemistry identical to XLH when active. KEY DISTINGUISHING FEATURE: fluctuating course — may remit, "
                    "especially when iron-replete. Iron deficiency suppresses FGF23 cleavage → phenotype worsens; "
                    "iron repletion → FGF23 cleaved → phosphate normalises. "
                    "Check ferritin + transferrin sat in all ADHR patients; replete iron before escalating treatment. "
                    "Burosumab effective. Phosphate + calcitriol second-line (same rules as XLH)."
                ),
            },
            {
                "term": "Hereditary Hypophosphatemic Rickets with Hypercalciuria (SLC34A3/HHRH) — FGF23 NORMAL/LOW, 1,25-D ELEVATED",
                "definition": (
                    "HHRH: SLC34A3 (NaPi-IIc) LOF → primary renal phosphate wasting → hypophosphatemia → "
                    "CYP27B1 appropriately upregulated → 1,25(OH)2D3 ELEVATED. Elevated 1,25-D → intestinal Ca absorption + "
                    "bone resorption → HYPERCALCIURIA → nephrolithiasis/nephrocalcinosis. FGF23 NORMAL or LOW "
                    "(hypophosphatemia would suppress FGF23 production — contrast with XLH where FGF23 is the driver). "
                    "CRITICAL: calcitriol CI (1,25-D already high → worsens hypercalciuria). Burosumab NOT indicated (FGF23 not elevated). "
                    "Treat: phosphate only + low-calcium diet ± thiazide diuretic for hypercalciuria. Renal US monitoring essential."
                ),
            },
            {
                "term": "CASR Disorders — FHH1 (LOF benign) / ADH1 (GOF hypocalcemia) / NSHPT (biallelic LOF severe)",
                "definition": (
                    "CaSR is a GPCR that senses extracellular Ca²⁺ and regulates PTH. "
                    "FHH1 (LOF, AD): setpoint shifted right → hypercalcemia with LOW urine calcium (UCaR <0.01). "
                    "Benign — NO end-organ damage. DO NOT parathyroidectomy (permanent hypoparathyroidism = worse). "
                    "UCaR <0.01 diagnoses FHH1; compare primary HPT (UCaR >0.02). "
                    "ADH1 (GOF, AD): setpoint shifted left → hypocalcemia + suppressed PTH. "
                    "Calcitriol therapy → nephrocalcinosis (urine Ca rises); target low-normal Ca only; "
                    "thiazide + low-Ca diet help; recombinant PTH experimental. "
                    "NSHPT (biallelic LOF): severe neonatal hypercalcemia → URGENT total parathyroidectomy or cinacalcet bridge."
                ),
            },
            {
                "term": "Multiple Endocrine Neoplasia Type 1 (MEN1/Menin) — Multiglandular HPT + pNETs + Pituitary",
                "definition": (
                    "MEN1 encodes Menin — a tumor suppressor scaffold for MLL/SET1 histone methyltransferase. "
                    "2-hit; >600 variants; no phenotype-genotype correlation. "
                    "TRIAD: pHPT (90-95%, MULTIGLANDULAR 4-gland disease) + pNETs (60-70%: gastrinoma → ZE syndrome, insulinoma, non-functional) "
                    "+ pituitary adenoma (30-40%: prolactinoma most common). "
                    "SURGERY for HPT: subtotal 3.5-gland parathyroidectomy (NOT single-gland — recurs rapidly) OR total + autotransplantation. "
                    "Annual surveillance: Ca, PTH, gastrin, glucose, insulin, chromogranin A, prolactin, IGF-1, EUS from age 20. "
                    "Thymic carcinoid (5-10%): lethal if metastatic; annual chest CT; smoking greatly increases risk."
                ),
            },
            {
                "term": "Multiple Endocrine Neoplasia Type 2 (RET) — Prophylactic Thyroidectomy Timing by Codon",
                "definition": (
                    "RET GOF mutations → constitutive tyrosine kinase activation → MTC + pheo + (± HPT in MEN2A). "
                    "CODON-BASED RISK (ATA 2015): "
                    "Category D (highest — M918T, C634F/Y, A883F): prophylactic thyroidectomy within 6 months of life; "
                    "Category C (high — C618/620/630, C634G/R/S/W): by age 5; "
                    "Category B (intermediate — E768D, V804M/L, S891A etc.): by age 5 or when calcitonin rises. "
                    "MEN2B (M918T): earliest and most aggressive — thyroidectomy in first 6 months; mucosal neuromas from birth. "
                    "PHEO BEFORE THYROID SURGERY: mandatory alpha-blockade (phenoxybenzamine) before any neck surgery. "
                    "Metastatic MTC: selpercatinib or pralsetinib (selective RET inhibitors) now preferred."
                ),
            },
            {
                "term": "Autosomal Recessive Osteopetrosis (TCIRG1) — HSCT Before 3 Months; Bisphosphonates Absolutely CI",
                "definition": (
                    "TCIRG1 encodes V-ATPase a3 subunit on osteoclast ruffled border. LOF → osteoclasts cannot acidify "
                    "resorption lacuna → bone accumulates → marrow obliteration → pancytopenia + EMH (hepatosplenomegaly); "
                    "cranial foramina compressed → optic atrophy + facial palsy + hearing loss; dense brittle bones + fractures. "
                    "HSCT IS THE ONLY CURE: restores monocyte-osteoclast lineage → bone resorption resumes. "
                    "Timing critical: before age 3 months for best optic nerve outcomes — irreversible nerve damage occurs rapidly. "
                    "Peri-HSCT: restrict dietary calcium (sudden bone resorption post-HSCT → hypercalcemia). "
                    "BISPHOSPHONATES ABSOLUTELY CONTRAINDICATED. Gamma-IFN: bridge to HSCT."
                ),
            },
            {
                "term": "Autosomal Dominant Osteopetrosis Type 2 (CLCN7/ADO2 / Albers-Schönberg) — HSCT NOT Effective for AD Form",
                "definition": (
                    "CLCN7 (CLC-7) is a 2Cl-/H+ antiporter on osteoclast ruffled border (works with V-ATPase for lacunar acidification). "
                    "ADO2: dominant-negative LOF → most common osteopetrosis (~1:20,000); childhood/adult onset; "
                    "dense bones, fractures, cranial nerve palsies, mandibular osteomyelitis. "
                    "HSCT NOT EFFECTIVE for dominant form (dominant-negative mechanism persists in donor osteoclasts). "
                    "Contrast: AR-CLCN7 (biallelic) — more severe; HSCT may be beneficial. "
                    "Management: fracture prevention, dental hygiene (avoid extractions — osteomyelitis risk), "
                    "ophthalmology (optic canal decompression if acute), hearing aids. "
                    "BISPHOSPHONATES ABSOLUTELY CONTRAINDICATED. Calcium + vitamin D supplementation only."
                ),
            },
            {
                "term": "FGF23 Phosphatonin Pathway — XLH vs ADHR vs HHRH vs Oncogenic Osteomalacia",
                "definition": (
                    "FGF23 is produced by osteocytes → inhibits NaPi-IIa/IIc on proximal tubule → phosphaturia; "
                    "also inhibits CYP27B1 → suppresses 1,25(OH)2D3. "
                    "Elevated FGF23: XLH (PHEX LOF), ADHR (FGF23 GOF — resistant to cleavage), oncogenic osteomalacia (tumor-secreted). "
                    "All share: hypophosphatemia + phosphaturia + 1,25-D inappropriately LOW + Ca NORMAL + PTH NORMAL. "
                    "NORMAL/LOW FGF23 with hypophosphatemia: HHRH (SLC34A3), Fanconi syndrome, nutritional rickets. "
                    "In HHRH, 1,25-D is ELEVATED (FGF23 not elevated → CYP27B1 works normally → appropriate 1,25-D rise). "
                    "FGF23 assay: use INTACT FGF23 assay (Kainos) — C-terminal assay measures inactive fragments too."
                ),
            },
            {
                "term": "Urinary Calcium-to-Creatinine Ratio (UCaR) — Diagnosing FHH1 vs Primary HPT",
                "definition": (
                    "UCaR = 24h urine Ca (mmol or mg) / 24h urine creatinine (mmol or mg). "
                    "In primary HPT: UCaR typically >0.01–0.02 (hypercalciuria — PTH-driven renal Ca excretion). "
                    "In FHH1 (CASR LOF): UCaR <0.01 (DIAGNOSTIC) — kidneys retain Ca due to defective renal CaSR signalling. "
                    "Pitfall: vitamin D deficiency can lower urine Ca in PHPT → mimics FHH1 — correct Vit D first. "
                    "Pitfall: thiazide use lowers urine Ca — stop 1 month before UCaR measurement. "
                    "If UCaR <0.01 + family history of hypercalcemia: do CASR genetic testing before any parathyroid surgery. "
                    "Ca/Cr ratio can also be measured spot urine (less reliable — 24h preferred)."
                ),
            },
            {
                "term": "Osteopetrosis — Dense Bone Paradox and Radiological Hallmarks",
                "definition": (
                    "Osteopetrosis: impaired osteoclast resorption → bone accumulates but is POORLY REMODELLED → "
                    "'marble bone' — dense on X-ray but structurally BRITTLE (improperly organised Haversian systems). "
                    "X-ray hallmarks: "
                    "(1) Erlenmeyer flask deformity — distal femur/proximal tibia flared (metaphysis not shaped by resorption); "
                    "(2) Bone-within-bone (endobone) — growth arrest lines visible within medullary canal; "
                    "(3) 'Sandwich' vertebrae — dense endplates (H-vertebrae) around normal central body; "
                    "(4) Skull base sclerosis — foramina narrowed → cranial nerve entrapment. "
                    "ARO (TCIRG1): ALL features + marrow obliteration (pancytopenia). "
                    "ADO2 (CLCN7): milder but same X-ray pattern; no pancytopenia usually."
                ),
            },
            {
                "term": "MEN1 vs MEN2 — Key Differentiating Features",
                "definition": (
                    "MEN1: MEN1 gene (11q13.1); AD LOF; TRIAD = HPT (multiglandular) + pNETs (gastrinoma/insulinoma) + pituitary adenoma; "
                    "NO MTC, NO pheo in MEN1 proper. "
                    "MEN2A: RET gene (10q11.21); AD GOF; TRIAD = MTC (95-100%) + pheo (40-50%) + HPT (15-20% — single adenoma not multiglandular). "
                    "MEN2B: RET M918T most common; MTC (most aggressive) + pheo + MARFANOID HABITUS + MUCOSAL NEUROMAS — NO HPT. "
                    "Key DDx features: "
                    "pNETs → MEN1 not MEN2; "
                    "MTC → MEN2 not MEN1; "
                    "Mucosal neuromas + marfanoid → MEN2B; "
                    "Multiglandular HPT (4-gland) → MEN1; single adenoma → MEN2A or sporadic; "
                    "Gastrin elevated (ZE syndrome) → MEN1 gastrinoma."
                ),
            },
        ]
    }
