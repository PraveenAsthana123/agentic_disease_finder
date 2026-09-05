#!/usr/bin/env python3
"""Autoinflammatory-Atlas — Complete 8-Gene Hereditary Autoinflammatory Disorder Atlas
MEFV     (Familial Mediterranean Fever; ~781 aa; 16p13.3; pyrin/marenostrin;
          AR (most) or AD (some p.E148Q carriers); colchicine first-line — amyloid PREVENTED;
          anakinra/canakinumab for colchicine-resistant FMF) ·
NLRP3    (Cryopyrin-associated periodic syndromes — CAPS spectrum; ~1036 aa; 1q44;
          cryopyrin/NALP3; AD; FCAS → MWS → NOMID/CINCA;
          canakinumab FDA 2016 — ALL three CAPS subtypes; live vaccines ABSOLUTELY CI) ·
TNFRSF1A (TNF receptor-associated periodic syndrome — TRAPS; ~455 aa; 12p13.31;
          TNFR1; AD; attacks >5 days + migratory myalgia + periorbital oedema;
          corticosteroids WORSEN frequency long-term; canakinumab preferred) ·
MVK      (Mevalonate kinase deficiency — MKD / HIDS; ~396 aa; 12q24.11;
          mevalonate kinase; AR; hyperimmunoglobulinaemia D (historical);
          canakinumab FDA 2021 — MKD; geranyl/geranylgeranyl pyrophosphate depletion) ·
IL1RN    (Deficiency of IL-1 receptor antagonist — DIRA; ~177 aa; 2q14.1;
          IL-1Ra; AR; neonatal sterile osteomyelitis; anakinra ESSENTIALLY CURATIVE;
          without treatment — potentially fatal within months) ·
CECR1    (Deficiency of ADA2 — DADA2; ~511 aa; 22q11.1;
          adenosine deaminase 2; AR; childhood stroke (lacunar); splenomegaly;
          TNF inhibitors PREVENT STROKE; IVIG ineffective; HSCT curative) ·
NOD2     (Blau syndrome / early-onset sarcoidosis; ~1040 aa; 16q12.1;
          NOD2/CARD15; AD; classic triad: granulomatous arthritis + uveitis + skin rash;
          R334W most common; NOD2 controls NF-κB innate response) ·
PSMB8    (Proteasome-associated autoinflammatory syndromes — PRAAS; ~277 aa; 6p21.32;
          beta5i immunoproteasome; AR; Nakajo-Nishimura syndrome;
          lipodystrophy + panniculitis; JAK inhibitors: baricitinib/tofacitinib)
320-patient aggregate cohort (8 × 40, seeds 1174–1181)
"""

import random

SEED_BASE = 1174

AUTOINFLAMMATORY_GENES = [
    # ── MEFV — Familial Mediterranean Fever ─────────────────────────────────
    {
        "gene": "MEFV",
        "protein": "Pyrin (Marenostrin)",
        "alias": (
            "MEFV; OMIM gene 608107; 16p13.3; 781 aa; FMF OMIM #249100; "
            "AR (homozygous or compound het, most common) or AD (p.E148Q low-penetrance); "
            "prevalence: 1 in 200-500 Eastern Mediterranean (Armenian, Sephardic Jewish, "
            "Turkish, Arab); ~100,000 patients worldwide; "
            "most common monogenic autoinflammatory disorder"
        ),
        "aa": "781 aa",
        "kDa": "~86 kDa",
        "gene_class": (
            "Pyrin/marenostrin — TRIM superfamily; central regulator of inflammasome activation; "
            "contains TRIM domain (N-term), B-box zinc finger, coiled-coil, B30.2/SPRY (C-term); "
            "normal function: SUPPRESSES NLRP3/caspase-1 via 14-3-3 binding (phosphoserine); "
            "pathogenic variants LOF → pyrin inflammasome spontaneously activates → "
            "IL-1β and IL-18 overproduction → periodic sterile peritonitis/serositis; "
            "gene locus 16p13.3; constitutively expressed in myeloid cells (PMNs, monocytes)"
        ),
        "locus": "16p13.3",
        "omim_gene": 608107,
        "omim_disease": 249100,
        "phenotype": (
            "FMF: Episodic sterile febrile attacks (38-40°C) lasting 1-3 days; "
            "peritonitis (93% — acute abdomen; can mimic appendicitis requiring laparotomy); "
            "pleuritis (30%); arthritis (50%); erysipelas-like erythema (5-20%); "
            "pericarditis (1%); scrotal pain (males); "
            "attack frequency: weekly to every few months; "
            "triggers: minor infection, stress, physical exertion, menstruation; "
            "AA amyloidosis (kidneys) — #1 long-term complication — PREVENTED by colchicine; "
            "patients asymptomatic between attacks (unlike TRAPS which may have residual symptoms)"
        ),
        "hallmark": (
            "PERITONITIS (ACUTE ABDOMEN) IN FMF — mimics appendicitis; attacks 1-3 days; "
            "COLCHICINE prevents both attacks AND amyloid A amyloidosis — "
            "start at diagnosis, lifelong; M694V/M694V homozygote = MOST SEVERE phenotype; "
            "E148Q = low penetrance, often insufficient for diagnosis alone"
        ),
        "treatment_alert": (
            "COLCHICINE — First-line, lifelong; dose 1-2 mg/day; "
            "reduces attack frequency ≥65% and PREVENTS AA amyloidosis (near-complete); "
            "NSAIDs for acute pain only — do NOT replace colchicine; "
            "if colchicine-resistant/intolerant → anakinra (IL-1Ra) or canakinumab (IL-1β); "
            "COLCHICINE SAFE IN PREGNANCY — do NOT stop; stopping risks amyloidosis; "
            "CYCLOSPORINE with colchicine → MYOPATHY + RHABDOMYOLYSIS — avoid combination; "
            "statins + colchicine → MYOPATHY risk — monitor CK"
        ),
        "key_ddx": (
            "TRAPS: attacks >5 days (FMF 1-3 days); migratory myalgia; periorbital oedema; "
            "high ESR between attacks (FMF normalises between attacks); "
            "MKD/HIDS: elevated IgD (historical); lymphadenopathy prominent; "
            "Appendicitis: cannot distinguish acutely — laparo if in doubt first attack; "
            "Periodic fever aphthous pharyngitis adenitis (PFAPA): tonsillar exudates; "
            "corticosteroid-responsive (unlike FMF); "
            "Acute porphyria: urinary PBG; abdominal + neurological; AIP vs FMF"
        ),
        "gfr_pattern": (
            "Normal until AA amyloidosis develops (nephropathy) → proteinuria → CKD → ESRD; "
            "proteinuria in FMF = amyloidosis until proven otherwise; "
            "renal biopsy: apple-green birefringence Congo red = diagnostic amyloid"
        ),
        "proteinuria_pattern": (
            "Proteinuria = AA amyloid deposition in glomeruli; "
            "nephrotic syndrome → ESRD; "
            "colchicine prevents and may partially reverse early amyloid"
        ),
        "primary_complication": (
            "AA amyloidosis (kidneys > spleen > liver); ESRD is the primary mortality risk "
            "in non-treated or colchicine-non-adherent FMF; "
            "fully preventable with colchicine"
        ),
        "disease": (
            "MEFV encodes pyrin (marenostrin), a 781 aa TRIM superfamily protein expressed "
            "predominantly in myeloid cells. "
            "NORMAL FUNCTION: Pyrin acts as a PATTERN-RECOGNITION RECEPTOR for Rho GTPase "
            "modification by bacterial toxins (e.g., Clostridioides difficile TcdB, Burkholderia "
            "VopS). Under basal conditions, pyrin is INHIBITED by 14-3-3 proteins binding "
            "phosphoserine residues (Ser208 and Ser242, phosphorylated by PKN1/2). This keeps "
            "pyrin from activating its ASC-dependent inflammasome. "
            "PATHOMECHANISM: FMF variants cluster in the C-terminal B30.2/SPRY domain (M694V, "
            "M694I, M680I, V726A) and disrupt 14-3-3 binding → constitutive/lowered threshold "
            "pyrin inflammasome activation → caspase-1 cleavage → IL-1β and IL-18 release → "
            "STERILE PERITONITIS/SEROSITIS. The peritoneal cavity becomes flooded with neutrophils "
            "(chemotactic gradient) producing the 'acute abdomen' of FMF. "
            "GENOTYPE-PHENOTYPE: M694V/M694V (homozygous) — most severe; highest amyloid risk; "
            "M680I/M694V — severe; V726A/M694V — moderate; E148Q — low penetrance, often "
            "insufficient for FMF diagnosis alone, requires supportive clinical criteria; "
            "M694V allele frequency: 20-50% in Armenian, Sephardic Jewish populations. "
            "CLINICAL ATTACKS: Peritonitis (93% lifetime): sudden-onset colicky abdominal pain "
            "with fever 38-40°C; rigidity, rebound tenderness; resolves spontaneously within "
            "72 hours (key distinguishing feature from surgical abdomen — if in doubt, operate); "
            "Pleuritis (30%): pleuritic chest pain, effusion, unilateral; Arthritis (50%): "
            "monoarthritis, most commonly knee or ankle; self-limiting 1 week; "
            "Erysipelas-like erythema (ELE): well-demarcated erythematous warm plaque, shin/dorsum "
            "of foot — pathognomonic when present; Scrotal attacks (males): mimics torsion — "
            "urological emergency awareness; "
            "BETWEEN ATTACKS: completely asymptomatic (distinguishes from TRAPS); "
            "CRP/SAA normalise between attacks (SAA is the acute-phase precursor of amyloid). "
            "AA AMYLOIDOSIS: serum amyloid A (SAA) deposits in kidneys (mesangium + "
            "subendothelial) → proteinuria → nephrotic syndrome → CKD/ESRD; "
            "also spleen, liver, adrenals; colchicine prevents >95% of amyloidosis if started "
            "early and continued lifelong; SAA monitoring every 6-12 months. "
            "TREATMENT: Colchicine 1-2 mg/day (max 3 mg): inhibits microtubule polymerisation → "
            "disrupts neutrophil migration and IL-1β processing; reduces attacks ≥65%; "
            "reduces AA amyloidosis risk near-completely; safe in pregnancy and breastfeeding — "
            "NEVER STOP in pregnancy (amyloid risk + pregnancy is itself a trigger); "
            "Colchicine resistance (10-15%): add anakinra 100 mg/day SC or canakinumab 150 mg "
            "SC every 8 weeks (both FDA/EMA approved for colchicine-resistant FMF); "
            "AVOID: colchicine + cyclosporine (severe myopathy/rhabdomyolysis); "
            "colchicine + clarithromycin/erythromycin (increased colchicine toxicity); "
            "NEWBORN SCREENING: not performed routinely — diagnose by clinical criteria + genetics."
        ),
    },

    # ── NLRP3 — Cryopyrin-Associated Periodic Syndromes (CAPS) ──────────────
    {
        "gene": "NLRP3",
        "protein": "Cryopyrin (NALP3)",
        "alias": (
            "NLRP3; OMIM gene 606416; 1q44; 1036 aa; "
            "CAPS spectrum: FCAS OMIM #120100 (mildest) / MWS OMIM #191900 / "
            "NOMID (CINCA) OMIM #607115 (most severe); AD; de novo in NOMID; "
            "prevalence: FCAS ~1 in 1M; MWS ~1 in 1M; NOMID ~150-200 patients worldwide; "
            "gain-of-function mutations"
        ),
        "aa": "1036 aa",
        "kDa": "~118 kDa",
        "gene_class": (
            "Cryopyrin/NALP3 — NLR (NOD-like receptor) family, NLRP subfamily; "
            "components: N-terminal PYD (pyrin domain), central NACHT ATPase, "
            "C-terminal LRR (leucine-rich repeats); "
            "forms the NLRP3 inflammasome scaffold: NLRP3 + ASC + caspase-1; "
            "normal function: activated by PAMPs/DAMPs (uric acid, ATP, cholesterol crystals, "
            "K+ efflux) → processes pro-IL-1β → IL-1β; "
            "pathogenic GOF variants → lowered activation threshold → constitutive "
            "IL-1β/IL-18/IL-33 overproduction; "
            "1q44 locus; ubiquitously expressed in myeloid cells"
        ),
        "locus": "1q44",
        "omim_gene": 606416,
        "omim_disease": 607115,
        "phenotype": (
            "CAPS spectrum (same gene, different severity): "
            "FCAS (mildest) — cold-induced urticaria within 1-2h of cold exposure; fever; "
            "arthralgia; resolves 24h; no deafness; "
            "MWS (intermediate) — episodic urticaria (NOT cold-triggered); fever; "
            "SNHL (50%); amyloidosis risk; seronegative arthritis; "
            "NOMID/CINCA (most severe) — neonatal onset; chronic urticarial rash; "
            "aseptic meningitis (chronic CSF pleocytosis); frontal bossing; "
            "epiphyseal overgrowth (knees/wrists/ankles — characteristic radiograph); "
            "SNHL (90%); visual impairment (optic disc oedema); "
            "intellectual disability; stunted growth; amyloidosis; premature death untreated"
        ),
        "hallmark": (
            "CAPS SPECTRUM — FCAS (cold-triggered) → MWS (urticaria+SNHL) → NOMID (neonatal, "
            "chronic meningitis, epiphyseal overgrowth); "
            "canakinumab (anti-IL-1β monoclonal) FDA APPROVED 2016 for ALL THREE CAPS subtypes; "
            "LIVE VACCINES ABSOLUTELY CONTRAINDICATED on canakinumab (IL-1β blockade → "
            "impaired vaccine response; reactivation risk with live-attenuated pathogens)"
        ),
        "treatment_alert": (
            "LIVE VACCINES ABSOLUTELY CONTRAINDICATED with canakinumab — "
            "complete ALL live vaccines before starting; MMR/varicella/yellow fever/BCG; "
            "only inactivated/killed vaccines during treatment; "
            "canakinumab 150 mg SC every 8 weeks (adults); "
            "anakinra (IL-1Ra) daily SC alternative — shorter half-life useful in infections; "
            "rilonacept FDA approved for FCAS and MWS; "
            "do NOT use corticosteroids long-term (ineffective and side effects); "
            "colchicine NOT effective for NLRP3 CAPS"
        ),
        "key_ddx": (
            "Urticarial vasculitis: complement low, hypocomplementaemia; "
            "systemic-onset JIA (sJIA/SJIA): ANA, RF workup; ferritin very high; "
            "FCAS vs cold urticaria: urticaria in 20-30 min cold immersion (physical urticaria) "
            "vs FCAS 1-2h systemic cold exposure with fever/arthralgia; "
            "PFAPA: tonsillar, periodic, response to steroids; "
            "Schnitzler syndrome: monoclonal IgM + urticaria; adult onset; "
            "FMF: peritonitis dominant; no urticaria; Mediterranean ancestry"
        ),
        "gfr_pattern": (
            "Normal until AA amyloidosis (NOMID/MWS); renal function monitored 6-monthly "
            "in NOMID; canakinumab dramatically reduces amyloid risk if started early"
        ),
        "proteinuria_pattern": (
            "Proteinuria = AA amyloid; MWS 25% lifetime amyloid without treatment; "
            "NOMID higher risk; FCAS minimal amyloid risk"
        ),
        "primary_complication": (
            "NOMID: cognitive impairment, deafness, blindness, amyloidosis, premature death; "
            "MWS: SNHL progressive, amyloidosis; "
            "FCAS: urticarial attacks, minimal organ damage with treatment"
        ),
        "disease": (
            "NLRP3 encodes cryopyrin (NALP3), the core scaffold of the most studied "
            "inflammasome in innate immunity. "
            "NORMAL FUNCTION: NLRP3 inflammasome assembles in response to PAMPs and DAMPs "
            "(ATP, uric acid crystals, cholesterol crystals, K+ efflux, mitochondrial ROS); "
            "once assembled: NLRP3-ASC-caspase-1 → cleaves pro-IL-1β → active IL-1β; "
            "also activates gasdermin D (pyroptosis) and IL-18 processing. "
            "PATHOMECHANISM — CAPS: gain-of-function variants cluster in the NACHT ATPase domain "
            "(R260W, D303N, A439V, T350M); lower K+ threshold for inflammasome assembly → "
            "constitutive IL-1β/IL-18 production without physiological triggers; "
            "spectrum severity correlates with degree of GOF: "
            "FCAS: mildest — cold exposure lowers intracellular K+ → pushes cells past threshold; "
            "MWS: intermediate — spontaneous episodic activation; "
            "NOMID: most severe — continuous inflammasome activation → chronic organ inflammation. "
            "CLINICAL — FCAS: cold-triggered attacks (brief cold exposure, NOT contact); "
            "generalised urticaria + fever + polyarthralgia within 1-2 hours; "
            "resolves within 24 hours; NO ANGIOEDEMA (key — differentiates from hereditary "
            "angioedema); no SNHL in FCAS; "
            "MWS: episodic NOT cold-triggered urticaria; prolonged fever episodes; "
            "progressive SNHL (sensorineural, 50%); seronegative polyarthritis; "
            "eye: conjunctivitis, episcleritis; AA amyloidosis (25% without treatment); "
            "NOMID/CINCA: neonatal onset (birth or first weeks); "
            "TRIAD: chronic urticarial rash + chronic aseptic meningitis + "
            "arthropathy with epiphyseal overgrowth; "
            "chronic CSF: lymphocytic pleocytosis (can mimic chronic meningitis); "
            "brain MRI: white matter lesions, cerebral atrophy; "
            "epiphyseal/metaphyseal dysplasia: characteristic 'tram-track' calcification, "
            "knee/wrist/ankle knobby enlargement (radiograph pathognomonic); "
            "SNHL (90%), progressive; optic nerve oedema → visual loss; "
            "intellectual disability (without treatment); stunted growth; "
            "Hearing loss: cochlear + eighth-nerve inflammation — audiology every 6 months; "
            "GENETICS: de novo mutations common in NOMID (~75% de novo); "
            "somatic mosaicism occurs — Sanger may miss; deep sequencing required; "
            "TREATMENT: IL-1 blockade is TRANSFORMATIVE: "
            "canakinumab (anti-IL-1β mAb) 150-300 mg SC q8w — FDA approved 2016 FCAS/MWS/NOMID; "
            "sustained complete remission in >70% NOMID; halts SNHL progression; "
            "reduces meningeal inflammation; halts epiphyseal overgrowth; "
            "anakinra 1-2 mg/kg/day SC — daily dosing; useful in acute severe attacks and "
            "infections (short half-life = rapid offset); "
            "rilonacept (IL-1 TRAP) — approved FCAS/MWS adults; "
            "LIVE VACCINES: ALL live vaccines (MMR, varicella, BCG, yellow fever, nasal flu) "
            "ABSOLUTELY CONTRAINDICATED during IL-1 blockade — complete before starting; "
            "inactivated vaccines (flu shot, pneumococcal) are safe; "
            "MONITORING: audiometry 6-monthly; ophthalmology 6-monthly (optic disc); "
            "MRI brain annually (NOMID); SAA levels to monitor disease activity."
        ),
    },

    # ── TNFRSF1A — TNF Receptor-Associated Periodic Syndrome (TRAPS) ─────────
    {
        "gene": "TNFRSF1A",
        "protein": "TNFR1 (Tumour Necrosis Factor Receptor Superfamily Member 1A)",
        "alias": (
            "TNFRSF1A; OMIM gene 191190; 12p13.31; 455 aa; TRAPS OMIM #142680; "
            "AD; low-penetrance variants exist (R92Q, P46L); "
            "prevalence: ~1 in 1M worldwide; "
            "most common in Northern European ancestry (Irish, Scottish, Scottish-Irish); "
            "attacks characteristically LONG (>5 days, often 2-4 weeks)"
        ),
        "aa": "455 aa",
        "kDa": "~50 kDa (TNFR1 ectodomain shed)",
        "gene_class": (
            "TNFR1 — type I transmembrane glycoprotein; TNF receptor superfamily; "
            "55 kDa receptor with 4 extracellular cysteine-rich domains (CRD); "
            "normal: binds TNF → NF-κB activation → inflammation; "
            "ALSO signals via TRADD-FADD → apoptosis (caspase-8); "
            "normal function: receptor shedding (ADAM17/TACE) releases soluble TNFR1 (sTNFR1) → "
            "acts as decoy receptor, dampening TNF signalling; "
            "pathogenic TRAPS variants: impaired shedding → accumulated TNFR1 on cell surface → "
            "sustained NF-κB/MAPK signalling; also ER retention → misfolded protein → "
            "mitochondrial ROS → NLRP3 co-activation"
        ),
        "locus": "12p13.31",
        "omim_gene": 191190,
        "omim_disease": 142680,
        "phenotype": (
            "TRAPS: LONG febrile attacks (5-21 days, often 2-4 weeks — KEY distinguisher from FMF 1-3d); "
            "MIGRATORY myalgia (centrifugal migration from proximal to distal muscle — PATHOGNOMONIC); "
            "overlying erythematous skin (over affected muscle = migratory myalgia patch); "
            "PERIORBITAL OEDEMA (unilateral or bilateral — characteristic sign); "
            "abdominal pain + pleuritis; monoarthritis (large joints); "
            "conjunctivitis; "
            "AA amyloidosis risk 10-20% (higher with high-penetrance cysteine variants); "
            "high ESR/CRP BETWEEN attacks (FMF normalises — key distinction); "
            "low-penetrance variants (R92Q, P46L): milder, may not meet TRAPS criteria"
        ),
        "hallmark": (
            "LONG ATTACKS (>5 days to weeks) + MIGRATORY MYALGIA + PERIORBITAL OEDEMA — "
            "triad distinguishes TRAPS from FMF (short attacks, no periorbital oedema, no "
            "migratory myalgia); "
            "CORTICOSTEROIDS WORSEN LONG-TERM — reduce attack severity but INCREASE FREQUENCY; "
            "cause steroid dependence; switch to IL-1 blockade"
        ),
        "treatment_alert": (
            "CORTICOSTEROIDS: Use for acute attacks only — short-course; "
            "LONG-TERM CORTICOSTEROIDS INCREASE ATTACK FREQUENCY (steroid dependence); "
            "etanercept (TNF inhibitor): less effective than IL-1 inhibitors in TRAPS; "
            "anakinra or canakinumab preferred for high-penetrance variants; "
            "NSAID for mild attacks only; "
            "colchicine NOT effective for TRAPS (unlike FMF); "
            "AMYLOIDOSIS RISK: monitor SAA levels 6-monthly; "
            "high-penetrance cysteine variants = highest amyloid risk"
        ),
        "key_ddx": (
            "FMF: attacks 1-3 days (TRAPS 5+ days); peritonitis dominant; "
            "Mediterranean ancestry; MEFV mutation; "
            "Adult Still's disease (sJIA): quotidian fever, salmon-pink rash, "
            "arthritis, ferritin very high (>10,000); "
            "TRAPS low-penetrance R92Q: may be coincidental variant — need clinical criteria; "
            "Infectious fever: rule out with cultures, serology; "
            "MKD/HIDS: shorter attacks (3-7d); lymphadenopathy prominent; elevated IgD; AR"
        ),
        "gfr_pattern": (
            "Normal until AA amyloidosis; proteinuria = amyloid deposition; "
            "cysteine variant carriers: highest amyloid/renal risk; "
            "24h urine protein annually if high-penetrance variant"
        ),
        "proteinuria_pattern": (
            "Proteinuria = AA amyloid; renal biopsy if nephrotic-range; "
            "IL-1 blockade can stabilise renal function even after amyloid established"
        ),
        "primary_complication": (
            "AA amyloidosis (ESRD risk); disability from repeated long attacks; "
            "steroid dependence from inappropriate long-term corticosteroid use"
        ),
        "disease": (
            "TNFRSF1A encodes TNFR1 (p55/CD120a), the primary signalling receptor for TNF-α. "
            "NORMAL FUNCTION: TNF binds trimeric TNFR1 on cell surface → receptor trimerisation → "
            "TRADD recruitment → NF-κB (survival/inflammation) OR FADD-caspase-8 (apoptosis); "
            "crucially, TNFR1 ectodomain is constitutively shed by ADAM17 (TACE) producing "
            "soluble sTNFR1, which acts as a decoy receptor neutralising extracellular TNF. "
            "PATHOMECHANISM: TRAPS variants in CRD1/CRD2 (Cys30Arg, Cys30Ser, Cys33Gly, "
            "Thr50Met, His22Tyr — cysteine variants = HIGH PENETRANCE; "
            "non-cysteine: R92Q, P46L = LOW PENETRANCE): "
            "(1) Impaired ectodomain shedding → accumulated surface TNFR1 → amplified TNF "
            "signalling; reduced sTNFR1 decoy effect; "
            "(2) ER retention of misfolded TNFR1 → UPR → mitochondrial ROS → "
            "NLRP3 inflammasome co-activation → IL-1β release; "
            "(3) Impaired apoptosis → prolonged neutrophil survival; "
            "combined: prolonged, severe inflammatory episodes. "
            "CLINICAL: attacks last 5-21 days (mean 2 weeks); "
            "MIGRATORY MYALGIA: begins in proximal muscles (thigh, shoulder) and migrates "
            "distally over days — centrifugal pattern; skin over affected muscle is "
            "erythematous (20-40 cm patch) — the 'migratory erythema over myalgia' is "
            "PATHOGNOMONIC when combined with fever and family history; "
            "PERIORBITAL OEDEMA: unilateral or bilateral, non-painful; occurs in ~80% of "
            "high-penetrance variant attacks; can mimic Graves' ophthalmopathy; "
            "Abdominal pain: non-specific; less peritoneal than FMF; constipation common; "
            "Pleuritis (60-80%): pleuritic chest pain, effusion; "
            "Arthritis: mono- or oligoarthritis, large joints; non-destructive; "
            "Conjunctivitis: mild, unilateral; "
            "BETWEEN ATTACKS: elevated ESR/CRP/SAA (unlike FMF which normalises) — "
            "KEY clinical distinguisher; subclinical inflammation persists; "
            "GENETICS: autosomal dominant; penetrance variable; "
            "low-penetrance R92Q (common, ~10% of northern Europeans): may be coincidental; "
            "clinical criteria (Eurofever) must be met before attributing to R92Q; "
            "TREATMENT: "
            "Acute attacks: short-course prednisolone 0.5-1 mg/kg (tapers quickly); "
            "LONG-TERM CORTICOSTEROIDS INCREASE ATTACK FREQUENCY — corticosteroid dependence "
            "is a real hazard; must transition to steroid-sparing agents; "
            "Anakinra: 100 mg/day SC — highly effective; can taper to every-other-day; "
            "Canakinumab: 150-300 mg SC q4-8w — preferred for compliance (monthly or less); "
            "Etanercept: some benefit but INFERIOR to IL-1 inhibitors; not recommended in "
            "TRAPS as first biologic; "
            "Colchicine: NOT effective — do not substitute for FMF colchicine treatment; "
            "MONITORING: SAA 6-monthly; urinalysis annually; echocardiogram if pericarditis."
        ),
    },

    # ── MVK — Mevalonate Kinase Deficiency (MKD / HIDS) ─────────────────────
    {
        "gene": "MVK",
        "protein": "Mevalonate Kinase",
        "alias": (
            "MVK; OMIM gene 251170; 12q24.11; 396 aa; "
            "MKD spectrum: HIDS OMIM #260920 (mild) / Mevalonic aciduria OMIM #610377 (severe); "
            "AR (compound het or homozygous); "
            "HIDS: 1 in 200,000-500,000; most common in Western European ancestry (Dutch, French); "
            "p.Val377Ile most common HIDS variant (60-70%); "
            "p.Ile268Thr: severe MVA phenotype"
        ),
        "aa": "396 aa",
        "kDa": "~42 kDa",
        "gene_class": (
            "Mevalonate kinase — enzyme in isoprenoid/cholesterol biosynthesis pathway; "
            "phosphorylates mevalonic acid → mevalonate-5-phosphate; "
            "located downstream of HMG-CoA reductase (statin target); "
            "loss of function → mevalonic acid accumulates; "
            "downstream depletion: geranyl pyrophosphate (GPP), geranylgeranyl pyrophosphate (GGPP), "
            "farnesyl pyrophosphate (FPP) — required for prenylation of GTPases (Rac1, RhoA, Cdc42); "
            "unprenylated GTPases → NLRP3 inflammasome activation → IL-1β; "
            "cholesterol levels normal in HIDS (partial enzyme activity ~2-10% of normal)"
        ),
        "locus": "12q24.11",
        "omim_gene": 251170,
        "omim_disease": 260920,
        "phenotype": (
            "MKD/HIDS: episodic fever attacks (3-7 days); triggered by infection, "
            "vaccination, stress, surgery, trauma; "
            "LYMPHADENOPATHY (cervical, prominent — hallmark; also axillary/inguinal); "
            "abdominal pain (90%); nausea/vomiting/diarrhoea; "
            "arthralgia/myalgia (50-70%); "
            "skin: maculopapular rash, urticaria, petechiae; "
            "oral aphthous ulcers (50%); "
            "splenomegaly (50%); "
            "ELEVATED SERUM IgD >100 IU/mL (historical marker — not always elevated; "
            "not specific — also elevated in TRAPS and FMF); "
            "urinary mevalonic acid elevated during fever (DIAGNOSTIC); "
            "MVA (severe): dysmorphic features, cerebellar atrophy, psychomotor delay, "
            "mevalonic aciduria, failure to thrive, recurrent crises — a different disease spectrum"
        ),
        "hallmark": (
            "LYMPHADENOPATHY prominent in MKD/HIDS (especially cervical) — "
            "distinguishes from FMF and TRAPS where lymphadenopathy is absent or minor; "
            "URINARY MEVALONIC ACID elevated DURING fever episode — DIAGNOSTIC; "
            "canakinumab FDA 2021 approved for MKD; "
            "VACCINATIONS can TRIGGER attacks (but do NOT withhold vaccinations — pre-treat "
            "with anakinra or NSAIDs around vaccine day)"
        ),
        "treatment_alert": (
            "STATINS WORSEN MKD — HMG-CoA reductase inhibitors further deplete GPP/GGPP "
            "downstream of enzyme defect → more severe attacks; "
            "AVOID statins in MKD; "
            "canakinumab 150 mg SC q4-8w — FDA approved 2021; reduces attack frequency ≥70%; "
            "anakinra 100 mg/day SC — effective; shorter half-life useful around triggers; "
            "NSAIDs for mild acute attacks; "
            "prednisolone 1 mg/kg for acute attacks (short course); "
            "pre-treat with anakinra/NSAIDs before PLANNED vaccination to prevent attack; "
            "colchicine generally NOT effective for MKD"
        ),
        "key_ddx": (
            "FMF: Mediterranean; peritonitis; MEFV mutation; attacks 1-3 days; "
            "TRAPS: longer attacks (>5d); periorbital oedema; migratory myalgia; "
            "sJIA (systemic JIA): quotidian fever; salmon-pink rash; ferritin very high; "
            "PFAPA: tonsillitis + pharyngitis; periodic; corticosteroid aborts attack; "
            "lymphoma: persistent lymphadenopathy, constitutional symptoms; "
            "other mevalonate pathway disorders: Antley-Bixler (CYP51A1), "
            "Smith-Lemli-Opitz (DHCR7 — cholesterol synthesis defect)"
        ),
        "gfr_pattern": (
            "Usually normal in HIDS; MVA severe form can have renal involvement; "
            "AA amyloidosis rare in MKD (occurs but less common than FMF/TRAPS)"
        ),
        "proteinuria_pattern": (
            "Generally absent unless amyloidosis develops; "
            "urinalysis annually in long-standing disease"
        ),
        "primary_complication": (
            "Recurrent debilitating attack burden; AA amyloidosis (rare, less than FMF/TRAPS); "
            "MVA (severe MKD): cognitive impairment, cerebellar dysfunction, failure to thrive"
        ),
        "disease": (
            "MVK encodes mevalonate kinase, a cytoplasmic enzyme catalysing the phosphorylation "
            "of (R)-mevalonic acid → (R)-mevalonate-5-phosphate in the isoprenoid pathway. "
            "NORMAL FUNCTION: The mevalonate pathway produces: "
            "(1) Cholesterol (via lanosterol); "
            "(2) Isoprenoids: GPP (geranyl-PP), FPP (farnesyl-PP), GGPP (geranylgeranyl-PP); "
            "GPP/FPP/GGPP are essential for prenylation of small GTPases (Ras, Rho, Rac, Cdc42, "
            "Rab proteins), which require isoprenoid attachment for membrane anchorage and "
            "activation. "
            "PATHOMECHANISM: MVK loss-of-function → residual enzyme activity 1-20% of normal "
            "(HIDS 2-10%; MVA <1%); "
            "mevalonic acid accumulates (urinary mevalonic acid elevated during fever attacks); "
            "CRITICAL: depletion of GGPP → NLRP3 inflammasome activation; "
            "specifically: unprenylated (inactive) Rac1/RhoA → NLRP3 conformational change → "
            "IL-1β overproduction → fever attacks; "
            "FEVER ITSELF further reduces MVK enzyme activity (thermolabile) → "
            "vicious cycle: infection/fever → ↓ MVK activity → ↓ GGPP → ↑ IL-1β → more fever. "
            "CLINICAL: HIDS (mild end): episodic attacks 3-7 days; triggered by minor infections, "
            "vaccinations, trauma, emotional stress, surgery; "
            "CERVICAL LYMPHADENOPATHY — present in virtually all HIDS patients during attack; "
            "tender lymph nodes; "
            "Gastrointestinal: abdominal pain (90%), vomiting, diarrhoea; "
            "Skin: maculopapular rash, urticaria (50%); aphthous ulcers (50%); "
            "Arthralgia (50-70%): symmetric, large joint, non-destructive; "
            "Splenomegaly (50-70%); "
            "BETWEEN ATTACKS: asymptomatic; IgD may remain elevated but not reliable; "
            "urinary mevalonic acid normalises between attacks; "
            "MVA (severe, <1% MVK activity): neonatal-onset; dysmorphic face (frontal bossing, "
            "large fontanelle, triangular face, downslanting palpebral fissures); "
            "hypotonia; psychomotor retardation; cerebellar atrophy; "
            "episodic crises with vomiting, hepatosplenomegaly; "
            "failure to thrive; early death possible; "
            "LABORATORY: during fever: mevalonic acid ↑↑ (GC-MS); IgD often >100 IU/mL "
            "(not always — present in 80%); IgA also elevated (60%); "
            "CRP/ESR markedly elevated during attacks; "
            "cholesterol levels NORMAL (HIDS — sufficient residual activity); "
            "TREATMENT: "
            "Canakinumab 150 mg SC q4-8w — FDA approved June 2021 for HIDS/MKD — "
            "reduces attack frequency 70%, attack duration, CRP; "
            "anakinra 100 mg SC daily — effective; "
            "NSAIDs and prednisolone for acute attacks; "
            "STATINS: ABSOLUTELY AVOID — HMG-CoA reductase inhibitors reduce mevalonate "
            "production → further deplete GPP/GGPP → worsens attacks; "
            "Simvastatin INCREASES ATTACK FREQUENCY in MKD; "
            "VACCINATION STRATEGY: do NOT withhold vaccines — pre-treat with anakinra "
            "100 mg SC day before vaccine and 2-3 days after; "
            "or use NSAIDs peri-vaccination; vaccination refusal not recommended; "
            "NEWBORN SCREENING: mevalonic acid by MS/MS can be picked up on NBS "
            "(MVA form); HIDS not routinely screened."
        ),
    },

    # ── IL1RN — Deficiency of IL-1 Receptor Antagonist (DIRA) ──────────────
    {
        "gene": "IL1RN",
        "protein": "Interleukin-1 Receptor Antagonist (IL-1Ra)",
        "alias": (
            "IL1RN; OMIM gene 147679; 2q14.1; 177 aa (secreted form IL-1Ra1); "
            "DIRA OMIM #612852; AR (biallelic null); "
            "very rare: ~30-60 patients worldwide reported; "
            "most common in Puerto Rican, Dutch, Brazilian, Lebanese populations; "
            "Puerto Rican founder deletion (175kb deletion on 2q13); "
            "neonatal onset — can be fatal without treatment; "
            "anakinra (recombinant IL-1Ra) is ESSENTIALLY CURATIVE"
        ),
        "aa": "177 aa (secreted IL-1Ra1); intracellular forms also exist",
        "kDa": "~17-25 kDa (glycosylated)",
        "gene_class": (
            "IL-1Ra — endogenous anti-inflammatory protein; "
            "competitive antagonist of IL-1 receptor type I (IL-1RI); "
            "binds IL-1RI with same affinity as IL-1β but NO agonist activity; "
            "3 isoforms (secreted IL-1Ra1, intracellular IL-1Ra2/3); "
            "produced by monocytes, macrophages, neutrophils, hepatocytes; "
            "normal function: maintains IL-1 signalling balance — at physiological ratio, "
            "~10-100x molar excess of IL-1Ra required to block IL-1RI; "
            "complete LOF → UNOPPOSED IL-1α and IL-1β signalling → systemic and local "
            "hyperinflammation at neonatal stage"
        ),
        "locus": "2q14.1",
        "omim_gene": 147679,
        "omim_disease": 612852,
        "phenotype": (
            "DIRA: neonatal onset (birth to 4 weeks); "
            "STERILE MULTIFOCAL OSTEOMYELITIS (SMO) — bone pain, swelling, pathological fractures; "
            "periostitis (cortical bone reaction on X-ray); "
            "pustular skin rash (diffuse neutrophilic pustulosis — not varicella); "
            "mucous membrane involvement (oral, genital); "
            "pulmonary infiltrates (alveolar neutrophilia); "
            "hepatosplenomegaly; "
            "growth failure; "
            "extreme CRP elevation; "
            "death from systemic inflammatory failure in weeks-months WITHOUT treatment; "
            "ANAKINRA (IL-1Ra) → ESSENTIALLY CURATIVE within days to weeks"
        ),
        "hallmark": (
            "NEONATAL STERILE MULTIFOCAL OSTEOMYELITIS + PUSTULAR SKIN RASH — "
            "in first weeks of life; "
            "ANAKINRA IS ESSENTIALLY CURATIVE — response within 24-48 hours of starting; "
            "complete reversal of osteomyelitis lesions with treatment; "
            "without treatment: FATAL within months from sepsis-like systemic inflammation; "
            "do NOT delay treatment for genetic confirmation if DIRA clinically suspected"
        ),
        "treatment_alert": (
            "ANAKINRA (IL-1Ra): start IMMEDIATELY if DIRA suspected; "
            "1-4 mg/kg/day SC (neonates may need higher doses); "
            "response within 24-72 hours; do NOT wait for genetic result; "
            "LIFELONG treatment required — relapse on stopping; "
            "LIVE VACCINES ABSOLUTELY CONTRAINDICATED on anakinra; "
            "complete ALL live vaccines before anakinra; "
            "canakinumab can substitute for anakinra when compliance is an issue; "
            "NSAIDs and corticosteroids INSUFFICIENT — partial suppression only; "
            "HSCT is potentially curative for DIRA (eliminates need for lifelong anakinra)"
        ),
        "key_ddx": (
            "Neonatal sepsis: blood cultures; empirical antibiotics until DIRA confirmed; "
            "Neonatal chronic multifocal osteomyelitis (CRMO/CNO): onset usually older, "
            "not neonatal; no pustular rash; IL1RN normal; "
            "Deficiency of IL-36Ra (DITRA): GPR84 or IL36RN — pustular psoriasis; "
            "Langerhans cell histiocytosis: BRAF V600E; bone lesions; CD1a/CD207+ cells; "
            "Congenital syphilis: VDRL/RPR; periostitis + rash + hepatosplenomegaly; "
            "Majeed syndrome (LPIN2): CRMO + congenital dyserythropoietic anaemia + "
            "inflammation — AR; different gene"
        ),
        "gfr_pattern": (
            "Normal — renal involvement uncommon in DIRA when treated; "
            "untreated severe disease may develop amyloidosis; "
            "renal function monitored annually on anakinra"
        ),
        "proteinuria_pattern": (
            "Absent with early treatment; amyloidosis possible in delayed diagnosis"
        ),
        "primary_complication": (
            "FATAL systemic inflammatory crisis without treatment; "
            "pathological fractures from osteomyelitis; "
            "growth retardation; "
            "treated patients have near-normal life expectancy on anakinra"
        ),
        "disease": (
            "IL1RN encodes interleukin-1 receptor antagonist (IL-1Ra), the primary endogenous "
            "brake on IL-1 signalling. "
            "NORMAL FUNCTION: IL-1Ra binds IL-1 receptor type I (IL-1RI) with the same affinity "
            "as IL-1α and IL-1β but WITHOUT transducing any signal — purely competitive "
            "antagonism; ~100-1000x molar excess required to fully block IL-1 signalling; "
            "IL-1Ra is secreted constitutively by monocytes, macrophages, neutrophils, and "
            "hepatocytes, and is massively upregulated during infection/inflammation. "
            "PATHOMECHANISM — DIRA: biallelic null mutations (deletions, frameshift, nonsense) → "
            "complete absence of IL-1Ra → UNOPPOSED IL-1α and IL-1β signalling through IL-1RI → "
            "constitutive NF-κB activation in osteoblasts, osteoclasts, keratinocytes, "
            "endothelium, hepatocytes, neutrophils → "
            "STERILE SYSTEMIC INFLAMMATION from birth. "
            "Puerto Rican founder variant: 175 kb deletion on 2q13 encompassing entire IL1RN "
            "locus — homozygous in affected Puerto Rican patients; heterozygous carrier "
            "frequency ~1 in 200 in Puerto Rican population. "
            "CLINICAL: presentations from first days to first month of life; "
            "STERILE MULTIFOCAL OSTEOMYELITIS: rib, vertebral, long bone cortical destruction; "
            "periostitis on plain X-ray (cortical thickening, onion-skin layering); "
            "no organisms on culture — sterile; nuclear medicine bone scan shows multiple foci; "
            "SKIN: diffuse neutrophilic pustulosis; generalised pustular eruption; "
            "can cover 50-80% of BSA; distinguishable from neonatal HSV/impetigo; "
            "SYSTEMIC: CRP >200 mg/L typical; hepatosplenomegaly; anaemia of inflammation; "
            "pulmonary alveolar neutrophilia → respiratory distress; "
            "NO FEVER (or low-grade only) — unlike other autoinflammatory diseases; "
            "UNTREATED: sepsis-like deterioration → multi-organ failure → "
            "DEATH within weeks to months; "
            "GENETICS: AR; consanguinity in many families; Puerto Rican, Newfoundland, "
            "Lebanese, Dutch affected; "
            "DIAGNOSIS: genetic testing IL1RN; confirm with IL-1Ra serum level (absent); "
            "skin/bone biopsy: sterile neutrophilic infiltrate; "
            "TREATMENT: "
            "Anakinra (Kineret) — recombinant human IL-1Ra; 1-4 mg/kg/day SC; "
            "response: CRP normalises within 24-72h; skin clears in 1-2 weeks; "
            "bone lesions heal over months on treatment; "
            "LIFELONG therapy required — discontinuation causes relapse; "
            "HSCT: potentially curative — reported cases of successful HSCT providing "
            "donor-derived IL-1Ra; considered in families where lifelong injection is not feasible; "
            "LIVE VACCINES: ABSOLUTELY CONTRAINDICATED on anakinra — "
            "switch to canakinumab if better dosing schedule needed; "
            "MONITORING: CRP/SAA monthly; bone imaging annually (MRI or bone scan); "
            "growth parameters closely; annual immunoglobulin levels."
        ),
    },

    # ── CECR1 — Deficiency of ADA2 (DADA2) ──────────────────────────────────
    {
        "gene": "CECR1",
        "protein": "Adenosine Deaminase 2 (ADA2)",
        "alias": (
            "CECR1; OMIM gene 607575; 22q11.1; 511 aa; DADA2 OMIM #615688; "
            "AR (biallelic); prevalence: <200 patients worldwide reported; "
            "multiple ethnic groups: Georgian Jewish (founder p.Gly47Arg), Turkish, "
            "European; "
            "RENAMED from CECR1 (cat eye syndrome chromosome region 1) to ADA2; "
            "the most CLINICALLY IMPORTANT pitfall: childhood STROKE from vasculitis"
        ),
        "aa": "511 aa",
        "kDa": "~59 kDa (signal peptide cleaved; secreted homodimer ~118 kDa)",
        "gene_class": (
            "ADA2 — secreted adenosine deaminase; deaminates adenosine → inosine; "
            "expressed in myeloid cells (monocytes, macrophages, NK cells, plasma cells); "
            "structurally unrelated to ADA1 (ADA1 deficiency = SCID); "
            "normal function: ADA2 promotes M2 (anti-inflammatory) macrophage polarisation; "
            "promotes regulatory T-cell differentiation; endothelial homeostasis; "
            "LOF → M1 (pro-inflammatory) macrophage predominance → NF-κB and STAT1 activation; "
            "endothelial dysfunction → VASCULITIS → stroke and organ involvement; "
            "locus 22q11.1 — note: NOT the 22q11.2 deletion (DiGeorge) locus"
        ),
        "locus": "22q11.1",
        "omim_gene": 607575,
        "omim_disease": 615688,
        "phenotype": (
            "DADA2: highly variable — overlap of autoinflammatory + immunodeficiency + vasculitis; "
            "STROKE (lacunar infarcts, most severe manifestation): childhood or young adult onset; "
            "recurrent; deep grey matter (basal ganglia, brainstem, thalamus); "
            "LIVEDO RETICULARIS (70%) — racemosa pattern on skin; "
            "Raynaud phenomenon; "
            "splenomegaly (60%); hepatomegaly; "
            "polyarteritis nodosa-like vasculitis (PAN phenotype): aneurysms, infarcts; "
            "pure red cell aplasia / pancytopenia (immunodeficiency overlap); "
            "common variable immunodeficiency (CVID) phenotype (20%); "
            "fever attacks; lymphoproliferation; "
            "PLASMA ADA2 ACTIVITY LOW — diagnostic biomarker"
        ),
        "hallmark": (
            "CHILDHOOD STROKE (LACUNAR INFARCTS) + LIVEDO RETICULARIS — "
            "DADA2 must be excluded in any child or young adult with unexplained stroke; "
            "TNF INHIBITORS PREVENT STROKE — TNF-α blockade dramatically reduces "
            "stroke recurrence; etanercept/adalimumab prevent further strokes; "
            "IVIG IS INEFFECTIVE — do not use for stroke prevention; "
            "HSCT IS CURATIVE (bone marrow transplant eliminates underlying defect)"
        ),
        "treatment_alert": (
            "TNF INHIBITORS (etanercept or adalimumab): START IMMEDIATELY to prevent stroke; "
            "dramatically reduces stroke recurrence; "
            "IVIG: DOES NOT PREVENT STROKE — ineffective for vascular/vasculitis manifestations; "
            "do NOT substitute IVIG for TNF blockade in stroke-prone DADA2; "
            "HSCT is potentially curative — consider early in severe phenotype; "
            "LIVE VACCINES: CONTRAINDICATED on TNF inhibitors; "
            "plasma ADA2 supplementation (fresh frozen plasma): reported to help temporarily; "
            "PURE RED CELL APLASIA: HSCT is best; may respond to cyclosporine + EPO"
        ),
        "key_ddx": (
            "ADA1 deficiency (ADA): SCID (severe combined immunodeficiency) — very low T/B/NK; "
            "ADA2 deficiency is NOT a SCID (T/B cells present); "
            "Polyarteritis nodosa (PAN): clinically overlapping; CECR1 mutation testing mandatory "
            "in childhood PAN; "
            "Childhood vasculitis stroke DDx: CNS vasculitis, moyamoya, antiphospholipid syndrome; "
            "PAPA syndrome (PSTPIP1): pyogenic arthritis, pyoderma, acne; "
            "Sneddon syndrome (APLA + livedo): antiphospholipid antibodies"
        ),
        "gfr_pattern": (
            "Renal vasculitis (PAN-like): renal artery aneurysms → renovascular hypertension; "
            "haematuria, proteinuria from renal infarction; "
            "ACE inhibitors for renovascular hypertension"
        ),
        "proteinuria_pattern": (
            "Proteinuria from renal infarction or vasculitis; "
            "not amyloid-mediated primarily; "
            "renal angiography if hypertension"
        ),
        "primary_complication": (
            "STROKE (recurrent lacunar infarcts) — most feared complication; "
            "pure red cell aplasia / pancytopenia; "
            "vasculitis-mediated organ damage"
        ),
        "disease": (
            "CECR1 (ADA2) encodes adenosine deaminase 2, a secreted homodimeric enzyme that "
            "deaminates adenosine to inosine in the extracellular space. "
            "IMPORTANT: ADA2 is structurally unrelated to ADA1 (the enzyme deficient in "
            "ADA-SCID) despite sharing the adenosine deaminase reaction. "
            "NORMAL FUNCTION: ADA2 is highly expressed in monocytes, macrophages, plasma cells, "
            "and dendritic cells; "
            "promotes differentiation of monocytes towards an M2-like (anti-inflammatory/repair) "
            "phenotype; regulates adenosine receptor signalling on endothelium; "
            "supports endothelial cell survival and maintenance of vascular wall integrity; "
            "contributes to regulatory T-cell generation; "
            "low plasma ADA2 activity is measurable and correlates with disease. "
            "PATHOMECHANISM: biallelic LOF → absent/very low plasma ADA2 → "
            "(1) M1 macrophage polarisation predominates → NF-κB overactivation → IL-6, TNF, "
            "IL-12 overproduction → autoinflammatory phenotype; "
            "(2) Endothelial dysfunction → small-vessel vasculitis → LACUNAR INFARCTS; "
            "(3) Lymphoid dysregulation → lymphoproliferation, hypogammaglobulinaemia; "
            "TNF-α is a key mediator — TNF inhibition dramatically reduces vascular events; "
            "Georgian Jewish founder variant p.Gly47Arg (c.139G>A) — carrier frequency ~1/80. "
            "CLINICAL — HIGHLY VARIABLE: 3 main phenotypic clusters: "
            "(A) Inflammatory/Vasculitis: livedo reticularis (racemosa pattern — "
            "net-like violaceous mottling of skin, NOT benign cutis marmorata); "
            "Raynaud; polyarteritis nodosa-like (visceral aneurysms, bowel ischaemia); "
            "STROKE: lacunar infarcts in deep grey matter — hallmark and most feared; "
            "can occur from toddler age; recurrent without TNF inhibition; "
            "(B) Immunodeficiency: common variable immunodeficiency (CVID)-like; "
            "recurrent bacterial infections; hypogammaglobulinaemia; "
            "(C) Haematological: pure red cell aplasia; pancytopenia; "
            "bone marrow failure; lymphoproliferation; "
            "OVERLAP between phenotypes is common; same mutation can cause different phenotypes "
            "even within the same family; "
            "DIAGNOSIS: plasma ADA2 enzyme activity (low or absent); CECR1 gene sequencing; "
            "MRI brain (lacunar infarcts); "
            "skin biopsy: small-vessel fibrinoid necrosis; "
            "TREATMENT: "
            "TNF INHIBITORS: etanercept 0.8 mg/kg SC weekly or adalimumab — FIRST CHOICE; "
            "reduces stroke recurrence dramatically (from ~70% to <10% in series); "
            "start as soon as DADA2 diagnosed if any vasculitis/stroke manifestation; "
            "HSCT: curative — recommended for bone marrow failure, severe phenotype, "
            "or refractory disease; provides donor-derived ADA2; "
            "IVIG: for hypogammaglobulinaemia (CVID phenotype) — NOT for stroke prevention; "
            "PLASMA ADA2 (FFP infusion): temporary bridge while awaiting HSCT; "
            "LIVE VACCINES: CI on TNF inhibitors; inactivated vaccines safe; "
            "MONITORING: MRI brain 6-12 monthly (stroke surveillance); "
            "plasma ADA2 activity; immunoglobulin levels; CBC (haematological involvement)."
        ),
    },

    # ── NOD2 — Blau Syndrome / Early-Onset Sarcoidosis (EOS) ────────────────
    {
        "gene": "NOD2",
        "protein": "NOD2 (Nucleotide-Binding Oligomerisation Domain 2 / CARD15)",
        "alias": (
            "NOD2; OMIM gene 605956; 16q12.1; 1040 aa; "
            "Blau syndrome OMIM #186580 (familial, AD); "
            "early-onset sarcoidosis OMIM #609464 (sporadic, de novo); "
            "AD gain-of-function in NF-κB pathway; "
            "prevalence: <300 families worldwide; "
            "different from IBD-associated NOD2 variants (Crohn disease): "
            "Blau GOF ≠ Crohn LOF"
        ),
        "aa": "1040 aa",
        "kDa": "~114 kDa",
        "gene_class": (
            "NOD2/CARD15 — intracellular PRR (pattern-recognition receptor); "
            "NLR family; central regulator of innate NF-κB response to bacterial muramyl dipeptide (MDP); "
            "domain structure: 2 N-terminal CARDs (caspase activation/recruitment domains) → "
            "ASC/RIP2 recruitment; central NACHT ATPase; C-terminal LRR (MDP sensing); "
            "normal LOF (Crohn): impaired mucosal innate immunity → chronic GI inflammation; "
            "Blau GOF: R334W/Q, L469F variants → constitutive NF-κB activation without MDP → "
            "GRANULOMA formation in skin, joints, eyes; "
            "locus 16q12.1 — SAME as MEFV at 16p13.3 but different chromosome arm"
        ),
        "locus": "16q12.1",
        "omim_gene": 605956,
        "omim_disease": 186580,
        "phenotype": (
            "Blau syndrome classic TRIAD (incomplete penetrance): "
            "1) GRANULOMATOUS POLYARTHRITIS: symmetric, boggy/camptodactyly; "
            "non-destructive early but can develop flexion contractures; "
            "wrist/ankle/IP joints; periarticular cysts; synovial biopsy: non-caseating granulomata; "
            "2) UVEITIS (pan-uveitis): often bilateral; can lead to band keratopathy, "
            "cataract, glaucoma, visual loss; "
            "3) SKIN (ichthyosiform/lichenoid rash): maculopapular, tan-coloured; "
            "eruptions on trunk and extremities; biopsy: epithelioid granulomata; "
            "onset usually <4 years of age; "
            "systemic: fever, lymphadenopathy; "
            "rare: GRANULOMATOUS VASCULITIS — aneurysms of large vessels; "
            "NORMAL ACE and serum calcium (distinguishes from pulmonary sarcoidosis)"
        ),
        "hallmark": (
            "CLASSIC TRIAD ONSET <4 YEARS: granulomatous arthritis + uveitis + skin rash; "
            "NON-CASEATING GRANULOMATA on biopsy (skin, synovium, liver) = PATHOGNOMONIC histology; "
            "ACE and CALCIUM usually NORMAL (pulmonary sarcoid ACE elevated, "
            "hypercalcaemia common — key DDx); "
            "R334W/R334Q most common Blau variants; "
            "sporadic EOS = same gene, same lesions, but de novo mutation"
        ),
        "treatment_alert": (
            "UVEITIS: monitor EVERY 3-6 MONTHS even when disease appears clinically quiet — "
            "uveitis can be asymptomatic and progress to blindness; "
            "topical corticosteroids (eye drops) for anterior uveitis; "
            "methotrexate/azathioprine for steroid-sparing; "
            "TNF inhibitors (adalimumab, infliximab) for refractory uveitis; "
            "adalimumab FDA approved for non-infectious uveitis (Humira); "
            "GRANULOMATOUS VASCULITIS: high risk — angiography for unexplained hypertension; "
            "IL-6 inhibitors (tocilizumab) increasingly used; "
            "systemic corticosteroids: moderately effective but steroid-sparing essential; "
            "colchicine NOT effective for Blau granulomatous inflammation"
        ),
        "key_ddx": (
            "Pulmonary sarcoidosis: ACE elevated; hypercalcaemia; pulmonary hilar adenopathy; "
            "bilateral hilar lymphadenopathy on CXR — these features ABSENT in Blau; "
            "JIA (juvenile idiopathic arthritis): RF/ANA; no granulomata on synovial biopsy; "
            "Crohn disease (NOD2 LOF): gastrointestinal; NOD2 LOF variants differ from Blau GOF; "
            "Tuberculosis: caseating granulomata; ZN stain/culture/PCR positive; "
            "PAPA syndrome (PSTPIP1): pyoderma, pyogenic arthritis, acne — no granulomata; "
            "Enthesitis-related arthritis: HLA-B27; sacroiliitis; no granulomata"
        ),
        "gfr_pattern": (
            "Usually normal; granulomatous interstitial nephritis if renal granulomata; "
            "rare renal involvement (<10%); urinalysis annually"
        ),
        "proteinuria_pattern": (
            "Usually absent; granulomatous nephritis if renal involvement"
        ),
        "primary_complication": (
            "BLINDNESS from uveitis (most feared — insidious if untreated); "
            "joint contractures; "
            "large vessel vasculitis (aneurysm rupture — rare but catastrophic)"
        ),
        "disease": (
            "NOD2 encodes nucleotide-binding oligomerisation domain 2 (also called CARD15), "
            "an intracellular pattern-recognition receptor that senses bacterial muramyl "
            "dipeptide (MDP), a component of both Gram-positive and Gram-negative cell walls. "
            "NORMAL FUNCTION: MDP binds the NOD2 LRR domain → conformational change → "
            "oligomerisation via NACHT domain → CARD domain recruits RIPK2 (RIP2 kinase) → "
            "NF-κB and MAPK activation → pro-inflammatory cytokine production + "
            "autophagy induction for bacterial clearance; "
            "also involved in intestinal mucosal homeostasis (Paneth cell NOD2 expression "
            "regulates antimicrobial peptide production). "
            "PATHOMECHANISM — BLAU vs CROHN distinction: "
            "IBD/Crohn NOD2 variants (R702W, G908R, L1007fsinsC): LOF → impaired MDP sensing → "
            "defective mucosal innate response → chronic dysregulated intestinal inflammation; "
            "Blau NOD2 variants (R334W, R334Q, L469F, G481D — 16q12.1 NACHT domain): GOF → "
            "constitutive NF-κB activation WITHOUT MDP binding → "
            "spontaneous mononuclear cell activation → EPITHELIOID GRANULOMA formation; "
            "granulomata are non-caseating (histiocytes + lymphocytes, no central necrosis); "
            "target organs: skin, synovium, uveal tract (uvea = iris + ciliary body + choroid). "
            "CLINICAL: onset 2-4 years of age; "
            "ARTHRITIS: boggy, non-destructive symmetric polyarthritis; "
            "camptodactyly (fixed flexion deformity of fingers from synovial granulomata); "
            "periarticular cysts (bony erosions from granulomata visible on MRI); "
            "UVEITIS: THE MOST DANGEROUS MANIFESTATION; "
            "insidious pan-uveitis (anterior + intermediate + posterior); "
            "BILATERAL in 80%; often asymptomatic — children may lose vision without complaints; "
            "band keratopathy (calcium deposits in cornea), cataract, secondary glaucoma; "
            "visual loss in 10-20% despite treatment; "
            "OPHTHALMOLOGY REVIEW EVERY 3-6 MONTHS MANDATORY even in remission; "
            "SKIN: ichthyosiform lichenoid eruption; small tan-coloured papules on trunk/limbs; "
            "skin biopsy 1 cm punch → non-caseating granulomata = EASIEST diagnostic tissue; "
            "ACE: NORMAL (elevated in pulmonary sarcoid — critical DDx point); "
            "CALCIUM: NORMAL (hypercalcaemia of macrophage 1α-hydroxylase = pulmonary sarcoid); "
            "TREATMENT: "
            "Systemic corticosteroids: prednisolone for acute inflammation; "
            "Methotrexate or azathioprine: steroid-sparing, disease-modifying; "
            "TNF inhibitors: adalimumab (preferred; FDA approved for uveitis) or infliximab — "
            "for refractory uveitis and granulomatous arthritis; "
            "IL-6 inhibitor (tocilizumab): increasingly used for granulomatous vasculitis; "
            "Topical corticosteroid eye drops + mydriatics: anterior uveitis acute management; "
            "BLINDING can occur — zero tolerance for missed uveitis surveillance; "
            "NEWBORN SCREENING: not applicable — genetic testing family members of Blau propositus."
        ),
    },

    # ── PSMB8 — Proteasome-Associated Autoinflammatory Syndromes (PRAAS) ─────
    {
        "gene": "PSMB8",
        "protein": "Proteasome Subunit Beta Type-8 (Beta5i / LMP7)",
        "alias": (
            "PSMB8; OMIM gene 177046; 6p21.32; 277 aa; "
            "PRAAS1 / Nakajo-Nishimura syndrome OMIM #256040; "
            "CANDLE syndrome (chronic atypical neutrophilic dermatosis with lipodystrophy and "
            "elevated temperature) OMIM #256040; "
            "JMP (joint contractures, muscle atrophy, microcytic anaemia, panniculitis-induced "
            "lipodystrophy) OMIM #614252; "
            "AR biallelic; very rare <100 patients worldwide; "
            "Japanese founder p.Gly201Val; Puerto Rican/Mediterranean cases also reported; "
            "JAK1/2 inhibitors: baricitinib, tofacitinib — transformative treatment"
        ),
        "aa": "277 aa",
        "kDa": "~30 kDa (mature cleaved form)",
        "gene_class": (
            "PSMB8/beta5i/LMP7 — inducible proteasome catalytic subunit; "
            "immunoproteasome (IP) subunit beta5i (incorporated in IP replacing beta5); "
            "IP expressed in immune cells and cells exposed to IFN-γ; "
            "function: degrades ubiquitinated proteins into peptides for MHC class I presentation; "
            "PSMB8 LOF → dysfunctional immunoproteasome → accumulation of ubiquitinated proteins → "
            "ER stress → unfolded protein response (UPR) → ISG (interferon-stimulated gene) upregulation; "
            "high type I and II interferon signature: INF-α/β/γ → JAK-STAT1/2 pathway → "
            "chronic sterile inflammation; 6p21.32 within MHC class III region"
        ),
        "locus": "6p21.32",
        "omim_gene": 177046,
        "omim_disease": 256040,
        "phenotype": (
            "PRAAS/Nakajo-Nishimura/CANDLE: neonatal or early infantile onset; "
            "LIPODYSTROPHY (progressive, partial — face + trunk + limbs); "
            "PANNICULITIS: recurrent erythematous/nodular skin plaques (sterile, neutrophilic); "
            "PERIODIC FEVER: daily or near-daily low-grade fevers; "
            "JOINT CONTRACTURES: progressive, especially fingers (camptodactyly); "
            "muscle atrophy (myopathy); "
            "hepatosplenomegaly; lymphadenopathy; "
            "anaemia (microcytic, normocytic); "
            "high type I/II interferon signature: elevated ISG15, IFIT1, OASL in blood; "
            "high ESR/CRP; "
            "CALCIFICATION of basal ganglia (brain CT); "
            "progressive — without treatment: severely disabled by adolescence"
        ),
        "hallmark": (
            "LIPODYSTROPHY + PANNICULITIS + PERIODIC FEVER from NEONATAL/EARLY INFANTILE ONSET — "
            "triad in PRAAS/CANDLE; "
            "HIGH TYPE I INTERFERON SIGNATURE (IFN score) — blood RNA-seq or ISG15/IFIT1 — "
            "PATHOGNOMONIC for interferonopathies including PRAAS; "
            "JAK INHIBITORS (baricitinib, tofacitinib) — TRANSFORMATIVE; "
            "reduce interferon score within weeks; reduce fever, panniculitis, lipodystrophy "
            "progression; dramatically improve quality of life"
        ),
        "treatment_alert": (
            "JAK INHIBITORS: baricitinib 2-4 mg/day or tofacitinib — FIRST-LINE for PRAAS; "
            "reduce interferon score; suppress panniculitis and fever; "
            "LIVE VACCINES ABSOLUTELY CONTRAINDICATED on JAK inhibitors; "
            "screening for TB/hepatitis before starting JAK inhibitor (reactivation risk); "
            "corticosteroids: partially effective, cause lipodystrophy worsening; "
            "NSAIDs/colchicine: inadequate; "
            "HSCT has been attempted — variable outcomes; "
            "IFN-γ blockade (emapalumab): studied; may help some PRAAS patients"
        ),
        "key_ddx": (
            "CANDLE from other PRAAS genes: PSMB9, PSMB4, PSMA3, POMP mutations — "
            "same phenotype, different proteasome gene; "
            "Systemic lupus erythematosus: ANA/anti-dsDNA positive; "
            "panniculitis from SLE (lupus panniculitis) — distinguish by IFN score; "
            "Congenital lipodystrophy (Berardinelli-Wiedemann AGPAT2/BSCL2): "
            "generalised, not inflammatory; "
            "NLRC4-MAS (macrophage activation syndrome): "
            "very high ferritin, haemophagocytosis; "
            "Neutral lipid storage disease (ATGL): ichthyosis + myopathy + lipid vacuoles; "
            "Aicardi-Goutières syndrome: interferonopathy; "
            "brain calcification + CSF lymphocytosis + IFN-α elevated"
        ),
        "gfr_pattern": (
            "Generally preserved; proteinuria from amyloid if late; "
            "renal function annual monitoring; "
            "metabolic derangements from lipodystrophy: insulin resistance → diabetes → CKD risk"
        ),
        "proteinuria_pattern": (
            "Usually absent; metabolic nephropathy from lipodystrophy-associated insulin resistance "
            "possible over decades"
        ),
        "primary_complication": (
            "Progressive lipodystrophy and muscle atrophy; joint contractures; "
            "metabolic complications (insulin resistance, diabetes, hypertriglyceridaemia) "
            "from lipodystrophy; brain calcification (basal ganglia)"
        ),
        "disease": (
            "PSMB8 encodes the beta5i (LMP7) catalytic subunit of the immunoproteasome (IP), "
            "which replaces the constitutive beta5 subunit upon IFN-γ stimulation. "
            "NORMAL FUNCTION: The immunoproteasome is the primary protease in immune cells "
            "for degrading ubiquitinated proteins into 8-10 aa peptides → loaded onto MHC "
            "class I molecules by TAP → presented to CD8+ T cells for adaptive immunity; "
            "beta5i has chymotrypsin-like cleavage activity after hydrophobic residues; "
            "the IP generates immunopeptides more efficiently than the constitutive proteasome. "
            "PATHOMECHANISM: PSMB8 biallelic LOF → absent/dysfunctional beta5i → "
            "IP assembly defect → accumulation of polyubiquitinated proteins in cells → "
            "ER STRESS + UNFOLDED PROTEIN RESPONSE (UPR) → "
            "IRF3/IRF7 activation → TYPE I INTERFERON PRODUCTION (IFN-α/β); "
            "IFN-α/β → JAK1-TYK2 → STAT1/2 → interferon-stimulated gene (ISG) expression: "
            "ISG15, IFIT1, IFIT2, MX1, OASL — collectively the 'interferon signature' or "
            "'IFN score'; "
            "this is the common mechanism across all PRAAS-associated proteasome gene mutations; "
            "persistent high IFN signalling → PROGRESSIVE tissue inflammation + lipodystrophy. "
            "CLINICAL — PRAAS/Nakajo-Nishimura: onset neonatal to early infantile; "
            "PANNICULITIS: recurrent painful erythematous plaques; sterile neutrophilic and "
            "lymphocytic infiltrate on biopsy; affects face, trunk, limbs; "
            "LIPODYSTROPHY: progressive loss of subcutaneous fat; starts face (lipoatrophy) → "
            "spreads to trunk and limbs; Cushingoid fat redistribution paradox; "
            "leads to severe metabolic complications: insulin resistance, diabetes mellitus, "
            "mixed hyperlipidaemia (hypertriglyceridaemia, low HDL); "
            "JOINT: camptodactyly from periarticular inflammation; muscle atrophy; "
            "FEVER: daily or near-daily low-grade fever (38-39°C); malaise; "
            "HEPATOSPLENOMEGALY; "
            "HAEMATOLOGICAL: anaemia (microcytic or normocytic); leukocytosis; "
            "BRAIN: basal ganglia calcification on CT (calcium deposits visible at 5-10y); "
            "GENETICS: AR; PSMB8 p.Gly201Val — Japanese founder variant (Nakajo-Nishimura); "
            "other genes causing CANDLE: PSMB9 (beta1i), PSMB4 (beta3), PSMA3 (alpha7), "
            "POMP — all produce identical phenotype via same mechanism; "
            "IFN SCORE: blood RNA-seq or RT-PCR panel (ISG15, IFIT1, MX1, OASL, IFIT3) — "
            "score >2 SD above normal = interferon signature = DIAGNOSTICALLY SUPPORTIVE; "
            "TREATMENT: "
            "JAK INHIBITORS: baricitinib (JAK1/2 inhibitor) 2-4 mg/day — most evidence; "
            "suppresses JAK-STAT interferon pathway; reduces panniculitis, fever, improves "
            "lipodystrophy (partially reverses); normalises IFN score within weeks; "
            "tofacitinib (JAK1/3) also effective; ruxolitinib (JAK1/2) used; "
            "LIVE VACCINES: ABSOLUTELY CONTRAINDICATED on JAK inhibitors; "
            "TB and hepatitis screening mandatory before starting; "
            "METABOLIC: metformin/insulin for diabetes; fibrates for hypertriglyceridaemia; "
            "MONITORING: IFN score 6-monthly; skin; metabolic panel; brain MRI/CT annually; "
            "lipid panel 6-monthly; DEXA for fat distribution; "
            "NEWBORN SCREENING: IFN score may be measurable on NBS blood spot in future."
        ),
    },
]


def _make_cohort(gene: dict, seed: int, n: int = 40) -> list:
    """Generate a synthetic patient cohort for a gene."""
    rng = random.Random(seed)
    g = gene["gene"]
    patients = []
    for i in range(n):
        age = rng.randint(1, 65)
        sex = rng.choice(["M", "F"])
        # Gene-specific severity and features
        if g == "MEFV":
            subtype = rng.choice(["M694V/M694V", "M694V/M680I", "M694V/V726A", "E148Q/M694V"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[25, 40, 35])[0]
            esrd = rng.random() < 0.12   # AA amyloidosis → ESRD in non-adherent
            htn = rng.random() < 0.20
            drug_error = rng.random() < 0.15  # colchicine stopped or dose-reduced
            dx_delayed = rng.random() < 0.40  # often labelled 'recurrent appendicitis'
            transplant = rng.random() < 0.08
            adherent = rng.random() < 0.72
        elif g == "NLRP3":
            subtype = rng.choice(["FCAS (mild)", "MWS (intermediate)", "NOMID/CINCA (severe)", "MWS+SNHL"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[30, 35, 35])[0]
            esrd = rng.random() < 0.08   # AA amyloid in MWS/NOMID
            htn = rng.random() < 0.15
            drug_error = rng.random() < 0.25  # live vaccine given on canakinumab
            dx_delayed = rng.random() < 0.55  # rare disease, often late diagnosis
            transplant = rng.random() < 0.03
            adherent = rng.random() < 0.75
        elif g == "TNFRSF1A":
            subtype = rng.choice(["Cys-variant (high-penetrance)", "R92Q (low-penetrance)", "T50M", "Non-Cys high-penetrance"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[30, 38, 32])[0]
            esrd = rng.random() < 0.10   # AA amyloid (cysteine variants especially)
            htn = rng.random() < 0.18
            drug_error = rng.random() < 0.32  # long-term corticosteroids → attack increase
            dx_delayed = rng.random() < 0.60  # often diagnosed late or as FMF
            transplant = rng.random() < 0.06
            adherent = rng.random() < 0.68
        elif g == "MVK":
            subtype = rng.choice(["HIDS (p.Val377Ile/Val377Ile)", "HIDS compound het", "MVA severe", "HIDS mild"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[35, 40, 25])[0]
            esrd = rng.random() < 0.05   # AA amyloid rare in MKD
            htn = rng.random() < 0.12
            drug_error = rng.random() < 0.20  # statin given → worsens attacks
            dx_delayed = rng.random() < 0.65  # rare; often labelled PFAPA initially
            transplant = rng.random() < 0.03
            adherent = rng.random() < 0.70
        elif g == "IL1RN":
            subtype = rng.choice(["DIRA Puerto Rican founder del", "DIRA Dutch variant", "DIRA Lebanese", "DIRA other biallelic"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[10, 20, 70])[0]
            esrd = rng.random() < 0.08   # amyloid if delayed treatment
            htn = rng.random() < 0.10
            drug_error = rng.random() < 0.30  # delayed anakinra start
            dx_delayed = rng.random() < 0.70  # often initially treated as neonatal sepsis
            transplant = rng.random() < 0.08  # HSCT for curative intent
            adherent = rng.random() < 0.82   # anakinra-treated adhere well once diagnosed
        elif g == "CECR1":
            subtype = rng.choice(["DADA2 vasculitis/stroke", "DADA2 CVID phenotype", "DADA2 PAN-like", "DADA2 haematological"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[20, 35, 45])[0]
            esrd = rng.random() < 0.12   # renovascular hypertension → CKD
            htn = rng.random() < 0.35    # renovascular (renal artery involvement)
            drug_error = rng.random() < 0.40  # IVIG given instead of TNF inhibitor
            dx_delayed = rng.random() < 0.70  # stroke in child → extensive workup before DADA2
            transplant = rng.random() < 0.15  # HSCT for severe/haematological
            adherent = rng.random() < 0.70
        elif g == "NOD2":
            subtype = rng.choice(["Blau R334W", "Blau R334Q", "Blau L469F", "EOS de novo"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[30, 40, 30])[0]
            esrd = rng.random() < 0.05   # renal granulomata rare
            htn = rng.random() < 0.15
            drug_error = rng.random() < 0.22  # uveitis missed because painless
            dx_delayed = rng.random() < 0.55  # labelled JIA initially
            transplant = rng.random() < 0.03
            adherent = rng.random() < 0.72
        else:  # PSMB8
            subtype = rng.choice(["PRAAS p.Gly201Val (Japanese)", "CANDLE compound het", "JMP variant", "PRAAS other PSMB8"])
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[15, 30, 55])[0]
            esrd = rng.random() < 0.07   # metabolic nephropathy from lipodystrophy
            htn = rng.random() < 0.22
            drug_error = rng.random() < 0.28  # live vaccine given on JAK inhibitor
            dx_delayed = rng.random() < 0.75  # extremely rare; often labelled panniculitis
            transplant = rng.random() < 0.10  # HSCT attempted
            adherent = rng.random() < 0.68

        patients.append({
            "id": f"{g}-{seed}-{i+1:03d}",
            "gene": g,
            "age": age,
            "sex": sex,
            "subtype": subtype,
            "severity": severity,
            "esrd": esrd,
            "hypertension": htn,
            "drug_error": drug_error,
            "dx_delayed": dx_delayed,
            "transplant": transplant,
            "adherent": adherent,
        })
    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    return {
        "n": n,
        "esrd_pct": round(100 * sum(p["esrd"] for p in patients) / n, 1),
        "htn_pct": round(100 * sum(p["hypertension"] for p in patients) / n, 1),
        "drug_error_pct": round(100 * sum(p["drug_error"] for p in patients) / n, 1),
        "dx_delayed_pct": round(100 * sum(p["dx_delayed"] for p in patients) / n, 1),
        "transplant_pct": round(100 * sum(p["transplant"] for p in patients) / n, 1),
        "adherent_pct": round(100 * sum(p["adherent"] for p in patients) / n, 1),
        "severity": {
            "Mild":     round(100 * sum(p["severity"] == "Mild"     for p in patients) / n, 1),
            "Moderate": round(100 * sum(p["severity"] == "Moderate" for p in patients) / n, 1),
            "Severe":   round(100 * sum(p["severity"] == "Severe"   for p in patients) / n, 1),
        },
    }


def _build_all_patients():
    all_patients = []
    for idx, gene in enumerate(AUTOINFLAMMATORY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(gene, seed=seed, n=40)
        all_patients.extend(cohort)
    return all_patients


ALL_PATIENTS = _build_all_patients()


def get_overview() -> dict:
    n = len(ALL_PATIENTS)
    agg = _cohort_stats(ALL_PATIENTS)
    return {
        "atlas_name": "Autoinflammatory-Atlas",
        "atlas_subtitle": (
            "Complete 8-Gene Hereditary Autoinflammatory Disorder Atlas — "
            "MEFV · NLRP3 · TNFRSF1A · MVK · IL1RN · CECR1 · NOD2 · PSMB8"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1174–1181",
        "description": (
            "Comprehensive hereditary autoinflammatory reference covering the 8 most clinically "
            "significant monogenic autoinflammatory disorders: "
            "FMF (MEFV — colchicine prevents AA amyloidosis; M694V/M694V most severe); "
            "CAPS spectrum FCAS/MWS/NOMID (NLRP3 GOF — canakinumab FDA 2016; live vaccines CI); "
            "TRAPS (TNFRSF1A — long attacks >5d; corticosteroids worsen long-term; "
            "canakinumab preferred); "
            "MKD/HIDS (MVK — cervical lymphadenopathy; statins WORSEN; canakinumab FDA 2021); "
            "DIRA (IL1RN — neonatal sterile osteomyelitis; anakinra ESSENTIALLY CURATIVE); "
            "DADA2 (CECR1 — childhood stroke; TNF inhibitors PREVENT stroke; IVIG ineffective); "
            "Blau syndrome (NOD2 GOF — granulomata triad; uveitis BLINDNESS risk; "
            "surveillance mandatory); "
            "PRAAS/CANDLE (PSMB8 — lipodystrophy + panniculitis + IFN signature; "
            "JAK inhibitors transformative). "
            "320-patient aggregate cohort, 8 × 40 patients, seeds 1174–1181."
        ),
        "drug_alerts": [
            {
                "type": "danger",
                "title": "NLRP3 / IL1RN / PSMB8: LIVE VACCINES ABSOLUTELY CONTRAINDICATED on IL-1 inhibitors / JAK inhibitors",
                "body": (
                    "Canakinumab (anti-IL-1β), anakinra (IL-1Ra), and JAK inhibitors (baricitinib, "
                    "tofacitinib) — LIVE VACCINES ABSOLUTELY CONTRAINDICATED. "
                    "Complete ALL live vaccines (MMR, varicella, BCG, yellow fever, nasal flu) "
                    "BEFORE starting any of these biologics. "
                    "Inactivated/killed vaccines (flu shot, pneumococcal, meningococcal) are safe during treatment."
                ),
            },
            {
                "type": "danger",
                "title": "CECR1 (DADA2): IVIG DOES NOT PREVENT STROKE — use TNF inhibitors",
                "body": (
                    "In DADA2, childhood lacunar stroke is the most feared complication. "
                    "IVIG is INEFFECTIVE for stroke prevention in DADA2. "
                    "TNF inhibitors (etanercept or adalimumab) dramatically reduce stroke recurrence. "
                    "Start immediately when DADA2 is diagnosed with any vasculitis or stroke manifestation. "
                    "HSCT is potentially curative."
                ),
            },
            {
                "type": "danger",
                "title": "TNFRSF1A (TRAPS): LONG-TERM CORTICOSTEROIDS INCREASE ATTACK FREQUENCY",
                "body": (
                    "In TRAPS, corticosteroids reduce acute attack severity but INCREASE attack "
                    "frequency with long-term use — causing steroid dependence. "
                    "Use short-course only for acute attacks. "
                    "Transition to anakinra or canakinumab for maintenance. "
                    "Colchicine is NOT effective for TRAPS (unlike FMF)."
                ),
            },
            {
                "type": "danger",
                "title": "MVK (MKD/HIDS): STATINS WORSEN ATTACKS — ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "HMG-CoA reductase inhibitors (statins) further deplete GPP/GGPP downstream "
                    "of the MVK enzyme defect → worsens attack frequency and severity. "
                    "Do NOT prescribe statins for cardiovascular prevention in MVK/MKD patients. "
                    "Alternative lipid-lowering if needed."
                ),
            },
            {
                "type": "warning",
                "title": "MEFV (FMF): COLCHICINE — lifelong; prevents AA amyloidosis; SAFE in pregnancy",
                "body": (
                    "Colchicine prevents both FMF attacks and AA amyloidosis — "
                    "do NOT stop colchicine in pregnancy (stopping risks amyloidosis; "
                    "colchicine is safe in pregnancy — teratogenicity risk is theoretical, "
                    "NOT observed in clinical practice). "
                    "Avoid colchicine + cyclosporine combination (severe myopathy/rhabdomyolysis)."
                ),
            },
            {
                "type": "warning",
                "title": "NOD2 (Blau): UVEITIS — 3-6 monthly surveillance MANDATORY even when asymptomatic",
                "body": (
                    "Uveitis in Blau syndrome is INSIDIOUS — often asymptomatic until vision is lost. "
                    "Ophthalmology review every 3-6 months is mandatory even when the patient feels well. "
                    "Adalimumab (FDA approved) for refractory non-infectious uveitis. "
                    "ACE and calcium are usually NORMAL in Blau (elevated in pulmonary sarcoidosis — "
                    "critical DDx)."
                ),
            },
        ],
        "critical_rules": [
            "MEFV (FMF): COLCHICINE lifelong — prevents both attacks AND AA amyloidosis; "
            "safe in pregnancy; M694V/M694V = most severe; E148Q = low penetrance",
            "NLRP3 (CAPS): spectrum FCAS (cold-triggered) → MWS (SNHL) → NOMID (neonatal, "
            "meningitis, epiphyseal overgrowth); LIVE VACCINES ABSOLUTELY CI on canakinumab",
            "TNFRSF1A (TRAPS): attacks >5 days + migratory myalgia + periorbital oedema; "
            "CORTICOSTEROIDS WORSEN long-term — increase attack frequency; colchicine ineffective",
            "MVK (MKD/HIDS): cervical lymphadenopathy hallmark; statins ABSOLUTELY CI (deplete GGPP); "
            "urinary mevalonic acid elevated during fever = diagnostic",
            "IL1RN (DIRA): neonatal sterile osteomyelitis + pustular rash; "
            "anakinra ESSENTIALLY CURATIVE within 24-72h; FATAL without treatment",
            "CECR1 (DADA2): childhood lacunar stroke + livedo reticularis; "
            "IVIG DOES NOT PREVENT STROKE; TNF inhibitors (etanercept/adalimumab) prevent stroke; "
            "HSCT is curative",
            "NOD2 (Blau): AD GOF; triad arthritis + uveitis + skin granulomata; "
            "ACE NORMAL (unlike pulmonary sarcoid); OPHTHALMOLOGY 3-6 monthly mandatory",
            "PSMB8 (PRAAS): lipodystrophy + panniculitis + IFN signature; "
            "JAK inhibitors (baricitinib/tofacitinib) reduce IFN score within weeks; "
            "LIVE VACCINES CI on JAK inhibitors",
        ],
        "kpis": [
            {"label": "Total Patients",    "value": str(n)},
            {"label": "Genes Covered",     "value": "8"},
            {"label": "Drug Error Rate",   "value": f"{agg['drug_error_pct']}%"},
            {"label": "Delayed Dx Rate",   "value": f"{agg['dx_delayed_pct']}%"},
            {"label": "ESRD Rate",         "value": f"{agg['esrd_pct']}%"},
            {"label": "Transplant Rate",   "value": f"{agg['transplant_pct']}%"},
            {"label": "Severe Cases",      "value": f"{agg['severity']['Severe']}%"},
            {"label": "Adherent",          "value": f"{agg['adherent_pct']}%"},
        ],
        "aggregate_clinical": {
            "esrd_pct":                  agg["esrd_pct"],
            "hypertension_pct":          agg["htn_pct"],
            "transplant_rate_pct":       agg["transplant_pct"],
            "drug_error_pct":            agg["drug_error_pct"],
            "diagnosis_delayed_pct":     agg["dx_delayed_pct"],
            "surveillance_adherent_pct": agg["adherent_pct"],
            "severity_mild_pct":         agg["severity"]["Mild"],
            "severity_moderate_pct":     agg["severity"]["Moderate"],
            "severity_severe_pct":       agg["severity"]["Severe"],
        },
        "pathway_targets": {
            "MEFV":     "Pyrin inflammasome → IL-1β; colchicine (microtubule); "
                        "anakinra/canakinumab (IL-1 blockade) for colchicine-resistant",
            "NLRP3":    "NLRP3 inflammasome GOF → IL-1β/IL-18; "
                        "canakinumab (anti-IL-1β); rilonacept (IL-1 TRAP); MCC950 (investigational)",
            "TNFRSF1A": "TNFR1 shedding impaired → sustained NF-κB/MAPK; "
                        "canakinumab/anakinra (IL-1); etanercept (less effective)",
            "MVK":      "Mevalonate kinase → GGPP depletion → NLRP3 → IL-1β; "
                        "canakinumab FDA 2021; AVOID statins (deplete GGPP further)",
            "IL1RN":    "IL-1Ra absent → unopposed IL-1α/β → NF-κB; "
                        "anakinra (recombinant IL-1Ra) ESSENTIALLY CURATIVE; HSCT curative",
            "CECR1":    "ADA2 absent → M1 macrophage → NF-κB + endothelial dysfunction; "
                        "TNF inhibitors prevent stroke; HSCT curative",
            "NOD2":     "NOD2 GOF → constitutive NF-κB → granuloma; "
                        "MTX/AZA; TNF inhibitors (adalimumab); IL-6 inhibitors for vasculitis",
            "PSMB8":    "Immunoproteasome defect → UPR → IFN signature; "
                        "JAK inhibitors (baricitinib/tofacitinib) → JAK-STAT blockade",
        },
        "disease_category_breakdown": {
            "IL-1 Inflammasome (MEFV/NLRP3/MVK)":  37.5,
            "TNF/TNFR Pathway (TNFRSF1A)":          12.5,
            "IL-1Ra Deficiency (IL1RN)":             12.5,
            "ADA2/Vasculitis (CECR1)":               12.5,
            "NOD2/NF-κB Granulomatous (NOD2)":       12.5,
            "Interferonopathy/PRAAS (PSMB8)":        12.5,
        },
    }


def get_breakdown() -> dict:
    genes = []
    for idx, gene_def in enumerate(AUTOINFLAMMATORY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(gene_def, seed=seed, n=40)
        stats = _cohort_stats(cohort)
        g = gene_def["gene"]
        if g == "CECR1":
            inheritance = "AR (biallelic LOF); highly variable phenotype"
        elif g in ("NLRP3", "TNFRSF1A", "NOD2"):
            inheritance = "AD (GOF); de novo common in NOMID (NLRP3)" if g == "NLRP3" else "AD"
        elif g in ("MVK", "IL1RN", "PSMB8"):
            inheritance = "AR (biallelic LOF)"
        else:  # MEFV
            inheritance = "AR (most); low-penetrance AD (E148Q heterozygote)"
        genes.append({
            "gene":              gene_def["gene"],
            "protein":           gene_def["protein"],
            "alias":             gene_def["alias"],
            "aa":                gene_def["aa"],
            "kDa":               gene_def["kDa"],
            "locus":             gene_def["locus"],
            "omim_gene":         gene_def["omim_gene"],
            "omim_disease":      gene_def["omim_disease"],
            "gene_class":        gene_def["gene_class"],
            "inheritance":       inheritance,
            "phenotype":         gene_def["phenotype"],
            "hallmark":          gene_def["hallmark"],
            "treatment_alert":   gene_def["treatment_alert"],
            "key_ddx":           gene_def["key_ddx"],
            "gfr_pattern":       gene_def["gfr_pattern"],
            "proteinuria_pattern": gene_def["proteinuria_pattern"],
            "primary_complication": gene_def["primary_complication"],
            "disease_detail":    gene_def["disease"],
            "cohort_n":          40,
            "cohort_stats": {
                "esrd_pct":       stats["esrd_pct"],
                "htn_pct":        stats["htn_pct"],
                "drug_error_pct": stats["drug_error_pct"],
                "dx_delayed_pct": stats["dx_delayed_pct"],
                "transplant_pct": stats["transplant_pct"],
                "adherent_pct":   stats["adherent_pct"],
                "severity":       stats["severity"],
            },
        })
    return {"genes": genes, "n_genes": len(genes)}


def get_definitions() -> list:
    return [
        {
            "term": "AA Amyloidosis",
            "full": "Amyloid A amyloidosis — long-term complication of autoinflammatory disorders",
            "explanation": (
                "Serum amyloid A (SAA), an acute-phase reactant, deposits in organs (kidneys #1, "
                "spleen, liver) as amyloid fibrils when chronically elevated; "
                "SAA → amyloid A (AA) → apple-green birefringence on Congo red staining under "
                "polarised light (PATHOGNOMONIC for amyloid); "
                "renal amyloid → proteinuria → nephrotic syndrome → ESRD; "
                "in FMF: COMPLETELY PREVENTABLE by lifelong colchicine; "
                "SAA monitoring every 6-12 months is the standard of care; "
                "IL-1 blockade (anakinra/canakinumab) can stabilise and partially reverse "
                "early amyloid even after establishment."
            ),
        },
        {
            "term": "Colchicine Mechanism in FMF",
            "full": "Colchicine — microtubule polymerisation inhibitor; FMF first-line",
            "explanation": (
                "Colchicine binds tubulin dimers → prevents microtubule polymerisation → "
                "disrupts neutrophil chemotaxis (PMN cannot migrate to peritoneum) and "
                "pyrin inflammasome assembly (disrupts cytoskeletal scaffold required for "
                "ASC speck formation); "
                "dose 1-2 mg/day (max 3 mg); "
                "reduces attack frequency ≥65%; "
                "PREVENTS AA amyloidosis — near-complete protection with adherent use; "
                "SAFE IN PREGNANCY (do not stop; risk of amyloidosis without it); "
                "avoid with cyclosporine (myopathy/rhabdomyolysis); "
                "avoid with macrolide antibiotics (clarithromycin) — increased colchicine levels."
            ),
        },
        {
            "term": "Canakinumab (Ilaris)",
            "full": "Anti-IL-1β monoclonal antibody — CAPS, TRAPS, MKD, FMF",
            "explanation": (
                "Fully human anti-IL-1β IgG1 monoclonal antibody (Novartis); "
                "FDA approved 2009 for systemic JIA; 2016 for CAPS (FCAS/MWS/NOMID); "
                "2020 for TRAPS, MKD/HIDS, colchicine-resistant FMF; "
                "dose: 150-300 mg SC every 4-8 weeks (condition dependent); "
                "onset: clinical response within 1-3 days; normalises CRP/SAA; "
                "LIVE VACCINES ABSOLUTELY CONTRAINDICATED — complete before starting; "
                "TB screening mandatory; hepatitis B/C; "
                "does NOT block IL-1α (unlike anakinra which blocks both IL-1α and IL-1β)."
            ),
        },
        {
            "term": "Anakinra (Kineret)",
            "full": "Recombinant IL-1 receptor antagonist — DIRA, FMF, CAPS, TRAPS",
            "explanation": (
                "Recombinant human IL-1Ra (interleukin-1 receptor antagonist); "
                "blocks both IL-1α AND IL-1β at the IL-1 receptor type I (IL-1RI); "
                "dose: 100 mg SC daily (adults); 1-4 mg/kg/day in neonates (DIRA); "
                "short half-life (~4-6 hours) — useful when rapid offset needed (infections, surgery); "
                "FDA approved for rheumatoid arthritis; DIRA (off-label but standard of care); "
                "colchicine-resistant FMF; acute gout (off-label); "
                "LIVE VACCINES ABSOLUTELY CONTRAINDICATED; "
                "DIRA: ESSENTIALLY CURATIVE — response within 24-72 hours of starting."
            ),
        },
        {
            "term": "CAPS Spectrum (FCAS/MWS/NOMID)",
            "full": "Cryopyrin-Associated Periodic Syndrome — NLRP3 GOF spectrum",
            "explanation": (
                "CAPS is a SPECTRUM of three phenotypes caused by NLRP3 gain-of-function variants: "
                "FCAS (Familial Cold Autoinflammatory Syndrome) — mildest; cold-triggered systemic "
                "urticaria + fever + arthralgia; resolves within 24h; NO SNHL; "
                "MWS (Muckle-Wells Syndrome) — intermediate; episodic NOT cold-triggered; "
                "progressive SNHL (50%); amyloidosis risk 25%; "
                "NOMID/CINCA (Neonatal Onset Multisystem Inflammatory Disease) — most severe; "
                "neonatal onset; TRIAD: chronic urticaria + aseptic meningitis + arthropathy "
                "with epiphyseal overgrowth; SNHL (90%); visual loss; intellectual disability; "
                "de novo mutations in ~75% NOMID; deep sequencing needed if Sanger negative."
            ),
        },
        {
            "term": "Interferon Signature (IFN Score)",
            "full": "Elevated ISG expression — marker of type I/II interferonopathy including PRAAS",
            "explanation": (
                "Interferon-stimulated genes (ISGs) including ISG15, IFIT1, IFIT2, MX1, OASL, IFIT3 "
                "are quantified by RT-PCR or RNA-seq on whole blood; "
                "IFN score >2 standard deviations above healthy controls = ELEVATED; "
                "elevated in PRAAS (PSMB8/PSMB9/PSMA3), Aicardi-Goutières syndrome, "
                "STING-associated vasculopathy (SAVI), SLE, systemic JIA, DADA2; "
                "in PRAAS: IFN score is a TREATMENT MONITORING TOOL — "
                "normalises within weeks of baricitinib/JAK inhibitor start; "
                "a persistently high IFN score despite JAK inhibitor = insufficient dose or "
                "wrong diagnosis; "
                "NOT elevated in FMF, TRAPS, NLRP3 CAPS (IL-1-dominated, not IFN-dominated)."
            ),
        },
        {
            "term": "Migratory Myalgia (TRAPS)",
            "full": "Centrifugal migratory myalgia — TRAPS pathognomonic feature",
            "explanation": (
                "In TRAPS, the myalgia MIGRATES from proximal muscle groups (thigh, shoulder girdle) "
                "to distal (calf, forearm) over the course of the attack — a centrifugal pattern; "
                "the overlying skin becomes erythematous and warm (reactive erythema over affected muscle); "
                "area 20-40 cm; persists for days; "
                "this combination (fever + migratory myalgia + overlying erythema) is "
                "PATHOGNOMONIC for TRAPS when combined with attack duration >5 days; "
                "PERIORBITAL OEDEMA (unilateral or bilateral) also characteristic — "
                "seen in 80% of high-penetrance variant attacks; "
                "absent in FMF (key distinguishing feature)."
            ),
        },
        {
            "term": "Urinary Mevalonic Acid (MKD Diagnostic)",
            "full": "Elevated urinary mevalonic acid during fever — MKD/HIDS diagnostic",
            "explanation": (
                "Mevalonate kinase (MVK) catalyses mevalonic acid → mevalonate-5-phosphate; "
                "LOF → mevalonic acid accumulates → excreted in urine; "
                "measurement: GC-MS (gas chromatography-mass spectrometry) urine organic acids; "
                "collect DURING a fever attack for highest sensitivity; "
                "between attacks: may normalise (reduces sensitivity); "
                "ALSO elevated in mevalonic aciduria (MVA) — severe end of spectrum; "
                "IgD >100 IU/mL (historical 'HIDS' marker): present in 80% but NOT specific "
                "(also elevated in FMF/TRAPS); IgD alone insufficient for diagnosis."
            ),
        },
        {
            "term": "Blau Syndrome vs Pulmonary Sarcoidosis (ACE / Calcium)",
            "full": "NOD2 Blau — ACE normal, calcium normal; pulmonary sarcoid — ACE elevated, hypercalcaemia",
            "explanation": (
                "Blau syndrome (NOD2 GOF) and sarcoidosis share histological non-caseating "
                "epithelioid granulomata, but are distinguished by: "
                "ACE (angiotensin-converting enzyme): NORMAL in Blau; elevated in 60-80% of "
                "active pulmonary sarcoidosis (macrophage-derived ACE from granulomata); "
                "CALCIUM: NORMAL in Blau; hypercalcaemia in 10-30% sarcoidosis "
                "(macrophage 1α-hydroxylase converts 25-OHD → 1,25-OHD → increased gut Ca absorption); "
                "CHEST: no bilateral hilar lymphadenopathy in Blau; "
                "AGE: Blau onset <4 years; sarcoid onset typically adult; "
                "GENETICS: NOD2 R334W/Q GOF in Blau; NOD2 LOF variants in Crohn (different locus!)"
            ),
        },
        {
            "term": "DADA2 Stroke Prevention (TNF inhibitors)",
            "full": "CECR1 (ADA2) — TNF inhibitors prevent childhood lacunar stroke; IVIG does not",
            "explanation": (
                "In DADA2 (deficiency of adenosine deaminase 2), lacunar infarcts in deep grey matter "
                "(basal ganglia, thalamus, brainstem) are the most feared complication — "
                "can occur from toddler age and recur; "
                "mechanism: ADA2 absence → M1 macrophage polarisation → TNF-α-mediated "
                "endothelial injury → small-vessel vasculitis → thrombotic/ischaemic infarcts; "
                "IVIG: NO BENEFIT for stroke prevention; addresses hypogammaglobulinaemia only; "
                "do NOT substitute IVIG for TNF inhibitor in stroke-prone DADA2; "
                "ETANERCEPT or ADALIMUMAB: reduces stroke recurrence from ~70% to <10% in series; "
                "start immediately when DADA2 diagnosed with vasculitis/stroke phenotype; "
                "HSCT: curative by providing donor-derived ADA2; consider early in severe cases."
            ),
        },
    ]
