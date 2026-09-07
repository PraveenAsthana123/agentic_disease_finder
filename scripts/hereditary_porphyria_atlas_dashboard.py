#!/usr/bin/env python3
"""Hereditary-Porphyria-Atlas — Complete 8-Gene Hereditary Porphyria Atlas
HMBS    (hydroxymethylbilane synthase; 361 aa; 11q23.3; AD;
         Acute Intermittent Porphyria — most common acute porphyria worldwide;
         >300 drug triggers; hemin (Panhematin) IV first-line acute attack;
         givosiran siRNA FDA 2019 monthly prevention; seed SEED_BASE+0) ·
CPOX    (coproporphyrinogen oxidase; 454 aa; 3q11.2; AD;
         Hereditary Coproporphyria — acute neurovisceral + blistering skin;
         same drug triggers as AIP; hemin acute; givosiran prevention; seed SEED_BASE+1) ·
PPOX    (protoporphyrinogen oxidase; 477 aa; 1q23.3; AD;
         Variegate Porphyria — R59W founder South Africa; acute + skin;
         do NOT give local anaesthetics containing prilocaine; seed SEED_BASE+2) ·
UROS    (uroporphyrinogen III synthase; 265 aa; 10q25.2; AR;
         Congenital Erythropoietic Porphyria — Gunther disease; severe mutilating
         blistering; pink/red urine and teeth PATHOGNOMONIC; HSCT curative; seed SEED_BASE+3) ·
FECH    (ferrochelatase; 423 aa; 18q21.31; AR/low-penetrance;
         Erythropoietic Protoporphyria — burning pain within minutes of sunlight PATHOGNOMONIC;
         NO blistering; liver failure 2–5%; afamelanotide SC FDA 2019; seed SEED_BASE+4) ·
ALAS2   (5-aminolevulinic acid synthase 2; 587 aa; Xp11.21; XLD-GOF/XLR-LOF;
         X-linked Protoporphyria — GOF identical to EPP clinically; LOF sideroblastic anaemia;
         afamelanotide for XLP-GOF; phlebotomy for XLSA-LOF; seed SEED_BASE+5) ·
UROD    (uroporphyrinogen decarboxylase; 367 aa; 1p34.1; AD;
         Porphyria Cutanea Tarda — most common porphyria overall; blistering photosensitivity;
         phlebotomy + low-dose hydroxychloroquine FIRST-LINE; alcohol/oestrogen triggers; seed SEED_BASE+6) ·
ALAD    (5-aminolevulinic acid dehydratase; 330 aa; 9q32; AR;
         ALA-Dehydratase Deficiency Porphyria — rarest human porphyria (<10 reported cases);
         AR inheritance (unlike all other acute porphyrias which are AD); lead inhibits ALAD; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1758–1765)
"""

import random

SEED_BASE = 1758

PORPHYRIA_GENES = [
    # ── HMBS — Acute Intermittent Porphyria ──────────────────────────────────
    {
        "gene": "HMBS",
        "protein": (
            "HMBS — 11q23.3 AD — Hydroxymethylbilane-Synthase-361aa — "
            "Acute-Intermittent-Porphyria-Most-Common-Acute-Porphyria — "
            "Neurovisceral-Attacks-PATHOGNOMONIC — Port-Wine-Urine-During-Attack — "
            "Hemin-IV-First-Line-Acute — Givosiran-siRNA-FDA2019-Monthly-Prevention — "
            ">300-Drug-Triggers-Mandatory-Check"
        ),
        "alias": (
            "HMBS (hydroxymethylbilane synthase / porphobilinogen deaminase); OMIM gene 609806; "
            "Acute Intermittent Porphyria (AIP) OMIM 176000. "
            "11q23.3; 361 aa; ~42 kDa; autosomal dominant; penetrance ~10–20% (most carriers never attack). "
            "FUNCTION: HMBS catalyses the third step of haem synthesis — "
            "condensation of four porphobilinogen (PBG) monomers → hydroxymethylbilane (HMB); "
            "HMBS haploinsufficiency → ↓ HMB synthetic capacity → "
            "when ALAS1 (rate-limiting enzyme) is induced, ALA and PBG accumulate → neurotoxicity. "
            "MUTATION SPECTRUM: >400 pathogenic variants; missense, nonsense, frameshift, splice; "
            "the W283X variant is a common founder in Northern Europe; "
            "essentially all functional — no single dominant variant globally; "
            "90% of HMBS mutation carriers are ASYMPTOMATIC throughout life — penetrance ~10–20%; "
            "attacks occur only when a second trigger induces ALAS1 (see below). "
            "TRIGGERS (INDUCING ALAS1 → ALA/PBG ACCUMULATION): "
            "DRUGS — most important: >300 drugs classified; critical triggers: "
            "barbiturates (HIGH RISK — absolute prohibition in AIP), rifampicin, "
            "carbamazepine, phenytoin, phenobarbital, valproate (HIGH RISK), primidone, "
            "rifamycins, ergots, griseofulvin, certain sulfonamides, "
            "progestogens (including combined OCP progesterone component), estrogen-containing OCP; "
            "SAFE anticonvulsants: gabapentin, pregabalin, levetiracetam, lacosamide; "
            "SAFE antibiotics: amoxicillin, penicillins, erythromycin, doxycycline; "
            "ALWAYS check the European Porphyria Network (EPNET) drug database before prescribing; "
            "HORMONAL: natural progesterone rise in luteal phase → cyclical pre-menstrual attacks; "
            "oestrogen and progesterone both potential triggers; GnRH analogues for cyclical AIP; "
            "FASTING/CALORIC RESTRICTION: reduced glucose → ↑ PGC-1α → ↑ ALAS1; "
            "carbohydrate loading (glucose IV/oral) reduces ALAS1 — traditional 'glucose effect'; "
            "SURGERY: fasting + stress → attack; schedule surgery with IV glucose + porphyria team; "
            "INFECTIONS: any systemic illness can trigger; "
            "ALCOHOL: acute ingestion induces ALAS1. "
            "CLINICAL ATTACK (neurovisceral): "
            "Severe abdominal pain (visceral neuropathy): 90% of attacks; colicky; no peritoneal signs; "
            "normal or decreased bowel sounds; frequently misdiagnosed as 'surgical abdomen'; "
            "Tachycardia, hypertension (autonomic neuropathy): 60–80% of attacks; "
            "Dark/port-wine urine: darkening on standing (PBG oxidises to porphyrins → dark pigments); "
            "Motor neuropathy: acute flaccid paralysis (can be severe — respiratory paralysis); "
            "Psychiatric: anxiety, agitation, confusion, psychosis; "
            "Hyponatraemia: SIADH (syndrome of inappropriate ADH) — serum Na can fall to <115 mmol/L → seizures; "
            "Seizures: paradox — almost ALL anticonvulsants trigger AIP (except levetiracetam/lacosamide); "
            "levetiracetam IV is the ONLY safe first-line in AIP-related seizures; "
            "DIAGNOSIS DURING ATTACK: "
            "Urinary PBG (spot or 24h) — MARKEDLY elevated (often >50× ULN) during attack; "
            "Urinary ALA also elevated; "
            "Urinary PBG normalises between attacks — DON'T rely on inter-attack specimens; "
            "Faecal porphyrins: normal in AIP (key DDx from VP and HCP which show raised faecal); "
            "Plasma porphyrins: may be elevated but less specific; "
            "HMBS erythrocyte enzyme activity: reduced ~50% in most variants (not universal); "
            "HMBS molecular genetic testing: definitive. "
            "TREATMENT: "
            "ACUTE ATTACK: "
            "1. STOP all triggering drugs IMMEDIATELY; "
            "2. IV haemin (hemin, Panhematin): 3–4 mg/kg/day IV × 4 days — represses ALAS1 → ↓ ALA/PBG; "
            "haem arginate (Normosang) — preferred in Europe (more stable); "
            "give via central line (phlebitis); administer within 24h of severe attack onset; "
            "3. IV glucose (300–400 g/day or 10% dextrose at ≥2 mg/kg/min) if hemin not available or mild; "
            "4. IV pain management: opioids (morphine) safe; "
            "5. Beta-blockers for tachycardia/hypertension (safe); "
            "6. Hyponatraemia: fluid restrict + treat SIADH; "
            "7. Seizures: IV levetiracetam (safe); magnesium sulphate for eclamptic pattern; "
            "8. Respiratory paralysis: ICU, intubation, ventilation; "
            "PREVENTION: "
            "Givosiran (GIVLAARI, Alnylam) SC monthly: siRNA targeting ALAS1 mRNA → ↓ hepatic ALAS1 → "
            "↓ ALA and PBG even with triggers; FDA approved 2019; EMA approved 2020; "
            "for recurrent acute attacks (≥2 attacks/year requiring hospitalisation or hemin); "
            "adverse effects: hepatotoxicity (↑ transaminases — monitor monthly LFTs first year), "
            "homocysteine elevation (supplement B6), injection site reactions; "
            "Haem arginate prophylaxis: weekly IV before menstrual trigger cycle in cyclical AIP; "
            "GnRH analogues (leuprolide, buserelin): for intractable cyclical pre-menstrual attacks; "
            "add-back oestrogen (NOT progestogen) for bone protection; "
            "Liver transplantation: curative in severe recurrent AIP (hepatic HMBS replaced); "
            "long-term neuropathic damage may not reverse post-transplant; "
            "LONG-TERM COMPLICATIONS: "
            "Chronic pain syndrome (peripheral neuropathy); "
            "Hypertension (chronic sympathetic activation); "
            "CKD (tubular damage from ALA — monitor eGFR annually); "
            "Hepatocellular carcinoma (↑ risk vs population — annual liver ultrasound from age 50 in AIP)."
        ),
        "locus": "11q23.3",
        "aa": 361,
        "kDa": 42,
        "omim_gene": "609806",
        "omim_disease": "176000",
        "inheritance": "AD; haploinsufficiency; penetrance ~10–20%; attacks require second trigger",
        "gene_class": "Haem Biosynthesis Step 3 — PBG Deaminase / HMBS — ALA/PBG Accumulation",
        "key_alerts": [
            "HMBS-BARBITURATES-ABSOLUTE-PROHIBITION: barbiturates (thiopental, phenobarbital, primidone) are HIGH RISK triggers for acute attacks in AIP — a single dose of thiopental can trigger a potentially lethal attack; anaesthetists MUST check AIP status before any GA; use propofol or ketamine instead",
            "HMBS-DRUG-CHECK-EPNET-MANDATORY: >300 drugs trigger AIP by inducing ALAS1; always check the EPNET/American Porphyria Foundation drug database before any new prescription; critically: rifampicin, CBZ, PHT, VPA, progesterone-OCP are HIGH RISK; levetiracetam and gabapentin are SAFE anticonvulsants",
            "HMBS-GIVOSIRAN-FDA2019-PREVENTION: givosiran (GIVLAARI) SC monthly siRNA targeting ALAS1 mRNA is FDA/EMA approved for prevention of recurrent acute porphyria attacks; indicates ≥2 attacks/year; monitor LFTs monthly (Year 1) and homocysteine; NOT for acute treatment",
            "HMBS-HYPONATRAEMIA-SIADH-SEVERE: SIADH is a major complication of acute AIP attack (SIADH → Na can fall to <115 mmol/L → seizures); treat with fluid restriction ± hypertonic saline; seizures in AIP require IV levetiracetam (almost all other AEDs trigger AIP)",
        ],
        "etiologies": {
            "Classic_Neurovisceral_AIP": {"pct": 68, "phenotype": "abdominal pain + autonomic neuropathy + port-wine urine — drug/hormone triggered"},
            "Cyclical_Premenstrual_AIP": {"pct": 22, "phenotype": "luteal-phase progesterone triggers recurrent attacks; GnRH agonist therapy"},
            "Silent_Carrier_AIP": {"pct": 82, "phenotype": "HMBS mutation carrier — never attacks in lifetime; penetrance ~10–20%"},
            "Severe_Motor_Neuropathy_AIP": {"pct": 10, "phenotype": "acute flaccid paralysis, respiratory involvement — ICU, prolonged recovery"},
        },
        "stats": {
            "mean_onset_age_y": 28.5,
            "mean_dx_delay_months": 18.2,
            "attack_hospitalisation_rate_pct": 88,
            "hemin_treated_pct": 74,
        },
        "dx_delay_distribution": {"<6mo": 28, "6-24mo": 42, ">24mo": 30},
        "patients": [],
    },
    # ── CPOX — Hereditary Coproporphyria ─────────────────────────────────────
    {
        "gene": "CPOX",
        "protein": (
            "CPOX — 3q11.2 AD — Coproporphyrinogen-Oxidase-454aa — "
            "Hereditary-Coproporphyria-Acute-AND-Cutaneous — "
            "Same-Drug-Triggers-as-AIP — "
            "Raised-Faecal-Coproporphyrin-III-KEY-DDx-from-AIP — "
            "Hemin-Acute-Givosiran-Prevention — "
            "Blistering-Skin-30pct-of-Attacks"
        ),
        "alias": (
            "CPOX (coproporphyrinogen oxidase); OMIM gene 612732; "
            "Hereditary Coproporphyria (HCP) OMIM 121300. "
            "3q11.2; 454 aa; ~50 kDa; autosomal dominant; penetrance ~1–5% (most carriers asymptomatic). "
            "FUNCTION: CPOX catalyses step 6 of haem synthesis — "
            "oxidative decarboxylation of coproporphyrinogen III → protoporphyrinogen IX; "
            "CPOX haploinsufficiency → accumulation of coproporphyrinogen III (and its oxidised product "
            "coproporphyrin III) in all tissues including skin → photosensitive skin lesions. "
            "KEY BIOCHEMICAL DDX FROM AIP: "
            "AIP: ↑↑ urinary ALA + PBG; faecal porphyrins NORMAL; "
            "HCP: ↑↑ urinary ALA + PBG (during attack) + ↑↑ faecal coproporphyrin III (ALWAYS elevated, "
            "even inter-attack) + urinary coproporphyrin III elevated; "
            "VP: ↑↑ urinary ALA + PBG (during attack) + ↑↑ faecal protoporphyrin + coproporphyrin; "
            "plasma porphyrin fluorescence peak at 626 nm (VP) vs 619 nm (HCP) — spectroscopy distinction. "
            "CLINICAL FEATURES: "
            "Neurovisceral attacks: IDENTICAL to AIP — abdominal pain, autonomic, motor neuropathy, SIADH; "
            "Drug triggers: IDENTICAL to AIP (barbiturates, rifampicin, CBZ/PHT/PB/VPA, progestogens); "
            "Skin manifestations (30% of attacks): "
            "blistering photosensitive dermatitis on sun-exposed skin (hands, face); "
            "subepidermal bullae with milia; fragile skin; scarring; hypertrichosis; "
            "same mechanism as PCT (UROD) — excess porphyrins in skin photosensitise and generate ROS; "
            "CHARACTERISTIC: HCP can present as isolated skin disease without neurovisceral attacks; "
            "DIAGNOSIS: "
            "Urinary PBG + ALA (during attack — same as AIP); "
            "Faecal porphyrins: ↑ coproporphyrin III (HCP) vs ↑ protoporphyrin + coproporphyrin (VP); "
            "Plasma porphyrin fluorescence emission scan: peak at 619 nm in HCP (626 nm in VP); "
            "24h urine coproporphyrin III: elevated, especially Type III isomer; "
            "CPOX erythrocyte enzyme activity: ~50% of normal; "
            "CPOX molecular testing: definitive. "
            "TREATMENT: "
            "Acute attacks: identical to AIP — haemin IV + IV glucose + supportive care; "
            "Prevention: givosiran SC monthly (same indication as AIP); "
            "Skin: photoprotection (high-factor sunblock, protective clothing); "
            "avoid vasodilatory alcohol which worsens skin photosensitivity; "
            "cholestyramine (binds porphyrins in gut — reduces faecal porphyrin load and skin symptoms); "
            "SAFETY IN PREGNANCY: "
            "acute attacks can be life-threatening in pregnancy; "
            "haem arginate is used in pregnancy (case series — safer than haemin/Panhematin formulation); "
            "progesterone in pregnancy is a major trigger — close monitoring; "
            "givosiran: teratogenicity unknown — generally avoided in pregnancy (use haem arginate)."
        ),
        "locus": "3q11.2",
        "aa": 454,
        "kDa": 50,
        "omim_gene": "612732",
        "omim_disease": "121300",
        "inheritance": "AD; haploinsufficiency; penetrance ~1–5%; triggers required for attack",
        "gene_class": "Haem Biosynthesis Step 6 — Coproporphyrinogen III Oxidase — Dual Acute+Cutaneous",
        "key_alerts": [
            "CPOX-FAECAL-COPROPORPHYRIN-KEY-DDX: ↑ faecal coproporphyrin III distinguishes HCP from AIP (which has NORMAL faecal porphyrins) — mandatory faecal porphyrin measurement in all acute porphyria workup; this test is inter-attack stable so can be done outside an attack",
            "CPOX-SAME-DRUG-TRIGGERS-AIP: HCP shares ALL drug triggers with AIP — check EPNET database; barbiturates, rifampicin, CBZ, PHT, VPA, progestogens are HIGH RISK; give same drug-safety education as AIP patients",
            "CPOX-SKIN-BLISTERING-PHOTOPROTECTION: 30% of HCP patients develop blistering photosensitivity similar to PCT — SUNBLOCK factor 50+ and protective clothing from the first diagnosis; alcohol and oestrogen worsen skin; cholestyramine binds gut porphyrins",
            "CPOX-PLASMA-FLUORESCENCE-619nm-VP-626nm: plasma porphyrin fluorescence emission scan distinguishes HCP (619 nm peak) from VP (626 nm peak) — critical when faecal porphyrins are borderline; request at specialist porphyria centre",
        ],
        "etiologies": {
            "Acute_Neurovisceral_HCP": {"pct": 70, "phenotype": "attacks identical to AIP — drug/hormone triggered; faecal coproporphyrin always elevated"},
            "Mixed_Acute_Cutaneous_HCP": {"pct": 30, "phenotype": "neurovisceral attack + blistering photosensitive skin; skin may be sole manifestation"},
            "Isolated_Cutaneous_HCP": {"pct": 12, "phenotype": "blistering skin without acute attacks — rare; diagnosed on faecal/urine porphyrin profile"},
            "Silent_Carrier_HCP": {"pct": 85, "phenotype": "carrier never attacks; faecal coproporphyrin III elevated on biochemistry but asymptomatic"},
        },
        "stats": {
            "mean_onset_age_y": 32.1,
            "mean_dx_delay_months": 22.4,
            "skin_blistering_rate_pct": 30,
            "hemin_treated_pct": 70,
        },
        "dx_delay_distribution": {"<6mo": 22, "6-24mo": 38, ">24mo": 40},
        "patients": [],
    },
    # ── PPOX — Variegate Porphyria ────────────────────────────────────────────
    {
        "gene": "PPOX",
        "protein": (
            "PPOX — 1q23.3 AD — Protoporphyrinogen-Oxidase-477aa — "
            "Variegate-Porphyria-Acute-AND-Skin — "
            "R59W-South-Africa-1-in-300-Founder-PATHOGNOMONIC — "
            "Plasma-Fluorescence-626nm-DISTINGUISHES-from-HCP-619nm — "
            "Hemin-IV-Acute — Givosiran-Prevention — "
            "Skin-More-Prominent-than-HCP"
        ),
        "alias": (
            "PPOX (protoporphyrinogen oxidase); OMIM gene 600923; "
            "Variegate Porphyria (VP) OMIM 176200. "
            "1q23.3; 477 aa; ~51 kDa; autosomal dominant; penetrance ~20% in South Africa (high R59W carrier frequency). "
            "FUNCTION: PPOX catalyses step 7 of haem synthesis — "
            "oxidation of protoporphyrinogen IX → protoporphyrin IX; "
            "PPOX haploinsufficiency → accumulation of protoporphyrinogen (and other upstream porphyrins) → "
            "excreted as faecal protoporphyrin (dominant) and coproporphyrin; "
            "protoporphyrin in skin → photosensitisation (same mechanism as EPP). "
            "SOUTH AFRICAN R59W FOUNDER: "
            "A single founder mutation (p.Arg59Trp) was introduced to South Africa by a Dutch settler "
            "around 1688 — carrier frequency in Afrikaner population: 1 in 300; "
            "VP was dubbed 'the Royal Malady' because King George III of Great Britain may have had VP "
            "(historical debate — not definitively proven); "
            "R59W: substitutes a fully conserved arginine — impairs FAD binding at the active site. "
            "BIOCHEMICAL SIGNATURE: "
            "Faecal porphyrins: ↑↑ protoporphyrin + coproporphyrin (both elevated, protoporphyrin dominant); "
            "contrasts with HCP: coproporphyrin III dominant; "
            "Plasma porphyrin fluorescence: 626 nm peak (PATHOGNOMONIC for VP — same as AIP in attack "
            "but plasma peak 626 nm is VP, 619 nm is HCP — stable inter-attack); "
            "Urinary ALA + PBG: elevated during attack (same as AIP), near-normal inter-attack; "
            "urinary coproporphyrin III: intermediate between AIP and HCP. "
            "CLINICAL FEATURES: "
            "Neurovisceral attacks: identical to AIP and HCP — trigger-dependent; "
            "Cutaneous photosensitivity: PROMINENT in VP — often more severe than HCP; "
            "fragile skin on dorsum of hands + face; subepidermal bullae; milia; hyperpigmentation; "
            "can persist even without acute neurovisceral symptoms; "
            "DISTINCTIVE: cutaneous disease in VP can be entirely independent of acute neurovisceral disease; "
            "DRUG TRIGGERS: identical to AIP/HCP — EPNET drug database mandatory; "
            "prilocaine (local anaesthetic) is specific additional risk in VP; "
            "ANAESTHETIC CONSIDERATIONS: "
            "avoid thiopental, etomidate (both CONTRAINDICATED in VP); "
            "propofol is safe; isoflurane is safe; halothane causes hepatic CYP induction — use with caution; "
            "DIAGNOSIS: "
            "Plasma porphyrin fluorescence emission: 626 nm stable peak (most reliable inter-attack test); "
            "Faecal porphyrins: ↑ protoporphyrin (dominant) + coproporphyrin; "
            "PPOX enzyme assay: ~50% activity; "
            "PPOX molecular testing: definitive; "
            "TREATMENT: "
            "Acute attacks: hemin IV (haem arginate/Panhematin) + IV glucose + supportive; "
            "Prevention: givosiran SC monthly; avoid ALL porphyria-inducing drugs; "
            "Skin: sunblock (mineral-based, not chemical), protective clothing; "
            "Beta-carotene: reduces cutaneous photosensitivity modestly in VP (200–300 mg/day); "
            "No afamelanotide approval for VP skin (approved for EPP only); "
            "GENETIC COUNSELLING: 50% risk to offspring; cascade testing in South African Afrikaner families."
        ),
        "locus": "1q23.3",
        "aa": 477,
        "kDa": 51,
        "omim_gene": "600923",
        "omim_disease": "176200",
        "inheritance": "AD; haploinsufficiency; R59W founder South Africa (1:300 Afrikaner); penetrance ~20%",
        "gene_class": "Haem Biosynthesis Step 7 — Protoporphyrinogen IX Oxidase — Dual Acute+Cutaneous",
        "key_alerts": [
            "PPOX-R59W-SOUTH-AFRICA-1-IN-300: R59W is a South African Afrikaner founder mutation with carrier frequency 1 in 300 — VP is the most common porphyria in South Africa; any patient of Afrikaner descent with acute abdominal pain must have VP excluded urgently with plasma porphyrin fluorescence",
            "PPOX-PLASMA-FLUORESCENCE-626nm-INTER-ATTACK: plasma porphyrin fluorescence emission at 626 nm is positive EVEN BETWEEN ATTACKS in VP — this is the most reliable inter-attack diagnostic test; stable marker unlike urine PBG which normalises; request at specialist centre",
            "PPOX-ETOMIDATE-THIOPENTAL-CONTRAINDICATED: etomidate and thiopental are contraindicated in VP anaesthesia; use propofol (safe) + opioid or ketamine; prilocaine local anaesthetic also carries specific risk in VP — use lidocaine instead",
            "PPOX-SKIN-INDEPENDENT-FROM-ACUTE-ATTACKS: cutaneous blistering photosensitivity in VP can occur entirely independently of neurovisceral acute attacks — do NOT reassure a VP patient with 'only skin disease' that they are safe from drug triggers; full drug avoidance applies regardless",
        ],
        "etiologies": {
            "Classic_Dual_VP": {"pct": 55, "phenotype": "acute neurovisceral + cutaneous blistering photosensitivity — most common combined presentation"},
            "Cutaneous_Only_VP": {"pct": 25, "phenotype": "fragile blistering skin without neurovisceral; plasma 626nm positive; mistaken for PCT"},
            "Acute_Only_VP": {"pct": 20, "phenotype": "neurovisceral attacks without significant skin; identical to AIP biochemically except plasma 626nm"},
            "R59W_Founder_VP": {"pct": 62, "phenotype": "South African Afrikaner R59W variant; high carrier frequency in that population"},
        },
        "stats": {
            "mean_onset_age_y": 26.8,
            "mean_dx_delay_months": 25.6,
            "skin_disease_pct": 80,
            "hemin_treated_pct": 65,
        },
        "dx_delay_distribution": {"<6mo": 20, "6-24mo": 35, ">24mo": 45},
        "patients": [],
    },
    # ── UROS — Congenital Erythropoietic Porphyria ───────────────────────────
    {
        "gene": "UROS",
        "protein": (
            "UROS — 10q25.2 AR — Uroporphyrinogen-III-Synthase-265aa — "
            "Congenital-Erythropoietic-Porphyria-Gunther-Disease — "
            "Pink-Red-Urine-AND-Teeth-Erythrodontia-PATHOGNOMONIC — "
            "Severe-Mutilating-Blistering-NO-Neurovisceral — "
            "HSCT-Curative-Only-Definitive-Treatment — "
            "Afamelanotide-Reduces-Photosensitivity"
        ),
        "alias": (
            "UROS (uroporphyrinogen III synthase / cosynthetase); OMIM gene 606938; "
            "Congenital Erythropoietic Porphyria (CEP) / Günther disease OMIM 263700. "
            "10q25.2; 265 aa; ~29 kDa; autosomal recessive; very rare (~300 cases worldwide). "
            "FUNCTION: UROS catalyses step 4 of haem synthesis — "
            "cyclisation of hydroxymethylbilane → uroporphyrinogen III (the correct isomer for haem); "
            "UROS deficiency → spontaneous non-enzymatic cyclisation forms uroporphyrinogen I (wrong isomer) → "
            "cannot be converted to haem → accumulates as uroporphyrin I (highly photosensitising) → "
            "deposits in skin, teeth, bones; "
            "erythrocyte free porphyrins also dramatically elevated (ring sideroblast-like cells in bone marrow). "
            "NEONATAL/INFANTILE PRESENTATION: "
            "Pink/red urine staining nappies at birth or shortly after — FIRST SIGN; "
            "PATHOGNOMONIC: 'pink nappies' in a newborn should trigger immediate porphyrin testing; "
            "Erythrodontia: reddish-brown/orange discolouration of teeth (uroporphyrin I deposits in dentine); "
            "teeth fluoresce bright red/pink under Wood's (UV) lamp — PATHOGNOMONIC; "
            "SEVERE BLISTERING PHOTOSENSITIVITY: "
            "blistering dermatitis begins in early infancy; minor sun exposure → bullae, erosions, scarring; "
            "progressive mutilation of hands, face: loss of nasal cartilage, ears, eyelids ('Werewolf syndrome'); "
            "corneal scarring → visual impairment; "
            "hypertrichosis (excess facial hair); "
            "alopecia; "
            "NO NEUROVISCERAL ATTACKS (unlike AIP/HCP/VP): CEP is purely cutaneous/haematological; "
            "HAEMATOLOGICAL COMPLICATIONS: "
            "Haemolytic anaemia: intra- and extra-vascular haemolysis; splenomegaly; "
            "erythroid hyperplasia → secondary expansion of porphyrin production → worsens clinical disease; "
            "Transfusion-dependent anaemia in severe cases; "
            "Splenectomy: rarely indicated (transfusion-dependent haemolysis refractory to other measures); "
            "DIAGNOSIS: "
            "Urinary porphyrins: ↑↑ uroporphyrin I + coproporphyrin I (type I isomers; contrast with "
            "normal porphyrin excretion which is type III); "
            "Erythrocyte free porphyrins: markedly elevated (uroporphyrin I); "
            "Bone marrow: type I isomer porphyrins; erythroblasts fluoresce red under UV; "
            "UROS enzyme activity: severely reduced (<10% of normal in severe forms); "
            "Molecular UROS testing: definitive. "
            "TREATMENT: "
            "PHOTOPROTECTION: total sunlight avoidance — blackout curtains, UV-blocking film on windows, "
            "full-body UV-protective clothing outdoors; LED/fluorescent bulbs preferred (no UV); "
            "Haematin / blood transfusions: suppress bone marrow erythropoiesis → ↓ porphyrin production; "
            "regular transfusions can reduce porphyrin output but iron overload requires chelation; "
            "Afamelanotide (Scenesse): MC1R agonist; increases melanin → reduces photosensitivity in EPP; "
            "used off-label in CEP — modest benefit; approved for EPP/XLP in some jurisdictions; "
            "Activated charcoal + cholestyramine: bind gut porphyrins → reduce entero-hepatic cycle; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): "
            "ONLY CURATIVE TREATMENT — replaces haematopoietic cells with donor (normal UROS) → "
            "porphyrin production normalises; "
            "best outcomes in early childhood before severe mutilation; "
            "allogeneic HSCT from HLA-matched sibling preferred; "
            "consider gene therapy (investigational UROS-corrected autologous HSC); "
            "SUPPORTIVE: "
            "Vitamin E, beta-carotene (antioxidants); "
            "dental protection (avoid white light exposure during dental procedures — use amber tinted filters)."
        ),
        "locus": "10q25.2",
        "aa": 265,
        "kDa": 29,
        "omim_gene": "606938",
        "omim_disease": "263700",
        "inheritance": "AR; biallelic UROS LOF; very rare (~300 cases worldwide); de novo ~rare",
        "gene_class": "Haem Biosynthesis Step 4 — Uroporphyrinogen III Synthase — Type-I-Isomer Accumulation",
        "key_alerts": [
            "UROS-PINK-NAPPIES-ERYTHRODONTIA-PATHOGNOMONIC: pink/red urine staining in a neonate combined with erythrodontia (red-brown teeth fluorescent under UV) is PATHOGNOMONIC for CEP — do not dismiss as haematuria; urgent urine porphyrin isomer analysis required; refer immediately to metabolic porphyria centre",
            "UROS-HSCT-ONLY-CURATIVE: haematopoietic stem cell transplantation is the ONLY curative treatment for CEP — consider in any child with severe disease before mutilating changes; outcomes best <2 years of age with HLA-matched sibling donor; enrol in research registry for gene therapy trials",
            "UROS-TOTAL-PHOTOPROTECTION-MANDATORY: even minimal UV exposure causes irreversible bullae and mutilation in CEP; patients need blackout curtains, UV-film on windows, full-body UV protective garments outdoors — standard SPF50 sunblock alone is INSUFFICIENT for CEP severity",
            "UROS-NO-NEUROVISCERAL-ATTACKS: CEP is purely cutaneous/haematological — there are NO neurovisceral attacks; drugs that trigger AIP/HCP/VP do NOT trigger CEP attacks; do NOT restrict safe medications unnecessarily in CEP patients",
        ],
        "etiologies": {
            "Classic_Severe_CEP": {"pct": 60, "phenotype": "biallelic null/severe alleles — pink nappies, erythrodontia, severe blistering, haemolytic anaemia, mutilation"},
            "Moderate_CEP": {"pct": 30, "phenotype": "compound het with milder allele — later onset, less mutilation; may respond to haematin therapy"},
            "Late_Onset_CEP": {"pct": 10, "phenotype": "mild alleles — photosensitivity first apparent in adulthood; rare; may be misdiagnosed as PCT"},
            "CEP_with_MDS": {"pct": 8, "phenotype": "somatic UROS mutation in erythroid clone with MDS — acquired CEP in adults; differentiate from congenital"},
        },
        "stats": {
            "mean_onset_age_y": 0.5,
            "mean_dx_delay_months": 3.2,
            "haemolytic_anaemia_pct": 80,
            "hsct_considered_pct": 55,
        },
        "dx_delay_distribution": {"<6mo": 75, "6-24mo": 18, ">24mo": 7},
        "patients": [],
    },
    # ── FECH — Erythropoietic Protoporphyria ─────────────────────────────────
    {
        "gene": "FECH",
        "protein": (
            "FECH — 18q21.31 AR-Quantitative-Trait-Locus/Low-Penetrance-AD — "
            "Ferrochelatase-423aa — "
            "Erythropoietic-Protoporphyria-Most-Common-Erythropoietic-Porphyria — "
            "Burning-Pain-Within-Minutes-Sunlight-PATHOGNOMONIC-NOT-Blistering — "
            "No-Urine-Porphyrins-KEY-DDx — "
            "Afamelanotide-SC-FDA-Approved-2019-USA — "
            "Liver-Failure-2-5pct-ANNUAL-LFTs-Mandatory"
        ),
        "alias": (
            "FECH (ferrochelatase); OMIM gene 612386; "
            "Erythropoietic Protoporphyria (EPP) OMIM 177000. "
            "18q21.31; 423 aa; ~47 kDa; complex inheritance. "
            "FECH GENETICS — UNIQUE QTL MECHANISM: "
            "EPP usually requires biallelic FECH dysfunction: "
            "(1) one pathogenic null/missense allele (inherited) + "
            "(2) the low-expression IVS3-48C hypomorphic allele on the trans chromosome "
            "    (present in ~10% of population — a QTL lowering FECH expression ~50%); "
            "RESULT: affected individuals have ~25% residual ferrochelatase activity; "
            "clinical penetrance ~90% in compound het (pathogenic allele + IVS3-48C). "
            "RARE TRUE DOMINANT: ~5% of EPP families have a dominant-negative FECH mutation — "
            "true heterozygous AD with 50% reduced activity sufficient for disease. "
            "FUNCTION: FECH is the final enzyme of haem synthesis — "
            "inserts Fe2+ into protoporphyrin IX → haem; "
            "FECH deficiency → protoporphyrin IX (PP9) accumulates in erythrocytes, plasma, liver; "
            "PP9 is highly photosensitising (absorbs at Soret band ~408 nm). "
            "CLINICAL FEATURES — CHARACTERISTIC: "
            "Burning pain in sunlight (NOT itching, NOT blistering initially): "
            "IMMEDIATE onset — burning, tingling, stinging within minutes (5–30 min) of sun exposure; "
            "no skin blistering (distinguishes from VP, HCP, CEP, PCT); "
            "skin looks NORMAL or mildly red/swollen after exposure (oedema); "
            "prolonged severe exposure → persistent oedema for 24–48h; "
            "chronic sun avoidance → vitamin D deficiency (supplement); "
            "Onset: typically childhood (3–5 years) — children cannot explain burning pain; "
            "school avoidance, anxiety around outdoor activities; often initially dismissed; "
            "LATE SEVERE ONSET: minimal skin changes → little suspicion → average dx delay >16 years; "
            "LIVER DISEASE: "
            "PP9 accumulates in hepatocytes → biliary excretion → "
            "gallstones (PP9 stones) — 20–30%; cholecystitis; "
            "Progressive liver disease in 2–5%: "
            "PP9 crystallisation in hepatocytes → hepatocyte death → cirrhosis → liver failure; "
            "annual LFTs mandatory; liver transplantation in liver failure "
            "(but transplant does NOT cure EPP — erythrocyte PP9 production continues; re-accumulation inevitable); "
            "combined liver + stem cell transplantation considered in severe EPP liver failure; "
            "DIAGNOSIS: "
            "Erythrocyte protoporphyrin: markedly elevated (primarily PP9 — free, not zinc-chelated; "
            "KEY DDx: zinc protoporphyrin elevated in iron deficiency and lead poisoning, "
            "but FREE protoporphyrin elevated specifically in EPP/XLP); "
            "Urine porphyrins: NORMAL (KEY DDx from acute porphyrias — no urine abnormality); "
            "Faecal porphyrins: elevated (PP9 excreted via bile); "
            "FECH enzyme assay in erythrocytes: reduced; "
            "Molecular FECH testing: definitive; "
            "TREATMENT: "
            "Afamelanotide (Scenesse, CLINUVEL): SC implant every 60 days (16 mg); "
            "alpha-MSH analogue → stimulates melanocortin-1 receptor (MC1R) → ↑ melanin production → "
            "UV protection via pigmentation; "
            "European Medicines Agency approved 2014; FDA approved 2019 (USA); "
            "patients gain 50–70 extra minutes of sun tolerance per day; "
            "does NOT reduce erythrocyte PP9 — only increases sun tolerance; "
            "Beta-carotene (Lumitene): 120–180 mg/day oral; mild benefit; turns skin orange; "
            "Cholestyramine + activated charcoal: interrupts PP9 enterohepatic circulation → "
            "reduces liver PP9 load; used in liver disease; "
            "Vitamin D supplementation: deficiency common due to sun avoidance; "
            "Annual LFTs + liver USS: mandatory; escalate to hepatologist if LFTs rising; "
            "Liver + HSC transplantation: rare; for end-stage EPP liver failure."
        ),
        "locus": "18q21.31",
        "aa": 423,
        "kDa": 47,
        "omim_gene": "612386",
        "omim_disease": "177000",
        "inheritance": "AR compound het (pathogenic allele + IVS3-48C QTL in trans); rare true AD dominant-negative; ~10%",
        "gene_class": "Haem Biosynthesis Final Step — Ferrochelatase — Free Protoporphyrin IX Accumulation",
        "key_alerts": [
            "FECH-BURNING-PAIN-IMMEDIATE-NOT-BLISTERING-PATHOGNOMONIC: EPP photosensitivity is BURNING PAIN within minutes of sunlight with no immediate blistering (distinguishes from VP, HCP, CEP) — the 'burning pain' is the diagnostic clue; children may describe sun as 'hurting their hands'; do not dismiss as psychosomatic",
            "FECH-AFAMELANOTIDE-FDA2019-APPROVED: afamelanotide (Scenesse) SC implant every 60 days is FDA (2019) and EMA (2014) approved for EPP/XLP — it increases melanin pigmentation allowing longer sun tolerance (~50-70 min/day extra); does NOT cure EPP or reduce erythrocyte PP9; refer to specialist porphyria centre for implant",
            "FECH-LIVER-FAILURE-2-5PCT-ANNUAL-SURVEILLANCE: protoporphyrin IX crystallises in hepatocytes in 2-5% of EPP patients causing cirrhosis and liver failure; annual LFTs + liver ultrasound MANDATORY; rising LFTs → immediate referral to hepatologist + gastroenterologist; liver transplant does NOT cure EPP (erythrocyte production continues)",
            "FECH-URINE-NORMAL-KEY-DDX: urine porphyrins are NORMAL in EPP/XLP (unlike all acute porphyrias) — erythrocyte FREE protoporphyrin is the diagnostic test; check free vs zinc-chelated fraction (zinc-PP elevated in iron deficiency — not EPP)",
        ],
        "etiologies": {
            "Classic_EPP_QTL": {"pct": 88, "phenotype": "compound het FECH pathogenic + IVS3-48C — childhood burning pain; no blistering"},
            "EPP_True_AD": {"pct": 5, "phenotype": "dominant-negative FECH — true AD; similar clinical EPP but pedigree shows vertical transmission"},
            "EPP_Liver_Complication": {"pct": 3, "phenotype": "progressive cholestatic liver disease → cirrhosis → liver failure; rising PP9 in hepatocytes"},
            "EPP_Gallstones": {"pct": 25, "phenotype": "PP9 gallstones; cholecystitis; biliary colic"},
        },
        "stats": {
            "mean_onset_age_y": 4.2,
            "mean_dx_delay_months": 196.8,
            "liver_disease_pct": 4,
            "afamelanotide_treated_pct": 42,
        },
        "dx_delay_distribution": {"<6mo": 8, "6-24mo": 12, ">24mo": 80},
        "patients": [],
    },
    # ── ALAS2 — X-linked Protoporphyria / X-linked Sideroblastic Anaemia ─────
    {
        "gene": "ALAS2",
        "protein": (
            "ALAS2 — Xp11.21 XLD-GOF/XLR-LOF — "
            "ALA-Synthase-2-587aa — "
            "GOF-Exon11-C-Terminal-Deletion-X-linked-Protoporphyria-XLP — "
            "Clinically-IDENTICAL-to-EPP-Burning-Pain-No-Blistering — "
            "LOF-X-linked-Sideroblastic-Anaemia-XLSA — "
            "Afamelanotide-Effective-XLP — "
            "Pyridoxine-B6-Treatment-XLSA-Male"
        ),
        "alias": (
            "ALAS2 (5-aminolevulinic acid synthase 2, erythroid-specific); OMIM gene 301300; "
            "X-linked Protoporphyria (XLP) OMIM 300752; "
            "X-linked Sideroblastic Anaemia (XLSA) OMIM 300751. "
            "Xp11.21; 587 aa; ~65 kDa; X-linked. "
            "ALAS2 DUAL PHENOTYPE: "
            "GAIN OF FUNCTION mutations in ALAS2 → X-LINKED PROTOPORPHYRIA (XLP): "
            "C-terminal deletions in exon 11 (most common) prevent autoinhibitory feedback → "
            "constitutively active ALAS2 → ↑↑ ALA production → ↑↑↑ erythrocyte protoporphyrin → "
            "identical photosensitivity to EPP (FECH deficiency). "
            "LOSS OF FUNCTION mutations in ALAS2 → X-LINKED SIDEROBLASTIC ANAEMIA (XLSA): "
            "insufficient ALA for haem synthesis in erythroid progenitors → "
            "iron accumulates in mitochondria of erythroid cells → ring sideroblasts in bone marrow → "
            "microcytic hypochromic anaemia (iron-loaded); elevated serum iron + ferritin; "
            "Serum iron and ferritin HIGH (contrast with iron deficiency where they are low). "
            "XLP CLINICAL FEATURES: "
            "Burning photosensitivity: IDENTICAL to EPP — burning pain within minutes of sun; "
            "NO blistering; skin looks normal or oedematous; "
            "inheritance: X-linked — males fully affected (hemizygous GOF); "
            "females: heterozygous — variable expression (lyonisation); "
            "erythrocyte protoporphyrin: ↑↑↑ (very high — higher than EPP in many cases); "
            "liver complications: same risk as EPP (PP9 hepatopathy); annual LFTs mandatory; "
            "afamelanotide: effective (FDA approved for XLP alongside EPP — see EPNET/FDA labelling). "
            "XLSA CLINICAL FEATURES: "
            "Males: moderate to severe microcytic anaemia from infancy/childhood; "
            "ring sideroblasts in bone marrow (diagnostic); splenomegaly; "
            "progressive iron overload: cardiac, hepatic (transfusion + iron accumulation from gut); "
            "Pyridoxine (vitamin B6) therapy: most XLSA patients respond partially to pyridoxine 50–200 mg/day; "
            "pyridoxal-5-phosphate is a cofactor for ALAS2; some variants are pyridoxine-responsive; "
            "phlebotomy once anaemia stabilised on pyridoxine (to reduce iron overload); "
            "transfusion for severe anaemia; "
            "females: typically mild anaemia (carrier); "
            "XLSA DIAGNOSIS: "
            "CBC: microcytic hypochromic anaemia; iron studies: serum iron ↑, transferrin saturation ↑; "
            "bone marrow: ring sideroblasts (≥15% erythroid precursors); "
            "contrast with iron deficiency (iron LOW), thalassaemia (iron NORMAL), "
            "lead poisoning (basophilic stippling, exposure history); "
            "ALAS2 molecular testing: definitive."
        ),
        "locus": "Xp11.21",
        "aa": 587,
        "kDa": 65,
        "omim_gene": "301300",
        "omim_disease": "300752",
        "inheritance": "X-linked; GOF = XLP (XLD in expression); LOF = XLSA (XLR — males affected, females carrier)",
        "gene_class": "Haem Biosynthesis Rate-Limiting Step 1 — ALAS2 Erythroid-Specific — GOF or LOF Phenotype",
        "key_alerts": [
            "ALAS2-GOF-XLP-CLINICALLY-IDENTICAL-EPP: X-linked protoporphyria (ALAS2 exon-11 GOF) is clinically indistinguishable from FECH-EPP (burning pain in sunlight, no blistering) — differentiate by genetic testing; both are treated with afamelanotide; erythrocyte protoporphyrin can be even higher than in EPP",
            "ALAS2-LOF-XLSA-RING-SIDEROBLASTS-B6: ALAS2 LOF causes X-linked sideroblastic anaemia in males — ring sideroblasts on bone marrow biopsy + elevated serum iron differentiates from iron deficiency anaemia; pyridoxine 50-200 mg/day is first-line treatment (most patients partially pyridoxine-responsive)",
            "ALAS2-SERUM-IRON-HIGH-NOT-LOW-XLSA: XLSA presents with microcytic anaemia but serum iron/ferritin are ELEVATED (not low) — this immediately rules out iron deficiency anaemia; failing to recognise this leads to inappropriate iron supplementation worsening iron overload",
            "ALAS2-LIVER-SURVEILLANCE-XLP-SAME-EPP: XLP patients have the same risk of PP9-induced liver disease as EPP — annual LFTs and liver ultrasound mandatory; refer to hepatologist if LFTs trend upward; combined liver + HSC transplantation occasionally required for end-stage XLP hepatopathy",
        ],
        "etiologies": {
            "XLP_GOF_Males": {"pct": 55, "phenotype": "hemizygous ALAS2 exon-11 deletion GOF — severe EPP-like burning photosensitivity from childhood"},
            "XLP_GOF_Females": {"pct": 45, "phenotype": "heterozygous GOF — variable expression (lyonisation); may have full EPP phenotype or mild"},
            "XLSA_LOF_Males": {"pct": 40, "phenotype": "hemizygous ALAS2 LOF — moderate-severe microcytic anaemia; ring sideroblasts; pyridoxine-responsive"},
            "XLSA_Carrier_Females": {"pct": 60, "phenotype": "heterozygous LOF — mild anaemia; occasional macrocytosis; rarely severe"},
        },
        "stats": {
            "mean_onset_age_y": 5.8,
            "mean_dx_delay_months": 168.0,
            "xlsa_b6_response_pct": 70,
            "afamelanotide_treated_pct": 38,
        },
        "dx_delay_distribution": {"<6mo": 10, "6-24mo": 15, ">24mo": 75},
        "patients": [],
    },
    # ── UROD — Porphyria Cutanea Tarda ────────────────────────────────────────
    {
        "gene": "UROD",
        "protein": (
            "UROD — 1p34.1 AD/AR — Uroporphyrinogen-Decarboxylase-367aa — "
            "Porphyria-Cutanea-Tarda-MOST-COMMON-Porphyria-Worldwide — "
            "Blistering-Photosensitivity-Dorsal-Hands-CLASSIC — "
            "Phlebotomy-Hydroxychloroquine-FIRST-LINE — "
            "Iron-Overload-Alcohol-HCV-HIV-Oestrogen-Key-Triggers — "
            "HFE-C282Y-STRONG-CO-FACTOR — "
            "NO-Neurovisceral-Attacks"
        ),
        "alias": (
            "UROD (uroporphyrinogen decarboxylase); OMIM gene 613521; "
            "Porphyria Cutanea Tarda (PCT) OMIM 176090. "
            "1p34.1; 367 aa; ~41 kDa; autosomal dominant (type II PCT) or acquired (type I PCT, most common). "
            "PCT — TWO TYPES: "
            "TYPE I (SPORADIC, 75–80%): UROD gene normal; hepatic UROD enzyme activity reduced to ~20% "
            "due to ACCUMULATED INHIBITOR (uroporphomethene — generated by iron-dependent oxidation); "
            "TYPE II (FAMILIAL, 20–25%): UROD pathogenic variant (heterozygous → 50% enzyme activity); "
            "second hit (iron, HCV, etc.) further reduces to ~20% → PCT threshold. "
            "PCT is the MOST COMMON PORPHYRIA WORLDWIDE: prevalence ~1 in 10,000. "
            "TRIGGERS THAT PRECIPITATE BOTH TYPE I AND II PCT: "
            "(1) IRON OVERLOAD: the most important trigger; HFE C282Y co-mutation dramatically increases risk "
            "    (homozygous HFE C282Y + heterozygous UROD → very high PCT risk); "
            "    MECHANISM: excess hepatic iron generates reactive oxygen species → "
            "    oxidises UROD active site → inhibited UROD → uroporphyrin I + III accumulation; "
            "(2) ALCOHOL: direct hepatic iron mobilisation + CYP induction; "
            "(3) HCV infection: 10–30% of PCT patients are HCV+; "
            "(4) HIV infection: HCV co-infection common; also direct iron dysregulation; "
            "(5) OESTROGENS (exogenous): OCP, HRT → increase hepatic iron turnover; "
            "    PCT in men on feminising HRT is well-documented; "
            "(6) Polychlorinated biphenyls (PCBs): Turkish epidemic 1956–1961 (hexachlorobenzene). "
            "CLINICAL FEATURES: "
            "Blistering photosensitive dermatitis: fragile bullae on dorsal hands and face — most classic; "
            "subepidermal bullae (NOT epidermal, hence deeper scarring); "
            "Milia (white inclusion cysts in healing areas); "
            "Hypertrichosis (excess facial hair — females pathognomonic); "
            "Hyperpigmentation + hypopigmentation of sun-exposed skin; "
            "Scleroderma-like changes (chronic cases — skin induration); "
            "NO NEUROVISCERAL ATTACKS: PCT is purely a cutaneous porphyria; "
            "differentiates from all acute porphyrias (AIP, HCP, VP). "
            "BIOCHEMISTRY: "
            "Urine porphyrins: ↑↑↑ uroporphyrin (8-COOH) + heptacarboxylporphyrin; "
            "Faecal porphyrins: ↑ isocoproporphyrin (SPECIFIC for PCT — not present in other porphyrias); "
            "Urine ALA + PBG: NORMAL (distinguishes from acute porphyrias); "
            "Serum iron: ↑ ferritin, transferrin saturation; "
            "HFE genotyping: C282Y/H63D; "
            "LFTs: often mildly elevated; check for HCV, HIV, ALD. "
            "TREATMENT: "
            "IDENTIFY AND REMOVE TRIGGERS: "
            "Alcohol cessation (mandatory); "
            "Discontinue oestrogen-containing OCP or HRT; "
            "Treat HCV if present (direct-acting antivirals → remission of PCT after SVR); "
            "FIRST-LINE TREATMENT: "
            "1. Phlebotomy: 450 mL every 2–4 weeks until ferritin 15–30 µg/L and transferrin sat <20%; "
            "    typically 5–10 phlebotomies; clinical remission in >90% (months); "
            "2. Low-dose hydroxychloroquine: 100–200 mg twice weekly (NOT full anti-malarial dose!); "
            "    chelates uroporphyrins in liver → urinary excretion; "
            "    effective as phlebotomy; preferred when phlebotomy poorly tolerated (anaemia); "
            "    FULL DOSE hydroxychloroquine CONTRAINDICATED in PCT (hepatotoxicity with porphyrins); "
            "combined phlebotomy + low-dose HCQ: faster remission in severe PCT; "
            "RELAPSE: treat triggers; repeat phlebotomy; annual skin check; LFTs."
        ),
        "locus": "1p34.1",
        "aa": 367,
        "kDa": 41,
        "omim_gene": "613521",
        "omim_disease": "176090",
        "inheritance": "AD (type II familial ~20%); acquired inhibition (type I ~80%); triggers mandatory for both",
        "gene_class": "Haem Biosynthesis Step 5 — Uroporphyrinogen Decarboxylase — Blistering Skin Only",
        "key_alerts": [
            "UROD-PHLEBOTOMY-LOW-DOSE-HCQ-FIRST-LINE: PCT is treated with phlebotomy (target ferritin 15-30) OR low-dose hydroxychloroquine 100-200 mg twice weekly (NOT standard full dose — hepatotoxic); full-dose HCQ is CONTRAINDICATED in PCT; remove ALL triggers first (alcohol, oestrogens, HCV)",
            "UROD-HCV-HIV-SCREEN-MANDATORY: HCV is found in 10-30% of PCT patients; HIV co-infection also increases risk; all PCT patients must be screened for HCV/HIV; treating HCV with DAAs causes remission of PCT after SVR — both conditions treated simultaneously",
            "UROD-HFE-C282Y-MAJOR-COFACTOR: HFE C282Y co-mutation dramatically amplifies PCT risk (especially homozygous HFE C282Y + UROD variant); screen all PCT patients for HFE mutation; check transferrin saturation and ferritin; phlebotomy targets both PCT and HFE haemochromatosis simultaneously",
            "UROD-NO-NEUROVISCERAL-ATTACKS-DRUG-TRIGGERS-NOT-APPLICABLE: PCT is purely cutaneous — there are NO neurovisceral attacks and the drug trigger rules of AIP/HCP/VP do NOT apply to PCT; do not restrict safe medications from PCT patients based on acute porphyria drug lists",
        ],
        "etiologies": {
            "PCT_Type_I_Sporadic": {"pct": 78, "phenotype": "acquired UROD inhibition by iron — alcohol, HCV, iron overload, oestrogen triggers; UROD gene normal"},
            "PCT_Type_II_Familial": {"pct": 22, "phenotype": "heterozygous UROD pathogenic variant; second hit (iron/HCV/alcohol) precipitates clinical disease"},
            "PCT_with_HFE_C282Y": {"pct": 35, "phenotype": "co-existing HFE C282Y (het or hom) — amplifies iron loading; phlebotomy addresses both PCT and haemochromatosis"},
            "PCT_HCV_Associated": {"pct": 25, "phenotype": "HCV-triggered PCT; treating HCV with DAAs leads to PCT remission after SVR"},
        },
        "stats": {
            "mean_onset_age_y": 45.2,
            "mean_dx_delay_months": 14.8,
            "alcohol_association_pct": 65,
            "phlebotomy_remission_pct": 90,
        },
        "dx_delay_distribution": {"<6mo": 38, "6-24mo": 42, ">24mo": 20},
        "patients": [],
    },
    # ── ALAD — ALA-Dehydratase Deficiency Porphyria ───────────────────────────
    {
        "gene": "ALAD",
        "protein": (
            "ALAD — 9q32 AR — ALA-Dehydratase-330aa — "
            "ALA-Dehydratase-Deficiency-Porphyria-RAREST-Human-Porphyria — "
            "AR-Inheritance-UNIQUE-Among-Acute-Porphyrias — "
            "<10-Cases-Worldwide-Reported — "
            "Lead-Inhibits-ALAD-Enzyme-DDx-Lead-Poisoning — "
            "Hemin-IV-Acute-Attack — "
            "Normal-Urinary-PBG-KEY-DDx-from-HMBS-AIP"
        ),
        "alias": (
            "ALAD (5-aminolevulinic acid dehydratase / porphobilinogen synthase); OMIM gene 125270; "
            "ALA-Dehydratase Deficiency Porphyria (ADP) OMIM 612740. "
            "9q32; 330 aa; ~37 kDa (monomer; functional octamer); autosomal recessive. "
            "RARITY: ADP is the RAREST human porphyria — approximately <10 cases documented worldwide "
            "since 1979 first description (Doss et al., Germany); all cases have been severe. "
            "UNIQUE FEATURES: "
            "(1) AUTOSOMAL RECESSIVE — unlike HMBS (AIP), CPOX (HCP), PPOX (VP) which are all AD; "
            "requires biallelic ALAD LOF for disease; both parents are obligate carriers; "
            "(2) LEAD COMPETITIVE INHIBITION: "
            "lead (Pb2+) is a potent competitive inhibitor of ALAD (displaces Zn2+ from active site); "
            "lead poisoning produces a biochemical picture IDENTICAL to ADP: "
            "↑↑ urinary ALA; ↑↑ erythrocyte zinc protoporphyrin; normal urinary PBG; "
            "KEY DDx: in any patient with apparent ADP-like biochemistry, EXCLUDE LEAD POISONING FIRST; "
            "blood lead level mandatory before diagnosing ADP; "
            "FUNCTION: ALAD catalyses step 2 of haem synthesis — "
            "condensation of 2 ALA → porphobilinogen (PBG) + asymmetric; "
            "ALAD deficiency → ALA accumulates → neurotoxicity (same mechanism as AIP/HCP/VP); "
            "CRUCIALLY: PBG is NOT elevated (ALAD deficiency is upstream of PBG synthesis); "
            "KEY DDx FROM OTHER ACUTE PORPHYRIAS: "
            "AIP/HCP/VP: ↑ urinary ALA + ↑ urinary PBG; "
            "ADP: ↑ urinary ALA ONLY; urinary PBG NORMAL; "
            "lead poisoning: identical to ADP on urine + erythrocyte findings; "
            "check blood lead to distinguish. "
            "CLINICAL FEATURES: "
            "Neurovisceral attacks: identical to AIP — severe abdominal pain, motor neuropathy, SIADH; "
            "onset typically childhood or adolescence (all reported cases presented young); "
            "severe course in most reported patients; "
            "DRUG TRIGGERS: same class as AIP (ALAS1-inducing drugs); "
            "EPNET drug database applies to ADP also. "
            "DIAGNOSIS: "
            "Urinary ALA: markedly elevated; "
            "Urinary PBG: NORMAL (critical DDx — PBG not elevated because ALAD is upstream of PBG step); "
            "Blood lead level: MUST exclude lead poisoning first; "
            "ALAD erythrocyte enzyme activity: severely reduced (often <5% of normal); "
            "Erythrocyte zinc protoporphyrin: elevated (same as lead poisoning); "
            "ALAD molecular testing: biallelic pathogenic variants. "
            "TREATMENT: "
            "Acute attacks: hemin IV (same as AIP — ALAS1 repression reduces substrate flux); "
            "IV glucose loading (weaker effect than hemin); "
            "Supportive care for neuropathy/SIADH; "
            "Liver transplantation: has been performed in severe recurrent ADP "
            "(case reports — partial benefit; liver ALAD replaced but erythrocyte ALAD still deficient); "
            "ERYTHROCYTE TRANSFUSION: reduces erythrocyte ZnPP burden transiently; "
            "Avoid all ALAS1-inducing drugs (EPNET list applies); "
            "PROGNOSIS: all reported cases have had severe progressive course; "
            "management is challenging due to rarity."
        ),
        "locus": "9q32",
        "aa": 330,
        "kDa": 37,
        "omim_gene": "125270",
        "omim_disease": "612740",
        "inheritance": "AR; biallelic ALAD LOF; parents obligate carriers; <10 cases worldwide",
        "gene_class": "Haem Biosynthesis Step 2 — ALA Dehydratase / PBG Synthase — ALA Accumulation Only",
        "key_alerts": [
            "ALAD-EXCLUDE-LEAD-POISONING-FIRST: the biochemical picture of ADP (↑ urine ALA, ↑ erythrocyte ZnPP, NORMAL urine PBG) is IDENTICAL to lead poisoning — blood lead level is MANDATORY before diagnosing ADP; failure to check lead level has led to diagnostic errors",
            "ALAD-PBG-NORMAL-KEY-DDX-FROM-AIP: urinary PBG is NORMAL in ADP (ALA dehydratase is upstream of PBG synthesis) — this is the critical biochemical DDx from AIP/HCP/VP where PBG is markedly elevated; an acute porphyria with elevated ALA but normal PBG should immediately raise ADP vs lead poisoning",
            "ALAD-RAREST-PORPHYRIA-AR-UNIQUE: ADP is the ONLY acute porphyria with autosomal recessive inheritance (unlike AIP, HCP, VP which are all AD); <10 cases ever reported worldwide; biallelic ALAD testing required; refer to international porphyria expert centres",
            "ALAD-SAME-DRUG-TRIGGERS-AIP: despite different enzyme, ADP has the same drug trigger list as AIP/HCP/VP (ALAS1-inducing drugs accelerate flux and worsen ALA accumulation); EPNET drug database applies to ADP; hemin IV is first-line acute treatment",
        ],
        "etiologies": {
            "Classic_Severe_ADP": {"pct": 70, "phenotype": "biallelic null/severe alleles — severe childhood/adolescent neurovisceral attacks; ↑ ALA, normal PBG"},
            "ADP_Moderate": {"pct": 20, "phenotype": "compound het with milder allele — intermittent attacks; responds to haem; residual enzyme 5-15%"},
            "DDx_Lead_Poisoning": {"pct": 0, "phenotype": "NOT ADP — lead inhibits ALAD mimicking ADP biochemistry; always exclude with blood lead FIRST"},
            "ADP_Latent": {"pct": 10, "phenotype": "biallelic ALAD LOF but no clinical attacks in carrier — very rare; monitored"},
        },
        "stats": {
            "mean_onset_age_y": 12.4,
            "mean_dx_delay_months": 28.6,
            "hemin_response_pct": 72,
            "worldwide_cases_approx": 8,
        },
        "dx_delay_distribution": {"<6mo": 22, "6-24mo": 36, ">24mo": 42},
        "patients": [],
    },
]


def _generate_patients():
    for idx, gene_data in enumerate(PORPHYRIA_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        patients = []
        gene = gene_data["gene"]
        for i in range(40):
            if gene == "HMBS":
                trigger = rng.choice(["drug_barbiturate", "drug_other", "hormonal_premenstrual", "fasting", "infection"])
                onset_age = rng.randint(18, 45)
                age_at_dx = onset_age + rng.randint(0, 4)
                dx_delay = max(2, (age_at_dx - onset_age) * 12 + rng.randint(0, 24))
                attacks_per_year = rng.choice([1, 2, 2, 3, 5, 8])
                givosiran = rng.random() < 0.35 and attacks_per_year >= 2
                patients.append({
                    "patient_id": f"HMBS-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": trigger,
                    "attacks_per_year": attacks_per_year,
                    "givosiran_treated": givosiran,
                    "gene": gene, "seed": seed,
                })
            elif gene == "CPOX":
                has_skin = rng.random() < 0.30
                trigger = rng.choice(["drug", "hormonal", "fasting"])
                onset_age = rng.randint(22, 48)
                age_at_dx = onset_age + rng.randint(1, 5)
                dx_delay = max(6, (age_at_dx - onset_age) * 12 + rng.randint(0, 30))
                patients.append({
                    "patient_id": f"CPOX-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": "acute_plus_cutaneous" if has_skin else "acute_only",
                    "skin_blistering": has_skin,
                    "trigger": trigger,
                    "gene": gene, "seed": seed,
                })
            elif gene == "PPOX":
                r59w = rng.random() < 0.62
                subtype = rng.choice(["dual", "dual", "cutaneous_only", "acute_only"])
                onset_age = rng.randint(18, 50)
                age_at_dx = onset_age + rng.randint(1, 6)
                dx_delay = max(6, (age_at_dx - onset_age) * 12 + rng.randint(0, 36))
                patients.append({
                    "patient_id": f"PPOX-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": subtype,
                    "variant_R59W": r59w,
                    "gene": gene, "seed": seed,
                })
            elif gene == "UROS":
                severity = rng.choice(["severe", "severe", "moderate", "late_onset"])
                onset_age = 0.0 if severity == "severe" else (rng.uniform(0.5, 3) if severity == "moderate" else rng.randint(18, 45))
                age_at_dx = onset_age + rng.uniform(0, 0.5)
                dx_delay = max(0, age_at_dx * 12)
                hsct = severity == "severe" and rng.random() < 0.55
                patients.append({
                    "patient_id": f"UROS-{i+1:03d}",
                    "onset_age": round(onset_age, 1),
                    "age_at_dx": round(age_at_dx, 1),
                    "dx_delay_months": round(dx_delay),
                    "phenotype": severity,
                    "hsct_performed": hsct,
                    "erythrodontia": severity in ("severe", "moderate"),
                    "gene": gene, "seed": seed,
                })
            elif gene == "FECH":
                has_liver = rng.random() < 0.04
                gallstones = rng.random() < 0.25
                onset_age = rng.randint(2, 8)
                age_at_dx = onset_age + rng.randint(5, 25)
                dx_delay = (age_at_dx - onset_age) * 12
                afamelanotide = rng.random() < 0.42
                patients.append({
                    "patient_id": f"FECH-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": "EPP_liver_disease" if has_liver else "EPP_classic",
                    "liver_complication": has_liver,
                    "gallstones": gallstones,
                    "afamelanotide_treated": afamelanotide,
                    "gene": gene, "seed": seed,
                })
            elif gene == "ALAS2":
                subtype = rng.choice(["XLP_GOF", "XLP_GOF", "XLSA_LOF"])
                sex = "M" if rng.random() < 0.6 else "F"
                onset_age = rng.randint(2, 10) if subtype == "XLP_GOF" else rng.randint(0, 5)
                age_at_dx = onset_age + rng.randint(8, 20) if subtype == "XLP_GOF" else onset_age + rng.randint(0, 3)
                dx_delay = max(12, (age_at_dx - onset_age) * 12)
                b6_resp = subtype == "XLSA_LOF" and rng.random() < 0.70
                patients.append({
                    "patient_id": f"ALAS2-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": subtype,
                    "sex": sex,
                    "pyridoxine_responsive": b6_resp,
                    "gene": gene, "seed": seed,
                })
            elif gene == "UROD":
                pct_type = rng.choice(["type_I", "type_I", "type_I", "type_II"])
                hfe_c282y = rng.random() < 0.35
                hcv_pos = rng.random() < 0.25
                alcohol = rng.random() < 0.65
                onset_age = rng.randint(30, 62)
                age_at_dx = onset_age + rng.randint(0, 3)
                dx_delay = max(2, (age_at_dx - onset_age) * 12 + rng.randint(0, 18))
                phlebotomy_remission = rng.random() < 0.90
                patients.append({
                    "patient_id": f"UROD-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": pct_type,
                    "hfe_c282y_carrier": hfe_c282y,
                    "hcv_positive": hcv_pos,
                    "alcohol_history": alcohol,
                    "phlebotomy_remission": phlebotomy_remission,
                    "gene": gene, "seed": seed,
                })
            else:  # ALAD
                onset_age = rng.randint(4, 22)
                age_at_dx = onset_age + rng.randint(1, 5)
                dx_delay = max(12, (age_at_dx - onset_age) * 12 + rng.randint(0, 36))
                lead_excluded = True
                hemin_response = rng.random() < 0.72
                patients.append({
                    "patient_id": f"ALAD-{i+1:03d}",
                    "onset_age": onset_age,
                    "age_at_dx": age_at_dx,
                    "dx_delay_months": dx_delay,
                    "phenotype": "ADP_severe",
                    "lead_poisoning_excluded": lead_excluded,
                    "hemin_response": hemin_response,
                    "gene": gene, "seed": seed,
                })
        gene_data["patients"] = patients


_generate_patients()


def overview():
    all_delays = [
        p.get("dx_delay_months", 0)
        for g in PORPHYRIA_GENES for p in g["patients"]
    ]
    all_ages = [
        p.get("onset_age", 0)
        for g in PORPHYRIA_GENES for p in g["patients"]
    ]
    total = sum(len(g["patients"]) for g in PORPHYRIA_GENES)
    return {
        "atlas": "Hereditary Porphyria Atlas — Complete 8-Gene Reference",
        "subtitle": (
            "HMBS (AIP) · CPOX (HCP) · PPOX (VP) · UROS (CEP) · "
            "FECH (EPP) · ALAS2 (XLP/XLSA) · UROD (PCT) · ALAD (ADP) — "
            "320 Patients (8×40, Seeds 1758–1765)"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "aggregate_stats": {
            "genes_covered": 8,
            "patients_per_gene": 40,
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
        },
        "genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "aa": g["aa"],
                "kDa": g["kDa"],
                "n_patients": len(g["patients"]),
                "mean_dx_age": round(
                    sum(p.get("onset_age", 0) for p in g["patients"]) / len(g["patients"]), 1
                ),
                "mean_dx_delay_months": round(
                    sum(p.get("dx_delay_months", 0) for p in g["patients"]) / len(g["patients"]), 1
                ),
            }
            for g in PORPHYRIA_GENES
        ],
        "top_alerts": [
            "HMBS-BARBITURATES-ABSOLUTE-PROHIBITION: barbiturates are HIGH RISK triggers for acute AIP attacks — any anaesthetist must check AIP status; use propofol instead; check EPNET drug database for >300 triggers; givosiran FDA 2019 for prevention",
            "CPOX-FAECAL-COPROPORPHYRIN-KEY-DDX: faecal coproporphyrin III is elevated EVEN BETWEEN ATTACKS in HCP (distinguishes from AIP where faecal porphyrins are normal); plasma fluorescence 619 nm (HCP) vs 626 nm (VP)",
            "PPOX-R59W-SOUTH-AFRICA-1IN300: R59W PPOX is a South African Afrikaner founder (1 in 300) — VP is the most common porphyria in South Africa; plasma porphyrin 626 nm is stable inter-attack diagnostic test",
            "UROS-PINK-NAPPIES-ERYTHRODONTIA-PATHOGNOMONIC: red urine staining nappies + erythrodontia = CEP (Günther disease) until proven otherwise; HSCT is the ONLY curative treatment",
            "FECH-BURNING-PAIN-IMMEDIATE-NOT-BLISTERING: EPP burning pain within minutes of sunlight with NO blistering is PATHOGNOMONIC; afamelanotide SC FDA 2019 approved; annual LFTs mandatory (liver failure 2-5%)",
            "ALAS2-GOF-XLP-IDENTICAL-EPP: ALAS2 exon-11 GOF = X-linked Protoporphyria — clinically identical to EPP; LOF = X-linked sideroblastic anaemia (ring sideroblasts + elevated serum iron — NOT iron deficiency)",
            "UROD-PHLEBOTOMY-LOW-HCQ-FIRST-LINE: PCT treatment — phlebotomy (target ferritin 15-30) OR low-dose HCQ 100-200 mg twice weekly; full-dose HCQ CONTRAINDICATED; screen for HCV/HIV/HFE; remove alcohol and oestrogens",
            "ALAD-EXCLUDE-LEAD-FIRST-NORMAL-PBG: ADP is the rarest porphyria (<10 cases worldwide); elevated ALA with NORMAL PBG (DDx from AIP/HCP/VP); blood lead level mandatory to exclude lead poisoning FIRST",
        ],
    }


def breakdown():
    result = []
    for idx, g in enumerate(PORPHYRIA_GENES):
        delays = [p.get("dx_delay_months", 0) for p in g["patients"]]
        ages = [p.get("onset_age", 0) for p in g["patients"]]
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "computed": {
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def definitions():
    return {
        "concepts": {
            "The Haem Biosynthesis Pathway — 8 Enzymes, 8 Porphyrias": (
                "Haem synthesis is an 8-step pathway beginning in mitochondria (steps 1, 7, 8), "
                "continuing in the cytosol (steps 2–5), and finishing in mitochondria. "
                "PATHWAY OVERVIEW: "
                "Step 1: ALAS2 (erythroid) / ALAS1 (ubiquitous) — "
                "succinyl-CoA + glycine → ALA (rate-limiting; ALAS1 is the porphyria regulation target); "
                "Step 2: ALAD — 2× ALA → porphobilinogen (PBG); inhibited by lead; "
                "Step 3: HMBS — 4× PBG → hydroxymethylbilane (HMB); "
                "Step 4: UROS — HMB → uroporphyrinogen III (correct isomer); deficiency → type I isomers; "
                "Step 5: UROD — uroporphyrinogen III → coproporphyrinogen III (4 decarboxylations); "
                "Step 6: CPOX — coproporphyrinogen III → protoporphyrinogen IX; "
                "Step 7: PPOX — protoporphyrinogen IX → protoporphyrin IX; "
                "Step 8: FECH — protoporphyrin IX + Fe2+ → haem. "
                "PORPHYRIA CLASSIFICATION: "
                "ACUTE (neurovisceral) = steps 1–4 (ALAD, HMBS, CPOX, PPOX) — ALA/PBG accumulation; "
                "CUTANEOUS (skin) = steps 5–8 (UROS, FECH, ALAS2-GOF, UROD) — photosensitising porphyrins; "
                "MIXED = CPOX, PPOX (both acute and cutaneous). "
                "REGULATION: ALAS1 is the rate-limiting enzyme; repressed by haem (feedback); "
                "induced by >300 drugs (CYP inducers, glucose deprivation, progesterone); "
                "hemin (IV haem) acutely represses ALAS1 → ↓ ALA production → ↓ acute attack."
            ),
            "The Acute Porphyria Attack — ALA Neurotoxicity Mechanism": (
                "THE ALA HYPOTHESIS: "
                "ALA (5-aminolevulinic acid) is a structural analogue of GABA; "
                "ALA inhibits GABA-A receptors → reduces inhibitory neurotransmission → "
                "excitatory neuropathy, autonomic dysfunction, and visceral pain; "
                "ALA also generates reactive oxygen species → oxidative neuronal injury; "
                "Both acute porphyrias that primarily accumulate ALA (ALAD/ADP) AND porphyrias that "
                "accumulate ALA + PBG (HMBS/AIP, CPOX/HCP, PPOX/VP) all share this neurovisceral phenotype. "
                "CLINICAL ATTACK SPECTRUM: "
                "Visceral: severe abdominal pain (colicky, poorly localised); nausea/vomiting; constipation; "
                "Autonomic: tachycardia, hypertension (sympathetic overactivation); "
                "SIADH: ↑ ADH → hyponatraemia → if uncorrected → seizures; "
                "Motor neuropathy: acute motor > sensory; proximal weakness → respiratory failure; "
                "Psychiatric: agitation, confusion, psychosis — may precede somatic symptoms; "
                "PORT-WINE URINE: ALA + PBG oxidise on standing → dark red/brown porphyrins — diagnostic clue; "
                "MANAGEMENT PRIORITIES: "
                "1. Identify and STOP triggering drug; "
                "2. Haem (hemin/haem arginate) IV: represses ALAS1 → ↓ ALA/PBG within 24–48h; "
                "3. IV glucose/dextrose: reduces PGC-1α → mild ALAS1 reduction (weaker than hemin); "
                "4. Opioid analgesia (safe); beta-blockers for autonomic; "
                "5. IV levetiracetam for seizures (avoid ALMOST ALL other AEDs in acute porphyria); "
                "6. ICU if motor neuropathy progresses to respiratory compromise."
            ),
            "Distinguishing the Four Acute Porphyrias Biochemically": (
                "All four acute porphyrias (ADP, AIP, HCP, VP) cause neurovisceral attacks with similar "
                "symptoms. Biochemical profile distinguishes them: "
                "ADP (ALAD): "
                "↑↑ urinary ALA; NORMAL urinary PBG; ↑ erythrocyte ZnPP; faecal porphyrins normal; "
                "NOTE: lead poisoning produces IDENTICAL profile — blood lead mandatory first. "
                "AIP (HMBS): "
                "↑↑↑ urinary ALA + PBG (both markedly elevated during attack); "
                "faecal porphyrins NORMAL; plasma porphyrins non-specific. "
                "HCP (CPOX): "
                "↑↑ urinary ALA + PBG (during attack, may normalise inter-attack); "
                "↑↑ faecal coproporphyrin III (ALWAYS elevated, stable inter-attack — KEY MARKER); "
                "urinary coproporphyrin III elevated; plasma fluorescence 619 nm. "
                "VP (PPOX): "
                "↑↑ urinary ALA + PBG (during attack, may normalise inter-attack); "
                "↑↑ faecal protoporphyrin + coproporphyrin (protoporphyrin dominant in VP, contrast HCP); "
                "plasma porphyrin fluorescence 626 nm (stable inter-attack — KEY MARKER FOR VP); "
                "PRACTICAL ALGORITHM: "
                "1. Spot urine PBG during attack (if positive → acute porphyria; if negative → ADP, lead, other); "
                "2. Faecal porphyrins: normal (AIP) vs elevated (HCP, VP); "
                "3. Plasma fluorescence: 619 nm (HCP) vs 626 nm (VP). "
                "Most clinical presentations need molecular testing for definitive diagnosis."
            ),
            "Givosiran (GIVLAARI) — siRNA Targeting ALAS1 for Recurrent Acute Porphyria": (
                "Givosiran (GIVLAARI, Alnylam Pharmaceuticals) is a subcutaneous siRNA "
                "targeting ALAS1 mRNA in hepatocytes, approved by FDA (2019) and EMA (2020) "
                "for prevention of recurrent acute attacks across the four acute porphyrias "
                "(AIP, HCP, VP, ADP). "
                "MECHANISM: "
                "GalNAc-siRNA conjugate → selectively taken up by hepatocytes (asialoglycoprotein receptor) → "
                "RISC complex formed → ALAS1 mRNA cleaved → ↓ ALAS1 protein → "
                "↓ haem synthesis flux → ↓ ALA and PBG production → ↓ attack frequency; "
                "effect persists between monthly doses (RISC is persistent). "
                "PIVOTAL TRIAL — ENVISION: "
                "Phase 3, RCT; givosiran 2.5 mg/kg SC monthly vs placebo; "
                "AHP-SSC composite: 74% reduction in composite attack rate vs placebo; "
                "urinary ALA ↓ 57%, PBG ↓ 60% from baseline; "
                "hemin use ↓ 70%; hospitalisation rate ↓; "
                "quality of life measures improved. "
                "ADVERSE EFFECTS: "
                "Elevated transaminases (ALT >3× ULN in ~30%): monitor LFTs monthly in year 1, "
                "then quarterly; dose reduction or interruption if significant elevation; "
                "Homocysteine elevation (↓ ALAS2 → ↓ PLP availability → ↓ CBS activity): "
                "supplement with pyridoxine 50 mg/day and folate + B12 prophylactically; "
                "Injection site reactions: local erythema; rotate sites; "
                "Renal function: monitor; glomerular involvement in some patients. "
                "ELIGIBILITY: ≥2 porphyria attacks/year requiring hospitalisation or IV hemin; "
                "not for acute treatment; continue hemin for breakthrough attacks; "
                "available via specialist porphyria centres."
            ),
            "Afamelanotide (Scenesse) — MC1R Agonist for Erythropoietic Porphyrias": (
                "Afamelanotide (Scenesse, CLINUVEL Pharmaceuticals) is an SC biodegradable implant "
                "releasing an MC1R agonist analogue over ~60 days. "
                "MECHANISM: "
                "Afamelanotide (NDP-α-MSH analogue) binds melanocortin-1 receptor (MC1R) on melanocytes → "
                "↑ eumelanin (dark pigment) production via MITF transcription factor → "
                "↑ skin photoprotection by melanin → attenuates Soret-band (408 nm) light absorption "
                "by erythrocyte protoporphyrins in dermal capillaries → ↓ porphyrin photoexcitation → "
                "↓ reactive oxygen species generation → ↓ burning pain in EPP/XLP. "
                "APPROVALS: "
                "EMA 2014 (Europe): EPP; "
                "FDA 2019 (USA): EPP and XLP; "
                "16 mg SC implant every 60 days (spring/summer scheduling for seasonal use). "
                "CLINICAL EVIDENCE: "
                "Phase 3 trial (Langendonk et al., NEJM 2015): "
                "significant increase in time patients could spend in direct sunlight (+69 min/day); "
                "quality of life measures improved; pain and erythema scores reduced; "
                "afamelanotide does NOT reduce erythrocyte protoporphyrin IX levels — "
                "it only increases melanin protection; "
                "patients must continue sun avoidance and annual liver monitoring. "
                "ADVERSE EFFECTS: "
                "Implant site reaction; transient nausea; hyperpigmentation of skin; "
                "temporary darkening of existing naevi (benign — monitor for melanoma change); "
                "does not appear to promote melanoma (reassuring data). "
                "USE IN CEP: off-label modest benefit; "
                "PHOTOPROTECTION IS STILL REQUIRED even with afamelanotide — it does not confer full protection."
            ),
        },
        "pharmacological_distinctions": [
            "Hemin / Haem arginate (Normosang/Panhematin) — all acute porphyria attacks: IV haem represses ALAS1 within 24-48h; preferred haem arginate (Normosang, EMA) vs hemin (Panhematin, FDA); administer via central line to avoid phlebitis; give within 24h of severe attack; 3-4 mg/kg/day × 4 days; thrombophlebitis risk; haem arginate more stable formulation",
            "Givosiran (GIVLAARI, Alnylam) SC monthly — prevention of recurrent acute porphyria (AIP/HCP/VP/ADP): ALAS1-targeting siRNA; FDA 2019; ≥2 attacks/year; monitor LFTs monthly (year 1); supplement B vitamins (homocysteine elevation); NOT for acute treatment",
            "Afamelanotide (Scenesse, CLINUVEL) SC implant q60d — EPP (FECH) and XLP (ALAS2 GOF): MC1R agonist increases melanin; EMA 2014; FDA 2019; 16 mg implant; does not reduce erythrocyte PP9; supplements sun tolerance (~50-70 min/day); annual liver surveillance continues",
            "Hydroxychloroquine (low-dose) — PCT (UROD): 100-200 mg twice weekly (NOT full 200-400 mg/day dose — hepatotoxic with porphyrins); chelates uroporphyrins; as effective as phlebotomy; preferred when phlebotomy not tolerated; check LFTs and G6PD before starting",
            "Phlebotomy — PCT (UROD): 450 mL every 2-4 weeks; target ferritin 15-30 µg/L + transferrin saturation <20%; ~5-10 phlebotomies for remission; remission in >90%; also treats HFE haemochromatosis if co-present",
            "GnRH analogues (leuprolide/buserelin) — cyclical premenstrual AIP (HMBS): inhibits endogenous progesterone cycling → prevents luteal-phase trigger; use add-back oestrogen (NOT progestogen) for bone protection; long-term use requires bone density monitoring (DEXA)",
            "Pyridoxine (B6) 50-200 mg/day — X-linked sideroblastic anaemia (ALAS2 LOF/XLSA): ALAS2 requires PLP (B6) cofactor; many XLSA patients are pyridoxine-responsive (partial); response determines whether phlebotomy is needed for iron overload; B6 supplementation also required with givosiran (homocysteine rise)",
            "Cholestyramine + activated charcoal — EPP/CEP/PCT: interrupts enterohepatic recirculation of porphyrins via bile; reduces liver porphyrin accumulation; used in EPP liver disease prevention and PCT skin management; 4 g cholestyramine before meals; 1 g charcoal between meals",
            "HSCT (haematopoietic stem cell transplantation) — CEP (UROS): ONLY curative treatment for congenital erythropoietic porphyria; replaces bone marrow with normal UROS-expressing cells; best outcomes in early childhood; HLA-matched sibling preferred; long-term iron overload monitoring post-HSCT",
        ],
        "key_standards": [
            "European Porphyria Network (EPNET) Drug Database — all acute porphyrias: >300 drugs classified as HIGH, POSSIBLE, NOT KNOWN, PROBABLY SAFE risk for triggering acute attacks; accessible at porphyria.eu/drug-database; ALL prescribers for AIP/HCP/VP/ADP patients MUST check before prescribing; printable patient card with known safe/unsafe drugs",
            "American Porphyria Foundation (APF) Drug Database: USA equivalent to EPNET; aporphyria.org; similar classification system; used by US pharmacists and prescribers",
            "Givosiran ENVISION Trial Eligibility: ≥2 documented acute attacks/year requiring hospitalisation or IV hemin; enrol in EXPLORE registry for long-term safety data; monitor LFTs, homocysteine, renal function; available through specialist porphyria centres and hospital pharmacy",
            "Afamelanotide Clinuvel Patient Access: EPP/XLP patients → refer to specialist erythropoietic porphyria centre; Scenesse implant administered by trained physician (SC implant in upper buttock); 60-day intervals; patient to maintain sun avoidance diary; annual erythrocyte PP9 and LFTs",
            "PCT Minimum Workup: spot urine porphyrin isomers (uroporphyrin I+III, heptacarboxylporphyrin); 24h urine uroporphyrin; faecal isocoproporphyrin; serum ferritin + transferrin saturation; LFTs; HCV serology; HIV serology; HFE genotyping (C282Y/H63D); alcohol use history; oestrogen medications review",
            "CEP (UROS) Neonatal Emergency Protocol: pink/red nappy staining in neonate → immediate urine porphyrin isomer analysis; blood porphyrins; skin biopsy not needed for diagnosis; start total photoprotection (blackout environment) before results; refer immediately to metabolic porphyria centre; HSCT evaluation within first months of life",
            "EPP Liver Surveillance Standard: annual LFTs (ALT, AST, GGT, bilirubin, ALP); annual liver ultrasound; if LFTs consistently elevated → fibroscan or liver biopsy; if cirrhosis/liver failure → hepatology MDT + combined liver + HSCT consideration; annual erythrocyte free protoporphyrin (monitor porphyrin burden)",
            "AIP/VP/HCP Anaesthesia Protocol: inform anaesthetic team; avoid thiopental (ABSOLUTE CI), etomidate (CI in VP/HCP), nitrous oxide uncertain; propofol is SAFE; morphine/fentanyl SAFE; use regional anaesthesia where possible; IV glucose perioperatively (minimum 300 g/day); porphyria emergency card to be with patient; emergency contact: national porphyria centre",
        ],
    }
