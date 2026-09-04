#!/usr/bin/env python3
"""Porphyria-Atlas — Complete 8-Gene Porphyria Disorders Atlas
HMBS   (AIP — Acute Intermittent Porphyria; AR dom; most common acute; hydroxymethylbilane synthase) ·
ALAD   (ADP — ALAD Porphyria; AR recessive; rarest; aminolevulinic acid dehydratase) ·
CPOX   (HCP — Hereditary Coproporphyria; AR dom; coproporphyrinogen oxidase) ·
PPOX   (VP — Variegate Porphyria; AR dom; protoporphyrinogen oxidase; South Africa founder) ·
FECH   (EPP — Erythropoietic Protoporphyria; AD/AR; most common erythropoietic; ferrochelatase) ·
UROD   (PCT — Porphyria Cutanea Tarda; AD/acquired; most common porphyria overall; uroporphyrinogen decarboxylase) ·
UROS   (CEP — Congenital Erythropoietic Porphyria / Günther disease; AR; rarest/most severe; uroporphyrinogen III synthase) ·
ALAS2  (XLP — X-linked Protoporphyria; X-linked GOF; exon 10-11 deletion; aminolevulinate synthase 2)
320-patient aggregate cohort (8 × 40, seeds 982–989)

Porphyria — Key Neurological Principles:
  - ACUTE HEPATIC PORPHYRIAS (AHPs): AIP (HMBS), HCP (CPOX), VP (PPOX), ADP (ALAD).
    All four cause neurovisceral crises: severe abdominal pain + autonomic dysfunction +
    peripheral motor neuropathy ± psychiatric/encephalopathic features.
    SEIZURES occur in ~20-30% of acute attacks (AIP most common cause of porphyric seizures).
    Mechanism: delta-aminolevulinic acid (ALA) is directly neurotoxic — inhibits GABA-R,
    damages mitochondria, and induces oxidative stress in peripheral and central neurons.
  - ABSOLUTE-CI AEDs IN ACUTE PORPHYRIA:
    Phenytoin (PHT), Phenobarbital (PB), Carbamazepine (CBZ), Primidone, Oxcarbazepine (OXC),
    Lamotrigine (LTG, contested but usually avoided). Mechanism: all are CYP450 inducers that
    upregulate hepatic ALAS1 → excess ALA/PBG production → worsens neurological crisis.
    These can be FATAL — prescribing them during an acute attack is a critical medical error.
  - SAFE AEDs (acute porphyria): Levetiracetam (LEV) — first-line.
    Gabapentin (GBP), Clonazepam (CZP), Magnesium, Propofol (acute status).
    Vigabatrin: theoretically safe (not a CYP inducer) but very limited data.
  - HEMATIN (Hemin / Panhematin, Normosang): IV hematin provides negative feedback
    on hepatic ALAS1 → reduces ALA/PBG production. Specific treatment for acute attack.
    Must be given within 24-48h of severe attack; delayed administration may not prevent
    neurological damage. Administered as 3-4 mg/kg/day IV for 4 days.
  - GIVOSIRAN (Alnylam Pharmaceuticals, FDA Nov 2019): siRNA targeting ALAS1 mRNA.
    First-ever RNA therapeutic for porphyria. Monthly SC injection reduces urinary ALA+PBG
    by ~90%. Approved for recurrent acute hepatic porphyria attacks (>2 attacks/year).
    Side effects: hepatotoxicity (AST/ALT), homocysteine elevation, nephrotoxicity.
  - GLUCOSE LOADING: High-carbohydrate IV therapy (300-400g glucose/day) suppresses
    ALAS1 via insulin signalling. First-line for mild attacks; hematin for moderate-severe.
  - GIVLAARI (givosiran) vs PANHEMATIN (hemin):
    Givosiran = prophylactic siRNA (monthly SC); Panhematin = acute attack IV rescue.
    Both work via ALAS1 suppression but at different molecular levels.
  - DIAGNOSTIC URINE TEST (acute attack):
    Porphobilinogen (PBG) + delta-aminolevulinic acid (ALA) in urine — markedly elevated
    during attack. Urine turns dark red/port-wine on standing (photooxidation of PBG).
    Watson-Schwartz test: classic but non-specific (also positive with urobilinogen, PBG).
    Quantitative urine PBG (spot or 24h) is the gold standard test during attack.
    Between attacks: urine PBG may normalise (especially in HMBS low-penetrance carriers).
  - SKIN PHOTOSENSITIVITY SPECTRUM:
    BLISTERING: PCT (UROD), CEP (UROS), HCP (CPOX, cutaneous attacks), VP (PPOX, dual).
    NON-BLISTERING (acute burning pain): EPP (FECH), XLP (ALAS2) — most severe acute pain.
    EPP/XLP: UV-visible light causes immediate burning, erythema, oedema within minutes;
    NO bullae; hepatic disease from protoporphyrin deposits (10% fatal EPP liver disease).

COHORT: 8 × 40 = 320 patient slots (seeds 982–989; gene-specific seeds)
"""

import random

SEED_BASE = 982

PORPHYRIA_GENES = [
    # ── HMBS — Acute Intermittent Porphyria (AIP) ─────────────────────────
    {
        "gene": "HMBS", "protein": "Hydroxymethylbilane Synthase (Porphobilinogen Deaminase)",
        "alias": "HMBS — AIP (Acute Intermittent Porphyria; OMIM #176000); most common acute porphyria; autosomal dominant low penetrance (~10%)",
        "aa": "361 aa", "kDa": "44 kDa",
        "gene_class": (
            "Hydroxymethylbilane synthase (HMBS; also called porphobilinogen deaminase, PBGD); "
            "catalyses the condensation of four porphobilinogen (PBG) molecules into "
            "hydroxymethylbilane (HMB) — step 3 of the heme biosynthesis pathway; located in "
            "cytoplasm; HMBS is haploinsufficient — one normal allele produces ~50% enzyme "
            "activity; under normal conditions this is sufficient, but precipitating factors "
            "(drugs, fasting, hormones) upregulate ALAS1, overwhelming the reduced HMBS capacity "
            "→ ALA + PBG accumulate → neurovisceral crisis; 11q23.3; OMIM gene 609806"
        ),
        "porphyria_group": "Acute Hepatic Porphyria (AHP)",
        "subtype": "AIP — Acute Intermittent Porphyria",
        "locus": "11q23.3", "omim_gene": 609806, "omim_disease": 176000,
        "inheritance": "Autosomal Dominant (AD). Low penetrance: only ~10-20% of carriers ever have an attack. Females > Males (3:1 due to hormonal triggers). De novo mutations rare.",
        "seed_offset": 0,
        "onset_range_y": (15.0, 45.0),
        "phenotype": (
            "Neurovisceral crisis: (1) SEVERE ABDOMINAL PAIN (colicky, generalised, most common "
            "presenting feature, >90%); (2) Peripheral motor neuropathy (ascending, axonal, "
            "can cause respiratory failure — ICU admission in severe attacks); "
            "(3) Autonomic dysfunction: tachycardia, hypertension, sweating, constipation/ileus; "
            "(4) SIADH: hyponatraemia (<130 mmol/L in severe attacks — major cause of "
            "SEIZURES; correct slowly to avoid osmotic demyelination); "
            "(5) Psychiatric: anxiety, confusion, hallucinations, paranoia; "
            "(6) Seizures: 20-30% of severe attacks (management hazard — porphyrinogenic AEDs "
            "absolutely contraindicated). Between attacks: many patients are entirely asymptomatic. "
            "Triggers: drugs (CYP inducers), fasting, alcohol, hormonal changes "
            "(luteal phase — cyclical attacks), infections, surgery, stress."
        ),
        "disease": (
            "AIP is the most common of the four acute hepatic porphyrias. Prevalence: "
            "~1:10,000-50,000 in Europe; higher in Finland, Sweden, UK (Scottish highlands). "
            "HMBS haploinsufficiency is required but not sufficient — a precipitating factor "
            "is always present during attacks. The hallmark biochemical finding: dramatically "
            "elevated urine PBG (often >100-fold normal) + elevated ALA during attack. "
            "Urine turns dark red/port-wine on standing (photodecomposition of PBG → porphyrins). "
            "Diagnosis: quantitative spot urine PBG (normalised to creatinine) — highly sensitive "
            "during attack. Between attacks, urine PBG may be normal; erythrocyte HMBS activity "
            "is ~50% of normal (not affected by attack activity). "
            "Treatment acute: IV glucose (300g/day) for mild attacks; IV hematin (3-4 mg/kg/day × 4 days) "
            "for moderate-severe. Prophylactic givosiran (monthly SC) for recurrent attacks (>2/year). "
            "Liver transplant: curative for severe recurrent AIP unresponsive to givosiran. "
            "Complication: chronic kidney disease (tubular damage from chronic ALA exposure — "
            "monitor eGFR; 40-fold increased risk of hepatocellular carcinoma — annual liver US from age 50)."
        ),
        "hallmark": "Urine PBG markedly elevated during attack (Watson-Schwartz + dark red urine). NO cutaneous photosensitivity. Abdominal pain + peripheral motor neuropathy + SIADH seizures.",
        "nbs_marker": "No NBS. Erythrocyte HMBS activity 50% (confirms carrier; not attack-sensitive).",
        "key_biomarker": "Urine PBG (>10× ULN during attack). Urine ALA elevated. Plasma PBG. Fecal porphyrins NORMAL (distinguishes from VP/HCP).",
        "severity_spectrum": "Latent (most carriers never attack) → Mild episodic (PBG urine mildly elevated, manageable at home) → Moderate (hospitalisation, IV hematin) → Severe (respiratory failure, ICU, quadriplegia from motor neuropathy)",
        "emergency": "Acute attack: STOP triggering drug IMMEDIATELY. IV high-glucose saline. IV hematin if moderate-severe. Monitor Na (SIADH). Seizure: LEV IV (NOT PHT, NOT PB, NOT CBZ — fatal). Monitor respiratory function (ascending neuropathy).",
        "ci_drugs": "PHT (ABSOLUTE), PB (ABSOLUTE), CBZ (ABSOLUTE), OXC (ABSOLUTE), Primidone (ABSOLUTE), LTG (RELATIVE, mostly avoided), Valproate (RELATIVE — some centres use cautiously), Rifampicin, Griseofulvin, Sulphonamides, Progesterone-only contraceptives, alcohol, many others — always check porphyria drug database (www.drugs-porphyria.org) before prescribing.",
        "gender": "female",
        "severity_weights": [0.50, 0.35, 0.15],  # Latent, Moderate, Severe
        "treatment_options": [
            "IV hematin + IV glucose (acute moderate-severe)",
            "IV glucose + IV saline + LEV (mild attack + seizure)",
            "Givosiran SC monthly (prophylaxis recurrent attacks)",
            "Liver transplantation (severe refractory)",
            "IV hematin (acute rescue)",
            "LEV + givosiran (seizure prevention + prophylaxis)",
        ],
        "outcome_options": [
            "Full resolution between attacks (latent phase)",
            "Chronic neuropathy (severe prior attack)",
            "Recurrent attacks — givosiran prophylaxis",
            "Dialysis-dependent CKD (chronic ALA nephrotoxicity)",
        ],
    },
    # ── ALAD — ALAD Porphyria (ADP) ────────────────────────────────────────
    {
        "gene": "ALAD", "protein": "Aminolevulinic Acid Dehydratase (Porphobilinogen Synthase)",
        "alias": "ALAD — ADP (ALAD Porphyria; OMIM #125270); rarest porphyria; autosomal recessive; delta-aminolevulinic acid dehydratase",
        "aa": "330 aa", "kDa": "36 kDa (monomer; active as octamer)",
        "gene_class": (
            "Aminolevulinic acid dehydratase (ALAD; also porphobilinogen synthase); "
            "catalyses step 2 of heme biosynthesis: condensation of two ALA molecules → "
            "porphobilinogen (PBG); zinc-dependent; active as a homo-octamer; "
            "located in cytoplasm; 9q32; OMIM gene 125270; "
            "autosomal RECESSIVE — both alleles must be non-functional; "
            "<10 documented cases worldwide as of 2026; "
            "ALAD is also the target of lead — lead poisoning mimics ADP biochemically "
            "(ALA accumulates, PBG normal in lead poisoning — distinguishes from AIP). "
            "Compound heterozygous or homozygous LOF → markedly elevated ALA (AIP-like crisis) "
            "but PBG NORMAL (ALAD block is upstream of PBG formation)"
        ),
        "porphyria_group": "Acute Hepatic Porphyria (AHP) — Autosomal Recessive",
        "subtype": "ADP — ALAD Porphyria (Doss Porphyria)",
        "locus": "9q32", "omim_gene": 125270, "omim_disease": 125270,
        "inheritance": "Autosomal Recessive (AR). Both alleles non-functional required. Heterozygous carriers: ~50% ALAD activity — clinically normal but susceptible to lead toxicity. <10 confirmed cases worldwide (2026).",
        "seed_offset": 1,
        "onset_range_y": (0.5, 20.0),
        "phenotype": (
            "AIP-like neurovisceral crisis: abdominal pain, peripheral neuropathy, autonomic "
            "dysfunction, psychiatric symptoms. BIOCHEMICAL DISTINCTION: urine ALA markedly "
            "elevated BUT urine PBG NORMAL (contrast AIP where BOTH ALA + PBG elevated — "
            "this is the key diagnostic DDx). Onset typically earlier than AIP (childhood or "
            "early adulthood). Erythrocyte ALA markedly elevated (accumulates because PBG "
            "cannot be formed). Erythrocyte protoporphyrin IX elevated (ZnPP, FePP — "
            "secondary to ALA accumulation driving the alternative pathway). "
            "Lead poisoning biochemical DDx: lead → ALAD inhibition → same urine ALA elevation "
            "but PBG normal; blood lead level diagnostic. "
            "Myeloproliferative overlap: some ADP cases occur in context of ALAD frameshift + "
            "polycythaemia/hepatocellular disease (somatic ALAD mutations in myeloid cells)."
        ),
        "disease": (
            "ADP is the rarest inherited porphyria — fewer than 10 molecularly confirmed cases. "
            "First described by Doss et al. (1979) in German patients — also called Doss porphyria. "
            "Biochemical fingerprint: urine ALA >>normal, PBG NORMAL (distinguishes from all "
            "other acute porphyrias where PBG is elevated). ALAD is ~0-5% activity in affected homozygotes. "
            "Treatment: same as AIP — IV hematin, IV glucose; givosiran theoretically applicable "
            "(targets ALAS1 → reduces substrate ALA available to mutant ALAD); liver transplant "
            "has been curative in one confirmed case. "
            "Drug contraindications: same as AIP — all CYP450 inducers (PHT, PB, CBZ, OXC) ABSOLUTELY "
            "CONTRAINDICATED. Lead avoidance mandatory (synergistic ALAD inhibition in heterozygous carriers)."
        ),
        "hallmark": "Urine ALA markedly elevated; PBG NORMAL (distinguishes from AIP). Erythrocyte protoporphyrin elevated. Lead poisoning must be excluded (identical biochemistry).",
        "nbs_marker": "No NBS. Erythrocyte ALAD activity <5% (both alleles; homozygote). Blood lead level (exclude lead poisoning DDx).",
        "key_biomarker": "Urine ALA (>>normal, no PBG elevation). Erythrocyte ALA. Erythrocyte ZnPP (elevated). Blood lead (exclude).",
        "severity_spectrum": "Very rare — all reported cases severe with neonatal/childhood onset; adult onset described. Erythropoietic overlap in some cases.",
        "emergency": "Same as AIP: STOP triggering drug. IV glucose. IV hematin. LEV for seizures. Lead avoidance. Measure blood lead to exclude lead poisoning. Liver transplant evaluation for refractory.",
        "ci_drugs": "PHT (ABSOLUTE), PB (ABSOLUTE), CBZ (ABSOLUTE), OXC (ABSOLUTE). All CYP inducers. Lead-containing preparations absolutely avoided. Check porphyria drug database.",
        "gender": "male",
        "severity_weights": [0.20, 0.50, 0.30],
        "treatment_options": [
            "IV hematin + IV glucose (acute attack)",
            "Liver transplantation (curative, case reports)",
            "Givosiran SC monthly (prophylaxis — off-label)",
            "LEV (safe AED for seizures)",
            "Lead chelation if concurrent lead toxicity",
            "Supportive neuropathy management",
        ],
        "outcome_options": [
            "Severe recurrent attacks (most common)",
            "Progressive neuropathy",
            "Liver transplant curative (rare)",
            "Dialysis (chronic ALA nephrotoxicity)",
        ],
    },
    # ── CPOX — Hereditary Coproporphyria (HCP) ─────────────────────────────
    {
        "gene": "CPOX", "protein": "Coproporphyrinogen III Oxidase",
        "alias": "CPOX — HCP (Hereditary Coproporphyria; OMIM #121300); dual neurovisceral + cutaneous; autosomal dominant",
        "aa": "354 aa", "kDa": "40 kDa (monomer; active as dimer)",
        "gene_class": (
            "Coproporphyrinogen III oxidase (CPOX); catalyses step 6 of heme biosynthesis: "
            "coproporphyrinogen III → protoporphyrinogen IX (two sequential oxidative decarboxylations); "
            "located in mitochondrial intermembrane space; oxygen-dependent; "
            "3q11.2; OMIM gene 612732; "
            "haploinsufficiency causes HCP — ~50% enzyme activity usually adequate; "
            "precipitating factors upregulate ALAS1 → coproporphyrinogen III accumulates → "
            "ALA + PBG elevation (neurovisceral) + skin photosensitivity (porphyrin-mediated "
            "phototoxicity from cutaneous coproporphyrins)"
        ),
        "porphyria_group": "Acute Hepatic Porphyria (AHP) with Cutaneous Overlap",
        "subtype": "HCP — Hereditary Coproporphyria",
        "locus": "3q11.2", "omim_gene": 612732, "omim_disease": 121300,
        "inheritance": "Autosomal Dominant (AD). Low penetrance (~10-20% symptomatic). Both sexes equally affected. Homozygous HCP (harderoporphyria) described — more severe with haemolytic anaemia.",
        "seed_offset": 2,
        "onset_range_y": (15.0, 40.0),
        "phenotype": (
            "DUAL phenotype (unique among acute porphyrias): (1) AIP-like neurovisceral "
            "crisis (abdominal pain, motor neuropathy, autonomic, psychiatric) during attacks; "
            "(2) CUTANEOUS PHOTOSENSITIVITY: blistering on sun-exposed skin (similar to PCT — "
            "bullae, fragile skin, milia, hypertrichosis) — present in ~30% of HCP patients "
            "and can occur INDEPENDENTLY of neurovisceral attacks. "
            "BIOCHEMICAL PATTERN: elevated urine ALA + PBG (like AIP) PLUS elevated "
            "fecal and urine coproporphyrins (markedly); this fecal coproporphyrin elevation "
            "distinguishes HCP from AIP (where fecal porphyrins are normal). "
            "HOMOZYGOUS HCP (harderoporphyria): severe, neonatal onset, haemolytic anaemia, "
            "skin disease, neurological involvement."
        ),
        "disease": (
            "HCP is the third most common acute porphyria (after AIP and VP). Prevalence: "
            "~1:100,000 in Europe. Coproporphyrinogen III accumulates in liver → excreted "
            "in bile/feces (elevated fecal coproporphyrins) and urine. "
            "Diagnostic fingerprint: elevated urine PBG + ALA PLUS disproportionately "
            "elevated fecal and urine coproporphyrins (isomer III/I ratio >0.9). "
            "HARDEROPORPHYRIA: homozygous CPOX mutations (usually K404E) → "
            "severe enzyme deficiency → childhood onset, haemolytic anaemia, skin disease — "
            "extremely rare. "
            "Treatment: same as AIP — hematin, glucose, givosiran for prophylaxis. "
            "Skin protection: sunscreens, physical sun barriers (zinc oxide), avoid sunlight. "
            "AED prescribing: same absolute contraindications as AIP. "
            "Pregnancy risk: attacks more frequent in third trimester; hematin safe in pregnancy."
        ),
        "hallmark": "AIP-like crisis + blistering cutaneous photosensitivity (30%). Fecal coproporphyrins markedly elevated (distinguishes from AIP). Urine PBG + ALA elevated during attack.",
        "nbs_marker": "No NBS. Erythrocyte CPOX activity ~50% in carriers. Fecal porphyrins (coproporphyrin III predominant).",
        "key_biomarker": "Fecal coproporphyrin III (elevated; isoform ratio III/I >0.9). Urine PBG + ALA (elevated during attack). Plasma porphyrins (cutaneous attacks).",
        "severity_spectrum": "Latent → Neurovisceral attacks only → Cutaneous only → Combined neurovisceral + cutaneous → Severe harderoporphyria (homozygous)",
        "emergency": "Acute attack: IV hematin, IV glucose, LEV for seizures. Skin: physical sunblock, avoid UV. Same drug CIs as AIP.",
        "ci_drugs": "PHT (ABSOLUTE), PB (ABSOLUTE), CBZ (ABSOLUTE), OXC (ABSOLUTE). All CYP inducers. Oral oestrogen (caution — can trigger cutaneous). Check porphyria database.",
        "gender": "both",
        "severity_weights": [0.45, 0.40, 0.15],
        "treatment_options": [
            "IV hematin + IV glucose (acute neurovisceral attack)",
            "Physical sunscreens + sun avoidance (cutaneous)",
            "Givosiran SC monthly (prophylaxis)",
            "LEV (safe AED)",
            "IV hematin (acute rescue)",
            "Phlebotomy (if iron overload complicates)",
        ],
        "outcome_options": [
            "Latent — no attacks (most carriers)",
            "Recurrent attacks — givosiran prophylaxis",
            "Chronic cutaneous disease",
            "Progressive neuropathy (severe attacks)",
        ],
    },
    # ── PPOX — Variegate Porphyria (VP) ────────────────────────────────────
    {
        "gene": "PPOX", "protein": "Protoporphyrinogen IX Oxidase",
        "alias": "PPOX — VP (Variegate Porphyria; OMIM #176200); dual phenotype; p.Arg59Trp South African founder; autosomal dominant",
        "aa": "477 aa", "kDa": "55 kDa (monomer; active as homodimer)",
        "gene_class": (
            "Protoporphyrinogen IX oxidase (PPOX); catalyses step 7 of heme biosynthesis: "
            "protoporphyrinogen IX → protoporphyrin IX (six-electron oxidation, FAD-dependent); "
            "located in inner mitochondrial membrane; "
            "1q23.3; OMIM gene 600923; "
            "haploinsufficiency → VP. The South African founder mutation p.Arg59Trp in PPOX "
            "traces to a single Dutch immigrant couple (1688) — carrier frequency in South Africa "
            "~1:300 (one of the most common single-gene disorders in that population). "
            "Both cutaneous and neurovisceral manifestations can occur in the SAME patient "
            "at DIFFERENT TIMES — giving rise to the name 'variegate'"
        ),
        "porphyria_group": "Acute Hepatic Porphyria (AHP) with Cutaneous Overlap",
        "subtype": "VP — Variegate Porphyria (South Africa: p.Arg59Trp founder)",
        "locus": "1q23.3", "omim_gene": 600923, "omim_disease": 176200,
        "inheritance": "Autosomal Dominant (AD). p.Arg59Trp: South African founder — all SA VP traces to 1688 Dutch immigrant couple; carrier frequency ~1:300 SA. Penetrance ~10% in SA.",
        "seed_offset": 3,
        "onset_range_y": (20.0, 50.0),
        "phenotype": (
            "VARIEGATE = variable: same gene, different phenotype at different times. "
            "(1) ACUTE NEUROVISCERAL attacks: same as AIP/HCP — abdominal pain, motor neuropathy, "
            "autonomic, SIADH-seizures; (2) CUTANEOUS BLISTERING: subepidermal bullae, fragile skin, "
            "milia, hypertrichosis on sun-exposed skin — can occur OUTSIDE of acute attacks "
            "(triggered by sun alone, not needing drug/fasting triggers). "
            "BIOCHEMICAL FINGERPRINT distinguishing VP from HCP and AIP: "
            "PLASMA porphyrins with characteristic fluorescence emission peak at 626-628 nm "
            "(the 'VP fluorescence peak') — most sensitive test for VP; "
            "elevated fecal protoporphyrins + coproporphyrins "
            "(both elevated in VP; coproporphyrins predominate in HCP). "
            "Between attacks: urine PBG may be normal; PLASMA PORPHYRINS remain abnormal "
            "(unlike AIP where plasma porphyrins are normal between attacks — KEY DDx)."
        ),
        "disease": (
            "VP is the second most common acute porphyria globally (~1:75,000) and the most "
            "common in South Africa (p.Arg59Trp founder; ~30,000 carriers). "
            "The plasma porphyrin fluorescence peak at 626-628nm is pathognomonic for VP "
            "(AIP peak ~615nm; HCP ~617nm — spectrally distinct). "
            "DIAGNOSTIC ALGORITHM: During attack — urine PBG + ALA + plasma porphyrins. "
            "Between attacks — plasma porphyrins (remain elevated even when urine PBG normalises). "
            "Fecal porphyrins: protoporphyrins predominate (unlike HCP where coproporphyrins predominate). "
            "HOMOZYGOUS VP: rare, severe; childhood onset, dermatoses, neurological impairment; "
            "plasma porphyrins markedly elevated; erythrocyte protoporphyrin elevated. "
            "Treatment: same acute management as AIP. Skin: sun avoidance, physical sunblock. "
            "Givosiran approved for AHP includes VP. Liver transplant: curative for neurovisceral "
            "component but does not cure cutaneous (cutaneous VP driven by non-hepatic porphyrin "
            "production — erythrocyte + skin porphyrin overload)."
        ),
        "hallmark": "Dual: acute neurovisceral attacks + blistering skin. Plasma porphyrin fluorescence peak 626-628nm PATHOGNOMONIC. Fecal protoporphyrins elevated. Between attacks: plasma porphyrins remain abnormal (unlike AIP).",
        "nbs_marker": "No NBS. Plasma porphyrins (fluorescence peak 626-628nm; most sensitive test — stays abnormal between attacks). Erythrocyte PPOX activity ~50%.",
        "key_biomarker": "Plasma porphyrin fluorescence peak 626-628nm (pathognomonic). Fecal protoporphyrins + coproporphyrins. Urine PBG + ALA (during attack).",
        "severity_spectrum": "Latent → Cutaneous only → Neurovisceral attacks only → Combined (most clinically complex) → Severe (respiratory failure, paralysis)",
        "emergency": "Acute attack: same as AIP — hematin, glucose, LEV. Skin crisis: physical sun block, avoid UV, consider low-dose hydroxychloroquine (for cutaneous disease only — NOT during acute attack). Never give PHT/PB/CBZ.",
        "ci_drugs": "PHT (ABSOLUTE), PB (ABSOLUTE), CBZ (ABSOLUTE), OXC (ABSOLUTE). Same as AIP. Hydroxychloroquine: SAFE for skin disease but do NOT use during acute neurovisceral attack (hepatotoxic porphyrin mobilisation risk). Check porphyria drug database.",
        "gender": "both",
        "severity_weights": [0.45, 0.40, 0.15],
        "treatment_options": [
            "IV hematin + IV glucose (acute attack)",
            "Physical sunblock + sun avoidance (cutaneous)",
            "Givosiran SC monthly (prophylaxis)",
            "Hydroxychloroquine low-dose (chronic cutaneous disease — NOT during acute attack)",
            "LEV (safe AED)",
            "Liver transplant (severe refractory neurovisceral component)",
        ],
        "outcome_options": [
            "Latent — no attacks (most SA carriers)",
            "Cutaneous disease only (mild)",
            "Recurrent neurovisceral attacks — givosiran",
            "Progressive neuropathy (severe attacks)",
        ],
    },
    # ── FECH — Erythropoietic Protoporphyria (EPP) ─────────────────────────
    {
        "gene": "FECH", "protein": "Ferrochelatase",
        "alias": "FECH — EPP (Erythropoietic Protoporphyria; OMIM #177000); most common erythropoietic porphyria; non-blistering acute photosensitivity; 10% fatal liver disease",
        "aa": "423 aa", "kDa": "48 kDa (monomer; active as homodimer; [2Fe-2S] cluster)",
        "gene_class": (
            "Ferrochelatase (FECH); catalyses the final step of heme biosynthesis: "
            "insertion of Fe²⁺ into protoporphyrin IX → heme; "
            "located in inner mitochondrial membrane; contains [2Fe-2S] iron-sulfur cluster; "
            "18q21.31; OMIM gene 612386; "
            "EPP genetics: UNIQUE — most EPP patients are compound heterozygous: "
            "one null (LOF) FECH allele + one common low-expression allele (IVS3-48T/C); "
            "IVS3-48C allele has ~65% normal expression; combined → ~35% residual activity → EPP. "
            "Ferrochelatase deficiency → protoporphyrin IX (PP IX) accumulates in erythrocytes, "
            "plasma, and liver → phototoxic (PP IX absorbs at 400nm Soret band → "
            "reactive oxygen species on sun exposure → cell membrane damage → burning pain)"
        ),
        "porphyria_group": "Erythropoietic Porphyria (non-acute; cutaneous photosensitivity; hepatic risk)",
        "subtype": "EPP — Erythropoietic Protoporphyria",
        "locus": "18q21.31", "omim_gene": 612386, "omim_disease": 177000,
        "inheritance": "AD (trans-heterozygous: FECH null + IVS3-48C low-expression allele) in most cases. Autosomal recessive (homozygous null) rare. FECH IVS3-48C low-expression allele frequency: ~10% Europeans. No neurovisceral attacks — purely cutaneous + hepatic.",
        "seed_offset": 4,
        "onset_range_y": (0.5, 6.0),
        "phenotype": (
            "ACUTE PHOTOSENSITIVITY: immediate, severe burning/stinging pain on sun exposure "
            "(onset within minutes of sunlight). NOT blistering (distinguishes EPP from PCT, CEP, HCP, VP). "
            "Erythema, oedema, petechiae; skin appears swollen, waxy, with thickened knuckle skin "
            "(characteristic chronic skin thickening — 'wax' appearance). "
            "SEVERELY disabling: children avoid sunlight entirely; social isolation, school refusal. "
            "HEPATIC DISEASE: protoporphyrin deposits → gallstones (porphyrin bilestones — "
            "brown pigment stones, ~5%), and in ~5-10% → progressive cholestatic liver disease "
            "→ hepatic failure (FATAL in ~5-10%). "
            "NO NEUROLOGICAL ATTACKS — purely cutaneous + hepatic. "
            "Erythrocyte free protoporphyrin (PP IX) markedly elevated — "
            "this is the diagnostic test (bright orange-red fluorescence under UV/Wood's light). "
            "Anaemia: mild hypochromic anaemia in some patients."
        ),
        "disease": (
            "EPP is the most common erythropoietic porphyria and the most common porphyria "
            "presenting in childhood. Global prevalence: ~1:75,000-200,000. "
            "Erythrocyte PPIX (protoporphyrin IX): >100× normal in EPP patients (gold standard diagnostic test). "
            "Metal-free PP IX predominates (free PP IX, not zinc-chelated) — distinguishes EPP from "
            "other conditions with elevated erythrocyte PP IX. "
            "TREATMENT: AFAMELANOTIDE (Scenesse, Clinuvel Pharmaceuticals): "
            "alpha-MSH analogue; implant; induces melanogenesis → increased melanin → "
            "photoprotection; FDA approved Dec 2019 (first FDA approval for EPP); "
            "reduces pain episodes significantly. "
            "Cholestyramine (bile acid sequestrant): reduces enterohepatic recirculation of PP IX → "
            "decreases liver PP IX burden. "
            "Liver transplantation: for advanced hepatic EPP — does NOT cure EPP "
            "(erythroid PP IX overproduction continues); re-transplantation may be required. "
            "Haematopoietic stem cell transplant (HSCT): curative (addresses erythroid overproduction); "
            "reserved for severe hepatic EPP. "
            "Beta-carotene (Lumitene): weak photoprotection; limited efficacy; largely replaced by afamelanotide."
        ),
        "hallmark": "Immediate non-blistering burning pain on sunlight exposure (onset minutes). Erythrocyte free PP IX >100x ULN (bright orange-red fluorescence Wood's light). 5-10% fatal protoporphyric hepatic disease.",
        "nbs_marker": "No NBS. Erythrocyte free protoporphyrin (PP IX) — markedly elevated. Plasma PP IX (peak 634nm fluorescence).",
        "key_biomarker": "Erythrocyte free PP IX (>100× ULN; not zinc-chelated — DDx lead poisoning where ZnPP elevated). Plasma porphyrins (PP IX peak 634nm). Liver enzymes (cholestasis indicator).",
        "severity_spectrum": "Mild (occasional burning, manageable) → Moderate (significant sun avoidance, quality of life severely impaired) → Severe hepatic disease (cholestasis, cirrhosis, liver failure)",
        "emergency": "Acute phototoxic attack: move patient from sunlight immediately; IV fluids; opioid analgesia. Hepatic crisis: cholestyramine, vitamin E, riboflavin; liver transplant evaluation. HSCT for curative option.",
        "ci_drugs": "No drug CI specific to EPP (no acute porphyria). All hepatotoxic drugs with caution (already compromised liver). Oral oestrogens: avoid (may worsen liver porphyrin burden). Iron supplementation contraindicated without documented iron deficiency.",
        "gender": "both",
        "severity_weights": [0.40, 0.45, 0.15],
        "treatment_options": [
            "Afamelanotide implant (Scenesse) — FDA 2019 photoprotection",
            "Sun avoidance + physical sunscreens (zinc oxide, titanium dioxide)",
            "Cholestyramine (reduce hepatic PP IX load)",
            "Liver transplantation (advanced hepatic EPP)",
            "HSCT (curative for severe hepatic EPP)",
            "Beta-carotene + cysteine (mild photoprotection)",
        ],
        "outcome_options": [
            "Good QoL with sun avoidance + afamelanotide",
            "Severe sun avoidance — social isolation",
            "Progressive hepatic disease → transplant",
            "Fatal liver failure (5-10%)",
        ],
    },
    # ── UROD — Porphyria Cutanea Tarda (PCT) ────────────────────────────────
    {
        "gene": "UROD", "protein": "Uroporphyrinogen III Decarboxylase",
        "alias": "UROD — PCT (Porphyria Cutanea Tarda; OMIM #176100); most common porphyria overall; blistering cutaneous; phlebotomy + chloroquine curative; iron + hepatitis C triggers",
        "aa": "367 aa", "kDa": "41 kDa (monomer; active as homodimer)",
        "gene_class": (
            "Uroporphyrinogen III decarboxylase (UROD); catalyses step 5 of heme biosynthesis: "
            "uroporphyrinogen III → coproporphyrinogen III (four sequential decarboxylations); "
            "cytoplasmic enzyme; ubiquitously expressed; "
            "1p34.1; OMIM gene 613521; "
            "PCT types: Type 1 (sporadic, 75%) — normal UROD gene but enzyme inhibited by "
            "iron-catalysed mechanism (uroporphomethene formation) in liver; "
            "Type 2 (familial, 25%) — UROD heterozygous LOF mutation (further reduces activity "
            "below threshold when combined with iron/HCV/alcohol/oestrogen triggers); "
            "UROD requires hepatic iron overload to inhibit — HFE C282Y co-mutation "
            "dramatically increases risk (30-50% of familial PCT have HFE C282Y). "
            "Hepatitis C virus direct UROD inhibition via oxidative stress — HCV most common trigger."
        ),
        "porphyria_group": "Cutaneous Porphyria — Hepatic (no acute neurovisceral attacks)",
        "subtype": "PCT — Porphyria Cutanea Tarda (Type 1 sporadic; Type 2 familial UROD LOF)",
        "locus": "1p34.1", "omim_gene": 613521, "omim_disease": 176100,
        "inheritance": "AD for familial PCT (Type 2; UROD heterozygous LOF). Sporadic PCT (Type 1) has normal UROD gene — environmental triggers (iron, HCV, alcohol, oestrogen) inhibit normal UROD. HFE C282Y hemochromatosis allele major risk factor for both types.",
        "seed_offset": 5,
        "onset_range_y": (30.0, 60.0),
        "phenotype": (
            "CUTANEOUS BLISTERING on sun-exposed skin — NO neurovisceral attacks (distinguishes PCT "
            "from acute porphyrias). Subepidermal bullae on dorsal hands, forearms, face — "
            "fragile skin, heals with milia (tiny inclusion cysts) and scarring. "
            "Hypertrichosis (excess hair growth — especially malar region in women — pathognomonic). "
            "Hyperpigmentation. Scleroderma-like skin changes in chronic/severe cases. "
            "ONSET: typically adulthood (30-60y); often triggered by specific precipitant. "
            "PRECIPITANTS: hepatitis C (most common trigger worldwide), iron overload "
            "(especially HFE C282Y hemochromatosis), alcohol, oestrogen therapy, "
            "HIV, hexachlorobenzene (environmental), polychlorinated biphenyls. "
            "URINE: dark red/brown (elevated uroporphyrins) — 'port wine urine' appearance. "
            "DIAGNOSIS: elevated urine uroporphyrins (>1000 nmol/24h) + plasma porphyrins "
            "(emission peak at ~617 nm) + clinical picture. Liver biopsy: porphyrin fluorescence "
            "under UV light (orange-red fluorescence of hepatocytes)."
        ),
        "disease": (
            "PCT is the most common porphyria globally (1:10,000-25,000). "
            "Uroporphyrins accumulate in liver, plasma, skin — light-activated → phototoxic bullae. "
            "TREATMENT: Simple and highly effective: "
            "(1) PHLEBOTOMY: removes excess hepatic iron → removes the inhibitor of UROD → "
            "uroporphyrin production falls → remission in 4-8 months. "
            "Phlebotomy units: 450 mL every 2 weeks until serum ferritin <15-20 μg/L; then "
            "haemoglobin is maintained (target Hb 120 g/L women, 130 g/L men). "
            "(2) LOW-DOSE CHLOROQUINE (or hydroxychloroquine): mobilises hepatic porphyrins "
            "for urinary excretion; used when phlebotomy contraindicated (anaemia) or in "
            "those without iron overload. Chloroquine 125 mg twice weekly. "
            "BOTH treatments are curative in most patients. "
            "HEPATITIS C TREATMENT: if HCV positive, treatment with DAAs (direct-acting antivirals) "
            "often produces PCT remission without phlebotomy. "
            "Alcohol abstinence + oestrogen cessation: required to achieve and maintain remission. "
            "Relapse: occurs if triggers resume; re-treat with phlebotomy."
        ),
        "hallmark": "Blistering cutaneous photosensitivity (NO acute neurovisceral attacks). Hypertrichosis pathognomonic. Dark urine (uroporphyrins). Phlebotomy + chloroquine CURATIVE. HCV + iron overload as key triggers.",
        "nbs_marker": "No NBS. Urine uroporphyrins >1000 nmol/24h. Plasma porphyrins (peak 617nm). Serum ferritin (iron overload). HFE genotype (C282Y). HCV antibody.",
        "key_biomarker": "Urine uroporphyrins (markedly elevated; Type I isomer predominates). Plasma porphyrins (peak 617nm). Serum ferritin (iron overload). HFE C282Y genotype.",
        "severity_spectrum": "Mild (few bullae, manageable) → Moderate (extensive skin disease, significant scarring) → Severe (sclerodermatous skin, significant hepatic involvement)",
        "emergency": "No acute porphyria crisis. Severe blistering: wound care, sun avoidance. Start phlebotomy or chloroquine. Treat HCV. Alcohol abstinence. Stop oestrogen.",
        "ci_drugs": "No acute porphyria AED CIs. Oral oestrogens (precipitate PCT). Alcohol (precipitate). Excess iron supplementation. Hexachlorobenzene. Tamoxifen (some cases). Check drug database for specific PCT triggers.",
        "gender": "both",
        "severity_weights": [0.50, 0.40, 0.10],
        "treatment_options": [
            "Phlebotomy (450mL biweekly until ferritin <15 μg/L) — first-line",
            "Low-dose chloroquine 125mg twice weekly (if phlebotomy contraindicated)",
            "HCV DAA therapy (if HCV positive)",
            "Alcohol abstinence + oestrogen cessation",
            "Sun avoidance + physical sunscreens",
            "Hydroxychloroquine 200mg twice weekly (alternative to chloroquine)",
        ],
        "outcome_options": [
            "Full remission with phlebotomy (4-8 months)",
            "Remission with chloroquine (6-12 months)",
            "Relapse after trigger re-exposure",
            "Chronic scarring + hypertrichosis (untreated)",
        ],
    },
    # ── UROS — Congenital Erythropoietic Porphyria / Günther disease (CEP) ─
    {
        "gene": "UROS", "protein": "Uroporphyrinogen III Synthase (Uroporphyrinogen III Cosynthase)",
        "alias": "UROS — CEP (Congenital Erythropoietic Porphyria / Günther disease; OMIM #263700); rarest and most severe; AR; erythrodontia pathognomonic; <500 cases worldwide",
        "aa": "265 aa", "kDa": "29 kDa",
        "gene_class": (
            "Uroporphyrinogen III synthase (UROS); catalyses step 4 of heme biosynthesis: "
            "hydroxymethylbilane (HMB) → uroporphyrinogen III (asymmetric macrocycle); "
            "this asymmetric cyclisation is critical — without UROS, HMB spontaneously "
            "cyclises to uroporphyrinogen I (the WRONG isomer) → "
            "uroporphyrin I + coproporphyrin I accumulate (non-metabolisable dead-end products); "
            "cytoplasmic; expressed in erythroid precursors and all tissues; "
            "10q26.2; OMIM gene 606938; "
            "biallelic LOF → severe CEP; residual UROS activity determines severity "
            "(some missense mutations have 1-5% residual activity → severe disease; "
            "null mutations → most severe haemolytic/mutilating phenotype)"
        ),
        "porphyria_group": "Erythropoietic Porphyria — Congenital, Severe (AR)",
        "subtype": "CEP — Congenital Erythropoietic Porphyria (Günther disease)",
        "locus": "10q26.2", "omim_gene": 606938, "omim_disease": 263700,
        "inheritance": "Autosomal Recessive (AR). Both alleles LOF required. <500 cases worldwide (2026). Prenatal diagnosis essential (severe mutilating disease). Most common mutation: p.Cys73Arg (C73R).",
        "seed_offset": 6,
        "onset_range_y": (0.0, 1.0),
        "phenotype": (
            "MOST SEVERE PORPHYRIA. Neonatal/infantile onset. "
            "HAEMOLYTIC ANAEMIA: severe, often transfusion-dependent; erythroid hyperplasia → "
            "splenomegaly; porphyrin I isomers incorporated into erythrocytes → haemolysis. "
            "CUTANEOUS MUTILATION: severe blistering on minimal sun exposure → "
            "progressive scarring → loss of digits, ears, nose, eyelids. "
            "ERYTHRODONTIA: PATHOGNOMONIC — reddish-brown fluorescent teeth under UV light "
            "(porphyrin I deposition in dentine during development); pink-red milk teeth. "
            "PINK/RED URINE: from birth (uroporphyrin I in urine — nappy staining classic). "
            "CORNEAL ULCERATION + blindness from eyelid loss/photophobia. "
            "BONE: pathological fractures (porphyrin infiltration + haemolytic anaemia-driven "
            "erythroid hyperplasia eroding cortical bone). "
            "NEUROLOGICAL: seizures in some patients (from severe anaemia/haemodynamic instability); "
            "intellectual development often normal if anaemia managed."
        ),
        "disease": (
            "CEP (Günther disease) is the rarest and most severe porphyria. <500 confirmed cases "
            "worldwide. First described by Hans Günther (1911). "
            "URINARY PORPHYRINS: uroporphyrin I and coproporphyrin I (non-metabolisable isomers) "
            "markedly elevated from birth — pink/red nappies in newborn period. "
            "HYDROPS FETALIS: most severe null mutations → severe intrauterine anaemia → "
            "fetal hydrops (diagnosis in utero by amniocentesis porphyrin assay). "
            "DIAGNOSIS: urine uroporphyrin I isomer (predominant) + erythrocyte uroporphyrin I; "
            "erythrodontia (red fluorescent teeth under Wood's UV light — PATHOGNOMONIC). "
            "TREATMENT: "
            "Mild (some residual UROS activity): chronic transfusion to suppress erythropoiesis "
            "+ sun avoidance; "
            "Severe: HSCT (haematopoietic stem cell transplant) — potentially curative; "
            "corrects erythroid porphyrin overproduction; early HSCT before mutilation is essential; "
            "Gene therapy: UROS lentiviral correction of haematopoietic stem cells in clinical trial "
            "(Institut Imagine Paris); "
            "Chronic transfusion: reduces haematopoiesis and porphyrin production; "
            "Splenectomy: reduces haemolysis in some patients."
        ),
        "hallmark": "Erythrodontia (red fluorescent teeth UV light) PATHOGNOMONIC. Pink nappies from birth. Severe haemolytic anaemia + photomutilation. <500 cases worldwide. HSCT potentially curative.",
        "nbs_marker": "No standard NBS. Neonatal pink urine (uroporphyrin I). Erythrocyte uroporphyrin I (markedly elevated). Teeth: UV wood lamp fluorescence (pathognomonic). Prenatal: amniocentesis porphyrins.",
        "key_biomarker": "Urine uroporphyrin I (>> isomer III; isomer I predominates — DDx PCT where isomer I also elevated but less extreme). Erythrocyte uroporphyrin I. Plasma uroporphyrins. Haemoglobin (anaemia).",
        "severity_spectrum": "Mild (some residual UROS — manageable anaemia, moderate skin disease) → Moderate (transfusion-dependent, progressive scarring) → Severe (hydrops fetalis, mutilation, blindness) → Fatal (intrauterine)",
        "emergency": "Neonatal pink urine: immediate dermatology + haematology consult. Sun avoidance: UV-protective clothing + sunblock from birth. Chronic transfusion to suppress erythropoiesis. Early HSCT evaluation. Corneal protection (lubricants, sunglasses). Wound care for bullae.",
        "ci_drugs": "No specific AED contraindications (unlike acute porphyrias). Haemotoxic drugs caution. Iron supplementation CI unless documented iron deficiency (anaemia of porphyria — NOT iron deficient typically).",
        "gender": "both",
        "severity_weights": [0.20, 0.50, 0.30],
        "treatment_options": [
            "HSCT (haematopoietic stem cell transplant) — potentially curative",
            "Chronic red cell transfusion (suppress erythropoiesis)",
            "Sun avoidance + UV-protective clothing (lifelong)",
            "Splenectomy (reduce haemolysis in severe cases)",
            "Lentiviral gene therapy (clinical trial — Institut Imagine)",
            "Corneal protection + ophthalmology monitoring",
        ],
        "outcome_options": [
            "HSCT remission (curative if early)",
            "Transfusion-dependent stable disease",
            "Progressive photomutilation (untreated/late)",
            "Fatal intrauterine (most severe null mutations)",
        ],
    },
    # ── ALAS2 — X-linked Protoporphyria (XLP) ───────────────────────────────
    {
        "gene": "ALAS2", "protein": "Aminolevulinate Synthase 2 (Erythroid-Specific)",
        "alias": "ALAS2 — XLP (X-linked Protoporphyria; OMIM #300752); GAIN-of-function exon 10-11 deletion; X-linked; EPP-like but earlier + more severe hepatic risk; aminolevulinate synthase 2",
        "aa": "587 aa", "kDa": "65 kDa (monomer; active as homodimer; PLP-dependent)",
        "gene_class": (
            "Aminolevulinate synthase 2 (ALAS2; erythroid-specific isoform); "
            "catalyses the FIRST COMMITTED step of heme biosynthesis: "
            "glycine + succinyl-CoA → delta-aminolevulinic acid (ALA) + CO₂; "
            "pyridoxal phosphate (PLP, B6)-dependent; "
            "located in mitochondrial matrix; expressed in erythroid cells only "
            "(ALAS1 is the ubiquitous isoform; ALAS2 is erythroid-specific); "
            "Xp11.21; OMIM gene 301300; "
            "XLP MECHANISM: UNIQUE — GAIN-of-function (GOF). "
            "Exon 10 or 11 deletions remove C-terminal autoinhibitory clamp → "
            "ALAS2 is constitutively active → excess ALA production → "
            "excess protoporphyrin IX (PP IX) in erythroid cells → "
            "same phenotype as EPP but caused by OVERPRODUCTION rather than "
            "UNDERUTILISATION of PP IX (EPP/FECH deficiency = underutilisation). "
            "X-linked: males affected (hemizygous GOF); females carriers can be affected "
            "(random X-inactivation — severe skewed lyonisation → symptomatic females possible)"
        ),
        "porphyria_group": "Erythropoietic Porphyria (non-acute; cutaneous photosensitivity; hepatic risk; X-linked GOF)",
        "subtype": "XLP — X-linked Protoporphyria (ALAS2 gain-of-function; exon 10-11 deletion)",
        "locus": "Xp11.21", "omim_gene": 301300, "omim_disease": 300752,
        "inheritance": "X-linked (XL). Hemizygous males severely affected (GOF). Female carriers: variable depending on X-inactivation pattern; severe skewed lyonisation → symptomatic females (earlier onset, worse than EPP). Higher hepatic risk than EPP.",
        "seed_offset": 7,
        "onset_range_y": (0.5, 5.0),
        "phenotype": (
            "CLINICALLY IDENTICAL TO EPP (FECH deficiency) but typically more severe "
            "and with higher hepatic risk: "
            "Immediate burning photosensitivity (minutes of sunlight; non-blistering); "
            "erythema, oedema, petechiae; skin thickening (waxy knuckle changes); "
            "severe lifestyle impairment. "
            "HEPATIC DISEASE: protoporphyrin deposits in bile ducts and hepatocytes → "
            "gallstones + cholestasis → hepatic failure; "
            "hepatic risk in XLP estimated at 20-30% (higher than EPP 5-10%) — "
            "this is the most clinically important distinction. "
            "BIOCHEMISTRY: erythrocyte PP IX elevated — "
            "CRITICAL DISTINCTION from EPP: "
            "In EPP: zinc PP IX (ZnPP) is LOW, free PP IX is HIGH. "
            "In XLP: BOTH zinc PP IX AND free PP IX are elevated "
            "(excess ALA → excess PP IX → both chelated-Zn and free forms accumulate). "
            "This Zn-PP IX distinction is the key DDx between EPP and XLP. "
            "NO NEUROVISCERAL ATTACKS (unlike acute porphyrias)."
        ),
        "disease": (
            "XLP (X-linked protoporphyria) was described in 2008 by Whatley et al. "
            "as a distinct entity from EPP, caused by ALAS2 GOF exon deletions. "
            "Prevalence: less common than EPP; estimated 1:50 of EPP-like presentations. "
            "ALAS2 exon 10 deletion (most common) removes last 33 amino acids → "
            "loss of C-terminal autoinhibitory cysteine pair → constitutively active ALAS2 → "
            "excess erythroid ALA → excess PP IX. "
            "KEY DIAGNOSTIC DISTINCTION: "
            "XLP vs EPP: in XLP, zinc PP IX is elevated (in addition to free PP IX); "
            "in EPP, zinc PP IX is normal or low. "
            "ALAS2 GOF gene sequencing confirms XLP. FECH activity: NORMAL in XLP. "
            "TREATMENT: same as EPP — afamelanotide (FDA 2019 for EPP, likely effective in XLP), "
            "sun avoidance; "
            "Liver monitoring: more aggressive in XLP (ALT, bilirubin quarterly; "
            "erythrocyte PP IX levels; liver biopsy threshold lower than EPP); "
            "Liver transplantation: indicated for hepatic failure; "
            "HSCT: potentially curative (corrects erythroid ALAS2 overexpression); "
            "Givosiran: not approved for XLP (targets hepatic ALAS1, not erythroid ALAS2); "
            "Phlebotomy: NOT effective (no iron overload mechanism unlike PCT)."
        ),
        "hallmark": "EPP-like acute non-blistering photosensitivity. BOTH free PP IX AND zinc-PP IX elevated (EPP: zinc-PP IX NORMAL). Higher hepatic risk (20-30%) than EPP. ALAS2 exon 10-11 deletion GOF. X-linked males most severely affected.",
        "nbs_marker": "No NBS. Erythrocyte PP IX (total + fractionated: both free AND Zn-PP IX elevated). ALAS2 gene sequencing (GOF exon 10-11 deletion). Liver enzymes (quarterly monitoring). FECH activity NORMAL (DDx EPP).",
        "key_biomarker": "Erythrocyte zinc PP IX (elevated — KEY DDx vs EPP where ZnPP normal). Erythrocyte free PP IX (elevated). Plasma PP IX (634nm fluorescence peak). ALAS2 exon 10-11 deletion on gene panel.",
        "severity_spectrum": "Mild (symptomatic males with manageable photosensitivity) → Moderate (severe sun avoidance, social isolation) → Severe hepatic disease (20-30% risk) → Fatal liver failure",
        "emergency": "Acute phototoxic episode: remove from sunlight, IV analgesia, supportive. Hepatic crisis: urgent transplant evaluation. Avoid phlebotomy (no iron mechanism). Monitor ZnPP + ALT quarterly.",
        "ci_drugs": "No AED CIs (no acute neurovisceral attacks). Hepatotoxic drugs: significant caution given higher hepatic risk. Givosiran NOT applicable (targets hepatic ALAS1, not erythroid ALAS2). Iron supplementation CI unless documented iron deficiency.",
        "gender": "male",
        "severity_weights": [0.35, 0.45, 0.20],
        "treatment_options": [
            "Afamelanotide implant (Scenesse) — photoprotection (used off-label for XLP)",
            "Sun avoidance + physical sunscreens (zinc oxide, titanium dioxide)",
            "Liver transplantation (advanced hepatic XLP)",
            "HSCT (potentially curative — erythroid correction)",
            "Cholestyramine (reduce hepatic PP IX burden)",
            "Quarterly liver monitoring (ALT + bilirubin + erythrocyte PP IX)",
        ],
        "outcome_options": [
            "Managed photosensitivity with afamelanotide + sun avoidance",
            "Severe sun avoidance — social isolation",
            "Progressive hepatic disease → transplant (20-30% risk)",
            "Fatal liver failure (higher risk than EPP)",
        ],
    },
]


def _make_patients(gd):
    rng = random.Random(SEED_BASE + gd["seed_offset"])
    n = 40
    pts = []
    sev_labels = ["Mild", "Moderate", "Severe"]
    sev_weights = gd.get("severity_weights", [0.40, 0.40, 0.20])
    gender_bias = gd.get("gender", "both")
    for i in range(n):
        sid = f"{gd['gene']}-{SEED_BASE + gd['seed_offset']:03d}-{i+1:03d}"
        sev = rng.choices(sev_labels, weights=sev_weights, k=1)[0]
        lo, hi = gd["onset_range_y"]
        age_onset = round(rng.uniform(lo, hi), 1)
        # diagnosis delay depends on group and severity
        if gd["porphyria_group"].startswith("Acute"):
            delay = round(rng.uniform(0.5, 8.0), 1)  # often delayed — misdiagnosed as IBS/psychiatric
        elif gd["gene"] == "UROS":
            delay = round(rng.uniform(0.0, 0.5), 1)   # pink nappies = early diagnosis
        else:
            delay = round(rng.uniform(0.5, 5.0), 1)
        # Gender
        if gender_bias == "male":
            sex = "M" if rng.random() < 0.80 else "F"
        elif gender_bias == "female":
            sex = "F" if rng.random() < 0.70 else "M"
        else:
            sex = rng.choice(["M", "F"])
        # AED safety event (for acute porphyrias)
        if gd["porphyria_group"].startswith("Acute"):
            unsafe_aed_given = rng.random() < 0.18  # ~18% received unsafe AED historically
        else:
            unsafe_aed_given = False
        # Urine PBG fold change (acute only)
        if gd["porphyria_group"].startswith("Acute") and gd["gene"] != "ALAD":
            pbg_fold = round(rng.uniform(5, 200), 1)
        elif gd["gene"] == "ALAD":
            pbg_fold = round(rng.uniform(0.8, 1.5), 1)  # PBG NORMAL in ADP
        else:
            pbg_fold = None
        # ALA fold change
        if gd["porphyria_group"].startswith("Acute"):
            ala_fold = round(rng.uniform(3, 100), 1)
        else:
            ala_fold = None
        # Erythrocyte PP IX (EPP / XLP / CEP)
        if gd["gene"] in ("FECH", "ALAS2"):
            ery_ppix_fold = round(rng.uniform(40, 300), 1)
        elif gd["gene"] == "UROS":
            ery_ppix_fold = round(rng.uniform(20, 150), 1)
        else:
            ery_ppix_fold = None
        # Liver involvement
        if gd["gene"] in ("FECH",):
            hepatic_disease = rng.random() < 0.08   # 5-10%
        elif gd["gene"] in ("ALAS2",):
            hepatic_disease = rng.random() < 0.22   # 20-30%
        elif gd["gene"] in ("HMBS", "CPOX", "PPOX", "ALAD"):
            hepatic_disease = rng.random() < 0.15   # CKD/HCC risk
        elif gd["gene"] == "UROD":
            hepatic_disease = rng.random() < 0.10   # HCV-related
        else:
            hepatic_disease = rng.random() < 0.05
        treatment = rng.choice(gd["treatment_options"])
        outcome = rng.choice(gd["outcome_options"])
        # Trigger (acute porphyrias)
        if gd["porphyria_group"].startswith("Acute"):
            trigger = rng.choice(["Drug (CYP inducer)", "Fasting/low carbohydrate", "Hormonal (luteal phase)",
                                   "Alcohol", "Infection/illness", "Surgery/anaesthesia", "Stress",
                                   "Oral contraceptive pill"])
        elif gd["gene"] == "UROD":
            trigger = rng.choice(["Hepatitis C", "Iron overload (HFE C282Y)", "Alcohol",
                                   "Oral oestrogen", "HIV", "Environmental toxin", "Idiopathic"])
        else:
            trigger = "Sunlight exposure"
        pts.append({
            "id": sid,
            "gene": gd["gene"],
            "sex": sex,
            "age_onset_y": age_onset,
            "dx_delay_y": delay,
            "severity": sev,
            "urine_pbg_fold_uln": pbg_fold,
            "urine_ala_fold_uln": ala_fold,
            "ery_ppix_fold_uln": ery_ppix_fold,
            "hepatic_disease": hepatic_disease,
            "unsafe_aed_given": unsafe_aed_given,
            "primary_trigger": trigger,
            "treatment": treatment,
            "outcome": outcome,
        })
    return pts


def get_overview():
    all_pts = []
    gene_summary = {}
    group_counts = {}
    for gd in PORPHYRIA_GENES:
        pts = _make_patients(gd)
        all_pts.extend(pts)
        grp = gd["porphyria_group"]
        group_counts[grp] = group_counts.get(grp, 0) + 1
        gene_summary[gd["gene"]] = {
            "subtype": gd["subtype"],
            "group": grp,
            "locus": gd["locus"],
            "inheritance": gd["inheritance"].split(".")[0],
            "hallmark": gd["hallmark"][:120] + "…",
            "n_patients": len(pts),
        }
    n = len(all_pts)
    avg_onset = round(sum(p["age_onset_y"] for p in all_pts) / n, 1)
    avg_delay = round(sum(p["dx_delay_y"] for p in all_pts) / n, 1)
    sev_dist = {"Mild": 0, "Moderate": 0, "Severe": 0}
    hepatic_n = 0
    unsafe_aed_n = 0
    for p in all_pts:
        sev_dist[p["severity"]] += 1
        if p["hepatic_disease"]:
            hepatic_n += 1
        if p["unsafe_aed_given"]:
            unsafe_aed_n += 1
    # Acute vs cutaneous split
    acute_n = sum(1 for p in all_pts if gd_by_gene(p["gene"])["porphyria_group"].startswith("Acute"))
    return {
        "title": "Porphyria-Atlas — Complete 8-Gene Porphyria Disorders Atlas",
        "subtitle": "HMBS/AIP · ALAD/ADP · CPOX/HCP · PPOX/VP · FECH/EPP · UROD/PCT · UROS/CEP · ALAS2/XLP",
        "genes": [gd["gene"] for gd in PORPHYRIA_GENES],
        "subtypes": [gd["subtype"] for gd in PORPHYRIA_GENES],
        "total_patients": n,
        "avg_onset_y": avg_onset,
        "avg_dx_delay_y": avg_delay,
        "severity_distribution": sev_dist,
        "hepatic_disease_n": hepatic_n,
        "hepatic_disease_pct": round(100 * hepatic_n / n, 1),
        "unsafe_aed_n": unsafe_aed_n,
        "unsafe_aed_pct": round(100 * unsafe_aed_n / n, 1),
        "acute_n": acute_n,
        "cutaneous_erythropoietic_n": n - acute_n,
        "porphyria_groups": group_counts,
        "gene_summary": gene_summary,
        "key_facts": [
            "4 ACUTE HEPATIC PORPHYRIAS (AHP): AIP/HMBS, HCP/CPOX, VP/PPOX, ADP/ALAD — neurovisceral crises",
            "ABSOLUTE CI AEDs: PHT, PB, CBZ, OXC (CYP inducers upregulate ALAS1 → worsens crisis)",
            "SAFE AED: Levetiracetam (LEV) — first-line for porphyric seizures",
            "IV HEMATIN (Panhematin/Normosang): specific acute attack treatment — suppresses ALAS1",
            "GIVOSIRAN (Alnylam, FDA Nov 2019): siRNA targeting ALAS1 — prophylaxis for recurrent AHP (>2 attacks/year)",
            "PBG urine: diagnostic for acute attacks (AIP, HCP, VP) — Watson-Schwartz test",
            "ALAD PORPHYRIA unique: urine ALA elevated but PBG NORMAL (DDx all other acute porphyrias)",
            "VP FLUORESCENCE PEAK 626-628nm: pathognomonic for variegate porphyria",
            "PHLEBOTOMY + chloroquine: curative for PCT (most common porphyria)",
            "ERYTHRODONTIA (UV-fluorescent teeth): pathognomonic for CEP/Günther disease",
            "AFAMELANOTIDE (Scenesse): FDA 2019 — photoprotection for EPP/FECH deficiency",
            "XLP (ALAS2 GOF): both free PP IX AND zinc PP IX elevated — KEY DDx from EPP (ZnPP normal)",
            "SIADH + hyponatraemia: complication of acute attacks → SEIZURES — correct slowly",
            "CEP/UROS: HSCT potentially curative — initiate early before mutilation",
        ],
        "critical_distinctions": {
            "ALAD vs AIP": "ALA elevated, PBG NORMAL in ALAD; ALA + PBG both elevated in AIP",
            "VP vs AIP": "Plasma porphyrin fluorescence 626nm (VP); fecal protoporphyrins elevated; VP elevated between attacks",
            "HCP vs AIP": "Fecal coproporphyrins markedly elevated (HCP); dual cutaneous + neurovisceral (30%)",
            "EPP vs XLP": "ZnPP: normal in EPP, elevated in XLP; FECH deficiency vs ALAS2 GOF",
            "PCT vs CEP": "PCT: adult onset, iron triggers, phlebotomy curative; CEP: neonatal, erythrodontia, HSCT",
        },
        "seeds_used": f"{SEED_BASE}–{SEED_BASE + len(PORPHYRIA_GENES) - 1}",
    }


_gd_cache = {gd["gene"]: gd for gd in PORPHYRIA_GENES}


def gd_by_gene(gene):
    return _gd_cache[gene]


def get_breakdown():
    result = []
    for gd in PORPHYRIA_GENES:
        pts = _make_patients(gd)
        sev = {"Severe": 0, "Moderate": 0, "Mild": 0}
        for p in pts:
            sev[p["severity"]] += 1
        avg_delay = round(sum(p["dx_delay_y"] for p in pts) / len(pts), 1)
        avg_onset = round(sum(p["age_onset_y"] for p in pts) / len(pts), 1)
        hepatic_n = sum(1 for p in pts if p["hepatic_disease"])
        unsafe_aed_n = sum(1 for p in pts if p["unsafe_aed_given"])
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        top_tx = sorted(treatments.items(), key=lambda x: -x[1])[:3]
        outcomes = {}
        for p in pts:
            outcomes[p["outcome"]] = outcomes.get(p["outcome"], 0) + 1
        triggers = {}
        for p in pts:
            triggers[p["primary_trigger"]] = triggers.get(p["primary_trigger"], 0) + 1
        top_triggers = sorted(triggers.items(), key=lambda x: -x[1])[:4]
        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "porphyria_group": gd["porphyria_group"],
            "subtype": gd["subtype"],
            "n_patients": len(pts),
            "severity_distribution": sev,
            "avg_onset_y": avg_onset,
            "avg_dx_delay_y": avg_delay,
            "hepatic_disease_n": hepatic_n,
            "hepatic_disease_pct": round(100 * hepatic_n / len(pts), 1),
            "unsafe_aed_n": unsafe_aed_n,
            "top_treatments": [{"treatment": t, "n": n_} for t, n_ in top_tx],
            "outcome_distribution": outcomes,
            "top_triggers": [{"trigger": t, "n": n_} for t, n_ in top_triggers],
            "hallmark": gd["hallmark"],
            "nbs_marker": gd["nbs_marker"],
            "key_biomarker": gd["key_biomarker"],
            "severity_spectrum": gd["severity_spectrum"],
            "emergency": gd["emergency"],
            "ci_drugs": gd["ci_drugs"],
            "disease_summary": gd["disease"][:600] + "…",
            "phenotype_summary": gd["phenotype"][:400] + "…",
            "gene_class": gd["gene_class"][:400] + "…",
        })
    return {"breakdown": result, "total_genes": len(result)}


def get_definitions():
    return {
        "definitions": [
            {
                "term": "Acute Hepatic Porphyria (AHP)",
                "definition": "Four inherited porphyrias causing neurovisceral crises: AIP (HMBS), HCP (CPOX), VP (PPOX), ADP (ALAD). Mechanism: precipitating factors (drugs, fasting, hormones, infections) upregulate hepatic ALAS1 → excess ALA/PBG → neurotoxicity. Clinical triad: (1) severe abdominal pain; (2) peripheral motor neuropathy (ascending, axonal); (3) autonomic dysfunction (tachycardia, hypertension, SIADH). Seizures occur in 20-30% of attacks (dangerous — most effective AEDs are contraindicated). SIADH-driven hyponatraemia is the commonest cause of porphyric seizures."
            },
            {
                "term": "ALAS1 (Aminolevulinate Synthase 1) — Hepatic Isoform",
                "definition": "Rate-limiting enzyme of hepatic heme biosynthesis; catalyses ALA synthesis from glycine + succinyl-CoA. ALAS1 is regulated by hepatic heme via feedback inhibition — when heme is depleted, ALAS1 is upregulated. CYP450-inducing drugs deplete heme (CYP enzymes require heme) → ALAS1 upregulation → excess ALA production (the porphyrinogenic mechanism). Hematin/givosiran both suppress ALAS1 (hematin = direct heme product feedback; givosiran = siRNA degrading ALAS1 mRNA). ALAS2 is the erythroid-specific isoform (XLP)."
            },
            {
                "term": "Porphobilinogen (PBG) — The Diagnostic Marker",
                "definition": "PBG is the second intermediate in heme biosynthesis (ALA → PBG via ALAD). In acute attacks of AIP, HCP, and VP, urine PBG is markedly elevated (often >100× normal). PBG in urine turns dark red/port-wine on standing (photooxidation to porphyrins). Watson-Schwartz test: colorimetric rapid test for urine PBG (qualitative). Quantitative urine PBG (normalised to creatinine, spot sample): gold standard during attack. CRITICAL: PBG is NORMAL in ALAD porphyria (ADP) — ALA is elevated but PBG cannot form because ALAD is deficient. Between attacks: PBG may normalise in AIP (unlike VP where plasma porphyrins remain abnormal)."
            },
            {
                "term": "Hematin / Panhematin / Normosang (Intravenous Heme)",
                "definition": "IV heme preparation used specifically to treat acute porphyria attacks. Mechanism: provides exogenous heme → negative feedback on ALAS1 → suppresses ALA/PBG production within 24-48 hours. Dose: 3-4 mg/kg/day IV × 4 days (as haematin dissolved in albumin — improves stability and reduces thrombophlebitis). Must be given EARLY in attack (within 24-48h of moderate-severe attack) — delayed administration does not prevent established neurological damage. Caution: coagulopathy (haematin is phlebotoxic; infuse into large vein or central line); anaphylaxis (rare). Normosang (haeme arginate, Europe) is more stable than Panhematin (US)."
            },
            {
                "term": "Givosiran (Givlaari, Alnylam Pharmaceuticals)",
                "definition": "FDA approved November 2019. First RNA therapeutic approved for porphyria. Mechanism: siRNA targeting ALAS1 mRNA in hepatocytes → sustained ALAS1 mRNA degradation → reduced ALA/PBG production (~90% reduction). Administration: 2.5 mg/kg SC monthly. Approved for recurrent AHP (>2 attacks/year requiring hospitalisation). Clinical trial: ENVISION trial (NEJM 2020) — significant reduction in attack rate vs placebo. Side effects: hepatotoxicity (ALT/AST elevation — monitor monthly LFTs), homocysteine elevation (B6 supplementation may help), nephrotoxicity (creatinine monitoring). NOT approved for ALAD porphyria (off-label) or erythropoietic porphyrias (targets hepatic ALAS1, not erythroid ALAS2)."
            },
            {
                "term": "Porphyrinogenic Drugs — Absolute Contraindications",
                "definition": "Drugs that upregulate hepatic ALAS1 by inducing CYP450 enzymes (depleting heme pool → ALAS1 upregulation) or by other mechanisms. In acute porphyria patients, these can precipitate or dramatically worsen neurovisceral crises — some cases fatal. ABSOLUTE CI: Phenytoin (PHT), Phenobarbital (PB), Carbamazepine (CBZ), Oxcarbazepine (OXC), Primidone (mechanisms overlap). RELATIVE CI: Lamotrigine (generally avoided despite some centres using cautiously), Valproate (complex — some evidence of mild porphyrinogenicity but used in some centres; check database). SAFE AEDs: Levetiracetam (LEV) — drug of choice for porphyric seizures; Gabapentin; Clonazepam. Clinical database: www.drugs-porphyria.org (up-to-date drug safety classifications — always verify before prescribing)."
            },
            {
                "term": "Plasma Porphyrin Fluorescence Peak — Differential Diagnosis",
                "definition": "When plasma porphyrins are elevated, fluorescence emission spectrometry identifies the porphyrin type by its peak emission wavelength: VP (PPOX): 626-628 nm (pathognomonic — stays abnormal even BETWEEN attacks, unlike AIP); HCP (CPOX): ~617 nm; AIP (HMBS): ~615 nm (plasma porphyrins normal between attacks); PCT (UROD): ~617 nm (uroporphyrin); EPP/XLP (FECH/ALAS2): ~634 nm (protoporphyrin). The VP 626-628 nm peak is the single most specific test for VP — it persists between attacks and distinguishes VP from AIP and HCP even when urine PBG has normalised."
            },
            {
                "term": "Erythrodontia — Pathognomonic Sign of CEP",
                "definition": "Reddish-brown discolouration of teeth with intense orange-red fluorescence under ultraviolet (UV/Wood's lamp) light. Mechanism: uroporphyrin I (from UROS deficiency) deposits in dentine and enamel during tooth development. PATHOGNOMONIC for Congenital Erythropoietic Porphyria (CEP/UROS deficiency) — no other condition causes this specific combination of dental discolouration + UV fluorescence. Primary (milk) teeth may appear pink-red in affected neonates — an early diagnostic clue. Permanent teeth also affected (deposited during odontogenesis). Dental manifestation is permanent (does not regress with treatment)."
            },
            {
                "term": "Afamelanotide (Scenesse, Clinuvel Pharmaceuticals)",
                "definition": "Alpha-melanocyte-stimulating hormone (α-MSH) analogue; biodegradable subcutaneous implant (16 mg) placed every 2 months. FDA approved December 2019 for EPP (erythropoietic protoporphyria / FECH deficiency) — first pharmacological treatment for EPP. Mechanism: binds melanocortin-1 receptor (MC1R) → induces melanin synthesis → increased melanin in skin → photoprotection (absorbs UV + visible light that activates PP IX). Reduces annualised phototoxic episodes by ~50% and significantly increases direct sunlight exposure time tolerated. Also used off-label for XLP (ALAS2 GOF) with similar rationale. Not effective for PCT or acute porphyrias."
            },
            {
                "term": "Phlebotomy for PCT — Mechanism and Targets",
                "definition": "First-line treatment for porphyria cutanea tarda (UROD deficiency/inhibition). Mechanism: removes excess iron → removes the co-factor required for the uroporphomethene inhibitory species that blocks UROD → UROD activity recovers → uroporphyrin production falls. Protocol: 450 mL whole blood removed every 2 weeks; target serum ferritin 15-20 μg/L (relative iron depletion — not full iron deficiency). Haemoglobin maintained >110-120 g/L. Remission typically achieved in 4-8 months. Monitoring: ferritin, Hb, and urine uroporphyrins monthly. Re-treatment needed if triggers return (HCV, alcohol, oestrogen). HCV DAA therapy achieves PCT remission without phlebotomy in many HCV-positive patients."
            },
            {
                "term": "SIADH in Acute Porphyria — Seizure Mechanism",
                "definition": "Syndrome of inappropriate antidiuretic hormone secretion (SIADH) is a common complication of acute porphyria attacks, occurring in up to 60% of severe AIP episodes. Mechanism: ALA toxicity to hypothalamic neurones → inappropriate ADH release → free water retention → dilutional hyponatraemia. Hyponatraemia <130 mmol/L → cerebral oedema → SEIZURES. CRITICAL PRESCRIBING HAZARD: the clinician attempts to treat porphyric seizures with a standard AED (PHT, PB) which are all porphyrinogenic → worsens the underlying crisis → more SIADH → more severe seizures. TREATMENT: correct hyponatraemia slowly (≤8-10 mmol/L/24h to avoid osmotic demyelination); isotonic saline ± fluid restriction; hematin to treat the underlying attack. LEV IV for seizures (safe in porphyria)."
            },
            {
                "term": "Heme Biosynthesis Pathway — Step Order",
                "definition": "8-step pathway from glycine/succinyl-CoA to heme: Step 1 (ALAS1/2): ALA synthesis (mitochondrial, rate-limiting). Step 2 (ALAD): ALA → PBG (cytoplasmic; blocked in ADP). Step 3 (HMBS): 4×PBG → hydroxymethylbilane (cytoplasmic; blocked in AIP). Step 4 (UROS): HMB → uroporphyrinogen III (cytoplasmic; blocked in CEP). Step 5 (UROD): uroporphyrinogen III → coproporphyrinogen III (cytoplasmic; blocked in PCT). Step 6 (CPOX): coproporphyrinogen III → protoporphyrinogen IX (mitochondrial IMS; blocked in HCP). Step 7 (PPOX): protoporphyrinogen IX → protoporphyrin IX (mitochondrial inner membrane; blocked in VP). Step 8 (FECH): PP IX + Fe²⁺ → heme (mitochondrial inner membrane; blocked in EPP; overdriven in XLP by ALAS2 GOF)."
            },
            {
                "term": "South African Variegate Porphyria Founder Mutation",
                "definition": "PPOX p.Arg59Trp (c.175C>T): present in ALL South African VP patients, traceable to a single couple — Gerrit Jansz van Deventer (Dutch) and Ariaantje Jacobs — who emigrated to Cape Town in 1688. Their descendant Willem Adriaan van der Stel (Governor of Cape Colony) fathered many children, spreading the mutation. Current estimated carrier frequency: ~1:300 South Africans (~30,000 carriers in South Africa; total VP global prevalence ~1:75,000). This founder effect makes VP one of the most common single-gene disorders in South Africa. The same mutation is rare outside South Africa (European VP is caused by diverse private mutations)."
            },
            {
                "term": "Zinc-PP IX (ZnPP) vs Free PP IX — EPP/XLP Distinction",
                "definition": "In erythropoietic porphyrias, the chelation state of elevated erythrocyte protoporphyrin IX (PP IX) distinguishes EPP from XLP: EPP (FECH deficiency): FECH cannot insert Fe²⁺ into PP IX → excess FREE PP IX. Zinc (Zn²⁺) can partially insert into PP IX by non-enzymatic chelation → mild ZnPP elevation, but FREE PP IX predominates. ZnPP:free PP IX ratio: LOW in EPP. XLP (ALAS2 GOF): excess ALA → excess PP IX production. PP IX overwhelms both FECH (→ free PP IX) and zinc chelation (→ ZnPP). BOTH free PP IX AND ZnPP are elevated. ZnPP:free PP IX ratio: HIGHER than EPP. Lead poisoning: ZnPP elevated, free PP IX normal (ALAD inhibition → ALA accumulation → ZnPP via non-enzymatic chelation; NOT via PP IX overproduction)."
            },
        ]
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print("=== PORPHYRIA ATLAS OVERVIEW ===")
    print(f"Genes: {ov['genes']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Avg onset: {ov['avg_onset_y']} y")
    print(f"Avg dx delay: {ov['avg_dx_delay_y']} y")
    print(f"Hepatic disease: {ov['hepatic_disease_n']} ({ov['hepatic_disease_pct']}%)")
    print(f"Unsafe AED events: {ov['unsafe_aed_n']} ({ov['unsafe_aed_pct']}%)")
    print(f"Porphyria groups: {ov['porphyria_groups']}")
    bd = get_breakdown()
    print(f"\n=== BREAKDOWN ({bd['total_genes']} genes) ===")
    for g in bd["breakdown"]:
        print(f"  {g['gene']}: {g['n_patients']} pts, onset {g['avg_onset_y']}y, delay {g['avg_dx_delay_y']}y, hepatic {g['hepatic_disease_pct']}%")
    defs = get_definitions()
    print(f"\n=== DEFINITIONS: {len(defs['definitions'])} terms ===")
