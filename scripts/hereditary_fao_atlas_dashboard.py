#!/usr/bin/env python3
"""Hereditary-FAO-Atlas — Complete 8-Gene Hereditary Fatty Acid Oxidation Disorders Atlas
ACADM   (medium-chain acyl-CoA dehydrogenase; 421 aa; 1p31.1; AR;
         MCAD deficiency — MOST COMMON FAO disorder in NBS populations;
         c.985A>G (p.K329E) ~90% of Northern European alleles;
         NBS-detectable (C8-acylcarnitine elevated);
         hypoketotic hypoglycaemia triggered by fasting/intercurrent illness;
         FASTING-ABSOLUTELY-CI; seed SEED_BASE+0) .
ACADVL  (very-long-chain acyl-CoA dehydrogenase; 655 aa; 17p13.1; AR;
         VLCAD deficiency;
         severe neonatal cardiac form (HCM, arrhythmia, early lethality);
         hepatic/hypoglycaemia form (childhood);
         adult myopathic form (exercise-induced rhabdomyolysis);
         FASTING-CI, EXERCISE-PACING mandatory;
         seed SEED_BASE+1) .
HADHA   (trifunctional protein alpha; 763 aa; 2p23.3; AR;
         LCHAD / TFP deficiency;
         peripheral neuropathy + pigmentary retinopathy PATHOGNOMONIC combination;
         rhabdomyolysis triggered by fasting/illness;
         maternal acute fatty liver of pregnancy (AFLP) or HELLP in LCHAD-carrier mothers;
         seed SEED_BASE+2) .
CPT1A   (carnitine palmitoyltransferase 1A; 773 aa; 11q13.3; AR;
         CPT1A deficiency;
         hypoketotic hypoglycaemia + hepatomegaly;
         Arctic Inuit/First Nations founder c.1436C>T (p.P479L) — very common, relatively benign;
         MEDIUM-CHAIN-TRIGLYCERIDES-CI in contrast to other FAO defects;
         seed SEED_BASE+3) .
CPT2    (carnitine palmitoyltransferase 2; 658 aa; 1p32.3; AR;
         CPT2 muscle form — most common FAO disorder in adults;
         MYOGLOBINURIA triggered by prolonged exercise/cold/fasting PATHOGNOMONIC;
         rhabdomyolysis → acute kidney injury;
         statins + NSAIDs CI (worsen rhabdomyolysis);
         seed SEED_BASE+4) .
SLC25A20 (carnitine-acylcarnitine translocase; 301 aa; 3p21.31; AR;
         CACT deficiency;
         neonatal/severe onset; cardiac arrhythmia; hyperammonaemia;
         very low free carnitine; rarely survives to adulthood without transplant;
         seed SEED_BASE+5) .
ETFA    (electron transfer flavoprotein alpha; 333 aa; 15q24.2; AR;
         MADD / GA2 (Multiple Acyl-CoA Dehydrogenation Deficiency / Glutaric Aciduria Type 2);
         three clinical forms: severe neonatal (dysmorphic features + congenital anomalies),
         severe neonatal (without anomalies), and late-onset;
         RIBOFLAVIN-RESPONSIVE 25-30% — trial mandatory;
         sweaty feet/cabbage odour;
         seed SEED_BASE+6) .
HMGCL   (3-hydroxymethylglutaryl-CoA lyase; 325 aa; 1p36.11; AR;
         HMG-CoA Lyase deficiency;
         HypoglycAemia WITHOUT ketosis PATHOGNOMONIC — cannot generate ketones;
         metabolic acidosis (organic acidemia);
         Saudi Arabian/Portuguese founder;
         leucine-rich foods CI (leucine → HMG-CoA pathway);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1830–1837)
"""

import random

SEED_BASE = 1830

FAO_GENES = [
    # -- ACADM -- MCAD deficiency -----------------------------------------------
    {
        "gene": "ACADM",
        "protein": (
            "ACADM -- 1p31.1 AR -- Medium-Chain-Acyl-CoA-Dehydrogenase-421aa -- "
            "MCAD-Deficiency-Most-Common-FAO-NBS-Populations -- "
            "c985AG-p.K329E-90pct-Northern-European-Alleles -- "
            "C8-Octanoylcarnitine-Elevated-NBS-PATHOGNOMONIC -- "
            "Hypoketotic-Hypoglycaemia-Fasting-Triggered -- "
            "FASTING-ABSOLUTELY-CI-AVOID-Prolonged-Fast"
        ),
        "alias": (
            "ACADM (medium-chain acyl-CoA dehydrogenase); OMIM gene 607008; "
            "MCAD deficiency OMIM 201450. "
            "1p31.1; 421 aa; ~45 kDa; mitochondrial matrix; FAD-dependent; autosomal recessive. "
            "FUNCTION: ACADM catalyses the first step of mitochondrial beta-oxidation for "
            "medium-chain fatty acids (C6-C12): dehydrogenation of the CoA ester → "
            "trans-2-enoyl-CoA + FADH2. The electrons are transferred to electron transfer "
            "flavoprotein (ETF, encoded by ETFA/ETFB) and then to the respiratory chain. "
            "Without ACADM: medium-chain fatty acids accumulate as acylcarnitines "
            "(C6, C8 octanoylcarnitine, C10, C10:1 decenoylcarnitine) and as "
            "free fatty acids / dicarboxylic acids (suberylglycine, hexanoylglycine "
            "in urine = PATHOGNOMONIC organic acid pattern). "
            "PATHOPHYSIOLOGY: "
            "Beta-oxidation is the key metabolic pathway during fasting: "
            "when glucose runs out, fatty acids are mobilised from adipose tissue → "
            "in MCAD: medium-chain fatty acids cannot be oxidised → "
            "impaired acetyl-CoA/NADH generation from medium-chain FA → "
            "impaired hepatic gluconeogenesis (substrate deficiency) → "
            "HYPOKETOTIC hypoglycaemia (low/absent ketones = diagnostic, because "
            "ketogenesis itself requires beta-oxidation of longer chains which is intact, "
            "but MCAD block prevents medium-chain-derived acetyl-CoA from reaching ketogenesis "
            "and diverts substrate away from ketone production → REDUCED ketones despite hypoglycaemia). "
            "Brain depends on glucose and ketones as fuel: "
            "in MCAD crisis: hypoglycaemia + inadequate ketone fuel → cerebral energy failure → "
            "encephalopathy, seizures, coma, hepatomegaly, elevated transaminases; "
            "without treatment: rapid deterioration to death (historically the most common "
            "cause of sudden unexpected death in infants before NBS era). "
            "EPIDEMIOLOGY: "
            "Most common fatty acid oxidation disorder in NBS programmes (Western European/Australian/US): "
            "incidence ~1:10,000-15,000 live births; "
            "carrier frequency ~1:40-1:80 in Northern Europeans. "
            "COMMON VARIANT: c.985A>G (p.Lys329Glu / p.K329E): "
            "accounts for ~90% of Northern European MCAD alleles; "
            "Classic MCAD: homozygous c.985A>G (classic NBS presentation); "
            "compound heterozygous c.985A>G + other variant: same clinical risk; "
            "rare genotypes may have milder biochemical phenotype. "
            "CLINICAL PRESENTATION: "
            "Pre-NBS: acute metabolic crisis at age 3-24 months — "
            "intercurrent illness (viral URTI most common trigger) → fasting → crisis; "
            "hypoketotic hypoglycaemia → vomiting, lethargy, hepatomegaly, seizures, coma, death; "
            "previously: mortality ~25% per episode; "
            "survivors: ~25% had significant neurological sequelae; "
            "post-NBS: most patients diagnosed asymptomatically; crisis avoidable with diet counselling; "
            "presentation as an adult is rare (usually mild genotype) but occurs. "
            "NBS: "
            "C8-acylcarnitine (octanoylcarnitine) markedly elevated on MS/MS; "
            "C8/C10 and C8/C2 ratios elevated; C10:1 decenoylcarnitine elevated; "
            "reflex testing: urine organic acids (hexanoylglycine, suberylglycine, "
            "phenylpropionylglycine after medium-chain fat load); plasma acylcarnitines; gene sequencing. "
            "DIAGNOSIS: "
            "NBS C8 elevation → confirm with plasma acylcarnitines + urine organic acids + ACADM sequencing; "
            "in crisis: glucose very low, ketones absent/low, "
            "liver enzymes elevated, hyperammonaemia (mild-moderate); "
            "plasma carnitine: low total + free (secondary carnitine depletion). "
            "TREATMENT: "
            "Preventive (mainstay): NEVER FAST — ensure glucose supply continuously; "
            "sick-day protocol: any illness → glucose drinks/polymer (Maxijul/Polycose) every 3-4h; "
            "if not tolerating orally → IV dextrose 10% at maintenance + bolus IMMEDIATELY; "
            "fasting guidelines: "
            "0-3 months: maximum 4h fast; 3-6 months: 6h; 6-12 months: 8h; >1 year: 10-12h; "
            "high-fat (ketogenic) diet CONTRAINDICATED (requires beta-oxidation to generate ketones); "
            "L-carnitine supplementation: "
            "controversial — benefits secondary carnitine depletion; "
            "most metabolic centres supplement to maintain plasma free carnitine >30 µmol/L; "
            "diet is otherwise unrestricted (do NOT restrict fat globally — fat is needed); "
            "MCT oil (medium-chain triglyceride): "
            "CONTRAINDICATED in MCAD (MCT directly loads the deficient pathway); "
            "Acute crisis: IV dextrose 10% 2 mL/kg bolus → maintenance 8-10 mg/kg/min; "
            "intralipid: avoid until stable (long-chain fat OK when well). "
            "KEY CLINICAL FACTS: "
            "MCAD crisis can be triggered by: viral illness, gastroenteritis, any cause of reduced intake; "
            "NBS has reduced MCAD mortality to near zero in screened populations; "
            "a normal result on NBS from a premature infant may be false-negative (immature); "
            "suberylglycine and hexanoylglycine in urine during crisis = most specific markers; "
            "adults with MCAD: can present with rhabdomyolysis/exercise intolerance but rare; "
            "prognosis with NBS + education: excellent — normal development."
        ),
        "age_of_onset": "Infancy 3-24 months (triggered by fasting/illness); NBS detects neonatal",
        "inheritance": "AR",
        "locus": "1p31.1",
        "protein_size": "421 aa",
        "key_biomarker": "C8-octanoylcarnitine elevated on NBS MS/MS; hexanoylglycine + suberylglycine in urine",
        "pathognomonic": "Hypoketotic hypoglycaemia + C8 acylcarnitine elevated + suberylglycine/hexanoylglycine urine",
        "treatment": "NEVER FAST; sick-day glucose protocol; IV dextrose in crisis; avoid MCT",
        "critical_flags": [
            "FASTING-ABSOLUTELY-CI — sick-day protocol = glucose polymer every 3-4h",
            "C8-OCTANOYLCARNITINE-NBS-PATHOGNOMONIC — screen all NBS abnormal C8",
            "MCT-OIL-CONTRAINDICATED — directly loads deficient pathway",
            "KETOGENIC-DIET-CONTRAINDICATED — cannot generate ketones via MCAD pathway",
            "IV-DEXTROSE-10PCT-IMMEDIATELY — any comatose infant with hypoglycaemia + low ketones",
            "SECONDARY-CARNITINE-DEPLETION — supplement to free carnitine >30 µmol/L",
            "NBS-NEAR-ELIMINATES-MORTALITY — pre-NBS 25% died per first episode",
            "SUBERYLGLYCINE-HEXANOYLGLYCINE-URINE-MOST-SPECIFIC — organic acid markers",
        ],
    },
    # -- ACADVL -- VLCAD deficiency ---------------------------------------------
    {
        "gene": "ACADVL",
        "protein": (
            "ACADVL -- 17p13.1 AR -- Very-Long-Chain-Acyl-CoA-Dehydrogenase-655aa -- "
            "VLCAD-Deficiency-Three-Phenotypes-Cardiac-Hepatic-Myopathic -- "
            "C14:1-Tetradecenoylcarnitine-Elevated-NBS-MOST-SPECIFIC -- "
            "Severe-Neonatal-Cardiac-Form-HCM-Arrhythmia-High-Lethality -- "
            "FASTING-CI-EXERCISE-PACING-Mandatory-MCT-Oil-Allowed"
        ),
        "alias": (
            "ACADVL (very-long-chain acyl-CoA dehydrogenase); OMIM gene 609575; "
            "VLCAD deficiency OMIM 201475. "
            "17p13.1; 655 aa; ~70 kDa; inner mitochondrial membrane; FAD-dependent; autosomal recessive. "
            "FUNCTION: ACADVL is the rate-limiting enzyme in mitochondrial beta-oxidation of "
            "very-long-chain fatty acids (C14-C20). Unlike ACADM (soluble matrix), "
            "ACADVL is located on the inner mitochondrial membrane. "
            "ACADVL dehydrogenates very-long-chain acyl-CoA esters to 2-enoyl-CoA + FADH2; "
            "electrons pass to ETF. "
            "In VLCAD deficiency: long-chain acylcarnitines accumulate "
            "(C14:1 tetradecenoylcarnitine = most specific NBS marker; "
            "also C14, C16, C18:1 elevated). "
            "Long-chain fatty acids (LCFA) cannot be properly oxidised → "
            "cardiac and skeletal muscle energy failure (heart and skeletal muscle are "
            "highly LCFA-dependent at rest and exercise). "
            "PATHOPHYSIOLOGY: "
            "Unlike MCAD: VLCAD uses C14-C20 (very long chains); "
            "medium-chain triglycerides (MCT, C8-C10) bypass VLCAD → "
            "MCT supplementation IS indicated in VLCAD (contrast MCAD where MCT is CI). "
            "Three clinical phenotypes: "
            "1. SEVERE NEONATAL CARDIAC FORM: "
            "hypertrophic cardiomyopathy (HCM) in utero or neonatal; "
            "life-threatening arrhythmias; pericardial effusion; heart failure; "
            "hypoglycaemia; lactic acidosis; hepatomegaly; "
            "WITHOUT NBS + treatment → high early lethality; "
            "with MCT + strict LCFA restriction + carnitine + treatment of arrhythmia → survival. "
            "2. HEPATIC/HYPOGLYCAEMIA FORM (childhood): "
            "hypoketotic hypoglycaemia triggered by fasting/illness; "
            "hepatomegaly with elevated transaminases; "
            "similar to MCAD but typically less severe acute presentation; "
            "cardiac involvement less prominent; "
            "rhabdomyolysis possible. "
            "3. ADULT MYOPATHIC FORM: "
            "exercise-induced rhabdomyolysis and myalgia; "
            "myoglobinuria with strenuous exercise; "
            "no/minimal hypoglycaemia; "
            "NBS may be positive even for milder genotypes. "
            "DIAGNOSIS: "
            "NBS: C14:1 tetradecenoylcarnitine most specific; also C14, C16, C18:1; "
            "plasma acylcarnitines confirm; ACADVL gene sequencing; "
            "in crisis: glucose low, ketones absent/low, troponin elevated (cardiac form), "
            "CK markedly elevated (myopathic/rhabdo form). "
            "TREATMENT: "
            "DIETARY: "
            "Long-chain fat restriction mandatory: "
            "reduce long-chain fat to <10% total energy (some centres <20%); "
            "MCT supplementation: MCT oil (C8-C10) bypasses ACADVL → is the therapeutic fat source; "
            "fat-modified formula (e.g. Monogen, Lipistart) for infants; "
            "uncooked cornstarch (complex carbohydrate) before bed/exercise in older children/adults; "
            "FASTING: avoid prolonged fast per MCAD protocol but VLCAD typically allows slightly longer; "
            "EXERCISE: aerobic exercise allowed + encouraged at moderate intensity; "
            "MANDATORY carbohydrate loading before strenuous exercise; "
            "PHARMACOLOGICAL: "
            "L-carnitine: supplement to prevent secondary depletion; "
            "triheptanoin (C7 triglyceride — Dojolvi™, FDA 2020): "
            "anaplerotic medium-odd-chain fat; provides propionyl-CoA + acetyl-CoA "
            "to replenish TCA cycle intermediates depleted in FAO defects; "
            "reduces hypoglycaemia episodes and hospitalisation; "
            "ACUTE CRISIS: IV dextrose 10% + stop LCFA intake; "
            "cardiac care (antiarrhythmic, diuretics for HCM); "
            "avoid propofol (PRIS risk — propofol inhibits mitochondrial FAO); "
            "ANAESTHESIA: pre-operative dextrose; avoid fasting >4-6h. "
            "KEY CLINICAL FACTS: "
            "MCT IS ALLOWED (contrast MCAD where MCT is CI — critical distinction); "
            "C14:1 is the best NBS marker — C14 can be spuriously low or normal in mild cases; "
            "neonatal HCM in VLCAD: cardiomyopathy is REVERSIBLE with treatment (metabolic HCM); "
            "ECG: pre-excitation (WPW-like) pattern occasionally; "
            "triheptanoin approved for long-chain FAO disorders (includes VLCAD, LCHAD, CPT2, CACT)."
        ),
        "age_of_onset": "Neonatal (cardiac form) / childhood (hepatic) / adult (myopathic)",
        "inheritance": "AR",
        "locus": "17p13.1",
        "protein_size": "655 aa",
        "key_biomarker": "C14:1 tetradecenoylcarnitine elevated on NBS (most specific)",
        "pathognomonic": "Neonatal HCM + C14:1 elevated + LCFA accumulation",
        "treatment": "Long-chain fat restriction; MCT supplementation; triheptanoin (Dojolvi); fasting CI",
        "critical_flags": [
            "MCT-OIL-ALLOWED-THERAPEUTIC — contrast MCAD where MCT is CI",
            "C14:1-NBS-MOST-SPECIFIC — do not miss on acylcarnitine panel",
            "NEONATAL-HCM-REVERSIBLE — metabolic cardiomyopathy improves with treatment",
            "TRIHEPTANOIN-FDA2020 — anaplerotic C7 for long-chain FAO defects",
            "FASTING-CI — carbohydrate load before exercise; dextrose in illness",
            "LCFA-RESTRICTION-MANDATORY — <10% total energy from long-chain fat",
            "PROPOFOL-AVOID — PRIS risk (propofol inhibits mitochondrial FAO)",
            "EXERCISE-PACING-MANDATORY — rhabdomyolysis risk; carbohydrate pre-exercise",
        ],
    },
    # -- HADHA -- LCHAD / TFP deficiency ----------------------------------------
    {
        "gene": "HADHA",
        "protein": (
            "HADHA -- 2p23.3 AR -- Trifunctional-Protein-Alpha-763aa -- "
            "LCHAD-TFP-Deficiency-Peripheral-Neuropathy+Pigmentary-Retinopathy-PATHOGNOMONIC -- "
            "Maternal-AFLP-HELLP-Carrier-Mothers-PATHOGNOMONIC -- "
            "Rhabdomyolysis-LCFA-Triggered -- "
            "C16-OH-3-Hydroxy-Palmitoylcarnitine-NBS"
        ),
        "alias": (
            "HADHA (hydroxyacyl-CoA dehydrogenase/3-ketoacyl-CoA thiolase/enoyl-CoA hydratase, "
            "alpha subunit / mitochondrial trifunctional protein alpha); OMIM gene 600890; "
            "LCHAD deficiency OMIM 609016; TFP deficiency OMIM 609015. "
            "2p23.3; 763 aa; ~79 kDa; inner mitochondrial membrane; autosomal recessive. "
            "FUNCTION: The mitochondrial trifunctional protein (MTP) is a heterooctamer of "
            "four alpha subunits (HADHA) and four beta subunits (HADHB) embedded in the "
            "inner mitochondrial membrane. "
            "HADHA encodes THREE enzymatic activities: "
            "1. Long-chain enoyl-CoA hydratase (LCEH): 2-enoyl-CoA → 3-L-hydroxyacyl-CoA; "
            "2. Long-chain 3-hydroxyacyl-CoA dehydrogenase (LCHAD): "
            "3-L-hydroxyacyl-CoA → 3-ketoacyl-CoA + NADH; "
            "3. Long-chain 3-ketoacyl-CoA thiolase (LCKT, on HADHB): thiolytic cleavage. "
            "LCHAD is the most critical enzymatic step — "
            "HADHA p.G1528C (c.1528G>C, p.Glu510Gln) is the common LCHAD-specific mutation "
            "(in the LCHAD active site, OMIM p.E474Q numbering); "
            "this LCHAD-isolated deficiency is distinguishable from complete TFP deficiency "
            "(mutations affecting entire protein). "
            "In HADHA deficiency: 3-hydroxy-long-chain acylcarnitines accumulate "
            "(C16-OH, C18:1-OH, C18-OH — key NBS markers). "
            "PATHOPHYSIOLOGY AND UNIQUE FEATURES: "
            "1. PROGRESSIVE PERIPHERAL NEUROPATHY: "
            "accumulation of 3-hydroxy-long-chain fatty acids causes Schwann cell toxicity; "
            "peripheral axonal neuropathy — absent tendon reflexes, reduced nerve conduction; "
            "progressive over years despite dietary treatment; "
            "combination of neuropathy + retinopathy in a child with FAO disorder = PATHOGNOMONIC for LCHAD/TFP. "
            "2. PIGMENTARY RETINOPATHY: "
            "photoreceptor and RPE degeneration; onset childhood-adolescence; "
            "pigmentary changes on fundoscopy; ERG abnormal; "
            "can progress to blindness if untreated or in severe cases; "
            "MANDATORY ophthalmology review annually in all HADHA patients. "
            "3. MATERNAL COMPLICATIONS in LCHAD CARRIER MOTHERS: "
            "if a LCHAD-deficient fetus (compound heterozygous or homozygous LCHAD): "
            "3-hydroxy fatty acids from the fetus cross the placenta → "
            "maternal AFLP (acute fatty liver of pregnancy): hepatic failure, "
            "coagulopathy, encephalopathy in third trimester; "
            "or maternal HELLP syndrome (haemolysis, elevated liver enzymes, low platelets); "
            "PATHOGNOMONIC epidemiological link: "
            "mother presenting with AFLP/HELLP → test baby for LCHAD; "
            "baby with LCHAD → inform mother of increased future AFLP risk in subsequent pregnancies; "
            "HADHB mutations do not cause maternal AFLP (LCHAD-specific). "
            "4. EPISODIC RHABDOMYOLYSIS: "
            "triggered by prolonged exercise, fasting, or febrile illness; "
            "elevated CK, myoglobinuria; AKI risk. "
            "DIAGNOSIS: "
            "NBS: C16-OH 3-OH-palmitoylcarnitine, C18:1-OH, C18-OH elevated; "
            "distinguish isolated LCHAD vs TFP by genotype or enzyme assay; "
            "HADHA/HADHB gene sequencing; "
            "ophthalmology and neurophysiology baseline at diagnosis. "
            "TREATMENT: "
            "Long-chain fat restriction (similar to VLCAD): reduce LCFA to <10% total energy; "
            "MCT supplementation (bypasses HADHA — C8-C10 uses MCAD not HADHA); "
            "DHA supplementation: docosahexaenoic acid (C22:6n-3) — "
            "DHA cannot be synthesised normally in LCHAD deficiency (requires long-chain FAO); "
            "DHA deficiency may contribute to retinopathy — supplement long-term; "
            "L-carnitine supplementation; "
            "triheptanoin (Dojolvi) — approved for long-chain FAO defects including TFP/LCHAD; "
            "ophthalmology: annual fundoscopy + ERG; retinal laser if neovascularisation; "
            "neurology: annual neurophysiology (NCS/EMG); "
            "FASTING CI; sick-day protocol same as VLCAD; "
            "KEY CLINICAL FACTS: "
            "Retinopathy + neuropathy = PATHOGNOMONIC for LCHAD/TFP vs other FAO defects; "
            "maternal AFLP/HELLP in LCHAD carrier mother is a diagnostic red flag; "
            "DHA supplementation may slow retinal progression (evidence from small trials); "
            "prognosis: neonatal cardiac form severe; late-onset myopathic form more favourable."
        ),
        "age_of_onset": "Neonatal (cardiac/hepatic) or childhood (neuropathy/retinopathy)",
        "inheritance": "AR",
        "locus": "2p23.3",
        "protein_size": "763 aa",
        "key_biomarker": "C16-OH 3-OH-palmitoylcarnitine elevated on NBS; C18:1-OH, C18-OH",
        "pathognomonic": "Peripheral neuropathy + pigmentary retinopathy in child with FAO disorder",
        "treatment": "LCFA restriction; MCT + DHA supplementation; triheptanoin; annual ophthalmology/neurology",
        "critical_flags": [
            "NEUROPATHY+RETINOPATHY-PATHOGNOMONIC — unique to LCHAD/TFP among FAO defects",
            "MATERNAL-AFLP-HELLP-CARRIER — mother with AFLP → test baby for LCHAD",
            "DHA-SUPPLEMENTATION-MANDATORY — prevents/slows retinal degeneration",
            "ANNUAL-OPHTHALMOLOGY-MANDATORY — early retinal changes treatable",
            "MCT-THERAPEUTIC — bypasses HADHA (C8-C10 uses MCAD not TFP)",
            "C16-OH-NBS-SPECIFIC — distinguish from VLCAD (C14:1)",
            "TRIHEPTANOIN-FDA2020 — approved for long-chain FAO including LCHAD/TFP",
            "LCHAD-SPECIFIC-p.G1528C-COMMON — isolated LCHAD vs complete TFP different prognosis",
        ],
    },
    # -- CPT1A -- CPT1A deficiency -----------------------------------------------
    {
        "gene": "CPT1A",
        "protein": (
            "CPT1A -- 11q13.3 AR -- Carnitine-Palmitoyltransferase-1A-773aa -- "
            "CPT1A-Deficiency-Hypoketotic-Hypoglycaemia-Hepatomegaly -- "
            "Arctic-Inuit-First-Nations-Founder-c.1436C>T-p.P479L-Very-Common -- "
            "MCT-CI-Contrast-VLCAD-LCHAD -- "
            "C0-Free-Carnitine-VERY-HIGH-KEY-DDx"
        ),
        "alias": (
            "CPT1A (carnitine palmitoyltransferase 1A, liver isoform); OMIM gene 600528; "
            "CPT1A deficiency OMIM 255120. "
            "11q13.3; 773 aa; ~88 kDa; outer mitochondrial membrane; autosomal recessive. "
            "FUNCTION: CPT1A is the rate-limiting enzyme for the carnitine shuttle system "
            "that transports long-chain fatty acids into mitochondria for beta-oxidation. "
            "Mechanism: "
            "Cytoplasmic long-chain fatty acyl-CoA + L-carnitine → acylcarnitine (CPT1A) → "
            "crosses inner mitochondrial membrane via CACT (SLC25A20) → "
            "re-esterified to acyl-CoA by CPT2 inside mitochondria → "
            "undergoes beta-oxidation. "
            "CPT1A is the liver/kidney isoform (CPT1B = muscle; CPT1C = brain). "
            "There are THREE CPT1 isoforms: "
            "CPT1A (liver, kidney): most important for hepatic FA oxidation and ketogenesis; "
            "CPT1B (muscle, heart): skeletal and cardiac muscle FA oxidation; "
            "CPT1C (brain): neuronal energy sensing. "
            "In CPT1A deficiency: ONLY liver isoform affected → "
            "long-chain FA cannot enter hepatic mitochondria → "
            "impaired hepatic beta-oxidation and ketogenesis → "
            "hypoketotic hypoglycaemia + hepatomegaly (hepatic FA accumulation); "
            "cardiac and skeletal muscle FA oxidation preserved (CPT1B intact); "
            "heart is NOT primarily affected (DDx VLCAD/LCHAD). "
            "UNIQUE NBS MARKER: "
            "C0 (free carnitine) VERY HIGH — because acylcarnitines cannot form "
            "(CPT1A makes acylcarnitines from acyl-CoA; without CPT1A, "
            "acylcarnitines do not form from long-chain FA; free carnitine accumulates); "
            "C0/C16 + C18 ratio markedly elevated; "
            "long-chain acylcarnitines NORMAL or LOW (DDx other FAO: long-chain acylcarnitines high). "
            "ARCTIC INUIT / FIRST NATIONS FOUNDER VARIANT: "
            "c.1436C>T (p.Pro479Leu / p.P479L): "
            "VERY HIGH frequency in circumpolar Indigenous populations: "
            "Greenland Inuit, Canadian Inuit, First Nations (Cree, Ojibwe): "
            "carrier frequency up to 1:3 in some communities; "
            "homozygous frequency up to ~1:6 in some Inuit communities; "
            "CLINICAL SIGNIFICANCE: "
            "Homozygous p.P479L has approximately 40% residual CPT1A activity; "
            "in these communities: RELATIVELY BENIGN phenotype in most homozygotes "
            "— most are asymptomatic; "
            "hypothesis: advantageous in Inuit diet (high fat/protein diet — "
            "may reduce excessive ketogenesis from a high-fat diet); "
            "HOWEVER: some p.P479L homozygotes DO have clinically significant hypoglycaemia; "
            "NBS in these communities: very high CPT1A screen positives — need protocols. "
            "IMPORTANT DISTINCTION: "
            "MCT OIL CONTRAINDICATED in CPT1A deficiency: "
            "MCT (C8-C10) enters mitochondria via MCAD directly (bypassing CPT1A); "
            "BUT: loading with MCT increases medium-chain FA entering mitochondria; "
            "however, the HEPATIC load is the problem — MCT supplements increase acyl-CoA flux; "
            "practically: most metabolic centres AVOID MCT in CPT1A; "
            "this contrasts with VLCAD/LCHAD where MCT is THERAPEUTIC. "
            "DIAGNOSIS: "
            "NBS: very high C0 free carnitine; C0/(C16+C18) ratio key; "
            "normal or low long-chain acylcarnitines (DDx other FAO with HIGH long-chain acylcarnitines); "
            "plasma carnitine total elevated; organic acids urine: dicarboxylic acids in crisis; "
            "CPT1A gene sequencing; enzyme assay if uncertain. "
            "TREATMENT: "
            "Fasting avoidance + sick-day protocol (same as MCAD); "
            "high-carbohydrate, LOW-fat diet in severe cases; "
            "IV dextrose in crisis; "
            "MCT avoidance (CI in CPT1A, unlike VLCAD/LCHAD); "
            "L-carnitine: uncertain benefit (carnitine is already high); "
            "prognosis for standard variants: good with treatment; "
            "p.P479L Arctic variant: often asymptomatic but counselling essential. "
            "KEY CLINICAL FACTS: "
            "Free C0 carnitine ELEVATED (opposite to most FAO defects where carnitine is low); "
            "no cardiomyopathy (cardiac CPT1B intact); "
            "MCT CONTRAINDICATED — critical difference from VLCAD/LCHAD/CPT2; "
            "Arctic Inuit founder: most common FAO disorder variant in circumpolar populations."
        ),
        "age_of_onset": "Infancy (triggered by fasting/illness); often asymptomatic if p.P479L Inuit founder",
        "inheritance": "AR",
        "locus": "11q13.3",
        "protein_size": "773 aa",
        "key_biomarker": "C0 free carnitine VERY HIGH on NBS; C0/(C16+C18) ratio elevated; long-chain acylcarnitines normal/low",
        "pathognomonic": "Hypoketotic hypoglycaemia + hepatomegaly + C0 elevated + NO cardiac involvement",
        "treatment": "Fasting CI; high-carb diet; IV dextrose; MCT CI; avoid fasting",
        "critical_flags": [
            "MCT-CONTRAINDICATED — opposite to VLCAD/LCHAD where MCT is therapeutic",
            "C0-FREE-CARNITINE-VERY-HIGH — opposite pattern to most FAO (usually carnitine low)",
            "NO-CARDIAC-INVOLVEMENT — CPT1B (cardiac) intact; DDx VLCAD/LCHAD",
            "ARCTIC-INUIT-FOUNDER-p.P479L — high frequency; often relatively benign in homozygotes",
            "LONG-CHAIN-ACYLCARNITINES-NORMAL — DDx VLCAD (high C14:1), LCHAD (high C16-OH)",
            "FASTING-ABSOLUTELY-CI — sick-day protocol mandatory",
            "L-CARNITINE-NOT-NEEDED — carnitine already elevated",
            "HIGH-CARB-LOW-FAT-DIET — opposite to ketogenic diet",
        ],
    },
    # -- CPT2 -- CPT2 deficiency ------------------------------------------------
    {
        "gene": "CPT2",
        "protein": (
            "CPT2 -- 1p32.3 AR -- Carnitine-Palmitoyltransferase-2-658aa -- "
            "CPT2-Muscle-Form-Most-Common-FAO-Adults -- "
            "MYOGLOBINURIA-Exercise-Cold-Fasting-Triggered-PATHOGNOMONIC -- "
            "Rhabdomyolysis-AKI-Risk -- "
            "Statins-NSAIDs-CI-Worsen-Rhabdomyolysis"
        ),
        "alias": (
            "CPT2 (carnitine palmitoyltransferase 2); OMIM gene 600650; "
            "CPT2 deficiency OMIM 255110. "
            "1p32.3; 658 aa; ~74 kDa; inner mitochondrial membrane; autosomal recessive. "
            "FUNCTION: CPT2 is the second enzyme of the carnitine shuttle. "
            "Located on the matrix face of the inner mitochondrial membrane, "
            "CPT2 re-converts long-chain acylcarnitine → long-chain acyl-CoA + free carnitine "
            "(the reverse reaction of CPT1). "
            "Without CPT2: long-chain fatty acids cannot be re-esterified inside mitochondria → "
            "long-chain acylcarnitines accumulate inside the mitochondrial matrix AND "
            "in plasma (C16, C18, C18:1, C14:1 acylcarnitines elevated). "
            "CPT2 has THREE clinical phenotypes: "
            "1. LETHAL NEONATAL FORM: "
            "virtually no CPT2 activity; "
            "hypoglycaemia, cardiomyopathy, hepatomegaly, renal dysgenesis; "
            "death within days-weeks; rare. "
            "2. SEVERE INFANTILE HEPATO-CARDIOMUSCULAR FORM: "
            "onset 3-24 months; "
            "triggered by febrile illness; "
            "hypoketotic hypoglycaemia + HCM + hepatomegaly + rhabdomyolysis; "
            "rare; severe. "
            "3. MUSCLE FORM (CPT2-MYOPATHY): "
            "MOST COMMON form of CPT2 deficiency; "
            "MOST COMMON inherited disorder of mitochondrial FAO presenting in adults; "
            "onset typically adolescence-early adulthood (range: childhood to 60s); "
            "partial residual CPT2 activity (~30%); "
            "TRIGGERS: prolonged aerobic exercise + cold + fasting (individually or combined); "
            "PRESENTATION: "
            "exercise-induced myalgia and stiffness → rhabdomyolysis → MYOGLOBINURIA; "
            "urine turns dark brown-red (myoglobin = PATHOGNOMONIC); "
            "CK markedly elevated (>10,000 IU/L in episodes); "
            "acute kidney injury (myoglobin nephrotoxicity); "
            "between episodes: completely NORMAL neurological and muscular function; "
            "inter-episode CK: normal or mildly elevated. "
            "COMMON VARIANT (CPT2 muscle form): "
            "p.S113L (c.338C>T): accounts for ~60-70% of CPT2 muscle form alleles; "
            "also p.P50H, p.F352C; "
            "p.S113L: partial activity retained → muscle form only (sufficient for hepatic/cardiac). "
            "TRIGGERS AND MANAGEMENT: "
            "EXERCISE: transition from anaerobic → aerobic metabolism (around 10-20 min exercise) "
            "= the most common trigger (lipids become primary fuel); "
            "aerobic exercise pacing with carbohydrate loading; warm-up gradually; "
            "COLD: cold temperature ↑ LCFA demand in muscle (shivering thermogenesis); "
            "FASTING: depletes glycogen → switches to LCFA-dependent metabolism; "
            "ILLNESS: infection raises LCFA demand + reduces intake. "
            "ANAESTHESIA TRIGGER: "
            "general anaesthesia + muscle relaxants → rhabdomyolysis risk; "
            "anaesthetist must be informed; propofol AVOID (PRIS); "
            "ensure glucose infusion peri-operatively. "
            "MEDICATIONS CI: "
            "STATINS: inhibit mevalonate pathway → reduced CoQ10 + secondary CPT2 inhibition "
            "→ markedly worsen rhabdomyolysis in CPT2; CI; "
            "NSAIDs (ibuprofen, diclofenac): directly inhibit CPT2 enzyme activity → CI; "
            "valproate: inhibits FAO; CI; "
            "propofol: PRIS risk; CI. "
            "TREATMENT: "
            "PREVENTIVE: "
            "Carbohydrate loading before exercise (cornstarch or glucose polymer 30g before strenuous); "
            "avoid prolonged fasting; "
            "maintain warmth in cold environments; "
            "low-fat, high-carbohydrate diet; "
            "MCT supplementation (medium-chain bypasses CPT2 — enters via MCAD): "
            "THERAPEUTIC — use before exercise; "
            "L-carnitine supplementation (secondary depletion); "
            "triheptanoin (Dojolvi): not formally approved for CPT2 but used; "
            "ACUTE RHABDOMYOLYSIS: "
            "IV normal saline or Hartmann 3-5 mL/kg/h → urine output >1-2 mL/kg/h; "
            "alkalinise urine (sodium bicarbonate) to reduce myoglobin nephrotoxicity; "
            "IV glucose 10%; "
            "discontinue all statins/NSAIDs; "
            "monitor urine myoglobin and creatinine closely; "
            "dialysis if AKI oliguria despite fluids. "
            "KEY CLINICAL FACTS: "
            "Most common adult-onset FAO disorder; "
            "inter-episode CK: normal (useful DDx from other myopathies); "
            "NBS: can pick up elevated C16/C18 acylcarnitines (variable); "
            "prognosis: excellent with lifestyle modification; "
            "patient education: carry carbohydrates during exercise; "
            "family: siblings need CPT2 testing (same risk)."
        ),
        "age_of_onset": "Adolescence-early adulthood (muscle form); neonatal (rare severe forms)",
        "inheritance": "AR",
        "locus": "1p32.3",
        "protein_size": "658 aa",
        "key_biomarker": "C16, C18:1, C14:1 acylcarnitines in crisis; CK markedly elevated; urine myoglobin",
        "pathognomonic": "Exercise/cold-triggered myoglobinuria + rhabdomyolysis with NORMAL inter-episode CK",
        "treatment": "Carb loading before exercise; MCT therapeutic; statins/NSAIDs/valproate CI; hydration in crisis",
        "critical_flags": [
            "MYOGLOBINURIA-PATHOGNOMONIC — dark urine after exercise/cold = CPT2 until proven otherwise",
            "STATINS-CI — markedly worsen rhabdomyolysis in CPT2",
            "NSAIDs-CI — directly inhibit CPT2 enzyme",
            "VALPROATE-CI — inhibits FAO",
            "MCT-THERAPEUTIC — medium-chain bypasses CPT2",
            "CARBOHYDRATE-LOAD-BEFORE-EXERCISE — primary prevention of rhabdomyolysis",
            "IV-SALINE-HYDRATION — urine output >2 mL/kg/h in acute rhabdomyolysis",
            "PROPOFOL-AVOID — PRIS risk; inform anaesthesia",
        ],
    },
    # -- SLC25A20 -- CACT deficiency --------------------------------------------
    {
        "gene": "SLC25A20",
        "protein": (
            "SLC25A20 -- 3p21.31 AR -- Carnitine-Acylcarnitine-Translocase-CACT-301aa -- "
            "CACT-Deficiency-Neonatal-Severe-Cardiac-Arrhythmia-Hyperammonaemia -- "
            "Free-Carnitine-VERY-LOW-All-Long-Chain-Acylcarnitines-VERY-HIGH -- "
            "Rarely-Survives-Without-Early-Transplant-Consideration"
        ),
        "alias": (
            "SLC25A20 (solute carrier family 25 member 20; carnitine/acylcarnitine translocase; "
            "CACT); OMIM gene 613698; CACT deficiency OMIM 212138. "
            "3p21.31; 301 aa; ~33 kDa; inner mitochondrial membrane carrier protein; autosomal recessive. "
            "FUNCTION: SLC25A20 encodes CACT, the inner mitochondrial membrane translocase "
            "that exchanges long-chain acylcarnitines (from CPT1 reaction) for free carnitine "
            "(from CPT2 reaction). "
            "CACT is the central shuttle: "
            "Cytoplasm: long-chain acyl-CoA + carnitine → acylcarnitine (CPT1A) "
            "→ CACT transports acylcarnitine INTO mitochondria (antiport with free carnitine); "
            "Mitochondria: acylcarnitine → acyl-CoA + free carnitine (CPT2) "
            "→ CACT transports free carnitine OUT to cytoplasm. "
            "In CACT deficiency: the carnitine cycle is completely blocked — "
            "long-chain acylcarnitines cannot enter mitochondria → "
            "ALL long-chain FA oxidation abolished → "
            "severe energy failure in all LCFA-dependent tissues: heart, liver, skeletal muscle. "
            "BIOCHEMICAL SIGNATURE: "
            "Plasma free carnitine (C0): VERY LOW — carnitine trapped in cytoplasm as acylcarnitines; "
            "Long-chain acylcarnitines (C16, C18, C18:1, C14:1): VERY HIGH "
            "(unlike CPT1A where these are normal); "
            "Severe hyperammonaemia: liver cannot detoxify ammonia without FA energy; "
            "hypoglycaemia (hypoketotic); lactic acidosis. "
            "CLINICAL PRESENTATION: "
            "NEONATAL ONSET in most cases: "
            "hours to days after birth; "
            "cardiac arrhythmia (ventricular tachycardia, fibrillation) — severe risk of sudden death; "
            "hyperammonaemia (often >500 µmol/L — greater than most urea cycle defects); "
            "hypoketotic hypoglycaemia; "
            "hepatomegaly with liver failure; "
            "profound hypotonia; "
            "neonatal death without aggressive treatment. "
            "MILDER FORMS: "
            "Late neonatal/infantile onset with episodic decompensation; "
            "some patients survive to childhood or adulthood with treatment. "
            "DIAGNOSIS: "
            "NBS: severely elevated C16, C18, C18:1 + very low C0 (distinguishes from CPT2, VLCAD); "
            "plasma acylcarnitine profile: "
            "very high long-chain acylcarnitines + very low free C0 = CACT or CPT2 pattern; "
            "CPT2 usually milder (muscle form) vs CACT (severe neonatal); "
            "SLC25A20 vs CPT2 gene sequencing is definitive; "
            "ammonia: very high (unusual for FAO defects — degree of hyperammonaemia suggests CACT). "
            "TREATMENT: "
            "ACUTE: "
            "IV glucose (10-15%) at high rate + L-carnitine IV/oral; "
            "stop all oral feeds containing long-chain fat; "
            "CACT-safe formula (MCT-based, minimal LCFA); "
            "antiarrhythmic agents for ventricular arrhythmia (amiodarone); "
            "ammonia scavengers (sodium benzoate/phenylbutyrate) for hyperammonaemia; "
            "dialysis for severe hyperammonaemia/metabolic acidosis; "
            "CHRONIC: "
            "MCT-based formula (C8-C10 bypasses CACT); "
            "strict long-chain fat restriction; "
            "frequent feeds (avoid fasting completely); "
            "L-carnitine supplementation (IV then oral, high dose); "
            "triheptanoin (C7) in some centres; "
            "cardiac monitoring: ECG, ECHO (arrhythmia risk persists); "
            "TRANSPLANT: "
            "liver transplantation has been performed in some CACT-deficient patients "
            "with improvement of metabolic crises; cardiac benefit uncertain. "
            "PROGNOSIS: "
            "Without aggressive treatment: very high neonatal mortality; "
            "even with treatment: significant morbidity from cardiac and neurological complications; "
            "rare patients with milder genotypes survive to adulthood. "
            "KEY CLINICAL FACTS: "
            "CACT deficiency = the most severe carnitine cycle defect (complete block of shuttle); "
            "very low C0 + very high long-chain acylcarnitines = CACT or severe CPT2; "
            "hyperammonaemia greater than typical FAO defects = clue to CACT; "
            "cardiac arrhythmia (ventricular) is the leading cause of acute death; "
            "MCT formula IS the safe nutrient (bypasses the CACT transporter)."
        ),
        "age_of_onset": "Neonatal (hours-days after birth)",
        "inheritance": "AR",
        "locus": "3p21.31",
        "protein_size": "301 aa",
        "key_biomarker": "C0 free carnitine VERY LOW + C16/C18/C18:1 VERY HIGH on NBS; severe hyperammonaemia",
        "pathognomonic": "Neonatal cardiac arrhythmia + hyperammonaemia + very low C0 + very high long-chain acylcarnitines",
        "treatment": "IV glucose; IV L-carnitine; MCT formula (LCFA strictly restricted); antiarrhythmics; ammonia scavengers",
        "critical_flags": [
            "CARDIAC-ARRHYTHMIA-LEADING-KILLER — ventricular tachycardia/fibrillation neonatal",
            "C0-VERY-LOW-LONG-CHAIN-VERY-HIGH — complete carnitine shuttle block",
            "HYPERAMMONAEMIA-SEVERE — >500 µmol/L; greater than typical urea cycle defects",
            "MCT-FORMULA-BYPASSES-CACT — C8-C10 enters via MCAD; only safe energy fat",
            "LCFA-ABSOLUTELY-CI — any long-chain fat → carnitine shuttle blocked",
            "MOST-SEVERE-CARNITINE-CYCLE-DEFECT — complete block vs CPT1A/CPT2 (partial)",
            "NEONATAL-DEATH-WITHOUT-TREATMENT — aggressive IV glucose + carnitine IMMEDIATELY",
            "IV-CARNITINE-HIGH-DOSE — L-carnitine 100-200 mg/kg/day IV initially",
        ],
    },
    # -- ETFA -- MADD / GA2 deficiency ------------------------------------------
    {
        "gene": "ETFA",
        "protein": (
            "ETFA -- 15q24.2 AR -- Electron-Transfer-Flavoprotein-Alpha-333aa -- "
            "MADD-GA2-Glutaric-Aciduria-Type-2-Multiple-Acyl-CoA-Dehydrogenation-Deficiency -- "
            "Three-Forms-Neonatal-Dysmorphic-Neonatal-Non-Dysmorphic-Late-Onset -- "
            "RIBOFLAVIN-RESPONSIVE-25-30pct-Trial-MANDATORY -- "
            "Sweaty-Feet-Cabbage-Isovaleric-Odour"
        ),
        "alias": (
            "ETFA (electron transfer flavoprotein alpha subunit); OMIM gene 608053; "
            "MADD (Multiple Acyl-CoA Dehydrogenation Deficiency) OMIM 231680; "
            "also called GA2 (Glutaric Aciduria Type 2). "
            "15q24.2; 333 aa; ~35 kDa; mitochondrial matrix; autosomal recessive. "
            "Also caused by ETFB (OMIM 130410) and ETFDH (OMIM 231675) mutations. "
            "FUNCTION: The electron transfer flavoprotein (ETF) is a heterodimer of "
            "alpha (ETFA) and beta (ETFB) subunits in the mitochondrial matrix. "
            "ETF accepts electrons from MULTIPLE acyl-CoA dehydrogenases: "
            "SCAD (short chain), MCAD (medium chain), LCAD, VLCAD, LCHAD, "
            "IVD (isovaleryl-CoA — Leu catabolism), "
            "2-MCADeH, GluD (glutaryl-CoA), "
            "also proline oxidase and dimethylglycine dehydrogenase. "
            "ETF passes electrons to ETFDH (electron transfer flavoprotein dehydrogenase, "
            "on inner mitochondrial membrane) → coenzyme Q10 → CIII → ATP. "
            "In ETFA/ETFB/ETFDH deficiency: "
            "ALL the above dehydrogenases are secondarily impaired → "
            "MULTIPLE acyl-CoA dehydrogenation deficiency; "
            "multiple acylcarnitines accumulate simultaneously (short, medium, long chain + glutarylcarnitine). "
            "BIOCHEMICAL SIGNATURE: "
            "Urine organic acids: "
            "ethylmalonic acid + adipic acid + suberic acid + glutaric acid (=GA2, not GA1); "
            "isovalerylglycine; isobutyrylglycine; hexanoylglycine; "
            "lactic acid; "
            "sweaty feet odour: isovaleric + butyric/hexanoic acid metabolites. "
            "NBS acylcarnitines: "
            "C4 (butyrylcarnitine) elevated; C5 (isovalerylcarnitine) elevated; "
            "C6, C8, C10, C14:1 all elevated — MULTIPLE species elevated = diagnostic clue. "
            "THREE CLINICAL FORMS: "
            "1. NEONATAL FORM WITH CONGENITAL ANOMALIES: "
            "virtually absent ETF activity; "
            "dysmorphic features: ear abnormalities, facial dysmorphism, scrotal hypoplasia; "
            "cystic renal disease; "
            "brain abnormalities (migration defects, periventricular heterotopia); "
            "metabolic acidosis, hypoglycaemia, hypotonia immediately after birth; "
            "die within days-weeks; riboflavin unresponsive. "
            "2. NEONATAL FORM WITHOUT ANOMALIES: "
            "absent ETF activity without structural defects; "
            "metabolic crisis from birth; hypoglycaemia, acidosis, cardiomyopathy; "
            "often fatal in neonatal period; some survive short-term; riboflavin unresponsive. "
            "3. LATE-ONSET FORM (most common surviving phenotype): "
            "residual ETF activity; "
            "onset childhood to adulthood; "
            "episodic: vomiting, hypoglycaemia, metabolic acidosis, myopathy triggered by illness/fasting; "
            "muscle weakness, exercise intolerance; "
            "sweaty feet/isovaleric odour especially during crises; "
            "RIBOFLAVIN-RESPONSIVE: ~25-30% of late-onset MADD patients respond to "
            "riboflavin (vitamin B2) 150-300 mg/day; "
            "ETFDH mutations most commonly riboflavin-responsive (CoQ10 deficiency pathway); "
            "riboflavin trial mandatory in all MADD patients before declaring non-responsive. "
            "DIAGNOSIS: "
            "NBS: multiple acylcarnitine elevations (C4, C5, C6, C8, C10, C14:1 simultaneously); "
            "urine organic acids: multiple dicarboxylic + glutaric acid; "
            "plasma acylcarnitines confirm; "
            "ETFA/ETFB/ETFDH gene panel; "
            "riboflavin trial with metabolite monitoring to assess response. "
            "TREATMENT: "
            "Riboflavin (vitamin B2): 150-300 mg/day — trial mandatory (2-4 weeks minimum); "
            "responders: dramatic improvement in metabolic stability and function; "
            "CoQ10 supplementation (especially in ETFDH/riboflavin-responsive): "
            "100-300 mg/day (CoQ10 deficiency secondary in ETFDH defects); "
            "DIETARY: "
            "Low-fat, high-carbohydrate diet; "
            "protein restriction (reduces acyl-CoA substrate load from amino acid catabolism); "
            "L-carnitine supplementation (secondary depletion); "
            "fasting avoidance; sick-day protocols; "
            "ACUTE CRISIS: IV glucose; L-carnitine IV; "
            "discontinue fasting; "
            "riboflavin IV if responder; "
            "KEY CLINICAL FACTS: "
            "Multiple acylcarnitines on NBS = diagnostic clue (most FAO show one predominant species); "
            "sweaty feet odour (isovaleric acid) = characteristic clinical clue; "
            "riboflavin trial MANDATORY before any dietary restriction; "
            "ETFDH mutations: most commonly riboflavin-responsive (secondary CoQ10 deficiency); "
            "glutaric acid type 2 ≠ glutaric aciduria type 1 (GA1/GCDH — different disease)."
        ),
        "age_of_onset": "Neonatal (severe forms) or childhood-adult (late-onset, riboflavin-responsive)",
        "inheritance": "AR",
        "locus": "15q24.2",
        "protein_size": "333 aa",
        "key_biomarker": "Multiple acylcarnitines elevated (C4/C5/C6/C8/C10) on NBS; ethylmalonic + glutaric acids urine",
        "pathognomonic": "Multiple acyl-CoA dehydrogenase deficiency + sweaty feet odour + riboflavin-responsiveness",
        "treatment": "Riboflavin 150-300 mg/day (trial mandatory); CoQ10; low-fat/protein diet; L-carnitine; fasting CI",
        "critical_flags": [
            "RIBOFLAVIN-TRIAL-MANDATORY-ALL-PATIENTS — 25-30% late-onset respond dramatically",
            "MULTIPLE-ACYLCARNITINES-NBS — C4+C5+C6+C8+C10 all elevated = MADD/GA2",
            "CoQ10-SUPPLEMENTATION-ETFDH — secondary CoQ10 deficiency in ETFDH mutations",
            "GA2-NOT-GA1 — glutaric aciduria TYPE 2 (ETFA/B/DH) ≠ type 1 (GCDH)",
            "SWEATY-FEET-ODOUR — isovaleric acid metabolite; characteristic clinical clue",
            "NEONATAL-DYSMORPHIC-FORM-FATAL — structural brain/renal anomalies; riboflavin ineffective",
            "FASTING-CI — avoid prolonged fast; sick-day glucose protocol",
            "ETFDH-MOST-RIBOFLAVIN-RESPONSIVE — check gene first to predict response",
        ],
    },
    # -- HMGCL -- HMG-CoA Lyase deficiency -------------------------------------
    {
        "gene": "HMGCL",
        "protein": (
            "HMGCL -- 1p36.11 AR -- 3-Hydroxy-3-Methylglutaryl-CoA-Lyase-325aa -- "
            "HMGCoA-Lyase-Deficiency-Hypoglycaemia-WITHOUT-Ketosis-PATHOGNOMONIC -- "
            "Organic-Acidaemia-WITH-Metabolic-Acidosis-NO-Ketones -- "
            "Saudi-Arabian-Portuguese-Founder -- "
            "Leucine-Rich-Foods-CI-LEUCINE-MOST-DANGEROUS"
        ),
        "alias": (
            "HMGCL (3-hydroxy-3-methylglutaryl-CoA lyase, mitochondrial); OMIM gene 613898; "
            "HMG-CoA lyase deficiency OMIM 246450. "
            "1p36.11; 325 aa; ~35 kDa; mitochondrial matrix; autosomal recessive. "
            "FUNCTION: HMGCL catalyses the final step of ketogenesis: "
            "HMG-CoA (3-hydroxy-3-methylglutaryl-CoA) → Acetoacetate + Acetyl-CoA. "
            "It also functions in leucine catabolism: "
            "Leucine → alpha-KIC → isovaleryl-CoA → β-methylcrotonyl-CoA → "
            "methylglutaconyl-CoA → HMG-CoA → Acetoacetate + Acetyl-CoA (via HMGCL). "
            "HMGCL is therefore REQUIRED for BOTH: "
            "1. Ketogenesis (hepatic: FA → acetyl-CoA → ketones); "
            "2. Leucine catabolism (final step). "
            "In HMGCL deficiency: "
            "HMG-CoA accumulates → excreted as 3-hydroxy-3-methylglutaric acid (HMG), "
            "3-methylglutaric acid, 3-methylglutaconic acid, "
            "3-hydroxyisovaleric acid — organic aciduria; "
            "KETOGENESIS IS ABOLISHED: "
            "no ketones can be generated (the ketogenic final step is blocked); "
            "during fasting/illness: glucose becomes depleted → "
            "normally, ketones provide brain energy during hypoglycaemia; "
            "in HMGCL: NO ketones generated → brain energy failure at lower glucose threshold; "
            "HYPOGLYCAEMIA WITHOUT KETONES = PATHOGNOMONIC for HMG-CoA lyase deficiency "
            "(and also for FAO defects that are hypoketotic — "
            "but HMGCL is an ORGANIC ACIDEMIA with acidosis, not a primary FAO defect; "
            "however metabolically causes hypoketotic hypoglycaemia like FAO). "
            "LEUCINE IS THE KEY TRIGGER: "
            "high-leucine meals → leucine enters catabolism → blocks at HMG-CoA → "
            "HMG-CoA accumulates → organic acids surge → metabolic acidosis; "
            "LEUCINE-RICH FOODS ARE CONTRAINDICATED (dairy, eggs, meat in excess, leucine supplements); "
            "CLINICAL PRESENTATION: "
            "Typically age 3-11 months (weaning onto protein-rich diet increases leucine load): "
            "acute encephalopathy (lethargy → coma); "
            "metabolic acidosis (anion gap); "
            "hypoglycaemia WITHOUT ketones (KEY diagnostic clue); "
            "hyperammonaemia (mild-moderate); "
            "hepatomegaly with elevated transaminases; "
            "vomiting, poor feeding; "
            "WITHOUT TREATMENT: "
            "cerebral oedema, death, or neurological sequelae. "
            "DIAGNOSIS: "
            "Urine organic acids: "
            "3-hydroxy-3-methylglutaric acid (HMG) — most specific; "
            "3-methylglutaric acid; 3-methylglutaconic acid; 3-hydroxyisovaleric acid; "
            "Plasma acylcarnitines: "
            "C6-carnitine (3-methylglutarylcarnitine) may be elevated (NBS); "
            "some NBS programmes detect via acylcarnitines; organic acids are more sensitive; "
            "Blood glucose: LOW; blood ketones (beta-hydroxybutyrate): ABSENT/VERY LOW "
            "(in the context of hypoglycaemia — this combination is the diagnostic hallmark); "
            "Plasma leucine: elevated during crisis; "
            "HMGCL gene sequencing. "
            "TREATMENT: "
            "ACUTE CRISIS: "
            "IV glucose 10% (dextrose) — correct hypoglycaemia immediately; "
            "stop all oral intake temporarily (especially protein/leucine); "
            "IV fluids with bicarbonate for metabolic acidosis; "
            "L-carnitine IV (secondary depletion); "
            "CHRONIC: "
            "LEUCINE-RESTRICTED DIET mandatory (limit leucine from dairy, eggs, meat); "
            "leucine target plasma <200 µmol/L; "
            "leucine supplements: ABSOLUTELY CONTRAINDICATED (bodybuilding supplements etc.); "
            "protein from low-leucine or amino-acid formula; "
            "FASTING: avoid prolonged fasting (no ketone reserve); "
            "sick-day protocol: glucose polymer + stop high-leucine protein in illness; "
            "DIET: high-carbohydrate; adequate fat (fat does not worsen HMGCL acutely); "
            "avoid very high fat/ketogenic diet (intended to increase ketones — paradoxically "
            "the added acetyl-CoA overloads HMGCL-dependent pathway); "
            "prognosis: excellent with treatment; "
            "without treatment: intellectual disability common. "
            "FOUNDER MUTATIONS: "
            "Saudi Arabian: c.109G>A (p.Gly37Arg) — most common Saudi/Arab variant; "
            "Portuguese/Iberian: p.E37X (splice site); "
            "these populations have disproportionately higher incidence. "
            "KEY CLINICAL FACTS: "
            "HMGCL = organic acidemia that behaves like an FAO disorder (hypoketotic); "
            "hypoglycaemia WITHOUT ketones = hallmark (shared with FAO defects but also HMGCL); "
            "leucine restriction is THE critical treatment (unique among organic acidemias); "
            "LEUCINE SUPPLEMENTS ABSOLUTELY CI — athletes/bodybuilders with HMGCL must avoid BCAA/leucine; "
            "NBS: organic acid profile is more reliable than acylcarnitines for this condition."
        ),
        "age_of_onset": "Infancy 3-11 months (after protein introduction); rarely neonatal",
        "inheritance": "AR",
        "locus": "1p36.11",
        "protein_size": "325 aa",
        "key_biomarker": "3-HMG (3-hydroxy-3-methylglutaric acid) urine; hypoglycaemia WITHOUT ketones",
        "pathognomonic": "Metabolic acidosis + hypoglycaemia WITHOUT ketones + 3-HMG in urine",
        "treatment": "Leucine restriction (CI leucine supplements); IV glucose in crisis; fasting CI; L-carnitine",
        "critical_flags": [
            "HYPOGLYCAEMIA-WITHOUT-KETOSIS-PATHOGNOMONIC — cannot generate ketones (HMGCL = last step)",
            "LEUCINE-RICH-FOODS-CI — dairy/eggs/meat in excess trigger crisis",
            "LEUCINE-SUPPLEMENTS-ABSOLUTELY-CI — BCAA/bodybuilding supplements forbidden",
            "3-HMG-URINE-MOST-SPECIFIC — 3-hydroxy-3-methylglutaric acid diagnostic",
            "KETOGENIC-DIET-CONTRAINDICATED — overloads HMG-CoA pathway",
            "FASTING-CI — no ketone reserve = brain energy failure at lower glucose",
            "SAUDI-PORTUGUESE-FOUNDER — c.109G>A (Saudi); p.E37X (Iberian)",
            "ORGANIC-ACIDAEMIA-NOT-FAO — but behaves like FAO (hypoketotic); different mechanism",
        ],
    },
]


def _make_patients(gene_data, seed):
    rng = random.Random(seed)
    ages = [rng.randint(0, 50) for _ in range(40)]
    sexes = [rng.choice(["M", "F"]) for _ in range(40)]
    gene = gene_data["gene"]
    inheritance = gene_data["inheritance"]
    severity_choices = ["mild", "moderate", "severe"]
    # FAO disorders: severity weights depend on typical phenotype
    sev_weights = {
        "ACADM": [40, 40, 20],    # mostly mild-moderate with NBS
        "ACADVL": [20, 40, 40],   # cardiac form severe, myopathic mild
        "HADHA": [20, 40, 40],    # neuropathy/retinopathy severe
        "CPT1A": [50, 35, 15],    # Inuit variant often mild
        "CPT2": [35, 45, 20],     # muscle form moderate
        "SLC25A20": [10, 25, 65], # neonatal severe
        "ETFA": [25, 40, 35],     # three forms
        "HMGCL": [30, 45, 25],    # treatable
    }
    weights = sev_weights.get(gene, [25, 45, 30])
    patients = []
    for i in range(40):
        severity = rng.choices(severity_choices, weights=weights)[0]
        age = ages[i]
        sex = sexes[i]
        on_diet = rng.random() > 0.20
        rhabdo = gene in ("CPT2", "ACADVL", "HADHA") and rng.random() > 0.50
        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis": age,
            "sex": sex,
            "severity": severity,
            "inheritance": inheritance,
            "on_diet": on_diet,
            "rhabdomyolysis_history": rhabdo,
            "key_biomarker_abnormal": True,
            "family_cascade": rng.random() > 0.45,
        })
    return patients


def _build_cohort():
    all_patients = []
    for idx, gene_data in enumerate(FAO_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_patients(gene_data, seed))
    return all_patients


COHORT = _build_cohort()


# ── API response functions ─────────────────────────────────────────────────────

def overview():
    total = len(COHORT)
    gene_counts = {}
    severity_counts = {"mild": 0, "moderate": 0, "severe": 0}
    for p in COHORT:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1
        severity_counts[p["severity"]] = severity_counts.get(p["severity"], 0) + 1

    on_diet = sum(1 for p in COHORT if p.get("on_diet"))
    cascade = sum(1 for p in COHORT if p.get("family_cascade"))
    rhabdo  = sum(1 for p in COHORT if p.get("rhabdomyolysis_history"))

    genes_covered = len(FAO_GENES)
    ar_genes = sum(1 for g in FAO_GENES if g["inheritance"] == "AR")

    return {
        "atlas": (
            "Hereditary-FAO-Atlas — Complete 8-Gene Hereditary Fatty Acid Oxidation "
            "Disorders Atlas"
        ),
        "subtitle": (
            "ACADM (MCAD-Most-Common-FAO-NBS-Universal-c985AG-Fasting-CI) · "
            "ACADVL (VLCAD-C14:1-NBS-Cardiac-MCT-Therapeutic-Triheptanoin) · "
            "HADHA (LCHAD-TFP-Neuropathy+Retinopathy-PATHOGNOMONIC-Maternal-AFLP-DHA) · "
            "CPT1A (CPT1A-C0-HIGH-Inuit-Founder-p.P479L-MCT-CI) · "
            "CPT2 (CPT2-Myoglobinuria-Exercise-PATHOGNOMONIC-Statins-NSAIDs-CI) · "
            "SLC25A20 (CACT-Neonatal-Arrhythmia-Hyperammonaemia-C0-Very-Low) · "
            "ETFA (MADD-GA2-Multiple-Acylcarnitines-Riboflavin-25-30pct-Responsive) · "
            "HMGCL (HMGCoA-Lyase-Hypo-Without-Ketosis-PATHOGNOMONIC-Leucine-CI) — "
            f"320 Patients (8×40, Seeds {SEED_BASE}–{SEED_BASE+7})"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
        "aggregate_stats": {
            "genes_covered": genes_covered,
            "ar_genes": ar_genes,
            "x_linked_genes": 0,
            "ad_genes": 0,
            "patients_per_gene": total // genes_covered,
            "on_diet_pct": round(on_diet / total * 100, 1),
            "family_cascade_pct": round(cascade / total * 100, 1),
            "rhabdomyolysis_history_pct": round(rhabdo / total * 100, 1),
            "severity_mild_pct": round(severity_counts["mild"] / total * 100, 1),
            "severity_moderate_pct": round(severity_counts["moderate"] / total * 100, 1),
            "severity_severe_pct": round(severity_counts["severe"] / total * 100, 1),
        },
        "genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "disorder": g["pathognomonic"].split("+")[0].strip(),
                "treatment": g["treatment"],
                "n_patients": gene_counts.get(g["gene"], 0),
                "key_biomarker": g["key_biomarker"],
            }
            for g in FAO_GENES
        ],
        "top_alerts": [
            flag
            for g in FAO_GENES
            for flag in (g["critical_flags"][:2])
        ],
        "critical_treatment_alerts": [
            flag
            for g in FAO_GENES
            for flag in g["critical_flags"]
        ],
    }


def breakdown():
    per_gene = {}
    for g in FAO_GENES:
        gene = g["gene"]
        pts = [p for p in COHORT if p["gene"] == gene]
        mild     = sum(1 for p in pts if p["severity"] == "mild")
        moderate = sum(1 for p in pts if p["severity"] == "moderate")
        severe   = sum(1 for p in pts if p["severity"] == "severe")
        on_diet  = sum(1 for p in pts if p.get("on_diet"))
        rhabdo   = sum(1 for p in pts if p.get("rhabdomyolysis_history"))
        per_gene[gene] = {
            "gene": gene,
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "n_patients": len(pts),
            "severity": {"mild": mild, "moderate": moderate, "severe": severe},
            "on_diet": on_diet,
            "on_diet_pct": round(on_diet / len(pts) * 100, 1) if pts else 0,
            "rhabdomyolysis_history": rhabdo,
            "family_cascade": sum(1 for p in pts if p.get("family_cascade")),
            "protein_description": g["protein"],
            "age_of_onset": g["age_of_onset"],
        }
    return {
        "atlas": "Hereditary-FAO-Atlas — Per-Gene Breakdown",
        "genes": per_gene,
        "aggregate": {
            "total_patients": len(COHORT),
            "total_genes": len(FAO_GENES),
            "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
            "all_inheritance": list({g["inheritance"] for g in FAO_GENES}),
        },
    }


def definitions():
    defs = {}
    for g in FAO_GENES:
        defs[g["gene"]] = g["alias"]

    defs["Fatty Acid Oxidation — Overview and Pathway"] = (
        "Mitochondrial fatty acid beta-oxidation (FAO) is the primary source of energy "
        "during fasting in heart, skeletal muscle, liver, and kidney. "
        "BETA-OXIDATION PATHWAY (simplified): "
        "Step 1. Carnitine shuttle: "
        "Long-chain acyl-CoA + carnitine → acylcarnitine (CPT1A/B) → "
        "acylcarnitine into mitochondria (CACT/SLC25A20) → "
        "acyl-CoA + carnitine (CPT2). "
        "Step 2. Beta-oxidation cycle (4 enzymes per cycle): "
        "1. Acyl-CoA dehydrogenase (VLCAD for C14-C20; LCAD; MCAD for C6-C12; SCAD for C4-C6) "
        "→ 2-enoyl-CoA + FADH2 (→ ETF → ETFDH → CoQ10 → CIII → ATP); "
        "2. Enoyl-CoA hydratase → 3-hydroxyacyl-CoA; "
        "3. 3-Hydroxyacyl-CoA dehydrogenase → 3-ketoacyl-CoA + NADH; "
        "4. Thiolase → shorter acyl-CoA + acetyl-CoA. "
        "(Steps 2-4 for long chain: HADHA encodes hydratase + LCHAD; HADHB encodes thiolase). "
        "Step 3. Ketogenesis (liver only): "
        "Acetyl-CoA → acetoacetate + β-hydroxybutyrate (via HMGCS2 → HMGCL). "
        "DEFECTS AND THEIR CLASSIFICATION: "
        "Carnitine shuttle defects: CPT1A, CACT (SLC25A20), CPT2; "
        "Acyl-CoA dehydrogenase defects: VLCAD, MCAD, ETFA/ETFB/ETFDH (MADD); "
        "Trifunctional protein defects: HADHA (LCHAD), HADHB (TFP-thiolase); "
        "Ketogenesis defect: HMGCL (organic acidemia with hypoketotic hypoglycaemia). "
        "SHARED CLINICAL FEATURES: "
        "Hypoketotic hypoglycaemia (absent/low ketones despite hypoglycaemia); "
        "triggers: fasting, intercurrent illness, prolonged exercise; "
        "hepatomegaly with raised transaminases; "
        "cardiomyopathy (long-chain defects); "
        "rhabdomyolysis (long-chain + CPT2). "
        "KEY BIOCHEMICAL PATTERNS: "
        "C0 very high, long-chain acylcarnitines low/normal → CPT1A; "
        "C0 very low, long-chain acylcarnitines very high → CACT/CPT2 neonatal; "
        "C8 dominant → MCAD; C14:1 dominant → VLCAD; C16-OH dominant → LCHAD/TFP; "
        "Multiple acylcarnitines (C4+C5+C6+C8+C10) → MADD/GA2 (ETFA/B/DH); "
        "Hypoketotic hypoglycaemia + organic acidemia → HMGCL. "
    )

    defs["Hypoketotic Hypoglycaemia — Differential Diagnosis"] = (
        "Hypoketotic hypoglycaemia = blood glucose <2.6 mmol/L WITH absent or inappropriately "
        "low ketones (β-hydroxybutyrate <1 mmol/L during hypoglycaemia). "
        "NORMAL RESPONSE: during hypoglycaemia, ketones rise to >1 mmol/L as brain fuel; "
        "absence of this ketotic response = impairment of ketogenesis OR fatty acid oxidation. "
        "DIFFERENTIAL DIAGNOSIS: "
        "FAO DEFECTS: MCAD, VLCAD, LCHAD/TFP, CACT, CPT1A, CPT2 severe forms; "
        "KETOGENESIS DEFECTS: HMGCL, HMGCS2; "
        "HYPERINSULINISM: insulin suppresses lipolysis AND promotes glycolysis; "
        "insulin excess causes hypoketotic hypoglycaemia — check insulin at time of hypoglycaemia; "
        "GROWTH HORMONE / CORTISOL DEFICIENCY: impair gluconeogenesis + lipolysis; "
        "HEPATIC FAILURE: impairs gluconeogenesis and ketogenesis. "
        "CRITICAL TRIAGE QUESTION: "
        "At time of hypoglycaemia: "
        "β-hydroxybutyrate low + insulin low → FAO defect or ketogenesis defect; "
        "β-hydroxybutyrate low + insulin HIGH → hyperinsulinism; "
        "obtain blood glucose + insulin + β-hydroxybutyrate + fatty acids simultaneously; "
        "send plasma acylcarnitines + urine organic acids DURING crisis for best yield. "
        "EMERGENCY TREATMENT: "
        "Do NOT wait for confirmatory results in an acutely encephalopathic child with "
        "hypoketotic hypoglycaemia → give IV dextrose 10% 2 mL/kg bolus immediately."
    )

    defs["Carnitine Cycle — CPT1A vs CACT vs CPT2 Comparison"] = (
        "The carnitine cycle transports long-chain fatty acids into mitochondria. "
        "Three disorders disrupt different steps: "
        "CPT1A (outer mitochondrial membrane): "
        "converts acyl-CoA + carnitine → acylcarnitine → C0 FREE carnitine HIGH; "
        "long-chain acylcarnitines NORMAL/LOW (they cannot be formed in excess); "
        "no cardiac involvement (CPT1B cardiac intact); MCT CONTRAINDICATED. "
        "CACT/SLC25A20 (inner mitochondrial membrane translocase): "
        "cannot transport acylcarnitines into matrix → all accumulate; "
        "C0 VERY LOW + ALL long-chain acylcarnitines VERY HIGH; "
        "most severe: cardiac arrhythmia + hyperammonaemia; "
        "MCT therapeutic. "
        "CPT2 (matrix face of inner mitochondrial membrane): "
        "cannot reconvert acylcarnitine → acyl-CoA inside mitochondria; "
        "C16/C18 elevated + C0 low; "
        "muscle form most common (adult rhabdomyolysis); "
        "MYOGLOBINURIA is PATHOGNOMONIC for muscle CPT2; "
        "MCT therapeutic; statins/NSAIDs CI. "
    )

    defs["MCT — When to Use vs When to Avoid in FAO Defects"] = (
        "Medium-chain triglycerides (MCT, C8-C10) bypass the carnitine shuttle "
        "(they enter mitochondria without CPT1/CACT/CPT2) and bypass VLCAD/LCHAD "
        "(they are oxidised by MCAD, not VLCAD). "
        "MCT THERAPEUTIC (use to supplement): "
        "VLCAD deficiency — MCT is the safe fat source; "
        "LCHAD/TFP (HADHA deficiency) — MCT bypasses HADHA; "
        "CPT2 muscle form — MCT bypasses CPT2; "
        "CACT (SLC25A20) deficiency — MCT bypasses carnitine shuttle. "
        "MCT CONTRAINDICATED (avoid): "
        "MCAD deficiency — MCT is C8-C10 = LOADS the deficient MCAD pathway; "
        "CPT1A deficiency — MCT supplementation loads medium-chain FAO in context of "
        "already impaired hepatic FA handling; most centres avoid; "
        "MADD/ETFA — MCT provides additional medium-chain substrate to already overwhelmed system. "
        "PRACTICAL RULE: "
        "Long-chain FAO defect (VLCAD, LCHAD, CPT2, CACT) → MCT = therapeutic. "
        "Medium-chain FAO defect (MCAD) → MCT = CI. "
        "Carnitine synthesis defect (CPT1A) → MCT = CI. "
    )

    return {
        "atlas": "Hereditary-FAO-Atlas — Clinical Definitions",
        "definitions": defs,
        "total_genes": len(FAO_GENES),
        "total_definition_entries": len(defs),
    }
