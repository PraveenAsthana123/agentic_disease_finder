#!/usr/bin/env python3
"""Hereditary-Muscular-Dystrophy-Atlas — Complete 8-Gene Hereditary Muscular Dystrophy Atlas
DMD    (dystrophin; 3685 aa; Xp21.2; XLR;
         Duchenne MD [DMD] — frameshift/nonsense, wheelchair by 12y, DCM 90% by 18y, Elevidys/SRP-9001 FDA 2023;
         Becker MD [BMD] — in-frame, ambulatory into 40s;
         seed SEED_BASE+0) ·
DMPK   (myotonin protein kinase; 629 aa; 19q13.32; AD;
         Myotonic Dystrophy Type 1 [DM1/Steinert] — CTG repeat expansion;
         multisystem: myotonia + DCM + heart block + cataracts + endocrine;
         ANAESTHESIA EXTREME RISK — depolarising NMB absolute CI;
         seed SEED_BASE+1) ·
SMCHD1 (structural maintenance of chromosomes flexible hinge domain 1; 2005 aa; 18p11.32; AD;
         FSHD2 — epigenetic D4Z4 hypomethylation; requires permissive 4qA haplotype;
         asymmetric facioscapulohumeral weakness PATHOGNOMONIC; losmapimod trial;
         seed SEED_BASE+2) ·
EMD    (emerin; 254 aa; Xq28; XLR;
         EDMD1 Emery-Dreifuss MD — early contractures BEFORE weakness PATHOGNOMONIC;
         humeral-peroneal distribution; LETHAL cardiac arrhythmia; ICD mandatory;
         seed SEED_BASE+3) ·
LMNA   (lamin A/C; 664 aa; 1q22; AD;
         EDMD2 / LMNA-related DCM / CMD1A — MOST LETHAL MD gene;
         ICD regardless of LVEF (Padua score); non-missense highest risk; mean cardiac event age 36y;
         seed SEED_BASE+4) ·
CAPN3  (calpain 3; 821 aa; 15q15.1; AR;
         LGMD-R1 / calpainopathy — most common LGMD globally ~30%;
         pelvifemoral pattern; NO cardiac; CK 5-80x; normal CK possible (unique among LGMD);
         seed SEED_BASE+5) ·
DYSF   (dysferlin; 2080 aa; 2p13.2; AR;
         LGMD-R2 / Miyoshi Myopathy — CK 50-100x UPPER LIMIT PATHOGNOMONIC;
         distal posterior (Miyoshi) OR proximal (LGMD-R2); STEROIDS CONTRAINDICATED — worsen disease;
         seed SEED_BASE+6) ·
GNE    (UDP-GlcNAc 2-epimerase/ManNAc kinase; 722 aa; 9p13.3; AR;
         GNE Myopathy / IBM2 / Nonaka — QUADRICEPS SPARED PATHOGNOMONIC; rimmed vacuoles;
         sialic-acid synthesis; NeuAc/aceneuramic-acid trials;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1702–1709)
"""

import random

SEED_BASE = 1702

MD_GENES = [
    # ── DMD — Duchenne / Becker MD ────────────────────────────────────────────
    {
        "gene": "DMD",
        "protein": "DMD — Xp21.2 XLR — Dystrophin-3685aa — Duchenne-MD-Elevidys-FDA2023 — Becker-MD-In-Frame — Cardiac-DCM-90pct-Age18 — ICD-ACEi-from-Age10",
        "alias": (
            "DMD (dystrophin); OMIM gene 300377; "
            "Duchenne Muscular Dystrophy (DMD) OMIM 310200; Becker MD (BMD) OMIM 300376. "
            "Xp21.2; 3685 aa; ~427 kDa; X-linked recessive. "
            "FUNCTION: Dystrophin is the largest gene in the human genome (2.4 Mb). "
            "The protein links the intracellular actin cytoskeleton to the extracellular matrix via "
            "the dystrophin-associated protein complex (DAPC: dystroglycans, sarcoglycans, syntrophins, dystrobrevin). "
            "This mechanical link protects sarcolemma from contraction-induced stress. "
            "DMD LOF → membrane fragility → Ca2+ influx → necrosis-regeneration cycles → fibrosis. "
            "DUCHENNE MD (DMD): out-of-frame mutations (deletions exons 45-55 region most common, ~65%; "
            "duplications ~11%; point mutations ~24%) → NO functional dystrophin; "
            "onset: proximal weakness age 3-5y; Gowers sign (rising from floor using arms); "
            "Trendelenburg/waddling gait; CALF PSEUDOHYPERTROPHY PATHOGNOMONIC (fibrofatty replacement); "
            "Meryon sign (slipping through axillary grip); "
            "loss of ambulation by 12y without steroids; "
            "CK: massively elevated 10,000-100,000 IU/L (highest CK of all muscular dystrophies); "
            "CARDIAC: DCM in ~90% by age 18; most Duchenne die from cardiac/respiratory failure in 20s-30s; "
            "annual echo + cardiac MRI from age 10; ACE-inhibitor/ARB start at age 10 regardless of LVEF; "
            "beta-blockers when DCM confirmed; "
            "RESPIRATORY: FVC monitoring q6 months; NIV (BiPAP) when FVC <50% or SaO2 drops; "
            "cough-assist device; secretion management; "
            "TREATMENT — CORTICOSTEROIDS: deflazacort (preferred) or prednisone daily/weekend; "
            "delay loss of ambulation by 2-3 years; maintain upper limb function; "
            "EXON SKIPPING: FDA-approved antisense oligonucleotides: "
            "eteplirsen/Exondys 51 (exon 51, ~14% DMD); "
            "golodirsen/Vyondys 53 (exon 53, ~8%); "
            "viltolarsen/Viltepso (exon 53); casimersen (exon 45, ~8%); "
            "GENE THERAPY: SRP-9001 (Elevidys/delandistrogene moxeparvovec) — "
            "micro-dystrophin AAVrh74, FDA accelerated approval June 2023 for 4-17y ambulatory DMD; "
            "regular approval August 2024 for 4-17y; "
            "STOP CODON READTHROUGH: ataluren (Translarna) — for nonsense mutations; approved EU/UK not FDA; "
            "BECKER MD (BMD): in-frame mutations → truncated but semi-functional dystrophin; "
            "mild phenotype; ambulatory into 40s-50s; CK elevated 5-100x; cardiac DCM still occurs (later); "
            "female carriers: ~50% asymptomatic carriers; ~10% manifesting (lyonisation); "
            "carrier cardiac surveillance (echo) recommended."
        ),
        "locus": "Xp21.2",
        "aa": 3685,
        "kDa": 427,
        "omim_gene": "300377",
        "omim_disease": "310200",
        "inheritance": "XLR — X-linked Recessive — males fully affected; females carrier (50% risk sons affected); female manifesting carriers: ~10% with cardiac disease; de novo: ~30% DMD mutations",
        "gene_class": "Structural (cytoskeletal linker); connects F-actin to DAPC; sarcolemmal scaffold; absent = membrane fragility; truncated (BMD) = partial function",
        "key_alerts": [
            "DMD-ELEVIDYS-FDA2023-GENE-THERAPY: SRP-9001 (delandistrogene moxeparvovec) FDA-approved 2023-2024 for 4-17y ambulatory DMD; micro-dystrophin construct restores partial function; eligibility: confirmed DMD pathogenic variant, ambulatory, age 4-17; liver enzyme monitoring post-infusion",
            "DMD-ACEi-ARB-AGE10-CARDIAC-MANDATORY: ACE-inhibitor or ARB started at age 10 regardless of echo/LVEF — prevents DCM progression; annual cardiac MRI from age 10; ICD when LVEF <35% or sustained VT",
            "DMD-CK-10000-100000-HIGHEST-ALL-MD: CK 10,000-100,000 IU/L in Duchenne (highest of any MD); present from birth; elevated before symptoms; newborn screening feasible; CK >10x ULN in first 5 years = investigation mandatory",
            "DMD-EXON-SKIPPING-GENOTYPE-SPECIFIC: Eteplirsen (exon 51, ~14%), golodirsen/viltolarsen (exon 53, ~8%), casimersen (exon 45, ~8%); genetic report MUST state exact deletion/duplication exons before choosing drug; confirm reading-frame prediction",
            "BMD-CARDIAC-STILL-PRESENT-DESPITE-MILD-MUSCLE: Becker MD — mild proximal weakness does NOT mean no cardiac risk; DCM develops later (30s-50s); annual echo in all BMD from age 15; sudden death in ambulatory BMD males from DCM",
            "DMD-CALF-PSEUDOHYPERTROPHY-PATHOGNOMONIC: gastrocnemius enlargement with fibrofatty replacement; hard on palpation; early sign in toddlers; PATHOGNOMONIC for dystrophinopathy (also seen BMD, LGMD-R3/4 sarcoglycanopathies)",
            "DMD-STEROIDS-DEFLAZACORT-PREFERRED: deflazacort has lower weight-gain side effect vs prednisone; start when motor development plateau (typically age 4-6); daily regimen delays ambulation loss by ~2y",
            "DMD-RESPIRATORY-FVC-MONITOR-q6m: FVC q6 months; NIV (BiPAP) when FVC <50% predicted or nocturnal SaO2 drops; cough-assist (MI-E) device for secretion; prophylactic antibiotics during respiratory infections",
        ],
        "etiologies": {
            "Duchenne_frameshift_deletion": {"pct": 58, "phenotype": "severe", "dystrophin": "absent", "notes": "exon 45-55 deletion hotspot; CK 50,000+; loss ambulation by 12y"},
            "Duchenne_nonsense_point": {"pct": 18, "phenotype": "severe", "dystrophin": "absent", "notes": "ataluren eligible (EU/UK); ~11% of DMD"},
            "Duchenne_duplication_frame": {"pct": 9, "phenotype": "severe", "dystrophin": "absent", "notes": "duplications out-of-frame; sometimes exon skip restores frame"},
            "Becker_inframe_deletion": {"pct": 12, "phenotype": "mild_variable", "dystrophin": "truncated", "notes": "central rod domain deletions often mild; exon 48-49 classic mild BMD"},
            "Manifesting_carrier": {"pct": 3, "phenotype": "variable", "dystrophin": "mosaic", "notes": "lyonisation; cardiac surveillance essential"},
        },
        "stats": {
            "incidence": "1 in 3,500 live male births (DMD); 1 in 18,000 live male births (BMD)",
            "de_novo_rate_pct": 30,
            "mean_dx_age_DMD_y": 4.5,
            "mean_loss_ambulation_untreated_y": 10,
            "mean_loss_ambulation_on_steroids_y": 13,
            "cardiac_dcm_by_18y_pct": 90,
            "exon_skip_eligible_pct": 30,
            "gene_therapy_eligible_pct": 100,
        },
        "dx_delay_distribution": {"mean_months": 18, "median_months": 14, "range": "3-72", "notes": "often diagnosed after parental concern about Gowers sign; CK first test confirms"},
    },

    # ── DMPK — Myotonic Dystrophy Type 1 ─────────────────────────────────────
    {
        "gene": "DMPK",
        "protein": "DMPK — 19q13.32 AD — Myotonin-Protein-Kinase-629aa — DM1-Steinert-CTG-Repeat — Multisystem-Myotonia-Cardiac-Cataracts — ANAESTHESIA-EXTREME-RISK — Mexiletine-Myotonia",
        "alias": (
            "DMPK (DM protein kinase); OMIM gene 605377; "
            "Myotonic Dystrophy Type 1 (DM1, Steinert disease) OMIM 160900. "
            "19q13.32; 629 aa; ~69 kDa; autosomal dominant (CTG trinucleotide repeat expansion). "
            "FUNCTION: DMPK encodes myotonin protein kinase, a serine/threonine kinase expressed in skeletal muscle, "
            "cardiac muscle, and brain. The pathomechanism is RNA gain-of-function, NOT protein LOF: "
            "expanded CUG repeat tracts in DMPK mRNA → nuclear retention → sequestration of "
            "MBNL1 (muscleblind-like protein 1) → aberrant splicing of numerous downstream targets "
            "(CLCN1 → myotonia; TNNT2 → cardiomyopathy; IR/INSR → insulin resistance; etc.). "
            "REPEAT CATEGORIES: 5-34 CTG = normal; 35-49 = premutation (no symptoms); "
            "50-99 = mild DM1; 100-999 = classic DM1; ≥1000 = congenital DM1 (CDM). "
            "ANTICIPATION: repeat expands in maternal transmission (especially severe CDM via mother); "
            "offspring at risk of more severe disease than parent. "
            "CLINICAL FEATURES — MULTISYSTEM: "
            "Myotonia: inability to relax muscle after contraction; grip myotonia; "
            "percussion myotonia of tongue/thenar; worsened by cold; mexiletine first-line treatment; "
            "Weakness: distal-predominant initially (foot drop, grip weakness) → proximal later; "
            "hatchet face (temporalis/masseter wasting) + ptosis + dysarthria = classic facial dysmorphism; "
            "CARDIAC — MANDATORY SURVEILLANCE: "
            "Conduction defects (PR prolongation, AV block, bundle branch block) — "
            "sudden death from complete heart block in 30% of DM1 deaths; "
            "pacemaker if PR >200ms, Mobitz I/II, or 3rd degree AV block; "
            "ICD if low LVEF or sustained VT; annual ECG + 24h Holter; "
            "Cataracts: posterior subcapsular cataracts in 90% by age 40 — slit-lamp examination; "
            "Endocrine: insulin resistance, testicular atrophy, gonadal dysfunction; "
            "GI: dysphagia, constipation, gastroparesis; "
            "CNS: frontal lobe dysfunction, hypersomnia, cognitive decline; "
            "ANAESTHESIA — EXTREME RISK: "
            "DM1 patients have high anesthesia mortality; "
            "AVOID: succinylcholine (suxamethonium) → exaggerated hyperkalemia + myotonic crisis; "
            "AVOID: volatile anaesthetics → prolonged post-op weakness + respiratory failure; "
            "PREFER: regional/spinal anaesthesia; total IV anaesthesia (propofol + remifentanil) if GA needed; "
            "AVOID: NEOSTIGMINE (worsens myotonia); "
            "have cardiac defibrillator available; post-op ICU monitoring mandatory; "
            "CONGENITAL DM1 (CDM): ≥1000 CTG; maternal origin; severe neonatal hypotonia, "
            "respiratory failure, feeding difficulties; club feet; later intellectual disability; "
            "TREATMENT: mexiletine (mexitil) for myotonia — sodium channel blocker; "
            "no disease-modifying treatment approved; antisense + small molecule trials ongoing."
        ),
        "locus": "19q13.32",
        "aa": 629,
        "kDa": 69,
        "omim_gene": "605377",
        "omim_disease": "160900",
        "inheritance": "AD — Autosomal Dominant — CTG repeat expansion; ANTICIPATION (expanding in successive generations especially maternal transmission); de novo mutations rare (expansion of premutation)",
        "gene_class": "Serine/threonine kinase; pathomechanism = RNA gain-of-function (not protein LOF); expanded CUG tracts sequester MBNL1 → aberrant splicing of CLCN1, TNNT2, INSR, BIN1, etc.",
        "key_alerts": [
            "DMPK-ANAESTHESIA-EXTREME-RISK-DEPOLARISING-NMB-ABSOLUTE-CI: succinylcholine/suxamethonium ABSOLUTE CI in DM1 — triggers myotonic crisis + hyperkalemia; volatile agents worsen post-op weakness; AVOID neostigmine reversal; prefer spinal/epidural; if GA → TIVA (propofol+remifentanil); MANDATORY anesthesia risk disclosure pre-op",
            "DMPK-CARDIAC-PACEMAKER-SUDDEN-DEATH-30PCT: 30% of DM1 sudden deaths = complete heart block; annual ECG + 24h Holter; pacemaker for PR >200ms, any Mobitz block; ICD if LVEF <35% or VT; electrophysiology study if any AV block detected",
            "DMPK-ANTICIPATION-MATERNAL-CDM-RISK: CTG repeat EXPANDS in transmission especially maternal; mother with mild DM1 (200 CTG) can have child with CDM (>1000 CTG, severe neonatal); prenatal CTG sizing mandatory; early postnatal respiratory support if CDM suspected",
            "DMPK-MYOTONIA-MEXILETINE-FIRST-LINE: mexiletine (150-200mg TID) reduces myotonia; QTc monitoring; alternative: lamotrigine or carbamazepine (second-line); AVOID quinine (cardiac risk); cold worsens myotonia — warm environment",
            "DMPK-REPEAT-SIZE-PHENOTYPE-CORRELATION: 50-99 CTG = mild (cataracts ± mild myotonia); 100-999 = classic DM1 (full multisystem); ≥1000 = congenital (neonatal hypotonia, cognitive impairment); ANTICIPATION — repeat sizing MANDATORY in offspring of affected parent",
            "DMPK-CATARACTS-90PCT-BY-40Y: posterior subcapsular cataracts in 90% by age 40; annual slit-lamp; cataract surgery effective but anaesthesia risk applies",
            "DMPK-INSULIN-RESISTANCE-ENDOCRINE: insulin resistance independent of obesity; testicular atrophy + low testosterone in males; thyroid function check annually; DM-related hyperglycaemia requires standard management",
            "DMPK-HYPERSOMNIA-MODAFINIL: excessive daytime sleepiness (CNS involvement); modafinil off-label effective; CPAP if OSA co-exists; cognitive assessment at diagnosis",
        ],
        "etiologies": {
            "classic_DM1_100_999_CTG": {"pct": 55, "phenotype": "classic_multisystem", "notes": "onset 20-40y; full cardiac + weakness + myotonia + cataracts"},
            "mild_DM1_50_99_CTG": {"pct": 20, "phenotype": "mild", "notes": "cataracts dominant; mild grip myotonia; late cardiac issues"},
            "severe_DM1_1000plus_CTG": {"pct": 12, "phenotype": "severe_adult", "notes": "early onset; rapid progression; severe cardiac + cognitive"},
            "CDM_congenital_1000plus_CTG": {"pct": 8, "phenotype": "congenital", "notes": "neonatal hypotonia; respiratory failure at birth; maternal origin"},
            "childhood_onset_100_500_CTG": {"pct": 5, "phenotype": "childhood", "notes": "cognitive + motor delay; facial weakness; cardiac from adolescence"},
        },
        "stats": {
            "prevalence": "1 in 8,000",
            "mean_dx_age_y": 29,
            "cardiac_pacemaker_lifetime_pct": 30,
            "cataracts_by_40y_pct": 90,
            "anaesthesia_mortality_relative_risk": "2-5x general population",
            "anticipation_observed_pct": 85,
        },
        "dx_delay_distribution": {"mean_months": 36, "median_months": 24, "range": "6-180", "notes": "multisystem presentation delays diagnosis; myotonia often not reported by patient; cataracts sometimes first referral"},
    },

    # ── SMCHD1 — FSHD2 ───────────────────────────────────────────────────────
    {
        "gene": "SMCHD1",
        "protein": "SMCHD1 — 18p11.32 AD — 2005aa — FSHD2-Epigenetic-D4Z4-Hypomethylation — Requires-4qA-Permissive-Haplotype — Asymmetric-Weakness-PATHOGNOMONIC — Losmapimod-Trial",
        "alias": (
            "SMCHD1 (structural maintenance of chromosomes flexible hinge domain 1); OMIM gene 614982; "
            "FSHD2 (facioscapulohumeral muscular dystrophy type 2) OMIM 158901. "
            "18p11.32; 2005 aa; ~226 kDa; autosomal dominant (haploinsufficiency). "
            "FUNCTION: SMCHD1 is a chromatin modifier that maintains epigenetic silencing of D4Z4 "
            "macrosatellite repeats on chromosome 4q35 (and 10q26). "
            "Normal: SMCHD1 methylates D4Z4 → silences DUX4 transcription → no muscle toxicity. "
            "FSHD2 pathomechanism: SMCHD1 haploinsufficiency → D4Z4 hypomethylation → "
            "DUX4 (double homeobox 4) expression → DUX4 protein is highly toxic to muscle cells "
            "→ apoptosis + inflammation + aberrant differentiation → facioscapulohumeral weakness. "
            "DIGENIC MECHANISM: FSHD2 requires BOTH: "
            "(1) SMCHD1 pathogenic variant; AND (2) permissive 4qA haplotype (polyadenylation signal "
            "stabilises DUX4 mRNA — without 4qA, DUX4 mRNA is degraded even if expressed). "
            "~98% of FSHD cases are FSHD1 (D4Z4 contraction to ≤10 units + permissive haplotype); "
            "~2% are FSHD2 (SMCHD1/DNMT3B mutations + normal D4Z4 count + permissive haplotype). "
            "CLINICAL FEATURES: "
            "Facioscapulohumeral distribution: facial weakness (cannot whistle/close eyes fully) — "
            "scapular winging — humeral (biceps/triceps) weakness — peroneal/tibialis anterior weakness; "
            "ASYMMETRY — PATHOGNOMONIC: weakness dramatically asymmetric (right > left, no clear explanation); "
            "highly variable expressivity (gene penetrance ~95% but severity 1-100); "
            "inter- and intra-familial variability extreme (same mutation = mild cousin and wheelchair uncle); "
            "Abdominal: rectus abdominis weakness → Beevor sign (umbilicus moves upward on flexion neck); "
            "Retinal vasculopathy (Coats disease) in severe cases — ophthalmology screening; "
            "SNHL in ~75% of severe early-onset cases; "
            "TREATMENT: no approved disease-modifying treatment; "
            "losmapimod (p38 MAPK inhibitor) — phase 3 clinical trial; "
            "TREATMENT GOAL: maintain scapular stability (physiotherapy), "
            "scapular fixation surgery (scapulothoracic fusion) for severe winging; "
            "foot drop — ankle-foot orthosis (AFO); "
            "avoid repetitive shoulder overhead activities; "
            "FSHD2-specific: SMCHD1 mutation testing + D4Z4 methylation assay + 4qA haplotyping "
            "required for diagnosis (SMCHD1 panel alone insufficient — haplotype determination needed)."
        ),
        "locus": "18p11.32",
        "aa": 2005,
        "kDa": 226,
        "omim_gene": "614982",
        "omim_disease": "158901",
        "inheritance": "AD — Autosomal Dominant — haploinsufficiency; BUT DIGENIC: requires permissive 4qA haplotype on chr4; 50% transmission risk for SMCHD1 variant but phenotype only if 4qA inherited",
        "gene_class": "Chromatin modifier (SMC family); maintains D4Z4 methylation; haploinsufficiency → D4Z4 hypomethylation → DUX4 derepression → muscle toxicity",
        "key_alerts": [
            "SMCHD1-FSHD2-DIGENIC-4qA-HAPLOTYPE-MANDATORY: SMCHD1 variant alone does NOT confirm FSHD2 — permissive 4qA haplotype on chromosome 4 is required; D4Z4 methylation assay should be <18% (hypomethylated); genetic test must include SMCHD1 sequencing + 4qA haplotyping + D4Z4 methylation; incomplete testing = missed or false-positive diagnosis",
            "SMCHD1-ASYMMETRY-PATHOGNOMONIC: dramatic asymmetry of weakness (one limb/side much weaker than contralateral) is PATHOGNOMONIC for FSHD1 and FSHD2; symmetric facial/limb weakness → reconsider FSHD diagnosis",
            "SMCHD1-LOSMAPIMOD-PHASE3-TRIAL: losmapimod (p38α/β MAPK inhibitor) — ReDUX4 phase 3 trial ongoing; rationale: p38 MAPK promotes DUX4 expression; early trials showed DUX4 target gene suppression; no FDA approval yet; refer eligible patients to trial",
            "SMCHD1-VARIABLE-EXPRESSIVITY-EXTREME: same SMCHD1 variant can cause severe early-onset disease in one family member and completely asymptomatic carrier in another; penetrance ~95% but severity 1-100 on scale; genetic counselling must explicitly address this unpredictability",
            "SMCHD1-COATS-DISEASE-RETINAL-SCREENING: retinal vasculopathy (Coats-like exudative retinopathy) in ~1% overall FSHD but ~75% of severely affected early-onset cases; ophthalmology referral at diagnosis + every 2 years",
            "SMCHD1-SNHL-AUDIOMETRY-MANDATORY: sensorineural hearing loss in majority of early-onset and severe FSHD; audiometry at diagnosis + every 2-3 years; cochlear implants effective if severe",
            "SMCHD1-BEEVOR-SIGN-ABDOMINAL-WEAKNESS: Beevor sign (umbilicus migrates cephalad on neck flexion) = abdominal weakness; nearly pathognomonic for FSHD; distinguish from normal umbilicus at rest",
            "SMCHD1-SCAPULAR-FIXATION-SURGERY: scapulothoracic arthrodesis improves shoulder range and function in severe scapular winging; best results when deltoid strength preserved; discuss early before deltoid loss",
        ],
        "etiologies": {
            "SMCHD1_LOF_truncating": {"pct": 60, "phenotype": "moderate_severe", "notes": "frameshift/nonsense → null; complete SMCHD1 haploinsufficiency"},
            "SMCHD1_missense_hypomorphic": {"pct": 30, "phenotype": "mild_moderate", "notes": "partial loss of chromatin binding; milder D4Z4 hypomethylation"},
            "SMCHD1_splicing": {"pct": 10, "phenotype": "variable", "notes": "splice site; partial LOF; phenotype correlates with residual protein"},
        },
        "stats": {
            "prevalence_FSHD_overall": "1 in 8,300",
            "FSHD2_fraction_pct": 2,
            "mean_dx_age_y": 26,
            "wheelchair_by_60y_pct": 20,
            "retinal_vasculopathy_early_onset_pct": 75,
            "snhl_pct": 70,
        },
        "dx_delay_distribution": {"mean_months": 60, "median_months": 48, "range": "12-240", "notes": "FSHD2 delays longer than FSHD1 (no contraction on standard array); requires specialist lab for methylation assay; asymmetric onset confuses presentation"},
    },

    # ── EMD — EDMD1 Emery-Dreifuss ────────────────────────────────────────────
    {
        "gene": "EMD",
        "protein": "EMD — Xq28 XLR — Emerin-254aa — EDMD1-Emery-Dreifuss — Early-Contractures-BEFORE-Weakness-PATHOGNOMONIC — ICD-Mandatory-Lethal-Arrhythmia — Female-Carriers-Cardiac",
        "alias": (
            "EMD (emerin); OMIM gene 300384; "
            "Emery-Dreifuss Muscular Dystrophy type 1 (EDMD1) OMIM 310300. "
            "Xq28; 254 aa; ~29 kDa; X-linked recessive. "
            "FUNCTION: Emerin is an inner nuclear membrane protein that localises to the nuclear envelope. "
            "It is part of the LInker of Nucleoskeleton and Cytoskeleton (LINC complex) connecting "
            "nuclear lamins (including lamin A/C) to the cytoskeleton. "
            "Emerin regulates transcription factor access, nuclear mechanics, and mechanotransduction. "
            "EMD LOF → disrupted nuclear envelope integrity → aberrant mechanosensing → "
            "cardiac (arrhythmia, DCM) and muscle (contractures, weakness) phenotype. "
            "CLINICAL TRIAD — ALL THREE REQUIRED FOR EDMD DIAGNOSIS: "
            "(1) Early joint contractures (Achilles, elbow, spine) — BEFORE significant weakness; "
            "(2) Humeral-peroneal distribution weakness (biceps, triceps, tibialis anterior, peroneals); "
            "(3) Life-threatening cardiac arrhythmia and DCM. "
            "CONTRACTURES BEFORE WEAKNESS — PATHOGNOMONIC: "
            "Joint contractures typically appear in childhood BEFORE marked weakness — this temporal sequence "
            "distinguishes EDMD from other MD forms where contractures follow weakness; "
            "elbow flexion contractures; "
            "Achilles tendon contracture (toe-walking from early age); "
            "neck/spine rigidity ('rigid spine'); "
            "CARDIAC — THE LETHAL COMPONENT: "
            "Cardiac arrhythmia is the primary cause of death/morbidity in EDMD1; "
            "arrhythmias: atrial standstill (characteristic), AF, atrial flutter, AV block, VT/VF; "
            "sudden cardiac death WITHOUT prior palpitations or syncope — can be first manifestation; "
            "DCM in a subset; "
            "ICD MANDATORY for all affected males (even if mild muscle disease); "
            "by age 20-25 in most EDMD1 males; "
            "FEMALE CARRIERS: ~30% of EMD carrier females develop cardiac arrhythmia/DCM "
            "without muscle disease — cardiac surveillance and possible ICD in carriers; "
            "emerin mosaicism on IHC (immunofluorescence of blood cells) — carrier testing; "
            "DIAGNOSIS: "
            "Emerin IHC on muscle biopsy or blood cells (white cells/buccal cells) shows absent/reduced emerin; "
            "EMD sequencing (whole gene, including deletions); "
            "MANAGEMENT: "
            "Annual ECG, 24h Holter, echo; "
            "ICD implantation (pacemaker insufficient — need defibrillation capability); "
            "physiotherapy for contractures; "
            "heart transplantation for end-stage DCM."
        ),
        "locus": "Xq28",
        "aa": 254,
        "kDa": 29,
        "omim_gene": "300384",
        "omim_disease": "310300",
        "inheritance": "XLR — X-linked Recessive — males fully affected; female carriers ~30% cardiac disease; carrier testing by emerin IHC on peripheral blood cells; female carriers need cardiac surveillance",
        "gene_class": "Inner nuclear membrane protein (LEM domain family); part of LINC complex; connects lamin A/C to cytoskeleton; regulates mechanotransduction and nuclear envelope integrity",
        "key_alerts": [
            "EMD-ICD-MANDATORY-LETHAL-ARRHYTHMIA: ICD MANDATORY for all EDMD1 males — atrial standstill, AV block, VT/VF can cause sudden death without warning; pacemaker alone INSUFFICIENT — must be ICD (defibrillation capability); implant by age 20-25 even with mild muscle disease",
            "EMD-CONTRACTURES-BEFORE-WEAKNESS-PATHOGNOMONIC: joint contractures (elbow, Achilles, spine) appearing BEFORE significant weakness is the PATHOGNOMONIC temporal sequence of EDMD; weakness preceding contractures = consider alternative diagnosis",
            "EMD-FEMALE-CARRIERS-CARDIAC-SURVEILLANCE-30PCT: 30% of EMD female carriers develop cardiac arrhythmia/DCM WITHOUT muscle disease; all carrier females need annual ECG + echo; ICD if arrhythmia or LVEF reduction; emerin IHC on blood cells identifies carriers",
            "EMD-ATRIAL-STANDSTILL-CHARACTERISTIC: atrial standstill (atrial mechanical failure despite sinus rhythm or AF) is characteristic of EMD/LMNA EDMD; ventricular conduction maintained; sudden death risk high; look for absent P waves with narrow QRS",
            "EMD-EMERIN-IHC-DIAGNOSTIC: emerin immunohistochemistry on muscle biopsy or peripheral blood mononuclear cells (PBMC) or buccal swab — absent/granular emerin = EDMD1; partial/mosaic in carriers; fast, cheap, highly sensitive (avoids long sequencing wait)",
            "EMD-HUMERAL-PERONEAL-DISTRIBUTION: weakness in biceps/triceps (humeral) AND tibialis anterior/peroneals (peroneal) — distinctive distribution; spares deltoid early; distal tibialis anterior weakness = foot drop in EDMD (vs proximal-predominant in most MD)",
            "EMD-RIGID-SPINE-SCOLIOSIS: spinal rigidity and scoliosis from paraspinal contractures; may contribute to restrictive lung disease; regular FVC monitoring; refer orthopaedics if curve >30°",
            "EMD-EMERIN-LMNA-SAME-PHENOTYPE-DIFFERENT-GENE: EDMD1 (EMD, XLR) and EDMD2 (LMNA, AD) produce nearly identical EDMD triad; both require ICD; LMNA tends to cause MORE severe cardiac disease and DCM; EMD — cardiac arrhythmia dominant; overlap necessitates testing BOTH genes if one negative",
        ],
        "etiologies": {
            "EMD_deletion_large": {"pct": 40, "phenotype": "severe_classic_EDMD", "notes": "complete gene deletion or large deletion; absent emerin on IHC"},
            "EMD_nonsense_frameshift": {"pct": 35, "phenotype": "severe_classic_EDMD", "notes": "null allele; absent emerin on IHC"},
            "EMD_missense": {"pct": 20, "phenotype": "variable_milder", "notes": "missense at LEM domain or integral membrane region; may have residual emerin expression"},
            "EMD_splice": {"pct": 5, "phenotype": "variable", "notes": "partial LOF; emerin may be reduced not absent on IHC"},
        },
        "stats": {
            "prevalence": "1 in 100,000",
            "mean_contracture_onset_y": 5,
            "mean_weakness_onset_y": 8,
            "mean_cardiac_event_age_y": 22,
            "icd_implantation_pct_by_30y": 90,
            "female_carrier_cardiac_pct": 30,
        },
        "dx_delay_distribution": {"mean_months": 48, "median_months": 36, "range": "12-120", "notes": "contractures misattributed to joint laxity or juvenile arthritis; cardiac diagnosis sometimes before muscle diagnosis"},
    },

    # ── LMNA — EDMD2 / LMNA-CMD / LMNA-DCM ───────────────────────────────────
    {
        "gene": "LMNA",
        "protein": "LMNA — 1q22 AD — Lamin-A/C-664aa — MOST-LETHAL-MD — EDMD2-DCM-CMD1A — ICD-Regardless-LVEF-Padua-Score — Non-Missense-Highest-Risk — Mean-Cardiac-Age-36y",
        "alias": (
            "LMNA (lamin A/C); OMIM gene 150330; "
            "Emery-Dreifuss MD type 2 (EDMD2) OMIM 181350; LMNA-related DCM (CMD1A) OMIM 115200; "
            "congenital muscular dystrophy-LMNA (LMNA-CMD); progeroid syndromes (HGPS, MAD, Werner-like). "
            "1q22; 664 aa; ~74 kDa (lamin A) / ~65 kDa (lamin C, alternative splicing); autosomal dominant. "
            "FUNCTION: Lamin A and C are type V intermediate filaments forming the nuclear lamina — "
            "the structural scaffold of the inner nuclear membrane. "
            "Functions: nuclear shape/stiffness, chromatin organisation, gene expression regulation, "
            "DNA repair, mechanotransduction (LINC complex with SUN1/2, nesprin, emerin). "
            "LMNA mutations cause MOST LETHAL of all muscular dystrophies — cardiac disease dominates. "
            "LMNA CARDIAC PHENOTYPE: "
            "DCM (dilated cardiomyopathy) + arrhythmia (AF, VT/VF, AV block) → sudden cardiac death; "
            "MOST LETHAL LMNA MUTATIONS: non-missense (truncating/splicing) >> missense; "
            "PADUA RISK SCORE: ≥4 points → ICD regardless of LVEF; "
            "Padua score factors: non-missense variant (+2), LVEF <45% (+1), male sex (+1), "
            "NSVT on Holter (+1), AV block on ECG (+1); "
            "mean age of first major cardiac event: 36 years; "
            "AVOID: flecainide, propafenone (Ic antiarrhythmics) — worsen LMNA arrhythmia; "
            "amiodarone + ICD combination effective; "
            "heart transplantation for end-stage DCM (bridge with LVAD); "
            "LMNA MUSCULAR PHENOTYPE: "
            "Similar EDMD triad to EDMD1 (contractures + humeral-peroneal + cardiac); "
            "LMNA-CMD: congenital form — neonatal hypotonia, dropped head, spine rigidity from infancy, "
            "early respiratory involvement, relatively preserved cognitive function; "
            "LMNA STRIATED MUSCLE DISORDERS (spectrum): "
            "EDMD2 (classic triad); LMNA-DCM (cardiac dominant, minimal muscle); "
            "LMNA-CMD (congenital onset); LGMD1B (proximal-predominant); "
            "PROGEROID SYNDROMES: Hutchinson-Gilford progeria (HGPS, c.1824C>T p.Gly608Gly → "
            "splicing → progerin accumulation → premature ageing); Mandibuloacral dysplasia; Werner-like; "
            "DIAGNOSIS: LMNA sequencing (panel including large deletion analysis); "
            "cardiac MRI (late gadolinium enhancement mid-myocardial = LMNA-specific pattern); "
            "MANAGEMENT: ICD (Padua score); beta-blockers; ACE-i/ARB; "
            "physiotherapy for contractures; respiratory monitoring FVC; transplant evaluation."
        ),
        "locus": "1q22",
        "aa": 664,
        "kDa": 74,
        "omim_gene": "150330",
        "omim_disease": "181350",
        "inheritance": "AD — Autosomal Dominant — haploinsufficiency or dominant-negative; 50% transmission; de novo ~15% in CMD form; AR biallelic = severe multisystem laminopathy",
        "gene_class": "Nuclear lamina intermediate filament (type V); scaffolds inner nuclear membrane; LINC complex; chromatin organiser; mechanotransducer; mutations cause laminopathies spanning DCM to progeria",
        "key_alerts": [
            "LMNA-ICD-REGARDLESS-LVEF-PADUA-SCORE: LMNA ICD indication based on Padua Risk Score ≥4 — NOT just LVEF; non-missense variant = +2 points alone (often score ≥4 without other factors); do NOT wait for LVEF to drop before implanting ICD — sudden death can occur with normal LVEF",
            "LMNA-NON-MISSENSE-HIGHEST-RISK: truncating, frameshift, splice-site LMNA variants carry higher SCD risk than missense; all non-missense = Padua score +2 = near-certain ICD indication; variant classification CRITICAL — missense VUS interpretation affects ICD decision",
            "LMNA-MEAN-CARDIAC-EVENT-AGE-36Y: first major cardiac event (sustained VT, SCD, pacemaker for AV block) at mean age 36y (range 20-50); cardiac surveillance from diagnosis regardless of age; ICD implantation in 30s standard practice",
            "LMNA-FLECAINIDE-PROPAFENONE-AVOID: Class Ic antiarrhythmic agents (flecainide, propafenone) increase mortality in LMNA cardiomyopathy — avoid for rhythm control; amiodarone is safer choice for AF/VT; ICD primary prevention of SCD",
            "LMNA-CARDIAC-MRI-MIDWALL-LGE: late gadolinium enhancement on cardiac MRI shows mid-wall enhancement (not subendocardial/subepicardial) — distinctive LMNA pattern; LGE presence increases SCD risk; cardiac MRI recommended at diagnosis + every 2-3 years",
            "LMNA-CMD-DROPPED-HEAD-SPINE-RIGIDITY: congenital LMNA-CMD: neonatal hypotonia + dropped head syndrome (neck extensor weakness > flexor) + rigid spine + early respiratory insufficiency; FVC monitoring from infancy; NIV when FVC <60%; cardiac surveillance from diagnosis",
            "LMNA-HEART-TRANSPLANT-BRIDGE-LVAD: heart transplantation for LMNA end-stage DCM; LMNA patients transplant younger than other DCM (mean age ~40); LVAD bridge effective; post-transplant arrhythmia risk persists (VT storm) — maintain ICD until transplant",
            "LMNA-HGPS-SPLICING-VARIANT-PROGERIA: c.1824C>T (p.Gly608Gly) = silent splicing variant → activates cryptic splice site → 50aa internal deletion → progerin; HGPS = premature ageing, CAD, stroke by age 13; lonafarnib (FTI) approved FDA 2020 for HGPS; distinct from striated muscle LMNA",
        ],
        "etiologies": {
            "LMNA_nonsense_frameshift_truncating": {"pct": 35, "phenotype": "severe_cardiac_dominant", "notes": "Padua ≥4 guaranteed; ICD mandatory; early cardiac events"},
            "LMNA_missense_rod_domain": {"pct": 40, "phenotype": "variable_EDMD2_like", "notes": "R453W, R249W common; EDMD triad with cardiac; Padua score variable"},
            "LMNA_splicing": {"pct": 15, "phenotype": "variable_often_severe", "notes": "may produce truncated protein or exon skip; phenotype = truncating if LOF splice"},
            "LMNA_CMD_specific": {"pct": 7, "phenotype": "congenital_severe", "notes": "early onset rigid spine; R249W also causes CMD; de novo common"},
            "LMNA_HGPS_progeria": {"pct": 3, "phenotype": "progeroid_cardiovascular", "notes": "c.1824C>T; premature ageing; CAD/stroke; lonafarnib approved"},
        },
        "stats": {
            "prevalence": "1 in 50,000 (striated muscle laminopathy)",
            "mean_first_cardiac_event_age_y": 36,
            "sudden_death_without_ICD_pct": 45,
            "heart_transplant_rate_pct": 15,
            "de_novo_rate_pct": 15,
        },
        "dx_delay_distribution": {"mean_months": 42, "median_months": 30, "range": "6-120", "notes": "cardiac symptoms often precede muscle diagnosis; DCM may be labelled idiopathic for years before LMNA testing"},
    },

    # ── CAPN3 — LGMD-R1 / Calpainopathy ─────────────────────────────────────
    {
        "gene": "CAPN3",
        "protein": "CAPN3 — 15q15.1 AR — Calpain3-821aa — LGMD-R1-Most-Common-LGMD-30pct — Pelvifemoral-Pattern — NO-Cardiac — CK-5-80x-Normal-CK-POSSIBLE — No-Gene-Therapy-Approved",
        "alias": (
            "CAPN3 (calpain 3); OMIM gene 114240; "
            "LGMD-R1 / calpainopathy / LGMD2A OMIM 253600. "
            "15q15.1; 821 aa; ~94 kDa; autosomal recessive. "
            "FUNCTION: Calpain 3 (p94) is a muscle-specific calcium-activated cysteine protease. "
            "It is unique among calpains as it is skeletal-muscle-specific and interacts with titin. "
            "CAPN3 functions: sarcomere remodelling during exercise; titin-anchored protease; "
            "regulates IkBα (NF-kB pathway) → muscle regeneration; "
            "CAPN3 LOF → impaired sarcomere homeostasis → progressive muscle degeneration. "
            "CLINICAL FEATURES: "
            "Most common LGMD globally (~30% of all LGMD); "
            "onset: 8-30 years (highly variable); "
            "PELVIFEMORAL PATTERN: glutei, adductors, hip flexors, hamstrings affected first → "
            "proximal leg weakness → difficulty climbing stairs, rising from floor; "
            "trunk weakness: hyperlordosis, waddling gait; "
            "shoulder girdle involved later; "
            "NO CARDIAC INVOLVEMENT (major differentiating feature from sarcoglycanopathies and laminopathies); "
            "CK: elevated 5-80x ULN; "
            "NORMAL CK POSSIBLE IN CAPN3 — unique among LGMD; "
            "some CAPN3 mutations (especially p.Arg490Gln) cause 'pseudometabolic' LGMD with normal CK; "
            "normal CK does NOT exclude CAPN3 — genome/panel sequencing required; "
            "CONTRACTURES: uncommon (unlike EDMD/LMNA); "
            "CARDIAC: absent in CAPN3 (vs sarcoglycanopathies where cardiac may occur); "
            "MANAGEMENT: physio, hydrotherapy, orthotics, wheelchair; "
            "no approved disease-modifying treatment; "
            "gene therapy trials (AAV-mediated CAPN3 delivery) in early phase; "
            "GENETIC COMPLEXITY: CAPN3 has 24 exons; common mutations include: "
            "p.Arg490Gln (Mediterranean); del550-572 (Basque founder, 40% LGMD-R1 in Basques); "
            "compound heterozygous very common; in-trans confirmation required for compound het; "
            "Western BLOT CAPN3: absent or reduced CAPN3 on western blot of muscle biopsy; "
            "some missense maintain normal protein level but no enzymatic activity."
        ),
        "locus": "15q15.1",
        "aa": 821,
        "kDa": 94,
        "omim_gene": "114240",
        "omim_disease": "253600",
        "inheritance": "AR — Autosomal Recessive — biallelic LOF; compound heterozygous common; Basque founder del550-572; Mediterranean R490Q; parents: obligate heterozygous carriers",
        "gene_class": "Calcium-activated cysteine protease (calpain family); muscle-specific (p94 isoform); titin-associated; sarcomere remodelling; NF-kB regulation via IkBα cleavage",
        "key_alerts": [
            "CAPN3-MOST-COMMON-LGMD-GLOBALLY-30PCT: CAPN3 calpainopathy accounts for ~30% of all LGMD worldwide; first test in undiagnosed pelvifemoral MD; molecular diagnosis by sequencing (panel or WGS) + western blot for protein confirmation",
            "CAPN3-NORMAL-CK-POSSIBLE-DO-NOT-EXCLUDE: normal CK does NOT exclude CAPN3 LGMD-R1; especially p.Arg490Gln and some Mediterranean variants may have near-normal CK; sequencing required when CK borderline and pelvifemoral pattern; this is unique among LGMD",
            "CAPN3-NO-CARDIAC-MAJOR-DDx-FEATURE: ABSENCE of cardiac involvement distinguishes CAPN3 from dystrophinopathy, sarcoglycanopathy (alpha/beta/gamma), and laminopathy; routine echo NOT indicated in established CAPN3 LGMD-R1 (unless other cardiac risk factors); annual respiratory assessment from 30y",
            "CAPN3-BASQUE-FOUNDER-DEL550-572: deletion of exons 5-6 encoding aa 550-572 is the founder mutation accounting for ~40% of LGMD-R1 in Basque population; targeted deletion testing before full sequencing in appropriate ethnic background",
            "CAPN3-COMPOUND-HETEROZYGOUS-IN-TRANS-CONFIRMATION: compound heterozygous CAPN3 mutations require confirmation that variants are on DIFFERENT alleles (in trans); parental testing mandatory; both variants on same allele (cis) = carrier not affected",
            "CAPN3-WESTERN-BLOT-PROTEIN-CONFIRMATION: muscle biopsy CAPN3 western blot shows absent or reduced protein; normal western blot does NOT exclude CAPN3 (some missense = full protein, no enzyme activity); enzyme activity assay required if blot normal but sequencing positive",
            "CAPN3-PSEUDOMETABOLIC-EXERCISE-INTOLERANCE: some CAPN3 patients present with exercise intolerance, myalgia, elevated CK post-exercise — mimics metabolic myopathy; distinguish by sequencing; no myoglobinuria typically",
            "CAPN3-AAV-GENE-THERAPY-TRIAL: AAV-CAPN3 intramuscular and systemic delivery phase 1-2 trials underway; no approved therapy yet; refer to trial for ambulatory early-onset patients",
        ],
        "etiologies": {
            "CAPN3_del550572_Basque_founder": {"pct": 12, "phenotype": "moderate_pelvifemoral", "notes": "40% Basque LGMD-R1; del exons 5-6; reduced western blot"},
            "CAPN3_R490Q_Mediterranean": {"pct": 18, "phenotype": "variable_milder", "notes": "normal CK possible; western blot may be normal; enzymatic LOF"},
            "CAPN3_compound_het_other": {"pct": 55, "phenotype": "variable", "notes": "worldwide; compound heterozygous most common presentation; confirm in trans"},
            "CAPN3_homozygous_rare": {"pct": 10, "phenotype": "severe_early_onset", "notes": "consanguineous; early-onset severe; absent protein on western blot"},
            "CAPN3_AD_dominant_negative": {"pct": 5, "phenotype": "mild_adult_onset", "notes": "rare AD CAPN3; gain-of-function or dominant-negative; proximal weakness; distinct from AR"},
        },
        "stats": {
            "fraction_all_LGMD_pct": 30,
            "mean_dx_age_y": 19,
            "wheelchair_by_50y_pct": 40,
            "CK_normal_proportion_pct": 10,
            "cardiac_involvement_pct": 0,
        },
        "dx_delay_distribution": {"mean_months": 84, "median_months": 72, "range": "24-240", "notes": "longest delay of all hereditary MD (slowly progressive; pelvifemoral weakness initially attributed to deconditioning); normal CK subgroup often misdiagnosed as metabolic myopathy"},
    },

    # ── DYSF — LGMD-R2 / Miyoshi Myopathy ────────────────────────────────────
    {
        "gene": "DYSF",
        "protein": "DYSF — 2p13.2 AR — Dysferlin-2080aa — LGMD-R2-Miyoshi-Myopathy — CK-50-100x-ULN-PATHOGNOMONIC — STEROIDS-WORSEN-ABSOLUTE-CONTRAINDICATION — Distal-Posterior-Miyoshi-OR-Proximal-LGMD",
        "alias": (
            "DYSF (dysferlin); OMIM gene 603009; "
            "LGMD-R2 / dysferlinopathy / LGMD2B OMIM 253601; Miyoshi Myopathy OMIM 254130. "
            "2p13.2; 2080 aa; ~237 kDa; autosomal recessive. "
            "FUNCTION: Dysferlin is a calcium-activated membrane repair protein. "
            "It belongs to the ferlin family of vesicle fusion proteins. "
            "Dysferlin mediates plasma membrane resealing after contraction-induced micro-tears: "
            "Ca2+ influx at membrane lesion → dysferlin-positive vesicles fuse with sarcolemma → "
            "membrane patch forms → lesion resealed. "
            "DYSF LOF → membrane repair failure → chronic Ca2+ influx → fibre degeneration. "
            "CLINICAL FEATURES — TWO MAIN PRESENTATIONS: "
            "(A) Miyoshi Myopathy (MM): DISTAL posterior compartment → gastrocnemius and soleus affected FIRST; "
            "patient cannot rise on tiptoes (early sign); calf muscle atrophy; lower leg muscle wasting; "
            "later spreads to thigh, then proximal; "
            "(B) LGMD-R2 (LGMD2B): PROXIMAL pelvifemoral → glutei, hamstrings, hip flexors affected FIRST; "
            "mimics CAPN3 LGMD-R1; later distal spread; "
            "(C) Scapuloperoneal syndrome: rare variant; "
            "SAME gene, same mutations can cause EITHER pattern in different family members — unknown modifier; "
            "CK: MASSIVELY ELEVATED 50-100x ULN — PATHOGNOMONIC for dysferlinopathy; "
            "CK may reach 50,000-100,000 IU/L; "
            "CK elevated BEFORE symptoms — pre-symptomatic diagnosis possible; "
            "CK elevation may fluctuate; CK highest early in disease, may normalise as muscle atrophies; "
            "INFLAMMATORY APPEARANCE ON BIOPSY — CRITICAL: "
            "Muscle biopsy shows CD4+ T-cell inflammation mimicking polymyositis/dermatomyositis; "
            "STEROIDS PRESCRIBED IN ERROR in up to 20% of dysferlinopathy patients → WORSEN disease; "
            "STEROIDS ARE ABSOLUTE CONTRAINDICATION IN DYSFERLINOPATHY; "
            "re-biopsy shows muscle atrophy despite 'treatment'; genetic testing mandatory before steroids; "
            "DYSFERLIN FLOW CYTOMETRY: dysferlin expression on monocytes (peripheral blood) by flow cytometry — "
            "reduced/absent in dysferlinopathy; fast, non-invasive, sensitive; "
            "CARDIAC: absent in dysferlinopathy (major differentiator from dystrophinopathy); "
            "RESPIRATORY: late if at all; "
            "TREATMENT: no approved therapy; "
            "physiotherapy; avoid steroids; "
            "antisense, exon skipping trials ongoing; "
            "protein restoration via read-through or stabilisation investigated."
        ),
        "locus": "2p13.2",
        "aa": 2080,
        "kDa": 237,
        "omim_gene": "603009",
        "omim_disease": "253601",
        "inheritance": "AR — Autosomal Recessive — biallelic LOF; compound heterozygous common; founder mutations in specific ethnic groups (Libya/Israel/Spain/Japan); elevated CK in obligate heterozygous carriers sometimes",
        "gene_class": "Membrane repair protein (ferlin family); calcium-activated vesicle fusion; sarcolemmal resealing after micro-tears; C2 domains mediate Ca2+-dependent membrane binding; LOF = membrane repair failure",
        "key_alerts": [
            "DYSF-STEROIDS-ABSOLUTE-CONTRAINDICATION-WORSEN: STEROIDS (prednisolone, methylprednisolone) ABSOLUTELY CONTRAINDICATED in dysferlinopathy — accelerate muscle loss; inflammatory biopsy appearance mimics polymyositis but immune suppression HARMFUL; genetic testing mandatory BEFORE any immunosuppression in unexplained inflammatory myopathy",
            "DYSF-CK-50-100x-ULN-PATHOGNOMONIC: CK 50-100x ULN (up to 50,000-100,000 IU/L) is the HIGHEST CK of any LGMD and is PATHOGNOMONIC for dysferlinopathy; CK >10,000 in a young ambulatory patient without trauma → DYSF sequencing first-line",
            "DYSF-INFLAMMATORY-BIOPSY-MIMICS-POLYMYOSITIS: CD4+ T-cell infiltrate on muscle biopsy mimics polymyositis in 20%+ of dysferlinopathy; do NOT start steroids without dysferlin flow cytometry or sequencing; re-biopsy after failed steroid treatment often shows advanced atrophy",
            "DYSF-DYSFERLIN-FLOW-CYTOMETRY-MONOCYTES: dysferlin expression on peripheral blood monocytes by flow cytometry — reduced/absent in DYSF biallelic LOF; fast (2-3 days), non-invasive, highly sensitive and specific; use as first-line test before muscle biopsy in suspected dysferlinopathy",
            "DYSF-MIYOSHI-TIPTOE-INABILITY-EARLY-SIGN: inability to rise on tiptoes (gastrocnemius-soleus weakness) in a young person with very high CK = Miyoshi Myopathy until proven otherwise; distinguish from Charcot-Marie-Tooth (CMT) by CK (CMT: normal CK) and nerve conduction (DYSF: normal NCS)",
            "DYSF-SAME-GENE-TWO-PHENOTYPES-MODIFIER: Miyoshi distal pattern vs LGMD-R2 proximal pattern can occur in DIFFERENT family members with IDENTICAL DYSF mutations; phenotype cannot be predicted from genotype; modifier genes unknown; same patient may transition from one pattern to other",
            "DYSF-NO-CARDIAC-KEY-DDx: complete absence of cardiac involvement distinguishes dysferlinopathy from dystrophinopathy, sarcoglycanopathy, and laminopathy; if cardiac involvement present → reconsider DYSF diagnosis or check for second condition",
            "DYSF-PRE-SYMPTOMATIC-HIGH-CK: dysferlin protein functions from birth; CK elevated before muscle weakness clinically apparent; pre-symptomatic relatives of DYSF patients (normal CK rules out biallelic LOF carrier)",
        ],
        "etiologies": {
            "DYSF_compound_het_worldwide": {"pct": 55, "phenotype": "variable_LGMD2B_or_Miyoshi", "notes": "most common presentation worldwide; confirm both variants in trans"},
            "DYSF_Libyan_founder_del_exon32": {"pct": 10, "phenotype": "LGMD2B", "notes": "founder deletion; North African Jewish; proximal predominant"},
            "DYSF_Spanish_c.5979dupA": {"pct": 8, "phenotype": "Miyoshi_distal", "notes": "Spanish/Portuguese founder; distal posterior compartment"},
            "DYSF_homozygous_Japanese": {"pct": 7, "phenotype": "Miyoshi_classic", "notes": "Japanese founder R1905X; severe Miyoshi; severe CK elevation"},
            "DYSF_large_deletion": {"pct": 10, "phenotype": "severe_early", "notes": "whole gene or large exon deletion; absent dysferlin; early onset"},
            "DYSF_missense_partial_LOF": {"pct": 10, "phenotype": "milder_variable", "notes": "reduced not absent dysferlin; later onset; slower progression"},
        },
        "stats": {
            "prevalence": "1 in 100,000-200,000",
            "CK_50x_ULN_pct": 90,
            "misdiagnosed_polymyositis_pct": 20,
            "steroids_given_erroneously_pct": 15,
            "cardiac_involvement_pct": 0,
            "mean_dx_age_y": 22,
        },
        "dx_delay_distribution": {"mean_months": 54, "median_months": 42, "range": "6-180", "notes": "inflammatory biopsy leads to polymyositis diagnosis; steroid trial delays correct diagnosis; delay longer in Miyoshi (attributed to sports injury, compartment syndrome)"},
    },

    # ── GNE — GNE Myopathy / IBM2 ────────────────────────────────────────────
    {
        "gene": "GNE",
        "protein": "GNE — 9p13.3 AR — GNE-722aa-Bifunctional-Sialic-Acid-Enzyme — GNE-Myopathy-IBM2-Nonaka — QUADRICEPS-SPARED-PATHOGNOMONIC — Rimmed-Vacuoles-Biopsy — NeuAc-Sialic-Acid-Supplementation-Trial",
        "alias": (
            "GNE (UDP-GlcNAc 2-epimerase/N-acetylmannosamine kinase); OMIM gene 603824; "
            "GNE myopathy / Hereditary Inclusion Body Myopathy (HIBM/IBM2) / Nonaka myopathy OMIM 605820. "
            "9p13.3; 722 aa; ~79 kDa; autosomal recessive. "
            "FUNCTION: GNE is a bifunctional enzyme essential for sialic acid (N-acetylneuraminic acid, NeuAc) biosynthesis. "
            "Domain 1 (aa 1-406): UDP-GlcNAc 2-epimerase → converts UDP-GlcNAc to ManNAc; "
            "Domain 2 (aa 420-722): N-acetylmannosamine kinase (ManNAc kinase) → phosphorylates ManNAc to ManNAc-6-P; "
            "These are the first two committed steps of sialic acid biosynthesis. "
            "GNE LOF → reduced sialic acid production → hyposialylation of muscle glycoproteins → "
            "muscle degeneration via unclear mechanism (alpha-dystroglycan hyposialylation hypothesised). "
            "CLINICAL FEATURES: "
            "Onset: 20-40 years (adult-onset distal myopathy); "
            "DISTAL ANTERIOR COMPARTMENT: tibialis anterior, extensor digitorum longus affected first → "
            "foot drop → steppage gait; "
            "QUADRICEPS SPARED UNTIL VERY LATE — PATHOGNOMONIC: "
            "Quadriceps muscle preserved until advanced disease (often until wheelchair); "
            "the combination of tibialis anterior weakness (foot drop) with INTACT quadriceps strength "
            "distinguishes GNE myopathy from CMT, peroneal muscular atrophy, and other distal myopathies; "
            "RIMMED VACUOLES on muscle biopsy — HALLMARK: "
            "Modified Gomori trichrome stain shows rimmed (red-outlined) vacuoles; "
            "p62-positive inclusions; TDP-43 inclusions; amyloid deposits (Congo red positive); "
            "biopsy appearance similar to sporadic IBM (sIBM) — crucial DDx; "
            "CK: normal to mildly elevated (2-10x ULN) — very low CK for a dystrophy; "
            "ETHNIC FOUNDER MUTATIONS: "
            "Middle Eastern Jewish (Iranian/Iraqi Jewish): M712T/V727M compound heterozygous (>95% of MENA Jewish GNE); "
            "Japanese: V572L/A524V; "
            "Korean, European (various mutations); "
            "TREATMENT — NO APPROVED THERAPY: "
            "NeuAc (N-acetylneuraminic acid, sialic acid) oral supplementation trials: "
            "EPICEPT (ManNAc) phase 2; NeuAc (Ultragenyx/Sialix) phase 3 GRACE trial negative (2021); "
            "Aceneuramic acid (Ace-ER, ACENEURAMIC ACID) — FDA orphan designation; "
            "GRACE trial did not meet primary endpoint (walking ability); no approved treatment; "
            "investigational: intravenous ManNAc; "
            "symptomatic: AFO for foot drop, walking aids; "
            "PROGNOSIS: slowly progressive; most reach wheelchair 20-30 years after onset; "
            "upper limb + pharyngeal weakness later; relatively preserved proximal leg strength until late; "
            "cardiac usually spared; respiratory rarely involved."
        ),
        "locus": "9p13.3",
        "aa": 722,
        "kDa": 79,
        "omim_gene": "603824",
        "omim_disease": "605820",
        "inheritance": "AR — Autosomal Recessive — biallelic; founder effects in Middle Eastern Jewish and Japanese populations; compound heterozygous M712T/V727M in Iranian/Iraqi Jewish patients",
        "gene_class": "Bifunctional enzyme (epimerase + kinase domains); rate-limiting for sialic acid biosynthesis; hyposialylation of muscle glycoproteins drives pathology; not a structural protein or ion channel",
        "key_alerts": [
            "GNE-QUADRICEPS-SPARED-PATHOGNOMONIC: quadriceps muscle (vastus lateralis, rectus femoris) preserved until very late disease — PATHOGNOMONIC sign of GNE myopathy; foot drop with INTACT knee extension in young adult = GNE myopathy first-line investigation; if quadriceps weak early → reconsider diagnosis",
            "GNE-RIMMED-VACUOLES-BIOPSY-HALLMARK: modified Gomori trichrome showing rimmed vacuoles with p62, TDP-43, amyloid deposits = GNE myopathy; distinguishes from DYSF (inflammatory biopsy) and CAPN3 (necrosis-regeneration without vacuoles); sIBM (sporadic IBM) has same biopsy but age >50 and ENMC criteria",
            "GNE-CK-NORMAL-TO-MILDLY-ELEVATED: CK only 1-10x ULN in GNE myopathy (very low for a dystrophy); normal or near-normal CK does NOT exclude significant muscle disease; do NOT use CK to stratify severity; use functional assessment + imaging",
            "GNE-MIDDLE-EASTERN-JEWISH-FOUNDER-M712T-V727M: M712T/V727M compound heterozygous accounts for >95% of GNE myopathy in Iranian and Iraqi Jewish population; targeted mutation testing in appropriate ethnic background before full sequencing; partner carrier testing essential",
            "GNE-IBM2-VS-SPORADIC-IBM-DDx: GNE myopathy (IBM2) vs sporadic IBM (sIBM): GNE = biallelic mutation, onset <50y, foot drop dominant, quadriceps spared, NO response to immunosuppression; sIBM = no mutation, onset >50y, finger flexors + quadriceps BOTH weak, ENMC 2011 criteria; sequencing mandatory when rimmed vacuoles in <50y patient",
            "GNE-GRACE-TRIAL-NEGATIVE-NO-APPROVED-RX: GRACE phase 3 trial of aceneuramic acid (Ace-ER) failed to meet primary endpoint (2021); no FDA-approved disease-modifying treatment; symptomatic care + AFO + walking aids; refer to trial if ongoing; NeuAc oral supplementation investigational only",
            "GNE-RESPIRATORY-CARDIAC-USUALLY-SPARED: respiratory and cardiac involvement absent or very late in GNE myopathy; FVC monitoring from wheelchair stage; annual ECG as general monitoring (not specific to GNE); pharyngeal weakness (dysphagia) may require PEG in advanced cases",
            "GNE-MRI-SELECTIVE-MUSCLE-INVOLVEMENT: MRI lower limb shows selective involvement: tibialis anterior, long finger extensors early; quadriceps spared on MRI even when clinically inaccessible; thigh MRI pattern (quadriceps spared) diagnostic when biopsy not available",
        ],
        "etiologies": {
            "GNE_M712T_V727M_MENA_Jewish": {"pct": 25, "phenotype": "classic_GNE_myopathy", "notes": "Iranian/Iraqi Jewish; epimerase + kinase domain doubly affected; most well-characterized cohort"},
            "GNE_V572L_Japanese": {"pct": 15, "phenotype": "classic_Nonaka", "notes": "Japanese/Korean; kinase domain; Nonaka distal myopathy; quad spared"},
            "GNE_compound_het_other": {"pct": 45, "phenotype": "variable_distal", "notes": "worldwide; epimerase + kinase domain combinations; variable severity"},
            "GNE_homozygous_non_founder": {"pct": 10, "phenotype": "moderate", "notes": "non-founder regions; consanguineous; range of severity"},
            "GNE_epimerase_domain_only": {"pct": 5, "phenotype": "mild_to_moderate", "notes": "some residual kinase activity; milder; slower progression"},
        },
        "stats": {
            "prevalence": "1 in 500,000-1,000,000 (ultra-rare)",
            "MENA_Jewish_prevalence": "1 in 1,500 Iranian/Iraqi Jewish",
            "mean_dx_age_y": 30,
            "mean_wheelchair_y_after_onset": 25,
            "CK_normal_pct": 50,
            "cardiac_involvement_pct": 0,
        },
        "dx_delay_distribution": {"mean_months": 96, "median_months": 84, "range": "24-360", "notes": "longest diagnostic delay; foot drop attributed to CMT or orthopaedic cause; rimmed vacuoles sometimes misread as sIBM; ethnic origin not always volunteered"},
    },
]


def _generate_patients():
    for idx, gene in enumerate(MD_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        patients = []
        g = gene["gene"]
        for i in range(40):
            # Gene-specific patient simulation
            if g == "DMD":
                duchenne = rng.random() > 0.25
                age_at_dx = rng.randint(3, 6) if duchenne else rng.randint(8, 25)
                dx_delay = rng.randint(6, 36) if duchenne else rng.randint(12, 60)
                calf_pseudohypertrophy = duchenne and rng.random() > 0.1
                cardiac_dcm = duchenne and rng.random() > 0.3
                exon_skip_eligible = rng.random() > 0.65
                on_steroids = duchenne and rng.random() > 0.2
                on_gene_therapy = duchenne and rng.random() > 0.6
                lost_ambulation = duchenne and age_at_dx > 5 and rng.random() > 0.55
                ck_x_uln = rng.randint(15000, 100000) if duchenne else rng.randint(500, 15000)
                variant_class = rng.choice(["frameshift_deletion", "nonsense", "duplication", "inframe_becker"])
                patients.append({
                    "patient_id": f"{g}-{i+1:03d}", "duchenne": duchenne,
                    "age_at_dx": age_at_dx, "dx_delay_months": dx_delay,
                    "calf_pseudohypertrophy": calf_pseudohypertrophy,
                    "cardiac_dcm": cardiac_dcm, "exon_skip_eligible": exon_skip_eligible,
                    "on_steroids": on_steroids, "on_gene_therapy": on_gene_therapy,
                    "lost_ambulation": lost_ambulation, "ck_x_uln": ck_x_uln,
                    "variant_class": variant_class, "gene": g, "seed": seed,
                })

            elif g == "DMPK":
                ctg_class = rng.choice(["mild_50_99", "classic_100_999", "classic_100_999", "severe_1000plus", "CDM_1000plus"])
                onset_age = rng.randint(3, 12) if ctg_class == "CDM_1000plus" else \
                            rng.randint(10, 20) if ctg_class == "severe_1000plus" else \
                            rng.randint(25, 45) if ctg_class == "classic_100_999" else rng.randint(40, 65)
                pacemaker = rng.random() > 0.6
                cataracts = rng.random() > 0.2
                myotonia = rng.random() > 0.1
                on_mexiletine = myotonia and rng.random() > 0.45
                anaesthesia_event = rng.random() > 0.85
                anticipation_child = rng.random() > 0.5
                dx_delay = rng.randint(12, 120)
                ck_x_uln = rng.randint(1, 5)
                patients.append({
                    "patient_id": f"{g}-{i+1:03d}", "ctg_class": ctg_class,
                    "onset_age": onset_age, "dx_delay_months": dx_delay,
                    "pacemaker": pacemaker, "cataracts": cataracts,
                    "myotonia": myotonia, "on_mexiletine": on_mexiletine,
                    "anaesthesia_event": anaesthesia_event, "anticipation": anticipation_child,
                    "ck_x_uln": ck_x_uln, "gene": g, "seed": seed,
                })

            elif g == "SMCHD1":
                severity = rng.choice(["mild", "mild", "moderate", "severe"])
                onset_age = rng.randint(15, 50) if severity == "mild" else \
                            rng.randint(8, 30) if severity == "moderate" else rng.randint(2, 15)
                asymmetry = rng.random() > 0.05
                scapular_winging = rng.random() > 0.3
                facial_weakness = rng.random() > 0.25
                retinal_vasculopathy = severity == "severe" and rng.random() > 0.25
                snhl = rng.random() > 0.4 if severity in ("moderate", "severe") else rng.random() > 0.8
                losmapimod_trial = rng.random() > 0.85
                permissive_4qA = True
                dx_delay = rng.randint(36, 240)
                ck_x_uln = rng.randint(1, 15)
                patients.append({
                    "patient_id": f"{g}-{i+1:03d}", "severity": severity,
                    "onset_age": onset_age, "dx_delay_months": dx_delay,
                    "asymmetry": asymmetry, "scapular_winging": scapular_winging,
                    "facial_weakness": facial_weakness, "retinal_vasculopathy": retinal_vasculopathy,
                    "snhl": snhl, "losmapimod_trial": losmapimod_trial,
                    "permissive_4qA": permissive_4qA, "ck_x_uln": ck_x_uln,
                    "gene": g, "seed": seed,
                })

            elif g == "EMD":
                contractures_age = rng.randint(3, 10)
                weakness_age = rng.randint(8, 18)
                icd_implanted = rng.random() > 0.15
                arrhythmia_type = rng.choice(["atrial_standstill", "AF", "AV_block_2nd", "VT", "AF"])
                emerin_absent_ihc = rng.random() > 0.05
                female_carrier = rng.random() > 0.8
                carrier_cardiac = female_carrier and rng.random() > 0.65
                dx_delay = rng.randint(12, 120)
                ck_x_uln = rng.randint(3, 20)
                patients.append({
                    "patient_id": f"{g}-{i+1:03d}",
                    "contractures_onset_age": contractures_age,
                    "weakness_onset_age": weakness_age,
                    "icd_implanted": icd_implanted, "arrhythmia_type": arrhythmia_type,
                    "emerin_absent_ihc": emerin_absent_ihc,
                    "female_carrier": female_carrier, "carrier_cardiac": carrier_cardiac,
                    "dx_delay_months": dx_delay, "ck_x_uln": ck_x_uln,
                    "gene": g, "seed": seed,
                })

            elif g == "LMNA":
                variant_class = rng.choice(["truncating", "missense", "splicing", "CMD"])
                onset_age = rng.randint(0, 5) if variant_class == "CMD" else \
                            rng.randint(20, 45) if variant_class in ("truncating", "splicing") else rng.randint(25, 50)
                padua_score = rng.randint(4, 7) if variant_class in ("truncating", "splicing") else rng.randint(2, 5)
                icd_implanted = padua_score >= 4
                cardiac_event_age = rng.randint(25, 50)
                dcm = rng.random() > 0.35
                cardiac_transplant = dcm and rng.random() > 0.8
                flecainide_avoided = rng.random() > 0.1
                lge_mri = dcm and rng.random() > 0.3
                dx_delay = rng.randint(12, 120)
                ck_x_uln = rng.randint(2, 20)
                patients.append({
                    "patient_id": f"{g}-{i+1:03d}", "variant_class": variant_class,
                    "onset_age": onset_age, "dx_delay_months": dx_delay,
                    "padua_score": padua_score, "icd_implanted": icd_implanted,
                    "cardiac_event_age": cardiac_event_age, "dcm": dcm,
                    "cardiac_transplant": cardiac_transplant, "flecainide_avoided": flecainide_avoided,
                    "lge_mri": lge_mri, "ck_x_uln": ck_x_uln,
                    "gene": g, "seed": seed,
                })

            elif g == "CAPN3":
                founder = rng.choice(["del550572_Basque", "R490Q_Mediterranean", "other", "other", "other"])
                onset_age = rng.randint(8, 30)
                ck_x_uln = rng.randint(5, 80) if founder != "R490Q_Mediterranean" else rng.randint(1, 20)
                normal_ck = ck_x_uln < 3
                pelvifemoral = rng.random() > 0.1
                wheelchair = rng.random() > 0.6 and onset_age < 20
                cardiac = False
                compound_het = rng.random() > 0.4 and founder == "other"
                in_trans_confirmed = compound_het and rng.random() > 0.3
                western_blot_reduced = rng.random() > 0.15
                dx_delay = rng.randint(24, 240)
                patients.append({
                    "patient_id": f"{g}-{i+1:03d}", "founder": founder,
                    "onset_age": onset_age, "dx_delay_months": dx_delay,
                    "ck_x_uln": ck_x_uln, "normal_ck": normal_ck,
                    "pelvifemoral": pelvifemoral, "wheelchair": wheelchair,
                    "cardiac": cardiac, "compound_het": compound_het,
                    "in_trans_confirmed": in_trans_confirmed,
                    "western_blot_reduced": western_blot_reduced,
                    "gene": g, "seed": seed,
                })

            elif g == "DYSF":
                phenotype = rng.choice(["Miyoshi", "LGMD-R2", "LGMD-R2", "mixed"])
                onset_age = rng.randint(15, 35)
                ck_x_uln = rng.randint(30, 100)
                misdiagnosed_polymyositis = rng.random() > 0.8
                steroids_given = misdiagnosed_polymyositis and rng.random() > 0.4
                dysferlin_flow_absent = rng.random() > 0.05
                cardiac = False
                tiptoe_inability = phenotype in ("Miyoshi", "mixed")
                biopsy_inflammatory = rng.random() > 0.4
                dx_delay = rng.randint(12, 180)
                patients.append({
                    "patient_id": f"{g}-{i+1:03d}", "phenotype": phenotype,
                    "onset_age": onset_age, "dx_delay_months": dx_delay,
                    "ck_x_uln": ck_x_uln,
                    "misdiagnosed_polymyositis": misdiagnosed_polymyositis,
                    "steroids_given": steroids_given,
                    "dysferlin_flow_absent": dysferlin_flow_absent,
                    "cardiac": cardiac, "tiptoe_inability": tiptoe_inability,
                    "biopsy_inflammatory": biopsy_inflammatory,
                    "gene": g, "seed": seed,
                })

            else:  # GNE
                ethnic_founder = rng.choice(["MENA_Jewish", "Japanese", "other", "other"])
                onset_age = rng.randint(20, 45)
                foot_drop = rng.random() > 0.05
                quadriceps_spared = rng.random() > 0.02
                rimmed_vacuoles_biopsy = rng.random() > 0.1
                ck_x_uln = rng.randint(1, 8)
                normal_ck = ck_x_uln <= 2
                misdiagnosed_sIBM = rng.random() > 0.75
                neuroac_trial = rng.random() > 0.85
                wheelchair = rng.random() > 0.65 and onset_age < 35
                cardiac = False
                dx_delay = rng.randint(48, 360)
                patients.append({
                    "patient_id": f"{g}-{i+1:03d}", "ethnic_founder": ethnic_founder,
                    "onset_age": onset_age, "dx_delay_months": dx_delay,
                    "foot_drop": foot_drop, "quadriceps_spared": quadriceps_spared,
                    "rimmed_vacuoles_biopsy": rimmed_vacuoles_biopsy,
                    "ck_x_uln": ck_x_uln, "normal_ck": normal_ck,
                    "misdiagnosed_sIBM": misdiagnosed_sIBM,
                    "neuroac_trial": neuroac_trial,
                    "wheelchair": wheelchair, "cardiac": cardiac,
                    "gene": g, "seed": seed,
                })
        gene["patients"] = patients


_generate_patients()


def get_overview():
    all_delays = [p.get("dx_delay_months", 0) for g in MD_GENES for p in g["patients"]]
    genes = []
    for idx, g in enumerate(MD_GENES):
        delays = [p.get("dx_delay_months", 0) for p in g["patients"]]
        genes.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
            "key_alerts": g["key_alerts"],
            "n_patients": len(g["patients"]),
        })
    return {
        "atlas": "Hereditary-Muscular-Dystrophy-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Muscular Dystrophy Atlas — "
            "DMD / DMPK / SMCHD1 / EMD / LMNA / CAPN3 / DYSF / GNE — 320 Patients (8×40, Seeds 1702–1709)"
        ),
        "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
        "total_patients": sum(len(g["patients"]) for g in MD_GENES),
        "aggregate_stats": {
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "genes_covered": len(MD_GENES),
            "patients_per_gene": 40,
        },
        "genes": genes,
        "top_alerts": [
            "DMD-ELEVIDYS-FDA2023-GENE-THERAPY: SRP-9001 (delandistrogene moxeparvovec) FDA-approved June 2023 (accelerated) / August 2024 (regular) for ambulatory DMD age 4-17; micro-dystrophin AAVrh74; confirm pathogenic DMD variant before referral; liver monitoring mandatory",
            "DMPK-ANAESTHESIA-EXTREME-RISK-SUXAMETHONIUM-ABSOLUTE-CI: depolarising NMB (suxamethonium) ABSOLUTE CI in DM1; volatile agents worsen post-op weakness; prefer spinal/TIVA; neostigmine contraindicated; cardiac defibrillator available; post-op ICU monitoring",
            "EMD-LMNA-ICD-MANDATORY-LETHAL-ARRHYTHMIA: ICD mandatory in all EDMD1 (EMD) males and all LMNA striated-muscle-laminopathy patients with Padua score ≥4; sudden cardiac death without warning in EDMD; pacemaker ALONE is insufficient",
            "DYSF-STEROIDS-ABSOLUTE-CI-WORSEN-DISEASE: steroids ABSOLUTELY contraindicated in dysferlinopathy — inflammatory biopsy mimics polymyositis but steroid treatment accelerates muscle loss; genetic/dysferlin flow testing mandatory before any immunosuppression in myopathy",
            "GNE-QUADRICEPS-SPARED-PATHOGNOMONIC: quadriceps preserved despite significant tibialis anterior (foot drop) weakness = GNE myopathy PATHOGNOMONIC sign; if foot drop + intact knee extension + rimmed vacuoles biopsy + normal CK → GNE sequencing first-line",
            "CAPN3-NORMAL-CK-DOES-NOT-EXCLUDE: normal CK does NOT exclude CAPN3 LGMD-R1 (unique among LGMD); CK normal in p.Arg490Gln and some other variants; sequencing required despite normal CK in pelvifemoral pattern",
            "SMCHD1-FSHD2-DIGENIC-4qA-MANDATORY: SMCHD1 variant alone does NOT diagnose FSHD2; permissive 4qA haplotype + D4Z4 methylation assay REQUIRED; incomplete FSHD2 diagnosis without both components",
            "LMNA-NON-MISSENSE-HIGHEST-CARDIAC-RISK: truncating/frameshift/splice LMNA = Padua score +2 = near-certain ICD indication; non-missense LMNA requires early ICD regardless of LVEF or muscle severity; cardiac MRI mid-wall LGE confirms diagnosis",
        ],
    }


def get_breakdown():
    result = []
    for idx, g in enumerate(MD_GENES):
        delays = [p.get("dx_delay_months", 0) for p in g["patients"]]
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
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def get_definitions():
    return {
        "concepts": {
            "Muscular Dystrophy Classification — The 2021 ENMC Framework": (
                "Muscular dystrophies are hereditary progressive myopathies defined by primary muscle "
                "degeneration and necrosis (without primary neuropathy or metabolic defect). "
                "The 2021 ENMC reclassification organises LGMD by protein function rather than number: "
                "LGMD-R (recessive) vs LGMD-D (dominant); numbered by year of gene discovery. "
                "LGMD-R1 = CAPN3 (calpainopathy); LGMD-R2 = DYSF (dysferlinopathy); "
                "LGMD-R3/4/5/6 = sarcoglycanopathies (SGCA/SGCB/SGCG/SGCD). "
                "KEY DIAGNOSTIC ALGORITHM: "
                "CK >10x + proximal weakness: dystrophinopathy panel (DMD) → if negative: LGMD panel; "
                "CK >30x + distal posterior: DYSF (Miyoshi); "
                "CK normal + distal anterior + quad spared: GNE myopathy; "
                "Rimmed vacuoles <50y: GNE → sIBM (age >50); "
                "Early contractures + humeral-peroneal + cardiac arrhythmia: EMD/LMNA; "
                "Facial + shoulder asymmetric + scapular wing: FSHD1/2 (D4Z4 count + SMCHD1); "
                "CTG repeat + myotonia + multisystem: DMPK DM1. "
                "Muscle MRI: invaluable for pattern recognition before biopsy. "
                "Next-generation sequencing panels covering >200 myopathy genes now standard."
            ),
            "Dystrophinopathy — The Reading-Frame Rule and Exon-Skipping Precision Medicine": (
                "The reading-frame rule predicts Duchenne vs Becker severity from DMD mutation: "
                "OUT-OF-FRAME mutations (disrupt codon reading frame) → no dystrophin → Duchenne (severe); "
                "IN-FRAME mutations (preserve reading frame) → truncated but semi-functional dystrophin → Becker (mild). "
                "Accuracy: ~90% (some exceptions due to NMD, alternative splicing, tissue-specific isoforms). "
                "EXON SKIPPING — PRECISION MEDICINE: "
                "Antisense oligonucleotides (AONs) skip specific exons → convert out-of-frame to in-frame → "
                "produce BMD-like truncated dystrophin; "
                "Exon 51 skip: eteplirsen (FDA 2016) — eligible ~13% DMD; "
                "Exon 53 skip: golodirsen (FDA 2019), viltolarsen (FDA 2020); "
                "Exon 45 skip: casimersen (FDA 2021); "
                "Exon 44 skip: under development; "
                "Limitation: each AON covers ~10-15% of DMD only; patient's exact exons REQUIRED; "
                "GENE THERAPY — ELEVIDYS: "
                "SRP-9001 (delandistrogene moxeparvovec, Elevidys): AAVrh74-MHCK7 promoter-micro-dystrophin; "
                "micro-dystrophin: engineered 138-kDa version (full = 427 kDa); "
                "FDA accelerated approval June 2023; regular approval August 2024 for 4-17y ambulatory; "
                "single IV infusion; micro-dystrophin expression in muscle; "
                "functional improvement measured by North Star Ambulatory Assessment (NSAA); "
                "ATALUREN: ribosome readthrough of premature stop codons (nonsense mutations); "
                "approved EU/UK (Translarna); not FDA-approved; eligible ~11% DMD."
            ),
            "DM1 RNA Toxicity — Why One Gene Causes Multisystem Disease": (
                "Myotonic Dystrophy Type 1 (DM1) is caused by CTG repeat expansion in the 3'UTR of DMPK, "
                "but the protein kinase itself is not the culprit. "
                "RNA GAIN-OF-FUNCTION mechanism: "
                "Expanded CUG-repeat RNA → folds into hairpin structure → "
                "sequesters MBNL1 and MBNL2 (muscleblind-like proteins) in nuclear foci. "
                "MBNL1/2 are alternative splicing regulators. Their sequestration → "
                "dysregulation of hundreds of splicing events across all tissues: "
                "CLCN1 (muscle chloride channel): mis-splicing → reduced Cl- conductance → membrane hyperexcitability → MYOTONIA; "
                "TNNT2 (troponin T2): foetal isoform in adult heart → DCM + conduction disease; "
                "INSR (insulin receptor): mis-spliced → insulin resistance → DM-like hyperglycaemia; "
                "BIN1: mis-splicing → T-tubule abnormalities → weakness; "
                "APP: mis-splicing → Tau hyperphosphorylation → CNS involvement; "
                "MULTISYSTEM PREDICTABLY FROM MBNL1 TARGETS: "
                "myotonia (CLCN1), weakness (BIN1), cardiac (TNNT2), cataracts (unknown), "
                "endocrine (INSR), GI (DMPK in smooth muscle), CNS (APP, tau). "
                "THERAPEUTIC TARGET: "
                "Antisense oligonucleotides targeting CUG repeats → release MBNL1 → correct splicing; "
                "IONIS Pharma DM1-AON trials; PRISM trial; no approved RNA-targeted therapy yet."
            ),
            "Laminopathy — Why LMNA Requires ICD Before LVEF Drops": (
                "LMNA mutations cause the most lethal form of MD due to primary cardiac electrical instability "
                "that is disproportionate to and often precedes DCM severity. "
                "MECHANISM: Lamin A/C mutations → altered nuclear mechanics + aberrant gene expression + "
                "disrupted desmosome/emerin interactions → cardiomyocyte electrical instability → "
                "VT/VF even with normal or mildly reduced LVEF. "
                "THE PADUA RISK SCORE: Validated risk score for SCD in LMNA: "
                "Non-missense variant: +2 points; "
                "LVEF <45%: +1; Male sex: +1; NSVT on Holter: +1; "
                "AV block on ECG (first to third degree): +1. "
                "Score ≥4 → STRONG ICD indication regardless of LVEF. "
                "Score 2-3 → Electrophysiology study + close monitoring. "
                "WHY NOT WAIT FOR LVEF TO DROP: "
                "In LMNA, 30-50% of SCD events occur when LVEF still >45%; "
                "arrhythmia substrate (fibrosis + lamin disruption) precedes pump failure; "
                "waiting for LVEF criterion (as in non-genetic DCM) = preventable death; "
                "ENMC/HRS/ESC guidelines: ICD for LMNA Padua ≥4, even with preserved EF. "
                "ANTIARRHYTHMIC CAUTION: "
                "Flecainide/propafenone (Class Ic): proarrhythmic in LMNA → avoid; "
                "Amiodarone: acceptable for AF rate control; "
                "Beta-blockers: standard DCM management; reduces VT burden."
            ),
            "DYSF Membrane Repair — Why Inflammatory Biopsy Does NOT Mean Polymyositis": (
                "Dysferlinopathy presents a major diagnostic trap: the muscle biopsy appears inflammatory, "
                "mimicking polymyositis (PM) in >20% of cases, yet STEROIDS ARE CONTRAINDICATED. "
                "MECHANISM OF INFLAMMATION IN DYSFERLINOPATHY: "
                "Dysferlin mediates membrane resealing via Ca2+-triggered vesicle fusion. "
                "DYSF LOF → failed membrane repair → chronic Ca2+ influx → fibre necrosis → "
                "macrophage + CD4+ T-cell infiltrate as secondary response to ongoing necrosis. "
                "This is REACTIVE inflammation (not autoimmune like PM/DM). "
                "Steroids suppress the reactive inflammation temporarily → "
                "patient appears to improve initially → steroid-induced myopathy superimposed + "
                "underlying DYSF necrosis continues → net WORSENING; "
                "WHY STEROIDS WORSEN: "
                "1. Glucocorticoids promote muscle protein catabolism; "
                "2. Suppress satellite cell regeneration; "
                "3. Dysferlin expression may be reduced by steroids; "
                "4. No autoimmune target to suppress in DYSF. "
                "DYSFERLIN FLOW CYTOMETRY — THE FAST SOLUTION: "
                "Dysferlin protein measurable on peripheral blood monocytes by flow cytometry; "
                "turnaround 2-3 business days; sensitivity/specificity >95%; "
                "absent/severely reduced dysferlin on monocytes + high CK + myopathy = dysferlinopathy; "
                "use BEFORE muscle biopsy and BEFORE any steroids in suspected inflammatory myopathy with CK >3000."
            ),
        },
        "pharmacological_distinctions": [
            "DMD — Elevidys (SRP-9001): single IV gene therapy; 4-17y ambulatory; confirm pathogenic DMD variant; liver enzymes monitored; corticosteroids continued alongside; immunosuppressive prep (prednisolone) pre-infusion; not yet indicated for non-ambulatory or adult",
            "DMD — Exon skipping: eteplirsen (exon 51, IV weekly), golodirsen/viltolarsen (exon 53, IV weekly), casimersen (exon 45, IV weekly); genotype-specific; renal monitoring; no cardiac benefit demonstrated",
            "DMD — Deflazacort preferred over prednisone: deflazacort (0.9 mg/kg/day) → similar efficacy, less weight gain vs prednisone (0.75 mg/kg/day daily); both delay ambulation loss ~2y; both cause cataract, bone loss, growth suppression — monitor",
            "DM1 — Mexiletine for myotonia: sodium channel blocker; 150-200mg TID; reduces grip myotonia and percussion myotonia; QTc monitoring mandatory (pro-arrhythmic if QT prolonged); carbamazepine/lamotrigine second-line; avoid quinine (cardiac risk)",
            "LMNA — ICD (Padua score): Padua ≥4 → ICD regardless of LVEF; amiodarone for AF/VT rate control; beta-blockers (carvedilol/bisoprolol) for DCM; AVOID flecainide/propafenone (Class Ic); ACE-i/ARB; LVAD bridge to transplant for end-stage",
            "EMD — ICD mandatory all males: implant by age 20-25; ICD not pacemaker (defibrillation capability required); female carriers: ICD if arrhythmia confirmed; physiotherapy for contractures (stretching Achilles, elbow, spine daily)",
            "DYSF — NO steroids ever: steroids accelerate muscle loss; only symptomatic management; physiotherapy; AFO for foot drop; walking frame/wheelchair; no approved disease-modifying therapy; avoid NSAIDs if renal concern (very rare)",
            "GNE — NeuAc supplementation (investigational): GRACE phase 3 trial of Ace-ER (aceneuramic acid extended release) failed 2021; oral NeuAc/ManNAc still investigated; no approved therapy; AFO for foot drop; PEG tube for dysphagia in advanced disease",
            "CAPN3 — Supportive care only: physiotherapy, hydrotherapy, resistance training (moderate); ankle-foot orthoses; wheelchair when needed; gene therapy (AAV-CAPN3) phase 1-2 trials; no approved disease-modifying treatment",
            "SMCHD1-FSHD2 — Losmapimod trial: p38 MAPK inhibitor; currently phase 3 (ReDUX4); no approved therapy; scapular fixation surgery for severe winging; AFO for foot drop; physiotherapy; avoid overhead arm activities",
        ],
        "key_standards": [
            "McDonald et al. 2018 (Neuromuscular Disorders) — LGMD Standard of Care: European MD consortium recommendations; genotype-first approach; gene panel sequencing standard; protein studies (IHC, western blot) as complement; muscle MRI pattern recognition; cardiac surveillance by gene",
            "Bushby et al. 2010 (Lancet Neurology) — DMD Care Recommendations: Benchmark DMD management standards; steroid start age, cardiac ACE-i at age 10, respiratory FVC monitoring, exon-skipping genetic eligibility determination",
            "Johnson et al. 2023 (NEJM) — Elevidys (SRP-9001) Phase 3 EMBARK Trial: primary endpoint (NSAA) not met at 52 weeks in full cohort; secondary functional endpoints improved in 4-7y age group; FDA granted regular approval August 2024; ongoing debate about benefit magnitude",
            "Wahbi et al. 2019 (Circulation) — LMNA Padua Score Validation: multicenter LMNA cohort n=311; Padua score ≥4 = HR 8.16 for SCD; validated ICD indication beyond standard LVEF criteria; established clinical standard for LMNA ICD decision",
            "Turner et al. 2021 (Lancet Neurology) — FSHD Standards: FSHD1/2 diagnostic recommendations; genetic testing pathway (D4Z4 array + SMCHD1 sequencing + methylation assay + 4qA haplotyping); losmapimod trial design reference",
            "Bushby et al. 1995 (Neuromuscular Disorders) — EDMD Diagnostic Criteria: established EMD/LMNA triad (contractures + humeral-peroneal + cardiac); ICD recommendation predating modern trial data; foundational for EDMD surveillance",
            "Takahashi et al. 2021 (JAMA Neurology) — GRACE Trial (GNE Myopathy): phase 3 aceneuramic acid extended release; primary endpoint (modified Rankin motor scale) not met; secondary endpoints also negative; no approved therapy for GNE myopathy currently",
            "Magot et al. 2022 (ENMC) — DYSF Diagnostic Criteria: dysferlin flow cytometry validated as first-line test; biopsy Western blot as confirmatory; steroid contraindication explicitly stated; genetic testing gold standard for definitive diagnosis",
        ],
    }
