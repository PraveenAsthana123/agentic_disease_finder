#!/usr/bin/env python3
"""Hereditary-Skeletal-Muscle-Channelopathy-Myotonic-Disorder-Atlas -- Complete 8-Gene Reference
Hereditary channelopathies and myotonic disorders of skeletal muscle:
SCN4A (Nav1.4; HyperKPP/PMC/SCN4A-Myotonia/HypoPP2; carbonic-anhydrase-inhibitor/mexiletine),
CACNA1S (Cav1.1; HypoPP-type1-MOST-COMMON; MHS5; R528H/R1239H hotspots; volatile/succinylcholine ABSOLUTE CI),
KCNJ2 (Kir2.1; Andersen-Tawil-Syndrome ATS-LQT7; TRIAD PATHOGNOMONIC; bidirectional-VT PATHOGNOMONIC),
CLCN1 (ClC-1; Myotonia-Congenita Thomsen/Becker; warm-up-phenomenon PATHOGNOMONIC; mexiletine-first-line),
KCNJ18 (Kir2.6; Thyrotoxic-Periodic-Paralysis susceptibility; treat-thyrotoxicosis-FIRST; NOT-K+-alone),
RYR1 (RyR1; MHS1/CCD/MmD; volatile+succinylcholine ABSOLUTELY CI; dantrolene MANDATORY; IVCT/CHCT gold-standard),
DMPK (CTG-repeat; DM1; anticipation PATHOGNOMONIC; cardiac-conduction annual-ECG-Holter MANDATORY),
CNBP (CCTG-repeat; DM2-PROMM; proximal-weakness; NO-anticipation; pain-prominent).

SCN4A    (Sodium channel Nav1.4; 1836aa; 17q23.3; AD GOF/LOF;
          HyperKPP-type1 / Paramyotonia-Congenita / SCN4A-Myotonia / HypoPP-type2 / Normokalemic-PP;
          carbonic-anhydrase-inhibitor acetazolamide/mexiletine;
          seed SEED_BASE+0).
CACNA1S  (Calcium channel Cav1.1; 1873aa; 1q32.1; AD GOF;
          HypoPP-type1-MOST-COMMON; MHS5-Malignant-Hyperthermia-Susceptibility;
          R528H-R1239H-HOTSPOTS; volatile-anesthetics+succinylcholine ABSOLUTE CI without dantrolene;
          seed SEED_BASE+1).
KCNJ2    (Inward-rectifier Kir2.1; 427aa; 17q24.3; AD LOF;
          Andersen-Tawil-Syndrome-ATS-LQT7;
          TRIAD periodic-paralysis+LQT7+facial-dysmorphia PATHOGNOMONIC;
          bidirectional-VT PATHOGNOMONIC; avoid-class-I-antiarrhythmics;
          seed SEED_BASE+2).
CLCN1    (Chloride channel ClC-1; 988aa; 7q34; AD-Thomsen/AR-Becker;
          Myotonia-Congenita; warm-up-phenomenon PATHOGNOMONIC;
          lid-lag-percussion-myotonia; mexiletine-first-line;
          seed SEED_BASE+3).
KCNJ18   (Inward-rectifier Kir2.6; 445aa; 17p11.2; AD LOF;
          Thyrotoxic-Periodic-Paralysis susceptibility;
          treat-thyrotoxicosis-FIRST; propranolol-acute;
          NOT-hypokalemia-driven-DO-NOT-use-K+-alone;
          seed SEED_BASE+4).
RYR1     (Ryanodine receptor type 1; 5038aa; 19q13.2; AD-MHS1/AR-CCD-MmD;
          Malignant-Hyperthermia-Susceptibility-type1 + Central-Core-Disease;
          volatile-agents+succinylcholine ABSOLUTE CI; dantrolene-IV mandatory;
          IVCT/CHCT-GOLD-STANDARD;
          seed SEED_BASE+5).
DMPK     (Dystrophia myotonica protein kinase; 629aa; 19q13.32; AD CTG-expansion;
          Myotonic-Dystrophy-type-1-DM1; anticipation PATHOGNOMONIC;
          cardiac-conduction-defect MANDATORY annual-ECG-Holter;
          respiratory MANDATORY; cataracts; distal-weakness;
          seed SEED_BASE+6).
CNBP     (CCHC-type zinc finger nucleic acid binding protein; 19q13.11; AD CCTG-expansion;
          Myotonic-Dystrophy-type-2-DM2-PROMM; proximal-weakness contrast-DM1;
          NO-anticipation contrast-DM1; DM2-milder-cardiac; DM2-pain-prominent;
          seed SEED_BASE+7).
"""

import random
import math

SEED_BASE = 3734

GENES = ["SCN4A", "CACNA1S", "KCNJ2", "CLCN1", "KCNJ18", "RYR1", "DMPK", "CNBP"]

GENE_META = {
    "SCN4A": {
        "aa": 1836, "kda": 200, "locus": "17q23.3", "inh": "AD GOF/LOF",
        "syndrome": "HyperKPP-type1 / Paramyotonia-Congenita / SCN4A-Myotonia / HypoPP-type2 / Normokalemic-PP",
        "channel": "Nav1.4 — voltage-gated sodium channel alpha subunit",
        "pathognomonic": "Cold-triggered myotonia worsening with exercise (PMC) PATHOGNOMONIC; potassium-triggered flaccid paralysis (HyperKPP)",
        "key_rule": "Acetazolamide (carbonic anhydrase inhibitor) first-line for HyperKPP/PMC; mexiletine for disabling myotonia; avoid K+-containing IV fluids in paralytic attacks",
        "mean_age_dx": 14.2,
        "pct_female": 0.45,
        "attack_freq_per_year": 18.5,
        "dx_delay_years": 4.8,
        "severity_score": 6.2,
        "cardiac_involvement_pct": 0.05,
        "respiratory_pct": 0.08,
        "prevalence_note": "Most common cause of periodic paralysis overall; allelic conditions span HyperKPP to myotonia",
        "treatment": "Acetazolamide / mexiletine / avoid cold+exercise triggers",
        "trigger": "Potassium load / cold exposure / rest after exercise",
        "clinical_facts": [
            "SCN4A encodes Nav1.4 (200 kDa) — the dominant skeletal muscle voltage-gated Na channel; GOF mutations cause gain of inactivation failure (myotonia/HyperKPP/PMC), LOF mutations cause HypoPP-type2",
            "HyperKPP (GOF): episodic flaccid weakness triggered by high-K+ foods, rest after exercise, fasting; serum K+ ELEVATED or normal during attacks (normokalemic PP is allelic)",
            "Paramyotonia Congenita (PMC, GOF): cold-triggered myotonia that WORSENS with repeated exercise — opposite of classic myotonia congenita warm-up; PATHOGNOMONIC",
            "HypoPP-type2 (LOF gating-pore current): clinically identical to CACNA1S HypoPP-type1 but genotype distinguishes; acetazolamide effective for HypoPP-type2 also",
            "Acetazolamide (carbonic anhydrase inhibitor) is first-line for HyperKPP and PMC — reduces attack frequency and severity; dichlorphenamide is alternative",
            "Mexiletine (sodium channel blocker) controls disabling myotonia in SCN4A-myotonia and PMC; dose-escalate with ECG monitoring for QTc",
            "Potassium-containing IV fluids are CONTRAINDICATED during HyperKPP attack — worsen weakness; isotonic saline preferred; glucose may help",
            "Ice-water forearm immersion test: cold provokes myotonia / EMG after cold reproduces electrical myotonia in PMC — useful office test",
            "Allelic spectrum: HyperKPP (GOF I693T/T704M), PMC (GOF R1448C/H), SCN4A-myotonia (GOF I1160V), HypoPP-type2 (LOF R672G/H/S/C gating-pore mutations), normokalemic PP",
            "Neonatal episodic laryngospasm has been reported with SCN4A mutations — critical airway presentation in infants",
        ],
    },
    "CACNA1S": {
        "aa": 1873, "kda": 210, "locus": "1q32.1", "inh": "AD GOF",
        "syndrome": "HypoPP-type1-MOST-COMMON / MHS5-Malignant-Hyperthermia-Susceptibility-type5",
        "channel": "Cav1.1 — voltage-gated L-type calcium channel alpha1 subunit",
        "pathognomonic": "Hypokalemic paralysis + R528H or R1239H hotspot mutations; combined HypoPP1 + MH susceptibility in same patient",
        "key_rule": "Volatile anesthetics (halothane/sevoflurane/desflurane/isoflurane) + succinylcholine ABSOLUTE CONTRAINDICATION without dantrolene preloaded; TIVA mandatory",
        "mean_age_dx": 12.8,
        "pct_female": 0.48,
        "attack_freq_per_year": 14.2,
        "dx_delay_years": 5.1,
        "severity_score": 6.8,
        "cardiac_involvement_pct": 0.06,
        "respiratory_pct": 0.12,
        "prevalence_note": "Most common form of hypokalemic periodic paralysis (~60% of all HypoPP cases); MHS5 co-susceptibility in same patient",
        "treatment": "Oral KCl supplementation during attacks; acetazolamide prophylaxis; low-carbohydrate/salt diet",
        "trigger": "High-carbohydrate meals / insulin / exercise / cold / alcohol / stress",
        "clinical_facts": [
            "CACNA1S encodes Cav1.1 (210 kDa) — L-type voltage-gated calcium channel of the skeletal muscle dihydropyridine receptor (DHPR); critical for excitation-contraction coupling",
            "HypoPP-type1 is the MOST COMMON form of hypokalemic periodic paralysis (~60% of all familial HypoPP); R528H (exon 11) and R1239H (exon 30) are hotspot mutations accounting for >50% of cases",
            "Attacks: flaccid paralysis lasting hours to days; serum K+ drops to 1.5-2.5 mEq/L during attack; onset typically in the first or second decade; attacks decrease after age 30-40",
            "Cardiac arrhythmia risk during severe hypokalemia attacks — monitor ECG; U waves, prolonged QT, arrhythmia possible; admit for IV KCl if serum K+ <2.5 mEq/L",
            "MHS5 co-susceptibility: CACNA1S mutations confer MH susceptibility — volatile anesthetics (halothane, sevoflurane, desflurane, isoflurane) and succinylcholine are ABSOLUTE CONTRAINDICATIONS without dantrolene",
            "TIVA (total intravenous anesthesia with propofol + opioid) is MANDATORY for CACNA1S carriers — alert anesthesia team preoperatively",
            "Acetazolamide prophylaxis effective in HypoPP-type1 for reducing attack frequency; paradoxically may worsen HypoPP-type2 (KCNJ2 gating-pore mutations) — genetic testing distinguishes",
            "Progressive myopathy (proximal > distal weakness) develops in older patients; vacuolar myopathy on biopsy; correlates with cumulative attack frequency and severity",
            "Low-carbohydrate diet reduces attack frequency by minimizing post-meal insulin surge; avoid high-sugar meals, alcohol, strenuous exercise",
            "Thyrotoxic PP must be excluded in all new-onset HypoPP — check TSH/FT4; treat thyroid disease FIRST (KCNJ18 susceptibility gene)",
        ],
    },
    "KCNJ2": {
        "aa": 427, "kda": 48, "locus": "17q24.3", "inh": "AD LOF",
        "syndrome": "Andersen-Tawil-Syndrome-ATS / LQT7 / TRIAD-PATHOGNOMONIC",
        "channel": "Kir2.1 — inward-rectifier potassium channel",
        "pathognomonic": "TRIAD: episodic periodic paralysis + LQT7 with BIDIRECTIONAL VT PATHOGNOMONIC + facial/skeletal dysmorphia PATHOGNOMONIC for ATS",
        "key_rule": "Bidirectional ventricular tachycardia PATHOGNOMONIC for ATS-LQT7; avoid class-I antiarrhythmics (quinidine/flecainide) — worsen QT prolongation; ICD if sustained VT/VF",
        "mean_age_dx": 16.4,
        "pct_female": 0.52,
        "attack_freq_per_year": 8.4,
        "dx_delay_years": 7.2,
        "severity_score": 7.8,
        "cardiac_involvement_pct": 0.78,
        "respiratory_pct": 0.04,
        "prevalence_note": "Rare; ~1 in 1,000,000; 60% of ATS cases have KCNJ2 mutations; ~40% no mutation found",
        "treatment": "Oral KCl / beta-blockers / ICD for malignant arrhythmia; avoid class-I antiarrhythmics",
        "trigger": "Rest after exercise / fasting / emotional stress / carbohydrate meals",
        "clinical_facts": [
            "KCNJ2 encodes Kir2.1 (48 kDa) — the principal cardiac/skeletal muscle inward-rectifier K+ channel maintaining resting membrane potential; LOF mutations elevate resting potential and prolong cardiac action potential",
            "Andersen-Tawil Syndrome (ATS): CLASSIC TRIAD = periodic paralysis (normo/hypo/hyperkalemic) + LQT7 (prolonged QU interval) + facial/skeletal dysmorphia (low-set ears, micrognathia, syndactyly, scoliosis, short stature) — TRIAD PATHOGNOMONIC",
            "Bidirectional ventricular tachycardia (alternating QRS axis beat-to-beat) is PATHOGNOMONIC for ATS-KCNJ2 — a key distinguishing ECG finding from other LQT syndromes",
            "Cardiac risk in ATS is paradoxically lower than other LQT syndromes despite prominent QT prolongation — most VT is non-sustained, polymorphic; sudden cardiac death less common than LQT1/LQT2/LQT3",
            "Class-I antiarrhythmics (quinidine, flecainide, propafenone) MUST BE AVOIDED — block Kir2.1 directly and worsen QT prolongation; procainamide also contraindicated",
            "Beta-blockers are first-line for arrhythmia management; ICD indicated for documented sustained VT/VF or resuscitated cardiac arrest",
            "Potassium supplementation during attacks (serum K+ may be low, normal, or high — all variants occur in ATS); acetazolamide reduces attack frequency",
            "Dental/jaw anomalies (micrognathia, mandibular hypoplasia) require orthodontic/maxillofacial consultation; small hands/feet, clinodactyly are common dysmorphic features",
            "Only 60% of ATS cases have identifiable KCNJ2 mutations — diagnosis is clinical (triad-based) in mutation-negative cases; full gene sequencing including deletions required",
            "Heterozygous dominant-negative mechanism — mutant Kir2.1 incorporates into tetramers and suppresses wild-type Kir2.1 function (dominant-negative >haploinsufficiency)",
        ],
    },
    "CLCN1": {
        "aa": 988, "kda": 108, "locus": "7q34", "inh": "AD (Thomsen) / AR (Becker)",
        "syndrome": "Myotonia-Congenita-Thomsen-AD / Myotonia-Congenita-Becker-AR",
        "channel": "ClC-1 — voltage-gated chloride channel",
        "pathognomonic": "Warm-up phenomenon (myotonia improves with continued exercise) PATHOGNOMONIC for Becker MC; percussion myotonia; lid-lag on downgaze",
        "key_rule": "Mexiletine is first-line pharmacotherapy for disabling myotonia in CLCN1; warm-up differentiates MC from paramyotonia (PMC = cold-worsens); Becker AR is more severe than Thomsen AD",
        "mean_age_dx": 10.6,
        "pct_female": 0.50,
        "attack_freq_per_year": 365.0,  # daily stiffness events
        "dx_delay_years": 3.4,
        "severity_score": 5.4,
        "cardiac_involvement_pct": 0.02,
        "respiratory_pct": 0.03,
        "prevalence_note": "~1 in 25,000 (Becker AR more common in non-Scandinavian populations); Thomsen AD ~1 in 100,000",
        "treatment": "Mexiletine first-line; carbamazepine/phenytoin second-line; warm-up exercise; avoid sudden exertion",
        "trigger": "Rest after inactivity / cold / startle / sudden movement after rest",
        "clinical_facts": [
            "CLCN1 encodes ClC-1 (108 kDa) — the principal skeletal muscle voltage-gated chloride channel responsible for ~80% of resting membrane conductance; LOF causes membrane hyperexcitability and repetitive action potentials (myotonia)",
            "Warm-up phenomenon PATHOGNOMONIC for CLCN1 myotonia congenita (especially Becker AR): muscle stiffness severe after rest, dramatically improves with repeated exercise — exact opposite of paramyotonia congenita (SCN4A PMC where cold+exercise worsens myotonia)",
            "Thomsen disease (AD): milder, onset in infancy, minimal progression; Becker disease (AR, biallelic LOF): more severe, onset 4-12 years, transient weakness at onset of movement (may appear as \"paralysis\" lasting seconds), progressive",
            "Becker MC has a unique transient weakness feature at initiation of movement that CLEARS with continued exercise — can mimic periodic paralysis; EMG showing electrical myotonia confirms",
            "Percussion myotonia: tapping the thenar eminence or tongue causes sustained dimpling/contraction for 5-10 seconds — readily elicitable in clinic",
            "Lid-lag on rapid downgaze (von Graefe sign): myotonic lid lag on downgaze, upper eyelid remains elevated momentarily — present in most CLCN1 patients; useful bedside sign",
            "Mexiletine (sodium channel blocker) is first-line for disabling myotonia in CLCN1 — reduces electrical myotonia and clinical stiffness; monitor QTc; avoid in pre-existing cardiac conduction disease",
            "Mexiletine trial: 150 mg TID for 4 weeks; EMG myotonia score and Patient Global Impression before/after; effective in ~70-80% at therapeutic doses",
            "Carbamazepine and lamotrigine are alternatives when mexiletine contraindicated; phenytoin and acetazolamide also used historically but less effective",
            "No periodic paralysis in isolated CLCN1 myotonia congenita — if paralytic attacks present, consider SCN4A or KCNJ2 overlap; weakness in Becker is transient at initiation, not prolonged flaccid attacks",
        ],
    },
    "KCNJ18": {
        "aa": 445, "kda": 50, "locus": "17p11.2", "inh": "AD LOF",
        "syndrome": "Thyrotoxic-Periodic-Paralysis-Susceptibility / TPP",
        "channel": "Kir2.6 — inward-rectifier potassium channel (thyroid-hormone regulated)",
        "pathognomonic": "Hypokalemic periodic paralysis in setting of THYROTOXICOSIS — TREAT THYROID FIRST; paralysis resolves with euthyroid state",
        "key_rule": "Treat thyrotoxicosis FIRST (antithyroid drugs/radioiodine/surgery) — K+ supplementation alone is insufficient; propranolol for acute attack; DO NOT use K+ supplementation as sole long-term strategy",
        "mean_age_dx": 28.4,
        "pct_female": 0.20,
        "attack_freq_per_year": 9.8,
        "dx_delay_years": 2.1,
        "severity_score": 5.8,
        "cardiac_involvement_pct": 0.10,
        "respiratory_pct": 0.06,
        "prevalence_note": "TPP most common in Asian males (2% of hyperthyroid Asians); KCNJ18 LOF found in ~33% of TPP cases — susceptibility allele",
        "treatment": "Treat thyrotoxicosis (ATD/RAI/surgery); propranolol for acute attacks; K+ supplementation only adjunctive",
        "trigger": "Thyrotoxicosis + carbohydrate load / insulin / beta-agonists / strenuous exercise",
        "clinical_facts": [
            "KCNJ18 encodes Kir2.6 (50 kDa) — a thyroid-hormone-regulated inward-rectifier K+ channel; thyroid hormone (T3) upregulates KCNJ18 transcription; LOF mutations expose latent susceptibility in hyperthyroid state",
            "Thyrotoxic Periodic Paralysis (TPP): episodic flaccid paralysis with severe hypokalemia (K+ 1.0-2.5 mEq/L) occurring in overtly or subclinically thyrotoxic patients — clinical presentation IDENTICAL to familial HypoPP except biochemical hyperthyroidism",
            "TREAT THYROID DISEASE FIRST — restoration of euthyroid state with antithyroid drugs (carbimazole/methimazole/PTU), radioiodine, or thyroidectomy abolishes attacks; long-term K+ supplementation alone does not prevent attacks",
            "Propranolol (non-selective beta-blocker): reduces attack frequency acutely by blocking Na-K-ATPase stimulation and blocking beta2-receptor-mediated potassium shift into cells; use while awaiting euthyroid status",
            "DO NOT use K+ supplementation as sole management strategy — rebound hyperkalemia may occur as K+ redistributes post-attack; overcorrection risk is significant; only use for symptomatic correction of severe hypokalemia",
            "Strong male predominance (M:F ~20:1) in Asians with TPP despite thyrotoxicosis being more common in women — androgen-mediated Na-K-ATPase upregulation + thyroid-hormone synergism proposed mechanism",
            "ECG during attack: bradycardia, QU prolongation (misread as QT), U waves, ST depression — identical to HypoPP1/HypoPP2; cardiac arrhythmia during severe hypokalemia possible",
            "TSH + FT4 + FT3 MANDATORY in ALL new-onset hypokalemic periodic paralysis — TPP missed in up to 50% on first presentation because thyroid symptoms may be subtle",
            "Acetazolamide and dichlorphenamide used in familial HypoPP may be ineffective or harmful in KCNJ18-TPP — confirm thyroid etiology before using",
            "KCNJ18 variants found in ~33% of TPP patients — remaining cases may involve other Kir channels or regulatory variants; negative KCNJ18 sequencing does NOT exclude TPP if thyrotoxicosis present",
        ],
    },
    "RYR1": {
        "aa": 5038, "kda": 565, "locus": "19q13.2", "inh": "AD (MHS1) / AR (CCD/MmD)",
        "syndrome": "Malignant-Hyperthermia-Susceptibility-type1-MHS1 / Central-Core-Disease-CCD / Multiminicore-Myopathy-MmD",
        "channel": "RyR1 — ryanodine receptor type 1 (sarcoplasmic reticulum Ca2+ release channel)",
        "pathognomonic": "Malignant hyperthermia crisis: hyperthermia + rigidity + masseter spasm + rhabdomyolysis triggered by volatile anesthetics/succinylcholine PATHOGNOMONIC; dantrolene IV reverses crisis",
        "key_rule": "Volatile anesthetics (all halogenated agents) + succinylcholine ABSOLUTELY CONTRAINDICATED; dantrolene IV 2.5 mg/kg push MUST be immediately available at ALL anesthetic sites where RYR1-carrier may present; IVCT/CHCT GOLD STANDARD for susceptibility testing",
        "mean_age_dx": 22.6,
        "pct_female": 0.45,
        "attack_freq_per_year": 0.3,  # MH episodes (rare, anesthesia-triggered)
        "dx_delay_years": 8.4,
        "severity_score": 8.6,
        "cardiac_involvement_pct": 0.08,
        "respiratory_pct": 0.22,
        "prevalence_note": "MHS1 prevalence ~1 in 10,000-50,000; CCD ~1 in 200,000 (most have RYR1 mutations); AD for MHS, AR for CCD/MmD",
        "treatment": "Dantrolene IV + cooling + supportive; TIVA mandatory lifelong; alert bracelet",
        "trigger": "Volatile halogenated anesthetics (halothane/sevoflurane/desflurane/isoflurane/enflurane) / succinylcholine / (rarely) stress/extreme heat",
        "clinical_facts": [
            "RYR1 encodes the ryanodine receptor type 1 (565 kDa) — the giant SR Ca2+ release channel in skeletal muscle; the most mutation-rich gene in the human genome (>400 pathogenic variants); AD GOF mutations cause MHS1; biallelic AR LOF/hypomorphic mutations cause CCD/MmD",
            "Malignant Hyperthermia (MH): uncontrolled RyR1 Ca2+ release triggered by volatile anesthetics or succinylcholine → massive skeletal muscle hypermetabolism → hyperthermia (>39°C rising), masseter rigidity (jaw rigidity post-succinylcholine PATHOGNOMONIC), generalized rigidity, rhabdomyolysis, metabolic/respiratory acidosis, hyperkalemia, DIC",
            "Volatile anesthetics (halothane, sevoflurane, desflurane, isoflurane, enflurane) and depolarizing neuromuscular blockade (succinylcholine) are ABSOLUTELY CONTRAINDICATED in RYR1 carriers — safe anesthesia = TIVA (propofol + opioid + non-depolarizing NMB)",
            "Dantrolene IV: MUST be immediately available at EVERY anesthetic site; dose 2.5 mg/kg IV bolus push, repeat q5min until crisis resolves (max ~10 mg/kg); each vial 20 mg requires 60 mL sterile water for reconstitution — pre-reconstitution in crisis is critical",
            "IVCT/CHCT (In Vitro Contracture Test / Caffeine-Halothane Contracture Test): GOLD STANDARD for MH susceptibility diagnosis; muscle biopsy from vastus lateralis; caffeine and halothane dose-response; performed at only ~60 specialized centers worldwide",
            "Central Core Disease (CCD): AR biallelic RYR1 mutations; congenital hypotonia, proximal weakness, central cores on muscle biopsy (PATHOGNOMONIC); all CCD patients must be considered MH-susceptible regardless of C/C or C/H IVCT result",
            "Multiminicore Myopathy (MmD): biallelic RYR1 (and SEPN1); multiple cores on biopsy; axial and respiratory weakness more prominent than CCD; scoliosis common",
            "King-Denborough syndrome: MH susceptibility + myopathic facies + skeletal anomalies (short stature, kyphoscoliosis, cryptorchidism) — associated with RYR1 mutations",
            "MH Alert: patients/families MUST carry an MH alert card/bracelet; genetic testing of first-degree relatives mandatory (AD inheritance in MHS1 — 50% risk per offspring); MHAUS (Malignant Hyperthermia Association USA) hotline 1-800-644-9737",
            "Exercise-induced rhabdomyolysis (ExIR) and heat stroke susceptibility can occur in RYR1 carriers WITHOUT anesthesia — counsel on hydration, avoiding extreme heat, early CK monitoring after intense exercise",
        ],
    },
    "DMPK": {
        "aa": 629, "kda": 70, "locus": "19q13.32", "inh": "AD CTG-repeat expansion",
        "syndrome": "Myotonic-Dystrophy-type-1-DM1 / Steinert-Disease",
        "channel": "DMPK — dystrophia myotonica protein kinase (CTG trinucleotide repeat expansion in 3'UTR)",
        "pathognomonic": "Anticipation (earlier onset / greater severity in successive generations) PATHOGNOMONIC for DM1; myotonia + distal weakness + cataracts + cardiac conduction defect; congenital DM1 from maternal expansion",
        "key_rule": "Annual ECG + Holter MANDATORY (cardiac conduction defects / AV block / VT are leading cause of premature death); annual pulmonary function tests; annual slit-lamp for cataracts; repeat ECG at EVERY hospital presentation",
        "mean_age_dx": 28.6,
        "pct_female": 0.50,
        "attack_freq_per_year": 365.0,  # daily myotonia
        "dx_delay_years": 6.8,
        "severity_score": 7.4,
        "cardiac_involvement_pct": 0.65,
        "respiratory_pct": 0.55,
        "prevalence_note": "Most common adult muscular dystrophy; ~1 in 8,000; anticipation — congenital DM1 is most severe (maternal transmission; >1000 CTG repeats)",
        "treatment": "Mexiletine for myotonia; pacemaker/ICD for cardiac conduction; NIV/BiPAP for respiratory; annual surveillance; genetic counselling",
        "trigger": "Progressive disease — not episodic; CTG repeat length correlates with severity",
        "clinical_facts": [
            "DMPK CTG-repeat expansion in 3'UTR (normal <35 repeats; DM1 >50 repeats; congenital DM1 >1000 repeats) — expanded RNA forms nuclear foci sequestering MBNL1 splicing factor, causing toxic RNA gain-of-function with multisystem splicing dysregulation",
            "Anticipation PATHOGNOMONIC: each generation has earlier onset and greater severity; maternal transmission of large expansions causes congenital DM1 (floppy infant, neonatal respiratory failure, intellectual disability) — paternal transmission rarely causes congenital form",
            "Classic DM1 phenotype: distal muscle weakness (finger extensors, foot dorsiflexors — DISTAL CONTRAST TO DM2 which is proximal), grip myotonia, myopathic facies (ptosis, facial diplegia, temporal wasting), cataracts (posterior subcapsular, iridescent)",
            "Cardiac conduction disease (1st/2nd/3rd degree AV block, bundle branch block, VT, sudden cardiac death) is a LEADING CAUSE OF PREMATURE DEATH in DM1 — annual 12-lead ECG + 24-hour Holter MANDATORY; prophylactic pacemaker if PR >240 ms or any 2nd/3rd degree block",
            "Respiratory failure (diaphragm + intercostal weakness) is the second major cause of premature death; annual spirometry (FVC supine and upright); nocturnal hypoventilation precedes daytime respiratory failure; NIV (BiPAP) is life-prolonging",
            "Myotonia in DM1: grip myotonia (unable to release handshake), percussion myotonia; WARM-UP PHENOMENON present but less dramatic than CLCN1 myotonia congenita; mexiletine effective",
            "Gastrointestinal: smooth muscle involvement causes dysphagia, gastroparesis, pseudo-obstruction, constipation — underrecognized in DM1 management",
            "Endocrine: insulin resistance (not insulin deficiency) — diabetes mellitus or impaired glucose tolerance in ~20%; testicular atrophy in males; hypersomnolence (excessive daytime sleepiness) from CNS involvement",
            "DM1 vs DM2 key difference: DM1 = DISTAL weakness + anticipation; DM2 (CNBP) = PROXIMAL weakness + NO anticipation; DM2 generally milder cardiac; DM2 pain prominent",
            "Genetic testing: Southern blot (definitive CTG repeat quantification) or triplet repeat PCR; both alleles must be sized; repeat length in blood leukocytes may underestimate somatic mosaicism in muscle",
        ],
    },
    "CNBP": {
        "aa": 629, "kda": 53, "locus": "19q13.11", "inh": "AD CCTG-repeat expansion",
        "syndrome": "Myotonic-Dystrophy-type-2-DM2 / PROMM-Proximal-Myotonic-Myopathy",
        "channel": "CNBP — CCHC-type zinc finger nucleic acid binding protein (CCTG tetranucleotide repeat expansion in intron 1)",
        "pathognomonic": "Proximal weakness (hip flexors, neck flexors — contrast DM1 distal) + myotonia + NO anticipation + pain prominent; DM2 milder cardiac than DM1",
        "key_rule": "NO ANTICIPATION in DM2 (contrast DM1) — repeat size does not increase generation-to-generation; PROXIMAL not distal weakness; pain is prominent and often misdiagnosed as fibromyalgia; cardiac surveillance still mandatory but milder than DM1",
        "mean_age_dx": 42.8,
        "pct_female": 0.52,
        "attack_freq_per_year": 365.0,  # daily myotonia/pain
        "dx_delay_years": 9.4,
        "severity_score": 5.8,
        "cardiac_involvement_pct": 0.28,
        "respiratory_pct": 0.20,
        "prevalence_note": "~1 in 8,000-20,000 (similar to DM1 in European populations, particularly Finnish/German); often underdiagnosed due to late onset and milder phenotype",
        "treatment": "Mexiletine for myotonia; pain management; cardiac surveillance (less aggressive than DM1 but still required); genetic counselling",
        "trigger": "Progressive disease; CCTG repeat >75 intronic repeats (normal <26); no correlation of repeat size with clinical severity (unlike DM1)",
        "clinical_facts": [
            "CNBP encodes CCHC-type zinc finger nucleic acid binding protein (53 kDa); CCTG tetranucleotide repeat expansion in intron 1 (normal <26 CCTG repeats; DM2 >75 repeats, typically thousands) — same toxic RNA splicing mechanism as DM1 via MBNL1 sequestration",
            "NO ANTICIPATION in DM2 — repeat size does not systematically increase generation-to-generation (unlike DM1 CTG expansion); congenital DM2 does not exist; this is a key distinguishing feature from DM1",
            "DM2 weakness is PROXIMAL (hip flexors, hip extensors, thigh muscles, neck flexors) — CONTRAST DM1 which is DISTAL (finger extensors, wrist extensors, foot dorsiflexors); this distinction is clinically essential",
            "Myotonia in DM2 is present but often milder and more fluctuating than DM1; grip and percussion myotonia detectable on EMG; warm-up phenomenon present; mexiletine effective",
            "PAIN is a prominent and distinguishing feature of DM2 (deep aching muscle pain, fatigue) — often misdiagnosed as fibromyalgia, chronic fatigue syndrome, or seronegative inflammatory myopathy; diagnosis delay average 9 years",
            "Cardiac involvement in DM2 is MILDER than DM1 but NOT absent — AV block, arrhythmia, dilated cardiomyopathy reported; annual ECG mandatory but pacemaker prophylaxis less frequently required than DM1",
            "Posterior subcapsular cataracts: present in most DM2 patients by age 50; identical iridescent cataract pattern to DM1; annual slit-lamp examination from diagnosis",
            "Endocrine and metabolic: insulin resistance, testicular failure, thyroid dysfunction common; hypersomnolence and cognitive dysfunction (milder than DM1) present",
            "Muscle biopsy in DM2: highly variable fibre size, MBNL1 nuclear foci on FISH, type 2 fibre atrophy (OPPOSITE to DM1 which shows type 1 fibre atrophy) — biopsy differentiation from DM1 requires FISH or genetic testing",
            "Genetic testing: repeat-primed PCR detects expansion presence; Southern blot for confirmation and sizing; CCTG repeat in intron 1 too large for standard PCR sizing; diagnosis can be missed on routine panel sequencing without specific repeat assay",
        ],
    },
}

N_PER_GENE = 40


def _gen_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    m = GENE_META[gene]
    n = N_PER_GENE

    ages = [max(0.5, rng.gauss(m["mean_age_dx"], 5.2)) for _ in range(n)]
    female = [rng.random() < m["pct_female"] for _ in range(n)]
    severity = [max(1.0, min(10.0, rng.gauss(m["severity_score"], 1.5))) for _ in range(n)]
    dx_delay = [max(0.1, rng.gauss(m["dx_delay_years"], 2.1)) for _ in range(n)]
    cardiac = [rng.random() < m["cardiac_involvement_pct"] for _ in range(n)]
    respiratory = [rng.random() < m["respiratory_pct"] for _ in range(n)]

    return {
        "gene": gene,
        "n": n,
        "mean_age_dx": round(sum(ages) / n, 1),
        "pct_female": round(sum(female) / n, 3),
        "mean_severity_score": round(sum(severity) / n, 1),
        "mean_dx_delay_years": round(sum(dx_delay) / n, 1),
        "cardiac_involvement_pct": round(sum(cardiac) / n * 100, 1),
        "respiratory_pct": round(sum(respiratory) / n * 100, 1),
    }


def _build_atlas_genes() -> list:
    result = []
    for gene, m in GENE_META.items():
        result.append({
            "gene": gene,
            "locus": m["locus"],
            "aa": m["aa"],
            "kda": m["kda"],
            "inheritance": m["inh"],
            "syndrome": m["syndrome"],
            "channel": m["channel"],
            "pathognomonic": m["pathognomonic"],
            "key_rule": m["key_rule"],
            "treatment": m["treatment"],
            "trigger": m["trigger"],
            "prevalence_note": m["prevalence_note"],
        })
    return result


def generate_overview() -> dict:
    cohorts = {g: _gen_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)}
    total = sum(c["n"] for c in cohorts.values())

    cardiac_all = sum(
        cohorts[g]["cardiac_involvement_pct"] * cohorts[g]["n"] / 100
        for g in GENES
    )
    respiratory_all = sum(
        cohorts[g]["respiratory_pct"] * cohorts[g]["n"] / 100
        for g in GENES
    )
    severity_vals = [cohorts[g]["mean_severity_score"] for g in GENES]
    delay_vals = [cohorts[g]["mean_dx_delay_years"] for g in GENES]

    gene_summaries = {}
    for gene in GENES:
        m = GENE_META[gene]
        gene_summaries[gene] = {
            "locus": m["locus"],
            "inheritance": m["inh"],
            "syndrome": m["syndrome"].split("/")[0].strip(),
            "channel": m["channel"],
            "pathognomonic": m["pathognomonic"].split(";")[0],
            "key_rule": m["key_rule"],
            "prevalence_note": m["prevalence_note"],
        }

    return {
        "atlas": "Hereditary-Skeletal-Muscle-Channelopathy-Myotonic-Disorder-Atlas",
        "genes": GENES,
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "key_stats": {
            "total_cardiac_involvement_patients": round(cardiac_all),
            "cardiac_involvement_pct_overall": round(cardiac_all / total * 100, 1),
            "total_respiratory_patients": round(respiratory_all),
            "respiratory_pct_overall": round(respiratory_all / total * 100, 1),
            "mean_severity_score_overall": round(sum(severity_vals) / len(severity_vals), 1),
            "mean_dx_delay_years_overall": round(sum(delay_vals) / len(delay_vals), 1),
            "mh_critical_rule": "RYR1+CACNA1S: volatile anesthetics+succinylcholine ABSOLUTE CI — dantrolene IV 2.5 mg/kg MUST be available; TIVA mandatory",
            "ats_triad_rule": "KCNJ2-ATS: TRIAD (periodic paralysis + LQT7 + dysmorphia) PATHOGNOMONIC; bidirectional VT PATHOGNOMONIC; avoid class-I antiarrhythmics",
            "dm1_rule": "DMPK-DM1: anticipation PATHOGNOMONIC; annual ECG+Holter MANDATORY; cardiac conduction = leading cause of premature death",
            "dm2_rule": "CNBP-DM2: PROXIMAL weakness + NO anticipation (contrast DM1 distal + anticipation); pain prominent; milder cardiac",
            "mc_warmup_rule": "CLCN1-Myotonia-Congenita: warm-up phenomenon PATHOGNOMONIC (improves with exercise, opposite of SCN4A-PMC which worsens with cold+exercise)",
            "tpp_rule": "KCNJ18-Thyrotoxic-PP: TREAT THYROTOXICOSIS FIRST — K+ supplementation alone insufficient; propranolol acute; TSH mandatory in all new-onset HypoPP",
        },
        "gene_summaries": gene_summaries,
    }


def generate_breakdown() -> dict:
    cohorts = {g: _gen_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)}
    atlas_genes = _build_atlas_genes()

    gene_data = {}
    for gene in GENES:
        c = cohorts[gene]
        m = GENE_META[gene]
        gene_data[gene] = {
            "syndrome": m["syndrome"].split("/")[0].strip(),
            "n": c["n"],
            "mean_age_dx": c["mean_age_dx"],
            "pct_female": c["pct_female"],
            "mean_severity_score": c["mean_severity_score"],
            "mean_dx_delay_years": c["mean_dx_delay_years"],
            "cardiac_involvement_pct": c["cardiac_involvement_pct"],
            "respiratory_pct": c["respiratory_pct"],
            "channel": m["channel"],
            "trigger": m["trigger"],
            "treatment": m["treatment"],
            "clinical_facts": m["clinical_facts"],
        }

    return {
        "atlas": "Hereditary-Skeletal-Muscle-Channelopathy-Myotonic-Disorder-Atlas",
        "gene_data": gene_data,
        "atlas_genes": atlas_genes,
        "cohort_summary": {
            "total_patients": N_PER_GENE * len(GENES),
            "seeds": f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
            "genes_covered": len(GENES),
            "sodium_channel_genes": ["SCN4A"],
            "calcium_channel_genes": ["CACNA1S"],
            "potassium_channel_genes": ["KCNJ2", "KCNJ18"],
            "chloride_channel_genes": ["CLCN1"],
            "calcium_release_channel_genes": ["RYR1"],
            "trinucleotide_repeat_genes": ["DMPK"],
            "tetranucleotide_repeat_genes": ["CNBP"],
            "mh_susceptibility_genes": ["CACNA1S", "RYR1"],
            "periodic_paralysis_genes": ["SCN4A", "CACNA1S", "KCNJ2", "KCNJ18"],
            "myotonic_disorder_genes": ["SCN4A", "CLCN1", "DMPK", "CNBP"],
            "dystrophic_myotonia_genes": ["DMPK", "CNBP"],
            "anesthesia_danger_genes": ["RYR1", "CACNA1S"],
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Skeletal-Muscle-Channelopathy-Myotonic-Disorder-Atlas",
        "critical_clinical_rules": {
            "MH-VOLATILE-ABSOLUTE-CI": "RYR1/CACNA1S: volatile halogenated anesthetics (halothane/sevoflurane/desflurane/isoflurane/enflurane) + succinylcholine ABSOLUTELY CONTRAINDICATED — TIVA (propofol+opioid) mandatory; dantrolene IV 2.5 mg/kg MUST be immediately available",
            "DANTROLENE-IV-MANDATORY": "Dantrolene IV 2.5 mg/kg push, repeat q5min until crisis resolves (max ~10 mg/kg); MUST be stocked at every anesthetic site serving MH-susceptible patients; each 20 mg vial requires 60 mL sterile water reconstitution",
            "IVCT-CHCT-GOLD-STANDARD": "In Vitro Contracture Test (IVCT, European)/Caffeine-Halothane Contracture Test (CHCT, North American) is GOLD STANDARD for MH susceptibility — muscle biopsy from vastus lateralis at ~60 specialized centers; C/S or H/S = MHS; C/N and H/N = MHN",
            "KCNJ2-ATS-TRIAD-PATHOGNOMONIC": "Andersen-Tawil Syndrome TRIAD = periodic paralysis + LQT7 (QU prolongation) + facial/skeletal dysmorphia — PATHOGNOMONIC when all 3 present; bidirectional ventricular tachycardia PATHOGNOMONIC for ATS",
            "KCNJ2-AVOID-CLASS-I": "Class-I antiarrhythmics (quinidine, flecainide, propafenone, procainamide) MUST BE AVOIDED in KCNJ2-ATS — directly block Kir2.1 and worsen QT prolongation; beta-blockers are safe first-line; ICD for sustained VT/VF",
            "DM1-ANNUAL-ECG-HOLTER": "DMPK-DM1: annual 12-lead ECG + 24-hour Holter MANDATORY — cardiac conduction defect (AV block, VT, sudden cardiac death) is leading cause of premature death; pacemaker if PR >240ms or any 2nd/3rd degree block",
            "DM1-DM2-CONTRAST": "DM1 (DMPK): DISTAL weakness + anticipation + congenital form possible + severe cardiac. DM2 (CNBP): PROXIMAL weakness + NO anticipation + no congenital form + milder cardiac + pain prominent",
            "MC-WARMUP-PATHOGNOMONIC": "CLCN1-Myotonia-Congenita: warm-up phenomenon (myotonia IMPROVES with repeated exercise) PATHOGNOMONIC — opposite of SCN4A-Paramyotonia-Congenita (worsens with cold+exercise); mexiletine first-line for both",
            "TPP-TREAT-THYROID-FIRST": "KCNJ18-Thyrotoxic-PP: TREAT THYROTOXICOSIS FIRST (antithyroid drugs/RAI/surgery) — K+ supplementation alone insufficient and may cause rebound hyperkalemia; TSH+FT4 MANDATORY in ALL new-onset hypokalemic periodic paralysis",
            "HYPOPPP-ACETAZOLAMIDE-CAUTION": "Acetazolamide effective for CACNA1S-HypoPP1 and SCN4A-HypoPP2 — but may WORSEN HypoPP caused by gating-pore current mutations; genetic subtyping guides treatment; NEVER give K+ containing IV fluids in HyperKPP attack",
        },
        "definitions": [
            {
                "term": "Hypokalemic Periodic Paralysis (HypoPP)",
                "definition": "Episodic flaccid muscle weakness with low serum K+ (<3.5 mEq/L, often <2.5 mEq/L). CACNA1S (HypoPP-type1, most common) and SCN4A (HypoPP-type2). Triggers: carbohydrate meals, insulin, exercise, cold. Management: oral/IV KCl; acetazolamide prophylaxis; EXCLUDE thyrotoxicosis (KCNJ18-TPP) — TSH mandatory.",
            },
            {
                "term": "Hyperkalemic Periodic Paralysis (HyperKPP)",
                "definition": "Episodic weakness with elevated or normal serum K+ during attack (SCN4A GOF). Triggers: K+-rich foods, rest after exercise, fasting. Treatment: acetazolamide (carbonic anhydrase inhibitor); avoid K+-containing IV fluids; glucose may abort attack.",
            },
            {
                "term": "Paramyotonia Congenita (PMC)",
                "definition": "SCN4A GOF mutation. Cold-triggered myotonia that WORSENS with repeated exercise (paradoxical myotonia) — PATHOGNOMONIC contrast to Myotonia Congenita (improves = warm-up). Ice-water forearm test positive. Treatment: mexiletine; acetazolamide; avoid cold.",
            },
            {
                "term": "Andersen-Tawil Syndrome (ATS / LQT7)",
                "definition": "KCNJ2 LOF. TRIAD: episodic periodic paralysis + LQT7 (QU prolongation, bidirectional VT PATHOGNOMONIC) + facial/skeletal dysmorphia (micrognathia, syndactyly, short stature). 60% have KCNJ2 mutations. Avoid class-I antiarrhythmics. ICD for VT/VF.",
            },
            {
                "term": "Myotonia Congenita (Thomsen AD / Becker AR)",
                "definition": "CLCN1 LOF. Myotonia (inability to relax muscle after contraction). WARM-UP PHENOMENON PATHOGNOMONIC: stiffness worse after rest, clears with exercise. Percussion myotonia, lid-lag on downgaze. Becker (AR) more severe with transient weakness at initiation. Mexiletine first-line.",
            },
            {
                "term": "Malignant Hyperthermia (MH)",
                "definition": "Life-threatening hypermetabolic crisis triggered by volatile anesthetics or succinylcholine in RYR1/CACNA1S carriers. Features: hyperthermia (rising >2°C/hr), masseter spasm (PATHOGNOMONIC post-succinylcholine), generalized rigidity, rhabdomyolysis, metabolic/respiratory acidosis. Treatment: STOP trigger, dantrolene IV 2.5 mg/kg, cooling, bicarbonate.",
            },
            {
                "term": "Dantrolene Sodium (IV)",
                "definition": "RyR1 inhibitor — blocks SR Ca2+ release. MANDATORY treatment for MH: 2.5 mg/kg IV push q5min until crisis resolves (max ~10 mg/kg). Must be stocked at ALL anesthetic sites where MH-susceptible patients may present. Oral dantrolene NOT equivalent for acute crisis. Reconstitute each 20 mg vial with 60 mL sterile water.",
            },
            {
                "term": "Myotonic Dystrophy type 1 (DM1 / Steinert Disease)",
                "definition": "DMPK CTG-repeat expansion in 3'UTR. Most common adult muscular dystrophy (~1/8,000). Anticipation PATHOGNOMONIC. Features: DISTAL weakness, myotonia, cataracts (posterior subcapsular), cardiac conduction defects, respiratory failure, hypersomnia, intellectual involvement (variable). Annual ECG + Holter mandatory. Leading death cause = cardiac conduction disease.",
            },
            {
                "term": "Myotonic Dystrophy type 2 (DM2 / PROMM)",
                "definition": "CNBP CCTG-repeat expansion in intron 1. Similar multisystem features to DM1 but: PROXIMAL weakness (not distal), NO anticipation, no congenital form, milder cardiac involvement, PAIN is prominent (often misdiagnosed as fibromyalgia). Later onset (mean ~40 yr). Annual ECG still mandatory.",
            },
            {
                "term": "Thyrotoxic Periodic Paralysis (TPP)",
                "definition": "Hypokalemic periodic paralysis in thyrotoxic patients. KCNJ18 LOF allele confers susceptibility. Strong male predominance (M:F 20:1 in Asians). TREAT THYROTOXICOSIS FIRST — attacks resolve with euthyroid state. Acute: propranolol + KCl for severe hypokalemia. TSH mandatory in ALL new-onset hypokalemic PP.",
            },
            {
                "term": "Anticipation (Trinucleotide/Tetranucleotide Repeat)",
                "definition": "Earlier onset and greater severity in successive generations due to repeat expansion during germline transmission. PATHOGNOMONIC for DM1 (CTG, DMPK) and other repeat-expansion diseases. DM2 (CNBP, CCTG) does NOT show anticipation — a key DM1/DM2 distinguishing feature.",
            },
            {
                "term": "Warm-Up Phenomenon vs Paradoxical Myotonia",
                "definition": "Warm-up (CLCN1-MC): myotonia IMPROVES with repeated exercise — chloride channel restores with repetitive depolarization. Paradoxical myotonia (SCN4A-PMC): myotonia WORSENS with cold + exercise — sodium channel inactivation failure accumulates. Clinically distinguishing: ask patient if muscle stiffness improves or worsens with exercise.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(json.dumps(ov, indent=2)[:800])
    print("\n=== BREAKDOWN (sample) ===")
    br = generate_breakdown()
    print(json.dumps({"genes": list(br["gene_data"].keys()), "total": br["cohort_summary"]["total_patients"]}, indent=2))
    print("\n=== DEFINITIONS (critical rules) ===")
    df = generate_definitions()
    for k, v in list(df["critical_clinical_rules"].items())[:3]:
        print(f"  {k}: {v[:80]}...")
