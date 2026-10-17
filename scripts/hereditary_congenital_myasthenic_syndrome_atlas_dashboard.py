#!/usr/bin/env python3
"""Hereditary-Congenital-Myasthenic-Syndrome-Atlas -- Complete 8-Gene Reference
Neuromuscular junction disorders with defects in pre/post-synaptic and synaptic basal lamina
(CHRNE/AChR-epsilon-most-common-AR, COLQ/endplate-AChEI-pyridostigmine-CONTRAINDICATED,
 RAPSN/clustering-defect, DOK7/limbgirdle-pyridostigmine-CONTRAINDICATED-salbutamol-first-line,
 AGRN/agrin-congenital, MUSK/postsynaptic-kinase, SCN4A/sodium-channel-fast-channel,
 GFPT1/glycosylation-tubular-aggregates-PATHOGNOMONIC):

CHRNE  (Acetylcholine receptor epsilon subunit; 493aa; 17p13.2; AR;
         Most common AR CMS worldwide (40-50%);
         Slow-channel (GOF) and fast-channel (LOF) subtypes;
         Pyridostigmine effective for LOF; quinidine for GOF slow-channel;
         seed SEED_BASE+0).
COLQ   (Acetylcholinesterase collagen-like tail subunit; 461aa; 3p25.1; AR;
         Endplate acetylcholinesterase deficiency;
         Pupillary light reflex absent/sluggish PATHOGNOMONIC;
         Pyridostigmine ABSOLUTELY CONTRAINDICATED — accumulates ACh, worsens;
         Ephedrine first-line;
         seed SEED_BASE+1).
RAPSN  (Receptor-associated protein of the synapse; 412aa; 11p11.2; AR;
         AChR clustering defect;
         Neonatal hypotonia + arthrogryposis + facial diplegia;
         N88K founder mutation Northern European;
         Pyridostigmine effective; 3,4-DAP adjunct;
         seed SEED_BASE+2).
DOK7   (Downstream of kinase 7; 504aa; 4p16.3; AR;
         Limb-girdle pattern CMS — PATHOGNOMONIC clinical pattern;
         Pyridostigmine ABSOLUTELY CONTRAINDICATED — initial benefit then severe worsening;
         Salbutamol/albuterol 4-8 mg/day = first-line SPECIFIC THERAPY;
         seed SEED_BASE+3).
AGRN   (Agrin; 2045aa; 1p36.33; AR;
         Distal/limb-girdle CMS;
         Low-amplitude CMAP with decremental response;
         Salbutamol effective;
         seed SEED_BASE+4).
MUSK   (Muscle-specific kinase; 869aa; 9q31.3; AR;
         Postsynaptic kinase deficiency;
         Distal + ocular + bulbar involvement;
         Pyridostigmine variable — can worsen;
         Ephedrine + 3,4-DAP preferred;
         seed SEED_BASE+5).
SCN4A  (Sodium channel Nav1.4; 1836aa; 17q23.3; AD;
         Fast-channel CMS (sodium channel) — DISTINCT mechanism from nicotinic AChR;
         Episodic weakness + myotonic discharge on EMG;
         Acetazolamide first-line for episodic;
         seed SEED_BASE+6).
GFPT1  (Glutamine-fructose-6-phosphate transaminase 1; 699aa; 2p13.1; AR;
         Glycosylation defect CMS — limb-girdle;
         Tubular aggregates on muscle biopsy PATHOGNOMONIC;
         3,4-DAP + pyridostigmine effective;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3814-3821)
"""
import random

SEED_BASE = 3814

ATLAS_GENES = [
    {
        "gene": "CHRNE",
        "protein": (
            "CHRNE -- 17p13.2 AR LOF/GOF -- 493aa -- "
            "Acetylcholine-receptor-epsilon-subunit-pentameric-nAChR-gamma-to-epsilon-switch-"
            "most-common-AR-CMS-40-50pct-worldwide-"
            "slow-channel-GOF-quinidine-fluoxetine-vs-fast-channel-LOF-pyridostigmine-"
            "ptosis-ophthalmoplegia-fluctuating-weakness-NMJ-decrement-RNS"
        ),
        "locus": "17p13.2",
        "protein_size": (
            "493 aa / ~54 kDa / 17p13.2 CHRNE acetylcholine receptor epsilon subunit molecular context: "
            "STRUCTURE: "
            "  493 aa; part of pentameric nicotinic AChR (alpha2-beta-delta-epsilon); "
            "  Epsilon subunit replaces gamma at 33 weeks gestation (fetal-to-adult switch); "
            "  Mutations cause: LOF (fast-channel) or GOF (slow-channel) subtypes; "
            "  MOST COMMON AR CMS gene worldwide (40-50% of AR CMS); "
            "CLINICAL: "
            "  Ptosis + ophthalmoplegia + fluctuating fatigable weakness; "
            "  Bulbar involvement variable; respiratory crises possible; "
            "  Repetitive nerve stimulation (RNS): decremental response ≥10%; "
            "  Single-fiber EMG: increased jitter PATHOGNOMONIC for NMJ disorder; "
            "SUBTYPE DISTINCTION (CRITICAL): "
            "  Fast-channel (LOF): channel opens less/shorter — pyridostigmine EFFECTIVE; "
            "  Slow-channel (GOF): channel stays open too long — pyridostigmine CONTRAINDICATED; "
            "  Slow-channel treatment: quinidine 2-10 mg/kg/day or fluoxetine 80-120 mg/day; "
            "  Slow-channel hallmark: repetitive CMAP on single nerve stimulation PATHOGNOMONIC; "
            "FOUNDER MUTATIONS: "
            "  p.Arg279Cys (Mediterranean/Middle East), p.Glu143Lys (South Asian); "
            "  p.Arg64Ter (Turkish); epsilon-null mutations common; "
            "  Genotype crucial to determine treatment polarity"
        ),
        "syndrome": "AChR-epsilon-CMS / most-common-AR-CMS / slow-channel-GOF-quinidine vs fast-channel-LOF-pyridostigmine",
        "inheritance": "AR",
        "pathognomonic": "Repetitive CMAP on single stimulus = slow-channel GOF PATHOGNOMONIC; decremental RNS",
        "key_therapy": "Fast-channel: pyridostigmine; Slow-channel: quinidine or fluoxetine (pyridostigmine CONTRAINDICATED for slow)",
        "critical_rules": [
            "CHRNE-must-determine-slow-vs-fast-channel-BEFORE-treating-opposite-polarity",
            "CHRNE-slow-channel-repetitive-CMAP-PATHOGNOMONIC-quinidine-fluoxetine",
            "CHRNE-fast-channel-pyridostigmine-EFFECTIVE-3-4-DAP-adjunct",
            "CHRNE-pyridostigmine-CONTRAINDICATED-in-slow-channel-GOF-worsens",
            "CHRNE-single-fiber-EMG-PATHOGNOMONIC-increased-jitter",
            "CHRNE-gamma-to-epsilon-switch-explains-neonatal-form-with-arthrogryposis",
            "CHRNE-genotype-treatment-polarity-MANDATORY-before-prescribing",
            "CHRNE-most-common-AR-CMS-gene-test-first-in-panel",
        ],
    },
    {
        "gene": "COLQ",
        "protein": (
            "COLQ -- 3p25.1 AR LOF -- 461aa -- "
            "Collagen-like-tail-AChE-anchoring-subunit-endplate-AChE-deficiency-"
            "absent-sluggish-pupillary-light-reflex-PATHOGNOMONIC-"
            "pyridostigmine-ABSOLUTELY-CONTRAINDICATED-accumulates-ACh-paradoxically-worsens-"
            "ephedrine-first-line-salbutamol-alternative"
        ),
        "locus": "3p25.1",
        "protein_size": (
            "461 aa / ~54 kDa / 3p25.1 COLQ collagen tail subunit molecular context: "
            "STRUCTURE: "
            "  461 aa; collagen-like tail anchors AChE asymmetric form to endplate basal lamina; "
            "  COLQ LOF -> AChE absent at endplate -> ACh NOT cleared after release; "
            "  Persistent ACh in synapse -> prolonged channel activation -> endplate myopathy; "
            "PATHOGNOMONIC FEATURES: "
            "  Pupillary light reflex: absent or severely sluggish PATHOGNOMONIC; "
            "  (AChE normally terminates muscarinic ACh effect on pupil); "
            "  Small pupils relative to dark: CLINICAL CLUE; "
            "  Repetitive CMAP on single nerve stimulus (similar to slow-channel); "
            "  Profound endplate myopathy on biopsy: degenerative junctional changes; "
            "CRITICAL TREATMENT RULE: "
            "  Pyridostigmine (AChEI) ABSOLUTELY CONTRAINDICATED; "
            "  Mechanism: AChE already absent — AChEI further blocks non-existent AChE; "
            "  ACh accumulates further -> overstimulation -> cholinergicdepolarisation block; "
            "  Clinical: initial brief improvement then profound worsening — DANGEROUS; "
            "  First-line: ephedrine 25-75 mg TID or salbutamol 4-8 mg TID; "
            "  Mechanism of benefit: sympathomimetics stabilize NMJ via beta-2 receptor; "
            "DIAGNOSIS: "
            "  Absent endplate AChE on specialized AChE histochemistry; "
            "  COLQ gene panel; absent pupillary reflexes as bedside PATHOGNOMONIC clue; "
            "  AChE inhibitor trial: brief improvement then WORSENING = diagnostic COLQ"
        ),
        "syndrome": "Endplate-AChE-deficiency / pupillary-reflex-ABSENT-PATHOGNOMONIC / pyridostigmine-ABSOLUTELY-CI",
        "inheritance": "AR",
        "pathognomonic": "Absent/sluggish pupillary light reflex PATHOGNOMONIC + brief response then worsening with pyridostigmine",
        "key_therapy": "Ephedrine 25-75 mg TID or salbutamol 4-8 mg TID FIRST-LINE; pyridostigmine ABSOLUTELY CONTRAINDICATED",
        "critical_rules": [
            "COLQ-pyridostigmine-ABSOLUTELY-CONTRAINDICATED-accumulates-ACh-worsens",
            "COLQ-pupillary-light-reflex-ABSENT-PATHOGNOMONIC-bedside-clue",
            "COLQ-ephedrine-salbutamol-first-line-sympathomimetic-mechanism",
            "COLQ-initial-brief-pyridostigmine-response-then-worsening-diagnostic-COLQ",
            "COLQ-repetitive-CMAP-similar-to-slow-channel-CHRNE",
            "COLQ-AChE-histochemistry-absent-endplate-on-biopsy",
            "COLQ-3-4-DAP-may-be-tried-but-evidence-limited",
            "COLQ-endplate-myopathy-profound-degenerative-changes-biopsy",
        ],
    },
    {
        "gene": "RAPSN",
        "protein": (
            "RAPSN -- 11p11.2 AR LOF -- 412aa -- "
            "Receptor-associated-protein-of-synapse-43kDa-"
            "AChR-clustering-defect-"
            "neonatal-hypotonia-arthrogryposis-facial-diplegia-"
            "N88K-Northern-European-founder-mutation-"
            "pyridostigmine-EFFECTIVE-3-4-DAP-adjunct"
        ),
        "locus": "11p11.2",
        "protein_size": (
            "412 aa / ~43 kDa / 11p11.2 RAPSN receptor-associated protein of synapse molecular context: "
            "STRUCTURE: "
            "  412 aa; 43 kDa protein; clusters and anchors AChR at postsynaptic membrane; "
            "  Interacts with MuSK-AGRN signaling cascade for NMJ formation; "
            "  RAPSN LOF -> AChR cannot cluster properly -> reduced postsynaptic AChR density; "
            "CLINICAL: "
            "  Neonatal/early onset most common; "
            "  Neonatal hypotonia + respiratory distress + feeding difficulties; "
            "  Arthrogryposis (joint contractures): 30-40% — due to in utero immobility; "
            "  Facial diplegia + ptosis + ophthalmoplegia; "
            "  Episodes of APNEA triggered by fever/infection: HALLMARK (respiratory crises); "
            "  Fatigable weakness with RNS decremental response; "
            "  CK: NORMAL (distinguishes from muscular dystrophy); "
            "FOUNDER MUTATIONS: "
            "  p.Asn88Lys (N88K): MOST COMMON worldwide especially Northern European; "
            "  Homozygous N88K: classical early-onset CMS; "
            "  p.Glu152Lys: second most common; "
            "TREATMENT: "
            "  Pyridostigmine: EFFECTIVE — AChR clustering defect benefits from increased ACh; "
            "  3,4-Diaminopyridine (3,4-DAP): adjunct — increases ACh release presynaptically; "
            "  Ephedrine: rescue for respiratory crises; "
            "  AVOID: aminoglycosides, fluoroquinolones, beta-blockers, magnesium (NMJ blockers)"
        ),
        "syndrome": "RAPSN-AChR-clustering-defect / N88K-founder / neonatal-hypotonia+arthrogryposis+apnea / pyridostigmine-EFFECTIVE",
        "inheritance": "AR",
        "pathognomonic": "Neonatal hypotonia + arthrogryposis + episodic apnea with fever + N88K mutation",
        "key_therapy": "Pyridostigmine + 3,4-DAP; avoid aminoglycosides/quinolones/beta-blockers/Mg",
        "critical_rules": [
            "RAPSN-pyridostigmine-EFFECTIVE-AChR-clustering-defect",
            "RAPSN-N88K-most-common-worldwide-Northern-European-founder",
            "RAPSN-fever-triggered-apnea-HALLMARK-respiratory-crisis-warning",
            "RAPSN-arthrogryposis-30-40pct-in-utero-immobility",
            "RAPSN-CK-NORMAL-distinguish-from-muscular-dystrophy",
            "RAPSN-aminoglycosides-fluoroquinolones-AVOID-NMJ-blockers",
            "RAPSN-3-4-DAP-adjunct-increases-ACh-release",
            "RAPSN-respiratory-crises-need-emergency-plan-fever-protocol",
        ],
    },
    {
        "gene": "DOK7",
        "protein": (
            "DOK7 -- 4p16.3 AR LOF -- 504aa -- "
            "Downstream-of-kinase-7-MuSK-activator-"
            "limb-girdle-CMS-PATHOGNOMONIC-clinical-pattern-"
            "pyridostigmine-ABSOLUTELY-CONTRAINDICATED-initial-benefit-then-severe-worsening-"
            "salbutamol-albuterol-4-8mg-day-FIRST-LINE-SPECIFIC-THERAPY"
        ),
        "locus": "4p16.3",
        "protein_size": (
            "504 aa / ~56 kDa / 4p16.3 DOK7 downstream of kinase 7 molecular context: "
            "STRUCTURE: "
            "  504 aa; cytoplasmic adaptor protein; binds and activates MuSK; "
            "  DOK7 LOF -> MuSK underactivated -> NMJ formation/maintenance impaired; "
            "  Postsynaptic defect but limb-girdle pattern (UNLIKE ocular/bulbar dominant CMS); "
            "PATHOGNOMONIC CLINICAL PATTERN: "
            "  Limb-girdle weakness: proximal > distal, legs > arms; "
            "  Minimal or NO ptosis, minimal ophthalmoplegia (ATYPICAL for CMS); "
            "  Often misdiagnosed as limb-girdle muscular dystrophy (LGMD); "
            "  Respiratory involvement: 50-60% (respiratory failure late feature); "
            "  CK: MILDLY elevated in some (adds confusion with LGMD); "
            "  RNS decremental response often requires stimulation of proximal muscles; "
            "CRITICAL TREATMENT (MOST IMPORTANT DOK7 RULE): "
            "  Pyridostigmine ABSOLUTELY CONTRAINDICATED in DOK7-CMS; "
            "  Mechanism: increased ACh overstimulates already fragile NMJ -> NMJ destruction; "
            "  Clinical: Initial 2-4 week improvement then profound worsening (worse than baseline); "
            "  First documented as DOK7 case series: dramatic response then catastrophic decline on pyridostigmine; "
            "  SALBUTAMOL (albuterol) 4-8 mg TID: SPECIFIC FIRST-LINE THERAPY; "
            "  Beta-2 agonist mechanism: stabilizes NMJ, reduces NMJ damage; "
            "  EPHEDRINE 25-75 mg TID: alternative if salbutamol not tolerated; "
            "DIAGNOSIS: "
            "  Panel testing MANDATORY — easily misdiagnosed; "
            "  p.c.1124_1127dupTGCC: frameshift; most common European allele"
        ),
        "syndrome": "DOK7-limb-girdle-CMS-PATHOGNOMONIC / pyridostigmine-ABSOLUTELY-CI / salbutamol-FIRST-LINE-SPECIFIC",
        "inheritance": "AR",
        "pathognomonic": "Limb-girdle pattern WITHOUT ptosis/ophthalmoplegia + worsening on pyridostigmine = DOK7 PATHOGNOMONIC",
        "key_therapy": "Salbutamol 4-8 mg TID FIRST-LINE; ephedrine alternative; pyridostigmine ABSOLUTELY CONTRAINDICATED",
        "critical_rules": [
            "DOK7-pyridostigmine-ABSOLUTELY-CONTRAINDICATED-initial-improvement-then-catastrophic-worsening",
            "DOK7-salbutamol-albuterol-4-8mg-TID-FIRST-LINE-SPECIFIC-THERAPY",
            "DOK7-limb-girdle-WITHOUT-ptosis-PATHOGNOMONIC-clinical-pattern",
            "DOK7-misdiagnosed-LGMD-frequently-panel-testing-MANDATORY",
            "DOK7-respiratory-failure-50-60pct-late-feature-respiratory-monitoring",
            "DOK7-worsening-on-pyridostigmine-is-diagnostic-of-DOK7-if-suspected",
            "DOK7-ephedrine-alternative-if-salbutamol-not-tolerated",
            "DOK7-c1124_1127dupTGCC-most-common-European-founder-allele",
        ],
    },
    {
        "gene": "AGRN",
        "protein": (
            "AGRN -- 1p36.33 AR LOF -- 2045aa -- "
            "Agrin-heparan-sulfate-proteoglycan-NMJ-organizer-"
            "neural-agrin-activates-MuSK-via-LRP4-"
            "distal-limb-girdle-CMS-"
            "salbutamol-ephedrine-effective-"
            "pyridostigmine-variable"
        ),
        "locus": "1p36.33",
        "protein_size": (
            "2045 aa / ~225 kDa / 1p36.33 AGRN agrin molecular context: "
            "STRUCTURE: "
            "  2045 aa; largest CMS gene; heparan sulfate proteoglycan; "
            "  Neural agrin (z+ isoform) activates MuSK via LRP4 co-receptor; "
            "  AGRN LOF -> MuSK NOT activated -> AChR clustering defect (upstream of RAPSN/DOK7); "
            "  Muscle agrin also contributes to NMJ maintenance; "
            "CLINICAL: "
            "  Distal muscle weakness: distal > proximal (ATYPICAL — most CMS proximal); "
            "  Limb-girdle pattern: overlap with DOK7; "
            "  Onset: congenital to early childhood; "
            "  Ptosis and ophthalmoplegia: variable, often mild; "
            "  Low-amplitude CMAP with decrement; "
            "  Wrist/finger extensor weakness characteristic; "
            "TREATMENT: "
            "  Salbutamol: EFFECTIVE — beta-2 agonist stabilizes MuSK-deficient NMJ; "
            "  Ephedrine: alternative; "
            "  Pyridostigmine: VARIABLE — may help some, may worsen others; "
            "  Caution with pyridostigmine (not absolutely CI as in DOK7/COLQ, but variable); "
            "  3,4-DAP: limited evidence, may be tried; "
            "  Avoid: aminoglycosides, magnesium, fluoroquinolones"
        ),
        "syndrome": "AGRN-CMS / neural-agrin-MuSK-pathway / distal-weakness-ATYPICAL / salbutamol-EFFECTIVE",
        "inheritance": "AR",
        "pathognomonic": "Distal > proximal weakness + NMJ decrement + AGRN gene confirmation",
        "key_therapy": "Salbutamol or ephedrine FIRST-LINE; pyridostigmine variable (use cautiously)",
        "critical_rules": [
            "AGRN-distal-predominant-weakness-ATYPICAL-for-CMS-diagnostic-clue",
            "AGRN-salbutamol-ephedrine-effective-MuSK-pathway-stabilization",
            "AGRN-pyridostigmine-VARIABLE-not-absolutely-CI-but-cautious-use",
            "AGRN-neural-agrin-z-plus-isoform-specific-MuSK-activator",
            "AGRN-aminoglycosides-fluoroquinolones-AVOID",
            "AGRN-largest-CMS-gene-2045aa-WES-panel-MANDATORY",
            "AGRN-upstream-of-RAPSN-DOK7-in-MuSK-signaling-pathway",
            "AGRN-CMAP-low-amplitude-decrement-NMJ-electrophysiology",
        ],
    },
    {
        "gene": "MUSK",
        "protein": (
            "MUSK -- 9q31.3 AR LOF -- 869aa -- "
            "Muscle-specific-kinase-receptor-tyrosine-kinase-"
            "NMJ-formation-maintenance-master-regulator-"
            "distal-ocular-bulbar-involvement-"
            "pyridostigmine-variable-can-worsen-"
            "ephedrine-3-4-DAP-preferred"
        ),
        "locus": "9q31.3",
        "protein_size": (
            "869 aa / ~97 kDa / 9q31.3 MUSK muscle-specific kinase molecular context: "
            "STRUCTURE: "
            "  869 aa; receptor tyrosine kinase; master regulator of NMJ formation; "
            "  Activated by AGRN-LRP4 complex; phosphorylates DOK7; activates RAPSN; "
            "  MUSK LOF -> entire NMJ signaling cascade impaired; "
            "  AR MUSK: CMS (distinct from autoimmune anti-MuSK MG); "
            "CLINICAL: "
            "  Distal + ocular + bulbar: unusual combination (CMS usually ocular/proximal); "
            "  Facial weakness + ptosis + ophthalmoplegia often prominent; "
            "  Neck flexor weakness: characteristic (>80%); "
            "  Respiratory involvement: 30-40%; "
            "  CK: NORMAL; "
            "  RNS: decremental response distal muscles most prominent; "
            "TREATMENT: "
            "  Pyridostigmine: VARIABLE — can improve initially but may worsen; "
            "  Not as dangerous as DOK7/COLQ but uncertain; "
            "  Ephedrine: preferred first-line (25-75 mg TID); "
            "  3,4-DAP: adjunct (increases ACh release, compensates for post-synaptic defect); "
            "DISTINCTION FROM AUTOIMMUNE ANTI-MuSK MG: "
            "  Autoimmune: anti-MuSK antibodies (seronegative for AChR); "
            "  Congenital AR-MUSK: genetic defect, no antibodies; "
            "  Treatment OPPOSITE: autoimmune benefits from IVIg/rituximab; CMS does not; "
            "  Pyridostigmine: works for autoimmune anti-MuSK MG but variable in CMS MUSK"
        ),
        "syndrome": "MUSK-CMS / distal+ocular+bulbar-pattern / pyridostigmine-variable / ephedrine+3-4-DAP-preferred",
        "inheritance": "AR",
        "pathognomonic": "Distal+ocular+bulbar combined + AR MUSK gene + no anti-MuSK antibodies",
        "key_therapy": "Ephedrine 25-75 mg TID + 3,4-DAP; pyridostigmine variable (not absolute CI but caution)",
        "critical_rules": [
            "MUSK-distinguish-from-autoimmune-anti-MuSK-MG-no-antibodies-in-CMS",
            "MUSK-ephedrine-3-4-DAP-preferred-pyridostigmine-variable-uncertain",
            "MUSK-distal-plus-ocular-plus-bulbar-combined-pattern-UNUSUAL",
            "MUSK-neck-flexor-weakness-80pct-characteristic",
            "MUSK-pyridostigmine-NOT-absolutely-CI-but-use-cautiously",
            "MUSK-IVIg-rituximab-for-autoimmune-NOT-for-CMS-MUSK",
            "MUSK-CK-NORMAL-distinguish-from-inflammatory-myopathy",
            "MUSK-master-NMJ-signaling-regulator-AGRN-LRP4-MUSK-DOK7-RAPSN-cascade",
        ],
    },
    {
        "gene": "SCN4A",
        "protein": (
            "SCN4A -- 17q23.3 AD LOF/GOF -- 1836aa -- "
            "Nav1-4-voltage-gated-sodium-channel-skeletal-muscle-"
            "fast-channel-CMS-distinct-mechanism-nicotinic-AChR-"
            "episodic-weakness-myotonic-discharge-EMG-"
            "acetazolamide-first-line-episodic-"
            "NOT-treated-with-pyridostigmine"
        ),
        "locus": "17q23.3",
        "protein_size": (
            "1836 aa / ~211 kDa / 17q23.3 SCN4A Nav1.4 sodium channel molecular context: "
            "STRUCTURE: "
            "  1836 aa; voltage-gated sodium channel alpha subunit; skeletal muscle; "
            "  Nav1.4: triggers action potential in skeletal muscle; "
            "  LOF -> muscle cannot fire properly -> episodic weakness; "
            "  GOF -> channel inactivation impaired -> persistent Na+ entry -> myotonia; "
            "CMS MECHANISM (DISTINCT FROM NICOTINIC AChR): "
            "  SCN4A LOF at NMJ: Nav1.4 at endplate cannot propagate action potential; "
            "  This is a POSTSYNAPTIC muscle excitation defect, NOT a nicotinic receptor defect; "
            "  RNS: decremental response — mimics CMS but mechanism different; "
            "  SINGLE-FIBER EMG: increased jitter — NMJ appearance; "
            "CLINICAL SUBTYPES: "
            "  GOF (SCN4A): hyperkalemic periodic paralysis (HyperKPP), myotonia fluctuans; "
            "  LOF (SCN4A): hypokalemic periodic paralysis type 2 (HypoKPP2); "
            "  CMS subtype (specific variants): episodic apnea + weakness + decrement; "
            "  Neonatal onset: respiratory failure, hypotonia; "
            "TREATMENT: "
            "  Acetazolamide: FIRST-LINE for episodic CMS from SCN4A; "
            "  Mexiletine: sodium channel blocker for myotonic component; "
            "  Pyridostigmine: NOT effective (not a nicotinic AChR defect); "
            "  NOT a nicotinic receptor defect — pyridostigmine irrelevant/potentially harmful"
        ),
        "syndrome": "SCN4A-fast-channel-CMS / Nav1.4-NOT-nicotinic-AChR / episodic-apnea / acetazolamide-FIRST-LINE",
        "inheritance": "AD",
        "pathognomonic": "Episodic respiratory failure/apnea + myotonic discharges EMG + SCN4A gene + decrement RNS",
        "key_therapy": "Acetazolamide first-line; mexiletine for myotonia; pyridostigmine NOT indicated (wrong mechanism)",
        "critical_rules": [
            "SCN4A-NOT-nicotinic-AChR-defect-pyridostigmine-NOT-indicated",
            "SCN4A-acetazolamide-FIRST-LINE-episodic-CMS-carbonic-anhydrase-mechanism",
            "SCN4A-myotonic-discharges-EMG-CLUE-to-sodium-channel-CMS",
            "SCN4A-AD-inheritance-UNLIKE-most-AR-CMS-genes",
            "SCN4A-episodic-apnea-neonatal-life-threatening-presentation",
            "SCN4A-mexiletine-for-myotonic-component-sodium-channel-blocker",
            "SCN4A-decrement-on-RNS-mimics-nicotinic-CMS-different-mechanism",
            "SCN4A-same-gene-as-hyperkalemic-periodic-paralysis-different-variants",
        ],
    },
    {
        "gene": "GFPT1",
        "protein": (
            "GFPT1 -- 2p13.1 AR LOF -- 699aa -- "
            "Glutamine-fructose-6-phosphate-transaminase-1-"
            "N-glycosylation-defect-glycosylation-CMS-"
            "tubular-aggregates-muscle-biopsy-PATHOGNOMONIC-"
            "3-4-DAP-plus-pyridostigmine-effective-"
            "limb-girdle-pattern-minimal-ophthalmoplegia"
        ),
        "locus": "2p13.1",
        "protein_size": (
            "699 aa / ~79 kDa / 2p13.1 GFPT1 glutamine fructose-6-phosphate transaminase 1 molecular context: "
            "STRUCTURE: "
            "  699 aa; rate-limiting enzyme in hexosamine biosynthesis pathway; "
            "  Converts fructose-6-P + glutamine -> glucosamine-6-P; "
            "  Product enters N-glycosylation pathway for protein glycosylation; "
            "  GFPT1 LOF -> NMJ proteins (AChR subunits, MuSK, AGRN) underglycosylated; "
            "  Underglycosylated NMJ proteins: impaired clustering, stability, function; "
            "  GLYCOSYLATION CMS: distinct mechanism from structural NMJ protein defects; "
            "PATHOGNOMONIC FEATURE: "
            "  Tubular aggregates on muscle biopsy: PATHOGNOMONIC for glycosylation CMS; "
            "  Tubular aggregates: dilated sarcoplasmic reticulum tubules seen on EM; "
            "  NOT specific to GFPT1 only (also in other glycosylation CMS: ALG2, DPAGT1, ALG14); "
            "  But GFPT1 is MOST COMMON glycosylation CMS gene; "
            "CLINICAL: "
            "  Limb-girdle weakness: proximal > distal; "
            "  Minimal ptosis and ophthalmoplegia (UNLIKE AChR-defect CMS); "
            "  Exercise intolerance + fatigable weakness; "
            "  Late childhood / adolescent onset often; "
            "  CK: mildly elevated (glycosylation myopathy component); "
            "  RNS: decremental response (typical CMS pattern); "
            "TREATMENT: "
            "  3,4-DAP (3,4-diaminopyridine): EFFECTIVE — increases ACh release; "
            "  Pyridostigmine: often added as adjunct; "
            "  Combination effective (unlike DOK7/COLQ where pyridostigmine is CI)"
        ),
        "syndrome": "GFPT1-glycosylation-CMS / tubular-aggregates-PATHOGNOMONIC / limb-girdle / 3-4-DAP+pyridostigmine-effective",
        "inheritance": "AR",
        "pathognomonic": "Tubular aggregates on muscle biopsy PATHOGNOMONIC for glycosylation CMS (GFPT1 most common)",
        "key_therapy": "3,4-DAP + pyridostigmine combination; tubular aggregate biopsy triggers GFPT1/glycosylation CMS workup",
        "critical_rules": [
            "GFPT1-tubular-aggregates-biopsy-PATHOGNOMONIC-triggers-glycosylation-CMS-panel",
            "GFPT1-most-common-glycosylation-CMS-gene-ALG2-DPAGT1-ALG14-also-have-tubular-aggregates",
            "GFPT1-3-4-DAP-EFFECTIVE-pyridostigmine-adjunct-UNLIKE-DOK7-COLQ",
            "GFPT1-limb-girdle-minimal-ophthalmoplegia-UNLIKE-CHRNE-AChR-CMS",
            "GFPT1-NMJ-proteins-underglycosylated-AChR-MuSK-AGRN-all-affected",
            "GFPT1-hexosamine-biosynthesis-pathway-glycosylation-mechanism",
            "GFPT1-CK-mildly-elevated-glycosylation-myopathy-component",
            "GFPT1-late-childhood-adolescent-onset-often-delayed-diagnosis",
        ],
    },
]


def _make_cohort(gene_data: dict, seed: int) -> dict:
    """Generate a synthetic 40-patient cohort for one CMS gene."""
    rng = random.Random(seed)
    n = 40
    gene = gene_data["gene"]

    # Age at diagnosis (years) — gene-specific
    age_ranges = {
        "CHRNE": (0.01, 5.0),   # neonatal to childhood
        "COLQ":  (0.01, 3.0),   # neonatal to infancy
        "RAPSN": (0.01, 0.5),   # neonatal predominant
        "DOK7":  (0.5,  10.0),  # infancy to childhood; often delayed Dx
        "AGRN":  (0.01, 5.0),   # congenital to early childhood
        "MUSK":  (0.01, 10.0),  # variable onset
        "SCN4A": (0.01, 2.0),   # neonatal apnea presentation common
        "GFPT1": (5.0,  20.0),  # late childhood / adolescent
    }
    lo, hi = age_ranges.get(gene, (0.5, 10.0))
    ages = [round(rng.uniform(lo, hi), 2) for _ in range(n)]
    mean_age_dx = round(sum(ages) / n, 2)

    # Respiratory crises
    resp_rates = {
        "CHRNE": 0.30, "COLQ": 0.35, "RAPSN": 0.50,
        "DOK7": 0.55, "AGRN": 0.25, "MUSK": 0.35,
        "SCN4A": 0.65, "GFPT1": 0.15,
    }
    resp_n = sum(1 for _ in range(n) if rng.random() < resp_rates.get(gene, 0.3))

    # Ptosis/ophthalmoplegia
    oph_rates = {
        "CHRNE": 0.85, "COLQ": 0.70, "RAPSN": 0.75,
        "DOK7": 0.15, "AGRN": 0.40, "MUSK": 0.70,
        "SCN4A": 0.30, "GFPT1": 0.20,
    }
    oph_n = sum(1 for _ in range(n) if rng.random() < oph_rates.get(gene, 0.5))

    # Pyridostigmine adverse event (worsening on pyridostigmine)
    pyr_adverse_rates = {
        "CHRNE": 0.10, "COLQ": 0.90, "RAPSN": 0.05,  # COLQ almost always worsens
        "DOK7": 0.85, "AGRN": 0.20, "MUSK": 0.30,
        "SCN4A": 0.05, "GFPT1": 0.05,
    }
    pyr_adverse_n = sum(1 for _ in range(n) if rng.random() < pyr_adverse_rates.get(gene, 0.1))

    # Gene-specific extra features
    extra = {}
    if gene == "CHRNE":
        extra["slow_channel_n"] = sum(1 for _ in range(n) if rng.random() < 0.35)
        extra["fast_channel_n"] = n - extra["slow_channel_n"]
        extra["quinidine_treated_n"] = extra["slow_channel_n"]
    elif gene == "COLQ":
        extra["pupil_abnormal_n"] = sum(1 for _ in range(n) if rng.random() < 0.90)
        extra["ache_absent_biopsy_n"] = sum(1 for _ in range(n) if rng.random() < 0.85)
    elif gene == "RAPSN":
        extra["arthrogryposis_n"] = sum(1 for _ in range(n) if rng.random() < 0.35)
        extra["n88k_n"] = sum(1 for _ in range(n) if rng.random() < 0.65)
        extra["apnea_episode_n"] = sum(1 for _ in range(n) if rng.random() < 0.50)
    elif gene == "DOK7":
        extra["limb_girdle_n"] = sum(1 for _ in range(n) if rng.random() < 0.90)
        extra["misdiagnosed_lgmd_n"] = sum(1 for _ in range(n) if rng.random() < 0.55)
        extra["salbutamol_response_n"] = sum(1 for _ in range(n) if rng.random() < 0.85)
    elif gene == "SCN4A":
        extra["myotonic_discharge_n"] = sum(1 for _ in range(n) if rng.random() < 0.70)
        extra["episodic_apnea_n"] = sum(1 for _ in range(n) if rng.random() < 0.65)
        extra["acetazolamide_response_n"] = sum(1 for _ in range(n) if rng.random() < 0.75)
    elif gene == "GFPT1":
        extra["tubular_aggregates_n"] = sum(1 for _ in range(n) if rng.random() < 0.80)
        extra["ck_elevated_n"] = sum(1 for _ in range(n) if rng.random() < 0.45)
        extra["dap_response_n"] = sum(1 for _ in range(n) if rng.random() < 0.80)

    return {
        "gene": gene,
        "n": n,
        "mean_age_dx": mean_age_dx,
        "respiratory_crisis_n": resp_n,
        "ptosis_ophthalmoplegia_n": oph_n,
        "pyridostigmine_adverse_n": pyr_adverse_n,
        **extra,
        "syndrome": gene_data["syndrome"],
        "pathognomonic": gene_data["pathognomonic"],
        "key_therapy": gene_data["key_therapy"],
        "inheritance": gene_data["inheritance"],
        "locus": gene_data["locus"],
        "protein_size": gene_data["protein_size"],
        "critical_rules": gene_data["critical_rules"],
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    total = sum(c["n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Congenital-Myasthenic-Syndrome-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "description": (
            "Complete 8-gene Hereditary Congenital Myasthenic Syndrome reference atlas. "
            "CHRNE (most common AR CMS 40-50%; slow-channel GOF → quinidine MANDATORY, "
            "pyridostigmine CONTRAINDICATED in slow-channel; fast-channel LOF → pyridostigmine EFFECTIVE). "
            "COLQ (endplate AChE deficiency; absent pupillary reflex PATHOGNOMONIC; "
            "pyridostigmine ABSOLUTELY CONTRAINDICATED; ephedrine/salbutamol first-line). "
            "RAPSN (AChR clustering defect; N88K Northern European founder; "
            "neonatal hypotonia+arthrogryposis; fever-triggered apnea HALLMARK; "
            "pyridostigmine EFFECTIVE). "
            "DOK7 (limb-girdle WITHOUT ptosis PATHOGNOMONIC; "
            "pyridostigmine ABSOLUTELY CONTRAINDICATED; salbutamol/albuterol SPECIFIC first-line; "
            "misdiagnosed as LGMD frequently). "
            "AGRN (distal-predominant ATYPICAL; salbutamol/ephedrine effective; "
            "pyridostigmine variable). "
            "MUSK (distal+ocular+bulbar combined; distinguish from autoimmune anti-MuSK MG; "
            "ephedrine+3,4-DAP preferred; pyridostigmine variable). "
            "SCN4A (Nav1.4 sodium channel — NOT nicotinic AChR; episodic apnea; "
            "acetazolamide first-line; pyridostigmine NOT indicated; AD inheritance). "
            "GFPT1 (glycosylation CMS; tubular aggregates biopsy PATHOGNOMONIC; "
            "3,4-DAP+pyridostigmine EFFECTIVE; limb-girdle minimal ophthalmoplegia). "
            "320-patient aggregate, 8 x 40 patients, seeds 3814-3821."
        ),
        "gene_summaries": [
            {
                "gene": g["gene"],
                "syndrome": g["syndrome"],
                "key_therapy": g["key_therapy"],
                "pathognomonic": g["pathognomonic"],
                "inheritance": g["inheritance"],
                "locus": g["locus"],
                "protein_size": g["protein_size"].split("molecular context")[0].strip(),
            }
            for g in ATLAS_GENES
        ],
        "cohort_summary": {
            c["gene"]: {
                "n": c["n"],
                "mean_age_dx_yr": c["mean_age_dx"],
                "respiratory_crisis_n": c["respiratory_crisis_n"],
                "ptosis_ophthalmoplegia_n": c["ptosis_ophthalmoplegia_n"],
                "pyridostigmine_adverse_n": c["pyridostigmine_adverse_n"],
            }
            for c in cohorts
        },
    }


def generate_breakdown():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    return {
        "atlas": "Hereditary-Congenital-Myasthenic-Syndrome-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_data": {c["gene"]: c for c in cohorts},
        "atlas_genes": ATLAS_GENES,
        "total_patients": sum(c["n"] for c in cohorts),
    }


def generate_definitions():
    return {
        "atlas": "Hereditary-Congenital-Myasthenic-Syndrome-Atlas",
        "definitions": {
            "Congenital Myasthenic Syndrome (CMS)": (
                "Heterogeneous group of inherited neuromuscular junction (NMJ) disorders "
                "caused by mutations affecting NMJ structure or function. "
                "Distinguished from autoimmune myasthenia gravis: CMS is genetic, no antibodies."
            ),
            "Neuromuscular junction (NMJ)": (
                "Synapse between motor nerve terminal and skeletal muscle fiber. "
                "Three zones: presynaptic (ACh synthesis/release), synaptic cleft (AChE), "
                "postsynaptic (AChR, rapsyn, MuSK, DOK7 clustering apparatus)."
            ),
            "Repetitive nerve stimulation (RNS)": (
                "Electrophysiological test for NMJ disorders. "
                "A decremental CMAP response (≥10% amplitude reduction at 3Hz) "
                "indicates impaired NMJ transmission — present in all CMS subtypes."
            ),
            "Single-fiber EMG": (
                "Most sensitive NMJ test. Increased jitter (variability of interpotential interval) "
                "PATHOGNOMONIC for NMJ disorder (CMS or autoimmune MG). "
                "Cannot distinguish CMS from autoimmune — use antibody testing and genetics."
            ),
            "Pyridostigmine": (
                "Acetylcholinesterase inhibitor (AChEI). Prolongs ACh in synapse. "
                "EFFECTIVE: AChR deficiency (CHRNE fast-channel, RAPSN, GFPT1). "
                "ABSOLUTELY CONTRAINDICATED: COLQ (no AChE to inhibit), DOK7 (NMJ destruction). "
                "VARIABLE: MUSK, AGRN, CHRNE slow-channel GOF. "
                "Rule: know the gene subtype BEFORE prescribing pyridostigmine."
            ),
            "Salbutamol / albuterol": (
                "Beta-2 adrenergic agonist. SPECIFIC FIRST-LINE THERAPY for DOK7-CMS. "
                "Also effective in COLQ, AGRN, and some MUSK-CMS. "
                "Mechanism: beta-2 activation stabilizes NMJ, reduces NMJ degradation. "
                "Dose: 4-8 mg TID oral (not inhaled — systemic effect required for CMS)."
            ),
            "3,4-Diaminopyridine (3,4-DAP)": (
                "Presynaptic agent — blocks K+ channels, prolongs nerve terminal action potential, "
                "increases ACh release. Effective for postsynaptic CMS (RAPSN, GFPT1). "
                "Often combined with pyridostigmine. "
                "Requires special pharmacy (not standard formulary)."
            ),
            "Slow-channel CMS": (
                "CHRNE (or CHRNB1/CHRNA1) GOF mutation: channel open time PROLONGED. "
                "Persistent ACh-receptor occupancy → endplate myopathy. "
                "Repetitive CMAP on single nerve stimulus: PATHOGNOMONIC. "
                "Treatment: quinidine or fluoxetine (prolong refractory period). "
                "Pyridostigmine: CONTRAINDICATED (already too much ACh activity)."
            ),
            "Fast-channel CMS": (
                "AChR LOF: channel open time SHORTENED or conductance reduced. "
                "Less ACh effect per vesicle release. "
                "Treatment: pyridostigmine (increases synaptic ACh) + 3,4-DAP. "
                "Opposite of slow-channel — gene subtype determines polarity."
            ),
            "Endplate AChE deficiency (COLQ)": (
                "Absent AChE at endplate → ACh not cleared → persistent stimulation → "
                "calcium overload → endplate myopathy. "
                "Diagnostic clue: absent pupillary light reflex (muscarinic, not nicotinic). "
                "Pyridostigmine CONTRAINDICATED: AChE absent — inhibiting nothing, ACh accumulates."
            ),
            "Tubular aggregates": (
                "Dilated sarcoplasmic reticulum tubules seen on muscle biopsy (EM or Gomori stain). "
                "PATHOGNOMONIC for glycosylation-defect CMS (GFPT1, ALG2, DPAGT1, ALG14). "
                "GFPT1 is the most common glycosylation CMS gene. "
                "Finding tubular aggregates: order GFPT1 gene panel."
            ),
            "Glycosylation CMS": (
                "CMS caused by N-glycosylation pathway defects affecting NMJ proteins. "
                "Genes: GFPT1 (most common), ALG2, DPAGT1, ALG14, GMPPB. "
                "Limb-girdle pattern; tubular aggregates on biopsy PATHOGNOMONIC. "
                "Treatment: 3,4-DAP + pyridostigmine (effective unlike DOK7/COLQ)."
            ),
            "DOK7-CMS": (
                "Most important CMS diagnosis to recognize because: "
                "1. Misdiagnosed as LGMD (limb-girdle muscular dystrophy) commonly. "
                "2. Pyridostigmine CONTRAINDICATED — causes irreversible NMJ damage. "
                "3. Salbutamol SPECIFIC therapy: dramatic response in 70-90%. "
                "Clinical: limb-girdle weakness WITHOUT ptosis/ophthalmoplegia."
            ),
        },
        "critical_clinical_rules": {
            rule: True
            for gene_data in ATLAS_GENES
            for rule in gene_data["critical_rules"]
        },
        "cascade_testing": {
            "CHRNE": ["CHRNE_panel_slow_vs_fast_channel", "CHRNB1_CHRNA1_CHRND_if_CHRNE_negative"],
            "COLQ": ["COLQ_gene_panel", "AChE_histochemistry_biopsy_absent"],
            "RAPSN": ["RAPSN_gene_panel", "N88K_founder_screen_Northern_European"],
            "DOK7": ["DOK7_gene_panel", "c1124_1127dupTGCC_founder_European"],
            "AGRN": ["AGRN_gene_panel_WES_preferred_large_gene"],
            "MUSK": ["MUSK_gene_panel", "anti_MuSK_antibodies_NEGATIVE_confirm_congenital"],
            "SCN4A": ["SCN4A_gene_panel", "EMG_myotonic_discharges_clue"],
            "GFPT1": ["GFPT1_gene_panel", "ALG2_DPAGT1_ALG14_if_tubular_aggregates_GFPT1_negative"],
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Congenital Myasthenic Syndrome Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(
            f"  {gene}: n={d['n']}, age_dx={d['mean_age_dx']}yr, "
            f"resp_crisis={d['respiratory_crisis_n']}/40, "
            f"oph={d['ptosis_ophthalmoplegia_n']}/40, "
            f"pyr_adverse={d['pyridostigmine_adverse_n']}/40"
        )
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
