#!/usr/bin/env python3
"""Hereditary-Periodic-Fever-Atlas — Complete 8-Gene Hereditary Periodic Fever Syndrome Atlas
(MEFV · MVK · TNFRSF1A · NLRP3 · NOD2 · PSTPIP1 · IL1RN · NLRP12).

MEFV     (Pyrin; 781 aa; 86 kDa; 16p13.3; AR (AD-modifier M694V);
          Familial Mediterranean Fever (FMF) — most common monogenic periodic fever worldwide;
          attacks 12-72 h: fever + STERILE PERITONITIS + pleuritis + synovitis;
          COLCHICINE IS MANDATORY — prevents AA amyloidosis (leading cause of death pre-colchicine);
          M694V homozygous = most severe phenotype; E148Q = mild/uncertain significance;
          seed SEED_BASE+0).
MVK      (Mevalonate kinase; 396 aa; 43 kDa; 12q24.11; AR;
          Hyper-IgD Syndrome / Mevalonate Kinase Deficiency (HIDS/MKD);
          VACCINATION-TRIGGERED ATTACKS PATHOGNOMONIC; cervical lymphadenopathy KEY FEATURE;
          urinary mevalonic acid elevated during attacks;
          canakinumab FDA-approved (2016); seed SEED_BASE+1).
TNFRSF1A (TNF receptor superfamily member 1A; 455 aa; 51 kDa; 12p13.31; AD;
          TNF Receptor-Associated Periodic Syndrome (TRAPS);
          ATTACKS >7 DAYS + MIGRATORY MYALGIA + PERIORBITAL EDEMA PATHOGNOMONIC;
          amyloidosis risk untreated; etanercept preferred (not infliximab — resistance);
          R92Q/P46L = low-penetrance variants of uncertain significance;
          seed SEED_BASE+2).
NLRP3    (NLR family pyrin domain-containing 3 / Cryopyrin; 1036 aa; 118 kDa; 1q44; AD;
          Cryopyrin-Associated Periodic Syndrome (CAPS): FCAS → Muckle-Wells → NOMID;
          NON-PRURITIC URTICARIAL RASH = PATHOGNOMONIC (not allergic urticaria);
          COLD TRIGGER for FCAS = pathognomonic; NOMID = most severe (CNS, deafness);
          canakinumab + rilonacept FDA-approved; treat NOMID urgently;
          seed SEED_BASE+3).
NOD2     (Nucleotide-binding oligomerization domain-containing protein 2; 1040 aa; 114 kDa; 16q12.1; AD;
          Blau Syndrome / Early-Onset Sarcoidosis (EOS);
          CLASSIC TRIAD: GRANULOMATOUS ARTHRITIS + UVEITIS + SKIN GRANULOMAS PATHOGNOMONIC;
          R334W/R334Q hotspot variants; methotrexate + TNF inhibitors (infliximab best);
          distinct from Crohn NOD2 variants (those are low-penetrance common variants);
          seed SEED_BASE+4).
PSTPIP1  (Proline-serine-threonine phosphatase-interacting protein 1; 416 aa; 47 kDa; 15q24.3; AD;
          PAPA Syndrome: PYOGENIC ARTHRITIS + PYODERMA GANGRENOSUM + ACNE TRIAD PATHOGNOMONIC;
          PG lesions ulcerate — misdiagnosed as wounds/infections; anakinra/canakinumab;
          corticosteroids for acute flares; IL-1 blockade for prevention;
          seed SEED_BASE+5).
IL1RN    (IL-1 receptor antagonist; 177 aa; 25 kDa; 2q14.2; AR;
          Deficiency of IL-1 Receptor Antagonist (DIRA);
          NEONATAL ONSET (first days of life): multifocal osteomyelitis + pustulosis + periostitis;
          ANAKINRA IS CURATIVE AND LIFE-SAVING (physiological IL-1Ra replacement);
          must begin urgently in neonatal period — delay causes irreversible bone destruction;
          seed SEED_BASE+6).
NLRP12   (NLR family pyrin domain-containing 12; 1062 aa; 119 kDa; 19q13.42; AD;
          FCAS2 / NLRP12-Associated Periodic Fever;
          COLD-TRIGGERED ATTACKS (like CAPS/FCAS) but typically milder — less deafness;
          often misdiagnosed as FCAS or TRAPS; colchicine partially effective;
          anakinra/canakinumab second-line; heterozygous LOF;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2542-2549).
"""

import random

SEED_BASE = 2542

FEVER_GENES = [
    # -- MEFV -- Familial Mediterranean Fever ------------------------------------------------
    {
        "gene": "MEFV",
        "alt_name": (
            "MEFV (MEFV-781aa-16p13.3 / AR (AD-modifier) -- "
            "FMF-FAMILIAL-MEDITERRANEAN-FEVER-MOST-COMMON-MONOGENIC-PERIODIC-FEVER -- "
            "COLCHICINE-MANDATORY-PREVENTS-AA-AMYLOIDOSIS-LEADING-CAUSE-DEATH-PRE-COLCHICINE -- "
            "M694V-HOMOZYGOUS-MOST-SEVERE-E148Q-MILD-UNCERTAIN-SIGNIFICANCE -- "
            "ANAKINRA-CANAKINUMAB-COLCHICINE-RESISTANT)"
        ),
        "protein": (
            "MEFV -- 16p13.3 AR (AD-modifier) -- MEFV-781aa -- "
            "Pyrin-86kDa-TRIM20-B-box-Coiled-Coil-SPRY-Domain-B30.2 -- "
            "Pyrin-Inflammasome-Assembly-Regulation-NLRP3-Independent-Pathway -- "
            "PAAND-Pyrin-Associated-Autoinflammation-Neutrophilic-Dermatosis-Gain-Function -- "
            "B30.2-Domain-Missense-Destabilises-14-3-3-Binding-Releases-Pyrin-Inhibition -- "
            "Mediterranean-Middle-Eastern-Turkish-Armenian-Jewish-North-African-Founder -- "
            "OMIM-Gene-608107-Disease-FMF-249100"
        ),
        "locus": "16p13.3",
        "protein_size": "781 aa / 86 kDa",
        "inheritance": (
            "AR (biallelic pathogenic variants); some patients with single M694V have phenotype (modifier effect); "
            "FMF: attacks 12-72 h with complete resolution between attacks; "
            "STERILE PERITONITIS — rigid abdomen mimicking surgical emergency; "
            "high fever (39-40°C), pleuritis (unilateral), synovitis (ankle > knee); "
            "M694V/M694V homozygous = most severe (highest amyloidosis risk); "
            "E148Q = mild/uncertain significance (common in general population, often benign); "
            "carrier frequency 1 in 5-7 in Mediterranean populations"
        ),
        "disease_category": (
            "Familial Mediterranean Fever (FMF) — most common monogenic periodic fever; "
            "Mediterranean/Middle Eastern populations predominantly; "
            "COLCHICINE PREVENTS AA AMYLOIDOSIS — mandatory for life; "
            "sterile peritonitis + fever attacks 12-72 h; "
            "anakinra/canakinumab for colchicine-resistant"
        ),
        "disease_pathway": (
            "Pyrin (encoded by MEFV) is expressed primarily in neutrophils, monocytes, and serosal cells. "
            "Pyrin normally assembles a pyrin inflammasome in response to bacterial toxins that inactivate Rho GTPases "
            "(e.g., Clostridioides difficile toxin B inactivates RhoA). When Rho GTPases are active, "
            "PKN1/2 kinases phosphorylate pyrin at S208 and S242, allowing 14-3-3 proteins to bind and inhibit pyrin. "
            "FMF mutations (M694V, M680I, V726A, M694I, E148Q) cluster in the B30.2/SPRY domain — "
            "the domain critical for 14-3-3 binding. Missense mutations destabilise this interaction → "
            "14-3-3 binding reduced → pyrin released → pyrin inflammasome assembles constitutively → "
            "caspase-1 activation → IL-1β and IL-18 processing and secretion → intense sterile inflammation. "
            "The mechanism is NLRP3-independent — explaining why NLRP3-specific drugs are less effective than "
            "colchicine (which inhibits microtubule-dependent pyrin inflammasome assembly). "
            "AA amyloidosis: unchecked IL-1β drives hepatic serum amyloid A (SAA) production; "
            "chronic SAA elevation → AA fibril deposition in kidney (nephrotic syndrome → renal failure) — "
            "the pre-colchicine major cause of death in FMF. Colchicine prevents attacks → prevents SAA elevation → "
            "prevents amyloidosis."
        ),
        "pathognomonic": (
            "PERIODIC FEBRILE ATTACKS 12-72 H WITH COMPLETE RESOLUTION BETWEEN ATTACKS = FMF PATHOGNOMONIC; "
            "sterile peritonitis (rigid abdomen, elevated CRP, negative cultures — misdiagnosed as appendicitis); "
            "unilateral pleuritis; ankle synovitis; M694V homozygous = highest risk; "
            "colchicine response CONFIRMS diagnosis (attacks stop within weeks to months)"
        ),
        "treatment": (
            "COLCHICINE 1-2 mg/day: FIRST-LINE AND MANDATORY FOR LIFE — prevents AA amyloidosis; "
            "abrupt colchicine discontinuation triggers attack; "
            "ANAKINRA 100 mg/day SC: colchicine-resistant FMF or amyloidosis risk reduction; "
            "CANAKINUMAB 150 mg q8w SC: FDA-approved for FMF (2016); quarterly dosing preferred over daily anakinra; "
            "rilonacept: alternative IL-1 blocker; "
            "avoid NSAIDs as primary prevention (do not prevent amyloidosis); "
            "acute attack: colchicine dose not increased during attacks (not effective acutely); "
            "NSAIDs/colchicine for acute symptom relief"
        ),
        "key_features": [
            "Most common monogenic periodic fever worldwide — high in Mediterranean populations",
            "COLCHICINE MANDATORY AND LIFE-LONG — prevents AA amyloidosis (leading cause of death pre-colchicine era)",
            "Sterile peritonitis mimics acute abdomen — up to 40% undergo appendicectomy before diagnosis",
            "M694V homozygous = most severe; M694V/M694I = intermediate; E148Q = mild/uncertain significance",
            "Attacks 12-72 h with COMPLETE RESOLUTION between attacks — key diagnostic feature",
            "COLCHICINE RESPONSE CONFIRMS diagnosis — failure to respond suggests alternative diagnosis",
            "Anakinra/canakinumab for colchicine-resistant; canakinumab FDA-approved 2016",
        ],
        "key_ddx": [
            "Appendicitis (sterile peritonitis — cultures negative; FMF resolves without surgery)",
            "TRAPS (attacks >7 days; migratory myalgia; periorbital edema — MEFV attacks shorter)",
            "HIDS/MVK (vaccination trigger; cervical lymphadenopathy; urinary mevalonate elevated)",
            "PFAPA (responds to single dose prednisolone acutely; tonsillectomy curative — no genetic basis)",
        ],
        "attack_duration_h": "12–72 hours (median 24-48h)",
        "dominant_trigger": "Stress, exercise, menses, vaccination (less common than MVK)",
        "amyloid_risk": "High if colchicine non-compliant — M694V/M694V highest risk",
        "nbs_indicated": False,
        "severity": "moderate-severe",
    },

    # -- MVK -- Hyper-IgD Syndrome / Mevalonate Kinase Deficiency --------------------------------
    {
        "gene": "MVK",
        "alt_name": (
            "MVK (MVK-396aa-12q24.11 / AR -- "
            "HIDS-MKD-HYPER-IGD-SYNDROME-MEVALONATE-KINASE-DEFICIENCY -- "
            "VACCINATION-TRIGGERED-ATTACKS-PATHOGNOMONIC -- "
            "CERVICAL-LYMPHADENOPATHY-KEY-FEATURE-DDx-FMF -- "
            "CANAKINUMAB-FDA-APPROVED-2016-URINARY-MEVALONIC-ACID-DIAGNOSTIC)"
        ),
        "protein": (
            "MVK -- 12q24.11 AR -- MVK-396aa -- "
            "Mevalonate-Kinase-43kDa-ATP-Dependent-Phosphorylation-Mevalonate-to-Mevalonate-5P -- "
            "Third-Step-Mevalonate-Pathway-Cholesterol-Isoprenoid-Biosynthesis -- "
            "Residual-Activity-1-10pct-HIDS-vs-0pct-Classical-Mevalonic-Aciduria-MVA -- "
            "V377I-Most-Common-HIDS-Western-Europe-Dutch-Founder-80pct-Alleles -- "
            "I268T-Second-Most-Common-HIDS-Allele -- "
            "OMIM-Gene-251170-Disease-HIDS-260920-MVA-610377"
        ),
        "locus": "12q24.11",
        "protein_size": "396 aa / 43 kDa",
        "inheritance": (
            "AR (biallelic pathogenic variants); "
            "HIDS (Hyper-IgD Syndrome) / MKD: residual MVK activity 1-10% (partial deficiency); "
            "Classical Mevalonic Aciduria (MVA): 0% MVK activity — more severe, dysmorphic, cerebellar ataxia; "
            "V377I founder variant in Dutch/Western European HIDS; "
            "attacks: 3-7 days with fever, lymphadenopathy, abdominal pain, diarrhea; "
            "IgD >100 IU/mL (originally defining — present in ~80% of HIDS, not specific)"
        ),
        "disease_category": (
            "Hyper-IgD Syndrome / Mevalonate Kinase Deficiency (HIDS/MKD) — "
            "impaired isoprenoid biosynthesis → IL-1β dysregulation; "
            "VACCINATION-TRIGGERED ATTACKS PATHOGNOMONIC; "
            "CERVICAL LYMPHADENOPATHY KEY DISTINGUISHING FEATURE; "
            "canakinumab FDA-approved; urinary mevalonic acid elevated during attacks"
        ),
        "disease_pathway": (
            "Mevalonate kinase (MVK) catalyses the third step of the mevalonate pathway: "
            "mevalonate + ATP → mevalonate-5-phosphate. This pathway produces isoprenoids (farnesyl-PP, "
            "geranylgeranyl-PP) required for prenylation of small GTPases (Ras, Rho, Rac) and for "
            "sterol biosynthesis. In HIDS, residual MVK activity (1-10% of normal) is sufficient for "
            "baseline isoprenoid production but decompensates during fever/stress — fever inhibits "
            "MVK activity further → isoprenoid pool collapses → unprenylated Rac1 cannot inhibit "
            "caspase-1 → IL-1β processing uninhibited → cytokine storm. "
            "The VACCINATION TRIGGER is pathognomonic: vaccines induce fever → MVK decompensation → "
            "attack within 12-24h of vaccination. URINARY MEVALONIC ACID is elevated during attacks "
            "(substrate accumulates proximal to the defective step) — a useful biomarker. "
            "IgD elevation: the original disease-defining finding (Hyper-IgD), but now understood as "
            "a secondary marker — IgD >100 IU/mL in ~80% of HIDS patients; IgA also often elevated. "
            "MVK severity spectrum: HIDS (1-10% activity) → Classical MVA (<1% activity, "
            "severe dysmorphic disease with cerebellar ataxia, intellectual disability, retinitis pigmentosa)."
        ),
        "pathognomonic": (
            "VACCINATION-TRIGGERED FEBRILE ATTACK WITHIN 12-24H OF VACCINE = HIDS PATHOGNOMONIC; "
            "cervical lymphadenopathy (>2 cm, tender) distinguishes from FMF (no lymphadenopathy); "
            "urinary mevalonic acid elevated during attack (can be normal between attacks); "
            "IgD >100 IU/mL + IgA elevated; serum IgD may be normal in young children"
        ),
        "treatment": (
            "CANAKINUMAB 4 mg/kg (max 300 mg) q8w SC: FDA-approved (2016) — most effective; "
            "ANAKINRA 1-2 mg/kg/day SC: IL-1 blocker — effective for acute and prevention; "
            "SIMVASTATIN: may reduce attack frequency (increases isoprenoid flux) — second-line; "
            "NSAIDs/corticosteroids: limited acute benefit; "
            "etanercept (TNF blockade): modest benefit in some patients; "
            "COLCHICINE: generally NOT effective (unlike FMF — different inflammasome pathway); "
            "pre-vaccination anakinra cover: single dose 48h before/after vaccination reduces risk; "
            "avoid vaccinations during active flare"
        ),
        "key_features": [
            "VACCINATION-TRIGGERED ATTACKS PATHOGNOMONIC — attack within 12-24h of any vaccine",
            "CERVICAL LYMPHADENOPATHY (2+ cm, tender) KEY distinguishing feature from FMF",
            "Urinary mevalonic acid elevated during attack — diagnostic biomarker",
            "IgD >100 IU/mL in 80% — not specific; may be normal in young children",
            "V377I Dutch/Western European founder — most common HIDS allele",
            "COLCHICINE NOT effective (MVK not pyrin pathway); canakinumab is first-line",
            "Pre-vaccination cover: anakinra 48h before vaccination reduces vaccine-triggered attacks",
        ],
        "key_ddx": [
            "FMF (no lymphadenopathy; shorter attacks 12-72h vs 3-7 days; MEFV mutations)",
            "TRAPS (attacks >7 days; migratory myalgia; periorbital edema; TNFRSF1A mutations)",
            "PFAPA (responds to single prednisolone dose; tonsillectomy curative; no gene mutation)",
            "Classical MVA (<1% MVK activity; severe dysmorphic disease; cerebellar ataxia — more severe)",
        ],
        "attack_duration_h": "3–7 days (72-168h)",
        "dominant_trigger": "VACCINATION (most specific trigger), fever from any cause",
        "amyloid_risk": "Lower than FMF — rare AA amyloidosis",
        "nbs_indicated": False,
        "severity": "moderate",
    },

    # -- TNFRSF1A -- TRAPS -----------------------------------------------------------------------
    {
        "gene": "TNFRSF1A",
        "alt_name": (
            "TNFRSF1A (TNFRSF1A-455aa-12p13.31 / AD -- "
            "TRAPS-TNF-RECEPTOR-ASSOCIATED-PERIODIC-SYNDROME -- "
            "ATTACKS->7-DAYS-MIGRATORY-MYALGIA-PERIORBITAL-EDEMA-PATHOGNOMONIC -- "
            "ETANERCEPT-PREFERRED-NOT-INFLIXIMAB-ANTI-DRUG-ANTIBODIES -- "
            "R92Q-P46L-LOW-PENETRANCE-VARIANTS-UNCERTAIN-SIGNIFICANCE)"
        ),
        "protein": (
            "TNFRSF1A -- 12p13.31 AD -- TNFRSF1A-455aa -- "
            "TNF-Receptor-Superfamily-Member-1A-TNFR1-55kDa-Type-I-Transmembrane -- "
            "CRD1-CRD2-CRD3-CRD4-Cysteine-Rich-Domains-Extracellular -- "
            "C30S-C33Y-T50M-C70S-C88G-R92Q-P46L-Missense-Cluster-CRDs -- "
            "High-Penetrance-Cys-Substitutions-CRD2-CRD3-vs-Low-Penetrance-R92Q-P46L -- "
            "Misfolded-TNFR1-Retained-ER-Not-Secreted-Endogenous-TNF-Not-Neutralised -- "
            "OMIM-Gene-191190-Disease-TRAPS-142680"
        ),
        "locus": "12p13.31",
        "protein_size": "455 aa / 51 kDa",
        "inheritance": (
            "AD (autosomal dominant); "
            "HIGH-PENETRANCE variants: Cys-substitutions in CRD2/CRD3 (C30S, C33Y, C70S, C88G, T50M) — "
            "severe TRAPS with high amyloidosis risk; "
            "LOW-PENETRANCE variants: R92Q and P46L — very common in general population; "
            "clinical significance uncertain; may behave as 'carriers' or disease modifiers, "
            "not always pathogenic; R92Q frequency ~1% in European populations; "
            "attacks: typically >7 days (3-28 days), can last weeks; "
            "amyloidosis risk without treatment, especially with Cys-substitution variants"
        ),
        "disease_category": (
            "TNF Receptor-Associated Periodic Syndrome (TRAPS) — "
            "AD periodic fever with MIGRATORY MYALGIA + PERIORBITAL EDEMA + LONG ATTACKS >7 DAYS; "
            "etanercept preferred (not infliximab — anti-drug antibodies develop); "
            "canakinumab FDA-approved; AA amyloidosis risk"
        ),
        "disease_pathway": (
            "TNFRSF1A encodes TNFR1 (TNF receptor 1), the major signalling receptor for TNF-α, expressed "
            "ubiquitously. Normally, membrane-bound TNFR1 is proteolytically shed by TACE/ADAM17, releasing "
            "soluble TNFR1 (sTNFR1) which acts as a decoy receptor to neutralise circulating TNF. "
            "TRAPS pathomechanism (dominant-negative model): "
            "Missense mutations in the cysteine-rich domains (CRDs) cause TNFR1 misfolding → "
            "misfolded TNFR1 is retained in the ER → not expressed at cell surface → not shed → "
            "sTNFR1 levels are LOW → endogenous TNF-α not neutralised → unopposed TNF signalling. "
            "Additionally, ER-retained misfolded TNFR1 induces ER stress and reactive oxygen species → "
            "activates NF-κB and MAPK inflammatory pathways independently of TNF-ligand binding. "
            "MIGRATORY MYALGIA: characteristic of TRAPS — overlying fascia oedema with monocyte/macrophage "
            "infiltration causing the characteristic skin erythema that 'migrates' centripetally; "
            "muscle biopsy shows perifascial oedema, not myositis proper. "
            "PERIORBITAL OEDEMA: distinctive — caused by periorbital fat inflammation; "
            "misdiagnosed as allergic reaction or angioedema. "
            "ETANERCEPT vs INFLIXIMAB: etanercept (soluble TNFR2-Fc fusion) replaces the missing sTNFR1 "
            "decoy function; infliximab (anti-TNF monoclonal) triggers paradoxical reactions in TRAPS "
            "and anti-drug antibody development — NOT recommended."
        ),
        "pathognomonic": (
            "ATTACKS >7 DAYS + MIGRATORY MYALGIA + PERIORBITAL EDEMA = TRAPS PATHOGNOMONIC TRIAD; "
            "myalgia with overlying centripetal skin erythema mimicking cellulitis; "
            "periorbital edema misdiagnosed as allergy; "
            "low soluble TNFR1 during attacks (sTNFR1 <1 ng/mL = supporting finding); "
            "HIGH-PENETRANCE Cys-substitutions: severe, amyloidosis risk; "
            "R92Q/P46L: uncertain significance — clinical correlation required"
        ),
        "treatment": (
            "ETANERCEPT 25 mg SC 2x/week: preferred TNF blocker (replaces deficient sTNFR1 decoy); "
            "CANAKINUMAB 150 mg q8w SC: FDA-approved 2016 — excellent for Cys-variant severe TRAPS; "
            "ANAKINRA 100 mg/day SC: IL-1 blockade effective; "
            "DO NOT USE INFLIXIMAB (anti-TNF monoclonal — anti-drug antibodies + paradoxical reactions); "
            "NSAIDs for mild attacks; "
            "corticosteroids for acute severe attacks (not long-term); "
            "R92Q/P46L: treat based on attack burden, not just genotype (low penetrance)"
        ),
        "key_features": [
            "ATTACKS >7 DAYS (longest of all monogenic periodic fevers) — key distinguishing feature",
            "MIGRATORY MYALGIA with overlying fascia erythema PATHOGNOMONIC — misdiagnosed as cellulitis",
            "PERIORBITAL EDEMA PATHOGNOMONIC — misdiagnosed as allergy/angioedema",
            "Etanercept PREFERRED — replaces deficient soluble TNFR1 decoy neutralisation of TNF",
            "INFLIXIMAB NOT RECOMMENDED — anti-drug antibodies + paradoxical flares in TRAPS",
            "Cys-substitutions (C30S, C33Y, C70S) = high penetrance, high amyloidosis risk",
            "R92Q/P46L = LOW penetrance — present in 1% of Europeans; clinical significance uncertain",
        ],
        "key_ddx": [
            "Cellulitis (myalgia erythema in TRAPS migrates centripetally; no response to antibiotics)",
            "FMF (shorter attacks 12-72h; no migratory myalgia; no periorbital edema; MEFV mutations)",
            "Still's disease/systemic JIA (quotidian fever rash; arthritis; ferritin >1000)",
            "HIDS (vaccination trigger; lymphadenopathy; urinary mevalonic acid elevated)",
        ],
        "attack_duration_h": ">7 days (168h+); may last 3-28 days",
        "dominant_trigger": "Stress, minor infections, exercise, temperature change",
        "amyloid_risk": "High (Cys-substitution variants) — etanercept/canakinumab reduces risk",
        "nbs_indicated": False,
        "severity": "moderate-severe",
    },

    # -- NLRP3 -- CAPS (FCAS/Muckle-Wells/NOMID) ------------------------------------------------
    {
        "gene": "NLRP3",
        "alt_name": (
            "NLRP3 (NLRP3-1036aa-1q44 / AD -- "
            "CAPS-CRYOPYRIN-ASSOCIATED-PERIODIC-SYNDROME-FCAS-MUCKLE-WELLS-NOMID -- "
            "NON-PRURITIC-URTICARIAL-RASH-PATHOGNOMONIC-NOT-ALLERGIC-URTICARIA -- "
            "COLD-TRIGGER-FCAS-PATHOGNOMONIC -- "
            "CANAKINUMAB-RILONACEPT-FDA-APPROVED-NOMID-TREAT-URGENTLY)"
        ),
        "protein": (
            "NLRP3 -- 1q44 AD -- NLRP3-1036aa -- "
            "Cryopyrin-NLRP3-118kDa-PYD-NACHT-LRR-Domain-Architecture -- "
            "NLRP3-Inflammasome-ASC-Caspase-1-IL1B-IL18-Processing -- "
            "GOF-NLRP3-Constitutive-Inflammasome-Assembly-Without-Danger-Signal -- "
            "FCAS-Mild-R260W-CIAS1/CAPS1-Cold-Triggered -- "
            "Muckle-Wells-T350M-V198M-Progressive-SNHL-Amyloidosis -- "
            "NOMID-CINCA-Severe-D303N-E311K-CNS-Papilloedema-Epiphyseal-Overgrowth -- "
            "OMIM-Gene-606416-Disease-FCAS-120100-MWS-191900-NOMID-607115"
        ),
        "locus": "1q44",
        "protein_size": "1036 aa / 118 kDa",
        "inheritance": (
            "AD (autosomal dominant, gain-of-function); de novo mutations common in NOMID; "
            "CAPS severity spectrum (same gene — variant position determines severity): "
            "FCAS (Familial Cold Autoinflammatory Syndrome): mildest; cold-triggered; urticaria; no deafness; "
            "Muckle-Wells Syndrome: intermediate; progressive SNHL; episodic urticaria/fever; amyloidosis; "
            "NOMID/CINCA: most severe; neonatal onset; CNS involvement; papilloedema; epiphyseal overgrowth; "
            "same gene, same inflammasome pathway — IL-1β blockade effective across all three"
        ),
        "disease_category": (
            "Cryopyrin-Associated Periodic Syndrome (CAPS) — FCAS → Muckle-Wells → NOMID; "
            "NON-PRURITIC URTICARIAL RASH PATHOGNOMONIC (not allergic); "
            "COLD TRIGGER for FCAS; "
            "canakinumab FDA-approved; rilonacept FDA-approved; "
            "NOMID requires urgent treatment to prevent irreversible CNS damage and deafness"
        ),
        "disease_pathway": (
            "NLRP3 encodes Cryopyrin, the sensor component of the NLRP3 inflammasome. "
            "The NLRP3 inflammasome is a cytosolic multiprotein complex that assembles in response to "
            "danger signals (ATP, crystals, microbial products) via oligomerisation of NLRP3, "
            "the adaptor ASC (PYCARD), and caspase-1. Activated caspase-1 cleaves pro-IL-1β and pro-IL-18 "
            "to their active forms, driving systemic inflammation. "
            "CAPS gain-of-function mutations (predominantly in the NACHT domain): "
            "constitutive NLRP3 conformational change → inflammasome assembles without danger signal → "
            "continuous caspase-1 activation → continuous IL-1β and IL-18 secretion. "
            "COLD TRIGGER (FCAS): cold stress activates TRPM8/TRPA1 channels in peripheral tissues → "
            "intracellular signals that further lower the threshold for mutant NLRP3 activation → "
            "systemic urticaria + fever within 1-2 h of cold exposure. "
            "URTICARIAL RASH: neutrophil-rich perivascular infiltrate in dermis → non-pruritic urticaria; "
            "antihistamines are INEFFECTIVE (not mast-cell-mediated IgE mechanism — neutrophilic). "
            "NOMID (Neonatal-onset Multisystem Inflammatory Disease): "
            "constitutive CNS inflammation → meningitis → papilloedema → sensorineural hearing loss → "
            "epiphyseal overgrowth from IL-1β-driven chondrocyte activation → dysmorphic facies. "
            "TREATMENT: canakinumab (anti-IL-1β) or rilonacept (IL-1 trap) — both FDA-approved for CAPS; "
            "anakinra also effective but daily injections less convenient."
        ),
        "pathognomonic": (
            "NON-PRURITIC URTICARIAL RASH + FEVER = CAPS PATHOGNOMONIC (not allergic urticaria); "
            "COLD TRIGGER within 1-2h of cold exposure (urticaria + fever) = FCAS PATHOGNOMONIC; "
            "progressive SNHL + episodic urticaria/fever = Muckle-Wells; "
            "neonatal onset + papilloedema + epiphyseal overgrowth = NOMID; "
            "antihistamines INEFFECTIVE (neutrophilic, not IgE mast-cell) — confirms non-allergic nature"
        ),
        "treatment": (
            "CANAKINUMAB 150 mg q8w SC (adult): FDA-approved — anti-IL-1β monoclonal, quarterly dosing; "
            "RILONACEPT 160 mg SC loading then 80-160 mg weekly: FDA-approved IL-1 trap; "
            "ANAKINRA 1-2 mg/kg/day SC: effective; daily injections less convenient; "
            "DO NOT treat with antihistamines (urticaria is neutrophilic, not IgE-mediated — ineffective); "
            "colchicine: NOT effective for CAPS (NLRP3 does not rely on microtubule-dependent assembly); "
            "NOMID: start treatment URGENTLY — irreversible CNS damage and deafness occur without treatment; "
            "ophthalmology review for papilloedema; audiology for SNHL monitoring"
        ),
        "key_features": [
            "NON-PRURITIC URTICARIAL RASH PATHOGNOMONIC — misdiagnosed as allergic urticaria; antihistamines INEFFECTIVE",
            "COLD TRIGGER: urticaria + fever within 1-2h of cold exposure = FCAS PATHOGNOMONIC",
            "Three severity tiers (same gene): FCAS (mild) → Muckle-Wells (intermediate SNHL) → NOMID (severe CNS)",
            "Canakinumab + rilonacept FDA-approved; colchicine NOT effective (different pathway to FMF)",
            "NOMID: treat URGENTLY — papilloedema and SNHL are irreversible without treatment",
            "De novo NLRP3 mutations common in NOMID — no family history does NOT exclude",
            "Progressive SNHL in Muckle-Wells — monitor audiology 6-monthly; cochlear implant if severe",
        ],
        "key_ddx": [
            "Allergic urticaria (pruritic, antihistamine-responsive; CAPS urticaria is non-pruritic, antihistamine-INEFFECTIVE)",
            "Schnitzler syndrome (IgM/IgG monoclonal gammopathy + urticaria; acquired; NLRP3 not mutated)",
            "Angioedema/SERPING1-HAE (swelling without urticaria; cold water trigger differs from cold air FCAS)",
            "TRAPS (migratory myalgia; longer attacks; no cold trigger; TNFRSF1A mutation)",
        ],
        "attack_duration_h": "FCAS: 12-24h (cold-triggered); Muckle-Wells: 24-48h; NOMID: continuous",
        "dominant_trigger": "Cold exposure (FCAS), stress, physical exertion",
        "amyloid_risk": "Muckle-Wells: amyloidosis risk; NOMID: CNS damage risk; FCAS: low amyloid risk",
        "nbs_indicated": False,
        "severity": "mild-to-severe",
    },

    # -- NOD2 -- Blau Syndrome / Early-Onset Sarcoidosis ------------------------------------------
    {
        "gene": "NOD2",
        "alt_name": (
            "NOD2 (NOD2-1040aa-16q12.1 / AD -- "
            "BLAU-SYNDROME-EARLY-ONSET-SARCOIDOSIS -- "
            "GRANULOMATOUS-ARTHRITIS-UVEITIS-SKIN-GRANULOMAS-TRIAD-PATHOGNOMONIC -- "
            "R334W-R334Q-HOTSPOT-NACHT-DOMAIN -- "
            "METHOTREXATE-TNF-INHIBITORS-INFLIXIMAB-BEST-EVIDENCE)"
        ),
        "protein": (
            "NOD2 -- 16q12.1 AD -- NOD2-1040aa -- "
            "Nucleotide-Binding-Oligomerization-Domain-2-114kDa-CARD15 -- "
            "Two-N-Terminal-CARD-Domains-NACHT-Domain-LRR-Ligand-Sensing -- "
            "GOF-NACHT-Domain-Spontaneous-NF-kB-Activation -- "
            "Distinct-From-Crohn-NOD2-Risk-Alleles-Fs-Mutations-LOF -- "
            "MDP-Muramyl-Dipeptide-Sensor-Bacterial-Peptidoglycan -- "
            "R334W-R334Q-NACHT-Domain-Hotspot-Blau -- "
            "OMIM-Gene-605956-Disease-Blau-186580"
        ),
        "locus": "16q12.1",
        "protein_size": "1040 aa / 114 kDa",
        "inheritance": (
            "AD (autosomal dominant, gain-of-function); "
            "Blau syndrome: R334W, R334Q, M513T, H496L — GOF mutations in NACHT domain → "
            "constitutive NF-κB activation → granuloma formation; "
            "DISTINCT from Crohn NOD2 risk variants (R702W, G908R, fs1007insC) — those are LOF, "
            "common in general population, low individual risk; "
            "Blau = early-onset childhood granulomatosis (onset <4 years); "
            "EOS = sporadic de novo (same gene, same mutations — no family history)"
        ),
        "disease_category": (
            "Blau Syndrome / Early-Onset Sarcoidosis (EOS) — "
            "GRANULOMATOUS ARTHRITIS + UVEITIS + SKIN GRANULOMAS TRIAD PATHOGNOMONIC; "
            "AD; onset <4 years; R334W/R334Q hotspot; "
            "methotrexate + TNF inhibitors (infliximab best evidence); "
            "NOT Crohn disease (those NOD2 variants are LOF, low penetrance)"
        ),
        "disease_pathway": (
            "NOD2 encodes an intracellular pattern-recognition receptor that senses muramyl dipeptide (MDP), "
            "a component of bacterial peptidoglycan. Wildtype NOD2 is autoinhibited until MDP binding → "
            "NACHT domain oligomerisation → RIP2 recruitment → NF-κB and MAPK activation → innate immune response. "
            "BLAU GAIN-OF-FUNCTION: Missense mutations in the NACHT domain (R334W, R334Q, M513T) alter the "
            "auto-inhibitory conformation → constitutive NOD2 oligomerisation without MDP binding → "
            "constitutive NF-κB activation → pro-inflammatory cytokines (TNF-α, IL-12, IL-6) → granuloma formation. "
            "GRANULOMA HISTOLOGY: non-caseating epithelioid granulomas with multinucleated giant cells — "
            "identical to sarcoidosis histologically, but in a child (<4 years) with AD inheritance = Blau. "
            "THREE-TISSUE TRIAD: "
            "(1) Arthritis: boggy polyarthritis of wrists/ankles/knees with rice-body synovium; "
            "(2) Uveitis: chronic anterior/pan-uveitis — vision-threatening; requires ophthalmology follow-up; "
            "(3) Skin: tan/cream papular rash — 2-4mm papules with granuloma on biopsy — precedes arthritis. "
            "CONTRAST WITH CROHN NOD2 VARIANTS: Crohn NOD2 alleles are LOF frameshift/missense "
            "(Fs1007insC, R702W, G908R) reducing MDP sensing → impaired epithelial barrier → gut inflammation; "
            "these are common polymorphisms with modest individual Crohn risk — completely different from Blau GOF."
        ),
        "pathognomonic": (
            "ONSET <4 YEARS + GRANULOMATOUS ARTHRITIS + UVEITIS + SKIN GRANULOMAS = BLAU PATHOGNOMONIC; "
            "skin biopsy: non-caseating granuloma (confirms granulomatous nature in child); "
            "positive NOD2 NACHT-domain GOF variant (R334W/R334Q most common); "
            "distinguish from juvenile idiopathic arthritis (no granulomas in JIA); "
            "distinguish from childhood sarcoidosis by AD family history and NOD2 mutation (EOS = sporadic Blau)"
        ),
        "treatment": (
            "METHOTREXATE 10-15 mg/m²/week: first-line steroid-sparing; "
            "INFLIXIMAB: best evidence for Blau among TNF inhibitors (TNF-α is the dominant cytokine in granuloma); "
            "ETANERCEPT: less effective than infliximab for granulomatous disease (soluble receptor less effective); "
            "ADALIMUMAB: alternative TNF inhibitor if infliximab failure; "
            "CORTICOSTEROIDS: prednisolone for acute flares (not long-term — steroid toxicity in children); "
            "TOPICAL OPHTHALMIC STEROIDS + ophthalmology every 3 months (uveitis monitoring); "
            "canakinumab/anakinra: considered if TNF failure (limited evidence); "
            "colchicine: NOT effective for Blau (different pathway)"
        ),
        "key_features": [
            "CLASSIC TRIAD: GRANULOMATOUS ARTHRITIS + UVEITIS + SKIN GRANULOMAS PATHOGNOMONIC",
            "Onset <4 years — early childhood; sporadic = EOS (de novo), familial = Blau (AD)",
            "R334W/R334Q NACHT hotspot — most common Blau variants",
            "DISTINCT from Crohn NOD2 variants (those are LOF, common, low penetrance — NOT Blau)",
            "Skin biopsy: non-caseating granuloma — confirms diagnosis and differentiates from JIA",
            "Uveitis vision-threatening — ophthalmology follow-up every 3 months mandatory",
            "Infliximab BEST TNF inhibitor evidence for Blau (etanercept less effective for granulomas)",
        ],
        "key_ddx": [
            "Juvenile idiopathic arthritis JIA (no granulomas; no skin granuloma rash; no uveitis-rash-arthritis triad)",
            "Sarcoidosis (adult onset, rare in <4 years; sporadic; no NOD2 GOF)",
            "TRAPS (no granulomas; migratory myalgia; periorbital edema; TNFRSF1A mutation)",
            "Crohn disease NOD2 variants (LOF not GOF; gut inflammation not granulomatous triad)",
        ],
        "attack_duration_h": "Chronic continuous inflammation (not episodic fever attacks)",
        "dominant_trigger": "No clear trigger — chronic autoinflammatory process",
        "amyloid_risk": "Low (uncommon in Blau)",
        "nbs_indicated": False,
        "severity": "moderate-severe",
    },

    # -- PSTPIP1 -- PAPA Syndrome ----------------------------------------------------------------
    {
        "gene": "PSTPIP1",
        "alt_name": (
            "PSTPIP1 (PSTPIP1-416aa-15q24.3 / AD -- "
            "PAPA-SYNDROME-PYOGENIC-ARTHRITIS-PYODERMA-GANGRENOSUM-ACNE-TRIAD-PATHOGNOMONIC -- "
            "PG-LESIONS-ULCERATE-MISDIAGNOSED-AS-INFECTION -- "
            "ANAKINRA-CANAKINUMAB-IL1-BLOCKADE -- "
            "A230T-E250K-HOTSPOT)"
        ),
        "protein": (
            "PSTPIP1 -- 15q24.3 AD -- PSTPIP1-416aa -- "
            "Proline-Serine-Threonine-Phosphatase-Interacting-Protein-1-CD2BP1-47kDa -- "
            "FCH-F-BAR-Domain-SH3-Domain-PEST-Domain -- "
            "Scaffold-Protein-Actin-Cytoskeleton-Immunological-Synapse -- "
            "Hyperphosphorylation-Prevents-PSTPIP1-Pyrin-Interaction -- "
            "PAPA-Mutations-Increase-Pyrin-Binding-Aberrant-Pyrin-Inflammasome-Activation -- "
            "A230T-E250K-Most-Common-PAPA-Mutations -- "
            "OMIM-Gene-606347-Disease-PAPA-604416"
        ),
        "locus": "15q24.3",
        "protein_size": "416 aa / 47 kDa",
        "inheritance": (
            "AD (autosomal dominant); "
            "PAPA syndrome: A230T, E250K, E277D most common variants; "
            "PSTPIP1 mutations increase binding to pyrin → aberrant pyrin inflammasome assembly → "
            "caspase-1 activation → IL-1β → joint and skin inflammation; "
            "PAPA = PYOGENIC ARTHRITIS + PYODERMA GANGRENOSUM + ACNE; "
            "arthritis and PG can be destructive; onset childhood (arthritis) to adolescence/adulthood (PG)"
        ),
        "disease_category": (
            "PAPA Syndrome — PYOGENIC ARTHRITIS + PYODERMA GANGRENOSUM + ACNE TRIAD PATHOGNOMONIC; "
            "AD; A230T/E250K PSTPIP1 hotspot variants; "
            "IL-1 blockade (anakinra/canakinumab) effective; "
            "PG lesions misdiagnosed as infection — debridement worsens PG (pathergy)"
        ),
        "disease_pathway": (
            "PSTPIP1 (also known as CD2BP1) is an adaptor/scaffold protein expressed in haematopoietic cells, "
            "particularly T cells and neutrophils. It contains F-BAR, SH3, and PEST domains and normally "
            "interacts with CD2, WASP, PSTPIP2, and — importantly — pyrin (MEFV). "
            "Wildtype interaction: PSTPIP1 is regulated by Tyr-phosphorylation; phospho-PSTPIP1 has reduced "
            "pyrin binding, limiting pyrin inflammasome activity. "
            "PAPA mutations (A230T, E250K): increased PSTPIP1–pyrin binding affinity → "
            "enhanced pyrin inflammasome assembly → caspase-1 constitutive activation → IL-1β → "
            "pyogenic (sterile neutrophilic) inflammation in joints and skin. "
            "PYOGENIC ARTHRITIS: destructive arthritis with purulent synovial fluid (culture-negative) — "
            "misdiagnosed as septic arthritis; surgical washout + antibiotics without effect. "
            "PYODERMA GANGRENOSUM (PG): begins as painful red papule/pustule at site of minor trauma → "
            "ulcerates with undermined violaceous borders — PATHERGY (trauma triggers or worsens PG) → "
            "debridement WORSENS PG (pathergy effect); biopsy shows neutrophilic infiltrate. "
            "ACNE CONGLOBATA or nodulocystic acne: severe inflammatory acne; often responds to standard acne "
            "treatment but flares with PAPA autoinflammatory attacks."
        ),
        "pathognomonic": (
            "PYOGENIC ARTHRITIS + PYODERMA GANGRENOSUM + ACNE (PAPA TRIAD) PATHOGNOMONIC; "
            "culture-negative purulent joint fluid (sterile arthritis misdiagnosed as septic); "
            "PG with pathergy (trauma triggers ulceration — debridement worsens it) PATHOGNOMONIC; "
            "A230T/E250K PSTPIP1 variants confirm diagnosis; "
            "IL-1 blockade response (anakinra) confirms autoinflammatory mechanism"
        ),
        "treatment": (
            "ANAKINRA 100 mg/day SC: first-line IL-1 blocker — effective for arthritis and PG; "
            "CANAKINUMAB 150 mg q8w SC: quarterly dosing option; "
            "CORTICOSTEROIDS: high-dose IV methylprednisolone for acute severe PG or arthritis flare; "
            "for PG specifically: AVOID DEBRIDEMENT — pathergy worsens ulceration; "
            "topical/intralesional steroids + compression for PG; "
            "acne: isotretinoin + standard management; biologic cover during isotretinoin; "
            "INFLIXIMAB: reported benefit in PG (non-PAPA also); "
            "colchicine: NOT effective for PAPA"
        ),
        "key_features": [
            "PAPA TRIAD (PYOGENIC ARTHRITIS + PYODERMA GANGRENOSUM + ACNE) PATHOGNOMONIC",
            "Culture-negative purulent joint fluid — misdiagnosed as septic arthritis; antibiotics ineffective",
            "PG PATHERGY: minor trauma triggers/worsens PG — AVOID DEBRIDEMENT (worsens ulceration)",
            "A230T/E250K PSTPIP1 hotspot — increased pyrin binding → pyrin inflammasome activation",
            "IL-1 blockade (anakinra, canakinumab) effective — confirms autoinflammatory mechanism",
            "PG violaceous undermined borders — biopsy: sterile neutrophilic infiltrate",
            "Colchicine NOT effective (PSTPIP1 → pyrin inflammasome pathway differs from FMF)",
        ],
        "key_ddx": [
            "Septic arthritis (PAPA arthritis culture-negative; antibiotics ineffective; IL-1 responds)",
            "Hidradenitis suppurativa (similar distribution but no arthritis; different pathology)",
            "Cutaneous Crohn (granulomatous not neutrophilic; bowel disease usually present)",
            "Behçet disease (oral/genital ulcers; uveitis; pathergy also present; HLA-B51)",
        ],
        "attack_duration_h": "Variable — PG can be chronic; arthritis attacks days-weeks",
        "dominant_trigger": "Minor trauma/pathergy for PG; infections for arthritis",
        "amyloid_risk": "Low",
        "nbs_indicated": False,
        "severity": "moderate-severe",
    },

    # -- IL1RN -- DIRA ---------------------------------------------------------------------------
    {
        "gene": "IL1RN",
        "alt_name": (
            "IL1RN (IL1RN-177aa-2q14.2 / AR -- "
            "DIRA-DEFICIENCY-IL1-RECEPTOR-ANTAGONIST -- "
            "NEONATAL-ONSET-MULTIFOCAL-OSTEOMYELITIS-PUSTULOSIS-PERIOSTITIS -- "
            "ANAKINRA-CURATIVE-AND-LIFE-SAVING-PHYSIOLOGICAL-IL1RA-REPLACEMENT -- "
            "PUERTO-RICAN-NEWFOUNDLAND-DUTCH-FOUNDERS)"
        ),
        "protein": (
            "IL1RN -- 2q14.2 AR -- IL1RN-177aa -- "
            "IL-1-Receptor-Antagonist-IL1Ra-25kDa-Secreted-Form-IL1RA-Isoform-1 -- "
            "Endogenous-Competitive-IL-1R1-Blocker-No-Agonist-Activity -- "
            "Blocks-Both-IL-1alpha-and-IL1beta-At-IL1-Receptor-1 -- "
            "DIRA-Biallelic-LOF-No-IL1Ra-Unopposed-IL-1-Activity -- "
            "Large-Deletions-Frameshift-Most-Common-Mechanism -- "
            "Puerto-Rican-Newfoundland-Dutch-Founder-Deletions -- "
            "OMIM-Gene-147679-Disease-DIRA-612852"
        ),
        "locus": "2q14.2",
        "protein_size": "177 aa / 25 kDa (secreted isoform 1)",
        "inheritance": (
            "AR (biallelic loss-of-function); "
            "DIRA: complete absence of IL-1 receptor antagonist → unopposed IL-1α and IL-1β activity; "
            "neonatal onset (first days to weeks of life); "
            "Puerto Rican founder deletion (>30% of DIRA patients); "
            "Newfoundland and Dutch founder deletions also described; "
            "systemic neonatal multifocal sterile osteomyelitis + pustular skin eruption + periostitis"
        ),
        "disease_category": (
            "Deficiency of IL-1 Receptor Antagonist (DIRA) — "
            "NEONATAL ONSET multifocal sterile osteomyelitis + pustulosis + periostitis; "
            "ANAKINRA IS CURATIVE AND LIFE-SAVING (IL-1Ra replacement — the missing protein); "
            "delay causes irreversible bone destruction; AR biallelic LOF"
        ),
        "disease_pathway": (
            "IL-1 receptor antagonist (IL-1Ra), encoded by IL1RN, is the endogenous competitive antagonist of "
            "the IL-1 receptor type 1 (IL-1R1). IL-1Ra binds IL-1R1 with high affinity but induces no "
            "intracellular signalling — it simply blocks IL-1α and IL-1β from binding and activating IL-1R1. "
            "Physiological role: IL-1Ra is constitutively secreted by monocytes, neutrophils, and epithelial cells "
            "to limit IL-1 signalling duration and intensity. "
            "DIRA pathomechanism: Biallelic LOF of IL1RN → complete absence of IL-1Ra → "
            "both IL-1α and IL-1β bind IL-1R1 without competition → "
            "UNOPPOSED IL-1 SIGNALLING across all IL-1R1-expressing tissues → "
            "neutrophilic sterile inflammation in bone marrow (osteomyelitis), periosteum (periostitis), "
            "and skin (pustulosis). "
            "NEONATAL PRESENTATION: IL-1 is critical for normal immune homeostasis from birth; "
            "without IL-1Ra, the normal inflammatory responses from birth (birth canal, neonatal feeding) "
            "become uncontrolled → multifocal osteomyelitis (ribs, vertebrae, long bones) within first days of life. "
            "ANAKINRA (recombinant IL-1Ra): provides the missing physiological protein → "
            "immediate resolution of inflammation and bone pain → CURATIVE and life-saving; "
            "bone lesions stabilise/heal with prompt treatment; "
            "untreated DIRA causes severe bone deformation and can be fatal. "
            "IMPORTANT: anakinra must be continued life-long (IL1RN biallelic — cannot produce IL-1Ra naturally)."
        ),
        "pathognomonic": (
            "NEONATAL MULTIFOCAL STERILE OSTEOMYELITIS + PUSTULOSIS + PERIOSTITIS = DIRA PATHOGNOMONIC; "
            "bone lesions on X-ray/MRI: periosteal reaction (ribs, vertebrae, long bones); "
            "skin: pustular eruption with sterile neutrophilic infiltrate on biopsy; "
            "undetectable IL-1Ra in serum (diagnostic); "
            "ANAKINRA RESPONSE within 24-48h confirms diagnosis — dramatic improvement"
        ),
        "treatment": (
            "ANAKINRA 1-4 mg/kg/day SC (neonatal dose): CURATIVE AND LIFE-SAVING — "
            "physiological IL-1Ra replacement; start URGENTLY in neonatal period; "
            "dose increased as patient grows; continuous life-long treatment required; "
            "CANAKINUMAB: alternative if anakinra unavailable; "
            "NO ROLE for colchicine, steroids alone, NSAIDs (IL-1Ra replacement is the mechanism); "
            "bone lesions: heal with IL-1 blockade — orthopaedic surgery rarely needed if treated early; "
            "multidisciplinary: neonatology, rheumatology, orthopaedics, dermatology"
        ),
        "key_features": [
            "NEONATAL ONSET (first days of life) — multifocal sterile osteomyelitis + pustulosis + periostitis",
            "ANAKINRA IS CURATIVE AND LIFE-SAVING — physiological IL-1Ra replacement (the missing protein itself)",
            "Delay in diagnosis causes IRREVERSIBLE bone destruction and deformity",
            "AR biallelic LOF — complete absence of IL-1Ra → UNOPPOSED IL-1 signalling",
            "Puerto Rican / Newfoundland / Dutch founder large deletions — ancestry clue",
            "Undetectable serum IL-1Ra confirms diagnosis; anakinra response within 24-48h confirms",
            "Life-long anakinra required — patient cannot produce endogenous IL-1Ra",
        ],
        "key_ddx": [
            "Neonatal septic arthritis/osteomyelitis (cultures NEGATIVE in DIRA — sterile; antibiotics ineffective)",
            "Caffey disease (infantile cortical hyperostosis — neonatal periostitis; IL1A mutation; different genetics)",
            "Congenital infections (TORCH screen negative in DIRA; pustulosis sterile on culture)",
            "CAMP syndrome (NLRP3 somatic; later onset; not biallelic IL1RN)",
        ],
        "attack_duration_h": "Continuous from neonatal period (not episodic)",
        "dominant_trigger": "Continuous — no episodic trigger",
        "amyloid_risk": "Low (prompt treatment prevents chronic inflammation)",
        "nbs_indicated": True,
        "severity": "severe",
    },

    # -- NLRP12 -- FCAS2 / NLRP12-AU -------------------------------------------------------------
    {
        "gene": "NLRP12",
        "alt_name": (
            "NLRP12 (NLRP12-1062aa-19q13.42 / AD -- "
            "FCAS2-NLRP12-ASSOCIATED-AUTOINFLAMMATION -- "
            "COLD-TRIGGERED-ATTACKS-LIKE-CAPS-BUT-MILDER-LESS-DEAFNESS -- "
            "OFTEN-MISDIAGNOSED-AS-FCAS-NLRP3-OR-TRAPS -- "
            "COLCHICINE-PARTIALLY-EFFECTIVE-ANAKINRA-CANAKINUMAB-SECOND-LINE)"
        ),
        "protein": (
            "NLRP12 -- 19q13.42 AD -- NLRP12-1062aa -- "
            "NLR-Family-Pyrin-Domain-Containing-12-Monarch-1-119kDa -- "
            "PYD-NACHT-LRR-Domain-Architecture-Similar-NLRP3 -- "
            "Negative-Regulator-NF-kB-Inflammasome-Suppressor-In-Wildtype -- "
            "LOF-Heterozygous-Releases-NF-kB-Inhibition-Not-GOF-Inflammasome -- "
            "F402L-R352C-Q983X-Most-Common-NLRP12-FCAS2-Variants -- "
            "OMIM-Gene-609648-Disease-FCAS2-611762"
        ),
        "locus": "19q13.42",
        "protein_size": "1062 aa / 119 kDa",
        "inheritance": (
            "AD (autosomal dominant, heterozygous LOF); "
            "NLRP12 normally suppresses NF-κB and canonical inflammasome; "
            "LOF → NF-κB disinhibition → periodic fever and urticaria; "
            "NOTE: unlike NLRP3 (GOF — constitutively active inflammasome), "
            "NLRP12 mechanism is LOF disinhibition of NF-κB — important distinction; "
            "FCAS2: cold-triggered urticarial rash + fever (similar to FCAS/NLRP3); "
            "deafness less common than Muckle-Wells; milder than NOMID"
        ),
        "disease_category": (
            "FCAS2 / NLRP12-Associated Autoinflammatory Disorder — "
            "COLD-TRIGGERED urticaria + fever (like FCAS/NLRP3 but distinct gene); "
            "often misdiagnosed as FCAS (NLRP3) or TRAPS; "
            "colchicine partially effective; anakinra/canakinumab second-line; "
            "milder than CAPS overall — less deafness, no NOMID-equivalent"
        ),
        "disease_pathway": (
            "NLRP12 (Monarch-1) is an NLR family protein with a distinct regulatory role: "
            "unlike NLRP3 (a danger-sensing inflammasome assembler), NLRP12 primarily functions as "
            "a SUPPRESSOR of inflammatory signalling. Wildtype NLRP12 inhibits: "
            "(1) Canonical NF-κB activation by targeting IRAK-1 and NIK for degradation; "
            "(2) Non-canonical NF-κB pathway (NIK suppression); "
            "(3) ERK/MAPK signalling via interaction with ERK. "
            "FCAS2 pathomechanism (heterozygous LOF): "
            "haploinsufficiency of NLRP12 → insufficient NF-κB suppression → "
            "NF-κB hyperactivation at baseline and following cold stress → "
            "periodic fever, urticaria, fatigue, joint pain. "
            "The COLD TRIGGER mechanism: cold stress activates TRPM8 channels and sympathetic catecholamines → "
            "in NLRP12-deficient cells, the downstream NF-κB response is exaggerated. "
            "CONTRAST WITH NLRP3/CAPS: "
            "NLRP3 = GOF → constitutive caspase-1 activation → IL-1β processing → inflammasome-driven disease; "
            "NLRP12 = LOF → NF-κB disinhibition → broader cytokine dysregulation (TNF, IL-6, IL-1β); "
            "both respond to IL-1 blockade but NLRP12 often also responds to colchicine (different mechanism). "
            "Deafness (SNHL): less common than in Muckle-Wells — periodic audiological monitoring recommended but "
            "cochlear implant rarely needed. Amyloidosis risk low."
        ),
        "pathognomonic": (
            "COLD-TRIGGERED URTICARIA + FEVER in child without positive NLRP3 mutation = consider NLRP12; "
            "milder than Muckle-Wells (less deafness), no NOMID-like CNS manifestations; "
            "urticaria may be pruritic (less non-pruritic signature than NLRP3 CAPS); "
            "often multiple family members with variable expressivity; "
            "NLRP12 sequencing required (not on standard autoinflammatory panels in all centres)"
        ),
        "treatment": (
            "COLCHICINE 1 mg/day: partially effective — first-line (unlike CAPS/NLRP3, colchicine helps here); "
            "ANAKINRA 1-2 mg/kg/day SC: second-line if colchicine insufficient; "
            "CANAKINUMAB 150 mg q8w SC: second-line alternative; "
            "NSAIDs for mild attacks; "
            "cold avoidance (warm clothing, heated home environments); "
            "antihistamines: limited benefit (urticaria may be partly mast-cell); "
            "ophthalmology + audiology monitoring annually"
        ),
        "key_features": [
            "COLD-TRIGGERED urticaria + fever — similar to FCAS (NLRP3) but distinct gene and milder course",
            "Often misdiagnosed as FCAS (NLRP3), TRAPS, or cold urticaria — requires NLRP12 sequencing",
            "NLRP12 is a LOF suppressor (NF-κB disinhibition) — distinct from NLRP3 GOF inflammasome",
            "Colchicine PARTIALLY EFFECTIVE (unlike CAPS/NLRP3 where colchicine NOT effective)",
            "Less deafness than Muckle-Wells; no NOMID-equivalent — milder overall",
            "Heterozygous LOF haploinsufficiency — variable expressivity; AD family history",
            "Not always on standard autoinflammatory panels — request specifically if FCAS phenotype + NLRP3 negative",
        ],
        "key_ddx": [
            "FCAS/NLRP3 (NLRP3 GOF; more consistent deafness risk; non-pruritic urticaria; NLRP3 sequencing)",
            "Cold urticaria (acquired; no fever; positive ice-cube test; antihistamine-responsive)",
            "TRAPS (longer attacks; migratory myalgia; periorbital edema; TNFRSF1A mutation)",
            "Schnitzler syndrome (acquired; IgM/IgG monoclonal gammopathy; adult onset)",
        ],
        "attack_duration_h": "12-48h (cold-triggered episodes)",
        "dominant_trigger": "Cold exposure, physical exertion",
        "amyloid_risk": "Low",
        "nbs_indicated": False,
        "severity": "mild-moderate",
    },
]


# ---------------------------------------------------------------------------
# PATIENT COHORT GENERATOR
# ---------------------------------------------------------------------------

def _make_cohort(entry: dict, seed: int) -> list:
    """Generate 40 synthetic patients for a periodic fever gene."""
    rng = random.Random(seed)
    cohort = []

    gene = entry["gene"]

    for i in range(40):
        age = rng.randint(1, 55)
        sex = rng.choice(["M", "F", "F", "M"])

        # Gene-specific attack frequency and duration
        if gene == "MEFV":
            attacks_per_year = rng.randint(4, 24)
            attack_duration_h = rng.randint(12, 72)
            peak_temp = round(rng.uniform(38.5, 40.5), 1)
            on_colchicine = rng.random() < 0.92
            on_anakinra = rng.random() < 0.18 if on_colchicine else rng.random() < 0.08
            on_canakinumab = rng.random() < 0.12 if not on_anakinra else False
            amyloid = rng.random() < (0.12 if not on_colchicine else 0.02)
        elif gene == "MVK":
            attacks_per_year = rng.randint(3, 18)
            attack_duration_h = rng.randint(72, 168)
            peak_temp = round(rng.uniform(38.8, 40.8), 1)
            on_colchicine = rng.random() < 0.15
            on_anakinra = rng.random() < 0.30
            on_canakinumab = rng.random() < 0.55 if not on_anakinra else False
            amyloid = rng.random() < 0.03
        elif gene == "TNFRSF1A":
            attacks_per_year = rng.randint(2, 12)
            attack_duration_h = rng.randint(120, 504)
            peak_temp = round(rng.uniform(38.5, 40.5), 1)
            on_colchicine = rng.random() < 0.20
            on_anakinra = rng.random() < 0.28
            on_canakinumab = rng.random() < 0.38 if not on_anakinra else False
            amyloid = rng.random() < 0.08
        elif gene == "NLRP3":
            attacks_per_year = rng.randint(6, 365)
            attack_duration_h = rng.randint(12, 96)
            peak_temp = round(rng.uniform(38.0, 40.0), 1)
            on_colchicine = rng.random() < 0.08
            on_anakinra = rng.random() < 0.22
            on_canakinumab = rng.random() < 0.62 if not on_anakinra else False
            amyloid = rng.random() < 0.05
        elif gene == "NOD2":
            attacks_per_year = 0
            attack_duration_h = 0
            peak_temp = round(rng.uniform(37.0, 38.5), 1)
            on_colchicine = rng.random() < 0.05
            on_anakinra = rng.random() < 0.12
            on_canakinumab = rng.random() < 0.08
            amyloid = rng.random() < 0.01
        elif gene == "PSTPIP1":
            attacks_per_year = rng.randint(1, 8)
            attack_duration_h = rng.randint(48, 336)
            peak_temp = round(rng.uniform(38.0, 40.0), 1)
            on_colchicine = rng.random() < 0.10
            on_anakinra = rng.random() < 0.55
            on_canakinumab = rng.random() < 0.25 if not on_anakinra else False
            amyloid = rng.random() < 0.01
        elif gene == "IL1RN":
            attacks_per_year = 0
            attack_duration_h = 0
            peak_temp = round(rng.uniform(37.5, 40.0), 1)
            on_colchicine = False
            on_anakinra = rng.random() < 0.98
            on_canakinumab = rng.random() < 0.02 if not on_anakinra else False
            amyloid = rng.random() < 0.01
        else:  # NLRP12
            attacks_per_year = rng.randint(4, 20)
            attack_duration_h = rng.randint(12, 48)
            peak_temp = round(rng.uniform(38.0, 39.8), 1)
            on_colchicine = rng.random() < 0.55
            on_anakinra = rng.random() < 0.28 if not on_colchicine else rng.random() < 0.15
            on_canakinumab = rng.random() < 0.12 if not on_anakinra else False
            amyloid = rng.random() < 0.01

        crp_attack = round(rng.uniform(40, 250), 1)
        crp_baseline = round(rng.uniform(0.5, 8.0), 1)
        esr_attack = rng.randint(40, 120)
        snhl = rng.random() < (0.35 if gene == "NLRP3" else 0.05)
        uveitis = rng.random() < (0.70 if gene == "NOD2" else 0.05)
        skin_rash = rng.random() < (0.85 if gene in ("NLRP3", "NLRP12") else 0.20)
        lymphadenopathy = rng.random() < (0.85 if gene == "MVK" else 0.15)
        periostitis = rng.random() < (0.80 if gene == "IL1RN" else 0.02)
        pg_lesion = rng.random() < (0.70 if gene == "PSTPIP1" else 0.01)
        migratory_myalgia = rng.random() < (0.80 if gene == "TNFRSF1A" else 0.08)
        periorbital_edema = rng.random() < (0.75 if gene == "TNFRSF1A" else 0.03)
        genetic_dx = rng.random() < 0.88

        cohort.append({
            "patient_id": f"HPF-{gene[:4]}-{seed}-{i+1:03d}",
            "gene": gene,
            "age": age,
            "sex": sex,
            "attacks_per_year": attacks_per_year,
            "attack_duration_h": attack_duration_h,
            "peak_temp_C": peak_temp,
            "crp_attack_mg_L": crp_attack,
            "crp_baseline_mg_L": crp_baseline,
            "esr_attack_mm_h": esr_attack,
            "on_colchicine": on_colchicine,
            "on_anakinra": on_anakinra,
            "on_canakinumab": on_canakinumab,
            "has_amyloid": amyloid,
            "has_snhl": snhl,
            "has_uveitis": uveitis,
            "has_skin_rash": skin_rash,
            "has_lymphadenopathy": lymphadenopathy,
            "has_periostitis": periostitis,
            "has_pg_lesion": pg_lesion,
            "has_migratory_myalgia": migratory_myalgia,
            "has_periorbital_edema": periorbital_edema,
            "had_genetic_diagnosis": genetic_dx,
        })

    return cohort


# ---------------------------------------------------------------------------
# API DATA FUNCTIONS
# ---------------------------------------------------------------------------

def generate_overview():
    all_cohort = []
    gene_summary = []

    for idx, entry in enumerate(FEVER_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        all_cohort.extend(cohort)
        total = len(cohort)

        gene_summary.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_name": entry["disease_category"].split(" —")[0].strip(),
            "n_patients": total,
            "attack_duration_h": entry["attack_duration_h"],
            "dominant_trigger": entry["dominant_trigger"],
            "amyloid_risk": entry["amyloid_risk"],
            "severity": entry["severity"],
            "on_colchicine_pct": round(100 * sum(1 for p in cohort if p["on_colchicine"]) / total, 1),
            "on_anakinra_pct": round(100 * sum(1 for p in cohort if p["on_anakinra"]) / total, 1),
            "on_canakinumab_pct": round(100 * sum(1 for p in cohort if p["on_canakinumab"]) / total, 1),
            "amyloid_pct": round(100 * sum(1 for p in cohort if p["has_amyloid"]) / total, 1),
            "snhl_pct": round(100 * sum(1 for p in cohort if p["has_snhl"]) / total, 1),
            "uveitis_pct": round(100 * sum(1 for p in cohort if p["has_uveitis"]) / total, 1),
            "genetic_dx_pct": round(100 * sum(1 for p in cohort if p["had_genetic_diagnosis"]) / total, 1),
        })

    total = len(all_cohort)
    return {
        "atlas_name": "Hereditary Periodic Fever Syndrome Atlas",
        "atlas_id": "hereditary-periodic-fever-atlas",
        "subtitle": "Complete 8-Gene Hereditary Periodic Fever & Autoinflammatory Atlas",
        "genes": [e["gene"] for e in FEVER_GENES],
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": total,
        "n_genes": len(FEVER_GENES),
        "disease_classes": [
            "Pyrin Inflammasome (FMF — MEFV)",
            "Mevalonate Kinase Deficiency (HIDS — MVK)",
            "TNF Receptor Periodic Syndrome (TRAPS — TNFRSF1A)",
            "Cryopyrin-Associated Periodic Syndrome (CAPS — NLRP3)",
            "Blau/Early-Onset Sarcoidosis (NOD2)",
            "PAPA Syndrome (PSTPIP1)",
            "IL-1Ra Deficiency (DIRA — IL1RN)",
            "NLRP12-Associated Autoinflammation (NLRP12)",
        ],
        "aggregate_metrics": {
            "on_colchicine_pct": round(100 * sum(1 for p in all_cohort if p["on_colchicine"]) / total, 1),
            "on_anakinra_pct": round(100 * sum(1 for p in all_cohort if p["on_anakinra"]) / total, 1),
            "on_canakinumab_pct": round(100 * sum(1 for p in all_cohort if p["on_canakinumab"]) / total, 1),
            "amyloid_pct": round(100 * sum(1 for p in all_cohort if p["has_amyloid"]) / total, 1),
            "snhl_pct": round(100 * sum(1 for p in all_cohort if p["has_snhl"]) / total, 1),
            "uveitis_pct": round(100 * sum(1 for p in all_cohort if p["has_uveitis"]) / total, 1),
            "genetic_dx_pct": round(100 * sum(1 for p in all_cohort if p["had_genetic_diagnosis"]) / total, 1),
        },
        "gene_summary": gene_summary,
        "clinical_pearls": [
            "FMF/MEFV: COLCHICINE IS MANDATORY FOR LIFE — prevents AA amyloidosis; attacks 12-72h; sterile peritonitis mimics acute abdomen",
            "HIDS/MVK: VACCINATION-TRIGGERED ATTACK PATHOGNOMONIC; cervical lymphadenopathy KEY DDx from FMF; canakinumab FDA-approved",
            "TRAPS/TNFRSF1A: ATTACKS >7 DAYS + MIGRATORY MYALGIA + PERIORBITAL EDEMA PATHOGNOMONIC; etanercept preferred, NOT infliximab",
            "CAPS/NLRP3: NON-PRURITIC URTICARIA PATHOGNOMONIC (not allergic; antihistamines INEFFECTIVE); NOMID treat URGENTLY — irreversible",
            "Blau/NOD2: GRANULOMATOUS ARTHRITIS + UVEITIS + SKIN GRANULOMA TRIAD; NOT Crohn NOD2 variants (those are LOF)",
            "PAPA/PSTPIP1: PG PATHERGY — AVOID DEBRIDEMENT (worsens ulceration); culture-negative purulent arthritis not septic",
            "DIRA/IL1RN: NEONATAL ONSET; ANAKINRA CURATIVE AND LIFE-SAVING (physiological IL-1Ra replacement); start URGENTLY",
            "FCAS2/NLRP12: cold-triggered like FCAS/NLRP3 but distinct; colchicine partially effective (unlike CAPS); less deafness",
        ],
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(FEVER_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        total = len(cohort)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "attack_duration_h": entry["attack_duration_h"],
            "dominant_trigger": entry["dominant_trigger"],
            "amyloid_risk": entry["amyloid_risk"],
            "nbs_indicated": entry["nbs_indicated"],
            "severity": entry["severity"],
            "n_patients": total,
            "avg_peak_temp": round(sum(p["peak_temp_C"] for p in cohort) / total, 1),
            "avg_crp_attack": round(sum(p["crp_attack_mg_L"] for p in cohort) / total, 1),
            "avg_attacks_per_year": round(sum(p["attacks_per_year"] for p in cohort) / total, 1),
            "on_colchicine_pct": round(100 * sum(1 for p in cohort if p["on_colchicine"]) / total, 1),
            "on_anakinra_pct": round(100 * sum(1 for p in cohort if p["on_anakinra"]) / total, 1),
            "on_canakinumab_pct": round(100 * sum(1 for p in cohort if p["on_canakinumab"]) / total, 1),
            "amyloid_pct": round(100 * sum(1 for p in cohort if p["has_amyloid"]) / total, 1),
            "snhl_pct": round(100 * sum(1 for p in cohort if p["has_snhl"]) / total, 1),
            "uveitis_pct": round(100 * sum(1 for p in cohort if p["has_uveitis"]) / total, 1),
            "lymphadenopathy_pct": round(100 * sum(1 for p in cohort if p["has_lymphadenopathy"]) / total, 1),
            "periostitis_pct": round(100 * sum(1 for p in cohort if p["has_periostitis"]) / total, 1),
            "pg_pct": round(100 * sum(1 for p in cohort if p["has_pg_lesion"]) / total, 1),
            "myalgia_pct": round(100 * sum(1 for p in cohort if p["has_migratory_myalgia"]) / total, 1),
            "periorbital_pct": round(100 * sum(1 for p in cohort if p["has_periorbital_edema"]) / total, 1),
            "genetic_dx_pct": round(100 * sum(1 for p in cohort if p["had_genetic_diagnosis"]) / total, 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["alt_name"].split(" (")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:500],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "attack_duration_h": entry["attack_duration_h"],
                "dominant_trigger": entry["dominant_trigger"],
                "amyloid_risk": entry["amyloid_risk"],
                "nbs_indicated": entry["nbs_indicated"],
            }
            for entry in FEVER_GENES
        },
        "fever_glossary": {
            "Pyrin Inflammasome (MEFV/FMF)": (
                "Pyrin (encoded by MEFV) assembles a distinct inflammasome activated by bacterial effectors "
                "that inactivate RhoA GTPase. FMF mutations (B30.2 domain) destabilise 14-3-3 inhibition → "
                "constitutive pyrin inflammasome → IL-1β/IL-18. "
                "COLCHICINE: inhibits microtubule-dependent pyrin assembly — MANDATORY LIFELONG — "
                "prevents AA amyloidosis. Attacks 12-72h with complete resolution."
            ),
            "NLRP3 Inflammasome (NLRP3/CAPS)": (
                "NLRP3 is a danger-sensing inflammasome assembler. GOF CAPS mutations → constitutive "
                "caspase-1 → IL-1β. CAPS spectrum: FCAS (cold-triggered, mild) → Muckle-Wells "
                "(SNHL, amyloidosis) → NOMID (neonatal, CNS, treat urgently). "
                "NON-PRURITIC urticaria PATHOGNOMONIC — antihistamines INEFFECTIVE (neutrophilic not IgE). "
                "Canakinumab + rilonacept FDA-approved."
            ),
            "IL-1Ra Physiology and DIRA": (
                "IL-1Ra (IL1RN) is the endogenous competitive antagonist of IL-1R1 — no agonist activity. "
                "Biallelic LOF → DIRA: unopposed IL-1 signalling → neonatal sterile multifocal osteomyelitis + "
                "pustulosis + periostitis. ANAKINRA = recombinant IL-1Ra → CURATIVE AND LIFE-SAVING. "
                "Must start urgently — delay causes irreversible bone destruction."
            ),
            "Colchicine vs IL-1 Blocker Selection": (
                "COLCHICINE EFFECTIVE: FMF/MEFV (pyrin — microtubule-dependent assembly); NLRP12/FCAS2 (partial). "
                "COLCHICINE NOT EFFECTIVE: CAPS/NLRP3 (NLRP3 assembly microtubule-independent); "
                "HIDS/MVK, TRAPS, PAPA, DIRA, Blau. "
                "CANAKINUMAB FDA-APPROVED: FMF, HIDS, TRAPS (2016). "
                "ETANERCEPT PREFERRED IN TRAPS (not infliximab — anti-drug antibodies + paradoxical reactions)."
            ),
            "AA Amyloidosis in Periodic Fevers": (
                "AA (serum amyloid A) amyloidosis: highest risk in FMF (colchicine prevents), TRAPS (Cys-variants), "
                "Muckle-Wells. SAA is an acute-phase reactant — chronic SAA elevation → renal AA deposition → "
                "nephrotic syndrome → end-stage renal disease. Monitor: serum SAA + proteinuria annually in high-risk. "
                "Treatment: control underlying inflammation (colchicine/IL-1 blockade) + eprodisate."
            ),
            "Pathergy": (
                "Pathergy: exaggerated tissue response to minor trauma — diagnostic of PAPA/PSTPIP1 (PG) "
                "and Behçet disease. In PAPA: minor wound → PG ulceration. "
                "CRITICAL: AVOID DEBRIDEMENT of PG lesions — worsens ulceration by pathergy. "
                "Topical/intralesional steroids + compression; IL-1 blockade for prevention."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (gene 0 only) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first gene entry) ===")
    defn = generate_definitions()
    first_key = next(iter(defn["gene_entries"]))
    print(json.dumps(defn["gene_entries"][first_key], indent=2)[:1500])
