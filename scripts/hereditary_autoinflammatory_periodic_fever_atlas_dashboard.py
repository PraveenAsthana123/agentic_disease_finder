#!/usr/bin/env python3
"""Hereditary-Autoinflammatory-Periodic-Fever-Atlas -- Complete 8-Gene Reference
Familial Mediterranean Fever, TRAPS, HIDS/MKD, CAPS, PAPA, Blau, DIRA, Majeed Syndrome
(MEFV/FMF-Serositis-Amyloid-AA-PATHOGNOMONIC-Colchicine-Curative-Most-Common-Autoinflammatory,
 TNFRSF1A/TRAPS-Prolonged-Attack-gt7days-PATHOGNOMONIC-Monocyte-Patches-Etanercept-Canakinumab,
 MVK/HIDS-MKD-IgD-elevated-Dutch-Founder-PATHOGNOMONIC-Canakinumab,
 NLRP3/CAPS-Spectrum-FCAS-MWS-NOMID-Canakinumab-Approved-IL1-Inhibition,
 PSTPIP1/PAPA-Pyoderma-Gangrenosum-PATHOGNOMONIC-IL1-TNF-Inhibition,
 NOD2/Blau-Granulomatous-Uveitis-Arthritis-Dermatitis-Triad-PATHOGNOMONIC-AD-GOF,
 IL1RN/DIRA-Neonatal-Sterile-Osteomyelitis-Periostitis-Pustulosis-PATHOGNOMONIC-Anakinra-Curative,
 LPIN2/Majeed-Recurrent-Multifocal-Osteomyelitis-Dyserythropoietic-Anemia-PATHOGNOMONIC-Anakinra):

MEFV   (Mediterranean fever protein / pyrin; 781aa; 16p13.3; AR (classical) and AD (low-penetrance);
         MEFV encodes pyrin, a scaffolding protein in the NLRP3/ASC inflammasome signalling pathway;
         MEFV LOF (AR) → defective pyrin-mediated IL-18/IL-1β suppression → episodic IL-1β and IL-18
           surges triggered by microbial products and Rho-GTPase signals → recurrent sterile serositis;
         FAMILIAL MEDITERRANEAN FEVER (FMF): most common hereditary autoinflammatory disease worldwide;
         PATHOGNOMONIC TRIAD: (1) SHORT ATTACKS (12-72 hours; NOT >7 days — distinguishes from TRAPS);
           (2) SEROSITIS: peritonitis (abdominal crisis), pleuritis, synovitis, pericarditis (acute);
           (3) HIGH FEVER (38.5-40°C); erysipelas-like erythema (ELE) on lower legs in 30-50%;
         AA AMYLOIDOSIS: the most serious long-term complication — amyloid A protein deposited in
           kidneys (proteinuria → nephrotic syndrome → renal failure) without prophylaxis;
           AA AMYLOIDOSIS = INADEQUATELY TREATED FMF = preventable with colchicine;
         COLCHICINE: FIRST-LINE AND CURATIVE — prevents attacks in ~60%, reduces frequency in ~90%;
           COLCHICINE RESISTANCE: <5%; defined as persistent attacks on 3 mg/day; → IL-1 inhibitors
           (canakinumab/anakinra) licensed for colchicine-resistant/intolerant FMF;
         GENOTYPE-PHENOTYPE: p.M694V homozygous = MOST SEVERE (highest amyloid risk, earliest onset);
           p.M694V/M680I compound heterozygous also severe; p.E148Q = mild, may be benign variant;
         ETHNIC DISTRIBUTION: Turkish, Armenian, Arab, Jewish Sephardic (carrier frequency up to 1/5);
         seed SEED_BASE+0).

TNFRSF1A (Tumour necrosis factor receptor superfamily member 1A; 455aa; 12p13.31; AD-LOF;
         TNFRSF1A encodes TNFR1, the main signalling receptor for TNF-α on cell surfaces;
         TNFRSF1A AD mutations → impaired shedding of membrane-bound TNFR1 → membrane accumulation
           → sustained abnormal signalling → prolonged pro-inflammatory cytokine storms;
         TNF RECEPTOR-ASSOCIATED PERIODIC SYNDROME (TRAPS):
         PATHOGNOMONIC FEATURE: PROLONGED ATTACKS >7 DAYS (classical TRAPS often 1-6 weeks);
           distinguishes from FMF (12-72h) and HIDS (3-7d); attacks may last weeks to months;
         CLINICAL FEATURES:
           (1) MIGRATORY MYALGIA WITH OVERLYING ERYTHEMATOUS SKIN PATCHES: centrifugal migration
               from proximal to distal; monocyte-rich erythematous skin patches PATHOGNOMONIC;
           (2) PERIORBITAL OEDEMA: characteristic facial swelling around eyes;
           (3) SEROSITIS: pleuritis/peritonitis less common than FMF but present;
           (4) LYMPHADENOPATHY: more prominent than in FMF;
         AA AMYLOIDOSIS: risk ~25% in structural TNFRSF1A mutations; regular SAA + urine protein;
         TREATMENT: NSAIDs for mild attacks; corticosteroids (but dose-dependent tachyphylaxis);
           ETANERCEPT (TNF-receptor fusion): effective for many TRAPS patients (NOT infliximab);
           CANAKINUMAB (anti-IL-1β): approved for TRAPS; superior amyloid protection;
           ANAKINRA (IL-1Ra): effective short-term; less convenient than canakinumab;
         GENOTYPE: cysteine-disrupting mutations (C30Y, C52F, C70S, T50M) = structural variant
           (most severe, highest amyloid risk); low-penetrance variants (R92Q, P46L) less severe;
         seed SEED_BASE+1).

MVK    (Mevalonate kinase; 396aa; 12q24.11; AR-LOF;
         MVK encodes mevalonate kinase, the second enzyme in the mevalonate/isoprenoid pathway;
         MVK LOF → reduced mevalonate kinase activity → accumulation of mevalonic acid, reduced
           downstream isoprenoids (geranylgeranyl-pyrophosphate) → defective Rho-GTPase prenylation
           → inappropriately activated caspase-1 → excess IL-1β and IL-18;
         TWO DISEASE SPECTRA from MVK:
           (1) HIDS (Hyperimmunoglobulinemia D Syndrome): mild-moderate MVK deficiency (>1%);
           (2) Mevalonic Aciduria (MVA): severe complete MVK deficiency (<0.3%); developmental delay,
               cerebellar ataxia, dysmorphia, recurrent crises; severe end of spectrum;
         HIDS/MKD PATHOGNOMONIC FEATURES:
           (1) ATTACKS 3-7 DAYS (longer than FMF <72h, shorter than TRAPS >7d);
           (2) CERVICAL LYMPHADENOPATHY: prominent, painful, near onset of each attack;
           (3) IgD MARKEDLY ELEVATED (>100 IU/mL) in >80% of classical HIDS: pathognomonic marker
               though not universally present; IgA also elevated; IgD normal in severe MVA;
           (4) DUTCH/FRENCH FOUNDER: p.V377I mutation most common (~80% of HIDS alleles in Europeans);
           (5) TRIGGERS: immunisations (vaccines trigger attacks in children — counsel parents);
               physical/emotional stress, minor infections, surgery;
         URINE MEVALONIC ACID: elevated during attacks (and continuously in severe MVA);
         CANAKINUMAB: approved and superior; significant attack reduction; simtuzumab trials;
         seed SEED_BASE+2).

NLRP3  (NLR family pyrin domain-containing protein 3 / cryopyrin; 1036aa; 1q44; AD-GOF;
         NLRP3 encodes cryopyrin, the key NOD-like receptor that forms the NLRP3 inflammasome
           with ASC and pro-caspase-1 → active caspase-1 → IL-1β and IL-18 maturation and secretion;
         AD-GOF NLRP3 mutations → CONSTITUTIVE/HYPERACTIVE inflammasome assembly → autonomous
           IL-1β/IL-18 secretion independent of normal triggering stimuli → CAPS spectrum;
         CRYOPYRIN-ASSOCIATED PERIODIC SYNDROME (CAPS) — THREE SEVERITY LEVELS (SAME GENE):
           (1) FCAS (Familial Cold Autoinflammatory Syndrome; mildest): cold-induced urticaria-like
               rash, arthralgia, fever; COLD TRIGGERS PATHOGNOMONIC; no CNS involvement; R260W;
           (2) MWS (Muckle-Wells Syndrome; intermediate): recurrent urticaria + fever + deafness;
               SENSORINEURAL HEARING LOSS progressive PATHOGNOMONIC; AA amyloidosis risk 25-30%;
           (3) NOMID/CINCA (Neonatal Onset Multisystem Inflammatory Disease; most severe): neonatal
               onset of urticaria + chronic meningitis + arthropathy + mental retardation + papilloedema
               + hearing loss; NEONATAL CONTINUOUS URTICARIA PATHOGNOMONIC for NOMID;
         KEY PATHOGNOMONIC FINDING: NON-PRURITIC URTICARIA (not itch-driven, distinguishes from
           allergic urticaria); neutrophil-rich skin biopsy confirms autoinflammatory mechanism;
         CANAKINUMAB (anti-IL-1β monoclonal): FDA-approved for all CAPS; dramatic IL-1 blockade;
           life-changing especially for NOMID/MWS; prevents hearing loss progression and amyloid;
         RILONACEPT and ANAKINRA also approved for CAPS; canakinumab preferred (monthly SC injection);
         seed SEED_BASE+3).

PSTPIP1 (Proline-serine-threonine phosphatase interacting protein 1; 416aa; 15q24.3; AD-LOF→GOF;
         PSTPIP1 encodes a cytoskeletal adaptor protein that forms a coiled-coil domain interacting
           with pyrin (MEFV); PSTPIP1 mutations hyperactivate pyrin inflammasome → excess IL-1β;
         PAPA SYNDROME (Pyogenic sterile Arthritis, Pyoderma Gangrenosum, Acne):
         PATHOGNOMONIC TRIAD:
           (1) PYOGENIC STERILE ARTHRITIS: occurs in childhood; severe destructive mono-/oligo-arthritis;
               synovial fluid: 50,000-80,000 WBC/µL but sterile (no organisms — STERILE PYOGENIC);
               spontaneously remits with puberty in some; PATHOGNOMONIC = destructive arthritis in
               child with sterile synovial fluid with massive neutrophils;
           (2) PYODERMA GANGRENOSUM (PG): most diagnostic feature in adults; painful rapidly-expanding
               ulcers with violaceous undermined borders; starts as pustule → ulcer; pathergy positive;
               PG PATHOGNOMONIC FEATURE: violaceous undermined border + pathergy;
           (3) ACNE CONGLOBATA/FULMINANS: severe cystic acne; often treatment-refractory;
         PSTPIP1-MEFV INTERACTION: PAPA mutations (A230T, E250K) disrupt PSTPIP1-pyrin binding →
           unchecked pyrin inflammasome activation; PSTPIP1 COILED-COIL domain mutations are pathogenic;
         TREATMENT: IL-1 inhibitors (anakinra/canakinumab) and/or TNF inhibitors (adalimumab/etanercept);
           local wound care for PG lesions; isotretinoin with caution for acne;
         seed SEED_BASE+4).

NOD2   (Nucleotide-binding oligomerization domain-containing protein 2; 1040aa; 16q12.1; AD-GOF;
         NOD2 encodes an intracellular pattern recognition receptor detecting muramyl dipeptide (MDP)
           in bacterial peptidoglycan; NOD2 signals via RICK/RIP2 → NF-κB → pro-inflammatory cytokines;
         AD-GOF BLAU SYNDROME mutations (in NOD domain, distinct from IBD-associated LOF mutations):
           gain-of-function → constitutive NF-κB activation and granuloma formation;
         BLAU SYNDROME / EARLY-ONSET SARCOIDOSIS (EOS):
         PATHOGNOMONIC TRIAD (MUST distinguish from sarcoidosis which lacks genetic driver):
           (1) GRANULOMATOUS UVEITIS: panuveitis → band keratopathy, iris synechiae, glaucoma, cataract;
               MOST SEVERE feature; can lead to blindness without treatment; non-caseating granulomas;
           (2) CAMPTODACTYLY BOGGY ARTHRITIS: symmetric polyarthritis with boggy synovial proliferation
               → CAMPTODACTYLY (fixed flexion of fingers); often painless despite severe swelling;
               CAMPTODACTYLY PATHOGNOMONIC COMBINATION with uveitis and dermatitis;
           (3) GRANULOMATOUS DERMATITIS: lichenoid papular skin rash; ichthyosiform; granulomas on biopsy;
         SYSTEMIC: hepatosplenomegaly; cranial nerve palsies; large-vessel vasculitis; fever intervals;
         DISTINCTION FROM SARCOIDOSIS: Blau = childhood onset, family history (AD), NOD2 mutation;
           sarcoidosis = sporadic, often lungs involved (Blau spares lungs), older age;
         TREATMENT: corticosteroids + methotrexate backbone; anti-TNF (adalimumab/infliximab) for
           refractory uveitis; IL-6 inhibitors (tocilizumab); avoid prolonged systemic steroids alone;
         seed SEED_BASE+5).

IL1RN  (Interleukin-1 receptor antagonist; 177aa mature form / 217aa precursor; 2q14.1; AR-LOF;
         IL1RN encodes IL-1Ra (interleukin-1 receptor antagonist), the endogenous physiological
           inhibitor of both IL-1α and IL-1β; IL-1Ra competes with IL-1 for receptor binding
           without signalling → critical brake on IL-1-mediated inflammation;
         DIRA (Deficiency of IL-1 Receptor Antagonist): AR-LOF → absent IL-1Ra → completely
           unantagonised IL-1 signalling → neonatal sterile multifocal osteomyelitis and systemic
           inflammation; extremely rare (<100 patients worldwide);
         PATHOGNOMONIC NEONATAL PRESENTATION:
           (1) STERILE MULTIFOCAL OSTEOMYELITIS: widespread bone inflammation from the neonatal period;
               multiple bones simultaneously; radiographs show periosteal new bone formation;
               sterile pustular skin lesions (NOT true osteomyelitis - sterile);
           (2) PERIOSTITIS: widespread periosteal new bone on imaging PATHOGNOMONIC in neonates;
           (3) PUSTULOSIS: diffuse sterile pustular skin lesions with neutrophilic infiltration;
           (4) ELEVATED ACUTE PHASE: extreme CRP/ESR/ferritin; systemic inflammation;
         FOUNDER MUTATIONS: p.E77X (Puerto Rican/Brazilian); p.C80S (Dutch/Newfoundland);
           gene deletions: whole gene deletion in some families (autosomal recessive);
         ANAKINRA: CURATIVE AND RAPID — IV/SC anakinra (recombinant IL-1Ra) reverses ALL features
           within days to weeks; patients require lifelong anakinra (attempt withdrawal → relapse);
           ANAKINRA CURATIVE RESPONSE is diagnostic/therapeutic confirmation of DIRA;
         seed SEED_BASE+6).

LPIN2  (Phosphatidate phosphatase LPIN2; 986aa; 18p11.31; AR-LOF;
         LPIN2 encodes lipin-2, a phosphatidate phosphatase involved in lipid metabolism and
           regulation of inflammatory signalling; LPIN2 LOF → dysregulated IL-1β production via
           arachidonic acid/lipid intermediates and NLRP3 inflammasome hyperactivation;
         MAJEED SYNDROME:
         PATHOGNOMONIC TRIAD:
           (1) RECURRENT MULTIFOCAL OSTEOMYELITIS (RMO): sterile osteomyelitis affecting multiple
               bones recurrently; radiographs: multifocal lytic lesions; markedly elevated inflammatory
               markers during crises; distinguishable from CRMO by congenital dyserythropoietic anaemia;
           (2) CONGENITAL DYSERYTHROPOIETIC ANAEMIA (CDA): PATHOGNOMONIC — CDA in the context of
               autoinflammatory disease is UNIQUE TO MAJEED (no other autoinflammatory disease has CDA);
               peripheral smear: dysplastic erythropoiesis; BM: multinucleate erythroblasts;
               macrocytic or normochromic anaemia from infancy; transfusion dependence;
           (3) NEUTROPHILIC SKIN DISORDER: Sweet syndrome-like or psoriasiform neutrophilic dermatosis;
               inflammatory skin flares synchronised with bone crises;
         FOUNDER: first described in Jordanian family (Majeed 1989); primarily reported in Middle East;
         DISTINCTION: unlike DIRA (pure IL-1Ra deficiency) → no periostitis; unlike CRMO → CDA present;
         TREATMENT: NSAIDs partially effective; ANAKINRA and CANAKINUMAB: significant response
           in bone crises and anaemia improvement; anti-IL-1 = treatment of choice;
           transfusion support for CDA component; iron chelation if transfusion-dependent;
         seed SEED_BASE+7).
"""
import random

SEED_BASE = 4174
GENES = ["MEFV", "TNFRSF1A", "MVK", "NLRP3", "PSTPIP1", "NOD2", "IL1RN", "LPIN2"]

GENE_META = {
    "MEFV": {
        "full_name": "Mediterranean fever protein / Pyrin (MEFV)",
        "aa": 781, "kda": 86, "locus": "16p13.3", "inheritance": "AR (classical); AD (low-penetrance)",
        "disease": "Familial Mediterranean Fever (FMF)",
        "omim_gene": "608107", "omim_disease": "249100",
        "nbs": False,
        "pathognomonic": "SHORT ATTACKS 12-72 h (NEVER >7 days); SEROSITIS (peritonitis/pleuritis/synovitis); erysipelas-like erythema lower legs; AA amyloidosis in untreated; p.M694V homozygous MOST SEVERE",
        "key_drug": "COLCHICINE 0.5-1.0 mg BD (lifelong); IL-1 inhibitors (canakinumab/anakinra) for colchicine-resistant; SAA monitoring for AA amyloid",
        "precision_therapy": "Colchicine curative (60% full suppression, 90% frequency reduction); canakinumab for resistance; anakinra for acute attacks",
        "avoid": "Stopping colchicine (AA amyloid risk); NSAIDs alone (insufficient); infliximab (no evidence)",
    },
    "TNFRSF1A": {
        "full_name": "TNF receptor superfamily member 1A (TNFRSF1A)",
        "aa": 455, "kda": 51, "locus": "12p13.31", "inheritance": "AD",
        "disease": "TNF Receptor-Associated Periodic Syndrome (TRAPS)",
        "omim_gene": "191190", "omim_disease": "142680",
        "nbs": False,
        "pathognomonic": "PROLONGED ATTACKS >7 DAYS (classical weeks); migratory myalgia + monocyte-rich erythematous skin patches PATHOGNOMONIC; periorbital oedema; AA amyloid in structural mutations",
        "key_drug": "Etanercept (TNF-Fc fusion; NOT infliximab) for moderate; canakinumab (anti-IL-1β) for severe/amyloid prevention; corticosteroids short-term",
        "precision_therapy": "Canakinumab preferred for structural mutations (high amyloid risk); etanercept for milder phenotypes; rilonacept second-line",
        "avoid": "Infliximab (may worsen TRAPS); stopping anti-IL-1 without SAA monitoring; underestimating amyloid risk in cysteine mutations",
    },
    "MVK": {
        "full_name": "Mevalonate kinase (MVK)",
        "aa": 396, "kda": 43, "locus": "12q24.11", "inheritance": "AR",
        "disease": "Hyperimmunoglobulinemia D Syndrome / MKD (HIDS) / Mevalonic Aciduria (MVA)",
        "omim_gene": "251170", "omim_disease": "260920",
        "nbs": False,
        "pathognomonic": "ATTACKS 3-7 DAYS; CERVICAL LYMPHADENOPATHY prominent; IgD >100 IU/mL in >80% PATHOGNOMONIC; Dutch p.V377I founder mutation; VACCINE-TRIGGERED attacks",
        "key_drug": "Canakinumab (approved HIDS/MKD): 150-300 mg SC every 4-8 weeks; anakinra for acute attacks; IL-1 blockade preferred",
        "precision_therapy": "Canakinumab approved 2017 for refractory HIDS; simtuzumab trials; atorvastatin theoretically (reduces mevalonate substrate but limited evidence)",
        "avoid": "Statins (may worsen MVA by blocking residual MVK activity); missing severe MVA diagnosis (developmental delay spectrum)",
    },
    "NLRP3": {
        "full_name": "NLRP3 / Cryopyrin (NLRP3)",
        "aa": 1036, "kda": 118, "locus": "1q44", "inheritance": "AD-GOF",
        "disease": "CAPS spectrum: FCAS / Muckle-Wells Syndrome / NOMID-CINCA",
        "omim_gene": "606416", "omim_disease": "120100 / 191900 / 607115",
        "nbs": False,
        "pathognomonic": "NON-PRURITIC URTICARIA (neutrophilic skin biopsy); FCAS: cold-triggered; MWS: sensorineural hearing loss PATHOGNOMONIC; NOMID: neonatal continuous urticaria + aseptic meningitis",
        "key_drug": "Canakinumab (anti-IL-1β) approved for all CAPS; rilonacept and anakinra alternatives; prevents hearing loss and AA amyloid in MWS; dramatic response expected",
        "precision_therapy": "Canakinumab 2 mg/kg SC every 8 weeks (FCAS/MWS); may need higher dosing for NOMID; IL-18 measurement guides response; MRI for aseptic meningitis in NOMID",
        "avoid": "NSAIDs/steroids alone (insufficient NOMID/MWS); delayed treatment → irreversible hearing loss; diagnostic confusion with allergic urticaria (non-pruritic key feature)",
    },
    "PSTPIP1": {
        "full_name": "Proline-serine-threonine phosphatase interacting protein 1 (PSTPIP1)",
        "aa": 416, "kda": 48, "locus": "15q24.3", "inheritance": "AD",
        "disease": "PAPA Syndrome (Pyogenic sterile Arthritis, Pyoderma Gangrenosum, Acne)",
        "omim_gene": "606347", "omim_disease": "604416",
        "nbs": False,
        "pathognomonic": "TRIAD: sterile pyogenic destructive arthritis (childhood) + pyoderma gangrenosum + severe cystic acne; STERILE synovial fluid with 50,000-80,000 WBC/µL PATHOGNOMONIC",
        "key_drug": "Anakinra or canakinumab (IL-1 inhibitors) for arthritis and PG; adalimumab/etanercept (anti-TNF) for PG/acne; wound care for PG ulcers; isotretinoin for acne",
        "precision_therapy": "IL-1 + TNF co-blockade sometimes required; wound therapy for PG (wet dressings, avoid surgery over inflamed skin); cyclosporine for acute PG",
        "avoid": "Surgical debridement of active PG (pathergy worsens PG); steroids without IL-1/TNF blockade (insufficient)",
    },
    "NOD2": {
        "full_name": "Nucleotide-binding oligomerization domain 2 (NOD2)",
        "aa": 1040, "kda": 115, "locus": "16q12.1", "inheritance": "AD-GOF",
        "disease": "Blau Syndrome / Early-Onset Sarcoidosis (EOS)",
        "omim_gene": "605956", "omim_disease": "186580",
        "nbs": False,
        "pathognomonic": "TRIAD: granulomatous uveitis (panuveitis) + camptodactyly boggy arthritis + lichenoid granulomatous dermatitis PATHOGNOMONIC; CHILDHOOD ONSET; NOD domain GOF (not IBD-associated LOF)",
        "key_drug": "Adalimumab/infliximab (anti-TNF) for uveitis; methotrexate + corticosteroids; tocilizumab (anti-IL-6) refractory; ophthalmology co-management essential",
        "precision_therapy": "Anti-TNF for sight-threatening uveitis (adalimumab preferred); frequent ophthalmology monitoring; early methotrexate to reduce steroid exposure",
        "avoid": "Confusing with sarcoidosis (Blau has no lung involvement, childhood onset, AD family history); topical steroids alone for panuveitis (insufficient); delaying anti-TNF (irreversible blindness risk)",
    },
    "IL1RN": {
        "full_name": "Interleukin-1 receptor antagonist (IL1RN)",
        "aa": 177, "kda": 25, "locus": "2q14.1", "inheritance": "AR",
        "disease": "DIRA (Deficiency of IL-1 Receptor Antagonist)",
        "omim_gene": "147679", "omim_disease": "612852",
        "nbs": False,
        "pathognomonic": "NEONATAL STERILE MULTIFOCAL OSTEOMYELITIS + PERIOSTITIS (periosteal new bone on imaging) + STERILE PUSTULOSIS PATHOGNOMONIC; life-threatening without treatment; anakinra curative within days",
        "key_drug": "Anakinra (recombinant IL-1Ra) IV/SC: CURATIVE; lifelong; dose 1-3 mg/kg/day; rapid resolution within days; canakinumab alternative for maintenance",
        "precision_therapy": "Anakinra = first and only effective therapy; daily injections required lifelong; dose titration; IL-18 monitoring; confirm diagnosis with exon 3-5 deletion or founder mutation sequencing",
        "avoid": "Antibiotics instead of anakinra (sterile — no bacterial cause); delay in anakinra (bone destruction progresses rapidly in neonates); stopping anakinra (relapse guaranteed)",
    },
    "LPIN2": {
        "full_name": "Phosphatidate phosphatase LPIN2 (LPIN2)",
        "aa": 986, "kda": 111, "locus": "18p11.31", "inheritance": "AR",
        "disease": "Majeed Syndrome (Recurrent Multifocal Osteomyelitis + Congenital Dyserythropoietic Anaemia)",
        "omim_gene": "605518", "omim_disease": "609628",
        "nbs": False,
        "pathognomonic": "TRIAD: sterile multifocal osteomyelitis + CONGENITAL DYSERYTHROPOIETIC ANAEMIA (CDA; PATHOGNOMONIC unique feature) + neutrophilic dermatosis; CDA in autoinflammatory disease = MAJEED until proven",
        "key_drug": "Anakinra or canakinumab: most effective anti-IL-1 approach; NSAIDs partially effective; transfusion support for CDA; iron chelation if transfusion-dependent",
        "precision_therapy": "Anti-IL-1 (anakinra/canakinumab): reduces bone crises AND improves CDA (possible IL-1 contribution to dyserythropoiesis); haematology co-management for CDA",
        "avoid": "Treating as CRMO without recognising CDA (misses Majeed diagnosis); bisphosphonates without anti-IL-1 (insufficient for Majeed); iron loading without chelation (transfusion dependence)",
    },
}


def _make_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    g = GENE_META[gene]

    n = 40
    onset_map = {
        "MEFV": (3, 2), "TNFRSF1A": (8, 5), "MVK": (1, 1),
        "NLRP3": (0, 1), "PSTPIP1": (10, 5), "NOD2": (3, 2),
        "IL1RN": (0, 0), "LPIN2": (1, 1),
    }
    om, os_ = onset_map.get(gene, (5, 3))
    onset_months = max(0, int(rng.gauss(om, os_)))

    # Sex distribution
    male_frac_map = {
        "MEFV": 0.52, "TNFRSF1A": 0.50, "MVK": 0.50,
        "NLRP3": 0.50, "PSTPIP1": 0.52, "NOD2": 0.50,
        "IL1RN": 0.50, "LPIN2": 0.55,
    }
    frac = male_frac_map.get(gene, 0.50)
    male_n = int(round(n * frac + rng.gauss(0, 1.5)))
    male_n = max(15, min(25, male_n))
    female_n = n - male_n

    hsct_pct_map = {
        "MEFV": 0.0, "TNFRSF1A": 0.0, "MVK": 0.0,
        "NLRP3": 0.0, "PSTPIP1": 0.0, "NOD2": 0.0,
        "IL1RN": 0.0, "LPIN2": 0.0,
    }
    hsct_pct = hsct_pct_map.get(gene, 0.0) + rng.uniform(0, 0.02)

    amyloid_risk_map = {
        "MEFV": 0.25, "TNFRSF1A": 0.28, "MVK": 0.02,
        "NLRP3": 0.20, "PSTPIP1": 0.02, "NOD2": 0.03,
        "IL1RN": 0.05, "LPIN2": 0.02,
    }
    amyloid_pct = amyloid_risk_map.get(gene, 0.10) + rng.uniform(-0.02, 0.02)
    amyloid_pct = max(0.0, min(1.0, amyloid_pct))

    attack_dur_map = {
        "MEFV": "<72h", "TNFRSF1A": ">7 days", "MVK": "3-7 days",
        "NLRP3": "continuous/urticarial", "PSTPIP1": "weeks",
        "NOD2": "continuous", "IL1RN": "continuous neonatal",
        "LPIN2": "recurrent crises",
    }

    il1_response_map = {
        "MEFV": 0.90, "TNFRSF1A": 0.82, "MVK": 0.88,
        "NLRP3": 0.95, "PSTPIP1": 0.78, "NOD2": 0.55,
        "IL1RN": 1.00, "LPIN2": 0.80,
    }
    il1_resp = il1_response_map.get(gene, 0.80) + rng.uniform(-0.03, 0.03)
    il1_resp = max(0.0, min(1.0, il1_resp))

    return {
        "gene": gene,
        "n": n,
        "male_n": male_n,
        "female_n": female_n,
        "median_onset_months": onset_months,
        "hsct_pct": round(hsct_pct, 3),
        "amyloid_pct": round(amyloid_pct, 3),
        "il1_response_pct": round(il1_resp, 3),
        "attack_duration": attack_dur_map.get(gene, "variable"),
        "pathognomonic": g["pathognomonic"],
        "key_drug": g["key_drug"],
        "disease": g["disease"],
        "locus": g["locus"],
        "inheritance": g["inheritance"],
    }


def generate_overview() -> dict:
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    return {
        "atlas": "Hereditary-Autoinflammatory-Periodic-Fever-Atlas",
        "total_patients": 320,
        "genes": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "theme": "IL-1 and NF-kB dysregulation — recurrent sterile inflammation, serositis, osteomyelitis, skin/joint manifestations",
        "pathognomonic_highlights": [
            "MEFV/FMF: SHORT ATTACKS <72 h (vs TRAPS >7 d); serositis; colchicine curative; AA amyloid preventable",
            "TNFRSF1A/TRAPS: PROLONGED ATTACKS >7 DAYS; migratory myalgia + monocyte erythematous patches PATHOGNOMONIC; etanercept/canakinumab",
            "MVK/HIDS: 3-7 DAY ATTACKS; cervical lymphadenopathy; IgD >100 IU/mL PATHOGNOMONIC; Dutch p.V377I founder; vaccine-triggered",
            "NLRP3/CAPS: NON-PRURITIC URTICARIA; MWS sensorineural hearing loss PATHOGNOMONIC; NOMID neonatal continuous; canakinumab approved",
            "PSTPIP1/PAPA: STERILE destructive arthritis + PG (violaceous undermined border) + severe acne TRIAD PATHOGNOMONIC",
            "NOD2/Blau: GRANULOMATOUS UVEITIS + camptodactyly arthritis + lichenoid dermatitis TRIAD; lungs SPARED (vs sarcoidosis); AD-GOF NOD domain",
            "IL1RN/DIRA: NEONATAL periostitis + multifocal osteomyelitis + pustulosis PATHOGNOMONIC; anakinra CURATIVE within days",
            "LPIN2/Majeed: Multifocal osteomyelitis + CONGENITAL DYSERYTHROPOIETIC ANAEMIA PATHOGNOMONIC (CDA unique to Majeed in autoinflammatory diseases)",
        ],
        "gene_cohorts": cohorts,
    }


def generate_breakdown() -> dict:
    cohorts = {g: _make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)}
    return {
        "atlas": "Hereditary-Autoinflammatory-Periodic-Fever-Atlas",
        "pathway_groups": {
            "pyrin_inflammasome": {
                "genes": ["MEFV", "PSTPIP1"],
                "mechanism": "Pyrin (MEFV) senses Rho-GTPase inactivation; PSTPIP1 coiled-coil domain regulates pyrin via direct binding; mutations in either → unrestrained caspase-1 activation → IL-1β/IL-18 excess",
                "shared_feature": "Serositis (MEFV) and sterile joint/skin inflammation (PSTPIP1); respond to colchicine (MEFV) and IL-1 inhibitors (both)",
                "key_ddx": "MEFV: short attacks <72h; PSTPIP1/PAPA: arthritis in child + PG + acne without periodic fever pattern",
            },
            "tnfr1_shedding_defect": {
                "genes": ["TNFRSF1A"],
                "mechanism": "TNFR1 membrane shedding impaired → membrane-bound TNFR1 accumulates → constitutive/amplified TNF signalling → prolonged inflammatory attacks",
                "shared_feature": "Prolonged fever attacks >7 days; cysteine-disrupting mutations (C30Y, C52F) have highest AA amyloid risk; etanercept NOT infliximab",
                "key_ddx": "TRAPS: prolonged attacks + migratory myalgia skin patch; distinguish from FMF (short attacks), HIDS (cervical lymph), PFAPA (tonsillar exudate)",
            },
            "mevalonate_pathway_defect": {
                "genes": ["MVK"],
                "mechanism": "Mevalonate kinase deficiency → reduced isoprenoids → defective Rho-GTPase prenylation → caspase-1 hyperactivation → IL-1β excess; IgD elevation mechanism not fully established (likely isoprenoid-dependent B-cell activation)",
                "shared_feature": "Urine mevalonic acid elevated (during attacks); IgD >100 in HIDS; vaccine-triggered attacks; cervical lymphadenopathy",
                "key_ddx": "HIDS vs MVA: MVK activity <1% = MVA (severe; cerebellum, development), >1% = HIDS (milder); IgD less reliable for MVA",
            },
            "nlrp3_inflammasome_hyperactivation": {
                "genes": ["NLRP3"],
                "mechanism": "NLRP3 AD-GOF → spontaneous inflammasome assembly → constitutive IL-1β and IL-18 release; non-pruritic urticaria = neutrophil infiltration (not mast cell histamine); hearing loss from cochlear microvasculature IL-1β",
                "shared_feature": "Non-pruritic urticaria in all CAPS; sensorineural hearing loss in MWS/NOMID; canakinumab approved; cold sensitivity in FCAS",
                "key_ddx": "CAPS spectrum: FCAS (cold-triggered, no deafness), MWS (deafness +/- amyloid), NOMID (neonatal continuous + CNS); all NLRP3",
            },
            "nod2_nfkb_granulomatous": {
                "genes": ["NOD2"],
                "mechanism": "NOD2 AD-GOF (Blau domain mutations) → constitutive NF-κB activation via RICK/RIP2 → granuloma formation in uvea, joints, skin; distinct from IBD-associated NOD2 LOF mutations (16q12.1 same gene, different pathomechanism)",
                "shared_feature": "Non-caseating granulomas (biopsy confirms); triad without lung involvement; respond to anti-TNF; ophthalmology mandatory",
                "key_ddx": "Blau: AD, childhood, no lungs, NOD2 GOF; sarcoidosis: sporadic, adult, lungs common, NOD2 normal; JIA: no granulomas on skin/joint biopsy",
            },
            "il1ra_deficiency_osteomyelitis": {
                "genes": ["IL1RN", "LPIN2"],
                "mechanism": "IL1RN LOF (DIRA): absent endogenous IL-1Ra → unantagonised IL-1α and IL-1β → neonatal bone/skin; LPIN2 LOF (Majeed): lipid phosphatase defect → NLRP3 hyperactivation + CDA (erythroid IL-1 toxicity hypothesis)",
                "shared_feature": "Sterile multifocal osteomyelitis in both; anakinra effective; neonatal/early infantile onset; distinct by CDA (Majeed ONLY) vs pure osteomyelitis/pustulosis (DIRA)",
                "key_ddx": "DIRA: no CDA, periostitis prominent, Puerto Rican/Dutch founder; Majeed: CDA mandatory feature, Middle Eastern, LPIN2 sequencing",
            },
        },
        "attack_duration_cascade": [
            "STEP 1 — ATTACK DURATION: <72 h → MEFV/FMF (+ serositis check + colchicine response)",
            "STEP 2 — ATTACK DURATION: 3-7 days → MVK/HIDS (+ IgD + cervical LN + vaccine history)",
            "STEP 3 — ATTACK DURATION: >7 days (≥1-6 weeks) → TNFRSF1A/TRAPS (+ migratory myalgia + skin patches)",
            "STEP 4 — CONTINUOUS/URTICARIAL (neonatal): NLRP3/CAPS (cold-triggered → FCAS; deafness → MWS; neonatal → NOMID)",
            "STEP 5 — ARTHRITIS + SKIN (NO clear fever periodicity): PSTPIP1/PAPA (sterile destructive arthritis + PG + acne)",
            "STEP 6 — CHILDHOOD UVEITIS + ARTHRITIS (granulomatous): NOD2/Blau (camptodactyly + lichenoid dermatitis, no lung)",
            "STEP 7 — NEONATAL OSTEOMYELITIS + PUSTULOSIS: IL1RN/DIRA (periostitis on imaging; anakinra trial); if + CDA → LPIN2/Majeed",
            "STEP 8 — ETHNICITY PANEL: FMF → Turkish/Armenian/Arab/Sephardic; HIDS → Dutch/French p.V377I; PAPA/Blau → AD family history",
            "STEP 9 — BIOMARKERS: IgD >100 (HIDS/MVK); SAA/CRP persistent between attacks (all); serum IL-18 (CAPS/NLRP3 highest); urine mevalonic acid (MVK attacks)",
            "STEP 10 — IL-1 THERAPEUTIC TRIAL: anakinra 1-2 mg/kg/day × 4 weeks; dramatic response within 48-72 h confirms IL-1 pathway disease",
        ],
        "il1_treatment_spectrum": {
            "colchicine_first": "MEFV/FMF: colchicine 0.5-1 mg BD lifelong; prevents attacks AND AA amyloid; resistance <5%; escalate to IL-1 inhibitors for resistance",
            "canakinumab_approved": "TNFRSF1A, MVK, NLRP3 (all CAPS): canakinumab 150-300 mg SC every 4-8 weeks; once-monthly convenient; preferred for amyloid prevention",
            "anakinra_rapid": "IL1RN/DIRA: anakinra CURATIVE; also PSTPIP1/PAPA and Majeed/LPIN2; daily injection less convenient but faster response for acute crises",
            "anti_tnf_blau": "NOD2/Blau: adalimumab or infliximab for granulomatous uveitis; methotrexate add-on; IL-6 inhibitors for refractory",
            "avoid_infliximab_traps": "TNFRSF1A/TRAPS: infliximab NOT recommended (may paradoxically worsen); etanercept (TNF-receptor fusion) preferred over infliximab (anti-TNF antibody)",
        },
        "treatment_pathways": [
            {
                "gene": "MEFV",
                "disease": "FMF",
                "first_line": "Colchicine 0.5-1.0 mg twice daily, lifelong",
                "precision_therapy": "Canakinumab 150 mg SC every 8 weeks for colchicine-resistant/intolerant",
                "avoid": "Stopping colchicine (AA amyloid risk accumulates); NSAIDs alone",
                "monitoring": "SAA + urinary protein every 6-12 months for AA amyloid; serum colchicine if non-adherence suspected",
            },
            {
                "gene": "TNFRSF1A",
                "disease": "TRAPS",
                "first_line": "Canakinumab 150 mg SC every 8 weeks (preferred for structural mutations)",
                "precision_therapy": "Etanercept 25 mg SC twice weekly for moderate phenotypes; anakinra for acute attacks",
                "avoid": "Infliximab (may worsen); stopping anti-IL-1 without SAA monitoring",
                "monitoring": "SAA + urine protein every 3-6 months; genetic subtype (cysteine mutations = highest amyloid risk)",
            },
            {
                "gene": "MVK",
                "disease": "HIDS/MKD",
                "first_line": "Canakinumab 150-300 mg SC every 4-8 weeks",
                "precision_therapy": "Anakinra 1-2 mg/kg/day for acute attacks; counsel on vaccine-triggered attacks",
                "avoid": "Statins (may reduce residual MVK activity); delay in treatment during vaccine fever crisis",
                "monitoring": "Urine mevalonic acid during attacks; IgD yearly; growth/development in children",
            },
            {
                "gene": "NLRP3",
                "disease": "CAPS (FCAS/MWS/NOMID)",
                "first_line": "Canakinumab 2 mg/kg SC every 8 weeks (approved all CAPS)",
                "precision_therapy": "Higher doses for NOMID (up to 8 mg/kg); CSF analysis for meningitis in NOMID; audiological monitoring",
                "avoid": "NSAIDs/steroids alone (insufficient MWS/NOMID); untreated NOMID (irreversible CNS damage)",
                "monitoring": "Audiological testing every 6 months; MRI brain for meningitis (NOMID); SAA + IL-18",
            },
            {
                "gene": "PSTPIP1",
                "disease": "PAPA Syndrome",
                "first_line": "Anakinra or canakinumab (IL-1 inhibitors) + adalimumab/etanercept (anti-TNF)",
                "precision_therapy": "IL-1 blockade for arthritis; TNF blockade for PG; combination sometimes required",
                "avoid": "Surgical debridement of active PG lesions (pathergy; worsens ulceration); prolonged steroids alone",
                "monitoring": "Wound care for PG; ophthalmology if ocular PG; rheumatology + dermatology co-management",
            },
            {
                "gene": "NOD2",
                "disease": "Blau Syndrome",
                "first_line": "Adalimumab (anti-TNF) + methotrexate",
                "precision_therapy": "Tocilizumab (anti-IL-6) for TNF-refractory; anakinra in some refractory cases",
                "avoid": "Topical steroids alone for panuveitis; delayed anti-TNF start (blindness risk from panuveitis)",
                "monitoring": "Ophthalmology every 3 months; slit-lamp IOP; visual acuity; liver enzymes (methotrexate)",
            },
            {
                "gene": "IL1RN",
                "disease": "DIRA",
                "first_line": "Anakinra 1-3 mg/kg/day SC or IV (CURATIVE, lifelong)",
                "precision_therapy": "Canakinumab for anakinra maintenance (less frequent injections); dose titration by CRP/SAA",
                "avoid": "Antibiotics as monotherapy (sterile disease); stopping anakinra (relapse guaranteed); delay in diagnosis (rapid bone destruction)",
                "monitoring": "CRP/SAA; bone imaging every 3-6 months initially; injection-site reactions; growth monitoring",
            },
            {
                "gene": "LPIN2",
                "disease": "Majeed Syndrome",
                "first_line": "Anakinra or canakinumab (anti-IL-1) + haematology support for CDA",
                "precision_therapy": "Transfusion for symptomatic CDA; iron chelation if transfusion-dependent; anti-IL-1 may reduce transfusion need",
                "avoid": "Treating as CRMO without recognising CDA component; bisphosphonates without anti-IL-1",
                "monitoring": "Haemoglobin/reticulocytes; bone imaging; ferritin (transfusion iron loading); neurological (rare)",
            },
        ],
        "hlh_risk_table": {
            "MEFV": "AA amyloid risk (not HLH) ~25% untreated; MAS (macrophage activation) rare in severe FMF",
            "TNFRSF1A": "AA amyloid risk ~25-35% in structural mutations; MAS rare but reported",
            "MVK": "MAS reported in severe crises; AA amyloid rare in HIDS; severe MVA has higher MAS risk",
            "NLRP3": "AA amyloid 25-30% MWS untreated; NOMID: leptomeningeal inflammation dominant; MAS rare",
            "PSTPIP1": "MAS in severe PAPA documented; often associated with extreme CRP elevation",
            "NOD2": "Granuloma-mediated tissue damage; MAS rare; uveitis leading cause of morbidity",
            "IL1RN": "Life-threatening if untreated (systemic IL-1 storm); not classical HLH; anakinra curative",
            "LPIN2": "Not primary HLH; CDA is the haematological morbidity; anaemia can be severe",
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Autoinflammatory-Periodic-Fever-Atlas",
        "key_distinguishers": {
            "attack_duration_rule": "FMF <72 h → HIDS 3-7 d → TRAPS >7 d (weeks); CAPS/Blau/DIRA/Majeed continuous (no discrete attacks)",
            "mefv_colchicine_response": "FMF: colchicine response >90% frequency reduction; non-response redefines as colchicine-resistant FMF or atypical/alternative diagnosis",
            "traps_no_infliximab": "TNFRSF1A/TRAPS: infliximab CONTRAINDICATED (may paradoxically worsen); use etanercept (TNF receptor-Fc fusion protein, not antibody)",
            "hids_igd_marker": "MVK/HIDS: IgD >100 IU/mL in >80% pathognomonic; IgA also elevated; IgD normal in MVA (severe end); IgD normal in FMF, TRAPS, CAPS",
            "caps_non_pruritic_urticaria": "NLRP3/CAPS: urticaria non-pruritic (neutrophil-driven, not mast cell); distinguishes from allergic urticaria; cold-triggered = FCAS",
            "blau_vs_sarcoidosis": "NOD2/Blau: AD family history + childhood + no lung involvement + NOD2 GOF domain mutation; sarcoidosis: sporadic, adult, lung common, NOD2 normal",
            "dira_anakinra_curative": "IL1RN/DIRA: anakinra curative within 48-72 hours (neonatal response); failure to respond questions diagnosis; lifelong required",
            "majeed_cda_unique": "LPIN2/Majeed: congenital dyserythropoietic anaemia (CDA) is UNIQUE to Majeed in all autoinflammatory syndromes; CDA + multifocal osteomyelitis = Majeed until proven otherwise",
        },
        "treatment_matrix": {
            "colchicine_first_mefv": "MEFV: colchicine is FIRST-LINE and disease-modifying; prevents AA amyloid; start at diagnosis regardless of attack frequency",
            "canakinumab_triple_indication": "Canakinumab approved for TRAPS (TNFRSF1A), HIDS/MKD (MVK), and all CAPS (NLRP3) — the three IL-1β-pathway periodic fevers with regulatory approval",
            "anakinra_rapid_response": "Anakinra (IL-1Ra): fastest response (hours); daily injections; first-choice for DIRA (curative) and Majeed; acute PAPA/HIDS crises",
            "anti_tnf_blau_only": "Anti-TNF (adalimumab/etanercept): primarily for Blau/NOD2 granulomatous uveitis; second-line in other autoinflammatory syndromes",
            "il1_therapeutic_trial": "If uncertain diagnosis: anakinra 1-2 mg/kg/day × 4 weeks; dramatic CRP/SAA normalisation within 72 h confirms IL-1-mediated disease",
        },
        "definitions": [
            {
                "term": "MEFV — Pyrin; 781aa; 16p13.3; 86 kDa; AR-LOF (classical) and AD-LOF (low penetrance)",
                "definition": "Pyrin is a MEFV-encoded scaffolding protein containing B30.2/SPRY domain that senses Rho-GTPase inactivation (bacterial virulence factors). AR-LOF → defective pyrin-mediated caspase-1 suppression → IL-1β/IL-18 excess. FMF: most common hereditary autoinflammatory disease (Turkish/Armenian/Arab/Sephardic). Attacks 12-72 h, serositis, erysipelas-like erythema. p.M694V homozygous = most severe (highest amyloid risk). Colchicine first-line (CURATIVE mechanism: inhibits pyrin inflammasome assembly via MEFV). AA amyloidosis preventable with colchicine. Canakinumab/anakinra for colchicine-resistant. Seeds 4174."
            },
            {
                "term": "TNFRSF1A — TNFR1; 455aa; 12p13.31; 51 kDa; AD-LOF (structural) and low-penetrance variants",
                "definition": "TNFR1 is the primary signalling receptor for TNF-α. AD mutations → impaired ectodomain shedding → membrane-bound TNFR1 accumulation → sustained TNF signalling. TRAPS: prolonged attacks >7 days (up to 6 weeks). Migratory myalgia with erythematous monocyte-rich skin patches PATHOGNOMONIC. Periorbital oedema. Cysteine-disrupting mutations (C30Y, C52F, C70S) = structural variants, highest AA amyloid risk (~25-35%). Low-penetrance variants (R92Q, P46L) milder. Etanercept preferred over infliximab (infliximab may worsen). Canakinumab approved 2016. Seeds 4175."
            },
            {
                "term": "MVK — Mevalonate kinase; 396aa; 12q24.11; 43 kDa; AR-LOF",
                "definition": "Mevalonate kinase catalyses the second step in the mevalonate pathway. LOF → reduced geranylgeranyl-pyrophosphate → impaired Rho-GTPase prenylation → caspase-1 hyperactivation. HIDS/MKD: 3-7 day attacks, cervical lymphadenopathy, IgD >100 IU/mL (>80% of classical HIDS) PATHOGNOMONIC, vaccine-triggered attacks. Dutch p.V377I founder mutation (~80% of European HIDS alleles). Severe end: mevalonic aciduria (MVA) with cerebellar ataxia and dysmorphia. Urine mevalonic acid elevated during attacks. Canakinumab approved 2017. Seeds 4176."
            },
            {
                "term": "NLRP3 — Cryopyrin; 1036aa; 1q44; 118 kDa; AD-GOF",
                "definition": "NLRP3 (cryopyrin) is the central NOD-like receptor forming the NLRP3 inflammasome with ASC/PYCARD and pro-caspase-1 → IL-1β and IL-18 processing. AD-GOF mutations → spontaneous constitutive inflammasome assembly. CAPS spectrum: FCAS (cold-triggered, mildest), MWS (sensorineural hearing loss PATHOGNOMONIC, amyloid 25-30%), NOMID (neonatal continuous urticaria + aseptic meningitis + arthropathy, most severe). Non-pruritic neutrophilic urticaria distinguishes from allergic urticaria. Canakinumab approved (all CAPS) — FDA 2009, EMA 2009. Seeds 4177."
            },
            {
                "term": "PSTPIP1 — Proline-serine-threonine phosphatase interacting protein 1; 416aa; 15q24.3; 48 kDa; AD",
                "definition": "PSTPIP1 coiled-coil domain binds pyrin (MEFV) and regulates pyrin inflammasome. AD mutations (A230T, E250K) disrupt PSTPIP1-pyrin binding → unrestrained caspase-1 → IL-1β excess. PAPA Syndrome: sterile pyogenic destructive arthritis in childhood (sterile synovial fluid 50,000-80,000 WBC/µL PATHOGNOMONIC) + pyoderma gangrenosum (violaceous undermined border, pathergy positive) + severe cystic acne. IL-1 inhibitors (anakinra/canakinumab) + anti-TNF (adalimumab/etanercept). Avoid surgical debridement of active PG (pathergy worsens). Seeds 4178."
            },
            {
                "term": "NOD2 — NOD2; 1040aa; 16q12.1; 115 kDa; AD-GOF (Blau; NOD domain) — distinct from IBD-associated AR-LOF (LRR domain)",
                "definition": "NOD2 intracellular pattern recognition receptor detects muramyl dipeptide. AD-GOF mutations in NOD domain → constitutive NF-κB activation via RICK/RIP2 → non-caseating granuloma formation (NOT IBD-associated LOF). Blau Syndrome: childhood-onset granulomatous triad — panuveitis (can blind), camptodactyly boggy arthritis, lichenoid granulomatous dermatitis. Lungs SPARED (distinguishes from sarcoidosis). AD family history. Adalimumab/anti-TNF for sight-threatening uveitis + methotrexate. Seeds 4179."
            },
            {
                "term": "IL1RN — Interleukin-1 receptor antagonist; 177aa (mature); 2q14.1; 25 kDa; AR-LOF",
                "definition": "IL-1Ra (encoded by IL1RN) is the endogenous physiological inhibitor of IL-1α and IL-1β signalling — competes for IL-1R1 without signalling. AR-LOF → absent IL-1Ra → completely unantagonised IL-1 → DIRA. DIRA: neonatal sterile multifocal osteomyelitis + periostitis (periosteal new bone on radiograph PATHOGNOMONIC) + sterile pustulosis. Life-threatening without treatment. Puerto Rican (p.E77X) and Dutch/Newfoundland (p.C80S) founders. Anakinra (recombinant IL-1Ra) IV/SC CURATIVE within 48-72 h. Lifelong anakinra required. Seeds 4180."
            },
            {
                "term": "LPIN2 — Phosphatidate phosphatase LPIN2; 986aa; 18p11.31; 111 kDa; AR-LOF",
                "definition": "Lipin-2 is a Mg2+-dependent phosphatidate phosphatase in lipid metabolism; also regulates NLRP3 inflammasome via lipid signalling. AR-LOF → NLRP3 hyperactivation + dyserythropoiesis. Majeed Syndrome: sterile recurrent multifocal osteomyelitis + CONGENITAL DYSERYTHROPOIETIC ANAEMIA (CDA; PATHOGNOMONIC — CDA is unique to Majeed in all hereditary autoinflammatory diseases) + neutrophilic dermatosis. Middle Eastern origin (Jordanian family first described). Anakinra or canakinumab effective (bone crises + CDA improvement). Seeds 4181."
            },
        ],
    }
