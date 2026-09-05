#!/usr/bin/env python3
"""HSAN-Atlas — Complete 8-Gene Hereditary Sensory and Autonomic Neuropathy Atlas
SPTLC1  (Serine palmitoyltransferase long-chain base subunit 1; 473 aa; 9q22.33; AD;
          HSAN1A — most common hereditary sensory neuropathy worldwide;
          deoxy-sphingolipids neurotoxic → small fibre sensory loss → painless plantar ulcers → amputations;
          L-serine REDUCES deoxy-SL → partial neuroprotection;
          AVOID VINCRISTINE/CISPLATIN ABSOLUTELY — precipitates acute axonal collapse;
          Cys133Trp/Cys133Tyr and Val144Asp most common variants) ·
ELP1    (Elongator complex protein 1 / IKBKAP; 1332 aa; 9q31.3; AR;
          HSAN3 = Familial Dysautonomia (FD) / Riley–Day syndrome;
          c.2204+6T>C splice-site FOUNDER MUTATION > 99.5% of Ashkenazi Jewish alleles;
          AUTONOMIC CRISES: hypertensive episodes, cyclic vomiting, diaphoresis — management is mainstay;
          progressive cerebellar ataxia, dysphagia, scoliosis, reduced taste;
          TAUROURSODEOXYCHOLIC ACID (TUDCA): reduces ELP1-mRNA mis-splicing → partial protein restoration) ·
NTRK1   (Neurotrophic tyrosine receptor kinase 1; 796 aa; 1q23.1; AR;
          HSAN4 = CIPA (Congenital Insensitivity to Pain with Anhidrosis);
          HYPERTHERMIA IS THE KILLER — no sweating → core temperature 41 °C+ in warm environments → fatal;
          Dental/orthopaedic injuries undetected → Charcot joints, fractured teeth;
          Intellectual disability in ~50%; normal childhood a management challenge;
          Fever precautions, dental guards, protective footwear mandatory) ·
NGFB    (Nerve growth factor, beta subunit; 241 aa; 1p13.2; AR;
          HSAN5 — selective loss of high-threshold (deep pain) A-delta and C-fibre nociception;
          Light touch/vibration/proprioception PRESERVED — key DDx from other HSAN types;
          p.Arg221Trp Norwegian founder mutation; painless fractures/burns;
          NGF drives DRG survival during development — NGFB LOF → selective nociceptor loss) ·
FAM134B (Reticulophagy regulator 1 / RETREG1; 491 aa; 5p15.1; AR;
          HSAN2B — most severe HSAN2 subtype; ER-reticulophagy receptor;
          NEONATAL ONSET with self-mutilating behaviour from infancy;
          Severe pan-modal sensory loss + profound autonomic instability;
          Tongue biting, digit amputations, joint destruction;
          ER stress accumulation → DRG neuron apoptosis) ·
DNMT1   (DNA methyltransferase 1; 1616 aa; 19p13.2; AD;
          HSAN1E = Hereditary Sensory Neuropathy with Dementia and Hearing Loss (HSAN-D);
          TRIAD: adult-onset sensory neuropathy + SNHL (sensorineural) + dementia (frontotemporal pattern);
          Methyltransferase-targeting domain mutations → premature protein degradation → hypomethylation;
          UNIQUE among AD sensory neuropathies for cognitive + hearing involvement) ·
WNK1    (WNK lysine-deficient protein kinase 1; 2382 aa; 12p13.33; AR;
          HSAN2A — requires HSN2 isoform-specific mutations (neuronal exon 12);
          STANDARD SANGER MISSES IT: HSN2 exon not in reference panels — target deep intronic/exon sequencing;
          Severe pan-modal sensory loss, autonomic instability, lancinating pain in early disease;
          WNK1/HSN2 regulates ROMK/KCC2 in sensory neurons) ·
PRDM12  (PR/SET domain-containing protein 12; 516 aa; 9q34.12; AR;
          HSAN8 — global pain insensitivity WITHOUT anhidrosis;
          SWEATING PRESERVED — key DDx from CIPA/NTRK1 (anhidrotic);
          Temperature sensing IMPAIRED but autonomic function relatively preserved;
          PR domain zinc-finger TF regulates nociceptor specification during development;
          Life expectancy normal; corneal abrasions a major complication)
320-patient aggregate cohort (8 × 40, seeds 1366-1373)
"""

import random

SEED_BASE = 1366

HSAN_GENES = [
    # ── SPTLC1 — HSAN1A ──
    {
        "gene": "SPTLC1",
        "protein": "Serine Palmitoyltransferase Long-Chain Base Subunit 1",
        "alias": (
            "SPTLC1; OMIM gene 605712; HSAN type 1A #162400 (AD); "
            "9q22.33; 473 aa; ~57 kDa; catalytic subunit of the serine palmitoyltransferase (SPT) complex; "
            "SPT catalyses the first and rate-limiting step in de novo sphingolipid biosynthesis: "
            "palmitoyl-CoA + L-serine → 3-ketodihydrosphingosine (KDS); "
            "pathogenic variants (Cys133Trp, Cys133Tyr, Val144Asp, Gly387Ala) shift substrate selectivity from "
            "L-serine to L-alanine or L-glycine → deoxysphingolipids (1-deoxySL) accumulate; "
            "1-deoxySL are neurotoxic: selectively destroy small-calibre dorsal root ganglion neurons "
            "(C-fibres, A-delta fibres) → length-dependent sensory neuropathy; "
            "Cys133Trp is most common (50% of HSAN1 pedigrees); Val144Asp second most common; "
            "most common hereditary sensory neuropathy worldwide"
        ),
        "aa": "473 aa",
        "kDa": "~57 kDa",
        "locus": "9q22.33",
        "omim_gene": 605712,
        "omim_disease": 162400,
        "inheritance": (
            "AD — heterozygous pathogenic variant; incomplete penetrance but high (>80% by age 50); "
            "phenotype: sensory loss begins in feet in 3rd-4th decade → ascends; "
            "motor involvement minimal or absent (key DDx from CMT); "
            "pain insensitivity → plantar ulcers → osteomyelitis → amputation (untreated natural history)"
        ),
        "gene_class": (
            "SPTLC1 encodes the catalytic subunit of the obligate SPT heterodimer (SPTLC1 + SPTLC2). "
            "Pathogenic variants cause toxic gain-of-function: SPT accepts alanine instead of serine "
            "as substrate → 1-deoxysphinganine → 1-deoxysphingosine accumulate. "
            "These deoxy-SL cannot be degraded by canonical ceramidase/sphingomyelinase pathways "
            "and accumulate to cytotoxic levels in DRG neurons."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("Cys133Trp / p.C133W heterozygous (most common)", 0.50),
            ("Val144Asp / p.V144D heterozygous", 0.25),
            ("Cys133Tyr / p.C133Y heterozygous", 0.15),
            ("Gly387Ala / p.G387A heterozygous", 0.10),
        ],
        "age_onset_years_range": (15, 40),
        "sex_ratio_M": 0.55,
        "rates": {
            "sensory_loss": 1.00,
            "autonomic": 0.15,
            "pain_insensitivity": 0.85,
            "anhidrosis": 0.10,
            "plantar_ulcers": 0.70,
            "mutilations": 0.30,
            "cognitive_decline": 0.00,
            "hearing_loss": 0.00,
            "hyperthermia": 0.05,
            "gi_dysmotility": 0.10,
        },
        "hallmarks": [
            "Length-dependent sensory neuropathy (feet first): loss of pain + temperature, preserved vibration",
            "Painless plantar ulcers → osteomyelitis → amputation (untreated natural history)",
            "DEOXY-SPHINGOLIPID ACCUMULATION: neurotoxic; L-serine supplementation reduces deoxy-SL",
            "AVOID VINCRISTINE/CISPLATIN ABSOLUTELY: acute axonal collapse in SPTLC1 carriers",
            "Motor involvement minimal or absent (key DDx from CMT1/2)",
            "Lancinating shooting pains in early disease paradoxically (before sensory loss)",
            "Cys133Trp = most common variant (50% of HSAN1A pedigrees worldwide)",
            "L-serine oral supplementation (400 mg/kg/day): competes with alanine → reduces 1-deoxySL",
        ],
        "treatment_alerts": [
            "VINCRISTINE/CISPLATIN ABSOLUTELY CONTRAINDICATED: acute axonal degeneration",
            "L-serine supplementation: reduces 1-deoxySL by up to 50%; start early (pre-symptomatic)",
            "Foot care protocol: daily inspection, protective footwear, orthotics mandatory",
            "Wound care + orthopaedic surgery for plantar ulcers; osteomyelitis → 6-week IV antibiotics",
            "Pain management: despite sensory loss, neuropathic pain in early disease — gabapentin/duloxetine",
        ],
        "organ_system": "peripheral nervous system (sensory)",
        "primary_treatment": "L-serine supplementation + foot care; AVOID vincristine/cisplatin",
    },

    # ── ELP1/IKBKAP — HSAN3 / Familial Dysautonomia ──
    {
        "gene": "ELP1",
        "protein": "Elongator Complex Protein 1 (IKBKAP)",
        "alias": (
            "ELP1 (alias IKBKAP); OMIM gene 603722; Familial Dysautonomia (HSAN3) #223900 (AR); "
            "9q31.3; 1332 aa; ~150 kDa; scaffold subunit of the Elongator complex (six-subunit tRNA modification complex); "
            "Elongator regulates tRNA wobble-base uridine modification (mcm5s2U34) → affects translation of A/U-ending codons; "
            "c.2204+6T>C splice-site variant (p.R696PfsX7 at protein level) is the Ashkenazi Jewish founder: "
            ">99.5% of all FD alleles globally; p.P914L missense is the second allele; "
            "c.2204+6T>C → exon 20 skipping → truncated mRNA → dramatically reduced ELP1 protein (tissue-specific: "
            "more severe in neurons); "
            "FD affects Ashkenazi Jewish only (carrier frequency 1/27 Ashkenazi Jews); "
            "most famous hereditary autonomic neuropathy in history"
        ),
        "aa": "1332 aa",
        "kDa": "~150 kDa",
        "locus": "9q31.3",
        "omim_gene": 603722,
        "omim_disease": 223900,
        "inheritance": (
            "AR — compound heterozygous or homozygous; virtually always c.2204+6T>C / c.2204+6T>C in Ashkenazi Jews; "
            "ethnically restricted: FD occurs exclusively in Ashkenazi Jewish individuals; "
            "carrier frequency 1:27; birth prevalence ~1:3700 Ashkenazi births"
        ),
        "gene_class": (
            "ELP1 is the scaffold subunit of the 6-subunit Elongator complex, which catalyses tRNA modification "
            "at wobble position U34. Without proper U34 modification, translation elongation is impaired "
            "for A/U-rich codons → ribosome stalling → misfolded protein accumulation → neuronal apoptosis, "
            "particularly of autonomic and sensory neurons derived from neural crest."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("c.2204+6T>C homozygous (Ashkenazi founder)", 0.75),
            ("c.2204+6T>C / p.P914L compound heterozygous", 0.20),
            ("c.2204+6T>C / other splice variant", 0.05),
        ],
        "age_onset_years_range": (0, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "sensory_loss": 0.90,
            "autonomic": 1.00,
            "pain_insensitivity": 0.50,
            "anhidrosis": 0.30,
            "plantar_ulcers": 0.15,
            "mutilations": 0.05,
            "cognitive_decline": 0.10,
            "hearing_loss": 0.00,
            "hyperthermia": 0.40,
            "gi_dysmotility": 1.00,
        },
        "hallmarks": [
            "AUTONOMIC CRISES: episodic hypertension + vomiting + diaphoresis + tachycardia + retching (crisis management is mainstay)",
            "Reduced/absent tear production (alacrima): Schirmer test <5 mm PATHOGNOMONIC",
            "Absent fungiform papillae on tongue: NO taste on tip/sides of tongue",
            "Dysphagia + aspiration pneumonia: leading cause of death (supraglottic laryngoplasty + G-tube)",
            "Progressive cerebellar ataxia: wheelchair by 3rd-4th decade",
            "Scoliosis (80-90%): Cobb angle progression → spinal fusion mandatory",
            "Ashkenazi Jewish exclusively: c.2204+6T>C >99.5%; carrier 1:27 Ashkenazi",
            "TAUROURSODEOXYCHOLIC ACID (TUDCA): reduces exon 20 skipping → partial ELP1 restoration",
        ],
        "treatment_alerts": [
            "AUTONOMIC CRISIS PROTOCOL: IV benzodiazepam (diazepam 0.1 mg/kg) + antiemetic (ondansetron) + fluid; Clonidine for hypertension",
            "TUDCA (tauroursodeoxycholic acid): 15 mg/kg/day — reduces mis-splicing; start at diagnosis",
            "G-tube: consider early (aspiration protection); swallowing study every 2 years",
            "Scoliosis: spinal fusion when Cobb >40° (curve progression relentless)",
            "Ophthalmology: lubricant drops every 2 hours + moisture chamber overnight (corneal damage from alacrima)",
            "Blood pressure monitoring: supine hypertension + orthostatic hypotension; no single antihypertensive suits both",
        ],
        "organ_system": "autonomic + sensory nervous system",
        "primary_treatment": "TUDCA + autonomic crisis management + scoliosis surgery + G-tube",
    },

    # ── NTRK1 — HSAN4 / CIPA ──
    {
        "gene": "NTRK1",
        "protein": "Neurotrophic Tyrosine Receptor Kinase 1 (TrkA)",
        "alias": (
            "NTRK1 (alias TRKA); OMIM gene 191315; CIPA / HSAN4 #256800 (AR); "
            "1q23.1; 796 aa; ~87 kDa; transmembrane receptor tyrosine kinase; "
            "high-affinity receptor for Nerve Growth Factor (NGF); "
            "NGF/TrkA signalling is essential for nociceptor survival during embryonic development; "
            "NTRK1 LOF → absence of NGF → DRG nociceptors (pain, temp) and sympathetic neurons die in utero; "
            ">100 pathogenic variants reported; more prevalent in consanguineous populations (Middle East, Japan); "
            "CIPA = Congenital Insensitivity to Pain with Anhidrosis: "
            "no pain perception from birth + absence of sweating → hyperthermia"
        ),
        "aa": "796 aa",
        "kDa": "~87 kDa",
        "locus": "1q23.1",
        "omim_gene": 191315,
        "omim_disease": 256800,
        "inheritance": (
            "AR — homozygous or compound heterozygous; "
            "consanguineous families common; higher prevalence Middle East and Japan; "
            "no sex predilection"
        ),
        "gene_class": (
            "NTRK1 encodes the TrkA receptor tyrosine kinase for NGF. "
            "NGF-TrkA signalling during fetal development is mandatory for survival of nociceptor DRG neurons "
            "and sympathetic chain neurons. Without TrkA, these neurons undergo apoptosis before birth. "
            "Result: absence of A-delta and C-fibre nociceptors and sympathetic innervation of sweat glands."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("Truncating variant (nonsense/frameshift) homozygous", 0.45),
            ("Missense variant kinase domain (Arg649Gln/Ser) homozygous", 0.25),
            ("Splice-site variant homozygous", 0.15),
            ("Compound heterozygous truncating + missense", 0.15),
        ],
        "age_onset_years_range": (0, 2),
        "sex_ratio_M": 0.55,
        "rates": {
            "sensory_loss": 1.00,
            "autonomic": 0.95,
            "pain_insensitivity": 1.00,
            "anhidrosis": 1.00,
            "plantar_ulcers": 0.55,
            "mutilations": 0.60,
            "cognitive_decline": 0.45,
            "hearing_loss": 0.00,
            "hyperthermia": 0.90,
            "gi_dysmotility": 0.20,
        },
        "hallmarks": [
            "HYPERTHERMIA IS THE LEADING KILLER: no sweating (anhidrosis) → fever 41°C+ in warm environments → fatal",
            "Congenital total pain insensitivity: fractures, burns, dental injuries go undetected",
            "Anhidrosis: sweat glands intact but no sympathetic innervation — cooling by external means mandatory",
            "Charcot arthropathy: repeated undetected trauma → joint destruction",
            "Dental injuries: tongue/lip biting, fractured teeth — dental guard from infancy",
            "Intellectual disability ~50%: sympathetic chain neurons also depend on NGF",
            "Self-injurious behaviour: lip/finger biting, eye gouging; NOT psychiatric but pain-insensitive",
            "Hyperventilation and breath-holding spells: autonomic breathing regulation impaired",
        ],
        "treatment_alerts": [
            "HYPERTHERMIA PREVENTION: Cooling vest, misting fan, never leave unattended in heat; rectal temperature monitoring",
            "Dental guard from 6 months: prevents tongue/lip maceration",
            "Fracture surveillance: regular X-ray surveillance; treat fractures before Charcot joint develops",
            "Ophthalmology: corneal abrasions from analgesia-free eye injury; lubricant mandatory",
            "Developmental surveillance: ~50% intellectual disability — early educational support",
            "Orthopaedic monitoring: Charcot joint debridement vs arthrodesis when severe",
        ],
        "organ_system": "sensory + sympathetic nervous system",
        "primary_treatment": "hyperthermia prevention + dental guard + fracture surveillance; no disease-modifying therapy",
    },

    # ── NGFB — HSAN5 ──
    {
        "gene": "NGFB",
        "protein": "Nerve Growth Factor Beta Subunit",
        "alias": (
            "NGFB (alias NGF); OMIM gene 162030; HSAN5 #608654 (AR); "
            "1p13.2; 241 aa (~119 aa mature NGF after pre-pro processing); ~13 kDa mature dimer; "
            "secreted neurotrophin; ligand for NTRK1 (TrkA) and NGFR (p75-NTR); "
            "essential survival factor for nociceptive DRG neurons and sympathetic neurons; "
            "p.Arg221Trp (c.661C>T) is the Norwegian founder mutation (described in single large pedigree); "
            "HSAN5 phenotype: SELECTIVE loss of high-threshold (deep pain) nociceptors; "
            "critically: light touch, vibration, and proprioception PRESERVED; "
            "autonomic features milder than HSAN4 (NTRK1) — partial residual NGF signalling"
        ),
        "aa": "241 aa (119 aa mature)",
        "kDa": "~13 kDa mature NGF dimer",
        "locus": "1p13.2",
        "omim_gene": 162030,
        "omim_disease": 608654,
        "inheritance": (
            "AR — homozygous in the Norwegian pedigree (p.Arg221Trp founder); "
            "additional pedigrees with compound heterozygous variants described; "
            "heterozygous carriers: normal — no haploinsufficiency phenotype"
        ),
        "gene_class": (
            "NGFB encodes NGF, the prototypical neurotrophin. NGF is secreted by target tissues "
            "and retrogradely transported to DRG cell bodies to signal via TrkA/p75-NTR. "
            "p.Arg221Trp disrupts the NGF dimer interface → reduced secretion and TrkA binding affinity. "
            "Residual NGF function explains the SELECTIVE phenotype: only high-threshold nociceptors "
            "(most NGF-dependent) are lost; low-threshold mechanoreceptors are preserved."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("p.Arg221Trp (c.661C>T) Norwegian founder homozygous", 0.70),
            ("p.Arg221Trp compound heterozygous with missense", 0.15),
            ("Novel missense dimerisation domain compound heterozygous", 0.15),
        ],
        "age_onset_years_range": (0, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "sensory_loss": 0.90,
            "autonomic": 0.35,
            "pain_insensitivity": 0.85,
            "anhidrosis": 0.20,
            "plantar_ulcers": 0.35,
            "mutilations": 0.15,
            "cognitive_decline": 0.00,
            "hearing_loss": 0.00,
            "hyperthermia": 0.20,
            "gi_dysmotility": 0.15,
        },
        "hallmarks": [
            "SELECTIVE deep pain insensitivity: high-threshold nociceptors (A-delta, C-fibre) absent",
            "Light touch, vibration, and proprioception PRESERVED: key DDx from other HSAN types",
            "Painless deep-tissue injuries: muscle tears, visceral pain absent, bone fractures undetected",
            "Sweating mostly preserved (partial): milder anhidrosis than CIPA/NTRK1",
            "Norwegian founder: p.Arg221Trp in single large consanguineous pedigree",
            "Intellectual function normal: NGF5 does not affect sympathetic cognitive development",
            "Life expectancy approaches normal: morbidity from recurrent undetected injuries",
            "Corneal reflex preserved: light-touch fibres intact (unlike HSAN4)",
        ],
        "treatment_alerts": [
            "Deep injury surveillance: visceral and bone injury can be catastrophic and silent",
            "Fracture monitoring: regular bone X-ray, especially weight-bearing joints",
            "No disease-modifying therapy: NGF protein replacement theoretically rational but not clinically approved",
            "Genetic counselling: AR with Norwegian founder — cascade test family members",
            "Scoliosis: proprioception preserved but pain insensitivity may delay presentation",
        ],
        "organ_system": "peripheral nociceptive nervous system",
        "primary_treatment": "injury surveillance + fracture prevention; no disease-modifying therapy",
    },

    # ── FAM134B/RETREG1 — HSAN2B ──
    {
        "gene": "FAM134B",
        "protein": "Reticulophagy Regulator 1 (RETREG1)",
        "alias": (
            "FAM134B (alias RETREG1); OMIM gene 613114; HSAN2B #613115 (AR); "
            "5p15.1; 491 aa; ~54 kDa; ER-reticulophagy receptor; "
            "LIR (LC3-interacting region) motif + reticulon homology domain (RHD) — curves ER tubules + recruits autophagosome; "
            "ER-reticulophagy: selective autophagic degradation of excess/damaged ER; "
            "FAM134B LOF → ER expansion + UPR (unfolded protein response) → DRG neuron apoptosis; "
            "phenotype: most severe HSAN2 subtype — neonatal onset, pan-modal sensory loss + profound autonomic instability; "
            "self-mutilating behaviour from infancy (tongue biting, digit amputations); "
            "found in non-Ashkenazi populations globally"
        ),
        "aa": "491 aa",
        "kDa": "~54 kDa",
        "locus": "5p15.1",
        "omim_gene": 613114,
        "omim_disease": 613115,
        "inheritance": (
            "AR — homozygous or compound heterozygous; "
            "consanguineous pedigrees predominate; "
            "ethnically diverse (non-Ashkenazi); "
            "no founder mutation — multiple private variants"
        ),
        "gene_class": (
            "FAM134B/RETREG1 is an ER-resident receptor containing a reticulon homology domain "
            "that curves ER tubular membranes and an LIR motif that recruits LC3/ATG8-family proteins "
            "to initiate reticulophagy (selective ER autophagy). "
            "LOF → ER cannot be selectively degraded → protein aggregates accumulate → chronic ER stress → "
            "unfolded protein response (UPR) → IRE1/PERK/ATF6 activation → DRG neuron death. "
            "Motor neurons are relatively spared (motor neuron ER stress threshold is lower)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("Nonsense/frameshift homozygous (truncating LOF)", 0.55),
            ("Splice-site homozygous", 0.20),
            ("Missense LIR-motif disruption compound heterozygous", 0.15),
            ("Large deletion homozygous", 0.10),
        ],
        "age_onset_years_range": (0, 3),
        "sex_ratio_M": 0.50,
        "rates": {
            "sensory_loss": 1.00,
            "autonomic": 0.85,
            "pain_insensitivity": 0.90,
            "anhidrosis": 0.60,
            "plantar_ulcers": 0.50,
            "mutilations": 0.80,
            "cognitive_decline": 0.10,
            "hearing_loss": 0.00,
            "hyperthermia": 0.50,
            "gi_dysmotility": 0.55,
        },
        "hallmarks": [
            "NEONATAL/INFANTILE onset: self-mutilating behaviour from the first months of life",
            "Pan-modal sensory loss: pain, temperature, vibration, touch all severely reduced",
            "Tongue biting → self-amputation; digit biting → osteomyelitis; elbow/knee ulcers",
            "ER-RETICULOPHAGY: ER expansion → UPR → DRG neuron apoptosis (mechanistic basis)",
            "Profound autonomic instability: BP fluctuations, bradycardia/tachycardia, temperature dysregulation",
            "Motor function relatively preserved (distinguishes from SMA/ALS)",
            "Most severe HSAN2 subtype: earlier onset + more mutilation than WNK1/KIF1A HSAN2",
            "GI dysmotility: gastroparesis, constipation (autonomic enteric involvement)",
        ],
        "treatment_alerts": [
            "Physical restraint/protective padding from infancy: mittens, elbow pads, padded mouthguard",
            "Dental extraction: some centres extract primary teeth early to prevent tongue/lip injury",
            "Wound care: chronic ulcer management; osteomyelitis prophylaxis; orthopaedic review",
            "Autonomic monitoring: 24h BP, ECG (bradycardia risk); atropine for bradycardia episodes",
            "GI: jejunal feeding if severe gastroparesis; pro-motility agents",
            "No disease-modifying therapy: gene therapy under investigation",
        ],
        "organ_system": "sensory + autonomic nervous system",
        "primary_treatment": "protective padding + wound care + autonomic monitoring; no disease-modifying therapy",
    },

    # ── DNMT1 — HSAN1E ──
    {
        "gene": "DNMT1",
        "protein": "DNA Methyltransferase 1",
        "alias": (
            "DNMT1; OMIM gene 126375; HSAN1E / HSAND #614116 (AD); "
            "19p13.2; 1616 aa; ~183 kDa; maintenance DNA methyltransferase; "
            "catalyses transfer of methyl group from SAM to cytosine in CpG dinucleotides during DNA replication; "
            "replication-foci targeting sequence (RFTS domain) mutations cause HSAN1E: Tyr495Cys, Ala570Val, Val606Phe; "
            "RFTS mutations → DNMT1 protein misfolding → premature ubiquitin-proteasome degradation → "
            "hypomethylation in dividing neurons → genome instability; "
            "UNIQUE TRIAD: adult-onset peripheral sensory neuropathy + sensorineural hearing loss + "
            "frontotemporal dementia (no other HSAN type causes all three)"
        ),
        "aa": "1616 aa",
        "kDa": "~183 kDa",
        "locus": "19p13.2",
        "omim_gene": 126375,
        "omim_disease": 614116,
        "inheritance": (
            "AD — heterozygous pathogenic variant in RFTS domain; "
            "penetrance high (>95% by age 60); "
            "onset typically 3rd-5th decade; "
            "phenotype: hearing loss often first → then sensory neuropathy → then dementia (variable order)"
        ),
        "gene_class": (
            "DNMT1 is the major maintenance methyltransferase, restoring hemimethylated CpG sites "
            "during DNA replication. The RFTS domain autoinhibits catalytic activity when not at the "
            "replication fork. RFTS domain mutations (Tyr495Cys etc.) disrupt the autoinhibitory interface → "
            "the mutant protein is recognised as misfolded → accelerated proteasomal degradation → "
            "reduced DNMT1 activity → CpG hypomethylation in post-mitotic neurons → gene dysregulation → "
            "neuronal death in DRG (sensory neuropathy), cochlear hair cells (SNHL), and frontal/temporal cortex (dementia)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("p.Tyr495Cys (c.1484A>G) RFTS domain heterozygous", 0.45),
            ("p.Ala570Val (c.1709C>T) RFTS domain heterozygous", 0.30),
            ("p.Val606Phe (c.1816G>T) RFTS domain heterozygous", 0.25),
        ],
        "age_onset_years_range": (25, 55),
        "sex_ratio_M": 0.50,
        "rates": {
            "sensory_loss": 1.00,
            "autonomic": 0.20,
            "pain_insensitivity": 0.55,
            "anhidrosis": 0.10,
            "plantar_ulcers": 0.20,
            "mutilations": 0.05,
            "cognitive_decline": 0.85,
            "hearing_loss": 0.90,
            "hyperthermia": 0.05,
            "gi_dysmotility": 0.10,
        },
        "hallmarks": [
            "TRIAD: adult-onset sensory neuropathy + SNHL (sensorineural) + frontotemporal dementia (UNIQUE among HSAN)",
            "SNHL often FIRST (audiogram abnormal before neuropathy symptoms)",
            "Frontotemporal dementia pattern: executive dysfunction, personality change, language decline",
            "Length-dependent sensory loss: feet first, similar to other HSAN1 but with triple involvement",
            "RFTS domain mutations exclusively: Tyr495Cys/Ala570Val/Val606Phe — targeted testing",
            "Adult onset (25-55y): distinguishes from HSAN1A (similar sensory) by dementia + SNHL",
            "Anticipation: not demonstrated (AD, not trinucleotide repeat)",
            "Methyltransferase catalytic domain: NOT mutated — RFTS domain misfolding is the mechanism",
        ],
        "treatment_alerts": [
            "Hearing aids: SNHL progressive — audiology every 12 months; cochlear implant if profound loss",
            "Dementia management: frontotemporal dementia protocol; avoid anticholinergics",
            "Foot care: sensory neuropathy → plantar ulcer risk (identical to HSAN1A foot protocol)",
            "Genetic counselling: AD inheritance — 50% risk each child; presymptomatic testing available",
            "No disease-modifying therapy: DNMT1 enzyme replacement not feasible; epigenetic therapy investigational",
            "Cognitive assessment: baseline neuropsychology at diagnosis; annual reassessment",
        ],
        "organ_system": "peripheral sensory + central nervous system + auditory",
        "primary_treatment": "hearing aids + dementia management + foot care; no disease-modifying therapy",
    },

    # ── WNK1 — HSAN2A ──
    {
        "gene": "WNK1",
        "protein": "WNK Lysine-Deficient Protein Kinase 1",
        "alias": (
            "WNK1 (alias HSN2 isoform); OMIM gene 605232; HSAN2A #201300 (AR); "
            "12p13.33; 2382 aa (canonical); ~270 kDa; serine-threonine kinase; "
            "CRITICAL: ONLY neuronal HSN2 exon-containing isoform causes HSAN2A; "
            "HSN2 exon is NOT in standard RefSeq transcript — routine Sanger/clinical exome MISSES it; "
            "WNK1/HSN2 isoform regulates KCC2 (neuronal K-Cl cotransporter) and ROMK in sensory neurons; "
            "WNK1 canonical isoform mutations → Gordon syndrome (pseudohypoaldosteronism type IIA, hypertension); "
            "HSAN2A mutations are exclusively in HSN2 exon; "
            "severe pan-modal sensory loss from infancy; autonomic features present"
        ),
        "aa": "2382 aa (canonical); HSN2 isoform-specific exon",
        "kDa": "~270 kDa",
        "locus": "12p13.33",
        "omim_gene": 605232,
        "omim_disease": 201300,
        "inheritance": (
            "AR — homozygous or compound heterozygous in HSN2 exon; "
            "consanguineous pedigrees common; ethnically diverse; "
            "no single founder mutation — multiple private HSN2-exon variants; "
            "must specifically request HSN2 exon sequencing"
        ),
        "gene_class": (
            "WNK1 encodes a serine-threonine kinase whose catalytic domain lacks the conserved Lys "
            "(hence 'With-No-Lysine'). The HSN2 neuronal exon is inserted in intron 8 and is expressed "
            "only in DRG and other sensory neurons. WNK1/HSN2 regulates KCC2 activity through SPAK/OSR1 "
            "phosphorylation — critical for neuronal chloride homeostasis. LOF → aberrant DRG chloride "
            "gradient → loss of normal sensory signal transduction → DRG neuron apoptosis."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("Truncating variant in HSN2 exon homozygous", 0.55),
            ("Frameshift in HSN2 exon compound heterozygous", 0.25),
            ("Splice-site disrupting HSN2 exon inclusion", 0.20),
        ],
        "age_onset_years_range": (0, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "sensory_loss": 1.00,
            "autonomic": 0.60,
            "pain_insensitivity": 0.80,
            "anhidrosis": 0.35,
            "plantar_ulcers": 0.45,
            "mutilations": 0.40,
            "cognitive_decline": 0.05,
            "hearing_loss": 0.05,
            "hyperthermia": 0.35,
            "gi_dysmotility": 0.30,
        },
        "hallmarks": [
            "HSN2-EXON EXCLUSIVE: STANDARD EXOME/SANGER MISSES IT — request targeted HSN2-exon sequencing",
            "Severe pan-modal sensory loss: all modalities (pain, temperature, vibration, touch) severely reduced",
            "Early-onset (infantile): sensory loss detectable in first year of life",
            "Lancinating pains in early disease: before established sensory loss — diagnostic clue",
            "Autonomic features: anhidrosis, BP fluctuations (milder than FD/ELP1)",
            "Mutilations: secondary to pain insensitivity — tongue, digit, joint injuries",
            "NO COGNITIVE DECLINE: WNK1-HSN2 does not affect central neurons",
            "WNK1 canonical mutations → Gordon syndrome (NOT HSAN): entirely different exon",
        ],
        "treatment_alerts": [
            "DIAGNOSTIC ALERT: Must specifically request WNK1-HSN2 exon sequencing; standard panels miss it",
            "Gordon syndrome confusion: canonical WNK1 → hypertension; HSN2 isoform → neuropathy; different exons",
            "Foot care + wound management identical to HSAN1A protocol",
            "Lancinating pain: if present early, gabapentin/pregabalin (paradoxical pain relief)",
            "Autonomic monitoring: temperature regulation; cooling as needed",
            "No disease-modifying therapy available",
        ],
        "organ_system": "peripheral sensory nervous system",
        "primary_treatment": "wound care + injury prevention; MANDATORY HSN2-exon targeted sequencing",
    },

    # ── PRDM12 — HSAN8 ──
    {
        "gene": "PRDM12",
        "protein": "PR Domain Zinc Finger Protein 12",
        "alias": (
            "PRDM12; OMIM gene 616458; HSAN8 #616488 (AR); "
            "9q34.12; 516 aa; ~57 kDa; PR/SET domain zinc-finger transcription factor; "
            "expressed during embryonic DRG development; represses nociceptor fate-determining genes; "
            "PRDM12 LOF → failure of nociceptor specification during embryogenesis → absent nociceptors; "
            "phenotype: global pain insensitivity WITHOUT anhidrosis — sweat glands and autonomic innervation INTACT; "
            "KEY DDx FROM CIPA (NTRK1): sweating preserved → no hyperthermia crisis → better prognosis; "
            "temperature sensing impaired; corneal abrasions a major complication"
        ),
        "aa": "516 aa",
        "kDa": "~57 kDa",
        "locus": "9q34.12",
        "omim_gene": 616458,
        "omim_disease": 616488,
        "inheritance": (
            "AR — homozygous or compound heterozygous; "
            "multiple consanguineous pedigrees globally; "
            "PRDM12 described as novel HSAN gene in 2015 — clinically underrecognised; "
            "no major founder mutation"
        ),
        "gene_class": (
            "PRDM12 is a PR domain (related to SET domain) zinc-finger transcription factor. "
            "It is expressed in DRG progenitor cells during embryonic neurogenesis and is required for "
            "the specification/differentiation of nociceptors (TrkA+/CGRP+/substance P+ neurons). "
            "PRDM12 LOF → nociceptors fail to differentiate from common DRG progenitors → "
            "absent pain-sensing neurons. Autonomic neurons develop independently of PRDM12 → "
            "sweating, BP regulation, GI function INTACT (unlike NTRK1/CIPA)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("Truncating (nonsense/frameshift) homozygous", 0.50),
            ("Missense PR domain homozygous", 0.30),
            ("Compound heterozygous truncating + missense", 0.20),
        ],
        "age_onset_years_range": (0, 3),
        "sex_ratio_M": 0.50,
        "rates": {
            "sensory_loss": 1.00,
            "autonomic": 0.10,
            "pain_insensitivity": 1.00,
            "anhidrosis": 0.05,
            "plantar_ulcers": 0.40,
            "mutilations": 0.25,
            "cognitive_decline": 0.05,
            "hearing_loss": 0.00,
            "hyperthermia": 0.05,
            "gi_dysmotility": 0.05,
        },
        "hallmarks": [
            "PAIN INSENSITIVITY WITHOUT ANHIDROSIS: sweating preserved → hyperthermia NOT a major risk (key DDx from CIPA)",
            "Temperature sensing impaired: cannot detect tissue-damaging heat/cold despite intact sweat response",
            "Corneal abrasions: no corneal pain → undetected injury → recurrent ulcers → risk of vision loss",
            "Congenital — nociceptors never formed (developmental specification failure, not degeneration)",
            "Normal cognitive function: PRDM12 not expressed in CNS cognitive areas",
            "Normal autonomic function: BP, heart rate, GI motility, sweating all intact",
            "NOVEL (2015): underdiagnosed — consider in any child with congenital pain insensitivity + sweating",
            "Life expectancy near-normal: fewer hyperthermia crises than CIPA; morbidity from injuries/ulcers",
        ],
        "treatment_alerts": [
            "CORNEAL PROTECTION: lubricant drops every 2 hours; moisture chamber at night; ophthalmology every 6 months",
            "Injury surveillance: temperature probes on affected areas; protective clothing/gloves",
            "Dental guard: lip and tongue self-injury from biting",
            "Foot care: plantar ulcer surveillance identical to HSAN1A protocol",
            "No disease-modifying therapy: nociceptors absent from development — cannot be regenerated with current therapy",
            "Genetic counselling: AR; sibling risk 25%",
        ],
        "organ_system": "peripheral nociceptive nervous system (autonomic SPARED)",
        "primary_treatment": "corneal protection + injury prevention; no disease-modifying therapy",
    },
]


def _build_cohort(gene_def: dict) -> list:
    """Generate synthetic patient cohort for a gene."""
    rng = random.Random(gene_def["seed"])
    patients = []
    etiologies = gene_def["etiologies"]
    gene = gene_def["gene"]
    ages_years = gene_def.get("age_onset_years_range", (10, 40))
    sex_ratio_m = gene_def.get("sex_ratio_M", 0.50)
    rates = gene_def["rates"]

    for i in range(gene_def["n_patients"]):
        # Pick etiology
        r = rng.random()
        cumulative = 0.0
        etiology = etiologies[-1][0]
        for name, prob in etiologies:
            cumulative += prob
            if r <= cumulative:
                etiology = name
                break

        age_at_onset = rng.randint(ages_years[0], max(ages_years[0] + 1, ages_years[1]))
        age_at_diagnosis = age_at_onset + rng.randint(0, 8)
        sex = "M" if rng.random() < sex_ratio_m else "F"

        # Clinical features
        has_sensory_loss = rng.random() < rates["sensory_loss"]
        has_autonomic_dysfunction = rng.random() < rates["autonomic"]
        has_pain_insensitivity = rng.random() < rates["pain_insensitivity"]
        has_anhidrosis = rng.random() < rates["anhidrosis"]
        has_plantar_ulcers = rng.random() < rates["plantar_ulcers"]
        has_mutilations = rng.random() < rates["mutilations"]
        has_cognitive_decline = rng.random() < rates["cognitive_decline"]
        has_hearing_loss = rng.random() < rates["hearing_loss"]
        has_hyperthermia = rng.random() < rates["hyperthermia"]
        has_gi_dysmotility = rng.random() < rates["gi_dysmotility"]

        # Gene-specific additional features
        has_lancinating_pain = gene in ("SPTLC1", "WNK1") and rng.random() < 0.55
        has_autonomic_crisis = gene == "ELP1" and rng.random() < 0.80
        has_scoliosis = gene == "ELP1" and rng.random() < 0.85
        has_alacrima = gene == "ELP1" and rng.random() < 0.90
        has_charcot_joint = gene == "NTRK1" and rng.random() < 0.45
        has_corneal_abrasion = gene == "PRDM12" and rng.random() < 0.60

        # Treatment
        if gene == "SPTLC1":
            treatment = rng.choice([
                "L-serine supplementation + foot care",
                "L-serine + gabapentin + wound care",
                "Foot care only (pre-L-serine era)",
                "L-serine + protective footwear + orthotics",
            ])
        elif gene == "ELP1":
            treatment = rng.choice([
                "TUDCA + autonomic crisis protocol",
                "TUDCA + scoliosis surgery + G-tube",
                "Autonomic crisis management + G-tube",
                "TUDCA + clonidine + G-tube + scoliosis monitoring",
            ])
        elif gene == "NTRK1":
            treatment = rng.choice([
                "Cooling vest + dental guard + fracture surveillance",
                "Temperature monitoring + orthopaedic support",
                "Protective environment + developmental support",
                "Cooling protocol + corneal lubrication + orthopaedic review",
            ])
        elif gene == "NGFB":
            treatment = rng.choice([
                "Fracture surveillance + injury monitoring",
                "Supportive + orthopaedic support",
                "Deep injury surveillance + wound care",
            ])
        elif gene == "FAM134B":
            treatment = rng.choice([
                "Protective padding + wound care + autonomic monitoring",
                "Dental extraction + mouthguard + IV antibiotics (osteomyelitis)",
                "Protective mittens + G-tube + autonomic protocol",
            ])
        elif gene == "DNMT1":
            treatment = rng.choice([
                "Hearing aids + dementia management + foot care",
                "Cochlear implant (if profound SNHL) + neuropsychiatry",
                "Audiology + cognitive support + L-serine (investigational)",
                "Supportive dementia care + hearing aids + neuropathy monitoring",
            ])
        elif gene == "WNK1":
            treatment = rng.choice([
                "Wound care + injury prevention",
                "Gabapentin (lancinating pain) + foot care",
                "Supportive + autonomic monitoring",
            ])
        elif gene == "PRDM12":
            treatment = rng.choice([
                "Corneal lubrication + injury surveillance",
                "Moisture chamber + dental guard + foot care",
                "Protective clothing + ophthalmology monitoring",
            ])
        else:
            treatment = "supportive"

        # Diagnostic route
        diagnostic_route = rng.choice([
            "Clinical phenotype → HSAN gene panel",
            "Family history → targeted sequencing",
            "Skin punch biopsy (IENFD) + gene panel",
            "Nerve conduction study (absent SNAPs) + gene panel",
            "Sural nerve biopsy → DRG-specific gene panel",
        ])

        patients.append({
            "patient_id": f"{gene}-P{i+1:03d}",
            "gene": gene,
            "etiology": etiology,
            "age_at_onset": age_at_onset,
            "age_at_diagnosis": age_at_diagnosis,
            "sex": sex,
            "has_sensory_loss": has_sensory_loss,
            "has_autonomic_dysfunction": has_autonomic_dysfunction,
            "has_pain_insensitivity": has_pain_insensitivity,
            "has_anhidrosis": has_anhidrosis,
            "has_plantar_ulcers": has_plantar_ulcers,
            "has_mutilations": has_mutilations,
            "has_cognitive_decline": has_cognitive_decline,
            "has_hearing_loss": has_hearing_loss,
            "has_hyperthermia": has_hyperthermia,
            "has_gi_dysmotility": has_gi_dysmotility,
            "has_lancinating_pain": has_lancinating_pain,
            "has_autonomic_crisis": has_autonomic_crisis,
            "has_scoliosis": has_scoliosis,
            "has_alacrima": has_alacrima,
            "has_charcot_joint": has_charcot_joint,
            "has_corneal_abrasion": has_corneal_abrasion,
            "treatment_received": treatment,
            "diagnostic_route": diagnostic_route,
        })
    return patients


def get_overview():
    """Atlas overview: gene list, aggregate stats, key DDx anchors."""
    genes_summary = []
    total_patients = 0
    total_sensory = 0
    total_autonomic = 0
    total_pain_insensitivity = 0
    total_anhidrosis = 0
    total_plantar_ulcers = 0
    total_mutilations = 0
    total_cognitive = 0
    total_hearing = 0
    total_hyperthermia = 0
    total_gi = 0

    for gd in HSAN_GENES:
        cohort = _build_cohort(gd)
        n = len(cohort)
        total_patients += n

        sensory = sum(1 for p in cohort if p["has_sensory_loss"])
        autonomic = sum(1 for p in cohort if p["has_autonomic_dysfunction"])
        pain_ins = sum(1 for p in cohort if p["has_pain_insensitivity"])
        anhid = sum(1 for p in cohort if p["has_anhidrosis"])
        ulcers = sum(1 for p in cohort if p["has_plantar_ulcers"])
        mutilat = sum(1 for p in cohort if p["has_mutilations"])
        cognitive = sum(1 for p in cohort if p["has_cognitive_decline"])
        hearing = sum(1 for p in cohort if p["has_hearing_loss"])
        hyper = sum(1 for p in cohort if p["has_hyperthermia"])
        gi = sum(1 for p in cohort if p["has_gi_dysmotility"])

        total_sensory += sensory
        total_autonomic += autonomic
        total_pain_insensitivity += pain_ins
        total_anhidrosis += anhid
        total_plantar_ulcers += ulcers
        total_mutilations += mutilat
        total_cognitive += cognitive
        total_hearing += hearing
        total_hyperthermia += hyper
        total_gi += gi

        avg_onset = round(sum(p["age_at_onset"] for p in cohort) / n, 1)
        avg_diag_delay = round(
            sum(p["age_at_diagnosis"] - p["age_at_onset"] for p in cohort) / n, 1
        )

        genes_summary.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "n_patients": n,
            "sensory_loss_pct": round(100 * sensory / n, 1),
            "autonomic_pct": round(100 * autonomic / n, 1),
            "pain_insensitivity_pct": round(100 * pain_ins / n, 1),
            "anhidrosis_pct": round(100 * anhid / n, 1),
            "plantar_ulcers_pct": round(100 * ulcers / n, 1),
            "mutilations_pct": round(100 * mutilat / n, 1),
            "cognitive_decline_pct": round(100 * cognitive / n, 1),
            "hearing_loss_pct": round(100 * hearing / n, 1),
            "hyperthermia_pct": round(100 * hyper / n, 1),
            "gi_dysmotility_pct": round(100 * gi / n, 1),
            "avg_age_at_onset": avg_onset,
            "avg_diagnosis_delay_years": avg_diag_delay,
            "primary_organ_system": gd["organ_system"],
            "primary_treatment": gd["primary_treatment"],
            "hallmarks": gd["hallmarks"][:4],
            "top_treatment_alert": gd["treatment_alerts"][0],
        })

    return {
        "atlas": "HSAN-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Sensory and Autonomic Neuropathy Atlas",
        "api_path": "/api/hsan-atlas/",
        "genes": [g["gene"] for g in HSAN_GENES],
        "total_patients": total_patients,
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "aggregate_stats": {
            "sensory_loss_pct": round(100 * total_sensory / total_patients, 1),
            "autonomic_dysfunction_pct": round(100 * total_autonomic / total_patients, 1),
            "pain_insensitivity_pct": round(100 * total_pain_insensitivity / total_patients, 1),
            "anhidrosis_pct": round(100 * total_anhidrosis / total_patients, 1),
            "plantar_ulcers_pct": round(100 * total_plantar_ulcers / total_patients, 1),
            "mutilations_pct": round(100 * total_mutilations / total_patients, 1),
            "cognitive_decline_pct": round(100 * total_cognitive / total_patients, 1),
            "hearing_loss_pct": round(100 * total_hearing / total_patients, 1),
            "hyperthermia_pct": round(100 * total_hyperthermia / total_patients, 1),
            "gi_dysmotility_pct": round(100 * total_gi / total_patients, 1),
        },
        "genes_summary": genes_summary,
        "key_ddx_anchor": [
            "CIPA (NTRK1) vs HSAN8 (PRDM12): BOTH have global pain insensitivity — key DDx is ANHIDROSIS (NTRK1) vs SWEATING PRESERVED (PRDM12)",
            "SPTLC1-HSAN1A: deoxy-sphingolipid toxicity — VINCRISTINE/CISPLATIN ABSOLUTELY CONTRAINDICATED (acute axonal collapse)",
            "ELP1-FD: ASHKENAZI JEWISH EXCLUSIVELY — c.2204+6T>C >99.5% alleles; autonomous crises; alacrima; absent tongue fungiform papillae",
            "DNMT1-HSAN1E TRIAD: sensory neuropathy + SNHL + frontotemporal dementia — unique among all HSAN genes",
            "WNK1-HSAN2A DIAGNOSTIC ALERT: HSN2 exon NOT in standard panels — standard exome/Sanger MISSES IT; request targeted HSN2-exon sequencing",
            "NTRK1-CIPA HYPERTHERMIA: anhidrosis + warm environment → fatal hyperthermia; cooling vest + temperature monitoring mandatory",
            "FAM134B-HSAN2B: MOST SEVERE HSAN2; neonatal-onset self-mutilation; ER-reticulophagy failure → DRG apoptosis",
            "L-SERINE SUPPLEMENTATION (SPTLC1): reduces 1-deoxysphingolipid synthesis → partial neuroprotection — ONLY disease-modifying therapy in all 8 genes",
        ],
    }


def get_breakdown():
    """Per-gene detailed breakdown with cohort data."""
    result = []
    for gd in HSAN_GENES:
        cohort = _build_cohort(gd)
        n = len(cohort)

        sexes = {"M": sum(1 for p in cohort if p["sex"] == "M"),
                 "F": sum(1 for p in cohort if p["sex"] == "F")}
        etiology_counts = {}
        treatments = {}
        diagnostic_routes = {}

        for p in cohort:
            etiology_counts[p["etiology"]] = etiology_counts.get(p["etiology"], 0) + 1
            treatments[p["treatment_received"]] = treatments.get(p["treatment_received"], 0) + 1
            diagnostic_routes[p["diagnostic_route"]] = diagnostic_routes.get(p["diagnostic_route"], 0) + 1

        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"].split(";")[0].strip(),
            "n_patients": n,
            "sex_distribution": sexes,
            "avg_age_at_onset": round(sum(p["age_at_onset"] for p in cohort) / n, 1),
            "avg_diagnosis_delay_years": round(
                sum(p["age_at_diagnosis"] - p["age_at_onset"] for p in cohort) / n, 1
            ),
            "sensory_loss_pct": round(100 * sum(1 for p in cohort if p["has_sensory_loss"]) / n, 1),
            "autonomic_pct": round(100 * sum(1 for p in cohort if p["has_autonomic_dysfunction"]) / n, 1),
            "pain_insensitivity_pct": round(100 * sum(1 for p in cohort if p["has_pain_insensitivity"]) / n, 1),
            "anhidrosis_pct": round(100 * sum(1 for p in cohort if p["has_anhidrosis"]) / n, 1),
            "plantar_ulcers_pct": round(100 * sum(1 for p in cohort if p["has_plantar_ulcers"]) / n, 1),
            "mutilations_pct": round(100 * sum(1 for p in cohort if p["has_mutilations"]) / n, 1),
            "cognitive_decline_pct": round(100 * sum(1 for p in cohort if p["has_cognitive_decline"]) / n, 1),
            "hearing_loss_pct": round(100 * sum(1 for p in cohort if p["has_hearing_loss"]) / n, 1),
            "hyperthermia_pct": round(100 * sum(1 for p in cohort if p["has_hyperthermia"]) / n, 1),
            "gi_dysmotility_pct": round(100 * sum(1 for p in cohort if p["has_gi_dysmotility"]) / n, 1),
            "etiology_distribution": etiology_counts,
            "treatment_distribution": treatments,
            "diagnostic_route_distribution": diagnostic_routes,
            "hallmarks": gd["hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "primary_treatment": gd["primary_treatment"],
            "organ_system": gd["organ_system"],
        })
    return result


def get_definitions():
    """Key clinical definitions for HSAN conditions."""
    return {
        "definitions": [
            {
                "term": "Deoxysphingolipid Toxicity (SPTLC1) — L-serine as Substrate Competitor",
                "definition": (
                    "SPTLC1 encodes the catalytic subunit of serine palmitoyltransferase (SPT), which normally "
                    "condenses L-serine + palmitoyl-CoA → 3-ketodihydrosphingosine. "
                    "Pathogenic SPTLC1 variants (Cys133Trp, Val144Asp) shift substrate selectivity: SPT accepts "
                    "L-alanine or L-glycine instead of L-serine → generates 1-deoxysphingolipids (1-deoxySL). "
                    "1-DEOXYSPHINGOLIPIDS ARE NEUROTOXIC: they cannot enter the canonical ceramide/sphingomyelin "
                    "catabolic pathway (no C1-OH group for headgroup exchange) → accumulate in DRG neurons → "
                    "mitochondrial dysfunction → axonal degeneration. "
                    "L-SERINE THERAPEUTIC RATIONALE: Oral L-serine (400 mg/kg/day) floods the SPT active site "
                    "with the correct substrate, competing with alanine → reduces 1-deoxySL synthesis by up to 50% "
                    "in phase II trials. Plasma 1-deoxySL levels are the pharmacodynamic biomarker. "
                    "VINCRISTINE/CISPLATIN ABSOLUTELY CONTRAINDICATED: these agents are independently neurotoxic "
                    "and in SPTLC1 carriers precipitate acute axonal collapse — even a single vincristine dose "
                    "can cause fulminant neuropathy. Cancer teams must be warned before any chemotherapy."
                ),
            },
            {
                "term": "Familial Dysautonomia (ELP1/IKBKAP) — Ashkenazi Founder and Autonomic Crisis",
                "definition": (
                    "Familial Dysautonomia (FD) is the prototypic hereditary autonomic neuropathy, affecting "
                    "exclusively Ashkenazi Jewish individuals. The c.2204+6T>C splice-site variant in intron 20 "
                    "of ELP1 causes exon 20 skipping → truncated, unstable mRNA → severely reduced ELP1 protein. "
                    "ELP1 is the scaffold of the 6-subunit Elongator complex, which modifies wobble-base uridine "
                    "(U34) in tRNA anticodons → promotes accurate decoding of AA-codon mRNAs; "
                    "without ELP1, translation of many proteins is impaired → particularly catastrophic in neurons. "
                    "AUTONOMIC CRISIS: Episodic hypertension (systolic >180 mmHg) + cyclic vomiting + diaphoresis; "
                    "triggered by excitement, illness, or anaesthesia. Management: IV diazepam (first-line sedation), "
                    "ondansetron (anti-emetic), clonidine (blood pressure), IV fluids. "
                    "TUDCA (tauroursodeoxycholic acid): reduces ER stress + partially corrects mis-splicing → "
                    "restores some ELP1 protein; currently the only disease-modifying treatment in clinical use. "
                    "ALACRIMA (absent tears): Schirmer test <5 mm in virtually all FD patients; corneal lubrication mandatory."
                ),
            },
            {
                "term": "CIPA (NTRK1) — Hyperthermia as Primary Lethal Mechanism",
                "definition": (
                    "Congenital Insensitivity to Pain with Anhidrosis (CIPA, HSAN4) is caused by biallelic NTRK1 LOF. "
                    "TrkA (NTRK1) is the high-affinity NGF receptor: NGF-TrkA retrograde signalling from peripheral "
                    "target tissues to DRG cell bodies is mandatory for survival of nociceptors and sympathetic neurons "
                    "during fetal development. NTRK1 LOF → no NGF signal → nociceptors and sympathetic postganglionic "
                    "neurons undergo apoptosis in utero. "
                    "HYPERTHERMIA IS THE LEADING CAUSE OF DEATH IN CIPA: "
                    "Absent sympathetic innervation to sweat glands → no sweating → no evaporative cooling. "
                    "Core temperature can exceed 41°C within 30-60 minutes in a warm environment → brain damage → death. "
                    "Mandatory precautions: cooling vest (worn whenever ambient temperature >22°C), misting fan, "
                    "rectal temperature monitoring, NEVER leaving child unattended outdoors. "
                    "DENTAL/ORTHOPAEDIC: Undetected oral trauma → fractured teeth, tongue ulcers; "
                    "undetected fractures → Charcot joints. Dental guard from 6 months, weekly orthopaedic check."
                ),
            },
            {
                "term": "HSAN5 (NGFB) — Selective Nociceptor Loss with Preserved Mechanoreception",
                "definition": (
                    "NGFB encodes the beta subunit of NGF, the obligate TrkA ligand. Unlike NTRK1 LOF (which causes "
                    "complete absence of nociceptors AND sympathetics), NGFB pathogenic variants (p.Arg221Trp founder) "
                    "produce a partially functional NGF — reduced secretion and TrkA binding affinity. "
                    "SELECTIVE PHENOTYPE: Only high-threshold nociceptors (A-delta mechanical nociceptors and C-fibres, "
                    "most NGF-dependent) are lost. Low-threshold mechanoreceptors (Merkel, Meissner, Ruffini, Pacinian) "
                    "survive on alternative neurotrophins (NT-3, BDNF). "
                    "CLINICAL RESULT: Deep pain insensitivity (cannot feel bone fractures, muscle tears, visceral pain) "
                    "with PRESERVED light touch, vibration, and proprioception. "
                    "KEY DDx: In HSAN4 (NTRK1) ALL sensation is lost; in HSAN5 (NGFB) superficial sensation is intact. "
                    "Sweating mostly preserved (sympathetic neurons receive residual NGF). "
                    "Norwegian founder p.Arg221Trp — identified in a single large consanguineous pedigree."
                ),
            },
            {
                "term": "FAM134B/RETREG1 (HSAN2B) — ER-Reticulophagy and DRG Neuron Death",
                "definition": (
                    "FAM134B (RETREG1) encodes an ER-resident receptor that mediates reticulophagy: "
                    "selective autophagic degradation of excess or damaged ER. "
                    "FAM134B contains: (1) a reticulon homology domain (RHD) that curves ER tubules; "
                    "(2) two LIR (LC3-interacting region) motifs that recruit autophagosomes. "
                    "FAM134B LOF → reticulophagy fails → ER tubule network expands → misfolded proteins accumulate → "
                    "chronic ER stress (UPR activation: IRE1, PERK, ATF6 arms) → DRG neuron apoptosis. "
                    "DRG neurons are particularly vulnerable because they are post-mitotic and cannot replace lost ER capacity. "
                    "PHENOTYPE: Most severe HSAN2 subtype — neonatal onset, pan-modal sensory loss (pain, touch, "
                    "vibration, proprioception all absent), profound autonomic instability, and self-mutilating "
                    "behaviour from the first months of life. "
                    "MANAGEMENT: Physical restraint (padded gloves, elbow pads), dental care, autonomic monitoring, "
                    "GI support. No disease-modifying therapy available."
                ),
            },
            {
                "term": "DNMT1-HSAN1E — Epigenetic Neuropathy: Sensory + Hearing + Dementia Triad",
                "definition": (
                    "DNMT1 (DNA methyltransferase 1) maintains CpG methylation during DNA replication. "
                    "The replication-foci targeting sequence (RFTS domain) autoinhibits catalytic activity until "
                    "DNMT1 is recruited to hemimethylated CpG at the replication fork. "
                    "RFTS domain mutations (Tyr495Cys, Ala570Val, Val606Phe) cause HSAN1E by a dominant negative / "
                    "gain-of-function mechanism: mutant RFTS mis-folds → ubiquitin-proteasome degradation → "
                    "reduced DNMT1 → CpG hypomethylation → inappropriate gene activation in post-mitotic neurons. "
                    "THE TRIAD (pathognomonic): "
                    "(1) Adult-onset length-dependent sensory neuropathy (feet first, similar to HSAN1A); "
                    "(2) Sensorineural hearing loss — often the FIRST symptom (audiogram before neuropathy symptoms); "
                    "(3) Frontotemporal dementia — executive dysfunction, personality change, later aphasia. "
                    "NO OTHER HSAN GENE CAUSES ALL THREE. Patients presenting with unexplained sensory neuropathy + SNHL + "
                    "cognitive decline in the 3rd-5th decade should have DNMT1 RFTS domain sequencing. "
                    "No disease-modifying therapy; epigenetic modulation investigational."
                ),
            },
            {
                "term": "WNK1-HSAN2A — HSN2 Isoform: Why Standard Testing Misses It",
                "definition": (
                    "WNK1 (With-No-Lysine kinase 1) is a serine-threonine kinase famous for its role in "
                    "renal salt handling: canonical WNK1 mutations → Gordon syndrome (pseudohypoaldosteronism type IIA, "
                    "hypertension, hyperkalaemia). These are entirely different from HSAN2A mutations. "
                    "HSAN2A IS CAUSED EXCLUSIVELY BY MUTATIONS IN THE HSN2 ISOFORM-SPECIFIC EXON: "
                    "The HSN2 exon is a neuronal-specific exon inserted in intron 8 of the canonical WNK1 transcript. "
                    "It is expressed only in DRG, spinal cord, and brain. "
                    "THE HSN2 EXON IS NOT IN STANDARD REFSEQ TRANSCRIPTS used by most clinical exome pipelines. "
                    "Consequence: Sanger sequencing of exons 1-28 of WNK1 WILL NOT DETECT HSAN2A. "
                    "ACTION: When suspecting WNK1-HSAN2A, specifically request HSN2-exon sequencing "
                    "(deep-intronic or isoform-aware panel). Failure to do so results in false-negative in essentially "
                    "all standard panels. Gordon syndrome (canonical WNK1 mutations) does NOT cause neuropathy."
                ),
            },
            {
                "term": "PRDM12-HSAN8 — Congenital Pain Insensitivity with Sweating Preserved",
                "definition": (
                    "PRDM12 encodes a PR domain / zinc-finger transcription factor expressed in DRG progenitors "
                    "during embryonic neurogenesis (E10.5-E12.5 in mice; analogous human window). "
                    "PRDM12 is required for the specification/fate commitment of nociceptive neurons from common "
                    "DRG progenitors. PRDM12 LOF → nociceptors fail to differentiate → absent from birth. "
                    "CRITICAL DDx FROM CIPA (NTRK1): "
                    "NTRK1/CIPA: pain insensitivity + ANHIDROSIS (absent sweating) — hyperthermia is lethal. "
                    "PRDM12/HSAN8: pain insensitivity + SWEATING PRESERVED — hyperthermia is NOT a major risk. "
                    "The practical implication: HSAN8 patients do not require cooling vests or strict temperature "
                    "precautions that are mandatory in CIPA. "
                    "However: temperature SENSING is impaired (cannot feel burning heat) — tissue damage from "
                    "hot surfaces/liquids occurs silently. "
                    "CORNEAL ABRASION: With no corneal pain, eye injury goes undetected → recurrent ulcers → "
                    "corneal scarring/vision loss. Lubricant drops every 2 hours and ophthalmology every 6 months are mandatory."
                ),
            },
            {
                "term": "IENFD (Intraepidermal Nerve Fibre Density) — Diagnostic Biopsy in HSAN",
                "definition": (
                    "Intraepidermal nerve fibre density (IENFD) measurement by skin punch biopsy is the key "
                    "histological confirmation of small-fibre neuropathy in HSAN. "
                    "PROCEDURE: 3 mm punch biopsy of distal leg (10 cm above lateral malleolus) + proximal site. "
                    "Immunostaining with anti-PGP 9.5 (panaxonal marker) → count nerve fibre profiles per mm "
                    "of epidermis; reference range age/sex-adjusted. "
                    "IN HSAN: IENFD is severely reduced or absent at distal site, often sparing proximal site early; "
                    "confirms small-fibre pathology even when NCS is normal (NCS tests large fibres). "
                    "NCS IN HSAN: Absent sensory nerve action potentials (SNAPs) in large-fibre HSAN types; "
                    "NCS can be NORMAL in pure small-fibre types (SPTLC1, NGFB, PRDM12 early). "
                    "DIFFERENTIAL ROLE: Biopsy distinguishes HSAN (small fibre lost from birth/early life) "
                    "from acquired small-fibre neuropathy (diabetes, Sjögren, idiopathic) — history + IENFD + genetic panel."
                ),
            },
            {
                "term": "Cascade Genetic Testing — HSAN Panel Strategy",
                "definition": (
                    "For any index case with suspected HSAN, the diagnostic and family testing approach: "
                    "Step 1 — EXCLUDE COMMON ACQUIRED CAUSES: glucose/HbA1c (diabetes), Sjögren (anti-Ro, "
                    "lip biopsy), HIV, amyloid, toxic (B6 toxicity, chemotherapy). "
                    "Step 2 — NERVE CONDUCTION STUDY: Absent SNAPs confirm large-fibre sensory loss; "
                    "normal NCS does not exclude HSAN (pure small-fibre types). "
                    "Step 3 — SKIN PUNCH BIOPSY (IENFD): Confirms small-fibre pathology. "
                    "Step 4 — HSAN GENE PANEL: Must include SPTLC1, SPTLC2, ELP1, NTRK1, NGFB, FAM134B, "
                    "DNMT1, WNK1 (including HSN2 exon), PRDM12, KIF1A, ATL1/3, RETREG1. "
                    "WNK1-HSN2: Alert the laboratory to include the HSN2-specific exon. "
                    "Step 5 — CASCADE TESTING: First-degree relatives of confirmed cases (AD types: SPTLC1, DNMT1); "
                    "sibling testing for AR types (25% risk). "
                    "PRESYMPTOMATIC UTILITY: SPTLC1 — L-serine supplementation can be started before clinical neuropathy; "
                    "ELP1 — TUDCA initiated early may slow disease progression."
                ),
            },
            {
                "term": "L-serine Supplementation (SPTLC1-HSAN1A) — Only Disease-Modifying Therapy",
                "definition": (
                    "L-serine is the only disease-modifying therapy for any HSAN gene (as of 2024 evidence). "
                    "MECHANISM: Oral L-serine floods the SPT enzyme active site, competing with the misincorporated "
                    "L-alanine substrate. By mass action, more of the SPT condensation products are canonical "
                    "sphingolipids vs 1-deoxysphingolipids → plasma and DRG 1-deoxySL levels fall. "
                    "EVIDENCE: Phase II RCT (n=22, SPTLC1/SPTLC2, Fridman et al. 2019): "
                    "L-serine 400 mg/kg/day reduced plasma 1-deoxySL by 47%; neuropathy stabilisation trend. "
                    "BIOMARKER: Plasma 1-deoxysphingolipid levels (1-deoxySA + 1-deoxySO ratio); "
                    "target reduction >30% from baseline to assess pharmacological response. "
                    "SAFETY: Well tolerated; no serious adverse events in trials; aminotransferase monitoring. "
                    "START EARLY: Initiate at genetic diagnosis BEFORE clinical neuropathy onset — "
                    "axonal loss is irreversible; pre-symptomatic neuroprotection is the clinical goal. "
                    "NOT EFFECTIVE IN OTHER HSAN GENES: Mechanism is SPTLC1/SPTLC2-specific."
                ),
            },
            {
                "term": "Hereditary Sensory Neuropathy Classification (HSAN1-8)",
                "definition": (
                    "HSAN is classified by subtype (HSAN1-8), inheritance, and gene: "
                    "HSAN1 (AD): SPTLC1 (1A), SPTLC2 (1C), ATL1 (1D), DNMT1 (1E), ATL3 (1F); "
                    "onset 2nd-5th decade; length-dependent; sensory > motor; plantar ulcers. "
                    "HSAN2 (AR): WNK1/HSN2-exon (2A), FAM134B (2B), KIF1A (2C); "
                    "infantile onset; severe pan-modal loss; mutilations. "
                    "HSAN3 (AR): ELP1/IKBKAP; Ashkenazi Jewish only; Familial Dysautonomia; autonomic + sensory. "
                    "HSAN4 (AR): NTRK1; CIPA; pain + anhidrosis; hyperthermia. "
                    "HSAN5 (AR): NGFB; selective deep-pain loss; sweating partial. "
                    "HSAN7 (AD): SCN11A; episodic pain + GI dysmotility (gain-of-function Nav1.9). "
                    "HSAN8 (AR): PRDM12; pain insensitivity; sweating preserved. "
                    "Common misdiagnosis: idiopathic small-fibre neuropathy. "
                    "Key DDx by anhidrosis: HSAN4 (NTRK1) = anhidrotic; HSAN5, HSAN8 = sweating preserved. "
                    "Key DDx by onset: HSAN1 = adult; HSAN2/3/4/5/8 = congenital/infantile."
                ),
            },
        ],
        "pharmacological_distinctions": [
            "L-SERINE (SPTLC1/HSAN1A ONLY): Substrate competitor — oral L-serine competes with L-alanine at "
            "mutant SPT active site → reduces 1-deoxysphingolipid synthesis; 400 mg/kg/day; "
            "ONLY disease-modifying therapy across all 8 HSAN genes (Phase II evidence); no effect in other HSAN types",
            "TUDCA / TAUROURSODEOXYCHOLIC ACID (ELP1/FD ONLY): bile acid derivative that reduces ER stress + "
            "partially corrects ELP1 exon 20 mis-splicing → restores ELP1 protein; 15 mg/kg/day; "
            "in clinical use at FD specialised centres; slows disease progression",
            "BENZODIAZEPAM + ONDANSETRON (ELP1 AUTONOMIC CRISIS): IV diazepam 0.1 mg/kg (sedation) + "
            "ondansetron 0.15 mg/kg (antiemetic) + clonidine 0.05-0.1 mg (blood pressure control); "
            "mainstay of acute crisis management in FD; must be prescribed as a standing PRN crisis kit",
            "GABAPENTIN / PREGABALIN (SPTLC1, WNK1 lancinating pain): alpha-2-delta calcium channel ligands; "
            "effective for lancinating neuropathic pain in early HSAN1A/2A before established sensory loss; "
            "dose-limit: excessive sedation in already-ataxic patients",
            "COOLING VEST / EXTERNAL COOLING (NTRK1/CIPA): Not pharmacological but the critical life-saving "
            "intervention — evaporative cooling vest, misting fan, air-conditioned environment; "
            "mandatory whenever ambient temperature >22°C; rectal thermometry for febrile illness",
            "CORNEAL LUBRICANTS (PRDM12, NTRK1, ELP1): preservative-free artificial tears every 2 hours; "
            "viscous gel at night; moisture chamber spectacles overnight; fundamental for preventing "
            "corneal scarring from painless abrasion in all HSAN types with corneal anaesthesia",
            "CLONIDINE (ELP1 SUPINE HYPERTENSION): alpha-2 agonist; reduces sympathetic tone during "
            "autonomic crises; 0.1-0.3 mg orally PRN; also used as transdermal patch for overnight supine "
            "hypertension in FD; does not prevent orthostatic hypotension",
            "VINCRISTINE/CISPLATIN — ABSOLUTELY CONTRAINDICATED IN SPTLC1: These chemotherapy agents are "
            "independently neurotoxic and in SPTLC1 carriers trigger catastrophic acute axonal degeneration "
            "even at standard doses. Oncology teams MUST be informed of SPTLC1 status before any cancer treatment. "
            "Alternative chemotherapy regimens must be planned in consultation with neurology.",
        ],
    }
