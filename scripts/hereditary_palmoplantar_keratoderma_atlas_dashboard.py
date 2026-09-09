#!/usr/bin/env python3
"""Hereditary-Palmoplantar-Keratoderma-Atlas — Complete 8-Gene Hereditary PPK Atlas
(KRT9 · SLURP1 · CTSC · GJB2 · DSP · JUP · SERPINB7 · LORICRIN).

KRT9     (Keratin 9; 464 aa; 17q21.2; AD;
          Epidermolytic PPK / Vörner disease;
          Most common hereditary PPK;
          BIOPSY: suprabasal vacuolation + epidermolysis PALMS ONLY PATHOGNOMONIC;
          FEET SPARED — KRT10 on soles, not KRT9 — KEY DDx: palm-only involvement;
          p.Arg163Trp most common European mutation;
          Emollients + keratolytics; retinoids for severe;
          seed SEED_BASE+0).
SLURP1   (Secreted Ly6/uPAR-Related Protein 1; 103 aa; 8q24.13; AR;
          Mal de Meleda disease;
          TRANSGREDIENT PPK extending to dorsal hands/feet + erythematous border PATHOGNOMONIC;
          Hyperhidrosis (severe) + pseudoainhum + perioral erythema;
          Mljet Island (Croatia/Adriatic) founder population;
          p.W15R most common Adriatic founder;
          Emollients + retinoids; pseudoainhum release surgery if needed;
          seed SEED_BASE+1).
CTSC     (Cathepsin C; 463 aa; 11q14.2; AR;
          Papillon-Lefèvre syndrome (PLS);
          PPK + SEVERE EARLY-ONSET PERIODONTITIS WITH PREMATURE TOOTH LOSS PATHOGNOMONIC;
          Prophylactic antibiotics (amoxicillin/metronidazole) mandatory to preserve teeth;
          Haim-Munk syndrome variant: PLS + arachnodactyly + acro-osteolysis (CTSC same gene);
          Cathepsin C deficiency impairs neutrophil serine protease activation;
          seed SEED_BASE+2).
GJB2     (Connexin 26; 226 aa; 13q12.11; AD;
          Vohwinkel syndrome (mutilating PPK);
          HONEYCOMB PPK + STARFISH-SHAPED KNUCKLE PADS + PSEUDOAINHUM PATHOGNOMONIC;
          Sensorineural hearing loss (SNHL) in classic Vohwinkel;
          KID syndrome (GJB2 p.Asp50Asn): keratitis-ichthyosis-deafness;
          Connexin-26 gap junction; different mutations → different phenotypes;
          seed SEED_BASE+3).
DSP      (Desmoplakin; 2871 aa; 6p24.3; AR;
          Carvajal syndrome;
          PPK + WOOLLY HAIR + DILATED CARDIOMYOPATHY (DCM) PATHOGNOMONIC;
          Cardiac MRI + echocardiography mandatory — DCM causes sudden cardiac death;
          vs JUP (Naxos): ARVC not DCM — LEFT vs RIGHT ventricle KEY DDx;
          Desmoplakin (desmosomal protein); ICD if significant LV dysfunction;
          seed SEED_BASE+4).
JUP      (Plakoglobin; 745 aa; 17q21.2; AR;
          Naxos disease;
          PPK + WOOLLY HAIR + ARVC PATHOGNOMONIC;
          RIGHT ventricular involvement + ventricular arrhythmias + sudden cardiac death;
          vs DSP (Carvajal): DCM LEFT not RIGHT — RIGHT ARVC is KEY DDx;
          Greek (Naxos island) founder; plakoglobin desmosomal protein;
          ICD mandatory; sports restriction absolute;
          seed SEED_BASE+5).
SERPINB7 (Serine Protease Inhibitor B7; 394 aa; 18q21.33; AR;
          Bothnian type PPK / Non-epidermolytic PPK type 2 (NEPPK2);
          AQUAGENIC WRINKLING OF PALMS PATHOGNOMONIC;
          Palmar pits + punctate hyperkeratosis; mild disease — no systemic involvement;
          Swedish (Bothnian region) founder: p.Trp249Ter;
          No approved treatment; emollients + antiperspirants;
          seed SEED_BASE+6).
LORICRIN (Loricrin; 312 aa; 1q21.3; AD;
          Loricrin keratoderma / Vohwinkel variant without hearing loss;
          PPK + CONSTRICTION BANDS (pseudoainhum) + ICHTHYOTIC VEIL PATHOGNOMONIC;
          NO HEARING LOSS — KEY DDx from GJB2-Vohwinkel (Vohwinkel has SNHL);
          Loricrin is major cornified envelope protein (70-80% of CE);
          Retinoids partially effective; surgery for constriction bands;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2286-2293).
"""

import random

SEED_BASE = 2286

PPK_GENES = [
    # -- KRT9 — Epidermolytic PPK / Vörner disease (most common hereditary PPK) ---------------
    {
        "gene": "KRT9",
        "alt_name": (
            "KRT9 (KRT9-464aa-17q21.2 / AD — Epidermolytic-PPK-Vörner-Disease — "
            "MOST-COMMON-HEREDITARY-PPK — "
            "PALM-ONLY-INVOLVEMENT-FEET-SPARED-PATHOGNOMONIC — "
            "BIOPSY-SUPRABASAL-VACUOLATION-EPIDERMOLYSIS-CONFIRMS — "
            "p.Arg163Trp-Most-Common-European)"
        ),
        "protein": (
            "KRT9 -- 17q21.2 AD -- KRT9-464aa -- "
            "Keratin-9-56kDa-Type-I-Suprabasal-Palmar-Keratinocyte-Intermediate-Filament -- "
            "EPPK-Epidermolytic-PPK-OMIM-144200 -- "
            "EXPRESSED-PALMAR-EPIDERMIS-NOT-SOLES-Partners-KRT1-Suprabasal -- "
            "KRT9-LOF-Dominant-Negative-Suprabasal-IF-Collapse-Epidermolysis -- "
            "BIOPSY-SUPRABASAL-VACUOLATION-PERINUCLEAR-HALOS-EPIDERMOLYSIS-PATHOGNOMONIC -- "
            "PALM-ONLY-FEET-SPARED-KRT10-Covers-Soles-NOT-KRT9-KEY-DDx -- "
            "p.Arg163Trp-Most-Common-European-Mutation-Rod-Domain -- "
            "p.Arg162Trp-p.Ile162Asn-Multiple-Hotspot-Arg-Residues-1B-Domain -- "
            "EMOLLIENT-KERATOLYTIC-First-Line-RETINOID-Severe -- "
            "OMIM-Gene-KRT9-607606-Disease-EPPK-144200"
        ),
        "locus": "17q21.2",
        "protein_size": "464 aa / 56 kDa",
        "inheritance": (
            "AD (autosomal dominant); dominant-negative missense mutations in rod domain; "
            "p.Arg163Trp: most common European mutation (1B domain helix-initiation motif); "
            "p.Arg162Trp, p.Ile162Asn: other hotspot residues; "
            "Nearly 100% penetrance; variable expressivity (severity within families); "
            "De novo mutations occur in ~20% of new cases; "
            "Palm-only disease: KRT9 expressed in palmar epidermis, not plantar; "
            "KRT10 covers soles — feet spared in KRT9 (unlike KRT1-EPPK where PPK is palmoplantar); "
            "Neonatal onset: thickening begins in infancy, fully expressed by adulthood; "
            "Most common hereditary PPK worldwide"
        ),
        "key_features": [
            "PALM-ONLY PPK — FEET COMPLETELY SPARED — PATHOGNOMONIC for KRT9-EPPK; KRT9 expressed in palms (not soles) — unique anatomical restriction distinguishes from all other PPK forms",
            "EPIDERMOLYTIC HYPERKERATOSIS ON BIOPSY — suprabasal vacuolation + perinuclear halos + epidermolysis — PATHOGNOMONIC; confirms KRT9-EPPK without gene sequencing",
            "DIFFUSE PALMAR THICKENING — yellowish-white keratoderma; erythematous border at transition zone; not transgredient; surface may show fine fissuring",
            "p.Arg163Trp — most common European mutation; hotspot in 1B rod domain helix-initiation motif; dominant-negative collapse of suprabasal keratin IF network",
            "ONSET IN INFANCY — diffuse palmar thickening by age 1-2yr; fully expressed by childhood; lifelong; not progressive beyond adulthood",
            "SECONDARY HYPERHIDROSIS — palmar sweating worsens maceration; contributes to malodor and infection risk; antiperspirant adjunct helpful",
            "EMOLLIENTS + KERATOLYTICS (first-line): urea 20-40% palmar preparations; lactic acid 10-12%; soaking + mechanical debridement",
            "RETINOIDS (acitretin) for severe: reduces hyperkeratosis; teratogenic; monitor LFTs; dose 0.3-0.5 mg/kg/day",
        ],
        "treatment": (
            "First-line (all patients): "
            "Soaking + keratolytic: warm water soaks 15-20 min twice daily; mechanical debridement (pumice, Dremel); "
            "Urea 20-40% cream (keratolytic): applied twice daily after soaking; "
            "Lactic acid 10-12% lotion: alpha-hydroxy acid barrier keratolytic; "
            "Salicylic acid 5-10% ointment: adjunct keratolytic for thickened plaques; "
            "Petroleum jelly base for dry fissures. "
            "Antiperspirant: 20% aluminium chloride (Drysol) — reduces hyperhidrosis, maceration risk, malodor. "
            "Antiseptic: chlorhexidine wash — reduces secondary bacterial colonization. "
            "Systemic (severe/functionally impairing): "
            "Acitretin 0.3-0.5 mg/kg/day — reduces scale significantly; functional improvement; teratogenic; "
            "LFTs + lipids monthly (first 3 months), then 3-monthly; DEXA if long-term; "
            "Contraception mandatory for females during + 3yr after acitretin. "
            "Footwear: special gloves for occupation (hand PPK); padded gloves for grip. "
            "Genetics: AD — 50% recurrence per pregnancy; prenatal/PGD offered."
        ),
        "monitoring": [
            "Annual dermatology: keratoderma severity score; fissure infection; functional impact (grip, writing, occupation)",
            "Antiperspirant compliance: hyperhidrosis worsens maceration; monthly review first year",
            "Retinoid monitoring: LFTs + triglycerides 1-2 monthly (first 3 months), then 3-monthly; DEXA if long-term",
            "Infection screen: secondary S. aureus/candida colonization; swab if malodor or fissure inflammation",
            "QoL: DLQI; hand function score; occupational impact; psychosocial (hand visibility)",
            "Biopsy: EHK confirmed at diagnosis — no repeat needed unless phenotype changes",
            "Gene sequencing: confirm AD mutation; cascade family testing; prenatal/PGD counselling",
        ],
        "ppk_type": "Epidermolytic PPK (EPPK) — diffuse palmar, feet spared",
        "pathognomonic": "Palm-only PPK (feet completely spared) + EHK on biopsy (suprabasal vacuolation) = KRT9-EPPK",
        "treatment_highlight": "Urea 40% + keratolytic soaks; acitretin for severe; palm-only (no plantar) is pathognomonic; biopsy confirms EHK",
        "key_ddx": [
            "KRT1-EPPK (both palms + soles; EHK on biopsy — KRT9 feet spared; KRT1 feet involved)",
            "LORICRIN keratoderma (constriction bands + ichthyotic veil + NO SNHL — no biopsy EHK)",
            "Aquired PPK (keratoderma climactericum / arsenic — bilateral but acquired, no family history)",
            "Tyrosinemia type II (Richner-Hanhart — pseudoherpetic corneal ulcers + PPK + mental retardation; tyrosine level elevated)",
        ],
        "avg_age_at_dx_yrs": 1.5,
        "transgredient": False,
        "systemic_features": [],
        "cardiac_risk": False,
        "dental_risk": False,
    },
    # -- SLURP1 — Mal de Meleda disease ---------------------------------------------------
    {
        "gene": "SLURP1",
        "alt_name": (
            "SLURP1 (SLURP1-103aa-8q24.13 / AR — Mal-de-Meleda-Disease — "
            "TRANSGREDIENT-PPK-DORSAL-EXTENSION-ERYTHEMATOUS-BORDER-PATHOGNOMONIC — "
            "HYPERHIDROSIS-SEVERE-PSEUDOAINHUM-PERIORAL-ERYTHEMA — "
            "Mljet-Island-Adriatic-Croatia-Founder-p.W15R)"
        ),
        "protein": (
            "SLURP1 -- 8q24.13 AR -- SLURP1-103aa -- "
            "Secreted-Ly6-uPAR-Related-Protein-1-11kDa-GPI-Anchored-Cholinergic-Signalling -- "
            "Mal-de-Meleda-OMIM-248300 -- "
            "MODULATES-NICOTINIC-ACETYLCHOLINE-RECEPTOR-ALPHA7-Keratinocyte-Differentiation -- "
            "SLURP1-LOF-Aberrant-Cholinergic-Signalling-Epidermal-Hyperproliferation -- "
            "TRANSGREDIENT-PPK-DORSAL-EXTENSION-BEYOND-PALMOPLANTAR-MARGIN-PATHOGNOMONIC -- "
            "ERYTHEMATOUS-BORDER-Transition-Zone-Between-PPK-Normal-Skin -- "
            "HYPERHIDROSIS-SEVERE-Palms-Soles-Malodor-Maceration -- "
            "PSEUDOAINHUM-Constricting-Fibrous-Bands-Digits-Risk-Autoamputation -- "
            "PERIORAL-ERYTHEMA-Erythematous-Patch-Around-Mouth-Unique-Feature -- "
            "p.W15R-Most-Common-Adriatic-Mljet-Island-Founder -- "
            "OMIM-Gene-SLURP1-606119-Disease-MdM-248300"
        ),
        "locus": "8q24.13",
        "protein_size": "103 aa / 11 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF mutations; "
            "p.W15R (p.Trp15Arg): most common allele — Mljet Island (Adriatic Croatia) founder; "
            "Mal de Meleda = 'Mljet Island sickness' in Croatian (meleda=Mljet); "
            "High consanguinity in original Adriatic population; "
            "Rare worldwide; cases reported from Mediterranean, Middle East, North Africa; "
            "Severity: biallelic null → severe transgredient PPK with pseudoainhum; "
            "Hypomorphic compound heterozygotes → milder phenotype; "
            "SLURP1 GPI-anchored protein — regulates alpha7 nicotinic AChR signalling in keratinocytes; "
            "LOF → loss of cholinergic anti-proliferative signal → hyperproliferation + aberrant differentiation"
        ),
        "key_features": [
            "TRANSGREDIENT PPK — extends beyond palmoplantar margins to DORSAL SURFACES of hands and feet — PATHOGNOMONIC for Mal de Meleda; only a few PPK types are transgredient (SLURP1, JUP-Naxos rare)",
            "ERYTHEMATOUS BORDER — well-demarcated erythematous band at transition between PPK skin and normal dorsal skin — distinctive",
            "HYPERHIDROSIS (SEVERE) — profuse sweating of palms and soles; malodor; maceration; secondary candidal infection common",
            "PSEUDOAINHUM — constricting fibrous bands around digits; risk of digit autoamputation if untreated; surgical release required for severe constriction",
            "PERIORAL ERYTHEMA — erythematous patch around mouth (perioral distribution); distinguishes Mal de Meleda from other PPK types",
            "CONGENITAL ONSET — present at birth or within first weeks; progressive through childhood; lifelong",
            "p.W15R (Mljet/Adriatic founder): most common biallelic combination in Mediterranean Mal de Meleda; identifies founder population efficiently",
            "RETINOIDS (acitretin) most effective systemic therapy — reduces PPK thickness and transgredient extension; does not reverse pseudoainhum bands",
        ],
        "treatment": (
            "Topical (first-line): "
            "Urea 40-50% cream + occlusion overnight (keratolytic); "
            "Salicylic acid 10-20% paste (thick plaques); "
            "Lactic acid 12% lotion twice daily; "
            "Emollient base (petroleum jelly, Vaseline) for dry fissures. "
            "Hyperhidrosis management: "
            "20% aluminium chloride (Drysol) topical antiperspirant — reduces malodor, maceration; "
            "Botulinum toxin A injection — iontophoresis palmar/plantar — 3-6 monthly; "
            "Oral anticholinergics (glycopyrrolate) for severe — side-effect limited. "
            "Pseudoainhum: "
            "Surgical release of constricting fibrous bands — early intervention before vascular compromise; "
            "Dermatology + hand surgery co-management; monitor digits closely. "
            "Systemic: "
            "Acitretin 0.3-0.5 mg/kg/day — most effective; reduces PPK + transgredient spread; "
            "Neonates: emollient intensive; avoid tight clothing; "
            "Secondary infection: antifungals (candida) + antiseptic washes. "
            "Genetics: AR — 25% recurrence; prenatal/PGD available."
        ),
        "monitoring": [
            "3-6 monthly dermatology: PPK extent (dorsal spread quantification); hyperhidrosis severity; infection signs",
            "Annual: pseudoainhum surveillance — digit circumference measurement; early surgical referral if constriction detected",
            "Retinoid: LFTs + lipids monthly (first 3 months), then 3-monthly; teratogenicity counselling (females)",
            "QoL: DLQI; malodor/social impact VAS; occupational assessment",
            "Gene sequencing: biallelic SLURP1 confirmed; founder mutation p.W15R screen first in Mediterranean patients",
            "Family cascade: sibling screening; prenatal/PGD offered",
        ],
        "ppk_type": "Transgredient PPK (Mal de Meleda) — dorsal extension, hyperhidrosis, pseudoainhum",
        "pathognomonic": "Transgredient PPK with dorsal extension + erythematous border + hyperhidrosis + perioral erythema + pseudoainhum = SLURP1 Mal de Meleda",
        "treatment_highlight": "Retinoids most effective; pseudoainhum surgical release mandatory if constriction develops; hyperhidrosis management (aluminium chloride/botulinum)",
        "key_ddx": [
            "GJB2-Vohwinkel (honeycomb PPK + SNHL + starfish knuckle pads — not transgredient dorsal extension)",
            "LORICRIN keratoderma (constriction bands + ichthyotic veil — NO dorsal extension of PPK, NO SNHL)",
            "Tyrosinemia type II (Richner-Hanhart: corneal ulcers + PPK — serum tyrosine elevated; no transgredient)",
            "Naxos/JUP (ARVC + woolly hair + PPK — no hyperhidrosis; cardiac mandatory screen)",
        ],
        "avg_age_at_dx_yrs": 0.1,
        "transgredient": True,
        "systemic_features": ["Hyperhidrosis", "Pseudoainhum", "Perioral erythema"],
        "cardiac_risk": False,
        "dental_risk": False,
    },
    # -- CTSC — Papillon-Lefèvre syndrome --------------------------------------------------
    {
        "gene": "CTSC",
        "alt_name": (
            "CTSC (CTSC-463aa-11q14.2 / AR — Papillon-Lefèvre-Syndrome-PLS — "
            "PPK-SEVERE-EARLY-ONSET-PERIODONTITIS-PREMATURE-TOOTH-LOSS-PATHOGNOMONIC — "
            "PROPHYLACTIC-ANTIBIOTICS-MANDATORY-Amoxicillin-Metronidazole — "
            "Haim-Munk-Variant-PLS-Plus-Arachnodactyly-Acro-Osteolysis)"
        ),
        "protein": (
            "CTSC -- 11q14.2 AR -- CTSC-463aa -- "
            "Cathepsin-C-Dipeptidyl-Peptidase-I-DPPI-51kDa-Lysosomal-Cysteine-Protease -- "
            "PLS-OMIM-245000 -- "
            "ACTIVATES-NEUTROPHIL-SERINE-PROTEASES-Elastase-CG-PR3-NSP4-Essential-Immune -- "
            "CTSC-LOF-Impaired-NSP-Activation-Defective-Neutrophil-Pathogen-Killing -- "
            "PPK-DIFFUSE-PALMOPLANTAR-Transgredient-Variant-Some-Patients -- "
            "PERIODONTITIS-SEVERE-EARLY-ONSET-Primary-Teeth-3-4yr-Permanent-Teeth-By-14yr-ALL-LOST -- "
            "PROPHYLACTIC-ANTIBIOTICS-MANDATORY-Amoxicillin-Metronidazole-START-BEFORE-TEETH-ERUPT -- "
            "HAIM-MUNK-SYNDROME-CTSC-Same-Gene-PLS-Plus-Arachnodactyly-Acro-Osteolysis -- "
            "Hepatosplenomegaly-Recurrent-Skin-Infections-Pyogenic-Abscess -- "
            "OMIM-Gene-CTSC-602365-Disease-PLS-245000"
        ),
        "locus": "11q14.2",
        "protein_size": "463 aa / 51 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic missense/nonsense/splice; "
            "No major founder mutation — mutations distributed worldwide (consanguinity increases risk); "
            "PLS and Haim-Munk syndrome (HM) both caused by CTSC biallelic mutations — allelic disorders; "
            "Haim-Munk syndrome: PLS phenotype + arachnodactyly + acro-osteolysis (bone resorption at fingertips) + onychogryphosis — distinct clinical syndrome from same gene; "
            "Neutrophil serine protease (NSP) activation: cathepsin C activates elastase, cathepsin G, proteinase 3, NSP4 by cleaving propeptides; "
            "NSP deficiency → impaired neutrophil killing → severe periodontitis + recurrent infections; "
            "PPK severity varies: diffuse palmoplantar; some transgredient involvement; perioral + periorbital keratoderma in some"
        ),
        "key_features": [
            "SEVERE EARLY-ONSET PERIODONTITIS — aggressive destruction of periodontal attachment; primary (deciduous) teeth affected from age 3-4yr; ALL permanent teeth lost by age 14 without prophylactic antibiotics — PATHOGNOMONIC combination with PPK",
            "PPK — diffuse palmoplantar keratoderma; onset early childhood; variable severity; perioral/periorbital keratoderma variants",
            "PROPHYLACTIC ANTIBIOTICS (amoxicillin + metronidazole) — START BEFORE PERMANENT TEETH ERUPT (age 5-6yr) — MANDATORY intervention; preserves permanent dentition; consult paediatric periodontist immediately at diagnosis",
            "HAIM-MUNK SYNDROME — same CTSC gene; PLS + arachnodactyly + acro-osteolysis of distal phalanges + onychogryphosis — distinguished from classic PLS by radiological features",
            "RECURRENT PYOGENIC INFECTIONS — skin abscesses; furuncles; recurrent upper respiratory infections; pyogenic liver abscess (rare); due to neutrophil serine protease deficiency",
            "NEUTROPHIL SERINE PROTEASE DEFICIENCY — cathepsin C activates elastase/CG/PR3/NSP4; absent in PLS → neutrophils phagocytose but cannot kill certain pathogens efficiently",
            "HEPATOSPLENOMEGALY — 30-50% of PLS patients; related to recurrent infections + immune dysregulation; liver function generally preserved",
            "DENTAL SURVEILLANCE: dental panoramic X-ray annually from age 3yr; periodontal probing; plaque index; bacteria typing (A. actinomycetemcomitans) guides antibiotic choice",
        ],
        "treatment": (
            "DENTAL EMERGENCY (priority): "
            "Paediatric periodontist + dermatologist co-management mandatory from diagnosis; "
            "Prophylactic antibiotics: amoxicillin 250-500mg/day (or amoxicillin-clavulanate) + metronidazole 200-400mg/day; "
            "START before permanent teeth erupt (age 5-6yr); continue for 3+ years during teeth eruption period; "
            "Antiseptic mouth rinse: chlorhexidine 0.2% twice daily — mandatory; "
            "Dental scaling + root planing quarterly — remove subgingival biofilm; "
            "A. actinomycetemcomitans: if present, add ciprofloxacin; "
            "Orthodontic + prosthetic planning: if permanent teeth lost, implant rehabilitation after skeletal maturity. "
            "PPK treatment: "
            "Urea 40% cream + soaking + keratolytic — daily regimen; "
            "Salicylic acid 10-20% (thick plaques); "
            "Acitretin 0.3-0.5 mg/kg/day — reduces PPK; some evidence for periodontal benefit (reduces hyperkeratotic bacterial reservoir); "
            "Perioral/periorbital keratoderma: low-potency topical steroid + emollient intermittently. "
            "Infections: "
            "Prompt antibiotics for abscesses; drainage if needed; "
            "G-CSF (filgrastim) — case reports for severe recurrent infections. "
            "Genetics: AR — 25%; prenatal/PGD available."
        ),
        "monitoring": [
            "Dental: quarterly periodontal probing + scaling + root planing; annual OPG (panoramic X-ray); antibiotic compliance assessment",
            "Annual dermatology: PPK severity; perioral/periorbital keratoderma; fissure infection; retinoid monitoring",
            "Infection surveillance: recurrent abscess frequency; liver USS if hepatomegaly detected; neutrophil function testing",
            "Hepatic: liver function tests annually; USS if hepatosplenomegaly documented",
            "Retinoid: LFTs + triglycerides monthly (first 3 months), then 3-monthly; DEXA if long-term",
            "Dental prosthetics: orthodontic assessment from age 10yr; implant planning from age 18yr (after skeletal maturity)",
            "Gene panel: biallelic CTSC confirmed; Haim-Munk variant: hand X-ray (arachnodactyly, acro-osteolysis) + nail assessment",
            "QoL: psychosocial impact (tooth loss + visible hand PPK); school + social integration",
        ],
        "ppk_type": "PPK + Severe Periodontitis (Papillon-Lefèvre syndrome) — dental emergency",
        "pathognomonic": "PPK + severe early-onset periodontitis with premature primary/permanent tooth loss = CTSC Papillon-Lefèvre syndrome",
        "treatment_highlight": "Prophylactic antibiotics (amoxicillin+metronidazole) MANDATORY before teeth erupt — preserves dentition; retinoids for PPK; paediatric periodontist co-management",
        "key_ddx": [
            "Haim-Munk syndrome (same CTSC gene: PLS + arachnodactyly + acro-osteolysis — X-ray distinguishes)",
            "Chediak-Higashi (oculocutaneous albinism + neutrophil giant granules + periodontitis — albinism key DDx)",
            "Aggressive juvenile periodontitis (no PPK; no CTSC mutation — isolated dental; FPD/AML if RUNX1)",
            "Richner-Hanhart PPK (tyrosinemia II: corneal ulcers + PPK + mental retardation — serum tyrosine elevated; no periodontitis)",
        ],
        "avg_age_at_dx_yrs": 3.5,
        "transgredient": False,
        "systemic_features": ["Severe periodontitis", "Premature tooth loss", "Recurrent infections", "Hepatosplenomegaly"],
        "cardiac_risk": False,
        "dental_risk": True,
    },
    # -- GJB2 — Vohwinkel syndrome (mutilating PPK) / KID syndrome -------------------------
    {
        "gene": "GJB2",
        "alt_name": (
            "GJB2 (GJB2-226aa-13q12.11 / AD — Vohwinkel-Syndrome-Mutilating-PPK — "
            "HONEYCOMB-PPK-STARFISH-KNUCKLE-PADS-PSEUDOAINHUM-PATHOGNOMONIC — "
            "SNHL-Sensorineural-Hearing-Loss-Classic-Vohwinkel — "
            "KID-Syndrome-p.Asp50Asn-Keratitis-Ichthyosis-Deafness)"
        ),
        "protein": (
            "GJB2 -- 13q12.11 AD -- GJB2-226aa -- "
            "Connexin-26-Cx26-26kDa-Gap-Junction-Beta-2-Epidermis-Cochlea -- "
            "Vohwinkel-OMIM-124500-KID-OMIM-148210 -- "
            "FORMS-GAP-JUNCTION-CHANNELS-Cell-Cell-Ion-Metabolite-Small-Molecule-Communication -- "
            "GJB2-GOF/Missense-Disrupted-Channel-Gating-Dominant-Negative-Cx26-Hexamers -- "
            "HONEYCOMB-STARFISH-PPK-Palms-Soles-Pathognomonic-Pattern -- "
            "STARFISH-SHAPED-KNUCKLE-PADS-Over-PIP-MCP-Joints-Unique-Feature -- "
            "PSEUDOAINHUM-Constricting-Fibrous-Bands-Risk-Digit-Autoamputation -- "
            "SNHL-Sensorineural-Hearing-Loss-Classic-Vohwinkel-Cochlear-Cx26 -- "
            "p.Asp50Asn-D50N-KID-Syndrome-Keratitis-Ichthyosis-Deafness-Different-Phenotype -- "
            "DFNB1-AR-Biallelic-LOF-Isolated-Hearing-Loss-No-Skin -- "
            "OMIM-Gene-GJB2-121011-Disease-Vohwinkel-124500-KID-148210-DFNB1-220290"
        ),
        "locus": "13q12.11",
        "protein_size": "226 aa / 26 kDa",
        "inheritance": (
            "AD (autosomal dominant gain-of-function/dominant-negative missense for Vohwinkel + KID); "
            "AR (autosomal recessive loss-of-function for DFNB1 — isolated hearing loss, no skin); "
            "GJB2 p.Leu34Pro → Vohwinkel syndrome (classic): honeycomb PPK + SNHL; "
            "GJB2 p.Asp50Asn (D50N) → KID syndrome: Keratitis-Ichthyosis-Deafness — erythrokeratoderma + severe deafness + corneal vascularization (distinct phenotype); "
            "GJB2 biallelic LOF (e.g. 35delG) → DFNB1 isolated hearing loss — NO PPK, NO skin; "
            "Same gene, three completely different phenotypes depending on mutation type and zygosity; "
            "Vohwinkel + KID = dominant with dominant-negative/GOF channel dysfunction; "
            "Connexin-26 forms heteromeric gap junctions in epidermis + cochlear hair cells"
        ),
        "key_features": [
            "HONEYCOMB (STARFISH) PPK — keratoderma with honeycomb pattern at palms/soles — PATHOGNOMONIC for Vohwinkel; distinctive textured surface not seen in other PPK forms",
            "STARFISH-SHAPED KNUCKLE PADS — hyperkeratotic nodules over PIP + MCP joints in a starfish/sea-star distribution — PATHOGNOMONIC for Vohwinkel syndrome",
            "PSEUDOAINHUM — constricting fibrous bands around digits; risk of digit autoamputation if untreated; emergency surgical release if vascular compromise",
            "SENSORINEURAL HEARING LOSS (SNHL) — classic Vohwinkel has congenital/childhood SNHL; cochlear Connexin-26 gap junctions essential for endocochlear potential maintenance; hearing aids + cochlear implant",
            "KID SYNDROME (p.Asp50Asn / D50N) — DIFFERENT phenotype: keratitis (corneal vascularization → blindness risk) + generalised ichthyosiform erythroderma + deafness; NOT classic Vohwinkel honeycomb PPK",
            "DFNB1 DISTINCTION: biallelic GJB2 LOF (35delG most common worldwide) → isolated hearing loss — NO PPK, NO keratoderma — must be distinguished from GJB2-Vohwinkel/KID to avoid unnecessary dermatological evaluation",
            "CONNEXIN-26 CHANNEL DYSFUNCTION — GOF mutations increase intracellular Ca2+ flux and pro-apoptotic signalling in keratinocytes → aberrant differentiation + thickening",
            "LORICRIN DDx — LORICRIN keratoderma also has pseudoainhum + PPK but NO SNHL and ichthyotic veil; hearing test distinguishes",
        ],
        "treatment": (
            "PPK + knuckle pads: "
            "Keratolytic emollients: urea 40-50% cream feet/palms; salicylic acid 15-20% paste; daily soaking + debridement; "
            "Retinoids (acitretin 0.3-0.5 mg/kg/day): reduces PPK and knuckle pads; does not reverse pseudoainhum; "
            "Topical retinoids (tretinoin 0.05-0.1%): adjunct for knuckle pads. "
            "Pseudoainhum: "
            "Surgical release: early intervention; hand surgery + dermatology co-management; "
            "Monitor digit perfusion + circumference; "
            "Emergency release if digit ischaemia (cyanosis/necrosis onset). "
            "Hearing: "
            "Hearing assessment: audiometry at diagnosis + annually; "
            "Hearing aids: for mild-moderate SNHL; "
            "Cochlear implant: if profound SNHL — excellent outcomes with early implantation; "
            "Speech therapy + educational support. "
            "KID syndrome: "
            "Ophthalmology: corneal vascularization → keratitis → blindness; "
            "Topical ciclosporin + corneal grafting; regular slit-lamp. "
            "Genetics: AD — 50% recurrence; prenatal/PGD offered."
        ),
        "monitoring": [
            "Dermatology 3-6 monthly: PPK severity; knuckle pads assessment; pseudoainhum — digit circumference measurement",
            "Annual audiometry: SNHL progression; hearing aid fitting review; cochlear implant candidacy assessment",
            "Ophthalmology (KID): annual slit-lamp — corneal vascularization; topical treatment efficacy",
            "Pseudoainhum: digit vascular assessment at every visit; hand surgery review annually",
            "Retinoid: LFTs + lipids monthly (first 3 months), then 3-monthly; teratogenic",
            "QoL: DLQI; hearing QoL (HHI); hand function; psychosocial (digit deformity + hearing loss)",
            "Gene: confirm AD GJB2 mutation (distinguish Vohwinkel vs KID vs DFNB1 by specific variant)",
            "Audiology: cochlear implant follow-up; language acquisition assessment in children",
        ],
        "ppk_type": "Mutilating PPK (Vohwinkel) — honeycomb pattern, starfish knuckle pads, SNHL, pseudoainhum",
        "pathognomonic": "Honeycomb PPK + starfish-shaped knuckle pads + pseudoainhum + SNHL = GJB2 Vohwinkel syndrome",
        "treatment_highlight": "Pseudoainhum surgical release mandatory if constriction; cochlear implant for SNHL; acitretin for PPK; distinguish Vohwinkel (SNHL) from LORICRIN (no SNHL)",
        "key_ddx": [
            "LORICRIN keratoderma (pseudoainhum + ichthyotic veil — NO SNHL — hearing test distinguishes)",
            "SLURP1 Mal de Meleda (transgredient PPK + hyperhidrosis — no SNHL, no knuckle pads)",
            "GJB2 KID syndrome (same gene p.D50N: keratitis + ichthyosis + deafness — NOT honeycomb PPK; corneal involvement distinguishes)",
            "GJB2 DFNB1 (biallelic LOF: isolated SNHL — NO PPK; do not confuse with Vohwinkel)",
        ],
        "avg_age_at_dx_yrs": 2.0,
        "transgredient": False,
        "systemic_features": ["Sensorineural hearing loss", "Pseudoainhum", "Starfish knuckle pads"],
        "cardiac_risk": False,
        "dental_risk": False,
    },
    # -- DSP — Carvajal syndrome (PPK + woolly hair + dilated cardiomyopathy) ----------------
    {
        "gene": "DSP",
        "alt_name": (
            "DSP (DSP-2871aa-6p24.3 / AR — Carvajal-Syndrome — "
            "PPK-WOOLLY-HAIR-DILATED-CARDIOMYOPATHY-DCM-PATHOGNOMONIC — "
            "CARDIAC-MRI-MANDATORY-SCD-RISK-TEENAGERS — "
            "vs-JUP-Naxos-ARVC-NOT-DCM-LEFT-vs-RIGHT-KEY-DDx)"
        ),
        "protein": (
            "DSP -- 6p24.3 AR -- DSP-2871aa -- "
            "Desmoplakin-DP-332kDa-Largest-Desmosomal-Protein-Plakin-Family -- "
            "Carvajal-OMIM-605676 -- "
            "LINKS-DESMOSOMAL-CADHERINS-TO-INTERMEDIATE-FILAMENTS-Desmin-in-Heart-Keratins-in-Skin -- "
            "DSP-AR-LOF-Loss-Desmosomal-Integrity-Heart-Skin -- "
            "PPK-Diffuse-Palmoplantar-Keratoderma-Onset-Infancy -- "
            "WOOLLY-HAIR-Tightly-Curled-Kinky-Hair-All-Patients -- "
            "DILATED-CARDIOMYOPATHY-DCM-LEFT-VENTRICULAR-DYSFUNCTION-PATHOGNOMONIC -- "
            "SCD-Sudden-Cardiac-Death-Teenagers-Young-Adults-Without-ICD -- "
            "CARDIAC-MRI-MANDATORY-LGE-Subepicardial-Fibrosis-Early-Marker -- "
            "ICD-If-Significant-LV-Dysfunction-EF<35pct -- "
            "vs-JUP-Naxos-ARVC-RIGHT-VENTRICULAR-DIFFERENT-PATHOGNOMONIC -- "
            "OMIM-Gene-DSP-125647-Disease-Carvajal-605676"
        ),
        "locus": "6p24.3",
        "protein_size": "2871 aa / 332 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF mutations (premature stop, frameshift, splice); "
            "Carvajal syndrome: recessive DSP mutations → PPK + woolly hair + DCM; "
            "AD DSP mutations → different phenotype: ARVC (arrhythmogenic cardiomyopathy) ± skin — similar to JUP but dominant; "
            "Key distinction: AR DSP → DILATED (LV) cardiomyopathy (Carvajal); AD DSP → ARVC (RV); "
            "Original family described in Ecuador (Carvajal, 1998); now reported worldwide in consanguineous families; "
            "Woolly hair: kinky/tightly curled in all affected patients (both sexes); congenital; "
            "DCM onset: childhood to early adulthood; rapid progression to heart failure and SCD without intervention"
        ),
        "key_features": [
            "DILATED CARDIOMYOPATHY (DCM) — LEFT ventricular dilatation + systolic dysfunction — PATHOGNOMONIC cardiac phenotype in Carvajal syndrome; onset childhood/teenage; progressive → heart failure + SCD",
            "WOOLLY HAIR — tightly curled/kinky hair in all patients regardless of ethnicity; congenital; does not progress; shared with Naxos (JUP) but cardiac type distinguishes",
            "DIFFUSE PPK — palmoplantar keratoderma; onset infancy; moderate severity; variable extent; secondary to desmosomal loss in skin",
            "SUDDEN CARDIAC DEATH RISK IN TEENAGERS — WITHOUT ICD: ventricular arrhythmias → SCD; ICD mandatory if EF <35% or significant arrhythmia burden",
            "CARDIAC MRI — late gadolinium enhancement (LGE) — subepicardial fibrosis PATHOGNOMONIC early marker; detects cardiac involvement before echocardiographic dysfunction",
            "ICD (IMPLANTABLE CARDIOVERTER-DEFIBRILLATOR) — mandatory if significant LV dysfunction; electrophysiology assessment; family ICD discussion",
            "DSP vs JUP DDx — CRITICAL: DSP-AR → DILATED (LEFT ventricle) = Carvajal; JUP-AR → ARVC (RIGHT ventricle) = Naxos; LEFT vs RIGHT = KEY DDx; cardiac MRI distinguishes definitively",
            "DSP-AD (dominant) → ARVC pattern (similar to JUP/Naxos) — dominant DSP acts like JUP phenotypically; recessive DSP = DCM only",
        ],
        "treatment": (
            "Cardiac (PRIORITY — life-saving): "
            "Cardiology + cardiac genetics mandatory from diagnosis; "
            "Echocardiography at diagnosis: LVEF, LV dimensions, wall motion; "
            "Cardiac MRI: LGE mapping (subepicardial fibrosis early marker); CMR annually if LGE present; "
            "ICD: if LVEF <35% or sustained VT/VF; "
            "Heart failure therapy (if reduced EF): ACE-inhibitor/ARB + beta-blocker + MRA (aldosterone antagonist); "
            "Sports restriction: intense competitive sports ABSOLUTELY CONTRAINDICATED — SCD trigger; "
            "Annual 24-hour Holter + cardiac MRI until age 30, then 2-yearly if stable. "
            "Heart transplantation: if end-stage DCM refractory to therapy. "
            "Skin: "
            "Urea 40% cream + keratolytic soaks — daily regimen; "
            "Acitretin 0.3-0.5 mg/kg/day — reduces PPK; monitor LFTs + lipids + cardiac interaction; "
            "Emollients + salicylic acid for thick plaques. "
            "Genetics: AR — 25%; prenatal/PGD available; sibling cardiac screening mandatory."
        ),
        "monitoring": [
            "CARDIAC: 6-monthly echocardiography (LVEF, dimensions); annual 24-hour Holter; cardiac MRI with LGE annually until stable",
            "ICD check: device interrogation 3-6 monthly; shock appropriateness; lead status",
            "Sports restriction counselling: repeat annually; school/work physical activity plan",
            "Dermatology: 6 monthly — PPK severity; retinoid side effects (important in cardiac patients)",
            "Heart failure: fluid balance; exercise tolerance assessment; BNP/NT-proBNP monitoring",
            "Genetics: first-degree family cardiac screening (echo + cardiac MRI); cascade testing mandatory",
            "QoL: cardiac symptoms; psychosocial; ICD anxiety; school/occupation integration",
        ],
        "ppk_type": "PPK + Woolly Hair + Dilated Cardiomyopathy (Carvajal syndrome) — cardiac emergency",
        "pathognomonic": "PPK + woolly hair + dilated cardiomyopathy (LEFT ventricular) = DSP Carvajal syndrome",
        "treatment_highlight": "Cardiac MRI + ICD if EF<35% — life-saving; sports restriction absolute; DCM (LEFT ventricle) distinguishes from Naxos/JUP (RIGHT ARVC)",
        "key_ddx": [
            "JUP-Naxos (PPK + woolly hair + ARVC RIGHT ventricular — RIGHT not LEFT is KEY DDx; cardiac MRI distinguishes)",
            "AD DSP mutations (same gene; dominant → ARVC pattern not DCM; zygosity testing critical)",
            "Dilated cardiomyopathy without PPK (LMNA/TTN/FLNC — no woolly hair; no skin; genetic panel)",
            "ARVC without PPK (PKP2/DSG2/DSGDSP-AD — RIGHT ventricular; no skin involvement)",
        ],
        "avg_age_at_dx_yrs": 5.0,
        "transgredient": False,
        "systemic_features": ["Woolly hair", "Dilated cardiomyopathy", "Sudden cardiac death risk"],
        "cardiac_risk": True,
        "dental_risk": False,
    },
    # -- JUP — Naxos disease (PPK + woolly hair + ARVC) -----------------------------------
    {
        "gene": "JUP",
        "alt_name": (
            "JUP (JUP-745aa-17q21.2 / AR — Naxos-Disease — "
            "PPK-WOOLLY-HAIR-ARVC-RIGHT-VENTRICULAR-CARDIOMYOPATHY-PATHOGNOMONIC — "
            "ICD-MANDATORY-Sports-Restriction-Absolute — "
            "vs-DSP-Carvajal-DCM-LEFT-NOT-RIGHT-KEY-DDx)"
        ),
        "protein": (
            "JUP -- 17q21.2 AR -- JUP-745aa -- "
            "Plakoglobin-Junction-Plakoglobin-81kDa-Desmosome-Adherens-Junction-Dual-Role -- "
            "Naxos-OMIM-601214 -- "
            "STRUCTURAL-COMPONENT-DESMOSOMES-Epidermis-Heart-Links-Desmoglins-Desmoplakins -- "
            "DUAL-ROLE-Wnt-Beta-Catenin-Signalling-Nuclear-Localisation-Transcription -- "
            "JUP-AR-LOF-Desmosomal-Instability-Heart-Skin -- "
            "PPK-Diffuse-Palmoplantar-Onset-Infancy -- "
            "WOOLLY-HAIR-Tightly-Curled-ALL-Patients-Congenital -- "
            "ARVC-Arrhythmogenic-Right-Ventricular-Cardiomyopathy-RIGHT-Ventricle-PATHOGNOMONIC -- "
            "VENTRICULAR-ARRHYTHMIAS-Sustained-VT-SCD-Risk -- "
            "GREEK-NAXOS-ISLAND-FOUNDER-POPULATION-Greek-Aegean-c2157del2-Deletion -- "
            "ICD-MANDATORY-Sports-Restriction-ABSOLUTE -- "
            "vs-DSP-Carvajal-DCM-LEFT-DIFFERENT -- "
            "OMIM-Gene-JUP-173325-Disease-Naxos-601214"
        ),
        "locus": "17q21.2",
        "protein_size": "745 aa / 81 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic deletion/insertion causing frameshift; "
            "c.2157del2 (2-bp deletion, exon 17): founder mutation on Naxos island (Greek Aegean); "
            "Originally described in Greek island of Naxos population — hence 'Naxos disease'; "
            "Rare outside Greek/Mediterranean populations; cases in Turkish, Arab, Israeli populations; "
            "Plakoglobin: desmosomal protein linking desmosomal cadherins (Dsg/Dsc) to desmoplakin; "
            "Also participates in Wnt/beta-catenin signalling (nuclear plakoglobin); "
            "AR LOF → destabilised desmosomes → cardiomyocyte detachment → fibrofatty replacement (ARVC); "
            "100% penetrance for cardiac phenotype in males; lower penetrance in females"
        ),
        "key_features": [
            "ARVC — ARRHYTHMOGENIC RIGHT VENTRICULAR CARDIOMYOPATHY — RIGHT ventricular fibrofatty replacement + right ventricular dilatation/dysfunction — PATHOGNOMONIC cardiac phenotype in Naxos disease",
            "VENTRICULAR ARRHYTHMIAS — sustained ventricular tachycardia; ventricular fibrillation; sudden cardiac death in teenagers/young adults WITHOUT ICD intervention",
            "WOOLLY HAIR — tightly curled kinky hair; congenital; present in all affected patients; shared with Carvajal (DSP) but cardiac type distinguishes definitively",
            "DIFFUSE PPK — palmoplantar keratoderma; onset infancy; similar severity to Carvajal; variable; secondary desmosomal skin loss",
            "ICD (IMPLANTABLE CARDIOVERTER-DEFIBRILLATOR) MANDATORY — life-saving; indicate for documented ventricular arrhythmia or significant RV dysfunction; electrophysiology assessment",
            "SPORTS RESTRICTION ABSOLUTE — competitive/intense sports absolutely contraindicated; exercise is an arrhythmia trigger in ARVC; school physical education modification mandatory",
            "DSP vs JUP DDx — CRITICAL: JUP-AR → ARVC (RIGHT ventricle) = Naxos; DSP-AR → DCM (LEFT ventricle) = Carvajal; RIGHT vs LEFT = KEY DDx; cardiac MRI definitively distinguishes",
            "GREEK FOUNDER: c.2157del2 (2-bp deletion) in Greek Naxos island families; genetic testing identifies founder allele rapidly",
        ],
        "treatment": (
            "Cardiac (PRIORITY — life-saving): "
            "Arrhythmology + cardiac genetics mandatory from diagnosis; "
            "ECG: EPSILON WAVES (V1-V3), inverted T-waves V1-V4, prolonged QRS >110ms in V1 — ARVC diagnostic; "
            "Echocardiography: RV dimensions, RVEF, wall motion abnormalities (basal RV aneurysm); "
            "Cardiac MRI: RV fibrofatty infiltration + LGE; CMR annually; "
            "ICD: if VT, VF, or significant RV dysfunction — primary prevention discussed with EP; "
            "Antiarrhythmic: sotalol/amiodarone adjunct; "
            "Sports restriction: ABSOLUTE — competitive sports prohibited; moderate recreational walking only; "
            "Family screening: ECG + echo + CMR in ALL first-degree relatives from puberty. "
            "Skin: "
            "Urea 40% cream + keratolytic daily — PPK management; "
            "Acitretin 0.3-0.5 mg/kg/day — PPK reduction; monitor LFTs + cardiac interaction. "
            "Genetics: AR — 25%; c.2157del2 founder rapid screen in Greek families; prenatal/PGD available."
        ),
        "monitoring": [
            "CARDIAC: 6-monthly ECG + echocardiography (RV dimensions, RVEF); annual 24-hour Holter; annual cardiac MRI",
            "ICD: device interrogation 3-6 monthly; shock log review; lead status",
            "Sports restriction: enforced; school/work physical plan; repeat counselling annually",
            "Dermatology: 6 monthly — PPK severity; retinoid side effects",
            "Arrhythmia symptoms: palpitations/syncope/presyncope diary; ICD interrogation after each episode",
            "Family screening: ECG + echo + CMR in first-degree relatives; children screened from puberty",
            "QoL: cardiac anxiety; ICD adjustment; school/occupation integration",
        ],
        "ppk_type": "PPK + Woolly Hair + ARVC (Naxos disease) — cardiac emergency",
        "pathognomonic": "PPK + woolly hair + ARVC (RIGHT ventricular cardiomyopathy) = JUP Naxos disease",
        "treatment_highlight": "ICD mandatory + sports restriction absolute — life-saving; ARVC (RIGHT ventricle) distinguishes from Carvajal/DSP (LEFT DCM); Greek founder c.2157del2",
        "key_ddx": [
            "DSP-Carvajal (PPK + woolly hair + DCM LEFT ventricle — LEFT not RIGHT is KEY DDx; cardiac MRI distinguishes)",
            "ARVC without PPK (PKP2/DSG2/DSC2 — same ARVC phenotype but NO woolly hair + NO PPK)",
            "Brugada syndrome (RV ST changes — no woolly hair; no PPK; different genetics; fever trigger)",
            "AD-DSP mutations (same desmoplakin gene but dominant → ARVC phenotype similar to JUP; PPK variable; heterozygous)",
        ],
        "avg_age_at_dx_yrs": 4.5,
        "transgredient": False,
        "systemic_features": ["Woolly hair", "ARVC", "Ventricular arrhythmias", "Sudden cardiac death risk"],
        "cardiac_risk": True,
        "dental_risk": False,
    },
    # -- SERPINB7 — Bothnian type PPK / aquagenic PPK ------------------------------------
    {
        "gene": "SERPINB7",
        "alt_name": (
            "SERPINB7 (SERPINB7-394aa-18q21.33 / AR — Bothnian-Type-PPK-NEPPK2 — "
            "AQUAGENIC-WRINKLING-PALMS-PATHOGNOMONIC-Exaggerated-Wrinkling-Minutes-Water — "
            "Swedish-Bothnian-Region-Founder-p.Trp249Ter — "
            "MILD-DISEASE-No-Systemic-Involvement)"
        ),
        "protein": (
            "SERPINB7 -- 18q21.33 AR -- SERPINB7-394aa -- "
            "Serine-Protease-Inhibitor-B7-44kDa-Maspin-Related-Squamous-Cell-Carcinoma -- "
            "Bothnian-PPK-OMIM-612111 -- "
            "INHIBITS-SERINE-PROTEASES-Epidermal-Homeostasis-Barrier-Function -- "
            "SERPINB7-LOF-Impaired-Protease-Inhibition-Epidermal-Desquamation-Defect -- "
            "AQUAGENIC-WRINKLING-PALMS-Exaggerated-Rapid-Wrinkling-Minutes-Water-PATHOGNOMONIC -- "
            "PALMAR-PITS-Punctate-Depressions-Palmar-Skin-Distinctive -- "
            "PUNCTATE-HYPERKERATOSIS-Fine-Papular-Keratoses-Palms-Soles -- "
            "MILD-DISEASE-No-Cardiac-Dental-Systemic-Involvement -- "
            "p.Trp249Ter-W249X-Swedish-Bothnian-Region-Founder -- "
            "EMOLLIENT-ANTIPERSPIRANT-Symptomatic-No-Curative-Treatment -- "
            "OMIM-Gene-SERPINB7-603357-Disease-NEPPK2-612111"
        ),
        "locus": "18q21.33",
        "protein_size": "394 aa / 44 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF mutations; "
            "p.Trp249Ter (W249X): Swedish (Bothnian region) founder mutation — high frequency in northern Sweden; "
            "Bothnian type PPK named after Bothnian region of Sweden (Gulf of Bothnia, north); "
            "Relatively mild phenotype — no systemic features; "
            "SERPINB7 inhibits serine proteases involved in epidermal turnover; "
            "LOF → overactive serine proteases → accelerated corneodesmolysis → punctate/diffuse hyperkeratosis + water-triggered barrier failure; "
            "Aquagenic wrinkling mechanism: defective barrier allows water entry → rapid osmotic swelling of SC; "
            "Compound heterozygotes (W249X + other LOF) seen outside Bothnian region; "
            "Phenotypic penetrance near 100% in biallelic carriers"
        ),
        "key_features": [
            "AQUAGENIC WRINKLING OF PALMS — exaggerated rapid wrinkling within minutes of water exposure (hand washing, swimming) — PATHOGNOMONIC for SERPINB7-PPK; persists and worsens on prolonged water contact; normal population has some wrinkling but not within 2-3 minutes",
            "PALMAR PITS — multiple small punctate depressions on palmar surface — distinctive feature unique to SERPINB7 among common PPK forms",
            "PUNCTATE HYPERKERATOSIS — fine papular/punctate keratoses scattered across palms and soles; not diffuse thick plaque-like (distinguishes from EPPK types)",
            "MILD DISEASE — NO cardiac, dental, hearing, or other systemic involvement; quality of life affected mainly by cosmesis and aquagenic wrinkling discomfort",
            "p.Trp249Ter (W249X) — Swedish Bothnian founder mutation; efficient screening by targeted sequencing before full panel in northern Swedish patients",
            "NON-EPIDERMOLYTIC — biopsy shows no EHK (no suprabasal vacuolation/epidermolysis); hyperkeratosis without epidermolytic changes — distinguishes from KRT9-EPPK",
            "EMOLLIENT-BASED MANAGEMENT — no curative treatment; antiperspirant (aluminium chloride) reduces aquagenic wrinkling severity; avoidance of prolonged water exposure",
            "CYSTIC FIBROSIS CARRIER ASSOCIATION — aquagenic wrinkling of palms also occurs in CFTR carrier/CF patients (CFTR-related); check sweat chloride if SERPINB7 negative in aquagenic wrinkling presentation",
        ],
        "treatment": (
            "First-line (symptomatic — no curative treatment): "
            "Emollients: petroleum jelly / dimethicone-based barrier cream applied before water exposure — reduces aquagenic wrinkling; "
            "Antiperspirant: 20% aluminium chloride (Drysol) to palms — reduces sweating + aquagenic wrinkling severity; "
            "Keratolytic: urea 10-20% cream twice daily — reduces punctate hyperkeratosis; "
            "Salicylic acid 5-10% — adjunct for prominent hyperkeratotic papules. "
            "Aquagenic wrinkling management: "
            "Barrier gloves for prolonged water exposure (washing dishes, swimming); "
            "Apply emollient/barrier cream before handwashing; "
            "Limit water contact time where practical. "
            "Secondary measures: "
            "Botulinum toxin A injection — reduces hyperhidrosis contribution to aquagenic wrinkling; "
            "Iontophoresis — palmar anhidrosis adjunct. "
            "Genetics: AR — 25% recurrence; p.Trp249Ter founder allele rapid screen in Swedish patients; prenatal/PGD available (if desired — mild phenotype)."
        ),
        "monitoring": [
            "Annual dermatology: punctate hyperkeratosis extent; aquagenic wrinkling severity score; emollient efficacy",
            "Antiperspirant compliance: aquagenic wrinkling diary; response assessment",
            "QoL: DLQI; social impact (visible wrinkling, work with water); patient-reported outcome",
            "Cystic fibrosis screen: if aquagenic wrinkling prominent — sweat chloride + CFTR sequencing to exclude CF/carrier",
            "Gene sequencing: biallelic SERPINB7 confirmed; p.Trp249Ter rapid screen in Swedish patients",
            "Family cascade: sibling testing if biallelic confirmed; genetic counselling (mild phenotype — recurrence probability discussion)",
        ],
        "ppk_type": "Aquagenic PPK (Bothnian type / NEPPK2) — aquagenic wrinkling, palmar pits, mild",
        "pathognomonic": "Aquagenic wrinkling of palms within minutes of water exposure + palmar pits + punctate hyperkeratosis = SERPINB7 Bothnian PPK",
        "treatment_highlight": "Barrier cream before water exposure; antiperspirant (aluminium chloride); no systemic disease; Swedish p.Trp249Ter founder allele",
        "key_ddx": [
            "Cystic fibrosis/CFTR carrier (aquagenic wrinkling also in CF — sweat chloride + CFTR sequencing; no PPK in CF)",
            "KRT9-EPPK (thick diffuse palmar PPK — not aquagenic wrinkling; EHK on biopsy; feet spared; no pits)",
            "Palmoplantar punctate keratoderma type 1 (Coleman-Harber — punctate but no aquagenic feature; COL14A1)",
            "Darier disease (ATP2A2 — follicular keratoses; nails; no aquagenic wrinkling)",
        ],
        "avg_age_at_dx_yrs": 10.0,
        "transgredient": False,
        "systemic_features": ["Aquagenic wrinkling", "Palmar pits"],
        "cardiac_risk": False,
        "dental_risk": False,
    },
    # -- LORICRIN — Loricrin keratoderma / Vohwinkel variant without hearing loss ----------
    {
        "gene": "LORICRIN",
        "alt_name": (
            "LORICRIN (LORICRIN-312aa-1q21.3 / AD — Loricrin-Keratoderma-Vohwinkel-Variant — "
            "PPK-CONSTRICTION-BANDS-ICHTHYOTIC-VEIL-PATHOGNOMONIC — "
            "NO-HEARING-LOSS-KEY-DDx-GJB2-Vohwinkel-SNHL — "
            "Major-Cornified-Envelope-Protein-70-80pct-CE-Frameshift-Nuclear-Accumulation)"
        ),
        "protein": (
            "LORICRIN -- 1q21.3 AD -- LORICRIN-312aa -- "
            "Loricrin-37kDa-Major-Cornified-Envelope-Protein-70-80pct-CE-Mass -- "
            "Loricrin-Keratoderma-OMIM-604117 -- "
            "CROSS-LINKED-BY-TGase-1-TGase-3-Forms-CE-Backbone-Glutamine-Rich-Repeats -- "
            "LORICRIN-FRAMESHIFT-Altered-C-Terminus-Nuclear-Accumulation-Protein -- "
            "PPK-Diffuse-Palmoplantar-Onset-Infancy-Childhood -- "
            "CONSTRICTION-BANDS-Pseudoainhum-Fibrous-Digit-Constrictions-Autoamputation-Risk -- "
            "ICHTHYOTIC-VEIL-Fine-Generalised-Ichthyosis-Subtle-Scaly-Skin-All-Body -- "
            "NO-HEARING-LOSS-Critical-DDx-from-GJB2-Vohwinkel-SNHL -- "
            "RETINOIDS-Partially-Effective-Surgery-Constriction-Bands -- "
            "OMIM-Gene-LOR-152445-Disease-LK-604117"
        ),
        "locus": "1q21.3",
        "protein_size": "312 aa / 37 kDa",
        "inheritance": (
            "AD (autosomal dominant); frameshift mutations in loricrin C-terminal region; "
            "Frameshift mutations alter the C-terminal loricrin sequence → nuclear accumulation of mutant loricrin; "
            "Normal loricrin cytoplasmic → mutant frameshifted loricrin nuclear (p62-positive inclusions); "
            "Dominant effect: one mutant allele sufficient for disease; haploinsufficiency NOT the mechanism; "
            "Loricrin = major CE protein (70-80% of CE mass); "
            "Frameshifted protein has altered sequence that creates nuclear localisation signal or escapes cytoplasmic retention; "
            "Phenotypic overlap with GJB2-Vohwinkel: both have PPK + pseudoainhum; "
            "Key distinguishing feature: LORICRIN has NO SNHL; GJB2-Vohwinkel has SNHL; audiometry confirms; "
            "LORICRIN also has ichthyotic veil (fine generalised ichthyosis) not seen in GJB2-Vohwinkel"
        ),
        "key_features": [
            "NO HEARING LOSS — KEY DDx from GJB2 Vohwinkel syndrome (which has SNHL); audiometry normal in all LORICRIN keratoderma patients; this single finding separates the two diagnoses clinically",
            "CONSTRICTION BANDS (PSEUDOAINHUM) — fibrous constricting bands around digits; progressive risk of autoamputation; surgical release mandatory if vascular compromise; same feature as GJB2-Vohwinkel but WITHOUT SNHL",
            "PPK — diffuse palmoplantar keratoderma; honeycomb-like pattern (similar to Vohwinkel but less pronounced starfish knuckle pads); onset infancy-childhood",
            "ICHTHYOTIC VEIL — fine generalised ichthyosis; subtle scaly skin all over body; often not prominent; distinguishes from classic GJB2-Vohwinkel (no generalised ichthyosis in GJB2)",
            "NUCLEAR ACCUMULATION OF MUTANT LORICRIN — frameshift creates altered C-terminus targeting protein to nucleus instead of CE; immunohistochemistry shows p62-positive nuclear inclusions (research tool, not routine)",
            "RETINOIDS (ACITRETIN) — partially effective for PPK and ichthyotic veil; does not prevent constriction band formation; continued long-term",
            "SURGICAL RELEASE — for pseudoainhum bands; hand surgery + dermatology co-management; early intervention before vascular compromise; similar to GJB2-Vohwinkel management",
            "LORICRIN 1q21 EPIDERMAL DIFFERENTIATION COMPLEX — neighbour genes include FLG, HRNR, SPRR proteins; chromosomal region important in skin barrier",
        ],
        "treatment": (
            "PPK + ichthyotic veil: "
            "Emollients: urea 20-40% cream twice daily; lactic acid 10-12% lotion; "
            "Keratolytics: salicylic acid 10-15% paste for thick plaques; soaking + mechanical debridement; "
            "Retinoids: acitretin 0.3-0.5 mg/kg/day — partially effective for PPK + ichthyotic veil; "
            "monitor LFTs + lipids + teratogenicity. "
            "Constriction bands (pseudoainhum — priority if present): "
            "Surgical release: Z-plasty or simple division of constricting fibrous band; "
            "Early intervention before vascular compromise (digit cyanosis/ischaemia); "
            "Hand surgery + dermatology co-management; monitor digit circumference + perfusion quarterly; "
            "Emergency release if acute ischaemia (same-day intervention). "
            "Ichthyotic veil: "
            "Generalised emollients: urea 10-20% body lotion; lactic acid; "
            "Bath oil + soak-and-smear technique; "
            "Low-potency topical retinoid (tretinoin 0.025%) for ichthyotic skin adjunct. "
            "Hearing: audiometry confirms NO SNHL — reassure patient/family. "
            "Genetics: AD — 50% recurrence; frameshift variant in LORICRIN; prenatal/PGD offered."
        ),
        "monitoring": [
            "Dermatology 3-6 monthly: PPK severity; ichthyotic veil extent; constriction band detection; digit circumference measurement",
            "Annual audiometry: confirms NO SNHL — differentiates from GJB2-Vohwinkel; baseline + annual until adulthood",
            "Pseudoainhum: digit vascular assessment at every visit; surgical review annually; early referral if constriction detected",
            "Retinoid: LFTs + lipids monthly (first 3 months), then 3-monthly; teratogenic (females); DEXA if long-term",
            "QoL: DLQI; hand function (pseudoainhum impact); psychosocial (skin visibility + digit deformity)",
            "Gene sequencing: AD frameshift in LORICRIN; 50% first-degree risk; family cascade testing",
        ],
        "ppk_type": "Loricrin keratoderma (Vohwinkel variant) — pseudoainhum, ichthyotic veil, NO SNHL",
        "pathognomonic": "PPK + constriction bands (pseudoainhum) + ichthyotic veil + NO HEARING LOSS = LORICRIN keratoderma (vs GJB2-Vohwinkel with SNHL)",
        "treatment_highlight": "Audiometry confirms NO SNHL (DDx from GJB2-Vohwinkel); surgical release of constriction bands; retinoids partially effective",
        "key_ddx": [
            "GJB2-Vohwinkel (honeycomb PPK + starfish knuckle pads + pseudoainhum + SNHL — hearing test is the KEY DDx; LORICRIN has NO SNHL)",
            "SLURP1 Mal de Meleda (transgredient PPK + hyperhidrosis — no ichthyotic veil; no SNHL; no constriction bands predominantly)",
            "KRT9-EPPK (palm-only; EHK biopsy; no constriction bands; no ichthyotic veil)",
            "CTSC Papillon-Lefèvre (PPK + severe periodontitis — no constriction bands; dental emergency feature distinguishes)",
        ],
        "avg_age_at_dx_yrs": 2.5,
        "transgredient": False,
        "systemic_features": ["Pseudoainhum", "Ichthyotic veil (fine generalised ichthyosis)"],
        "cardiac_risk": False,
        "dental_risk": False,
    },
]


def _make_ppk_patient(gene_entry: dict, seed: int) -> dict:
    rng = random.Random(seed)
    gene = gene_entry["gene"]

    base_age = gene_entry["avg_age_at_dx_yrs"]
    age_dx = round(max(0.0, base_age + rng.gauss(0, 0.8)), 1)
    follow_up = round(rng.uniform(0.5, 14.0), 1)
    sex = rng.choice(["M", "F", "M", "F"])

    # PPK severity
    if gene in ("DSP", "JUP"):
        severity = rng.choice(["Moderate", "Moderate", "Severe"])
    elif gene in ("CTSC", "GJB2"):
        severity = rng.choice(["Moderate", "Moderate-Severe", "Severe", "Moderate"])
    elif gene in ("SERPINB7",):
        severity = rng.choice(["Mild", "Mild-Moderate", "Mild"])
    elif gene in ("SLURP1",):
        severity = rng.choice(["Moderate-Severe", "Severe", "Moderate"])
    else:
        severity = rng.choice(["Moderate", "Moderate-Severe", "Severe", "Moderate"])

    # Systemic features
    systemic = []
    if gene == "SLURP1":
        if rng.random() < 0.90:
            systemic.append("Hyperhidrosis")
        if rng.random() < 0.65:
            systemic.append("Pseudoainhum")
        if rng.random() < 0.70:
            systemic.append("Perioral_erythema")
    elif gene == "CTSC":
        if rng.random() < 0.98:
            systemic.append("Severe_periodontitis")
        if rng.random() < 0.90:
            systemic.append("Premature_tooth_loss")
        if rng.random() < 0.45:
            systemic.append("Recurrent_infections")
        if rng.random() < 0.35:
            systemic.append("Hepatosplenomegaly")
    elif gene == "GJB2":
        if rng.random() < 0.80:
            systemic.append("SNHL_sensorineural_HL")
        if rng.random() < 0.70:
            systemic.append("Starfish_knuckle_pads")
        if rng.random() < 0.55:
            systemic.append("Pseudoainhum")
    elif gene == "DSP":
        systemic.append("Woolly_hair")
        if rng.random() < 0.85:
            systemic.append("Dilated_cardiomyopathy_DCM")
        if rng.random() < 0.40:
            systemic.append("LV_dysfunction_ICD_indicated")
    elif gene == "JUP":
        systemic.append("Woolly_hair")
        if rng.random() < 0.90:
            systemic.append("ARVC")
        if rng.random() < 0.50:
            systemic.append("Ventricular_arrhythmia")
    elif gene == "SERPINB7":
        if rng.random() < 0.98:
            systemic.append("Aquagenic_wrinkling")
        if rng.random() < 0.75:
            systemic.append("Palmar_pits")
    elif gene == "LORICRIN":
        if rng.random() < 0.80:
            systemic.append("Pseudoainhum")
        if rng.random() < 0.85:
            systemic.append("Ichthyotic_veil")
    elif gene == "KRT9":
        if rng.random() < 0.65:
            systemic.append("Hyperhidrosis_palmar")

    # Transgredient
    transgredient = gene_entry["transgredient"]
    if gene == "SLURP1":
        transgredient = True
    elif gene == "CTSC" and rng.random() < 0.20:
        transgredient = True

    # Treatment
    treatment_map = {
        "KRT9": ["Urea40pct+Keratolytic", "Acitretin+Emollient", "Emollient_only"],
        "SLURP1": ["Acitretin+Keratolytic", "Keratolytic+Antiperspirant", "Acitretin+Surgery"],
        "CTSC": ["Antibiotics+Acitretin+Dental", "Antibiotics_only+Dental", "Acitretin+Keratolytic"],
        "GJB2": ["Acitretin+Keratolytic", "Keratolytic+Hearing_aid", "Surgery_pseudoainhum+Acitretin"],
        "DSP": ["Cardiac_ICD+Acitretin", "Heart_failure_therapy+Emollient", "Acitretin+Cardiac_monitor"],
        "JUP": ["ICD+Sports_restriction+Acitretin", "Antiarrhythmic+Keratolytic", "Acitretin+Holter"],
        "SERPINB7": ["Emollient+Antiperspirant", "Barrier_cream+Keratolytic", "Antiperspirant_only"],
        "LORICRIN": ["Acitretin+Keratolytic", "Surgery_pseudoainhum+Emollient", "Keratolytic_only"],
    }
    treatment = rng.choice(treatment_map.get(gene, ["Emollient"]))

    # Pathognomonic finding assigned per patient
    patho_map = {
        "KRT9": "Palm-only PPK (feet spared) + EHK on biopsy",
        "SLURP1": "Transgredient PPK dorsal extension + erythematous border + hyperhidrosis",
        "CTSC": "PPK + severe early-onset periodontitis + premature tooth loss",
        "GJB2": "Honeycomb PPK + starfish knuckle pads + pseudoainhum + SNHL",
        "DSP": "PPK + woolly hair + dilated cardiomyopathy (DCM, LEFT)",
        "JUP": "PPK + woolly hair + ARVC (RIGHT ventricular)",
        "SERPINB7": "Aquagenic wrinkling palms + palmar pits + punctate hyperkeratosis",
        "LORICRIN": "PPK + constriction bands + ichthyotic veil + NO hearing loss",
    }

    return {
        "patient_id": f"{gene}-{seed % 10000:04d}",
        "gene": gene,
        "ppk_type": gene_entry["ppk_type"],
        "locus": gene_entry["locus"],
        "inheritance": gene_entry["inheritance"].split(";")[0].strip(),
        "age_at_dx_yrs": age_dx,
        "follow_up_yrs": follow_up,
        "sex": sex,
        "severity": severity,
        "transgredient": transgredient,
        "systemic_features": systemic,
        "treatment": treatment,
        "cardiac_risk": gene_entry["cardiac_risk"],
        "dental_risk": gene_entry["dental_risk"],
        "pathognomonic_finding": patho_map.get(gene, "PPK"),
    }


def _build_cohort() -> list:
    patients = []
    for i, gene_entry in enumerate(PPK_GENES):
        base_seed = SEED_BASE + i
        for j in range(40):
            patients.append(_make_ppk_patient(gene_entry, base_seed * 100 + j))
    return patients


# ── Public API ──────────────────────────────────────────────────────────────────────────

def generate_overview() -> dict:
    """Aggregate statistics across all 8 PPK genes (320 patients)."""
    cohort = _build_cohort()
    gene_counts = {}
    type_counts = {}

    for p in cohort:
        g = p["gene"]
        gene_counts[g] = gene_counts.get(g, 0) + 1
        t = p["ppk_type"].split(" — ")[0][:40]
        type_counts[t] = type_counts.get(t, 0) + 1

    gene_summary = []
    for entry in PPK_GENES:
        g = entry["gene"]
        gene_summary.append({
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "ppk_type": entry["ppk_type"],
            "pathognomonic": entry["pathognomonic"],
            "n_patients": gene_counts.get(g, 0),
            "avg_age_at_dx_yrs": entry["avg_age_at_dx_yrs"],
            "cardiac_risk": entry["cardiac_risk"],
            "dental_risk": entry["dental_risk"],
            "transgredient": entry["transgredient"],
        })

    return {
        "title": "Hereditary-Palmoplantar-Keratoderma-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary PPK Atlas — "
            "KRT9 · SLURP1 · CTSC · GJB2 · DSP · JUP · SERPINB7 · LORICRIN — "
            "320 patients (8 × 40, seeds 2286-2293)"
        ),
        "n_patients": len(cohort),
        "n_genes": 8,
        "seed_range": "2286-2293",
        "ppk_categories": {
            "Epidermolytic PPK": "KRT9 (EPPK — palm-only, EHK biopsy)",
            "Transgredient PPK": "SLURP1 (Mal de Meleda — dorsal extension, hyperhidrosis)",
            "Syndromic-Dental PPK": "CTSC (Papillon-Lefèvre — PPK + periodontitis)",
            "Mutilating PPK": "GJB2 (Vohwinkel — honeycomb, SNHL, pseudoainhum)",
            "Cardiac PPK (DCM)": "DSP (Carvajal — woolly hair + dilated cardiomyopathy)",
            "Cardiac PPK (ARVC)": "JUP (Naxos — woolly hair + right ventricular ARVC)",
            "Aquagenic PPK": "SERPINB7 (Bothnian — aquagenic wrinkling, palmar pits)",
            "CE-Protein PPK": "LORICRIN (Loricrin keratoderma — pseudoainhum + ichthyotic veil, NO SNHL)",
        },
        "inheritance_map": {
            "KRT9": "AD", "SLURP1": "AR", "CTSC": "AR", "GJB2": "AD",
            "DSP": "AR", "JUP": "AR", "SERPINB7": "AR", "LORICRIN": "AD",
        },
        "key_clinical_pearls": [
            "KRT9: PALM-ONLY PPK (FEET COMPLETELY SPARED) PATHOGNOMONIC — KRT9 expressed in palms not soles; EHK on biopsy (suprabasal vacuolation) confirms; p.Arg163Trp most common European; most common hereditary PPK",
            "SLURP1: TRANSGREDIENT PPK EXTENDING TO DORSAL SURFACES OF HANDS/FEET + ERYTHEMATOUS BORDER PATHOGNOMONIC (Mal de Meleda); hyperhidrosis + pseudoainhum + perioral erythema; Mljet Island Adriatic founder; retinoids most effective",
            "CTSC: PPK + SEVERE EARLY-ONSET PERIODONTITIS WITH PREMATURE TOOTH LOSS PATHOGNOMONIC (Papillon-Lefèvre); PROPHYLACTIC ANTIBIOTICS (amoxicillin+metronidazole) MANDATORY before permanent teeth erupt; dental emergency",
            "GJB2: HONEYCOMB PPK + STARFISH-SHAPED KNUCKLE PADS + PSEUDOAINHUM PATHOGNOMONIC (Vohwinkel); SNHL in classic Vohwinkel (cochlear Cx26); KID syndrome (p.D50N): keratitis+ichthyosis+deafness — different phenotype same gene",
            "DSP: PPK + WOOLLY HAIR + DILATED CARDIOMYOPATHY (DCM) PATHOGNOMONIC (Carvajal); cardiac MRI + ICD if EF<35% mandatory; LEFT ventricular DCM distinguishes from JUP/Naxos (RIGHT ARVC); SCD in teenagers without ICD",
            "JUP: PPK + WOOLLY HAIR + ARVC (RIGHT VENTRICULAR CARDIOMYOPATHY) PATHOGNOMONIC (Naxos disease); ICD MANDATORY + sports restriction ABSOLUTE; RIGHT ventricle distinguishes from DSP/Carvajal (LEFT DCM); Greek Naxos Island founder",
            "SERPINB7: AQUAGENIC WRINKLING OF PALMS (exaggerated wrinkling within minutes of water exposure) PATHOGNOMONIC (Bothnian PPK); palmar pits; mild disease — no systemic involvement; Swedish Bothnian region founder p.Trp249Ter",
            "LORICRIN: PPK + CONSTRICTION BANDS (pseudoainhum) + ICHTHYOTIC VEIL (fine generalised ichthyosis) PATHOGNOMONIC; NO HEARING LOSS = KEY DDx from GJB2-Vohwinkel (which has SNHL); audiometry distinguishes; surgery for constriction bands",
        ],
        "gene_summary": gene_summary,
        "diagnostic_algorithm": {
            "Step_1": "Assess PPK extent: palm-only (KRT9-EPPK) vs palmoplantar (all others) vs transgredient dorsal (SLURP1) vs aquagenic wrinkling (SERPINB7)",
            "Step_2": "Systemic associations: (A) severe periodontitis + tooth loss → CTSC; (B) woolly hair + cardiac → DSP (DCM) or JUP (ARVC); (C) SNHL + knuckle pads + pseudoainhum → GJB2-Vohwinkel",
            "Step_3": "Hearing test (audiometry): SNHL → GJB2-Vohwinkel; NO SNHL + pseudoainhum + ichthyotic veil → LORICRIN keratoderma",
            "Step_4": "Cardiac workup (if woolly hair): ECG + echocardiography + cardiac MRI; RIGHT ARVC → JUP/Naxos; LEFT DCM → DSP/Carvajal",
            "Step_5": "Skin biopsy: EHK (suprabasal vacuolation) → KRT9-EPPK; non-epidermolytic → SLURP1/CTSC/GJB2/SERPINB7/LORICRIN; aquagenic water test → SERPINB7",
            "Step_6": "NGS PPK gene panel (KRT9, SLURP1, CTSC, GJB2, DSP, JUP, SERPINB7, LORICRIN + broader panel): confirms gene; dictates management (cardiac ICD, dental antibiotics, surgical pseudoainhum release)",
        },
        "cardiac_emergency_genes": ["DSP", "JUP"],
        "dental_emergency_genes": ["CTSC"],
        "pseudoainhum_genes": ["SLURP1", "GJB2", "LORICRIN"],
        "transgredient_genes": ["SLURP1"],
        "no_systemic_genes": ["KRT9", "SERPINB7"],
        "type_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10]),
    }


def generate_breakdown() -> dict:
    """Per-gene PPK profiles across all 8 genes."""
    cohort = _build_cohort()
    by_gene = {}
    for p in cohort:
        by_gene.setdefault(p["gene"], []).append(p)

    gene_breakdown = {}
    for entry in PPK_GENES:
        g = entry["gene"]
        pts = by_gene.get(g, [])

        severity_dist = {}
        treatment_dist = {}
        systemic_dist = {}
        for p in pts:
            sv = p["severity"]
            severity_dist[sv] = severity_dist.get(sv, 0) + 1
            tx = p["treatment"]
            treatment_dist[tx] = treatment_dist.get(tx, 0) + 1
            for sf in p["systemic_features"]:
                systemic_dist[sf] = systemic_dist.get(sf, 0) + 1

        avg_age = round(sum(p["age_at_dx_yrs"] for p in pts) / len(pts), 2) if pts else 0
        avg_fu = round(sum(p["follow_up_yrs"] for p in pts) / len(pts), 1) if pts else 0
        transgredient_n = sum(1 for p in pts if p["transgredient"])
        systemic_n = sum(1 for p in pts if p["systemic_features"])
        severe_n = sum(1 for p in pts if "Severe" in p["severity"])

        gene_breakdown[g] = {
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "ppk_type": entry["ppk_type"],
            "n_patients": len(pts),
            "avg_age_at_dx_yrs": avg_age,
            "avg_follow_up_yrs": avg_fu,
            "transgredient_pct": round(transgredient_n / len(pts) * 100, 1) if pts else 0,
            "systemic_pct": round(systemic_n / len(pts) * 100, 1) if pts else 0,
            "severe_pct": round(severe_n / len(pts) * 100, 1) if pts else 0,
            "pathognomonic": entry["pathognomonic"],
            "treatment_highlight": entry["treatment_highlight"],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:700],
            "monitoring": entry["monitoring"],
            "key_ddx": entry["key_ddx"],
            "severity_distribution": dict(sorted(severity_dist.items(), key=lambda x: -x[1])),
            "systemic_distribution": dict(sorted(systemic_dist.items(), key=lambda x: -x[1])),
            "treatment_distribution": dict(sorted(treatment_dist.items(), key=lambda x: -x[1])),
            "cardiac_risk": entry["cardiac_risk"],
            "dental_risk": entry["dental_risk"],
            "patients_sample": pts[:5],
        }

    return {
        "title": "Hereditary-Palmoplantar-Keratoderma-Atlas — Per-Gene Breakdown",
        "n_genes": 8,
        "n_patients": len(cohort),
        "gene_breakdown": gene_breakdown,
        "clinical_emergency_flags": [
            "DSP/Carvajal CARDIAC: PPK + WOOLLY HAIR + DILATED CARDIOMYOPATHY — CARDIAC MRI + ICD if LVEF <35% MANDATORY; sports restriction absolute; sudden cardiac death risk teenagers; LEFT ventricular DCM; cardiology + cardiac genetics from diagnosis",
            "JUP/Naxos CARDIAC: PPK + WOOLLY HAIR + ARVC — ICD MANDATORY if ventricular arrhythmia/significant RV dysfunction; SPORTS RESTRICTION ABSOLUTE (exercise triggers SCD in ARVC); RIGHT ventricular involvement; electrophysiology urgent",
            "CTSC/PLS DENTAL: PPK + SEVERE PERIODONTITIS — PROPHYLACTIC ANTIBIOTICS (amoxicillin+metronidazole) START BEFORE PERMANENT TEETH ERUPT (age 5-6yr) = MANDATORY; paediatric periodontist immediate; ALL permanent teeth lost by 14yr without intervention",
            "GJB2/Vohwinkel PSEUDOAINHUM: DIGIT CONSTRICTION BANDS — EMERGENCY SURGICAL RELEASE if digit ischaemia (cyanosis/necrosis); same-day hand surgery; monitor digit circumference at every visit",
            "LORICRIN/Vohwinkel-variant PSEUDOAINHUM: CONSTRICTION BANDS WITHOUT SNHL — surgical release mandatory if vascular compromise; audiometry confirms NO SNHL (DDx from GJB2-Vohwinkel with SNHL)",
            "SLURP1/Mal-de-Meleda PSEUDOAINHUM: CONSTRICTING FIBROUS BANDS — surgical release if band causes vascular compromise; monitor digits closely for progressive constriction",
        ],
    }


def generate_definitions() -> dict:
    """Glossary of PPK biology, gene definitions, treatments, and diagnostic tests."""
    return {
        "title": "Hereditary-Palmoplantar-Keratoderma-Atlas — Definitions & Glossary",
        "gene_entries": {
            entry["gene"]: {
                "full_name": entry["gene"],
                "protein_size": entry["protein_size"],
                "locus": entry["locus"],
                "inheritance": entry["inheritance"],
                "disease_name": entry["ppk_type"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"],
                "key_ddx": entry["key_ddx"],
                "key_features": entry["key_features"],
                "monitoring": entry["monitoring"],
            }
            for entry in PPK_GENES
        },
        "ppk_biology_glossary": {
            "Palmoplantar keratoderma (PPK)": "Diffuse or focal hyperkeratosis of the palms and soles; classified as (1) hereditary (genetic) or acquired; hereditary forms divided by histology (epidermolytic/non-epidermolytic), distribution (diffuse/focal/striate), extent (palmoplantar only vs transgredient dorsal extension), and associated features (syndromic vs non-syndromic)",
            "Cornified envelope (CE)": "Insoluble protein shell replacing plasma membrane in terminally differentiated corneocytes; cross-linked by transglutaminases (TGase-1/TGM1, TGase-3); major components: loricrin (70-80%), involucrin, SPRRs, elafin; LORICRIN mutations → defective CE in skin",
            "Epidermolytic hyperkeratosis (EHK)": "Histopathological pattern: suprabasal vacuolation + perinuclear halos + epidermolysis in granular layer + compact hyperkeratosis; PATHOGNOMONIC for KRT9-EPPK (palms) and KRT1/KRT10-EI; confirms keratin mutation; distinguishes epidermolytic from non-epidermolytic PPK types",
            "Transgredient PPK": "Palmoplantar keratoderma that extends beyond the palmoplantar margins to adjacent dorsal surfaces of hands and feet; NOT a feature of most PPK types; PATHOGNOMONIC for SLURP1-Mal de Meleda when combined with erythematous border; less commonly occurs in CTSC-PLS",
            "Pseudoainhum": "Constricting fibrous band encircling a digit; can cause progressive ischaemia and digit autoamputation if untreated; occurs in SLURP1, GJB2-Vohwinkel, LORICRIN keratoderma; named after tropical ainhum disease (Treponema pallidum infection) which it mimics; surgical release is mandatory treatment",
            "Desmosome": "Cell junction anchoring adjacent epithelial/cardiac cells; components: desmosomal cadherins (desmoglein/desmocollin), armadillo proteins (plakoglobin/JUP, plakophilin), plakin proteins (desmoplakin/DSP), intermediate filaments; disrupted in Naxos (JUP) and Carvajal (DSP) syndromes → PPK + cardiac disease",
            "Connexin-26 (GJB2)": "Gap junction protein forming hexameric hemichannels; critical in epidermis (cell-cell communication, differentiation) and cochlea (endocochlear potential maintenance, K+ recycling); AD missense → Vohwinkel syndrome (SNHL+PPK); biallelic LOF → DFNB1 (isolated SNHL, no skin); p.D50N → KID syndrome",
            "Cathepsin C (CTSC)": "Lysosomal cysteine protease (dipeptidyl peptidase I, DPPI); activates neutrophil serine proteases (elastase, cathepsin G, proteinase 3, NSP4) by cleaving propeptides; deficiency → impaired neutrophil killing → aggressive periodontitis; also expressed in skin; biallelic LOF → Papillon-Lefèvre + Haim-Munk syndromes",
            "Aquagenic wrinkling": "Exaggerated rapid wrinkling of palmar skin within minutes of water exposure; PATHOGNOMONIC for SERPINB7-Bothnian PPK; also occurs in cystic fibrosis (CFTR) carriers — sweat chloride test distinguishes; mechanism: defective SC barrier allows water entry → osmotic swelling",
        },
        "ppk_type_glossary": {
            "Epidermolytic PPK (EPPK)": "AD hereditary PPK with EHK on biopsy; classic form = KRT9 (palmar only, feet spared) and KRT1 (palmoplantar both); EHK = suprabasal vacuolation + epidermolysis + compact hyperkeratosis; distinguishes from non-epidermolytic forms; retinoids effective; biopsy mandatory at diagnosis",
            "Transgredient PPK (Mal de Meleda)": "AR PPK (SLURP1) extending to dorsal surfaces with erythematous border; hyperhidrosis; pseudoainhum; perioral erythema; named after Mljet (Meleda) island Croatia; retinoids most effective; pseudoainhum surgical release if needed; distinguished by dorsal extension pattern",
            "Syndromic PPK — Dental (PLS)": "AR PPK (CTSC) + severe early-onset periodontitis + premature tooth loss; prophylactic antibiotics mandatory before permanent teeth erupt; Haim-Munk variant: same CTSC gene + arachnodactyly + acro-osteolysis; neutrophil serine protease deficiency underlies immunodeficiency",
            "Mutilating PPK (Vohwinkel)": "AD PPK (GJB2) — honeycomb PPK + SNHL + starfish knuckle pads + pseudoainhum; connexin-26 channel dysfunction; KID syndrome (p.D50N): keratitis+ichthyosis+deafness (different phenotype same gene); cochlear implant if profound SNHL; surgical pseudoainhum release",
            "Cardiac PPK (Carvajal/Naxos)": "AR PPK (DSP-Carvajal: LEFT DCM; JUP-Naxos: RIGHT ARVC) + woolly hair; desmosomal disease; ICD mandatory; sports restriction absolute; cardiac MRI diagnostic; SCD in teenagers without intervention; KEY DDx: LEFT vs RIGHT ventricle",
            "Aquagenic PPK (Bothnian)": "AR PPK (SERPINB7) — aquagenic wrinkling of palms + palmar pits + punctate hyperkeratosis; mild disease; Swedish Bothnian founder p.Trp249Ter; no systemic features; emollients + antiperspirant; distinguish from CF-related aquagenic wrinkling (CFTR)",
            "Loricrin keratoderma (Vohwinkel variant)": "AD PPK (LORICRIN frameshift) — PPK + pseudoainhum + fine generalised ichthyotic veil; NO SNHL; KEY DDx from GJB2-Vohwinkel (which has SNHL); audiometry distinguishes; major CE protein (70-80%); nuclear accumulation of frameshifted loricrin",
        },
        "treatment_glossary": {
            "Acitretin (systemic retinoid for PPK)": "Oral aromatic retinoid 0.3-0.5 mg/kg/day; first-line systemic for severe hereditary PPK (KRT9, SLURP1, CTSC, GJB2-Vohwinkel, LORICRIN); reduces hyperkeratosis; teratogenic (Category X) — contraception during + 3yr after (females); monitor LFTs, triglycerides, DEXA; useful for transgredient PPK and mutilating forms",
            "Prophylactic antibiotics (CTSC-PLS)": "Amoxicillin (250-500mg/day) + metronidazole (200-400mg/day) — START before permanent teeth erupt (age 5-6yr) in all CTSC/PLS patients; destroys periodontal pathogens (A. actinomycetemcomitans) before established periodontitis; 3+ years continuous during teeth eruption; combine with chlorhexidine 0.2% rinse + quarterly scaling",
            "ICD (Implantable Cardioverter-Defibrillator)": "Mandatory for DSP/Carvajal patients with LVEF <35% or significant LV dysfunction; mandatory for JUP/Naxos patients with documented VT/VF or significant ARVC; primary prevention discussion for all cardiac PPK patients; prevents sudden cardiac death in teenagers/young adults with cardiac desmosomal disease",
            "Surgical pseudoainhum release": "Z-plasty or simple division of constricting fibrous bands around digits; emergency procedure if acute vascular compromise (cyanosis/necrosis); hand surgery + dermatology co-management; performed in SLURP1, GJB2-Vohwinkel, LORICRIN keratoderma when bands cause constriction; monitor digit circumference + perfusion quarterly",
            "Keratolytic emollients (urea/salicylic acid)": "Urea 20-50% cream: humectant + keratolytic; reduces SC thickness; urea 40-50% for PPK specifically; lactic acid 10-12%: alpha-hydroxy acid keratolytic; salicylic acid 5-20%: corneolytic; soak-and-smear technique (warm water 15-20 min soak then immediate emollient application) maximises keratolytic efficacy; first-line for all hereditary PPK",
            "Antiperspirant (aluminium chloride)": "20% aluminium chloride in ethanol (Drysol): most effective topical antiperspirant; reduces palmar hyperhidrosis in SLURP1-Mal de Meleda and KRT9-EPPK; apply to dry palms at night 2-3×/week; reduces maceration + malodor + aquagenic wrinkling in SERPINB7; botulinum toxin injection as alternative for severe hyperhidrosis",
        },
        "diagnostic_tests": {
            "Skin biopsy (EHK histology)": "H&E biopsy from palm: EHK (suprabasal vacuolation + perinuclear halos + epidermolysis) = PATHOGNOMONIC for KRT9-EPPK; distinguishes from non-epidermolytic PPK (SLURP1, CTSC, GJB2, DSP, JUP, SERPINB7, LORICRIN — all non-epidermolytic); mandatory at diagnosis to classify PPK type before gene panel; electron microscopy adds value for SERPINB7 (SC structural defect)",
            "Audiometry (PPK SNHL assessment)": "Pure-tone audiometry + tympanometry at PPK diagnosis: SNHL → GJB2-Vohwinkel; NORMAL hearing → LORICRIN keratoderma (PPK + pseudoainhum without SNHL) — SINGLE MOST IMPORTANT DDx test for pseudoainhum + PPK; baseline then annually; cochlear implant candidacy assessment if profound SNHL",
            "Cardiac MRI with LGE (cardiac PPK)": "Mandatory for all DSP/JUP patients; late gadolinium enhancement (LGE): DSP/Carvajal → subepicardial LGE in LV (early DCM marker); JUP/Naxos → RV fatty infiltration + LGE (ARVC); RIGHT vs LEFT involvement distinguishes Naxos from Carvajal; annual CMR in affected patients until stable; detects cardiac involvement before echocardiographic dysfunction",
            "Aquagenic challenge test (SERPINB7)": "Submerse hands in water for 3-5 minutes; aquagenic wrinkling developing within 3 minutes = POSITIVE; PATHOGNOMONIC for SERPINB7-Bothnian PPK; also positive in CF/CFTR carriers — combine with sweat chloride (iontophoresis) to exclude CF; simple bedside test; no equipment needed",
            "NGS PPK gene panel": "Panel covering: KRT9 (EPPK), KRT1 (EPPK-feet), SLURP1 (Mal de Meleda), CTSC (PLS), GJB2 (Vohwinkel/KID), DSP (Carvajal), JUP (Naxos), SERPINB7 (Bothnian), LORICRIN, AQP5 (focal), KRT16/17 (striate) + broader; essential for: distinguishing Vohwinkel (GJB2) from Loricrin keratoderma (LORICRIN); identifying cardiac PPK (DSP/JUP — immediate cardiology referral); confirming CTSC (dental emergency); prenatal diagnosis",
            "Dental panoramic X-ray (OPG — CTSC/PLS)": "Annual OPG from age 3yr in CTSC/PLS diagnosis; quantifies periodontal bone loss around primary + permanent teeth; identifies A. actinomycetemcomitans subgingival burden; guides antibiotic choice + surgical periodontal intervention; tracks response to prophylactic antibiotic therapy; baseline before permanent teeth erupt mandatory",
        },
    }


# Aliases for api_backend.py compatibility (uses overview/breakdown/definitions)
def overview() -> dict:
    return generate_overview()


def breakdown() -> dict:
    return generate_breakdown()


def definitions() -> dict:
    return generate_definitions()


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"  Title: {ov['title']}")
    print(f"  Patients: {ov['n_patients']}  |  Genes: {ov['n_genes']}")
    print(f"  Seeds: {ov['seed_range']}")
    print("  Key pearls:")
    for p in ov["key_clinical_pearls"][:4]:
        print(f"    - {p[:100]}")

    print("\n=== BREAKDOWN (gene counts) ===")
    bk = generate_breakdown()
    for g, info in bk["gene_breakdown"].items():
        print(f"  {g}: {info['n_patients']} pts | Type: {info['ppk_type'][:60]}")
