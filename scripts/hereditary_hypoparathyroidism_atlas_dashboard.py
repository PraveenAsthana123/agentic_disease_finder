#!/usr/bin/env python3
"""Hereditary-Hypoparathyroidism-Atlas — Complete 8-Gene Parathyroid / Calcium-Homeostasis Atlas
GCM2    (glial cells missing 2 transcription factor; 495 aa; 6p24.2; AD/AR;
         most common familial isolated hypoparathyroidism;
         parathyroid gland agenesis/hypoplasia; no extraparathyroid features;
         AD haploinsufficiency OR AR biallelic;
         seed SEED_BASE+0) ·
PTH     (parathyroid hormone; 115 aa preproPTH → 84 aa mature; 11p15.3; AR/AD;
         direct PTH-gene mutations → preproPTH misprocessing/non-secretion;
         isolated hypoparathyroidism; very rare; PTH undetectable;
         seed SEED_BASE+1) ·
CASR    (calcium-sensing receptor; 1078 aa; 3q13.3; AD GOF;
         activating mutations → receptor hypersensitive to Ca2+ → PTH suppressed
         at lower Ca2+ → Autosomal Dominant Hypocalcemia type 1 (ADH1);
         PATHOGNOMONIC: relative hypercalciuria despite hypocalcemia (renal CaSR activated);
         RISK: calcium + calcitriol → severe nephrocalcinosis;
         cinacalcet ABSOLUTE CI (sensitiser → worsens hypocalcemia);
         seed SEED_BASE+2) ·
GNA11   (Gα11 G-protein alpha subunit; 359 aa; 19p13.3; AD GOF;
         downstream of CaSR — GOF → constitutive PTH suppression;
         Autosomal Dominant Hypocalcemia type 2 (ADH2);
         cinacalcet ineffective (acts upstream of Gα11);
         seed SEED_BASE+3) ·
TBCE    (tubulin-specific chaperone E; 527 aa; 1q42.3; AR;
         HRD syndrome = Hypoparathyroidism + Retardation + Dysmorphism;
         also Sanjad-Sakati syndrome; Arab/Middle-Eastern founder IVS1-2A>G;
         also causes Kenny-Caffey syndrome type 1 (KCS1) — with cortical bone thickening;
         intellectual disability distinguishes HRD from KCS2 (FAM111A);
         seed SEED_BASE+4) ·
FAM111A (FAM111 protease family member A; 611 aa; 11q13.1; AD GOF;
         Kenny-Caffey syndrome type 2 (KCS2) — GOF serine-protease activity;
         short stature + cortical bone thickening + hypoparathyroidism;
         NO intellectual disability (vs TBCE/HRD — key discriminator);
         seed SEED_BASE+5) ·
GATA3   (GATA-binding protein 3; 444 aa; 10p14; AD LOF;
         HDR syndrome = Hypoparathyroidism + Deafness (SNHL bilateral) + Renal anomalies;
         also called Barakat syndrome; haploinsufficiency;
         bilateral SNHL often FIRST/most prominent feature;
         renal anomalies: dysplasia/aplasia/horseshoe/VUR;
         seed SEED_BASE+6) ·
SOX3    (SRY-box transcription factor 3; 446 aa; Xq27.1; XL;
         X-linked hypoparathyroidism — males ONLY clinically affected;
         insertion/deletion in SOX3 regulatory region → parathyroid aplasia/ectopia;
         females carriers (normal or very mild);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2926–2933)
"""
import random

SEED_BASE = 2926

ATLAS_GENES = [
    {
        "gene": "GCM2",
        "protein": (
            "GCM2 -- 6p24.2 AD/AR -- 495aa -- Glial-Cells-Missing-2-TF-"
            "47kDa-Parathyroid-Master-Transcription-Factor-"
            "Most-Common-Familial-Isolated-Hypoparathyroidism-"
            "OMIM-Gene-603716-Disease-OMIM-146200-307700"
        ),
        "locus": "6p24.2",
        "protein_size": (
            "495 aa / 47 kDa (GCM2 — glial cells missing 2; "
            "zinc-coordinating transcription factor; GCM domain (aa 1-150); "
            "FUNCTION: master regulator of parathyroid gland development; "
            "  required for parathyroid gland specification from 3rd/4th pharyngeal pouch; "
            "  LOF → parathyroid gland agenesis or severe hypoplasia; "
            "  no other organ expression relevant (unlike GATA3/TBCE) → ISOLATED HP; "
            "AD: single LOF allele → haploinsufficiency → partial parathyroid loss; "
            "AR: biallelic → complete parathyroid agenesis → profound hypoparathyroidism; "
            "AD mutation spectrum: missense in DNA-binding GCM domain (R47L, R110H) most common; "
            "AR: truncating / homozygous; "
            "encoded 6p24.2; "
            "MOST COMMON GENE for familial isolated hypoparathyroidism worldwide"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) HAPLOINSUFFICIENCY or AUTOSOMAL RECESSIVE (AR) — GCM2: "
            "  AD (LOF, haploinsufficiency): one pathogenic allele → partial reduction in parathyroid mass; "
            "    Incomplete penetrance ~60-70%; variable expressivity (from asymptomatic to severe CH); "
            "    Family history: parent often mildly affected or biochemically only; "
            "  AR (biallelic): complete loss → severe neonatal/infantile hypoparathyroidism; "
            "    Consanguinity a risk factor; "
            "CLINICAL FEATURES: "
            "    ISOLATED hypoparathyroidism — NO deafness, NO renal anomalies, NO dysmorphic features; "
            "    Low Ca2+, elevated phosphate, low/undetectable PTH; "
            "    Tetany, laryngospasm, carpopedal spasm, seizures; "
            "    Chvostek sign (facial twitch on tapping facial nerve) and Trousseau sign (carpal spasm with BP cuff); "
            "DIAGNOSIS: "
            "    Serum: hypocalcaemia + hyperphosphataemia + low/undetectable PTH; "
            "    Urine: hypocalciuria (normal renal CaSR → calcium reabsorption intact); "
            "    Imaging: parathyroid scan shows absent/small glands; "
            "    Gene panel: GCM2 first in isolated familial HP (most common gene)"
        ),
        "disease_category": (
            "ISOLATED FAMILIAL HYPOPARATHYROIDISM — PARATHYROID AGENESIS/HYPOPLASIA: "
            "  ONSET: "
            "    AR biallelic: neonatal (day 1-5) — profound hypocalcaemia; "
            "    AD haploinsufficiency: variable — neonatal, childhood, or adult; "
            "  CALCIUM HOMEOSTASIS FAILURE: "
            "    PTH normally: stimulates bone resorption (raises Ca2+) + renal Ca reabsorption + "
            "      1α-hydroxylase (→ 1,25(OH)2D → intestinal Ca absorption) + renal phosphate excretion; "
            "    GCM2 LOF: PTH absent → hypocalcaemia + hyperphosphataemia + low 1,25(OH)2D; "
            "  URINE CALCIUM: LOW (renal CaSR intact — normal Ca reabsorption in absence of PTH-driven hypercalciuria); "
            "  CRITICAL DISTINCTION from ADH1/2 (CASR/GNA11): "
            "    GCM2/PTH: LOW urine Ca; "
            "    CASR/GNA11 ADH: HIGH urine Ca despite hypocalcaemia — key differential; "
            "  TREATMENT: "
            "    Calcium + active vitamin D (calcitriol); "
            "    Recombinant PTH (1-34 = teriparatide or 1-84 = Natpara) — SUPERIOR to calcitriol alone; "
            "    Monitor: serum Ca, 24h urine Ca (target <6.25 mmol/24h), eGFR; "
            "    Risk of nephrocalcinosis with over-treatment → keep serum Ca in lower normal range"
        ),
        "disease_pathway": (
            "GCM2 LOF → PARATHYROID AGENESIS/HYPOPLASIA → PTH DEFICIENCY → HYPOCALCAEMIA: "
            "  Normal: PTH → PTH1R (bone + kidney) → "
            "    Bone: osteoclast activation → Ca2+/PO4 release; "
            "    Kidney DCT: CaSR + PTH1R → Ca reabsorption; "
            "    Kidney PCT: PTH1R → phosphaturia (FePO4 increased); "
            "    Kidney: 25(OH)D → 1α-OHase → 1,25(OH)2D → intestinal Ca + PO4 absorption; "
            "  GCM2 LOF: "
            "    PTH absent → Ca2+ falls → phosphate rises (no PTH-mediated phosphaturia); "
            "    1,25(OH)2D low (no PTH drive for 1α-hydroxylase); "
            "    Urine Ca LOW (kidney CaSR intact, reabsorbs Ca in absence of hypercalciuric drive); "
            "  KEY FORMULA: "
            "    Hypocalcaemia + hyperphosphataemia + LOW PTH + LOW 1,25(OH)2D + LOW urine Ca "
            "    = HYPOPARATHYROIDISM (not pseudohypoparathyroidism = PTH resistance where PTH is HIGH)"
        ),
    },
    {
        "gene": "PTH",
        "protein": (
            "PTH -- 11p15.3 AR/AD -- 115aa-preproPTH/84aa-mature -- "
            "Parathyroid-Hormone-9.4kDa-Mature-Direct-PTH-Gene-Mutation-"
            "Signal-Peptide-Prepro-Region-LOF-Non-Secretion-"
            "OMIM-Gene-168450-Disease-OMIM-146200"
        ),
        "locus": "11p15.3",
        "protein_size": (
            "115 aa preproPTH → 84 aa mature PTH / 9.4 kDa (mature form; "
            "STRUCTURE: signal peptide (aa 1-25) → propeptide (aa 26-31) → mature PTH (aa 32-115 = 1-84); "
            "PROCESSING: "
            "  Ribosome: 115 aa preproPTH synthesised; "
            "  ER: signal peptide cleaved → 90 aa proPTH; "
            "  Golgi: propeptide cleaved → 84 aa mature PTH stored in secretory granules; "
            "  Release: on hypocalcaemia signal (CaSR off → exocytosis); "
            "PTH GENE MUTATIONS causing isolated HP: "
            "  Signal peptide mutations: impair ER translocation → prePTH retained in cytoplasm → degraded; "
            "  Propeptide mutations: impair processing → inactive proPTH secreted; "
            "  Mature PTH mutations: direct loss of function; "
            "  AR (most) or AD (dominant negative signal peptide mutations); "
            "encoded 11p15.3; very rare cause of isolated HP (GCM2 far more common)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) biallelic OR AUTOSOMAL DOMINANT (AD) dominant-negative — PTH: "
            "  AR: biallelic PTH mutations → no PTH → severe neonatal hypoparathyroidism; "
            "    Most common: signal peptide mutation (Cys18Arg) — impairs ER translocation; "
            "  AD: dominant-negative signal peptide mutation → mutant preproPTH traps wild-type → reduced secretion; "
            "    Less severe than biallelic AR; "
            "CLINICAL: "
            "    Isolated hypoparathyroidism — no extraparathyroid features; "
            "    Neonatal or early childhood onset (biallelic); "
            "    Serum PTH: undetectable or extremely low; "
            "    Parathyroid glands: PRESENT on imaging (contrast with GCM2 — glands absent); "
            "DIAGNOSIS: "
            "    Same biochemistry as GCM2 (low Ca, high PO4, low PTH); "
            "    Distinction: PTH gene → glands present but non-functional; "
            "    Confirmed by PTH gene sequencing after GCM2 excluded; "
            "    Functional assay: PTH mRNA/protein expression (research setting)"
        ),
        "disease_category": (
            "ISOLATED HYPOPARATHYROIDISM — DIRECT PTH GENE DEFECT: "
            "  MECHANISM: "
            "    Signal peptide mutations: preproPTH misfolds/mislocalises → no mature PTH processed; "
            "    Dominant negative: mutant preproPTH aggregates with WT preproPTH in ER → ER stress → apoptosis; "
            "  PARATHYROID GLANDS: present (contrast with GCM2) but secrete non-functional PTH or none; "
            "  ONSET: neonatal (biallelic) or early childhood; "
            "  TREATMENT: identical to all forms of hypoparathyroidism: "
            "    Calcium supplementation + calcitriol; "
            "    Recombinant PTH 1-84 (Natpara / PTH1-84): FDA-approved for chronic HP in adults; "
            "    Recombinant PTH 1-34 (teriparatide): off-label for HP; superior urine Ca profile vs calcitriol; "
            "  URINE Ca: low (same as GCM2 — renal CaSR intact); "
            "  GENETIC COUNSELLING: AR → 25% recurrence; AD → 50% recurrence; "
            "  DIAGNOSIS ORDER: GCM2 → PTH → CASR (if PTH low + urine Ca low = exclude ADH first)"
        ),
        "disease_pathway": (
            "PTH GENE MUTATION → NO/DYSFUNCTIONAL PTH SECRETED: "
            "  Pre-pro-PTH biogenesis failure: "
            "    Signal peptide mutation (aa 1-25): impairs signal recognition particle (SRP) binding; "
            "      preproPTH not directed to ER → cytoplasmic degradation by proteasome; "
            "    Propeptide mutation: proPTH cleavage defective → inactive proPTH in circulation; "
            "    Mature PTH coding mutation: unstable or receptor-binding-defective mature PTH; "
            "  Downstream: identical to GCM2 — PTH absent → hypocalcaemia + hyperphosphataemia + low 1,25(OH)2D; "
            "  KEY DISTINCTION: "
            "    PTH glands PRESENT (on 4D-CT or 99mTc-MIBI) — dysfunctional; "
            "    GCM2 glands ABSENT — agenesis; "
            "    Both give low/undetectable PTH in circulation"
        ),
    },
    {
        "gene": "CASR",
        "protein": (
            "CASR -- 3q13.3 AD-GOF -- 1078aa -- Calcium-Sensing-Receptor-"
            "120kDa-GPCR-Class-C-7TM-Activating-Mutation-ADH1-"
            "Autosomal-Dominant-Hypocalcemia-Type-1-"
            "PATHOGNOMONIC-Relative-Hypercalciuria-Despite-Hypocalcemia-"
            "CINACALCET-ABSOLUTE-CI-NEPHROCALCINOSIS-Risk-Calcium-Calcitriol-"
            "OMIM-Gene-601199-Disease-OMIM-601198"
        ),
        "locus": "3q13.3",
        "protein_size": (
            "1078 aa / 120 kDa (CaSR — calcium-sensing receptor; "
            "GPCR Class C; homodimer; "
            "DOMAINS: "
            "  Large extracellular Venus Flytrap (VFT) domain (aa 1-599) — Ca2+ binding site; "
            "  Cysteine-rich domain (aa 600-640) — signal transduction; "
            "  7 transmembrane helices (aa 641-862); "
            "  Intracellular C-tail (aa 863-1078); "
            "NORMAL FUNCTION: "
            "  Parathyroid: CaSR senses extracellular Ca2+ → if elevated → Gαq → PLC → IP3 → intracellular Ca2+ → PTH secretion INHIBITED; "
            "  Kidney (TAL/DCT): CaSR senses tubular Ca2+ → if elevated → reduces Ca reabsorption → calciuresis; "
            "  Key concept: CaSR is the Ca2+ set-point governor for both PTH suppression and renal Ca handling; "
            "ADH1 GOF MECHANISM: "
            "  Activating mutation shifts CaSR Ca2+ sensitivity left → receptor activated at LOWER Ca2+; "
            "  Parathyroid: PTH suppressed even at normal/low Ca2+ → hypocalcaemia; "
            "  Kidney: renal CaSR also activated → Ca2+ NOT reabsorbed even at low Ca2+ → "
            "    RELATIVE HYPERCALCIURIA (24h urine Ca inappropriately high for serum Ca level) — PATHOGNOMONIC; "
            "encoded 3q13.3"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GAIN-OF-FUNCTION (GOF) — CASR ADH1: "
            "  Single activating allele → haploinsufficiency-like effect (GOF dominant over WT); "
            "  Mutation hotspots: VFT domain (aa 1-599) most common — R185Q, R185W, C129S; "
            "  De novo mutations common (~20%); "
            "  Penetrance: high (~90%); variable severity; "
            "CLINICAL FEATURES: "
            "    Hypocalcaemia (usually mild-moderate: Ca 1.9-2.1 mmol/L); "
            "    Hyperphosphataemia (mild); "
            "    RELATIVE HYPERCALCIURIA: 24h urine Ca inappropriately high relative to serum Ca; "
            "      Normal CaSR: hypocalcaemia → low urine Ca (kidney saves Ca); "
            "      ADH1: hypocalcaemia + HIGH urine Ca (renal CaSR activated → cannot retain Ca); "
            "    Serum Mg: may be LOW (CaSR also governs Mg reabsorption in TAL); "
            "    PTH: LOW but usually DETECTABLE (unlike GCM2/PTH — may be 1-20 pg/mL); "
            "CRITICAL TREATMENT RULES: "
            "    Cinacalcet ABSOLUTE CI (positive allosteric modulator of CaSR → further activates GOF → worsens hypocalcaemia); "
            "    Calcium + calcitriol: USE WITH CAUTION — risk of nephrocalcinosis/nephrolithiasis because "
            "      renal CaSR still activated → any Ca supplement → calciuria → kidney damage; "
            "    Recombinant PTH 1-34 or 1-84 PREFERRED — bypasses CaSR mechanism, restores renal Ca reabsorption; "
            "    Target: serum Ca 1.9-2.1 mmol/L (lower than usual HP target) to minimise calciuria"
        ),
        "disease_category": (
            "AUTOSOMAL DOMINANT HYPOCALCEMIA TYPE 1 (ADH1) — ACTIVATING CASR: "
            "  DISTINGUISHING FEATURES vs other HP causes: "
            "    Relative hypercalciuria: 24h urine Ca >6.25 mmol/24h despite serum Ca <2.0 mmol/L — PATHOGNOMONIC; "
            "      (All other HP causes → LOW urine Ca as renal CaSR not activated); "
            "    Low-normal serum Mg (CaSR in Henle → Mg reabsorption also reduced); "
            "    PTH: detectable but inappropriately low for serum Ca (usually 1-20 pg/mL); "
            "  DIAGNOSIS ALGORITHM: "
            "    Low Ca + low PTH → measure 24h urine Ca; "
            "    If ELEVATED urine Ca: → CASR/GNA11 ADH — gene panel; "
            "    If LOW urine Ca: → GCM2/PTH/TBCE/GATA3/FAM111A/SOX3; "
            "  NEPHROCALCINOSIS RISK: "
            "    Calcium + calcitriol treatment: urine Ca rises further → medullary nephrocalcinosis; "
            "    Monitor: renal ultrasound + 24h urine Ca every 6-12 months; "
            "    eGFR trending: chronic kidney disease in undertreated or overtreated ADH1; "
            "  TREATMENT: "
            "    rPTH (Natpara 1-84 or Forteo 1-34): restores renal Ca reabsorption → lower urine Ca vs calcitriol; "
            "    Keep serum Ca low-normal (1.9-2.1 mmol/L) in ADH1 — safe zone to limit nephrocalcinosis; "
            "    Thiazide diuretic + low-sodium diet: reduce urine Ca as adjunct"
        ),
        "disease_pathway": (
            "CASR GOF → CONSTITUTIVE PTH SUPPRESSION + RENAL Ca WASTING: "
            "  CaSR GOF: VFT domain mutation → Ca2+-binding site hypersensitive → "
            "    Activated at lower serum Ca → Gαq → PLC-β → IP3 → Ca2+ release → PTH exocytosis blocked; "
            "  Parathyroid: PTH suppressed at Ca 1.9-2.1 mmol/L (normal 2.1-2.6 mmol/L); "
            "  Kidney: "
            "    TAL CaSR GOF → ROMK (K+ channel) + claudin-16/19 (paracellular Ca/Mg) → "
            "      reduced TAL Ca2+ reabsorption; "
            "    DCT CaSR → TRPV5 (epithelial Ca channel) reduced → less Ca2+ entry; "
            "    Net: calciuria at serum Ca that should cause avidly low urine Ca; "
            "  Kidney Mg: TAL CaSR GOF → reduced Mg reabsorption → hypomagnesaemia; "
            "  TREATMENT RATIONALE: rPTH restores PTH1R-mediated renal Ca reabsorption independently of CaSR; "
            "    Calcitriol + Ca supplement: bypasses intestinal deficiency BUT cannot correct renal CaSR → "
            "      Ca absorbed → renal CaSR further activated → calciuria → nephrocalcinosis"
        ),
    },
    {
        "gene": "GNA11",
        "protein": (
            "GNA11 -- 19p13.3 AD-GOF -- 359aa -- Galpha11-G-Protein-Alpha-Subunit-"
            "42kDa-GPCR-Signal-Transducer-Downstream-CaSR-GOF-ADH2-"
            "Autosomal-Dominant-Hypocalcemia-Type-2-"
            "CINACALCET-INEFFECTIVE-Upstream-OMIM-Gene-139313-Disease-OMIM-615361"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "359 aa / 42 kDa (Gα11 — G-protein alpha subunit 11; "
            "Gαq/11 subfamily; "
            "NORMAL FUNCTION: "
            "  CaSR activates → GDP on Gα11 exchanged for GTP → Gα11 dissociates → activates PLC-β; "
            "  PLC-β → IP3 + DAG → intracellular Ca2+ release → PTH secretion suppressed; "
            "  Gα11 is the primary G-protein coupling CaSR in parathyroid chief cells; "
            "GNA11 GOF MECHANISM: "
            "  Activating mutations → Gα11 constitutively active (GTPase activity reduced → stays GTP-bound); "
            "  Downstream: same PLC-β → IP3 activation as CaSR GOF; "
            "  Net effect: PTH suppressed at lower Ca2+ levels — identical phenotype to ADH1 (CASR GOF); "
            "DISTINCTION from CASR: "
            "  CaSR GOF (ADH1) also activates Gα11 AND other signalling → full CaSR downstream cascade; "
            "  GNA11 GOF (ADH2) is DOWNSTREAM of CaSR → "
            "    cinacalcet (CaSR positive allosteric modulator) acts UPSTREAM of Gα11 → "
            "    cinacalcet may not correct ADH2 (contrast ADH1 where cinacalcet is CI, ADH2 where it is unhelpful); "
            "encoded 19p13.3"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GAIN-OF-FUNCTION (GOF) — GNA11 ADH2: "
            "  Single GOF allele → constitutive Gα11 signalling → PTH suppression; "
            "  De novo mutations frequent; "
            "  Mutation spectrum: residues around GTPase active site (Arg60, Phe341); "
            "  Phenotype: clinically indistinguishable from ADH1 (CASR GOF) on initial biochemistry; "
            "CLINICAL FEATURES: "
            "    Hypocalcaemia; "
            "    RELATIVE HYPERCALCIURIA (as for ADH1) — Gα11 expressed in kidney TAL too; "
            "    Hypomagnesaemia (same renal Mg wasting as ADH1); "
            "    Low-detectable PTH (similar to ADH1); "
            "TREATMENT: "
            "    rPTH 1-34 or 1-84 preferred; "
            "    Calcium + calcitriol: same nephrocalcinosis risk as ADH1; "
            "    Cinacalcet: NOT helpful (acts upstream of Gα11 at CaSR level — cannot modulate constitutively active Gα11); "
            "DIAGNOSIS: "
            "    Biochemistry identical to ADH1 (CASR GOF); "
            "    Distinguish by gene sequencing: ADH1 = CASR panel; ADH2 = GNA11 panel; "
            "    Order: CASR first (more common), then GNA11 if CASR negative"
        ),
        "disease_category": (
            "AUTOSOMAL DOMINANT HYPOCALCEMIA TYPE 2 (ADH2) — GNA11 GOF: "
            "  BIOCHEMICAL PROFILE: identical to ADH1 — "
            "    Hypocalcaemia + hyperphosphataemia + low PTH + RELATIVE HYPERCALCIURIA + low Mg; "
            "  DISTINGUISHING from ADH1: requires gene sequencing; "
            "    Clinical tip: if ADH phenotype + CASR negative → GNA11; "
            "  DISTINGUISHING from non-ADH HP: "
            "    Key test: 24h urine Ca in the context of serum Ca; "
            "    ADH1/ADH2: urine Ca HIGH despite low serum Ca; "
            "    Other HP (GCM2/PTH/TBCE/FAM111A/GATA3/SOX3): urine Ca LOW; "
            "  TREATMENT RULES (same as ADH1): "
            "    Cinacalcet: NOT indicated and theoretically unhelpful (different from ADH1 where it is ABSOLUTE CI — "
            "      in ADH1 cinacalcet is CI because it worsens CaSR GOF; in ADH2 CaSR is downstream, cinacalcet doesn't worsen it "
            "      but does nothing useful as the problem is Gα11); "
            "    rPTH preferred; low calcium target (1.9-2.1); renal monitoring"
        ),
        "disease_pathway": (
            "GNA11 GOF → CONSTITUTIVE PLC-BETA SIGNALLING → PTH SUPPRESSION: "
            "  Gα11 GOF: reduced GTPase activity → stays GTP-bound → persistent PLCβ activation; "
            "  PLCβ → PIP2 → IP3 + DAG; "
            "  IP3 → IP3R (ER receptor) → ER Ca2+ release → cytosolic Ca2+ ↑ → "
            "    calmodulin → CaMKII → phosphorylation of PTH secretory machinery → exocytosis blocked; "
            "  Parathyroid: constitutive signal → PTH suppressed at normal/low Ca2+; "
            "  Kidney TAL: Gα11 also expressed → Gα11 GOF → ROMK/claudin reduction → renal Ca/Mg wasting; "
            "  CINACALCET MECHANISM: "
            "    Cinacalcet → CaSR VFT domain → shifts CaSR activation curve left; "
            "    In ADH1: this WORSENS CaSR GOF → CI; "
            "    In ADH2: cinacalcet acts upstream of Gα11 → cannot modulate constitutively active Gα11 "
            "      → ineffective, not harmful per se, but not used"
        ),
    },
    {
        "gene": "TBCE",
        "protein": (
            "TBCE -- 1q42.3 AR -- 527aa -- Tubulin-Specific-Chaperone-E-"
            "59kDa-Microtubule-Assembly-Factor-HRD-Syndrome-"
            "Hypoparathyroidism-Retardation-Dysmorphism-Sanjad-Sakati-"
            "Kenny-Caffey-Syndrome-Type-1-KCS1-"
            "Arab-Founder-IVS1-2A>G-"
            "OMIM-Gene-604934-Disease-OMIM-241410-244460"
        ),
        "locus": "1q42.3",
        "protein_size": (
            "527 aa / 59 kDa (TBCE — tubulin-specific chaperone E; "
            "contains UBL (ubiquitin-like) domain + CAP-Gly motif; "
            "FUNCTION: "
            "  Required for de novo microtubule α/β-tubulin heterodimer assembly; "
            "  Pathway: TBCA → pre-folded β-tub; TBCE + TBCD → α-tubulin processing; "
            "    TBCC + TBCB activate GTPase on β-tub → correct GTP hydrolysis → stable αβ dimer; "
            "  TBCE LOF → microtubule assembly defect → affects rapidly dividing/differentiating cells; "
            "  Parathyroid: small glands depend on microtubule-dependent cell division during embryogenesis → "
            "    TBCE LOF → parathyroid hypoplasia → hypoparathyroidism; "
            "  Brain: neurodevelopmental role → intellectual disability; "
            "  Facial/skeletal: craniofacial + limb development; "
            "encoded 1q42.3; "
            "HRD/Sanjad-Sakati syndrome: common in Arab/Middle Eastern populations due to founder mutation IVS1-2A>G"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — TBCE: "
            "  Biallelic LOF mutations → HRD syndrome (Sanjad-Sakati) or Kenny-Caffey syndrome type 1 (KCS1); "
            "  FOUNDER MUTATION: IVS1-2A>G (splice acceptor) — highly prevalent in Arab/Middle Eastern/consanguineous; "
            "    Carrier frequency ~1/80 in some Saudi Arabian populations; "
            "  KCS1 variant: TBCE R262H — produces bone thickening but milder ID; "
            "HRD SYNDROME (= Sanjad-Sakati syndrome) FEATURES: "
            "    Hypoparathyroidism (neonatal onset); "
            "    Intellectual disability (mild-moderate); "
            "    Dysmorphic features: microcephaly, deep-set eyes, beaked nose, long philtrum, micrognathia, small hands/feet; "
            "    Growth retardation: severe short stature; "
            "    Eye abnormalities: microphthalmia, microphthalmos, corneal opacification; "
            "KENNY-CAFFEY SYNDROME TYPE 1 (KCS1): "
            "    Short stature + cortical bone thickening (tubular bones); "
            "    Eye: microphthalmia + hyperopia (common); "
            "    Hypoparathyroidism; "
            "    WITHOUT intellectual disability of same severity as HRD; "
            "TREATMENT: "
            "    Calcium + calcitriol (same HP treatment); "
            "    Developmental support, ophthalmological follow-up; "
            "    No specific TBCE therapy"
        ),
        "disease_category": (
            "SYNDROMIC HYPOPARATHYROIDISM — HRD/SANJAD-SAKATI AND KCS1: "
            "  KEY DISCRIMINATORS: "
            "    TBCE (HRD): INTELLECTUAL DISABILITY present — differentiates from FAM111A (KCS2, no ID); "
            "    TBCE (HRD): DYSMORPHIC FEATURES (microcephaly, micrognathia) — differentiates from isolated HP (GCM2/PTH); "
            "    TBCE (KCS1) vs FAM111A (KCS2): "
            "      KCS1 (TBCE, AR): intellectual disability +/-, microphthalmos, corneal opacification; "
            "      KCS2 (FAM111A, AD): NO intellectual disability, normal intelligence, telecanthus; "
            "  CONSANGUINITY: strongly associated with TBCE (AR — common in consanguineous populations); "
            "  ETHNIC CLUE: Arab/Middle Eastern ancestry → consider IVS1-2A>G TBCE founder; "
            "  URINE Ca: LOW (as for all non-ADH1/2 HP); "
            "  PTH: low/undetectable; "
            "  BONE: cortical thickening in KCS1 (tubular bones); density paradox — dense but fragile; "
            "  OPHTHALMOLOGY MANDATORY: microphthalmos, corneal pathology, high hyperopia in KCS1/HRD"
        ),
        "disease_pathway": (
            "TBCE LOF → MICROTUBULE ASSEMBLY FAILURE → MULTI-ORGAN HYPOPLASIA: "
            "  TBCE required for α-tubulin folding via TBC pathway (TBCA-B-C-D-E); "
            "  LOF → α-tubulin not properly folded → reduced αβ-tubulin heterodimer pool; "
            "  Impact by organ: "
            "    Parathyroid chief cells: mitotic spindle failure during embryonic gland development → "
            "      fewer chief cells → parathyroid hypoplasia; "
            "    Neurons: dendritic/axonal microtubule deficiency → intellectual disability; "
            "    Craniofacial: branchial arch neural crest cell migration/division affected → dysmorphology; "
            "    Bone/length: growth plate chondrocyte division → short stature; "
            "    Eye: lens fibre cell organisation → microphthalmos; "
            "  IVS1-2A>G: disrupts splice acceptor of intron 1 → exon 2 skipping → "
            "    premature stop codon → NMD → near-complete TBCE LOF in homozygotes"
        ),
    },
    {
        "gene": "FAM111A",
        "protein": (
            "FAM111A -- 11q13.1 AD-GOF -- 611aa -- FAM111-Protease-Family-Member-A-"
            "69kDa-Serine-Protease-PCNA-Interactor-Kenny-Caffey-Syndrome-Type-2-KCS2-"
            "Short-Stature-Dense-Tubular-Bones-Hypoparathyroidism-NO-Intellectual-Disability-"
            "Osteocraniostenosis-GOF-Severe-Allele-"
            "OMIM-Gene-615292-Disease-OMIM-127000"
        ),
        "locus": "11q13.1",
        "protein_size": (
            "611 aa / 69 kDa (FAM111A — FAM111 trypsin-like serine protease; "
            "DOMAINS: "
            "  UBL (ubiquitin-like) domain (aa 1-80); "
            "  PIP box / PCNA-interacting motif (aa 164-173) — key binding site; "
            "  Trypsin-like serine protease domain (aa 420-611); "
            "NORMAL FUNCTION (WT): "
            "  Binds PCNA at replication forks; "
            "  Recruits POL-η to stalled replication forks → restarts DNA synthesis; "
            "  Regulated protease: prodomains keep activity low; "
            "GOF MECHANISM (KCS2): "
            "  GOF mutations (R569H, S342R, Y468H) increase protease activity; "
            "  Hyperactive FAM111A → inappropriately cleaves PCNA-dependent replication factors; "
            "    → genome instability → uncontrolled cell cycle in some lineages; "
            "  Parathyroid: hyperstimulated protease → chief cell loss during embryogenesis; "
            "  Bone: cortical bone thickening (osteoblast hyperactivity); "
            "encoded 11q13.1; "
            "Osteocraniostenosis (OCS): severe GOF → lethal craniosynostosis + bone overgrowth"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GAIN-OF-FUNCTION (GOF) — FAM111A: "
            "  Kenny-Caffey syndrome type 2 (KCS2): moderate GOF (R569H most common de novo); "
            "  Osteocraniostenosis (OCS): severe GOF → lethal early; "
            "  De novo in most KCS2 cases (no family history expected); "
            "KCS2 FEATURES: "
            "    Short stature (severe prenatal-onset growth restriction); "
            "    Cortical bone thickening: dense tubular bones (radiograph: chalk-stick appearance); "
            "    Hypoparathyroidism: onset neonatal-early childhood; "
            "    Ophthalmologic: telecanthus, deep-set eyes (milder than TBCE); "
            "    INTELLIGENCE: NORMAL — key discriminator from KCS1 (TBCE) and HRD; "
            "    No microcephaly, no corneal opacification; "
            "  BIOCHEMISTRY: "
            "    Low Ca, high PO4, low PTH; "
            "    Urine Ca: LOW (renal CaSR not affected — same as all non-ADH HP); "
            "    ALP: may be elevated (bone turnover); "
            "TREATMENT: "
            "    HP: calcium + calcitriol / rPTH; "
            "    Bone: no specific therapy; fracture prevention (bones dense but brittle); "
            "    Growth: GH therapy may be considered for severe short stature"
        ),
        "disease_category": (
            "KENNY-CAFFEY SYNDROME TYPE 2 (KCS2) — FAM111A GOF — KEY DISCRIMINATORS: "
            "  KCS2 vs KCS1 (TBCE AR): "
            "    KCS2 (FAM111A AD): normal intelligence; DE NOVO dominant; telecanthus; no corneal opacity; "
            "    KCS1 (TBCE AR): intellectual disability; AR (need biallelic); corneal opacification; microphthalmos; "
            "    Both: dense tubular bones + HP + short stature; "
            "  KCS2 vs HRD/TBCE: "
            "    HRD: microcephaly + severe ID + dysmorphic; AR; consanguineous/Arab; "
            "    KCS2: normal intelligence; AD; de novo; bone thickening predominant; "
            "  KCS2 vs isolated HP (GCM2): "
            "    GCM2: no bone abnormality, normal height, no dysmorphology; "
            "    KCS2: characteristic short stature + dense bones = distinguish on X-ray; "
            "  RADIOLOGY KEY: dense tubular bones on X-ray (long bone cortex thickened) + "
            "    small medullary cavity + delayed bone age; "
            "  OPHTHALMOLOGY: dilated fundus exam (telecanthus + hyperopia in some KCS2)"
        ),
        "disease_pathway": (
            "FAM111A GOF → PROTEASE HYPERACTIVITY → PARATHYROID AND BONE PATHOLOGY: "
            "  GOF mutations in trypsin-like domain (R569H): increased intrinsic protease activity; "
            "  PCNA interaction retained → hyperactive FAM111A at replication forks; "
            "  Parathyroid: "
            "    During embryogenesis, chief cell progenitor proliferation requires PCNA-dependent DNA replication; "
            "    FAM111A GOF → replication stress → parathyroid progenitor cell death → gland hypoplasia; "
            "  Bone: "
            "    Osteoblast-lineage cells express FAM111A; GOF → altered osteoblast/osteoclast balance; "
            "    Net: excess cortical bone deposition (GOF promotes osteoblast survival); "
            "  Molecular parallel to CASR/GNA11: unrelated mechanism — replication stress vs Ca-sensing; "
            "  OCS (severe GOF): exaggerated bone overgrowth → craniosynostosis → fatal neonatal"
        ),
    },
    {
        "gene": "GATA3",
        "protein": (
            "GATA3 -- 10p14 AD-LOF -- 444aa -- GATA-Binding-Protein-3-"
            "48kDa-Dual-Zinc-Finger-TF-HDR-Syndrome-Barakat-Syndrome-"
            "Hypoparathyroidism-Deafness-Renal-Anomalies-Triad-"
            "SNHL-Bilateral-Often-First-Feature-"
            "OMIM-Gene-131320-Disease-OMIM-146255"
        ),
        "locus": "10p14",
        "protein_size": (
            "444 aa / 48 kDa (GATA3 — GATA-binding transcription factor 3; "
            "STRUCTURE: "
            "  Two zinc-finger domains (N-terminal ZF1 aa 253-277; C-terminal ZF2 aa 317-343); "
            "  ZF2: DNA binding to WGATAR (GATA motif); "
            "  ZF1: protein-protein interactions (FOG co-repressor); "
            "EXPRESSION: "
            "  Parathyroid gland development (3rd-4th pharyngeal pouch); "
            "  Inner ear (cochlear hair cell + stria vascularis development); "
            "  Kidney (ureteric bud + nephric duct); "
            "  T-lymphocytes (Th2 differentiation — hence T-cell immunodeficiency in some patients); "
            "LOF MECHANISM (HDR): "
            "  Haploinsufficiency — single allele LOF → parathyroid + ear + kidney; "
            "  All three organs develop simultaneously from GATA3-expressing progenitors; "
            "encoded 10p14"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT LOSS-OF-FUNCTION (HAPLOINSUFFICIENCY) — GATA3: "
            "  Single pathogenic allele → HDR syndrome; "
            "  Mutation types: missense (ZF domain), nonsense, frameshift, whole-gene deletion (FISH/array-CGH); "
            "  10p deletions: may also delete GATA3 → HDR + developmental delay if larger deletion; "
            "HDR / BARAKAT SYNDROME TRIAD: "
            "    (H) Hypoparathyroidism: neonatal or early childhood; degree varies; "
            "    (D) Deafness: bilateral sensorineural hearing loss (SNHL); "
            "      OFTEN MOST PROMINENT/FIRST feature — mild-profound; progressive; "
            "      Pre-lingual (congenital) in many; "
            "    (R) Renal anomalies: "
            "      Renal dysplasia (commonest); aplasia/hypoplasia; horseshoe kidney; "
            "      Pelviureteric junction (PUJ) obstruction; vesicoureteric reflux (VUR); "
            "      May present as neonatal renal failure or recurrent UTIs; "
            "PRESENTATION TIPS: "
            "    NOT all three features always present at once (variable penetrance per organ); "
            "    SNHL may be the presenting feature → check Ca/PTH in any child with bilateral SNHL; "
            "    Renal anomaly on antenatal USS → check hearing + Ca postnatally; "
            "TREATMENT: "
            "    HP: calcium + calcitriol; hearing aids / cochlear implants for SNHL; "
            "    Nephrology follow-up; hypertension screening; dialysis/transplant if severe renal disease; "
            "    Audiological assessment mandatory at diagnosis"
        ),
        "disease_category": (
            "HDR / BARAKAT SYNDROME — HYPOPARATHYROIDISM + DEAFNESS + RENAL ANOMALIES: "
            "  KEY DISCRIMINATORS: "
            "    GATA3 (HDR): SNHL bilateral + renal anomalies + HP = TRIAD; "
            "    GCM2: isolated HP, no deafness, no renal; "
            "    TBCE (HRD): HP + ID + dysmorphic, no deafness, no renal anomalies; "
            "    SLC26A4 (Pendred): HP is NOT a feature — SNHL + goiter + Mondini/EVA; "
            "  AUDIOLOGICAL CLUE: "
            "    Any bilateral SNHL in a child → reflexly check serum Ca + PTH; "
            "    GATA3 SNHL: affects all frequencies; may be progressive; aids/CI effective; "
            "  RENAL CLUE: "
            "    Antenatal hydronephrosis + bilateral SNHL → GATA3 until proven otherwise; "
            "  GENE PANEL: GATA3 should be in every panel for syndromic HP; "
            "  URINE Ca: LOW (same as GCM2/PTH/TBCE/FAM111A/SOX3 — renal CaSR not affected); "
            "  IMMUNE: rare severe GATA3 LOF → T-cell dysfunction (Th2 pathway) — check lymphocytes"
        ),
        "disease_pathway": (
            "GATA3 LOF → SIMULTANEOUS PARATHYROID + COCHLEA + KIDNEY DEVELOPMENTAL FAILURE: "
            "  GATA3 in pharyngeal pouch (parathyroid): "
            "    GATA3 activates GCM2 (among others) → parathyroid fate commitment; "
            "    LOF → GCM2 not fully activated → parathyroid gland hypoplasia → HP; "
            "  GATA3 in otocyst (inner ear): "
            "    Stria vascularis differentiation → endocochlear potential generation; "
            "    LOF → hair cell and stria vascularis degeneration → SNHL; "
            "  GATA3 in kidney: "
            "    Ureteric bud branching + nephric duct → collecting system and kidney; "
            "    LOF → ureteric bud morphogenesis failure → renal dysplasia/hypoplasia; "
            "  HAPLOINSUFFICIENCY: 50% reduction in GATA3 level → below threshold for all three organs; "
            "  TRIAD = three GATA3-dependent organs simultaneously affected at haploinsufficiency threshold"
        ),
    },
    {
        "gene": "SOX3",
        "protein": (
            "SOX3 -- Xq27.1 XL -- 446aa -- SRY-Box-TF-3-"
            "46kDa-HMG-Box-SOX-Family-B1-"
            "X-Linked-Hypoparathyroidism-Males-Only-Clinically-"
            "Insertion-Deletion-SOX3-Regulatory-Region-Parathyroid-Aplasia-Ectopia-"
            "Females-Carriers-Normal-OMIM-Gene-313430-Disease-OMIM-307700-300290"
        ),
        "locus": "Xq27.1",
        "protein_size": (
            "446 aa / 46 kDa (SOX3 — SRY-related HMG-box transcription factor 3; "
            "SOX family group B1 (with SOX1, SOX2, SOX3); "
            "DOMAINS: "
            "  HMG (high-mobility group) box DNA-binding domain (aa 150-220); "
            "    Binds AACAAAG motif; bends DNA 70-80°; "
            "  C-terminal transactivation domain; "
            "EXPRESSION: "
            "  Anterior pituitary / hypothalamus (GOF duplications → hypopituitarism); "
            "  Parathyroid gland primordia (during embryogenesis); "
            "  SOX3 REGULATORY REGION: in Xq27.1, ~200 kb upstream of SOX3 coding region; "
            "    Insertions/deletions here alter SOX3 expression timing/level → parathyroid aplasia; "
            "MECHANISM FOR HP: "
            "  Regulatory region insertion (polyA tract expansion) → ectopic SOX3 expression in parathyroid precursors; "
            "    → disrupts GCM2/GATA3 developmental programme → parathyroid aplasia or ectopia; "
            "  NOT a SOX3 coding mutation — regulatory disruption; "
            "encoded Xq27.1; "
            "X-LINKED: males have single X → hemizygous → clinically affected; females have two X → lyonization"
        ),
        "inheritance": (
            "X-LINKED (XL) REGULATORY INSERTION/DELETION — SOX3: "
            "  Insertion/deletion in SOX3 regulatory region on Xq27.1; "
            "  Males (XY): single X → hemizygous for insertion → clinically affected; "
            "  Females (XX): carrier → lyonisation (X-inactivation) → usually asymptomatic or very mild; "
            "  Transmission: carrier mother → 50% sons affected; 50% daughters carriers; "
            "CLINICAL FEATURES: "
            "    Isolated hypoparathyroidism (no ID, no dysmorphic features, no deafness — isolated); "
            "    Parathyroid glands: absent or ectopic on imaging; "
            "    Onset: neonatal (profound) or childhood; "
            "    Serum Ca: low; PTH: low/undetectable; phosphate: high; "
            "    Males only clinically significant; "
            "    Female carriers: usually normal biochemistry; "
            "DIAGNOSIS: "
            "    Clinical clue: X-linked pattern (maternal uncles affected); "
            "    Chromosomal microarray / Southern blot to detect Xq27.1 regulatory region insertion; "
            "    Standard gene sequencing of SOX3 coding region MISSES this (regulatory, not coding); "
            "    Must request SPECIFIC REGULATORY REGION ANALYSIS — key diagnostic pitfall; "
            "TREATMENT: standard HP management (calcium + calcitriol / rPTH)"
        ),
        "disease_category": (
            "X-LINKED HYPOPARATHYROIDISM — ISOLATED — REGULATORY SOX3 DISRUPTION: "
            "  KEY DISTINGUISHING FEATURES: "
            "    X-linked: males affected, females carriers (key family history clue); "
            "    ISOLATED HP: no deafness (vs GATA3), no dysmorphic features (vs TBCE), no renal anomalies; "
            "    DIAGNOSTIC PITFALL: "
            "      Standard SOX3 coding region sequencing NORMAL in X-linked HP; "
            "      Must request Xq27.1 regulatory region CNV/MLPA or array-CGH to detect insertion/deletion; "
            "  FAMILY HISTORY PATTERN: "
            "    Maternal uncles with HP → X-linked → SOX3 regulatory; "
            "    Autosomal HP (GCM2/PTH/CASR/GNA11/TBCE/FAM111A/GATA3) → father-to-son transmission possible; "
            "    X-linked: NO father-to-son transmission (father gives Y to sons); "
            "  URINE Ca: LOW (same as other non-ADH HP); "
            "  PARATHYROID GLANDS: absent or ectopic on 4D-CT or 99mTc-MIBI (contrast with PTH gene where glands present); "
            "  ANTERIOR PITUITARY: monitor — SOX3 coding duplications → hypopituitarism (not the same mutation; separate entity)"
        ),
        "disease_pathway": (
            "SOX3 REGULATORY INSERTION → ECTOPIC SOX3 EXPRESSION → PARATHYROID APLASIA: "
            "  Normal: SOX3 regulatory region (Xq27.1, ~200kb upstream) has defined enhancers; "
            "    SOX3 expressed in pituitary/hypothalamus but NOT in parathyroid primordia; "
            "  Regulatory insertion (polyA/CAG repeat expansion): "
            "    Disrupts chromatin loop or insulator → SOX3 expression ectopically in parathyroid primordia; "
            "  Ectopic SOX3 in parathyroid progenitor: "
            "    SOX3 → competes with or represses GCM2 transcriptional programme; "
            "    → parathyroid fate not established → aplasia or ectopia; "
            "  Male (hemizygous): single copy of insertion → complete parathyroid aplasia; "
            "  Female (heterozygous): X-inactivation → ~50% cells use normal X → sufficient parathyroid development; "
            "  CODING REGION NORMAL: explains why standard sequencing misses this"
        ),
    },
]


# ── patient cohort generator ──────────────────────────────────────────────────
def _make_patients(seed: int, gene: str, n: int = 40) -> list:
    rng = random.Random(seed)
    genders  = ["M", "F"]
    # Onset ages vary by gene
    onset_ranges = {
        "GCM2": (0, 48), "PTH": (0, 36), "CASR": (0, 72), "GNA11": (0, 72),
        "TBCE": (0, 12), "FAM111A": (0, 24), "GATA3": (0, 60), "SOX3": (0, 24),
    }
    onset_range = onset_ranges.get(gene, (0, 48))

    # Seizure rate (due to hypocalcemia)
    sz_rates = {"GCM2": 0.65, "PTH": 0.60, "CASR": 0.40, "GNA11": 0.38,
                "TBCE": 0.70, "FAM111A": 0.55, "GATA3": 0.50, "SOX3": 0.72}
    sz_rate = sz_rates.get(gene, 0.55)

    # Calcitriol/rPTH treatment response
    def tx_response():
        r = rng.random()
        if gene in ("CASR", "GNA11"):
            if r < 0.30: return "rPTH 1-34"
            if r < 0.55: return "Ca+calcitriol (low-target)"
            if r < 0.70: return "Ca+calcitriol+thiazide"
            if r < 0.85: return "rPTH 1-84"
            return "Ca+calcitriol (high-risk nephrocalcinosis)"
        else:
            if r < 0.50: return "Ca+calcitriol"
            if r < 0.75: return "rPTH 1-34"
            if r < 0.88: return "rPTH 1-84"
            if r < 0.94: return "Ca+calcitriol+Mg"
            return "Ca+calcitriol+phosphate-binder"

    # Urine Ca pattern
    def urine_ca():
        if gene in ("CASR", "GNA11"):
            val = rng.uniform(7.0, 14.0)
            return f"HIGH {val:.1f} mmol/24h"
        else:
            val = rng.uniform(1.0, 3.5)
            return f"LOW {val:.1f} mmol/24h"

    patients = []
    for i in range(n):
        onset_mo = rng.randint(*onset_range)
        age_dx   = onset_mo + rng.randint(0, 18)
        ca       = round(rng.uniform(1.55, 2.05), 2)
        pth      = round(rng.uniform(1.0, 18.0), 1)
        phos     = round(rng.uniform(1.6, 3.2), 2)
        mg       = round(rng.uniform(0.55, 0.95), 2) if gene in ("CASR", "GNA11") else round(rng.uniform(0.72, 1.10), 2)
        gender   = rng.choice(genders) if gene != "SOX3" else "M"
        seizures = rng.random() < sz_rate
        tetany   = rng.random() < 0.72
        nephrocalcinosis = (gene in ("CASR", "GNA11")) and rng.random() < 0.35
        patients.append({
            "id": f"{gene}-{i+1:02d}",
            "gene": gene,
            "gender": gender,
            "onset_months": onset_mo,
            "age_at_dx_months": age_dx,
            "serum_ca_mmol": ca,
            "serum_pth_pgml": pth,
            "serum_phos_mmol": phos,
            "serum_mg_mmol": mg,
            "urine_ca_24h": urine_ca(),
            "seizures": seizures,
            "tetany": tetany,
            "nephrocalcinosis": nephrocalcinosis,
            "treatment": tx_response(),
        })
    return patients


# ── API surface ───────────────────────────────────────────────────────────────
def generate_overview() -> dict:
    """Atlas overview — aggregate stats across all 8 HP genes."""
    all_patients = []
    for idx, g in enumerate(ATLAS_GENES):
        all_patients.extend(_make_patients(SEED_BASE + idx, g["gene"]))

    n = len(all_patients)
    n_seizures        = sum(1 for p in all_patients if p["seizures"])
    n_tetany          = sum(1 for p in all_patients if p["tetany"])
    n_nephrocalcinosis= sum(1 for p in all_patients if p["nephrocalcinosis"])
    n_high_urine_ca   = sum(1 for p in all_patients if p["urine_ca_24h"].startswith("HIGH"))
    mean_ca  = round(sum(p["serum_ca_mmol"]  for p in all_patients) / n, 2)
    mean_pth = round(sum(p["serum_pth_pgml"] for p in all_patients) / n, 1)

    gene_summary = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        gene_summary.append({
            "gene":       g["gene"],
            "locus":      g["locus"],
            "n_patients": len(pts),
            "mean_ca":    round(sum(p["serum_ca_mmol"]  for p in pts) / len(pts), 2),
            "mean_pth":   round(sum(p["serum_pth_pgml"] for p in pts) / len(pts), 1),
            "sz_rate_pct": round(100 * sum(1 for p in pts if p["seizures"]) / len(pts), 1),
            "high_urine_ca_pct": round(100 * sum(1 for p in pts if p["urine_ca_24h"].startswith("HIGH")) / len(pts), 1),
            "inheritance": (
                "AD/AR" if g["gene"] == "GCM2" else
                "AR/AD" if g["gene"] == "PTH" else
                "AD GOF" if g["gene"] in ("CASR", "GNA11", "FAM111A") else
                "AR"    if g["gene"] in ("TBCE",) else
                "AD LOF" if g["gene"] == "GATA3" else
                "XL"
            ),
        })

    return {
        "atlas": "Hereditary-Hypoparathyroidism-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": len(ATLAS_GENES),
        "n_patients": n,
        "seeds": f"{SEED_BASE}–{SEED_BASE + len(ATLAS_GENES) - 1}",
        "aggregate_metrics": {
            "mean_serum_ca_mmol": mean_ca,
            "mean_serum_pth_pgml": mean_pth,
            "seizures_pct": round(100 * n_seizures / n, 1),
            "tetany_pct": round(100 * n_tetany / n, 1),
            "nephrocalcinosis_pct": round(100 * n_nephrocalcinosis / n, 1),
            "high_urine_ca_pct": round(100 * n_high_urine_ca / n, 1),
        },
        "gene_summary": gene_summary,
        "key_discriminators": [
            "Urine Ca status: HIGH = ADH1/ADH2 (CASR/GNA11) — LOW = all other HP causes",
            "Cinacalcet: ABSOLUTE CI in ADH1 (CASR GOF) — worsens hypocalcaemia",
            "GATA3: HDR triad — HP + bilateral SNHL + renal anomalies",
            "TBCE HRD: HP + intellectual disability + dysmorphic (Middle Eastern founder)",
            "FAM111A KCS2: HP + dense bones + short stature — NORMAL intelligence vs KCS1",
            "SOX3 XL: males only — standard gene panel MISSES (regulatory, not coding)",
            "CASR/GNA11: nephrocalcinosis risk with Ca+calcitriol — use rPTH preferentially",
            "GCM2: most common familial isolated HP — first gene to test",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown with full patient-level data for each of the 8 HP genes."""
    genes_out = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        genes_out.append({
            "gene":             g["gene"],
            "protein":          g["protein"],
            "locus":            g["locus"],
            "protein_size":     g["protein_size"],
            "inheritance":      g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway":  g["disease_pathway"],
            "n_patients":       len(pts),
            "patients":         pts,
        })
    return {
        "atlas": "Hereditary-Hypoparathyroidism-Atlas",
        "count": len(genes_out),
        "genes": genes_out,
    }


def generate_definitions() -> dict:
    """Glossary of key concepts for the Hereditary Hypoparathyroidism Atlas."""
    return {
        "atlas": "Hereditary-Hypoparathyroidism-Atlas",
        "definitions": [
            {
                "term": "Urine Calcium as the Key ADH Discriminator",
                "definition": (
                    "All hypoparathyroidism causes LOW serum Ca2+ → EXPECTED: low urine Ca (kidneys conserve Ca). "
                    "ADH1 (CASR GOF) and ADH2 (GNA11 GOF): renal CaSR/Gα11 also constitutively activated → "
                    "  cannot reabsorb Ca even at low serum Ca → INAPPROPRIATELY HIGH urine Ca. "
                    "PATHOGNOMONIC RULE: "
                    "  Hypocalcaemia + HIGH 24h urine Ca (>6.25 mmol/24h) = ADH (CASR or GNA11) until proven otherwise; "
                    "  Hypocalcaemia + LOW urine Ca = all other HP (GCM2, PTH, TBCE, FAM111A, GATA3, SOX3). "
                    "CLINICAL IMPACT: "
                    "  ADH identified early → AVOID standard Ca+calcitriol dose escalation → prevent nephrocalcinosis; "
                    "  Start rPTH in ADH1/2 — bypasses renal CaSR, restores Ca reabsorption. "
                    "TEST: 24h urine Ca creatinine ratio (spot) or 24h collection, measured at diagnosis."
                ),
            },
            {
                "term": "Cinacalcet — Absolute Contraindication in ADH1 (CASR GOF)",
                "definition": (
                    "Cinacalcet mechanism: positive allosteric modulator (PAM) of CaSR → "
                    "  shifts CaSR activation curve left → CaSR activated at LOWER Ca2+. "
                    "In NORMAL CaSR: reduces PTH, used for primary hyperparathyroidism and CKD-MBD. "
                    "In ADH1 (CASR GOF): CaSR already hyperactivated → cinacalcet FURTHER sensitises → "
                    "  PTH further suppressed → serum Ca falls → LIFE-THREATENING HYPOCALCAEMIA. "
                    "ABSOLUTE CI in ADH1. "
                    "ADH2 (GNA11 GOF): cinacalcet ineffective (Gα11 constitutively active, upstream; "
                    "  cinacalcet cannot modulate constitutively active Gα11 directly) — not helpful, avoid. "
                    "KEY CLINICAL RULE: before using cinacalcet in any hypocalcaemia — "
                    "  EXCLUDE ADH1/ADH2 by 24h urine Ca + CASR/GNA11 gene panel FIRST."
                ),
            },
            {
                "term": "Nephrocalcinosis Risk in ADH1/ADH2 — Treatment Strategy",
                "definition": (
                    "Problem: ADH1/ADH2 → renal CaSR/Gα11 activated → any Ca absorbed → calciuria → "
                    "  calcium phosphate crystals in kidney tubules → medullary nephrocalcinosis. "
                    "Standard HP treatment (Ca + calcitriol): increases intestinal Ca absorption → "
                    "  more Ca in tubule → CaSR further activated → more calciuria → nephrocalcinosis. "
                    "SOLUTION: recombinant PTH (rPTH 1-34 teriparatide or 1-84 Natpara): "
                    "  PTH1R in kidney DCT: activates TRPV5 → Ca reabsorption → reduces calciuria even with normal serum Ca; "
                    "  Bypasses CaSR mechanism → can maintain serum Ca without driving calciuria. "
                    "MONITORING: renal ultrasound 6-12 monthly; 24h urine Ca target <6.25 mmol/24h; eGFR; "
                    "ADJUNCT: thiazide diuretic (reduces urine Ca by DCT NaCl reabsorption enhancement) "
                    "  + low-sodium diet (reduces Ca excretion); "
                    "SERUM Ca TARGET in ADH: 1.9-2.1 mmol/L (below normal range) — accept mild hypocalcaemia "
                    "  to prevent nephrocalcinosis."
                ),
            },
            {
                "term": "GCM2 — Most Common Familial Isolated Hypoparathyroidism",
                "definition": (
                    "GCM2 is the master transcription factor for parathyroid gland development. "
                    "LOF → parathyroid agenesis (biallelic AR) or hypoplasia (monoallelic AD haploinsufficiency). "
                    "FIRST GENE TO TEST in familial isolated HP (no dysmorphology, no deafness, no renal anomalies). "
                    "AD vs AR: "
                    "  AD (haploinsufficiency): incomplete penetrance ~60-70%; family member often only mildly biochemically affected; "
                    "  AR (biallelic): severe neonatal; consanguinity risk; "
                    "URINE Ca: LOW (renal CaSR not affected — Ca appropriately retained). "
                    "TREATMENT: Ca + calcitriol standard; rPTH improves urine Ca profile. "
                    "GCM2 panel note: GCM2 activates GATA3 in parathyroid primordia — both tested together in comprehensive panels."
                ),
            },
            {
                "term": "GATA3 HDR Syndrome — Bilateral SNHL as First Presenting Feature",
                "definition": (
                    "HDR (Hypoparathyroidism-Deafness-Renal) = Barakat syndrome. "
                    "GATA3 haploinsufficiency affects three organs simultaneously: "
                    "  Parathyroid (from pharyngeal pouch), inner ear (cochlear hair cells/stria vascularis), "
                    "  kidney (ureteric bud/nephron development). "
                    "CLINICAL PRESENTATION: "
                    "  Bilateral SNHL is OFTEN the FIRST recognised feature (congenital or early-onset); "
                    "  Any bilateral SNHL in a child → CHECK serum Ca + PTH; "
                    "  Renal anomaly on antenatal USS → CHECK hearing + Ca postnatally; "
                    "NOT ALL THREE FEATURES SIMULTANEOUSLY: partial phenotype in 30-40% (one or two organs at presentation). "
                    "CRITICAL DISTINCTION from SLC26A4 (Pendred): Pendred = SNHL + GOITER (not HP) + Mondini/EVA; "
                    "AUDIOLOGICAL MANAGEMENT: hearing aids; cochlear implants successful (inner ear anatomy variable). "
                    "RENAL MONITORING: eGFR, BP, urinalysis annually; renal dysplasia may → CKD."
                ),
            },
            {
                "term": "TBCE (HRD/Sanjad-Sakati) vs FAM111A (KCS2) — Key Differential",
                "definition": (
                    "BOTH: short stature + cortical bone thickening (KCS spectrum) + hypoparathyroidism. "
                    "TBCE (AR, HRD = Sanjad-Sakati): "
                    "  Intellectual disability (mild-moderate) + dysmorphic (microcephaly, beaked nose, micrognathia); "
                    "  Ophthalmology: microphthalmos, corneal opacification; "
                    "  Founder mutation: IVS1-2A>G in Arab/Middle Eastern consanguineous families; "
                    "  AR inheritance — BOTH parents must be carriers. "
                    "FAM111A (AD GOF, KCS2): "
                    "  NORMAL INTELLIGENCE — single most important differentiator; "
                    "  No microcephaly, no corneal opacification; telecanthus may be present; "
                    "  De novo dominant — no family history expected; "
                    "  Osteocraniostenosis (OCS): severe FAM111A GOF → lethal craniosynostosis. "
                    "DIAGNOSIS: "
                    "  AR family history + consanguinity + ID → TBCE; "
                    "  Normal IQ + dense bones + HP → FAM111A (AD); "
                    "  X-ray: long bone cortex thickening + small medullary cavity (both). "
                    "TREATMENT: both require HP management + orthopaedic/ophthalmological follow-up."
                ),
            },
            {
                "term": "SOX3 X-Linked HP — Regulatory Region Diagnostic Pitfall",
                "definition": (
                    "X-linked hypoparathyroidism caused by insertion/deletion in SOX3 REGULATORY REGION (Xq27.1). "
                    "NOT a coding mutation — STANDARD GENE SEQUENCING MISSES IT. "
                    "DIAGNOSTIC PITFALL: "
                    "  Patient presents with isolated HP + X-linked pattern; "
                    "  HP gene panel (coding): GCM2, PTH, CASR, GNA11, TBCE, FAM111A, GATA3 — ALL NEGATIVE; "
                    "  Test must specifically include: Xq27.1 regulatory region CNV/MLPA, array-CGH, or Southern blot. "
                    "CLINICAL CLUE: "
                    "  X-linked pattern: maternal uncles with HP; NO father-to-son transmission; "
                    "  Males ONLY clinically affected; females carriers (normal or borderline Ca). "
                    "PARATHYROID IMAGING: absent/ectopic glands (unlike PTH gene where glands present). "
                    "SOX3 duplications (different from regulatory insertion): cause hypopituitarism — "
                    "  different disease entity; do not confuse."
                ),
            },
            {
                "term": "Recombinant PTH in Hypoparathyroidism — Indications and Advantages",
                "definition": (
                    "Two approved/off-label rPTH formulations for HP management: "
                    "  Natpara (rPTH 1-84): FDA-approved for adults with HP, uncontrolled on Ca+calcitriol; "
                    "  Forteo (teriparatide 1-34): off-label for HP; both SC injection once or twice daily. "
                    "ADVANTAGES OF rPTH OVER Ca+CALCITRIOL: "
                    "  Restores PTH1R-mediated renal TRPV5 activation → Ca reabsorption → LESS CALCIURIA; "
                    "  Activates PTH1R-mediated phosphaturia → normalises PO4; "
                    "  Physiological fluctuation → closer to natural PTH pulsatility → better bone remodelling; "
                    "  Reduces urine Ca → CRITICAL in ADH1/ADH2 to prevent nephrocalcinosis. "
                    "SPECIFIC INDICATIONS: "
                    "  ADH1/ADH2: FIRST-LINE over Ca+calcitriol (nephrocalcinosis prevention); "
                    "  All HP uncontrolled on Ca+calcitriol (breakthrough hypocalcaemia or calciuria); "
                    "  HP with CKD: calcitriol dose limited by GFR → rPTH allows lower calcitriol. "
                    "MONITORING: Ca (weekly initially), 24h urine Ca, eGFR, bone turnover markers."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:500])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])}")
