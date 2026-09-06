#!/usr/bin/env python3
"""PP-Atlas — Complete 8-Gene Purine & Pyrimidine Metabolism Disorders Atlas
HPRT1 · ADA · ADSL · PNP · ATIC · APRT · XDH · UMPS
320-patient aggregate cohort (8 × 40, seeds 896–903)

Purine & Pyrimidine Metabolism Disorders facts:
  - Inherited defects in purine or pyrimidine synthesis/salvage/catabolism enzymes.
  - Clinically heterogeneous: ranges from SCID (ADA), self-mutilation (HPRT1),
    T-cell immunodeficiency (PNP), epilepsy (ADSL), kidney stones (APRT/XDH),
    to megaloblastic anemia (UMPS).
  - Collective incidence ~1/100,000–1/500,000 depending on disorder.
  - KEY TEACHING POINTS:
      HPRT1: X-linked; allopurinol controls gout/uric acid but NOT neurological features.
      ADA-SCID: First gene therapy disease (1990); PEGylated ADA (ADAGEN) available;
                gene therapy Strimvelis (EMA 2016). deoxyATP accumulates → T-cell toxicity.
      ADSL: SAICAR and SAICAr accumulate in CSF; seizures + ID + autism; no specific therapy.
      PNP: T-cell immunodeficiency + autoimmune hemolytic anemia; HSCT is curative.
      ATIC: AICA-ribosiduria; bifunctional enzyme; brain malformations + autistic features.
      APRT: 2,8-dihydroxyadenine (2,8-DHA) nephrolithiasis; allopurinol CURATIVE.
      XDH: Xanthinuria type I; allopurinol ABSOLUTELY CONTRAINDICATED
           (blocks XDH = the ONLY pathway left for xanthine catabolism → complete block).
      UMPS: Orotic aciduria type I; uridine monophosphate CURATIVE; NO hyperammonemia
            (DDx from OTC deficiency which causes secondary orotic aciduria WITH hyperammonemia).

COHORT: 8 × 40 = 320 patient slots (seeds 896–903; gene-specific seeds)
"""

import random

SEED_BASE = 896

# ── All 8 PP Genes ────────────────────────────────────────────────────────────────
PP_GENES = [
    # ── HPRT1 — Lesch-Nyhan syndrome ─────────────────────────────────────────────
    {
        "gene": "HPRT1", "alias": "HPRT1 — Lesch-Nyhan syndrome / HPRT deficiency (X-linked)",
        "aa": "218 aa", "kDa": "25 kDa",
        "gene_class": "Purine salvage enzyme (HGPRT)",
        "pp_subgroup": "Purine salvage pathway (HPRT1 · APRT)",
        "locus": "Xq26.2", "omim_gene": 308000,
        "phenotype": "Classic LNS: X-linked; self-mutilation, choreoathetosis, hyperuricemia, gout; allopurinol controls uric acid NOT neuro features",
        "disease": (
            "HPRT1 hemizygous loss → Lesch-Nyhan syndrome (LNS, OMIM #300322) or HPRT-related "
            "hyperuricemia/gout (partial deficiency). HPRT1 encodes hypoxanthine-guanine "
            "phosphoribosyltransferase (HGPRT, 218aa, 25kDa), the key enzyme in purine salvage: "
            "hypoxanthine + PRPP → IMP; guanine + PRPP → GMP. Without salvage, purines must be "
            "synthesised de novo (costly) and xanthine/uric acid accumulate via XDH/XO. "
            "Classic LNS (complete loss): severe neurological triad — choreoathetosis (basal ganglia "
            "dopamine deficit — HGPRT is required for dopamine neuron purine recycling), intellectual "
            "disability, and COMPULSIVE SELF-MUTILATION (biting lips/fingers — pathognomonic; "
            "distinguishes from other movement disorders; not volitional). Hyperuricemia → gout, "
            "tophi, uric acid nephrolithiasis, obstructive uropathy. Partial HPRT deficiency "
            "(Kelley-Seegmiller syndrome): hyperuricemia + gout but minimal/no neurological features. "
            "X-linked: virtually all affected patients are male; female carriers asymptomatic. "
            "Incidence: ~1/380,000 male births."
        ),
        "inheritance": "X-linked recessive (Xq26.2). Virtually all affected = males. Female carriers asymptomatic. De novo mutation rate ~50%.",
        "hallmark": (
            "HPRT1/LNS HALLMARKS: "
            "(1) SELF-MUTILATION PATHOGNOMONIC: compulsive self-biting of lips, fingers, buccal mucosa; "
            "not due to pain insensitivity — patients are distressed by their own behaviour; "
            "protective restraints requested by patients themselves; DDx from Lesch-Nyhan partial: no self-harm; "
            "(2) CHOREOATHETOSIS: basal ganglia dopamine deficiency (HGPRT required for dopaminergic neuron purine recycling); "
            "dopamine transporter imaging shows reduced uptake in caudate/putamen; "
            "(3) ALLOPURINOL CONTROLS URIC ACID BUT NOT NEUROLOGICAL FEATURES: critical point; "
            "allopurinol (XO inhibitor) normalises uric acid → prevents gout, nephrolithiasis, renal failure; "
            "does NOT reverse choreoathetosis, self-mutilation, or intellectual disability; "
            "start allopurinol early to prevent renal damage; do not expect neurological benefit; "
            "(4) HYPERURICEMIA: plasma uric acid >10 mg/dL often (normal <7); gout, tophi, uric acid crystals; "
            "(5) URIC ACID OVERPRODUCTION (not underexcretion): de novo purine synthesis runs unchecked "
            "without salvage pathway → massive uric acid production; 24h urine uric acid/creatinine ratio elevated; "
            "(6) X-LINKED: affected male, carrier female; check maternal uncle history; "
            "(7) DOPAMINE DEFICIT — NOT PRIMARY DEFECT: HGPRT indirectly required for dopamine synthesis "
            "(purine supply for dopaminergic neurons); L-DOPA trial historically unhelpful or worsened behaviour"
        ),
        "key_ddx": (
            "HPRT1 DDx: (1) Cerebral palsy: no hyperuricemia, no self-mutilation, normal uric acid; "
            "(2) Partial HPRT deficiency (Kelley-Seegmiller): same gene, higher residual HGPRT activity; "
            "hyperuricemia/gout but NO self-mutilation; distinguish by HGPRT enzyme assay in RBCs + HPRT1 sequencing; "
            "(3) Other dystonias/choreas: no uric acid elevation; "
            "(4) APRT deficiency: kidney stones (2,8-DHA, NOT uric acid); normal uric acid; "
            "(5) XDH deficiency (xanthinuria): low uric acid (OPPOSITE of LNS); xanthine stones; allopurinol CI"
        ),
        "diet_treatment": "Allopurinol (xanthine oxidase inhibitor): 10 mg/kg/day; normalises uric acid; prevents gout, nephrolithiasis, and renal failure. Does NOT help neurological features. High fluid intake (>2 L/m²/day) to prevent uric acid crystalluria. Low purine diet (limit organ meats, sardines, anchovies). Febuxostat alternative to allopurinol. Protective restraints for self-mutilation (patient-requested). Baclofen/clonazepam for spasticity. Dental extractions sometimes needed for severe self-biting.",
        "gene_therapy_status": "No approved gene therapy. HPRT1 lentiviral gene therapy trials ongoing (UCSF, 2020s). Haematopoietic stem cell gene therapy proof-of-concept in murine models. Small size of HPRT1 (657bp coding) makes AAV delivery straightforward. Allogeneic HSCT attempted historically — corrects haematopoietic compartment but does NOT rescue neurological features (basal ganglia already damaged).",
        "critical_ci": (
            "CRITICAL: (1) Expecting allopurinol to help neuro features — INCORRECT; "
            "allopurinol treats hyperuricemia only; neurological damage is irreversible; "
            "(2) Diagnosing as cerebral palsy — check uric acid in all unexplained choreoathetosis; "
            "(3) HSCT for neurological rescue — does NOT work; damaged basal ganglia cannot be restored; "
            "(4) Missing carrier females — often asymptomatic; genetic counselling mandatory; "
            "(5) Misidentifying self-mutilation as psychiatric — it is neurological/compulsive, not volitional"
        ),
        "nbs_marker": "Not in standard NBS. Uric acid elevated in plasma/urine (but not routinely screened at birth). Diagnosis: HGPRT enzyme assay in RBCs (< 1.5% activity = LNS; 1.5–10% = partial); confirmed by HPRT1 sequencing. Uric acid/creatinine ratio in urine (>0.75 mg/mg = overproduction). Neonatal hyperuricaemia in males should prompt HGPRT assay.",
        "key_biomarker": "Plasma uric acid >10 mg/dL. 24h urine uric acid elevated (>600 mg/1.73m²/day). Uric acid/creatinine ratio in spot urine >0.75 (>2y) or >2.0 (<1y). RBC HGPRT enzyme activity <1.5% (LNS) or 1.5-10% (partial). CSF uric acid low (despite elevated plasma — CSF purine metabolism autonomous). Dopamine transporter (DaT) SPECT: reduced caudate/putamen uptake.",
        "severity_spectrum": "Classic LNS (complete HGPRT loss <1.5%): choreoathetosis + self-mutilation + ID + hyperuricemia → Partial HPRT deficiency with neuro (1.5-8%): hyperuricemia + some neuro → Kelley-Seegmiller syndrome (8-60%): hyperuricemia/gout ONLY, no neuro → HPRT-related hyperuricemia (>60%): gout phenotype only.",
        "founder_variant": "No major founder. ~50% de novo. Missense, nonsense, splicing, large deletions. CpG hotspot variants: p.Asp201Asn, p.Val188Gly.",
        "key_variants": [
            "p.Asp201Asn — CpG hotspot; complete loss; classic LNS",
            "p.Val188Gly — CpG hotspot; complete loss; classic LNS",
            "p.Lys68Glu — partial function; Kelley-Seegmiller syndrome",
            "Large exon deletions — complete loss; classic LNS",
            "p.Ser109Leu — partial; intermediate phenotype",
        ],
        "seed": SEED_BASE + 0,
    },
    # ── ADA — ADA-SCID ────────────────────────────────────────────────────────────
    {
        "gene": "ADA", "alias": "ADA — ADA-SCID (first gene therapy disease; ERT and Strimvelis)",
        "aa": "363 aa", "kDa": "41 kDa",
        "gene_class": "Purine catabolism enzyme (adenosine deaminase)",
        "pp_subgroup": "Purine catabolism pathway (ADA · PNP · XDH)",
        "locus": "20q13.12", "omim_gene": 102700,
        "phenotype": "ADA-SCID: severe T-B-NK combined immunodeficiency; deoxyATP toxic to T-cells; PEGylated ADA (ADAGEN) ERT; gene therapy Strimvelis (EMA 2016)",
        "disease": (
            "ADA biallelic loss → ADA-SCID (OMIM #102700), the first genetic disease treated by gene therapy "
            "(Anderson et al., 1990). ADA encodes adenosine deaminase (363aa, 41kDa), which deaminates "
            "adenosine → inosine, and deoxyadenosine → deoxyinosine. Without ADA, deoxyadenosine "
            "accumulates → phosphorylated to deoxyadenosine triphosphate (dATP) by deoxycytidine kinase → "
            "dATP accumulates preferentially in lymphocytes (particularly T-cells) → inhibits ribonucleotide "
            "reductase (RNR, which requires dATP < total dNTP pool) → DNA synthesis blocked → T-cell apoptosis. "
            "B-cells and NK cells also depleted secondarily. Clinical: profound SCID (T-B-NK-) presenting "
            "in first months of life — recurrent/severe infections (Pneumocystis, CMV, EBV, adenovirus, "
            "BCG dissemination if inadvertently vaccinated), failure to thrive, absent lymph nodes/tonsils, "
            "lymphopenia (total lymphocytes <500/μL). Skeletal abnormalities (costochondral junctions, "
            "rib cupping) in ~50% — unique to ADA-SCID among SCID variants. ~15% of all SCID. "
            "Incidence: ~1/200,000–1/1,000,000. Delayed/late-onset ADA deficiency: milder alleles → "
            "onset in childhood/adulthood with combined immunodeficiency (not classic SCID)."
        ),
        "inheritance": "Autosomal recessive. ADA 20q13.12. ~15% of all SCID cases. All ethnicities; some founder alleles in specific populations.",
        "hallmark": (
            "ADA-SCID HALLMARKS: "
            "(1) dATP ACCUMULATION — MECHANISTIC KEY: deoxyadenosine → dATP via deoxycytidine kinase; "
            "dATP inhibits RNR (allosteric feedback); T-cells are uniquely sensitive because they cannot "
            "clear dATP via nucleotidase pathways efficiently; dATP >50% of total dNTP pool = RNR block; "
            "(2) T-B-NK- COMBINED SCID: all three lymphocyte lineages depleted (unlike X-linked SCID [IL2RG] "
            "which is T-B+NK- or RAG deficiency T-B-NK+); flow cytometry confirms; "
            "(3) COSTOCHONDRAL ABNORMALITIES: rib cupping/flaring on chest X-ray — unique to ADA-SCID "
            "among SCID subtypes; present in ~50%; radiographic clue; "
            "(4) PEGylated ADA (ADAGEN/elapegademase): bovine ADA conjugated to PEG; weekly IM injection; "
            "reduces dATP, restores partial immunity; NOT curative (does not correct lymphocyte development); "
            "used as bridge to HSCT or gene therapy; "
            "(5) GENE THERAPY STRIMVELIS (autologous HSC, γ-retroviral vector, EMA approved 2016): "
            "first approved gene therapy in Europe; autologous HSCs transduced with ADA cDNA → "
            "transplanted without myeloablation; curative in 70-80%; no graft failure or GVHD; "
            "PREFERRED over allogeneic HSCT if no matched sibling donor; "
            "(6) DO NOT GIVE LIVE VACCINES: BCG at birth can disseminate fatally; rotavirus, varicella CI; "
            "(7) IRRADIATED/CMV-NEGATIVE BLOOD PRODUCTS only until SCID diagnosed and treated"
        ),
        "key_ddx": (
            "ADA-SCID DDx among SCID subtypes: "
            "(1) X-linked SCID (IL2RG γc chain): T-B+NK- (B-cells present but non-functional); "
            "X-linked (males); no skeletal abnormalities; gene therapy also available; "
            "(2) RAG1/RAG2 deficiency: T-B-NK+ (NK cells present); AR; Omenn syndrome variant; "
            "(3) ARTEMIS/DNA-PKcs/LIG4 deficiency: T-B-NK+ or T-B-NK-; radiosensitive SCID; "
            "unique feature: radiation sensitivity — reduced-intensity conditioning ONLY; "
            "(4) JAK3 deficiency: T-B+NK- phenotype like IL2RG; AR; "
            "(5) PNP deficiency: T-cell deficiency (primarily) with autoimmune features; "
            "deoxyGTP accumulates (not dATP); less severe than ADA-SCID; "
            "(6) MHC class II deficiency (bare lymphocyte syndrome): CD4 cells absent; "
            "normal lymphocyte count but non-functional"
        ),
        "diet_treatment": "PEGylated ADA enzyme replacement (ADAGEN, elapegademase): 30-60 U/kg IM weekly. Gene therapy Strimvelis (EMA 2016) or clinical trials (lentiviral vector). Allogeneic HSCT (matched sibling = best; MUD acceptable). Prophylaxis: PCP prophylaxis (TMP-SMX), antifungal, antiviral (aciclovir, IVIG). NO live vaccines (BCG, rotavirus, varicella, MMR) — fatal dissemination risk. IVIG replacement until immune reconstitution.",
        "gene_therapy_status": "Strimvelis (EMA approved 2016): autologous HSC + γ-retroviral ADA vector; first approved gene therapy in EU. OTL-101 (lentiviral vector, Orchard Therapeutics): improved safety profile (no insertional oncogenesis with lentiviral vs retroviral); Phase III data 2022-2023 showing 90%+ immune reconstitution. PEGylated ADA (ADAGEN) bridge therapy. Allogeneic HSCT curative if matched sibling available (>90% survival).",
        "critical_ci": (
            "CRITICAL: (1) BCG vaccination at birth in undiagnosed ADA-SCID → disseminated BCG; fatal; "
            "do not administer BCG in countries with routine neonatal BCG until SCID excluded in symptomatic infants; "
            "(2) Live vaccines — absolutely contraindicated until immune reconstitution confirmed; "
            "(3) Non-irradiated blood products → transfusion-associated GVHD (fatal); "
            "(4) Expecting PEGylated ADA to cure — it bridges; it does not restore lymphocyte development fully; "
            "(5) Missing skeletal X-ray — costochondral rib abnormalities confirm ADA-SCID subtype among SCID"
        ),
        "nbs_marker": "ADA activity in dried blood spots (DBS): detectable by NBS in some programmes. T-cell receptor excision circles (TRECs) are the standard NBS SCID test — ADA-SCID has very low TRECs. ADA activity in RBCs: <1% of normal = complete deficiency. dATP/deoxyadenosine in RBCs/plasma elevated. Confirmed by ADA sequencing.",
        "key_biomarker": "RBC ADA enzyme activity <1% (complete ADA-SCID). dATP in RBCs/plasma markedly elevated (>50% of total dNTP pool). Lymphopenia: total lymphocytes <500/μL (normal >1500). CD3/CD4/CD8/CD16/CD19/CD56 flow cytometry: T-B-NK-. Costochondral rib abnormalities on chest X-ray (50%). TRECs near-zero on NBS.",
        "severity_spectrum": "Complete ADA-SCID (null/null): neonatal or early-infant T-B-NK- SCID; fatal without treatment → Partial ADA deficiency (some residual activity): delayed/late-onset combined immunodeficiency in childhood/adulthood → ADA deficiency with normal immunity (some hypomorphic alleles): detectable enzyme deficiency without clinical SCID.",
        "founder_variant": "No single founder. Hotspot: p.Arg156His (CpG), p.Arg101Gln, p.Glu337Lys (Ashkenazi Jewish enriched). ~120 pathogenic variants described.",
        "key_variants": [
            "p.Arg156His — CpG hotspot; null; complete ADA-SCID",
            "p.Arg101Gln — common; null; complete ADA-SCID",
            "p.Glu337Lys — Ashkenazi Jewish enriched; partial → delayed-onset",
            "p.Gln3Ter — early stop; null; complete ADA-SCID",
            "p.Gly216Arg — partial function; late-onset milder",
        ],
        "seed": SEED_BASE + 1,
    },
    # ── ADSL — Adenylosuccinate lyase deficiency ──────────────────────────────────
    {
        "gene": "ADSL", "alias": "ADSL — Adenylosuccinate lyase deficiency (SAICAR accumulation; seizures + ID + autism)",
        "aa": "484 aa", "kDa": "55 kDa",
        "gene_class": "Purine de novo synthesis enzyme (bifunctional lyase)",
        "pp_subgroup": "Purine de novo synthesis pathway (ADSL · ATIC)",
        "locus": "22q13.2", "omim_gene": 103050,
        "phenotype": "Seizures + ID + autism spectrum; SAICAR and SAICAr accumulate in CSF/urine; no specific therapy; succinyladenosine and SAICAR in CSF = pathognomonic",
        "disease": (
            "ADSL biallelic loss → Adenylosuccinate lyase (ADSL) deficiency (OMIM #103050). "
            "ADSL is a bifunctional enzyme (484aa, 55kDa, tetrameric) that catalyses two reactions "
            "in de novo purine synthesis: (1) SAICAR → AICAR + fumarate (step 8); "
            "(2) adenylosuccinate → AMP + fumarate (step 10). Both reactions produce fumarate. "
            "ADSL loss → accumulation of: SAICAR (succinylaminoimidazole carboxamide ribotide), "
            "SAICAr (succinylaminoimidazole carboxamide riboside, dephosphorylated form of SAICAR), "
            "and succinyladenosine (SA, the dephosphorylated adenylosuccinate product). "
            "These accumulate in CSF, urine, and (less) plasma. SAICAr and SA are NEUROTOXIC "
            "(mechanism unclear — possibly purinergic receptor interference or mitochondrial dysfunction). "
            "Clinical spectrum: (1) Severe neonatal form: fatal/severe encephalopathy, absent EEG activity; "
            "(2) Profound ID form: profoundly disabled, seizures, autistic features; "
            "(3) Mild-moderate form: developmental delay, autistic features, +/- seizures. "
            "Incidence: ~1/500,000. No sex predilection."
        ),
        "inheritance": "Autosomal recessive. ADSL 22q13.2. Pan-ethnic. No major founder allele in most populations; p.Arg426His (Belgian) enriched in Northern Europe.",
        "hallmark": (
            "ADSL HALLMARKS: "
            "(1) SUCCINYLADENOSINE (SA) IN CSF — PATHOGNOMONIC: SA detectable in CSF (normal = absent); "
            "also SAICAR and SAICAr in CSF and urine; Bratton-Marshall colorimetric test on urine screens "
            "for AICA/succinyl compounds (succinyl Bratton-Marshall positive); "
            "(2) AUTISM SPECTRUM FEATURES: stereotypies, poor social interaction, restricted interests — "
            "present in majority; ADSL deficiency is one of the few IEM with prominent autism features; "
            "(3) SEIZURES: multifocal myoclonic, tonic, absence; EEG: burst suppression in neonates; "
            "hypsarrhythmia possible; refractory to standard AEDs; "
            "(4) NO SPECIFIC TREATMENT: D-ribose supplementation (to restore purine synthesis) studied "
            "but no proven benefit; allopurinol has been tried (reduces de novo flux) — conflicting data; "
            "supportive care only; "
            "(5) BIFUNCTIONAL ENZYME — BOTH REACTIONS BLOCKED: one enzyme catalyses 2 steps; "
            "SAICAR accumulates (step 8 blocked) AND succinyladenosine accumulates (step 10 blocked); "
            "(6) LEAN/CACHEXIA: wasting phenotype in severe forms; "
            "(7) p.Arg426His SEVERITY GENOTYPE: R426H/R426H → profound ID; other combos → variable"
        ),
        "key_ddx": (
            "ADSL DDx: (1) Other IEM with autism/epilepsy: ATIC (AICAriboside in urine, not SA), "
            "GAMT deficiency (creatine deficiency, guanidinoacetate elevated), "
            "ARX mutations (X-linked, lissencephaly variants); "
            "(2) Rett syndrome: MECP2 mutation; female predominance; hand stereotypies; normal metabolites; "
            "(3) Angelman syndrome: UBE3A; happy affect; EEG pattern; maternal deletion 15q11; "
            "(4) ASD without IEM: normal metabolites; normal purine studies; "
            "(5) ATIC deficiency: AICA-ribosiduria (AICAriboside in urine); very similar phenotype — "
            "distinguish by urine organic acids + purine studies; AICAriboside vs succinyladenosine"
        ),
        "diet_treatment": "No proven specific treatment. D-ribose supplementation (0.5-1g/kg/day) — limited evidence. Allopurinol (reduces de novo purine flux, potentially decreasing SAICAR load) — small uncontrolled series, controversial. Seizure management (standard AEDs; partial control). Early developmental intervention, speech/occupational therapy. Nutritional support for wasting phenotype.",
        "gene_therapy_status": "No approved or clinical-stage therapy. Small enzyme (484aa) suitable for AAV delivery. No gene therapy trials as of 2025. D-ribose and dietary modifications have been tried empirically without definitive benefit. Substrate reduction via allopurinol theoretically reduces SAICAR but no RCT.",
        "critical_ci": (
            "CRITICAL: (1) Missing succinyladenosine in CSF — MUST check CSF for SAICAR/SA "
            "in any unexplained epileptic encephalopathy + autism; urine Bratton-Marshall test as screen; "
            "(2) Diagnosing as primary autism without metabolic workup — screen urine purines; "
            "(3) Expecting allopurinol to be curative — evidence very limited; "
            "(4) Confusing ADSL with ATIC: both have purine accumulation + autism; "
            "distinguish by specific metabolites (SA vs AICAriboside)"
        ),
        "nbs_marker": "Not in standard NBS. Urine organic acids (succinyladenosine detectable). Urine Bratton-Marshall test (succinyl compounds give positive colour). CSF: SAICAR, SAICAr, succinyladenosine — gold-standard diagnosis. ADSL sequencing.",
        "key_biomarker": "CSF succinyladenosine (SA): absent in normal, elevated in ADSL deficiency (pathognomonic). CSF SAICAR elevated. Urine succinyladenosine and SAICAR elevated. Urine Bratton-Marshall test: positive. RBC ADSL enzyme activity reduced. Plasma uric acid: may be mildly low (reduced de novo purine flux).",
        "severity_spectrum": "Neonatal severe (fatal, burst suppression EEG) → Profound ID + refractory seizures + autism (most common) → Mild-moderate ID with autistic features +/- seizures → Rare mild phenotype (mild DD). Genotype-phenotype correlation: R426H/R426H → severe; mild missense compound hets → moderate.",
        "founder_variant": "p.Arg426His — Northern European (Belgian, Dutch); associated with severe phenotype. Otherwise no pan-ethnic founder.",
        "key_variants": [
            "p.Arg426His — Northern European; severe profound ID phenotype",
            "p.Ser395Arg — moderate phenotype",
            "p.Ala269Val — partial function; mild-moderate",
            "p.Trp57Cys — severe neonatal form",
            "p.Tyr114Cys — moderate phenotype",
        ],
        "seed": SEED_BASE + 2,
    },
    # ── PNP — Purine nucleoside phosphorylase deficiency ──────────────────────────
    {
        "gene": "PNP", "alias": "PNP — PNP deficiency (T-cell immunodeficiency + autoimmune hemolytic anemia; HSCT curative)",
        "aa": "289 aa", "kDa": "32 kDa",
        "gene_class": "Purine catabolism enzyme (purine nucleoside phosphorylase)",
        "pp_subgroup": "Purine catabolism pathway (ADA · PNP · XDH)",
        "locus": "14q11.2", "omim_gene": 164050,
        "phenotype": "Progressive T-cell immunodeficiency (CD3 low; B-cells preserved); autoimmune hemolytic anemia; spastic diplegia; deoxyGTP toxic to T-cells; HSCT curative",
        "disease": (
            "PNP biallelic loss → PNP deficiency (OMIM #613179). PNP encodes purine nucleoside phosphorylase "
            "(289aa, 32kDa, homotrimeric), which converts inosine → hypoxanthine + ribose-1-P, "
            "guanosine → guanine + ribose-1-P, and analogously for deoxy-nucleosides. "
            "Without PNP, deoxyguanosine accumulates → phosphorylated by mitochondrial deoxyguanosine kinase "
            "(DGUOK) → dGTP accumulates in T-cells (T-cells have high DGUOK and low 5'-nucleotidase) → "
            "dGTP inhibits ribonucleotide reductase → T-cell apoptosis (same mechanism as ADA but with dGTP). "
            "T-cell immunodeficiency is progressive (not present at birth; develops in months-years). "
            "B-cells are PRESERVED (contrast ADA-SCID where B-cells also depleted). "
            "Clinical: (1) Recurrent opportunistic infections (PCP, viral — EBV, CMV, HSV, VZV); "
            "(2) Autoimmune complications (autoimmune hemolytic anemia [AIHA] in 50%, autoimmune thrombocytopenia); "
            "(3) Neurological: spastic diplegia, ataxia, intellectual disability (~50%) — unique among immunodeficiencies; "
            "occurs independently of infections (dGTP directly toxic to neurons?); "
            "(4) Low/absent uric acid (PNP is upstream of XDH → less xanthine/uric acid production); "
            "HYPOURICEMIA is a KEY CLUE — uric acid <1 mg/dL. Incidence: very rare (~1/10,000,000)."
        ),
        "inheritance": "Autosomal recessive. PNP 14q11.2. Extremely rare; <100 cases described. No consistent founder allele.",
        "hallmark": (
            "PNP HALLMARKS: "
            "(1) HYPOURICEMIA — KEY DIAGNOSTIC CLUE: plasma uric acid <1 mg/dL "
            "(PNP is upstream of XDH; less substrate for uric acid → hypouricaemia); "
            "same as XDH deficiency but completely different phenotype (immunodeficiency vs kidney stones); "
            "(2) T-CELL IMMUNODEFICIENCY (B-CELLS PRESERVED): CD3/CD4/CD8 low; CD19 normal or elevated; "
            "T-B+NK? immunophenotype (contrast ADA: T-B-NK-); NK cell variably affected; "
            "(3) AUTOIMMUNE HEMOLYTIC ANEMIA (AIHA): Coombs-positive; ~50% of patients; "
            "warm or cold AIHA; unique feature among primary immunodeficiencies; "
            "(4) NEUROLOGICAL FEATURES (spastic diplegia, ataxia): present in ~50%; "
            "unique among primary T-cell immunodeficiencies; mechanism uncertain (dGTP neurotoxicity); "
            "(5) dGTP ACCUMULATION: elevated in T-cells and erythrocytes; "
            "mechanistically identical to ADA/dATP but with deoxyguanosine as substrate; "
            "(6) HSCT CURATIVE for immunological features (allogenic HSCT); "
            "does NOT reliably reverse neurological features (CNS already damaged); "
            "(7) DO NOT GIVE LIVE VACCINES"
        ),
        "key_ddx": (
            "PNP DDx: (1) ADA-SCID: T-B-NK- (all lymphocytes depleted); hyperuricemia normal; "
            "skeletal anomalies; earlier onset; "
            "(2) Common variable immunodeficiency (CVID): hypogammaglobulinaemia; B-cell dysfunction; "
            "onset 20-30s; "
            "(3) XDH deficiency (xanthinuria): also hypouricaemia but NO immunodeficiency, NO AIHA; "
            "xanthine kidney stones; key separator; "
            "(4) SCID variants (RAG, Artemis, JAK3): lymphopenia but different immunophenotype; "
            "no autoimmune features; normal uric acid; "
            "(5) Autoimmune hemolytic anemia (warm): positive Coombs; if also lymphopenic → think PNP"
        ),
        "diet_treatment": "Allogeneic HSCT (matched sibling or matched unrelated donor): curative for immunological features; should be done promptly when donor available. PEGylated PNP enzyme replacement (investigational — has been attempted). IVIG replacement therapy before HSCT. PCP prophylaxis (TMP-SMX). Antiviral prophylaxis. NO live vaccines. Corticosteroids/IVIG for AIHA.",
        "gene_therapy_status": "No approved therapy. HSCT is the standard curative treatment. Gene therapy (lentiviral HSC approach analogous to ADA-SCID) in early development — no trials as of 2025. PEGylated bovine PNP tried as ERT in some patients; reduces deoxyguanosine exposure partially but not curative.",
        "critical_ci": (
            "CRITICAL: (1) Missing hypouricemia — plasma uric acid <1 mg/dL in T-cell immunodeficiency "
            "should immediately suggest PNP deficiency (or XDH); "
            "(2) Live vaccines — contraindicated; "
            "(3) Expecting HSCT to reverse neurological features — INCORRECT; "
            "HSCT corrects immune system; neurological damage from dGTP is not reversed; "
            "(4) Confusing T-B+ (PNP) with T-B- (ADA): immunophenotyping guides diagnosis; "
            "(5) Missing AIHA workup in immunodeficiency — AIHA + lymphopenia = PNP until proven otherwise"
        ),
        "nbs_marker": "Not in standard NBS. TREC screening may detect T-cell lymphopenia (later onset, may miss early). PNP enzyme activity in RBCs: absent in biallelic loss. Elevated deoxyguanosine in plasma/urine. Plasma uric acid <1 mg/dL (hypouricemia). PNP sequencing.",
        "key_biomarker": "Plasma uric acid <1 mg/dL (hypouricemia — key clue). RBC PNP enzyme activity <1% (null). Plasma deoxyguanosine elevated. T-cell lymphopenia (CD3/CD4/CD8 low). B-cells (CD19) preserved — T-B+ pattern. Coombs-positive AIHA (50%). dGTP elevated in RBCs.",
        "severity_spectrum": "Severe early-onset T-cell immunodeficiency (null/null, early infancy) → Progressive T-cell deficiency (later onset, childhood) with AIHA + neuro → Mild partial immunodeficiency (residual PNP activity). All associated with hypouricemia.",
        "founder_variant": "No major founder. Very rare globally. Missense and nonsense variants throughout gene. Arg234Pro described in multiple families.",
        "key_variants": [
            "p.Arg234Pro — recurrent; complete loss; severe immunodeficiency",
            "p.Arg234Gln — complete loss; severe",
            "p.Tyr160Cys — partial function; milder phenotype",
            "Exon deletions — rare; complete loss",
            "p.Ile197Ser — complete loss; classic phenotype",
        ],
        "seed": SEED_BASE + 3,
    },
    # ── ATIC — AICA-ribosiduria / de Brouwer syndrome ─────────────────────────────
    {
        "gene": "ATIC", "alias": "ATIC — AICA-ribosiduria / de Brouwer syndrome (bifunctional de novo purine enzyme; brain malformations)",
        "aa": "592 aa", "kDa": "65 kDa",
        "gene_class": "Purine de novo synthesis enzyme (bifunctional AICAR formyltransferase/IMP cyclohydrolase)",
        "pp_subgroup": "Purine de novo synthesis pathway (ADSL · ATIC)",
        "locus": "2q35", "omim_gene": 601312,
        "phenotype": "AICA-ribosiduria: AICAriboside in urine + severe ID + brain malformations + autistic features; bifunctional enzyme (AICAR formyltransferase + IMP cyclohydrolase); extremely rare",
        "disease": (
            "ATIC biallelic loss → AICA-ribosiduria (de Brouwer syndrome, OMIM #608381). "
            "ATIC encodes a bifunctional enzyme (592aa, 65kDa): (1) AICAR transformylase (ATIC-T domain): "
            "converts AICAR + 10-formyl-THF → FAICAR + THF (step 9 of de novo purine synthesis, "
            "requiring folate); and (2) IMP cyclohydrolase (ATIC-C domain): FAICAR → IMP + H₂O (step 10). "
            "Loss of ATIC → AICAR accumulates → dephosphorylated to AICAriboside (AICA-riboside, "
            "also known as acadesine or AICAR) → excreted in urine. AICAriboside is a potent AMPK activator "
            "(used experimentally in metabolic research); chronic AMPK activation in neurons may contribute to "
            "neurotoxicity. Clinical (described first by de Brouwer 2010, extremely rare — <20 cases): "
            "profound intellectual disability, epilepsy, autistic features, brain malformations on MRI "
            "(hypomyelination, simplified gyral pattern, corpus callosum abnormalities), dysmorphic features "
            "(broad forehead, hypertelorism, large ears). Also note: AICAR accumulation ALSO inhibits AMP deaminase "
            "and interferes with adenylate energy charge. No specific treatment."
        ),
        "inheritance": "Autosomal recessive. ATIC 2q35. Extremely rare (<20 cases globally). Consanguineous families over-represented.",
        "hallmark": (
            "ATIC HALLMARKS: "
            "(1) AICAriboside IN URINE — PATHOGNOMONIC: AICA-riboside (acadesine, AICAR dephosphorylated) "
            "is absent in normal urine; its presence = ATIC deficiency (no other condition); "
            "detected by urine organic acid or purine analysis (not standard OA — requires targeted assay); "
            "(2) BIFUNCTIONAL ENZYME — BOTH DOMAINS AFFECTED: step 9 (AICAR formyltransferase) "
            "AND step 10 (IMP cyclohydrolase) both fail; unique double block in de novo pathway; "
            "(3) BRAIN MALFORMATIONS: hypomyelination, corpus callosum dysgenesis, simplified gyri — "
            "more structural than ADSL (which is predominantly functional-metabolic encephalopathy); "
            "(4) DYSMORPHIC FEATURES: broad forehead, hypertelorism, large ears, wide-spaced eyes — "
            "dysmorphic features are more prominent than in ADSL; "
            "(5) AICAR IS AN AMPK ACTIVATOR: AICAriboside (AICAR) activates AMPK (AMP-activated kinase) "
            "mimicking energy depletion — chronic AMPK activation may disturb myelination/neuronal metabolism; "
            "(6) EXTREMELY RARE: fewer than 20 cases in literature; "
            "(7) NO APPROVED TREATMENT: AICAR accumulation cannot yet be reduced pharmacologically"
        ),
        "key_ddx": (
            "ATIC DDx: (1) ADSL deficiency: succinyladenosine in CSF (not AICAriboside); "
            "autism + seizures but LESS structural brain malformations; "
            "urine: SA vs AICAriboside — key discriminator; "
            "(2) Other epileptic encephalopathies with hypomyelination: "
            "Pelizaeus-Merzbacher (PLP1, X-linked, nystagmus, males), "
            "POLR3 deficiency (cerebellar atrophy + hypomyelination), "
            "AIMP1/EIF2B deficiencies; "
            "(3) Angelman/Rett: chromosomal/gene testing; normal purines; "
            "(4) CDG syndromes: transferrin isoelectrofocusing abnormal; different metabolites; "
            "(5) Folate-responsive disorders: AICAR formyltransferase needs 10-formyl-THF; "
            "folinic acid supplementation theoretically could partially restore step 9 — unproven"
        ),
        "diet_treatment": "No proven specific treatment. Folinic acid supplementation (10-formyl-THF donor) has theoretical basis for restoring AICAR formyltransferase step; empirical trials in few patients — no definitive benefit proven. Seizure management. Early developmental intervention. Standard AEDs for epilepsy.",
        "gene_therapy_status": "No approved or investigational therapy. Protein small enough for AAV delivery (592aa = 1776bp). No animal model with complete characterisation. Folinic acid supplementation is the only rationale-based empirical trial used clinically.",
        "critical_ci": (
            "CRITICAL: (1) Missing AICAriboside in urine — standard OA screens often miss it; "
            "must request targeted purine/AICA analysis in unexplained epileptic encephalopathy with brain malformations; "
            "(2) Confusing with ADSL — both have purine accumulation + autism/epilepsy; "
            "succinyladenosine (ADSL) vs AICAriboside (ATIC) — completely different metabolites; "
            "(3) Folate supplementation without monitoring — high-dose folate may worsen seizures in some IEM; "
            "(4) Missing consanguinity workup — most reported cases are consanguineous"
        ),
        "nbs_marker": "Not in standard NBS. Targeted urine purine analysis: AICAriboside (AICA-riboside) detectable by HPLC or LC-MS/MS. Standard urine OA often negative — specialised test required. ATIC sequencing confirms.",
        "key_biomarker": "Urine AICAriboside (AICA-riboside/acadesine): pathognomonic — absent in normal; elevated in ATIC deficiency. Plasma AICAR may be elevated. Brain MRI: hypomyelination + corpus callosum abnormalities + simplified gyri. ATIC enzyme activity in RBCs (reduced). ATIC sequencing.",
        "severity_spectrum": "All reported cases: profound ID + epilepsy + brain malformations. No mild form described (may be lethal prenatally at null/null). Intermediate forms unknown due to rarity.",
        "founder_variant": "No founder allele. Very rare. Consanguineous families predominate. p.Thr116Pro (first described de Brouwer case), homozygous.",
        "key_variants": [
            "p.Thr116Pro — first described homozygous case (de Brouwer 2010); complete loss",
            "Homozygous missense in consanguineous families — pattern consistent with complete loss",
            "No recurrent allele identified — too rare",
        ],
        "seed": SEED_BASE + 4,
    },
    # ── APRT — Adenine phosphoribosyltransferase deficiency ──────────────────────
    {
        "gene": "APRT", "alias": "APRT — APRT deficiency (2,8-DHA nephrolithiasis; allopurinol CURATIVE)",
        "aa": "179 aa", "kDa": "19.5 kDa",
        "gene_class": "Purine salvage enzyme (APRT)",
        "pp_subgroup": "Purine salvage pathway (HPRT1 · APRT)",
        "locus": "16q24.3", "omim_gene": 102600,
        "phenotype": "2,8-dihydroxyadenine (2,8-DHA) nephrolithiasis + nephropathy; allopurinol + dietary adenine restriction CURATIVE; 2,8-DHA stones are RADIOLUCENT (do not calcify, miss on plain X-ray)",
        "disease": (
            "APRT biallelic loss → APRT deficiency (OMIM #102600). APRT encodes adenine "
            "phosphoribosyltransferase (179aa, 19.5kDa, dimeric), which salvages free adenine: "
            "adenine + PRPP → AMP. Without APRT, free adenine cannot be salvaged → adenine oxidised "
            "by XDH: adenine → 2-hydroxyadenine → 2,8-dihydroxyadenine (2,8-DHA). "
            "2,8-DHA is EXTREMELY INSOLUBLE (pKa solubility very low) → crystallises in renal tubules → "
            "2,8-DHA nephrolithiasis, tubular obstruction, crystal nephropathy, end-stage renal disease (ESRD). "
            "KEY: 2,8-DHA stones are RADIOLUCENT (not calcium; X-ray negative); diagnosed by CT/ultrasound "
            "or stone composition analysis. 2,8-DHA crystals in urine sediment (brown, birefringent under "
            "polarised light) are pathognomonic. Type I APRT deficiency (null, complete loss, Northern European): "
            "recurrent stones from childhood. Type II APRT deficiency (partial — p.Met136Thr; Japanese, Icelandic): "
            "partial enzyme activity; slower course; recurrent stones into adulthood."
        ),
        "inheritance": "Autosomal recessive. APRT 16q24.3. Type I (null, Northern European + pan-ethnic). Type II (partial; p.Met136Thr; prevalent in Japan and Iceland).",
        "hallmark": (
            "APRT HALLMARKS: "
            "(1) 2,8-DHA STONES — RADIOLUCENT: 2,8-dihydroxyadenine does NOT calcify → X-ray negative stones; "
            "CT abdomen-pelvis (non-contrast) or renal ultrasound detects; plain abdominal X-ray MISSES; "
            "critical point for nephrolithiasis workup; "
            "(2) URINE CRYSTAL MORPHOLOGY — PATHOGNOMONIC: 2,8-DHA crystals are round/oval, "
            "yellowish-brown, birefringent under polarised light in urine sediment; "
            "characteristic shape distinguishes from uric acid crystals; "
            "(3) ALLOPURINOL IS CURATIVE: allopurinol inhibits XDH → blocks adenine oxidation to 2,8-DHA "
            "(adenine → 2-hydroxyadenine → 2,8-DHA step inhibited); "
            "adenine cannot be converted; accumulated adenine excreted directly in urine as adenine "
            "(less toxic); prevents stone recurrence; may allow partial renal recovery; "
            "(4) DIETARY ADENINE RESTRICTION: reduce high-adenine foods (organ meats, mushrooms, beer); "
            "combined with allopurinol = complete prevention; "
            "(5) NORMAL URIC ACID: plasma uric acid is NORMAL (adenine is different from guanine/hypoxanthine pathway); "
            "distinguishes from HPRT1 (high uric acid) and XDH (low uric acid); "
            "(6) TYPE II (p.Met136Thr, Japanese/Icelandic): partial enzyme; stones onset later; "
            "same treatment"
        ),
        "key_ddx": (
            "APRT DDx: (1) Uric acid nephrolithiasis: uric acid stones also radiolucent; "
            "plasma/urine uric acid ELEVATED in gout; NORMAL in APRT; "
            "stone composition analysis: uric acid vs 2,8-DHA (infrared spectroscopy of stone); "
            "(2) Cystinuria: radiolucent stones, hexagonal crystals; cystine elevated in urine; "
            "(3) Calcium oxalate stones: radiopaque; elevated oxalate; "
            "(4) Primary hyperoxaluria: recurrent stones from childhood; oxalate elevated; "
            "(5) HPRT1/LNS: uric acid stones; hyperuricemia; neurological features; "
            "2,8-DHA stones vs uric acid stones — stone composition analysis key"
        ),
        "diet_treatment": "Allopurinol: 5-10 mg/kg/day (XO inhibitor blocks adenine → 2,8-DHA conversion) — prevents stone formation and nephropathy. Dietary adenine restriction: limit organ meats (liver, kidney), yeast extract, beer, mushrooms. High fluid intake (>3L/day in adults). Urinary alkalinisation NOT helpful (2,8-DHA insoluble across pH range). Febuxostat alternative if allopurinol not tolerated. Renal transplantation for ESRD — recurrence in graft if allopurinol not continued.",
        "gene_therapy_status": "No gene therapy needed — allopurinol + diet is highly effective. Renal transplant may be needed for ESRD. After transplant, allopurinol mandatory to prevent graft recurrence.",
        "critical_ci": (
            "CRITICAL: (1) Plain abdominal X-ray to diagnose stones — MISSES 2,8-DHA (radiolucent); "
            "always use CT or ultrasound; "
            "(2) Missing APRT in stone workup — stone composition analysis (infrared spectroscopy) "
            "identifies 2,8-DHA; mandatory for all recurrent stone patients; "
            "(3) Urinary alkalinisation — does NOT dissolve 2,8-DHA (unlike uric acid stones); "
            "may give false sense of treatment; "
            "(4) Missing ESRD risk — untreated APRT deficiency → ESRD; "
            "(5) Renal transplant without continuing allopurinol → graft recurrence"
        ),
        "nbs_marker": "Not in standard NBS. Stone analysis (infrared spectroscopy): 2,8-DHA identified. Urine microscopy: 2,8-DHA crystals (round, brownish, birefringent). Urine purine analysis: 2,8-DHA, 8-hydroxyadenine, adenine in urine. RBC APRT enzyme activity: <1% (Type I) or 10-30% (Type II). APRT sequencing.",
        "key_biomarker": "Urine 2,8-DHA (elevated; by HPLC or LC-MS/MS). Urine 8-hydroxyadenine (elevated). Stone analysis: 2,8-DHA confirmed by infrared spectroscopy. Urine sediment: round brownish birefringent crystals. Plasma uric acid NORMAL. RBC APRT enzyme activity reduced/absent.",
        "severity_spectrum": "Type I (null, complete loss): recurrent stones from childhood/infancy; ESRD by adulthood if untreated → Type II (p.Met136Thr, 10-30% residual activity): recurrent stones onset childhood-adulthood; ESRD less common but occurs. Both respond completely to allopurinol + dietary restriction.",
        "founder_variant": "p.Met136Thr — Japanese and Icelandic enriched; Type II partial deficiency. Type I null: frame-shifts, early stops — pan-ethnic. No single European founder.",
        "key_variants": [
            "p.Met136Thr — Japanese/Icelandic; Type II partial deficiency; common allele",
            "p.Trp98Ter — Type I null; Northern European",
            "p.Gln22Ter — Type I null; loss-of-function",
            "c.407+3A>G — splice donor; Type I null",
            "p.Asp65Val — partial function; intermediate",
        ],
        "seed": SEED_BASE + 5,
    },
    # ── XDH — Xanthinuria type I ──────────────────────────────────────────────────
    {
        "gene": "XDH", "alias": "XDH — Xanthinuria type I (xanthine stones; allopurinol ABSOLUTELY CONTRAINDICATED)",
        "aa": "1335 aa", "kDa": "150 kDa",
        "gene_class": "Purine catabolism enzyme (xanthine dehydrogenase/oxidase)",
        "pp_subgroup": "Purine catabolism pathway (ADA · PNP · XDH)",
        "locus": "2p23.1", "omim_gene": 607633,
        "phenotype": "Xanthine nephrolithiasis + myopathy (xanthine deposition in muscle); HYPOURICEMIA; allopurinol ABSOLUTELY CONTRAINDICATED (blocks the ONLY xanthine catabolism pathway → complete xanthine accumulation)",
        "disease": (
            "XDH biallelic loss → Xanthinuria type I (OMIM #278300). XDH encodes xanthine dehydrogenase "
            "(1335aa, 150kDa, homodimer with FAD, molybdopterin, and Fe-S cofactors), the enzyme that converts "
            "hypoxanthine → xanthine → uric acid (final two steps of purine catabolism). Also exists "
            "as xanthine oxidase (XO) form. Without XDH/XO, purines accumulate as xanthine and hypoxanthine "
            "in blood, urine, and tissues. KEY: URIC ACID IS ABSENT (or near-absent, <1 mg/dL). "
            "Xanthine is only slightly more soluble than uric acid → xanthine nephrolithiasis, "
            "myopathy (xanthine crystals in muscle), rarely arthropathy. "
            "ALLOPURINOL IS ABSOLUTELY CONTRAINDICATED: allopurinol (and its active metabolite oxipurinol) "
            "inhibit XDH — the ONLY pathway for xanthine catabolism. In normal people, allopurinol reduces "
            "xanthine → uric acid conversion (this is the therapeutic effect in gout). "
            "In XDH deficiency: this pathway DOES NOT EXIST — xanthine is already accumulating. "
            "Giving allopurinol in xanthinuria would block any residual XDH and also inhibit the cofactor "
            "binding — but more critically, clinicians must not confuse xanthine stones (low uric acid) "
            "with uric acid stones (high uric acid) and prescribe allopurinol. "
            "Xanthinuria type II: MOCOS (molybdenum cofactor sulfurase) deficiency — both XDH AND AO "
            "(aldehyde oxidase) affected because MOCOS sulfurates the molybdopterin cofactor of both "
            "enzymes. Type I = XDH mutation only (AO normal); Type II = MOCOS mutation (XDH + AO both reduced)."
        ),
        "inheritance": "Autosomal recessive. XDH 2p23.1. Type I (XDH gene mutation; isolated xanthine oxidase deficiency; AO normal). Type II (MOCOS mutation; XDH + AO both reduced).",
        "hallmark": (
            "XDH HALLMARKS: "
            "(1) ALLOPURINOL ABSOLUTELY CONTRAINDICATED: "
            "allopurinol inhibits XDH = the enzyme that IS DEFICIENT; "
            "giving allopurinol to a xanthinuria patient provides no benefit (XDH already non-functional) "
            "and could worsen symptoms (blocks residual activity in Type I with partial function); "
            "CLINICIANS MUST NOT DIAGNOSE XANTHINURIA AS GOUT AND PRESCRIBE ALLOPURINOL; "
            "the hypouricemia DISTINGUISHES from gout (hyperuricemia); "
            "(2) HYPOURICEMIA (plasma uric acid <1 mg/dL): XDH makes uric acid; loss → no uric acid; "
            "same finding as PNP deficiency but completely different phenotype (no immunodeficiency in XDH); "
            "(3) XANTHINE STONES — RADIOLUCENT: xanthine does not calcify; "
            "CT or ultrasound required; plain X-ray misses; "
            "stone composition analysis: xanthine (yellow-brown, crystalline on stone analysis); "
            "(4) XANTHINE IN URINE: elevated xanthine + hypoxanthine in urine (by HPLC or LC-MS/MS); "
            "uric acid very low or absent; "
            "(5) MUSCLE DEPOSITION: xanthine crystals in muscle → myopathy, myalgia, creatine kinase elevated in ~30%; "
            "(6) MOSTLY BENIGN: majority asymptomatic; detected incidentally on low uric acid; "
            "stones are the main morbidity; "
            "(7) TYPE I vs TYPE II: AO (allopurinol metabolism) intact in Type I → "
            "allopurinol can still be metabolised to oxipurinol by AO; "
            "Type II (MOCOS): AO also absent → allopurinol clearance altered; "
            "allopurinol still contraindicated in both types"
        ),
        "key_ddx": (
            "XDH DDx: (1) PNP deficiency: also hypouricemia; but T-cell immunodeficiency + AIHA; "
            "urine deoxyguanosine elevated; no xanthine stones; "
            "(2) Molybdenum cofactor deficiency (MOCS1/MOCS2/GPHN): "
            "both XDH AND sulfite oxidase (SUOX) impaired; "
            "severe neonatal seizures, lens dislocation, xanthine + sulfite in urine; "
            "xanthinuria + sulfocysteine (key additional marker); LIFE-THREATENING; "
            "(3) Uric acid stones: hyperuricemia (opposite); gout; allopurinol HELPS (opposite); "
            "(4) APRT (2,8-DHA stones): also radiolucent; normal uric acid; 2,8-DHA in urine; "
            "(5) Xanthinuria type II (MOCOS): identical phenotype; AO also absent; "
            "type distinguished by AO activity testing (metabolise allopurinol differently)"
        ),
        "diet_treatment": "NO allopurinol (ABSOLUTELY CONTRAINDICATED). High fluid intake (>3L/day in adults): dilutes xanthine → prevents crystallisation. Low purine diet (reduce organ meats, sardines, yeast products → less hypoxanthine/xanthine substrate). Alkalinisation of urine: increases xanthine solubility slightly (weak effect; xanthine solubility improves with alkaline pH, unlike uric acid which also improves). Urinary alkalinisation (citrate, bicarbonate) may help mild cases. Stone removal as needed (percutaneous nephrolithotomy, ureteroscopy).",
        "gene_therapy_status": "No gene therapy. Large gene (XDH 1335aa = 4005bp) — challenging for AAV delivery. Most patients need only dietary management + high fluid intake. No drug therapies targeting xanthine clearance available.",
        "critical_ci": (
            "CRITICAL: (1) ALLOPURINOL — ABSOLUTELY CONTRAINDICATED in xanthinuria; "
            "do not give to any patient with hypouricemia + xanthine stones; "
            "(2) Missing hypouricemia — measure uric acid in all stone patients; "
            "xanthinuria detected by very low uric acid + xanthine in urine; "
            "(3) Diagnosing as gout — gout has hyperuricemia; xanthinuria has hypouricemia; "
            "opposite; "
            "(4) Molybdenum cofactor deficiency: also has xanthinuria + sulfite in urine; "
            "sulfocysteine elevated; neonatal seizures; severe — do NOT miss sulfocysteine; "
            "(5) Plain X-ray for stone diagnosis — misses radiolucent xanthine stones"
        ),
        "nbs_marker": "Not in standard NBS. Plasma uric acid near-zero (<1 mg/dL) on biochemistry (often incidental finding). Urine xanthine (elevated) + uric acid (absent/trace) by HPLC/LC-MS/MS. Stone composition analysis: xanthine. Urine sulfocysteine NORMAL (distinguishes from MoCoD). XDH sequencing.",
        "key_biomarker": "Plasma uric acid <1 mg/dL (hypouricemia — key clue). Urine xanthine elevated (>300 mg/day; normal <10 mg/day). Urine uric acid near-absent. Urine sulfocysteine NORMAL (critical DDx from molybdenum cofactor deficiency). Stone infrared spectroscopy: xanthine. XDH enzyme activity absent.",
        "severity_spectrum": "Mostly asymptomatic (many patients diagnosed incidentally on low plasma uric acid) → Xanthine nephrolithiasis (acute renal colic, recurrent) → Xanthine myopathy (exercise-induced myalgia, rhabdomyolysis) → Rarely: renal failure from chronic crystal nephropathy. No neurological or immune features.",
        "founder_variant": "No major founder. Pan-ethnic. Missense, nonsense, splicing throughout large XDH gene. Elevated incidence in Japan, Middle East (consanguinity).",
        "key_variants": [
            "p.Arg149Cys — complete loss; recurrent xanthine stones",
            "p.Gln337Ter — nonsense; null; xanthine stones + myopathy",
            "c.IVS4+1G>A — splice donor; null",
            "p.Ala1078Val — partial function; milder",
            "Large exon deletions — described in Japanese patients; complete loss",
        ],
        "seed": SEED_BASE + 6,
    },
    # ── UMPS — Orotic aciduria type I ─────────────────────────────────────────────
    {
        "gene": "UMPS", "alias": "UMPS — UMP synthase deficiency / Orotic aciduria type I (uridine CURATIVE; NO hyperammonemia; DDx OTC)",
        "aa": "480 aa", "kDa": "52 kDa",
        "gene_class": "Pyrimidine de novo synthesis enzyme (bifunctional OPRT + ODCase)",
        "pp_subgroup": "Pyrimidine de novo synthesis pathway (UMPS)",
        "locus": "3q13.2", "omim_gene": 613891,
        "phenotype": "Orotic aciduria type I: megaloblastic anemia + massive orotic acid in urine + growth retardation; NO hyperammonemia (DDx from OTC deficiency); uridine monophosphate supplement CURATIVE",
        "disease": (
            "UMPS biallelic loss → UMP synthase deficiency / Orotic aciduria type I (OMIM #613891). "
            "UMPS encodes a bifunctional enzyme (480aa, 52kDa): (1) Orotate phosphoribosyltransferase (OPRT) "
            "domain: orotate + PRPP → OMP; and (2) OMP decarboxylase (ODCase) domain: OMP → UMP. "
            "These are the final two steps of de novo pyrimidine synthesis. Without UMPS, orotic acid "
            "ACCUMULATES massively (orotic acid = orotate, the substrate for OPRT). "
            "Clinical triad: (1) MEGALOBLASTIC ANEMIA: pyrimidine deficiency → impaired DNA synthesis → "
            "megaloblastic erythropoiesis (macrocytic anemia, hypersegmented neutrophils, marrow megaloblasts); "
            "does NOT respond to B12 or folate (primary enzyme defect, not vitamin deficiency); "
            "(2) OROTIC ACID IN URINE: massive crystalluria; orange/white crystals; "
            "(3) Growth retardation and failure to thrive. "
            "CRITICAL DDx: OTC (ornithine transcarbamylase) deficiency — a UCD — ALSO causes secondary orotic aciduria "
            "(because carbamoyl phosphate shunted from blocked UCD into pyrimidine synthesis → "
            "orotate accumulates). KEY SEPARATOR: OTC deficiency has HYPERAMMONEMIA; UMPS does NOT. "
            "Also: OTC is X-linked, males primarily; UMPS is autosomal recessive. "
            "TREATMENT: Uridine monophosphate (UMP, 50-150 mg/kg/day oral) provides exogenous pyrimidine "
            "→ bypasses the block → normalises DNA synthesis → corrects anemia + orotic aciduria + growth. "
            "HIGHLY EFFECTIVE — essentially curative."
        ),
        "inheritance": "Autosomal recessive. UMPS 3q13.2. Very rare (~50 cases reported). Pan-ethnic. Both sexes equally affected.",
        "hallmark": (
            "UMPS HALLMARKS: "
            "(1) MEGALOBLASTIC ANEMIA NOT RESPONSIVE TO B12/FOLATE: "
            "macrocytic anemia, hypersegmented neutrophils; bone marrow megaloblasts; "
            "B12 and folate levels NORMAL; trial of B12/folate does NOT help; "
            "pyrimidine-deficient megaloblastosis is the mechanism; "
            "(2) OROTIC ACID CRYSTALLURIA: massive orotic acid in urine → crystals may obstruct ureters; "
            "urine dip: urobilinogen-like orange/brown staining; "
            "urine OA chromatography: orotic acid massively elevated; "
            "(3) NO HYPERAMMONEMIA — KEY DDx FROM OTC DEFICIENCY: "
            "OTC deficiency (X-linked UCD) also causes orotic aciduria (secondary) but WITH hyperammonemia; "
            "UMPS has NO hyperammonemia (urea cycle intact); always measure ammonia to distinguish; "
            "(4) URIDINE SUPPLEMENT CURATIVE: oral UMP → enters cells via nucleoside transporters → "
            "bypasses UMPS → restores uridine pool → normalises DNA synthesis; "
            "anemia corrects within weeks; orotic acid normalises; growth normalises; "
            "(5) BIFUNCTIONAL ENZYME — BOTH DOMAINS REQUIRED: OPRT + ODCase; mutations in either domain "
            "cause the same phenotype (orotic acid accumulation = OPRT step blocked, or OMP accumulates if ODCase blocked); "
            "(6) ALSO NOTE: uridine is the treatment — not orotic acid, not OMP; "
            "uridine (a nucleoside) is distinct from UMP but can be converted to UMP inside cells"
        ),
        "key_ddx": (
            "UMPS DDx for orotic aciduria: "
            "(1) OTC deficiency (X-linked UCD): orotic aciduria + HYPERAMMONEMIA (post-prandial or triggered by illness); "
            "X-linked; males primarily; low citrulline; plasma ammonia elevated; "
            "UMPS: normal ammonia — this is the key separator; "
            "(2) Purine nucleoside phosphorylase (PNP) deficiency: hypouricemia; T-cell immunodeficiency; "
            "no orotic aciduria; "
            "(3) B12 deficiency: macrocytic anemia; low B12; responds to B12 (UMPS does not); normal purines; "
            "(4) Folate deficiency: macrocytic anemia; low folate; responds to folic acid; "
            "(5) CPS1 deficiency (UCD): hyperammonemia + orotic aciduria may be mild or absent; "
            "plasma citrulline and argininosuccinate trace/absent; "
            "(6) Allopurinol-induced secondary orotic aciduria: allopurinol inhibits OPRT domain; "
            "transient; resolves with allopurinol cessation; no anemia"
        ),
        "diet_treatment": "Uridine monophosphate (UMP) supplementation: 50-150 mg/kg/day in 3-4 divided doses. Highly effective — corrects anemia, normalises orotic acid, restores growth. Uridine (as nucleoside) also used and absorbed efficiently. No dietary protein restriction needed (unlike UCD). No B12/folate needed (won't help). Monitor urine orotic acid as treatment response marker.",
        "gene_therapy_status": "No gene therapy needed — UMP supplementation is highly effective and essentially curative. Gene therapy research exists in principle (UMPS is 480aa, AAV-deliverable) but clinical need is low given oral UMP therapy works well. Neonatal diagnosis is important to prevent intellectual disability from prolonged anemia/pyrimidine deficiency.",
        "critical_ci": (
            "CRITICAL: (1) Treating with B12/folate — will NOT help UMPS; waste time before correct therapy; "
            "(2) Not measuring ammonia in orotic aciduria — missing OTC deficiency (hyperammonemia can be fatal); "
            "ALWAYS check plasma ammonia in any orotic aciduria; "
            "(3) Confusing with UCD (OTC): OTC = X-linked + hyperammonemia + orotic aciduria; "
            "UMPS = AR + normal ammonia + orotic aciduria; "
            "(4) Delaying UMP therapy — prolonged pyrimidine deficiency → intellectual disability, "
            "immune dysfunction; start UMP as soon as diagnosis confirmed; "
            "(5) Missed diagnosis of orotic acid crystalluria as UTI — orange crystals in nappy/diaper "
            "are often mistaken for blood or UTI; measure urine OA"
        ),
        "nbs_marker": "Not in standard NBS. Orange crystals in urine/nappy (orotic acid). Urine organic acids: orotic acid massively elevated (prominent on OA chromatography). Plasma ammonia: NORMAL (excludes OTC/UCD). CBC: macrocytic anemia, hypersegmented neutrophils. Bone marrow: megaloblasts. B12/folate: NORMAL. UMPS enzyme activity in RBCs (reduced). UMPS sequencing.",
        "key_biomarker": "Urine orotic acid: massively elevated (>10× normal; may be >1000× normal). Plasma ammonia: NORMAL (critical DDx). CBC: macrocytic anemia (MCV>100); hypersegmented neutrophils. Reticulocyte count: low (hypoproliferative). B12 and folate: NORMAL. RBC UMPS enzyme activity: reduced/absent. Response to UMP supplementation: diagnostic and therapeutic.",
        "severity_spectrum": "Classic severe (null/null): neonatal macrocytic anemia + growth failure + orotic crystalluria → Moderate (compound het with partial allele): childhood presentation → Mild (partial UMPS activity, rare): late childhood/adolescent orotic aciduria with minimal anemia. All respond to UMP supplementation regardless of severity.",
        "founder_variant": "No founder allele. Very rare globally. Most cases: homozygous or compound heterozygous missense.",
        "key_variants": [
            "p.Arg96Gln — partial OPRT domain function; moderate phenotype",
            "p.Trp326Ter — nonsense; null; severe classic phenotype",
            "p.Gly213Ser — ODCase domain; orotic acid accumulation (OMP → UMP blocked)",
            "p.Ile152Thr — severe; neonatal presentation",
            "p.Leu353Pro — moderate; compound het presentations",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _make_patients(gene_dict):
    """Generate 40 synthetic patient records for a given PP gene."""
    rng = random.Random(gene_dict["seed"])
    gene = gene_dict["gene"]

    # Phenotypic class probabilities per gene
    PHENO_PROBS = {
        "HPRT1": [0.55, 0.30, 0.15],   # Classic LNS / Partial-neuro / Kelley-Seegmiller
        "ADA":   [0.70, 0.20, 0.10],   # Complete SCID / Delayed-onset / Partial
        "ADSL":  [0.15, 0.65, 0.20],   # Severe neonatal / Moderate / Mild
        "PNP":   [0.60, 0.30, 0.10],   # Severe T-cell deficiency / Moderate / Mild
        "ATIC":  [0.90, 0.10, 0.00],   # Classic / Mild (no mild described — rare)
        "APRT":  [0.55, 0.45, 0.00],   # Type I null / Type II partial
        "XDH":   [0.25, 0.40, 0.35],   # Symptomatic stones / Myopathy-predominant / Asymptomatic
        "UMPS":  [0.60, 0.30, 0.10],   # Classic severe / Moderate / Mild
    }
    CLASS_NAMES = {
        "HPRT1": ["Classic LNS", "Partial neuro", "Kelley-Seegmiller"],
        "ADA":   ["Complete ADA-SCID", "Delayed-onset", "Partial deficiency"],
        "ADSL":  ["Severe neonatal", "Moderate ID+seizures", "Mild ID"],
        "PNP":   ["Severe T-cell deficiency", "Moderate", "Mild"],
        "ATIC":  ["Classic de Brouwer", "Mild", "Asymptomatic"],
        "APRT":  ["Type I null", "Type II partial", "Asymptomatic"],
        "XDH":   ["Symptomatic stones", "Myopathy-predominant", "Asymptomatic"],
        "UMPS":  ["Classic severe", "Moderate", "Mild"],
    }
    probs = PHENO_PROBS.get(gene, [0.50, 0.35, 0.15])
    classes = CLASS_NAMES.get(gene, ["Severe", "Moderate", "Mild"])

    patients = []
    for i in range(40):
        r = rng.random()
        if r < probs[0]:
            pheno = classes[0]
        elif r < probs[0] + probs[1]:
            pheno = classes[1]
        else:
            pheno = classes[2]

        is_severe = (pheno == classes[0])
        is_mod    = (pheno == classes[1])
        is_mild   = (pheno == classes[2])

        # Age at diagnosis (years)
        if gene == "HPRT1":
            age_dx = round(rng.uniform(0.0, 0.5), 1) if is_severe else round(rng.uniform(0.5, 10.0), 1)
        elif gene == "ADA":
            age_dx = round(rng.uniform(0.0, 0.5), 1) if is_severe else round(rng.uniform(0.5, 15.0), 1)
        elif gene == "ADSL":
            age_dx = round(rng.uniform(0.0, 0.3), 1) if is_severe else round(rng.uniform(0.3, 5.0), 1)
        elif gene == "PNP":
            age_dx = round(rng.uniform(0.1, 2.0), 1) if is_severe else round(rng.uniform(1.0, 10.0), 1)
        elif gene == "ATIC":
            age_dx = round(rng.uniform(0.0, 1.0), 1)
        elif gene == "APRT":
            age_dx = round(rng.uniform(1.0, 10.0), 1) if is_severe else round(rng.uniform(10.0, 40.0), 0)
        elif gene == "XDH":
            age_dx = round(rng.uniform(0.0, 50.0) if not is_severe else rng.uniform(5.0, 40.0), 1)
        elif gene == "UMPS":
            age_dx = round(rng.uniform(0.0, 0.5), 1) if is_severe else round(rng.uniform(0.5, 5.0), 1)
        else:
            age_dx = round(rng.uniform(0.5, 20.0), 1)

        # Gene-specific clinical flags
        if gene == "HPRT1":
            has_self_mutilation  = is_severe
            has_chorea_athetosis = is_severe or (is_mod and rng.random() < 0.4)
            has_hyperuricemia    = rng.random() < (0.97 if is_severe or is_mod else 0.75)
            has_gout             = rng.random() < (0.70 if (is_severe or is_mod) else 0.85)
            has_renal_stones     = rng.random() < 0.55
            allopurinol_rx       = rng.random() < 0.85
            has_seizures         = rng.random() < (0.30 if is_severe else 0.05)
            has_id               = is_severe or (is_mod and rng.random() < 0.5)
            sex = "M"  # X-linked; all males
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "has_self_mutilation": has_self_mutilation,
                "has_chorea_athetosis": has_chorea_athetosis,
                "has_hyperuricemia": has_hyperuricemia,
                "has_gout": has_gout,
                "has_renal_stones": has_renal_stones,
                "allopurinol_rx": allopurinol_rx,
                "has_seizures": has_seizures,
                "has_id": has_id,
            })
        elif gene == "ADA":
            has_scid             = is_severe
            has_opportunistic    = rng.random() < (0.85 if is_severe else 0.50 if is_mod else 0.20)
            has_skeletal_anomaly = is_severe and rng.random() < 0.52
            gene_therapy_rx      = rng.random() < (0.30 if is_severe else 0.10)
            peg_ada_rx           = rng.random() < (0.60 if is_severe else 0.20)
            hsct_rx              = rng.random() < (0.55 if is_severe else 0.15)
            has_lymphopenia      = rng.random() < (0.97 if is_severe else 0.70 if is_mod else 0.30)
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": rng.choice(["M","F"]),
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "has_scid": has_scid,
                "has_opportunistic_infection": has_opportunistic,
                "has_skeletal_anomaly": has_skeletal_anomaly,
                "gene_therapy_received": gene_therapy_rx,
                "peg_ada_received": peg_ada_rx,
                "hsct_received": hsct_rx,
                "has_lymphopenia": has_lymphopenia,
            })
        elif gene == "ADSL":
            has_seizures   = rng.random() < (0.95 if is_severe else 0.80 if is_mod else 0.40)
            has_autism     = rng.random() < (0.60 if is_severe else 0.75 if is_mod else 0.65)
            has_id         = True  # all affected have ID in described cases
            csf_sa_pos     = rng.random() < 0.97  # succinyladenosine in CSF virtually universal
            has_cachexia   = is_severe and rng.random() < 0.65
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": rng.choice(["M","F"]),
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "has_seizures": has_seizures,
                "has_autism_features": has_autism,
                "has_id": has_id,
                "csf_succinyladenosine_positive": csf_sa_pos,
                "has_cachexia": has_cachexia,
            })
        elif gene == "PNP":
            has_t_cell_deficiency = rng.random() < (0.97 if is_severe else 0.80 if is_mod else 0.50)
            has_aiha              = rng.random() < 0.50
            has_neuro             = rng.random() < (0.55 if is_severe else 0.35)
            hsct_rx               = rng.random() < 0.55
            has_opportunistic     = rng.random() < (0.75 if is_severe else 0.45)
            has_spastic_diplegia  = rng.random() < (0.50 if is_severe else 0.25)
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": rng.choice(["M","F"]),
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "has_t_cell_deficiency": has_t_cell_deficiency,
                "has_aiha": has_aiha,
                "has_neurological": has_neuro,
                "hsct_received": hsct_rx,
                "has_opportunistic_infection": has_opportunistic,
                "has_spastic_diplegia": has_spastic_diplegia,
            })
        elif gene == "ATIC":
            has_brain_malformation = rng.random() < 0.90
            has_seizures           = rng.random() < 0.85
            has_autism             = rng.random() < 0.80
            has_dysmorphic         = rng.random() < 0.85
            aica_urine_pos         = rng.random() < 0.97
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": rng.choice(["M","F"]),
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "has_brain_malformation": has_brain_malformation,
                "has_seizures": has_seizures,
                "has_autism_features": has_autism,
                "has_dysmorphic": has_dysmorphic,
                "aica_riboside_in_urine": aica_urine_pos,
            })
        elif gene == "APRT":
            has_kidney_stones = rng.random() < (0.90 if is_severe else 0.70)
            has_esrd          = rng.random() < (0.30 if is_severe else 0.15)
            allopurinol_rx    = rng.random() < 0.75
            dha_crystals      = rng.random() < (0.85 if is_severe else 0.65)
            type_label        = "Type I" if is_severe else "Type II"
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": rng.choice(["M","F"]),
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "aprt_type": type_label,
                "has_kidney_stones": has_kidney_stones,
                "has_esrd": has_esrd,
                "allopurinol_rx": allopurinol_rx,
                "has_dha_crystals_in_urine": dha_crystals,
            })
        elif gene == "XDH":
            has_kidney_stones = rng.random() < (0.75 if is_severe else 0.40 if is_mod else 0.10)
            has_myopathy      = rng.random() < (0.20 if is_severe else 0.45 if is_mod else 0.05)
            hypouricemia      = rng.random() < 0.97
            asymptomatic      = is_mild
            allopurinol_ci    = True  # always CI in XDH
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": rng.choice(["M","F"]),
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "has_kidney_stones": has_kidney_stones,
                "has_myopathy": has_myopathy,
                "has_hypouricemia": hypouricemia,
                "is_asymptomatic": asymptomatic,
                "allopurinol_contraindicated": allopurinol_ci,
            })
        elif gene == "UMPS":
            has_megaloblastic_anemia = rng.random() < (0.97 if is_severe else 0.80 if is_mod else 0.50)
            has_orotic_crystalluria  = rng.random() < (0.90 if is_severe else 0.70)
            has_growth_retardation   = rng.random() < (0.90 if is_severe else 0.65)
            uridine_rx               = rng.random() < 0.85
            ammonia_normal           = rng.random() < 0.97  # always normal in UMPS
            b12_folate_normal        = rng.random() < 0.97
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": rng.choice(["M","F"]),
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "has_megaloblastic_anemia": has_megaloblastic_anemia,
                "has_orotic_crystalluria": has_orotic_crystalluria,
                "has_growth_retardation": has_growth_retardation,
                "uridine_rx": uridine_rx,
                "ammonia_normal": ammonia_normal,
                "b12_folate_normal": b12_folate_normal,
            })
        else:
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": rng.choice(["M","F"]),
                "phenotypic_class": pheno, "age_dx_y": age_dx,
            })
    return patients


# ── Populate patient cohorts ──────────────────────────────────────────────────────
for _g in PP_GENES:
    _g["patients"] = _make_patients(_g)
    _g["n_patients"] = len(_g["patients"])

ALL_PATIENTS = [p for g in PP_GENES for p in g["patients"]]


# ─── API: get_overview ───────────────────────────────────────────────────────────
def get_overview():
    total = len(ALL_PATIENTS)

    gene_summary = []
    for g in PP_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "locus": g["locus"],
            "gene_class": g["gene_class"],
            "pp_subgroup": g["pp_subgroup"],
            "n_patients": g["n_patients"],
            "diet_treatment": g["diet_treatment"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })

    return {
        "atlas": "PP-Atlas — Complete 8-Gene Purine & Pyrimidine Metabolism Disorders Atlas",
        "n_genes": len(PP_GENES),
        "n_patients": total,
        "seeds": [g["seed"] for g in PP_GENES],
        "genes_covered": [g["gene"] for g in PP_GENES],
        "gene_subgroups": {
            "Purine salvage pathway (HPRT1 · APRT)": ["HPRT1", "APRT"],
            "Purine catabolism pathway (ADA · PNP · XDH)": ["ADA", "PNP", "XDH"],
            "Purine de novo synthesis pathway (ADSL · ATIC)": ["ADSL", "ATIC"],
            "Pyrimidine de novo synthesis pathway (UMPS)": ["UMPS"],
        },
        "critical_clinical_rules": [
            "HPRT1/LNS — ALLOPURINOL TREATS GOUT NOT NEURO: allopurinol normalises uric acid and prevents nephrolithiasis/renal failure; does NOT reverse choreoathetosis, self-mutilation, or intellectual disability — these are permanent from basal ganglia dopamine deficit; start allopurinol early for renal protection; counsel families honestly about neurological prognosis",
            "ADA-SCID — NO LIVE VACCINES, NO NON-IRRADIATED BLOOD: BCG given at birth in undiagnosed ADA-SCID → disseminated fatal BCG; rotavirus, varicella, MMR all contraindicated until immune reconstitution; non-irradiated blood → transfusion-associated GVHD; irradiated/CMV-negative products mandatory",
            "ADA-SCID — FIRST GENE THERAPY DISEASE: Strimvelis (EMA 2016) autologous HSC + γ-retroviral ADA vector; OTL-101 lentiviral improved safety; PEGylated ADA (ADAGEN/elapegademase) is bridge therapy only, not curative; gene therapy preferred over allogeneic HSCT when no matched sibling",
            "XDH XANTHINURIA — ALLOPURINOL ABSOLUTELY CONTRAINDICATED: allopurinol inhibits XDH — the enzyme that IS ABSENT in xanthinuria; giving allopurinol provides no benefit and removes any residual activity; key diagnostic clue is HYPOURICEMIA (uric acid <1 mg/dL) — the OPPOSITE of gout; never prescribe allopurinol for low-uric-acid nephrolithiasis",
            "UMPS OROTIC ACIDURIA — NO HYPERAMMONEMIA (DDx OTC): OTC deficiency (X-linked UCD) causes orotic aciduria WITH hyperammonemia; UMPS deficiency causes orotic aciduria WITHOUT hyperammonemia; always measure plasma ammonia in any orotic aciduria to distinguish; OTC is X-linked (males primarily); UMPS is AR (both sexes)",
            "UMPS — URIDINE CURATIVE, B12/FOLATE DO NOT HELP: megaloblastic anemia in UMPS does NOT respond to B12 or folate (B12/folate levels are normal; this is pyrimidine-deficient megaloblastosis, not B12/folate deficiency); oral UMP 50-150 mg/kg/day corrects anemia within weeks, normalises orotic acid, restores growth",
            "APRT DEFICIENCY — 2,8-DHA STONES ARE RADIOLUCENT: plain abdominal X-ray misses 2,8-DHA stones; always use CT or ultrasound; stone composition analysis (infrared spectroscopy) identifies 2,8-DHA; allopurinol is CURATIVE (blocks XDH → prevents adenine → 2,8-DHA conversion); normal uric acid (unlike gout)",
            "PNP DEFICIENCY — HYPOURICEMIA + T-CELL DEFICIENCY + AIHA: triad pathognomonic; hypouricemia because PNP is upstream of XDH (less substrate); T-B+ immunophenotype (B-cells preserved, unlike ADA-SCID T-B-); autoimmune hemolytic anemia (Coombs-positive) in 50%; spastic diplegia and ataxia unique among primary immunodeficiencies; HSCT curative for immune phenotype, NOT neuro",
            "ADSL DEFICIENCY — SUCCINYLADENOSINE IN CSF PATHOGNOMONIC: SA absent in normal CSF; present only in ADSL deficiency; urine Bratton-Marshall test screens; profound ID + autism + seizures; NO hyperammonemia; no specific therapy; do not confuse with ATIC (AICAriboside in urine) or UCD (hyperammonemia)",
            "ATIC DEFICIENCY — AICAriboside IN URINE PATHOGNOMONIC: acadesine/AICA-riboside absent in normal urine; standard OA screen may miss; targeted purine analysis required; extreme rarity (<20 cases); brain malformations more structural than ADSL; folinic acid supplementation empirical (theoretical basis: step 9 needs 10-formyl-THF)",
        ],
        "gene_summary": gene_summary,
    }


# ─── API: get_breakdown ──────────────────────────────────────────────────────────
def get_breakdown():
    gene_rows = []
    for g in PP_GENES:
        pts = g["patients"]
        gene_rows.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "gene_class": g["gene_class"],
            "pp_subgroup": g["pp_subgroup"],
            "n_patients": g["n_patients"],
            "seed": g["seed"],
            "phenotype": g["phenotype"],
            "inheritance": g["inheritance"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "diet_treatment": g["diet_treatment"],
            "gene_therapy_status": g["gene_therapy_status"],
            "critical_ci": g["critical_ci"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "key_variants": g["key_variants"],
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })
    return {
        "genes": gene_rows,
        "total": len(PP_GENES),
        "total_patients": len(ALL_PATIENTS),
    }


# ─── API: get_definitions ─────────────────────────────────────────────────────
def get_definitions():
    return {
        "atlas": "PP-Atlas — Complete 8-Gene Purine & Pyrimidine Metabolism Disorders Atlas",
        "pp_overview": {
            "full_name": "Purine & Pyrimidine Metabolism Disorders — inherited defects in synthesis, salvage, or catabolism of purines (adenine, guanine) or pyrimidines (uracil, cytosine, thymine)",
            "genes_in_atlas": 8,
            "collective_incidence": "Variable: ~1/380,000 for HPRT1/LNS; ~1/200,000–1,000,000 for ADA-SCID; very rare for PNP, ATIC; ~1/50,000–100,000 for APRT; rare for XDH, UMPS, ADSL",
            "nbs_note": "None in standard NBS panels. ADA-SCID detected by low TRECs on expanded NBS. Others require targeted metabolite testing (uric acid, orotic acid, purine panel).",
        },
        "definitions": [
            {
                "term": "Purine De Novo Synthesis Pathway — 10 Enzymatic Steps",
                "definition": "Purines (adenine + guanine) are synthesised de novo in 10 steps from phosphoribosyl pyrophosphate (PRPP) + glutamine → IMP (inosine monophosphate). Key enzymes relevant to this atlas: ADSL catalyses steps 8 (SAICAR→AICAR) and 10 (adenylosuccinate→AMP); ATIC catalyses steps 9 (AICAR formyltransferase; needs 10-formyl-THF) and 10 (IMP cyclohydrolase). Loss of ADSL → SAICAR and succinyladenosine accumulate. Loss of ATIC → AICAR/AICAriboside accumulates. Both pathways also converge at IMP → AMP (via ADSL) and IMP → XMP → GMP (via IMPDH/GMPS). AMP can be deaminated back to IMP by AMP deaminase (AMPD1). All de novo synthesis requires energy and folate cofactors.",
            },
            {
                "term": "Purine Salvage Pathway — HGPRT and APRT",
                "definition": "The purine salvage pathway recycles free purine bases (from nucleotide catabolism, dietary sources, or cell death) back to nucleoside monophosphates, saving the energetic cost of de novo synthesis (which requires 6 ATP per purine ring). Key enzymes: HGPRT (HPRT1): hypoxanthine + PRPP → IMP; guanine + PRPP → GMP. APRT: adenine + PRPP → AMP. Both require PRPP (phosphoribosyl pyrophosphate), synthesised by PRPS1/2. Without HGPRT: hypoxanthine and guanine cannot be salvaged → must be oxidised by XDH → uric acid overproduction (gout + LNS neurology). Without APRT: adenine cannot be salvaged → adenine oxidised by XDH to 2,8-dihydroxyadenine (2,8-DHA) → insoluble → nephropathy.",
            },
            {
                "term": "Purine Catabolism to Uric Acid — ADA, PNP, XDH Axis",
                "definition": "Catabolism: AMP → adenosine (5'-nucleotidase) → inosine (ADA: deaminates adenosine→inosine) → hypoxanthine (PNP: removes ribose-1-phosphate) → xanthine (XDH step 1: XO/XDH oxidises hypoxanthine→xanthine) → uric acid (XDH step 2: xanthine→uric acid). ADA also deaminates deoxyadenosine → deoxyinosine. PNP also converts guanosine → guanine and deoxyguanosine → deoxyguanine. Loss of ADA → deoxyadenosine/dATP accumulates (T-cell toxicity = SCID). Loss of PNP → deoxyguanosine/dGTP accumulates (T-cell toxicity + AIHA). Loss of XDH → xanthine accumulates (stones, myopathy) + uric acid absent (hypouricemia). All three genes are in the same linear catabolism pathway.",
            },
            {
                "term": "Lesch-Nyhan Syndrome — Basal Ganglia Dopamine Deficit Mechanism",
                "definition": "The neurological features of LNS (choreoathetosis, self-mutilation, intellectual disability) are caused by SELECTIVE vulnerability of dopaminergic neurons to HGPRT loss. Dopaminergic neurons of the basal ganglia (striatum, substantia nigra) have UNIQUELY high purine turnover and UNIQUELY low de novo synthesis capacity → they are entirely dependent on HGPRT for purine recycling. HGPRT loss → dopaminergic neurons cannot maintain adequate purine (ATP/GTP) pools → impaired dopamine synthesis and release. Evidence: (1) DaT (dopamine transporter) SPECT shows markedly reduced striatal DaT binding; (2) Post-mortem: reduced dopaminergic neurons and dopamine in caudate/putamen; (3) L-DOPA has variable/no benefit (receptor downregulation + developmental loss). Allopurinol reduces uric acid but cannot restore dopaminergic purine supply because the salvage enzyme (HGPRT) is absent — this is why neurological features are irreversible and unresponsive to allopurinol.",
            },
            {
                "term": "ADA-SCID — dATP Accumulation Mechanism of T-Cell Toxicity",
                "definition": "ADA deficiency → deoxyadenosine accumulates (not cleared by ADA) → phosphorylated by deoxycytidine kinase (DCK) to dAMP → dADP → dATP. T-lymphocytes are uniquely sensitive because: (1) high DCK activity + low 5'-nucleotidase activity → efficient phosphorylation of deoxyadenosine to dATP; (2) dATP accumulates to >50% of total dNTP pool in T-cells; (3) dATP allosterically inhibits ribonucleotide reductase (RNR), the rate-limiting enzyme of DNA synthesis → dNTP pool imbalance → replication block → apoptosis. Also: adenosine accumulates → inhibits S-adenosylhomocysteine hydrolase (SAHase) → SAH accumulates → inhibits SAM-dependent methylation reactions → lymphocyte dysfunction. B-cells and NK cells are secondarily depleted but less intrinsically sensitive.",
            },
            {
                "term": "Orotic Aciduria — UMPS (Primary) vs OTC Deficiency (Secondary)",
                "definition": "Two completely different diseases both manifest as orotic aciduria: (1) UMPS deficiency (Orotic aciduria type I, AR): primary failure of OPRT/ODCase → orotic acid cannot be converted to OMP → massively elevated urine orotic acid → megaloblastic anemia (pyrimidine deficiency). NO hyperammonemia (urea cycle intact). Both sexes. Treatment: UMP supplementation curative. (2) OTC deficiency (X-linked UCD): ornithine transcarbamylase deficient → carbamoyl phosphate accumulates in mitochondria → enters cytosol → enters pyrimidine synthesis (CPSII pathway) → orotic acid produced in excess → elevated urine orotic acid. WITH hyperammonemia (primary block in urea cycle). Males primarily (X-linked). Treatment: protein restriction + citrulline/arginine + nitrogen scavengers (sodium benzoate/phenylbutyrate). Key separator: plasma ammonia. Allopurinol loading test (allopurinol blocks OPRT → OA accumulation is exaggerated in OTC carriers but NOT in UMPS patients): used to detect OTC carrier females.",
            },
            {
                "term": "2,8-Dihydroxyadenine (2,8-DHA) Nephropathy — APRT",
                "definition": "In APRT deficiency, free adenine cannot be salvaged → oxidised by XDH/XO: adenine → 2-hydroxyadenine → 2,8-dihydroxyadenine (2,8-DHA). 2,8-DHA has extremely low aqueous solubility → precipitates in renal collecting tubules, pelvis, and ureter. Crystals: round/oval, yellow-brown, birefringent under polarised light (electron microscopy: laminated). X-ray NEGATIVE (radiolucent — no calcium; CT and ultrasound required). Urinary excretion: 2,8-DHA + 8-hydroxyadenine + adenine detectable by HPLC/LC-MS/MS. Allopurinol mechanism: inhibits XDH → blocks adenine oxidation → adenine excreted unchanged (safer than 2,8-DHA). Dietary adenine restriction (organ meats, yeast, beer, mushrooms) reduces substrate load. ESRD occurs in untreated patients. After renal transplant, allopurinol must continue to prevent graft recurrence.",
            },
            {
                "term": "Xanthinuria — Type I (XDH) vs Type II (MOCOS) vs Molybdenum Cofactor Deficiency",
                "definition": "Three entities cause xanthinuria (elevated urine xanthine + hypouricemia): (1) Xanthinuria Type I (XDH gene): XDH/XO absent → xanthine accumulates; AO (aldehyde oxidase) NORMAL — patient can still metabolise allopurinol to oxipurinol via AO. Benign in majority. (2) Xanthinuria Type II (MOCOS gene): molybdenum cofactor sulfurase deficiency → BOTH XDH AND AO reduced (MOCOS sulfurates molybdopterin cofactor of both enzymes); patient CANNOT metabolise allopurinol to oxipurinol. Otherwise same clinical phenotype as Type I. (3) Molybdenum cofactor deficiency (MOCS1/MOCS2/GPHN): affects molybdopterin cofactor → XDH + AO + SULFITE OXIDASE (SUOX) all absent. SUOX loss is the lethal component: sulfite accumulates → severe neonatal seizures, lens dislocation, progressive encephalopathy → FATAL or severe handicap; urine: xanthine + SULFOCYSTEINE (pathognomonic). Sulfocysteine absent in Type I/II xanthinuria — key discriminator.",
            },
            {
                "term": "AICAR / AICAriboside — ATIC Deficiency and AMPK Activation",
                "definition": "AICAR (5-aminoimidazole-4-carboxamide ribonucleotide) is a normal de novo purine synthesis intermediate (step 9). ATIC (bifunctional AICAR formyltransferase/IMP cyclohydrolase) converts AICAR to FAICAR (step 9) and then FAICAR to IMP (step 10). Without ATIC: AICAR accumulates → dephosphorylated to AICAriboside (AICA-riboside, acadesine). AICAriboside is a potent AMPK activator — it mimics AMP by activating AMP-activated kinase (AMPK), the master energy sensor. Chronic systemic AMPK activation in developing neurons and glia may impair myelination and neuronal differentiation. Acadesine (AICAriboside) is also used experimentally in metabolic research as an AMPK activator — this pharmacological effect explains some of the neurological toxicity. Detection: urine targeted purine analysis (AICAriboside); standard OA screening often misses it — specialised assay required.",
            },
            {
                "term": "Succinyladenosine and SAICAR — ADSL Deficiency Metabolites",
                "definition": "ADSL catalyses two reactions: (1) SAICAR (succinylaminoimidazole carboxamide ribotide) → AICAR + fumarate (de novo purine step 8); (2) adenylosuccinate → AMP + fumarate (de novo purine step 10). Loss of ADSL → two metabolites accumulate: (a) SAICAR and its dephosphorylated form SAICAr; (b) Succinyladenosine (SA, the dephosphorylated form of adenylosuccinate). Both are detected in CSF and urine. SA in CSF is absent in normals → its presence is pathognomonic for ADSL deficiency. Urine Bratton-Marshall test: succinyl compounds produce a positive colour reaction (red/pink) — used as initial screen. SAICAR and SA are neurotoxic (purinergic receptor modulation, mitochondrial interference proposed). These metabolites are DISTINCT from AICAriboside (ATIC) — both atlases have autism/epilepsy but different specific metabolites.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== PP Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"Subgroups: {list(ov['gene_subgroups'].keys())}")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
    print("OK")
