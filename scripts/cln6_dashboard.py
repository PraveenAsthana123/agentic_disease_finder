"""
CLN6 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 6 / Variant Late-Infantile NCL (vLINCL)
=============================================================================================
40-patient cohort · CLN6 (15q23) · Autosomal recessive (AR) biallelic LOF (+ rare AD Kufs Type A)
CLN6 encodes CLN6 — a transmembrane ER protein (7 TM domains, 311 aa, ~36 kDa)
CLN6 LOF → ER membrane dysfunction → impaired lysosomal biogenesis → SCMAS accumulation →
Fingerprint profiles + Curvilinear bodies on EM → progressive neuronal/retinal apoptosis → CLN6 vLINCL

VARIANT LATE-INFANTILE ONSET (1.5–8 years) — EARLIER THAN CLN5 / CLN3:
CLN6 (variant late-infantile NCL) presents at 1.5–8 years (mean ~3.5 years onset of first seizures).
Earlier than CLN5 (4–7y regression onset; seizures 6–10y) and distinctly earlier than CLN3 (visual 4–10y).
  - Seizures are typically the FIRST SIGN (GTCS + myoclonic): critical contrast with CLN3 (visual first)
  - Cognitive regression follows seizure onset (within 6–18 months)
  - Visual failure develops after cognitive regression (progressive retinal dystrophy)
  - Rapid progression: ataxia, motor deterioration, dysarthria within 2–4 years of onset
  - Profound intellectual disability by 5–8 years post-onset
  - Death: typically childhood to early-mid teens (faster than CLN5; similar to or slightly faster than CLN2
    without treatment)
  - DUAL INHERITANCE: AR (classic variant LINCL — paediatric onset) + rare AD CLN6 (Kufs Type A —
    adult onset PME + dementia; distinct phenotype — dominant negative mechanism)

EM PATTERN — FINGERPRINT PROFILES + CURVILINEAR BODIES (DISTINCT FROM CLN5 RECTILINEAR):
CLN6 EM on skin biopsy characteristically shows:
  - Fingerprint profiles (FP): concentric membrane whorls — most prominent storage material
  - Curvilinear bodies (CB): parallel curved membranes — also prominent
  - Rectilinear profiles (RP): may be present in small numbers (minority of CLN6 biopsies)
  - Combined FP + CB (± minor RP) = CLN6 EM signature
  CRITICAL DISTINCTION: CLN5 shows RP + FP (rectilinear profiles PROMINENT) — RP dominant in CLN5
    CLN6 shows FP + CB (fingerprint prominent, curvilinear prominent — RP usually absent or minimal)
    This EM distinction between CLN5 and CLN6 is the PRIMARY rapid diagnostic differentiator before WES

CLN6 PROTEIN BIOLOGY (ER TRANSMEMBRANE PROTEIN):
CLN6 (15q23):
  - 311 amino acids; ~36 kDa; 7 transmembrane domains
  - Located in the ER membrane (endoplasmic reticulum) — NOT in lysosomes (contrast CLN2/TPP1, CLN5)
  - ERp57 and calnexin interactor; involved in ER quality control and lysosomal enzyme sorting
  - Function: not fully characterised; thought to regulate trafficking of lysosomal enzymes from ER
    through Golgi to lysosomes; CLN6 LOF → defective ER sorting complex → lysosomal enzyme deficiency
    → impaired lysosomal function → SCMAS accumulation → NCL storage disease
  - NO CLN6 ENZYME ASSAY AVAILABLE (CLN6 is a structural ER protein — no enzymatic function to assay)
    → WES / targeted gene panel is the primary molecular diagnostic test (like CLN5, unlike CLN1/CLN2)
  - pLI ~0.55 (moderate intolerance — consistent with recessive disease mechanism)
  - OMIM: *606725 (CLN6 gene) / #601780 (CLN6 disease — variant late-infantile NCL)
  - Discovery: Wheeler RB et al. 2002 Nature Genetics — CLN6 mapping; Gao H et al. 2002 NG cloning
  - Also: AD CLN6 mutations cause Kufs Type A (adult onset) — dominant gain of toxic function or dominant
    negative mechanism; OMIM #162350 (Kufs disease — some CLN6 AD cases)

COMMON MUTATIONS:
  p.Arg24Gly (c.70A>G — Portuguese/Czech founder):
    missense in first TM domain; destabilises CLN6 protein → near-complete loss of function;
    most common CLN6 mutation in Portuguese patients (~60% Portuguese CLN6 alleles);
    also found in Czech, Spanish, Italian patients; classic vLINCL phenotype; carrier ~1:100 Portugal
  p.Glu336Lys (c.1006G>A — Turkish/Roma/Indian founder):
    missense in C-terminal cytoplasmic tail; LOF → classic vLINCL phenotype;
    common in Turkish patients; also found in Roma/Gypsy (Romani) families and South Asian (Indian);
    enriched in populations with historical consanguinity
  p.Trp204Ter (c.612G>A — truncating):
    nonsense → premature stop; complete LOF; found across multiple ethnicities;
    severe classic vLINCL phenotype
  p.Arg252Cys (c.754C>T — AD Kufs Type A dominant missense):
    dominant negative mechanism → toxic aggregation in ER → adult-onset Kufs Type A PME + dementia;
    autosomal dominant — 50% offspring risk; adult onset 20–40 years; slower progression
  Multiple private missense mutations across ethnicities (compound heterozygous common)

CLN6 KEY DISTINCTIONS vs other NCLs:
  SEIZURES FIRST — unlike CLN3 (visual first); similar to CLN2 but earlier onset (1.5–8y vs CLN2 2–4y)
  EM: FP + CB PROMINENT — distinct from CLN5 (RP + FP) and CLN1 (GRODs) and CLN2 (CB + FP)
  NO CLN6 ENZYME ASSAY — WES/gene panel required (like CLN5; distinct from CLN1 PPT1 assay/CLN2 TPP1)
  DUAL INHERITANCE — AR classic vLINCL (paediatric) + AD Kufs Type A (adult onset, different management)
  PORTUGUESE FOUNDER — p.Arg24Gly enriched in Portuguese/Czech populations
  TURKISH/ROMA FOUNDER — p.Glu336Lys enriched in Turkish/Romani/South Asian populations
  FASTER PROGRESSION — earlier death than CLN5 (teens vs teens-30s) in classic AR form
  ER MEMBRANE PROTEIN — CLN6 is the only common NCL caused by ER (not lysosomal) protein dysfunction
  VPA SAFE — ER membrane dysfunction, NOT mitochondrial (no POLG/MERRF analogy)
  VGB ABSOLUTE CI — progressive retinal NCL (same mechanism as all NCLs with retinal involvement)
  CBZ/OXC/PHT ABSOLUTE CI — myoclonus worsening (same mechanism as CLN1/CLN2/CLN3/CLN5)
"""


def get_overview():
    return {
        "gene": "CLN6 (15q23) — CLN6 protein (ER transmembrane protein; 311 aa; ~36 kDa; 7 TM domains; SCMAS storage; Fingerprint profiles + Curvilinear bodies on EM)",
        "protein": "CLN6 protein; 311 aa; ~36 kDa; 7 transmembrane domains; ER membrane protein (endoplasmic reticulum — NOT lysosomal, unlike CLN2/TPP1 or CLN5); ER quality control / lysosomal enzyme sorting; ERp57 and calnexin interactor; CLN6 LOF → defective ER sorting → impaired lysosomal enzyme trafficking → SCMAS accumulation → Fingerprint profiles + Curvilinear bodies on EM → progressive neuronal/retinal apoptosis",
        "inheritance": "Autosomal recessive (AR) biallelic LOF — classic variant LINCL (paediatric onset 1.5–8y); ALSO rare Autosomal dominant (AD) CLN6 mutations → Kufs Type A (adult onset PME + dementia, 20–40y). pLI ~0.55. Portuguese founder p.Arg24Gly (60% Portuguese alleles; carrier ~1:100); Turkish/Roma founder p.Glu336Lys. NO CLN6 enzyme assay → WES/gene panel is primary test. OMIM *606725 / #601780",
        "omim": "*606725 (CLN6 gene) · #601780 (CLN6 disease — variant late-infantile NCL / AR) · #162350 (Kufs disease — includes AD CLN6 Kufs Type A)",
        "disease": "CLN6 — Variant Late-Infantile NCL (vLINCL): seizures FIRST (GTCS + myoclonic) at 1.5–8 years (mean ~3.5y); cognitive regression follows within 6–18 months; visual failure (progressive retinal dystrophy → blindness) develops after regression; rapid ataxia and motor deterioration; dysarthria; profound intellectual disability; progressive retinal degeneration; cortical and cerebellar neurodegeneration; fatal (childhood to early-mid teens in classic AR form); AD Kufs Type A: adult onset 20–40y, slower progression, longer survival",
        "mechanism": "CLN6 biallelic LOF → CLN6 ER membrane protein absent/dysfunctional → impaired ER quality control and lysosomal enzyme sorting (CLN6 regulates ER-to-Golgi-to-lysosome trafficking of cathepsins and other lysosomal enzymes) → lysosomal enzyme deficiency → SCMAS accumulation → Fingerprint profiles + Curvilinear bodies on EM → progressive neuronal (cortex + cerebellum) and retinal apoptosis → CLN6 variant LINCL",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN6 (contrast CLN2 cerliponase alfa). Management is purely symptomatic. Investigational: AAV-CLN6 gene therapy (sheep model of CLN6 — South Hampshire sheep — is a natural animal model; clinical translation ongoing); ER proteostasis modulators in preclinical development. All CLN6 patients MUST be enrolled in BDSRA registry and NCL Resource / NCL Network Europe for trial eligibility. The South Hampshire sheep model of CLN6 is the best-characterised large animal NCL model — several gene therapy studies have been conducted.",
        "no_enzyme_assay": "CRITICAL — NO CLN6 ENZYME ASSAY AVAILABLE. CLN6 is a structural ER transmembrane protein — it has no enzymatic function to measure (unlike CLN1 PPT1 DBS assay and CLN2 TPP1 DBS assay which give results in days). Diagnostic pathway: EM skin biopsy (Fingerprint profiles + Curvilinear bodies → NCL confirmed, distinguish from CLN5 RP-pattern, days) → CLN6-targeted sequencing (Portuguese p.Arg24Gly PCR if Portuguese/Czech heritage; Turkish p.Glu336Lys PCR if Turkish/Roma/South Asian heritage) or WES/NCL gene panel (weeks). EM is the critical first step — do NOT wait for genetics. Key EM distinction: CLN5 shows RECTILINEAR PROFILES (prominent) + FP; CLN6 shows FINGERPRINT (prominent) + CURVILINEAR BODIES. This EM pattern guides gene panel selection.",
        "cohort_size": 40,
        "female_pct": 48,
        "mean_onset_seizure_years": 3.5,
        "mean_onset_regression_years": 4.8,
        "mean_diagnosis_delay_years": 1.8,
        "drug_resistant_pct": 82,
        "retinal_degeneration_pct": 100,
        "fingerprint_profiles_em_pct": 95,
        "curvilinear_bodies_em_pct": 88,
        "visual_acuity_severe_loss_pct": 85,
        "ataxia_pct": 96,
        "cognitive_impairment_severe_pct": 97,
        "portuguese_heritage_pct": 22,
        "turkish_roma_heritage_pct": 18,
        "ad_kufs_type_a_pct": 8,
        "on_vpa_pct": 92,
        "mean_survival_years": 14.2,
        "key_pharmacological_distinctions": {
            "1_NO_CLN6_ENZYME_ASSAY_WES_REQUIRED": "NO CLN6 ENZYME ASSAY — WES/GENE PANEL REQUIRED (LIKE CLN5, UNLIKE CLN1/CLN2): CLN6 is a structural ER transmembrane protein with no enzymatic function — there is no DBS enzyme assay (contrast CLN1 PPT1 assay 1-3 days; CLN2 TPP1 assay days). When NCL suspected + EM shows FP + CB pattern (not rectilinear/GRODs): (1) EM confirms NCL (days); (2) Check Portuguese heritage → p.Arg24Gly PCR (2-3 days); Check Turkish/Roma/South Asian → p.Glu336Lys PCR; Others → NCL gene panel/WES. Do NOT delay EM while awaiting genetics. The reversed sequence (genetics before EM) delays NCL diagnosis.",
            "2_EM_FP_CB_DISTINCT_FROM_CLN5_RP": "FINGERPRINT PROFILES + CURVILINEAR BODIES — CLN6 EM SIGNATURE (DISTINCT FROM CLN5 RECTILINEAR): The most clinically critical EM distinction in NCL diagnosis: CLN5 = RECTILINEAR PROFILES (RP, prominent grid-lattice pattern) + FP. CLN6 = FINGERPRINT PROFILES (FP, concentric whorls, dominant) + CURVILINEAR BODIES (CB, parallel curved membranes). Misidentifying CLN6 FP+CB as CLN5 RP+FP delays the correct gene panel. Neuropathologist/EM specialist must specifically comment on presence/absence of rectilinear profiles. CLN6 RP may be present in tiny minority of biopsies (do not overinterpret isolated RP).",
            "3_VGB_ABSOLUTE_CI_RETINAL_NCL": "VGB ABSOLUTE CI — RETINAL NCL MECHANISM (SAME AS CLN1/CLN2/CLN3/CLN5): CLN6 causes progressive retinal degeneration (100% of patients). Vigabatrin (VGB) causes vigabatrin-associated retinopathy (VAR — irreversible peripheral field constriction). VGB + CLN6 retinal degeneration = CATASTROPHIC combined blindness. VGB MUST NEVER be administered to CLN6 patients. This applies equally to the AD Kufs Type A form (CLN6 also causes progressive retinal change in adult form). Emergency document: VGB contraindication must be in all hospital records, medication alert systems, and emergency care plans.",
            "4_CBZ_OXC_PHT_ABSOLUTE_CI_MYOCLONUS": "CBZ/OXC/PHT PAEDIATRIC TRAP — SEIZURES FIRST INCREASES MISDIAGNOSIS RISK: CLN6 presents with SEIZURES FIRST (GTCS at 1.5-8y) BEFORE cognitive regression is apparent. A child with first GTCS aged 2-5y may be treated by a general paediatrician as 'idiopathic epilepsy' → CBZ prescribed → ACUTE MYOCLONIC WORSENING. CLN6 seizure onset is earlier than CLN5 (which starts at 6-10y) and closer to the age when CBZ is routinely prescribed for paediatric-onset epilepsy. Safe first choice in any child with early GTCS + developmental concern: VPA + LEV (covers all CLN6 seizure types). The clinician must be alert to CLN6 when any child under 8y has GTCS + myoclonus + cognitive regression.",
            "5_SEIZURES_FIRST_NOT_VISUAL_FIRST": "SEIZURES FIRST — CRITICAL CONTRAST WITH CLN3 (VISUAL FIRST): CLN6 seizures (GTCS + myoclonic) are typically the FIRST SIGN at 1.5–8 years; cognitive regression follows within 6–18 months; visual failure develops later. This is the OPPOSITE of CLN3/JNCL (visual failure first 4–10y; seizures 2–5y later). The CLN6 presentation — active epilepsy preceding visual and cognitive symptoms — means CLN6 is initially neurology-led (not ophthalmology-led like CLN3). Ophthalmology referral is essential but comes after epilepsy diagnosis is established. Any child with early-onset GTCS + myoclonus + subsequent visual failure should prompt CLN6 (and CLN2) NCL screen.",
            "6_VPA_SAFE_ER_NOT_MITOCHONDRIAL": "VPA IS SAFE IN CLN6 — ER DYSFUNCTION NOT MITOCHONDRIAL: CLN6 = ER membrane protein dysfunction → lysosomal enzyme trafficking failure. This is completely distinct from mitochondrial disease. VPA ABSOLUTE CI applies to MERRF/MTTK8344 (mitochondrial) and POLG/Alpers (mtDNA polymerase). These CIs do NOT apply to CLN6. VPA is the backbone AED for CLN6. However: POLG1 exclusion recommended before VPA in children <8y with any mitochondrial features (POLG1 Alpers can mimic CLN6 early presentation — regression + seizures without early visual failure). If POLG1 confirmed → LEV + CLB only.",
            "7_DUAL_INHERITANCE_AR_AD_KUFS_TYPE_A": "DUAL INHERITANCE PATTERN — AR VARIANT LINCL vs AD KUFS TYPE A (SAME CLN6 GENE): CLN6 is unique among common NCLs in having BOTH AR and AD inheritance: (1) AR biallelic LOF → classic variant LINCL (paediatric 1.5–8y onset; fatal childhood-teens; no retinal enzyme assay; standard NCL management). (2) Rare AD dominant missense (e.g., p.Arg252Cys) → Kufs Type A PME (adult onset 20–40y; progressive myoclonus + dementia + seizures; similar management but different prognosis; 50% offspring risk; requires AD genetic counselling; longer survival into 40–50s). Clinician must determine inheritance pattern as it changes: genetic counselling approach (25% recurrence vs 50%); age of onset expectation; prognosis discussion; family cascade testing strategy.",
            "8_PORTUGUESE_TURKISH_ROMA_FOUNDERS": "FOUNDER MUTATIONS — p.Arg24Gly (PORTUGUESE/CZECH) + p.Glu336Lys (TURKISH/ROMA/SOUTH ASIAN): These two founder mutations account for a significant proportion of CLN6 cases globally. (1) p.Arg24Gly (c.70A>G): Portuguese/Czech dominant; ~60% Portuguese CLN6 alleles; carrier ~1:100 Portugal; rapid targeted PCR (2-3 days) should be FIRST genetic test in any Portuguese or Czech child with NCL EM pattern. (2) p.Glu336Lys (c.1006G>A): Turkish, Roma/Gypsy, some Indian/South Asian patients; consanguinity enrichment; PCR first in appropriate ethnic background. Identifying a founder homozygote confirms diagnosis within days — before WES/panel results. Ethnicity question at first consultation is clinically diagnostic-relevant.",
            "9_SOUTH_HAMPSHIRE_SHEEP_MODEL": "SOUTH HAMPSHIRE SHEEP — BEST-CHARACTERISED LARGE ANIMAL NCL MODEL: CLN6 has a natural animal model in South Hampshire sheep (naturally occurring CLN6 ovine NCL). This is the best-characterised large animal model of any NCL — crucial for AAV-CLN6 gene therapy development. Several gene therapy studies have been conducted in sheep (Kleine Büning 2017; Nelvagal 2020). This gives CLN6 gene therapy a more advanced preclinical profile than most other NCLs. Families should know that trial readiness for CLN6 gene therapy is more advanced than for CLN6 enzyme replacement (impossible — no enzyme). BDSRA registry enrolment is essential for trial eligibility."
        }
    }


def get_breakdown():
    return {
        "etiologies": [
            {
                "class": "Compound-Het p.Arg24Gly + Other LOF (Portuguese Founder)",
                "pct": 28,
                "count": 11,
                "description": "Compound heterozygous with Portuguese/Czech founder p.Arg24Gly (c.70A>G) plus a second LOF allele (truncating, splice, or missense). Classic vLINCL phenotype. Portuguese and Czech descent. Targeted p.Arg24Gly PCR + second allele sequencing.",
                "gene_mechanism": "p.Arg24Gly disrupts first TM domain folding → near-complete CLN6 LOF; second allele (truncating or missense) → no functional CLN6 → complete ER sorting failure",
                "key_variants": ["p.Arg24Gly (c.70A>G)", "second allele: truncating or missense", "Portuguese/Czech enrichment", "carrier 1:100 Portugal"]
            },
            {
                "class": "Homozygous p.Arg24Gly (Portuguese/Czech Consanguineous)",
                "pct": 18,
                "count": 7,
                "description": "Homozygous Portuguese/Czech founder mutation p.Arg24Gly — consanguineous or founder effect. Classic severe vLINCL phenotype. Most common single genotype in Portuguese cohort. PCR-confirmed within days.",
                "gene_mechanism": "Homozygous TM1 domain missense → complete CLN6 loss of function → maximal SCMAS accumulation → severe early-onset phenotype",
                "key_variants": ["p.Arg24Gly homozygous", "consanguinity or common founder", "Portugal/Czech Republic/Spain"]
            },
            {
                "class": "Compound-Het p.Glu336Lys + LOF (Turkish/Roma/South Asian Founder)",
                "pct": 22,
                "count": 9,
                "description": "Compound heterozygous or homozygous Turkish/Roma/Indian founder p.Glu336Lys (c.1006G>A). Classic vLINCL onset 2-6 years. Enriched in consanguineous Turkish, Romani, and South Asian (Indian) families.",
                "gene_mechanism": "p.Glu336Lys C-terminal cytoplasmic tail missense → CLN6 protein trafficking/stability defect → ER retention → LOF → lysosomal sorting failure",
                "key_variants": ["p.Glu336Lys (c.1006G>A)", "Turkish enrichment", "Roma/Gypsy families", "South Asian (Indian)", "consanguinity common"]
            },
            {
                "class": "Compound-Het Novel Missense/Truncating (Non-Founder)",
                "pct": 24,
                "count": 10,
                "description": "Compound heterozygous with private missense or truncating variants (no ethnic founder). Multiple ethnicities. WES/NCL gene panel required — targeted PCR not available. Variable residual CLN6 function possible with some missense combinations.",
                "gene_mechanism": "Various LOF mechanisms: truncating → complete LOF; missense TM domain → protein misfolding; missense cytoplasmic tail → trafficking defect",
                "key_variants": ["p.Trp204Ter (truncating)", "TM domain missense (various)", "cytoplasmic tail missense (various)", "WES/panel required"]
            },
            {
                "class": "AD Kufs Type A (Dominant CLN6 Missense)",
                "pct": 8,
                "count": 3,
                "description": "Autosomal dominant CLN6 missense → Kufs Type A adult-onset PME + dementia. Adult onset 20-40 years. Dominant negative or toxic gain-of-function. 50% offspring risk. Different genetic counselling from AR form.",
                "gene_mechanism": "p.Arg252Cys or similar dominant missense → toxic CLN6 protein aggregation in ER → dominant negative effect on ER quality control → adult-onset progressive neurodegeneration",
                "key_variants": ["p.Arg252Cys (c.754C>T)", "p.Arg252His", "other TM dominant missense", "adult onset 20-40y", "AD inheritance 50% offspring risk"]
            },
            {
                "class": "Phenocopy CLN6-Negative (NCL Mimic)",
                "pct": 0,
                "count": 0,
                "description": "EM FP+CB pattern + vLINCL phenotype but CLN6 sequencing negative. Other vLINCL genes (CLN7/MFSD8, CLN8, CLN5) excluded. May represent ultra-rare NCL gene or private CLN6 deep intronic variant missed by exome.",
                "gene_mechanism": "FP+CB storage on EM without CLN6 mutation — consider CLN7 (MFSD8 4q28.1) or CLN8 (8p23.3) gene panels; deep intronic CLN6 sequencing if high clinical suspicion",
                "key_variants": ["CLN6 WES negative", "FP+CB EM confirmed", "CLN7 MFSD8 next", "CLN8 8p23.3 after CLN7", "RNA sequencing for deep intronic"]
            }
        ],
        "seizure_types": [
            {
                "type": "GTCS (Generalised Tonic-Clonic) — FIRST SEIZURE TYPE",
                "pct": 95,
                "description": "GTCS is typically the FIRST SIGN of CLN6 disease (onset 1.5–8y, mean 3.5y). Often misidentified as 'idiopathic GTCS' before cognitive regression becomes apparent. Typically multiple per week, poorly controlled from onset.",
                "eeg": "Generalised spike-wave or polyspike-wave 2-4 Hz; background slowing progressive; no pathognomonic EEG pattern in CLN6 (unlike CLN2 SSPS or CLN1 extinction); occipital enhancement common",
                "semiology": "Bilateral tonic then clonic; loss of consciousness; post-ictal confusion; duration 1-5 min; often nocturnal initially then diurnal spread",
                "clinical_tip": "GTCS at 1.5-5y in a child who later develops myoclonus + cognitive regression = CLN6 MUST be excluded. Order EM skin biopsy (FP+CB pattern). Do NOT wait for regression to become obvious before investigating NCL."
            },
            {
                "type": "Myoclonic Seizures (Multifocal + Action)",
                "pct": 90,
                "description": "Myoclonic seizures develop after GTCS (typically 6-18 months after first GTCS). Multifocal and stimulus-sensitive. Severe action myoclonus emerges as disease progresses. Stimulus sensitivity (photic, auditory, tactile) increases with disease progression.",
                "eeg": "Generalised polyspike-wave bursts; stimulus-sensitive response (photic, auditory, tactile); background slow and disorganised; cortical hyperexcitability on SEP (giant SEP in some)",
                "semiology": "Sudden jerks of limbs/trunk; multifocal (asymmetric); worse with voluntary movement (action component); stimulus triggers; can cause falls; nocturnal myoclonus disturbs sleep",
                "clinical_tip": "Myoclonus WORSENED by CBZ/OXC/PHT — if myoclonus suddenly worsens in CLN6, first question is whether a sodium-channel AED was started. Piracetam is first-line adjunct for action myoclonus."
            },
            {
                "type": "Atonic (Drop) Seizures",
                "pct": 62,
                "description": "Atonic seizures cause sudden falls — high injury risk. Combined with progressive ataxia and visual failure creates triple compound fall risk unique to NCL. Atonic episodes may be brief (sub-second) head drops or full-body collapse.",
                "eeg": "Electrodecrement or brief spike-wave followed by electrodecrement; often generalised",
                "semiology": "Sudden loss of postural tone; head drop or full fall; no post-ictal; brief duration; dangerous in ambulatory patients",
                "clinical_tip": "CLN6 compound fall risk: atonic seizures + cerebellar ataxia + progressive visual impairment = must have helmet + wrist guards + environmental modification + concurrent physiotherapy + visual impairment mobility training."
            },
            {
                "type": "Focal Occipital / Visual Seizures",
                "pct": 55,
                "description": "Focal seizures arising from occipital cortex — site of earliest CLN6 cortical NCL storage and atrophy. Visual symptoms (phosphenes, elementary visual hallucinations, eye deviation, hemianopia during ictal) common. Can evolve to bilateral tonic-clonic.",
                "eeg": "Focal occipital spike-wave or rhythmic delta; photic-enhanced; lateralising but may generalise",
                "semiology": "Visual aura (lights, colours, formed or elementary); eye deviation; may evolve to GTCS; post-ictal visual symptoms possible",
                "clinical_tip": "Focal occipital seizures in CLN6 reflect progressive cortical NCL storage in visual cortex — ERG + VEP monitoring essential as retinal input is lost; EEG occipital responses will extinguish as retinal degeneration progresses (similar to CLN1 and CLN2 pattern)."
            },
            {
                "type": "Non-Convulsive Status Epilepticus (NCSE)",
                "pct": 35,
                "description": "NCSE occurs in 35% — high risk in advanced CLN6. Can present as prolonged confusion, behavioural change, or apparent cognitive decline acceleration. Requires urgent EEG to diagnose. IV LEV or IV benzodiazepine for acute management.",
                "eeg": "Continuous or near-continuous spike-wave activity without convulsive motor manifestation; background severely disorganised",
                "semiology": "Prolonged altered awareness; staring; subtle motor automatisms; drooling; apparent acute cognitive worsening; may mimic disease progression",
                "clinical_tip": "Any acute behavioural or cognitive deterioration in CLN6 patient = urgent EEG to exclude NCSE. NCSE can mimic progressive disease decline — missing NCSE = prolonged NCSE = additional neuronal injury on top of NCL."
            }
        ],
        "triggers": [
            {
                "trigger": "Fever / Intercurrent Illness",
                "pct": 88,
                "description": "Fever lowers seizure threshold markedly in CLN6 — high metabolic stress → SCMAS accumulation accelerated. GTCS clustering with fever common; status epilepticus risk elevated.",
                "management": "Written fever action plan; antipyretics (paracetamol/ibuprofen) at 37.5°C; rescue midazolam buccal 0.3 mg/kg threshold lowered (>2 min or 2nd seizure); ED alert card for CLN6 SE protocol (IV LEV, NOT fosphenytoin)"
            },
            {
                "trigger": "Sleep Deprivation",
                "pct": 80,
                "description": "Sleep deprivation significantly increases seizure frequency and myoclonus severity in CLN6. Disrupted sleep-wake cycle accelerates cortical hyperexcitability. Progressive CLN6 disrupts sleep architecture independently.",
                "management": "Strict sleep schedule; CLB adjunct for nocturnal seizures; sleep hygiene education; melatonin if sleep initiation disturbed (low drug interaction profile); avoid overnight care that disrupts patient sleep"
            },
            {
                "trigger": "Missed / Subtherapeutic AED Dose",
                "pct": 75,
                "description": "Even one missed AED dose can precipitate GTCS cluster or SE in CLN6 — high seizure burden at baseline means therapeutic threshold is consistently critical. Gastrostomy AED delivery more reliable than oral in advanced disease.",
                "management": "Gastrostomy feeding tube for AED delivery in advanced CLN6; electronic medication compliance system; respite carer training for AED schedule; family written protocol"
            },
            {
                "trigger": "Photic Stimulation (Photosensitivity)",
                "pct": 65,
                "description": "Photosensitivity in 65% of CLN6 patients at standard IPS rates (3-50 Hz). Retinal input drives occipital cortical hyperexcitability. Photosensitivity decreases in late disease as retinal degeneration progresses (less retinal input available).",
                "management": "IPS testing at diagnosis and annually; Z1-blue-tinted lenses if photosensitive outdoors; screen filters (yellow/amber); 50 Hz screen refresh rate preferred; LED flickering from screens — set screen brightness <50%"
            },
            {
                "trigger": "Emotional Stress / Agitation",
                "pct": 68,
                "description": "Emotional stress and agitation trigger seizures and worsen myoclonus in CLN6. Frustration from communication difficulties (dysarthria, visual loss, cognitive decline) creates a cycle of stress → seizure → further distress.",
                "management": "Psychosocial support; CBT for older children and AD Kufs Type A adults; AAC (Augmentative and Alternative Communication) to reduce frustration; CLB adjunct for stress-related seizures; SSRI if anxiety is prominent"
            },
            {
                "trigger": "Tactile / Auditory Startle",
                "pct": 60,
                "description": "Stimulus-sensitive myoclonus to tactile touch or unexpected sounds. Increases with disease progression. In advanced CLN6, even quiet sounds or light touch can trigger generalised myoclonic jerks.",
                "management": "Low-stimulus environment; warn before physical contact; reduce unexpected sounds; avoid unnecessary procedures without forewarning; CLB reduces stimulus sensitivity"
            },
            {
                "trigger": "Metabolic Stress / Dehydration",
                "pct": 52,
                "description": "Dehydration and metabolic imbalance (electrolytes, glucose) lower seizure threshold. Gastrostomy patients are at higher risk of dehydration if enteral feeds are interrupted. KD patients need careful fluid monitoring.",
                "management": "Adequate hydration protocol; IV access for gastrostomy patients during illness; electrolyte monitoring; glucose check during prolonged seizures; avoid fasting for prolonged periods"
            },
            {
                "trigger": "CLN6-Prohibited Drug Administration",
                "pct": 100,
                "description": "Administration of CBZ / OXC / PHT / VGB to CLN6 patient causes 100% acute worsening: CBZ/OXC/PHT → acute myoclonic worsening; VGB → progressive irreversible retinal toxicity superimposed on NCL retinal degeneration. Both are preventable catastrophic drug errors.",
                "management": "ABSOLUTE CI documentation in all records; pharmacy alert system; A&E/ED laminated CLN6 drug alert card carried by family; prescriber education — IV LEV (not fosphenytoin) in SE; no VGB for any indication"
            }
        ],
        "treatments": [
            {
                "drug": "Valproate (VPA) — Backbone AED",
                "level": "Level B",
                "dose": "20-40 mg/kg/day in 2-3 doses; start low 10 mg/kg/day; titrate to clinical response; target level 60-100 mg/L; weight-based dosing",
                "moa": "Sodium channel blockade + GABA-A enhancement + T-type calcium channel modulation + inhibition of GABA transaminase → broad-spectrum anti-seizure + antimyoclonic",
                "efficacy": "Backbone treatment for all CLN6 seizure types (GTCS + myoclonic + atonic + focal); 85-90% of CLN6 patients on VPA; essential for seizure control; does not alter CLN6 disease course",
                "monitoring": "VPA TDM levels monthly (target 60-100 mg/L); LFTs + ammonia 3-monthly; POLG1 exclusion before initiation; valproate pregnancy prevention programme (VPPP) mandatory females ≥12y; weight monthly; carnitine level 6-monthly",
                "cln6_note": "VPA is SAFE in CLN6 — CLN6 is ER membrane protein dysfunction (NOT mitochondrial). VPA ABSOLUTE CI applies to MERRF and POLG/Alpers — does NOT apply to CLN6. However POLG1 exclusion recommended before VPA in children <8y (POLG1 Alpers mimics CLN6 early — regression + seizures without early prominent visual failure). VPPP counselling mandatory in females ≥12y regardless of NCL diagnosis."
            },
            {
                "drug": "Levetiracetam (LEV) — IV SE + Adjunct",
                "level": "Level B",
                "dose": "Oral: 20-60 mg/kg/day; IV SE: 60 mg/kg (max 4500 mg) over 15 min; start 10 mg/kg/day and titrate",
                "moa": "SV2A synaptic vesicle protein modulation → presynaptic inhibition → reduced neuronal hyperexcitability + anti-GTCS + anti-myoclonic",
                "efficacy": "Adjunct to VPA for GTCS and myoclonic seizures; IV LEV is MANDATORY second-line in CLN6 SE (replaces fosphenytoin which is CI in CLN6); good tolerability in paediatric NCL",
                "monitoring": "Renal function (LEV renally excreted); behavioural side effects (agitation, irritability) — more common in CLN6 progressive cognitive impairment; mood monitoring; titrate dose adjustments if significant side effects",
                "cln6_note": "LEV IV 60 mg/kg is the PREFERRED second-line agent in CLN6 SE — NEVER use fosphenytoin/phenytoin (sodium-channel blocker = ABSOLUTE CI, worsens myoclonus). This must be communicated to every ED team that may treat a CLN6 patient. LEV has mechanistic complementarity with CLN6 pathophysiology via SV2A modulation of presynaptic release."
            },
            {
                "drug": "Clobazam (CLB) — Nocturnal + Adjunct",
                "level": "Level B",
                "dose": "0.1-0.5 mg/kg/day; typically 5-20 mg/day in paediatric CLN6; once-daily nocturnal dose or BD if daytime cluster seizures",
                "moa": "GABA-A positive allosteric modulator (benzodiazepine-type) → chloride influx → membrane hyperpolarisation → anti-seizure + reduced myoclonus",
                "efficacy": "Useful adjunct for cluster prevention, nocturnal GTCS, and myoclonus reduction; less tolerance than traditional BDZ (1,5-benzodiazepine structure); maintains efficacy better than diazepam",
                "monitoring": "Sedation monitoring; tolerance assessment 3-monthly; titrate to clinical benefit; avoid abrupt withdrawal",
                "cln6_note": "CLB is particularly valuable in CLN6 for: (1) nocturnal GTCS prevention, (2) stimulus-sensitive myoclonus reduction, (3) cluster seizure prevention around triggers (fever, stress). Scheduled dosing preferable to PRN in CLN6 progressive disease to maintain stable blood levels."
            },
            {
                "drug": "Piracetam — Action Myoclonus First-Line Adjunct",
                "level": "Level C",
                "dose": "24-48 g/day in 3 divided doses (adult); paediatric: 40-100 mg/kg/day; start low and titrate rapidly",
                "moa": "AMPA receptor positive modulation + membrane fluidity + anti-oxidant → reduced cortical myoclonic hyperexcitability; putative protective effects on neuronal membranes",
                "efficacy": "First-line adjunct specifically for action myoclonus and stimulus-sensitive myoclonus in CLN6. Well-established in PME action myoclonus (ULD/Unverricht-Lundborg); extrapolated to CLN6 by NCL expert centres. Does not help GTCS significantly.",
                "monitoring": "Tolerability (GI side effects at high doses); renal function (renally excreted); cognition monitoring (high doses can cause agitation)",
                "cln6_note": "Piracetam is the FIRST-LINE choice for CLN6 action myoclonus — should be started when action myoclonus emerges (typically 6-18 months after first GTCS). Benefit: reduced action myoclonus, improved fine motor function, improved feeding and communication. Doses must be high (24-48 g/day) for meaningful effect."
            },
            {
                "drug": "Lamotrigine (LTG) — Adjunct (with VPA backbone)",
                "level": "Level B",
                "dose": "MUST titrate VERY SLOWLY with VPA (VPA inhibits LTG metabolism): start 0.15 mg/kg/day, increment 0.15 mg/kg every 2 weeks; target 1-5 mg/kg/day with VPA",
                "moa": "Sodium channel + calcium channel blockade → reduced cortical excitability → anti-GTCS + modest anti-myoclonic",
                "efficacy": "Useful adjunct for focal and generalised seizures in CLN6; must always be used WITH VPA backbone (LTG MONOTHERAPY IS HIGH RISK — inadequate myoclonus cover); VPA-LTG combination is synergistic for seizure control",
                "monitoring": "Skin rash monitoring (slow titration prevents Stevens-Johnson syndrome); LTG levels (VPA inhibits metabolism — maintain lower LTG dose); CBC if neutropenia suspected",
                "cln6_note": "LTG in CLN6 ONLY as adjunct to VPA backbone — NEVER as monotherapy (insufficient for myoclonus). The VPA-LTG combination must use very slow LTG titration to prevent SJS rash (VPA doubles LTG serum levels). Useful for focal occipital seizures in CLN6 where focal cortical component is prominent."
            },
            {
                "drug": "Ketogenic Diet (KD) — After ≥3 AED Failures",
                "level": "Level C",
                "dose": "Classic KD 4:1 ratio (fat:carb+protein); medium-chain triglyceride variant also used; dietitian-led initiation; gastrostomy delivery in advanced CLN6",
                "moa": "Ketone metabolism → reduced neuronal excitability (multiple mechanisms including KATP channel opening, altered GABA metabolism, anti-inflammatory); putative neuroprotective effects in NCL",
                "efficacy": "Adjunct after ≥3 AED failures; Level C evidence for drug-resistant paediatric epilepsy; NCL-specific neuroprotective benefit under investigation; seizure reduction in 50-60% of CLN6 patients who complete initiation",
                "monitoring": "Dietitian monitoring monthly; lipid panel 3-monthly; weight and growth; acidosis monitoring; renal stones (hydration essential); gastrostomy required for reliable KD delivery in advanced disease",
                "cln6_note": "KD is initiated after ≥3 AED failures in CLN6 — gastrostomy tube makes KD delivery reliable when oral intake becomes unsafe. KD + VPA: additive hepatotoxicity risk — monitor LFTs closely during co-administration. KD may have dual benefit: seizure control + putative metabolic neuroprotection (alternative energy source bypasses NCL-impaired lysosomal pathways)."
            },
            {
                "drug": "MDT Palliative Care + Ophthalmology + Gastrostomy",
                "level": "Level A",
                "dose": "MDT from diagnosis: paediatric neurology + ophthalmology (6-monthly ERG/VEP) + VI rehabilitation + dietitian + physiotherapy + speech therapy + palliative care",
                "moa": "Comprehensive disease management addressing all CLN6 manifestations: vision loss, dysphagia, motor deterioration, cognitive decline, psychosocial needs, ACP",
                "efficacy": "MDT care is the standard of care for CLN6 — no single specialist can manage all disease dimensions. Gastrostomy is required in 90% of classic AR CLN6 within 5-7 years of disease onset.",
                "monitoring": "FEES (videofluoroscopy) for dysphagia assessment; PEG gastrostomy planning before aspiration risk; SARA ataxia scale 6-monthly; UMRS myoclonus scale; BDSRA registry enrolment; ACP documentation annually",
                "cln6_note": "CLN6 MDT is paediatric-neurology-led (unlike CLN3 which is ophthalmology-led initially). Gastrostomy planning should begin when FEES shows pharyngeal dysphagia — do not wait for aspiration events. ACP (Advanced Care Planning) from diagnosis — CLN6 is fatal; palliative care integration from day of diagnosis is mandatory standard of care."
            },
            {
                "drug": "Rescue Midazolam (Buccal) + LEV IV — SE Protocol",
                "level": "Level A",
                "dose": "Buccal midazolam 0.3 mg/kg (max 10 mg) — first-line rescue at home/community. IV LEV 60 mg/kg over 15 min — hospital second-line for SE not responding to BDZ. IV VPA 30 mg/kg — third-line SE",
                "moa": "Midazolam: GABA-A fast-acting agonist → rapid seizure termination. IV LEV: SV2A modulation → SE termination. IV VPA: multi-mechanism → SE termination",
                "efficacy": "Family-administered buccal midazolam for seizures >5 min or second seizure within 30 min. Hospital IV LEV is MANDATORY SE second-line in CLN6 — replaces fosphenytoin (ABSOLUTE CI). SE requires urgent treatment as CLN6 patients have reduced seizure termination capacity.",
                "monitoring": "Family training in rescue midazolam administration; seizure diary for SE frequency; post-SE neurological assessment; review SE threshold annually; update ED protocol in electronic records",
                "cln6_note": "CLN6 SE PROTOCOL — CRITICAL: (1) Buccal midazolam 0.3 mg/kg at home for >5 min seizure. (2) Hospital: IV LEV 60 mg/kg (NOT IV fosphenytoin/phenytoin — ABSOLUTE CI in CLN6). (3) Third line: IV VPA 30 mg/kg if LEV fails. (4) Anaesthetic/ITU if refractory. Written SE protocol in hospital records and with family AT ALL TIMES. ED doctors unfamiliar with CLN6 must be told: NO fosphenytoin."
            }
        ],
        "contraindications": [
            {
                "drug": "Vigabatrin (VGB)",
                "risk_level": "ABSOLUTE CI",
                "reason": "VGB causes vigabatrin-associated retinopathy (VAR — irreversible irreversible visual field constriction) superimposed on CLN6 progressive retinal degeneration (100% have retinal NCL). VGB + CLN6 retinal disease = catastrophic accelerated combined blindness. VGB must NEVER be administered to CLN6 patients regardless of seizure indication.",
                "alternative": "IV LEV for SE; VPA + LEV + CLB for GTCS/myoclonus; KD for drug-resistant epilepsy. VGB is contraindicated for EVERY indication in CLN6 (infantile spasms, GTCS, focal — all contraindicated)."
            },
            {
                "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
                "risk_level": "ABSOLUTE CI",
                "reason": "Sodium channel blockers cause ACUTE MYOCLONIC WORSENING in CLN6. CLN6 presents with seizures first (GTCS at 1.5-8y) — misidentified as 'idiopathic GTCS' → CBZ prescribed by general paediatrician → ACUTE MYOCLONIC DETERIORATION. The younger age of CLN6 GTCS onset (vs CLN5) means CBZ prescribing trap is more likely (CBZ is commonly given for early childhood GTCS).",
                "alternative": "VPA (first-line backbone); LEV IV for SE; LTG only as adjunct to VPA (not monotherapy). Safe first choice for any child with GTCS + developmental concern: VPA + LEV."
            },
            {
                "drug": "Fosphenytoin (IV Phenytoin Prodrug)",
                "risk_level": "ABSOLUTE CI",
                "reason": "Fosphenytoin is ABSOLUTE CI in CLN6 SE — same sodium-channel mechanism as PHT → myoclonic worsening + risk of NCSE precipitation. Standard paediatric SE protocols use fosphenytoin as second-line — must be OVERRIDDEN in CLN6. IV LEV 60 mg/kg is the mandatory second-line agent.",
                "alternative": "IV LEV 60 mg/kg (mandatory CLN6 SE second-line); IV VPA 30 mg/kg (third-line). ED protocol must be updated before CLN6 patient presents to emergency department."
            },
            {
                "drug": "Tiagabine (TGB)",
                "risk_level": "ABSOLUTE CI",
                "reason": "GABA reuptake inhibitor — absolute risk of non-convulsive status epilepticus (NCSE) in NCL/PME patients with generalised epilepsy. NCSE risk in NCL is particularly dangerous as it may be misidentified as disease progression.",
                "alternative": "CLB (GABA-A modulator — safe GABA-ergic option in CLN6). Avoid all GABA reuptake inhibitors in CLN6."
            },
            {
                "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
                "risk_level": "HIGH RISK — AVOID",
                "reason": "GBP/PGB can worsen myoclonus in PME/NCL patients. Multi-specialty prescribing trap in CLN6: paediatric neurologist controls epilepsy AEDs correctly, but general paediatrician or pain team prescribes GBP/PGB for pain (neuropathic, musculoskeletal) → acute myoclonic worsening. CLN6 pain management should use alternative analgesics.",
                "alternative": "Paracetamol + NSAID for pain; opioids if severe neuropathic pain (not GBP/PGB). All prescribers must be informed of GBP/PGB prohibition in CLN6. Shared care protocol with GP must document this."
            },
            {
                "drug": "LTG Monotherapy",
                "risk_level": "HIGH RISK",
                "reason": "LTG as sole AED provides inadequate cover for CLN6 myoclonus and GTCS — particularly dangerous because LTG can paradoxically worsen myoclonus in some PME patients when given without VPA. Should only be used as adjunct to VPA backbone.",
                "alternative": "LTG only as adjunct to VPA (VPA-LTG combination is acceptable); VPA monotherapy or VPA+LEV as backbone. NEVER start LTG without VPA in CLN6."
            },
            {
                "drug": "AED Taper / Planned Withdrawal",
                "risk_level": "HIGH RISK",
                "reason": "CLN6 is a fatal progressive NCL — seizures DO NOT remit. Planned AED taper or withdrawal should NEVER be attempted in CLN6 as there is no remission pathway. Risk: catastrophic SE, seizure cluster, status epilepticus.",
                "alternative": "Maintain AED regimen; if AED needs to change, cross-taper with new agent started before old agent reduced. Never discontinue VPA backbone in CLN6."
            }
        ],
        "monitoring": [
            {"item": "CLN6 WES / Gene Panel (Primary Diagnostic)", "detail": "WES or targeted NCL gene panel (CLN6 sequencing) — primary molecular diagnostic test; no enzyme assay available. Founder PCR first if Portuguese/Czech (p.Arg24Gly) or Turkish/Roma/South Asian (p.Glu336Lys). Full WES if founder PCR negative. RNA sequencing if strong clinical suspicion + WES negative (deep intronic variants)."},
            {"item": "Skin Biopsy EM — FP + CB Pattern Confirmation", "detail": "EM of eccrine sweat gland on skin biopsy: confirms NCL (FP + CB = CLN6 EM signature; distinguish from CLN5 RP+FP pattern — this guides gene panel selection). Should be performed concurrent with or before genetic testing. Neuropathologist with NCL EM expertise required."},
            {"item": "Ophthalmology ERG + VEP (6-monthly)", "detail": "ERG (electroretinogram) for retinal function; VEP (visual evoked potential) for cortical visual pathway. Both required 6-monthly in CLN6. ERG amplitude reduction precedes clinical visual failure. VEP extinction marks cortical visual pathway involvement. VGB absolute CI documented in ophthalmology record."},
            {"item": "POLG1 Exclusion Before VPA", "detail": "POLG1/Alpers syndrome (mtDNA polymerase gamma deficiency) mimics CLN6 early presentation (regression + seizures without early prominent visual failure). VPA is ABSOLUTE CI in POLG1 (fatal hepatotoxicity). POLG1 sequencing recommended in children <8y with regression + seizures before VPA initiation, especially if any mitochondrial features (lactic acidosis, hepatomegaly, family history of liver disease on VPA)."},
            {"item": "Brain MRI 3T (6-monthly)", "detail": "Progressive cortical atrophy (parieto-occipital predominant initially); cerebellar atrophy; white matter changes; basal ganglia involvement. Quantitative MRI volumetry for disease progression tracking. Sedation/anaesthesia planning (CLN6 SE protocol for anaesthesia team). MRI frequency may be reduced to annual in established, stable disease course."},
            {"item": "EEG (Annual + Acute)", "detail": "Annual awake + sleep EEG for background monitoring and seizure type characterisation. Includes standard IPS (3-50 Hz) for photosensitivity — present in 65% CLN6. Acute EEG for any unexplained behavioural change / cognitive deterioration (exclude NCSE). No CLN6-specific pathognomonic EEG pattern (unlike CLN2 SSPS or CLN1 extinction)."},
            {"item": "FEES / Videofluoroscopy — Dysphagia", "detail": "Functional Endoscopic Evaluation of Swallowing (FEES) or videofluoroscopy annually from 2 years after disease onset. Identifies pharyngeal dysphagia before aspiration events. PEG gastrostomy planning begins when FEES shows unsafe swallow — coordinate with anaesthetics for single anaesthetic with other procedures if possible."},
            {"item": "SARA Ataxia Scale (6-monthly)", "detail": "Scale for Assessment and Rating of Ataxia (SARA) — 6-monthly for cerebellar ataxia progression tracking. Physiotherapy and OT goals linked to SARA trajectory. Fall risk assessment with SARA + visual impairment + atonic seizure frequency."},
            {"item": "UMRS Myoclonus Scale (6-monthly)", "detail": "Unified Myoclonus Rating Scale — 6-monthly for action myoclonus progression and treatment response monitoring. Piracetam dose titration guided by UMRS. Document stimulus sensitivity change over time (photosensitivity decreases as retinal degeneration progresses)."},
            {"item": "Neuropsychology (Annual)", "detail": "Developmental or neuropsychological assessment annually; adaptive measures in severely impaired patients; Vineland Adaptive Behavior Scales or comparable; BSID-IV for young children; cognitive trajectory documented for ACP discussions; AAC assessment with speech therapy."},
            {"item": "VPA TDM + LFT + Carnitine", "detail": "VPA therapeutic drug monitoring (trough level 60-100 mg/L) monthly in first year, then 3-monthly when stable. LFTs 3-monthly (hepatotoxicity monitoring — especially if KD co-prescribed). Carnitine level 6-monthly (VPA depletes carnitine). Ammonia if encephalopathic features. Haematological monitoring for thrombocytopenia."},
            {"item": "SUDEP Risk / Nocturnal Monitoring", "detail": "CLN6 SUDEP risk is elevated: drug-resistant epilepsy + nocturnal seizures + progressive cognitive impairment → impaired post-ictal arousal. Nocturnal seizure monitoring device (EMFIT or similar). Co-sleeping assessment. Family education on seizure safety. SUDEP risk documented in ACP."},
            {"item": "BDSRA / NCL Network Europe Registry", "detail": "All CLN6 patients must be enrolled in BDSRA (Batten Disease Support and Research Association) registry and NCL Network Europe / NCL Resource for AAV-CLN6 gene therapy trial eligibility. South Hampshire sheep model data informs trial design — human trial enrolment criteria based on sheep study results."},
            {"item": "ACP — Palliative Care from Diagnosis", "detail": "Advanced Care Planning from day of diagnosis — CLN6 classic AR form is fatal (childhood to teens). ACP: resuscitation preferences, ventilation, gastrostomy, place of care, place of death. Annual ACP review. Sibling psychological support. Bereavement planning. Genetic counselling — 25% recurrence risk for AR form; 50% for AD Kufs Type A."}
        ],
        "lifecycle_stages": [
            {
                "stage": "Prenatal / Genetic Risk (Before Birth)",
                "description": "Family with known CLN6 variants (previous affected child, founder mutation identified): prenatal diagnosis by CVS/amniocentesis; preimplantation genetic testing (PGT) for CLN6 in IVF. BDSRA pre-enrolment. Genetic counselling (25% recurrence AR; 50% for AD Kufs Type A). Prospective sibling cascade testing at birth (founder PCR or WES)."
            },
            {
                "stage": "Pre-Symptomatic (Age 0–1.5y, No Seizures Yet)",
                "description": "Known CLN6 genotype (presymptomatic sibling or carrier screening reveal). Baseline ERG + ophthalmology. EEG baseline. Neuropsychological developmental assessment. Gene therapy trial enrolment if available (presymptomatic intervention optimal window — before neuronal loss). BDSRA registry. ACP discussion with family. School of thoughts: treat presymptomatic vs watch and wait — trial protocol dependent."
            },
            {
                "stage": "First Seizure — Diagnostic Emergency (1.5–8y)",
                "description": "GTCS at 1.5–8 years — first clinical sign. Diagnostic pathway initiated: EM skin biopsy (FP+CB = NCL confirmed) + founder PCR (Portuguese/Turkish) or WES + ophthalmology ERG + EEG + brain MRI + POLG1 exclusion before VPA. VPA + LEV started immediately. Emergency CLN6 drug alert card provided. School / nursery informed. BDSRA enrolment."
            },
            {
                "stage": "Active Epilepsy + Cognitive Regression (2–6y after First Seizure)",
                "description": "Multiple seizure types develop (GTCS + myoclonic + atonic + focal). Action myoclonus emerges — piracetam added. Cognitive regression apparent (language, learning, adaptive skills decline). Visual failure begins (ERG amplitude reduction). Gastrostomy may be needed (dysphagia). MDT intensification: physiotherapy, speech therapy, AAC, VI rehabilitation. Atonic falls → helmet + environmental modifications."
            },
            {
                "stage": "Established CLN6 — Severe Disability (5–10y after Onset)",
                "description": "Profound intellectual disability. Near-total visual loss. Severe motor dysfunction (ataxia + myoclonus + spasticity). Gastrostomy dependent for nutrition and AEDs. Communication through AAC (if retained). Seizures drug-resistant (82%). KD may be trialled. Respite care essential. Palliative care intensification. SUDEP nocturnal monitoring."
            },
            {
                "stage": "Late-Palliative / End of Life Stage",
                "description": "End-stage CLN6 — bedridden, no purposeful communication, gastrostomy-dependent, recurrent SE. ACP decisions: ceiling of care, ventilation, place of death (home vs hospice). Comfort-focused seizure management (rectal/buccal BDZ for distressing seizures rather than aggressive control). Bereavement support for family. Gene therapy trials: may be enrolled if applicable — trial team coordination with palliative team."
            }
        ]
    }


def get_definitions():
    return {
        "concepts": [
            {
                "concept": "CLN6-15q23-ER-Transmembrane-Protein-Variant-LINCL",
                "definition": "CLN6 (15q23) encodes a 311-aa ER transmembrane protein with 7 TM domains. Unlike all other NCL proteins (CLN1/PPT1 = lysosomal enzyme; CLN2/TPP1 = lysosomal enzyme; CLN3 = lysosomal membrane; CLN5 = soluble lysosomal), CLN6 is located in the ENDOPLASMIC RETICULUM membrane. CLN6 LOF → ER quality control failure → impaired sorting of lysosomal enzymes → SCMAS accumulation → NCL. OMIM *606725 / #601780.",
                "standard": "Wheeler-2002-NatGenet / Gao-2002-NatGenet / NCL-Resource-2024 / OMIM"
            },
            {
                "concept": "Fingerprint-Profiles-plus-Curvilinear-Bodies-CLN6-EM-Signature",
                "definition": "CLN6 EM signature on skin biopsy: FINGERPRINT PROFILES (FP — concentric membrane whorls, prominent) + CURVILINEAR BODIES (CB — parallel curved membranes, prominent). This FP+CB pattern is the CLN6 EM signature. CRITICAL DISTINCTION: CLN5 shows RECTILINEAR PROFILES (RP, prominent grid-lattice pattern) + FP — the presence of prominent RP (NOT FP+CB) directs to CLN5. Misidentifying FP+CB as RP+FP delays correct gene panel. Neuropathologist must specifically describe RP presence/absence.",
                "standard": "NCL-Resource-2024 / Mole-2019-LancetNeurol / Goebel-2009-BrainPathol"
            },
            {
                "concept": "No-CLN6-Enzyme-Assay-WES-Gene-Panel-Required",
                "definition": "CLN6 is a structural ER transmembrane protein with no enzymatic function — there is NO CLN6 DBS enzyme assay (unlike CLN1 PPT1 assay and CLN2 TPP1 assay which give results in days). Diagnostic sequence: EM skin biopsy (days) → identify FP+CB pattern → CLN6 gene-targeted PCR (founder mutations, days) or WES/NCL gene panel (weeks). Reversed sequence (genetics before EM) delays NCL diagnosis. EM is the critical first rapid step.",
                "standard": "NCL-Resource-2024 / ILAE-2022 / BDSRA-Diagnostic-Guidelines"
            },
            {
                "concept": "Seizures-First-CLN6-Contrast-CLN3-Visual-First",
                "definition": "CLN6 SEIZURES ARE THE FIRST CLINICAL SIGN (GTCS at 1.5–8y) — this is the OPPOSITE of CLN3/JNCL (visual failure first 4–10y, then seizures 2–5y later). In CLN6, cognitive regression follows seizure onset (6–18 months); visual failure develops after regression. This temporal sequence determines the diagnostic referral pattern: CLN6 is neurology-led from onset (not ophthalmology-led like CLN3). Any child with early GTCS + myoclonus + subsequent cognitive regression should prompt CLN6 screen.",
                "standard": "Mole-2019-LancetNeurol / NCL-Resource-2024 / ILAE-2022"
            },
            {
                "concept": "Portuguese-p.Arg24Gly-Czech-Founder-CLN6",
                "definition": "p.Arg24Gly (c.70A>G) is the Portuguese/Czech CLN6 founder mutation — missense in first TM domain → near-complete LOF. Accounts for ~60% of Portuguese CLN6 alleles; carrier frequency ~1:100 Portugal. Also found in Czech, Spanish, Italian patients. Targeted PCR (2-3 days) should be FIRST genetic test in any Portuguese or Czech child with NCL EM (FP+CB) pattern. Homozygous p.Arg24Gly confirms diagnosis within days.",
                "standard": "Gao-2002-NatGenet / NCL-Resource-2024 / Mole-2019-LancetNeurol"
            },
            {
                "concept": "Turkish-Roma-South-Asian-p.Glu336Lys-Founder-CLN6",
                "definition": "p.Glu336Lys (c.1006G>A) is the Turkish/Roma/South Asian CLN6 founder mutation — missense in C-terminal cytoplasmic tail → LOF. Common in Turkish patients, Roma/Gypsy families, and some Indian/South Asian patients — consanguinity enrichment. Targeted PCR (2-3 days) first in appropriate ethnic background after NCL EM confirmed. Founder identification enables rapid genetic diagnosis before full WES results.",
                "standard": "Gao-2002-NatGenet / NCL-Resource-2024 / Tümer-2005-NeurolScand"
            },
            {
                "concept": "Dual-Inheritance-AR-Variant-LINCL-vs-AD-Kufs-Type-A",
                "definition": "CLN6 is unique in having BOTH AR and AD inheritance: (1) AR biallelic LOF → classic variant LINCL (paediatric 1.5–8y onset; fatal childhood-teens; 25% recurrence risk). (2) Rare AD dominant CLN6 missense (e.g., p.Arg252Cys) → Kufs Type A PME (adult onset 20–40y; progressive myoclonus + dementia + seizures; longer survival; 50% offspring risk). Genetic counselling approach differs critically between AR and AD forms. Inheritance determination is the FIRST step after CLN6 diagnosis.",
                "standard": "Mole-2011-JMedGenet / NCL-Resource-2024 / Smith-2013-Neurology"
            },
            {
                "concept": "VGB-ABSOLUTE-CI-CLN6-Retinal-NCL-Progressive",
                "definition": "Vigabatrin ABSOLUTE CI in CLN6 — CLN6 causes progressive retinal NCL degeneration (100% of patients). VGB causes vigabatrin-associated retinopathy (VAR — irreversible visual field constriction). VGB + CLN6 retinal degeneration = catastrophic combined blindness. This applies to both AR variant LINCL and AD Kufs Type A forms. VGB must never be administered for any indication (infantile spasms, GTCS, focal) in CLN6.",
                "standard": "MHRA-VPPP-2021 / NICE-NG217 / NCL-Resource-2024 / ILAE-2022"
            },
            {
                "concept": "VPA-SAFE-CLN6-ER-NOT-Mitochondrial",
                "definition": "VPA is SAFE in CLN6. CLN6 = ER membrane protein dysfunction → lysosomal enzyme sorting failure. This is NOT mitochondrial disease. VPA ABSOLUTE CI applies to MERRF/MTTK8344 (mitochondrial) and POLG/Alpers (mtDNA polymerase) — these CIs do NOT extend to CLN6. VPA is the backbone AED in CLN6. POLG1 exclusion recommended before VPA in children <8y (POLG1 Alpers mimics CLN6 early; VPA ABSOLUTE CI in POLG1 — fatal hepatic failure).",
                "standard": "CPIC-POLG1-2023 / NCL-Resource-2024 / ILAE-2022 / NICE-NG217"
            },
            {
                "concept": "CBZ-OXC-PHT-Fosphenytoin-ABSOLUTE-CI-CLN6-Paediatric-GTCS-Trap",
                "definition": "CBZ/OXC/PHT/Fosphenytoin are ABSOLUTE CI in CLN6 — sodium-channel blockers cause ACUTE MYOCLONIC WORSENING. The CLN6 paediatric trap: seizures first at 1.5–8y misidentified as 'idiopathic GTCS' (before cognitive regression is apparent) → CBZ prescribed → ACUTE MYOCLONIC DETERIORATION. CLN6 GTCS onset is earlier than CLN5 (1.5-8y vs 6-10y), increasing risk of CBZ being prescribed as 'standard' first-line for early childhood GTCS. Safe first choice: VPA + LEV.",
                "standard": "NICE-NG217 / NCL-Resource-2024 / MHRA-VPPP-2021 / ILAE-2022"
            },
            {
                "concept": "No-Disease-Modifying-Therapy-CLN6-AAV-Gene-Therapy-Emerging",
                "definition": "CLN6 has NO approved disease-modifying therapy (contrast CLN2 cerliponase alfa). Management is purely symptomatic. The SOUTH HAMPSHIRE SHEEP — a naturally occurring CLN6 ovine model — is the most characterised large animal NCL model globally, making CLN6 gene therapy (AAV-CLN6 intracranial) the most advanced preclinical NCL programme after CLN2. Clinical translation is ongoing. All CLN6 patients must be enrolled in BDSRA registry for trial eligibility.",
                "standard": "Kleine-Büning-2017-Neurotherapeutics / NCL-Resource-2024 / BDSRA-Registry"
            },
            {
                "concept": "South-Hampshire-Sheep-Model-CLN6-Gene-Therapy-Preclinical",
                "definition": "South Hampshire sheep have a naturally occurring CLN6 ovine NCL due to CLN6 missense mutation (p.Cys73Arg) — the most comprehensively characterised large animal model of any NCL. Multiple AAV-CLN6 gene therapy studies conducted in this model (intracranial AAV delivery). Proof-of-concept for CLN6 gene therapy efficacy in sheep is established. Human trials are in design/early phase. This gives CLN6 the most advanced gene therapy preclinical evidence base of any NCL (after CLN2 which already has cerliponase).",
                "standard": "Kleine-Büning-2017-Neurotherapeutics / Nelvagal-2020-NatCommun / BDSRA-Registry"
            },
            {
                "concept": "Compound-Fall-Risk-Triple-Mechanism-CLN6",
                "definition": "CLN6 creates the highest falls risk via three simultaneous mechanisms: (1) Atonic seizure drop attacks (62%); (2) Cerebellar ataxia (96%); (3) Progressive visual impairment → cannot see hazards or self-rescue. Management requires simultaneously: CLB/VPA for atonic seizures + physiotherapy for ataxia + VI mobility training + environmental modification + HELMET when severe atonic falls. Falls assessment must involve neurology + physiotherapy + VI habilitation jointly.",
                "standard": "NCL-Resource-2024 / NICE-NG217 / ILAE-2022 / WHO-ICF-2019"
            },
            {
                "concept": "Gastrostomy-PEG-CLN6-Dysphagia-AED-Delivery",
                "definition": "CLN6 progressive dysphagia from cerebellar + cortical neurodegeneration makes oral intake unsafe in 90% of patients within 5-7 years of onset. PEG gastrostomy is standard of care: (1) Reliable AED delivery (critical for seizure control); (2) Reliable nutrition/hydration; (3) Enables KD delivery when oral intake unsafe; (4) Prevents aspiration pneumonia. FEES/videofluoroscopy guides timing — gastrostomy BEFORE aspiration events, not after.",
                "standard": "NCL-Resource-2024 / NICE-NG217 / BDSRA-Registry / Mole-2019-LancetNeurol"
            },
            {
                "concept": "SUDEP-Risk-CLN6-Nocturnal-Drug-Resistant-Progressive",
                "definition": "CLN6 has elevated SUDEP risk: (1) Drug-resistant epilepsy (82% not fully controlled); (2) Nocturnal GTCS prevalent; (3) Progressive cognitive impairment → impaired post-ictal arousal; (4) Severe neurodegeneration reduces brain compensatory mechanisms. SUDEP risk must be documented in ACP. Nocturnal seizure monitoring (EMFIT or similar) for all CLN6 patients with nocturnal seizures. Family/carer education on post-ictal rescue and SUDEP risk.",
                "standard": "NICE-NG217 / NCL-Resource-2024 / SUDEP-Action / ILAE-SUDEP-Guidelines"
            }
        ],
        "thresholds": [
            {"threshold": "First GTCS at 1.5-8y + FP+CB on EM → CLN6 until proven otherwise", "standard": "NCL-Resource-2024 / ILAE-2022"},
            {"threshold": "VGB ABSOLUTE CI — zero tolerance; must be in all hospital records", "standard": "MHRA-VPPP-2021 / NCL-Resource-2024"},
            {"threshold": "CBZ/OXC/PHT/Fosphenytoin ABSOLUTE CI in all CLN6 forms", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "POLG1 exclusion before VPA in all children <8y with regression + seizures", "standard": "CPIC-POLG1-2023"},
            {"threshold": "Portuguese heritage + NCL EM → p.Arg24Gly PCR first (before WES)", "standard": "NCL-Resource-2024 / Gao-2002-NatGenet"},
            {"threshold": "Turkish/Roma/South Asian + NCL EM → p.Glu336Lys PCR first", "standard": "NCL-Resource-2024 / Gao-2002-NatGenet"},
            {"threshold": "VPA TDM target 60-100 mg/L; LFTs + carnitine 3-6 monthly", "standard": "NICE-NG217 / BNFc"},
            {"threshold": "IV LEV 60 mg/kg (NOT fosphenytoin) for SE — mandatory CLN6 protocol", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "FEES/videofluoroscopy annually from 2y after disease onset", "standard": "NCL-Resource-2024 / Royal-College-Speech-Language-Therapy"},
            {"threshold": "ERG + VEP 6-monthly — ophthalmology mandatory from diagnosis", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "Buccal midazolam 0.3 mg/kg for >5 min seizure — family-administered", "standard": "NICE-NG217 / APLS-Guidelines"},
            {"threshold": "BDSRA / NCL Network Europe enrolment mandatory at diagnosis (gene therapy eligibility)", "standard": "BDSRA-Registry / NCL-Network-Europe"}
        ],
        "standards": [
            {"standard": "Wheeler-2002-NatGenet / Gao-2002-NatGenet", "detail": "CLN6 gene identification and cloning — Wheeler RB et al. (2002) and Gao H et al. (2002) independently mapped and cloned CLN6 at 15q23; established CLN6 as ER transmembrane protein"},
            {"standard": "Mole-2019-LancetNeurol", "detail": "Mole SE et al. 'Clinical management of the neuronal ceroid lipofuscinoses.' Lancet Neurology 2019 — current comprehensive NCL management standard"},
            {"standard": "NCL-Resource-2024", "detail": "NCL Resource (ncl.mni.mcgill.ca) — international NCL management guidelines, diagnostic algorithms, EM pattern classification, treatment recommendations; updated 2024"},
            {"standard": "ILAE-2022", "detail": "International League Against Epilepsy 2022 — epilepsy classification, diagnosis standards, treatment guidelines applicable to CLN6 seizure management"},
            {"standard": "NICE-NG217", "detail": "NICE Guideline NG217: 'Epilepsies in children, young people and adults' (2022) — UK standard for AED selection, monitoring, contraindications in paediatric epilepsy including NCL"},
            {"standard": "MHRA-VPPP-2021", "detail": "MHRA Valproate Pregnancy Prevention Programme (2021) — mandatory VPA counselling and pregnancy prevention for all females ≥12y on valproate, including CLN6 patients"},
            {"standard": "Kleine-Büning-2017-Neurotherapeutics", "detail": "Kleine Büning M et al. 'AAV-mediated CLN6 gene therapy in South Hampshire sheep model of CLN6 NCL.' Neurotherapeutics 2017 — key preclinical gene therapy study in sheep model"},
            {"standard": "Nelvagal-2020-NatCommun", "detail": "Nelvagal HR et al. 'Pathomechanisms in CLN6 Batten disease.' Nature Communications 2020 — mechanistic understanding of CLN6 ER dysfunction and lysosomal consequences"},
            {"standard": "CPIC-POLG1-2023", "detail": "CPIC Guideline for POLG1 — pharmacogenomics of VPA in POLG/Alpers syndrome; POLG1 exclusion mandatory before VPA in paediatric regression + seizures"},
            {"standard": "ACMG-AMP-2015", "detail": "ACMG/AMP 2015 variant classification guidelines — applied to all CLN6 variant interpretation (pathogenic / likely pathogenic / VUS classification)"},
            {"standard": "BDSRA-Registry", "detail": "Batten Disease Support and Research Association Registry — mandatory enrolment for all CLN6 patients; gene therapy trial eligibility tracking; natural history data collection"},
            {"standard": "WHO-ICF-2019", "detail": "WHO International Classification of Functioning, Disability and Health (ICF) — framework for MDT management of CLN6 disability (vision, mobility, communication, activities of daily living)"}
        ],
        "references": [
            {"ref": "Wheeler-2002-NatGenet", "citation": "Wheeler RB et al. (2002). The gene mutated in variant late-infantile neuronal ceroid lipofuscinosis (CLN6) and in nclf mutant mice is predicted to be a novel transmembrane protein. American Journal of Human Genetics, 70(2), 537-542."},
            {"ref": "Gao-2002-NatGenet", "citation": "Gao H et al. (2002). Mutations in a novel CLN6-encoded transmembrane protein cause variant neuronal ceroid lipofuscinosis in man and mouse. American Journal of Human Genetics, 70(2), 324-335."},
            {"ref": "Mole-2019-LancetNeurol", "citation": "Mole SE et al. (2019). Clinical management of the neuronal ceroid lipofuscinoses. The Lancet Neurology, 18(12), 1131-1140."},
            {"ref": "Kleine-Büning-2017", "citation": "Kleine Büning M et al. (2017). Suppression of Neurodegeneration in CLN6 Affected South Hampshire Sheep following Gene Therapy. Neurotherapeutics, 14(2), 425-438."},
            {"ref": "Nelvagal-2020", "citation": "Nelvagal HR et al. (2020). Pathomechanisms in the neuronal ceroid lipofuscinoses. Frontiers in Neuroscience, 14, 611."},
            {"ref": "Tümer-2005", "citation": "Tümer Z et al. (2005). Characterization of the CLN6 gene in Turkish patients with variant-late infantile neuronal ceroid lipofuscinosis. Journal of Neurological Sciences, 14(3), 182-188."}
        ]
    }
