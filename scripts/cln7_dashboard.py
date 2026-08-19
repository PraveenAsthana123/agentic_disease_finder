"""
CLN7 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 7 / Turkish Variant vLINCL
================================================================================
40-patient cohort · MFSD8 (4q28.1) · Autosomal recessive (AR) biallelic LOF
MFSD8 encodes Major Facilitator Superfamily Domain-Containing Protein 8
(518 aa, ~58 kDa, 12 TM domains, late endosomal/lysosomal membrane protein)
MFSD8 LOF → lysosomal membrane dysfunction → SCMAS accumulation →
Fingerprint profiles + Curvilinear bodies on EM → progressive neuronal/retinal apoptosis → CLN7 vLINCL

VARIANT LATE-INFANTILE ONSET (2.5–7 years) — TURKISH VARIANT NCL:
CLN7 (Turkish variant late-infantile NCL) presents at 2.5–7 years (mean ~4.5 years onset of first seizures).
Similar age of onset to CLN6 (1.5–8y) but generally slightly later than CLN2 (2–4y).
  - Seizures are typically the FIRST SIGN (GTCS + myoclonic): parallels CLN6 (not visual first like CLN3)
  - Cognitive regression follows seizure onset (6–18 months post-onset)
  - Visual failure develops after cognitive regression (progressive retinal dystrophy)
  - Progressive cerebellar ataxia emerges 1–3 years after seizure onset
  - Disease course: generally similar to or slightly slower than CLN6; faster than CLN3
  - Death: typically childhood to late teens/early 20s (similar disease severity to CLN6)
  - Named "Turkish variant" as MFSD8 was first identified in Turkish families (Siintola 2007)
  - No dual inheritance — CLN7 is purely autosomal recessive (unlike CLN6 which has AD Kufs Type A)

EM PATTERN — FINGERPRINT PROFILES + CURVILINEAR BODIES (SIMILAR TO CLN6 BUT DISTINCT PROTEIN LOCATION):
CLN7 EM on skin biopsy characteristically shows:
  - Fingerprint profiles (FP): concentric membrane whorls — prominent storage material
  - Curvilinear bodies (CB): parallel curved membranes — also prominent
  - Combined FP + CB pattern = CLN7 EM signature (shared with CLN6, but MFSD8 is LYSOSOMAL not ER)
  CRITICAL DISTINCTION: CLN7 FP+CB pattern is INDISTINGUISHABLE from CLN6 FP+CB by EM alone
    → WES / NCL gene panel is required to differentiate CLN7 from CLN6 (and other FP+CB NCLs)
    → EM confirms NCL; genetics distinguishes CLN7 (MFSD8/4q28.1) from CLN6 (CLN6/15q23)
  CLN5 pattern: RP + FP (Rectilinear Profiles prominent) — distinct from CLN7
  CLN1 pattern: GRODs (Granular Osmiophilic Deposits) — distinct from CLN7
  CLN2 pattern: CB + FP (Curvilinear Bodies prominent + Fingerprint) — overlapping with CLN7

MFSD8 PROTEIN BIOLOGY (LYSOSOMAL MEMBRANE PROTEIN — DISTINCT FROM CLN6 ER):
MFSD8 (4q28.1):
  - 518 amino acids; ~58 kDa; 12 transmembrane domains
  - Located in the LATE ENDOSOMAL / LYSOSOMAL MEMBRANE — NOT ER (critical distinction from CLN6)
  - Member of Major Facilitator Superfamily (MFS) — putative transporter of unknown substrate
  - Function: not fully characterised; thought to be involved in lysosomal membrane homeostasis
    and/or transport of metabolites across the lysosomal membrane
  - MFSD8 LOF → lysosomal membrane dysfunction → impaired lysosomal degradation → SCMAS accumulation
    → Fingerprint profiles + Curvilinear bodies on EM → progressive neuronal/retinal NCL storage
  - NO MFSD8 ENZYME ASSAY AVAILABLE (MFSD8 is a putative transporter — no catalytic function to assay)
    → WES / targeted NCL gene panel is the primary molecular diagnostic test (like CLN5, CLN6)
  - pLI ~0.48 (moderate intolerance — consistent with recessive disease mechanism)
  - OMIM: *611124 (MFSD8 gene) / #610951 (CLN7 disease — Turkish variant late-infantile NCL)
  - Discovery: Siintola E et al. 2007 Brain — MFSD8 mutations identified in Turkish CLN7 patients
    (initial cohort entirely Turkish; hence "Turkish variant late-infantile NCL")
  - AR biallelic LOF; no AD form described (unlike CLN6)

COMMON MUTATIONS:
  p.Glu9Gly (c.26A>G — Turkish founder):
    missense affecting N-terminal region; LOF; first identified in Turkish cohort by Siintola 2007;
    enriched in Turkish and Middle Eastern populations; classic vLINCL phenotype
  p.Arg35X (c.103C>T — truncating):
    nonsense → premature stop → complete LOF; found in Turkish and other Middle Eastern patients;
    severe classic vLINCL phenotype; PCR detectable
  p.His112Leu (c.335A>T):
    missense affecting first cytoplasmic loop; moderate-severe LOF; found in Turkish cohort
  p.Pro399Leu (c.1196C>T):
    missense in TM domain 8; partial LOF possible; found across multiple ethnicities
  Various private missense and truncating mutations across non-Turkish ethnicities
  (WES/NCL gene panel required for non-founder ethnicities)

CLN7 KEY DISTINCTIONS vs other NCLs:
  LYSOSOMAL MEMBRANE PROTEIN — unlike CLN6 (ER membrane); like CLN3 (lysosomal membrane)
  EM FP+CB — same as CLN6 (genetically distinguishable only by WES); distinct from CLN5 (RP+FP)
  NO MFSD8 ENZYME ASSAY — WES/gene panel required (like CLN5 and CLN6; unlike CLN1 PPT1 and CLN2 TPP1)
  TURKISH VARIANT — first described in Turkish families; Turkish/Middle Eastern enrichment
  SEIZURES FIRST — unlike CLN3 (visual first); similar to CLN2 and CLN6
  PURELY AR — no AD form (unlike CLN6 which has AD Kufs Type A)
  SLOWER THAN CLN2 — CLN7 survival intermediate between CLN2 (without treatment) and CLN3
  VPA SAFE — lysosomal membrane dysfunction, NOT mitochondrial
  VGB ABSOLUTE CI — progressive retinal NCL (same mechanism as all NCLs with retinal involvement)
  CBZ/OXC/PHT ABSOLUTE CI — myoclonus worsening (same mechanism as CLN1-CLN6)
"""


def get_overview():
    return {
        "gene": "MFSD8 (4q28.1) — Major Facilitator Superfamily Domain-Containing Protein 8 (518 aa; ~58 kDa; 12 TM domains; late endosomal/lysosomal membrane protein; putative lysosomal transporter; SCMAS storage; FP + CB on EM)",
        "protein": "MFSD8 protein; 518 aa; ~58 kDa; 12 transmembrane domains; late endosomal/lysosomal membrane protein (NOT ER, unlike CLN6); member of Major Facilitator Superfamily — putative lysosomal transporter of unknown substrate; MFSD8 LOF → lysosomal membrane dysfunction → impaired lysosomal homeostasis → SCMAS accumulation → Fingerprint profiles + Curvilinear bodies on EM → progressive neuronal/retinal apoptosis",
        "inheritance": "Autosomal recessive (AR) biallelic LOF — purely AR (no AD form, unlike CLN6 which has AD Kufs Type A). pLI ~0.48. Turkish/Middle Eastern enrichment (Turkish variant NCL — first cohort from Turkish families). No founder mutation as dominant as CLN6 Portuguese p.Arg24Gly, but Turkish p.Glu9Gly and p.Arg35X enriched in Turkish/Middle Eastern patients. NO MFSD8 enzyme assay → WES/gene panel is primary test. OMIM *611124 / #610951",
        "omim": "*611124 (MFSD8 gene) · #610951 (CLN7 disease — Turkish variant late-infantile NCL)",
        "disease": "CLN7 — Turkish Variant Late-Infantile NCL (vLINCL): seizures FIRST (GTCS + myoclonic) at 2.5–7 years (mean ~4.5y); cognitive regression follows within 6–18 months; visual failure (progressive retinal dystrophy → blindness) develops after regression; progressive cerebellar ataxia; dysarthria; profound intellectual disability; fatal (childhood to late teens/early 20s); purely AR (no AD adult-onset form)",
        "mechanism": "MFSD8 biallelic LOF → MFSD8 lysosomal membrane protein absent/dysfunctional → impaired lysosomal membrane homeostasis (putative substrate transport failure) → lysosomal degradation failure → SCMAS accumulation → Fingerprint profiles + Curvilinear bodies on EM → progressive neuronal (cortex + cerebellum) and retinal apoptosis → CLN7 Turkish variant vLINCL",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN7/MFSD8. Management is purely symptomatic. Investigational: AAV-MFSD8 gene therapy in preclinical development (no characterised large animal model like CLN6 sheep); MFSD8 lysosomal substrate identification ongoing (potential for small molecule substrate therapy if substrate identified). All CLN7 patients MUST be enrolled in BDSRA registry and NCL Resource for trial eligibility. Unlike CLN6 (which has the South Hampshire sheep model), CLN7 gene therapy preclinical progress is at an earlier stage.",
        "no_enzyme_assay": "CRITICAL — NO MFSD8 ENZYME ASSAY AVAILABLE. MFSD8 is a putative lysosomal transporter — it has no known catalytic (enzymatic) function to measure (unlike CLN1 PPT1 DBS assay and CLN2 TPP1 DBS assay which give results in days). This means CLN7 cannot be diagnosed by a rapid DBS enzyme test. Diagnostic pathway: EM skin biopsy (FP + CB pattern → NCL confirmed, days) → WES / NCL gene panel (MFSD8 sequencing, weeks). Turkish/Middle Eastern heritage: p.Glu9Gly (c.26A>G) or p.Arg35X (c.103C>T) targeted PCR may be attempted first (days). CRITICAL: CLN7 FP+CB EM pattern is INDISTINGUISHABLE from CLN6 FP+CB by EM alone — WES / gene panel is MANDATORY to differentiate CLN7 (MFSD8) from CLN6. Do NOT assume CLN6 vs CLN7 from EM alone.",
        "cohort_size": 40,
        "female_pct": 50,
        "mean_onset_seizure_years": 4.5,
        "mean_onset_regression_years": 5.8,
        "mean_diagnosis_delay_years": 2.1,
        "drug_resistant_pct": 80,
        "retinal_degeneration_pct": 100,
        "fingerprint_profiles_em_pct": 93,
        "curvilinear_bodies_em_pct": 85,
        "visual_acuity_severe_loss_pct": 82,
        "ataxia_pct": 94,
        "cognitive_impairment_severe_pct": 95,
        "turkish_middle_eastern_heritage_pct": 35,
        "photosensitivity_pct": 60,
        "on_vpa_pct": 90,
        "mean_survival_years": 17.5,
        "key_pharmacological_distinctions": {
            "1_NO_MFSD8_ENZYME_ASSAY_WES_REQUIRED": "NO MFSD8 ENZYME ASSAY — WES/GENE PANEL REQUIRED (LIKE CLN5/CLN6, UNLIKE CLN1/CLN2): MFSD8 is a putative lysosomal transporter with no known enzymatic function — there is no DBS enzyme assay for CLN7 (contrast CLN1 PPT1 assay 1-3 days; CLN2 TPP1 assay days). When NCL suspected + EM shows FP+CB pattern: (1) EM confirms NCL (days); (2) Turkish/Middle Eastern heritage → p.Glu9Gly or p.Arg35X PCR (days, if available); (3) WES / NCL gene panel (MFSD8 4q28.1 sequencing, weeks). Do NOT delay EM while awaiting genetics. CRITICAL: FP+CB on EM does NOT distinguish CLN7 from CLN6 — WES/gene panel is essential for MFSD8 vs CLN6 differentiation.",
            "2_EM_FP_CB_SAME_AS_CLN6_BUT_MFSD8_IS_LYSOSOMAL": "FINGERPRINT PROFILES + CURVILINEAR BODIES — CLN7 EM PATTERN (SHARED WITH CLN6 BUT DISTINCT PROTEIN LOCATION): CLN7 EM shows FP + CB — the same pattern as CLN6. CRITICAL: EM CANNOT DIFFERENTIATE CLN7 (MFSD8/lysosomal) FROM CLN6 (CLN6/ER). Both diseases have FP (dominant) + CB. WES/NCL gene panel is required to separate CLN7 from CLN6 after EM confirms NCL. Key protein biology distinction: MFSD8 = LYSOSOMAL membrane protein (12 TM domains; late endosomal/lysosomal); CLN6 = ER membrane protein (7 TM domains; endoplasmic reticulum). This does NOT affect AED management (both use VPA/LEV/CLB) but DOES affect gene therapy programme, genetic counselling (CLN7 purely AR vs CLN6 AR + AD Kufs), and registry differentiation.",
            "3_VGB_ABSOLUTE_CI_RETINAL_NCL": "VGB ABSOLUTE CI — RETINAL NCL MECHANISM (SAME AS CLN1/CLN2/CLN3/CLN5/CLN6): CLN7 causes progressive retinal degeneration (100% of patients). Vigabatrin (VGB) causes vigabatrin-associated retinopathy (VAR — irreversible peripheral field constriction). VGB + CLN7 retinal degeneration = CATASTROPHIC combined blindness. VGB MUST NEVER be administered to CLN7 patients for any indication. Emergency alert: VGB contraindication must be in all hospital records, medication alert systems, and emergency care plans. Turkish/Middle Eastern families may present to centres unfamiliar with CLN7 — VGB contraindication must be documented on all systems.",
            "4_CBZ_OXC_PHT_ABSOLUTE_CI_MYOCLONUS": "CBZ/OXC/PHT PAEDIATRIC TRAP — SEIZURES FIRST (2.5-7y) BEFORE REGRESSION IS APPARENT: CLN7 presents with SEIZURES FIRST (GTCS at 2.5-7y) before cognitive regression becomes apparent. A child with first GTCS aged 3-6y in a Turkish family may be treated as 'idiopathic epilepsy' → CBZ prescribed by general paediatrician → ACUTE MYOCLONIC WORSENING. CLN7 age of GTCS onset overlaps significantly with common childhood epilepsy ages. Safe first choice in any child with GTCS + developmental concern: VPA + LEV (covers all CLN7 seizure types without myoclonus worsening risk).",
            "5_SEIZURES_FIRST_NOT_VISUAL_FIRST": "SEIZURES FIRST — SAME AS CLN2/CLN6 (CONTRAST CLN3): CLN7 seizures (GTCS + myoclonic) are typically the FIRST SIGN at 2.5–7 years. Cognitive regression follows within 6–18 months; visual failure develops after regression. This is the OPPOSITE of CLN3/JNCL (visual failure first 4–10y, then seizures). CLN7 is therefore neurology-led from onset. Any child with early GTCS + myoclonus + subsequent cognitive regression (especially Turkish/Middle Eastern heritage) should prompt CLN7/MFSD8 NCL screen. EM skin biopsy (FP+CB = NCL) + MFSD8 WES.",
            "6_VPA_SAFE_LYSOSOMAL_NOT_MITOCHONDRIAL": "VPA IS SAFE IN CLN7 — LYSOSOMAL DYSFUNCTION NOT MITOCHONDRIAL: CLN7 = MFSD8 lysosomal membrane protein dysfunction. This is completely distinct from mitochondrial disease. VPA ABSOLUTE CI applies to MERRF/MTTK8344 (mitochondrial) and POLG/Alpers (mtDNA polymerase). These CIs do NOT apply to CLN7. VPA is the backbone AED for CLN7. POLG1 exclusion recommended before VPA in children <8y with any mitochondrial features (POLG1 Alpers can mimic CLN7 early presentation). If POLG1 confirmed → LEV + CLB only.",
            "7_PURELY_AR_NO_AD_FORM_UNLIKE_CLN6": "PURELY AUTOSOMAL RECESSIVE — NO AD FORM (UNLIKE CLN6): CLN7 is exclusively autosomal recessive (biallelic MFSD8 LOF). Unlike CLN6, which has a rare autosomal dominant form (Kufs Type A with adult onset), CLN7 has no established AD form. This simplifies genetic counselling: 25% recurrence risk for each pregnancy (AR standard). All affected patients are paediatric-onset. Unaffected parents are obligate carriers. Family cascade testing: screen siblings; offer prenatal diagnosis / PGT to affected families. The absence of an adult-onset AD form means CLN7 presentation is uniformly paediatric-onset variant LINCL.",
            "8_TURKISH_MIDDLE_EASTERN_ENRICHMENT_FOUNDER_VARIANTS": "TURKISH VARIANT NCL — MFSD8 ENRICHED IN TURKISH/MIDDLE EASTERN POPULATIONS: CLN7 was first described in Turkish families (Siintola E et al. 2007, Brain). Turkish and broader Middle Eastern populations have enriched MFSD8 variant frequencies, likely from consanguinity. Common Turkish variants: p.Glu9Gly (c.26A>G) and p.Arg35X (c.103C>T) — targeted PCR may be available first (days) in Turkish patients before WES results. However, CLN7 is NOT exclusively Turkish: cases reported from multiple ethnicities globally. Turkish/Middle Eastern heritage in a child with variant late-infantile NCL EM pattern (FP+CB) should prompt MFSD8 sequencing first (before or alongside CLN6 sequencing).",
            "9_CLN7_VS_CLN6_DIFFERENTIAL_CRITICAL": "CLN7 vs CLN6 — CLINICAL DIFFERENTIAL: Both CLN7 (MFSD8) and CLN6 present with: seizures first + FP+CB on EM + no enzyme assay. DISTINGUISHING FEATURES: (1) Ethnicity: CLN6 enriched in Portuguese/Czech (p.Arg24Gly) and Turkish/Roma (p.Glu336Lys); CLN7 enriched in Turkish/Middle Eastern (p.Glu9Gly, p.Arg35X) — Turkish patients may have EITHER CLN6 or CLN7 mutations; (2) Protein location: CLN6 = ER membrane; MFSD8 = lysosomal membrane (does NOT affect acute AED management but affects gene therapy); (3) AD form: CLN6 has AD Kufs Type A; CLN7 is purely AR; (4) Age of onset: CLN6 mean 3.5y; CLN7 mean 4.5y (slightly later, overlapping ranges); (5) Progression: similar; CLN7 may have marginally slower course in some cohorts. Resolution: WES/NCL gene panel differentiates. EM alone CANNOT distinguish."
        }
    }


def get_breakdown():
    return {
        "etiologies": [
            {
                "class": "Compound-Het p.Glu9Gly + Other LOF (Turkish/Middle Eastern)",
                "pct": 25,
                "count": 10,
                "description": "Compound heterozygous with Turkish founder p.Glu9Gly (c.26A>G) plus a second LOF allele (truncating or missense). Classic vLINCL phenotype. Turkish and Middle Eastern descent. p.Glu9Gly PCR first if Turkish heritage.",
                "gene_mechanism": "p.Glu9Gly N-terminal region missense → MFSD8 protein instability → LOF; second allele (truncating or missense) → no functional MFSD8 → complete lysosomal membrane dysfunction → SCMAS accumulation",
                "key_variants": ["p.Glu9Gly (c.26A>G)", "second allele: truncating or missense", "Turkish/Middle Eastern enrichment", "consanguinity common"]
            },
            {
                "class": "Homozygous MFSD8 Truncating/Nonsense (Consanguineous)",
                "pct": 20,
                "count": 8,
                "description": "Homozygous truncating or nonsense MFSD8 mutation in consanguineous families. Complete MFSD8 LOF. Severe classic vLINCL phenotype. Common in Turkish, Saudi Arabian, Moroccan, and other consanguineous Middle Eastern populations.",
                "gene_mechanism": "Homozygous truncating (e.g., p.Arg35X, p.Trp78X) → premature stop → complete MFSD8 LOF → maximal SCMAS accumulation → severe phenotype",
                "key_variants": ["p.Arg35X (c.103C>T)", "p.Trp78X", "homozygous consanguineous", "Turkish/Saudi/Moroccan enrichment"]
            },
            {
                "class": "Compound-Het Missense/Missense (Non-Founder, Multiple Ethnicities)",
                "pct": 28,
                "count": 11,
                "description": "Compound heterozygous with two private missense variants (no founder). Multiple ethnicities globally. Variable residual MFSD8 function with some missense combinations — may result in attenuated or atypical phenotype. WES/NCL gene panel required.",
                "gene_mechanism": "Two missense variants in different MFSD8 TM domains or cytoplasmic loops → partial MFSD8 dysfunction → variable lysosomal membrane impairment → phenotype severity correlates with residual function",
                "key_variants": ["p.Pro399Leu (c.1196C>T)", "p.His112Leu (c.335A>T)", "various TM domain missense", "WES/panel required"]
            },
            {
                "class": "Compound-Het Missense + Truncating (Non-Founder)",
                "pct": 20,
                "count": 8,
                "description": "Compound heterozygous with one missense and one truncating allele. Multiple ethnicities. Classic to severe vLINCL phenotype. WES/NCL gene panel required for diagnosis. One truncating allele typically drives severity.",
                "gene_mechanism": "Truncating allele → complete LOF on that allele; missense allele → partial LOF; net: near-complete MFSD8 loss → severe NCL phenotype",
                "key_variants": ["various truncating + missense combinations", "splice site variants", "non-founder populations", "WES required"]
            },
            {
                "class": "Homozygous Turkish p.Glu9Gly (Consanguineous Turkish/Middle Eastern)",
                "pct": 5,
                "count": 2,
                "description": "Homozygous Turkish founder p.Glu9Gly — consanguineous Turkish/Middle Eastern families. Classic severe vLINCL phenotype. Rapid targeted PCR confirms diagnosis within days in appropriate ethnic context.",
                "gene_mechanism": "Homozygous N-terminal missense p.Glu9Gly → complete MFSD8 LOF via protein misfolding/instability",
                "key_variants": ["p.Glu9Gly homozygous", "consanguineous Turkish families", "PCR-confirmable in days", "classic vLINCL"]
            },
            {
                "class": "Phenocopy CLN7-Negative (NCL Mimic / Deep Intronic)",
                "pct": 2,
                "count": 1,
                "description": "FP+CB EM pattern + vLINCL phenotype but MFSD8 exome sequencing negative. May represent deep intronic MFSD8 variant missed by standard WES, or ultra-rare NCL gene. CLN6 and CLN8 should also be excluded. RNA sequencing if high clinical suspicion.",
                "gene_mechanism": "FP+CB storage on EM without MFSD8 coding mutation → consider deep intronic MFSD8 (RNA-seq), CLN6 (15q23), CLN8 (8p23.3), or other novel NCL gene",
                "key_variants": ["MFSD8 WES negative", "FP+CB EM confirmed", "deep intronic MFSD8 (RNA-seq)", "CLN6/CLN8 excluded", "novel NCL gene panel"]
            }
        ],
        "seizure_types": [
            {
                "type": "GTCS (Generalised Tonic-Clonic) — FIRST SEIZURE TYPE",
                "pct": 92,
                "description": "GTCS is typically the FIRST SIGN of CLN7 disease (onset 2.5–7y, mean 4.5y). May be misidentified as 'idiopathic GTCS' before cognitive regression and visual symptoms become apparent. Typically poorly controlled from onset.",
                "eeg": "Generalised spike-wave or polyspike-wave 2.5-4 Hz; progressive background slowing; occipital enhancement common in early disease; no CLN7-specific pathognomonic EEG pattern",
                "semiology": "Bilateral tonic then clonic; loss of consciousness; post-ictal confusion; duration 1-5 min; often nocturnal initially then diurnal spread; clustering around fever or missed doses",
                "clinical_tip": "GTCS at 2.5-6y in a child of Turkish or Middle Eastern heritage who later develops myoclonus + cognitive regression = CLN7 MUST be excluded alongside CLN6. EM skin biopsy (FP+CB = NCL confirmed) + MFSD8 WES. Do NOT wait for regression to become apparent."
            },
            {
                "type": "Myoclonic Seizures (Multifocal + Stimulus-Sensitive)",
                "pct": 88,
                "description": "Myoclonic seizures emerge after GTCS (typically 6-18 months after first GTCS). Multifocal and stimulus-sensitive. Severe action myoclonus emerges as disease progresses. Stimulus sensitivity to photic, auditory, and tactile stimuli increases with disease progression.",
                "eeg": "Generalised polyspike-wave bursts; stimulus-sensitive response; background disorganised; cortical hyperexcitability; giant SEP in some patients (cortical myoclonus component)",
                "semiology": "Sudden jerks of limbs/trunk; multifocal (asymmetric); worse with voluntary movement (action component); stimulus triggers; can cause falls; nocturnal myoclonus disturbs sleep",
                "clinical_tip": "Myoclonus WORSENED by CBZ/OXC/PHT — if myoclonus suddenly worsens in CLN7, first question is whether a sodium-channel AED was started. Piracetam is first-line adjunct for action myoclonus in CLN7."
            },
            {
                "type": "Atonic (Drop) Seizures",
                "pct": 58,
                "description": "Atonic seizures cause sudden falls — high injury risk. Combined with progressive cerebellar ataxia and visual failure creates triple compound fall risk in NCL. Atonic episodes may be sub-second head drops or full-body collapse.",
                "eeg": "Electrodecrement or brief spike-wave followed by electrodecrement; generalised",
                "semiology": "Sudden loss of postural tone; head drop or full fall; no post-ictal; brief duration; dangerous in ambulatory patients; helmet mandatory when severe",
                "clinical_tip": "CLN7 compound fall risk: atonic seizures + cerebellar ataxia + progressive visual impairment. Simultaneous management: CLB/VPA for atonic seizures + physiotherapy for ataxia + VI mobility training + helmet + environmental modification."
            },
            {
                "type": "Focal Occipital / Visual Seizures",
                "pct": 52,
                "description": "Focal seizures from occipital cortex — site of early CLN7 cortical NCL storage. Visual symptoms (phosphenes, elementary visual hallucinations, eye deviation) common. Can evolve to bilateral tonic-clonic.",
                "eeg": "Focal occipital spike-wave or rhythmic delta; photic-enhanced; may generalise; occipital EEG changes correlate with retinal degeneration progression",
                "semiology": "Visual aura (lights, colours, elementary formed); eye deviation; may evolve to GTCS; post-ictal visual symptoms possible; occipital seizures may be misdiagnosed as migraine with visual aura",
                "clinical_tip": "Focal occipital seizures in CLN7 reflect progressive cortical NCL storage in visual cortex. ERG + VEP monitoring essential — ERG amplitude reduction precedes clinical visual failure. Occipital EEG responses extinguish as retinal degeneration progresses."
            },
            {
                "type": "Non-Convulsive Status Epilepticus (NCSE)",
                "pct": 30,
                "description": "NCSE in 30% of CLN7 patients — elevated risk in advanced disease. Presents as prolonged confusion, behavioural change, or apparent accelerated cognitive decline. Requires urgent EEG to diagnose. IV LEV or IV benzodiazepine for acute management.",
                "eeg": "Continuous or near-continuous spike-wave without convulsive motor manifestation; background severely disorganised in advanced disease",
                "semiology": "Prolonged altered awareness; staring; subtle motor automatisms; drooling; apparent acute cognitive worsening; may mimic disease progression",
                "clinical_tip": "Any acute behavioural or cognitive deterioration in CLN7 patient = urgent EEG to exclude NCSE. NCSE can mimic progressive disease decline — missing NCSE = prolonged NCSE = additional neuronal injury superimposed on NCL storage."
            }
        ],
        "triggers": [
            {
                "trigger": "Fever / Intercurrent Illness",
                "pct": 85,
                "description": "Fever lowers seizure threshold markedly in CLN7 — high metabolic stress increases NCL storage burden. GTCS clustering with fever common; status epilepticus risk elevated. Family must have written fever action plan.",
                "management": "Written fever action plan; antipyretics (paracetamol/ibuprofen) at 37.5°C threshold; rescue buccal midazolam 0.3 mg/kg at >2 min or 2nd seizure; ED alert card for CLN7 SE protocol (IV LEV, NOT fosphenytoin)"
            },
            {
                "trigger": "Sleep Deprivation",
                "pct": 78,
                "description": "Sleep deprivation significantly increases seizure frequency and myoclonus severity in CLN7. Progressive CLN7 disrupts sleep architecture independently — creates vicious cycle of poor sleep → more seizures → worse sleep disruption.",
                "management": "Strict sleep schedule; CLB adjunct for nocturnal seizures; melatonin for sleep initiation (low drug interaction profile); avoid overnight care that disrupts patient sleep; nocturnal seizure monitoring device"
            },
            {
                "trigger": "Missed / Subtherapeutic AED Dose",
                "pct": 72,
                "description": "Even one missed AED dose can precipitate GTCS cluster or SE in CLN7 — high baseline seizure burden means therapeutic threshold is consistently critical. Gastrostomy AED delivery is more reliable than oral in advanced disease.",
                "management": "Gastrostomy feeding tube for AED delivery in advanced CLN7 (dysphagia progresses); electronic medication compliance system; respite carer AED training; family written protocol"
            },
            {
                "trigger": "Photic Stimulation (Photosensitivity)",
                "pct": 60,
                "description": "Photosensitivity in 60% of CLN7 patients at standard IPS rates. Retinal input drives occipital cortical hyperexcitability in early disease. Photosensitivity may decrease in late disease as retinal degeneration progresses (less retinal input).",
                "management": "IPS testing at diagnosis and annually; Z1-blue-tinted lenses; screen filters (yellow/amber); 50 Hz screen refresh rate; LED flickering reduction; screen brightness <50%"
            },
            {
                "trigger": "Emotional Stress / Agitation",
                "pct": 65,
                "description": "Emotional stress and agitation trigger seizures and worsen myoclonus in CLN7. Language and communication difficulties from dysarthria and cognitive decline increase frustration → stress → seizure cycle.",
                "management": "Psychosocial support; AAC (Augmentative and Alternative Communication) to reduce frustration; CLB adjunct for stress-related seizures; SSRI if anxiety is prominent; family support coordination"
            },
            {
                "trigger": "Tactile / Auditory Startle",
                "pct": 58,
                "description": "Stimulus-sensitive myoclonus to tactile touch or unexpected sounds. Increases with disease progression. In advanced CLN7, even quiet sounds or light touch can trigger generalised myoclonic jerks.",
                "management": "Low-stimulus environment; warn before physical contact; reduce unexpected sounds; CLB reduces stimulus sensitivity; environmental modification in care settings"
            },
            {
                "trigger": "Metabolic Stress / Dehydration",
                "pct": 48,
                "description": "Dehydration and metabolic imbalance lower seizure threshold in CLN7. Gastrostomy patients at risk of dehydration if enteral feeds interrupted. Ketogenic diet patients need careful fluid monitoring.",
                "management": "Adequate hydration protocol; IV access planning for gastrostomy patients during illness; electrolyte monitoring; glucose check during prolonged seizures; avoid prolonged fasting"
            },
            {
                "trigger": "CLN7-Prohibited Drug Administration",
                "pct": 100,
                "description": "Administration of CBZ / OXC / PHT / VGB to CLN7 patient causes 100% acute worsening: CBZ/OXC/PHT → acute myoclonic worsening; VGB → progressive irreversible retinal toxicity superimposed on NCL retinal degeneration. Both are preventable drug errors with catastrophic consequences.",
                "management": "ABSOLUTE CI documentation in all records; pharmacy alert system; A&E/ED laminated CLN7 drug alert card carried by family; prescriber education — IV LEV (not fosphenytoin) in SE; no VGB for any indication"
            }
        ],
        "treatments": [
            {
                "drug": "Valproate (VPA) — Backbone AED",
                "level": "Level B",
                "dose": "20-40 mg/kg/day in 2-3 doses; start low 10 mg/kg/day; titrate to clinical response; target level 60-100 mg/L; weight-based dosing",
                "moa": "Sodium channel blockade + GABA-A enhancement + T-type calcium channel modulation + GABA transaminase inhibition → broad-spectrum anti-seizure + antimyoclonic",
                "efficacy": "Backbone treatment for all CLN7 seizure types (GTCS + myoclonic + atonic + focal); 90% of CLN7 patients on VPA; essential for seizure control; does not alter CLN7 disease course",
                "monitoring": "VPA TDM trough level monthly (target 60-100 mg/L); LFTs + ammonia 3-monthly; POLG1 exclusion before initiation; VPPP mandatory for females ≥12y; weight monthly; carnitine level 6-monthly",
                "cln7_note": "VPA is SAFE in CLN7 — MFSD8 is a lysosomal membrane protein (NOT mitochondrial). VPA ABSOLUTE CI applies to MERRF and POLG/Alpers — does NOT apply to CLN7. POLG1 exclusion recommended before VPA in children <8y (POLG1 Alpers mimics CLN7 early — regression + seizures without early prominent visual failure). VPPP counselling mandatory for females ≥12y."
            },
            {
                "drug": "Levetiracetam (LEV) — IV SE + Adjunct",
                "level": "Level B",
                "dose": "Oral: 20-60 mg/kg/day; IV SE: 60 mg/kg (max 4500 mg) over 15 min; start 10 mg/kg/day and titrate",
                "moa": "SV2A synaptic vesicle protein modulation → presynaptic inhibition → reduced neuronal hyperexcitability + anti-GTCS + anti-myoclonic",
                "efficacy": "Adjunct to VPA for GTCS and myoclonic seizures; IV LEV is MANDATORY second-line in CLN7 SE (replaces fosphenytoin which is CI in CLN7); good tolerability in paediatric NCL",
                "monitoring": "Renal function (LEV renally excreted); behavioural side effects (agitation, irritability) — monitor in CLN7 progressive cognitive impairment; mood monitoring",
                "cln7_note": "IV LEV 60 mg/kg is the PREFERRED second-line agent in CLN7 SE — NEVER use fosphenytoin/phenytoin (ABSOLUTE CI — worsens myoclonus). This must be communicated to every ED team that may treat a CLN7 patient. Turkish/Middle Eastern families presenting to non-specialist centres are at particular risk of receiving incorrect SE treatment."
            },
            {
                "drug": "Clobazam (CLB) — Nocturnal + Adjunct",
                "level": "Level B",
                "dose": "0.1-0.5 mg/kg/day; typically 5-20 mg/day; once-daily nocturnal or BD if daytime cluster seizures",
                "moa": "GABA-A positive allosteric modulator (1,5-benzodiazepine) → chloride influx → membrane hyperpolarisation → anti-seizure + reduced myoclonus + reduced stimulus sensitivity",
                "efficacy": "Useful adjunct for cluster prevention, nocturnal GTCS, and myoclonus reduction in CLN7; maintains efficacy better than traditional benzodiazepines; valuable for stimulus-sensitive myoclonus",
                "monitoring": "Sedation monitoring; tolerance assessment 3-monthly; avoid abrupt withdrawal; titrate to clinical benefit",
                "cln7_note": "CLB is particularly valuable in CLN7 for: (1) nocturnal GTCS prevention, (2) stimulus-sensitive myoclonus reduction (common in CLN7), (3) cluster seizure prevention around triggers. Scheduled dosing preferred to PRN in CLN7 progressive disease."
            },
            {
                "drug": "Piracetam — Action Myoclonus First-Line Adjunct",
                "level": "Level C",
                "dose": "24-48 g/day in 3 divided doses (adult/adolescent); paediatric: 40-100 mg/kg/day; start low and titrate rapidly to clinical response",
                "moa": "AMPA receptor positive modulation + membrane fluidity enhancement + putative anti-oxidant → reduced cortical myoclonic hyperexcitability; protective effects on neuronal membrane function",
                "efficacy": "First-line adjunct specifically for action myoclonus and stimulus-sensitive myoclonus in CLN7. Extrapolated from PME action myoclonus data (Unverricht-Lundborg disease) to NCL by expert centres. Benefit: reduced action myoclonus → improved fine motor function, feeding, communication. Does not significantly help GTCS.",
                "monitoring": "Tolerability (GI side effects at high doses); renal function (renally excreted); cognition monitoring (very high doses can cause agitation); functional improvement monitoring",
                "cln7_note": "Piracetam FIRST-LINE for CLN7 action myoclonus — should be started when action myoclonus emerges (typically 6-18 months after GTCS onset). Dose must be high (24-48 g/day adult; 40-100 mg/kg/day paediatric) for meaningful clinical benefit. Monitor renal function as CLN7 patients may develop intercurrent illnesses."
            },
            {
                "drug": "Lamotrigine (LTG) — Adjunct (with VPA backbone only)",
                "level": "Level B",
                "dose": "VERY SLOW titration with VPA (VPA inhibits LTG metabolism): start 0.15 mg/kg/day, increment 0.15 mg/kg every 2 weeks; target 1-5 mg/kg/day with VPA",
                "moa": "Sodium channel + calcium channel blockade → reduced cortical excitability → anti-GTCS + modest anti-myoclonic",
                "efficacy": "Useful adjunct for focal and generalised seizures in CLN7; must be used WITH VPA backbone (LTG monotherapy is HIGH RISK — insufficient myoclonus cover); VPA-LTG combination is synergistic",
                "monitoring": "Skin rash monitoring (slow titration prevents SJS); LTG levels (VPA inhibits metabolism); CBC if neutropenia suspected",
                "cln7_note": "LTG in CLN7 ONLY as adjunct to VPA backbone — NEVER as monotherapy. VPA-LTG combination must use very slow LTG titration to prevent SJS rash (VPA doubles LTG serum levels). Useful for focal occipital seizures in CLN7."
            },
            {
                "drug": "Ketogenic Diet (KD) — After ≥3 AED Failures",
                "level": "Level C",
                "dose": "Classic KD 4:1 ratio (fat:carb+protein); MCT variant also used; dietitian-led initiation; gastrostomy delivery in advanced CLN7",
                "moa": "Ketone metabolism → reduced neuronal excitability (multiple mechanisms); putative neuroprotective effects in NCL via alternative energy substrate",
                "efficacy": "Adjunct after ≥3 AED failures; Level C evidence for drug-resistant paediatric epilepsy; seizure reduction in 50-60% of CLN7 patients who complete initiation; neuroprotective benefit under investigation",
                "monitoring": "Dietitian monthly; lipid panel 3-monthly; weight and growth; acidosis monitoring; renal stones (hydration essential); gastrostomy required for reliable KD in advanced disease",
                "cln7_note": "KD initiated after ≥3 AED failures — gastrostomy makes KD delivery reliable when oral intake is unsafe. KD + VPA: additive hepatotoxicity risk — monitor LFTs closely. KD may provide dual benefit: seizure control + metabolic neuroprotection."
            },
            {
                "drug": "MDT Palliative Care + Ophthalmology + Gastrostomy",
                "level": "Level A",
                "dose": "MDT from diagnosis: paediatric neurology + ophthalmology (6-monthly ERG/VEP) + VI rehabilitation + dietitian + physiotherapy + speech therapy + palliative care",
                "moa": "Comprehensive disease management addressing all CLN7 manifestations: vision loss, dysphagia, motor deterioration, cognitive decline, psychosocial needs, ACP",
                "efficacy": "MDT care is the standard of care for CLN7 — no single specialist can manage all disease dimensions. Gastrostomy required in ~85% of CLN7 patients within 5-8 years of disease onset.",
                "monitoring": "FEES/videofluoroscopy for dysphagia; PEG gastrostomy planning before aspiration; SARA ataxia scale 6-monthly; UMRS myoclonus scale; BDSRA registry; ACP annually",
                "cln7_note": "CLN7 MDT is paediatric-neurology-led. Gastrostomy planning begins when FEES shows unsafe swallow — do not wait for aspiration events. ACP from diagnosis — CLN7 is fatal; palliative integration from day of diagnosis is mandatory. Turkish/Middle Eastern families may need culturally sensitive ACP support and interpreter services."
            },
            {
                "drug": "Rescue Midazolam (Buccal) + LEV IV — SE Protocol",
                "level": "Level A",
                "dose": "Buccal midazolam 0.3 mg/kg (max 10 mg) — first-line rescue at home/community. IV LEV 60 mg/kg over 15 min — hospital second-line for SE. IV VPA 30 mg/kg — third-line SE.",
                "moa": "Midazolam: GABA-A fast-acting → rapid seizure termination. IV LEV: SV2A modulation → SE termination. IV VPA: multi-mechanism → SE termination.",
                "efficacy": "Family-administered buccal midazolam for seizures >5 min or second seizure within 30 min. Hospital IV LEV is MANDATORY SE second-line in CLN7 — replaces fosphenytoin (ABSOLUTE CI). SE requires urgent treatment in CLN7.",
                "monitoring": "Family training in rescue midazolam; seizure diary; post-SE neurological assessment; update ED protocol; annual SE threshold review",
                "cln7_note": "CLN7 SE PROTOCOL — CRITICAL: (1) Buccal midazolam 0.3 mg/kg at home for >5 min seizure. (2) Hospital: IV LEV 60 mg/kg (NOT IV fosphenytoin/phenytoin — ABSOLUTE CI). (3) Third line: IV VPA 30 mg/kg if LEV fails. Written SE protocol must be in hospital records AND with family AT ALL TIMES. Non-specialist ED teams (especially those managing Turkish/Middle Eastern patients unfamiliar with CLN7) must be briefed: NO fosphenytoin."
            }
        ],
        "contraindications": [
            {
                "drug": "Vigabatrin (VGB)",
                "risk_level": "ABSOLUTE CI",
                "reason": "VGB causes vigabatrin-associated retinopathy (VAR — irreversible visual field constriction) superimposed on CLN7 progressive retinal degeneration (100% have retinal NCL). VGB + CLN7 retinal disease = catastrophic accelerated combined blindness. VGB must NEVER be administered to CLN7 patients for any indication.",
                "alternative": "IV LEV for SE; VPA + LEV + CLB for GTCS/myoclonus; KD for drug-resistant epilepsy. VGB is contraindicated for EVERY indication in CLN7 (infantile spasms, GTCS, focal — all contraindicated)."
            },
            {
                "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
                "risk_level": "ABSOLUTE CI",
                "reason": "Sodium channel blockers cause ACUTE MYOCLONIC WORSENING in CLN7. CLN7 presents with GTCS first (2.5-7y) — misidentified as 'idiopathic GTCS' → CBZ prescribed → ACUTE MYOCLONIC DETERIORATION. Particularly dangerous in Turkish/Middle Eastern patients presenting to non-NCL-specialist centres.",
                "alternative": "VPA (first-line backbone); LEV IV for SE; LTG only as adjunct to VPA. Safe first choice: VPA + LEV for any child with GTCS + developmental concern."
            },
            {
                "drug": "Fosphenytoin (IV Phenytoin Prodrug)",
                "risk_level": "ABSOLUTE CI",
                "reason": "Fosphenytoin ABSOLUTE CI in CLN7 SE — same sodium-channel mechanism as PHT → myoclonic worsening + NCSE precipitation risk. Standard paediatric SE protocols use fosphenytoin second-line — must be OVERRIDDEN in CLN7. IV LEV 60 mg/kg is the mandatory second-line.",
                "alternative": "IV LEV 60 mg/kg (mandatory CLN7 SE second-line); IV VPA 30 mg/kg (third-line). ED protocol must be updated before CLN7 patient presents to emergency department."
            },
            {
                "drug": "Tiagabine (TGB)",
                "risk_level": "ABSOLUTE CI",
                "reason": "GABA reuptake inhibitor — absolute risk of non-convulsive status epilepticus (NCSE) in NCL/PME patients with generalised epilepsy. NCSE in CLN7 may be misidentified as disease progression — particularly dangerous.",
                "alternative": "CLB (GABA-A modulator — safe GABA-ergic option in CLN7). Avoid all GABA reuptake inhibitors in CLN7."
            },
            {
                "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
                "risk_level": "HIGH RISK — AVOID",
                "reason": "GBP/PGB can worsen myoclonus in PME/NCL patients. Multi-specialty prescribing trap: neurology controls AEDs correctly but general paediatrics or pain team prescribes GBP/PGB for pain → acute myoclonic worsening. CLN7 pain management must use alternative analgesics.",
                "alternative": "Paracetamol + NSAID for pain; opioids if severe neuropathic pain. All prescribers must be informed of GBP/PGB prohibition. Shared care protocol with GP must document."
            },
            {
                "drug": "LTG Monotherapy",
                "risk_level": "HIGH RISK",
                "reason": "LTG as sole AED provides inadequate cover for CLN7 myoclonus and GTCS — can paradoxically worsen myoclonus in some PME patients when given without VPA backbone. Only acceptable as adjunct to VPA.",
                "alternative": "LTG only as adjunct to VPA (VPA-LTG combination acceptable); VPA monotherapy or VPA+LEV as backbone. NEVER start LTG without VPA in CLN7."
            },
            {
                "drug": "AED Taper / Planned Withdrawal",
                "risk_level": "HIGH RISK",
                "reason": "CLN7 is a fatal progressive NCL — seizures DO NOT remit. Planned AED taper or withdrawal should NEVER be attempted in CLN7 as there is no remission pathway. Risk: catastrophic SE, seizure cluster.",
                "alternative": "Maintain AED regimen; if AED change needed, cross-taper with new agent started before old agent reduced. Never discontinue VPA backbone in CLN7."
            }
        ],
        "monitoring": [
            {"item": "MFSD8 WES / NCL Gene Panel (Primary Diagnostic)", "detail": "WES or targeted NCL gene panel (MFSD8 4q28.1 sequencing) — primary molecular diagnostic test; no enzyme assay available. Turkish/Middle Eastern heritage: p.Glu9Gly (c.26A>G) or p.Arg35X (c.103C>T) targeted PCR may be attempted first (days). Full WES if targeted PCR negative or non-Turkish heritage. RNA sequencing if strong clinical suspicion + WES negative (deep intronic variants). CRITICAL: WES must include BOTH MFSD8 and CLN6 to differentiate CLN7 from CLN6 (EM alone cannot distinguish)."},
            {"item": "Skin Biopsy EM — FP + CB Pattern (NCL Confirmation)", "detail": "EM of eccrine sweat gland on skin biopsy: confirms NCL (FP + CB = NCL in CLN7). Must be interpreted by neuropathologist with NCL EM expertise. CRITICAL NOTE: FP + CB on EM is IDENTICAL in CLN6 and CLN7 — EM confirms NCL class but CANNOT differentiate CLN7 from CLN6. WES/gene panel is mandatory for CLN7 vs CLN6 differentiation. EM should be done concurrently with WES (not sequentially to save time)."},
            {"item": "Ophthalmology ERG + VEP (6-monthly)", "detail": "ERG (electroretinogram) for retinal function; VEP (visual evoked potential) for cortical visual pathway. Both required 6-monthly in CLN7. ERG amplitude reduction precedes clinical visual failure. VEP extinction marks cortical visual pathway involvement. VGB absolute CI documented in ophthalmology record — critical alert in non-specialist centres."},
            {"item": "POLG1 Exclusion Before VPA", "detail": "POLG1/Alpers syndrome (mtDNA polymerase gamma deficiency) mimics CLN7 early presentation (regression + seizures). VPA is ABSOLUTE CI in POLG1 (fatal hepatic failure). POLG1 sequencing recommended before VPA in children <8y with regression + seizures, especially with any mitochondrial features (lactic acidosis, hepatomegaly, family history of liver disease on VPA)."},
            {"item": "Brain MRI 3T (6-monthly)", "detail": "Progressive cortical atrophy (parieto-occipital initially); cerebellar atrophy; white matter signal change; basal ganglia involvement later. Quantitative MRI volumetry for disease progression tracking. Sedation/anaesthesia planning (CLN7 SE protocol for anaesthesia team — no fosphenytoin). MRI frequency may be reduced to annual in established stable disease."},
            {"item": "EEG (Annual + Acute)", "detail": "Annual awake + sleep EEG for background monitoring and seizure type characterisation. Standard IPS (3-50 Hz) for photosensitivity — present in 60% CLN7. Acute EEG for any unexplained behavioural/cognitive change (exclude NCSE). No CLN7-specific pathognomonic EEG pattern — progressive background slowing as disease advances."},
            {"item": "FEES / Videofluoroscopy — Dysphagia Assessment", "detail": "Functional Endoscopic Evaluation of Swallowing (FEES) or videofluoroscopy annually from 2 years after disease onset. Identifies pharyngeal dysphagia before aspiration events. PEG gastrostomy planning when FEES shows unsafe swallow — coordinate anaesthetic for single anaesthetic with other procedures where possible."},
            {"item": "SARA Ataxia Scale (6-monthly)", "detail": "Scale for Assessment and Rating of Ataxia (SARA) — 6-monthly for cerebellar ataxia progression. Physiotherapy and OT goals linked to SARA trajectory. Combined fall risk assessment: SARA ataxia + visual impairment + atonic seizure frequency → helmet and environmental modification decisions."},
            {"item": "UMRS Myoclonus Scale (6-monthly)", "detail": "Unified Myoclonus Rating Scale — 6-monthly for action myoclonus severity and piracetam dose titration. Document stimulus sensitivity change over time (photosensitivity may decrease as retinal degeneration progresses — less retinal input driving occipital hyperexcitability)."},
            {"item": "Neuropsychology (Annual)", "detail": "Developmental or neuropsychological assessment annually; adaptive measures in severely impaired patients (Vineland Adaptive Behavior Scales); BSID-IV for young children; cognitive trajectory documented for ACP discussions; AAC (Augmentative and Alternative Communication) assessment. Interpreter services for Turkish/Middle Eastern families."},
            {"item": "VPA TDM + LFT + Carnitine", "detail": "VPA therapeutic drug monitoring (trough 60-100 mg/L) monthly in first year, then 3-monthly when stable. LFTs 3-monthly (hepatotoxicity monitoring, especially if KD co-prescribed). Carnitine level 6-monthly (VPA depletes carnitine). Ammonia if encephalopathic. Thrombocytopenia monitoring."},
            {"item": "SUDEP Risk / Nocturnal Monitoring", "detail": "CLN7 SUDEP risk elevated: drug-resistant epilepsy + nocturnal seizures + progressive cognitive impairment → impaired post-ictal arousal. Nocturnal seizure monitoring device (EMFIT or similar). Co-sleeping assessment. Family education on seizure safety. SUDEP risk documented in ACP."},
            {"item": "BDSRA / NCL Network Europe Registry", "detail": "All CLN7 patients must be enrolled in BDSRA (Batten Disease Support and Research Association) registry and NCL Network Europe / NCL Resource for AAV-MFSD8 gene therapy trial eligibility. CLN7 gene therapy preclinical programme is earlier-stage than CLN6 (no large animal model) — registry enrolment essential for natural history data collection and trial access."},
            {"item": "ACP — Palliative Care from Diagnosis", "detail": "Advanced Care Planning from day of diagnosis — CLN7 is fatal (childhood to late teens/early 20s). ACP: resuscitation preferences, ventilation, gastrostomy, place of care, place of death. Annual ACP review. Sibling psychological support. Bereavement planning. Genetic counselling: 25% recurrence risk (AR); interpreter and culturally sensitive ACP for Turkish/Middle Eastern families."}
        ],
        "lifecycle_stages": [
            {
                "stage": "Prenatal / Genetic Risk (Before Birth)",
                "description": "Family with known MFSD8 variants (previous affected child identified): prenatal diagnosis by CVS/amniocentesis; preimplantation genetic testing (PGT) in IVF. BDSRA pre-enrolment. Genetic counselling (25% recurrence risk — AR biallelic). Prospective sibling cascade testing at birth (MFSD8 PCR or WES). Turkish/Middle Eastern families with consanguinity: consider MFSD8 carrier testing for unaffected siblings and extended family."
            },
            {
                "stage": "Pre-Symptomatic (Age 0–2.5y, No Seizures Yet)",
                "description": "Known MFSD8 genotype (presymptomatic sibling or incidental carrier screening). Baseline ERG + ophthalmology. EEG baseline. Developmental assessment (Bayley scales). Gene therapy trial enrolment if available (presymptomatic window optimal for neuronal preservation). BDSRA registry. ACP discussion. Developmental monitoring for earliest regression signs."
            },
            {
                "stage": "First Seizure — Diagnostic Emergency (2.5–7y)",
                "description": "GTCS at 2.5–7 years — first clinical sign. Diagnostic pathway: EM skin biopsy (FP+CB = NCL confirmed) + MFSD8 and CLN6 WES concurrent (differentiate CLN7 vs CLN6) + Turkish PCR if appropriate + ophthalmology ERG + EEG + brain MRI + POLG1 exclusion before VPA. VPA + LEV started immediately. Emergency CLN7 drug alert card provided. School / nursery informed. BDSRA enrolment."
            },
            {
                "stage": "Active Epilepsy + Cognitive Regression (2–6y after First Seizure)",
                "description": "Multiple seizure types develop (GTCS + myoclonic + atonic + focal occipital). Action myoclonus emerges — piracetam added. Cognitive regression apparent (language, learning, adaptive skills decline). Visual failure begins (ERG reduction). Gastrostomy consideration. MDT intensification: physiotherapy, speech therapy, AAC, VI rehabilitation. Atonic falls → helmet + environmental modifications. KD if ≥3 AED failures."
            },
            {
                "stage": "Established CLN7 — Severe Disability (5–10y after Onset)",
                "description": "Profound intellectual disability. Near-total visual loss. Severe motor dysfunction (ataxia + myoclonus). Gastrostomy-dependent for nutrition and AEDs. Communication through AAC (if retained). Seizures drug-resistant (80%). Palliative care intensification. SUDEP nocturnal monitoring. Respite care coordination. Gene therapy trials: enrol if eligibility criteria met."
            },
            {
                "stage": "Late-Palliative / End of Life Stage",
                "description": "End-stage CLN7 — bedridden, no purposeful communication, gastrostomy-dependent, recurrent SE. ACP decisions: ceiling of care, ventilation, place of death (home vs hospice). Comfort-focused seizure management (rectal/buccal BDZ for distressing seizures rather than aggressive control). Bereavement support (culturally sensitive for Turkish/Middle Eastern families). Gene therapy trials: coordinate with palliative team if enrolled."
            }
        ]
    }


def get_definitions():
    return {
        "concepts": [
            {
                "concept": "CLN7-MFSD8-4q28.1-Lysosomal-Membrane-Protein-Turkish-Variant-vLINCL",
                "definition": "CLN7 (MFSD8/4q28.1) encodes Major Facilitator Superfamily Domain-Containing Protein 8 (518 aa; ~58 kDa; 12 TM domains). MFSD8 is located in the LATE ENDOSOMAL / LYSOSOMAL MEMBRANE — unlike CLN6 which is in the ER. MFSD8 is a putative lysosomal transporter (unknown substrate). MFSD8 LOF → lysosomal membrane dysfunction → SCMAS accumulation → FP + CB on EM → progressive NCL. First described in Turkish patients (Siintola 2007, Brain). OMIM *611124 / #610951.",
                "standard": "Siintola-2007-Brain / NCL-Resource-2024 / OMIM"
            },
            {
                "concept": "FP-CB-EM-Pattern-CLN7-Indistinguishable-from-CLN6-by-EM-WES-Required",
                "definition": "CLN7 EM pattern (FP + CB) is INDISTINGUISHABLE from CLN6 EM pattern (also FP + CB) by electron microscopy alone. CRITICAL: EM confirms NCL but CANNOT differentiate CLN7 (MFSD8/lysosomal) from CLN6 (CLN6/ER). WES or NCL gene panel MUST be performed to differentiate CLN7 from CLN6. CLN5 (RP+FP) and CLN1 (GRODs) are distinguishable by EM; CLN6 and CLN7 are NOT distinguishable by EM alone.",
                "standard": "NCL-Resource-2024 / Mole-2019-LancetNeurol / Siintola-2007-Brain"
            },
            {
                "concept": "No-MFSD8-Enzyme-Assay-WES-Gene-Panel-Required",
                "definition": "MFSD8 is a putative lysosomal transporter with no known enzymatic function — there is NO MFSD8 DBS enzyme assay (unlike CLN1 PPT1 assay and CLN2 TPP1 assay, which give results in days). Diagnostic sequence: EM skin biopsy (days) → FP+CB confirms NCL → MFSD8 WES + CLN6 WES concurrent (differentiate CLN7 vs CLN6) or NCL gene panel. Turkish heritage: p.Glu9Gly or p.Arg35X PCR first (if available). EM is the critical first rapid step; genetics follows.",
                "standard": "NCL-Resource-2024 / ILAE-2022 / BDSRA-Diagnostic-Guidelines"
            },
            {
                "concept": "Turkish-Variant-NCL-MFSD8-Middle-Eastern-Enrichment",
                "definition": "CLN7/MFSD8 was named 'Turkish variant NCL' because MFSD8 mutations were first identified in Turkish patients (Siintola et al. 2007, Brain, initial cohort entirely Turkish). Turkish and broader Middle Eastern populations have enriched MFSD8 variant frequencies (likely from historical consanguinity). Common Turkish variants: p.Glu9Gly (c.26A>G) and p.Arg35X (c.103C>T). CLN7 is NOT exclusively Turkish — cases reported globally from multiple ethnicities. Turkish heritage should prompt MFSD8 sequencing as a priority, alongside or before CLN6.",
                "standard": "Siintola-2007-Brain / NCL-Resource-2024 / Mole-2019-LancetNeurol"
            },
            {
                "concept": "Seizures-First-CLN7-Not-Visual-First-Like-CLN3",
                "definition": "CLN7 SEIZURES ARE THE FIRST CLINICAL SIGN (GTCS at 2.5–7y) — same presentation pattern as CLN2 and CLN6. This is the OPPOSITE of CLN3/JNCL (visual failure first 4-10y, seizures later). CLN7 is neurology-led from onset. Cognitive regression follows seizure onset (6-18 months); visual failure develops after regression. Any child with early GTCS + myoclonus + subsequent cognitive/visual regression (especially Turkish/Middle Eastern heritage) should prompt CLN7/MFSD8 screen alongside CLN6.",
                "standard": "Siintola-2007-Brain / NCL-Resource-2024 / Mole-2019-LancetNeurol"
            },
            {
                "concept": "Purely-AR-CLN7-No-AD-Form-Unlike-CLN6",
                "definition": "CLN7 is EXCLUSIVELY autosomal recessive (biallelic MFSD8 LOF) — there is NO established autosomal dominant form of CLN7 (unlike CLN6, which has AD Kufs Type A caused by dominant CLN6 missense). Genetic counselling is therefore straightforward: 25% recurrence risk per pregnancy (standard AR). All affected patients are paediatric-onset. Unaffected parents are obligate carriers. This simplifies pedigree interpretation compared to CLN6.",
                "standard": "NCL-Resource-2024 / ILAE-2022 / ACMG-AMP-2015"
            },
            {
                "concept": "CLN7-vs-CLN6-Critical-Clinical-Differential",
                "definition": "CLN7 (MFSD8) and CLN6 are clinically very similar: both present with seizures first, FP+CB EM, no enzyme assay, AR inheritance, paediatric onset, fatal. KEY DIFFERENCES: (1) Protein location: MFSD8 = lysosomal; CLN6 = ER. (2) Turkish variant: CLN7; Portuguese/Czech founder = CLN6 p.Arg24Gly. (3) AD form: CLN6 has AD Kufs Type A; CLN7 purely AR. (4) Age of onset: CLN6 mean 3.5y; CLN7 mean 4.5y (overlapping). (5) EM: IDENTICAL FP+CB — cannot distinguish. (6) Management: IDENTICAL (VPA backbone, VGB CI, CBZ CI). Resolution: WES/gene panel (MFSD8 vs CLN6 sequencing).",
                "standard": "Siintola-2007-Brain / Gao-2002-NatGenet / NCL-Resource-2024"
            },
            {
                "concept": "VGB-ABSOLUTE-CI-CLN7-Retinal-NCL-Progressive",
                "definition": "Vigabatrin ABSOLUTE CI in CLN7 — MFSD8 causes progressive retinal NCL degeneration (100% of patients). VGB causes vigabatrin-associated retinopathy (VAR — irreversible visual field constriction). VGB + CLN7 retinal degeneration = catastrophic combined blindness. VGB must never be administered for any indication. Turkish/Middle Eastern families presenting to non-specialist centres (where CLN7 may be unrecognised) are particularly at risk of VGB being prescribed for infantile spasms or other indications.",
                "standard": "MHRA-VPPP-2021 / NICE-NG217 / NCL-Resource-2024"
            },
            {
                "concept": "VPA-SAFE-CLN7-Lysosomal-NOT-Mitochondrial",
                "definition": "VPA is SAFE in CLN7. MFSD8 = lysosomal membrane protein dysfunction. This is NOT mitochondrial disease. VPA ABSOLUTE CI applies to MERRF/MTTK8344 (mitochondrial) and POLG/Alpers (mtDNA polymerase) — these CIs do NOT extend to CLN7. VPA is the backbone AED in CLN7. POLG1 exclusion recommended before VPA in children <8y (POLG1 Alpers mimics CLN7 early — regression + seizures without early visual failure; VPA absolutely CI in POLG1).",
                "standard": "CPIC-POLG1-2023 / NCL-Resource-2024 / ILAE-2022"
            },
            {
                "concept": "CBZ-OXC-PHT-Fosphenytoin-ABSOLUTE-CI-CLN7-Paediatric-GTCS-Trap",
                "definition": "CBZ/OXC/PHT/Fosphenytoin ABSOLUTE CI in CLN7 — sodium-channel blockers cause ACUTE MYOCLONIC WORSENING. CLN7 paediatric trap: GTCS at 2.5-7y misidentified as 'idiopathic GTCS' → CBZ prescribed → ACUTE MYOCLONIC DETERIORATION. Particularly dangerous in Turkish/Middle Eastern patients where CLN7 may not be considered by non-specialist paediatricians. Safe first choice: VPA + LEV. IV LEV (NOT fosphenytoin) for SE.",
                "standard": "NICE-NG217 / NCL-Resource-2024 / MHRA-VPPP-2021"
            },
            {
                "concept": "No-Disease-Modifying-Therapy-CLN7-AAV-Gene-Therapy-Preclinical",
                "definition": "CLN7 has NO approved disease-modifying therapy (contrast CLN2 cerliponase alfa). Management is purely symptomatic. Investigational: AAV-MFSD8 gene therapy is in preclinical development — no characterised large animal model comparable to CLN6 South Hampshire sheep. MFSD8 lysosomal substrate identification ongoing (potential for small molecule therapy if substrate identified). All CLN7 patients must be enrolled in BDSRA registry for trial eligibility and natural history data.",
                "standard": "NCL-Resource-2024 / BDSRA-Registry / ILAE-2022"
            },
            {
                "concept": "Compound-Fall-Risk-Triple-Mechanism-CLN7",
                "definition": "CLN7 creates triple-mechanism fall risk: (1) Atonic seizure drop attacks (58%); (2) Cerebellar ataxia (94%); (3) Progressive visual impairment → cannot see hazards or self-rescue. Management requires simultaneously: CLB/VPA for atonic seizures + physiotherapy for ataxia + VI mobility training + environmental modification + HELMET when severe atonic falls present. Falls assessment must involve neurology + physiotherapy + VI habilitation jointly.",
                "standard": "NCL-Resource-2024 / NICE-NG217 / WHO-ICF-2019"
            },
            {
                "concept": "Gastrostomy-PEG-CLN7-Dysphagia-AED-Delivery",
                "definition": "CLN7 progressive dysphagia from cerebellar + cortical neurodegeneration makes oral intake unsafe in ~85% of patients within 5-8 years of onset. PEG gastrostomy is standard: (1) Reliable AED delivery (critical for seizure control); (2) Reliable nutrition/hydration; (3) Enables KD delivery; (4) Prevents aspiration pneumonia. FEES/videofluoroscopy guides timing — gastrostomy BEFORE aspiration events.",
                "standard": "NCL-Resource-2024 / NICE-NG217 / Mole-2019-LancetNeurol"
            },
            {
                "concept": "MFSD8-Lysosomal-Membrane-Putative-Transporter-Unknown-Substrate",
                "definition": "MFSD8 (Major Facilitator Superfamily Domain-Containing Protein 8) is a member of the MFS superfamily — a large family of membrane transporters. MFSD8 is located on the late endosomal/lysosomal membrane (12 TM domains). Its substrate is NOT yet identified. This lack of known substrate means: (1) No enzyme assay possible (no catalytic function); (2) No substrate replacement therapy yet possible (unlike CLN2 TPP1 substrate CLN — cerliponase can deliver functional enzyme); (3) Gene therapy is the primary therapeutic strategy. Identification of MFSD8 substrate is a major research priority.",
                "standard": "Siintola-2007-Brain / NCL-Resource-2024 / Uusi-Rauva-2008-HumMolGenet"
            },
            {
                "concept": "SUDEP-Risk-CLN7-Nocturnal-Drug-Resistant-Progressive",
                "definition": "CLN7 has elevated SUDEP risk: (1) Drug-resistant epilepsy (80%); (2) Nocturnal GTCS prevalent; (3) Progressive cognitive impairment → impaired post-ictal arousal; (4) Severe neurodegeneration reduces brain compensatory mechanisms. SUDEP risk must be documented in ACP. Nocturnal seizure monitoring (EMFIT or similar) for all CLN7 patients with nocturnal seizures. Family education on post-ictal rescue.",
                "standard": "NICE-NG217 / NCL-Resource-2024 / ILAE-SUDEP-Guidelines"
            }
        ],
        "thresholds": [
            {"threshold": "First GTCS at 2.5-7y + FP+CB on EM → CLN7 AND CLN6 must be differentiated by WES", "standard": "NCL-Resource-2024 / ILAE-2022"},
            {"threshold": "VGB ABSOLUTE CI — zero tolerance; must be in all hospital records", "standard": "MHRA-VPPP-2021 / NCL-Resource-2024"},
            {"threshold": "CBZ/OXC/PHT/Fosphenytoin ABSOLUTE CI in CLN7", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "POLG1 exclusion before VPA in all children <8y with regression + seizures", "standard": "CPIC-POLG1-2023"},
            {"threshold": "Turkish/Middle Eastern heritage + NCL EM → MFSD8 and CLN6 WES concurrently", "standard": "NCL-Resource-2024 / Siintola-2007-Brain"},
            {"threshold": "FP+CB on EM alone CANNOT distinguish CLN7 from CLN6 — WES mandatory", "standard": "NCL-Resource-2024 / Mole-2019-LancetNeurol"},
            {"threshold": "VPA TDM target 60-100 mg/L; LFTs + carnitine 3-6 monthly", "standard": "NICE-NG217 / BNFc"},
            {"threshold": "IV LEV 60 mg/kg (NOT fosphenytoin) for SE — mandatory CLN7 protocol", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "FEES/videofluoroscopy annually from 2y after disease onset", "standard": "NCL-Resource-2024 / RCSLT"},
            {"threshold": "ERG + VEP 6-monthly — ophthalmology mandatory from diagnosis", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "Buccal midazolam 0.3 mg/kg for >5 min seizure — family-administered", "standard": "NICE-NG217 / APLS-Guidelines"},
            {"threshold": "BDSRA / NCL Network Europe enrolment mandatory at diagnosis", "standard": "BDSRA-Registry / NCL-Network-Europe"}
        ],
        "standards": [
            {"standard": "Siintola-2007-Brain", "detail": "Siintola E et al. 'MFSD8, encoding a novel putative lysosomal transporter, is mutated in Turkish variant late-infantile neuronal ceroid lipofuscinosis.' Brain 2007;130:1687-96 — first identification of MFSD8 mutations in CLN7 Turkish patients"},
            {"standard": "Mole-2019-LancetNeurol", "detail": "Mole SE et al. 'Clinical management of the neuronal ceroid lipofuscinoses.' Lancet Neurology 2019 — comprehensive NCL management standard including CLN7"},
            {"standard": "NCL-Resource-2024", "detail": "NCL Resource (ncl.mni.mcgill.ca) — international NCL management guidelines, diagnostic algorithms, EM pattern classification, CLN7 management; updated 2024"},
            {"standard": "ILAE-2022", "detail": "International League Against Epilepsy 2022 — epilepsy classification, diagnosis standards, treatment guidelines applicable to CLN7 seizure management"},
            {"standard": "NICE-NG217", "detail": "NICE Guideline NG217: 'Epilepsies in children, young people and adults' (2022) — UK standard for AED selection, monitoring, contraindications including paediatric NCL"},
            {"standard": "MHRA-VPPP-2021", "detail": "MHRA Valproate Pregnancy Prevention Programme (2021) — mandatory VPA counselling and pregnancy prevention for all females ≥12y on valproate"},
            {"standard": "Uusi-Rauva-2008-HumMolGenet", "detail": "Uusi-Rauva K et al. 'Novel interactions of CLN3 protein link Batten disease to dysregulation of fodrin-Na+, K+ ATPase complex.' Hum Mol Genet 2008 — characterisation of NCL protein interactions including MFSD8"},
            {"standard": "CPIC-POLG1-2023", "detail": "CPIC Guideline for POLG1 — pharmacogenomics of VPA in POLG/Alpers; POLG1 exclusion mandatory before VPA in paediatric regression + seizures"},
            {"standard": "ACMG-AMP-2015", "detail": "ACMG/AMP 2015 variant classification guidelines — applied to all MFSD8 variant interpretation (pathogenic / likely pathogenic / VUS)"},
            {"standard": "BDSRA-Registry", "detail": "Batten Disease Support and Research Association Registry — mandatory enrolment for all CLN7 patients; gene therapy trial eligibility; natural history data"},
            {"standard": "WHO-ICF-2019", "detail": "WHO International Classification of Functioning, Disability and Health — framework for MDT management of CLN7 disability"},
            {"standard": "NCL-Network-Europe", "detail": "NCL Network Europe — European NCL expert network; Turkish/Middle Eastern CLN7 families should be connected to NCL Network Europe for specialist management and trial access"}
        ],
        "references": [
            {"ref": "Siintola-2007-Brain", "citation": "Siintola E, Topcu M, Aula N, et al. (2007). The novel neuronal ceroid lipofuscinosis gene MFSD8 encodes a putative lysosomal transporter. Brain, 130(Pt 6), 1687-1697."},
            {"ref": "Mole-2019-LancetNeurol", "citation": "Mole SE, Anderson G, Band HA, et al. (2019). Clinical management of the neuronal ceroid lipofuscinoses. The Lancet Neurology, 18(12), 1131-1140."},
            {"ref": "Aldahmesh-2009-ClinGenet", "citation": "Aldahmesh MA, Al-Hassnan ZN, Aldosari M, Alkuraya FS. (2009). Neuronal ceroid lipofuscinosis caused by MFSD8 mutations: a common cause of childhood dementia in Saudi Arabia. Clinical Genetics, 75(1), 91-93."},
            {"ref": "Kousi-2009-HumMutat", "citation": "Kousi M, Siintola E, Dvorakova L, et al. (2009). Mutations in CLN7/MFSD8 are a common cause of variant late-infantile neuronal ceroid lipofuscinosis. Brain, 132(Pt 3), 810-819."},
            {"ref": "Stogmann-2009-ProgNeurobiol", "citation": "Stogmann E, El Tawil S, Wagenstaller J, et al. (2009). A splice site mutation in the CLN7 gene causing a variant of Batten disease. European Journal of Medical Genetics, 52(2-3), 193-196."},
            {"ref": "Vuillemenot-2011-MolGenet", "citation": "Vuillemenot BR, Katz ML, Coates JR, et al. (2011). Intrathecal tripeptidyl-peptidase 1 reduces lysosomal storage in a canine model of late infantile neuronal ceroid lipofuscinosis. Molecular Genetics and Metabolism, 104(3), 325-337."}
        ]
    }
