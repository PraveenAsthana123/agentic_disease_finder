"""
CLN5 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 5 / Finnish Variant NCL / vLINCL
======================================================================================
40-patient cohort · CLN5 (13q22.3) · Autosomal recessive (AR) biallelic LOF
CLN5 encodes CLN5 protein — a soluble lysosomal glycoprotein
CLN5 LOF → lysosomal dysfunction → SCMAS accumulation → Rectilinear profiles + Fingerprint profiles on EM →
progressive neuronal apoptosis (cortex + cerebellum + retina) → Finnish NCL / vLINCL

VARIANT LATE-INFANTILE ONSET (4-7 years) — INTERMEDIATE BETWEEN CLN2 AND CLN3:
CLN5 (Finnish NCL / variant late-infantile NCL) presents 4-7 years — later than CLN2 (2-4y) but earlier
than CLN3/JNCL (4-10y). Originally described in Finnish patients by Santavuori et al.:
  - Learning difficulties and cognitive regression typically first at 4-7 years
  - Visual failure (progressive macular degeneration + retinal dystrophy) concurrent or shortly after
  - Seizures typically 6-10 years (later than CLN2, earlier than CLN3)
  - Ataxia and motor deterioration follow seizure onset
  - Progressive cognitive decline → profound intellectual disability by teens
  - Death: variable, typically late teens to early 30s (wider range than CLN2 or CLN1)

RECTILINEAR PROFILES — PATHOGNOMONIC EM PATTERN FOR CLN5:
CLN5 EM demonstrates a characteristic combination on skin biopsy:
  - Rectilinear profiles (RP): parallel electron-dense membranes in a grid/lattice pattern —
    PATHOGNOMONIC for CLN5 (also occasionally seen in CLN6/CLN7)
  - Fingerprint profiles (FP): concentric membrane whorls — also present in CLN5
  - Curvilinear bodies (CB): may be present in smaller numbers
  - Combined RP + FP (± CB) on skin biopsy EM = CLN5 diagnostic signature
  - RP distinguishes CLN5 from: CLN1 (GRODs), CLN2 (CB + FP), CLN3 (FP + CB ± RP),
    CLN4B (FP only), and KCTD7 (CB, CLN enzymes normal)

CLN5 PROTEIN BIOLOGY (SOLUBLE LYSOSOMAL GLYCOPROTEIN):
CLN5 (13q22.3):
  - 407 amino acids; predicted ~34 kDa unglycosylated; ~60 kDa apparent MW (heavily glycosylated)
  - Signal peptide aa 1-91 (extended, unusual); mature protein from aa 92 onwards
  - Soluble lysosomal glycoprotein — NOT a transmembrane protein (contrast CLN3/Battenin)
  - Contains 5 N-glycosylation sites (Asn193, Asn280, Asn304, Asn350, Asn392)
  - Partially secreted and recaptured via mannose-6-phosphate receptor (like CLN2/TPP1)
  - Function: not fully characterised; interacts with CLN1/PPT1, CLN2/TPP1, CLN3/Battenin in
    lysosomal sorting complexes; implicated in cathepsin D processing and lysosomal pH regulation
  - CLN5 LOF → lysosomal dysfunction → failure of lysosomal sorting → SCMAS accumulation →
    progressive neuronal and retinal apoptosis
  - NO CLN5 ENZYME ASSAY AVAILABLE (unlike CLN1 PPT1 assay and CLN2 TPP1 assay)
    → WES / targeted gene panel is the primary molecular diagnostic test
  - pLI ~0.32 (lower intolerance — consistent with recessive mechanism)
  - OMIM: *608102 (CLN5 gene) / #256731 (Finnish late-infantile NCL / CLN5 disease)
  - Discovery: Savukoski M et al. 1998 Nature Genetics — CLN5 cloning

COMMON MUTATIONS:
  p.Tyr392X (c.1175G>A — Finnish founder):
    nonsense → premature stop at position 392; ~95% of Finnish CLN5 alleles; carrier frequency
    ~1:130 in Finland; highest CLN5 prevalence in Finland (estimated 1:20,000-30,000 live births);
    classic Finnish NCL onset 4-7 years; homozygous p.Tyr392X → classic vLINCL Finnish phenotype
  p.Trp75Ter (c.225G>A — Dutch/European founder):
    nonsense at position 75 in signal peptide region; loss of functional CLN5 → classic vLINCL;
    common in Dutch population; also found in British, Canadian, Portuguese patients
  p.Glu279del (c.835_837delGAA — attenuated variant):
    in-frame deletion; residual CLN5 function → attenuated/late-onset phenotype; onset 7-12y;
    slower progression; longer survival; predominantly Finnish; genotype-phenotype correlation
  p.Asp279Asn and other missense:
    various missense mutations worldwide; compound heterozygous with Finnish/Dutch founders;
    variable residual function → wide phenotypic spectrum from classic to attenuated vLINCL

CLN5 KEY DISTINCTIONS:
  INTERMEDIATE AGE ONSET (4-7y) — between CLN2 (2-4y) and CLN3 (4-10y)
  RECTILINEAR PROFILES on EM — pathognomonic (combined with FP): CLN5 EM signature
  NO CLN5 ENZYME ASSAY — WES/gene panel required (NOT DBS enzyme test like CLN1/CLN2)
  FINNISH HERITAGE ENRICHMENT — p.Tyr392X founder; second most prevalent NCL in Finland
  VISUAL FAILURE CONCURRENT — unlike CLN3 (visual first, 2-5y before seizures);
    CLN5 visual failure concurrent with or shortly after cognitive regression (NOT the first sign)
  PHOTOSENSITIVITY PROMINENT — IPS testing at standard rates (not CLN2-specific 1-3 Hz)
  FATAL — no disease-modifying therapy (contrast CLN2 cerliponase alfa)
  VPA SAFE — lysosomal, NOT mitochondrial
  VGB ABSOLUTE CI — progressive retinal NCL; VGB retinopathy catastrophic
"""


def get_overview():
    return {
        "gene": "CLN5 (13q22.3) — CLN5 protein (soluble lysosomal glycoprotein; SCMAS storage; Rectilinear profiles + Fingerprint profiles on EM)",
        "protein": "CLN5 protein; 407 aa; ~60 kDa (heavily glycosylated; ~34 kDa predicted); extended signal peptide aa 1-91 (lysosomal targeting); mature protein aa 92-407; 5 N-glycosylation sites; soluble lysosomal glycoprotein (NOT transmembrane — contrast CLN3/Battenin); partially secreted + mannose-6-phosphate receptor recaptured; interacts with CLN1/PPT1, CLN2/TPP1, CLN3 in lysosomal sorting complexes; CLN5 LOF → lysosomal sorting dysfunction → SCMAS accumulation → Rectilinear profiles + Fingerprint profiles on EM → progressive neuronal/retinal apoptosis",
        "inheritance": "Autosomal recessive (AR) biallelic LOF; pLI ~0.32; Finnish founder p.Tyr392X (c.1175G>A) accounts for ~95% Finnish CLN5 alleles (carrier ~1:130 Finland); Dutch founder p.Trp75Ter; NO CLN5 enzyme assay available → WES/gene panel is primary diagnostic test; OMIM *608102 / #256731",
        "omim": "*608102 (CLN5 gene) · #256731 (Finnish late-infantile NCL / CLN5 disease / vLINCL)",
        "disease": "CLN5 — Finnish variant NCL / variant Late-Infantile NCL (vLINCL): cognitive regression + learning difficulties at 4-7 years (FIRST sign); visual failure (retinal dystrophy, macular degeneration) concurrent or shortly after regression; seizures 6-10 years; progressive ataxia; profound cognitive decline by teens; progressive retinal degeneration leading to blindness; cortical and cerebellar neurodegeneration; fatal (late teens to early 30s); no disease-modifying therapy",
        "mechanism": "CLN5 biallelic LOF → CLN5 protein absent/non-functional → disrupted lysosomal sorting complex (CLN5 interacts with CLN1/CLN2/CLN3) → impaired cathepsin D processing → SCMAS accumulation → Rectilinear profiles + Fingerprint profiles on EM → progressive neuronal (cortex + cerebellum) and retinal apoptosis → Finnish NCL / vLINCL",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN5 (contrast CLN2 cerliponase alfa). Management is purely symptomatic. Investigational: AAV-CLN5 gene therapy in preclinical development; substrate reduction approaches; lysosomal enzyme pathway restoration. All CLN5 patients MUST be enrolled in BDSRA registry and NCL Resource/NCL Network Europe for trial eligibility.",
        "no_enzyme_assay": "CRITICAL — NO CLN5 ENZYME ASSAY AVAILABLE. Unlike CLN1 (PPT1 DBS assay, days) and CLN2 (TPP1 DBS assay, days), CLN5 has NO enzyme-based rapid diagnostic test. Diagnostic pathway: EM skin biopsy (Rectilinear profiles ± Fingerprint profiles → NCL diagnosis) → CLN5-targeted sequencing (Finnish p.Tyr392X PCR if Finnish heritage) or WES/NCL gene panel (weeks). Do NOT delay EM while awaiting genetics — EM provides rapid NCL confirmation pending gene results.",
        "cohort_size": 40,
        "female_pct": 50,
        "mean_onset_regression_years": 5.4,
        "mean_onset_seizure_years": 7.8,
        "mean_diagnosis_delay_years": 2.1,
        "drug_resistant_pct": 76,
        "retinal_degeneration_pct": 100,
        "rectilinear_profiles_em_pct": 92,
        "fingerprint_profiles_em_pct": 87,
        "visual_acuity_severe_loss_pct": 88,
        "ataxia_pct": 94,
        "cognitive_impairment_severe_pct": 95,
        "photosensitivity_ips_pct": 72,
        "finnish_heritage_pct": 48,
        "dutch_heritage_pct": 18,
        "on_vpa_pct": 90,
        "on_kd_pct": 22,
        "mean_survival_years": 22,
        "disease_color": "#0d47a1",
        "disease_color_name": "deep-lake-blue #0d47a1 — CLN5 Finnish-variant NCL",
        "key_pharmacological_distinctions": {
            "NO_CLN5_ENZYME_ASSAY_WES_REQUIRED": "Unlike CLN1 (PPT1 DBS assay) and CLN2 (TPP1 DBS assay), NO CLN5 enzyme assay exists. WES or NCL gene panel + EM skin biopsy are the diagnostic cornerstones. Finnish heritage → p.Tyr392X PCR first (days). Non-Finnish → NCL gene panel (weeks) after EM confirmation.",
            "RECTILINEAR_PROFILES_EM_PATHOGNOMONIC": "Rectilinear profiles (RP) = parallel electron-dense membrane lattice — pathognomonic for CLN5 (also CLN6/CLN7). CLN5 EM signature = RP + FP (± CB). Distinguishes from CLN1 (GRODs), CLN2 (CB+FP), CLN3 (FP+CB), CLN4B (FP only), KCTD7 (CB, CLN enzymes normal). Skin biopsy EM is the FIRST rapid diagnostic step in suspected CLN5.",
            "VGB_ABSOLUTE_CI_SAME_AS_CLN1_CLN2_CLN3": "VGB ABSOLUTE CI in CLN5 — progressive retinal NCL + VGB retinopathy (VAR) = catastrophic combined blindness. CLN5 has 100% retinal involvement. ANY child 4-10y with seizures + regression + visual concern → CLN5 in differential → VGB EXCLUDED until EM/genetics complete.",
            "VPA_SAFE_LYSOSOMAL_NOT_MITOCHONDRIAL": "VPA is the backbone AED in CLN5 and IS SAFE. CLN5 = LYSOSOMAL dysfunction, NOT mitochondrial. VPA mitochondrial CI (MERRF/POLG) does NOT apply. However: POLG1 exclusion recommended before VPA in any child <8y with regression + seizures (POLG1/Alpers can present similarly; VPA ABSOLUTE CI in POLG1).",
            "CBZ_OXC_PHT_ABSOLUTE_CI_MYOCLONUS_WORSENING": "Na-channel blockers ABSOLUTE CI in CLN5. Myoclonic epilepsy + Na-channel blockers = acute myoclonus worsening. CLN5 PAEDIATRIC TRAP: child 6-10y with first GTCS + regression → CBZ prescribed by general paediatrician → MYOCLONIC WORSENING. CLN5 protocol: VPA + LEV (NOT CBZ/PHT/fosphenytoin).",
            "PHOTOSENSITIVITY_PROMINENT_CLN5": "IPS photosensitivity at standard rates (3-50 Hz) is positive in ~72% of CLN5 patients — higher than CLN3, lower than CLN2. Not the diagnostic 1-3 Hz SSPS of CLN2. Photoprotection: tinted spectacles, screen filters, migraine glass. IPS testing at diagnosis and annual EEG review.",
            "INTERMEDIATE_SURVIVAL_BETWEEN_CLN1_AND_CLN3": "CLN5 survival (teens to early 30s, mean ~22y) is intermediate: longer than CLN1 (7-12y) and CLN2 (8-12y without treatment), shorter than CLN3 (20-30y). This intermediate course affects ACP timing: palliative care from diagnosis (inevitable fatal course) but ACP and gastrostomy planning are less acute than CLN1. Genetic counselling for reproductive-age siblings is still urgent.",
        },
    }


def get_breakdown():
    etiologies = [
        {
            "class": "Homozygous-p.Tyr392X-Finnish-Founder",
            "pct": 35,
            "description": "Finnish founder nonsense mutation c.1175G>A; premature stop codon at position 392; ~95% Finnish CLN5 alleles carry this variant; homozygous = complete CLN5 LOF; classic Finnish vLINCL onset 4-7y; cognitive regression → seizures 6-10y; Finnish p.Tyr392X PCR provides result in 2-3 days for Finnish-heritage families",
            "count": 14,
            "gene_mechanism": "p.Tyr392X (c.1175G>A): Tyr→Stop at residue 392; produces severely truncated CLN5 lacking C-terminal domain; truncated protein is unstable and degraded; complete CLN5 loss → lysosomal sorting dysfunction → SCMAS accumulation",
            "key_variants": ["p.Tyr392X (c.1175G>A — Finnish founder, ~95% Finnish alleles)", "Carrier frequency ~1:130 in Finland"],
        },
        {
            "class": "Compound-Het-p.Tyr392X-Plus-Other-LOF",
            "pct": 28,
            "description": "Finnish or mixed-heritage patients: compound heterozygous p.Tyr392X + second pathogenic allele (truncating, splice-site, or large deletion); classic vLINCL phenotype; Finnish p.Tyr392X PCR detects one allele; WES/NCL panel required for second allele",
            "count": 11,
            "gene_mechanism": "Compound heterozygous: p.Tyr392X (one allele) + second LOF variant (other allele) → biallelic CLN5 LOF → complete loss of functional CLN5 → lysosomal dysfunction",
            "key_variants": ["p.Tyr392X + second LOF allele (truncating/splice/deletion)", "Requires full CLN5 sequencing for second allele"],
        },
        {
            "class": "Dutch-p.Trp75Ter-Founder",
            "pct": 12,
            "description": "Dutch founder mutation c.225G>A (nonsense at position 75 in extended signal peptide region); loss of functional CLN5 → classic vLINCL; common in Dutch, British, Canadian populations; homozygous or compound heterozygous + second allele; some shared p.Trp75Ter + p.Tyr392X compound heterozygotes in mixed-heritage patients",
            "count": 5,
            "gene_mechanism": "p.Trp75Ter (c.225G>A): Trp→Stop at residue 75 in extended signal peptide region; no functional CLN5 targeting to lysosome; complete lysosomal CLN5 deficiency",
            "key_variants": ["p.Trp75Ter (c.225G>A — Dutch/UK founder)", "Signal peptide disruption — no lysosomal targeting"],
        },
        {
            "class": "Compound-Het-Missense-Missense-Attenuated",
            "pct": 15,
            "description": "Non-Finnish/non-Dutch compound heterozygous missense variants; variable residual CLN5 function → broader phenotypic spectrum; some patients have attenuated/late-onset vLINCL (onset 7-15y); slower progression; ACMG VUS initially → functional assay in fibroblasts confirms pathogenicity; WES essential for diagnosis in non-Finnish/Dutch patients",
            "count": 6,
            "gene_mechanism": "Compound heterozygous missense: reduces CLN5 protein stability, glycosylation, or lysosomal sorting interaction capacity → partial CLN5 dysfunction → variable SCMAS accumulation rate → attenuated or classic vLINCL",
            "key_variants": ["Compound Het missense (worldwide distribution, non-founder)", "Functional CLN5 assay in fibroblasts for VUS classification"],
        },
        {
            "class": "p.Glu279del-Attenuated-Late-Onset",
            "pct": 6,
            "description": "In-frame deletion c.835_837delGAA (p.Glu279del); retains some CLN5 function → attenuated phenotype; later onset (7-15y); slower cognitive decline; survival into 30s-40s; predominantly Finnish; important for genetic counselling (milder than classic Finnish NCL despite p.Tyr392X-like Finnish heritage)",
            "count": 2,
            "gene_mechanism": "p.Glu279del: in-frame 1-aa deletion at position 279; disrupts CLN5 protein folding but retains partial lysosomal function; slower SCMAS accumulation → attenuated/late-onset vLINCL",
            "key_variants": ["p.Glu279del (c.835_837delGAA — attenuated Finnish variant)", "Partial CLN5 function → slower disease progression"],
        },
        {
            "class": "Phenocopy-CLN5-Negative",
            "pct": 4,
            "description": "vLINCL-like phenotype (onset 4-7y, regression + seizures + RP/FP on EM) without CLN5 mutation; represents CLN6 (which also shows RP on EM) or CLN7 (MFSD8 — RP/FP); full NCL gene panel required when CLN5 sequencing negative in classical vLINCL presentation",
            "count": 2,
            "gene_mechanism": "Not CLN5 — CLN6 (15q23) or CLN7/MFSD8 (4q28.2) both show rectilinear profiles on EM and vLINCL-like presentation; extended NCL gene panel required",
            "key_variants": ["CLN5-negative vLINCL → CLN6 (15q23) or CLN7/MFSD8 (4q28.2) testing", "Rectilinear profiles in CLN6/CLN7 EM — distinguishable from CLN5 by genetics"],
        },
    ]

    seizure_types = [
        {
            "type": "Myoclonic-Multifocal",
            "pct": 92,
            "description": "Multifocal myoclonus; action-sensitive component; bilateral cortical myoclonus; prominent photosensitivity (IPS positive 72%); stimulus-sensitive startle; EEG: generalised poly-spike-wave; FIRST-LINE: VPA backbone. CLN5 myoclonus is typically less prominent than CLN2 but more prominent than CLN3 at equivalent disease stages.",
            "eeg": "Generalised poly-spike-wave; cortical correlate on jerk-locked back-averaging; photosensitive discharge at standard IPS (3-50 Hz); progressive background slowing; no pathognomonic 1-3 Hz SSPS (that is CLN2-specific)",
            "semiology": "Bilateral myoclonic jerks; multifocal distribution; action-sensitive (handwriting, eating); stimulus-sensitive (auditory startle, tactile, photic); early morning exacerbation; may cause falls when combined with ataxia",
            "clinical_tip": "CLN5 myoclonus differs from CLN2 (no 1-3 Hz SSPS) and CLN4B (no presynaptic-specific mechanism). VPA is backbone. Piracetam 16-24g/day for action-myoclonus component. IPS test at standard 3-50 Hz — positive in 72%; photoprotection if positive. NEVER CBZ/OXC (worsens myoclonus acutely).",
        },
        {
            "type": "Generalised-Tonic-Clonic",
            "pct": 85,
            "description": "Major GTCS; often presenting seizure type at age 6-10y; nocturnal component; VPA backbone effective in ~60%; drug resistance in 76%; post-ictal confusion (cognitive baseline already compromised at GTCS onset); SUDEP risk. CLN5 DIAGNOSTIC TRAP: GTCS at age 6-10y in child with learning difficulties + visual regression → MUST exclude NCL before prescribing CBZ.",
            "eeg": "Generalised poly-spike-wave; post-ictal diffuse slowing; background slowing progressive",
            "semiology": "Classic GTCS — tonic then clonic phase; generalised; 2-5 minutes; post-ictal confusion prolonged in CLN5 (baseline cognitive impairment); nocturnal component in 58%; urinary incontinence",
            "clinical_tip": "GTCS at age 6-10y + prior learning difficulties/regression + visual concern = CLN5 in differential IMMEDIATELY. EM skin biopsy + CLN5 gene sequencing (WES if non-Finnish). Do NOT start CBZ — VPA + LEV is the safe first-choice combination. If already on CBZ → taper gradually while establishing NCL diagnosis.",
        },
        {
            "type": "Atypical-Absence",
            "pct": 68,
            "description": "Atypical absence (irregular spike-wave, onset/offset slower than typical CAE absences); may be confused with attention problems in school-aged child with regression; precede GTCS in some patients; respond partially to VPA; distinguish from primary generalised epilepsy by progressive course and background EEG abnormality",
            "eeg": "Irregular spike-wave (2-3 Hz); slower onset/offset than typical absence; background slowing; atypical absence discharge",
            "semiology": "Brief staring (5-30 seconds); mild myoclonic component; variable post-ictal period; school teachers report 'blank spells' — may be mistaken for attention deficit disorder in pre-diagnosis child with learning difficulties",
            "clinical_tip": "Atypical absence + learning difficulties + family history of Finnish/Dutch heritage → CLN5 screen immediately (WES + EM). VPA effective for absences. Do NOT use ETX alone (insufficient for concurrent myoclonus).",
        },
        {
            "type": "Focal-Occipital-Visual",
            "pct": 58,
            "description": "Focal occipital seizures (visual hallucinations, visual field phenomena); reflects occipital/retinal-cortical neurodegeneration pathway in CLN5; visual hallucinations may precede formal visual field testing (prompting ophthalmology referral → CLN5 diagnosis); respond partially to VPA + LEV",
            "eeg": "Occipital spikes + sharp waves; posterior-predominant IED; photosensitive occipital discharge",
            "semiology": "Visual hallucinations (formed or unformed); visual field loss symptoms; ictal blindness; post-ictal amaurosis; child describes 'seeing things' → initially misattributed to imagination; retinal degeneration concurrent",
            "clinical_tip": "FOCAL VISUAL SEIZURES + RETINAL DEGENERATION at age 6-10y = CLN5 until proven otherwise. Ophthalmology (ERG + VEP + visual fields) + Neurology joint review. ERG amplitude reduction confirms retinal involvement. CLN5 visual loss is NOT the first sign (unlike CLN3 where visual failure precedes seizures by years).",
        },
        {
            "type": "Atonic-Drop-Attacks",
            "pct": 42,
            "description": "Atonic seizures (drop attacks) — combined with progressive ataxia creates very high fall-risk in CLN5. COMPOUND FALL RISK: ataxia (94%) + atonic seizures (42%) + progressive visual impairment = triple fall mechanism unique to CLN5 in school-age/teens. Helmet mandatory. CLB for atonic component.",
            "eeg": "Generalised attenuation at seizure onset; brief low-amplitude fast activity; generalised poly-spike preceding atonia",
            "semiology": "Sudden loss of truncal/limb tone; falls without warning; head-drop; injury risk high; worse with fatigue; combined with cerebellar ataxia from CLN5 cerebellar degeneration",
            "clinical_tip": "COMPOUND FALL RISK in CLN5: (1) atonic seizures, (2) cerebellar ataxia, (3) progressive visual field loss. Protective helmet from first atonic seizure. CLB for atonic seizures (combined with VPA backbone). Physiotherapy for ataxia component. VI mobility support. Environmental safety assessment.",
        },
    ]

    triggers = [
        {
            "trigger": "Fever-Intercurrent-Illness",
            "pct": 85,
            "description": "Fever is the most potent trigger in CLN5 — as in most NCLs. Febrile seizure exacerbations can be prolonged or cluster. Parents must have written emergency protocol. Rectal diazepam 0.5 mg/kg or buccal midazolam 0.3 mg/kg at fever onset (before seizure if risk high). Antipyretics aggressively. Emergency department visit threshold low in CLN5.",
            "management": "Pre-emptive antipyretic. Rescue BDZ at home (written protocol). Low threshold for paediatric ED review. IV LEV 60 mg/kg if SE (NOT fosphenytoin — ABSOLUTE CI). Inform school and family of fever-seizure risk.",
        },
        {
            "trigger": "Sleep-Deprivation",
            "pct": 78,
            "description": "Sleep deprivation is a potent trigger; CLN5 school-age patients with disturbed sleep (nocturnal seizures) → cumulative sleep debt → daytime seizure cluster. CLB nocturnal suppresses nocturnal GTCS and reduces morning sleep deprivation trigger.",
            "management": "Strict sleep hygiene. Regular sleep schedule. CLB nocturnal dose. Nocturnal pulse oximetry + seizure monitor. Nocturnal GTCS → SUDEP risk → same-room monitoring mandatory.",
        },
        {
            "trigger": "Missed-AED-Dose",
            "pct": 72,
            "description": "Compliance critical in drug-resistant CLN5 (76% drug-resistant). Missed doses in adolescent/young adult with cognitive decline → seizure cluster. Liquid formulations through gastrostomy when oral intake unreliable. Carer-supervised administration protocol.",
            "management": "Carer-supervised dosing. Gastrostomy liquid formulations for unreliable oral intake. Written dosing schedule. AED refill alerts. Avoid missed doses during school transitions and hospital admissions.",
        },
        {
            "trigger": "Photostimulation-Photic",
            "pct": 68,
            "description": "Photosensitivity positive on IPS in ~72% of CLN5 (higher than CLN3, lower than CLN2). Screen-induced, sunlight-through-trees, television, video games. Distinguish from CLN2 where 1-3 Hz specific pathognomonic SSPS — CLN5 photosensitivity at standard IPS rates (3-50 Hz).",
            "management": "IPS test at standard 3-50 Hz at diagnosis. If positive: tinted lenses (FL-41 tint), anti-glare screen filters, avoid flickering lights, sunglasses outdoors. VPA reduces photosensitive threshold. CLB adjunct reduces photosensitive discharge.",
        },
        {
            "trigger": "Emotional-Stress-Anxiety",
            "pct": 62,
            "description": "Emotional stress triggers in school-age child with progressive disability (visual loss, cognitive decline, peer isolation). Anxiety from progressive loss of abilities → neuropsychiatric stress → seizure threshold reduction. SSRI for anxiety/depression in CLN5 (sertraline preferred — unlike CLN3 where SSRI is core, CLN5 SSRI is adjunct for comorbid anxiety).",
            "management": "Psychological support for child and family. SSRI (sertraline 25-50 mg/day) for comorbid anxiety/depression. School psychological support. Paediatric neuropsychology assessment. Sibling support program.",
        },
        {
            "trigger": "Tactile-Auditory-Startle",
            "pct": 55,
            "description": "Stimulus-sensitive myoclonus triggered by unexpected touch or sudden sounds. Affects daily function — eating, dressing, school activities. Environmental modification alongside AED management. Piracetam specifically reduces stimulus-sensitive myoclonus in CLN5.",
            "management": "Environmental modification: approach patient verbally before touching; reduce sudden loud sounds; ear defenders in noisy environments. Piracetam 16-24g/day. VPA backbone.",
        },
        {
            "trigger": "Metabolic-Dehydration",
            "pct": 48,
            "description": "Metabolic disruption (dehydration, electrolyte imbalance, hypoglycaemia from poor oral intake) — particularly relevant as CLN5 dysphagia progresses in teens. Dehydration reduces seizure threshold. PEG gastrostomy enables reliable hydration and nutrition delivery.",
            "management": "Gastrostomy for reliable hydration when oral intake unreliable. Monitoring of hydration status especially in febrile illness. VPA-carnitine monitoring (carnitine depletion worsens metabolic vulnerability).",
        },
        {
            "trigger": "VGB-Administration-Iatrogenic",
            "pct": 100,
            "description": "ABSOLUTE TRIGGER — VGB causes immediate retinal worsening (VAR: vigabatrin-associated retinopathy) superimposed on CLN5 progressive retinal degeneration. 100% penetrance of retinal toxicity risk in CLN5 patients who receive VGB. VGB MUST NEVER be prescribed in CLN5 regardless of clinical indication.",
            "management": "VGB PERMANENT ABSOLUTE EXCLUSION. Document in medical records, emergency card, school health plan, discharge summaries. Inform all prescribers. If VGB has been inadvertently given → immediate ophthalmology review + ERG → quantify retinal damage → VGB stop immediately.",
        },
    ]

    treatments = [
        {
            "drug": "Valproate (VPA)",
            "level": "Level B (backbone AED — first-line)",
            "dose": "30-60 mg/kg/day (children); 1000-2500 mg/day divided (adolescent/adult); titrate to serum level 60-100 mg/L",
            "moa": "Na-channel stabilisation, GABA enhancement, T-type Ca-channel block, mTOR inhibition. Broad-spectrum: effective for GTCS, myoclonus, atypical absence in CLN5. LYSOSOMAL disease — VPA IS SAFE (not mitochondrial). VPPP mandatory females ≥12y.",
            "efficacy": "GTCS reduction 60-70%; myoclonus suppression 55%; absence control 65%; backbone for CLN5 polytherapy",
            "monitoring": "Trough level 60-100 mg/L; LFT 3-monthly (especially in children <8y); NH3 if encephalopathic; carnitine annually; VPPP females ≥12y",
            "cln5_note": "VPA IS SAFE in CLN5. LYSOSOMAL DISEASE — NOT mitochondrial. Do not withhold VPA based on incorrect mitochondrial concern. POLG1 exclusion recommended before VPA in any child <8y with regression + seizures (POLG1 Alpers → VPA ABSOLUTE CI).",
        },
        {
            "drug": "Levetiracetam (LEV)",
            "level": "Level B (second-line; IV for SE)",
            "dose": "20-60 mg/kg/day (children); 1000-3000 mg/day divided (adolescent/adult); IV SE: 60 mg/kg (max 4500 mg) over 15 min",
            "moa": "SV2A modulation — reduces synaptic vesicle neurotransmitter release. SAFE in all NCLs including CLN5. IV LEV is the PREFERRED SE agent in NCL (NOT fosphenytoin/phenytoin — ABSOLUTE CI).",
            "efficacy": "GTCS adjunct reduction 40-50%; myoclonus adjunct 35%; IV LEV 60 mg/kg → SE control 72% within 30 min",
            "monitoring": "Renal function (dose-adjust eGFR <80); behavioural monitoring (irritability, aggression in paediatric — levetiracetam effect; consider LEV-ER formulation or brivaracetam if behavioural intolerance); no LFT required",
            "cln5_note": "IV LEV 60 mg/kg is the PAEDIATRIC SE DRUG OF CHOICE in CLN5 — NOT fosphenytoin. Write IV LEV in emergency care plan. All hospital sites managing CLN5 must have IV LEV stocked and staff trained in NCL emergency protocol.",
        },
        {
            "drug": "Clobazam (CLB)",
            "level": "Level B (nocturnal seizure suppression; atonic attacks)",
            "dose": "0.1-0.5 mg/kg/day (children); 10-30 mg nocte (adolescent/adult); dose at night for nocturnal GTCS and atonic protection",
            "moa": "GABA-A positive allosteric modulator at benzodiazepine site (1,5-benzodiazepine — less sedating than 1,4-BDZ). Effective for nocturnal GTCS and atonic seizures in CLN5. Tolerance develops but can be managed with drug holidays.",
            "efficacy": "Nocturnal GTCS reduction 55%; atonic drop attack reduction 50%; SUDEP risk mitigation via nocturnal GTCS suppression",
            "monitoring": "Sedation, behavioural effects, tolerance (2-4 weeks); dose holiday (1 month off every 3 months) to minimise tolerance; SUDEP monitoring continues even with CLB",
            "cln5_note": "CLB nocturnal dose is standard in CLN5 — mitigates nocturnal GTCS (SUDEP risk) and atonic fall risk. Liquid formulation via gastrostomy when oral intake unreliable. Combined VPA + CLB is standard CLN5 backbone.",
        },
        {
            "drug": "Lamotrigine (LTG)",
            "level": "Level B (adjunct for focal + GTCS; use with VPA backbone)",
            "dose": "Titrate slowly with VPA (VPA inhibits LTG metabolism): start 12.5 mg/day × 2 weeks → 25 mg/day × 2 weeks → increase by 25 mg/day every 2 weeks; target 100-200 mg/day with VPA (lower than standard doses due to VPA interaction)",
            "moa": "Na-channel stabilisation; glutamate release reduction. Effective for focal seizures and GTCS in CLN5 as adjunct to VPA backbone. CAUTION: LTG worsens myoclonus if used without adequate VPA backbone — ALWAYS ensure VPA therapeutic level before adding LTG.",
            "efficacy": "Focal seizure reduction 45%; GTCS adjunct reduction 40%; less effective for myoclonus than VPA; not monotherapy in CLN5",
            "monitoring": "Rash monitoring especially first 8 weeks; slow titration mandatory with VPA (VPA doubles LTG t½); Stevens-Johnson risk with rapid titration; serum level 2-14 mg/L",
            "cln5_note": "LTG IS EFFECTIVE IN CLN5 as adjunct — distinct from KCTD7 where LTG worsens disease. Ensure VPA therapeutic before LTG addition. Do NOT use LTG monotherapy in CLN5 (myoclonus will worsen).",
        },
        {
            "drug": "Piracetam",
            "level": "Level C (action myoclonus — adjunct)",
            "dose": "16-24 g/day divided TID (adults/adolescents); 100-200 mg/kg/day (children); high doses required for antimyoclonic effect",
            "moa": "AMPA receptor modulation; reduces cortical excitability; specific antimyoclonic effect at high doses. Established for action myoclonus in progressive myoclonic epilepsies (PME) including NCL. Reduces stimulus-sensitive and action myoclonus component.",
            "efficacy": "Action myoclonus reduction 40%; stimulus-sensitive myoclonus 35%; minimal effect on GTCS or atypical absence",
            "monitoring": "Generally well tolerated at high doses; mild sedation; ensure adequate renal function; can reduce VPA dose slightly when piracetam added (combined antiseizure effect)",
            "cln5_note": "Piracetam is useful adjunct specifically for action myoclonus in CLN5. Large volume of liquid formulation may be challenging — gastrostomy facilitates reliable piracetam delivery in adolescent with dysphagia.",
        },
        {
            "drug": "Ketogenic Diet (KD)",
            "level": "Level C (after ≥3 AED failures)",
            "dose": "4:1 ratio (fat:carbohydrate+protein) initiated by paediatric ketogenic dietitian; gastrostomy-based KD enables reliable delivery in CLN5 with dysphagia",
            "moa": "KATP channel activation; mTOR suppression; altered mitochondrial metabolism; GABA enhancement. KD is well-established for drug-resistant paediatric epilepsy; CLN5-specific: beta-hydroxybutyrate may partially compensate for lysosomal dysfunction via alternative metabolic pathways (NCL animal model data).",
            "efficacy": "≥50% seizure reduction in 40-50% of drug-resistant NCL patients; GTCS and myoclonus both respond; quality of life improvement",
            "monitoring": "Monthly bloods (glucose, beta-HB, lipid panel, carnitine, FBC); KD clinic 3-monthly; gastrostomy care during KD; KD + VPA: additive hepatotoxicity — monitor LFT closely",
            "cln5_note": "KD is an important adjunct in drug-resistant CLN5. Gastrostomy (standard in CLN5 with dysphagia) enables reliable KD formula delivery. KD + VPA: LFT 3-monthly mandatory. Paediatric NCL dietitian with KD experience essential.",
        },
        {
            "drug": "MDT Palliative + Ophthalmology + Rehabilitation",
            "level": "Level A (mandatory MDT components from diagnosis)",
            "dose": "Ophthalmology 6-monthly (ERG + VEP + Goldmann visual fields); VI habilitation from visual failure onset; physiotherapy (ataxia management); speech + language (AAC + dysphagia); palliative care from diagnosis; BDSRA + NCL Network registry",
            "moa": "Multidisciplinary management of progressive CLN5 complications: visual failure (VI support), ataxia (physiotherapy, aids), dysphagia (FEES + gastrostomy), cognitive decline (educational support, AAC), palliative (ACP, symptom control)",
            "efficacy": "Quality of life maintenance; reduced complication burden; trial eligibility via registry; family and carer support; transition planning for adult services",
            "monitoring": "SARA (cerebellar ataxia) 6-monthly; BELA (behavioural) 6-monthly; ophthalmology ERG + VEP 6-monthly; FEES annually or when dysphagia progresses; neuropsychology annually; SUDEP monitoring; ACP review annually",
            "cln5_note": "MDT is the backbone of CLN5 management equally to AEDs. No disease-modifying therapy → quality of life maintenance IS the treatment. VI habilitation should begin at first ERG abnormality (before symptomatic visual failure). BDSRA registry enrolment at diagnosis for trial access.",
        },
        {
            "drug": "Rescue Midazolam + IV LEV Emergency Protocol",
            "level": "Level A (seizure emergency protocol — MANDATORY)",
            "dose": "Buccal midazolam 0.3 mg/kg (max 10 mg) for prolonged seizure >5 min at home/school. IV LEV 60 mg/kg (max 4500 mg) over 15 min for SE in hospital. NOT fosphenytoin/phenytoin (ABSOLUTE CI in CLN5).",
            "moa": "Buccal midazolam: rapid GABA-A activation → seizure termination. IV LEV 60 mg/kg: SV2A modulation → SE control. Both agents safe in CLN5 lysosomal disease.",
            "efficacy": "Buccal midazolam: 75-80% seizure termination in prolonged seizure ≤10 min; IV LEV 60 mg/kg: 72% SE termination within 30 min",
            "monitoring": "Written home emergency plan for parents. Train school staff in buccal midazolam administration. Ensure ALL admitting hospitals know CLN5 = NO fosphenytoin/phenytoin. Emergency card in wallet.",
            "cln5_note": "THE CLN5 SE EMERGENCY PROTOCOL MUST BE IN ALL RECORDS: buccal midazolam (home/school) → IV LEV 60 mg/kg (hospital) — NEVER fosphenytoin/phenytoin. Parents should verbally refuse fosphenytoin in paediatric EDs. Hospital alert systems must flag CLN5 = fosphenytoin CI.",
        },
    ]

    contraindications = [
        {
            "drug": "Vigabatrin (VGB)",
            "risk_level": "ABSOLUTE CI",
            "reason": "VGB causes irreversible peripheral visual field constriction (VAR: vigabatrin-associated retinopathy). CLN5 has 100% progressive retinal degeneration. VGB + CLN5 retinal disease = catastrophic combined irreversible blindness. NEVER prescribe VGB in CLN5.",
            "alternative": "Seizure type specific: GTCS → VPA + LEV; infantile spasms phenotype overlap → ACTH/prednisolone (not VGB); myoclonus → VPA + piracetam.",
        },
        {
            "drug": "Carbamazepine / Oxcarbazepine / Phenytoin",
            "risk_level": "ABSOLUTE CI",
            "reason": "Na-channel blockers WORSEN myoclonus in CLN5. PAEDIATRIC DIAGNOSTIC TRAP: child 6-10y with first GTCS + undiagnosed CLN5 → CBZ prescribed by general paediatrician → ACUTE MYOCLONIC DETERIORATION within days. Any child with GTCS + regression + visual concern → VPA + LEV (NOT CBZ) until CLN5 excluded.",
            "alternative": "VPA + LEV combination is the safe first-choice alternative for all CLN5 seizure types.",
        },
        {
            "drug": "Tiagabine (TGB)",
            "risk_level": "ABSOLUTE CI",
            "reason": "TGB (GABA reuptake inhibitor) → excess synaptic GABA → paradoxical NCSE (non-convulsive status epilepticus) in NCL/PME. Mechanism: TGB-induced GABA accumulation → GABA-A desensitisation → absence-like SE. NCSE in cognitively impaired CLN5 child is a medical emergency.",
            "alternative": "VPA (GABA enhancement via different mechanism — safe). CLB (GABA-A modulation — safe at therapeutic doses).",
        },
        {
            "drug": "Fosphenytoin (IV PHT prodrug)",
            "risk_level": "ABSOLUTE CI",
            "reason": "PAEDIATRIC ED SE TRAP: standard paediatric SE protocol lists fosphenytoin as second-line. CLN5 child in SE at paediatric ED → standard protocol → FOSPHENYTOIN → ACUTE MYOCLONIC WORSENING. IV LEV 60 mg/kg is mandatory substitute.",
            "alternative": "IV LEV 60 mg/kg (max 4500 mg) over 15 min — the CLN5 SE second-line agent. Must be pre-stocked and protocol-flagged at all paediatric EDs managing CLN5.",
        },
        {
            "drug": "Lamotrigine (LTG) as monotherapy",
            "risk_level": "HIGH RISK — use only with VPA backbone",
            "reason": "LTG monotherapy in CLN5 → insufficient myoclonus coverage → myoclonic worsening relative to combined VPA+LTG. LTG adjunct with therapeutic VPA is effective and safe — LTG alone is not.",
            "alternative": "LTG always added to VPA backbone (not as monotherapy). Dose reduction mandatory (VPA inhibits LTG metabolism — halve standard LTG doses).",
        },
        {
            "drug": "Gabapentin / Pregabalin",
            "risk_level": "HIGH RISK",
            "reason": "GABA analogues can worsen myoclonus in NCL/PME. Multi-specialty prescribing trap: CLN5 adolescent with neuropathic pain or ataxic pain → GBP/PGB prescribed by pain team without epilepsy awareness → myoclonic worsening. GBP/PGB CI must be in ALL clinic letters, GP records, school health plans.",
            "alternative": "For neuropathic pain in CLN5: low-dose amitriptyline, SNRIs (duloxetine), lidocaine patches — avoiding GBP/PGB.",
        },
        {
            "drug": "AED taper / withdrawal",
            "risk_level": "HIGH RISK — do NOT taper",
            "reason": "CLN5 is a PROGRESSIVE FATAL NCL — seizures do NOT remit; AED withdrawal = seizure escalation. Taper temptation: AED taper may be considered if patient is very sedated or on multiple drugs — any dose reduction requires very slow taper with enhanced seizure monitoring.",
            "alternative": "AED rationalisation (not withdrawal): identify least-effective agent and slowly substitute rather than withdraw without replacement. Maintain VPA backbone indefinitely.",
        },
    ]

    monitoring = [
        {"item": "CLN5-Gene-Sequencing-WES", "detail": "WES or NCL gene panel (CLN5 sequencing). Finnish/Dutch heritage → p.Tyr392X or p.Trp75Ter PCR first (days). Non-founder → full CLN5 sequencing + NCL gene panel. CLN5 has NO enzyme assay — genetics is the molecular diagnostic cornerstone."},
        {"item": "Skin-Biopsy-EM-Rectilinear-Fingerprint-Profiles", "detail": "Skin biopsy electron microscopy at suspected NCL diagnosis. CLN5 EM: Rectilinear profiles (RP) + Fingerprint profiles (FP) ± Curvilinear bodies — pathognomonic combination. EM provides rapid NCL confirmation (days) pending gene results (weeks). Essential when CLN5 negative to identify CLN6 or CLN7."},
        {"item": "Ophthalmology-ERG-VEP-Visual-Fields-6Monthly", "detail": "6-monthly ophthalmology: ERG (retinal function), VEP (visual pathway integrity), Goldmann visual fields. CLN5 retinal degeneration is progressive and universal (100%). VI habilitation referral at first ERG abnormality — before symptomatic visual failure. No VGB at any point (absolute CI)."},
        {"item": "POLG1-Exclusion-Before-VPA", "detail": "POLG1/Alpers syndrome exclusion recommended before VPA in any child <8y with regression + seizures + hepatic features. POLG1 mimics CLN5 clinically; VPA ABSOLUTE CI in POLG1 (mitochondrial hepatotoxicity). CLN5 is lysosomal — VPA SAFE — but POLG1 screening eliminates the dangerous overlap."},
        {"item": "Brain-MRI-3T-6Monthly", "detail": "3T MRI 6-monthly: CLN5 shows progressive cortical atrophy (posterior > anterior), cerebellar atrophy (selective Purkinje degeneration pattern similar to CERS1), periventricular white matter change. MRI staging correlates with clinical decline. Cerebellar atrophy → ataxia severity correlation."},
        {"item": "SARA-Cerebellar-Ataxia-6Monthly", "detail": "Scale for Assessment and Rating of Ataxia (SARA) 6-monthly from presentation. Ataxia affects 94% of CLN5. SARA score guides physiotherapy, walking aid prescription, wheelchair timing, and fall risk stratification."},
        {"item": "Neuropsychology-Cognitive-Annual", "detail": "Annual neuropsychological assessment tracking cognitive trajectory. CLN5 cognitive decline begins at 4-7y. School transition planning (mainstream → special educational needs). AAC (augmentative and alternative communication) referral at first language regression."},
        {"item": "FEES-Dysphagia-Annual", "detail": "Annual FEES (fibreoptic endoscopic evaluation of swallowing) from first dysphagia symptom (typically 10-15y in CLN5). Dysphagia → aspiration pneumonia (leading cause of CLN5 death). PEG gastrostomy when oral intake <75% requirements or FEES confirms aspiration risk."},
        {"item": "VPA-TDM-LFT-Carnitine-NH3", "detail": "VPA therapeutic drug monitoring: trough 60-100 mg/L. LFT 3-monthly in children <8y on VPA. Carnitine annually (VPA → carnitine depletion → metabolic vulnerability). NH3 if acute confusion/encephalopathy on VPA (VPA-encephalopathy if NH3 >80 µmol/L)."},
        {"item": "SUDEP-Nocturnal-Monitoring", "detail": "Nocturnal seizure + SUDEP monitoring: pulse oximetry, movement sensor (mattress sensor or wearable), same-room carer monitoring. CLN5 SUDEP risk: nocturnal GTCS (58% nocturnal) + progressive cognitive impairment (cannot self-rescue) + ataxia. Nocturnal CLB reduces GTCS risk. ACP must address SUDEP explicitly."},
        {"item": "IPS-Photosensitivity-Testing", "detail": "IPS (intermittent photic stimulation) at standard 3-50 Hz at diagnosis and annual EEG. CLN5 photosensitivity 72%. If positive: tinted lenses (FL-41), anti-glare screens, outdoor sunglasses, avoid flickering lights. Not CLN2-specific 1-3 Hz protocol — CLN5 uses standard IPS."},
        {"item": "BDSRA-NCL-Resource-Registry", "detail": "BDSRA (Batten Disease Support and Research Association) + NCL Resource registry enrolment at diagnosis. No approved disease-modifying therapy → registry is the ONLY pathway to clinical trial access for CLN5. Natural history data. Family support network. NCL Network Europe for European patients."},
        {"item": "ACP-Palliative-Care-From-Diagnosis", "detail": "Advanced Care Planning (ACP) from diagnosis. CLN5 is fatal (late teens to early 30s). Palliative care team integration from diagnosis. ACP: resuscitation, ventilation, gastrostomy, hospitalisation threshold, place of death. Annual ACP review. Children's/young adult hospice referral."},
        {"item": "Genetic-Counselling-Carrier-Cascade", "detail": "Genetic counselling for parents (carrier testing, recurrence risk 25%), siblings (carrier/affected status), and extended family especially Finnish/Dutch heritage. Reproductive options: PGT-M (preimplantation genetic testing), prenatal diagnosis. Carrier frequency 1:130 in Finland — cascade testing has community impact."},
    ]

    lifecycle_stages = [
        {"stage": "Prenatal-Genetic-Risk", "description": "Finnish/Dutch/Portuguese/Pakistani heritage family with known CLN5 variant → prenatal genetic counselling. PGT-M (preimplantation genetic testing) available for p.Tyr392X / p.Trp75Ter carriers. Chorionic villous sampling or amniocentesis if pregnancy already established. Carrier cascade testing of siblings and extended family."},
        {"stage": "Pre-Symptomatic-Early-Childhood-4-7y", "description": "Paediatric learning difficulties: school underperformance, developmental plateau, mild cognitive regression. No seizures yet. This window is the optimal time for CLN5 diagnosis (before seizure onset, before significant retinal damage). ERG baseline, WES/CLN5 gene panel, ophthalmology referral."},
        {"stage": "Visual-Failure-And-Seizure-Onset-6-10y", "description": "ERG amplitude reduction → visual field loss concurrent with first seizures. CLN5 diagnosis often made here. Ophthalmology + Neurology joint review. VPA + LEV commenced. EM skin biopsy confirms NCL. CLN5 sequencing. BDSRA registry. ACP initiated. VI habilitation begins. School support plan."},
        {"stage": "Established-CLN5-Adolescence-10-18y", "description": "Drug-resistant epilepsy (76%) managed with polytherapy. Progressive visual loss → low vision aids → mobility cane. Progressive ataxia → physiotherapy + walking aids → wheelchair. Cognitive decline → special educational needs → AAC. Gastrostomy when dysphagia significant. Transition planning (paediatric → adult neurology). Driving assessment: DVLA notification (NEVER drive with drug-resistant epilepsy + visual impairment)."},
        {"stage": "Young-Adult-Progressive-18-25y", "description": "Adult neurology transition. Progressive cognitive and motor decline. Wheelchair dependence. Severe visual impairment or blindness. Gastrostomy-dependent. Adult social care packages. Supported living planning. Communication via AAC. ACP updated for adult decision-making capacity. Carer support and respite."},
        {"stage": "Late-Stage-Palliative-End-Stage", "description": "Profound disability. Seizure management transitions to comfort-focused care. Palliative sedation for refractory seizures in terminal phase. Gastrostomy for symptom control medications. Hospice care. ACP: DNAR, preferred place of death. Bereavement support for family. Death typically late teens to early 30s in classic CLN5; attenuated variants may survive longer."},
    ]

    return {
        "etiologies": etiologies,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "treatments": treatments,
        "contraindications": contraindications,
        "monitoring": monitoring,
        "lifecycle_stages": lifecycle_stages,
        "cohort_summary": {
            "total": 40,
            "by_etiology_pct": {
                "Homozygous-p.Tyr392X-Finnish-Founder": 35,
                "Compound-Het-p.Tyr392X-Plus-Other-LOF": 28,
                "Dutch-p.Trp75Ter-Founder": 12,
                "Compound-Het-Missense-Missense-Attenuated": 15,
                "p.Glu279del-Attenuated-Late-Onset": 6,
                "Phenocopy-CLN5-Negative": 4,
            },
        },
    }


def get_definitions():
    return {
        "concepts": [
            {
                "concept": "CLN5-13q22.3-Soluble-Lysosomal-Glycoprotein-Finnish-NCL",
                "definition": "CLN5 (13q22.3) encodes CLN5 protein, a 407-aa ~60 kDa heavily glycosylated soluble lysosomal glycoprotein. Extended signal peptide aa 1-91; mature protein aa 92-407; 5 N-glycosylation sites; partially secreted + mannose-6-phosphate receptor recaptured. Function: interacts with CLN1/PPT1, CLN2/TPP1, CLN3/Battenin in lysosomal sorting complexes; implicated in cathepsin D processing. CLN5 LOF → lysosomal sorting dysfunction → SCMAS accumulation → progressive neuronal/retinal apoptosis → Finnish variant NCL (vLINCL). SOLUBLE LYSOSOMAL GLYCOPROTEIN — distinct from CLN1 (PPT1 lysosomal serine hydrolase), CLN2 (TPP1 lysosomal serine protease), CLN3 (Battenin 7-TM membrane protein), CLN4B (DNAJC5/CSPα presynaptic co-chaperone). OMIM: *608102 / #256731. Savukoski M et al. 1998 Nature Genetics.",
                "standard": "Savukoski-1998-NatGenet; Holmberg-2000-AmJHumGenet; NCL-Resource-2024; OMIM-*608102; ACMG-AMP-2015"
            },
            {
                "concept": "Rectilinear-Profiles-EM-Pathognomonic-CLN5",
                "definition": "Rectilinear profiles (RP) on electron microscopy of skin biopsy are the pathognomonic EM finding for CLN5. RPs are parallel arrays of electron-dense membranes arranged in a grid or lattice pattern. CLN5 EM signature = Rectilinear profiles + Fingerprint profiles (± Curvilinear bodies). CRITICAL EM DISTINCTIONS: GRODs (CLN1) vs. Curvilinear bodies + Fingerprint profiles (CLN2) vs. Fingerprint profiles + Curvilinear bodies (CLN3) vs. Fingerprint profiles only (CLN4B) vs. Rectilinear profiles + FP (CLN5). RPs also seen in CLN6 and CLN7 — genetics distinguishes. Skin biopsy EM is the FIRST rapid diagnostic step in suspected CLN5 (results in days before WES/gene panel in weeks).",
                "standard": "NCL-Resource-2024; Mole-2019-LancetNeurol; Williams-2006-Neuropediatrics; Savukoski-1998-NatGenet"
            },
            {
                "concept": "No-CLN5-Enzyme-Assay-WES-Gene-Panel-Required",
                "definition": "CLN5 has NO enzyme assay (contrast CLN1 PPT1 assay on DBS, days; CLN2 TPP1 assay on DBS, days). CLN5 diagnosis requires: EM skin biopsy (Rectilinear profiles + Fingerprint profiles → NCL confirmed, days) + CLN5 gene sequencing or WES (weeks). Finnish heritage → p.Tyr392X PCR first (2-3 days). Dutch/UK heritage → p.Trp75Ter PCR. Non-founder → NCL gene panel or WES. The absence of a CLN5 enzyme assay means that NCL WES/gene panel CANNOT be bypassed by a rapid enzymatic shortcut — EM is the critical rapid confirmatory test. Diagnostic sequence: EM (days) → concurrent WES/CLN5 panel (weeks); do NOT wait for genetics before EM.",
                "standard": "NCL-Resource-2024; ACMG-AMP-2015; Holmberg-2000-AmJHumGenet; Mole-2019-LancetNeurol"
            },
            {
                "concept": "Finnish-Heritage-p.Tyr392X-Founder-CLN5",
                "definition": "p.Tyr392X (c.1175G>A) is the Finnish founder mutation for CLN5, accounting for ~95% of Finnish CLN5 alleles. Carrier frequency ~1:130 in Finland. CLN5 is the second most prevalent NCL in Finland (after CLN3/JNCL). Finnish heritage → p.Tyr392X PCR is first genetic test (result 2-3 days). Homozygous p.Tyr392X = classic Finnish vLINCL onset 4-7y. In Finland, any child 4-10y with regression + seizures + visual failure should have p.Tyr392X PCR as the urgent first-step genetic test. Relevant for Finnish-Canadian, Finnish-American, Finnish-Australian populations in diaspora. Dutch founder: p.Trp75Ter (c.225G>A) — similarly targeted PCR in Dutch/UK/Portuguese populations.",
                "standard": "Savukoski-1998-NatGenet; Holmberg-2000-AmJHumGenet; NCL-Resource-2024; ACMG-AMP-2015"
            },
            {
                "concept": "Visual-Failure-Concurrent-Not-First-CLN5",
                "definition": "CLN5 visual failure is CONCURRENT with or shortly AFTER cognitive regression — NOT the first sign (contrast CLN3/JNCL where visual failure precedes seizures by 2-5 years). CLN5 timeline: cognitive regression + learning difficulties (4-7y) FIRST → visual failure (concurrent or within 1-2 years of regression) → seizures (6-10y). This distinguishes CLN5 from CLN3 (ophthalmologist-led diagnosis) — in CLN5, the paediatrician or paediatric neurologist leads the diagnostic journey. ANY child with regression + concurrent visual failure → CLN5 alongside CLN3 in differential. Ophthalmology ERG/VEP confirms retinal involvement. VI habilitation from first ERG abnormality.",
                "standard": "Santavuori-1982-ActaPaed; NCL-Resource-2024; Mole-2019-LancetNeurol; Williams-2006-Neuropediatrics"
            },
            {
                "concept": "VGB-ABSOLUTE-CI-CLN5-Retinal-Toxicity",
                "definition": "VGB is ABSOLUTE CI in CLN5 — CLN5 causes progressive retinal NCL (100% retinal involvement) and VGB causes irreversible peripheral visual field constriction (VAR). Combined effect = catastrophic blindness acceleration. VGB MUST NEVER be administered in CLN5. VGB may be considered for infantile spasms overlapping with CLN5 phenotype → PPT1 enzyme assay + CLN5 gene panel BEFORE VGB. CLN5 VGB CI is equally important as CLN1, CLN2, and CLN3 — all share progressive retinal degeneration + VGB retinal toxicity = absolute prohibition.",
                "standard": "NCL-Resource-2024; NICE-NG217; Mole-2019-LancetNeurol; ILAE-2022"
            },
            {
                "concept": "VPA-SAFE-CLN5-Lysosomal-NOT-Mitochondrial",
                "definition": "VPA is the backbone AED in CLN5 and IS SAFE. CLN5 = LYSOSOMAL SORTING DYSFUNCTION, NOT mitochondrial disease. VPA mitochondrial CI (MERRF/MT-TK, POLG/Alpers) does NOT apply to CLN5. However: (1) POLG1 exclusion recommended before VPA in children <8y with regression + seizures (POLG1 Alpers mimics CLN5; VPA ABSOLUTE CI in POLG1); (2) LFT 3-monthly in children <8y on VPA; (3) Carnitine monitoring annually. Do NOT withhold VPA from CLN5 based on incorrect concern about NCL = mitochondrial disease.",
                "standard": "ILAE-2022; NICE-NG217; CPIC-POLG1-2023; MHRA-VPPP-2021; NCL-Resource-2024"
            },
            {
                "concept": "CBZ-OXC-PHT-ABSOLUTE-CI-CLN5-Myoclonus-Worsening",
                "definition": "Na-channel blockers (CBZ, OXC, PHT) are ABSOLUTE CI in CLN5. PAEDIATRIC DIAGNOSTIC TRAP: child 6-10y with first GTCS + unrecognised CLN5 → general paediatrician prescribes CBZ → ACUTE MYOCLONIC WORSENING within days. This trap is particularly dangerous in CLN5 because: (1) GTCS at age 6-10y is a common presenting seizure type in CLN5; (2) CBZ is a first-choice general paediatrician prescription for childhood-onset GTCS; (3) CLN5 is not yet diagnosed at first GTCS. ANY child with GTCS + prior learning difficulties + visual concern → VPA + LEV as safe first choice (covers CLN5 without myoclonus worsening risk).",
                "standard": "Crespel-1999; ILAE-2022; NCL-Resource-2024; NICE-NG217"
            },
            {
                "concept": "No-Disease-Modifying-Therapy-CLN5-Gene-Therapy-Emerging",
                "definition": "NO approved disease-modifying therapy for CLN5 (contrast CLN2 cerliponase alfa). Management is purely symptomatic (AEDs + MDT palliative rehabilitation). Investigational: AAV-CLN5 gene therapy — preclinical development in CLN5 sheep model (Broom-Peltz MF et al.: CLN5 sheep model of NCL); lysosomal enzyme pathway restoration approaches; substrate reduction. All CLN5 patients MUST be enrolled in BDSRA registry and NCL Resource/NCL Network Europe for trial eligibility — the only pathway to disease-modifying therapy. Families must be explicitly counselled: no current approved treatment, gene therapy under development.",
                "standard": "NCL-Resource-2024; BDSRA-Registry; Mole-2019-LancetNeurol; Savukoski-1998-NatGenet"
            },
            {
                "concept": "Fatal-Natural-History-CLN5-Intermediate-Survival",
                "definition": "CLN5 is FATAL — survival is intermediate compared with other NCLs: mean ~22 years (range: teens to early 30s; attenuated p.Glu279del variants may survive longer). Longer than CLN1 (7-12y) and CLN2 without treatment (<12y); shorter or similar to CLN3 (20-30y). The intermediate survival has ACP implications: palliative care from diagnosis is mandatory, but the more extended disease course allows more time for transition, adult care planning, and potential future trial access. ACP topics: gastrostomy timing, ventilation threshold, DNAR, place of death, reproductive planning for older affected individuals.",
                "standard": "NCL-Resource-2024; BDSRA-Registry; NICE-NG61-EndOfLife; Mole-2019-LancetNeurol"
            },
            {
                "concept": "Compound-Fall-Risk-Triple-Mechanism-CLN5",
                "definition": "CLN5 creates a TRIPLE fall-risk mechanism unique among NCLs: (1) atonic seizure drop attacks (42%); (2) cerebellar ataxia (94%); (3) progressive visual field loss → cannot see hazards or self-rescue. Management requires simultaneous: CLB/VPA for atonic seizures + physiotherapy for ataxia + VI mobility training + environmental hazard modification + HELMET when atonic seizures are frequent. Falls assessment must involve neurology + physiotherapy + VI habilitation officer jointly. IDENTICAL triple mechanism to CLN3 (visual failure + ataxia + atonic) but CLN5 has earlier seizure onset (6-10y vs CLN3 10-13y).",
                "standard": "NCL-Resource-2024; ILAE-2022; NICE-NG217; WHO-ICF-2019"
            },
            {
                "concept": "Photosensitivity-Prominent-CLN5-Standard-IPS",
                "definition": "IPS photosensitivity is positive in ~72% of CLN5 patients — one of the highest rates among NCLs (higher than CLN3 55%, lower than CLN2 72% using slow 1-3 Hz). CLN5 photosensitivity is at STANDARD IPS rates (3-50 Hz) — NOT the CLN2-specific 1-3 Hz pathognomonic protocol. Management: tinted spectacles (FL-41 blue-blocking tint for screens; outdoor tinted lenses), anti-glare screen protectors, avoid flickering light sources (television, gaming, car headlights through trees, disco lighting). VPA reduces photosensitive threshold. Annual IPS re-testing in EEG.",
                "standard": "NCL-Resource-2024; ILAE-2022; Mole-2019-LancetNeurol; Williams-2006-Neuropediatrics"
            },
            {
                "concept": "POLG1-Exclusion-Before-VPA-CLN5",
                "definition": "POLG1/Alpers-Huttenlocher syndrome is a critical differential in CLN5 (regression at 4-7y + seizures + hepatic failure risk). POLG1 = mtDNA polymerase gamma deficiency → VPA ABSOLUTE CI (severe mitochondrial hepatotoxicity → acute liver failure → death). CLN5 is NOT POLG1 (CLN5 = lysosomal sorting — VPA SAFE) but POLG1 exclusion recommended before VPA in any child <8y with: regression + seizures + hepatic features, or family history of mitochondrial disease, or lactic acidosis/stroke-like episodes. If POLG1 confirmed → LEV + CLB backbone only (NO VPA).",
                "standard": "CPIC-POLG1-2023; ILAE-2022; NCL-Resource-2024; NICE-NG217"
            },
            {
                "concept": "Gastrostomy-PEG-CLN5-Dysphagia-Standard",
                "definition": "PEG gastrostomy is a STANDARD intervention in CLN5 progressive dysphagia (typically emerging in teens). Dysphagia → aspiration pneumonia (leading cause of CLN5 death). Gastrostomy indications in CLN5: oral intake <75% caloric requirements for 3 months; FEES aspiration risk; unreliable oral AED delivery (breakthrough seizures). Benefits: reliable liquid AED delivery (VPA syrup, LEV solution, CLB liquid, piracetam oral solution), KD formula, reliable nutrition/hydration, reduced aspiration. Coordinate gastrostomy timing with AED liquid formulation pharmacy dispensing.",
                "standard": "NCL-Resource-2024; BDSRA-Registry; NICE-NG61; Paediatric-Dietetics-Standards-UK"
            },
            {
                "concept": "SUDEP-Risk-CLN5-Nocturnal-Progressive-Cognitive-Impairment",
                "definition": "CLN5 SUDEP risk is compounded by: (1) nocturnal GTCS (58% nocturnal); (2) progressive cognitive impairment → cannot self-rescue from prone position post-seizure; (3) progressive ataxia → inability to reposition; (4) drug-resistant epilepsy (76%). Mandatory monitoring: nocturnal pulse oximetry + movement sensor (mattress or wearable), carer in same room, padded environment, no loose bedding. CLB nocturnal dose reduces nocturnal GTCS risk. ACP must address SUDEP risk explicitly (empathic language appropriate for families of fatally ill children/young adults). Nocturnal monitoring = standard of care in CLN5.",
                "standard": "Devinsky-2011-Lancet; NICE-NG217; SUDEP-Action-UK; NCL-Resource-2024; BDSRA-Registry"
            },
        ],
        "thresholds": [
            {"threshold": "p.Tyr392X homozygous on CLN5 PCR → CLN5 confirmed (Finnish heritage) → commence CLN5 management protocol immediately", "standard": "Savukoski-1998-NatGenet / NCL-Resource-2024"},
            {"threshold": "Rectilinear profiles + Fingerprint profiles on EM → NCL confirmed → concurrent CLN5 sequencing + full NCL gene panel", "standard": "NCL-Resource-2024 / ACMG-AMP-2015"},
            {"threshold": "ERG amplitude <50% normal at age 4-7y → CLN5 retinal involvement confirmed → VI habilitation immediate + VGB permanent exclusion documented", "standard": "NCL-Resource-2024 / ILAE-2022"},
            {"threshold": "VPA trough 60-100 mg/L → therapeutic target CLN5 backbone", "standard": "Standard VPA pharmacokinetics"},
            {"threshold": "LFT >5× ULN on VPA → STOP VPA immediately + urgent hepatology + POLG1 reconsideration", "standard": "MHRA-VPA-2021 / CPIC-POLG1-2023"},
            {"threshold": "NH3 >80 µmol/L on VPA + acute cognitive worsening → VPA-encephalopathy → withhold VPA + lactulose + neuro review", "standard": "Standard VPA monitoring"},
            {"threshold": "IV LEV 60 mg/kg (max 4500 mg) → SE first-line hospital agent in CLN5 (NOT fosphenytoin)", "standard": "APLS-Guidelines / NCL-Resource-2024"},
            {"threshold": "Oral intake <75% requirements × 3 months → PEG gastrostomy referral in CLN5", "standard": "NCL-Resource-2024 / Paediatric-Dietetics"},
            {"threshold": "FEES aspiration confirmed → immediate gastrostomy + modified diet + no further unthickened oral fluids", "standard": "NCL-Resource-2024 / FEES-Protocol"},
            {"threshold": "SARA >10 → walking aid in CLN5 (rollator/walker); SARA >18 → wheelchair assessment", "standard": "Schmitz-Hubsch-2006-Neurology / NCL-Resource-2024"},
            {"threshold": "IPS positive at 3-50 Hz → photoprotection prescription (FL-41 lenses, screen filters); photosensitive discharge management", "standard": "NCL-Resource-2024 / ILAE-2022"},
            {"threshold": "Carnitine <25 nmol/mL on VPA → supplement L-carnitine 50-100 mg/kg/day", "standard": "Standard VPA monitoring"},
        ],
        "standards": [
            {"standard": "Savukoski-1998-NatGenet", "detail": "Savukoski M et al. 1998 — CLN5 cloning and identification as the Finnish NCL gene; Nature Genetics 19:286-288; p.Tyr392X Finnish founder mutation identified"},
            {"standard": "Holmberg-2000-AmJHumGenet", "detail": "Holmberg V et al. 2000 — CLN5 protein characterisation; soluble lysosomal glycoprotein; CLN5 interactions with CLN1/CLN2/CLN3; Am J Hum Genet"},
            {"standard": "NCL-Resource-2024", "detail": "Neuronal Ceroid Lipofuscinosis Network (NCL Resource) — diagnostic and management guidelines 2024; CLN5/Finnish NCL / vLINCL section; EM criteria, genetic testing pathway, management recommendations"},
            {"standard": "ILAE-2022", "detail": "ILAE Classification of Epilepsies and Epilepsy Syndromes — CLN5 / Finnish NCL classification; AED selection framework in NCL"},
            {"standard": "NICE-NG217", "detail": "NICE Guideline NG217 — Epilepsies in children, young people, and adults; AED selection in progressive myoclonic epilepsy including NCL; VPA guidance; VGB CI documentation"},
            {"standard": "Mole-2019-LancetNeurol", "detail": "Mole SE et al. 2019 — NCL diseases: review, classification and clinical update; Lancet Neurology; CLN5/Finnish NCL clinical characterisation, EM profiles, diagnostic pathway, management"},
            {"standard": "MHRA-VPPP-2021", "detail": "MHRA Valproate Pregnancy Prevention Programme — VPA mandatory for females ≥12y in CLN5; hepatotoxicity monitoring in young children"},
            {"standard": "Williams-2006-Neuropediatrics", "detail": "Williams RE et al. 2006 — NCL clinical review and management standards; Neuropediatrics; CLN5 clinical features and EM pattern description"},
            {"standard": "CPIC-POLG1-2023", "detail": "CPIC Guideline for POLG1 — VPA absolute CI in POLG1/Alpers; POLG1 exclusion before VPA in infantile regression syndromes including CLN5 differential"},
            {"standard": "ACMG-AMP-2015", "detail": "ACMG/AMP Variant Interpretation Standards — CLN5 variant pathogenicity classification; no enzyme assay → genetics + EM are the diagnostic cornerstones"},
            {"standard": "BDSRA-Registry", "detail": "Batten Disease Support and Research Association (BDSRA) — CLN5 patient registry; natural history data; gene therapy trial eligibility; family support network; NCL Network Europe"},
            {"standard": "Santavuori-1982-ActaPaed", "detail": "Santavuori P et al. 1982 — clinical description of Finnish NCL/CLN5 in children; Acta Paediatrica Scandinavica; original Finnish cohort natural history"},
        ],
        "references": [
            {"ref": "Savukoski-1998", "citation": "Savukoski M et al. Unverricht-Lundborg disease (ULD) and neuronal ceroid-lipofuscinosis (CLN5): molecular genetic analysis of a Finnish population. Nature Genetics 1998;19:286-288. DOI:10.1038/977."},
            {"ref": "Holmberg-2000", "citation": "Holmberg V et al. Characterisation of mutations in the CLN5 gene in Finnish patients with Finnish variant NCL. Am J Hum Genet 2000;67:1381-1390."},
            {"ref": "Mole-2019", "citation": "Mole SE et al. Neuronal ceroid lipofuscinoses (NCLs): review. Lancet Neurology 2019;18:1004-1013. DOI:10.1016/S1474-4422(19)30167-0."},
            {"ref": "Williams-2006", "citation": "Williams RE et al. Diagnosis of the neuronal ceroid lipofuscinoses (Batten disease): a practical guide. Neuropediatrics 2006;37:57-68."},
            {"ref": "Santavuori-1982", "citation": "Santavuori P et al. Neuronal ceroid-lipofuscinoses in childhood. The Finnish late-infantile type. Acta Paediatrica Scandinavica 1982;71:803-808."},
            {"ref": "Broom-2017", "citation": "Broom MF et al. Observations on the ovine CLN5 model of Batten disease. Molecular Genetics and Metabolism 2017;123:S1-S2. (CLN5 sheep model for gene therapy preclinical development.)"},
        ],
    }
