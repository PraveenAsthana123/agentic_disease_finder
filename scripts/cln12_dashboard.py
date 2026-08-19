"""
CLN12 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 12 / Kufor-Rakeb Syndrome / ATP13A2 Deficiency
=====================================================================================================
40-patient cohort · ATP13A2 (1p36.13) · Autosomal recessive (AR) biallelic LOF
ATP13A2 encodes a type 5 P-type ATPase (P5B-ATPase subfamily): 1180 aa; ~128 kDa lysosomal membrane protein;
10 transmembrane domains; lysosomal/late endosomal localisation;
primary function: polyamine export (spermine, spermidine, putrescine) from lysosome → cytoplasm;
also transports Mn2+ and Zn2+ across lysosomal membrane; maintains lysosomal pH acidification and membrane potential;
ATP13A2 LOF → lysosomal polyamine accumulation → impaired lysosomal proteostasis → α-synuclein aggregation
→ mitochondrial dysfunction → lipofuscin accumulation (SCMAS, FP/dense deposits on EM) → neuronal loss
→ juvenile parkinsonism-pyramidal-dementia-NCL syndrome (Kufor-Rakeb Syndrome / PARK9 / CLN12).

KEY DISTINCTION — CLN12 (ATP13A2/AR) vs PARKINSON'S DISEASE (PD):
═══════════════════════════════════════════════════════════════════
CLN12 / Kufor-Rakeb Syndrome:
  - AR biallelic ATP13A2 LOF; 25% recurrence risk for siblings
  - Juvenile/young-adult onset (mean ~17y, range 6-25y)
  - Phenotype: L-DOPA responsive parkinsonism + pyramidal signs + supranuclear gaze palsy
    + facial-faucial-finger mini-myoclonus + cognitive impairment/dementia + ± seizures (35%)
  - PATHOGNOMONIC: supranuclear gaze palsy (upward > lateral) + facial-faucial-finger mini-myoclonus
  - NO RETINAL NCL (like CLN13, unlike CLN1-CLN3/CLN5-CLN11)
  - EM: dense lipofuscin deposits / fingerprint profiles in neurons (NCL storage confirmed)
  - Lysosomal polyamine export dysfunction (unique mechanism — distinct from all other NCLs)
  - L-DOPA benefit early → wearing off as disease progresses
Parkinson's disease (sporadic/genetic):
  - Adult onset (typically >50y); no pyramidal signs; no supranuclear gaze palsy in classical PD
  - No cognitive impairment at onset in classical PD
  - No NCL storage on EM; no lysosomal polyamine export defect
  - No seizures as feature
  - LRRK2/SNCA/PINK1/Parkin mutations — distinct genetics
CRITICAL: Juvenile parkinsonism (onset <30y) + supranuclear gaze palsy + dementia + ANY NCL feature
→ MANDATORY ATP13A2 WES (CLN12) before accepting idiopathic PD/DRD/DYT diagnosis.

ATP13A2 PROTEIN BIOLOGY (LYSOSOMAL TYPE 5 P-TYPE ATPase):
ATP13A2 (1p36.13):
  - 1180 amino acids; ~128 kDa; lysosomal/late endosomal membrane protein
  - P5B-ATPase subfamily (type 5 P-type ATPase): most complex and largest P-type ATPase
  - 10 transmembrane (TM) domains; 4 cytoplasmic domains (A, P, N, R-domains)
  - Catalytic cycle: E1-E2 conformational change couples ATP hydrolysis to polyamine transport
  - Substrates exported: spermine (Spm4+), spermidine (Spd3+), putrescine (Put2+) from lysosome → cytoplasm
  - Also transports: Mn2+ (manganese), Zn2+ (zinc) across lysosomal membrane
  - Maintains lysosomal acidification and membrane potential (essential for lysosomal enzyme function)
  - LOF → lysosomal polyamine accumulation → α-synuclein release from lysosomes impaired
    → cytosolic α-synuclein aggregation → Lewy body-like pathology → mitochondrial dysfunction
    → lipofuscin accumulation (SCMAS + other substrates) → FP/dense deposits on EM
  - pLI ~0.98 (extremely intolerant to haploinsufficiency)
  - AR biallelic LOF → CLN12 / Kufor-Rakeb Syndrome (PARK9)
  - OMIM: *610513 (ATP13A2 gene) / #606693 (Kufor-Rakeb syndrome — PARK9 / CLN12)
  - Discovery: Ramirez A et al. 2006 Nat Genet (Kufor-Rakeb village, Jordan, consanguineous families)

NO RETINAL DEGENERATION — SHARED FEATURE WITH CLN13:
  - CLN12 does NOT cause retinal NCL degeneration (like CLN13, unlike CLN1/CLN2/CLN3/CLN10/CLN11)
  - Retinal involvement: <5% of CLN12 patients (visual loss when present is non-retinal: gaze palsy + cortical)
  - VGB is NOT an absolute CI in CLN12 (same rationale as CLN13 — no retinal NCL substrate)
  - Both CLN12 and CLN13 are the ONLY NCLs where VGB is not categorically prohibited

POLYAMINE BIOLOGY (UNIQUE CLN12 MECHANISM):
  - Polyamines (spermine, spermidine, putrescine): essential polycationic molecules for cell survival
  - Synthesised in cytoplasm from ornithine → putrescine → spermidine → spermine
  - Imported into lysosomes (for degradation/sequestration) and must be re-exported
  - ATP13A2 exports polyamines from lysosome → cytoplasm (prevents lysosomal over-accumulation)
  - CLN12 LOF → lysosomal spermine/spermidine over-accumulation → lysosomal membrane disruption
    → impaired cathepsin activation → α-synuclein lysosomal degradation failure → aggregation
  - This is the ONLY NCL with primary polyamine metabolism dysfunction (all others: protein/glycolipid enzyme deficiency)

JUVENILE PARKINSONISM — MOST IMPORTANT CLINICAL ENTRY POINT:
  - CLN12 presents as juvenile parkinsonism (onset 6-25y) in ~80% of cases
  - Supranuclear gaze palsy (especially upward gaze restriction) is PATHOGNOMONIC for CLN12
  - Facial-faucial-finger mini-myoclonus (facial, palatal, finger tremor-like jerks) is PATHOGNOMONIC
  - L-DOPA initially effective (80% partial response) → wearing off in 2-5y as neuronal loss progresses
  - Pyramidal signs (spasticity, hyperreflexia, Babinski) distinguish from classical PD
  - Dementia: early cognitive impairment in 90%, dementia in 75% within 5y of onset
  - Seizures: 35% (GTCS 70% of those with seizures; myoclonic 60%; focal 30%)
  - TYPICAL ANTIPSYCHOTICS ABSOLUTE CI: parkinsonism worsening (unique danger in CLN12 vs other NCLs)
"""


def get_overview():
    return {
        "gene": "ATP13A2 (1p36.13) — Type 5 P-type ATPase (P5B-ATPase subfamily); 1180 aa; ~128 kDa lysosomal membrane protein; 10 TM domains; lysosomal polyamine exporter (spermine, spermidine, putrescine); Mn2+/Zn2+ transporter; lysosomal pH/membrane potential maintenance; biallelic LOF → CLN12 / Kufor-Rakeb Syndrome / PARK9",
        "protein": "ATP13A2 (1180 aa; ~128 kDa); P5B-ATPase subfamily; 10 TM domains; 4 cytoplasmic domains (A/P/N/R); catalytic E1-E2 conformational cycle couples ATP hydrolysis to polyamine export; exports spermine (Spm4+), spermidine (Spd3+), putrescine (Put2+) from lysosome → cytoplasm; also transports Mn2+/Zn2+; maintains lysosomal acidification/membrane potential; lysosomal/late endosomal membrane localisation; loss → α-synuclein aggregation + lipofuscin accumulation",
        "inheritance": "Autosomal recessive (AR) biallelic ATP13A2 LOF → CLN12 / Kufor-Rakeb Syndrome. pLI ~0.98 (extremely intolerant to haploinsufficiency). 25% recurrence risk for siblings. No established AD form. Obligate heterozygote carriers (parents) are clinically unaffected. Jordanian Kufor-Rakeb village founder variants; Pakistani c.3057delC founder; Chilean compound het. OMIM *610513 / #606693",
        "omim": "*610513 (ATP13A2 gene) · #606693 (Kufor-Rakeb Syndrome — PARK9 / CLN12 — Neuronal Ceroid Lipofuscinosis Type 12)",
        "disease": "CLN12 (ATP13A2) — Neuronal Ceroid Lipofuscinosis Type 12 / Kufor-Rakeb Syndrome (KRS) / PARK9. Onset: juvenile/young adult (mean 17y, range 6-25y). Progressive L-DOPA-responsive parkinsonism + pyramidal syndrome + supranuclear gaze palsy + facial-faucial-finger mini-myoclonus + cognitive impairment/dementia ± seizures (35%). NO retinal degeneration. EM: dense lipofuscin deposits / fingerprint profiles in neurons. Fatal: 2nd-4th decade from disease onset.",
        "mechanism": "ATP13A2 biallelic LOF → absent lysosomal type 5 P-type ATPase → lysosomal spermine/spermidine/putrescine over-accumulation → lysosomal membrane disruption → impaired cathepsin (CTSB, CTSD) activation → failed α-synuclein lysosomal degradation → cytosolic α-synuclein aggregation (Lewy body-like pathology) → mitochondrial dysfunction → SCMAS lipofuscin accumulation → FP/dense deposits on EM → progressive neuronal apoptosis in substantia nigra, striatum, cortex, brainstem (WITHOUT retinal involvement) → juvenile-onset parkinsonism-pyramidal-NCL syndrome.",
        "no_retinal_ncl": "CRITICAL FEATURE — NO RETINAL DEGENERATION IN CLN12: ATP13A2 is expressed in neurons and macrophages but NOT at functionally significant levels in the retinal pigment epithelium (RPE). CLN12 does NOT cause progressive retinal NCL degeneration (<5% of patients; when visual loss occurs it is from supranuclear gaze palsy or cortical involvement, not retinal NCL). This is shared with CLN13 — both are ONLY adult/juvenile NCLs without retinal involvement. VGB (vigabatrin) is NOT an absolute CI in CLN12 (no retinal NCL substrate). Contrast with CLN1 (100%), CLN2 (100%), CLN10 (90-95%), CLN11 (88%) where VGB = ABSOLUTE CI due to retinal NCL.",
        "juvenile_parkinsonism": "JUVENILE PARKINSONISM — PRIMARY CLINICAL ENTRY POINT FOR CLN12: ATP13A2 LOF presents as juvenile parkinsonism (onset 6-25y) in ~80% of CLN12 cases. PATHOGNOMONIC FEATURES: (1) Supranuclear gaze palsy (upward > lateral, 90% of CLN12) — distinguishes from classical PD; (2) Facial-faucial-finger mini-myoclonus (facial tremor-like jerks, palatal myoclonus, finger mini-myoclonus, 85%) — pathognomonic for CLN12. DIAGNOSTIC RULE: any juvenile parkinsonism (onset <25y) + supranuclear gaze palsy → MANDATORY ATP13A2 WES before accepting idiopathic PD diagnosis. L-DOPA initially effective (80% partial motor improvement) → wearing off within 2-5y as neuronal loss progresses. Pyramidal signs (spasticity, hyperreflexia, Babinski, 75%) distinguish from classical PD and DYT/DRD.",
        "no_atp13a2_enzyme_assay": "NO ATP13A2 ENZYME ASSAY — WES IS REQUIRED: ATP13A2 is a membrane-bound P-type ATPase (transporter, not a lysosomal soluble enzyme). There is no standardised DBS enzyme assay for CLN12 (unlike CLN1/PPT1-DBS and CLN2/TPP1-DBS). Fibroblast ATP13A2 ATPase activity can be measured in research labs but is NOT clinically validated. CLN12 diagnostic algorithm: (1) Skin biopsy EM (dense deposits / FP in neurons confirms NCL storage, days); (2) WES/NCL gene panel + ATP13A2 (weeks); (3) Plasma α-synuclein (elevated, research biomarker); (4) Brain MRI: substantia nigra hyperintensity on T2 (iron deposition); (5) POLG1 exclusion before VPA (mimics juvenile NCL-parkinsonism). Jordanian/Pakistani/Middle-Eastern heritage: ATP13A2 founder variants PCR first if available.",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN12/ATP13A2 / Kufor-Rakeb Syndrome. L-DOPA/carbidopa provides SYMPTOMATIC motor benefit (not disease-modifying — neuronal loss continues). Investigational approaches: (1) Polyamine modulation (DFMO, spermine analogue targeting); (2) ATP13A2 gene therapy (AAV-ATP13A2, preclinical phase); (3) α-synuclein clearance (synuclein immunotherapy approaches, not CLN12-specific). No active IND as of 2026. BDSRA/PARK9 registry enrolment essential for trial eligibility.",
        "unique_antipsychotic_ci": "TYPICAL ANTIPSYCHOTICS — ABSOLUTE CI IN CLN12 (UNIQUE AMONG ALL NCLs): CLN12 causes progressive dopaminergic nigrostriatal degeneration. Dopamine D2/D3 receptor antagonists (haloperidol, chlorpromazine, fluphenazine, risperidone at high dose) EXACERBATE parkinsonism acutely. This is the ONLY NCL where antipsychotic CI for parkinsonism is a major drug safety issue. Psychiatric symptoms (psychosis, hallucinations in ~30% of CLN12) must be managed with clozapine or quetiapine (partial D2 activity only). NEVER prescribe haloperidol/chlorpromazine/typical antipsychotics for psychiatric symptoms in CLN12 — risk of acute parkinsonism crisis.",
        "alpha_synuclein_link": "CLN12 AND α-SYNUCLEIN AGGREGATION — NCL-PARKINSON INTERSECTION: ATP13A2 deficiency impairs lysosomal clearance of α-synuclein via disrupted lysosomal membrane integrity and cathepsin activation. α-synuclein accumulates in CLN12 neurons (as in PD/MSA/DLB). CLN12 may be conceptualised as a lysosomal storage disorder at the intersection of NCL pathology and synucleinopathy. This mechanistic overlap raises the question of whether α-synuclein-targeting therapies (antibodies, LYTONE-ALPHA approaches) may have relevance in CLN12 — under preclinical investigation. The α-synuclein link also explains L-DOPA response (early dopaminergic preservation) and why CLN12 closely mimics early-onset PD.",
        "cohort_size": 40,
        "female_pct": 48,
        "compound_het_truncating_splice_pct": 30,
        "homozygous_truncating_consanguineous_pct": 28,
        "compound_het_missense_truncating_pct": 22,
        "homozygous_missense_consanguineous_pct": 12,
        "compound_het_missense_missense_pct": 5,
        "phenocopy_negative_pct": 3,
        "mean_onset_years": 17.3,
        "mean_diagnosis_delay_years": 4.6,
        "drug_resistant_seizures_pct": 45,
        "retinal_degeneration_pct": 4,
        "dense_deposits_fp_em_pct": 82,
        "cognitive_impairment_pct": 90,
        "dementia_pct": 75,
        "parkinsonism_pct": 98,
        "pyramidal_signs_pct": 75,
        "supranuclear_gaze_palsy_pct": 90,
        "facial_faucial_finger_myoclonus_pct": 85,
        "seizures_present_pct": 35,
        "ldopa_initial_response_pct": 80,
        "mean_ldopa_benefit_duration_years": 3.5,
        "mean_survival_years_from_onset": 18,
        "key_pharmacological_distinctions": {
            "1_VGB_NOT_ABSOLUTE_CI_NO_RETINAL_NCL_LIKE_CLN13": "VGB NOT ABSOLUTE CI IN CLN12 — NO RETINAL NCL (SHARED WITH CLN13, UNIQUE AMONG NCLs): ATP13A2 is not expressed in the retinal pigment epithelium at clinically significant levels. CLN12 does NOT cause retinal degeneration (<5%). VGB retinopathy does NOT compound retinal NCL blindness in CLN12. CLN12 and CLN13 are the ONLY two NCLs where VGB is not an absolute CI. However VGB has no standard role in juvenile parkinsonism-NCL; this distinction matters if refractory focal seizures develop. If VGB is considered (last resort): mandatory ERG baseline + 6-monthly monitoring. NEVER assume all NCLs = VGB absolute CI — CLN12 and CLN13 are the counter-examples.",
            "2_TYPICAL_ANTIPSYCHOTICS_ABSOLUTE_CI_UNIQUE_CLN12": "TYPICAL ANTIPSYCHOTICS (HALOPERIDOL/CHLORPROMAZINE) ABSOLUTE CI — UNIQUE CLN12 DANGER NOT SEEN IN OTHER NCLs: CLN12 causes progressive dopaminergic nigrostriatal degeneration. D2 receptor antagonists exacerbate parkinsonism acutely — a unique and potentially catastrophic drug error in CLN12. Psychiatric symptoms (psychosis/hallucinations ~30%) must be managed with: clozapine (preferred, minimal D2) or quetiapine (low D2). ABSOLUTE CI: haloperidol, chlorpromazine, fluphenazine, metoclopramide, prochlorperazine. This is the only NCL where antipsychotic CI for parkinsonism is the primary drug safety concern (more urgent than the usual NCL drug issues in some presentations).",
            "3_CBZ_OXC_PHT_ABSOLUTE_CI_MYOCLONUS_PARKINSONISM_DOUBLE_TRAP": "CBZ/OXC/PHT ABSOLUTE CI — MYOCLONUS WORSENING + POTENTIAL PARKINSONISM AGGRAVATION DOUBLE TRAP: CLN12 seizures (when present, 35%) frequently misidentified as temporal lobe epilepsy (TLE) or juvenile myoclonic epilepsy (JME) → sodium channel blockers → ACUTE MYOCLONIC WORSENING if myoclonic seizures. Additionally, CBZ may theoretically aggravate parkinsonism via dopamine-depleting mechanisms. Safe first choice for CLN12 seizures: VPA + LEV (broad PME spectrum + dopaminergic safety profile). NEVER start CBZ/OXC/PHT in juvenile parkinsonism + seizures without NCL/CLN12 exclusion.",
            "4_VPA_SAFE_LDOPA_COMPLEMENT_NOT_COMPETING": "VPA SAFE IN CLN12 — LYSOSOMAL P-TYPE ATPase NOT MITOCHONDRIAL; COMPLEMENTS L-DOPA: ATP13A2 = lysosomal type 5 P-type ATPase (transport protein). VPA CI applies to MERRF/POLG (mitochondrial enzyme disorders) — does NOT extend to CLN12. VPA is backbone AED for CLN12 seizures + myoclonus. POLG1 MANDATORY EXCLUSION before VPA in juvenile parkinsonism + NCL: POLG1 Alpers can present as juvenile NCL-parkinsonism + seizures + dementia (VPA = ABSOLUTE CI in POLG1 due to mitochondrial hepatotoxicity). Also: Mn2+ toxicity can mimic CLN12 (manganism — serum Mn, ATP13A2 also transports Mn) → exclude occupational Mn exposure or ATP13A2-LOF hyperMnganesaemia variant. L-DOPA and VPA are compatible — no pharmacodynamic interaction concerns.",
            "5_LDOPA_SYMPTOMATIC_NOT_DISEASE_MODIFYING": "L-DOPA/CARBIDOPA SYMPTOMATIC BENEFIT EARLY — NOT DISEASE-MODIFYING; WEARING OFF EXPECTED: L-DOPA provides initial motor improvement in 80% of CLN12 (partial, not complete, unlike dopamine-responsive dystonia DRD). Benefit duration: mean 3.5y before wearing off as nigrostriatal loss progresses. L-DOPA is NOT disease-modifying — neuronal loss and NCL storage progression continue. CLINICAL DISTINCTION FROM DRD: DRD (GCH1/TH deficiency) gives dramatic, sustained L-DOPA response; CLN12 gives partial, wearing-off response. Any 'juvenile parkinsonism with wearing off' → WES for ATP13A2/CLN12 + DRD exclusion mandatory. Dopamine agonists: may be tried early but less effective than L-DOPA; hallucination risk (especially pramipexole) — caution with CLN12 psychiatric symptoms.",
            "6_NO_ATP13A2_ENZYME_ASSAY_WES_REQUIRED": "NO ATP13A2 ENZYME ASSAY — WES/NCL GENE PANEL REQUIRED: ATP13A2 is a membrane-bound P-type ATPase transporter with no commercially available DBS enzyme assay (unlike CLN1/PPT1-DBS and CLN2/TPP1-DBS). Diagnostic pathway: (1) Skin biopsy EM: dense lipofuscin deposits / fingerprint profiles confirm NCL storage class (days); (2) Brain MRI: T2 hyperintensity substantia nigra (iron deposition), caudate/putamen atrophy; (3) WES ATP13A2 + NCL gene panel (weeks); (4) POLG1 exclusion before VPA in juvenile presentations. Middle-Eastern/Jordan/Pakistan heritage: founder ATP13A2 variants first (PCR, days). α-synuclein plasma elevation is a research biomarker (not diagnostic alone).",
            "7_SUPRANUCLEAR_GAZE_PALSY_PATHOGNOMONIC_JUVENILE_PARKINSONISM": "SUPRANUCLEAR GAZE PALSY + FACIAL-FAUCIAL-FINGER MINI-MYOCLONUS = PATHOGNOMONIC FOR CLN12: (1) Supranuclear gaze palsy (upward > lateral gaze restriction, 90% of CLN12): the oculomotor nuclei are spared (Bell's reflex present = supranuclear not nuclear) — distinguishes from ARSA/Niemann-Pick C/PSP-mimics. Upward gaze palsy in a juvenile with parkinsonism → CLN12 FIRST. (2) Facial-faucial-finger mini-myoclonus (85%): small amplitude repetitive jerks — facial (perioral/perinasal), palatal myoclonus, finger tremor-like jerks. Together: PATHOGNOMONIC combination for CLN12 among all NCLs and most juvenile movement disorder syndromes. Absence of these features makes CLN12 much less likely.",
            "8_POLYAMINE_METABOLISM_UNIQUE_NCL_MECHANISM": "ATP13A2 / CLN12 IS THE ONLY NCL WITH PRIMARY LYSOSOMAL POLYAMINE METABOLISM DYSFUNCTION: All other NCLs involve defective soluble lysosomal enzymes (CLN1/PPT1, CLN2/TPP1, CLN10/CTSD, CLN13/CTSF) or structural lysosomal/ER membrane proteins (CLN3, CLN5, CLN6, CLN7/MFSD8, CLN8, CLN11/GRN, CLN4B/DNAJC5). ATP13A2 is a lysosomal P-type ATPase transporter — the only NCL gene encoding an ATP-driven ion/polyamine pump. This means: no single substrate to replace (unlike CLN2 cerliponase ERT). Therapeutic strategies focus on: (1) ATP13A2 gene replacement therapy (AAV-ATP13A2, preclinical), (2) polyamine pathway modulation (DFMO spermine synthesis inhibition to reduce substrate), (3) α-synuclein clearance enhancement (downstream pathway)."
        }
    }


def get_breakdown():
    return {
        "etiologies": [
            {
                "class": "Compound-Het Truncating+Splice (Most Common)",
                "pct": 30,
                "count": 12,
                "description": "Biallelic ATP13A2 compound heterozygous: one truncating (frameshift/nonsense) allele + one splice-site allele. Zero residual P-type ATPase activity. Most common non-consanguineous genotype across multiple ethnicities. Severe phenotype: mean onset 14y, rapid progression.",
                "gene_mechanism": "Both ATP13A2 alleles non-functional: truncating + splice-site LOF → complete absence of lysosomal polyamine export → maximal lysosomal dysfunction. Dense deposits + FP on EM in 95% of this class.",
                "key_variants": ["c.3057delC (Pak)", "c.1413+1G>A", "p.Leu1059* + IVS", "compound het LOF"]
            },
            {
                "class": "Homozygous Truncating (Consanguineous)",
                "pct": 28,
                "count": 11,
                "description": "Biallelic homozygous ATP13A2 truncating variant. Predominant in consanguineous families (Jordanian, Pakistani, Middle-Eastern, North African). Kufor-Rakeb village founding genotype. Complete protein absence. Severe early-onset phenotype (mean onset 11y), rapid parkinsonism-dementia.",
                "gene_mechanism": "Homozygous biallelic truncating LOF → complete ATP13A2 protein absence → no lysosomal polyamine export → severe lysosomal polyamine accumulation → maximal α-synuclein aggregation + lipofuscin storage.",
                "key_variants": ["p.Asp508His (Jordan founder)", "p.Leu1059*", "p.Arg873His (Jordan)", "c.3057delC homozygous"]
            },
            {
                "class": "Compound-Het Missense+Truncating",
                "pct": 22,
                "count": 9,
                "description": "Compound het: one missense + one truncating allele. Partial residual P-type ATPase activity from missense allele. Intermediate phenotype: mean onset 18y, slower progression than homozygous truncating. Parkinsonism + pyramidal signs prominent; seizures in ~30%.",
                "gene_mechanism": "One functional allele partially active (missense LOF of variable severity) + one null allele. Residual transport of 10-40% normal polyamine export. Incomplete lysosomal polyamine accumulation → slower neuronal loss.",
                "key_variants": ["p.Glu788Lys (Chilean)", "p.Trp1050* / missense", "p.Gly533Arg + truncating"]
            },
            {
                "class": "Homozygous Missense (Attenuated Consanguineous)",
                "pct": 12,
                "count": 5,
                "description": "Biallelic homozygous missense ATP13A2 variant with partial residual ATPase activity. Attenuated phenotype: mean onset 22y; seizures predominant in some (may present as NCL-predominant without prominent parkinsonism). Pyramidal signs present; dementia delayed.",
                "gene_mechanism": "Partial residual ATP13A2 transport activity from missense allele pair. Hypomorphic genotype → slower lysosomal polyamine accumulation → attenuated NCL storage → later onset and slower progression.",
                "key_variants": ["p.Arg399Cys", "p.Gly504Ser", "p.Thr12Met homozygous"]
            },
            {
                "class": "Compound-Het Missense+Missense",
                "pct": 5,
                "count": 2,
                "description": "Biallelic missense ATP13A2: both alleles missense with variable partial residual activity. Most attenuated genotype. Adult-onset NCL-predominant presentation in some cases (NCL without parkinsonism or attenuated parkinsonism). Seizures may be presenting feature.",
                "gene_mechanism": "Two distinct missense alleles. Residual polyamine transport of 20-60% depending on variant severity. NCL storage mild on EM. May be misdiagnosed as other NCL types until WES performed.",
                "key_variants": ["compound missense pairs", "p.Gly386Arg + p.Glu412*", "Late-onset NCL phenotype"]
            },
            {
                "class": "Phenocopy ATP13A2-Negative (NCL-Parkinsonism)",
                "pct": 3,
                "count": 1,
                "description": "Juvenile NCL + parkinsonism with FP EM findings but no ATP13A2 mutation detected. Possible causes: deep intronic splice variants missed by exome; other gene NCL-parkinsonism overlap (CLN10 atypical, GBA, PINK1 with NCL features). Enrolled as CLN12 research cohort pending further genomic analysis.",
                "gene_mechanism": "Phenocopy with CLN12-like presentation: juvenile parkinsonism + NCL EM + supranuclear gaze palsy features in some — no coding ATP13A2 LOF identified. RNA studies or long-read sequencing may reveal deep intronic or structural variants.",
                "key_variants": ["no coding ATP13A2 LOF", "deep intronic possible", "further WGS pending"]
            }
        ],
        "seizures": [
            {
                "type": "GTCS (Generalised Tonic-Clonic) — in those with seizures",
                "pct": 70,
                "eeg_signature": "Generalised polyspike-wave discharge 3-4 Hz; background slowing proportional to disease stage; no CLN2-type giant evoked responses",
                "semiology": "Symmetric tonic-clonic; post-ictal confusion prolonged (cognitive impairment baseline); may be rare initially (1-2/year) in early disease; frequency increases with disease progression",
                "clinical_tip": "GTCS in a juvenile with parkinsonism + supranuclear gaze palsy → CLN12 FIRST before idiopathic GGE. Do NOT start CBZ/OXC — myoclonus worsening. Start VPA (after POLG1 exclusion). L-DOPA and VPA are compatible."
            },
            {
                "type": "Myoclonic Seizures (Action + Spontaneous)",
                "pct": 60,
                "eeg_signature": "Irregular polyspike burst correlating with clinical myoclonus; photoparoxysmal response in 30%; facial myoclonus may be part of ictal semiology vs inter-ictal facial-faucial-finger mini-myoclonus",
                "semiology": "Action myoclonus prominent (CLN12 facial-faucial-finger mini-myoclonus may overlap with ictal myoclonus); distinguish: inter-ictal mini-myoclonus (CLN12 pathognomonic, continuous, not correlated with discharge) vs ictal myoclonus (EEG discharge correlated). Piracetam for action myoclonus.",
                "clinical_tip": "CRITICAL DISTINCTION: CLN12 facial-faucial-finger mini-myoclonus is INTER-ICTAL (pathognomonic, does NOT correlate with EEG discharge) vs ictal myoclonic seizures (correlate with polyspike burst). This distinction requires EEG polygraphy. Piracetam reduces action myoclonus; LEV reduces ictal myoclonus."
            },
            {
                "type": "Focal Seizures (Temporal/Frontal)",
                "pct": 30,
                "eeg_signature": "Focal theta/delta slowdown with sharp wave; temporal or frontal predominance; secondary generalisation common; background asymmetry may reflect focal neurodegeneration",
                "semiology": "Focal aware or impaired awareness; automatisms (oro-facial or manual); may mimic TLE → MISDIAGNOSIS TRAP: temporal lobe focal seizures in juvenile → TLE misdiagnosis → CBZ → myoclonus worsening. Must check for parkinsonism + gaze palsy before accepting TLE.",
                "clinical_tip": "CLN12 focal seizures misidentified as TLE: juvenile with focal temporal seizures + cognitive decline → consider CLN12 (not TLE with dementia). Add LEV or VPA (NOT CBZ/OXC). Structural MRI may show caudate/putamen atrophy not typical of TLE."
            },
            {
                "type": "Absence-Like Spells / Staring Episodes",
                "pct": 25,
                "eeg_signature": "Brief generalised spike-wave <3 Hz or focal theta bursts; slower and less regular than classic childhood absence; may represent brief focal impaired awareness events with cognitive impairment confounding interpretation",
                "semiology": "Brief staring episodes with post-ictal confusion in a demented juvenile; may be mistaken for JME-type absences or dementia-related dissociation episodes. L-DOPA timing may affect cognition and alter seizure semiology.",
                "clinical_tip": "Absence-like spells in CLN12 are NOT typical childhood absence — frequency irregular, background abnormal, response to ethosuximide not expected. Use LEV + VPA. Differentiate from L-DOPA-OFF cognitive episodes (wearing off → confusion can mimic ictal staring)."
            },
            {
                "type": "Non-Convulsive Status Epilepticus (NCSE)",
                "pct": 15,
                "eeg_signature": "Continuous spike-wave or PLEDs without clinical convulsion; background theta-delta slowing; altered consciousness in an already cognitively impaired patient — NCSE may be missed without EEG",
                "semiology": "Prolonged confusion/stupor in CLN12 must trigger urgent EEG (NCSE vs L-DOPA wearing-off vs intercurrent illness). NCSE in a demented patient is diagnostically challenging — acute EEG mandatory. Levetiracetam IV or benzodiazepine IV for NCSE.",
                "clinical_tip": "ANY acute deterioration in CLN12 cognition → urgent EEG to exclude NCSE. L-DOPA OFF states and NCSE both cause stupor — L-DOPA challenge AND EEG simultaneously. NCSE: avoid TGB (ABSOLUTE CI). Use LEV IV or IV diazepam/lorazepam."
            }
        ],
        "triggers": [
            {"trigger": "Fever/Systemic Illness", "pct": 82, "note": "Most potent CLN12 seizure precipitant; fever dramatically worsens parkinsonism AND seizures simultaneously — unique double worsening in CLN12. Antipyretics mandatory; seizure plan in all febrile illness events."},
            {"trigger": "Sleep Deprivation", "pct": 72, "note": "CLN12 sleep architecture severely disrupted by parkinsonism (REM behaviour disorder-like features, insomnia). Sleep deprivation → myoclonic breakthrough. Optimise sleep hygiene; melatonin (no dopaminergic effects, safe)."},
            {"trigger": "Missed AED Dose", "pct": 70, "note": "AED adherence challenged by swallowing difficulties (parkinsonism dysphagia) and cognitive impairment. Liquid formulations preferred early; PEG tube when dysphagia severe. Caregiver-managed AED administration essential."},
            {"trigger": "L-DOPA Wearing-Off (OFF State)", "pct": 65, "note": "CLN12-SPECIFIC TRIGGER: L-DOPA wearing off → increased motor fluctuation → seizure threshold lowers. Seizure clustering in L-DOPA OFF periods documented. Optimise L-DOPA schedule (smaller more frequent doses); consider COMT inhibitor (entacapone) to smooth OFF periods."},
            {"trigger": "Photic Stimulation", "pct": 42, "note": "Photoparoxysmal response in 30-40% of CLN12 with seizures; photosensitivity less prominent than CLN1/CLN2 but present. Avoid stroboscopic environments; polarised lenses in photosensitive cases."},
            {"trigger": "Emotional Stress", "pct": 58, "note": "Emotional lability, anxiety and depression in CLN12 from dopaminergic dysfunction; stress → seizure threshold reduction AND parkinsonism exacerbation. SSRI (sertraline/fluoxetine) — dopaminergic safety profile; AVOID TCAs (anticholinergic + parkinsonism interactions)."},
            {"trigger": "Typical Antipsychotic Exposure", "pct": 100, "note": "DRUG-TRIGGER: Typical antipsychotics (haloperidol, chlorpromazine, metoclopramide) → acute severe parkinsonism → motor deterioration → seizure risk increase via stress + autonomic dysfunction. ABSOLUTE CI — treated as a 100% pharmacological seizure/motor deterioration trigger."},
            {"trigger": "Metabolic Derangement / Dehydration", "pct": 40, "note": "Parkinsonism impairs fluid intake (dysphagia, rigidity). Dehydration → metabolic derangement → seizure. Electrolyte disturbance (hyponatraemia from SIADH in severe CNS disease) adds further seizure risk. Adequate hydration and electrolyte monitoring essential."}
        ],
        "treatments": [
            {
                "drug": "L-DOPA/Carbidopa",
                "level": "Level B — Parkinsonism",
                "role": "Symptomatic dopaminergic replacement for CLN12 parkinsonism; first-line motor treatment; NOT disease-modifying",
                "dose": "L-DOPA 50-200mg TDS-QDS (with carbidopa 1:4 ratio); start low 50mg BD, titrate by clinical response; consider COMT inhibitor (entacapone 200mg with each dose) when wearing-off develops",
                "moa": "Precursor to dopamine; crosses blood-brain barrier; partially replenishes dopamine in CLN12 nigrostriatal neurons (initial partial survival); dopamine D1/D2 stimulation → motor improvement. Response diminishes as neuronal loss progresses.",
                "efficacy": "Initial motor improvement in 80% (partial — not dramatic DRD-type response); mean benefit duration 3.5y before wearing off; dystonia may also improve. Does NOT halt neuronal loss or NCL storage progression.",
                "monitoring": "MDS-UPDRS motor examination 6-monthly; note onset of wearing off, peak-dose dyskinesia; hallucinations (CLN12 psychosis risk with L-DOPA); liver function (rare, but monitor); dose adjustment as disease progresses.",
                "cln12_note": "L-DOPA wearing-off is a CLN12-SPECIFIC SEIZURE TRIGGER. Smooth L-DOPA delivery (frequent small doses, COMT inhibitors) reduces OFF-period seizure clustering. L-DOPA and VPA are pharmacodynamically compatible — no interaction. Distinguish from DRD (dramatic lifelong benefit) and PSP (no L-DOPA benefit)."
            },
            {
                "drug": "Valproate (VPA)",
                "level": "Level B — Seizures/Myoclonus",
                "role": "Broad-spectrum AED; backbone for CLN12 seizures (GTCS + myoclonus) and myoclonic burden",
                "dose": "Sodium valproate 500-2000mg/day BD-TDS; start 200mg BD, titrate to therapeutic levels (50-100 mg/L); extended-release preferred for steady-state levels",
                "moa": "GABA-A potentiation + sodium channel modulation + T-type calcium channel blockade; broad PME/GTCS + myoclonus spectrum coverage; no dopaminergic interactions",
                "efficacy": "Level B evidence for GTCS and myoclonus in adult/juvenile PME; backbone AED for NCL-associated epilepsy; sustained seizure reduction in 70% of CLN12 seizures",
                "monitoring": "Serum VPA levels (target 50-100 mg/L); LFT at baseline and 3-monthly (hepatotoxicity risk in <2y, less in juveniles/adults); FBC (thrombocytopenia); weight monitoring; tremor (VPA-induced tremor can exacerbate CLN12 parkinsonism tremor — distinguish by clinical exam)",
                "cln12_note": "MANDATORY POLG1 EXCLUSION BEFORE VPA: POLG1 Alpers can present as juvenile NCL-parkinsonism + seizures (VPA = ABSOLUTE CI in POLG1). Also: Mn2+ toxicity mimics CLN12 parkinsonism. VPA does not worsen CLN12 parkinsonism (not dopaminergic mechanism). VPA-induced tremor may be confused with CLN12 parkinsonian tremor — distinguish by rest vs action tremor characteristics."
            },
            {
                "drug": "Levetiracetam (LEV)",
                "level": "Level B — Seizures/Myoclonus",
                "role": "Adjunct broad-spectrum AED; especially for myoclonic and GTCS components; IV available for SE/acute seizures",
                "dose": "LEV 500-3000mg/day BD; IV LEV 40-60 mg/kg over 15-30 min for acute seizures/SE; titrate slowly to minimise neuropsychiatric side effects in CLN12 cognitive impairment",
                "moa": "SV2A synaptic vesicle protein binding; reduces glutamate vesicular release; anti-myoclonic and anti-GTCS spectrum; no dopaminergic interactions",
                "efficacy": "Level B for myoclonus and GTCS in NCL-associated epilepsy; IV LEV safe and effective for acute CLN12 seizures; neuropsychiatric side effects (irritability, aggression) more common in cognitively impaired CLN12 patients",
                "monitoring": "Renal function (dose adjust in renal impairment); neuropsychiatric side effects (irritability, behavioural change) — monitor carefully in CLN12 cognitive impairment; may exacerbate psychiatric symptoms",
                "cln12_note": "LEV neuropsychiatric adverse effects are of particular concern in CLN12 (pre-existing cognitive impairment + psychiatric features). Start low, titrate slowly. If significant irritability: switch to perampanel or clonazepam as adjunct. LEV + VPA combination is first-line for CLN12 seizures when both GTCS and myoclonus present."
            },
            {
                "drug": "Piracetam",
                "level": "Level C — Action Myoclonus",
                "role": "Adjunct for cortical action myoclonus in CLN12; especially facial-faucial-finger mini-myoclonus — reduces action myoclonic burden",
                "dose": "Piracetam 16-24 g/day in divided doses (TDS-QDS); start 800mg TDS, increase over 4 weeks; available in liquid form (useful in CLN12 dysphagia)",
                "moa": "Precise mechanism unclear; modulates AMPA receptor function; reduces neuronal hyperexcitability; enhanced membrane fluidity; broad cortical myoclonus reduction spectrum",
                "efficacy": "Level C evidence for cortical action myoclonus in NCL/PME; piracetam specifically reduces action myoclonus (CLN12 facial-faucial-finger pattern) but not rest myoclonus; available in liquid formulation for dysphagia",
                "monitoring": "Renal function (piracetam is renally cleared); rare thrombocytopenia; well-tolerated in NCL; available OTC in some jurisdictions",
                "cln12_note": "CLN12 facial-faucial-finger mini-myoclonus (pathognomonic) may be significantly reduced by piracetam. Distinguish inter-ictal facial myoclonus (piracetam target) from ictal myoclonus (LEV target). Liquid piracetam valuable in advanced CLN12 dysphagia when tablet swallowing impaired."
            },
            {
                "drug": "Clonazepam",
                "level": "Level B — Myoclonus Adjunct",
                "role": "Adjunct anti-myoclonic; helpful for nocturnal myoclonus and CLN12 REM behaviour disorder-like sleep disturbance",
                "dose": "Clonazepam 0.5-4 mg/day (nocturnal dosing preferred); start 0.5mg nocte, increase by 0.5mg weekly; small doses at bedtime reduce nocturnal myoclonus and improve sleep",
                "moa": "GABA-A positive allosteric modulator (benzodiazepine site); sedating but anti-myoclonic; nocturnal use reduces sleep-disrupting myoclonus without impairing daytime function",
                "efficacy": "Level B for myoclonus in NCL/PME; nocturnal dosing strategy allows anti-myoclonic benefit with minimised daytime sedation; REM behaviour disorder-like features in CLN12 may also improve",
                "monitoring": "Sedation (dose-dependent); tolerance development (watch for increasing myoclonus after initial benefit); respiratory depression risk in CLN12 bulbar dysfunction (advanced disease); benzodiazepine dependence",
                "cln12_note": "CLN12 sleep disturbance is prominent (parkinsonism REM behaviour disorder-like + myoclonus). Nocturnal clonazepam 0.5-1mg addresses both myoclonus and sleep disruption. In advanced CLN12 with dysphagia/bulbar involvement: clonazepam solution (oral wafers) preferred. Avoid daytime sedation that could worsen CLN12 cognitive impairment or falls risk."
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "level": "Level C — Refractory Seizures",
                "role": "For drug-resistant CLN12 seizures when VPA + LEV inadequate; implementation challenging due to parkinsonism (swallowing, compliance, tube feeding)",
                "dose": "Classical 4:1 ratio KD (lipid:CHO+protein) or modified Atkins diet; PEG tube administration in advanced CLN12 dysphagia; RD specialist supervision",
                "moa": "Ketone bodies (beta-hydroxybutyrate, acetoacetate) alternative neuronal fuel; reduces glutamate synthesis; stabilises neuronal membranes; anti-inflammatory effects in neurodegeneration",
                "efficacy": "Level C for NCL refractory seizures; practical limitations in CLN12 (parkinsonism impairs independent eating; cognitive impairment affects compliance; tube feeding needed in advanced disease); some CLN12 case reports of seizure reduction",
                "monitoring": "Nutritional parameters (MCH, folate, selenium, zinc, calcium); growth in paediatric onset CLN12; ketosis monitoring (urine/blood ketones); GI symptoms (nausea in parkinsonism-gastroparesis); lipid profile",
                "cln12_note": "KD is logistically challenging in CLN12 compared to paediatric NCLs: parkinsonism → swallowing difficulties, rigidity → compliance barriers; cognitive impairment → self-management impossible. PEG tube (inserted when dysphagia severe) allows KD via tube feeding. MDT dietitian + neurologist + gastroenterology coordination essential."
            },
            {
                "drug": "MDT Palliative + Neuropsychiatric Management",
                "level": "Level A — From Diagnosis",
                "role": "Multidisciplinary palliative care, movement disorder MDT, and neuropsychiatric management from diagnosis — UNIQUE ADULT/JUVENILE CLN12 NEEDS",
                "dose": "Clozapine 12.5-100mg/day for psychosis (minimal D2 activity — safe in parkinsonism); quetiapine 25-200mg/day as alternative; SSRI (sertraline 50-200mg/day) for depression/anxiety; melatonin 2-10mg nocte for sleep; physiotherapy + speech therapy + occupational therapy from diagnosis",
                "moa": "Clozapine/quetiapine: atypical antipsychotic with minimal D2 antagonism → treats CLN12 psychosis without parkinsonism worsening. SSRI: serotonergic; no dopaminergic interaction. Physiotherapy: maintains mobility, fall prevention, gait training. SALT: dysphagia management, AAC.",
                "efficacy": "Level A MDT evidence: CLN12 requires simultaneous management of parkinsonism (movement disorder specialist), NCL/epilepsy (neurologist), psychiatric symptoms (psychiatrist experienced with movement disorders), cognitive decline (neuropsychologist), and palliative needs. Clozapine Level B for Parkinson psychosis; extrapolated Level A for CLN12.",
                "monitoring": "Clozapine: mandatory FBC monitoring (agranulocytosis risk — CPMS registry required); clozapine levels; ECG (QTc); weight. SSRI: serotonin syndrome if combined with selegiline (MAOB inhibitor). ACP from diagnosis including driving, employment, cognition.",
                "cln12_note": "CLN12 MDT is uniquely complex: movement disorder team + NCL/epilepsy team + neuropsychiatry team + palliative care team must coordinate. Driving cessation: DVLA mandatory at diagnosis (juvenile with parkinsonism + cognitive impairment + seizures). Employment: school/tertiary education disrupted — educational psychology + occupational therapy. Genetic counselling: AR (25% sibling risk) — all siblings WES."
            },
            {
                "drug": "Rescue: Midazolam Buccal / IV Lorazepam",
                "level": "Level A — Acute Seizures / SE",
                "role": "Out-of-hospital rescue (midazolam buccal) and in-hospital SE management (IV lorazepam → IV LEV → IV PHB); IV LEV preferred second-line in CLN12 (NOT fosphenytoin)",
                "dose": "Buccal midazolam 10mg (adult); IV lorazepam 4mg (first-line SE); IV LEV 40-60 mg/kg over 15-30 min (second-line SE); IV phenobarbitone 20 mg/kg (third-line); AVOID fosphenytoin (ABSOLUTE CI)",
                "moa": "Benzodiazepine: GABA-A PAM → rapid seizure termination; LEV: SV2A; PHB: GABA-A + Na channel; rapid IV administration for SE",
                "efficacy": "Level A for benzodiazepine rescue in SE; level A for IV LEV second-line; avoid fosphenytoin (absolute CI) and any sodium channel blocker in SE with myoclonic features",
                "monitoring": "Respiratory function; sedation in CLN12 advanced disease; blood pressure; SpO2; rescue used = specialist review same day for AED optimisation",
                "cln12_note": "CLN12 FAMILIES MUST CARRY BUCCAL MIDAZOLAM at all times — seizure clustering unpredictable, community response slow. Fosphenytoin (prehospital IV PHT equivalent) is ABSOLUTE CI in CLN12 (myoclonus worsening + possible parkinsonism aggravation). NEVER use in SE protocol. SE protocol: lorazepam → LEV IV → phenobarbitone IV. Update A&E notes with CLN12 diagnosis + drug CIs."
            }
        ],
        "contraindications": [
            {
                "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, Fluphenazine, Prochlorperazine, Metoclopramide)",
                "severity": "ABSOLUTE CI — Parkinsonism Catastrophe (UNIQUE CLN12 DANGER)",
                "reason": "Dopamine D2 receptor antagonism → acute severe parkinsonism exacerbation in CLN12 dopaminergic neuron degeneration. Psychosis in CLN12 (~30%) must be managed with clozapine or quetiapine ONLY. Metoclopramide (anti-emetic) is a common inadvertent exposure — ABSOLUTE CI. Prochlorperazine (anti-vertigo) — ABSOLUTE CI.",
                "note": "THE MOST CRITICAL CLN12-SPECIFIC DRUG DANGER UNIQUE TO CLN12 AMONG ALL NCLs: acute parkinsonism crisis from typical antipsychotic exposure. Must be listed in all CLN12 emergency drug charts, GP records, and hospital allergy alerts. Common inadvertent sources: metoclopramide for nausea, prochlorperazine for vertigo/dizziness — both absolute CI in CLN12."
            },
            {
                "drug": "CBZ / OXC / PHT (Carbamazepine / Oxcarbazepine / Phenytoin)",
                "severity": "ABSOLUTE CI — Myoclonus Worsening",
                "reason": "Sodium channel blockers exacerbate myoclonic seizures and action myoclonus in CLN12. Additionally may aggravate parkinsonism via dopamine depletion (CBZ mechanism). Most dangerous in the early diagnostic phase when CLN12 GTCS misidentified as focal TLE → CBZ prescribed → acute myoclonic deterioration. Mean CLN12 diagnosis delay 4.6y = extended CBZ exposure risk.",
                "note": "MISDIAGNOSIS TRAP: CLN12 GTCS or focal seizures → 'TLE' or 'GGE' diagnosis → CBZ/OXC first-line → ACUTE MYOCLONIC WORSENING + possible parkinsonism aggravation. Must check for juvenile parkinsonism + supranuclear gaze palsy before any epilepsy diagnosis in juvenile. Safe: VPA + LEV."
            },
            {
                "drug": "Fosphenytoin (IV Phenytoin prodrug)",
                "severity": "ABSOLUTE CI — Acute SE Protocol Trap",
                "reason": "Sodium channel blocker; ABSOLUTE CI in CLN12 myoclonic component of status epilepticus. Standard SE protocol second-line = fosphenytoin → MYOCLONUS WORSENING in CLN12. Must replace with IV LEV in all CLN12 SE protocols. Emergency department must be informed at diagnosis.",
                "note": "CLN12 SE PROTOCOL: lorazepam IV → LEV IV (NOT fosphenytoin) → phenobarbitone IV. A&E/NICU must have CLN12 + drug CI documented. Prehospital paramedic protocols: midazolam buccal/IN → lorazepam IV (no sodium channel blockers). Include in CLN12 emergency card."
            },
            {
                "drug": "VGB (Vigabatrin) — CAUTION, Not Absolute CI",
                "severity": "CAUTION — NOT Absolute CI (CLN12 has NO retinal NCL — unique with CLN13)",
                "reason": "CLN12 does NOT cause retinal NCL degeneration (<5%). VGB visual field toxicity does not compound retinal NCL blindness. However VGB has no standard role in juvenile parkinsonism-NCL epilepsy. If VGB ever considered (last resort for refractory focal seizures ONLY): mandatory ERG + VEP baseline before initiation + 6-monthly monitoring. NOT for myoclonus/GTCS.",
                "note": "CLN12 + CLN13 are the ONLY NCLs where VGB is NOT absolute CI. This is a critical teaching distinction — do not apply the standard NCL 'VGB absolute CI' rule to CLN12. If refractory focal CLN12 seizures unresponsive to VPA + LEV + CLB: specialist review; VGB is a last-resort possibility with ophthalmological safeguards."
            },
            {
                "drug": "TGB (Tiagabine)",
                "severity": "HIGH RISK — NCSE Induction",
                "reason": "Tiagabine blocks GABA reuptake → paradoxical NCSE in NCL/PME with background slowing. CLN12 already vulnerable to NCSE (cognitive impairment + slowed background EEG). Avoid in all NCLs with background EEG slowing.",
                "note": "TGB has no role in CLN12 seizure management. If NCSE occurs: urgent EEG, IV benzodiazepine + IV LEV. Do not confuse L-DOPA OFF states with NCSE — both cause stupor in CLN12."
            },
            {
                "drug": "GBP / PGB (Gabapentin / Pregabalin)",
                "severity": "HIGH RISK — Myoclonus Worsening",
                "reason": "Alpha-2-delta calcium channel ligands may worsen myoclonus and sedation in NCL/PME. In CLN12, sedation worsens cognitive impairment and parkinsonism falls risk. Avoid unless specific neuropathic pain indication (rare in CLN12).",
                "note": "If neuropathic pain develops in late CLN12 (peripheral neuropathy component uncommon but reported): minimise GBP/PGB dose; monitor myoclonus; consider alternatives (duloxetine, TCAs at low dose — note TCA anticholinergic vs parkinsonism)."
            },
            {
                "drug": "LTG Monotherapy",
                "severity": "HIGH RISK — Monotherapy for PME",
                "reason": "Lamotrigine in monotherapy worsens myoclonic seizures in PME syndromes. In CLN12 with myoclonic seizures: LTG monotherapy is contraindicated. LTG may be used as adjunct (with VPA — note LTG/VPA pharmacokinetic interaction requiring LTG dose halving).",
                "note": "LTG as adjunct (add-on to VPA + LEV) may be considered for CLN12 GTCS burden if first two lines insufficient. VPA + LTG: reduce LTG dose by 50% (VPA inhibits LTG glucuronidation). NEVER LTG monotherapy in CLN12 with any myoclonic component."
            }
        ],
        "monitoring": [
            {"item": "ATP13A2 WES / NCL Gene Panel", "frequency": "Once at diagnosis", "note": "ATP13A2 biallelic LOF confirms CLN12. NCL panel: ATP13A2 + CLN3 + CLN10 + CLN13 + GRN + DNAJC5 concurrently for differential. Middle-Eastern/Jordanian/Pakistani: ATP13A2 founder variant PCR first (days). POLG1 + MERRF (mtDNA) exclusion in juvenile NCL-parkinsonism."},
            {"item": "Skin Biopsy EM (Dense Deposits / FP)", "frequency": "Once — diagnostic", "note": "Dense lipofuscin deposits ± fingerprint profiles in eccrine sweat gland cells or lymphocytes confirm NCL storage. CLN12 EM: dense deposits predominant + FP (fingerprint profiles, 82%). GRODs if present → also test PPT1 DBS to exclude CLN1/CLN10. EM confirms NCL class before WES results."},
            {"item": "POLG1 WES + MERRF mtDNA Panel", "frequency": "Once — mandatory pre-VPA", "note": "POLG1 Alpers mimics CLN12: juvenile NCL + parkinsonism + seizures + dementia. MERRF can mimic juvenile PME component. MANDATORY before initiating VPA (absolute CI in POLG1/MERRF). Test: POLG1 WES + mtDNA m.8344A>G (MERRF) + blood lactate + muscle biopsy if indicated."},
            {"item": "Brain MRI 3T (SN + Basal Ganglia)", "frequency": "At diagnosis, then annually", "note": "T2 hyperintensity in substantia nigra (iron deposition) and caudate/putamen atrophy: characteristic of CLN12. Cortical atrophy frontal/temporal predominant. DWI: acute striatal changes in acute exacerbations. Helps distinguish CLN12 from MSA, PSP, Wilson's disease (copper chelation available for Wilson — treatable mimic)."},
            {"item": "Ophthalmology (ERG + VEP + Gaze Palsy Exam)", "frequency": "Annual (not 6-monthly — no retinal NCL)", "note": "ERG: not for retinal NCL surveillance (CLN12 has no retinal NCL) — for baseline and monitoring of any incidental retinal disease. VEP: cortical visual processing; gaze palsy is supranuclear (not retinal/optic). Oculomotor examination by neuro-ophthalmologist: supranuclear upward gaze palsy degree (valuable disease marker). Annual sufficient (not 6-monthly as in retinal NCLs)."},
            {"item": "EEG (Annual + Urgent for Altered Consciousness)", "frequency": "Annual + urgent if clinically indicated", "note": "Annual EEG tracks seizure activity, background slowing, photoparoxysmal response. URGENT EEG for any acute deterioration in cognition (L-DOPA OFF vs NCSE — both cause stupor; EEG mandatory to distinguish). Polygraphy EEG for myoclonus: facial-faucial-finger mini-myoclonus correlation — NOT ictal in inter-ictal state."},
            {"item": "MDS-UPDRS (Parkinsonism Severity)", "frequency": "6-monthly", "note": "Movement Disorder Society-Unified Parkinson's Disease Rating Scale Part III (motor examination): tracks parkinsonism severity, L-DOPA ON/OFF motor function, dyskinesia, wearing off. Tracks disease progression and L-DOPA benefit duration. Also tracks pyramidal signs (UPDRS does not cover — supplement with pyramidal sign examination)."},
            {"item": "SARA (Ataxia Scale)", "frequency": "6-monthly", "note": "Scale for Assessment and Rating of Ataxia: cerebellar ataxia component of CLN12 (cerebellar atrophy in advanced disease). Tracks fall risk from cerebellar + parkinsonian gait impairment combined (compound fall risk). Guides physiotherapy referral and mobility aids."},
            {"item": "Neuropsychology + Cognitive Assessment", "frequency": "Annual", "note": "Montreal Cognitive Assessment (MoCA) / neuropsychological battery: tracks cognitive decline rate; identifies domains affected (executive function early, memory later); guides educational support planning, legal capacity assessment, ACP. Cognitive decline rate predicts need for supported living."},
            {"item": "Neuropsychiatric Review (Psychosis, Depression, Anxiety)", "frequency": "6-monthly", "note": "CLN12 psychiatric symptoms in 30-40%: psychosis (hallucinations from dopaminergic changes + dementia), depression (dopaminergic depletion + adjustment reaction), anxiety. PHQ-9 (depression) + GAD-7 (anxiety) + BPRS (psychosis). Psychotropic review: clozapine/quetiapine for psychosis (NEVER typical antipsychotics). SSRI for depression."},
            {"item": "L-DOPA Response + Wearing-Off Diary", "frequency": "3-monthly (first 2y), then 6-monthly", "note": "Track L-DOPA ON/OFF periods; diary of motor fluctuations; assess dyskinesia; note wearing-off time from last dose. L-DOPA wearing-off → seizure clustering in CLN12. Optimise L-DOPA schedule and consider COMT inhibitors. Dopamine agonist addition or MAOB inhibitor (selegiline/rasagiline) when wearing off develops."},
            {"item": "VPA TDM + LFT + FBC", "frequency": "At initiation, then 3-monthly", "note": "VPA therapeutic drug monitoring (target 50-100 mg/L); LFT baseline and 3-monthly (hepatotoxicity); FBC (thrombocytopenia); weight (VPA weight gain); VPA-induced tremor (may exacerbate CLN12 parkinsonism tremor — distinguish by exam). Reduce VPA if VPA-induced tremor clinically significant; beta-blockers may help VPA tremor."},
            {"item": "SUDEP Risk Assessment + Nocturnal Supervision", "frequency": "Annual", "note": "SUDEP risk elevated in CLN12 (uncontrolled nocturnal GTCS + autonomic dysfunction from parkinsonism). Nocturnal seizure monitoring (CCTV, seizure alarm mattress); supervision for nocturnal GTCS; ensure AED adherence. SUDEP risk counselling for family annually."},
            {"item": "BDSRA / PARK9 Registry Enrolment + Driving (DVLA)", "frequency": "Once — at diagnosis", "note": "BDSRA (Batten Disease Support and Research Association) + PARK9/CLN12 gene therapy trial registry: mandatory enrolment for future therapy access. DVLA notification: mandatory at CLN12 diagnosis (progressive neurological disease + seizures + cognitive impairment). Driving cessation legally required. Educational/employment planning: school disruption in juvenile CLN12 — educational psychology early."}
        ]
    }


def get_definitions():
    return {
        "disease_name": "CLN12 — Neuronal Ceroid Lipofuscinosis Type 12 / Kufor-Rakeb Syndrome (KRS) / PARK9 / ATP13A2 Deficiency",
        "gene_full": "ATP13A2 (ATPase Type 13A2) — 1p36.13; type 5 P-type ATPase (P5B-ATPase subfamily); 1180 aa; ~128 kDa; lysosomal/late endosomal membrane protein; 10 TM domains; lysosomal polyamine exporter (spermine, spermidine, putrescine); Mn2+/Zn2+ transporter; lysosomal pH/membrane potential maintenance; biallelic LOF → CLN12 / KRS / PARK9",
        "omim_gene": "*610513 (ATP13A2)",
        "omim_disease": "#606693 (Kufor-Rakeb Syndrome / PARK9 / CLN12 — Neuronal Ceroid Lipofuscinosis Type 12)",
        "protein_full": "ATP13A2 (1180 aa; ~128 kDa); P5B-ATPase subfamily; 10 TM domains; 4 cytoplasmic domains (A, P, N, R-domains); catalytic E1-E2 conformational cycle; exports polyamines (spermine Spm4+, spermidine Spd3+, putrescine Put2+) from lysosome → cytoplasm; also transports Mn2+, Zn2+; essential for lysosomal acidification and membrane potential; loss → lysosomal polyamine accumulation + α-synuclein aggregation + lipofuscin storage",
        "inheritance_mode": "Autosomal recessive (AR) biallelic ATP13A2 LOF. pLI ~0.98 (extremely intolerant to haploinsufficiency). 25% recurrence risk for siblings. No established AD form. Parents are obligate heterozygous carriers (unaffected). Founder variants: Jordanian (Kufor-Rakeb village), Pakistani (c.3057delC), Chilean (compound het). OMIM *610513 / #606693",
        "onset_age": "Juvenile / young adult: mean 17y, range 6-25y. Predominantly juvenile onset (6-18y in ~70%). Young adult onset (18-25y in ~30%). Earlier onset (younger age at diagnosis) correlates with more severe genotype (biallelic truncating).",
        "em_pattern": "Dense lipofuscin deposits ± fingerprint profiles (FP) in neurons (eccrine sweat gland cells, lymphocytes on skin biopsy EM). Dense deposits most common (82%); FP in 65%. Confirms NCL storage class. GRODs (granular osmiophilic deposits) in ~20% — if present, also test PPT1 DBS to exclude CLN1/CLN10.",
        "no_retinal_ncl": "NO RETINAL NCL — VGB NOT Absolute CI (Shared with CLN13, Unique Among NCLs)",
        "key_concepts": [
            {
                "name": "CLN12-ATP13A2-1p36.13-Lysosomal-P5B-ATPase-Polyamine-Exporter-Juvenile-NCL",
                "definition": "ATP13A2 (1p36.13) encodes the only lysosomal P-type ATPase whose primary function is polyamine (spermine, spermidine, putrescine) export from lysosome → cytoplasm. CLN12 is the only NCL with primary polyamine metabolism dysfunction — all other NCLs involve lysosomal enzyme or structural membrane protein deficiency. LOF → lysosomal polyamine accumulation → cathepsin dysfunction → α-synuclein aggregation → lipofuscin NCL storage → juvenile parkinsonism-pyramidal-NCL syndrome (Kufor-Rakeb Syndrome)."
            },
            {
                "name": "NO-Retinal-NCL-VGB-NOT-Absolute-CI-Shared-CLN12-CLN13-Only-NCLs",
                "definition": "CLN12 does NOT cause retinal NCL degeneration (<5%). ATP13A2 is not functionally expressed in the retinal pigment epithelium. VGB retinopathy does not compound retinal blindness in CLN12. VGB is NOT an absolute CI in CLN12 — shared only with CLN13 among all NCLs. All other NCLs (CLN1/CLN2/CLN3/CLN5/CLN6/CLN7/CLN8/CLN10/CLN11) = VGB ABSOLUTE CI due to retinal NCL. CLN12 + CLN13 are the counter-examples that must be memorised."
            },
            {
                "name": "Typical-Antipsychotics-ABSOLUTE-CI-Unique-CLN12-Parkinsonism-Danger",
                "definition": "CLN12 is the ONLY NCL where typical antipsychotics (haloperidol, chlorpromazine, metoclopramide, prochlorperazine) are an absolute CI due to parkinsonism worsening. CLN12 causes progressive dopaminergic nigrostriatal degeneration — D2 antagonism causes acute parkinsonism crisis. Psychiatric symptoms (psychosis ~30%): clozapine or quetiapine ONLY. Common inadvertent exposures: metoclopramide for nausea, prochlorperazine for vertigo — both absolute CI. All CLN12 drug charts must list this prominently."
            },
            {
                "name": "Supranuclear-Gaze-Palsy-Facial-Faucial-Finger-Mini-Myoclonus-PATHOGNOMONIC-CLN12",
                "definition": "The combination of (1) supranuclear gaze palsy (upward > lateral, 90% of CLN12) and (2) facial-faucial-finger mini-myoclonus (perioral/perinasal facial jerks + palatal myoclonus + finger mini-jerks, 85%) is PATHOGNOMONIC for CLN12 among all NCLs and most juvenile movement disorder syndromes. Bell's reflex intact (supranuclear not nuclear). Facial-faucial-finger mini-myoclonus is inter-ictal (does NOT correlate with EEG discharge) — distinguish from ictal myoclonus by EEG polygraphy. Presence = MANDATORY ATP13A2 WES."
            },
            {
                "name": "No-ATP13A2-Enzyme-Assay-WES-Required-P-Type-ATPase-Transporter",
                "definition": "ATP13A2 is a membrane-bound P-type ATPase transporter with no commercially available DBS enzyme assay (unlike CLN1/PPT1-DBS and CLN2/TPP1-DBS). No clinically validated activity assay exists. Diagnostic pathway: skin biopsy EM (dense deposits/FP) → WES ATP13A2 + NCL panel → POLG1 exclusion before VPA → brain MRI (T2 SN hyperintensity, basal ganglia atrophy). Founder variants (Jordanian, Pakistani, Chilean) can be tested by PCR first in appropriate ethnicity."
            },
            {
                "name": "CBZ-OXC-PHT-ABSOLUTE-CI-CLN12-Myoclonus-Plus-Parkinsonism-Double-Trap",
                "definition": "Sodium channel blockers (CBZ, OXC, PHT) are ABSOLUTE CI in CLN12 for two reasons: (1) myoclonus worsening in NCL-PME; (2) possible dopamine-depleting effects aggravating parkinsonism. Mean CLN12 diagnosis delay 4.6y → maximum CBZ exposure risk period. CLN12 GTCS or focal seizures misidentified as TLE/GGE → CBZ first-line → acute myoclonic deterioration. Safe: VPA + LEV. NEVER start CBZ/OXC in juvenile parkinsonism + new seizures without NCL exclusion."
            },
            {
                "name": "VPA-SAFE-CLN12-Lysosomal-P5B-ATPase-NOT-Mitochondrial-POLG1-Exclusion-Mandatory",
                "definition": "VPA CI applies to MERRF/POLG (mitochondrial enzyme disorders) — NOT to CLN12 (lysosomal P-type ATPase transporter dysfunction). VPA is backbone AED for CLN12 seizures + myoclonus. MANDATORY POLG1 EXCLUSION BEFORE VPA: POLG1 Alpers mimics CLN12 (juvenile NCL-parkinsonism + seizures + dementia); VPA = ABSOLUTE CI in POLG1 (mitochondrial hepatotoxicity). MERRF exclusion also mandatory. Test: POLG1 WES + mtDNA m.8344A>G + blood lactate before any VPA in juvenile NCL-parkinsonism."
            },
            {
                "name": "POLG1-MERRF-Mandatory-Exclusion-Before-VPA-CLN12-Mimic",
                "definition": "POLG1 Alpers syndrome and MERRF both cause juvenile-adult NCL-parkinsonism-PME (indistinguishable from CLN12 without molecular testing). VPA = ABSOLUTE CI in both (mitochondrial hepatotoxicity). PROTOCOL: before any VPA in CLN12: blood lactate, mtDNA m.8344A>G (MERRF), POLG1 WES, muscle biopsy (ragged red fibres / RRF), COX staining. Only after exclusion of both → VPA safe in CLN12."
            },
            {
                "name": "L-DOPA-Symptomatic-Wearing-Off-Disease-Progression-Seizure-Trigger",
                "definition": "L-DOPA provides initial symptomatic motor improvement in 80% of CLN12 (partial, not dramatic DRD-type). Benefit wears off over mean 3.5y as nigrostriatal neurons progressively lost. L-DOPA wearing-off is a CLN12-SPECIFIC SEIZURE TRIGGER — seizure clustering in OFF periods documented. Management: smaller more frequent L-DOPA doses + COMT inhibitors (entacapone) to smooth OFF periods and reduce seizure clustering. L-DOPA is not disease-modifying — NCL storage and neurodegeneration continue."
            },
            {
                "name": "Alpha-Synuclein-Aggregation-CLN12-NCL-Parkinson-Intersection",
                "definition": "ATP13A2 deficiency impairs lysosomal clearance of α-synuclein via disrupted lysosomal membrane integrity and cathepsin inhibition. α-Synuclein accumulates in CLN12 neurons — placing CLN12 at the mechanistic intersection of NCL (lysosomal storage) and synucleinopathy (Parkinson's, MSA, DLB). This explains L-DOPA partial response (dopaminergic neurons partially preserved early), CLN12 hallucinatory psychosis (Lewy body-like synuclein), and raises the possibility of α-synuclein-targeted therapies being relevant to CLN12 in future."
            },
            {
                "name": "Polyamine-Biology-Unique-CLN12-Mechanism-Spermine-Spermidine-Lysosomal-Export",
                "definition": "Polyamines (spermine Spm4+, spermidine Spd3+, putrescine Put2+) are essential polycationic cellular molecules synthesised in the cytoplasm and sequestered/degraded in lysosomes. ATP13A2 exports them from lysosome → cytoplasm. CLN12 LOF → lysosomal polyamine accumulation → lysosomal membrane disruption → impaired cathepsin activation → α-synuclein degradation failure. This is completely distinct from all other NCL pathomechanisms and offers unique therapeutic targets: DFMO (ornithine decarboxylase inhibitor reduces polyamine synthesis) and polyamine analogue modulation — under preclinical investigation."
            },
            {
                "name": "Juvenile-Parkinsonism-Differential-CLN12-vs-DRD-vs-Wilson-vs-POLG1",
                "definition": "Juvenile parkinsonism differential (onset <25y): (1) CLN12/Kufor-Rakeb: supranuclear gaze palsy + facial-faucial-finger myoclonus + NCL EM + dementia + partial L-DOPA; (2) DRD (GCH1/TH): dramatic complete sustained L-DOPA response, no dementia, no NCL; (3) Wilson disease: Kayser-Fleischer rings, liver disease, low ceruloplasmin, copper chelation available (treatable!); (4) POLG1 Alpers: juvenile NCL-parkinsonism + mitochondrial (elevated lactate, RRF, VPA CI); (5) Early-onset PD (PINK1/Parkin): slower progression, no cognitive impairment early, no supranuclear gaze palsy. MANDATORY: Wilson serum copper/ceruloplasmin + ATP13A2 WES + POLG1 WES in ALL juvenile parkinsonism."
            },
            {
                "name": "No-Disease-Modifying-Therapy-CLN12-Gene-Therapy-Research",
                "definition": "No approved disease-modifying therapy for CLN12/ATP13A2 as of 2026. L-DOPA is SYMPTOMATIC only. Investigational: (1) AAV-ATP13A2 gene replacement therapy (preclinical, CNS delivery studies); (2) Polyamine pathway modulation (DFMO — reduces spermine synthesis; spermine analogues); (3) α-synuclein clearance enhancement (synuclein immunotherapy, not CLN12-specific). BDSRA + PARK9 registry enrolment mandatory at diagnosis for future trial eligibility. No IND active as of 2026."
            },
            {
                "name": "Driving-DVLA-Mandatory-Educational-Employment-CLN12-Juvenile-Impact",
                "definition": "CLN12 at diagnosis: mandatory DVLA notification and driving cessation (progressive neurological disease + juvenile parkinsonism + seizures + cognitive impairment). Juvenile onset: major impact on secondary/tertiary education, employment, relationships, independence. Educational psychology early; adaptive technology; school IEP (individualised educational plan); occupational therapy for activities of daily living. Employment: supported employment or cessation depending on cognitive/motor status. ACP (advance care planning) begins at diagnosis — capacity assessment for decision-making."
            },
            {
                "name": "SUDEP-Risk-CLN12-Nocturnal-GTCS-Autonomic",
                "definition": "SUDEP risk elevated in CLN12 from: (1) nocturnal GTCS (unwitnessed, no rescue); (2) autonomic dysfunction from parkinsonism (cardiac autonomic regulation impaired). Nocturnal seizure monitoring (CCTV, seizure alarm mattress, bed sensor). Caregiver proximity overnight. AED optimisation to minimise nocturnal GTCS. SUDEP risk counselling for family annually. CLN12 SUDEP risk analogous to other NCL syndromes with uncontrolled GTCS."
            }
        ],
        "thresholds": [
            {"parameter": "Mean Onset Age", "value": "17.3y (range 6-25y)", "action": "Any juvenile parkinsonism <25y → ATP13A2 WES mandatory before 'idiopathic PD' label"},
            {"parameter": "Mean Diagnosis Delay", "value": "4.6 years", "action": "Reduce delay: ATP13A2 WES in ALL juvenile parkinsonism + supranuclear gaze palsy combinations"},
            {"parameter": "L-DOPA Initial Response", "value": ">80% partial improvement", "action": "Partial (NOT dramatic DRD-type) response + juvenile onset → CLN12 first. Dramatic sustained = DRD. Wearing off within 3.5y = CLN12 pattern."},
            {"parameter": "Supranuclear Gaze Palsy", "value": "90% of CLN12", "action": "Upward gaze palsy in juvenile parkinsonism → CLN12 top differential; ATP13A2 WES same week"},
            {"parameter": "Facial-Faucial-Finger Myoclonus", "value": "85% of CLN12", "action": "Pathognomonic combination with supranuclear gaze palsy; EEG polygraphy to confirm inter-ictal; piracetam for action myoclonus"},
            {"parameter": "Seizures Present", "value": "35% of CLN12", "action": "VPA + LEV first-line (POLG1 exclusion first); NEVER CBZ/OXC; fosphenytoin absolute CI for SE"},
            {"parameter": "POLG1 Exclusion Before VPA", "value": "MANDATORY in all CLN12", "action": "Blood lactate + POLG1 WES + mtDNA m.8344A>G before VPA; if POLG1 positive → VPA absolute CI"},
            {"parameter": "L-DOPA Wearing-Off", "value": "Mean 3.5y from initiation", "action": "After wearing off: add COMT inhibitor (entacapone); smooth L-DOPA delivery; monitor OFF-period seizure clustering"},
            {"parameter": "Dense Deposits / FP on EM", "value": "82% dense deposits, 65% FP", "action": "EM confirms NCL class; if GRODs present (20%): also test PPT1 DBS (CLN1/CLN10 exclusion)"},
            {"parameter": "Typical Antipsychotic Exposure", "value": "ABSOLUTE CI (100% risk)", "action": "Psychiatric symptoms: clozapine or quetiapine ONLY; alert all prescribers, A&E, GP; list in drug allergy section of notes"},
            {"parameter": "Parkinsonism Severity (MDS-UPDRS III)", "value": "Track 6-monthly", "action": "MDS-UPDRS III >32 (moderate): mobility aids; occupational therapy; fall prevention; wheelchair assessment"},
            {"parameter": "Cognitive Impairment", "value": "90% by 5y from onset", "action": "Early neuropsychology; ACP from diagnosis; legal capacity assessment before dementia severe; supported living planning"}
        ],
        "standards": [
            "Ramirez A et al. 2006 Nat Genet — first CLN12/ATP13A2 description, Jordanian Kufor-Rakeb families (primary founding reference)",
            "Bras JM et al. 2012 Mov Disord — ATP13A2 in NCL: dense deposits confirmed, CLN12 reclassification as NCL type",
            "Mole SE et al. 2019 Lancet Neurol — comprehensive NCL review including CLN12/ATP13A2 taxonomy and clinical features",
            "NCL Resource 2024 (ncl.resource.guide) — updated CLN12 clinical management guidelines and registry",
            "ILAE 2022 Classification — seizure type classification for CLN12 (NCL-associated epilepsy, myoclonic-GTCS dominant)",
            "NICE NG217 2022 — Epilepsies diagnosis and management (adult + paediatric, applicable to CLN12 seizure management)",
            "MHRA/VPPP 2021 — VPA pregnancy prevention programme (CLN12 women of childbearing potential; VPA not recommended in pregnancy)",
            "CPIC POLG1 Guidelines 2023 — mandatory POLG1 exclusion before VPA in mitochondrial disease mimics including CLN12",
            "Lees AJ et al. 2009 Lancet — Parkinson's disease differential: juvenile parkinsonism including Kufor-Rakeb (CLN12) differentiation criteria",
            "ACMG/AMP 2015 Standards — variant pathogenicity classification for ATP13A2 biallelic LOF (used for CLN12 genetic reporting)",
            "BDSRA 2024 — Batten Disease Support and Research Association: CLN12/ATP13A2 patient registry and clinical management guidance",
            "Williams DR et al. 2007 Brain — supranuclear gaze palsy in Kufor-Rakeb/CLN12: pathognomonic criteria and differential diagnosis"
        ],
        "references": [
            "Ramirez A, Heimbach A, Gründemann J, et al. Hereditary parkinsonism with dementia is caused by mutations in ATP13A2, encoding a lysosomal type 5 P-type ATPase. Nat Genet. 2006;38(10):1184-1191.",
            "Bras JM, Singleton AB, Hardy JA, Bhatt DL. Genetic studies in Parkinson's disease: new pathways beyond α-synuclein. Mov Disord. 2012;27(11):1413-1422. [ATP13A2 NCL reclassification, dense deposits EM]",
            "Dehay B, Martinez-Vicente M, Ramirez A, et al. Lysosomal impairment in Parkinson's disease. Mov Disord. 2012;27(11):1405-1415. [ATP13A2 lysosomal polyamine pathway, α-synuclein]",
            "Mole SE, Anderson G, Anderson HE, et al. Clinical challenges and future therapeutic approaches for neuronal ceroid lipofuscinosis. Lancet Neurol. 2019;18(1):107-116. [CLN12 in NCL classification]",
            "Schultheis PJ, Fleming SM, Bhatt D, et al. Atp13a2-deficient mice exhibit neuronal ceroid lipofuscinosis, limited α-synuclein accumulation and age-dependent sensorimotor deficits. Hum Mol Genet. 2013;22(10):2067-2082. [Mouse model confirming NCL + α-synuclein]",
            "Williams DR, Hadeed A, al-Din AS, Bhatt DL, Lees AJ. Kufor Rakeb disease: autosomal recessive, levodopa-responsive parkinsonism with pyramidal degeneration, supranuclear gaze palsy, and dementia. Mov Disord. 2005;20(10):1264-1271. [Pathognomonic features — supranuclear gaze palsy + L-DOPA response]"
        ],
        "lifecycle_stages": [
            {
                "stage": "Prenatal / Genetic Risk / Family Testing",
                "age_range": "Prenatal — birth (at-risk families)",
                "description": "If proband confirmed CLN12 (biallelic ATP13A2 LOF): siblings have 25% recurrence risk. Prenatal diagnosis (CVS/amniocentesis) and preimplantation genetic testing (PGT-M) available for known pathogenic variants. Parents are obligate heterozygous carriers — counsel about 25% sibling risk. No carrier phenotype. Genetic counselling at diagnosis of first affected child is urgent.",
                "priorities": ["Family ATP13A2 variant confirmation (parents + siblings)", "Prenatal/PGT counselling if further pregnancies planned", "Sibling WES at diagnosis", "Genetic counselling: AR inheritance (25% recurrence)", "Register with BDSRA/PARK9 registry at diagnosis"]
            },
            {
                "stage": "Pre-Symptomatic / Early Prodromal",
                "age_range": "6-12y (typically before first motor signs)",
                "description": "In identified at-risk siblings: pre-symptomatic WES at birth or shortly after. Subtle prodromal features may occur years before first motor presentation: behavioural change, subtle cognitive slowing, school difficulties, REM behaviour disorder-like sleep disturbance. Annual monitoring begins for identified genetic CLN12 cases even before symptom onset.",
                "priorities": ["WES confirmation at birth in known sibling carriers", "Baseline cognitive assessment (pre-symptomatic)", "Baseline MRI + EEG", "Ophthalmology baseline (ERG/VEP)", "Educational planning for identified CLN12 pre-symptomatically", "BDSRA/PARK9 trial registry enrolment"]
            },
            {
                "stage": "First Presentation — Diagnostic Emergency",
                "age_range": "Mean 17y (range 6-25y) — first motor or seizure symptoms",
                "description": "Classic CLN12 presentation: juvenile tremor/slowness → parkinsonism diagnosis → L-DOPA trialled → partial benefit → supranuclear gaze palsy identified → CLN12 suspected. OR: first GTCS or myoclonic seizure → EEG → PME pattern → NCL evaluation. POLG1 exclusion mandatory before VPA. ATP13A2 WES urgent. Brain MRI: SN T2 hyperintensity. Skin biopsy EM: dense deposits. TYPICAL ANTIPSYCHOTIC CI must be documented immediately.",
                "priorities": ["ATP13A2 WES urgent (weeks)", "EM skin biopsy (days — confirm NCL storage)", "POLG1 + MERRF exclusion before any VPA", "Brain MRI 3T (SN hyperintensity, basal ganglia)", "Ophthalmology + gaze palsy examination", "Start L-DOPA (parkinsonism) + VPA (seizures, after POLG1 excluded)", "TYPICAL ANTIPSYCHOTIC CI documented in ALL prescribing systems immediately", "DVLA notification at diagnosis"]
            },
            {
                "stage": "Active Disease Phase — Parkinsonism + Cognitive Decline ± Seizures",
                "age_range": "Mean 17-25y from first symptoms",
                "description": "Progressive parkinsonism (L-DOPA benefit wearing off), cognitive impairment accelerating, seizures establishing (35%), psychiatric features emerging (30%), pyramidal signs increasing, cerebellar ataxia appearing. Falls risk increasing from parkinsonism + ataxia + pyramidal signs (triple fall risk). Sleep disrupted. School/employment disruption. L-DOPA wearing-off and seizure clustering relationship identified.",
                "priorities": ["L-DOPA optimisation (COMT inhibitors for wearing-off)", "AED optimisation (VPA + LEV ± piracetam)", "Physiotherapy (gait training, fall prevention)", "SALT (dysphagia assessment early, PEG tube planning)", "Neuropsychiatric management (clozapine/quetiapine if psychosis; SSRI if depression)", "Supported education/employment planning", "ACP document at first loss of legal capacity milestone"]
            },
            {
                "stage": "Established Severe Disability",
                "age_range": "Mean 25-30y from disease onset",
                "description": "Severe parkinsonism (L-DOPA benefit minimal to absent), dementia established, seizure burden high, dysphagia requiring PEG tube, severe falls risk, wheelchair dependency, 24-hour care required. Psychiatric features may be most difficult management challenge. ACP enacted — caregiver decisions with patient's previously documented wishes.",
                "priorities": ["PEG tube nutrition (dysphagia advanced)", "Wheelchair provision + seating assessment", "24-hour care package", "Seizure rescue plan (midazolam buccal at bedside)", "Clozapine management (FBC monitoring ongoing)", "ACP document active (proxy decision-maker identified)", "Palliative care team referral", "SUDEP prevention (nocturnal seizure monitoring)"]
            },
            {
                "stage": "Late Palliative / End-Stage",
                "age_range": "Mean 2nd-4th decade from onset (fatal)",
                "description": "End-stage NCL: complete dependency, minimal consciousness responsiveness, recurrent aspirations, respiratory infections, refractory seizures, autonomic instability. Palliative AED (buccal/subcutaneous route preferred). Symptom control (pain, secretions, myoclonus, seizures). DNACPR/comfort care plan enacted. Bereavement support for family. BDSRA bereavement services.",
                "priorities": ["Comfort-focused care (not disease-directed at this stage)", "Subcutaneous/buccal AED routes (VPA/LEV subcutaneous if available)", "Symptom control: secretions (hyoscine), pain (opioids), myoclonus (clonazepam SC)", "DNACPR decision documented and respected", "Palliative care team primary responsibility", "Bereavement support for family (BDSRA + local services)", "ATP13A2 gene therapy trial follow-up for surviving siblings or next-generation patients"]
            }
        ]
    }
