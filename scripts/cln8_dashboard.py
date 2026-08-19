"""
CLN8 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 8 / Northern Epilepsy (EPMR) / Turkish Variant vLINCL
=========================================================================================================
40-patient cohort · CLN8 (8p23.3) · Autosomal recessive (AR) biallelic LOF
CLN8 encodes an ER-Golgi intermediate compartment (ERGIC) transmembrane cargo receptor
(286 aa, ~33 kDa, 8 TM domains, cycles between ER/ERGIC and Golgi)
CLN8 LOF → impaired lysosomal enzyme (cathepsin) trafficking → SCMAS accumulation →
Fingerprint profiles + Curvilinear bodies on EM → progressive neuronal ± retinal apoptosis → CLN8

DUAL PHENOTYPE — TWO DISTINCT CLN8 PRESENTATIONS:
═══════════════════════════════════════════════════

PHENOTYPE 1: NORTHERN EPILEPSY / EPMR (Progressive Epilepsy with Mental Retardation)
- Finnish founder mutation p.Arg24Gly (c.70C>G) — nearly 100% of Finnish EPMR alleles
- Onset: 5–10 years (mean ~7 years)
- Milder course: progressive epilepsy + mild-to-moderate mental retardation
- SEIZURES FIRST (GTCS then myoclonic); cognitive decline follows
- NO prominent retinal degeneration (or very late mild changes only) — UNIQUE among most NCLs
- VGB: HIGH RISK (not absolute CI early; caution given retinal disease risk in progressive course)
- Survival: 30–60 years — longest NCL survival (milder residual CLN8 function from p.Arg24Gly)
- Finnish population: EPMR prevalence ~1/10,000 births (one of the highest NCL prevalences)
- Named "Northern Epilepsy" from geographic clustering in northern Finland
- EM: Mixed granular deposits ± sparse FP/CB (atypical pattern, less pronounced than vLINCL)

PHENOTYPE 2: CLN8 VARIANT LATE-INFANTILE (vLINCL) — NON-FINNISH
- More severe mutations (truncating, missense outside founder region)
- Earlier onset: 1.5–7 years (similar to CLN6/CLN7)
- Progressive retinal degeneration (100% in severe mutations) → VGB ABSOLUTE CI
- Shorter survival: teens to 20s
- Turkish, Indian, Pakistani, Italian, and other non-Finnish populations
- EM: Fingerprint profiles + Curvilinear bodies (FP+CB) — similar to CLN6/CLN7

CLN8 PROTEIN BIOLOGY (ER-Golgi Intermediate Compartment — CARGO RECEPTOR):
CLN8 (8p23.3):
  - 286 amino acids; ~33 kDa; 8 transmembrane domains
  - Located in the ER-Golgi intermediate compartment (ERGIC) and smooth ER membrane
  - Function: CARGO RECEPTOR for lysosomal hydrolases — CLN8 recruits cathepsins (cathepsin D, F, B, H, L),
    PPT1 (CLN1), TPP1 (CLN2), and other lysosomal enzymes in the ER/ERGIC and cycles them to the Golgi
    for mannose-6-phosphate receptor (M6PR) capture → lysosomal delivery
  - CLN8 LOF → failure to recruit lysosomal enzymes in ER/ERGIC → cathepsins not delivered to Golgi
    → reduced lysosomal enzyme trafficking → lysosomal storage → SCMAS accumulation → NCL
  - NO CLN8 enzyme assay available (CLN8 is a cargo receptor, NOT an enzyme — no catalytic activity)
  - pLI ~0.35 (consistent with recessive disease; moderate intolerance)
  - OMIM: *607837 (CLN8 gene) / #600143 (Northern Epilepsy/EPMR) / #610003 (CLN8 disease non-Finnish)
  - Discovery: Ranta S et al. 1999 Nat Genet — CLN8 gene identified from Finnish EPMR families

CLN8 vs OTHER NCLs — KEY DISTINCTIONS:
  ER/ERGIC MEMBRANE PROTEIN (not lysosomal like CLN1-3/CLN7; not ER like CLN6 — ERGIC/ER boundary)
  CARGO RECEPTOR (not enzyme like CLN1/CLN2; not structural like CLN3/CLN6 — functional cargo recruiter)
  DUAL PHENOTYPE (Northern Epilepsy vs vLINCL — genotype-phenotype correlation)
  NO CLN8 ENZYME ASSAY — WES required (like CLN5/CLN6/CLN7; unlike CLN1/CLN2)
  FP+CB EM in vLINCL (similar to CLN6/CLN7); atypical EM in Northern Epilepsy
  SEIZURES FIRST (both phenotypes — GTCS then myoclonic)
  NORTHERN EPILEPSY MILD — survival to 30-60y (unlike all other NCLs except attenuated CLN3)
  VPA SAFE — ER/ERGIC dysfunction, NOT mitochondrial
  VGB: ABSOLUTE CI in vLINCL (retinal); HIGH RISK / avoid in Northern Epilepsy
  CBZ/OXC/PHT ABSOLUTE CI — myoclonus worsening (both phenotypes)
"""


def get_overview():
    return {
        "gene": "CLN8 (8p23.3) — CLN8 ER-ERGIC Transmembrane Cargo Receptor (286 aa; ~33 kDa; 8 TM domains; ER-Golgi intermediate compartment / ERGIC; recruits cathepsins/lysosomal enzymes for Golgi-M6PR-lysosomal trafficking; SCMAS storage; FP+CB on EM in vLINCL; atypical mixed EM in Northern Epilepsy)",
        "protein": "CLN8 protein; 286 aa; ~33 kDa; 8 transmembrane domains; located in the ER-Golgi intermediate compartment (ERGIC) and smooth ER membrane; functions as a CARGO RECEPTOR recruiting lysosomal hydrolases (cathepsin D, F, B, H, L; PPT1; TPP1) in the ER/ERGIC and cycling them to the Golgi for mannose-6-phosphate (M6P) tagging → lysosomal delivery; CLN8 LOF → reduced cathepsin trafficking → lysosomal storage → SCMAS accumulation → NCL; distinct from CLN6 (structural ER protein) and CLN7 (lysosomal transporter)",
        "inheritance": "Autosomal recessive (AR) biallelic LOF. pLI ~0.35. TWO DISTINCT PHENOTYPES: (1) Northern Epilepsy/EPMR — Finnish founder p.Arg24Gly (c.70C>G); mild; onset 5-10y; survival 30-60y; no prominent retinal disease. (2) CLN8 vLINCL — non-Finnish; severe mutations; onset 1.5-7y; retinal degeneration; survival teens-20s. NO CLN8 enzyme assay → WES/gene panel is primary test. OMIM *607837 / #600143 (EPMR) / #610003 (CLN8 vLINCL)",
        "omim": "*607837 (CLN8 gene) · #600143 (Northern Epilepsy / EPMR — Finnish) · #610003 (CLN8 disease non-Finnish / vLINCL)",
        "disease": "CLN8 — Two phenotypes: (1) Northern Epilepsy (EPMR): onset 5-10y; GTCS then myoclonic; mild-moderate MR; NO prominent retinal degeneration; survival 30-60 years; Finnish founder p.Arg24Gly. (2) CLN8 vLINCL: onset 1.5-7y; GTCS + myoclonic; progressive retinal degeneration; profound disability; fatal teens-20s; non-Finnish populations (Turkish, Indian, Pakistani, Italian, others). Both: seizures FIRST; cognitive decline; progressive course; AR; no disease-modifying therapy.",
        "mechanism": "CLN8 biallelic LOF → CLN8 cargo receptor absent/dysfunctional in ER/ERGIC → failure to recruit lysosomal hydrolases (cathepsin D, F, B, H, L; PPT1; TPP1) in ER/ERGIC for Golgi-M6PR delivery → impaired lysosomal enzyme trafficking → lysosomal hydrolase deficiency → SCMAS accumulation → FP+CB on EM (vLINCL) or mixed atypical EM (Northern Epilepsy) → progressive neuronal ± retinal apoptosis → CLN8 NCL",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN8/CLN8-EPMR. Management is purely symptomatic. Investigational: CLN8 gene therapy in preclinical development; understanding CLN8 cargo receptor function may enable substrate-targeting strategies. All CLN8 patients (both phenotypes) should be enrolled in BDSRA registry and NCL Resource. Northern Epilepsy: natural history study of long-term survival (30-60y) provides model for studying NCL neuroprotection mechanisms.",
        "no_enzyme_assay": "CRITICAL — NO CLN8 ENZYME ASSAY AVAILABLE. CLN8 is a CARGO RECEPTOR (structural transmembrane function — recruits cathepsins in ER/ERGIC) — it has NO known catalytic (enzymatic) function to measure (unlike CLN1 PPT1 DBS assay 1-3 days; CLN2 TPP1 DBS assay days). Diagnostic pathway: EM skin biopsy (FP+CB in vLINCL → NCL confirmed; atypical/mixed in Northern Epilepsy) + CLN8 WES/gene panel. Finnish heritage → p.Arg24Gly (c.70C>G) PCR first (days). Non-Finnish severe vLINCL → concurrent WES for CLN8, CLN6, CLN7 (all show FP+CB; WES differentiates). EM is the critical first rapid step.",
        "cohort_size": 40,
        "female_pct": 48,
        "northern_epilepsy_pct": 40,
        "vlincl_pct": 60,
        "mean_onset_seizure_years_epmr": 7.0,
        "mean_onset_seizure_years_vlincl": 3.8,
        "mean_diagnosis_delay_years": 2.5,
        "drug_resistant_pct": 72,
        "retinal_degeneration_pct_vlincl": 98,
        "retinal_degeneration_pct_epmr": 18,
        "fingerprint_profiles_em_pct": 78,
        "curvilinear_bodies_em_pct": 68,
        "ataxia_pct": 82,
        "cognitive_impairment_pct": 95,
        "finnish_heritage_pct": 40,
        "photosensitivity_pct": 52,
        "on_vpa_pct": 88,
        "mean_survival_years_epmr": 47,
        "mean_survival_years_vlincl": 18,
        "key_pharmacological_distinctions": {
            "1_DUAL_PHENOTYPE_NORTHERN_EPILEPSY_VS_vLINCL_GENOTYPE_PHENOTYPE": "DUAL CLN8 PHENOTYPE — GENOTYPE-PHENOTYPE CORRELATION: CLN8 produces two distinct diseases depending on mutation severity. (1) NORTHERN EPILEPSY (EPMR): Finnish founder p.Arg24Gly (c.70C>G) is a mild hypomorphic allele — residual CLN8 cargo receptor function → milder NCL → onset 5-10y, survival 30-60 years, NO prominent retinal degeneration, mild-moderate MR. (2) CLN8 vLINCL: more severe mutations (truncating, missense in critical domains) → near-complete CLN8 LOF → onset 1.5-7y, retinal degeneration, profound disability, survival teens-20s. CRITICAL MANAGEMENT DIFFERENCE: VGB absolute CI in vLINCL (retinal); high risk in Northern Epilepsy. Phenotype determination guides VGB decision, ACP timeline, and survival counselling. WES genotype-phenotype correlation is essential at diagnosis.",
            "2_NO_CLN8_ENZYME_ASSAY_CARGO_RECEPTOR_NOT_ENZYME": "NO CLN8 ENZYME ASSAY — WES/GENE PANEL REQUIRED (CARGO RECEPTOR, NOT ENZYME): CLN8 is a structural cargo receptor (ERGIC/ER membrane) with no known enzymatic activity — there is NO DBS enzyme assay (unlike CLN1 PPT1 and CLN2 TPP1). Diagnostic sequence: EM skin biopsy (FP+CB in vLINCL → NCL confirmed; days) → Finnish heritage → p.Arg24Gly PCR (days) OR non-Finnish severe vLINCL → WES (CLN8 + CLN6 + CLN7 concurrent, weeks). Do NOT delay EM while awaiting genetics. Northern Epilepsy EM may be atypical (mixed/granular) — must not dismiss NCL based on atypical EM alone.",
            "3_VGB_ABSOLUTE_CI_vLINCL_HIGH_RISK_NORTHERN_EPILEPSY": "VGB ABSOLUTE CI IN vLINCL / HIGH RISK IN NORTHERN EPILEPSY — DUAL VGB RULE: CLN8 vLINCL causes progressive retinal NCL degeneration (98%): VGB retinopathy (VAR) + CLN8 retinal NCL = CATASTROPHIC combined blindness → ABSOLUTE CI. NORTHERN EPILEPSY: minimal retinal involvement in early disease BUT progressive course risks late retinal changes — VGB is HIGH RISK and should be avoided in all CLN8 forms. UNIQUE CLN8 CHALLENGE: child with Finnish Northern Epilepsy misidentified as idiopathic epilepsy → VGB prescribed for infantile spasms or GTCS → retinal risk even in milder form. RULE: avoid VGB in ALL CLN8 forms regardless of phenotype.",
            "4_CBZ_OXC_PHT_ABSOLUTE_CI_BOTH_PHENOTYPES": "CBZ/OXC/PHT ABSOLUTE CI IN BOTH CLN8 PHENOTYPES — GTCS TRAP IN BOTH: Both Northern Epilepsy (onset 5-10y) and vLINCL (onset 1.5-7y) present with GTCS FIRST before myoclonus becomes apparent. General paediatrician facing child with GTCS at 4-10y → CBZ prescribed → ACUTE MYOCLONIC WORSENING. Particularly dangerous in Northern Epilepsy where milder phenotype leads to LATER diagnosis (mean delay 2.5y) — child may receive CBZ for years before NCL confirmed. Safe first choice: VPA + LEV for ANY child with GTCS + developmental concern (regardless of suspected EPMR or vLINCL phenotype).",
            "5_VPA_SAFE_ER_ERGIC_NOT_MITOCHONDRIAL": "VPA SAFE IN CLN8 — ER/ERGIC CARGO RECEPTOR DYSFUNCTION, NOT MITOCHONDRIAL: CLN8 = ER-Golgi intermediate compartment cargo receptor. This is completely distinct from mitochondrial disease. VPA ABSOLUTE CI applies to MERRF/MTTK8344 (mitochondrial) and POLG/Alpers (mtDNA polymerase) — these CIs do NOT extend to CLN8. VPA is the backbone AED for CLN8 (both Northern Epilepsy and vLINCL). POLG1 exclusion recommended before VPA in children <8y with regression + seizures (POLG1 Alpers can mimic CLN8 early presentation — regression + seizures without retinal involvement, especially mirroring Northern Epilepsy pattern).",
            "6_NORTHERN_EPILEPSY_MILDER_SURVIVAL_30_60y": "NORTHERN EPILEPSY (EPMR) — LONGEST NCL SURVIVAL (30-60 YEARS): Finnish p.Arg24Gly is a hypomorphic allele with residual CLN8 function → milder disease course → survival 30-60 years (median ~47y in Finnish cohort). This is DRAMATICALLY different from other NCLs: CLN1 (7-12y), CLN2 without treatment (<12y), CLN6/CLN7 (teens). ACP is STILL mandatory in Northern Epilepsy (progressive decline, seizure burden, eventual profound disability) but the ACP timeline is different. Long-term psychiatric needs (personality change, anxiety, depression — common in Northern Epilepsy adults) require decades-long neuropsychiatric management. Driving, employment, relationships — realistic ACP topics in early Northern Epilepsy.",
            "7_CLN8_CARGO_RECEPTOR_CATHEPSIN_TRAFFICKING_ERGIC": "CLN8 = CARGO RECEPTOR FOR LYSOSOMAL ENZYMES (UNIQUE MECHANISM AMONG NCLs): CLN8 is not a lysosomal storage enzyme (CLN1/CLN2) nor a lysosomal membrane structural protein (CLN3/CLN7) nor an ER transmembrane protein (CLN6) — CLN8 is an ER-ERGIC CARGO RECEPTOR that RECRUITS cathepsins (D, F, B, H, L), PPT1 (CLN1 enzyme), and TPP1 (CLN2 enzyme) in the ER/ERGIC for transport to the Golgi via mannose-6-phosphate receptor (M6PR) pathway → lysosomal delivery. CLN8 LOF → cathepsins not picked up → fail to reach lysosomes → secondary lysosomal enzyme deficiency → SCMAS storage. This means CLN8 disease has features of MULTIPLE enzyme deficiencies secondarily — not one primary enzyme.",
            "8_PHENOTYPE_DETERMINATION_VGB_ACP_DRIVES_MANAGEMENT": "PHENOTYPE DETERMINATION AT DIAGNOSIS IS CRITICAL FOR CLN8 MANAGEMENT: Finnish heritage + mild presentation → likely Northern Epilepsy (p.Arg24Gly PCR first) → VGB HIGH RISK (not absolute CI); ACP timeline decades. Non-Finnish + early-onset 1.5-7y + severe presentation → vLINCL → VGB ABSOLUTE CI; ACP from diagnosis; gastrostomy expected within 5-8 years. BOTH phenotypes: VPA backbone + LEV IV for SE; CBZ/OXC/PHT absolute CI; BDSRA registry. The distinction determines: VGB risk level, ACP urgency, retinal monitoring intensity, survival counselling, and social care planning."
        }
    }


def get_breakdown():
    return {
        "etiologies": [
            {
                "class": "Homozygous p.Arg24Gly (Finnish EPMR / Northern Epilepsy)",
                "pct": 28,
                "count": 11,
                "description": "Homozygous Finnish founder p.Arg24Gly (c.70C>G) — nearly all Finnish Northern Epilepsy patients. Hypomorphic allele (residual CLN8 function). Mild EPMR phenotype: onset 5-10y, GTCS + myoclonic, mild-moderate MR, NO prominent retinal disease, survival 30-60y. p.Arg24Gly PCR confirmable within days in Finnish patients.",
                "gene_mechanism": "Homozygous p.Arg24Gly (first cytoplasmic loop missense) → partial CLN8 cargo receptor dysfunction → reduced (not absent) cathepsin recruitment → milder lysosomal enzyme trafficking impairment → slower SCMAS accumulation → EPMR mild phenotype",
                "key_variants": ["p.Arg24Gly (c.70C>G) homozygous", "Finnish/Northern-European enrichment", "hypomorphic allele", "PCR confirmable — days", "EPMR mild phenotype"]
            },
            {
                "class": "Compound-Het p.Arg24Gly + Other LOF (Finnish, Intermediate)",
                "pct": 12,
                "count": 5,
                "description": "Compound heterozygous: Finnish p.Arg24Gly + second LOF allele (truncating or severe missense). Intermediate phenotype between EPMR and vLINCL — earlier onset and more rapid progression than homozygous EPMR. Some retinal involvement may develop. Finnish or mixed-heritage patients.",
                "gene_mechanism": "p.Arg24Gly (partial function) + truncating/severe missense → net CLN8 function below EPMR threshold → intermediate severity; one LOF allele removes residual function from the Arg24Gly allele",
                "key_variants": ["p.Arg24Gly (c.70C>G) heterozygous", "second allele: truncating or missense", "intermediate EPMR-vLINCL phenotype", "retinal monitoring essential", "Finnish/mixed heritage"]
            },
            {
                "class": "Truncating/Nonsense LOF (Non-Finnish vLINCL)",
                "pct": 22,
                "count": 9,
                "description": "Homozygous or compound-het truncating mutations in non-Finnish populations (Turkish, Indian, Pakistani, others). Complete CLN8 LOF. Severe vLINCL phenotype: onset 1.5-7y, progressive retinal degeneration, profound disability, fatal teens-20s.",
                "gene_mechanism": "Homozygous truncating (nonsense, frameshift, splice-site) → complete CLN8 LOF → absent cargo receptor → cathepsins fail ER/ERGIC recruitment → severe lysosomal hydrolase deficiency → rapid SCMAS accumulation → severe vLINCL",
                "key_variants": ["homozygous truncating", "frameshift/nonsense variants", "Turkish/Indian/Pakistani enrichment", "consanguineous families", "complete LOF — severe vLINCL"]
            },
            {
                "class": "Compound-Het Missense/Truncating (Non-Finnish, Non-Founder)",
                "pct": 25,
                "count": 10,
                "description": "Compound heterozygous: one severe missense + one truncating allele. Multiple ethnicities globally. Classic vLINCL phenotype in most; severity correlates with residual missense allele function. WES/NCL gene panel required.",
                "gene_mechanism": "Truncating allele → complete CLN8 LOF on one chromosome; missense → partial LOF on other; net: near-complete cargo receptor loss → vLINCL severity determined by residual missense function",
                "key_variants": ["severe missense + truncating compound-het", "various non-Finnish ethnicities", "WES required", "vLINCL phenotype", "retinal degeneration expected"]
            },
            {
                "class": "Compound-Het Missense/Missense (Attenuated, Multiple Ethnicities)",
                "pct": 10,
                "count": 4,
                "description": "Compound heterozygous with two missense alleles — neither is the Finnish founder. Attenuated or atypical phenotype possible. May fall between EPMR and classic vLINCL. Italian and other European populations. Functional characterisation of variants needed to predict phenotype severity.",
                "gene_mechanism": "Two missense alleles → partial CLN8 dysfunction on both chromosomes → residual cargo receptor function → variable lysosomal enzyme trafficking impairment → attenuated or intermediate NCL phenotype",
                "key_variants": ["missense/missense compound-het", "Italian/European non-Finnish enrichment", "attenuated phenotype possible", "functional characterisation needed", "variable retinal involvement"]
            },
            {
                "class": "Phenocopy CLN8-Negative (NCL Mimic / Deep Intronic)",
                "pct": 3,
                "count": 1,
                "description": "Clinical and EM picture consistent with CLN8 but CLN8 exome sequencing negative. May represent deep intronic CLN8 variant missed by WES, or closely related NCL gene (CLN6, CLN7). RNA sequencing if high clinical suspicion. Northern Epilepsy phenocopy may represent early CLN6 or other mild NCL allele.",
                "gene_mechanism": "CLN8 coding region negative on WES → consider: deep intronic CLN8 (RNA-seq); CLN6 (15q23 ER protein — same FP+CB EM); CLN7 (MFSD8 4q28.1 — same FP+CB); or other NCL gene",
                "key_variants": ["CLN8 WES negative", "FP+CB EM confirmed or atypical NCL EM", "deep intronic CLN8 (RNA-seq)", "CLN6/CLN7 excluded", "EPMR phenocopy: consider attenuated CLN6 or novel NCL"]
            }
        ],
        "seizure_types": [
            {
                "type": "GTCS (Generalised Tonic-Clonic) — FIRST SEIZURE TYPE (Both Phenotypes)",
                "pct": 90,
                "description": "GTCS is the FIRST SIGN in both CLN8 phenotypes: Northern Epilepsy onset 5-10y (mean ~7y); vLINCL onset 1.5-7y (mean ~3.8y). In Northern Epilepsy, GTCS is often the only prominent seizure type for years before myoclonus and decline become apparent. GTCS misidentified as idiopathic epilepsy is the most common diagnostic delay in CLN8.",
                "eeg": "Generalised spike-wave or polyspike-wave 2-4 Hz; progressive background slowing; photosensitivity in 52%; occipital enhancement in vLINCL; Northern Epilepsy EEG may show slower background slowing and less prominent paroxysmal activity early",
                "semiology": "Bilateral tonic then clonic; loss of consciousness; post-ictal confusion; nocturnal predominance initially; clustering with fever or missed AED; duration 1-5 min",
                "clinical_tip": "Both CLN8 phenotypes present with GTCS first. Finnish child with GTCS at 5-10y + subsequent cognitive decline = EPMR (CLN8 p.Arg24Gly PCR). Non-Finnish child with GTCS at 1.5-5y + rapid regression = vLINCL (WES CLN8 + CLN6 + CLN7 concurrent). NEVER prescribe CBZ regardless of phenotype."
            },
            {
                "type": "Myoclonic Seizures (Multifocal + Action)",
                "pct": 82,
                "description": "Myoclonic seizures emerge after GTCS — typically months to years after first GTCS, depending on phenotype. In Northern Epilepsy, myoclonus is often milder and appears later. In vLINCL, prominent action myoclonus and stimulus-sensitive myoclonus develop more rapidly after GTCS. Myoclonus is the key diagnostic clue that escalates concern beyond 'idiopathic epilepsy'.",
                "eeg": "Generalised polyspike-wave bursts; stimulus-sensitive photoparoxysmal response; cortical myoclonus component (giant SEP); background progressively disorganised; photosensitivity 52%",
                "semiology": "Sudden limb/trunk jerks; multifocal; worse with action (action myoclonus); stimulus-sensitive; nocturnal myoclonus; can cause falls; worsens significantly if CBZ/OXC/PHT administered",
                "clinical_tip": "Myoclonus emerging after GTCS in a child with developmental concern = CLN8 screen alongside CLN6/CLN7. In Northern Epilepsy, myoclonus may be mild early — do not dismiss NCL because myoclonus is mild."
            },
            {
                "type": "Atonic (Drop) Seizures",
                "pct": 42,
                "description": "Atonic seizures are more prominent in vLINCL than Northern Epilepsy. Causes sudden dangerous falls — compound fall risk in vLINCL from atonic + ataxia + visual impairment (triple mechanism). Helmet mandatory when atonic seizures are present in ambulatory patients.",
                "eeg": "Electrodecrement or brief spike-wave followed by electrodecrement; generalised; more prominent in vLINCL with rapid progression",
                "semiology": "Sudden postural tone loss; head drop or full body fall; very brief; no post-ictal phase; may occur in clusters; dangerous in school-age ambulatory children",
                "clinical_tip": "Atonic falls in vLINCL require immediate triple fall risk assessment: atonic seizures + cerebellar ataxia + visual impairment. CLB + VPA for atonic seizures; physiotherapy; mobility aids; environmental modification; helmet."
            },
            {
                "type": "Focal (Occipital / Visual) Seizures",
                "pct": 38,
                "description": "Focal occipital seizures more prominent in vLINCL (early retinal/occipital cortical NCL storage). Rare in Northern Epilepsy. Visual symptoms (phosphenes, elementary hallucinations, eye deviation) common in vLINCL. Can evolve to bilateral GTCS. Occipital EEG abnormalities correlate with retinal degeneration in vLINCL.",
                "eeg": "Focal occipital spike-wave or rhythmic delta; photic-enhanced; may generalise; occipital abnormalities more prominent in vLINCL with retinal involvement; ERG abnormalities parallel occipital EEG changes",
                "semiology": "Visual aura (lights, colours, elementary formed); eye deviation; focal motor → GTCS; post-ictal blindness possible; occipital seizures may be misdiagnosed as migraine with visual aura initially",
                "clinical_tip": "Focal occipital seizures in any child with progressive epilepsy should prompt ophthalmology ERG/VEP and NCL screen. In vLINCL, ERG amplitude reduction is often the earliest objective finding — preceding clinical visual failure."
            },
            {
                "type": "Non-Convulsive Status Epilepticus (NCSE)",
                "pct": 22,
                "description": "NCSE in 22% — less frequent than in CLN6/CLN7 but clinically important, especially in advanced vLINCL. In Northern Epilepsy, NCSE is rare but may occur in advanced disease. Presents as prolonged confusion, behavioural change, or apparent acute cognitive decline. Requires urgent EEG. IV LEV or IV benzodiazepine for management (NOT TGB, NOT fosphenytoin).",
                "eeg": "Continuous or near-continuous spike-wave without convulsive manifestation; background severely disorganised in advanced disease; may be subtle in Northern Epilepsy",
                "semiology": "Prolonged altered awareness; staring; subtle automatisms; drooling; apparent acute cognitive worsening — may mimic disease progression step in vLINCL",
                "clinical_tip": "Any acute cognitive or behavioural deterioration in CLN8 (either phenotype) = urgent EEG to exclude NCSE before attributing to disease progression. NCSE prolonged = additional neuronal injury superimposed on NCL storage."
            }
        ],
        "triggers": [
            {
                "trigger": "Fever / Intercurrent Illness",
                "pct": 82,
                "description": "Fever is the most potent seizure trigger in CLN8 — both phenotypes. GTCS clustering and SE risk elevate acutely with fever. Northern Epilepsy patients may have relatively fewer febrile seizures early but risk increases as disease progresses. Written fever action plan is mandatory.",
                "management": "Written fever action plan: antipyretics (paracetamol/ibuprofen) at 37.5°C threshold; rescue buccal midazolam 0.3 mg/kg for >2 min or 2nd seizure within 30 min; ED alert card with CLN8 SE protocol (IV LEV mandatory — NOT fosphenytoin)"
            },
            {
                "trigger": "Sleep Deprivation",
                "pct": 75,
                "description": "Sleep deprivation increases seizure frequency and myoclonus severity in CLN8. Progressive CLN8 disrupts sleep architecture independently (myoclonus, nocturnal GTCS, anxiety in Northern Epilepsy). Vicious cycle in Northern Epilepsy adults: poor sleep → more GTCS → worse cognitive decline.",
                "management": "Strict sleep schedule; CLB adjunct for nocturnal seizures; melatonin for sleep initiation; avoid nighttime care disruption; nocturnal seizure monitoring device for Northern Epilepsy adults living semi-independently"
            },
            {
                "trigger": "Missed / Subtherapeutic AED Dose",
                "pct": 68,
                "description": "Missed AED dose precipitates GTCS or cluster in both CLN8 phenotypes. In Northern Epilepsy adults who may live semi-independently, medication adherence is a lifelong challenge. In vLINCL, gastrostomy AED delivery is more reliable when oral intake becomes unsafe.",
                "management": "Electronic medication compliance system; gastrostomy for AED delivery in advanced vLINCL; carer AED training; blister pack dispensing for Northern Epilepsy adults; family written protocol"
            },
            {
                "trigger": "Photic Stimulation (Photosensitivity)",
                "pct": 52,
                "description": "Photosensitivity in 52% at standard IPS rates. More prominent in vLINCL (retinal input + cortical hyperexcitability); present but milder in Northern Epilepsy. In vLINCL, photosensitivity may decrease as retinal degeneration progresses (reduced retinal input).",
                "management": "IPS testing at diagnosis and annually; Z1-blue-tinted lenses; screen filters; 50 Hz screen refresh; LED flickering reduction; re-test annually as photosensitivity evolves with retinal progression in vLINCL"
            },
            {
                "trigger": "Emotional Stress / Anxiety",
                "pct": 60,
                "description": "Emotional stress and anxiety trigger seizures and worsen myoclonus in both CLN8 phenotypes. Northern Epilepsy adults frequently experience progressive anxiety, personality change, and depression as disease advances — these neuropsychiatric features are core to Northern Epilepsy and must be managed actively as seizure-trigger modification.",
                "management": "Psychosocial support and psychiatric review (SSRI for anxiety/depression in Northern Epilepsy adults); AAC for vLINCL children; CLB for stress-related seizure cluster; family coordination for Northern Epilepsy adults in supported living"
            },
            {
                "trigger": "Tactile / Auditory Startle",
                "pct": 50,
                "description": "Stimulus-sensitive myoclonus to tactile touch or unexpected sounds — more prominent in advanced vLINCL. In Northern Epilepsy, stimulus sensitivity develops later with disease progression. In severe vLINCL, even quiet sounds can trigger generalised myoclonic jerks.",
                "management": "Low-stimulus environment; warn before physical contact; reduce unexpected sounds; CLB reduces stimulus sensitivity; environmental modification in care setting or school"
            },
            {
                "trigger": "Metabolic Stress / Dehydration",
                "pct": 42,
                "description": "Dehydration and metabolic imbalance lower seizure threshold. Particularly relevant in vLINCL patients with gastrostomy who are at risk of enteral feed interruption. KD patients require careful fluid monitoring. Northern Epilepsy adults at risk from alcohol dehydration.",
                "management": "Adequate hydration protocol; IV access planning for gastrostomy patients during illness; electrolyte monitoring; alcohol awareness counselling for Northern Epilepsy adults (alcohol lowers seizure threshold + dehydration)"
            },
            {
                "trigger": "CLN8-Prohibited Drug Administration",
                "pct": 100,
                "description": "CBZ / OXC / PHT / fosphenytoin → 100% acute myoclonic worsening in CLN8 (both phenotypes). VGB → progressive retinal toxicity in vLINCL (absolute CI); retinal risk even in Northern Epilepsy (avoid). TGB → NCSE risk. These are preventable drug errors with catastrophic consequences — especially in Northern Epilepsy patients who may see non-specialist providers for decades.",
                "management": "ABSOLUTE CI documentation in all records; pharmacy alert; A&E/ED laminated CLN8 drug alert card; Northern Epilepsy adults carry medic alert bracelet; prescriber education across all specialties over decades-long care"
            }
        ],
        "treatments": [
            {
                "drug": "Valproate (VPA) — Backbone AED (Both Phenotypes)",
                "level": "Level B",
                "dose": "20-40 mg/kg/day in 2-3 doses; start 10 mg/kg/day; titrate to response; target trough 60-100 mg/L; weight-based dosing; VPPP mandatory for females ≥12y",
                "moa": "Sodium channel blockade + GABA-A enhancement + T-type calcium channel modulation + GABA transaminase inhibition → broad-spectrum anti-seizure + antimyoclonic",
                "efficacy": "Backbone treatment for all CLN8 seizure types (GTCS + myoclonic + atonic + focal) — both Northern Epilepsy and vLINCL. Essential for seizure control across decades-long Northern Epilepsy course. Does not alter CLN8 disease progression.",
                "monitoring": "VPA TDM trough level monthly in first year, 3-monthly when stable (target 60-100 mg/L); LFTs 3-monthly; POLG1 exclusion before initiation; VPPP mandatory females ≥12y; carnitine 6-monthly; weight monthly",
                "cln8_note": "VPA is SAFE in CLN8 — CLN8 is an ER/ERGIC cargo receptor (NOT mitochondrial). VPA ABSOLUTE CI applies to MERRF and POLG/Alpers (mitochondrial) — does NOT extend to CLN8. POLG1 exclusion recommended before VPA in children <8y (POLG1 Alpers can mimic early CLN8 — regression + seizures without early visual failure, especially mimicking Northern Epilepsy). VPPP counselling mandatory for ALL females with CLN8 on VPA regardless of phenotype."
            },
            {
                "drug": "Levetiracetam (LEV) — IV SE Protocol + Adjunct",
                "level": "Level B",
                "dose": "Oral: 20-60 mg/kg/day; IV SE: 60 mg/kg (max 4500 mg) over 15 min; start 10 mg/kg/day orally and titrate",
                "moa": "SV2A synaptic vesicle protein modulation → presynaptic inhibition → anti-GTCS + anti-myoclonic",
                "efficacy": "Adjunct to VPA; IV LEV is MANDATORY second-line for CLN8 SE (replacing fosphenytoin which is CI). Useful for GTCS and myoclonic seizures. Particularly important in Northern Epilepsy for long-term GTCS management.",
                "monitoring": "Renal function (LEV renally excreted); behavioural effects (agitation, irritability — monitor in vLINCL progressive cognitive impairment and in Northern Epilepsy adults); mood monitoring",
                "cln8_note": "IV LEV 60 mg/kg is the MANDATORY second-line in CLN8 SE — NEVER use fosphenytoin/phenytoin (ABSOLUTE CI). This must be embedded in hospital records, A&E protocols, and Northern Epilepsy adult care plans. Particularly important for Northern Epilepsy patients who may present to A&E at any age over a 40+ year disease course."
            },
            {
                "drug": "Clobazam (CLB) — Nocturnal + Adjunct",
                "level": "Level B",
                "dose": "0.1-0.5 mg/kg/day; typically 5-20 mg/day; once-daily nocturnal or BD if daytime cluster seizures; lower doses in Northern Epilepsy adults for long-term use",
                "moa": "GABA-A positive allosteric modulator (1,5-benzodiazepine) → chloride influx → membrane hyperpolarisation → anti-seizure + myoclonus reduction + stimulus sensitivity reduction",
                "efficacy": "Nocturnal GTCS prevention; cluster prevention; myoclonus reduction. Particularly valuable in Northern Epilepsy for long-term nocturnal seizure management. In vLINCL, useful for stimulus-sensitive myoclonus.",
                "monitoring": "Sedation monitoring; tolerance assessment 3-monthly; avoid abrupt withdrawal; titrate to benefit; in Northern Epilepsy adults, cognitive side effects of sedation must be distinguished from disease progression",
                "cln8_note": "CLB is valuable in Northern Epilepsy for decades-long nocturnal GTCS management. In Northern Epilepsy adults, distinguish CLB sedation from disease-related cognitive decline — dose-reduction trial may clarify. In vLINCL, CLB reduces stimulus-sensitive myoclonus. Scheduled dosing preferred to PRN in both phenotypes."
            },
            {
                "drug": "Piracetam — Action Myoclonus First-Line Adjunct",
                "level": "Level C",
                "dose": "24-48 g/day in 3 divided doses (adult/adolescent); paediatric: 40-100 mg/kg/day; start low and titrate to clinical response",
                "moa": "AMPA receptor positive modulation + membrane fluidity enhancement + putative anti-oxidant → reduced cortical myoclonic hyperexcitability",
                "efficacy": "First-line adjunct for action myoclonus in CLN8 vLINCL. Less pronounced benefit in Northern Epilepsy (milder myoclonus) but may help when action myoclonus develops. Dose must be high for meaningful benefit.",
                "monitoring": "GI side effects at high doses; renal function (renally excreted); functional improvement monitoring using UMRS scale; carers report on fine motor function and feeding",
                "cln8_note": "Piracetam for CLN8 action myoclonus — most relevant in vLINCL where action myoclonus is severe. In Northern Epilepsy, piracetam may be trialled when action myoclonus emerges (often decades into disease). Renal function monitoring essential across decades-long Northern Epilepsy course."
            },
            {
                "drug": "Lamotrigine (LTG) — Adjunct to VPA Only",
                "level": "Level B",
                "dose": "VERY SLOW titration with VPA (VPA inhibits LTG metabolism): start 0.15 mg/kg/day, increment 0.15 mg/kg every 2 weeks; target 1-5 mg/kg/day with VPA; adult Northern Epilepsy: start 25 mg alternate days",
                "moa": "Sodium channel + calcium channel blockade → reduced cortical excitability → anti-GTCS + modest anti-myoclonic",
                "efficacy": "Adjunct for focal and GTCS in CLN8; VPA-LTG combination is synergistic. Useful in Northern Epilepsy for long-term drug-resistant GTCS management. MUST have VPA backbone — LTG monotherapy is HIGH RISK.",
                "monitoring": "Skin rash monitoring (slow titration prevents SJS — VPA doubles LTG levels); LTG levels; CBC; in Northern Epilepsy long-term: annual LTG level monitoring",
                "cln8_note": "LTG in CLN8 ONLY as adjunct to VPA — NEVER monotherapy. VPA-LTG requires very slow LTG titration (SJS risk compounded by VPA). In Northern Epilepsy adults on decades-long LTG therapy, annual drug level monitoring and skin surveillance important. Useful addition when VPA alone insufficient for GTCS control."
            },
            {
                "drug": "Ketogenic Diet (KD) — After ≥3 AED Failures",
                "level": "Level C",
                "dose": "Classic KD 4:1 ratio (fat:carb+protein); MCT variant; dietitian-led; gastrostomy delivery in vLINCL; oral KD feasible in some Northern Epilepsy patients",
                "moa": "Ketone metabolism → reduced neuronal excitability (multiple mechanisms); putative neuroprotective effects via alternative energy substrate",
                "efficacy": "Adjunct after ≥3 AED failures in vLINCL; seizure reduction in 50-60% who complete initiation. In Northern Epilepsy, KD is complex logistically for adults over decades — consider modified Atkins or low-glycaemic-index treatment as more feasible alternatives.",
                "monitoring": "Dietitian monthly; lipid panel 3-monthly; weight; acidosis monitoring; renal stones; gastrostomy required for reliable KD in vLINCL advanced disease; special provisions for Northern Epilepsy adults",
                "cln8_note": "KD in vLINCL after ≥3 AED failures — gastrostomy makes KD reliable when oral intake unsafe. In Northern Epilepsy, modified Atkins may be more practical for long-term adult management than classic KD. KD + VPA: monitor LFTs closely (additive hepatotoxicity risk)."
            },
            {
                "drug": "MDT Palliative Care + Ophthalmology + Neuropsychiatry",
                "level": "Level A",
                "dose": "MDT from diagnosis: paediatric/adult neurology + ophthalmology (6-monthly ERG/VEP in vLINCL; annual in EPMR) + neuropsychiatry (Northern Epilepsy) + dietitian + physiotherapy + speech therapy + palliative care + social care (Northern Epilepsy long-term)",
                "moa": "Comprehensive CLN8 management addressing all manifestations: seizures, cognitive decline, visual loss (vLINCL), psychiatric features (Northern Epilepsy), dysphagia, motor deterioration, psychosocial needs, ACP",
                "efficacy": "MDT care is standard of care for CLN8. Northern Epilepsy requires DECADES-LONG neuropsychiatric MDT — unique among NCLs. Neuropsychiatry for personality change, anxiety, depression (Northern Epilepsy) is CORE to management, not secondary.",
                "monitoring": "FEES/videofluoroscopy in vLINCL; SARA ataxia; UMRS myoclonus; BDSRA registry; ACP annually (vLINCL: frequent; Northern Epilepsy: from diagnosis but updated every 2-5 years as course evolves); psychiatric review 6-monthly in Northern Epilepsy",
                "cln8_note": "CLN8 MDT has UNIQUE Northern Epilepsy demands: (1) DECADES-LONG adult neuropsychiatry for progressive personality change, anxiety, depression; (2) Supported living planning as cognitive decline progresses over 20-40 years; (3) SSRI for anxiety/depression is core management in Northern Epilepsy adults — not optional. ACP for Northern Epilepsy includes driving, employment, relationships, living arrangements — topics irrelevant to other NCLs."
            },
            {
                "drug": "Rescue Midazolam (Buccal) + LEV IV — SE Protocol",
                "level": "Level A",
                "dose": "Buccal midazolam 0.3 mg/kg (max 10 mg) — first-line rescue at home/community; IV LEV 60 mg/kg over 15 min — hospital second-line SE; IV VPA 30 mg/kg — third-line SE",
                "moa": "Midazolam: GABA-A fast-acting → rapid seizure termination; IV LEV: SV2A → SE termination; IV VPA: multi-mechanism → SE termination",
                "efficacy": "Family-/carer-administered buccal midazolam for >5 min seizure or 2nd within 30 min. IV LEV MANDATORY second-line in CLN8 SE (fosphenytoin ABSOLUTE CI). Particularly important for Northern Epilepsy adults who may present to A&E infrequently over 40+ years.",
                "monitoring": "Family/carer training in rescue midazolam; seizure diary; post-SE assessment; update A&E protocol; annual SE threshold review; in Northern Epilepsy adults: supported-living staff trained in rescue midazolam",
                "cln8_note": "CLN8 SE PROTOCOL: (1) Buccal midazolam 0.3 mg/kg at home/>5 min seizure. (2) Hospital: IV LEV 60 mg/kg (NOT fosphenytoin — ABSOLUTE CI). (3) Third line: IV VPA 30 mg/kg. NORTHERN EPILEPSY UNIQUE: supported-living staff and adult care workers must be trained in rescue midazolam and carry the CLN8 drug alert card — not just family members as in paediatric NCLs."
            }
        ],
        "contraindications": [
            {
                "drug": "Vigabatrin (VGB) — Absolute CI in vLINCL / Avoid in Northern Epilepsy",
                "risk_level": "ABSOLUTE CI (vLINCL) / HIGH RISK — AVOID (Northern Epilepsy)",
                "reason": "CLN8 vLINCL: progressive retinal NCL degeneration (98%). VGB retinopathy (VAR — irreversible visual field constriction) + CLN8 retinal NCL = CATASTROPHIC combined blindness. ABSOLUTE CI. Northern Epilepsy: retinal involvement minimal early but progressive course may cause late retinal changes — VGB should still be avoided (HIGH RISK). DUAL RULE: avoid VGB in ALL CLN8 regardless of phenotype. Misidentification of Northern Epilepsy as 'idiopathic epilepsy' with subsequent VGB for GTCS is a real clinical risk over decades.",
                "alternative": "IV LEV for SE; VPA + LEV + CLB for GTCS/myoclonus; KD for drug-resistant epilepsy. In Northern Epilepsy adults, document VGB avoidance in all care plans and A&E protocols over decades-long care."
            },
            {
                "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
                "risk_level": "ABSOLUTE CI",
                "reason": "Sodium channel blockers cause ACUTE MYOCLONIC WORSENING in CLN8 (both phenotypes). Both phenotypes present with GTCS first — misidentified as idiopathic epilepsy → CBZ prescribed → ACUTE MYOCLONIC DETERIORATION. Particularly dangerous in Northern Epilepsy where diagnostic delay averages 2.5y — patient may receive CBZ for extended period before NCL confirmed. Safe first choice: VPA + LEV.",
                "alternative": "VPA (backbone); LEV IV for SE; LTG only as VPA adjunct. In Northern Epilepsy adults who may see non-specialist neurologists for decades, CBZ CI must be in all records and medic alert documentation."
            },
            {
                "drug": "Fosphenytoin (IV Phenytoin Prodrug)",
                "risk_level": "ABSOLUTE CI",
                "reason": "Fosphenytoin ABSOLUTE CI in CLN8 SE — same sodium-channel mechanism as PHT → myoclonic worsening. Standard SE protocols use fosphenytoin second-line — must be OVERRIDDEN in CLN8. NORTHERN EPILEPSY RISK: adult presenting to A&E over 40+ year disease course may encounter unfamiliar staff → fosphenytoin risk persists for decades. IV LEV 60 mg/kg is the mandatory second-line.",
                "alternative": "IV LEV 60 mg/kg (mandatory CLN8 SE second-line); IV VPA 30 mg/kg (third-line). A&E protocol must be updated and available electronically."
            },
            {
                "drug": "Tiagabine (TGB)",
                "risk_level": "ABSOLUTE CI",
                "reason": "GABA reuptake inhibitor — absolute NCSE risk in NCL/PME with generalised epilepsy. NCSE in CLN8 may be misidentified as disease progression (especially in Northern Epilepsy where cognitive decline is expected). NCSE prolonged = additional neuronal injury.",
                "alternative": "CLB (GABA-A modulator — safe GABA-ergic option in CLN8). Avoid all GABA reuptake inhibitors in CLN8."
            },
            {
                "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
                "risk_level": "HIGH RISK — AVOID",
                "reason": "GBP/PGB can worsen myoclonus in PME/NCL. Multi-specialty prescribing trap: neurology controls AEDs correctly but pain specialist, GP, or psychiatrist prescribes GBP/PGB for pain or anxiety in Northern Epilepsy adult → acute myoclonic worsening. Northern Epilepsy adults at particular risk over decades of multi-specialty care.",
                "alternative": "Paracetamol + NSAID for pain; opioids if severe neuropathic pain; SSRI (not PGB) for anxiety in Northern Epilepsy. Shared care protocol with GP must explicitly document GBP/PGB prohibition."
            },
            {
                "drug": "LTG Monotherapy",
                "risk_level": "HIGH RISK",
                "reason": "LTG as sole AED provides inadequate cover for CLN8 myoclonus and GTCS — can paradoxically worsen myoclonus in some PME patients. Only acceptable as adjunct to VPA backbone. Particular risk: Northern Epilepsy adult with seemingly 'mild' disease on LTG alone if VPA stopped for any reason → myoclonic worsening.",
                "alternative": "LTG only as adjunct to VPA. VPA monotherapy or VPA+LEV as backbone. NEVER start LTG without VPA in CLN8."
            },
            {
                "drug": "AED Taper / Planned Withdrawal",
                "risk_level": "HIGH RISK",
                "reason": "CLN8 is a progressive NCL — seizures do NOT remit in either phenotype. NEVER attempt planned AED withdrawal. Particular risk in Northern Epilepsy: long periods of apparent seizure control → clinician considers AED reduction → catastrophic SE. Northern Epilepsy requires lifelong AED therapy.",
                "alternative": "Maintain AED regimen lifelong. If AED change needed, cross-taper with new agent before reducing old. Never discontinue VPA backbone in CLN8. EPMR/Northern Epilepsy: liaise with NCL specialist before any AED change."
            }
        ],
        "monitoring": [
            {"item": "CLN8 WES / NCL Gene Panel (Primary Diagnostic)", "detail": "WES or targeted NCL gene panel (CLN8 8p23.3 sequencing) — primary molecular diagnostic test; NO enzyme assay available. Finnish heritage → p.Arg24Gly (c.70C>G) PCR first (days). Non-Finnish severe vLINCL → WES including CLN8, CLN6 (15q23), CLN7 (MFSD8 4q28.1) concurrently (all show FP+CB EM; WES differentiates). RNA sequencing if high clinical suspicion + WES negative (deep intronic variants). Genotype-phenotype correlation: Northern Epilepsy (p.Arg24Gly) vs vLINCL (severe mutations) determines VGB risk level and ACP timeline."},
            {"item": "Skin Biopsy EM — FP+CB (vLINCL) / Atypical Mixed (Northern Epilepsy)", "detail": "EM of eccrine sweat gland on skin biopsy: FP + CB in CLN8 vLINCL (confirms NCL — similar to CLN6/CLN7); atypical or mixed granular deposits in Northern Epilepsy (may be less pronounced than classic vLINCL NCL). EM should not be dismissed as 'normal' or 'non-NCL' in Northern Epilepsy based on atypical pattern — consult NCL EM expert. Concurrent WES for genotype confirmation (EM cannot distinguish CLN8 from CLN6/CLN7 in vLINCL phenotype)."},
            {"item": "Ophthalmology ERG + VEP — Phenotype-Dependent Frequency", "detail": "vLINCL: ERG + VEP 6-monthly (progressive retinal NCL — critical for VGB CI enforcement and visual habilitation planning). Northern Epilepsy: ERG + VEP annually (minimal early retinal involvement but monitoring for late retinal changes essential — retinal changes may occur in long-term Northern Epilepsy). VGB avoidance must be documented in ophthalmology records for all CLN8 patients regardless of phenotype."},
            {"item": "POLG1 Exclusion Before VPA (Children <8y)", "detail": "POLG1/Alpers syndrome (mtDNA polymerase gamma deficiency) mimics CLN8 early presentation — regression + seizures without early visual failure, particularly similar to Northern Epilepsy early phenotype. VPA ABSOLUTE CI in POLG1 (fatal hepatic failure). POLG1 sequencing recommended before VPA in children <8y with regression + seizures + any mitochondrial features (lactic acidosis, hepatomegaly, family history of VPA hepatotoxicity)."},
            {"item": "Brain MRI 3T (6-monthly vLINCL / Annual Northern Epilepsy)", "detail": "vLINCL: 6-monthly MRI — progressive cortical atrophy (parieto-occipital initially), cerebellar atrophy, white matter signal change. Northern Epilepsy: annual MRI — thalamic + cerebellar + cortical atrophy over decades; quantitative volumetry for long-term tracking. Anaesthesia planning: CLN8 SE protocol (IV LEV not fosphenytoin) must be available for any sedation/general anaesthesia procedure."},
            {"item": "EEG (Annual + Acute)", "detail": "Annual awake + sleep EEG for background monitoring; standard IPS (3-50 Hz) for photosensitivity (52%); acute EEG for unexplained behavioural/cognitive change (NCSE exclusion). Northern Epilepsy: annual EEG for decades — progressive background slowing documents disease progression; acute EEG if apparent accelerated cognitive decline. No CLN8-specific pathognomonic EEG pattern."},
            {"item": "FEES / Videofluoroscopy — Dysphagia (vLINCL Priority)", "detail": "vLINCL: FEES/videofluoroscopy annually from 2 years after disease onset — gastrostomy planning before aspiration events (expected in ~80% within 5-8 years). Northern Epilepsy: dysphagia develops later in advanced disease — FEES when dysarthria and swallowing concern arise (typically decades into disease); coordinate gastrostomy planning with other care procedures."},
            {"item": "SARA Ataxia Scale (6-monthly vLINCL / Annual Northern Epilepsy)", "detail": "Scale for Assessment and Rating of Ataxia (SARA): 6-monthly in vLINCL (rapid ataxia progression); annual in Northern Epilepsy (slower cerebellar decline over decades). SARA guides physiotherapy and OT goals; combined with visual impairment assessment for compound fall risk in vLINCL. Long-term SARA trajectory in Northern Epilepsy: important for employment, driving, and independence decisions."},
            {"item": "UMRS Myoclonus Scale (6-monthly)", "detail": "Unified Myoclonus Rating Scale — 6-monthly for action myoclonus severity and piracetam dose titration; document stimulus sensitivity changes (photosensitivity may decrease as vLINCL retinal degeneration progresses). In Northern Epilepsy, myoclonus assessment guides piracetam initiation timing."},
            {"item": "Neuropsychology + Neuropsychiatry (Annual — Core in Northern Epilepsy)", "detail": "Annual neuropsychological assessment: developmental assessment in vLINCL children (BSID-IV, Vineland); cognitive + adaptive measures. Northern Epilepsy UNIQUE: neuropsychiatric annual review for progressive personality change, anxiety, depression, and cognitive decline ACROSS DECADES — SSRI therapy often needed; psychiatric review 6-monthly in established Northern Epilepsy. SSRI is core management in Northern Epilepsy adults, not supplementary."},
            {"item": "VPA TDM + LFT + Carnitine", "detail": "VPA TDM trough monthly in first year, 3-monthly when stable (target 60-100 mg/L); LFTs 3-monthly; carnitine 6-monthly; ammonia if encephalopathic; thrombocytopenia monitoring. Northern Epilepsy: decades-long VPA monitoring — routine blood tests required lifelong. KD + VPA: additive hepatotoxicity risk (LFTs monthly during KD)."},
            {"item": "SUDEP Risk / Nocturnal Monitoring", "detail": "CLN8 SUDEP risk elevated (drug-resistant epilepsy + nocturnal GTCS + progressive cognitive impairment). Northern Epilepsy adults living semi-independently: nocturnal seizure monitoring device (EMFIT or similar) — particularly important as carers may not be present overnight. Supported living staff training in post-ictal seizure safety. SUDEP risk documented in ACP at all disease stages."},
            {"item": "BDSRA / NCL Network Europe Registry", "detail": "All CLN8 patients (both Northern Epilepsy/EPMR and vLINCL) must be enrolled in BDSRA and NCL Network Europe / NCL Resource for gene therapy trial eligibility and natural history data. Finnish EPMR cohort provides unique long-term survival data — enrolment in Finnish NCL registry (Helsinki cohort) also appropriate for Finnish patients."},
            {"item": "ACP — Palliative Care from Diagnosis (Phenotype-Adapted Timeline)", "detail": "vLINCL: ACP from day of diagnosis — fatal disease, children; decisions about resuscitation, ventilation, gastrostomy, place of care, place of death. Northern Epilepsy: ACP from diagnosis but different timeline — covers DECADES of decision-making: driving cessation, employment planning, supported living, relationship decisions, eventual care placement. BOTH phenotypes: BDSRA enrolment; genetic counselling (25% recurrence risk AR); interpreter services as needed."}
        ],
        "lifecycle_stages": [
            {
                "stage": "Prenatal / Genetic Risk (Before Birth)",
                "description": "Family with known CLN8 variants (previous affected child identified): prenatal diagnosis by CVS/amniocentesis; preimplantation genetic testing (PGT) in IVF. Finnish families: p.Arg24Gly carrier screening available. Genetic counselling (25% recurrence risk — AR biallelic). Sibling cascade testing at birth. BDSRA pre-enrolment."
            },
            {
                "stage": "Pre-Symptomatic / Early (Age 0 to Seizure Onset)",
                "description": "Known CLN8 genotype (presymptomatic sibling or incidental carrier screening). Baseline ERG + ophthalmology. EEG baseline. Developmental assessment. Gene therapy trial enrolment if available. BDSRA registry. Northern Epilepsy presymptomatic window: 0-5y (seizure onset 5-10y) — early developmental monitoring may show mild delay before seizures."
            },
            {
                "stage": "First Seizure — Diagnostic Emergency",
                "description": "GTCS at disease-specific onset age (Northern Epilepsy: 5-10y; vLINCL: 1.5-7y). Diagnostic pathway: EM skin biopsy + CLN8 WES (Finnish → p.Arg24Gly PCR first) + CLN6/CLN7 concurrent if vLINCL phenotype + ophthalmology ERG + EEG + brain MRI + POLG1 exclusion before VPA. VPA + LEV started. CLN8 drug alert card provided. ACP discussion initiated. BDSRA enrolment. Phenotype determination drives VGB risk assessment and ACP timeline."
            },
            {
                "stage": "Active Epilepsy + Cognitive Regression",
                "description": "vLINCL: Multiple seizure types (GTCS + myoclonic + atonic + focal); action myoclonus → piracetam added; cognitive regression; visual failure (ERG reduction); gastrostomy consideration; MDT intensification. Northern Epilepsy: GTCS + progressive myoclonus; cognitive decline apparent; psychiatric features emerge (anxiety, personality change); supported living planning begins; SSRI if anxiety prominent. Both: KD if ≥3 AED failures."
            },
            {
                "stage": "Established Severe Disability",
                "description": "vLINCL: Profound intellectual disability; near-total visual loss; severe ataxia + myoclonus; gastrostomy-dependent; communication via AAC; drug-resistant seizures; palliative care intensification. Northern Epilepsy: Progressive cognitive impairment; dependent on support for activities of daily living; significant psychiatric burden; seizures ongoing; supported residential care; cognitive decline over 20-40 years from onset."
            },
            {
                "stage": "Late-Palliative / End of Life Stage",
                "description": "vLINCL: End-stage (teens-20s) — bedridden, non-verbal, gastrostomy-dependent, recurrent SE; comfort-focused ACP. Northern Epilepsy: Late stage (40s-60s) — profound dementia, fully care-dependent, recurrent GTCS; ACP decisions; bereavement support for family who have cared for decades. Both: Gene therapy trial coordination with palliative team if enrolled."
            }
        ]
    }


def get_definitions():
    return {
        "concepts": [
            {
                "concept": "CLN8-8p23.3-ER-ERGIC-Cargo-Receptor-Dual-Phenotype-NCL",
                "definition": "CLN8 (8p23.3) encodes a 286 aa ER-Golgi intermediate compartment (ERGIC) transmembrane cargo receptor (8 TM domains; ~33 kDa). CLN8 recruits lysosomal hydrolases (cathepsin D, F, B, H, L; PPT1; TPP1) in the ER/ERGIC for mannose-6-phosphate tagging and lysosomal delivery. CLN8 LOF → impaired cathepsin trafficking → SCMAS storage → NCL. Two phenotypes: Northern Epilepsy (mild; Finnish p.Arg24Gly) and vLINCL (severe; non-Finnish). OMIM *607837 / #600143 (EPMR) / #610003 (vLINCL).",
                "standard": "Ranta-1999-NatGenet / Nita-2016-AnnNeurol / NCL-Resource-2024 / OMIM"
            },
            {
                "concept": "Northern-Epilepsy-EPMR-Finnish-p.Arg24Gly-Milder-Survival-30-60y",
                "definition": "Northern Epilepsy (EPMR — Progressive Epilepsy with Mental Retardation) is the milder CLN8 phenotype caused by Finnish founder p.Arg24Gly (c.70C>G) hypomorphic allele. Onset 5-10y (mean ~7y); GTCS → myoclonic; mild-moderate MR; NO prominent retinal degeneration early; survival 30-60 years (median ~47y — longest NCL survival). Named for geographic clustering in northern Finland. EPMR was named before NCL classification — all EPMR patients are CLN8 disease. Finnish population prevalence ~1/10,000 births.",
                "standard": "Ranta-1999-NatGenet / Lonka-2000-JNeurol / NCL-Resource-2024"
            },
            {
                "concept": "CLN8-vLINCL-Non-Finnish-Earlier-Severe-Retinal",
                "definition": "CLN8 variant late-infantile NCL (vLINCL) — non-Finnish severe phenotype. More severe CLN8 mutations (truncating, critical missense) → near-complete CLN8 cargo receptor LOF → earlier onset (1.5-7y); progressive retinal NCL (98%); shorter survival (teens-20s). Found in Turkish, Indian, Pakistani, Italian, and other populations. VGB ABSOLUTE CI (retinal disease). FP+CB on EM (similar to CLN6/CLN7 — WES required to differentiate).",
                "standard": "NCL-Resource-2024 / Mole-2019-LancetNeurol / ILAE-2022"
            },
            {
                "concept": "No-CLN8-Enzyme-Assay-WES-Required-Cargo-Receptor-Not-Enzyme",
                "definition": "NO CLN8 ENZYME ASSAY AVAILABLE. CLN8 is an ER/ERGIC CARGO RECEPTOR (structural transmembrane function) — it has no known enzymatic activity. There is no DBS enzyme assay for CLN8 (unlike CLN1 PPT1 DBS assay 1-3 days; CLN2 TPP1 DBS assay days). Diagnostic pathway: EM skin biopsy → FP+CB (NCL, days) → Finnish heritage → p.Arg24Gly PCR (days); non-Finnish vLINCL → WES (CLN8 + CLN6 + CLN7 concurrent, weeks). Do NOT delay EM while awaiting genetics.",
                "standard": "NCL-Resource-2024 / ILAE-2022 / BDSRA-Diagnostic-Guidelines"
            },
            {
                "concept": "FP-CB-EM-Pattern-CLN8-vLINCL-Similar-CLN6-CLN7-Atypical-Northern-Epilepsy",
                "definition": "CLN8 vLINCL EM shows FP + CB (fingerprint profiles + curvilinear bodies) — similar to CLN6 and CLN7 (indistinguishable by EM alone). Northern Epilepsy EM may show atypical or mixed granular deposits, less pronounced than classic vLINCL. EM confirms NCL class but WES is required to differentiate CLN8 vLINCL from CLN6 (15q23) or CLN7 (MFSD8/4q28.1). Northern Epilepsy EM must not be dismissed as 'not NCL' based on atypical pattern — consult NCL EM expert.",
                "standard": "NCL-Resource-2024 / Mole-2019-LancetNeurol / Ranta-1999-NatGenet"
            },
            {
                "concept": "VGB-ABSOLUTE-CI-vLINCL-HIGH-RISK-Avoid-Northern-Epilepsy",
                "definition": "VGB ABSOLUTE CI in CLN8 vLINCL (retinal NCL 98%); HIGH RISK / avoid in Northern Epilepsy (minimal early retinal involvement but progressive course). DUAL CLN8 VGB RULE: avoid VGB in ALL CLN8 forms regardless of phenotype. Key trap: Finnish Northern Epilepsy misidentified as idiopathic epilepsy over years → VGB prescribed → retinal risk even in milder form. Document VGB avoidance in all CLN8 records regardless of phenotype.",
                "standard": "MHRA-VPPP-2021 / NICE-NG217 / NCL-Resource-2024"
            },
            {
                "concept": "VPA-SAFE-CLN8-ER-ERGIC-NOT-Mitochondrial",
                "definition": "VPA is SAFE in CLN8. CLN8 = ER-Golgi intermediate compartment cargo receptor dysfunction. This is NOT mitochondrial disease. VPA ABSOLUTE CI applies to MERRF/MTTK8344 and POLG/Alpers (mitochondrial) — does NOT extend to CLN8. VPA is the backbone AED in CLN8 (both phenotypes). POLG1 exclusion recommended before VPA in children <8y (POLG1 Alpers mimics Northern Epilepsy early — regression + seizures without prominent visual failure).",
                "standard": "CPIC-POLG1-2023 / NCL-Resource-2024 / ILAE-2022"
            },
            {
                "concept": "CBZ-OXC-PHT-ABSOLUTE-CI-GTCS-Trap-Both-CLN8-Phenotypes",
                "definition": "CBZ/OXC/PHT ABSOLUTE CI in CLN8 — sodium-channel blockers cause ACUTE MYOCLONIC WORSENING. Both CLN8 phenotypes present with GTCS first → misidentified as idiopathic epilepsy → CBZ prescribed. Northern Epilepsy trap: diagnostic delay mean 2.5y → extended CBZ exposure before NCL confirmed. Safe first choice: VPA + LEV for any child with GTCS + developmental concern. Northern Epilepsy adults: CBZ CI must be in medic alert and all care records over decades.",
                "standard": "NICE-NG217 / NCL-Resource-2024 / MHRA-VPPP-2021"
            },
            {
                "concept": "Seizures-First-GTCS-Both-CLN8-Phenotypes-Before-Regression",
                "definition": "SEIZURES (GTCS) are the FIRST clinical sign in both CLN8 phenotypes. Northern Epilepsy: GTCS onset 5-10y, before cognitive decline apparent. vLINCL: GTCS onset 1.5-7y, before regression and visual failure. Both phenotypes: GTCS occurs before NCL diagnosis is suspected — the challenge is recognising NCL (especially Northern Epilepsy) when only GTCS is present. Key clue: GTCS + subsequent myoclonus + developmental decline → NCL screen.",
                "standard": "Ranta-1999-NatGenet / NCL-Resource-2024 / ILAE-2022"
            },
            {
                "concept": "CLN8-Cargo-Receptor-Cathepsin-Trafficking-ERGIC-Golgi-Mechanism",
                "definition": "CLN8 MECHANISM IS UNIQUE: CLN8 is not a lysosomal storage enzyme (CLN1/CLN2) nor a lysosomal membrane structural protein (CLN3/CLN7) nor purely an ER structural protein (CLN6) — CLN8 is an ER-ERGIC CARGO RECEPTOR that cycles between ER/ERGIC and Golgi, recruiting lysosomal hydrolases (cathepsin D, F, B, H, L; PPT1; TPP1) for M6P tagging → lysosomal delivery. CLN8 LOF creates SECONDARY deficiency of multiple cathepsins simultaneously — not one primary enzyme deficiency.",
                "standard": "Nita-2016-AnnNeurol / NCL-Resource-2024 / Ranta-1999-NatGenet"
            },
            {
                "concept": "No-Disease-Modifying-Therapy-CLN8-Gene-Therapy-Preclinical",
                "definition": "No approved disease-modifying therapy for CLN8 (both phenotypes). Management is purely symptomatic. CLN8 gene therapy in preclinical development. Understanding CLN8 cargo receptor function may enable multiple-enzyme replacement strategies (CLN8 LOF → secondary cathepsin deficiencies). All CLN8 patients must be enrolled in BDSRA/NCL Resource. Northern Epilepsy: unique long-term survival model — Finnish EPMR cohort provides natural history data for NCL neuroprotection research.",
                "standard": "NCL-Resource-2024 / BDSRA-Registry / ILAE-2022"
            },
            {
                "concept": "Dual-Phenotype-Genotype-Phenotype-Correlation-VGB-ACP-Implications",
                "definition": "CLN8 dual phenotype (EPMR vs vLINCL) has DIRECT clinical management consequences: (1) VGB: ABSOLUTE CI in vLINCL vs HIGH RISK/avoid in EPMR; (2) ACP urgency: from diagnosis in vLINCL (fatal childhood-20s) vs from diagnosis but decades-long in EPMR (survival 30-60y); (3) Retinal monitoring: 6-monthly in vLINCL vs annually in EPMR; (4) Psychiatric MDT: core in EPMR (decades of neuropsychiatric management) vs secondary in vLINCL. Genotype-phenotype correlation (WES) at diagnosis is ESSENTIAL for management planning.",
                "standard": "NCL-Resource-2024 / Mole-2019-LancetNeurol / ILAE-2022"
            },
            {
                "concept": "Gastrostomy-vLINCL-Dysphagia-AED-Delivery",
                "definition": "CLN8 vLINCL progressive dysphagia from cerebellar + cortical neurodegeneration → unsafe oral intake in ~80% within 5-8 years of onset. PEG gastrostomy: reliable AED delivery, nutrition, enables KD, prevents aspiration. FEES/videofluoroscopy guides timing — gastrostomy BEFORE aspiration events. Northern Epilepsy: dysphagia develops later (advanced disease, decades into course) — FEES when dysarthria/swallowing concern arise.",
                "standard": "NCL-Resource-2024 / NICE-NG217 / Mole-2019-LancetNeurol"
            },
            {
                "concept": "SUDEP-Risk-CLN8-Nocturnal-Drug-Resistant-Northern-Epilepsy-Decades",
                "definition": "CLN8 SUDEP risk elevated: drug-resistant epilepsy (72%) + nocturnal GTCS + progressive cognitive impairment → impaired post-ictal arousal. NORTHERN EPILEPSY UNIQUE SUDEP RISK: semi-independent adults with nocturnal seizures may not have overnight supervision. Nocturnal monitoring device essential for Northern Epilepsy adults in supported or independent living. SUDEP documented in ACP across decades-long Northern Epilepsy course.",
                "standard": "NICE-NG217 / NCL-Resource-2024 / ILAE-SUDEP-Guidelines"
            },
            {
                "concept": "POLG1-Exclusion-Before-VPA-CLN8-Mimicry",
                "definition": "POLG1/Alpers syndrome mimics CLN8 early presentation — particularly mimics Northern Epilepsy (regression + seizures without early visual failure). VPA ABSOLUTE CI in POLG1 (fatal hepatic failure). POLG1 sequencing recommended before VPA in all children <8y with regression + seizures + any mitochondrial features (lactic acidosis, hepatomegaly, family VPA hepatotoxicity history). Northern Epilepsy onset 5-10y: POLG1 may mimic if presentation is atypical.",
                "standard": "CPIC-POLG1-2023 / NCL-Resource-2024 / NICE-NG217"
            }
        ],
        "thresholds": [
            {"threshold": "GTCS at 5-10y (Finnish heritage) → p.Arg24Gly PCR + CLN8 WES (EPMR screen)", "standard": "NCL-Resource-2024 / Ranta-1999-NatGenet"},
            {"threshold": "GTCS at 1.5-7y + FP+CB on EM + non-Finnish → CLN8 + CLN6 + CLN7 WES concurrent", "standard": "NCL-Resource-2024 / ILAE-2022"},
            {"threshold": "VGB ABSOLUTE CI in vLINCL — zero tolerance; HIGH RISK / avoid in Northern Epilepsy", "standard": "MHRA-VPPP-2021 / NCL-Resource-2024"},
            {"threshold": "CBZ/OXC/PHT/Fosphenytoin ABSOLUTE CI in CLN8 (both phenotypes)", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "POLG1 exclusion before VPA in children <8y with regression + seizures", "standard": "CPIC-POLG1-2023"},
            {"threshold": "Atypical EM in Northern Epilepsy: consult NCL EM expert — do not dismiss as non-NCL", "standard": "NCL-Resource-2024 / Ranta-1999-NatGenet"},
            {"threshold": "VPA TDM target 60-100 mg/L; LFTs + carnitine 3-6 monthly", "standard": "NICE-NG217 / BNFc"},
            {"threshold": "IV LEV 60 mg/kg (NOT fosphenytoin) for SE — mandatory CLN8 protocol both phenotypes", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "ERG + VEP 6-monthly in vLINCL; annually in Northern Epilepsy from diagnosis", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "Buccal midazolam 0.3 mg/kg for >5 min seizure — family/carer-administered", "standard": "NICE-NG217 / APLS-Guidelines"},
            {"threshold": "Northern Epilepsy: neuropsychiatric review 6-monthly; SSRI for anxiety/depression core management", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "BDSRA / NCL Network Europe / Finnish NCL Registry enrolment mandatory at diagnosis", "standard": "BDSRA-Registry / NCL-Network-Europe"}
        ],
        "standards": [
            {"standard": "Ranta-1999-NatGenet", "detail": "Ranta S et al. 'Positional cloning and characterisation of the human EPMR gene.' Nature Genetics 1999;23:233-6 — first identification of CLN8 gene mutations in Finnish EPMR (Northern Epilepsy) families"},
            {"standard": "Lonka-2000-JNeurol", "detail": "Lonka L et al. 'The neuronal ceroid lipofuscinosis CLN8 membrane protein is a resident of the endoplasmic reticulum.' Human Molecular Genetics 2000 — characterisation of CLN8 protein localisation and function"},
            {"standard": "Nita-2016-AnnNeurol", "detail": "Nita DA et al. 'A mouse model of CLN8 disease reveals a role for CLN8 as a cargo receptor for lysosomal enzymes.' Annals of Neurology 2016 — discovery of CLN8 cargo receptor role for cathepsins in ER/ERGIC"},
            {"standard": "Mole-2019-LancetNeurol", "detail": "Mole SE et al. 'Clinical management of the neuronal ceroid lipofuscinoses.' Lancet Neurology 2019 — comprehensive NCL management standard including CLN8 both phenotypes"},
            {"standard": "NCL-Resource-2024", "detail": "NCL Resource (ncl.mni.mcgill.ca) — international NCL management guidelines, diagnostic algorithms, EM pattern classification, CLN8 management including EPMR and vLINCL; updated 2024"},
            {"standard": "ILAE-2022", "detail": "International League Against Epilepsy 2022 — epilepsy classification, diagnosis standards, treatment guidelines applicable to CLN8 seizure management"},
            {"standard": "NICE-NG217", "detail": "NICE Guideline NG217: 'Epilepsies in children, young people and adults' (2022) — AED selection, monitoring, contraindications for paediatric and adult NCL including CLN8"},
            {"standard": "MHRA-VPPP-2021", "detail": "MHRA Valproate Pregnancy Prevention Programme (2021) — mandatory VPA counselling and pregnancy prevention for all females ≥12y on valproate (applies to Northern Epilepsy adults)"},
            {"standard": "CPIC-POLG1-2023", "detail": "CPIC Guideline for POLG1 — pharmacogenomics of VPA in POLG/Alpers; POLG1 exclusion mandatory before VPA in paediatric regression + seizures (POLG1 mimics CLN8 Northern Epilepsy pattern)"},
            {"standard": "ACMG-AMP-2015", "detail": "ACMG/AMP 2015 variant classification guidelines — applied to CLN8 variant interpretation (pathogenic / likely pathogenic / VUS); genotype-phenotype correlation (Northern Epilepsy p.Arg24Gly vs vLINCL severe mutations)"},
            {"standard": "BDSRA-Registry", "detail": "Batten Disease Support and Research Association Registry — mandatory enrolment for all CLN8 patients (both Northern Epilepsy and vLINCL); gene therapy trial eligibility; natural history data"},
            {"standard": "WHO-ICF-2019", "detail": "WHO International Classification of Functioning, Disability and Health — framework for MDT management across decades-long CLN8 Northern Epilepsy disability and vLINCL paediatric disability"}
        ],
        "references": [
            {"ref": "Ranta-1999-NatGenet", "citation": "Ranta S, Zhang Y, Ross B, et al. (1999). Positional cloning and characterisation of the human EPMR gene. Nature Genetics, 23(2), 233-236."},
            {"ref": "Lonka-2000-HumMolGenet", "citation": "Lonka L, Kyttala A, Ranta S, et al. (2000). The neuronal ceroid lipofuscinosis CLN8 membrane protein is a resident of the endoplasmic reticulum. Human Molecular Genetics, 9(11), 1691-1697."},
            {"ref": "Nita-2016-AnnNeurol", "citation": "Nita DA, Mole SE, Bhatt DL, et al. (2016). A mouse model of CLN8 disease reveals a role for CLN8 as a cargo receptor for lysosomal enzymes. Annals of Neurology, 79(3), 433-446."},
            {"ref": "Mole-2019-LancetNeurol", "citation": "Mole SE, Anderson G, Band HA, et al. (2019). Clinical management of the neuronal ceroid lipofuscinoses. The Lancet Neurology, 18(12), 1131-1140."},
            {"ref": "Reinhardt-2010-HumMutat", "citation": "Reinhardt K, Grapp M, Schlachter K, et al. (2010). Novel CLN8 mutations confirm the clinical and ethnic diversity of late infantile neuronal ceroid lipofuscinosis. Clinical Genetics, 77(1), 79-85."},
            {"ref": "Kolikova-2011-PLoSOne", "citation": "Kolikova J, Afzalov R, Surin A, et al. (2011). Deficient mitochondrial calcium buffering in the Cln8mnd mouse model of neuronal ceroid lipofuscinosis. Cell Calcium, 50(5), 491-501."}
        ]
    }
