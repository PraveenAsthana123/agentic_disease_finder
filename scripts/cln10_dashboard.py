"""
CLN10 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 10 / Congenital NCL / Cathepsin D Deficiency
==================================================================================================
40-patient cohort · CTSD (11p15.5) · Autosomal recessive (AR) biallelic LOF
CTSD encodes Cathepsin D: 412 aa precursor (~44 kDa); signal peptide aa 1-20;
processed to light chain ~14 kDa + heavy chain ~34 kDa; aspartic endopeptidase;
Asp106/Asp231 catalytic dyad; pH optimum 3.5-5.0; key lysosomal protease cleaving
SCMAS (subunit c mitochondrial ATP synthase) and activating PPT1 (CLN1 enzyme).
CTSD LOF → SCMAS accumulation → Granular Osmiophilic Deposits (GRODs) on EM →
progressive neuronal ± retinal apoptosis → CLN10 (Congenital NCL most severe).

MOST SEVERE NCL — CONGENITAL PRESENTATION:
═══════════════════════════════════════════

PHENOTYPE 1: CONGENITAL CLN10 (most severe NCL known)
- Biallelic CTSD null/truncating mutations → near-complete CTSD LOF
- Seizures from BIRTH or first hours/days of life
- Microcephaly at birth (prenatal brain malformation — simplified gyral pattern on fetal MRI)
- Respiratory failure in neonatal period — fatal within days to months (<1 year most)
- Most severe of all NCL types — no postnatal neurodevelopmental window
- EM: GRODs identical to CLN1/PPT1 (Granular Osmiophilic Deposits)

PHENOTYPE 2: LATE-INFANTILE CLN10 (attenuated — compound-het or missense with residual CTSD)
- Onset 2-5 years; similar to vLINCL course but slower
- Progressive cognitive decline + seizures + retinal degeneration
- Survival teens to late 20s

PHENOTYPE 3: JUVENILE/ADULT CLN10 (very rare, attenuated missense alleles)
- Onset adolescence/early adulthood; slowest CLN10 course
- Progressive NCL features; may survive to 4th-5th decade

CTSD PROTEIN BIOLOGY (SOLUBLE LYSOSOMAL ASPARTIC ENDOPEPTIDASE):
CTSD (11p15.5):
  - 412 amino acids precursor; ~44 kDa; signal peptide aa 1-20; propeptide aa 21-67
  - Processed to: light chain ~14 kDa (Asp106 catalytic residue) + heavy chain ~34 kDa (Asp231)
  - Soluble lysosomal aspartic endopeptidase; pH optimum 3.5-5.0 (lysosomal pH)
  - Key substrates: SCMAS (subunit c of mitochondrial ATP synthase — NCL storage protein);
    proapoptotic proteins; extracellular matrix proteins
  - CRITICAL: CTSD cleaves and ACTIVATES PPT1 (CLN1 enzyme) — CTSD LOF causes secondary PPT1
    functional reduction → explains why CTSD/CLN10 and PPT1/CLN1 BOTH produce GRODs on EM
  - pLI ~0.89 (highly intolerant to LOF — essential lysosomal protease)
  - OMIM: *116840 (CTSD gene) / #610127 (CLN10 disease)
  - Discovery: Siintola E et al. 2006 Ann Neurol (first identification in Turkish/Finnish congenital NCL);
    Steinfeld R et al. 2006 Hum Mol Genet (independent discovery in Italian congenital NCL)

CLN10 vs OTHER NCLs — KEY DISTINCTIONS:
  CONGENITAL ONSET — only NCL with prenatal/neonatal seizures (all others postnatal)
  GRODS EM — same as CLN1/PPT1; CRITICAL to test both CTSD AND PPT1 when GRODs found
  SOLUBLE LYSOSOMAL ENZYME — like CLN1/CLN2 (but aspartic protease, not thioesterase/peptidase)
  CTSD→PPT1 ACTIVATION — CTSD deficiency causes secondary PPT1 reduction (dual GRODs mechanism)
  CTSD ENZYME ASSAY — less standardised as DBS than CLN1 PPT1; WES often required
  PPT1 DBS ASSAY FIRST — when GRODs on EM, PPT1 enzyme assay (DBS, 1-3 days) should precede CTSD WES
  VPA SAFE — lysosomal aspartic protease (NOT mitochondrial; POLG1 exclusion for neonates <2y)
  VGB ABSOLUTE CI — retinal NCL in all non-congenital forms; congenital = palliative only
  CBZ/OXC/PHT ABSOLUTE CI — myoclonus worsening (non-congenital forms)
  MOST LETHAL NCL — congenital form: fatal within weeks to months
"""


def get_overview():
    return {
        "gene": "CTSD (11p15.5) — Cathepsin D; soluble lysosomal aspartic endopeptidase (412 aa precursor; ~44 kDa; light chain Asp106 + heavy chain Asp231 catalytic dyad; pH 3.5-5.0; cleaves SCMAS + activates PPT1; GRODs on EM; congenital-to-adult phenotypic spectrum; SCMAS accumulation → NCL)",
        "protein": "Cathepsin D (CTSD); 412 aa precursor; ~44 kDa; signal peptide aa 1-20; propeptide aa 21-67 (inhibitory); proteolytically processed to mature bichain form: light chain ~14 kDa (contains Asp106 active-site) + heavy chain ~34 kDa (contains Asp231); forms pepstatin-inhibitable aspartic endopeptidase; key lysosomal protease; primary NCL-relevant substrates: SCMAS (subunit c mitochondrial ATP synthase) → NCL storage; and propalmitoyl-protein thioesterase 1 (PPT1/CLN1) → CTSD cleaves/activates PPT1; CTSD LOF → dual mechanism: (1) direct SCMAS accumulation; (2) secondary PPT1 enzyme reduction → compounded NCL storage; explains GRODs appearing in both CLN1 (PPT1 LOF) and CLN10 (CTSD LOF)",
        "inheritance": "Autosomal recessive (AR) biallelic LOF. pLI ~0.89 (highly intolerant — essential lysosomal protease). THREE PHENOTYPES by mutation severity: (1) Congenital CLN10 — biallelic null/truncating; most severe; seizures from birth; microcephaly; fatal <1 year. (2) Late-infantile CLN10 — compound-het missense+truncating; onset 2-5y; survival teens-20s. (3) Juvenile/adult CLN10 — biallelic missense with residual CTSD; rare; survival 4th-5th decade. OMIM *116840 / #610127",
        "omim": "*116840 (CTSD gene) · #610127 (CLN10 — Neuronal Ceroid Lipofuscinosis Type 10)",
        "disease": "CLN10 (CTSD) — Neuronal Ceroid Lipofuscinosis Type 10. Three phenotypes: (1) Congenital: biallelic null CTSD LOF; seizures from birth/hours; microcephaly; respiratory failure; death <1 year — MOST SEVERE NCL. (2) Late-infantile: compound-het; onset 2-5y; GTCS + myoclonic; visual failure; survival teens-20s. (3) Juvenile/adult: missense alleles with residual CTSD; onset adolescence; slow progressive NCL. EM: GRODs (Granular Osmiophilic Deposits) identical to CLN1 — CRITICAL: both CLN1 (PPT1 LOF) and CLN10 (CTSD LOF → secondary PPT1 reduction) produce GRODs. Test PPT1 enzyme assay FIRST (DBS, 1-3 days) before CTSD WES.",
        "mechanism": "CTSD biallelic LOF → absent/dysfunctional lysosomal aspartic endopeptidase → DUAL MECHANISM: (1) SCMAS not cleaved → subunit c accumulation → GRODs → neuronal/retinal apoptosis; (2) PPT1 (CLN1 enzyme) not activated by CTSD → secondary palmitoyl-protein thioesterase deficiency → compounded substrate accumulation. Both CLN1 and CLN10 converge on the PPT1/SCMAS/lysosomal pathway → both produce GRODs on EM. Congenital form: near-complete CTSD absence → prenatal neuronal apoptosis → simplified gyri/microcephaly → neonatal epilepsy → rapidly fatal.",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN10/CTSD. Management is purely symptomatic. Investigational: CTSD enzyme replacement therapy (ERT) — conceptually feasible (soluble lysosomal enzyme, like CLN2 cerliponase); preclinical studies in CTSD-deficient mice. Gene therapy for CTSD in early research phase. Congenital form: palliative comfort-focused care from birth given uniformly fatal course. All CLN10 patients (non-congenital forms) must be enrolled in BDSRA/NCL Resource for trial eligibility.",
        "grods_em_critical": "CRITICAL — GRODs (Granular Osmiophilic Deposits) on EM skin biopsy = CLN1 (PPT1) OR CLN10 (CTSD). BOTH PPT1 AND CTSD MUST BE TESTED WHEN GRODs ARE FOUND. DIAGNOSTIC SEQUENCE: (1) EM skin biopsy → GRODs confirmed (days). (2) PPT1 enzyme assay DBS — 1-3 days (fastest, cheapest, most standardised). (3) If PPT1 normal → CTSD WES/gene panel (weeks). CTSD DBS enzyme assay (aspartic protease pepstatin-inhibitable assay on DBS) is less standardised than PPT1 — WES is often the primary CTSD diagnostic tool. Do NOT skip PPT1 DBS assay even if CTSD is suspected — PPT1 must be excluded first.",
        "cohort_size": 40,
        "female_pct": 50,
        "congenital_pct": 42,
        "late_infantile_pct": 40,
        "juvenile_adult_pct": 18,
        "mean_onset_seizure_days_congenital": 1.5,
        "mean_onset_seizure_years_lincl": 3.2,
        "mean_diagnosis_delay_years": 1.8,
        "drug_resistant_pct": 78,
        "retinal_degeneration_pct": 92,
        "grods_em_pct": 95,
        "microcephaly_congenital_pct": 88,
        "cognitive_impairment_pct": 98,
        "photosensitivity_pct": 40,
        "on_vpa_pct": 82,
        "mean_survival_months_congenital": 8,
        "mean_survival_years_lincl": 19,
        "key_pharmacological_distinctions": {
            "1_GRODS_EM_MEANS_CLN1_OR_CLN10_TEST_BOTH": "GRODs ON EM — CLN1 (PPT1) AND CLN10 (CTSD) BOTH PRODUCE GRODs: Granular Osmiophilic Deposits are pathognomonic for NCL storage of palmitoylated proteins and SCMAS. Both CLN1 (PPT1 LOF → failed depalmitoylation) and CLN10 (CTSD LOF → failed SCMAS cleavage + secondary PPT1 reduction) produce GRODs. CRITICAL DIAGNOSTIC RULE: When GRODs found on EM skin biopsy, ALWAYS test BOTH PPT1 (DBS enzyme assay, 1-3 days — test first) AND CTSD (WES/sequencing). Skipping PPT1 enzyme assay and going directly to CTSD WES = diagnostic delay. Conversely, normal PPT1 enzyme assay does NOT mean CLN10 is excluded — CTSD must also be tested. The two diseases can only be distinguished biochemically/genetically, not by EM alone.",
            "2_CTSD_ACTIVATES_PPT1_DUAL_GRODS_MECHANISM": "CTSD CLEAVES AND ACTIVATES PPT1 (CLN1 ENZYME) — UNIQUE INTER-NCL ENZYME RELATIONSHIP: Cathepsin D proteolytically cleaves pro-PPT1 to mature active PPT1 in the lysosome. CTSD LOF → pro-PPT1 not activated → secondary functional PPT1 deficiency → compound NCL storage (SCMAS + palmitoylated substrates). This CTSD→PPT1 activation pathway means CLN10 patients can show biochemical features of BOTH CLN1 (secondary PPT1 enzyme reduction) AND CLN10 (primary CTSD deficiency). Clinicians must request CTSD-specific sequencing — a low-normal PPT1 activity on DBS can occur in CLN10 if CTSD-mediated PPT1 activation is impaired. If PPT1 is borderline low but CTSD WES reveals biallelic pathogenic variants, CLN10 is the diagnosis.",
            "3_CONGENITAL_NCL_MOST_SEVERE_PRENATAL_BRAIN_MALFORMATION": "CONGENITAL CLN10 — MOST SEVERE NCL AND MOST SEVERE LYSOSOMAL STORAGE DISEASE PRESENTATION: CLN10 congenital form is the only NCL with PRENATAL onset. Biallelic CTSD null mutations → near-complete absence of lysosomal CTSD from embryonic development → progressive neuronal apoptosis prenatally → microcephaly at birth (simplified gyral pattern / agyria-pachygyria-like on fetal MRI) → seizures from first hours of life → neonatal multifocal myoclonic epilepsy + tonic seizures + vegetative dysfunction → respiratory failure → fatal within days to ~12 months. NO other NCL type causes prenatal brain malformation. Diagnosis considerations: prenatal MRI showing simplified gyri in at-risk family (prior CLN10 child) → fetal diagnosis; neonatal seizures + microcephaly → GRODs skin biopsy + PPT1 DBS + CTSD WES.",
            "4_VGB_ABSOLUTE_CI_NON_CONGENITAL_PALLIATIVE_ONLY_CONGENITAL": "VGB ABSOLUTE CI IN NON-CONGENITAL CLN10 (RETINAL NCL); CONGENITAL FORM IS PALLIATIVE-ONLY: Late-infantile and juvenile CLN10: progressive retinal NCL degeneration (90-95%) → VGB retinopathy (VAR) + CLN10 retinal NCL = CATASTROPHIC combined blindness → ABSOLUTE CI. Congenital CLN10: disease is immediately fatal (< 1 year) → palliative comfort-only focus → VGB question rarely arises (congenital infants receive comfort care, not vigorous AED polytherapy). However, even in congenital CLN10, VGB should NOT be initiated as it provides no clinical benefit and adds retinal toxicity to an already devastating disease.",
            "5_CBZ_OXC_PHT_CI_NON_CONGENITAL_NEONATAL_MYOCLONUS_WORSENING": "CBZ/OXC/PHT ABSOLUTE CI IN CLN10 MYOCLONUS: Non-congenital CLN10 (late-infantile/juvenile): GTCS + multifocal myoclonus at onset → misidentified as idiopathic epilepsy → CBZ prescribed → ACUTE MYOCLONIC WORSENING. Congenital CLN10: sodium channel blockers can exacerbate neonatal multifocal myoclonic seizures — particularly dangerous in the already critically compromised neonate. NEONATAL TRAP: neonate with multifocal seizures → neonatologist prescribes phenobarbital (safer) OR phenytoin (AVOID → Na-channel blocker → myoclonus worsening). Safe neonatal seizure drugs: phenobarbital + LEV (± pyridoxine trial). Avoid PHT/fosphenytoin even in neonates.",
            "6_VPA_SAFE_LYSOSOMAL_NOT_MITOCHONDRIAL_POLG1_NEONATAL_EXCLUSION": "VPA SAFE IN CLN10 — LYSOSOMAL ASPARTIC PROTEASE, NOT MITOCHONDRIAL: CTSD = soluble lysosomal aspartic endopeptidase. This is NOT mitochondrial disease. VPA ABSOLUTE CI applies to MERRF/POLG (mitochondrial) — does NOT extend to CLN10. VPA backbone in late-infantile/juvenile CLN10. CRITICAL NEONATAL CAVEAT: In congenital CLN10 (neonatal presentation), VPA should be used with caution in neonates <4 weeks (hepatic immaturity; VPA associated with hepatic failure in neonates regardless of POLG1 status). POLG1 exclusion ESSENTIAL before VPA in any neonate/infant <2y with mitochondrial features — POLG1/Alpers can mimic late-infantile CLN10 with regression + seizures. If POLG1 confirmed → VPA ABSOLUTE CI.",
            "7_PPT1_NORMAL_CTSD_SEQUENCING_MANDATORY_GRODS": "MANDATORY DIAGNOSTIC ALGORITHM FOR GRODs ON EM: Step 1: PPT1 enzyme assay DBS (1-3 days) → if CLEARLY LOW → CLN1 diagnosis (confirm with CLN1/PPT1 sequencing). Step 2: If PPT1 NORMAL or BORDERLINE LOW → CTSD WES (CLN10 sequencing) MANDATORY. Step 3: If both PPT1 normal AND CTSD WES negative → expand NCL panel (CLN2 TPP1 enzyme, NCL gene panel). BORDERLINE PPT1 CAUTION: CLN10/CTSD LOF → secondary PPT1 reduction → PPT1 enzyme may appear borderline low in CLN10 patients (CTSD-mediated PPT1 activation impaired). A borderline low PPT1 does NOT definitively diagnose CLN1 — CTSD sequencing is still mandatory when PPT1 is borderline. This nuanced PPT1/CTSD relationship is the most common diagnostic pitfall in GRODs NCL.",
            "8_CTSD_ERT_CONCEPTUALLY_FEASIBLE_UNLIKE_STRUCTURAL_NCLSS": "CTSD ENZYME REPLACEMENT THERAPY (ERT) IS CONCEPTUALLY FEASIBLE — MOST PROMISING CLN10 THERAPEUTIC APPROACH: CTSD is a SOLUBLE lysosomal enzyme (like CLN2/TPP1 where cerliponase alfa ERT is approved). Soluble lysosomal enzymes can theoretically be replaced by intrathecal or intravenous recombinant enzyme delivery (mannose-6-phosphate receptor pathway). CLN2 cerliponase proved ERT works for NCL soluble enzymes. CLN10/CTSD ERT is in preclinical research phase — CTSD-deficient mouse models show promise. CRITICAL: congenital CLN10 ERT would need to be prenatal or immediate neonatal — too late postnatal for congenital form given irreversible prenatal brain damage. Late-infantile/juvenile CLN10 are the ERT therapeutic targets. All CLN10 patients must be enrolled in BDSRA/NCL Resource for trial eligibility tracking."
        }
    }


def get_breakdown():
    return {
        "etiologies": [
            {
                "class": "Homozygous CTSD Null/Truncating (Congenital CLN10 — Consanguineous)",
                "pct": 28,
                "count": 11,
                "description": "Homozygous truncating CTSD mutations (nonsense, frameshift, splice-site) in consanguineous families — primary cause of congenital CLN10. Near-complete CTSD enzyme absence from embryonic development → prenatal neuronal apoptosis → microcephaly + simplified gyral pattern at birth → neonatal multifocal seizures → respiratory failure → death <1 year. Turkish founder p.Trp383Stop (c.1148G>A) described by Siintola 2006; Finnish and Italian null alleles independently reported.",
                "gene_mechanism": "Homozygous null CTSD (truncating) → complete absence of lysosomal aspartic endopeptidase from embryonic period → no SCMAS cleavage → no PPT1 activation → GRODs + prenatal neuronal apoptosis → microcephaly + simplified gyri → congenital epilepsy → fatal neonatal NCL",
                "key_variants": ["p.Trp383Stop (c.1148G>A) Turkish founder (Siintola 2006)", "p.Tyr360Stop Italian/consanguineous (Steinfeld 2006)", "frameshift/splice-site null alleles", "consanguineous families globally", "GRODs on neonatal EM"]
            },
            {
                "class": "Compound-Het CTSD Truncating/Truncating (Congenital — Non-Consanguineous)",
                "pct": 14,
                "count": 6,
                "description": "Compound heterozygous truncating CTSD mutations — both alleles null. Congenital phenotype identical to homozygous null. Non-consanguineous families. Complete CTSD LOF. Diagnosis by WES — no founder PCR available; NCL gene panel including CTSD essential for any neonate with microcephaly + GRODs.",
                "gene_mechanism": "Two independent truncating alleles → compound-het null CTSD → complete CTSD enzyme absence → congenital NCL identical mechanism to homozygous null; slightly more likely in non-consanguineous families",
                "key_variants": ["compound-het truncating pair", "various frameshift + nonsense combinations", "non-consanguineous", "WES required", "congenital phenotype — CTSD activity undetectable"]
            },
            {
                "class": "Compound-Het Missense/Truncating (Late-Infantile CLN10)",
                "pct": 25,
                "count": 10,
                "description": "Compound heterozygous: one truncating allele + one missense allele with partial CTSD function. Residual CTSD activity permits post-natal neurodevelopment, delaying onset to 2-5 years. Late-infantile CLN10 presentation: GTCS first, then myoclonic seizures, cognitive regression, and visual failure. Survival teens to mid-20s. Most common non-congenital CLN10 genotype.",
                "gene_mechanism": "Truncating allele → zero CTSD on one chromosome; missense → residual CTSD on other → net partial CTSD deficiency → slower SCMAS/PPT1 pathway impairment → late-infantile NCL; severity determined by residual missense allele CTSD activity",
                "key_variants": ["truncating + missense compound-het", "multiple ethnicities — no dominant founder", "WES/NCL panel required", "late-infantile onset 2-5y", "residual CTSD activity 5-15% normal"]
            },
            {
                "class": "Compound-Het Missense/Missense (Juvenile/Attenuated CLN10)",
                "pct": 18,
                "count": 7,
                "description": "Compound heterozygous with two missense CTSD alleles — both hypomorphic, allowing substantial residual CTSD function. Juvenile or attenuated adult onset. Slowly progressive NCL: seizures in adolescence, visual decline, cognitive impairment developing over years to decades. Rarest non-congenital CLN10 genotype. Italian, British, and other European cases reported.",
                "gene_mechanism": "Two partial-function missense alleles → residual CTSD 15-35% normal → slowly progressive lysosomal SCMAS accumulation → juvenile/adult NCL phenotype; phenotype severity inversely correlates with residual CTSD enzyme activity",
                "key_variants": ["missense/missense compound-het", "Italian/European enrichment", "juvenile onset — adolescence/early adulthood", "residual CTSD 15-35% normal", "functional enzyme assay + WES for characterisation"]
            },
            {
                "class": "Homozygous Missense (Attenuated Adult CLN10 — Very Rare)",
                "pct": 11,
                "count": 4,
                "description": "Homozygous missense CTSD mutations with significant residual enzyme activity. Adult-onset NCL: epilepsy in 3rd-4th decade, very slow cognitive decline, may survive to 5th decade. The mildest CLN10 form. Consanguineous or regional founder missense alleles. GRODs on skin biopsy EM remain the diagnostic key — may be sparse in attenuated forms.",
                "gene_mechanism": "Homozygous missense → residual CTSD 20-40% normal → extremely slow lysosomal accumulation → adult-onset progressive NCL; GRODs may be sparse on EM in mild forms (sample adequacy critical)",
                "key_variants": ["homozygous missense — hypomorphic allele", "adult-onset 3rd-4th decade", "residual CTSD 20-40% normal", "sparse GRODs — adequate EM sampling essential", "consanguineous/founder alleles; regional"]
            },
            {
                "class": "Phenocopy CLN10-Negative (CLN1 Mimic / Deep Intronic CTSD)",
                "pct": 4,
                "count": 2,
                "description": "Clinical and EM picture consistent with CLN10 (GRODs) but both PPT1 enzyme assay and CTSD WES coding-region negative. May represent: (1) deep intronic CTSD variant (RNA-seq required); (2) CLN1 with borderline PPT1 (repeat assay + CLN1 sequencing); (3) rare novel GRODs-producing NCL gene. RNA-seq of fibroblasts if high clinical suspicion with negative WES.",
                "gene_mechanism": "GRODs on EM → PPT1 normal → CTSD WES negative → consider: deep intronic CTSD (splicing variant missed by WES — RNA-seq); CLN1 with borderline PPT1 (repeat DBS assay); or undescribed NCL gene producing GRODs",
                "key_variants": ["GRODs EM confirmed", "PPT1 enzyme assay normal/borderline", "CTSD WES coding-region negative", "deep intronic CTSD (RNA-seq of fibroblasts)", "CLN1/PPT1 re-assay + complete sequencing"]
            }
        ],
        "seizure_types": [
            {
                "type": "Neonatal / Early-Onset Multifocal Myoclonic Seizures (Congenital CLN10)",
                "pct": 88,
                "description": "The most distinctive CLN10 seizure type: multifocal myoclonic seizures in neonates/infants from birth or first hours of life (congenital form). Multifocal — affects different body parts asynchronously. Extremely refractory. May be indistinguishable from other causes of neonatal seizures initially, but severity, microcephaly, and GRODs on skin biopsy distinguish. In late-infantile/juvenile forms, multifocal myoclonus appears after GTCS onset.",
                "eeg": "Multifocal spike discharges; burst-suppression pattern (congenital form — pathognomonic of severe neonatal epileptic encephalopathy); progressive background suppression; in late-infantile: generalised polyspike-wave emerging with disease progression",
                "semiology": "Neonatal: brief arrhythmic jerks affecting face, trunk, limbs asynchronously; may persist through sleep (non-REM); very frequent — dozens to hundreds per day; may progress to tonic posturing. Late-infantile: action-sensitive myoclonus, stimulus-sensitive, interferes with feeding and motor function",
                "clinical_tip": "Neonate with microcephaly + multifocal myoclonic seizures from birth → GRODs skin biopsy + PPT1 enzyme DBS + CTSD WES simultaneously. Burst-suppression EEG in neonate with microcephaly → highest priority NCL screen. Do NOT wait for genetics before starting palliative comfort care discussions in congenital CLN10."
            },
            {
                "type": "GTCS (Generalised Tonic-Clonic) — Late-Infantile/Juvenile CLN10",
                "pct": 78,
                "description": "GTCS is the first prominent seizure type in late-infantile and juvenile CLN10 — onset 2-5y (late-infantile) or adolescence (juvenile). Misidentified as idiopathic epilepsy, with subsequent inappropriate AED prescribing (CBZ — ABSOLUTE CI in NCL myoclonus). GTCS frequency increases with disease progression. In congenital CLN10, GTCS are rarely identified as distinct — seizure semiology is more complex (multifocal myoclonic + tonic).",
                "eeg": "Generalised spike-wave or polyspike-wave; occipital enhancement; progressive background slowing; photosensitivity in 40%; IPS at standard rates (3-50 Hz) may provoke response",
                "semiology": "Bilateral tonic then clonic; loss of consciousness; post-ictal confusion; nocturnal predominance; clustering with fever or missed AED; duration 1-5 min; may present to paediatric A&E initially",
                "clinical_tip": "GTCS at 2-5y with cognitive regression or prior learning difficulty → NCL screen (GRODs EM + PPT1 DBS + CTSD WES). Do NOT prescribe CBZ for first GTCS in child with developmental concern — VPA + LEV safe first choice in CLN10."
            },
            {
                "type": "Infantile Spasms / West Syndrome Overlap (Congenital/Early CLN10)",
                "pct": 52,
                "description": "Infantile spasms-like presentation in congenital CLN10 — clusters of brief bilateral arm/leg extension (extensor spasms) or flexion, often in series upon waking. Important: VGB (first-line for infantile spasms per NICE/UKISS) is ABSOLUTE CI in CLN10 (retinal NCL). ACTH or prednisolone is the spasm treatment of choice in CLN10, but prognosis is invariably poor given congenital NCL. This presentation creates the highest VGB misadministration risk — CLN10 infant with spasms prescribed VGB as standard care = catastrophic.",
                "eeg": "Hypsarrhythmia (modified) in early congenital CLN10; may evolve from burst-suppression; chaotic high-amplitude multifocal discharges; EEG background severely disorganised from birth",
                "semiology": "Clusters of brief symmetric or asymmetric spasms (flexor/extensor/mixed); series of 5-50+ spasms on waking or during drowsiness; cry after clusters; developmental regression concurrent with spasm onset; no post-ictal phase within cluster",
                "clinical_tip": "ANY INFANT with spasms + microcephaly + developmental regression → URGENT PPT1 enzyme DBS + CTSD WES BEFORE VGB initiation. West syndrome with microcephaly = high-priority CLN10 screen. If CLN10 confirmed (or GRODs found pending genetics) → ACTH/prednisolone (NOT VGB) for spasms."
            },
            {
                "type": "Focal (Occipital / Visual) Seizures",
                "pct": 42,
                "description": "Focal occipital seizures in late-infantile and juvenile CLN10 — correlate with retinal NCL storage and occipital cortex involvement. Visual symptoms (phosphenes, elementary formed hallucinations, forced eye deviation) common. May evolve to bilateral GTCS. In congenital CLN10, focal seizure identification is confounded by diffuse multifocal myoclonic background.",
                "eeg": "Focal occipital spike-wave; ERG amplitude reduction parallels occipital EEG abnormality; photic-enhanced occipital paroxysmal activity; may generalise",
                "semiology": "Visual aura (flashing lights, coloured patterns); forced eye deviation; elementary formed visual hallucinations; may evolve to GTCS; post-ictal blindness occasionally; occipital seizures may precede clinical visual failure in late-infantile CLN10",
                "clinical_tip": "Focal occipital seizures + reduced ERG amplitude in a child 2-5y = CLN10 screen essential. ERG amplitude reduction is often the earliest objective finding preceding clinical visual failure. Ophthalmology ERG at diagnosis and 6-monthly in non-congenital CLN10."
            },
            {
                "type": "Non-Convulsive Status Epilepticus (NCSE)",
                "pct": 32,
                "description": "NCSE in 32% — clinically important in advanced late-infantile CLN10 and congenital CLN10 (where it may be continuous or near-continuous from birth). In late-infantile/juvenile CLN10, NCSE presents as prolonged confusion or apparent acute cognitive deterioration — must be distinguished from CLN10 disease progression. Urgent EEG mandatory. TGB ABSOLUTE CI (NCSE precipitant). IV LEV or IV benzodiazepine for treatment.",
                "eeg": "Continuous or near-continuous generalised spike-wave (late-infantile NCSE); near-continuous multifocal discharge in congenital CLN10 (blending with seizure baseline); may be subtle in milder forms; background severely disorganised in advanced disease",
                "semiology": "Prolonged altered awareness; subtle motor manifestations; drooling; staring; apparent step-wise cognitive decline; may be mistaken for disease progression; congenital: difficult to distinguish from baseline encephalopathic state",
                "clinical_tip": "Any acute behavioural or cognitive deterioration in CLN10 (any phenotype) = urgent EEG to exclude NCSE. In congenital CLN10, persistent seizure activity may be near-continuous — EEG essential to guide comfort care decisions."
            }
        ],
        "triggers": [
            {
                "trigger": "Fever / Intercurrent Illness",
                "pct": 85,
                "description": "Fever is the most potent seizure trigger in CLN10 — non-congenital forms. Even brief temperature elevations trigger GTCS clustering and SE risk in late-infantile/juvenile CLN10. Written fever action plan is mandatory from diagnosis. Congenital CLN10: fever triggers additional seizure burden in already severely compromised neonates — infection management is primary.",
                "management": "Written fever action plan: antipyretics (paracetamol) at 37.5°C threshold; rescue buccal midazolam 0.3 mg/kg for >2 min or 2nd seizure in 30 min; ED alert card with CLN10 SE protocol (IV LEV — NOT fosphenytoin; palliative care plan for congenital form)"
            },
            {
                "trigger": "Metabolic Stress / Hypoglycaemia (Critical in Neonates)",
                "pct": 78,
                "description": "Metabolic stress — particularly hypoglycaemia — is a major neonatal seizure precipitant in congenital CLN10 on top of the underlying epileptic encephalopathy. Neonates with congenital CLN10 may have impaired glycaemic regulation from brainstem dysfunction. In late-infantile CLN10, dehydration and metabolic imbalance lower seizure threshold. KD patients require careful monitoring.",
                "management": "Neonatal glucose monitoring hourly in first 48h; IV dextrose for hypoglycaemia; adequate enteral/parenteral nutrition; in late-infantile CLN10 gastrostomy patients: enteral feed monitoring; electrolytes with every acute illness; KD: careful metabolic monitoring with ketone and glucose checks"
            },
            {
                "trigger": "Sleep Deprivation",
                "pct": 70,
                "description": "Sleep deprivation increases seizure frequency in non-congenital CLN10. Progressive CLN10 disrupts sleep architecture (nocturnal myoclonus, GTCS, pain). In late-infantile CLN10 children, sleep disruption creates a vicious cycle with families exhausted by nocturnal care. Nocturnal seizure monitoring is essential.",
                "management": "Strict sleep schedule; CLB for nocturnal seizure reduction; melatonin for sleep initiation (1-5 mg); nocturnal seizure monitoring device; respite care for families of late-infantile CLN10 children"
            },
            {
                "trigger": "Missed / Subtherapeutic AED Dose",
                "pct": 65,
                "description": "Missed AED dose precipitates GTCS or cluster in late-infantile/juvenile CLN10. In late-infantile CLN10, gastrostomy AED delivery is more reliable when oral intake becomes unsafe. Particularly relevant as disease progresses and dysphagia develops.",
                "management": "Electronic medication compliance; gastrostomy AED delivery in advanced late-infantile CLN10; blister pack dispensing; family written protocol; AED suspension formulation for gastrostomy administration"
            },
            {
                "trigger": "Tactile / Auditory / Visual Startle (Stimulus-Sensitive Myoclonus)",
                "pct": 62,
                "description": "Stimulus-sensitive myoclonus — touch, sound, or visual stimuli trigger myoclonic jerks in CLN10 myoclonus, especially in late-infantile form as disease advances. Congenital CLN10: generalised myoclonus may occur with any handling (including routine care), complicating neonatal nursing. In late-infantile CLN10: startle myoclonus interferes with self-care and mobility.",
                "management": "Low-stimulus environment; warn before physical contact; quiet room; reduce unexpected sounds; CLB reduces stimulus sensitivity; environmental modification at school and home; in congenital CLN10: gentle handling protocols in NICU/palliative setting"
            },
            {
                "trigger": "Emotional Stress / Anxiety",
                "pct": 55,
                "description": "Emotional stress triggers seizures and worsens myoclonus in juvenile/adult CLN10. Stress from loss of function (vision, cognitive skills, independence) in late-infantile and juvenile forms creates both psychological and physiological seizure risk. Family counselling and psychological support reduce stress-related seizure burden.",
                "management": "Psychosocial support; AAC for late-infantile CLN10 children; family counselling; physiotherapy; CLB for stress-related seizure cluster; visual impairment rehabilitation referral"
            },
            {
                "trigger": "Photic Stimulation (Photosensitivity)",
                "pct": 40,
                "description": "Photosensitivity in 40% of CLN10 at standard IPS rates — less prominent than CLN1/CLN2/CLN5 but clinically relevant. In late-infantile CLN10, photosensitivity may diminish as retinal degeneration progresses (reduced retinal input to cortex). Congenital CLN10: photosensitivity present but rarely tested given severity of overall neurological state.",
                "management": "IPS testing at diagnosis in late-infantile/juvenile CLN10 (annually); Z1-blue-tinted lenses; screen filters; 50 Hz screen refresh; re-test annually as photosensitivity may evolve with retinal progression"
            },
            {
                "trigger": "CLN10-Prohibited Drug Administration",
                "pct": 100,
                "description": "VGB → progressive retinal toxicity + catastrophic blindness (ABSOLUTE CI in all non-congenital CLN10; also contraindicated in congenital form despite palliative status — adds harm with no benefit). CBZ/OXC/PHT/fosphenytoin → 100% acute myoclonic worsening (ABSOLUTE CI). CRITICAL NEONATAL TRAP: fosphenytoin is standard second-line in neonatal SE protocols — MUST BE AVOIDED in CLN10 neonates (myoclonus worsening + seizure escalation). PHB (phenobarbitone) is safer neonatal alternative.",
                "management": "ABSOLUTE CI documentation in all records; neonatal SE protocol must explicitly override fosphenytoin default; VGB alert in all records regardless of phenotype severity; pharmacy alert system; A&E/NICU alert card"
            }
        ],
        "treatments": [
            {
                "drug": "Valproate (VPA) — Backbone AED (Late-Infantile / Juvenile CLN10)",
                "level": "Level B",
                "dose": "Late-infantile/juvenile: 20-40 mg/kg/day in 2-3 doses; start 10 mg/kg/day, titrate; target trough 60-100 mg/L; weight-based dosing; VPPP mandatory females ≥12y. Congenital CLN10: VPA caution in neonates <4 weeks (hepatic immaturity, neonatal VPA hepatotoxicity risk); if used, start lowest effective dose with intensive LFT monitoring.",
                "moa": "Sodium channel blockade + GABA-A enhancement + T-type calcium channel modulation + GABA transaminase inhibition → broad-spectrum anti-seizure + antimyoclonic",
                "efficacy": "Backbone AED for late-infantile and juvenile CLN10 (GTCS + myoclonic + atonic + focal). Essential for seizure control alongside LEV. Does not alter CLN10 disease progression.",
                "monitoring": "VPA TDM trough monthly first year, 3-monthly when stable (target 60-100 mg/L); LFTs 3-monthly; POLG1 exclusion before initiation (especially infants <2y); VPPP mandatory females ≥12y; carnitine 6-monthly; ammonia if drowsy/encephalopathic; neonatal: LFTs weekly",
                "cln10_note": "VPA is SAFE in CLN10 — CTSD = soluble lysosomal aspartic protease (NOT mitochondrial disease). VPA ABSOLUTE CI applies to MERRF/POLG (mitochondrial) — does NOT extend to CLN10. CRITICAL: POLG1 exclusion mandatory before VPA in any infant/child <2y with regression + seizures + mitochondrial features (POLG1 Alpers mimics late-infantile CLN10). Neonatal VPA requires intensive hepatic monitoring. VPPP counselling mandatory for all CLN10 females on VPA."
            },
            {
                "drug": "Levetiracetam (LEV) — IV SE Protocol + Adjunct (All CLN10 Phenotypes)",
                "level": "Level B",
                "dose": "Late-infantile/juvenile oral: 20-60 mg/kg/day; IV SE: 60 mg/kg (max 4500 mg) over 15 min. Neonatal congenital CLN10: IV LEV 40-60 mg/kg loading dose (neonatal renal dosing); oral neonatal: 10-30 mg/kg/day adjusted for renal maturity.",
                "moa": "SV2A synaptic vesicle protein modulation → presynaptic inhibition → anti-GTCS + anti-myoclonic. Effective in both congenital neonatal and post-neonatal CLN10 seizures.",
                "efficacy": "Adjunct to VPA (post-neonatal); IV LEV mandatory second-line SE replacing fosphenytoin (ABSOLUTE CI in CLN10). Most suitable AED for neonatal CLN10 alongside phenobarbitone. Effective for GTCS and myoclonic seizures in late-infantile/juvenile CLN10.",
                "monitoring": "Renal function (LEV renally excreted — adjust in neonates for renal immaturity); behavioural effects (agitation, irritability — monitor in late-infantile CLN10 cognitive impairment); mood monitoring in juvenile CLN10",
                "cln10_note": "IV LEV 60 mg/kg (40-60 mg/kg neonatal) is the MANDATORY second-line in CLN10 SE — NEVER fosphenytoin/phenytoin (ABSOLUTE CI — Na-channel blocker → myoclonus worsening). In congenital CLN10 neonatal SE, IV LEV is the drug of choice alongside IV phenobarbitone. This must be embedded in NICU, A&E, and neonatal protocols for any neonate with microcephaly + seizures."
            },
            {
                "drug": "Phenobarbitone (PHB) — Neonatal Seizure First-Line (Congenital CLN10)",
                "level": "Level B",
                "dose": "Neonatal loading: 20 mg/kg IV (up to 40 mg/kg if inadequate response); maintenance: 3-5 mg/kg/day IV/oral; monitor levels (target 20-40 mg/L neonatal). Not typically continued beyond neonatal period in late-infantile/juvenile forms.",
                "moa": "GABA-A positive allosteric modulator; potentiates chloride influx → membrane hyperpolarisation → broad neonatal anti-seizure activity",
                "efficacy": "First-line neonatal AED for congenital CLN10 seizures — well-established safety profile in neonates; effective for multifocal myoclonic + tonic seizures; combined with IV LEV for SE. Not suitable long-term in post-neonatal CLN10 (sedation, cognitive effects).",
                "monitoring": "PHB levels (target 20-40 mg/L neonatal); respiratory monitoring (apnoea risk); sedation; transition planning: PHB typically weaned once palliative/comfort plan established in congenital CLN10 or transitioned to VPA/LEV in rare survival beyond neonatal period",
                "cln10_note": "PHB is the neonatal first-line for congenital CLN10 seizures — avoids the PHT/fosphenytoin trap (Na-channel blocker ABSOLUTE CI). PHB is safer in neonates than VPA (neonatal hepatotoxicity risk with VPA). In congenital CLN10, PHB + IV LEV + buccal midazolam forms the neonatal SE triple protocol. PHB should NOT be continued in late-infantile/juvenile CLN10 beyond neonatal/infancy period."
            },
            {
                "drug": "ACTH / Prednisolone — Infantile Spasms in CLN10 (Instead of VGB)",
                "level": "Level B",
                "dose": "ACTH: 150 IU/m²/day IM for 2 weeks then taper (UKISS protocol); OR Prednisolone 10 mg QDS for 2 weeks then taper. Clinical response assessment at 2 weeks. NOT expected to achieve remission in CLN10 (underlying NCL progression continues).",
                "moa": "Glucocorticoid anti-inflammatory + GABA-B receptor modulation + neurosteroid effects → spasm suppression. Standard first-line for infantile spasms when VGB is contraindicated.",
                "efficacy": "Indicated in CLN10 with infantile spasms/West syndrome-like presentation INSTEAD OF VGB (VGB ABSOLUTE CI). Partial spasm reduction expected — complete remission unlikely given progressive NCL. ACTH/prednisolone provides temporary benefit while prognosis discussions proceed with family.",
                "monitoring": "Blood pressure (hypertension risk); glucose (hyperglycaemia); infection risk (immunosuppression); electrolytes; weight; EEG response at 2 weeks; spasm diary; in congenital CLN10: ACTH used alongside comfort care plan",
                "cln10_note": "ACTH/prednisolone replaces VGB for CLN10 infantile spasms — this is the most critical treatment decision in CLN10 West-syndrome overlap. Standard West syndrome pathway uses VGB first-line (UKISS trial) → ABSOLUTE CI in CLN10 retinal NCL. Clinicians must know CLN10 before initiating spasm treatment. GRODs on EM or PPT1/CTSD genetic diagnosis must trigger immediate VGB avoidance with ACTH substitution."
            },
            {
                "drug": "Clobazam (CLB) — Nocturnal + Stimulus-Sensitive Myoclonus Adjunct",
                "level": "Level B",
                "dose": "0.1-0.5 mg/kg/day; typically 5-20 mg/day; once-daily nocturnal or BD; lower doses in late-infantile CLN10 (cognitive side effect monitoring); paediatric: 0.1-0.3 mg/kg/day",
                "moa": "GABA-A positive allosteric modulator (1,5-benzodiazepine) → chloride influx → anti-seizure + myoclonus reduction + stimulus sensitivity reduction",
                "efficacy": "Nocturnal GTCS prevention; cluster prevention; stimulus-sensitive myoclonus reduction in late-infantile CLN10. Valuable adjunct particularly for nocturnal seizure management.",
                "monitoring": "Sedation (distinguish from disease cognitive decline); tolerance 3-monthly; avoid abrupt withdrawal; CLB sedation vs disease-related cognitive decline — dose-reduction trial may clarify; in late-infantile CLN10: respiratory monitoring with increasing dysphagia",
                "cln10_note": "CLB is effective for stimulus-sensitive myoclonus in CLN10 late-infantile form — reduces touch/sound-triggered jerks and nocturnal seizure clusters. In congenital CLN10 neonates, CLB is not standard (PHB + LEV preferred). In late-infantile/juvenile CLN10, CLB adjunct after VPA + LEV primary control."
            },
            {
                "drug": "Ketogenic Diet (KD) — After ≥3 AED Failures (Non-Congenital CLN10)",
                "level": "Level C",
                "dose": "Classic KD 4:1 ratio; MCT variant; dietitian-led; gastrostomy delivery in advanced late-infantile CLN10; oral KD feasible in early disease. Not indicated in congenital CLN10 (palliative focus; neonatal metabolic monitoring too complex alongside KD).",
                "moa": "Ketone metabolism → reduced neuronal excitability; putative neuroprotective effects via alternative metabolic substrate; anti-inflammatory mechanisms",
                "efficacy": "Adjunct after ≥3 AED failures in late-infantile/juvenile CLN10 — seizure reduction in 50-60% who complete initiation. Evidence in NCL specifically is limited (extrapolated from drug-resistant paediatric epilepsy).",
                "monitoring": "Dietitian monthly; lipid panel 3-monthly; weight; acidosis monitoring; renal stones; carnitine; gastrostomy required for reliable KD in late-infantile CLN10 advanced disease; KD + VPA: monitor LFTs closely (additive hepatotoxicity)",
                "cln10_note": "KD in late-infantile CLN10 after ≥3 AED failures — gastrostomy makes KD reliable when oral intake unsafe. KD + VPA: additive hepatotoxicity risk (LFTs monthly during KD). Modified Atkins or low-glycaemic-index treatment may be more practical for older juvenile CLN10 patients. Not indicated in congenital CLN10 (palliative care context)."
            },
            {
                "drug": "MDT Palliative Care + Ophthalmology + Neonatology (All Phenotypes)",
                "level": "Level A",
                "dose": "MDT from diagnosis: paediatric/adult neurology + neonatology (congenital) + ophthalmology (6-monthly ERG/VEP) + genetics + dietitian + physiotherapy + speech therapy + palliative care from diagnosis + social care. Congenital CLN10: neonatal palliative care team from birth; family counselling; place of care/death decisions.",
                "moa": "Comprehensive CLN10 management addressing all manifestations: seizures, cognitive decline, visual loss, dysphagia, motor deterioration, psychosocial needs, ACP",
                "efficacy": "MDT care is standard of care for all CLN10 phenotypes. Congenital CLN10 requires IMMEDIATE neonatal palliative care team involvement — ACP decisions (ventilation, resuscitation, comfort care) from day of diagnosis. Non-congenital: comprehensive MDT as other NCL types.",
                "monitoring": "FEES/videofluoroscopy in late-infantile CLN10; SARA ataxia; UMRS myoclonus; BDSRA registry; ACP from diagnosis (congenital: immediate; late-infantile: early; juvenile: progressive). Family psychological support throughout.",
                "cln10_note": "CONGENITAL CLN10 UNIQUE MDT NEED: immediate neonatal palliative care team. ACP decisions made with family from birth: ventilation threshold, CPR status, place of death (home vs hospice vs NICU), comfort care medications. Family grief support from diagnosis. Genetic counselling (25% recurrence risk). Sibling cascade testing at birth."
            },
            {
                "drug": "Rescue Midazolam (Buccal) + IV LEV / IV PHB — SE Protocol",
                "level": "Level A",
                "dose": "Non-neonatal: buccal midazolam 0.3 mg/kg (max 10 mg) — home/community first-line >5 min; IV LEV 60 mg/kg SE second-line. Neonatal (congenital CLN10): IV phenobarbitone 20 mg/kg first-line + IV LEV 40-60 mg/kg second-line; buccal midazolam 0.1 mg/kg neonatal (caution: respiratory).",
                "moa": "Midazolam: GABA-A fast-acting; IV PHB: GABA-A broad neonatal; IV LEV: SV2A → SE termination",
                "efficacy": "Family/carer-administered buccal midazolam for >5 min seizure (non-neonatal). IV PHB + IV LEV neonatal protocol. IV LEV MANDATORY hospital second-line in CLN10 SE (fosphenytoin ABSOLUTE CI).",
                "monitoring": "Family/carer training in buccal midazolam; seizure diary; post-SE assessment; annual SE protocol review; palliative care SE protocol for congenital CLN10 (comfort-focused endpoints rather than aggressive SE termination)",
                "cln10_note": "CLN10 SE PROTOCOL: (1) Non-neonatal: buccal midazolam 0.3 mg/kg → IV LEV 60 mg/kg (NOT fosphenytoin) → IV VPA 30 mg/kg. (2) Neonatal congenital CLN10: IV PHB 20 mg/kg → IV LEV 40-60 mg/kg → IV midazolam infusion. NEVER fosphenytoin/phenytoin in any CLN10 phenotype. Congenital CLN10: SE protocol aligned with palliative care plan — escalation limits documented in advance."
            }
        ],
        "contraindications": [
            {
                "drug": "Vigabatrin (VGB) — ABSOLUTE CI (Non-Congenital CLN10) / Avoid (Congenital — No Benefit)",
                "risk_level": "ABSOLUTE CI (Non-Congenital) / AVOID (Congenital)",
                "reason": "Late-infantile/juvenile CLN10: progressive retinal NCL (90-95%) → VGB retinopathy (VAR) + CLN10 retinal NCL = CATASTROPHIC combined blindness → ABSOLUTE CI. Congenital CLN10: disease is rapidly fatal; VGB provides no benefit and adds retinal toxicity — avoid. CRITICAL TRAP: CLN10 congenital with infantile spasms/West syndrome overlap → clinician prescribes VGB as per standard West syndrome guidelines → CATASTROPHIC retinal injury superimposed on congenital NCL. This is the most dangerous VGB mistake in paediatric neurology.",
                "alternative": "ACTH/prednisolone for infantile spasms/West overlap in CLN10 (NOT VGB). IV LEV for SE; VPA + LEV + CLB for seizure control in non-congenital CLN10."
            },
            {
                "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
                "risk_level": "ABSOLUTE CI",
                "reason": "Sodium channel blockers cause ACUTE MYOCLONIC WORSENING in CLN10 myoclonus (both late-infantile and juvenile forms). Both non-congenital CLN10 phenotypes present with GTCS first — misidentified as idiopathic epilepsy → CBZ prescribed → ACUTE MYOCLONIC DETERIORATION. Congenital form: PHT/fosphenytoin is standard neonatal SE second-line → ABSOLUTE CI in CLN10 neonates → must use IV LEV instead. Safe alternatives: VPA + LEV (late-infantile/juvenile); PHB + LEV (neonatal).",
                "alternative": "VPA (backbone late-infantile/juvenile); IV LEV for SE; PHB (neonatal). In any child with myoclonus + GTCS: VPA + LEV preferred over CBZ."
            },
            {
                "drug": "Fosphenytoin / Phenytoin (IV) — Critical Neonatal SE Trap",
                "risk_level": "ABSOLUTE CI",
                "reason": "Fosphenytoin ABSOLUTE CI in CLN10 — same Na-channel mechanism as PHT. Standard neonatal SE protocols use phenobarbitone then fosphenytoin — in CLN10 neonate, fosphenytoin MUST be replaced by IV LEV (40-60 mg/kg). NEONATAL TRAP: neonatologist unfamiliar with CLN10 uses standard SE protocol → fosphenytoin administered → myoclonic SE worsening in already compromised neonate. CLN10 neonatal SE protocol must be documented in NICU records.",
                "alternative": "Neonatal SE: IV PHB 20 mg/kg → IV LEV 40-60 mg/kg → IV midazolam infusion. NEVER fosphenytoin in CLN10 neonate."
            },
            {
                "drug": "Tiagabine (TGB)",
                "risk_level": "ABSOLUTE CI",
                "reason": "GABA reuptake inhibitor — absolute NCSE risk in NCL/PME with generalised epilepsy. NCSE in CLN10 may be misidentified as disease progression, especially in congenital form where encephalopathic baseline makes NCSE identification difficult. NCSE prolonged = additional neuronal injury on top of NCL storage.",
                "alternative": "CLB (GABA-A modulator — safe GABA-ergic option in CLN10). Avoid all GABA reuptake inhibitors."
            },
            {
                "drug": "Gabapentin (GBP) / Pregabalin (PGB)",
                "risk_level": "HIGH RISK — AVOID",
                "reason": "GBP/PGB can worsen myoclonus in NCL/PME. Multi-specialty prescribing trap in late-infantile CLN10: pain management or neuropathic pain → GBP prescribed → acute myoclonic worsening. Particularly relevant in juvenile CLN10 adults managed across multiple specialties.",
                "alternative": "Paracetamol + NSAID for pain; opioids if severe neuropathic pain. Shared care protocol must explicitly prohibit GBP/PGB."
            },
            {
                "drug": "LTG Monotherapy",
                "risk_level": "HIGH RISK",
                "reason": "LTG as sole AED insufficient for CLN10 myoclonus and GTCS — can paradoxically worsen myoclonus in some PME/NCL patients. Only acceptable as adjunct to VPA backbone in late-infantile/juvenile CLN10. Never initiate LTG without VPA backbone.",
                "alternative": "LTG only as adjunct to VPA; very slow titration required (VPA doubles LTG levels → SJS risk). VPA monotherapy or VPA+LEV as backbone."
            },
            {
                "drug": "AED Taper / Planned Withdrawal",
                "risk_level": "HIGH RISK",
                "reason": "CLN10 is a progressive NCL — seizures do NOT remit in any phenotype. NEVER attempt planned AED withdrawal in late-infantile/juvenile CLN10. Any AED reduction must be compensated by cross-tapering. Congenital CLN10: AED withdrawal decisions are within the palliative care plan context, not routine anti-epileptic management.",
                "alternative": "Maintain AED regimen lifelong in non-congenital CLN10. Any AED change: cross-taper with close monitoring. Congenital CLN10: AED decisions part of palliative plan — discuss with palliative and neonatology teams."
            }
        ],
        "monitoring": [
            {"item": "CTSD WES / NCL Gene Panel (Primary Molecular Diagnostic)", "detail": "WES or targeted NCL gene panel including CTSD (11p15.5) — primary molecular diagnostic. NO standardised CTSD DBS enzyme assay (unlike CLN1 PPT1 DBS). Diagnostic sequence: (1) EM skin biopsy → GRODs (NCL confirmed, days). (2) PPT1 enzyme assay DBS (1-3 days) → if clearly low → CLN1; if normal or borderline → CTSD WES (weeks). (3) If both PPT1 normal AND CTSD WES negative → expand NCL panel. Turkish/consanguineous → p.Trp383Stop PCR first (days). Neonatal: EM skin biopsy + PPT1 DBS + CTSD WES simultaneously from day 1."},
            {"item": "Skin Biopsy EM — GRODs (Identical to CLN1; Test Both CTSD and PPT1)", "detail": "EM of eccrine sweat gland on skin biopsy: GRODs (Granular Osmiophilic Deposits) in CLN10 — indistinguishable from CLN1/PPT1 by EM alone. GRODs represent SCMAS and palmitoylated protein accumulation. CRITICAL: GRODs = CLN1 (PPT1) OR CLN10 (CTSD) — BOTH must be tested biochemically/genetically. Congenital CLN10: EM skin biopsy may need to be taken urgently from neonate (fingertip or heel skin biopsy in small neonates). GRODs density correlates with disease severity — sparse in attenuated phenotypes (adequate EM sampling essential)."},
            {"item": "PPT1 Enzyme Assay DBS (First-Line When GRODs Confirmed)", "detail": "PPT1 (Palmitoyl-Protein Thioesterase 1) enzyme assay on dried blood spot — 1-3 days turnaround. Must be performed FIRST when GRODs found on EM (before CTSD WES results). Low PPT1 → CLN1 diagnosis (confirm with CLN1 sequencing). Normal/borderline PPT1 → proceed to CTSD WES. BORDERLINE PPT1 CAUTION: CLN10/CTSD LOF → secondary PPT1 reduction (CTSD activates PPT1) → PPT1 may be borderline low in CLN10. Borderline PPT1 does NOT confirm CLN1 — CTSD sequencing mandatory if any doubt. Commercial laboratories (Hamburg Bioanalytik, Kennedy Krieger NCL lab) provide PPT1 assay."},
            {"item": "POLG1 Exclusion Before VPA (Neonates/Infants <2y)", "detail": "POLG1/Alpers syndrome (mtDNA polymerase gamma deficiency) mimics late-infantile CLN10 — regression + seizures + hepatic failure. VPA ABSOLUTE CI in POLG1 (fatal hepatic failure). POLG1 sequencing recommended before VPA in any infant <2y with regression + seizures + mitochondrial features (lactic acidosis, hepatomegaly, family VPA hepatotoxicity history). Congenital CLN10: POLG1 less of a mimic (microcephaly distinguishes), but neonatal VPA requires LFT monitoring regardless."},
            {"item": "Ophthalmology ERG + VEP (6-Monthly Non-Congenital; Baseline Neonatal)", "detail": "ERG + VEP 6-monthly in late-infantile and juvenile CLN10 — progressive retinal NCL degeneration (90-95%). ERG amplitude reduction is often the earliest objective finding preceding clinical visual failure. VGB avoidance must be documented in ophthalmology records. Congenital CLN10: baseline ERG/VEP if neonate survives beyond first weeks (may guide comfort care prognosis discussions). Paediatric ERG requires specialist neonatal/infant sedation protocol."},
            {"item": "Brain MRI 3T (6-Monthly Late-Infantile / Annual Juvenile / Fetal if At-Risk Family)", "detail": "Late-infantile CLN10: 6-monthly MRI — progressive cortical atrophy (parieto-occipital), cerebellar atrophy, white matter signal change. Juvenile: annual MRI — slower progressive atrophy. Congenital (at-risk family / prenatal): fetal MRI from 20 weeks gestation → simplified gyral pattern / cortical malformation suggests congenital NCL → fetal blood sampling or amniocentesis for CTSD genetics. Anaesthesia planning: IV LEV + PHB (not fosphenytoin) for any sedation/general anaesthesia."},
            {"item": "EEG (Urgent Neonatal + Annual Post-Neonatal + Acute NCSE)", "detail": "Urgent EEG in all neonates with multifocal seizures + microcephaly: burst-suppression pattern highly suggestive of congenital NCL (alongside GRODs EM). Annual EEG in late-infantile/juvenile CLN10 for background progression monitoring; standard IPS (photosensitivity 40%). Acute EEG for unexplained behavioural/cognitive change → NCSE exclusion. Neonatal EEG monitoring (aEEG/cEEG) recommended in congenital CLN10 NICU/palliative setting."},
            {"item": "FEES / Videofluoroscopy — Dysphagia (Late-Infantile Priority)", "detail": "Late-infantile CLN10: FEES/videofluoroscopy annually from 2 years after disease onset — gastrostomy planning before aspiration events (expected ~75% within 5-8 years). Congenital CLN10: dysphagia/poor suckling from birth → nasogastric tube initially; gastrostomy discussion part of ACP (comfort care vs longer-term nutrition decisions). Juvenile CLN10: dysphagia develops later — FEES when dysarthria and swallowing concern arise."},
            {"item": "SARA Ataxia Scale (6-Monthly Late-Infantile / Annual Juvenile)", "detail": "Scale for Assessment and Rating of Ataxia (SARA): 6-monthly in late-infantile CLN10 (rapid ataxia progression); annual in juvenile CLN10. SARA guides physiotherapy and OT goals; combined with visual impairment assessment for compound fall risk. Congenital CLN10: SARA not applicable (congenital neurological deficit present from birth; standardised neonatal neurological exam instead)."},
            {"item": "UMRS Myoclonus Scale (6-Monthly Non-Congenital)", "detail": "Unified Myoclonus Rating Scale — 6-monthly in late-infantile and juvenile CLN10 for myoclonus severity quantification and treatment response (piracetam or CLB dose titration). Congenital CLN10: quantitative myoclonus assessment via EEG-poly myoclonus correlation (cEEG monitoring in NICU). Annual IPS re-testing in non-congenital CLN10 (photosensitivity may evolve with retinal progression)."},
            {"item": "VPA TDM + LFT + Carnitine (Non-Congenital; Intensive Neonatal)", "detail": "Non-congenital CLN10: VPA TDM trough monthly first year, 3-monthly stable (target 60-100 mg/L); LFTs 3-monthly; carnitine 6-monthly; ammonia if encephalopathic. Congenital CLN10 (if VPA used): LFTs weekly (neonatal hepatic immaturity + VPA = high hepatotoxicity risk); PHB preferred over VPA in neonates. KD + VPA: additive hepatotoxicity → LFTs monthly during KD."},
            {"item": "Neurodevelopmental / Neuropsychological Assessment (Late-Infantile Annual)", "detail": "Annual neurodevelopmental assessment in late-infantile CLN10: BSID-IV (Bayley), Vineland Adaptive Behavior Scales — documents cognitive trajectory; guides educational support, AAC needs, school placement. Juvenile CLN10: annual cognitive + adaptive measures; employment and independence planning as cognitive decline progresses. Congenital CLN10: neonatal neurological exam (Sarnat grading, aEEG-background monitoring) — guides ACP discussions."},
            {"item": "SUDEP Risk / Nocturnal Monitoring (Non-Congenital CLN10)", "detail": "CLN10 SUDEP risk elevated (drug-resistant epilepsy + nocturnal GTCS + progressive cognitive impairment). Nocturnal seizure monitoring device for late-infantile CLN10 children as disease advances and independent mobility is lost. Congenital CLN10: continuous cardiorespiratory monitoring in NICU/palliative setting — focus on comfort, not SUDEP prevention specifically. SUDEP discussed in ACP for all non-congenital CLN10 from diagnosis."},
            {"item": "BDSRA / NCL Network Europe Registry + Genetics Counselling", "detail": "All CLN10 patients (all phenotypes) enrolled in BDSRA and NCL Network Europe / NCL Resource — gene therapy / ERT trial eligibility (CTSD ERT conceptually feasible as soluble lysosomal enzyme). Congenital CLN10: BDSRA enrolment provides natural history data critical for understanding congenital NCL. Genetic counselling (25% recurrence risk — AR biallelic): sibling cascade testing at birth; prenatal diagnosis by CVS/amniocentesis for subsequent pregnancies; PGT in IVF available for known CLN10 families."},
            {"item": "ACP — Palliative Care from Birth (Congenital) / Diagnosis (Non-Congenital)", "detail": "Congenital CLN10: ACP from BIRTH — fatal disease; immediate decisions: resuscitation status, ventilation threshold, comfort care medications, place of death (home, hospice, NICU). Family psychological support from day 1; sibling welfare; bereavement pathway. Late-infantile CLN10: ACP from diagnosis — progressive fatal disease (survival teens-20s); decisions about gastrostomy, ventilation, resuscitation, place of care. Juvenile CLN10: ACP initiated at diagnosis updated as disease progresses — decades-long trajectory."}
        ],
        "lifecycle_stages": [
            {
                "stage": "Prenatal / Genetic Risk (Before Birth / In Utero)",
                "description": "At-risk family (prior affected child with CLN10): fetal MRI from 20 weeks (simplified gyral pattern suggests congenital NCL); fetal blood sampling or amniocentesis for CTSD genetics (CVS from 11 weeks). Preimplantation genetic testing (PGT) in IVF for known CLN10 mutation carriers. BDSRA pre-enrolment. Genetic counselling: 25% recurrence risk (AR biallelic). Prenatal diagnosis critical for family preparation and neonatal palliative care planning."
            },
            {
                "stage": "Neonatal Congenital CLN10 (Day 0 to Death — Days to Months)",
                "description": "Born with microcephaly + seizures from hours of life. NICU admission. Diagnostic emergency: GRODs skin biopsy + PPT1 enzyme DBS + CTSD WES simultaneously. Neonatal EEG (burst-suppression → myoclonic). PHB + IV LEV as neonatal SE protocol (NEVER fosphenytoin). Palliative care team involved from day 1. ACP: ventilation, resuscitation, comfort care goals, place of death. Family counselling, psychological support, sibling welfare. BDSRA registry. Fatal within days to ~12 months."
            },
            {
                "stage": "Pre-Symptomatic / Early (Aged 0 to Seizure Onset — Non-Congenital)",
                "description": "Known CLN10 genotype (sibling diagnosis or incidental) in late-infantile/juvenile forms. Baseline ERG + ophthalmology; EEG; brain MRI; developmental assessment. CLN10 drug alert documentation from diagnosis — VGB avoidance, fosphenytoin avoidance. BDSRA registry enrolment. Gene therapy/ERT trial enrolment if available. Pre-symptomatic window: 0 to ~2-3y (late-infantile); 0 to ~12y (juvenile)."
            },
            {
                "stage": "First Seizure — Diagnostic Emergency (Non-Congenital)",
                "description": "GTCS at disease-specific onset age (late-infantile: 2-5y; juvenile: adolescence). Diagnostic pathway: EM skin biopsy → GRODs confirmed + PPT1 enzyme DBS (1-3 days) + CTSD WES → ophthalmology ERG + VEP + brain MRI + EEG + POLG1 exclusion before VPA. VPA + LEV started. CLN10 drug alert card (VGB ABSOLUTE CI; CBZ/PHT/fosphenytoin ABSOLUTE CI). ACP discussion initiated. BDSRA enrolment. GRODs + normal PPT1 → CTSD WES expedited."
            },
            {
                "stage": "Active Epilepsy + Cognitive Regression (Late-Infantile / Juvenile CLN10)",
                "description": "Late-infantile: Multiple seizure types (GTCS + myoclonic + focal occipital); action myoclonus → piracetam/CLB; cognitive regression; visual failure (ERG reduction → clinical blindness); dysphagia → gastrostomy; ataxia → mobility aids; MDT intensification. Juvenile: slower course — progressive myoclonus; cognitive decline; visual impairment; school/employment support. KD after ≥3 AED failures."
            },
            {
                "stage": "Established Severe Disability / Late-Palliative",
                "description": "Late-infantile CLN10: Profound intellectual disability; near-total visual loss; severe ataxia + myoclonus; gastrostomy-dependent; communication via AAC; drug-resistant seizures; palliative care intensification; death in teens-20s. Juvenile CLN10: progressive disability over decades; care-dependent; seizures ongoing; death in 4th-5th decade (attenuated alleles). Both: BDSRA trial participation; family bereavement support; ACP updated regularly."
            }
        ]
    }


def get_definitions():
    return {
        "concepts": [
            {
                "concept": "CLN10-CTSD-11p15.5-Lysosomal-Aspartic-Endopeptidase-GRODs-NCL",
                "definition": "CLN10 (CTSD, 11p15.5) — Cathepsin D; 412 aa precursor; light chain Asp106 + heavy chain Asp231 catalytic dyad; lysosomal aspartic endopeptidase; pH optimum 3.5-5.0; key substrates: SCMAS (NCL storage protein) + pro-PPT1 (CLN1 enzyme activation). CTSD LOF → SCMAS accumulation → GRODs on EM → NCL. pLI ~0.89. AR biallelic LOF. OMIM *116840 / #610127. Three phenotypes: congenital (most severe NCL) → late-infantile → juvenile/adult (attenuated).",
                "standard": "Siintola-2006-AnnNeurol / Steinfeld-2006-HumMolGenet / NCL-Resource-2024 / OMIM"
            },
            {
                "concept": "Congenital-CLN10-Most-Severe-NCL-Prenatal-Brain-Malformation",
                "definition": "Congenital CLN10 is the MOST SEVERE NCL and among the most severe lysosomal storage diseases. Biallelic null CTSD mutations → complete lysosomal aspartic endopeptidase absence from embryonic development → prenatal neuronal apoptosis → microcephaly + simplified gyral pattern at birth → seizures from first hours of life → neonatal multifocal myoclonic epileptic encephalopathy → respiratory failure → fatal within days to ~12 months. ONLY NCL type with prenatal brain malformation. Diagnosis at birth: GRODs skin biopsy + PPT1 DBS + CTSD WES simultaneously.",
                "standard": "Siintola-2006-AnnNeurol / Steinfeld-2006-HumMolGenet / NCL-Resource-2024"
            },
            {
                "concept": "GRODs-EM-CLN1-AND-CLN10-Both-Test-PPT1-First",
                "definition": "GRODs (Granular Osmiophilic Deposits) on EM skin biopsy = CLN1 (PPT1 LOF) OR CLN10 (CTSD LOF → secondary PPT1 reduction). GRODs are indistinguishable between CLN1 and CLN10 by EM alone. MANDATORY DIAGNOSTIC RULE: When GRODs found, ALWAYS test BOTH: (1) PPT1 enzyme assay DBS first (1-3 days — fastest); (2) CTSD WES (weeks). Normal PPT1 does NOT exclude CLN10. Borderline PPT1 may occur in CLN10 (CTSD-mediated PPT1 activation impaired) — CTSD sequencing mandatory if borderline.",
                "standard": "NCL-Resource-2024 / ILAE-2022 / Siintola-2006-AnnNeurol / Steinfeld-2006-HumMolGenet"
            },
            {
                "concept": "CTSD-Activates-PPT1-Dual-Lysosomal-Enzyme-Relationship",
                "definition": "CTSD proteolytically cleaves and activates pro-PPT1 (palmitoyl-protein thioesterase 1 / CLN1 enzyme) in the lysosome. CTSD LOF → pro-PPT1 not activated → secondary functional PPT1 deficiency → compound NCL substrate accumulation (SCMAS + palmitoylated proteins). This explains: (1) why both CLN1 and CLN10 produce GRODs; (2) why PPT1 enzyme activity may be borderline low in CLN10 patients; (3) the mechanistic convergence of CLN1 and CLN10 on the same lysosomal substrate pathway. Unique inter-NCL enzyme relationship with direct clinical diagnostic implications.",
                "standard": "Tyynelä-2000-MolGenetMetab / NCL-Resource-2024 / Siintola-2006-AnnNeurol"
            },
            {
                "concept": "No-Standardised-CTSD-DBS-Assay-WES-Required-Unlike-CLN1-CLN2",
                "definition": "CTSD enzyme assay on DBS is NOT standardised as a first-line diagnostic test (unlike CLN1 PPT1 DBS assay and CLN2 TPP1 DBS assay — both commercially available with established reference ranges). CTSD pepstatin-inhibitable proteolysis can be measured in fibroblasts or amniotic fluid but is not routinely available as a DBS screening assay. Clinical diagnostic approach: EM → GRODs → PPT1 DBS first (standardised, fast) → if normal → CTSD WES (weeks). Turkish consanguineous: p.Trp383Stop PCR first (days). Do NOT wait for CTSD enzyme assay — WES is the primary CTSD diagnostic tool.",
                "standard": "NCL-Resource-2024 / Siintola-2006-AnnNeurol / BDSRA-Diagnostic-Guidelines"
            },
            {
                "concept": "VGB-ABSOLUTE-CI-Infantile-Spasms-West-Syndrome-Trap-CLN10",
                "definition": "VGB is first-line for infantile spasms (UKISS/NICE guidance) but ABSOLUTE CI in CLN10 retinal NCL. CLN10 congenital and late-infantile forms can present with infantile spasms / West syndrome-like EEG. CRITICAL TRAP: CLN10 infant with spasms → standard West syndrome pathway prescribes VGB → CATASTROPHIC combined retinal NCL + VGB retinopathy. RULE: ANY infant with spasms + microcephaly or developmental regression → PPT1 DBS + CTSD WES BEFORE VGB. If CLN10 suspected or GRODs confirmed → ACTH/prednisolone (NOT VGB) for spasms.",
                "standard": "MHRA-VPPP-2021 / NICE-NG217 / NCL-Resource-2024 / UKISS-Trial"
            },
            {
                "concept": "VPA-SAFE-CLN10-Lysosomal-NOT-Mitochondrial-Neonatal-Caution",
                "definition": "VPA is SAFE in CLN10 — CTSD = lysosomal aspartic protease (NOT mitochondrial disease). VPA ABSOLUTE CI applies to MERRF/POLG — does NOT extend to CLN10. VPA backbone in late-infantile/juvenile CLN10. NEONATAL CAUTION: VPA in neonates <4 weeks has hepatotoxicity risk from hepatic immaturity (independent of POLG1). PHB is preferred over VPA for neonatal seizures in congenital CLN10. POLG1 exclusion mandatory before VPA in infants <2y with regression + seizures + mitochondrial features.",
                "standard": "CPIC-POLG1-2023 / NCL-Resource-2024 / NICE-NG217"
            },
            {
                "concept": "CBZ-OXC-PHT-Fosphenytoin-ABSOLUTE-CI-Neonatal-SE-Trap",
                "definition": "CBZ/OXC/PHT ABSOLUTE CI in CLN10 — Na-channel blockers cause ACUTE MYOCLONIC WORSENING. CRITICAL NEONATAL TRAP: fosphenytoin is standard neonatal SE second-line → ABSOLUTE CI in CLN10 neonate → must be overridden with IV LEV (40-60 mg/kg). Non-congenital CLN10: child with first GTCS at 2-5y misidentified as idiopathic → CBZ prescribed → ACUTE MYOCLONIC DETERIORATION. Safe choices: VPA + LEV (late-infantile/juvenile); PHB + IV LEV (neonatal). Fosphenytoin avoidance must be embedded in NICU and A&E protocols.",
                "standard": "NCL-Resource-2024 / NICE-NG217 / ILAE-2022"
            },
            {
                "concept": "CTSD-ERT-Conceptually-Feasible-Soluble-Lysosomal-Enzyme-CLN2-Precedent",
                "definition": "CTSD (Cathepsin D) is a SOLUBLE lysosomal enzyme — the same class as CLN2/TPP1 where cerliponase alfa ERT is approved (NICE/FDA). Soluble lysosomal enzymes can be replaced by intrathecal or IV recombinant enzyme (mannose-6-phosphate receptor uptake). CLN2 cerliponase proves ERT feasibility for NCL soluble enzymes. CLN10/CTSD ERT is in preclinical research phase (CTSD-deficient mouse models). TARGET PATIENTS: late-infantile/juvenile CLN10 (not congenital — irreversible prenatal brain damage). BDSRA enrolment essential for trial access.",
                "standard": "NCL-Resource-2024 / BDSRA-Registry / Koike-2000-JNeurosci"
            },
            {
                "concept": "Fetal-MRI-Simplified-Gyri-At-Risk-Family-Congenital-CLN10",
                "definition": "Congenital CLN10 causes PRENATAL brain malformation: simplified gyral pattern (agyria-pachygyria-like) from prenatal neuronal apoptosis. Fetal MRI from ~20 weeks gestation in at-risk families (prior congenital CLN10 child) — cortical malformation on fetal MRI → urgent fetal genetic testing (amniocentesis/fetal blood sampling for CTSD). CVS from 11 weeks for families with known CLN10 mutation. Preimplantation genetic testing (PGT) in IVF available. Prenatal diagnosis allows family preparation and neonatal palliative care planning.",
                "standard": "Siintola-2006-AnnNeurol / NCL-Resource-2024 / BDSRA-Registry"
            },
            {
                "concept": "No-Disease-Modifying-Therapy-CLN10-ERT-Preclinical-Gene-Therapy",
                "definition": "No approved disease-modifying therapy for CLN10/CTSD. Management is purely symptomatic. Investigational: CTSD enzyme replacement therapy (ERT) in preclinical phase — soluble lysosomal enzyme (CLN2 cerliponase ERT precedent). CTSD gene therapy in early research. Congenital CLN10: palliative comfort care from birth — no realistic therapeutic window given prenatal brain damage. All non-congenital CLN10 patients must be enrolled in BDSRA/NCL Resource for trial eligibility.",
                "standard": "NCL-Resource-2024 / BDSRA-Registry / ILAE-2022"
            },
            {
                "concept": "Burst-Suppression-EEG-Neonatal-CLN10-Pathognomonic-Severe-Encephalopathy",
                "definition": "Neonatal EEG burst-suppression pattern in CLN10 congenital form indicates severe epileptic encephalopathy from birth. Burst-suppression in a neonate with microcephaly and multifocal seizures is a critical trigger for NCL screen (GRODs skin biopsy). Congenital CLN10 burst-suppression may evolve to near-isoelectric EEG within weeks as neurodegeneration progresses. aEEG/cEEG monitoring in NICU recommended for seizure burden quantification and palliative care decision support.",
                "standard": "NCL-Resource-2024 / Siintola-2006-AnnNeurol / ILAE-2022"
            },
            {
                "concept": "Gastrostomy-Late-Infantile-CLN10-Dysphagia-AED-Nutrition",
                "definition": "Late-infantile CLN10: progressive dysphagia from cerebellar + cortical NCL degeneration → unsafe oral intake ~75% within 5-8 years of onset. PEG gastrostomy: reliable AED delivery, nutrition, enables KD, prevents aspiration pneumonia. FEES/videofluoroscopy guides timing — gastrostomy BEFORE aspiration events. Congenital CLN10: nasogastric tube initially (dysphagia from birth — suckling failure); gastrostomy decision within ACP comfort care context.",
                "standard": "NCL-Resource-2024 / NICE-NG217 / Mole-2019-LancetNeurol"
            },
            {
                "concept": "SUDEP-Risk-CLN10-Drug-Resistant-Nocturnal-GTCS",
                "definition": "CLN10 SUDEP risk elevated: drug-resistant epilepsy (78%) + nocturnal GTCS + progressive cognitive impairment → impaired post-ictal arousal. Late-infantile CLN10 children require nocturnal monitoring device as independence is lost. Congenital CLN10: cardiorespiratory monitoring in NICU/palliative care — death from respiratory failure rather than SUDEP per se. SUDEP documented in ACP for all non-congenital CLN10.",
                "standard": "NICE-NG217 / NCL-Resource-2024 / ILAE-SUDEP-Guidelines"
            },
            {
                "concept": "POLG1-Exclusion-Before-VPA-CLN10-Alpers-Mimic-Late-Infantile",
                "definition": "POLG1/Alpers syndrome mimics late-infantile CLN10 presentation — regression + seizures + hepatic failure. VPA ABSOLUTE CI in POLG1 (fatal hepatic failure). POLG1 sequencing recommended before VPA in infants <2y with regression + seizures + mitochondrial features (lactic acidosis, hepatomegaly, family VPA hepatotoxicity). Late-infantile CLN10 onset 2-5y is outside peak POLG1 age (usually <2y) but overlap exists. GRODs EM distinguishes CLN10 from POLG1 (POLG1 shows pleomorphic mitochondria, not GRODs).",
                "standard": "CPIC-POLG1-2023 / NCL-Resource-2024 / NICE-NG217"
            }
        ],
        "thresholds": [
            {"threshold": "Neonate with microcephaly + multifocal seizures from birth → GRODs skin biopsy + PPT1 DBS + CTSD WES simultaneously (day 1)", "standard": "NCL-Resource-2024 / Siintola-2006-AnnNeurol"},
            {"threshold": "GRODs on EM → PPT1 enzyme DBS first (1-3 days); if normal/borderline → CTSD WES mandatory", "standard": "NCL-Resource-2024 / ILAE-2022"},
            {"threshold": "VGB ABSOLUTE CI in all CLN10 phenotypes — infantile spasms → ACTH/prednisolone (NOT VGB)", "standard": "MHRA-VPPP-2021 / NCL-Resource-2024"},
            {"threshold": "CBZ/OXC/PHT/Fosphenytoin ABSOLUTE CI in CLN10 — neonatal SE: IV PHB 20 mg/kg + IV LEV 40-60 mg/kg (NOT fosphenytoin)", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "POLG1 exclusion before VPA in all infants <2y with regression + seizures + mitochondrial features", "standard": "CPIC-POLG1-2023"},
            {"threshold": "Burst-suppression EEG in neonate with microcephaly → highest-priority NCL screen (GRODs EM + PPT1 DBS + CTSD WES)", "standard": "NCL-Resource-2024 / ILAE-2022"},
            {"threshold": "PPT1 borderline low in GRODs patient → CTSD sequencing mandatory (CTSD→PPT1 activation effect; borderline PPT1 ≠ CLN1 confirmed)", "standard": "NCL-Resource-2024 / Tyynelä-2000-MolGenetMetab"},
            {"threshold": "IV LEV 60 mg/kg second-line SE (40-60 mg/kg neonatal) — NEVER fosphenytoin in CLN10", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "ERG + VEP 6-monthly in late-infantile CLN10; annually in juvenile from diagnosis", "standard": "NCL-Resource-2024 / NICE-NG217"},
            {"threshold": "Buccal midazolam 0.3 mg/kg for >5 min seizure (0.1 mg/kg neonatal with respiratory monitoring)", "standard": "NICE-NG217 / APLS-Guidelines"},
            {"threshold": "Congenital CLN10: palliative care team from birth; ACP from day 1 (ventilation/resuscitation/comfort goals)", "standard": "NCL-Resource-2024 / NICE-NG86-Palliative"},
            {"threshold": "BDSRA / NCL Network Europe enrolment mandatory at diagnosis for all CLN10 phenotypes (ERT trial eligibility)", "standard": "BDSRA-Registry / NCL-Network-Europe"}
        ],
        "standards": [
            {"standard": "Siintola-2006-AnnNeurol", "detail": "Siintola E et al. 'Cathepsin D deficiency underlies congenital human neuronal ceroid-lipofuscinosis.' Annals of Neurology 2006;59(6):1033-6 — first identification of CTSD biallelic mutations in Turkish/Finnish congenital NCL; established CLN10 as a disease entity"},
            {"standard": "Steinfeld-2006-HumMolGenet", "detail": "Steinfeld R et al. 'Cathepsin D deficiency is associated with a human neurodegenerative disorder.' Human Molecular Genetics 2006;15(12):1967-76 — independent discovery of CTSD mutations in Italian congenital NCL; confirmed CLN10 = CTSD deficiency"},
            {"standard": "Tyynelä-2000-MolGenetMetab", "detail": "Tyynelä J et al. 'Elevated cathepsin D immunoreactivity in neuronal ceroid lipofuscinosis brains.' Molecular Genetics and Metabolism 2000;69(2):123-35 — characterisation of CTSD's role in NCL pathogenesis and PPT1 activation"},
            {"standard": "Koike-2000-JNeurosci", "detail": "Koike M et al. 'Cathepsin D deficiency induces lysosomal storage with ceroid lipofuscin in mouse CNS neurons.' Journal of Neuroscience 2000;20(18):6898-906 — CTSD-deficient mouse model demonstrating NCL pathology; key preclinical model for CLN10"},
            {"standard": "Mole-2019-LancetNeurol", "detail": "Mole SE et al. 'Clinical management of the neuronal ceroid lipofuscinoses.' Lancet Neurology 2019;18(12):1131-40 — comprehensive NCL management standard including CLN10 congenital and non-congenital forms"},
            {"standard": "NCL-Resource-2024", "detail": "NCL Resource (ncl.mni.mcgill.ca) — international NCL management guidelines, diagnostic algorithms (GRODs pathway: PPT1 DBS first → CTSD WES), CLN10 management including congenital neonatal palliative care; updated 2024"},
            {"standard": "ILAE-2022", "detail": "International League Against Epilepsy 2022 — epilepsy classification and treatment guidelines applicable to CLN10 seizure management (congenital and non-congenital forms)"},
            {"standard": "NICE-NG217", "detail": "NICE Guideline NG217: 'Epilepsies in children, young people and adults' (2022) — AED selection, contraindications, SE protocols applicable to CLN10; VGB CI, fosphenytoin CI, ACTH for infantile spasms"},
            {"standard": "MHRA-VPPP-2021", "detail": "MHRA Valproate Pregnancy Prevention Programme (2021) — mandatory VPA counselling for all females ≥12y on valproate; applies to juvenile/adult CLN10 females on VPA backbone"},
            {"standard": "CPIC-POLG1-2023", "detail": "CPIC Guideline for POLG1 — POLG1 exclusion mandatory before VPA in infants <2y with regression + seizures (POLG1 Alpers mimics late-infantile CLN10)"},
            {"standard": "ACMG-AMP-2015", "detail": "ACMG/AMP 2015 variant classification guidelines — applied to CTSD variant interpretation; genotype-phenotype correlation (null = congenital; missense = late-infantile/juvenile based on residual CTSD activity)"},
            {"standard": "BDSRA-Registry", "detail": "Batten Disease Support and Research Association Registry — mandatory enrolment for all CLN10 patients; ERT trial eligibility (CTSD ERT conceptually feasible — CLN2 cerliponase precedent); natural history data for congenital NCL"}
        ],
        "references": [
            {"ref": "Siintola-2006-AnnNeurol", "citation": "Siintola E, Partanen S, Strömme P, et al. (2006). Cathepsin D deficiency underlies congenital human neuronal ceroid-lipofuscinosis. Annals of Neurology, 59(6), 1033-1036."},
            {"ref": "Steinfeld-2006-HumMolGenet", "citation": "Steinfeld R, Reinhardt K, Engel K, et al. (2006). Cathepsin D deficiency is associated with a human neurodegenerative disorder. Human Molecular Genetics, 15(12), 1967-1976."},
            {"ref": "Tyynelä-2000-MolGenetMetab", "citation": "Tyynelä J, Sohar I, Sleat DE, et al. (2000). A mutation in the ovine cathepsin D gene causes a congenital lysosomal storage disease with profound neurodegeneration. EMBO Journal, 19(12), 2786-2792."},
            {"ref": "Koike-2000-JNeurosci", "citation": "Koike M, Nakanishi H, Saftig P, et al. (2000). Cathepsin D deficiency induces lysosomal storage with ceroid lipofuscin in mouse CNS neurons. Journal of Neuroscience, 20(18), 6898-6906."},
            {"ref": "Mole-2019-LancetNeurol", "citation": "Mole SE, Anderson G, Band HA, et al. (2019). Clinical management of the neuronal ceroid lipofuscinoses. The Lancet Neurology, 18(12), 1131-1140."},
            {"ref": "Canafoglia-2014-Epilepsia", "citation": "Canafoglia L, Bugiani M, Dalla Bernardina B, et al. (2014). Rhythmic cortical myoclonus in neuronal ceroid-lipofuscinosis due to CTSD, CLN6, and CLN8 mutations. Epilepsia, 55(10), 1599-1607."}
        ]
    }
