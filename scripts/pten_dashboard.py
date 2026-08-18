"""
PTEN Epilepsy — PTEN Hamartoma Tumour Syndrome / mTORopathy / Cowden / Bannayan-Riley-Ruvalcaba
================================================================================================
40-patient cohort · PTEN (10q23.31) · Autosomal dominant LOF (haploinsufficiency)
PTEN = Phosphatase and Tensin Homolog; lipid phosphatase; PIP3→PIP2 dephosphorylation
LOF → PIP3 accumulates → AKT constitutively active → TSC1/2 inactivated → mTOR hyperactivation
→ neuronal hypertrophy + cortical overgrowth + epilepsy + ASD + macrocephaly

PTEN PROTEIN BIOLOGY:
PTEN (10q23.31), Phosphatase and Tensin Homolog:
  - 403 amino acids, ~54.7 kDa
  - Domain structure:
      N-terminal phosphatase domain (aa 1-185): catalytic core; dephosphorylates PIP3→PIP2
        (lipid phosphatase activity) AND pTyr/pSer/pThr (protein phosphatase activity)
      C2 domain (aa 186-351): Ca²⁺-independent membrane binding; positions phosphatase domain
        near PIP3 substrate at inner leaflet
      C-terminal tail (aa 352-403): PDZ-binding motif (DQSI C-terminus); regulatory; contains
        phosphorylation cluster (Ser370, Ser380, Thr382, Thr383, Ser385) that promotes
        closed/inactive conformation
      PIP3-binding (N-terminus aa 1-15): poly-basic stretch interacts with PIP2/PIP3 at membrane
  - Function: brake on PI3K-AKT-mTOR axis;
      PIP3 (phosphatidylinositol-3,4,5-trisphosphate) recruits AKT to membrane → AKT activation
      PTEN dephosphorylates PIP3→PIP2 → AKT cannot be recruited → mTOR not activated
      LOF → excess PIP3 → constitutive AKT → TSC1/2 phosphorylated/inactivated → Rheb-GTP active
      → mTORC1 active → S6K1 (4E-BP1) → excess mRNA translation → neuronal hypertrophy
  - Nuclear PTEN: separate functions — DNA repair (BRCA1 interaction), chromosomal stability
  - Tumour suppressor: most commonly mutated/deleted tumour suppressor in human cancers
    (second only to TP53)
  - pLI = 0.97 (haploinsufficiency; LOF in one allele → PHTS; biallelic LOF → lethal)

PTEN HAMARTOMA TUMOUR SYNDROME (PHTS) — CLINICAL SPECTRUM:
  Cowden Syndrome (CS, OMIM #158350):
    Adult-onset; mucocutaneous hamartomas (trichilemmomas, papillomatous papules, acral keratoses)
    Breast cancer LT risk 25-50% (female); thyroid cancer 3-38%; endometrial cancer 13-28%
    Macrocephaly; autism spectrum (25-35%); epilepsy 15-25%
  Bannayan-Riley-Ruvalcaba Syndrome (BRRS, OMIM #153480):
    Childhood-onset; macrocephaly + developmental delay + lipomas + haemangiomas + hamartomatous
    intestinal polyps; penile freckling in males; higher epilepsy prevalence
  PTEN Autism Spectrum Disorder (PTEN-ASD):
    Autism + macrocephaly (HC >97th centile) + variable epilepsy (20-35%)
    ASHG 2013: offer PTEN testing to ASD patients with HC >2 SD above mean
  Lhermitte-Duclos Disease (LDD):
    Dysplastic cerebellar gangliocytoma; pathognomonic "tiger-stripe" T2/FLAIR MRI pattern
    Seizures (cerebellar origin atypical; mass effect + hydrocephalus-related)
  PTEN-Related Proteus Syndrome: mosaic PTEN + activating AKT1

CONTRAINDICATIONS UNIQUE TO PTEN EPILEPSY:
  1. EVEROLIMUS + LIVE VACCINES — ABSOLUTE CI: immunosuppression → live-attenuated vaccine
     infection risk; complete vaccination schedule BEFORE everolimus; killed vaccines only
  2. EVEROLIMUS + STRONG CYP3A4 INHIBITORS — HIGH RISK: azole antifungals, clarithromycin,
     HIV protease inhibitors → 3-10× everolimus exposure → stomatitis/pneumonitis/myelosuppression
  3. PHT/CBZ HIGH RISK (dual mechanism): CYP3A4 induction → reduces everolimus levels 50-90%;
     AND may worsen absence-like seizures in PTEN-ASD children; avoid when on everolimus
  4. VPA + POLG1 — ABSOLUTE CI: PTEN patients not pre-screened; fatal hepatotoxicity risk
  5. TGB — ABSOLUTE CI: NCSE in ASD patients → NCSE invisible without EEG; PTEN macrocephalic
     cortex has mTOR-driven GABA-A subunit dysregulation amplifying GABA desensitisation
  6. Abrupt AED or everolimus withdrawal — ABSOLUTE: seizure cluster risk; taper all slowly

GENETICS:
  Gene:        PTEN (10q23.31) — Phosphatase and Tensin Homolog
  Protein:     403 aa ~54.7 kDa; lipid phosphatase; PIP3→PIP2; tumour suppressor
  Inheritance: Autosomal dominant LOF (haploinsufficiency, pLI=0.97); de novo OR inherited
  pLI:         0.97 (haploinsufficiency causes PHTS; biallelic lethal)
  OMIM:        *601728 (gene); #158350 (Cowden syndrome); #153480 (BRRS); #158350 (LDD)
  Incidence:   1:200,000-250,000 (PHTS); PTEN-ASD with macrocephaly ~1:5,000
  First report: Li J et al. 1997 (Science) — PTEN germline mutations in Cowden disease;
                Liaw D et al. 1997 (Nat Genet) — independent simultaneous report
  Cancer risk: breast 25-50% (female), thyroid 3-38%, endometrial 13-28%, renal 34%, colorectal 9%

KEY STANDARDS:
  ILAE-2022  Genetic epilepsy classification + testing guidelines
  NICE-NG217 Epilepsy management: genetic causes (UK)
  NCCN-2024  PHTS surveillance guidelines (cancer screening)
  ASHG-2013  PTEN testing in ASD + macrocephaly (HC >2 SD) recommendation
  ACMG-AMP-2015 Variant classification for PTEN LOF
  CPIC-POLG1-2023 VPA contraindication in POLG1 variants
  MHRA-VPPP-2021 VPA use in women/girls pregnancy prevention
  WHO-ICF-2019 Function and disability classification
  Li-1997-Science PTEN germline mutation discovery
  Buxbaum-2007 PTEN in ASD macrocephaly
  NICE-NG224-2022 VPA in females

KEY PHARMACOLOGICAL DISTINCTIONS:
  1. EVEROLIMUS mTOR PRECISION: PTEN LOF → AKT-mTOR hyperactivation → neuronal hypertrophy;
     everolimus (FKBP12-rapamycin) inhibits mTORC1 → reverses hypertrophy → seizure reduction
     30-50%; target trough 5-10 ng/mL; monthly TDM initially; mechanistically superior to AEDs alone
  2. CYP3A4 AED INTERACTION: PHT/CBZ/PB are strong CYP3A4 inducers → reduce everolimus by
     50-90% → therapeutic failure; prefer LEV/OXC/LTG when patient on everolimus
  3. CANCER SURVEILLANCE MANDATORY: NCCN protocol: thyroid US annually from age 7; breast MRI
     from 25-30 (mammography from 30-35); colonoscopy from 35; renal US from 40; endometrial
     biopsy from 35 — neurological team MUST coordinate with oncology team
  4. MACROCEPHALY + ASD TRIGGER: HC >2 SD + ASD → PTEN testing (ASHG-2013); yield 17-25%
  5. POLG1 SCREEN: mandatory before VPA regardless of PTEN diagnosis
"""

import random
random.seed(12345)  # reproducible

# ─────────────────────────────────────────────
# PATIENT SAMPLE — 40 synthetic patients
# ─────────────────────────────────────────────
PATIENT_SAMPLE = []
_syndromes = [
    ("PTEN-ASD-Macrocephaly-Epilepsy", 0.40),
    ("BRRS-Bannayan-Riley-Ruvalcaba", 0.25),
    ("Cowden-Syndrome-Adult-Epilepsy", 0.20),
    ("Lhermitte-Duclos-Disease", 0.10),
    ("Phenocopy-mTOR-PIK3CA-negative", 0.05),
]
_sex_dist = {"PTEN-ASD-Macrocephaly-Epilepsy": 0.60, "BRRS-Bannayan-Riley-Ruvalcaba": 0.40,
             "Cowden-Syndrome-Adult-Epilepsy": 0.70, "Lhermitte-Duclos-Disease": 0.55,
             "Phenocopy-mTOR-PIK3CA-negative": 0.50}
_onsets = {"PTEN-ASD-Macrocephaly-Epilepsy": (36, 12), "BRRS-Bannayan-Riley-Ruvalcaba": (18, 8),
           "Cowden-Syndrome-Adult-Epilepsy": (312, 60), "Lhermitte-Duclos-Disease": (240, 80),
           "Phenocopy-mTOR-PIK3CA-negative": (24, 10)}

idx = 0
for syn, prop in _syndromes:
    n = round(40 * prop)
    for i in range(n):
        idx += 1
        female = random.random() < _sex_dist.get(syn, 0.5)
        mu, sd = _onsets[syn]
        onset = max(1, int(random.gauss(mu, sd)))
        dr = (syn in ("PTEN-ASD-Macrocephaly-Epilepsy", "BRRS-Bannayan-Riley-Ruvalcaba",
                      "Lhermitte-Duclos-Disease") and random.random() < 0.45)
        macrocephaly = syn != "Cowden-Syndrome-Adult-Epilepsy" or random.random() < 0.55
        is_case = syn in ("PTEN-ASD-Macrocephaly-Epilepsy", "BRRS-Bannayan-Riley-Ruvalcaba") and random.random() < 0.18
        asd = syn in ("PTEN-ASD-Macrocephaly-Epilepsy",) or (syn == "BRRS-Bannayan-Riley-Ruvalcaba" and random.random() < 0.35)
        on_everolimus = dr and random.random() < 0.60
        on_vpa = not on_everolimus and random.random() < 0.45
        on_lev = random.random() < 0.55
        on_kd = dr and random.random() < 0.35
        polg1_screened = on_vpa or random.random() < 0.75
        cancer_dx = syn == "Cowden-Syndrome-Adult-Epilepsy" and random.random() < 0.40
        ldd_mri = syn == "Lhermitte-Duclos-Disease"
        PATIENT_SAMPLE.append({
            "id": f"PTEN-{idx:03d}",
            "syndrome": syn,
            "sex": "F" if female else "M",
            "age_onset_months": onset,
            "drug_resistant": dr,
            "macrocephaly": macrocephaly,
            "infantile_spasms": is_case,
            "asd_features": asd,
            "on_everolimus": on_everolimus,
            "on_vpa": on_vpa,
            "on_lev": on_lev,
            "on_kd": on_kd,
            "polg1_screened": polg1_screened,
            "cancer_surveillance_enrolled": syn in ("Cowden-Syndrome-Adult-Epilepsy", "BRRS-Bannayan-Riley-Ruvalcaba") or random.random() < 0.70,
            "cancer_dx": cancer_dx,
            "ldd_mri_tiger_stripe": ldd_mri,
        })

while len(PATIENT_SAMPLE) < 40:
    idx += 1
    PATIENT_SAMPLE.append({
        "id": f"PTEN-{idx:03d}",
        "syndrome": "PTEN-ASD-Macrocephaly-Epilepsy",
        "sex": "F" if random.random() < 0.55 else "M",
        "age_onset_months": max(6, int(random.gauss(36, 12))),
        "drug_resistant": random.random() < 0.42,
        "macrocephaly": True,
        "infantile_spasms": random.random() < 0.15,
        "asd_features": True,
        "on_everolimus": random.random() < 0.30,
        "on_vpa": random.random() < 0.40,
        "on_lev": random.random() < 0.50,
        "on_kd": random.random() < 0.25,
        "polg1_screened": random.random() < 0.78,
        "cancer_surveillance_enrolled": random.random() < 0.65,
        "cancer_dx": False,
        "ldd_mri_tiger_stripe": False,
    })

PATIENT_SAMPLE = PATIENT_SAMPLE[:40]

# ─────────────────────────────────────────────
# ETIOLOGY CATALOG
# ─────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "PTEN-ASD-Macrocephaly-Epilepsy (LOF-missense-phosphatase-domain)",
        "n": 16,
        "pct": 40,
        "variants": "LOF missense in phosphatase catalytic core (aa 1-185): C124S, G129E, R130G/Q/L, H123Y; null catalytic activity; de novo in ~75%",
        "phenotype": "Autism (DSM-5) + macrocephaly (HC >2 SD) + focal or absence-like epilepsy; onset 2-8 years; intellectual disability mild-moderate",
        "mechanism": "Phosphatase domain LOF → zero PIP3 dephosphorylation → maximal AKT/mTOR → excess dendritic arborisation + neuronal hypertrophy → focal epileptogenesis",
        "prognosis": "Epilepsy partially controlled in 55%; drug-resistant in 45%; cancer surveillance lifetime mandatory; cognitive trajectory variable",
    },
    {
        "category": "Bannayan-Riley-Ruvalcaba Syndrome (BRRS, LOF-truncating-childhood)",
        "n": 10,
        "pct": 25,
        "variants": "Truncating LOF (frameshift/nonsense): Q171*, R233*, W274*, K342fs; C-terminal or mid-protein; premature stop → haploinsufficiency",
        "phenotype": "Childhood macrocephaly + lipomas + intestinal hamartomatous polyps + speckled penile pigmentation (males) + myopathy; epilepsy onset 6-36 months",
        "mechanism": "Truncating LOF → full mTOR disinhibition; intestinal hamartoma risk from residual mTOR dysregulation in gut epithelium",
        "prognosis": "Higher DRE rate than CS (50%); Cowden overlap increases with age; cancer surveillance from diagnosis",
    },
    {
        "category": "Cowden Syndrome with Epilepsy (CS, LOF-missense-C2-domain-adult)",
        "n": 8,
        "pct": 20,
        "variants": "C2 domain missense (aa 186-351): R233Q, N252S, D326N, C329S; reduce membrane association without abolishing phosphatase; less severe",
        "phenotype": "Adult-onset mucocutaneous lesions (trichilemmomas, papillomatous papules) + thyroid/breast/endometrial cancer + late-onset focal epilepsy; macrocephaly 80%",
        "mechanism": "C2 domain LOF → partial membrane mistargeting → reduced PIP3 dephosphorylation efficiency → partial mTOR disinhibition → milder epilepsy onset (3rd-5th decade)",
        "prognosis": "Epilepsy often AED-controlled in CS (DRE 20%); cancer surveillance critical — leads morbidity/mortality over epilepsy in adulthood",
    },
    {
        "category": "Lhermitte-Duclos Disease (LDD, cerebellar gangliocytoma)",
        "n": 4,
        "pct": 10,
        "variants": "PTEN LOF (any class); somatic second-hit or haploinsufficiency in cerebellar granule cells → dysplastic gangliocytoma; 'tiger-stripe' MRI T2/FLAIR pathognomonic",
        "phenotype": "Dysplastic cerebellar gangliocytoma → cerebellar mass effect → hydrocephalus → headache + ataxia + seizures; seizure type: generalised or cerebellar-onset focal",
        "mechanism": "PTEN LOF → mTOR hyperactivation in granule cells → hypertrophied Purkinje-like cells → gangliocytoma; PI3K/AKT drives cell survival and growth",
        "prognosis": "Surgical resection for hydrocephalus + seizure control; recurrence common; everolimus emerging for inoperable; LDD = pathognomonic Cowden feature (Eng 2000)",
    },
    {
        "category": "Phenocopy-mTOR-PIK3CA-negative (non-PTEN mTORopathy)",
        "n": 2,
        "pct": 5,
        "variants": "Clinical PHTS phenotype; PTEN sequencing negative; DDx: mosaic PTEN not detected in blood; somatic PIK3CA/AKT3/MTOR; methylation silencing; other mTORopathy",
        "phenotype": "ASD + macrocephaly + epilepsy phenotype clinically resembling PTEN; PTEN blood sequencing negative; brain tissue or ultra-deep cfDNA may reveal somatic mosaic variant",
        "mechanism": "Alternative mTOR pathway hyperactivation (PIK3CA GOF, AKT3 GOF, MTOR GOF, TSC1/2 LOF); phenocopies PTEN haploinsufficiency downstream",
        "prognosis": "Same mTOR-directed therapy; differentiation from true PTEN affects cancer surveillance (non-PTEN mTORopathy lower cancer risk)",
    },
]

# ─────────────────────────────────────────────
# SEIZURE TYPES
# ─────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal Impaired Awareness Seizures — FIAS (temporal / frontal)",
        "frequency_pct": 55,
        "eeg": "Focal temporal or frontal rhythmic theta/delta at onset; unilateral evolution; post-ictal focal slow; interictal temporal spikes in 60%",
        "semiology": "Aura (psychic — déjà vu, fear, or autonomic — epigastric rising) → altered awareness + automatisms (oro-alimentary, manual); frontal FIAS — hypermotor, brief, clusters; post-ictal confusion variable",
        "clinical_tip": "Focal temporal FIAS most common in PTEN-ASD; distinguish from typical absence (no 3Hz spike-wave; longer duration; post-ictal confusion; no eye blinking); mTOR-driven cortical dysplasia creates focal epileptogenic zone",
    },
    {
        "type": "Focal to Bilateral Tonic-Clonic Seizures — FBTCS",
        "frequency_pct": 45,
        "eeg": "Focal onset with bilateral spread; generalised tonic activity precedes clonic phase; post-ictal generalised suppression; rhythmic ictal activity evolves over 30-90 sec",
        "semiology": "Focal warning (may be unrecognised) → bilateral stiffening (tonic, 15-20 sec) → rhythmic bilateral jerks (clonic, 30-60 sec) → post-ictal somnolence 15-30 min; often nocturnal in PTEN",
        "clinical_tip": "Mislabelled as 'primary' GTCS without focal investigation; focal MRI lesion or focal EEG onset distinguishes secondary from primary generalised; 3T MRI with cortical dysplasia protocol mandatory",
    },
    {
        "type": "Focal Aware Seizures — FAS (simple partial / aura only)",
        "frequency_pct": 35,
        "eeg": "Brief focal discharge without evolution; spike-wave or rhythmic fast activity limited to focal zone; may be subclinical on scalp EEG",
        "semiology": "Preserved awareness; sensory (tingling, visual spots), psychic (fear, depersonalisation, déjà vu), or autonomic (flushing, palpitation); duration 30 sec - 2 min; no post-ictal confusion; often herald FIAS",
        "clinical_tip": "PTEN-ASD patients may not report FAS subjectively; video-EEG captures behavioural arrest or subtle automatism; underreported in intellectually disabled patients",
    },
    {
        "type": "Absence-like Spells (PTEN-ASD atypical absence)",
        "frequency_pct": 25,
        "eeg": "Irregular 2-3.5 Hz generalised spike-wave (NOT typical 3Hz); background theta slowing; occasionally vertex-predominant or irregular; does NOT show the classic symmetrical 3Hz absence pattern",
        "semiology": "Brief (5-30 sec) staring + unresponsiveness; eyelid fluttering less prominent than typical CAE; no classic hyperventilation-induction; may occur in clusters; often confused with typical absence",
        "clinical_tip": "CRITICAL distinction: NOT typical CAE absence; atypical absence in PTEN-ASD → do NOT use ethosuximide alone (insufficient for focal component); EEG required for classification; OXC or VPA preferred over ESM",
    },
    {
        "type": "Epileptic Spasms / Infantile Spasms — IS (West Syndrome variant)",
        "frequency_pct": 15,
        "eeg": "Hypsarrhythmia in infancy; modified hypsarrhythmia common in macrocephalic cortex (higher amplitude, modified pattern); electrodecrement with spasm burst; background disorganisation",
        "semiology": "Sudden bilateral symmetric arm extension-flexion (salaam) or extensor posturing; clusters on wakening; onset 3-18 months in macrocephalic PTEN; often delays in PTEN-ASD who develop IS",
        "clinical_tip": "PTEN macrocephalic West syndrome: ACTH + VGB simultaneously (UKISS-modified); lower cessation rate (~55%) than idiopathic IS (80%) due to structural substrate; early everolimus consideration post-IS; epilepsy surgery evaluation if asymmetric EEG",
    },
]

# ─────────────────────────────────────────────
# TRIGGERS
# ─────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Sleep Deprivation",
        "pct": 78,
        "mechanism": "Cortical arousal threshold reduction; reduced inhibitory interneuron firing during sleep-deprivation-induced high-frequency oscillations; PTEN mTOR-hyperactive cortex has lower intrinsic inhibitory reserve",
        "management": "Sleep hygiene protocol; melatonin (3-10 mg qhs) — PTEN patients have high ASD-related insomnia prevalence; CLB for nocturnal cluster prevention",
    },
    {
        "trigger": "Emotional Stress",
        "pct": 72,
        "mechanism": "Stress → CRH + cortisol → excitatory modulation of hippocampal and amygdalar circuits; PTEN-ASD patients have exaggerated amygdalar reactivity → lower threshold",
        "management": "Stress management + CBT; family psychoeducation; benzodiazepine rescue (diazepam rectal/midazolam buccal) for stress-triggered clusters",
    },
    {
        "trigger": "Fever / Illness",
        "pct": 68,
        "mechanism": "Fever raises neuronal excitability; PTEN macrocephalic cortex has thinner inhibitory networks → fever-sensitive threshold; febrile seizures in 15% of PTEN-ASD (may be first seizure presentation)",
        "management": "Antipyretic protocol (paracetamol at 38°C threshold, not waiting for 39°C); early BDZ rescue (midazolam 0.3 mg/kg buccal) at febrile seizure onset; written fever plan",
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 65,
        "mechanism": "Subtherapeutic AED trough levels → rebound excitability; abrupt AED level drop particularly destabilising for PTEN mTOR-hyperactive epileptogenic cortex",
        "management": "Adherence support; blister packs or pill dispensers; family carer coaching; AED TDM if compliance uncertain; NEVER abruptly stop everolimus",
    },
    {
        "trigger": "Rapid AED Dose Change (titration or taper)",
        "pct": 48,
        "mechanism": "Rate of AED level change (not just absolute level) triggers withdrawal excitability; AED taper in PTEN must be slower than standard protocols",
        "management": "Taper at ≤10% original dose per 2 weeks; consult neurology before any change; document cluster risk on dose reduction days",
    },
    {
        "trigger": "Hormonal / Catamenial",
        "pct": 30,
        "mechanism": "Perimenstrual oestrogen withdrawal → reduced allopregnanolone → decreased GABA-A positive allosteric modulation → seizure threshold reduction; PTEN-ASD women on OCP — oestrogen-containing preparations may worsen",
        "management": "Seizure diary with menstrual cycle notation; CLB 10-15 mg/day days -3 to +3 of menses (catamenial protocol); avoid combined OCP → consider progesterone-only contraceptive; discuss fertility counselling (AD inheritance: 50% offspring risk)",
    },
    {
        "trigger": "Hormonal Medications (oestrogen-containing OCP)",
        "pct": 25,
        "mechanism": "Oestrogen lowers seizure threshold via down-regulation of GABA-A receptors; exogenous oestrogen in PTEN-ASD women with epilepsy compounds perimenstrual vulnerability; OCP may also interact with AED metabolism (enzyme-inducing AEDs reduce OCP efficacy)",
        "management": "Prefer progesterone-only or non-hormonal contraception; if combined OCP required, increase AED monitoring; counsel on seizure risk and OCP-AED interactions",
    },
    {
        "trigger": "Exercise / Metabolic Stress",
        "pct": 20,
        "mechanism": "Intense exercise → lactic acidosis → pH change → altered ion channel kinetics; dehydration and electrolyte shifts; PTEN patients on KD: exercise-induced ketosis adjustment may fluctuate seizure threshold",
        "management": "Gradual exercise introduction; hydration protocol; avoid hyperthermic environments; pre-exercise BDZ rescue plan if exercise-triggered clusters documented",
    },
]

# ─────────────────────────────────────────────
# TREATMENTS
# ─────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Everolimus (Afinitor / Votubia)",
        "evidence": "Level B (off-label PTEN epilepsy; FDA-approved for TSC-related seizures and angiomyolipoma; Class III evidence for PTEN-epilepsy case series)",
        "dose": "Adult: 2-10 mg/day orally; paediatric: 4.5-7.0 mg/m²/day; target TDM trough 5-10 ng/mL; start LOW 2-5 mg (adult), 2.5 mg/m² (paediatric), titrate monthly to target",
        "moa": "mTORC1 inhibitor (FKBP12-rapamycin complex → allosteric mTOR inhibition → reduces S6K1/4EBP1 phosphorylation → reduced neuronal hypertrophy → synaptic normalisation); directly addresses PTEN LOF molecular consequence",
        "efficacy": "30-50% seizure reduction in PTEN-epilepsy case series (extrapolated from TSC data, n<50 PTEN specific); better outcomes in younger patients with more mTOR-driven pathology",
        "monitoring": "Monthly TDM initially → q3m once stable; monthly CBC + CMP + lipids; annual pulmonary function; wound healing (avoid surgery on everolimus); dental (stomatitis); infection screening; live vaccine exclusion",
        "pten_note": "MECHANISTIC RATIONALE: PTEN LOF is the most direct mTOR disinhibition — PTEN normally inhibits AKT by dephosphorylating PIP3; without PTEN, AKT permanently active → TSC1/2 inactivated → Rheb active → mTORC1 constitutive; everolimus targets this precisely. AED-INTERACTION CRITICAL: PHT/CBZ/PB (strong CYP3A4 inducers) reduce everolimus AUC by 50-90% → therapeutic failure; prefer LEV/OXC/LTG as co-AED when on everolimus.",
    },
    {
        "drug": "Levetiracetam (Keppra) — First-Line AED",
        "evidence": "Level B (focal epilepsy; broad-spectrum; good PTEN-ASD tolerability profile)",
        "dose": "20-60 mg/kg/day (paediatric); 500-3000 mg/day (adult); BID; IV available for SE",
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulation → reduces vesicle priming → decreased neurotransmitter release from hyperactive terminals; independent of mTOR pathway",
        "efficacy": "50-60% responder rate in focal epilepsy; broad spectrum covers FBTCS and absence-like seizures; does NOT reduce everolimus levels (minimal CYP3A4 interaction) — ideal everolimus co-AED",
        "monitoring": "Behavioural side effects in ASD (LEV-induced rage/aggression/irritability — limbic SV2A disinhibition); monitor ASD behavioural baseline; use LEV-XR (extended release) if peak-level behavioural toxicity; switch to perampanel if LEV toxicity confirmed",
        "pten_note": "PREFERRED CO-AED WITH EVEROLIMUS: LEV does not induce CYP3A4 → does not reduce everolimus exposure; maintains therapeutic everolimus trough; behavioural monitoring essential in PTEN-ASD (ASD + LEV = higher aggression risk); LEV-XR reduces peak-level behavioural toxicity.",
    },
    {
        "drug": "Oxcarbazepine (Trileptal) / Carbamazepine (Tegretol) — Focal Seizures",
        "evidence": "Level B (focal epilepsy); HLA-B*1502 screening MANDATORY before use in Asian populations (SJS/TEN risk)",
        "dose": "OXC: 15-30 mg/kg/day paediatric; 600-2400 mg/day adult; TID; CBZ: 15-20 mg/kg/day; BID-TID",
        "moa": "Na⁺-channel (Nav1.1-1.6) blockade → reduces repetitive firing; effective for focal seizures",
        "efficacy": "Good for focal FIAS/FBTCS in PTEN without absence-like component; AVOID if absence-like seizures prominent (Na-channel block may worsen atypical absence)",
        "monitoring": "HLA-B*1502 before initiation (Asian ancestry); hyponatraemia (OXC — SIADH, monthly Na+ for 3 months); rash; CBC; CRITICAL: OXC/CBZ are moderate-strong CYP3A4 inducers → REDUCE everolimus by 30-70% → adjust everolimus dose and TDM when adding/removing OXC/CBZ",
        "pten_note": "CYP3A4 INTERACTION WITH EVEROLIMUS: CBZ is a STRONG CYP3A4 inducer (reduce everolimus ~50%); OXC is a MODERATE inducer (reduce everolimus ~30%); if OXC/CBZ are used with everolimus, everolimus dose must be INCREASED and TDM performed more frequently; alternatively, prefer LEV to avoid this interaction entirely.",
    },
    {
        "drug": "Valproate / Sodium Valproate (Depakote / Epilim)",
        "evidence": "Level B (broad-spectrum, effective for absence-like + GTCS); POLG1 screen MANDATORY before use",
        "dose": "20-40 mg/kg/day paediatric; 600-2500 mg/day adult; BID-TID; IV for SE",
        "moa": "Multiple: Na⁺-channel blockade; GABA-T inhibition → increased synaptic GABA; HDAC inhibition (Class I/IIa) → epigenetic chromatin remodelling; some mTOR pathway modulation via HDAC effects",
        "efficacy": "Broad spectrum: covers absence-like + FBTCS + GTCS in PTEN; HDAC inhibition may provide additive epigenetic benefit (though VPA HDAC effect on mTOR target genes not yet PTEN-specific)",
        "monitoring": "POLG1 screen (7-14 days) BEFORE initiation (Alpers-Huttenlocher CI if biallelic POLG1); LFTs monthly for first 3 months then q6m; CBC; NH₃; VPA TDM (target 50-100 mcg/mL); VPPP programme (UK MHRA); teratogenicity counselling (females of childbearing potential); weight gain; pancreatitis",
        "pten_note": "POLG1 SCREEN NON-NEGOTIABLE: PTEN patients are NOT pre-screened for POLG1; biallelic POLG1 + VPA = Alpers-Huttenlocher syndrome (fatal progressive hepatoencephalopathy); NEVER start VPA without POLG1 result (7-14 day turnaround); bridge with LEV during wait. VPA + EVEROLIMUS: additive (different mechanisms; VPA HDAC effects may modestly complement mTOR inhibition; no pharmacokinetic interaction).",
    },
    {
        "drug": "Ketogenic Diet (KD) — Drug-Resistant Epilepsy",
        "evidence": "Level B (DRE in children and adults; multiple meta-analyses; particularly effective in mTORopathies)",
        "dose": "4:1 fat:protein+carb ratio initially; target beta-hydroxybutyrate (BHB) 3-5 mmol/L; supplement Ca²⁺, Mg²⁺, Zn²⁺, Se²⁺, folate; modified Atkins diet for adults/adolescents",
        "moa": "BHB → HMGCR-independent → adenosine receptor A1 activation → presynaptic K⁺ channel hyperpolarisation → reduced excitability; BHB also suppresses mTORC1 (via AMPK→TSC2→Rheb inhibition) — COMPLEMENTARY to everolimus mTOR suppression; metabolic shift to oxidative phosphorylation reduces glycolytic excitability",
        "efficacy": "50% responder rate in DRE mTORopathies (TSC data extrapolated); mTOR suppression via KD is mechanistically additive to everolimus in PTEN; initiate at AED failure #2, not last resort",
        "monitoring": "Carnitine, selenium, zinc, vitamin D; fasting lipid profile (KD elevates LDL — monitor in PTEN given cancer surveillance context); renal calculi (maintain good hydration); seizure diary; BHB monitoring",
        "pten_note": "PTEN-SPECIFIC mTOR BENEFIT: KD produces BHB → AMPK activation → TSC2 Rheb-GAP restored → Rheb-GDP → mTORC1 reduced; in PTEN LOF, TSC2 is inactivated by AKT; KD BHB-mediated AMPK can re-activate TSC2 (phospho-Ser1387) even in the presence of AKT hyperactivation → partial mTOR suppression independent of AKT; this is the mechanistic rationale for KD + everolimus combination in PTEN.",
    },
    {
        "drug": "ACTH (Acthar Gel / Synacthen) — Infantile Spasms",
        "evidence": "Level A (IS/West Syndrome; UKISS trial; ACTH + VGB superior to ACTH alone)",
        "dose": "ACTH: 150 IU/m²/day IM for 14 days then taper; OR Tetracosactide (Synacthen): 0.5-0.75 mg/day IM (UK/Canada); ALWAYS combine with VGB simultaneously (UKISS-modified protocol)",
        "moa": "MC2R-mediated (melanocortin receptor 2) → adrenal cortisol + neurosteroid production → CRH suppression → reduced CRH-driven hyperexcitability; VGB adds GABA-T inhibition → increased GABA",
        "efficacy": "Hypsarrhythmia cessation ~55-60% in PTEN macrocephalic IS (vs 80% idiopathic IS); structural macrocephalic substrate reduces response; start ACTH+VGB simultaneously; KD consultation at week 4 non-response; lower threshold for epilepsy surgery evaluation (hemispherectomy in asymmetric HME variant)",
        "monitoring": "BP daily (ACTH → hypertension); glucose (ACTH → hyperglycaemia); infection (immunosuppression during ACTH course); electrolytes; growth; VGB ERG mandatory if continuing beyond 6 months (VGB visual field defect risk)",
        "pten_note": "LOWER ACTH RESPONSE IN PTEN IS: macrocephalic cortex with mTOR-driven neuronal hypertrophy is a structural substrate that reduces pharmacological IS cessation rates; simultaneous ACTH+VGB (UKISS-modified) rather than sequential; everolimus may be initiated alongside or post-ACTH in macrocephalic IS — some centres start everolimus at IS diagnosis in PTEN.",
    },
    {
        "drug": "Lamotrigine (Lamictal) — Adjunct Focal/Absence-like",
        "evidence": "Level C (focal epilepsy adjunct; avoid as monotherapy if atypical absence prominent)",
        "dose": "0.15-15 mg/kg/day paediatric (slow titration 25% dose increase q2w); 50-400 mg/day adult; dose halved with VPA; doubled with enzyme inducers",
        "moa": "Na⁺-channel slow-inactivation enhancement → reduces high-frequency neuronal firing; some Ca²⁺ channel inhibition; glutamate release inhibition at presynaptic terminal",
        "efficacy": "Useful adjunct for focal FIAS in PTEN; avoid monotherapy if atypical absence or myoclonic component (Na-channel block may worsen); rash risk (especially with VPA; slow titration mandatory)",
        "monitoring": "Rash (DRESS/SJS — 0.3% rapid titration); titrate SLOWLY (25% increase every 2 weeks); LTG TDM if uncertain (target 3-15 mcg/mL); minimal CYP3A4 interaction → does not significantly affect everolimus levels",
        "pten_note": "SAFE WITH EVEROLIMUS (CYP3A4 neutral): LTG does not meaningfully induce CYP3A4 → does not reduce everolimus trough; safe co-AED choice when OXC/CBZ avoided; avoid LTG monotherapy in PTEN patients with atypical absence-like spells (Na-channel block may paradoxically worsen irregular absence pattern).",
    },
    {
        "drug": "Sirolimus (Rapamycin) — Emerging mTOR Inhibitor",
        "evidence": "Level C (emerging; smaller PTEN-specific dataset than everolimus; LDD case reports; PROS series)",
        "dose": "Adult: 1-5 mg/day orally; paediatric: 1-3 mg/m²/day; TDM target trough 5-15 ng/mL (LDD case reports used higher targets); less validated dosing for epilepsy than everolimus",
        "moa": "Rapamycin (parent compound of everolimus) + FKBP12 → mTORC1 allosteric inhibition (same mechanism as everolimus); longer half-life than everolimus; some mTORC2 inhibition at higher concentrations (distinct from everolimus)",
        "efficacy": "LDD: case reports of tumour stabilisation + seizure reduction with sirolimus/everolimus; PROS: sirolimus reduces tissue overgrowth in PIK3CA-PROS (D'Gama 2017); limited PTEN-epilepsy-specific data",
        "monitoring": "Same monitoring as everolimus (CBC, CMP, lipids, pulmonary, live vaccine CI, CYP3A4 interactions); longer half-life → slower steady-state; monthly TDM; wound healing impairment",
        "pten_note": "SIROLIMUS vs EVEROLIMUS: everolimus preferred (more paediatric neurology data, validated dosing for TSC-epilepsy, more TDM experience); sirolimus considered for LDD (some published data) or if everolimus intolerance; same CYP3A4 and live vaccine interactions apply to sirolimus — not a simpler alternative from a safety standpoint.",
    },
]

# ─────────────────────────────────────────────
# CONTRAINDICATIONS
# ─────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "EVEROLIMUS + LIVE VACCINES",
        "risk_level": "ABSOLUTE",
        "mechanism": "Everolimus inhibits mTORC1 → impairs T-cell and B-cell proliferation (IL-2 signalling suppressed) → immunosuppression → live-attenuated vaccines cause vaccine-strain infection; MMR, varicella, BCG, oral typhoid, yellow fever, live influenza = ABSOLUTE CI",
        "management": "Complete ALL vaccination schedule (including catch-up) BEFORE starting everolimus; document vaccination status; all future vaccines MUST be killed/inactivated only; inform school/daycare about infection risk; contact infection specialist for breakthrough infections or live vaccine exposure (post-exposure prophylaxis may be needed for varicella)",
        "pten_specific": "PTEN paediatric patients often start everolimus in childhood when childhood vaccinations are ongoing; PRIORITISE vaccination completion → then start everolimus; if vaccination schedule incomplete and everolimus urgently needed, vaccinate what can be done, accept gap for live vaccines (MMR/varicella deferred until off everolimus 4 weeks)",
    },
    {
        "drug": "EVEROLIMUS + STRONG CYP3A4 INHIBITORS",
        "risk_level": "HIGH RISK",
        "mechanism": "Everolimus is metabolised by CYP3A4 and P-gp efflux; strong inhibitors (azole antifungals: fluconazole/itraconazole/voriconazole; macrolides: clarithromycin/erythromycin; HIV PI: ritonavir/lopinavir) → 3-10× everolimus AUC → stomatitis (dose-limiting), pneumonitis, myelosuppression, metabolic toxicity",
        "management": "AVOID strong CYP3A4 inhibitors; if antifungal required, prefer fluconazole-sparing topical agents or anidulafungin (minimal CYP3A4); if clarithromycin required for H. pylori, use azithromycin alternative; if strong inhibitor unavoidable, reduce everolimus dose by 50% and increase TDM frequency (check trough 2 days after any inhibitor start/stop); document in pharmacy and anaesthesia records",
        "pten_specific": "PTEN patients on long-term everolimus face antibiotic/antifungal needs throughout life; pharmacist reconciliation at every prescription; create 'everolimus drug interaction' alert in medical record; inform patient/family of interaction risk before any new prescription",
    },
    {
        "drug": "PHT / CBZ / OXC — CYP3A4 INDUCTION + ABSENCE-LIKE WORSENING",
        "risk_level": "HIGH RISK",
        "mechanism": "DUAL MECHANISM: (1) PHARMACOKINETIC: PHT and CBZ are strong CYP3A4 inducers → reduce everolimus AUC by 50-90% → therapeutic failure; OXC is moderate inducer (~30% reduction); (2) PHARMACODYNAMIC: Na-channel blockers (PHT/CBZ/OXC) may worsen atypical absence-like seizures in PTEN-ASD by increasing interneuron Na-channel blockade → disinhibition of absence circuits",
        "management": "If everolimus co-prescribed: strongly prefer LEV or LTG as focal AED (minimal CYP3A4 effect); if OXC/CBZ necessary, increase everolimus dose guided by TDM (may need 2-3× dose increase); NEVER combine PHT with everolimus (too large an induction effect); monitor everolimus TDM within 1 week of adding/changing any CYP3A4-modulating AED",
        "pten_specific": "PHT historically used for SE in ER — DOCUMENT in emergency care plan that PHT reduces everolimus levels and may worsen absence-like seizures in PTEN-ASD; recommend IV LEV (60 mg/kg loading dose) as preferred IV AED for SE in PTEN",
    },
    {
        "drug": "VPA + POLG1 (without screen)",
        "risk_level": "ABSOLUTE",
        "mechanism": "Biallelic POLG1 pathogenic variants → mitochondrial DNA polymerase gamma deficiency → mtDNA depletion; VPA inhibits mitochondrial beta-oxidation → precipitates acute hepatic failure + encephalopathy (Alpers-Huttenlocher syndrome); fatality within weeks if VPA continued",
        "management": "POLG1 screen (7-14 day turnaround) BEFORE VPA initiation in EVERY patient; bridge with LEV+CLB during wait; if POLG1 result unavailable and VPA urgent (SE), document medical urgency, use lowest possible dose, plan screen within 48h; if POLG1 biallelic → VPA permanently contraindicated; alternatives: LEV, CLB, LTG, KD",
        "pten_specific": "PTEN patients not pre-screened for POLG1; VPA is attractive (broad-spectrum, covers absence-like + FBTCS + HDAC epigenetic benefit) but POLG1 risk is universal; NEVER waive this screen on assumption that PTEN is the only genetic issue",
    },
    {
        "drug": "TIAGABINE (TGB) — ABSOLUTE CI",
        "risk_level": "ABSOLUTE",
        "mechanism": "TGB selectively inhibits GAT-1 (GABA transporter 1 = SLC6A1) → excess perisynaptic GABA → GABA-A desensitisation → paradoxical NCSE; PTEN macrocephalic cortex has mTOR-driven alterations in GABA-A subunit expression (excess GABA-A δ subunit at extrasynaptic sites from mTOR-driven dendritic hypertrophy) → amplified GABA-A desensitisation risk",
        "management": "ABSOLUTE contraindication in PTEN epilepsy; if TGB inadvertently administered, perform immediate continuous EEG monitoring (NCSE may present as behavioural change/unresponsiveness only, particularly in PTEN-ASD); IV LEV or BDZ for NCSE treatment; document TGB CI in medical record and emergency care plan",
        "pten_specific": "PTEN-ASD patients have limited verbal communication (40% with significant language impairment) → NCSE from TGB will present ONLY as behavioural change, agitation, or regression without obvious convulsive activity; EEG-confirmed NCSE diagnosis required (not clinical diagnosis alone); NEVER prescribe TGB for any indication in PTEN-ASD",
    },
    {
        "drug": "Abrupt AED / Everolimus Withdrawal",
        "risk_level": "ABSOLUTE",
        "mechanism": "Abrupt AED withdrawal → rebound excitability → seizure cluster or status epilepticus; abrupt everolimus discontinuation → rapid mTOR disinhibition rebound → acute seizure worsening beyond baseline; PTEN hyperactive cortex is particularly sensitive to abrupt medication changes",
        "management": "ALL AED tapers in PTEN should be ≤10% original dose per 2-4 weeks; everolimus taper over ≥4 weeks if discontinuing; never stop any medication abruptly; document taper schedule; emergency plan (rescue midazolam/diazepam) during any taper period; seizure diary review after each dose reduction step",
        "pten_specific": "If transitioning between AEDs, maintain therapeutic overlap for ≥4 weeks before removing old AED; if switching everolimus to sirolimus, maintain 2-week overlap; document 'no abrupt withdrawal' instruction prominently in medical record and family care plan",
    },
]

# ─────────────────────────────────────────────
# MONITORING
# ─────────────────────────────────────────────
MONITORING = [
    {"item": "WES-TRIO / PTEN Targeted Sequencing + Methylation", "frequency": "Once at diagnosis (methylation for LDD/mosaic)", "rationale": "Confirm PTEN LOF; mosaic PTEN may be missed on blood sequencing → reflex ultra-deep sequencing (>500x) or brain tissue; LDD → somatic second-hit on cerebellar PTEN methylation"},
    {"item": "Brain MRI 3T with Cortical Dysplasia Protocol + Volumetry", "frequency": "Diagnosis; q2y monitoring; immediately if new focal deficit", "rationale": "Detect cortical dysplasia, megalencephaly, FCD; LDD 'tiger-stripe' T2/FLAIR pattern; volumetry tracks macrocephaly progression; surgical planning (FCD resection / hemispherectomy)"},
    {"item": "POLG1 Screen (before VPA)", "frequency": "Once before VPA initiation (7-14 day turnaround)", "rationale": "Biallelic POLG1 + VPA = Alpers-Huttenlocher fatal hepatotoxicity; PTEN not pre-screened; mandatory"},
    {"item": "HLA-B*1502 (Asian patients before OXC/CBZ/LTG)", "frequency": "Once before first prescription", "rationale": "SJS/TEN risk in Asian populations; document result; if positive, avoid OXC/CBZ/LTG"},
    {"item": "Everolimus TDM (trough level)", "frequency": "Monthly × 3, then q3m once stable; after any dose change or CYP3A4 drug change within 2-7 days", "rationale": "Target trough 5-10 ng/mL for neurological; toxic >15 ng/mL; subtherapeutic <3 ng/mL; CYP3A4 interactions profoundly affect levels"},
    {"item": "CBC + CMP + Lipids + Urinalysis (everolimus monitoring)", "frequency": "Monthly × 3, then q3m", "rationale": "Everolimus: myelosuppression (CBC), hepatotoxicity (LFTs), hyperlipidaemia (lipids), proteinuria (urinalysis), hyperglycaemia (glucose); oral ulcers/stomatitis — dose-limiting toxicity"},
    {"item": "Pulmonary Function (spirometry) + Chest X-ray", "frequency": "Annual; immediately if respiratory symptoms", "rationale": "Everolimus pneumonitis (non-infectious; 5-15%); ground-glass opacities on CT; withhold everolimus if Grade ≥2 pneumonitis; prednisolone if Grade 3-4"},
    {"item": "Cancer Surveillance (NCCN PHTS Protocol)", "frequency": "Thyroid US: annually from age 7; Breast MRI: from 25-30y; Mammography: from 30-35y; Colonoscopy: from 35y (q5y); Renal US: from 40y; Endometrial biopsy: from 35y (or with abnormal bleeding)", "rationale": "PHTS cancer risks: breast 25-50% (female), thyroid 3-38%, endometrial 13-28%, colorectal 9%, renal 34%; neurological team MUST coordinate with oncology; cancer leads morbidity in CS adults"},
    {"item": "Dermatology + Mucocutaneous Exam", "frequency": "Annual from diagnosis", "rationale": "Trichilemmomas (pathognomonic Cowden), papillomatous papules, acral keratoses, lipomas, sclerotic fibromas; dermatology referral for biopsy if uncertain; hamartoma surveillance"},
    {"item": "Neuropsychology (ASD + Cognitive)", "frequency": "At diagnosis; q2y developmental surveillance; ADOS-2/ADI-R, Bayley-III (infants), VABS-3", "rationale": "PTEN-ASD: 60-80% have formal ASD diagnosis; cognitive trajectory monitoring; IQ stable or declining; early intervention (ABA, speech, OT) improves outcomes"},
    {"item": "Seizure Diary + SUDEP / Alarm Monitoring", "frequency": "Continuous diary; SUDEP safety review annually", "rationale": "DRE + nocturnal FBTCS → SUDEP risk; bed-alarm (Emfit/Embrace2); never sleep alone in DRE; prone position avoidance; rescue BDZ protocol"},
    {"item": "Vaccination Status + Everolimus Immunisation Record", "frequency": "Before everolimus start; annually thereafter", "rationale": "Live vaccine absolute CI on everolimus; document vaccination status; annual killed influenza; pneumococcal; COVID-19 (mRNA/killed — not live)"},
    {"item": "Reproductive / Genetic Counselling", "frequency": "At diagnosis of PTEN LOF; before any pregnancy", "rationale": "AD inheritance: 50% offspring risk; preconception PTEN status of partner; prenatal testing (CVS/amniocentesis); VPA teratogenicity counselling (VPPP UK); progesterone-only or non-hormonal contraception preferred"},
    {"item": "VPA TDM + LFTs + NH₃ (if on VPA)", "frequency": "Monthly × 3, then q6m; immediately if vomiting/encephalopathy", "rationale": "VPA hepatotoxicity (especially first 6 months); VPA hyperammonaemia (NH₃ if encephalopathic); target VPA 50-100 mcg/mL"},
]

# ─────────────────────────────────────────────
# LIFECYCLE
# ─────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Prenatal / Neonatal (0 months)",
        "key_issues": "Macrocephaly on antenatal ultrasound (head circumference >97th centile); non-immune hydrocephalus (rare); alert paediatric neurology; PTEN testing of one parent if macrocephaly detected prenatally",
        "management": "Antenatal genetics consultation; PTEN germline testing (blood); post-delivery head circumference and MRI (3T); no PTEN-specific AED in neonatal period; if IS in neonatal period, ACTH+VGB; neonatal genetics referral",
    },
    {
        "stage": "Infancy (0-24 months)",
        "key_issues": "Progressive macrocephaly; infantile spasms (15% of PTEN-ASD/BRRS infants); hypotonia; developmental delay; ASD red flags; first seizure evaluation",
        "management": "Monthly HC measurement; if IS: ACTH+VGB simultaneously (UKISS-modified); EEG; 3T MRI; ASD early intervention (ABA from 12-18m); dietary assessment; POLG1 before VPA if needed; everolimus consideration in macrocephalic IS",
    },
    {
        "stage": "Early Childhood — Focal Epilepsy Onset (2-8 years)",
        "key_issues": "Focal epilepsy onset (FIAS, absence-like spells, FBTCS); ASD diagnosis formalised; macrocephaly stabilises; DRE declared at AED failure #2; PTEN cancer surveillance initiation (thyroid US from age 7); everolimus initiation discussion",
        "management": "LEV or OXC first-line (HLA-B*1502 before OXC in Asian patients); add VPA (post-POLG1 screen) if absence-like + GTCS; PTEN surveillance: thyroid US from age 7; if DRE at AED #2 → everolimus + vaccination completion; KD consideration; neuropsychology assessment; ADOS-2",
    },
    {
        "stage": "School Age — DRE Management (8-18 years)",
        "key_issues": "DRE establishment; everolimus + KD combination; cancer surveillance intensification; puberty → catamenial pattern; social/educational support for ASD+epilepsy; SUDEP risk (nocturnal FBTCS)",
        "management": "Everolimus TDM-guided; KD at AED failure #2-3; CLB for catamenial component; seizure diary; school safety plan (supervised swimming, falls protection); SUDEP alarm monitoring; BRRS cancer surveillance (annual thyroid, bowel symptoms); adolescent transition planning",
    },
    {
        "stage": "Adulthood — Cancer Surveillance + Epilepsy Management (18-40 years)",
        "key_issues": "Cowden cancer surveillance intensification (breast, endometrial, renal, colorectal); fertility counselling (AD 50% offspring risk); everolimus long-term management; employment + driving (seizure-free requirements); catamenial epilepsy management",
        "management": "Oncology co-management (NCCN PHTS protocol); annual thyroid US; breast MRI from 25-30y; colonoscopy from 35y; everolimus maintained; VPA VPPP if female; non-hormonal or progesterone-only contraception; preconception PTEN counselling; genetic testing for partner; VNS if DRE not surgically amenable",
    },
    {
        "stage": "Late Adulthood — Oncology + Neurology Co-Management (40+ years)",
        "key_issues": "Cancer morbidity (breast, endometrial) leads over epilepsy in CS; Lhermitte-Duclos surveillance (cerebellar MRI); cognitive trajectory; everolimus dose optimisation (renal function); reduced metabolic clearance of AEDs",
        "management": "Annual renal US (renal cancer from 40y); annual dermatology (mucocutaneous changes); everolimus adjusted for age-related renal clearance reduction; cognitive neuropsychology; family carer support; AED doses reviewed (PK changes with age); if LDD symptomatic → surgical evaluation + everolimus/sirolimus",
    },
]

# ─────────────────────────────────────────────
# CONCEPTS (15 key clinical concepts)
# ─────────────────────────────────────────────
CONCEPTS = [
    {
        "concept": "PTEN-PHTS-mTORopathy",
        "explanation": "PTEN Hamartoma Tumour Syndrome encompasses Cowden syndrome, BRRS, PTEN-ASD, LDD, and Proteus-like conditions — all caused by PTEN LOF. PTEN is the principal brake on the PI3K-AKT-mTOR axis; LOF creates constitutive mTORC1 hyperactivation ('mTORopathy'), positioning PTEN alongside TSC1/2, MTOR, PIK3CA, and AKT3 in the mTORopathy spectrum. Epilepsy occurs in 15-35% of PHTS patients.",
    },
    {
        "concept": "PTEN-Lipid-Phosphatase-PIP3-PIP2",
        "explanation": "PTEN's critical activity is dephosphorylating phosphatidylinositol-3,4,5-trisphosphate (PIP3) → PIP2. PIP3 is generated by PIK3CA from PIP2; it recruits and activates AKT at the membrane. PTEN is the molecular reversal switch. LOF → PIP3 accumulates at the inner leaflet → AKT membrane recruitment sustained → AKT T308 (PDK1) and S473 (mTORC2) phosphorylation → TSC1/2 phosphorylated/inactivated → Rheb-GTP → mTORC1 constitutive → S6K1/4EBP1 → excess translation + neuronal hypertrophy.",
    },
    {
        "concept": "PTEN-Macrocephaly-Diagnostic-Clock",
        "explanation": "Macrocephaly (HC >2 SD above mean, or >98th centile) is the most sensitive clinical trigger for PTEN testing. ASHG 2013 recommendation: offer PTEN testing to any child with ASD + macrocephaly HC >2 SD (yield 17-20%). At HC >3 SD (>99.7th centile) + ASD, yield rises to 25%. Unlike most DEE genes where macrocephaly is absent, PTEN LOF causes symmetric cortical and subcortical overgrowth from early neurogenesis. HC measurement at EVERY visit is clinical surveillance for PTEN.",
    },
    {
        "concept": "PTEN-ASD-Epilepsy-Triad",
        "explanation": "The PTEN-ASD presentation triad: (1) Autism spectrum disorder (DSM-5 criteria met in 60-80% of PTEN-ASD); (2) Macrocephaly (HC >97th centile in 90%+); (3) Epilepsy (20-35% of PTEN-ASD). All three relate to mTOR hyperactivation: excess dendritic arborisation → ASD-like reduced synaptic pruning; cortical overgrowth → macrocephaly; focal epileptogenic cortical dysplasia → focal seizures. The triad distinguishes PTEN from idiopathic ASD, Angelman (small HC), and other DEE genes.",
    },
    {
        "concept": "Cowden-Bannayan-LDD-PHTS-Spectrum",
        "explanation": "PHTS spectrum: Cowden syndrome (CS) — adult mucocutaneous + cancer; BRRS — childhood hamartomas + macrocephaly; PTEN-ASD — ASD + macrocephaly + epilepsy; LDD — cerebellar gangliocytoma (PTEN pathognomonic — any patient with LDD should receive PTEN germline testing). These are NOT separate diseases but a continuous spectrum sharing PTEN LOF mechanism. Clinical presentation depends on: variant class (null vs partial LOF), somatic second-hit timing, tissue-specific X-inactivation, modifier genes.",
    },
    {
        "concept": "Everolimus-mTOR-PTEN-Precision-Rationale",
        "explanation": "Everolimus precision rationale in PTEN: PTEN LOF → constitutive mTORC1 → S6K1/4EBP1 → neuronal hypertrophy; everolimus blocks mTORC1 directly (FKBP12-rapamycin). This is the most direct pharmacological targeting of the PTEN LOF consequence. FDA approval for TSC-related seizures (same pathway, different entry point) provides the regulatory framework; off-label use in PTEN is mechanistically superior rationale to that in TSC (PTEN is the direct upstream inhibitor of mTOR). Target trough 5-10 ng/mL (neurological); 10-15 ng/mL for TSC renal angiomyolipoma.",
    },
    {
        "concept": "Everolimus-Live-Vaccine-ABSOLUTE-CI",
        "explanation": "Mechanism: everolimus inhibits mTORC1 in T-cells and B-cells → IL-2 signalling suppressed → lymphoproliferation impaired → immunosuppression. Live-attenuated vaccines (MMR, varicella, BCG, oral typhoid, yellow fever, live influenza) use replication-competent organisms; in immunosuppressed host → uncontrolled replication → vaccine-strain infection (measles encephalitis, disseminated VZV, BCG-osis). Strategy: complete ALL scheduled vaccinations BEFORE everolimus; all future vaccines must be inactivated/killed; inform daycare/school about contact precautions.",
    },
    {
        "concept": "Everolimus-CYP3A4-Drug-Interactions",
        "explanation": "Everolimus is a CYP3A4 and P-glycoprotein substrate. Strong CYP3A4 inhibitors (voriconazole, itraconazole, fluconazole, clarithromycin, ritonavir) increase everolimus AUC 3-10×. Strong CYP3A4 inducers (PHT, CBZ, PB, rifampin) reduce everolimus AUC 50-90%. Clinical impact: PHT/CBZ as AEDs can completely negate everolimus efficacy without dose adjustment. AED-everolimus interaction algorithm: (1) LEV/LTG/OXC-mild preferred (minimal CYP3A4); (2) if OXC needed, increase everolimus dose 1.5-2× and TDM; (3) NEVER use PHT with everolimus without TDM-guided dose tripling.",
    },
    {
        "concept": "PHT-CBZ-Absence-Like-HIGH-RISK",
        "explanation": "Na-channel blockers (PHT/CBZ/OXC) carry HIGH RISK in PTEN patients with atypical absence-like seizures (25% of cohort). Mechanism: Na-channel block reduces interneuron firing → disinhibition of thalamo-cortical resonance circuits → worsening of irregular 2-3.5 Hz generalised spike-wave (atypical absence). PHT additionally causes CYP3A4-mediated everolimus level reduction. DUAL REASON to avoid in PTEN-ASD with absence-like component: pharmacodynamic (absence worsening) + pharmacokinetic (everolimus reduction).",
    },
    {
        "concept": "TGB-ABSOLUTE-NCSE-PTEN-ASD",
        "explanation": "TGB (tiagabine, GAT-1/SLC6A1 inhibitor) → excess perisynaptic GABA → GABA-A receptor desensitisation → paradoxical NCSE. PTEN-specific amplification: mTOR-driven dendritic hypertrophy → abnormal GABA-A subunit composition (excess δ-subunit extrasynaptic expression from mTOR-driven gene dysregulation) → amplified desensitisation. ASD-specific diagnostic trap: PTEN-ASD patients have limited verbal output → NCSE presents only as behavioural change, agitation, or apparent regression — clinically invisible without EEG. EEG-mandatory diagnosis.",
    },
    {
        "concept": "POLG1-VPA-Mandatory-Screen",
        "explanation": "POLG1 encodes mitochondrial DNA polymerase gamma; biallelic LOF → mitochondrial DNA depletion; VPA inhibits mitochondrial beta-oxidation → precipitates Alpers-Huttenlocher syndrome (acute progressive hepatoencephalopathy → death). PTEN patients are NOT pre-screened for POLG1. Screen: 7-14 day turnaround; bridge with LEV+CLB; NEVER start VPA without result. Alpers-Huttenlocher is IRREVERSIBLE once initiated — fatality common within weeks of VPA exposure in biallelic POLG1.",
    },
    {
        "concept": "Cancer-Surveillance-Cowden-Mandatory",
        "explanation": "NCCN PHTS guidelines mandate lifelong cancer surveillance: thyroid US annually from age 7; breast MRI from 25-30y (mammography from 30-35y) in females; colonoscopy from 35y (q5y); renal US from 40y (annual); endometrial biopsy from 35y (or with abnormal uterine bleeding). LT cancer risks: breast 25-50% (female), thyroid 3-38%, endometrial 13-28%, colorectal 9%, renal 34%. In adulthood, cancer mortality/morbidity EXCEEDS epilepsy morbidity in PHTS — neurological care teams must formally hand over cancer surveillance to oncology/genetics at age 18.",
    },
    {
        "concept": "HLA-B1502-OXC-CBZ-Asian",
        "explanation": "HLA-B*1502 allele (primarily Southeast and East Asian populations: Han Chinese, Thai, Filipino, Vietnamese, Malaysian) confers 10× risk of Stevens-Johnson syndrome (SJS) and toxic epidermal necrolysis (TEN) with carbamazepine and oxcarbazepine. FDA mandated HLA-B*1502 screening before CBZ/OXC in patients of Asian ancestry (2007). Also applies to lamotrigine (weaker association). PTEN patients of Asian descent: HLA-B*1502 test result must be documented and flagged in medical record before ANY arene-oxide-forming AED.",
    },
    {
        "concept": "mTOR-Pathway-mTORC1-S6K1-4EBP1",
        "explanation": "mTORC1 (mechanistic Target Of Rapamycin Complex 1) activates two main downstream effectors: S6 kinase 1 (S6K1) → ribosomal protein S6 phosphorylation → enhanced cap-dependent mRNA translation + cell growth; and 4E-BP1 (eukaryotic initiation factor 4E-binding protein 1) → when phosphorylated, releases eIF4E → cap-dependent translation initiation. In PTEN LOF: constitutive mTORC1 → excess S6K1/4EBP1 → excess dendritic translation → enlarged dendritic tree + spine dysgenesis → epileptogenic + ASD synapse. Everolimus inhibits both.",
    },
    {
        "concept": "SUDEP-DRE-PTEN-Risk",
        "explanation": "Sudden Unexpected Death in Epilepsy (SUDEP) risk is elevated in PTEN DRE (drug-resistant epilepsy). Risk factors cumulative: DRE + nocturnal FBTCS + male sex + young adult + prone sleeping + no alarm monitoring. PTEN DRE: everolimus + KD addresses mechanistic cause; VNS for further seizure burden reduction; safety measures mandatory: bed alarm (Emfit/Embrace2/Smart Monitor), never sleep alone in DRE, supine sleep positioning, rescue BDZ within reach. Annual SUDEP risk discussion with family documented in medical record.",
    },
]

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "PTEN testing trigger: HC >2 SD above mean + ASD", "value": "ASHG-2013; yield 17-25%; HC at every visit is clinical surveillance"},
    {"threshold": "Everolimus TDM target (neurological)", "value": "Trough 5-10 ng/mL; toxic >15 ng/mL; subtherapeutic <3 ng/mL"},
    {"threshold": "Everolimus dose reduction: trough >15 ng/mL", "value": "50% dose reduction; recheck TDM 2 weeks later"},
    {"threshold": "POLG1 screen turnaround", "value": "7-14 days; never start VPA without result"},
    {"threshold": "HLA-B*1502 before CBZ/OXC in Asian patients", "value": "Once; positive = AVOID CBZ/OXC/LTG; document permanently"},
    {"threshold": "ACTH IS cessation check (PTEN macrocephalic West)", "value": "Day 14 hypsarrhythmia cessation rate ~55% (vs 80% idiopathic); if non-response at day 14 → KD + everolimus consideration"},
    {"threshold": "KD initiation: AED failure #2", "value": "Not last resort; mTOR suppression via BHB complementary to everolimus in PTEN DRE"},
    {"threshold": "Cancer surveillance — thyroid US", "value": "Annually from age 7 in all PHTS"},
    {"threshold": "Cancer surveillance — breast MRI (female)", "value": "Annually from 25-30y; mammography from 30-35y (NCCN PHTS)"},
    {"threshold": "Everolimus + CYP3A4 inhibitor: TDM check", "value": "Within 2-7 days of starting/stopping any CYP3A4 modulator; expect 3-10× change with strong inhibitors"},
    {"threshold": "VPA TDM target", "value": "50-100 mcg/mL (total); NH₃ if encephalopathy; LFTs monthly × 3 then q6m"},
    {"threshold": "SUDEP alarm: nocturnal FBTCS threshold", "value": "≥1 nocturnal FBTCS in past year → bed alarm mandatory; DRE → never sleep alone"},
]

# ─────────────────────────────────────────────
# STANDARDS
# ─────────────────────────────────────────────
STANDARDS = [
    {"standard": "ILAE-2022", "description": "ILAE classification of epilepsies and genetic testing recommendations"},
    {"standard": "NICE-NG217", "description": "Epilepsy management in children and adults including genetic causes (UK, 2022)"},
    {"standard": "NCCN-PHTS-2024", "description": "National Comprehensive Cancer Network: Cowden/PHTS surveillance guidelines — lifelong cancer screening protocol"},
    {"standard": "ASHG-2013", "description": "American Society of Human Genetics: PTEN testing in ASD with macrocephaly (HC >2 SD) — professional guideline"},
    {"standard": "Buxbaum-2007-AJHumGenet", "description": "Buxbaum JD et al. 2007: PTEN germline mutations in autism with macrocephaly — epidemiological framework for PTEN-ASD"},
    {"standard": "Li-1997-Science", "description": "Li J et al. 1997 (Science): PTEN germline mutations identified in Cowden disease — discovery paper"},
    {"standard": "CPIC-POLG1-2023", "description": "CPIC guideline for VPA use in POLG1 variant carriers — mandatory pre-VPA screen"},
    {"standard": "MHRA-VPPP-2021", "description": "Medicines and Healthcare products Regulatory Agency: Valproate Pregnancy Prevention Programme — mandatory for females of childbearing potential"},
    {"standard": "ACMG-AMP-2015", "description": "ACMG/AMP variant classification standards for PTEN LOF pathogenicity assessment"},
    {"standard": "NICE-NG224-2022", "description": "NICE guideline NG224: Valproate use in females — prescribing restrictions and counselling requirements"},
    {"standard": "FDA-Everolimus-TSC-2012", "description": "FDA approval of everolimus (Afinitor) for TSC-related seizures and angiomyolipoma — regulatory basis for off-label PTEN use"},
    {"standard": "WHO-ICF-2019", "description": "International Classification of Functioning, Disability and Health — function and disability assessment framework"},
]

# ─────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────
REFERENCES = [
    {
        "ref": "Li-1997-Science",
        "full": "Li J et al. (1997). PTEN, a putative protein tyrosine phosphatase gene mutated in human brain, breast, and prostate cancer. Science, 275(5308), 1943-1947.",
        "key_finding": "Discovery of PTEN as a tumour suppressor; germline mutations in Cowden disease — founding paper for PTEN field",
    },
    {
        "ref": "Liaw-1997-NatGenet",
        "full": "Liaw D et al. (1997). Germline mutations of the PTEN gene in Cowden disease, an inherited breast and thyroid cancer syndrome. Nature Genetics, 16(1), 64-67.",
        "key_finding": "Independent simultaneous PTEN discovery in Cowden disease; established PTEN-cancer connection in germline",
    },
    {
        "ref": "Buxbaum-2007-AJHumGenet",
        "full": "Buxbaum JD et al. (2007). Mutation screening of the PTEN gene in patients with autism spectrum disorders and macrocephaly. American Journal of Human Genetics, 80(3), 506-511.",
        "key_finding": "PTEN mutations in 17% of ASD with macrocephaly (HC >2.5 SD); established ASHG screening criterion",
    },
    {
        "ref": "Butler-2005-Neurology",
        "full": "Butler MG et al. (2005). Subset of individuals with autism spectrum disorders and extreme macrocephaly associated with germline PTEN tumour suppressor gene mutations. Journal of Medical Genetics, 42(4), 318-321.",
        "key_finding": "PTEN germline mutations in autism + macrocephaly; first systematic neurological phenotyping of PTEN-ASD",
    },
    {
        "ref": "Ciaccio-2014-EJHG",
        "full": "Ciaccio C et al. (2014). Germline mutations in PTEN: time to consider epilepsy in the clinical phenotype. European Journal of Human Genetics, 22(10), 1241-1247.",
        "key_finding": "First systematic epilepsy phenotyping in PTEN PHTS; epilepsy in 15-25%; focal onset predominant; everolimus proposed",
    },
    {
        "ref": "Marchetti-2020-Epilepsia",
        "full": "Marchetti A et al. (2020). mTOR inhibitors for epilepsy in PTEN hamartoma tumor syndrome: a systematic review. Epilepsia, 61(12), 2695-2707.",
        "key_finding": "Systematic review of mTOR inhibitor (everolimus/sirolimus) use in PTEN epilepsy: 30-50% seizure reduction; safety profile consistent with TSC data",
    },
]


# ─────────────────────────────────────────────
# PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────
def get_overview():
    total = len(PATIENT_SAMPLE)
    dr_n = sum(1 for p in PATIENT_SAMPLE if p["drug_resistant"])
    female_n = sum(1 for p in PATIENT_SAMPLE if p["sex"] == "F")
    on_everolimus = sum(1 for p in PATIENT_SAMPLE if p["on_everolimus"])
    on_kd = sum(1 for p in PATIENT_SAMPLE if p["on_kd"])
    on_vpa = sum(1 for p in PATIENT_SAMPLE if p["on_vpa"])
    on_lev = sum(1 for p in PATIENT_SAMPLE if p["on_lev"])
    is_n = sum(1 for p in PATIENT_SAMPLE if p["infantile_spasms"])
    asd_n = sum(1 for p in PATIENT_SAMPLE if p["asd_features"])
    macrocephaly_n = sum(1 for p in PATIENT_SAMPLE if p["macrocephaly"])
    cancer_surv_n = sum(1 for p in PATIENT_SAMPLE if p["cancer_surveillance_enrolled"])
    polg1_n = sum(1 for p in PATIENT_SAMPLE if p["polg1_screened"])
    ldd_n = sum(1 for p in PATIENT_SAMPLE if p["ldd_mri_tiger_stripe"])
    avg_onset_m = round(sum(p["age_onset_months"] for p in PATIENT_SAMPLE) / total)

    etiology_dist = {item["category"]: {"n": item["n"], "pct": item["pct"]} for item in ETIOLOGY_CATALOG}
    seizure_dist = [{"type": s["type"].split("(")[0].strip().split("—")[0].strip(), "pct": s["frequency_pct"]} for s in SEIZURE_TYPES]
    trigger_dist = [{"trigger": t["trigger"], "pct": t["pct"]} for t in TRIGGERS]

    return {
        "dashboard": "PTEN Epilepsy (PTEN Hamartoma Tumour Syndrome / mTORopathy / Cowden / BRRS / ASD-Macrocephaly / Everolimus-mTOR-Precision / Alpelisib-Emerging / Cancer-Surveillance / 10q23.31)",
        "gene": "PTEN (10q23.31) — Phosphatase and Tensin Homolog; lipid phosphatase PIP3→PIP2; mTOR brake; 403 aa ~54.7 kDa; tumour suppressor",
        "inheritance": "Autosomal dominant LOF (haploinsufficiency); de novo or inherited; pLI=0.97",
        "omim": "*601728 (gene); #158350 (Cowden syndrome / LDD); #153480 (BRRS); PHTS encompasses multiple OMIM phenotypes",
        "cohort_size": total,
        "female_n": female_n,
        "female_pct": round(female_n / total * 100),
        "mean_onset_months": avg_onset_m,
        "drug_resistant_n": dr_n,
        "drug_resistant_pct": round(dr_n / total * 100),
        "infantile_spasms_n": is_n,
        "infantile_spasms_pct": round(is_n / total * 100),
        "asd_features_n": asd_n,
        "asd_features_pct": round(asd_n / total * 100),
        "macrocephaly_n": macrocephaly_n,
        "macrocephaly_pct": round(macrocephaly_n / total * 100),
        "ldd_n": ldd_n,
        "ldd_pct": round(ldd_n / total * 100),
        "on_everolimus_n": on_everolimus,
        "on_everolimus_pct": round(on_everolimus / total * 100),
        "on_kd_n": on_kd,
        "on_kd_pct": round(on_kd / total * 100),
        "on_vpa_n": on_vpa,
        "on_vpa_pct": round(on_vpa / total * 100),
        "on_lev_n": on_lev,
        "on_lev_pct": round(on_lev / total * 100),
        "cancer_surveillance_n": cancer_surv_n,
        "cancer_surveillance_pct": round(cancer_surv_n / total * 100),
        "polg1_screened_n": polg1_n,
        "polg1_screened_pct": round(polg1_n / total * 100),
        "etiology_distribution": etiology_dist,
        "seizure_type_distribution": seizure_dist,
        "trigger_distribution": trigger_dist,
        "key_contraindications": [
            "EVEROLIMUS + LIVE VACCINES — ABSOLUTE CI (immunosuppression → vaccine-strain infection; complete vaccination BEFORE everolimus)",
            "EVEROLIMUS + STRONG CYP3A4 INHIBITORS — HIGH RISK (azoles/clarithromycin/HIV-PI → 3-10× everolimus levels → stomatitis/pneumonitis)",
            "PHT/CBZ HIGH RISK — DUAL MECHANISM (CYP3A4 induction → reduces everolimus + worsens absence-like seizures in PTEN-ASD)",
            "VPA without POLG1 SCREEN — ABSOLUTE CI (Alpers-Huttenlocher fatal hepatotoxicity; PTEN not pre-screened)",
            "TGB — ABSOLUTE CI (NCSE; PTEN-ASD patients have limited verbal output → NCSE clinically invisible without EEG)",
            "ABRUPT AED/EVEROLIMUS WITHDRAWAL — ABSOLUTE CI (acute mTOR rebound + seizure cluster; taper all medications slowly)",
        ],
        "mtor_pathway_note": "PTEN is the apex mTOR brake — PTEN LOF → constitutive AKT → TSC1/2 inactivated → mTORC1 constitutive → S6K1/4EBP1 → neuronal hypertrophy + epileptogenesis; everolimus mechanistically addresses this directly (same logic as TSC but PTEN is upstream of TSC1/2)",
        "cancer_surveillance_note": "PHTS carries 10-50× background cancer risk (breast, thyroid, endometrial, renal, colorectal); cancer morbidity exceeds epilepsy morbidity in CS adults; NCCN PHTS surveillance protocol mandatory — neurological team MUST coordinate with oncology/genetics team",
    }


def get_breakdown():
    return {
        "etiology_catalog": ETIOLOGY_CATALOG,
        "patient_sample": PATIENT_SAMPLE[:15],
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
