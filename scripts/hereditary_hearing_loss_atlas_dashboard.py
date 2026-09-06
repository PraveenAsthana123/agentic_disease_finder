#!/usr/bin/env python3
"""Hereditary-Hearing-Loss-Atlas — Complete 8-Gene Hereditary Sensorineural Hearing Loss Atlas
GJB2    (Connexin 26; 226 aa; 13q12.11; AR DFNB1 / AD DFNA3;
          Most common cause of hereditary SNHL — 50% of AR congenital severe–profound SNHL;
          35delG founder Europe; 235delC East Asian; 167delT Ashkenazi Jewish;
          V37I mild–moderate — Asian populations; test GJB6 deletion before labelling monoallelic) ·
SLC26A4 (Pendrin; 780 aa; 7q22.3; AR;
          DFNB4 — non-syndromic bilateral SNHL with enlarged vestibular aqueduct (EVA);
          Pendred syndrome — EVA + goitre + partial thyroid organification defect;
          IVS7-2A>G founder Europe; L236P+T416P digenic with EVA; avoid head trauma;
          Perchlorate discharge test >10% abnormal; CI effective; EVA fluctuates with trauma/infection) ·
OTOF    (Otoferlin; 1997 aa; 2p23.3; AR;
          DFNB9 — auditory neuropathy spectrum disorder (ANSD) without peripheral neuropathy;
          DPOAE present (outer hair cells intact), ABR absent/grossly abnormal;
          Q829X Spain/Turkey founder; cochlear implant EXCELLENT outcome unlike neuropathy-ANSD;
          Otoferlin mediates synaptic vesicle fusion at inner hair cell ribbon synapse) ·
COCH    (Cochlin; 550 aa; 14q12-q13; AD;
          DFNA9 — adult-onset bilateral progressive SNHL + vestibular (Meniere-like);
          P51S and V66G LCCL domain founders Belgium/Netherlands; no specific treatment;
          Eosinophilic deposits in spiral ligament and membranous labyrinth on histology;
          Annual audiogram + vestibular testing; hearing aids; COCH deafness not reversible) ·
TMC1    (Transmembrane Channel-Like 1; 757 aa; 9q21.13; AR DFNB7/11 / AD DFNA36;
          Mechanoelectrical transduction channel at stereocilia tips of hair cells;
          M298K AD dominant-negative — Beethoven mouse model; Tunisian/Iranian AR founder;
          Gene therapy (AAV-TMC1) in phase I/II — first TMC1 gene therapy trials active 2024;
          AR: pre-lingual profound deafness; AD: progressive onset 2nd decade) ·
MYO7A   (Myosin VIIA; 2215 aa; 11q13.5; AR DFNB2 / Usher 1B;
          Most common Usher type 1 gene (Usher 1B) — deaf at birth + vestibular areflexia + RP;
          DFNB2 non-syndromic AR SNHL (milder alleles, no RP);
          Usher 1B: CI effective early (before 12 months ideal); annual ERG for RP;
          Vestibular areflexia — delayed walking (18 months+), no compensation; low-vision aids) ·
CDH23   (Cadherin 23; 3354 aa; 10q22.1; AR;
          Usher 1D — stereocilia tip-link cadherin; hypomorphic alleles → DFNB12 (non-syndromic);
          Null alleles → Usher 1D (profound deafness + vestibular areflexia + RP);
          c.7903C>T p.Arg2636Ter founder Finland; CI effective; no vestibular compensation;
          ERG mandatory — RP often appears teens/early 20s after deafness diagnosis) ·
PCDH15  (Protocadherin 15; 1955 aa; 10q21.1; AR;
          Usher 1F — R245X Ashkenazi Jewish founder (1 in 148 carriers);
          Most common Usher type 1 gene in Ashkenazi Jewish and some South Asian populations;
          Tip-link lower component; DFNB23 non-syndromic (milder alleles);
          Early CI; ophthalmology annually; vestibular PT; PCDH15 panel before general Usher panel in Ashkenazi)
320-patient aggregate cohort (8 × 40, seeds 1430–1437)
"""

import random

SEED_BASE = 1430

HHL_GENES = [
    # ── GJB2 — Most common AR SNHL ──
    {
        "gene": "GJB2",
        "protein": "Connexin 26 (Gap Junction Protein Beta-2)",
        "alias": (
            "GJB2; OMIM gene 121011; DFNB1 #220290, DFNA3 #601544; 13q12.11; 226 aa; ~26 kDa; "
            "Most common cause of hereditary SNHL — ~50% of AR congenital severe–profound SNHL worldwide; "
            "35delG (c.35delG) European founder — 1/31 carriers Northern European; "
            "235delC East Asian founder — most common Asian GJB2 variant; "
            "167delT Ashkenazi Jewish founder; V37I mild–moderate (Asian); "
            "Forms hexameric connexon gap junctions in cochlear supporting cells — K+ recycling; "
            "GJB2 LOF → K+ accumulation in endolymph → hair cell death; "
            "GJB6 deletion (del(GJB6-D13S1830)) must be excluded before calling monoallelic GJB2"
        ),
        "aa": "226 aa",
        "kDa": "~26 kDa",
        "locus": "13q12.11",
        "omim_gene": 121011,
        "omim_disease": 220290,
        "inheritance": "AR (DFNB1) — biallelic LOF; AD (DFNA3) — dominant-negative; biallelic most common clinical scenario",
        "gene_class": (
            "GJB2 encodes connexin 26, a gap junction protein essential for potassium recycling in the "
            "cochlear supporting cell network. Following mechanotransduction, K+ enters hair cells through "
            "apical MET channels and must be recycled back to the endolymph via connexin gap junctions "
            "in supporting cells. Biallelic GJB2 LOF prevents K+ recycling, causing hair cell death and "
            "congenital profound SNHL. GJB2 is the most common cause of hereditary congenital SNHL in "
            "most populations. The GJB6 gene (connexin 30) occupies the same chromosomal region — a "
            "large GJB6 deletion that removes part of GJB2's regulatory region can act as a second allele, "
            "so testing must include GJB6 deletion analysis before calling a GJB2 monoallelic case."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("35delG homozygous (c.35delG/c.35delG) — European", 0.28),
            ("35delG + second GJB2 pathogenic variant (compound heterozygous) — European", 0.30),
            ("235delC compound heterozygous — East Asian", 0.15),
            ("V37I compound heterozygous — mild–moderate (Asian)", 0.12),
            ("167delT Ashkenazi Jewish compound heterozygous", 0.08),
            ("Novel/other biallelic GJB2 pathogenic variant", 0.07),
        ],
        "age_onset_years_range": (0, 1),
        "sex_ratio_M": 0.50,
        "rates": {
            "congenital_profound_snhl":         0.75,
            "congenital_severe_snhl":           0.15,
            "mild_moderate_snhl_v37i":          0.10,
            "bilateral_symmetric":              0.95,
            "unilateral_asymmetric":            0.05,
            "ci_implanted":                     0.60,
            "excellent_ci_outcome":             0.55,
            "hearing_aid_user":                 0.35,
            "gjb6_deletion_coallele":           0.15,
            "monoallelic_initial_report":       0.18,
            "vestibular_dysfunction":           0.05,
            "associated_peripheral_neuropathy": 0.00,
            "skin_manifestation_keratoderma":   0.03,
            "normal_mri_cochlea":               0.92,
            "speech_delay_preverbal_diagnosis": 0.40,
        },
        "hallmarks": [
            "Most common hereditary SNHL gene worldwide — GJB2 must be tested FIRST in all congenital SNHL",
            "35delG European founder — 1 in 31 carrier frequency Northern Europeans; offer to siblings",
            "GJB6 large deletion exclusion MANDATORY before labelling case monoallelic GJB2",
            "V37I: mild–moderate SNHL not profound — will be missed if only severe–profound panel tested",
            "Cochlear implant outcomes EXCELLENT — predict language outcomes comparable to hearing peers if early",
            "MRI cochlea typically normal — no structural abnormality unlike SLC26A4",
            "K+ recycling gap junction defect — no treatment beyond amplification/CI and rehabilitation",
        ],
        "treatment_alerts": [
            "GJB6 DELETION EXCLUSION: test del(GJB6-D13S1830) before reporting monoallelic GJB2 — may be second allele",
            "NEWBORN HEARING SCREEN: bilateral refer → GJB2 sequencing + GJB6 deletion before age 3 months",
            "CI TIMING: implant before 12 months for optimal language outcomes (critical period)",
            "V37I ALLELE: mild variant — counsel about progressive risk; annual audiogram; hearing aids first",
            "CASCADE TESTING: all first-degree relatives; siblings 25% affected if both parents carriers",
            "SYNDROME EXCLUSION: GJB2 non-syndromic; if skin or corneal findings → consider GJB2 dominant-negative DFNA3",
            "GENETIC COUNSELLING: AR 25% recurrence per pregnancy; prenatal diagnosis available",
        ],
        "primary_treatment": (
            "No cochlear therapy — GJB2 SNHL is not reversible. "
            "Amplification: hearing aids for mild–moderate (V37I); CI for severe–profound (standard first-line). "
            "CI timing: before 12 months strongly preferred for language outcomes. "
            "Rehabilitation: auditory-verbal therapy, speech therapy, sign language if family chooses. "
            "Annual audiogram (hearing aid users); monitor for progressive loss in V37I. "
            "Genetic counselling: cascade testing all first-degree relatives; prenatal diagnosis. "
            "Multidisciplinary: ENT, audiology, genetics, speech-language pathology, education."
        ),
    },

    # ── SLC26A4 — Pendred Syndrome / DFNB4 ──
    {
        "gene": "SLC26A4",
        "protein": "Pendrin (Sulfate-Chloride Anion Transporter)",
        "alias": (
            "SLC26A4; OMIM gene 605646; DFNB4 #600791, Pendred #274600; 7q22.3; 780 aa; ~86 kDa; "
            "Second most common cause of AR hereditary SNHL — enlarged vestibular aqueduct (EVA) hallmark; "
            "Pendred syndrome: bilateral SNHL + EVA + diffuse goitre + thyroid organification defect; "
            "IVS7-2A>G (c.919-2A>G) — most common European pathogenic splice variant; "
            "L236P + T416P digenic compound heterozygote common; H723R Asian founder; "
            "EVA: CT/MRI mandatory — enlargement >1.5 mm at midpoint lateral semicircular canal; "
            "AVOID HEAD TRAUMA — Valsalva, contact sports, barotrauma cause fluctuating SNHL"
        ),
        "aa": "780 aa",
        "kDa": "~86 kDa",
        "locus": "7q22.3",
        "omim_gene": 605646,
        "omim_disease": 600791,
        "inheritance": "AR — biallelic LOF; very rarely one pathogenic variant + one EVA-causing allele (digenic with SLC26A4 region)",
        "gene_class": (
            "SLC26A4 encodes pendrin, an anion exchanger expressed in the endolymph-producing epithelium "
            "of the inner ear (endolymphatic sac) and in thyroid follicular cells. In the cochlea, pendrin "
            "mediates Cl-/HCO3- exchange maintaining endolymph homeostasis. LOF causes endolymphatic hydrops "
            "and the characteristic EVA. Thyroid pendrin participates in iodide organification — its loss "
            "produces a partial organification defect detectable by perchlorate discharge test, though most "
            "patients are euthyroid. The EVA is the radiological hallmark — CT shows >1.5 mm at midpoint. "
            "Hearing loss fluctuates with minor head trauma, Valsalva manoeuvres, or infections in EVA."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("IVS7-2A>G (c.919-2A>G) compound heterozygous — European", 0.30),
            ("H723R compound heterozygous — East Asian", 0.20),
            ("L236P + T416P compound heterozygous — digenic", 0.15),
            ("p.Leu445Trp compound heterozygous", 0.10),
            ("Other biallelic SLC26A4 pathogenic variants", 0.15),
            ("SLC26A4 monoallelic + EVA (possible somatic/regulatory second allele)", 0.10),
        ],
        "age_onset_years_range": (0, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "severe_profound_snhl":             0.70,
            "moderate_severe_snhl":             0.20,
            "mild_moderate_snhl":               0.10,
            "bilateral_symmetric_eva":          0.85,
            "unilateral_eva":                   0.10,
            "fluctuating_hearing_loss":         0.55,
            "goitre_pendred_syndrome":          0.40,
            "euthyroid_despite_goitre":         0.35,
            "thyroid_organification_defect":    0.40,
            "perchlorate_discharge_positive":   0.38,
            "vestibular_dysfunction":           0.30,
            "ci_implanted":                     0.55,
            "hearing_aid_user":                 0.35,
            "head_trauma_fluctuation_event":    0.45,
            "valsalva_triggered_loss":          0.30,
        },
        "hallmarks": [
            "EVA on temporal bone CT/MRI: >1.5 mm at midpoint — MANDATORY imaging in all bilateral SNHL children",
            "AVOID HEAD TRAUMA — contact sports, diving, trampolines cause irreversible step-wise HL progression",
            "Pendred syndrome: EVA + goitre + organification defect — goitre may appear in 2nd–3rd decade",
            "Perchlorate discharge test: >10% discharge = partial organification defect (Pendred hallmark)",
            "Hearing fluctuates with minor head trauma or infection — acute management: steroids not proven but used",
            "CI effective — perform before 3 years in profound cases; hearing aids in moderate–severe",
            "SLC26A4 monoallelic cases: 35% have EVA — other allele may be in regulatory or deep intronic region",
        ],
        "treatment_alerts": [
            "HEAD TRAUMA PROHIBITION: no contact sports, diving, trampolines, Valsalva manoeuvres — written instruction every appointment",
            "IODINE SUPPLEMENTATION NOT RECOMMENDED: most Pendred patients are euthyroid; iodine can worsen goitre",
            "THYROID MONITORING: annual TSH; thyroidectomy only for large symptomatic goitre (rare)",
            "EVA IMAGING: CT temporal bones mandatory in all children with bilateral SNHL before labelling idiopathic",
            "FLUCTUATION MANAGEMENT: high-dose steroids (1 mg/kg/day prednisolone × 2 weeks) for acute deterioration — evidence limited",
            "CI TIMING: implant before 3 years in profound cases; EVA does not preclude CI",
            "CASCADE TESTING: siblings 25% affected; prenatal diagnosis available",
        ],
        "primary_treatment": (
            "Avoid head trauma strictly — written activity restrictions provided at every visit. "
            "Amplification: hearing aids for moderate–severe; CI for severe–profound. "
            "Thyroid: annual TSH monitoring; treatment only if hypothyroid (uncommon in Pendred). "
            "Acute fluctuation: oral prednisolone trial (evidence limited); prevent re-injury. "
            "Education: dedicated deaf education support; sign language if family chooses. "
            "Genetic counselling: cascade family testing; 25% recurrence per pregnancy."
        ),
    },

    # ── OTOF — Auditory Neuropathy ANSD / DFNB9 ──
    {
        "gene": "OTOF",
        "protein": "Otoferlin (Synaptic Vesicle Fusion Protein at Inner Hair Cell Ribbon Synapse)",
        "alias": (
            "OTOF; OMIM gene 603681; DFNB9 #601071; 2p23.3; 1997 aa; ~227 kDa; "
            "Auditory neuropathy spectrum disorder (ANSD) — outer hair cell function preserved; "
            "DPOAE present; ABR absent or grossly abnormal; CM (cochlear microphonic) present; "
            "Q829X (c.2485C>T) Spain and Turkish founder; p.Arg1939Gln mild-temperature-sensitive; "
            "Otoferlin is the Ca2+ sensor for synaptic vesicle fusion at inner hair cell ribbon synapse; "
            "CI EXCELLENT outcome — unlike ANSD from neuropathy (AN2-8 genes); neural ANSD CI less effective; "
            "Temperature-sensitive variants: hearing worsens with fever — may recover with temperature normalisation"
        ),
        "aa": "1997 aa",
        "kDa": "~227 kDa",
        "locus": "2p23.3",
        "omim_gene": 603681,
        "omim_disease": 601071,
        "inheritance": "AR — biallelic LOF most severe; compound heterozygous with mild allele → partial function",
        "gene_class": (
            "Otoferlin mediates Ca2+-triggered exocytosis at the ribbon synapse of cochlear inner hair cells. "
            "Unlike neuronal synapses using synaptotagmin, inner hair cells require otoferlin as the primary "
            "Ca2+ sensor for high-rate, temporally precise synaptic vesicle fusion. OTOF LOF ablates inner "
            "hair cell synaptic transmission while leaving outer hair cells (and thus DPOAEs) intact — the "
            "physiological signature of auditory neuropathy. Critically, OTOF-ANSD responds well to cochlear "
            "implantation because the defect is pre-neural (synaptic, not in the auditory nerve itself), "
            "distinguishing it from neuropathy-type ANSD (e.g. AIFM1, DIAPH3) where CI outcomes are variable. "
            "Temperature-sensitive OTOF variants cause transient worsening of hearing with fever — characteristic."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("Q829X (c.2485C>T) compound heterozygous — Spanish/Turkish founder", 0.30),
            ("Biallelic null OTOF variants — severe–profound ANSD", 0.35),
            ("Compound heterozygous including mild allele — temperature-sensitive phenotype", 0.15),
            ("p.Arg1939Gln compound heterozygous — temperature-sensitive SNHL", 0.10),
            ("Novel/other biallelic OTOF pathogenic variants", 0.10),
        ],
        "age_onset_years_range": (0, 2),
        "sex_ratio_M": 0.50,
        "rates": {
            "ansd_phenotype_dpoae_present_abr_absent": 0.95,
            "cochlear_microphonic_present":             0.90,
            "severe_profound_snhl":                    0.80,
            "moderate_snhl":                           0.15,
            "temperature_sensitive_worsening":         0.20,
            "fever_triggered_acute_loss":              0.18,
            "ci_implanted":                            0.65,
            "excellent_ci_outcome_open_set_speech":    0.60,
            "hearing_aid_user_partial":                0.25,
            "bilateral_symmetric":                     0.90,
            "normal_mri_auditory_nerve":               0.88,
            "speech_delay_preverbal":                  0.55,
            "auditory_processing_difficulty":          0.70,
            "peripheral_neuropathy_absent":            0.98,
        },
        "hallmarks": [
            "ANSD audiological profile: DPOAEs PRESENT (outer hair cells intact), ABR absent — inner hair cell synaptic defect",
            "OTOF-ANSD: cochlear implant EXCELLENT — unlike nerve-level ANSD; CI is first-line for severe–profound OTOF-ANSD",
            "Temperature-sensitive phenotype: hearing worsens dramatically with fever — fever plan for all OTOF patients",
            "Cochlear microphonic (CM) present on ABR — confirms hair cell function; tells you ANSD is pre-neural",
            "Distinguish from neurological ANSD: no peripheral neuropathy, no MRI nerve hypoplasia in OTOF",
            "FM system + hearing aids may partially help in milder OTOF cases while awaiting CI decision",
        ],
        "treatment_alerts": [
            "ANSD AUDIOLOGICAL ASSESSMENT: always perform DPOAE + ABR + CM — do NOT assume ANSD without CM confirmation",
            "CI IS FIRST-LINE for severe–profound OTOF-ANSD — do not withhold CI due to 'ANSD' label; CI works well",
            "DIFFERENTIATE ANSD CAUSE: OTOF (pre-neural, CI excellent) vs neuropathy ANSD (post-neural, CI variable) — critical for counselling",
            "TEMPERATURE MANAGEMENT: written fever plan every patient — antipyretics early; fever can cause acute profound hearing loss",
            "HEARING AIDS: limited benefit in profound ANSD (signal not transmitted); FM system + soundfield amplification may help",
            "NO FM-ONLY APPROACH: FM system insufficient for severe–profound OTOF-ANSD; CI timing should not be delayed",
            "MRI NORMAL: no CI contraindication from OTOF; cochlear anatomy normal",
        ],
        "primary_treatment": (
            "Cochlear implantation: preferred treatment for severe–profound OTOF-ANSD. "
            "Implant as early as possible (6–12 months); outcomes comparable to GJB2-SNHL CI outcomes. "
            "Temperature management: antipyretics immediately at fever onset — written plan; "
            "hearing may deteriorate acutely with fever then recover as temperature normalises. "
            "Hearing aids: FM systems and hearing aids for mild–moderate cases or while awaiting CI. "
            "Rehabilitation: auditory-verbal therapy post-CI; speech-language pathology. "
            "Genetic counselling: AR 25% recurrence; prenatal diagnosis available."
        ),
    },

    # ── COCH — Adult Progressive SNHL / DFNA9 ──
    {
        "gene": "COCH",
        "protein": "Cochlin (LCCL Domain Extracellular Matrix Glycoprotein)",
        "alias": (
            "COCH; OMIM gene 603196; DFNA9 #601369; 14q12-q13; 550 aa; ~63 kDa; "
            "AD adult-onset bilateral progressive SNHL + vestibular dysfunction (Meniere-like); "
            "P51S and V66G — LCCL domain founders Belgium, Netherlands, USA families; "
            "W117R, G87W, I109N — other AD LCCL pathogenic variants; "
            "Age at onset: 2nd–4th decade; profound by 5th–6th decade; "
            "Eosinophilic deposits in spiral ligament and stria vascularis on histology; "
            "No disease-modifying treatment — hearing aids then CI; vestibular rehabilitation"
        ),
        "aa": "550 aa",
        "kDa": "~63 kDa",
        "locus": "14q12-q13",
        "omim_gene": 603196,
        "omim_disease": 601369,
        "inheritance": "AD — dominant pathogenic variants in LCCL domain cause protein misfolding and accumulation",
        "gene_class": (
            "Cochlin is the most abundant protein in the inner ear extracellular matrix (stria vascularis, "
            "spiral ligament, spiral ganglion). Its N-terminal LCCL domain (Limulus factor C, Coch-5b2, "
            "LGL1) is the mutational hotspot — missense variants cause protein misfolding and eosinophilic "
            "deposit accumulation in inner ear connective tissues. The mechanism parallels amyloidosis — "
            "misfolded cochlin aggregates progressively destroy inner ear architecture. The resulting "
            "DFNA9 phenotype includes both sensorineural hearing loss (progressive from high frequencies) "
            "and endolymphatic hydrops-like vestibular dysfunction (episodic vertigo, progressive vestibular "
            "loss). There is no specific treatment beyond amplification and vestibular management."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("P51S (c.151C>T) LCCL domain — Belgian/Dutch founder", 0.35),
            ("V66G (c.197T>G) LCCL domain — Netherlands/USA", 0.25),
            ("W117R (c.349T>C) LCCL domain", 0.15),
            ("G87W (c.259G>T) LCCL domain", 0.12),
            ("Other LCCL domain AD pathogenic variant", 0.13),
        ],
        "age_onset_years_range": (15, 50),
        "sex_ratio_M": 0.50,
        "rates": {
            "progressive_bilateral_snhl":           0.98,
            "high_frequency_onset":                 0.90,
            "profound_snhl_by_5th_decade":          0.70,
            "vestibular_dysfunction_meniere_like":  0.75,
            "episodic_vertigo":                     0.60,
            "progressive_vestibular_loss":          0.65,
            "tinnitus":                             0.75,
            "aural_fullness_pressure":              0.40,
            "hearing_aid_user":                     0.75,
            "ci_implanted":                         0.35,
            "vestibular_rehabilitation_required":   0.55,
            "positive_family_history_ad":           0.85,
            "normal_mri_inner_ear":                 0.70,
        },
        "hallmarks": [
            "Adult-onset progressive SNHL in a family — always sequence COCH when AD SNHL + vestibular features",
            "Meniere-like vestibular attacks (episodic vertigo) + progressive SNHL — COCH is a major cause",
            "No disease-modifying treatment — counsel family proactively about inevitable progression",
            "Hearing aids early (as soon as functional loss); CI for profound bilateral (outcomes good)",
            "Vestibular rehabilitation: vestibular physio for progressive vestibular loss; fall prevention",
            "Annual audiogram: track progression; fitting amplification at functional threshold (40 dB HL)",
        ],
        "treatment_alerts": [
            "COCH IS INCURABLE: no disease-modifying therapy — set honest expectations from diagnosis",
            "HEARING AID FITTING: fit early when loss crosses 35–40 dB HL — do not wait for severe loss",
            "VESTIBULAR REHABILITATION: enrol in vestibular physio programme for episodic vertigo/progressive loss",
            "FALL PREVENTION: significant bilateral vestibular loss — Cawthorne-Cooksey exercises; bone-anchored systems",
            "CI CANDIDACY: when profound bilateral — COCH CI outcomes are good (neural function preserved)",
            "CASCADE TESTING: 50% first-degree relative risk (AD); genetic counselling; prenatal testing available",
            "MRI INNER EAR: endolymphatic space changes may be seen; normal cochlear nerve; CI not contraindicated",
        ],
        "primary_treatment": (
            "Amplification: hearing aids fitted promptly when thresholds cross 40 dB HL (high-frequency first). "
            "Cochlear implant: when bilateral profound (standard CI indications). COCH CI outcomes excellent. "
            "Vestibular: vestibular rehabilitation physiotherapy for episodic vertigo; fall-prevention programme. "
            "Acute vertigo: antiemetics short-term; no long-term betahistine evidence for genetic COCH. "
            "Monitoring: annual audiogram; annual vestibular assessment (caloric, video-HIT). "
            "Genetic counselling: 50% offspring risk (AD inheritance); cascade family testing."
        ),
    },

    # ── TMC1 — Mechanoelectrical Transduction / DFNB7/11 + DFNA36 ──
    {
        "gene": "TMC1",
        "protein": "Transmembrane Channel-Like Protein 1 (MET Channel Subunit)",
        "alias": (
            "TMC1; OMIM gene 606706; DFNB7 #600974, DFNB11, DFNA36 #606705; 9q21.13; 757 aa; ~87 kDa; "
            "Mechanoelectrical transduction (MET) channel subunit at stereocilia tips; "
            "M298K (c.892A>C) — AD DFNA36 dominant-negative (Beethoven mouse); "
            "R34X Tunisian AR founder; H604R Iranian founder; "
            "AR DFNB7/11: pre-lingual profound deafness; AD DFNA36: progressive 2nd decade onset; "
            "Gene therapy target: AAV-TMC1 inner ear injection — Phase I/II trials active 2024; "
            "TMC1 and TMC2 both contribute to MET channel pore; TMC1 dominant in adult hair cells"
        ),
        "aa": "757 aa",
        "kDa": "~87 kDa",
        "locus": "9q21.13",
        "omim_gene": 606706,
        "omim_disease": 600974,
        "inheritance": "AR (DFNB7/11) — biallelic LOF; AD (DFNA36) — dominant-negative M298K pore mutation",
        "gene_class": (
            "TMC1 (transmembrane channel-like 1) forms the core pore of the mechanoelectrical transduction "
            "(MET) channel at the tips of cochlear hair cell stereocilia. Tip-link tension deflects stereocilia, "
            "gating the MET channel to allow K+ and Ca2+ entry — the first step in converting mechanical "
            "sound vibration to electrical signal. TMC1 is the dominant channel subunit in adult cochlear "
            "hair cells (replacing TMC2, which predominates neonatally). Biallelic LOF abolishes MET "
            "transduction → profound pre-lingual deafness. The AD dominant-negative M298K variant "
            "(Beethoven mouse) interferes with channel pore selectivity → progressive high-frequency loss. "
            "TMC1 is a leading gene therapy target: AAV-TMC1 injection into mouse endolymph restores "
            "hearing, and Phase I/II human clinical trials are underway as of 2024."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("R34X (c.100C>T) biallelic — Tunisian AR founder", 0.25),
            ("H604R (c.1811A>G) biallelic — Iranian AR founder", 0.20),
            ("Other biallelic TMC1 null variants — AR profound", 0.30),
            ("M298K (c.892A>C) heterozygous — AD DFNA36 progressive", 0.15),
            ("Novel AR compound heterozygous TMC1 variants", 0.10),
        ],
        "age_onset_years_range": (0, 20),
        "sex_ratio_M": 0.50,
        "rates": {
            "congenital_profound_snhl_ar":          0.70,
            "progressive_snhl_onset_2nd_decade_ad": 0.15,
            "moderate_severe_snhl":                 0.10,
            "bilateral_symmetric":                  0.95,
            "ci_implanted":                         0.60,
            "good_ci_outcome":                      0.55,
            "hearing_aid_user":                     0.30,
            "vestibular_normal":                    0.88,
            "tinnitus":                             0.20,
            "positive_family_history_ar":           0.60,
            "consanguineous_family":                0.30,
            "gene_therapy_trial_eligible_2024":     0.10,
            "normal_mri_cochlea":                   0.90,
        },
        "hallmarks": [
            "MET channel subunit — TMC1 is the direct transduction channel; audiogram flat-profound (AR) or high-frequency progressive (AD M298K)",
            "Beethoven mouse (M298K): AD progressive SNHL from 2nd decade — monitor annually from childhood",
            "Gene therapy target: first TMC1 human trial 2024 — check ClinicalTrials.gov for eligibility",
            "AR founder variants: R34X Tunisian, H604R Iranian — targeted testing before full sequencing",
            "CI first-line for AR profound — no vestibular component; excellent CI outcomes",
            "TMC2 provides neonatal backup — neonatal hearing screen may be normal in some TMC1 AR cases",
        ],
        "treatment_alerts": [
            "NEONATAL HEARING SCREEN FALSE NEGATIVE: TMC2 backup in neonates — screen may be normal; ABR retest at 3 months",
            "AD M298K MONITORING: annual audiogram from age 10; fit hearing aids early (before 35 dB HL profound)",
            "GENE THERAPY ELIGIBILITY: check ClinicalTrials.gov NCT numbers for AAV-TMC1 trials — counsel appropriately",
            "CI TIMING: AR cases — implant before 12 months; CI is current standard of care",
            "FOUNDER VARIANTS: R34X (Tunisian), H604R (Iranian) — test specifically in at-risk populations before full sequencing",
            "CASCADE TESTING: AR 25% sibling risk; AD 50% offspring risk; prenatal testing available",
        ],
        "primary_treatment": (
            "CI: standard first-line for AR DFNB7/11 profound deafness — implant before 12 months. "
            "Hearing aids: for AD DFNA36 progressive loss (M298K) — fit early at functional threshold. "
            "Annual audiogram: mandatory for AD M298K from age 10. "
            "Gene therapy: check clinical trial eligibility (AAV-TMC1 in Phase I/II 2024); "
            "counsel about trial status without raising unrealistic expectations. "
            "Rehabilitation: auditory-verbal therapy post-CI; speech-language pathology. "
            "Genetic counselling: AR 25% / AD 50% recurrence; cascade testing."
        ),
    },

    # ── MYO7A — Usher Syndrome Type 1B / DFNB2 ──
    {
        "gene": "MYO7A",
        "protein": "Myosin VIIA (Unconventional Myosin — Stereocilia + Retinal Pigment Epithelium)",
        "alias": (
            "MYO7A; OMIM gene 276903; DFNB2 #600060, Usher1B #276900, DFNA11; 11q13.5; 2215 aa; ~254 kDa; "
            "Most common Usher type 1 gene — Usher 1B (>50% Usher 1 in most populations); "
            "Usher syndrome type 1: profound SNHL at birth + vestibular areflexia + RP (onset teens); "
            "DFNB2: AR non-syndromic SNHL without RP — milder alleles, some residual MYO7A function; "
            "DFNA11: AD progressive SNHL — dominant-negative; "
            "CI effective in Usher 1B — implant before 12 months; RP is not treatable; "
            "Vestibular areflexia: delayed walking >18 months; no vestibular compensation; "
            "Annual ERG mandatory from diagnosis — RP onset typically teens"
        ),
        "aa": "2215 aa",
        "kDa": "~254 kDa",
        "locus": "11q13.5",
        "omim_gene": 276903,
        "omim_disease": 276900,
        "inheritance": "AR (Usher 1B + DFNB2) — biallelic LOF; AD (DFNA11) — dominant-negative; X-linked excluded (11q)",
        "gene_class": (
            "Myosin VIIA is an unconventional myosin motor critical for stereocilia development and maintenance "
            "in cochlear hair cells, and for retinal pigment epithelium (RPE) phagocytosis of photoreceptor "
            "outer segments. In the cochlea, MYO7A maintains ankle-link structure and locates protein "
            "complexes within stereocilia. In the RPE, MYO7A is required for melanosome transport and "
            "efficient outer segment phagocytosis. Biallelic null alleles abolish both cochlear and retinal "
            "MYO7A function → Usher type 1B (deaf/blind/vestibular areflexia). Milder alleles preserving "
            "partial MYO7A function produce DFNB2 (non-syndromic deafness without RP). The tri-syndrome "
            "of Usher type 1 (deafness + RP + vestibular) is progressive blindness on a background of "
            "congenital deafness — the dual sensory loss requires coordinated deaf-blind management."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("Biallelic MYO7A null — Usher 1B (profound SNHL + RP + vestibular areflexia)", 0.55),
            ("MYO7A compound heterozygous mild allele — DFNB2 (non-syndromic)", 0.20),
            ("MYO7A truncating homozygous — Usher 1B founder (e.g. p.Arg212His Spain)", 0.12),
            ("AD MYO7A — DFNA11 progressive (dominant-negative)", 0.08),
            ("Novel biallelic MYO7A", 0.05),
        ],
        "age_onset_years_range": (0, 3),
        "sex_ratio_M": 0.50,
        "rates": {
            "congenital_profound_snhl":             0.82,
            "moderate_severe_snhl_dfnb2":           0.15,
            "vestibular_areflexia_bilateral":       0.78,
            "delayed_walking_over_18_months":       0.70,
            "retinitis_pigmentosa":                 0.75,
            "rp_onset_before_age_25":               0.70,
            "night_blindness_nyctalopia":           0.68,
            "visual_field_constriction":            0.60,
            "ci_implanted":                         0.70,
            "good_ci_speech_outcome":               0.65,
            "low_vision_aids_required":             0.45,
            "deaf_blind_services_required":         0.30,
            "vestibular_pt_required":               0.70,
            "balance_impaired_dark":                0.72,
            "positive_erg_abnormality":             0.78,
        },
        "hallmarks": [
            "Usher 1B triad: profound congenital deafness + bilateral vestibular areflexia + RP (onset teens) — all three required for Usher 1",
            "DELAYED WALKING >18 months: vestibular areflexia → balance on visual cues; dark = fall risk",
            "ERG mandatory from diagnosis — RP onset typically teens; annual ERG to track; predict legal blindness",
            "CI EARLY: implant before 12 months; CI excellent for hearing; hearing is the better-preserved sense",
            "Low vision services: as RP progresses — magnifiers, orientation-mobility training, deaf-blind services",
            "No RP treatment currently (2024) — gene therapy in preclinical; counsel honestly without false hope",
        ],
        "treatment_alerts": [
            "USHER 1 OPHTHALMOLOGY: annual ERG from diagnosis — do not wait for visual symptoms; catch RP before legal blindness",
            "VESTIBULAR PT: balance physiotherapy from diagnosis; swimming + dark environments = fall hazard",
            "CI TIMING: implant before 12 months in Usher 1B — CI is the most impactful intervention; hearing saves deafblind independence",
            "GENETIC TESTING: 5-gene Usher 1 panel (MYO7A, CDH23, PCDH15, SANS, HARMONIN) simultaneously — more efficient than sequential",
            "LOW VISION REFERRAL: when visual field <10° or acuity <6/60 — deaf-blind services; orientation-mobility training",
            "NO VITAMIN A MEGADOSE: no longer recommended for Usher RP (safety concerns; insufficient evidence)",
            "AVOID BRIGHT LIGHT + UV: photoreceptor stress; sunglasses with UV protection mandatory",
        ],
        "primary_treatment": (
            "Hearing: CI before 12 months — most important single intervention in Usher 1B. "
            "Auditory-verbal therapy post-CI; sign language (visual communication backup as RP progresses). "
            "Vestibular: physiotherapy from infancy; swimming lessons with close supervision; fall prevention; "
            "vestibular rehabilitation as bilateral loss progresses. "
            "Vision: annual ERG + visual fields from diagnosis; low-vision services when RP progresses; "
            "orientation-mobility training; UV-blocking sunglasses. "
            "Deaf-blind services: as RP + deafness progress — dedicated dual-sensory support. "
            "Genetic counselling: AR 25% recurrence; cascade testing all first-degree relatives; prenatal diagnosis."
        ),
    },

    # ── CDH23 — Usher Syndrome Type 1D / DFNB12 ──
    {
        "gene": "CDH23",
        "protein": "Cadherin 23 (Stereocilia Tip-Link Upper Component)",
        "alias": (
            "CDH23; OMIM gene 605516; DFNB12 #601386, Usher1D #601067; 10q22.1; 3354 aa; ~371 kDa; "
            "Tip-link upper component — CDH23 (upper) + PCDH15 (lower) = mechanoelectrical transduction link; "
            "Hypomorphic alleles → DFNB12 (non-syndromic AR SNHL, no RP); null → Usher 1D; "
            "c.7903C>T (p.Arg2636Ter) Finnish founder — Usher 1D; "
            "CDH23 harmonin-binding cadherin repeat 27 critical for tip-link stability; "
            "CDH23 and PCDH15 interact via EC1-EC2 handshake — tip-link heterodimer; "
            "ERG mandatory — RP onset 2nd–3rd decade even after cochlear diagnosis"
        ),
        "aa": "3354 aa",
        "kDa": "~371 kDa",
        "locus": "10q22.1",
        "omim_gene": 605516,
        "omim_disease": 601067,
        "inheritance": "AR — biallelic null → Usher 1D; biallelic hypomorphic → DFNB12 (non-syndromic)",
        "gene_class": (
            "Cadherin 23 forms the upper half of the stereocilia tip-link — the filament connecting "
            "adjacent stereocilia that transmits tension to gate the MET channel. CDH23 (upper) forms "
            "an antiparallel heterodimer with PCDH15 (lower) via their first two extracellular cadherin "
            "repeats (EC1-EC2 handshake). Null CDH23 alleles completely abolish tip-links → MET channel "
            "cannot open → profound deafness + vestibular areflexia + RP (Usher 1D). Hypomorphic alleles "
            "that retain partial CDH23 function produce DFNB12 — non-syndromic deafness without RP. "
            "Genotype-phenotype correlation is relatively predictable: truncating variants → Usher; "
            "missense hypomorphic → DFNB12. This makes CDH23 one of the clearest examples of "
            "allelic heterogeneity determining syndromic vs non-syndromic phenotype."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("Biallelic null CDH23 — Usher 1D (profound SNHL + vestibular areflexia + RP)", 0.45),
            ("p.Arg2636Ter Finnish founder — Usher 1D", 0.20),
            ("Biallelic hypomorphic — DFNB12 non-syndromic SNHL", 0.20),
            ("Compound heterozygous null + hypomorphic — intermediate phenotype", 0.10),
            ("Novel biallelic CDH23 variants", 0.05),
        ],
        "age_onset_years_range": (0, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "congenital_profound_snhl_usher1d":     0.65,
            "severe_snhl_dfnb12_nonsyndromic":      0.20,
            "vestibular_areflexia_usher1d":         0.62,
            "retinitis_pigmentosa_usher1d":         0.65,
            "rp_onset_2nd_3rd_decade":              0.60,
            "night_blindness_onset":                0.58,
            "delayed_walking":                      0.55,
            "ci_implanted":                         0.60,
            "good_ci_outcome":                      0.55,
            "low_vision_services":                  0.35,
            "annual_erg_performed":                 0.70,
            "dfnb12_no_rp_confirmed":               0.20,
            "consanguineous_family":                0.25,
            "positive_family_history_ar":           0.55,
        },
        "hallmarks": [
            "Genotype predicts phenotype: null CDH23 → Usher 1D (RP + deaf + vestibular); hypomorphic → DFNB12 (deaf only)",
            "Tip-link upper component — CDH23-PCDH15 heterodimer; test BOTH when Usher 1 suspected",
            "ERG mandatory even if DFNB12 suspected — exclude RP with certainty; RP onset teens-20s",
            "Finnish p.Arg2636Ter founder — targeted test first in Finnish families with Usher 1",
            "Vestibular areflexia → delayed walking; balance relies on vision; no vestibular compensation",
            "CI effective for Usher 1D — early implantation before RP onset gives best dual-sensory outcome",
        ],
        "treatment_alerts": [
            "DFNB12 vs USHER 1D DISCRIMINATION: genotype determines RP risk — null → annual ERG; hypomorphic → monitor with ERG annually until 30",
            "USHER 1 PANEL: test CDH23 + PCDH15 + MYO7A + SANS + HARMONIN simultaneously — sequential testing delays diagnosis",
            "CI TIMING: before 12 months in Usher 1D; CI outcomes equivalent to non-syndromic deafness",
            "ERG ANNUAL MANDATORY: even after DFNB12 confirmed (hypomorphic alleles); RP can appear in late-onset form",
            "VESTIBULAR PT: balance physiotherapy; safe swimming; night-time fall risk assessment",
            "LOW VISION: early referral as RP advances; deaf-blind services when dual-sensory loss significant",
            "GENETIC COUNSELLING: AR 25% recurrence; differentiate Usher 1D from DFNB12 by allele severity",
        ],
        "primary_treatment": (
            "Hearing: CI for Usher 1D profound deafness before 12 months — maximise auditory function while vision intact. "
            "Hearing aids for DFNB12 moderate-severe loss. "
            "Vision: annual ERG from diagnosis; low-vision services as RP progresses; sunglasses UV-blocking. "
            "Vestibular: physiotherapy; safe mobility assessment; balance training. "
            "Deaf-blind: services when both senses significantly affected. "
            "Genetic counselling: cascade family testing; prenatal diagnosis; distinguish Usher 1D from DFNB12."
        ),
    },

    # ── PCDH15 — Usher Syndrome Type 1F / DFNB23 ──
    {
        "gene": "PCDH15",
        "protein": "Protocadherin 15 (Stereocilia Tip-Link Lower Component)",
        "alias": (
            "PCDH15; OMIM gene 605514; DFNB23 #609533, Usher1F #602083; 10q21.1; 1955 aa; ~215 kDa; "
            "Tip-link lower component — CDH23 (upper) + PCDH15 (lower) = MET tip-link heterodimer; "
            "R245X (c.733C>T) Ashkenazi Jewish founder — 1 in 148 carrier frequency; "
            "Most common Usher type 1 gene in Ashkenazi Jewish populations; "
            "PCDH15 CD2 isoform C-terminal binding harmonin (USH1C) → tripartite Usher 1 complex; "
            "Hypomorphic alleles → DFNB23 (non-syndromic SNHL, no RP); "
            "Null → Usher 1F: congenital profound SNHL + vestibular areflexia + RP onset teens"
        ),
        "aa": "1955 aa",
        "kDa": "~215 kDa",
        "locus": "10q21.1",
        "omim_gene": 605514,
        "omim_disease": 602083,
        "inheritance": "AR — biallelic null → Usher 1F; biallelic hypomorphic → DFNB23 (non-syndromic)",
        "gene_class": (
            "Protocadherin 15 forms the lower half of the stereocilia tip-link, connecting the "
            "tip of each shorter stereocilium to the side of the adjacent taller stereocilium. "
            "With CDH23, PCDH15 constitutes the complete tip-link — the critical mechanical coupling "
            "element that gates the MET channel. PCDH15 also scaffolds the USH1 protein complex at "
            "stereocilia tips via its C-terminal binding to harmonin (USH1C). Null PCDH15 alleles "
            "destroy tip-links and the entire USH1 complex → profound deafness + vestibular areflexia "
            "+ RP (Usher 1F). The R245X Ashkenazi Jewish founder allele makes PCDH15 the most common "
            "Usher type 1 gene in Ashkenazi Jewish patients; targeted R245X testing is cost-effective "
            "before a comprehensive Usher panel in this population."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("R245X (c.733C>T) Ashkenazi Jewish founder — Usher 1F biallelic", 0.35),
            ("Other biallelic PCDH15 null — Usher 1F (various populations)", 0.30),
            ("Biallelic hypomorphic PCDH15 — DFNB23 non-syndromic", 0.18),
            ("Compound heterozygous null + hypomorphic — intermediate phenotype", 0.10),
            ("Novel biallelic PCDH15 variants", 0.07),
        ],
        "age_onset_years_range": (0, 3),
        "sex_ratio_M": 0.50,
        "rates": {
            "congenital_profound_snhl":             0.80,
            "vestibular_areflexia":                 0.75,
            "retinitis_pigmentosa_usher1f":         0.78,
            "rp_onset_teens_20s":                   0.72,
            "night_blindness":                      0.70,
            "visual_field_loss":                    0.60,
            "delayed_walking_18_months":            0.68,
            "ci_implanted":                         0.68,
            "excellent_ci_speech_outcome":          0.63,
            "low_vision_services":                  0.40,
            "deaf_blind_services":                  0.28,
            "annual_erg_performed":                 0.75,
            "ashkenazi_jewish_ancestry":            0.35,
            "r245x_carrier_detected_in_family":     0.30,
        },
        "hallmarks": [
            "R245X Ashkenazi Jewish founder — 1 in 148 carrier frequency; test R245X FIRST in Ashkenazi before full panel",
            "Usher 1F triad: congenital profound deafness + vestibular areflexia + RP (teens onset) — same as Usher 1B/1D",
            "DFNB23 vs Usher 1F: allele severity predicts RP; null → Usher 1F; hypomorphic → DFNB23",
            "ERG annual from diagnosis — RP onset teens in Usher 1F; must not be missed",
            "CI before 12 months — Usher 1F CI outcomes excellent; maximise auditory function",
            "Tip-link lower component: CDH23-PCDH15 heterodimer — test both in any Usher 1 suspicion",
        ],
        "treatment_alerts": [
            "ASHKENAZI JEWISH TARGETED TEST: R245X (c.733C>T) PCDH15 before full Usher panel — cost-effective in Ashkenazi populations",
            "USHER 1 PANEL: simultaneous PCDH15 + CDH23 + MYO7A + SANS + HARMONIN — sequential testing delays diagnosis",
            "CI BEFORE 12 MONTHS: Usher 1F — early CI is the most impactful intervention; do not delay pending RP confirmation",
            "ERG ANNUAL: from diagnosis — RP often not visible funduscopically until late teens; ERG detects earlier",
            "DEAF-BLIND PLAN: discuss trajectory from diagnosis — prepare family for progressive dual-sensory loss",
            "CASCADE TESTING: R245X carrier frequency 1/148 in Ashkenazi — offer expanded carrier screening to relatives",
            "AVOID BRIGHT LIGHT: UV-protective sunglasses mandatory; photoreceptor vulnerability in RP",
        ],
        "primary_treatment": (
            "Hearing: CI before 12 months — most impactful intervention for Usher 1F. "
            "Auditory-verbal therapy post-CI; sign language as backup communication as RP progresses. "
            "Vision: annual ERG from diagnosis; low-vision aids as RP progresses; UV-blocking sunglasses; "
            "orientation-mobility training; deaf-blind services when both senses significantly affected. "
            "Vestibular: physiotherapy; safe mobility; fall prevention. "
            "Genetic counselling: R245X carrier testing in Ashkenazi family members; "
            "cascade testing; 25% recurrence per pregnancy; prenatal diagnosis available."
        ),
    },
]


def _simulate_patients(gene_def: dict) -> list:
    rng = random.Random(gene_def["seed"])
    patients = []
    ages = list(range(gene_def["age_onset_years_range"][0], gene_def["age_onset_years_range"][1] + 1))
    if not ages:
        ages = [0]
    n = gene_def["n_patients"]

    for i in range(n):
        age_onset = rng.choice(ages)

        r = rng.random()
        cum = 0.0
        etiology = gene_def["etiologies"][-1][0]
        for label, frac in gene_def["etiologies"]:
            cum += frac
            if r < cum:
                etiology = label
                break

        features = {}
        for feat, rate in gene_def["rates"].items():
            features[feat] = rng.random() < rate

        sex = "M" if rng.random() < gene_def["sex_ratio_M"] else "F"

        patients.append({
            "id": i + 1,
            "gene": gene_def["gene"],
            "age_onset": age_onset,
            "sex": sex,
            "etiology": etiology,
            "features": features,
        })
    return patients


def _aggregate_stats(patients: list, rates: dict) -> dict:
    if not patients:
        return {}
    n = len(patients)
    return {k: round(sum(p["features"].get(k, False) for p in patients) / n * 100, 1) for k in rates}


# ─────────────────────────────────────────────────────────────────────────────
# Build all cohorts once
# ─────────────────────────────────────────────────────────────────────────────
_ALL_PATIENTS: dict = {}
_ALL_STATS: dict = {}

for _gd in HHL_GENES:
    _pts = _simulate_patients(_gd)
    _ALL_PATIENTS[_gd["gene"]] = _pts
    _ALL_STATS[_gd["gene"]] = _aggregate_stats(_pts, _gd["rates"])


# ─────────────────────────────────────────────────────────────────────────────
# API Data Functions
# ─────────────────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """Overview — aggregate stats across all 320 patients."""
    all_pts = [p for pts in _ALL_PATIENTS.values() for p in pts]
    n = len(all_pts)

    def _pct(key: str) -> float:
        return round(sum(p["features"].get(key, False) for p in all_pts) / n * 100, 1)

    genes = [g["gene"] for g in HHL_GENES]

    top_alerts = [
        "GJB2-FIRST-TEST: most common hereditary SNHL gene — test GJB2 first in all congenital bilateral SNHL",
        "GJB6-DELETION-MANDATORY: exclude del(GJB6-D13S1830) before calling monoallelic GJB2 — may be second allele",
        "SLC26A4-HEAD-TRAUMA-AVOID: written prohibition — contact sports, diving, trampolines cause irreversible step-wise progression in EVA",
        "OTOF-CI-WORKS: OTOF-ANSD = pre-neural (synaptic) — CI excellent; do NOT withhold CI for 'ANSD' label",
        "OTOF-TEMPERATURE-SENSITIVE: fever causes acute hearing deterioration — antipyretics immediately; written fever plan",
        "USHER1-ERG-MANDATORY: annual ERG from diagnosis in MYO7A/CDH23/PCDH15 — RP onset teens; catch before blindness",
        "USHER1-CI-EARLY: CI before 12 months in all Usher type 1 — maximise auditory function while vision intact",
        "PCDH15-R245X-ASHKENAZI: 1 in 148 carrier frequency — test R245X first in Ashkenazi before full panel",
        "COCH-NO-CURE: DFNA9 is incurable progressive; set expectations early; fit hearing aids at 40 dB HL",
        "TMC1-GENE-THERAPY: AAV-TMC1 Phase I/II trials active 2024 — check ClinicalTrials.gov for eligibility",
        "USHER1-PANEL-SIMULTANEOUS: test MYO7A + CDH23 + PCDH15 + SANS + HARMONIN simultaneously — sequential is too slow",
        "CI-BEFORE-12-MONTHS: universal rule for all congenital profound SNHL — critical language acquisition window",
    ]

    diseases = {}
    for g in HHL_GENES:
        alias_parts = g["alias"].split(";")
        diseases[g["gene"]] = alias_parts[3].strip() + " — " + alias_parts[4].strip() if len(alias_parts) > 4 else g["alias"][:120]

    return {
        "atlas_name": "Hereditary-Hearing-Loss-Atlas — Complete 8-Gene Hereditary Sensorineural Hearing Loss Atlas",
        "subtitle": "GJB2 · SLC26A4 · OTOF · COCH · TMC1 · MYO7A · CDH23 · PCDH15",
        "total_patients": n,
        "genes": genes,
        "seed_range": "1430–1437",
        "aggregate_stats": {
            "congenital_profound_snhl":         round(sum(
                any(p["features"].get(k, False) for k in [
                    "congenital_profound_snhl", "congenital_profound_snhl_ar",
                    "congenital_profound_snhl_usher1d", "congenital_profound_snhl_ar",
                ]) for p in all_pts) / n * 100, 1),
            "ci_implanted":                     _pct("ci_implanted"),
            "retinitis_pigmentosa":             round(sum(
                any(p["features"].get(k, False) for k in [
                    "retinitis_pigmentosa", "retinitis_pigmentosa_usher1d", "retinitis_pigmentosa_usher1f",
                ]) for p in all_pts) / n * 100, 1),
            "vestibular_areflexia":             round(sum(
                any(p["features"].get(k, False) for k in [
                    "vestibular_areflexia", "vestibular_areflexia_bilateral", "vestibular_areflexia_usher1d",
                ]) for p in all_pts) / n * 100, 1),
            "bilateral_symmetric_loss":         round(sum(
                any(p["features"].get(k, False) for k in [
                    "bilateral_symmetric", "bilateral_symmetric_eva", "bilateral_symmetric",
                ]) for p in all_pts) / n * 100, 1),
            "progressive_snhl":                 _pct("progressive_bilateral_snhl"),
            "hearing_aid_user":                 _pct("hearing_aid_user"),
            "ansd_phenotype":                   _pct("ansd_phenotype_dpoae_present_abr_absent"),
            "fluctuating_hearing_loss_eva":     _pct("fluctuating_hearing_loss"),
            "usher_triad_any":                  round(sum(
                any(p["features"].get(k, False) for k in [
                    "retinitis_pigmentosa", "retinitis_pigmentosa_usher1d", "retinitis_pigmentosa_usher1f",
                ]) for p in all_pts) / n * 100, 1),
        },
        "top_alerts": top_alerts,
        "diseases": diseases,
    }


def get_breakdown() -> dict:
    """Per-gene breakdown for Gene Table and Clinical Atlas tabs."""
    result = {}
    for gd in HHL_GENES:
        gene = gd["gene"]
        pts = _ALL_PATIENTS[gene]
        stats = _ALL_STATS[gene]

        etiology_distribution = [
            {"etiology": label, "fraction": round(frac, 3)}
            for label, frac in gd["etiologies"]
        ]

        result[gene] = {
            "gene":                 gene,
            "protein":              gd["protein"],
            "aa":                   gd["aa"],
            "locus":                gd["locus"],
            "omim_gene":            gd["omim_gene"],
            "omim_disease":         gd["omim_disease"],
            "inheritance":          gd["inheritance"],
            "organ_system":         "Inner ear / Cochlea / Stereocilia / Retina / Vestibular",
            "n_patients":           gd["n_patients"],
            "seed":                 gd["seed"],
            "gene_class":           gd["gene_class"],
            "hallmarks":            gd["hallmarks"],
            "treatment_alerts":     gd["treatment_alerts"],
            "primary_treatment":    gd["primary_treatment"],
            "stats":                stats,
            "etiology_distribution": etiology_distribution,
        }
    return result


def get_definitions() -> dict:
    """Disease classification, diagnostic rules, and treatment hierarchies."""
    return {
        "classification": {
            "gap_junction_disorders": {
                "GJB2_DFNB1": "AR — connexin 26 LOF; K+ recycling failure; most common hereditary SNHL; CI excellent",
            },
            "endolymph_homeostasis": {
                "SLC26A4_DFNB4_Pendred": "AR — pendrin LOF; EVA + ± goitre + organification defect; head trauma avoidance mandatory",
            },
            "synaptic_disorders_ansd": {
                "OTOF_DFNB9": "AR — otoferlin LOF; auditory neuropathy ANSD (pre-neural synaptic); CI EXCELLENT unlike neuropathy ANSD",
            },
            "extracellular_matrix": {
                "COCH_DFNA9": "AD — cochlin LCCL domain; adult progressive SNHL + Meniere-like vestibular; no cure",
            },
            "met_channel": {
                "TMC1_DFNB7_DFNA36": "AR/AD — MET channel subunit; profound pre-lingual (AR) or progressive (AD M298K); gene therapy trials 2024",
            },
            "usher_syndrome_type1": {
                "MYO7A_Usher1B": "AR — myosin VIIA; most common Usher 1; deaf + vestibular areflexia + RP; CI early; annual ERG",
                "CDH23_Usher1D": "AR — cadherin 23 tip-link upper; null = Usher 1D; hypomorphic = DFNB12; Finnish R2636X founder",
                "PCDH15_Usher1F": "AR — protocadherin 15 tip-link lower; R245X Ashkenazi founder 1/148; null = Usher 1F; hypomorphic = DFNB23",
            },
        },
        "key_diagnostic_rules": {
            "GJB2_GJB6_MONOALLELIC": (
                "When only one GJB2 pathogenic variant is found by sequencing, the case cannot be declared "
                "monoallelic until GJB6 deletion analysis is complete. The del(GJB6-D13S1830) deletion "
                "removes GJB2's regulatory region and acts as a functional second allele. In Northern "
                "European GJB2 heterozygotes, this deletion accounts for a significant proportion of "
                "'monoallelic' cases. MLPA or targeted PCR for GJB6 deletion must be performed as part "
                "of the standard GJB2 workup in all populations."
            ),
            "SLC26A4_HEAD_TRAUMA_PROHIBITION": (
                "Enlarged vestibular aqueduct (EVA) associated with SLC26A4 variants is uniquely vulnerable "
                "to minor head trauma, Valsalva manoeuvres, contact sports, diving, and barotrauma. Each "
                "event can cause step-wise irreversible deterioration of hearing. Written activity "
                "restrictions must be provided at EVERY clinical encounter. The mechanism is pressure "
                "transmission via the enlarged endolymphatic duct to the cochlear epithelium. There is "
                "no treatment once deterioration occurs — prevention is the only strategy."
            ),
            "OTOF_ANSD_CI_DISTINCTION": (
                "Auditory neuropathy spectrum disorder (ANSD) has two fundamentally different origins: "
                "(1) Pre-neural/synaptic ANSD (e.g. OTOF): outer hair cells intact (DPOAEs present), "
                "defect is at the inner hair cell ribbon synapse. CI bypasses the defective synapse and "
                "directly stimulates the spiral ganglion neurons — outcomes EXCELLENT, comparable to "
                "non-ANSD CI. (2) Neural ANSD (e.g. AIFM1, DIAPH3, auditory nerve hypoplasia): CI "
                "stimulates abnormal spiral ganglion neurons — outcomes variable and often poor. "
                "Genetic diagnosis of OTOF MANDATES CI counselling as first-line, not withholding CI."
            ),
            "USHER1_ERG_MANDATORY": (
                "All patients with Usher type 1 (MYO7A, CDH23, PCDH15, SANS, HARMONIN biallelic) must "
                "have annual electroretinography (ERG) from diagnosis, regardless of current visual "
                "symptoms. Retinitis pigmentosa (RP) in Usher 1 typically begins in the teens but may "
                "not be symptomatic (night blindness, field loss) until significant photoreceptor loss "
                "has already occurred. ERG detects RP years before symptoms or funduscopic changes. "
                "Early RP detection enables: low-vision planning, orientation-mobility training, "
                "deaf-blind service engagement, and family counselling — all of which improve outcomes."
            ),
            "PCDH15_R245X_ASHKENAZI_TARGETED": (
                "PCDH15 c.733C>T (p.Arg245Ter / R245X) has a carrier frequency of approximately 1 in 148 "
                "in Ashkenazi Jewish individuals, making it the most common Usher type 1 founder allele "
                "in this population. In Ashkenazi patients with suspected Usher syndrome or severe "
                "congenital deafness, R245X targeted testing should be performed FIRST before ordering "
                "a comprehensive Usher panel. If R245X is found heterozygous, the second allele "
                "requires sequencing of PCDH15. This targeted approach is faster and more cost-effective "
                "than full panel in Ashkenazi populations."
            ),
            "TMC1_NEONATAL_SCREEN_FALSE_NEGATIVE": (
                "TMC1 is the dominant MET channel subunit in adult cochlear hair cells. However, in "
                "neonates, TMC2 provides functional redundancy and can partially compensate for TMC1 "
                "absence. This means the newborn hearing screen (ABR or OAE) may be NORMAL or near-normal "
                "in some TMC1 biallelic cases, with profound deafness appearing in the first 1–2 years "
                "as TMC2 is developmentally downregulated. Refer all children with bilateral 'refer' "
                "on newborn screen for formal audiological assessment by 3 months — and repeat testing "
                "at 12 months even if initial screen passed in familial deafness cases."
            ),
            "CI_CRITICAL_PERIOD": (
                "Cochlear implantation before 12 months of age consistently produces superior language "
                "outcomes across all congenital profound SNHL aetiologies — GJB2, OTOF, SLC26A4, TMC1, "
                "Usher type 1, and others. The critical period for auditory cortex development is maximal "
                "in the first 12–18 months. CI before 12 months: language outcomes approach hearing peers. "
                "Delayed CI (>3 years): significantly worse spoken language outcomes. This is universal — "
                "the gene aetiology does not change the timing recommendation for CI in profound SNHL."
            ),
        },
        "treatment_hierarchy": {
            "GJB2_DFNB1": [
                "1. CI: before 12 months for severe–profound; standard first-line",
                "2. Hearing aids: for mild–moderate (V37I allele); trial before CI in moderate cases",
                "3. Auditory-verbal therapy post-CI; speech-language pathology",
                "4. Annual audiogram for hearing aid users; no progression in most GJB2",
                "5. Genetic counselling: GJB6 exclusion; 25% recurrence; cascade testing",
            ],
            "SLC26A4_Pendred_DFNB4": [
                "1. ABSOLUTE: avoid head trauma, contact sports, diving, Valsalva — written prohibition",
                "2. CI for severe–profound; hearing aids for moderate–severe",
                "3. Annual TSH; thyroid treatment only if hypothyroid",
                "4. Acute fluctuation: oral prednisolone trial (evidence limited)",
                "5. Genetic counselling: cascade testing; 25% recurrence",
            ],
            "OTOF_DFNB9": [
                "1. CI: first-line for severe–profound OTOF-ANSD — do not withhold CI for 'ANSD' label",
                "2. Fever plan: antipyretics immediately; written emergency instruction",
                "3. FM system + soundfield amplification: modest benefit while awaiting CI",
                "4. Auditory-verbal therapy post-CI",
                "5. Genetic counselling: 25% recurrence; cascade testing",
            ],
            "COCH_DFNA9": [
                "1. Hearing aids: fit promptly at 40 dB HL threshold — do not wait for severe",
                "2. CI: when bilateral profound (COCH CI outcomes good)",
                "3. Vestibular rehabilitation: physiotherapy for progressive vestibular loss",
                "4. Annual audiogram + annual vestibular assessment (caloric, video-HIT)",
                "5. Genetic counselling: 50% offspring risk (AD); cascade testing",
            ],
            "TMC1_DFNB7_DFNA36": [
                "1. CI: before 12 months for AR profound DFNB7/11",
                "2. Hearing aids: for AD DFNA36 progressive (M298K); annual audiogram from age 10",
                "3. Gene therapy: check ClinicalTrials.gov for AAV-TMC1 trial eligibility",
                "4. Auditory-verbal therapy post-CI",
                "5. Genetic counselling: AR 25% / AD 50% recurrence",
            ],
            "Usher1_MYO7A_CDH23_PCDH15": [
                "1. CI: before 12 months — most impactful intervention; maximise auditory function",
                "2. Annual ERG from diagnosis — detect RP years before symptoms",
                "3. Vestibular physiotherapy: balance training; fall prevention; safe mobility",
                "4. Low-vision services: as RP advances; orientation-mobility training",
                "5. Deaf-blind services: when both senses significantly affected",
                "6. Genetic counselling: 25% recurrence; cascade Usher 1 panel; prenatal testing",
            ],
        },
    }
