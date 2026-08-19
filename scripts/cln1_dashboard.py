"""
CLN1 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 1 / Infantile Batten Disease / Santavuori-Haltia Disease
=============================================================================================================
40-patient cohort · CLN1/PPT1 (1p34.2) · Autosomal recessive (AR) biallelic LOF
PPT1 encodes Palmitoyl-Protein Thioesterase 1 — a lysosomal serine hydrolase
PPT1 LOF → fatty-acid-modified protein accumulation → GRODS (granular osmiophilic deposits) on EM →
progressive neuronal apoptosis (cortex + cerebellum + retina) → CLN1 / Infantile NCL (INCL)

EARLIEST-ONSET NCL — INFANTILE RAPID REGRESSION:
CLN1 (Santavuori-Haltia disease) is the most severe and earliest-onset NCL, presenting at 6-24 months
with RAPID PSYCHOMOTOR REGRESSION — the FIRST sign in virtually all affected infants:
  - Deceleration of development then loss of milestones from 6-24 months (mean 10 months)
  - Visual failure emerging simultaneously or immediately after developmental regression
  - Myoclonic jerks and irritability follow milestone loss
  - Characteristic EEG progression: normal → high-amplitude → rapid background suppression → isoelectric
  - Mean survival 7-12 years (range 6-15 years); FATAL in all classic infantile CLN1

GRODS — PATHOGNOMONIC EM FINDING FOR CLN1:
Granular Osmiophilic Deposits (GRODs) in skin biopsy eccrine gland epithelial cells, endothelial
cells, lymphocytes:
  - Small, round, electron-dense granular deposits — PATHOGNOMONIC for CLN1/PPT1 deficiency
  - Distinct from CLN2 (curvilinear bodies + fingerprint profiles) and CLN4B (fingerprint profiles only)
  - GRODs represent accumulated palmitate-modified proteins that PPT1 normally removes
  - Skin biopsy EM (GRODs) + PPT1 enzyme assay (DBS): the two-test diagnostic confirmation for CLN1

PPT1 PROTEIN BIOLOGY (LYSOSOMAL SERINE HYDROLASE):
CLN1/PPT1 (1p34.2), PPT1 (Palmitoyl-Protein Thioesterase 1):
  - 306 aa preproprotein; ~34-35 kDa mature form after signal peptide cleavage
  - Signal peptide aa 1-25 (lysosomal targeting); Mature protein aa 26-306
  - Catalytic triad: Ser115-Asp233-His289 (serine hydrolase active site)
  - Thioesterase reaction: cleaves thioester bonds between palmitate and Cys residues in
    palmitoylated proteins — removes fatty acids from S-acylated cysteine residues
  - pH-optimum: 4.0-5.0 (lysosomal — activated by acidic environment)
  - Ubiquitous expression; highest in neurons and retinal pigment epithelium
  - Lysosomal localisation; partially secreted and recaptured via mannose-6-phosphate receptor
  - PPT1 LOF → failure to deprotect palmitoylated substrates → accumulation of fatty-acid-
    modified proteins → ceroid storage (GRODs on EM) → progressive neuronal and retinal apoptosis
  - pLI ~0.85 (moderate-high intolerance)
  - OMIM: *600722 (PPT1 gene) / #256730 (CLN1 disease / Santavuori-Haltia disease / INCL)
  - Discovery: Vesa J et al. 1995 Nature — PPT1 identified as CLN1 disease gene
    (Santavuori P et al. 1971 — original clinical description of infantile NCL)

COMMON MUTATIONS:
  p.Arg122Trp (c.364C>T — Finnish founder mutation):
    most common CLN1 allele (~60% of Finnish CLN1 alleles); Arg122 in beta-4 strand flanking
    active site; Arg→Trp disrupts hydrogen bonding network around catalytic triad → complete PPT1
    LOF; classic infantile Santavuori-Haltia onset 6-18 months; Finnish heritage enrichment
    (carrier frequency ~1:70 in Finland); homozygous → severe classic infantile phenotype
  p.Leu10X (c.29_30delAT — Northern European truncating):
    2-bp frameshift deletion → premature stop at position 10 → no functional PPT1 protein;
    severe loss-of-function; classic infantile phenotype
  p.Thr75Pro (attenuated / late-infantile-juvenile variant):
    residual PPT1 activity → milder phenotype; onset 2-7 years; slower progression; late-infantile
    or juvenile CLN1 (CLN1 variant) — distinct from classic infantile Santavuori-Haltia
  p.Arg151X (truncating — nonsense):
    severe LOF; classic infantile phenotype; widely distributed across populations

CLN1 KEY DISTINCTIONS:
  EARLIEST INFANTILE ONSET (6-24 months) — fastest progression of any NCL
  GRODS on EM — PATHOGNOMONIC (distinct from CLN2 curvilinear, CLN3 fingerprint+curvilinear, CLN4B fingerprint)
  PPT1 ENZYME ASSAY on DBS — rapid diagnostic test (days) — MUST come before WES
  EEG EXTINCTION — progressive background suppression to isoelectric by age 2-3y — PATHOGNOMONIC
  VEP EXTINCTION — visual evoked potentials extinguished early (age 12-18 months) — diagnostic
  VISUAL FAILURE EARLY — concurrent with developmental regression (simultaneous, not sequential as in CLN3)
  FATAL — no disease-modifying therapy (contrast CLN2 cerliponase alfa)
  FINNISH HERITAGE ENRICHMENT — p.Arg122Trp founder mutation; highest CLN1 prevalence in Finland
"""


def get_overview():
    return {
        "gene": "CLN1/PPT1 (1p34.2) — PPT1 (Palmitoyl-Protein Thioesterase 1; lysosomal serine hydrolase)",
        "protein": "PPT1 (Palmitoyl-Protein Thioesterase 1); 306 aa preproprotein; ~34-35 kDa mature; signal peptide aa 1-25 (lysosomal targeting); Ser115-Asp233-His289 catalytic triad (serine hydrolase); cleaves thioester bonds between palmitate and cysteine residues in S-acylated proteins; pH-optimum 4.0-5.0 (lysosomal activation); ubiquitous, highest in neurons + retinal pigment epithelium; partially secreted and recaptured via mannose-6-phosphate receptor; PPT1 LOF → accumulated fatty-acid-modified proteins → GRODS (granular osmiophilic deposits) on EM → progressive neuronal and retinal apoptosis",
        "inheritance": "Autosomal recessive (AR) biallelic LOF; pLI ~0.85; typically compound heterozygous or homozygous pathogenic variants; Finnish founder mutation p.Arg122Trp (c.364C>T) accounts for ~60% of Finnish CLN1 alleles; PPT1 enzyme assay on DBS (dried blood spot) is the rapid diagnostic test; OMIM *600722 / #256730",
        "omim": "*600722 (PPT1 gene) · #256730 (CLN1 disease / Santavuori-Haltia disease / Infantile NCL / INCL)",
        "disease": "CLN1 — Infantile Neuronal Ceroid Lipofuscinosis (INCL / Santavuori-Haltia disease): RAPID PSYCHOMOTOR REGRESSION at 6-24 months (mean 10 months) — FIRST sign; simultaneous visual failure (ERG abnormal from 6-12 months; VEP extinguished by 12-18 months); myoclonic jerks, irritability, hypotonia; progressive retinal degeneration, optic atrophy, cortical and cerebellar neurodegeneration; EEG: rapid progression from high-amplitude to suppressed isoelectric by age 2-3y; fatal 7-12 years; no disease-modifying therapy",
        "mechanism": "CLN1/PPT1 biallelic LOF → PPT1 absent/non-functional → failure to cleave thioester-linked fatty acids (palmitate) from S-acylated lysosomal substrate proteins → progressive accumulation of fatty-acid-modified proteins → GRODS (granular osmiophilic deposits) formation in neurons, retina, eccrine sweat glands → progressive apoptosis cortex + cerebellum + retina → INCL / Santavuori-Haltia disease",
        "no_disease_modifying_therapy": "CONFIRMED — NO approved disease-modifying therapy for CLN1. Investigational: AAV-PPT1 gene therapy (preclinical + early phase trials; CLN1 mouse model efficacy demonstrated; Macauley SL 2018); enzyme replacement therapy (lysosomal delivery analogous to CLN2 cerliponase but not yet approved); substrate reduction approaches. Register with BDSRA, NCL Resource, NCL Network Europe for trial eligibility. All CLN1 patients must be offered BDSRA registry enrolment to enable future trial access.",
        "cohort_size": 40,
        "female_pct": 48,
        "mean_onset_regression_months": 10.2,
        "mean_onset_seizure_months": 16.8,
        "mean_death_years": 9.4,
        "drug_resistant_pct": 88,
        "photosensitivity_pct": 35,
        "retinal_degeneration_pct": 100,
        "grods_skin_biopsy_pct": 92,
        "eeg_suppression_by_age_3y_pct": 98,
        "vep_extinction_pct": 95,
        "finnish_heritage_pct": 22,
        "cognitive_decline_pct": 100,
        "on_vpa_pct": 82,
        "on_lev_pct": 72,
        "on_kd_pct": 28,
        "discovery": "CLN1/PPT1 gene identified by Vesa J et al. 1995 (Nature 376:584-587) in Santavuori-Haltia disease; original Infantile NCL clinical description: Santavuori P et al. 1971 (Acta Paediatrica Scandinavica); GRODs as pathognomonic EM finding: Haltia M et al. 1973; PPT1 enzyme assay diagnostic test: Hofmann SL et al. 1999",
        "unique_feature": "EARLIEST-ONSET NCL (6-24 months) — most severe and fastest-progressing NCL. GRODS on EM are the pathognomonic storage pattern (unique to CLN1 — distinct from CLN2 curvilinear, CLN3 fingerprint+curvilinear, CLN4B fingerprint). PPT1 ENZYME ASSAY on DBS provides diagnosis in days — must come before gene panel. EEG EXTINCTION (progressive isoelectric by age 2-3y) is pathognomonic. VEP EXTINCTION by 12-18 months is the earliest objective sign. VISUAL FAILURE CONCURRENT with developmental regression (not sequential as in CLN3). FATAL in first decade. Finnish heritage enrichment (p.Arg122Trp founder). NO disease-modifying therapy (distinct from CLN2 which has cerliponase). LYSOSOMAL SERINE HYDROLASE disease (distinct from CLN4B presynaptic chaperone).",
        "absolute_ci": [
            "Carbamazepine / Oxcarbazepine / Phenytoin (CBZ/OXC/PHT) — ABSOLUTE CI: Na-channel blockers WORSEN myoclonus in CLN1. Misdiagnosis trap: infantile seizures (GTCS/focal) → CBZ prescribed → ACUTE MYOCLONIC DETERIORATION. VPA+LEV mandatory backbone.",
            "Vigabatrin (VGB) — ABSOLUTE CI: VGB retinopathy (irreversible peripheral visual field constriction) superimposed on CLN1 retinal NCL (100% retinal degeneration) = catastrophic, irreversible accelerated blindness. ABSOLUTE CI at CLN1 diagnosis. Document in all ED records, AED card, paediatric emergency plan.",
            "Tiagabine (TGB) — ABSOLUTE CI: GABA-reuptake inhibition → NCSE in myoclonic epilepsies; ABSOLUTE CI across all NCL/PME syndromes.",
        ],
        "high_risk_ci": [
            "GBP / Pregabalin — HIGH RISK: worsen myoclonus (Crespel 1999); NOT age-appropriate for infants/toddlers; listed for completeness if CLN1 variant (juvenile form) sees adult specialists.",
            "LTG Monotherapy — HIGH RISK: LTG monotherapy worsens myoclonus in NCL epilepsy; NEVER monotherapy in CLN1; if add-on adjunct needed, ensure VPA backbone maintained.",
            "Fosphenytoin-IV — HIGH RISK: IV phenytoin/fosphenytoin for infantile SE = CI (Na-channel blocker worsen myoclonus); use IV LEV 60 mg/kg as first-line SE rescue; document in ED paediatric resuscitation notes.",
            "AED-Taper-Remission-Expectation — HIGH RISK: CLN1 is fatal progressive neurodegenerative disease — seizures do NOT remit; AED taper = inevitable severe GTCS + SUDEP risk; maintain AED indefinitely; late disease AED rationalisation only for palliative comfort (not seizure remission expectation).",
        ],
    }


def get_breakdown():
    etiologies = [
        {
            "class": "Homozygous-p.Arg122Trp-Finnish-Founder",
            "pct": 22,
            "description": "Finnish founder mutation p.Arg122Trp (c.364C>T) homozygous; Arg122 in beta-4 strand flanking catalytic triad; Arg→Trp disrupts hydrogen bonding network → complete PPT1 LOF; classic Santavuori-Haltia infantile onset 6-18 months; Finnish heritage pedigrees; GRODs on EM; PPT1 enzyme assay DBS = 0-2% residual activity; carrier frequency ~1:70 in Finland (highest CLN1 prevalence in world)",
            "count": 9,
            "gene_mechanism": "p.Arg122Trp disrupts Arg122 hydrogen bonding stabilising beta-4 strand adjacent to Ser115 catalytic nucleophile → complete PPT1 active-site collapse → no thioesterase activity → PPT1 complete LOF",
            "key_variants": ["p.Arg122Trp (c.364C>T — Finnish founder, ~60% Finnish CLN1 alleles)", "beta-4 strand active-site-adjacent residue Arg122"],
        },
        {
            "class": "Compound-Het-p.Arg122Trp-Plus-Other",
            "pct": 26,
            "description": "Compound heterozygous: p.Arg122Trp on one allele + pathogenic missense/nonsense on second allele; non-Finnish northern European distribution; classic infantile phenotype similar to homozygous p.Arg122Trp; GRODs on EM; PPT1 enzyme near-zero on DBS; PPT1 sequencing required (deletion PCR misses second allele if non-Finnish)",
            "count": 10,
            "gene_mechanism": "Trans compound heterozygosity — p.Arg122Trp (active site disruption) + independent LOF variant on second allele → complete biallelic PPT1 deficiency → classic infantile INCL phenotype",
            "key_variants": ["p.Arg122Trp + p.Arg151X (most common compound het)", "p.Arg122Trp + p.Leu10X (c.29_30delAT frameshift)", "p.Arg122Trp + novel missense"],
        },
        {
            "class": "Biallelic-Truncating-LOF-Null",
            "pct": 20,
            "description": "Biallelic null/truncating variants (frameshift + nonsense, or homozygous nonsense); complete PPT1 protein absence; early-onset severe infantile phenotype (onset 6-10 months, earliest in cohort); rapid EEG suppression; GRODs heavy on EM; PPT1 enzyme assay DBS = 0-1% residual activity; enzyme confirmation before gene sequencing to expedite diagnosis",
            "count": 8,
            "gene_mechanism": "Frameshift + nonsense compound het or homozygous nonsense → premature stop → NMD (nonsense-mediated mRNA decay) → no PPT1 protein produced → complete lysosomal serine hydrolase deficiency → severe INCL",
            "key_variants": ["p.Leu10X (c.29_30delAT — 2-bp deletion)", "p.Arg151X (c.451C>T)", "p.Tyr109X", "Various null compound heterozygous"],
        },
        {
            "class": "Compound-Het-Missense-Missense",
            "pct": 18,
            "description": "Compound heterozygous missense pairs; variable residual PPT1 activity (2-10%) → variable phenotype severity; classic infantile (0-5% residual) or late-infantile/juvenile variant (5-15% residual); PPT1 enzyme assay essential — enzyme level predicts phenotype severity; may overlap with 'late-infantile CLN1' if onset 18-36 months",
            "count": 7,
            "gene_mechanism": "Trans compound heterozygous missense — each allele individually may retain partial activity; combined biallelic residual activity determines phenotype: <5% → classic infantile; 5-15% → late-infantile; >15% → juvenile CLN1 variant with milder course",
            "key_variants": ["p.Thr75Pro (attenuated activity — late-infantile if trans)", "p.Tyr247His (catalytic domain missense)", "Various compound het missense combinations"],
        },
        {
            "class": "Late-Infantile-Juvenile-CLN1-Variant",
            "pct": 10,
            "description": "CLN1 variant with residual PPT1 activity (5-20%); onset 18 months - 6 years; milder phenotype — slower visual decline, longer ambulation, later EEG suppression; may be misdiagnosed as CLN2 or CLN3 (different onset age); PPT1 enzyme assay essential — residual activity explains phenotypic mitigation; GRODs on EM; management same as classic CLN1 (no disease-modifying therapy)",
            "count": 4,
            "gene_mechanism": "Hypomorphic biallelic PPT1 variants with partial residual enzyme activity (5-20% of normal) → slower lysosomal substrate accumulation → delayed/attenuated NCL phenotype → late-infantile or juvenile CLN1 variant",
            "key_variants": ["p.Thr75Pro (partial activity — attenuated CLN1)", "Exonic splice-site with partial skipping", "Missense pairs with combined 5-20% residual activity"],
        },
        {
            "class": "Phenocopy-CLN1-Negative",
            "pct": 4,
            "description": "Infantile NCL phenotype (rapid psychomotor regression + GRODs-like EM + infantile seizures) with normal PPT1 enzyme assay; represents CLN1 phenocopy from other causes (CLN10/CTSD infantile variant — cathepsin D deficiency; CLN13/CTSF; congenital variant CLN1-phenocopy); full NCL gene panel after negative PPT1 enzyme; cathepsin D enzyme assay next if CLN1 negative",
            "count": 2,
            "gene_mechanism": "PPT1 enzyme normal → CLN1 excluded; phenocopy from CLN10/CTSD (cathepsin D — lysosomal aspartyl protease; congenital NCL or early-infantile); CTSD enzyme assay + CTSD/CLN13 sequencing next",
            "key_variants": ["PPT1 enzyme normal → CLN10/CTSD next", "CTSD congenital NCL — profound neonatal/infantile onset", "CLN13/CTSF progressive epilepsy (older onset usually)"],
        },
    ]

    seizure_types = [
        {
            "type": "Myoclonic-Seizures-Multifocal",
            "pct": 95,
            "description": "Multifocal myoclonic jerks emerging from 8-20 months; initially subtle (eyelid flutter, limb twitch) → progressive generalised myoclonus; erratic, asynchronous, continuous myoclonus; photosensitive component in 35%; FIRST-LINE: VPA backbone (broad spectrum, covers myoclonus + GTCS) + LEV adjunct. Action myoclonus less prominent than in adult PMEs (infant lacks fine motor task baseline). Myoclonus becomes continuous (epilepsia partialis continua) in later disease.",
            "eeg": "Progressive background amplitude suppression; multifocal sharp waves and spikes; initially high-amplitude polyspike then rapid suppression; GRODS EM confirms CLN1. EEG background progressively attenuates to isoelectric by age 2-3y — PATHOGNOMONIC progression.",
            "semiology": "Erratic multifocal jerks — eyelids, limbs, trunk; not action-triggered (unlike adult PMEs); continuous erratic myoclonus; worse during wakefulness; stimulus-sensitive (sound, touch, photic — 35%); carer describes 'constant twitching' or 'jerking'; irritability accompanying myoclonus is typical",
            "clinical_tip": "VPA (20-40 mg/kg/day) backbone for myoclonus control in CLN1. PIracetam and clonazepam adjuncts for refractory myoclonus. AVOID CBZ/OXC/PHT — ABSOLUTE CI (worsen myoclonus). AVOID VGB — ABSOLUTE CI (retinal toxicity superimposed on CLN1 retinal degeneration). Document CI in paediatric emergency resuscitation notes, AED card, and hospital alert.",
        },
        {
            "type": "GTCS-Generalised-Tonic-Clonic",
            "pct": 88,
            "description": "GTCS emerging from 10-24 months; nocturnal predominance; triggered by fever, illness, missed AED; often the presenting epileptic event before NCL recognition; VPA + LEV backbone; IV LEV 60 mg/kg for SE. CRITICAL: GTCS in an infant with developmental regression → IMMEDIATE PPT1 ENZYME ASSAY + EM referral (do NOT wait for genetics results). SUDEP risk significant in CLN1.",
            "eeg": "Generalised poly-spike-wave burst; background suppression rapid; post-ictal flat/suppressed background (progressive background attenuation); differentiate from Ohtahara syndrome (earlier, suppression-burst pattern distinct)",
            "semiology": "Classic GTCS — tonic extension then clonic jerking; nocturnal predominance; post-ictal flaccidity; duration 2-5 minutes; may be brief in early disease; SE risk elevated by fever",
            "clinical_tip": "SE in CLN1: IV LEV 60 mg/kg first-line (NOT fosphenytoin — Na-channel blocker CI). If IV access fails: buccal midazolam (0.3 mg/kg). Paediatric ketamine as second-line for refractory SE. Document in paediatric resuscitation plan: 'CLN1 — NO phenytoin/fosphenytoin.' SUDEP risk: nocturnal monitoring, pulse oximetry, position monitoring (infant cannot reposition).",
        },
        {
            "type": "Focal-Seizures-Occipital-Visual",
            "pct": 72,
            "description": "Occipital-onset focal seizures from visual cortex involvement (early retino-cortical NCL degeneration); eye deviation, eyelid flutter, visual phenomena (where developmental level allows description), occipital delta slowing; VPA + LEV; VGB ABSOLUTE CI (retinal degeneration). Occipital focal features may initially prompt VGB consideration (infantile spasms differential) — CLN1 diagnosis MUST exclude VGB before any treatment.",
            "eeg": "Occipital delta slowing; posterior sharp waves; focal occipital spikes; VEP extinguished by 12-18 months — PATHOGNOMONIC. SSPS (slow-rate IPS) absent in CLN1 (differs from CLN2 where 1-3 Hz SSPS pathognomonic)",
            "semiology": "Eye deviation (lateral/upward); eyelid flickering; vomiting (occipital seizure); head version; brief visual phenomena if old enough to describe; post-ictal gaze palsy; often brief (30-90 seconds); clusters during febrile illness",
            "clinical_tip": "Occipital seizures + infantile regression + retinal signs → CLN1 until proven otherwise. ERG urgently — abnormal ERG in infancy + developmental regression = CLN1 diagnostic pathway. PPT1 enzyme assay DBS (same-day result) + ophthalmology referral. DO NOT prescribe VGB for occipital seizures in this phenotype.",
        },
        {
            "type": "Infantile-Spasms-West-Syndrome-Overlap",
            "pct": 42,
            "description": "Infantile spasms or West syndrome overlap — flexion/extension spasms in early CLN1 (6-14 months); may precede recognition as NCL; TRAP: VGB is FIRST-LINE for infantile spasms but ABSOLUTE CI in CLN1 (retinal toxicity). CLN1 must be excluded BEFORE VGB administered. PPT1 enzyme assay (DBS) in any infant with spasms + developmental regression + visual concern = mandatory.",
            "eeg": "Hypsarrhythmia may be present early in CLN1 spasms; CLN1 EEG rapidly transitions through: hypsarrhythmia (if spasms) → high-amplitude polyspike-wave → progressive background suppression → isoelectric by 2-3y. This sequential background attenuation is CLN1-specific.",
            "semiology": "Flexion (or mixed flex/extend) spasms; clusters on waking; may mimic early infantile spasms of other causes; 4-15 seconds per spasm; 10-100 per cluster; developmental stagnation noted by parents",
            "clinical_tip": "INFANTILE SPASMS + REGRESSION = CLN1 SCREEN FIRST. PPT1 enzyme assay DBS + ophthalmology (ERG/VEP) before ANY treatment decision. If CLN1 confirmed → VGB ABSOLUTE CI → use ACTH or vigabatrin-alternative corticosteroids for spasms component (ACTH 2-8 units/day or prednisolone 40 mg/day); VPA for ongoing seizures. NEVER VGB in CLN1 — even if spasms are the predominant seizure type.",
        },
        {
            "type": "Atonic-Drop-Attacks",
            "pct": 58,
            "description": "Atonic seizures with head drops or full body falls from 12-30 months; compound fall risk (ataxia + atonic drops + progressive hypotonia); helmet mandatory; padded protective equipment. CLB for atonic component; VPA backbone continues. Falls assessment with physiotherapy + OT is standard CLN1 MDT task.",
            "eeg": "Brief generalised slow-wave or spike-wave with electromyographic silence (atonic component); background attenuation between events; rapid background suppression progression",
            "semiology": "Sudden head nod or full body fall (atonic collapse); brief (< 2 seconds); no post-ictal confusion; may cluster; compound with progressive hypotonia; infant may not exhibit tonic phase; helmet recommended from atonic attack onset",
            "clinical_tip": "Protective helmet mandatory at atonic attack onset in CLN1 — head injury risk compounded by progressive ataxia and hypotonia. CLB 0.05-0.1 mg/kg/day for atonic seizures. Physiotherapy for hypotonia + ataxia management. Padded clothing + fall-protection environment. SUDEP risk: atonic fall into prone position in infant/toddler — monitoring essential.",
        },
    ]

    triggers = [
        {
            "trigger": "Fever-Febrile-Illness",
            "pct": 88,
            "description": "Fever and intercurrent illness are the most potent seizure triggers in CLN1; febrile SE common (febrile illness → myoclonic cluster + GTCS → prolonged SE); CLN1-specific: immune stress additionally accelerates NCL pathology during illness; aggressive antipyretics at FIRST sign of fever; rescue midazolam buccal for clusters",
            "management": "Antipyretics (paracetamol + ibuprofen alternating) aggressively at fever onset; temperature threshold for rescue AED: >38.5°C → parent-administered buccal midazolam 0.3 mg/kg; written febrile seizure action plan; SE plan: IV LEV 60 mg/kg in hospital (NOT fosphenytoin); hospital alert for CLN1 febrile SE",
        },
        {
            "trigger": "Missed-AED-Dose",
            "pct": 78,
            "description": "AED omission in infants/toddlers (feeding difficulties, medication palatability, vomiting) → seizure breakthrough; CLN1-specific: progressive dysphagia makes oral medication increasingly difficult; liquid formulations essential; NG/gastrostomy tube medication route planning required",
            "management": "Liquid formulations for all AEDs (VPA syrup/solution, LEV solution, CLB suspension); gastrostomy consideration when oral medication unreliable; electronic medication tracking (parent app); home nursing support for complex medication regimen; pharmacy liaison for compounding if needed",
        },
        {
            "trigger": "Sleep-Deprivation-Disruption",
            "pct": 72,
            "description": "Sleep deprivation lowers seizure threshold; CLN1-specific: seizure activity disrupts infant/toddler sleep; nocturnal GTCS disrupt sleep → daytime fatigue → lower threshold → cycle; myoclonus during drowsiness; progressive cortical suppression reduces sleep architecture",
            "management": "Sleep hygiene; melatonin 0.5-3 mg nocte for circadian support; CLB nocturnal dose reduces nocturnal GTCS; nocturnal pulse oximetry mandatory; SUDEP monitoring (infant cannot self-rescue); padded cot sides; family sleep support services",
        },
        {
            "trigger": "Photic-Visual-Stimulation",
            "pct": 35,
            "description": "Photosensitive component in 35% of CLN1 — less prominent than in adult PMEs; TV/screen flickering, sunlight through trees, flashing toys; note: CLN1 is NOT CLN2 (no pathognomonic 1-3 Hz SSPS response — CLN2-specific); VEP extinction makes formal IPS testing unreliable after 18 months (no retinal response to drive visual cortex)",
            "management": "Screen filters for TV/computer; avoid photosensitising toys; tinted glasses if photosensitive; standard IPS at diagnosis (when VEP still present); after VEP extinction, IPS results unreliable; educate carers re: photosensitive triggers",
        },
        {
            "trigger": "Tactile-Auditory-Startle",
            "pct": 62,
            "description": "Auditory and tactile startle triggers cortical myoclonus in CLN1; sudden sounds (clapping, door slam, vacuum cleaner), unexpected touch; carer education on predictable approach; environmental modification; less prominent than in CLN4B but significant especially in later disease when cortical hyperexcitability is severe",
            "management": "Predictable approach protocol (always speak before touching); reduce unexpected loud sounds in home; ear defenders for noisy environments; CLB for severe stimulus-sensitive myoclonus; sensory integration occupational therapy",
        },
        {
            "trigger": "Emotional-Stress-Agitation",
            "pct": 55,
            "description": "Agitation, crying, frustration (progressive cognitive decline → reduced communication → frustration behaviours) trigger myoclonic clusters; CLN1-specific: as cognition and communication deteriorate, unexpressed pain/discomfort/frustration manifest as agitation → seizure trigger; pain assessment critical (Paediatric Pain Profile for non-verbal children)",
            "management": "Augmentative and alternative communication (AAC) from early disease to maintain communication as long as possible; paediatric pain assessment (Paediatric Pain Profile) — identify and treat pain as seizure trigger; child psychology/play therapy; sensory comfort strategies; anti-epileptic management optimised to reduce seizure-driven agitation cycle",
        },
        {
            "trigger": "Metabolic-Stress-Dehydration",
            "pct": 48,
            "description": "Metabolic stress (dehydration, hypoglycaemia, electrolyte disturbance from GI illness) precipitates GTCS in CLN1; dysphagia in later disease reduces fluid intake → dehydration risk; VPA can cause hyperammonaemia under metabolic stress; ammonia level in acute deterioration on VPA",
            "management": "Adequate hydration maintenance; early NG/PEG tube feeding when dysphagia impairs fluid intake; electrolyte monitoring during illness; VPA hyperammonaemia monitoring during metabolic stress (ammonia level); carnitine supplementation if VPA-related deficiency; ER plan includes IV fluids + glucose",
        },
        {
            "trigger": "VGB-Administration",
            "pct": 100,
            "description": "VGB administration is ITSELF a trigger of catastrophic irreversible visual deterioration in CLN1 (and a seizure aggravator by GABAergic excess mechanism). This is listed as a trigger to emphasise: VGB must be prevented from reaching a CLN1 patient. Any infant with possible CLN1 presenting to ED must have VGB excluded from treatment protocols before diagnosis confirmed.",
            "management": "ABSOLUTE CI — VGB must not be administered in any clinical setting. Document in: hospital medication allergy field, ED alert, AED card, school medical records, GP letter, out-of-hours handover, paediatric resuscitation plan. Carry CLN1 diagnostic card. Parents trained to refuse VGB verbally in emergency settings.",
        },
    ]

    treatments = [
        {
            "drug": "Valproate (VPA)",
            "level": "Level-B",
            "dose": "Infantile: 20-40 mg/kg/day divided BD-TID; liquid formulation (VPA syrup/solution 200 mg/5 mL or 40 mg/mL); titrate to seizure control; trough target 60-100 mg/L; extended-release not available in very young infants — use immediate-release liquid; maintain throughout disease course (do not taper in progressive NCL)",
            "moa": "Multiple mechanisms: Na-channel, GABA-A potentiation, T-type Ca channel, HCN — broad-spectrum; effective for GTCS + myoclonus + infantile spasms component. SAFE in CLN1: PPT1 is a LYSOSOMAL SERINE HYDROLASE — NOT mitochondrial. VPA mitochondrial CI (MERRF/POLG) does NOT apply to CLN1. Backbone AED for infantile CLN1.",
            "efficacy": "82% of CLN1 cohort on VPA; good initial myoclonus + GTCS control; remains backbone despite progressive drug resistance as neurodegeneration advances; maintains seizure control longer than most alternatives in early-mid CLN1",
            "monitoring": "LFT + ammonia + FBC baseline and 3-monthly in infants (hepatotoxicity risk highest in children under 3 years on polytherapy); VPA trough 60-100 mg/L; weight; carnitine level (VPA-related carnitine depletion — supplement if <25 nmol/mL); hyperammonaemia monitoring (especially important in CLN1 — differentiate VPA encephalopathy from NCL progression); VPPP NOT applicable in infantile/toddler CLN1 (VPPP for females ≥12y)",
            "cln1_note": "VPA is SAFE in CLN1 (lysosomal serine hydrolase disease — NOT mitochondrial). Hepatotoxicity risk is highest in children under 3 on polytherapy — LFT monitoring 3-monthly essential. Hyperammonaemia may mimic CLN1 cognitive deterioration — check ammonia if acute worsening on VPA. Carnitine supplementation standard in CLN1 (long-term VPA + malnutrition risk from dysphagia). Liquid formulation essential.",
        },
        {
            "drug": "Levetiracetam (LEV)",
            "level": "Level-B",
            "dose": "Infantile/toddler: 20-60 mg/kg/day divided BD; IV LEV 60 mg/kg for SE; oral solution 100 mg/mL available (essential for infant dosing); weight-based dosing reviewed at every clinic; renally cleared — adjust for renal impairment (rare in young child but documented in late CLN1 multi-organ stress)",
            "moa": "SV2A modulation → reduced vesicle cycling → broad-spectrum AED. In CLN1 context: PPT1 lysosomal substrate accumulation → presynaptic dysfunction (secondary to neuronal degeneration); LEV SV2A modulation provides adjunct presynaptic stabilisation. Not as mechanistically specific as in CLN4B (where SV2A-CSPα direct complementarity) but empirically effective in infantile NCL epilepsy.",
            "efficacy": "72% of CLN1 cohort on LEV; effective adjunct to VPA for GTCS + myoclonus; IV LEV 60 mg/kg = first-line SE rescue (not fosphenytoin); oral solution aids late-disease medication delivery via gastrostomy; relatively well-tolerated in infants",
            "monitoring": "Behavioural side effects (irritability, aggression) in infants — important in CLN1 where agitation is a primary manifestation; differentiate LEV-induced irritability from CLN1 disease; dose-reduce or add CLB if significant behavioural activation; renal function monitoring in late disease",
            "cln1_note": "IV LEV 60 mg/kg is the MANDATORY first-line SE rescue in CLN1 — replace fosphenytoin in all emergency protocols (Na-channel blocker CI). Document in ED paediatric resuscitation notes. Oral solution (100 mg/mL) essential for infant dosing and later gastrostomy administration. Behavioural activation monitoring — irritability from LEV vs. CLN1 disease agitation: if onset coincides with LEV dose increase → dose-reduce.",
        },
        {
            "drug": "Clobazam (CLB)",
            "level": "Level-B",
            "dose": "0.1-0.3 mg/kg/day OD nocte (nocturnal GTCS prevention); 0.05-0.2 mg/kg/day for atonic + myoclonic component; titrate to seizure control; liquid formulation 1 mg/mL or suspension; tolerance monitoring every 3-6 months",
            "moa": "GABA-A PAM (benzodiazepine site) — 1,5-benzodiazepine (less sedating than clonazepam at comparable doses); nocturnal GTCS prevention; atonic drop attack reduction; myoclonus adjunct. Nocturnal dosing reduces SUDEP risk from nocturnal GTCS in CLN1.",
            "efficacy": "Adjunct for atonic drops + nocturnal GTCS + myoclonic clusters when VPA + LEV insufficient; tolerance a clinical management challenge (increase dose interval or drug holiday); useful in late disease for palliative comfort",
            "monitoring": "Tolerance (3-6 months — prescribe tolerance management plan); sedation (distinguish CLB sedation from CLN1 cognitive decline); paradoxical excitation in young children; feeding/suckling impairment if over-sedated; respiratory monitoring in late disease",
            "cln1_note": "CLB nocturnal dose is important for SUDEP risk reduction in CLN1 (compound SUDEP risk: nocturnal GTCS + infant cannot reposition). Tolerance management: consider 2-3 day drug holiday every 6-8 weeks; alternative to clonazepam for prolonged use (less tolerance accumulation). Liquid formulation essential in infants.",
        },
        {
            "drug": "Piracetam",
            "level": "Level-C",
            "dose": "Children: 50-100 mg/kg/day divided TID; start low (30 mg/kg/day) and titrate; piracetam oral solution available; renal dose adjustment (renally cleared); response assessment by UMRS or clinical myoclonus frequency",
            "moa": "AMPA positive modulation → reduced cortical reflex myoclonus amplitude; evidence in PME action myoclonus across syndromes; in CLN1, myoclonus is less action-triggered (unlike adult PMEs) but cortical component still present; adjunct benefit for multifocal cortical myoclonus reduction",
            "efficacy": "Level C evidence specifically in CLN1 (stronger evidence in adult PMEs); adjunct for refractory myoclonus when VPA + LEV + CLB insufficient; functional benefit limited in CLN1 due to rapid cognitive/motor decline (unlike adult PMEs where functional gains in handwriting etc. measurable)",
            "monitoring": "Renal function (age-appropriate — eGFR in older CLN1 patients); behavioural activation at high doses; adequate hydration; antiplatelet effect at high doses; UMRS response assessment",
            "cln1_note": "Piracetam is adjunct-only in CLN1 (Level C vs Level B in adult PMEs). Functional gains are harder to measure in rapidly deteriorating infants/toddlers vs. adult PME patients. Use when multifocal cortical myoclonus is frequent and causing distress despite VPA+LEV+CLB. Renal monitoring from 2 years.",
        },
        {
            "drug": "Ketogenic-Diet (KD)",
            "level": "Level-C",
            "dose": "Classical KD ratio 3:1 or 4:1 (fat:carb+protein); commenced under metabolic dietitian supervision; close metabolic monitoring (acidosis, hypoglycaemia, renal calculi, dyslipidaemia); requires specialist metabolic team co-management",
            "moa": "Ketosis → beta-hydroxybutyrate replaces glucose as CNS energy substrate → reduces neuronal excitability (multiple mechanisms: KATP channel activation, mTOR suppression, mitochondrial efficiency, GABA enhancement); evidence for drug-resistant infantile epilepsy broadly; may have additional neuroprotective benefit in lysosomal storage diseases (PPT1 substrate accumulation partially reduced by metabolic shift)",
            "efficacy": "Level C evidence in CLN1 specifically; 28% of cohort on KD as adjunct; more evidence in drug-resistant infantile epilepsies generally (Dravet, GLUT1 deficiency); CLN1-specific neuroprotective benefit under investigation (mouse model data suggests partial benefit); consider when ≥3 AEDs have failed in CLN1",
            "monitoring": "Urine ketones 3× daily; blood glucose daily (hypoglycaemia risk in infants); LFT + lipid profile 3-monthly; renal calculi (urine calcium:creatinine ratio); metabolic acidosis monitoring; growth parameters; vitamin and mineral supplementation mandatory; dietitian review monthly",
            "cln1_note": "KD is particularly useful in drug-resistant CLN1 infantile epilepsy — consider after ≥3 AED failures. CLN1 lysosomal storage disease: KD may reduce substrate accumulation by alternative metabolic pathway (under investigation in NCL mouse models). Requires metabolic dietitian + neurologist co-management. Gastrostomy enables reliable KD delivery when oral intake becomes unreliable. Avoid in VPA-KD combination (additive hepatotoxicity risk — monitor LFT closely).",
        },
        {
            "drug": "MDT-Palliative-Ophthalmology-Communication",
            "level": "Level-A",
            "dose": "Ophthalmology: ERG + VEP + fundoscopy 6-monthly until VEP extinguished, then 12-monthly; visual rehabilitation (sensory enrichment while vision present); AAC from diagnosis; palliative care team integration from diagnosis; gastroenterology for PEG/gastrostomy (dysphagia); physiotherapy for hypotonia + spasticity; child life specialist; family support services",
            "moa": "CLN1 is FATAL — no disease-modifying therapy. Quality of life maximisation: vision aids (magnification, contrast enhancement) while vision present; AAC preserves communication as language declines; physiotherapy maintains comfort and prevents contractures; PEG enables reliable nutrition and medication delivery; palliative care optimises symptom management (pain, seizures, comfort); family support prevents caregiver breakdown",
            "efficacy": "MDT palliative care from diagnosis is MANDATORY STANDARD OF CARE in CLN1. Early ophthalmology involvement captures the window of visual function before ERG/VEP extinction. AAC enables communication until very late disease. PEG gastrostomy standard at CLN1 dysphagia onset. ACP discussions (resuscitation, hospitalisation thresholds, place of care, end-of-life) initiated from diagnosis.",
            "monitoring": "ERG + VEP 6-monthly; ophthalmology fundoscopy; FEES (fibreoptic endoscopic evaluation of swallowing) for dysphagia monitoring; weight + growth charts; pain assessment (Paediatric Pain Profile); ACP review 6-monthly; sibling + parent psychological support; bereavement planning",
            "cln1_note": "Palliative care from DIAGNOSIS is non-optional in CLN1 (fatal disease, no disease-modifying therapy). ACP must be completed while parents have capacity to consider resuscitation/ventilation decisions — do not defer to terminal phase. BDSRA registry enrolment enables future trial eligibility (PPT1 gene therapy trials emerging). Ophthalmology relationship is key in CLN1 — visual rehabilitation specialist (VI habilitation) should be involved from earliest possible point to maximise visual learning before ERG extinction.",
        },
        {
            "drug": "Rescue-Midazolam-Buccal-IV-Lorazepam",
            "level": "Level-A",
            "dose": "Buccal midazolam 0.3 mg/kg (infant weight-based — standard paediatric SE protocol); maximum dose 10 mg; parent-trained administration; IV lorazepam 0.1 mg/kg if IV access established; SE protocol: buccal midazolam → IV LEV 60 mg/kg (NOT phenytoin/fosphenytoin) → anaesthetic team if refractory",
            "moa": "Rapid buccal absorption → CNS benzodiazepine site → terminates SE and prolonged seizures; buccal route avoids need for IV access in community SE; weight-based infant dosing critical",
            "efficacy": "Standard SE rescue for all paediatric epilepsy patients with GTCS; CLN1-specific: SE is common (febrile illness trigger) and potentially fatal; parent-administered buccal midazolam is life-saving; IV LEV 60 mg/kg for in-hospital SE",
            "monitoring": "Parent competency assessment (annual or with each clinic); buccal midazolam expiry date; respiration monitoring post-administration (brief apnoea possible); call 999 if: seizure >5 minutes + buccal midazolam given; document SE protocols in hospital notes and GP letter",
            "cln1_note": "SE IN CLN1 PAEDIATRIC EMERGENCY: (1) Buccal midazolam 0.3 mg/kg community. (2) IV LEV 60 mg/kg hospital (NOT phenytoin/fosphenytoin). (3) Paediatric anaesthetic team for refractory SE. Document in ED alert: 'CLN1 — AVOID phenytoin/fosphenytoin — use IV LEV.' Ensure hospital pharmacy stocks IV LEV in paediatric doses. AED emergency card carried by family at all times.",
        },
        {
            "drug": "BDSRA-Registry-Trial-Enrolment",
            "level": "Level-A",
            "dose": "BDSRA registry enrolment at diagnosis; NCL Resource contact; NCL Network Europe registration; natural history study participation; biobank sample donation (blood, CSF, fibroblasts for research); PPT1 gene therapy trial eligibility assessment",
            "moa": "No disease-modifying therapy currently approved for CLN1. Registry enrolment: enables natural history data contribution; trial eligibility assessment for emerging PPT1 gene therapy (AAV-PPT1 — preclinical efficacy demonstrated in CLN1 mouse model; Macauley SL et al. 2018 J Clin Invest); enzyme replacement therapy (PPT1 ICV delivery — analogous to CLN2 cerliponase alfa approach — under development); family support network access; bereavement support for sibling families",
            "efficacy": "Registry participation is the ONLY pathway to disease-modifying trial access in CLN1. AAV-PPT1 gene therapy shows efficacy in CLN1 mouse model (Macauley 2018) — human trials emerging. Early enrolment maximises trial eligibility window (early-stage disease has most therapeutic benefit).",
            "monitoring": "Annual update of registry phenotype data; trial eligibility re-assessment annually; natural history sample collection protocol; biobank consent maintenance; family contact with BDSRA support services",
            "cln1_note": "BDSRA enrolment is a clinical PRIORITY at CLN1 diagnosis — not optional. Families should understand that: (1) no current disease-modifying therapy; (2) gene therapy is emerging and the only realistic near-term option; (3) early enrolment maximises trial eligibility while disease is less advanced; (4) BDSRA provides peer support network for CLN1 families (a unique and essential source of support for a fatal paediatric disease).",
        },
    ]

    contraindications = [
        {
            "drug": "Vigabatrin (VGB)",
            "severity": "ABSOLUTE",
            "reason": "VGB irreversible peripheral visual field constriction (vigabatrin-associated retinopathy, VAR) + CLN1 progressive retinal NCL (100% retinal degeneration) = CATASTROPHIC, IRREVERSIBLE, ACCELERATED BLINDNESS. CLN1 retinal degeneration is already progressive and fatal to visual function; VGB superimposes irreversible additional retinal toxicity. ABSOLUTE CI — more severe than CLN2/CLN3 because CLN1 onset is infantile (longer potential life with compounded blindness). INFANTILE SPASMS TRAP: VGB is first-line for infantile spasms — MUST exclude CLN1 before VGB prescribed. PPT1 enzyme assay (DBS) before VGB in any infant with spasms + developmental regression + visual concern.",
            "alternative": "ACTH (2-8 units/day paediatric) + prednisolone for infantile spasms component. VPA + LEV backbone for ongoing CLN1 seizures. Ketogenic diet adjunct for refractory infantile epilepsy. NEVER VGB in CLN1 regardless of seizure type.",
        },
        {
            "drug": "Carbamazepine-Oxcarbazepine-Phenytoin",
            "severity": "ABSOLUTE",
            "reason": "Na-channel blockers WORSEN myoclonus in CLN1. Misdiagnosis trap: infantile seizures (GTCS, focal occipital seizures) → CBZ/PHT prescribed by general paediatrician before NCL recognised → ACUTE MYOCLONIC DETERIORATION. PAEDIATRIC ED TRAP: fosphenytoin is standard paediatric SE protocol BUT ABSOLUTE CI in CLN1 (Na-channel blocker — worsen myoclonus). IV LEV 60 mg/kg must replace fosphenytoin in ALL CLN1 SE protocols. Document in paediatric resuscitation notes: 'CLN1 — NO phenytoin/fosphenytoin — use IV LEV.'",
            "alternative": "VPA + LEV backbone (broad-spectrum: covers GTCS + myoclonus + focal seizures). IV LEV 60 mg/kg for SE. Stop CBZ/OXC immediately if patient already receiving it — VPA+LEV overlap.",
        },
        {
            "drug": "Tiagabine (TGB)",
            "severity": "ABSOLUTE",
            "reason": "NCSE risk in myoclonic epilepsies — Tiagabine GABA-reuptake inhibition → excessive CSF GABA → NCSE. ABSOLUTE CI across all NCL/PME syndromes including CLN1.",
            "alternative": "GABA-A modulation via benzodiazepines (CLB/clonazepam) — safe GABA-A PAM without NCSE risk of TGB reuptake inhibition.",
        },
        {
            "drug": "Fosphenytoin-IV-Phenytoin",
            "severity": "ABSOLUTE",
            "reason": "IV phenytoin/fosphenytoin in paediatric SE protocols is standard BUT ABSOLUTE CI in CLN1 (Na-channel blocker — worsens myoclonus). PAEDIATRIC ED TRAP: CLN1 child arrives in SE → paediatric ED team follows standard SE protocol → fosphenytoin administered → MYOCLONIC STATUS WORSENS. CLN1 diagnosis card must be carried; ED alert must specify: 'CLN1 — NO fosphenytoin — use IV LEV 60 mg/kg.' Parent education on refusing fosphenytoin in emergency.",
            "alternative": "IV LEV 60 mg/kg is the SE first-line in CLN1 (proven safety, no myoclonus aggravation, broad-spectrum). If refractory SE: IV clonazepam → anaesthetic team (propofol or ketamine — ketamine preferred if Na-channel blocker avoided).",
        },
        {
            "drug": "LTG-Monotherapy",
            "severity": "HIGH-RISK",
            "reason": "LTG monotherapy worsens myoclonus in CLN1. If LTG add-on considered for refractory focal seizures (3rd/4th line, with VPA backbone — note VPA halves LTG clearance, very low starting dose needed): NEVER first-line monotherapy. Risk in CLN1: infantile seizures + developmental regression may be misdiagnosed as Dravet syndrome → LTG → acute myoclonic worsening.",
            "alternative": "VPA + LEV backbone handles focal + generalised + myoclonic seizure spectrum in CLN1. Ketogenic diet for drug-resistant CLN1 before LTG add-on attempt.",
        },
        {
            "drug": "GBP-Pregabalin",
            "severity": "HIGH-RISK",
            "reason": "GBP/PGB worsen myoclonus (Crespel 1999). Rare in pure CLN1 (infantile disease). Relevant for CLN1 variant (late-infantile/juvenile form — older children with pain or spasticity may be seen by pain services). Document CI in all correspondence if CLN1 variant in older age group.",
            "alternative": "Amitriptyline low dose for neuropathic pain in older CLN1 variant; non-pharmacological pain management; palliative team for late-disease pain.",
        },
        {
            "drug": "VPA-Polytherapy-Under-3-Years",
            "severity": "HIGH-RISK",
            "reason": "VPA hepatotoxicity risk is HIGHEST in children under 2-3 years of age on polytherapy — Alpers-Huttenlocher mimicry risk. CLN1 is NOT POLG/mitochondrial (VPA safe) but the age-related hepatotoxicity risk still applies. POLG1 screening recommended before VPA initiation in any infant under 2 years (to exclude coincidental POLG1 mitochondrial disease where VPA is ABSOLUTE CI). Liver function monitoring 3-monthly in this age group. Carnitine supplementation standard in CLN1 on VPA.",
            "alternative": "If VPA hepatotoxicity suspected (acute LFT rise, vomiting, decreased consciousness): STOP VPA IMMEDIATELY; switch to LEV + CLB backbone; avoid re-challenge; POLG1 sequencing to exclude coincidental mitochondrial disease.",
        },
    ]

    monitoring = [
        {"item": "PPT1-Enzyme-Assay-DBS-Diagnosis", "rationale": "PPT1 enzyme assay on dried blood spot (DBS) — FIRST diagnostic test; results in 1-3 days; PPT1 activity <1% of normal = CLN1 confirmed; PPT1 10-30% residual = late-infantile/juvenile CLN1 variant; normal PPT1 (>30%) → exclude CLN1 → next: TPP1 enzyme assay for CLN2, then NCL gene panel. DIAGNOSTIC PRIORITY SEQUENCE: PPT1 enzyme → TPP1 enzyme → PPT1/TPP1 gene sequencing → NCL gene panel. PPT1 enzyme assay MUST come before genetic sequencing to expedite diagnosis (days vs. weeks)."},
        {"item": "PPT1-Gene-CLN1-Sequencing", "rationale": "CLN1/PPT1 full gene sequencing (exons + splice sites + deletion/duplication analysis) after confirmatory enzyme assay; p.Arg122Trp targeted PCR first (if Finnish heritage — 2-3 days result); ACMG variant classification; parental carrier testing; prenatal testing options; population-specific founder mutation panel; BDSRA registry variant submission"},
        {"item": "GRODs-Skin-Biopsy-EM", "rationale": "Skin biopsy electron microscopy: GRANULAR OSMIOPHILIC DEPOSITS (GRODs) in eccrine sweat gland epithelial cells, endothelial cells, lymphocytes — PATHOGNOMONIC for CLN1/PPT1 deficiency. GRODs are small, round, electron-dense granular deposits — distinct from CLN2 (curvilinear bodies + fingerprint profiles), CLN3 (combined fingerprint + curvilinear), CLN4B (fingerprint profiles only). Skin biopsy EM performed alongside PPT1 enzyme assay — confirmatory in 92% of CLN1. GRODs confirm NCL diagnosis while enzyme/genetic results pending."},
        {"item": "Ophthalmology-ERG-VEP-6Monthly", "rationale": "Electroretinogram (ERG) + Visual Evoked Potentials (VEP) + fundoscopy 6-monthly from diagnosis. ERG: abnormal from 6-12 months (retinal degeneration onset); progressive amplitude reduction → ERG extinction. VEP: extinguished by 12-18 months in classic CLN1 — PATHOGNOMONIC progression (complete VEP loss by 18 months is the pathognomonic CLN1 EEG-ophthalmology finding). Fundoscopy: macular pallor → optic atrophy → pigmentary retinopathy. VEP extinction correlates with IPS protocol failure (cannot formally assess photosensitivity after VEP extinction). CRITICAL: ERG/VEP trajectory documents disease progression; guides visual rehabilitation timing."},
        {"item": "EEG-Background-Suppression-Progression", "rationale": "Serial EEG every 6 months: document progressive background attenuation sequence — HIGH AMPLITUDE → PROGRESSIVE ATTENUATION → NEAR-ISOELECTRIC → ISOELECTRIC (by age 2-3y in classic CLN1). This progressive background suppression is PATHOGNOMONIC for CLN1 (distinct from CLN2 where SSPS at 1-3 Hz is pathognomonic; distinct from CLN3 with less rapid EEG deterioration). Document: background amplitude, sleep architecture, epileptiform discharge density, bursts. Isoelectric EEG milestone = advanced CLN1 — modify treatment to palliative comfort focus."},
        {"item": "Brain-MRI-3T-Cortical-Cerebellar", "rationale": "Brain MRI 3T every 6-12 months: progressive cortical atrophy (frontal > occipital in CLN1 — differs from CLN2 occipital predominance); cerebellar atrophy; white matter T2 changes (periventricular myelin loss); thalamic signal change; MRI trajectory correlates with clinical decline rate; useful for MDT prognosis discussions and ACP timing; MRI distress management (sedation/anaesthesia increasing in progressive CLN1)"},
        {"item": "Developmental-Assessment-Neuropsychology", "rationale": "Paediatric neuropsychological assessment 6-monthly: Bayley Scales of Infant and Toddler Development (Bayley-4) — motor + cognitive + language subscales; document milestone loss trajectory (regression monitoring); communication level assessment for AAC planning; functional vision assessment (alongside ophthalmology); pain/comfort assessment as language is lost (Paediatric Pain Profile); quality of life tool (Pediatric QoL — parent proxy)"},
        {"item": "Dysphagia-FEES-Gastrostomy-Planning", "rationale": "FEES (Fibreoptic Endoscopic Evaluation of Swallowing) 6-monthly from 12-18 months or at first dysphagia signs; modified diet textures when FEES shows aspiration risk; PEG gastrostomy planning: insert when: (a) inadequate oral caloric intake <75% estimated requirements for 3 months; (b) aspiration risk on FEES; (c) reliable medication delivery needed. Gastrostomy enables: reliable VPA/LEV/CLB liquid delivery, ketogenic diet feeds, reliable nutrition, reduced infection risk from aspiration. Do not delay PEG beyond the point when anaesthetic risk becomes prohibitive."},
        {"item": "VPA-TDM-LFT-Carnitine-NH3", "rationale": "VPA trough 60-100 mg/L; LFT + ammonia + FBC 3-monthly (heightened frequency in <3 years due to hepatotoxicity risk); carnitine level (supplement if <25 nmol/mL — VPA-related carnitine depletion common in CLN1 on long-term VPA with dysphagia); ammonia level if acute cognitive/behavioural deterioration on VPA (distinguish VPA encephalopathy from CLN1 progression); LFT — stop VPA and urgent hepatology if transaminases >5× ULN"},
        {"item": "SUDEP-Nocturnal-Infant-Monitoring", "rationale": "Compound SUDEP risk in CLN1: (1) nocturnal GTCS (88%); (2) infant/toddler cannot self-rescue prone position; (3) progressive hypotonia (cannot reposition after seizure); (4) rapid EEG suppression (diminished cortical arousal response). MANDATORY monitoring: nocturnal pulse oximetry + movement sensor; cot sensor/apnoea monitor; firm mattress; no pillow (suffocation risk in hypotonic infant post-seizure); padded cot sides; carer sleeping in room monitoring; nocturnal CLB dose. SUDEP counselling in ACP (age-appropriate language for parents)."},
        {"item": "BDSRA-NCL-Registry-Trial-Eligibility", "rationale": "BDSRA (Batten Disease Support and Research Association) registry + NCL Resource enrolment at diagnosis; phenotype + genotype data submission; natural history sample collection (blood, fibroblasts); PPT1 gene therapy trial eligibility assessment (AAV-PPT1 trials emerging; Macauley 2018 mouse model efficacy); annual trial eligibility update; BDSRA family support network (siblings, grandparent support); bereavement support planning; sibling carrier testing (if parents are carriers, each sibling has 25% risk)"},
        {"item": "ACP-Palliative-Care-From-Diagnosis", "rationale": "Advance Care Planning initiated at CLN1 diagnosis — not deferred to terminal phase. Decisions: resuscitation (DNAR), hospital admission thresholds, ventilation, gastrostomy, where the child will die. Palliative care team integration from diagnosis: symptom management (seizures, pain, secretions, dysphagia), quality of life maximisation, respite care, family psychological support. Children's hospice referral at diagnosis (not at end of life). Review ACP 6-monthly as disease advances. Sibling psychological support essential (witness to sibling's decline)."},
        {"item": "Genetic-Counselling-Carrier-Testing", "rationale": "AR inheritance → 25% sibling recurrence risk; both parents are obligate carriers (PPT1 heterozygous); sibling carrier testing + antenatal diagnosis options; prenatal testing for future pregnancies (PPT1 enzyme assay + sequencing on CVS or amniocentesis); PGT-M (preimplantation genetic testing for monogenic disease) for couples seeking unaffected children; extended family carrier testing (each parent's siblings at 50% carrier risk); Finnish heritage families: p.Arg122Trp targeted testing of extended family"},
        {"item": "VGB-Prevention-Alert-System", "rationale": "VGB ABSOLUTE CI in CLN1 — prevention system mandatory: (a) Hospital medication allergy field: 'Vigabatrin — CLN1 ABSOLUTE CI'; (b) ED paediatric alert system; (c) GP medication summary and allergy field; (d) School/respite health record; (e) Ambulance 'special patient note'; (f) AED emergency card carried by family; (g) Parent verbal training to refuse VGB in emergencies. Annual verification that VGB alert remains in ALL systems. Out-of-hours and locum clinician communication."},
    ]

    lifecycle = [
        {
            "stage": "Prenatal-Genetic-Risk",
            "age": "Prenatal / neonatal",
            "description": "AR inheritance → 25% sibling recurrence risk; antenatal diagnosis if CLN1 confirmed in family (PPT1 enzyme assay or targeted mutation on CVS/amniocentesis); PGT-M for couples planning future pregnancies; Finnish heritage families: p.Arg122Trp carrier screening programme; neonatal genetic testing if CLN1 family history confirmed.",
        },
        {
            "stage": "Pre-Symptomatic-Infancy",
            "age": "0-6 months",
            "description": "If known CLN1 family history: presymptomatic PPT1 enzyme assay (DBS) in neonatal period; genetic confirmation; ophthalmology baseline (ERG + VEP + fundoscopy); neurology enrolment; BDSRA registry; ACP preliminary discussion; trial enrolment if emerging PPT1 gene therapy trial available (earliest intervention window). No specific treatment required pre-symptomatically except optimising nutrition + developmental support.",
        },
        {
            "stage": "Developmental-Regression-Onset",
            "age": "6-18 months",
            "description": "FIRST SIGN: developmental regression or stagnation (motor/cognitive milestones loss); parental concern about development ± visual concern. DIAGNOSTIC PATHWAY: PPT1 enzyme assay DBS (1-3 days) + ophthalmology (ERG + VEP) + EEG + skin biopsy EM. VPA + LEV started when first seizures emerge. MDT assembled: neurology + ophthalmology + dietitian + physiotherapy + speech therapy + palliative care + genetics. ACP initiated.",
        },
        {
            "stage": "Active-Epilepsy-Rapid-Decline",
            "age": "12-36 months",
            "description": "Multifocal myoclonus + GTCS + atonic drops → AED optimisation (VPA + LEV + CLB ± piracetam ± KD). Progressive visual failure: ERG → VEP extinguished by 18 months. EEG: high-amplitude → progressive background suppression. Dysphagia emerging → FEES → PEG planning. AAC implementation as language declines. Progressive hypotonia: physiotherapy + postural support. Seizure management intensified. ACP reviewed.",
        },
        {
            "stage": "Established-CLN1-Severe-Disability",
            "age": "2-6 years",
            "description": "Profound cognitive + motor impairment; minimal voluntary movement; EEG near-isoelectric or isoelectric. Gastrostomy established. Seizure management: palliative-focus alongside seizure control (comfort vs. aggressive AED escalation discussion with family). Spasticity/dystonia management. Secretion management. Children's hospice input. Bereavement preparation support for family. ACP enacted for hospital admission decisions.",
        },
        {
            "stage": "Late-Palliative-End-Stage",
            "age": "5-12 years",
            "description": "End-stage CLN1: minimal/absent cortical activity (isoelectric EEG); vegetative state; gastrostomy feeding; maximal nursing care. Palliative AED management (symptom comfort, not seizure frequency reduction). Children's hospice care. ACP enacted: DNAR confirmed; home/hospice death vs. hospital decision made. Median survival 9-10 years (range 6-15y). Bereavement support planning. Sibling psychological support.",
        },
    ]

    return {
        "etiologies": etiologies,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "treatments": treatments,
        "contraindications": contraindications,
        "monitoring": monitoring,
        "lifecycle": lifecycle,
        "cohort_size": 40,
    }


def get_definitions():
    return {
        "concepts": [
            {
                "concept": "CLN1-PPT1-1p34.2-Lysosomal-Serine-Hydrolase-Infantile-NCL",
                "definition": "CLN1/PPT1 (1p34.2) encodes PPT1 (Palmitoyl-Protein Thioesterase 1), a 306-aa ~34 kDa lysosomal serine hydrolase. Catalytic triad: Ser115-Asp233-His289. Cleaves thioester bonds between palmitate and S-acylated cysteine residues in palmitoylated proteins. pH-optimum 4.0-5.0 (lysosomal activation). Ubiquitous; highest in neurons + retinal pigment epithelium. CLN1/PPT1 biallelic LOF → accumulated fatty-acid-modified proteins → GRODS on EM → progressive neuronal/retinal apoptosis → INCL (Santavuori-Haltia disease). LYSOSOMAL SERINE HYDROLASE disease — distinct from CLN2 (TPP1 lysosomal serine protease), CLN3 (Battenin lysosomal membrane protein), CLN4B (DNAJC5/CSPα presynaptic co-chaperone). OMIM: *600722 / #256730. Vesa J et al. 1995 Nature.",
                "standard": "Vesa-1995-Nature; Hofmann-1999-JBiolChem; NCL-Resource-2024; OMIM-*600722; ACMG-AMP-2015"
            },
            {
                "concept": "GRODs-Granular-Osmiophilic-Deposits-Pathognomonic-CLN1",
                "definition": "Granular Osmiophilic Deposits (GRODs) on electron microscopy of skin biopsy (eccrine sweat gland epithelial cells, endothelial cells) are PATHOGNOMONIC for CLN1/PPT1 deficiency. GRODs are small, round, electron-dense, granular deposits representing accumulated fatty-acid-modified proteins that PPT1 normally removes. CRITICAL DISTINCTION: GRODs (CLN1) vs. curvilinear bodies + fingerprint profiles (CLN2) vs. combined fingerprint + curvilinear (CLN3) vs. fingerprint profiles only (CLN4B). GRODs identify CLN1 even when enzyme assay is pending. Present in 92% of CLN1 skin biopsies. Skin biopsy EM (GRODs) + PPT1 enzyme assay (DBS) = the two-test CLN1 diagnostic confirmation.",
                "standard": "NCL-Resource-2024; Haltia-1973; Santavuori-1971; Mole-2019-LancetNeurol; Vesa-1995-Nature"
            },
            {
                "concept": "PPT1-Enzyme-Assay-DBS-Diagnostic-Priority-CLN1",
                "definition": "PPT1 enzyme assay on dried blood spot (DBS) provides CLN1 diagnosis in 1-3 days — MUST come before NCL gene panel (weeks) when CLN1 is suspected. Sequence: PPT1 enzyme (DBS, days) → if low → CLN1 confirmed → PPT1 gene sequencing for family counselling. If PPT1 normal → TPP1 enzyme assay (CLN2) → PPT1/TPP1 gene panel → NCL gene panel. CLN1 is TREATABLE in future (gene therapy trials emerging) — analogous urgency argument to CLN2 cerliponase (where TPP1 enzyme assay comes first). Reversed diagnostic sequence (genetics before enzyme) = delayed CLN1 diagnosis = missed trial enrolment window = potentially missed treatment. PPT1 enzyme assay is a simple, rapid, affordable DBS test available in most paediatric metabolic genetics laboratories.",
                "standard": "Hofmann-1999-JBiolChem; NCL-Resource-2024; ACMG-AMP-2015; Mole-2019-LancetNeurol"
            },
            {
                "concept": "EEG-Extinction-Progressive-Suppression-CLN1-Pathognomonic",
                "definition": "CLN1 EEG follows a PATHOGNOMONIC progression: (1) initially normal (0-8 months); (2) high-amplitude cortical activity (6-18 months — exuberant but disorganised); (3) progressive background attenuation (12-24 months — diminishing amplitude as cortical neurons die); (4) near-isoelectric (2-3 years); (5) complete isoelectric/EEG silence (3-5 years in classic CLN1). This sequential background suppression is pathognomonic for CLN1 (distinct from CLN2 where 1-3 Hz SSPS is pathognomonic; CLN3 where EEG attenuation is slower over years; CLN4B where EEG suppression occurs in adulthood). EEG isoelectric milestone = advanced disease — modify treatment to palliative/comfort focus; aggressive AED escalation has diminishing benefit at this stage.",
                "standard": "Santavuori-1971; NCL-Resource-2024; Mole-2019-LancetNeurol; ILAE-2022"
            },
            {
                "concept": "VEP-ERG-Extinction-Early-CLN1-Diagnostic",
                "definition": "Visual Evoked Potentials (VEP) are EXTINGUISHED by 12-18 months in classic CLN1 — among the earliest objective signs of CLN1 disease. ERG (electroretinogram) shows progressive amplitude reduction from 6-12 months, preceding VEP extinction. VEP extinction at <18 months of age is pathognomonic for CLN1 among infantile neurodegenerative diseases. Clinical sequence: normal ERG/VEP → ERG amplitude reduction (6-12 months) → complete ERG extinction → VEP extinction (12-18 months) → ophthalmoscopic changes (macular pallor, optic atrophy). VEP extinction distinguishes CLN1 from other infantile epilepsies (where VEP is preserved). ERG abnormality in infancy + developmental regression = CLN1 diagnostic pathway immediately.",
                "standard": "Santavuori-1971; NCL-Resource-2024; Mole-2019-LancetNeurol; ILAE-2022"
            },
            {
                "concept": "VGB-ABSOLUTE-CI-CLN1-Infantile-Spasms-Trap",
                "definition": "VGB is ABSOLUTE CI in CLN1 — causes irreversible peripheral visual field constriction (VAR) superimposed on CLN1 progressive retinal degeneration (100% retinal involvement) → catastrophic blindness acceleration. THE INFANTILE SPASMS TRAP: VGB is the first-line treatment for infantile spasms (UKISS trial; NICE guidance) — CLN1 may initially present as infantile spasms-like phenotype (regression + spasms). ANY INFANT with spasms + developmental regression + visual concern (abnormal ERG, poor fixation, nystagmus) MUST have PPT1 enzyme assay (DBS) before VGB is administered. If CLN1 confirmed → VGB PERMANENTLY EXCLUDED; use ACTH/prednisolone for spasms component. This is the most clinically dangerous drug-disease interaction in CLN1 management.",
                "standard": "NCL-Resource-2024; NICE-NG127; UKISS-2004; Mole-2019-LancetNeurol; ILAE-2022"
            },
            {
                "concept": "Finnish-Heritage-p.Arg122Trp-Founder-Mutation",
                "definition": "p.Arg122Trp (c.364C>T) is the Finnish founder mutation for CLN1/PPT1, accounting for ~60% of Finnish CLN1 alleles. CLN1 has the highest prevalence in Finland (estimated 1:20,000 live births vs. ~1:100,000 in non-Finnish European populations), attributable to this founder. Homozygous p.Arg122Trp = classic Santavuori-Haltia infantile NCL (onset 6-18 months, severe, fatal 7-10 years). Finnish heritage → targeted p.Arg122Trp PCR is first genetic test (result in 2-3 days before full PPT1 sequencing). Finnish healthcare pathway: Santavuori-Haltia disease has specific Finnish clinical guidelines; patients may be referred to Finnish NCL centres of excellence for consultation. Important for Canadian/North American populations: Finnish descent families (especially Ontario Mennonite / Finnish-Canadian heritage).",
                "standard": "Vesa-1995-Nature; NCL-Resource-2024; Santavuori-1971; ACMG-AMP-2015"
            },
            {
                "concept": "VPA-SAFE-CLN1-Lysosomal-NOT-Mitochondrial",
                "definition": "VPA is the backbone AED in CLN1 and IS SAFE. CLN1/PPT1 is a LYSOSOMAL SERINE HYDROLASE disease — NOT mitochondrial. VPA mitochondrial CI (MERRF/MT-TK, POLG/Alpers) does NOT apply to CLN1. However: (1) POLG1 screening recommended before VPA in children <2 years (to exclude coincidental POLG1 mitochondrial disease — POLG1 Alpers syndrome can present similarly with infantile regression + seizures; VPA ABSOLUTE CI in POLG1); (2) VPA hepatotoxicity risk is higher in children under 3 years on polytherapy — LFT 3-monthly mandatory; (3) Carnitine supplementation standard in CLN1 on VPA (malnutrition risk from dysphagia + VPA-related carnitine depletion). VPA must NOT be withheld from CLN1 patients based on incorrect mitochondrial concern.",
                "standard": "ILAE-2022; NICE-NG217; CPIC-POLG1-2023; MHRA-VPPP-2021; NCL-Resource-2024"
            },
            {
                "concept": "CBZ-OXC-PHT-FOSPHENYTOIN-ABSOLUTE-CI-Paediatric-ED-Trap",
                "definition": "Na-channel blockers are ABSOLUTE CI in CLN1. The PAEDIATRIC ED TRAP: CLN1 infant presents in SE → standard paediatric SE protocol → FOSPHENYTOIN administered → ACUTE MYOCLONIC WORSENING. Fosphenytoin is listed in standard paediatric SE algorithms but is ABSOLUTE CI in CLN1. CLN1 emergency protocol: buccal midazolam 0.3 mg/kg → IV LEV 60 mg/kg (NOT phenytoin/fosphenytoin) → anaesthetic team. Parents must be trained to verbally refuse fosphenytoin in ED. Hospital alert system must flag CLN1 = NO phenytoin/fosphenytoin. This is particularly dangerous because CLN1 infants frequently present to EDs before diagnosis is confirmed — symptom (infantile SE + regression) must trigger CLN1 screen AND fosphenytoin avoidance.",
                "standard": "Crespel-1999; ILAE-2022; NICE-NG217; NCL-Resource-2024; APLS-Guidelines"
            },
            {
                "concept": "No-Disease-Modifying-Therapy-CLN1-PPT1-Gene-Therapy-Emerging",
                "definition": "NO approved disease-modifying therapy for CLN1/PPT1 disease (in contrast to CLN2 which has cerliponase alfa). Management is purely symptomatic (AED + palliative MDT). However: PPT1 gene therapy (AAV-PPT1) shows efficacy in CLN1 mouse model (Macauley SL et al. 2018 J Clin Invest: AAV-PPT1 extended survival and reduced pathology in CLN1 mice). Human trials: AAV-PPT1 intracranial and intrathecal delivery trials are in early phase. Enzyme replacement: PPT1 ICV enzyme replacement analogous to cerliponase (CLN2) under investigation. MANDATORY: all CLN1 patients must be enrolled in BDSRA registry for trial eligibility — the ONLY pathway to disease-modifying therapy in the absence of approved treatment. Families must be explicitly told: no current treatment, but gene therapy is actively under development.",
                "standard": "Macauley-2018-JClinInvest; NCL-Resource-2024; BDSRA-Registry; Vesa-1995-Nature; ILAE-2022"
            },
            {
                "concept": "Fatal-Natural-History-CLN1-Infantile-ACP-Mandatory",
                "definition": "CLN1 is FATAL — median survival 9.4 years (range 6-15 years) in classic infantile form. Unlike CLN4B (adult onset; longer survival) and adult PMEs (non-fatal), CLN1 death occurs in childhood. Mandatory implications: (1) Palliative care from DIAGNOSIS — not from terminal phase; (2) ACP must be completed while parents have cognitive clarity and emotional capacity (at diagnosis/early disease, not at terminal crisis); (3) Children's hospice referral at diagnosis — primary centre of non-acute care; (4) Sibling psychological support mandatory — witnessing sibling's progressive decline; (5) Resuscitation decisions: DNAR, ventilation threshold, hospital admission threshold must be documented and regularly reviewed; (6) Place of death: home/hospice/hospital — plan in advance. CLN1 ACP framework is the same standard as CLN2 (both fatal) but in paediatric setting.",
                "standard": "NCL-Resource-2024; BDSRA-Registry; NICE-NG61-EndOfLife; WHO-PediatricPalliative-2018; ILAE-2022"
            },
            {
                "concept": "POLG1-Mandatory-Exclusion-Before-VPA-Infantile",
                "definition": "POLG1/Alpers-Huttenlocher syndrome is a critically important differential diagnosis for CLN1 (infantile regression + seizures + hepatic failure risk). POLG1 = mtDNA polymerase gamma deficiency → progressive neuronal degeneration (POLG1 syndrome). VPA is ABSOLUTE CI in POLG1 (severe mitochondrial hepatotoxicity → acute liver failure → death). CLN1 is NOT POLG1 (PPT1 = lysosomal serine hydrolase, NOT mitochondrial) — but coincidental POLG1 mutation can rarely coexist. POLG1 sequencing recommended before VPA in: any infant with regression + seizures + hepatic dysfunction, or family history of mitochondrial disease, or clinical features suggesting mitochondrial dysfunction (lactic acidosis, stroke-like episodes, ophthalmoplegia). If POLG1 is confirmed → VPA ABSOLUTE CI → LEV + CLB backbone only (NO VPA).",
                "standard": "CPIC-POLG1-2023; ILAE-2022; NCL-Resource-2024; NICE-NG217; Naviaux-2004"
            },
            {
                "concept": "Ketogenic-Diet-Adjunct-CLN1-Drug-Resistant",
                "definition": "Ketogenic diet (KD) is a Level C adjunct in CLN1 drug-resistant epilepsy. Rationale: (1) KD reduces neuronal hyperexcitability via KATP channel activation, mTOR suppression, GABA enhancement; (2) CLN1-specific: beta-hydroxybutyrate may reduce lysosomal substrate accumulation via alternative metabolic pathways (NCL mouse model data emerging); (3) KD is well-established for drug-resistant infantile epilepsies (Dravet, GLUT1 deficiency, tuberous sclerosis). Consider KD after ≥3 AED failures in CLN1. Gastrostomy enables reliable KD delivery when oral intake unreliable (standard in CLN1 progressive dysphagia). KD + VPA: additive hepatotoxicity risk — monitor LFT closely if co-administered; consider reducing VPA dose when KD commenced.",
                "standard": "Kossoff-2018-Epilepsia; NCL-Resource-2024; NICE-NG217; Macauley-2018-JClinInvest"
            },
            {
                "concept": "Gastrostomy-PEG-CLN1-Dysphagia-Standard",
                "definition": "PEG gastrostomy is a STANDARD intervention in CLN1 (not an option). CLN1 progressive dysphagia: swallowing dysfunction emerges from 12-24 months due to progressive bulbar neurodegeneration. Implications: (1) unreliable oral AED delivery → breakthrough seizures; (2) aspiration risk → aspiration pneumonia (leading cause of CLN1 death); (3) malnutrition from inadequate oral intake. PEG insertion: recommended when oral caloric intake <75% of estimated requirements for 3 months OR FEES shows aspiration risk OR reliable medication delivery fails. Insert before respiratory compromise makes anaesthesia prohibitive. Benefits: reliable liquid AED delivery, KD feeds, reliable nutrition, reduced aspiration. Coordinate with VPA/LEV/CLB liquid formulation pharmacy dispensing for gastrostomy delivery.",
                "standard": "NCL-Resource-2024; BDSRA-Registry; NICE-NG61; Paediatric-Dietetics-Standards-UK"
            },
            {
                "concept": "SUDEP-Risk-Infant-CLN1-Compound",
                "definition": "CLN1 SUDEP risk is compounded by four factors unique to the infantile setting: (1) nocturnal GTCS (88%); (2) infant/toddler CANNOT self-rescue from prone position post-seizure; (3) progressive hypotonia (infant cannot reposition independently); (4) rapid EEG background suppression (diminished cortical arousal response). Monitoring mandatory from onset: nocturnal pulse oximetry + movement sensor (mattress sensor or wearable); no pillow (suffocation risk in hypotonic infant); padded cot sides; firm mattress; carer sleeping in same room; CLB nocturnal dose reduces nocturnal GTCS frequency. SUDEP counselling in ACP (empathic, non-traumatising language appropriate for devastated parents of a fatally ill child). Nocturnal monitoring = standard of care in CLN1.",
                "standard": "Devinsky-2011-Lancet; NICE-NG217; SUDEP-Action-UK; NCL-Resource-2024; BDSRA-Registry"
            },
        ],
        "thresholds": [
            {"threshold": "PPT1 enzyme activity <1% of normal on DBS → CLN1 diagnosis confirmed", "standard": "Hofmann-1999-JBiolChem / NCL-Resource-2024"},
            {"threshold": "PPT1 enzyme activity 1-10% residual → late-infantile/juvenile CLN1 variant (slower course)", "standard": "NCL-Resource-2024 / ACMG-AMP-2015"},
            {"threshold": "VEP extinguished by 18 months → CLN1 pathognomonic — ophthalmology + EEG urgent", "standard": "Santavuori-1971 / NCL-Resource-2024"},
            {"threshold": "EEG isoelectric by age 3 years → advanced CLN1 — ACP review + palliative focus", "standard": "NCL-Resource-2024 / ILAE-2022"},
            {"threshold": "IV LEV 60 mg/kg → SE first-line rescue in CLN1 (NOT fosphenytoin)", "standard": "APLS-Guidelines / NCL-Resource-2024"},
            {"threshold": "VPA trough 60-100 mg/L → therapeutic target CLN1", "standard": "Standard VPA pharmacokinetics"},
            {"threshold": "LFT >5× ULN on VPA → STOP VPA immediately + urgent hepatology", "standard": "MHRA-VPA-2021 / CPIC-2023"},
            {"threshold": "Carnitine <25 nmol/mL on VPA → supplement with L-carnitine 50-100 mg/kg/day", "standard": "Standard VPA monitoring"},
            {"threshold": "Oral intake <75% requirements × 3 months → PEG gastrostomy referral", "standard": "NCL-Resource-2024 / Paediatric-Dietetics"},
            {"threshold": "FEES aspiration risk → immediate PEG referral + modified diet", "standard": "NCL-Resource-2024 / FEES-Protocol"},
            {"threshold": "Ammonia >80 µmol/L on VPA + acute cognitive worsening → VPA-encephalopathy suspected", "standard": "Standard VPA pharmacology"},
            {"threshold": "Spasms + regression + abnormal ERG/VEP → PPT1 enzyme assay (DBS) BEFORE VGB", "standard": "NCL-Resource-2024 / ILAE-2022"},
        ],
        "standards": [
            {"standard": "Vesa-1995-Nature", "detail": "Vesa J et al. 1995 — PPT1 identification as CLN1 disease gene; Nature 376:584-587; infantile neuronal ceroid lipofuscinosis gene cloning"},
            {"standard": "Santavuori-1971-ActaPaed", "detail": "Santavuori P et al. 1971 — original clinical description of Infantile NCL (Santavuori-Haltia disease); Acta Paediatrica Scandinavica"},
            {"standard": "Hofmann-1999-JBiolChem", "detail": "Hofmann SL et al. 1999 — PPT1 enzyme assay development and diagnostic criteria; Journal of Biological Chemistry; DBS assay validation"},
            {"standard": "NCL-Resource-2024", "detail": "Neuronal Ceroid Lipofuscinosis Network (NCL Resource) — diagnostic and management guidelines 2024; CLN1/PPT1 infantile NCL section"},
            {"standard": "ILAE-2022", "detail": "ILAE Classification of Epilepsies and Epilepsy Syndromes — CLN1 / infantile NCL classification; AED selection in infantile NCL"},
            {"standard": "NICE-NG217", "detail": "NICE Guideline NG217 — Epilepsies in children, young people, and adults; AED selection including infantile NCL"},
            {"standard": "Macauley-2018-JClinInvest", "detail": "Macauley SL et al. 2018 — AAV-PPT1 gene therapy efficacy in CLN1 mouse model; J Clin Invest; preclinical basis for human gene therapy trials"},
            {"standard": "MHRA-VPPP-2021", "detail": "MHRA Valproate Pregnancy Prevention Programme — VPA monitoring; hepatotoxicity risk in young children (applicable to CLN1 VPA use)"},
            {"standard": "Mole-2019-LancetNeurol", "detail": "Mole SE et al. 2019 — NCL diseases: review, classification and clinical update; Lancet Neurology; CLN1 clinical characterisation and diagnostic criteria"},
            {"standard": "CPIC-POLG1-2023", "detail": "CPIC Guideline for POLG1 — VPA absolute CI in POLG1/Alpers; mandatory exclusion before VPA in infantile regression syndromes"},
            {"standard": "ACMG-AMP-2015", "detail": "ACMG/AMP Variant Interpretation Standards — PPT1 variant pathogenicity classification; enzyme activity as functional evidence"},
            {"standard": "BDSRA-Registry", "detail": "Batten Disease Support and Research Association (BDSRA) — CLN1 patient registry; natural history data; AAV-PPT1 trial eligibility; family support network"},
        ],
        "references": [
            {"ref": "Vesa-1995", "citation": "Vesa J et al. Mutations in the palmitoyl protein thioesterase gene causing infantile neuronal ceroid lipofuscinosis. Nature 1995;376:584-587."},
            {"ref": "Santavuori-1971", "citation": "Santavuori P et al. A new form of neuronal ceroid-lipofuscinosis in childhood. Acta Paediatrica Scandinavica 1971;233(Suppl):1-7."},
            {"ref": "Hofmann-1999", "citation": "Hofmann SL et al. Palmitoyl-protein thioesterase 1 (PPT1) defect in infantile neuronal ceroid lipofuscinosis. Molecular Genetics and Metabolism 1999;66:288-292."},
            {"ref": "Mole-2019", "citation": "Mole SE et al. Neuronal ceroid lipofuscinoses (NCLs): review. Lancet Neurology 2019;18:1004-1013. DOI:10.1016/S1474-4422(19)30167-0."},
            {"ref": "Macauley-2018", "citation": "Macauley SL et al. Intrathecal delivery of CLN1 rescues disease pathology in a mouse model of infantile neuronal ceroid lipofuscinosis. J Clin Invest 2018;128:3707-3720."},
            {"ref": "Haltia-1973", "citation": "Haltia M et al. Infantile type of so-called neuronal ceroid-lipofuscinosis. 2. Morphological and biochemical studies. Acta Neuropathologica 1973;26:157-170."},
        ],
    }
