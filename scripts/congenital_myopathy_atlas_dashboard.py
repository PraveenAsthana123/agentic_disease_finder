#!/usr/bin/env python3
"""Congenital Myopathy Atlas — Complete 8-Gene Congenital Myopathy Atlas
NEB    (Nebulin; ~8045 aa; 2q23.3; Nemaline Myopathy type 2 NEM2; AR; largest sarcomeric gene ~183 exons; exon 55 deletion Ashkenazi founder; nemaline rods Gomori trichrome PATHOGNOMONIC; respiratory failure before limb weakness; NIV cornerstone) ·
RYR1   (Ryanodine Receptor 1; ~5038 aa; 19q13.2; Central Core Disease CCD + Multi-minicore; AD/AR; central cores NADH-TR PATHOGNOMONIC; MALIGNANT HYPERTHERMIA MHS1 — volatile anaesthetics + succinylcholine ABSOLUTELY CONTRAINDICATED; dantrolene emergency) ·
ACTA1  (Skeletal muscle alpha-actin; ~377 aa; 1q42.13; Nemaline Myopathy type 3 NEM3; AD de novo/AR; intranuclear rods PATHOGNOMONIC; neonatal lethal form 75% de novo; wide phenotypic spectrum) ·
TPM2   (Beta-tropomyosin; ~284 aa; 9p13.3; Nemaline Myopathy type 4 NEM4 + Distal Arthrogryposis; AD; trismus PATHOGNOMONIC; cap disease; distal arthrogryposis type 1) ·
TPM3   (Alpha-tropomyosin slow; ~284 aa; 1q21.3; Nemaline Myopathy type 1 NEM1; AD/AR; type 1 fiber uniformity/hypotrophy hallmark; CFTD; mild proximal weakness; head drop) ·
MTM1   (Myotubularin 1; ~603 aa; Xq28; X-linked Myotubular Myopathy XLMTM; XLR; neonatal onset profound hypotonia respiratory failure; centrally placed nuclei PATHOGNOMONIC; hepatopathy 10%; female carriers usually unaffected) ·
DNM2   (Dynamin 2; ~870 aa; 19p13.2; Centronuclear Myopathy type 1 CNM1; AD; centrally placed nuclei + perinuclear halo PATHOGNOMONIC; ptosis + ophthalmoplegia; CMT2M allelic; milder than XLMTM) ·
SELENON (Selenoprotein N; ~590 aa; 1p36.11; Rigid Spine Muscular Dystrophy 1 RSMD1; AR; rigid spine + contractures BEFORE limb weakness; respiratory failure precedes limb weakness; NIV MANDATORY early)
320-patient aggregate cohort (8 × 40, seeds 1086–1093)
"""

import random

SEED_BASE = 1086

CM_GENES = [
    # ── NEB — Nemaline Myopathy 2 ────────────────────────────────────────────
    {
        "gene": "NEB", "protein": "Nebulin (NEB)",
        "alias": "NEB; OMIM gene 161650; 2q23.3; ~8045 aa; Nemaline Myopathy type 2 (NEM2; OMIM #256030); AR; largest sarcomeric gene ~183 exons; exon 55 Ashkenazi Jewish founder deletion",
        "aa": "~8045 aa", "kDa": "~773 kDa",
        "mechanism": (
            "NEB encodes nebulin, one of the largest human proteins and the largest sarcomeric protein. "
            "It spans the entire length of the thin filament (actin) in skeletal muscle, acting as a "
            "molecular ruler that determines thin filament length and regulates actin dynamics. "
            "NORMAL FUNCTION: nebulin is a ~800 kDa modular protein composed of ~185 actin-binding "
            "modules; it provides structural backbone for the thin filament, regulates actin-myosin "
            "interaction, and participates in force production. "
            "NEM2 PATHOMECHANISM: biallelic loss-of-function variants (truncating, missense, "
            "exon deletions) → nebulin absent or truncated → thin filament length dysregulated → "
            "sarcomere dysfunction → weakness. Concurrently, nemaline rods form from disorganised "
            "thin filament components (predominantly Z-disc material — alpha-actinin + actin) that "
            "accumulate in subsarcolemmal and nuclear regions. "
            "GENE SIZE: NEB is the largest gene in the human genome (~249 kb genomic DNA, ~183 exons); "
            "standard sequencing panels may miss deep intronic variants and large deletions — "
            "MLPA (Multiplex Ligation-dependent Probe Amplification) for large exonic deletions is essential. "
            "ASHKENAZI FOUNDER: deletion of exon 55 (c.24465+1G>T or similar) is present in "
            "~1 in 108 Ashkenazi Jewish carriers — by far the most common NEB pathogenic variant worldwide. "
            "FIBRE TYPE: type 1 fibre predominance and type 1 fibre hypotrophy on biopsy — characteristic "
            "even in the absence of visible rods in some fibres."
        ),
        "disease_type": "Nemaline Myopathy type 2 (NEM2; AR biallelic NEB LOF; nemaline rods on Gomori trichrome PATHOGNOMONIC; largest sarcomeric gene)",
        "locus": "2q23.3", "omim_gene": 161650, "omim_disease": 256030,
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic pathogenic variants. "
            "NEB mutations account for approximately 50% of all genetically confirmed nemaline myopathy cases. "
            "Most patients are compound heterozygotes; the exon 55 deletion allele (p.Ser3029Leufs) "
            "is the most common single allele in Ashkenazi Jewish patients (~1:108 carrier frequency). "
            "FAMILY SCREENING: parents are obligate carriers; siblings have 25% recurrence risk. "
            "GENETIC TESTING STRATEGY: sequencing alone misses large exonic deletions → "
            "MLPA or array-CGH MANDATORY as complementary test; "
            "comprehensive NEB panel covering all 183 exons required. "
            "PRENATAL: available if both parental variants identified."
        ),
        "phenotype": (
            "NEMALINE MYOPATHY TYPE 2 (NEM2): "
            "ONSET: Congenital (most common) to childhood; occasionally milder/later. "
            "MOTOR: proximal > distal weakness; facial weakness (ptosis, nasal speech, "
            "open mouth posture); neck flexor weakness; foot drop in ambulant patients; "
            "severity ranges from neonatal lethal to mild adult-onset. "
            "RESPIRATORY: RESPIRATORY FAILURE may PRECEDE significant limb weakness in severe forms — "
            "diaphragm and intercostal muscle involvement; annual spirometry + sleep study MANDATORY; "
            "NIV is the cornerstone of life-prolonging treatment. "
            "CARDIAC: usually NOT involved (distinguishes from ACTA1 in some alleles; "
            "cardiac monitoring still recommended). "
            "BIOPSY HALLMARK: NEMALINE RODS on Gomori trichrome (red/purple rods on green background) — "
            "PATHOGNOMONIC; predominantly subsarcolemmal and perinuclear distribution; "
            "electron microscopy (EM) confirms rod structure (Z-disc-derived). "
            "MUSCLE MRI: diffuse involvement; no pathognomonic pattern (distinguishes from RYR1 cores). "
            "FEEDING: feeding difficulties + nasogastric tube feeding in neonatal severe form."
        ),
        "treatment_options": [
            "Non-invasive ventilation (NIV) — BiPAP/CPAP: CORNERSTONE of treatment; "
            "MANDATORY annual spirometry (FVC) + overnight sleep study (oximetry + CO2); "
            "start NIV at FVC <50% or evidence of nocturnal hypoventilation (desaturation, hypercapnia); "
            "tracheostomy if NIV fails or bulbar weakness prevents effective non-invasive support; "
            "respiratory management is the primary determinant of survival in NEB-NEM2",
            "Physiotherapy and orthotic management: regular physiotherapy to maintain muscle strength "
            "and prevent contractures; ankle-foot orthoses (AFOs) for foot drop; "
            "hydrotherapy beneficial for proximal weakness; avoid prolonged bed rest",
            "Nutritional support: feeding assessment by speech-language pathologist (SLP); "
            "nasogastric or gastrostomy tube feeding (PEG) for severe dysphagia or inadequate caloric intake; "
            "high-calorie dietary supplementation; weight monitoring — malnutrition accelerates weakness",
            "Scoliosis management: spinal monitoring from early childhood; "
            "spinal fusion surgery when Cobb angle >40-50 degrees or progressive; "
            "scoliosis worsens restrictive lung disease — coordinate spine and respiratory care",
            "L-tyrosine: anecdotal and small-series reports of benefit in nemaline myopathy; "
            "not proven in RCT; considered experimental but low risk; some centres prescribe; "
            "mechanism unclear (possible actin-myosin interaction modulation)",
            "Salbutamol (albuterol): beta-2 agonist; some open-label evidence of mild functional "
            "benefit in nemaline myopathy; not standard of care; "
            "used at some centres in ambulant patients with preserved function",
            "NO DISEASE-MODIFYING THERAPY: no approved cure; management is supportive; "
            "gene therapy and exon skipping strategies in preclinical/early clinical development; "
            "trial participation encouraged; registry enrolment (Congenital Myopathy Research Registry)",
        ],
        "key_ddx": [
            "ACTA1 nemaline myopathy (NEM3) — de novo AD; intranuclear rods on EM; "
            "neonatal lethal form more common; actin accumulation bodies; WES/panel distinguishes",
            "TPM2/TPM3 nemaline myopathy (NEM4/NEM1) — AD; milder; type 1 fibre uniformity; "
            "cap disease (TPM2); diurnal variation not present; genetic panel required",
            "SELENON myopathy (RSMD1) — AR; rigid spine; minicore lesions NOT nemaline rods; "
            "respiratory failure disproportionate; spine radiograph shows rigidity",
            "Congenital muscular dystrophy (various) — elevated CK; necrosis on biopsy; "
            "dystrophic changes vs nemaline rods; lamin/collagen subtypes",
            "Spinal muscular atrophy (SMA — SMN1) — motor neuron disease; "
            "EMG denervation pattern; no nemaline rods on biopsy; SMN1 deletion on genetic test",
        ],
        "onset_range_y": (0.0, 2.0),
        "sex_female_prob": 0.50,
        "myopathy_type": "Nemaline rods (Gomori trichrome PATHOGNOMONIC) / Type 1 fibre predominance",
        "severity_dist": {"Severe": 0.60, "Moderate": 0.30, "Mild": 0.10},
        "ventilator_dependent_rate": 0.55,
        "mh_risk": False,
        "hepatopathy_rate": 0.0,
        "progression_rate": 0.70,
        "drug_error_rate": 0.08,
        "targeted_therapy_available": False,
        "first_line_drug": "NIV (BiPAP/CPAP) — annual spirometry + sleep study mandatory",
        "critical_avoid": "RESPIRATORY FAILURE can PRECEDE significant limb weakness — MANDATORY annual spirometry + sleep study; start NIV at FVC <50% or nocturnal hypoventilation; MLPA MANDATORY (exon 55 deletion missed by sequencing alone); EXON 55 DELETION: most common NEB allele in ASHKENAZI patients; AVOID succinylcholine (myopathic hyperkalaemia risk — universal congenital myopathy precaution)",
    },
    # ── RYR1 — Central Core Disease / Malignant Hyperthermia ─────────────────
    {
        "gene": "RYR1", "protein": "Ryanodine Receptor 1 (RYR1)",
        "alias": "RYR1; OMIM gene 180901; 19q13.2; ~5038 aa; Central Core Disease (CCD; OMIM #117000) + Multi-minicore Disease; AD/AR; MALIGNANT HYPERTHERMIA SUSCEPTIBILITY MHS1 — highest risk",
        "aa": "~5038 aa", "kDa": "~565 kDa",
        "mechanism": (
            "RYR1 encodes the ryanodine receptor type 1, the principal calcium release channel of the "
            "sarcoplasmic reticulum (SR) in skeletal muscle. "
            "NORMAL FUNCTION: RYR1 forms a homotetrameric channel spanning the SR membrane; "
            "during excitation-contraction (EC) coupling, action potential-induced DHPR (voltage sensor) "
            "conformational change directly gates RYR1 → SR calcium release → actomyosin activation → contraction. "
            "PATHOMECHANISM (two distinct mechanisms by mutation class): "
            "(1) GAIN-OF-FUNCTION (hypersensitive/leaky channel — MH + Central Core): "
            "AD variants (p.Arg614Cys, p.Arg2163Cys, p.Arg2458His etc.) → RYR1 opens abnormally → "
            "sustained calcium release → depletion of SR calcium → muscle weakness AND "
            "under anaesthetic triggers (volatile agents + succinylcholine): "
            "uncontrolled calcium release → MALIGNANT HYPERTHERMIA (MH) crisis — "
            "hyperthermia, rigidity, rhabdomyolysis, hypermetabolic state, fatal if untreated. "
            "(2) LOSS-OF-FUNCTION (reduced channel opening — biallelic AR multi-minicore): "
            "AR variants → reduced or absent RYR1 → impaired EC coupling → severe myopathy without MH. "
            "HISTOLOGY: CENTRAL CORES — circumscribed regions devoid of mitochondria and oxidative enzyme "
            "activity in type 1 fibres, visible as pale central areas on NADH-TR (diaphorase) stain. "
            "Type 1 fibre predominance universal. MH LOCUS: MHS1 (19q13.2) — highest MH risk of all MH loci."
        ),
        "disease_type": "Central Core Disease (CCD; AD/AR RYR1; PATHOGNOMONIC central cores on NADH-TR; MHS1 MALIGNANT HYPERTHERMIA — volatile agents + succinylcholine ABSOLUTELY CONTRAINDICATED)",
        "locus": "19q13.2", "omim_gene": 180901, "omim_disease": 117000,
        "inheritance": (
            "AUTOSOMAL DOMINANT (typical Central Core Disease — AD GOF variants) AND "
            "AUTOSOMAL RECESSIVE (multi-minicore disease, severe myopathy — AR biallelic LOF variants). "
            "AD variants: many are de novo; penetrance near complete for cores; MH risk 100% for AD GOF variants. "
            "AR variants: biallelic; severe multi-minicore disease with ophthalmoplegia; MH risk variable. "
            "MH SUSCEPTIBILITY TESTING: In vitro contracture test (IVCT — Europe) / "
            "Caffeine-Halothane Contracture Test (CHCT — North America) confirms MH susceptibility "
            "in families without genetic confirmation; positive IVCT/CHCT = MH susceptible. "
            "FAMILY SCREENING: all first-degree relatives of AD CCD/MH patients → IVCT/CHCT or genetic testing; "
            "MH alert card should be issued to all RYR1 AD GOF carriers. "
            "RYR1 is the most commonly mutated gene in congenital myopathy overall."
        ),
        "phenotype": (
            "CENTRAL CORE DISEASE (CCD) — AD form: "
            "ONSET: Congenital to early childhood; mild presentation common (many diagnosed incidentally). "
            "MOTOR: proximal > distal weakness; hypotonia; delayed motor milestones; "
            "most patients remain ambulant; mild-moderate functional limitation in typical AD CCD. "
            "HIP DISLOCATION: congenital hip dislocation common — may be presenting feature. "
            "OPHTHALMOPLEGIA: NOT typical in AD CCD (distinguishes from AR multi-minicore). "
            "RESPIRATORY: usually mild in AD CCD; more significant in severe AR forms. "
            "CK: usually normal or mildly elevated (distinguishes from muscular dystrophies). "
            "BIOPSY HALLMARK: CENTRAL CORES — pale circular/oval areas devoid of mitochondria "
            "in type 1 fibres on NADH-TR (oxidative stain); PATHOGNOMONIC; type 1 fibre uniformity. "
            "MULTI-MINICORE DISEASE (AR biallelic): SEVERE; widespread small cores on biopsy; "
            "ophthalmoplegia (in some AR RYR1 forms); scoliosis; significant respiratory involvement; "
            "overlap with SELENON myopathy. "
            "MALIGNANT HYPERTHERMIA: explosive onset during/after anaesthesia with volatile agents "
            "or succinylcholine; masseter spasm may be first sign; "
            "tachycardia → hyperthermia → rigidity → rhabdomyolysis → metabolic acidosis → death if untreated."
        ),
        "treatment_options": [
            "MALIGNANT HYPERTHERMIA EMERGENCY — DANTROLENE: "
            "Dantrolene sodium 2.5 mg/kg IV bolus IMMEDIATELY on MH diagnosis; "
            "repeat 1mg/kg bolus every 5-10 min until MH resolves (max 10 mg/kg initial episode); "
            "MECHANISM: dantrolene blocks RYR1 calcium release → terminates MH; "
            "MUST BE AVAILABLE in ALL operating theatres; MH crisis drills mandatory",
            "ANAESTHESIA ALERT: ABSOLUTELY CONTRAINDICATED in all RYR1 AD GOF/MH susceptible patients — "
            "volatile anaesthetic agents (halothane, sevoflurane, desflurane, isoflurane, enflurane) + "
            "succinylcholine (suxamethonium); "
            "SAFE ALTERNATIVES: total intravenous anaesthesia (TIVA) with propofol + non-depolarising "
            "NMBDs (rocuronium, vecuronium, atracurium); "
            "MH ALERT CARD issued to all patients and families; medical alert bracelet recommended",
            "Physiotherapy and orthotics: regular physiotherapy; orthopaedic management of hip dislocation "
            "(congenital hip dislocation — orthopaedic referral at diagnosis); scoliosis monitoring; "
            "AFOs for distal weakness; hydrotherapy beneficial",
            "Respiratory monitoring: annual spirometry for moderate/severe cases; "
            "sleep study if symptomatic nocturnal hypoventilation; "
            "NIV if FVC <50% or nocturnal hypoventilation; "
            "respiratory involvement less prominent in typical AD CCD vs AR multi-minicore",
            "IVCT/CHCT testing for family members: all first-degree relatives of index patient → "
            "IVCT (Europe) or CHCT (North America) for MH susceptibility confirmation; "
            "genetic cascade testing if pathogenic variant identified; "
            "EUROMAC registry participation recommended",
            "Ophthalmology referral: assess for ophthalmoplegia (especially AR RYR1 multi-minicore forms); "
            "distinguish from mitochondrial myopathy (lactate, muscle biopsy, mtDNA).",
        ],
        "key_ddx": [
            "NEB nemaline myopathy (NEM2) — AR; nemaline rods not cores; diffuse biopsy MRI; NEB gene",
            "SELENON (RSMD1) — AR; minicore lesions; rigid spine syndrome; 1p36.11 gene",
            "Mitochondrial myopathy (MELAS, MERRF, CPEO) — ragged red fibres; lactic acidosis; "
            "abnormal COX staining; mtDNA or nuclear mtDNA gene mutations",
            "Bethlem myopathy (COL6A1/2/3) — proximal contractures + distal hyperlaxity; collagen VI IHC",
            "Congenital muscular dystrophy (LAMA2, COLQ, FKRP) — dystrophic biopsy; elevated CK; "
            "white matter changes (LAMA2); no cores on NADH-TR",
        ],
        "onset_range_y": (0.0, 20.0),
        "sex_female_prob": 0.50,
        "myopathy_type": "Central cores (NADH-TR PATHOGNOMONIC) / Type 1 fibre uniformity / MH susceptibility",
        "severity_dist": {"Severe": 0.30, "Moderate": 0.50, "Mild": 0.20},
        "ventilator_dependent_rate": 0.25,
        "mh_risk": True,
        "hepatopathy_rate": 0.0,
        "progression_rate": 0.45,
        "drug_error_rate": 0.20,
        "targeted_therapy_available": False,
        "first_line_drug": "TIVA (propofol + non-depolarising NMBD) — NEVER volatile agents or succinylcholine; dantrolene 2.5 mg/kg IV for MH emergency",
        "critical_avoid": "VOLATILE ANAESTHETIC AGENTS (halothane, sevoflurane, desflurane, isoflurane) + SUCCINYLCHOLINE ABSOLUTELY CONTRAINDICATED — MALIGNANT HYPERTHERMIA (MHS1 = highest risk locus); DANTROLENE 2.5mg/kg IV BOLUS emergency treatment; MH ALERT CARD to ALL patients; IVCT/CHCT for all first-degree relatives; AVOID succinylcholine universally in all congenital myopathies",
    },
    # ── ACTA1 — Nemaline Myopathy type 3 ────────────────────────────────────
    {
        "gene": "ACTA1", "protein": "Skeletal Muscle Alpha-Actin (ACTA1)",
        "alias": "ACTA1; OMIM gene 102610; 1q42.13; ~377 aa; Nemaline Myopathy type 3 (NEM3; OMIM #161800); AD de novo (75%) / AR; intranuclear rods PATHOGNOMONIC on EM; neonatal lethal form",
        "aa": "~377 aa", "kDa": "~42 kDa",
        "mechanism": (
            "ACTA1 encodes skeletal muscle alpha-actin, the principal thin filament protein of skeletal "
            "muscle sarcomeres. NORMAL FUNCTION: ACTA1 forms filamentous actin (F-actin) that polymerises "
            "with myosin cross-bridges during muscle contraction; regulated by troponin-tropomyosin complex. "
            "ACTA1-NEM3 PATHOMECHANISM: mutations produce multiple distinct histological lesions — "
            "(1) NEMALINE RODS (most common): mutant actin misfolds → aggregates into rods "
            "(Z-disc-derived protein accumulations); "
            "(2) INTRANUCLEAR RODS: rods form WITHIN myonuclei — characteristic of ACTA1; "
            "ALMOST PATHOGNOMONIC for ACTA1 myopathy on electron microscopy; "
            "(3) ACTIN FILAMENT ACCUMULATION: masses of thin filaments without sarcomeric organisation "
            "(filamentous actin clumps = 'actin myopathy'); "
            "(4) CONGENITAL FIBRE TYPE DISPROPORTION (CFTD): type 1 fibres smaller than type 2 by >12%. "
            "DOMINANT NEGATIVE (AD de novo): ~75% of severe ACTA1 cases are de novo dominant; "
            "mutant actin incorporated into thin filament → disrupts normal filament function → "
            "dominant negative effect; severity correlates with proportion of mutant actin incorporated. "
            "PHENOTYPIC SPECTRUM: genotype-phenotype is wide; same mutation may cause "
            "neonatal lethal → adult-onset mild myopathy."
        ),
        "disease_type": "Nemaline Myopathy type 3 (NEM3; ACTA1 de novo AD / AR; intranuclear rods PATHOGNOMONIC on EM; neonatal lethal dominant form; wide phenotypic spectrum)",
        "locus": "1q42.13", "omim_gene": 102610, "omim_disease": 161800,
        "inheritance": (
            "AUTOSOMAL DOMINANT DE NOVO: ~75% of SEVERE neonatal-lethal and infantile ACTA1-NEM3 cases "
            "arise from de novo dominant variants (parent testing negative — new mutation); "
            "dominant negative mechanism — one mutant allele disrupts filament function. "
            "AUTOSOMAL RECESSIVE: ~25% of ACTA1-NEM3; biallelic mutations; often milder phenotype "
            "(LOF rather than dominant negative); compound heterozygotes common. "
            "RECURRENCE RISK: de novo AD → recurrence risk in next sibling very low (<1%); "
            "if parent is mosaic → higher recurrence risk — parental testing recommended. "
            "AR: 25% recurrence risk for siblings; family cascade. "
            "GENETIC COUNSELLING: CRITICAL to distinguish de novo AD from AR — "
            "affects recurrence risk and family planning substantially."
        ),
        "phenotype": (
            "NEMALINE MYOPATHY TYPE 3 (NEM3) — WIDE SPECTRUM: "
            "NEONATAL LETHAL (75% de novo AD): born floppy (severe neonatal hypotonia); "
            "minimal spontaneous movement; respiratory failure requiring immediate mechanical ventilation; "
            "poor respiratory drive; facial diplegia; feeding impossible — NG tube from birth; "
            "ICU management from delivery; goals of care discussion MANDATORY; "
            "survival without intensive support is unlikely; "
            "SEVERE INFANTILE: requires ventilatory support; never ambulant; "
            "MODERATE: delayed milestones; ambulant with support; moderate respiratory involvement; "
            "MILD (AR forms): adult-onset; limb-girdle or distal pattern; normal lifespan. "
            "CARDIAC: typically NOT prominently involved (distinguishes from sarcomere diseases HCM/DCM). "
            "BIOPSY: nemaline rods (Gomori trichrome); INTRANUCLEAR RODS visible on EM — "
            "ALMOST PATHOGNOMONIC for ACTA1 (distinguish from NEB); "
            "actin accumulation bodies in some; CFTD pattern. "
            "KEY CLINICAL PEARL: ACTA1 has the widest phenotypic range of all nemaline myopathy genes — "
            "same mutation may be lethal in one patient and mild in another."
        ),
        "treatment_options": [
            "NEONATAL LETHAL FORM — Emergency respiratory support: "
            "intubation and mechanical ventilation at birth or immediately postnatally; "
            "GOALS OF CARE DISCUSSION MANDATORY early — prognosis for severe ACTA1 de novo cases is poor; "
            "involve palliative care, neonatology, and neurology; "
            "support family decision-making around ventilator dependence and tracheostomy",
            "Respiratory management (survivors): tracheostomy and home ventilator for ventilator-dependent; "
            "NIV (BiPAP) for those with partial respiratory reserve; "
            "annual spirometry + sleep oximetry for all patients; "
            "respiratory physiotherapy + cough assist devices",
            "Nutritional support: gastrostomy tube (PEG) for enteral nutrition in severe forms; "
            "speech-language pathology assessment of feeding and swallowing; "
            "caloric supplementation to prevent malnutrition and maintain body weight",
            "Physiotherapy: passive range of motion to prevent contractures; "
            "hydrotherapy for ambulant/partially ambulant patients; postural management",
            "Genetic counselling: distinguish de novo AD (low sibling recurrence) from AR (25% recurrence); "
            "parental mosaicism testing; prenatal diagnosis for subsequent pregnancies if mutations known",
            "NO DISEASE-MODIFYING THERAPY: supportive management only; "
            "gene therapy research in progress (gene replacement + exon skipping approaches); "
            "trial participation where available; Congenital Myopathy Research Registry enrolment",
        ],
        "key_ddx": [
            "NEB nemaline myopathy (NEM2) — AR; rods subsarcolemmal not intranuclear; "
            "exon 55 Ashkenazi founder; MLPA for NEB deletions; gene panel",
            "TPM2/TPM3 nemaline myopathy — AD milder; cap disease (TPM2); CFTD (TPM3); "
            "no intranuclear rods; genetic testing differentiates",
            "Myotubular myopathy (MTM1 XLMTM) — XLR males; centrally placed nuclei not rods; "
            "fetal myotubule appearance; hepatopathy 10%; X-linked inheritance",
            "Pontocerebellar hypoplasia type 1 (PCH1) — anterior horn + cerebellar; "
            "denervation atrophy not rods; EMG denervation; SMA overlap",
            "Centronuclear myopathy (DNM2, BIN1) — centrally placed nuclei; perinuclear halo on NADH-TR; "
            "ptosis + ophthalmoplegia; no rods; genetic panel",
        ],
        "onset_range_y": (0.0, 1.0),
        "sex_female_prob": 0.50,
        "myopathy_type": "Nemaline rods + Intranuclear rods (EM PATHOGNOMONIC) + Actin accumulation",
        "severity_dist": {"Severe": 0.40, "Moderate": 0.35, "Mild": 0.25},
        "ventilator_dependent_rate": 0.70,
        "mh_risk": False,
        "hepatopathy_rate": 0.0,
        "progression_rate": 0.60,
        "drug_error_rate": 0.10,
        "targeted_therapy_available": False,
        "first_line_drug": "Mechanical ventilation (neonatal lethal) / NIV (intermediate) / Physio (mild) — Goals of care MANDATORY in severe de novo",
        "critical_avoid": "NEONATAL LETHAL FORM: ICU from birth — respiratory failure; goals of care MANDATORY discussion; INTRANUCLEAR RODS on EM = ACTA1 almost pathognomonic (not NEB); de novo AD (~75% severe) vs AR (~25% milder) — CRITICAL genetic counselling distinction (different recurrence risk); AVOID succinylcholine (myopathic hyperkalaemia)",
    },
    # ── TPM2 — Nemaline Myopathy type 4 / Distal Arthrogryposis ─────────────
    {
        "gene": "TPM2", "protein": "Beta-Tropomyosin (TPM2)",
        "alias": "TPM2; OMIM gene 190990; 9p13.3; ~284 aa; Nemaline Myopathy type 4 (NEM4; OMIM #609285) + Distal Arthrogryposis type 1 (DA1); AD; trismus PATHOGNOMONIC; cap disease; milder prognosis",
        "aa": "~284 aa", "kDa": "~33 kDa",
        "mechanism": (
            "TPM2 encodes beta-tropomyosin (beta-TM), a coiled-coil protein that runs along the actin "
            "thin filament and regulates actin-myosin interaction. "
            "NORMAL FUNCTION: tropomyosin (dimers of alpha-TM/TPM3 and beta-TM/TPM2) wraps around the "
            "actin filament; at rest, tropomyosin blocks myosin binding sites; "
            "troponin-calcium-mediated movement of tropomyosin exposes binding sites → contraction. "
            "TPM2 PATHOMECHANISM: "
            "GOF (GAIN-OF-FUNCTION) variants — mutant beta-TM has increased actin affinity or "
            "altered tropomyosin position → myosin binding disrupted → "
            "thin filament over-activated (GOF) → paradoxically causes contractures + cap disease; "
            "LOF variants → nemaline rods (similar to NEB/ACTA1 mechanism). "
            "CAP DISEASE: characteristic TPM2 lesion — 'caps' of pale material (rods + myofibrillar debris) "
            "at the periphery of muscle fibres, visible as homogeneous caps on Gomori trichrome + NADH-TR. "
            "DISTAL ARTHROGRYPOSIS (DA1): GOF TPM2 variants → contractures predominantly at distal joints "
            "(fingers, wrists, ankles, toes) from fetal life; "
            "TRISMUS (restricted mouth opening from pterygoid/masseter involvement) "
            "is almost pathognomonic for TPM2 distal arthrogryposis."
        ),
        "disease_type": "Nemaline Myopathy type 4 (NEM4; AD TPM2; cap disease) + Distal Arthrogryposis type 1 (DA1; trismus PATHOGNOMONIC); generally milder phenotype",
        "locus": "9p13.3", "omim_gene": 190990, "omim_disease": 609285,
        "inheritance": (
            "AUTOSOMAL DOMINANT — heterozygous pathogenic variants. "
            "Most cases: GOF variants causing cap disease or distal arthrogryposis. "
            "De novo variants common in distal arthrogryposis; familial cases also reported. "
            "PENETRANCE: near-complete for contractures; variable expressivity for muscle weakness. "
            "GENETIC TESTING: thin filament gene panel (ACTA1, NEB, TPM2, TPM3, TNNT1, CFL2, KBTBD13); "
            "sequencing detects most TPM2 variants; large deletions rare. "
            "PROGNOSIS: TPM2 has the MILDEST prognosis among nemaline myopathy genes — "
            "most patients ambulant, no significant respiratory involvement typical."
        ),
        "phenotype": (
            "NEMALINE MYOPATHY TYPE 4 / DISTAL ARTHROGRYPOSIS TYPE 1: "
            "ONSET: Congenital (contractures at birth — DA1) or early childhood (weakness). "
            "DA1 FORM: distal joint contractures at birth: "
            "CAMPTODACTYLY (flexion contractures fingers), CLUBFOOT (talipes equinovarus), "
            "wrist contractures; TRISMUS — restricted mouth opening — ALMOST PATHOGNOMONIC for TPM2-DA1 "
            "(pterygoid muscle involvement causes jaw contracture); "
            "spine may be involved (scoliosis); knees may be hyperextended (genu recurvatum). "
            "MOTOR WEAKNESS: usually MILD proximal limb-girdle pattern; "
            "most patients ambulant throughout life; "
            "contractures may limit function more than weakness. "
            "RESPIRATORY: typically NOT significantly involved — normal or near-normal FVC; "
            "important differentiating feature from NEB and ACTA1 severe forms. "
            "CARDIAC: not involved. "
            "BIOPSY: cap disease (pale peripheral caps) OR nemaline rods; "
            "cap structures contain desmin and actin; type 1 fibre predominance. "
            "PROGNOSIS: BEST among nemaline myopathy genes — good ambulatory prognosis, "
            "near-normal lifespan; contractures are the primary management challenge."
        ),
        "treatment_options": [
            "Orthopaedic/physiotherapy management of contractures: "
            "serial casting from birth for talipes equinovarus (clubfoot); "
            "orthopaedic surgery for persistent/severe contractures (tendon lengthening, "
            "joint release); physiotherapy to maintain joint range of motion; "
            "AFOs for foot drop/ankle contractures; hand splinting for camptodactyly",
            "Trismus management: maxillofacial/dental assessment; "
            "soft diet and food modification for restricted mouth opening; "
            "physiotherapy for jaw contracture; "
            "dental extraction or procedures may require general anaesthesia "
            "with appropriate congenital myopathy precautions (avoid succinylcholine)",
            "Physiotherapy and exercise: regular physiotherapy to maintain strength; "
            "hydrotherapy well-tolerated; avoid prolonged immobilisation; "
            "swimming and aquatic therapy beneficial given mild weakness",
            "Scoliosis monitoring: spinal X-ray annually from childhood; "
            "bracing for curves 20-40 degrees; surgical consideration for progressive curves >40-50 degrees",
            "Respiratory monitoring: annual spirometry in all patients; "
            "sleep study if symptomatic; respiratory involvement uncommon but possible in severe alleles",
            "NO DISEASE-MODIFYING THERAPY: supportive; gene therapy in preclinical development; "
            "prognosis generally excellent with appropriate orthopaedic management",
        ],
        "key_ddx": [
            "Distal arthrogryposis type 2A (TNNI2, TNNT3) — similar DA phenotype; no trismus; "
            "biopsy different; genetically distinct; thin filament panel required",
            "Sheldon-Hall syndrome (TPM2, TNNI2, TNNT3, MYH3) — DA2B overlap; "
            "contractures + craniofacial; genotype differentiates",
            "Freeman-Sheldon syndrome (MYH3) — AD; severe craniofacial + distal contractures; "
            "whistling face; myopathic with MYH3 mutations",
            "NEB nemaline myopathy (NEM2) — AR; more severe; no trismus typically; "
            "rods not caps on biopsy; gene panel required",
            "TPM3 nemaline myopathy (NEM1) — AD/AR; type 1 fibre uniformity; CFTD; "
            "no trismus; head drop characteristic; genetic testing differentiates",
        ],
        "onset_range_y": (0.0, 5.0),
        "sex_female_prob": 0.50,
        "myopathy_type": "Cap disease (Gomori/NADH-TR) / Nemaline rods / Distal arthrogryposis (DA1) / Trismus PATHOGNOMONIC",
        "severity_dist": {"Severe": 0.10, "Moderate": 0.30, "Mild": 0.60},
        "ventilator_dependent_rate": 0.05,
        "mh_risk": False,
        "hepatopathy_rate": 0.0,
        "progression_rate": 0.30,
        "drug_error_rate": 0.05,
        "targeted_therapy_available": False,
        "first_line_drug": "Orthopaedic management (serial casting, tendon release) / Physiotherapy / Trismus management — excellent prognosis",
        "critical_avoid": "TRISMUS (restricted mouth opening) is ALMOST PATHOGNOMONIC for TPM2 distal arthrogryposis — check jaw opening in all DA1 cases; NO significant respiratory or cardiac involvement typical (distinguish from NEB/ACTA1/SELENON); AVOID succinylcholine (universal congenital myopathy precaution); mild prognosis — avoid over-medicalisation",
    },
    # ── TPM3 — Nemaline Myopathy type 1 ─────────────────────────────────────
    {
        "gene": "TPM3", "protein": "Alpha-Tropomyosin Slow (TPM3)",
        "alias": "TPM3; OMIM gene 191030; 1q21.3; ~284 aa; Nemaline Myopathy type 1 (NEM1; OMIM #609284); AD/AR; type 1 fibre uniformity + hypotrophy hallmark; CFTD; head drop; mild proximal weakness",
        "aa": "~284 aa", "kDa": "~33 kDa",
        "mechanism": (
            "TPM3 encodes alpha-tropomyosin slow (alpha-TM slow), the tropomyosin isoform expressed "
            "predominantly in slow (type 1) skeletal muscle fibres. "
            "NORMAL FUNCTION: TPM3 product forms dimers with TPM2 (beta-TM) along actin thin filaments "
            "in slow fibres; regulates calcium-dependent actomyosin interaction in type 1 fibres. "
            "PATHOMECHANISM: "
            "AD variants (dominant negative): mutant TPM3 incorporated into slow fibre thin filaments → "
            "thin filament dysfunction → type 1 fibre-specific myopathy → "
            "TYPE 1 FIBRE UNIFORMITY (all fibres become type 1 and smaller) → "
            "characteristic biopsy of CONGENITAL FIBRE TYPE DISPROPORTION (CFTD): "
            "type 1 fibres >12% smaller than type 2 fibres. "
            "AR variants: biallelic LOF → nemaline rods in some + CFTD. "
            "TYPE 1 FIBRE HYPOTROPHY: the hallmark of TPM3 myopathy; "
            "all fibres are type 1 (uniformity) and they are uniformly small (hypotrophy) — "
            "this unique biopsy pattern should trigger TPM3 testing. "
            "NEMALINE RODS: present in some TPM3 variants (mixed CFTD + rods phenotype)."
        ),
        "disease_type": "Nemaline Myopathy type 1 (NEM1; AD/AR TPM3; type 1 fibre uniformity/hypotrophy HALLMARK; CFTD; generally mild; head drop characteristic)",
        "locus": "1q21.3", "omim_gene": 191030, "omim_disease": 609284,
        "inheritance": (
            "AUTOSOMAL DOMINANT (most common) — heterozygous pathogenic variants; "
            "dominant negative mechanism affecting slow fibre tropomyosin. "
            "AUTOSOMAL RECESSIVE — biallelic variants; some with more severe/neonatal onset. "
            "AD variants often show variable expressivity within families. "
            "PENETRANCE: high for biopsy changes; variable for clinical weakness severity. "
            "GENETIC TESTING: thin filament gene panel; sequencing detects most variants; "
            "MLPA for potential large deletions. "
            "PROGNOSIS: TPM3 has GOOD prognosis — most patients remain ambulant; "
            "lifespan generally normal; respiratory involvement mild."
        ),
        "phenotype": (
            "NEMALINE MYOPATHY TYPE 1 / CFTD (TPM3): "
            "ONSET: Congenital to childhood; occasionally late childhood-adult. "
            "MOTOR PATTERN: "
            "PROXIMAL WEAKNESS — hip girdle and shoulder girdle; "
            "HEAD DROP / NECK FLEXOR WEAKNESS: characteristic feature of TPM3 myopathy — "
            "weakness of sternocleidomastoid and neck flexors → inability to lift head from bed; "
            "finger extension weakness in some; "
            "most patients AMBULANT throughout life; "
            "functional limitations predominantly proximal. "
            "FACIAL WEAKNESS: mild; ptosis uncommon (distinguishes from centronuclear). "
            "RESPIRATORY: usually MILD — FVC typically preserved; "
            "nocturnal hypoventilation in a minority; "
            "annual spirometry recommended. "
            "BIOPSY HALLMARK: "
            "TYPE 1 FIBRE UNIFORMITY (all fibres stain as type 1 on ATPase/myosin IHC) — "
            "HALLMARK of TPM3; "
            "TYPE 1 FIBRE HYPOTROPHY (type 1 fibres >12% smaller than type 2 — CFTD criterion); "
            "nemaline rods variably present; "
            "CFTD (congenital fibre type disproportion) is the dominant biopsy pattern. "
            "PROGNOSIS: GOOD — ambulation maintained; near-normal lifespan."
        ),
        "treatment_options": [
            "Physiotherapy: regular physiotherapy focusing on proximal muscle strengthening; "
            "hydrotherapy; postural management to address head drop; "
            "neck orthoses/collars for severe head drop if needed",
            "Orthotics: AFOs for distal foot involvement; "
            "neck support (soft collar or rigid depending on severity of head drop); "
            "functional hand orthoses if needed",
            "Respiratory monitoring: annual spirometry; sleep oximetry if symptomatic; "
            "NIV if FVC <50% or nocturnal hypoventilation (uncommon in TPM3 but possible); "
            "cough assist for any respiratory involvement",
            "Scoliosis monitoring: spinal radiograph annually; "
            "bracing or surgical management if progressive",
            "Nutritional assessment: dysphagia assessment if facial/bulbar weakness; "
            "nutritional support as needed; weight monitoring",
            "Exercise and activity: aerobic exercise beneficial and well-tolerated; "
            "swimming particularly recommended; avoid prolonged immobilisation; "
            "patient education on activity modification",
            "NO DISEASE-MODIFYING THERAPY: supportive management; "
            "prognosis is good; avoid unnecessary over-investigation; "
            "genetic counselling for family planning",
        ],
        "key_ddx": [
            "NEB nemaline myopathy (NEM2) — AR; more severe; rods prominent not CFTD pattern; "
            "no type 1 uniformity without rods; exon 55 Ashkenazi; NEB gene",
            "TPM2 nemaline myopathy (NEM4) — AD; cap disease; distal arthrogryposis; trismus; "
            "no CFTD pattern; TPM2 gene",
            "Congenital fibre type disproportion (other genes: ACTA1, RYR1, SELENON) — "
            "CFTD on biopsy but different genes; genetic panel required",
            "Congenital muscular dystrophy (LAMA2, COLQ) — dystrophic biopsy; elevated CK; "
            "no type 1 uniformity pattern; different genes",
            "MTM1 myotubular myopathy (XLMTM) — centrally placed nuclei not type 1 hypotrophy; "
            "X-linked males; hepatopathy; much more severe neonatal",
        ],
        "onset_range_y": (0.0, 10.0),
        "sex_female_prob": 0.55,
        "myopathy_type": "Type 1 fibre uniformity + hypotrophy (CFTD HALLMARK) / Nemaline rods (variable) / Head drop",
        "severity_dist": {"Severe": 0.10, "Moderate": 0.35, "Mild": 0.55},
        "ventilator_dependent_rate": 0.08,
        "mh_risk": False,
        "hepatopathy_rate": 0.0,
        "progression_rate": 0.35,
        "drug_error_rate": 0.05,
        "targeted_therapy_available": False,
        "first_line_drug": "Physiotherapy + Neck orthosis (head drop) / NIV if respiratory — generally mild, good prognosis",
        "critical_avoid": "HEAD DROP / NECK FLEXOR WEAKNESS is CHARACTERISTIC of TPM3 — assess neck flexors at each visit; TYPE 1 FIBRE UNIFORMITY on biopsy = TPM3 until proven otherwise; CFTD pattern should trigger TPM3 genetic testing; generally MILD — avoid overly aggressive ventilatory intervention early; AVOID succinylcholine (universal congenital myopathy precaution)",
    },
    # ── MTM1 — X-linked Myotubular Myopathy ──────────────────────────────────
    {
        "gene": "MTM1", "protein": "Myotubularin 1 (MTM1)",
        "alias": "MTM1; OMIM gene 300415; Xq28; ~603 aa; X-linked Myotubular Myopathy (XLMTM; OMIM #310400); XLR; neonatal onset profound hypotonia + respiratory failure; centrally placed nuclei PATHOGNOMONIC; hepatopathy 10%",
        "aa": "~603 aa", "kDa": "~69 kDa",
        "mechanism": (
            "MTM1 encodes myotubularin, a phosphoinositide phosphatase that dephosphorylates "
            "phosphatidylinositol 3-phosphate (PI3P) and PI(3,5)P2. "
            "NORMAL FUNCTION: MTM1 regulates endosomal sorting, membrane tubulation, "
            "and autophagy via PI3P/PI(3,5)P2 homeostasis; critical for muscle fibre maturation "
            "from myotubes to mature myofibres during development. "
            "XLMTM PATHOMECHANISM: hemizygous loss of MTM1 → PI3P accumulates → "
            "arrest of muscle fibre maturation at myotube stage → "
            "fibres morphologically resemble foetal myotubes (centrally placed nuclei, "
            "perinuclear halo of organelles). "
            "HISTOLOGY: CENTRALLY PLACED NUCLEI in the majority of type 1 fibres — "
            "PATHOGNOMONIC of MTM1 myopathy; "
            "necklace fibres (in DNM2 centronuclear myopathy — distinguish). "
            "HEPATOPATHY: MTM1 is expressed in hepatocytes; ~10% of XLMTM patients develop "
            "hepatic involvement (elevated transaminases, hepatic fibrosis); "
            "screen LIVER ENZYMES in ALL XLMTM patients. "
            "DNM2-MTM1 INTERACTION: Dynamin 2 (DNM2) is a modifier — overactive DNM2 in the "
            "absence of MTM1 worsens the phenotype; DNM2 reduction is a validated therapeutic target "
            "(antisense oligonucleotide to DNM2 ameliorates XLMTM in mouse models — in clinical trials)."
        ),
        "disease_type": "X-linked Myotubular Myopathy (XLMTM; XLR MTM1; neonatal ventilator dependence; centrally placed nuclei PATHOGNOMONIC; hepatopathy 10%; DNM2 therapeutic target)",
        "locus": "Xq28", "omim_gene": 300415, "omim_disease": 310400,
        "inheritance": (
            "X-LINKED RECESSIVE — hemizygous males SEVERELY affected. "
            "FEMALES: heterozygous females are usually UNAFFECTED CARRIERS; "
            "however, SOME CARRIER FEMALES develop MILD MYOPATHY due to X-inactivation skewing — "
            "assess female carriers clinically (spirometry, strength). "
            "CARRIER DETECTION: maternal carrier testing by MTM1 sequencing + MLPA; "
            "~25% of cases are de novo (new mutations in the proband, mother not carrier). "
            "FAMILY SCREENING: maternal brothers at 50% risk; maternal sisters at 50% carrier risk. "
            "PRENATAL/PREIMPLANTATION: available once mutation identified. "
            "ASPLENISM: functional asplenia reported in some XLMTM patients — check splenic function. "
            "PROGNOSIS: survival largely dependent on ventilatory support — "
            "without aggressive respiratory support, neonatal death within days-weeks."
        ),
        "phenotype": (
            "X-LINKED MYOTUBULAR MYOPATHY (XLMTM): "
            "ONSET: NEONATAL — profound hypotonia at birth (floppy infant); "
            "respiratory failure requiring MECHANICAL VENTILATION at birth or within hours; "
            "almost universally require ventilation at birth. "
            "MOTOR: no spontaneous antigravity movements; hyporeflexia; ophthalmoplegia (ptosis + EOM paresis); "
            "facial diplegia; no independent sitting or standing without support; "
            "some patients achieve minimal voluntary movement with intensive support. "
            "RESPIRATORY: VENTILATOR DEPENDENT from birth; "
            "spontaneous breathing may develop in some (milder mutations); "
            "tracheostomy for long-term ventilation in survivors; "
            "without ventilator support, survival is measured in weeks. "
            "FEEDING: severe feeding difficulties; gastrostomy tube required. "
            "HEPATOPATHY: ~10% — elevated ALT/AST, hepatic fibrosis; "
            "SCREEN LIVER ENZYMES AT DIAGNOSIS and periodically. "
            "BIOPSY: CENTRALLY PLACED NUCLEI in type 1 fibres — PATHOGNOMONIC; "
            "'necklace fibres' not present (distinguish from DNM2 centronuclear). "
            "EM: nuclear centricity with surrounding halo of mitochondria/glycogen/ER. "
            "ASPLENISM in subset — Howell-Jolly bodies on blood film; "
            "FEMALE CARRIERS: check CK, spirometry, strength — mild involvement possible."
        ),
        "treatment_options": [
            "Mechanical ventilation from birth: "
            "neonatal intubation and ventilation IMMEDIATELY at birth or within hours; "
            "transition to home tracheostomy + ventilator for surviving patients; "
            "ventilator weaning rarely successful in severe XLMTM; "
            "nocturnal NIV occasionally sufficient in milder cases (mild mutations); "
            "respiratory goals of care discussion early — family support critical",
            "Hepatic monitoring: "
            "LIVER ENZYMES (ALT, AST, GGT, bilirubin) at diagnosis and every 6 months; "
            "HEPATOPATHY in ~10% of XLMTM — hepatic fibrosis, peliosis hepatis; "
            "liver ultrasound in patients with elevated enzymes; "
            "avoid hepatotoxic medications; consider hepatology referral if elevated",
            "Nutritional support: gastrostomy tube (PEG) essential for adequate nutrition; "
            "dietitian involvement from infancy; prevent malnutrition to support respiratory function",
            "Physiotherapy and rehabilitation: passive ROM to prevent contractures; "
            "standing frames and powered wheelchairs for mobility; "
            "communication aids (AAC) if ventilated",
            "DNM2-ASO (XLMTM-101): antisense oligonucleotide reducing DNM2 expression — "
            "Phase 2 clinical trial (ASPIRO); showed functional improvements (ventilator-free time); "
            "ASP1000 (formerly AT132) gene therapy (AAV8-MTM1): showed early promise but "
            "associated with fatal liver toxicity in some patients — trial paused; ongoing investigation; "
            "trial participation at specialist XLMTM centres where available",
            "Asplenism management: "
            "check for functional asplenia (Howell-Jolly bodies); "
            "if asplenic: pneumococcal, meningococcal, Hib vaccines; "
            "antibiotic prophylaxis per asplenia protocol; "
            "patient/carer education on fever management in asplenia",
        ],
        "key_ddx": [
            "DNM2 centronuclear myopathy (AD CNM1) — centrally placed nuclei BUT milder; "
            "NECKLACE FIBRES on oxidative stain; CMT2M allelic; AD not X-linked; "
            "onset childhood-adulthood; genetic testing differentiates",
            "BIN1 centronuclear myopathy (CNM2) — AR; centrally placed nuclei; "
            "no necklace fibres; T-tubule abnormality; BIN1 gene",
            "Nemaline myopathy (NEB, ACTA1) — nemaline rods not centrally placed nuclei; "
            "biopsy completely different; gene panel required",
            "Congenital muscular dystrophy (LAMA2) — dystrophic biopsy not centrally placed nuclei; "
            "elevated CK; white matter changes on MRI brain (LAMA2 MDC1A)",
            "SMA type 1 (SMN1) — anterior horn cell disease; EMG denervation; "
            "no centrally placed nuclei on biopsy; SMN1 deletion on genetic test",
        ],
        "onset_range_y": (0.0, 0.5),
        "sex_female_prob": 0.05,
        "myopathy_type": "Centrally placed nuclei (PATHOGNOMONIC) / Perinuclear halo / Fetal myotube pattern",
        "severity_dist": {"Severe": 0.90, "Moderate": 0.10, "Mild": 0.00},
        "ventilator_dependent_rate": 0.90,
        "mh_risk": False,
        "hepatopathy_rate": 0.10,
        "progression_rate": 0.50,
        "drug_error_rate": 0.12,
        "targeted_therapy_available": True,
        "first_line_drug": "Mechanical ventilation from birth (tracheostomy) / DNM2-ASO trial / Liver enzyme monitoring MANDATORY",
        "critical_avoid": "HEPATOPATHY in ~10% of XLMTM — screen LIVER ENZYMES ALL patients at diagnosis and every 6 months; FEMALE CARRIERS: ~some develop mild myopathy (X-inactivation skewing) — assess clinically; ASPLENISM — check Howell-Jolly bodies; vaccinate and antibiotics if asplenic; DNM2-ASO gene therapy aspirations are active — refer to trial centre; NEONATAL VENTILATOR DEPENDENCE = MTM1 until proven otherwise in severely affected male neonates",
    },
    # ── DNM2 — Centronuclear Myopathy type 1 ─────────────────────────────────
    {
        "gene": "DNM2", "protein": "Dynamin 2 (DNM2)",
        "alias": "DNM2; OMIM gene 602378; 19p13.2; ~870 aa; Centronuclear Myopathy type 1 (CNM1; OMIM #160150); AD; milder than XLMTM; centrally placed nuclei + perinuclear halo PATHOGNOMONIC; CMT2M allelic",
        "aa": "~870 aa", "kDa": "~96 kDa",
        "mechanism": (
            "DNM2 encodes dynamin 2, a large GTPase involved in membrane fission and tubulation. "
            "NORMAL FUNCTION: DNM2 pinches off vesicles from membranes (endocytosis, clathrin-mediated); "
            "regulates T-tubule biogenesis in skeletal muscle (T-tubules = invaginations of sarcolemma "
            "critical for EC coupling and calcium distribution); "
            "membrane trafficking (Golgi, endosomes). "
            "DNM2-CNM1 PATHOMECHANISM: "
            "AD GOF mutations → hyperactive DNM2 → excessive membrane fission → "
            "T-tubule abnormalities (irregular, dilated, disorganised) → impaired EC coupling; "
            "concurrent misregulation of phosphoinositide metabolism (DNM2 interacts with MTM1 pathway). "
            "CENTRALLY PLACED NUCLEI: DNM2 variants → arrest of myotube maturation → "
            "nuclei fail to migrate to periphery → remain centrally placed; "
            "PERINUCLEAR HALO: accumulation of mitochondria/glycogen around central nuclei "
            "= characteristic 'perinuclear halo' on NADH-TR stain. "
            "NECKLACE FIBRES: pale 'necklace' surrounding a central core on oxidative stain — "
            "CHARACTERISTIC of DNM2 centronuclear myopathy (distinguishes from MTM1). "
            "CMT2M ALLELIC: DNM2 also causes Charcot-Marie-Tooth disease type 2M (axonal CMT); "
            "CHECK FOR NEUROPATHY in DNM2-CNM patients and their families."
        ),
        "disease_type": "Centronuclear Myopathy type 1 (CNM1; AD DNM2; centrally placed nuclei + necklace fibres PATHOGNOMONIC; ptosis + ophthalmoplegia; CMT2M allelic; milder than XLMTM)",
        "locus": "19p13.2", "omim_gene": 602378, "omim_disease": 160150,
        "inheritance": (
            "AUTOSOMAL DOMINANT — heterozygous pathogenic variants; "
            "most common missense at p.Arg465Trp (~40% of reported DNM2-CNM cases). "
            "DE NOVO variants occur; familial cases with variable expressivity also described. "
            "PENETRANCE: high for centronuclear biopsy findings; "
            "variable expressivity for clinical severity — same family can have mildly and moderately affected. "
            "ALLELIC DISORDERS: same gene, different variants cause: "
            "(1) Centronuclear Myopathy (CNM1) — dominant; "
            "(2) Charcot-Marie-Tooth disease type 2M (CMT2M) — axonal neuropathy, dominant; "
            "(3) Lethal-infantile centronuclear myopathy (de novo severe alleles — rare). "
            "FAMILY SCREENING: NCS/EMG for peripheral neuropathy in CNM patients AND families "
            "(CMT2M allele in same family possible)."
        ),
        "phenotype": (
            "CENTRONUCLEAR MYOPATHY TYPE 1 (DNM2-CNM1): "
            "ONSET: Childhood to early adulthood (later onset than MTM1 — distinguishing feature). "
            "MOTOR: slowly progressive proximal limb weakness; "
            "most patients AMBULANT throughout life; "
            "foot drop in some; progressive but slow decline. "
            "FACIAL: PTOSIS — unilateral or bilateral; OPHTHALMOPLEGIA (EOM paresis) — "
            "both common; distinguish from mitochondrial myopathy "
            "(normal lactate, no ragged red fibres in DNM2-CNM1). "
            "RESPIRATORY: mild to moderate; FVC monitoring recommended; "
            "NIV in advanced cases; less severe than XLMTM. "
            "CK: moderately elevated (distinguishes from MTM1 — also mild-moderate). "
            "BIOPSY HALLMARK: "
            "CENTRALLY PLACED NUCLEI + PERINUCLEAR HALO on NADH-TR — PATHOGNOMONIC of centronuclear; "
            "NECKLACE FIBRES (pale ring surrounding type 1 fibre core on oxidative stain) — "
            "CHARACTERISTIC of DNM2 specifically (NOT seen in MTM1); "
            "type 1 fibre predominance. "
            "CMT2M NEUROPATHY: check NCS/EMG — axonal neuropathy may be present; "
            "foot deformity (pes cavus) from neuropathy; "
            "family members may have neuropathy without myopathy."
        ),
        "treatment_options": [
            "Physiotherapy and activity management: regular physiotherapy; "
            "ambulatory aids (cane, walker) as weakness progresses; "
            "powered mobility when required; avoid prolonged immobilisation; "
            "aquatic therapy beneficial for proximal weakness",
            "Ptosis management: ptosis props (crutch spectacles) for functional ptosis; "
            "ophthalmic assessment; ptosis surgery when significant and functional; "
            "ophthalmoplegia monitoring (distinguish from mitochondrial myopathy — "
            "check lactate and muscle biopsy in diagnostic uncertainty)",
            "Respiratory monitoring: annual spirometry; sleep oximetry if symptomatic; "
            "NIV if FVC <50% or nocturnal hypoventilation; "
            "respiratory involvement present in moderate/severe cases",
            "Peripheral neuropathy assessment: "
            "NCS/EMG at diagnosis AND for first-degree relatives; "
            "CMT2M axonal neuropathy may coexist (same gene, different mutation class); "
            "foot orthotics for pes cavus from neuropathy component; "
            "neurologist + neuromuscular specialist co-management",
            "Scoliosis monitoring: spinal radiograph annually; "
            "orthopaedic consultation for progressive curves; "
            "spinal fusion if required",
            "DNM2-targeted therapy (investigational): "
            "DNM2 reduction strategy (antisense oligonucleotide to DNM2) is the primary "
            "therapeutic approach for XLMTM (MTM1) — DNM2 reduction ALSO being explored for DNM2-CNM; "
            "clinical trials ongoing; enrolment at specialist centres",
        ],
        "key_ddx": [
            "MTM1 X-linked myotubular myopathy (XLMTM) — MUCH more severe; neonatal onset; "
            "centrally placed nuclei BUT NO NECKLACE FIBRES; XLR males; hepatopathy 10%; "
            "different gene — genetic testing essential",
            "BIN1 centronuclear myopathy (CNM2) — AR; centrally placed nuclei; "
            "T-tubule abnormality; NO necklace fibres; BIN1 ampliphysin gene; "
            "early childhood onset; genetic panel required",
            "Mitochondrial myopathy (CPEO, MELAS) — ptosis + ophthalmoplegia overlap; "
            "RAGGED RED FIBRES on Gomori trichrome (not centrally placed nuclei); "
            "elevated lactate; mtDNA/nuclear MT gene mutations",
            "Myotonic dystrophy type 1 (DMPK CTG repeat) — myotonia; cataracts; cardiac; "
            "different clinical and biopsy pattern; ring fibres; myotonic discharges on EMG",
            "Oculopharyngeal muscular dystrophy (PABPN1) — late adult onset; "
            "filamentous intranuclear inclusions on EM (not centrally placed nuclei); "
            "dysphagia prominent; PABPN1 gene",
        ],
        "onset_range_y": (5.0, 30.0),
        "sex_female_prob": 0.50,
        "myopathy_type": "Centrally placed nuclei + Necklace fibres (CHARACTERISTIC DNM2) / Ptosis + Ophthalmoplegia / CMT2M allelic",
        "severity_dist": {"Severe": 0.15, "Moderate": 0.35, "Mild": 0.50},
        "ventilator_dependent_rate": 0.15,
        "mh_risk": False,
        "hepatopathy_rate": 0.0,
        "progression_rate": 0.55,
        "drug_error_rate": 0.08,
        "targeted_therapy_available": True,
        "first_line_drug": "Physiotherapy + Ptosis management + Respiratory monitoring / NCS/EMG for CMT2M neuropathy",
        "critical_avoid": "CMT2M ALLELIC DISEASE — DNM2 variants cause BOTH centronuclear myopathy AND CMT2M axonal neuropathy — NCS/EMG MANDATORY for ALL DNM2-CNM patients AND first-degree relatives; NECKLACE FIBRES on oxidative stain = DNM2 CNM (distinguishes from MTM1 centrally placed nuclei); PTOSIS + OPHTHALMOPLEGIA — distinguish from mitochondrial (lactate, RRF on biopsy); AVOID succinylcholine",
    },
    # ── SELENON — Rigid Spine Muscular Dystrophy 1 ────────────────────────────
    {
        "gene": "SELENON", "protein": "Selenoprotein N (SELENON / SEPN1)",
        "alias": "SELENON (formerly SEPN1); OMIM gene 606210; 1p36.11; ~590 aa; Rigid Spine Muscular Dystrophy 1 (RSMD1; OMIM #602771); AR; rigid spine + contractures BEFORE limb weakness; respiratory failure precedes limb weakness; NIV MANDATORY early",
        "aa": "~590 aa", "kDa": "~65 kDa",
        "mechanism": (
            "SELENON (formerly SEPN1) encodes selenoprotein N, an endoplasmic reticulum (ER) resident "
            "glycoprotein containing a selenocysteine residue (the 21st amino acid, encoded by UGA codon "
            "decoded as Sec via SECIS element). "
            "NORMAL FUNCTION: selenoprotein N participates in ER calcium homeostasis and "
            "oxidative stress defence (redox function via selenocysteine active site); "
            "regulates ryanodine receptor (RYR1/RYR2) activity in ER membrane; "
            "important for muscle fibre development and maintenance. "
            "RSMD1 PATHOMECHANISM: biallelic loss-of-function → loss of ER redox protection → "
            "oxidative damage to muscle fibres; disrupted ER calcium regulation → "
            "aberrant RYR activity → muscle fibre injury. "
            "BIOPSY: MULTICORE LESIONS (mini-cores) — multiple small areas devoid of mitochondria "
            "on NADH-TR (mini-cores = SELENON; distinguish from central cores = RYR1); "
            "Mallory-like bodies (desmin aggregates) in some SELENON patients. "
            "RIGID SPINE: paraspinal muscle weakness + contractures → "
            "rigid spine early in the disease course, BEFORE significant limb weakness — "
            "KEY CLINICAL PEARL; "
            "spinal rigidity → restrictive thoracic cage → restrictive lung disease → "
            "RESPIRATORY FAILURE despite apparently preserved limb function. "
            "SELENIUM SUPPLEMENTATION: despite gene name, no consistent benefit — selenium "
            "supplementation does not correct SELENON LOF."
        ),
        "disease_type": "Rigid Spine Muscular Dystrophy 1 (RSMD1; AR SELENON LOF; rigid spine + contractures BEFORE limb weakness; RESPIRATORY FAILURE precedes limb weakness — NIV MANDATORY early)",
        "locus": "1p36.11", "omim_gene": 606210, "omim_disease": 602771,
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic pathogenic variants. "
            "Prevalence: uncommon; seen across ethnic groups. "
            "Most patients compound heterozygotes; homozygotes in consanguineous families. "
            "GENETIC TESTING: SELENON sequencing + MLPA for deletions; "
            "part of standard congenital myopathy gene panel. "
            "FAMILY SCREENING: parents obligate carriers; siblings 25% recurrence risk. "
            "PRENATAL DIAGNOSIS: available once mutations identified. "
            "GENETIC COUNSELLING: no parent typically affected (AR); "
            "selenium supplementation based on gene name is INCORRECT and NOT beneficial — "
            "SELENON LOF is not selenium deficiency."
        ),
        "phenotype": (
            "RIGID SPINE MUSCULAR DYSTROPHY 1 (RSMD1): "
            "ONSET: Infancy to early childhood (typically 1-15 years). "
            "HALLMARK — RIGID SPINE SYNDROME: "
            "early and prominent SPINAL RIGIDITY — limited forward flexion of cervical and thoracic spine; "
            "CONTRACTURES of spine develop BEFORE significant limb weakness — key diagnostic clue; "
            "scoliosis common. "
            "LIMB WEAKNESS: proximal > distal; milder than spinal involvement; "
            "most patients ambulant but may develop difficulty with stairs/running. "
            "RESPIRATORY — CRITICAL: "
            "RESPIRATORY FAILURE PRECEDES SIGNIFICANT LIMB WEAKNESS — "
            "the rigid spine + scoliosis reduces chest wall compliance → restrictive lung disease → "
            "DYSPNEA MAY BE MASKED because limited mobility reduces oxygen demand; "
            "patient and family may not report respiratory symptoms despite significant restriction; "
            "ANNUAL SPIROMETRY + SLEEP STUDY MANDATORY regardless of perceived clinical state; "
            "nocturnal hypoventilation precedes daytime respiratory failure — "
            "sleep study detects this before symptoms; "
            "EARLY NIV is critical — start before symptomatic respiratory failure. "
            "BIOPSY: multicore lesions (mini-cores on NADH-TR) — distinguish from central cores (RYR1); "
            "Mallory-like bodies in some. "
            "SELENIUM: NO benefit from selenium supplementation despite gene name."
        ),
        "treatment_options": [
            "Respiratory management — MANDATORY: "
            "ANNUAL SPIROMETRY (FVC — forced vital capacity) and OVERNIGHT SLEEP STUDY "
            "(oximetry + transcutaneous CO2) from diagnosis in ALL RSMD1 patients — "
            "regardless of perceived respiratory symptoms; "
            "START NIV (BiPAP) at: FVC <60% predicted, OR sleep study showing nocturnal desaturation, "
            "OR transcutaneous CO2 >6 kPa overnight — do NOT wait for daytime symptoms; "
            "RIGID SPINE MASKS DYSPNEA — patients underreport because limited mobility reduces demand; "
            "tracheostomy for NIV failure or severe bulbar dysfunction",
            "Scoliosis management: "
            "spinal radiograph (AP + lateral) every 6-12 months; "
            "rigid spinal bracing for curves 20-40 degrees (may worsen respiratory function — "
            "coordinate with respiratory team); "
            "spinal fusion surgery for progressive Cobb angle >40-50 degrees; "
            "surgery timing: AFTER lung function established on NIV — "
            "co-ordinate timing with pulmonologist; "
            "rigid spine limits surgical correction — specialist spinal orthopaedic team required",
            "Physiotherapy: range of motion exercises for spine and limbs; "
            "hydrotherapy; swimming beneficial; avoid prolonged immobilisation; "
            "postural management",
            "Selenium supplementation: AVOID — no consistent benefit despite gene name; "
            "SELENON LOF is loss of protein function, NOT selenium deficiency; "
            "selenium toxicity risk if supplemented excessively",
            "Nutritional support: dysphagia assessment (bulbar involvement in some); "
            "nutritional monitoring; PEG if required; "
            "avoid obesity which worsens restrictive lung disease",
            "NO DISEASE-MODIFYING THERAPY: supportive management; "
            "trial participation where available; "
            "RYR1 and SELENON share ER calcium pathways — shared therapeutic approaches in research",
        ],
        "key_ddx": [
            "RYR1 multi-minicore disease (AR) — also has minicore lesions on biopsy; "
            "but CENTRAL CORES (not mini-cores); MH risk (RYR1); "
            "ophthalmoplegia in RYR1 AR; genetic panel differentiates",
            "LMNA muscular dystrophy / EDMD — rigid spine can occur; "
            "but LETHAL arrhythmia / cardiac conduction; "
            "LMNA gene; cardiac involvement mandatory monitoring",
            "Emery-Dreifuss muscular dystrophy (EMD) — rigid spine + contractures; "
            "but XLR emerin gene; cardiac arrhythmia PATHOGNOMONIC (atrial standstill); "
            "genetic testing differentiates",
            "Nemaline myopathy (NEB, ACTA1) — minicore lesions NOT nemaline rods; "
            "Gomori trichrome negative for rods in SELENON; genetic panel required",
            "Bethlem myopathy (COL6A1) — contractures + proximal weakness; "
            "but distal HYPERLAXITY + proximal contractures PATHOGNOMONIC for Bethlem; "
            "different biopsy; COL6A gene panel",
        ],
        "onset_range_y": (1.0, 15.0),
        "sex_female_prob": 0.50,
        "myopathy_type": "Multicore lesions (mini-cores on NADH-TR) / Rigid spine syndrome / Respiratory failure BEFORE limb weakness",
        "severity_dist": {"Severe": 0.30, "Moderate": 0.50, "Mild": 0.20},
        "ventilator_dependent_rate": 0.60,
        "mh_risk": False,
        "hepatopathy_rate": 0.0,
        "progression_rate": 0.65,
        "drug_error_rate": 0.18,
        "targeted_therapy_available": False,
        "first_line_drug": "NIV MANDATORY before symptomatic respiratory failure — annual spirometry + sleep study; spinal fusion for scoliosis",
        "critical_avoid": "RESPIRATORY FAILURE PRECEDES LIMB WEAKNESS — RIGID SPINE MASKS DYSPNEA; MANDATORY annual spirometry + sleep study regardless of perceived symptoms; START NIV at FVC <60% or nocturnal hypoventilation — DO NOT WAIT for daytime symptoms; SELENIUM SUPPLEMENTATION: NO benefit despite gene name — do not prescribe; MINI-CORES on NADH-TR (not central cores — distinguish from RYR1); AVOID succinylcholine",
    },
]


# ── Patient generator ───────────────────────────────────────────────────────

def _gen_patients(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    patients = []
    onset_lo, onset_hi = gene_data["onset_range_y"]

    for i in range(40):
        onset = round(rng.uniform(onset_lo, max(onset_lo + 0.1, onset_hi)), 2)

        # Severity
        r = rng.random()
        cumulative = 0.0
        sev = "Severe"
        for label, prob in gene_data["severity_dist"].items():
            cumulative += prob
            if r < cumulative:
                sev = label
                break

        # Sex (MTM1 is X-linked recessive — predominantly male)
        sex = "F" if rng.random() < gene_data["sex_female_prob"] else "M"

        # Clinical booleans
        vent_dep = rng.random() < gene_data["ventilator_dependent_rate"]
        mh_risk  = gene_data["mh_risk"]
        hepatop  = rng.random() < gene_data["hepatopathy_rate"]
        drug_err = rng.random() < gene_data["drug_error_rate"]
        on_tgt   = gene_data["targeted_therapy_available"] and rng.random() < 0.40
        progress = rng.random() < gene_data["progression_rate"]
        cog_imp  = sev == "Severe" and rng.random() < (
            0.20 if gene in ("NEB", "ACTA1", "MTM1") else
            0.10 if gene in ("RYR1", "DNM2") else 0.05
        )

        # Myopathy type (primary descriptor)
        mv = gene_data["myopathy_type"].split(" / ")[0]

        # Treatment
        fl = gene_data["first_line_drug"].split(" / ")[0]
        if on_tgt:
            tx = fl + " (targeted/disease-specific)"
        elif drug_err:
            tx = "CONTRAINDICATED or incorrect drug prescribed (error detected)"
        else:
            tx = fl + (" + adjunctive therapy" if rng.random() < 0.50 else "")

        age_at_dx = round(min(onset + rng.uniform(0.2, 3.0), max(onset_hi, 1.0) + 5.0), 2)

        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "onset_age_y": onset,
            "diagnosis_age_y": age_at_dx,
            "sex": sex,
            "severity": sev,
            "myopathy_type": mv,
            "ventilator_dependent": vent_dep,
            "mh_risk": mh_risk,
            "hepatopathy": hepatop,
            "drug_avoid_prescribed_error": drug_err,
            "on_targeted_therapy": on_tgt,
            "disease_progression": progress,
            "cognitive_impairment": cog_imp,
            "treatment": tx,
            "first_line_drug": gene_data["first_line_drug"],
            "critical_avoid": gene_data["critical_avoid"],
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gene_data in enumerate(CM_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients(gene_data, seed))
    return all_pts


# ── Public API ──────────────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] += 1

    vent_n    = sum(1 for p in patients if p["ventilator_dependent"])
    mh_n      = sum(1 for p in patients if p["mh_risk"])
    hepat_n   = sum(1 for p in patients if p["hepatopathy"])
    prog_n    = sum(1 for p in patients if p["disease_progression"])
    targeted_n = sum(1 for p in patients if p["on_targeted_therapy"])
    drug_err_n = sum(1 for p in patients if p["drug_avoid_prescribed_error"])

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 2)
    mean_dx_age = round(sum(p["diagnosis_age_y"] for p in patients) / n, 2)

    return {
        "atlas": "Congenital-Myopathy-Atlas",
        "full_name": "Complete 8-Gene Congenital Myopathy Atlas",
        "subtitle": (
            "NEB·RYR1·ACTA1·TPM2·TPM3·MTM1·DNM2·SELENON — "
            "320 patients (8×40, seeds 1086–1093)"
        ),
        "description": (
            "Comprehensive atlas of 8 major genetic congenital myopathies encompassing: "
            "NEMALINE MYOPATHY 2 (NEB — AR; ~8045 aa; largest sarcomeric gene ~183 exons; "
            "nemaline rods Gomori trichrome PATHOGNOMONIC; exon 55 Ashkenazi founder deletion; "
            "respiratory failure before limb weakness; NIV cornerstone; no DMT); "
            "CENTRAL CORE DISEASE (RYR1 — AD/AR; SR calcium release channel; "
            "central cores NADH-TR PATHOGNOMONIC; MHS1 MALIGNANT HYPERTHERMIA — "
            "VOLATILE AGENTS + SUCCINYLCHOLINE ABSOLUTELY CONTRAINDICATED; dantrolene emergency); "
            "NEMALINE MYOPATHY 3 (ACTA1 — AD de novo ~75% / AR; skeletal alpha-actin; "
            "INTRANUCLEAR RODS on EM PATHOGNOMONIC; neonatal lethal dominant form; wide spectrum; "
            "goals of care MANDATORY in severe); "
            "NEMALINE MYOPATHY 4 / DISTAL ARTHROGRYPOSIS (TPM2 — AD; beta-tropomyosin; "
            "TRISMUS almost PATHOGNOMONIC for DA1; cap disease; mild prognosis); "
            "NEMALINE MYOPATHY 1 (TPM3 — AD/AR; alpha-tropomyosin slow; "
            "TYPE 1 FIBRE UNIFORMITY/HYPOTROPHY HALLMARK; CFTD; HEAD DROP characteristic; mild); "
            "X-LINKED MYOTUBULAR MYOPATHY (MTM1 — XLR; myotubularin phosphatase; "
            "neonatal ventilator dependence; CENTRALLY PLACED NUCLEI PATHOGNOMONIC; "
            "HEPATOPATHY ~10% — screen liver enzymes ALL patients; female carriers check spirometry); "
            "CENTRONUCLEAR MYOPATHY 1 (DNM2 — AD; dynamin 2 GTPase; "
            "centrally placed nuclei + NECKLACE FIBRES characteristic; ptosis + ophthalmoplegia; "
            "CMT2M ALLELIC — NCS/EMG MANDATORY); "
            "RIGID SPINE MUSCULAR DYSTROPHY 1 (SELENON — AR; ER redox/calcium; "
            "RIGID SPINE + CONTRACTURES BEFORE LIMB WEAKNESS; "
            "RESPIRATORY FAILURE PRECEDES LIMB WEAKNESS — NIV MANDATORY EARLY; "
            "selenium supplementation NO benefit despite gene name)."
        ),
        "total_patients": n,
        "genes_covered": len(CM_GENES),
        "patients_per_gene": 40,
        "seed_range": "1086–1093",
        "gene_list": [g["gene"] for g in CM_GENES],
        "disease_category_breakdown": {
            "Nemaline Myopathy 2 (AR NEB; ~183 exons; nemaline rods PATHOGNOMONIC; exon55 Ashkenazi; NIV)": ["NEB"],
            "Central Core Disease (AD/AR RYR1; MHS1-MH; volatile+succinylcholine ABSOLUTELY CI; dantrolene)": ["RYR1"],
            "Nemaline Myopathy 3 (ACTA1 de novo AD/AR; intranuclear rods EM PATHOGNOMONIC; neonatal lethal)": ["ACTA1"],
            "Nemaline Myopathy 4 / Distal Arthrogryposis DA1 (AD TPM2; trismus PATHOGNOMONIC; cap disease; mild)": ["TPM2"],
            "Nemaline Myopathy 1 (AD/AR TPM3; type1 fibre uniformity HALLMARK; CFTD; head drop; mild)": ["TPM3"],
            "X-linked Myotubular Myopathy XLMTM (XLR MTM1; neonatal ventilator; centrally placed nuclei; hepatopathy)": ["MTM1"],
            "Centronuclear Myopathy 1 (AD DNM2; necklace fibres; ptosis+ophthalmoplegia; CMT2M allelic)": ["DNM2"],
            "Rigid Spine Muscular Dystrophy 1 RSMD1 (AR SELENON; rigid spine; respiratory failure before limb weakness)": ["SELENON"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "mean_diagnosis_age_y": mean_dx_age,
        "kpis": [
            {"label": "Total Patients", "value": n, "color": "#37474f"},
            {"label": "Genes Covered", "value": len(CM_GENES), "color": "#2e7d32"},
            {"label": "Patients/Gene", "value": 40, "color": "#6a1b9a"},
            {"label": "Ventilator-Dependent", "value": f"{round(100 * vent_n / n, 1)}%", "color": "#b71c1c"},
            {"label": "MH Risk (RYR1)", "value": f"{round(100 * mh_n / n, 1)}%", "color": "#e65100"},
            {"label": "Seeds", "value": "1086–1093", "color": "#37474f"},
        ],
        "clinical_features_prevalence": {
            "Disease Progression": round(100 * prog_n / n, 1),
            "Ventilator Dependent": round(100 * vent_n / n, 1),
            "Malignant Hyperthermia Risk": round(100 * mh_n / n, 1),
            "Hepatopathy (MTM1)": round(100 * hepat_n / n, 1),
            "On Targeted/Trial Therapy": round(100 * targeted_n / n, 1),
            "Drug-Prescribing Error Detected": round(100 * drug_err_n / n, 1),
        },
        "drug_alerts": [
            "RYR1 (Central Core Disease / MHS1): VOLATILE ANAESTHETIC AGENTS (halothane, sevoflurane, "
            "desflurane, isoflurane, enflurane) + SUCCINYLCHOLINE ABSOLUTELY CONTRAINDICATED — "
            "MALIGNANT HYPERTHERMIA (MHS1 = highest risk MH locus); "
            "DANTROLENE 2.5 mg/kg IV BOLUS is the emergency antidote — MUST be in all ORs; "
            "MH ALERT CARD issued to ALL RYR1 patients; IVCT/CHCT for all first-degree relatives; "
            "SAFE ANAESTHESIA: TIVA (propofol + non-depolarising NMBD — rocuronium/vecuronium)",
            "ALL CONGENITAL MYOPATHIES: SUCCINYLCHOLINE (suxamethonium) CONTRAINDICATED — "
            "myopathic muscle membrane instability → hyperkalaemia → cardiac arrest; "
            "use ROCURONIUM (reversed with sugammadex) or vecuronium instead; "
            "volatile agents additionally contraindicated in RYR1 (see above)",
            "MTM1 (XLMTM): HEPATOPATHY in ~10% — screen LIVER ENZYMES (ALT, AST, GGT) "
            "AT DIAGNOSIS and every 6 months in ALL XLMTM patients; "
            "avoid hepatotoxic medications; hepatology referral if elevated; "
            "FEMALE CARRIERS: some develop mild myopathy — assess clinically; "
            "ASPLENISM — check Howell-Jolly bodies; vaccinate if asplenic",
            "NEB (Nemaline Myopathy 2): RESPIRATORY FAILURE can PRECEDE significant limb weakness — "
            "MANDATORY annual spirometry + sleep study regardless of limb function; "
            "start NIV at FVC <50% or nocturnal hypoventilation; "
            "MLPA MANDATORY (exon 55 deletion missed by sequencing alone); "
            "exon 55 deletion screen first in ASHKENAZI JEWISH patients",
            "SELENON (RSMD1): RESPIRATORY FAILURE PRECEDES LIMB WEAKNESS — "
            "RIGID SPINE MASKS DYSPNEA (limited mobility = reduced oxygen demand = no perceived breathlessness); "
            "MANDATORY annual spirometry + sleep study; "
            "START NIV at FVC <60% or nocturnal hypoventilation — do NOT wait for daytime symptoms; "
            "SELENIUM SUPPLEMENTATION: NO benefit — SELENON LOF is protein function loss, not selenium deficiency",
            "ACTA1 (NEM3 — neonatal lethal): GOALS OF CARE DISCUSSION MANDATORY in severe neonatal de novo cases — "
            "INTRANUCLEAR RODS on EM = ACTA1 almost pathognomonic; "
            "distinguish de novo AD (~75% severe) from AR (~25% milder) — "
            "critical for recurrence risk counselling; "
            "neonatal ICU from birth in lethal form",
            "DNM2 (Centronuclear Myopathy 1): CMT2M ALLELIC DISEASE — "
            "NCS/EMG MANDATORY for ALL DNM2-CNM patients AND first-degree relatives "
            "(axonal neuropathy may coexist); "
            "NECKLACE FIBRES on oxidative stain = DNM2 CNM (distinguishes from MTM1 without necklace fibres)",
        ],
        "diagnostic_pearls": [
            "NEB (NEM2): NEMALINE RODS on Gomori trichrome (red/purple rods) PATHOGNOMONIC; "
            "MLPA mandatory — exon 55 deletion (Ashkenazi founder) missed by sequencing; "
            "respiratory failure may precede limb weakness in severe NEB",
            "RYR1 (CCD): CENTRAL CORES on NADH-TR (pale oval areas in type 1 fibres) PATHOGNOMONIC; "
            "MHS1 — highest MH risk; volatile agents + succinylcholine ABSOLUTELY CI; "
            "congenital hip dislocation may be the presenting feature in AD CCD",
            "ACTA1 (NEM3): INTRANUCLEAR RODS on ELECTRON MICROSCOPY — almost pathognomonic for ACTA1; "
            "wide phenotypic spectrum — neonatal lethal (de novo AD) to mild adult (AR); "
            "distinguish de novo AD from AR (affects recurrence risk)",
            "TPM2 (NEM4/DA1): TRISMUS (restricted mouth opening) is ALMOST PATHOGNOMONIC for TPM2 DA1; "
            "check jaw opening in ALL distal arthrogryposis patients; "
            "MILDEST prognosis among nemaline myopathy genes",
            "TPM3 (NEM1): TYPE 1 FIBRE UNIFORMITY + HYPOTROPHY on biopsy = CFTD — HALLMARK of TPM3; "
            "HEAD DROP / neck flexor weakness is characteristic; "
            "generally MILD — ambulation maintained throughout life",
            "MTM1 (XLMTM): SEVERELY AFFECTED MALE NEONATE with hypotonia + respiratory failure at birth → "
            "XLMTM until proven otherwise; CENTRALLY PLACED NUCLEI on biopsy PATHOGNOMONIC; "
            "screen LIVER ENZYMES at diagnosis (hepatopathy 10%); check Howell-Jolly bodies (asplenism)",
            "DNM2 (CNM1): CENTRALLY PLACED NUCLEI + NECKLACE FIBRES on oxidative stain — "
            "necklace fibres DISTINGUISH DNM2 from MTM1; MILDER than XLMTM; childhood-adult onset; "
            "ptosis + ophthalmoplegia — check NCS/EMG (CMT2M allelic)",
            "SELENON (RSMD1): SPINE RIGID + CONTRACTURES BEFORE LIMB WEAKNESS — "
            "limited mobility masks dyspnea; MANDATORY sleep study even without symptoms; "
            "mini-cores on NADH-TR (NOT central cores — distinguish from RYR1); "
            "selenium supplementation does NOT help despite gene name",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    breakdown = {}
    for gene_data in CM_GENES:
        gene = gene_data["gene"]
        gene_pts = [p for p in patients if p["gene"] == gene]
        n = len(gene_pts)
        sev = {s: sum(1 for p in gene_pts if p["severity"] == s) for s in ("Mild", "Moderate", "Severe")}
        f_n = sum(1 for p in gene_pts if p["sex"] == "F")

        breakdown[gene] = {
            "gene": gene,
            "protein": gene_data["protein"],
            "alias": gene_data["alias"],
            "n_patients": n,
            "disease_type": gene_data["disease_type"],
            "locus": gene_data["locus"],
            "omim_gene": gene_data["omim_gene"],
            "omim_disease": gene_data["omim_disease"],
            "inheritance": gene_data["inheritance"],
            "phenotype": gene_data["phenotype"],
            "treatment_options": gene_data["treatment_options"],
            "key_ddx": gene_data["key_ddx"],
            "mechanism": gene_data["mechanism"],
            "myopathy_type": gene_data["myopathy_type"],
            "first_line_drug": gene_data["first_line_drug"],
            "critical_avoid": gene_data["critical_avoid"],
            "severity_distribution": {
                "mild_pct": round(100 * sev["Mild"] / n, 1),
                "moderate_pct": round(100 * sev["Moderate"] / n, 1),
                "severe_pct": round(100 * sev["Severe"] / n, 1),
            },
            "mean_onset_age_y": round(sum(p["onset_age_y"] for p in gene_pts) / n, 2),
            "mean_diagnosis_age_y": round(sum(p["diagnosis_age_y"] for p in gene_pts) / n, 2),
            "sex_pct_female": round(100 * f_n / n, 1),
            "ventilator_dependent_pct": round(100 * sum(1 for p in gene_pts if p["ventilator_dependent"]) / n, 1),
            "mh_risk_pct": round(100 * sum(1 for p in gene_pts if p["mh_risk"]) / n, 1),
            "hepatopathy_pct": round(100 * sum(1 for p in gene_pts if p["hepatopathy"]) / n, 1),
            "drug_error_pct": round(100 * sum(1 for p in gene_pts if p["drug_avoid_prescribed_error"]) / n, 1),
            "on_targeted_therapy_pct": round(100 * sum(1 for p in gene_pts if p["on_targeted_therapy"]) / n, 1),
            "progression_pct": round(100 * sum(1 for p in gene_pts if p["disease_progression"]) / n, 1),
            "cognitive_impairment_pct": round(100 * sum(1 for p in gene_pts if p["cognitive_impairment"]) / n, 1),
        }
    return {
        "atlas": "Congenital-Myopathy-Atlas",
        "subtitle": "Per-gene clinical breakdown — 320 patients (8×40, seeds 1086–1093)",
        "genes": breakdown,
        "gene_order": [g["gene"] for g in CM_GENES],
    }


def get_definitions() -> dict:
    return {
        "atlas": "Congenital-Myopathy-Atlas",
        "subtitle": "Clinical and genetic terminology definitions for Congenital Myopathy Atlas",
        "definitions": {
            "Congenital Myopathy": (
                "A heterogeneous group of inherited skeletal muscle diseases presenting at birth or "
                "early childhood, characterised by: (1) hypotonia ('floppy infant'); "
                "(2) characteristic histological findings on muscle biopsy (rods, cores, centrally "
                "placed nuclei, fibre type disproportion); (3) non-progressive or slowly progressive course. "
                "Aetiology: mutations in sarcomeric, membrane, or organelle proteins. "
                "Key distinguishing features from muscular dystrophies: "
                "normal or near-normal CK, non-necrotic biopsy, congenital onset, characteristic "
                "histological marker (rods/cores/central nuclei). "
                "Treatment: predominantly supportive (NIV, physiotherapy, orthopaedic) — "
                "no approved disease-modifying therapy for most congenital myopathies."
            ),
            "Nemaline Rods": (
                "Pathological protein aggregates (rods) visible on Gomori trichrome stain "
                "as red-purple structures on a green background — PATHOGNOMONIC of nemaline myopathy. "
                "Composition: Z-disc proteins (predominantly alpha-actinin + actin + titin) "
                "that aggregate due to sarcomeric disorganisation. "
                "Distribution: subsarcolemmal (beneath the sarcolemma) and perinuclear. "
                "Genes causing nemaline rods: NEB (most common), ACTA1, TPM2, TPM3, TNNT1, CFL2, KBTBD13. "
                "INTRANUCLEAR RODS: rods within the nucleus — ALMOST PATHOGNOMONIC for ACTA1 myopathy. "
                "Electron microscopy (EM) confirms rod structure and location. "
                "Rod number does NOT correlate with clinical severity."
            ),
            "Central Cores (RYR1)": (
                "Circumscribed regions in the centre of type 1 muscle fibres, devoid of mitochondria "
                "and oxidative enzyme activity. Visible as pale oval/circular areas on NADH-TR "
                "(tetrazolium reductase) oxidative stain — PATHOGNOMONIC of Central Core Disease (CCD). "
                "Mechanism: RYR1 mutations → calcium dysregulation → mitochondrial depletion in the "
                "core region → absence of oxidative activity. "
                "Type 1 fibre predominance universal. "
                "DISTINGUISH from: mini-cores (SELENON — multiple small cores); "
                "nemaline rods (NEB/ACTA1 — red on Gomori). "
                "Electron microscopy: structural Z-disc disorganisation in the core."
            ),
            "Malignant Hyperthermia (MH)": (
                "Life-threatening pharmacogenomic crisis triggered by volatile anaesthetic agents "
                "(halothane, sevoflurane, desflurane, isoflurane) ± succinylcholine. "
                "MECHANISM: in MH-susceptible individuals (RYR1 or CACNA1S mutations), "
                "these triggers cause uncontrolled RYR1 calcium release → "
                "sustained muscle contraction → hypermetabolism → hyperthermia, rigidity, "
                "rhabdomyolysis, metabolic acidosis, hyperkalaemia, DIC → death if untreated. "
                "DANTROLENE: specific antidote — blocks RYR1; 2.5 mg/kg IV bolus IMMEDIATELY; "
                "continue 1 mg/kg q5-10min until resolution; "
                "MUST be in every operating theatre. "
                "MH SUSCEPTIBILITY TESTING: IVCT (in vitro contracture test) / CHCT (caffeine-halothane). "
                "MHS1 locus (RYR1 19q13.2) = highest MH risk of all MH loci. "
                "SAFE ANAESTHESIA: TIVA (propofol + non-depolarising NMBD)."
            ),
            "Centrally Placed Nuclei": (
                "Myonuclei located in the centre of muscle fibres instead of the normal peripheral "
                "position beneath the sarcolemma. "
                "NORMAL: >97% of nuclei are peripheral (sarcolemmal). "
                "PATHOLOGICAL: >3-5% centrally placed = centronuclear myopathy. "
                "Genes: MTM1 (XLMTM — XLR, neonatal, centrally placed nuclei + PATHOGNOMONIC; "
                "NO necklace fibres); DNM2 (CNM1 — AD, milder, centrally placed nuclei + "
                "NECKLACE FIBRES characteristic); BIN1 (CNM2 — AR). "
                "DISTINGUISH MTM1 from DNM2: "
                "MTM1 = neonatal severe, no necklace fibres; "
                "DNM2 = childhood-adult, NECKLACE FIBRES on NADH-TR. "
                "Fetal resemblance: centrally placed nuclei resemble fetal myotubes → "
                "arrest of muscle maturation."
            ),
            "Necklace Fibres (DNM2)": (
                "Pale ring of reduced oxidative activity surrounding a central area of normal activity "
                "in type 1 muscle fibres, visible on NADH-TR (oxidative stain). "
                "CHARACTERISTIC of DNM2 centronuclear myopathy (CNM1). "
                "DISTINGUISHES DNM2-CNM from MTM1-XLMTM (MTM1 does NOT have necklace fibres). "
                "Mechanism: T-tubule abnormalities from DNM2 hyperactivity → "
                "ring of altered membrane domain around fibre core. "
                "Combined with centrally placed nuclei = highly suggestive of DNM2-CNM."
            ),
            "Congenital Fibre Type Disproportion (CFTD)": (
                "Biopsy pattern: type 1 muscle fibres are >12% smaller than type 2 fibres "
                "(type 1 fibre hypotrophy relative to type 2). "
                "HALLMARK of TPM3 nemaline myopathy / NEM1: "
                "all fibres are type 1 (type 1 uniformity) AND they are uniformly small (hypotrophy). "
                "Also seen in: ACTA1, RYR1, SELENON (as secondary finding). "
                "ATPase staining (pH 9.4 and 4.3) distinguishes fibre types. "
                "Clinical correlate: proximal weakness, usually mild; head drop in TPM3; "
                "generally good prognosis. "
                "CFTD on biopsy → genetic panel (ACTA1, RYR1, TPM3, SELENON, RYR2)."
            ),
            "Rigid Spine Syndrome (SELENON)": (
                "Early and prominent spinal rigidity — limited forward flexion of the cervical and "
                "thoracic spine — developing BEFORE significant limb weakness. "
                "HALLMARK of SELENON (SEPN1) myopathy — RSMD1. "
                "Also seen in: EMD (Emery-Dreifuss — but with cardiac arrhythmia PATHOGNOMONIC); "
                "LMNA muscular dystrophy. "
                "CRITICAL CLINICAL PEARL: rigid spine → restrictive chest wall → "
                "restrictive lung disease BEFORE the patient is wheelchair bound or severely weak; "
                "LIMITED MOBILITY MASKS DYSPNEA (patient doesn't feel breathless because activity is reduced). "
                "MANAGEMENT: annual spirometry + sleep study MANDATORY; "
                "NIV at FVC <60% or nocturnal hypoventilation — do NOT wait for symptoms."
            ),
            "X-linked Myotubular Myopathy (XLMTM)": (
                "Severe congenital myopathy caused by hemizygous MTM1 mutations on Xq28. "
                "X-LINKED RECESSIVE: males severely affected; females usually carriers (unaffected) "
                "but some carrier females develop mild myopathy from X-inactivation skewing. "
                "NEONATAL ONSET: profound hypotonia + respiratory failure at birth → "
                "mechanical ventilation within hours of birth. "
                "CENTRALLY PLACED NUCLEI: PATHOGNOMONIC on biopsy — fibres resemble fetal myotubes. "
                "HEPATOPATHY: ~10% — screen liver enzymes ALL patients. "
                "ASPLENISM: check Howell-Jolly bodies; vaccinate if asplenic. "
                "DNM2-MTM1 INTERACTION: DNM2 overactivity worsens XLMTM phenotype → "
                "DNM2-ASO (antisense oligonucleotide) reduces DNM2 as therapeutic strategy."
            ),
            "Distal Arthrogryposis Type 1 (DA1 — TPM2)": (
                "Congenital contractures of DISTAL joints (fingers, wrists, ankles, toes) "
                "present at birth. Caused by TPM2 (beta-tropomyosin) GOF variants. "
                "CLINICAL FEATURES: camptodactyly (finger flexion contractures); "
                "talipes equinovarus (clubfoot); wrist contractures; "
                "TRISMUS (restricted jaw opening — masseter/pterygoid involvement) — "
                "ALMOST PATHOGNOMONIC for TPM2 DA1 (assess jaw opening in all DA1 patients). "
                "PROGNOSIS: good — most patients ambulant; "
                "contractures are the primary management challenge (serial casting, orthopaedic surgery). "
                "DISTINCTION: DA1 (TPM2) = distal contractures + trismus; "
                "DA2B/Sheldon-Hall = other thin filament genes; "
                "Freeman-Sheldon (MYH3) = severe craniofacial + distal."
            ),
            "Minicore Lesions (SELENON)": (
                "Multiple small focal areas of reduced oxidative activity within muscle fibres, "
                "visible on NADH-TR stain as small pale zones. "
                "CHARACTERISTIC of SELENON (RSMD1) myopathy. "
                "DISTINGUISH from CENTRAL CORES (RYR1): "
                "central cores = single large central area per fibre; "
                "mini-cores = multiple small areas distributed throughout fibre. "
                "Mechanism: focal mitochondrial depletion + sarcomeric disorganisation "
                "from selenoprotein N deficiency. "
                "Mallory-like bodies (desmin aggregates) may accompany mini-cores in SELENON. "
                "Also seen in: RYR1 (multi-minicore disease — biallelic AR — alongside ophthalmoplegia); "
                "SELENON is the most common cause of pure multi-minicore disease."
            ),
            "Intranuclear Rods (ACTA1)": (
                "Nemaline rods located WITHIN myonuclei — detected by electron microscopy (EM). "
                "ALMOST PATHOGNOMONIC for ACTA1 myopathy (skeletal muscle alpha-actin mutations). "
                "Mechanism: mutant ACTA1 protein mislocalises to nucleus → forms rod-like "
                "aggregates within nuclear compartment. "
                "NOT seen in NEB or TPM2/TPM3 nemaline myopathies (subsarcolemmal/cytoplasmic only). "
                "BIOPSY PROTOCOL: routine Gomori trichrome + electron microscopy required to detect "
                "intranuclear rods (light microscopy may miss nuclear localisation). "
                "Clinical context: severe ACTA1 myopathy, especially de novo dominant neonatal lethal form."
            ),
            "Non-invasive Ventilation (NIV) in Congenital Myopathy": (
                "Mechanical respiratory support delivered via mask (nasal or full-face) without intubation. "
                "CORNERSTONE of treatment in NEB nemaline myopathy, SELENON RSMD1, and advanced forms. "
                "INDICATIONS: FVC <50% (NEB/ACTA1/MTM1/DNM2) or FVC <60% (SELENON) OR "
                "nocturnal hypoventilation on sleep study (desaturation or hypercapnia). "
                "MANDATORY MONITORING: annual spirometry (FVC) + overnight sleep oximetry/CO2 in ALL "
                "congenital myopathy patients — respiratory failure can precede limb weakness "
                "(especially NEB and SELENON); "
                "do NOT rely on symptom reporting (limited mobility masks dyspnea). "
                "MODES: BiPAP (bilevel positive airway pressure) most common; CPAP for milder cases. "
                "TRACHEOSTOMY: if NIV fails or bulbar dysfunction prevents effective NIV."
            ),
            "Malignant Hyperthermia Susceptibility Testing (IVCT/CHCT)": (
                "In vitro contracture test (IVCT — European protocol) / "
                "Caffeine-Halothane Contracture Test (CHCT — North American protocol). "
                "DIAGNOSTIC: muscle biopsy from vastus lateralis exposed to caffeine and halothane — "
                "MH-susceptible muscle contracts abnormally. "
                "INDICATIONS: RYR1 mutation (especially AD GOF) identified; "
                "family member of known MH patient without identified mutation; "
                "unexplained perioperative crisis. "
                "RESULT INTERPRETATION: MH-susceptible (MHS), MH-equivocal (MHE), MH-normal (MHN). "
                "All RYR1 AD GOF patients are presumed MH-susceptible even without IVCT/CHCT — "
                "MH precautions MANDATORY. "
                "MH ALERT CARD: issued to all patients and families; "
                "medical alert bracelet recommended. "
                "EUROMAC registry: European MH registry for case reporting."
            ),
            "Charcot-Marie-Tooth disease type 2M (CMT2M — DNM2)": (
                "Axonal peripheral neuropathy caused by DNM2 mutations — ALLELIC with DNM2 centronuclear myopathy. "
                "IMPORTANT: same gene (DNM2 on 19p13.2), different variants cause either: "
                "(1) Centronuclear Myopathy (CNM1 — muscle-predominant), or "
                "(2) CMT2M (neuropathy-predominant), or "
                "(3) BOTH in some families. "
                "CLINICAL: axonal neuropathy — distal weakness, areflexia, pes cavus, "
                "foot drop; nerve conduction studies show reduced CMAP amplitudes with "
                "relatively preserved conduction velocities (axonal pattern). "
                "MANAGEMENT: NCS/EMG at diagnosis of DNM2-CNM; "
                "ankle-foot orthoses for foot drop; referral to CMT clinic if neuropathy significant. "
                "GENETIC COUNSELLING: DNM2 families may have members with myopathy, neuropathy, or both."
            ),
        },
    }
