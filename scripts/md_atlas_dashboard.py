#!/usr/bin/env python3
"""MD-Atlas — Complete 8-Gene Muscular Dystrophy Atlas
DMD    (Dystrophin; 3685 aa; Xp21.2; Duchenne/Becker MD; exon-skipping/ataluren/gene-therapy; STEROIDS MANDATORY) ·
DMPK   (Dystrophia Myotonica Protein Kinase; 629 aa; 19q13.32; DM1; CTG repeat; RNA toxicity; AVOID GA class Ic) ·
CNBP   (Cellular Nucleic Acid-Binding Protein; 177 aa; 3q21.3; DM2; CCTG repeat; NO neonatal form; milder) ·
EMD    (Emerin; 254 aa; Xq28; EDMD1 X-linked; LAMIN PATHWAY; cardiac LETHAL arrhythmia; ICD MANDATORY) ·
SMCHD1 (Structural Maintenance of Chromosomes Flexible Hinge Domain-Containing 1; 2005 aa; 18p11.32; FSHD2; DUX4 derepression) ·
COL6A1 (Collagen Type VI α1; 1028 aa; 21q22.3; Bethlem Myopathy AD mild / Ullrich CMD AR severe; Collagen 6 spectrum) ·
LAMA2  (Laminin α2; 3112 aa; 6q22.33; MDC1A merosin-deficient CMD; WHITE MATTER T2 UNIVERSAL; stable adult wheelchair) ·
POMT1  (Protein O-Mannosyltransferase 1; 747 aa; 9q34.13; Walker-Warburg Syndrome; cobblestone lissencephaly; FATAL infantile)
320-patient aggregate cohort (8 × 40, seeds 1062–1069)
"""

import random

SEED_BASE = 1062

MD_GENES = [
    # ── DMD — Dystrophin ──────────────────────────────────────────────────
    {
        "gene": "DMD", "protein": "Dystrophin",
        "alias": "DMD; OMIM gene 300377; Xp21.2; 3685 aa; 427 kDa; Duchenne MD (OMIM #310200); Becker MD (OMIM #300376); X-linked recessive; exon-skipping; ataluren; delandistrogene moxeparvovec",
        "aa": "3685 aa", "kDa": "427 kDa",
        "mechanism": (
            "DMD encodes dystrophin, the largest known human protein (427 kDa; 3685 aa), forming a "
            "critical structural link between the intracellular actin cytoskeleton and the extracellular "
            "matrix via the dystrophin-associated protein complex (DAPC). Absence of dystrophin → "
            "sarcolemmal fragility → contraction-induced membrane tears → Ca2+ influx → muscle fibre "
            "necrosis → progressive replacement with fat and fibrosis. "
            "READING-FRAME RULE: OUT-OF-FRAME deletions/duplications → no functional protein → DUCHENNE (severe). "
            "IN-FRAME mutations → truncated but partially functional protein → BECKER (milder). "
            "MOST COMMON MUTATION: deletion of one or more exons (60–70% DMD patients); "
            "exon 51 skip-amenable = ~13% of all DMD. "
            "EXON-SKIPPING THERAPIES: eteplirsen (exon 51), golodirsen/viltolarsen (exon 53), "
            "casimersen (exon 45), mavodersen (exon 44) — antisense oligonucleotides that restore "
            "in-frame reading → truncated 'Becker-like' dystrophin. "
            "ATALUREN: nonsense mutation suppression (ribosomal read-through); PTC124; "
            "FDA approved for nmDMD (premature stop codon, ~10–15% of patients). "
            "DELANDISTROGENE MOXEPARVOVEC (Elevidys): AAV-rh74-MHCK7-micro-dystrophin gene therapy; "
            "FDA accelerated approval 2023 for ages 4–5; delivers functional micro-dystrophin. "
            "CORTICOSTEROIDS: deflazacort (first-line; superior; less weight gain) or prednisolone — "
            "MANDATORY for ALL DMD. Delay loss of ambulation by 2–5 years; preserve respiratory function; "
            "reduce cardiomyopathy onset. START at plateau phase (maximum strength, before decline)."
        ),
        "disease_type": "Duchenne/Becker Muscular Dystrophy (XLR)",
        "locus": "Xp21.2", "omim_gene": 300377, "omim_disease": 310200,
        "inheritance": (
            "X-LINKED RECESSIVE (XLR). Males affected; females are carriers (heterozygous). "
            "1/3 of DMD cases are de novo mutations (no family history). "
            "Carrier females: CK elevated in ~70%; mild myopathy in ~8% (manifesting carriers); "
            "dilated cardiomyopathy risk in ~20% of carriers — CARDIAC SCREENING MANDATORY. "
            "INHERITANCE PEARLS: 2/3 of affected boys have carrier mothers; "
            "1/3 have new mutations. If mother is carrier: 50% sons affected, 50% daughters are carriers."
        ),
        "phenotype": (
            "DUCHENNE (out-of-frame): Gowers sign by age 3–5; loss of ambulation by age 12 (without steroids); "
            "respiratory failure (VC <1L) in teens; dilated cardiomyopathy (DCM) in ALL by 18 years. "
            "BECKER (in-frame): onset variable (childhood to adult); loss of ambulation variable (>16 years or never); "
            "cardiomyopathy is the DOMINANT morbidity in Becker (sometimes presenting feature in adults). "
            "CARDIAC: DCM universal in DMD; onset ~10 years; arrhythmias; heart failure; "
            "ACE inhibitor/ARB + beta-blocker from age 10 REGARDLESS of LV function. "
            "RESPIRATORY: FVC monitoring from age 6; NIV when FVC <50% or FVC declining; "
            "tracheostomy in some; cough assist devices MANDATORY. "
            "COGNITIVE: IQ ~1 SD below average (intrinsic, not secondary to respiratory); "
            "ADHD, autism, learning disability co-occur. "
            "CK: MASSIVELY ELEVATED at birth (10,000–50,000 IU/L in DMD); "
            "falls with disease progression as muscle is replaced by fat."
        ),
        "treatment_options": [
            "Deflazacort 0.9 mg/kg/day: FIRST-LINE steroid for DMD; superior to prednisolone; "
            "less weight gain; better bone density; preserves ambulation + respiratory function; monitor cataracts",
            "Prednisolone 0.75 mg/kg/day: alternative if deflazacort unavailable; similar efficacy; "
            "monitor weight/blood pressure/bone mineral density/glycaemia",
            "Exon-skipping ASOs: eteplirsen (exon 51, ~13% DMD); golodirsen/viltolarsen (exon 53, ~8%); "
            "casimersen (exon 45, ~8%); requires deletion/mutation amenable to that specific skip",
            "Ataluren (PTC124) 40 mg/kg/day: for premature stop codon DMD (~10-15%); "
            "read-through of UGA/UAA/UAG nonsense mutations; NOT for frameshift/deletion",
            "Delandistrogene moxeparvovec (Elevidys): AAV-micro-dystrophin gene therapy; "
            "FDA approved 2023 for age 4-5; single IV infusion; neutralising antibody screen required",
            "ACE inhibitor (lisinopril/perindopril) + beta-blocker: MANDATORY from age 10 years; "
            "from diagnosis if DCM present; cardioprotective regardless of echo findings",
            "NIV (BiPAP): when FVC <50% or symptomatic hypoventilation; nocturnal first; "
            "cough assist (MI-E) from early stage; invasive ventilation decision counselling",
            "Physiotherapy + orthoses: maintain ambulation; prevent contractures; ankle-foot orthoses; "
            "scoliosis spine fusion when Cobb >20° (improves comfort + respiratory function)",
        ],
        "key_ddx": [
            "Becker MD (in-frame DMD mutation) — same gene; milder; adult-onset cardiomyopathy dominant",
            "LGMD R1 (CAPN3) / LGMD R2 (DYSF) — autosomal recessive; normal dystrophin IHC; WES",
            "Spinal muscular atrophy (SMA, SMN1) — proximal weakness in infants; no CK elevation",
            "Metabolic myopathy (e.g. Pompe, GSD) — enzyme assay distinguishes; CK variable",
            "X-linked dilated cardiomyopathy (XL-DCM, DMD cardiac alleles) — heart-only BMD variant",
        ],
        "onset_range_y": (2, 6),
        "cardiac_risk": True,
        "dcm_risk": True,
        "steroid_responsive": True,
        "exon_skipping": True,
        "respiratory_failure": True,
        "cognitive_involvement": True,
        "contractures": True,
        "scoliosis_risk": True,
        "ck_range": (5000, 30000),
        "first_line_drug": "Deflazacort",
        "critical_avoid": "Succinylcholine (hyperkalaemia/rhabdo); uncorrected scoliosis before NIV",
    },
    # ── DMPK — Myotonic Dystrophy Type 1 (DM1) ───────────────────────────
    {
        "gene": "DMPK", "protein": "Myotonic Dystrophy Protein Kinase (DMPK)",
        "alias": "DMPK; OMIM gene 605377; 19q13.32; 629 aa; Myotonic Dystrophy type 1 DM1 (OMIM #160900); CTG trinucleotide repeat; MBNL1 sequestration; RNA toxicity; MOST COMMON adult muscular dystrophy",
        "aa": "629 aa", "kDa": "69 kDa",
        "mechanism": (
            "DMPK encodes myotonic dystrophy protein kinase. The disease is caused by CTG repeat "
            "expansion in the 3'UTR of DMPK. Normal: 5–37 CTG. Premutation: 38–49. "
            "DM1 mild: 50–150. Classic DM1: 100–1000. Congenital: >1000 (maternal inheritance). "
            "MECHANISM: Expanded CUG RNA transcript folds into hairpin structures → sequesters "
            "MBNL1 (muscleblind-like splicing regulator) in nuclear foci → loss of MBNL1 splicing function "
            "→ aberrant splicing of multiple downstream targets: "
            "ClC-1 (CLCN1) → myotonia; IR (INSR) → insulin resistance; "
            "cTNT (TNNT2) → cardiac conduction; Tau (MAPT) → CNS effects. "
            "CONCURRENT CELF1 UPREGULATION further disrupts splicing. "
            "ANTICIPATION: CTG repeat EXPANDS between generations — offspring more severely affected, "
            "especially maternal transmission (congenital DM1 almost exclusively via mother). "
            "RNA TOXICITY: unlike most trinucleotide repeat diseases, the expanded DMPK RNA "
            "itself (not protein) is the pathogenic agent — RNA gain-of-function."
        ),
        "disease_type": "Myotonic Dystrophy Type 1 / DM1 (AD, CTG repeat)",
        "locus": "19q13.32", "omim_gene": 605377, "omim_disease": 160900,
        "inheritance": (
            "AUTOSOMAL DOMINANT. CTG repeat instability → anticipation (each generation more severe). "
            "CONGENITAL DM1: almost exclusively MATERNAL transmission (maternal CTG >1000); "
            "father with DM1 RARELY transmits congenital form (paternal meiotic expansion bias lower). "
            "PENETRANCE: near 100% with >100 CTG; variable clinical expression. "
            "Repeat-primed PCR (RP-PCR) + Southern blot for large expansions. "
            "Standard fragment analysis misses large expansions (>200 CTG)."
        ),
        "phenotype": (
            "MULTISYSTEM DISORDER — 'tip of the iceberg' phenotype (myotonia often the presenting symptom). "
            "MYOTONIA: grip myotonia + percussion myotonia; WARM-UP phenomenon (improves with exercise); "
            "paradoxical myotonia in cold (unlike CLCN1 where warm-up is ALSO present). "
            "WEAKNESS: distal onset (foot drop, hand weakness); facial weakness (ptosis, hatchet face, "
            "wasting temporalis/SCM); dysarthria; dysphagia. "
            "CARDIAC: 1st degree AV block → complete heart block (sudden death); "
            "ventricular arrhythmia; ANNUAL ECG + HOLTER MANDATORY; "
            "pacemaker/ICD threshold LOW in DM1 (sudden death risk). "
            "RESPIRATORY: diaphragm + intercostal weakness → nocturnal hypoventilation → NIV. "
            "CNS: COGNITIVE IMPAIRMENT (executive function); hypersomnia (excessive daytime sleepiness — DISABLING); "
            "apathy; white matter lesions on MRI. "
            "ENDOCRINE: insulin resistance; early cataracts (slit lamp MANDATORY — posterior subscapular 'Christmas tree'). "
            "GI: dysmotility (constipation, dysphagia, pseudo-obstruction); cholangiopathy. "
            "CONGENITAL DM1: neonatal hypotonia ('floppy baby'); respiratory failure at birth; "
            "clubfoot; intellectual disability; mother always affected (often mildly)."
        ),
        "treatment_options": [
            "Mexiletine 150–300 mg TDS: FIRST-LINE for DM1 myotonia; reduces grip myotonia; "
            "NOT a disease-modifying agent but improves function; monitor for cardiac conduction effects",
            "Cardiac surveillance: annual ECG + Holter 24h; pacemaker if PR >240ms or 2nd-degree AV block; "
            "ICD if ventricular arrhythmia; LOW threshold for pacemaker in DM1",
            "NIV (BiPAP): nocturnal hypoventilation (FVC <50% or symptomatic); "
            "annual respiratory function tests (spirometry + sniff nasal pressure)",
            "Modafinil 100–400 mg/day: excessive daytime sleepiness (EDS) is DISABLING; "
            "modafinil addresses hypersomnia; sodium oxybate evidence emerging",
            "Cataracts: annual slit-lamp exam; posterior subscapular 'Christmas tree' cataracts pathognomonic; "
            "surgical extraction when visual impairment occurs",
            "AVOID class Ia antiarrhythmics (quinidine, procainamide, disopyramide): worsen myotonia; "
            "AVOID class Ic (flecainide, propafenone) in cardiac DM1: proarrhythmic",
            "Annual ophthalmology, endocrine (insulin resistance, thyroid), gastroenterology review",
            "Genetic counselling: anticipation; maternal transmission for congenital form; "
            "prenatal genetic diagnosis by CVS/amniocentesis; preimplantation genetic testing available",
        ],
        "key_ddx": [
            "DM2 (CNBP) — CCTG repeat; proximal weakness (NOT distal); milder; NO congenital form; "
            "CKMM elevated; neck flexors SPARED (weak in DM1)",
            "CLCN1 Myotonia Congenita — pure myotonia; NO systemic features; NO cardiac/respiratory/CNS",
            "SCN4A Paramyotonia Congenita — cold-triggered; no systemic features",
            "PROMM — myotonic-like disorder without DM1/DM2; exclude by molecular testing",
            "Ocular/pharyngeal myopathy (OPMD, GAN, VCP) — proximal weakness; different clinical context",
        ],
        "onset_range_y": (10, 45),
        "cardiac_risk": True,
        "dcm_risk": False,
        "steroid_responsive": False,
        "exon_skipping": False,
        "respiratory_failure": True,
        "cognitive_involvement": True,
        "contractures": False,
        "scoliosis_risk": False,
        "ck_range": (100, 600),
        "first_line_drug": "Mexiletine (myotonia)",
        "critical_avoid": "Class Ia/Ic antiarrhythmics; uncorrected cardiac conduction disease",
    },
    # ── CNBP — Myotonic Dystrophy Type 2 (DM2) ───────────────────────────
    {
        "gene": "CNBP", "protein": "Cellular Nucleic Acid-Binding Protein (CNBP / ZNF9)",
        "alias": "CNBP (formerly ZNF9); OMIM gene 116955; 3q21.3; 177 aa; Myotonic Dystrophy type 2 DM2 (OMIM #602668); CCTG tetranucleotide repeat; PROXIMAL weakness; NO congenital form; milder",
        "aa": "177 aa", "kDa": "19 kDa",
        "mechanism": (
            "CNBP (ZNF9) encodes cellular nucleic acid-binding protein, a CCCH zinc-finger RNA-binding protein. "
            "DM2 is caused by CCTG tetranucleotide repeat expansion in intron 1 of CNBP. "
            "Normal: <26 CCTG. DM2 affected: 75–11,000 CCTG. "
            "MECHANISM: CCUG-repeat RNA hairpins (from CCTG expansion) sequester MBNL1 nuclear foci "
            "→ same downstream splicing dysregulation as DM1 (ClC-1, INSR, TNNT2). "
            "RNA GAIN-OF-FUNCTION: CCUG RNA foci detectable in muscle biopsies. "
            "DETECTION: repeat-primed PCR detects the expansion; "
            "SOUTHERN BLOT required to size expansion (can exceed 11,000 CCTG). "
            "KEY DIFFERENCES FROM DM1: CCTG not CTG; tetranucleotide not trinucleotide; "
            "PROXIMAL weakness not distal; cardiac conduction disease less common; "
            "NO congenital form (no neonatal DM2 exists — critical clinical rule). "
            "DM2 is often UNDERDIAGNOSED due to milder course and rarity of testing."
        ),
        "disease_type": "Myotonic Dystrophy Type 2 / DM2 (AD, CCTG repeat)",
        "locus": "3q21.3", "omim_gene": 116955, "omim_disease": 602668,
        "inheritance": (
            "AUTOSOMAL DOMINANT. CCTG repeat expansion. "
            "ANTICIPATION LESS MARKED than DM1 (tetranucleotide instability different from CTG). "
            "NO congenital form — most important distinguishing feature from DM1. "
            "Prevalence: comparable to DM1 in some European populations (Germany/Finland); "
            "rare in East Asian/African populations. "
            "Standard PCR misses large CCTG expansions → repeat-primed PCR required."
        ),
        "phenotype": (
            "MULTISYSTEM DISORDER — milder than DM1 overall. "
            "WEAKNESS: PROXIMAL pattern (hip-flexors, elbow-flexors, finger-flexors — opposite to DM1 distal pattern); "
            "neck FLEXORS RELATIVELY SPARED (weak in DM1 — 'chin to chest' test). "
            "MYOTONIA: present but often less prominent than DM1; grip myotonia; warm-up phenomenon. "
            "PAIN: myalgia is a prominent, often underappreciated feature of DM2 (muscle pain + cramping). "
            "CARDIAC: AV conduction abnormalities LESS FREQUENT than DM1 but still present; "
            "annual ECG recommended. "
            "COGNITIVE: generally MILDER than DM1; hypersomnia less severe. "
            "CATARACTS: same posterior subscapular pattern as DM1; annual slit-lamp. "
            "ENDOCRINE: insulin resistance; elevated CK (often 5–10× normal — CK ELEVATED is a key clue). "
            "NO CONGENITAL FORM: neonatal disease DOES NOT OCCUR in DM2 — "
            "if suspected, reconsider DM1 (maternal) or myotonia congenita (CLCN1). "
            "DM2 is often misdiagnosed as 'fibromyalgia' or 'idiopathic myalgia'."
        ),
        "treatment_options": [
            "Mexiletine: for myotonia (same as DM1); often less symptomatic in DM2",
            "Pain management: myalgia is prominent; NSAIDs/gabapentin/duloxetine for chronic myalgia; "
            "avoid opioids as first-line",
            "Cardiac: annual ECG; lower intervention threshold than DM1 (AV block less common but reported)",
            "Cataracts: annual slit-lamp; surgical when visually impaired",
            "Endocrine: screen for insulin resistance (HbA1c); thyroid function annually",
            "Genetic counselling: AD; anticipation less marked; NO neonatal form (critical distinction from DM1)",
            "Repeat-primed PCR required for diagnosis — standard PCR and sequencing MISS the expansion",
            "Physiotherapy: proximal strengthening; avoid extreme fatigue; aerobic exercise beneficial",
        ],
        "key_ddx": [
            "DM1 (DMPK) — DISTAL weakness; congenital form possible; CTG not CCTG; CK lower; more cardiac",
            "LGMD (CAPN3, DYSF, FKRP) — AR; no myotonia; CK pattern; molecular genetics",
            "CLCN1 Myotonia Congenita — NO systemic features; NO weakness progression; pure myotonia",
            "Fibromyalgia/idiopathic myalgia — DM2 frequently misdiagnosed; CK elevation + myotonia clue",
            "Hypothyroid myopathy — secondary myotonia/myalgia; TSH screen mandatory in all",
        ],
        "onset_range_y": (20, 60),
        "cardiac_risk": True,
        "dcm_risk": False,
        "steroid_responsive": False,
        "exon_skipping": False,
        "respiratory_failure": False,
        "cognitive_involvement": True,
        "contractures": False,
        "scoliosis_risk": False,
        "ck_range": (200, 1500),
        "first_line_drug": "Mexiletine (myotonia) / NSAIDs (myalgia)",
        "critical_avoid": "Opioids first-line; statins (worsens myalgia); missed diagnosis (often idiopathic myalgia label)",
    },
    # ── EMD — Emery-Dreifuss Muscular Dystrophy Type 1 ───────────────────
    {
        "gene": "EMD", "protein": "Emerin",
        "alias": "EMD (STA); OMIM gene 300384; Xq28; 254 aa; 29 kDa; Emery-Dreifuss MD Type 1 EDMD1 (OMIM #310300); X-linked recessive; LEM-domain nuclear envelope protein; LETHAL arrhythmia; ICD MANDATORY",
        "aa": "254 aa", "kDa": "29 kDa",
        "mechanism": (
            "EMD encodes emerin, a LEM (Lap2-Emerin-MAN1) domain protein of the inner nuclear membrane. "
            "Emerin interacts with LMNA (lamin A/C), BAF (barrier-to-autointegration factor), nesprin-1, "
            "and actin — forming the LINC complex (Linker of Nucleoskeleton and Cytoskeleton). "
            "MECHANISM OF DISEASE: absence of emerin → loss of nuclear envelope integrity → "
            "nuclear shape abnormalities → altered mechanosensing + chromatin organisation → "
            "muscle-specific gene dysregulation. "
            "CARDIAC: emerin loss particularly severe in cardiac conduction system → "
            "progressive atrioventricular block → complete heart block → SUDDEN CARDIAC DEATH. "
            "CARDIAC ARRHYTHMIA IS THE LEADING CAUSE OF DEATH in EDMD1 and EDMD2 (LMNA). "
            "MUSCLE: three hallmarks — (1) elbow joint contractures (EARLY, before weakness); "
            "(2) rigid spine; (3) scapuloperoneal weakness (humeroperoneal distribution). "
            "CONNECTION TO LMNA: EDMD2 (lamin A/C, LMNA) has IDENTICAL clinical phenotype to EDMD1 "
            "but autosomal dominant — cardiac prognosis MORE SEVERE in LMNA-EDMD than EMD-EDMD."
        ),
        "disease_type": "Emery-Dreifuss Muscular Dystrophy Type 1 / EDMD1 (XLR)",
        "locus": "Xq28", "omim_gene": 300384, "omim_disease": 310300,
        "inheritance": (
            "X-LINKED RECESSIVE. Males severely affected; female carriers usually asymptomatic "
            "but may have cardiac involvement (rare manifesting carriers with atrial standstill). "
            "De novo mutations occur. IHC (immunohistochemistry): emerin absent from nuclear envelope "
            "in affected males → DIAGNOSTIC CONFIRMATION on muscle/skin biopsy. "
            "EDMD2 (LMNA) is autosomal dominant with identical clinical phenotype — "
            "gene panel required to distinguish; LMNA-EDMD has MORE AGGRESSIVE cardiac course."
        ),
        "phenotype": (
            "TRIAD: (1) EARLY JOINT CONTRACTURES (elbow — 'tennis elbow' posture from childhood; "
            "rigid spine; ankle equinus). "
            "(2) SLOWLY PROGRESSIVE WEAKNESS (humeroperoneal distribution — biceps, triceps, "
            "tibialis anterior, peronei — scapular winging). "
            "(3) LIFE-THREATENING CARDIAC ARRHYTHMIA: progressive AV block → complete heart block → "
            "ATRIAL STANDSTILL (pathognomonic low/absent P-waves on ECG); ventricular arrhythmia; "
            "sudden cardiac death. "
            "CARDIAC IS THE DOMINANT MORBIDITY: patients may walk until adult life but die from "
            "arrhythmia if cardiac monitoring is not established. "
            "CK: mildly elevated (300–1500 IU/L). "
            "COGNITION: NORMAL (distinguishes from FKRP dystroglycanopathy). "
            "ONSET: contractures in childhood; weakness in teens; cardiac in young adulthood. "
            "FEMALES: carrier females MUST be screened for cardiac involvement annually — "
            "atrial standstill can occur in female EMD carriers."
        ),
        "treatment_options": [
            "ICD (implantable cardioverter-defibrillator): MANDATORY when ventricular arrhythmia detected; "
            "pacemaker for complete heart block; ICD preferred over pacemaker-only given VT/VF risk",
            "Annual ECG + 48h Holter from diagnosis: cardiac surveillance; "
            "ATRIAL STANDSTILL (flat baseline, no P-waves) is pathognomonic of EDMD — "
            "pacemaker URGENTLY required",
            "Physiotherapy: stretching for contractures; AFOs for equinus; spine bracing for rigid spine",
            "Genetic counselling: X-linked; carrier females need cardiac screening; "
            "distinguish from EDMD2 (LMNA) — same phenotype, worse cardiac in LMNA",
            "Serial echocardiography: dilated cardiomyopathy develops in some; "
            "ACE inhibitor if EF reduced",
            "Emerin IHC on muscle/skin biopsy: absent nuclear staining DIAGNOSTIC; "
            "faster than genetics in emergency",
            "Female EMD carriers: annual cardiac review regardless of symptoms; "
            "arrhythmia can be the first and fatal presentation",
            "AVOID antiarrhythmics without cardiology review — risk of pro-arrhythmia in structurally abnormal heart",
        ],
        "key_ddx": [
            "EDMD2 (LMNA) — autosomal dominant; identical phenotype; MORE aggressive cardiac; "
            "emerin IHC NORMAL in LMNA-EDMD — molecular diagnosis required",
            "LGMD D1 (LMNA-LGMD) — same gene, different clinical pattern; less prominent contractures",
            "Rigid spine muscular dystrophy (SELENON) — rigid spine + respiratory disproportionate; different distribution",
            "Congenital fiber type disproportion — neonatal hypotonia; early contractures; muscle biopsy",
            "Bethlem myopathy (COL6A1) — contractures + weakness; different distribution; autosomal dominant",
        ],
        "onset_range_y": (5, 20),
        "cardiac_risk": True,
        "dcm_risk": True,
        "steroid_responsive": False,
        "exon_skipping": False,
        "respiratory_failure": False,
        "cognitive_involvement": False,
        "contractures": True,
        "scoliosis_risk": False,
        "ck_range": (200, 1500),
        "first_line_drug": "ICD (cardiac) / physiotherapy (contractures)",
        "critical_avoid": "Missing cardiac surveillance; pacemaker-only when VT risk present",
    },
    # ── SMCHD1 — FSHD Type 2 ─────────────────────────────────────────────
    {
        "gene": "SMCHD1", "protein": "Structural Maintenance of Chromosomes Flexible Hinge Domain-Containing 1 (SMCHD1)",
        "alias": "SMCHD1; OMIM gene 614982; 18p11.32; 2005 aa; 232 kDa; FSHD Type 2 (OMIM #158901); DUX4 derepression; D4Z4 epigenetic hypomethylation; digenic inheritance with permissive 4qA allele",
        "aa": "2005 aa", "kDa": "232 kDa",
        "mechanism": (
            "SMCHD1 encodes a chromatin regulator (structural maintenance of chromosomes protein) "
            "that maintains epigenetic silencing of D4Z4 macrosatellite repeats at chromosome 4q35. "
            "D4Z4 repeats normally harbour DUX4 (double homeobox 4) — a transcription factor toxic "
            "to somatic cells. D4Z4 is normally silenced by methylation. "
            "FSHD TYPE 1 (D4Z4 DELETION): D4Z4 repeat contraction on 4q35 → <11 units → "
            "loss of methylation → DUX4 derepression. "
            "FSHD TYPE 2 (SMCHD1 MUTATION): SMCHD1 LOF → epigenetic failure → D4Z4 hypomethylation → "
            "DUX4 derepression → DUX4 protein production → activation of germline/cancer genes "
            "in muscle → muscle cell death. "
            "DIGENIC INHERITANCE: SMCHD1 mutation alone is INSUFFICIENT — must co-inherit a "
            "permissive allele (4qA haplotype with polyadenylation signal that stabilises DUX4 mRNA). "
            "~5% of FSHD cases are FSHD2 (SMCHD1 mutation + 4qA allele). "
            "D4Z4 methylation level at 4q35: <20% → FSHD2 (confirmed by methylation analysis). "
            "SMCHD1 variants also cause BAMS (Bosma arhinia microphthalmia syndrome) — "
            "different mechanism (FSHD2 = hypomorphic LOF; BAMS = GOF → same gene, opposite effect)."
        ),
        "disease_type": "Facioscapulohumeral Muscular Dystrophy Type 2 / FSHD2 (Digenic AD)",
        "locus": "18p11.32", "omim_gene": 614982, "omim_disease": 158901,
        "inheritance": (
            "DIGENIC: SMCHD1 mutation (18p11.32, autosomal) + permissive 4qA allele (chromosome 4q35). "
            "FSHD2 accounts for ~5% of FSHD cases. "
            "SMCHD1 mutation alone: asymptomatic without permissive 4qA allele. "
            "4qA allele alone: asymptomatic without SMCHD1 mutation. "
            "FSHD1 (90-95% of FSHD): D4Z4 contraction (<11 units on 4qA haplotype) — "
            "diagnose by Southern blot (D4Z4 repeat sizing); NOT a single-gene mutation. "
            "SMCHD1 BAMS (Bosma arhinia): DIFFERENT missense GOF mutations → absent/hypoplastic nose; "
            "distinguish from FSHD2 LOF mutations."
        ),
        "phenotype": (
            "FACIOSCAPULOHUMERAL PATTERN: "
            "FACE: facial weakness (cannot whistle/blow; orbicularis oculi weakness — sleep with eyes open); "
            "mask-like facies. "
            "SCAPULAR: winging (rhomboids, serratus anterior weakness) — Beevor sign (umbilicus moves UP "
            "with neck flexion — rectus abdominis asymmetric weakness) is PATHOGNOMONIC. "
            "HUMERAL: biceps/triceps weakness before deltoid (Beevor + humeral weakness = FSHD until proven otherwise). "
            "PERONEAL: foot drop (tibialis anterior weakness) in advanced cases. "
            "ASYMMETRY: HIGHLY CHARACTERISTIC — one side often significantly weaker than the other. "
            "EXTRAMUSCULAR: Coats retinal vasculopathy (rare, mainly congenital/severe early cases); "
            "high-frequency sensorineural deafness (audiogram MANDATORY in congenital FSHD). "
            "FSHD2 tends to be MILDER than FSHD1 (smaller D4Z4 contraction = more severe in FSHD1). "
            "NO CARDIAC OR RESPIRATORY INVOLVEMENT as primary feature (unlike DMD/EDMD/DM1)."
        ),
        "treatment_options": [
            "Losmapimod (p38α/β MAPK inhibitor): phase 3 FSHD trial (REACH trial — losmapimod); "
            "targets DUX4-induced muscle inflammation; most advanced disease-modifying therapy in FSHD",
            "Scapular fixation surgery: surgical stabilisation of scapula to thorax; "
            "improves shoulder elevation and cosmesis; consider when scapular winging severe and arm elevation limited",
            "Physiotherapy: scapular stabilisation exercises; avoid eccentric exercise that worsens weakness; "
            "aerobic exercise (swimming, cycling) beneficial and safe",
            "AFOs (ankle-foot orthoses): for foot drop (peroneal weakness in advanced FSHD)",
            "Respiratory: FVC annually if advanced; NIV for nocturnal hypoventilation (uncommon in FSHD2)",
            "Ophthalmology: Coats disease (fundal assessment in congenital/severe early onset FSHD)",
            "Audiology: high-frequency SNHL screen, especially in early-onset severe FSHD",
            "Genetic counselling: digenic; SMCHD1 partner must be counselled about 4qA allele co-inheritance; "
            "D4Z4 methylation analysis confirms FSHD2; distinguish FSHD1 (Southern blot) from FSHD2",
        ],
        "key_ddx": [
            "FSHD1 (D4Z4 deletion, 4qA allele) — 90-95% of FSHD; Southern blot D4Z4 sizing; same phenotype",
            "LGMD (CAPN3, DYSF) — no facial weakness; AR; different pattern",
            "Polymyositis/dermatomyositis — inflammatory; EMG; biopsy; CK very high; steroid-responsive",
            "Pompe disease (GAA) — proximal + respiratory; enzyme assay",
            "Scapuloperoneal syndromes (EMD, EDMD) — contractures + cardiac; different genetics",
        ],
        "onset_range_y": (10, 40),
        "cardiac_risk": False,
        "dcm_risk": False,
        "steroid_responsive": False,
        "exon_skipping": False,
        "respiratory_failure": False,
        "cognitive_involvement": False,
        "contractures": False,
        "scoliosis_risk": True,
        "ck_range": (100, 500),
        "first_line_drug": "Losmapimod (trial) / physiotherapy / scapular fixation",
        "critical_avoid": "Missing Beevor sign (pathognomonic of abdominal asymmetry in FSHD); Coats retinal disease unmonitored",
    },
    # ── COL6A1 — Bethlem Myopathy / Ullrich CMD ──────────────────────────
    {
        "gene": "COL6A1", "protein": "Collagen Type VI Alpha-1 Chain (COL6A1)",
        "alias": "COL6A1; OMIM gene 120220; 21q22.3; 1028 aa; Bethlem Myopathy (OMIM #158810, AD mild) / Ullrich Congenital Muscular Dystrophy (OMIM #254090, AR severe); Collagen 6 spectrum disorder; COL6A1/COL6A2/COL6A3 trimer",
        "aa": "1028 aa", "kDa": "116 kDa",
        "mechanism": (
            "COL6A1 encodes the α1-chain of collagen VI (COL6), a microfibrillar collagen of the "
            "extracellular matrix (ECM). COL6 forms a homotetrameric triple helix (α1:α2:α3 chains) "
            "secreted into the endomysial ECM. COL6 bridges the sarcolemma to the ECM, "
            "and also localises to the mitochondria outer membrane, regulating autophagy. "
            "MECHANISM: COL6 deficiency → ECM instability → impaired sarcolemmal integrity + "
            "mitochondrial dysfunction → increased apoptosis (especially in congenital Ullrich form). "
            "GENOTYPE-PHENOTYPE: "
            "BETHLEM MYOPATHY: AD (dominant negative COL6A1/A2 mutations); mild/moderate; "
            "adult-onset or childhood; proximal weakness + distal joint contractures (finger flexor 'prayer sign'); "
            "life expectancy normal; <10% need wheelchair. "
            "ULLRICH CMD: AR (biallelic null mutations → no COL6 protein); severe; neonatal hypotonia; "
            "proximal contractures + distal hyperlaxity (PATHOGNOMONIC COMBINATION); "
            "loss of ambulation in childhood; respiratory failure (NIV by teens). "
            "PARADOX: PROXIMAL CONTRACTURES + DISTAL JOINT HYPERLAXITY = Ullrich CMD pathognomonic sign. "
            "BETHLEM FINGER CONTRACTURES: inability to extend finger interphalangeal joints with wrist extended "
            "('prayer sign') — early diagnostic clue in Bethlem."
        ),
        "disease_type": "Collagen VI Spectrum Disorder: Bethlem Myopathy (AD) / Ullrich CMD (AR)",
        "locus": "21q22.3", "omim_gene": 120220, "omim_disease": 158810,
        "inheritance": (
            "BETHLEM: AUTOSOMAL DOMINANT (de novo or familial). COL6A1, COL6A2, or COL6A3 mutations. "
            "ULLRICH: AUTOSOMAL RECESSIVE (biallelic null mutations in COL6A1, COL6A2, or COL6A3). "
            "Both Bethlem and Ullrich can result from mutations in any of the 3 COL6 genes. "
            "De novo dominant mutations common in Bethlem. "
            "COL6A1 is the most frequently mutated gene in collagen VI myopathy. "
            "Skin fibroblast COL6 western blot/IHC: absent in Ullrich; reduced or misfolded in Bethlem."
        ),
        "phenotype": (
            "BETHLEM MYOPATHY (AD, mild): "
            "Onset: childhood (often mild until 4th–5th decade). "
            "Weakness: proximal limb-girdle; usually slow progression; <10% require wheelchair. "
            "CONTRACTURES: finger flexors (prayer sign); elbows; Achilles — EARLY AND PROMINENT. "
            "Skin: follicular hyperkeratosis (keratosis pilaris-like); keloid scarring. "
            "Respiratory: usually mild; FVC monitoring from ~age 40. "
            "ULLRICH CMD (AR, severe): "
            "Neonatal: hypotonia; proximal contractures (hip, knee, elbow) + distal HYPERLAXITY "
            "(hyperextensible wrists, ankles — PATHOGNOMONIC). "
            "Motor: sit independently; some walk briefly (10–50%); loss of ambulation childhood-teens. "
            "Respiratory: NIV required by teenage years; respiratory failure is leading cause of death. "
            "Spine: PROGRESSIVE SCOLIOSIS requiring surgery. "
            "Skin: keratosis pilaris, keloid scarring. "
            "COL6 IHC: completely absent in Ullrich; reduced/normal in Bethlem."
        ),
        "treatment_options": [
            "Physiotherapy: contracture management CRITICAL — stretching, hydrotherapy; "
            "prevent fixed contractures that impair function",
            "NIV (BiPAP): MANDATORY in Ullrich CMD by teenage years (respiratory failure); "
            "annual FVC/spirometry; FVC <50% or symptoms = start NIV; also monitor in Bethlem from age 40",
            "Scoliosis surgery: Ullrich CMD — Cobb angle >25–30°; spinal fixation improves respiratory compliance",
            "Orthoses: AFOs for equinus/foot drop; wrist splints; resting splints",
            "Cyclosporin A: mitochondria-targeted; neuroprotective in animal Ullrich models; "
            "limited evidence in humans; compassionate use reported",
            "Deflazacort: some evidence in Ullrich CMD for ambulation preservation; trial data limited",
            "Occupational therapy: adaptive devices for contractures (splints, feeding aids)",
            "Genetic counselling: Bethlem (AD — 50% risk); Ullrich (AR — 25% recurrence)",
        ],
        "key_ddx": [
            "EDMD1/EDMD2 (EMD/LMNA) — contractures + cardiac arrhythmia; emerin IHC; different distribution",
            "Congenital fiber type disproportion (SELENON/RYR1) — rigid spine; no distal hyperlaxity",
            "Bethlem vs EDMD: Bethlem = finger flexor contractures + COL6; EDMD = elbow contractures + cardiac",
            "MDC1A (LAMA2) — merosin absent; white matter T2 signal; no distal hyperlaxity",
            "Nemaline myopathy (NEB/ACTA1) — neonatal hypotonia; biopsy rods; no contractures",
        ],
        "onset_range_y": (0, 30),
        "cardiac_risk": False,
        "dcm_risk": False,
        "steroid_responsive": False,
        "exon_skipping": False,
        "respiratory_failure": True,
        "cognitive_involvement": False,
        "contractures": True,
        "scoliosis_risk": True,
        "ck_range": (100, 800),
        "first_line_drug": "Physiotherapy (contractures) / NIV (respiratory)",
        "critical_avoid": "Missing respiratory monitoring in Ullrich CMD; unfixed scoliosis before respiratory deterioration",
    },
    # ── LAMA2 — MDC1A Merosin-Deficient Congenital MD ────────────────────
    {
        "gene": "LAMA2", "protein": "Laminin Alpha-2 (Merosin)",
        "alias": "LAMA2 (formerly LAMA2); OMIM gene 156225; 6q22.33; 3112 aa; 343 kDa; Merosin-Deficient Congenital Muscular Dystrophy type 1A MDC1A (OMIM #607855); AR; WHITE MATTER T2 SIGNAL UNIVERSAL; stable adult course",
        "aa": "3112 aa", "kDa": "343 kDa",
        "mechanism": (
            "LAMA2 encodes laminin α2 subunit (merosin), the α-chain of laminin-211 (α2β1γ1) — "
            "the major laminin isoform in skeletal muscle basement membrane. "
            "Laminin-211 is the primary extracellular ligand for α-dystroglycan (dystroglycanopathy pathway). "
            "MEROSIN FUNCTION: anchors myofibre basement membrane to ECM; "
            "connects α-dystroglycan (DAPC) to type IV collagen of ECM; "
            "essential for sarcolemmal stability, nerve myelination, and BBB integrity. "
            "MDC1A MECHANISM: biallelic LAMA2 null mutations → absent merosin → "
            "sarcolemmal fragility → progressive muscle fibre necrosis (similar to DMD pathway). "
            "WHITE MATTER: merosin is expressed in CNS white matter → "
            "T2 white matter signal abnormalities (UNIVERSAL in MDC1A, even in mildest cases). "
            "T2 white matter changes DO NOT CORRELATE WITH COGNITIVE FUNCTION — "
            "patients can have normal cognition with abnormal MRI. "
            "PARTIAL MEROSIN DEFICIENCY: hypomorphic alleles → PARTIAL merosin on IHC → "
            "milder phenotype (LGMD-like: later-onset limb-girdle weakness). "
            "DIAGNOSIS: merosin IHC on muscle biopsy (absent in complete MDC1A; "
            "reduced in partial) + molecular confirmation."
        ),
        "disease_type": "Merosin-Deficient Congenital Muscular Dystrophy Type 1A / MDC1A (AR)",
        "locus": "6q22.33", "omim_gene": 156225, "omim_disease": 607855,
        "inheritance": (
            "AUTOSOMAL RECESSIVE. Biallelic null/severe mutations → complete absence → MDC1A (severe). "
            "Partial null/missense alleles → reduced merosin → milder phenotype (LGMD-like in adults). "
            "Prevalence: MDC1A is the most common congenital muscular dystrophy in Northern Europe. "
            "DIAGNOSIS PATHWAY: neonatal hypotonia → muscle biopsy merosin IHC absent → "
            "brain MRI (T2 white matter signal, stable) → LAMA2 gene sequencing. "
            "Merosin IHC correlates with LAMA2 biallelic severity."
        ),
        "phenotype": (
            "CLASSIC MDC1A (complete absence): "
            "NEONATAL: severe hypotonia ('floppy infant'); respiratory insufficiency at birth; "
            "poor feeding requiring NGT/PEG. "
            "MOTOR DEVELOPMENT: delayed; many sit independently; most NEVER WALK (complete MDC1A). "
            "RESPIRATORY: NIV required in first decade; leading cause of death. "
            "COGNITIVE: usually NORMAL intelligence (white matter changes ≠ cognitive impairment). "
            "EPILEPSY: in ~30–40% (occipital focus; cortical dysplasia in some). "
            "CK: markedly elevated in infancy (5–50× ULN); falls with fibrosis. "
            "BRAIN MRI: T2 periventricular white matter signal — UNIVERSAL, STABLE, NOT PROGRESSIVE "
            "(critical: parents must be told white matter changes ≠ brain injury). "
            "PERIPHERAL NERVE: demyelinating peripheral neuropathy on NCS (merosin also expressed in PNS). "
            "ADULT COURSE: motor function generally STABLE after early childhood (wheelchair users live to "
            "adult life with respiratory support). "
            "PARTIAL MEROSIN DEFICIENCY: ambulatory; LGMD-like; normal MRI in some; milder."
        ),
        "treatment_options": [
            "NIV (BiPAP): respiratory support from first decade; annual lung function from diagnosis; "
            "FVC <50% or symptoms = start NIV; invasive ventilation if NIV fails",
            "PEG/NGT: nutritional support in neonates and when bulbar involvement; "
            "PEG preferred for long-term feeding in non-ambulant MDC1A",
            "Antiepileptics: seizure management (~30-40% have epilepsy); levetiracetam/sodium valproate",
            "Physiotherapy: hydrotherapy; prevent contractures (hips, knees, spine); "
            "postural management; adaptive seating",
            "Scoliosis: spine surveillance from early childhood; surgical fixation when Cobb >25°",
            "Gene therapy trials (ongoing): merosin delivery via AAV; mini-laminin α2 constructs; "
            "preclinical MDC1A models show efficacy — clinical trials emerging",
            "Explanation of MRI: parents counselled that T2 white matter changes are STABLE and "
            "NOT brain injury — cognitive prognosis is separate from MRI findings",
            "Genetic counselling: AR; 25% recurrence; merosin IHC is fast diagnostic screen",
        ],
        "key_ddx": [
            "Other CMD (RYR1, SELENON, COL6A1) — merosin PRESENT on IHC; different clinical pattern",
            "DMD — CK much higher; no white matter changes; males only; dystrophin absent (not merosin)",
            "SMA — no CK elevation; EMG denervation; SMN1 deletion",
            "Dystroglycanopathy (POMT1, FKRP) — α-DG glycosylation reduced not absent merosin; "
            "brain structural anomalies more common (cobblestone, PMG)",
            "Congenital hypomyelination — white matter changes but different pattern; nerve conduction",
        ],
        "onset_range_y": (0, 2),
        "cardiac_risk": False,
        "dcm_risk": False,
        "steroid_responsive": False,
        "exon_skipping": False,
        "respiratory_failure": True,
        "cognitive_involvement": False,
        "contractures": True,
        "scoliosis_risk": True,
        "ck_range": (1000, 10000),
        "first_line_drug": "NIV (respiratory) / antiepileptics (if seizures)",
        "critical_avoid": "Misinterpreting white matter MRI as brain injury; missed respiratory monitoring",
    },
    # ── POMT1 — Walker-Warburg Syndrome / Dystroglycanopathy ─────────────
    {
        "gene": "POMT1", "protein": "Protein O-Mannosyltransferase 1 (POMT1)",
        "alias": "POMT1; OMIM gene 607423; 9q34.13; 747 aa; Walker-Warburg Syndrome WWS (OMIM #236670); AR; dystroglycanopathy; cobblestone lissencephaly; cerebellar/pontine hypoplasia; FATAL first 3 years; eyes involved",
        "aa": "747 aa", "kDa": "84 kDa",
        "mechanism": (
            "POMT1 encodes protein O-mannosyltransferase 1, an enzyme in the ER that catalyses "
            "the first step of O-mannosylation of α-dystroglycan (α-DG). "
            "POMT1 and POMT2 form a heteromeric complex for O-mannosyltransferase activity. "
            "α-DYSTROGLYCAN O-MANNOSYLATION PATHWAY: "
            "POMT1/POMT2 → O-mannose on α-DG → POMGNT1 → LARGE1/LARGE2 → "
            "mature heavily glycosylated α-DG → binding to LAMA2, agrin, perlecan, neurexin. "
            "DYSTROGLYCANOPATHY MECHANISM: POMT1 mutations → absent/reduced O-mannosylation → "
            "hypoglycosylated α-DG → loss of α-DG–laminin binding → basement membrane instability "
            "in MUSCLE, BRAIN, AND EYE. "
            "BRAIN: normally α-DG anchors radial glial fibres at the basement membrane; "
            "without α-DG glycosylation → neuronalover-migration through disrupted glia limitans → "
            "COBBLESTONE LISSENCEPHALY (type II — bumpy, nodular cortical surface, opposite of type I smooth). "
            "CLINICAL SPECTRUM BY RESIDUAL GLYCOSYLATION: "
            "Walker-Warburg (WWS) — most severe; Muscle-Eye-Brain (MEB) — intermediate; "
            "FCMD (Fukutin) — Japan-specific; LGMD with brain involvement (FKRP) — milder. "
            "POMT1 mutations → PREDOMINANTLY WWS (most severe phenotype in the spectrum)."
        ),
        "disease_type": "Walker-Warburg Syndrome / Dystroglycanopathy (AR POMT1)",
        "locus": "9q34.13", "omim_gene": 607423, "omim_disease": 236670,
        "inheritance": (
            "AUTOSOMAL RECESSIVE. Biallelic POMT1 mutations → Walker-Warburg (most severe) "
            "or Muscle-Eye-Brain disease. "
            "Consanguinity increases risk. Prevalence 1.2–4 per 100,000 births (WWS all causes). "
            "POMT1 accounts for ~20% of WWS cases. Other WWS genes: POMT2, POMGNT1, LARGE1, FKRP, FKTN. "
            "Reduced α-DG glycosylation on muscle/brain IHC is a diagnostic biomarker "
            "(IIH6 antibody for glycosylated α-DG — ABSENT in WWS). "
            "Muscle enzyme panel + brain MRI before molecular → NARROWS gene panel."
        ),
        "phenotype": (
            "WALKER-WARBURG SYNDROME (WWS) — MOST SEVERE DYSTROGLYCANOPATHY: "
            "BRAIN: cobblestone lissencephaly type II (agyria/pachygyria with nodular cortex on MRI); "
            "cerebellar and pontine hypoplasia/agenesis; corpus callosum absent; hydrocephalus. "
            "EYES: OCULAR STRUCTURAL ABNORMALITIES — microphthalmia; coloboma; Peters anomaly; "
            "retinal dysplasia; anterior segment malformations. "
            "MUSCLE: congenital muscular dystrophy — profound hypotonia at birth; "
            "markedly elevated CK. "
            "EPILEPSY: virtually universal; often intractable. "
            "PROGNOSIS: FATAL — most patients die before age 3 years (median ~9 months for WWS); "
            "no children with full WWS survive beyond childhood. "
            "MUSCLE-EYE-BRAIN (MEB) — intermediate: some ambulation; less severe brain; "
            "longer survival (late childhood); cognitive impairment. "
            "DIAGNOSIS: brain MRI (cobblestone lissencephaly pathognomonic) → "
            "muscle biopsy (IIH6 absent) → gene panel (POMT1, POMT2, POMGNT1, LARGE1, etc.)."
        ),
        "treatment_options": [
            "Palliative/comfort care: WWS is FATAL within years; goals of care discussion EARLY and MANDATORY; "
            "quality of life focus; multidisciplinary team (neurology, ophthalmology, palliative care)",
            "Antiepileptics: seizure control is a priority for comfort; valproate, levetiracetam, "
            "clonazepam; refractory epilepsy common — ketogenic diet consideration",
            "NIV/mechanical ventilation: respiratory support per family goals; tracheostomy considered "
            "only if consistent with family's goals of care",
            "PEG (percutaneous endoscopic gastrostomy): nutritional support; NGT initially",
            "Ophthalmology: structural anomalies; glaucoma monitoring; "
            "spectacles if any visual acuity; retinal assessment",
            "Physiotherapy/OT: supportive; comfort-based range of motion; positioning",
            "Genetic counselling: AR; 25% recurrence; prenatal diagnosis by CVS/amniocentesis; "
            "PGT-M (preimplantation genetic testing-monogenic) available; "
            "brain MRI at 20 weeks gestation detects cobblestone lissencephaly prenatally",
            "Ribose 5-phosphate supplementation: experimental; mannose supplementation limited evidence; "
            "no approved disease-modifying therapy currently exists for WWS",
        ],
        "key_ddx": [
            "MDC1A (LAMA2) — merosin absent; WHITE MATTER (not cobblestone lissencephaly); longer survival",
            "Fukuyama CMD (FKTN) — Japan; cobblestone lissencephaly; milder than WWS; survival to teens",
            "MEB disease (POMGNT1) — intermediate; eye involvement + cerebellar hypoplasia; longer survival",
            "FKRP LGMD R9 — mostly LGMD; brain structural anomalies rare; cardiac involvement",
            "Lissencephaly type 1 (LIS1/YWHAE) — SMOOTH cortex (no cobblestones); different mechanism",
        ],
        "onset_range_y": (0, 1),
        "cardiac_risk": False,
        "dcm_risk": False,
        "steroid_responsive": False,
        "exon_skipping": False,
        "respiratory_failure": True,
        "cognitive_involvement": True,
        "contractures": True,
        "scoliosis_risk": False,
        "ck_range": (500, 5000),
        "first_line_drug": "Palliative (antiepileptics + NIV by goals of care)",
        "critical_avoid": "Aggressive intervention without goals-of-care discussion; false reassurance on prognosis",
    },
]


def _gen_patients(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    patients = []
    ck_lo, ck_hi = gene_data["ck_range"]
    onset_lo, onset_hi = gene_data["onset_range_y"]

    for i in range(40):
        onset = round(rng.uniform(onset_lo, onset_hi), 1)
        ck_val = round(rng.uniform(ck_lo, ck_hi))

        r = rng.random()
        if gene == "DMD":
            sev = "Severe" if r < 0.55 else ("Moderate" if r < 0.80 else "Mild")
        elif gene == "DMPK":
            sev = "Moderate" if r < 0.50 else ("Mild" if r < 0.75 else "Severe")
        elif gene == "CNBP":
            sev = "Mild" if r < 0.55 else ("Moderate" if r < 0.85 else "Severe")
        elif gene == "EMD":
            sev = "Moderate" if r < 0.45 else ("Severe" if r < 0.72 else "Mild")
        elif gene == "SMCHD1":
            sev = "Mild" if r < 0.50 else ("Moderate" if r < 0.82 else "Severe")
        elif gene == "COL6A1":
            sev = "Mild" if r < 0.42 else ("Moderate" if r < 0.75 else "Severe")
        elif gene == "LAMA2":
            sev = "Severe" if r < 0.60 else ("Moderate" if r < 0.85 else "Mild")
        else:  # POMT1
            sev = "Severe"  # WWS — uniformly fatal

        cardiac_ev  = gene_data["cardiac_risk"] and rng.random() < 0.55
        dcm         = gene_data["dcm_risk"] and rng.random() < 0.50
        icd         = cardiac_ev and gene in ("EMD", "DMPK") and rng.random() < 0.50
        steroid_rx  = gene_data["steroid_responsive"] and rng.random() < 0.92
        exon_skip   = gene_data["exon_skipping"] and rng.random() < 0.40
        niv         = gene_data["respiratory_failure"] and rng.random() < (
            0.85 if gene in ("LAMA2", "POMT1") else 0.50 if gene == "DMPK" else 0.70
        )
        contracture = gene_data["contractures"] and rng.random() < 0.78
        scoliosis   = gene_data["scoliosis_risk"] and rng.random() < 0.55
        cognitive   = gene_data["cognitive_involvement"] and rng.random() < (
            0.95 if gene == "POMT1" else 0.70 if gene == "DMD" else 0.55
        )
        ambulant    = gene not in ("POMT1", "LAMA2") or (gene == "LAMA2" and sev == "Mild" and rng.random() < 0.30)

        # Treatment
        if gene == "DMD":
            tx = "Deflazacort" + ("; exon-skip ASO" if exon_skip else "") + ("; NIV" if niv else "")
        elif gene == "DMPK":
            tx = "Mexiletine + cardiac monitoring" + ("; pacemaker/ICD" if icd else "")
        elif gene == "CNBP":
            tx = "Mexiletine (if myotonia); pain management"
        elif gene == "EMD":
            tx = "Pacemaker/ICD (cardiac)" + ("; cardiac monitoring" if not icd else "")
        elif gene == "SMCHD1":
            tx = "Physiotherapy + scapular fixation"
        elif gene == "COL6A1":
            tx = "Physiotherapy" + ("; NIV" if niv else "") + ("; scoliosis surgery" if scoliosis else "")
        elif gene == "LAMA2":
            tx = "NIV + anti-epileptics" + ("; PEG" if not ambulant else "")
        else:  # POMT1
            tx = "Palliative care + antiepileptics"

        sex = "M" if (gene in ("DMD", "EMD") and rng.random() < 0.95) else rng.choice(["M", "F"])
        age_at_dx = min(onset + rng.uniform(1, 5), onset_hi + 5)

        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "onset_age_y": onset,
            "diagnosis_age_y": round(age_at_dx, 1),
            "sex": sex,
            "severity": sev,
            "ck_iu_l": ck_val,
            "cardiac_involvement": cardiac_ev,
            "dcm": dcm,
            "icd_implanted": icd,
            "on_steroids": steroid_rx,
            "exon_skipping_therapy": exon_skip,
            "niv": niv,
            "contractures": contracture,
            "scoliosis": scoliosis,
            "cognitive_involvement": cognitive,
            "ambulant": ambulant,
            "treatment": tx,
            "first_line_drug": gene_data["first_line_drug"],
            "critical_avoid": gene_data["critical_avoid"],
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gene_data in enumerate(MD_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients(gene_data, seed))
    return all_pts


def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] += 1

    cardiac_n   = sum(1 for p in patients if p["cardiac_involvement"])
    dcm_n       = sum(1 for p in patients if p["dcm"])
    icd_n       = sum(1 for p in patients if p["icd_implanted"])
    steroid_n   = sum(1 for p in patients if p["on_steroids"])
    exon_n      = sum(1 for p in patients if p["exon_skipping_therapy"])
    niv_n       = sum(1 for p in patients if p["niv"])
    contract_n  = sum(1 for p in patients if p["contractures"])
    scolio_n    = sum(1 for p in patients if p["scoliosis"])
    cog_n       = sum(1 for p in patients if p["cognitive_involvement"])
    ambulant_n  = sum(1 for p in patients if p["ambulant"])

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 1)
    mean_ck = round(sum(p["ck_iu_l"] for p in patients) / n)

    return {
        "atlas": "MD-Atlas",
        "full_name": "Complete 8-Gene Muscular Dystrophy Atlas",
        "subtitle": "DMD·DMPK·CNBP·EMD·SMCHD1·COL6A1·LAMA2·POMT1 — 320 patients (8×40, seeds 1062–1069)",
        "description": (
            "Comprehensive atlas of 8 major genetic muscular dystrophies not covered in LGMD or Congenital "
            "Myopathy atlases. Encompasses: DUCHENNE/BECKER (DMD — exon-skipping, deflazacort, gene therapy); "
            "MYOTONIC DYSTROPHY type 1 (DMPK — CTG repeat, RNA toxicity, cardiac monitoring); "
            "MYOTONIC DYSTROPHY type 2 (CNBP — CCTG repeat, proximal weakness, milder); "
            "EMERY-DREIFUSS type 1 (EMD — nuclear envelope, LETHAL arrhythmia, ICD mandatory); "
            "FSHD type 2 (SMCHD1 — DUX4 derepression, digenic, Beevor sign pathognomonic); "
            "COLLAGEN VI spectrum (COL6A1 — Bethlem mild AD / Ullrich CMD severe AR); "
            "MDC1A (LAMA2 — merosin absent, white matter T2 universal, cognitive normal); "
            "WALKER-WARBURG (POMT1 — cobblestone lissencephaly, fatal, eyes involved). "
            "CRITICAL RULES: DMD deflazacort MANDATORY; DM1 cardiac monitoring life-saving; "
            "EMD ICD MANDATORY (atrial standstill pathognomonic); FSHD Beevor sign pathognomonic; "
            "MDC1A white matter ≠ cognitive injury (counsel parents); "
            "POMT1 Walker-Warburg = palliative from diagnosis."
        ),
        "total_patients": n,
        "genes_covered": len(MD_GENES),
        "patients_per_gene": 40,
        "seed_range": "1062–1069",
        "gene_list": [g["gene"] for g in MD_GENES],
        "disease_category_breakdown": {
            "Dystrophinopathy / Duchenne-Becker MD (XLR — deflazacort + exon-skip + gene therapy)": ["DMD"],
            "Myotonic Dystrophy type 1 (AD CTG repeat — multisystem; cardiac pacemaker/ICD)": ["DMPK"],
            "Myotonic Dystrophy type 2 (AD CCTG repeat — proximal weakness; no congenital form)": ["CNBP"],
            "Emery-Dreifuss MD type 1 (XLR emerin — LETHAL arrhythmia; ICD MANDATORY)": ["EMD"],
            "FSHD type 2 (Digenic — SMCHD1 + 4qA allele; DUX4 derepression; Beevor sign)": ["SMCHD1"],
            "Collagen VI spectrum (COL6A1 — Bethlem AD mild / Ullrich CMD AR severe)": ["COL6A1"],
            "MDC1A merosin-deficient CMD (AR LAMA2 — white matter T2; normal cognition)": ["LAMA2"],
            "Walker-Warburg / Dystroglycanopathy (AR POMT1 — cobblestone; fatal; palliative)": ["POMT1"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "mean_ck_iu_l": mean_ck,
        "kpis": [
            {"label": "Total Patients", "value": n, "color": "#1565c0"},
            {"label": "Genes Covered", "value": len(MD_GENES), "color": "#2e7d32"},
            {"label": "Patients/Gene", "value": 40, "color": "#6a1b9a"},
            {"label": "Cardiac Risk Genes", "value": 3, "color": "#b71c1c"},
            {"label": "Mean Onset (y)", "value": mean_onset, "color": "#e65100"},
            {"label": "Seeds", "value": "1062–1069", "color": "#37474f"},
        ],
        "clinical_features_prevalence": {
            "On Corticosteroids (DMD)": round(100 * steroid_n / n, 1),
            "Exon-Skipping Therapy (DMD)": round(100 * exon_n / n, 1),
            "Non-Invasive Ventilation (NIV)": round(100 * niv_n / n, 1),
            "Cardiac Involvement": round(100 * cardiac_n / n, 1),
            "DCM (Dilated Cardiomyopathy)": round(100 * dcm_n / n, 1),
            "ICD Implanted": round(100 * icd_n / n, 1),
            "Contractures": round(100 * contract_n / n, 1),
            "Scoliosis": round(100 * scolio_n / n, 1),
            "Cognitive Involvement": round(100 * cog_n / n, 1),
            "Ambulant": round(100 * ambulant_n / n, 1),
        },
        "drug_alerts": [
            "DMD MANDATORY: deflazacort 0.9 mg/kg/day (superior to prednisolone; less weight gain); "
            "ACE inhibitor + beta-blocker from age 10 regardless of echo",
            "DM1 (DMPK): AVOID class Ia antiarrhythmics (quinidine/procainamide/disopyramide) and class Ic "
            "(flecainide/propafenone) — pro-arrhythmic + worsen myotonia in DM1",
            "EMD (EDMD1): ICD MANDATORY when ventricular arrhythmia present; atrial standstill "
            "(flat baseline on ECG, no P-waves) is PATHOGNOMONIC — pacemaker URGENTLY required",
            "SUCCINYLCHOLINE ABSOLUTELY CONTRAINDICATED in DMD/BMD — life-threatening hyperkalaemia "
            "and rhabdomyolysis; use rocuronium + sugammadex instead",
            "POMT1 Walker-Warburg: PALLIATIVE from diagnosis — median survival <3 years; "
            "goals-of-care discussion MANDATORY at first visit",
            "MDC1A white matter T2 signal DOES NOT indicate brain injury — "
            "counsel parents that cognitive prognosis is INDEPENDENT of MRI white matter changes",
        ],
        "diagnostic_pearls": [
            "DMD: Gowers sign + CK 10,000–50,000 IU/L + X-linked → immediate dystrophin IHC + gene panel",
            "DM1: DISTAL weakness + MYOTONIA + MULTISYSTEM (cataracts + cardiac + cognitive + GI) → CTG repeat",
            "DM2: PROXIMAL weakness + myotonia + PAIN + NO congenital form + CK elevated → CCTG repeat",
            "EMD: TRIAD = early ELBOW CONTRACTURES + scapuloperoneal weakness + CARDIAC ARRHYTHMIA → emerin IHC absent",
            "FSHD: BEEVOR SIGN (umbilicus rises on neck flexion) + facial weakness + ASYMMETRY → D4Z4 sizing + SMCHD1",
            "MDC1A: neonatal CMD + WHITE MATTER T2 (universal) + NORMAL COGNITION → merosin IHC absent → LAMA2",
            "Walker-Warburg: COBBLESTONE LISSENCEPHALY + eye anomalies + CMD → IIH6 absent → POMT1 panel",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    breakdown = {}
    for gene_data in MD_GENES:
        gene = gene_data["gene"]
        gene_pts = [p for p in patients if p["gene"] == gene]
        n = len(gene_pts)
        sev = {s: sum(1 for p in gene_pts if p["severity"] == s) for s in ("Mild", "Moderate", "Severe")}
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
            "first_line_drug": gene_data["first_line_drug"],
            "critical_avoid": gene_data["critical_avoid"],
            "severity_distribution": {
                "mild_pct": round(100 * sev["Mild"] / n, 1),
                "moderate_pct": round(100 * sev["Moderate"] / n, 1),
                "severe_pct": round(100 * sev["Severe"] / n, 1),
            },
            "mean_onset_age_y": round(sum(p["onset_age_y"] for p in gene_pts) / n, 1),
            "mean_ck_iu_l": round(sum(p["ck_iu_l"] for p in gene_pts) / n),
            "cardiac_pct": round(100 * sum(1 for p in gene_pts if p["cardiac_involvement"]) / n, 1),
            "niv_pct": round(100 * sum(1 for p in gene_pts if p["niv"]) / n, 1),
            "contracture_pct": round(100 * sum(1 for p in gene_pts if p["contractures"]) / n, 1),
            "ambulant_pct": round(100 * sum(1 for p in gene_pts if p["ambulant"]) / n, 1),
        }
    return {
        "atlas": "MD-Atlas",
        "subtitle": "Per-gene clinical breakdown — 320 patients (8×40, seeds 1062–1069)",
        "genes": breakdown,
        "gene_order": [g["gene"] for g in MD_GENES],
    }


def get_definitions() -> dict:
    return {
        "atlas": "MD-Atlas",
        "subtitle": "Clinical and genetic terminology definitions for MD-Atlas",
        "definitions": {
            "Dystrophinopathy": "Disease caused by mutations in the DMD gene (dystrophin); includes Duchenne MD (out-of-frame), Becker MD (in-frame), X-linked dilated cardiomyopathy, and DMD-related myalgia.",
            "Reading-Frame Rule": "DMD out-of-frame mutations → no functional dystrophin → Duchenne (severe). In-frame mutations → truncated dystrophin → Becker (mild). Key predictor of severity but exceptions exist.",
            "Exon-Skipping": "Antisense oligonucleotide (ASO) therapy that skips a specific exon during RNA splicing, converting an out-of-frame to an in-frame reading frame → truncated 'Becker-like' dystrophin. Each ASO targets a specific exon.",
            "Deflazacort": "Corticosteroid (oxazoline derivative of prednisolone) — first-line for DMD; superior to prednisolone in preservation of ambulation with less weight gain. Dose: 0.9 mg/kg/day.",
            "Gowers Sign": "Manoeuvre where a child uses hands to push off thighs when rising from floor, compensating for proximal hip-extensor weakness. Classic DMD sign from ~age 3-5.",
            "CTG Repeat (DM1)": "Trinucleotide CTG repeat expansion in 3'UTR of DMPK. Normal <37; DM1 >50 (mild 50-150, classic 100-1000, congenital >1000). Shows ANTICIPATION — expands and worsens in successive generations.",
            "CCTG Repeat (DM2)": "Tetranucleotide CCTG repeat expansion in intron 1 of CNBP. Normal <26; DM2 75-11,000. NO congenital form exists. Shows mild anticipation. Causes proximal (not distal) weakness.",
            "RNA Toxicity": "Disease mechanism in myotonic dystrophies where the expanded CUG/CCUG repeat RNA sequesters MBNL1 splicing factors in nuclear foci, causing widespread splicing dysregulation of multiple targets.",
            "MBNL1": "Muscleblind-Like Splicing Regulator 1 — sequestered by expanded CUG/CCUG RNA in DM1/DM2, leading to loss of splicing regulation for ClC-1, INSR, TNNT2, MAPT, and other critical transcripts.",
            "Anticipation": "Worsening of disease phenotype in successive generations due to trinucleotide/tetranucleotide repeat expansion. Classic in DM1 (CTG expansion). Congenital DM1 almost exclusively from maternal transmission.",
            "LEM Domain": "Protein domain in Lap2, Emerin, and MAN1 that binds BAF (barrier-to-autointegration factor). Emerin (EMD gene) is a LEM-domain protein of the inner nuclear membrane.",
            "Atrial Standstill": "ECG finding pathognomonic of Emery-Dreifuss MD — absent P-waves on ECG due to complete atrial paralysis/fibrosis. Requires URGENT pacemaker. Emerin and LMNA mutations.",
            "D4Z4 Repeat": "Macrosatellite repeat unit at 4q35 harbouring DUX4. Normal ≥11 units. FSHD1: <11 units on 4qA haplotype. FSHD2: D4Z4 methylation <20% due to SMCHD1 mutations. Both → DUX4 derepression.",
            "DUX4": "Double Homeobox 4 — transcription factor normally silenced in somatic cells. Derepressed in FSHD → activates germline/embryonic genes → muscle cell death. Common endpoint for FSHD1 and FSHD2.",
            "Beevor Sign": "Pathognomonic of FSHD — umbilicus moves UPWARD on neck flexion (chin to chest) due to asymmetric rectus abdominis weakness (lower > upper). Positive Beevor sign = FSHD until proven otherwise.",
            "Bethlem Myopathy": "Autosomal dominant COL6 spectrum disorder — mild/moderate; adult or childhood onset; proximal weakness + DISTAL CONTRACTURES (finger flexors — 'prayer sign'); life expectancy normal.",
            "Ullrich CMD": "Autosomal recessive COL6 spectrum disorder — severe; neonatal hypotonia; PROXIMAL CONTRACTURES + DISTAL HYPERLAXITY (pathognomonic); respiratory failure in teens.",
            "Merosin (Laminin α2)": "α2 subunit of laminin-211, the major basement membrane protein of skeletal muscle. Absent in MDC1A (LAMA2 biallelic null). Merosin IHC on muscle biopsy: diagnostic screen.",
            "MDC1A": "Merosin-Deficient Congenital Muscular Dystrophy Type 1A — most common CMD in Northern Europe. LAMA2 biallelic mutations → absent merosin → CMD + universal white matter T2 MRI changes + normal cognition.",
            "Cobblestone Lissencephaly": "Type II lissencephaly — bumpy, nodular cortical surface on MRI (opposite of type I smooth). Pathognomonic of dystroglycanopathies (POMT1, POMT2, POMGNT1, LARGE1, FKRP, FKTN). Caused by neuronal over-migration.",
            "α-Dystroglycan (α-DG)": "Cell-surface receptor that links extracellular matrix (laminin) to the DAPC (dystrophin-associated protein complex). Requires O-mannosylation by POMT1/POMT2 for laminin binding. Hypoglycosylated in dystroglycanopathies.",
            "IIH6 Antibody": "Monoclonal antibody that recognises the glycosylated (functional) form of α-dystroglycan. IIH6 staining absent on muscle biopsy IHC = dystroglycanopathy (POMT1, POMT2, POMGNT1, etc.).",
            "Walker-Warburg Syndrome (WWS)": "Most severe dystroglycanopathy — cobblestone lissencephaly + ocular structural anomalies + CMD. POMT1 is the most common gene. Median survival <3 years. Fatal.",
            "Dystroglycanopathy": "Group of muscular dystrophies caused by deficient O-mannosylation of α-dystroglycan. Includes WWS (most severe), MEB, FCMD, LGMD-R9 (FKRP), LGMD-R11 (ANO5). Gene panel required.",
            "LINC Complex": "Linker of Nucleoskeleton and Cytoskeleton — includes emerin, lamin A/C, nesprin-1, SUN1/2, BAF. Connects nuclear lamina to actin cytoskeleton. Disrupted in EDMD (EMD, LMNA mutations).",
            "Losmapimod": "p38α/β MAP kinase inhibitor in phase 3 clinical trial for FSHD (REACH trial). Targets DUX4-induced inflammatory signalling in muscle. Most advanced disease-modifying therapy for FSHD.",
            "Delandistrogene Moxeparvovec (Elevidys)": "AAV-rh74-MHCK7-micro-dystrophin gene therapy for DMD; FDA accelerated approval 2023 for age 4-5; delivers functional micro-dystrophin to skeletal and cardiac muscle.",
        }
    }


if __name__ == "__main__":
    import json
    print("=== MD-Atlas Overview ===")
    ov = get_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Full name: {ov['full_name']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Genes: {', '.join(ov['gene_list'])}")
    print(f"Mean onset: {ov['mean_onset_age_y']} y")
    print(f"Mean CK: {ov['mean_ck_iu_l']} IU/L")
    print(f"Severity: mild={ov['severity']['mild_pct']}% mod={ov['severity']['moderate_pct']}% severe={ov['severity']['severe_pct']}%")
    print("Clinical features (%):")
    for k, v in ov['clinical_features_prevalence'].items():
        print(f"  {k}: {v}%")
    print("Drug alerts:")
    for a in ov['drug_alerts']:
        print(f"  - {a[:80]}...")
    bk = get_breakdown()
    print(f"\n=== Breakdown: {len(bk['genes'])} genes ===")
    for g, data in bk['genes'].items():
        print(f"  {g}: mean onset {data['mean_onset_age_y']}y, mean CK {data['mean_ck_iu_l']}, "
              f"severity mild={data['severity_distribution']['mild_pct']}% severe={data['severity_distribution']['severe_pct']}%")
    df = get_definitions()
    print(f"\n=== Definitions: {len(df['definitions'])} terms ===")
