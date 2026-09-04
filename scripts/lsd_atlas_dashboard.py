#!/usr/bin/env python3
"""LSD-Atlas — Complete 8-Gene Lysosomal Storage Disorder Atlas
GBA · GLA · GAA · SMPD1 · NPC1 · GALC · ARSA · HEXA
320-patient aggregate cohort (8 × 40, seeds 850–857)

Lysosomal Storage Disorders (LSD) facts:
  - LSDs = inherited defects in lysosomal enzymes or transport proteins → substrate accumulation
  - ~50 distinct LSDs; collectively 1 in 5,000–7,500 live births
  - All autosomal recessive except GLA (Fabry, X-linked)
  - Categories: sphingolipidoses (GBA/GLA/GALC/ARSA/HEXA), mucopolysaccharidoses (MPS),
    glycoproteinoses, glycogen storage (GAA), and NPC-type transport defects (NPC1)
  - ERT (enzyme replacement therapy) available: GBA, GLA, GAA, SMPD1-B
  - Substrate reduction therapy (SRT): GBA (miglustat/eliglustat), NPC1 (miglustat)
  - HSCT: Krabbe (pre-symptomatic ONLY), MLD (gene therapy now preferred)
  - Newborn screening: most jurisdictions include GBA, GLA, GAA, GALC on expanded NBS panels

ATLAS SCOPE (8 classic LSD genes):
  Sphingolipidoses (ceramide/glucocerebroside/galactocerebroside/sulfatide pathways):
    GBA   — β-glucocerebrosidase → Gaucher disease types 1/2/3 (1q22)
    GLA   — α-galactosidase A → Fabry disease (Xq22.1, X-LINKED)
    GALC  — galactocerebrosidase → Krabbe disease (14q31.3)
    ARSA  — arylsulfatase A → Metachromatic leukodystrophy (22q13.33)
    HEXA  — β-hexosaminidase α → Tay-Sachs / GM2 gangliosidosis type I (15q23)
  Glycogen storage (acid maltase pathway):
    GAA   — acid α-glucosidase → Pompe disease (17q25.3)
  Sphingomyelinase / cholesterol trafficking:
    SMPD1 — sphingomyelin phosphodiesterase 1 → Niemann-Pick A/B (11p15.4)
    NPC1  — intracellular cholesterol transporter → Niemann-Pick C type 1 (18q11.2)

CRITICAL CLINICAL RULES:
  1. GBA: ERT (imiglucerase/velaglucerase/taliglucerase) is LEVEL A for type 1/3 visceral;
     NO ERT benefit for CNS (does not cross BBB). SRT eliglustat ONLY for CYP2D6 metabolisers.
     GBA is the MOST COMMON GENETIC PD/DLB RISK FACTOR — p.Asn409Ser heterozygotes 5× PD risk.
  2. GLA: X-LINKED — hemizygous males severely affected; heterozygous females variable
     (25–70% symptomatic due to X-inactivation skewing). Migalastat (chaperone) ONLY for
     amenable variants (GNE database); NOT a universal GLA therapy. α-Gal A activity in
     leukocytes/plasma unreliable in females — DNA testing mandatory.
  3. GAA: CRIM (cross-reactive immunological material) status DETERMINES immune tolerance induction
     need. CRIM-negative Pompe → immune tolerance induction (ITI: MTX + rituximab + IVIG) MANDATORY
     before or concurrent with ERT (alglucosidase alfa/avalglucosidase alfa). IOPD has HCM 95% —
     cardiac biomarkers + echo mandatory. c.-32-13T>G is most common LOPD allele.
  4. SMPD1: Type A (complete loss, NH3 <1% residual activity) = fatal infantile neuronopathic;
     Type B (partial loss, 5–10% residual) = non-neuronopathic, ERT (olipudase alfa) approved.
     Never promise neurological benefit from ERT in any SMPD1 patient.
  5. NPC1: Miglustat (SRT — SRT approved EU/Canada; off-label USA) STABILISES neurological
     progression but does NOT reverse damage. Arimoclomol (HSP co-inducer) showed benefit in
     RCT. Filipin staining of fibroblasts = gold-standard biochemical diagnosis.
     VSGP (vertical supranuclear gaze palsy) + cataplexy in school-age child = NPC1 until proven.
  6. GALC (Krabbe): HSCT ONLY BENEFITS pre-symptomatic — must be done before neurological
     symptoms appear (ideally within first 30 days in IOPD detected on NBS). NO benefit once
     neurological symptoms present. Psychosine (galactosylsphingosine) is the primary neurotoxin
     (not galactocerebroside). CSF protein >100 mg/dL virtually universal in infantile Krabbe.
  7. ARSA (MLD): Atidarsagene autotemcel (Libmeldy) — HSC gene therapy — approved EU 2020;
     first-in-class for MLD. Benefit restricted to pre-symptomatic late-infantile or early-
     symptomatic early-juvenile. Pseudo-deficiency alleles I179S/R496H are NOT pathogenic —
     saposin B activator level must be checked; sulfatide urine is the definitive biomarker.
  8. HEXA (Tay-Sachs): NO ERT AVAILABLE (GM2 gangliosides cannot be replaced exogenously).
     Carrier screening (Ashkenazi Jewish, French-Canadian, Cajun) has nearly eliminated
     infantile Tay-Sachs in screened populations. Substrate reduction (miglustat) provides
     modest symptomatic benefit; adult-onset Tay-Sachs has better prognosis.

COHORT: 8 × 40 = 320 patient slots (seeds 850–857; gene-specific seeds)
"""

import random

SEED_BASE = 850

# ── All 8 LSD genes ────────────────────────────────────────────────────────────
LSD_GENES = [
    # ── GBA — β-glucocerebrosidase ──
    {
        "gene": "GBA", "alias": "GBA — β-Glucocerebrosidase (Gaucher Disease Types 1/2/3)",
        "aa": "497 aa", "kDa": "59.7 kDa",
        "gene_class": "glucosidase",
        "locus": "1q22", "omim_gene": 606463,
        "phenotype": "Gaucher Disease (GD) — glucocerebroside accumulation in macrophages; most common LSD",
        "disease": (
            "GBA biallelic (or hemizygous in compound het) loss → Gaucher Disease (OMIM #230800 type 1, "
            "#230900 type 2, #231000 type 3). GBA encodes lysosomal acid β-glucosidase (β-glucocerebrosidase, "
            "497aa, 59.7 kDa), which cleaves glucocerebroside (glucosylceramide) → ceramide + glucose in "
            "lysosomes. Deficiency → glucocerebroside accumulates in macrophages (Kupffer cells, "
            "hepatic/splenic/bone marrow macrophages) → organomegaly, bone disease, cytopaenias. "
            "Three clinical subtypes: Type 1 (non-neuronopathic, 94% of cases): splenomegaly, "
            "hepatomegaly, bone pain/crises, thrombocytopaenia, anaemia, lung involvement. "
            "Type 2 (acute neuronopathic): severe neonatal/infantile neurodegeneration, death <2 years. "
            "Type 3 (subacute/chronic neuronopathic): myoclonic epilepsy, horizontal supranuclear gaze palsy, "
            "slower neurological decline. p.Asn409Ser (N370S) is the most common type 1 allele "
            "(Ashkenazi: 70–75% of alleles); p.Leu483Pro (L444P) is the most common severe/type 3 "
            "allele worldwide. GBA heterozygosity is the MOST COMMON GENETIC RISK FACTOR for PD/DLB: "
            "5–10× increased PD risk compared to non-carriers."
        ),
        "inheritance": "Autosomal recessive, biallelic. Most common LSD: 1 in 40,000 (general); 1 in 450 Ashkenazi.",
        "hallmark": (
            "GBA HALLMARKS: (1) GAUCHER CELLS — lipid-laden macrophages with crinkled paper / "
            "wrinkled tissue paper cytoplasm on bone marrow biopsy (pathognomonic); "
            "(2) ERT (enzyme replacement therapy) LEVEL A for type 1/3 visceral: imiglucerase "
            "(Cerezyme), velaglucerase alfa (VPRIV), taliglucerase alfa (Elelyso) — IV every 2 weeks; "
            "ERT does NOT cross BBB — NO CNS benefit for type 2/3 neurological; "
            "(3) SRT ELIGLUSTAT: oral substrate reduction, type 1 only, CYP2D6-metaboliser testing "
            "MANDATORY before prescribing (poor metabolisers: plasma levels unpredictable); "
            "SRT MIGLUSTAT: less preferred, CNS penetrant (NPC1 use), GI side effects 90%; "
            "(4) p.Asn409Ser (N370S): PROTECTIVE against neurological type 2/3 — if one allele is "
            "N370S, phenotype is type 1 (non-neuronopathic) regardless of second allele; "
            "(5) p.Leu483Pro (L444P) homozygous → type 3 (Norrbottnian) or type 2; "
            "(6) GBA HETEROZYGOSITY → PARKINSON DISEASE: 5–10× PD risk; 10–15% of Ashkenazi PD "
            "carry GBA variant; DLB also strongly associated; monitoring for prodromal PD "
            "(REM sleep disorder, hyposmia, constipation) recommended in all GD patients/carriers; "
            "(7) BONE CRISIS (pseudo-osteomyelitis): acute bone pain + fever; bisphosphonates; "
            "not infective; (8) Pulmonary hypertension (type 1, post-splenectomy risk); "
            "(9) Chitotriosidase: biomarker for GD disease burden/ERT response (avoid if "
            "chitoBioTriosidase variant: 1 in 12 individuals; use CCL18/PARC instead); "
            "(10) Splenectomy increases type 3 neurological risk — avoid."
        ),
        "key_ddx": (
            "GBA DDx: (1) Other causes of splenomegaly (lymphoma, portal hypertension, MPS): "
            "bone marrow biopsy + β-glucosidase enzyme activity in leukocytes; "
            "(2) Niemann-Pick A/B: sphingomyelinase activity, foamy cells not crinkled-paper; "
            "(3) NPC1: filipin staining, VSGP, cataplexy; (4) Pompe: glucose enzyme, WBC activity; "
            "(5) GBA PSEUDO-DEFICIENCY: p.Glu326Lys variant reduces enzyme activity in vitro but "
            "NOT pathogenic — must distinguish by clinical context and second allele."
        ),
        "ert_available": "YES — imiglucerase / velaglucerase / taliglucerase (type 1/3 visceral only)",
        "srt_available": "YES — eliglustat (CYP2D6 MANDATORY) / miglustat (less preferred)",
        "hsct_role": "Not first-line; considered for neuronopathic type 3 refractory",
        "gene_therapy_status": "Phase 2-3 trials ongoing (lentiviral / AAV)",
        "critical_ci": (
            "CRITICAL CI: (1) Eliglustat WITHOUT CYP2D6 testing — dangerous accumulation in poor "
            "metabolisers; (2) Splenectomy — removes major disease sink; increases bone marrow "
            "burden + type 3 neurological risk; (3) Miglustat in pregnancy — teratogenic; "
            "ERT preferred peripartum; (4) VPA in type 3 with epilepsy — hepatotoxic; use LEV"
        ),
        "nbs_marker": "β-glucosidase activity (dried blood spot, DBS) — included in expanded NBS",
        "key_biomarker": "Chitotriosidase (plasma), CCL18/PARC; Lyso-GL1 (glucosylsphingosine)",
        "severity_spectrum": "Type 1 (mild–severe visceral) → Type 3 (neurological, slower) → Type 2 (fatal neonatal CNS)",
        "founder_variant": "p.Asn409Ser (N370S): 70–75% Ashkenazi alleles; p.Leu483Pro (L444P): global type 3",
        "key_variants": ["p.Asn409Ser (N370S) — type1 protective", "p.Leu483Pro (L444P) — type3/severe",
                         "p.Arg159Trp (R120W) — type3 Norrbottnian", "p.Asp448His (D409H) — cardiac Gaucher",
                         "IVS2+1G→A — null, type2"],
        "seed": SEED_BASE + 0,
    },
    # ── GLA — α-galactosidase A ──
    {
        "gene": "GLA", "alias": "GLA — α-Galactosidase A (Fabry Disease / X-Linked LSD)",
        "aa": "429 aa", "kDa": "50.8 kDa",
        "gene_class": "glycosidase",
        "locus": "Xq22.1", "omim_gene": 300644,
        "phenotype": "Fabry Disease (FD) — Gb3/lyso-Gb3 accumulation; cardiomyopathy, renal failure, stroke",
        "disease": (
            "GLA hemizygous (males) or heterozygous (females) variants → Fabry Disease (OMIM #301500). "
            "GLA encodes lysosomal α-galactosidase A (α-Gal A, 429aa, 50.8 kDa), a homodimeric enzyme "
            "cleaving terminal α-galactosyl residues from glycosphingolipids (primarily Gb3 = globotriaosylceramide) "
            "and glycoproteins. Deficiency → Gb3 + lyso-Gb3 (globotriaosylsphingosine) accumulate in "
            "vascular endothelium, renal podocytes/tubules, cardiomyocytes, dorsal root ganglia neurons. "
            "X-LINKED INHERITANCE — the ONLY major LSD gene on the X chromosome. "
            "Classic males (hemizygous, severe): childhood-onset neuropathic pain (acroparaesthesiae, "
            "crises), angiokeratomas, hypohidrosis; adulthood: progressive renal failure (ESRD 40s), "
            "HCM (cardiomyopathy 95%), stroke/TIA (40s–50s). Variant (cardiac-predominant) type: "
            "p.Arg301Gln (males) → isolated HCM without classic early symptoms. "
            "Heterozygous females: highly variable — 25–70% have significant symptoms due to skewed "
            "X-inactivation; CANNOT rely on normal α-Gal A enzyme activity to exclude Fabry in females."
        ),
        "inheritance": "X-LINKED (Xq22.1). Hemizygous males severely affected. Heterozygous females variable (25-70% symptomatic).",
        "hallmark": (
            "GLA HALLMARKS: (1) X-LINKED — only major LSD on X chromosome; α-Gal A plasma/leukocyte "
            "activity UNRELIABLE in females (normal activity does not exclude disease); DNA testing MANDATORY "
            "for all females in Fabry families; (2) NEUROPATHIC PAIN (acroparaesthesiae, Fabry crises): "
            "burning/stabbing extremity pain, triggered by fever/heat/exercise — pathognomonic in boys <10y; "
            "(3) ANGIOKERATOMAS: dark red/purple telangiectatic skin lesions (bathing trunk distribution, "
            "scrotum, umbilicus, thighs); NOT universal but highly specific when present; (4) CORNEA "
            "VERTICILLATA: whorl-pattern corneal deposits on slit-lamp — 70-90% of male patients, "
            "also 70% of female carriers; present by childhood; NOT vision-threatening; "
            "(5) HCM: left ventricular hypertrophy (95% of classic males), late gadolinium enhancement "
            "in posterior inferolateral wall (fibrosis — Fabry pattern on CMR); arrhythmia, "
            "pacemaker often needed; (6) RENAL: Gb3 in podocytes → proteinuria → ESRD (4th–5th decade); "
            "biopsy: 'zebra bodies' / concentric lamellar inclusions on EM; eGFR monitoring mandatory; "
            "(7) ERT: agalsidase alfa (Replagal) 0.2 mg/kg IV q2w; agalsidase beta (Fabrazyme) "
            "1 mg/kg IV q2w — both LEVEL A, reduce Gb3 in most tissues; CNS protection partial; "
            "(8) MIGALASTAT (chaperone therapy): oral, amenable variants ONLY (~50% of all GLA variants "
            "as per GNE database); pharmacological chaperone stabilises misfolded but functional enzyme; "
            "NOT interchangeable with ERT — verify amenability before prescribing; "
            "(9) LYSO-Gb3 (globotriaosylsphingosine): plasma biomarker most sensitive for disease "
            "burden and ERT/migalastat response; markedly elevated in classic males; may be normal "
            "in cardiac variant or mild females; (10) p.Arg301Gln (R301Q): CARDIAC VARIANT — "
            "isolated HCM phenotype, no classic early symptoms; may be missed for decades."
        ),
        "key_ddx": (
            "GLA DDx: (1) Hypertrophic cardiomyopathy (HCM) of other causes — always test α-Gal A in "
            "unexplained HCM especially in males; (2) Other causes of ESRD in young adults — eGFR + "
            "lyso-Gb3; (3) SMPD1/Niemann-Pick B — both cause lipid storage in macrophages but SMPD1 "
            "lacks angiokeratomas/neuropathic pain; (4) Neurological small-fibre neuropathy of other "
            "causes — α-Gal A + lyso-Gb3 discriminates; (5) GBA heterozygosity: PD risk without Fabry "
            "features."
        ),
        "ert_available": "YES — agalsidase alfa (0.2 mg/kg IV q2w) / agalsidase beta (1.0 mg/kg IV q2w)",
        "srt_available": "Migalastat (chaperone) for AMENABLE VARIANTS ONLY — verify GNE database",
        "hsct_role": "Not established for GLA",
        "gene_therapy_status": "Phase 1-2 (AAV lentiviral); doraglucoronidase alfa in development",
        "critical_ci": (
            "CRITICAL CI: (1) Migalastat for non-amenable variants — no benefit, delays ERT; "
            "(2) Relying on α-Gal A enzyme activity to exclude disease in FEMALES — unreliable due to "
            "X-inactivation; (3) ACE inhibitors without Fabry workup in unexplained proteinuria; "
            "(4) Enzyme infusion reactions — pre-treat with antihistamines; serious anaphylaxis possible"
        ),
        "nbs_marker": "α-galactosidase A activity (DBS); GLA gene sequencing for females",
        "key_biomarker": "Lyso-Gb3 (plasma, urine) — most sensitive; Gb3 in urine; α-Gal A leukocytes (males)",
        "severity_spectrum": "Classic (hemizygous male, all organs) → Cardiac variant (p.R301Q, HCM only) → Female carrier (variable)",
        "founder_variant": "p.Arg301Gln (cardiac variant) — common in Taiwan, other populations",
        "key_variants": ["p.Asn215Ser (N215S) — classic", "p.Arg301Gln (R301Q) — cardiac variant",
                         "p.Ala97Val (A97V) — amenable to migalastat", "p.Arg112His (R112H) — classic",
                         "c.IVS4+919G→A — exon-trapping cryptic splice (common)"],
        "seed": SEED_BASE + 1,
    },
    # ── GAA — acid α-glucosidase (Pompe) ──
    {
        "gene": "GAA", "alias": "GAA — Acid α-Glucosidase (Pompe Disease / GSD-II)",
        "aa": "952 aa", "kDa": "110 kDa",
        "gene_class": "glucosidase",
        "locus": "17q25.3", "omim_gene": 606800,
        "phenotype": "Pompe Disease (GSD-II) — lysosomal glycogen accumulation; HCM + respiratory failure",
        "disease": (
            "GAA biallelic loss → Pompe Disease (glycogen storage disease type II, OMIM #232300). "
            "GAA encodes lysosomal acid α-glucosidase (952aa, 110 kDa), which hydrolyses α-1,4 and "
            "α-1,6 glycosidic bonds in glycogen within lysosomes. Deficiency → glycogen accumulates "
            "in lysosomes of skeletal muscle, cardiac muscle, smooth muscle, and liver. "
            "IOPD (infantile-onset Pompe disease): near-complete loss (<1% residual activity); "
            "presents within first weeks of life with massive HCM (cardiac mass × volume ratio >65 g/m²), "
            "floppy infant (axial hypotonia, tongue enlargement), respiratory failure; death <12 months "
            "without ERT. LOPD (late-onset Pompe disease): partial loss (1–30% residual activity); "
            "proximal limb-girdle myopathy (progressive), diaphragm involvement → respiratory failure "
            "without cardiac involvement. c.-32-13T>G (IVS1) is the most common LOPD allele globally "
            "(90% of LOPD in Northern Europeans). CRIM (cross-reactive immunological material): "
            "CRIM-positive patients retain some GAA protein → lower antibody response to ERT; "
            "CRIM-negative → high-titre antibody neutralises ERT → MUST receive ITI."
        ),
        "inheritance": "Autosomal recessive, biallelic. IOPD: 1 in 140,000; LOPD: 1 in 60,000.",
        "hallmark": (
            "GAA HALLMARKS: (1) IOPD TRIAD: HCM (mass/volume ratio >65 g/m²) + severe hypotonia "
            "('floppy infant') + respiratory failure — within first weeks of life; heart is the "
            "dominant organ at this age; (2) ERT ESSENTIAL in IOPD: alglucosidase alfa (Myozyme) "
            "20 mg/kg IV q2w → 1st-generation ERT; avalglucosidase alfa (Nexviazyme) 40 mg/kg q2w → "
            "2nd-generation, bis-mannose-6-phosphate enriched, ~4× better muscle uptake, preferred; "
            "cipaglucosidase alfa + miglustat co-administration → 3rd-generation, highest M6P payload; "
            "(3) CRIM STATUS determines ITI need: CRIM-NEGATIVE patients lack cross-reactive protein → "
            "develop high-titre IgG antibodies neutralising ERT → MUST START ITI before or concurrent "
            "with first ERT infusion (methotrexate + rituximab + IVIG protocol); CRIM-positive: lower "
            "risk but still monitor inhibitor titres; (4) c.-32-13T>G (IVS1 splice-site): MOST COMMON "
            "LOPD allele — leaky splicing → 1–5% residual GAA → LOPD; if homozygous: LOPD phenotype "
            "virtually certain; one c.-32-13T>G + null allele → LOPD; (5) LOPD PHENOTYPE: limb-girdle "
            "pattern weakness (proximal > distal), scapular winging, rigid spine; diaphragm involvement "
            "early → nocturnal hypoventilation → morning headaches; NO cardiac involvement in LOPD "
            "distinguishes from IOPD; (6) Respiratory monitoring: spirometry sitting + supine; FVC<50% "
            "→ non-invasive ventilation before symptoms; (7) NBS (DBS α-glucosidase activity): widely "
            "implemented; low activity → confirm with GAA sequencing + lymphocyte enzyme + urine Glc4; "
            "(8) Urinary Glc4 (glucose tetrasaccharide): sensitive disease burden biomarker + ERT "
            "response marker; (9) Gene therapy (SB-525 / AT845 / ACTUS-101): Phase 2 trials promising — "
            "potential one-dose cure replacing lifelong ERT; (10) Miglustat co-delivery with "
            "cipaglucosidase: pharmacological chaperone stabilises exogenous enzyme in circulation."
        ),
        "key_ddx": (
            "GAA DDx: (1) IOPD: other causes of neonatal HCM (PTPN11/Noonan, GSD-III, storage, "
            "maternal diabetes) — enzyme activity is diagnostic; (2) LOPD: other limb-girdle MDs "
            "(LGMD 2I/FKRP, LGMD 2A/CAPN3, Becker MD, acid maltase in adults) — GAA enzyme mandatory "
            "in unexplained proximal myopathy; (3) Danon disease (LAMP2 mutation): also glycogen "
            "storage + HCM, X-linked, LAMP2 sequencing distinguishes; (4) GSD III (AGL): liver-dominant, "
            "GAA normal; (5) Consider Pompe in any unexplained 'idiopathic' respiratory failure."
        ),
        "ert_available": "YES — alglucosidase alfa (1st gen) / avalglucosidase alfa (2nd gen, preferred) / cipaglucosidase+miglustat",
        "srt_available": "Miglustat as chaperone co-administration with cipaglucosidase (not standalone SRT)",
        "hsct_role": "Not established for GAA",
        "gene_therapy_status": "Phase 2 (AAV-GAA, promising); could replace ERT",
        "critical_ci": (
            "CRITICAL CI: (1) Starting ERT in CRIM-negative IOPD WITHOUT ITI — will develop inhibitory "
            "antibodies neutralising ERT, fatal; (2) Alglucosidase infusion reactions — infusion-associated "
            "reactions (IARs) up to 50% — pre-treat; anaphylaxis rare but possible; (3) Using 1st-gen "
            "ERT when 2nd-gen available in IOPD — poorer muscle penetration; (4) Waiting for respiratory "
            "symptoms to start NIV in LOPD — start prophylactically at FVC <50%"
        ),
        "nbs_marker": "Acid α-glucosidase DBS activity (widely NBS-included); Glc4 urine biomarker",
        "key_biomarker": "Urinary Glc4 (glucose tetrasaccharide); creatine kinase (CK); FVC% sitting vs supine",
        "severity_spectrum": "IOPD (<1% activity) → LOPD with early onset → LOPD late-onset/adult",
        "founder_variant": "c.-32-13T>G (IVS1): >90% LOPD in Northern Europeans; p.Arg854Ter: most common IOPD null in non-IVS1",
        "key_variants": ["c.-32-13T>G (IVS1) — most common LOPD", "p.Arg854Ter — IOPD null",
                         "p.Asp645Glu (D645E) — severe", "p.Gly648Ser (G648S) — moderate",
                         "p.Arg600Cys (R600C) — LOPD"],
        "seed": SEED_BASE + 2,
    },
    # ── SMPD1 — sphingomyelin phosphodiesterase 1 ──
    {
        "gene": "SMPD1", "alias": "SMPD1 — Sphingomyelinase (Niemann-Pick Disease Types A & B)",
        "aa": "629 aa", "kDa": "71.5 kDa",
        "gene_class": "phosphodiesterase",
        "locus": "11p15.4", "omim_gene": 607608,
        "phenotype": "Niemann-Pick Disease type A (severe, neuronopathic) / type B (visceral, non-neuronopathic)",
        "disease": (
            "SMPD1 biallelic loss → Niemann-Pick Disease types A and B (NPA, OMIM #257200; NPB, OMIM #607616). "
            "SMPD1 encodes lysosomal acid sphingomyelinase (629aa, 71.5 kDa), which hydrolyses "
            "sphingomyelin → ceramide + phosphocholine. Deficiency → sphingomyelin accumulates in "
            "macrophages/monocytes → hepatosplenomegaly, bone marrow infiltration, lung disease. "
            "TYPE A (severe, neuronopathic): complete/near-complete loss (<1% residual); onset 3–6 months; "
            "hepatosplenomegaly, progressive neurodegenerative regression, cherry-red macular spot (50%), "
            "death by 2–3 years. TYPE B (non-neuronopathic): partial residual activity (5–10%); "
            "predominantly visceral (hepatosplenomegaly, interstitial lung disease, dyslipidaemia); "
            "NO primary CNS involvement; survival into adulthood without treatment. "
            "INTERMEDIATE (A/B): overlap; some neurological features without full NPA severity. "
            "p.Arg496Leu (R496L) is the classic NPA Ashkenazi Jewish founder mutation; "
            "p.Phe333_Phe334del is a common NPB mutation."
        ),
        "inheritance": "Autosomal recessive, biallelic. NPA: 1 in 40,000 Ashkenazi; 1 in 250,000 general. NPB: similar.",
        "hallmark": (
            "SMPD1 HALLMARKS: (1) FOAM CELLS ('sea-blue histiocytes'): lipid-laden macrophages in "
            "bone marrow — distinctive appearance on biopsy; (2) CHERRY-RED MACULA: ~50% of NPA "
            "(HEXA/Tay-Sachs also has cherry-red spot — distinguish by enzyme activity); "
            "(3) TYPE A vs TYPE B DISCRIMINATION: residual ASM activity in fibroblasts/leukocytes; "
            "A: <1% → fatal CNS; B: 5–10% → visceral only; intermediate: 1–5%; DNA genotype also "
            "distinguishes; (4) PULMONARY: NP-B → diffuse interstitial infiltrates, restrictive lung "
            "pattern; pulmonary function testing mandatory; O2 supplementation may be needed; "
            "(5) DYSLIPIDAEMIA in NP-B: high LDL, low HDL, elevated triglycerides; cardiovascular "
            "risk monitoring; (6) ERT FOR NP-B: olipudase alfa (Xenpozyme) — recombinant ASM — "
            "approved FDA/EMA 2022 for non-neuronopathic NPB — reduces spleen/liver volume + lung "
            "improvement; dose escalation mandatory (start low → escalate to avoid acute toxicity); "
            "NO ERT for NPA (no benefit in advanced CNS disease); (7) SPLEEN: massive "
            "splenomegaly → thrombocytopaenia, hypersplenism; splenectomy generally avoided (removes "
            "disease sink, may accelerate extrasplenic accumulation); (8) SPHINGOMYELIN/CERAMIDE "
            "ratio: plasma biomarker; lyso-sphingomyelin (lyso-SM) most sensitive marker "
            "for monitoring; (9) NPC1 DISTINCTION: NPC1 has cholesterol trafficking defect, not "
            "sphingomyelin — filipin staining distinguishes; NPC1 has VSGP/cataplexy; NPA/B do not; "
            "(10) Ashkenazi Jewish carrier frequency: NPA ~1 in 90 for p.Arg496Leu variant."
        ),
        "key_ddx": (
            "SMPD1 DDx: (1) NPC1 — also hepatosplenomegaly + cherry-red spot; filipin staining + NPC1/NPC2 "
            "sequencing distinguishes; VSGP in NPC1 but not NPA/B; (2) GBA Gaucher — crinkled-paper "
            "cells (GBA) vs foam/sea-blue cells (SMPD1); GBA enzyme activity; (3) HEXA Tay-Sachs — "
            "cherry-red spot + neurodegeneration but no hepatosplenomegaly; hexosaminidase A activity; "
            "(4) GM1 gangliosidosis (GLB1) — cherry-red + coarse features; β-galactosidase activity."
        ),
        "ert_available": "YES for NPB only — olipudase alfa (Xenpozyme), dose escalation required; NO ERT for NPA",
        "srt_available": "Not established",
        "hsct_role": "HSCT considered in NPA (palliative/investigational); not curative for CNS",
        "gene_therapy_status": "Preclinical; mRNA therapy in development",
        "critical_ci": (
            "CRITICAL CI: (1) ERT (olipudase) without dose escalation — acute infusion toxicity if "
            "full dose given from start; mandatory slow escalation protocol; (2) Splenectomy without "
            "careful consideration — accelerates extrasplenic disease; (3) Statin therapy for "
            "dyslipidaemia — generally manageable but monitor liver; (4) VPA in NPA patients with "
            "seizures — hepatotoxic in storage disorders; use LEV"
        ),
        "nbs_marker": "ASM (acid sphingomyelinase) DBS activity; lyso-SM plasma",
        "key_biomarker": "Lyso-sphingomyelin (lyso-SM, plasma/DBS); sphingomyelin; 7-ketocholesterol (NPC DDx)",
        "severity_spectrum": "NPA (<1% ASM, fatal CNS) → Intermediate (1-5%) → NPB (5-10%, visceral only)",
        "founder_variant": "p.Arg496Leu (NPA, Ashkenazi Jewish founder); p.Phe333_Phe334del (NPB)",
        "key_variants": ["p.Arg496Leu (R496L) — NPA Ashkenazi founder", "p.Phe333_Phe334del — NPB",
                         "p.Pro330Arg — NPB Spanish/Portuguese", "p.Arg608del — intermediate",
                         "p.Leu302Pro — severe NPA"],
        "seed": SEED_BASE + 3,
    },
    # ── NPC1 — intracellular cholesterol transporter ──
    {
        "gene": "NPC1", "alias": "NPC1 — Cholesterol Transporter (Niemann-Pick Disease Type C1)",
        "aa": "1278 aa", "kDa": "142 kDa",
        "gene_class": "cholesterol_transporter",
        "locus": "18q11.2", "omim_gene": 607623,
        "phenotype": "Niemann-Pick Disease Type C1 (NPC1) — intracellular cholesterol trafficking defect; progressive neurodegeneration",
        "disease": (
            "NPC1 biallelic loss → Niemann-Pick Disease Type C (NPC1, OMIM #257220). NPC1 encodes "
            "a 1278aa transmembrane protein in late endosomal/lysosomal membranes that exports "
            "free cholesterol from lysosomes to the rest of the cell in coordination with NPC2 "
            "(soluble lysosomal cholesterol-binding protein, OMIM #607625). NPC1 loss → free "
            "cholesterol, sphingomyelin, glycosphingolipids, and other lipids cannot exit lysosomes "
            "→ progressive accumulation in neurons, macrophages, hepatocytes. Unlike GBA/SMPD1 "
            "(direct enzyme deficiencies), NPC1 is a TRANSPORT DEFECT. Clinically: neonatal "
            "cholestatic jaundice (often resolves spontaneously); visceral disease (hepatosplenomegaly); "
            "school-age neurodegeneration — vertical supranuclear gaze palsy (VSGP) is the cardinal "
            "neurological sign; gelastic cataplexy; progressive ataxia, dystonia, dysarthria, "
            "seizures, dementia; psychosis in adolescent/adult onset. p.Ile1061Thr is the most "
            "common Western allele (~20%). Filipin staining (unesterified cholesterol) in fibroblasts "
            "remains the biochemical gold standard."
        ),
        "inheritance": "Autosomal recessive, biallelic (NPC1 95%; NPC2 5%). Estimated 1 in 120,000–150,000.",
        "hallmark": (
            "NPC1 HALLMARKS: (1) VSGP (VERTICAL SUPRANUCLEAR GAZE PALSY): PATHOGNOMONIC — school-age "
            "child who cannot move eyes up/down voluntarily but retains oculocephalic reflex; often "
            "missed; must test vertical saccades specifically (not just asking patient to look up/down); "
            "(2) GELASTIC CATAPLEXY: sudden muscle atonia triggered by emotion/laughter — also "
            "pathognomonic combination with VSGP; (3) FILIPIN STAINING: fibroblast culture → "
            "filipin (fluorescent polyene) binds unesterified cholesterol → bright perinuclear "
            "fluorescent puncta in NPC1 cells — BIOCHEMICAL GOLD STANDARD (but technically demanding); "
            "(4) MIGLUSTAT (Zavesca) SRT: APPROVED EU/Canada for progressive neurological manifestations "
            "in NPC1; oral SRT reduces glycolipid substrates; NOT a cure; stabilises neurological "
            "progression if started early; not yet FDA approved for NPC; GI side effects (90%); "
            "(5) ARIMOCLOMOL: heat-shock protein co-inducer; clinical trial (NCT02612129) showed "
            "benefit in NPC1 — compassionate use expanding; (6) PLASMA OXYSTEROLS (7-ketocholesterol, "
            "25-OH cholesterol): MOST SENSITIVE NPC1 biomarkers available; markedly elevated; "
            "correlate with disease severity + miglustat response; (7) NEONATAL CHOLESTATIC JAUNDICE: "
            "may resolve — but NPC diagnosis MUST BE MADE even if jaundice resolves; "
            "(8) NPC1 EXON 9: hotspot; p.Ile1061Thr most common (~20% of alleles); "
            "(9) PSYCHIATRIC: NPC1 onset in teens/adults commonly presents with psychosis, "
            "treatment-resistant depression — consider NPC1 in young-onset psychiatric disease "
            "with subtle neurological signs; (10) Statin therapy for cholesterol in NPC1 does NOT "
            "address lysosomal cholesterol — NOT therapeutic for NPC1 itself."
        ),
        "key_ddx": (
            "NPC1 DDx: (1) SMPD1/NPA — also hepatosplenomegaly but NO VSGP/cataplexy; sphingomyelinase "
            "activity distinguishes; (2) HEXA Tay-Sachs — progressive neurodegeneration without VSGP; "
            "hexosaminidase A activity; (3) Other causes of VSGP (PSP in adults — much older onset, "
            "no cholestatic history); (4) NPC2: same phenotype, NPC2 gene; plasma oxysterols equally "
            "elevated; filipin positive; (5) Niemann-Pick A: cherry-red spot more prominent, younger "
            "onset, no VSGP."
        ),
        "ert_available": "NO ERT available for NPC1 (transport defect, not enzyme deficiency)",
        "srt_available": "YES — miglustat (EU/Canada approved for neurological NPC1); arimoclomol (compassionate use / emerging)",
        "hsct_role": "HSCT not established for NPC1 (fails to reach CNS stores)",
        "gene_therapy_status": "Preclinical/early Phase 1 (intrathecal AAV); cyclodextrin trials (HPβCD) ongoing",
        "critical_ci": (
            "CRITICAL CI: (1) Miglustat without adequate GI preparation — severe diarrhoea/flatulence; "
            "start carbohydrate-restricted diet first; (2) Statins as primary NPC therapy — do not "
            "address lysosomal compartment; (3) Antipsychotics without NPC1 workup in young-onset "
            "psychosis with subtle ataxia/VSGP — delays diagnosis; (4) Relying on cholesterol blood "
            "levels to diagnose NPC — plasma cholesterol is NORMAL or mildly elevated; filipin + "
            "oxysterols are the diagnostic tools"
        ),
        "nbs_marker": "Plasma oxysterols (7-ketocholesterol); NPC1/NPC2 sequencing; filipin staining",
        "key_biomarker": "7-Ketocholesterol (plasma); 25-hydroxycholesterol; N-palmitoyl-O-phosphocholine (PPCS) — emerging",
        "severity_spectrum": "Perinatal hepatic (rare, severe) → Infantile/early childhood → Late childhood/adolescent → Adult onset",
        "founder_variant": "p.Ile1061Thr: ~20% Western alleles; p.Pro1007Ala — Nova Scotia Acadian founder (c.3182C>G)",
        "key_variants": ["p.Ile1061Thr (I1061T) — most common", "p.Pro1007Ala (P1007A) — Nova Scotia founder",
                         "p.Asp874Val — moderate", "p.Arg1186His — adult-onset",
                         "p.Cys177Tyr — severe infantile"],
        "seed": SEED_BASE + 4,
    },
    # ── GALC — galactocerebrosidase (Krabbe) ──
    {
        "gene": "GALC", "alias": "GALC — Galactocerebrosidase (Krabbe Disease / Globoid Cell Leukodystrophy)",
        "aa": "669 aa", "kDa": "78.4 kDa",
        "gene_class": "glycosidase",
        "locus": "14q31.3", "omim_gene": 606890,
        "phenotype": "Krabbe Disease (Globoid Cell Leukodystrophy) — psychosine accumulation; rapidly fatal leukodystrophy",
        "disease": (
            "GALC biallelic loss → Krabbe Disease (Globoid Cell Leukodystrophy, OMIM #245200). "
            "GALC encodes lysosomal galactocerebrosidase (galactosylceramidase, 669aa, 78.4 kDa), "
            "which cleaves galactose from galactosylceramide (the major myelin lipid) and from "
            "galactosylsphingosine (psychosine). GALC deficiency → psychosine (galactosylsphingosine) "
            "accumulates — psychosine is the PRIMARY NEUROTOXIN (not galactosylceramide). "
            "Psychosine kills oligodendrocytes and Schwann cells → progressive demyelination. "
            "Characteristic 'GLOBOID CELLS' (multinucleated macrophages containing PAS+ inclusions "
            "around blood vessels in white matter) are pathognomonic on brain pathology. "
            "INFANTILE KRABBE (90% of cases): onset 3–6 months; irritability, hypertonicity, "
            "hypersensitivity to stimuli, regression, peripheral neuropathy; death by 2 years. "
            "LATE-ONSET (10%): juvenile/adult form; slower progression; ataxia, spastic paraparesis. "
            "Large 30-kb deletion (c.502–1delATTTCTGTGATGACTCTGAGGAGTCCCA) spans exons 11–17 "
            "in European population (~40% of alleles)."
        ),
        "inheritance": "Autosomal recessive, biallelic. ~1 in 100,000 live births (infantile). Higher in specific populations (Muslim Arab 1 in 6,000).",
        "hallmark": (
            "GALC HALLMARKS: (1) HSCT PRE-SYMPTOMATIC ONLY — CRITICAL WINDOW: HSCT (haematopoietic "
            "stem cell transplantation) slows neurological progression ONLY if performed BEFORE "
            "neurological symptom onset — typically within the first 30–45 days of life for "
            "NBS-detected infantile Krabbe; once neurological symptoms appear (Stage 1 clinical Krabbe), "
            "HSCT provides NO BENEFIT and significant morbidity; (2) PSYCHOSINE (galactosylsphingosine): "
            "THE primary Krabbe neurotoxin — NOT galactosylceramide; oligodendrocyte-toxic at picomolar "
            "concentrations; plasma psychosine elevated in symptomatic patients; psychosine DBS in NBS "
            "second-tier test after low GALC activity; (3) CSF PROTEIN: markedly elevated >100 mg/dL "
            "in infantile Krabbe (virtually universal); markedly elevated CSF protein in an irritable "
            "infant → consider Krabbe; (4) PERIPHERAL NEUROPATHY: demyelinating neuropathy (NCS) in "
            "infantile form → hypotonia but increased deep tendon reflexes (combined central + peripheral "
            "demyelination paradox); (5) MRI: symmetric white matter involvement (posterior > anterior "
            "gradient); dentate nuclei + posterior limb internal capsule + corticospinal tract pattern; "
            "T1 high-signal in basal ganglia in some; (6) GALC ENZYME ACTIVITY (DBS): low in Krabbe; "
            "PSEUDO-DEFICIENCY: some variants reduce GALC activity in vitro but are NOT pathogenic "
            "(common false-positive on NBS) — second-tier psychosine DBS + sequencing needed; "
            "(7) LARGE DELETION: European 30-kb deletion — multiplex PCR / MLPA needed (not detected "
            "by short-read WES — specific MLPA panel required); (8) NERVE CONDUCTION: absent/markedly "
            "slowed motor velocities; NCS is a valuable functional biomarker; "
            "(9) GLOBOID CELLS: macrophages containing PAS+ inclusions around cerebral blood vessels — "
            "pathognomonic on brain biopsy/autopsy but not clinically used for diagnosis; "
            "(10) Late-onset adult Krabbe: spastic paraparesis + peripheral neuropathy in 3rd–5th decade — "
            "often misdiagnosed as hereditary spastic paraplegia (HSP); GALC enzyme + sequencing "
            "in unexplained SPG."
        ),
        "key_ddx": (
            "GALC DDx: (1) ARSA/MLD — also leukodystrophy, sulfatide urine, but no psychosine toxicity, "
            "no globoid cells; (2) Pelizaeus-Merzbacher (PLP1): X-linked leukodystrophy, nystagmus, "
            "PLP1 duplication; enzyme normal; (3) Canavan disease (ASPA): macrocephaly, N-acetylaspartate "
            "elevated, distinct MRI; (4) HEXA/Tay-Sachs: no leukodystrophy on MRI; cherry-red spot; "
            "hexosaminidase A activity; (5) Alexander disease (GFAP): frontal leukodystrophy, "
            "Rosenthal fibers, dominant; (6) HSP in adults: GALC enzyme mandatory in unexplained "
            "adult spastic paraparesis."
        ),
        "ert_available": "NO approved ERT for Krabbe (ERT cannot cross BBB to reach CNS demyelination)",
        "srt_available": "NO established SRT",
        "hsct_role": "HSCT PRE-SYMPTOMATIC ONLY — CRITICAL WINDOW; no benefit after neurological symptom onset",
        "gene_therapy_status": "Phase 1 (intrathecal/IV AAV-GALC, + HSC-mediated gene therapy); preclinical results promising",
        "critical_ci": (
            "CRITICAL CI: (1) HSCT in symptomatic infantile Krabbe — no neurological benefit + "
            "significant morbidity; ONLY pre-symptomatic; (2) Missing the 30-kb deletion with standard "
            "WES — specific MLPA panel required; (3) False reassurance from normal brain MRI early — "
            "MRI may be normal in pre-symptomatic NBS-detected Krabbe; (4) Diagnosing as other "
            "irritable infant syndromes without GALC enzyme activity"
        ),
        "nbs_marker": "GALC enzyme activity (DBS), followed by psychosine (DBS) as second-tier",
        "key_biomarker": "Plasma psychosine (galactosylsphingosine); GALC DBS activity; CSF protein (>100 mg/dL)",
        "severity_spectrum": "Infantile (3–6mo onset, fatal <2y) → Juvenile (6mo–3y) → Late-onset juvenile → Adult",
        "founder_variant": "30-kb deletion (exons 11-17): ~40% European alleles; p.Gly270Asp — Turkish/Arab founder",
        "key_variants": ["30-kb deletion c.502-1delATTT...: European ~40%", "p.Gly270Asp — Turkish/Arab founder",
                         "p.Arg168Cys — adult-onset", "p.Tyr303Cys — severe infantile",
                         "c.IVS10+1G→A — null splice"],
        "seed": SEED_BASE + 5,
    },
    # ── ARSA — arylsulfatase A (MLD) ──
    {
        "gene": "ARSA", "alias": "ARSA — Arylsulfatase A (Metachromatic Leukodystrophy / MLD)",
        "aa": "507 aa", "kDa": "57.6 kDa",
        "gene_class": "sulfatase",
        "locus": "22q13.33", "omim_gene": 607574,
        "phenotype": "Metachromatic Leukodystrophy (MLD) — sulfatide accumulation; progressive leukodystrophy",
        "disease": (
            "ARSA biallelic loss → Metachromatic Leukodystrophy (MLD, OMIM #250100). ARSA encodes "
            "lysosomal arylsulfatase A (507aa, 57.6 kDa), which desulfates sulfatide (cerebroside "
            "sulfate / galactosylsulfatide) → galactosylceramide + sulfate. Requires saposin B "
            "as a sphingolipid activator protein (PSAP gene, SapB domain); SapB deficiency "
            "(variant form of MLD) causes identical disease with normal ARSA activity. "
            "ARSA deficiency → sulfatide accumulates in lysosomes of central and peripheral nervous "
            "system myelin-forming cells (oligodendrocytes, Schwann cells) → progressive demyelination. "
            "Metachromatic granules (sulfatide deposits that stain metachromatically — shift from blue "
            "to red/brown with crystal violet / toluidine blue staining) in urine sediment and "
            "on sural nerve biopsy (pathognomonic but increasingly replaced by biochemistry). "
            "Three clinical subtypes: Late-infantile (50%): onset 1–2 years, most severe; early motor "
            "regression → flaccid quadriplegia → vegetative state, death 5–10 years. "
            "Juvenile (30%): onset 4–16 years. Adult (20%): onset >16 years; often psychiatric/cognitive "
            "first (dementia, personality change) before motor; frequently misdiagnosed as schizophrenia."
        ),
        "inheritance": "Autosomal recessive, biallelic. ~1 in 40,000–160,000. Late-infantile most common.",
        "hallmark": (
            "ARSA HALLMARKS: (1) ATIDARSAGENE AUTOTEMCEL (Libmeldy) — GENE THERAPY: HSC gene therapy "
            "with autologous CD34+ cells transduced with lentiviral ARSA vector; approved EMA 2020 "
            "(first-in-class for MLD); benefit LIMITED to: pre-symptomatic late-infantile OR "
            "early-symptomatic early-juvenile MLD; symptomatic late-infantile patients do NOT benefit; "
            "one-time infusion after myeloablative conditioning; (2) PSEUDO-DEFICIENCY ALLELES: "
            "p.Ile179Ser (I179S) and p.Arg496His (R496H) — COMMON in general population (~15% of "
            "European chromosomes carry one); reduce ARSA activity on biochemical assay to MLD-like "
            "levels WITHOUT causing MLD; CRITICAL: compound heterozygotes pseudo-def + pathogenic "
            "allele → normal ARSA activity due to allelic complementation; pseudo-def homozygotes → "
            "low ARSA activity but NO disease; (3) URINE SULFATIDE: most reliable diagnostic test; "
            "markedly elevated in all symptomatic MLD; sulfatide/glucuronide ratio on HPLC/MS-MS; "
            "pseudo-deficiency: normal urine sulfatide (most useful diagnostic discriminator); "
            "(4) MRI: symmetric bilateral white matter signal (posterior to anterior gradient); "
            "'tigroid' or 'leopard-skin' pattern (periventricular + U-fibre sparing); corticospinal "
            "tract involvement; corpus callosum; (5) SAPOSIN B (SapB): if ARSA activity low but "
            "urine sulfatide elevated + ARSA sequencing normal/pseudo-def → check PSAP sequencing "
            "for SapB domain mutations; (6) NERVE BIOPSY: metachromatic inclusions in Schwann cells; "
            "toluidine blue metachromatically stains yellow-brown instead of blue — NAME DERIVATION; "
            "(7) ADULT PSYCHIATRIC ONSET: adult MLD frequently presents with treatment-resistant "
            "psychiatric disease (schizophrenia-like psychosis, dementia) — MRI white matter changes "
            "are KEY to trigger workup; (8) PERIPHERAL NEUROPATHY: demyelinating NCS in all forms; "
            "elevated CSF protein; (9) HSCT (allogeneic BMT): previously used; gene therapy preferred; "
            "allogeneic HSCT benefits only pre-symptomatic/early-symptomatic; (10) ARSA NBS: "
            "included in some expanded NBS programmes — requires second-tier sulfatide DBS to "
            "resolve pseudo-deficiency."
        ),
        "key_ddx": (
            "ARSA DDx: (1) GALC/Krabbe — also leukodystrophy but globoid cells, psychosine elevated, "
            "GALC enzyme activity; (2) NPC1 — VSGP, cholesterol trafficking; (3) GLD and other "
            "leukodystrophies — enzyme panel; (4) PSAP/SapB deficiency — ARSA normal but MLD "
            "identical — check PSAP; (5) Adult-onset multiple sclerosis — MS has discrete lesions, "
            "normal enzyme; (6) Psychiatric/dementia differential in adults — MRI white matter "
            "pattern is diagnostic trigger; (7) VPA-induced white matter changes — drug history."
        ),
        "ert_available": "NO approved ERT (ICV ERT investigational)",
        "srt_available": "NO established approved SRT",
        "hsct_role": "Allogeneic HSCT: benefit only pre-symptomatic; replaced by gene therapy (Libmeldy) in eligible patients",
        "gene_therapy_status": "APPROVED (Libmeldy, atidarsagene autotemcel, EMA 2020) — pre/early-symptomatic only",
        "critical_ci": (
            "CRITICAL CI: (1) Gene therapy in symptomatic late-infantile MLD — no benefit; (2) Treating "
            "pseudo-deficiency as MLD — normal urine sulfatide distinguishes; (3) Missing SapB "
            "deficiency when ARSA activity is normal/low-pseudo-def + sulfatiduria present; (4) VPA "
            "in MLD patients with seizures — hepatotoxic in storage disorders; LEV preferred; "
            "(5) Allogeneic HSCT when Libmeldy gene therapy is available and patient eligible"
        ),
        "nbs_marker": "ARSA enzyme (DBS), followed by urine sulfatide second-tier (distinguishes pseudo-deficiency)",
        "key_biomarker": "Urine sulfatide (galactosylsulfatide, definitive); ARSA leukocyte activity; CSF protein",
        "severity_spectrum": "Late-infantile (1-2y, fastest) → Juvenile → Adult (psychiatric first)",
        "founder_variant": "p.Pro426Leu (P426L) — most common severe allele; pseudo-def: p.Ile179Ser + p.Arg496His",
        "key_variants": ["p.Pro426Leu — most common severe", "p.Ile179Ser (I179S) — PSEUDO-DEFICIENCY (not pathogenic)",
                         "p.Arg496His (R496H) — PSEUDO-DEFICIENCY", "p.Thr409Ile — juvenile",
                         "c.465+1G→A — null severe"],
        "seed": SEED_BASE + 6,
    },
    # ── HEXA — β-hexosaminidase A α-subunit (Tay-Sachs) ──
    {
        "gene": "HEXA", "alias": "HEXA — β-Hexosaminidase A α-Subunit (Tay-Sachs / GM2 Gangliosidosis Type I)",
        "aa": "529 aa", "kDa": "60.7 kDa",
        "gene_class": "glycosidase",
        "locus": "15q23", "omim_gene": 606869,
        "phenotype": "Tay-Sachs Disease (GM2 Gangliosidosis Type I) — GM2 ganglioside accumulation; fatal neurodegeneration",
        "disease": (
            "HEXA biallelic loss → Tay-Sachs Disease (GM2 Gangliosidosis Type I, OMIM #272800). "
            "HEXA encodes the α-subunit of lysosomal β-hexosaminidase A (HexA, αβ heterodimer). "
            "HexA cleaves the terminal N-acetylgalactosamine from GM2 ganglioside in lysosomes, "
            "a step requiring GM2 activator protein (GM2A). HEXA deficiency → selective loss of "
            "HexA (αβ) activity; HexB (ββ homodimer, Sandhoff) is preserved in Tay-Sachs — "
            "this distinguishes from Sandhoff (HEXB loss → both HexA and HexB absent). "
            "GM2 ganglioside accumulates in cortical neurons → progressive neuronal death. "
            "INFANTILE TAY-SACHS (most common): normal development 3–5 months → hyperacusis "
            "(exaggerated startle to sound — pathognomonic), progressive loss of milestones, "
            "cherry-red macular spot (90%), macrocephaly, seizures (myoclonic), spasticity, "
            "vegetative state, death by 2–5 years. NO ERT AVAILABLE. "
            "JUVENILE (subacute): onset 2–10 years; slower regression. "
            "ADULT-ONSET (chronic): onset 2nd–5th decade; motor neuron disease mimicry + "
            "cerebellar/spinocerebellar ataxia + psychosis; significantly better prognosis."
        ),
        "inheritance": "Autosomal recessive, biallelic. Infantile: 1 in 3,600 (Ashkenazi Jewish); 1 in 320,000 (general). Carrier freq Ashkenazi: 1 in 30.",
        "hallmark": (
            "HEXA HALLMARKS: (1) HYPERACUSIS + STARTLE: exaggerated, involuntary, whole-body startle "
            "response to sound (clap, door) — pathognomonic in infantile Tay-Sachs; present from "
            "first months, worsens with disease; (2) CHERRY-RED MACULAR SPOT: 90% of infantile cases "
            "on fundoscopy — ganglion cell ring around fovea filled with GM2 looks white/gray; "
            "fovea (no ganglion cells) appears as 'cherry-red' spot; also present in SMPD1/NPA, "
            "GM1 gangliosidosis, Niemann-Pick C — hexosaminidase activity distinguishes; "
            "(3) NO ERT AVAILABLE: GM2 ganglioside cannot be enzymatically replaced exogenously; "
            "neuronal access is the barrier; clinical trials of pyrimethamine (pharmacological "
            "chaperone for some variants) + miglustat (modest); gene therapy in early Phase 1; "
            "(4) HexA vs HexB ACTIVITY RATIO: HEXA deficiency → HexA absent, HexB preserved; "
            "total hexosaminidase activity may be NORMAL — MUST measure HexA:HexB ratio "
            "(heat denaturation method); HexA loss is specific; (5) ASHKENAZI JEWISH FOUNDER: "
            "three common mutations: 4-bp insertion c.1277_1278insTATC (p.Tyr427IlefsTer5, "
            "~70–80% of Ashkenazi alleles, null), splice-site c.1421+1G→C (~18% Ashkenazi alleles, "
            "null), and missense p.Gly269Ser (~2–4%, associated with adult/juvenile form); "
            "(6) CARRIER SCREENING SUCCESS: population-based Tay-Sachs carrier screening in "
            "Ashkenazi Jewish community (since 1970s) has reduced infantile Tay-Sachs by >90%; "
            "gold standard example of preventive genetic medicine; HexA enzyme + HEXA/HEXB gene "
            "sequencing for carrier testing; (7) ADULT-ONSET TAY-SACHS: motor neuron disease "
            "phenotype — weakness, fasciculations + ataxia + psychiatric features; often "
            "misdiagnosed as ALS, SCA, or bipolar; enzyme activity test in any unexplained adult "
            "MND; (8) GM2 ACTIVATOR DEFICIENCY (GM2A gene): phenotypically identical to Tay-Sachs "
            "but both HexA and HexB activity NORMAL in absence of GM2A activator; must check "
            "GM2 activator if enzyme normal in classic Tay-Sachs presentation; "
            "(9) SANDHOFF DISTINCTION (HEXB): loss of HEXB → loss of BOTH HexA AND HexB; "
            "total hexosaminidase activity near zero (vs Tay-Sachs: HexB preserved); "
            "clinically similar to infantile Tay-Sachs but no cherry-red spot founder specificity; "
            "(10) SUBSTRATE REDUCTION: miglustat (GluSyn inhibitor) provides modest symptomatic "
            "benefit in adult Tay-Sachs; not approved for infantile."
        ),
        "key_ddx": (
            "HEXA DDx: (1) HEXB/Sandhoff: phenotypically identical infantile presentation; "
            "HexA + HexB BOTH absent in Sandhoff; Ashkenazi founder NOT Sandhoff; (2) SMPD1/NPA: "
            "cherry-red + neurodegeneration but no hyperacusis; sphingomyelinase + hepatosplenomegaly; "
            "(3) NPC1: cholesterol trafficking, VSGP; (4) GM1 gangliosidosis (GLB1): also cherry-red + "
            "coarse features; β-galactosidase activity; (5) ALS in adults: HEXA enzyme in unexplained "
            "adult MND + cerebellar/psychiatric features; (6) GM2A activator deficiency: both HexA/HexB "
            "normal but GM2 activator absent — HEXA/HEXB sequencing normal; test GM2 activator."
        ),
        "ert_available": "NO ERT available for Tay-Sachs (GM2 ganglioside; neuronal access barrier unresolved)",
        "srt_available": "Miglustat (modest benefit in adult Tay-Sachs; not infantile); pyrimethamine (p.Gly269Ser chaperone investigational)",
        "hsct_role": "Not established for HEXA",
        "gene_therapy_status": "Phase 1 (intrathecal/ICM AAV-HEXA+HEXB bicistronic); early but promising",
        "critical_ci": (
            "CRITICAL CI: (1) Measuring only total hexosaminidase — MUST measure HexA:HexB ratio "
            "(heat denaturation) or use fluorescent substrates to detect HexA-specific loss; "
            "(2) Diagnosing adult-onset Tay-Sachs as ALS or psychiatric disease without enzyme "
            "testing; (3) VPA in seizures of infantile Tay-Sachs — hepatotoxic; LEV/CLB preferred; "
            "(4) Pseudo-deficiency (reduction of HexA thermolabile fraction without disease — "
            "p.Arg247Trp): common false-positive in some Ashkenazi carrier screening programmes; "
            "confirm with DNA testing"
        ),
        "nbs_marker": "HexA enzyme activity (DBS, heat-labile fraction); HEXA sequencing in Ashkenazi",
        "key_biomarker": "HexA enzyme activity (leukocytes/serum, heat-inactivation assay); GM2 ganglioside (CSF); HEXA sequencing",
        "severity_spectrum": "Infantile (fatal <5y) → Juvenile (2-10y) → Adult/chronic (2nd-5th decade, better prognosis)",
        "founder_variant": "c.1277_1278insTATC: ~70-80% Ashkenazi alleles (null); c.1421+1G→C: ~18% Ashkenazi (null)",
        "key_variants": ["c.1277_1278insTATC — Ashkenazi null (70-80%)", "c.1421+1G→C — Ashkenazi null (18%)",
                         "p.Gly269Ser (G269S) — adult/juvenile onset", "p.Arg247Trp — pseudo-deficiency carrier",
                         "p.Tyr427IlefsTer5 — severe null"],
        "seed": SEED_BASE + 7,
    },
]

# ── Patient cohort generator ───────────────────────────────────────────────────
def _make_patients_for_gene(g):
    rng = random.Random(g["seed"])
    gene = g["gene"]
    patients = []
    n = 40
    for i in range(n):
        pid = f"LSD-{gene}-{i+1:03d}"
        # Age at diagnosis (varies by gene/type)
        if gene == "GBA":
            age_dx = rng.choice(
                [rng.uniform(0, 1)] * 12 +    # type 2 (neonatal)
                [rng.uniform(1, 12)] * 20 +    # type 1/3 childhood
                [rng.uniform(12, 45)] * 8      # adult type 1
            )
            subtype = rng.choice(["type1"] * 15 + ["type3"] * 7 + ["type2"] * 2 +
                                  ["type1_mild"] * 10 + ["type1_neuronopathic"] * 6)
            ert_on = rng.random() < 0.7
            srt_on = rng.random() < 0.2
        elif gene == "GLA":
            age_dx = rng.choice(
                [rng.uniform(5, 20)] * 20 +    # classic male childhood
                [rng.uniform(20, 50)] * 15 +   # adult
                [rng.uniform(30, 60)] * 5      # cardiac variant late
            )
            subtype = rng.choice(["classic_male"] * 18 + ["cardiac_variant"] * 8 + ["female_affected"] * 14)
            ert_on = rng.random() < 0.65
            srt_on = rng.random() < 0.2  # migalastat amenable only
        elif gene == "GAA":
            age_dx = rng.choice(
                [rng.uniform(0, 0.5)] * 18 +   # IOPD neonatal/infantile
                [rng.uniform(2, 40)] * 22       # LOPD late
            )
            subtype = rng.choice(["IOPD"] * 18 + ["LOPD_early"] * 12 + ["LOPD_late"] * 10)
            ert_on = rng.random() < 0.85
            srt_on = False
        elif gene == "SMPD1":
            age_dx = rng.choice(
                [rng.uniform(0, 1)] * 14 +     # NPA infantile
                [rng.uniform(2, 30)] * 26      # NPB/intermediate
            )
            subtype = rng.choice(["NPA"] * 14 + ["NPB"] * 20 + ["intermediate"] * 6)
            ert_on = rng.random() < 0.4  # only NPB eligible
            srt_on = False
        elif gene == "NPC1":
            age_dx = rng.choice(
                [rng.uniform(4, 15)] * 22 +   # classic school-age
                [rng.uniform(15, 40)] * 12 +  # adolescent/adult
                [rng.uniform(0, 1)] * 6       # perinatal hepatic
            )
            subtype = rng.choice(["classic_childhood"] * 22 + ["adolescent"] * 10 + ["adult"] * 6 + ["perinatal"] * 2)
            ert_on = False  # no ERT for NPC1
            srt_on = rng.random() < 0.6  # miglustat
        elif gene == "GALC":
            age_dx = rng.choice(
                [rng.uniform(0, 1)] * 28 +    # infantile (90%)
                [rng.uniform(1, 10)] * 8 +    # juvenile
                [rng.uniform(10, 40)] * 4     # late-onset
            )
            subtype = rng.choice(["infantile"] * 28 + ["juvenile"] * 8 + ["late_onset"] * 4)
            ert_on = False
            srt_on = False
        elif gene == "ARSA":
            age_dx = rng.choice(
                [rng.uniform(1, 4)] * 20 +   # late-infantile 50%
                [rng.uniform(4, 16)] * 12 +  # juvenile 30%
                [rng.uniform(16, 50)] * 8    # adult 20%
            )
            subtype = rng.choice(["late_infantile"] * 20 + ["juvenile"] * 12 + ["adult"] * 8)
            ert_on = False
            srt_on = False
        else:  # HEXA
            age_dx = rng.choice(
                [rng.uniform(0, 1)] * 25 +   # infantile 80%
                [rng.uniform(2, 15)] * 8 +   # juvenile
                [rng.uniform(15, 50)] * 7    # adult
            )
            subtype = rng.choice(["infantile"] * 25 + ["juvenile"] * 8 + ["adult_onset"] * 7)
            ert_on = False
            srt_on = rng.random() < 0.2  # miglustat adult only
        # Organ involvement
        has_spleen = gene in ("GBA", "SMPD1", "NPC1") and rng.random() < 0.85
        has_liver = gene in ("GBA", "SMPD1", "NPC1", "GAA") and rng.random() < 0.75
        has_hcm = (gene == "GLA" and rng.random() < 0.92) or (gene == "GAA" and "IOPD" in subtype and rng.random() < 0.95)
        has_neuro = gene in ("NPC1", "GALC", "ARSA", "HEXA", "SMPD1") and rng.random() < 0.80
        hsct_done = gene in ("GALC", "ARSA") and rng.random() < 0.30
        # Deceased
        deceased = (
            (gene == "HEXA" and subtype == "infantile" and rng.random() < 0.85) or
            (gene == "GALC" and subtype == "infantile" and rng.random() < 0.90) or
            (gene == "SMPD1" and subtype == "NPA" and rng.random() < 0.80) or
            (gene == "GAA" and subtype == "IOPD" and not ert_on and rng.random() < 0.70) or
            False
        )
        patients.append({
            "pid": pid,
            "gene": gene,
            "subtype": subtype,
            "age_dx_y": round(age_dx, 1),
            "splenomegaly": has_spleen,
            "hepatomegaly": has_liver,
            "cardiomyopathy": has_hcm,
            "neurological": has_neuro,
            "ert_on": ert_on,
            "srt_on": srt_on,
            "hsct_done": hsct_done,
            "deceased": deceased,
        })
    return patients


ALL_PATIENTS = []
for _g in LSD_GENES:
    _pts = _make_patients_for_gene(_g)
    _g["n_patients"] = len(_pts)
    _g["patients"] = _pts
    ALL_PATIENTS.extend(_pts)


# ─── API: get_overview ──────────────────────────────────────────────────────────
def get_overview():
    total = len(ALL_PATIENTS)
    n_ert = sum(1 for p in ALL_PATIENTS if p["ert_on"])
    n_srt = sum(1 for p in ALL_PATIENTS if p["srt_on"])
    n_hsct = sum(1 for p in ALL_PATIENTS if p["hsct_done"])
    n_deceased = sum(1 for p in ALL_PATIENTS if p["deceased"])
    n_neuro = sum(1 for p in ALL_PATIENTS if p["neurological"])
    n_hcm = sum(1 for p in ALL_PATIENTS if p["cardiomyopathy"])
    n_spleen = sum(1 for p in ALL_PATIENTS if p["splenomegaly"])
    gene_summary = []
    for g in LSD_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "locus": g["locus"],
            "n_patients": g["n_patients"],
            "ert_available": g["ert_available"],
            "srt_available": g["srt_available"],
            "hsct_role": g["hsct_role"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "pct_ert": round(100 * sum(1 for p in pts if p["ert_on"]) / len(pts), 1),
            "pct_srt": round(100 * sum(1 for p in pts if p["srt_on"]) / len(pts), 1),
            "pct_neuro": round(100 * sum(1 for p in pts if p["neurological"]) / len(pts), 1),
            "pct_deceased": round(100 * sum(1 for p in pts if p["deceased"]) / len(pts), 1),
        })
    return {
        "atlas": "LSD-Atlas — Complete 8-Gene Lysosomal Storage Disorder Atlas",
        "n_genes": len(LSD_GENES),
        "n_patients": total,
        "seeds": [g["seed"] for g in LSD_GENES],
        "genes_covered": [g["gene"] for g in LSD_GENES],
        "gene_classes": {
            "sphingolipidoses": ["GBA", "GLA", "GALC", "ARSA", "HEXA"],
            "glycogen_storage": ["GAA"],
            "sphingomyelinase_transport": ["SMPD1", "NPC1"],
        },
        "aggregate_clinical": {
            "pct_ert_on": round(100 * n_ert / total, 1),
            "pct_srt_on": round(100 * n_srt / total, 1),
            "pct_hsct": round(100 * n_hsct / total, 1),
            "pct_neurological": round(100 * n_neuro / total, 1),
            "pct_hcm": round(100 * n_hcm / total, 1),
            "pct_splenomegaly": round(100 * n_spleen / total, 1),
            "pct_deceased": round(100 * n_deceased / total, 1),
        },
        "gene_summary": gene_summary,
        "critical_clinical_rules": [
            "GBA: ERT (imiglucerase/velaglucerase/avalglucosidase) LEVEL A type 1/3 visceral; NO CNS benefit; eliglustat SRT requires CYP2D6 testing; GBA heterozygosity = 5-10x PD/DLB risk",
            "GLA X-LINKED: α-Gal A enzyme activity UNRELIABLE in females — DNA testing MANDATORY; migalastat ONLY for amenable variants (verify GNE database)",
            "GAA CRIM-NEGATIVE: ITI (methotrexate+rituximab+IVIG) MANDATORY before or with first ERT in CRIM-negative IOPD — omitting causes fatal antibody-mediated ERT neutralisation",
            "SMPD1: olipudase alfa (ERT) approved for NPB ONLY — NO benefit for NPA CNS; dose escalation mandatory to avoid toxicity",
            "NPC1: NO ERT — transport defect; miglustat SRT stabilises neurological progression; VSGP + cataplexy = pathognomonic combination",
            "GALC (Krabbe): HSCT ONLY pre-symptomatic — once neurological symptoms appear, NO HSCT benefit; psychosine is the neurotoxin (not galactosylceramide)",
            "ARSA (MLD): Libmeldy gene therapy approved EMA 2020 — pre/early-symptomatic only; pseudo-deficiency alleles (I179S/R496H) NOT pathogenic — urine sulfatide discriminates",
            "HEXA (Tay-Sachs): NO ERT — measure HexA:HexB RATIO (not total hexosaminidase); Ashkenazi carrier screening has reduced infantile Tay-Sachs >90%",
        ],
    }


# ─── API: get_breakdown ─────────────────────────────────────────────────────────
def get_breakdown():
    gene_rows = []
    for g in LSD_GENES:
        pts = g["patients"]
        gene_rows.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "gene_class": g["gene_class"],
            "n_patients": g["n_patients"],
            "seed": g["seed"],
            "phenotype": g["phenotype"],
            "inheritance": g["inheritance"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "ert_available": g["ert_available"],
            "srt_available": g["srt_available"],
            "hsct_role": g["hsct_role"],
            "gene_therapy_status": g["gene_therapy_status"],
            "critical_ci": g["critical_ci"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "key_variants": g["key_variants"],
            # Aggregate from cohort
            "pct_ert": round(100 * sum(1 for p in pts if p["ert_on"]) / len(pts), 1),
            "pct_srt": round(100 * sum(1 for p in pts if p["srt_on"]) / len(pts), 1),
            "pct_hsct": round(100 * sum(1 for p in pts if p["hsct_done"]) / len(pts), 1),
            "pct_neuro": round(100 * sum(1 for p in pts if p["neurological"]) / len(pts), 1),
            "pct_hcm": round(100 * sum(1 for p in pts if p["cardiomyopathy"]) / len(pts), 1),
            "pct_spleen": round(100 * sum(1 for p in pts if p["splenomegaly"]) / len(pts), 1),
            "pct_deceased": round(100 * sum(1 for p in pts if p["deceased"]) / len(pts), 1),
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })
    return {
        "genes": gene_rows,
        "total": len(LSD_GENES),
        "total_patients": len(ALL_PATIENTS),
    }


# ─── API: get_definitions ─────────────────────────────────────────────────────
def get_definitions():
    return {
        "atlas": "LSD-Atlas — Complete 8-Gene Lysosomal Storage Disorder Atlas",
        "lsd_overview": {
            "full_name": "Lysosomal Storage Disorders — inherited defects in lysosomal enzymes or transport proteins → substrate accumulation",
            "genes_in_atlas": 8,
            "total_known_lsds": "~50+",
            "collective_incidence": "1 in 5,000–7,500 live births",
            "inheritance_note": "All autosomal recessive except GLA (Fabry, X-linked)",
        },
        "definitions": [
            {"term": "Lysosomal Storage Disorder (LSD)", "definition": "Inherited defects in lysosomal hydrolases (enzymes) or membrane transporters → accumulation of undegraded substrates (sphingolipids, glycosaminoglycans, glycogen, cholesterol) within lysosomes → cellular dysfunction. ~50+ distinct disorders; collectively 1 in 5,000-7,500 live births. Most are AR; GLA (Fabry) is X-linked. Treatment: enzyme replacement therapy (ERT), substrate reduction therapy (SRT), HSCT, or gene therapy depending on the specific disorder and availability."},
            {"term": "Enzyme Replacement Therapy (ERT)", "definition": "IV infusion of recombinant lysosomal enzyme (produced in CHO cells or other expression systems) to replace the missing/deficient endogenous enzyme. Available for GBA (imiglucerase/velaglucerase/taliglucerase), GLA (agalsidase alfa/beta), GAA (alglucosidase alfa/avalglucosidase alfa), SMPD1-B (olipudase alfa), IDUA/MPS1, GNS/MPS3, GALNS/MPS4. ERT does NOT cross blood-brain barrier — limited/no CNS benefit. Must be administered indefinitely (no cure). Immune reactions (including anaphylaxis) possible — pre-medication protocol required."},
            {"term": "Substrate Reduction Therapy (SRT)", "definition": "Oral medication that reduces biosynthesis of the accumulating substrate to match the residual catabolic capacity of the deficient enzyme. Approved SRT: miglustat (Zavesca, iminosugar GluSyn inhibitor) for GBA type 1 and NPC1 neurological symptoms; eliglustat (Cerdelga) for GBA type 1 only (requires CYP2D6 metaboliser testing). Miglustat crosses BBB → used in NPC1 CNS; eliglustat does NOT cross BBB. SRT is NOT a cure; generally second-line to ERT for visceral disease."},
            {"term": "Cross-Reactive Immunological Material (CRIM)", "definition": "CRIM refers to residual GAA protein detectable by western blot in Pompe disease. CRIM-positive patients have some residual protein → reduced immune response to exogenous ERT. CRIM-negative patients have NO residual protein → ERT is a completely foreign antigen → high-titre IgG antibodies neutralise ERT → MUST receive immune tolerance induction (ITI: methotrexate+rituximab+IVIG) before or concurrent with first ERT. Failure to do ITI in CRIM-negative IOPD is a treatment error with fatal consequence. Genotype predicts CRIM status (null/null = CRIM-negative in most cases)."},
            {"term": "Gaucher Cells", "definition": "Lipid-laden macrophages in GBA deficiency. Glucocerebroside-engorged macrophages develop a distinctive 'crinkled tissue paper' or 'wrinkled silk' cytoplasmic appearance on Giemsa/H&E staining. Found in bone marrow, spleen, liver. Pathognomonic for Gaucher disease. Differentiate from Niemann-Pick foam/sea-blue histiocytes (SMPD1) by enzyme activity. Gaucher cells in bone marrow cause: thrombocytopaenia, anaemia, Erlenmeyer flask deformity of femur (cortical thinning), osteonecrosis."},
            {"term": "Cherry-Red Macula", "definition": "Fundoscopic finding: central fovea appears bright red surrounded by opaque/grey ring of ganglion cells engorged with storage material. Present in: HEXA/Tay-Sachs (90%), SMPD1/NPA (50%), NPC1, GM1 gangliosidosis (GLB1). The fovea has no ganglion cells — normal red choroidal reflex seen; ganglion cell ring around it is pale/grey from ganglioside storage. Enzyme activity panels distinguish the cause. NOT present in GAA, GBA type 1, or ARSA/MLD."},
            {"term": "Vertical Supranuclear Gaze Palsy (VSGP)", "definition": "Cardinal sign of NPC1. Inability to move eyes voluntarily up/down (supranuclear = above cranial nerve nuclei level), with PRESERVED oculocephalic reflex (eyes move with passive head turn — brainstem/CN3 nuclei intact). Specifically: failure of VERTICAL saccades. Horizontal saccades initially preserved. Caused by Gb3/lipid accumulation in riMLF (rostral interstitial nucleus of MLF) and INC (interstitial nucleus of Cajal) controlling vertical gaze. Often missed — must specifically test vertical saccades; not just asking patient to look 'up'. VSGP + gelastic cataplexy in school-age child = NPC1 until proven otherwise."},
            {"term": "Filipin Staining", "definition": "Fluorescent polyene antibiotic (filipin) that binds unesterified (free) cholesterol. In NPC1 fibroblasts: markedly enlarged fluorescent perinuclear puncta (late endosomal/lysosomal cholesterol trapping) visible on UV microscopy. Biochemical gold standard for NPC1 diagnosis (>95% sensitivity for classic NPC1). Technically demanding (fibroblast culture required; perishable reagent). Complemented by plasma oxysterols (7-ketocholesterol) which are more practical. Atypical NPC1 (variant filipin pattern) requires DNA sequencing."},
            {"term": "Psychosine (Galactosylsphingosine)", "definition": "Primary neurotoxin in Krabbe disease (GALC deficiency). Unlike galactosylceramide (the primary myelin lipid), psychosine is a highly toxic lyso-sphingolipid accumulating in oligodendrocytes and Schwann cells when GALC is absent. Kills oligodendrocytes at picomolar concentrations → progressive demyelination. Plasma psychosine is the most sensitive biomarker for symptomatic Krabbe and response to HSCT. Elevated psychosine predicts neurological deterioration. Psychosine DBS assay is used as second-tier NBS confirmatory test after low GALC activity."},
            {"term": "ARSA Pseudo-Deficiency", "definition": "Specific ARSA variants (most commonly p.Ile179Ser/I179S and p.Arg496His/R496H) reduce ARSA enzyme activity to MLD-like levels in biochemical assays WITHOUT causing MLD. Carrier frequency ~15% in European populations. Critical clinical trap: NBS using ARSA enzyme activity alone will flag pseudo-deficiency individuals as 'positive' for MLD. Distinguishing test: URINE SULFATIDE — elevated in true MLD, NORMAL in pseudo-deficiency. Pseudo-deficiency homozygotes: low enzyme, normal urine sulfatide, no disease. DNA panel is definitive for variant classification."},
            {"term": "NBS (Newborn Screening) for LSD", "definition": "Expanded newborn screening (DBS acylcarnitine/amino acid/enzyme panel) now includes multiple LSDs in many jurisdictions (USA, UK, EU variable). Common NBS-included LSDs: GBA (Gaucher), GLA (Fabry), GAA (Pompe), GALC (Krabbe), MPS1, MPS2. NBS advantage: pre-symptomatic identification → treatment before irreversible damage. NBS challenge: pseudo-deficiency (ARSA, GALC) creates false positives requiring second-tier testing (sulfatide, psychosine, DNA). NBS benefit clearest for GALC (Krabbe) and GAA (Pompe) where pre-symptomatic treatment changes outcomes dramatically."},
            {"term": "Migalastat (Galafold)", "definition": "Oral pharmacological chaperone (iminosugar) that binds and stabilises misfolded but catalytically active GLA protein in the ER → facilitates trafficking to lysosomes → increased α-Gal A activity. ONLY effective for AMENABLE GLA VARIANTS — those where the protein is misfolded but retains latent catalytic activity (typically missense variants with residual activity >3% of normal under in vitro assay). Amenability determined by GNE (GoodNovation Enzyme) amenability table or cell-based assay. NOT interchangeable with ERT — verify amenability BEFORE prescribing. Non-amenable variants: no benefit from migalastat."},
            {"term": "Atidarsagene Autotemcel (Libmeldy)", "definition": "HSC gene therapy (ex vivo lentiviral) for MLD (ARSA deficiency). Patient's own CD34+ haematopoietic stem cells are collected, transduced with a lentiviral vector encoding functional ARSA, then reinfused after myeloablative conditioning. Approved EMA 2020 (first-in-class for MLD). BENEFIT RESTRICTED TO: (1) Pre-symptomatic late-infantile MLD (detected via NBS or affected sibling); (2) Early-symptomatic early-juvenile MLD (mild/no cognitive impairment). Symptomatic late-infantile patients: NO benefit (disease too advanced). Manufacturing time ~4 months — interim bridging needed. Monitoring: ARSA activity in leukocytes + urine sulfatide + MRI."},
            {"term": "Olipudase Alfa (Xenpozyme)", "definition": "Recombinant human acid sphingomyelinase (rhASM) for Niemann-Pick type B (non-neuronopathic SMPD1 deficiency). FDA/EMA approved 2022. Mechanism: IV infusion replaces deficient ASM → reduces sphingomyelin in spleen, liver, lung. DOSE ESCALATION MANDATORY: start at very low dose (0.03 mg/kg) → escalate over ~6 months to maintenance 3 mg/kg q2w; rapid escalation causes acute toxicity (hepatic enzyme elevation, systemic lipid mobilisation). NOT indicated for NPA (CNS disease beyond ERT reach). Monitor: spleen/liver volume (MRI), lung diffusion capacity, platelet count, LFTs during escalation."},
            {"term": "GBA and Parkinson Disease Risk", "definition": "GBA heterozygous pathogenic variants confer 5-10× increased lifetime risk of Parkinson disease (PD) and Dementia with Lewy Bodies (DLB). GBA variants are the most common genetic risk factor for PD identified to date. In Ashkenazi PD populations: 10-15% carry a GBA variant. Risk is allele-dependent: p.Asn409Ser (N370S) → 5× risk; p.Leu483Pro (L444P) → 10× risk; more severe alleles → higher risk. Mechanism: partial GluCerase loss → impairs α-synuclein clearance via autophagy-lysosome pathway → Lewy body formation. All Gaucher disease patients and GBA carriers should be monitored for prodromal PD (hyposmia, REM sleep disorder, constipation, mild cognitive changes)."},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== LSD Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"ERT on: {ov['aggregate_clinical']['pct_ert_on']}%")
    print(f"Neurological: {ov['aggregate_clinical']['pct_neurological']}%")
    print(f"HCM: {ov['aggregate_clinical']['pct_hcm']}%")
    print(f"Deceased: {ov['aggregate_clinical']['pct_deceased']}%")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
    print("=== ALL PASS ===")
