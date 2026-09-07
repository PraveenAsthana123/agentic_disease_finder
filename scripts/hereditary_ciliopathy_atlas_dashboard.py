#!/usr/bin/env python3
"""Hereditary-Ciliopathy-Atlas — Complete 8-Gene Atlas (Primary Cilia Disorders)
BBS1    (BBS1 — Bardet-Biedl Syndrome subunit 1; 593 aa; 11q13.2; AR;
         Bardet-Biedl Syndrome type 1; most common BBS gene ~23%;
         retinal dystrophy ALWAYS first (rod-cone); obesity; polydactyly;
         renal anomalies; cognitive impairment; M1L founder Caucasian;
         seed SEED_BASE+0) .
CEP290  (Centrosomal Protein 290; 2479 aa; 12q21.32; AR;
         Joubert Syndrome (JBTS5) + Meckel-Gruber (MKS4) + LCA10 + NPHP6 + Senior-Løken;
         c.2991+1655A>G intronic Leber Congenital Amaurosis — deep intronic MOST COMMON LCA variant;
         MOLAR TOOTH SIGN on MRI brainstem PATHOGNOMONIC Joubert;
         seed SEED_BASE+1) .
NPHP1   (Nephrocystin-1; 732 aa; 2q13; AR;
         Nephronophthisis type 1 (NPHP1) — most common genetic cause of renal failure in children;
         deletion 2q13 homozygous ~85% NPHP1 — detected by MLPA NOT standard sequencing;
         corticomedullary cysts + tubular basement membrane disruption;
         Senior-Løken if retinal involvement; Joubert if cerebellar;
         seed SEED_BASE+2) .
RPGR    (Retinitis Pigmentosa GTPase Regulator; 815 aa; Xp11.3; XLR;
         X-linked Retinitis Pigmentosa (XLRP) — most common severe RP;
         RPGR-ORF15 exon: 70-75% of all XLRP mutations; hotspot requiring specific sequencing;
         female carriers: 20% symptomatic (variable X-inactivation);
         cone-rod dystrophy variant; combined RP + recurrent respiratory infections;
         seed SEED_BASE+3) .
AHI1    (Abelson Helper Integration site 1; 1196 aa; 6q23.3; AR;
         Joubert Syndrome type 3 (JBTS3); MOLAR TOOTH SIGN required;
         high rate retinal involvement (80%) + nephronophthisis;
         AHI1 Joubert more severe ocular phenotype than INPP5E Joubert;
         R830W Ashkenazi Jewish founder — screening indicated;
         seed SEED_BASE+4) .
ALMS1   (Alström Syndrome protein 1; 4169 aa; 2p13.1; AR;
         Alström Syndrome — unique ciliopathy: NO polydactyly; NO cognitive impairment;
         cone-rod dystrophy by 1yr; dilated cardiomyopathy in infancy (1st presentation);
         type 2 diabetes in childhood + obesity + hearing loss;
         DISTINGUISH from BBS: normal intelligence; no extra digits; cardiomyopathy;
         seed SEED_BASE+5) .
KIF7    (Kinesin Family Member 7; 1343 aa; 15q26.1; AR;
         Acrocallosal Syndrome (ACLS) + Hydrolethalus Syndrome 2 + Joubert type 12;
         Hedgehog pathway ciliary gatekeeper — KIF7 null → constitutive Hedgehog activation;
         polydactyly + agenesis corpus callosum + facial dysmorphism;
         phenotypic spectrum: lethal (hydrolethalus) to mild (JBTS12) by variant type;
         seed SEED_BASE+6) .
DYNC2H1 (Dynein Cytoplasmic 2 Heavy Chain 1; 4307 aa; 11q22.3; AR;
         Short-Rib Thoracic Dysplasia / Jeune Asphyxiating Thoracic Dystrophy (SRTD3/ATD);
         narrow thorax causing neonatal respiratory failure — #1 lethal outcome;
         retrograde intraflagellar transport (IFT-B); polydactyly variable;
         renal, hepatic, retinal involvement in survivors;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1950–1957)
"""

import random

SEED_BASE = 1950

CILIOPATHY_GENES = [
    # -- BBS1 — Bardet-Biedl Syndrome 1 ----------------------------------------
    {
        "gene": "BBS1",
        "alt_name": "BBS1 (Bardet-Biedl Syndrome 1)",
        "protein": (
            "BBS1 -- 11q13.2 AR -- BBS1-593aa -- "
            "Bardet-Biedl-Syndrome-Type1-Most-Common-23pct -- "
            "Retinal-Dystrophy-ALWAYS-First-Rod-Cone-Onset-5-15yr -- "
            "Obesity-Truncal-Onset-Infancy -- Polydactyly-Postaxial-Most-Common -- "
            "Renal-Anomalies-50-100pct -- M1L-Caucasian-Founder-Frequent -- "
            "BBSome-Component-Cilia-Trafficking"
        ),
        "locus": "11q13.2",
        "protein_size": "593 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Retinal: rod-cone dystrophy onset 5-15 yr (nyctalopia first), legal blindness by 3rd decade; "
            "Obesity: truncal adiposity from infancy; "
            "Polydactyly: postaxial (6th digit), present at birth; "
            "Renal: structural anomalies from fetal life (calyceal clubbing, cysts); "
            "Cognitive: borderline to mild impairment in ~50%"
        ),
        "key_biomarker": (
            "Ophthalmology: rod-cone ERG (severely reduced rod responses before cone); "
            "fundus: salt-and-pepper retinopathy, disc pallor, attenuated vessels; "
            "renal ultrasound: calyceal clubbing, corticomedullary cysts, structural anomalies; "
            "molecular: BBS1 pathogenic variant — M1L p.Met1Leu founder in Caucasians; "
            "BMI: obesity from infancy; "
            "endocrine: hypogonadism (hypergonadotropic males; hypogonadotropic females)"
        ),
        "pathognomonic": (
            "BBS pentad: rod-cone dystrophy + truncal obesity + postaxial polydactyly + "
            "renal anomalies + cognitive impairment = BBS (not all in every patient); "
            "CLINICAL DIAGNOSIS: 4 primary OR 3 primary + 2 secondary features; "
            "PRIMARY features: rod-cone dystrophy, polydactyly, obesity, renal, learning disability, hypogonadism; "
            "DISTINGUISH from Alström: Alström has NO polydactyly, NO cognitive impairment; "
            "DISTINGUISH from NPHP: NPHP is slim, NO retinal in isolated NPHP1"
        ),
        "treatment": (
            "Retinal: no disease-modifying therapy; low-vision aids; avoid photosensitizing drugs; "
            "Obesity: early dietitian + behavioural programme; GLP-1 agonists studied; "
            "Renal: monitor GFR annually; nephroprotective (ACE-i/ARB); renal transplantation if ESRD; "
            "Hypogonadism: testosterone replacement males; oestrogen-progesterone females; "
            "Learning: special educational needs assessment early; "
            "Polydactyly: surgical removal neonatal period; "
            "Genetic counselling: AR — 25% recurrence; molecular confirmation in proband before cascade"
        ),
        "critical_flags": [
            "BBS1-RETINAL-IRREVERSIBLE: no neuroprotective therapy approved; early diagnosis allows low-vision planning but NOT reversal; gene therapy trials (BBS1 AAV) — enrol if eligible",
            "BBS1-RENAL-SILENT-PROGRESSION: renal involvement may be asymptomatic until late; annual GFR + urine ACR mandatory; ESRD in ~25% by age 35 if untreated",
            "BBS1-M1L-FOUNDER: p.Met1Leu (c.1A>G) recurrent Caucasian founder; standard Sanger may miss — confirm with allele-specific assay or NGS",
            "BBS1-HYPOGONADISM-MALES: males almost universally hypogonadotropic hypogonadism; testosterone supplementation → fertility NOT restored (tubular failure); sperm banking if ANY residual spermatogenesis",
            "BBS1-BBSome-CARGO: BBS1 is core BBSome subunit; trafficking of GPCRs (somatostatin, melanocortin) from cilia explains obesity mechanism — distinguish from simple dietary obesity",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- CEP290 — Centrosomal Protein 290 / LCA10 / Joubert ----------------------
    {
        "gene": "CEP290",
        "alt_name": "CEP290 (Joubert/LCA10/NPHP6/Meckel)",
        "protein": (
            "CEP290 -- 12q21.32 AR -- CEP290-2479aa -- "
            "Joubert-JBTS5-Molar-Tooth-Sign-PATHOGNOMONIC-MRI-Brainstem -- "
            "LCA10-c.2991+1655A>G-Deep-Intronic-Most-Common-LCA-Variant -- "
            "Meckel-Gruber-MKS4-Lethal-Neural-Tube-Polydactyly-Cystic-Kidneys -- "
            "NPHP6-Senior-Løken-Renal-Retinal -- "
            "Sepofarsen-Antisense-IVS26-Mutation-Phase2"
        ),
        "locus": "12q21.32",
        "protein_size": "2479 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "LCA10: congenital/neonatal visual impairment (c.2991+1655A>G intronic); "
            "Joubert: hypotonia + ataxia from infancy; molar tooth sign on neonatal MRI; "
            "Meckel-Gruber: lethal in utero or neonatal (neural tube + polydactyly + cystic kidneys); "
            "NPHP6/Senior-Løken: nephronophthisis + retinal dystrophy — renal failure by 2nd decade"
        ),
        "key_biomarker": (
            "MRI brainstem: molar tooth sign (MTS) = elongated/thickened superior cerebellar peduncles "
            "+ deepened interpeduncular fossa + vermis hypoplasia — PATHOGNOMONIC Joubert; "
            "ERG: LCA10 — flat/severely reduced; "
            "renal function: GFR + urine concentrating ability (NPHP6); "
            "molecular: c.2991+1655A>G deep intronic variant in IVS26 — creates cryptic splice site; "
            "CRITICAL: STANDARD exon sequencing MISSES this variant — request DEEP INTRONIC testing"
        ),
        "pathognomonic": (
            "Molar Tooth Sign on MRI = PATHOGNOMONIC Joubert spectrum (any causative gene); "
            "CEP290 LCA10: nystagmus + fixed pupils + absent ERG in neonate — LCA until proven otherwise; "
            "c.2991+1655A>G in trans with any pathogenic CEP290 variant = LCA10 diagnosis; "
            "Meckel-Gruber triad: encephalocele (posterior) + polydactyly (postaxial) + polycystic kidneys = lethal ciliopathy; "
            "DISTINGUISH from RPGR LCA: RPGR X-linked; no renal/cerebellar; male-predominant"
        ),
        "treatment": (
            "LCA10 c.2991+1655A>G: sepofarsen (antisense oligonucleotide) — Phase 2/3 intravitreal injection; "
            "reduces aberrant splicing; sustained visual improvement in trials; "
            "Joubert: supportive — physiotherapy, respiratory support (apnoea monitoring); "
            "renal: nephroprotective; transplant if ESRD; "
            "AVXS-201 (AAV gene therapy CEP290) — early trials for LCA10; "
            "Meckel-Gruber: no curative therapy; lethal — family counselling + prenatal diagnosis essential; "
            "Genetic counselling: ALL subtypes AR; prenatal exome if prior affected child"
        ),
        "critical_flags": [
            "CEP290-INTRONIC-VARIANT-MISSED: c.2991+1655A>G is DEEP INTRONIC (IVS26); NOT detected by standard exome/panel; request RNA splicing analysis or targeted deep-intronic assay; most common single LCA variant worldwide",
            "CEP290-MOLAR-TOOTH-MRI-MANDATORY: any infant with hypotonia + ataxia + nystagmus requires brain MRI with dedicated brainstem sequences; MTS absence does NOT exclude CEP290 disease (other subtypes exist)",
            "CEP290-PHENOTYPIC-SPECTRUM-SAME-GENE: Joubert (mild) → Senior-Løken (renal+retinal) → Meckel-Gruber (lethal) from different variants in SAME gene — genotype-phenotype only partial; clinical vigilance for renal in all CEP290 Joubert",
            "CEP290-SEPOFARSEN-ELIGIBILITY: only c.2991+1655A>G homozygous or compound het eligible; confirm BOTH alleles before trial enrolment",
            "CEP290-RENAL-SURVEILLANCE: ALL CEP290 patients (not just NPHP6 label) need annual renal function; subclinical nephronophthisis present in many Joubert CEP290 patients",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- NPHP1 — Nephrocystin-1 / Nephronophthisis 1 ----------------------------
    {
        "gene": "NPHP1",
        "alt_name": "NPHP1 (Nephronophthisis type 1)",
        "protein": (
            "NPHP1 -- 2q13 AR -- NPHP1-732aa -- "
            "Nephronophthisis-Type1-Most-Common-Genetic-ESRD-Children -- "
            "Homozygous-2q13-Deletion-85pct-MLPA-Required-NOT-Standard-Sequencing -- "
            "Corticomedullary-Cysts-Tubular-Basement-Membrane-Disruption -- "
            "Senior-Løken-If-Retinal-Involvement -- "
            "Joubert-JBTS4-If-Cerebellar-Vermis-Involvement"
        ),
        "locus": "2q13",
        "protein_size": "732 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Isolated nephronophthisis: renal failure median age 13 yr (NPHP1 = juvenile type); "
            "polyuria + polydipsia FIRST symptoms (concentrating defect precedes GFR decline); "
            "anaemia out of proportion to GFR loss; "
            "Senior-Løken: simultaneous retinal dystrophy + nephronophthisis; "
            "Joubert-NPHP1: cerebellar involvement from infancy"
        ),
        "key_biomarker": (
            "Renal ultrasound: normal or small kidneys (NOT enlarged as in ADPKD); "
            "corticomedullary cysts (not always visible early); loss of corticomedullary differentiation; "
            "tubular basement membrane disruption on biopsy (not polycystic on biopsy); "
            "MOLECULAR: homozygous deletion 2q13 ~85% of NPHP1 — MLPA required (NOT sequencing); "
            "urine osmolality: impaired concentrating ability (max <300 mOsm early); "
            "GFR decline: predictable — ESRD median age 13 yr"
        ),
        "pathognomonic": (
            "Child with ESRD + normal-sized kidneys + polyuria + anaemia = NPHP until proven otherwise; "
            "corticomedullary cysts + tubular basement membrane disruption on biopsy = NPHP; "
            "homozygous 2q13 deletion detected by MLPA = NPHP1 confirmed; "
            "DISTINGUISH from ADPKD: ADPKD = enlarged cystic kidneys + family history + adult ESRD; "
            "DISTINGUISH from MUC1-ADTKD: ADTKD = adult onset; no cysts visible; normal urine"
        ),
        "treatment": (
            "No disease-modifying therapy approved; nephroprotective: "
            "ACE-i/ARB for proteinuria; strict BP control (<50th percentile); "
            "anaemia: erythropoietin + iron supplementation early; "
            "avoid nephrotoxins (NSAIDs, contrast, aminoglycosides); "
            "renal transplantation — NPHP does NOT recur post-transplant (ciliopathy, not immune); "
            "growth hormone if height SDS < -2; "
            "regular ophthalmology for Senior-Løken patients; "
            "MLPA screening of siblings — 25% risk (AR)"
        ),
        "critical_flags": [
            "NPHP1-MLPA-MANDATORY: homozygous 2q13 deletion in ~85%; standard WES/gene panel reads THROUGH the deletion as homozygous reference — MLPA or SNP-array required; missed diagnosis = delayed transplant listing",
            "NPHP1-KIDNEYS-NOT-ENLARGED: renal ultrasound normal-to-small kidneys; cysts may not be visible until late; normal ultrasound does NOT exclude NPHP — tubular function tests + biopsy if suspicious",
            "NPHP1-POLYURIA-FIRST: urine concentrating defect is the FIRST abnormality (before GFR decline); nocturia/enuresis in a school-age child with no UTI = check urine osmolality after overnight fast",
            "NPHP1-TRANSPLANT-NON-RECURRENT: NPHP is structural ciliary defect, NOT immune-mediated; does NOT recur in renal transplant; transplant should not be delayed by concern about recurrence",
            "NPHP1-EXTRARENAL-SURVEILLANCE: all NPHP1 patients need ophthalmology (Senior-Løken) and brain MRI (Joubert) at diagnosis; extrarenal involvement changes management",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- RPGR — Retinitis Pigmentosa GTPase Regulator ---------------------------
    {
        "gene": "RPGR",
        "alt_name": "RPGR (X-linked RP / XLRP)",
        "protein": (
            "RPGR -- Xp11.3 XLR -- RPGR-815aa -- "
            "X-linked-Retinitis-Pigmentosa-Most-Common-Severe-RP -- "
            "RPGR-ORF15-Exon-70-75pct-All-XLRP-Hotspot-Specific-Sequencing-Required -- "
            "Female-Carriers-20pct-Symptomatic-Variable-X-Inactivation -- "
            "Cone-Rod-Dystrophy-Variant-RPGR -- "
            "Combined-RP-Recurrent-Respiratory-Infections-Ciliary-Dyskinesia"
        ),
        "locus": "Xp11.3",
        "protein_size": "815 aa",
        "inheritance": "XLR",
        "age_of_onset": (
            "Typical XLRP: nyctalopia (night blindness) onset 1st-2nd decade; "
            "visual field constriction — tunnel vision; central vision preserved until 4th-5th decade; "
            "cone-rod dystrophy variant: central vision loss FIRST (more like ACHM); "
            "carrier females: 20% develop rod-cone dystrophy (X-inactivation skewing); "
            "respiratory: recurrent sinopulmonary infections in subset (ciliary function)"
        ),
        "key_biomarker": (
            "ERG: markedly reduced rod responses first; cone responses later; "
            "fundus: bone-spicule pigmentation, disc pallor, attenuated arterioles; "
            "OCT: RPE/IS-OS disruption; macular involvement late (vs central in cone-rod); "
            "visual field: ring scotoma → tunnel; "
            "MOLECULAR: ORF15 exon sequencing required — poly-purine hotspot; "
            "standard gene panels may NOT sequence ORF15 adequately — confirm with lab; "
            "carrier females: obligate carrier testing (maternal + paternal cascade)"
        ),
        "pathognomonic": (
            "X-linked pattern (males affected, females carriers, no male-to-male transmission) + RP = XLRP; "
            "ORF15 frameshift/nonsense in RPGR + X-linked RP = RPGR-XLRP confirmed; "
            "DISTINGUISH from PRPF31 XLRP: PRPF31 can have normal ERG carriers; different penetrance; "
            "carrier females with sectoral RP (asymmetric, sector-shaped): RPGR carrier until proven otherwise; "
            "combined RP + bronchiectasis: RPGR ciliary dyskinesia variant (primary ciliary dyskinesia overlap)"
        ),
        "treatment": (
            "Gene therapy: RPGR-ORF15 AAV (subretinal injection) — Phase 2/3 trials (botaretigene sparoparvovec); "
            "no approved therapy yet; "
            "visual rehabilitation: low-vision aids; orientation + mobility; "
            "vitamin A palmitate 15,000 IU/day — modest ERG preservation (avoid in liver disease); "
            "avoid vitamin E high-dose (accelerates decline in some RP); "
            "sun protection: amber/dark lenses reduce photoreceptor stress; "
            "carrier female surveillance: annual ERG + visual fields (20% progression risk); "
            "respiratory: chest physiotherapy if ciliary dyskinesia variant"
        ),
        "critical_flags": [
            "RPGR-ORF15-NOT-SEQUENCED: ORF15 exon contains a poly-purine/poly-pyrimidine repeat hotspot; difficult Sanger sequencing; many NGS panels under-cover this region; request specific lab confirmation of ORF15 coverage",
            "RPGR-CARRIER-FEMALE-RISK: 20% of female carriers are symptomatic; carrier females need annual ophthalmology + ERG; do not dismiss visual symptoms in a presumed carrier",
            "RPGR-CONE-ROD-VARIANT: some RPGR variants cause cone-rod dystrophy (central vision first) rather than classic rod-cone; misdiagnosed as macular dystrophy; X-linked pattern should trigger RPGR testing",
            "RPGR-GENE-THERAPY-WINDOW: subretinal gene therapy being tested; vision must be sufficient for benefit; do NOT delay referral to trials until vision is too poor to measure",
            "RPGR-RESPIRATORY-CILIARY: RPGR ciliary dyskinesia — recurrent sinopulmonary infections + bronchiectasis + situs inversus; nasal nitric oxide (low) + ciliary biopsy; treat as PCD",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- AHI1 — Abelson Helper Integration site 1 / Joubert 3 -------------------
    {
        "gene": "AHI1",
        "alt_name": "AHI1 (Joubert Syndrome type 3 / JBTS3)",
        "protein": (
            "AHI1 -- 6q23.3 AR -- AHI1-1196aa -- "
            "Joubert-Syndrome-JBTS3-Molar-Tooth-Sign-REQUIRED -- "
            "High-Rate-Retinal-Involvement-80pct-MORE-Than-Other-Joubert-Genes -- "
            "Nephronophthisis-30-40pct -- "
            "R830W-Ashkenazi-Jewish-Founder-Cascade-Screening-Indicated -- "
            "Jouberin-Scaffold-Protein-Transition-Zone-Cilia"
        ),
        "locus": "6q23.3",
        "protein_size": "1196 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Joubert hypotonia + ataxia: from infancy; "
            "molar tooth sign: present from birth (detectable on neonatal MRI); "
            "retinal dystrophy: AHI1 has HIGHER retinal involvement rate (80%) vs other JBTS genes; "
            "rod-cone dystrophy onset 1st-2nd decade; "
            "nephronophthisis: renal failure in 30-40%; median ESRD 2nd decade"
        ),
        "key_biomarker": (
            "MRI brainstem: molar tooth sign (MTS) — REQUIRED for Joubert classification; "
            "ERG: rod-cone dystrophy in 80% — AHI1-JBTS has highest retinal rate of all JBTS genes; "
            "renal ultrasound + GFR: nephronophthisis in ~35%; "
            "molecular: AHI1 pathogenic variants; R830W (c.2488C>T) in Ashkenazi Jewish population; "
            "ophthalmology: nystagmus + poor visual behaviour from infancy"
        ),
        "pathognomonic": (
            "Molar tooth sign + nystagmus + hypotonia + retinal dystrophy = AHI1 Joubert until proven otherwise; "
            "AHI1 = Joubert gene with highest retinal penetrance — if Joubert + severe retinal = AHI1 priority; "
            "R830W homozygous or compound het in Ashkenazi Jewish child with Joubert = AHI1-JBTS3 confirmed; "
            "DISTINGUISH from CEP290 Joubert: CEP290 more variable; AHI1 more retinal-predominant; "
            "DISTINGUISH from INPP5E Joubert: INPP5E less retinal; more cerebellar-predominant"
        ),
        "treatment": (
            "No disease-modifying therapy; supportive: "
            "physiotherapy for hypotonia/ataxia; "
            "occupational therapy; speech therapy; "
            "respiratory: apnoea monitoring in infancy (episodic hyperpnoea); "
            "ophthalmology: low-vision aids; gene therapy trials emerging; "
            "nephrology: nephroprotective + renal transplant if ESRD (AHI1 does NOT recur); "
            "Ashkenazi Jewish population: R830W carrier frequency ~1/93 — population screening possible; "
            "family cascade testing for AHI1 variants in affected pedigrees"
        ),
        "critical_flags": [
            "AHI1-RETINAL-HIGHEST-RISK: AHI1-JBTS has 80% retinal involvement — higher than any other JBTS gene; ALL AHI1 patients need formal ERG at diagnosis regardless of visual symptoms; early diagnosis enables trials enrolment",
            "AHI1-MTS-REQUIRED: molar tooth sign is required for Joubert classification; without MTS, diagnosis is not Joubert — consider other cerebellar ataxia genes",
            "AHI1-R830W-ASHKENAZI: R830W is a founder variant in Ashkenazi Jewish population (~1/93 carrier frequency); cascade testing in extended AJ family after a proband identified; pre-conception carrier testing offered",
            "AHI1-RENAL-ANNUAL: 30-40% develop nephronophthisis; annual GFR + urine osmolality + renal ultrasound from diagnosis; subclinical renal involvement may predate clinical detection by years",
            "AHI1-APNOEA-INFANCY: episodic hyperpnoea/apnoea is a Joubert hallmark in neonates; cardiorespiratory monitoring for first 6-12 months; respiratory failure is leading cause of early death",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- ALMS1 — Alström Syndrome protein 1 -------------------------------------
    {
        "gene": "ALMS1",
        "alt_name": "ALMS1 (Alström Syndrome)",
        "protein": (
            "ALMS1 -- 2p13.1 AR -- ALMS1-4169aa -- "
            "Alstrom-Syndrome-Unique-Ciliopathy-NO-Polydactyly-NO-Cognitive-Impairment -- "
            "Cone-Rod-Dystrophy-Nystagmus-First-Year-Of-Life -- "
            "Dilated-Cardiomyopathy-Infancy-FIRST-Presentation-May-Resolve -- "
            "Type2-Diabetes-Childhood-Plus-Obesity-Plus-Sensorineural-Hearing-Loss -- "
            "Distinguish-BBS-Normal-Intelligence-No-Extra-Digits-Cardiomyopathy"
        ),
        "locus": "2p13.1",
        "protein_size": "4169 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Cone-rod dystrophy: photophobia + nystagmus within first year; "
            "dilated cardiomyopathy: can present in infancy (often resolves by age 5 spontaneously); "
            "type 2 diabetes: typically childhood/adolescence; "
            "sensorineural hearing loss: childhood; "
            "obesity: infancy onwards; "
            "hepatic steatosis → fibrosis: childhood to adulthood"
        ),
        "key_biomarker": (
            "ERG: cone responses severely reduced early (cone-rod, NOT rod-cone — opposite of BBS); "
            "echocardiogram: dilated cardiomyopathy (EF reduced); "
            "audiogram: sensorineural hearing loss bilateral; "
            "fasting glucose + insulin: type 2 diabetes / insulin resistance; "
            "liver enzymes + ultrasound: steatosis/fibrosis; "
            "molecular: ALMS1 large gene — next-generation sequencing + CNV; "
            "urinary albumin:creatinine — progressive nephropathy"
        ),
        "pathognomonic": (
            "Infant with photophobia + nystagmus + dilated cardiomyopathy = Alström FIRST; "
            "triad: cone-rod dystrophy + dilated cardiomyopathy + type 2 diabetes in childhood = Alström; "
            "DISTINGUISH from BBS: Alström has NO polydactyly, NO cognitive impairment, NO renal structural anomalies; "
            "Alström cardiomyopathy may SPONTANEOUSLY RESOLVE (2nd cardiomyopathy in adolescence re-emerges); "
            "DISTINGUISH from Wolfram: Wolfram = optic atrophy + DI + DM + deafness — no retinopathy"
        ),
        "treatment": (
            "Cone-rod dystrophy: no curative therapy; low-vision aids; UV protection; "
            "cardiomyopathy infancy: standard HF therapy (diuretics, ACE-i); often resolves by age 5; "
            "second cardiomyopathy in adolescence: early HF therapy; cardiac transplant in refractory; "
            "type 2 diabetes: metformin + GLP-1 agonists + insulin as needed; "
            "hearing: hearing aids; cochlear implant if profound; "
            "hepatic: NASH management — weight reduction + vitamin E (NASH-CRN score); "
            "renal: annual ACR + GFR; ACE-i/ARB for proteinuria; "
            "multidisciplinary: ophthalmology + cardiology + endocrinology + nephrology + audiology"
        ),
        "critical_flags": [
            "ALMS1-CARDIOMYOPATHY-TWO-EPISODES: dilated cardiomyopathy in infancy often resolves — may falsely reassure; second cardiomyopathy episode in adolescence is life-threatening; lifelong cardiac surveillance mandatory",
            "ALMS1-NOT-BBS: Alström is often misdiagnosed as BBS due to retinal dystrophy + obesity overlap; KEY DIFFERENCE = Alström has cone-rod (central first) NOT rod-cone; NO polydactyly; normal intelligence; cardiomyopathy",
            "ALMS1-CONE-ROD-NOT-ROD-CONE: cone responses lost FIRST (photophobia, central vision loss first); opposite to BBS/NPHP; impacts visual rehabilitation planning and trial eligibility",
            "ALMS1-LARGE-GENE-SEQUENCING: ALMS1 is 23 exons; 4169 aa; large gene — ensure sequencing covers full gene including large exon 16 (highest variant density); CNV analysis included",
            "ALMS1-MULTI-ORGAN-SURVEILLANCE: annual surveillance protocol = ophthalmology + echo + audiogram + fasting glucose + HbA1c + liver enzymes + renal function; missing any organ risks silent progression",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- KIF7 — Kinesin Family Member 7 -----------------------------------------
    {
        "gene": "KIF7",
        "alt_name": "KIF7 (Acrocallosal / Hydrolethalus / JBTS12)",
        "protein": (
            "KIF7 -- 15q26.1 AR -- KIF7-1343aa -- "
            "Acrocallosal-Syndrome-ACLS-Polydactyly-Corpus-Callosum-Agenesis-Facial-Dysmorphism -- "
            "Hydrolethalus-Syndrome2-Lethal-Hydrocephaly-Polydactyly -- "
            "Joubert-JBTS12-Molar-Tooth-Sign-Milder-End-Spectrum -- "
            "Hedgehog-Pathway-Ciliary-Gatekeeper-KIF7-Null-Constitutive-Gli-Activation -- "
            "Phenotypic-Spectrum-Lethal-To-Mild-By-Variant-Severity"
        ),
        "locus": "15q26.1",
        "protein_size": "1343 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Hydrolethalus: lethal in utero/neonatal — severe hydrocephaly + polydactyly + cardiac; "
            "Acrocallosal: neonatal — polydactyly + corpus callosum agenesis + facial dysmorphism; "
            "JBTS12: childhood — molar tooth sign + variable intellectual disability; "
            "Spectrum severity correlates with variant type: "
            "null/truncating → hydrolethalus; missense → Joubert/ACLS"
        ),
        "key_biomarker": (
            "Prenatal/neonatal MRI: hydrocephaly (hydrolethalus); corpus callosum agenesis (ACLS); "
            "molar tooth sign (JBTS12); "
            "hands/feet: polydactyly (postaxial hallucal/preaxial digital); "
            "facial: hypertelorism, short nose, cleft lip/palate (ACLS); "
            "molecular: KIF7 variants — truncating more severe; "
            "hedgehog pathway: GLI processing assay (research)"
        ),
        "pathognomonic": (
            "Hydrolethalus: macrohydrocephaly + polydactyly + absent midbrain structure + cardiac = KIF7 or HYLS1; "
            "Acrocallosal: polydactyly (pre + postaxial) + corpus callosum agenesis + facial = ACLS (KIF7 or GLI3); "
            "JBTS12: molar tooth sign + polydactyly + intellectual disability without severe retinal = KIF7 priority; "
            "DISTINGUISH from GLI3: GLI3 = Greig cephalopolysyndactyly or Pallister-Hall; "
            "DISTINGUISH from other JBTS genes: KIF7 has higher polydactyly rate; Hedgehog involvement"
        ),
        "treatment": (
            "Hydrolethalus: palliative / lethal; prenatal diagnosis + family counselling; "
            "Acrocallosal: polydactyly — surgical correction; "
            "corpus callosum agenesis: neurodevelopmental support; seizure management; "
            "JBTS12: physiotherapy + occupational therapy + special education; "
            "apnoea monitoring in Joubert spectrum; "
            "ophthalmology (retinal involvement in ~30% JBTS12 KIF7); "
            "hedgehog pathway: theoretical SMO inhibitors (vismodegib) — NOT approved for ciliopathy; "
            "genetic counselling: AR — 25% recurrence; prenatal diagnosis feasible (exome)"
        ),
        "critical_flags": [
            "KIF7-PHENOTYPIC-SPECTRUM-WIDE: same AR gene causes lethal hydrolethalus, acrocallosal syndrome, and mild Joubert; variant type (truncating vs missense) is the strongest predictor; cannot give prognosis from gene name alone",
            "KIF7-HEDGEHOG-CHECKPOINT: KIF7 is the ANTEROGRADE motor of the Hedgehog pathway ciliary checkpoint; null KIF7 = constitutive GLI2/3 activation; Hedgehog overactivation causes polydactyly + neural tube phenotypes",
            "KIF7-GLI3-DDX: acrocallosal syndrome also caused by GLI3 mutations; GLI3 should be sequenced in ACLS; KIF7 vs GLI3 ACLS: similar phenotype; molecular confirmation required",
            "KIF7-PRENATAL-DIAGNOSIS: hydrolethalus and ACLS detectable by detailed fetal anomaly ultrasound (level 2); fetal MRI helps for corpus callosum agenesis; offer exome to parents of affected fetus",
            "KIF7-RETINAL-SURVEILLANCE: KIF7-JBTS12 has ~30% retinal dystrophy rate; annual ophthalmology + ERG from diagnosis; lower than AHI1 but not negligible",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- DYNC2H1 — Dynein Cytoplasmic 2 Heavy Chain 1 ---------------------------
    {
        "gene": "DYNC2H1",
        "alt_name": "DYNC2H1 (Short-Rib Thoracic Dysplasia / Jeune / SRTD3)",
        "protein": (
            "DYNC2H1 -- 11q22.3 AR -- DYNC2H1-4307aa -- "
            "Short-Rib-Thoracic-Dysplasia-Jeune-SRTD3-Narrow-Thorax-Neonatal-Respiratory-Failure -- "
            "Retrograde-Intraflagellar-Transport-IFT-A -- "
            "Polydactyly-Variable-Pre-Postaxial -- "
            "Renal-Hepatic-Retinal-Involvement-Survivors -- "
            "Most-Common-Gene-SRTD-45pct-All-SRTD-Cases"
        ),
        "locus": "11q22.3",
        "protein_size": "4307 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "Neonatal: narrow thorax → respiratory failure is the primary lethal outcome; "
            "perinatal: prenatal ultrasound detects short ribs + narrow thorax + limb shortening; "
            "survivors: progressive renal insufficiency + liver fibrosis + retinal dystrophy; "
            "polydactyly: variable (present at birth if present)"
        ),
        "key_biomarker": (
            "Prenatal/neonatal X-ray: short horizontal ribs + narrow thorax + trident acetabulum; "
            "thoracic circumference: below 3rd percentile at birth; "
            "chest X-ray: 'champagne cork' metaphyseal changes; "
            "molecular: DYNC2H1 pathogenic variants — most common SRTD gene (~45%); "
            "renal ultrasound (survivors): nephronophthisis/cysts; "
            "liver biopsy: biliary dysgenesis/fibrosis; "
            "ERG (survivors): rod-cone dystrophy in subset"
        ),
        "pathognomonic": (
            "Narrow thorax + short ribs + limb shortening on prenatal ultrasound = skeletal ciliopathy (SRTD/ATD); "
            "DYNC2H1 = most common SRTD gene; short ribs + narrow chest + trident pelvis + polydactyly = Jeune ATD; "
            "respiratory failure in neonate with radiological short-rib dysplasia = SRTD until proven otherwise; "
            "DISTINGUISH from thanatophoric dysplasia: FGFR3 mutation; telephone-handset femora; LETHAL (not survivable); "
            "DISTINGUISH from Ellis-van Creveld: EVC gene; mesomelic shortening; cardiac defects prominent"
        ),
        "treatment": (
            "Neonatal respiratory: aggressive ventilatory support (CPAP/mechanical ventilation); "
            "thoracic expansion surgery: lateral thoracic expansion (VEPTR/Titanium rib device); "
            "multiple surgical expansions needed as child grows (every 1-2 yr); "
            "survivors: multidisciplinary — nephrology + hepatology + ophthalmology; "
            "renal: nephroprotective + transplant if ESRD (does NOT recur); "
            "liver: ursodeoxycholic acid; transplant if hepatic failure; "
            "retinal: low-vision aids; gene therapy trials emerging; "
            "genetic counselling: AR — 25% recurrence; prenatal diagnosis by exome or targeted"
        ),
        "critical_flags": [
            "DYNC2H1-RESPIRATORY-LETHAL: 30-50% of SRTD/Jeune infants die from respiratory failure in neonatal period; thoracic expansion surgery (VEPTR) must be planned prenatally; refer to specialist centre before delivery",
            "DYNC2H1-MOST-COMMON-SRTD: DYNC2H1 accounts for ~45% of all SRTD cases; sequence DYNC2H1 first in any skeletal short-rib dysplasia; large gene (4307aa) — requires comprehensive sequencing",
            "DYNC2H1-IFT-A-RETROGRADE: DYNC2H1 is the heavy chain of IFT dynein-2 (retrograde IFT-A); loss → ciliary tip accumulation of IFT-B components; cilia are short/bulgy — detectable on electron microscopy",
            "DYNC2H1-SURVIVORS-MULTI-ORGAN: children surviving the neonatal respiratory period face progressive renal + hepatic + retinal involvement; lifelong surveillance essential; do not falsely reassure parents after respiratory stabilisation",
            "DYNC2H1-VEPTR-CENTRES: lateral thoracic expansion (VEPTR) requires specialist paediatric orthopaedic centre; multiple operations needed; outcome data: 70% of operated children survive to adulthood with adequate pulmonary function",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _make_patient(gene_data: dict, idx: int) -> dict:
    rng = random.Random(gene_data["seed"] * 1000 + idx)
    gene = gene_data["gene"]
    inh = gene_data["inheritance"]
    locus = gene_data["locus"]
    prot_size = gene_data["protein_size"]

    age_map = {
        "BBS1": (4, 18),
        "CEP290": (0, 5),
        "NPHP1": (6, 20),
        "RPGR": (8, 30),
        "AHI1": (0, 6),
        "ALMS1": (0, 4),
        "KIF7": (0, 3),
        "DYNC2H1": (0, 1),
    }
    a_min, a_max = age_map.get(gene, (2, 25))
    age = rng.randint(a_min, a_max)

    sex_choices = {"XLR": (["M"] * 9 + ["F"]), "XLD": (["F"] * 7 + ["M"] * 3)}
    sex = rng.choice(sex_choices.get(inh, ["M", "F"]))

    severity_weights = {
        "BBS1": ["moderate", "severe", "moderate", "mild", "severe"],
        "CEP290": ["severe", "severe", "moderate", "critical"],
        "NPHP1": ["moderate", "severe", "moderate"],
        "RPGR": ["moderate", "severe", "mild", "moderate"],
        "AHI1": ["moderate", "severe", "moderate"],
        "ALMS1": ["moderate", "severe", "moderate"],
        "KIF7": ["severe", "critical", "moderate", "moderate"],
        "DYNC2H1": ["critical", "severe", "severe", "critical"],
    }
    severity = rng.choice(severity_weights.get(gene, ["moderate", "severe", "mild"]))

    biomarker_map = {
        "BBS1": {
            "rod_ERG_uV": round(rng.uniform(0.2, 8.0), 1),
            "BMI_SDS": round(rng.uniform(2.0, 4.5), 1),
            "renal_GFR": round(rng.uniform(25, 100), 0),
        },
        "CEP290": {
            "ERG_flat": rng.choice([True, True, False]),
            "MTS_present": True,
            "GFR": round(rng.uniform(20, 110), 0),
        },
        "NPHP1": {
            "GFR_mL_min": round(rng.uniform(5, 60), 0),
            "urine_osm_max": round(rng.uniform(150, 400), 0),
            "del_2q13_homozygous": rng.choice([True, True, True, False]),
        },
        "RPGR": {
            "rod_ERG_uV": round(rng.uniform(0.1, 10.0), 1),
            "VF_remaining_deg": round(rng.uniform(2, 30), 0),
            "ORF15_variant": rng.choice([True, True, False]),
        },
        "AHI1": {
            "rod_ERG_uV": round(rng.uniform(0.2, 6.0), 1),
            "MTS_present": True,
            "GFR": round(rng.uniform(15, 110), 0),
        },
        "ALMS1": {
            "cone_ERG_uV": round(rng.uniform(0.1, 5.0), 1),
            "EF_pct": round(rng.uniform(20, 65), 0),
            "HbA1c_pct": round(rng.uniform(6.5, 12.0), 1),
        },
        "KIF7": {
            "MTS_present": rng.choice([True, False]),
            "CC_agenesis": rng.choice([True, True, False]),
            "polydactyly": True,
        },
        "DYNC2H1": {
            "thoracic_circ_pc": round(rng.uniform(1, 10), 0),
            "rib_count_visible": rng.randint(6, 10),
            "ventilated": rng.choice([True, True, False]),
        },
    }
    biomarkers = biomarker_map.get(gene, {})

    tx_map = {
        "BBS1": rng.choice(["Low-vision-aids + renal-surveillance", "GFR-decline → ACE-i", "Testosterone-replacement + low-vision"]),
        "CEP290": rng.choice(["Sepofarsen-eligible (IVS26)", "Supportive-Joubert + renal-transplant", "Meckel-lethal-palliative"]),
        "NPHP1": rng.choice(["Renal-transplant-listed", "ACE-i + erythropoietin", "MLPA-confirmed → nephrologist"]),
        "RPGR": rng.choice(["AAV-gene-therapy-trial (ORF15)", "Vitamin-A palmitate + low-vision", "Carrier-female-surveillance"]),
        "AHI1": rng.choice(["Physiotherapy + low-vision", "Renal-transplant + ophthalmology", "R830W-AJ-cascade-screening"]),
        "ALMS1": rng.choice(["GLP-1 + hearing-aid + cardiac-surveillance", "VEPTR-not-applicable + low-vision", "Liver-NASH + renal-ACE-i"]),
        "KIF7": rng.choice(["Surgical-polydactyly + neurodevelopment", "Palliative (hydrolethalus)", "Physiotherapy + seizure-management"]),
        "DYNC2H1": rng.choice(["VEPTR-thoracic-expansion", "CPAP + VEPTR-planned", "Palliative (lethal-thorax)"]),
    }
    treatment = tx_map.get(gene, "Supportive")

    flag_map = {
        "BBS1": rng.choice(["Renal-silent-progression", "M1L-founder-assay-needed", "Retinal-irreversible-trial-enrol"]),
        "CEP290": rng.choice(["Intronic-variant-deep-sequencing", "Sepofarsen-trial-eligible", "Renal-surveillance-mandatory"]),
        "NPHP1": rng.choice(["MLPA-deletion-not-sequencing", "Polyuria-early-sign", "Transplant-non-recurrent"]),
        "RPGR": rng.choice(["ORF15-coverage-confirm", "Carrier-female-20pct-risk", "Gene-therapy-trial-window"]),
        "AHI1": rng.choice(["Retinal-80pct-ERG-mandatory", "Apnoea-monitoring-infancy", "R830W-AJ-screen"]),
        "ALMS1": rng.choice(["Cardiomyopathy-2-episodes", "Cone-rod-not-rod-cone", "Multi-organ-annual-panel"]),
        "KIF7": rng.choice(["Phenotypic-spectrum-wide", "Prenatal-diagnosis-feasible", "Hydrolethalus-lethal"]),
        "DYNC2H1": rng.choice(["Respiratory-VEPTR-centre", "Survivors-multi-organ", "IFT-A-retrograde-ciliary"]),
    }
    flag = flag_map.get(gene, "Surveillance-required")

    return {
        "patient_id": f"{gene}-{idx+1:03d}",
        "gene": gene,
        "locus": locus,
        "protein_size": prot_size,
        "inheritance": inh,
        "age_at_diagnosis": age,
        "sex": sex,
        "severity": severity,
        "biomarkers": biomarkers,
        "treatment": treatment,
        "critical_flag": flag,
        "alt_name": gene_data["alt_name"],
    }


def _all_patients() -> list:
    out = []
    for gd in CILIOPATHY_GENES:
        for i in range(40):
            out.append(_make_patient(gd, i))
    return out


# ── Public API ──────────────────────────────────────────────────────────────

def overview() -> dict:
    patients = _all_patients()
    total = len(patients)
    by_gene = {}
    for p in patients:
        g = p["gene"]
        by_gene.setdefault(g, {"count": 0, "severities": []})
        by_gene[g]["count"] += 1
        by_gene[g]["severities"].append(p["severity"])
    gene_summary = []
    for gd in CILIOPATHY_GENES:
        g = gd["gene"]
        rec = by_gene.get(g, {"count": 0, "severities": []})
        sevs = rec["severities"]
        sev_counts = {s: sevs.count(s) for s in set(sevs)}
        gene_summary.append({
            "gene": g,
            "alt_name": gd["alt_name"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance": gd["inheritance"],
            "patient_count": rec["count"],
            "severity_distribution": sev_counts,
        })
    return {
        "atlas": "Hereditary-Ciliopathy-Atlas",
        "subtitle": "Complete-8-Gene-Primary-Cilia-Disorder-Atlas",
        "total_patients": total,
        "genes_covered": len(CILIOPATHY_GENES),
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
        "inheritance_modes": sorted({gd["inheritance"] for gd in CILIOPATHY_GENES}),
        "gene_summary": gene_summary,
        "key_clinical_facts": [
            "BBS1: most common BBS gene (~23%); rod-cone dystrophy ALWAYS first; M1L Caucasian founder",
            "CEP290: c.2991+1655A>G deep intronic IVS26 — most common LCA variant; MISSED by standard exome",
            "NPHP1: homozygous 2q13 deletion ~85% — MLPA required; most common genetic childhood ESRD",
            "RPGR-ORF15: 70-75% of all XLRP; ORF15 hotspot requires specific sequencing coverage",
            "AHI1: highest retinal penetrance in Joubert (80%); R830W Ashkenazi founder",
            "ALMS1: NO polydactyly, NO cognitive impairment; cone-rod (not rod-cone); cardiomyopathy 2 episodes",
            "KIF7: Hedgehog ciliary gatekeeper; phenotypic spectrum = lethal (hydrolethalus) → mild (JBTS12)",
            "DYNC2H1: most common SRTD gene (45%); narrow thorax = neonatal respiratory failure",
        ],
    }


def breakdown() -> dict:
    patients = _all_patients()
    per_gene = {}
    for gd in CILIOPATHY_GENES:
        g = gd["gene"]
        gene_pts = [p for p in patients if p["gene"] == g]
        ages = [p["age_at_diagnosis"] for p in gene_pts]
        sevs = [p["severity"] for p in gene_pts]
        sexes = [p["sex"] for p in gene_pts]
        per_gene[g] = {
            "gene": g,
            "alt_name": gd["alt_name"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance": gd["inheritance"],
            "patient_count": len(gene_pts),
            "age_stats": {
                "min": min(ages) if ages else 0,
                "max": max(ages) if ages else 0,
                "mean": round(sum(ages) / len(ages), 1) if ages else 0,
            },
            "severity_counts": {s: sevs.count(s) for s in set(sevs)},
            "sex_counts": {s: sexes.count(s) for s in set(sexes)},
            "sample_patients": gene_pts[:3],
            "pathognomonic": gd["pathognomonic"],
            "treatment": gd["treatment"],
            "critical_flags": gd["critical_flags"],
        }
    return {"per_gene_breakdown": per_gene}


def definitions() -> dict:
    defs = {}
    for gd in CILIOPATHY_GENES:
        defs[gd["gene"]] = {
            "gene": gd["gene"],
            "alt_name": gd["alt_name"],
            "locus": gd["locus"],
            "protein": gd["protein"],
            "protein_size": gd["protein_size"],
            "inheritance": gd["inheritance"],
            "age_of_onset": gd["age_of_onset"],
            "key_biomarker": gd["key_biomarker"],
            "pathognomonic": gd["pathognomonic"],
            "treatment": gd["treatment"],
            "critical_flags": gd["critical_flags"],
        }
    return {
        "atlas": "Hereditary-Ciliopathy-Atlas",
        "gene_definitions": defs,
        "glossary": {
            "Ciliopathy": "Disease caused by dysfunction of primary or motile cilia — affecting multiple organs (kidney, retina, brain, skeleton, liver)",
            "Molar-Tooth-Sign-MTS": "MRI brainstem finding in Joubert Syndrome — elongated superior cerebellar peduncles + vermis hypoplasia; PATHOGNOMONIC for Joubert spectrum",
            "BBSome": "Complex of BBS proteins (BBS1/2/4/5/7/8/9/18) required for trafficking of GPCRs and other cargo into and out of cilia",
            "IFT-A-Retrograde": "Intraflagellar transport complex A drives retrograde (tip→base) ciliary traffic; DYNC2H1 is its dynein heavy chain motor",
            "Rod-cone vs Cone-rod": "Rod-cone: rods fail first (night blindness first) = BBS/NPHP; Cone-rod: cones fail first (central vision first) = Alström, RPGR-CRD",
            "NPHP": "Nephronophthisis — autosomal recessive tubulo-interstitial nephritis; small kidneys; corticomedullary cysts; leading genetic ESRD in children",
            "MLPA": "Multiplex Ligation-dependent Probe Amplification — detects copy number variants (deletions/duplications); required for NPHP1 2q13 deletion diagnosis",
            "ORF15": "Open reading frame 15 — unique poly-purine/pyrimidine exon of RPGR; hotspot for 70-75% of XLRP mutations; requires specific sequencing attention",
            "Deep-intronic-variant": "Variant in intron far from splice site that creates a cryptic splice site; NOT detected by exome sequencing; e.g. CEP290 c.2991+1655A>G",
            "SRTD-Jeune": "Short-Rib Thoracic Dysplasia / Jeune Asphyxiating Thoracic Dystrophy — skeletal ciliopathy with narrow chest and neonatal respiratory failure",
            "VEPTR": "Vertical Expandable Prosthetic Titanium Rib — titanium device surgically placed to expand thorax in SRTD; requires multiple expansions during growth",
            "Hedgehog-pathway": "Developmental signalling pathway; KIF7 is the anterograde Hedgehog ciliary transport motor; KIF7 loss → constitutive GLI activation → polydactyly + CNS anomalies",
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = breakdown()
    first_gene = list(bd["per_gene_breakdown"].keys())[0]
    print(json.dumps(bd["per_gene_breakdown"][first_gene], indent=2)[:2000])
    print("\n=== DEFINITIONS (first gene) ===")
    df = definitions()
    print(json.dumps(df["gene_definitions"]["BBS1"], indent=2)[:1000])
