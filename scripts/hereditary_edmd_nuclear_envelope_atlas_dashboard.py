#!/usr/bin/env python3
"""Hereditary-EDMD-Nuclear-Envelope-Atlas — Complete 8-Gene EDMD / Nuclear Envelope Myopathy Spectrum Atlas
(EMD · LMNA · SYNE1 · SYNE2 · TMEM43 · FHL1 · SUN1 · LEMD3).

EMD     (Emerin; 254 aa; Xq28; XLR;
         EDMD1 — X-linked EDMD; emerin absent on IHC (muscle + skin punch biopsy);
         ELBOW FLEXION CONTRACTURES EARLIEST SIGN PATHOGNOMONIC; rigid spine; cardiac mandatory;
         PACEMAKER/ICD — SCD risk; emerin absent = diagnostic;
         seed SEED_BASE+0).
LMNA    (Lamin A/C; 664 aa; 1q22; AD;
         EDMD2 / LGMD1B / DCM-CD / FPLD2 — MOST ALLELIC GENE IN HUMAN DISEASE;
         CARDIAC MANDATORY — DCM + CONDUCTION DEFECTS + SCD RISK; ICD early;
         15+ distinct phenotypes from same gene — laminopathy spectrum;
         seed SEED_BASE+1).
SYNE1   (Nesprin-1; 8797 aa; 6q25.2; AR;
         EDMD4 / ARCA1 — autosomal recessive cerebellar ataxia (Quebec founder);
         LINC complex outer nuclear membrane giant protein; allelic ARCA1;
         seed SEED_BASE+2).
SYNE2   (Nesprin-2; 6885 aa; 14q23.2; AD/AR;
         EDMD5 — LINC complex outer nuclear membrane; KASH domain SUN-binding;
         similar EDMD phenotype to SYNE1 but rarer; cardiac involvement;
         seed SEED_BASE+3).
TMEM43  (LUMA; 400 aa; 3p25.1; AD;
         EDMD7 / ARVC5 — arrhythmogenic right ventricular cardiomyopathy;
         S358L Newfoundland founder — 100% penetrance males — LETHAL ARRHYTHMIA;
         ICD MANDATORY ALL MUTATION CARRIERS — SCD PRIMARY CAUSE OF DEATH;
         seed SEED_BASE+4).
FHL1    (FHL1; 323 aa; Xq26.3; XLR;
         EDMD6 / Scapuloperoneal Myopathy / Reducing Body Myopathy / HCM;
         REDUCING BODY MYOPATHY — eosinophilic inclusions on biopsy PATHOGNOMONIC;
         XLR males most affected; females carriers can manifest HCM;
         seed SEED_BASE+5).
SUN1    (SUN1; 916 aa; 7q32.2; AR/AD;
         LINC complex inner nuclear membrane SUN domain; EDMD-like + DCM;
         LINC complex = nuclear-cytoskeletal force transduction pathway;
         seed SEED_BASE+6).
LEMD3   (MAN1; 922 aa; 12q14.3; AD;
         Buschke-Ollendorff syndrome — OSTEOPOIKILOSIS + DERMATOFIBROSIS LENTICULARIS PATHOGNOMONIC;
         Melorheostosis allelic; TGF-β antagonism loss → bone overgrowth;
         DERMATOLOGY referral mandatory — classic 'white dot' skin lesions;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2198-2205).
"""

import random

SEED_BASE = 2198

EDMD_GENES = [
    # -- EMD — Emerin, X-linked EDMD1 -------------------------------------------------------
    {
        "gene": "EMD",
        "alt_name": (
            "EMD (EMD-254aa-Xq28 / XLR — EDMD1-X-linked-Emery-Dreifuss-Muscular-Dystrophy — "
            "ELBOW-FLEXION-CONTRACTURES-EARLIEST-PATHOGNOMONIC — "
            "Emerin-IHC-ABSENT-Muscle-AND-Skin-Punch-Biopsy-Diagnostic — "
            "PACEMAKER-ICD-MANDATORY-SCD-Risk-Cardiac-Surveillance)"
        ),
        "protein": (
            "EMD -- Xq28 XLR -- EMD-254aa -- "
            "Emerin-LEM-Domain-Inner-Nuclear-Membrane-Protein-29kDa -- "
            "EDMD1-X-linked-Emery-Dreifuss-Muscular-Dystrophy-OMIM-310300 -- "
            "ELBOW-FLEXION-CONTRACTURES-EARLIEST-SIGN-PATHOGNOMONIC-Before-Weakness -- "
            "Rigid-Spine-Cervical-Spine-Contracture-Early -- "
            "CARDIAC-CONDUCTION-DEFECTS-MANDATORY-AF-AV-Block-SCD-Risk -- "
            "Emerin-IHC-ABSENT-Muscle-Biopsy-AND-Skin-Punch-Biopsy-Diagnostic-Screening -- "
            "PACEMAKER-ICD-Mandatory-SCD-Risk-Even-Before-Cardiomyopathy -- "
            "Dilated-Cardiomyopathy-DCM-Late-Develops-After-Conduction-Disease -- "
            "XLR-Males-Fully-Affected-Females-May-Have-Cardiac-Only -- "
            "Humeroperoneal-Distribution-Weakness-Humeral-Leg-Peroneal -- "
            "CK-Normal-to-5x-Mildly-Elevated -- "
            "Onset-Childhood-Adolescence-Joint-Contractures-First -- "
            "OMIM-Gene-EMD-300384-Disease-EDMD1-310300"
        ),
        "locus": "Xq28",
        "protein_size": "254 aa / 29 kDa",
        "inheritance": (
            "XLR (X-linked recessive); males fully affected; "
            "females obligate carriers — may have cardiac conduction defects (check female relatives); "
            "de novo mutations documented; carrier females: cardiac surveillance mandatory; "
            "Contractures precede weakness — diagnostic clue; onset childhood/adolescence"
        ),
        "key_features": [
            "ELBOW FLEXION CONTRACTURES — earliest sign, precedes weakness PATHOGNOMONIC",
            "Rigid spine — cervical + thoracolumbar early",
            "CARDIAC CONDUCTION DEFECTS — AF, AV block, SCD risk; mandatory surveillance",
            "Emerin IHC ABSENT on muscle biopsy AND skin punch biopsy — diagnostic",
            "PACEMAKER + ICD mandatory — SCD before cardiomyopathy develops",
            "Humeroperoneal weakness distribution (humeral + peroneal muscles)",
            "DCM late — develops after conduction disease established",
            "XLR: males affected; female carriers need cardiac surveillance",
        ],
        "treatment": (
            "Cardiac: PACEMAKER for AV block; ICD for SCD prevention — DO NOT WAIT for DCM; "
            "annual Holter + echo mandatory; "
            "Contractures: physiotherapy, serial casting, surgical release in severe cases; "
            "no disease-modifying skeletal therapy; "
            "genetic counselling family cascade"
        ),
        "contraindications": [
            "DO NOT DEFER PACEMAKER/ICD — SCD occurs before cardiomyopathy; do not wait for DCM to implant",
            "Skin punch biopsy screen: emerin IHC absent confirms diagnosis — do not skip",
        ],
        "critical_pearls": [
            "ELBOW contractures in a boy before weakness = EDMD until EMD+LMNA proved negative",
            "Skin punch biopsy: emerin IHC absent — simple, non-invasive, diagnostic screening tool",
            "Carrier females: 10-20% have cardiac conduction disease — cascade screen all female relatives",
            "SCD precedes cardiomyopathy in EMD EDMD1: implant ICD early, do not wait for DCM",
            "Rigid spine: EMD + LMNA + SELENON (SEPN1-related) — triad of rigid spine myopathies",
        ],
        "mean_age_onset": 12,
        "mean_age_dx": 20,
        "sex_ratio_m_f": "9:1",
    },
    # -- LMNA — Lamin A/C, most allelic gene in human disease ------------------------------
    {
        "gene": "LMNA",
        "alt_name": (
            "LMNA (LMNA-664aa-1q22 / AD — EDMD2-LGMD1B-DCM-CD-FPLD2-Progeria-Laminopathy-Spectrum — "
            "MOST-ALLELIC-GENE-15-DISTINCT-PHENOTYPES — "
            "DCM-CARDIAC-MANDATORY-ICD-EARLY-SCD-Risk — "
            "Lamin-A-C-Nuclear-Lamina-Structural-Protein)"
        ),
        "protein": (
            "LMNA -- 1q22 AD/AR -- LMNA-664aa -- "
            "Lamin-A-C-Type-A-Nuclear-Lamina-Intermediate-Filament-74kDa -- "
            "EDMD2-Autosomal-Dominant-OMIM-181350 -- "
            "LGMD1B-Limb-Girdle-Muscular-Dystrophy-R1-OMIM-159001 -- "
            "DCM-1A-Familial-Dilated-Cardiomyopathy-Conduction-Disease-OMIM-115200 -- "
            "FPLD2-Familial-Partial-Lipodystrophy-Dunnigan-OMIM-151660 -- "
            "Progeria-HGPS-Hutchinson-Gilford-de-Novo-p.G608G-Cryptic-Splice -- "
            "CMT2B1-Charcot-Marie-Tooth-AR-OMIM-605588 -- "
            "MOST-ALLELIC-GENE-HUMAN-15+-DISTINCT-PHENOTYPES-SAME-GENE -- "
            "DCM-CARDIAC-CONDUCTION-DISEASE-MOST-COMMON-LMNA-PHENOTYPE -- "
            "ICD-MANDATORY-EARLY-SCD-Risk->25yr-if-DCM-Penetrance -- "
            "STRIATED-MUSCLE-LAMINOPATHY-Skeletal+Cardiac -- "
            "Hinge-Domain-Missense-Most-Common-Pathogenic-Variants -- "
            "CK-Normal-to-5x-ULN-AD-forms -- "
            "OMIM-Gene-LMNA-150330-Disease-EDMD2-181350"
        ),
        "locus": "1q22",
        "protein_size": "664 aa / 74 kDa (Lamin A) / 60 kDa (Lamin C splice)",
        "inheritance": (
            "AD (most EDMD2/DCM/LGMD1B — missense/splice); "
            "AR (CMT2B1, mandibuloacral dysplasia — biallelic truncating); "
            "de novo (HGPS progeria — p.G608G cryptic splice in exon 11); "
            "Penetrance: striated muscle disease >90%; lipodystrophy females (FPLD2); "
            "15+ distinct phenotypes — allele and modifier-dependent"
        ),
        "key_features": [
            "MOST ALLELIC GENE IN HUMAN DISEASE — 15+ distinct phenotypes",
            "DCM + cardiac conduction disease — most common laminopathy phenotype",
            "ICD mandatory early — SCD risk even with preserved EF; do not wait",
            "LGMD1B: proximal girdle weakness + cardiac; onset adulthood",
            "EDMD2: contractures + rigid spine + humeroperoneal + cardiac",
            "FPLD2: female lipodystrophy — limb fat loss, central adiposity, metabolic syndrome",
            "HGPS: de novo p.G608G — progerin accumulation, accelerated aging, childhood SCD",
            "AR forms: CMT2B1 (motor neuropathy) — compound heterozygous biallelic",
        ],
        "treatment": (
            "Cardiac: ICD for SCD prevention — primary prevention even with preserved EF if significant mutation; "
            "annual echo + Holter + cardiac MRI; "
            "skeletal: physiotherapy, contracture management; "
            "FPLD2: metformin/insulin sensitisers, lipid management; "
            "HGPS: lonafarnib (farnesyltransferase inhibitor) — FDA approved 2020 extends life 2.5yr"
        ),
        "contraindications": [
            "DO NOT DEFER ICD IN LMNA DCM — SCD occurs at any EF; preserved EF does NOT exclude SCD risk",
            "Do not reassure on cardiac risk without annual surveillance — penetrance increases with age",
        ],
        "critical_pearls": [
            "LMNA DCM: ICD for primary SCD prevention — do NOT wait for EF <35% guideline; LMNA-specific rule",
            "FPLD2: female patients only (fat distribution sex-dependent); metabolic syndrome + lipodystrophy = LMNA first",
            "HGPS p.G608G: de novo cryptic splice exon 11 — activates cryptic donor site; progerin protein; childhood diagnosis",
            "EDMD2 vs EMD EDMD1: LMNA AD (one copy) vs EMD XLR (males); emerin IHC normal in LMNA EDMD2",
            "15 phenotypes: when phenotype overlaps muscular dystrophy + cardiac + lipodystrophy → LMNA panel mandatory",
        ],
        "mean_age_onset": 28,
        "mean_age_dx": 35,
        "sex_ratio_m_f": "1:1",
    },
    # -- SYNE1 — Nesprin-1, EDMD4 / ARCA1 ------------------------------------------------
    {
        "gene": "SYNE1",
        "alt_name": (
            "SYNE1 (SYNE1-8797aa-6q25.2 / AR — EDMD4-ARCA1-Autosomal-Recessive-Cerebellar-Ataxia-1 — "
            "LINC-Complex-Outer-Nuclear-Membrane-Giant-Protein — "
            "Quebec-Founder-q.1A-ARCA1-Pure-Cerebellar-Ataxia — "
            "Nesprin-1-KASH-Domain-Nuclear-Cytoskeletal-Coupling)"
        ),
        "protein": (
            "SYNE1 -- 6q25.2 AR/AD -- SYNE1-8797aa -- "
            "Nesprin-1-Nuclear-Envelope-Spectrin-Repeat-Protein-1-LINC-Complex -- "
            "EDMD4-Autosomal-Dominant-Emery-Dreifuss-Muscular-Dystrophy-OMIM-612998 -- "
            "ARCA1-Autosomal-Recessive-Cerebellar-Ataxia-1-Pure-OMIM-610743 -- "
            "Quebec-Founder-c.228C>A-p.Cys76* -- "
            "LINC-Complex-Outer-Nuclear-Membrane-KASH-Domain-SUN-Domain-Binding -- "
            "Largest-Known-Human-Protein-8797aa-Spectrin-Repeats -- "
            "ARCA1-Pure-Non-Progressive-Cerebellar-Ataxia-Quebec-Prevalence -- "
            "EDMD4-AD-Heterozygous-Similar-EMD-EDMD1-Phenotype -- "
            "CK-Normal-or-Mildly-Elevated -- "
            "WES-WGS-SYNE1-Challenging-Large-Gene-Many-VUS -- "
            "OMIM-Gene-SYNE1-608441-Disease-EDMD4-612998-ARCA1-610743"
        ),
        "locus": "6q25.2",
        "protein_size": "8797 aa / ~1 MDa (nesprin-1 giant isoform)",
        "inheritance": (
            "AR (ARCA1 — biallelic truncating; Quebec founder c.228C>A); "
            "AD (EDMD4 — missense in conserved KASH domain); "
            "ARCA1: pure cerebellar ataxia, non-progressive, onset childhood/adult; "
            "EDMD4: rare, AD, rigid spine + contractures + cardiac similar to EMD EDMD1"
        ),
        "key_features": [
            "ARCA1 (AR biallelic): pure non-progressive cerebellar ataxia — Quebec founder",
            "EDMD4 (AD): rigid spine + contractures + humeroperoneal + cardiac",
            "LINC complex outer nuclear membrane — KASH domain binds SUN1/2",
            "Largest known human protein (8797 aa) — WES/WGS bioinformatics challenging",
            "Quebec founder c.228C>A — prevalence ~1:4000 in Beauce region Quebec",
            "ARCA1 vs SYNE1-EDMD: AR vs AD; ataxia vs myopathy; gene panel identifies both",
            "CK normal or mildly elevated",
            "Nuclear-cytoskeletal force transduction — LINC complex biology",
        ],
        "treatment": (
            "ARCA1: supportive cerebellar ataxia management — physiotherapy, balance aids; "
            "EDMD4: cardiac surveillance (pacemaker/ICD as for EDMD1 EMD); "
            "physiotherapy contracture management; "
            "no disease-modifying therapy; "
            "genetic counselling essential (AR ARCA1 vs AD EDMD4)"
        ),
        "contraindications": [
            "Do not classify SYNE1 VUS as pathogenic without functional evidence — gene is large with many benign variants",
            "ARCA1 cardiac: less prominent than EDMD4 — but annual check warranted for all SYNE1 biallelic",
        ],
        "critical_pearls": [
            "Quebec autosomal recessive cerebellar ataxia = ARCA1/SYNE1 first in French-Canadian heritage",
            "LINC complex: SYNE1/SYNE2 (outer nuclear membrane KASH) + SUN1/SUN2 (inner nuclear membrane SUN) — 4-gene panel",
            "SYNE1 8797aa: one of the largest human proteins; challenging bioinformatics; many VUS in WES",
            "ARCA1 non-progressive: important distinction from degenerative ataxias — prognosis counselling critical",
            "EDMD4 AD: phenotypically overlaps EMD EDMD1 — emerin IHC NORMAL (unlike EMD EDMD1)",
        ],
        "mean_age_onset": 18,
        "mean_age_dx": 26,
        "sex_ratio_m_f": "1:1",
    },
    # -- SYNE2 — Nesprin-2, EDMD5 --------------------------------------------------------
    {
        "gene": "SYNE2",
        "alt_name": (
            "SYNE2 (SYNE2-6885aa-14q23.2 / AD — EDMD5-Autosomal-Dominant-EDMD — "
            "LINC-Complex-Outer-Nuclear-Membrane-Nesprin-2 — "
            "KASH-Domain-SUN1-SUN2-Binding-Nuclear-Cytoskeletal-Bridge — "
            "Similar-Phenotype-SYNE1-EDMD4-But-Rarer)"
        ),
        "protein": (
            "SYNE2 -- 14q23.2 AD/AR -- SYNE2-6885aa -- "
            "Nesprin-2-Nuclear-Envelope-Spectrin-Repeat-Protein-2-LINC-Complex -- "
            "EDMD5-Autosomal-Dominant-Emery-Dreifuss-Muscular-Dystrophy-OMIM-612999 -- "
            "LINC-Complex-Outer-Nuclear-Membrane-KASH-Domain -- "
            "Rigid-Spine-Elbow-Contractures-Humeroperoneal-Weakness -- "
            "Cardiac-Conduction-Disease-DCM-Surveillance-Mandatory -- "
            "Second-Largest-Human-Protein-6885aa-Spectrin-Repeats -- "
            "Rarer-Than-SYNE1-EDMD4-Few-Families-Reported -- "
            "CK-Normal-to-3x-ULN -- "
            "OMIM-Gene-SYNE2-608442-Disease-EDMD5-612999"
        ),
        "locus": "14q23.2",
        "protein_size": "6885 aa / ~800 kDa (nesprin-2 giant isoform)",
        "inheritance": (
            "AD (EDMD5 — heterozygous missense/truncating KASH domain); "
            "AR forms: rare dilated cardiomyopathy reports; "
            "Phenotype similar to SYNE1 EDMD4 and EMD EDMD1 — rigid spine + contractures + cardiac; "
            "Very rare globally — few large families reported"
        ),
        "key_features": [
            "EDMD5: rigid spine + elbow contractures + humeroperoneal weakness — AD phenotype",
            "Cardiac conduction disease + DCM — annual surveillance mandatory",
            "LINC complex outer nuclear membrane — KASH domain bridges to SUN1/SUN2",
            "Second-largest human protein (6885 aa) — WES challenging",
            "Very rare — fewer than 20 families reported globally",
            "Phenotypically identical to SYNE1 EDMD4 — gene panel required to distinguish",
            "Emerin IHC NORMAL — unlike EMD EDMD1",
            "Cardiac: pacemaker/ICD as for EDMD1/EDMD2 — SCD risk",
        ],
        "treatment": (
            "Cardiac: Holter + echo annual; pacemaker/ICD for conduction disease/SCD prevention; "
            "physiotherapy contracture management; "
            "rigid spine: cervical collar if severe; "
            "no disease-modifying therapy; "
            "gene panel for all EDMD-phenotype families to distinguish from EMD/LMNA/SYNE1"
        ),
        "contraindications": [
            "Do not stop at EMD/LMNA negative — SYNE2 (EDMD5) must be included in EDMD gene panel",
            "Cardiac surveillance cannot be deferred — conduction disease + SCD risk same as EMD EDMD1",
        ],
        "critical_pearls": [
            "EDMD phenotype with EMD/LMNA negative = extend panel to SYNE1/SYNE2/FHL1/TMEM43",
            "SYNE2 EDMD5: AD inheritance distinguishes from SYNE1 AR (ARCA1) — AD vs AR critical for counselling",
            "Emerin IHC NORMAL in SYNE2 — do not exclude EDMD5 on normal emerin IHC",
            "LINC complex panel: SYNE1 + SYNE2 + SUN1 + SUN2 — nuclear envelope connectome",
            "Very rare: report to international EDMD registry if confirmed SYNE2 EDMD5",
        ],
        "mean_age_onset": 15,
        "mean_age_dx": 24,
        "sex_ratio_m_f": "1:1",
    },
    # -- TMEM43 — LUMA, EDMD7 / ARVC5 ---------------------------------------------------
    {
        "gene": "TMEM43",
        "alt_name": (
            "TMEM43 (TMEM43-400aa-3p25.1 / AD — EDMD7-ARVC5-Arrhythmogenic-RV-Cardiomyopathy — "
            "S358L-Newfoundland-Founder-100pct-Penetrance-Males-LETHAL — "
            "ICD-MANDATORY-ALL-CARRIERS-SCD-PRIMARY-CAUSE-OF-DEATH — "
            "LUMA-Inner-Nuclear-Membrane-WNT-Signalling)"
        ),
        "protein": (
            "TMEM43 -- 3p25.1 AD -- TMEM43-400aa -- "
            "LUMA-LEM-Domain-Inner-Nuclear-Membrane-Protein-Transmembrane-43 -- "
            "EDMD7-Autosomal-Dominant-Emery-Dreifuss-Muscular-Dystrophy-OMIM-614302 -- "
            "ARVC5-Arrhythmogenic-Right-Ventricular-Cardiomyopathy-Type-5-OMIM-604400 -- "
            "S358L-Newfoundland-Atlantic-Canada-Founder-c.1073C>T-p.Ser358Leu -- "
            "100pct-Penetrance-Males-Lethal-Cardiac-Arrhythmia-Mean-Death-41yr-Untreated -- "
            "ICD-MANDATORY-ALL-MUTATION-CARRIERS-Male-AND-Female -- "
            "Biventricular-Cardiomyopathy-Not-RV-Only-In-Most-Carriers -- "
            "Skeletal-Myopathy-Mild-Or-Absent-In-Many -- "
            "WNT-Signalling-Pathway-LUMA-Scaffold-Function -- "
            "SCD-Ventricular-Fibrillation-VF-Primary-Presentation -- "
            "OMIM-Gene-TMEM43-612048-Disease-ARVC5-604400"
        ),
        "locus": "3p25.1",
        "protein_size": "400 aa / 44 kDa",
        "inheritance": (
            "AD (autosomal dominant); S358L Newfoundland founder — Atlantic Canada prevalence ~1:3000; "
            "100% penetrance in males (mean death 41yr untreated); "
            "Females: cardiac disease with reduced penetrance/later onset; "
            "ICD mandatory ALL carriers regardless of sex; "
            "ARVC5 + EDMD7: same gene, cardiac phenotype dominant"
        ),
        "key_features": [
            "S358L NEWFOUNDLAND FOUNDER — 100% penetrance males; mean death 41yr untreated",
            "ICD MANDATORY ALL MUTATION CARRIERS — primary SCD prevention",
            "ARVC5: biventricular cardiomyopathy; VF/SCD primary cause of death",
            "Skeletal myopathy mild or absent — cardiac dominates",
            "EDMD7: when skeletal myopathy present, contractures + humeroperoneal",
            "SCD risk: ventricular fibrillation — ICD implant even in asymptomatic carriers",
            "Annual cardiac MRI + Holter mandatory for all carriers",
            "Atlantic Canada ancestry: any cardiomyopathy → TMEM43 S358L screen",
        ],
        "treatment": (
            "ICD: MANDATORY ALL MUTATION CARRIERS — do not defer; "
            "anti-arrhythmic: amiodarone adjunct for VT burden; "
            "heart failure: standard HF therapy (ACEI/ARB, beta-blocker, MRA); "
            "exercise restriction: avoid competitive sports; "
            "annual cardiac MRI + Holter; "
            "genetic cascade: screen all first-degree relatives urgently"
        ),
        "contraindications": [
            "NEVER DEFER ICD IN S358L CARRIERS — SCD is the presenting event in many; even asymptomatic carriers need ICD",
            "DO NOT REASSURE FEMALE CARRIERS — female penetrance lower but SCD risk real; ICD still mandatory",
        ],
        "critical_pearls": [
            "S358L Newfoundland: any Atlantic Canadian with unexplained cardiomyopathy = TMEM43 S358L screen urgently",
            "100% penetrance males: all male S358L carriers will develop lethal disease without ICD — no exceptions",
            "ARVC5 TMEM43: biventricular pattern atypical for ARVC — do not exclude on RV-only criteria",
            "Athlete SCD: TMEM43 is underdiagnosed cause — screen athletes with cardiomyopathy from Atlantic Canada",
            "S358L founder: founder effect narrow geography; outside Atlantic Canada, full TMEM43 sequencing needed",
        ],
        "mean_age_onset": 30,
        "mean_age_dx": 38,
        "sex_ratio_m_f": "2:1",
    },
    # -- FHL1 — Four-and-a-half LIM domain protein 1, EDMD6 / Reducing Body Myopathy ------
    {
        "gene": "FHL1",
        "alt_name": (
            "FHL1 (FHL1-323aa-Xq26.3 / XLR — EDMD6-Scapuloperoneal-Myopathy-Reducing-Body-Myopathy — "
            "REDUCING-BODY-MYOPATHY-PATHOGNOMONIC-Eosinophilic-Inclusions-Biopsy — "
            "XLR-Males-Affected-Female-Carriers-HCM-Risk — "
            "HCM-Hypertrophic-Cardiomyopathy-Some-Alleles)"
        ),
        "protein": (
            "FHL1 -- Xq26.3 XLR -- FHL1-323aa -- "
            "Four-and-a-half-LIM-Domains-Protein-1-Sarcomere-Z-Disc-Nuclear-Scaffolding -- "
            "EDMD6-X-linked-Emery-Dreifuss-Muscular-Dystrophy-OMIM-300696 -- "
            "Reducing-Body-Myopathy-RBM-OMIM-300718-OMIM-300280 -- "
            "Scapuloperoneal-Myopathy-XLR-OMIM-300695 -- "
            "HCM-Hypertrophic-Cardiomyopathy-Some-FHL1-Mutations -- "
            "REDUCING-BODY-INCLUSIONS-Eosinophilic-Menadione-Nitro-BT-Stain-PATHOGNOMONIC -- "
            "XLR-Males-Fully-Affected-Females-Carriers-Can-Manifest-HCM -- "
            "Scapular-Winging-Early-Sign-Scapuloperoneal-Phenotype -- "
            "CK-Normal-to-5x-ULN -- "
            "OMIM-Gene-FHL1-300163-Disease-EDMD6-300696"
        ),
        "locus": "Xq26.3",
        "protein_size": "323 aa / 32 kDa",
        "inheritance": (
            "XLR (X-linked recessive); males fully affected; "
            "female carriers: some manifest HCM (cardiac involvement); "
            "Genotype-phenotype: EDMD6 vs reducing body myopathy vs scapuloperoneal vs HCM — allele-dependent; "
            "Onset: variable, childhood to adult"
        ),
        "key_features": [
            "REDUCING BODY MYOPATHY — eosinophilic inclusions on biopsy (menadione-nitro BT stain) PATHOGNOMONIC",
            "EDMD6: rigid spine + contractures + humeroperoneal + cardiac",
            "Scapuloperoneal myopathy: scapular winging + peroneal weakness",
            "HCM in some alleles — especially female carriers",
            "XLR: males fully affected; female carriers may manifest HCM",
            "Cardiac: conduction disease + HCM — annual surveillance mandatory",
            "CK normal to mildly elevated",
            "Reducing body inclusions: positive menadione-nitro BT stain — hallmark of FHL1 myopathy",
        ],
        "treatment": (
            "Cardiac: echo annual for HCM screening; pacemaker/ICD for conduction disease; "
            "HCM: beta-blocker, disopyramide; ICD for HCM-SCD risk; "
            "contractures: physiotherapy, serial casting; "
            "scapuloperoneal: scapular stabilisation surgery if severe; "
            "no disease-modifying therapy"
        ),
        "contraindications": [
            "Do not miss female carriers — check echo in all female FHL1 obligate carriers for HCM",
            "Reducing body myopathy biopsy: special stain (menadione-nitro BT) required — standard H&E misses",
        ],
        "critical_pearls": [
            "Reducing body inclusions: menadione-nitro BT stain required — standard H&E may not show; request specifically",
            "FHL1 XLR: distinguish from EMD XLR EDMD1 by emerin IHC (normal in FHL1, absent in EMD)",
            "Female carriers HCM: FHL1 is X-linked but female carriers need echo — HCM penetrance in carriers real",
            "Scapular winging + peroneal weakness in male = FHL1/EDMD6 differential alongside FSHD",
            "Genotype-phenotype: EDMD vs reducing body vs HCM — consult specialist; allele-specific risk",
        ],
        "mean_age_onset": 16,
        "mean_age_dx": 24,
        "sex_ratio_m_f": "8:1",
    },
    # -- SUN1 — SUN domain protein 1, LINC complex inner nuclear membrane ------------------
    {
        "gene": "SUN1",
        "alt_name": (
            "SUN1 (SUN1-916aa-7q32.2 / AR-AD — LINC-Complex-Inner-Nuclear-Membrane-SUN-Domain — "
            "EDMD-Like-Phenotype-Dilated-Cardiomyopathy — "
            "SUN-Domain-Perinuclear-Space-SUN-KASH-Bridge-SYNE1-SYNE2-Binding — "
            "Nuclear-Cytoskeletal-Force-Transduction)"
        ),
        "protein": (
            "SUN1 -- 7q32.2 AR/AD -- SUN1-916aa -- "
            "SUN-Domain-Containing-Protein-1-Inner-Nuclear-Membrane-LINC-Complex -- "
            "SUN-Domain-Spans-Perinuclear-Space-Binds-KASH-Domain-Nesprin-1-2 -- "
            "EDMD-Like-Phenotype-Rigid-Spine-Contractures-Humeroperoneal -- "
            "Dilated-Cardiomyopathy-DCM-Cardiac-Component -- "
            "LINC-Complex-Mechanical-Coupling-Cytoskeleton-Nucleus -- "
            "Loss-LINC-Complex-Integrity-Muscle-Nuclear-Mechanics-Failure -- "
            "Very-Rare-Few-Families-Reported-Globally -- "
            "CK-Normal-or-Mildly-Elevated -- "
            "Compound-Heterozygous-AR-or-Heterozygous-AD-Variant-Classification-Challenging -- "
            "OMIM-Gene-SUN1-607723-Disease-Laminopathy-EDMD-Like"
        ),
        "locus": "7q32.2",
        "protein_size": "916 aa / 103 kDa",
        "inheritance": (
            "AR (biallelic truncating/missense) or AD (heterozygous dominant-negative) reported; "
            "Very rare — fewer than 10 families globally reported with confirmed SUN1 disease; "
            "Phenotype: EDMD-like (contractures + rigid spine + humeroperoneal) + DCM; "
            "LINC complex: if SUN1 disrupted, nesprin-KASH cannot anchor — nuclear mechanics failure"
        ),
        "key_features": [
            "LINC complex inner nuclear membrane — SUN domain bridges to SYNE1/SYNE2 KASH domains",
            "EDMD-like phenotype: rigid spine + contractures + humeroperoneal weakness",
            "Dilated cardiomyopathy — cardiac surveillance mandatory",
            "Very rare — confirmed cases <10 families worldwide",
            "Nuclear mechanics: LINC complex disruption → impaired force transduction nucleus-cytoskeleton",
            "AR or AD — mode depends on allele type (biallelic truncating vs dominant-negative missense)",
            "CK normal or mildly elevated",
            "SUN1 + SUN2: often assessed together as inner nuclear membrane LINC complex pair",
        ],
        "treatment": (
            "Cardiac: echo + Holter annual; pacemaker/ICD for DCM/conduction disease; "
            "contractures: physiotherapy; "
            "LINC complex completeness: assess SYNE1/SYNE2 in parallel — combinatorial defects possible; "
            "no disease-modifying therapy; "
            "research setting: LINC complex reconstitution strategies experimental"
        ),
        "contraindications": [
            "Do not classify SUN1 variants as pathogenic without co-segregation — VUS interpretation needs specialist",
            "Cardiac cannot be deferred — DCM risk even when myopathy mild",
        ],
        "critical_pearls": [
            "LINC complex: SUN1/2 (inner NM) + SYNE1/2 (outer NM) — 4 components; panel all 4 in suspected nuclear myopathy",
            "SUN domain function: SUN1 trimers span perinuclear space, KASH domain of nesprin plugs in — mechanical bridge",
            "Very rare: register confirmed SUN1 disease cases in international EDMD registry",
            "AD dominant-negative: single copy loss-of-function disrupts SUN1 trimer — dominant-negative mechanism",
            "Distinguish SUN1 from EMD/LMNA: emerin/lamin IHC usually preserved in SUN1 disease",
        ],
        "mean_age_onset": 18,
        "mean_age_dx": 28,
        "sex_ratio_m_f": "1:1",
    },
    # -- LEMD3 — MAN1, Buschke-Ollendorff syndrome / Melorheostosis -----------------------
    {
        "gene": "LEMD3",
        "alt_name": (
            "LEMD3 (LEMD3-MAN1-922aa-12q14.3 / AD — Buschke-Ollendorff-Syndrome-Melorheostosis — "
            "OSTEOPOIKILOSIS-DERMATOFIBROSIS-LENTICULARIS-PATHOGNOMONIC-Dual — "
            "TGF-Beta-SMAD-Antagonism-Loss-Bone-Overgrowth — "
            "DERMATOLOGY-ORTHOPAEDICS-DUAL-Referral-Mandatory)"
        ),
        "protein": (
            "LEMD3 -- 12q14.3 AD -- LEMD3-MAN1-922aa -- "
            "LEM-Domain-Containing-Protein-3-MAN1-Inner-Nuclear-Membrane-TGF-Beta-Regulator -- "
            "Buschke-Ollendorff-Syndrome-BOS-OMIM-166700-Osteopoikilosis-Dermatofibrosis -- "
            "Melorheostosis-OMIM-155950-Sclerotome-Somatic-Mosaic-OR-Germline -- "
            "OSTEOPOIKILOSIS-Sclerotic-Bone-Dots-Epiphyses-Pelvis-PATHOGNOMONIC-X-Ray -- "
            "DERMATOFIBROSIS-LENTICULARIS-White-Connective-Tissue-Naevus-Skin-Lesions -- "
            "TGF-Beta-SMAD-Antagonism-Loss-Bone-Overgrowth-Fibrosis-Skin-Bone -- "
            "Melorheostosis-Flowing-Candle-Wax-Cortex-X-Ray-Cortical-Hyperostosis -- "
            "Somatic-Mosaic-Melorheostosis-MAP2K1-SMAD3-KRAS-Somatic-Drivers-Also -- "
            "Mostly-Benign-BOS-But-Joint-Pain-Restriction-Management-Needed -- "
            "OMIM-Gene-LEMD3-607844-Disease-BOS-166700-Melorheostosis-155950"
        ),
        "locus": "12q14.3",
        "protein_size": "922 aa / 103 kDa",
        "inheritance": (
            "AD (autosomal dominant — Buschke-Ollendorff syndrome); "
            "Somatic mosaic or germline (melorheostosis — also MAP2K1/SMAD3/KRAS somatic variants); "
            "BOS: osteopoikilosis + skin lesions; variable expression within families; "
            "Penetrance: osteopoikilosis near 100%; skin lesions variable; "
            "Melorheostosis: may be somatic mosaic LEMD3 OR non-LEMD3 somatic (MAP2K1)"
        ),
        "key_features": [
            "OSTEOPOIKILOSIS — sclerotic dots at epiphyses/metaphyses on X-ray PATHOGNOMONIC",
            "DERMATOFIBROSIS LENTICULARIS — white connective tissue naevus skin lesions",
            "MELORHEOSTOSIS — 'flowing candle wax' cortical hyperostosis on X-ray",
            "TGF-β/SMAD antagonism: LEMD3 inhibits R-SMAD; loss → TGF-β overactivation → bone + skin fibrosis",
            "Dual DERMATOLOGY + ORTHOPAEDICS referral mandatory",
            "Mostly benign but joint pain/restriction from bone lesions",
            "Somatic mosaic melorheostosis: somatic MAP2K1/SMAD3/KRAS — NOT all LEMD3 germline",
            "NOT primarily a myopathy — bone and skin disease distinguishes from other nuclear envelope genes",
        ],
        "treatment": (
            "Mostly conservative: analgesia, physiotherapy for joint restriction; "
            "orthopaedic surgery for melorheostosis-related joint compression (rare); "
            "anti-remodelling trials (bisphosphonates) limited evidence; "
            "dermatology: cosmetic review for skin lesions; "
            "no systemic treatment needed for BOS; "
            "genetic counselling — mostly benign prognosis"
        ),
        "contraindications": [
            "Do not over-treat BOS — mostly benign; avoid unnecessary bone intervention",
            "Do not miss somatic mosaic melorheostosis — somatic testing of affected bone tissue required if germline negative",
        ],
        "critical_pearls": [
            "Osteopoikilosis incidental X-ray finding (pelvis/hands/feet): check skin for connective tissue naevus = BOS/LEMD3",
            "LEMD3 is a nuclear envelope gene in the EDMD gene family — very different phenotype (bone + skin, not myopathy)",
            "Melorheostosis: if LEMD3 germline negative, order somatic panel on bone biopsy (MAP2K1/SMAD3/KRAS hotspots)",
            "TGF-β overactivation: overlap with Loeys-Dietz/MFS (FBN1) and BOS (LEMD3) — both TGF-β pathway dysregulation",
            "BOS prognosis excellent — most patients require only reassurance + orthopaedic surveillance",
        ],
        "mean_age_onset": 20,
        "mean_age_dx": 28,
        "sex_ratio_m_f": "1:1",
    },
]


def _make_patients(gene_data, seed):
    """Generate 40 realistic synthetic patients for a given EDMD / nuclear envelope gene."""
    rng = random.Random(seed)
    gene = gene_data["gene"]
    mean_onset = gene_data["mean_age_onset"]
    mean_dx_delay = gene_data["mean_age_dx"] - mean_onset

    outcomes = {
        "EMD":    {"ambulant_10yr": 0.80, "alive_20yr": 0.75, "treated_pct": 0.75},
        "LMNA":   {"ambulant_10yr": 0.72, "alive_20yr": 0.70, "treated_pct": 0.80},
        "SYNE1":  {"ambulant_10yr": 0.82, "alive_20yr": 0.90, "treated_pct": 0.35},
        "SYNE2":  {"ambulant_10yr": 0.80, "alive_20yr": 0.82, "treated_pct": 0.40},
        "TMEM43": {"ambulant_10yr": 0.85, "alive_20yr": 0.55, "treated_pct": 0.90},
        "FHL1":   {"ambulant_10yr": 0.75, "alive_20yr": 0.80, "treated_pct": 0.55},
        "SUN1":   {"ambulant_10yr": 0.78, "alive_20yr": 0.82, "treated_pct": 0.45},
        "LEMD3":  {"ambulant_10yr": 0.98, "alive_20yr": 0.96, "treated_pct": 0.20},
    }
    out = outcomes.get(gene, {"ambulant_10yr": 0.78, "alive_20yr": 0.80, "treated_pct": 0.50})

    # CK ranges: nuclear envelope myopathies generally mild CK
    ck_low = {"EMD": 1, "LMNA": 1, "SYNE1": 1, "SYNE2": 1, "TMEM43": 1, "FHL1": 1, "SUN1": 1, "LEMD3": 0.5}.get(gene, 1)
    ck_high = {"EMD": 5, "LMNA": 5, "SYNE1": 3, "SYNE2": 3, "TMEM43": 3, "FHL1": 5, "SUN1": 3, "LEMD3": 1.5}.get(gene, 3)

    patients = []
    for i in range(40):
        age_onset = max(0, int(rng.gauss(mean_onset, max(mean_onset * 0.25, 3))))
        dx_delay = max(0, int(rng.gauss(mean_dx_delay, 4)))
        age_dx = age_onset + dx_delay
        current_age = age_dx + rng.randint(1, 20)
        ck_mult = rng.uniform(ck_low, ck_high)
        ck_val = round(ck_mult * rng.uniform(0.8, 1.2) * 200, 0)
        ambulant = rng.random() < out["ambulant_10yr"]
        alive = rng.random() < out["alive_20yr"]
        treated = rng.random() < out["treated_pct"]
        attacks_py = round(rng.uniform(0, 0.5), 1)  # mostly cardiac events, not myopathic attacks
        patients.append({
            "id": f"{gene}_{seed}_{i:02d}",
            "gene": gene,
            "age_onset": age_onset,
            "age_dx": age_dx,
            "dx_delay_yr": dx_delay,
            "current_age": current_age,
            "ck_iu_l": int(ck_val),
            "ambulant_at_10yr": ambulant,
            "alive": alive,
            "treated": treated,
            "attacks_per_year": attacks_py,
            "sex": rng.choice(["M", "F"]) if gene not in ("EMD", "FHL1") else rng.choices(["M", "F"], weights=[9, 1])[0],
        })
    return patients


def overview():
    """Aggregate overview across all 8 genes (320 patients)."""
    all_patients = []
    for i, gd in enumerate(EDMD_GENES):
        all_patients += _make_patients(gd, SEED_BASE + i)

    n = len(all_patients)
    alive = sum(1 for p in all_patients if p["alive"])
    treated = sum(1 for p in all_patients if p["treated"])
    ambulant = sum(1 for p in all_patients if p["ambulant_at_10yr"])
    mean_onset = round(sum(p["age_onset"] for p in all_patients) / n, 1)
    mean_dx_delay = round(sum(p["dx_delay_yr"] for p in all_patients) / n, 1)
    mean_ck = round(sum(p["ck_iu_l"] for p in all_patients) / n, 0)

    gene_summaries = []
    for i, gd in enumerate(EDMD_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        gene_summaries.append({
            "gene": gd["gene"],
            "n_patients": len(pts),
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance": gd["inheritance"].split(";")[0].strip(),
            "mean_onset_yr": round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "mean_ck": round(sum(p["ck_iu_l"] for p in pts) / len(pts), 0),
            "alive_pct": round(sum(1 for p in pts if p["alive"]) / len(pts) * 100, 1),
            "ambulant_10yr_pct": round(sum(1 for p in pts if p["ambulant_at_10yr"]) / len(pts) * 100, 1),
        })

    return {
        "atlas": "Hereditary-EDMD-Nuclear-Envelope-Atlas",
        "subtitle": (
            "Complete 8-Gene EDMD / Nuclear Envelope Myopathy Spectrum "
            "(EMD · LMNA · SYNE1 · SYNE2 · TMEM43 · FHL1 · SUN1 · LEMD3)"
        ),
        "genes_covered": [gd["gene"] for gd in EDMD_GENES],
        "n_genes": 8,
        "n_patients": n,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "alive_pct": round(alive / n * 100, 1),
        "treated_pct": round(treated / n * 100, 1),
        "ambulant_10yr_pct": round(ambulant / n * 100, 1),
        "mean_age_onset_yr": mean_onset,
        "mean_dx_delay_yr": mean_dx_delay,
        "mean_ck_iu_l": int(mean_ck),
        "key_spectrum_facts": [
            "EMD: ELBOW CONTRACTURES EARLIEST SIGN PATHOGNOMONIC; emerin IHC ABSENT on muscle+skin biopsy; PACEMAKER/ICD mandatory; SCD before DCM",
            "LMNA: MOST ALLELIC GENE 15+ phenotypes; ICD mandatory EARLY in DCM-CD; SCD risk even preserved EF; lonafarnib for HGPS",
            "SYNE1: ARCA1 Quebec founder (AR cerebellar ataxia) + EDMD4 (AD myopathy); LINC complex outer NM; 8797aa largest human protein",
            "SYNE2: EDMD5 AD; LINC complex outer NM; very rare; phenotype identical to SYNE1 EDMD4 — gene panel mandatory",
            "TMEM43: S358L Newfoundland founder 100% penetrance males; ICD MANDATORY ALL CARRIERS; ARVC5; mean death 41yr untreated",
            "FHL1: XLR; REDUCING BODY MYOPATHY biopsy PATHOGNOMONIC (menadione-nitro BT stain); female carriers HCM risk; EDMD6",
            "SUN1: LINC complex inner NM SUN domain; EDMD-like+DCM; very rare; SUN1+SUN2+SYNE1+SYNE2 full LINC panel",
            "LEMD3: Buschke-Ollendorff OSTEOPOIKILOSIS+DERMATOFIBROSIS DUAL PATHOGNOMONIC; TGF-β dysregulation; mostly benign",
        ],
        "critical_ddx": {
            "EMD_vs_LMNA": "EMD: XLR, emerin IHC ABSENT; LMNA: AD, emerin IHC NORMAL — emerin IHC is the first-line discriminator",
            "EDMD_vs_FSHD": "EDMD: contractures + rigid spine + cardiac; FSHD: asymmetric facial + humeral + D4Z4 contraction; no cardiac",
            "TMEM43_vs_PKP2_ARVC": "TMEM43 ARVC5: biventricular pattern, founder S358L; PKP2 ARVC: RV-predominant, desmosome gene panel",
            "SYNE1_ARCA1_vs_FRDA": "SYNE1 ARCA1: pure non-progressive AR ataxia; FRDA: progressive + cardiomyopathy + GAA repeat FXN",
            "FHL1_vs_EMD": "FHL1 XLR: emerin IHC NORMAL, reducing bodies biopsy; EMD XLR: emerin IHC ABSENT, no reducing bodies",
            "LEMD3_vs_Tuberous_Sclerosis": "LEMD3 BOS: osteopoikilosis + skin naevus; TSC: angiofibromas + ash-leaf + renal AML; gene panel distinguishes",
            "LMNA_vs_VCP_DCM": "LMNA DCM-CD: conduction disease early, ICD mandatory; VCP: IBMPFD triad (myopathy+Paget+FTD) — very different",
        },
        "gene_summaries": gene_summaries,
    }


def breakdown():
    """Per-gene clinical breakdown with key metrics."""
    result = {}
    for i, gd in enumerate(EDMD_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        result[gd["gene"]] = {
            "gene": gd["gene"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "n_patients": len(pts),
            "alive_pct": round(sum(1 for p in pts if p["alive"]) / len(pts) * 100, 1),
            "treated_pct": round(sum(1 for p in pts if p["treated"]) / len(pts) * 100, 1),
            "ambulant_10yr_pct": round(sum(1 for p in pts if p["ambulant_at_10yr"]) / len(pts) * 100, 1),
            "mean_age_onset": round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "mean_dx_delay_yr": round(sum(p["dx_delay_yr"] for p in pts) / len(pts), 1),
            "mean_ck_iu_l": int(sum(p["ck_iu_l"] for p in pts) / len(pts)),
            "mean_attacks_per_year": round(sum(p["attacks_per_year"] for p in pts) / len(pts), 1),
            "key_features": gd["key_features"],
            "treatment": gd["treatment"],
            "contraindications": gd["contraindications"],
            "critical_pearls": gd["critical_pearls"],
            "inheritance": gd["inheritance"][:200],
        }
    return result


def definitions():
    """Gene definitions, biopsy patterns, cardiac surveillance, DDx tables, LINC complex panel."""
    gene_defs = {}
    for gd in EDMD_GENES:
        gene_defs[gd["gene"]] = {
            "full_name": gd["protein"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance_detail": gd["inheritance"],
            "alt_name": gd["alt_name"],
        }

    return {
        "gene_definitions": gene_defs,
        "onset_age_spectrum": {
            "EMD_EDMD1":       "Childhood/adolescence — elbow contractures first, before weakness",
            "FHL1_EDMD6":      "Variable — childhood to adult onset (reducing body myopathy severe early)",
            "SYNE1_EDMD4":     "Adolescence to early adult (EDMD4) / Childhood-adult (ARCA1 cerebellar ataxia)",
            "SYNE2_EDMD5":     "Adolescence to early adult — similar to SYNE1 EDMD4",
            "SUN1":            "Late adolescence to early adult",
            "LMNA_EDMD2":      "Young adult 20-40yr (striated muscle laminopathy); cardiac may precede",
            "TMEM43_ARVC5":    "Adult 25-45yr — cardiac events dominant; SCD risk peak 30-50yr",
            "LEMD3_BOS":       "Incidental discovery any age — osteopoikilosis often asymptomatic",
        },
        "biopsy_pattern_table": {
            "EMD":    "Myopathic non-specific; emerin IHC ABSENT (inner nuclear membrane staining lost); skin punch biopsy also shows absent emerin",
            "LMNA":   "Myopathic non-specific; rimmed vacuoles in some; lamin A/C IHC varies by allele; emerin NORMAL",
            "SYNE1":  "Myopathic non-specific (EDMD4); cerebellar cortex atrophy on MRI (ARCA1); muscle biopsy non-diagnostic",
            "SYNE2":  "Myopathic non-specific; emerin/lamin IHC normal; LINC complex IHC research tool",
            "TMEM43": "Fibrofatty replacement RV on cardiac biopsy/MRI; skeletal myopathy mild-absent; fibro-adipose",
            "FHL1":   "REDUCING BODIES — eosinophilic granular inclusions; positive menadione-nitro BT stain PATHOGNOMONIC; HE may miss",
            "SUN1":   "Myopathic non-specific; SUN1 IHC research tool; emerin normal",
            "LEMD3":  "Skin biopsy: increased collagen dermis (dermatofibrosis); bone biopsy: sclerotic foci (not routinely needed)",
        },
        "cardiac_surveillance_table": {
            "EMD":    "Holter 6-monthly + echo annual — PACEMAKER for AV block; ICD for SCD prevention even before DCM; start at diagnosis",
            "LMNA":   "Echo + Holter annual; cardiac MRI if LGE assessment needed; ICD for primary SCD prevention — do NOT defer; start at diagnosis",
            "SYNE1":  "Echo annual (EDMD4 AD); ARCA1 AR — echo annual even in ataxia phenotype; less cardiac than EMD/LMNA",
            "SYNE2":  "Echo + Holter annual — cardiac conduction disease + DCM risk; pacemaker/ICD as for EMD/LMNA",
            "TMEM43": "Echo + Holter 6-monthly + cardiac MRI annual — ICD MANDATORY ALL CARRIERS; do not defer; electrophysiology if VT",
            "FHL1":   "Echo annual for HCM — female carriers especially; pacemaker/ICD for conduction disease; HCM management per guidelines",
            "SUN1":   "Echo + Holter annual — DCM risk; pacemaker/ICD for conduction disease; less data than EMD/LMNA",
            "LEMD3":  "No cardiac involvement — no cardiac surveillance required for BOS; annual review for joint symptoms",
        },
        "ddx_table": {
            "EMD_vs_LMNA":              "Emerin IHC ABSENT in EMD; NORMAL in LMNA — first discriminator; XLR (EMD) vs AD (LMNA)",
            "TMEM43_vs_PKP2_ARVC":      "TMEM43 ARVC5: biventricular, S358L Newfoundland; PKP2 most common ARVC: RV dominant, desmosome",
            "SYNE1_ARCA1_vs_FRDA":      "SYNE1 ARCA1: AR non-progressive cerebellar; FRDA: progressive ataxia + HCM + GAA repeat FXN",
            "FHL1_vs_FSHD":             "FHL1: reducing bodies biopsy; XLR; no D4Z4; FSHD: D4Z4 contraction; facial; AD",
            "LMNA_vs_TTN_DCM":          "LMNA DCM: conduction disease early; ICD mandatory; TTN-titinopathy DCM: less conduction disease; different gene panel",
            "LEMD3_vs_Melorheostosis":   "BOS (germline LEMD3): osteopoikilosis + skin; melorheostosis (somatic mosaic LEMD3/MAP2K1/KRAS): unilateral candle-wax",
            "EDMD_vs_Congenital_Myopathy": "EDMD: contractures + cardiac + humeroperoneal; congenital myopathy: cores/nemaline biopsy; RYR1/NEB/ACTA1",
        },
        "founder_mutations": {
            "TMEM43_S358L":   "TMEM43 p.Ser358Leu — Newfoundland/Atlantic Canada founder; 100% penetrance males; ICD mandatory all carriers",
            "SYNE1_c228CA":   "SYNE1 c.228C>A p.Cys76* — Quebec French-Canadian founder; ARCA1 non-progressive AR cerebellar ataxia",
            "LMNA_R377H":     "LMNA p.Arg377His — West African founder; EDMD/DCM-CD in African-American and West African families",
            "LMNA_del_K32":   "LMNA c.93-99del p.Lys32del — mandibuloacral dysplasia; AR form; lipodystrophy + mandibular hypoplasia",
            "LMNA_G608G":     "LMNA p.G608G (c.1824C>T) — de novo cryptic splice exon 11; progerin; Hutchinson-Gilford Progeria HGPS",
        },
        "linc_complex_panel": {
            "outer_nuclear_membrane": {
                "SYNE1": "Nesprin-1 — 8797aa; KASH domain binds SUN1/2; EDMD4 (AD) + ARCA1 (AR)",
                "SYNE2": "Nesprin-2 — 6885aa; KASH domain; EDMD5 (AD/AR)",
            },
            "inner_nuclear_membrane": {
                "SUN1": "SUN1 — 916aa; SUN domain spans perinuclear space; EDMD-like + DCM",
                "SUN2": "SUN2 — 717aa; SUN domain; partner of SUN1; EDMD-like reports",
                "EMD":  "Emerin — 254aa; LEM domain; IHC absent in EDMD1; interacts with lamin A",
                "LEMD3": "MAN1/LEMD3 — 922aa; LEM domain; TGF-β SMAD antagonist; BOS phenotype",
            },
            "nuclear_lamina": {
                "LMNA": "Lamin A/C — 664aa; nuclear lamina scaffold; most allelic human gene; EDMD2/LGMD1B/DCM/FPLD2/progeria",
            },
            "clinical_note": (
                "LINC complex defects share: rigid spine, elbow contractures, humeroperoneal weakness, cardiac conduction disease. "
                "Order a multi-gene EDMD panel (EMD + LMNA + SYNE1 + SYNE2 + FHL1 + TMEM43 + SUN1 + LEMD3) for any EDMD phenotype."
            ),
        },
        "emerin_ihc_decision_tree": {
            "emerin_IHC_absent_XLR_male": "EMD EDMD1 — confirm with EMD gene sequencing",
            "emerin_IHC_absent_AD_female": "LMNA (rare) or EMD carrier female — sequence both",
            "emerin_IHC_normal_EDMD_phenotype": "LMNA / SYNE1 / SYNE2 / FHL1 / SUN1 — extend to full nuclear envelope panel",
            "reducing_bodies_biopsy": "FHL1 — XLR; check menadione-nitro BT stain specifically; sequence FHL1",
            "osteopoikilosis_skin_naevus": "LEMD3 BOS — predominantly benign; bone + skin referral",
            "S358L_Atlantic_Canada": "TMEM43 — ICD urgently; cascade all first-degree relatives",
        },
        "glossary": {
            "LINC_complex":          "Linker of Nucleoskeleton and Cytoskeleton — SUN1/2 (inner NM) + nesprin-1/2 (outer NM) + lamins; force transduction nucleus↔cytoskeleton",
            "Emerin":                "LEM-domain inner nuclear membrane protein; binds lamin A; absent IHC in EMD EDMD1 (XLR); also absent skin punch biopsy",
            "Laminopathy":           "Disease caused by LMNA/B mutations; spectrum includes EDMD2, LGMD1B, DCM-CD, FPLD2, CMT2B1, HGPS — all from same gene",
            "EDMD":                  "Emery-Dreifuss Muscular Dystrophy — clinical triad: joint contractures (elbow/spine) + humeroperoneal weakness + cardiac disease",
            "Reducing_body_myopathy": "FHL1-related myopathy; eosinophilic inclusions positive on menadione-nitro BT stain; XLR; severe early-onset variant",
            "Osteopoikilosis":       "Sclerotic dots at epiphyses/metaphyses on X-ray — LEMD3 BOS pathognomonic; usually asymptomatic; benign",
            "Dermatofibrosis_lenticularis": "White connective tissue naevus skin lesions in BOS (LEMD3); occurs over elbows/nape/buttocks",
            "HGPS_progerin":         "Hutchinson-Gilford Progeria; de novo LMNA p.G608G; cryptic splice → truncated lamin A = progerin; accelerated ageing; SCD childhood",
            "KASH_domain":           "Klarsicht-ANC-SYNE homology domain — C-terminal domain of nesprins (SYNE1/2); anchored in outer nuclear membrane; binds SUN domain",
            "SUN_domain":            "Sad1/UNC-84 domain — C-terminal domain of SUN1/2; spans perinuclear space; binds KASH of nesprins; LINC complex bridge",
            "FPLD2":                 "Familial Partial Lipodystrophy Dunnigan — LMNA missense in Ig-fold (R482Q/W hotspot); fat loss limbs + central adiposity; females > males; metabolic syndrome",
            "Melorheostosis":        "Flowing cortical hyperostosis 'candle wax on bone' X-ray; somatic mosaic LEMD3 or MAP2K1/SMAD3/KRAS; asymmetric; limb pain/deformity",
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = overview()
    print(json.dumps({k: v for k, v in ov.items() if k != "gene_summaries"}, indent=2))
    print("\n=== BREAKDOWN keys ===")
    br = breakdown()
    for gene, data in br.items():
        print(f"  {gene}: {data['n_patients']} patients, alive={data['alive_pct']}%, ambulant_10yr={data['ambulant_10yr_pct']}%")
    print("\n=== DEFINITIONS keys ===")
    defs = definitions()
    print(list(defs.keys()))
