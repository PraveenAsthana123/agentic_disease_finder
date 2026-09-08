#!/usr/bin/env python3
"""Hereditary-Nemaline-Myopathy-Atlas — Complete 8-Gene Nemaline Myopathy (NM) Spectrum Atlas
(NEB · ACTA1 · TPM2 · TPM3 · TNNT1 · CFL2 · KBTBD13 · LMOD3).

NEB     (Nebulin; 6669 aa; 2q23.3; AR;
         Typical/Severe nemaline myopathy — most common AR NM (50% of all NM);
         NEMALINE RODS on Gomori trichrome biopsy PATHOGNOMONIC;
         congenital hypotonia; facial + bulbar weakness; respiratory failure critical;
         CK normal–mildly elevated; NEB exon 55 deletion founder Ashkenazi Jewish;
         seed SEED_BASE+0).
ACTA1   (Actin α1 skeletal muscle; 375 aa; 1q42.13; AD de novo / AR;
         Typical to severe/congenital nemaline myopathy — 2nd most common NM gene;
         INTRANUCLEAR RODS (actin aggregates inside nucleus) PATHOGNOMONIC for ACTA1;
         de novo dominant = most severe; AR = milder; CK normal–2×;
         seed SEED_BASE+1).
TPM2    (Tropomyosin β; 284 aa; 9p13.3; AD / AR;
         Nemaline myopathy type 4 + Cap Disease + Distal Arthrogryposis DA2A/DA2B;
         ALLELIC (same gene different phenotypes); TPM2 GOF → distal arthrogryposis;
         TPM2 LOF → nemaline; CAP DISEASE (thin filament cap at rods) on biopsy;
         seed SEED_BASE+2).
TPM3    (Tropomyosin α-slow; 285 aa; 1q21.2; AD / AR;
         Typical nemaline myopathy type 1; Slow-twitch fiber-selective tropomyosin;
         AD dominant cap disease; AR typical NM; fiber-type disproportion overlap;
         slower progression than NEB/ACTA1; CK normal; exercise intolerance;
         seed SEED_BASE+3).
TNNT1   (Troponin T type 1 slow skeletal; 328 aa; 19q13.42; AR;
         Nemaline Myopathy Amish — LETHAL INFANTILE in Amish founder population;
         E180X Amish founder mutation (homozygous); congenital hypotonia; rapidly progressive;
         FATAL BY 2-3 YEARS typical; non-Amish: milder E180K alleles;
         seed SEED_BASE+4).
CFL2    (Cofilin-2; 166 aa; 14q13.1; AR;
         Nemaline myopathy type 4 — actin-depolymerizing factor;
         Myopathy with CORES and RODS on biopsy; proximal > distal weakness;
         respiratory failure moderate; childhood to adult onset; CK mildly elevated;
         seed SEED_BASE+5).
KBTBD13 (Kelch repeat and BTB domain-containing protein 13; 571 aa; 15q22.31; AD;
         Nemaline myopathy type 6 — SLOW FIBER-SELECTIVE disease;
         EXERCISE INTOLERANCE + SLOW RELAXATION PATHOGNOMONIC;
         mild course; CK normal; slow-twitch fiber predominance on biopsy;
         Dutch founder haplotype; slow relaxation time (grip myotonia-like but electrically silent);
         seed SEED_BASE+6).
LMOD3   (Leiomodin-3; 547 aa; 3p14.1; AR;
         Nemaline myopathy type 10 — thin filament elongation factor;
         severe congenital NM; cardiac involvement (DCM possible); respiratory failure early;
         CK mildly elevated; rods and minimal filaments on biopsy;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2222-2229).
"""

import random

SEED_BASE = 2222

NM_GENES = [
    # -- NEB — Nebulin, most common AR NM -----------------------------------------------
    {
        "gene": "NEB",
        "alt_name": (
            "NEB (NEB-6669aa-2q23.3 / AR — Typical-Severe-NM-Most-Common-AR-NM-50pct — "
            "NEMALINE-RODS-Gomori-Trichrome-PATHOGNOMONIC — "
            "Congenital-Hypotonia-Facial-Bulbar-Weakness-Respiratory-Failure — "
            "NEB-Exon55-Deletion-Ashkenazi-Jewish-Founder)"
        ),
        "protein": (
            "NEB -- 2q23.3 AR -- NEB-6669aa -- "
            "Nebulin-800kDa-Sarcomeric-Thin-Filament-Template-Actin-Polymerization -- "
            "NM-OMIM-256030 -- "
            "NEMALINE-RODS-Subsarcolemmal-Z-Disc-Derived-Gomori-Trichrome-Modified-PATHOGNOMONIC -- "
            "MOST-COMMON-AR-NEMALINE-MYOPATHY-50pct-All-NM -- "
            "Congenital-Hypotonia-Facial-Weakness-Bulbar-Weakness-Respiratory-Failure -- "
            "CK-Normal-To-Mildly-Elevated-1-3x -- "
            "NEB-Exon55-Deletion-Ashkenazi-Jewish-Founder-5pct-Carrier-Frequency -- "
            "Gene-Size-Largest-Human-Gene-Exons-183-Sequencing-Challenge -- "
            "Cardiac-Usually-Spared-Monitor -- "
            "OMIM-Gene-NEB-161650-Disease-NM2-256030"
        ),
        "locus": "2q23.3",
        "protein_size": "6669 aa / ~800 kDa",
        "inheritance": (
            "AR (biallelic loss-of-function); "
            "Most common AR nemaline myopathy (50% of all genetically confirmed NM); "
            "NEB exon 55 deletion: Ashkenazi Jewish founder (~5% carrier frequency); "
            "CK: normal to mildly elevated (1-3×); "
            "Onset: congenital (classic), but spectrum to childhood; "
            "Respiratory failure: critical — NIV often 1st decade; FVC monitoring mandatory; "
            "Facial + bulbar weakness: dysarthria, dysphagia, nasal speech; "
            "Cardiac: usually spared; "
            "Motor milestone delay: walking achieved in typical cases (ambulant form)"
        ),
        "key_features": [
            "NEMALINE RODS on Gomori modified trichrome — subsarcolemmal + intrasarcoplasmic; Z-disc-derived; PATHOGNOMONIC",
            "Most common AR NM gene — 50% of all genetically confirmed nemaline myopathy",
            "NEB exon 55 deletion — Ashkenazi Jewish founder; ~5% carrier frequency; homozygous = typical NM",
            "Largest human gene (183 exons) — targeted panel/WES may miss deep intronic variants; RNA studies needed",
            "Congenital hypotonia — floppy infant, poor feeding, nasal voice, ptosis",
            "Facial + bulbar weakness — dysarthria, dysphagia, nasal speech; aspiration risk",
            "Respiratory failure — NIV often 1st decade; CK normal or mildly elevated (not markedly raised)",
            "Cardiac: usually spared (unlike ACTA1/MFM); echocardiogram baseline advised",
        ],
        "treatment": (
            "Respiratory: NIV monitoring (FVC every 6-12 months); nocturnal hypoventilation screen. "
            "Nutrition: gastrostomy if swallowing unsafe; caloric support. "
            "Physiotherapy: avoid contractures; maintain ambulation. "
            "Cardiac: baseline echo; annual follow-up. "
            "Genetic: cascade testing siblings; NEB exon 55 deletion PCR screen if Ashkenazi. "
            "No disease-modifying therapy approved 2026; "
            "Salbutamol (albuterol): open-label trials show strength improvement in some NM subtypes (ACTA1/NEB); Level C evidence. "
            "L-tyrosine: anecdotal reports; Level C."
        ),
        "monitoring": [
            "Respiratory: FVC sitting+supine every 6 months; nocturnal oximetry; sleep study if symptomatic",
            "Swallowing: videofluoroscopy; modified barium swallow; SLP assessment",
            "Cardiac: baseline echocardiogram + ECG; repeat annually",
            "Nutrition: weight gain velocity; micronutrient screen",
            "Motor: physiotherapy + occupational therapy; AFO/orthoses",
            "Genetic: family cascade; NEB exon 55 deletion PCR if Ashkenazi Jewish",
            "Eyes: ptosis assessment — surgical if vision compromised",
            "NBS: not current standard; research protocols emerging",
        ],
    },
    # -- ACTA1 — Actin α1 skeletal, 2nd most common NM ----------------------------------
    {
        "gene": "ACTA1",
        "alt_name": (
            "ACTA1 (ACTA1-375aa-1q42.13 / AD-de-novo / AR — Typical-to-Severe-Congenital-NM — "
            "INTRANUCLEAR-RODS-ACTA1-Specific-PATHOGNOMONIC — "
            "De-Novo-Dominant-Most-Severe-Congenital-Form — "
            "2nd-Most-Common-NM-Gene)"
        ),
        "protein": (
            "ACTA1 -- 1q42.13 AD-de-novo / AR -- ACTA1-375aa -- "
            "Actin-Alpha-1-Skeletal-Muscle-Isoform-42kDa-Thin-Filament-Main-Component -- "
            "NM3-OMIM-161800 -- "
            "INTRANUCLEAR-RODS-Actin-Aggregate-Inside-Nucleus-PATHOGNOMONIC-ACTA1-Specific -- "
            "DE-NOVO-DOMINANT-MOST-SEVERE-Congenital-Respiratory-Failure-Lethal-Without-Ventilation -- "
            "AR-ACTA1-MILDER-Typical-NM-Ambulant-Form -- "
            "CK-Normal-to-2x -- "
            "Salbutamol-Level-C-Evidence-Open-Label-Trials -- "
            "OMIM-Gene-ACTA1-102610-Disease-NM3-161800"
        ),
        "locus": "1q42.13",
        "protein_size": "375 aa / 42 kDa",
        "inheritance": (
            "AD de novo (most common, most severe) or AR (milder); "
            "~50% de novo dominant: severe/congenital NM, respiratory failure at birth; "
            "~50% AR (biallelic LOF): typical NM, ambulant, milder course; "
            "CK: normal to 2× ULN; "
            "Onset: de novo dominant = at birth (congenital); AR = infancy-childhood; "
            "Intranuclear rods: pathognomonic for ACTA1 (actin aggregates inside nucleus on EM); "
            "Respiratory failure: critical in dominant severe form; NIV/ventilator from birth"
        ),
        "key_features": [
            "INTRANUCLEAR RODS — actin aggregates inside nucleus on electron microscopy; PATHOGNOMONIC for ACTA1",
            "De novo dominant = most severe: congenital respiratory failure, life-threatening without immediate ventilation",
            "AR ACTA1 = milder typical NM — ambulant; slower progression than dominant severe",
            "2nd most common NM gene after NEB; accounts for ~25% of NM overall",
            "CK: normal to mildly elevated (unlike muscular dystrophies with markedly elevated CK)",
            "Salbutamol (albuterol) — open-label trials: some patients gain strength; Level C evidence",
            "Biopsy: nemaline rods + intranuclear rods + type 1 fiber predominance",
            "Cardiac: usually spared but monitor; rare DCM reports in dominant forms",
        ],
        "treatment": (
            "Respiratory: immediate ventilatory support at birth if dominant severe form; long-term NIV/ventilator. "
            "Salbutamol (albuterol): 4 mg TID → 8 mg TID — open-label evidence of strength improvement; Level C. "
            "L-tyrosine: 100-200 mg/kg/day — some case reports, Level C. "
            "Physiotherapy: stretching, strengthening, contracture prevention. "
            "Feeding: gastrostomy if bulbar weakness severe. "
            "Cardiac: baseline echocardiogram; annual follow-up. "
            "Gene therapy: research phase for ACTA1."
        ),
        "monitoring": [
            "Respiratory: FVC monthly in severe forms; nocturnal oximetry; sleep study",
            "Cardiac: baseline echo + ECG; annual; Holter if palpitations",
            "Nutrition: gastrostomy assessment; SLP swallow study",
            "Physiotherapy: motor milestone tracking; contracture surveillance",
            "Genetic: de novo confirmation (parental testing); recurrence risk counseling",
            "Salbutamol: dose titration; heart rate monitoring (tachycardia side effect)",
            "Ophthalmology: ptosis — surgical if visual axis compromised",
        ],
    },
    # -- TPM2 — Tropomyosin β, allelic: NM + Cap Disease + DA2A/B -----------------------
    {
        "gene": "TPM2",
        "alt_name": (
            "TPM2 (TPM2-284aa-9p13.3 / AD / AR — Nemaline-Myopathy-Type4-Cap-Disease-DA2A-DA2B — "
            "ALLELIC-DISORDERS-One-Gene-Three-Phenotypes — "
            "CAP-DISEASE-Biopsy-PATHOGNOMONIC-TPM2 — "
            "GOF-Distal-Arthrogryposis-LOF-Nemaline)"
        ),
        "protein": (
            "TPM2 -- 9p13.3 AD / AR -- TPM2-284aa -- "
            "Tropomyosin-Beta-Chain-33kDa-Thin-Filament-Actin-Regulatory-Coiled-Coil -- "
            "NM-OMIM-609285 -- "
            "ALLELIC-NM-Type4-Cap-Disease-DA2A-DA2B -- "
            "CAP-DISEASE-Thin-Filament-Cap-at-Z-Disc-End-PATHOGNOMONIC-TPM2 -- "
            "TPM2-GOF-Mutations-Distal-Arthrogryposis-DA2A-DA2B -- "
            "TPM2-LOF-Mutations-Nemaline-Myopathy -- "
            "Fiber-Type-Disproportion-Overlap -- "
            "CK-Normal-to-2x -- "
            "OMIM-Gene-TPM2-190990-Disease-NM4-609285"
        ),
        "locus": "9p13.3",
        "protein_size": "284 aa / 33 kDa",
        "inheritance": (
            "AD or AR depending on mutation; "
            "ALLELIC gene: GOF mutations → distal arthrogryposis (DA2A/DA2B); "
            "LOF mutations → nemaline myopathy; "
            "Cap disease: AD dominant-negative; thin-filament capping at Z-disc; "
            "CK: normal to 2× ULN; "
            "Onset: congenital to childhood; "
            "Respiratory: mild-moderate; "
            "Facial weakness less prominent than NEB/ACTA1"
        ),
        "key_features": [
            "CAP DISEASE on muscle biopsy — abnormal thin filament caps at Z-disc ends; PATHOGNOMONIC for TPM2 cap disease alleles",
            "ALLELIC disorders: one gene → three distinct phenotypes depending on mutation class",
            "GOF mutations → Distal Arthrogryposis DA2A/DA2B (Freeman-Sheldon/Sheldon-Hall syndrome)",
            "LOF mutations → Typical nemaline myopathy (NM type 4)",
            "Fiber type disproportion overlap — type 1 fiber predominance",
            "CK normal to mildly elevated; slower progression than NEB/ACTA1 severe forms",
            "Contractures: joint contractures prominent in DA2A/DA2B forms",
            "Respiratory: mild-moderate; facial weakness less than other NM genes",
        ],
        "treatment": (
            "NM form: physiotherapy; respiratory monitoring; NIV if needed. "
            "DA2 form: serial casting for contractures; surgical release if severe; hand surgery. "
            "Splinting: ankles, hands — prevent fixed deformities. "
            "Salbutamol: limited data for TPM2 NM specifically. "
            "Feeding: usually not severely compromised; SLP assessment. "
            "Genetic counseling: distinguish GOF (DA2) from LOF (NM) — management differs critically."
        ),
        "monitoring": [
            "Respiratory: FVC every 12 months; nocturnal oximetry if symptomatic",
            "Joint: orthopedic review for contracture progression (DA2 forms)",
            "Physiotherapy: contracture prevention; stretching programme",
            "Genetic: phenotype-genotype correlation critical (GOF vs LOF)",
            "Swallowing: SLP assessment; aspiration risk low in mild forms",
            "Growth: annual anthropometrics",
        ],
    },
    # -- TPM3 — Tropomyosin α-slow, dominant/recessive NM --------------------------------
    {
        "gene": "TPM3",
        "alt_name": (
            "TPM3 (TPM3-285aa-1q21.2 / AD / AR — Typical-Nemaline-Myopathy-Type1 — "
            "SLOW-TWITCH-FIBER-SELECTIVE-Tropomyosin — "
            "Fiber-Type-Disproportion-Overlap-Cap-Disease-AD — "
            "Slower-Progression-Exercise-Intolerance)"
        ),
        "protein": (
            "TPM3 -- 1q21.2 AD / AR -- TPM3-285aa -- "
            "Tropomyosin-Alpha-Slow-Chain-33kDa-Slow-Twitch-Fiber-Selective-Thin-Filament -- "
            "NM1-OMIM-609284 -- "
            "SLOW-TWITCH-FIBER-SELECTIVE-TPM3-Predominantly-Type-1-Fibers -- "
            "AD-CAP-DISEASE-Thin-Filament-Caps-Biopsy-PATHOGNOMONIC-TPM3 -- "
            "AR-TYPICAL-NM-Rods -- "
            "FIBER-TYPE-DISPROPORTION-Overlap-Phenotype -- "
            "Exercise-Intolerance-Proximal-Weakness-Slower-Progression -- "
            "CK-Normal -- "
            "OMIM-Gene-TPM3-191030-Disease-NM1-609284"
        ),
        "locus": "1q21.2",
        "protein_size": "285 aa / 33 kDa",
        "inheritance": (
            "AD or AR; "
            "AD: cap disease + fiber type disproportion; dominant-negative effect; "
            "AR: typical nemaline myopathy with rods; "
            "CK: normal (distinguishing feature); "
            "Onset: congenital to childhood; some adult-onset; "
            "Slower progression than NEB/ACTA1 severe forms; "
            "Exercise intolerance prominent; "
            "Respiratory: mild-moderate; "
            "Facial weakness variable"
        ),
        "key_features": [
            "SLOW-TWITCH FIBER SELECTIVE — TPM3 is expressed only in slow-twitch (type 1) fibers; selective weakness pattern",
            "AD cap disease — dominant-negative; thin filament caps on biopsy; PATHOGNOMONIC TPM3 cap alleles",
            "AR form — typical NM with rods; milder than NEB/ACTA1",
            "Fiber type disproportion — type 1 fiber predominance + small size (congenital fiber type disproportion overlap)",
            "CK: NORMAL — differentiates from dystrophies; exercise intolerance without markedly raised CK",
            "Slower progression — many patients ambulant into adulthood",
            "Respiratory: mild-moderate; NIV sometimes needed; slower trajectory",
            "Exercise intolerance: fatigue after mild exertion; PTH (exercise testing) useful",
        ],
        "treatment": (
            "Physiotherapy: strengthening; aerobic conditioning; contracture prevention. "
            "Respiratory: FVC monitoring; NIV if nocturnal hypoventilation. "
            "Splinting: AFO for footdrop. "
            "Salbutamol: limited data for TPM3; may benefit. "
            "Genetic counseling: AD vs AR — offspring risk differs. "
            "No disease-modifying therapy approved 2026."
        ),
        "monitoring": [
            "Respiratory: FVC every 12 months; nocturnal oximetry",
            "Motor: physiotherapy; 6-minute walk test; motor milestone tracking",
            "Nutrition: body weight; height velocity",
            "Cardiac: baseline echocardiogram; repeat if symptoms",
            "Genetic: cascade testing in AD families; prenatal counseling",
        ],
    },
    # -- TNNT1 — Troponin T1 slow, Amish Lethal NM ---------------------------------------
    {
        "gene": "TNNT1",
        "alt_name": (
            "TNNT1 (TNNT1-328aa-19q13.42 / AR — Nemaline-Myopathy-Amish — "
            "E180X-AMISH-FOUNDER-LETHAL-INFANTILE-PATHOGNOMONIC — "
            "FATAL-BY-2-3-YEARS-Amish-Homozygous — "
            "Non-Amish-Milder-E180K-Alleles)"
        ),
        "protein": (
            "TNNT1 -- 19q13.42 AR -- TNNT1-328aa -- "
            "Troponin-T1-Slow-Skeletal-Isoform-36kDa-Thin-Filament-Regulatory-Complex -- "
            "NM-Amish-OMIM-605355 -- "
            "E180X-AMISH-FOUNDER-NONSENSE-Homozygous-Lethal -- "
            "LETHAL-BY-2-3-YEARS-Amish-Homozygous-Rapidly-Progressive -- "
            "NON-AMISH-E180K-MILDER-MISSENSE-Longer-Survival -- "
            "Congenital-Hypotonia-Nemaline-Rods-Biopsy -- "
            "CK-Normal-to-Mildly-Elevated -- "
            "OMIM-Gene-TNNT1-191041-Disease-NM5-605355"
        ),
        "locus": "19q13.42",
        "protein_size": "328 aa / 36 kDa",
        "inheritance": (
            "AR (biallelic); "
            "E180X (Glu180Stop): Amish founder mutation — homozygous → lethal infantile NM; "
            "Fatal typically by 2-3 years in Amish homozygotes (progressive respiratory failure); "
            "E180K (Glu180Lys): non-Amish missense → milder, longer survival; "
            "CK: normal to mildly elevated; "
            "Onset: at birth; "
            "Nemaline rods on biopsy; "
            "Respiratory failure: dominant feature; progressive; "
            "No intellectual disability"
        ),
        "key_features": [
            "E180X AMISH FOUNDER — homozygous nonsense mutation; ~1/800 Amish births affected; LETHAL INFANTILE",
            "FATAL BY 2-3 YEARS in Amish E180X homozygotes — rapidly progressive respiratory failure; palliative care discussions critical",
            "E180K non-Amish allele — missense; milder phenotype; longer survival possible",
            "Congenital hypotonia — profound at birth; non-ambulant in severe forms",
            "Nemaline rods on Gomori trichrome — Z-disc derived; troponin complex disruption",
            "Respiratory failure — progressive; NIV rarely sufficient in E180X; ventilator often needed",
            "CK normal — not a dystrophy; enzyme pattern helps exclude dystrophies",
            "Genetic counseling: Amish founder effect; population carrier screening feasible",
        ],
        "treatment": (
            "E180X Amish: palliative care discussion early; goals of care (ventilator/non-ventilator); "
            "Respiratory: aggressive NIV initiation; mechanical ventilation consideration. "
            "Feeding: gastrostomy early given bulbar + respiratory compromise. "
            "Physiotherapy: comfort-focused; range of motion. "
            "Genetic: cascade Amish community; carrier screening available. "
            "No disease-modifying therapy 2026; gene therapy research phase. "
            "Non-Amish E180K: standard NM management with respiratory surveillance."
        ),
        "monitoring": [
            "Respiratory: monthly FVC in E180X; nocturnal polysomnography; blood gas",
            "Feeding: weight gain; swallow study; gastrostomy timing discussion",
            "Goals of care: family meetings; advance care planning (E180X Amish)",
            "Genetic: Amish community carrier testing; prenatal diagnosis if E180X family",
            "Physiotherapy: contracture prevention; comfort positioning",
        ],
    },
    # -- CFL2 — Cofilin-2, NM type 4 with cores and rods --------------------------------
    {
        "gene": "CFL2",
        "alt_name": (
            "CFL2 (CFL2-166aa-14q13.1 / AR — Nemaline-Myopathy-Type4-Cofilin-2 — "
            "CORES-AND-RODS-Biopsy-PATHOGNOMONIC-CFL2 — "
            "Actin-Depolymerizing-Factor-Thin-Filament-Regulation — "
            "Proximal-Weakness-Respiratory-Moderate-CK-Mildly-Elevated)"
        ),
        "protein": (
            "CFL2 -- 14q13.1 AR -- CFL2-166aa -- "
            "Cofilin-2-Muscle-Isoform-19kDa-Actin-Depolymerizing-Factor-Thin-Filament-Dynamics -- "
            "NM7-OMIM-610687 -- "
            "CORES-AND-RODS-On-Biopsy-Combination-PATHOGNOMONIC-CFL2 -- "
            "Proximal-Weakness-Greater-Than-Distal -- "
            "Respiratory-Failure-Moderate -- "
            "CK-Normal-to-Mildly-Elevated -- "
            "Childhood-to-Adult-Onset -- "
            "OMIM-Gene-CFL2-601443-Disease-NM7-610687"
        ),
        "locus": "14q13.1",
        "protein_size": "166 aa / 19 kDa",
        "inheritance": (
            "AR (biallelic loss-of-function); "
            "Rare; CFL2 = muscle-specific isoform of cofilin (actin-depolymerizing factor); "
            "CK: normal to mildly elevated; "
            "Onset: childhood to adult; "
            "Biopsy: CORES + RODS (combination — distinguishing from pure NEB/ACTA1); "
            "Proximal > distal weakness; "
            "Respiratory failure: moderate; slower trajectory than NEB/ACTA1 severe"
        ),
        "key_features": [
            "CORES AND RODS on biopsy — combination of cores (like central core disease) + nemaline rods; PATHOGNOMONIC for CFL2",
            "CFL2 = muscle-specific cofilin — actin dynamics (depolymerization); disrupted thin filament turnover",
            "Proximal > distal weakness — proximal limb girdle pattern; pelvic + shoulder girdle",
            "Respiratory failure: moderate; slower than NEB/ACTA1; NIV often deferred",
            "CK mildly elevated or normal; differentiates from muscular dystrophies",
            "Childhood to adult onset — later than typical NEB/ACTA1",
            "Rare gene: WES important — panel may miss atypical intronic/splice variants",
            "Genetic counseling: AR pattern; sibling risk 25%",
        ],
        "treatment": (
            "Respiratory: FVC monitoring; NIV if nocturnal hypoventilation. "
            "Physiotherapy: proximal muscle strengthening; aerobic conditioning. "
            "Splinting: AFO for footdrop if distal involvement. "
            "Genetic counseling: AR; sibling cascade; prenatal options. "
            "No specific therapy 2026."
        ),
        "monitoring": [
            "Respiratory: FVC every 12 months; nocturnal oximetry",
            "Motor: 6-minute walk test; timed up-and-go",
            "Cardiac: baseline echocardiogram; repeat if symptoms",
            "Physiotherapy: annual functional assessment",
            "Genetic: WES if panel negative; RNA studies if splice variants suspected",
        ],
    },
    # -- KBTBD13 — NM type 6, slow fiber, exercise intolerance -------------------------
    {
        "gene": "KBTBD13",
        "alt_name": (
            "KBTBD13 (KBTBD13-571aa-15q22.31 / AD — Nemaline-Myopathy-Type6 — "
            "SLOW-RELAXATION-EXERCISE-INTOLERANCE-PATHOGNOMONIC — "
            "Slow-Fiber-Selective-Dutch-Founder — "
            "Mild-Course-CK-Normal-Electrically-Silent-Slow-Relaxation)"
        ),
        "protein": (
            "KBTBD13 -- 15q22.31 AD -- KBTBD13-571aa -- "
            "Kelch-BTB-Domain-Protein-13-64kDa-Ubiquitin-Substrate-Adaptor-Thin-Filament-Regulation -- "
            "NM6-OMIM-609273 -- "
            "SLOW-RELAXATION-TIME-PATHOGNOMONIC-Grip-Myotonia-Like-Electrically-Silent-EMG -- "
            "EXERCISE-INTOLERANCE-Fatigue-With-Minimal-Exertion -- "
            "SLOW-TWITCH-FIBER-SELECTIVE-Disease -- "
            "DUTCH-FOUNDER-HAPLOTYPE -- "
            "CK-Normal -- "
            "Mild-Course-Ambulant-Adulthood -- "
            "OMIM-Gene-KBTBD13-613727-Disease-NM6-609273"
        ),
        "locus": "15q22.31",
        "protein_size": "571 aa / 64 kDa",
        "inheritance": (
            "AD (dominant-negative or haploinsufficiency); "
            "Dutch founder haplotype; "
            "SLOW RELAXATION TIME — pathognomonic; electrically silent (NOT myotonia on EMG); "
            "CK: NORMAL; "
            "Onset: childhood; "
            "Mild course — ambulant throughout adulthood; "
            "Exercise intolerance: fatigue out of proportion to weakness; "
            "Slow-twitch fiber-selective disease"
        ),
        "key_features": [
            "SLOW RELAXATION — pathognomonic: grip-myotonia-like but ELECTRICALLY SILENT on EMG (not true myotonia)",
            "EXERCISE INTOLERANCE — fatigue disproportionate to weakness; key clinical clue",
            "Slow-twitch fiber selective — type 1 fiber predominance on biopsy; slow fiber atrophy",
            "Dutch founder haplotype — concentrated in Netherlands; founder effect population",
            "CK: NORMAL — important distinguishing feature; NOT a muscular dystrophy",
            "Mild course — ambulant throughout adulthood; not wheelchair-dependent typically",
            "EMG distinguishes from myotonic disorders: NO repetitive discharges; electrically silent slow relaxation",
            "Biopsy: nemaline rods (may be sparse) + type 1 fiber predominance",
        ],
        "treatment": (
            "Physiotherapy: aerobic conditioning; fatigue management strategies; pacing. "
            "Slow relaxation: avoid situations requiring rapid grip release (machinery, driving awareness). "
            "Occupational therapy: adaptation for slow relaxation in daily activities. "
            "No pharmacological therapy proven 2026; salbutamol trial data minimal for KBTBD13. "
            "Respiratory: monitor (mild risk); FVC yearly. "
            "Genetic counseling: AD; 50% offspring risk."
        ),
        "monitoring": [
            "Respiratory: FVC every 12 months; nocturnal oximetry if symptomatic",
            "Motor: fatigue scales; exercise tolerance testing",
            "EMG: repeat if EMG-myotonia suspected — should be electrically silent in KBTBD13",
            "Genetic: family cascade; prenatal counseling if requested",
            "Physiotherapy: annual fatigue + strength assessment",
        ],
    },
    # -- LMOD3 — Leiomodin-3, NM type 10, severe congenital ----------------------------
    {
        "gene": "LMOD3",
        "alt_name": (
            "LMOD3 (LMOD3-547aa-3p14.1 / AR — Nemaline-Myopathy-Type10 — "
            "Thin-Filament-Elongation-Factor-Leiomodin-3 — "
            "Severe-Congenital-NM-DCM-Possible-Respiratory-Failure-Early — "
            "Rods-Minimal-Filaments-Biopsy)"
        ),
        "protein": (
            "LMOD3 -- 3p14.1 AR -- LMOD3-547aa -- "
            "Leiomodin-3-64kDa-Sarcomeric-Thin-Filament-Elongation-Nucleation-Factor -- "
            "NM10-OMIM-616165 -- "
            "THIN-FILAMENT-ELONGATION-Pointed-End-Actin-Polymerization-Opposite-Formin -- "
            "SEVERE-CONGENITAL-NM-Respiratory-Failure-Birth -- "
            "DCM-POSSIBLE-Cardiac-Involvement-Mandatory-Monitor -- "
            "RODS-AND-MINIMAL-FILAMENTS-Biopsy -- "
            "CK-Normal-to-Mildly-Elevated -- "
            "OMIM-Gene-LMOD3-616112-Disease-NM10-616165"
        ),
        "locus": "3p14.1",
        "protein_size": "547 aa / 64 kDa",
        "inheritance": (
            "AR (biallelic loss-of-function); "
            "LMOD3 = pointed-end actin polymerization factor (opposite to formin at barbed end); "
            "Severe congenital NM with respiratory failure at birth; "
            "Cardiac: DCM possible — mandatory monitoring (distinguishes from many NM genes); "
            "CK: normal to mildly elevated; "
            "Biopsy: rods + minimal thin filaments (reduced thin filament content); "
            "Onset: birth; congenital severe"
        ),
        "key_features": [
            "THIN FILAMENT ELONGATION FACTOR — LMOD3 polymerizes actin at pointed end; LOF → short thin filaments → rods + reduced filament content",
            "SEVERE CONGENITAL NM — respiratory failure at birth; life-threatening without ventilation",
            "DCM POSSIBLE — cardiac involvement; echocardiogram mandatory from diagnosis (distinguishes from NEB/ACTA1 cardiac-sparing)",
            "RODS + MINIMAL FILAMENTS on biopsy — reduced thin filament density visible on EM (unusual biopsy pattern)",
            "CK normal to mildly elevated; not markedly raised",
            "AR — both parents carriers; sibling risk 25%",
            "Respiratory: immediate ventilatory support at birth; long-term NIV/ventilator",
            "Rare gene: WES important; RNA studies if splice variants",
        ],
        "treatment": (
            "Respiratory: immediate ventilatory support at birth; assess goals of care early. "
            "Cardiac: echocardiogram baseline; ACE inhibitor + beta-blocker if DCM. "
            "Feeding: gastrostomy early given respiratory + bulbar compromise. "
            "Physiotherapy: comfort-focused in severe forms; range of motion. "
            "Genetic counseling: AR; parental carrier testing; prenatal diagnosis available. "
            "No disease-modifying therapy 2026; research phase."
        ),
        "monitoring": [
            "Cardiac: echocardiogram every 6 months; ECG; Holter if DCM",
            "Respiratory: blood gas; nocturnal polysomnography; ventilator settings review",
            "Feeding: weight gain; gastrostomy function; SLP review",
            "Goals of care: family meetings; advance care planning",
            "Genetic: sibling cascade; prenatal counseling; WES if panel negative",
        ],
    },
]

# ──────────────────────────────────────────────────────────────────
# Patient simulation
# ──────────────────────────────────────────────────────────────────
def _make_cohort(gene_entry: dict, seed: int) -> list[dict]:
    random.seed(seed)
    gene = gene_entry["gene"]
    n = 40
    patients = []
    for i in range(n):
        pid = f"{gene}-{seed}-{i+1:03d}"
        onset_age_options = [0, 0, 0, 1, 2, 3, 5, 8, 12, 18, 25, 35]
        onset_age = random.choice(onset_age_options)
        current_age = onset_age + random.randint(2, 40)
        ck_map = {
            "NEB":     (lambda: random.randint(50, 350)),
            "ACTA1":   (lambda: random.randint(50, 250)),
            "TPM2":    (lambda: random.randint(40, 200)),
            "TPM3":    (lambda: random.randint(40, 180)),
            "TNNT1":   (lambda: random.randint(50, 300)),
            "CFL2":    (lambda: random.randint(60, 350)),
            "KBTBD13": (lambda: random.randint(40, 120)),
            "LMOD3":   (lambda: random.randint(50, 280)),
        }
        ck = ck_map.get(gene, lambda: random.randint(50, 300))()
        ambulant = random.random() < (0.75 if gene not in ("TNNT1", "LMOD3") else 0.25)
        niv = random.random() < (0.80 if gene in ("NEB", "ACTA1", "TNNT1", "LMOD3") else 0.45)
        cardiac = random.random() < (0.25 if gene == "LMOD3" else 0.08)
        patients.append({
            "id": pid,
            "gene": gene,
            "onset_age": onset_age,
            "current_age": current_age,
            "ck_iul": ck,
            "ambulant": ambulant,
            "niv": niv,
            "cardiac_involvement": cardiac,
            "biopsy_rods": True,
            "biopsy_intranuclear_rods": gene == "ACTA1" and random.random() < 0.60,
            "biopsy_cores_and_rods": gene == "CFL2" and random.random() < 0.80,
            "biopsy_cap_disease": gene in ("TPM2", "TPM3") and random.random() < 0.45,
            "slow_relaxation": gene == "KBTBD13" and random.random() < 0.95,
            "respiratory_support": niv,
        })
    return patients


def _aggregate_cohort():
    all_patients = []
    for idx, entry in enumerate(NM_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))
    return all_patients


# ──────────────────────────────────────────────────────────────────
# API response builders
# ──────────────────────────────────────────────────────────────────
def overview() -> dict:
    cohort = _aggregate_cohort()
    total = len(cohort)
    ambulant = sum(1 for p in cohort if p["ambulant"])
    niv = sum(1 for p in cohort if p["niv"])
    cardiac = sum(1 for p in cohort if p["cardiac_involvement"])
    intranuclear = sum(1 for p in cohort if p["biopsy_intranuclear_rods"])
    cores_and_rods = sum(1 for p in cohort if p["biopsy_cores_and_rods"])
    slow_relax = sum(1 for p in cohort if p["slow_relaxation"])
    avg_onset = round(sum(p["onset_age"] for p in cohort) / total, 1)
    avg_ck = round(sum(p["ck_iul"] for p in cohort) / total, 0)

    gene_counts = {}
    for g in NM_GENES:
        gn = g["gene"]
        subset = [p for p in cohort if p["gene"] == gn]
        gene_counts[gn] = {
            "n": len(subset),
            "ambulant_pct": round(100 * sum(1 for p in subset if p["ambulant"]) / len(subset)),
            "niv_pct":      round(100 * sum(1 for p in subset if p["niv"]) / len(subset)),
            "cardiac_pct":  round(100 * sum(1 for p in subset if p["cardiac_involvement"]) / len(subset)),
            "avg_onset":    round(sum(p["onset_age"] for p in subset) / len(subset), 1),
            "avg_ck":       round(sum(p["ck_iul"] for p in subset) / len(subset), 0),
        }

    return {
        "atlas": "Hereditary-Nemaline-Myopathy-Atlas",
        "subtitle": "Complete 8-Gene Nemaline Myopathy (NM) Spectrum Atlas",
        "genes": [g["gene"] for g in NM_GENES],
        "gene_count": len(NM_GENES),
        "total_patients": total,
        "seeds": list(range(SEED_BASE, SEED_BASE + len(NM_GENES))),
        "kpis": {
            "total_patients": total,
            "ambulant_pct": round(100 * ambulant / total),
            "niv_pct": round(100 * niv / total),
            "cardiac_involvement_pct": round(100 * cardiac / total),
            "intranuclear_rods_pct": round(100 * intranuclear / total),
            "cores_and_rods_pct": round(100 * cores_and_rods / total),
            "slow_relaxation_pct": round(100 * slow_relax / total),
            "avg_onset_years": avg_onset,
            "avg_ck_iul": int(avg_ck),
        },
        "gene_summary": gene_counts,
        "pathognomonic_features": {
            "NEB":     "NEMALINE RODS on Gomori trichrome — Z-disc-derived rods",
            "ACTA1":   "INTRANUCLEAR RODS — actin aggregates inside nucleus on EM",
            "TPM2":    "CAP DISEASE — thin filament caps at Z-disc ends on biopsy",
            "TPM3":    "FIBER TYPE DISPROPORTION + slow fiber selectivity",
            "TNNT1":   "E180X AMISH FOUNDER — lethal infantile; E180K milder non-Amish",
            "CFL2":    "CORES AND RODS — combination on biopsy pathognomonic for CFL2",
            "KBTBD13": "SLOW RELAXATION — electrically silent (not myotonia) + exercise intolerance",
            "LMOD3":   "RODS + MINIMAL THIN FILAMENTS — reduced filament density on EM",
        },
        "inheritance_map": {
            "NEB": "AR", "ACTA1": "AD-de-novo/AR", "TPM2": "AD/AR",
            "TPM3": "AD/AR", "TNNT1": "AR", "CFL2": "AR",
            "KBTBD13": "AD", "LMOD3": "AR",
        },
        "protein_sizes": {g["gene"]: g["protein_size"] for g in NM_GENES},
        "loci": {g["gene"]: g["locus"] for g in NM_GENES},
        "key_distinctions": [
            "ACTA1 intranuclear rods (EM) — only gene with actin inside nucleus",
            "KBTBD13 slow relaxation — electrically silent, NOT myotonia on EMG",
            "TNNT1 Amish E180X — lethal infantile; founder effect counseling",
            "LMOD3 DCM possible — only NM gene with significant cardiac risk",
            "CFL2 cores+rods — overlap with core myopathy on biopsy",
            "NEB exon 55 deletion — Ashkenazi Jewish founder PCR screen available",
            "TPM2 allelic: GOF→DA2, LOF→NM — phenotype depends on mutation class",
            "KBTBD13 CK NORMAL — distinguishes from dystrophies; fatigue out of proportion",
        ],
        "critical_treatments": {
            "NEB":     "Respiratory NIV monitoring; salbutamol Level C",
            "ACTA1":   "Salbutamol Level C (strength improvement); respiratory support from birth",
            "TPM2":    "DA2 form: serial casting contractures; NM form: respiratory monitor",
            "TPM3":    "Physiotherapy aerobic conditioning; FVC annually",
            "TNNT1":   "E180X: palliative/goals of care; ventilator consideration; gastrostomy",
            "CFL2":    "Respiratory monitor; physiotherapy; cascade testing AR",
            "KBTBD13": "Pacing/fatigue management; slow relaxation safety (machinery/driving)",
            "LMOD3":   "Cardiac (ACE-i + BB if DCM); respiratory from birth; gastrostomy",
        },
    }


def breakdown() -> dict:
    cohort = _aggregate_cohort()
    patients_out = []
    for p in cohort:
        entry = next(g for g in NM_GENES if g["gene"] == p["gene"])
        patients_out.append({
            **p,
            "protein": entry["protein"],
            "alt_name": entry["alt_name"],
            "inheritance": entry["inheritance"],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:300],
        })
    return {
        "atlas": "Hereditary-Nemaline-Myopathy-Atlas",
        "total": len(patients_out),
        "patients": patients_out,
        "gene_profiles": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "key_features": g["key_features"],
                "treatment": g["treatment"],
                "monitoring": g["monitoring"],
            }
            for g in NM_GENES
        ],
    }


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Nemaline-Myopathy-Atlas",
        "glossary": {
            "Nemaline Myopathy (NM)": (
                "Hereditary congenital myopathy defined by nemaline rods on muscle biopsy (Gomori modified trichrome). "
                "Caused by mutations in thin filament proteins (NEB, ACTA1, TPM2, TPM3, TNNT1, CFL2, KBTBD13, LMOD3). "
                "Clinical spectrum: lethal congenital → mild adult. "
                "OMIM: multiple loci; unifying pathology = nemaline rods."
            ),
            "Nemaline Rods": (
                "Electron-dense bodies derived from Z-disc material (alpha-actinin, titin, actin). "
                "Seen on Gomori modified trichrome as red-purple rods in pale sarcoplasm. "
                "Subsarcolemmal + intrasarcoplasmic locations. "
                "NOT specific to one gene — seen in NEB, ACTA1, TPM2, TPM3, TNNT1, CFL2, LMOD3."
            ),
            "Intranuclear Rods (ACTA1)": (
                "Actin aggregates inside the nucleus — PATHOGNOMONIC for ACTA1 NM. "
                "Visible on electron microscopy (EM). "
                "Not seen in other NM genes. "
                "De novo dominant ACTA1 = most severe."
            ),
            "Cap Disease (TPM2/TPM3)": (
                "Abnormal caps of thin filament material at sarcomere Z-disc ends. "
                "Distinct from typical nemaline rods. "
                "PATHOGNOMONIC for TPM2/TPM3 cap-disease alleles. "
                "Seen in dominant-negative mutations causing dominant inheritance."
            ),
            "Cores and Rods (CFL2)": (
                "Combination of core myopathy features (mitochondria-depleted areas) + nemaline rods. "
                "PATHOGNOMONIC combination for CFL2 mutations. "
                "Overlap with central core / multi-minicore disease on biopsy."
            ),
            "Slow Relaxation (KBTBD13)": (
                "Delayed muscle relaxation after contraction — grip myotonia-like clinically. "
                "ELECTRICALLY SILENT on EMG (NOT true myotonia). "
                "Pathognomonic for KBTBD13. "
                "Due to slow-twitch fiber-selective disease; slow fiber kinetics impaired."
            ),
            "NEB Exon 55 Deletion": (
                "Founder deletion in NEB gene — Ashkenazi Jewish population ~5% carrier frequency. "
                "Homozygous deletion → typical nemaline myopathy (NM2). "
                "PCR-based test available for rapid screening. "
                "NEB is the largest human gene (183 exons, 6669 aa)."
            ),
            "E180X Amish Founder (TNNT1)": (
                "Nonsense mutation Glu180Stop in TNNT1. "
                "Amish founder mutation — 1/800 Amish births affected. "
                "Homozygous → lethal infantile NM; fatal typically 2-3 years. "
                "E180K = non-Amish missense → milder, longer survival."
            ),
            "Thin Filament": (
                "Sarcomeric filament composed of actin (ACTA1), tropomyosin (TPM2/TPM3), troponin complex (TNNT1). "
                "Regulated by Ca²⁺ during muscle contraction. "
                "NM genes encode thin filament proteins → impaired regulation → nemaline rods."
            ),
            "Gomori Modified Trichrome": (
                "Histochemical stain for muscle biopsy. "
                "Nemaline rods appear red-purple in pale sarcoplasm. "
                "Gold standard for NM diagnosis. "
                "Electron microscopy (EM) confirms rod ultrastructure + intranuclear rods."
            ),
            "Salbutamol (NM treatment)": (
                "β2-adrenoceptor agonist. "
                "Open-label trials show strength improvement in NEB/ACTA1 NM. "
                "Level C evidence (expert opinion + open-label trials). "
                "Mechanism: anabolic effect on skeletal muscle via β2-adrenoceptor. "
                "Side effects: tachycardia, tremor — dose-titrate (4 mg → 8 mg TID)."
            ),
            "Leiomodin-3 (LMOD3)": (
                "Thin filament elongation/nucleation factor at pointed end of actin. "
                "Opposite function to formin (which works at barbed end). "
                "LOF → short thin filaments → nemaline rods + reduced filament content on EM. "
                "Only NM gene associated with significant DCM risk."
            ),
            "DCM in NM (LMOD3)": (
                "Dilated cardiomyopathy — rare in most NM genes. "
                "LMOD3: DCM possible (~25% in severe forms); mandatory echocardiogram monitoring. "
                "Treatment: ACE inhibitor + beta-blocker standard HF protocol. "
                "Distinguishes LMOD3 from other NM genes."
            ),
        },
        "diagnostic_algorithm": [
            "1. Congenital hypotonia + proximal weakness → muscle biopsy (Gomori modified trichrome)",
            "2. Nemaline rods confirmed → NM diagnosis; classify severity (congenital/typical/mild)",
            "3. CK: if >10× ULN → consider dystrophy DDx; NM CK usually normal–mild elevation",
            "4. Biopsy subtypes: intranuclear rods (ACTA1), cores+rods (CFL2), cap disease (TPM2/TPM3)",
            "5. KBTBD13: slow relaxation + electrically silent EMG → targeted sequencing",
            "6. Targeted panel: NEB (include exon 55 deletion assay if Ashkenazi), ACTA1, TPM2, TPM3, TNNT1, CFL2, KBTBD13, LMOD3",
            "7. WES if panel negative; RNA studies for NEB intronic/splice variants",
            "8. TNNT1 E180X → Amish population-specific screening",
            "9. LMOD3 confirmed → echocardiogram mandatory",
            "10. Respiratory assessment: FVC sitting+supine, nocturnal oximetry, sleep study",
        ],
        "references": [
            "Sewry CA et al. Nemaline myopathies: a current view. Neuropathol Appl Neurobiol. 2019.",
            "Wallgren-Pettersson C et al. Nemaline myopathies. Semin Pediatr Neurol. 2011.",
            "Nowak KJ et al. Mutations in the skeletal muscle alpha-actin gene in patients with actin myopathy and nemaline myopathy. Nat Genet. 1999.",
            "Johnston JJ et al. A point mutation in the human murine leukemia virus receptor gene Pit2 causes thrombocythemia. Blood. 2004. [TNNT1 ref: Donner K et al. Mutations in the β-tropomyosin (TPM2) gene. Neuromuscul Disord. 2002.]",
            "Ravenscroft G et al. Mutations in LMOD3 cause nemaline myopathy and implicate LMOD3 in thin filament assembly. Am J Hum Genet. 2013.",
            "Garg A et al. KBTBD13 interacts with Cullin3 to form a functional ubiquitin ligase. Sci Rep. 2014.",
        ],
        "standards": [
            "ENMC NM International Consortium (Neuromuscul Disord)",
            "TREAT-NMD NM Network — diagnosis/management guidelines",
            "European Neuromuscular Centre (ENMC) Workshop Reports — NM",
            "ACMG/AMP Variant Classification 2015 — pathogenicity criteria",
            "ILAE neonatal seizure classification (TNNT1 Amish — seizures possible)",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== DEFINITIONS ===")
    print(json.dumps(definitions(), indent=2)[:2000])
