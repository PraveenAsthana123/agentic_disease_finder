#!/usr/bin/env python3
"""Hereditary-Congenital-Myopathy-Atlas — Complete 8-Gene Hereditary Congenital Myopathy Atlas
RYR1    (ryanodine receptor 1; 5037 aa; 19q13.2; AD/AR;
         Central Core Disease (CCD) / Multi-minicore Disease (MmD) / Malignant Hyperthermia Susceptibility (MHS1);
         ANAESTHESIA EMERGENCY — MHS triggers succinylcholine + volatile agents;
         dantrolene IV mandatory on MH cart; seed SEED_BASE+0) ·
MYH7    (myosin heavy chain 7 (slow/beta); 1935 aa; 14q11.2; AD;
         Laing Distal Myopathy (MPD1) / Hyaline Body Myopathy / Scapulo-limb-girdle + DCM;
         CARDIAC SURVEILLANCE MANDATORY — same gene causes DCM and LVNC;
         annual ECG + echo; seed SEED_BASE+1) ·
ACTA1   (alpha-skeletal actin; 377 aa; 1q42.13; AD 70% de novo / AR;
         Nemaline Myopathy 3 (NEM3) — rod bodies on biopsy;
         RESPIRATORY SURVEILLANCE MANDATORY — neonatal critical in severe form;
         70% de novo AD; seed SEED_BASE+2) ·
NEB     (nebulin; 6669 aa; 2q23.3; AR;
         Nemaline Myopathy 2 (NEM2) — LARGEST human gene;
         rod bodies + ankle contractures + foot drop;
         seed SEED_BASE+3) ·
SELENON (selenoprotein N; 590 aa; 1p36.11; AR;
         Rigid Spine Muscular Dystrophy (RSMD1) / Multi-minicore Disease;
         RIGID SPINE + EARLY SCOLIOSIS — NO cardiac involvement KEY DDx from EMD/LMNA;
         seed SEED_BASE+4) ·
MTM1    (myotubularin; 603 aa; Xq28; X-linked recessive;
         X-linked Myotubular Myopathy (XLMTM) — most severe congenital myopathy;
         HIGH NEONATAL MORTALITY — ventilator dependent; MTM1 gene therapy trials;
         seed SEED_BASE+5) ·
DNM2    (dynamin 2; 870 aa; 19p13.2; AD;
         Centronuclear Myopathy (CNM3) / CMT-dominant intermediate type B (CMTDIB);
         GTPase dynamin — central nuclei on biopsy; CMT overlap;
         seed SEED_BASE+6) ·
BIN1    (bridging integrator 1 / amphiphysin 2; 452 aa; 2q14.3; AR / AD rare;
         Centronuclear Myopathy (CNM2) — T-tubule biogenesis;
         NO cardiac involvement; milder than MTM1;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1726–1733)
"""

import random

SEED_BASE = 1726

CM_GENES = [
    # ── RYR1 — CCD / MmD / MHS ────────────────────────────────────────────────
    {
        "gene": "RYR1",
        "protein": (
            "RYR1 — 19q13.2 AD/AR — Ryanodine-Receptor-1-5037aa — "
            "CentralCoreDisease-CCD / Multi-minioreCoreDisease-MmD / "
            "MalignantHyperthermia-MHS1 — "
            "ANAESTHESIA-EMERGENCY-Dantrolene-MH-Cart-Mandatory"
        ),
        "alias": (
            "RYR1 (ryanodine receptor 1, skeletal muscle); OMIM gene 180901; "
            "CCD (central core disease) OMIM 117000; MmD (multi-minicore disease) OMIM 255320; "
            "MHS1 (malignant hyperthermia susceptibility 1) OMIM 145600. "
            "19q13.2; 5037 aa; ~565 kDa; autosomal dominant (CCD, MHS1) or AR (MmD, severe CCD). "
            "FUNCTION: RYR1 encodes the skeletal muscle ryanodine receptor — a massive homotetrameric "
            "calcium release channel in the sarcoplasmic reticulum (SR) membrane. "
            "It is the principal SR Ca2+ release channel mediating excitation-contraction (E-C) coupling: "
            "action potential → DHPR (dihydropyridine receptor in T-tubule) → RYR1 opening → "
            "SR Ca2+ flood → myosin-actin cross-bridge cycling → muscle contraction. "
            "CCD MECHANISM: AD GOF mutations in C-terminal transmembrane domain cause constitutively "
            "open RYR1 channels → SR Ca2+ depletion → impaired E-C coupling → muscle weakness; "
            "central cores (amorphous, mitochondria-free zones) on Gomori trichrome biopsy; "
            "cores form because RYR1 dysfunction impairs mitochondrial function locally. "
            "MmD MECHANISM: AR LOF mutations (or some AD) cause channel hypersensitivity or leakiness; "
            "multiple small cores ('minicores') on biopsy rather than single central core. "
            "MHS MECHANISM: GOF mutations cause RYR1 hyperactivation on exposure to "
            "volatile anaesthetics (halothane, sevoflurane, isoflurane, desflurane) or "
            "succinylcholine → uncontrolled SR Ca2+ release → sustained contraction → "
            "hyperthermia, rigidity, acidosis, rhabdomyolysis, hyperkalemia, arrhythmia, death. "
            "CLINICAL FEATURES (CCD): "
            "Neonatal/infantile hypotonia (floppy infant); delayed motor milestones; "
            "proximal > distal weakness (hip girdle, shoulder girdle); "
            "facial weakness mild; extraocular muscles SPARED (KEY DDx from CNM); "
            "scoliosis common; foot deformities (pes planus, pes cavus); "
            "NO cardiac involvement (CK mildly elevated or normal); "
            "intelligence NORMAL; slowly progressive or stable; "
            "MHS co-segregates with CCD in ~50% of RYR1-CCD families — "
            "ALL first-degree relatives must be counselled as MHS-susceptible. "
            "MHS EMERGENCY PROTOCOL: "
            "Avoid all volatile anaesthetic agents (halothane, isoflurane, sevoflurane, desflurane); "
            "avoid succinylcholine (suxamethonium); "
            "total intravenous anaesthesia (TIVA) with propofol is safe; "
            "dantrolene 2.5 mg/kg IV IMMEDIATELY if MH suspected; "
            "European MH Group and MHAUS protocols; "
            "MH susceptibility is lifelong — inform ALL surgeons/anaesthetists; "
            "MedicAlert bracelet mandatory. "
            "GENETIC COUNSELLING: AD — 50% offspring risk; AR (biallelic) — 25% sibling risk; "
            "de novo mutations occur; genetic testing of at-risk relatives mandatory for anaesthesia safety. "
            "TESTING: RYR1 sequencing + CNV; caffeine-halothane contracture test (CHCT or IVCT) for MHS "
            "when genetic testing inconclusive; "
            "muscle biopsy (Gomori trichrome) for cores; oxidative enzyme stains."
        ),
        "locus": "19q13.2",
        "aa": 5037,
        "kDa": 565,
        "omim_gene": "180901",
        "omim_disease": "117000",
        "inheritance": (
            "AD (CCD, MHS1 — dominant GOF); AR (severe MmD, recessive CCD — biallelic LOF); "
            "some heterozygous carriers are MHS-susceptible only (subclinical myopathy); "
            "de novo AD mutations occur; 50% offspring risk AD; 25% sibling risk AR"
        ),
        "gene_class": (
            "Ryanodine receptor (RyR); homotetrameric SR Ca2+ release channel; "
            "E-C coupling; T-tubule–SR junction (triad); largest known single-chain protein (565 kDa monomer); "
            "N-terminal regulatory scaffold (SPRY domains) + transmembrane pore domain (C-terminal)"
        ),
        "key_alerts": [
            "RYR1-ANAESTHESIA-EMERGENCY-MHS: ALL RYR1 mutation carriers and CCD patients must be assumed "
            "malignant hyperthermia susceptible (MHS) until proven otherwise — "
            "AVOID volatile anaesthetics (halothane, isoflurane, sevoflurane, desflurane, enflurane) "
            "and succinylcholine; use TIVA (propofol-based) exclusively; "
            "dantrolene 2.5 mg/kg IV stat if MH occurs; "
            "MH cart must be immediately available in any OR where RYR1 patient is treated; "
            "INFORM surgeon, anaesthetist, and nurse on every admission — this is a life-or-death alert",
            "RYR1-FAMILY-MHS-COUNSELLING-MANDATORY: CCD and MHS co-segregate in ~50% of RYR1-CCD families; "
            "ALL first-degree relatives (parents, siblings, children) require MHS counselling "
            "and ideally genetic testing before any general anaesthesia; "
            "do NOT assume family member is safe without testing",
            "RYR1-CORES-ON-BIOPSY: central cores on Gomori trichrome + NADH-TR (amorphous cores lacking "
            "mitochondria and oxidative enzyme activity) — pathognomonic for CCD; "
            "multi-minicore disease (MmD) shows multiple small cores; "
            "cores are NOT specific to RYR1 alone — SELENON (RSMD1) also causes minicores",
            "RYR1-CARDIAC-SPARING: CCD/MmD due to RYR1 does NOT involve cardiac muscle — "
            "cardiac RyR2 is a different gene; do NOT confuse with MYH7-related DCM/cardiomyopathy; "
            "annual echo NOT required for RYR1 (unless co-inherited cardiac mutation)",
            "RYR1-EXTRAOCULAR-SPARING: extraocular muscles are SPARED in CCD — "
            "this distinguishes RYR1-CCD from centronuclear myopathy (MTM1, DNM2, BIN1) "
            "where ophthalmoplegia and ptosis are prominent features",
            "RYR1-SCOLIOSIS-ORTHOPAEDIC: scoliosis is common in RYR1 CCD — "
            "annual clinical scoliosis assessment; spine X-ray if clinical curvature; "
            "bracing vs surgical decision requires neuromuscular scoliosis specialist",
        ],
        "etiologies": [
            "CCD (central core disease) AD — GOF mutations in transmembrane domain (C-terminal) — "
            "constitutively open RYR1 → SR Ca2+ depletion; I4898T, G4898R, R163C hotspots; "
            "variable penetrance; most common cause of CCD",
            "MHS1 only — missense GOF mutations; no clinical myopathy; "
            "only triggered by volatile anaesthetics/succinylcholine; "
            "caffeine-halothane contracture test positive; important for anaesthesia risk",
            "Multi-minicore disease (MmD) AR — biallelic LOF; both alleles mutated; "
            "severe neonatal presentation; respiratory compromise; "
            "axial weakness + ophthalmoplegia in some AR cases",
            "King-Denborough syndrome — RYR1-related dysmorphic features + MH susceptibility; "
            "facial dysmorphism, ptosis, low-set ears, short stature, skeletal abnormalities",
            "Exertional rhabdomyolysis RYR1 — AD heterozygous; triggered by exercise/heat/drugs; "
            "recurrent episodes; myoglobinuria; renal failure risk; avoid strenuous unaccustomed exercise",
            "Centronuclear RYR1 — rare; central nuclei on biopsy without typical core pattern; "
            "biallelic AR variants; may be confused with MTM1/DNM2",
        ],
        "stats": {
            "mhs_co_segregation_pct": 50,
            "scoliosis_pct": 55,
            "foot_deformity_pct": 45,
            "cardiac_involvement_pct": 0,
            "extraocular_sparing_pct": 98,
        },
    },
    # ── MYH7 — Laing / Hyaline Body / DCM ────────────────────────────────────
    {
        "gene": "MYH7",
        "protein": (
            "MYH7 — 14q11.2 AD — SlowBetaMyosinHeavyChain-1935aa — "
            "LaingDistalMyopathy-MPD1 / HyalineBodyMyopathy / "
            "DCM-LVNC-CARDIAC-SURVEILLANCE-MANDATORY — "
            "AnnualECG-Echo-AllCarriers"
        ),
        "alias": (
            "MYH7 (myosin heavy chain 7, cardiac muscle beta / slow skeletal); OMIM gene 160760; "
            "MPD1 (Laing distal myopathy) OMIM 160500; "
            "SAME gene causes HCM (hypertrophic cardiomyopathy) OMIM 192600 and DCM (dilated cardiomyopathy) OMIM 613765 "
            "depending on mutation class. "
            "14q11.2; 1935 aa; ~223 kDa; autosomal dominant (all known pathogenic variants). "
            "FUNCTION: MYH7 encodes the beta-myosin heavy chain (β-MHC), "
            "the predominant myosin isoform in slow-twitch (type 1) skeletal muscle fibres "
            "and in adult ventricular cardiomyocytes (cardiac β-MHC). "
            "β-MHC is the molecular motor that drives muscle contraction: "
            "ATP-powered cross-bridge cycling between myosin head and actin filament. "
            "SKELETAL MYOPATHY MECHANISM: "
            "Disease-causing MYH7 variants differ from HCM/DCM variants: "
            "Laing distal myopathy (MPD1) — mutations in rod domain (coiled-coil) or C-terminal; "
            "cause abnormal myosin filament assembly; selective slow fibre atrophy; "
            "distal weakness beginning in childhood; "
            "Hyaline body myopathy — mutations causing protein aggregation; "
            "'hyaline bodies' = amorphous eosinophilic myosin aggregates within fibres on biopsy. "
            "CLINICAL FEATURES (SKELETAL): "
            "MPD1 (Laing distal myopathy): childhood onset (1-25y); "
            "DISTAL weakness onset — great toe + ankle dorsiflexors first (foot drop); "
            "then finger extensors, neck flexors; scapular winging; "
            "slowly progressive; proximal weakness late; "
            "NORMAL cognition; normal life expectancy if no cardiac involvement; "
            "CARDIAC SURVEILLANCE MANDATORY — same gene causes DCM and LVNC in different pedigrees; "
            "even MPD1 families can develop DCM; annual ECG + echo for ALL MYH7 carriers. "
            "Hyaline body myopathy: scapuloperoneal distribution; "
            "proximal shoulder + distal legs; scapular winging; hyaline bodies on biopsy. "
            "CARDIAC OVERLAP: "
            "HCM mutations (MYH7): S403stop, R453C, R719W, R723G, R869H, R870H in globular head — "
            "cause hypercontractility and hypertrophic response; "
            "DCM mutations (MYH7): R894X, K850N in rod domain — cause dilated heart; "
            "LVNC (left ventricular non-compaction): non-compaction mutations in MYH7; "
            "For skeletal myopathy pedigrees, cardiac surveillance essential "
            "because some carriers may manifest only cardiac disease. "
            "KEY DIAGNOSTIC TIP: MYH7 on both HCM/DCM panels AND neuromuscular myopathy panels — "
            "if a distal myopathy patient has a MYH7 variant, check cardiac phenotype carefully."
        ),
        "locus": "14q11.2",
        "aa": 1935,
        "kDa": 223,
        "omim_gene": "160760",
        "omim_disease": "160500",
        "inheritance": (
            "AD — all known pathogenic skeletal myopathy variants; "
            "variable expressivity; 50% offspring risk; "
            "intrafamilial variability: some family members may manifest cardiac only, others skeletal only"
        ),
        "gene_class": (
            "Class II non-muscle / sarcomeric myosin heavy chain (MyHC); "
            "ATPase motor domain (globular head) + neck + coiled-coil rod; "
            "slow-twitch skeletal (type 1 fibres) + cardiac ventricular isoform; "
            "thick filament constituent; cross-bridge cycling ATP hydrolysis"
        ),
        "key_alerts": [
            "MYH7-CARDIAC-SURVEILLANCE-MANDATORY: MYH7 is expressed in BOTH skeletal muscle AND cardiac "
            "ventricle — ALL MYH7 variant carriers regardless of skeletal phenotype require "
            "annual ECG + echocardiogram; even 'pure' Laing distal myopathy families can develop "
            "DCM or LVNC; cardiac death can be the first manifestation in a family; "
            "refer ALL to cardiologist at diagnosis",
            "MYH7-DISTAL-ONSET-FOOT-DROP: Laing distal myopathy begins with great toe extensor weakness "
            "(inability to extend big toe) and ankle dorsiflexion weakness (foot drop) — "
            "this is the signature presentation; in any child with foot drop, "
            "MYH7 should be on the panel alongside CMT genes",
            "MYH7-HCM-DCM-SAME-GENE: MYH7 is the most common single gene causing familial HCM (35-40%) "
            "AND a significant cause of familial DCM — the myopathy mutations differ from cardiac mutations "
            "but the gene overlaps with both cardiomyopathy panels; "
            "always cross-check variant location (globular head = HCM risk; rod domain = DCM risk)",
            "MYH7-SCAPULAR-WINGING-LATE: scapular winging can appear later in MYH7 myopathy — "
            "asymmetric in some; document and assess regularly; "
            "scapulothoracic fusion can improve function in selected patients",
            "MYH7-NECK-FLEXOR-WEAKNESS: neck flexors are disproportionately affected in MYH7 Laing — "
            "test neck flexion strength in every assessment; "
            "this is a helpful clinical differentiator from other distal myopathies",
            "MYH7-BIOPSY-TYPE1-FIBRE-PREDOMINANCE: type 1 slow fibre predominance and atrophy on biopsy "
            "is characteristic; hyaline bodies are eosinophilic cytoplasmic inclusions on H&E; "
            "electron microscopy confirms myosin thick filament disorganisation",
        ],
        "etiologies": [
            "Laing distal myopathy (MPD1) — rod/C-terminal MYH7 mutations; "
            "distal lower limb onset; foot drop + great toe extensor weakness; "
            "K1617del most common; childhood to early adult onset",
            "Hyaline body myopathy — protein aggregation variants; "
            "eosinophilic hyaline bodies on biopsy; scapuloperoneal distribution",
            "Scapuloperoneal myopathy — predominantly scapular girdle + peroneal weakness; "
            "MYH7 variant in rod domain; overlaps HBM phenotype",
            "DCM-related skeletal myopathy — rod domain variants; "
            "dilated cardiomyopathy + skeletal myopathy co-occurring; "
            "some family members manifest cardiac only",
            "HCM pedigree with subclinical skeletal myopathy — globular head variants; "
            "skeletal myopathy mild or subclinical; HCM is dominant feature",
            "LVNC + distal myopathy — non-compaction mutations; "
            "rare but important: stroke risk from LV thrombus; anticoagulation required",
        ],
        "stats": {
            "cardiac_involvement_pct": 35,
            "foot_drop_onset_pct": 80,
            "neck_flexor_weakness_pct": 60,
            "scapular_winging_pct": 50,
            "normal_cognition_pct": 100,
        },
    },
    # ── ACTA1 — Nemaline Myopathy 3 ──────────────────────────────────────────
    {
        "gene": "ACTA1",
        "protein": (
            "ACTA1 — 1q42.13 AD(70%deNovo)/AR — AlphaSkeletalActin-377aa — "
            "NemalineMyopathy3-NEM3 — RodBodiesOnBiopsy-PATHOGNOMONIC — "
            "RESPIRATORY-SURVEILLANCE-MANDATORY — NeonatalCriticalForm"
        ),
        "alias": (
            "ACTA1 (actin, alpha 1, skeletal muscle); OMIM gene 102610; "
            "NEM3 (nemaline myopathy 3) OMIM 161800. "
            "1q42.13; 377 aa; ~42 kDa; autosomal dominant (~70% de novo AD) or AR. "
            "FUNCTION: ACTA1 encodes alpha-skeletal muscle actin, "
            "the predominant actin isoform in adult skeletal muscle thin filaments. "
            "Actin is the scaffold of the sarcomere thin filament: "
            "it binds tropomyosin-troponin complex (regulating Ca2+ sensitivity), "
            "interacts with myosin heads during cross-bridge cycling, "
            "and forms the structural backbone of the sarcomere. "
            "NEM3 MECHANISM: "
            "Dominant or biallelic ACTA1 mutations disrupt: "
            "(1) thin filament assembly → rod body formation (nemaline rods = polymerised actin aggregates); "
            "(2) actin-myosin interaction → contractile failure; "
            "(3) actin-tropomyosin binding → altered Ca2+ sensitivity. "
            "Nemaline rods are pathognomonic on Gomori trichrome biopsy — "
            "electron dense Z-disc-derived aggregates visible by EM. "
            "CLINICAL SPECTRUM: "
            "Neonatal lethal (severe AD de novo): profound neonatal hypotonia, "
            "respiratory failure from birth, unable to breathe or swallow, "
            "death within days-weeks without aggressive ventilatory support; "
            "no motor activity; face + limbs completely hypotonic; "
            "Severe congenital: neonatal onset, ventilator dependent, "
            "survive with tracheostomy + NIV; wheelchair; "
            "Typical congenital: infantile onset, delayed milestones, "
            "proximal > distal weakness, facial weakness, high arched palate, "
            "scoliosis, respiratory involvement; "
            "walks with aids; may survive to adulthood with support; "
            "Childhood onset mild: onset 2-10y, mild proximal weakness, "
            "near-normal life; slowly progressive; "
            "Adult onset: late-onset nemaline, progressive proximal weakness, "
            "some sporadic (possible somatic mosaicism). "
            "RESPIRATORY: respiratory involvement is the most important determinant of survival — "
            "FVC must be monitored at every visit; "
            "early NIV + chest physio prolongs survival; "
            "tracheostomy for severe cases; "
            "aggressive management of respiratory infections. "
            "FACIAL FEATURES: open mouth posture, elongated face, high arched palate, "
            "nasal speech; feeding difficulties in infancy. "
            "GENETIC COUNSELLING: ~70% of AD cases are de novo mutations; "
            "parents tested negative does NOT exclude recurrence (germline mosaicism risk); "
            "AR biallelic: 25% sibling risk; "
            "counselling for germline mosaicism essential in all de novo cases."
        ),
        "locus": "1q42.13",
        "aa": 377,
        "kDa": 42,
        "omim_gene": "102610",
        "omim_disease": "161800",
        "inheritance": (
            "AD (~70% de novo; 30% familial) or AR (biallelic); "
            "de novo AD accounts for most severe cases; "
            "germline mosaicism possible — recurrence risk ~4-5% for de novo cases; "
            "AR — 25% sibling risk; 100% carrier parents"
        ),
        "gene_class": (
            "Sarcomeric alpha-actin (skeletal muscle isoform); "
            "G-actin monomer → F-actin thin filament polymerisation; "
            "ATPase (actin-activated myosin ATPase); "
            "tropomyosin-troponin binding scaffold; "
            "sarcomere thin filament; Z-disc anchoring"
        ),
        "key_alerts": [
            "ACTA1-RESPIRATORY-SURVEILLANCE-MANDATORY: respiratory compromise is the major cause of "
            "morbidity and mortality in ACTA1 nemaline myopathy — "
            "FVC (sitting + lying) at EVERY clinic visit; "
            "lying FVC <50% sitting FVC suggests diaphragm weakness; "
            "early proactive NIV (BiPAP) before symptomatic failure; "
            "aggressive infection management; "
            "tracheostomy discussion early in severe cases",
            "ACTA1-NEONATAL-CRITICAL-FORM: severe AD de novo ACTA1 mutations cause neonatal lethal myopathy — "
            "complete absence of spontaneous movement, respiratory failure from birth; "
            "immediate NICU referral + mechanical ventilation decision; "
            "early genetic diagnosis essential for family counselling",
            "ACTA1-NEMALINE-RODS-PATHOGNOMONIC: nemaline rods on Gomori trichrome biopsy "
            "(magenta-staining rod bodies within myofibres) are pathognomonic for nemaline myopathy; "
            "confirm with electron microscopy (Z-disc derived dense bodies); "
            "found in ACTA1 (NEM3), NEB (NEM2), TPM2, TPM3, TNNT1, LMOD3",
            "ACTA1-FACIAL-FEATURES-FEEDING: facial weakness, open mouth, elongated face, "
            "high arched palate, nasal speech, and feeding difficulties in infancy are characteristic; "
            "nasogastric or gastrostomy tube may be required in severe/moderate forms",
            "ACTA1-DE-NOVO-GERMLINE-MOSAICISM: ~70% of ACTA1 cases are de novo AD — "
            "parents may test negative but recurrence risk is NOT zero due to germline mosaicism; "
            "counsel families that 4-5% recurrence risk exists even with negative parental testing; "
            "prenatal or preimplantation genetic testing recommended for subsequent pregnancies",
            "ACTA1-SCOLIOSIS-ORTHOPAEDIC: scoliosis is common and can compromise respiratory function "
            "further — annual clinical assessment; early referral for bracing or surgical correction "
            "before curves become severe; respiratory function must guide surgical timing",
        ],
        "etiologies": [
            "Neonatal lethal nemaline (AD de novo ACTA1) — null or severe GOF variants; "
            "no movement at birth; mechanical ventilation from delivery; "
            "rods on biopsy (post-mortem or open biopsy); "
            "counselling: near-universal neonatal/early infantile death without tracheostomy",
            "Severe congenital (AD de novo ACTA1) — pronounced neonatal hypotonia; "
            "respiratory compromise requiring NIV/CPAP; "
            "survive with intensive support; "
            "gastrostomy feeding; wheelchair bound",
            "Typical congenital (AR or AD) — infantile onset; delayed milestones; "
            "proximal weakness; facial weakness; scoliosis; "
            "ambulates with aids; NIV in second decade; typical lifespan shortened",
            "Childhood onset mild (AD inherited or AR) — onset 2-10y; "
            "mild proximal weakness; foot drop possible; near-normal lifespan; "
            "FVC monitoring; physiotherapy",
            "Adult onset nemaline — late onset (20-50y); progressive proximal weakness; "
            "some sporadic with somatic mosaicism; dysphagia possible; "
            "rule out inflammatory myopathy (sporadic adult nemaline can mimic polymyositis)",
            "NEM3 intranuclear rod body variant — rods within nuclei in addition to cytoplasm; "
            "specific ACTA1 mutation cluster; cardiac involvement rare but reported",
        ],
        "stats": {
            "de_novo_rate_pct": 70,
            "respiratory_compromise_pct": 75,
            "facial_weakness_pct": 80,
            "scoliosis_pct": 60,
            "neonatal_critical_pct": 30,
        },
    },
    # ── NEB — Nemaline Myopathy 2 (largest human gene) ───────────────────────
    {
        "gene": "NEB",
        "protein": (
            "NEB — 2q23.3 AR — Nebulin-6669aa-LARGEST-HUMAN-GENE — "
            "NemalineMyopathy2-NEM2 — RodBodies-AnkleContractures-FootDrop — "
            "AR-SiblingRisk25pct"
        ),
        "alias": (
            "NEB (nebulin); OMIM gene 161650; "
            "NEM2 (nemaline myopathy 2) OMIM 256030. "
            "2q23.3; 6669 aa; ~800 kDa; autosomal recessive (biallelic variants). "
            "FUNCTION: NEB encodes nebulin — the LARGEST known human protein (6669 amino acids, ~800 kDa). "
            "Nebulin is a giant ruler protein that spans the entire thin filament from Z-disc to pointed end, "
            "determining thin filament length and maintaining sarcomere geometry. "
            "Functions: (1) thin filament stabilisation and length determination; "
            "(2) actin filament polymerisation regulation; "
            "(3) myosin-actin interaction — nebulin acts as a scaffold coordinating cross-bridge cycling. "
            "NEM2 MECHANISM: "
            "Biallelic LOF variants in NEB → truncated or absent nebulin → "
            "thin filament shortening and disorganisation → nemaline rod formation (Z-disc derived) → "
            "impaired force generation. "
            "The NEB gene is exceptionally large (249 exons) with a "
            "triplicate-repeat region (exons 82-105 encoding the actin-binding modules) — "
            "standard sequencing can MISS DELETIONS in the triplicate region; "
            "MLPA or CNV analysis is MANDATORY to exclude triplicate deletions. "
            "CLINICAL FEATURES: "
            "Typical congenital (most common): mild to moderate neonatal hypotonia; "
            "delayed motor milestones; proximal > distal weakness; "
            "facial weakness mild; ANKLE CONTRACTURES prominent (Achilles tendon shortening); "
            "foot drop + toe-walking in early childhood; "
            "high arched palate; mild respiratory involvement (usually not severe); "
            "intelligence NORMAL; most ambulate into adulthood; "
            "slowly progressive or stable; "
            "Severe congenital (rarer): neonatal respiratory compromise; "
            "ventilator dependence; earlier than typical. "
            "BIOPSY: nemaline rods on Gomori trichrome (same appearance as ACTA1-NEM3); "
            "type 1 fibre predominance and uniformity. "
            "JOINT CONTRACTURES: Achilles tendon contractures cause toe-walking and gait difficulty — "
            "serial casting, orthotics (AFOs), or surgical release; "
            "ankle contractures are more prominent in NEB vs ACTA1 nemaline. "
            "TESTING: next-generation sequencing panel mandatory + "
            "MLPA for NEB triplicate repeat deletions (exons 82-105) — "
            "failure to test MLPA misses ~10-15% of NEB cases."
        ),
        "locus": "2q23.3",
        "aa": 6669,
        "kDa": 800,
        "omim_gene": "161650",
        "omim_disease": "256030",
        "inheritance": (
            "AR — biallelic LOF variants; "
            "25% sibling risk; both parents are obligate carriers; "
            "consanguinity increases risk; rare compound heterozygotes common"
        ),
        "gene_class": (
            "Giant sarcomeric ruler protein (6669 aa, ~800 kDa); "
            "thin filament length determinant; "
            "actin-binding super-repeat modules (exons 82-105 triplicate); "
            "Z-disc to barbed end spanning; myosin-S1 interaction; "
            "F-actin polymerisation scaffold"
        ),
        "key_alerts": [
            "NEB-MLPA-TRIPLICATE-MANDATORY: NEB has a triplicate-repeat region (exons 82-105) "
            "that causes alignment/mapping artefacts in NGS — "
            "standard sequencing MISSES deletions in this region in ~10-15% of NEB cases; "
            "MLPA (multiplex ligation-dependent probe amplification) targeting NEB triplicate exons "
            "is MANDATORY in all patients with nemaline myopathy biopsy but negative standard NEB sequencing; "
            "never exclude NEB without MLPA",
            "NEB-ANKLE-CONTRACTURES-DISTINCTIVE: ankle contractures (Achilles tendon shortening) are "
            "characteristic and early in NEB nemaline myopathy — "
            "toe-walking and equinus deformity in young children; "
            "serial casting from early infancy can delay contracture progression; "
            "AFOs mandatory to maintain heel-toe gait; "
            "surgical lengthening considered if casting fails",
            "NEB-LARGEST-HUMAN-GENE: nebulin (6669 aa, 249 exons) is the largest known human gene — "
            "comprehensive panel sequencing of NEB requires specific bioinformatics handling; "
            "WGS may be preferable to WES for detecting complex intronic variants; "
            "CNV analysis essential for multi-exon deletions beyond the triplicate region",
            "NEB-RESPIRATORY-MILDER-THAN-ACTA1: NEB nemaline typically has milder respiratory involvement "
            "than ACTA1 severe forms — but FVC monitoring is still mandatory; "
            "NIV may be required in second-third decade especially in severe congenital subtype; "
            "annual pulmonary function testing",
            "NEB-NEMALINE-RODS-SAME-BIOPSY-PATTERN: nemaline rods on Gomori trichrome are identical "
            "regardless of gene (NEB, ACTA1, TPM2, TPM3, TNNT1, LMOD3) — "
            "genetic testing is required to determine the specific gene; "
            "'nemaline myopathy' is a pathological diagnosis, not a genetic diagnosis",
            "NEB-SIBLING-TESTING: both parents are obligate NEB carriers; "
            "25% recurrence risk per pregnancy; prenatal/preimplantation genetic testing available; "
            "cascade carrier testing of siblings who are planning families",
        ],
        "etiologies": [
            "Typical congenital NEM2 (biallelic NEB truncating/LOF) — mild neonatal hypotonia; "
            "delayed milestones; ankle contractures; foot drop; ambulates; "
            "most common NEB presentation (~80%)",
            "Severe congenital NEM2 (biallelic NEB null) — neonatal respiratory compromise; "
            "ventilator dependence; earlier milestones delay; "
            "rare; both alleles null or severe truncating",
            "NEB triplicate deletion compound heterozygote — one truncating allele + "
            "triplicate region deletion (detected by MLPA only); "
            "phenotype variable; missed on standard NGS",
            "Distal nemaline with ankle contractures (NEB) — distal-predominant weakness; "
            "foot drop and finger extensor weakness; ankle contractures; "
            "rare distal phenotype subset",
            "Core-rod myopathy NEB — combined nemaline rods + minicore changes on biopsy; "
            "overlap with RYR1-related core-rod; biallelic NEB hypomorphic variants",
            "Childhood onset NEM2 — later presentation 2-5y; gait difficulty; "
            "ankle contractures; mild proximal weakness; near-normal lifespan",
        ],
        "stats": {
            "ankle_contracture_pct": 85,
            "respiratory_compromise_pct": 35,
            "facial_weakness_pct": 55,
            "ambulation_maintained_pct": 80,
            "triplicate_deletion_missed_by_ngs_pct": 12,
        },
    },
    # ── SELENON — Rigid Spine Muscular Dystrophy / RSMD1 ─────────────────────
    {
        "gene": "SELENON",
        "protein": (
            "SELENON — 1p36.11 AR — SelenoproteinN-590aa — "
            "RigidSpineMuscleDystrophy-RSMD1 / MultiminioreCoreDisease — "
            "NO-Cardiac-KEY-DDx-EMD-LMNA — "
            "EarlyRigidSpine-Scoliosis-Surveillance-Mandatory"
        ),
        "alias": (
            "SELENON (selenoprotein N; formerly SEPN1); OMIM gene 606210; "
            "RSMD1 (rigid spine muscular dystrophy 1) OMIM 602771; "
            "also causes Multi-minicore Disease (MmD) OMIM 255320 and Mallory body myopathy. "
            "1p36.11; 590 aa; ~67 kDa; autosomal recessive (biallelic LOF). "
            "FUNCTION: SELENON is an ER-resident selenoprotein (contains selenocysteine, "
            "encoded by UGA stop codon read-through in selenoprotein context). "
            "Proposed functions: "
            "(1) ER calcium homeostasis — SELENON modulates SERCA (sarco-ER Ca2+ ATPase) activity, "
            "regulating Ca2+ re-uptake into the ER lumen; "
            "(2) oxidative stress protection — selenoproteins are redox enzymes; "
            "(3) muscle development — required for normal muscle fibre maintenance. "
            "RSMD1 MECHANISM: "
            "Biallelic loss-of-function → dysregulated ER calcium → "
            "muscle fibre dysfunction with specific morphology: "
            "minicores (on biopsy) + RIGID SPINE is the cardinal clinical feature. "
            "Rigid spine due to selective paraspinal muscle involvement (fibrous infiltration of "
            "paraspinal erector spinae early), causing lumbar hyperlordosis → progressive loss of "
            "thoracolumbar spine mobility. "
            "CLINICAL FEATURES: "
            "Early childhood onset (2-5y typical); "
            "RIGID SPINE — paraspinal muscle involvement: "
            "inability to flex lumbar spine (cannot touch toes); "
            "lumbar hyperlordosis (hyperextended posture); "
            "progressive rigidity of cervicothoracic spine; "
            "scoliosis — early onset and progressive; "
            "proximal limb weakness (mild-moderate); "
            "RESPIRATORY COMPROMISE — early + progressive respiratory muscle weakness; "
            "diaphragm involved; FVC must be monitored; "
            "nocturnal hypoventilation before daytime symptoms; "
            "NIV early (often before scoliosis surgery); "
            "KEY DDx — NO CARDIAC INVOLVEMENT: "
            "This distinguishes SELENON RSMD1 from EMD (Emerin-EDMD) and LMNA (EDMD2, DCM) — "
            "both of which also cause rigid spine with LETHAL cardiac arrhythmias; "
            "SELENON heart is SPARED — no ICD needed (unlike EMD/LMNA); "
            "echocardiogram and ECG should still be performed to confirm cardiac sparing; "
            "however, ICD is NOT routinely indicated for SELENON."
        ),
        "locus": "1p36.11",
        "aa": 590,
        "kDa": 67,
        "omim_gene": "606210",
        "omim_disease": "602771",
        "inheritance": (
            "AR — biallelic LOF; truncating variants most common; "
            "25% sibling risk; consanguinity increases prevalence; "
            "no dominant disease reported"
        ),
        "gene_class": (
            "ER-resident selenoprotein (selenocysteine at catalytic site); "
            "SERCA regulator; ER calcium homeostasis; "
            "oxidative stress protection; "
            "muscle development and maintenance; "
            "redox enzyme — selenium-dependent"
        ),
        "key_alerts": [
            "SELENON-NO-CARDIAC-KEY-DDx: SELENON RSMD1 does NOT involve the heart — "
            "this is the critical distinguisher from EMD (EDMD1) and LMNA (EDMD2/DCM) "
            "which also cause rigid spine but with LETHAL cardiac arrhythmias requiring ICD; "
            "ECG + echo to confirm cardiac sparing; "
            "do NOT routinely implant ICD for SELENON (contrast with EMD/LMNA mandatory ICD)",
            "SELENON-RIGID-SPINE-EARLY: inability to flex lumbar spine (cannot bend forward to touch toes) "
            "is the hallmark of RSMD1 — paraspinal erector spinae are selectively affected early; "
            "lumbar hyperlordosis is characteristic posture; "
            "assess spinal mobility at every visit (Schober test); "
            "rigid spine + early scoliosis in a child = SELENON on the differential",
            "SELENON-RESPIRATORY-BEFORE-SCOLIOSIS-SURGERY: respiratory compromise can precede "
            "or co-occur with scoliosis; "
            "NIV may be needed BEFORE scoliosis surgery is considered; "
            "anaesthesia for scoliosis correction requires pre-operative respiratory optimisation; "
            "multidisciplinary neuromuscular-respiratory assessment mandatory pre-operatively",
            "SELENON-SCOLIOSIS-SURVEILLANCE: scoliosis in SELENON is early onset and rapidly progressive — "
            "6-monthly spinal assessment in growing children; "
            "Cobb angle X-ray; early referral to neuromuscular scoliosis surgeon; "
            "bracing may slow progression; surgical correction before severe pulmonary compromise",
            "SELENON-MINICORE-BIOPSY: multi-minicore disease (MmD) pattern on biopsy "
            "(multiple small oxidase-deficient cores per fibre on NADH-TR); "
            "same biopsy pattern as RYR1-related MmD; "
            "genetic testing required to distinguish (SELENON vs RYR1); "
            "SELENON MmD: rigid spine + early respiratory + NO cardiac; "
            "RYR1 MmD: MHS risk, NO rigid spine, possible ophthalmoplegia in AR form",
            "SELENON-SELENOPROTEIN-TESTING: standard molecular testing (NGS) must include "
            "analysis of the UGA selenocysteine codon — some informatics pipelines flag this as "
            "a stop codon; ensure the lab is aware that SELENON has a programmed UGA read-through "
            "for selenocysteine incorporation (selenium-dependent); "
            "incorrect annotation of Sec codon as pathogenic stop must be avoided",
        ],
        "etiologies": [
            "RSMD1 typical (biallelic SELENON truncating) — early childhood rigid spine; "
            "paraspinal weakness + scoliosis; mild proximal limb weakness; "
            "respiratory decline 5-15y; most common SELENON phenotype",
            "Multi-minicore Disease SELENON variant — minicore biopsy + rigid spine; "
            "ophthalmoplegia absent (unlike RYR1 AR MmD subtype); "
            "respiratory involvement prominent",
            "SELENON early severe (compound heterozygous null+null) — earlier onset; "
            "more rapid respiratory progression; NIV by age 10y; "
            "severe scoliosis; wheelchair-dependent",
            "Mallory body myopathy SELENON — desmin + sarcoplasmic protein aggregates "
            "(Mallory-Denk bodies) on biopsy; "
            "overlapping SELENON phenotype; rare",
            "Congenital onset SELENON — neonatal or early infantile rigid spine; "
            "earlier scoliosis; severe respiratory compromise",
            "Adult presentation SELENON (mild alleles) — diagnosis delayed to adulthood "
            "due to mild phenotype; mild rigid spine; slow progression; FVC preserved longer",
        ],
        "stats": {
            "cardiac_involvement_pct": 0,
            "rigid_spine_pct": 98,
            "scoliosis_pct": 85,
            "respiratory_compromise_pct": 80,
            "nih_dcm_pct": 0,
        },
    },
    # ── MTM1 — X-linked Myotubular Myopathy ──────────────────────────────────
    {
        "gene": "MTM1",
        "protein": (
            "MTM1 — Xq28 XLR — Myotubularin-603aa — "
            "XlinkedMyotubularMyopathy-XLMTM — MOST-SEVERE-CongenitalMyopathy — "
            "HighNeonatalMortality-VentilatorDependent — "
            "AspireRx-GeneTherapy-Trial"
        ),
        "alias": (
            "MTM1 (myotubularin 1); OMIM gene 300415; "
            "XLMTM (X-linked myotubular myopathy) / CNMX OMIM 310400. "
            "Xq28; 603 aa; ~69 kDa; X-linked recessive (males severely affected; females carriers, occasionally symptomatic). "
            "FUNCTION: MTM1 encodes myotubularin — a phosphoinositide 3-phosphatase "
            "that dephosphorylates PI(3)P and PI(3,5)P2 on endosomal membranes. "
            "Critical roles: "
            "(1) endosomal trafficking and membrane recycling; "
            "(2) T-tubule (transverse tubule) biogenesis — PI(3)P metabolism is essential for "
            "correct T-tubule formation and the triadic SR-T-tubule junction; "
            "(3) autophagy regulation; "
            "(4) triad maturation in developing muscle (myotubes → mature fibres). "
            "XLMTM MECHANISM: "
            "MTM1 loss-of-function → abnormal PI(3)P accumulation → "
            "defective T-tubule formation → triad disorganisation → "
            "failure of E-C coupling → severe neonatal hypotonia. "
            "Histology: centronuclear myopathy pattern — central nuclei in myofibres "
            "(resembling foetal myotubes, hence 'myotubular' myopathy); "
            "type 1 fibre smallness and predominance; "
            "oxidative enzyme staining: 'necklace fibres' (ring of oxidative activity). "
            "CLINICAL FEATURES: "
            "MOST SEVERE congenital myopathy — neonatal onset universal; "
            "profound hypotonia at birth ('floppy infant'); "
            "severely impaired respiratory effort from birth; "
            "almost all males require ventilatory support (>80% ventilator dependent); "
            "high neonatal mortality without aggressive ventilatory support; "
            "facial diplegia (facial weakness bilateral); "
            "ophthalmoplegia + ptosis (extraocular + levator weakness); "
            "NO or minimal spontaneous limb movements; "
            "survival into childhood/adulthood POSSIBLE with tracheostomy + continuous ventilation; "
            "some ambulatory function with long-term support; "
            "biliary disease (cholestasis, peliosis hepatis) — visceral involvement; "
            "FEMALES: carrier females are generally asymptomatic but "
            "~10-15% manifest mild-moderate myopathy due to skewed X-inactivation. "
            "GENE THERAPY: "
            "Aspiro gene therapy (ASPIRO trial, Solid Biosciences + Astellas): "
            "AT132 (AAV8-MTM1 gene therapy) — Phase 1/2 ASPIRO trial; "
            "severe adverse events (including deaths) led to trial pause; "
            "reanalysis and ongoing development; "
            "mRNA therapy approaches also in development. "
            "PROGNOSIS: severe; most males die in infancy without ventilatory support; "
            "with intensive support, some survive 10-20+ years; "
            "cognitive function is PRESERVED — brain is not affected."
        ),
        "locus": "Xq28",
        "aa": 603,
        "kDa": 69,
        "omim_gene": "300415",
        "omim_disease": "310400",
        "inheritance": (
            "X-linked recessive; males hemizygous (severely affected); "
            "females heterozygous carriers (usually asymptomatic; "
            "~10-15% manifest myopathy with skewed X-inactivation); "
            "de novo mutations occur; "
            "mothers of affected sons are usually carriers — test maternal family"
        ),
        "gene_class": (
            "Myotubularin phosphoinositide 3-phosphatase (PI3P phosphatase); "
            "PTP/DSP dual-specificity phosphatase family; "
            "PI(3)P → PI dephosphorylation; "
            "endosomal membrane trafficking; "
            "T-tubule biogenesis; triad formation; autophagy"
        ),
        "key_alerts": [
            "MTM1-MOST-SEVERE-CONGENITAL-MYOPATHY: XLMTM is the most severe congenital myopathy — "
            "all affected males require immediate neonatal ICU management; "
            "anticipate ventilatory support from birth; "
            "tracheostomy decision should be discussed antenatally when XLMTM is known prenatally; "
            "without ventilatory support, most males die within days-weeks of birth",
            "MTM1-COGNITIVE-PRESERVATION: despite profound motor weakness, "
            "cognition is PRESERVED in XLMTM — brain is structurally and functionally normal; "
            "affected males (and surviving females) have normal intelligence; "
            "AAC (augmentative/alternative communication) and assistive technology enable "
            "meaningful quality of life; do not conflate motor severity with cognitive impairment",
            "MTM1-OPHTHALMOPLEGIA-PTOSIS: external ophthalmoplegia (EOM weakness) and ptosis are "
            "characteristic of XLMTM and centronuclear myopathies — "
            "distinguish from CCD/NEM (where EOMs are spared); "
            "ptosis may require surgical correction to maintain visual axis; "
            "strabismus surgery if required for binocular vision",
            "MTM1-BILIARY-HEPATIC-COMPLICATION: visceral involvement including cholestasis, "
            "peliosis hepatis, and biliary problems can occur in XLMTM — "
            "liver function tests at diagnosis and periodically; "
            "hepatic involvement may complicate gene therapy approaches (liver-directed vectors)",
            "MTM1-FEMALE-CARRIERS-SCREENING: ~10-15% of MTM1 carrier females develop symptomatic "
            "myopathy due to skewed X-inactivation — "
            "all carrier females should have a neuromuscular assessment; "
            "muscle biopsy can show centronuclear changes; "
            "screening X-inactivation assay may help predict symptomatic risk",
            "MTM1-GENE-THERAPY-ASPIRO-CAUTION: AT132 ASPIRO trial demonstrated preclinical and "
            "early clinical benefit but was paused after serious adverse events; "
            "refer to specialist MTM1 centres for latest clinical trial updates; "
            "do NOT present gene therapy as approved; discuss as investigational with families",
        ],
        "etiologies": [
            "XLMTM null (frameshift/nonsense MTM1) — most severe; "
            "no residual protein; neonatal respiratory failure; "
            "ventilator dependent from birth; tracheostomy; "
            "high early mortality without support",
            "XLMTM missense hypomorphic — partial protein function; "
            "slightly milder; some may breathe independently briefly; "
            "still severe; variable progression",
            "XLMTM de novo — maternal test negative; "
            "de novo hemizygous mutation; "
            "maternal testing negative does not exclude recurrence "
            "(germline mosaicism possible in mother)",
            "XLMTM carrier female symptomatic — skewed X-inactivation in affected female; "
            "mild-moderate proximal weakness; ptosis; "
            "biopsy may show centronuclear changes; "
            "respiratory monitoring required",
            "XLMTM with severe biliary complication — peliosis hepatis + "
            "liver dysfunction; more common with certain null variants; "
            "liver transplant considered in extreme cases",
            "MTM1 exon deletion/duplication — large genomic rearrangement; "
            "MLPA or CGH-array required; standard sequencing misses; "
            "phenotype variable depending on affected domain",
        ],
        "stats": {
            "ventilator_dependent_pct": 85,
            "ophthalmoplegia_pct": 75,
            "neonatal_death_without_support_pct": 70,
            "cognition_normal_pct": 100,
            "female_carrier_symptomatic_pct": 12,
        },
    },
    # ── DNM2 — Centronuclear Myopathy 3 / CMT-DIB ────────────────────────────
    {
        "gene": "DNM2",
        "protein": (
            "DNM2 — 19p13.2 AD — Dynamin2-870aa — "
            "CentronuclearMyopathy-CNM3 / CMTdominantIntermediateTypeB-CMTDIB — "
            "CentralNuclei-GTPase — "
            "NeuropathyOverlap-PeripheralNerve"
        ),
        "alias": (
            "DNM2 (dynamin 2); OMIM gene 602378; "
            "CNM3 (centronuclear myopathy 3) OMIM 160150; "
            "CMTDIB (Charcot-Marie-Tooth disease, dominant intermediate type B) OMIM 606482. "
            "19p13.2; 870 aa; ~98 kDa; autosomal dominant. "
            "FUNCTION: DNM2 encodes dynamin 2 — a ubiquitous GTPase mediating membrane tubulation "
            "and vesicle scission at multiple subcellular sites: "
            "(1) endocytosis — clathrin-coated vesicle scission; "
            "(2) T-tubule biogenesis and maintenance — DNM2 forms helical oligomers around "
            "T-tubule necks, regulating T-tubule topology; "
            "(3) Golgi trafficking; "
            "(4) centrosome regulation; "
            "(5) actin remodelling. "
            "CNM3 MECHANISM: "
            "AD DNM2 mutations disrupt GTPase function or membrane-binding domains → "
            "abnormal T-tubule structure → triad disorganisation → impaired E-C coupling → "
            "centronuclear histopathology (central nuclei + hub-and-spoke oxidative pattern "
            "on NADH-TR staining). "
            "CMTDIB MECHANISM: same gene, overlapping but distinct mutations → "
            "peripheral nerve Schwann cell dysfunction → intermediate NCS (between demyelinating "
            "and axonal) → CMT phenotype. "
            "CLINICAL FEATURES (CNM3): "
            "Congenital to childhood onset (variable); "
            "proximal muscle weakness; "
            "facial weakness (milder than MTM1); "
            "PTOSIS + OPHTHALMOPLEGIA (variable, milder than MTM1); "
            "slowly progressive; ambulation maintained in most; "
            "respiratory involvement (moderate; NIV may be needed); "
            "intelligence NORMAL; "
            "peripheral neuropathy in some (CMT overlap); "
            "DNM2 mutations are genotype-specific: "
            "R465W — most common CNM mutation; severe neonatal; "
            "R522H — milder adult onset. "
            "CLINICAL FEATURES (CMTDIB): "
            "Intermediate NCS (MCV 25-45 m/s); "
            "distal motor > sensory weakness; foot drop; CMT phenotype "
            "WITHOUT centronuclear myopathy (or with subclinical myopathy). "
            "KEY POINT: the same DNM2 gene causes both CNM3 (myopathy) and CMTDIB (neuropathy) — "
            "some patients have BOTH (neuromyopathy); "
            "full assessment requires both NCS + muscle biopsy."
        ),
        "locus": "19p13.2",
        "aa": 870,
        "kDa": 98,
        "omim_gene": "602378",
        "omim_disease": "160150",
        "inheritance": (
            "AD — haploinsufficiency or dominant negative; "
            "de novo mutations frequent; "
            "50% offspring risk; "
            "variable expressivity — some carriers have subclinical myopathy"
        ),
        "gene_class": (
            "Large GTPase (dynamin family); "
            "membrane tubulation and vesicle fission enzyme; "
            "T-tubule biogenesis; "
            "clathrin-mediated endocytosis; "
            "GTPase domain + middle domain + PH domain + GED; "
            "self-assembles into helical polymers around membrane tubes"
        ),
        "key_alerts": [
            "DNM2-CNM-CMTDIB-SAME-GENE: DNM2 causes TWO distinct disease phenotypes — "
            "CNM3 (centronuclear myopathy) and CMTDIB (dominant intermediate CMT neuropathy); "
            "some patients have both (neuromyopathy); "
            "genotype-phenotype correlations exist but imperfect; "
            "all DNM2 patients require BOTH NCS (nerve conduction) + muscle biopsy assessment",
            "DNM2-CENTRAL-NUCLEI-HUB-SPOKE: biopsy shows central nuclei (centronuclear pattern) "
            "with characteristic hub-and-spoke (radial) NADH-TR oxidative pattern — "
            "distinguish from MTM1 (necklace fibres) and BIN1 (similar centronuclear but AR); "
            "electron microscopy shows T-tubule dilation and triad disorganisation",
            "DNM2-OPHTHALMOPLEGIA-PTOSIS-VARIABLE: external ophthalmoplegia and ptosis are "
            "present but milder than MTM1 (XLMTM); "
            "facial weakness also milder; "
            "distinguish from congenital CPEO syndromes (mitochondrial) by family history, "
            "biopsy (no ragged red fibres), normal lactate",
            "DNM2-R465W-MOST-SEVERE: the R465W variant is the most common and most severe DNM2 "
            "CNM mutation — neonatal onset, ventilator dependence, severe hypotonia; "
            "R522H and other variants cause milder adult-onset CNM; "
            "genotype-informed counselling essential",
            "DNM2-PERIPHERAL-NEUROPATHY-ASSESS: NCS must be performed in ALL DNM2 mutation carriers — "
            "intermediate NCS (MCV 25-45 m/s) indicates CMTDIB; "
            "pure myopathy patients may still have subclinical NCS slowing; "
            "this guides physiotherapy (foot drop orthoses) and differential diagnosis",
            "DNM2-RESPIRATORY-MONITOR: NIV may be required in CNM3 patients — "
            "annual FVC assessment; sleep study if symptomatic; "
            "respiratory decline is slower than MTM1 but still progressive in severe cases",
        ],
        "etiologies": [
            "CNM3 severe (R465W DNM2) — neonatal/congenital onset; "
            "severe hypotonia; ventilator dependence; ophthalmoplegia; "
            "central nuclei + hub-and-spoke on biopsy",
            "CNM3 moderate (other CNM DNM2 variants) — childhood onset; "
            "proximal weakness; ptosis/ophthalmoplegia; slowly progressive; "
            "ambulation maintained; NIV in third-fourth decade",
            "CNM3 adult onset (R522H DNM2) — presentation 20-50y; "
            "mild proximal weakness; late respiratory involvement; "
            "cardiomyopathy rarely co-reported",
            "CMTDIB DNM2 neuropathy — intermediate NCS; "
            "distal motor weakness; foot drop; minimal or no myopathy on biopsy; "
            "CMT phenotype clinically",
            "DNM2 neuromyopathy — combined CNM3 + CMTDIB; "
            "myopathy + peripheral neuropathy co-occurring; "
            "rare but diagnostically challenging",
            "De novo DNM2 — no family history; de novo haploinsufficiency; "
            "severity depends on variant class; "
            "germline mosaicism documented",
        ],
        "stats": {
            "ophthalmoplegia_pct": 55,
            "ptosis_pct": 65,
            "peripheral_neuropathy_pct": 40,
            "ambulation_maintained_pct": 75,
            "respiratory_niv_needed_pct": 40,
        },
    },
    # ── BIN1 — Centronuclear Myopathy 2 ──────────────────────────────────────
    {
        "gene": "BIN1",
        "protein": (
            "BIN1 — 2q14.3 AR(typical)/ADrare — BridgingIntegrator1-452aa — "
            "CentronuclearMyopathy2-CNM2 — TtubuleBiogenesis — "
            "NO-Cardiac-NeuropathyRare — MilderThanMTM1"
        ),
        "alias": (
            "BIN1 (bridging integrator 1; amphiphysin 2); OMIM gene 601248; "
            "CNM2 (centronuclear myopathy 2) OMIM 255200. "
            "2q14.3; 452 aa; ~52 kDa; typically autosomal recessive (biallelic), "
            "rarely autosomal dominant (some variants). "
            "FUNCTION: BIN1 (amphiphysin 2) is a membrane-curvature sensing protein "
            "with a BAR domain (Bin/Amphiphysin/Rvs domain) that: "
            "(1) detects and induces membrane curvature; "
            "(2) is essential for T-tubule biogenesis in skeletal muscle — "
            "BIN1 tubulates the sarcolemma to form T-tubule invaginations; "
            "(3) interacts with DNM2 and MTM1 — forms a regulatory complex "
            "that coordinates T-tubule formation and maintenance; "
            "(4) clathrin-mediated endocytosis (in non-muscle cells). "
            "CNM2 MECHANISM: "
            "Biallelic BIN1 mutations (AR) → defective membrane tubulation → "
            "T-tubule disorganisation → triad dysfunction → impaired E-C coupling → "
            "centronuclear histopathology. "
            "CLINICAL FEATURES: "
            "Typically congenital or early infantile onset; "
            "hypotonia; motor delay; "
            "proximal > distal weakness; "
            "facial weakness (variable); "
            "PTOSIS + ophthalmoplegia (variable — present in some, absent in others); "
            "MILDER than MTM1 (XLMTM) — respiratory failure less severe; "
            "most patients are ambulatory; "
            "NO cardiac involvement (BIN1 heart spared); "
            "slowly progressive or relatively stable; "
            "intelligence NORMAL; "
            "RARE AD dominant: some heterozygous BIN1 variants cause mild myopathy + "
            "possible cardiac arrhythmia susceptibility (BIN1 expressed in cardiac myocytes); "
            "BIN1-DNM2-MTM1 MOLECULAR TRIAD: "
            "all three CNM genes (BIN1, DNM2, MTM1) interact in T-tubule biology — "
            "understanding their shared pathway is essential for emerging therapeutic targeting."
        ),
        "locus": "2q14.3",
        "aa": 452,
        "kDa": 52,
        "omim_gene": "601248",
        "omim_disease": "255200",
        "inheritance": (
            "AR — biallelic LOF (most cases; both parents carriers; 25% sibling risk); "
            "rare AD — heterozygous dominant-negative variants described; "
            "most BIN1 myopathy is AR with negative family history"
        ),
        "gene_class": (
            "BAR-domain membrane-curvature sensor protein (Bin/Amphiphysin/Rvs family); "
            "T-tubule biogenesis effector; "
            "interacts with DNM2 (GTPase) and MTM1 (phosphatase) in T-tubule maintenance complex; "
            "clathrin adaptor; PI(4,5)P2 binding (PH domain)"
        ),
        "key_alerts": [
            "BIN1-DNM2-MTM1-MOLECULAR-TRIAD: BIN1, DNM2, and MTM1 form an interconnected T-tubule "
            "biogenesis complex — all three cause centronuclear myopathy by disrupting the same pathway; "
            "genetic testing must include all three genes when centronuclear histology is found; "
            "BIN1 (AR, milder), DNM2 (AD, intermediate), MTM1 (XLR, severe) differ in severity "
            "and inheritance pattern but share biologic mechanism",
            "BIN1-MILDER-THAN-MTM1: AR-CNM2 (BIN1) is significantly milder than XLMTM (MTM1) — "
            "most BIN1 patients are ambulatory; respiratory failure is less common; "
            "do not apply MTM1 prognosis to BIN1 patients; "
            "lifespan can be near-normal with appropriate support",
            "BIN1-AR-SIBLING-RISK: both parents of a BIN1 CNM2 patient are obligate carriers; "
            "25% recurrence risk per pregnancy; "
            "prenatal/preimplantation genetic testing available; "
            "cascade testing of siblings planning families",
            "BIN1-OPHTHALMOPLEGIA-VARIABLE: ptosis and ophthalmoplegia are present in some BIN1 patients "
            "but variable — less consistent than in MTM1; "
            "facial weakness also variable; "
            "document ocular motor findings at every assessment; "
            "ptosis surgery if axis is compromised",
            "BIN1-NO-CARDIAC-TYPICALLY: BIN1 AR myopathy does NOT cause cardiac involvement — "
            "this distinguishes from MYH7 and LMNA; "
            "ECG + echo to confirm cardiac sparing; "
            "rare AD BIN1 variants may have arrhythmia susceptibility (Brugada-like) in some pedigrees — "
            "if AD suspected, cardiac evaluation warranted",
            "BIN1-BIOPSY-CENTRONUCLEAR: central nuclei on routine H&E + oxidative enzyme staining — "
            "radial sarcoplasmic strands on NADH-TR; "
            "similar to DNM2-CNM3 histology; "
            "MTM1 has 'necklace fibres'; "
            "electron microscopy for T-tubule architecture distinction",
        ],
        "etiologies": [
            "AR-CNM2 typical (biallelic BIN1 truncating/LOF) — congenital onset; "
            "neonatal hypotonia; motor delay; proximal weakness; "
            "ambulatory most cases; mild-moderate course; most common BIN1 phenotype",
            "AR-CNM2 severe (biallelic BIN1 null) — severe neonatal hypotonia; "
            "early respiratory compromise; ventilator requirement; "
            "rare; worse prognosis; facial + ocular weakness",
            "AR-CNM2 mild (biallelic BIN1 hypomorphic) — childhood or later presentation; "
            "mild proximal weakness; near-normal function; slowly progressive",
            "AD-CNM dominant (heterozygous BIN1 dominant-negative) — rare; "
            "late onset; mild myopathy; possible arrhythmia in some pedigrees; "
            "family history positive; genotype-specific",
            "BIN1 exon deletion AR — large genomic deletion; "
            "detected by MLPA/array-CGH; "
            "standard NGS may miss; severe phenotype",
            "BIN1 compound heterozygote (truncating + missense) — variable severity; "
            "depends on residual BAR domain function; "
            "most common molecular explanation in AR cases",
        ],
        "stats": {
            "ambulation_maintained_pct": 80,
            "ophthalmoplegia_pct": 40,
            "respiratory_niv_pct": 25,
            "cardiac_involvement_pct": 2,
            "cognition_normal_pct": 100,
        },
    },
]


def _make_cohort(gene_dict, seed):
    rng = random.Random(seed)
    gene = gene_dict["gene"]
    patients = []
    for pid in range(40):
        if gene == "RYR1":
            onset_age = rng.gauss(3.5, 4.0)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(30, 14)
            dx_delay = max(3, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "mhs_susceptible": rng.random() < 0.85,
                "cores_on_biopsy": rng.random() < 0.90,
                "scoliosis": rng.random() < 0.55,
                "cardiac_involvement": False,
                "extraocular_sparing": True,
            })
        elif gene == "MYH7":
            onset_age = rng.gauss(7.0, 8.0)
            onset_age = max(0.5, onset_age)
            dx_delay = rng.gauss(48, 18)
            dx_delay = max(6, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "foot_drop": rng.random() < 0.80,
                "cardiac_echo_abnormal": rng.random() < 0.35,
                "scapular_winging": rng.random() < 0.50,
                "neck_flexor_weakness": rng.random() < 0.60,
            })
        elif gene == "ACTA1":
            onset_age = rng.gauss(0.2, 0.5)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(18, 10)
            dx_delay = max(1, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "neonatal_critical": rng.random() < 0.30,
                "de_novo": rng.random() < 0.70,
                "respiratory_niv": rng.random() < 0.75,
                "nemaline_rods_biopsy": True,
                "facial_weakness": rng.random() < 0.80,
            })
        elif gene == "NEB":
            onset_age = rng.gauss(0.5, 1.0)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(28, 12)
            dx_delay = max(4, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "ankle_contractures": rng.random() < 0.85,
                "nemaline_rods_biopsy": True,
                "respiratory_niv": rng.random() < 0.35,
                "ambulation_maintained": rng.random() < 0.80,
                "triplicate_mlpa_tested": rng.random() < 0.60,
            })
        elif gene == "SELENON":
            onset_age = rng.gauss(3.0, 2.5)
            onset_age = max(0.3, onset_age)
            dx_delay = rng.gauss(42, 16)
            dx_delay = max(6, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "rigid_spine": rng.random() < 0.98,
                "scoliosis": rng.random() < 0.85,
                "cardiac_involvement": False,
                "respiratory_niv": rng.random() < 0.80,
                "minicores_biopsy": rng.random() < 0.90,
            })
        elif gene == "MTM1":
            onset_age = rng.gauss(0.0, 0.1)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(8, 5)
            dx_delay = max(1, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "ventilator_dependent": rng.random() < 0.85,
                "ophthalmoplegia": rng.random() < 0.75,
                "ptosis": rng.random() < 0.80,
                "cognition_normal": True,
                "neonatal_death_without_support": rng.random() < 0.70,
                "biliary_complication": rng.random() < 0.15,
            })
        elif gene == "DNM2":
            onset_age = rng.gauss(5.0, 8.0)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(60, 24)
            dx_delay = max(6, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "ophthalmoplegia": rng.random() < 0.55,
                "ptosis": rng.random() < 0.65,
                "peripheral_neuropathy": rng.random() < 0.40,
                "ambulation_maintained": rng.random() < 0.75,
                "respiratory_niv": rng.random() < 0.40,
            })
        else:  # BIN1
            onset_age = rng.gauss(1.5, 2.5)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(36, 16)
            dx_delay = max(4, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "ophthalmoplegia": rng.random() < 0.40,
                "ptosis": rng.random() < 0.45,
                "cardiac_involvement": rng.random() < 0.02,
                "ambulation_maintained": rng.random() < 0.80,
                "respiratory_niv": rng.random() < 0.25,
            })

    ages = [p["onset_age"] for p in patients]
    delays = [p["dx_delay_months"] for p in patients]
    return {
        "gene": gene_dict["gene"],
        "protein": gene_dict["protein"],
        "alias": gene_dict["alias"],
        "locus": gene_dict["locus"],
        "aa": gene_dict["aa"],
        "kDa": gene_dict["kDa"],
        "omim_gene": gene_dict["omim_gene"],
        "omim_disease": gene_dict.get("omim_disease", ""),
        "inheritance": gene_dict["inheritance"],
        "gene_class": gene_dict["gene_class"],
        "key_alerts": gene_dict["key_alerts"],
        "etiologies": gene_dict["etiologies"],
        "stats": gene_dict["stats"],
        "sample_patients": patients[:10],
        "computed": {
            "n_patients": len(patients),
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
        },
    }


def _build_all():
    cohorts = []
    for i, gene in enumerate(CM_GENES):
        cohorts.append(_make_cohort(gene, SEED_BASE + i))
    return cohorts


def get_overview():
    cohorts = _build_all()
    all_ages = [p["onset_age"] for c in cohorts for p in c["sample_patients"]]
    all_delays = [p["dx_delay_months"] for c in cohorts for p in c["sample_patients"]]
    top_alerts = []
    for c in cohorts:
        if c["key_alerts"]:
            top_alerts.append(c["key_alerts"][0])
    return {
        "atlas": "Hereditary-Congenital-Myopathy-Atlas — Complete 8-Gene Hereditary Congenital Myopathy Atlas",
        "subtitle": (
            "RYR1 (CCD/MmD/MHS1 — Anaesthesia Emergency) · MYH7 (Laing Distal/DCM — Cardiac Surveillance) · "
            "ACTA1 (NEM3 — Nemaline Rods / Respiratory) · NEB (NEM2 — Largest Gene / Ankle Contractures) · "
            "SELENON (RSMD1 — Rigid Spine / No Cardiac) · MTM1 (XLMTM — Most Severe / Gene Therapy) · "
            "DNM2 (CNM3/CMT-DIB — Central Nuclei / Neuropathy Overlap) · "
            "BIN1 (CNM2 — AR / T-tubule / Milder) — 320 Patients (8×40, Seeds 1726–1733)"
        ),
        "total_patients": 320,
        "seed_range": "1726–1733",
        "aggregate_stats": {
            "genes_covered": 8,
            "patients_per_gene": 40,
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
        },
        "genes": [
            {
                "gene": c["gene"],
                "locus": c["locus"],
                "aa": c["aa"],
                "kDa": c["kDa"],
                "mean_dx_age": c["computed"]["mean_dx_age"],
                "mean_dx_delay_months": c["computed"]["mean_dx_delay_months"],
                "n_patients": c["computed"]["n_patients"],
            }
            for c in cohorts
        ],
        "top_alerts": top_alerts,
    }


def get_breakdown():
    cohorts = _build_all()
    return [
        {
            "gene": c["gene"],
            "protein": c["protein"],
            "locus": c["locus"],
            "aa": c["aa"],
            "kDa": c["kDa"],
            "omim_gene": c["omim_gene"],
            "omim_disease": c["omim_disease"],
            "inheritance": c["inheritance"],
            "gene_class": c["gene_class"],
            "key_alerts": c["key_alerts"],
            "etiologies": c["etiologies"],
            "alias": c["alias"],
            "stats": c["stats"],
            "sample_patients": c["sample_patients"],
            "computed": c["computed"],
        }
        for c in cohorts
    ]


def get_definitions():
    return {
        "concepts": {
            "Congenital Myopathy vs Muscular Dystrophy — Critical Structural Distinction": (
                "Congenital myopathies (RYR1, MYH7, ACTA1, NEB, SELENON, MTM1, DNM2, BIN1) differ "
                "fundamentally from muscular dystrophies (DMD, DYSF, CAPN3, LMNA) in histopathology: "
                "Congenital myopathies: structural sarcomere/T-tubule abnormalities (cores, rods, central nuclei); "
                "minimal or no necrosis-regeneration cycle; CK usually normal or mildly elevated. "
                "Muscular dystrophies: sarcolemmal or cytoskeletal protein loss → necrosis-regeneration cycle → "
                "fibrosis; CK often markedly elevated (especially Duchenne). "
                "This distinction guides biopsy interpretation and genetic testing panel selection."
            ),
            "Centronuclear Myopathy Triad — MTM1, DNM2, BIN1 Molecular Pathway": (
                "MTM1, DNM2, and BIN1 are genetically distinct but share a common molecular pathway: "
                "T-tubule biogenesis and membrane trafficking. "
                "BIN1 (amphiphysin 2) initiates T-tubule tubulation (membrane curvature sensing); "
                "DNM2 (dynamin 2) finalises T-tubule scission (GTPase membrane fission); "
                "MTM1 (myotubularin) regulates PI(3)P on endosomes adjacent to T-tubules. "
                "Disruption of any of these three → disorganised T-tubules → "
                "failed E-C coupling → centronuclear histopathology. "
                "Key clinical distinguishers: severity (MTM1 > DNM2 > BIN1); "
                "inheritance (XLR vs AD vs AR); ophthalmoplegia severity."
            ),
            "Nemaline Myopathy — Multi-Gene Biopsy-Driven Diagnosis": (
                "Nemaline rods on Gomori trichrome biopsy are PATHOGNOMONIC for nemaline myopathy "
                "but NOT gene-specific — rods occur in NEB (NEM2), ACTA1 (NEM3), TPM2, TPM3, "
                "TNNT1 (Finnish nemaline), LMOD3, CFL2, MYPN, and others. "
                "Comprehensive nemaline panel sequencing is required after biopsy diagnosis. "
                "NEB and ACTA1 are the two most common causes of congenital nemaline. "
                "NEB (AR) — ankle contractures prominent; mild respiratory; largest gene (need MLPA for triplicate). "
                "ACTA1 (AD 70% de novo / AR) — more severe respiratory; neonatal lethal subtype; "
                "de novo mutations common; facial weakness prominent."
            ),
            "RYR1 Malignant Hyperthermia — Anaesthesia Emergency Protocol": (
                "Malignant hyperthermia (MH) is a life-threatening pharmacogenomic disorder triggered by "
                "volatile anaesthetics (halothane, sevoflurane, isoflurane, desflurane) and succinylcholine "
                "in RYR1 mutation carriers. "
                "Mechanism: uncontrolled SR Ca2+ release → sustained muscle contraction → "
                "hyperthermia, rigidity, hypercarbia, acidosis, hyperkalemia, arrhythmia, rhabdomyolysis. "
                "Treatment: dantrolene 2.5 mg/kg IV immediately (then repeat doses); "
                "cooling; bicarbonate; hyperkalemia management; ICU. "
                "Prevention: total IV anaesthesia (propofol + remifentanil); avoid all triggering agents; "
                "pre-operative MH flag essential; MH cart stocked in all ORs. "
                "ALL CCD patients and ALL RYR1 variant carriers must be treated as MHS — "
                "MH susceptibility is lifelong and fully preventable with correct anaesthesia."
            ),
            "SELENON Rigid Spine vs EMD/LMNA — No ICD for SELENON": (
                "Rigid spine with respiratory involvement can occur in SELENON (RSMD1), EMD (EDMD1), "
                "and LMNA (EDMD2, DCM). The KEY distinguisher is cardiac involvement: "
                "SELENON: NO cardiac involvement — heart is completely spared; NO ICD required. "
                "EMD (emerin, Xq28 XLR): rigid spine + early contractures + LETHAL cardiac arrhythmia; "
                "ICD mandatory; first-degree cardiac event can be sudden death without prior symptoms. "
                "LMNA (lamin A/C, 1q22 AD): rigid spine + DCM + atrial arrhythmias; "
                "ICD regardless of LVEF using Padua score; most lethal MD. "
                "Misclassifying SELENON as LMNA would lead to unnecessary ICD implantation; "
                "misclassifying LMNA as SELENON would risk preventable sudden cardiac death."
            ),
            "MYH7 Dual Panel Inclusion — Cardiac and Neuromuscular Overlap": (
                "MYH7 is uniquely present on BOTH cardiac (HCM/DCM panels) and neuromuscular (myopathy) panels. "
                "The same gene encodes both skeletal (slow-twitch) and cardiac (ventricular) beta-myosin. "
                "Cardiac mutations (globular head): HCM (hypercontractility) or DCM (dilated). "
                "Skeletal myopathy mutations (rod domain): Laing distal myopathy (distal-onset); "
                "hyaline body myopathy (protein aggregation). "
                "ALL MYH7 carriers require cardiac surveillance regardless of skeletal phenotype — "
                "some family members develop only cardiac disease, others only skeletal, others both."
            ),
            "MTM1 Gene Therapy — Investigational Status": (
                "X-linked myotubular myopathy (XLMTM) has been the target of AAV-based gene therapy "
                "due to clear genetic cause and severe unmet need. "
                "AT132 (Aspiro trial, Solid Biosciences): AAV8 vector delivering MTM1 cDNA; "
                "Phase 1/2 ASPIRO trial demonstrated improved ventilator-free breathing in survivors; "
                "trial was paused following serious adverse events including deaths potentially related to "
                "hepatotoxicity or immune reactions in patients with pre-existing liver disease. "
                "The trial and programme underwent safety review and continue in modified form. "
                "Do NOT present gene therapy as approved or available outside trial context. "
                "Families should be connected to MTM1-specialty centres and monitored trial registries."
            ),
            "Largest Human Gene — NEB Sequencing Challenge": (
                "Nebulin (NEB) is the largest known human protein (6669 amino acids, ~800 kDa, 249 exons). "
                "The triplicate-repeat region (exons 82-105) encodes homologous actin-binding modules — "
                "NGS reads from this region are difficult to map uniquely, causing 'alignment collapse.' "
                "This means standard WES or gene panel sequencing misses deletions in 10-15% of NEB cases. "
                "MLPA targeting NEB exons 82-105 is mandatory in any patient with nemaline biopsy "
                "but negative standard NEB sequencing. "
                "Some bioinformatics pipelines have developed NEB-specific alignment tools — "
                "ask your diagnostic lab whether they have validated NEB triplicate analysis."
            ),
        },
        "pharmacological_distinctions": [
            "RYR1/MHS — Dantrolene (Dantrium) IV 2.5 mg/kg: ACUTE MH TREATMENT — NOT prophylactic; "
            "given immediately on MH diagnosis; repeat doses every 5-10 min up to 10 mg/kg; "
            "cool, treat acidosis, hyperkalemia; ICU; ORAL dantrolene pre-treatment for elective surgery "
            "in high-risk cases (controversial — most prefer TIVA avoidance strategy)",
            "RYR1/CCD — Anaesthesia management: TIVA (propofol + remifentanil) is SAFE; "
            "avoid all volatile agents and succinylcholine; "
            "all OR staff must be briefed; MH cart stocked; dantrolene available within 5 minutes",
            "MYH7/DCM overlap — ACE inhibitor/ARB + beta-blocker for DCM; "
            "anticoagulation if LVEF <35% or AF; ICD for LVEF <35%; "
            "cardiac transplant in end-stage DCM",
            "ACTA1/NEM3 — No disease-modifying therapy; supportive: NIV (BiPAP), "
            "chest physiotherapy, gastrostomy, physiotherapy, orthoses; "
            "trofinetide and other actin-filament stabilisers in preclinical investigation",
            "NEB/NEM2 — No disease-modifying therapy; ankle contracture management: "
            "serial casting, AFOs, surgical Achilles tendon lengthening; "
            "NIV for respiratory compromise; physiotherapy",
            "SELENON/RSMD1 — No disease-modifying therapy; scoliosis management: "
            "bracing, surgical correction; NIV essential (respiratory muscle weakness prominent); "
            "selenium supplementation — NOT proven effective; avoid as unproven",
            "MTM1/XLMTM — No approved therapy; investigational: AT132 gene therapy (ASPIRO trial — "
            "paused for safety review; consult specialist centres); "
            "supportive: tracheostomy, ventilator, gastrostomy, physiotherapy; "
            "biliary surveillance; avoid valproate (hepatotoxicity risk with underlying liver disease)",
            "DNM2/CNM3 — No disease-modifying therapy; supportive: NIV, physiotherapy, AFOs for foot drop; "
            "neuropathy management (CMT-DIB overlap); annual cardiac + pulmonary assessment",
            "BIN1/CNM2 — No disease-modifying therapy; supportive: physiotherapy, AFOs; "
            "respiratory monitoring; ptosis surgery if indicated; "
            "milder than MTM1 — prognosis counselling should reflect this",
        ],
        "key_standards": [
            "CONGENITAL MYOPATHY STANDARDS: North Star Network (UK) and TREAT-NMD standards for congenital myopathy; "
            "annual multidisciplinary review (neurology, pulmonology, orthopaedics, nutrition, genetics, physio); "
            "FVC + MIP/MEP at every visit; scoliosis surveillance; physiotherapy-led",
            "MH STANDARDS: European MH Group (EMHG) and MHAUS (North America) clinical guidelines; "
            "CHCT (caffeine-halothane contracture test) or IVCT (in-vitro contracture test) for MHS when genetic result uncertain; "
            "MedicAlert bracelet mandatory for ALL MHS-confirmed individuals",
            "MTM1 STANDARDS: A Foundation Building Strength (AFBS) and MTM/CNM family support groups; "
            "ASPIRO trial site listings for gene therapy referrals; "
            "palliative care integration early given severity of XLMTM; "
            "tracheostomy discussion antenatally when XLMTM suspected prenatally",
            "NEMALINE MYOPATHY STANDARDS: ENMC (European Neuromuscular Centre) workshops on nemaline myopathy; "
            "Myoseq and large gene panels; NEB triplicate MLPA mandatory; "
            "ACTA1 de novo recurrence risk counselling; respiratory surveillance from diagnosis",
            "MYH7 CARDIAC STANDARDS: Cardio-neuromuscular joint clinic; "
            "annual ECG + echo for all MYH7 carriers; "
            "ACC/AHA HCM/DCM guidelines for cardiac MYH7 variants; "
            "ICD per cardiac guideline criteria (NOT per neuromuscular myopathy criteria)",
            "RESPIRATORY MONITORING ALL CONGENITAL MYOPATHY: FVC sitting + lying at every visit; "
            "overnight oximetry if symptoms; formal polysomnography for sleep hypoventilation; "
            "early NIV initiation improves survival and quality of life across all subtypes",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = get_breakdown()
    print(json.dumps(bd[0], indent=2)[:2000])
    print("\n=== DEFINITIONS ===")
    print(json.dumps(get_definitions(), indent=2)[:1000])
