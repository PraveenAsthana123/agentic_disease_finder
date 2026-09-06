#!/usr/bin/env python3
"""Hereditary-Muscular-Dystrophy-Atlas — Complete 8-Gene Hereditary Muscular Dystrophy Atlas
DMD    (Dystrophin; 3685 aa; Xp21.2; XLR;
         Duchenne MD — most common hereditary MD; out-of-frame deletions/duplications/nonsense;
         Exon-skip therapy mutation-specific (eteplirsen exon51, golodirsen/viltolarsen exon53,
         casimersen exon45, ataluren PTC); AVOID succinylcholine (hyperkalemia→cardiac arrest);
         Glucocorticoids + ACE-i + beta-blocker mandatory; annual cardiopulmonary surveillance) ·
DYSF   (Dysferlin; 2080 aa; 2p13.2; AR;
         LGMD2B/R2 — dysferlinopathy — membrane repair deficit;
         Absent dysferlin on Western blot/immunostaining DIAGNOSTIC;
         AVOID statins (accelerate weakness); CK very high 5000–100000;
         Miyoshi myopathy (distal calf) vs LGMD2B (proximal) — same gene) ·
CAPN3  (Calpain-3; 821 aa; 15q15.1; AR;
         LGMD2A/R1 — most common AR LGMD globally;
         Scapular winging + hip flexor weakness; CAPN3 Western blot reduced/absent;
         Founder mutation Arg490Gln Basque population; CK 5–30× normal) ·
LMNA   (Lamin A/C; 664 aa; 1q22; AD;
         EDMD2 + LMNA-DCM + LMNA-LGMD1B;
         AV block → ventricular arrhythmia LIFE-THREATENING — ICD mandatory when arrhythmia;
         Annual Holter + Echo from diagnosis regardless of muscle severity;
         Arg453Trp, Glu161Lys arrhythmia hotspots → ICD early) ·
EMD    (Emerin; 254 aa; Xq28; XLR;
         EDMD1 — X-linked Emery-Dreifuss MD;
         Emerin absent from nuclear membrane immunostaining DIAGNOSTIC;
         Pacemaker/ICD mandatory by 3rd–4th decade; scapulohumeral-peroneal distribution;
         Female carriers: manifesting carrier — cardiac surveillance mandatory) ·
SGCA   (Sarcoglycan Alpha; 387 aa; 17q21.33; AR;
         LGMD2D/R3 — alpha-sarcoglycanopathy;
         Secondary loss of all 4 sarcoglycans (SGCA/B/C/D) on immunostaining;
         TEST ALL 4 sarcoglycans as a panel — primary vs secondary loss distinguishes gene;
         Calf pseudo-hypertrophy; CK 10–150× normal; Arg77Cys founder in French-Canadian) ·
DMPK   (Myotonic Protein Kinase; 629 aa; 19q13.32; AD;
         DM1 — Myotonic Dystrophy type 1 — most common adult-onset MD;
         CTG trinucleotide repeat: >50 pathogenic, 200–999 classic, ≥1000 congenital DM1;
         ANTICIPATION — worsens each generation; ANAESTHESIA RISK — propofol preferred,
         AVOID succinylcholine (hyperkalemia), AVOID depolarising agents;
         Mexiletine for myotonia; annual cardiac Holter; cataracts mandatory ophthalmology) ·
CNBP   (CCHC-Type Zinc Finger Nucleic Acid Binding Protein; 347 aa; 3q21.3; AD;
         DM2 — Myotonic Dystrophy type 2 — CCTG repeat in intron 1;
         Proximal > distal weakness; milder than DM1; NO congenital form;
         Myotonia often subclinical; cataracts; Mexiletine for myotonia;
         CK mild; DM2-specific: proximal leg pain, fluctuating weakness, neck flexor)
320-patient aggregate cohort (8 × 40, seeds 1478–1485)
"""

import random

SEED_BASE = 1478

MUSCULAR_DYSTROPHY_GENES = [
    # ── DMD — Duchenne / Becker MD ──
    {
        "gene": "DMD",
        "protein": "Dystrophin — Sub-Sarcolemmal Cytoskeletal Linker Protein",
        "alias": (
            "DMD; OMIM gene 300377; Duchenne MD OMIM 310200; Becker MD OMIM 300376; Xp21.2; 3685 aa; ~427 kDa; "
            "Dystrophin — largest human gene (2.4 Mb, 79 exons); links intracellular actin cytoskeleton "
            "to extracellular matrix via dystroglycan complex; X-linked recessive; "
            "Duchenne MD (DMD): out-of-frame deletions/duplications/nonsense → absent dystrophin → "
            "onset <5y, loss of ambulation by ~12y, cardiac/respiratory failure by 20s–30s; "
            "Becker MD (BMD): in-frame deletions → reduced/truncated dystrophin → milder, onset >5y; "
            "EXON-SKIP THERAPY (mutation-specific): eteplirsen (exon 51, ~13% DMD), "
            "golodirsen + viltolarsen (exon 53, ~8%), casimersen (exon 45, ~8%), "
            "ataluren (PTC readthrough, nonsense ~10%); "
            "SUCCINYLCHOLINE ABSOLUTE CI: hyperkalemic cardiac arrest risk in dystrophinopathy — "
            "ALL anaesthesia must flag 'known/suspected dystrophinopathy'; "
            "Glucocorticoids (deflazacort preferred/prednisone) mandatory — slow ambulation loss by 2–5y; "
            "ACE inhibitor + beta-blocker from diagnosis (cardiomyopathy prevents); "
            "Nocturnal NIV when FVC <50% or pCO2 rising; spinal surveillance for scoliosis; "
            "exon 45–55 deletions most common (45% of all DMD deletions)"
        ),
        "aa": "3685 aa",
        "kDa": "~427 kDa",
        "locus": "Xp21.2",
        "omim_gene": 300377,
        "omim_disease": 310200,
        "inheritance": "XLR — out-of-frame (DMD) or in-frame (BMD) deletions/duplications/nonsense",
        "gene_class": (
            "DMD encodes dystrophin, the largest known human protein and the most structurally complex "
            "component of the dystrophin-associated protein complex (DAPC). Dystrophin acts as a "
            "mechanical shock-absorber during muscle contraction, linking filamentous actin inside "
            "the myofibre to the extracellular matrix via beta-dystroglycan and laminin-211. "
            "Frame-disrupting mutations cause absent dystrophin (Duchenne), whereas in-frame "
            "mutations produce a truncated but partially functional protein (Becker). "
            "Dystrophin deficiency triggers cycles of necrosis-regeneration, progressive fibrosis, "
            "and eventual respiratory and cardiac failure. The reading-frame rule predicts phenotype "
            "with ~92% accuracy: out-of-frame = Duchenne, in-frame = Becker. "
            "Modern mutation-specific exon skipping (phosphorodiamidate morpholino oligomers) "
            "converts an out-of-frame deletion into an in-frame deletion by skipping the flanking exon, "
            "producing Becker-like dystrophin. Eligibility requires knowing the exact deletion/duplication "
            "boundaries — MLPA is the mandatory first step before sequencing for DMD."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("Duchenne — out-of-frame exon deletion (hotspot 45–55)", 0.48),
            ("Duchenne — out-of-frame exon duplication", 0.12),
            ("Duchenne — nonsense mutation (PTC, ataluren amenable)", 0.10),
            ("Becker — in-frame deletion", 0.22),
            ("DMD VUS / mosaic / deep intronic", 0.08),
        ],
        "onset_range": (2, 25),
        "key_alerts": [
            "SUCCINYLCHOLINE-ABSOLUTE-CI: ALL anaesthesia → hyperkalemia → cardiac arrest in dystrophinopathy",
            "DMD-EXON-SKIP: mutation-specific — confirm exact exon boundaries (MLPA) before selecting therapy",
            "GLUCOCORTICOIDS-MANDATORY: deflazacort/prednisone slows ambulation loss by 2-5 years",
            "CARDIAC-ACE-I-FROM-DIAGNOSIS: prevent DMD cardiomyopathy — DO NOT wait for LV dysfunction",
            "DMD-MLPA-FIRST: MLPA detects 65-70% deletions/duplications; sequencing for remainder",
            "BMD-CARDIAC-DISPROPORTIONATE: Becker cardiac failure can exceed skeletal muscle severity",
            "NOCTURNAL-NIV: start when FVC <50% or symptomatic nocturnal hypoventilation",
            "CASCADE-Testing: All maternal relatives — CK + MLPA for carrier status",
        ],
    },
    # ── DYSF — Dysferlinopathy ──
    {
        "gene": "DYSF",
        "protein": "Dysferlin — Membrane Repair C2-Domain Calcium-Sensing Protein",
        "alias": (
            "DYSF; OMIM gene 603009; LGMD2B OMIM 253601; Miyoshi Myopathy OMIM 254130; 2p13.2; 2080 aa; ~237 kDa; "
            "Dysferlin — type II transmembrane protein containing 7 C2 domains; expressed in sarcolemma; "
            "mediates calcium-dependent membrane repair by sensing microlesions and "
            "fusing lysosomes/vesicles to patch sarcolemmal tears; "
            "biallelic LOF → dysferlinopathy: CK very high 5000–100000 IU/L; "
            "Two main phenotypes: LGMD2B (proximal hip-girdle) and Miyoshi Myopathy (distal calf, "
            "unable to stand on tiptoe — distinctive test); same gene, different phenotypes even in "
            "same family; inflammation on biopsy mimics polymyositis → AVOID immunosuppression; "
            "Absent dysferlin on Western blot (muscle) or monocyte flow cytometry (blood) DIAGNOSTIC; "
            "AVOID STATINS: accelerate dysferlinopathy weakness (sarcolemmal membrane further stressed); "
            "No disease-modifying therapy; exercise: swimming/aquatic therapy beneficial (low-impact); "
            "Founder mutations: Arg1000Trp (Libyan Jewish), p.del32+1G>A intron 32"
        ),
        "aa": "2080 aa",
        "kDa": "~237 kDa",
        "locus": "2p13.2",
        "omim_gene": 603009,
        "omim_disease": 253601,
        "inheritance": "AR — biallelic LOF (missense/nonsense/frameshift/splice)",
        "gene_class": (
            "DYSF encodes dysferlin, a 237 kDa protein belonging to the ferlin family of calcium sensor "
            "proteins. Its seven tandem C2 domains (C2A–C2G) bind phospholipids and membranes in a "
            "calcium-dependent manner, facilitating lysosomal exocytosis and membrane resealing after "
            "mechanical micro-tears during muscle contraction. Dysferlin deficiency results in "
            "an inability to patch sarcolemmal lesions, leading to calcium influx, "
            "mitochondrial dysfunction, and progressive myofibre necrosis. "
            "The inflammatory infiltrate seen on muscle biopsy can be prominent and misleads clinicians "
            "into a polymyositis misdiagnosis — immunosuppressive therapy in dysferlinopathy is harmful. "
            "Western blot of muscle biopsy showing absent dysferlin protein is the gold-standard diagnostic "
            "test, complemented by monocyte flow cytometry as a rapid blood-based screen. "
            "The clinical presentation ranges from distal (Miyoshi — toe-standing test) to proximal "
            "(LGMD2B), with the 'tiptoe sign' — inability to stand on tiptoes despite walking — "
            "being a useful early clinical clue in Miyoshi phenotype."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("LGMD2B — proximal hip-girdle onset", 0.45),
            ("Miyoshi Myopathy — distal calf onset (tiptoe sign)", 0.35),
            ("DYSF compound heterozygous VUS", 0.12),
            ("Dysferlinopathy — overlapping proximal + distal phenotype", 0.08),
        ],
        "onset_range": (15, 40),
        "key_alerts": [
            "DYSF-WESTERN-BLOT: absent dysferlin on muscle Western blot or monocyte flow cytometry DIAGNOSTIC",
            "AVOID-STATINS: statins accelerate dysferlin-deficient membrane fragility — ABSOLUTE CI",
            "MIYOSHI-TIPTOE-TEST: inability to stand on tiptoes is pathognomonic for distal dysferlinopathy",
            "NOT-POLYMYOSITIS: inflammatory biopsy mimics PM — DO NOT immunosuppress dysferlinopathy",
            "CK-VERY-HIGH: 5000-100000 IU/L typical — highest CK in any LGMD subtype",
            "DYSF-AQUATIC-THERAPY: swimming/low-impact beneficial — avoids sarcolemmal mechanical stress",
            "FOUNDER-Arg1000Trp: Libyan Jewish population — targeted test first",
            "CASCADE-Testing: DYSF biallelic — sibling risk 25% — CK + targeted sequencing",
        ],
    },
    # ── CAPN3 — LGMD2A / Calpainopathy ──
    {
        "gene": "CAPN3",
        "protein": "Calpain-3 — Muscle-Specific Calcium-Activated Neutral Protease",
        "alias": (
            "CAPN3; OMIM gene 114240; LGMD2A OMIM 253600; 15q15.1; 821 aa; ~94 kDa; "
            "Calpain-3 — muscle-specific calcium-dependent cysteine protease; "
            "most common AR LGMD subtype globally (30–40% of all AR LGMD); "
            "biallelic mutations → progressive limb-girdle weakness: scapular winging + hip flexors; "
            "CK 5–30× normal; CAPN3 Western blot shows reduced or absent protein; "
            "NO pseudo-hypertrophy (distinguishes from sarcoglycanopathies); "
            "Rigid spine may develop; cardiac involvement rare (important DDx from laminopathy); "
            "Basque founder mutation: Arg490Gln (c.1468C>T) present in 80% Basque LGMD2A; "
            "Autosomal recessive — compound heterozygous or homozygous; "
            "No disease-modifying therapy; physiotherapy; avoid prolonged inactivity"
        ),
        "aa": "821 aa",
        "kDa": "~94 kDa",
        "locus": "15q15.1",
        "omim_gene": 114240,
        "omim_disease": 253600,
        "inheritance": "AR — biallelic (homozygous or compound heterozygous)",
        "gene_class": (
            "CAPN3 encodes calpain-3, a muscle-specific member of the calpain superfamily of "
            "calcium-activated neutral cysteine proteases. Calpain-3 interacts with titin at the "
            "M-line and N2 line of the sarcomere, where it is thought to regulate sarcomeric "
            "protein turnover during adaptation to exercise and injury. Loss of calpain-3 activity "
            "disrupts sarcomere remodelling, leading to progressive myofibre degeneration predominantly "
            "affecting the hip girdle, thigh, and shoulder girdle muscles. "
            "LGMD2A (calpainopathy) is the most prevalent AR LGMD in Europe and globally. "
            "Key clinical features distinguishing it from sarcoglycanopathy include: "
            "absence of calf hypertrophy, rare cardiac involvement, frequent scapular winging, "
            "and a characteristic adductor-predominant thigh pattern on MRI. "
            "CAPN3 Western blot showing reduced or absent protein is the recommended first diagnostic "
            "test, complemented by genetic panel sequencing. Some pathogenic variants cause reduced "
            "but not absent calpain-3 activity (secondary reduction can also occur in other LGMDs)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("LGMD2A — Arg490Gln Basque founder (homozygous)", 0.22),
            ("LGMD2A — compound heterozygous missense", 0.42),
            ("LGMD2A — compound heterozygous (missense + splice/frameshift)", 0.26),
            ("CAPN3 VUS under evaluation", 0.10),
        ],
        "onset_range": (8, 35),
        "key_alerts": [
            "CAPN3-MOST-COMMON-AR-LGMD: 30-40% of all AR LGMD globally — test first in suspected AR LGMD",
            "CAPN3-WESTERN-BLOT: reduced/absent calpain-3 protein on muscle biopsy Western blot",
            "NO-CALF-HYPERTROPHY: distinguishes calpainopathy from sarcoglycanopathy",
            "SCAPULAR-WINGING: characteristic early sign — LGMD2A vs DMD carrier",
            "CARDIAC-RARE: cardiac spared in most CAPN3 — if AV block present → exclude LMNA first",
            "Arg490Gln-BASQUE: >80% Basque/Iberian LGMD2A carries this allele — targeted first",
            "MRI-ADDUCTOR-PATTERN: thigh adductor + posterior compartment predominance on MRI LGMD2A",
            "CASCADE-Testing: AR — sibling 25% risk — CK + Western blot + CAPN3 panel",
        ],
    },
    # ── LMNA — EDMD2 / LMNA Cardiomyopathy ──
    {
        "gene": "LMNA",
        "protein": "Lamin A/C — Nuclear Lamina Structural Protein and Mechano-Transducer",
        "alias": (
            "LMNA; OMIM gene 150330; EDMD2 OMIM 181350; LMNA-DCM OMIM 115200; LGMD1B OMIM 159001; 1q22; 664 aa; ~74 kDa; "
            "Lamin A/C — intermediate filament protein; nuclear envelope structural component; "
            "AD (haploinsufficiency or dominant-negative); "
            "Three striated muscle phenotypes: EDMD2 (scapulohumeral-peroneal + elbow contractures), "
            "LMNA-DCM (dilated cardiomyopathy + conduction disease, minimal myopathy), "
            "LGMD1B (proximal limb-girdle); all overlap; "
            "CARDIAC IS THE LETHAL THREAT: AV block + ventricular arrhythmia/VT/VF — "
            "ICD mandatory when arrhythmia or significant AV block present; "
            "Annual Holter + echo from diagnosis regardless of muscle severity; "
            "Arrhythmia hotspot mutations: Arg453Trp, Glu161Lys, Arg190Trp → ICD early; "
            "Joint contractures (elbow, Achilles) precede weakness — early hallmark; "
            "LMNA should be sequenced in ALL dilated cardiomyopathy with conduction disease"
        ),
        "aa": "664 aa",
        "kDa": "~74 kDa",
        "locus": "1q22",
        "omim_gene": 150330,
        "omim_disease": 181350,
        "inheritance": "AD — haploinsufficiency or dominant-negative missense",
        "gene_class": (
            "LMNA encodes lamins A and C via alternative splicing of the same transcript. "
            "Lamins A and C are type V intermediate filaments that form the nuclear lamina — "
            "a protein meshwork lining the inner nuclear membrane that maintains nuclear shape, "
            "chromatin organisation, DNA repair, and mechanotransduction. "
            "Pathogenic LMNA variants cause one of the most severe hereditary cardiomyopathy syndromes: "
            "unlike many genetic cardiomyopathies, laminopathy carries a high risk of lethal ventricular "
            "arrhythmia and AV block that can precede or be disproportionate to the cardiomyopathy. "
            "The 2019 HRS/EHRA expert consensus recommends prophylactic ICD implantation in LMNA "
            "mutation carriers with two or more of: non-sustained VT, LVEF <45%, male sex, "
            "non-missense mutation — irrespective of LVEF meeting traditional ICD thresholds. "
            "LMNA mutation should be actively sought in all familial DCM with AV block, "
            "as the arrhythmic risk mandates different management than idiopathic DCM."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("LMNA-DCM — dilated cardiomyopathy + AV block (minimal myopathy)", 0.40),
            ("EDMD2 — scapulohumeral-peroneal + elbow contractures", 0.32),
            ("LGMD1B — proximal limb-girdle + cardiac", 0.18),
            ("LMNA VUS under cardiology evaluation", 0.10),
        ],
        "onset_range": (20, 55),
        "key_alerts": [
            "LMNA-ICD-MANDATORY: AV block or ventricular arrhythmia → ICD — do NOT wait for LVEF <35%",
            "LMNA-ANNUAL-HOLTER: cardiac surveillance from diagnosis regardless of muscle severity",
            "LMNA-AV-BLOCK-LETHAL: sudden cardiac death from AV block → pacemaker/ICD is life-saving",
            "Arg453Trp-HOTSPOT: high arrhythmia risk mutation — ICD early even with preserved EF",
            "LMNA-ALL-DCM-WITH-CONDUCTION: sequence LMNA in ALL familial DCM + conduction disease",
            "ELBOW-CONTRACTURES-EARLY: elbow flexion contractures precede weakness — early hallmark EDMD",
            "LMNA-HRS-CRITERIA-2019: ICD if 2+ of: nsVT, LVEF<45%, male sex, non-missense mutation",
            "CASCADE-Testing: AD — 50% offspring risk — annual cardiac monitoring all first-degree",
        ],
    },
    # ── EMD — EDMD1 (X-linked Emery-Dreifuss) ──
    {
        "gene": "EMD",
        "protein": "Emerin — Nuclear Inner Membrane LEM-Domain Adaptor Protein",
        "alias": (
            "EMD; OMIM gene 300384; EDMD1 OMIM 310300; Xq28; 254 aa; ~29 kDa; "
            "Emerin — integral inner nuclear membrane protein; LEM domain binds BAF (barrier-to-autointegration factor); "
            "X-linked recessive; EDMD1: scapulohumeral-peroneal muscular dystrophy + "
            "rigid spine + early joint contractures (elbow, Achilles) + "
            "life-threatening cardiac conduction defects (AV block → VT/VF); "
            "Emerin absent from nuclear membrane on immunostaining of any nucleated cell DIAGNOSTIC — "
            "blood lymphocytes sufficient (avoids muscle biopsy); "
            "Pacemaker/ICD typically required by 3rd–4th decade (mandatory when AV block develops); "
            "Female carriers: manifesting carrier possible (cardiac surveillance mandatory); "
            "Phenotypically overlaps with LMNA-EDMD2 — same triad (contractures + myopathy + cardiac); "
            "EDMD1 cardiac can be lethal before significant myopathy develops"
        ),
        "aa": "254 aa",
        "kDa": "~29 kDa",
        "locus": "Xq28",
        "omim_gene": 300384,
        "omim_disease": 310300,
        "inheritance": "XLR — females are carriers (manifesting in ~15%)",
        "gene_class": (
            "EMD encodes emerin, a 254 aa integral protein of the inner nuclear membrane. "
            "Its N-terminal LEM domain interacts with BAF (BANF1), anchoring emerin to the "
            "inner nuclear membrane and linking it to the nuclear lamina, chromatin, and "
            "the LINC (Linker of Nucleoskeleton and Cytoskeleton) complex. "
            "Loss of emerin disrupts nuclear mechanical support and mechanosensitive gene regulation, "
            "particularly in post-mitotic cells like myofibres and cardiomyocytes. "
            "EDMD1 presents with the pathognomonic triad: (1) early muscle contractures "
            "(elbow fixed flexion, Achilles shortening, rigid cervical spine) preceding weakness; "
            "(2) slowly progressive scapulohumeral-peroneal muscular dystrophy; "
            "(3) cardiac arrhythmias and AV block that are potentially lethal independent of myopathy severity. "
            "Immunostaining of any accessible nucleated cell (skin fibroblasts, blood lymphocytes) "
            "for emerin is the most practical diagnostic test — absent nuclear membrane staining "
            "is highly diagnostic and avoids muscle biopsy in most cases. "
            "Female carriers can manifest cardiac disease requiring surveillance and ICD."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("EDMD1 — emerin-absent (large deletion/frameshift)", 0.52),
            ("EDMD1 — emerin-absent (nonsense)", 0.28),
            ("EDMD1 — emerin-reduced (missense, partial)", 0.12),
            ("EDMD1 — manifesting female carrier", 0.08),
        ],
        "onset_range": (5, 45),
        "key_alerts": [
            "EMD-IMMUNOSTAINING-DIAGNOSTIC: absent emerin from nuclear membrane on lymphocytes — avoids biopsy",
            "EMD-ICD-MANDATORY: AV block → pacemaker/ICD mandatory (typically 3rd-4th decade)",
            "EMD-ELBOW-CONTRACTURES-FIRST: contractures precede weakness — elbow + Achilles + cervical spine",
            "FEMALE-CARRIER-CARDIAC: manifesting carrier in ~15% — ALL female carriers need cardiac surveillance",
            "EMD-vs-LMNA: same triad (contractures + myopathy + cardiac) — immunostaining distinguishes",
            "EMD-CARDIAC-LETHAL-EARLY: AV block can cause sudden death before myopathy is severe",
            "RIGID-CERVICAL-SPINE: prevents neck flexion — early clinical clue in EDMD1",
            "CASCADE-Testing: XLR — maternal relatives — female carriers need cardiac monitoring + ICD evaluation",
        ],
    },
    # ── SGCA — LGMD2D / Alpha-Sarcoglycanopathy ──
    {
        "gene": "SGCA",
        "protein": "Sarcoglycan Alpha — Transmembrane Dystrophin-Associated Complex Subunit",
        "alias": (
            "SGCA; OMIM gene 600119; LGMD2D OMIM 608099; 17q21.33; 387 aa; ~43 kDa; "
            "Sarcoglycan-alpha — single-pass transmembrane glycoprotein; part of sarcoglycan complex "
            "(alpha/SGCA + beta/SGCB + gamma/SGCC + delta/SGCD) within dystrophin-associated complex; "
            "AR (LGMD2D); progressive proximal weakness; calf pseudo-hypertrophy; "
            "CK very high 10–150× normal; "
            "SECONDARY LOSS: primary mutation in any one sarcoglycan causes secondary loss of ALL 4 "
            "on immunostaining — PANEL TEST ALL 4 (SGCA/B/C/D) to identify primary gene; "
            "Founder mutation: Arg77Cys (c.229C>T) most common — French-Canadian + Southern European; "
            "Cardiac involvement uncommon but possible (annual echo); "
            "Phenotype similar to DMD but XLR excluded by female involvement + dystrophin normal; "
            "Gene therapy trials ongoing (SGT-003 clinical phase)"
        ),
        "aa": "387 aa",
        "kDa": "~43 kDa",
        "locus": "17q21.33",
        "omim_gene": 600119,
        "omim_disease": 608099,
        "inheritance": "AR — biallelic LOF",
        "gene_class": (
            "SGCA encodes the alpha subunit of the sarcoglycan complex — a heterotetrameric assembly "
            "of four single-pass transmembrane glycoproteins (SGCA, SGCB, SGCC, SGCD) that forms the "
            "structural core of the dystrophin-associated protein complex (DAPC) at the sarcolemma. "
            "The sarcoglycan complex bridges the extracellular matrix (via dystroglycan) with "
            "the intracellular dystrophin-actin network, providing mechanical reinforcement during "
            "muscle contraction and transmitting force laterally across the sarcolemma. "
            "Pathogenic mutations in any single sarcoglycan subunit destabilise the entire complex, "
            "causing secondary degradation of the other three sarcoglycans — which means that "
            "immunohistochemistry shows absent staining for all four sarcoglycans regardless of which "
            "gene is primarily mutated. Genetic panel testing of all four genes (SGCA/B/C/D) "
            "is therefore mandatory to identify the primary mutation and guide accurate recurrence "
            "risk counselling. Alpha-sarcoglycanopathy (LGMD2D) is the most common of the four, "
            "particularly in French-Canadian populations carrying the Arg77Cys founder allele."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("LGMD2D — Arg77Cys founder (homozygous)", 0.28),
            ("LGMD2D — compound heterozygous missense", 0.38),
            ("LGMD2D — compound heterozygous (missense + frameshift)", 0.22),
            ("SGCA VUS — secondary sarcoglycan loss pending primary identification", 0.12),
        ],
        "onset_range": (3, 25),
        "key_alerts": [
            "SGCA-PANEL-ALL-4: primary mutation in any sarcoglycan → secondary loss of ALL — test SGCA+B+C+D",
            "SGCA-SECONDARY-LOSS: absent immunostaining ALL 4 sarcoglycans = sarcoglycanopathy NOT diagnosis of SGCA specifically",
            "CALF-PSEUDO-HYPERTROPHY: distinguishes sarcoglycanopathy from calpainopathy (no hypertrophy)",
            "CK-VERY-HIGH: 10-150× normal — similar to DMD; dystrophin NORMAL distinguishes from DMD",
            "Arg77Cys-FOUNDER: targeted test first in French-Canadian or Southern European patients",
            "CARDIAC-ANNUAL-ECHO: sarcoglycanopathy can involve cardiac muscle — annual echo mandatory",
            "FEMALE-AFFECTED: AR inheritance — females equally affected (distinguishes from X-linked DMD)",
            "CASCADE-Testing: AR — sibling 25% risk — CK + SGCA/B/C/D panel",
        ],
    },
    # ── DMPK — DM1 (Myotonic Dystrophy type 1) ──
    {
        "gene": "DMPK",
        "protein": "Myotonic Protein Kinase — CTG Trinucleotide Repeat Regulator",
        "alias": (
            "DMPK; OMIM gene 605377; Myotonic Dystrophy 1 OMIM 160900; 19q13.32; 629 aa; ~69 kDa; "
            "Myotonic Protein Kinase — serine/threonine kinase; 3'UTR CTG trinucleotide repeat expansion; "
            "most common adult-onset muscular dystrophy (~1:8000); "
            "Repeat categories: 5–34 normal; 35–49 premutation; 50–999 classic DM1; ≥1000 congenital DM1; "
            "ANTICIPATION — CTG repeat expands each generation → earlier + more severe onset; "
            "Myotonia (grip myotonia — slow hand release) often first symptom; "
            "Multisystem disease: distal > proximal weakness; cataracts (subcapsular, mandatory slit-lamp); "
            "cardiac conduction (AV block → SCD); insulin resistance; hypersomnia; frontal balding (males); "
            "ANAESTHESIA RISK ABSOLUTE: propofol preferred (TIVA); "
            "AVOID succinylcholine (hyperkalaemia + prolonged paralysis); "
            "AVOID volatile anaesthetics where possible (prolonged myotonic response); "
            "Mexiletine for symptomatic myotonia (level B); "
            "Annual Holter mandatory (AV block → SCD before severe myopathy); "
            "Congenital DM1 (maternal inheritance only): neonatal hypotonia, respiratory failure, "
            "intellectual disability — if mother has DM1 and neonate hypotonic: IMMEDIATE DMPK testing"
        ),
        "aa": "629 aa",
        "kDa": "~69 kDa",
        "locus": "19q13.32",
        "omim_gene": 605377,
        "omim_disease": 160900,
        "inheritance": "AD — CTG trinucleotide repeat expansion in 3'UTR; anticipation",
        "gene_class": (
            "DMPK encodes myotonic protein kinase, but the disease mechanism in DM1 is not "
            "loss of DMPK kinase function but rather a toxic RNA gain-of-function. "
            "Expanded CUG repeats in DMPK 3'UTR form hairpin RNA structures that sequester "
            "the splicing regulator MBNL1 (muscleblind-like protein 1) into nuclear foci. "
            "MBNL1 sequestration causes mis-splicing of hundreds of downstream targets, "
            "including ClC-1 (chloride channel — myotonia), IR (insulin receptor — insulin resistance), "
            "TNNT2/3 (troponin T — cardiac/skeletal), and many others. "
            "This RNA toxicity mechanism accounts for the multisystem phenotype that extends far "
            "beyond the muscle (lens, heart, brain, endocrine, reproductive systems). "
            "DM1 is the archetypal triplet repeat expansion disease showing anticipation: "
            "maternal transmission carries the highest risk of further expansion and "
            "congenital DM1 (presenting with neonatal hypotonia and respiratory failure) "
            "occurs almost exclusively via maternal inheritance."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("DM1 Classic — CTG 50-999 repeats (adult onset)", 0.55),
            ("DM1 Mild — CTG 50-150 repeats (late onset/cataracts only)", 0.20),
            ("Congenital DM1 — CTG ≥1000 (maternal, neonatal hypotonia)", 0.15),
            ("DM1 — CTG repeat sizing pending (index referral)", 0.10),
        ],
        "onset_range": (0, 65),
        "key_alerts": [
            "DMPK-ANAESTHESIA-ABSOLUTE: propofol TIVA preferred — AVOID succinylcholine + volatile agents",
            "DMPK-ANTICIPATION: repeat expands each generation — warn all family members of earlier onset",
            "DMPK-ANNUAL-HOLTER: AV block → sudden cardiac death — Holter MANDATORY annually",
            "DMPK-MEXILETINE: mexiletine for symptomatic grip myotonia (level B evidence)",
            "CONGENITAL-DM1-MATERNAL: neonatal hypotonia + DM1 mother = IMMEDIATE DMPK sizing",
            "DMPK-CATARACTS: subcapsular — slit-lamp mandatory at diagnosis + 2-yearly",
            "DMPK-GRIP-MYOTONIA: percussion myotonia of thenar eminence — clinical bedside test",
            "CASCADE-Testing: repeat sizing all first-degree relatives — MANDATORY for maternal relatives of CDM1",
        ],
    },
    # ── CNBP — DM2 (Myotonic Dystrophy type 2) ──
    {
        "gene": "CNBP",
        "protein": "CCHC-Type Zinc Finger Nucleic Acid Binding Protein — CCTG Repeat Regulator",
        "alias": (
            "CNBP; OMIM gene 116955; Myotonic Dystrophy 2 OMIM 602668; 3q21.3; 347 aa; ~37 kDa; "
            "CCHC-type zinc finger nucleic acid binding protein; CCTG tetranucleotide repeat expansion "
            "in intron 1 (normal <26 repeats; DM2 ≥75, typically >5000 repeats); "
            "DM2 — Proximal Myotonic Myopathy (PROMM); milder than DM1; "
            "Proximal > distal weakness (hip flexors, thigh, finger flexors) — opposite distribution to DM1; "
            "NO congenital form — unlike DM1; NO intellectual disability typically; "
            "Myotonia often subclinical or elicited only on percussion; "
            "Fluctuating proximal leg pain and stiffness — characteristic and under-recognised; "
            "Cataracts (earlier than age-matched controls); CK mild elevation; "
            "Mexiletine for symptomatic myotonia; annual Holter (AV block risk lower than DM1 but present); "
            "Founder mutation: Central European (German/Polish) — founder haplotype 3q21.3; "
            "Genetic testing: Southern blot or repeat-primed PCR (standard PCR misses large expansions)"
        ),
        "aa": "347 aa",
        "kDa": "~37 kDa",
        "locus": "3q21.3",
        "omim_gene": 116955,
        "omim_disease": 602668,
        "inheritance": "AD — CCTG tetranucleotide repeat expansion in intron 1; mild anticipation",
        "gene_class": (
            "CNBP (formerly ZNF9) encodes a CCHC-type zinc finger protein that binds single-stranded "
            "nucleic acids. In DM2, a massive CCTG repeat expansion (typically >5000 repeats, "
            "compared to <26 normal) in intron 1 of CNBP generates toxic CCUG repeat-containing RNA "
            "that sequesters MBNL1, mirroring the DM1 mechanism but with distinct clinical consequences "
            "due to different tissue expression patterns and repeat structure. "
            "DM2 is clinically distinct from DM1 in several important ways: proximal > distal weakness, "
            "no congenital form (fetal transmission does not cause neonatal disease), "
            "absence of intellectual disability in most patients, milder cardiac conduction disease, "
            "and the characteristic fluctuating proximal leg pain (often misdiagnosed as fibromyalgia "
            "or inflammatory myopathy). Repeat-primed PCR is required for diagnosis as the enormous "
            "expansion exceeds the range detectable by standard PCR; Southern blot remains the reference. "
            "DM2 is particularly prevalent in Central European populations (German, Polish) due to "
            "a founder effect, and should be considered in all adult-onset myotonic myopathy "
            "not explained by DMPK expansion."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("DM2 Classic — CCTG >5000 repeats (Central European)", 0.50),
            ("DM2 — CCTG 75-999 repeats (milder)", 0.25),
            ("DM2 — proximal pain-predominant (PROMM phenotype)", 0.18),
            ("DM2 — repeat-primed PCR pending confirmation", 0.07),
        ],
        "onset_range": (25, 65),
        "key_alerts": [
            "CNBP-PROXIMAL-NOT-DISTAL: DM2 is proximal > distal — opposite to DM1 distal preference",
            "CNBP-NO-CONGENITAL-FORM: DM2 paternal/maternal transmission does NOT cause neonatal disease",
            "CNBP-REPEAT-PRIMED-PCR: standard PCR misses large expansions — use repeat-primed PCR or Southern blot",
            "CNBP-FLUCTUATING-PAIN: proximal leg pain + stiffness — characteristic, misdiagnosed as fibromyalgia",
            "CNBP-CATARACTS: subcapsular cataracts (earlier onset) — slit-lamp at diagnosis",
            "CNBP-MEXILETINE: mexiletine for symptomatic myotonia in DM2",
            "CNBP-ANNUAL-HOLTER: AV block risk lower than DM1 but real — annual cardiac monitoring",
            "CNBP-vs-DM1: DM2 milder — no intellectual disability, no congenital, proximal pattern",
        ],
    },
]


def _rng(seed):
    r = random.Random(seed)
    return r


def _make_cohort(gene_data):
    r = _rng(gene_data["seed"])
    gene = gene_data["gene"]
    patients = []
    etiologies = gene_data["etiologies"]
    etiol_labels = [e[0] for e in etiologies]
    etiol_weights = [e[1] for e in etiologies]
    onset_lo, onset_hi = gene_data["onset_range"]

    for i in range(gene_data["n_patients"]):
        etiol = r.choices(etiol_labels, weights=etiol_weights, k=1)[0]
        age_onset = round(r.uniform(onset_lo, onset_hi), 1)
        age_current = round(age_onset + r.uniform(2, 30), 1)
        sex = r.choice(["M", "F", "M", "M"]) if gene in ("DMD", "EMD", "DMPK") else r.choice(["M", "F"])
        dx_delay = round(r.uniform(0.5, 8.0), 1)

        # Gene-specific flags
        flags = {}
        if gene == "DMD":
            flags["succcinyl_ci"] = True
            flags["glucocorticoids"] = r.random() < 0.88
            flags["exon_skip_eligible"] = r.random() < 0.39
            flags["cardiac_ace_i"] = r.random() < 0.82
            flags["niv_started"] = r.random() < 0.45
            flags["ambulatory"] = r.random() > 0.55
        elif gene == "DYSF":
            flags["western_blot_absent"] = r.random() < 0.91
            flags["monocyte_flow_confirmed"] = r.random() < 0.78
            flags["statin_prescribed_erroneously"] = r.random() < 0.09
            flags["pm_misdiagnosis"] = r.random() < 0.14
            flags["ck_peak"] = int(r.uniform(5000, 100000))
        elif gene == "CAPN3":
            flags["western_blot_reduced"] = r.random() < 0.85
            flags["scapular_winging"] = r.random() < 0.72
            flags["calf_hypertrophy"] = False
            flags["cardiac_spared"] = r.random() > 0.07
            flags["ck_fold"] = round(r.uniform(5, 30), 1)
        elif gene == "LMNA":
            flags["av_block"] = r.random() < 0.68
            flags["icd_implanted"] = r.random() < 0.58
            flags["lvef_reduced"] = r.random() < 0.62
            flags["elbow_contractures"] = r.random() < 0.78
            flags["annual_holter"] = r.random() < 0.91
        elif gene == "EMD":
            flags["emerin_absent_immunostaining"] = r.random() < 0.94
            flags["av_block"] = r.random() < 0.72
            flags["pacemaker_icd"] = r.random() < 0.61
            flags["elbow_contractures"] = r.random() < 0.85
            flags["female_carrier_cardiac"] = sex == "F" and r.random() < 0.15
        elif gene == "SGCA":
            flags["all_4_sarcoglycans_tested"] = r.random() < 0.72
            flags["calf_hypertrophy"] = r.random() < 0.74
            flags["cardiac_echo_annual"] = r.random() < 0.81
            flags["ck_fold"] = round(r.uniform(10, 150), 1)
            flags["founder_Arg77Cys"] = r.random() < 0.28
        elif gene == "DMPK":
            flags["repeat_size"] = int(r.uniform(50, 2000))
            flags["anticipation_family"] = r.random() < 0.78
            flags["myotonia_clinical"] = r.random() < 0.88
            flags["mexiletine_prescribed"] = r.random() < 0.64
            flags["anaesthesia_flagged"] = r.random() < 0.71
            flags["annual_holter"] = r.random() < 0.84
            flags["cataracts"] = r.random() < 0.76
        elif gene == "CNBP":
            flags["repeat_sized_correctly"] = r.random() < 0.78
            flags["proximal_pain"] = r.random() < 0.82
            flags["myotonia_subclinical"] = r.random() < 0.58
            flags["cataracts"] = r.random() < 0.65
            flags["mexiletine_prescribed"] = r.random() < 0.44
            flags["misdiagnosed_fibromyalgia"] = r.random() < 0.19

        patients.append({
            "pid": f"{gene}-{i+1:03d}",
            "gene": gene,
            "etiology": etiol,
            "sex": sex,
            "age_onset": age_onset,
            "age_current": age_current,
            "dx_delay_y": dx_delay,
            **flags,
        })
    return patients


def get_overview():
    all_patients = []
    gene_summaries = []

    for gd in MUSCULAR_DYSTROPHY_GENES:
        pts = _make_cohort(gd)
        all_patients.extend(pts)

        g = gd["gene"]
        gene_summaries.append({
            "gene": g,
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "etiologies": [e[0] for e in gd["etiologies"]],
            "key_alerts": gd["key_alerts"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
        })

    n = len(all_patients)
    dmd_pts = [p for p in all_patients if p["gene"] == "DMD"]
    dysf_pts = [p for p in all_patients if p["gene"] == "DYSF"]
    capn3_pts = [p for p in all_patients if p["gene"] == "CAPN3"]
    lmna_pts = [p for p in all_patients if p["gene"] == "LMNA"]
    emd_pts = [p for p in all_patients if p["gene"] == "EMD"]
    sgca_pts = [p for p in all_patients if p["gene"] == "SGCA"]
    dmpk_pts = [p for p in all_patients if p["gene"] == "DMPK"]
    cnbp_pts = [p for p in all_patients if p["gene"] == "CNBP"]

    def pct(lst, key, val=True):
        if not lst:
            return 0.0
        return round(100 * sum(1 for p in lst if p.get(key) == val) / len(lst), 1)

    def pct_thresh(lst, key, thresh):
        if not lst:
            return 0.0
        return round(100 * sum(1 for p in lst if p.get(key, 0) >= thresh) / len(lst), 1)

    agg = {
        "total_patients": n,
        "total_genes": 8,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        # DMD
        "dmd_succcinyl_ci_pct": 100.0,
        "dmd_glucocorticoids_pct": pct(dmd_pts, "glucocorticoids"),
        "dmd_exon_skip_eligible_pct": pct(dmd_pts, "exon_skip_eligible"),
        "dmd_cardiac_ace_i_pct": pct(dmd_pts, "cardiac_ace_i"),
        "dmd_niv_pct": pct(dmd_pts, "niv_started"),
        "dmd_ambulatory_pct": pct(dmd_pts, "ambulatory"),
        # DYSF
        "dysf_western_blot_absent_pct": pct(dysf_pts, "western_blot_absent"),
        "dysf_statin_prescribed_erroneously_pct": pct(dysf_pts, "statin_prescribed_erroneously"),
        "dysf_pm_misdiagnosis_pct": pct(dysf_pts, "pm_misdiagnosis"),
        # CAPN3
        "capn3_western_blot_reduced_pct": pct(capn3_pts, "western_blot_reduced"),
        "capn3_scapular_winging_pct": pct(capn3_pts, "scapular_winging"),
        "capn3_cardiac_spared_pct": pct(capn3_pts, "cardiac_spared"),
        # LMNA
        "lmna_av_block_pct": pct(lmna_pts, "av_block"),
        "lmna_icd_implanted_pct": pct(lmna_pts, "icd_implanted"),
        "lmna_lvef_reduced_pct": pct(lmna_pts, "lvef_reduced"),
        "lmna_elbow_contractures_pct": pct(lmna_pts, "elbow_contractures"),
        "lmna_annual_holter_pct": pct(lmna_pts, "annual_holter"),
        # EMD
        "emd_emerin_absent_pct": pct(emd_pts, "emerin_absent_immunostaining"),
        "emd_av_block_pct": pct(emd_pts, "av_block"),
        "emd_pacemaker_icd_pct": pct(emd_pts, "pacemaker_icd"),
        "emd_elbow_contractures_pct": pct(emd_pts, "elbow_contractures"),
        # SGCA
        "sgca_all_4_tested_pct": pct(sgca_pts, "all_4_sarcoglycans_tested"),
        "sgca_calf_hypertrophy_pct": pct(sgca_pts, "calf_hypertrophy"),
        "sgca_founder_Arg77Cys_pct": pct(sgca_pts, "founder_Arg77Cys"),
        "sgca_cardiac_echo_pct": pct(sgca_pts, "cardiac_echo_annual"),
        # DMPK
        "dmpk_myotonia_pct": pct(dmpk_pts, "myotonia_clinical"),
        "dmpk_anaesthesia_flagged_pct": pct(dmpk_pts, "anaesthesia_flagged"),
        "dmpk_mexiletine_pct": pct(dmpk_pts, "mexiletine_prescribed"),
        "dmpk_annual_holter_pct": pct(dmpk_pts, "annual_holter"),
        "dmpk_cataracts_pct": pct(dmpk_pts, "cataracts"),
        # CNBP
        "cnbp_proximal_pain_pct": pct(cnbp_pts, "proximal_pain"),
        "cnbp_misdiagnosed_fibromyalgia_pct": pct(cnbp_pts, "misdiagnosed_fibromyalgia"),
        "cnbp_repeat_sized_correctly_pct": pct(cnbp_pts, "repeat_sized_correctly"),
        "cnbp_cataracts_pct": pct(cnbp_pts, "cataracts"),
        # Cross-gene
        "any_succcinyl_ci_pct": round(100 * sum(
            1 for p in all_patients if p.get("succcinyl_ci") or p.get("gene") == "DMPK"
        ) / n, 1),
        "any_icd_pacemaker_pct": round(100 * sum(
            1 for p in all_patients if p.get("icd_implanted") or p.get("pacemaker_icd")
        ) / n, 1),
        "any_contractures_pct": round(100 * sum(
            1 for p in all_patients if p.get("elbow_contractures")
        ) / n, 1),
    }

    top_alerts = [
        alert for gd in MUSCULAR_DYSTROPHY_GENES for alert in gd["key_alerts"]
    ]

    return {
        "title": "Hereditary-Muscular-Dystrophy-Atlas — Complete 8-Gene Hereditary Muscular Dystrophy Atlas",
        "subtitle": (
            "DMD · DYSF · CAPN3 · LMNA · EMD · SGCA · DMPK · CNBP — "
            "320 patients (8×40, seeds 1478–1485)"
        ),
        "aggregate_stats": agg,
        "genes": gene_summaries,
        "top_alerts": top_alerts,
    }


def get_breakdown():
    breakdown = []
    for gd in MUSCULAR_DYSTROPHY_GENES:
        pts = _make_cohort(gd)
        etiol_counts = {}
        for p in pts:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        sex_dist = {"M": sum(1 for p in pts if p["sex"] == "M"),
                    "F": sum(1 for p in pts if p["sex"] == "F")}
        mean_onset = round(sum(p["age_onset"] for p in pts) / len(pts), 1)
        mean_delay = round(sum(p["dx_delay_y"] for p in pts) / len(pts), 1)

        breakdown.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "mean_onset_y": mean_onset,
            "mean_dx_delay_y": mean_delay,
            "sex_distribution": sex_dist,
            "etiology_counts": etiol_counts,
            "key_alerts": gd["key_alerts"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
            "patients": pts,
        })
    return {"breakdown": breakdown}


def get_definitions():
    return {
        "definitions": [
            {
                "term": "Reading-Frame Rule (DMD vs BMD)",
                "definition": (
                    "The Koenig reading-frame rule predicts phenotype from DMD deletion/duplication with ~92% accuracy: "
                    "out-of-frame mutations (disrupt open reading frame) → absent dystrophin → Duchenne (severe); "
                    "in-frame mutations (preserve ORF, allow truncated protein) → reduced/truncated dystrophin → "
                    "Becker (milder). Exon-skipping antisense oligonucleotide therapy converts out-of-frame to "
                    "in-frame by skipping the flanking exon, converting Duchenne to a Becker-like phenotype."
                ),
            },
            {
                "term": "Succinylcholine Absolute Contraindication — Dystrophinopathy",
                "definition": (
                    "Succinylcholine (suxamethonium) causes massive potassium efflux through upregulated "
                    "extra-junctional acetylcholine receptors in denervated or dystrophic muscle. "
                    "In dystrophinopathy (DMD/BMD), this leads to potentially lethal hyperkalemia and "
                    "cardiac arrest. All anaesthetic charts for known or suspected dystrophinopathy MUST "
                    "be flagged to avoid succinylcholine. DMPK (DM1) also carries a prolonged paralysis "
                    "risk with succinylcholine. Propofol TIVA is the preferred induction method."
                ),
            },
            {
                "term": "Dysferlin Western Blot — LGMD2B Diagnostic Standard",
                "definition": (
                    "Absent dysferlin protein on Western blot of muscle biopsy lysate is the gold-standard "
                    "diagnostic test for dysferlinopathy (LGMD2B). Monocyte flow cytometry (peripheral blood) "
                    "is a rapid, non-invasive alternative that also shows absent/reduced dysferlin and can "
                    "distinguish primary dysferlinopathy from secondary reduction. The inflammatory infiltrate "
                    "visible on light microscopy in LGMD2B closely mimics polymyositis — Western blot "
                    "prevents erroneous immunosuppression, which is harmful in dysferlinopathy."
                ),
            },
            {
                "term": "Secondary Sarcoglycan Loss — Sarcoglycanopathy Panel Rule",
                "definition": (
                    "Primary pathogenic mutation in any one sarcoglycan subunit (SGCA, SGCB, SGCC, or SGCD) "
                    "destabilises the entire heterotetrameric sarcoglycan complex, causing secondary "
                    "immunohistochemical loss of all four subunits. Therefore, absent staining for all four "
                    "sarcoglycans on muscle immunohistochemistry does NOT identify which gene is primarily "
                    "mutated — genetic panel testing of all four genes (SGCA/B/C/D) is MANDATORY to "
                    "identify the causative gene and inform accurate recurrence risk counselling."
                ),
            },
            {
                "term": "Laminopathy Arrhythmia Risk — ICD Criteria (HRS/EHRA 2019)",
                "definition": (
                    "LMNA pathogenic variants carry a disproportionately high risk of malignant ventricular "
                    "arrhythmia and AV block relative to the degree of left ventricular dysfunction. "
                    "The 2019 HRS/EHRA consensus recommends prophylactic ICD in LMNA mutation carriers "
                    "with ≥2 of: (1) non-sustained VT on Holter, (2) LVEF <45%, (3) male sex, "
                    "(4) non-missense mutation (deletion/insertion/frameshift/splice). "
                    "This threshold is lower than standard ICD criteria (LVEF <35%) because LMNA-associated "
                    "SCD can occur well before severe LV dysfunction develops."
                ),
            },
            {
                "term": "Emerin Immunostaining — EDMD1 Diagnostic",
                "definition": (
                    "Emerin is expressed in the inner nuclear membrane of all nucleated cells. "
                    "In X-linked EDMD1, pathogenic EMD mutations result in absent or markedly reduced "
                    "emerin protein, detectable by immunofluorescence or immunohistochemistry on any "
                    "nucleated tissue — most conveniently skin punch biopsy fibroblasts or peripheral "
                    "blood lymphocytes, avoiding invasive muscle biopsy. "
                    "Nuclear membrane pattern (vs. cytoplasmic) must be confirmed with co-staining."
                ),
            },
            {
                "term": "DM1 Anticipation — Congenital Risk",
                "definition": (
                    "Myotonic Dystrophy type 1 (DM1) shows marked anticipation: CTG repeat length expands "
                    "with successive generations, particularly through maternal transmission, causing "
                    "progressively earlier onset and more severe disease. Congenital DM1 (CDM1), the most "
                    "severe form (CTG ≥1000), occurs almost exclusively through maternal inheritance and "
                    "presents with neonatal hypotonia, respiratory failure, and intellectual disability. "
                    "Paternal transmission rarely produces CDM1. Any neonate with hypotonia born to a "
                    "mother with DM1 requires immediate DMPK repeat sizing."
                ),
            },
            {
                "term": "DM1 Anaesthesia Protocol",
                "definition": (
                    "DM1 patients carry multiple anaesthetic risks: (1) succinylcholine causes hyperkalaemia "
                    "and prolonged neuromuscular blockade — ABSOLUTE CONTRAINDICATION; "
                    "(2) volatile anaesthetic agents (sevoflurane, isoflurane) can trigger myotonia and "
                    "prolong recovery — propofol TIVA (total intravenous anaesthesia) is preferred; "
                    "(3) respiratory muscles are weakened — post-operative NIV planning mandatory; "
                    "(4) cardiac conduction monitoring required perioperatively — anaesthetists must review "
                    "ECG for AV block before any procedure. All DM1 patients must carry an anaesthesia "
                    "alert card and any planned surgery must be pre-discussed with neuromuscular team."
                ),
            },
            {
                "term": "DM2 — Repeat-Primed PCR Requirement",
                "definition": (
                    "The CCTG expansion in DM2 typically exceeds 5000 tetranucleotide repeats — far larger "
                    "than the range detectable by standard fluorescent PCR (which reliably detects up to ~200 bp). "
                    "Standard PCR in DM2 therefore shows only the normal allele and gives a false-negative result. "
                    "Repeat-primed PCR (RP-PCR), which generates a characteristic stutter ladder pattern "
                    "at the repeat, or Southern blot are required to detect DM2 expansions. "
                    "Clinicians ordering 'myotonic dystrophy panel' must confirm the laboratory includes "
                    "RP-PCR for CNBP as well as DMPK."
                ),
            },
            {
                "term": "LGMD Classification (2018 ENMC Consensus)",
                "definition": (
                    "The 2018 European Neuromuscular Centre consensus renamed LGMD subtypes from "
                    "LGMD1 (AD) + LGMD2 (AR) + letter suffixes to R (recessive) or D (dominant) "
                    "numbered by gene discovery order: LGMD2A→R1 (CAPN3), LGMD2B→R2 (DYSF), "
                    "LGMD2D→R3 (SGCA), LGMD1B→D2 (LMNA). Both nomenclatures remain in clinical use. "
                    "This atlas uses the legacy LGMD2A/2B/2D nomenclature alongside OMIM disease numbers "
                    "as the primary identifiers, with the 2018 R/D classification noted parenthetically."
                ),
            },
            {
                "term": "Cascade Testing — Hereditary Muscular Dystrophy",
                "definition": (
                    "First-degree relatives of any hereditary muscular dystrophy proband should be offered: "
                    "(1) genetic counselling; (2) targeted genetic testing (or full panel if index mutation "
                    "not yet identified); (3) CK measurement as a rapid pre-test screen. "
                    "For X-linked disorders (DMD, EMD): all maternal relatives (sisters, maternal aunts) "
                    "need carrier CK + genetic testing. For LMNA/DMPK: all first-degree relatives need "
                    "annual cardiac monitoring once carrier status confirmed. "
                    "For DMPK: maternal relatives of any CDM1 proband need URGENT repeat sizing."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(json.dumps(ov["aggregate_stats"], indent=2))
    print(f"\nTop alerts ({len(ov['top_alerts'])}):")
    for a in ov["top_alerts"]:
        print(f"  • {a}")
