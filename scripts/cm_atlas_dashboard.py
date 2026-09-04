#!/usr/bin/env python3
"""CM-Atlas — Complete 8-Gene Congenital Myopathy Atlas
RYR1     (Ryanodine Receptor 1; AR/AD; 5038 aa; 19q13.2; Central Core Disease / Multiminicore / MALIGNANT HYPERTHERMIA — VOLATILE AGENTS + SUCCINYLCHOLINE ABSOLUTELY AVOID) ·
NEB      (Nebulin; AR; 8537 aa; 2q23.3; Nemaline Myopathy 2; largest sarcomeric gene; MLPA mandatory for exon deletions) ·
ACTA1    (Actin α1 Skeletal Muscle; AD de novo/AR; 377 aa; 1q42.13; Nemaline Myopathy 3; MULTIPLE HISTOLOGICAL PATTERNS — nemaline/intranuclear rods/actin aggregate/CFTD) ·
TPM2     (Tropomyosin β; AD; 284 aa; 9p13.3; Nemaline/Cap disease/CFTD; often milder; arthrogryposis possible) ·
MYH7     (Myosin Heavy Chain 7; AD; 1935 aa; 14q11.2; Laing distal/Myosin storage; ALLELIC HCM + DCM — cardiac mandatory) ·
SELENON  (Selenoprotein N; AR; 590 aa; 1p36.11; RSMD1 Rigid Spine; RESPIRATORY DISPROPORTIONATE — NIV MANDATORY EARLY before severe limb weakness) ·
MTM1     (Myotubularin 1; XLR; 603 aa; Xq28; XLMTM X-Linked Myotubular Myopathy; neonatal severe; ventilator-dependent from birth in most) ·
DNM2     (Dynamin 2; AD; 870 aa; 19p13.2; Autosomal Dominant Centronuclear Myopathy; ptosis + ophthalmoparesis hallmark; CMT-allelic)
320-patient aggregate cohort (8 × 40, seeds 1038–1045)

Congenital Myopathy — Key Neurological Principles:
  - DEFINITION: Clinically and genetically heterogeneous group of muscle diseases presenting at birth or
    in early infancy with hypotonia ("floppy infant"), weakness, and characteristic biopsy findings.
    Subdivided by histological pattern: Nemaline Myopathy (rods), Core Myopathy (central/minicores),
    Centronuclear/Myotubular Myopathy (central nuclei), Congenital Fiber Type Disproportion (CFTD).
    Same gene can cause multiple histological patterns (RYR1, ACTA1).
  - CRITICAL TREATMENT RULES:
    (1) RYR1 — MALIGNANT HYPERTHERMIA: ABSOLUTELY AVOID volatile anaesthetic agents (halothane,
        isoflurane, sevoflurane, desflurane) AND succinylcholine. Use propofol + N2O + regional/
        neuraxial anaesthesia. MH emergency: dantrolene IV immediately (2.5 mg/kg bolus, repeat
        q5–10 min to 10 mg/kg). MH alert bracelet + crisis card for all RYR1 patients and families.
    (2) SELENON/RSMD1 — NIV MANDATORY EARLY: Respiratory failure is disproportionate to limb
        weakness; FVC can be <50% predicted while patient still walking. Do NOT wait for severe
        respiratory symptoms. Annual FVC + overnight pulse oximetry from diagnosis; start NIV when
        FVC <60% predicted OR nocturnal desaturation confirmed.
    (3) MTM1/XLMTM — VENTILATOR FROM BIRTH: Most affected males require ventilator support from
        birth or within weeks. Many never achieve unsupported breathing. Aspiration pneumonia is
        the leading cause of early mortality. Percutaneous gastrostomy + ventilator support are
        life-sustaining interventions, not optional.
    (4) ALL CONGENITAL MYOPATHY: Succinylcholine contraindicated (rhabdomyolysis/hyperkalaemia
        risk regardless of MH susceptibility; worsened in MTM1/XLMTM with existing ventilatory failure).

COHORT: 8 × 40 = 320 patient slots (seeds 1038–1045; gene-specific seeds)
"""

import random

SEED_BASE = 1038

CM_GENES = [
    # ── RYR1 — Ryanodine Receptor 1 ───────────────────────────────────────
    {
        "gene": "RYR1", "protein": "Ryanodine Receptor 1",
        "alias": "RYR1; OMIM gene 180901; AR: Central Core Disease #117000; AD: Malignant Hyperthermia Susceptibility 1 #145600; 19q13.2; most common congenital myopathy gene",
        "aa": "5038 aa", "kDa": "565 kDa",
        "gene_class": (
            "RYR1 encodes the ryanodine receptor type 1, a massive homotetrameric calcium-release channel "
            "(each subunit 5038 aa, 565 kDa) located on the sarcoplasmic reticulum (SR) membrane of "
            "skeletal muscle. It forms the SR-junctional foot structure that physically contacts DHPR "
            "(Cav1.1) at the T-tubule, coupling excitation to calcium release (ECRC). "
            "MECHANISM OF DISEASE: AR biallelic LOF mutations → reduced or absent RYR1 protein → "
            "deficient calcium release → muscle weakness + characteristic core lesions on biopsy "
            "(regions of absent oxidative enzyme activity ± mitochondria on electron microscopy). "
            "AD GOF mutations → abnormally prolonged/excessive calcium release in response to "
            "volatile anaesthetic agents or succinylcholine → MALIGNANT HYPERTHERMIA (MH). "
            "MUTATION SPECTRUM: >600 pathogenic variants. AR mutations throughout the gene; "
            "MH-associated mutations cluster in 3 hot-spots (MH1: aa 35–614; MH2: aa 2163–2458; "
            "MH3/CCD: aa 4864–4973, C-terminal channel domain). "
            "RYR1 is the MOST COMMON congenital myopathy gene; also most common MH susceptibility gene."
        ),
        "cm_group": "SR Calcium Release Channel — Core Myopathy",
        "cm_type": "Central Core Disease (CCD) / Multiminicore Disease (MmD) / MH Susceptibility",
        "locus": "19q13.2", "omim_gene": 180901, "omim_disease": 117000,
        "inheritance": (
            "Bimodal: AR (biallelic, CCD/MmD/myopathy) + AD (dominant-negative/GOF, MH susceptibility + "
            "milder myopathy). De novo AD possible. X-linked excluded. "
            "AR Central Core Disease (CCD): OMIM #117000 (confusingly listed as AD historically; "
            "now reclassified — majority of severe CCD is AR). "
            "AD Malignant Hyperthermia Susceptibility-1 (MHS1): OMIM #145600. "
            "Some RYR1 carriers (heterozygous for mild variant) may have subclinical MH risk."
        ),
        "phenotype": (
            "ONSET: Neonatal-infantile (AR severe) or childhood-adolescent (AR/AD milder). "
            "Hypotonia (floppy infant) + proximal weakness + delayed motor milestones. "
            "FACIAL WEAKNESS: mild ptosis and facial weakness possible. "
            "OPHTHALMOPARESIS: in multiminicore subtype (MmD); a key distinguishing feature from CCD. "
            "RESPIRATORY: variable — severe AR forms require NIV; AD forms often walk without NIV. "
            "SCOLIOSIS: progressive; surgical management often needed. "
            "JOINT HYPERMOBILITY/CONTRACTURES: hip dislocation, club foot, kyphoscoliosis. "
            "MALIGNANT HYPERTHERMIA CRISIS: can be the FIRST presentation in a previously asymptomatic "
            "carrier — explosive perioperative hyperthermia + rigidity + rhabdomyolysis; potentially lethal. "
            "CK: normal to mildly elevated (unlike LGMD)."
        ),
        "disease": (
            "RYR1-related myopathy — Central Core Disease (CCD) / Multiminicore Disease (MmD). "
            "Most common congenital myopathy gene. Diagnosis: muscle biopsy (cores on NADH/COX) + "
            "RYR1 gene sequencing + in vitro contracture test (IVCT) for MH assessment. "
            "No approved disease-modifying therapy. Gene therapy trials in early stages. "
            "KEY MANAGEMENT: MH-safe anaesthesia protocol; dantrolene stocked perioperatively; "
            "physiotherapy for weakness/scoliosis; NIV if respiratory compromise."
        ),
        "treatment_options": [
            "MH EMERGENCY: Dantrolene IV 2.5 mg/kg bolus → repeat q5-10 min → max 10 mg/kg; "
            "STOP volatile agents; hyperventilate 100% O2; correct hyperthermia + acidosis + hyperkalaemia",
            "ABSOLUTELY AVOID volatile anaesthetic agents (halothane, isoflurane, sevoflurane, desflurane) "
            "AND succinylcholine — use propofol + N2O + regional/neuraxial anaesthesia (MH-safe protocol)",
            "MH alert bracelet + anaesthetic crisis card for patient and all first-degree relatives",
            "Physiotherapy: strength maintenance; stretching; hydrotherapy",
            "Scoliosis monitoring + surgical correction (Cobb >40°)",
            "Annual FVC + overnight pulse oximetry; NIV if respiratory compromise",
            "Cardiac echo (MYH7-allelic concern excluded; RYR1 cardiac involvement rare but possible in MmD)",
            "Genetic counselling: AR 25% recurrence; AD family cascade MH testing (IVCT gold standard)",
        ],
        "key_ddx": [
            "Central Core Disease: SEPN1/SELENON (but SELENON has rigid spine; no MH risk)",
            "Multiminicore Disease: SELENON (also MmD pattern; no MH), MYH7 (myosin storage myopathy)",
            "Nemaline Myopathy: RYR1 rarely causes nemaline pattern; NEB, ACTA1 first",
            "Malignant Hyperthermia: CACNA1S (MHS5), STAC3, KCNJ2 (Andersen-Tawil); confirm by IVCT + genotype",
            "King-Denborough syndrome: RYR1 (rare — MH + dysmorphic features + CK elevation)",
        ],
        "onset_range_y": (0, 15),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "contractures": True,
        "facial_weakness": True,
        "ophthalmoparesis": False,
        "rigid_spine": False,
        "mh_risk": True,
        "neonatal_severe": False,
        "ad_inheritance": True,
        "very_high_ck": False,
        "ptosis": True,
    },

    # ── NEB — Nebulin ─────────────────────────────────────────────────────
    {
        "gene": "NEB", "protein": "Nebulin",
        "alias": "NEB; OMIM gene 161650; Nemaline Myopathy 2 #256030; 2q23.3; largest sarcomeric gene (8537 aa); AR; most common AR nemaline myopathy",
        "aa": "8537 aa", "kDa": "800 kDa",
        "gene_class": (
            "NEB encodes nebulin (8537 aa, ~800 kDa; 183 exons), the largest structural muscle protein "
            "and the longest known sarcomeric ruler. Each nebulin molecule spans the entire thin filament "
            "from the Z-disc to the pointed end, setting thin filament length and regulating actin-tropomyosin "
            "assembly. MECHANISM OF DISEASE: AR biallelic LOF mutations → absent or truncated nebulin → "
            "shorter thin filaments → impaired force generation → NEMALINE RODS (electron-dense Z-disc "
            "material accumulation visible on Gomori trichrome and EM as nemaline rods). "
            "NEB IS THE LARGEST GENE IN HUMAN GENOME targeted for disease: 183 exons including "
            "a large (~8 kb) triplicated region (exons 82–105) that makes PCR amplification difficult. "
            "MUTATION SPECTRUM: hundreds of pathogenic variants; missense, nonsense, frameshift, "
            "splice-site, and LARGE DELETIONS (especially exon 55, triplicated region). "
            "MLPA MANDATORY: standard sequencing misses large deletions in ~20% of NEB cases. "
            "NEB is the most common AR nemaline myopathy gene, accounting for ~50% of all nemaline cases."
        ),
        "cm_group": "Thin Filament — Nemaline Myopathy (AR, most common)",
        "cm_type": "Nemaline Myopathy 2 (NEM2) — Nebulin Deficiency",
        "locus": "2q23.3", "omim_gene": 161650, "omim_disease": 256030,
        "inheritance": (
            "Autosomal Recessive (AR). Biallelic LOF. Compound heterozygous common (one missense + "
            "one null, or two missense). LARGE DELETIONS: exon 55 deletion is the most common founder "
            "variant in Ashkenazi Jewish population (~1:108 carriers). Triplicated region deletions: "
            "missed by standard sequencing — MLPA or gene dosage analysis MANDATORY."
        ),
        "phenotype": (
            "ONSET: Neonatal (severe congenital) to childhood (typical). "
            "Neonatal: hypotonia, feeding difficulties, respiratory distress at birth. "
            "Typical: proximal + axial weakness (head lag; spine rigidity); delayed walking. "
            "FACIAL WEAKNESS: prominent; open-mouth facies; myopathic facies; high-arched palate. "
            "RESPIRATORY: significant — many require NIV in childhood/adolescence; tracheostomy in severe cases. "
            "SCOLIOSIS: progressive; early surgical consideration. "
            "NON-AMBULATORY: ~30% of typical NEB cases lose ambulation by adulthood. "
            "CARDIAC: usually spared (unlike TPM2/MYH7 allelic variants). "
            "CK: normal to mildly elevated."
        ),
        "disease": (
            "Nemaline Myopathy 2 (NEM2) — NEB deficiency. Most common AR nemaline myopathy (~50% of NEM). "
            "Diagnosis: muscle biopsy (nemaline rods on Gomori trichrome + EM; type I fibre predominance); "
            "NEB panel sequencing + MLPA. "
            "Exon 55 deletion screen should be performed FIRST in Ashkenazi Jewish patients. "
            "No approved disease-modifying therapy. Supportive: physiotherapy, NIV, scoliosis surgery. "
            "Troponin activator tirasemtiv (Phase 2) and fast-troponin activators in trials."
        ),
        "treatment_options": [
            "MLPA/gene dosage mandatory: standard sequencing misses ~20% of NEB large deletions",
            "Exon 55 deletion screen first in Ashkenazi Jewish patients (founder allele ~1:108 carriers)",
            "Respiratory monitoring: annual FVC; overnight oximetry; NIV when FVC <50% predicted or nocturnal desaturation",
            "Feeding support: nasogastric or gastrostomy if swallowing failure; high-calorie diet",
            "Physiotherapy: maintenance of strength; stretching; hydrotherapy; avoid immobility",
            "Scoliosis: monitor Cobb angle; surgical correction if >40° or rapid progression",
            "Troponin activators (tirasemtiv/CK-2066260): Phase 2 trials in nemaline myopathy (investigational)",
            "Genetic counselling: AR 25% recurrence; exon 55 carrier testing in Ashkenazi families",
        ],
        "key_ddx": [
            "Nemaline Myopathy: ACTA1 (intranuclear rods + actin aggregate — severe), TPM2/TPM3, TNNT1 (Amish NEM), KBTBD13, KLHL40/41",
            "Centronuclear Myopathy: MTM1, DNM2, BIN1 (central nuclei, not rods — biopsy distinction)",
            "Spinal Muscular Atrophy (SMA): SMN1 deletion (lower motor neuron; normal biopsy structure; genetic)",
            "Congenital Myotonic Dystrophy (DM1): CTG repeat; maternal; facial + distal; myotonia on EMG",
            "Pompe Disease (GSD II): acid alpha-glucosidase deficiency; lysosomal vacuoles on biopsy (not nemaline rods)",
        ],
        "onset_range_y": (0, 10),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "contractures": True,
        "facial_weakness": True,
        "ophthalmoparesis": False,
        "rigid_spine": False,
        "mh_risk": False,
        "neonatal_severe": True,
        "ad_inheritance": False,
        "very_high_ck": False,
        "ptosis": False,
    },

    # ── ACTA1 — Actin Alpha 1 Skeletal Muscle ─────────────────────────────
    {
        "gene": "ACTA1", "protein": "Actin Alpha-1 Skeletal Muscle",
        "alias": "ACTA1; OMIM gene 102610; Nemaline Myopathy 3 #161800; 1q42.13; AD de novo (most); AR; MULTIPLE HISTOLOGICAL PATTERNS from one gene",
        "aa": "377 aa", "kDa": "42 kDa",
        "gene_class": (
            "ACTA1 encodes alpha-skeletal actin (377 aa, 42 kDa), the predominant actin isoform in adult "
            "skeletal muscle and the principal component of the thin filament. Alpha-skeletal actin "
            "interacts with tropomyosin, troponin, nebulin, and myosin heads to generate force. "
            "MECHANISM OF DISEASE: Diverse — ACTA1 mutations cause MULTIPLE distinct histological patterns "
            "depending on mutation type and location: "
            "(1) Nemaline rods (Z-disc material accumulation) — most common. "
            "(2) Intranuclear rods — dominant de novo mutations → rods inside myonuclei → severe/lethal. "
            "(3) Actin aggregate myopathy — abundant intracytoplasmic actin filament accumulations → severe. "
            "(4) Congenital Fibre Type Disproportion (CFTD) — type I fibres ≥12% smaller than type II → milder. "
            "(5) Core-rod myopathy — central cores + nemaline rods combined. "
            "AD DE NOVO: ~90% of ACTA1 nemaline myopathy is de novo dominant — de novo mutation screening "
            "mandatory in simplex cases with no family history."
        ),
        "cm_group": "Thin Filament — Nemaline Myopathy (AD de novo, AR)",
        "cm_type": "Nemaline Myopathy 3 (NEM3) — Actin Alpha-1 Deficiency",
        "locus": "1q42.13", "omim_gene": 102610, "omim_disease": 161800,
        "inheritance": (
            "Bimodal: AD (de novo dominant, ~90% of ACTA1 nemaline cases) + AR (recessive, rarer). "
            "DE NOVO: most simplex severe cases are new dominant mutations — check parental blood samples. "
            "AR: both parents carriers; milder phenotype common. "
            "Intranuclear rod myopathy: almost always de novo dominant (most severe, early lethal). "
            "Actin aggregate myopathy: AR or de novo; very severe."
        ),
        "phenotype": (
            "ONSET: Predominantly neonatal (severe forms) — 50% die in first year without ventilator support. "
            "SPECTRUM: extreme — from lethal neonatal (intranuclear rod myopathy) to ambulatory adults (CFTD). "
            "Severe forms: severe neonatal hypotonia, respiratory failure, areflexia, arthrogryposis. "
            "INTRANUCLEAR ROD MYOPATHY: most severe — rods in myonuclei; often fatal neonatal period. "
            "ACTIN AGGREGATE MYOPATHY: severe — abundant cytoplasmic actin inclusions; early death or vent. "
            "NEMALINE TYPICAL (NEM3): infantile hypotonia, facial weakness, respiratory compromise, scoliosis. "
            "CFTD (mildest): delayed motor milestones, proximal weakness, ambulation usually maintained. "
            "FACIAL WEAKNESS: prominent in all forms. RESPIRATORY: severe in most forms."
        ),
        "disease": (
            "Nemaline Myopathy 3 / ACTA1-related myopathy. Second most common congenital myopathy gene (~15–20%). "
            "Diagnosis: muscle biopsy (Gomori trichrome + EM mandatory to distinguish rod subtypes) + ACTA1 sequencing. "
            "De novo dominant: ~90% — parental testing important (recurrence risk near 0 vs 25% if AR). "
            "No approved therapy. Ventilator support life-sustaining in severe forms. Supportive care."
        ),
        "treatment_options": [
            "Parental ACTA1 testing MANDATORY: de novo dominant (~90%) vs AR (25% recurrence risk) distinction critical for counselling",
            "Muscle biopsy EM mandatory: Gomori alone cannot distinguish intranuclear rods (most severe) from cytoplasmic rods",
            "Ventilatory support: neonatal — endotracheal intubation → tracheostomy; chronic — NIV from infancy in severe forms",
            "Gastrostomy: feeding failure common; percutaneous gastrostomy prevents aspiration + improves nutrition",
            "Physiotherapy: passive range of motion; prevention of contractures; hydrotherapy",
            "Scoliosis: early spinal support + surgical correction",
            "Genetic counselling: distinguish de novo AD (very low recurrence in parents) vs AR (25% sibling risk)",
        ],
        "key_ddx": [
            "Nemaline Myopathy: NEB (AR, milder, rods only, no intranuclear), TPM2/TPM3 (milder),  TNNT1 (Amish NEM)",
            "Intranuclear rod myopathy: ACTA1 dominant — confirm by EM (nuclei contain rods); unique ACTA1 de novo",
            "Actin aggregate myopathy: ACTA1 AR/de novo — confirm by EM (abundant cytoplasmic actin filaments)",
            "SMA type 1 (Werdnig-Hoffmann): SMN1 deletion; lower motor neuron; normal biopsy architecture (no rods)",
            "Myotubular Myopathy (MTM1): central nuclei on biopsy (not rods); X-linked males; MTM1 gene",
        ],
        "onset_range_y": (0, 5),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "contractures": True,
        "facial_weakness": True,
        "ophthalmoparesis": False,
        "rigid_spine": False,
        "mh_risk": False,
        "neonatal_severe": True,
        "ad_inheritance": True,
        "very_high_ck": False,
        "ptosis": False,
    },

    # ── TPM2 — Tropomyosin Beta ────────────────────────────────────────────
    {
        "gene": "TPM2", "protein": "Tropomyosin Beta (β-tropomyosin)",
        "alias": "TPM2; OMIM gene 190990; Nemaline Myopathy 4 #609285; 9p13.3; AD; Cap disease; CFTD; nemaline; often milder spectrum; arthrogryposis possible",
        "aa": "284 aa", "kDa": "33 kDa",
        "gene_class": (
            "TPM2 encodes β-tropomyosin (284 aa, 33 kDa), which forms αβ-heterodimers with TPM1 "
            "(α-tropomyosin) on the thin filament. Tropomyosin occupies the grooves of actin filaments "
            "and regulates calcium-dependent actomyosin interaction in concert with troponin. "
            "MECHANISM OF DISEASE: TPM2 mutations → disrupted thin filament regulatory mechanism → "
            "abnormal myosin binding kinetics → diverse histological outcomes: "
            "(1) Nemaline myopathy (rods) — most common. "
            "(2) CAP DISEASE — subsarcolemmal protein caps (actin + tropomyosin aggregates) at periphery "
            "of myofibres; unique to TPM2 and ACTA1 among congenital myopathies. "
            "(3) Congenital Fibre Type Disproportion (CFTD) — type I fibres disproportionately small. "
            "(4) Core-rod myopathy — cores + rods combined. "
            "AD dominant-negative: most common mechanism. AR: rare."
        ),
        "cm_group": "Thin Filament — Nemaline/Cap/CFTD (AD)",
        "cm_type": "Nemaline Myopathy 4 (NEM4) / Cap Disease — Tropomyosin β Deficiency",
        "locus": "9p13.3", "omim_gene": 190990, "omim_disease": 609285,
        "inheritance": (
            "Autosomal Dominant (AD). Dominant-negative mechanism most common. De novo occasional. "
            "AR cases: rarer, more severe. "
            "ALLELIC DISORDERS: Distal Arthrogryposis Type 1 (DA1) — OMIM #108120 (contractures + "
            "foot deformity; skeletal rather than muscle-weakness predominant). "
            "Same TPM2 mutation → variable expressivity (cap disease in one family member, "
            "nemaline in another)."
        ),
        "phenotype": (
            "ONSET: Neonatal-infantile. SPECTRUM: milder than NEB/ACTA1 overall. "
            "Proximal + axial weakness; delayed motor milestones; most patients achieve ambulation. "
            "FACIAL WEAKNESS: mild to moderate. "
            "RESPIRATORY: usually mild; rarely requires NIV unless scoliosis severe. "
            "CONTRACTURES/ARTHROGRYPOSIS: characteristic — foot deformities (talipes equinovarus), "
            "finger contractures, jaw contracture possible. "
            "Cap disease subtype: subsarcolemmal caps on biopsy — unique and recognisable histological finding. "
            "CK: normal. CARDIAC: usually spared. "
            "LONG-TERM: most ambulatory adults with variable weakness."
        ),
        "disease": (
            "Nemaline Myopathy 4 / Cap Disease / CFTD — TPM2-related myopathy. "
            "Diagnosis: muscle biopsy (caps on H&E/Gomori trichrome + EM; nemaline rods if NEM4) + TPM2 sequencing. "
            "Usually milder than NEB/ACTA1. No disease-modifying therapy approved. "
            "Physiotherapy; contracture management; scoliosis surveillance. "
            "DA1 allele: orthopaedic management of arthrogryposis."
        ),
        "treatment_options": [
            "Physiotherapy: stretching (contracture prevention); strengthening; hydrotherapy",
            "Contracture management: serial casting; orthotics for foot deformities; rarely surgical release",
            "Scoliosis: monitor Cobb angle; surgical if >40°",
            "Annual FVC: usually mild; NIV rarely required unless severe scoliosis compresses thorax",
            "Cardiac: usually spared; echo if clinically indicated",
            "Genetic counselling: AD — 50% transmission risk; de novo check parental DNA",
            "Allelic DA1 (arthrogryposis): orthopaedic consultation for joint contractures at birth",
        ],
        "key_ddx": [
            "Cap disease: TPM2-specific (caps) + ACTA1 (actin aggregate — EM distinguishes); no other gene causes classic caps",
            "Nemaline Myopathy: NEB (AR, most common), ACTA1 (de novo, severe), TPM3 (α-tropomyosin, similar)",
            "CFTD: ACTA1, MYH7, SEPN1 (all cause CFTD on biopsy with different clinical features)",
            "Distal Arthrogryposis (DA): TPM2 (DA1), TNNI2/TNNT3 (DA2A/DA2B/Sheldon-Hall)",
            "Congenital Myotonic Dystrophy (DM1): maternal CTG repeat; facial diplegia; myotonia later",
        ],
        "onset_range_y": (0, 10),
        "cardiac_risk": False,
        "respiratory_risk": False,
        "contractures": True,
        "facial_weakness": True,
        "ophthalmoparesis": False,
        "rigid_spine": False,
        "mh_risk": False,
        "neonatal_severe": False,
        "ad_inheritance": True,
        "very_high_ck": False,
        "ptosis": False,
    },

    # ── MYH7 — Myosin Heavy Chain 7 ───────────────────────────────────────
    {
        "gene": "MYH7", "protein": "Myosin Heavy Chain 7 (β-myosin heavy chain)",
        "alias": "MYH7; OMIM gene 160760; Laing Distal Myopathy #160500; Myosin Storage Myopathy #608358; 14q11.2; AD; ALLELIC with HCM #192600 + DCM #613765 — CARDIAC MANDATORY",
        "aa": "1935 aa", "kDa": "223 kDa",
        "gene_class": (
            "MYH7 encodes β-myosin heavy chain (1935 aa, 223 kDa), the predominant myosin isoform "
            "in slow-twitch (type I) skeletal muscle fibres AND in the adult heart (ventricular myosin). "
            "MECHANISM OF DISEASE: "
            "(1) SKELETAL MUSCLE: AD dominant-negative mutations → structurally abnormal myosin → "
            "impaired thick filament assembly → weakness preferentially affecting type I fibres. "
            "LAING DISTAL MYOPATHY: foot drop (great toe/ankle dorsiflexion first) + finger extension "
            "weakness + neck flexor weakness (characteristic triad); early age of onset. "
            "MYOSIN STORAGE MYOPATHY: accumulation of β-myosin (hyaline bodies/inclusions) in type I "
            "fibres on biopsy; typically milder, proximal or distal. "
            "(2) CARDIAC (ALLELIC): same MYH7 gene causes HCM (hypertrophic cardiomyopathy, most common "
            "sarcomeric HCM gene) and DCM (dilated cardiomyopathy). Skeletal + cardiac mutations often "
            "different, but OVERLAP exists — patients with skeletal MYH7 myopathy MAY DEVELOP CARDIOMYOPATHY. "
            "ANNUAL CARDIAC ECHO MANDATORY for all MYH7-myopathy patients regardless of cardiac symptoms."
        ),
        "cm_group": "Thick Filament — Distal/Storage Myopathy (AD, CARDIAC-allelic)",
        "cm_type": "Laing Distal Myopathy (MPD1) / Myosin Storage Myopathy — β-Myosin HC",
        "locus": "14q11.2", "omim_gene": 160760, "omim_disease": 160500,
        "inheritance": (
            "Autosomal Dominant (AD). Dominant-negative mechanism. De novo frequent in severe childhood cases. "
            "ALLELIC DISORDERS (same MYH7 gene): "
            "HCM (Hypertrophic Cardiomyopathy): OMIM #192600 — most common sarcomeric HCM gene (30-40%). "
            "DCM (Dilated Cardiomyopathy): OMIM #613765. "
            "Laing Distal Myopathy: OMIM #160500 — skeletal weakness (distal-first). "
            "Myosin Storage Myopathy: OMIM #608358 — hyaline body myopathy. "
            "Some mutations cause BOTH skeletal AND cardiac phenotype in the same individual."
        ),
        "phenotype": (
            "LAING DISTAL MYOPATHY ONSET: childhood (2–25 y). "
            "CHARACTERISTIC WEAKNESS DISTRIBUTION: great toe and ankle dorsiflexion FIRST (foot drop); "
            "finger extension weakness early; NECK FLEXOR WEAKNESS (cannot lift head from pillow). "
            "Proximal shoulder/pelvic girdle weakness develops later. Facial weakness mild. "
            "CK: normal to mildly elevated. Slow progression; most remain ambulatory for decades. "
            "CARDIAC: HCM or DCM risk — cardiac evaluation MANDATORY. "
            "MYOSIN STORAGE MYOPATHY ONSET: infantile-childhood; proximal > distal; hyaline bodies on biopsy. "
            "SCOLIOSIS: moderate frequency. RESPIRATORY: mild unless severe scoliosis."
        ),
        "disease": (
            "Laing Distal Myopathy / Myosin Storage Myopathy — MYH7 mutation. "
            "Diagnosis: biopsy (hyaline bodies on Gomori/PAS; type I fibre hypertrophy/predominance; "
            "EM: thick filament aggregates) + MYH7 sequencing. "
            "IMPORTANT: cardiac MRI/echo + ECG mandatory given HCM/DCM allelic risk. "
            "No approved disease-modifying therapy for skeletal myopathy. "
            "AFOs for foot drop; physiotherapy; cardiac management per HCM/DCM guidelines if applicable."
        ),
        "treatment_options": [
            "Annual cardiac echo + ECG: MANDATORY — MYH7 is allelic with HCM + DCM; cardiac involvement in subset",
            "AFOs (ankle-foot orthoses): for foot drop/great-toe extension weakness — most common presenting complaint",
            "Physiotherapy: targeted strengthening; hydrotherapy; avoid disuse atrophy",
            "HCM management (if cardiac involvement confirmed): beta-blocker/verapamil/disopyramide; ICD if SCD risk high",
            "DCM management (if confirmed): ACE inhibitor + beta-blocker; ICD; cardiac transplant if refractory",
            "Scoliosis: monitor and correct if progressive (Cobb >40°)",
            "Genetic counselling: AD 50% recurrence; cardiac cascade screening of first-degree relatives mandatory",
        ],
        "key_ddx": [
            "Distal Myopathies: GNE myopathy (Nonaka; distal-leg-first; rimmed vacuoles; Japanese founder); "
            "Miyoshi Myopathy (DYSF; calf-wasting; very high CK); Welander (TIA1; finger extension; Scandinavian)",
            "HCM: MYH7 (most common sarcomeric), MYBPC3 (most common overall), TNNT2, TNNI3 — echo + genetic panel",
            "LGMD: LGMD D1 (LMNA — laminopathy; cardiac first; no foot drop triad); CMT (neuropathy — NCV)",
            "Myosin Storage Myopathy: MYH7 + MYH2 — hyaline bodies on biopsy + genotyping",
            "Congenital Fibre Type Disproportion (CFTD): MYH7 → type I fibre hypertrophy + proportional defect",
        ],
        "onset_range_y": (2, 25),
        "cardiac_risk": True,
        "respiratory_risk": False,
        "contractures": False,
        "facial_weakness": False,
        "ophthalmoparesis": False,
        "rigid_spine": False,
        "mh_risk": False,
        "neonatal_severe": False,
        "ad_inheritance": True,
        "very_high_ck": False,
        "ptosis": False,
    },

    # ── SELENON — Selenoprotein N ──────────────────────────────────────────
    {
        "gene": "SELENON", "protein": "Selenoprotein N (SelN/SEPN1)",
        "alias": "SELENON (formerly SEPN1); OMIM gene 606210; RSMD1 Rigid Spine Muscular Dystrophy 1 #602771; 1p36.11; AR; RESPIRATORY DISPROPORTIONATE — NIV MANDATORY EARLY even before severe limb weakness",
        "aa": "590 aa", "kDa": "65 kDa",
        "gene_class": (
            "SELENON (formerly SEPN1) encodes selenoprotein N (590 aa, 65 kDa), a glycoprotein of the "
            "endoplasmic/sarcoplasmic reticulum membrane containing a selenocysteine (Sec) residue "
            "at its catalytic site. Selenoprotein N functions as a reductase/oxidoreductase involved in "
            "redox homeostasis and regulation of calcium homeostasis in the ER/SR. "
            "MECHANISM OF DISEASE: AR biallelic LOF → absence of selenoprotein N → impaired ER redox "
            "regulation → oxidative stress in muscle fibres → characteristic myopathy. "
            "HISTOLOGICAL PATTERNS (same gene): Rigid Spine Muscular Dystrophy 1 (RSMD1, most common), "
            "Multiminicore Disease (MmD), Desmin-related myopathy with Mallory Body-like inclusions, "
            "CFTD. "
            "CRITICAL FEATURE: RESPIRATORY INVOLVEMENT IS DISPROPORTIONATE to limb weakness. "
            "Patients may have FVC <50% predicted while still walking — respiratory failure can "
            "precede ambulation loss. RIGID SPINE limits chest expansion further. "
            "Selenocysteine incorporation requires SECIS element in 3'UTR — standard sequencing may miss "
            "3'UTR variants; gene coverage must include 3'UTR."
        ),
        "cm_group": "ER Redox/Calcium — Rigid Spine Myopathy (AR)",
        "cm_type": "Rigid Spine Muscular Dystrophy 1 (RSMD1) — Selenoprotein N Deficiency",
        "locus": "1p36.11", "omim_gene": 606210, "omim_disease": 602771,
        "inheritance": (
            "Autosomal Recessive (AR). Biallelic LOF. Compound heterozygous common. "
            "SECIS element (3'UTR): selenocysteine insertion sequence — variants in 3'UTR may be pathogenic "
            "and require specific 3'UTR sequencing coverage. "
            "Consanguineous families: homozygous null common. Prevalence: pan-ethnic, no major founder."
        ),
        "phenotype": (
            "ONSET: Infantile-early childhood (birth to 2 y in severe; up to 10 y in mild). "
            "RIGID SPINE: the CARDINAL feature — inability to flex the neck and trunk; "
            "spine rigidity precedes significant limb weakness in many patients. "
            "PROXIMAL WEAKNESS: shoulder and hip girdle predominant; walking usually maintained for years. "
            "RESPIRATORY: DISPROPORTIONATE — FVC declines out of proportion to limb weakness. "
            "SCOLIOSIS: severe + progressive; exacerbates respiratory compromise. "
            "FACIAL WEAKNESS: mild to moderate. "
            "CARDIAC: usually spared but monitor. "
            "CK: normal to mildly elevated. "
            "LATE STAGE: NIV-dependent; scoliosis surgery complex due to spinal rigidity + respiratory compromise."
        ),
        "disease": (
            "Rigid Spine Muscular Dystrophy 1 (RSMD1) — SELENON deficiency. "
            "CRITICAL WARNING: respiratory failure is the leading cause of early death in RSMD1. "
            "Delay in starting NIV is the most avoidable mortality cause. "
            "Diagnosis: clinical (rigid spine + disproportionate respiratory failure) + biopsy (minicores "
            "± mallory-like inclusions) + SELENON sequencing (include 3'UTR). "
            "No approved disease-modifying therapy. Early NIV mandatory."
        ),
        "treatment_options": [
            "NIV MANDATORY EARLY: start when FVC <60% predicted OR nocturnal desaturation — DO NOT WAIT for severe symptoms",
            "Annual FVC + nocturnal oximetry from diagnosis: FVC decline is the main mortality predictor",
            "Rigid spine + scoliosis: complex surgical planning (general anaesthesia risk from respiratory compromise); pre-operative NIV optimisation mandatory",
            "Spinal bracing: temporary benefit; delays scoliosis progression but not definitive",
            "Physiotherapy: trunk and respiratory muscle exercises; swimming/hydrotherapy",
            "Cardiac echo: usually normal but recommended annually given myopathy diagnosis",
            "3'UTR sequencing: standard panels may miss SECIS element variants — confirm coverage",
            "Genetic counselling: AR 25% recurrence; prenatal diagnosis available",
        ],
        "key_ddx": [
            "RSMD: Other rigid-spine myopathies — LMNA (laminopathy; cardiac arrhythmia; contractures), FHL1 (Emery-Dreifuss-like)",
            "Multiminicore Disease: RYR1 (most common MmD; MH risk; different gene) vs SELENON (rigid spine distinguishes)",
            "Emery-Dreifuss Muscular Dystrophy: EMD (emerin, XL), LMNA (AD/AR) — contractures + cardiac first; no rigid spine",
            "Congenital Muscular Dystrophy (CMD): COL6A1-3 (Ullrich; contractures; proximal>distal; skin-fragility); "
            "SELENON does NOT cause CMD-level hypotonia/neonatal severe as a rule",
            "Pompe Disease: GAA deficiency; lysosomal vacuoles; elevated CK; enzyme assay diagnostic",
        ],
        "onset_range_y": (0, 10),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "contractures": False,
        "facial_weakness": True,
        "ophthalmoparesis": False,
        "rigid_spine": True,
        "mh_risk": False,
        "neonatal_severe": False,
        "ad_inheritance": False,
        "very_high_ck": False,
        "ptosis": False,
    },

    # ── MTM1 — Myotubularin 1 ─────────────────────────────────────────────
    {
        "gene": "MTM1", "protein": "Myotubularin 1",
        "alias": "MTM1; OMIM gene 300415; XLMTM X-Linked Myotubular Myopathy #310400; Xq28; XLR; NEONATAL SEVERE — most males require ventilator from birth; prenatal polyhydramnios; gene therapy (AT132/resamirigene beremagene geperparvovec) Phase 3",
        "aa": "603 aa", "kDa": "68 kDa",
        "gene_class": (
            "MTM1 encodes myotubularin 1 (603 aa, 68 kDa), a phosphoinositide phosphatase that "
            "dephosphorylates PI(3)P and PI(3,5)P2 at endosomal/lysosomal membranes. "
            "Myotubularin is essential for the development and maintenance of muscle fibre architecture — "
            "specifically for the biogenesis and organisation of the T-tubule system during myotube "
            "maturation (hence the name 'myotubular' — arrested myotube differentiation). "
            "MECHANISM OF DISEASE: XLR hemizygous LOF in males → absent myotubularin → failure of "
            "T-tubule maturation → disorganised excitation-contraction coupling → severe congenital "
            "hypotonia with centrally placed nuclei and T-tubule/SR disorganisation. "
            "HISTOLOGY: Central nuclei in the majority of fibres; necklace fibres; small fibre diameter; "
            "type I fibre predominance — resembles FETAL myotubes (arrested development). "
            "CARRIER FEMALES: usually asymptomatic but can be severely affected due to skewed "
            "X-inactivation (unfavourable lyonisation) → evaluate carrier females clinically. "
            "GENE THERAPY: AT132 (resamirigene beremagene geperparvovec, AAV8-MTM1) — Phase 3 trials; "
            "early deaths in trial due to pre-existing liver disease/high AAV8 dose → currently on hold."
        ),
        "cm_group": "PI(3)P Phosphatase — Centronuclear/Myotubular Myopathy (XLR, most severe)",
        "cm_type": "X-Linked Myotubular Myopathy (XLMTM) — Myotubularin Deficiency",
        "locus": "Xq28", "omim_gene": 300415, "omim_disease": 310400,
        "inheritance": (
            "X-Linked Recessive (XLR). Hemizygous males: severely affected (classic XLMTM). "
            "Heterozygous females: usually asymptomatic; rarely severely affected (skewed X-inactivation). "
            "NEW MUTATIONS: de novo in ~1/3 of index cases. "
            "PRENATAL: polyhydramnios (fetal akinesia/reduced swallowing), decreased fetal movements, "
            "thin ribs on ultrasound in affected males."
        ),
        "phenotype": (
            "NEONATAL ONSET (affected males): "
            "Severe global hypotonia + respiratory failure from birth — most require intubation at delivery. "
            "Facial diplegia (bilateral facial weakness). Ptosis + ophthalmoparesis (complete ophthalmoplegia common). "
            "Areflexia. Long face + macrocephaly (relative). "
            "VENTILATOR DEPENDENCE: >80% require long-term ventilator support; many never achieve unsupported breathing. "
            "ASPIRATION: major mortality risk — gastrostomy mandatory in most. "
            "UNDESCENDED TESTES: frequent. "
            "HEPATIC PELIOSIS: surveillance needed (found at autopsy in some patients). "
            "PROGNOSIS WITHOUT VENTILATION: very high early mortality (80% die <1y without mechanical ventilation). "
            "WITH VENTILATION + SUPPORTIVE CARE: survival to childhood + adolescence possible; "
            "some achieve minimal motor function (wheelchair users)."
        ),
        "disease": (
            "X-Linked Myotubular Myopathy (XLMTM) — most severe congenital myopathy. "
            "Diagnosis: central nuclei on biopsy (≥25% fibres; T-tubule disorganisation on EM); "
            "myotubularin IHC (absent in affected males); MTM1 sequencing + dosage. "
            "GENE THERAPY: AT132/resamirigene beremagene geperparvovec (AAV8-MTM1) — Phase 3 ASPIRO trial; "
            "trial on partial clinical hold after deaths in patients with pre-existing liver disease — "
            "liver function testing mandatory before enrolment. "
            "Supportive care remains standard."
        ),
        "treatment_options": [
            "Ventilatory support IMMEDIATELY at birth: endotracheal intubation → tracheostomy for long-term ventilation; NIV rarely sufficient in severe XLMTM",
            "Gastrostomy: aspiration prevention + nutritional support — near-universal need in affected males",
            "Hepatic surveillance: liver USS + LFTs annually (hepatic peliosis risk; critical before gene therapy)",
            "Physiotherapy: passive range of motion; positioning; prevent contractures; supported sitting",
            "Undescended testes: urology referral; orchidopexy if bilateral",
            "Gene therapy (AT132/resamirigene beremagene geperparvovec, AAV8-MTM1): Phase 3 trial ASPIRO (partial hold — liver disease exclusion mandatory)",
            "Carrier female evaluation: FVC, strength assessment, CK — severe in ~5% due to skewed X-inactivation",
            "Genetic counselling: X-linked — 50% of sons affected; 50% of daughters carriers; prenatal diagnosis via CVS/amnio",
        ],
        "key_ddx": [
            "Centronuclear Myopathy: DNM2 (AD, milder, adult ptosis); BIN1 (AR, intermediate severity); CCDC78",
            "SMA Type 1 (Werdnig-Hoffmann): SMN1 deletion; lower motor neuron (anterior horn cell); tongue fasciculations; no central nuclei on biopsy",
            "Pompe Disease (Infantile-Onset): GAA deficiency; hypertrophic cardiomyopathy; lysosomal vacuoles; alpha-glucosidase enzyme assay",
            "Congenital Myotonic Dystrophy (DM1): maternal CTG repeat; respiratory failure + facial diplegia + club feet; myotonia on EMG of mother",
            "Spinal Muscular Atrophy with Respiratory Distress (SMARD1): IGHMBP2; respiratory failure + distal weakness + phrenic nerve palsy",
        ],
        "onset_range_y": (0, 0),
        "cardiac_risk": False,
        "respiratory_risk": True,
        "contractures": False,
        "facial_weakness": True,
        "ophthalmoparesis": True,
        "rigid_spine": False,
        "mh_risk": False,
        "neonatal_severe": True,
        "ad_inheritance": False,
        "very_high_ck": False,
        "ptosis": True,
    },

    # ── DNM2 — Dynamin 2 ──────────────────────────────────────────────────
    {
        "gene": "DNM2", "protein": "Dynamin 2",
        "alias": "DNM2; OMIM gene 602378; Autosomal Dominant Centronuclear Myopathy #160150; 19p13.2; AD; PTOSIS + OPHTHALMOPARESIS hallmark; slowly progressive; CMT2M allelic",
        "aa": "870 aa", "kDa": "98 kDa",
        "gene_class": (
            "DNM2 encodes dynamin 2 (870 aa, 98 kDa), a large GTPase that mediates membrane tubulation "
            "and vesicle scission in multiple intracellular trafficking pathways. In muscle, dynamin 2 is "
            "essential for T-tubule biogenesis and maintenance — the same structural compartment disrupted "
            "in MTM1/XLMTM. Myotubularin (MTM1) and dynamin 2 are in a functional pathway: MTM1 regulates "
            "PI(3)P levels that modulate DNM2 recruitment to membranes. "
            "MECHANISM OF DISEASE: AD dominant-negative DNM2 mutations (hotspot in PH domain/GED) → "
            "impaired dynamin GTPase activity → abnormal T-tubule biogenesis → disorganised EC coupling "
            "→ central nuclei myopathy (adult-onset, milder than XLMTM). "
            "HISTOLOGY: Central nuclei (often perinuclear halos + radial spoke pattern on oxidative stains); "
            "type I fibre predominance; necklace fibres. "
            "ALLELIC DISORDERS: CMT2M (Charcot-Marie-Tooth type 2M) and CMT4B3 — axonal/demyelinating "
            "neuropathy from same DNM2 gene; neuropathy mutations distinct from myopathy mutations. "
            "MUTATION HOTSPOT: p.Arg465Trp (most common myopathy allele, ~40% of cases) in PH domain."
        ),
        "cm_group": "GTPase / T-Tubule Biogenesis — Centronuclear Myopathy (AD, milder)",
        "cm_type": "Autosomal Dominant Centronuclear Myopathy (ADCNM) — Dynamin 2",
        "locus": "19p13.2", "omim_gene": 602378, "omim_disease": 160150,
        "inheritance": (
            "Autosomal Dominant (AD). Dominant-negative mechanism. p.Arg465Trp (PH domain) most common (~40%). "
            "ALLELIC DISORDERS: CMT2M (axonal neuropathy; different mutation spectrum) and CMT4B3 "
            "(demyelinating neuropathy — rare). Check EMG/NCS for neuropathy overlap in DNM2 patients. "
            "De novo: rare; most have AD family history."
        ),
        "phenotype": (
            "ONSET: Infantile-childhood-adult (variable). "
            "PTOSIS + OPHTHALMOPARESIS: HALLMARK — bilateral ptosis and limited extraocular movements; "
            "can be mistaken for CPEO (chronic progressive external ophthalmoplegia) or OPMD. "
            "PROXIMAL WEAKNESS: mild to moderate; slowly progressive; most patients remain ambulatory. "
            "FACIAL WEAKNESS: mild. BULBAR: mild in severe cases. "
            "RESPIRATORY: usually mild; surveillance required. "
            "CK: normal to 2× ULN. "
            "PROGRESSION: slowly progressive over decades; most patients maintain independent function. "
            "CMT OVERLAP: check NCS/EMG for subclinical neuropathy in all DNM2 patients."
        ),
        "disease": (
            "Autosomal Dominant Centronuclear Myopathy (ADCNM) — DNM2. "
            "Diagnosis: biopsy (central nuclei; radial spokes on NADH; necklace fibres on EM); "
            "DNM2 sequencing (p.Arg465Trp screening first). "
            "EMG/NCS: exclude CMT overlap. "
            "No approved disease-modifying therapy. Slowly progressive — most patients live normal lifespan. "
            "Physiotherapy; ptosis surgery if severe; annual FVC."
        ),
        "treatment_options": [
            "Ptosis surgery: blepharoplasty or frontalis suspension if ptosis impairs vision; timing based on functional impact",
            "Physiotherapy: strength maintenance; hydrotherapy; avoid prolonged immobility",
            "Annual FVC + respiratory review: mild decline expected; NIV if significant compromise (rare)",
            "NCS/EMG: check for CMT overlap (CMT2M/CMT4B3 — neuropathic changes may be subclinical)",
            "Statin caution: myopathy may worsen; CK monitoring if statins prescribed",
            "Genetic counselling: AD 50% transmission; family members should be screened for ptosis",
            "MTM1 functional rescue strategy (experimental): DNM2 antisense oligonucleotides (DNM2-ASO) show rescue in murine MTM1 model — early translational research",
        ],
        "key_ddx": [
            "CPEO (Chronic Progressive External Ophthalmoplegia): mitochondrial DNA deletions; Kearns-Sayre; ragged red fibres; cardiac conduction defect",
            "OPMD (Oculopharyngeal Muscular Dystrophy): PABPN1 trinucleotide repeat; onset >40y; swallowing difficulty early; French-Canadian founder",
            "MTM1/XLMTM: X-linked males; neonatal severe; full ophthalmoplegia; central nuclei (same biopsy pattern — but XLMTM is neonatal + much more severe)",
            "BIN1-centronuclear: AR; intermediate severity between MTM1 and DNM2; BIN1 encodes amphiphysin 2 (same T-tubule pathway)",
            "MYH7-myopathy: occasional ptosis; proximal + distal (Laing); no radial spokes on biopsy; cardiac allelic",
        ],
        "onset_range_y": (0, 40),
        "cardiac_risk": False,
        "respiratory_risk": False,
        "contractures": False,
        "facial_weakness": True,
        "ophthalmoparesis": True,
        "rigid_spine": False,
        "mh_risk": False,
        "neonatal_severe": False,
        "ad_inheritance": True,
        "very_high_ck": False,
        "ptosis": True,
    },
]


def _gen_patients(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    patients = []
    gene = gene_data["gene"]
    o_min, o_max = gene_data["onset_range_y"]
    n = 40

    for i in range(n):
        onset = round(rng.uniform(o_min, max(o_max, o_min + 0.1)), 1)

        # Severity biased by gene
        if gene in ("MTM1",):
            sev_weights = [0.05, 0.15, 0.80]  # mostly Severe
        elif gene in ("ACTA1",):
            sev_weights = [0.15, 0.30, 0.55]
        elif gene in ("NEB", "SELENON"):
            sev_weights = [0.20, 0.45, 0.35]
        elif gene in ("RYR1",):
            sev_weights = [0.25, 0.45, 0.30]
        elif gene in ("TPM2",):
            sev_weights = [0.40, 0.45, 0.15]
        elif gene in ("DNM2",):
            sev_weights = [0.50, 0.40, 0.10]
        else:  # MYH7
            sev_weights = [0.45, 0.40, 0.15]

        sev = rng.choices(["Mild", "Moderate", "Severe"], weights=sev_weights)[0]

        # Clinical features
        resp = (
            gene_data["respiratory_risk"] and
            rng.random() < (0.85 if gene in ("MTM1", "SELENON") else
                            0.55 if gene in ("NEB", "ACTA1") else 0.30)
        )
        cardiac = (
            gene_data["cardiac_risk"] and
            rng.random() < (0.30 if gene == "MYH7" else 0.10)
        )
        mh_event = (
            gene_data["mh_risk"] and
            rng.random() < 0.20  # ~20% had a perioperative MH event or near-miss
        )
        vent_dependent = (gene == "MTM1" and rng.random() < 0.80)
        contractures = (
            gene_data["contractures"] and rng.random() < 0.55
        )
        facial_w = (
            gene_data["facial_weakness"] and rng.random() < 0.70
        )
        ophthalmo = (
            gene_data["ophthalmoparesis"] and rng.random() < 0.75
        )
        ptosis = (
            gene_data["ptosis"] and rng.random() < 0.80
        )
        rigid_sp = (
            gene_data["rigid_spine"] and rng.random() < 0.90
        )
        scoliosis = rng.random() < (0.75 if gene in ("SELENON", "NEB") else 0.40)

        # CK: typically normal/mildly elevated in congenital myopathy
        ck_val = int(rng.uniform(50, 400))
        if gene == "ACTA1" and sev == "Severe":
            ck_val = int(rng.uniform(300, 800))

        # Treatment selection
        if gene == "RYR1":
            if mh_event:
                tx = "MH-safe anaesthesia (propofol/N2O); dantrolene available; MH alert card"
            else:
                tx = "MH-safe anaesthesia protocol; physiotherapy; AFOs if foot drop"
        elif gene == "MTM1":
            if vent_dependent:
                tx = "Tracheostomy + long-term ventilation; PEG; physiotherapy"
            else:
                tx = "NIV; PEG; physiotherapy; gene therapy trial eligibility assessment"
        elif gene == "SELENON":
            tx = "NIV (early start); physiotherapy; scoliosis monitoring; spinal surgery planning"
        elif gene == "MYH7":
            tx = "Annual cardiac echo + ECG; AFOs for foot drop; physiotherapy"
        elif gene == "NEB":
            tx = "NIV if FVC declining; physiotherapy; gastrostomy if swallowing impaired"
        elif gene == "ACTA1":
            tx = "Ventilatory support (severity-dependent); physiotherapy; gastrostomy"
        elif gene == "TPM2":
            tx = "Physiotherapy; AFOs; contracture management; scoliosis surveillance"
        elif gene == "DNM2":
            tx = "Ptosis surgery if vision impaired; physiotherapy; annual FVC"
        else:
            tx = "Physiotherapy; supportive care"

        pid = f"CM-{gene}-{seed}-{i+1:03d}"
        sex = "M" if gene == "MTM1" else rng.choice(["M", "F"])  # XLMTM = males
        patients.append({
            "id": pid, "gene": gene, "sex": sex,
            "onset_age_y": onset, "severity": sev,
            "respiratory_decline": resp,
            "cardiac_event": cardiac,
            "mh_event": mh_event,
            "ventilator_dependent": vent_dependent,
            "contractures": contractures,
            "facial_weakness": facial_w,
            "ophthalmoparesis": ophthalmo,
            "ptosis": ptosis,
            "rigid_spine": rigid_sp,
            "scoliosis": scoliosis,
            "ck_iu_l": ck_val,
            "current_treatment": tx,
            "inheritance": gene_data["inheritance"].split(".")[0],
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gene_data in enumerate(CM_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients(gene_data, seed))
    return all_pts


def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)
    gene_counts = {}
    for p in patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    resp_n = sum(1 for p in patients if p["respiratory_decline"])
    cardiac_n = sum(1 for p in patients if p["cardiac_event"])
    mh_n = sum(1 for p in patients if p["mh_event"])
    vent_n = sum(1 for p in patients if p["ventilator_dependent"])
    contract_n = sum(1 for p in patients if p["contractures"])
    facial_n = sum(1 for p in patients if p["facial_weakness"])
    ophthalmo_n = sum(1 for p in patients if p["ophthalmoparesis"])
    ptosis_n = sum(1 for p in patients if p["ptosis"])
    rigid_n = sum(1 for p in patients if p["rigid_spine"])
    scolio_n = sum(1 for p in patients if p["scoliosis"])
    for p in patients:
        sev[p["severity"]] += 1

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 1)
    mean_ck = round(sum(p["ck_iu_l"] for p in patients) / n)

    return {
        "atlas": "CM-Atlas",
        "full_name": "Complete 8-Gene Congenital Myopathy (CM) Atlas",
        "subtitle": "RYR1·NEB·ACTA1·TPM2·MYH7·SELENON·MTM1·DNM2 — 320 patients (8×40, seeds 1038–1045)",
        "description": (
            "Comprehensive atlas of the 8 most clinically important Congenital Myopathy genes. "
            "Covers: SR calcium release channel (RYR1), thin filament genes (NEB, ACTA1, TPM2), "
            "thick filament (MYH7), ER redox/calcium (SELENON), and centronuclear/T-tubule "
            "biogenesis genes (MTM1, DNM2). "
            "CRITICAL TREATMENT DISTINCTIONS: RYR1 = Malignant Hyperthermia (MH-safe anaesthesia MANDATORY); "
            "SELENON = respiratory failure disproportionate to limb weakness (NIV MANDATORY EARLY); "
            "MTM1 = neonatal severe (ventilator from birth in most males); "
            "ALL = succinylcholine CONTRAINDICATED."
        ),
        "total_patients": n,
        "genes_covered": len(CM_GENES),
        "patients_per_gene": 40,
        "seed_range": "1038–1045",
        "gene_list": [g["gene"] for g in CM_GENES],
        "cm_category_breakdown": {
            "SR Calcium Release (Core Myopathy)": ["RYR1"],
            "Thin Filament / Nemaline (AR)": ["NEB"],
            "Thin Filament / Nemaline (AD de novo/AR)": ["ACTA1"],
            "Thin Filament / Cap-Nemaline (AD)": ["TPM2"],
            "Thick Filament / Distal (AD, Cardiac-allelic)": ["MYH7"],
            "ER Redox/Calcium (Rigid Spine, AR)": ["SELENON"],
            "Centronuclear / Myotubular (XLR, Severe)": ["MTM1"],
            "Centronuclear / GTPase (AD, Milder)": ["DNM2"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "mean_ck_iu_l": mean_ck,
        "clinical_features_prevalence": {
            "respiratory_decline_pct": round(100 * resp_n / n, 1),
            "cardiac_event_pct": round(100 * cardiac_n / n, 1),
            "mh_event_pct": round(100 * mh_n / n, 1),
            "ventilator_dependent_pct": round(100 * vent_n / n, 1),
            "contractures_pct": round(100 * contract_n / n, 1),
            "facial_weakness_pct": round(100 * facial_n / n, 1),
            "ophthalmoparesis_pct": round(100 * ophthalmo_n / n, 1),
            "ptosis_pct": round(100 * ptosis_n / n, 1),
            "rigid_spine_pct": round(100 * rigid_n / n, 1),
            "scoliosis_pct": round(100 * scolio_n / n, 1),
        },
        "key_teaching_points": [
            "RYR1: MALIGNANT HYPERTHERMIA — volatile anaesthetics (halothane/isoflurane/sevoflurane/desflurane) + succinylcholine ABSOLUTELY CONTRAINDICATED; MH-safe protocol (propofol/N2O/regional); dantrolene stocked; MH alert card",
            "SELENON/RSMD1: RESPIRATORY DISPROPORTIONATE — FVC <50% predicted while patient still walking; NIV MANDATORY when FVC <60% or nocturnal desaturation; do NOT wait for severe limb weakness",
            "MTM1/XLMTM: NEONATAL SEVERE — most males require ventilator from birth; aspiration = leading mortality cause; tracheostomy + gastrostomy life-sustaining; AT132 gene therapy (Phase 3, partial hold — liver exclusion mandatory)",
            "ALL CONGENITAL MYOPATHY: Succinylcholine CONTRAINDICATED — rhabdomyolysis/hyperkalaemia risk regardless of MH status",
            "NEB: LARGEST MUSCLE GENE (8537 aa, 183 exons) — MLPA mandatory (misses ~20% of large deletions by sequencing); exon 55 deletion first screen in Ashkenazi Jewish patients (carrier 1:108)",
            "ACTA1: MULTIPLE HISTOLOGICAL PATTERNS — intranuclear rods (de novo, most severe), actin aggregate (AR/de novo, severe), nemaline rods, CFTD (mildest); same gene → widely different prognosis",
            "MYH7: ALLELIC CARDIAC — same gene causes HCM + DCM; annual cardiac echo + ECG mandatory in all MYH7-myopathy patients; foot drop + finger extension + neck flexor triad = Laing distal myopathy",
            "DNM2: PTOSIS + OPHTHALMOPARESIS hallmark — mistaken for CPEO/OPMD; slowly progressive AD; p.Arg465Trp (PH domain) most common allele (~40%); CMT2M allelic (check NCS)",
        ],
        "drug_alerts": [
            "RYR1: VOLATILE ANAESTHETICS (halothane/isoflurane/sevoflurane/desflurane) ABSOLUTELY AVOID — MH crisis (fatal without dantrolene)",
            "ALL CONGENITAL MYOPATHY: Succinylcholine CONTRAINDICATED — hyperkalaemia + rhabdomyolysis; especially dangerous in MTM1 with existing respiratory failure",
            "SELENON/RSMD1: Delay in NIV = avoidable respiratory death — do NOT wait for dyspnoea at rest",
            "MTM1: Aminoglycosides caution in ventilated patients; sedatives/opioids require ventilator backup; gene therapy (AT132) — exclude liver disease first",
            "MYH7: Statins — myopathy risk; if HCM confirmed, avoid negative inotropes (caution with verapamil + disopyramide combination)",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    gene_profiles = []
    for gene_data in CM_GENES:
        gene_pts = [p for p in patients if p["gene"] == gene_data["gene"]]
        n = len(gene_pts)
        sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for p in gene_pts:
            sev[p["severity"]] += 1
        mean_ck_g = round(sum(p["ck_iu_l"] for p in gene_pts) / n)
        gene_profiles.append({
            "gene": gene_data["gene"],
            "protein": gene_data["protein"],
            "alias": gene_data["alias"],
            "locus": gene_data["locus"],
            "omim_gene": gene_data["omim_gene"],
            "omim_disease": gene_data["omim_disease"],
            "inheritance": gene_data["inheritance"],
            "cm_group": gene_data["cm_group"],
            "cm_type": gene_data["cm_type"],
            "aa": gene_data["aa"],
            "kDa": gene_data["kDa"],
            "gene_class": gene_data["gene_class"],
            "phenotype": gene_data["phenotype"],
            "disease": gene_data["disease"],
            "treatment_options": gene_data["treatment_options"],
            "key_ddx": gene_data["key_ddx"],
            "onset_range_y": list(gene_data["onset_range_y"]),
            "n_patients": n,
            "cardiac_risk": gene_data["cardiac_risk"],
            "respiratory_risk": gene_data["respiratory_risk"],
            "contractures": gene_data["contractures"],
            "facial_weakness": gene_data["facial_weakness"],
            "ophthalmoparesis": gene_data["ophthalmoparesis"],
            "ptosis": gene_data["ptosis"],
            "rigid_spine": gene_data["rigid_spine"],
            "mh_risk": gene_data["mh_risk"],
            "neonatal_severe": gene_data["neonatal_severe"],
            "ad_inheritance": gene_data["ad_inheritance"],
            "very_high_ck": gene_data["very_high_ck"],
            "mean_ck_iu_l": mean_ck_g,
            "severity_distribution": {
                "mild_pct": round(100 * sev["Mild"] / n, 1),
                "moderate_pct": round(100 * sev["Moderate"] / n, 1),
                "severe_pct": round(100 * sev["Severe"] / n, 1),
            },
            "clinical_features": {
                "respiratory_decline_pct": round(100 * sum(1 for p in gene_pts if p["respiratory_decline"]) / n, 1),
                "cardiac_event_pct": round(100 * sum(1 for p in gene_pts if p["cardiac_event"]) / n, 1),
                "mh_event_pct": round(100 * sum(1 for p in gene_pts if p["mh_event"]) / n, 1),
                "ventilator_dependent_pct": round(100 * sum(1 for p in gene_pts if p["ventilator_dependent"]) / n, 1),
                "contractures_pct": round(100 * sum(1 for p in gene_pts if p["contractures"]) / n, 1),
                "facial_weakness_pct": round(100 * sum(1 for p in gene_pts if p["facial_weakness"]) / n, 1),
                "ophthalmoparesis_pct": round(100 * sum(1 for p in gene_pts if p["ophthalmoparesis"]) / n, 1),
                "ptosis_pct": round(100 * sum(1 for p in gene_pts if p["ptosis"]) / n, 1),
                "rigid_spine_pct": round(100 * sum(1 for p in gene_pts if p["rigid_spine"]) / n, 1),
                "scoliosis_pct": round(100 * sum(1 for p in gene_pts if p["scoliosis"]) / n, 1),
            },
            "sample_patients": gene_pts[:3],
        })
    return {
        "atlas": "CM-Atlas",
        "genes": gene_profiles,
        "total_patients": len(patients),
    }


def get_definitions() -> dict:
    return {
        "atlas": "CM-Atlas",
        "definitions": [
            {
                "term": "Congenital Myopathy",
                "definition": (
                    "Clinically and genetically heterogeneous group of primary muscle diseases presenting "
                    "at birth or in early infancy with hypotonia, weakness, and characteristic histological "
                    "findings on muscle biopsy. Subdivided by biopsy morphology: Nemaline Myopathy (rods), "
                    "Core Myopathy (central/minicores), Centronuclear/Myotubular Myopathy (central nuclei), "
                    "Congenital Fibre Type Disproportion (CFTD). Same gene (e.g., RYR1, ACTA1) can cause "
                    "multiple histological subtypes. Diagnosis requires biopsy + electron microscopy + gene panel."
                ),
            },
            {
                "term": "Malignant Hyperthermia (MH) — RYR1",
                "definition": (
                    "Life-threatening pharmacogenetic crisis triggered by volatile anaesthetic agents "
                    "(halothane, isoflurane, sevoflurane, desflurane) AND succinylcholine in susceptible "
                    "individuals. RYR1 mutations (dominant GOF hotspots) → uncontrolled SR calcium release "
                    "→ sustained muscle contraction → hyperthermia, rigidity, rhabdomyolysis, acidosis, "
                    "hyperkalaemia, cardiac arrhythmia. TREATMENT: Dantrolene IV 2.5 mg/kg bolus (repeat "
                    "to 10 mg/kg) + STOP triggering agent + hyperventilate 100% O2 + cooling. "
                    "PREVENTION: MH-safe anaesthesia (propofol + N2O + regional); dantrolene stocked. "
                    "ALL RYR1 patients + first-degree relatives need MH anaesthesia card."
                ),
            },
            {
                "term": "Dantrolene — MH Treatment",
                "definition": (
                    "Dantrolene sodium — ryanodine receptor antagonist. Directly inhibits RYR1-mediated "
                    "SR calcium release → terminates MH crisis. DOSE: 2.5 mg/kg IV bolus → repeat every "
                    "5–10 min until signs abate; cumulative max 10 mg/kg. Post-crisis: oral dantrolene "
                    "maintenance 1 mg/kg q6h for 24-72h to prevent recrudescence. "
                    "SIDE EFFECTS: hepatotoxicity (rare with acute use), muscle weakness. "
                    "STOCKING REQUIREMENT: all anaesthetic rooms where RYR1 patients may be treated "
                    "must have dantrolene and mixing supplies immediately available."
                ),
            },
            {
                "term": "Nemaline Myopathy (NEM) — Rods",
                "definition": (
                    "Congenital myopathy characterised by nemaline rods (electron-dense accumulations of "
                    "Z-disc material — α-actinin, actin, tropomyosin, nebulin fragments) visible on "
                    "modified Gomori trichrome as red-purple rods and on electron microscopy as "
                    "rectangular lattice structures. GENES: NEB (most common AR, ~50%), ACTA1 (most common "
                    "AD de novo), TPM2, TPM3, TNNT1 (Amish), KBTBD13, KLHL40, KLHL41, CFL2. "
                    "RODS in nuclei (intranuclear rod myopathy) = severe; ACTA1 dominant de novo. "
                    "Rod number does NOT correlate with severity — severity correlates with gene and variant type."
                ),
            },
            {
                "term": "X-Linked Myotubular Myopathy (XLMTM) — MTM1",
                "definition": (
                    "Severe congenital myopathy caused by MTM1 mutations (Xq28). Affected males present "
                    "with neonatal profound hypotonia, respiratory failure requiring ventilation from birth, "
                    "facial diplegia, ophthalmoplegia, areflexia. Biopsy: central nuclei + necklace fibres "
                    "resembling fetal myotubes. >80% require long-term mechanical ventilation. "
                    "WITHOUT VENTILATION: ~80% die within first year. "
                    "Carrier females: usually asymptomatic; ~5% severely affected (skewed X-inactivation). "
                    "Gene therapy: AT132 (AAV8-MTM1, resamirigene beremagene geperparvovec) — Phase 3 ASPIRO "
                    "trial (partial hold — liver toxicity exclusion mandatory)."
                ),
            },
            {
                "term": "Rigid Spine Muscular Dystrophy 1 (RSMD1) — SELENON",
                "definition": (
                    "SELENON (SEPN1)-related myopathy; AR. Cardinal features: rigid spine (inability to "
                    "flex cervical and thoracic spine) + DISPROPORTIONATE respiratory failure. "
                    "CRITICAL: FVC can be <50% predicted while patient walks independently — "
                    "respiratory failure is NOT proportionate to limb weakness. "
                    "NIV must begin when FVC <60% predicted OR nocturnal desaturation confirmed; "
                    "do NOT wait for dyspnoea at rest (too late). "
                    "Biopsy: multiminicore disease pattern or CFTD. Selenocysteine residue — "
                    "3'UTR SECIS element must be sequenced."
                ),
            },
            {
                "term": "Centronuclear Myopathy (CNM)",
                "definition": (
                    "Group of congenital myopathies characterised by pathologically central nuclei "
                    "(≥25% of fibres in the central position; normal: subsarcolemmal). "
                    "Resembles fetal myotubes ('myotubular myopathy'). "
                    "GENES by severity: MTM1 (XLR, most severe, neonatal) > BIN1 (AR, intermediate) > "
                    "DNM2 (AD, milder, adult-onset ptosis/ophthalmoparesis). "
                    "BIOPSY FEATURES: central nuclei + radial spoke pattern on NADH + necklace fibres on EM. "
                    "Electron microscopy distinguishes CNM from NEM (rods) and CCD (cores)."
                ),
            },
            {
                "term": "Succinylcholine Contraindication in Congenital Myopathy",
                "definition": (
                    "Succinylcholine (suxamethonium) — depolarising neuromuscular blocker — is "
                    "CONTRAINDICATED in all congenital myopathies for two distinct reasons: "
                    "(1) MALIGNANT HYPERTHERMIA TRIGGER (RYR1 GOF mutations) — can precipitate fulminant MH. "
                    "(2) HYPERKALAEMIA + RHABDOMYOLYSIS — depolarisation releases potassium from abnormal "
                    "muscle → hyperkalaemic cardiac arrest (risk in any chronic myopathy with upregulated "
                    "acetylcholine receptors — especially dangerous in MTM1 with existing ventilatory failure). "
                    "USE: rocuronium (with sugammadex reversal available) as non-depolarising alternative."
                ),
            },
            {
                "term": "Core Myopathy — Central Core Disease (CCD)",
                "definition": (
                    "Congenital myopathy with well-circumscribed regions of absent oxidative enzyme activity "
                    "and mitochondria in the centre of type I muscle fibres ('cores'). "
                    "On NADH and SDH staining: discrete central pale cores. On EM: disorganised/absent "
                    "sarcomere structure within the core + absent mitochondria. "
                    "GENE: RYR1 (AR biallelic LOF most common in CCD). "
                    "MULTIMINICORE DISEASE (MmD): Multiple small cores (rather than one central) — "
                    "RYR1 (AR, most common MmD) + SELENON/SEPN1 (AR, + rigid spine). "
                    "CK: usually normal (unlike LGMD)."
                ),
            },
            {
                "term": "Nebulin (NEB) — Largest Sarcomeric Ruler",
                "definition": (
                    "Nebulin (8537 aa, ~800 kDa) is the largest structural protein in the sarcomere, "
                    "spanning the entire thin filament from Z-disc to pointed end. Functions as a "
                    "molecular ruler — sets thin filament length and regulates actin-tropomyosin assembly. "
                    "NEB = 183 exons including a large triplicated region (exons 82–105) that makes "
                    "PCR-based sequencing unreliable. MLPA/gene dosage MANDATORY: standard sequencing "
                    "misses large deletions in ~20% of cases. Exon 55 deletion = most common Ashkenazi "
                    "Jewish founder allele (~1:108 carrier frequency)."
                ),
            },
            {
                "term": "ACTA1 Histological Spectrum",
                "definition": (
                    "ACTA1 (alpha-skeletal actin, 377 aa) mutations cause FOUR distinct biopsy phenotypes, "
                    "all from the same gene: "
                    "(1) Nemaline rods — most common; cytoplasmic rods; moderate severity. "
                    "(2) Intranuclear rods — rods INSIDE myonuclei (EM required); almost always de novo "
                    "dominant; most severe subtype (often fatal neonatal). "
                    "(3) Actin aggregate myopathy — abundant cytoplasmic actin filament accumulations "
                    "replacing sarcomere structure; severe; AR or de novo. "
                    "(4) CFTD — type I fibres ≥12% smaller than type II; mildest; AR or AD. "
                    "EM IS MANDATORY to distinguish intranuclear rods (affects prognosis and counselling)."
                ),
            },
            {
                "term": "MYH7 Allelic Spectrum",
                "definition": (
                    "MYH7 (β-myosin heavy chain, 1935 aa) is expressed in slow-twitch skeletal muscle "
                    "and adult ventricular heart muscle. The SAME gene causes multiple distinct diseases: "
                    "(1) Laing Distal Myopathy (MPD1): foot drop + finger extension + neck flexor triad. "
                    "(2) Myosin Storage Myopathy: hyaline body inclusions on biopsy; proximal predominant. "
                    "(3) HCM (Hypertrophic Cardiomyopathy): second most common sarcomeric HCM gene (30-40%). "
                    "(4) DCM (Dilated Cardiomyopathy): rarer MYH7 allele. "
                    "IMPLICATION: Annual cardiac echo + ECG mandatory for ALL MYH7-myopathy patients "
                    "regardless of whether their mutation is known to cause cardiac disease."
                ),
            },
            {
                "term": "Dynamin 2 (DNM2) — p.Arg465Trp Hotspot",
                "definition": (
                    "Dynamin 2 (870 aa) is a large GTPase required for membrane tubulation/scission, "
                    "T-tubule biogenesis, and endosomal trafficking in muscle. "
                    "AD centronuclear myopathy mutations cluster in the PH domain (membrane binding) "
                    "and GED domain. p.Arg465Trp (PH domain): most common ADCNM allele (~40%). "
                    "ALLELIC: CMT2M (axonal CMT) and CMT4B3 (demyelinating CMT) — different mutation "
                    "cluster; NCS/EMG should be performed in all DNM2 patients to detect subclinical neuropathy. "
                    "DNM2-ASO (antisense oligonucleotide): reduces DNM2 expression → rescues MTM1 "
                    "murine model (MTM1/DNM2 pathway interaction — early translational research)."
                ),
            },
        ],
    }
