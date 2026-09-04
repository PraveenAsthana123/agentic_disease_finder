#!/usr/bin/env python3
"""CMS-Atlas — Complete 8-Gene Congenital Myasthenic Syndromes Atlas
CHRNE   (ε-AChR; AR; 493 aa; 17p13.2; most common CMS gene worldwide; AChR-deficient postsynaptic) ·
RAPSN   (Rapsyn; AR; 412 aa; 11p11.2; AChR clustering; N88K European founder; neonatal crisis) ·
DOK7    (DOK7; AR; 504 aa; 4p16.3; limb-girdle CMS; MuSK pathway; salbutamol FIRST-LINE; pyridostigmine WORSENS) ·
COLQ    (ColQ; AR; 457 aa; 3p24.3; AChE-deficient; PYRIDOSTIGMINE ABSOLUTELY CONTRAINDICATED; slow pupils PATHOGNOMONIC) ·
CHAT    (ChAT; AR; 748 aa; 10q11.23; presynaptic; episodic apnea; fever-triggered crisis; inter-ictal strength NORMAL) ·
GFPT1   (GFPT1; AR; 699 aa; 2p13.1; glycosylation-deficient limb-girdle CMS; tubular aggregates on biopsy; responds AChEI) ·
AGRN    (Agrin; AR; 2045 aa; 1p36.33; presynaptic agrin; activates MuSK-DOK7-rapsyn cascade; ephedrine/salbutamol) ·
SCN4A   (Nav1.4; AD or AR; 1836 aa; 17q23.3; myasthenic sodium channel syndrome; cold-sensitive; quinidine)
320-patient aggregate cohort (8 × 40, seeds 1022–1029)

Congenital Myasthenic Syndromes — Key Neurological Principles:
  - CMS DEFINITION: Inherited disorders of neuromuscular junction (NMJ) transmission. NOT autoimmune
    (AChR/MuSK/LRP4 antibodies NEGATIVE). GENETIC — diagnosed by molecular genetics ± EMG/SFEMG ± biopsy.
    CMS ≠ Myasthenia Gravis (MG): MG is autoimmune; CMS is genetic. Critical diagnostic distinction.
  - CMS CLASSIFICATION by anatomical defect:
    Presynaptic: CHAT (ChAT), AGRN (agrin at nerve terminal).
    Synaptic basal lamina: COLQ (ColQ-anchors AChE).
    Postsynaptic: CHRNE/CHRND/CHRNB1/CHRNA1 (AChR subunits), RAPSN (clustering), DOK7 (MuSK pathway).
    Glycosylation: GFPT1 (rate-limiting hexosamine biosynthesis enzyme).
    Ion channel: SCN4A (Nav1.4 myasthenic variant).
  - TREATMENT RULES (MANDATORY TO KNOW):
    (1) COLQ-CMS: Pyridostigmine ABSOLUTELY CONTRAINDICATED — ColQ-anchored AChE already absent;
        AChE inhibitor → depolarisation block → paradoxical WORSENING; use ephedrine ± salbutamol.
    (2) DOK7-CMS: Pyridostigmine typically worsens (not absolute CI but counterproductive);
        salbutamol (albuterol) or ephedrine — β2-agonists upregulate DOK7/MuSK AChR density.
    (3) Slow-channel CMS (SCN4A GOF, CHRND GOF): quinidine or fluoxetine (open-channel blockers);
        pyridostigmine CONTRAINDICATED (prolonged channel opening → depolarisation block).
    (4) Fast-channel CMS (CHRNE, RAPSN LOF): 3,4-DAP + pyridostigmine first-line.
    (5) CHAT-CMS: prophylactic ventilation / fever management; 3,4-DAP; avoid triggers (fever/GA).
    (6) GFPT1-CMS: standard pyridostigmine ± 3,4-DAP; responds unlike DOK7.
    (7) AGRN-CMS: ephedrine/salbutamol; 3,4-DAP sometimes helpful.
  - COLQ HALLMARK: Slow pupillary light response (cholinergic pupil) — cholinergic autonomic sign
    from diffuse AChE deficiency — PATHOGNOMONIC for COLQ-CMS diagnosis.
  - CHAT HALLMARK: Inter-episodic strength NORMAL (or near-normal); crisis precipitated by fever,
    infection, general anaesthesia → sudden onset bulbar/respiratory failure → ICU.
  - DOK7 HALLMARK: Limb-girdle pattern (proximal > distal) with NECK FLEXOR WEAKNESS;
    ptosis and ophthalmoplegia variable (not obligate); EMG: decremental → incremental paradox.
  - GENETIC COUNSELLING: Most CMS are AR (biallelic) → recurrence risk 25%; CHRND/SCN4A slow-channel
    may be AD GOF (de novo or familial). Cascade family testing warranted.

COHORT: 8 × 40 = 320 patient slots (seeds 1022–1029; gene-specific seeds)
"""

import random

SEED_BASE = 1022

CMS_GENES = [
    # ── CHRNE — ε-AChR (most common CMS gene) ────────────────────────────────
    {
        "gene": "CHRNE", "protein": "Acetylcholine Receptor ε-Subunit",
        "alias": "AChR ε (epsilon); ε-AChR-deficient CMS; OMIM #616313; AR; most common CMS gene worldwide; 17p13.2",
        "aa": "493 aa", "kDa": "56 kDa",
        "gene_class": (
            "CHRNE encodes the epsilon (ε) subunit of the adult-type nicotinic acetylcholine receptor (nAChR). "
            "The adult nAChR pentamer is (α1)₂β1δε — the ε subunit replaces the fetal γ subunit at birth. "
            "MECHANISM OF DISEASE: LOF mutations (frameshift, nonsense, splice-site, missense) reduce ε-subunit "
            "expression → diminished AChR density at the end-plate → impaired ACh-triggered channel opening → "
            "reduced miniature end-plate potential (MEPP) amplitude → decremental EMG response. "
            "MUTATION SPECTRUM: ε1267delG (frameshift) most common Mediterranean/North-African; "
            "εP121L missense (Gulf Arab founder); >100 pathogenic variants catalogued. "
            "CHRNE 17p13.2; OMIM gene 100725."
        ),
        "cms_group": "Postsynaptic CMS — AChR-deficient (most common)",
        "cms_type": "AChR-Deficient Postsynaptic CMS",
        "locus": "17p13.2", "omim_gene": 100725, "omim_disease": 616313,
        "inheritance": "Autosomal Recessive (AR). Biallelic LOF. Compound heterozygous common. High consanguinity rate in North-African/Middle-Eastern populations.",
        "phenotype": (
            "Onset: birth to 2 years (most <1 y). Ptosis ± ophthalmoplegia (frequent). Bulbar weakness "
            "(feeding difficulties in infancy). Fluctuating limb weakness, fatigability. Respiratory crises "
            "precipitated by fever/infection. Severity varies: mild (ocular only) to severe (ventilator-dependent). "
            "Crises can be life-threatening in neonates."
        ),
        "disease": (
            "CHRNE-CMS — AChR ε-Subunit Deficiency. Most common CMS gene globally. "
            "Pyridostigmine + 3,4-DAP first-line: excellent clinical response in most patients. "
            "North-African, Gypsy, and Gulf Arab populations have founder mutations with high carrier frequencies."
        ),
        "treatment_options": [
            "Pyridostigmine (AChEI): first-line; 30-60 mg TDS-QDS; improves NMJ transmission",
            "3,4-DAP (amifampridine): first-line adjunct; increases ACh quantal release; 10-20 mg TDS",
            "3,4-DAP + pyridostigmine: combination more effective than either alone",
            "Salbutamol: second-line adjunct if suboptimal response; may upregulate AChR density",
            "Avoid: AMINOGLYCOSIDE antibiotics (NMJ blockers); NEUROMUSCULAR BLOCKING AGENTS (suxamethonium, vecuronium) — anesthesia risk",
        ],
        "key_ddx": [
            "MG (autoimmune): AChR/MuSK/LRP4 antibodies distinguish — always send in apparent seronegative MG before age 5",
            "RAPSN-CMS: similar phenotype; N88K founder allele; RNT abnormal",
            "CHRND-CMS: delta subunit; Escobar (multiple pterygium) overlap; slow-channel GOF → quinidine NOT pyridostigmine",
            "COLQ-CMS: slow pupils; pyridostigmine ABSOLUTELY CI; distinguish by CHRNE panel",
            "Neonatal MG (transient): maternal AChR-Ab positive; resolves in weeks",
        ],
        "onset_range_y": (0, 5),
        "slow_pupils": False,
        "episodic_apnea": False,
        "limb_girdle_pattern": False,
        "pyridostigmine_safe": True,
        "three_four_dap_indicated": True,
        "salbutamol_indicated": False,
        "quinidine_indicated": False,
    },

    # ── RAPSN — Rapsyn ──────────────────────────────────────────────────────
    {
        "gene": "RAPSN", "protein": "Receptor-Associated Protein of the Synapse",
        "alias": "Rapsyn; MASC protein; 43K protein; OMIM #616313 (same locus group); OMIM disease #608931; AR; 11p11.2; N88K European founder",
        "aa": "412 aa", "kDa": "43 kDa",
        "gene_class": (
            "RAPSN (rapsyn / MASC) is a peripheral membrane protein essential for clustering AChR at the NMJ. "
            "Rapsyn is stoichiometrically co-expressed with AChR (1:1 ratio) and directly tethers AChR to "
            "the cytoskeletal scaffold via ankyrin repeats. "
            "MECHANISM: rapsyn binds the AChR β and δ subunits and the cytoplasmic scaffold (α-dystrobrevin, "
            "F-actin). LOF mutations → AChR fails to cluster at densities required (>10,000/µm²) for reliable "
            "quantal transmission → reduced MEPP amplitude + reduced end-plate AChR density on electron microscopy. "
            "N88K (c.264A>G, p.Asn88Lys): most common pathogenic allele; founder in North-European, Mexican, "
            "Turkish, Pakistani populations; present in ~50% of RAPSN alleles worldwide. "
            "RAPSN 11p11.2; OMIM gene 601592."
        ),
        "cms_group": "Postsynaptic CMS — AChR-Clustering Defect",
        "cms_type": "AChR-Clustering Deficient Postsynaptic CMS",
        "locus": "11p11.2", "omim_gene": 601592, "omim_disease": 608931,
        "inheritance": "Autosomal Recessive (AR). N88K homozygous or compound heterozygous. N88K is the most frequent CMS allele in European populations.",
        "phenotype": (
            "Onset: neonatal/infancy (most common CMS presenting at birth). Neonatal arthrogryposis (joint "
            "contractures from reduced fetal movement) — DISTINCTIVE for RAPSN. "
            "Respiratory distress at birth (may require ventilation). Episodic myasthenic crisis precipitated "
            "by fever/infection — sudden severe respiratory decompensation. Ptosis. Bulbar weakness. "
            "Limb weakness variable. Response to treatment usually good."
        ),
        "disease": (
            "RAPSN-CMS — Rapsyn Deficiency. Second most common CMS gene. "
            "N88K founder allele: diagnostic key — if RAPSN-CMS suspected, sequence N88K first (rapid, cheap). "
            "Episodic myasthenic crises: fever/infection → rapid decompensation → ICU. "
            "Neonatal arthrogryposis without RAPSN diagnosis is a missed opportunity for life-saving treatment."
        ),
        "treatment_options": [
            "Pyridostigmine: first-line; good response; 30-60 mg TDS-QDS",
            "3,4-DAP: first-line adjunct; increases ACh quantal release",
            "Salbutamol: helpful second-line adjunct",
            "Crisis management: early intubation at first sign of respiratory decompensation (fever lowers crisis threshold significantly)",
            "Prophylactic sick-day plan: written action plan for febrile illness (early hospital presentation, increase pyridostigmine, ICU readiness)",
            "Avoid: aminoglycosides, NMJ blockers, general anesthesia without senior anesthetist briefing",
        ],
        "key_ddx": [
            "CHRNE-CMS: no neonatal arthrogryposis; N88K distinguishes (CHRNE has different mutations)",
            "Neonatal MG (transient): maternal Ab positive; resolves; AChR-Ab workup mandatory",
            "DOK7-CMS: limb-girdle > ocular; salbutamol preferred over pyridostigmine",
            "Arthrogryposis multiplex congenita (AMC): RAPSN-CMS is a treatable cause — do not miss",
            "COLQ-CMS: slow pupils; arthrogryposis possible but slow pupils distinguish",
        ],
        "onset_range_y": (0, 2),
        "slow_pupils": False,
        "episodic_apnea": False,
        "limb_girdle_pattern": False,
        "pyridostigmine_safe": True,
        "three_four_dap_indicated": True,
        "salbutamol_indicated": False,
        "quinidine_indicated": False,
    },

    # ── DOK7 — DOK7 Myasthenia (Limb-Girdle CMS) ────────────────────────────
    {
        "gene": "DOK7", "protein": "Downstream of Kinase 7",
        "alias": "DOK7 myasthenia; limb-girdle CMS; OMIM #610542; AR; 4p16.3; MuSK-activating adaptor; salbutamol-responsive",
        "aa": "504 aa", "kDa": "57 kDa",
        "gene_class": (
            "DOK7 is a cytoplasmic adaptor protein that binds and dimerises the kinase domain of MuSK "
            "(muscle-specific kinase), amplifying agrin-MuSK signalling for AChR clustering. "
            "DOK7 contains a PH domain (membrane anchoring), PTB domain (MuSK binding), and C-terminal "
            "substrate region. MECHANISM: biallelic DOK7 LOF → impaired MuSK activation → "
            "failure to cluster AChR at normal density (end-plate AChR density reduced ~3-fold) → "
            "simplified synaptic folds on EM + simplified end-plate → decremental EMG. "
            "DISTINCTIVE: AChR immunostaining normal (antibodies normal); clinical limb-girdle pattern "
            "with NECK FLEXOR and FACIAL weakness, ptosis variable. "
            "FRAMESHIFT: c.1124_1127dupTGCC (p.Ala376Serfs*30) — most common worldwide allele. "
            "DOK7 4p16.3; OMIM gene 610285."
        ),
        "cms_group": "Postsynaptic CMS — MuSK Signalling Defect (Limb-Girdle)",
        "cms_type": "DOK7 Myasthenia — Limb-Girdle CMS",
        "locus": "4p16.3", "omim_gene": 610285, "omim_disease": 610542,
        "inheritance": "Autosomal Recessive (AR). Biallelic. c.1124_1127dupTGCC most common allele (>50% of pathogenic alleles).",
        "phenotype": (
            "ONSET: variable (neonatal to adult). Limb-girdle pattern: PROXIMAL weakness, "
            "NECK FLEXOR weakness (distinctive — test in clinic). Ptosis in ~60% (variable). "
            "Ophthalmoplegia: unusual (<30%). NO significant ocular involvement in many cases — key DDx from CHRNE. "
            "Bulbar weakness present but often mild. Respiratory involvement: may be severe and disproportionate "
            "to limb weakness. EMG shows decremental response."
        ),
        "disease": (
            "DOK7-CMS — DOK7 Myasthenia (Limb-Girdle CMS). CRITICAL TREATMENT RULE: "
            "PYRIDOSTIGMINE TYPICALLY WORSENS DOK7-CMS (counterproductive; reduces MuSK-activated AChR density). "
            "SALBUTAMOL (oral, 2-4 mg BD-TDS) or EPHEDRINE first-line — β2-agonists upregulate "
            "downstream MuSK-DOK7 pathway, increasing AChR density and NMJ function. "
            "Dramatic clinical improvement with salbutamol is highly characteristic of DOK7-CMS."
        ),
        "treatment_options": [
            "Salbutamol (oral): FIRST-LINE; 2-4 mg BD-TDS; β2-agonist upregulates MuSK-DOK7-AChR density",
            "Ephedrine: first-line alternative; 25-75 mg/day; similar mechanism to salbutamol",
            "PYRIDOSTIGMINE: AVOID — typically WORSENS (counterproductive); causes further simplification of NMJ",
            "3,4-DAP: may transiently help during crisis but NOT first-line for maintenance",
            "Monitor: cardiac (ephedrine tachycardia); bone density (long-term salbutamol)",
            "Avoid: aminoglycosides; NMJ blocking agents; pyridostigmine in maintenance",
        ],
        "key_ddx": [
            "LIMB-GIRDLE MUSCULAR DYSTROPHY (LGMD): CK usually normal in DOK7 (mildly raised possible); SFEMG decrement distinguishes",
            "MuSK-MG (autoimmune): MuSK-Ab positive; autoimmune (not genetic) — different treatment (immunotherapy)",
            "CHRNE-CMS: ocular > limb-girdle; pyridostigmine helps (not worsens); CHRNE panel distinguishes",
            "AGRN-CMS: similar limb-girdle + presynaptic; ephedrine effective in both; AGRN vs DOK7 panel",
            "Seronegative MG: AChR/MuSK/LRP4 negative; SFEMG + panel clarifies",
        ],
        "onset_range_y": (0, 30),
        "slow_pupils": False,
        "episodic_apnea": False,
        "limb_girdle_pattern": True,
        "pyridostigmine_safe": False,
        "three_four_dap_indicated": False,
        "salbutamol_indicated": True,
        "quinidine_indicated": False,
    },

    # ── COLQ — ColQ (AChE-Deficient CMS) ──────────────────────────────────────
    {
        "gene": "COLQ", "protein": "Collagen Q — AChE Anchoring Protein",
        "alias": "ColQ; AChE-deficient CMS; OMIM #603034; AR; 3p24.3; PYRIDOSTIGMINE ABSOLUTELY CONTRAINDICATED; slow pupils PATHOGNOMONIC",
        "aa": "457 aa", "kDa": "51 kDa",
        "gene_class": (
            "COLQ encodes the collagen tail subunit that anchors asymmetric acetylcholinesterase (AChE) to the "
            "synaptic basal lamina of the NMJ via perlecan. AChE in the synapse exists as A12 asymmetric form "
            "(three tetramers attached to three ColQ tails). "
            "MECHANISM: biallelic COLQ LOF → absence of synaptically-anchored AChE → ACh accumulates in the "
            "synaptic cleft and cannot be hydrolysed → persistent channel opening → depolarization block of "
            "the end-plate → paradoxical transmission failure despite EXCESS ACh. "
            "The synaptic cleft is FLOODED with ACh — adding pyridostigmine (AChEI) makes this WORSE "
            "by blocking the already absent enzyme (no effect on ColQ) while activating muscarinic receptors "
            "and inducing cholinergic crisis. "
            "PATHOGNOMONIC: SLOW PUPILLARY LIGHT RESPONSE — diffuse autonomic AChE deficiency "
            "(smooth muscle sphincter pupillae: cholinergic) → slow pupil light reflex. Measure in clinic. "
            "COLQ 3p24.3; OMIM gene 603033."
        ),
        "cms_group": "Synaptic Basal Lamina CMS — AChE-Deficient",
        "cms_type": "AChE-Deficient CMS — COLQ Anchoring Defect",
        "locus": "3p24.3", "omim_gene": 603033, "omim_disease": 603034,
        "inheritance": "Autosomal Recessive (AR). Biallelic LOF. Higher prevalence in Middle-Eastern consanguineous families.",
        "phenotype": (
            "Onset: neonatal to childhood. Generalised weakness + fatigability. Ptosis. "
            "Ophthalmoplegia (common). Respiratory crises. "
            "DISTINCTIVE SIGNS: (1) SLOW PUPILLARY LIGHT RESPONSE — pathognomonic; examine with "
            "torch in dim room — pupil constricts slowly and incompletely. "
            "(2) REPETITIVE COMPOUND MUSCLE ACTION POTENTIAL (RCMAP) on EMG — repetitive stimulation "
            "evokes double/triple compound action potential (unique among CMS types). "
            "Musarinic symptoms possible: excessive secretions, bradycardia."
        ),
        "disease": (
            "COLQ-CMS — AChE-Deficient CMS. CRITICAL TREATMENT RULE: "
            "PYRIDOSTIGMINE ABSOLUTELY CONTRAINDICATED — AChE already absent; AChEI → depolarisation block "
            "→ paradoxical severe worsening → potentially fatal cholinergic crisis. NEVER prescribe. "
            "TREATMENT: Ephedrine ± salbutamol (β2-agonists and sympathomimetics; mechanism: upregulate "
            "NMJ components independent of AChE pathway). EMG finding of RCMAP is diagnostic."
        ),
        "treatment_options": [
            "Ephedrine: FIRST-LINE; 25-75 mg/day; α/β-adrenergic; upregulates NMJ function",
            "Salbutamol (oral): FIRST-LINE alternative; 2-4 mg TDS; similar to ephedrine",
            "PYRIDOSTIGMINE: ABSOLUTELY CONTRAINDICATED — AChE already absent → depolarisation block → fatal worsening",
            "3,4-DAP: GENERALLY CONTRAINDICATED — increases quantal ACh release into already ACh-flooded synapse",
            "Azithromycin: reported in single cases as adjunct (mechanism: NMJ upregulation) — experimental only",
            "Ventilatory support: as required; RCMAP is diagnostic marker for COLQ",
        ],
        "key_ddx": [
            "CHRNE-CMS: no slow pupils; no RCMAP; pyridostigmine safe and effective (opposite of COLQ)",
            "Organophosphate poisoning: also causes AChE inhibition; slow pupils; BUT not genetic, acute onset, toxicological history",
            "DOK7-CMS: limb-girdle pattern; no slow pupils; salbutamol preferred; no RCMAP",
            "Cholinergic crisis (iatrogenic): pyridostigmine overdose mimics — but COLQ-CMS is spontaneous",
            "Other CMS types: RCMAP is almost unique to COLQ-CMS (also AGRN rarely) — specific diagnostic marker",
        ],
        "onset_range_y": (0, 5),
        "slow_pupils": True,
        "episodic_apnea": False,
        "limb_girdle_pattern": False,
        "pyridostigmine_safe": False,
        "three_four_dap_indicated": False,
        "salbutamol_indicated": True,
        "quinidine_indicated": False,
    },

    # ── CHAT — Choline Acetyltransferase (Presynaptic CMS) ─────────────────
    {
        "gene": "CHAT", "protein": "Choline Acetyltransferase",
        "alias": "ChAT; presynaptic CMS; OMIM #254210; AR; 10q11.23; episodic apnea; inter-episodic strength NORMAL",
        "aa": "748 aa", "kDa": "83 kDa",
        "gene_class": (
            "CHAT encodes choline acetyltransferase (ChAT), the enzyme that synthesises acetylcholine (ACh) "
            "from choline and acetyl-CoA in the presynaptic nerve terminal. "
            "MECHANISM: biallelic CHAT LOF → reduced ACh synthesis → reduced quantal content of synaptic "
            "vesicles → impaired NMJ transmission. During sustained neural activity or increased demand "
            "(fever, infection) → ACh stores rapidly depleted → sudden severe NMJ failure → apnoea. "
            "At baseline with infrequent motor activity: ACh resynthesis keeps pace → NORMAL strength. "
            "This explains the PARADOX: inter-episodic strength is NORMAL (or near-normal) while "
            "crises cause life-threatening apnoeic episodes. "
            "Repetitive EMG stimulation at high frequency (>10 Hz) reveals progressive decremental response. "
            "CHAT 10q11.23; OMIM gene 118490."
        ),
        "cms_group": "Presynaptic CMS — ACh Synthesis Defect",
        "cms_type": "ChAT-Deficient Presynaptic CMS — Episodic Apnea",
        "locus": "10q11.23", "omim_gene": 118490, "omim_disease": 254210,
        "inheritance": "Autosomal Recessive (AR). Biallelic LOF. De novo rare.",
        "phenotype": (
            "ONSET: neonatal (apnoea at birth or early infancy). "
            "HALLMARK: EPISODIC SUDDEN APNOEA — precipitated by FEVER, infection, general anaesthesia, "
            "emotion, exercise, or sleep. Between episodes: strength largely NORMAL (may have mild ptosis only). "
            "Neonatal: apnoea + bradycardia + hypotonia → may require ventilatory support from birth. "
            "Infancy: sudden unexpected apnoeic attacks (may mimic ALTE/SUDI). "
            "Childhood/adulthood: fever → sudden respiratory decompensation; patient can be fine minutes earlier. "
            "CRITICAL: misdiagnosis as SUDI (sudden unexpected death in infancy) risk if CHAT not tested."
        ),
        "disease": (
            "CHAT-CMS — ChAT Presynaptic CMS. CRITICAL RISK: fever/illness → sudden apnoeic crisis → "
            "ICU admission mandatory. Written sick-day action plan is life-saving. "
            "3,4-DAP (amifampridine) increases quantal ACh release (presynaptic) — first-line. "
            "Pyridostigmine: second-line (modest benefit; increases available ACh). "
            "PREVENTIVE: all fever/illness → emergency hospital, prophylactic 3,4-DAP increase. "
            "General anaesthesia: EXTREME RISK — briefing of anaesthetist + senior neurologist mandatory."
        ),
        "treatment_options": [
            "3,4-DAP (amifampridine): FIRST-LINE; increases ACh quantal release from presynaptic terminal; 10-20 mg TDS",
            "Pyridostigmine: second-line adjunct; improves ACh availability; 30-60 mg TDS-QDS",
            "SICK-DAY ACTION PLAN: early hospital attendance at any fever (>37.5°C); increase 3,4-DAP; ICU readiness",
            "Prophylactic ventilation: during elective procedures and major illness (BiPAP/NIV precautionary)",
            "TRIGGERS TO AVOID: high fever (manage aggressively with antipyretics), general anaesthesia (use RA/spinal where possible), extreme exercise",
            "Family training: airway management, bag-valve-mask for home use in known patients",
        ],
        "key_ddx": [
            "ALTE (apparent life-threatening event) / SIDS risk: CHAT-CMS is a treatable cause; send CMS panel in ALTE",
            "Central apnea (neurological): MRI brain normal in CHAT-CMS; EMG decrement distinguishes",
            "COLQ-CMS: slow pupils; RCMAP; pyridostigmine CI; no episodic pattern — continuous weakness",
            "RAPSN-CMS: arthrogryposis; no inter-episodic normality; pyridostigmine effective",
            "Mitochondrial disorders: apnea + lactic acidosis; muscle biopsy COX-negative fibres; CHAT-CMS is distinct",
        ],
        "onset_range_y": (0, 1),
        "slow_pupils": False,
        "episodic_apnea": True,
        "limb_girdle_pattern": False,
        "pyridostigmine_safe": True,
        "three_four_dap_indicated": True,
        "salbutamol_indicated": False,
        "quinidine_indicated": False,
    },

    # ── GFPT1 — Glycosylation-Deficient Limb-Girdle CMS ───────────────────
    {
        "gene": "GFPT1", "protein": "Glutamine:Fructose-6-Phosphate Aminotransferase 1",
        "alias": "GFPT1; glycosylation-deficient CMS; limb-girdle CMS-tubular aggregates; OMIM #615120; AR; 2p13.1; hexosamine pathway; tubular aggregates on biopsy",
        "aa": "699 aa", "kDa": "79 kDa",
        "gene_class": (
            "GFPT1 encodes glutamine:fructose-6-phosphate aminotransferase 1 — the rate-limiting enzyme of "
            "the hexosamine biosynthesis pathway (HBP). The HBP produces UDP-GlcNAc, the substrate for "
            "O-GlcNAc modification and N-glycosylation of proteins including α-dystroglycan (α-DG). "
            "MECHANISM: biallelic GFPT1 LOF → impaired N-glycosylation of NMJ-anchoring proteins "
            "(α-DG, AChR, agrin receptors) → failure of NMJ organisation and maintenance → "
            "simplified synaptic folds, reduced AChR density, NMJ transmission failure. "
            "DISTINCTIVE BIOPSY FEATURE: TUBULAR AGGREGATES in muscle fibres (accumulation of "
            "sarcoplasmic reticulum membrane tubules — seen on Gomori or NADH staining) — "
            "present in GFPT1-CMS but NOT in most other CMS types. "
            "GFPT1 2p13.1; OMIM gene 138292."
        ),
        "cms_group": "Glycosylation-Deficient CMS — Hexosamine Pathway",
        "cms_type": "GFPT1-CMS — Limb-Girdle CMS with Tubular Aggregates",
        "locus": "2p13.1", "omim_gene": 138292, "omim_disease": 615120,
        "inheritance": "Autosomal Recessive (AR). Biallelic. Often later onset than other CMS types (late childhood to young adult).",
        "phenotype": (
            "ONSET: late childhood to young adulthood (often 10–25 years — LATER than most CMS). "
            "LIMB-GIRDLE PATTERN: proximal weakness, waddling gait, difficulty climbing stairs. "
            "Ptosis: present in many; ophthalmoplegia: mild or absent. "
            "Bulbar involvement: variable. Respiratory: mild to moderate. "
            "Fatigability: marked. "
            "MUSCLE BIOPSY (if done): tubular aggregates on Gomori trichrome or NADH stain — DISTINCTIVE. "
            "EMG: decremental response. CK: usually normal or mildly elevated."
        ),
        "disease": (
            "GFPT1-CMS — Glycosylation-Deficient Limb-Girdle CMS. "
            "Unlike DOK7-CMS, GFPT1-CMS responds to standard AChEI therapy. "
            "Pyridostigmine ± 3,4-DAP first-line with good clinical response. "
            "Tubular aggregates on muscle biopsy + limb-girdle CMS + AChEI response = GFPT1 until proven otherwise. "
            "Late adolescent/young adult onset can lead to delayed diagnosis (misdiagnosed as LGMD)."
        ),
        "treatment_options": [
            "Pyridostigmine: FIRST-LINE; responds well (unlike DOK7); 30-60 mg TDS-QDS",
            "3,4-DAP: second-line adjunct; useful for incomplete pyridostigmine response",
            "Salbutamol: not typically first-line but may provide modest benefit",
            "Physiotherapy: proximal strengthening; stair training; avoid prolonged inactivity",
            "Avoid: aminoglycosides; NMJ blocking agents under anesthesia",
            "Genetic counselling: AR; 25% recurrence; panel testing for GFPT1 + other glycosylation-CMS genes (ALG2, ALG14, GMPPB, DPAGT1)",
        ],
        "key_ddx": [
            "LGMD (Limb-Girdle Muscular Dystrophy): CK elevation in LGMD; no EMG decrement; GFPT1 does not elevate CK significantly",
            "DOK7-CMS: also limb-girdle CMS; BUT pyridostigmine WORSENS DOK7 while HELPS GFPT1; salbutamol first for DOK7",
            "Periodic Paralysis: tubular aggregates also seen; but episodic not fatigable pattern",
            "Mitochondrial myopathy: ragged-red fibres on Gomori (not tubular aggregates) in mito; no EMG decrement",
            "Other glycosylation CMS (GMPPB, DPAGT1, ALG2, ALG14): similar phenotype; CMS gene panel distinguishes",
        ],
        "onset_range_y": (10, 30),
        "slow_pupils": False,
        "episodic_apnea": False,
        "limb_girdle_pattern": True,
        "pyridostigmine_safe": True,
        "three_four_dap_indicated": True,
        "salbutamol_indicated": False,
        "quinidine_indicated": False,
    },

    # ── AGRN — Agrin (Presynaptic CMS) ────────────────────────────────────
    {
        "gene": "AGRN", "protein": "Agrin",
        "alias": "Agrin; presynaptic CMS; OMIM #615120 (series); OMIM disease #615120; AR; 1p36.33; NMJ organiser; MuSK-DOK7-rapsyn cascade activator",
        "aa": "2045 aa", "kDa": "225 kDa",
        "gene_class": (
            "AGRN encodes agrin — a large heparan sulphate proteoglycan secreted by motor nerve terminals "
            "into the synaptic basal lamina. Agrin activates MuSK kinase (via LRP4 co-receptor) → "
            "MuSK autophosphorylation → DOK7 binding → rapsyn activation → AChR clustering. "
            "Agrin is the primary NMJ organising signal from the motor neuron to the muscle. "
            "MECHANISM: biallelic AGRN LOF → impaired MuSK activation → AChR fails to cluster → "
            "simplified NMJ with reduced quantal content and reduced end-plate AChR density. "
            "The NMJ remains a simplified structure resembling embryonic non-clustered junctions. "
            "DISTINCTIVE: presynaptic release of agrin — loss causes defective CLUSTER FORMATION, "
            "different from COLQ (AChE anchor) or RAPSN (direct AChR clustering protein). "
            "AGRN 1p36.33; OMIM gene 103320."
        ),
        "cms_group": "Presynaptic / Synaptic CMS — NMJ Organising Factor",
        "cms_type": "Agrin-Deficient CMS — Presynaptic NMJ Organiser Defect",
        "locus": "1p36.33", "omim_gene": 103320, "omim_disease": 615120,
        "inheritance": "Autosomal Recessive (AR). Biallelic. Rare — fewer than 30 families reported worldwide.",
        "phenotype": (
            "Onset: neonatal to early childhood. Proximal > distal weakness (limb-girdle-like). "
            "Ptosis and ophthalmoplegia: variable. Bulbar symptoms. Respiratory involvement. "
            "Mild facial weakness. EMG: decremental response. "
            "Phenotype overlaps with DOK7-CMS (both involve MuSK pathway activation). "
            "Muscle biopsy: simplified NMJ structure on electron microscopy; "
            "AChR staining normal density (unlike RAPSN/CHRNE)."
        ),
        "disease": (
            "AGRN-CMS — Agrin-Deficient Presynaptic CMS. Very rare. "
            "Treatment: ephedrine/salbutamol (β2-agonists) as for DOK7-CMS (same MuSK pathway). "
            "3,4-DAP: may help transiently (increases presynaptic ACh release to compensate). "
            "Pyridostigmine: limited benefit; some patients show modest improvement. "
            "Genetic diagnosis: essential — AGRN mutations confirm diagnosis and guide therapy."
        ),
        "treatment_options": [
            "Ephedrine: first-line; 25-75 mg/day; similar mechanism to DOK7-CMS (β-adrenergic MuSK upregulation)",
            "Salbutamol (oral): first-line alternative; 2-4 mg TDS",
            "3,4-DAP: useful adjunct (presynaptic mechanism); 10-20 mg TDS",
            "Pyridostigmine: modest benefit in some; worth trial if ephedrine/salbutamol insufficient",
            "Physiotherapy: essential for proximal strengthening",
            "Avoid: aminoglycosides; NMJ blockers under anesthesia; briefing of anaesthetist for all procedures",
        ],
        "key_ddx": [
            "DOK7-CMS: same MuSK pathway → very similar phenotype; panel distinguishes (DOK7 vs AGRN genes)",
            "LRP4-CMS: LRP4 is the MuSK co-receptor for agrin; very similar phenotype; panel separates",
            "MuSK-MG (autoimmune): MuSK-Ab positive; responds to immunotherapy; NOT genetic",
            "RAPSN-CMS: more neonatal + arthrogryposis; N88K founder; AChR cluster protein (downstream of agrin)",
            "CHRNE-CMS: ocular more prominent; pyridostigmine first-line (unlike AGRN)",
        ],
        "onset_range_y": (0, 10),
        "slow_pupils": False,
        "episodic_apnea": False,
        "limb_girdle_pattern": True,
        "pyridostigmine_safe": True,
        "three_four_dap_indicated": True,
        "salbutamol_indicated": True,
        "quinidine_indicated": False,
    },

    # ── SCN4A — Nav1.4 Myasthenic Sodium-Channel CMS ────────────────────────
    {
        "gene": "SCN4A", "protein": "Voltage-Gated Sodium Channel Nav1.4",
        "alias": "Nav1.4; SCN4A-CMS; sodium channel myasthenic syndrome; cold-sensitive; OMIM #616069; AD or AR; 17q23.3; quinidine treatment",
        "aa": "1836 aa", "kDa": "208 kDa",
        "gene_class": (
            "SCN4A encodes Nav1.4, the principal voltage-gated sodium channel of skeletal muscle. "
            "Nav1.4 generates the action potential of muscle fibres. "
            "In CMS, SCN4A GOF mutations cause SLOW-CHANNEL-LIKE MYASTHENIC SYNDROME "
            "(distinct from myotonia/periodic paralysis). "
            "MECHANISM: selected SCN4A GOF variants → prolonged inactivation → persistent Na⁺ influx "
            "→ membrane depolarisation block → failure of muscle fibre to respond to ACh → "
            "myasthenic phenotype WITHOUT direct NMJ structural defect. "
            "LOF variants (AR biallelic): Nav1.4 hypo-excitability → flaccid paralysis crises "
            "± COLD SENSITIVITY (cold lowers membrane potential, worsens LOF phenotype). "
            "KEY PHARMACOLOGY: GOF SCN4A-CMS → quinidine (Na⁺-channel open-channel blocker) or "
            "mexiletine (Na⁺-channel stabiliser) — same as for slow-channel syndrome and myotonia. "
            "Pyridostigmine: CONTRAINDICATED in GOF type (already in depolarisation block). "
            "SCN4A 17q23.3; OMIM gene 603967."
        ),
        "cms_group": "Ion-Channel CMS — Nav1.4 Myasthenic Syndrome",
        "cms_type": "SCN4A-CMS — Sodium Channel Myasthenic Syndrome",
        "locus": "17q23.3", "omim_gene": 603967, "omim_disease": 616069,
        "inheritance": "Autosomal Dominant (GOF variants) or Autosomal Recessive (LOF/LOF compound heterozygous). De novo possible.",
        "phenotype": (
            "ONSET: variable (neonatal to adult). "
            "GOF type: episodic weakness + stiffness; cold-induced worsening (paramyotonic features); "
            "myasthenic fatigability; EMG: decremental + myotonic discharges. "
            "LOF type: episodic flaccid paralysis (hypokalaemia-independent); cold-sensitive; "
            "crises of generalised paralysis with preserved consciousness; inter-ictal strength normal. "
            "Respiratory crises can occur. NOT typical limb-girdle, ocular, or neonatal arthrogryposis pattern. "
            "COLD SENSITIVITY: diagnostic clue — symptoms precipitated by cold exposure or cold meals."
        ),
        "disease": (
            "SCN4A-CMS — Sodium Channel Myasthenic Syndrome. TREATMENT DIVERGES FROM OTHER CMS: "
            "QUINIDINE or MEXILETINE (Na⁺ channel stabilisers/blockers) — specifically for GOF type. "
            "PYRIDOSTIGMINE: CONTRAINDICATED in GOF SCN4A-CMS (depolarisation block; worsens). "
            "LOF type: mexiletine may help; avoid cold; carbonic anhydrase inhibitors (acetazolamide) "
            "sometimes beneficial. "
            "IMPORTANT DDx: SCN4A GOF also causes paramyotonia congenita and HyperkPP2 (different phenotypes). "
            "CMS phenotype is the 'myasthenic variant' of SCN4A disease spectrum."
        ),
        "treatment_options": [
            "Quinidine: FIRST-LINE for GOF SCN4A-CMS; open-channel Na⁺ blocker; reduces prolonged depolarisation; monitor QTc",
            "Mexiletine: alternative Na⁺-channel stabiliser; 150-300 mg TDS; fewer cardiac side effects than quinidine",
            "PYRIDOSTIGMINE: CONTRAINDICATED in GOF type (depolarisation block → worsening)",
            "3,4-DAP: generally not indicated (presynaptic mechanism; not useful for membrane Na⁺ channel defect)",
            "Acetazolamide: may help LOF type (hyperpolarising shift); not first-line",
            "COLD AVOIDANCE: mandatory; warm clothing; avoid cold swimming, cold food/drinks precipitating crisis",
        ],
        "key_ddx": [
            "Paramyotonia Congenita (PMC/SCN4A GOF): cold stiffness > weakness; no NMJ defect; mexiletine; no myasthenic EMG",
            "HyperkPP2 (SCN4A GOF): episodic paralysis + high K+; attacks post-exercise/fasting; no NMJ decrement",
            "Myasthenia Gravis (MG): autoimmune Ab positive; no cold sensitivity as prominent feature",
            "COLQ-CMS: slow pupils; RCMAP; no cold sensitivity; ephedrine treatment",
            "Periodic Paralysis (CACNA1S/KCNJ2): K+ abnormalities during attacks; different channel; different treatment",
        ],
        "onset_range_y": (0, 40),
        "slow_pupils": False,
        "episodic_apnea": False,
        "limb_girdle_pattern": False,
        "pyridostigmine_safe": False,
        "three_four_dap_indicated": False,
        "salbutamol_indicated": False,
        "quinidine_indicated": True,
    },
]


def _gen_patients(gene_data: dict, seed: int) -> list:
    """Generate 40 synthetic CMS patients for a single gene."""
    rng = random.Random(seed)
    gene = gene_data["gene"]
    onset_lo, onset_hi = gene_data["onset_range_y"]
    patients = []
    for i in range(40):
        onset = round(rng.uniform(onset_lo, max(onset_lo + 0.5, onset_hi)), 1)
        # severity weighting
        r = rng.random()
        if gene in ("COLQ", "CHAT", "SCN4A"):
            sev = "Severe" if r < 0.40 else ("Moderate" if r < 0.72 else "Mild")
        elif gene in ("DOK7", "AGRN"):
            sev = "Severe" if r < 0.25 else ("Moderate" if r < 0.65 else "Mild")
        elif gene == "GFPT1":
            sev = "Severe" if r < 0.15 else ("Moderate" if r < 0.55 else "Mild")
        else:
            sev = "Severe" if r < 0.30 else ("Moderate" if r < 0.65 else "Mild")

        ptosis = rng.random() < (0.85 if gene in ("CHRNE", "RAPSN", "COLQ") else
                                  0.60 if gene in ("CHAT", "GFPT1") else
                                  0.35 if gene == "DOK7" else 0.50)
        ophthalmo = rng.random() < (0.65 if gene in ("COLQ",) else
                                     0.45 if gene in ("CHRNE", "RAPSN") else 0.25)
        bulbar = rng.random() < (0.75 if gene in ("CHAT", "RAPSN") else
                                  0.55 if gene in ("CHRNE", "COLQ") else 0.35)
        resp_crisis = rng.random() < (0.80 if gene == "CHAT" else
                                       0.55 if gene in ("RAPSN", "COLQ") else
                                       0.35 if gene in ("CHRNE",) else 0.20)
        limb_girdle = rng.random() < (0.90 if gene in ("DOK7", "GFPT1", "AGRN") else 0.35)
        arthrogryposis = rng.random() < (0.45 if gene == "RAPSN" else 0.05)
        slow_pupils_obs = rng.random() < (0.92 if gene == "COLQ" else 0.02)
        cold_sensitivity = rng.random() < (0.75 if gene == "SCN4A" else 0.05)
        decrement = round(rng.uniform(15, 55) if sev in ("Moderate", "Severe") else rng.uniform(5, 25), 1)
        emg_rcmap = gene == "COLQ" and rng.random() < 0.88

        # treatment response flag
        if gene in ("CHRNE", "RAPSN", "GFPT1"):
            treatment = "Pyridostigmine ± 3,4-DAP"
        elif gene in ("DOK7", "AGRN"):
            treatment = "Salbutamol/Ephedrine"
        elif gene == "COLQ":
            treatment = "Ephedrine/Salbutamol (NO pyridostigmine)"
        elif gene == "CHAT":
            treatment = "3,4-DAP + sick-day plan"
        else:
            treatment = "Quinidine/Mexiletine"

        pid = f"CMS-{gene}-{seed}-{i+1:03d}"
        sex = rng.choice(["M", "F"])
        patients.append({
            "id": pid, "gene": gene, "sex": sex,
            "onset_age_y": onset, "severity": sev,
            "ptosis": ptosis, "ophthalmoplegia": ophthalmo, "bulbar_weakness": bulbar,
            "respiratory_crisis": resp_crisis, "limb_girdle_pattern": limb_girdle,
            "arthrogryposis": arthrogryposis, "slow_pupils": slow_pupils_obs,
            "cold_sensitivity": cold_sensitivity, "emg_rcmap": emg_rcmap,
            "emg_decrement_pct": decrement, "current_treatment": treatment,
            "inheritance": gene_data["inheritance"].split(".")[0],
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gene_data in enumerate(CMS_GENES):
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
    ptosis_n = sum(1 for p in patients if p["ptosis"])
    ophthalmo_n = sum(1 for p in patients if p["ophthalmoplegia"])
    bulbar_n = sum(1 for p in patients if p["bulbar_weakness"])
    resp_n = sum(1 for p in patients if p["respiratory_crisis"])
    lg_n = sum(1 for p in patients if p["limb_girdle_pattern"])
    arthrog_n = sum(1 for p in patients if p["arthrogryposis"])
    slow_p_n = sum(1 for p in patients if p["slow_pupils"])
    cold_n = sum(1 for p in patients if p["cold_sensitivity"])
    for p in patients:
        sev[p["severity"]] += 1

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 1)

    return {
        "atlas": "CMS-Atlas",
        "full_name": "Complete 8-Gene Congenital Myasthenic Syndromes (CMS) Atlas",
        "subtitle": "CHRNE·RAPSN·DOK7·COLQ·CHAT·GFPT1·AGRN·SCN4A — 320 patients (8×40, seeds 1022–1029)",
        "description": (
            "Comprehensive atlas of the 8 most clinically important Congenital Myasthenic Syndrome (CMS) genes. "
            "CMS = inherited NMJ transmission disorders (NOT autoimmune — AChR/MuSK/LRP4 antibodies NEGATIVE). "
            "Covers postsynaptic AChR-deficient (CHRNE), AChR-clustering (RAPSN), MuSK-pathway (DOK7, AGRN), "
            "AChE-anchoring (COLQ), presynaptic ACh-synthesis (CHAT), glycosylation (GFPT1), "
            "and ion-channel (SCN4A) subtypes. "
            "CRITICAL TREATMENT DISTINCTIONS: COLQ = pyridostigmine ABSOLUTELY CONTRAINDICATED; "
            "DOK7 = pyridostigmine worsens, salbutamol first; SCN4A-GOF = quinidine; "
            "CHAT = fever-trigger sick-day plan mandatory."
        ),
        "total_patients": n,
        "genes_covered": len(CMS_GENES),
        "patients_per_gene": 40,
        "seed_range": "1022–1029",
        "gene_list": [g["gene"] for g in CMS_GENES],
        "cms_category_breakdown": {
            "Postsynaptic AChR-Deficient": ["CHRNE"],
            "Postsynaptic AChR-Clustering": ["RAPSN"],
            "Postsynaptic MuSK-Pathway (limb-girdle)": ["DOK7"],
            "Synaptic Basal Lamina AChE-Deficient": ["COLQ"],
            "Presynaptic ACh-Synthesis": ["CHAT"],
            "Glycosylation-Deficient": ["GFPT1"],
            "Presynaptic NMJ-Organiser": ["AGRN"],
            "Ion-Channel (Nav1.4)": ["SCN4A"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "clinical_features_prevalence": {
            "ptosis_pct": round(100 * ptosis_n / n, 1),
            "ophthalmoplegia_pct": round(100 * ophthalmo_n / n, 1),
            "bulbar_weakness_pct": round(100 * bulbar_n / n, 1),
            "respiratory_crisis_pct": round(100 * resp_n / n, 1),
            "limb_girdle_pattern_pct": round(100 * lg_n / n, 1),
            "arthrogryposis_pct": round(100 * arthrog_n / n, 1),
            "slow_pupils_pct": round(100 * slow_p_n / n, 1),
            "cold_sensitivity_pct": round(100 * cold_n / n, 1),
        },
        "key_teaching_points": [
            "CMS ≠ MG: CMS is GENETIC (no AChR/MuSK/LRP4 antibodies); MG is AUTOIMMUNE — send antibodies before age 5 in apparent seronegative MG",
            "COLQ-CMS: pyridostigmine ABSOLUTELY CONTRAINDICATED — AChE absent → depolarisation block; SLOW PUPILS pathognomonic; use ephedrine/salbutamol",
            "DOK7-CMS: pyridostigmine WORSENS (avoid); SALBUTAMOL first-line; limb-girdle + neck flexor weakness; c.1124_1127dup most common allele",
            "CHAT-CMS: inter-episodic strength NORMAL; fever → sudden apnoea crisis → ICU; written sick-day plan life-saving",
            "RAPSN-CMS: N88K most common pathogenic allele; neonatal arthrogryposis TREATABLE cause; episodic crisis with fever",
            "SCN4A-CMS (GOF): QUINIDINE/mexiletine; pyridostigmine CI; cold-sensitive; same gene as paramyotonia/HyperkPP",
            "GFPT1-CMS: glycosylation-deficient; limb-girdle + tubular aggregates on biopsy; responds to pyridostigmine (unlike DOK7)",
            "AGRN-CMS: presynaptic agrin; MuSK-pathway (like DOK7); ephedrine/salbutamol; extremely rare",
        ],
        "drug_alerts": [
            "COLQ-CMS: Pyridostigmine ABSOLUTELY CONTRAINDICATED — fatal cholinergic crisis if given",
            "DOK7-CMS: Pyridostigmine WORSENS — use salbutamol/ephedrine instead",
            "SCN4A-CMS (GOF): Pyridostigmine CONTRAINDICATED — use quinidine/mexiletine",
            "ALL CMS: Aminoglycoside antibiotics CONTRAINDICATED — NMJ blockade worsens all CMS types",
            "ALL CMS: Non-depolarising NMJ blockers (vecuronium etc.) — extreme sensitivity; use with caution under expert anaesthesia",
            "CHAT-CMS: General anaesthesia EXTREME RISK — senior anaesthetist + neurologist mandatory",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    gene_profiles = []
    for gene_data in CMS_GENES:
        gene_pts = [p for p in patients if p["gene"] == gene_data["gene"]]
        n = len(gene_pts)
        sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for p in gene_pts:
            sev[p["severity"]] += 1
        gene_profiles.append({
            "gene": gene_data["gene"],
            "protein": gene_data["protein"],
            "alias": gene_data["alias"],
            "locus": gene_data["locus"],
            "omim_gene": gene_data["omim_gene"],
            "omim_disease": gene_data["omim_disease"],
            "inheritance": gene_data["inheritance"],
            "cms_group": gene_data["cms_group"],
            "cms_type": gene_data["cms_type"],
            "aa": gene_data["aa"],
            "kDa": gene_data["kDa"],
            "gene_class": gene_data["gene_class"],
            "phenotype": gene_data["phenotype"],
            "disease": gene_data["disease"],
            "treatment_options": gene_data["treatment_options"],
            "key_ddx": gene_data["key_ddx"],
            "onset_range_y": list(gene_data["onset_range_y"]),
            "n_patients": n,
            "pyridostigmine_safe": gene_data["pyridostigmine_safe"],
            "three_four_dap_indicated": gene_data["three_four_dap_indicated"],
            "salbutamol_indicated": gene_data["salbutamol_indicated"],
            "quinidine_indicated": gene_data["quinidine_indicated"],
            "slow_pupils": gene_data["slow_pupils"],
            "episodic_apnea": gene_data["episodic_apnea"],
            "limb_girdle_pattern": gene_data["limb_girdle_pattern"],
            "severity_distribution": {
                "mild_pct": round(100 * sev["Mild"] / n, 1),
                "moderate_pct": round(100 * sev["Moderate"] / n, 1),
                "severe_pct": round(100 * sev["Severe"] / n, 1),
            },
            "clinical_features": {
                "ptosis_pct": round(100 * sum(1 for p in gene_pts if p["ptosis"]) / n, 1),
                "ophthalmoplegia_pct": round(100 * sum(1 for p in gene_pts if p["ophthalmoplegia"]) / n, 1),
                "bulbar_weakness_pct": round(100 * sum(1 for p in gene_pts if p["bulbar_weakness"]) / n, 1),
                "respiratory_crisis_pct": round(100 * sum(1 for p in gene_pts if p["respiratory_crisis"]) / n, 1),
                "limb_girdle_pct": round(100 * sum(1 for p in gene_pts if p["limb_girdle_pattern"]) / n, 1),
                "arthrogryposis_pct": round(100 * sum(1 for p in gene_pts if p["arthrogryposis"]) / n, 1),
                "slow_pupils_pct": round(100 * sum(1 for p in gene_pts if p["slow_pupils"]) / n, 1),
                "cold_sensitivity_pct": round(100 * sum(1 for p in gene_pts if p["cold_sensitivity"]) / n, 1),
            },
            "sample_patients": gene_pts[:3],
        })
    return {
        "atlas": "CMS-Atlas",
        "genes": gene_profiles,
        "total_patients": len(patients),
    }


def get_definitions() -> dict:
    return {
        "atlas": "CMS-Atlas",
        "definitions": [
            {
                "term": "Congenital Myasthenic Syndrome (CMS)",
                "definition": (
                    "Heterogeneous group of inherited neuromuscular junction (NMJ) disorders caused by "
                    "mutations in genes encoding proteins essential for NMJ structure and function. "
                    "CMS is GENETIC (not autoimmune) — acetylcholine receptor (AChR), MuSK, and LRP4 "
                    "antibodies are NEGATIVE. Presenting features: ptosis, ophthalmoplegia, bulbar weakness, "
                    "fatigable limb weakness, respiratory crises. Onset: birth to adulthood."
                ),
            },
            {
                "term": "Neuromuscular Junction (NMJ)",
                "definition": (
                    "Synapse between motor neuron axon terminal and skeletal muscle fibre. "
                    "Presynaptic: ACh synthesis (ChAT), storage (vesicles), and release (quantal). "
                    "Synaptic cleft: ACh diffusion; AChE (anchored by ColQ) hydrolyses ACh. "
                    "Postsynaptic: AChR clusters (maintained by rapsyn/DOK7/MuSK/agrin); "
                    "ACh binds → channel opens → end-plate potential → muscle action potential."
                ),
            },
            {
                "term": "Pyridostigmine (AChEI)",
                "definition": (
                    "Acetylcholinesterase inhibitor — the most common CMS treatment. "
                    "Mechanism: inhibits AChE → ACh accumulates in synaptic cleft → more AChR activation. "
                    "SAFE in: CHRNE, RAPSN, GFPT1, CHAT, AGRN (with caveats). "
                    "ABSOLUTELY CONTRAINDICATED: COLQ-CMS (AChE already absent → depolarisation block → fatal). "
                    "WORSENS: DOK7-CMS (reduces NMJ function by reducing MuSK-DOK7 dependent AChR density). "
                    "CONTRAINDICATED: SCN4A-CMS GOF type (depolarisation block)."
                ),
            },
            {
                "term": "3,4-DAP (Amifampridine / Firdapse)",
                "definition": (
                    "3,4-Diaminopyridine — blocks presynaptic voltage-gated K⁺ channels → "
                    "prolonged action potential → increased Ca²⁺ influx → increased ACh quantal release. "
                    "Indicated in: CHRNE, RAPSN, CHAT, GFPT1, AGRN CMS. "
                    "Generally not useful in: COLQ-CMS (flooding synapse with more ACh worsens depolarisation block). "
                    "Monitoring: QTc; seizure risk at high doses; avoid in epilepsy."
                ),
            },
            {
                "term": "Salbutamol / Ephedrine (β2-agonists in CMS)",
                "definition": (
                    "β2-adrenergic agonists — first-line treatment in DOK7-CMS and COLQ-CMS. "
                    "Mechanism: activate β2-adrenergic receptor → cAMP → protein kinase A → "
                    "upregulate HDAC4/PGC-1α pathway → increase utrophin and AChR expression; "
                    "may directly upregulate MuSK-DOK7 NMJ signalling cascade. "
                    "Oral salbutamol (2-4 mg TDS) or ephedrine (25-75 mg/day). "
                    "FIRST-LINE in DOK7-CMS (pyridostigmine worsens). "
                    "FIRST-LINE in COLQ-CMS (pyridostigmine absolutely CI)."
                ),
            },
            {
                "term": "Quinidine / Mexiletine (in CMS)",
                "definition": (
                    "Na⁺-channel open-channel blockers. "
                    "Quinidine: blocks prolonged-open Nav1.4 channels in SCN4A-GOF CMS (and slow-channel syndrome); "
                    "reduces depolarisation block; QTc monitoring mandatory (Class IA antiarrhythmic). "
                    "Mexiletine: similar mechanism; fewer cardiac side effects; 150-300 mg TDS. "
                    "CONTRAINDICATED: COLQ-CMS (no channel defect). NOT useful in most postsynaptic AChR-deficient CMS."
                ),
            },
            {
                "term": "AChE (Acetylcholinesterase)",
                "definition": (
                    "Enzyme that hydrolyses acetylcholine (ACh → choline + acetate) in the synaptic cleft. "
                    "Anchored to synaptic basal lamina by ColQ (COLQ gene). "
                    "In COLQ-CMS: ColQ absent → AChE not anchored → ACh not degraded → "
                    "persistent depolarisation → depolarisation block → myasthenic weakness. "
                    "DIAGNOSTIC: RCMAP (repetitive compound muscle action potential) on EMG is a hallmark "
                    "of AChE deficiency; SLOW PUPIL response (autonomic AChE deficiency)."
                ),
            },
            {
                "term": "RCMAP (Repetitive CMAP)",
                "definition": (
                    "Repetitive compound muscle action potential — an EMG finding where a single nerve "
                    "stimulus evokes 2 or more muscle action potentials separated by ~5-10 ms. "
                    "Pathognomonic for AChE-deficient CMS (COLQ-CMS). "
                    "Mechanism: persistent ACh in synapse (no AChE) activates AChR multiple times "
                    "as the membrane re-excites before the ACh clears. "
                    "Also very rarely seen in AGRN-CMS. Diagnostic EMG finding."
                ),
            },
            {
                "term": "Rapsyn (RAPSN)",
                "definition": (
                    "Receptor-associated protein of the synapse (43K protein). "
                    "Peripheral membrane protein stoichiometrically co-expressed 1:1 with AChR. "
                    "Directly tethers AChR to cytoskeletal scaffold via ankyrin repeats. "
                    "N88K (Asn→Lys): most common RAPSN pathogenic allele worldwide (North-European, Turkish, Mexican). "
                    "LOF → AChR clustering failure → AChR density < required threshold → CMS."
                ),
            },
            {
                "term": "DOK7 Myasthenia",
                "definition": (
                    "Limb-girdle CMS caused by biallelic DOK7 mutations. "
                    "DOK7 is a cytoplasmic adaptor that activates MuSK kinase → AChR clustering. "
                    "c.1124_1127dupTGCC: most common pathogenic allele (>50% of alleles worldwide). "
                    "PHENOTYPE: proximal > distal weakness; NECK FLEXORS weak (clinically test specifically); "
                    "ptosis variable; NO significant ophthalmoplegia. "
                    "TREATMENT: salbutamol/ephedrine FIRST-LINE; pyridostigmine WORSENS."
                ),
            },
            {
                "term": "ColQ (COLQ)",
                "definition": (
                    "Collagenic tail subunit that forms the A12 form of AChE (three AChE tetramers + three ColQ tails). "
                    "Anchors AChE to synaptic basal lamina via perlecan. "
                    "Without ColQ: AChE is not anchored; ACh not degraded; persistent depolarisation → block. "
                    "PATHOGNOMONIC SIGNS: (1) SLOW PUPIL light response (cholinergic pupil — test in clinic); "
                    "(2) RCMAP on EMG. "
                    "TREATMENT: ephedrine/salbutamol ONLY. Pyridostigmine ABSOLUTELY CI."
                ),
            },
            {
                "term": "MuSK (Muscle-Specific Kinase) Pathway in CMS",
                "definition": (
                    "MuSK is the receptor tyrosine kinase activated by agrin (AGRN) via LRP4 co-receptor. "
                    "MuSK activation → phosphorylates DOK7 → rapsyn activation → AChR clustering. "
                    "Genes in this pathway: AGRN (agrin ligand), LRP4 (co-receptor), MUSK (kinase), "
                    "DOK7 (kinase adaptor), RAPSN (cluster scaffolding). "
                    "CMS in DOK7 and AGRN involve this pathway (both treated with salbutamol/ephedrine). "
                    "MuSK-MG (autoimmune) is different: autoantibodies — DO NOT CONFUSE with MUSK/DOK7 CMS (genetic)."
                ),
            },
            {
                "term": "Agrin (AGRN)",
                "definition": (
                    "Large heparan sulphate proteoglycan (2045 aa) secreted by motor nerve terminals. "
                    "Neural agrin activates MuSK (via LRP4) → NMJ organisation. "
                    "AGRN-CMS: biallelic LOF → impaired MuSK signalling → simplified NMJ. "
                    "Very rare (~<30 families worldwide). Treatment: ephedrine/salbutamol (same as DOK7-CMS). "
                    "Presynaptic origin (motor nerve) — the signal FROM neuron TO muscle for NMJ formation."
                ),
            },
            {
                "term": "ChAT (Choline Acetyltransferase — CHAT)",
                "definition": (
                    "Presynaptic enzyme synthesising ACh from choline + acetyl-CoA. "
                    "CHAT-CMS: biallelic LOF → reduced ACh synthesis → quantal content low at baseline "
                    "but compensated at rest; CRISIS when demand exceeds synthesis capacity "
                    "(fever, infection, anaesthesia, exercise). "
                    "PATHOGNOMONIC: inter-episodic strength near-NORMAL → sudden apnoea crisis. "
                    "HIGH-FREQUENCY EMG decrement (>10 Hz) reveals defect. "
                    "MISSED DIAGNOSIS RISK: SUDI/ALTE in infants. TREATABLE."
                ),
            },
            {
                "term": "GFPT1 (Glutamine:Fructose-6-Phosphate Aminotransferase 1)",
                "definition": (
                    "Rate-limiting enzyme of the hexosamine biosynthesis pathway (HBP). "
                    "Produces UDP-GlcNAc → N-glycosylation of NMJ proteins (α-dystroglycan, AChR). "
                    "GFPT1-CMS: biallelic LOF → impaired glycosylation → NMJ instability. "
                    "DISTINCTIVE BIOPSY: tubular aggregates (Gomori/NADH stain) in muscle fibres. "
                    "Later onset than most CMS (teens/young adult). Limb-girdle pattern. "
                    "RESPONDS to pyridostigmine (UNLIKE DOK7-CMS which worsens)."
                ),
            },
            {
                "term": "Nav1.4 (SCN4A) Myasthenic Syndrome",
                "definition": (
                    "Rare CMS caused by SCN4A mutations (same gene as paramyotonia congenita / HyperkPP). "
                    "GOF: prolonged Na⁺ channel inactivation → persistent depolarisation → block → myasthenia. "
                    "LOF: Nav1.4 hypo-excitability → flaccid paralysis crises, cold-sensitive. "
                    "COLD SENSITIVITY: precipitates weakness (cold lowers membrane potential in LOF). "
                    "TREATMENT: quinidine or mexiletine (Na⁺-channel stabilisers). "
                    "PYRIDOSTIGMINE CI in GOF type (worsens depolarisation block)."
                ),
            },
            {
                "term": "Arthrogryposis in CMS",
                "definition": (
                    "Joint contractures (arthrogryposis) from reduced fetal movement in utero. "
                    "Indicates onset during fetal life. "
                    "TREATABLE cause of neonatal arthrogryposis: RAPSN-CMS (most common CMS with arthrogryposis). "
                    "Also COLQ-CMS and CHRNE-CMS (severe alleles). "
                    "CLINICAL RULE: neonatal arthrogryposis → send full CMS gene panel + "
                    "AChR antibodies (to exclude neonatal MG) — missing CMS = denying effective treatment."
                ),
            },
            {
                "term": "Aminoglycosides in CMS",
                "definition": (
                    "Aminoglycoside antibiotics (gentamicin, tobramycin, amikacin, neomycin) block "
                    "presynaptic ACh release AND postsynaptic AChR at high doses. "
                    "CONTRAINDICATED in ALL CMS subtypes — may precipitate severe myasthenic crisis. "
                    "Substitute: β-lactams (penicillins, cephalosporins) or fluoroquinolones "
                    "(but fluoroquinolones also have some NMJ-blocking activity — use cautiously)."
                ),
            },
        ],
    }
