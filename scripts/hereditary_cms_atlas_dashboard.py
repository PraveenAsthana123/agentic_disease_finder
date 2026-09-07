#!/usr/bin/env python3
"""Hereditary-CMS-Atlas — Complete 8-Gene Hereditary Congenital Myasthenic Syndrome Atlas
CHRNE   (AChR epsilon-subunit; 473 aa; 22q11.21; AR;
         CMS4 — endplate AChR deficiency; most common AR CMS in Europe/Middle East;
         pyridostigmine 1st line; 3,4-DAP adjunct; seed SEED_BASE+0) ·
RAPSN   (Rapsyn; 412 aa; 11p11.2; AR;
         CMS11 — post-synaptic AChR clustering; neonatal-onset apneic crises;
         antenatal fetal akinesia deformation sequence; pyridostigmine + 3,4-DAP; seed SEED_BASE+1) ·
DOK7    (Dok-7; 504 aa; 4p16.3; AR;
         CMS10 — limb-girdle phenotype; salbutamol/albuterol 1ST LINE; AChEI WORSEN — ABSOLUTELY CI;
         Dok-7 activates MuSK; seed SEED_BASE+2) ·
COLQ    (ColQ; 528 aa; 3p24.3; AR;
         CMS5 — endplate AChE deficiency; AChEI ABSOLUTELY CI — worsens dramatically;
         ephrin receptor-anchored AChE; 3,4-DAP + salbutamol; pupillary involvement; seed SEED_BASE+3) ·
CHAT    (ChAT; 748 aa; 10q11.23; AR;
         CMS6 — pre-synaptic; episodic potentially fatal apnea; temperature-sensitive crisis;
         low ACh resynthesis; prophylactic pyridostigmine mandatory; seed SEED_BASE+4) ·
SCN4A   (NaV1.4; 1836 aa; 17q23.3; AD/AR;
         CMS16 / periodic paralysis overlap; sodium channel myasthenia;
         quinidine or acetazolamide; AChEI partial response; seed SEED_BASE+5) ·
AGRN    (Agrin; 2045 aa; 1p36.33; AR;
         CMS8 — LRP4-MuSK-Dok-7 signalling pathway defect; complex limb-girdle + distal weakness;
         salbutamol useful; seed SEED_BASE+6) ·
MUSK    (MuSK; 869 aa; 9q31.3; AR;
         CMS9 — pre-synaptic AChR clustering defect; different from autoimmune MuSK-MG (antibody-driven);
         ephedrine or albuterol preferred; AChEI response variable; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1734–1741)
"""

import random

SEED_BASE = 1734

CMS_GENES = [
    # ── CHRNE — AChR epsilon-subunit / CMS4 endplate AChR deficiency ────────
    {
        "gene": "CHRNE",
        "protein": (
            "CHRNE — 22q11.21 AR — AChR-epsilon-subunit-473aa — "
            "CMS4-EndplateAChRDeficiency — Most-Common-AR-CMS-Europe-MiddleEast — "
            "Pyridostigmine-1stLine — 3,4-DAP-Adjunct"
        ),
        "alias": (
            "CHRNE (cholinergic receptor nicotinic epsilon subunit); OMIM gene 100725; "
            "CMS4 (congenital myasthenic syndrome 4, endplate acetylcholine receptor deficiency) OMIM 616313. "
            "22q11.21; 473 aa; ~53 kDa; autosomal recessive (homozygous or compound heterozygous). "
            "FUNCTION: CHRNE encodes the epsilon (ε) subunit of the adult-type muscle nicotinic acetylcholine "
            "receptor (nAChR), a pentameric ligand-gated ion channel at the neuromuscular junction (NMJ). "
            "Adult AChR composition: (α1)2·β1·δ·ε. The ε-subunit replaced the fetal γ-subunit "
            "in mature NMJ shortly after birth. "
            "LOF mutations in CHRNE → decreased AChR density at the endplate → reduced miniature endplate "
            "potential (MEPP) amplitude → impaired neuromuscular transmission → fatigable weakness. "
            "CLINICAL FEATURES: "
            "Onset: neonatal to early childhood (majority symptomatic before age 2); "
            "ptosis (bilateral, asymmetric) — hallmark presenting feature; "
            "external ophthalmoplegia (EOG) — distinguishes CMS from many adult MG presentations; "
            "fatigable proximal limb weakness; facial weakness (orbicularis oculi, orbicularis oris); "
            "bulbar weakness (feeding difficulties, dysarthria, dysphagia); "
            "respiratory compromise variable (depends on severity); "
            "fluctuating weakness exacerbated by infection, fever, exercise; "
            "cognitive development NORMAL; lifespan potentially normal with treatment. "
            "EPIDEMIOLOGY: most common AR CMS gene in Europe (especially Spanish Romani population — "
            "ε1267delG founder mutation); also common in Middle East. "
            "TREATMENT: Pyridostigmine (anticholinesterase) 1st-line — increases ACh dwell time; "
            "3,4-diaminopyridine (3,4-DAP) — enhances ACh release (pre-synaptic); "
            "combination often more effective than monotherapy. "
            "MONITORING: FVC, bulbar assessment, ptosis, QMGS (quantitative myasthenia gravis score) at each visit."
        ),
        "locus": "22q11.21",
        "aa": 473,
        "kDa": 53,
        "omim_gene": "100725",
        "omim_disease": "616313",
        "inheritance": (
            "AR (autosomal recessive); homozygous or compound heterozygous CHRNE mutations; "
            "25% sibling recurrence risk; parents typically asymptomatic carriers; "
            "de novo mutations uncommon; carrier testing parents mandatory for family planning"
        ),
        "gene_class": (
            "Nicotinic acetylcholine receptor (nAChR) subunit; pentameric ligand-gated ion channel; "
            "adult epsilon subunit (replaces fetal gamma postnatally); "
            "NMJ endplate structure; post-synaptic membrane"
        ),
        "key_alerts": [
            "CHRNE-PYRIDOSTIGMINE-1ST-LINE: Pyridostigmine (Mestinon) is first-line for CHRNE CMS4 — "
            "start at 1 mg/kg/day in 3-4 divided doses, titrate to clinical response; "
            "3,4-DAP (amifampridine) useful adjunct — enhances ACh release; "
            "combination may be more effective than either alone; "
            "NMJ physiological testing (SFEMG, CMAP decrement) guides dosing",
            "CHRNE-PTOSIS-OPHTHALMOPLEGIA-DIAGNOSIS: bilateral ptosis + ophthalmoplegia in a child "
            "without diurnal variation or antibodies should trigger CMS genetic panel; "
            "DO NOT diagnose seronegative MG in a child without CMS exclusion; "
            "CHRNE is the most common AR CMS gene — always include in first-tier panel",
            "CHRNE-FOUNDER-MUTATION: ε1267delG is a founder mutation in Spanish Romani (Gypsy) population; "
            "homozygosity rate high in this ethnic group; "
            "single-gene targeted testing cost-effective in Romani patients before full panel",
            "CHRNE-RESPIRATORY-MONITORING: FVC (sitting + supine) at every visit; "
            "bulbar assessment essential; respiratory failure can occur with fever/infection; "
            "emergency plan for intercurrent illness mandatory; "
            "NIV readiness education for family",
            "CHRNE-NORMAL-COGNITION: cognitive development is completely normal in CMS4; "
            "intellectual disability should trigger search for alternative/additional diagnosis; "
            "mental status is always preserved in uncomplicated CMS",
        ],
        "etiologies": [
            "CMS4 — endplate AChR deficiency (epsilon-subunit null/hypomorphic mutations): "
            "most common AR CMS in European and Middle Eastern populations; "
            "point mutations, small insertions/deletions in CHRNE; "
            "founder mutation ε1267delG in Spanish Romani; "
            "phenotype: ptosis, ophthalmoplegia, fatigable limb and bulbar weakness from neonatal/infant period",
            "Slow-channel syndrome (SCCMS) mimicry — some CHRNE gain-of-function mutations prolong "
            "channel opening → endplate myopathy; rare; phenotype differs from typical LOF CMS4",
            "CHRNE null — complete loss of epsilon subunit; fetal gamma-subunit cannot fully compensate "
            "in adult NMJ → very low AChR density; severe phenotype; ventilator-dependent rare but possible",
            "Compound heterozygous CHRNE — different mutations on each allele; "
            "phenotype depends on residual function of each allele; "
            "variable severity even within same family if different compound mutations",
        ],
        "stats": {
            "ptosis_prevalence": "~95%",
            "ophthalmoplegia_prevalence": "~75%",
            "pyridostigmine_response": "~80%",
            "neonatal_onset_pct": "~65%",
            "respiratory_compromise_pct": "~25%",
        },
    },

    # ── RAPSN — Rapsyn / CMS11 AChR-clustering ──────────────────────────────
    {
        "gene": "RAPSN",
        "protein": (
            "RAPSN — 11p11.2 AR — Rapsyn-412aa — "
            "CMS11-PostSynapticAChRClustering — Neonatal-Onset-Apneic-Crises — "
            "AntenatalFetal-Akinesia-DeformationSequence — Pyridostigmine-3,4-DAP"
        ),
        "alias": (
            "RAPSN (receptor associated protein of the synapse); OMIM gene 601592; "
            "CMS11 (congenital myasthenic syndrome 11) OMIM 616326. "
            "11p11.2; 412 aa; ~43 kDa; autosomal recessive. "
            "FUNCTION: Rapsyn is a post-synaptic scaffolding protein essential for clustering and anchoring "
            "AChR at the NMJ. It forms a stoichiometric 1:1 complex with AChR at the endplate. "
            "Rapsyn links AChR to the subsynaptic cytoskeleton (via β-dystroglycan) and is required "
            "for AChR aggregation in response to Agrin-MuSK signalling. "
            "LOF mutations → failure of AChR clustering → reduced endplate AChR density → "
            "impaired neuromuscular transmission. "
            "CLINICAL FEATURES: "
            "Most common presentation: neonatal hypotonia + apnea; "
            "episodic life-threatening apneic crises (respiratory arrests) triggered by fever/infection; "
            "antenatal form: fetal akinesia deformation sequence (FADS) — reduced fetal movements, "
            "arthrogryposis multiplex congenita, hydrops fetalis (severe); "
            "ptosis and ophthalmoplegia common; "
            "facial weakness, bulbar symptoms; "
            "strength fluctuates — better at rest, worse with exertion or intercurrent illness; "
            "cognitive development NORMAL. "
            "N88K FOUNDER MUTATION: p.Asn88Lys is the commonest RAPSN mutation worldwide, "
            "found in compound heterozygosity with a second mutation in most affected individuals. "
            "TREATMENT: Pyridostigmine + 3,4-DAP combination; "
            "respiratory support (NIV or ventilator) as needed; "
            "emergency protocol for fever/infection essential — anticipate respiratory decompensation."
        ),
        "locus": "11p11.2",
        "aa": 412,
        "kDa": 43,
        "omim_gene": "601592",
        "omim_disease": "616326",
        "inheritance": (
            "AR (autosomal recessive); N88K common mutation — often compound heterozygous with a second allele; "
            "25% sibling recurrence risk; antenatal RAPSN — severe fetal akinesia can occur; "
            "prenatal diagnosis important in previously affected families"
        ),
        "gene_class": (
            "Post-synaptic scaffolding protein; AChR clustering protein; "
            "tetratricopeptide repeat (TPR) domain family; "
            "β-dystroglycan linker; essential for NMJ maturation and AChR density maintenance"
        ),
        "key_alerts": [
            "RAPSN-APNEA-CRISIS-EMERGENCY: RAPSN CMS11 patients face life-threatening episodic apnea, "
            "especially during fever or infection — always have emergency management plan; "
            "parents/carers must know how to manage respiratory arrest; "
            "anticipate respiratory decompensation with ANY intercurrent illness; "
            "ICU admission threshold should be low; increase pyridostigmine dose during illness",
            "RAPSN-ANTENATAL-FADS: severe RAPSN LOF mutations can cause fetal akinesia deformation sequence — "
            "reduced fetal movements, arthrogryposis, talipes, micrognathia, hydrops; "
            "consider RAPSN panel in unexplained arthrogryposis + myasthenic phenotype post-birth; "
            "prenatal diagnosis for recurrence risk counselling",
            "RAPSN-N88K-FOUNDER: p.Asn88Lys (N88K) is the most common RAPSN mutation globally; "
            "usually found compound heterozygous with a second pathogenic allele; "
            "targeted sequencing for N88K cost-effective as first screen before full gene sequencing",
            "RAPSN-FEVER-WORSENING: RAPSN CMS11 worsens markedly with fever (temperature-sensitive NMJ); "
            "any fever should prompt increased monitoring and possible hospitalisation; "
            "all parents must have an emergency plan signed by the neurologist",
            "RAPSN-STRENGTH-RECOVERY: unlike progressive muscular dystrophies, RAPSN CMS11 is NOT progressive "
            "— strength improves with treatment and remains relatively stable; "
            "ambulatory in most cases with adequate pyridostigmine dosing",
        ],
        "etiologies": [
            "CMS11 — rapsyn-deficient endplate: loss-of-function RAPSN mutations → "
            "AChR clustering failure → endplate AChR deficiency; "
            "common mutations: N88K, E150K, various frameshift/splice; "
            "phenotype: neonatal onset, apneic crises, arthrogryposis in severe cases",
            "Antenatal FADS variant — severe LOF RAPSN; fetal akinesia, arthrogryposis multiplex congenita, "
            "pterygia, polyhydramnios, reduced fetal movement; stillbirth or severe neonatal compromise",
            "Late-onset RAPSN — rare milder mutations presenting in childhood or adolescence "
            "without neonatal crisis; heterogeneous severity depending on residual rapsyn function",
            "Compound heterozygous RAPSN — N88K + second null allele most common genotype worldwide; "
            "genotype-phenotype correlation: two severe alleles → earlier/more severe phenotype",
        ],
        "stats": {
            "n88k_prevalence": "~60% of RAPSN alleles worldwide",
            "apneic_crisis_rate": "~70% of RAPSN patients",
            "arthrogryposis_pct": "~25%",
            "ambulatory_pct": "~80% (treated)",
            "neonatal_onset_pct": "~75%",
        },
    },

    # ── DOK7 — Dok-7 / CMS10 limb-girdle / salbutamol ──────────────────────
    {
        "gene": "DOK7",
        "protein": (
            "DOK7 — 4p16.3 AR — Dok7-504aa — "
            "CMS10-LimbGirdle-PreSynapticAChRClustering — "
            "SALBUTAMOL-ALBUTEROL-1ST-LINE — AChEI-WORSEN-ABSOLUTELY-CI"
        ),
        "alias": (
            "DOK7 (downstream of tyrosine kinase 7); OMIM gene 610285; "
            "CMS10 (congenital myasthenic syndrome 10) OMIM 254300. "
            "4p16.3; 504 aa; ~56 kDa; autosomal recessive. "
            "FUNCTION: Dok-7 is an adaptor protein that binds and activates MuSK (muscle-specific kinase) "
            "at the NMJ. It bridges MuSK auto-phosphorylation with downstream signalling required for "
            "AChR clustering and NMJ maturation. Dok-7 contains a PH domain and a PTB domain. "
            "LOF mutations → reduced MuSK activation → small poorly-organised NMJ endplates → "
            "reduced AChR density → neuromuscular transmission failure. "
            "DISTINCTIVE CLINICAL PHENOTYPE — LIMB-GIRDLE CMS: "
            "Proximal > distal limb-girdle weakness (hips > shoulders); "
            "ptosis MILD or absent (distinguishes from CHRNE and RAPSN); "
            "ophthalmoplegia ABSENT or minimal (KEY DDx from CHRNE/RAPSN); "
            "fatigable limb weakness; walking difficulties; waddling gait; "
            "Trendelenburg sign; difficulty climbing stairs/rising from floor; "
            "stridor and respiratory weakness common (laryngeal involvement); "
            "bulbar weakness variable; cognitive development NORMAL. "
            "TREATMENT — CRITICAL PHARMACOLOGY: "
            "Salbutamol (albuterol) β2-agonist: FIRST-LINE for DOK7 CMS10; "
            "mechanism: promotes NMJ maturation and AChR clustering via upregulation of downstream pathways; "
            "dose: 2 mg TID (adults); paediatric dose adjusted; response typically within weeks to months; "
            "AChEI (pyridostigmine) — ABSOLUTELY CONTRAINDICATED in DOK7 CMS: causes clinical WORSENING; "
            "mechanism unclear but consistently worsens NMJ function in DOK7-CMS; "
            "THIS IS A LIFE-THREATENING PRESCRIBING ERROR — verify gene before prescribing AChEI."
        ),
        "locus": "4p16.3",
        "aa": 504,
        "kDa": 56,
        "omim_gene": "610285",
        "omim_disease": "254300",
        "inheritance": (
            "AR (autosomal recessive); c.1124_1127dupTGCC (p.Ala378Leufs*30) founder duplication — "
            "most common DOK7 mutation worldwide; 25% sibling recurrence risk; "
            "parents asymptomatic carriers"
        ),
        "gene_class": (
            "MuSK adaptor protein; PH domain + PTB domain; "
            "downstream of tyrosine kinase (DOK family); "
            "MuSK kinase activator; NMJ formation and maintenance; "
            "Agrin-LRP4-MuSK-Dok-7 signalling cascade"
        ),
        "key_alerts": [
            "DOK7-AChEI-ABSOLUTELY-CI-LIFE-THREATENING: Pyridostigmine and all anticholinesterase drugs "
            "are ABSOLUTELY CONTRAINDICATED in DOK7-CMS10 — they cause rapid clinical deterioration; "
            "if pyridostigmine has been prescribed by mistake, STOP IMMEDIATELY and monitor closely; "
            "ALWAYS confirm gene diagnosis before prescribing AChEI in any CMS patient; "
            "this is one of the most important prescribing errors in neuromuscular disease",
            "DOK7-SALBUTAMOL-1ST-LINE: salbutamol (albuterol) β2-agonist is first-line treatment; "
            "start at low dose (e.g. 2 mg OD, titrate to 2 mg TID in adults); "
            "response may take weeks to months — monitor FVC, walking ability, ptosis, QMGS; "
            "ephedrine is an alternative if salbutamol not tolerated; "
            "DO NOT stop without specialist review",
            "DOK7-LIMB-GIRDLE-NO-EOM: DOK7-CMS10 characteristically spares extraocular muscles — "
            "limb-girdle weakness without significant ptosis/ophthalmoplegia is the hallmark; "
            "this distinguishes DOK7 from most other CMS subtypes; "
            "facial weakness variable; stridor/respiratory important to assess",
            "DOK7-RESPIRATORY-STRIDOR: laryngeal/respiratory muscle involvement is common in DOK7; "
            "stridor can be the presenting symptom; "
            "FVC monitoring + respiratory assessment mandatory; "
            "NIV may be needed before limb weakness becomes severe",
            "DOK7-FOUNDER-MUTATION: c.1124_1127dupTGCC (4-bp duplication in exon 7) is the most common "
            "DOK7 mutation worldwide; targeted testing for this mutation first is cost-effective",
        ],
        "etiologies": [
            "CMS10 — Dok-7 deficiency: loss-of-function DOK7 mutations → "
            "impaired MuSK activation → small endplates → reduced AChR density; "
            "limb-girdle phenotype, stridor, respiratory involvement; "
            "c.1124_1127dupTGCC most common mutation; typically childhood-onset",
            "Adult-onset DOK7-CMS — rare; some mutations present in 2nd-3rd decade; "
            "initial misdiagnosis as seronegative MG common; "
            "AChEI trials by mistake → worsening triggers DOK7 diagnosis",
            "DOK7 + MuSK activation complex — mutations disrupting Dok-7 PH or PTB domain; "
            "severity correlates with degree of MuSK activation impairment; "
            "biallelic null mutations → most severe phenotype with congenital onset and respiratory failure",
        ],
        "stats": {
            "founder_dup_prevalence": "~50% of DOK7 alleles globally",
            "salbutamol_response_rate": "~75%",
            "ophthalmoplegia_pct": "<10% (KEY distinguisher)",
            "respiratory_niv_pct": "~30%",
            "stridor_pct": "~45%",
        },
    },

    # ── COLQ — ColQ / CMS5 endplate AChE deficiency ─────────────────────────
    {
        "gene": "COLQ",
        "protein": (
            "COLQ — 3p24.3 AR — ColQ-528aa — "
            "CMS5-EndplateAChEDeficiency — AChEI-ABSOLUTELY-CI-Worsens-Dramatically — "
            "PupillaryInvolvement-Hallmark — 3,4-DAP-Salbutamol"
        ),
        "alias": (
            "COLQ (collagen-like tail subunit of asymmetric acetylcholinesterase); OMIM gene 603033; "
            "CMS5 (congenital myasthenic syndrome 5, endplate acetylcholinesterase deficiency) OMIM 603034. "
            "3p24.3; 528 aa; ~57 kDa; autosomal recessive. "
            "FUNCTION: ColQ is the collagen-like scaffolding tail protein of asymmetric AChE at the NMJ. "
            "It anchors the AChE tetramers to the basal lamina via perlecan. "
            "LOF COLQ mutations → AChE absent or markedly reduced at the endplate → "
            "ACh accumulates in the synaptic cleft → prolonged, repeated, desensitising stimulation of AChR → "
            "progressive AChR inactivation (desensitisation block) → endplate myopathy → weakness. "
            "PARADOX — AChEI MAKES COLQ WORSE: "
            "Adding pyridostigmine (AChEI) when AChE is already absent → even MORE ACh accumulation → "
            "more desensitisation block → rapid clinical deterioration — sometimes catastrophic. "
            "AChEI is ABSOLUTELY CONTRAINDICATED in COLQ-CMS5. "
            "DISTINCTIVE FEATURES: "
            "Pupillary abnormalities (slow light reflexes, miosis) — HALLMARK; "
            "endplate myopathy on biopsy (degeneration of junctional folds); "
            "SFEMG shows decremental response with repetitive nerve stimulation; "
            "weakness: proximal limb + axial + bulbar; "
            "ptosis and ophthalmoplegia variable; "
            "respiratory compromise common. "
            "TREATMENT: 3,4-DAP (amifampridine) — pre-synaptic ACh release enhancement; "
            "salbutamol (albuterol) — NMJ maturation; "
            "combination 3,4-DAP + salbutamol often most effective. "
            "Ephedrine alternative."
        ),
        "locus": "3p24.3",
        "aa": 528,
        "kDa": 57,
        "omim_gene": "603033",
        "omim_disease": "603034",
        "inheritance": (
            "AR (autosomal recessive); homozygous or compound heterozygous COLQ mutations; "
            "25% sibling recurrence risk; parents asymptomatic carriers"
        ),
        "gene_class": (
            "Collagen-like AChE tail protein; asymmetric AChE (A12 form); "
            "NMJ basal lamina anchorage (perlecan-COLQ-AChE complex); "
            "endplate AChE is abolished or severely reduced"
        ),
        "key_alerts": [
            "COLQ-AChEI-ABSOLUTELY-CI-CATASTROPHIC: Pyridostigmine, neostigmine, and all anticholinesterases "
            "are ABSOLUTELY CONTRAINDICATED in COLQ-CMS5 — they cause dramatic worsening and can be fatal; "
            "AChE is already absent in COLQ-CMS5; adding AChEI causes ACh overload → desensitisation block; "
            "THIS IS ONE OF THE MOST DANGEROUS PRESCRIBING ERRORS IN CMS — always confirm gene before AChEI",
            "COLQ-PUPILLARY-HALLMARK: pupillary abnormalities (slow light reflex, anisocoria, miosis) "
            "are characteristic of COLQ-CMS5 and distinguish it from most other CMS subtypes; "
            "always examine pupils carefully in any child with fatigable weakness; "
            "pupil involvement in CMS = COLQ until proven otherwise",
            "COLQ-3,4-DAP-SALBUTAMOL-TREATMENT: 3,4-DAP (amifampridine) + salbutamol combination "
            "is the treatment of choice for COLQ-CMS5; "
            "3,4-DAP enhances ACh release pre-synaptically without worsening endplate overload; "
            "salbutamol promotes NMJ maturation; "
            "monitor ECG with 3,4-DAP (QTc prolongation risk — low but monitor)",
            "COLQ-ENDPLATE-MYOPATHY-BIOPSY: muscle biopsy shows endplate myopathy — "
            "degeneration of junctional folds with vacuolar changes and nuclear proliferation; "
            "AChE staining absent at endplate (Karnovsky-Roots stain) — PATHOGNOMONIC; "
            "SFEMG shows increased jitter + decrement on RNS",
            "COLQ-RESPIRATORY-MONITORING: respiratory compromise is common; "
            "FVC sitting + supine at every visit; "
            "intercurrent infection is a major risk — have emergency plan; "
            "NIV readiness education mandatory",
        ],
        "etiologies": [
            "CMS5 — ColQ null / severe LOF: absent endplate AChE; severe desensitisation block; "
            "early-onset, severe weakness; pupillary abnormalities; endplate myopathy on biopsy; "
            "AChEI dramatically worsens",
            "COLQ partial LOF — reduced (not absent) AChE; milder phenotype; "
            "late-onset possible; pupillary involvement may be subtle; "
            "still CONTRAINDICATED for AChEI even with partial LOF",
            "COLQ + proline-rich attachment domain mutations — disrupts COLQ anchoring to basal lamina; "
            "some genotype-phenotype correlation with residual anchoring function",
        ],
        "stats": {
            "pupillary_involvement_pct": "~80%",
            "ache_absent_biopsy_pct": "~95%",
            "rnsdecrement_pct": "~90%",
            "respiratory_niv_pct": "~40%",
            "dai_salbutamol_response": "~70%",
        },
    },

    # ── CHAT — ChAT / CMS6 pre-synaptic / episodic apnea ───────────────────
    {
        "gene": "CHAT",
        "protein": (
            "CHAT — 10q11.23 AR — ChAT-748aa — "
            "CMS6-PreSynapticACh-Resynthesis — EPISODIC-FATAL-APNEA-Hallmark — "
            "Temperature-Sensitive-Crisis — Prophylactic-Pyridostigmine-Mandatory"
        ),
        "alias": (
            "CHAT (choline acetyltransferase); OMIM gene 118490; "
            "CMS6 (congenital myasthenic syndrome 6, presynaptic) OMIM 254210. "
            "10q11.23; 748 aa; ~82 kDa; autosomal recessive. "
            "FUNCTION: ChAT catalyses the synthesis of acetylcholine (ACh) from choline and acetyl-CoA "
            "in the cytoplasm of motor nerve terminals. LOF mutations → impaired ACh resynthesis → "
            "quantal ACh store becomes depleted during sustained activity → neuromuscular block. "
            "The defect is PRE-SYNAPTIC — the problem is in the motor nerve terminal, not the endplate. "
            "DISTINCTIVE CLINICAL FEATURE — EPISODIC APNEA: "
            "Sudden, potentially fatal respiratory arrest is the defining clinical hallmark of CHAT-CMS6. "
            "These apneic episodes occur spontaneously, in sleep, or triggered by fever/infection/exertion; "
            "between episodes, patients may appear relatively normal or have mild weakness; "
            "episodes can be the FIRST presentation — sudden death in previously well-looking child; "
            "temperature sensitivity — high fever dramatically worsens NMJ transmission; "
            "TREATMENT: Prophylactic pyridostigmine (AChEI) is mandatory — "
            "maintains sufficient ACh levels to prevent crisis; "
            "do NOT wait for symptoms before treating; treat prophylactically even in asymptomatic periods; "
            "3,4-DAP adjunct useful to enhance pre-synaptic ACh release; "
            "home monitoring (apnea alarm, SpO2 monitoring) mandatory for all CHAT-CMS6 patients."
        ),
        "locus": "10q11.23",
        "aa": 748,
        "kDa": 82,
        "omim_gene": "118490",
        "omim_disease": "254210",
        "inheritance": (
            "AR (autosomal recessive); homozygous or compound heterozygous CHAT mutations; "
            "25% sibling recurrence risk; carrier parents: no symptoms; "
            "family history of sudden infant death or unexplained apnea — consider CHAT"
        ),
        "gene_class": (
            "Choline acetyltransferase; ACh biosynthesis enzyme; "
            "motor nerve terminal cytoplasm; vesicular ACh synthesis; "
            "pre-synaptic CMS (rare category — most CMS are post-synaptic)"
        ),
        "key_alerts": [
            "CHAT-EPISODIC-FATAL-APNEA: CHAT-CMS6 can cause sudden potentially fatal apnea — "
            "this is the most dangerous CMS for unexpected death; "
            "episodes occur in sleep or at rest without warning; "
            "home apnea monitoring (pulse oximeter alarm + apnea monitor) MANDATORY for all CHAT patients; "
            "parents must be trained in BLS/CPR; "
            "any infant with unexplained apnea must have CHAT on the differential",
            "CHAT-PROPHYLACTIC-PYRIDOSTIGMINE-MANDATORY: Pyridostigmine must be given prophylactically "
            "even when the patient appears well — do NOT wait for weakness or apnea before treating; "
            "ACh stores can deplete rapidly; continuous AChEI maintains quantal ACh; "
            "NEVER stop pyridostigmine without specialist guidance",
            "CHAT-TEMPERATURE-CRISIS: fever dramatically impairs CHAT function and worsens ACh synthesis; "
            "ANY febrile illness requires emergency plan activation — increase pyridostigmine dose, "
            "hospitalise if temperature >38.5°C, monitor SpO2 continuously; "
            "aggressive antipyretic management is part of CMS6 care",
            "CHAT-DIAGNOSIS-PRE-SYNAPTIC: CHAT is pre-synaptic CMS (rare); "
            "decrement on low-frequency RNS may be absent between episodes; "
            "single-fibre EMG (SFEMG) shows increased jitter; "
            "diagnosis often triggered by episodic apnea + family history + CMS panel; "
            "SFEMG during or after exercise stress most sensitive",
            "CHAT-SUDDEN-INFANT-DEATH-LINK: unexplained SIDS in a previous sibling + surviving infant with "
            "episodes = CMS6 until proven otherwise; "
            "genetic testing of surviving sibling URGENT; "
            "retrospective CHAT testing in SIDS families should be considered",
        ],
        "etiologies": [
            "CMS6 — CHAT null/severe LOF: severely reduced ACh synthesis; "
            "episodic fatal apnea; temperature-sensitive crisis; "
            "between episodes patients may seem near-normal (deceptive); "
            "emergency presentation common as first diagnosis",
            "CHAT hypomorphic mutations — partial reduction in ChAT activity; "
            "milder phenotype with episodic exacerbations rather than constant baseline weakness; "
            "apnea risk still present — still requires prophylactic pyridostigmine",
            "CHAT with prominent bulbar — some CHAT mutations present with "
            "severe feeding difficulties, choking, swallowing crisis; "
            "apnea + bulbar = high-risk combination for aspiration pneumonia",
        ],
        "stats": {
            "apneic_crisis_pct": "~90%",
            "temperature_sensitivity_pct": "~85%",
            "sids_family_history_pct": "~20%",
            "pyridostigmine_response_pct": "~85%",
            "home_monitoring_required_pct": "100%",
        },
    },

    # ── SCN4A — NaV1.4 / CMS16 / periodic paralysis ─────────────────────────
    {
        "gene": "SCN4A",
        "protein": (
            "SCN4A — 17q23.3 AD/AR — NaV1.4-1836aa — "
            "CMS16-SodiumChannel-Myasthenia — PeriodicParalysis-Overlap — "
            "Quinidine-Acetazolamide — AChEI-PartialResponse-Only"
        ),
        "alias": (
            "SCN4A (sodium voltage-gated channel alpha subunit 4); OMIM gene 603967; "
            "CMS16 (congenital myasthenic syndrome 16) OMIM 614198; "
            "also causes Hyperkalemic Periodic Paralysis (HypPP, OMIM 170500), "
            "Paramyotonia Congenita (PC, OMIM 168300), Hypokalemic Periodic Paralysis type 2 (HypoPP2). "
            "17q23.3; 1836 aa; ~208 kDa; autosomal dominant (GOF) or autosomal recessive (LOF for CMS). "
            "FUNCTION: SCN4A encodes NaV1.4, the principal voltage-gated sodium channel in skeletal muscle. "
            "NaV1.4 is responsible for the action potential upstroke in muscle fibres. "
            "Mutations cause two distinct disease categories: "
            "GOF mutations (dominant, AD): channel remains open too long → membrane hyperexcitability → "
            "myotonia, paramyotonia, periodic paralysis (HypPP2), exercise intolerance; "
            "LOF mutations (recessive, AR): channel cannot generate sufficient action potentials → "
            "muscle inexcitability → congenital myasthenic phenotype (CMS16) — "
            "patients appear clinically similar to CMS but EMG pattern differs. "
            "CMS16 PHENOTYPE: "
            "Neonatal/infantile weakness and hypotonia; "
            "ptosis and ophthalmoplegia variable; "
            "bulbar involvement; "
            "RNS may show decrement; "
            "SFEMG increased jitter; "
            "importantly: overlapping with ion channel periodic paralysis family — "
            "episodic weakness attacks can occur on top of baseline weakness. "
            "TREATMENT: quinidine (membrane stabiliser — modulates NaV1.4); "
            "acetazolamide (used in periodic paralysis); "
            "pyridostigmine partial response in some CMS16 patients; "
            "treatment must be guided by specialist and electrophysiology."
        ),
        "locus": "17q23.3",
        "aa": 1836,
        "kDa": 208,
        "omim_gene": "603967",
        "omim_disease": "614198",
        "inheritance": (
            "AR for CMS16 (biallelic LOF mutations — channel cannot generate action potentials); "
            "AD for periodic paralysis/myotonia/paramyotonia (GOF — channel stays open/fast-reactivation); "
            "25% sibling risk (AR CMS16); 50% offspring risk (AD channelopathies)"
        ),
        "gene_class": (
            "Voltage-gated sodium channel alpha subunit; skeletal muscle NaV1.4; "
            "fast-inactivating sodium channel; "
            "action potential upstroke in muscle membrane; "
            "ion channel (channelopathy family)"
        ),
        "key_alerts": [
            "SCN4A-CMS16-VS-PERIODIC-PARALYSIS-SAME-GENE: LOF SCN4A mutations → CMS16 (AR, congenital); "
            "GOF SCN4A mutations → periodic paralysis/myotonia/paramyotonia (AD); "
            "careful genotype classification needed to guide treatment — mechanism-based therapy differs; "
            "a child with episodic weakness + baseline weakness may have CMS16 or periodic paralysis overlap",
            "SCN4A-QUINIDINE-MEMBRANE-STABILISER: quinidine (sodium channel blocker) is useful in GOF "
            "SCN4A channelopathies; also has evidence in some CMS16 patients; "
            "QTc prolongation risk with quinidine — ECG monitoring mandatory; "
            "avoid other QTc-prolonging drugs concurrently",
            "SCN4A-AChEI-PARTIAL-RESPONSE-ONLY: pyridostigmine may provide partial symptomatic benefit "
            "in CMS16 but is NOT the primary treatment; "
            "the pathology is in the channel (muscle inexcitability) not in ACh deficiency; "
            "do not expect full response to AChEI as in post-synaptic CMS subtypes",
            "SCN4A-ELECTROPHYSIOLOGY-ESSENTIAL: EMG shows specific patterns in SCN4A channelopathies "
            "(myotonic discharges in GOF, muscle inexcitability pattern in LOF); "
            "nerve conduction studies normal; "
            "EMG is critical to confirm diagnosis and guide genetic testing focus",
            "SCN4A-COLD-EXACERBATION: paramyotonia congenita (GOF SCN4A) worsens in cold; "
            "patients should avoid cold water, cold environments; "
            "warm-up improves strength in paramyotonia; "
            "CMS16 does NOT show prominent cold sensitivity — use this to distinguish clinically",
        ],
        "etiologies": [
            "CMS16 — SCN4A biallelic LOF: muscle inexcitability due to non-functional NaV1.4; "
            "neonatal/congenital onset; overlapping myasthenic phenotype; "
            "RNS decrement variable; quinidine or acetazolamide may help",
            "Hyperkalemic Periodic Paralysis type 2 (HypPP2) — GOF SCN4A AD: "
            "episodic weakness with high serum K+; triggered by rest-after-exercise, K+-rich foods; "
            "acetazolamide or dichlorphenamide; avoid propranolol in acute attack",
            "Paramyotonia Congenita (PC) — GOF SCN4A AD: "
            "myotonia worsened by cold; paradoxical myotonia (worsens with repeated contractions); "
            "mexiletine or tocainide for myotonia; warm environment protective",
            "Myotonia Congenita SCN4A-related — rare; primarily myotonia without episodic paralysis; "
            "mexiletine first-line; phenotype distinguishable from CLCN1 myotonia congenita by EMG pattern",
        ],
        "stats": {
            "episodic_weakness_pct": "~60%",
            "myotonia_pct": "~40%",
            "cold_sensitivity_pct": "~35%",
            "quinidine_response_pct": "~55%",
            "neonatal_onset_pct": "~50%",
        },
    },

    # ── AGRN — Agrin / CMS8 / LRP4-MuSK pathway ────────────────────────────
    {
        "gene": "AGRN",
        "protein": (
            "AGRN — 1p36.33 AR — Agrin-2045aa — "
            "CMS8-AgrinLRP4MuSKPathway — LimbGirdle-DistalWeakness — "
            "Salbutamol-3,4-DAP-Useful — AChEI-PartialResponse"
        ),
        "alias": (
            "AGRN (agrin); OMIM gene 103320; "
            "CMS8 (congenital myasthenic syndrome 8, agrin deficiency) OMIM 615120. "
            "1p36.33; 2045 aa; ~215 kDa; autosomal recessive. "
            "FUNCTION: Agrin is a heparan sulfate proteoglycan secreted by motor neurons at the NMJ. "
            "Neural agrin (z+agrin) is the critical isoform that activates MuSK via LRP4. "
            "The Agrin → LRP4 → MuSK → Dok-7 signalling cascade is essential for AChR clustering "
            "and NMJ formation. LOF mutations in AGRN → impaired LRP4-MuSK activation → "
            "reduced AChR clustering → NMJ formation defect → fatigable weakness. "
            "CLINICAL FEATURES: "
            "Limb-girdle + distal weakness (proximal and distal combined — 'complex' phenotype); "
            "variable ptosis; ophthalmoplegia variable; "
            "bulbar involvement (dysphagia, dysarthria); "
            "respiratory compromise may occur; "
            "cognitive development NORMAL; "
            "phenotype overlaps with Dok-7 CMS but may have more distal involvement. "
            "TREATMENT: Salbutamol (β2-agonist) helpful for promoting NMJ maturation; "
            "3,4-DAP (amifampridine) may improve ACh release; "
            "pyridostigmine partial response (mechanism different from pure post-synaptic deficiency); "
            "combination therapy often required. "
            "GENETIC NOTE: AGRN is a large gene (2045 aa) — comprehensive sequencing needed; "
            "some AGRN variants are incidental (agrin polymorphisms common in general population); "
            "functional validation or strong segregation needed for variant interpretation."
        ),
        "locus": "1p36.33",
        "aa": 2045,
        "kDa": 215,
        "omim_gene": "103320",
        "omim_disease": "615120",
        "inheritance": (
            "AR (autosomal recessive); homozygous or compound heterozygous AGRN mutations; "
            "25% sibling recurrence risk; large gene — full sequencing required; "
            "variant interpretation requires functional evidence given high population variant frequency"
        ),
        "gene_class": (
            "Heparan sulfate proteoglycan; secreted by motor neurons; "
            "NMJ basal lamina component; "
            "Agrin-LRP4-MuSK signalling cascade activator; "
            "AChR clustering initiator"
        ),
        "key_alerts": [
            "AGRN-LRP4-MUSK-PATHWAY: AGRN, LRP4, MUSK, and DOK7 form a linked signalling pathway; "
            "mutations in any of these four genes disrupt AChR clustering; "
            "overlapping phenotypes — limb-girdle weakness, minimal EOM involvement; "
            "panel sequencing all four simultaneously is cost-effective in limb-girdle CMS",
            "AGRN-LARGE-GENE-VARIANT-INTERPRETATION: AGRN is a large gene and AGRN variants are "
            "common in the general population; many AGRN variants are benign; "
            "be cautious about variant of uncertain significance (VUS) interpretation; "
            "functional studies, segregation analysis, and ClinVar data essential before diagnosing CMS8",
            "AGRN-SALBUTAMOL-HELPFUL: salbutamol (albuterol) β2-agonist is useful for AGRN-CMS8 "
            "by promoting NMJ maturation downstream of the AGRN-MuSK defect; "
            "do NOT avoid AChEI as strongly as in DOK7 or COLQ — pyridostigmine may help partially; "
            "combination salbutamol + pyridostigmine often tried",
            "AGRN-RESPIRATORY-BULBAR: respiratory compromise and dysphagia occur in AGRN-CMS8; "
            "FVC monitoring and bulbar assessment at every visit; "
            "speech + language therapy for dysphagia; "
            "NIV if FVC declining",
            "AGRN-AUTOIMMUNE-CONFUSION: neural agrin is different from muscle agrin used in autoimmune MG; "
            "anti-LRP4 autoimmune MG is a distinct autoimmune condition — not to be confused with CMS8; "
            "CMS8 is GENETIC (AR), not autoimmune; no role for immunotherapy in CMS8",
        ],
        "etiologies": [
            "CMS8 — agrin deficiency: biallelic AGRN LOF mutations → "
            "impaired LRP4-MuSK signalling → reduced AChR clustering → NMJ deficiency; "
            "complex limb-girdle + distal weakness; variable EOM involvement",
            "AGRN splice mutations — common pathogenic mechanism; "
            "z-exon splice variants critical for neural agrin function; "
            "mutations affecting z-exon most severely affect NMJ clustering",
            "AGRN with developmental joint deformities — severe neonatal AGRN LOF; "
            "fetal akinesia deformation sequence overlap (severe cases); "
            "milder genotypes present in childhood with limb-girdle weakness",
        ],
        "stats": {
            "distal_weakness_pct": "~60%",
            "ptosis_pct": "~55%",
            "bulbar_pct": "~50%",
            "respiratory_compromise_pct": "~35%",
            "salbutamol_response_pct": "~60%",
        },
    },

    # ── MUSK — MuSK / CMS9 / pre-synaptic AChR clustering ───────────────────
    {
        "gene": "MUSK",
        "protein": (
            "MUSK — 9q31.3 AR — MuSK-869aa — "
            "CMS9-PreSynapticAChR-Clustering-Kinase — "
            "DifferentFrom-AutoimmuneMuSK-MG — "
            "Ephedrine-Albuterol-Preferred-AChEI-VariableResponse"
        ),
        "alias": (
            "MUSK (muscle specific kinase); OMIM gene 601296; "
            "CMS9 (congenital myasthenic syndrome 9, MuSK deficiency) OMIM 616325. "
            "9q31.3; 869 aa; ~97 kDa; autosomal recessive (LOF for CMS9). "
            "FUNCTION: MuSK is a receptor tyrosine kinase expressed exclusively at the NMJ endplate. "
            "It is the central transducer of Agrin-LRP4 signalling. "
            "Activated MuSK recruits Dok-7, phosphorylates downstream targets (including Rapsyn/Tid1), "
            "and drives AChR clustering and NMJ maturation. "
            "LOF MUSK mutations (CMS9) → kinase-dead or loss of AChR clustering → CMS phenotype. "
            "CRITICAL DISTINCTION — CMS9 vs AUTOIMMUNE MuSK-MG: "
            "CMS9 (MUSK-CMS): GENETIC — biallelic AR LOF MUSK mutations; "
            "congenital/childhood onset; no MuSK antibodies; no role for immunotherapy. "
            "Autoimmune MuSK-MG: AUTOIMMUNE — anti-MuSK IgG4 antibodies; "
            "adult onset (predominantly young women); immunotherapy (rituximab) 1st-line; "
            "AChEI often poorly tolerated in autoimmune MuSK-MG; "
            "These two conditions MUST NOT be confused — opposite treatment approaches. "
            "CMS9 TREATMENT: "
            "Ephedrine (sympathomimetic) or albuterol (salbutamol) — "
            "promote NMJ maturation via adenylyl cyclase pathways; "
            "pyridostigmine variable response (depends on residual MuSK activity); "
            "3,4-DAP may help; "
            "immunotherapy NOT indicated — this is not autoimmune."
        ),
        "locus": "9q31.3",
        "aa": 869,
        "kDa": 97,
        "omim_gene": "601296",
        "omim_disease": "616325",
        "inheritance": (
            "AR (autosomal recessive) for CMS9 — biallelic LOF MUSK mutations; "
            "AD GOF MUSK variants are not established as disease-causing; "
            "autoimmune MuSK-MG is NOT genetic — different condition entirely; "
            "25% sibling recurrence risk for CMS9"
        ),
        "gene_class": (
            "Receptor tyrosine kinase; NMJ-specific kinase; "
            "Agrin-LRP4-MuSK signalling cascade central transducer; "
            "Dok-7 binding partner; AChR clustering orchestrator; "
            "cysteine-rich domain + kringle domain + kinase domain"
        ),
        "key_alerts": [
            "MUSK-CMS9-VS-AUTOIMMUNE-MuSK-MG-CRITICAL-DISTINCTION: "
            "MUSK-CMS9 (genetic, AR LOF) and autoimmune MuSK-MG (anti-MuSK antibody) are DIFFERENT diseases; "
            "CMS9 = AR genetic disorder of infancy/childhood, no antibodies, treat with ephedrine/salbutamol; "
            "autoimmune MuSK-MG = adult-onset, anti-MuSK IgG4+, treat with rituximab/immunotherapy; "
            "test anti-MuSK antibodies to exclude autoimmune form; "
            "NEVER give immunosuppression for CMS9 — ineffective and harmful",
            "MUSK-EPHEDRINE-ALBUTEROL-PREFERRED: ephedrine or albuterol (salbutamol) are preferred "
            "treatments for MUSK-CMS9; promote NMJ maturation via cAMP pathways; "
            "pyridostigmine response variable (may help partially but not reliable); "
            "avoid high-dose AChEI if clinical response poor — reassess diagnosis",
            "MUSK-AChEI-AUTOIMMUNE-WARNING: in autoimmune MuSK-MG (NOT CMS9), "
            "pyridostigmine is often poorly tolerated and may worsen muscarinic symptoms; "
            "this is the OPPOSITE of post-synaptic CHRNE-CMS where AChEI helps; "
            "precise genetic/antibody diagnosis essential before committing to AChEI therapy",
            "MUSK-ANTIBODY-TESTING-MANDATORY: anti-AChR and anti-MuSK antibodies must be tested "
            "in ALL patients with suspected CMS before genetic panel — "
            "to exclude seronegative MG and autoimmune MuSK-MG; "
            "CMS9 is seronegative (no antibodies)",
            "MUSK-AGRIN-LRP4-DOK7-PATHWAY: MUSK, AGRN, LRP4, and DOK7 are four components of "
            "the same NMJ signalling cascade; mutations in any one cause similar limb-girdle CMS; "
            "panel test all four simultaneously in limb-girdle CMS without EOM involvement",
        ],
        "etiologies": [
            "CMS9 — MuSK kinase-dead LOF: biallelic MUSK mutations abolish or severely reduce kinase activity; "
            "AChR clustering fails; NMJ formation defective; "
            "congenital/early childhood onset; limb-girdle + bulbar pattern; "
            "no anti-MuSK antibodies — genetic, not autoimmune",
            "MUSK partial LOF — residual kinase activity; milder phenotype; "
            "childhood-onset with fatigable proximal weakness; "
            "ephedrine/albuterol responsive; better prognosis than null",
            "MUSK with prominent bulbar — some MUSK LOF mutations give severe dysphagia/dysarthria; "
            "may be misdiagnosed as seronegative MG; "
            "genetic testing essential in any childhood-onset seronegative MG phenotype",
        ],
        "stats": {
            "antibody_seronegative_pct": "100%",
            "limb_girdle_pct": "~75%",
            "bulbar_pct": "~55%",
            "ephedrine_response_pct": "~65%",
            "ophthalmoplegia_pct": "~40%",
        },
    },
]


def _make_cohort(gene_dict, seed):
    rng = random.Random(seed)
    gene = gene_dict["gene"]
    patients = []
    for pid in range(40):
        if gene == "CHRNE":
            onset_age = rng.gauss(0.5, 1.2)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(24, 12)
            dx_delay = max(3, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "ptosis": rng.random() < 0.95,
                "ophthalmoplegia": rng.random() < 0.75,
                "pyridostigmine_response": rng.random() < 0.80,
                "respiratory_niv": rng.random() < 0.25,
                "cognition_normal": True,
            })
        elif gene == "RAPSN":
            onset_age = rng.gauss(0.1, 0.3)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(18, 10)
            dx_delay = max(1, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "apneic_crisis": rng.random() < 0.70,
                "arthrogryposis": rng.random() < 0.25,
                "n88k_allele": rng.random() < 0.60,
                "ambulatory": rng.random() < 0.80,
                "fever_crisis_history": rng.random() < 0.65,
            })
        elif gene == "DOK7":
            onset_age = rng.gauss(5.0, 4.5)
            onset_age = max(0.3, onset_age)
            dx_delay = rng.gauss(60, 24)
            dx_delay = max(6, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "limb_girdle_weakness": True,
                "ophthalmoplegia": rng.random() < 0.10,
                "salbutamol_response": rng.random() < 0.75,
                "achei_worsened": rng.random() < 0.80,
                "stridor": rng.random() < 0.45,
            })
        elif gene == "COLQ":
            onset_age = rng.gauss(1.0, 2.0)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(36, 16)
            dx_delay = max(4, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "pupillary_involvement": rng.random() < 0.80,
                "ache_absent_biopsy": rng.random() < 0.95,
                "dai_response": rng.random() < 0.70,
                "respiratory_niv": rng.random() < 0.40,
                "achei_contraindicated": True,
            })
        elif gene == "CHAT":
            onset_age = rng.gauss(0.2, 0.5)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(14, 8)
            dx_delay = max(1, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "apneic_crisis": rng.random() < 0.90,
                "temperature_sensitive": rng.random() < 0.85,
                "home_monitoring": True,
                "pyridostigmine_prophylactic": True,
                "sids_family_history": rng.random() < 0.20,
            })
        elif gene == "SCN4A":
            onset_age = rng.gauss(4.0, 6.0)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(48, 24)
            dx_delay = max(4, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "episodic_weakness": rng.random() < 0.60,
                "myotonia": rng.random() < 0.40,
                "cold_sensitivity": rng.random() < 0.35,
                "quinidine_response": rng.random() < 0.55,
                "periodic_paralysis_overlap": rng.random() < 0.45,
            })
        elif gene == "AGRN":
            onset_age = rng.gauss(3.0, 4.0)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(48, 20)
            dx_delay = max(4, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "limb_girdle_weakness": rng.random() < 0.90,
                "distal_weakness": rng.random() < 0.60,
                "bulbar": rng.random() < 0.50,
                "salbutamol_response": rng.random() < 0.60,
                "respiratory_niv": rng.random() < 0.35,
            })
        else:  # MUSK
            onset_age = rng.gauss(2.0, 3.5)
            onset_age = max(0.0, onset_age)
            dx_delay = rng.gauss(42, 18)
            dx_delay = max(3, dx_delay)
            patients.append({
                "id": f"{gene}-{seed}-{pid:03d}",
                "onset_age": round(onset_age, 1),
                "dx_delay_months": round(dx_delay, 1),
                "seronegative": True,
                "limb_girdle_weakness": rng.random() < 0.75,
                "bulbar": rng.random() < 0.55,
                "ephedrine_response": rng.random() < 0.65,
                "ophthalmoplegia": rng.random() < 0.40,
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
    for i, gene in enumerate(CMS_GENES):
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
        "atlas": "Hereditary-CMS-Atlas — Complete 8-Gene Hereditary Congenital Myasthenic Syndrome Atlas",
        "subtitle": (
            "CHRNE (CMS4 — AChR Deficiency / Pyridostigmine) · RAPSN (CMS11 — Apneic Crises / Neonatal) · "
            "DOK7 (CMS10 — Limb-Girdle / AChEI-CI / Salbutamol) · COLQ (CMS5 — AChE Absent / AChEI-CI / Pupils) · "
            "CHAT (CMS6 — Pre-Synaptic / Fatal Apnea / Prophylactic AChEI) · "
            "SCN4A (CMS16 — Sodium Channel / Periodic Paralysis Overlap) · "
            "AGRN (CMS8 — Agrin-MuSK Pathway / Limb-Girdle+Distal) · "
            "MUSK (CMS9 — MuSK-Kinase / NOT Autoimmune MuSK-MG) — 320 Patients (8×40, Seeds 1734–1741)"
        ),
        "total_patients": 320,
        "seed_range": "1734–1741",
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
            "CMS vs Myasthenia Gravis — Genetic vs Autoimmune NMJ Disease": (
                "Congenital Myasthenic Syndromes (CMS) are GENETIC disorders of neuromuscular transmission "
                "caused by mutations in NMJ-related proteins (AChR subunits, clustering proteins, enzymes). "
                "Myasthenia Gravis (MG) is AUTOIMMUNE — caused by antibodies against AChR, MuSK, or LRP4. "
                "Key distinguishing features: "
                "CMS: genetic (AR/AD), congenital/childhood onset, antibody-NEGATIVE (seronegative), "
                "responds to specific pharmacological CMS therapy, NOT immunotherapy. "
                "MG: acquired autoimmune, adult-predominant (though juvenile MG exists), antibody-POSITIVE "
                "in 90% (anti-AChR or anti-MuSK), responds to immunotherapy (steroids, azathioprine, rituximab). "
                "Critical error to avoid: giving immunotherapy for CMS (ineffective) or "
                "missing CMS diagnosis in a child labelled seronegative MG."
            ),
            "AChEI Contraindications in CMS — DOK7 and COLQ": (
                "Anticholinesterase (AChEI) drugs (pyridostigmine, neostigmine) are first-line for "
                "most post-synaptic AChR-deficiency CMS (e.g. CHRNE), but are ABSOLUTELY CONTRAINDICATED "
                "in DOK7-CMS10 and COLQ-CMS5. "
                "DOK7 — AChEI cause rapid clinical worsening: mechanism unclear but consistently harmful; "
                "patients placed on pyridostigmine by mistake deteriorate within days; "
                "this is one of the most dangerous CMS prescribing errors. "
                "COLQ — endplate AChE is already absent: adding AChEI further increases ACh accumulation "
                "in the cleft → more prolonged receptor depolarisation → worsening desensitisation block. "
                "Rule: always establish the GENETIC subtype before prescribing AChEI in CMS. "
                "In a child with limb-girdle weakness and minimal EOM involvement — suspect DOK7 first; "
                "check GENE before prescribing."
            ),
            "Agrin-LRP4-MuSK-Dok7 Signalling Cascade — Four-Gene CMS Family": (
                "Neural agrin (AGRN) released by motor neurons binds LRP4 on the muscle surface. "
                "LRP4 activates MuSK (muscle-specific kinase). "
                "MuSK recruits Dok-7, which activates downstream clustering events including Rapsyn recruitment. "
                "Rapsyn (RAPSN) anchors AChR (subunits CHRNA1, CHRNB1, CHRND, CHRNE, CHRNG) into clusters. "
                "Mutations in AGRN, LRP4 (not in this atlas), MUSK, DOK7 — all cause similar "
                "limb-girdle CMS phenotypes with preserved extraocular motility (distinguishes from CHRNE/RAPSN). "
                "Treatment for this pathway family: salbutamol/ephedrine (promote NMJ maturation); "
                "avoid AChEI in DOK7 and COLQ; pyridostigmine variable in others. "
                "Key insight: same pathway, different steps — overlapping phenotypes require genetic diagnosis."
            ),
            "CHAT Pre-Synaptic CMS — Episodic Fatal Apnea and Temperature Sensitivity": (
                "CHAT-CMS6 is PRE-SYNAPTIC: the problem is INSIDE the motor nerve terminal (ACh synthesis), "
                "not at the endplate (post-synaptic). This has unique implications: "
                "Between episodes, patients can appear relatively normal or only mildly weak — "
                "the NMJ works adequately at rest when ACh stores are maintained. "
                "During sustained activity, fever, or sleep, ACh stores deplete below the safety factor → "
                "sudden neuromuscular block → potentially fatal respiratory arrest. "
                "This deceptive inter-episodic normality leads to delayed diagnosis and underestimated risk. "
                "The key principle: treat PROPHYLACTICALLY (pyridostigmine 24h/day, home monitoring) "
                "to maintain quantal ACh even when the patient seems well."
            ),
            "SCN4A Channelopathy — CMS16 vs Periodic Paralysis vs Myotonia": (
                "SCN4A (NaV1.4) mutations cause three distinct disease categories depending on mechanism: "
                "LOF (biallelic AR) → CMS16: muscle inexcitability, congenital weakness, myasthenic phenotype. "
                "GOF (heterozygous AD, rapid inactivation failure) → Hyperkalemic Periodic Paralysis type 2: "
                "episodic weakness triggered by rest-after-exercise or K+-rich foods; serum K+ elevated during attack. "
                "GOF (heterozygous AD, cold-sensitive) → Paramyotonia Congenita: "
                "myotonia worsened by cold (paradoxical — worsens with activity in cold). "
                "GOF (heterozygous AD, various) → Myotonia fluctuans / Sodium channel myotonia. "
                "Understanding mechanism (LOF vs GOF) is critical: treatment is opposite — "
                "LOF CMS16: quinidine or symptom-directed; GOF: membrane-stabilisers (mexiletine, acetazolamide). "
                "A patient can have compound heterozygous (LOF + partial GOF) presenting with mixed phenotype."
            ),
            "MuSK-CMS9 vs Autoimmune MuSK-MG — Critical Not-to-Confuse": (
                "MUSK-CMS9 (this atlas) and autoimmune MuSK-MG are completely different diseases targeting the "
                "same protein — confusion is clinically dangerous. "
                "CMS9: GENETIC (AR LOF mutations in MUSK gene); childhood onset; antibody-NEGATIVE; "
                "treatment: ephedrine/salbutamol; NO immunotherapy. "
                "Autoimmune MuSK-MG: AUTOIMMUNE (anti-MuSK IgG4 antibodies); adult predominance, female-biased; "
                "antibody-POSITIVE (anti-MuSK); treatment: rituximab (B-cell depleting); "
                "pyridostigmine poorly tolerated (muscarinic side effects prominent, worsening in some). "
                "Both can present with fatigable weakness, bulbar involvement, and variable EOM involvement. "
                "Anti-MuSK antibody testing is MANDATORY to distinguish these two before treatment."
            ),
        },
        "pharmacological_distinctions": [
            "CHRNE/RAPSN (CMS4/11 post-synaptic AChR deficiency) — "
            "Pyridostigmine 1st-line: 1-5 mg/kg/day in 3-4 divided doses (paediatric); "
            "3,4-DAP (amifampridine) adjunct; "
            "monitor QTc with 3,4-DAP (risk of long QT — low but check baseline ECG); "
            "combination pyridostigmine + 3,4-DAP often superior to monotherapy",
            "DOK7 (CMS10) — Salbutamol (albuterol) β2-agonist FIRST-LINE: 2 mg TID adults; "
            "paediatric dose lower (specialist guidance); "
            "AChEI (pyridostigmine) ABSOLUTELY CONTRAINDICATED — causes rapid worsening; "
            "ephedrine alternative if salbutamol not tolerated; "
            "response assessment at 3 months minimum",
            "COLQ (CMS5) — 3,4-DAP + salbutamol combination: "
            "3,4-DAP enhances ACh release pre-synaptically without increasing endplate ACh overload; "
            "AChEI ABSOLUTELY CONTRAINDICATED; "
            "ECG baseline for 3,4-DAP; "
            "ephedrine alternative",
            "CHAT (CMS6) — Pyridostigmine PROPHYLACTIC (not PRN): "
            "continuous round-the-clock dosing essential even without symptoms; "
            "dose increases mandatory during fever (consult specialist); "
            "3,4-DAP adjunct pre-synaptically; "
            "home SpO2 monitoring mandatory; antipyretics aggressively",
            "SCN4A (CMS16/channelopathies) — depends on LOF vs GOF: "
            "LOF-CMS16: quinidine (membrane stabiliser) or specialist-directed; "
            "GOF-HypPP2: acetazolamide or dichlorphenamide; avoid K+ supplements/high K+ foods; "
            "GOF-Paramyotonia: mexiletine (sodium channel blocker) for myotonia; "
            "avoid cold; keep environment warm; "
            "GOF-Myotonia: mexiletine 1st-line; carbamazepine alternative",
            "AGRN (CMS8) — Salbutamol + pyridostigmine (variable) combination: "
            "salbutamol promotes NMJ maturation downstream; "
            "pyridostigmine NOT contraindicated (unlike DOK7) but response variable; "
            "3,4-DAP adjunct; "
            "no strong contraindications — trial of combined approach under specialist",
            "MUSK (CMS9) — Ephedrine or albuterol/salbutamol preferred: "
            "cAMP pathway NMJ maturation; "
            "pyridostigmine variable — do NOT expect reliable full response; "
            "immunotherapy NOT indicated (this is genetic, not autoimmune); "
            "anti-MuSK antibody testing FIRST to exclude autoimmune MuSK-MG before treatment",
        ],
        "key_standards": [
            "CMS DIAGNOSTIC STANDARD: CMS panel sequencing (CHRNE, RAPSN, DOK7, COLQ, CHAT, SCN4A, AGRN, MUSK, "
            "plus CHRNA1, CHRNB1, CHRND, CHRNG, LRP4, GFPT1, ALG2, ALG14, DPAGT1, and others) "
            "is the gold standard; preceded by anti-AChR and anti-MuSK antibody testing to exclude autoimmune MG",
            "NMJ ELECTROPHYSIOLOGY: repetitive nerve stimulation (RNS) at 3 Hz for decrement; "
            "SFEMG (single-fibre EMG) for jitter — most sensitive NMJ test; "
            "post-exercise facilitation vs decrement pattern guides CMS subtype; "
            "EMG myotonic discharges suggest SCN4A channelopathy",
            "RESPIRATORY MONITORING ALL CMS: FVC sitting + supine at every visit; "
            "bulbar assessment; overnight SpO2 in moderate/severe cases; "
            "early NIV consideration; emergency plans for intercurrent illness",
            "ANAESTHESIA PRECAUTIONS CMS: CMS patients have increased anaesthesia sensitivity; "
            "depolarising NMBs (succinylcholine) should generally be avoided; "
            "non-depolarising NMBs used with dose reduction and neuromuscular monitoring; "
            "anaesthetist must be informed of CMS diagnosis and specific gene before any surgery",
            "FEVER MANAGEMENT PLAN: all CMS patients (especially CHAT, RAPSN) must have "
            "a written emergency fever management plan including: "
            "when to seek emergency care; dose adjustments; respiratory monitoring protocol; "
            "emergency contact for neuromuscular specialist",
            "GENETIC COUNSELLING: all CMS subtypes require formal genetic counselling; "
            "cascade testing of siblings (25% risk for AR); "
            "prenatal testing offered for severe CMS (especially RAPSN-FADS, CHAT-apnea, CHRNE-severe); "
            "carrier testing for parents; adolescent patients counselled about family planning",
            "MULTIDISCIPLINARY TEAM: neuromuscular neurologist + respiratory physician + "
            "speech and language therapist + physiotherapist + occupational therapist + dietitian + "
            "psychologist + specialist nurse (NMJ disease nurse); "
            "annual MDT review minimum; more frequent in rapidly changing or severe cases",
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
