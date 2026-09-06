#!/usr/bin/env python3
"""Hereditary-Autoinflammatory-Atlas — Complete 8-Gene Hereditary Autoinflammatory Disorders Atlas
MEFV     (Pyrin / Marenostrin; 681 aa; ~78 kDa; 16p13.3; AR (classic) / AD (MEFV-AD);
          OMIM gene 608107; Familial Mediterranean Fever OMIM 249100;
          most common hereditary periodic fever — 1:200–1:1,000 in Mediterranean/Middle-Eastern populations;
          colchicine FIRST LINE — prevents attacks and amyloidosis; IL-1 biologics for colchicine-resistant;
          SAA amyloidosis: the lethal complication — serum amyloid A monitoring mandatory;
          seed SEED_BASE+0) ·
TNFRSF1A (TNF Receptor Superfamily 1A / TNFR1; 455 aa; ~51 kDa; 12p13.31; AD;
          OMIM gene 191190; TNF Receptor Associated Periodic Syndrome OMIM 142680;
          prolonged fever attacks (>7 days vs FMF <3 days); migratory myalgia; periorbital oedema;
          low-penetrance variants (R92Q, P46L) require SAA-guided treatment;
          etanercept FIRST LINE, canakinumab for refractory;
          seed SEED_BASE+1) ·
MVK      (Mevalonate kinase; 396 aa; ~42 kDa; 12q24.11; AR;
          OMIM gene 251170; Mevalonate Kinase Deficiency / HIDS OMIM 260920;
          hyperimmunoglobulinaemia D (IgD often >100 IU/mL); cervical lymphadenopathy PATHOGNOMONIC;
          urine mevalonic acid elevated during attacks; urinary MMA elevated in severe (MA);
          canakinumab FDA-approved 2021 for MKD; geranylgeranyl pyrophosphate deficit mechanism;
          seed SEED_BASE+2) ·
NLRP3    (NOD-Like Receptor Protein 3 / Cryopyrin; 1036 aa; ~118 kDa; 1q44; AD;
          OMIM gene 606416; CAPS spectrum — FCAS OMIM 120100, MWS OMIM 191900, NOMID OMIM 607115;
          canakinumab FIRST LINE for all CAPS — 150 mg q8w (MWS/NOMID); cold-triggered urticarial rash;
          NOMID: chronic meningitis + optic disc swelling + cochlear damage + arthropathy — most severe;
          IL-18 and IL-1β both elevated; rilonacept and anakinra also effective;
          seed SEED_BASE+3) ·
NOD2     (Nucleotide-binding Oligomerisation Domain 2; 1040 aa; ~113 kDa; 16q12.1; AD;
          OMIM gene 605956; Blau Syndrome OMIM 186580; Early-Onset Sarcoidosis OMIM 609464;
          TRIAD: granulomatous skin rash + symmetric granulomatous polyarthritis + granulomatous uveitis PATHOGNOMONIC;
          onset <4 years (earlier than adult sarcoidosis); NOD2 somatic variants → de novo early-onset sarcoidosis;
          TNF-alpha blockers (adalimumab/infliximab) or methotrexate for uveitis;
          seed SEED_BASE+4) ·
IL1RN    (Interleukin-1 receptor antagonist; 177 aa / 152 aa mature; ~17 kDa; 2q14.1; AR;
          OMIM gene 147679; DIRA (Deficiency of IL-1 Receptor Antagonist) OMIM 612852;
          neonatal-onset: pustular skin rash + periostitis + multifocal osteomyelitis — no fever paradox;
          anakinra (IL-1Ra) CURATIVE — dramatic response within days; must be given continuously lifelong;
          bone changes (periosteal elevation, flared metaphyses) on X-ray PATHOGNOMONIC;
          seed SEED_BASE+5) ·
ADA2     (Adenosine Deaminase 2; 511 aa; ~59 kDa; 22q11.1; AR;
          OMIM gene 607575; DADA2 (Deficiency of ADA2) OMIM 615688;
          recurrent ischaemic stroke in childhood (lacunar infarcts) + livedo racemosa + vasculitis;
          TNF blockers (etanercept/adalimumab) PREVENT STROKE — start immediately on diagnosis;
          HSCT CURATIVE for haematological complications (cytopenias, lymphoma risk);
          ADA2 enzyme activity assay (not genetics alone) required for diagnosis;
          seed SEED_BASE+6) ·
PSTPIP1  (Proline-Serine-Threonine Phosphatase Interacting Protein 1; 416 aa; ~47 kDa; 15q24.3; AD;
          OMIM gene 606347; PAPA Syndrome OMIM 604416; PAMI Syndrome OMIM 617960;
          PAPA TRIAD: Pyogenic Arthritis (sterile, destructive) + Pyoderma Gangrenosum + Acne (severe);
          IL-1 biologics (anakinra/canakinumab) OR TNF blockers (adalimumab) — evidence for both;
          path-e-rgy reaction positive (pathergy test); skin biopsy shows neutrophilic infiltrate;
          seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1550–1557)
"""

import random

SEED_BASE = 1550

AUTOINFLAMMATORY_GENES = [
    # ── MEFV — Familial Mediterranean Fever ──
    {
        "gene": "MEFV",
        "protein": "Pyrin / Marenostrin — Familial Mediterranean Fever, Colchicine First-Line, IL-1 for Resistance, SAA Amyloidosis Prevention",
        "alias": (
            "MEFV; OMIM gene 608107; FMF OMIM 249100; 16p13.3; 681 aa; ~78 kDa; "
            "MEFV encodes pyrin (also called marenostrin), a 781-amino acid protein (681 aa in most "
            "databases; transcript-dependent) expressed in neutrophils, monocytes, dendritic cells, "
            "and synovial fibroblasts. Pyrin functions as a critical negative regulator of "
            "the NLRP3 inflammasome and independently assembles its own pyrin inflammasome. "
            "Normal pyrin: binds 14-3-3 proteins (phosphorylated at Ser208/Ser242) to maintain "
            "an inactive state; bacterial toxins (RhoA-inactivating toxins, e.g. C. difficile TcdB) "
            "or FMF-mutant pyrin releases 14-3-3, activating the pyrin inflammasome → IL-1β + IL-18 "
            "processing and secretion → systemic inflammation. Most common hereditary periodic fever "
            "worldwide: 1:200 in Armenians, Sephardic Jews, Arabs, Turks; 1:1,000 in other groups. "
            "Classic pathogenic variants: M694V (most common, most severe — amyloidosis risk highest), "
            "M680I (second most common), V726A, M694I, E148Q (low-penetrance, often compound with other). "
            "Inheritance: classically AR (biallelic pathogenic variants); MEFV-AD phenotype recognised "
            "(monoallelic M694V/M694I/M680I — attenuated but real risk). Clinical: episodic attacks of "
            "fever (38–40°C, 12 hours to 3 days) + serositis — peritonitis (most common, acute abdomen), "
            "pleuritis (unilateral, resolves with fever), pericarditis (rare), synovitis (large joints — "
            "knee/ankle), erysipelas-like erythema (dorsum of foot/ankle). SAA amyloidosis (AA type): "
            "the catastrophic long-term complication — renal amyloidosis → nephrotic syndrome → ESRD; "
            "risk: M694V homozygous >> compound heterozygous >> other; colchicine prevents amyloidosis. "
            "Treatment: COLCHICINE 0.5–1 mg twice daily — reduces attack frequency, prevents SAA "
            "amyloidosis, safe in pregnancy; start at diagnosis and continue lifelong. IL-1 inhibitors "
            "(anakinra, canakinumab, rilonacept) for colchicine-resistant or intolerant patients. "
            "SAA monitoring: serum amyloid A (SAA) monthly — target SAA < 10 mg/L to prevent amyloidosis; "
            "colchicine dose escalation or add IL-1 inhibitor if persistently elevated."
        ),
        "aa": "681 aa",
        "kDa": "~78 kDa",
        "locus": "16p13.3",
        "omim_gene": 608107,
        "omim_disease": 249100,
        "inheritance": "AR (biallelic pathogenic variants); MEFV-AD phenotype (monoallelic M694V/M680I); most common hereditary periodic fever worldwide",
        "gene_class": (
            "MEFV (pyrin/marenostrin) is a 681-amino acid innate immune regulator expressed in "
            "myeloid cells. Pyrin contains: PYRIN domain (PYD, residues 1–92) — interacts with ASC "
            "adapter; B-box zinc finger; coiled-coil domain; C-terminal B30.2/SPRY domain (site of "
            "most pathogenic variants). MEFV variants cluster in exon 10 (B30.2 domain): M694V, "
            "M680I, V726A, A744S, R761H. B30.2 domain variants impair GTPase binding that normally "
            "restrains pyrin → constitutive low-level inflammasome activation. The pyrin inflammasome "
            "is distinct from NLRP3: pyrin recruits ASC → caspase-1 → pro-IL-1β cleavage; also "
            "triggers gasdermin D-mediated pyroptosis. 14-3-3ε/τ binding (phosphorylated Ser208/242) "
            "is the molecular switch: normal pyrin is phosphorylated (inactive); RhoA-toxin exposure "
            "or FMF mutations destabilise 14-3-3 binding → pyrin activation."
        ),
        "n_patients": 40,
        "key_alerts": [
            "MEFV-COLCHICINE-FIRST-LINE: Start at diagnosis; prevents attacks AND SAA amyloidosis — the lethal long-term complication",
            "MEFV-SAA-MONITOR-MONTHLY: Serum amyloid A target < 10 mg/L even between attacks; elevated SAA = occult inflammation causing renal amyloidosis",
            "MEFV-M694V-HIGHEST-RISK: M694V homozygous = highest amyloidosis risk; escalate colchicine to maximum tolerated dose",
            "MEFV-IL1-RESISTANT: Anakinra/canakinumab for colchicine-resistant FMF (attack ≥1/month despite max-dose colchicine)",
            "MEFV-COLCHICINE-PREGNANCY-SAFE: Continue colchicine throughout pregnancy — teratogenicity data does NOT support stopping; stopping risks SAA surge",
            "MEFV-PERITONITIS-LAPAROTOMY-AVOID: Acute abdomen in FMF patient = likely FMF attack NOT appendicitis; careful diagnosis before surgery",
            "MEFV-E148Q-LOW-PENETRANCE: E148Q alone rarely causes disease; biallelic with M694V/M680I can be pathogenic; do not over-diagnose",
        ],
    },
    # ── TNFRSF1A — TRAPS ──
    {
        "gene": "TNFRSF1A",
        "protein": "TNF Receptor 1 — TRAPS, Prolonged Fever >7 Days, Migratory Myalgia, Periorbital Oedema, Etanercept/Canakinumab",
        "alias": (
            "TNFRSF1A; OMIM gene 191190; TRAPS OMIM 142680; 12p13.31; 455 aa; ~51 kDa; "
            "TNFRSF1A encodes TNF Receptor Superfamily member 1A (TNFR1, CD120a), the major "
            "signalling receptor for TNF-α. TNFR1 is expressed ubiquitously and mediates the "
            "inflammatory, apoptotic, and NF-κB signalling effects of TNF-α. TNFR1 normally "
            "undergoes ectodomain shedding (metalloprotease TACE/ADAM17) upon TNF binding — "
            "soluble TNFR1 (sTNFR1) acts as a decoy receptor dampening TNF signalling. "
            "TRAPS variants (missense in cysteine-rich extracellular domains, especially Cys residues "
            "forming disulphide bonds): impair receptor shedding → cell surface TNFR1 accumulates → "
            "sustained NF-κB and MAPK signalling, AND misfolded TNFR1 triggers ER stress and "
            "mitochondrial ROS → amplified NLRP3-dependent and -independent IL-1β release. "
            "Clinical: LONGEST fever episodes of hereditary periodic fevers — typically 7–21 days "
            "(FMF <3 days, HIDS 3–7 days). DISTINCTIVE features: migratory myalgia (descends from "
            "shoulder/arm to hand); periorbital oedema (unilateral or bilateral, periorbital erythema); "
            "centrifugal skin rash (red, painful, migratory, follows myalgia down limb). Serositis "
            "(peritonitis, pleuritis). Conjunctivitis, episcleritis. Lymphadenopathy. "
            "Variants: high-penetrance (Cys mutations e.g. C33Y, C52F, T61I) — structural, severe "
            "disease; low-penetrance (R92Q, P46L) — associated with inflammatory phenotypes but "
            "often not sufficient alone for TRAPS diagnosis (check SAA, family history). SAA "
            "amyloidosis risk for high-penetrance variants: ~10–25% lifetime; monitor SAA. "
            "Treatment: NSAIDs for mild attacks; corticosteroids (prednisolone) for attacks "
            "(note: some patients become corticosteroid-dependent); ETANERCEPT (TNF blocker) "
            "— first-line biologic (TNF-receptor fusion protein, effective in TRAPS); "
            "canakinumab (anti-IL-1β) — superior to etanercept in refractory TRAPS; "
            "infliximab/adalimumab can paradoxically WORSEN TRAPS — avoid monoclonal anti-TNF."
        ),
        "aa": "455 aa",
        "kDa": "~51 kDa",
        "locus": "12p13.31",
        "omim_gene": 191190,
        "omim_disease": 142680,
        "inheritance": "AD — autosomal dominant; high-penetrance (Cys mutations) vs low-penetrance (R92Q/P46L); de novo ~15%",
        "gene_class": (
            "TNFRSF1A (TNFR1/CD120a) is a 455-amino acid type I transmembrane receptor. Structure: "
            "signal peptide (cleaved); 4 cysteine-rich domains (CRD1-4) in extracellular region — "
            "disulphide bonds essential for TNF-binding conformation; transmembrane domain; "
            "intracellular death domain (DD, ~80 aa) recruits TRADD → RIPK1 → NF-κB and caspase-8. "
            "High-penetrance TRAPS variants cluster in CRD1/CRD2 (exons 2-4): disrupt disulphide "
            "bonds → receptor misfolding → impaired ectodomain shedding by ADAM17 → sustained TNFR1 "
            "signalling. R92Q (exon 4) and P46L (exon 3): not predicted to disrupt Cys bonds; "
            "mechanism incompletely understood; low-penetrance; classified as 'variants of uncertain "
            "significance for TRAPS' by some guidelines — use SAA levels and clinical criteria. "
            "Anti-TNF monoclonal antibodies (infliximab, adalimumab) block TNF but do not correct "
            "impaired shedding and may worsen disease — avoid; etanercept (TNFR2-Fc fusion) is "
            "preferred as it mimics the shed sTNFR1 decoy receptor mechanism."
        ),
        "n_patients": 40,
        "key_alerts": [
            "TNFRSF1A-FEVER-7DAYS: TRAPS attacks last 7–21 days — longest of hereditary periodic fevers; FMF < 3 days, HIDS 3–7 days",
            "TNFRSF1A-MONOCLONAL-ANTI-TNF-AVOID: Infliximab/adalimumab can PARADOXICALLY WORSEN TRAPS — use etanercept (TNFR-Fc) or canakinumab instead",
            "TNFRSF1A-PERIORBITAL-OEDEMA: Unilateral periorbital oedema + migratory myalgia = PATHOGNOMONIC combination for TRAPS",
            "TNFRSF1A-SAA-HIGH-PENETRANCE: Cysteine-substituting variants carry 10–25% lifetime amyloidosis risk — monthly SAA monitoring mandatory",
            "TNFRSF1A-R92Q-LOW-PENETRANCE: R92Q/P46L are low-penetrance variants; require elevated SAA + characteristic clinical features before treatment",
            "TNFRSF1A-CANAKINUMAB-REFRACTORY: Canakinumab superior to etanercept for steroid-dependent or frequently relapsing TRAPS",
        ],
    },
    # ── MVK — Mevalonate Kinase Deficiency / HIDS ──
    {
        "gene": "MVK",
        "protein": "Mevalonate Kinase — HIDS/MKD, Cervical Lymphadenopathy PATHOGNOMONIC, IgD Elevated, Canakinumab FDA-2021",
        "alias": (
            "MVK; OMIM gene 251170; MKD/HIDS OMIM 260920; 12q24.11; 396 aa; ~42 kDa; "
            "MVK encodes mevalonate kinase, the second enzyme of the mevalonate (cholesterol "
            "biosynthesis) pathway: converts mevalonic acid to mevalonic acid-5-phosphate using ATP. "
            "MVK is a critical enzyme because the mevalonate pathway generates not only cholesterol "
            "and sterols, but also isoprenoids — geranylgeranyl pyrophosphate (GGPP) and farnesyl "
            "pyrophosphate (FPP) — required for protein prenylation (Rho GTPases, Rab proteins), "
            "heme A synthesis, dolichol, and coenzyme Q10. Loss-of-function MVK variants cause a "
            "SPECTRUM: severe (mevalonic aciduria/MA, MVK activity <1%) — dysmorphic features + "
            "cerebellar atrophy + developmental delay + elevated urine mevalonic acid lifelong + "
            "periodic fever; mild (HIDS, MVK activity 1–10%) — predominantly periodic fever without "
            "severe dysmorphic/neurological features. HIDS mechanism: during fever, pyrexia "
            "further inhibits temperature-sensitive MVK → acute GGPP deficit → Rho GTPase "
            "defarnesylation/degeranylgeranylation → impaired Rho activation → pyrin inflammasome "
            "activation (same final pathway as FMF!) → IL-1β hypersecretion. PATHOGNOMONIC finding: "
            "CERVICAL LYMPHADENOPATHY — present in virtually all attacks (>95%) — distinguishes "
            "MVK/HIDS from FMF (rare) and TRAPS (uncommon). IgD > 100 IU/mL (100 mg/dL) in most "
            "adult HIDS patients; not diagnostic alone (elevated in ~20% of normal adults, absent in "
            "some young children with HIDS). Urine mevalonic acid elevated DURING attacks (not "
            "always interictal). Urine mevalonic acid 1000-fold elevated in MA (severe) — diagnostic. "
            "Treatment: canakinumab (anti-IL-1β) FDA-approved 2021 for MKD — 150 mg q8w; anakinra "
            "also effective. Simvastatin (inhibits HMG-CoA reductase upstream, reduces mevalonate "
            "substrate accumulation) — third-line, some benefit. HSCT curative in MA."
        ),
        "aa": "396 aa",
        "kDa": "~42 kDa",
        "locus": "12q24.11",
        "omim_gene": 251170,
        "omim_disease": 260920,
        "inheritance": "AR — autosomal recessive; severe (MA) <1% MVK activity; mild (HIDS) 1–10% activity; V377I most common HIDS variant",
        "gene_class": (
            "MVK (mevalonate kinase) is a 396-amino acid cytoplasmic enzyme classified in the GHMP "
            "kinase superfamily. Catalytic mechanism: ATP-dependent phosphorylation of (R)-mevalonate "
            "at the 5-hydroxyl group; active site contains conserved Asp304 (catalytic base) and "
            "Lys13. Protein structure: two domains forming ATP-binding and substrate-binding cleft. "
            "The most common HIDS variant V377I (c.1129G>A, exon 11) reduces enzyme thermostability "
            "— activity is borderline at physiological temperature but catastrophically reduced at "
            "fever temperatures (39–40°C), explaining the fever-triggered attack cycle. Other common "
            "HIDS variants: I268T, A334T, H20P. MA variants: more severe truncating/null variants. "
            "Enzyme activity assay (peripheral blood mononuclear cells or fibroblasts) at 37°C is "
            "the gold standard — genetics alone may miss temperature-sensitive variants. Urine "
            "mevalonic acid by GCMS is the metabolite biomarker."
        ),
        "n_patients": 40,
        "key_alerts": [
            "MVK-CERVICAL-LYMPHADENOPATHY-PATHOGNOMONIC: Present in >95% of attacks — most reliable clinical distinguisher from FMF/TRAPS",
            "MVK-IgD-NOT-DIAGNOSTIC-ALONE: IgD > 100 IU/mL supports diagnosis but present in ~20% normals and absent in young children; use with MVK enzyme activity",
            "MVK-CANAKINUMAB-FDA-2021: Only FDA-approved treatment for MKD (canakinumab 150 mg q8w); anakinra effective but off-label",
            "MVK-URINE-MMA-DURING-ATTACK: Collect urine for mevalonic acid DURING fever episode — may be normal between attacks",
            "MVK-FEVER-WORSENS-MVK: Pyrexia further inhibits temperature-sensitive MVK → cycle; antipyretics can help break the fever-attack loop",
            "MVK-ENZYME-ASSAY-MANDATORY: MVK enzyme activity assay required; genetics alone insufficient (V377I activity-reduced but not null)",
            "MVK-HSCT-MA-SEVERE: HSCT curative for mevalonic aciduria (severe form, <1% activity) with haematological/developmental complications",
        ],
    },
    # ── NLRP3 — CAPS Spectrum ──
    {
        "gene": "NLRP3",
        "protein": "Cryopyrin — CAPS Spectrum (FCAS/MWS/NOMID), Cold-Triggered Urticaria, Canakinumab FIRST LINE, Cochlear Damage",
        "alias": (
            "NLRP3; OMIM gene 606416; FCAS OMIM 120100, MWS OMIM 191900, NOMID OMIM 607115; "
            "1q44; 1036 aa; ~118 kDa; "
            "NLRP3 encodes cryopyrin (also called NALP3/PYPAF1), the sensor component of the "
            "NLRP3 inflammasome — the most extensively studied inflammasome complex. Cryopyrin "
            "structure: N-terminal PYD (pyrin domain, residues 1–89) recruits ASC adapter; "
            "NACHT domain (residues 217–534) with ATPase activity — necessary for inflammasome "
            "oligomerisation; C-terminal LRR (leucine-rich repeat, residues 535–1036) — autoinhibitory; "
            "senses pathogen signals (urate crystals, ATP, cholesterol crystals, silica, β-amyloid). "
            "Normal NLRP3 activation: signal 1 (NF-κB → pro-IL-1β, pro-IL-18, NLRP3 transcription) + "
            "signal 2 (K⁺ efflux, mitochondrial ROS, lysosomal destabilisation → NLRP3 oligomerisation "
            "→ ASC speck → pro-caspase-1 cleavage → IL-1β + IL-18 + gasdermin D). CAPS variants: "
            "gain-of-function (GOF) missense in NACHT domain or LRR — constitutive NLRP3 activation "
            "WITHOUT signal 2, IL-1β/IL-18 continuously secreted. CAPS SPECTRUM (genotype-phenotype): "
            "1) FCAS (Familial Cold Autoinflammatory Syndrome) — MILDEST; urticarial rash triggered by "
            "generalised cold exposure (not local cold); fever + arthralgia; no hearing loss; "
            "2) MWS (Muckle-Wells Syndrome) — INTERMEDIATE; urticaria + fever + sensorineural hearing loss "
            "(progressive, cochlear amyloid deposition); nephritis; progressive; "
            "3) NOMID (Neonatal-Onset Multisystem Inflammatory Disease, also called CINCA) — MOST SEVERE; "
            "neonatal-onset; chronic meningitis → intracranial hypertension; optic disc papilloedema → "
            "optic atrophy → visual loss; cochlear damage → deafness; destructive arthropathy "
            "(epiphyseal overgrowth, bony deformity); intellectual disability; characteristic facies "
            "(frontal bossing). IL-18 levels: massively elevated in NOMID (>10,000 pg/mL) — diagnostic. "
            "Treatment: CANAKINUMAB (anti-IL-1β) 150 mg q8w — FIRST LINE for all CAPS; dramatic clinical "
            "and biomarker (CRP, SAA, IL-18) response; prevents cochlear damage and amyloidosis. "
            "Rilonacept (IL-1 Trap) and anakinra also effective. Treatment must be continuous lifelong."
        ),
        "aa": "1036 aa",
        "kDa": "~118 kDa",
        "locus": "1q44",
        "omim_gene": 606416,
        "omim_disease": 191900,
        "inheritance": "AD — autosomal dominant GOF variants; NOMID often de novo; somatic mosaicism in ~35% of NOMID patients (sequencing of multiple tissues/high-sensitivity)",
        "gene_class": (
            "NLRP3 (cryopyrin) is a 1036-amino acid multi-domain innate immune sensor. Domain "
            "organisation: PYD (1–89) — homotypic interactions with ASC-PYD; NACHT/NAIP, "
            "CIITA, HET-E, TP1 domain (217–534) — ATPase activity, Walker A (GxxxxGKS/T) and "
            "Walker B (hhhhDE) motifs; LRR (535–1036) — autoinhibition via intramolecular contact "
            "with NACHT. CAPS-causing GOF variants: R262W (most common), R260W, A439V, L353P "
            "cluster in NACHT domain — disrupt LRR-NACHT autoinhibitory contacts → constitutive "
            "activation. Inflammasome structure: 7 NLRP3 subunits (NACHT-mediated ring) + ASC "
            "speck (oligomeric PYD + CARD filaments) + pro-caspase-1 dimers. Caspase-1 cleaves: "
            "pro-IL-1β (inactive, 37 kDa) → IL-1β (active, 17 kDa); pro-IL-18 → IL-18; gasdermin D "
            "N-terminal fragment → pore-forming membrane insertion → pyroptotic cell death + "
            "IL-1α/HMGB1 release. MCC950 (NLRP3 NACHT inhibitor): investigational; selective "
            "NLRP3 inhibitor binding Walker B motif — in phase 2 trials for CAPS/other inflammasomopathies."
        ),
        "n_patients": 40,
        "key_alerts": [
            "NLRP3-CANAKINUMAB-FIRST-LINE: Canakinumab 150 mg q8w is first-line for all CAPS — prevents cochlear damage, amyloidosis, and meningitis in NOMID",
            "NLRP3-NOMID-AUDIOGRAM-MANDATORY: Annual audiogram for all MWS/NOMID patients — cochlear damage is progressive and irreversible without treatment",
            "NLRP3-OPTIC-DISC-NOMID: Papilloedema in NOMID = intracranial hypertension — ophthalmology urgent + MRI brain + CSF pressure",
            "NLRP3-COLD-TRIGGER-FCAS: Urticarial rash triggered by GENERALISED cold exposure (not just local cold as in cold urticaria); avoid cold environments",
            "NLRP3-SOMATIC-MOSAIC-NOMID: ~35% of NOMID patients have somatic mosaicism — standard sequencing may miss; request ultra-deep sequencing if clinically suspected",
            "NLRP3-IL18-MASSIVELY-ELEVATED: IL-18 > 10,000 pg/mL = NOMID biomarker; CRP + SAA elevated in all CAPS during disease activity",
            "NLRP3-CONTINUOUS-TREATMENT: Stopping canakinumab → rapid disease flare + irreversible cochlear/CNS damage — treatment must be lifelong",
        ],
    },
    # ── NOD2 — Blau Syndrome ──
    {
        "gene": "NOD2",
        "protein": "NOD2 / CARD15 — Blau Syndrome, Granulomatous TRIAD (Skin + Joint + Eye) PATHOGNOMONIC, Onset <4 Years, Anti-TNF Treatment",
        "alias": (
            "NOD2; OMIM gene 605956; Blau Syndrome OMIM 186580; EOS OMIM 609464; "
            "16q12.1; 1040 aa; ~113 kDa; "
            "NOD2 (also known as CARD15) encodes a cytosolic pattern recognition receptor for "
            "bacterial muramyl dipeptide (MDP) — the minimal bioactive fragment of peptidoglycan "
            "from both gram-positive and gram-negative bacteria. NOD2 structure: "
            "2 N-terminal CARDs (caspase activation and recruitment domains) — recruit RIP2 kinase; "
            "NACHT/NBD domain — ATPase, oligomerisation; C-terminal LRR domain — MDP sensing. "
            "Normal NOD2 signalling: MDP binds LRR → NACHT conformational change → CARD recruitment "
            "of RIP2 kinase → XIAP-mediated RIP2 ubiquitination → TAK1 → NF-κB + MAPK → antimicrobial "
            "peptide production + autophagy induction. Blau syndrome (BS) / Early-onset sarcoidosis (EOS): "
            "GOF variants in NACHT domain of NOD2 → constitutive NF-κB activation without MDP → "
            "granuloma formation (CD68+ macrophages fused into Langhans giant cells + CD4+ T cells). "
            "CLINICAL TRIAD — the PATHOGNOMONIC combination: "
            "1) SKIN: Widespread lichenoid/papular rash — biopsy shows non-caseating granulomas; "
            "begins as yellow-brown papules (tangerine skin texture); precedes other manifestations; "
            "2) JOINT: Symmetric granulomatous polyarthritis — boggy synovial tissue cysts (not "
            "fluid); characteristically affects wrists, ankles, small joints of hands/feet; remarkable "
            "preservation of function despite longstanding disease in some; "
            "3) EYE: Granulomatous panuveitis (anterior + intermediate + posterior + panuveitis) — "
            "most serious complication; band keratopathy (calcium deposition, horizontal) PATHOGNOMONIC "
            "of chronic anterior uveitis in children; posterior synechiae; cystoid macular oedema → "
            "visual loss. Onset: typically < 4 years (median 18 months) — distinguishes BS from adult "
            "sarcoidosis (onset >40 years). NOD2 somatic variants (not inherited) → sporadic EOS — "
            "same phenotype, germline negative. Treatment: anti-TNF (adalimumab FIRST LINE for uveitis "
            "in BS; infliximab second-line); methotrexate for skin/joints; ophthalmology co-management "
            "essential. Corticosteroids for acute flares; long-term toxicity limits use."
        ),
        "aa": "1040 aa",
        "kDa": "~113 kDa",
        "locus": "16q12.1",
        "omim_gene": 605956,
        "omim_disease": 186580,
        "inheritance": "AD — autosomal dominant GOF in NACHT domain; de novo ~20% (sporadic EOS has NOD2 somatic variants); Crohn disease risk SNPs (R702W, G908R, L1007fs) are NOT Blau variants",
        "gene_class": (
            "NOD2 (CARD15) is a 1040-amino acid cytosolic innate immune sensor. Domain architecture: "
            "CARD1 (1–115) + CARD2 (116–213) — tandem CARDs recruit RIP2 kinase via homotypic "
            "CARD-CARD interaction; NBD/NACHT (214–577) — ATPase activity essential for "
            "oligomerisation; LRR (578–1040) — bacterial MDP sensing, ~7 LRRs. Blau/EOS variants "
            "cluster in NACHT domain (R334W/Q most common, E383K, W490L, L469F) — distinct from "
            "Crohn disease SNPs (in LRR, loss-of-function). Blau NACHT variants are gain-of-function "
            "(constitutive NF-κB) — mechanistically opposite to Crohn disease loss-of-function. "
            "RIP2 kinase: serine/threonine kinase, phosphorylates TAK1 complex; XIAP E3 ubiquitin "
            "ligase ubiquitinates RIP2 (K63-linkage) → signalling amplification. Granuloma histology: "
            "non-caseating epithelioid granulomas with Langhans giant cells (unlike Crohn disease "
            "granulomas — small, poorly formed); no AFB, no caseation → differentiates from TB."
        ),
        "n_patients": 40,
        "key_alerts": [
            "NOD2-TRIAD-PATHOGNOMONIC: Granulomatous skin rash + polyarthritis + uveitis in child <4 years = Blau Syndrome until proven otherwise",
            "NOD2-UVEITIS-VISUAL-LOSS: Granulomatous panuveitis → band keratopathy + posterior synechiae + CMO → visual loss without aggressive treatment",
            "NOD2-ADALIMUMAB-UVEITIS: Adalimumab (anti-TNF-α) FIRST LINE for Blau uveitis — better evidence than infliximab for paediatric granulomatous uveitis",
            "NOD2-NOT-CROHN-VARIANTS: Blau variants in NACHT domain (R334W/Q) are DIFFERENT from Crohn disease SNPs (R702W/G908R/L1007fs in LRR); do not conflate",
            "NOD2-SOMATIC-EOS: Sporadic early-onset sarcoidosis = NOD2 somatic variants (not inherited); germline negative → test somatic in skin biopsy",
            "NOD2-JOINT-BIOPSY: Boggy synovial cysts (NOT effusions) — aspiration shows minimal fluid; biopsy shows granulomas — confirms diagnosis",
            "NOD2-ONSET-4YEARS: Onset < 4 years (median 18 months) distinguishes Blau from adult sarcoidosis; early diagnosis prevents visual loss",
        ],
    },
    # ── IL1RN — DIRA ──
    {
        "gene": "IL1RN",
        "protein": "IL-1 Receptor Antagonist — DIRA, Neonatal Pustulosis + Periostitis PATHOGNOMONIC, Anakinra CURATIVE Lifelong",
        "alias": (
            "IL1RN; OMIM gene 147679; DIRA OMIM 612852; 2q14.1; 177 aa (secreted); 152 aa (mature); "
            "~17–25 kDa (glycosylated); "
            "IL1RN encodes interleukin-1 receptor antagonist (IL-1Ra), a naturally occurring "
            "competitive antagonist of both IL-1α and IL-1β at the IL-1 receptor type I (IL-1RI). "
            "IL-1Ra structure: shares IL-1 fold (12-stranded beta-trefoil) with IL-1α and IL-1β; "
            "binds IL-1RI with same affinity as IL-1β BUT does not recruit the IL-1 receptor "
            "accessory protein (IL-1RAcP) → NO signal transduction — pure receptor blockade. "
            "Normal physiology: IL-1Ra is the endogenous brake on IL-1 signalling — produced by "
            "monocytes, macrophages, hepatocytes, neutrophils, fibroblasts; circulates in blood; "
            "IL-1Ra/IL-1β molar ratio ~100:1 required to suppress IL-1β signalling. "
            "DIRA mechanism: biallelic loss-of-function IL1RN variants (gene deletions, truncating) "
            "→ absent IL-1Ra → UNCHECKED IL-1α and IL-1β signalling in skin + bone + systemic "
            "inflammation → neonatal-onset disease. CLINICAL features (all present at birth or "
            "within first weeks): "
            "1) SKIN: Generalised sterile pustular rash — non-infectious; neutrophilic infiltrate "
            "on biopsy; may resemble neonatal pustular melanosis or sepsis → delay in diagnosis; "
            "2) BONE: Multifocal periostitis (periosteal elevation on X-ray) + osteolysis — "
            "long bones, ribs (ANCA-like rib deformities), spine; X-ray: periosteal reaction = "
            "PATHOGNOMONIC for DIRA; bone pain → poor feeding + crying; "
            "3) SYSTEMIC: Elevated CRP, neutrophilia, elevated SAA; NO fever in most patients "
            "(paradoxical — despite severe inflammation); heterozygous carriers healthy. "
            "Anakinra (recombinant IL-1Ra, Kineret): CURATIVE — dramatic response within 24–72 hours "
            "of first injection; skin rash resolves completely, periostitis heals; must be given "
            "CONTINUOUSLY LIFELONG — stopping → relapse within days. Dose: 1–4 mg/kg/day SC. "
            "Canakinumab less studied but reported effective. GEOGRAPHICAL PREVALENCE: Newfoundland "
            "Canada founder population (IL1RN exon 4-10 deletion, allele frequency ~1:6 in some "
            "communities); Puerto Rican, Lebanese, Dutch populations also reported."
        ),
        "aa": "177 aa",
        "kDa": "~17 kDa",
        "locus": "2q14.1",
        "omim_gene": 147679,
        "omim_disease": 612852,
        "inheritance": "AR — autosomal recessive; biallelic loss-of-function (deletions > point mutations); heterozygous carriers healthy; Newfoundland founder deletion common",
        "gene_class": (
            "IL1RN (interleukin-1 receptor antagonist) is a 177-amino acid secreted cytokine (mature "
            "form 152 aa after signal peptide cleavage). Structure: single 12-stranded β-trefoil fold "
            "identical to IL-1α/β — three pseudo-symmetric beta-barrel units; binds IL-1RI through "
            "same binding site as IL-1β (Kd ~200 pM) but lacks 'trigger loop' contact with IL-1RAcP "
            "→ receptor occupancy without signalling. Multiple isoforms: secreted IL-1Ra (sIL-1Ra, "
            "transcript 1, main circulating form); intracellular IL-1Ra type I/II/III (icIL-1Ra) — "
            "nuclear and cytoplasmic; the secreted form is lost in DIRA pathogenic variants (deletion "
            "of exons 4-10 removes secreted isoform while intracellular isoforms may be partially "
            "retained — explains why absence of secreted IL-1Ra causes extracellular IL-1 "
            "hypersensitivity particularly in bone/skin). Anakinra (Kineret): E. coli-expressed "
            "recombinant human IL-1Ra with Met-1 addition; half-life 4–6 hours (once-daily SC "
            "injection required); binds IL-1RI and IL-1RII equally."
        ),
        "n_patients": 40,
        "key_alerts": [
            "IL1RN-ANAKINRA-CURATIVE: Anakinra (IL-1Ra) is CURATIVE for DIRA — dramatic response within 24–72 hours; must be given CONTINUOUSLY LIFELONG",
            "IL1RN-PERIOSTITIS-PATHOGNOMONIC: Periosteal elevation on X-ray (long bones + ribs) in neonatal pustulosis = DIRA until proven otherwise",
            "IL1RN-NO-FEVER-PARADOX: DIRA presents WITHOUT fever despite severe systemic inflammation — do not exclude diagnosis due to afebrile status",
            "IL1RN-SEPSIS-MIMIC: Neonatal pustular rash + elevated CRP → often initially treated as neonatal sepsis (cultures negative); IL1RN sequencing essential",
            "IL1RN-STOP-RELAPSE-DAYS: Stopping anakinra → relapse within 24–72 hours; patients must understand lifelong requirement before discharge",
            "IL1RN-NEWFOUNDLAND-FOUNDER: Founder deletion (exons 4-10) prevalent in Newfoundland Canada — targeted deletion PCR faster than WGS in this population",
        ],
    },
    # ── ADA2 — DADA2 ──
    {
        "gene": "ADA2",
        "protein": "Adenosine Deaminase 2 — DADA2, Childhood Stroke + Livedo Racemosa, TNF Blockers PREVENT STROKE, HSCT Curative",
        "alias": (
            "ADA2; OMIM gene 607575; DADA2 OMIM 615688; 22q11.1; 511 aa; ~59 kDa (monomer); "
            "ADA2 (formerly CECR1 — Cat Eye Syndrome Chromosome Region gene 1) encodes "
            "adenosine deaminase 2, a secreted dimeric enzyme that deaminates adenosine and "
            "2'-deoxyadenosine to inosine/2'-deoxyinosine. ADA2 is evolutionarily and structurally "
            "distinct from ADA1 (OMIM 608958 — causes ADA-SCID): ADA2 is a member of the adenosine "
            "deaminase growth factor (ADGF) family; predominantly extracellular; expressed by "
            "monocytes/macrophages; binds heparan sulfate proteoglycans. "
            "DADA2 pathophysiology (complex, incompletely understood): biallelic ADA2 LOF → "
            "adenosine accumulation in extracellular space → adenosine receptor (A2A) stimulation "
            "on endothelial cells + macrophages → dysregulated macrophage polarisation (M1 vs M2 "
            "imbalance → pro-inflammatory M1 predominance) → endothelial damage → small vessel "
            "vasculitis (medium-small arteries) → LACUNAR CEREBRAL INFARCTS + skin vasculitis. "
            "DISTINCTIVE CLINICAL FEATURES: "
            "1) STROKE: Recurrent ischaemic stroke in CHILDHOOD (median onset 5 years) — lacunar "
            "infarcts in basal ganglia/thalamus/brainstem on MRI; may present as hemiplegia; "
            "2) SKIN: Livedo racemosa (branching/reticular fixed skin discolouration — distinguishes "
            "from livedo reticularis which is more regular/net-like); polyarteritis nodosa (PAN)-like "
            "skin nodules; "
            "3) HAEMATOLOGICAL: Cytopenia (neutropenia, anaemia, thrombocytopenia); bone marrow "
            "failure; aplastic anaemia; lymphoma/haematological malignancy risk; "
            "4) SYSTEMIC INFLAMMATION: Elevated CRP, SAA, IL-6; intermittent fever; "
            "5) IMMUNODEFICIENCY: Hypogammaglobulinaemia; reduced NK cells. "
            "ADA2 PLASMA ENZYME ACTIVITY: must be measured — activity < 2 nmol/hr/mL plasma "
            "required for diagnosis (genetics alone insufficient — some variants affect enzyme "
            "activity without classic genotype-phenotype correlation). "
            "Treatment: TNF BLOCKERS (etanercept or adalimumab) — PREVENT STROKE; dramatic "
            "reduction in cerebrovascular events; START IMMEDIATELY on diagnosis. "
            "FRESH FROZEN PLASMA (FFP): provides ADA2 protein — useful for acute management before "
            "biologic is available; ERT (enzyme replacement) not commercially available. "
            "HSCT: CURATIVE for haematological complications (cytopenias, bone marrow failure) but "
            "does NOT prevent stroke reliably — TNF blockers still needed post-HSCT for vasculopathy."
        ),
        "aa": "511 aa",
        "kDa": "~59 kDa",
        "locus": "22q11.1",
        "omim_gene": 607575,
        "omim_disease": 615688,
        "inheritance": "AR — autosomal recessive; biallelic LOF variants; wide ethnic distribution; G47A (Turkey/Middle East), R169Q (North Africa), Y453C (Armenia) common variants",
        "gene_class": (
            "ADA2 (adenosine deaminase 2 / CECR1) is a 511-amino acid secreted glycoprotein. "
            "Structure: signal peptide (cleaved); inhibitory domain (ID, catalytically inactive — "
            "enzyme regulation); deaminase domain (DD) with adenosine-deaminase fold (TIM barrel "
            "modified); putative receptor-binding domain (RB). Active enzyme is a HOMODIMER — "
            "dimerisation is required for catalytic activity; disease variants that disrupt "
            "dimerisation or active-site Zn²⁺ coordination (Asp185, His87, His238, His250) "
            "abolish enzyme activity. ADA2 is structurally distinct from ADA1: ADA1 is cytoplasmic "
            "monomer (363 aa); ADA2 is secreted dimer (511 aa); ADA-SCID is caused by ADA1 LOF "
            "(accumulation of deoxyadenosine → dATP → lymphotoxic) — different mechanism from DADA2 "
            "where ADA2 LOF → extracellular adenosine accumulation → receptor-mediated vasculopathy. "
            "ADA2 plasma enzyme activity assay: substrate (adenosine + EHNA to inhibit ADA1) → "
            "measure inosine production by HPLC; < 2 nmol/hr/mL = diagnostic threshold. "
            "TNF-α promotes endothelial survival/repair; anti-TNF mechanism in DADA2 may be via "
            "improved endothelial function + reduced macrophage-mediated damage."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ADA2-TNF-BLOCKERS-PREVENT-STROKE: Etanercept/adalimumab PREVENT STROKE — start IMMEDIATELY on diagnosis; TNF blockers are the primary treatment for cerebrovascular disease",
            "ADA2-ENZYME-ACTIVITY-ASSAY: ADA2 plasma enzyme activity (< 2 nmol/hr/mL) REQUIRED — genetics alone insufficient as some variants affect activity unexpectedly",
            "ADA2-CHILDHOOD-LACUNAR-STROKE: Recurrent lacunar infarcts in basal ganglia/thalamus/brainstem in child = DADA2 until proven otherwise (test ADA2 enzyme + genetics)",
            "ADA2-LIVEDO-RACEMOSA: Branching/reticular fixed livedo (racemosa) + childhood stroke + cytopenia = DADA2 triad; confirm with enzyme activity",
            "ADA2-HSCT-HAEMATOLOGICAL-ONLY: HSCT cures cytopenias/bone marrow failure but does NOT reliably prevent stroke — continue TNF blocker post-HSCT",
            "ADA2-FFP-ACUTE-MANAGEMENT: Fresh Frozen Plasma provides ADA2 enzyme — bridge therapy while awaiting biologic; not for long-term use",
            "ADA2-NOT-ADA1-SCID: ADA2 ≠ ADA1; DADA2 ≠ ADA-SCID (Strimvelis); different genes, different mechanisms, different treatments",
        ],
    },
    # ── PSTPIP1 — PAPA Syndrome ──
    {
        "gene": "PSTPIP1",
        "protein": "PSTPIP1 / CD2BP1 — PAPA Syndrome, Pyogenic Arthritis + Pyoderma Gangrenosum + Acne TRIAD, IL-1/TNF Biologics",
        "alias": (
            "PSTPIP1; OMIM gene 606347; PAPA Syndrome OMIM 604416; 15q24.3; 416 aa; ~47 kDa; "
            "PSTPIP1 (Proline-Serine-Threonine Phosphatase Interacting Protein 1, also known as "
            "CD2-binding protein 1/CD2BP1) encodes a cytoskeletal scaffold protein expressed "
            "predominantly in myeloid cells (monocytes, neutrophils) and T cells. "
            "PSTPIP1 structure: N-terminal F-BAR domain (membrane curvature sensing) + coiled-coil "
            "domain + SH3 domain (C-terminal). Normal PSTPIP1 function: "
            "- Interacts with pyrin (MEFV protein) via PSTPIP1 coiled-coil + pyrin B-box — "
            "this interaction RESTRAINS pyrin inflammasome activity; "
            "- Interacts with PTPD1 (protein tyrosine phosphatase D1) to regulate Rho GTPase "
            "signalling; involved in podosome/lamellipodia formation and T-cell cytoskeletal "
            "dynamics at immune synapses. "
            "PAPA syndrome mechanism: GOF variants (A230T, E250K — in coiled-coil domain) → "
            "altered PSTPIP1-pyrin binding → ENHANCED pyrin inflammasome activation → IL-1β + IL-18 "
            "hypersecretion in neutrophils/monocytes → tissue-destructive neutrophilic inflammation. "
            "Additional pathway: PSTPIP1 variants → increased PSTPIP1-PTPD1 interaction → "
            "reduced WASP dephosphorylation → actin cytoskeleton dysregulation → neutrophil "
            "hyperactivation. CLINICAL TRIAD (PAPA): "
            "1) PYOGENIC ARTHRITIS: Sterile, destructive, episodic; begins in childhood (mean 6 years); "
            "large joints (knee >> ankle, hip); synovial biopsy = massive neutrophilic infiltrate, "
            "NO bacteria, NO crystals; destructive if untreated (joint ankylosis, cartilage loss); "
            "2) PYODERMA GANGRENOSUM (PG): Ulcerative neutrophilic dermatosis; begins as "
            "erythematous papule/pustule → rapidly enlarging painful ulcer with undermined purple "
            "border; PATHERGY POSITIVE — minor trauma triggers PG; usually trunk/lower limbs; "
            "3) ACNE: Severe cystic acne (nodulocystic, conglobata pattern); onset around puberty; "
            "resistant to standard acne treatments; significant scarring. "
            "Variants: A230T (most common) → milder; E250K → more severe. "
            "PAMI syndrome (PSTPIP1-associated myeloid-related proteinemia inflammatory syndrome): "
            "higher PSTPIP1 expression variants → monocytic inflammation + systemic autoinflammation "
            "without the PAPA triad — different clinical variant. "
            "Treatment: IL-1 biologics (anakinra/canakinumab) effective for arthritis + some PG; "
            "TNF blockers (adalimumab/infliximab) also used; colchicine rarely effective; "
            "systemic corticosteroids for acute PG; wound care for PG ulcers (NO debridement — "
            "pathergy worsens lesions)."
        ),
        "aa": "416 aa",
        "kDa": "~47 kDa",
        "locus": "15q24.3",
        "omim_gene": 606347,
        "omim_disease": 604416,
        "inheritance": "AD — autosomal dominant GOF variants (A230T most common, E250K more severe); de novo reported; reduced penetrance for PG component",
        "gene_class": (
            "PSTPIP1 (CD2BP1) is a 416-amino acid F-BAR/coiled-coil/SH3 domain scaffold protein. "
            "F-BAR domain (1–280): membrane-tubulating module that senses/induces membrane curvature; "
            "dimerises and binds negatively charged phospholipids (PIP2/PIP3). Coiled-coil (200–300): "
            "mediates pyrin binding — PAPA variants A230T and E250K reside here; coiled-coil "
            "interactions determine strength of PSTPIP1-pyrin complex. SH3 domain (330–390): "
            "binds PTPD1 proline-rich motifs. Biochemistry of PAPA variants: A230T → "
            "disrupted PSTPIP1-pyrin interaction → pyrin released from inhibitory complex → "
            "constitutive pyrin inflammasome activation — mechanistically similar to FMF but "
            "via upstream regulators rather than pyrin itself. This shared pathway explains "
            "why colchicine (which targets microtubule dynamics and neutrophil chemotaxis) "
            "may help mildly, but direct IL-1 blockade is more effective. Pathergy test: "
            "intradermal injection of saline or needle prick → induces PG lesion at injection site "
            "— positive in >50% of PAPA patients; reflects neutrophil hyperactivation."
        ),
        "n_patients": 40,
        "key_alerts": [
            "PSTPIP1-PAPA-TRIAD: Pyogenic Arthritis (sterile, destructive) + Pyoderma Gangrenosum + Severe Cystic Acne = PAPA until proven otherwise",
            "PSTPIP1-PG-NO-DEBRIDEMENT: NEVER debride PG lesions — pathergy positive = surgical trauma WORSENS PG; wound care only + systemic biologics",
            "PSTPIP1-PATHERGY-POSITIVE: Pathergy test positive >50% — minor trauma triggers PG; warn patients + surgical teams; any elective surgery requires preoperative biologic cover",
            "PSTPIP1-IL1-OR-TNF: IL-1 biologics (anakinra/canakinumab) most evidence for arthritis; TNF blockers (adalimumab) for PG — trial both if one fails",
            "PSTPIP1-JOINT-BIOPSY-STERILE: Synovial biopsy shows massive neutrophilic infiltrate, NO organisms, NO crystals — differentiates from septic arthritis and gout",
            "PSTPIP1-ACNE-RESISTANT: Severe cystic acne resistant to isotretinoin alone — IL-1 biologic may be needed; skin biopsy shows neutrophilic folliculitis",
            "PSTPIP1-PAMI-DISTINCT: PAMI syndrome (higher-expression variants) presents with monocytic inflammation without PAPA triad — different PSTPIP1 variant class",
        ],
    },
]


def _make_cohort(gene_data):
    rng = random.Random(SEED_BASE + AUTOINFLAMMATORY_GENES.index(gene_data))
    gene = gene_data["gene"]
    pts = []
    for i in range(gene_data["n_patients"]):
        if gene == "MEFV":
            age_dx = rng.gauss(12, 8)
            delay = rng.gauss(30, 18)
            colchicine = rng.random() < 0.95
            colchicine_resistant = rng.random() < 0.20
            il1_biologic = colchicine_resistant and rng.random() < 0.80
            amyloidosis = rng.random() < 0.12
            saa_elevated = rng.random() < 0.30
            cascade_tested = rng.random() < 0.72
            p = {
                "id": f"MEFV-{i+1:03d}",
                "gene": "MEFV",
                "etiology": rng.choice(["M694V/M694V", "M694V/M680I", "M694V/V726A", "M680I/V726A", "M694V/E148Q"]),
                "age_at_diagnosis": max(1, round(age_dx, 1)),
                "dx_delay_months": max(2, round(delay, 0)),
                "attack_duration_days": round(rng.gauss(1.5, 0.5), 1),
                "attacks_per_year": round(rng.gauss(6, 3), 0),
                "colchicine": colchicine,
                "colchicine_resistant": colchicine_resistant,
                "il1_biologic": il1_biologic,
                "amyloidosis": amyloidosis,
                "saa_elevated": saa_elevated,
                "peritonitis": rng.random() < 0.90,
                "pleuritis": rng.random() < 0.40,
                "arthritis": rng.random() < 0.65,
                "erysipelas_like_erythema": rng.random() < 0.20,
                "cascade_tested": cascade_tested,
            }
        elif gene == "TNFRSF1A":
            age_dx = rng.gauss(14, 10)
            delay = rng.gauss(60, 36)
            etanercept = rng.random() < 0.55
            canakinumab = rng.random() < 0.30
            amyloidosis = rng.random() < 0.10
            p = {
                "id": f"TNFRSF1A-{i+1:03d}",
                "gene": "TNFRSF1A",
                "etiology": rng.choice(["C33Y", "C52F", "T61I", "R92Q", "P46L", "C88Y"]),
                "age_at_diagnosis": max(1, round(age_dx, 1)),
                "dx_delay_months": max(6, round(delay, 0)),
                "attack_duration_days": round(rng.gauss(12, 5), 1),
                "attacks_per_year": round(rng.gauss(4, 2), 0),
                "etanercept": etanercept,
                "canakinumab": canakinumab,
                "corticosteroid_dependent": rng.random() < 0.35,
                "amyloidosis": amyloidosis,
                "periorbital_oedema": rng.random() < 0.75,
                "migratory_myalgia": rng.random() < 0.80,
                "serositis": rng.random() < 0.60,
                "conjunctivitis": rng.random() < 0.35,
                "lymphadenopathy": rng.random() < 0.50,
                "cascade_tested": rng.random() < 0.55,
            }
        elif gene == "MVK":
            age_dx = rng.gauss(8, 5)
            delay = rng.gauss(48, 24)
            canakinumab = rng.random() < 0.60
            p = {
                "id": f"MVK-{i+1:03d}",
                "gene": "MVK",
                "etiology": rng.choice(["V377I/V377I", "V377I/I268T", "I268T/A334T", "V377I/H20P"]),
                "age_at_diagnosis": max(1, round(age_dx, 1)),
                "dx_delay_months": max(6, round(delay, 0)),
                "attack_duration_days": round(rng.gauss(5, 2), 1),
                "attacks_per_year": round(rng.gauss(8, 4), 0),
                "canakinumab": canakinumab,
                "igd_elevated": rng.random() < 0.82,
                "cervical_lymphadenopathy": rng.random() < 0.96,
                "abdominal_pain": rng.random() < 0.85,
                "splenomegaly": rng.random() < 0.55,
                "aphthous_stomatitis": rng.random() < 0.50,
                "urine_mva_elevated": rng.random() < 0.70,
                "cascade_tested": rng.random() < 0.60,
            }
        elif gene == "NLRP3":
            age_dx = rng.gauss(3, 5)
            delay = rng.gauss(36, 24)
            caps_type = rng.choice(["FCAS", "MWS", "NOMID"])
            snhl = caps_type in ["MWS", "NOMID"] and rng.random() < 0.75
            p = {
                "id": f"NLRP3-{i+1:03d}",
                "gene": "NLRP3",
                "etiology": rng.choice(["R262W", "R260W", "A439V", "L353P", "V200M", "E311K"]),
                "caps_type": caps_type,
                "age_at_diagnosis": max(0, round(age_dx, 1)),
                "dx_delay_months": max(1, round(delay, 0)),
                "canakinumab": rng.random() < 0.88,
                "urticarial_rash": rng.random() < 0.95,
                "cold_triggered": caps_type == "FCAS" or rng.random() < 0.40,
                "sensorineural_hearing_loss": snhl,
                "optic_disc_swelling": caps_type == "NOMID" and rng.random() < 0.55,
                "chronic_meningitis": caps_type == "NOMID" and rng.random() < 0.65,
                "arthropathy": rng.random() < 0.45,
                "amyloidosis": rng.random() < 0.05,
                "somatic_mosaic": rng.random() < 0.25,
                "cascade_tested": rng.random() < 0.68,
            }
        elif gene == "NOD2":
            age_dx = rng.gauss(2, 1.5)
            delay = rng.gauss(24, 18)
            anti_tnf = rng.random() < 0.78
            p = {
                "id": f"NOD2-{i+1:03d}",
                "gene": "NOD2",
                "etiology": rng.choice(["R334W", "R334Q", "E383K", "W490L", "L469F"]),
                "age_at_diagnosis": max(0.1, round(age_dx, 1)),
                "dx_delay_months": max(3, round(delay, 0)),
                "anti_tnf": anti_tnf,
                "methotrexate": rng.random() < 0.55,
                "granulomatous_skin_rash": rng.random() < 0.98,
                "granulomatous_arthritis": rng.random() < 0.95,
                "granulomatous_uveitis": rng.random() < 0.90,
                "band_keratopathy": rng.random() < 0.35,
                "visual_impairment": rng.random() < 0.20,
                "somatic_variant": rng.random() < 0.18,
                "cascade_tested": rng.random() < 0.75,
            }
        elif gene == "IL1RN":
            age_dx = rng.gauss(0.1, 0.2)
            delay = rng.gauss(3, 2)
            p = {
                "id": f"IL1RN-{i+1:03d}",
                "gene": "IL1RN",
                "etiology": rng.choice(["Ex4-10del", "Ex4-10del/Ex4-10del", "Ex4-10del/point", "p.Q54*/Q54*"]),
                "age_at_diagnosis": max(0.01, round(age_dx, 2)),
                "dx_delay_months": max(0.5, round(delay, 1)),
                "anakinra": rng.random() < 0.98,
                "pustular_rash": rng.random() < 0.99,
                "periostitis": rng.random() < 0.95,
                "multifocal_osteomyelitis": rng.random() < 0.80,
                "fever": rng.random() < 0.25,
                "elevated_crp": rng.random() < 0.99,
                "elevated_saa": rng.random() < 0.99,
                "initial_sepsis_diagnosis": rng.random() < 0.65,
                "cascade_tested": rng.random() < 0.82,
            }
        elif gene == "ADA2":
            age_dx = rng.gauss(7, 4)
            delay = rng.gauss(24, 18)
            tnf_blocker = rng.random() < 0.82
            hsct = rng.random() < 0.25
            p = {
                "id": f"ADA2-{i+1:03d}",
                "gene": "ADA2",
                "etiology": rng.choice(["G47A/G47A", "R169Q/G47A", "Y453C/G47A", "p.Leu351Phe/R169Q"]),
                "age_at_diagnosis": max(0.5, round(age_dx, 1)),
                "dx_delay_months": max(6, round(delay, 0)),
                "tnf_blocker": tnf_blocker,
                "hsct": hsct,
                "ischaemic_stroke": rng.random() < 0.78,
                "livedo_racemosa": rng.random() < 0.70,
                "cytopenias": rng.random() < 0.55,
                "hypogammaglobulinaemia": rng.random() < 0.40,
                "pan_like_vasculitis": rng.random() < 0.30,
                "ada2_activity_low": rng.random() < 0.99,
                "recurrent_strokes": rng.random() < 0.45,
                "cascade_tested": rng.random() < 0.65,
            }
        else:  # PSTPIP1
            age_dx = rng.gauss(10, 5)
            delay = rng.gauss(48, 24)
            il1 = rng.random() < 0.60
            anti_tnf = rng.random() < 0.55
            p = {
                "id": f"PSTPIP1-{i+1:03d}",
                "gene": "PSTPIP1",
                "etiology": rng.choice(["A230T", "A230T", "A230T", "E250K"]),
                "age_at_diagnosis": max(1, round(age_dx, 1)),
                "dx_delay_months": max(6, round(delay, 0)),
                "il1_biologic": il1,
                "anti_tnf": anti_tnf,
                "pyogenic_arthritis": rng.random() < 0.98,
                "pyoderma_gangrenosum": rng.random() < 0.82,
                "severe_acne": rng.random() < 0.88,
                "pathergy_positive": rng.random() < 0.55,
                "joint_destruction": rng.random() < 0.40,
                "cascade_tested": rng.random() < 0.62,
            }
        pts.append(p)
    return pts


def _pct(pts, key):
    if not pts:
        return 0.0
    return round(100.0 * sum(1 for p in pts if p.get(key)) / len(pts), 1)


def get_overview():
    all_pts = []
    gene_pts = {}
    for gd in AUTOINFLAMMATORY_GENES:
        pts = _make_cohort(gd)
        gene_pts[gd["gene"]] = pts
        all_pts.extend(pts)

    mefv = gene_pts["MEFV"]
    tnfrsf1a = gene_pts["TNFRSF1A"]
    mvk = gene_pts["MVK"]
    nlrp3 = gene_pts["NLRP3"]
    nod2 = gene_pts["NOD2"]
    il1rn = gene_pts["IL1RN"]
    ada2 = gene_pts["ADA2"]
    pstpip1 = gene_pts["PSTPIP1"]

    s = {
        "total_patients": len(all_pts),
        "mean_dx_age_years": round(sum(p["age_at_diagnosis"] for p in all_pts) / len(all_pts), 1),
        "mean_dx_delay_months": round(sum(p["dx_delay_months"] for p in all_pts) / len(all_pts), 0),
        "cascade_tested_pct": _pct(all_pts, "cascade_tested"),
        # MEFV FMF
        "mefv_colchicine_pct": _pct(mefv, "colchicine"),
        "mefv_colchicine_resistant_pct": _pct(mefv, "colchicine_resistant"),
        "mefv_il1_biologic_pct": _pct(mefv, "il1_biologic"),
        "mefv_amyloidosis_pct": _pct(mefv, "amyloidosis"),
        "mefv_peritonitis_pct": _pct(mefv, "peritonitis"),
        # TNFRSF1A TRAPS
        "tnfrsf1a_periorbital_oedema_pct": _pct(tnfrsf1a, "periorbital_oedema"),
        "tnfrsf1a_migratory_myalgia_pct": _pct(tnfrsf1a, "migratory_myalgia"),
        "tnfrsf1a_etanercept_pct": _pct(tnfrsf1a, "etanercept"),
        "tnfrsf1a_corticosteroid_dependent_pct": _pct(tnfrsf1a, "corticosteroid_dependent"),
        # MVK HIDS
        "mvk_cervical_lymphadenopathy_pct": _pct(mvk, "cervical_lymphadenopathy"),
        "mvk_igd_elevated_pct": _pct(mvk, "igd_elevated"),
        "mvk_canakinumab_pct": _pct(mvk, "canakinumab"),
        # NLRP3 CAPS
        "nlrp3_canakinumab_pct": _pct(nlrp3, "canakinumab"),
        "nlrp3_urticarial_rash_pct": _pct(nlrp3, "urticarial_rash"),
        "nlrp3_snhl_pct": _pct(nlrp3, "sensorineural_hearing_loss"),
        # NOD2 Blau
        "nod2_granulomatous_uveitis_pct": _pct(nod2, "granulomatous_uveitis"),
        "nod2_anti_tnf_pct": _pct(nod2, "anti_tnf"),
        "nod2_visual_impairment_pct": _pct(nod2, "visual_impairment"),
        # IL1RN DIRA
        "il1rn_anakinra_pct": _pct(il1rn, "anakinra"),
        "il1rn_periostitis_pct": _pct(il1rn, "periostitis"),
        "il1rn_initial_sepsis_pct": _pct(il1rn, "initial_sepsis_diagnosis"),
        # ADA2 DADA2
        "ada2_tnf_blocker_pct": _pct(ada2, "tnf_blocker"),
        "ada2_ischaemic_stroke_pct": _pct(ada2, "ischaemic_stroke"),
        "ada2_livedo_racemosa_pct": _pct(ada2, "livedo_racemosa"),
        "ada2_hsct_pct": _pct(ada2, "hsct"),
        # PSTPIP1 PAPA
        "pstpip1_pyogenic_arthritis_pct": _pct(pstpip1, "pyogenic_arthritis"),
        "pstpip1_pyoderma_gangrenosum_pct": _pct(pstpip1, "pyoderma_gangrenosum"),
        "pstpip1_il1_biologic_pct": _pct(pstpip1, "il1_biologic"),
        "pstpip1_pathergy_positive_pct": _pct(pstpip1, "pathergy_positive"),
    }

    genes_out = []
    for gd in AUTOINFLAMMATORY_GENES:
        pts = gene_pts[gd["gene"]]
        genes_out.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "key_alerts": gd["key_alerts"],
            "mean_dx_age": round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1),
            "mean_dx_delay_months": round(sum(p["dx_delay_months"] for p in pts) / len(pts), 0),
        })

    top_alerts = []
    for gd in AUTOINFLAMMATORY_GENES:
        for alert in gd["key_alerts"][:2]:
            top_alerts.append({"gene": gd["gene"], "alert": alert})

    return {
        "dashboard": "Hereditary Autoinflammatory Disorders Atlas",
        "subtitle": "Complete 8-Gene Hereditary Autoinflammatory Reference — MEFV/TNFRSF1A/MVK/NLRP3/NOD2/IL1RN/ADA2/PSTPIP1",
        "seeds": list(range(SEED_BASE, SEED_BASE + 8)),
        "aggregate_stats": s,
        "top_alerts": top_alerts,
        "genes": genes_out,
    }


def get_breakdown():
    out = {}
    for gd in AUTOINFLAMMATORY_GENES:
        pts = _make_cohort(gd)
        gene = gd["gene"]

        etiol_counts = {}
        for p in pts:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        age_buckets = {"<2": 0, "2–5": 0, "6–10": 0, "11–20": 0, "21–40": 0, ">40": 0}
        for p in pts:
            a = p["age_at_diagnosis"]
            if a < 2:
                age_buckets["<2"] += 1
            elif a < 6:
                age_buckets["2–5"] += 1
            elif a < 11:
                age_buckets["6–10"] += 1
            elif a < 21:
                age_buckets["11–20"] += 1
            elif a < 41:
                age_buckets["21–40"] += 1
            else:
                age_buckets[">40"] += 1

        delay_buckets = {"<6m": 0, "6–12m": 0, "1–2y": 0, "2–5y": 0, ">5y": 0}
        for p in pts:
            d = p["dx_delay_months"]
            if d < 6:
                delay_buckets["<6m"] += 1
            elif d < 12:
                delay_buckets["6–12m"] += 1
            elif d < 24:
                delay_buckets["1–2y"] += 1
            elif d < 60:
                delay_buckets["2–5y"] += 1
            else:
                delay_buckets[">5y"] += 1

        stat_keys = ["cascade_tested"]
        if gene == "MEFV":
            stat_keys += ["colchicine", "colchicine_resistant", "il1_biologic", "amyloidosis",
                          "peritonitis", "pleuritis", "arthritis", "erysipelas_like_erythema"]
        elif gene == "TNFRSF1A":
            stat_keys += ["etanercept", "canakinumab", "corticosteroid_dependent", "amyloidosis",
                          "periorbital_oedema", "migratory_myalgia", "serositis", "conjunctivitis"]
        elif gene == "MVK":
            stat_keys += ["canakinumab", "igd_elevated", "cervical_lymphadenopathy",
                          "abdominal_pain", "splenomegaly", "aphthous_stomatitis", "urine_mva_elevated"]
        elif gene == "NLRP3":
            stat_keys += ["canakinumab", "urticarial_rash", "cold_triggered",
                          "sensorineural_hearing_loss", "optic_disc_swelling", "chronic_meningitis",
                          "arthropathy", "amyloidosis", "somatic_mosaic"]
        elif gene == "NOD2":
            stat_keys += ["anti_tnf", "methotrexate", "granulomatous_skin_rash",
                          "granulomatous_arthritis", "granulomatous_uveitis",
                          "band_keratopathy", "visual_impairment", "somatic_variant"]
        elif gene == "IL1RN":
            stat_keys += ["anakinra", "pustular_rash", "periostitis", "multifocal_osteomyelitis",
                          "fever", "elevated_crp", "elevated_saa", "initial_sepsis_diagnosis"]
        elif gene == "ADA2":
            stat_keys += ["tnf_blocker", "hsct", "ischaemic_stroke", "livedo_racemosa",
                          "cytopenias", "hypogammaglobulinaemia", "pan_like_vasculitis",
                          "ada2_activity_low", "recurrent_strokes"]
        else:
            stat_keys += ["il1_biologic", "anti_tnf", "pyogenic_arthritis",
                          "pyoderma_gangrenosum", "severe_acne", "pathergy_positive", "joint_destruction"]

        stats = {k: _pct(pts, k) for k in stat_keys}
        stats["mean_dx_age"] = round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1)
        stats["mean_dx_delay_months"] = round(sum(p["dx_delay_months"] for p in pts) / len(pts), 0)

        out[gene] = {
            "gene": gene,
            "protein": gd["protein"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
            "n_patients": len(pts),
            "etiologies": etiol_counts,
            "age_at_diagnosis_distribution": age_buckets,
            "dx_delay_distribution": delay_buckets,
            "stats": stats,
            "key_alerts": gd["key_alerts"],
            "patients": pts[:10],
        }
    return out


def get_definitions():
    return {
        "atlas": "Hereditary Autoinflammatory Disorders Atlas — Complete 8-Gene Reference",
        "genes_covered": [gd["gene"] for gd in AUTOINFLAMMATORY_GENES],
        "concepts": {
            "Hereditary_Periodic_Fever": (
                "Group of monogenic disorders characterised by recurrent episodes of systemic "
                "inflammation (fever + serositis + rash + arthritis) without evidence of infection "
                "or autoimmunity. Caused by dysregulated innate immune activation — primarily IL-1β "
                "hypersecretion via inflammasome pathways. Key fevers: FMF (MEFV, <3 days, "
                "Mediterranean), TRAPS (TNFRSF1A, >7 days, AD, migratory myalgia), HIDS/MKD "
                "(MVK, 3–7 days, AR, cervical LAP, IgD elevated), CAPS (NLRP3, AD GOF, cold-triggered). "
                "SAA amyloidosis: long-term complication of inadequately controlled FMF/TRAPS/MKD — "
                "monitor serum amyloid A (SAA target < 10 mg/L)."
            ),
            "Pyrin_Inflammasome": (
                "Cytosolic multi-protein complex activated by MEFV (pyrin) and regulated by PSTPIP1. "
                "Distinct from NLRP3 inflammasome: pyrin senses bacterial RhoA-modifying toxins (e.g. "
                "C. difficile TcdB) via 14-3-3 release → ASC recruitment → caspase-1 → IL-1β + IL-18. "
                "FMF variants (M694V/M680I in B30.2 domain): impair 14-3-3 binding → constitutive low-level "
                "activation. PAPA variants (PSTPIP1 A230T): altered pyrin-PSTPIP1 binding → same downstream "
                "consequence. Colchicine targets microtubule dynamics + pyrin SUMOylation — therapeutic "
                "mechanism in FMF. Both FMF and PAPA respond to IL-1 blockade."
            ),
            "NLRP3_Inflammasome": (
                "The major innate immune sensor complex responding to PAMPs/DAMPs (urate crystals, ATP, "
                "cholesterol crystals, silica, β-amyloid). CAPS GOF variants activate NLRP3 without signal 2 "
                "→ constitutive IL-1β + IL-18 secretion. Spectrum from FCAS (cold-triggered, mild) → MWS "
                "(hearing loss) → NOMID (neonatal, meningitis, blindness, deafness — most severe). "
                "Canakinumab (anti-IL-1β, 150 mg q8w) is first-line for all CAPS. IL-18 massively elevated "
                "in NOMID (>10,000 pg/mL) — diagnostic and monitoring biomarker."
            ),
            "SAA_Amyloidosis": (
                "Systemic AA (serum amyloid A) amyloidosis — secondary amyloidosis complicating chronic "
                "recurrent inflammation. SAA is an acute-phase reactant produced by the liver; when chronically "
                "elevated, SAA deposits as amyloid fibrils in kidneys (most common — nephrotic syndrome → "
                "ESRD), liver, spleen, heart. Prevention: maintain SAA < 10 mg/L (colchicine in FMF; "
                "biologics in FMF-resistant/TRAPS/MKD). Monitoring: serum SAA monthly (not just during "
                "attacks); renal function annual (creatinine + urine protein-creatinine ratio). Treatment "
                "of established amyloidosis: control underlying inflammation + eprodisate (limited evidence)."
            ),
            "Canakinumab": (
                "Fully human IgG1/κ monoclonal antibody (anti-IL-1β, Novartis/Ilaris). FDA-approved for: "
                "CAPS (all subtypes), SJIA, AOSD, TRAPS, HIDS/MKD, FMF (colchicine-resistant). "
                "Mechanism: binds and neutralises human IL-1β with high affinity (Kd ~0.8 pM); "
                "does not bind IL-1α. Dosing: 150 mg SC q8w (adults/children ≥40 kg); weight-based in "
                "smaller children. Long half-life (~26 days) — once-monthly dosing alternative. "
                "Monitoring: LFTs, CBC, TB screening before initiation (as with all biologics). "
                "Distinguished from anakinra (short-acting, daily SC, blocks both IL-1α and IL-1β "
                "by competitive receptor antagonism) — canakinumab preferred for chronic therapy."
            ),
            "Anakinra": (
                "Recombinant human IL-1 receptor antagonist (IL-1Ra, Kineret). Daily SC injection "
                "(short half-life 4–6 hours). FDA-approved: RA, NOMID, DIRA, SJIA (EU). Blocks BOTH "
                "IL-1α and IL-1β at IL-1RI receptor. CURATIVE for DIRA (IL1RN LOF) by replacing absent "
                "endogenous IL-1Ra. Also effective for CAPS, FMF, TRAPS, MKD — often used when "
                "canakinumab unavailable or in acute settings. Must be given CONTINUOUSLY for DIRA — "
                "stopping → relapse within 24–72 hours. Injection site reactions common (erythema, "
                "induration) — rotate injection sites."
            ),
            "Blau_Syndrome": (
                "AD granulomatous autoinflammatory disorder (NOD2 GOF, NACHT domain). Clinical TRIAD: "
                "(1) lichenoid granulomatous skin rash (non-caseating granulomas on biopsy); "
                "(2) granulomatous symmetric polyarthritis (boggy synovial cysts, NOT fluid effusions); "
                "(3) granulomatous panuveitis → band keratopathy → visual loss. "
                "Onset < 4 years (median 18 months). Somatic NOD2 variants → sporadic early-onset "
                "sarcoidosis (EOS) — same phenotype, germline negative. Anti-TNF (adalimumab) "
                "FIRST LINE for uveitis — preserves vision. Band keratopathy = calcium deposition "
                "on cornea (horizontal white line) = PATHOGNOMONIC of chronic anterior uveitis in children."
            ),
            "DADA2_Vasculopathy": (
                "Deficiency of ADA2 — recessive loss of secreted ADA2 enzyme → extracellular adenosine "
                "accumulation → macrophage M1 polarisation + endothelial dysfunction → small/medium vessel "
                "vasculitis. KEY FEATURES: recurrent childhood lacunar ischaemic stroke (basal ganglia/ "
                "thalamus/brainstem on MRI); livedo racemosa (branching, fixed); cytopenias; "
                "hypogammaglobulinaemia. TNF blockers (etanercept/adalimumab) PREVENT STROKE — must be "
                "started immediately on diagnosis. ADA2 plasma enzyme activity assay required (< 2 nmol/hr/mL). "
                "HSCT curative for haematological complications but does NOT prevent vasculopathy reliably."
            ),
            "DIRA_Periostitis": (
                "Deficiency of IL-1 Receptor Antagonist — biallelic IL1RN LOF → absent IL-1Ra → "
                "unchecked IL-1 signalling → neonatal sterile pustulosis + periostitis + osteomyelitis. "
                "PATHOGNOMONIC X-ray: periosteal elevation of long bones + rib periostitis → 'flared "
                "metaphyses' appearance. NO fever in most patients (paradox). Anakinra CURATIVE — "
                "dramatic response within 24–72 hours; MUST be given continuously lifelong — stops = "
                "relapse within 24–72 hours. Newfoundland founder deletion (exon 4-10) prevalent. "
                "Initial misdiagnosis as neonatal sepsis common (cultures negative but CRP/SAA very high)."
            ),
            "PAPA_Neutrophilic": (
                "PAPA syndrome (PSTPIP1 GOF) — pyrin inflammasome activation via altered PSTPIP1-pyrin "
                "binding → neutrophilic tissue-destructive inflammation in skin + joints. TRIAD: pyogenic "
                "arthritis (sterile, destructive) + pyoderma gangrenosum (PG) + acne. CRITICAL RULE: "
                "NEVER debride PG lesions — pathergy positive = surgical trauma worsens PG dramatically. "
                "IL-1 biologics (anakinra/canakinumab) and TNF blockers (adalimumab) both effective. "
                "Pathergy test: needle prick → new PG lesion = positive. Joint biopsy: neutrophilic "
                "infiltrate, NO organisms, NO crystals — confirms sterile pyogenic arthritis."
            ),
            "Serum_Amyloid_A_Monitoring": (
                "SAA (acute-phase reactant, >1000-fold increase over baseline during acute phase) is "
                "the substrate for AA amyloid deposition. Target: SAA < 10 mg/L BETWEEN attacks "
                "(intercritical period) — chronic low-grade elevation causes subclinical amyloid. "
                "Frequency: monthly SAA monitoring for FMF/TRAPS/MKD patients. Elevated SAA despite "
                "colchicine → add IL-1 biologic. Annual renal monitoring (eGFR + urine PCR) in "
                "patients with >10 years of disease or any SAA elevation."
            ),
        },
        "key_standards": [
            "Infevers database (infevers.iuis.org) — mutation registry for hereditary autoinflammatory diseases",
            "Ter Haar NM et al. Ann Rheum Dis 2016 — Eurofever/PRINTO classification criteria for FMF, TRAPS, MKD, CAPS",
            "De Benedetti F et al. NEJM 2018 — canakinumab for TRAPS, MKD, FMF (CLUSTER trial)",
            "Gattorno M et al. Ann Rheum Dis 2019 — canakinumab sustained response in periodic fevers",
            "Kastner DL et al. NEJM 2010 — hereditary autoinflammatory diseases — comprehensive review",
            "Aksentijevich I et al. Arthritis Rheum 2009 — DIRA (IL1RN) original description",
            "Zhou Q et al. Science 2014 — DADA2 (ADA2) original description",
            "Lindor NM et al. Mayo Clin Proc 1997 — PAPA syndrome original description",
            "Manthiram K et al. Nat Rev Rheumatol 2017 — NOD2 Blau syndrome comprehensive review",
            "Ben-Chetrit E et al. Semin Arthritis Rheum 2016 — FMF clinical features and management update",
        ],
        "pharmacological_distinctions": [
            "MEFV-FMF: Colchicine 0.5–1 mg twice daily — FIRST LINE; prevents attacks AND amyloidosis; safe in pregnancy; add anakinra/canakinumab for colchicine-resistant (≥1 attack/month)",
            "TNFRSF1A-TRAPS: Etanercept (TNF-Fc fusion) FIRST LINE biologic — NOT infliximab/adalimumab (monoclonal anti-TNF can WORSEN TRAPS); canakinumab for etanercept-resistant",
            "MVK-HIDS: Canakinumab 150 mg q8w FDA-approved 2021 — dramatic attack reduction; anakinra off-label effective; simvastatin third-line; HSCT for severe MA",
            "NLRP3-CAPS: Canakinumab 150 mg q8w FIRST LINE for MWS/NOMID; 150 mg q8w (or q12w for FCAS) — ALL CAPS require continuous treatment; rilonacept/anakinra alternatives",
            "NOD2-BLAU: Adalimumab (anti-TNF-α) FIRST LINE for uveitis; methotrexate for skin/joints; infliximab second-line; corticosteroids for acute flares only",
            "IL1RN-DIRA: Anakinra (IL-1Ra) 1–4 mg/kg/day SC — CURATIVE; must be CONTINUOUS LIFELONG; canakinumab less-studied alternative; NO stopping ever",
            "ADA2-DADA2: TNF blockers (etanercept/adalimumab) PREVENT STROKE — start IMMEDIATELY; HSCT for haematological complications; FFP as bridge therapy",
            "PSTPIP1-PAPA: IL-1 biologics (anakinra/canakinumab) for arthritis; adalimumab/infliximab for PG; NEVER debride PG (pathergy); no colchicine benefit",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(f"Total patients: {ov['aggregate_stats']['total_patients']}")
    print(f"Genes: {[g['gene'] for g in ov['genes']]}")
    print(f"Seeds: {ov['seeds']}")
    print("\n=== BREAKDOWN (gene list) ===")
    bd = get_breakdown()
    for gene, info in bd.items():
        print(f"  {gene}: {info['n_patients']} pts, mean dx age {info['stats']['mean_dx_age']}y, delay {info['stats']['mean_dx_delay_months']}m")
    print("\n=== DEFINITIONS ===")
    df = get_definitions()
    print(f"Concepts: {len(df['concepts'])}")
    print(f"Standards: {len(df['key_standards'])}")
