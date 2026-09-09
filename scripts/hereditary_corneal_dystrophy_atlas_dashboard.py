#!/usr/bin/env python3
"""Hereditary-Corneal-Dystrophy-Atlas — Complete 8-Gene Hereditary Corneal Dystrophy Atlas
(TGFBI · SLC4A11 · ZEB1 · OVOL2 · TCF4 · COL8A2 · VSX1 · KRT12).

TGFBI    (Transforming Growth Factor Beta-Induced; 683 aa; ~68 kDa; 5q31.1; AD;
          Multiple stromal/Bowman corneal dystrophies — variant-specific:
          R124C → Lattice Dystrophy Type I (LCD-I); R555W → Granular CD Type 1 (GCD1);
          R124H → Avellino (combined granular-lattice, GCD2); R555Q → Thiel-Behnke;
          R124L → Reis-Bücklers. MOST COMMON corneal dystrophy gene worldwide;
          seed SEED_BASE+0).
SLC4A11  (Sodium-Borate Cotransporter; 891 aa; ~99 kDa; 20p13; AR;
          CHED2 — Congenital Hereditary Endothelial Dystrophy type 2;
          BILATERAL DIFFUSE GROUND-GLASS CORNEAL OPACITY AT BIRTH PATHOGNOMONIC;
          nystagmus + photophobia in neonates; most common congenital endothelial dystrophy;
          seed SEED_BASE+1).
ZEB1     (Zinc Finger E-Box Binding Homeobox 1; 1124 aa; ~125 kDa; 10p11.22; AD;
          PPCD3 — Posterior Polymorphous Corneal Dystrophy type 3, most common PPCD gene;
          BAND-LIKE VESICULAR LESIONS ON ENDOTHELIUM PATHOGNOMONIC;
          also Fuchs early-onset; haploinsufficiency → endothelial-to-epithelial metaplasia;
          seed SEED_BASE+2).
OVOL2    (OVO-Like Zinc Finger 2; 323 aa; ~36 kDa; 20p13; AD;
          PPCD1 — Posterior Polymorphous Corneal Dystrophy type 1;
          REGULATORY REGION VARIANTS (5'-UTR / promoter) — coding region usually normal;
          ICE syndrome (iridocorneal endothelial) DDx — ICE is acquired/unilateral, PPCD bilateral;
          seed SEED_BASE+3).
TCF4     (Transcription Factor 4; 667 aa; ~73 kDa; 18q21.2; AD;
          Fuchs Endothelial Corneal Dystrophy type 3 — MOST COMMON HEREDITARY ADULT CORNEAL DYSTROPHY;
          CTG18.1 trinucleotide repeat expansion (>40 repeats) in 79% of Fuchs patients;
          DESCEMET MEMBRANE GUTTAE PATHOGNOMONIC; endothelial cell loss → corneal oedema;
          seed SEED_BASE+4).
COL8A2   (Collagen Type VIII Alpha-2 Chain; 703 aa; ~78 kDa; 1p34.3; AD;
          Fuchs Endothelial Corneal Dystrophy type 1 — EARLY-ONSET FUCHS (onset <40 y);
          L450W and Q455K most common early-onset Fuchs variants; PPCD also reported;
          earlier PKP/DMEK indication than late-onset Fuchs; seed SEED_BASE+5).
VSX1     (Visual System Homeobox 1; 365 aa; ~41 kDa; 20p11.21; AD;
          Keratoconus type 1 (KTCN1) + PPCD type 2;
          CORNEAL THINNING + IRREGULAR ASTIGMATISM + FLEISCHER RING (iron deposition) PATHOGNOMONIC;
          Vogt striae; corneal hydrops emergency; cross-linking (CXL) halts progression;
          seed SEED_BASE+6).
KRT12    (Keratin 12; 505 aa; ~55 kDa; 17q21.2; AD;
          Meesmann Epithelial Corneal Dystrophy (MECD) — most common epithelial corneal dystrophy;
          INTRAEPITHELIAL MICROCYSTS (< 0.1 mm) THROUGHOUT CORNEA PATHOGNOMONIC on slit-lamp;
          fragile epithelium; recurrent erosions rare (milder than KRT3); PAS+ glycogen in cysts;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2366-2373).
"""

import random

SEED_BASE = 2366

CORNEAL_DYSTROPHY_GENES = [
    # -- TGFBI -- Multiple Stromal/Bowman Dystrophies ------------------------------------------
    {
        "gene": "TGFBI",
        "alt_name": (
            "TGFBI (TGFBI-683aa-5q31.1 / AD -- "
            "MOST-COMMON-CORNEAL-DYSTROPHY-GENE-WORLDWIDE -- "
            "VARIANT-SPECIFIC-PHENOTYPES: R124C=LCD-I / R555W=GCD1 / R124H=AVELLINO / R555Q=THIEL-BEHNKE / R124L=REIS-BUCKLERS -- "
            "KERATOCYTE-DERIVED-TGFBI-PROTEIN-ACCUMULATES-IN-STROMA)"
        ),
        "protein": (
            "TGFBI -- 5q31.1 AD -- TGFBI-683aa -- "
            "Transforming-Growth-Factor-Beta-Induced-Protein-ig-h3-68kDa-Extracellular-Matrix -- "
            "Four-Fasciclin-1-FAS1-Domains-F1-F2-F3-F4-Integrin-Binding-RGD-Motif -- "
            "Arginine-124-R124-in-FAS1-Domain-4-Hotspot-80pct-of-TGFBI-variants -- "
            "R124C: Eosinophilic-Lattice-Lines-Amyloid-LCD-I-Congo-Red-Birefringent -- "
            "R124H: Mixed-Granular-Lattice-Deposits-Avellino-Masson-Trichrome-Both-Stain -- "
            "R555W: Discrete-Bread-Crumb-Granular-Deposits-Hyaline-Masson-Trichrome-RED -- "
            "R555Q: Honeycomb-Thiel-Behnke-Bowman-Layer-Fibrocellular-Material -- "
            "OMIM-Gene-601805"
        ),
        "locus": "5q31.1",
        "protein_size": "683 aa / 68 kDa",
        "inheritance": (
            "AD (autosomal dominant); heterozygous gain-of-function / dominant-negative extracellular matrix effect; "
            "complete penetrance for most variants; variable expressivity (LCD-I: R124C homozygous = severe childhood onset); "
            "variant-specific disease — genotype predicts phenotype precisely; "
            "consanguineous populations: homozygous R555W (GCD1) most severe"
        ),
        "disease_category": "TGFBI-related corneal dystrophies — Lattice (LCD-I), Granular (GCD1/GCD2/Avellino), Reis-Bücklers, Thiel-Behnke (IC3D Category 1 all)",
        "disease_pathway": (
            "TGFBI protein (keratoepithelin) is an extracellular matrix glycoprotein secreted by corneal epithelium and keratocytes. "
            "Pathogenic variants cause misfolded TGFBI that accumulates in the stroma or Bowman layer "
            "instead of being degraded by extracellular proteases. "
            "Phenotype is determined by variant location: R124 variants → amyloid or mixed deposits; R555 variants → hyaline deposits. "
            "LCD-I (R124C): amyloid fibrils → branching lattice lines, Congo Red+, recurrent erosions major problem. "
            "GCD1 (R555W): hyaline deposits → discrete grey-white bread-crumb opacities in anterior stroma, "
            "rarely between opacities — 'clear' intervals pathognomonic. "
            "Avellino/GCD2 (R124H): both granular (hyaline) and lattice (amyloid) — Masson and Congo Red both positive. "
            "Reis-Bücklers (R124L): Bowman layer fibrocellular replacement — ring-shaped opacities, "
            "severe recurrent erosions in childhood. "
            "Thiel-Behnke (R555Q): honeycomb subepithelial opacification — milder than Reis-Bücklers."
        ),
        "pathognomonic": (
            "VARIANT-SPECIFIC SLIT-LAMP FINDINGS (all AD, all bilateral, all in clear cornea between lesions initially): "
            "LCD-I (R124C): Branching lattice lines + refractile lines + subepithelial haze; amyloid Congo Red birefringence; "
            "recurrent erosions most frequent complication. "
            "GCD1 (R555W): Discrete grey-white BREAD-CRUMB or 'snowflake' deposits in anterior stroma; "
            "intervening cornea CLEAR pathognomonic (vs PPCD/Fuchs diffuse); Masson Trichrome RED. "
            "GCD2/Avellino (R124H): COMBINED granular + lattice deposits; BOTH Congo Red+ AND Masson Trichrome+; "
            "most common in Avellino Italy and Korea. "
            "Thiel-Behnke (R555Q): HONEYCOMB crumb-like subepithelial opacification; milder erosions than Reis-Bücklers. "
            "Reis-Bücklers (R124L): Geographical grey-white subepithelial opacities; Bowman layer destruction; "
            "severe recurrent erosions childhood onset PATHOGNOMONIC."
        ),
        "treatment": (
            "Recurrent erosions (all TGFBI types): lubricants, bandage contact lens, anterior stromal puncture, "
            "phototherapeutic keratectomy (PTK) with excimer laser — TREATMENT OF CHOICE for erosions and subepithelial lesions; "
            "PTK removes superficial deposits and smooths Bowman layer. "
            "Vision loss from central deposits: PTK for anterior lesions; "
            "Deep Anterior Lamellar Keratoplasty (DALK) or Penetrating Keratoplasty (PKP) for deep stromal deposits. "
            "RECURRENCE AFTER TRANSPLANT: all TGFBI dystrophies recur in graft (host keratocytes migrate and deposit) — "
            "patient counselling mandatory. Timeline: LCD-I 2-14 years, GCD2 fastest recurrence. "
            "No approved pharmacotherapy; siRNA and CRISPR approaches in development."
        ),
        "key_features": [
            "Most common corneal dystrophy gene worldwide",
            "Phenotype precisely determined by specific variant (R124C/R555W/R124H/R555Q/R124L)",
            "All bilateral, symmetric, progressive from second decade",
            "PTK excimer laser first-line for anterior lesions and recurrent erosions",
            "Recurrence in corneal graft universal — counsel all patients pre-transplant",
        ],
        "key_ddx": [
            "GCD1 vs macular corneal dystrophy (MCD/CHST6): GCD1 clear between opacities, MCD diffuse haze",
            "LCD-I vs gelatinous drop-like (GDLD/TACSTD2): GDLD severe early, mulberry surface lesions",
            "Avellino: confirm by genetic testing (R124H) — clinically intermediate",
            "Thiel-Behnke vs Reis-Bücklers: EM shows curly fibers (TB) vs rod-shaped (RB)",
            "Acquired vs hereditary: always bilateral family history; acquired typically asymmetric",
        ],
        "corneal_layer": "Stroma + Bowman layer (variant-dependent)",
        "ic3d_category": "Category 1 (genetic basis well-established, all 5 phenotypes)",
        "recurrence_in_graft": True,
        "ptk_effective": True,
        "erosion_frequency": "high",
        "onset_age": "2nd–3rd decade (adult); Reis-Bücklers: childhood",
        "autosomal_recessive_risk": False,
    },
    # -- SLC4A11 -- CHED2 ------------------------------------------------------------------
    {
        "gene": "SLC4A11",
        "alt_name": (
            "SLC4A11 (SLC4A11-891aa-20p13 / AR -- "
            "CHED2-MOST-COMMON-CONGENITAL-HEREDITARY-ENDOTHELIAL-DYSTROPHY -- "
            "BILATERAL-DIFFUSE-GROUND-GLASS-OPACITY-AT-BIRTH-PATHOGNOMONIC -- "
            "NYSTAGMUS+PHOTOPHOBIA-IN-NEONATES -- "
            "FUCHS-ASSOCIATION-HETEROZYGOUS-CARRIERS)"
        ),
        "protein": (
            "SLC4A11 -- 20p13 AR -- SLC4A11-891aa -- "
            "Sodium-Borate-Cotransporter-NaBC1-99kDa-Electrogenic-Borate-Transporter -- "
            "10-Transmembrane-Spans-Large-Cytoplasmic-N-Terminal-Cytoplasmic-C-Terminal -- "
            "Primary-Function-NH3-H2O-Transport-Endothelial-Fluid-Regulation -- "
            "LOF-Disrupts-Corneal-Endothelial-Hydration-Pump-Stroma-Oedema -- "
            "Heterozygous-Carriers-Fuchs-Risk-2-3x-Elevated -- "
            "p.Arg125His-Most-Common-European-Founder -- "
            "OMIM-Gene-610206-Disease-CHED2-217700"
        ),
        "locus": "20p13",
        "protein_size": "891 aa / 99 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF; CHED2; "
            "heterozygous carriers: Fuchs endothelial corneal dystrophy association (2-3x elevated risk); "
            "p.Arg125His common European founder; p.Glu399Lys and p.Asp485Asn also recurrent; "
            "AD heterozygous variants occasionally FECD (Fuchs) or Harboyan syndrome (CHED + sensorineural hearing loss)"
        ),
        "disease_category": "CHED2 — Congenital Hereditary Endothelial Corneal Dystrophy type 2 (most common congenital endothelial dystrophy); Harboyan syndrome (CHED2 + SNHL) in some biallelic cases",
        "disease_pathway": (
            "SLC4A11 is an electrogenic borate/NH3/H2O cotransporter critical for corneal endothelial pump function. "
            "The endothelium maintains corneal deturgescence by active ion transport (Na/K-ATPase) balanced by SLC4A11-mediated fluid regulation. "
            "Biallelic LOF → endothelial pump failure → stromal oedema → diffuse corneal opacification. "
            "The opacity is present at BIRTH (congenital) distinguishing CHED from adult-onset Fuchs. "
            "Endothelial cells are reduced in number and abnormal ultrastructurally. "
            "Harboyan variant: biallelic severe LOF → also inner ear endolymph transport disrupted → progressive SNHL in addition to CHED. "
            "Heterozygous carriers show corneal endothelial changes compatible with Fuchs spectrum — "
            "possibly due to haploinsufficiency of endothelial pump reserve."
        ),
        "pathognomonic": (
            "BILATERAL DIFFUSE GROUND-GLASS OR MILK-WHITE CORNEAL OPACITY PRESENT AT BIRTH — PATHOGNOMONIC FOR CHED2. "
            "Distribution: entire corneal stroma uniformly affected (unlike TGFBI focal deposits). "
            "Nystagmus: horizontal pendular nystagmus in infants secondary to poor visual fixation — "
            "present in 50-70% and helps distinguish CHED from congenital glaucoma. "
            "Photophobia prominent in first weeks of life. "
            "NO PAIN or recurrent erosions (endothelial not epithelial — unlike epithelial dystrophies). "
            "Corneal thickness (pachymetry) markedly elevated (>700 µm, often >1000 µm) at birth. "
            "Descemet membrane thickened and multilaminar on electron microscopy. "
            "IOP NORMAL — critical DDx from congenital glaucoma (buphthalmos, elevated IOP, Haab striae). "
            "CHED1 (SLC4A11-independent, ABCA1-related?) now reclassified — CHED2 = primary SLC4A11 entity."
        ),
        "treatment": (
            "Corneal transplantation is definitive treatment: "
            "Descemet Membrane Endothelial Keratoplasty (DMEK) or Descemet-Stripping Automated Endothelial Keratoplasty (DSAEK) — "
            "transplant diseased endothelium only, preserving recipient stroma; preferred over PKP when stroma is uninvolved. "
            "Timing: early PKP/DMEK in infancy to prevent amblyopia — patching and optical correction mandatory post-operatively. "
            "Dense amblyopia risk if opacity not cleared before age 7-8 (critical period). "
            "Hearing evaluation: screen all CHED2 patients for Harboyan syndrome (progressive SNHL) — "
            "audiometry from diagnosis annually. "
            "Graft prognosis: excellent with DMEK (endothelial replacement only); "
            "recurrence rate LOW (vs TGFBI) since SLC4A11 is expressed by donor endothelium only."
        ),
        "key_features": [
            "Opacity present at birth — most important distinguishing feature from adult-onset dystrophies",
            "Bilateral diffuse ground-glass cornea — no focal deposits or lattice lines",
            "Nystagmus and photophobia in neonates — visual behaviour abnormal",
            "IOP normal — critical DDx from congenital glaucoma (elevated IOP, Haab striae)",
            "DMEK/DSAEK preferred — stroma usually structurally normal",
            "Screen for Harboyan syndrome (progressive SNHL) in all CHED2 patients",
        ],
        "key_ddx": [
            "Congenital glaucoma: elevated IOP, buphthalmos, Haab striae — CHED2 normal IOP",
            "Peter's anomaly: central corneal opacity with iris adhesions (iridocorneal strands) — anterior segment dysgenesis",
            "CHED1 (now reclassified, ABCA1-related?): stationary vs progressive CHED2 — genetic testing required",
            "Sclerocornea: white opacity extends to limbus, vascularised — CHED2 clear limbus",
            "Infantile nystagmus syndrome (INS): cornea clear — CHED2 has corneal clouding as primary cause",
        ],
        "corneal_layer": "Endothelium + Descemet membrane",
        "ic3d_category": "Category 1 (genetic basis well-established)",
        "recurrence_in_graft": False,
        "ptk_effective": False,
        "erosion_frequency": "none",
        "onset_age": "Congenital (birth)",
        "autosomal_recessive_risk": True,
    },
    # -- ZEB1 -- PPCD3 + Fuchs -------------------------------------------------------------
    {
        "gene": "ZEB1",
        "alt_name": (
            "ZEB1 (ZEB1-1124aa-10p11.22 / AD -- "
            "PPCD3-MOST-COMMON-PPCD-GENE -- "
            "BAND-LIKE-VESICULAR-LESIONS-ENDOTHELIUM-PATHOGNOMONIC -- "
            "ENDOTHELIAL-TO-EPITHELIAL-METAPLASIA-MECHANISM -- "
            "EARLY-ONSET-FUCHS-ASSOCIATION)"
        ),
        "protein": (
            "ZEB1 -- 10p11.22 AD -- ZEB1-1124aa -- "
            "Zinc-Finger-E-Box-Binding-Homeobox-1-ZEB1-ZFHX1B-SIP1-125kDa -- "
            "Two-Zinc-Finger-Clusters-N-Term-C-Term-Central-Homeodomain -- "
            "E-Box-Repressor-Binds-CACCTG-Suppress-Epithelial-Genes-E-Cadherin-KRT12 -- "
            "LOF-Haploinsufficiency-Derepresses-Epithelial-Program-in-Endothelium -- "
            "Endothelial-to-Epithelial-Metaplasia-Fibrous-Tissue-on-Endothelium -- "
            "Also-Expressed-Trabecular-Meshwork-Glaucoma-Risk -- "
            "OMIM-Gene-189909-Disease-PPCD3-609141"
        ),
        "locus": "10p11.22",
        "protein_size": "1124 aa / 125 kDa",
        "inheritance": (
            "AD (autosomal dominant); haploinsufficiency; most common PPCD gene (~50% of PPCD cases); "
            "early-onset Fuchs association (heterozygous ZEB1 variants in Fuchs patients); "
            "frameshift/nonsense most common pathogenic class; "
            "also associated with Mowat-Wilson syndrome (severe neurodevelopmental — usually de novo, different from PPCD variants)"
        ),
        "disease_category": "PPCD3 — Posterior Polymorphous Corneal Dystrophy type 3 (most common PPCD); early-onset Fuchs endothelial corneal dystrophy",
        "disease_pathway": (
            "ZEB1 is a zinc-finger transcriptional repressor that normally maintains corneal endothelial identity "
            "by suppressing epithelial gene expression (E-cadherin, keratins including KRT12, KRT14). "
            "Haploinsufficiency → derepression of epithelial program in endothelial cells → "
            "endothelial-to-epithelial metaplasia → cells acquire epithelial characteristics. "
            "Metaplastic endothelium grows in sheets across posterior cornea, "
            "may extend over trabecular meshwork → secondary glaucoma. "
            "Clinical result: PPCD3 — band-like vesicular lesions on endothelium, iridocorneal adhesions possible. "
            "In Fuchs context: ZEB1 haploinsufficiency sensitises endothelium to additional insults → "
            "guttae formation and earlier endothelial decompensation."
        ),
        "pathognomonic": (
            "BAND-LIKE VESICULAR LESIONS AND BROAD SHEETS ON CORNEAL ENDOTHELIUM PATHOGNOMONIC FOR PPCD3. "
            "Three slit-lamp patterns (Cibis classification): "
            "1. VESICULAR: groups of grey vesicles with surrounding haze ring — most distinctive of PPCD; "
            "2. BAND-LIKE: irregular grey opaque bands with undulating scalloped edges; "
            "3. DIFFUSE: generalised endothelial haze (least specific). "
            "IRIDOCORNEAL ADHESIONS (broad-based, extending to line of Schwalbe, iris processes) in 25-30% — "
            "secondary glaucoma risk distinguishes PPCD from Fuchs; "
            "ICE (iridocorneal endothelial) syndrome DDx: ICE unilateral, acquired, no family history. "
            "BILATERAL (vs ICE unilateral). "
            "Corneal oedema relatively mild unless secondary glaucoma develops. "
            "Endothelial specular microscopy: cells pleomorphic, dark banding, epithelial-like multi-nucleation."
        ),
        "treatment": (
            "Glaucoma management (25-30% of PPCD3): topical IOP-lowering (prostaglandins, beta-blockers); "
            "trabeculectomy or tube shunt for refractory — iridocorneal adhesions complicate drainage surgery. "
            "Corneal decompensation (less common than Fuchs): "
            "DMEK or DSAEK endothelial keratoplasty; PKP if stroma significantly involved. "
            "Many PPCD patients have minimal visual symptoms for decades — conservative management with annual review. "
            "Graft prognosis: reasonable but PPCD endothelial disease recurs in graft epithelialised endothelium — "
            "slower recurrence than TGFBI but documented. "
            "Glaucoma treatment must be optimised before corneal surgery — uncontrolled IOP predicts graft failure."
        ),
        "key_features": [
            "Most common PPCD gene (~50% of genetic PPCD cases)",
            "Band-like vesicular lesions on endothelium — pathognomonic PPCD pattern",
            "Iridocorneal adhesions in ~25-30% → secondary glaucoma risk",
            "Bilateral — critical DDx from ICE syndrome (unilateral, acquired)",
            "Often asymptomatic for years — annual surveillance for IOP mandatory",
            "Early-onset Fuchs association in heterozygous carriers",
        ],
        "key_ddx": [
            "ICE syndrome (iridocorneal endothelial): unilateral, acquired, progressive, no family history — PPCD bilateral, hereditary",
            "Fuchs endothelial: guttae predominant, no iridocorneal adhesions, no vesicular lesions",
            "PPCD1 (OVOL2): clinically similar — genetic testing distinguishes; OVOL2 regulatory variants",
            "PPCD2 (COL8A2): early-onset Fuchs overlap; COL8A2 L450W/Q455K",
            "Mowat-Wilson syndrome (ZEB1 de novo severe): intellectual disability + Hirschsprung — different clinical setting",
        ],
        "corneal_layer": "Endothelium + Descemet membrane",
        "ic3d_category": "Category 1 (genetic basis well-established)",
        "recurrence_in_graft": True,
        "ptk_effective": False,
        "erosion_frequency": "rare",
        "onset_age": "Variable (childhood to adult); often subclinical until 4th–5th decade",
        "autosomal_recessive_risk": False,
    },
    # -- OVOL2 -- PPCD1 --------------------------------------------------------------------
    {
        "gene": "OVOL2",
        "alt_name": (
            "OVOL2 (OVOL2-323aa-20p13 / AD -- "
            "PPCD1-POSTERIOR-POLYMORPHOUS-CORNEAL-DYSTROPHY-TYPE-1 -- "
            "REGULATORY-REGION-VARIANTS-PROMOTER-5-UTR-PATHOGENIC -- "
            "CODING-REGION-USUALLY-NORMAL-KEY-DIAGNOSTIC-PITFALL)"
        ),
        "protein": (
            "OVOL2 -- 20p13 AD -- OVOL2-323aa -- "
            "OVO-Like-Zinc-Finger-2-36kDa-C2H2-Zinc-Finger-Transcription-Factor -- "
            "Represses-Epithelial-Mesenchymal-Transition-Maintains-Epithelial-State -- "
            "In-Endothelium-Represses-Inappropriate-Epithelial-Gene-Expression -- "
            "Similar-to-ZEB1-Mechanism-Both-Maintain-Endothelial-Identity -- "
            "PPCD1-Regulatory-Variants-Reduce-OVOL2-Promoter-Activity -- "
            "Ectopic-Expression-of-OVOL2-in-Endothelium-via-De-Novo-Regulatory-Element -- "
            "OMIM-Gene-616441-Disease-PPCD1-122000"
        ),
        "locus": "20p13",
        "protein_size": "323 aa / 36 kDa",
        "inheritance": (
            "AD (autosomal dominant); regulatory region variants (promoter/5'-UTR) — most important diagnostic caveat: "
            "coding sequence sequencing alone will MISS PPCD1; dedicated promoter/regulatory region sequencing or "
            "MLPA/CNV analysis required for complete genetic diagnosis; "
            "rare de novo regulatory variants also cause PPCD1; "
            "some families show apparent de novo occurrence due to low penetrance parents"
        ),
        "disease_category": "PPCD1 — Posterior Polymorphous Corneal Dystrophy type 1 (earliest described PPCD locus, chromosome 20p13)",
        "disease_pathway": (
            "OVOL2 is a C2H2-type zinc finger transcription factor that represses epithelial-mesenchymal transition (EMT) genes. "
            "In corneal endothelium: OVOL2 suppresses inappropriate epithelial gene programmes, maintaining endothelial cell identity. "
            "PPCD1 pathogenic regulatory variants create or enhance a promoter/enhancer element that drives "
            "inappropriate OVOL2 overexpression in developing endothelium OR, alternatively, "
            "haploinsufficiency of OVOL2 regulatory control derepresses endothelial EMT — "
            "the exact mechanism (gain vs loss of function via regulatory element) is still being characterised. "
            "Net result: endothelial cells acquire epithelial characteristics (metaplasia), forming abnormal sheets. "
            "Pathomechanism parallels ZEB1/PPCD3 — both converge on endothelial-to-epithelial metaplasia. "
            "OVOL2 and ZEB1 together account for majority of genetically solved PPCD cases."
        ),
        "pathognomonic": (
            "CLINICALLY INDISTINGUISHABLE FROM PPCD3 (ZEB1) — genetic testing required for PPCD1 vs PPCD3 distinction. "
            "Slit-lamp findings: vesicular lesions, band-like opacities, or diffuse endothelial haze (Cibis types 1-3). "
            "BILATERAL. Iridocorneal adhesions possible (secondary angle-closure/glaucoma risk). "
            "KEY DIAGNOSTIC PITFALL: standard coding-sequence gene panel will NOT detect PPCD1 "
            "if only exons are sequenced — regulatory region sequencing or MLPA mandatory for PPCD workup. "
            "Specular microscopy: endothelial pleomorphism, epithelial-like cells, dark banding. "
            "PPCD1 tends to have milder corneal oedema than CHED2 and less glaucoma than PPCD3 on average "
            "but clinical overlap is significant."
        ),
        "treatment": (
            "Same principles as PPCD3 (ZEB1): "
            "IOP monitoring and glaucoma treatment if iridocorneal adhesions cause angle compromise. "
            "Conservative: lubricants and observation if visually asymptomatic. "
            "Endothelial keratoplasty (DMEK/DSAEK) for visually significant corneal oedema. "
            "PKP rarely required unless stroma involved. "
            "Genetic counselling: alert family members that standard exome/panel may miss PPCD1 — "
            "regulatory region-specific testing must be explicitly requested. "
            "No disease-modifying pharmacotherapy available."
        ),
        "key_features": [
            "Regulatory region variants — coding sequence NGS alone will MISS this diagnosis",
            "Clinically identical to PPCD3 — genetic testing essential for subtype",
            "PPCD1 = chromosome 20p13 locus (same chromosome as SLC4A11 at 20p13)",
            "Iridocorneal adhesions may cause secondary glaucoma",
            "Often milder than PPCD3 but significant clinical overlap",
        ],
        "key_ddx": [
            "PPCD3 (ZEB1): clinically identical — ZEB1 coding variants detected by standard sequencing",
            "ICE syndrome: unilateral, acquired, progressive — PPCD bilateral, hereditary",
            "Fuchs endothelial: guttae, no iridocorneal adhesions, no vesicles",
            "CHED2 (SLC4A11): congenital onset, diffuse ground-glass — PPCD adult onset",
            "Diagnosis confirmation: requires regulatory region / promoter sequencing or MLPA for 20p13",
        ],
        "corneal_layer": "Endothelium + Descemet membrane",
        "ic3d_category": "Category 1 (genetic basis well-established)",
        "recurrence_in_graft": True,
        "ptk_effective": False,
        "erosion_frequency": "rare",
        "onset_age": "Variable; often subclinical until 3rd–6th decade",
        "autosomal_recessive_risk": False,
    },
    # -- TCF4 -- Fuchs FECD3 ---------------------------------------------------------------
    {
        "gene": "TCF4",
        "alt_name": (
            "TCF4 (TCF4-667aa-18q21.2 / AD -- "
            "FUCHS-ENDOTHELIAL-CORNEAL-DYSTROPHY-TYPE-3-MOST-COMMON -- "
            "CTG18.1-REPEAT-EXPANSION->40-REPEATS-79pct-OF-FECD-PATIENTS -- "
            "DESCEMET-MEMBRANE-GUTTAE-PATHOGNOMONIC -- "
            "LEADING-INDICATION-FOR-CORNEAL-TRANSPLANTATION-WORLDWIDE)"
        ),
        "protein": (
            "TCF4 -- 18q21.2 AD -- TCF4-667aa -- "
            "Transcription-Factor-4-E2-2-ITF2-73kDa-Class-I-bHLH-E-Protein -- "
            "Binds-E-Box-CANNTG-Heterodimerises-with-Class-II-bHLH-Factors -- "
            "Expressed-Corneal-Endothelium-Neural-Tissue-Lymphocytes -- "
            "FECD3-Due-to-CTG18.1-Trinucleotide-Repeat-Expansion-in-Intron-3 -- "
            "Normal:<40-Repeats-Pathogenic:>40-Repeats-Mean-Affected:~100-350-Repeats -- "
            "RNA-Gain-of-Function-CUG-Repeat-RNA-Forms-Nuclear-Foci-Sequesters-MBNL-Proteins -- "
            "Splicing-Dysregulation-Endothelial-Cell-Death -- "
            "OMIM-Gene-189894-Disease-FECD3-613267"
        ),
        "locus": "18q21.2",
        "protein_size": "667 aa / 73 kDa",
        "inheritance": (
            "AD (autosomal dominant); trinucleotide repeat expansion CTG18.1 in intron 3 of TCF4; "
            ">40 repeats pathogenic (79% of Fuchs patients of European ancestry); "
            "mean allele size in FECD: ~100-350 repeats; "
            "sex difference: female:male = 3:1 in Fuchs (oestrogen modifies endothelial cell biology); "
            "repeat size correlates imperfectly with severity; "
            "missense variants in TCF4 coding sequence also reported but less common"
        ),
        "disease_category": "FECD3 — Fuchs Endothelial Corneal Dystrophy type 3 (most common hereditary corneal dystrophy in adults; leading indication for corneal transplantation worldwide)",
        "disease_pathway": (
            "The CTG18.1 intronic expansion generates CUG-repeat RNA that forms nuclear RNA foci in endothelial cells. "
            "Nuclear foci sequester Muscleblind-like proteins (MBNL1, MBNL2) → alternative splicing dysregulation → "
            "transcriptome-wide splicing errors in endothelial cells → mitochondrial dysfunction, oxidative stress, unfolded protein response. "
            "Endothelial cell apoptosis and loss → compensatory Descemet membrane thickening and guttata formation. "
            "Guttae: collagen excrescences on posterior Descemet membrane deposited by dysfunctional endothelium. "
            "As endothelial density falls below ~500 cells/mm² (normal 2500): corneal oedema → stroma → epithelium (bullous keratopathy). "
            "Oestrogen promotes TEAD/YAP endothelial proliferation partially explaining female predominance. "
            "Central to pericentral guttae distribution early → spreads peripherally in late disease. "
            "RNA-gain-of-function mechanism (not protein loss) — similar to myotonic dystrophy DM1 (DMPK CTG repeats)."
        ),
        "pathognomonic": (
            "DESCEMET MEMBRANE GUTTAE PATHOGNOMONIC — seen on slit-lamp retroillumination as dark 'beaten metal' pattern. "
            "Grading: Krachmer grading scale (Grade 0-5) or modified scales. "
            "Early (Grade 1-2): central guttae ≤1 mm zone; no symptoms; specular microscopy shows dark 'drop-like' lesions. "
            "Moderate (Grade 3-4): guttae ≥1 mm, corneal oedema developing; morning blurring (diurnal variation — worse on waking); "
            "halos/glare. "
            "Advanced (Grade 5): diffuse endothelial failure; stromal oedema; bullous keratopathy; pain from ruptured bullae. "
            "SLIT-LAMP RETROILLUMINATION: central beaten-metal guttae pattern unique to Fuchs. "
            "SPECULAR MICROSCOPY: endothelial cell loss, dark non-reflective areas (guttae), polymegethism/pleomorphism. "
            "MORNING WORSENING PATHOGNOMONIC: overnight closed eye → less evaporation → corneal oedema worse on waking → "
            "improves through day as tears evaporate excess fluid. "
            "Anterior OCT: Descemet membrane irregularity and thickening."
        ),
        "treatment": (
            "Conservative (early-moderate): 5% hypertonic NaCl drops/ointment (osmotic dehydration, especially morning); "
            "hair dryer held at arm's length to promote epithelial evaporation; UV-blocking glasses for photosensitivity. "
            "Surgical (endothelial decompensation): "
            "DESCEMET MEMBRANE ENDOTHELIAL KERATOPLASTY (DMEK) — gold standard; "
            "fastest recovery, best visual outcomes (20/20 possible); "
            "DSAEK (thicker graft, lower rebubbling rate — preferred if posterior segment issues). "
            "PKP (penetrating keratoplasty): reserved for failed endothelial keratoplasty or significant stromal disease. "
            "TIMING: DMEK BEFORE stromal/epithelial oedema stage for best outcomes — early referral to cornea specialist. "
            "Future: Rho-kinase inhibitors (netarsudil — Y-27632) promote endothelial proliferation; "
            "SB-623 stem cell therapy; CRISPR-targeted CTG18.1 repeat knockout under investigation. "
            "Avoid contact lens fitting over guttae (hypoxia accelerates). "
            "AVOID PROLONGED INTRAOCULAR SURGERY (cataract, vitreoretinal) — endothelial trauma precipitates decompensation."
        ),
        "key_features": [
            "Most common hereditary corneal dystrophy in adults; leading cause of corneal transplantation worldwide",
            "CTG18.1 repeat expansion (>40 repeats) — found in 79% of Fuchs patients",
            "Female:male = 3:1 (oestrogen effect on endothelial biology)",
            "Morning worsening of vision (diurnal variation) pathognomonic early symptom",
            "DMEK gold-standard treatment — best visual outcomes of any keratoplasty",
            "RNA gain-of-function mechanism (not protein level): similar to myotonic dystrophy",
        ],
        "key_ddx": [
            "Early guttae vs Hassall-Henle bodies: peripheral endothelial excrescences are age-related/normal — Fuchs central",
            "PPCD: vesicular lesions, iridocorneal adhesions — not guttae pattern",
            "Pseudophakic bullous keratopathy: post-cataract, no guttae, unilateral",
            "ICE syndrome: unilateral, no guttae, iridocorneal adhesions, iris changes",
            "Secondary endothelial failure: trauma, uveitis, post-surgical — history distinguishes",
        ],
        "corneal_layer": "Endothelium + Descemet membrane",
        "ic3d_category": "Category 1 (genetic basis well-established)",
        "recurrence_in_graft": False,
        "ptk_effective": False,
        "erosion_frequency": "late (bullous keratopathy)",
        "onset_age": "4th–6th decade (later onset than COL8A2 early-onset Fuchs)",
        "autosomal_recessive_risk": False,
    },
    # -- COL8A2 -- Fuchs FECD1 Early-onset ------------------------------------------------
    {
        "gene": "COL8A2",
        "alt_name": (
            "COL8A2 (COL8A2-703aa-1p34.3 / AD -- "
            "FUCHS-ENDOTHELIAL-CORNEAL-DYSTROPHY-TYPE-1-EARLY-ONSET -- "
            "ONSET-BEFORE-40-YEARS-PATHOGNOMONIC -- "
            "L450W-Q455K-MOST-COMMON-EARLY-ONSET-FUCHS-VARIANTS -- "
            "PPCD-ASSOCIATION)"
        ),
        "protein": (
            "COL8A2 -- 1p34.3 AD -- COL8A2-703aa -- "
            "Collagen-Type-VIII-Alpha-2-Chain-78kDa-Short-Chain-Non-Fibrillar-Collagen -- "
            "NC1-Domain-Trimerisation-Collagenous-COL-Domain-Triple-Helix -- "
            "Component-of-Descemet-Membrane-Hexagonal-Lattice-Basement-Membrane -- "
            "Heterotrimers-COL8A2-COL8A1-COL8A1 or COL8A2-COL8A2-COL8A2 -- "
            "LOF-Disrupts-Descemet-Lattice-Endothelial-Stability-Reduced -- "
            "L450W-Trp-NC1-Domain-Hydrophobic-Core-Disruption-Most-Severe -- "
            "Q455K-Less-Severe-Less-Common -- "
            "OMIM-Gene-120252-Disease-FECD1-136800"
        ),
        "locus": "1p34.3",
        "protein_size": "703 aa / 78 kDa",
        "inheritance": (
            "AD (autosomal dominant); missense variants in NC1 domain; "
            "L450W most common early-onset Fuchs variant (worldwide); Q455K less common; "
            "incomplete penetrance observed in some families; "
            "FECD1 = early-onset Fuchs (<40 years) distinguishing it from TCF4 repeat-associated FECD3 (>40 years); "
            "PPCD also reported with COL8A2 — same gene, different variant class"
        ),
        "disease_category": "FECD1 — Fuchs Endothelial Corneal Dystrophy type 1 (early-onset, onset <40 years); posterior polymorphous corneal dystrophy (PPCD)",
        "disease_pathway": (
            "COL8A2 is the predominant alpha-2 chain of collagen type VIII (short-chain collagen), "
            "a major structural component of Descemet membrane (posterior corneal basement membrane). "
            "COL8A2 forms a regular hexagonal lattice with COL8A1 providing Descemet membrane structural integrity. "
            "Early-onset Fuchs variants (L450W, Q455K) in the NC1 domain: "
            "missense disrupts collagen VIII trimerisation and secretion → "
            "endoplasmic reticulum stress → unfolded protein response (UPR) → endothelial apoptosis. "
            "Additionally, abnormal Descemet lattice → guttae formation similar to TCF4-Fuchs. "
            "PPCD COL8A2 variants: different mechanism, endothelial metaplasia pathway. "
            "Earlier endothelial failure than TCF4-Fuchs: "
            "patients often require transplantation in 30s-40s vs 60s-70s for TCF4."
        ),
        "pathognomonic": (
            "EARLY-ONSET FUCHS GUTTAE — presentation in 20s-30s PATHOGNOMONIC for COL8A2 FECD1 (vs TCF4 FECD3 onset 40s-60s). "
            "Slit-lamp: central guttae identical to late-onset Fuchs; retroillumination beaten-metal appearance. "
            "FAMILY HISTORY OF EARLY CORNEAL DISEASE (<40 years) is the key distinguishing feature from TCF4-FECD. "
            "Specular microscopy: endothelial cell loss and dark guttae-corresponding areas in young patients. "
            "Vision symptoms (diurnal variation, glare) occurring in 3rd-4th decade — "
            "any patient with Fuchs features before age 40 requires COL8A2 genetic testing. "
            "Descemet membrane thickening on anterior OCT in young patients. "
            "SURGICAL INTERVENTION EARLIER: most require DMEK/DSAEK in 4th decade (vs 7th for TCF4-Fuchs)."
        ),
        "treatment": (
            "Same as TCF4-Fuchs endothelial dystrophy but EARLIER TIMELINE: "
            "DMEK preferred for primary endothelial keratoplasty — typically required in 30s-40s. "
            "Conservative measures: hypertonic NaCl 5% drops/ointment morning; hair dryer for epithelial drying. "
            "DSAEK if DMEK not available or technically difficult (small pupil, posterior segment complications). "
            "Cataract surgery planning: if cataract co-exists, combined phacoemulsification + DSAEK/DMEK "
            "in a single operation (triple procedure) reduces surgical risk vs staged. "
            "Post-DMEK: positioning face-down for 1 hour mandatory to float graft against stroma; "
            "rebubbling if graft detachment occurs (30-day window critical). "
            "Long-term prognosis post-DMEK excellent — Fuchs does NOT recur in donor graft "
            "(donor endothelium lacks COL8A2 pathogenic variant if from normal donor)."
        ),
        "key_features": [
            "EARLY-ONSET Fuchs — guttae appearing in 20s-30s is the critical feature",
            "L450W and Q455K are the most common pathogenic variants",
            "Earlier DMEK/DSAEK indication than TCF4-Fuchs (typically 4th decade)",
            "Also associated with PPCD — different variant class, different mechanism",
            "Corneal transplant prognosis excellent — Fuchs does NOT recur in donor graft",
            "ER stress / UPR pathway — L450W disrupts NC1 domain trimerisation",
        ],
        "key_ddx": [
            "TCF4-FECD3: late onset (>40 years), CTG18.1 repeat expansion — genetic testing separates",
            "PPCD (ZEB1/OVOL2): vesicular lesions, iridocorneal adhesions — not central guttae pattern",
            "Secondary endothelial failure: trauma, uveitis, post-surgical — no guttae, history distinguishes",
            "Iridocorneal endothelial (ICE) syndrome: unilateral, acquired, iris changes — FECD bilateral, hereditary",
            "Hassall-Henle bodies: peripheral, age-related — FECD central, pathological",
        ],
        "corneal_layer": "Endothelium + Descemet membrane",
        "ic3d_category": "Category 1 (genetic basis well-established)",
        "recurrence_in_graft": False,
        "ptk_effective": False,
        "erosion_frequency": "late (bullous keratopathy)",
        "onset_age": "2nd–4th decade (early-onset, unlike TCF4-FECD3)",
        "autosomal_recessive_risk": False,
    },
    # -- VSX1 -- Keratoconus + PPCD2 -------------------------------------------------------
    {
        "gene": "VSX1",
        "alt_name": (
            "VSX1 (VSX1-365aa-20p11.21 / AD -- "
            "KERATOCONUS-TYPE-1-KTCN1 -- "
            "CORNEAL-THINNING+IRREGULAR-ASTIGMATISM+FLEISCHER-RING-PATHOGNOMONIC -- "
            "PPCD2-POSTERIOR-POLYMORPHOUS-ALSO-REPORTED -- "
            "CROSS-LINKING-CXL-HALTS-PROGRESSION)"
        ),
        "protein": (
            "VSX1 -- 20p11.21 AD -- VSX1-365aa -- "
            "Visual-System-Homeobox-1-41kDa-Paired-Like-Homeodomain-Transcription-Factor -- "
            "Homeodomain-CVC-Subdomain-CRALBP-Promoter-Binding -- "
            "Expressed-Bipolar-Photoreceptors-Retina-Corneal-Stroma-Keratocytes -- "
            "LOF-Disrupts-Corneal-Keratocyte-Identity-Stromal-Collagen-Assembly -- "
            "p.Pro247Arg-Most-Studied-Lebanese-Kindred-Variant -- "
            "p.His244Arg-Canady-p.Asp144Glu-also-Reported -- "
            "OMIM-Gene-605020-Disease-KTCN1-148300"
        ),
        "locus": "20p11.21",
        "protein_size": "365 aa / 41 kDa",
        "inheritance": (
            "AD (autosomal dominant); variable penetrance; incomplete in some families; "
            "keratoconus is genetically heterogeneous — VSX1 accounts for 5-10% of familial keratoconus; "
            "most keratoconus is complex multifactorial (ZNF469, DOCK9, IPO5, VSX1, TGFBI modifier, many loci); "
            "VSX1 also linked to PPCD2 (posterior polymorphous corneal dystrophy type 2); "
            "sporadic keratoconus more common than familial — VSX1 more relevant in family-clustered cases"
        ),
        "disease_category": "KTCN1 — Keratoconus type 1 (hereditary familial keratoconus); PPCD2 — Posterior Polymorphous Corneal Dystrophy type 2",
        "disease_pathway": (
            "VSX1 is a paired-like homeodomain transcription factor expressed in corneal keratocytes and retinal bipolar cells. "
            "In the cornea, VSX1 regulates expression of genes critical for keratocyte identity and stromal collagen organisation. "
            "LOF → keratocyte dysfunction → abnormal collagen fibril assembly in stroma → corneal ectasia. "
            "Keratoconus hallmarks: thinning of paracentral/central cornea, irregular astigmatism, "
            "anterior protrusion (ectasia), altered collagen fibril geometry on electron microscopy. "
            "Secondary changes: iron deposition at epithelial base (Fleischer ring — haemosiderin from tear flow anomaly), "
            "Vogt striae (stress lines in posterior stroma), apical corneal scarring (late). "
            "Corneal hydrops: acute Descemet membrane tear → aqueous floods stroma → acute pain, blurring, "
            "white cornea — emergency requiring conservative management. "
            "PPCD2 mechanism (VSX1): separate from keratoconus — endothelial metaplasia pathway."
        ),
        "pathognomonic": (
            "KERATOCONUS CLASSIC TRIAD PATHOGNOMONIC: "
            "1. CORNEAL THINNING: paracentral inferior thinning (inferior > superior) on pachymetry maps; "
            "thinnest point <500 µm typically; progressive. "
            "2. IRREGULAR ASTIGMATISM: scissor reflex on retinoscopy; corrected acuity limited by irregular surface; "
            "Placido topography shows inferior steepening 'skewed radial axes' (KISA% index). "
            "3. FLEISCHER RING: iron deposition (haemosiderin) at base of epithelium in ring pattern at cone base — "
            "PATHOGNOMONIC for keratoconus; visible on cobalt blue slit-lamp (fluorescein enhances). "
            "VOGT STRIAE: vertical stress lines in deep stroma at cone apex; disappear with digital pressure. "
            "MUNSON SIGN: V-shaped deformation of lower lid on down-gaze (advanced keratoconus). "
            "ACUTE HYDROPS: sudden-onset whitening, pain — Descemet rupture emergency. "
            "Keratometry: K-readings >47D (steep), asymmetric between eyes, rapid progression on serial maps."
        ),
        "treatment": (
            "CORNEAL CROSS-LINKING (CXL) — STANDARD OF CARE FOR PROGRESSIVE KERATOCONUS: "
            "riboflavin + UV-A irradiation stiffens stromal collagen by inducing covalent cross-links; "
            "HALTS PROGRESSION in >90% (epithelium-off CXL); "
            "eligibility: age 14-35, progression documented (≥1D change in 12 months), "
            "thinnest point >400 µm (conventional) or >300 µm (accelerated). "
            "REFRACTIVE CORRECTION: "
            "Spectacles (early); rigid gas-permeable (RGP) contact lenses (irregular surface correction); "
            "scleral lenses (advanced — vault over ectatic apex, excellent vision). "
            "CORNEAL TRANSPLANTATION (advanced/scarred): "
            "DALK (deep anterior lamellar keratoplasty) preferred — preserves recipient Descemet/endothelium; "
            "PKP if DALK fails or Descemet torn; keratoconus DOES NOT RECUR IN DALK/PKP graft. "
            "Intracorneal rings (ICRS/Intacs): moderate keratoconus, steepen inferiorly → flatten cone; "
            "adjunct to CXL in some centres. "
            "ACUTE HYDROPS: cycloplegics, hypertonic saline, conservative 2-3 months; "
            "corneal scarring after resolution may paradoxically improve topo by scarring the ectactic apex. "
            "Eye rubbing STRICTLY CONTRAINDICATED — triggers and accelerates keratoconus; allergy management mandatory."
        ),
        "key_features": [
            "Fleischer ring (iron deposition at cone base) pathognomonic for keratoconus",
            "VSX1 accounts for 5-10% of familial keratoconus (genetically heterogeneous disease)",
            "Corneal cross-linking (CXL) halts progression in >90% — early intervention optimal",
            "DALK preferred over PKP — preserves recipient endothelium, keratoconus does not recur in graft",
            "Eye rubbing strictly contraindicated — most modifiable progression risk factor",
            "Acute hydrops emergency — Descemet tear, requires conservative management",
        ],
        "key_ddx": [
            "Pellucid marginal degeneration (PMD): inferior peripheral thinning 'crab claw' topography; Fleischer ring absent",
            "Keratoglobus: diffuse global thinning vs cone-shaped focal; different topography",
            "Post-LASIK ectasia: LASIK history, no Fleischer ring, iatrogenic",
            "Fuchs (early): central guttae not thinning; topography different",
            "VSX1 vs multifactorial sporadic keratoconus: family history and genetic testing",
        ],
        "corneal_layer": "Stroma (keratocyte dysfunction)",
        "ic3d_category": "Category 1 for PPCD2; Category 2 for KTCN1 (genes identified but not yet fully validated as monogenic)",
        "recurrence_in_graft": False,
        "ptk_effective": False,
        "erosion_frequency": "moderate (acute hydrops)",
        "onset_age": "Puberty to 3rd decade (keratoconus peak 15-30 years)",
        "autosomal_recessive_risk": False,
    },
    # -- KRT12 -- Meesmann Epithelial Corneal Dystrophy ------------------------------------
    {
        "gene": "KRT12",
        "alt_name": (
            "KRT12 (KRT12-505aa-17q21.2 / AD -- "
            "MEESMANN-EPITHELIAL-CORNEAL-DYSTROPHY-MECD -- "
            "MOST-COMMON-EPITHELIAL-CORNEAL-DYSTROPHY -- "
            "INTRAEPITHELIAL-MICROCYSTS-THROUGHOUT-CORNEA-PATHOGNOMONIC -- "
            "PAS-POSITIVE-GLYCOGEN-IN-CYSTS)"
        ),
        "protein": (
            "KRT12 -- 17q21.2 AD -- KRT12-505aa -- "
            "Keratin-12-55kDa-Type-I-Acidic-Keratin-Corneal-Epithelium-Specific -- "
            "Intermediate-Filament-Protein-KRT12-KRT3-Heterodimer-Obligate -- "
            "1A-Helix-rod-domain-most-common-pathogenic-variant-location -- "
            "p.Arg135Thr-Most-Common-European-KRT12-Variant -- "
            "Dominant-Negative-Mutant-KRT12-Disrupts-Tonofilament-Network -- "
            "Epithelial-Fragility-Cytolysis-Microcyst-Formation -- "
            "OMIM-Gene-601687-Disease-MECD-122100"
        ),
        "locus": "17q21.2",
        "protein_size": "505 aa / 55 kDa",
        "inheritance": (
            "AD (autosomal dominant); dominant-negative mechanism — mutant KRT12 poisons KRT3/KRT12 heterodimer; "
            "complete penetrance, variable expressivity; "
            "mutations in helix initiation motif (1A domain) most pathogenic; "
            "also KRT3 mutations (12q13) cause Meesmann — genetic testing covers both KRT12 and KRT3; "
            "no clear genotype-phenotype correlation for severity within MECD"
        ),
        "disease_category": "MECD — Meesmann Epithelial Corneal Dystrophy (most common hereditary epithelial corneal dystrophy)",
        "disease_pathway": (
            "KRT12 is the corneal epithelium-specific type I keratin that obligately pairs with KRT3 (type II) "
            "to form the intermediate filament network of corneal epithelial cells (superficial and wing cells). "
            "Pathogenic KRT12 variants → dominant-negative disruption of tonofilament assembly → "
            "cytoskeletal fragility → epithelial cells lyse → contents form intraepithelial microcysts. "
            "Cysts contain cellular debris, glycogen (PAS positive) and lipid remnants. "
            "The microcysts are scattered throughout the entire corneal epithelium from limbus to limbus "
            "but are concentrated in the interpalpebral zone. "
            "Microcysts slowly migrate toward surface, rupture → transient recurrent erosions (usually mild). "
            "Unlike EBMD (epithelial basement membrane dystrophy/Map-Dot-Fingerprint), "
            "MECD has no abnormal basement membrane reduplication. "
            "Vision usually minimally affected until later decades if cysts become confluent centrally."
        ),
        "pathognomonic": (
            "INTRAEPITHELIAL MICROCYSTS THROUGHOUT THE ENTIRE CORNEA — BILATERAL DIFFUSE DISTRIBUTION PATHOGNOMONIC. "
            "Slit-lamp: hundreds to thousands of tiny (<0.1 mm) clear grey-white dots in corneal epithelium. "
            "RETROILLUMINATION best demonstrates cysts — appear as multiple small clear bubbles. "
            "Distribution: ENTIRE corneal epithelium from limbus to limbus (vs EBMD — central/paracentral only). "
            "Histopathology: intraepithelial cysts containing PAS+ glycogen material. "
            "Electron microscopy: disrupted tonofilaments in basal and wing cells. "
            "ONSET: neonatal or early childhood — microcysts present from birth, detected in infancy with slit-lamp. "
            "Symptoms: usually MINIMAL — mild photophobia, occasional foreign body sensation; "
            "visual acuity usually PRESERVED until late disease. "
            "KEY DDx: EBMD (map-dot-fingerprint) — paracentral, no glycogen cysts, basement membrane changes, "
            "older onset, often unilateral."
        ),
        "treatment": (
            "Largely supportive — most MECD patients have good visual prognosis for decades: "
            "Lubricant drops: for dry eye symptoms and mild erosion prevention. "
            "Bandage contact lens: for recurrent erosion episodes (usually mild). "
            "Cyclosporine 0.05% drops: may reduce epithelial inflammation. "
            "PTK (phototherapeutic keratectomy): anterior stroma ablation — not standard for MECD "
            "as cysts involve full epithelial thickness; limited benefit for cyst removal. "
            "Corneal transplantation: rarely required — only if central confluent cysts cause significant visual loss; "
            "PKP or superficial anterior lamellar keratoplasty (SALK); "
            "RECURRENCE IN GRAFT: MECD recurs (KRT12 is in recipient's cells migrating to graft — "
            "same mechanism as TGFBI recurrence). "
            "Genetic counselling: AD with full penetrance — 50% offspring risk. "
            "REASSURANCE: most patients do NOT require surgery and maintain good vision through adult life."
        ),
        "key_features": [
            "Most common hereditary epithelial corneal dystrophy",
            "Microcysts throughout entire cornea — from limbus to limbus (unlike EBMD which is central)",
            "PAS-positive glycogen in cysts on histopathology — diagnostic on biopsy",
            "Usually MILD — good visual prognosis for most of adult life",
            "Retroillumination slit-lamp technique best for visualising microcysts",
            "KRT3 mutations (12q13) cause the same phenotype — both genes must be tested",
        ],
        "key_ddx": [
            "EBMD (map-dot-fingerprint/Cogan): central/paracentral only, basement membrane reduplications, older adults",
            "Thygeson superficial punctate keratitis: acquired inflammatory, response to steroids — MECD hereditary, non-inflammatory",
            "Recurrent corneal erosion syndrome (RCES): no microcysts, irregular epithelium — KRT12 + microcysts",
            "Macular dystrophy (CHST6 AR): deep stroma, not epithelium — MECD superficial layer only",
            "KRT12 vs KRT3 MECD: clinically identical — genetic testing distinguishes (KRT12 17q21.2 vs KRT3 12q13)",
        ],
        "corneal_layer": "Epithelium",
        "ic3d_category": "Category 1 (genetic basis well-established)",
        "recurrence_in_graft": True,
        "ptk_effective": False,
        "erosion_frequency": "mild (occasional)",
        "onset_age": "Neonatal / early childhood (cysts present from birth)",
        "autosomal_recessive_risk": False,
    },
]

PATIENTS_PER_GENE = 40


def _make_cohort(entry, seed):
    """Generate a 40-patient synthetic cohort for one gene."""
    rng = random.Random(seed)
    gene = entry["gene"]
    patients = []
    layer = entry["corneal_layer"]

    for i in range(PATIENTS_PER_GENE):
        # Age at diagnosis varies by gene
        if gene == "SLC4A11":
            age_dx = round(rng.gauss(1.0, 1.2), 1)  # congenital / neonatal
            age_dx = max(0.0, min(age_dx, 5.0))
        elif gene == "KRT12":
            age_dx = round(rng.gauss(6.0, 4.0), 1)  # childhood
            age_dx = max(0.5, min(age_dx, 20.0))
        elif gene == "TCF4":
            age_dx = round(rng.gauss(58.0, 10.0), 1)  # late onset
            age_dx = max(35.0, min(age_dx, 80.0))
        elif gene == "COL8A2":
            age_dx = round(rng.gauss(32.0, 7.0), 1)  # early-onset Fuchs
            age_dx = max(18.0, min(age_dx, 50.0))
        elif gene == "VSX1":
            age_dx = round(rng.gauss(20.0, 5.0), 1)  # keratoconus
            age_dx = max(12.0, min(age_dx, 40.0))
        else:
            age_dx = round(rng.gauss(35.0, 12.0), 1)
            age_dx = max(5.0, min(age_dx, 70.0))

        # Corneal oedema risk
        if gene in ("TCF4", "COL8A2", "SLC4A11"):
            oedema = rng.random() < 0.72
        elif gene in ("ZEB1", "OVOL2"):
            oedema = rng.random() < 0.35
        else:
            oedema = rng.random() < 0.15

        # PTK performed
        if gene == "TGFBI":
            ptk = rng.random() < 0.65
        else:
            ptk = False

        # Corneal transplant
        if gene in ("TCF4", "COL8A2", "SLC4A11"):
            transplant = rng.random() < 0.55
        elif gene == "TGFBI":
            transplant = rng.random() < 0.40
        elif gene == "VSX1":
            transplant = rng.random() < 0.35
        else:
            transplant = rng.random() < 0.18

        # Glaucoma (PPCD genes)
        glaucoma = gene in ("ZEB1", "OVOL2") and rng.random() < 0.28

        # Recurrent erosions
        if gene == "TGFBI":
            erosions = rng.random() < 0.72
        elif gene == "KRT12":
            erosions = rng.random() < 0.30
        else:
            erosions = rng.random() < 0.08

        # CXL for VSX1 keratoconus
        cxl = gene == "VSX1" and rng.random() < 0.70

        # Current VA (logMAR proxy)
        if oedema or (gene == "SLC4A11" and not transplant):
            bcva_worse_than_6_12 = rng.random() < 0.60
        elif gene == "TGFBI" and not ptk and not transplant:
            bcva_worse_than_6_12 = rng.random() < 0.35
        else:
            bcva_worse_than_6_12 = rng.random() < 0.15

        # Bilateral
        bilateral = True  # all hereditary corneal dystrophies in this atlas are bilateral

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis_years": age_dx,
            "corneal_layer": layer,
            "corneal_oedema": oedema,
            "ptk_performed": ptk,
            "corneal_transplant": transplant,
            "transplant_type": (
                "DMEK" if (gene in ("TCF4", "COL8A2") and transplant) else
                "DSAEK" if (gene == "SLC4A11" and transplant) else
                "DALK" if (gene == "VSX1" and transplant) else
                "PKP" if (gene == "TGFBI" and transplant) else
                "DMEK" if transplant else "none"
            ),
            "glaucoma": glaucoma,
            "recurrent_erosions": erosions,
            "cross_linking_cxl": cxl,
            "bcva_worse_than_6_12": bcva_worse_than_6_12,
            "bilateral": bilateral,
        })
    return patients


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(CORNEAL_DYSTROPHY_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))

    total = len(all_patients)
    transplant_count = sum(1 for p in all_patients if p["corneal_transplant"])
    oedema_count = sum(1 for p in all_patients if p["corneal_oedema"])
    erosion_count = sum(1 for p in all_patients if p["recurrent_erosions"])
    glaucoma_count = sum(1 for p in all_patients if p["glaucoma"])
    ptk_count = sum(1 for p in all_patients if p["ptk_performed"])
    cxl_count = sum(1 for p in all_patients if p["cross_linking_cxl"])

    gene_summary = {}
    for idx, entry in enumerate(CORNEAL_DYSTROPHY_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "corneal_layer": entry["corneal_layer"],
            "n_patients": len(cohort),
            "transplant_pct": round(100 * sum(1 for p in cohort if p["corneal_transplant"]) / len(cohort), 1),
            "oedema_pct": round(100 * sum(1 for p in cohort if p["corneal_oedema"]) / len(cohort), 1),
            "erosion_pct": round(100 * sum(1 for p in cohort if p["recurrent_erosions"]) / len(cohort), 1),
            "glaucoma_pct": round(100 * sum(1 for p in cohort if p["glaucoma"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Corneal-Dystrophy-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Corneal Dystrophy Reference -- TGFBI/SLC4A11/ZEB1/OVOL2/TCF4/COL8A2/VSX1/KRT12",
        "genes_covered": [e["gene"] for e in CORNEAL_DYSTROPHY_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "transplant_pct": round(100 * transplant_count / total, 1),
            "corneal_oedema_pct": round(100 * oedema_count / total, 1),
            "recurrent_erosions_pct": round(100 * erosion_count / total, 1),
            "glaucoma_pct": round(100 * glaucoma_count / total, 1),
            "ptk_pct": round(100 * ptk_count / total, 1),
            "cxl_pct": round(100 * cxl_count / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(CORNEAL_DYSTROPHY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "corneal_layer": entry["corneal_layer"],
            "ic3d_category": entry["ic3d_category"],
            "recurrence_in_graft": entry["recurrence_in_graft"],
            "ptk_effective": entry["ptk_effective"],
            "erosion_frequency": entry["erosion_frequency"],
            "onset_age": entry["onset_age"],
            "n_patients": len(cohort),
            "transplant_pct": round(100 * sum(1 for p in cohort if p["corneal_transplant"]) / len(cohort), 1),
            "oedema_pct": round(100 * sum(1 for p in cohort if p["corneal_oedema"]) / len(cohort), 1),
            "erosion_pct": round(100 * sum(1 for p in cohort if p["recurrent_erosions"]) / len(cohort), 1),
            "glaucoma_pct": round(100 * sum(1 for p in cohort if p["glaucoma"]) / len(cohort), 1),
            "ptk_pct": round(100 * sum(1 for p in cohort if p["ptk_performed"]) / len(cohort), 1),
            "cxl_pct": round(100 * sum(1 for p in cohort if p["cross_linking_cxl"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:400],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "corneal_layer": entry["corneal_layer"],
                "ic3d_category": entry["ic3d_category"],
                "recurrence_in_graft": entry["recurrence_in_graft"],
                "ptk_effective": entry["ptk_effective"],
                "erosion_frequency": entry["erosion_frequency"],
                "onset_age": entry["onset_age"],
            }
            for entry in CORNEAL_DYSTROPHY_GENES
        },
        "corneal_dystrophy_glossary": {
            "IC3D Classification — International Committee for Classification of Corneal Dystrophies": (
                "IC3D (2008, updated 2015) provides the evidence-based classification of corneal dystrophies. "
                "Category 1: well-defined genetic basis, multiple families, gene characterised (TGFBI, SLC4A11, ZEB1, TCF4, COL8A2, KRT12 — all in this atlas). "
                "Category 2: gene mapped but phenotype-genotype confirmation in multiple families incomplete (VSX1/KTCN1 borderline). "
                "Category 3: gene not yet identified, locus mapped. "
                "Category 4: no genetic data — descriptive only. "
                "IC3D replaced older classifications (Groenouw, Bücklers, Hogan) — always use IC3D nomenclature in publications."
            ),
            "TGFBI — Variant-Specific Phenotypes and DDx": (
                "TGFBI (keratoepithelin) variants at R124 and R555 cause 5 distinct corneal dystrophies: "
                "R124C → LCD-I (Lattice CD type I): amyloid fibril deposits, branching lattice lines, Congo Red birefringent. "
                "R124H → Avellino/GCD2: combined lattice+granular; both Congo Red+ and Masson Trichrome RED. "
                "R124L → Reis-Bücklers: Bowman layer destruction, childhood recurrent erosions, most severe. "
                "R555W → GCD1: discrete bread-crumb hyaline deposits, intervening cornea CLEAR (DDx key). "
                "R555Q → Thiel-Behnke: honeycomb Bowman opacities, milder than Reis-Bücklers. "
                "ALL recur in corneal grafts — PTK for anterior lesions, DALK/PKP for deep; CXL not indicated."
            ),
            "Fuchs Endothelial Corneal Dystrophy — Types and Transplant Timing": (
                "Fuchs FECD is the most common hereditary corneal dystrophy in adults. "
                "Types: FECD1 (COL8A2 L450W/Q455K): early-onset <40 y; FECD3 (TCF4 CTG18.1 repeat): late-onset 40-70 y. "
                "Key genetic test: TCF4 repeat expansion by triplet-primed PCR (standard exome MISSES trinucleotide repeats). "
                "Grading: Krachmer grades 1-5; grade 1-2 = guttae only; grade 3-4 = guttae + oedema; grade 5 = bullae. "
                "DMEK is gold standard surgery — faster vision recovery than DSAEK (1-2 months vs 3-6); "
                "DMEK rebubbling 20-40% but easily treated; DMEK vs DSAEK: DSAEK preferred with posterior segment disease, "
                "vitrectomised eye, poor compliance. "
                "Fuchs does NOT recur in donor graft (donor endothelium is from normal individual). "
                "Morning worsening (diurnal variation) pathognomonic early symptom — resolve with evaporating tears by afternoon."
            ),
            "PPCD — Posterior Polymorphous Corneal Dystrophy Subtypes and ICE DDx": (
                "PPCD has three confirmed genetic subtypes: PPCD1 (OVOL2 regulatory), PPCD2 (COL8A2), PPCD3 (ZEB1). "
                "All share endothelial-to-epithelial metaplasia mechanism. Clinically similar — genetic testing required for subtype. "
                "CRITICAL ICE SYNDROME DDx: "
                "ICE (iridocorneal endothelial) syndrome — ACQUIRED, progressive, UNILATERAL, no family history; "
                "three clinical subtypes (Chandler's, essential iris atrophy, Cogan-Reese/iris naevus); "
                "secondary glaucoma common; treatment = glaucoma + keratoplasty for oedema. "
                "PPCD — HEREDITARY, often BILATERAL (may be asymmetric), family history, static or slowly progressive. "
                "Key slit-lamp DDx: PPCD has VESICULAR lesions (ICE does not); PPCD bilateral. "
                "PPCD1 trap: coding-region exome will MISS PPCD1 (OVOL2 regulatory variants) — "
                "must request dedicated promoter/5'-UTR sequencing or specific OVOL2 MLPA."
            ),
            "Corneal Cross-Linking (CXL) — Keratoconus Standard of Care": (
                "CXL (riboflavin + UV-A, wavelength 365 nm) induces covalent cross-links between collagen fibrils via "
                "reactive oxygen species generation → stiffens stroma → halts ectasia progression. "
                "Epithelium-off CXL (standard Dresden protocol): epithelium removed, riboflavin 0.1% instilled 30 min, "
                "UV-A 3 mW/cm² for 30 min (5.4 J/cm²). "
                "Eligibility: documented progression (≥1D Kmax change per year), thinnest point >400 µm, age <40 typically. "
                "Efficacy: 90%+ halt progression at 5 years; some regression of ectasia (flattening) in young patients. "
                "Accelerated CXL (9-45 mW, shorter time): equivalent efficacy at 1 year; may be less effective at 5 years. "
                "Epi-on CXL (transepithelial): riboflavin penetrates intact epithelium — less stromal penetration, lower efficacy. "
                "Contraindications: severe apical scarring, hydrops, thinnest point <400 µm (standard protocol). "
                "After CXL: corneal haze 1-3 months, resolves; RGP contact lens fit after 3-6 months for refractive correction."
            ),
            "DMEK vs DSAEK vs PKP — Corneal Transplant Decision Algorithm": (
                "DMEK (Descemet Membrane Endothelial Keratoplasty): "
                "Transplants only Descemet membrane + endothelium (~10-20 µm thick tissue); "
                "fastest visual recovery (20/20 in 65%, 20/25 in 90% at 1 year); lower rejection rate; "
                "higher rebubbling rate (20-40%) — easily treated with air injection; "
                "technically demanding surgical preparation; preferred for Fuchs. "
                "DSAEK (Descemet-Stripping Automated Endothelial Keratoplasty): "
                "Transplants ~100-150 µm of posterior stroma + Descemet + endothelium; "
                "slower recovery (6-12 months); lower rebubbling; preferred for complex cases, vitrectomised eyes, CHED2. "
                "DALK (Deep Anterior Lamellar Keratoplasty): "
                "Replaces anterior stroma only; preserves recipient Descemet + endothelium; "
                "preferred for keratoconus, stromal dystrophies (TGFBI deep), no endothelial rejection risk; "
                "Big Bubble technique (Anwar) creates cleavage plane at Descemet level — gold standard DALK approach. "
                "PKP (Penetrating Keratoplasty): Full-thickness corneal transplant; higher rejection risk (1%/year); "
                "reserved for failed lamellar, scarred Descemet, combined disease."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (gene 0 only) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first key) ===")
    defn = generate_definitions()
    first_key = next(iter(defn["gene_entries"]))
    print(json.dumps(defn["gene_entries"][first_key], indent=2)[:1500])
