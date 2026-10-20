#!/usr/bin/env python3
"""Hereditary-NF-kB-Innate-Immune-Signaling-Defect-Atlas -- Complete 8-Gene Reference
Hereditary NF-κB / TIR-Pathway Innate Immune Signaling Defects: Pyogenic Bacterial Susceptibility,
EDA-Immunodeficiency, and Autoinflammatory-Immunodeficiency Overlap
(IRAK4-378aa-12q12-AR-LOF-Pyogenic-All-TIR-Receptors-IRAK4-Kinase-Absent-S.pneumo-S.aureus-Age-Related-Resistance,
 MYD88-296aa-3p22.2-AR-LOF-MyD88-Adaptor-TLR1-4-7-9-IL1R-IL18R-Phenocopy-IRAK4-Mortality-Infancy,
 TIRAP-221aa-11q24.2-AR-LOF-TLR2-TLR4-Bridging-Adaptor-S180L-Heterozygous-Protective-Malaria-TB,
 IKBKG-419aa-Xq28-XL-Hypomorphic-NEMO-EDA-ID-Ectodermal-Dysplasia-Anhidrosis-Conical-Teeth-Pyogenic-Mycobacteria,
 IKBKB-756aa-8p11.21-AR-LOF-IKK-Beta-NF-kB-Absent-ALL-TIR-TNFR-SCID-Like-B-Cell-Nonfunctional,
 IRAK1-712aa-Xq28-XL-LOF-IL-1R-TLR-Signaling-Kinase-IRAK4-Sequential-Rare-Inflammatory-Phenotype,
 RELA-551aa-11q13.1-AD-LOF-p65-NF-kB-Haploinsufficiency-IBD-Like-Aphthous-Ulcers-Paradoxical-Autoinflammation,
 NFKBIA-317aa-14q13.2-AD-GOF-IkBa-PEST-Undegraded-NF-kB-Trapped-Cytoplasm-EDA-ID-Identical-NEMO):

IRAK4   (IL-1 Receptor-Associated Kinase 4; 378aa; 12q12;
           AR LOF — most common hereditary defect of the TIR-pathway; first identified by
             Picard/Casanova 2003; prevalence ~1:50,000–1:100,000;
           MECHANISM: IRAK4 is the master kinase at the proximal node of ALL MyD88-dependent TIR
             signaling; structure: N-terminal death domain (DD) + kinase domain; SIGNALING SEQUENCE:
             (1) TLR/IL-1R ligand binding → TIR domain-mediated receptor dimerization → MyD88
             recruitment via TIR-TIR interaction → MyD88 death domain oligomerizes and recruits
             IRAK4 via DD-DD interaction; (2) IRAK4 autophosphorylates (kinase activation) → IRAK4
             phosphorylates IRAK1 (on Thr209/Thr387) → IRAK1 autophosphorylates (hyperphosphorylation)
             → IRAK1 dissociates from Myddosome → recruits TRAF6 → K63-polyubiquitin chain on TRAF6
             → TAK1 (MAP3K7)/TAB complex activation → IKK complex (IKKα/β/γ-NEMO) activation →
             IκBα phosphorylation → proteasomal degradation → NF-κB (p50/RelA) nuclear translocation
             → pro-inflammatory gene expression (TNF, IL-6, IL-1β, CXCL8);
           RECEPTORS AFFECTED BY IRAK4 LOF: TLR1, TLR2, TLR4, TLR5, TLR6, TLR7, TLR8, TLR9, TLR10
             (ALL except TLR3 which uses TRIF/TICAM1 adaptor → TRIF-dependent pathway → IRF3 is
             TLR3-specific and IRAK4-independent); IL-1R, IL-18R, IL-33R, IL-36R, IL-37R (ALL IL-1
             superfamily receptors); ST2 (IL-33 receptor);
           TLR3/TRIF IS SPARED: IRAK4-deficient patients respond NORMALLY to TLR3 stimulation (poly
             I:C → IFN-β production intact via TRIF→TRAF3→TBK1→IRF3 pathway); THIS IS THE KEY
             DIAGNOSTIC DISCRIMINATOR vs UNC93B1 deficiency (UNC93B1 LOF = ALL endosomal TLRs 3/7/8/9
             lost; IRAK4 LOF = TLR7/8/9 lost but TLR3 INTACT);
           SUSCEPTIBILITY PATTERN: INVASIVE BACTERIAL INFECTIONS by encapsulated bacteria (S. pneumoniae
             #1 — invasive pneumococcal disease (IPD) with bacteremia; S. aureus #2 — invasive
             staphylococcal disease; Pseudomonas aeruginosa, Salmonella, Shigella also reported; rarely
             fungi); VIRAL INFECTIONS: SPARED (IRAK4-deficient patients have normal antiviral immunity
             because type I IFN production in response to viruses predominantly uses IRF3/IRF7 pathways
             via STING and RIG-I/MDA5 which are IRAK4-independent); influenza/COVID outcomes normal;
           AGE-RELATED RESISTANCE — UNIQUE IN IMMUNODEFICIENCY: IRAK4 deficiency is the ONLY known
             primary immunodeficiency where the clinical phenotype IMPROVES with age; children <8 yr:
             life-threatening invasive bacterial infections; children 8-12 yr: decreasing frequency;
             adults: majority have FEW or NO infections; MECHANISM: immature immune cells are absolutely
             TLR-dependent for antimicrobial responses; adult immune system develops TLR-independent
             redundant pathways (NLR-dependent, complement-opsonization, B cell memory antibodies,
             trained innate immunity, NK cell memory) that compensate for IRAK4 absence;
           PRACTICAL CONSEQUENCE: HSCT is NOT routinely indicated for IRAK4 deficiency (unlike SCID)
             because spontaneous clinical improvement is expected; MANAGEMENT: (1) aggressive antibiotic
             prophylaxis in childhood (penicillin prophylaxis); (2) vaccination (PCV13/PPSV23 and all
             encapsulated bacteria vaccines — but response may be suboptimal due to impaired TLR-
             dependent B cell activation; T cell-independent polysaccharide responses blunted);
             (3) IVIG (anti-pneumococcal/anti-staphylococcal antibodies); (4) G-CSF in severe
             infections; avoid live vaccines in infancy;
           DIAGNOSTIC: IRAK4 protein by Western blot (absent or truncated); IRAK4 kinase activity assay
             (phosphorylation of IRAK1 substrate); TLR stimulation panel (PBMCs: TLR1/2/4/5/6/7/9 →
             NO IL-6/TNF/IL-12; TLR3 → NORMAL IFN-β); IRAK4 gene sequencing):

MYD88   (Myeloid Differentiation Primary Response 88; 296aa; 3p22.2;
           AR LOF — phenotypic copy of IRAK4 deficiency; identified by Picard/Casanova 2008;
             slightly less common than IRAK4 deficiency;
           MECHANISM: MyD88 is the UNIVERSAL ADAPTOR PROTEIN for ALL MyD88-dependent TIR receptors;
             structure: N-terminal Death Domain (DD) + intermediate domain (ID) + C-terminal TIR
             domain; SIGNALING: TIR domain of TLR/IL-1R → TIR domain of MyD88 (TIR-TIR homophilic
             interaction) → MyD88 DD oligomerizes with other MyD88 DDs + IRAK4 DDs → IRAK4 → IRAK1 →
             TRAF6 → NF-κB (as above); ABSENT MYD88 → IRAK4 cannot be recruited (no platform for
             DD-DD interaction) → identical downstream signaling failure as IRAK4 LOF;
           MYD88 vs IRAK4 DISTINCTION: TLR1/2/6 (which signal via TIRAP-MyD88 in conjunction)
             are EQUALLY defective in both; IL-1R/IL-18R/IL-33R defects are identical; VIRAL
             IMMUNITY IS SPARED IN BOTH (TLR3/TRIF intact, STING intact, RIG-I/MDA5 intact);
           CLINICAL PHENOTYPE IDENTICAL TO IRAK4: invasive S. pneumoniae (meningitis, bacteremia,
             pneumonia) + S. aureus invasive disease; mortality highest in infancy (25% in pre-
             antibiotic era series); SAME AGE-RELATED RESISTANCE (improves after ~8-12 yr);
           ONE DISTINCTION: MYD88 LOF patients may have slightly MORE inflammation-prone phenotype
             in adult life vs IRAK4 LOF — some reports of inflammatory bowel disease (IBD) elements,
             possibly because MyD88 signaling normally SUPPRESSES gut epithelial inflammatory
             responses via regulatory IL-10-dependent pathways that require MyD88;
           ALSO: MYD88 GOF variants (not LOF) are found in 30-38% of Waldenström's macroglobulinemia
             (WM) and diffuse large B cell lymphoma (DLBCL) as somatic activating mutations
             (L265P — L265P is the dominant somatic GOF; NOT the same as germline AR LOF causing
             immunodeficiency); PATHOGNOMONIC PEARL: L265P is the WM/DLBCL oncogenic mutation
             (ibrutinib target), NOT the immunodeficiency mutation; germline AR LOF variants are
             different pathogenic variants causing immune deficiency;
           DIAGNOSTIC: MYD88 protein by Western blot (absent); TLR stimulation panel (same as IRAK4
             — no response to TLR1/2/4/5/7/9; normal TLR3 response); MYD88 sequencing;
           KEY PEARL: IRAK4 vs MYD88 deficiency cannot be distinguished clinically; only molecular
             testing differentiates them; BOTH are excellent candidates for the "asplenic patient
             with invasive pneumococcal disease" extended workup):

TIRAP   (TIR Domain-Containing Adaptor Protein; also called MAL/TIRAP; 221aa; 11q24.2;
           AR LOF (complete deficiency — extremely rare; biallelic LOF);
             heterozygous S180L variant (rs8177374) = MALARIA/TB PROTECTIVE — population frequency
             14-20% in sub-Saharan African, 10% South Asian; first described by Bhatt 2003;
           MECHANISM: TIRAP (MAL) is a BRIDGING ADAPTOR specific for TLR2 and TLR4; TIRAP has its
             own TIR domain + N-terminal PIP2-binding domain that anchors TIRAP to the plasma membrane
             PI(4,5)P2-enriched signaling platforms; TIRAP bridges TLR2 and TLR4 TIR domains to
             MyD88 TIR domain (MyD88 alone cannot directly bind TLR2/4 without TIRAP); TIRAP is NOT
             required for TLR7/8/9 (endosomal) → MyD88 (direct interaction for endosomal TLRs);
           CONSEQUENCE OF TIRAP LOF: TLR2 (recognizes lipoteichoic acid/LTA from gram-positive bacteria,
             peptidoglycan, lipopeptides, fungal β-glucan) and TLR4 (recognizes LPS from gram-negative
             bacteria, heat shock proteins, respiratory syncytial virus fusion protein) → IMPAIRED
             SIGNALING specifically (other TLRs intact); CLINICAL: predominantly gram-positive
             (S. pneumoniae, S. aureus via TLR2 LTA) and gram-negative (E. coli, K. pneumoniae,
             Salmonella via TLR4 LPS) pyogenic susceptibility;
           TIRAP S180L HETEROZYGOUS — MALARIA/TB PROTECTION MECHANISM: heterozygous S180L → ~50%
             TIRAP function → REDUCED (not absent) TLR2/4-mediated IL-6/TNF production → LESS
             pathological cytokine storm in malaria (Plasmodium falciparum) and TB (Mycobacterium
             tuberculosis); EVOLUTIONARY ADVANTAGE: malaria/TB killed >50% of children in sub-Saharan
             Africa historically → heterozygous S180L survivors → reproductive advantage → high allele
             frequency; THIS IS A CLASSIC "HETEROZYGOUS ADVANTAGE" or "BALANCED POLYMORPHISM" example
             in immunology genetics analogous to sickle cell (HBB S/A) and malaria; CLINICAL PEARL:
             patient from sub-Saharan Africa with unusual susceptibility pattern — check TIRAP; patient
             with recurrent malaria/TB — consider TIRAP S180L;
           BIALLELIC TIRAP LOF (true AR): very rare; 5-10 pedigrees reported; features: similar to
             but less severe than IRAK4/MYD88 deficiency; specifically TLR2/4 pathway impairment;
             gram-positive and gram-negative invasive bacterial disease; viral immunity intact;
           DIAGNOSTIC: TIRAP protein expression; TLR2/4 stimulation with Pam3CSK4 (TLR1/2) and LPS
             (TLR4) → impaired IL-6/TNF; TLR7 (R848) → normal; TIRAP sequencing;
           TREATMENT: same as IRAK4/MYD88 — antibiotic prophylaxis, vaccination, IVIG in childhood):

IKBKG   (Inhibitor of Nuclear Factor Kappa B Kinase Regulatory Subunit Gamma / NEMO; 419aa; Xq28;
           X-linked — COMPLETE LOF = INCONTINENTIA PIGMENTI (IP), lethal in males (in utero);
             HYPOMORPHIC mutations = EDA-ID (Anhidrotic Ectodermal Dysplasia with Immunodeficiency);
           NEMO STRUCTURE: coiled-coil domains (CC1, CC2) + leucine zipper (LZ) + ZINC finger (ZF);
             NEMO (IKKγ) is the regulatory subunit of the IKK complex (IKKα + IKKβ + NEMO); NEMO
             bridges activated upstream kinases (TAK1, RIP1, TRAF6, NIK) to catalytic IKKα/β →
             IKKβ phosphorylates IκBα (Ser32/Ser36) → SCF-βTrCP E3 ligase → K48-polyubiquitin →
             26S proteasome → IκBα degraded → NF-κB (p50/RelA) released → nucleus → gene expression;
           EDA-ID (ECTODERMAL DYSPLASIA WITH IMMUNODEFICIENCY): features:
             (1) ECTODERMAL DYSPLASIA TRIAD: (a) ANHIDROSIS (inability to sweat — absence/deficiency
             of eccrine sweat glands; NF-κB signaling required for sweat gland morphogenesis via
             EDAR-EDARADD-TRAF6-NF-κB pathway; CLINICAL DANGER: heat stroke in summer/exercise;
             PATHOGNOMONIC: affected infant cannot sweat even in intense heat → hyperthermia → heat
             stroke → first presentation at 6-12 months in summer); (b) HYPOTRICHOSIS (sparse/absent
             scalp and body hair; all ectodermal appendages require NF-κB for development); (c)
             OLIGODONTIA/CONICAL TEETH (absent or peg-shaped teeth especially anteriors; dental
             X-rays at 3-4 yr to assess tooth bud development);
             (2) IMMUNODEFICIENCY: COMBINED (T cell-dependent and innate); pyogenic bacteria
             (S. pneumoniae, S. aureus) + mycobacteria (BCG vaccine ABSOLUTELY CONTRAINDICATED
             — can cause disseminated BCGosis; non-tuberculous mycobacteria; M. tuberculosis);
             some patients: hyper-IgM phenotype (high IgM, low IgG, IgA) due to impaired class switch
             recombination (CSR requires NF-κB downstream of CD40L-CD40 in B cells);
           INCONTINENTIA PIGMENTI (COMPLETE NEMO LOF): X-linked dominant; lethal in males; females
             mosaic (X-inactivation); presents in neonatal females with: (1) blistering skin lesions
             following Blaschko lines (stage I) → verrucous hyperkeratotic plaques (II) → hyperpigmented
             swirling (III) → atrophic/pallid (IV); (2) CNS: seizures, intellectual disability; (3)
             retinal: vascular abnormality; NEMO complete LOF → NF-κB absent in all tissues → embryonic
             lethal in hemizygous males without compensatory X; female mosaic survives;
           DIAGNOSTIC: NEMO protein by flow cytometry and Western blot; NF-κB activation assay after
             TLR stimulation; ectodermal features (anhidrosis by sweat test, dental panoramic X-ray,
             hair follicle biopsy); IKBKG sequencing (targeted + MLPA for deletion screening);
           MANAGEMENT: HSCT for severe immunodeficiency phenotype; BCG vaccination ABSOLUTELY
             CONTRAINDICATED; cooling vest/environment modification for anhidrosis; dental reconstruction;
             IVIG; prophylactic antibiotics + antifungals; G-CSF):

IKBKB   (Inhibitor of Nuclear Factor Kappa B Kinase Subunit Beta / IKKβ; 756aa; 8p11.21;
           AR LOF — extremely rare; ~20 patients reported; severe combined immunodeficiency
             with NF-κB defect;
           MECHANISM: IKBKB encodes IKK-beta (IKKβ), the primary catalytic kinase in the IKK complex
             that phosphorylates IκBα → degradation → canonical NF-κB activation; IKKβ is ESSENTIAL
             for inflammatory (canonical) NF-κB pathway; IKKα handles non-canonical NF-κB (RelB/p52)
             independently; IKBKB LOF → canonical NF-κB (p65/p50) signaling ABSENT downstream of
             ALL TIR receptors (TLR1-9, IL-1R, IL-18R, IL-33R) AND ALL TNFR family receptors (TNFR1/2,
             FasR (CD95), CD40, RANK, BAFF-R, BCMA) → essentially ABSENT NF-κB in ALL inflammatory
             and lymphocyte-survival pathways;
           CLINICAL PHENOTYPE — SCID-LIKE COMBINED IMMUNODEFICIENCY:
             (1) B CELLS NUMERICALLY PRESENT BUT NONFUNCTIONAL: B cells develop normally from bone
             marrow (pro-B → pre-B → immature B; early B cell survival signals are BCR-PI3K/AKT
             dependent, not NF-κB dependent); BUT functional differentiation requires NF-κB: B cell
             activation via BCR+CD40L → NF-κB → antibody production, class switch, plasma cell
             differentiation ALL blocked; BAFF-R → NF-κB survival signal absent → peripheral B cell
             survival impaired; RESULT: B cells present on CBC/flow but hypogammaglobulinemia (IgG/
             IgA/IgM all low); IVIG-dependent;
             (2) T CELLS: CD4 and CD8 present; T cell activation requires NF-κB for IL-2 transcription
             and T cell survival in activation-induced cell death (AICD) regulation; T cell responses
             impaired but cells present (milder than SCID);
             (3) NK CELLS: present; NK cytotoxicity partially preserved;
             (4) MONOCYTE/MACROPHAGE: cytokine production absent (TNF, IL-6, IL-12 cannot be made
             in response to TLR stimulation);
           DISTINCTION FROM NEMO EDA-ID: IKBKB deficiency lacks ectodermal dysplasia features (EDA
             pathway uses EDAR-TRAF6-NEMO, so NEMO deficiency disrupts EDA morphogenesis; but IKKβ
             and IKKα are downstream of EDAR/TRAF6 and are not exclusively required for sweat gland
             morphogenesis at the early fetal stage); IKBKB = immunodeficiency without EDA;
           DIAGNOSTIC: NF-κB activation ABSENT after TLR/TNF stimulation (EMSA / NF-κB reporter);
             IKKβ Western blot (absent/severely reduced); IKBKB sequencing;
           TREATMENT: HSCT curative (stem cells with normal NF-κB signaling restore both lymphocyte
             function and innate immune responses); IVIG + antibiotic prophylaxis pre-HSCT; BCG
             ABSOLUTELY CONTRAINDICATED; avoid live vaccines):

IRAK1   (IL-1 Receptor-Associated Kinase 1; 712aa; Xq28;
           X-linked (Xq28 — same chromosome region as IKBKG/NEMO but distinct locus);
             LOF is extremely rare; most variants are hemizygous in males;
           MECHANISM: IRAK1 is the second kinase in the TIR signaling cascade, acting DOWNSTREAM of
             IRAK4; IRAK4 phosphorylates IRAK1 → IRAK1 undergoes rapid hyperphosphorylation (>20
             serine/threonine sites) → hyperphosphorylated IRAK1 dissociates from Myddosome →
             recruits and activates TRAF6 via IRAK1 C-terminal domain → TRAF6 K63-polyubiquitin
             chains → TAK1 → IKK → NF-κB; IRAK1 also independently activates IRF5 (in TLR7/9
             pathway → type I IFN and IL-12) and interacts with TRAF3 in some contexts;
           IRAK1 LOF vs IRAK4 LOF: IRAK4 kinase activity is ABSOLUTELY required for IRAK1
             phosphorylation and Myddosome signaling → IRAK4 deficiency more severe; IRAK1 LOF
             has PARTIAL redundancy because IRAK2 can partially substitute for IRAK1 in TRAF6
             recruitment (IRAK2 lacks kinase activity but has TRAF6-binding capability); THEREFORE:
             IRAK1 deficiency is clinically LESS severe than IRAK4 deficiency;
           CLINICAL FEATURES: variable; most IRAK1-deficient males reported have had inflammatory
             phenotypes (recurrent bacterial infections + elevated inflammatory markers); some
             patients with IRAK1 LOF have features of immune dysregulation rather than pure
             immunodeficiency; inflammatory bowel disease-like features; recurrent aphthous ulcers;
           IRAK1 AND CLONAL DISEASE: IRAK1 somatic GOF variants associated with myelodysplastic
             syndrome (MDS) and clonal hematopoiesis; distinct from germline LOF;
           X-LINKED PATTERN: no father-to-son transmission; carrier mothers have 50% risk for
             affected sons; sisters of affected males have 50% carrier risk; genetic counseling for
             X-linked family pedigrees;
           DIAGNOSTIC: IRAK1 protein Western blot; IRAK1 kinase activity; TLR7/9 stimulation panel
             (R848, CpG): mildly impaired IL-6/TNF production; IRAK1 sequencing; X-linkage pedigree;
           TREATMENT: antibiotic prophylaxis for bacterial infections; anti-inflammatory therapy
             (NSAIDs, short corticosteroids) for inflammatory phenotype; IVIG if hypogammaglobulinemia;
             HSCT not typically required (phenotype less severe)):

RELA    (v-rel Reticuloendotheliosis Viral Oncogene Homolog A / NF-κB p65; 551aa; 11q13.1;
           AD LOF haploinsufficiency — RELA is the predominant Rel homology domain (RHD) subunit
             forming the classical p65/p50 NF-κB heterodimer; first described by Badran/Sancho/
             Forbes-Hernandez et al. in IBD cohorts 2019-2020;
           MECHANISM: RELA (p65) is the transactivation subunit of canonical NF-κB; contains:
             N-terminal RHD (DNA binding + dimerization + IκBα interaction + nuclear localization
             signal) + C-terminal transactivation domains (TA1, TA2); RELA haploinsufficiency →
             50% p65 protein → quantitatively reduced NF-κB target gene induction; KEY TARGETS
             REQUIRING FULL P65 DOSE: (1) anti-apoptotic genes (BCL2, BCL-XL, XIAP, c-FLIP) in
             intestinal epithelial cells (IEC) → IEC apoptosis increased → epithelial barrier defect
             → dysbiosis → inflammation; (2) TNFAIP3 (A20, the deubiquitinase that terminates NF-κB
             signaling) → REDUCED A20 expression → PROLONGED/HYPERACTIVATED NF-κB in RESPONSE TO
             TNF/IL-1 (paradox: haploinsufficiency of p65 → more inflammation because A20 feedback
             is lost); (3) MUC2 (intestinal mucin) and tight junction proteins (claudin, occludin) →
             barrier dysfunction;
           CLINICAL PHENOTYPE — PARADOXICAL AUTOINFLAMMATORY-IMMUNODEFICIENCY:
             (1) EARLY-ONSET INFLAMMATORY BOWEL DISEASE: severe, treatment-resistant IBD (Crohn-like
             ileocolitis, pan-colitis); onset typically <5 years; HISTOLOGY: increased IEC apoptosis
             (PATHOGNOMONIC feature distinguishing from classical Crohn/UC — high apoptotic body count
             on biopsy); inflammatory infiltrate; crypt abscess;
             (2) RECURRENT SEVERE APHTHOUS STOMATITIS/APHTHOUS ULCERS: oral aphthae are often the
             presenting feature; can precede gut disease; LARGE aphthous ulcers (herpetiform or major
             subtype); extremely refractory to standard treatments (steroids, thalidomide);
             (3) SUSCEPTIBILITY TO BACTERIAL/VIRAL INFECTIONS: concurrent immunodeficiency component
             (reduced p65 → reduced transcription of antimicrobial peptides, reduced cytokine storm →
             bacterial/fungal susceptibility alongside the inflammatory phenotype);
             (4) SKIN: eczematous/psoriasiform lesions in some pedigrees;
           TREATMENT: anti-TNF therapy (infliximab/adalimumab) MOST EFFECTIVE for IBD component
             (reduces TNF-driven inflammation that amplifies via the broken A20 feedback loop);
             JAK inhibitors (tofacitinib, upadacitinib) emerging; bone marrow transplant/HSCT for
             refractory cases; biologics (vedolizumab, ustekinumab); TPN support for malnutrition;
           DIAGNOSTIC: RELA sequencing; p65 protein quantification by Western blot (50% level);
             NF-κB activation assay (normal activation to TNF/IL-1 but altered kinetics and A20
             expression); intestinal biopsy (increased IEC apoptosis); FcγR-mediated pathway intact):

NFKBIA  (Nuclear Factor of Kappa Light Polypeptide Gene Enhancer in B-Cells Inhibitor Alpha /
           IκBα; 317aa; 14q13.2;
           AD GOF — gain-of-function mutations (usually in PEST domain or Ser32/Ser36 phosphorylation
             sites that target IκBα for degradation); first described by Courtois/Smahi group 2003;
           MECHANISM: IκBα (NFKBIA) is the primary inhibitor of canonical NF-κB; IκBα structure:
             N-terminal signal receiving domain (contains Ser32/Ser36 phosphorylation sites) + central
             ankyrin repeats (AR; 7 ankyrin repeats that interact with NF-κB RHD and mask nuclear
             localization signal (NLS) of p65/p50) + C-terminal PEST domain (proline-glutamate-serine-
             threonine rich; substrate recognition for constitutive IκBα turnover by CBP/p300-independent
             mechanisms); DEGRADATION SEQUENCE: IKKβ phosphorylates Ser32+Ser36 → SCF-βTrCP E3
             ubiquitin ligase recognizes phospho-Ser32/36 → K48 polyubiquitin at Lys21/Lys22 → 26S
             proteasome degrades IκBα → NF-κB translocates to nucleus;
           NFKBIA GOF MUTATIONS: variants that PREVENT IκBα phosphorylation (Ser32A, Ser36A) or
             PREVENT IκBα degradation (PEST domain deletions, Lys21/22 mutation, frameshift):
             → IκBα cannot be degraded upon IKKβ activation → NF-κB remains TRAPPED in cytoplasm
             → NF-κB SIGNALING ABSENT despite normal IKK activation → phenotypically IDENTICAL to
             NEMO (IKBKG) hypomorphic EDA-ID; CONVERGENT PATHWAY PHENOTYPE;
           CLINICAL — EDA-ID (IDENTICAL TO NEMO EDA-ID):
             (1) ECTODERMAL DYSPLASIA: anhidrosis (heat stroke risk), hypotrichosis, oligodontia/
               conical teeth; same ectodermal triad as NEMO/EDA-ID; MECHANISM: EDAR-TRAF6-NEMO-IKKβ-
               IκBα → NF-κB required for sweat gland morphogenesis; IκBα GOF blocks NF-κB → same
               EDA as NEMO hypomorphic;
             (2) IMMUNODEFICIENCY: pyogenic bacteria (S. pneumoniae, S. aureus) + mycobacteria
               (BCG ABSOLUTELY CONTRAINDICATED); hyper-IgM (impaired CSR); T cell dysfunction;
               pneumocystis jirovecii pneumonia;
           KEY CLINICAL DISTINCTION NEMO vs NFKBIA:
             (1) INHERITANCE: NEMO/IKBKG = X-linked (males affected >> females mosaic carrier);
               NFKBIA = AUTOSOMAL DOMINANT GOF (males AND females equally affected from first generation);
             (2) SEVERITY: NFKBIA GOF can be more severe (complete NF-κB block in ALL cells vs NEMO
               hypomorphic partial block); NFKBIA null = EMBRYONIC LETHAL in mice (NF-κB uncontrolled
               during development);
             (3) DIAGNOSTIC DISCRIMINATION: Western blot — NEMO: reduced NEMO protein; NFKBIA GOF:
               NORMAL IκBα protein level but FAILS TO DEGRADE after TNF/IL-1 stimulation (use
               timecourse Western blot: IκBα level fails to drop at 5-15 min post-stimulation = GOF);
               NEMO: IκBα fails to degrade because NEMO absent (cannot activate IKKβ); NFKBIA GOF:
               IκBα fails to degrade because IκBα resistant to phosphorylation/ubiquitination;
           DIAGNOSTIC: NFKBIA sequencing (Ser32/36 codons, PEST domain); IκBα degradation assay
             (timecourse: TNF stimulation → IκBα should be undetectable at 15 min by Western; if
             persistent = GOF confirmed); NF-κB EMSA/reporter (absent after stimulation);
           MANAGEMENT: HSCT curative; BCG ABSOLUTELY CONTRAINDICATED; IVIG; antibiotic prophylaxis;
             dental/ectodermal management identical to NEMO EDA-ID; cooling protocols for anhidrosis):
"""

import random
import statistics

ATLAS = "Hereditary-NF-kB-Innate-Immune-Signaling-Defect-Atlas"
GENES = ["IRAK4", "MYD88", "TIRAP", "IKBKG", "IKBKB", "IRAK1", "RELA", "NFKBIA"]
SEEDS = {
    "IRAK4":  4094,
    "MYD88":  4095,
    "TIRAP":  4096,
    "IKBKG":  4097,
    "IKBKB":  4098,
    "IRAK1":  4099,
    "RELA":   4100,
    "NFKBIA": 4101,
}

PROFILES = {
    "IRAK4": {
        "syndrome":        "IRAK4-Pyogenic-Bacterial-Susceptibility",
        "locus":           "12q12",
        "inheritance":     "AR LOF",
        "size_aa":         378,
        "eda":             False,
        "age_resistance":  True,
        "hsct_indicated":  False,
        "bcg_ci":          False,
        "hyper_igm":       False,
        "ectodermal":      False,
        "autoinflammation":False,
        "age_onset_range": (0, 3),
        "attack_types":    ["invasive_pneumococcal", "invasive_staphylococcal"],
        "viral_spared":    True,
        "tlr3_intact":     True,
    },
    "MYD88": {
        "syndrome":        "MYD88-Pyogenic-Bacterial-Susceptibility",
        "locus":           "3p22.2",
        "inheritance":     "AR LOF",
        "size_aa":         296,
        "eda":             False,
        "age_resistance":  True,
        "hsct_indicated":  False,
        "bcg_ci":          False,
        "hyper_igm":       False,
        "ectodermal":      False,
        "autoinflammation":False,
        "age_onset_range": (0, 2),
        "attack_types":    ["invasive_pneumococcal", "invasive_staphylococcal", "meningitis"],
        "viral_spared":    True,
        "tlr3_intact":     True,
    },
    "TIRAP": {
        "syndrome":        "TIRAP-TLR2-TLR4-Bridging-Adaptor-Defect",
        "locus":           "11q24.2",
        "inheritance":     "AR LOF",
        "size_aa":         221,
        "eda":             False,
        "age_resistance":  True,
        "hsct_indicated":  False,
        "bcg_ci":          False,
        "hyper_igm":       False,
        "ectodermal":      False,
        "autoinflammation":False,
        "age_onset_range": (0, 5),
        "attack_types":    ["gram_positive_invasive", "gram_negative_invasive"],
        "viral_spared":    True,
        "tlr3_intact":     True,
    },
    "IKBKG": {
        "syndrome":        "NEMO-EDA-ID-Ectodermal-Dysplasia-Immunodeficiency",
        "locus":           "Xq28",
        "inheritance":     "X-linked Hypomorphic",
        "size_aa":         419,
        "eda":             True,
        "age_resistance":  False,
        "hsct_indicated":  True,
        "bcg_ci":          True,
        "hyper_igm":       True,
        "ectodermal":      True,
        "autoinflammation":False,
        "age_onset_range": (0, 2),
        "attack_types":    ["invasive_pneumococcal", "mycobacterial", "gram_negative_invasive"],
        "viral_spared":    False,
        "tlr3_intact":     False,
    },
    "IKBKB": {
        "syndrome":        "IKK-Beta-SCID-NF-kB-Absent-All-Pathways",
        "locus":           "8p11.21",
        "inheritance":     "AR LOF",
        "size_aa":         756,
        "eda":             False,
        "age_resistance":  False,
        "hsct_indicated":  True,
        "bcg_ci":          True,
        "hyper_igm":       False,
        "ectodermal":      False,
        "autoinflammation":False,
        "age_onset_range": (0, 1),
        "attack_types":    ["invasive_pneumococcal", "mycobacterial", "fungal", "viral_severe"],
        "viral_spared":    False,
        "tlr3_intact":     False,
    },
    "IRAK1": {
        "syndrome":        "IRAK1-X-Linked-TIR-Kinase-Defect",
        "locus":           "Xq28",
        "inheritance":     "X-linked LOF",
        "size_aa":         712,
        "eda":             False,
        "age_resistance":  True,
        "hsct_indicated":  False,
        "bcg_ci":          False,
        "hyper_igm":       False,
        "ectodermal":      False,
        "autoinflammation":True,
        "age_onset_range": (0, 8),
        "attack_types":    ["invasive_bacterial", "inflammatory_bowel"],
        "viral_spared":    True,
        "tlr3_intact":     True,
    },
    "RELA": {
        "syndrome":        "RELA-p65-Haploinsufficiency-IBD-Autoinflammatory",
        "locus":           "11q13.1",
        "inheritance":     "AD LOF",
        "size_aa":         551,
        "eda":             False,
        "age_resistance":  False,
        "hsct_indicated":  False,
        "bcg_ci":          False,
        "hyper_igm":       False,
        "ectodermal":      False,
        "autoinflammation":True,
        "age_onset_range": (0, 4),
        "attack_types":    ["ibd_colitis", "aphthous_ulcers", "skin_inflammation"],
        "viral_spared":    True,
        "tlr3_intact":     True,
    },
    "NFKBIA": {
        "syndrome":        "IkBa-GOF-EDA-ID-NF-kB-Trapped",
        "locus":           "14q13.2",
        "inheritance":     "AD GOF",
        "size_aa":         317,
        "eda":             True,
        "age_resistance":  False,
        "hsct_indicated":  True,
        "bcg_ci":          True,
        "hyper_igm":       True,
        "ectodermal":      True,
        "autoinflammation":False,
        "age_onset_range": (0, 2),
        "attack_types":    ["invasive_pneumococcal", "mycobacterial", "pcp"],
        "viral_spared":    False,
        "tlr3_intact":     False,
    },
}


def _make_patients(gene: str, n: int = 40) -> list[dict]:
    prof = PROFILES[gene]
    rng = random.Random(SEEDS[gene])
    patients = []
    for i in range(n):
        age_min, age_max = prof["age_onset_range"]
        age_onset = round(rng.uniform(age_min, age_max + 2), 1)
        age_current = round(rng.uniform(max(age_onset + 1, age_onset), age_onset + 35), 1)
        sex = "M" if (prof["inheritance"].startswith("X-linked") or rng.random() < 0.5) else "F"
        if prof["inheritance"] == "AD LOF" and gene == "RELA":
            sex = rng.choice(["M", "F"])

        invasive_bacterial = rng.random() < (0.90 if gene in ["IRAK4", "MYD88"] else
                                              0.75 if gene in ["TIRAP"] else
                                              0.85 if gene in ["IKBKG", "IKBKB"] else
                                              0.50 if gene == "IRAK1" else
                                              0.30 if gene == "RELA" else 0.88)
        mycobacterial = rng.random() < (0.0 if gene in ["IRAK4", "MYD88", "TIRAP", "IRAK1"] else
                                         0.55 if gene in ["IKBKG"] else
                                         0.45 if gene in ["IKBKB"] else
                                         0.0 if gene == "RELA" else 0.50)
        viral_severe = rng.random() < (0.05 if prof["viral_spared"] else 0.40)
        eda_features = rng.random() < (0.85 if prof["ectodermal"] else 0.0)
        autoinflammation = rng.random() < (0.80 if prof["autoinflammation"] else 0.05)
        hyper_igm = rng.random() < (0.60 if prof["hyper_igm"] else 0.10)
        heat_intolerance = rng.random() < (0.90 if gene in ["IKBKG", "NFKBIA"] else 0.05)
        age_resistance_observed = (rng.random() < 0.70 and age_current > 12
                                   if prof["age_resistance"] else False)
        infections_per_yr = round(rng.uniform(3, 8) if invasive_bacterial and age_current < 10
                                   else rng.uniform(0.5, 3), 1)
        ivig_dependent = rng.random() < (0.50 if gene in ["IRAK4", "MYD88"] else
                                          0.80 if gene in ["IKBKG", "IKBKB", "NFKBIA"] else
                                          0.30 if gene == "IRAK1" else
                                          0.20 if gene in ["TIRAP", "RELA"] else 0.50)
        hsct_done = rng.random() < (0.45 if prof["hsct_indicated"] else 0.02)
        tlr_panel_done = rng.random() < 0.75
        tlr3_normal = prof["tlr3_intact"] if tlr_panel_done else None

        patients.append({
            "gene":                  gene,
            "age_onset":             age_onset,
            "age_current":           age_current,
            "sex":                   sex,
            "invasive_bacterial":    invasive_bacterial,
            "mycobacterial_disease": mycobacterial,
            "viral_severe":          viral_severe,
            "eda_features":          eda_features,
            "heat_intolerance":      heat_intolerance,
            "autoinflammatory":      autoinflammation,
            "hyper_igm":             hyper_igm,
            "age_resistance":        age_resistance_observed,
            "infections_per_yr":     infections_per_yr,
            "ivig_dependent":        ivig_dependent,
            "hsct_done":             hsct_done,
            "tlr_panel_done":        tlr_panel_done,
            "tlr3_normal":           tlr3_normal,
        })
    return patients


def _cohort_stats(patients: list[dict]) -> dict:
    n = len(patients)
    return {
        "n":                      n,
        "invasive_bacterial_n":   sum(1 for p in patients if p["invasive_bacterial"]),
        "mycobacterial_n":        sum(1 for p in patients if p["mycobacterial_disease"]),
        "viral_severe_n":         sum(1 for p in patients if p["viral_severe"]),
        "eda_n":                  sum(1 for p in patients if p["eda_features"]),
        "heat_intolerance_n":     sum(1 for p in patients if p["heat_intolerance"]),
        "autoinflammatory_n":     sum(1 for p in patients if p["autoinflammatory"]),
        "hyper_igm_n":            sum(1 for p in patients if p["hyper_igm"]),
        "age_resistance_n":       sum(1 for p in patients if p["age_resistance"]),
        "ivig_n":                 sum(1 for p in patients if p["ivig_dependent"]),
        "hsct_n":                 sum(1 for p in patients if p["hsct_done"]),
        "tlr_panel_n":            sum(1 for p in patients if p["tlr_panel_done"]),
        "tlr3_normal_n":          sum(1 for p in patients if p["tlr3_normal"] is True),
        "mean_age_onset":         round(statistics.mean(p["age_onset"] for p in patients), 1),
        "mean_infections_yr":     round(statistics.mean(p["infections_per_yr"] for p in patients), 1),
    }


def generate_overview():
    all_patients: list[dict] = []
    cohort_stats: dict[str, dict] = {}
    for g in GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        cohort_stats[g] = _cohort_stats(pts)

    return {
        "atlas":          ATLAS,
        "subtitle":       "Hereditary NF-κB/TIR Innate Immune Signaling Defects: 8-Gene Complete Reference",
        "total_patients": len(all_patients),
        "genes":          GENES,
        "seeds":          SEEDS,
        "cohort_summary": {
            g: {
                "syndrome":            PROFILES[g]["syndrome"],
                "inheritance":         PROFILES[g]["inheritance"],
                "locus":               PROFILES[g]["locus"],
                "n":                   cohort_stats[g]["n"],
                "invasive_bact_pct":   round(cohort_stats[g]["invasive_bacterial_n"] / 40 * 100, 1),
                "mycobacterial_pct":   round(cohort_stats[g]["mycobacterial_n"] / 40 * 100, 1),
                "viral_spared":        PROFILES[g]["viral_spared"],
                "eda_pct":             round(cohort_stats[g]["eda_n"] / 40 * 100, 1),
                "autoinflammatory_pct":round(cohort_stats[g]["autoinflammatory_n"] / 40 * 100, 1),
                "age_resistance":      PROFILES[g]["age_resistance"],
                "hsct_indicated":      PROFILES[g]["hsct_indicated"],
                "bcg_ci":              PROFILES[g]["bcg_ci"],
                "mean_age_onset":      cohort_stats[g]["mean_age_onset"],
            } for g in GENES
        },
        "key_facts": {
            "irak4_myd88_age_resistance":
                "IRAK4 + MYD88 deficiency: ONLY primary immunodeficiency where clinical phenotype spontaneously IMPROVES with age; TLR-independent immunity compensates after age ~8-12 yr.",
            "tlr3_intact_irak4":
                "IRAK4/MYD88-deficient patients have INTACT TLR3/TRIF-IFN-β response — viral immunity spared; TLR3 uses TRIF adaptor (not MyD88) → IRAK4-INDEPENDENT pathway.",
            "eda_id_two_genes":
                "EDA-ID (ectodermal dysplasia + immunodeficiency): caused by BOTH IKBKG (NEMO hypomorphic, X-linked) AND NFKBIA GOF (IκBα resistant to degradation, AD); CONVERGENT PATHWAY — different gene, identical phenotype.",
            "nemo_incontinentia_pigmenti":
                "IKBKG COMPLETE LOF = Incontinentia Pigmenti: lethal in males; females mosaic (Blaschko-line skin pigmentation, CNS involvement). HYPOMORPHIC = EDA-ID. Test ALL IKBKG variants for hypomorphic vs complete.",
            "bcg_absolutely_ci":
                "BCG vaccination ABSOLUTELY CONTRAINDICATED in IKBKG (NEMO), IKBKB (IKKβ), and NFKBIA GOF — disseminated BCGosis risk; mycobacterial susceptibility in all three.",
            "rela_paradox":
                "RELA haploinsufficiency: PARADOXICAL — p65 LOF causes AUTOINFLAMMATION (IBD, aphthous ulcers) NOT just immunodeficiency; reduced A20 (TNFAIP3) feedback → TNF hyperactivation; anti-TNF is first-line.",
            "tirap_malaria_protection":
                "TIRAP S180L heterozygous (14-20% of sub-Saharan Africans): PROTECTIVE against malaria and severe TB — evolutionary balanced polymorphism; biallelic = immunodeficiency.",
            "ikbkb_b_cells_present":
                "IKBKB deficiency: B CELLS NUMERICALLY PRESENT but nonfunctional — NF-κB required for B cell activation/CSR/plasma cell, not for B cell development; hypogammaglobulinemia with normal B cell count = NF-κB defect clue.",
        },
    }


def generate_breakdown():
    all_patients: list[dict] = []
    cohort_stats: dict[str, dict] = {}
    for g in GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        cohort_stats[g] = _cohort_stats(pts)

    pathway_groups = {
        "myddosome_kinases": {
            "genes":       ["IRAK4", "MYD88", "TIRAP", "IRAK1"],
            "description": "Proximal TIR signaling: MyD88 adaptor (MYD88), TIRAP bridging adaptor (TLR2/4), IRAK4 master kinase, IRAK1 downstream effector. All cause pyogenic bacterial susceptibility. IRAK4/MYD88: phenotypically identical; age-related improvement unique. TIRAP: S180L population variant protective against malaria. IRAK1: X-linked, milder.",
            "key_test":    "TLR stimulation panel: PBMCs stimulated with Pam3CSK4 (TLR1/2), LPS (TLR4), R848 (TLR7/8), CpG (TLR9), poly I:C (TLR3) → measure IL-6/TNF/IFN-β; IRAK4/MYD88: all TLR responses absent except TLR3; TIRAP: TLR2/4 specifically impaired; IRAK1: partial TLR7/9 impairment",
        },
        "ikk_complex": {
            "genes":       ["IKBKG", "IKBKB"],
            "description": "IKK complex: NEMO (IKBKG, regulatory) and IKKβ (IKBKB, catalytic). Both cause NF-κB absent in canonical pathway. IKBKG hypomorphic = EDA-ID (X-linked). IKBKB LOF = SCID-like without EDA. Both: mycobacterial susceptibility, BCG CI, HSCT indicated.",
            "key_test":    "NEMO protein by Western + flow; IKKβ Western; NF-κB EMSA after TNF stimulation; ectodermal features (sweat test, dental panoramic); BCG history",
        },
        "nfkb_downstream": {
            "genes":       ["NFKBIA", "RELA"],
            "description": "NF-κB regulation: IκBα GOF (NFKBIA — traps NF-κB; EDA-ID, AD) and p65 haploinsufficiency (RELA — IBD/autoinflammation, AD). Convergent on NF-κB nuclear entry failure (NFKBIA) vs reduced target gene expression (RELA). NFKBIA = immunodeficiency + EDA. RELA = paradoxical autoinflammatory IBD.",
            "key_test":    "IκBα degradation assay (TNF stimulation timecourse: IκBα persists at 15 min = NFKBIA GOF); RELA p65 quantification 50% level; biopsy IEC apoptosis count; anti-TNF response",
        },
    }

    return {
        "atlas":          ATLAS,
        "total_patients": len(all_patients),
        "cohort_stats":   cohort_stats,
        "pathway_groups": pathway_groups,
        "gene_summary": {
            g: {
                "gene":              g,
                "syndrome":          PROFILES[g]["syndrome"],
                "n":                 cohort_stats[g]["n"],
                "seed":              SEEDS[g],
                "invasive_bact_pct": round(cohort_stats[g]["invasive_bacterial_n"] / 40 * 100, 1),
                "eda_pct":           round(cohort_stats[g]["eda_n"] / 40 * 100, 1),
                "mean_age_onset":    cohort_stats[g]["mean_age_onset"],
                "bcg_ci":            PROFILES[g]["bcg_ci"],
                "hsct_indicated":    PROFILES[g]["hsct_indicated"],
            } for g in GENES
        },
    }


def generate_definitions():
    return {
        "definitions": [
            {
                "term": "IRAK4 Kinase — Master Node of ALL MyD88-Dependent TIR Signaling (IRAK4 vs UNC93B1 Discriminator)",
                "definition": "IRAK4 is the single most critical kinase in the MyD88-dependent branch of innate immune signaling, making its deficiency the best characterized cause of selective susceptibility to pyogenic (pus-forming) bacteria. SIGNALING HIERARCHY: upon TLR/IL-1R ligation → MyD88 death domain (DD) oligomerizes with IRAK4 DD → IRAK4 autophosphorylates within the Myddosome complex → phosphorylates IRAK1 (Thr209, Thr387, Ser376) → IRAK1 hyperphosphorylation → IRAK1 recruits TRAF6 → K63-ubiquitin chain on TRAF6 → TAK1 → IKKβ → NF-κB. RECEPTORS REQUIRING IRAK4: TLR1, 2, 4, 5, 6, 7, 8, 9, 10 (all MyD88-dependent TLRs); ALL IL-1 superfamily receptors (IL-1R, IL-18R, IL-33R [ST2], IL-36R, IL-37R, TIR8/SIGIRR). TLR3 (which uses TRIF, not MyD88) is IRAK4-INDEPENDENT → IFN-β production via TLR3 is INTACT in IRAK4 deficiency. DIAGNOSTIC PANEL (PBMC STIMULATION): TLR4 (LPS) → IL-6/TNF = ABSENT in IRAK4; TLR7/8 (R848) → IL-6/TNF = ABSENT; TLR9 (CpG) → IL-6/IL-12 = ABSENT; TLR3 (poly I:C) → IFN-β = NORMAL (key discriminator); DISTINGUISHES FROM UNC93B1: UNC93B1 LOF loses ALL endosomal TLRs (3+7+8+9 all absent) while IRAK4 LOF has TLR3 intact. AGE-RELATED RESISTANCE: the only primary immunodeficiency that self-improves clinically; children <8yr: dangerous invasive bacterial infections; adults: largely asymptomatic because TLR-independent innate pathways (NK cells, complement, phagocyte oxidative burst, trained innate immunity) provide redundant protection; PRACTICAL IMPLICATION: HSCT is NOT routinely recommended for IRAK4 deficiency (unlike SCID) — manage with prophylaxis and monitor for clinical improvement."
            },
            {
                "term": "EDA-ID — Ectodermal Dysplasia with Immunodeficiency: Two Genes, One Phenotype (IKBKG vs NFKBIA)",
                "definition": "EDA-ID is the paradigm example of convergent pathway genetics in primary immunodeficiency, where defects in two different genes — IKBKG (NEMO, X-linked hypomorphic) and NFKBIA (IκBα GOF, autosomal dominant) — produce clinically indistinguishable EDA-ID phenotype. CONVERGENT MECHANISM: EDAR (ectodysplasin receptor on ectodermal cells) → EDARADD → TRAF6 → NEMO/IKKγ → IKKβ → IκBα phosphorylation → IκBα degradation → NF-κB nuclear entry → sweat gland, hair follicle, tooth bud morphogenesis genes expressed; NEMO hypomorphic: reduced NEMO → IKKβ less efficiently activated → IκBα incompletely degraded → NF-κB reduced → EDA; NFKBIA GOF: IκBα resistant to phosphorylation/degradation → NF-κB trapped in cytoplasm → same EDA; CLINICAL EDA TRIAD (both): (1) ANHIDROSIS (absent eccrine sweat glands → unable to sweat → LIFE-THREATENING HEAT STROKE in infants in summer; presenting emergency at 6-12 months; temperature >40°C → seizures; cooling vest/air conditioning mandatory); (2) HYPOTRICHOSIS (sparse/absent hair; eyebrows, scalp; often not noticed in infancy); (3) OLIGODONTIA + CONICAL TEETH (absent or peg-shaped teeth; dental panoramic X-ray at 3-4 yr mandatory — tooth buds absent = confirmation); IMMUNODEFICIENCY (both): pyogenic bacteria (S. pneumo, S. aureus) + mycobacteria (BCG ABSOLUTELY CI in both); hyper-IgM pattern (class switch recombination requires NF-κB downstream of CD40 → impaired CSR → high IgM, low IgG, IgA); combined T + B cell dysfunction; DISTINGUISHING FEATURE: INHERITANCE — NEMO/IKBKG: X-linked (only males affected clinically; females are mosaic asymptomatic carriers — unless complete LOF = Incontinentia Pigmenti in females); NFKBIA GOF: autosomal dominant (males AND females affected equally from first generation; vertical transmission; no sex bias). WESTERN BLOT DISCRIMINATION: NEMO absent → IκBα degradation impaired because upstream IKKβ cannot be assembled; NFKBIA GOF → IκBα PRESENT but fails to degrade at 15 min post-TNF (timecourse critical). TREATMENT: HSCT for severe phenotype in both; BCG absolutely contraindicated; dental reconstruction; cooling protocols."
            },
            {
                "term": "RELA Haploinsufficiency — Paradoxical Autoinflammatory IBD from NF-κB LOF",
                "definition": "RELA (p65) haploinsufficiency is counterintuitive: losing HALF of the NF-κB p65 transactivation subunit causes AUTOINFLAMMATION (IBD, aphthous ulcers) rather than pure immunodeficiency. The paradox arises from two mechanisms. MECHANISM 1 — A20 FEEDBACK LOSS: RELA is required for transcription of TNFAIP3 (A20), the deubiquitinase that terminates NF-κB signaling by removing K63-ubiquitin chains from TRAF6, RIP1, and NEMO; RELA LOF → reduced A20 expression → PROLONGED NF-κB activation in response to TNF/IL-1 (less A20 = less negative feedback = more NF-κB activity per stimulus, despite lower baseline p65); NET EFFECT: individual NF-κB stimulation events cause EXAGGERATED and PROLONGED responses → TNF-driven enterocolitis, gut inflammation; MECHANISM 2 — INTESTINAL EPITHELIAL CELL APOPTOSIS: RELA (p65) is required for transcription of anti-apoptotic genes BCL-2, BCL-XL, XIAP, c-FLIP in intestinal epithelial cells (IEC); IEC survival absolutely requires NF-κB anti-apoptotic signaling against constitutive TNF from gut bacteria; RELA LOF → IEC apoptosis increased → epithelial barrier disrupted → luminal bacteria enter submucosa → inflammatory cascade; HISTOPATHOLOGICAL CLUE: INCREASED IEC APOPTOSIS on biopsy (higher apoptotic body density than classical Crohn/UC) — PATHOGNOMONIC feature of RELA deficiency vs other IBD causes. CLINICAL: early-onset severe IBD (<5 yr); MAJOR APHTHOUS ULCERS (large oral ulcers, often presenting sign); growth failure; malnutrition (TPN often required). TREATMENT: anti-TNF (infliximab/adalimumab) is MOST EFFECTIVE (reduces the TNF-driven hyperactivation component) — NOT standard IBD treatment with aminosalicylates/azathioprine which work poorly; vedolizumab (gut-selective integrin inhibitor), ustekinumab (IL-12/23 inhibitor) emerging; tofacitinib (JAK inhibitor) for refractory; HSCT for the most severe cases."
            },
            {
                "term": "BCG Vaccination Contraindication — IKBKG, IKBKB, NFKBIA: Disseminated BCGosis Risk",
                "definition": "BCG (Bacillus Calmette-Guérin, live attenuated Mycobacterium bovis) vaccination is ABSOLUTELY CONTRAINDICATED in IKBKG hypomorphic (NEMO EDA-ID), IKBKB LOF (IKKβ), and NFKBIA GOF (IκBα GOF EDA-ID) — all three NF-κB pathway defects impair mycobacterial immunity. MECHANISM: control of M. tuberculosis and related mycobacteria requires: (1) TLR2/4-MyD88-IRAK4-NF-κB (macrophage cytokine production: IL-12, TNF-α, IL-6 → drive Th1 differentiation); (2) IFNGR-JAK1/2-STAT1 → macrophage activation/killing; (3) TNFR-NF-κB → granuloma formation maintenance (critical: TNF-NF-κB maintains the granuloma wall surrounding mycobacteria); NF-κB DEFECTS IN IKBKG/IKBKB/NFKBIA: impaired IL-12 production (TLR2/4-NF-κB pathway blunted) → reduced IFN-γ production from T cells (insufficient IL-12 drive) + impaired granuloma maintenance (NF-κB absent in macrophages → TNF response blunted); BCGosis (disseminated BCG disease) results when BCG vaccinitis spreads beyond injection site to nodes, blood, and distant organs; CLINICAL BCGOSIS: fever, lymphadenopathy at injection site → regional nodes → hepatosplenomegaly → lung infiltrates → FATAL without recognition and treatment; TREATMENT OF BCGOSIS: isoniazid + rifampicin + ethambutol (3-drug regimen) for 12-18 months; HSCT urgently if BCGosis in IKBKG/IKBKB; ALSO CONTRAINDICATED IN: SCID (T- combined defects), CGD (phagocyte oxidative burst absent), MSMD (Mendelian susceptibility to mycobacterial disease: IFNGR1/2, IL12RB1/2, STAT1, NEMO, IL12B, IKBKG, CYBB, IRF8, ISG15, TYK2). PROTOCOL: before any live vaccine in any infant with family history of EDA/immunodeficiency, exclude NEMO/IκBα GOF; defer BCG until immune workup complete."
            },
            {
                "term": "TLR Stimulation Panel — Diagnostic Protocol for TIR/IRAK4/MYD88 Pathway Defects",
                "definition": "The TLR stimulation panel is the definitive functional assay for diagnosing defects in the TIR → MyD88 → IRAK4 → NF-κB signaling cascade. PROTOCOL: isolate PBMCs from patient blood; stimulate with: (1) Pam3CSK4 (TLR1/2 agonist; bacterial lipopeptide) → expected IL-6/TNF (TIRAP + MyD88 dependent); (2) LPS E. coli (TLR4 agonist) → expected IL-6/TNF (TIRAP + MyD88 dependent); (3) Poly I:C (TLR3 agonist) → expected IFN-β (TRIF dependent; IRAK4-INDEPENDENT); (4) R848 (TLR7/8 agonist; single-stranded RNA mimetic) → expected IL-6/TNF (MyD88 dependent); (5) CpG ODN (TLR9 agonist; unmethylated CpG DNA) → expected IL-6/IL-12 (MyD88 dependent); (6) IL-1β cytokine → expected IL-6 (IL-1R → MyD88 → IRAK4 → NF-κB); INTERPRETATION — IRAK4 LOF: stimuli 1/2/4/5/6 = ABSENT responses; stimulus 3 (TLR3) = NORMAL IFN-β; INTERPRETATION — MYD88 LOF: identical to IRAK4 (all MyD88-dependent absent; TLR3 normal); INTERPRETATION — TIRAP LOF: stimuli 1+2 (TLR2/4) = ABSENT; stimuli 4+5+6 (TLR7/8/9 + IL-1R) = NORMAL (TIRAP-independent); INTERPRETATION — IKBKG/IKBKB/NFKBIA: ALL stimuli (including TLR3 and IL-1R) = ABSENT NF-κB activation; IFN-β via IRF3/TRIF may still be partially intact in NEMO/IκBα defects (IRF3 uses a different downstream arm); KEY DISCRIMINATOR: TLR3 response is the most informative — intact TLR3 = TIR/MyD88/IRAK4 defect; absent TLR3 = broader innate immune defect (NEMO, IκBα, UNC93B1); PRACTICAL NOTE: PBMCs must be fresh (< 6 hours from collection); stimulation time 6-18 hours; read IL-6/TNF/IFN-β by ELISA or multiplex bead assay; cell viability >85% required."
            },
            {
                "term": "TIRAP S180L — Evolutionary Balanced Polymorphism: Heterozygous Malaria/TB Protection",
                "definition": "TIRAP S180L (rs8177374, Ser180Leu; also written as S180L or MAL S180L) is one of the clearest examples of a balanced polymorphism in human innate immunity — heterozygosity is advantageous while homozygosity causes disease. POPULATION FREQUENCY: 14-20% in sub-Saharan African populations, 10-12% in South Asian, 8% in East Asian; maintained by positive selection. MECHANISM: TIRAP S180L → ~50% TIRAP function → REDUCED (not absent) TLR2/4-driven IL-6/TNF/IL-12 → LESS pathological CYTOKINE STORM in malaria and tuberculosis: (1) P. falciparum malaria: severe malaria cerebral edema and organ damage driven partly by TNF-α and IL-6 overproduction; S180L → lower cytokine → better-tolerated malaria (infectious challenge still cleared by other mechanisms); (2) M. tuberculosis: paradoxically, TB disease in developing regions partly driven by excessive inflammatory tissue destruction; S180L → less IL-6/TNF-driven immunopathology → less tissue damage despite similar bacillary burden; heterozygotes have 40% lower odds of severe malaria vs wild-type (BHATT 2003 NEJM); HOMOZYGOUS S180L (rare): biallelic TLR2/4 pathway defect → pyogenic bacterial susceptibility; ANALOGY TO SICKLE CELL: HbAS (sickle cell trait, heterozygous) = malaria protection; HbSS = sickle cell disease; TIRAP S180L/WT = malaria/TB protection; TIRAP S180L/S180L = immunodeficiency; CLINICAL APPLICATION: (1) patient from malaria-endemic region with unusual susceptibility to gram-positive/negative bacteria → TIRAP biallelic sequencing; (2) counseling populations where S180L is common: heterozygosity = protective, not pathogenic; (3) evolutionary medicine: this explains why S180L has not been selected out despite causing homozygous disease — heterozygous advantage in malaria belt maintains allele frequency."
            },
            {
                "term": "IκBα Degradation Assay — Western Blot Timecourse for NFKBIA GOF Diagnosis",
                "definition": "The IκBα degradation assay (timecourse Western blot) is the definitive functional test for distinguishing NFKBIA GOF from IKBKG hypomorphic and from normal, as IκBα behavior post-stimulation is uniquely diagnostic. PROTOCOL: (1) culture primary fibroblasts or PBMCs from patient; (2) stimulate with TNF-α (10 ng/mL) or IL-1β (10 ng/mL) → IKKβ is activated by TNFR1-TRADD-RIP1-TRAF2 or IL-1R-MyD88-IRAK4-TRAF6; (3) collect cells at 0, 5, 10, 15, 20, 30, 60 minutes post-stimulation; (4) Western blot for IκBα (antibody to total IκBα protein); NORMAL TIMECOURSE: IκBα baseline (T0) → DISAPPEARS at T5-T15 min (phosphorylation by IKKβ → ubiquitination → proteasomal degradation) → REAPPEARS at T30-60 min (NF-κB transcribes new IκBα as negative feedback = resynthesis); NFKBIA GOF RESULT: IκBα level REMAINS CONSTANT at all timepoints — does NOT degrade; IKKβ IS ACTIVE (NEMO present, IKK activation normal) but IκBα RESISTS PHOSPHORYLATION (Ser32/36 mutant) or RESISTS UBIQUITINATION (PEST domain variant) → NF-κB remains trapped; IKBKG (NEMO) HYPOMORPHIC RESULT: IκBα also FAILS TO DEGRADE but for DIFFERENT REASON — NEMO absent → IKKβ cannot be assembled/activated → IκBα not phosphorylated → also persists; CRITICAL DISCRIMINATION: phospho-IκBα Western (antibody to phospho-Ser32/36): NFKBIA GOF → IκBα IS NOT PHOSPHORYLATED; NEMO hypomorphic → IκBα is also NOT phosphorylated; FURTHER DISCRIMINATION: IKKβ kinase assay (use NEMO-independent substrate) → NEMO hypomorphic: IKKβ kinase activity LOW; NFKBIA GOF: IKKβ kinase activity PRESENT; PRACTICAL NOTE: fibroblast cultures preferred (stable, reproducible); PBMCs also work but more variable. GENETIC CONFIRMATION required regardless of functional result."
            },
            {
                "term": "Cascade Diagnostic Algorithm — NF-kB/TIR Pathway Defect Workup",
                "definition": "STEP 1 — CLINICAL PHENOTYPE TRIAGE: (A) Invasive bacterial infections (pyogenic: S.pneumo/S.aureus) WITHOUT ectodermal features + WITHOUT autoinflammation → IRAK4/MYD88 panel FIRST; age at onset <5yr; viral sparing noted; (B) Invasive bacterial + mycobacterial + ECTODERMAL DYSPLASIA (anhidrosis/oligodontia/hypotrichosis) → IKBKG (NEMO) + NFKBIA panel; BCG vaccination history; (C) Males only affected (no father-to-son transmission) → X-linked genes first: IKBKG (Xq28), IRAK1 (Xq28); NEMO and IRAK1 are BOTH on Xq28 — full Xq28 sequencing; (D) Early-onset IBD + aphthous ulcers WITHOUT clear EDA → RELA haploinsufficiency; p65 Western; intestinal biopsy IEC apoptosis; anti-TNF response; (E) BOTH gram-positive AND gram-negative susceptibility + family from malaria-endemic region → TIRAP S180L genotyping; TLR2/4 stimulation panel. STEP 2 — TLR PANEL (all suspected TIR defects): poly I:C (TLR3) + LPS (TLR4) + Pam3CSK4 (TLR1/2) + R848 (TLR7/8) + CpG (TLR9) + IL-1β → cytokines measured; TLR3 intact = MyD88/IRAK4/TIRAP; TLR3 absent = NEMO/IKKβ/IκBα. STEP 3 — PROTEIN LEVELS: IRAK4 Western (absent = IRAK4 deficiency); MYD88 Western (absent = MYD88); NEMO flow cytometry (absent/reduced); IKKβ Western; IκBα degradation assay (timecourse: persistent = NFKBIA GOF); p65 level (50% = RELA haploinsufficiency). STEP 4 — SEQUENCING: targeted gene panel (all 8 genes) or exome; MLPA for IKBKG deletion. STEP 5 — MANAGEMENT DECISION: HSCT indicated: IKBKG severe / IKBKB / NFKBIA severe; NOT routinely HSCT: IRAK4, MYD88 (age-resistance expected); RELA (anti-TNF first); BCG REVIEW: if BCG already given to IKBKG/IKBKB/NFKBIA → immediate ATT prophylaxis, monitor for BCGosis."
            },
        ],
        "key_distinguishers": {
            "irak4_age_resistance":   "IRAK4 + MYD88: ONLY immunodeficiency that spontaneously IMPROVES with age (~8-12 yr); HSCT NOT routinely indicated; manage with prophylaxis.",
            "tlr3_intact_irak4":      "IRAK4/MYD88 deficiency: TLR3/TRIF-IFN-β pathway INTACT; viral immunity SPARED; confirms TIR/MyD88 branch defect (not pan-innate immune).",
            "eda_id_two_genes":       "EDA-ID from IKBKG (X-linked hypomorphic) OR NFKBIA GOF (AD): same phenotype, different inheritance; IκBα timecourse WB discriminates.",
            "bcg_ci_nemo_ikkb":       "BCG ABSOLUTELY CONTRAINDICATED in IKBKG + IKBKB + NFKBIA; mycobacterial susceptibility; disseminated BCGosis risk in all three.",
            "rela_anti_tnf_first":    "RELA haploinsufficiency: anti-TNF (infliximab) is MOST EFFECTIVE treatment — NOT conventional IBD therapy; p65 LOF paradoxically causes autoinflammation via reduced A20.",
            "tirap_malaria_balance":  "TIRAP S180L heterozygous: MALARIA/TB PROTECTIVE (balanced polymorphism); biallelic = TLR2/4-specific pyogenic susceptibility.",
            "ikbkb_b_cells_present":  "IKBKB deficiency: B cells numerically present but nonfunctional (NF-κB required for B cell activation not development); hypogammaglobulinemia + normal B cell count = NF-κB clue.",
            "nemo_ip_vs_eda_id":      "IKBKG COMPLETE LOF = Incontinentia Pigmenti (females, mosaic, lethal in males); IKBKG HYPOMORPHIC = EDA-ID (males clinically); distinguish by variant class + functional testing.",
        },
        "treatment_matrix": {
            "prophylactic_antibiotics":   "IRAK4 + MYD88 + TIRAP + IKBKG: penicillin prophylaxis in childhood; co-trimoxazole (TMP-SMX) if broad spectrum needed; stop when age-resistance achieved (IRAK4/MYD88).",
            "ivig":                       "IKBKG + IKBKB + NFKBIA + IRAK4 + MYD88 (if hypogammaglobulinemia): 400-600mg/kg q3-4 weeks; anti-pneumococcal/staphylococcal IgG titers critical.",
            "hsct":                       "IKBKG severe + IKBKB + NFKBIA: HSCT curative; conditioning required; matched sibling or MUD; fix immunodeficiency but NOT the ectodermal features (EDA persists post-HSCT).",
            "anti_tnf":                   "RELA haploinsufficiency: infliximab (anti-TNF mAb) or adalimumab for IBD component; most effective treatment; address A20-driven TNF hyperactivation.",
            "bcg_protocol":               "BCG ABSOLUTELY CONTRAINDICATED in IKBKG + IKBKB + NFKBIA; if given inadvertently → immediate INH + rifampicin + ethambutol ATT; monitor nodes + imaging.",
            "cooling_eda":                "IKBKG EDA-ID + NFKBIA EDA-ID: cooling vests/room air conditioning critical; alert caregivers re: summer heat stroke; MedicAlert bracelet; school cooling protocol.",
            "dental_reconstruction":      "IKBKG + NFKBIA EDA-ID: dental implants at skeletal maturity (18-20 yr); interim dentures in childhood; annual panoramic dental X-ray from age 3; orthodontic consultation.",
            "vaccination_schedule":       "All NF-κB defects: avoid all live vaccines; PCV13/PPSV23 (give even if response suboptimal); yearly inactivated influenza; conjugate Hib + meningococcal ACWY and B.",
        },
    }
