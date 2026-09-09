#!/usr/bin/env python3
"""Hereditary-Phagocyte-Deficiency-Atlas — Complete 8-Gene Hereditary Phagocyte Deficiency Atlas
(CYBB · NCF1 · CYBA · NCF2 · ITGB2 · FERMT3 · ELANE · HAX1).

CYBB     (Cytochrome b-245 Beta Chain / gp91phox / NOX2; 570 aa; Xp21.1-p11.4; XLR;
          Chronic Granulomatous Disease type X1 (CGD-X1);
          MOST COMMON CGD — accounts for ~65% of all CGD worldwide;
          XLR: males severely affected; female carriers can have lyonisation → partial CGD;
          DHR FLOW CYTOMETRY ZERO OXIDATIVE BURST PATHOGNOMONIC;
          RECURRENT CATALASE-POSITIVE ORGANISM INFECTIONS: Aspergillus + Staphylococcus aureus;
          seed SEED_BASE+0).
NCF1     (Neutrophil Cytosol Factor 1 / p47phox; 390 aa; 7q11.23; AR;
          Chronic Granulomatous Disease type AR2 (CGD-AR2);
          MOST COMMON AR CGD — ~25% of all CGD worldwide;
          GT DELETION FOUNDER in pseudogenes NCF1B/NCF1C — 85% of NCF1 CGD carry this;
          seed SEED_BASE+1).
CYBA     (Cytochrome b-245 Alpha Chain / p22phox; 195 aa; 16q24.2; AR;
          Chronic Granulomatous Disease type AR1 (CGD-AR1);
          p22phox stabilises gp91phox → absent p22phox = absent cytochrome b558 complex;
          seed SEED_BASE+2).
NCF2     (Neutrophil Cytosol Factor 2 / p67phox; 526 aa; 1q25.3; AR;
          Chronic Granulomatous Disease type AR3 (CGD-AR3);
          p67phox activation domain required for gp91phox NOX2 electron transfer;
          seed SEED_BASE+3).
ITGB2    (Integrin Subunit Beta 2 / CD18; 769 aa; 21q22.3; AR;
          Leukocyte Adhesion Deficiency type I (LAD-I); most common LAD (>80%);
          CD18 pairs with CD11a/b/c → all beta-2 integrins absent;
          OMPHALITIS + DELAYED CORD SEPARATION (>21 days) + LEUKOCYTOSIS WITHOUT PUS PATHOGNOMONIC;
          seed SEED_BASE+4).
FERMT3   (Fermitin Family Member 3 / Kindlin-3; 667 aa; 11q13.1; AR;
          Leukocyte Adhesion Deficiency type III (LAD-III);
          Kindlin-3 activates both beta-2 integrins (neutrophils) AND beta-3 integrins (platelets);
          LAD + GLANZMANN-LIKE BLEEDING DIATHESIS COMBINATION PATHOGNOMONIC;
          seed SEED_BASE+5).
ELANE    (Elastase Neutrophil Expressed / Neutrophil Elastase / NE; 256 aa; 19p13.3; AD;
          Severe Congenital Neutropenia type 1 (SCN1) and Cyclic Neutropenia (CyN);
          ELANE misfolding → ER stress → UPR → neutrophil apoptosis at promyelocyte stage;
          21-DAY CYCLE ANC NADIR + ORAL ULCERS + FEVER PATHOGNOMONIC for Cyclic Neutropenia;
          G-CSF LIFELONG FIRST-LINE TREATMENT;
          seed SEED_BASE+6).
HAX1     (HCLS1-Associated Protein X-1; 279 aa; 1q21.3; AR;
          Severe Congenital Neutropenia type 3 / Kostmann Disease;
          HAX1 inhibits granulocyte apoptosis; HAX1 deficiency → massive neutrophil apoptosis;
          NEUROLOGICAL INVOLVEMENT (epilepsy, cognitive) PATHOGNOMONIC in HAX1 isoform B loss;
          Original 1956 Kostmann Swedish pedigree — first SCN ever described;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2326-2333).
"""

import random

SEED_BASE = 2326

PHAGOCYTE_GENES = [
    # -- CYBB — CGD-X1 (gp91phox) -------------------------------------------------------
    {
        "gene": "CYBB",
        "alt_name": (
            "CYBB (CYBB-570aa-Xp21.1 / XLR — CGD-X1 — "
            "MOST-COMMON-CGD-65pct-All-CGD — "
            "DHR-ZERO-OXIDATIVE-BURST-PATHOGNOMONIC — "
            "ASPERGILLUS+Staphylococcus-Catalase-Positive-Organisms-PRIMARY-THREATS — "
            "TMP-SMX+Itraconazole+IFN-gamma-LIFELONG-PROPHYLAXIS-HSCT-Curative)"
        ),
        "protein": (
            "CYBB -- Xp21.1-p11.4 XLR -- CYBB-570aa -- "
            "Cytochrome-b-245-Beta-Chain-gp91phox-NOX2-91kDa-Heavily-Glycosylated-Transmembrane -- "
            "Large-Subunit-NADPH-Oxidase-Core-Electron-Transfer-Chain-O2-to-Superoxide -- "
            "6-TM-Domains-FAD-NADPH-Binding-Sites-Haem-Groups-B558-Complex-With-p22phox -- "
            "CYBB-Deficiency-No-Superoxide-O2•−-No-H2O2-No-HOCl-No-Oxidative-Killing -- "
            "CATALASE-POSITIVE-BACTERIA-Staph-aureus-Serratia-Burkholderia-Nocardia-Survive -- "
            "ASPERGILLUS-Most-Common-Fungal-Killer-In-CGD-Pneumonia-Dissemination-Lethal -- "
            "DHR-Dihydrorhodamine-Flow-Cytometry-ZERO-Shift-Males-Gold-Standard-Diagnosis -- "
            "FEMALE-CARRIERS-Lyonisation-Variable-DHR-30-60pct-Shift-Carrier-Testing-Mandatory -- "
            "OMIM-Gene-300481-Disease-CGD-X1-306400"
        ),
        "locus": "Xp21.1-p11.4",
        "protein_size": "570 aa / 91 kDa",
        "inheritance": (
            "XLR (X-linked recessive); hemizygous males severely affected (complete CGD); "
            "heterozygous females = carriers (usually asymptomatic); "
            "lyonisation in carriers → variable DHR oxidative burst (30-60% → partial CGD in some carriers); "
            "de novo mutations account for ~10% of CYBB CGD; "
            "CYBB mutations: missense, nonsense, splice, large deletions/rearrangements; "
            "genotype-phenotype: residual gp91phox expression (hypomorphic) → milder CGD; null → severe; "
            "MOST COMMON CGD GENE: ~65% of all CGD worldwide"
        ),
        "phagocyte_category": "CGD type X1 (CYBB/gp91phox; NADPH oxidase core; X-linked; most common CGD ~65%)",
        "pathognomonic": (
            "DHR FLOW CYTOMETRY — ZERO OXIDATIVE BURST: neutrophils exposed to PMA → no DHR fluorescence shift in affected males (normal = bright green shift); female carriers show bimodal pattern (two populations: normal + absent); "
            "RECURRENT CATALASE-POSITIVE ORGANISM INFECTIONS: Staphylococcus aureus, Serratia marcescens, Burkholderia cepacia (lung cavities), Nocardia, Aspergillus fumigatus; "
            "ASPERGILLUS PNEUMONIA: most common fungal cause of death in CGD; cavitating lung lesions; requires voriconazole + surgical resection if extensive; "
            "INFLAMMATORY GRANULOMA FORMATION: hyperinflammatory response even without infection → GI tract (Crohn-like), bladder (obstructive uropathy), lymph nodes (compressive)"
        ),
        "treatment": (
            "PROPHYLAXIS TRIPLE (MANDATORY LIFELONG): "
            "(1) TMP-SMX (Bactrim) 5mg/kg/day trimethoprim — antibacterial (Staph/Serratia/Burkholderia/Nocardia); "
            "(2) ITRACONAZOLE — antifungal (Aspergillus prevention; 200mg/day adult; 5mg/kg/day child); "
            "(3) IFN-GAMMA (Actimmune) 3x/week SQ — immunomodulatory; reduces infection frequency 70% vs placebo; "
            "ACUTE INFECTIONS: broad-spectrum IV antibiotics (cover catalase-positive organisms); voriconazole for Aspergillus; "
            "GRANULOMA: corticosteroids (prednisolone 1mg/kg) — effective for GI/bladder/airway granulomas; "
            "HSCT CURATIVE: allogeneic HSCT (HLA-matched sibling or MUD); best outcomes age <10 + no active infection; myeloablative conditioning (busulfan + fludarabine); "
            "GENE THERAPY: under investigation; haematopoietic stem cell gene correction (lentiviral vectors); "
            "GRANULOCYTE TRANSFUSION: life-threatening refractory infections → irradiated granulocytes (CMV-negative donor)"
        ),
        "key_features": [
            "MOST COMMON CGD GENE: ~65% of all CGD worldwide (X-linked); remaining 35% = AR forms (NCF1 25%, CYBA/NCF2/NCF4 10%)",
            "DHR FLOW CYTOMETRY ZERO OXIDATIVE BURST: dihydrorhodamine stimulated with PMA → no fluorescence shift (normal = bright fluorescent) = DIAGNOSTIC; bimodal in female carriers; must perform before ruling out CGD",
            "CATALASE-POSITIVE ORGANISMS UNIQUELY PATHOGENIC: these organisms destroy their own H2O2 → CGD neutrophils (no ROS) cannot kill them; Staphylococcus aureus = most common bacterial infection; Aspergillus = most common fungal killer",
            "HYPERINFLAMMATORY GRANULOMA = SECOND CLINICAL PROBLEM: CGD causes excessive inflammation even without infection; GI granulomas (Crohn-like — strictures/fistulae); bladder granulomas (obstructive uropathy); requires steroids",
            "TRIPLE PROPHYLAXIS MANDATORY: TMP-SMX + itraconazole + IFN-gamma ALL LIFELONG — each component has evidence; reduction in infection rates 70% (IFN-gamma RCT); non-adherence = life-threatening",
            "HSCT CURATIVE: allogeneic HSCT cures CGD; best outcomes before age 10; active Aspergillus infection increases transplant mortality; gene therapy in development",
        ],
        "monitoring": [
            "DHR oxidative burst testing at diagnosis (annually in carriers for lyonisation progression)",
            "Pulmonary function tests + HRCT chest annually (Aspergillus + granuloma surveillance)",
            "LFTs + serum itraconazole trough level every 6 months (itraconazole hepatotoxicity monitoring)",
            "CBC monthly × 6 months, then 3-monthly (baseline leukocytosis common; rising eosinophils = granuloma/GI)",
            "Upper + lower GI endoscopy if GI symptoms (Crohn-like granulomatous colitis — diagnosed early → steroids)",
            "Renal USS + post-void residual (bladder outlet obstruction from granuloma) annually",
        ],
        "key_ddx": [
            "NCF1-CGD-AR2 (most common AR CGD 25%; GT deletion founder; same infections; DHR reduced not zero; AR not XLR)",
            "Hyper-IgE syndrome / STAT3 (recurrent Staph skin abscesses + eczema + skeletal features; DHR normal; IgE very high)",
            "Myeloperoxidase deficiency (MPO) (mild CGD-like; usually asymptomatic; DHR reduced; MPO stain absent; routine labs incidental)",
            "Specific granule deficiency (SGD) (bilobed neutrophil nuclei; absent specific granules; DHR reduced; CBC)",
        ],
        "phagocyte_pathway": "NADPH oxidase (NOX2/gp91phox core) — cytochrome b558 electron chain → superoxide generation",
        "onset_age": "infant_to_early_childhood",
        "granuloma_risk": True,
        "fungal_risk": True,
        "bleeding_risk": False,
        "neurological_risk": False,
        "severity_options": [
            "Severe CGD-X1 (null CYBB; complete absence gp91phox; recurrent Aspergillus + bacterial infections; first year of life onset)",
            "Moderate CGD-X1 (hypomorphic CYBB; residual 1-5% gp91phox; delayed onset; milder infections; some survive to adulthood without HSCT)",
        ],
        "complication_options": ["Aspergillus pneumonia", "Staphylococcal liver abscess", "Burkholderia sepsis", "GI granulomas (Crohn-like)", "Bladder outlet obstruction", "Recurrent lymphadenitis"],
        "dhr_options": ["zero_burst", "severely_reduced", "absent"],
        "treatments_used": ["TMP-SMX (daily antibacterial)", "Itraconazole (daily antifungal)", "IFN-gamma 3x/week", "Voriconazole (Aspergillus treatment)", "Prednisolone (granuloma)", "HSCT (curative)"],
    },
    # -- NCF1 — CGD-AR2 (p47phox) -------------------------------------------------------
    {
        "gene": "NCF1",
        "alt_name": (
            "NCF1 (NCF1-390aa-7q11.23 / AR — CGD-AR2 — "
            "MOST-COMMON-AR-CGD-25pct-All-CGD — "
            "GT-DELETION-FOUNDER-NCF1B-NCF1C-Pseudogenes-85pct — "
            "DHR-MARKEDLY-REDUCED-NOT-ZERO-Key-DDx-CYBB)"
        ),
        "protein": (
            "NCF1 -- 7q11.23 AR -- NCF1-390aa -- "
            "Neutrophil-Cytosol-Factor-1-p47phox-47kDa-Cytosolic-Organiser-NADPH-Oxidase -- "
            "SH3-Domains-PX-Domain-Binds-Phosphoinositides-Membrane-Translocation-On-Activation -- "
            "NCF1-Phosphorylation-Triggers-Cytosol-Complex-p47phox-p67phox-p40phox-Membrane-Move -- "
            "GT-Deletion-Recurrent-NCF1-Exon-2-NCF1B-NCF1C-Pseudogenes-Intrachromosomal-Recombination -- "
            "GT-DELETION-FOUNDER-MUTATION-85pct-NCF1-CGD-Most-Common-Single-CGD-Mutation-Worldwide -- "
            "AR-CGD-AR2-CGD-Less-Severe-Than-CYBB-Residual-p47phox-Activity-Hypomorphic-Some -- "
            "OMIM-Gene-608512-Disease-CGD-AR2-233700"
        ),
        "locus": "7q11.23",
        "protein_size": "390 aa / 47 kDa",
        "inheritance": (
            "AR; biallelic NCF1 mutations; "
            "GT deletion (c.75_76delGT / p.Tyr26Cys frameshift) in exon 2: recurrent founder mutation; "
            "arises from intrachromosomal recombination between NCF1 and its pseudogenes NCF1B + NCF1C (same chromosome 7q11.23 location); "
            "85% of NCF1-CGD patients are homozygous or compound heterozygous for this GT deletion; "
            "most common AR-CGD worldwide; most common AR CGD in non-consanguineous populations; "
            "carrier frequency 1/100 in some populations (explains high NGF1-CGD prevalence); "
            "milder DHR phenotype (markedly reduced, not zero) compared to CYBB-CGD"
        ),
        "phagocyte_category": "CGD type AR2 (NCF1/p47phox; NADPH oxidase cytosolic component; most common AR CGD ~25%)",
        "pathognomonic": (
            "DHR FLOW CYTOMETRY MARKEDLY REDUCED (not zero): partial oxidative burst distinguishes NCF1-CGD-AR2 from CYBB-CGD-X1 (zero burst); "
            "GT DELETION FOUNDER MUTATION: c.75_76delGT in NCF1 exon 2 → detectable on targeted sequencing; arises from pseudogene recombination; "
            "SAME INFECTIONS AS CYBB-CGD: Staphylococcus aureus, Aspergillus, Serratia, Burkholderia — same prophylaxis required; "
            "MILDER CLINICAL COURSE on average vs CYBB: later onset (18 months vs 6 months typical), less severe Aspergillus burden; some patients diagnosed in adolescence/adulthood"
        ),
        "treatment": (
            "PROPHYLAXIS TRIPLE (IDENTICAL to CYBB-CGD — MANDATORY LIFELONG): "
            "TMP-SMX (antibacterial) + itraconazole (antifungal) + IFN-gamma (Actimmune 3x/week); "
            "ACUTE INFECTIONS: broad-spectrum IV antibiotics + voriconazole for Aspergillus; "
            "GRANULOMA: corticosteroids (prednisolone 1mg/kg) — GI/bladder/airway granulomas; "
            "HSCT: considered for severe phenotype / refractory infections; "
            "GT DELETION TESTING: NCF1 pseudogene-aware gene panel needed (standard WES may miss intragenic recombinations); "
            "CARRIER TESTING: obligate carriers (parents) + reproductive options (PGD available); "
            "PROGNOSIS: slightly better than CYBB-CGD; many patients survive to adulthood with prophylaxis"
        ),
        "key_features": [
            "MOST COMMON AR CGD: ~25% of all CGD worldwide; ~70% of AR CGD cases",
            "GT DELETION FOUNDER MUTATION: c.75_76delGT (exon 2) = most common single CGD mutation worldwide (recurrent pseudogene recombination); 85% of NCF1-CGD have this GT deletion",
            "DHR MARKEDLY REDUCED (NOT ZERO): distinguishes NCF1-CGD from CYBB-CGD; partial oxidative burst present (residual p47phox fragments or partial function)",
            "PSEUDOGENE DIAGNOSTIC PITFALL: standard WES misses GT deletion due to pseudogene homology; requires pseudogene-aware sequencing or specific NCF1 GT deletion PCR assay",
            "MILDER THAN CYBB: later onset, less severe Aspergillus pneumonia on average — but still requires triple lifelong prophylaxis; some cases diagnosed late (adolescent/adult)",
            "IDENTICAL PROPHYLAXIS REGIMEN TO CYBB-CGD: TMP-SMX + itraconazole + IFN-gamma — same treatment regardless of CGD subtype",
        ],
        "monitoring": [
            "DHR flow cytometry at diagnosis (markedly reduced, not zero — baseline for follow-up)",
            "NCF1 GT deletion PCR (pseudogene-aware) — confirm before WES negative report",
            "Pulmonary HRCT annually (Aspergillus surveillance; NCF1 milder but still significant risk)",
            "Itraconazole trough levels every 6 months (hepatotoxicity + efficacy monitoring)",
            "GI evaluation if diarrhoea/abdominal pain (NCF1 CGD GI granulomas similar to CYBB)",
            "HSCT assessment if 2+ major infections in 1 year despite prophylaxis",
        ],
        "key_ddx": [
            "CYBB-CGD-X1 (most common CGD; XLR; zero DHR burst; same infections; females only carriers; gene sequencing discriminates)",
            "CYBA-CGD-AR1 (AR like NCF1; cytochrome b558 absent rather than cytosol organiser; DHR also reduced; gene panel discriminates)",
            "NCF2-CGD-AR3 (rarest AR CGD; p67phox; AR; same DHR picture; gene panel discriminates)",
            "Myeloperoxidase deficiency (MPO) (reduced DHR but different mechanism; mostly asymptomatic; MPO stain confirms)",
        ],
        "phagocyte_pathway": "NADPH oxidase (p47phox cytosolic organiser) — cytosol-to-membrane translocation complex activation",
        "onset_age": "infant_to_childhood",
        "granuloma_risk": True,
        "fungal_risk": True,
        "bleeding_risk": False,
        "neurological_risk": False,
        "severity_options": [
            "Severe NCF1-CGD (homozygous GT deletion; complete p47phox absence; early onset; multiple hospitalisations)",
            "Moderate NCF1-CGD (compound heterozygous; partial residual function; later onset; responds to prophylaxis)",
        ],
        "complication_options": ["Aspergillus pneumonia", "Recurrent bacterial lymphadenitis", "GI granulomas", "Liver abscess", "Recurrent skin abscesses", "Osteomyelitis"],
        "dhr_options": ["markedly_reduced", "severely_reduced"],
        "treatments_used": ["TMP-SMX (daily)", "Itraconazole (daily)", "IFN-gamma 3x/week", "Voriconazole (Aspergillus)", "Prednisolone (granuloma)", "IV antibiotics (acute)"],
    },
    # -- CYBA — CGD-AR1 (p22phox) -------------------------------------------------------
    {
        "gene": "CYBA",
        "alt_name": (
            "CYBA (CYBA-195aa-16q24.2 / AR — CGD-AR1 — "
            "p22phox-Stabilises-gp91phox-Cytochrome-b558-Complex-Required — "
            "ABSENT-CYTOCHROME-b558-BOTH-gp91phox-AND-p22phox-ABSENT — "
            "AR-CGD-Similar-Phenotype-CYBB-Same-Triple-Prophylaxis)"
        ),
        "protein": (
            "CYBA -- 16q24.2 AR -- CYBA-195aa -- "
            "Cytochrome-b-245-Alpha-Chain-p22phox-22kDa-Transmembrane-Small-Subunit -- "
            "Partners-With-gp91phox-CYBB-To-Form-Cytochrome-b558-Heterodimer-Core-NADPH-Oxidase -- "
            "p22phox-Required-For-Stability-And-Membrane-Integration-Of-gp91phox -- "
            "CYBA-Deficiency-gp91phox-ALSO-Absent-Degraded-Cytochrome-b558-Complex-Lost -- "
            "Complete-Absence-Cytochrome-b558-Both-Subunits-Absent-In-CYBA-Deficiency -- "
            "CYBA-Provides-Docking-Site-For-Cytosolic-Components-p47phox-p67phox-Membrane -- "
            "AR-CGD-AR1-Clinically-Indistinguishable-From-CYBB-CGD-Same-Infections-Same-Rx -- "
            "OMIM-Gene-608508-Disease-CGD-AR1-233690"
        ),
        "locus": "16q24.2",
        "protein_size": "195 aa / 22 kDa",
        "inheritance": (
            "AR; biallelic CYBA mutations; "
            "consanguineous families enriched; "
            "mutations: missense, nonsense, splice, small insertions/deletions; "
            "consequence: p22phox absent → gp91phox (CYBB) also destabilised and absent → complete loss of cytochrome b558; "
            "DHR flow cytometry: zero oxidative burst (identical to CYBB-CGD); "
            "Western blot: both gp91phox AND p22phox absent (critical distinguishing feature from CYBB mutations where p22phox may still be detectable); "
            "~5-8% of all CGD"
        ),
        "phagocyte_category": "CGD type AR1 (CYBA/p22phox; cytochrome b558 small subunit; both subunits absent when CYBA mutated; ~5-8% of CGD)",
        "pathognomonic": (
            "DHR FLOW CYTOMETRY ZERO OXIDATIVE BURST (same as CYBB-CGD): no fluorescence shift; "
            "WESTERN BLOT — BOTH gp91phox AND p22phox ABSENT: key diagnostic finding; in CYBB mutations, p22phox detected but gp91phox absent; in CYBA mutations, BOTH absent (mutual destabilisation); "
            "IDENTICAL CLINICAL PRESENTATION TO CYBB-CGD: same catalase-positive organisms; same age of onset; same granuloma formation; "
            "GENE PANEL REQUIRED: DHR alone cannot distinguish CYBA-CGD from CYBB-CGD; Western blot pattern guides → gene panel confirms"
        ),
        "treatment": (
            "PROPHYLAXIS TRIPLE (IDENTICAL — MANDATORY LIFELONG): "
            "TMP-SMX + itraconazole + IFN-gamma (Actimmune 3x/week); "
            "ACUTE INFECTIONS: same as CYBB-CGD — broad-spectrum IV antibiotics; voriconazole for Aspergillus; "
            "GRANULOMA: corticosteroids — same management as CYBB; "
            "HSCT: curative — same as CYBB-CGD; consider for severe phenotype; "
            "WESTERN BLOT INTERPRETATION: pathologist must know to test both gp91phox AND p22phox bands; "
            "GENETIC COUNSELLING: AR — 25% recurrence; both parents obligate carriers; "
            "PROGNOSIS: identical to CYBB-CGD with equivalent severity"
        ),
        "key_features": [
            "CYTOCHROME b558 MUTUAL DESTABILISATION: p22phox (CYBA) is required to stabilise gp91phox (CYBB) — when CYBA is absent, BOTH subunits are degraded; Western blot shows both absent (unlike CYBB mutations where p22phox is present but gp91phox absent)",
            "DHR ZERO BURST: same zero oxidative burst as CYBB-CGD; cannot distinguish by DHR alone — Western blot pattern is the key differentiator before gene panel",
            "CLINICALLY INDISTINGUISHABLE FROM CYBB-CGD: same spectrum (Aspergillus, Staphylococcus, Burkholderia, granulomas); same prophylaxis; same HSCT indication",
            "CONSANGUINITY ENRICHED: AR inheritance — increased frequency in consanguineous populations (Middle East, South Asia); founder mutations described in specific populations",
            "GENE PANEL OVER WES ALONE: standard WES can detect CYBA mutations but Western blot pattern guides prioritisation; AR pattern on family history → test CYBA before assuming CYBB (CYBB is XLR)",
            "SAME TRIPLE PROPHYLAXIS REGIMEN: TMP-SMX + itraconazole + IFN-gamma — identical regardless of CYBB vs CYBA vs NCF1/NCF2",
        ],
        "monitoring": [
            "DHR at diagnosis + annually during clinical surveillance (baseline zero; any increase = consider gene correction or donor chimaerism post-HSCT)",
            "Western blot (gp91phox + p22phox) if diagnosis uncertain — confirms CYBA (both absent) vs CYBB (p22phox present only)",
            "HRCT chest annually (Aspergillus + granuloma); PFTs from age 6",
            "CBC + acute-phase reactants monthly × 6 months, 3-monthly thereafter",
            "Gastroenterology review if GI symptoms (identical GI granuloma risk to CYBB-CGD)",
            "Genetic counselling at diagnosis: cascade carrier testing of both parents + siblings",
        ],
        "key_ddx": [
            "CYBB-CGD-X1 (XLR; zero DHR; p22phox PRESENT but gp91phox absent on Western blot; males predominantly)",
            "NCF1-CGD-AR2 (AR; p47phox cytosolic; DHR markedly reduced not zero; both b558 subunits present on Western blot)",
            "NCF2-CGD-AR3 (AR; p67phox; DHR reduced; cytochrome b558 present; rarest AR CGD; gene panel)",
            "Glutathione peroxidase deficiency (GPx1) (very rare ROS pathway defect; CGD-like; specific enzyme assay)",
        ],
        "phagocyte_pathway": "NADPH oxidase (p22phox/cytochrome b558 small subunit) — stabilises gp91phox → mutual destabilisation when absent",
        "onset_age": "infant_to_early_childhood",
        "granuloma_risk": True,
        "fungal_risk": True,
        "bleeding_risk": False,
        "neurological_risk": False,
        "severity_options": [
            "Severe CYBA-CGD (biallelic null; complete cytochrome b558 absence; early onset; Aspergillus dominant)",
            "Moderate CYBA-CGD (hypomorphic allele; residual p22phox; milder clinical course)",
        ],
        "complication_options": ["Aspergillus pneumonia", "Liver abscess", "GI granulomas", "Perianal abscess", "Lymphadenitis", "Osteomyelitis"],
        "dhr_options": ["zero_burst", "absent"],
        "treatments_used": ["TMP-SMX (daily)", "Itraconazole (daily)", "IFN-gamma 3x/week", "Voriconazole (Aspergillus)", "Corticosteroids (granuloma)", "HSCT (curative)"],
    },
    # -- NCF2 — CGD-AR3 (p67phox) -------------------------------------------------------
    {
        "gene": "NCF2",
        "alt_name": (
            "NCF2 (NCF2-526aa-1q25.3 / AR — CGD-AR3 — "
            "RAREST-AR-CGD-p67phox-Activation-Domain-NOX2 — "
            "Cytochrome-b558-PRESENT-On-Western-Blot-KEY-DDx-CYBA-CYBB — "
            "AR-CGD-Same-Clinical-Profile-Same-Triple-Prophylaxis)"
        ),
        "protein": (
            "NCF2 -- 1q25.3 AR -- NCF2-526aa -- "
            "Neutrophil-Cytosol-Factor-2-p67phox-67kDa-Cytosolic-Regulatory-Subunit -- "
            "Activation-Domain-AD-Required-For-Electron-Transfer-From-NADPH-To-FAD-NOX2 -- "
            "Two-Tetratricopeptide-Repeat-TPR-Motifs-Bind-RAC-GTPase-Activated-Form -- "
            "NCF2-p67phox-p47phox-p40phox-Cytosolic-Heterotrimeric-Complex-Membrane-Translocation -- "
            "NCF2-Deficiency-No-Activation-Domain-NADPH-Oxidase-Assembly-Fails-No-O2-Burst -- "
            "Cytochrome-b558-gp91phox-p22phox-PRESENT-on-Western-Blot-KEY-DDx-CYBB-CYBA -- "
            "RAREST-AR-CGD-5pct-All-AR-CGD-More-Common-Middle-East-North-Africa -- "
            "OMIM-Gene-608515-Disease-CGD-AR3-233710"
        ),
        "locus": "1q25.3",
        "protein_size": "526 aa / 67 kDa",
        "inheritance": (
            "AR; biallelic NCF2 mutations; "
            "rarest AR CGD subtype (~2-5% of all CGD; ~5-8% of AR CGD); "
            "enriched in Middle Eastern and North African consanguineous pedigrees; "
            "mutations: missense, nonsense, splice — diverse; "
            "cytochrome b558 (gp91phox + p22phox) PRESENT but non-functional → key Western blot discriminator; "
            "DHR flow cytometry: zero or markedly reduced oxidative burst (p67phox activation domain absent); "
            "RAC-GTPase binding intact but electron transfer activation domain absent → complex assembly fails"
        ),
        "phagocyte_category": "CGD type AR3 (NCF2/p67phox; NADPH oxidase activation domain; rarest AR CGD ~2-5%)",
        "pathognomonic": (
            "WESTERN BLOT — CYTOCHROME b558 PRESENT (gp91phox + p22phox detectable): key discriminator from CYBB (p22phox absent) and CYBA (both absent); NCF2 deficiency = cytosolic component missing → membrane complex intact but non-functional; "
            "DHR FLOW CYTOMETRY ZERO/MARKEDLY REDUCED: same oxidative burst absence as other CGD subtypes; "
            "IDENTICAL CLINICAL PRESENTATION TO OTHER CGD SUBTYPES: Aspergillus, Staphylococcus, catalase-positive organisms; granuloma formation; "
            "GENE PANEL ESSENTIAL: NCF2-CGD cannot be distinguished clinically or by DHR from NCF1/CYBA/CYBB — requires Western blot pattern + gene sequencing"
        ),
        "treatment": (
            "PROPHYLAXIS TRIPLE (IDENTICAL — MANDATORY LIFELONG): "
            "TMP-SMX + itraconazole + IFN-gamma 3x/week; "
            "same acute and chronic management as CYBB/CYBA/NCF1; "
            "GENE PANEL: NCF2 often confirmed by next-generation sequencing panel (clinical + CYBB/NCF1/CYBA negative → test NCF2); "
            "HSCT: curative — same indication as other CGD subtypes; "
            "WESTERN BLOT PROTOCOL: always test gp91phox + p22phox + p47phox + p67phox to assign subtype; "
            "PROGNOSIS: similar to NCF1-CGD (milder than CYBB generally, but data limited by rarity)"
        ),
        "key_features": [
            "RAREST AR CGD: ~2-5% of all CGD; cytochrome b558 PRESENT (Western blot normal for gp91phox + p22phox) but complex non-functional (p67phox absent = activation domain missing)",
            "WESTERN BLOT PATTERN DIAGNOSTIC: gp91phox present + p22phox present + p67phox ABSENT = NCF2-CGD (distinguishes from CYBB where p22phox present only, and CYBA where both absent)",
            "RAC-GTPase INTERACTION: p67phox contains tetratricopeptide repeat (TPR) motifs that bind activated RAC-GTPase — NCF2 loss disconnects RAS signalling from NOX2 activation",
            "SAME CLINICAL SPECTRUM: Aspergillus pneumonia, bacterial liver abscesses, GI granulomas — identical to other CGD forms; same triple prophylaxis required",
            "MIDDLE EAST / NORTH AFRICA ENRICHMENT: consanguineous pedigrees in these regions disproportionately represent NCF2-CGD; index of suspicion higher in consanguineous AR CGD",
            "GENE PANEL ESSENTIAL: no clinical feature distinguishes NCF2-CGD from NCF1/CYBA/CYBB — Western blot + gene panel required for subtype assignment and genetic counselling",
        ],
        "monitoring": [
            "Western blot (gp91phox + p22phox + p47phox + p67phox) at diagnosis — assign CGD subtype",
            "DHR flow cytometry baseline at diagnosis; follow-up if clinical deterioration",
            "HRCT chest annually (same Aspergillus + granuloma surveillance as other CGD)",
            "Itraconazole trough + LFTs 6-monthly (standard antifungal monitoring)",
            "Genetic counselling: cascade testing of siblings (25% recurrence risk) + carrier parents",
            "HSCT evaluation if refractory infections (same criteria as CYBB-CGD)",
        ],
        "key_ddx": [
            "NCF1-CGD-AR2 (most common AR CGD; GT deletion; p47phox absent on Western blot; DHR markedly reduced; pseudogene pitfall)",
            "CYBA-CGD-AR1 (both gp91phox + p22phox absent on Western blot; NCF2 has both PRESENT — key discriminator)",
            "CYBB-CGD-X1 (XLR; gp91phox absent, p22phox present; males; DHR zero)",
            "NCF4 deficiency (p40phox) (extremely rare AR CGD variant; similar Western blot to NCF2 — p40phox absent; gene panel discriminates)",
        ],
        "phagocyte_pathway": "NADPH oxidase (p67phox activation domain) — cytosolic complex activation signal for electron transfer",
        "onset_age": "infant_to_childhood",
        "granuloma_risk": True,
        "fungal_risk": True,
        "bleeding_risk": False,
        "neurological_risk": False,
        "severity_options": [
            "Severe NCF2-CGD (null alleles; early onset; recurrent Aspergillus)",
            "Moderate NCF2-CGD (hypomorphic alleles; later onset; milder course than CYBB)",
        ],
        "complication_options": ["Aspergillus pneumonia", "Hepatic abscess", "Lymphadenitis suppurativa", "GI granulomas", "Serratia osteomyelitis", "Recurrent skin infection"],
        "dhr_options": ["zero_burst", "markedly_reduced"],
        "treatments_used": ["TMP-SMX (daily)", "Itraconazole (daily)", "IFN-gamma 3x/week", "Voriconazole (Aspergillus)", "IV meropenem (Burkholderia/Serratia)", "HSCT (curative)"],
    },
    # -- ITGB2 — LAD-I (CD18) -----------------------------------------------------------
    {
        "gene": "ITGB2",
        "alt_name": (
            "ITGB2 (ITGB2-769aa-21q22.3 / AR — LAD-I — "
            "MOST-COMMON-LAD-MORE-THAN-80pct — "
            "OMPHALITIS-DELAYED-CORD-SEPARATION->21-DAYS-LEUKOCYTOSIS-WITHOUT-PUS-PATHOGNOMONIC — "
            "CD18-ALL-Beta-2-Integrins-ABSENT-LFA-1-Mac-1-p150-95 — "
            "HSCT-Curative-Severe-<1pct-CD18)"
        ),
        "protein": (
            "ITGB2 -- 21q22.3 AR -- ITGB2-769aa -- "
            "Integrin-Subunit-Beta-2-CD18-95kDa-Beta-Subunit-All-Beta-2-Integrins -- "
            "CD18-Partners-CD11a-LFA-1-alphaL-beta2-T-Cell-Adhesion-Inflammation -- "
            "CD18-Partners-CD11b-Mac-1-alphaM-beta2-Phagocytosis-C3b-Opsonisation -- "
            "CD18-Partners-CD11c-p150-95-alphaX-beta2-Dendritic-Cell-Function -- "
            "ITGB2-Deficiency-ALL-Three-Beta-2-Integrins-LFA-1-Mac-1-p150-Absent -- "
            "Neutrophils-Cannot-Adhere-Endothelium-Cannot-Diapedesis-Cannot-Migrate-To-Infection -- "
            "Tissue-Infections-Proceed-Without-Neutrophil-Exudate-No-Pus-Formation-LEUKOCYTOSIS -- "
            "CORD-SEPARATION-Requires-Beta-2-Integrin-Mediated-Inflammation-Absent-Delay -- "
            "OMIM-Gene-600065-Disease-LAD-I-116920"
        ),
        "locus": "21q22.3",
        "protein_size": "769 aa / 95 kDa",
        "inheritance": (
            "AR; biallelic ITGB2 mutations; "
            "severe LAD-I: <1% CD18 surface expression → recurrent life-threatening infections; "
            "moderate LAD-I: 1-10% CD18 → less severe course; "
            "mild LAD-I: >10% CD18 → minimal symptoms detected incidentally; "
            "mutations: missense, nonsense, splice, deletions — diverse; no single founder; "
            "incidence: 1/million; more common in regions with higher consanguinity; "
            "ALL 3 beta-2 integrins (LFA-1/CD11a, Mac-1/CD11b, p150.95/CD11c) absent when CD18 absent; "
            "screening: flow cytometry (CD18 expression on neutrophils) — rapid diagnostic"
        ),
        "phagocyte_category": "LAD type I (ITGB2/CD18; beta-2 integrin common subunit; most common LAD >80%; neutrophil adhesion defect)",
        "pathognomonic": (
            "DELAYED UMBILICAL CORD SEPARATION (>21 days; normal <7-14 days): umbilical cord separation requires neutrophil-mediated inflammation → absent in LAD-I; early presentation of severe LAD-I; "
            "OMPHALITIS: umbilical stump infection (neonate) with minimal or no pus despite high leukocyte count; "
            "LEUKOCYTOSIS WITHOUT PUS AT INFECTION SITE: WBC 20-100 × 10⁹/L even at baseline; severe infection → WBC 50-200 × 10⁹/L; but infected tissues show NO neutrophil exudate (no pus); "
            "FLOW CYTOMETRY CD18 <1% (SEVERE): neutrophil CD18 expression <1% of normal on stimulated flow cytometry = diagnostic of severe LAD-I"
        ),
        "treatment": (
            "HSCT CURATIVE (DEFINITIVE TREATMENT FOR SEVERE LAD-I): "
            "allogeneic HSCT (HLA-matched sibling or MUD); "
            "best outcomes when transplanted before major organ damage (early in life); "
            "myeloablative or reduced-intensity conditioning; "
            "PROPHYLAXIS (BRIDGE TO HSCT / MODERATE LAD-I): "
            "TMP-SMX (anti-bacterial); fluconazole (antifungal); "
            "G-CSF: may transiently raise CD18-deficient neutrophil numbers (no effect on function); "
            "GRANULOCYTE TRANSFUSIONS: acute life-threatening infections → irradiated granulocytes (functional, CD18-expressing donor cells); "
            "GENE THERAPY: under investigation (lentiviral CD18 correction in haematopoietic stem cells); "
            "WOUND CARE: meticulous wound and skin/mucosal care; "
            "PERIODONTOLOGY: severe periodontitis in survivors (gingival neutrophil LAD = periodontitis)"
        ),
        "key_features": [
            "DELAYED CORD SEPARATION (>21 days) PATHOGNOMONIC: umbilical cord separation depends on neutrophil-mediated inflammation; LAD-I neutrophils cannot migrate → cord remains attached (neonate presenting sign); omphalitis often concurrent",
            "LEUKOCYTOSIS WITHOUT PUS: WBC >25 × 10⁹/L at baseline in severe LAD-I; infection → WBC 50-200 × 10⁹/L; but infected tissues show no neutrophil exudate (NO PUS) — neutrophils in blood cannot enter tissue; paradoxical sign is PATHOGNOMONIC",
            "CD18 FLOW CYTOMETRY: express CD18 on neutrophil surface < 1% = severe LAD-I; 1-10% = moderate; >10% = mild; rapid diagnostic test (same-day result); all 3 beta-2 integrins (LFA-1, Mac-1, p150.95) absent when CD18 absent",
            "SEVERE (<1% CD18): death in early childhood without HSCT (recurrent overwhelming bacterial infections — Staphylococcus, Pseudomonas, Klebsiella, Candida); moderate/mild (1-10% / >10%): survive longer, less severe infections",
            "PERIODONTITIS DISTINCTIVE COMPLICATION: in moderate/mild LAD-I survivors — severe early-onset destructive periodontitis (gingival neutrophil chemotaxis absent); tooth loss in childhood; intensive dental/periodontal management required",
            "HSCT CURATIVE FOR SEVERE LAD-I: allogeneic transplant corrects neutrophil CD18 expression; best outcome if transplanted early (before recurrent infections cause organ damage); gene therapy under development",
        ],
        "monitoring": [
            "CD18 flow cytometry (neutrophils + lymphocytes) at diagnosis — define severity (<1% severe, 1-10% moderate, >10% mild)",
            "CBC weekly (severe LAD-I) / monthly (moderate) — extreme leukocytosis at baseline; track WBC",
            "Wound surveillance: umbilical stump, skin, perianal, gingival — weekly nursing review (neonatal/infancy)",
            "Periodontal assessment from age 2: 6-monthly dental + periodontist review (prevent tooth loss)",
            "HSCT assessment urgently in severe LAD-I (<1% CD18): timing critical before cumulative organ damage",
            "Granulocyte crossmatch testing if granulocyte transfusion anticipated (prepare donor panel)",
        ],
        "key_ddx": [
            "FERMT3-LAD-III (LAD + BLEEDING DIATHESIS combination pathognomonic; CD18 reduced but not zero; platelet aggregation absent — Glanzmann-like; Kindlin-3 activates beta-2 AND beta-3 integrins)",
            "Severe CGD (CYBB/NCF1) (DHR zero; normal CD18 expression; different organism spectrum — catalase-positive; no cord delay or leukocytosis without pus)",
            "Chediak-Higashi (LYST) (partial albinism + giant granules neutrophils; accelerated phase HLH; CD18 normal; DHR reduced)",
            "Glycogen storage disease type Ib (SLC37A4) (neutropenia + GSD; GI symptoms; opposite to LAD — low, not high WBC; IBD-like)",
        ],
        "phagocyte_pathway": "Beta-2 integrin adhesion (CD18/ITGB2 common subunit) — neutrophil-endothelium adhesion → diapedesis → tissue migration",
        "onset_age": "neonatal_to_infant",
        "granuloma_risk": False,
        "fungal_risk": True,
        "bleeding_risk": False,
        "neurological_risk": False,
        "severity_options": [
            "Severe LAD-I (<1% CD18; death in infancy/early childhood without HSCT; omphalitis + sepsis)",
            "Moderate LAD-I (1-10% CD18; recurrent bacterial infections; survive to adulthood with care)",
            "Mild LAD-I (>10% CD18; minimal infections; periodontitis dominant; detected incidentally)",
        ],
        "complication_options": ["Delayed cord separation + omphalitis", "Recurrent bacterial sepsis", "Severe periodontitis", "Candida oesophagitis", "Perianal abscess without pus", "Persistent leukocytosis (50-200 ×10⁹/L)"],
        "dhr_options": ["normal_burst"],
        "treatments_used": ["HSCT (curative, severe)", "TMP-SMX prophylaxis", "Granulocyte transfusions (acute)", "Intensive periodontal care", "Wound care (antimicrobial dressings)", "Fluconazole (antifungal)"],
    },
    # -- FERMT3 — LAD-III (Kindlin-3) ---------------------------------------------------
    {
        "gene": "FERMT3",
        "alt_name": (
            "FERMT3 (FERMT3-667aa-11q13.1 / AR — LAD-III / Kindlin-3-Deficiency — "
            "LAD+GLANZMANN-LIKE-BLEEDING-PATHOGNOMONIC-COMBINATION — "
            "Kindlin-3-Activates-Beta-2-AND-Beta-3-Integrins-DUAL-DEFECT — "
            "HSCT-Curative-Both-Leukocyte-AND-Platelet-Function-Corrected)"
        ),
        "protein": (
            "FERMT3 -- 11q13.1 AR -- FERMT3-667aa -- "
            "Fermitin-Family-Member-3-Kindlin-3-74kDa-FERM-Domain-Integrin-Activator -- "
            "Expressed-Haematopoietic-Cells-Neutrophils-Platelets-T-B-Cells -- "
            "Kindlin-3-Binds-Beta-Integrin-Cytoplasmic-Tail-NPXY-Motif-Inside-Out-Signalling -- "
            "FERMT3-Required-Activation-BOTH-Beta-2-Integrins-Leukocytes-AND-Beta-3-Integrins-Platelets -- "
            "FERMT3-Deficiency-Beta-2-Integrins-Inactive-Leukocyte-Adhesion-Defect-LAD-III -- "
            "FERMT3-Deficiency-Beta-3-Integrins-Inactive-Platelet-GPIIb-GPIIIa-Failure-Bleeding -- "
            "CD18-Expression-REDUCED-NOT-ZERO-Key-DDx-ITGB2-LAD-I-CD18-Absent -- "
            "OMIM-Gene-607901-Disease-LAD-III-612840"
        ),
        "locus": "11q13.1",
        "protein_size": "667 aa / 74 kDa",
        "inheritance": (
            "AR; biallelic FERMT3 mutations; "
            "Kindlin-3 expressed exclusively in haematopoietic cells (neutrophils, platelets, lymphocytes, monocytes); "
            "FERMT3 deficiency affects ALL integrin-dependent haematopoietic functions; "
            "mutations: missense, nonsense — diverse; founder mutations described in Turkish, Lebanese, German populations; "
            "CD18 surface expression REDUCED (not absent) on flow cytometry — distinguishes from LAD-I (CD18 absent); "
            "platelet aggregation test: ABSENT with ADP/collagen/thrombin (Glanzmann-like) — GPIIb/GPIIIa expressed but inactive (inside-out signalling failure)"
        ),
        "phagocyte_category": "LAD type III (FERMT3/Kindlin-3; integrin inside-out signalling activator; LAD + Glanzmann-like platelet dysfunction; AR)",
        "pathognomonic": (
            "LAD FEATURES + GLANZMANN-LIKE BLEEDING DIATHESIS COMBINATION PATHOGNOMONIC: "
            "leukocyte adhesion defect (same signs as LAD-I: delayed cord separation, omphalitis, leukocytosis without pus) PLUS "
            "severe mucocutaneous bleeding (petechiae, epistaxis, gingival bleeding, GI bleeding) — Glanzmann thrombasthenia-like; "
            "CD18 REDUCED (NOT ABSENT): flow cytometry shows reduced CD18 expression (not zero as in LAD-I) — Kindlin-3 is needed for inside-out activation not surface expression; "
            "PLATELET AGGREGATION ABSENT: ADP, collagen, thrombin-induced aggregation all absent (GPIIb/GPIIIa expressed on surface but cannot be activated without Kindlin-3)"
        ),
        "treatment": (
            "HSCT CURATIVE: allogeneic HSCT corrects BOTH leukocyte adhesion defect AND platelet function (same Kindlin-3 deficiency in all haematopoietic cells — HSCT provides normal Kindlin-3+ cells); "
            "PRE-HSCT MANAGEMENT: "
            "PLATELET TRANSFUSIONS: for acute bleeding episodes (major surgery, trauma); "
            "RECOMBINANT FACTOR VIIa (rFVIIa): used for refractory bleeding (bypasses platelet aggregation requirement); "
            "TMP-SMX + antifungal PROPHYLAXIS: as per LAD-I; "
            "GRANULOCYTE TRANSFUSIONS: life-threatening infections; "
            "PLATELET CROSSMATCH: anti-HLA antibodies develop with repeated platelet transfusions — crossmatched donors required; "
            "NO ASPIRIN / NSAIDs: contraindicated (further impair platelet function); "
            "GENETIC COUNSELLING: AR — 25% recurrence"
        ),
        "key_features": [
            "DUAL INTEGRIN DEFECT PATHOGNOMONIC: Kindlin-3 activates BOTH beta-2 integrins (neutrophil/leukocyte adhesion) AND beta-3 integrins (platelet GPIIb/GPIIIa) — deficiency causes BOTH LAD features AND Glanzmann thrombasthenia-like bleeding simultaneously",
            "CD18 REDUCED (NOT ABSENT): distinguishes FERMT3-LAD-III from ITGB2-LAD-I; in LAD-III, CD18 is synthesised and expressed at reduced levels on the surface (inside-out activation failure, not protein absence)",
            "PLATELET AGGREGATION TEST KEY: ADP + collagen + thrombin-induced aggregation ALL ABSENT (Glanzmann thrombasthenia pattern) — GPIIb/GPIIIa expressed but not activated; ristocetin (VWF-dependent) NORMAL (distinguishes from VWD)",
            "BLEEDING DIATHESIS IN AN LAD PATIENT: any LAD patient with severe mucocutaneous bleeding (petechiae, epistaxis, gingival bleeding) → test FERMT3 (Kindlin-3) and platelet aggregation — this combination is diagnostic for LAD-III",
            "HSCT CORRECTS BOTH DEFECTS: allogeneic HSCT provides Kindlin-3+ haematopoietic cells → restores both neutrophil adhesion AND platelet activation — single treatment for dual phenotype",
            "NO ASPIRIN/NSAIDs EVER: patients with LAD-III have severe baseline platelet dysfunction; any additional platelet inhibition is life-threatening; contraindicated absolutely",
        ],
        "monitoring": [
            "CD18 flow cytometry (reduced, not absent — distinguish from LAD-I); platelet flow cytometry (GPIIb/GPIIIa expression normal; activation markers absent)",
            "Platelet aggregation testing at diagnosis (ADP/collagen/thrombin absent; ristocetin normal) — confirm Glanzmann-like platelet defect",
            "CBC weekly (severe): track leukocytosis + thrombocytopaenia (platelet count normal; function abnormal)",
            "Platelet antibody screen (HLA antibody; anti-GPIIb/GPIIIa) — develop with repeated platelet transfusions; crossmatched donors essential",
            "Bleeding assessment: ISTH bleeding score at each visit; frequency of platelet transfusions required",
            "HSCT evaluation: early discussion — timing before major haemorrhagic or infectious events",
        ],
        "key_ddx": [
            "ITGB2-LAD-I (CD18 ABSENT on flow cytometry; no bleeding diathesis; omphalitis; leukocytosis without pus; same LAD features but no platelet defect)",
            "Glanzmann Thrombasthenia (ITGA2B/ITGB3) (same platelet aggregation pattern; NO leukocyte adhesion defect; no omphalitis; normal CD18; ITGA2B/ITGB3 mutations)",
            "Kindlin-3 vs Talin-1: Talin-1 also activates integrins — Talin-1 mutations described (extremely rare) with similar dual defect; gene panel discriminates",
            "LYST/Chediak-Higashi (giant granules; albinism; HLH; different platelet defect — delta granule; different CD18; DHR reduced)",
        ],
        "phagocyte_pathway": "Integrin inside-out signalling (Kindlin-3/FERMT3) — activates beta-2 (leukocyte) AND beta-3 (platelet) integrins simultaneously",
        "onset_age": "neonatal_to_infant",
        "granuloma_risk": False,
        "fungal_risk": True,
        "bleeding_risk": True,
        "neurological_risk": False,
        "severity_options": [
            "Severe LAD-III (null FERMT3; complete adhesion + bleeding defect; early death without HSCT; petechiae + sepsis)",
            "Moderate LAD-III (hypomorphic alleles; partial function; survive infancy; recurrent bleeding + infections)",
        ],
        "complication_options": ["Severe mucocutaneous bleeding (petechiae/epistaxis)", "Delayed cord separation + omphalitis", "Intracranial haemorrhage", "Gastrointestinal bleeding", "Recurrent bacterial sepsis", "Platelet refractoriness (anti-HLA)"],
        "dhr_options": ["normal_burst"],
        "treatments_used": ["HSCT (curative — corrects both defects)", "Platelet transfusions (acute bleeding)", "Recombinant FVIIa (refractory bleeding)", "TMP-SMX prophylaxis", "Granulocyte transfusions (acute infection)", "Meticulous wound care"],
    },
    # -- ELANE — SCN1 / Cyclic Neutropenia -----------------------------------------------
    {
        "gene": "ELANE",
        "alt_name": (
            "ELANE (ELANE-256aa-19p13.3 / AD — SCN1-Severe-Congenital-Neutropenia-1 / Cyclic-Neutropenia — "
            "ELANE-Misfolding-ER-Stress-UPR-Neutrophil-Apoptosis-Promyelocyte-Stage — "
            "21-DAY-CYCLE-ANC-NADIR-ORAL-ULCERS-FEVER-PATHOGNOMONIC-Cyclic-Neutropenia — "
            "G-CSF-FILGRASTIM-LIFELONG-FIRST-LINE-Raises-ANC-Above-1000 — "
            "MDS-AML-Transformation-Risk-Annual-BM-Biopsy-Mandatory)"
        ),
        "protein": (
            "ELANE -- 19p13.3 AD -- ELANE-256aa -- "
            "Elastase-Neutrophil-Expressed-Neutrophil-Elastase-NE-29kDa-Serine-Protease -- "
            "Stored-Azurophil-Granules-Primary-Granules-Neutrophils-Bactericidal-Degradation -- "
            "ELANE-Mutations-Cause-Misfolded-NE-Protein-ER-Retention-Unfolded-Protein-Response-UPR -- "
            "UPR-Caspase-Activation-Neutrophil-Apoptosis-At-Promyelocyte-Myelocyte-Stage -- "
            "SCN1-Sustained-Severe-ANC-<200-Cells-µL-Neonatal-Or-Infantile-Onset-No-Cycling -- "
            "Cyclic-Neutropenia-CyN-21-Day-Cycle-ANC-<200-3-5-Days-Then-Normal-Periodic -- "
            "G-CSF-Corrects-ANC->1000-Dramatically-Reduces-Infection-Risk-Lifelong -- "
            "G-CSF-Failure-OR-RUNX1-CSF3R-Mutation-MDS-AML-Transformation-Risk-Annual-BM -- "
            "OMIM-Gene-130130-Disease-SCN1-202700-CyN-162800"
        ),
        "locus": "19p13.3",
        "protein_size": "256 aa / 29 kDa",
        "inheritance": (
            "AD (autosomal dominant); "
            "ELANE mutations cause misfolding of neutrophil elastase (NE) → ER stress → UPR → caspase activation → neutrophil apoptosis at promyelocyte stage; "
            "de novo mutations common (sporadic SCN1/CyN cases); "
            "SCN1: ANC persistently <200 cells/µL (severe, constant neutropenia; azurophil granule loss); "
            "Cyclic Neutropenia (CyN): same ELANE mutations but different phenotype — ANC cycles every ~21 days (nadir <200 for 3-5 days, then recovers); "
            "genotype-phenotype correlation: some mutations preferentially cause SCN1 vs CyN (e.g. p.Gly214Arg → CyN; p.Cys151Tyr → severe SCN1); "
            "secondary mutations in CSF3R (G-CSF receptor) → risk of MDS/AML transformation"
        ),
        "phagocyte_category": "SCN type 1 / Cyclic Neutropenia (ELANE/NE; serine protease misfolding; ER stress → neutrophil apoptosis; AD dominant negative)",
        "pathognomonic": (
            "21-DAY CYCLING PATTERN (Cyclic Neutropenia): ANC nadir <200 cells/µL lasting 3-5 days every 21 days ± 7 days; serial ANC monitoring (3x/week for 6-8 weeks) to demonstrate cycle; "
            "ORAL ULCERS DURING NADIR: aphthous-like oral ulcers appearing at ANC nadir → heals with ANC recovery (cyclic aphthous ulcers = clinical signature); "
            "FEVER AT ANC NADIR: cyclic fevers coinciding with neutropenia nadir; "
            "BONE MARROW — MATURATION ARREST AT PROMYELOCYTE: BM biopsy shows abundant promyelocytes/myelocytes but absent mature neutrophils (myeloid maturation arrest); "
            "G-CSF RESPONSE: ANC rises above 1000 within 24-48 hours of G-CSF = highly characteristic (discriminates from other SCN causes)"
        ),
        "treatment": (
            "G-CSF (FILGRASTIM) LIFELONG — FIRST LINE: "
            "daily SQ filgrastim (3-5 µg/kg/day initially; titrate to ANC >1000); "
            "dramatically reduces infection rate (>90% reduction in severe infections vs untreated); "
            "SCN1: continuous daily G-CSF required; "
            "Cyclic Neutropenia: G-CSF flattens cycle (prevents nadirs); can give every-other-day in mild CyN; "
            "TARGET: maintain ANC >500-1000 (minimum threshold for infection protection); "
            "ANTIBIOTIC PROPHYLAXIS: TMP-SMX during active neutropenic phase (cyclic) or continuous (SCN1); "
            "HSCT: for G-CSF non-responders (ANC <1000 despite >20µg/kg/day) + MDS/AML transformation; "
            "BM MONITORING (MANDATORY ANNUAL): RUNX1/CSF3R mutation panel + BM biopsy — detect transformation early; "
            "DENTAL CARE: aggressive periodontal management; G-CSF dramatically reduces gingival disease"
        ),
        "key_features": [
            "21-DAY CYCLE PATHOGNOMONIC (Cyclic Neutropenia): serial ANC (3× weekly for 6-8 weeks) demonstrates cycle; oral ulcers at nadir + recovery with ANC rise; febrile episodes at nadir; cycle period 21 ± 7 days in most ELANE-CyN",
            "ER STRESS / UPR MECHANISM: ELANE mutations → misfolded NE protein → retained in ER → UPR activates → caspase-mediated neutrophil apoptosis at promyelocyte stage; all mature neutrophil lineage downstream absent",
            "G-CSF RESPONSE DISTINGUISHES ELANE-SCN FROM OTHER SCN: ANC rises >1000 within 24-48 hours; most ELANE-SCN1 respond to 5-10 µg/kg/day; G-CSF failure suggests alternative diagnosis or acquired G-CSF receptor mutation",
            "MDS/AML TRANSFORMATION RISK: cumulative 10-20% lifetime risk in SCN1 (less in CyN); G-CSF receptor CSF3R mutations are pre-malignant events; RUNX1 mutations = highest risk; annual BM biopsy + CSF3R/RUNX1 panel mandatory",
            "SCN1 vs CYCLIC NEUTROPENIA: both ELANE mutations; SCN1 = constant ANC <200 (severe); CyN = cycling ANC (nadir <200 for 3-5 days); cycle period 21 ± 7 days; both respond to G-CSF but CyN less severely affected overall",
            "BONE MARROW MATURATION ARREST AT PROMYELOCYTE: BM biopsy shows abundant early myeloid precursors (promyelocytes/myelocytes) but absent mature neutrophils — myeloid maturation arrest at specific stage; distinguishes from aplastic anaemia (empty marrow) and myelodysplasia",
        ],
        "monitoring": [
            "Serial ANC (3× weekly for 6-8 weeks): demonstrate 21-day cycle (Cyclic Neutropenia) OR sustained <200 (SCN1); essential for diagnosis",
            "Annual BM biopsy + cytogenetics + CSF3R sequencing + RUNX1 mutation panel (MDS/AML transformation surveillance — mandatory for all SCN1 patients on G-CSF)",
            "G-CSF dose titration: maintain ANC 1000-2000; CBC weekly until stable dose; monthly thereafter",
            "Periodontal assessment 6-monthly (severe gingival disease with neutropenia; G-CSF improves but does not normalise)",
            "Growth monitoring (G-CSF in children — rare splenic enlargement/splenomegaly from G-CSF)",
            "LDH + uric acid (bone marrow hyper-cellularity on G-CSF; monitor transformation markers)",
        ],
        "key_ddx": [
            "HAX1-SCN3 / Kostmann disease (AR; neurological involvement with isoform B loss; same G-CSF response; no cycling; BM maturation arrest similar; HAX1 gene; consanguineous families)",
            "Autoimmune neutropenia (AIN) (acquired; anti-neutrophil antibodies; usually resolves by age 3-4; ANA screen; bone marrow hyperplastic not arrested)",
            "Reticular dysgenesis (AK2 mutations) (most severe; AR; absent all granulocytes AND lymphocytes (SCID+SCN); sensorineural hearing loss; HSCT mandatory)",
            "Cyclic haematopoiesis differential: VPS13B mutations (Cohen syndrome); Barth syndrome (TAZ/G4.5); all non-cycling — history + genetics discriminate",
        ],
        "phagocyte_pathway": "Granulopoiesis (ELANE/NE serine protease) — misfolding → ER stress → UPR → neutrophil maturation arrest at promyelocyte",
        "onset_age": "neonatal_to_infant",
        "granuloma_risk": False,
        "fungal_risk": True,
        "bleeding_risk": False,
        "neurological_risk": False,
        "severity_options": [
            "SCN1-severe (persistent ANC <200; neonatal onset; recurrent bacterial infections; omphalitis)",
            "Cyclic Neutropenia (21-day cycle; oral ulcers + fever at nadir; otherwise well between cycles; G-CSF effective)",
        ],
        "complication_options": ["Bacterial pneumonia at ANC nadir", "Oral ulcers (cyclic)", "Septicaemia (Gram-negative)", "Severe periodontitis", "MDS/AML transformation", "Stomatitis"],
        "dhr_options": ["normal_burst"],
        "treatments_used": ["G-CSF filgrastim (daily lifelong)", "TMP-SMX (antibiotic during nadir)", "HSCT (G-CSF failure / MDS)", "Annual BM biopsy (monitoring)", "Intensive dental care", "IV antibiotics (acute infections)"],
    },
    # -- HAX1 — SCN3 / Kostmann Disease -------------------------------------------------
    {
        "gene": "HAX1",
        "alt_name": (
            "HAX1 (HAX1-279aa-1q21.3 / AR — SCN3-Kostmann-Disease — "
            "NEUROLOGICAL-INVOLVEMENT-EPILEPSY-COGNITIVE-ISOFORM-B-PATHOGNOMONIC — "
            "ORIGINAL-1956-KOSTMANN-SWEDISH-PEDIGREE-FIRST-SCN-EVER-DESCRIBED — "
            "G-CSF-RESPONSIVE-SAME-As-ELANE-SCN1 — "
            "HSCT-Curative-G-CSF-Failure-MDS)"
        ),
        "protein": (
            "HAX1 -- 1q21.3 AR -- HAX1-279aa -- "
            "HCLS1-Associated-Protein-X-1-31kDa-Anti-Apoptotic-Mitochondrial-Protein -- "
            "HAX1-Localises-Mitochondria-Inner-Membrane-Stabilises-Mitochondrial-Membrane-Potential -- "
            "HAX1-Sequesters-PARL-Presenilin-Protease-Prevents-HtrA2-Serine-Protease-Release -- "
            "HAX1-Deficiency-HtrA2-Released-Mitochondria-Caspase-9-Activation-Granulocyte-Apoptosis -- "
            "Isoform-A-Ubiquitous-Isoform-B-Expressed-Brain-Neurons-Additional-HAX1-Isoform -- "
            "ISOFORM-B-Loss-NEUROLOGICAL-FEATURES-Epilepsy-Intellectual-Disability-UNIQUE-HAX1-SCN3 -- "
            "Homozygous-c.256C>T-p.Gln86Ter-FOUNDER-Swedish-Kostmann-Pedigree-1956 -- "
            "OMIM-Gene-605998-Disease-SCN3-Kostmann-610738"
        ),
        "locus": "1q21.3",
        "protein_size": "279 aa / 31 kDa",
        "inheritance": (
            "AR; biallelic HAX1 mutations; "
            "original Kostmann 1956 pedigree: large Swedish consanguineous family; founder mutation c.256C>T (p.Gln86Ter) in HAX1 exon 3; "
            "Kostmann originally described without molecular diagnosis; HAX1 gene identified 2007 (Klein et al.); "
            "HAX1 has multiple isoforms: isoform A (all tissues) + isoform B (brain/neurons); "
            "mutations truncating ONLY isoform A → SCN3 without neurological features; "
            "mutations truncating BOTH isoform A AND B → SCN3 with epilepsy/intellectual disability; "
            "founder mutation p.Gln86Ter (exon 3) truncates both isoforms → neurological involvement; "
            "identical BM picture to ELANE-SCN1: promyelocyte arrest; same G-CSF response; distinguished by AR inheritance + neurological features + gene panel"
        ),
        "phagocyte_category": "SCN type 3 / Kostmann disease (HAX1; anti-apoptotic mitochondrial protein; AR; original Kostmann pedigree 1956; neurological isoform B)",
        "pathognomonic": (
            "NEUROLOGICAL INVOLVEMENT (EPILEPSY + INTELLECTUAL DISABILITY) IN AR SCN PATIENT: epilepsy + cognitive impairment in an SCN patient = HAX1 isoform B truncation PATHOGNOMONIC; not seen in ELANE-SCN1 (AD) or WAS-related SCN; "
            "SWEDISH KOSTMANN PEDIGREE (HISTORICAL): original 1956 Swedish consanguineous pedigree (Rolf Kostmann description) — first SCN ever described; c.256C>T founder; "
            "SAME BM PICTURE AS ELANE-SCN1: promyelocyte arrest; ANC <200 (sustained, no cycling); "
            "G-CSF RESPONSE: identical to ELANE-SCN1 — ANC rises promptly on filgrastim; "
            "AR INHERITANCE IN CONSANGUINEOUS FAMILY: distinguishes from ELANE-SCN1 (AD) at pedigree level"
        ),
        "treatment": (
            "G-CSF (FILGRASTIM) LIFELONG — FIRST LINE (same as ELANE-SCN1): "
            "daily SQ filgrastim; target ANC >1000; "
            "most patients respond to G-CSF (G-CSF resistance + MDS risk similar to ELANE-SCN1); "
            "NEUROLOGICAL MANAGEMENT: "
            "ANTI-EPILEPTICS: for epilepsy (valproate, levetiracetam); "
            "NEURODEVELOPMENTAL SUPPORT: cognitive/educational support; physiotherapy; speech therapy; "
            "HSCT: curative for SCN3 (corrects neutropenia — does NOT correct neurological features if isoform B loss established); "
            "HSCT TIMING: before major infection complications; neurological features persist post-HSCT (brain cells not replaced by HSCT); "
            "ANNUAL BM BIOPSY: CSF3R + RUNX1 mutation panel (same MDS/AML risk as ELANE-SCN1); "
            "GENETIC COUNSELLING: AR — 25% recurrence; isoform B truncation = neurological risk in offspring"
        ),
        "key_features": [
            "ORIGINAL KOSTMANN 1956 PEDIGREE: first SCN ever described (Rolf Kostmann, Swedish consanguineous family); HAX1 gene identified 2007 — 51 years after clinical description; c.256C>T (p.Gln86Ter) founder mutation",
            "NEUROLOGICAL INVOLVEMENT (ISOFORM B LOSS) PATHOGNOMONIC: epilepsy + intellectual disability in SCN patient = HAX1 bilateral isoform A+B truncation — unique among SCN genes; HAX1 isoform B expressed in neurons; not seen in ELANE, G6PC3, or WAS-related SCN",
            "HAX1 MITOCHONDRIAL ANTI-APOPTOSIS: HAX1 stabilises mitochondrial membrane potential → sequesters PARL protease → prevents HtrA2 release → blocks caspase-9 activation → neutrophil survival; HAX1 absence → massive neutrophil apoptosis at promyelocyte stage (same BM picture as ELANE)",
            "G-CSF RESPONSE IDENTICAL TO ELANE-SCN1: ANC rises promptly on filgrastim; effective; lifelong therapy required; G-CSF resistance + MDS transformation risk similar to ELANE-SCN1 (annual BM biopsy mandatory)",
            "AR vs AD DISTINCTION FROM ELANE: HAX1-SCN3 is AR (consanguineous pedigrees); ELANE-SCN1 is AD (de novo or familial); pedigree analysis alone gives important clue; gene panel confirms",
            "HSCT CORRECTS NEUTROPENIA BUT NOT NEUROLOGICAL FEATURES: HSCT replaces haematopoietic stem cells → normal neutrophils; but established brain neurological features (epilepsy/cognitive) persist post-HSCT as neurons are not replaced; HSCT should be timed before major infectious damage",
        ],
        "monitoring": [
            "Serial ANC (3× weekly initially) — confirm sustained ANC <200 (not cyclic); distinguish from Cyclic Neutropenia",
            "Neurological assessment: EEG (epilepsy; HAX1 isoform B mutations); cognitive developmental testing annually",
            "G-CSF dose titration to ANC >1000; CBC monthly on stable dose",
            "Annual BM biopsy + CSF3R + RUNX1 sequencing (MDS/AML risk same as ELANE-SCN1)",
            "Periodontal assessment 6-monthly; dental surveillance (severe periodontitis with neutropenia)",
            "HAX1 isoform analysis (which isoforms truncated) — guides neurological prognosis; inform family of neurological risk",
        ],
        "key_ddx": [
            "ELANE-SCN1 (AD; same BM picture; same G-CSF response; NO neurological features; de novo or familial AD; cycling = CyN variant; gene panel discriminates)",
            "G6PC3-SCN (AR; glycogen storage type Ib-related neutropenia; GI symptoms + neutropenia; congenital heart defects; structural malformations; distinct)",
            "Reticular dysgenesis (AK2) (AR; SCID + SCN; sensorineural hearing loss; absent lymphocytes; HSCT only curative; distinct profile)",
            "WAS protein-related SCN (WIPF1 mutations) (Wiskott-Aldrich spectrum + neutropenia; X-linked; thrombocytopaenia + eczema + immunodeficiency; combined immunodeficiency picture)",
        ],
        "phagocyte_pathway": "Granulopoiesis (HAX1 mitochondrial anti-apoptosis) — HAX1 deficiency → HtrA2 release → caspase-9 → neutrophil apoptosis at promyelocyte",
        "onset_age": "neonatal_to_infant",
        "granuloma_risk": False,
        "fungal_risk": True,
        "bleeding_risk": False,
        "neurological_risk": True,
        "severity_options": [
            "SCN3 with neurological features (HAX1 isoform A+B truncated; epilepsy + cognitive impairment + severe neutropenia)",
            "SCN3 without neurological features (HAX1 isoform A truncated only; isolated severe neutropenia; no epilepsy)",
        ],
        "complication_options": ["Septicaemia (Gram-negative)", "Epilepsy (isoform B)", "Intellectual disability", "Severe periodontitis", "MDS/AML transformation", "Recurrent bacterial pneumonia"],
        "dhr_options": ["normal_burst"],
        "treatments_used": ["G-CSF filgrastim (daily lifelong)", "Anti-epileptics (valproate/levetiracetam)", "HSCT (G-CSF failure / MDS)", "Annual BM biopsy (monitoring)", "Neurodevelopmental support", "IV antibiotics (acute)"],
    },
]


# ---------------------------------------------------------------------------
# Patient cohort generator
# ---------------------------------------------------------------------------
def _build_cohort() -> list:
    cohort = []
    for idx, entry in enumerate(PHAGOCYTE_GENES):
        rng = random.Random(SEED_BASE + idx)
        gene = entry["gene"]
        n = 40
        for i in range(n):
            sev = rng.choice(entry["severity_options"])
            comp = rng.sample(entry["complication_options"], k=min(rng.randint(1, 3), len(entry["complication_options"])))
            dhr = rng.choice(entry["dhr_options"])
            treat = rng.sample(entry["treatments_used"], k=min(rng.randint(2, 4), len(entry["treatments_used"])))
            age_yr = round(rng.uniform(0.2, 20.0), 1)
            onset_yr = round(rng.uniform(0.0, min(age_yr, 2.0)), 2)
            delay_mo = max(0, round((age_yr - onset_yr) * 12 - rng.uniform(3, 24), 1))
            anc_nadir = round(rng.uniform(0, 150)) if gene in ("ELANE", "HAX1") else round(rng.uniform(500, 1200)) if gene in ("ITGB2", "FERMT3") else round(rng.uniform(0, 200))
            patient = {
                "patient_id": f"{gene}-{SEED_BASE + idx}-{i + 1:03d}",
                "gene": gene,
                "age_at_assessment_yr": age_yr,
                "age_at_symptom_onset_yr": onset_yr,
                "diagnosis_delay_months": delay_mo,
                "severity_label": sev.split("(")[0].strip(),
                "complications": comp,
                "dhr_result": dhr,
                "treatments_used": treat,
                "granuloma_present": entry["granuloma_risk"] and rng.random() < 0.40,
                "fungal_infection_present": entry["fungal_risk"] and rng.random() < 0.35,
                "bleeding_present": entry["bleeding_risk"] and rng.random() < 0.75,
                "neurological_present": entry["neurological_risk"] and rng.random() < 0.60,
                "anc_nadir_cells_ul": anc_nadir,
                "phagocyte_pathway": entry["phagocyte_pathway"],
            }
            cohort.append(patient)
    return cohort


# ---------------------------------------------------------------------------
# API data generators
# ---------------------------------------------------------------------------
def generate_overview() -> dict:
    cohort = _build_cohort()
    gene_counts = {}
    cgd_genes = []
    lad_genes = []
    scn_genes = []
    fungal_risk_genes = []
    neurological_genes = []
    bleeding_genes = []
    gene_summary = {}

    for entry in PHAGOCYTE_GENES:
        g = entry["gene"]
        pts = [p for p in cohort if p["gene"] == g]
        gene_counts[g] = len(pts)

        cat = entry["phagocyte_category"].lower()
        if "cgd" in cat:
            cgd_genes.append(g)
        elif "lad" in cat:
            lad_genes.append(g)
        elif "scn" in cat or "kostmann" in cat:
            scn_genes.append(g)

        if entry["fungal_risk"]:
            fungal_risk_genes.append(g)
        if entry["neurological_risk"]:
            neurological_genes.append(g)
        if entry["bleeding_risk"]:
            bleeding_genes.append(g)

        gene_summary[g] = {
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "phagocyte_category": entry["phagocyte_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "phagocyte_pathway": entry["phagocyte_pathway"],
            "granuloma_risk": entry["granuloma_risk"],
            "fungal_risk": entry["fungal_risk"],
            "bleeding_risk": entry["bleeding_risk"],
            "neurological_risk": entry["neurological_risk"],
            "onset_age": entry["onset_age"],
        }

    return {
        "title": "Hereditary-Phagocyte-Deficiency-Atlas — Complete 8-Gene Hereditary Phagocyte Deficiency Atlas",
        "n_genes": len(PHAGOCYTE_GENES),
        "n_patients": len(cohort),
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "disease_categories": {
            "CGD (Chronic Granulomatous Disease)": ["CYBB", "NCF1", "CYBA", "NCF2"],
            "LAD (Leukocyte Adhesion Deficiency)": ["ITGB2", "FERMT3"],
            "SCN (Severe Congenital Neutropenia)": ["ELANE", "HAX1"],
        },
        "inheritance_map": {
            "CYBB": "XLR", "NCF1": "AR", "CYBA": "AR", "NCF2": "AR",
            "ITGB2": "AR", "FERMT3": "AR", "ELANE": "AD", "HAX1": "AR",
        },
        "phagocyte_pathway_map": {
            "CYBB": "NADPH oxidase — gp91phox core (cytochrome b558 large subunit)",
            "NCF1": "NADPH oxidase — p47phox cytosolic organiser",
            "CYBA": "NADPH oxidase — p22phox (cytochrome b558 small subunit; mutual destabilisation)",
            "NCF2": "NADPH oxidase — p67phox activation domain",
            "ITGB2": "Beta-2 integrin adhesion (CD18 common subunit) — neutrophil migration",
            "FERMT3": "Integrin inside-out signalling (Kindlin-3) — beta-2 AND beta-3 integrins",
            "ELANE": "Granulopoiesis (NE serine protease misfolding → ER stress → apoptosis)",
            "HAX1": "Granulopoiesis (mitochondrial anti-apoptosis → HtrA2 → caspase-9)",
        },
        "key_clinical_pearls": [
            "DHR FLOW CYTOMETRY — UNIVERSAL CGD SCREEN: Dihydrorhodamine 123 flow cytometry with PMA stimulation = gold standard CGD diagnosis; ZERO fluorescence shift (males) or bimodal (female carriers) = CGD; CYBB-CGD = zero burst; NCF1-CGD = markedly reduced (not zero); CYBA/NCF2 = zero or near-zero; DHR must be done BEFORE diagnosing 'recurrent Staphylococcus aureus' without source",
            "CATALASE-POSITIVE ORGANISMS UNIQUELY DANGEROUS IN CGD: Staphylococcus aureus (liver abscesses), Aspergillus fumigatus (lung cavities — most common fungal killer), Serratia marcescens, Burkholderia cepacia (lung), Nocardia, Chromobacterium — all produce catalase (destroys H2O2); CGD neutrophils have no alternative killing mechanism; triple prophylaxis MANDATORY: TMP-SMX + itraconazole + IFN-gamma LIFELONG",
            "DELAYED CORD SEPARATION (>21 days) = IMMEDIATE LAD-I WORKUP: umbilical cord normally separates by 7-14 days via neutrophil-mediated inflammation; delay >21 days = leukocyte adhesion defect until proven otherwise; order CD18 flow cytometry (same day); LAD-I (ITGB2) = CD18 absent; LAD-III (FERMT3) = CD18 reduced + platelet aggregation absent; leukocytosis >25 × 10⁹/L + no pus = pathognomonic combination",
            "SERIAL ANC (3× WEEKLY) FOR 6-8 WEEKS = CYCLIC NEUTROPENIA DIAGNOSIS: oral ulcers + fever recurring every 3 weeks = Cyclic Neutropenia until proven otherwise; serial ANC demonstrates 21-day cycle with nadir <200; same-day ANC at presentation misses the cycle; G-CSF (filgrastim) corrects cycle AND lifts nadir; ELANE sequencing confirms",
            "HAX1 NEUROLOGICAL INVOLVEMENT DISTINGUISHES FROM ELANE-SCN: AR SCN (consanguineous family) + epilepsy + cognitive impairment = HAX1 isoform B truncation PATHOGNOMONIC; ELANE-SCN1 (AD) has no neurological features; neurological features persist post-HSCT (neurons not replaced); HAX1 gene panel required in any AR SCN with neurological findings",
            "MDS/AML TRANSFORMATION IN SCN1/SCN3 ON G-CSF: G-CSF use in SCN carries 10-20% lifetime MDS/AML risk; G-CSF receptor (CSF3R) gain-of-function mutations are pre-malignant events; RUNX1 mutations signal impending transformation; annual BM biopsy + CSF3R/RUNX1 sequencing MANDATORY in all SCN patients on G-CSF — never omit this surveillance",
            "FERMT3-LAD-III = ONLY LAD WITH BLEEDING: LAD + Glanzmann-like bleeding (petechiae + absent platelet aggregation + normal GPIIb/GPIIIa expression) = Kindlin-3 deficiency; no other phagocyte disorder combines these features; HSCT corrects BOTH leukocyte AND platelet defects (Kindlin-3 restored in all haematopoietic lineages); NO aspirin/NSAIDs ever",
        ],
        "clinical_emergency_flags": [
            "ASPERGILLUS PNEUMONIA IN CGD — LIFE-THREATENING: any CGD patient with fever + new pulmonary infiltrate → CT chest (tree-in-bud nodules + consolidation = Aspergillus); immediate voriconazole IV; galactomannan serum + BAL; surgical consultation if cavitating/large lesion; do not wait for culture (Aspergillus grows slowly); delay = death; consider HSCT if recurrent Aspergillus",
            "OMPHALITIS + LEUKOCYTOSIS >50 × 10⁹/L IN NEONATE: omphalitis (neonate) + extreme leukocytosis WITHOUT pus at infection site = LAD-I (ITGB2) or LAD-III (FERMT3); check CD18 flow cytometry same day; also examine platelet aggregation (LAD-III bleeding); if LAD-I severe (<1% CD18): urgent HSCT assessment; granulocyte transfusions for acute sepsis",
            "CGD PATIENT WITH GI OBSTRUCTION: CGD causes granulomatous colitis (Crohn-like) and bladder outlet obstruction via granuloma; abdominal CT if obstructive symptoms; high-dose corticosteroids (prednisolone 1mg/kg) = highly effective for GI/bladder granuloma; avoid surgery if responsive to steroids; itraconazole must continue during steroid course",
            "SCN PATIENT FEVER WITH ANC <200: any SCN patient on G-CSF with fever → immediate broad-spectrum antibiotics (gram-negative cover essential: Pseudomonas, E. coli, Klebsiella); blood cultures × 2; add G-CSF dose acutely; escalate to antifungals (voriconazole/caspofungin) if no response at 48h; neutrophil-dose escalation target ANC >500",
            "FERMT3-LAD-III INTRACRANIAL HAEMORRHAGE: LAD-III patients with severe platelet dysfunction → ICH risk (especially with trauma or fever); any neurological change = urgent head CT; platelet transfusions + recombinant FVIIa (rFVIIa); neurosurgical consultation; HSCT urgently; no aspirin/NSAIDs ever",
        ],
        "gene_summary": gene_summary,
        "cgd_genes": cgd_genes,
        "lad_genes": lad_genes,
        "scn_genes": scn_genes,
        "fungal_risk_genes": fungal_risk_genes,
        "neurological_genes": neurological_genes,
        "bleeding_genes": bleeding_genes,
        "diagnostic_algorithm": [
            "Step 1 — Suspect Phagocyte Deficiency: recurrent catalase-positive bacterial/fungal infections (CGD), omphalitis + leukocytosis without pus (LAD), or cyclic fevers/ulcers + severe neutropenia (SCN); consider in any child with recurrent life-threatening infections",
            "Step 2 — Immediate screening tests: (a) DHR flow cytometry with PMA — if abnormal → CGD workup; (b) CD18 flow cytometry on neutrophils — if absent/reduced → LAD workup; (c) serial ANC (3× weekly for 6-8 weeks) — 21-day cycle → ELANE; sustained <200 → HAX1 or ELANE-SCN1",
            "Step 3 — CGD subtype (if DHR abnormal): Western blot (gp91phox + p22phox + p47phox + p67phox); CYBB: p22phox present, gp91phox absent; CYBA: both absent; NCF1/NCF2: b558 present, cytosolic component absent; XLR vs AR family history; gene panel (CYBB/CYBA/NCF1/NCF2)",
            "Step 4 — LAD subtype (if CD18 abnormal): CD18 absent = ITGB2 (LAD-I); CD18 reduced = FERMT3 (LAD-III) — then check platelet aggregation (ADP/collagen/thrombin absent in LAD-III); ITGB2 gene vs FERMT3 gene; severity by CD18 expression level (<1% severe, 1-10% moderate)",
            "Step 5 — SCN diagnosis: BM biopsy (promyelocyte maturation arrest); G-CSF trial (response confirms SCN); ELANE sequencing (AD — SCN1 or Cyclic Neutropenia); HAX1 sequencing (AR + neurological → Kostmann); G6PC3, WAS protein panel if ELANE/HAX1 negative",
            "Step 6 — Initiate treatment: CGD → triple prophylaxis (TMP-SMX + itraconazole + IFN-gamma lifelong) + HSCT evaluation; Severe LAD-I/LAD-III → urgent HSCT evaluation + granulocyte transfusions; SCN → G-CSF filgrastim titration + annual BM biopsy; all → annual infectious disease surveillance",
        ],
        "gene_counts": gene_counts,
    }


def generate_breakdown() -> dict:
    cohort = _build_cohort()
    gene_breakdown = {}
    for entry in PHAGOCYTE_GENES:
        g = entry["gene"]
        pts = [p for p in cohort if p["gene"] == g]

        complication_counts = {}
        for p in pts:
            for c in p["complications"]:
                complication_counts[c] = complication_counts.get(c, 0) + 1

        dhr_distribution = {}
        for p in pts:
            dhr_distribution[p["dhr_result"]] = dhr_distribution.get(p["dhr_result"], 0) + 1

        treatment_counts = {}
        for p in pts:
            for t in p["treatments_used"]:
                treatment_counts[t] = treatment_counts.get(t, 0) + 1

        sev_counts = {}
        for p in pts:
            sev_counts[p["severity_label"]] = sev_counts.get(p["severity_label"], 0) + 1

        avg_delay = round(sum(p["diagnosis_delay_months"] for p in pts) / len(pts), 1) if pts else 0
        avg_anc = round(sum(p["anc_nadir_cells_ul"] for p in pts) / len(pts)) if pts else 0

        gene_breakdown[g] = {
            "gene": g,
            "n_patients": len(pts),
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0],
            "phagocyte_category": entry["phagocyte_category"],
            "phagocyte_pathway": entry["phagocyte_pathway"],
            "pathognomonic": entry["pathognomonic"][:200],
            "onset_age": entry["onset_age"],
            "complication_distribution": dict(sorted(complication_counts.items(), key=lambda x: -x[1])),
            "dhr_distribution": dhr_distribution,
            "treatment_distribution": dict(sorted(treatment_counts.items(), key=lambda x: -x[1])[:6]),
            "severity_distribution": sev_counts,
            "avg_diagnosis_delay_months": avg_delay,
            "avg_anc_nadir": avg_anc,
            "pct_granuloma": round(sum(1 for p in pts if p["granuloma_present"]) / len(pts) * 100, 1) if pts else 0,
            "pct_fungal": round(sum(1 for p in pts if p["fungal_infection_present"]) / len(pts) * 100, 1) if pts else 0,
            "pct_bleeding": round(sum(1 for p in pts if p["bleeding_present"]) / len(pts) * 100, 1) if pts else 0,
            "pct_neurological": round(sum(1 for p in pts if p["neurological_present"]) / len(pts) * 100, 1) if pts else 0,
            "granuloma_risk": entry["granuloma_risk"],
            "fungal_risk": entry["fungal_risk"],
            "bleeding_risk": entry["bleeding_risk"],
            "neurological_risk": entry["neurological_risk"],
        }
    return {"gene_breakdown": gene_breakdown, "n_genes": len(PHAGOCYTE_GENES), "n_patients": len(cohort)}


def generate_definitions() -> dict:
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["phagocyte_category"],
                "phagocyte_pathway": entry["phagocyte_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:400],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "granuloma_risk": entry["granuloma_risk"],
                "fungal_risk": entry["fungal_risk"],
                "bleeding_risk": entry["bleeding_risk"],
                "neurological_risk": entry["neurological_risk"],
                "onset_age": entry["onset_age"],
            }
            for entry in PHAGOCYTE_GENES
        },
        "phagocyte_glossary": {
            "Chronic Granulomatous Disease (CGD)": (
                "A primary immunodeficiency caused by mutations in NADPH oxidase components (CYBB/gp91phox, CYBA/p22phox, NCF1/p47phox, NCF2/p67phox). "
                "NADPH oxidase generates superoxide (O2•−) required to kill catalase-positive bacteria and fungi. "
                "CGD neutrophils fail to mount an oxidative burst → recurrent life-threatening infections with Staphylococcus aureus, Aspergillus fumigatus, Serratia marcescens, Burkholderia cepacia. "
                "Hyperinflammatory granuloma formation (GI tract, bladder) = second CGD phenotype. "
                "Diagnosed by DHR flow cytometry. Treated with triple prophylaxis (TMP-SMX + itraconazole + IFN-gamma) + HSCT for cure."
            ),
            "DHR (Dihydrorhodamine) Flow Cytometry": (
                "Gold standard diagnostic test for CGD. Neutrophils are stimulated with PMA (phorbol myristate acetate) → normal neutrophils generate ROS → DHR 123 is oxidised to fluorescent rhodamine 123 (bright green shift on flow). "
                "CGD neutrophils: no ROS → DHR remains non-fluorescent → zero or markedly reduced fluorescence shift. "
                "CYBB-CGD: zero shift (complete absence). NCF1-CGD: markedly reduced (partial). Female carriers: bimodal pattern. "
                "Must be performed on fresh blood within 4 hours."
            ),
            "Leukocyte Adhesion Deficiency (LAD)": (
                "Primary immunodeficiency caused by absent or dysfunctional beta-2 integrins on leukocytes. "
                "Neutrophils cannot adhere to activated endothelium → cannot migrate to sites of infection → infections proceed without pus. "
                "LAD-I (ITGB2): CD18 absent; LAD-III (FERMT3): CD18 reduced + platelet dysfunction. "
                "Pathognomonic: delayed umbilical cord separation (>21 days) + omphalitis + leukocytosis without pus formation. "
                "Severe LAD-I (<1% CD18): lethal in infancy without HSCT."
            ),
            "Severe Congenital Neutropenia (SCN)": (
                "Group of primary immunodeficiencies characterised by ANC persistently <200 cells/µL from birth due to granulopoiesis arrest at promyelocyte stage. "
                "Major genes: ELANE (AD — NE misfolding → UPR → apoptosis), HAX1 (AR — mitochondrial anti-apoptosis; original Kostmann disease 1956). "
                "Cyclic Neutropenia (ELANE): 21-day ANC cycling with nadir <200. "
                "G-CSF (filgrastim) lifelong = first-line; MDS/AML transformation risk 10-20% lifetime → annual BM biopsy mandatory."
            ),
            "Triple Prophylaxis (CGD)": (
                "Mandatory lifelong infection prophylaxis for ALL CGD patients regardless of subtype: "
                "(1) TMP-SMX (trimethoprim-sulfamethoxazole) — antibacterial (Staphylococcus, Serratia, Nocardia, Burkholderia); "
                "(2) Itraconazole — antifungal (Aspergillus primary prophylaxis, most effective antifungal in CGD); "
                "(3) IFN-gamma (Actimmune, 50 µg/m² 3× weekly SQ) — immunomodulatory (reduces infection frequency 70% in RCT). "
                "All three components have independent evidence. Non-adherence = life-threatening."
            ),
            "G-CSF (Filgrastim) in SCN": (
                "Granulocyte colony-stimulating factor — first-line lifelong treatment for ELANE-SCN1 and HAX1-SCN3 (Kostmann). "
                "Mechanism: stimulates myeloid progenitor proliferation → increases neutrophil output despite maturation arrest. "
                "Target: maintain ANC >1000 cells/µL. Dramatically reduces infection rate (>90% reduction). "
                "Risk: MDS/AML transformation via acquired G-CSF receptor (CSF3R) gain-of-function mutations → RUNX1 mutations → leukaemia. "
                "Annual BM biopsy + CSF3R/RUNX1 mutation panel mandatory for all SCN1/SCN3 patients on G-CSF."
            ),
            "Kindlin-3 (FERMT3) Dual Integrin Activation": (
                "Kindlin-3 is a FERM-domain protein exclusively expressed in haematopoietic cells (neutrophils, platelets, lymphocytes). "
                "Kindlin-3 activates integrin cytoplasmic tails via NPXY motif binding (inside-out signalling). "
                "FERMT3 mutations → Kindlin-3 absent → BOTH beta-2 integrins (leukocytes: LFA-1, Mac-1, p150.95) AND beta-3 integrins (platelets: GPIIb/GPIIIa) inactive simultaneously. "
                "Clinical result: LAD features (leukocyte adhesion defect) + Glanzmann thrombasthenia-like bleeding simultaneously. "
                "HSCT corrects both phenotypes (Kindlin-3 restored in all haematopoietic lineages)."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seed_range']}")
    print(f"CGD: {ov['cgd_genes']}")
    print(f"LAD: {ov['lad_genes']}")
    print(f"SCN: {ov['scn_genes']}")
    print("=== BREAKDOWN sample ===")
    bd = generate_breakdown()
    for g in ["CYBB", "ITGB2", "ELANE", "HAX1"]:
        gd = bd["gene_breakdown"][g]
        print(f"  {g}: {gd['n_patients']} pts, avg delay {gd['avg_diagnosis_delay_months']} mo")
    print("=== DEFINITIONS OK ===")
    df = generate_definitions()
    print(f"Gene entries: {list(df['gene_entries'].keys())}")
    print("All 3 endpoints OK.")
