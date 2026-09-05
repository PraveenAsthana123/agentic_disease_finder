#!/usr/bin/env python3
"""Amyloidosis-Atlas — Complete 8-Gene Hereditary Amyloidosis Atlas
TTR     (Transthyretin; 147 aa; 18q12.1; AD;
          ATTR-FAP (hereditary transthyretin amyloidosis with polyneuropathy) and ATTR-CM (cardiomyopathy);
          Val30Met founder (Portugal/Japan/Sweden); patisiran (siRNA)/inotersen (ASO) for neuropathy;
          tafamidis/acoramidis for cardiac stabilisation;
          LIVER TRANSPLANT: removes mutant TTR source BUT cardiac worsens — wild-type TTR from donor
          template replaces mutant; cardiac deposits continue to grow on wild-type scaffold;
          most common hereditary amyloidosis worldwide) ·
APOA1   (Apolipoprotein A-I; 267 aa; 11q23.3; AD;
          AApoAI amyloidosis — renal, hepatic, cardiac, neuropathic variants by mutation site;
          Leu64Pro → hepatic/renal; Leu75Pro → renal; Gly26Arg → neuropathic;
          LOW HDL PATHOGNOMONIC — apoA-I is the structural scaffold of HDL;
          NO approved amyloid-specific therapy) ·
APOA2   (Apolipoprotein A-II; 100 aa; 1q23.3; AD;
          AApoAII amyloidosis; RENAL PREDOMINANT — nephrotic syndrome → ESRD;
          frameshift creates C-terminal extension of 21 aa (Ostertag-type);
          no neuropathy, no cardiomyopathy; renal transplant standard of care for ESRD) ·
LYZ     (Lysozyme; 148 aa; 2q11.2; AD;
          ALys amyloidosis — systemic non-neuropathic;
          Ile56Thr and Asp67His most common (destabilise hydrophobic core);
          RENAL + hepatomegaly + GI bleeding (mesenteric involvement) + splenomegaly;
          NO peripheral neuropathy; organ transplant for end-stage organs) ·
GSN     (Gelsolin; 782 aa; 9q34.13; AD;
          AGel amyloidosis = Meretoja syndrome (Finnish type);
          Asp187Asn Finnish founder and Asp187Tyr;
          CRANIAL NEUROPATHY PATHOGNOMONIC — facial nerve FIRST (bilateral, progressive from 3rd decade);
          LATTICE CORNEAL DYSTROPHY type II PATHOGNOMONIC;
          proteinuria and cutis laxa late; unique cranial nerve + cornea diagnostic pair) ·
FGA     (Fibrinogen Alpha Chain; 866 aa; 4q28.1; AD;
          AFib amyloidosis (Ostertag); RENAL PREDOMINANT — nephrotic syndrome → ESRD;
          Glu526Val most common; NO neuropathy; NO cardiomyopathy (key DDx from TTR);
          LIVER TRANSPLANT CURATIVE — liver is exclusive source of amyloidogenic fibrinogen alpha chain;
          renal transplant alone FAILS — new fibrinogen deposits damage new kidney) ·
CST3    (Cystatin C; 146 aa; 20p11.21; AD;
          ACys amyloidosis (Icelandic type, HCHWA-I);
          Glu68Gln ONLY mutation described — Icelandic founder;
          CEREBROVASCULAR — intracerebral haemorrhage + subarachnoid haemorrhage in patients <40 y;
          cognitive decline/dementia; NO renal, NO peripheral neuropathy (key DDx);
          FATAL — median death age 30 without intervention) ·
B2M     (Beta-2 Microglobulin; 119 aa; 15q21.1; acquired/genetics-relevant;
          AB2M dialysis-related amyloidosis;
          CARPAL TUNNEL SYNDROME PATHOGNOMONIC (first sign in dialysis patients);
          destructive spondyloarthropathy; amyloid arthropathy of large joints;
          HIGH-FLUX DIALYSIS + online haemodiafiltration dramatically reduces AB2M burden;
          PREVENTION > TREATMENT; eliminated by kidney transplant)
320-patient aggregate cohort (8 x 40, seeds 1358-1365)
"""

import random

SEED_BASE = 1358

AMYLOIDOSIS_GENES = [
    # ── TTR — Hereditary ATTR Amyloidosis ──
    {
        "gene": "TTR",
        "protein": "Transthyretin (Prealbumin)",
        "alias": (
            "TTR; OMIM gene 176300; Hereditary Transthyretin Amyloidosis #176300 (AD); "
            "18q12.1; 147 aa; ~14 kDa monomer (55 kDa tetramer); plasma transport protein for thyroxine and retinol; "
            "synthesised predominantly in liver (90%) + choroid plexus + retinal pigment epithelium; "
            "TTR circulates as a homo-tetramer — tetramer stability is the key therapeutic target; "
            "pathogenic variants (>130 described) destabilise the tetramer → monomers misfold → amyloid fibrils; "
            "Val30Met (p.Val50Met new nomenclature) = MOST COMMON pathogenic variant worldwide (Portugal/Japan/Sweden); "
            "ATTR-FAP (polyneuropathy) and ATTR-CM (cardiomyopathy) = two major phenotypes"
        ),
        "aa": "147 aa",
        "kDa": "~14 kDa monomer / ~55 kDa tetramer",
        "locus": "18q12.1",
        "omim_gene": 176300,
        "omim_disease": 176300,
        "inheritance": (
            "AD — heterozygous pathogenic variants; variable penetrance (Val30Met: 5-10% lifetime penetrance in Portugal, "
            "higher in endemic regions Japan/Sweden); age-dependent penetrance increases markedly after age 50; "
            "late-onset variants (Val122Ile in 3-4% of African Americans) often present as 'sporadic' cardiomyopathy; "
            "homozygous ATTR described but does not clearly worsen phenotype (tetramer still formed from both alleles)"
        ),
        "gene_class": (
            "TTR encodes transthyretin, a homo-tetrameric plasma transport protein assembled in the liver. "
            "Monomers fold into a beta-sheet-rich structure; four monomers associate around two thyroxine-binding channels. "
            "PATHOGENESIS: Pathogenic variants (or ageing in wild-type ATTR) destabilise the quaternary tetramer → "
            "tetramer dissociates into monomers → partial unfolding → misassembly into amyloid fibrils → tissue deposition. "
            "PHENOTYPES: Val30Met → ATTR-FAP (length-dependent sensorimotor/autonomic polyneuropathy, earliest small-fibre "
            "neuropathy, onset 30-40s); Val122Ile + Thr60Ala → ATTR-CM (restrictive cardiomyopathy, onset 60-70s). "
            "Many variants cause mixed neuropathy + cardiomyopathy. "
            "LIVER-HEART PARADOX: Liver is the dominant TTR source (90%). Liver transplant removes mutant TTR-producing "
            "hepatocytes → mutant TTR production stops. However, existing cardiac amyloid deposits serve as a template; "
            "wild-type TTR (from donor liver) continues to deposit onto the existing cardiac matrix → cardiac disease "
            "WORSENS or stabilises at best post-transplant. Hence LTx is no longer first-line for ATTR-CM. "
            "THERAPIES: Patisiran/inotersen target TTR mRNA in liver → reduce total TTR production. "
            "Tafamidis/acoramidis kinetically stabilise the TTR tetramer → prevent dissociation → prevent amyloidogenesis."
        ),
        "hallmarks": [
            "Length-dependent sensorimotor polyneuropathy — small fibre (pain/temperature) first",
            "Autonomic dysfunction — orthostatic hypotension, GI dysmotility (alternating diarrhoea/constipation), ED",
            "Restrictive cardiomyopathy (ATTR-CM) — low-voltage ECG + thickened walls on echo",
            "Bilateral carpal tunnel syndrome — often the earliest manifestation (years before neuropathy)",
            "Vitreous opacities — deposits in vitreous humour; 'ghost vessel' appearance on slit lamp",
            "Leptomeningeal amyloidosis — choroid plexus TTR source; subarachnoid deposits",
            "Val30Met: Early onset (30-40s, Portugal/Japan); Late-onset (60-70s, Sweden) — same gene, different penetrance",
            "Val122Ile: ATTR-CM in African Americans; 3-4% carrier frequency — NOT rare",
        ],
        "treatment_alerts": [
            "PATISIRAN (siRNA): Approved for ATTR-FAP; hepatically delivered LNP-encapsulated siRNA; reduces TTR >80%",
            "INOTERSEN (ASO): Approved for ATTR-FAP; subcutaneous; monitor platelets (immune thrombocytopenia risk)",
            "TAFAMIDIS / ACORAMIDIS: TTR tetramer kinetic stabiliser; approved ATTR-CM; slows progression; start early",
            "LIVER TRANSPLANT PARADOX: Removes mutant TTR source BUT cardiac amyloid deposits continue to grow "
            "on wild-type TTR template from donor liver — cardiac disease often WORSENS post-transplant",
            "AVOID DIFLUNISAL in ATTR-CM: Older stabiliser; cardiac fluid retention risk outweighs benefit",
            "CARDIAC PACEMAKER/ICD: High-grade AV block and ventricular arrhythmias common — electrophysiology referral",
            "AVOID DIGOXIN: TTR amyloid deposits bind digoxin → unpredictable toxicity at standard doses",
            "NIS (technetium pyrophosphate) SCAN: Non-invasive diagnosis of ATTR-CM; >Grade 2 = diagnostic",
        ],
        "key_ddx": [
            "Wild-type ATTR (ATTRwt): Elderly males; same cardiac phenotype; NO pathogenic variant (ageing TTR)",
            "AL amyloidosis (plasma cell dyscrasia): Serum FLC assay + SPEP + UPEP; bone marrow biopsy",
            "AA amyloidosis: Chronic inflammation/infection history; serum amyloid A; renal predominant",
            "Hereditary ATTR vs ATTRwt: Genetic testing mandatory — treatment identical but family implications differ",
            "Diabetic neuropathy: Most common cause of sensorimotor neuropathy; autonomic overlap — check TTR",
            "CIDP (chronic inflammatory demyelinating polyneuropathy): Demyelinating pattern on NCS (ATTR = axonal/mixed)",
        ],
        "clinical_pearls": [
            "Val30Met penetrance is population-dependent: 5% lifetime penetrance in Portugal general population, >50% in endemic foci",
            "Bilateral carpal tunnel syndrome in a non-obese patient: think ATTR (years before neuropathy diagnosis)",
            "Low-voltage ECG + increased wall thickness on echo = ATTR-CM until proved otherwise — do NIS scan",
            "NIS (technetium-99m pyrophosphate or DPD) scan: Grade 2-3 = ATTR-CM with >99% specificity if FLC negative",
            "Vutrisiran (subcutaneous siRNA): Next-generation patisiran with less frequent dosing; approved 2022 FAP",
            "All first-degree relatives of confirmed TTR carriers must be offered genetic counselling and testing",
        ],
        "seed": 1358,
        "n_patients": 40,
        "etiologies": [
            ("Val30Met heterozygous (Portugal/Japan/Sweden founder)", 0.40),
            ("Val122Ile heterozygous (African American founder)", 0.18),
            ("Thr60Ala heterozygous (Irish founder)", 0.12),
            ("Other missense destabilising tetramer", 0.20),
            ("Compound het / two pathogenic variants (rare)", 0.10),
        ],
        "age_onset_years_range": (30, 75),
        "sex_ratio_M": 0.55,
        # Gene-specific feature rates
        "rates": {
            "neuropathy": 0.80,
            "autonomic": 0.70,
            "cardiomyopathy": 0.50,
            "renal": 0.15,
            "hepatic": 0.10,
            "carpal_tunnel": 0.30,
            "vitreous": 0.15,
            "cranial_neuropathy": 0.05,
            "corneal_dystrophy": 0.00,
            "cerebrovascular": 0.03,
        },
        "organ_system": "multisystem",
        "primary_treatment": "patisiran/inotersen (FAP) or tafamidis/acoramidis (CM)",
    },
    # ── APOA1 — AApoAI Amyloidosis ──
    {
        "gene": "APOA1",
        "protein": "Apolipoprotein A-I (ApoA-I)",
        "alias": (
            "APOA1; OMIM gene 107680; AApoAI Amyloidosis #107680 (AD); "
            "11q23.3; 267 aa; ~28 kDa; major structural protein of HDL particles; "
            "synthesised in liver and intestine; reverse cholesterol transport mediator; "
            "pathogenic variants cluster in N-terminal domain (residues 26-107) → amyloidogenic fragments; "
            "organotropism determined by mutation position: N-terminal variants → renal/hepatic; "
            "residues 170-178 variants → cardiac/neuropathic; "
            "LOW HDL is PATHOGNOMONIC — ApoA-I is the structural backbone of HDL; variants impair HDL assembly"
        ),
        "aa": "267 aa",
        "kDa": "~28 kDa",
        "locus": "11q23.3",
        "omim_gene": 107680,
        "omim_disease": 107680,
        "inheritance": (
            "AD — heterozygous pathogenic missense or deletion variants; "
            "variable penetrance and expressivity; "
            "multiple founders described (Leu64Pro, Leu75Pro, Gly26Arg, Trp50Arg among others); "
            "all first-degree relatives at 50% risk — cascade testing mandatory"
        ),
        "gene_class": (
            "APOA1 encodes Apolipoprotein A-I, the major structural protein of high-density lipoprotein (HDL) particles. "
            "ApoA-I is synthesised in hepatocytes and intestinal epithelium. "
            "PRIMARY FUNCTION: Facilitates reverse cholesterol transport; activates LCAT (lecithin-cholesterol acyltransferase); "
            "structural scaffold for HDL discs → mature spherical HDL particles. "
            "AMYLOIDOSIS MECHANISM: Pathogenic variants (>20 described) render the protein or proteolytic fragments "
            "amyloidogenic. The N-terminal 93 residues are particularly prone to forming fibrils when destabilised. "
            "ORGANOTROPISM BY MUTATION: Different residue positions confer different organ tropism — "
            "N-terminal region (Leu64Pro, Leu75Pro): renal + hepatic amyloid. "
            "Mid-region (Gly26Arg): neuropathic variant (peripheral nerve deposition). "
            "C-terminal (residues 170-178): cardiac and cutaneous. "
            "LOW HDL: Since ApoA-I is the structural backbone of HDL, pathogenic variants impair HDL assembly → "
            "serum HDL-C is markedly LOW — this is a cardinal diagnostic clue. "
            "NO APPROVED AMYLOID-SPECIFIC THERAPY: Management is organ-directed (renal transplant, liver transplant)."
        ),
        "hallmarks": [
            "LOW HDL cholesterol — PATHOGNOMONIC diagnostic clue (ApoA-I is the HDL scaffold protein)",
            "Renal amyloidosis — nephrotic syndrome → CKD → ESRD (most common presentation)",
            "Hepatic amyloidosis — hepatomegaly, progressive liver dysfunction (Leu64Pro variant)",
            "Cardiac amyloidosis — restrictive cardiomyopathy (less common than TTR but real)",
            "Peripheral neuropathy — Gly26Arg variant specifically (not universal)",
            "Mutations in N-terminal domain: renal/hepatic; mid-region: neuropathic; C-terminal: cardiac",
            "NO approved amyloid-specific therapy — organ transplantation for end-stage disease",
            "Family history of young-onset renal failure + low HDL = suspect AApoAI until proved otherwise",
        ],
        "treatment_alerts": [
            "NO AMYLOID-SPECIFIC THERAPY: No approved TTR stabiliser, siRNA, or ASO for AApoAI",
            "RENAL TRANSPLANT: Standard of care for ESRD; however amyloid may recur in allograft from ongoing ApoA-I production",
            "LIVER TRANSPLANT: May reduce amyloidogenic ApoA-I production if liver is the dominant source",
            "ACE INHIBITOR / ARB: Standard nephroprotection for proteinuric renal amyloidosis",
            "STATIN: Treat dyslipidaemia — low HDL + elevated LDL common; cardiovascular risk elevated",
            "CARDIAC MONITORING: Echo annually if cardiac variant mutation; low-threshold for cardiac MRI",
            "CASCADE GENETIC TESTING: All first-degree relatives of confirmed AApoAI index cases",
            "AVOID HIGH-DOSE FIBRATES: Modest HDL raise but may not help amyloid; safety uncertain",
        ],
        "key_ddx": [
            "AL amyloidosis: Most common systemic amyloidosis; serum FLC + SPEP mandatory before labelling hereditary",
            "TTR amyloidosis: More common hereditary; TTR genetic panel + NIS scan + FLC should be first-line",
            "AA amyloidosis: Renal predominant + low serum amyloid A; inflammatory history",
            "Familial hypercholesterolaemia (APOB/LDLR): HIGH LDL not LOW HDL; no amyloid",
            "Tangier disease (ABCA1): Very low HDL + no amyloid; cholesterol ester accumulation in tonsils",
            "Hereditary renal amyloidosis DDx: APOA1, APOA2, LYZ, FGA, CST3 — gene panel essential",
        ],
        "clinical_pearls": [
            "Low HDL in a patient with unexplained proteinuria or hepatomegaly: order hereditary amyloidosis panel",
            "ApoA-I mutation organotropism: know which mutation hits which organ (N-terminal = renal/hepatic first)",
            "Congo red staining + mass spectrometry peptide analysis is gold-standard amyloid typing",
            "Serum amyloid P (SAP) scintigraphy quantifies total body amyloid burden — available in specialised centres",
            "Recurrence in renal allograft is possible but slower — transplant still offers survival benefit",
            "Avoid labelling as 'idiopathic' low HDL without genetic workup in families with renal/hepatic disease",
        ],
        "seed": 1359,
        "n_patients": 40,
        "etiologies": [
            ("Leu64Pro (hepatic/renal founder)", 0.30),
            ("Leu75Pro (renal predominant)", 0.25),
            ("Gly26Arg (neuropathic variant)", 0.20),
            ("Trp50Arg or other N-terminal missense", 0.15),
            ("C-terminal deletion (cardiac variant)", 0.10),
        ],
        "age_onset_years_range": (35, 65),
        "sex_ratio_M": 0.50,
        "rates": {
            "neuropathy": 0.15,
            "autonomic": 0.10,
            "cardiomyopathy": 0.25,
            "renal": 0.75,
            "hepatic": 0.60,
            "carpal_tunnel": 0.10,
            "vitreous": 0.02,
            "cranial_neuropathy": 0.02,
            "corneal_dystrophy": 0.00,
            "cerebrovascular": 0.02,
        },
        "organ_system": "renal",
        "primary_treatment": "renal/liver transplant (no amyloid-specific therapy)",
    },
    # ── APOA2 — AApoAII Amyloidosis ──
    {
        "gene": "APOA2",
        "protein": "Apolipoprotein A-II (ApoA-II)",
        "alias": (
            "APOA2; OMIM gene 107670; AApoAII Amyloidosis #105200 (AD); "
            "1q23.3; 100 aa (mature peptide 77 aa); ~17 kDa; HDL-associated apolipoprotein; "
            "second most abundant protein in HDL after ApoA-I; "
            "pathogenic frameshift mutations create C-terminal extension of 21 additional amino acids; "
            "Ostertag-family type amyloidosis — first described in Ostertag family 1932; "
            "RENAL PREDOMINANT — nephrotic syndrome progressing to ESRD; "
            "NO peripheral neuropathy, NO cardiomyopathy — pure renal phenotype distinguishes from APOA1"
        ),
        "aa": "100 aa",
        "kDa": "~17 kDa",
        "locus": "1q23.3",
        "omim_gene": 107670,
        "omim_disease": 105200,
        "inheritance": (
            "AD — heterozygous frameshift mutation in APOA2 creating stop codon read-through + C-terminal extension; "
            "full penetrance in most described pedigrees; "
            "late-onset (5th-7th decade); "
            "rare disorder — fewer than 20 pedigrees described worldwide; "
            "Ostertag-type = generic name for non-neuropathic, non-cardiac, renal-predominant hereditary amyloidosis"
        ),
        "gene_class": (
            "APOA2 encodes Apolipoprotein A-II, the second most abundant protein of HDL. "
            "Mature ApoA-II (77 aa) exists as a homodimer linked by a disulfide bond at Cys6. "
            "PRIMARY FUNCTION: Modulates HDL metabolism; activates hepatic lipase; "
            "APOA2 is less critical for HDL structure than APOA1. "
            "AMYLOIDOSIS MECHANISM: Frameshift mutations in exon 3 of APOA2 disrupt the stop codon → "
            "the ribosome reads through into the 3'UTR → creates an abnormal C-terminal extension of ~21 amino acids. "
            "This elongated ApoA-II (Ostertag type) is amyloidogenic. The extra C-terminal sequence is hydrophobic "
            "and aggregation-prone → deposits in renal glomeruli + interstitium. "
            "PHENOTYPE: Purely renal — nephrotic-range proteinuria → progressive CKD → ESRD (5th-7th decade). "
            "No neuropathy (key DDx from TTR/APOA1). No cardiomyopathy. No hepatomegaly. "
            "Mild hepatic involvement in some cases (<20%). "
            "DIAGNOSIS: Renal biopsy → Congo red birefringence → mass spectrometry peptide typing identifies ApoA-II peptides."
        ),
        "hallmarks": [
            "RENAL PREDOMINANT — nephrotic syndrome → ESRD (5th-7th decade) — 100% have renal involvement",
            "Frameshift creates C-terminal extension of 21 aa (Ostertag mechanism) — unique molecular basis",
            "NO peripheral neuropathy — key DDx from TTR and APOA1 (neuropathic variants)",
            "NO cardiomyopathy — key DDx from TTR-CM",
            "Mild hepatomegaly in minority (<20%) — hepatic involvement is secondary",
            "Family history of late-onset renal failure (Ostertag family pattern)",
            "Rare disorder — <20 pedigrees worldwide; diagnosis requires mass spectrometry amyloid typing",
            "HDL may be mildly low but not as striking as in APOA1 amyloidosis",
        ],
        "treatment_alerts": [
            "NO AMYLOID-SPECIFIC THERAPY: Supportive and organ-directed only",
            "RENAL TRANSPLANT: Standard of care for ESRD; amyloid recurrence in allograft is a risk",
            "ACE INHIBITOR / ARB: Standard nephroprotection for nephrotic-range proteinuria",
            "BLOOD PRESSURE CONTROL: Strict (<125/75) to slow CKD progression",
            "AVOID NEPHROTOXIC AGENTS: NSAIDs, aminoglycosides, contrast — accelerate CKD in amyloid",
            "RENAL BIOPSY + MASS SPECTROMETRY: Essential for amyloid subtype diagnosis before organ referral",
            "CASCADE GENETIC TESTING: All first-degree relatives; autosomal dominant — 50% risk",
            "HEMODIALYSIS PLANNING: Early arteriovenous fistula planning at CKD Stage 4",
        ],
        "key_ddx": [
            "AApoAI: Also renal + hepatic; but neuropathic variants exist; very low HDL; APOA1 mutations",
            "ALys amyloidosis (LYZ): Renal + hepatomegaly + GI bleeding + splenomegaly; LYZ mutations",
            "AFib amyloidosis (FGA): Renal only; NO neuropathy; NO cardiac; liver Tx curative — FGA mutations",
            "AL amyloidosis: Most common systemic; FLC + SPEP essential; plasma cell dyscrasia",
            "AA amyloidosis: Renal predominant; inflammatory history; serum amyloid A elevated",
            "IgA nephropathy: Renal; no Congo red birefringence; different histology",
        ],
        "clinical_pearls": [
            "Ostertag-type amyloidosis = pure renal amyloidosis in a family with AD pattern: think APOA2, FGA, LYZ",
            "Frameshift at APOA2 stop codon creates the amyloidogenic extension — molecular basis unique",
            "Mass spectrometry (laser-capture + MS/MS) on renal biopsy amyloid extracts identifies ApoA-II peptides",
            "Renal transplant offers good medium-term outcomes; follow up for allograft recurrence (annual proteinuria)",
            "Genetic counselling mandatory — full penetrance; all children at 50% risk",
            "No approved targeted amyloid therapy; international registries important for trial eligibility",
        ],
        "seed": 1360,
        "n_patients": 40,
        "etiologies": [
            ("Frameshift stop codon read-through — C-terminal 21 aa extension (classic Ostertag)", 0.60),
            ("Alternative frameshift same mechanism — different breakpoint", 0.25),
            ("Missense destabilising (rare non-Ostertag mechanism)", 0.10),
            ("Novel frameshift variant — not yet published", 0.05),
        ],
        "age_onset_years_range": (45, 70),
        "sex_ratio_M": 0.50,
        "rates": {
            "neuropathy": 0.00,
            "autonomic": 0.05,
            "cardiomyopathy": 0.05,
            "renal": 1.00,
            "hepatic": 0.20,
            "carpal_tunnel": 0.15,
            "vitreous": 0.00,
            "cranial_neuropathy": 0.00,
            "corneal_dystrophy": 0.00,
            "cerebrovascular": 0.00,
        },
        "organ_system": "renal",
        "primary_treatment": "renal transplant (supportive; no amyloid-specific therapy)",
    },
    # ── LYZ — ALys Amyloidosis ──
    {
        "gene": "LYZ",
        "protein": "Lysozyme (1,4-beta-N-acetylmuramidase C)",
        "alias": (
            "LYZ; OMIM gene 153450; ALys Amyloidosis #105200 (AD); "
            "2q11.2; 148 aa (mature 130 aa); ~16 kDa; antimicrobial muramidase enzyme; "
            "expressed in granulocytes, monocytes, macrophages, epithelial cells, liver (minor); "
            "Ile56Thr and Asp67His most common pathogenic variants (hydrophobic core destabilisation); "
            "systemic non-neuropathic amyloidosis — RENAL + hepatic + GI + splenic involvement; "
            "NO peripheral neuropathy; NO cardiomyopathy — pure visceral systemic amyloidosis"
        ),
        "aa": "148 aa",
        "kDa": "~16 kDa",
        "locus": "2q11.2",
        "omim_gene": 153450,
        "omim_disease": 105200,
        "inheritance": (
            "AD — heterozygous pathogenic missense variants in LYZ gene; "
            "Ile56Thr and Asp67His account for majority of cases; "
            "first described in a British family in 1993 (Pepys et al.); "
            "fewer than 50 pedigrees described worldwide; "
            "all first-degree relatives at 50% risk — cascade genetic testing mandatory"
        ),
        "gene_class": (
            "LYZ encodes human lysozyme, a 14.3 kDa antimicrobial enzyme of the muramidase family. "
            "Lysozyme cleaves the beta-1,4 linkage between N-acetylmuramic acid and N-acetylglucosamine in bacterial cell walls. "
            "AMYLOIDOSIS MECHANISM: Ile56Thr (destabilises beta-domain) and Asp67His (ionisable at neutral pH → "
            "partial unfolding) disrupt the hydrophobic core → thermodynamically unstable monomers → "
            "amyloid fibrils form under physiological conditions. Unlike TTR, lysozyme amyloid forms from the monomeric state. "
            "PHENOTYPE: Systemic non-neuropathic visceral amyloidosis. "
            "RENAL: Glomerular and vascular amyloid → proteinuria → CKD → ESRD (~80%). "
            "HEPATIC: Hepatomegaly; liver amyloid deposits in sinusoids and walls (~70%). "
            "GI: Mesenteric vessel amyloid → GI bleeding (haematemesis/malaena); small bowel involvement (~60%). "
            "SPLENIC: Splenomegaly with amyloid deposits (~50%). "
            "IMPORTANT: No peripheral neuropathy (key DDx from TTR/APOA1); no cardiomyopathy. "
            "Diagnosis: renal or hepatic biopsy Congo red + mass spectrometry identifies lysozyme peptides."
        ),
        "hallmarks": [
            "Renal amyloidosis — nephrotic syndrome → CKD → ESRD (~80%)",
            "Hepatomegaly — liver amyloid deposition in sinusoids (~70%)",
            "GI bleeding — mesenteric vascular amyloid → mucosal ischaemia/haemorrhage (~60%)",
            "Splenomegaly — amyloid splenic deposition (~50%)",
            "NO peripheral neuropathy — key DDx from TTR and APOA1 neuropathic variants",
            "NO cardiomyopathy — key DDx from ATTR-CM",
            "Ile56Thr and Asp67His: hydrophobic core destabilisation — most common variants",
            "Systemic visceral amyloidosis without nervous system involvement (pure Ostertag pattern)",
        ],
        "treatment_alerts": [
            "NO AMYLOID-SPECIFIC THERAPY: No approved TTR stabilisers, siRNA, or ASO applicable to ALys",
            "RENAL TRANSPLANT: Standard of care for ESRD; amyloid may recur in allograft (ongoing lysozyme production)",
            "LIVER TRANSPLANT: May reduce amyloidogenic lysozyme burden if liver contribution is significant",
            "GI BLEEDING MANAGEMENT: PPI + endoscopic haemostasis; angiography/embolisation for mesenteric bleeding",
            "ACE INHIBITOR / ARB: Nephroprotection for proteinuric renal amyloidosis",
            "IRON SUPPLEMENTATION: Chronic GI blood loss → iron deficiency anaemia; oral iron ± IV iron",
            "SPLEEN MONITORING: Hypersplenism if massive splenomegaly; cytopenia monitoring",
            "CASCADE GENETIC TESTING: All first-degree relatives; 50% risk (AD)",
        ],
        "key_ddx": [
            "AL amyloidosis: Most common systemic; plasma cell dyscrasia; FLC + SPEP + UPEP mandatory",
            "AA amyloidosis: Chronic inflammatory disease history; serum amyloid A; renal predominant",
            "AApoAI: Renal + hepatic but may have neuropathic variant; low HDL; APOA1 mutations",
            "AFib (FGA): Renal only; NO GI bleeding; NO hepatomegaly; fibrinogen A-chain peptides on MS",
            "TTR amyloidosis: Neuropathy + cardiomyopathy + vitreous; renal involvement uncommon",
            "Hereditary GI bleeding DDx: Vascular malformations; PRSS1 pancreatitis; amyloid-specific Congo red",
        ],
        "clinical_pearls": [
            "GI bleeding + hepatomegaly + proteinuria in an AD family pattern: think ALys until proved otherwise",
            "Mass spectrometry (laser-capture MS on biopsy) identifies lysozyme peptides — essential for subtyping",
            "Serum lysozyme is elevated in monocyte/macrophage disorders (leukaemia) — not a diagnostic test for ALys",
            "Ile56Thr thermostability test: elevated temperature destabilises variant more than wild-type (research use)",
            "GI bleeding can be life-threatening and is the most morbid complication (mesenteric ischaemia)",
            "Renal transplant offers survival benefit despite risk of allograft recurrence — individualise decision",
        ],
        "seed": 1361,
        "n_patients": 40,
        "etiologies": [
            ("Ile56Thr heterozygous (most common)", 0.45),
            ("Asp67His heterozygous", 0.35),
            ("Trp64Arg or other rare missense", 0.12),
            ("Novel missense (hydrophobic core)", 0.08),
        ],
        "age_onset_years_range": (40, 70),
        "sex_ratio_M": 0.50,
        "rates": {
            "neuropathy": 0.00,
            "autonomic": 0.05,
            "cardiomyopathy": 0.05,
            "renal": 0.80,
            "hepatic": 0.70,
            "carpal_tunnel": 0.10,
            "vitreous": 0.00,
            "cranial_neuropathy": 0.00,
            "corneal_dystrophy": 0.00,
            "cerebrovascular": 0.02,
        },
        "organ_system": "multisystem",
        "primary_treatment": "renal transplant; GI management; no amyloid-specific therapy",
    },
    # ── GSN — Meretoja Syndrome / AGel Amyloidosis ──
    {
        "gene": "GSN",
        "protein": "Gelsolin (Actin-modulating protein)",
        "alias": (
            "GSN; OMIM gene 137350; AGel Amyloidosis / Meretoja Syndrome #105120 (AD); "
            "9q34.13; 782 aa (cytoplasmic) / 755 aa (plasma, secreted); ~80 kDa; "
            "actin-severing and capping protein; calcium-regulated; "
            "Asp187Asn (c.559G>A) — Finnish founder mutation; Asp187Tyr — non-Finnish variant; "
            "CRANIAL NEUROPATHY PATHOGNOMONIC — facial nerve (CN VII) bilateral, progressive from 3rd decade; "
            "LATTICE CORNEAL DYSTROPHY type II (LCD type II) PATHOGNOMONIC from early adulthood; "
            "unique diagnostic pair: cranial neuropathy + lattice corneal dystrophy"
        ),
        "aa": "782 aa",
        "kDa": "~80 kDa",
        "locus": "9q34.13",
        "omim_gene": 137350,
        "omim_disease": 105120,
        "inheritance": (
            "AD — heterozygous pathogenic missense at Asp187 (furin cleavage site in plasma gelsolin); "
            "Asp187Asn = Finnish founder (c.559G>A); Asp187Tyr = non-Finnish variant (c.559G>T); "
            "same codon, different nucleotide substitution → different amino acid but identical mechanism; "
            "penetrance essentially 100% by 6th decade; "
            "all first-degree relatives at 50% risk; prenatal diagnosis available"
        ),
        "gene_class": (
            "GSN encodes gelsolin, a calcium-regulated actin-severing and filament-capping protein. "
            "Two isoforms: cytoplasmic (782 aa, intracellular) and plasma/secreted (755 aa, N-terminally truncated). "
            "AMYLOIDOSIS MECHANISM: Asp187 lies within the furin cleavage site of plasma gelsolin. "
            "Asp187Asn or Asp187Tyr disrupts Ca²⁺ binding → abnormal furin cleavage of plasma gelsolin → "
            "generates an amyloidogenic 71-aa C-terminal fragment (C68 fragment) → deposits in cornea, cranial nerves, "
            "skin, blood vessels, and kidney. "
            "PHENOTYPE: Uniquely distinctive — LATTICE CORNEAL DYSTROPHY type II (earliest sign, begins 20s-30s) + "
            "CRANIAL NEUROPATHY (facial nerve bilateral and progressive, begins 3rd decade, later other cranial nerves) + "
            "CUTIS LAXA (skin laxity and redundancy from amyloid in dermis, begins 4th-5th decade) + "
            "PROTEINURIA (kidney involvement, 3rd-4th decade). "
            "The cranial neuropathy + corneal dystrophy pair is essentially pathognomonic for AGel amyloidosis. "
            "Systemic amyloid burden is modest — rarely fatal directly; morbidity from cranial nerve palsy + corneal disease."
        ),
        "hallmarks": [
            "LATTICE CORNEAL DYSTROPHY type II — PATHOGNOMONIC: lattice lines in corneal stroma from 20s-30s",
            "FACIAL NERVE (CN VII) PALSY — bilateral, progressive, PATHOGNOMONIC from 3rd decade",
            "Cranial nerve involvement progresses: CN V, IX, X later — dysphagia, dysarthria",
            "CUTIS LAXA — skin redundancy/laxity from dermal amyloid (4th-5th decade)",
            "Proteinuria — renal amyloid (3rd-4th decade); rarely reaches ESRD",
            "Asp187Asn (Finnish founder) — endemic in Finland; Asp187Tyr in non-Finnish families",
            "Both Asp187 variants have identical clinical phenotype — same codon, different nucleotide",
            "Cranial neuropathy + lattice corneal dystrophy = DIAGNOSTIC PAIR for AGel (no other hereditary amyloidosis)",
        ],
        "treatment_alerts": [
            "NO AMYLOID-SPECIFIC THERAPY: No approved siRNA, ASO, or stabiliser for AGel amyloidosis",
            "CORNEAL TRANSPLANT (penetrating keratoplasty): For advanced corneal dystrophy impairing vision; recurrence in graft",
            "FACIAL NERVE DECOMPRESSION / TARSORRHAPHY: For corneal exposure from CN VII palsy — prevent exposure keratopathy",
            "ARTIFICIAL TEARS + LUBRICANT EYE DROPS: Mandatory for CN VII palsy → incomplete eye closure → dry eye",
            "SWALLOWING ASSESSMENT: Formal SLT assessment when CN IX/X involved — aspiration pneumonia risk",
            "OPHTHALMOLOGY REVIEW: Annual slit-lamp examination from diagnosis; corneal topography",
            "RENAL MONITORING: Annual urine protein:creatinine ratio + eGFR",
            "GENETIC COUNSELLING: 50% risk to all first-degree relatives; test at age 18-20",
        ],
        "key_ddx": [
            "Lattice corneal dystrophy type I (TGFBI): Common corneal dystrophy; NO cranial neuropathy; no amyloid",
            "Lattice corneal dystrophy type III (TGFBI): Similar cornea; no systemic features",
            "Bell's palsy (idiopathic CN VII): Unilateral; acute; recovers — AGel is BILATERAL, PROGRESSIVE, permanent",
            "TTR amyloidosis: Peripheral neuropathy + cardiac; cranial nerve palsy rare; no corneal dystrophy",
            "Facial palsy DDx: Sarcoidosis, Lyme, bilateral Bell — but all unilateral + recoverable in most cases",
            "Cutis laxa DDx: ATP7A (Menkes), FBLN5, ELN elastic fibre disorders — no corneal/cranial nerve involvement",
        ],
        "clinical_pearls": [
            "If bilateral progressive facial palsy + lattice corneal lines on slit lamp → diagnose AGel until proved otherwise",
            "Finnish ancestry + any of the three features (cornea/facial palsy/cutis laxa) → test GSN Asp187Asn first",
            "Corneal transplant recurs — lattice deposits recur in donor graft tissue; counsel patient pre-operatively",
            "Facial palsy management: daily eye care essential — Bell's phenomenon, corneal exposure, tarsorrhaphy if needed",
            "Dysphagia (CN IX/X palsy) is a late but dangerous complication: initiate SLT early",
            "Genetic test: full exon sequencing identifies Asp187Asn/Tyr; no MLPA needed (point mutations only described)",
        ],
        "seed": 1362,
        "n_patients": 40,
        "etiologies": [
            ("Asp187Asn (c.559G>A) Finnish founder", 0.60),
            ("Asp187Tyr (c.559G>T) non-Finnish", 0.35),
            ("Other Asp187 substitution (novel, same codon)", 0.05),
        ],
        "age_onset_years_range": (25, 55),
        "sex_ratio_M": 0.50,
        "rates": {
            "neuropathy": 0.10,
            "autonomic": 0.10,
            "cardiomyopathy": 0.05,
            "renal": 0.40,
            "hepatic": 0.05,
            "carpal_tunnel": 0.15,
            "vitreous": 0.05,
            "cranial_neuropathy": 1.00,
            "corneal_dystrophy": 0.95,
            "cerebrovascular": 0.02,
        },
        "organ_system": "neurologic",
        "primary_treatment": "corneal transplant; facial nerve care; no amyloid-specific therapy",
    },
    # ── FGA — AFib Amyloidosis ──
    {
        "gene": "FGA",
        "protein": "Fibrinogen Alpha Chain (FGA)",
        "alias": (
            "FGA; OMIM gene 134820; AFib Amyloidosis / Ostertag Type #105200 (AD); "
            "4q28.1; 866 aa (mature chain 610 aa); ~95 kDa; coagulation factor; "
            "synthesised EXCLUSIVELY in the liver — hepatocytes are the only fibrinogen source; "
            "Glu526Val most common pathogenic variant (90% of described families); "
            "RENAL PREDOMINANT — nephrotic syndrome → ESRD (dominant organ affected); "
            "NO neuropathy; NO cardiomyopathy — key DDx from TTR and APOA1; "
            "LIVER TRANSPLANT CURATIVE — removes ONLY source of amyloidogenic fibrinogen; "
            "RENAL TRANSPLANT ALONE FAILS — donor kidney is destroyed by ongoing FGA amyloid deposition"
        ),
        "aa": "866 aa",
        "kDa": "~95 kDa",
        "locus": "4q28.1",
        "omim_gene": 134820,
        "omim_disease": 105200,
        "inheritance": (
            "AD — heterozygous pathogenic missense variants (Glu526Val accounts for ~90% of pedigrees); "
            "full penetrance for renal disease by 6th-7th decade; "
            "Glu526Val Indiana/Swiss/French founder — multiple independent families worldwide; "
            "first-degree relatives at 50% risk; genetic testing should precede renal transplant planning"
        ),
        "gene_class": (
            "FGA encodes the alpha-chain of fibrinogen, the coagulation zymogen cleaved by thrombin to form fibrin clots. "
            "Fibrinogen (a dimeric hexamer: [AalphaB betaGamma]₂) is synthesised EXCLUSIVELY in hepatocytes — "
            "no extrahepatic source exists. This biological exclusivity makes liver transplant uniquely curative. "
            "AMYLOIDOSIS MECHANISM: Glu526Val and other C-terminal alpha-chain variants generate an amyloidogenic "
            "C-terminal fragment after proteolytic cleavage of circulating fibrinogen. This fragment deposits preferentially "
            "in renal glomeruli → mesangial and vascular amyloid → nephrotic syndrome → progressive CKD → ESRD (5th-7th decade). "
            "PHENOTYPE: Exclusively renal in Glu526Val. NO peripheral neuropathy. NO cardiomyopathy. "
            "Hepatic involvement minimal despite liver origin of the protein. "
            "LIVER TRANSPLANT: Curative because it eliminates the ONLY source of amyloidogenic fibrinogen alpha chain. "
            "Renal function stabilises/improves after liver Tx (no new FGA deposits). "
            "RENAL TRANSPLANT ALONE: The new kidney is destroyed by ongoing amyloid deposition from the patient's "
            "own FGA — renal Tx alone fails within 5-8 years consistently. "
            "COMBINED LIVER-KIDNEY TRANSPLANT: Preferred at centres with experience."
        ),
        "hallmarks": [
            "RENAL PREDOMINANT — nephrotic syndrome → ESRD (100% have renal involvement by 6th-7th decade)",
            "NO neuropathy — key DDx from TTR (key examination point)",
            "NO cardiomyopathy — key DDx from ATTR-CM (NIS scan and FLC negative)",
            "Glu526Val — accounts for ~90% of all AFib amyloidosis pedigrees worldwide",
            "LIVER IS THE EXCLUSIVE SOURCE — all amyloidogenic fibrinogen alpha chain comes from hepatocytes",
            "LIVER TRANSPLANT CURATIVE: Eliminates source; renal deposits stop growing; stabilisation/improvement",
            "RENAL TRANSPLANT ALONE FAILS: New kidney destroyed by ongoing Glu526Val fibrinogen deposition (~5-8 yr)",
            "Normal coagulation parameters despite variant (heterozygous variant does not impair clotting)",
        ],
        "treatment_alerts": [
            "LIVER TRANSPLANT — CURATIVE: First-line recommendation for eligible patients before ESRD develops",
            "RENAL TRANSPLANT ALONE — FAILS: Document clearly in patient records; transplant teams must know AFib diagnosis",
            "COMBINED LIVER-KIDNEY TRANSPLANT: Preferred strategy in centres experienced with combined Tx",
            "TIMING OF LIVER TRANSPLANT: Before ESRD (CKD Stage 3-4); better outcomes if renal function preserved",
            "ACE INHIBITOR / ARB: Nephroprotection until liver transplant can be arranged",
            "NO AMYLOID-SPECIFIC MEDICAL THERAPY: No approved siRNA/ASO/stabiliser for AFib",
            "FAMILY TESTING: All first-degree relatives; identify pre-symptomatic carriers for surveillance",
            "COAGULATION SCREEN: Routine screen normal — Glu526Val does not impair fibrinogen polymerisation",
        ],
        "key_ddx": [
            "AL amyloidosis: Most common; FLC + SPEP + UPEP + bone marrow biopsy; renal predominant variants",
            "AA amyloidosis: Chronic inflammatory disease; serum amyloid A; renal predominant",
            "AApoAI: Renal + hepatic; neuropathic variants exist; low HDL; APOA1 gene",
            "AApoAII: Renal only; frameshift C-terminal extension; APOA2 gene",
            "TTR amyloidosis: Neuropathy + cardiac; renal involvement uncommon; TTR gene",
            "Focal segmental glomerulosclerosis (FSGS): Nephrotic syndrome; Congo red negative; different histology",
        ],
        "clinical_pearls": [
            "Pure renal amyloidosis (nephrotic + CKD) with AD inheritance + no neuropathy + no cardiac = AFib until proved otherwise",
            "Renal biopsy: Congo red + mass spectrometry identifies fibrinogen alpha-chain peptides",
            "FGA amyloidosis is the ONLY hereditary amyloidosis where liver Tx is explicitly curative and renal Tx alone explicitly fails",
            "Coagulation studies are NORMAL — the Glu526Val fibrinogen polymerises normally; amyloidogenicity is a separate property",
            "Combined liver-kidney transplant: renal function recovers if done before ESRD; 10-year survival good in experienced centres",
            "Glu526Val: test all siblings and children of index case; pre-symptomatic carriers can be transplanted electively",
        ],
        "seed": 1363,
        "n_patients": 40,
        "etiologies": [
            ("Glu526Val heterozygous (Indiana/Swiss/French founder)", 0.90),
            ("Other FGA missense (non-Glu526Val pathogenic variant)", 0.07),
            ("Novel FGA variant — pathogenicity confirmed by functional assay", 0.03),
        ],
        "age_onset_years_range": (40, 70),
        "sex_ratio_M": 0.50,
        "rates": {
            "neuropathy": 0.00,
            "autonomic": 0.03,
            "cardiomyopathy": 0.00,
            "renal": 1.00,
            "hepatic": 0.10,
            "carpal_tunnel": 0.08,
            "vitreous": 0.00,
            "cranial_neuropathy": 0.00,
            "corneal_dystrophy": 0.00,
            "cerebrovascular": 0.02,
        },
        "organ_system": "renal",
        "primary_treatment": "liver transplant (curative); renal transplant alone FAILS",
    },
    # ── CST3 — ACys Amyloidosis (Icelandic type) ──
    {
        "gene": "CST3",
        "protein": "Cystatin C (Cystatin-3)",
        "alias": (
            "CST3; OMIM gene 604312; ACys Amyloidosis / HCHWA-I (Hereditary Cerebral Haemorrhage with "
            "Amyloidosis — Icelandic type) #105150 (AD); "
            "20p11.21; 146 aa (mature 120 aa); ~13 kDa; cysteine protease inhibitor; "
            "synthesised ubiquitously (all nucleated cells); secreted into CSF, plasma, urine; "
            "serum cystatin C = GFR surrogate marker in nephrology; "
            "Glu68Gln (p.E68Q) — the ONLY pathogenic variant described; Icelandic founder; "
            "CEREBROVASCULAR DISEASE — intracerebral haemorrhage + subarachnoid haemorrhage in patients <40 y; "
            "FATAL — median death age 30; NO renal; NO peripheral neuropathy"
        ),
        "aa": "146 aa",
        "kDa": "~13 kDa",
        "locus": "20p11.21",
        "omim_gene": 604312,
        "omim_disease": 105150,
        "inheritance": (
            "AD — heterozygous Glu68Gln (p.E68Q) ONLY described mutation; Icelandic founder; "
            "essentially ONLY described in Iceland (and Icelandic descendants); "
            "full penetrance — all carriers develop cerebral haemorrhage by 4th-5th decade without intervention; "
            "median death age 30 without management; "
            "first-degree relatives at 50% risk — genetic testing is a medical priority in affected families"
        ),
        "gene_class": (
            "CST3 encodes Cystatin C (also known as Cystatin-3 or gamma-trace protein), a 13 kDa secreted cysteine "
            "protease inhibitor expressed ubiquitously. "
            "NORMAL FUNCTION: Cystatin C inhibits cysteine proteases (cathepsins B, H, L, S) in extracellular fluids; "
            "a major endogenous inhibitor of lysosomal enzyme escape; also used clinically as a GFR surrogate marker. "
            "AMYLOIDOSIS MECHANISM: Glu68Gln (the Icelandic mutation) reduces the net negative charge of cystatin C at "
            "the active-site loop → destabilises the protein → promotes dimerisation and oligomerisation → amyloid fibrils "
            "that deposit in small cerebral arterial walls and leptomeningeal vessels. "
            "PHENOTYPE: PURE CEREBROVASCULAR disease. "
            "Amyloid deposits in cerebral arterioles and leptomeningeal arteries → vessel wall fragility → "
            "intracerebral haemorrhage (ICH) and subarachnoid haemorrhage (SAH) from young adulthood (2nd-4th decade). "
            "Recurrent strokes → progressive cognitive decline → dementia → death (median age 30). "
            "NO renal involvement. NO peripheral neuropathy. NO cardiomyopathy. NO corneal disease. "
            "KEY DDx: cerebral amyloid angiopathy (CAA-TTR, APP-mutation HCHWA-D) — CST3-HCHWA-I is Icelandic-specific. "
            "TREATMENT: No approved amyloid-specific therapy; strict blood pressure control to reduce haemorrhage risk."
        ),
        "hallmarks": [
            "INTRACEREBRAL HAEMORRHAGE — onset before age 40 (often 20s-30s); PATHOGNOMONIC in Icelandic families",
            "SUBARACHNOID HAEMORRHAGE — small leptomeningeal artery rupture",
            "COGNITIVE DECLINE / DEMENTIA — progressive after recurrent strokes",
            "FATAL — median death age 30 without blood pressure management (most lethal hereditary amyloidosis)",
            "Glu68Gln — the ONLY mutation described; Icelandic founder (virtually exclusive to Iceland)",
            "NO renal involvement — key DDx from AFib/AApoAI/APOA2/LYZ",
            "NO peripheral neuropathy — key DDx from TTR and APOA1",
            "Serum cystatin C is ELEVATED (abnormal secretion of p.E68Q variant) — diagnostic clue",
        ],
        "treatment_alerts": [
            "BLOOD PRESSURE CONTROL — CRITICAL: Strict BP <130/80 to reduce haemorrhage risk; lifetime adherence",
            "AVOID ANTICOAGULANTS: Warfarin, DOACs — dramatically increase haemorrhage risk in cystatin C angiopathy",
            "AVOID ANTIPLATELETS unless absolutely necessary: Aspirin/clopidogrel increase ICH risk",
            "NO AMYLOID-SPECIFIC THERAPY: No approved siRNA, stabiliser, or ASO for ACys",
            "AVOID COCAINE/STIMULANTS/ALCOHOL EXCESS: BP elevation triggers haemorrhage; lifestyle counselling",
            "GENETIC COUNSELLING + TESTING: All first-degree relatives of Icelandic families; testing is urgent",
            "NEUROSURGICAL EVALUATION: For large/accessible haematoma with mass effect — surgical evacuation",
            "STATIN: Reduces risk of haemorrhagic stroke from vascular inflammation (indirect benefit)",
        ],
        "key_ddx": [
            "HCHWA-D (Dutch type, APP Glu693Gln): Cerebral haemorrhage + dementia — APP mutation, Dutch ancestry",
            "Sporadic cerebral amyloid angiopathy (CAA): Elderly; lobular ICH; APOE epsilon4; NOT hereditary",
            "Cerebral cavernous malformations (CCM1/2/3): Young ICH; KRIT1/CCM2/PDCD10; family history",
            "Hypertensive intracerebral haemorrhage: Deep ganglia; no hereditary amyloid; HTN present",
            "CADASIL (NOTCH3): Subcortical strokes + migraine + CADASIL; NOT haemorrhagic primarily",
            "AL amyloidosis with CNS involvement: Plasma cell dyscrasia; FLC + SPEP; peripheral > cerebral",
        ],
        "clinical_pearls": [
            "Young Icelander (20s-30s) with ICH or SAH: CST3 Glu68Gln testing is mandatory — diagnose before next bleed",
            "Serum cystatin C levels are elevated in carriers (abnormally secreted Glu68Gln variant) — useful screening tool",
            "Blood pressure is the ONLY modifiable risk factor — lifetime adherence is life-saving",
            "Anticoagulation is absolutely contraindicated — even for AF; prefer rate control strategies",
            "Median death age 30 without BP management — most lethal of all hereditary amyloidoses by age of death",
            "All affected families should have a confirmed molecular diagnosis before obstetric decisions",
        ],
        "seed": 1364,
        "n_patients": 40,
        "etiologies": [
            ("Glu68Gln (p.E68Q) heterozygous — Icelandic founder (the only described pathogenic variant)", 1.00),
        ],
        "age_onset_years_range": (20, 40),
        "sex_ratio_M": 0.50,
        "rates": {
            "neuropathy": 0.00,
            "autonomic": 0.05,
            "cardiomyopathy": 0.00,
            "renal": 0.00,
            "hepatic": 0.00,
            "carpal_tunnel": 0.03,
            "vitreous": 0.00,
            "cranial_neuropathy": 0.08,
            "corneal_dystrophy": 0.00,
            "cerebrovascular": 1.00,
        },
        "organ_system": "cerebrovascular",
        "primary_treatment": "blood pressure control; no amyloid-specific therapy; avoid anticoagulation",
    },
    # ── B2M — AB2M Dialysis-Related Amyloidosis ──
    {
        "gene": "B2M",
        "protein": "Beta-2 Microglobulin (β₂M)",
        "alias": (
            "B2M; OMIM gene 109700; AB2M Dialysis-Related Amyloidosis (DRA) / Haemodialysis-Related Amyloidosis; "
            "15q21.1; 119 aa (mature 99 aa); ~12 kDa; MHC class I light chain; "
            "non-covalently associated with all MHC class I molecules on nucleated cell surfaces; "
            "predominantly acquired (not inherited), but genetics-relevant: "
            "serum beta-2 microglobulin is renally excreted — ESRD causes >50-fold accumulation; "
            "CARPAL TUNNEL SYNDROME PATHOGNOMONIC — earliest manifestation in dialysis patients; "
            "HIGH-FLUX DIALYSIS + haemodiafiltration (HDF) dramatically reduces serum β₂M — PREVENTION > TREATMENT; "
            "ELIMINATED BY KIDNEY TRANSPLANT (restores renal β₂M clearance)"
        ),
        "aa": "119 aa",
        "kDa": "~12 kDa",
        "locus": "15q21.1",
        "omim_gene": 109700,
        "omim_disease": 109700,
        "inheritance": (
            "Acquired (not inherited) — AB2M amyloidosis develops in patients with long-term renal replacement therapy; "
            "genetics-relevant because: (1) B2M gene variants affect amyloidogenicity threshold; "
            "(2) rare hereditary B2M variants (D76N) cause systemic non-dialysis amyloidosis; "
            "for this atlas, AB2M dialysis-related amyloidosis is included as the definitive acquired amyloidosis "
            "with clear preventable/reversible mechanism; all included patients have ESRD on long-term dialysis"
        ),
        "gene_class": (
            "B2M encodes beta-2 microglobulin (β₂M), the 12 kDa light chain subunit non-covalently associated with "
            "all HLA class I (A, B, C, E, F, G) molecules on the surface of virtually all nucleated cells. "
            "NORMAL PHYSIOLOGY: β₂M is constitutively shed from cell surfaces and circulates freely in plasma. "
            "Normally cleared by glomerular filtration → proximal tubule reabsorption and catabolism → "
            "serum β₂M maintains ~1-2 mg/L with normal renal function. "
            "DRA MECHANISM: In ESRD, renal clearance of β₂M is abolished → serum β₂M rises to 50-100 mg/L "
            "(>50-fold accumulation). At high concentrations, β₂M undergoes amyloidogenesis (especially in the "
            "presence of advanced glycation end-products (AGE), Cu²⁺, collagen VIII). "
            "DEPOSITION SITES: Articular cartilage, synovium, carpal tunnel (flexor tenosynovium), intervertebral discs, "
            "long bone cysts (destructive spondyloarthropathy, amyloid arthropathy). "
            "HIGH-FLUX MEMBRANES + HDF: Remove significantly more β₂M than low-flux dialysis → lower serum levels → "
            "delayed/reduced amyloid deposition. PREVENTION IS THE ONLY EFFECTIVE STRATEGY. "
            "KIDNEY TRANSPLANT: Restores glomerular filtration → serum β₂M normalises rapidly → amyloid stops growing; "
            "existing deposits may partially resorb over years. "
            "Rare D76N variant: Hereditary β₂M amyloidosis (non-dialysis) — visceral deposits."
        ),
        "hallmarks": [
            "CARPAL TUNNEL SYNDROME — PATHOGNOMONIC: First manifestation in dialysis patients (typically after 5-7 yr on dialysis)",
            "DESTRUCTIVE SPONDYLOARTHROPATHY — disco-vertebral destruction, cervical spine especially dangerous",
            "AMYLOID ARTHROPATHY OF LARGE JOINTS — shoulders (dialysis shoulder), hips, knees",
            "SUBCHONDRAL BONE CYSTS — lytic cysts in carpal bones, femoral head, humeral head",
            "ALL PATIENTS ARE ON LONG-TERM DIALYSIS — AB2M does not occur with normal renal function",
            "SERUM β₂M MARKEDLY ELEVATED — >50 mg/L (normal <2 mg/L); years of ESRD",
            "HIGH-FLUX DIALYSIS + HDF PREVENTS OR DELAYS — prevention is the primary strategy",
            "KIDNEY TRANSPLANT ELIMINATES — serum β₂M normalises; deposits stop growing; partial resorption",
        ],
        "treatment_alerts": [
            "HIGH-FLUX MEMBRANES: MANDATORY for all dialysis patients — reduces β₂M accumulation substantially",
            "ONLINE HAEMODIAFILTRATION (HDF): Further reduces serum β₂M vs high-flux HD alone; preferred where available",
            "KIDNEY TRANSPLANT: Definitive therapy — eliminates DRA substrate by restoring β₂M clearance",
            "CARPAL TUNNEL DECOMPRESSION: Surgical carpal tunnel release relieves pain/neuropathy; amyloid recurs slowly",
            "CERVICAL SPINE SURVEILLANCE: X-ray annually; destructive spondyloarthropathy can cause cord compression",
            "AVOID LOW-FLUX MEMBRANES: Very poor β₂M clearance — high-flux is the minimum standard now",
            "SHOULDER JOINT INJECTION: Corticosteroid injection for amyloid arthropathy pain (temporary benefit)",
            "REGULAR β₂M MONITORING: Serial serum β₂M levels — high levels predict earlier DRA onset",
        ],
        "key_ddx": [
            "De Quervain tenosynovitis: Radial wrist pain; no bilateral carpal tunnel; no β₂M elevation",
            "Rheumatoid arthritis: Symmetric inflammatory arthropathy; RF/anti-CCP; different distribution",
            "Dialysis-associated crystal arthropathy (calcium pyrophosphate): Chondrocalcinosis on X-ray; fluid crystals",
            "Septic arthritis in dialysis patient: Fever; elevated WBC; culture mandatory",
            "Hereditary B2M amyloidosis (D76N): Systemic, non-dialysis, young patients; B2M D76N genetic variant",
            "Calcium phosphate deposition (calciphylaxis): Different distribution; skin/vessel calcification; not synovial",
        ],
        "clinical_pearls": [
            "Bilateral carpal tunnel syndrome in a dialysis patient >5 years: AB2M until proved otherwise",
            "Prevention is the ONLY effective strategy — start high-flux HD or HDF from dialysis initiation",
            "Serum β₂M level correlates with DRA risk: >30 mg/L significantly increases risk; >50 mg/L = high risk",
            "Kidney transplant is the best treatment — serum β₂M normalises within days; clinical improvement follows",
            "Destructive spondyloarthropathy: Cervical spine lesions can cause quadriplegia — early MRI if neck symptoms",
            "Online HDF achieves β₂M clearance 40-60% higher than high-flux HD alone — target in all incident patients",
        ],
        "seed": 1365,
        "n_patients": 40,
        "etiologies": [
            ("ESRD on haemodialysis >5 years (low-flux membrane)", 0.40),
            ("ESRD on haemodialysis >5 years (high-flux membrane, suboptimal clearance)", 0.30),
            ("ESRD on peritoneal dialysis — lower clearance of β₂M", 0.15),
            ("ESRD >10 years — long-duration dialysis regardless of modality", 0.10),
            ("Rare hereditary D76N variant — systemic non-dialysis AB2M", 0.05),
        ],
        "age_onset_years_range": (40, 70),
        "sex_ratio_M": 0.55,
        "rates": {
            "neuropathy": 0.10,
            "autonomic": 0.05,
            "cardiomyopathy": 0.05,
            "renal": 1.00,  # all patients have ESRD by definition
            "hepatic": 0.03,
            "carpal_tunnel": 0.90,
            "vitreous": 0.00,
            "cranial_neuropathy": 0.02,
            "corneal_dystrophy": 0.00,
            "cerebrovascular": 0.03,
        },
        "organ_system": "multisystem",
        "primary_treatment": "high-flux dialysis / HDF; kidney transplant eliminates",
    },
]


def _build_cohort(gene_def: dict) -> list:
    """Generate synthetic patient cohort for a gene."""
    rng = random.Random(gene_def["seed"])
    patients = []
    etiologies = gene_def["etiologies"]
    gene = gene_def["gene"]
    ages_years = gene_def.get("age_onset_years_range", (40, 65))
    sex_ratio_m = gene_def.get("sex_ratio_M", 0.50)
    rates = gene_def["rates"]

    for i in range(gene_def["n_patients"]):
        # Pick etiology via cumulative probability
        r = rng.random()
        cumulative = 0.0
        etiology = etiologies[-1][0]
        for name, prob in etiologies:
            cumulative += prob
            if r <= cumulative:
                etiology = name
                break

        age_at_onset = rng.randint(ages_years[0], max(ages_years[0] + 1, ages_years[1]))
        age_at_diagnosis = age_at_onset + rng.randint(0, 10)
        sex = "M" if rng.random() < sex_ratio_m else "F"

        # Clinical features — gene-specific rates
        has_neuropathy = rng.random() < rates["neuropathy"]
        has_autonomic_dysfunction = rng.random() < rates["autonomic"]
        has_cardiomyopathy = rng.random() < rates["cardiomyopathy"]
        has_renal_involvement = rng.random() < rates["renal"]
        has_hepatic_involvement = rng.random() < rates["hepatic"]
        has_carpal_tunnel = rng.random() < rates["carpal_tunnel"]
        has_vitreous_opacity = rng.random() < rates["vitreous"]
        has_cranial_neuropathy = rng.random() < rates["cranial_neuropathy"]
        has_corneal_dystrophy = rng.random() < rates["corneal_dystrophy"]
        has_cerebrovascular_event = rng.random() < rates["cerebrovascular"]

        # B2M: spondyloarthropathy and large joint amyloid arthropathy
        has_spondyloarthropathy = gene == "B2M" and rng.random() < 0.60
        has_large_joint_arthropathy = gene == "B2M" and rng.random() < 0.70
        # LYZ: GI bleeding and splenomegaly
        has_gi_bleeding = gene == "LYZ" and rng.random() < 0.60
        has_splenomegaly = gene == "LYZ" and rng.random() < 0.50
        # CST3: cognitive decline
        has_cognitive_decline = gene == "CST3" and rng.random() < 0.80
        # GSN: cutis laxa
        has_cutis_laxa = gene == "GSN" and rng.random() < 0.30

        # Organ system primary
        organ_system = gene_def["organ_system"]

        # Treatment received (gene-appropriate)
        if gene == "TTR":
            if has_neuropathy:
                treatment_options = ["patisiran", "inotersen", "vutrisiran", "liver transplant + patisiran"]
            else:
                treatment_options = ["tafamidis", "acoramidis", "tafamidis + diuretics"]
            treatment = rng.choice(treatment_options)
        elif gene == "APOA1":
            treatment = rng.choice(["ACE inhibitor/ARB", "renal transplant", "liver transplant", "supportive"])
        elif gene == "APOA2":
            treatment = rng.choice(["ACE inhibitor/ARB", "renal transplant", "haemodialysis", "supportive"])
        elif gene == "LYZ":
            treatment = rng.choice(["renal transplant", "liver transplant", "supportive + GI management", "ACE inhibitor/ARB"])
        elif gene == "GSN":
            treatment = rng.choice(["corneal transplant", "facial nerve care", "artificial tears", "supportive"])
        elif gene == "FGA":
            if rng.random() < 0.50:
                treatment = "liver transplant (curative)"
            elif rng.random() < 0.60:
                treatment = "combined liver-kidney transplant"
            else:
                treatment = rng.choice(["ACE inhibitor/ARB", "renal transplant (failed — relisted)"])
        elif gene == "CST3":
            treatment = rng.choice(["blood pressure control (antihypertensive)", "amlodipine", "ACE inhibitor", "supportive only"])
        elif gene == "B2M":
            treatment = rng.choice([
                "high-flux haemodialysis",
                "online haemodiafiltration (HDF)",
                "kidney transplant",
                "carpal tunnel release + high-flux HD",
            ])
        else:
            treatment = "supportive"

        # Serum amyloid P (SAP) scan performed
        sap_scan = rng.random() < 0.45 and gene != "CST3" and gene != "B2M"

        # Congo red biopsy confirmed
        congo_red_positive = rng.random() < 0.90

        # Genetic testing route
        genetic_route = rng.choice([
            "index case clinical diagnosis → gene panel",
            "family history → predictive testing",
            "incidental gene panel finding",
            "post-transplant unexpected amyloid typing",
        ])

        patients.append({
            "patient_id": f"{gene}-P{i+1:03d}",
            "gene": gene,
            "etiology": etiology,
            "age_at_onset": age_at_onset,
            "age_at_diagnosis": age_at_diagnosis,
            "sex": sex,
            "has_neuropathy": has_neuropathy,
            "has_autonomic_dysfunction": has_autonomic_dysfunction,
            "has_cardiomyopathy": has_cardiomyopathy,
            "has_renal_involvement": has_renal_involvement,
            "has_hepatic_involvement": has_hepatic_involvement,
            "has_carpal_tunnel": has_carpal_tunnel,
            "has_vitreous_opacity": has_vitreous_opacity,
            "has_cranial_neuropathy": has_cranial_neuropathy,
            "has_corneal_dystrophy": has_corneal_dystrophy,
            "has_cerebrovascular_event": has_cerebrovascular_event,
            "has_spondyloarthropathy": has_spondyloarthropathy,
            "has_large_joint_arthropathy": has_large_joint_arthropathy,
            "has_gi_bleeding": has_gi_bleeding,
            "has_splenomegaly": has_splenomegaly,
            "has_cognitive_decline": has_cognitive_decline,
            "has_cutis_laxa": has_cutis_laxa,
            "organ_system": organ_system,
            "treatment_received": treatment,
            "sap_scan_performed": sap_scan,
            "congo_red_positive": congo_red_positive,
            "genetic_testing_route": genetic_route,
        })
    return patients


def get_overview():
    """Atlas overview: gene list, aggregate stats, key DDx anchors."""
    genes_summary = []
    total_patients = 0
    total_neuropathy = 0
    total_cardiomyopathy = 0
    total_renal = 0
    total_cerebrovascular = 0
    total_carpal_tunnel = 0
    total_cranial_neuropathy = 0
    total_corneal_dystrophy = 0
    total_hepatic = 0
    total_autonomic = 0

    for gd in AMYLOIDOSIS_GENES:
        cohort = _build_cohort(gd)
        n = len(cohort)
        total_patients += n

        neuro = sum(1 for p in cohort if p["has_neuropathy"])
        cardio = sum(1 for p in cohort if p["has_cardiomyopathy"])
        renal = sum(1 for p in cohort if p["has_renal_involvement"])
        cerebro = sum(1 for p in cohort if p["has_cerebrovascular_event"])
        carpal = sum(1 for p in cohort if p["has_carpal_tunnel"])
        cranial = sum(1 for p in cohort if p["has_cranial_neuropathy"])
        corneal = sum(1 for p in cohort if p["has_corneal_dystrophy"])
        hepatic = sum(1 for p in cohort if p["has_hepatic_involvement"])
        autonomic = sum(1 for p in cohort if p["has_autonomic_dysfunction"])

        total_neuropathy += neuro
        total_cardiomyopathy += cardio
        total_renal += renal
        total_cerebrovascular += cerebro
        total_carpal_tunnel += carpal
        total_cranial_neuropathy += cranial
        total_corneal_dystrophy += corneal
        total_hepatic += hepatic
        total_autonomic += autonomic

        avg_onset = round(sum(p["age_at_onset"] for p in cohort) / n, 1)
        avg_diag_delay = round(
            sum(p["age_at_diagnosis"] - p["age_at_onset"] for p in cohort) / n, 1
        )

        organ_dist = {}
        for p in cohort:
            organ_dist[p["organ_system"]] = organ_dist.get(p["organ_system"], 0) + 1

        genes_summary.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "n_patients": n,
            "neuropathy_pct": round(100 * neuro / n, 1),
            "autonomic_pct": round(100 * autonomic / n, 1),
            "cardiomyopathy_pct": round(100 * cardio / n, 1),
            "renal_pct": round(100 * renal / n, 1),
            "hepatic_pct": round(100 * hepatic / n, 1),
            "carpal_tunnel_pct": round(100 * carpal / n, 1),
            "cranial_neuropathy_pct": round(100 * cranial / n, 1),
            "corneal_dystrophy_pct": round(100 * corneal / n, 1),
            "cerebrovascular_pct": round(100 * cerebro / n, 1),
            "avg_age_at_onset": avg_onset,
            "avg_diagnosis_delay_years": avg_diag_delay,
            "primary_organ_system": gd["organ_system"],
            "primary_treatment": gd["primary_treatment"],
            "hallmarks": gd["hallmarks"][:4],
            "top_treatment_alert": gd["treatment_alerts"][0],
        })

    return {
        "atlas": "Amyloidosis-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Amyloidosis Atlas",
        "api_path": "/api/amyloidosis-atlas/",
        "genes": [g["gene"] for g in AMYLOIDOSIS_GENES],
        "total_patients": total_patients,
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "aggregate_stats": {
            "neuropathy_pct": round(100 * total_neuropathy / total_patients, 1),
            "cardiomyopathy_pct": round(100 * total_cardiomyopathy / total_patients, 1),
            "renal_involvement_pct": round(100 * total_renal / total_patients, 1),
            "cerebrovascular_pct": round(100 * total_cerebrovascular / total_patients, 1),
            "carpal_tunnel_pct": round(100 * total_carpal_tunnel / total_patients, 1),
            "cranial_neuropathy_pct": round(100 * total_cranial_neuropathy / total_patients, 1),
            "corneal_dystrophy_pct": round(100 * total_corneal_dystrophy / total_patients, 1),
            "hepatic_involvement_pct": round(100 * total_hepatic / total_patients, 1),
            "autonomic_dysfunction_pct": round(100 * total_autonomic / total_patients, 1),
        },
        "genes_summary": genes_summary,
        "key_ddx_anchor": [
            "TTR vs AL: ALL hereditary amyloidosis — exclude AL (FLC+SPEP+UPEP+BM biopsy) FIRST before labelling hereditary",
            "TTR liver transplant PARADOX: Removes mutant source but CARDIAC WORSENS — wild-type TTR from donor deposits on existing cardiac template",
            "FGA liver transplant CURATIVE: Liver is ONLY source of amyloidogenic fibrinogen — renal transplant alone FAILS (deposits destroy new kidney)",
            "GSN DIAGNOSTIC PAIR: Bilateral lattice corneal dystrophy type II + bilateral facial palsy = Meretoja syndrome (AGel) — pathognomonic combination",
            "CST3 CEREBROVASCULAR: Young Icelander + ICH = CST3 Glu68Gln until proved otherwise; AVOID anticoagulation; BP control is lifesaving",
            "B2M PREVENTION > TREATMENT: High-flux HD + HDF reduces AB2M burden; kidney transplant eliminates; carpal tunnel = first sign",
            "APOA1 LOW HDL CLUE: ApoA-I is the HDL scaffold — markedly low HDL in a patient with renal amyloidosis = test APOA1",
        ],
    }


def get_breakdown():
    """Per-gene detailed breakdown with cohort data."""
    result = []
    for gd in AMYLOIDOSIS_GENES:
        cohort = _build_cohort(gd)
        n = len(cohort)

        # Etiology counts
        etiol_counts = {}
        for p in cohort:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        # Sex breakdown
        males = sum(1 for p in cohort if p["sex"] == "M")

        # Feature rates
        feature_rates = {
            "neuropathy_pct": round(100 * sum(1 for p in cohort if p["has_neuropathy"]) / n, 1),
            "autonomic_dysfunction_pct": round(100 * sum(1 for p in cohort if p["has_autonomic_dysfunction"]) / n, 1),
            "cardiomyopathy_pct": round(100 * sum(1 for p in cohort if p["has_cardiomyopathy"]) / n, 1),
            "renal_involvement_pct": round(100 * sum(1 for p in cohort if p["has_renal_involvement"]) / n, 1),
            "hepatic_involvement_pct": round(100 * sum(1 for p in cohort if p["has_hepatic_involvement"]) / n, 1),
            "carpal_tunnel_pct": round(100 * sum(1 for p in cohort if p["has_carpal_tunnel"]) / n, 1),
            "vitreous_opacity_pct": round(100 * sum(1 for p in cohort if p["has_vitreous_opacity"]) / n, 1),
            "cranial_neuropathy_pct": round(100 * sum(1 for p in cohort if p["has_cranial_neuropathy"]) / n, 1),
            "corneal_dystrophy_pct": round(100 * sum(1 for p in cohort if p["has_corneal_dystrophy"]) / n, 1),
            "cerebrovascular_pct": round(100 * sum(1 for p in cohort if p["has_cerebrovascular_event"]) / n, 1),
            "spondyloarthropathy_pct": round(100 * sum(1 for p in cohort if p["has_spondyloarthropathy"]) / n, 1),
            "large_joint_arthropathy_pct": round(100 * sum(1 for p in cohort if p["has_large_joint_arthropathy"]) / n, 1),
            "gi_bleeding_pct": round(100 * sum(1 for p in cohort if p["has_gi_bleeding"]) / n, 1),
            "splenomegaly_pct": round(100 * sum(1 for p in cohort if p["has_splenomegaly"]) / n, 1),
            "cognitive_decline_pct": round(100 * sum(1 for p in cohort if p["has_cognitive_decline"]) / n, 1),
            "cutis_laxa_pct": round(100 * sum(1 for p in cohort if p["has_cutis_laxa"]) / n, 1),
            "sap_scan_performed_pct": round(100 * sum(1 for p in cohort if p["sap_scan_performed"]) / n, 1),
            "congo_red_positive_pct": round(100 * sum(1 for p in cohort if p["congo_red_positive"]) / n, 1),
        }

        # Treatment distribution
        tx_dist = {}
        for p in cohort:
            tx_dist[p["treatment_received"]] = tx_dist.get(p["treatment_received"], 0) + 1
        tx_list = sorted(
            [{"treatment": k, "n": v, "pct": round(100 * v / n, 1)} for k, v in tx_dist.items()],
            key=lambda x: -x["n"],
        )

        # Etiology distribution as list
        etiol_list = sorted(
            [{"etiology": k, "n": v, "pct": round(100 * v / n, 1)} for k, v in etiol_counts.items()],
            key=lambda x: -x["n"],
        )

        # Organ system distribution
        organ_dist = {}
        for p in cohort:
            organ_dist[p["organ_system"]] = organ_dist.get(p["organ_system"], 0) + 1

        # Diagnosis delay
        avg_diag_delay = round(
            sum(p["age_at_diagnosis"] - p["age_at_onset"] for p in cohort) / n, 1
        )
        avg_onset = round(sum(p["age_at_onset"] for p in cohort) / n, 1)

        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "gene_class": gd["gene_class"],
            "hallmarks": gd["hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "key_ddx": gd["key_ddx"],
            "clinical_pearls": gd["clinical_pearls"],
            "n_patients": n,
            "males_pct": round(100 * males / n, 1),
            "avg_age_at_onset": avg_onset,
            "avg_diagnosis_delay_years": avg_diag_delay,
            "primary_organ_system": gd["organ_system"],
            "primary_treatment": gd["primary_treatment"],
            "feature_rates": feature_rates,
            "treatment_distribution": tx_list,
            "etiology_distribution": etiol_list,
            "organ_system_distribution": {
                k: {"n": v, "pct": round(100 * v / n, 1)} for k, v in organ_dist.items()
            },
        })
    return result


def get_definitions():
    """Clinical definitions and pharmacological distinctions for the atlas."""
    return {
        "atlas": "Amyloidosis-Atlas",
        "definitions": [
            {
                "term": "Hereditary Amyloidosis (Overview)",
                "definition": (
                    "Hereditary amyloidoses are autosomal dominant (or rarely acquired/genetics-relevant) disorders "
                    "in which a single amino acid substitution or frameshift renders a normally soluble protein "
                    "amyloidogenic. The misfolded protein (or its proteolytic fragment) assembles into insoluble, "
                    "beta-sheet-rich amyloid fibrils that deposit in organ interstitium, vessel walls, and stroma, "
                    "progressively impairing organ function. Fibrils bind Congo red dye and produce apple-green "
                    "birefringence under polarised light — this remains the gold-standard diagnostic test. "
                    "Mass spectrometry (laser-capture microdissection + LC-MS/MS) on biopsy-derived amyloid extracts "
                    "identifies the specific amyloid protein (proteomic typing), replacing immunohistochemistry as the "
                    "standard of care for amyloid subtyping. Serum amyloid P (SAP) scintigraphy quantifies total "
                    "body amyloid burden. Hereditary amyloidoses include ATTR, AApoAI, AApoAII, ALys, AGel, AFib, "
                    "ACys, and AB2M (dialysis-related). AL amyloidosis (plasma cell dyscrasia) is the most common "
                    "systemic amyloidosis overall and must be excluded before diagnosing any hereditary subtype."
                ),
            },
            {
                "term": "TTR Amyloidosis (ATTR) and the Liver-Heart Paradox",
                "definition": (
                    "Transthyretin (TTR) is a homo-tetrameric transport protein synthesised 90% in the liver. "
                    "Pathogenic variants (>130 described) or ageing (wild-type ATTR) destabilise the tetramer → "
                    "monomeric misfolding → amyloid fibrils. "
                    "LIVER TRANSPLANT PARADOX: Liver Tx removes the source of mutant TTR — mutant TTR production ceases. "
                    "However, existing amyloid deposits in the heart and blood vessels serve as a structural TEMPLATE. "
                    "Wild-type TTR produced by the donor liver is recruited onto this template and continues to deposit → "
                    "CARDIAC AMYLOID CONTINUES TO GROW on the wild-type scaffold. "
                    "Consequence: Liver transplant stabilises or improves neuropathy (nerves heal) but cardiac disease "
                    "WORSENS post-transplant. This paradox is why liver Tx is no longer recommended for ATTR-CM, "
                    "and why tafamidis/acoramidis (which stabilise the tetramer regardless of source) are preferred for cardiac. "
                    "ATTR-FAP (polyneuropathy, Val30Met): patisiran/inotersen suppress hepatic TTR mRNA → most effective. "
                    "ATTR-CM: tafamidis/acoramidis prevent tetramer dissociation → first-line cardiac therapy."
                ),
            },
            {
                "term": "Meretoja Syndrome / AGel Amyloidosis (GSN) — Cranial Neuropathy + Lattice Corneal Dystrophy",
                "definition": (
                    "GSN (gelsolin) Asp187Asn/Tyr generates an amyloidogenic C-terminal fragment (C68 fragment) after "
                    "aberrant furin cleavage of plasma gelsolin. Deposits localise to cornea, cranial nerve sheaths, "
                    "dermis, and renal vasculature. "
                    "PATHOGNOMONIC DIAGNOSTIC PAIR: "
                    "(1) LATTICE CORNEAL DYSTROPHY type II — visible on slit-lamp as lattice lines in corneal stroma; "
                    "begins in 20s-30s; bilaterally symmetrical; does not occur in any other hereditary amyloidosis. "
                    "(2) CRANIAL NERVE VII (FACIAL) PALSY — bilateral, progressive, beginning in 3rd decade; "
                    "evolves to involve CN V (trigeminal), IX, X — dysphagia and dysarthria follow. "
                    "Cutis laxa (dermal amyloid) appears in 4th-5th decade. "
                    "Proteinuria in 3rd-4th decade but ESRD is uncommon. "
                    "DIAGNOSIS: Both features together are essentially pathognomonic; GSN sequencing confirms Asp187 mutation. "
                    "No amyloid-specific therapy approved. Corneal transplant recurs. Facial nerve care is primary management."
                ),
            },
            {
                "term": "AFib Amyloidosis (FGA) — Liver Transplant Curative; Renal Transplant Alone Fails",
                "definition": (
                    "FGA (fibrinogen alpha chain) encodes the alpha chain of fibrinogen, synthesised EXCLUSIVELY in "
                    "hepatocytes. No extrahepatic source exists. Glu526Val is the sole mutation in ~90% of pedigrees. "
                    "Glu526Val fibrinogen generates an amyloidogenic C-terminal fragment → renal glomerular deposition → "
                    "nephrotic syndrome → CKD → ESRD. "
                    "LIVER TRANSPLANT IS CURATIVE: Replaces the ONLY source of amyloidogenic fibrinogen alpha chain → "
                    "no new deposits → renal function stabilises/recovers (if done before ESRD). "
                    "RENAL TRANSPLANT ALONE FAILS: The transplanted kidney is exposed to ongoing Glu526Val fibrinogen "
                    "from the recipient's own liver → amyloid deposits destroy the allograft within 5-8 years. "
                    "This is the starkest organ-transplant rule in all hereditary amyloidosis. "
                    "Combined liver-kidney transplant is the preferred strategy in experienced centres. "
                    "KEY DDx FROM TTR: AFib = NO neuropathy, NO cardiomyopathy (distinguishes from ATTR-CM/FAP)."
                ),
            },
            {
                "term": "ACys Amyloidosis (CST3) — Icelandic, Cerebrovascular, Young Stroke",
                "definition": (
                    "CST3 (cystatin C) Glu68Gln (p.E68Q) is the ONLY described pathogenic variant — an Icelandic founder. "
                    "Cystatin C is a secreted cysteine protease inhibitor. Glu68Gln destabilises the protein → "
                    "amyloid deposition in cerebral arterioles and leptomeningeal vessels → vessel wall fragility → "
                    "intracerebral and subarachnoid haemorrhage. "
                    "PRESENTATION: Young adults (20s-30s); recurrent cerebral haemorrhages → progressive cognitive "
                    "decline → dementia → death. Median death age 30 without management — most lethal hereditary amyloidosis "
                    "by age of death. "
                    "KEY DDx: NO renal involvement. NO peripheral neuropathy. Purely cerebrovascular. "
                    "TREATMENT: No amyloid-specific therapy. Strict blood pressure control is the ONLY effective intervention. "
                    "ABSOLUTE CONTRAINDICATION: anticoagulants — dramatically increase haemorrhage risk. "
                    "Serum cystatin C is elevated in carriers (abnormal secretion) — useful screening adjunct. "
                    "HCHWA-D (Dutch type, APP): Similar haemorrhagic amyloidosis but Dutch ancestry, APP mutation."
                ),
            },
            {
                "term": "Patisiran vs Inotersen vs Tafamidis — Mechanism Differences",
                "definition": (
                    "Three TTR-targeted therapies with distinct mechanisms: "
                    "PATISIRAN (siRNA, LNP-encapsulated): Targets ATTR mRNA in hepatocytes → RNA interference → "
                    "mRNA degradation → reduced TTR protein synthesis (~80-85% reduction). Approved ATTR-FAP. "
                    "Vutrisiran (second-generation subcutaneous siRNA) achieves same efficacy with quarterly dosing. "
                    "INOTERSEN (ASO, antisense oligonucleotide): Targets ATTR mRNA in hepatocytes → RNase-H cleavage → "
                    "mRNA degradation (~70-80% TTR reduction). Subcutaneous. Platelet count monitoring mandatory "
                    "(immune thrombocytopenia risk — black box warning). Approved ATTR-FAP. "
                    "TAFAMIDIS / ACORAMIDIS (kinetic stabilisers): Bind to the thyroxine-binding channel at the "
                    "tetramer dimer-dimer interface → kinetically stabilise the tetramer → prevent dissociation "
                    "into amyloidogenic monomers. They do NOT reduce TTR production but prevent fibril formation. "
                    "Approved for ATTR-CM (cardiomyopathy). Acoramidis (AG10) shows higher stabilisation than tafamidis. "
                    "KEY DIFFERENCE: siRNA/ASO work upstream (reduce TTR synthesis); stabilisers work downstream (prevent misfolding)."
                ),
            },
            {
                "term": "Congo Red Birefringence and Amyloid Diagnosis",
                "definition": (
                    "Congo red staining of tissue sections remains the gold-standard diagnostic test for amyloid. "
                    "Congo red dye intercalates into the periodic beta-sheet cross structure of amyloid fibrils. "
                    "Under polarised light, bound Congo red produces APPLE-GREEN BIREFRINGENCE — this is the "
                    "definitive diagnostic finding. Standard light microscopy shows salmon-pink staining. "
                    "SENSITIVITY: Dependent on amyloid load and section thickness; false-negatives occur with "
                    "small deposits. Thioflavin S/T fluorescence is more sensitive for small deposits. "
                    "SUBTYPE IDENTIFICATION: Congo red positivity identifies amyloid but does NOT specify the protein. "
                    "IMMUNOHISTOCHEMISTRY (IHC): Panel includes anti-TTR, anti-kappa, anti-lambda, anti-AA, anti-ApoA-I, "
                    "anti-fibrinogen — limited sensitivity/specificity for rare subtypes. "
                    "MASS SPECTROMETRY (GOLD STANDARD for subtyping): Laser-capture microdissection of Congo-red-positive "
                    "areas → LC-MS/MS peptide identification → unambiguous amyloid protein designation. "
                    "Replaced IHC as standard of care in specialised centres; essential for hereditary amyloidosis diagnosis."
                ),
            },
            {
                "term": "Serum Amyloid P Scintigraphy (SAP Scan)",
                "definition": (
                    "Serum amyloid P (SAP) is a pentraxin plasma protein that reversibly binds amyloid fibrils "
                    "in a calcium-dependent manner. SAP scintigraphy uses radiolabelled 123I-SAP to quantify and "
                    "localise total body amyloid deposits. "
                    "PROCEDURE: IV injection of 123I-SAP → whole-body SPECT or planar imaging at 24 hours. "
                    "Uptake in liver, spleen, kidneys, adrenals, bone marrow indicates amyloid at those sites. "
                    "ADVANTAGES: (1) Quantifies total amyloid burden non-invasively. (2) Serial scans track regression "
                    "or progression in response to treatment. (3) Identifies amyloid in sites not easily biopsied. "
                    "LIMITATIONS: Not widely available (specialised centres only: UK National Amyloidosis Centre). "
                    "Cardiac and cerebral amyloid incompletely detected (cardiac pool of blood-borne SAP obscures). "
                    "CLINICAL USE: TTR, AA, AApoAI, AFib, ALys amyloidosis — tracks visceral burden. "
                    "NIS scan (technetium pyrophosphate/DPD) is cardiac-specific for ATTR-CM (SAP scan is not cardiac-specific)."
                ),
            },
            {
                "term": "Genetic Amyloidosis Diagnostic Approach",
                "definition": (
                    "Step 1 — EXCLUDE AL AMYLOIDOSIS FIRST: Serum FLC assay (kappa/lambda ratio) + SPEP + UPEP + "
                    "bone marrow biopsy in any suspected systemic amyloidosis. AL is most common and most treatable. "
                    "Step 2 — TISSUE BIOPSY + AMYLOID TYPING: Abdominal fat pad aspirate (sensitivity ~70-85%); "
                    "target organ biopsy; Congo red stain + mass spectrometry proteomic typing. "
                    "Step 3 — TARGETED GENETIC PANEL: Once mass spectrometry identifies the protein, test the "
                    "corresponding gene (TTR, APOA1, APOA2, LYZ, GSN, FGA, CST3, B2M variants). "
                    "Step 4 — CASCADE FAMILY TESTING: All first-degree relatives of confirmed hereditary amyloidosis "
                    "index cases; pre-symptomatic diagnosis enables early treatment (especially TTR-FAP with patisiran). "
                    "PITFALLS: (1) Do not assume ATTR = wild-type without sequencing — pathogenic variant in 10-15% "
                    "of 'ATTR-CM' over age 60. (2) Multiple myeloma with coincidental ATTR: separate diseases. "
                    "(3) Negative FLC does not exclude hereditary amyloidosis — FLC is the AL screen, not universal."
                ),
            },
            {
                "term": "AApoAI Amyloidosis — Low HDL as Diagnostic Clue",
                "definition": (
                    "Apolipoprotein A-I (ApoA-I) is the major structural scaffold of HDL particles, activating LCAT "
                    "and facilitating reverse cholesterol transport. "
                    "LOW HDL IS PATHOGNOMONIC: Pathogenic ApoA-I variants impair HDL assembly → serum HDL-C is "
                    "markedly reduced (often <20 mg/dL). This is a cardinal bedside diagnostic clue — any patient "
                    "with unexplained renal amyloidosis or hepatomegaly + very low HDL should have APOA1 sequencing. "
                    "ORGANOTROPISM: Mutation position determines organ tropism. "
                    "N-terminal residues 26-107 (Leu64Pro, Leu75Pro): renal + hepatic amyloid. "
                    "Mid-region (Gly26Arg): neuropathic variant — peripheral nerve amyloid. "
                    "C-terminal (residues 170-178): cardiac and cutaneous. "
                    "No approved amyloid-specific therapy. Management is organ-directed (renal/liver transplant). "
                    "Differentiating from TTR: TTR has normal HDL, neuropathy + cardiac; AApoAI has low HDL, renal + hepatic dominant."
                ),
            },
            {
                "term": "AB2M Dialysis-Related Amyloidosis — Prevention Greater than Treatment",
                "definition": (
                    "Beta-2 microglobulin (β₂M, 12 kDa) is the MHC class I light chain, cleared by glomerular "
                    "filtration in normal kidneys. In ESRD, renal clearance is abolished → serum β₂M accumulates "
                    "50-fold above normal (>50 mg/L). At high concentrations + advanced glycation end-products + "
                    "collagen, β₂M forms amyloid fibrils in musculoskeletal tissues. "
                    "CARPAL TUNNEL SYNDROME: First and most common manifestation; bilateral; begins after 5-7 years "
                    "on dialysis; flexor tenosynovium amyloid compresses median nerve. "
                    "PREVENTION IS THE ONLY EFFECTIVE STRATEGY: "
                    "(1) High-flux dialysis membranes (polysulfone): superior β₂M clearance vs low-flux. "
                    "(2) Online haemodiafiltration (HDF): removes 40-60% more β₂M per session than high-flux HD. "
                    "(3) Kidney transplant: restores glomerular filtration → serum β₂M normalises within days → "
                    "existing deposits partially resorb over months-years. "
                    "TREATMENT ONCE ESTABLISHED: Carpal tunnel release (surgical decompression) — symptomatic relief; "
                    "amyloid recurs slowly. No pharmacological therapy clears established deposits effectively."
                ),
            },
            {
                "term": "Cascade Genetic Testing — Hereditary Amyloidosis Panel",
                "definition": (
                    "For all index cases: once hereditary amyloidosis subtype confirmed by mass spectrometry amyloid typing "
                    "and genetic sequencing, offer cascade testing to all first-degree relatives (50% risk — AD pattern). "
                    "Pre-symptomatic identification is clinically meaningful because: "
                    "(1) TTR-FAP: patisiran/inotersen started before axonal loss achieves near-normal quality of life; "
                    "delay beyond significant neuropathy = irreversible axonal loss. "
                    "(2) FGA: liver Tx before ESRD achieves renal recovery; after ESRD, outcome is worse. "
                    "(3) CST3 (Icelandic): blood pressure control started before first haemorrhage is lifesaving. "
                    "(4) B2M: high-flux/HDF from dialysis initiation prevents DRA in future ESRD patients. "
                    "Prenatal diagnosis: available for families with confirmed pathogenic variant. "
                    "Genetic counselling: mandatory before testing (psychological implications of learning carrier status "
                    "for a late-onset lethal disease such as CST3 require specialist support)."
                ),
            },
            {
                "term": "Amyloid Fibril Classification — AL vs AA vs ATTR vs Hereditary",
                "definition": (
                    "Amyloid fibrils are classified by the precursor protein: "
                    "AL (Amyloid Light chain): Immunoglobulin kappa or lambda free light chains. Plasma cell dyscrasia. "
                    "Most common systemic amyloidosis. Renal, cardiac, neuropathic, hepatic, soft tissue. "
                    "AA (Amyloid A): Serum amyloid A (acute-phase reactant). Chronic inflammation, TB, RA, IBD, FMF. "
                    "Renal predominant. "
                    "ATTRwt (wild-type transthyretin): Ageing TTR tetramer instability. Elderly males. Cardiomyopathy + carpal tunnel. "
                    "ATTRv (variant transthyretin): Hereditary TTR mutations. Neuropathy ± cardiomyopathy. "
                    "AApoAI: Apolipoprotein A-I variants. Renal, hepatic, cardiac, neuropathic by mutation site. "
                    "AApoAII: Apolipoprotein A-II frameshift. Renal only (Ostertag type). "
                    "ALys: Lysozyme variants. Renal + hepatic + GI + splenic (systemic visceral). "
                    "AGel: Gelsolin Asp187. Cranial neuropathy + lattice corneal dystrophy (Meretoja). "
                    "AFib: Fibrinogen alpha chain Glu526Val. Renal only. Liver Tx curative. "
                    "ACys: Cystatin C Glu68Gln. Cerebrovascular haemorrhage (Icelandic). "
                    "AB2M: Beta-2 microglobulin. Dialysis-related. Carpal tunnel + spondyloarthropathy."
                ),
            },
        ],
        "pharmacological_distinctions": [
            "PATISIRAN / VUTRISIRAN (siRNA): RNAi-mediated hepatic TTR mRNA knockdown — reduces TTR production ~80-85%; "
            "approved ATTR-FAP neuropathy; LNP-delivered IV (patisiran) or subcutaneous quarterly (vutrisiran)",
            "INOTERSEN (ASO): Antisense oligonucleotide targeting hepatic TTR mRNA via RNase-H; ~70-80% TTR reduction; "
            "subcutaneous weekly; PLATELET MONITORING MANDATORY — immune thrombocytopenia black-box warning",
            "TAFAMIDIS / ACORAMIDIS (kinetic stabilisers): Bind thyroxine-binding channel of TTR tetramer → prevent "
            "tetramer dissociation → prevent amyloidogenesis; approved ATTR-CM; do NOT reduce TTR production; "
            "acoramidis (AG10) achieves near-complete stabilisation vs ~50-60% for tafamidis",
            "TTR LIVER TRANSPLANT PARADOX: Removes mutant TTR source (hepatic) but cardiac deposits continue to grow "
            "on wild-type TTR template from donor liver — cardiac disease WORSENS; not recommended for ATTR-CM; "
            "may benefit pure FAP (polyneuropathy) variants with no or minimal cardiac involvement",
            "FGA LIVER TRANSPLANT CURATIVE: Liver is the ONLY source of amyloidogenic fibrinogen alpha chain — "
            "liver Tx eliminates amyloid substrate entirely; renal Tx alone FAILS (destroys allograft in 5-8 years); "
            "combined liver-kidney Tx is the preferred strategy in AFib amyloidosis",
            "CST3: NO approved amyloid therapy; strict blood pressure control (<130/80 mmHg) is the ONLY evidence-based "
            "intervention to reduce haemorrhage risk; anticoagulants and antiplatelets are ABSOLUTELY CONTRAINDICATED",
            "GSN (AGel): No amyloid-specific therapy; corneal transplant for corneal disease (recurrence in graft expected); "
            "nerve decompression for facial palsy-induced exposure keratopathy; SLT for dysphagia (CN IX/X involvement)",
            "B2M: Prevention by high-flux (polysulfone) dialysis membranes and online haemodiafiltration (HDF) — "
            "reduces serum β₂M 40-60% more than low-flux; kidney transplant eliminates DRA by restoring renal clearance; "
            "carpal tunnel decompression surgery provides symptomatic relief (amyloid recurs slowly)",
            "DOXYCYCLINE + TUDCA: Proposed amyloid fibril disruptor combination (doxycycline disrupts fibril structure; "
            "tauroursodeoxycholic acid TUDCA reduces ER stress/misfolding); used adjunctively in TTR and AL; "
            "evidence limited to small trials; not approved; used off-label in specialised centres",
        ],
    }
