#!/usr/bin/env python3
"""Vascular-Malformation-Atlas — Complete 8-Gene Hereditary Vascular Malformation Atlas
ENG      (Endoglin; 658 aa; 9q34.11; AD;
          HHT type 1 / Osler-Weber-Rendu syndrome;
          Pulmonary AVMs commonest serious manifestation — screen with contrast echo + CXR;
          Cerebral AVMs 10% — brain MRI at least once;
          GI telangiectasia → chronic iron-deficiency: IV iron PREFERRED over oral;
          BEVACIZUMAB (anti-VEGF) for severe epistaxis + hepatic AVMs;
          Most common HHT gene in northern European populations) ·
ACVRL1   (Activin receptor-like kinase 1 / ALK1; 503 aa; 12q13.13; AD;
          HHT type 2;
          Hepatic AVMs more common than in HHT1 — high-output cardiac failure;
          Pulmonary arterial hypertension 8% — early echocardiogram screen;
          BEVACIZUMAB effective, especially for hepatic shunting;
          Epistaxis typically less severe but hepatic disease more prominent) ·
SMAD4    (SMAD family member 4; 552 aa; 18q21.2; AD;
          Combined HHT + Juvenile Polyposis Syndrome (JP-HHT);
          SMAD4 TESTING FIRST whenever HHT + GI polyps (hamartomatous) coexist;
          Colorectal cancer risk 40% by age 40 — colonoscopy from age 15, every 1-3 years;
          Aortic root dilation 25% of carriers — annual echocardiogram;
          Gastric polyps + fundic gland: ANNUAL gastroscopy from diagnosis) ·
KRIT1    (KRIT1 / CCM1; 736 aa; 7q21.2; AD;
          Cerebral Cavernous Malformation type 1 (familial);
          POPCORN lesion on MRI (mixed-signal T1/T2 core + hemosiderin rim on GRE/SWI) pathognomonic;
          Autosomal dominant with 2-hit somatic mechanism — multiple lesions in familial form;
          Seizures 30-50%, focal deficits 25%, headache 30%, incidental 25%;
          Annual brain MRI; surgical resection for symptomatic or enlarging lesions;
          Most common familial CCM gene in northern European ancestry) ·
CCM2     (Malcavernin / CCM2; 380 aa; 7p13; AD;
          Cerebral Cavernous Malformation type 2;
          Clinically similar to CCM1; de novo variants 20%;
          Scaffolding protein linking KRIT1 to PDCD10 — loss destabilises entire CCM complex;
          Haemorrhage rate 2-3% per lesion per year;
          Conservative management if asymptomatic; annual MRI) ·
PDCD10   (Programmed cell death protein 10 / CCM3; 212 aa; 3q26.1; AD;
          Cerebral Cavernous Malformation type 3 — MOST SEVERE CCM subtype;
          Earlier onset (childhood), more lesions, faster growth, higher haemorrhage rate;
          FAMILIAL MENINGIOMAS in some PDCD10 families — if CCM + meningioma, test PDCD10 first;
          Cutaneous and spinal cavernomas more common than CCM1/2;
          Lower threshold for surgery given aggressive natural history) ·
TEK      (Tyrosine kinase with immunoglobulin and EGF homology domains / TIE2; 1124 aa; 9p21.2; AD / somatic mosaic;
          Venous Malformations (VM) — sporadic somatic mutations or germline AD;
          Blue, soft, compressible, non-pulsatile lesion; refills slowly after emptying (DDx from AVM);
          LOCALIZED INTRAVASCULAR COAGULOPATHY (LIC) — D-dimer elevated; thrombosis within VM;
          SIROLIMUS (mTOR inhibitor): reduces pain, volume, D-dimer — Level B evidence;
          Sclerotherapy (ethanol / polidocanol / STS) for accessible lesions;
          AVOID warfarin — increases LIC paradoxically in some VM) ·
RASA1    (RAS p21 protein activator 1; 1047 aa; 5q14.3; AD;
          CM-AVM: Capillary Malformation-Arteriovenous Malformation syndrome + PARKES WEBER syndrome;
          Pink-red, macular FAST-FLOW skin lesion (port-wine appearance but fast-flow on Doppler);
          PARKES WEBER: diffuse AVM + limb overgrowth + cutaneous flushing;
          DO NOT EMBOLIZE ALONE — AVMs recruit collaterals, worsen post-embolization;
          SIROLIMUS investigational — reduces flow and size in RASA1-AVM;
          Cascade testing: 1st-degree relatives of CM-AVM probands)
320-patient aggregate cohort (8 × 40, seeds 1382-1389)
"""

import random

SEED_BASE = 1382

VM_GENES = [
    # ── ENG — HHT type 1 ──
    {
        "gene": "ENG",
        "protein": "Endoglin",
        "alias": (
            "ENG; OMIM gene 131195; HHT1 #187300; "
            "9q34.11; 658 aa; ~68 kDa type I transmembrane glycoprotein; "
            "accessory receptor for TGF-β superfamily (BMP9/BMP10) — complexes with ALK1/ACVRL1 on endothelium; "
            "LOF → impaired angiogenesis/vascular remodelling → fragile telangiectasia and AVMs; "
            "Most common HHT gene worldwide — 50-60% of all HHT families; "
            "AD: one null allele sufficient; haploinsufficiency mechanism; "
            "Penetrance nearly complete by age 40 but variable expressivity even within families; "
            "Epistaxis (nosebleeds) in >95%, GI telangiectasia in 25-30%, PAVMs in 30-50%, "
            "hepatic AVMs in 30%, cerebral AVMs in 10%"
        ),
        "aa": "658 aa",
        "kDa": "~68 kDa",
        "locus": "9q34.11",
        "omim_gene": 131195,
        "omim_disease": 187300,
        "inheritance": "AD — haploinsufficiency; near-complete penetrance, variable expressivity",
        "gene_class": (
            "ENG encodes Endoglin, a co-receptor that partners with ALK1 (ACVRL1) to transduce BMP9/BMP10 "
            "signalling in vascular endothelium. This pathway is critical for arteriovenous specification: "
            "BMP9→ENG/ALK1→SMAD1/5/8 promotes quiescent 'arterial tone' in mature vessels. "
            "LOF creates dysfunctional endothelium that cannot maintain vessel wall integrity → compensatory "
            "angiogenesis via VEGF overstimulation → fragile telangiectasia → direct A-V shunting."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("Frameshift/nonsense → haploinsufficiency (most common ENG class)", 0.45),
            ("Missense (cysteine substitution or ECD domain disruption)", 0.25),
            ("Large deletion / intragenic rearrangement (MLPA-detected)", 0.15),
            ("Splice site variant → exon skipping → premature truncation", 0.15),
        ],
        "age_onset_years_range": (10, 40),
        "sex_ratio_M": 0.48,
        "rates": {
            "epistaxis": 0.97,
            "gi_telangiectasia": 0.30,
            "pulmonary_avm": 0.42,
            "hepatic_avm": 0.32,
            "cerebral_avm": 0.10,
            "stroke_or_brain_abscess": 0.12,
            "limb_overgrowth": 0.00,
            "cutaneous_cm": 0.05,
            "venous_malformation": 0.00,
        },
        "hallmarks": [
            "EPISTAXIS: recurrent, bilateral, begins in childhood (median age 12); worsens with age",
            "GI TELANGIECTASIA: small-bowel most common; iron-deficiency anaemia despite adequate diet",
            "PAVM: transthoracic contrast echo (bubble study) — gold standard screen; CXR misses 25%",
            "PAVM → paradoxical embolism: stroke, brain abscess; antibiotic prophylaxis for dental/GI procedures",
            "IV IRON PREFERRED: GI telangiectasia impairs oral iron absorption; IV iron ferric carboxymaltose",
            "BEVACIZUMAB (anti-VEGF): reduces epistaxis frequency and severity; IV 5 mg/kg q4-6 weeks",
            "PAVM embolization (coil/vascular plug): definitive treatment for PAVM ≥3 mm feeding artery",
            "CEREBRAL AVM: at least 1 MRI; risk of haemorrhage ~2% per year if unruptured",
        ],
        "treatment_alerts": [
            "IV IRON (ferric carboxymaltose): preferred over oral due to GI telangiectasia malabsorption",
            "BEVACIZUMAB: effective for severe/transfusion-dependent epistaxis and hepatic AVM high-output failure",
            "PAVM EMBOLIZATION: indicated for feeding artery ≥3 mm; prevents stroke and brain abscess",
            "ANTIBIOTIC PROPHYLAXIS: before dental / GI procedures — prevents brain abscess via PAVM shunt",
            "AVOID OESTROGEN-ONLY HRT: worsens telangiectasia; progesterone-containing preparations preferred",
            "SCREEN FIRST-DEGREE RELATIVES: contrast echo + clinical exam from age 8; MRI brain at least once",
        ],
        "organ_system": "vascular (telangiectasia / AVM: pulmonary, hepatic, cerebral, GI)",
        "primary_treatment": "IV iron + bevacizumab (medical); PAVM embolization (interventional); cascade screening",
    },

    # ── ACVRL1 — HHT type 2 ──
    {
        "gene": "ACVRL1",
        "protein": "Activin Receptor-Like Kinase 1 (ALK1)",
        "alias": (
            "ACVRL1; OMIM gene 601284; HHT2 #600376; "
            "12q13.13; 503 aa; ~57 kDa type I TGF-β receptor serine/threonine kinase; "
            "forms signalling complex with ENG + BMPR2 on endothelium; "
            "BMP9→ACVRL1→SMAD1/5/8 → arterial identity + vascular quiescence; "
            "LOF → same pathway as ENG but phenotypic emphasis differs: "
            "hepatic AVMs more common (>50%); pulmonary AVMs less common than HHT1; "
            "PAH in 8% — ACVRL1 mutations also cause familial PAH (FPAH); "
            "Spanish (Canary Islands) and French founder mutations described; "
            "20-30% of all HHT families"
        ),
        "aa": "503 aa",
        "kDa": "~57 kDa",
        "locus": "12q13.13",
        "omim_gene": 601284,
        "omim_disease": 600376,
        "inheritance": "AD — haploinsufficiency; allelic with FPAH",
        "gene_class": (
            "ACVRL1 (ALK1) is the type I receptor that directly binds BMP9 and BMP10 in partnership with ENG. "
            "Activated ALK1 phosphorylates SMAD1/5/8 → transcription of genes promoting arterial quiescence. "
            "In HHT2, LOF tips the balance toward excessive angiogenesis (unopposed SMAD2/3 via ALK5). "
            "Phenotypic differences from HHT1 likely reflect tissue-specific expression ratios of ENG vs ALK1."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("Missense (kinase domain or extracellular domain — e.g. p.Arg374Gln Canary Islands founder)", 0.40),
            ("Frameshift/nonsense → haploinsufficiency", 0.35),
            ("Splice site variant → aberrant transcript", 0.15),
            ("Large deletion (MLPA)", 0.10),
        ],
        "age_onset_years_range": (15, 50),
        "sex_ratio_M": 0.46,
        "rates": {
            "epistaxis": 0.92,
            "gi_telangiectasia": 0.25,
            "pulmonary_avm": 0.20,
            "hepatic_avm": 0.55,
            "cerebral_avm": 0.08,
            "stroke_or_brain_abscess": 0.06,
            "pulmonary_hypertension": 0.08,
            "limb_overgrowth": 0.00,
            "cutaneous_cm": 0.04,
            "venous_malformation": 0.00,
        },
        "hallmarks": [
            "HEPATIC AVM: high-output cardiac failure if large shunt; hepatic bruit + warm extremities",
            "PULMONARY HTN (PAH): occurs in 8% — ACVRL1 is a major PAH gene; early echo screen mandatory",
            "EPISTAXIS typically less severe than HHT1 but hepatic involvement more prominent",
            "BEVACIZUMAB: particularly effective for hepatic AVM-related high-output cardiac failure",
            "HEPATIC AVM management: medical first (diuretics + bevacizumab); AVOID hepatic artery embolization",
            "AVOID HEPATIC ARTERY EMBOLIZATION: risk of ischaemic hepatitis + biliary necrosis in hepatic HHT",
            "ECHOCARDIOGRAM: screen for PAH annually; if PAH, manage per FPAH guidelines",
            "ALLELIC with FPAH: ACVRL1 families can present as pulmonary arterial hypertension alone",
        ],
        "treatment_alerts": [
            "AVOID HEPATIC ARTERY EMBOLIZATION: biliary ischaemia + hepatic necrosis risk in hepatic AVM",
            "BEVACIZUMAB: first-line for symptomatic hepatic AVM with high-output failure — IV dosing",
            "PAH SCREEN: annual echo; if PAH, refer to PH specialist — endothelin antagonists + PDE5i",
            "IV IRON: as for HHT1 — oral absorption impaired by GI telangiectasia",
            "PAVM embolization: lower yield than HHT1 but indicated for ≥3 mm feeding artery",
            "LIVER TRANSPLANT: last resort for refractory hepatic HHT with cardiac failure",
        ],
        "organ_system": "vascular (telangiectasia / AVM: hepatic > pulmonary, GI, cerebral)",
        "primary_treatment": "Bevacizumab (hepatic/epistaxis); PAH drugs if PAH; IV iron; cascade screening",
    },

    # ── SMAD4 — HHT + Juvenile Polyposis combined ──
    {
        "gene": "SMAD4",
        "protein": "SMAD Family Member 4",
        "alias": (
            "SMAD4; OMIM gene 600993; JP-HHT #175050; JPS alone #174900; "
            "18q21.2; 552 aa; ~60 kDa MH1 + linker + MH2 domains; "
            "common intracellular mediator for both TGF-β (ALK5) and BMP (ALK1) signalling; "
            "MH1 binds DNA, MH2 mediates R-SMAD interaction; "
            "JP-HHT: combined Juvenile Polyposis (hamartomatous GI polyps) + HHT phenotype; "
            "Colorectal cancer: 40% by age 40 (highest GI cancer risk in polyposis syndromes after FAP); "
            "Gastric polyps (fundic gland type) in >50%; "
            "Aortic root dilation in 25% — distinct from Loeys-Dietz/Marfan; "
            "~15% of HHT families — SMAD4 screen mandatory if HHT + polyps"
        ),
        "aa": "552 aa",
        "kDa": "~60 kDa",
        "locus": "18q21.2",
        "omim_gene": 600993,
        "omim_disease": 175050,
        "inheritance": "AD — haploinsufficiency; tumour suppressor (LOH in polyps)",
        "gene_class": (
            "SMAD4 is the common mediator that R-SMADs (SMAD1/5/8 for BMP; SMAD2/3 for TGF-β) heterodimerize with "
            "before nuclear entry. In the vascular endothelium, loss of SMAD4 → impaired BMP9/ALK1 quiescence → "
            "HHT phenotype. In GI epithelium, SMAD4 LOH acts as a tumour suppressor lost in adenoma→carcinoma "
            "progression. The co-occurrence of HHT + JPS is essentially pathognomonic of a SMAD4 variant."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("Frameshift/nonsense → haploinsufficiency (MH2 domain — most common)", 0.45),
            ("Missense (MH2 domain — impairs R-SMAD binding)", 0.25),
            ("Large deletion including SMAD4 (18q MLPA)", 0.20),
            ("Splice site → aberrant MH1 or MH2 truncation", 0.10),
        ],
        "age_onset_years_range": (5, 35),
        "sex_ratio_M": 0.50,
        "rates": {
            "epistaxis": 0.85,
            "gi_telangiectasia": 0.25,
            "pulmonary_avm": 0.25,
            "hepatic_avm": 0.20,
            "cerebral_avm": 0.06,
            "gi_polyps_juvenile": 0.95,
            "colorectal_cancer": 0.40,
            "aortic_root_dilation": 0.25,
            "gastric_polyps": 0.55,
            "limb_overgrowth": 0.00,
        },
        "hallmarks": [
            "JP-HHT: SMAD4 FIRST if HHT + hamartomatous GI polyps — pathognomonic combination",
            "COLORECTAL CANCER RISK 40% by age 40: colonoscopy from age 15, every 1-3 years",
            "GASTRIC POLYPS: annual gastroscopy from diagnosis — fundic gland polyps → adenocarcinoma risk",
            "AORTIC ROOT DILATION 25%: annual echocardiogram — prophylactic surgery if >5.0 cm or rapid growth",
            "EPISTAXIS: managed as per HHT (bevacizumab, iron, nasal humidification)",
            "PAVM: screen as per HHT1 protocol (contrast echo + CXR); PAVM embolization if indicated",
            "SMAD4 TEST: mandatory if HHT + GI polyps — testing ENG/ACVRL1 first misses the diagnosis",
            "GI BLEEDING: may be from polyps (large, carpet-like) or telangiectasia",
        ],
        "treatment_alerts": [
            "SMAD4-FIRST RULE: if HHT phenotype + juvenile polyps, test SMAD4 before ENG/ACVRL1",
            "COLONOSCOPY from age 15: 1-3 year intervals; polypectomy for polyps >1 cm or symptomatic",
            "GASTROSCOPY ANNUALLY: gastric fundic gland polyps progress to cancer — annual scope mandatory",
            "ECHO ANNUALLY: aortic root dilation in 25%; prophylactic aortic surgery if >5.0 cm",
            "BEVACIZUMAB: as for HHT (epistaxis, hepatic AVM) — same dosing protocol",
            "GENETIC COUNSELLING: 50% inheritance risk; children need polyp surveillance from age 8",
        ],
        "organ_system": "vascular (HHT phenotype) + GI (polyposis + cancer) + aorta",
        "primary_treatment": "GI surveillance (colonoscopy/gastroscopy) + HHT protocol + aortic monitoring",
    },

    # ── KRIT1/CCM1 — Cerebral Cavernous Malformation type 1 ──
    {
        "gene": "KRIT1",
        "protein": "KRIT1 / CCM1 (Krev Interaction Trapped 1)",
        "alias": (
            "KRIT1; OMIM gene 604214; CCM1 #116860; "
            "7q21.2; 736 aa; ~92 kDa; FERM + ankyrin repeat domains; "
            "interacts with HEG1, RAP1, TLN1 → endothelial junction integrity + cell polarity; "
            "CCM complex (KRIT1-CCM2-PDCD10) stabilises VE-cadherin at endothelial cell junctions; "
            "LOF → increased endothelial permeability + iron deposition → haemosiderin ring; "
            "Most common familial CCM gene (CCM1): ~40-50% of familial CCM; "
            "AD with somatic 2nd hit — multiple lesions in germline carriers (vs single in sporadic); "
            "HISPANIC AMERICAN FOUNDER: p.Arg51Gln / c.742C>T (exon 4) — common in South/Central American ancestry"
        ),
        "aa": "736 aa",
        "kDa": "~92 kDa",
        "locus": "7q21.2",
        "omim_gene": 604214,
        "omim_disease": 116860,
        "inheritance": "AD with somatic 2nd hit (Knudson 2-hit) — haploinsufficiency not sufficient alone",
        "gene_class": (
            "KRIT1 is the membrane-proximal scaffold of the CCM signalling complex. It binds RAP1-GTP at "
            "cell-cell junctions, linking integrin signalling to junction stabilisation. KRIT1 also maintains "
            "RhoA-ROCK inhibition in endothelium — loss leads to RhoA activation, cytoskeletal contraction, "
            "increased permeability, and extravasation of red blood cells → haemosiderin deposition → "
            "the pathognomonic MRI 'popcorn lesion'."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("Frameshift/nonsense → haploinsufficiency + somatic 2nd hit", 0.50),
            ("Hispanic founder p.Arg51Gln (c.742C>T exon 4)", 0.20),
            ("Missense (FERM domain disruption)", 0.15),
            ("Large deletion (MLPA — chromosome 7q21 region)", 0.15),
        ],
        "age_onset_years_range": (20, 45),
        "sex_ratio_M": 0.50,
        "rates": {
            "seizures": 0.40,
            "focal_neurological_deficit": 0.25,
            "headache": 0.30,
            "incidental_mri_finding": 0.20,
            "intracerebral_haemorrhage": 0.15,
            "multiple_lesions": 0.80,
            "spinal_cavernoma": 0.10,
            "retinal_cavernoma": 0.05,
            "skin_cavernoma": 0.03,
            "limb_overgrowth": 0.00,
        },
        "hallmarks": [
            "POPCORN LESION: mixed-signal T1+T2 core (blood products of different ages) + haemosiderin rim on GRE/SWI",
            "MULTIPLE LESIONS: familial CCM1 → dozens to hundreds of cavernomas (vs 1-2 in sporadic)",
            "HAEMORRHAGE RATE: ~2-3% per lesion per year — higher with brainstem location",
            "SEIZURES (40%): first-line AED; consider surgery if drug-resistant or lesion accessible",
            "ANNUAL MRI: 3T MRI with SWI/GRE sequence most sensitive; count lesions; assess for growth",
            "SURGICAL RESECTION: for symptomatic, enlarging, or accessible lesions; stereotactic guidance",
            "HISPANIC FOUNDER p.Arg51Gln: screen in South/Central American ancestry with familial CCM",
            "RETINAL CAVERNOMA: fundoscopic exam + OCT — usually asymptomatic but can cause visual loss",
        ],
        "treatment_alerts": [
            "ANNUAL MRI with SWI/GRE: most sensitive sequence; count all lesions; track growth",
            "AVOID ANTICOAGULATION unless mandatory: increases haemorrhage risk in cavernomas",
            "ASPIRIN: acceptable if coronary/stroke indication — low-dose not clearly harmful",
            "SURGERY: symptomatic brainstem CCM requires expert centre; approach angle critical",
            "AED: seizure control with standard agents; levetiracetam commonly used post-haemorrhage",
            "CASCADE TESTING: 1st-degree relatives with MRI (SWI/GRE); Hispanic ancestry → test for p.Arg51Gln",
        ],
        "organ_system": "CNS vascular (cerebral, spinal, retinal cavernous malformations)",
        "primary_treatment": "Annual MRI surveillance; surgical resection for symptomatic/accessible lesions; AEDs for seizures",
    },

    # ── CCM2 — Cerebral Cavernous Malformation type 2 ──
    {
        "gene": "CCM2",
        "protein": "Malcavernin (CCM2)",
        "alias": (
            "CCM2; OMIM gene 607929; CCM2 #603284; "
            "7p13; 380 aa; ~48 kDa; "
            "contains PTB domain (phosphotyrosine-binding) + HBD (harmonin N-like) + C-terminal CCM3-binding; "
            "central scaffold of CCM complex: bridges KRIT1 and PDCD10; "
            "also interacts with MEKK3 → MAP3K kinase suppression in endothelium; "
            "De novo pathogenic variants ~20% — important for genetic counselling; "
            "20-30% of familial CCM; "
            "Clinically indistinguishable from CCM1 — same surveillance protocol; "
            "Mutation spectrum: mostly truncating (frameshift + nonsense)"
        ),
        "aa": "380 aa",
        "kDa": "~48 kDa",
        "locus": "7p13",
        "omim_gene": 607929,
        "omim_disease": 603284,
        "inheritance": "AD with somatic 2nd hit; de novo in ~20%",
        "gene_class": (
            "CCM2/Malcavernin acts as the central adaptor of the trimeric CCM complex. Its PTB domain binds "
            "phosphorylated tyrosines on HEG1 (Heart of Glass), anchoring the complex to endothelial cell junctions. "
            "Its C-terminus binds PDCD10/CCM3 directly. Without CCM2, neither KRIT1 nor PDCD10 is properly "
            "recruited to junctions → full loss of CCM complex function even with intact KRIT1 and PDCD10 alleles."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("Frameshift/nonsense → haploinsufficiency + 2nd hit (most common)", 0.55),
            ("Splice site variant → exon skipping", 0.20),
            ("Missense (PTB domain disruption)", 0.15),
            ("De novo variant (no family history)", 0.10),
        ],
        "age_onset_years_range": (25, 50),
        "sex_ratio_M": 0.50,
        "rates": {
            "seizures": 0.38,
            "focal_neurological_deficit": 0.22,
            "headache": 0.28,
            "incidental_mri_finding": 0.25,
            "intracerebral_haemorrhage": 0.13,
            "multiple_lesions": 0.78,
            "spinal_cavernoma": 0.08,
            "retinal_cavernoma": 0.04,
            "skin_cavernoma": 0.02,
            "de_novo_no_family_hx": 0.20,
        },
        "hallmarks": [
            "DE NOVO 20%: no family history in 1-in-5 CCM2 patients — MRI siblings still recommended",
            "CLINICALLY IDENTICAL TO CCM1: same MRI features (popcorn + haemosiderin), same management",
            "MULTIPLE LESIONS: familial pattern; dozens of cavernomas on SWI/GRE",
            "HAEMORRHAGE RATE: ~2% per lesion per year; brainstem lesions carry higher risk",
            "MEKK3 PATHWAY: CCM2 loss activates MEKK3→KLF2/4 → endothelial-mesenchymal transition in endothelium",
            "ANNUAL MRI surveillance: 3T with SWI/GRE; count and compare lesions longitudinally",
            "SURGICAL INDICATIONS: same as CCM1 — symptomatic, enlarging, or accessible single dominant lesion",
            "GENETIC COUNSELLING: 50% risk to offspring even if one parent asymptomatic",
        ],
        "treatment_alerts": [
            "DE NOVO 20%: request MRI for all 1st-degree relatives regardless of family history",
            "SAME PROTOCOL AS CCM1: annual MRI (SWI/GRE), avoid anticoagulation, AED for seizures",
            "MEKK3 INHIBITION: fasudil (ROCK inhibitor) investigational in CCM — not standard of care yet",
            "STATINS: pre-clinical data on CCM stabilisation — no RCT evidence; avoid withholding if otherwise indicated",
            "PREGNANCY: case reports of accelerated haemorrhage in pregnancy — close MRI follow-up",
            "CASCADE TESTING: 1st-degree relatives; positive → MRI; negative → discharged",
        ],
        "organ_system": "CNS vascular (cerebral, spinal, retinal cavernous malformations)",
        "primary_treatment": "Annual MRI surveillance (SWI/GRE); surgical resection for symptomatic lesions; cascade testing",
    },

    # ── PDCD10/CCM3 — Most severe CCM ──
    {
        "gene": "PDCD10",
        "protein": "Programmed Cell Death Protein 10 (CCM3)",
        "alias": (
            "PDCD10; OMIM gene 609118; CCM3 #603285; "
            "3q26.1; 212 aa; ~25 kDa; FFAT motif + homodimerisation domain + C-terminal CCM2-binding; "
            "interacts with STK24/25 (Sterile-20 kinases) → cytoskeletal organisation; "
            "MOST SEVERE CCM SUBTYPE: earlier onset, more lesions, faster haemorrhage rate; "
            "MENINGIOMA ASSOCIATION: familial spinal meningiomas in some PDCD10 pedigrees; "
            "CCM3 + CEREBRAL CAVERNOMA + MENINGIOMA = test PDCD10 first; "
            "Cutaneous cavernomas and spinal cavernomas more common than CCM1/2; "
            "~20% of familial CCM"
        ),
        "aa": "212 aa",
        "kDa": "~25 kDa",
        "locus": "3q26.1",
        "omim_gene": 609118,
        "omim_disease": 603285,
        "inheritance": "AD with somatic 2nd hit; most severe natural history of the three CCM genes",
        "gene_class": (
            "PDCD10/CCM3 is the smallest CCM complex subunit. It bridges CCM2 and STK24/STK25 kinases, "
            "coordinating cytoskeletal dynamics and endothelial-mesenchymal transition. PDCD10 also regulates "
            "apoptosis in endothelial cells via its interaction with Bcl-2 and MST kinases. Loss of PDCD10 "
            "produces the most dysregulated endothelial phenotype — explaining the earlier onset, greater lesion "
            "burden, and higher haemorrhage rate compared to CCM1 and CCM2."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("Frameshift/nonsense → haploinsufficiency + somatic 2nd hit", 0.55),
            ("Splice site variant → aberrant transcript", 0.20),
            ("Missense (homodimerisation or CCM2-binding domain)", 0.15),
            ("Large deletion (3q26 region — MLPA)", 0.10),
        ],
        "age_onset_years_range": (10, 35),
        "sex_ratio_M": 0.50,
        "rates": {
            "seizures": 0.50,
            "focal_neurological_deficit": 0.35,
            "headache": 0.35,
            "incidental_mri_finding": 0.12,
            "intracerebral_haemorrhage": 0.25,
            "multiple_lesions": 0.92,
            "spinal_cavernoma": 0.20,
            "retinal_cavernoma": 0.10,
            "skin_cavernoma": 0.10,
            "familial_meningioma": 0.10,
        },
        "hallmarks": [
            "MOST SEVERE CCM: earlier onset (childhood-adolescence), more lesions, faster haemorrhage rate",
            "MENINGIOMA ASSOCIATION: familial spinal meningiomas — CCM + meningioma = test PDCD10 FIRST",
            "CUTANEOUS CAVERNOMAS: clinically visible vascular lesions on skin — more frequent in PDCD10",
            "SPINAL CAVERNOMAS 20%: thoracic cord most common; myelopathy risk; spinal MRI mandatory",
            "HIGH HAEMORRHAGE BURDEN: multiple bleeds by 4th decade if untreated; aggressive surgical threshold",
            "LOWER SURGERY THRESHOLD: given aggressive natural history vs CCM1/CCM2",
            "STK24/25 PATHWAY: ongoing research into kinase inhibition as disease-modifying therapy",
            "RETINAL CAVERNOMA 10%: fundoscopic + OCT examination on diagnosis",
        ],
        "treatment_alerts": [
            "LOWER SURGICAL THRESHOLD: PDCD10 has worse natural history — earlier surgery for accessible lesions",
            "SPINAL MRI MANDATORY: 20% spinal cavernomas — complete spine MRI at diagnosis",
            "MENINGIOMA SCREEN: if family history of meningioma + CCM, test PDCD10 + spine MRI",
            "PAEDIATRIC ONSET: children with multiple CCM on MRI — test PDCD10 preferentially",
            "AVOID ANTICOAGULATION: as per CCM1/CCM2 — increased haemorrhage risk",
            "CASCADE TESTING: 1st-degree relatives; children need MRI from age 5-8",
        ],
        "organ_system": "CNS vascular (cerebral + spinal + retinal + cutaneous cavernous malformations)",
        "primary_treatment": "MRI surveillance (more frequent: 6-monthly for active disease); earlier surgery; cascade testing",
    },

    # ── TEK/TIE2 — Venous Malformations ──
    {
        "gene": "TEK",
        "protein": "Tyrosine Kinase with Immunoglobulin and EGF Homology Domains (TIE2)",
        "alias": (
            "TEK; OMIM gene 600221; Venous Malformation #600195; "
            "9p21.2; 1124 aa; ~140 kDa; RTK with IgG-like extracellular domain + FNIII + kinase; "
            "TIE2 receptor tyrosine kinase — activated by angiopoietins (Ang-1 quiescent, Ang-2 disruptive); "
            "Germline LOF → loss of TIE2 signalling → defective pericyte recruitment → enlarged blood-filled venous channels; "
            "Sporadic VM: often biallelic (germline + somatic gain-of-function in TIE2 or PIK3CA); "
            "LOCALIZED INTRAVASCULAR COAGULOPATHY (LIC) in large VM: D-dimer elevated, fibrinogen low; "
            "SIROLIMUS (mTOR inhibitor): robust evidence for pain reduction + volume decrease + D-dimer normalisation; "
            "Soft, blue, compressible, non-pulsatile mass — refills slowly (DDx AVM: pulsatile, warm, fast)"
        ),
        "aa": "1124 aa",
        "kDa": "~140 kDa",
        "locus": "9p21.2",
        "omim_gene": 600221,
        "omim_disease": 600195,
        "inheritance": "AD (germline LOF) or somatic mosaic (somatic GOF); sporadic mostly biallelic somatic",
        "gene_class": (
            "TIE2/TEK is the major regulator of vascular stability and quiescence via Angiopoietin signalling. "
            "Ang-1 (from perivascular cells) activates TIE2 → PI3K/AKT → vascular maturation and pericyte retention. "
            "Germline TIE2 LOF impairs pericyte recruitment → smooth muscle-poor, thin-walled venous channels that "
            "expand progressively. The stagnant blood in large VMs triggers coagulation activation → LIC → "
            "painful thrombosis and phleboliths. TIE2 mutations also activate the same mTOR pathway that sirolimus targets."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("Germline TEK LOF (AD familial VM) + somatic 2nd hit", 0.30),
            ("Somatic TEK GOF (p.Leu914Phe or p.Arg849Trp — most common somatic)", 0.40),
            ("Biallelic somatic (germline LOF + somatic PIK3CA GOF)", 0.20),
            ("Sporadic de novo germline", 0.10),
        ],
        "age_onset_years_range": (0, 20),
        "sex_ratio_M": 0.48,
        "rates": {
            "compressible_blue_mass": 1.00,
            "phleboliths_on_xray": 0.50,
            "localized_intravascular_coagulopathy": 0.65,
            "pain_at_rest_or_activity": 0.70,
            "skin_involvement": 0.55,
            "deep_tissue": 0.70,
            "bone_involvement": 0.15,
            "gi_vm": 0.10,
            "airway_vm": 0.08,
            "arteriovenous_shunting": 0.00,
        },
        "hallmarks": [
            "BLUE SOFT COMPRESSIBLE NON-PULSATILE: classical VM — refills slowly on compression (DDx AVM)",
            "PHLEBOLITHS: calcified thrombi within VM — pathognomonic on plain X-ray or CT",
            "LIC: D-dimer elevated in large VM; fibrinogen low; risk of DIC if major surgery without pre-treatment",
            "SIROLIMUS: Level B evidence — reduces pain, volume, D-dimer, phleboliths; 0.8 mg/m² BD oral",
            "SCLEROTHERAPY: ethanol (most effective but highest risk), STS, polidocanol — for accessible lesions",
            "PRE-OPERATIVE MANAGEMENT: LMWH 5-7 days before surgery to prevent perioperative DIC",
            "AVOID WARFARIN: paradoxical worsening of LIC in some VM patients — LMWH preferred if anticoagulation needed",
            "MRI: T2/STIR high signal ('lightbulb' on T2); no flow voids (DDx AVM); phleboliths on CT",
        ],
        "treatment_alerts": [
            "SIROLIMUS 0.8 mg/m² BD: monitor trough levels (5-15 ng/mL); reduces pain + volume + D-dimer",
            "PRE-SURGICAL LMWH 5-7 DAYS: prevents DIC from LIC — mandatory before any surgical or sclerotherapy procedure",
            "AVOID WARFARIN: use LMWH if anticoagulation needed — warfarin paradoxically activates LIC in some VM",
            "SCLEROTHERAPY: ethanol most effective but risk of skin necrosis + nerve injury; STS/polidocanol safer",
            "AIRWAY VM: urgent management if tongue/pharynx — risk of life-threatening obstruction",
            "D-DIMER MONITORING: track LIC activity; rising D-dimer signals enlargement or thrombosis",
        ],
        "organ_system": "venous vascular (soft tissue, skin, GI, bone, airway venous malformations)",
        "primary_treatment": "Sirolimus (medical); sclerotherapy (interventional); LMWH pre-procedure; D-dimer monitoring",
    },

    # ── RASA1 — CM-AVM + Parkes Weber ──
    {
        "gene": "RASA1",
        "protein": "RAS p21 Protein Activator 1 (p120RasGAP)",
        "alias": (
            "RASA1; OMIM gene 139150; CM-AVM1 #608354; Parkes Weber syndrome #608355; "
            "5q14.3; 1047 aa; ~120 kDa; SH2-SH3-SH2-PH-C2-RasGAP domain; "
            "RasGAP: accelerates GTP hydrolysis on RAS → keeps RAS-MAPK pathway off in endothelium; "
            "LOF → constitutive RAS/MAPK/PI3K activation → VEGF hypersensitivity → abnormal AV connections; "
            "CM-AVM: multiple capillary malformations (pink/red skin lesions) + underlying AVMs; "
            "PARKES WEBER: CM-AVM + limb overgrowth + diffuse AVM of an extremity; "
            "DO NOT EMBOLIZE ALONE: AVMs recruit collaterals post-embolization — worsen outcome; "
            "SIROLIMUS investigational — mTOR downstream of RAS/PI3K"
        ),
        "aa": "1047 aa",
        "kDa": "~120 kDa",
        "locus": "5q14.3",
        "omim_gene": 139150,
        "omim_disease": 608354,
        "inheritance": "AD — haploinsufficiency; somatic 2nd hit in AVM nidus",
        "gene_class": (
            "RASA1 encodes p120RasGAP, the major negative regulator of Ras GTPase activity in endothelial cells. "
            "When ligands activate RTKs (e.g. VEGFR), Ras→GTP accumulates transiently; RASA1 terminates this by "
            "hydrolysing GTP→GDP. LOF → persistent Ras-GTP → MAPK + PI3K/mTOR over-activation → excessive "
            "endothelial proliferation, impaired arteriovenous specification, and formation of multiple AVMs. "
            "The 2-hit mechanism (germline + somatic) explains why not every capillary malformation contains an AVM."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("Frameshift/nonsense → haploinsufficiency + somatic 2nd hit", 0.50),
            ("Splice site variant → premature termination", 0.20),
            ("Missense (RasGAP catalytic domain disruption)", 0.20),
            ("Large deletion (5q14 — MLPA)", 0.10),
        ],
        "age_onset_years_range": (0, 20),
        "sex_ratio_M": 0.48,
        "rates": {
            "pink_red_capillary_malformation": 1.00,
            "fast_flow_on_doppler": 0.70,
            "underlying_avm": 0.60,
            "parkes_weber_limb_overgrowth": 0.25,
            "intracranial_avm": 0.08,
            "spinal_avm": 0.05,
            "high_output_cardiac_failure": 0.10,
            "haemorrhage_from_avm": 0.08,
            "arteriovenous_shunting": 0.70,
            "limb_length_discrepancy": 0.25,
        },
        "hallmarks": [
            "PINK-RED FAST-FLOW SKIN LESION: looks like port-wine stain but warm + pulsatile on Doppler",
            "MULTIPLE CM: many pink/red macular lesions on skin — DDx Sturge-Weber (somatic GNAQ, single lesion)",
            "AVM NIDUS: below or adjacent to CM on MRI/angiogram — DSA gold standard for AVM characterisation",
            "PARKES WEBER: limb AVM + overgrowth + hyperhidrosis + cutaneous flushing of affected limb",
            "DO NOT EMBOLIZE ALONE: nidus recruitment post-embolization worsens AVM; combined surgical resection",
            "SIROLIMUS: investigational for RASA1-AVM — reduces RAS/mTOR-driven angiogenesis",
            "HIGH-OUTPUT CARDIAC FAILURE: large extremity AVM → cardiac volume overload; echocardiogram",
            "CASCADE TESTING: 1st-degree relatives; full-body skin exam + Doppler if CM present",
        ],
        "treatment_alerts": [
            "NEVER EMBOLIZE ALONE: always plan combined embolization + surgical resection for AVM — nidus recruits",
            "DOPPLER MANDATORY: any pink/red skin lesion in CM-AVM family must have Doppler — fast-flow = AVM",
            "SIROLIMUS: investigational; evidence from PIK3CA-related overgrowth (same pathway) supports use",
            "PARKES WEBER: orthopaedic management (epiphysiodesis) for limb-length discrepancy",
            "INTRACRANIAL AVM: neurovascular MDT; same Spetzler-Martin grading approach; haemorrhage risk ~2%/year",
            "CARDIAC ECHO: for large-limb AVM to assess output; high-output failure needs prompt AVM treatment",
        ],
        "organ_system": "vascular (capillary malformation + AVM: skin, extremity, intracranial, spinal)",
        "primary_treatment": "Combined embolization + surgery (AVM); sirolimus investigational; orthopaedic for PWS; Doppler surveillance",
    },
]


def _make_patients(gene_data: dict) -> list[dict]:
    rng = random.Random(gene_data["seed"])
    gene   = gene_data["gene"]
    rates  = gene_data["rates"]
    ptx    = []

    for pid in range(1, gene_data["n_patients"] + 1):
        age_lo, age_hi = gene_data["age_onset_years_range"]
        sex = "M" if rng.random() < gene_data["sex_ratio_M"] else "F"
        onset_age = rng.randint(age_lo, age_hi)

        etio_choices = [e[0] for e in gene_data["etiologies"]]
        etio_probs   = [e[1] for e in gene_data["etiologies"]]
        etiology = rng.choices(etio_choices, weights=etio_probs, k=1)[0]

        clinical = {k: (rng.random() < v) for k, v in rates.items()}

        ptx.append({
            "id": f"{gene}-{pid:03d}",
            "gene": gene,
            "sex": sex,
            "onset_age_years": onset_age,
            "etiology": etiology,
            **clinical,
        })
    return ptx


def _aggregate(patients: list[dict], rate_keys: list[str]) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    return {k: round(sum(1 for p in patients if p.get(k, False)) / n * 100, 1)
            for k in rate_keys}


ALL_PATIENTS = []
for _gd in VM_GENES:
    ALL_PATIENTS.extend(_make_patients(_gd))


def get_overview() -> dict:
    n = len(ALL_PATIENTS)
    rate_keys = list(VM_GENES[0]["rates"].keys())
    agg = _aggregate(ALL_PATIENTS, rate_keys)
    return {
        "atlas": "Hereditary Vascular Malformation Atlas",
        "subtitle": "Complete 8-Gene Hereditary Vascular Malformation & Telangiectasia Reference",
        "total_patients": n,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes": [g["gene"] for g in VM_GENES],
        "diseases": {
            "ENG":    "HHT1 (Osler-Weber-Rendu) — Pulmonary/Hepatic/Cerebral AVMs, IV Iron, Bevacizumab",
            "ACVRL1": "HHT2 (ALK1) — Hepatic AVM > Pulmonary AVM, PAH 8%, Avoid Hepatic Embolization",
            "SMAD4":  "HHT3+JPS — Polyps + HHT combined, Colorectal Cancer 40% by 40, Aortic Root Dilation",
            "KRIT1":  "CCM1 — Cerebral Cavernous Malformation, Popcorn Lesion, Hispanic Founder Arg51Gln",
            "CCM2":   "CCM2 — Malcavernin, De Novo 20%, CCM1-identical phenotype, Central CCM Complex Scaffold",
            "PDCD10": "CCM3 — Most Severe CCM, Meningioma Association, Cutaneous/Spinal Cavernomas",
            "TEK":    "Venous Malformation (TIE2) — LIC, Sirolimus, Avoid Warfarin, Pre-Surgical LMWH",
            "RASA1":  "CM-AVM/Parkes Weber — Fast-Flow CM, Never Embolize Alone, Limb Overgrowth",
        },
        "aggregate_stats": {
            k: agg.get(k, 0)
            for k in rate_keys
        },
        "top_alerts": [
            "ENG: IV iron PREFERRED over oral (GI telangiectasia impairs absorption)",
            "ACVRL1: AVOID hepatic artery embolization — biliary ischaemia risk",
            "SMAD4: TEST FIRST if HHT + GI polyps; colonoscopy from age 15; aortic echo annually",
            "KRIT1: HISPANIC FOUNDER p.Arg51Gln — test in South/Central American ancestry",
            "PDCD10: MOST SEVERE CCM — earlier surgery threshold; spinal MRI mandatory",
            "CCM2: DE NOVO 20% — MRI 1st-degree relatives even without family history",
            "TEK: SIROLIMUS reduces pain + volume + D-dimer; LMWH pre-procedure; AVOID warfarin",
            "RASA1: NEVER embolize alone — combined embolization + resection mandatory for AVM",
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for gd in VM_GENES:
        pts = [p for p in ALL_PATIENTS if p["gene"] == gd["gene"]]
        rate_keys = list(gd["rates"].keys())
        agg = _aggregate(pts, rate_keys)
        result[gd["gene"]] = {
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "organ_system": gd["organ_system"],
            "primary_treatment": gd["primary_treatment"],
            "stats": agg,
            "hallmarks": gd["hallmarks"],
            "treatment_alerts": gd["treatment_alerts"],
            "etiology_distribution": [
                {"etiology": e[0], "fraction": e[1]} for e in gd["etiologies"]
            ],
        }
    return result


def get_definitions() -> dict:
    return {
        "atlas": "Hereditary Vascular Malformation Atlas",
        "classification": {
            "HHT_subtypes": {
                "HHT1_ENG": "Hereditary Haemorrhagic Telangiectasia type 1 — endoglin; PAVM dominant",
                "HHT2_ACVRL1": "HHT type 2 — ALK1; hepatic AVM dominant; PAH 8%",
                "JPHT_SMAD4": "Juvenile Polyposis-HHT combined — SMAD4; polyps + cancer + HHT + aorta",
            },
            "CCM_subtypes": {
                "CCM1_KRIT1": "Cerebral Cavernous Malformation 1 — most common familial; Hispanic founder",
                "CCM2_CCM2": "CCM2 — malcavernin; 20% de novo; clinically identical to CCM1",
                "CCM3_PDCD10": "CCM3 — most severe; meningioma association; earlier onset",
            },
            "other_vascular": {
                "VM_TEK": "Venous Malformation — TIE2/TEK; LIC; sirolimus",
                "CM_AVM_RASA1": "Capillary Malformation-AVM — RASA1; Parkes Weber; never embolize alone",
            },
        },
        "key_diagnostic_rules": {
            "HHT_CURACAO": (
                "HHT Curaçao Criteria (3/4 = definite): (1) epistaxis recurrent spontaneous, "
                "(2) telangiectasia (lips/fingers/oral mucosa), (3) visceral AVM (pulmonary/hepatic/cerebral/GI), "
                "(4) family history 1st-degree"
            ),
            "IV_IRON_RULE": (
                "HHT patients with iron deficiency: IV iron PREFERRED over oral — GI telangiectasia "
                "impairs iron absorption; ferric carboxymaltose single-dose most convenient"
            ),
            "SMAD4_FIRST": (
                "HHT + GI hamartomatous polyps → test SMAD4 FIRST before ENG/ACVRL1 — "
                "otherwise the diagnosis is missed or delayed"
            ),
            "HEPATIC_AVM_AVOID_EMBOLIZATION": (
                "ACVRL1/HHT2 hepatic AVMs: AVOID hepatic artery embolization — "
                "risk of ischaemic hepatitis, biliary necrosis, acute liver failure; bevacizumab first"
            ),
            "CCM_POPCORN": (
                "CCM on MRI: POPCORN LESION — mixed T1/T2 signal (blood products of different ages) "
                "surrounded by haemosiderin rim (dark on GRE/SWI); SWI most sensitive sequence"
            ),
            "PDCD10_MENINGIOMA": (
                "CCM + familial meningioma → test PDCD10 first — PDCD10 families can present with "
                "cerebral cavernomas AND multiple spinal meningiomas"
            ),
            "VM_DDX_AVM": (
                "VM vs AVM clinical DDx: VM = blue, soft, compressible, non-pulsatile, refills slowly; "
                "AVM = warm, pulsatile, bruit present, fast-flow on Doppler"
            ),
            "VM_LIC": (
                "Large VM → Localised Intravascular Coagulopathy (LIC): D-dimer elevated, fibrinogen low; "
                "LMWH 5-7 days pre-procedure; monitor D-dimer; AVOID warfarin in VM"
            ),
            "SIROLIMUS_VM": (
                "VM (TEK): sirolimus 0.8 mg/m² BD oral — reduces pain, volume, D-dimer; "
                "Level B evidence from Vase Kids trial; trough 5-15 ng/mL"
            ),
            "RASA1_NEVER_EMBOLIZE": (
                "CM-AVM / RASA1: NEVER embolize an AVM alone — nidus recruits new feeding vessels post-embolization; "
                "always plan combined approach: embolization + surgical resection"
            ),
            "CASCADE_TESTING": (
                "All 8 genes: 50% risk per 1st-degree relative; genetic testing + MRI/contrast echo "
                "as indicated; children from age 8 (HHT) or age 5-8 (CCM/VM)"
            ),
        },
        "treatment_hierarchy": {
            "HHT": [
                "1. Bevacizumab (anti-VEGF): severe epistaxis, hepatic AVM — IV 5 mg/kg q4-6 weeks",
                "2. IV iron: ferric carboxymaltose for GI iron-deficiency (oral poorly absorbed)",
                "3. PAVM embolization: feeding artery ≥3 mm — coil or vascular plug",
                "4. Laser/electrocautery: nasal telangiectasia (Young's procedure)",
                "5. Liver transplant (last resort): refractory HHT-hepatic AVM with cardiac failure",
            ],
            "CCM": [
                "1. Annual MRI (SWI/GRE) surveillance — baseline + annual",
                "2. AED: seizure control — levetiracetam or valproate",
                "3. Surgery: symptomatic or enlarging lesion — stereotactic guidance",
                "4. Avoid anticoagulation unless mandatory",
                "5. PDCD10/CCM3: lower surgical threshold (more aggressive disease)",
            ],
            "VM": [
                "1. Sirolimus 0.8 mg/m² BD: pain reduction + volume decrease + D-dimer normalisation",
                "2. Sclerotherapy: ethanol (most effective) / STS / polidocanol",
                "3. LMWH pre-procedure (5-7 days): prevent DIC from LIC",
                "4. Surgery: combined with sclerotherapy for large/symptomatic VM",
                "5. Avoid warfarin in VM-LIC",
            ],
            "CM_AVM": [
                "1. Doppler imaging: all capillary malformations in RASA1 family — fast-flow = AVM",
                "2. DSA: gold standard AVM characterisation before treatment",
                "3. Combined embolization + surgical resection: NEVER embolize alone",
                "4. Sirolimus: investigational; downregulates RAS/mTOR-driven angiogenesis",
                "5. Orthopaedic: epiphysiodesis for Parkes Weber limb-length discrepancy",
            ],
        },
        "genes_summary": {g["gene"]: g["alias"] for g in VM_GENES},
    }
