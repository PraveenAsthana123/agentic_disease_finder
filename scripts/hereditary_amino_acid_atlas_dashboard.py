#!/usr/bin/env python3
"""Hereditary-Amino-Acid-Atlas — Complete 8-Gene Hereditary Aminoacidopathy Atlas
PAH     (phenylalanine hydroxylase; 452 aa; 12q23.2; AR;
         PKU — phenylketonuria; most common aminoacidopathy;
         Phe-restricted diet mandatory lifelong;
         BH4 (sapropterin) responsive ~30% of classical PKU alleles;
         pegvaliase (enzyme substitution) for refractory adult PKU;
         seed SEED_BASE+0) .
CBS     (cystathionine beta-synthase; 551 aa; 21q22.3; AR;
         Classical homocystinuria;
         B6 (pyridoxine) responsive ~50% — mandatory 3-month trial;
         thromboembolic risk PATHOGNOMONIC — lifelong anticoagulation in B6-non-responders;
         ectopia lentis (downward subluxation DDx Marfan = upward);
         seed SEED_BASE+1) .
BCKDHA  (branched-chain alpha-keto acid dehydrogenase E1-alpha; 445 aa; 19q13.2; AR;
         MSUD type 1A — maple syrup urine disease;
         alloisoleucine in plasma = PATHOGNOMONIC biomarker;
         leucine most neurotoxic BCAA — acute leucine crises → cerebral oedema;
         maple syrup odour urine/cerumen;
         Mennonite founder p.Tyr393Asn — 1 in 380;
         seed SEED_BASE+2) .
FAH     (fumarylacetoacetase; 419 aa; 15q25.1; AR;
         Tyrosinemia type 1 (HT1);
         succinylacetone in urine/plasma = PATHOGNOMONIC;
         nitisinone (NTBC) + low Phe/Tyr diet = standard of care;
         HCC risk 10-40x (pre-NTBC era) → AFP surveillance mandatory;
         renal Fanconi syndrome;
         seed SEED_BASE+3) .
HGD     (homogentisate 1,2-dioxygenase; 445 aa; 3q13.33; AR;
         Alkaptonuria (AKU);
         urine darkens on standing PATHOGNOMONIC;
         ochronosis — dark pigment deposits in cartilage, sclerae, tendons;
         nitisinone (NTBC) approved (SONIA-2 trial 2019);
         no acute metabolic crisis;
         seed SEED_BASE+4) .
GLDC    (glycine decarboxylase; 1020 aa; 9p24.1; AR;
         Nonketotic hyperglycinaemia (NKH) / glycine encephalopathy;
         CSF:plasma glycine ratio >0.08 = PATHOGNOMONIC;
         burst-suppression EEG in neonatal form;
         sodium benzoate (glycine sink) + dextromethorphan (NMDA antagonist);
         no cure — severe neurodevelopmental outcome in classic neonatal form;
         seed SEED_BASE+5) .
OAT     (ornithine aminotransferase; 439 aa; 10q26.13; AR;
         Gyrate atrophy of choroid and retina;
         plasma ornithine >10× normal;
         progressive chorioretinal degeneration → tunnel vision from age 20-30s;
         B6 (pyridoxine) responsive in ~5%;
         arginine-restricted diet reduces ornithine; creatine supplementation;
         seed SEED_BASE+6) .
TAT     (tyrosine aminotransferase; 454 aa; 16q22.2; AR;
         Tyrosinemia type 2 — Richner-Hanhart syndrome;
         palmar/plantar keratosis + pseudodendritic keratitis (TRIAD);
         plasma tyrosine >1000 µmol/L;
         low Phe/Tyr diet corrects ALL features — corneal lesions resolve within weeks;
         no hepatocellular involvement (contrast HT1);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1822-1829)
"""

import random

SEED_BASE = 1822

AA_GENES = [
    # -- PAH -- PKU / Phenylketonuria ------------------------------------------
    {
        "gene": "PAH",
        "protein": (
            "PAH -- 12q23.2 AR -- Phenylalanine-Hydroxylase-452aa -- "
            "PKU-Phenylketonuria-Most-Common-Aminoacidopathy -- "
            "Phe-Restricted-Diet-MANDATORY-Lifelong -- "
            "BH4-Sapropterin-Responsive-30pct-Classical-PKU -- "
            "Pegvaliase-Enzyme-Substitution-Refractory-Adult-PKU -- "
            "NBS-DBS-Universal-Screening-Heel-Prick"
        ),
        "alias": (
            "PAH (phenylalanine hydroxylase); OMIM gene 612349; "
            "PKU (phenylketonuria) OMIM 261600; HPA (hyperphenylalaninaemia). "
            "12q23.2; 452 aa; ~52 kDa; hepatic cytoplasmic enzyme; autosomal recessive. "
            "FUNCTION: PAH hydroxylates L-phenylalanine (Phe) to L-tyrosine (Tyr) "
            "using tetrahydrobiopterin (BH4) as cofactor and O2 as co-substrate. "
            "This is the rate-limiting step of phenylalanine catabolism. "
            "Without PAH: Phe accumulates → phenylpyruvate, phenylacetate, phenyllactate "
            "(alternative metabolites detectable in urine — phenylketonuria namesake). "
            "PATHOPHYSIOLOGY: "
            "Phe competitively inhibits aromatic amino acid transporters at the blood-brain barrier; "
            "large neutral amino acid (LNAA) transporter is saturated by Phe → "
            "reduced brain uptake of Tyr, Trp, Phe → reduced dopamine, serotonin, "
            "norepinephrine synthesis; myelin synthesis also impaired; "
            "classic untreated PKU: severe intellectual disability (IQ <50), "
            "seizures, autism-spectrum behaviours, fair hair/skin (melanin requires Tyr), "
            "musty odour (phenylacetate). "
            "CLASSIFICATION (blood Phe on unrestricted diet): "
            "Classic PKU: Phe >1200 µmol/L (>20 mg/dL); "
            "Mild PKU: Phe 600-1200 µmol/L; "
            "Mild HPA: Phe 120-600 µmol/L (Phe:Tyr ratio >3 on NBS); "
            "Tyr < normal in PKU (cannot synthesise Tyr from Phe). "
            "NBS: heel-prick day 3-5; blood Phe (MS/MS) — gold standard; "
            "all children born in high-income countries screened since ~1963 (Guthrie test). "
            "TREATMENT: "
            "1. Phe-restricted diet: Phe intake restricted to individual tolerance "
            "(typical 200-400 mg Phe/day; normal diet ~2500 mg/day); "
            "protein mainly from Phe-free amino acid supplements; "
            "Tyr supplementation required; diet is lifelong (not just childhood). "
            "2. BH4 (sapropterin — Kuvan™): "
            "BH4-responsive variants carry at least one mild (missense) allele with residual PAH activity; "
            "~30% of classical PKU and ~80% of mild PKU are BH4-responsive; "
            "sapropterin 10-20 mg/kg/day; Phe loading test (blood Phe −30% at 24h) confirms response; "
            "reduces dietary Phe restriction burden; not a cure. "
            "3. Pegvaliase (Palynziq™): subcutaneous enzyme substitution therapy; "
            "phenylalanine ammonia lyase conjugated to PEGylated carrier; "
            "converts Phe → trans-cinnamic acid + ammonia (does NOT require BH4); "
            "FDA 2018 for adults with uncontrolled blood Phe >600 µmol/L; "
            "injection site reactions common; arthralgia; "
            "achieves normal Phe in >50% — most effective therapy for refractory PKU. "
            "4. Maternal PKU: women with PKU must achieve blood Phe <360 µmol/L "
            "before conception and throughout pregnancy — maternal hyperphenylalaninaemia "
            "causes congenital heart disease, microcephaly, intellectual disability in the fetus "
            "(fetus is PAH heterozygous, NOT at risk from own genotype — maternal Phe is the teratogen). "
            "KEY CLINICAL FACTS: "
            "Most common aminoacidopathy in high-income countries (~1:10,000 live births); "
            "PAH >1000 known disease-causing variants in PAHdb; "
            "genotype-phenotype: two mild alleles = mild HPA; one mild + one severe = intermediate; "
            "two severe (null) alleles = classic PKU; "
            "PKU does NOT cause cirrhosis or renal disease (no toxic hepatic metabolite); "
            "BH4 deficiency (DHPR, PTPS, GTPCH, PCD deficiencies) causes HPA but also neurotransmitter "
            "deficiency — neurotransmitter precursors (DOPA/5HTP) mandatory alongside Phe restriction."
        ),
        "age_of_onset": "Neonatal (NBS day 3-5)",
        "inheritance": "AR",
        "locus": "12q23.2",
        "protein_size": "452 aa",
        "key_biomarker": "Blood phenylalanine >120 µmol/L (NBS)",
        "pathognomonic": "Phe >1200 µmol/L + musty odour untreated",
        "treatment": "Phe-restricted diet; sapropterin (BH4); pegvaliase",
        "critical_flags": [
            "MATERNAL-PKU — achieve Phe <360 µmol/L before conception",
            "LIFELONG-DIET — not just childhood",
            "BH4-TRIAL-MANDATORY — 10-20mg/kg for 30 days before declaring non-responsive",
            "NBS-UNIVERSAL — heel-prick day 3-5",
            "PHE-RESTRICTED — not protein-restricted (Phe-free AA supplements provide protein)",
            "TYR-SUPPLEMENTATION — Tyr becomes essential in PKU",
        ],
    },
    # -- CBS -- Classical Homocystinuria ----------------------------------------
    {
        "gene": "CBS",
        "protein": (
            "CBS -- 21q22.3 AR -- Cystathionine-Beta-Synthase-551aa -- "
            "Classical-Homocystinuria-HCU -- "
            "B6-Pyridoxine-Responsive-50pct-3-Month-Trial-Mandatory -- "
            "Ectopia-Lentis-Downward-DDx-Marfan-Upward -- "
            "Thromboembolic-Risk-PATHOGNOMONIC-VTE-Arterial -- "
            "Betaine-Methionine-Restriction-B6-Non-Responders"
        ),
        "alias": (
            "CBS (cystathionine beta-synthase); OMIM gene 613381; "
            "Classical homocystinuria (HCU) OMIM 236200. "
            "21q22.3; 551 aa; ~63 kDa; pyridoxal 5-phosphate (PLP) dependent; "
            "cytoplasmic; hepatic and other tissues; autosomal recessive. "
            "FUNCTION: CBS catalyses the first committed step of the transsulfuration pathway: "
            "homocysteine + serine → cystathionine (using PLP/B6 as cofactor). "
            "Cystathionine is then cleaved by cystathionase (CTH) to cysteine. "
            "In CBS deficiency: homocysteine cannot enter transsulfuration; "
            "homocysteine accumulates; remethylation pathway saturated → "
            "methionine accumulates (elevated plasma methionine a key NBS marker); "
            "homocysteine re-exported → plasma total homocysteine (tHcy) >100 µmol/L. "
            "PATHOPHYSIOLOGY: "
            "High homocysteine causes: "
            "1. Endothelial damage → premature atherosclerosis + thrombosis; "
            "2. Cross-linking of fibrillin-1 → ectopia lentis (downward lens subluxation); "
            "3. Skeletal: Marfanoid habitus, scoliosis, osteoporosis, genu valgum; "
            "4. CNS: intellectual disability (~50% without treatment), psychiatric disorder, seizures; "
            "5. Thromboembolism: DVT, PE, arterial stroke, coronary artery disease — "
            "MOST COMMON CAUSE OF DEATH; events can occur in childhood. "
            "CLINICAL FEATURES: "
            "Ectopia lentis (lens subluxation downward — KEY DDx: Marfan/FBN1 = upward); "
            "Marfanoid habitus; malar flush; fine fair hair; thin skin with mottled complexion; "
            "vascular occlusions at any age; intellectual disability if untreated; "
            "psychiatric features: personality disorder, anxiety, schizophrenia-like psychosis. "
            "NBS: elevated methionine on MS/MS; confirmation with plasma tHcy + amino acids; "
            "CBS gene sequencing. "
            "TREATMENT: "
            "B6 (pyridoxine): 150-750 mg/day (PLP-cofactor — improves residual enzyme activity); "
            "3-month trial mandatory before classifying non-responsive; "
            "~50% of cases are B6-responsive (at least one missense allele with residual activity); "
            "B6-responsive patients: pyridoxine alone + supplemental B12/folate (remethylation support); "
            "B6-non-responsive patients: "
            "Methionine-restricted + cystine-supplemented diet (low-Met amino acid formula); "
            "betaine (Cystadane™): N,N,N-trimethylglycine → donates methyl group "
            "for homocysteine remethylation → re-converts Hcy to methionine (raises Met but lowers Hcy); "
            "folate + B12 supplementation. "
            "Anticoagulation: aspirin or warfarin in B6-non-responders (high thrombotic risk); "
            "peri-operative anticoagulation mandatory (surgery significantly raises thrombotic risk). "
            "KEY CLINICAL FACTS: "
            "Lens dislocation distinguishes HCU from Marfan: direction (down vs up), "
            "plus elevated plasma tHcy in HCU; "
            "methionine-restricted diet ≠ protein-restricted diet — "
            "Met-free amino acid formula provides amino acids; "
            "betaine paradox: lowers Hcy but raises methionine — "
            "cerebral oedema reported if methionine >1000 µmol/L; "
            "adult untreated HCU: 50% have had a thromboembolic event by age 30; "
            "most common inherited disorder of sulfur amino acid metabolism."
        ),
        "age_of_onset": "Infancy-childhood; NBS detects early",
        "inheritance": "AR",
        "locus": "21q22.3",
        "protein_size": "551 aa",
        "key_biomarker": "Plasma total homocysteine >100 µmol/L + elevated methionine",
        "pathognomonic": "Ectopia lentis (downward) + tHcy >100 µmol/L",
        "treatment": "B6 trial; methionine restriction; betaine; anticoagulation",
        "critical_flags": [
            "B6-TRIAL-3-MONTHS-MANDATORY — before classifying non-responsive",
            "ECTOPIA-LENTIS-DOWNWARD — DDx Marfan (upward/temporal)",
            "THROMBOEMBOLISM-LEADING-CAUSE-DEATH — peri-op anticoagulation mandatory",
            "BETAINE-CEREBRAL-OEDEMA — if methionine >1000 µmol/L",
            "B12-FOLATE-REMETHYLATION-SUPPORT — all patients",
            "CYSTEINE-CONDITIONALLY-ESSENTIAL — supplement in Met-restricted diet",
        ],
    },
    # -- BCKDHA -- MSUD / Maple Syrup Urine Disease ----------------------------
    {
        "gene": "BCKDHA",
        "protein": (
            "BCKDHA -- 19q13.2 AR -- BCAA-Dehydrogenase-E1alpha-445aa -- "
            "MSUD-Type-1A-Maple-Syrup-Urine-Disease -- "
            "Alloisoleucine-Plasma-PATHOGNOMONIC-Biomarker -- "
            "Leucine-Most-Neurotoxic-Cerebral-Oedema-Crisis -- "
            "Mennonite-Founder-pTyr393Asn-1in380 -- "
            "Liver-Transplant-Metabolic-Cure"
        ),
        "alias": (
            "BCKDHA (branched-chain alpha-keto acid dehydrogenase E1-alpha subunit); "
            "OMIM gene 608348; MSUD type 1A OMIM 248600. "
            "19q13.2; 445 aa; ~46 kDa; mitochondrial matrix; PLP-dependent; autosomal recessive. "
            "FUNCTION: BCKDHA encodes the alpha subunit of BCKDH E1 component (a thiamine-dependent "
            "decarboxylase). The BCKDH complex (E1+E2+E3) catalyses the second step of "
            "branched-chain amino acid (BCAA = leucine, isoleucine, valine) catabolism: "
            "oxidative decarboxylation of the respective alpha-ketoacids (alpha-KIC, alpha-KMV, alpha-KIV). "
            "BCKDHA and BCKDHB encode E1 alpha and beta; DBT encodes E2 (dihydrolipoamide acyltransferase); "
            "DLD encodes E3. Mutations in any component cause MSUD. "
            "In BCKDH deficiency: BCAAs and their keto acids accumulate; "
            "keto acids detectable by DNPH test (dinitrophenylhydrazine = yellow precipitate); "
            "maple syrup odour from sotolone (keto acid derivative). "
            "PATHOPHYSIOLOGY: "
            "Leucine is the primary neurotoxin: high Leu competitively inhibits "
            "LNAA transport at BBB → reduced brain Tyr, Trp, Phe → neurotransmitter depletion; "
            "leucine itself is directly toxic to glial cells; "
            "keto acids inhibit mitochondrial function; "
            "accumulation of alloisoleucine: a stereoisomer of isoleucine produced only when "
            "alpha-KMV cannot be decarboxylated → "
            "plasma alloisoleucine >5 µmol/L = PATHOGNOMONIC for BCKDH deficiency. "
            "CLINICAL PRESENTATION: "
            "Classic MSUD (most severe, BCKDHA typically): "
            "encephalopathy onset day 4-7 (BCAAs accumulate as protein catabolism rises after birth); "
            "poor feeding, lethargy, stereotyped cycling movements, opisthotonos; "
            "sweet/maple syrup odour in cerumen AND urine; "
            "if untreated: coma, cerebral oedema, respiratory failure, death within weeks; "
            "survivors untreated: severe intellectual disability. "
            "Intermediate MSUD: milder, less common. "
            "Thiamine-responsive MSUD: E2 (DBT) mutations; thiamine 10-300 mg/day improves. "
            "DIAGNOSIS: "
            "NBS: elevated leucine + isoleucine + valine (Leu>Ile>>Val); "
            "confirm: plasma amino acids (alloisoleucine >5 µmol/L = diagnostic); "
            "DNPH urine test positive; organic acids: branched-chain keto acids (qualitative screen). "
            "TREATMENT: "
            "Acute crisis: eliminate BCAAs from diet completely for 24-48h; "
            "high-calorie IV dextrose + lipid (suppress catabolism); "
            "isoleucine + valine supplementation (prevent depletion; leucine-free formula); "
            "leucine target blood <200 µmol/L (neonatal) / <300 µmol/L (stable); "
            "haemodialysis for severe hyperleucinaemia; "
            "Chronic: BCAA-restricted + BCAA-free amino acid formula; "
            "frequent meals (prevent catabolism); "
            "thiamine trial 10-300 mg/day (regardless of variant — small proportion responsive); "
            "sick-day protocols critical (infection raises BCAAs rapidly); "
            "Liver transplantation: curative metabolically — transplanted liver provides enough "
            "BCKDH activity to maintain near-normal BCAA levels on an unrestricted diet; "
            "neurological damage already sustained is NOT reversed; "
            "transplant does not prevent low-level CNS vulnerability. "
            "KEY CLINICAL FACTS: "
            "Alloisoleucine is the single most reliable acute diagnostic marker; "
            "maple syrup odour: check cerumen (earwax) — often smelled before urine; "
            "Mennonite population: pTyr393Asn (c.1179C>A) in BCKDHA — carrier frequency 1:10; "
            "leucine (not isoleucine or valine) drives acute encephalopathy — "
            "target leucine specifically in acute crisis; "
            "infection is the most common precipitant of acute crisis in known MSUD."
        ),
        "age_of_onset": "Neonatal day 4-7",
        "inheritance": "AR",
        "locus": "19q13.2",
        "protein_size": "445 aa",
        "key_biomarker": "Plasma alloisoleucine >5 µmol/L (PATHOGNOMONIC)",
        "pathognomonic": "Alloisoleucine >5 µmol/L + maple syrup odour cerumen/urine",
        "treatment": "BCAA restriction; leucine-free formula; liver transplant (metabolic cure)",
        "critical_flags": [
            "ALLOISOLEUCINE-PATHOGNOMONIC — >5 µmol/L diagnostic",
            "LEUCINE-MOST-NEUROTOXIC — target leucine <200 µmol/L acutely",
            "MAPLE-SYRUP-CERUMEN — check earwax early presentation",
            "LIVER-TRANSPLANT-METABOLIC-CURE — neurological damage NOT reversed",
            "THIAMINE-TRIAL-ALWAYS — regardless of variant subtype",
            "SICK-DAY-PROTOCOL-CRITICAL — infection precipitates acute crisis",
            "MENNONITE-FOUNDER — pTyr393Asn BCKDHA 1:380 prevalence",
        ],
    },
    # -- FAH -- Tyrosinemia Type 1 (HT1) ----------------------------------------
    {
        "gene": "FAH",
        "protein": (
            "FAH -- 15q25.1 AR -- Fumarylacetoacetase-419aa -- "
            "Tyrosinemia-Type-1-HT1-Most-Severe-Tyrosinaemia -- "
            "Succinylacetone-Urine-Plasma-PATHOGNOMONIC -- "
            "Nitisinone-NTBC-Standard-of-Care-1992 -- "
            "HCC-Risk-AFP-Surveillance-Mandatory -- "
            "Renal-Fanconi-Syndrome-Rickets"
        ),
        "alias": (
            "FAH (fumarylacetoacetase); OMIM gene 613871; "
            "Tyrosinemia type 1 (HT1) OMIM 276700. "
            "15q25.1; 419 aa; ~46 kDa; cytoplasmic; hepatic; autosomal recessive. "
            "FUNCTION: FAH catalyses the last step of tyrosine catabolism: "
            "fumarylacetoacetate (FAA) → fumarate + acetoacetate. "
            "Without FAH: fumarylacetoacetate and maleylacetoacetate accumulate; "
            "spontaneous cyclisation → succinylacetone (SA) = the critical toxic metabolite; "
            "SA is a potent inhibitor of porphobilinogen synthase (ALAD) "
            "→ ALA accumulates → porphyria-like neurological crises; "
            "SA also inhibits DNA repair → hepatocellular carcinoma risk; "
            "FAA alkylates proteins and DNA → direct cellular damage. "
            "PATHOPHYSIOLOGY: "
            "Hepatic accumulation of FAA/SA → hepatocyte necrosis → acute hepatic failure; "
            "chronic SA → cirrhosis → HCC (HCC risk 37-60x pre-nitisinone era); "
            "renal tubule accumulation → Fanconi syndrome (phosphaturia, glucosuria, aminoaciduria) "
            "→ renal tubular acidosis → rickets (hypophosphataemic); "
            "ALA accumulation (SA blocks ALAD) → acute neurological crises mimicking AIP "
            "(acute intermittent porphyria): severe abdominal pain, peripheral neuropathy, "
            "respiratory failure — a porphyria-like crisis WITHOUT elevated PBG "
            "(porphobilinogen) — this distinguishes HT1 from AIP. "
            "CLINICAL PRESENTATION: "
            "Neonatal/infantile (most common): acute hepatic failure in first months; "
            "jaundice, coagulopathy, hypoglycaemia, ascites; "
            "cabbage-like odour (methionine metabolites); "
            "Chronic form (later onset, slower liver disease): "
            "rickets, hepatosplenomegaly, portal hypertension; "
            "Porphyria-like crises: acute abdominal pain, self-mutilation (pain), "
            "peripheral neuropathy, respiratory failure. "
            "KEY DIAGNOSTIC MARKER: "
            "Succinylacetone (SA) in urine or dried blood spot — "
            "PATHOGNOMONIC for HT1 (not found in any other disorder); "
            "SA on newborn screening now incorporated in many programmes (NBS MS/MS); "
            "plasma tyrosine elevated (but HT2, HT3 also have elevated Tyr — "
            "SA distinguishes HT1 from all others); "
            "FAH gene sequencing; "
            "liver biopsy only if SA negative and diagnosis uncertain. "
            "TREATMENT: "
            "1. Nitisinone (NTBC, Orfadin™): "
            "Inhibits 4-hydroxyphenylpyruvate dioxygenase (HPPD), two steps UPSTREAM of FAH; "
            "blocks production of FAA, MAA, and succinylacetone; "
            "dose 1-2 mg/kg/day; "
            "SA normalises within days; liver function stabilises rapidly; "
            "started immediately at diagnosis — before dietary modification; "
            "2. Low Phe/Tyr diet: "
            "NTBC raises plasma tyrosine (blocked catabolism); "
            "tyrosine is a LNAA toxic to eye and brain at >1000 µmol/L; "
            "low-Phe + low-Tyr amino acid formula required; "
            "3. AFP surveillance: "
            "AFP every 3-6 months; "
            "rising AFP despite NTBC → liver transplant indication (HCC concern); "
            "liver transplant curative for HT1 — restored FAH activity; "
            "4. Coagulation support acutely (FFP, vitamin K); "
            "5. Renal tubular acidosis: sodium bicarbonate, phosphate, vitamin D. "
            "KEY CLINICAL FACTS: "
            "NTBC was discovered as a herbicide (inhibits HPPD in plants); "
            "HT1 pre-NTBC: liver transplant by age 2 in most; "
            "HT1 post-NTBC: most patients avoid transplant but lifelong AFP monitoring; "
            "SA blocks ALAD → porphyria-like crisis but PBG normal "
            "(critical DDx from AIP which has elevated PBG and normal SA); "
            "founder mutations: Saguenay-Lac-Saint-Jean Quebec (IVS12+5G>A) — 1:1846; "
            "Scandinavian (p.Pro261Leu). "
        ),
        "age_of_onset": "Neonatal-infancy (acute form); childhood (chronic)",
        "inheritance": "AR",
        "locus": "15q25.1",
        "protein_size": "419 aa",
        "key_biomarker": "Succinylacetone in urine/plasma (PATHOGNOMONIC)",
        "pathognomonic": "Succinylacetone + hepatic failure; porphyria crisis + normal PBG",
        "treatment": "Nitisinone (NTBC) + low Phe/Tyr diet; AFP surveillance; liver transplant",
        "critical_flags": [
            "SUCCINYLACETONE-PATHOGNOMONIC — only found in HT1",
            "NTBC-START-IMMEDIATELY — before diet change",
            "AFP-EVERY-3-6-MONTHS — rising AFP = transplant evaluation",
            "PORPHYRIA-CRISIS-NORMAL-PBG — DDx from AIP (AIP has elevated PBG)",
            "TYROSINE-LOW-DIET-WHILE-ON-NTBC — raised Tyr from NTBC causes eye/brain toxicity",
            "LIVER-TRANSPLANT-CURATIVE — FAH activity restored",
            "SAGUENAY-FOUNDER — IVS12+5G>A Quebec 1:1846",
        ],
    },
    # -- HGD -- Alkaptonuria ---------------------------------------------------
    {
        "gene": "HGD",
        "protein": (
            "HGD -- 3q13.33 AR -- Homogentisate-1,2-Dioxygenase-445aa -- "
            "Alkaptonuria-AKU -- "
            "Urine-Darkens-on-Standing-PATHOGNOMONIC -- "
            "Ochronosis-Dark-Pigment-Cartilage-Sclerae-Tendons -- "
            "Nitisinone-NTBC-SONIA-2-Trial-2019 -- "
            "No-Acute-Metabolic-Crisis-Degenerative"
        ),
        "alias": (
            "HGD (homogentisate 1,2-dioxygenase); OMIM gene 607474; "
            "Alkaptonuria (AKU) OMIM 203500. "
            "3q13.33; 445 aa; ~52 kDa; homohexamer; cytoplasmic; hepatic; autosomal recessive. "
            "FUNCTION: HGD catalyses the ring-opening step of tyrosine catabolism: "
            "homogentisate (HGA) + O2 → maleylacetoacetate. "
            "Without HGD: HGA accumulates; excreted in urine; "
            "HGA oxidises in alkaline conditions → benzoquinone acetic acid (BQA) → "
            "dark brown/black pigment polymer (alkapton = 'greedy for alkali'). "
            "BQA binds to collagen in connective tissues → ochronosis "
            "(ochra = yellow → dark brown-black pigmentation seen on pathology). "
            "PATHOPHYSIOLOGY: "
            "HGA is relatively non-toxic acutely; "
            "chronic accumulation: HGA-derived polymers deposit in "
            "articular cartilage, intervertebral discs, tendons, ligaments, heart valves, "
            "ear cartilage, sclerae, skin; "
            "cartilage deposition → brittle, structurally weakened cartilage → "
            "severe degenerative joint disease (ochronotic arthropathy); "
            "spine: ochronotic spondyloarthropathy (fused vertebrae, disc calcification) — "
            "may look like ankylosing spondylitis on X-ray; "
            "cardiac: aortic valve stenosis/regurgitation; "
            "urolithiasis (kidney/prostate stones, dark urine stones). "
            "CLINICAL PRESENTATION: "
            "Dark urine at birth (nappies stain — parents may notice); "
            "grey-black pigmentation of sclerae (pinguecula) — visible from 20-30s; "
            "ear cartilage: grey-blue tinge visible; "
            "skin: dark patches over nose, malar area, axillae; "
            "joint disease: large joint arthropathy from 30s-40s; "
            "spine: severe ochronotic spondylitis; "
            "cardiovascular: aortic stenosis, mitral valve disease. "
            "DIAGNOSIS: "
            "Urine left to stand → darkens on oxidation in alkaline urine "
            "(addition of NaOH accelerates darkening) = PATHOGNOMONIC; "
            "urine HGA quantitation by GC-MS or HPLC; "
            "HGD gene sequencing; "
            "urinary HGA >1 g/day (normal <20 mg/day); "
            "NOT detected on routine NBS (no acylcarnitine/amino acid abnormality on MS/MS). "
            "TREATMENT: "
            "Nitisinone (NTBC): "
            "inhibits HPPD (upstream of HGD) → blocks HGA production; "
            "SONIA-2 trial (2018, Nat Med 2019): nitisinone 2 mg/day × 4 years; "
            "HGA excretion reduced 95%; ochronosis progression significantly slowed; "
            "nitisinone raises plasma tyrosine → low-Tyr diet supplementation recommended; "
            "approved by EMA 2020 (Orfadin™ 2 mg/day for AKU); "
            "Supportive: NSAIDs for joint pain; joint replacement surgery; "
            "cardiac valve surveillance by echo (aortic valve); "
            "vitamin C (ascorbic acid): antioxidant — theoretical benefit "
            "(prevents HGA oxidation); used historically; "
            "KEY CLINICAL FACTS: "
            "AKU is rare (~1:250,000 in most populations; 1:19,000 in Slovakia); "
            "no acute metabolic crisis — purely degenerative (unlike PKU, MSUD, HT1); "
            "ochronosis on histology: Prussian-blue-positive deposits in tissue; "
            "sclerae pigmentation: triangular grey patches temporal to corneoscleral limbus; "
            "Alexander the Great may have had AKU (historical dark urine reports); "
            "alkaptonuria (first described by Garrod 1902) — one of the original 'inborn errors of metabolism'; "
            "AKU is NOT associated with intellectual disability (no BBB-crossing toxic metabolite)."
        ),
        "age_of_onset": "Birth (dark urine); symptoms from 20-30s",
        "inheritance": "AR",
        "locus": "3q13.33",
        "protein_size": "445 aa",
        "key_biomarker": "Urine HGA >1 g/day; urine darkens on standing",
        "pathognomonic": "Urine darkens on standing + ochronotic pigmentation (grey sclerae)",
        "treatment": "Nitisinone (NTBC) 2 mg/day (EMA 2020); supportive joint care",
        "critical_flags": [
            "URINE-DARKENS-ON-STANDING-PATHOGNOMONIC — alkali (NaOH) accelerates",
            "NO-ACUTE-METABOLIC-CRISIS — purely degenerative",
            "NTBC-EMA-2020 — 2 mg/day reduces HGA 95% (SONIA-2)",
            "LOW-TYR-DIET-ON-NTBC — raised Tyr from blocked catabolism",
            "AORTIC-VALVE-SURVEILLANCE — echo monitoring",
            "NBS-MISSES-AKU — no abnormal acylcarnitines/amino acids on MS/MS",
            "GARROD-FIRST-INBORN-ERROR — historical landmark disease 1902",
        ],
    },
    # -- GLDC -- Nonketotic Hyperglycinaemia (NKH) / Glycine Encephalopathy ----
    {
        "gene": "GLDC",
        "protein": (
            "GLDC -- 9p24.1 AR -- Glycine-Decarboxylase-P-Protein-1020aa -- "
            "Nonketotic-Hyperglycinaemia-NKH-Glycine-Encephalopathy -- "
            "CSF-Plasma-Glycine-Ratio-gt0.08-PATHOGNOMONIC -- "
            "Burst-Suppression-EEG-Neonatal-Form -- "
            "Sodium-Benzoate-Dextromethorphan -- "
            "No-Cure-Severe-Neurodevelopmental-Outcome"
        ),
        "alias": (
            "GLDC (glycine decarboxylase, P protein); OMIM gene 238300; "
            "Nonketotic hyperglycinaemia (NKH) OMIM 605899. "
            "9p24.1; 1020 aa; ~114 kDa; mitochondrial matrix; autosomal recessive. "
            "FUNCTION: GLDC encodes the P protein of the glycine cleavage system (GCS). "
            "The GCS is a 4-component mitochondrial complex: "
            "P protein (GLDC, glycine decarboxylase) — decarboxylates glycine using PLP; "
            "H protein (GCSH, lipoamide-bearing) — shuttles methylamine group; "
            "T protein (AMT, aminomethyltransferase) — transfers methylene to THF; "
            "L protein (DLD, dihydrolipoamide dehydrogenase) — reoxidises lipoamide. "
            "The GCS decomposes glycine: glycine → CO2 + NH3 + N5N10-methylene-THF; "
            "the methylene-THF enters one-carbon metabolism. "
            "GLDC mutations: most common cause of NKH (~80% of cases); AMT ~15%; GCSH rare. "
            "Without GCS: glycine accumulates in all tissues, particularly brain and CSF; "
            "glycine is an inhibitory neurotransmitter (GlyR, strychnine-sensitive) in spinal cord "
            "AND an excitatory co-agonist at NMDA glutamate receptors in brain "
            "(D-serine and glycine are co-agonists at NMDA glycine-B site) — "
            "NMDA over-activation → excitotoxicity → neonatal seizures + encephalopathy. "
            "CLINICAL FORMS: "
            "Classic neonatal NKH (most common, 80%): "
            "onset hours-days; encephalopathy, hypotonia, hiccups, apnoeic spells; "
            "burst-suppression pattern on EEG (pathognomonic for neonatal form); "
            "most die or require ventilatory support; "
            "survivors have profound intellectual disability, spastic quadriplegia, intractable epilepsy; "
            "Late-onset/mild NKH (10-20%): childhood onset; spastic paraplegia, intellectual disability; "
            "Transient NKH: plasma glycine elevations normalise within weeks; "
            "not a true GCS defect — maturation of GCS; benign prognosis. "
            "DIAGNOSIS: "
            "Plasma glycine: markedly elevated (>1000 µmol/L in classic; normal <380 µmol/L); "
            "CSF glycine: markedly elevated; "
            "CSF:plasma glycine ratio >0.08 = PATHOGNOMONIC for NKH "
            "(normal CSF:plasma ratio <0.02); "
            "ketones normal (distinguishes from ketotic hyperglycinaemia of organic acidaemias); "
            "GLDC/AMT/GCSH gene sequencing; "
            "glycine cleavage enzyme activity in liver tissue (if gene panel inconclusive). "
            "TREATMENT: "
            "No curative therapy; treatment is palliative. "
            "Sodium benzoate: conjugates with glycine → hippurate (excreted in urine) "
            "→ reduces systemic glycine; dose 250-500 mg/kg/day; "
            "Dextromethorphan (DM): NMDA receptor antagonist; "
            "blocks glycine co-agonism at NMDA-B site; "
            "reduces seizures in some patients; dose 5-22 mg/kg/day; "
            "ketamine (NMDA blocker) used acutely in neonates; "
            "Strychnine (glycine receptor antagonist): historical use; no longer recommended; "
            "Anticonvulsants: phenobarbital, clonazepam, levetiracetam; "
            "benzodiazepines may worsen (potentiate glycine inhibition at GlyR). "
            "PROGNOSIS: "
            "Classic neonatal form: severe disability universal in survivors; "
            "most families choose supportive care only after neurological assessment; "
            "partial enzyme activity predicts milder phenotype; "
            "some late-onset patients have near-normal intelligence. "
            "KEY CLINICAL FACTS: "
            "Hiccups in a hypotonic neonate = NKH until proven otherwise; "
            "burst suppression on neonatal EEG — distinguish from HIE; "
            "NOT ketotic hyperglycinaemia (no ketosis) — "
            "ketotic hyperglycinaemia = propionic/methylmalonic/isovaleric acidaemias "
            "(which ALSO have elevated glycine but WITH ketosis and organic aciduria); "
            "NBS misses classic NKH (glycine not flagged on standard MS/MS panels "
            "in all programmes); "
            "GLDC gene is one of the largest in the disease panel (1020 aa, 25 exons); "
            "threonine-low diet (threonine is a major glycine precursor): "
            "modest adjunct but limited efficacy."
        ),
        "age_of_onset": "Neonatal (classic); childhood (late-onset)",
        "inheritance": "AR",
        "locus": "9p24.1",
        "protein_size": "1020 aa",
        "key_biomarker": "CSF:plasma glycine ratio >0.08 (PATHOGNOMONIC)",
        "pathognomonic": "CSF:plasma glycine >0.08 + burst suppression EEG neonatal",
        "treatment": "Sodium benzoate (glycine sink); dextromethorphan (NMDA block); no cure",
        "critical_flags": [
            "CSF-PLASMA-GLYCINE-RATIO-gt0.08-PATHOGNOMONIC — lumbar puncture essential",
            "BURST-SUPPRESSION-EEG-NEONATAL — classic NKH signature",
            "HICCUPS-HYPOTONIC-NEONATE — NKH key clinical clue",
            "NOT-KETOTIC-HYPERGLYCINAEMIA — no ketosis no organic aciduria (DDx organic acidaemias)",
            "NO-CURE — palliative sodium benzoate + dextromethorphan",
            "BENZODIAZEPINES-CAUTION — may worsen glycine-mediated inhibition",
            "NBS-MISSES-NKH — glycine not on all MS/MS NBS panels",
        ],
    },
    # -- OAT -- Gyrate Atrophy of Choroid and Retina ---------------------------
    {
        "gene": "OAT",
        "protein": (
            "OAT -- 10q26.13 AR -- Ornithine-Aminotransferase-439aa -- "
            "Gyrate-Atrophy-Choroid-Retina -- "
            "Plasma-Ornithine-10x-Normal -- "
            "Progressive-Chorioretinal-Degeneration-Tunnel-Vision-20s-30s -- "
            "Arginine-Restricted-Diet-Reduces-Ornithine -- "
            "B6-Responsive-5pct"
        ),
        "alias": (
            "OAT (ornithine aminotransferase); OMIM gene 613349; "
            "Gyrate atrophy of choroid and retina OMIM 258870. "
            "10q26.13; 439 aa; ~48 kDa; PLP-dependent; mitochondrial matrix; "
            "liver and most tissues; autosomal recessive. "
            "FUNCTION: OAT catalyses: ornithine + alpha-KG → glutamate semialdehyde + glutamate. "
            "This is the main route for ornithine catabolism; "
            "ornithine also serves as substrate for the urea cycle (OTC, carbamoyl phosphate → citrulline); "
            "OAT links the urea cycle to proline synthesis (glutamate semialdehyde → proline; "
            "or P5C reductase → ornithine recycling). "
            "Without OAT: ornithine accumulates systemically; "
            "plasma ornithine: 400-1000 µmol/L (normal <100 µmol/L) = >10× normal. "
            "PATHOPHYSIOLOGY: "
            "Ornithine accumulation in choroid and retinal pigment epithelium (RPE): "
            "inhibits the OAT enzyme in the RPE → "
            "progressive RPE atrophy → photoreceptor degeneration → choroidal atrophy; "
            "mitochondrial structural abnormalities in RPE and skeletal muscle; "
            "creatine deficiency: OAT also interconverts arginine + glycine → guanidinoacetate; "
            "reduced creatine synthesis; "
            "ornithine inhibits cerebral enzyme activities (hyperornithinaemia); "
            "cerebral effects mild compared to retinal (some patients have mild cognitive impairment). "
            "CLINICAL PRESENTATION: "
            "Myopia (refractive error from infancy); "
            "night blindness from late childhood (rod dysfunction); "
            "progressive concentric visual field constriction → tunnel vision; "
            "fundoscopy: scalloped atrophic patches of chorioretinal degeneration "
            "(initially mid-peripheral, coalesce centrally — 'gyrate' scalloped borders); "
            "posterior subcapsular cataracts (common); "
            "vision preserved centrally until 40s-50s; legal blindness typically 40-55y; "
            "skeletal muscle: type II fibre atrophy (tubular aggregates on biopsy); "
            "usually no acute metabolic crises. "
            "DIAGNOSIS: "
            "Plasma ornithine >400 µmol/L (markedly elevated; DDx: HHH syndrome = SLC25A15, "
            "also elevated ornithine but differs by homocitrullinuria and protein intolerance); "
            "urine amino acids: ornithinuria; "
            "OAT enzyme activity in lymphocytes or fibroblasts; "
            "OAT gene sequencing; "
            "NOT on standard NBS (ornithine not routinely flagged). "
            "TREATMENT: "
            "Arginine-restricted diet: "
            "arginine is converted to ornithine in the gut (arginase); "
            "low-arginine diet reduces ornithine production; "
            "Arg <30 mg/day; protein largely from branched-chain or essential AAs; "
            "target plasma ornithine <200 µmol/L; "
            "B6 (pyridoxine): "
            "PLP is OAT cofactor; B6-responsive variants (5% of cases) show "
            "significant ornithine reduction on pyridoxine 15-20 mg/kg/day; "
            "mandatory B6 trial in all patients; "
            "Creatine supplementation: 1.5 g/day (addresses creatine deficiency); "
            "Proline supplementation (theoretically useful; evidence limited); "
            "Ophthalmology: "
            "annual fundoscopy and visual field testing; "
            "low-vision aids; eventually cane/guide dog; "
            "Gene therapy: clinical trial in progress (AAV-OAT subretinal); "
            "Liver transplant: not indicated (retinal damage not primarily hepatic). "
            "KEY CLINICAL FACTS: "
            "Gyrate atrophy is the ONLY hereditary aminoacidopathy with primary ocular manifestation; "
            "not associated with hyperammonaemia (urea cycle intact — "
            "DDx from UCD disorders); "
            "Finnish population: high prevalence (1:50,000); founder mutation p.Leu402Pro; "
            "ornithine >10× normal is the hallmark — "
            "this level distinguishes from other causes of mild ornithinaemia; "
            "diet compliance is critical but difficult (arginine in all proteins); "
            "retinal degeneration precedes treatment effect — start diet at diagnosis, not when blind."
        ),
        "age_of_onset": "Childhood (myopia/night blindness); blindness 40-55y",
        "inheritance": "AR",
        "locus": "10q26.13",
        "protein_size": "439 aa",
        "key_biomarker": "Plasma ornithine >400 µmol/L (>10× normal)",
        "pathognomonic": "Plasma ornithine >10× + gyrate scalloped chorioretinal atrophy on fundoscopy",
        "treatment": "Arginine-restricted diet; B6 trial; creatine supplementation; annual ophthalmology",
        "critical_flags": [
            "PLASMA-ORNITHINE-10x-NORMAL — hallmark (>400 µmol/L)",
            "B6-TRIAL-MANDATORY — 5% responsive; pyridoxine 15-20 mg/kg",
            "ARGININE-RESTRICTED-DIET — target ornithine <200 µmol/L",
            "ANNUAL-FUNDOSCOPY-VF — progressive and irreversible",
            "NO-HYPERAMMONAEMIA — DDx from UCDs (urea cycle intact)",
            "CREATINE-SUPPLEMENTATION — ornithine-creatine pathway disrupted",
            "GYRATE-ONLY-AMINOACIDOPATHY-OCULAR-PRIMARY — unique clinical niche",
        ],
    },
    # -- TAT -- Tyrosinemia Type 2 (Richner-Hanhart) ---------------------------
    {
        "gene": "TAT",
        "protein": (
            "TAT -- 16q22.2 AR -- Tyrosine-Aminotransferase-454aa -- "
            "Tyrosinemia-Type-2-Richner-Hanhart-Syndrome -- "
            "Palmar-Plantar-Keratosis-Pseudodendritic-Keratitis-ID-TRIAD -- "
            "Plasma-Tyrosine-gt1000-uM -- "
            "Low-Phe-Tyr-Diet-Corrects-ALL-Features -- "
            "No-Hepatocellular-Involvement-DDx-HT1"
        ),
        "alias": (
            "TAT (tyrosine aminotransferase); OMIM gene 613229; "
            "Tyrosinemia type 2 (Richner-Hanhart syndrome, HT2) OMIM 276600. "
            "16q22.2; 454 aa; ~53 kDa; PLP-dependent; hepatic cytoplasm; autosomal recessive. "
            "FUNCTION: TAT catalyses the transamination of L-tyrosine to 4-hydroxyphenylpyruvate "
            "(4-HPP), the first committed step of tyrosine catabolism in the liver. "
            "Without TAT: tyrosine accumulates (plasma Tyr >1000 µmol/L; normal <130); "
            "4-HPP not produced → downstream metabolites (homogentisate, fumarylacetoacetate) not made; "
            "tyrosine crystal deposits form in cornea and skin. "
            "PATHOPHYSIOLOGY: "
            "TAT is ONLY expressed in hepatocytes (unlike HT1 where FAH = ubiquitous); "
            "liver: no toxic metabolite accumulates (tyrosine itself is benign to hepatocytes); "
            "NO hepatic disease — contrasts sharply with HT1; "
            "Cornea: tyrosine crystals deposit in corneal epithelium → inflammatory response "
            "→ pseudodendritic keratitis (HSV-like branching lesion pattern — "
            "crucial DDx: HT2 is NOT herpetic keratitis); "
            "Skin: palmar/plantar hyperkeratotic plaques (non-pruritic); "
            "CNS: intellectual disability in ~50% (mechanism uncertain — "
            "high Tyr may inhibit dopamine/serotonin synthesis; "
            "Tyr is LNAA but competitive BBB transport less severe than in PKU). "
            "CLINICAL PRESENTATION: "
            "TRIAD: "
            "1. Palmar/plantar keratosis — painful hyperkeratotic plaques on hands/feet; "
            "may be so painful they impair walking or grasping; "
            "2. Pseudodendritic keratitis — bilateral; onset infancy to childhood; "
            "photophobia, lacrimation, eye pain; looks like HSV dendritic ulcer but "
            "is NOT herpetic (no fluorescent staining pattern of true dendritic); "
            "3. Intellectual disability — variable (50%); "
            "usually mild-moderate. "
            "NO hepatomegaly (contrasts HT1). "
            "NO liver failure, no coagulopathy, no cirrhosis. "
            "NO succinylacetone (pathway blocked upstream). "
            "DIAGNOSIS: "
            "Plasma tyrosine >1000 µmol/L (markedly elevated); "
            "other plasma amino acids normal (methionine normal — DDx CBS/HCU); "
            "succinylacetone NEGATIVE (distinguishes HT2 from HT1); "
            "urinary 4-HPP and related metabolites (minimal); "
            "urine organic acids: N-acetyltyrosine, 4-hydroxyphenylacetate; "
            "TAT gene sequencing; "
            "elevated Tyr on NBS → confirm with plasma amino acids. "
            "TREATMENT: "
            "Dietary: low phenylalanine + low tyrosine diet (Phe is precursor of Tyr). "
            "Phe-free + Tyr-free amino acid formula for protein. "
            "On diet: plasma Tyr falls to <500 µmol/L → "
            "corneal lesions resolve within WEEKS; "
            "keratoderma resolves within months; "
            "intellectual outcome improved if started early. "
            "No nitisinone (NTBC not indicated — block upstream of FAH would not help "
            "in TAT deficiency which is UPSTREAM of HPPD). "
            "No specific pharmacotherapy; diet is the sole intervention. "
            "KEY CLINICAL FACTS: "
            "Named after Richner (1938) and Hanhart (1947); "
            "commonest in Italy (particularly Sardinia — founder) and North Africa; "
            "pseudodendritic keratitis: frequently MISDIAGNOSED as HSV keratitis → "
            "aciclovir does NOT help; ophthalmologist should check Tyr in any bilateral dendritic-like keratitis; "
            "NO hepatic disease: this is the most important distinguishing feature from HT1; "
            "early dietary treatment (neonatal NBS → prompt diet) prevents ALL complications; "
            "late-treated or untreated: keratoderma + keratitis regress but "
            "intellectual disability may be permanent."
        ),
        "age_of_onset": "Infancy-early childhood",
        "inheritance": "AR",
        "locus": "16q22.2",
        "protein_size": "454 aa",
        "key_biomarker": "Plasma tyrosine >1000 µmol/L; succinylacetone NEGATIVE",
        "pathognomonic": "Palmar/plantar keratosis + pseudodendritic keratitis + elevated Tyr",
        "treatment": "Low Phe/Tyr diet (corrects ALL features); no NTBC; no hepatic involvement",
        "critical_flags": [
            "NO-HEPATIC-DISEASE — critical DDx from HT1 (no succinylacetone, no liver failure)",
            "PSEUDODENDRITIC-KERATITIS-NOT-HERPETIC — aciclovir will NOT help",
            "LOW-PHE-TYR-DIET-CORRECTS-ALL — corneal lesions resolve weeks",
            "SUCCINYLACETONE-NEGATIVE — distinguishes HT2 from HT1",
            "NO-NTBC-INDICATED — TAT is upstream of HPPD (nitisinone target)",
            "MISDIAGNOSIS-AS-HSV-KERATITIS — bilateral dendritic keratitis → check plasma Tyr",
            "SARDINIAN-NORTH-AFRICAN-FOUNDER — high prevalence in specific populations",
        ],
    },
]


def _make_patients(gene_data, seed):
    rng = random.Random(seed)
    ages = [rng.randint(0, 45) for _ in range(40)]
    sexes = [rng.choice(["M", "F"]) for _ in range(40)]
    gene = gene_data["gene"]
    inheritance = gene_data["inheritance"]
    severity_choices = ["mild", "moderate", "severe"]
    patients = []
    for i in range(40):
        severity = rng.choices(severity_choices, weights=[25, 45, 30])[0]
        age = ages[i]
        sex = sexes[i]
        # Sex-linked adjustment: OAT (AR), all others AR — no X-linked in this atlas
        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis": age,
            "sex": sex,
            "severity": severity,
            "inheritance": inheritance,
            "on_diet": rng.random() > 0.15,
            "key_biomarker_abnormal": True,
            "family_cascade": rng.random() > 0.45,
        })
    return patients


def _build_cohort():
    all_patients = []
    for idx, gene_data in enumerate(AA_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_patients(gene_data, seed))
    return all_patients


COHORT = _build_cohort()


# ── API response functions ────────────────────────────────────────────────────

def overview():
    total = len(COHORT)
    gene_counts = {}
    severity_counts = {"mild": 0, "moderate": 0, "severe": 0}
    for p in COHORT:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1
        severity_counts[p["severity"]] = severity_counts.get(p["severity"], 0) + 1

    on_diet = sum(1 for p in COHORT if p.get("on_diet"))
    cascade = sum(1 for p in COHORT if p.get("family_cascade"))

    genes_covered = len(AA_GENES)
    ar_genes = sum(1 for g in AA_GENES if g["inheritance"] == "AR")
    x_linked_genes = sum(1 for g in AA_GENES if g["inheritance"] in ("XLR", "XLD"))

    return {
        "atlas": "Hereditary Amino Acid Disorders Atlas (Aminoacidopathies)",
        "subtitle": (
            "Complete 8-gene hereditary aminoacidopathy reference — "
            "PAH (PKU), CBS (homocystinuria), BCKDHA (MSUD), FAH (HT1), "
            "HGD (alkaptonuria), GLDC (NKH), OAT (gyrate atrophy), TAT (HT2) — "
            "320 patients (8×40, seeds 1822-1829)"
        ),
        "total_patients": total,
        "seed_range": "1822-1829",
        "aggregate_stats": {
            "genes_covered": genes_covered,
            "ar_genes": ar_genes,
            "x_linked_genes": x_linked_genes,
            "ad_genes": 0,
            "patients_per_gene": total // genes_covered,
            "on_diet_pct": round(on_diet / total * 100, 1),
            "family_cascade_pct": round(cascade / total * 100, 1),
            "severity_mild_pct": round(severity_counts["mild"] / total * 100, 1),
            "severity_moderate_pct": round(severity_counts["moderate"] / total * 100, 1),
            "severity_severe_pct": round(severity_counts["severe"] / total * 100, 1),
        },
        "genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "disorder": g["pathognomonic"].split("+")[0].strip(),
                "treatment": g["treatment"],
                "n_patients": gene_counts.get(g["gene"], 0),
                "key_biomarker": g["key_biomarker"],
            }
            for g in AA_GENES
        ],
        "critical_treatment_alerts": [
            flag
            for g in AA_GENES
            for flag in g["critical_flags"]
        ],
    }


def breakdown():
    per_gene = {}
    for g in AA_GENES:
        gene = g["gene"]
        pts = [p for p in COHORT if p["gene"] == gene]
        mild = sum(1 for p in pts if p["severity"] == "mild")
        moderate = sum(1 for p in pts if p["severity"] == "moderate")
        severe = sum(1 for p in pts if p["severity"] == "severe")
        on_diet = sum(1 for p in pts if p.get("on_diet"))
        per_gene[gene] = {
            "gene": gene,
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "n_patients": len(pts),
            "severity": {"mild": mild, "moderate": moderate, "severe": severe},
            "on_diet": on_diet,
            "on_diet_pct": round(on_diet / len(pts) * 100, 1) if pts else 0,
            "family_cascade": sum(1 for p in pts if p.get("family_cascade")),
            "protein_description": g["protein"],
            "age_of_onset": g["age_of_onset"],
        }
    return {
        "atlas": "Hereditary Amino Acid Disorders Atlas — Per-Gene Breakdown",
        "genes": per_gene,
        "aggregate": {
            "total_patients": len(COHORT),
            "total_genes": len(AA_GENES),
            "seed_range": "1822-1829",
            "all_inheritance": list({g["inheritance"] for g in AA_GENES}),
        },
    }


def definitions():
    defs = {}
    for g in AA_GENES:
        defs[g["gene"]] = g["alias"]

    defs["Aminoacidopathy — Overview"] = (
        "An aminoacidopathy is an inherited metabolic disorder caused by a deficiency in an "
        "enzyme or transporter involved in amino acid catabolism, transport, or cofactor metabolism, "
        "leading to accumulation of the affected amino acid (and/or its metabolites) to toxic concentrations. "
        "CLASSIFICATION FRAMEWORK: "
        "By amino acid group: "
        "Aromatic: PAH (Phe), TAT (Tyr — HT2), FAH (Tyr — HT1), HGD (Tyr — AKU); "
        "Sulfur: CBS (Met/Hcy); "
        "Branched-chain: BCKDHA (Leu/Ile/Val — MSUD); "
        "Glycine: GLDC (Gly — NKH); "
        "Ornithine: OAT (Orn — gyrate atrophy). "
        "DIFFERENTIAL APPROACH (by key biomarker): "
        "Elevated Phe → PKU (PAH) or BH4 deficiency (DHPR/PTPS/GTPCH); "
        "Elevated Tyr + succinylacetone → HT1 (FAH); "
        "Elevated Tyr - succinylacetone + keratoderma → HT2 (TAT); "
        "Elevated Tyr + dark urine → AKU (HGD); "
        "Elevated Met + Hcy + ectopia lentis → HCU (CBS); "
        "Elevated Leu+Ile+Val + alloisoleucine → MSUD (BCKDHA/B or DBT); "
        "CSF:plasma Gly >0.08 + encephalopathy → NKH (GLDC/AMT); "
        "Elevated ornithine >10× + chorioretinal degeneration → Gyrate atrophy (OAT). "
        "TREATMENT PRINCIPLES: "
        "Substrate reduction: restrict dietary precursor amino acid (Phe in PKU, Met in HCU, "
        "BCAAs in MSUD, Arg in gyrate atrophy, Phe+Tyr in HT2); "
        "Enzyme replacement: pegvaliase (PKU), nitisinone (HT1, AKU — upstream block); "
        "Cofactor supplementation: sapropterin/BH4 (PKU), pyridoxine/B6 (CBS, OAT); "
        "Alternative pathway: sodium benzoate (NKH — glycine conjugation); "
        "Curative: liver transplant (HT1 — restores FAH; MSUD — provides BCKDH). "
    )

    defs["Newborn Screening (NBS) for Aminoacidopathies"] = (
        "Tandem mass spectrometry (MS/MS) on dried blood spots (DBS) allows simultaneous "
        "quantification of amino acids and acylcarnitines, enabling early detection of "
        "aminoacidopathies before clinical symptoms develop. "
        "DETECTED ON STANDARD MS/MS NBS: "
        "PKU (PAH): elevated Phe, elevated Phe:Tyr ratio — most reliably detected; "
        "MSUD (BCKDHA/B, DBT): elevated Leu+Ile+Val; alloisoleucine on reflex; "
        "HT1 (FAH): succinylacetone (SA) on expanded NBS panels; "
        "tyrosine elevation (non-specific); "
        "HCU (CBS): elevated methionine — late NBS marker (often missed day 1-3); "
        "HT2 (TAT): elevated tyrosine — overlaps with transient neonatal tyrosinaemia; "
        "NOT DETECTED ON STANDARD MS/MS NBS: "
        "AKU (HGD): no MS/MS analyte flagged; detected by urine organic acids or clinical presentation; "
        "NKH (GLDC): glycine not reliably flagged on all programmes; "
        "Gyrate atrophy (OAT): ornithine often not elevated significantly enough at birth. "
        "REFLEX TESTING: "
        "Elevated Phe → plasma amino acids + BH4-loading test; "
        "Elevated Leu → plasma amino acids (alloisoleucine) + urine organics; "
        "Elevated methionine → plasma tHcy + amino acids. "
    )

    defs["Ectopia Lentis — Aminoacidopathy DDx"] = (
        "Ectopia lentis (lens subluxation) in the context of aminoacidopathies: "
        "CBS (HCU): downward subluxation; temporal or inferior; "
        "associated with Marfanoid habitus, fair hair, intellectual disability; "
        "plasma tHcy markedly elevated; methionine elevated. "
        "KEY DDx: "
        "FBN1 (Marfan syndrome): upward (superior) subluxation PATHOGNOMONIC; "
        "normal plasma homocysteine; tall stature + arachnodactyly + aortic dilatation; "
        "ADAMTSL4 mutations: isolated ectopia lentis (no systemic features); "
        "Weill-Marchesani: DOWNWARD but short stature (opposite to Marfan/HCU); "
        "In clinical practice: any ectopia lentis → check plasma tHcy + ophthalmology. "
    )

    defs["Phenylalanine Catabolism Pathway"] = (
        "Phenylalanine → Tyrosine (PAH, cofactor BH4): PKU if blocked. "
        "Tyrosine → 4-HPP (TAT, cofactor PLP): HT2 (Richner-Hanhart) if blocked. "
        "4-HPP → Homogentisate (HPPD, cofactor ascorbate): "
        "NTBC/nitisinone target — blocks this step in HT1 and AKU treatment. "
        "Homogentisate → Maleylacetoacetate (HGD): AKU if blocked. "
        "Maleylacetoacetate → Fumarylacetoacetate (MAAI, isomerase). "
        "Fumarylacetoacetate → Fumarate + Acetoacetate (FAH): HT1 if blocked; "
        "if FAH absent → FAA → succinylacetone (diagnostic, toxic). "
        "CLINICAL IMPLICATIONS: "
        "NTBC blocks HPPD (step 3) — prevents FAA and SA production regardless of FAH; "
        "NTBC also used in AKU (blocks HGA at step 3 by blocking step 2 output); "
        "step 1 (PAH) block → Phe accumulates; Tyr becomes conditionally essential; "
        "step 2 (TAT) block → Tyr accumulates; steps 3-6 unaffected (no succinylacetone). "
    )

    return {
        "atlas": "Hereditary Amino Acid Disorders Atlas — Clinical Definitions",
        "definitions": defs,
        "total_genes": len(AA_GENES),
        "total_definition_entries": len(defs),
    }
