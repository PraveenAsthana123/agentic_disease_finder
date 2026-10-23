#!/usr/bin/env python3
"""Hereditary-CDG-N-Linked-Glycosylation-Atlas -- Complete 8-Gene Reference
Congenital Disorders of Glycosylation (N-linked + mixed) Atlas:
PMM2-MPI-ALG6-SRD5A3-SLC35C1-PGM1-TMEM165-ATP6AP1
(PMM2/CDG-Ia-259aa-16p13.2-AR-Phosphomannomutase2-MostCommon-65-70pct-InvertedNipples-PeauD'Orange-PATHOGNOMONIC-CerebellarHypoplasia,
 MPI/CDG-Ib-423aa-15q24.1-AR-MannosePhosphateIsomerase-NO-Neurological-PATHOGNOMONIC-MannoseSupplement-CURATIVE,
 ALG6/CDG-Ic-507aa-1p31.3-AR-Glucosyltransferase-2ndMostCommon-Milder-Hypotonia-Coagulopathy,
 SRD5A3/CDG-Iq-318aa-4q11-AR-PolyprenolReductase-Ichthyosis-PATHOGNOMONIC-CerebellarHypoplasia-DolicholDeficiency,
 SLC35C1/CDG-IIc-364aa-11p11.2-AR-GDPFucoseTransporter-BombayBloodGroup-PATHOGNOMONIC-LAD-II,
 PGM1/PGM1-CDG-562aa-1p31.3-AR-ExerciseIntolerance-Rhabdomyolysis-MannoseGalactose-RESPONSIVE,
 TMEM165/TMEM165-CDG-324aa-4q12-AR-GolgiMnCaPump-TransferrinIEF-Pattern-PATHOGNOMONIC-ManganeseResponsive,
 ATP6AP1/ATP6AP1-CDG-394aa-Xq28-XLR-VATPase-LiverDisease-DOMINANT-Males-CopperAbnormal):

PMM2 (Phosphomannomutase 2; 259aa; 16p13.2; AR-LOF;
     PMM2 encodes phosphomannomutase 2, catalyzing conversion of mannose-6-phosphate to mannose-1-phosphate,
     the direct precursor of GDP-mannose, essential for N-linked glycosylation;
     PMM2 DEFICIENCY -- CDG-Ia (CDG type Ia) -- OMIM 212065;
     MOST COMMON CDG: accounts for 65-70% of all CDG cases worldwide;
     MECHANISM: PMM2 LOF → GDP-mannose deficient → underglycosylated proteins accumulate → multisystem disease;
     PATHOGNOMONIC CLINICAL FEATURES (infantile period):
       (1) Inverted nipples PATHOGNOMONIC in infants/young children;
       (2) Abnormal subcutaneous fat distribution (lipodystrophy): 'peau d'orange' skin, fat pads over buttocks;
       these two features together in a hypotonic infant = CDG-Ia until proven otherwise;
     CLINICAL PHASES:
       Infantile period: severe hypotonia, developmental delay, cerebellar hypoplasia, strabismus, hepatopathy;
       Childhood: ataxia, intellectual disability, epilepsy (5-8%), cardiomyopathy (rare);
       Adult: stable disability; peripheral neuropathy; hypergonadotropic hypogonadism;
     CEREBELLAR HYPOPLASIA: present in >90%; progressive atrophy; explains ataxia + gaze abnormalities;
     COAGULOPATHY: common — Factor XI/antithrombin deficiency; AVOID warfarin (extreme sensitivity);
       PT/PTT may be prolonged; protein C/S reduced; CHECK coagulation before procedures;
     DIAGNOSIS:
       Transferrin isoelectrofocusing (IEF): type 1 pattern (asialo + disialo-transferrin elevated) PATHOGNOMONIC for PMM2-CDG;
       PMM enzyme assay (leucocytes/fibroblasts); PMM2 sequencing;
       Most common variants: p.R141H (50% European alleles), p.F119L, p.D65Y;
       p.R141H homozygotes: NEVER reported alive — viable alleles always compound heterozyg;
     TREATMENT: no approved pharmacotherapy; mannose supplement FAILS (PMM2 cannot use mannose efficiently);
       physiotherapy, seizure management, dietary support;
       acemannan and other substrate modification: investigational;
     seed SEED_BASE+0).

MPI (Mannose phosphate isomerase; 423aa; 15q24.1; AR-LOF;
     MPI encodes mannose phosphate isomerase, catalyzing fructose-6-phosphate ↔ mannose-6-phosphate interconversion;
     MPI DEFICIENCY -- CDG-Ib -- OMIM 602579;
     THE TREATABLE CDG: mannose supplementation dramatically effective (PO mannose 1g/kg/day);
     PATHOGNOMONIC DISTINGUISHER FROM PMM2: NO NEUROLOGICAL INVOLVEMENT;
       No cerebellar hypoplasia; No cognitive impairment; No inverted nipples; No fat distribution abnormalities;
       Pure visceral/metabolic disease;
     CLINICAL FEATURES:
       Congenital hepatic fibrosis + portal hypertension (DOMINANT): liver disease is defining feature;
       Protein-losing enteropathy: severe hypoalbuminemia; edema; recurrent infections;
       Coagulopathy: PT/PTT prolonged; deficiency of factors II, V, VII, IX, X, XI;
       Hypoglycemia: recurrent fasting/postprandial; often presenting complaint;
       Vomiting, diarrhea; failure to thrive; cyclic vomiting may occur;
     TRANSFERRIN IEF: type 1 pattern (same as PMM2) — MPI-CDG and PMM2-CDG cannot be distinguished by IEF alone;
       ENZYME ACTIVITY distinguishes: MPI enzyme assay (leucocytes) low in MPI-CDG; PMM2 enzyme normal;
     TREATMENT:
       Oral mannose: 1 g/kg/day in 5-6 divided doses; CURATIVE for most manifestations;
       hepatic disease and enteropathy resolve in most patients;
       coagulopathy corrects; hypoglycemia resolves;
       liver transplantation reserved for end-stage portal hypertension (mannose fails if too advanced);
     seed SEED_BASE+1).

ALG6 (UDP-glucose:glycoprotein glucosyltransferase; 507aa; 1p31.3; AR-LOF;
     ALG6 encodes the enzyme adding the first glucose to the lipid-linked oligosaccharide (LLO) precursor in the ER;
     ALG6 DEFICIENCY -- CDG-Ic -- OMIM 603147;
     SECOND MOST COMMON CDG after PMM2 (though much less frequent);
     MECHANISM: ALG6 LOF → glucosylation of Man9GlcNAc2-PP-Dol fails → LLO truncated →
       calnexin/calreticulin quality control escapes → misfolded glycoproteins secreted;
     CLINICAL FEATURES (MILDER than PMM2):
       Hypotonia: moderate; often presenting complaint;
       Intellectual disability: mild-to-moderate; some patients have near-normal cognition;
       Epilepsy: more prominent than PMM2 — 20-30%; may be refractory;
       No cerebellar hypoplasia (distinguishes from PMM2 in many cases);
       Coagulopathy: coagulation factor deficiencies (same mechanism as PMM2);
       Strabismus; feeding difficulties; growth delay;
     TRANSFERRIN IEF: type 1 pattern;
     NOTE: No inverted nipples in ALG6-CDG — useful distinguisher from PMM2;
     TREATMENT: no approved therapy; supportive;
     seed SEED_BASE+2).

SRD5A3 (Polyprenol reductase; 318aa; 4q11; AR-LOF;
     SRD5A3 encodes the enzyme converting polyprenol to dolichol (by reducing the alpha-isoprene unit);
     dolichol = carrier lipid for N-linked glycosylation in the ER (dolichol-PP-oligosaccharide);
     SRD5A3 DEFICIENCY -- CDG-Iq -- OMIM 612379;
     PATHOGNOMONIC DUAL FEATURE: ICHTHYOSIS + CEREBELLAR HYPOPLASIA in same patient;
       Ichthyosis: lamellar/scaling skin (keratinization defect from inadequate dolichol in skin);
       Cerebellar hypoplasia + ataxia: N-glycosylation failure of cerebellar proteins;
       This combination is essentially diagnostic for SRD5A3-CDG;
     ADDITIONAL FEATURES:
       Ocular coloboma (iris/retinal coloboma): 50-70% — pathognomonic in context;
       Congenital eye abnormalities: microphthalmia, coloboma, nystagmus;
       Intellectual disability: moderate-to-severe;
       Hypotonia;
       Retinitis pigmentosa in some (progressive);
     BIOCHEMISTRY:
       Transferrin IEF: type 1 pattern;
       Plasma dolichol-18/dolichol-19 ratio: INVERTED (polyprenol accumulates, dolichol reduced);
       Urine dolichols: polyprenol elevated;
       SKIN: polyprenol accumulates (cannot make dolichol from polyprenol without SRD5A3);
     TREATMENT: no pharmacotherapy; skin emollients for ichthyosis; supportive;
     seed SEED_BASE+3).

SLC35C1 (GDP-fucose transporter 1; 364aa; 11p11.2; AR-LOF;
     SLC35C1 encodes the Golgi GDP-fucose transporter, importing GDP-fucose into Golgi for fucosylation;
     SLC35C1 DEFICIENCY -- CDG-IIc / LADII (Leukocyte Adhesion Deficiency type II) / Ramón syndrome -- OMIM 266265;
     PATHOGNOMONIC: BOMBAY (h/h) BLOOD GROUP;
       Normally: H antigen on RBCs is fucosylated (Fuc-α1,2-Gal); AB0 antigens built on H antigen;
       SLC35C1 LOF: no fucose transport → NO H antigen on RBCs → Bombay blood group;
       Anti-H antibody present → patient is incompatible with ALL ABO blood types (ONLY other Bombay patients compatible);
       Blood transfusion planning: CRITICAL and complex;
     CLINICAL FEATURES:
       Leukocyte adhesion deficiency type II (LAD-II):
         Recurrent bacterial infections (Staphylococcus, Streptococcus, Klebsiella);
         Delayed wound healing; omphalitis in neonates;
         Leukocytosis (persistent >20,000-40,000 even between infections — selectin deficiency);
         Neutrophils cannot migrate to tissues (fucose-dependent selectin ligands absent);
       SHORT STATURE + INTELLECTUAL DISABILITY: growth retardation (fucosylation required for growth factors);
       DISTINCTIVE FACE: coarse facies (similar to I-cell disease appearance);
     BIOCHEMISTRY:
       Transferrin IEF: type 2 pattern (Golgi defect — different from type 1 PMM2);
       Flowcytometry: CD15s (sialyl-Lewis X) absent on neutrophils PATHOGNOMONIC;
     TREATMENT:
       Oral fucose supplementation: dramatically effective for infections + leukocytosis in SOME patients;
         (p.T308R variant: fucose responsive; other variants may not respond);
       Does NOT reverse intellectual disability or blood group abnormality;
       Pre-transfusion: must use Bombay-compatible blood (extremely rare donors);
     seed SEED_BASE+4).

PGM1 (Phosphoglucomutase 1; 562aa; 1p31.3; AR-LOF;
     PGM1 catalyzes interconversion of glucose-1-phosphate ↔ glucose-6-phosphate;
     provides glucose-1-phosphate for both glycogen synthesis AND UDP-glucose/UDP-galactose (Leloir pathway);
     PGM1 DEFICIENCY -- PGM1-CDG -- OMIM 614921;
     UNIQUE: MIXED N-LINKED + O-LINKED GLYCOSYLATION DEFECT;
       PGM1 LOF reduces both glucose-1-P (→ UDP-glucose/galactose) and glucose-6-P supply;
       Impairs BOTH N-linked and mucin-type O-linked glycosylation;
     DISTINCTIVE CLINICAL FEATURES:
       EXERCISE INTOLERANCE + RHABDOMYOLYSIS: fixed myopathy + exertional; lactate rises on ischemic forearm test;
         (GSD component: PGM1 also affects glycogen metabolism — glucose-1-P for glycogen);
       Bifid uvula / cleft palate: 50-60% — structural defect from glycosylation failure during embryogenesis;
       DILATED CARDIOMYOPATHY: 20-30%; potentially life-threatening; echocardiogram MANDATORY;
       Short stature; hypoglycemia (neonatal/infantile);
       Hepatopathy: transaminases elevated; fatty liver;
       Coagulopathy;
     TRANSFERRIN IEF: type 1 AND type 2 pattern (mixed — both N and O glycosylation);
     TREATMENT — DIETARY (RESPONSIVE):
       Mannose supplementation: partially corrects N-glycosylation;
       Galactose supplementation (D-galactose): corrects O-glycosylation; reduces cardiomyopathy progression;
       COMBINED mannose + galactose: best response;
       Avoid prolonged fasting (myopathy + hypoglycemia risk);
     seed SEED_BASE+5).

TMEM165 (Transmembrane protein 165; 324aa; 4q12; AR-LOF;
     TMEM165 encodes a Golgi Ca²⁺/Mn²⁺ transporter (UPF0016 family);
     TMEM165 is essential for maintaining Golgi luminal Mn²⁺ and Ca²⁺ levels required for glycosyltransferase activity;
     many glycosyltransferases require Mn²⁺ as cofactor;
     TMEM165 DEFICIENCY -- TMEM165-CDG -- OMIM 614727;
     PATHOGNOMONIC TRANSFERRIN IEF: type 2 pattern (Golgi-level defect);
     MANGANESE RESPONSIVE: unlike most CDG types;
       Oral Mn²⁺ supplementation corrects Golgi manganese, restores glycosyltransferase activity;
       MnCl₂ or Mn-enriched formula: normalizes transferrin IEF pattern in some patients;
       FIRST CDG type where mineral supplementation corrects biochemistry;
     CLINICAL FEATURES:
       Moderate intellectual disability;
       Moderate-to-severe hypotonia;
       Short stature;
       Mild dysmorphic features;
       Liver involvement: mild transaminase elevation;
       Skeletal dysplasia in some patients;
     NOTE: TMEM165 also important for autophagy and lysosomal function (pleiotropic membrane transporter);
     TREATMENT: Mn supplementation (investigational/expert center); supportive;
     seed SEED_BASE+6).

ATP6AP1 (ATPase H+-transporting accessory protein 1; 394aa; Xq28; XLR;
     ATP6AP1 encodes accessory subunit 1 of the V-type H+-ATPase (V-ATPase), also called subunit AC45;
     V-ATPase acidifies endosomes, lysosomes, and Golgi lumen — required for proper Golgi glycosylation;
     ATP6AP1 DEFICIENCY -- ATP6AP1-CDG -- OMIM 300911;
     X-LINKED: affects males predominantly (females can be mildly affected as carriers);
     LIVER DISEASE DOMINANT:
       Hepatosplenomegaly + liver disease is the PRIMARY presenting feature;
       Elevated transaminases; cholestasis; hepatic fibrosis; cirrhosis in severe cases;
       Liver biopsy: electron microscopy shows dilated ER and abnormal Golgi;
       Liver failure requiring transplantation in severe cases;
     IMMUNODEFICIENCY: recurrent infections — V-ATPase required for lysosomal bactericidal activity;
       neutrophil dysfunction; antibody deficiency;
     ABNORMAL COPPER METABOLISM:
       Serum ceruloplasmin: low (Golgi copper loading defect for ceruloplasmin glycosylation+copper insertion);
       Serum copper: may be low; excess hepatic copper accumulation;
       Resembles partial ATP7B (Wilson disease): important DDx;
     TRANSFERRIN IEF: type 2 pattern (Golgi defect);
     TREATMENT: liver transplantation may be required and corrects hepatic disease;
       copper supplementation investigated;
       supportive antibiotics for immunodeficiency;
     seed SEED_BASE+7)."""

import random, hashlib, json
from typing import List, Dict, Any

ATLAS_ID  = "hereditary-cdg-n-linked-glycosylation-atlas"
SEED_BASE = 4246
GENES     = ["PMM2", "MPI", "ALG6", "SRD5A3", "SLC35C1", "PGM1", "TMEM165", "ATP6AP1"]

LLO_ASSEMBLY_GROUP = ["PMM2", "MPI", "ALG6", "SRD5A3"]
GOLGI_TRANSPORT_GROUP = ["SLC35C1", "TMEM165", "ATP6AP1"]
MIXED_GLYCOSYLATION_GROUP = ["PGM1"]

GENE_META = {
    "PMM2":    {"aa": 259,  "locus": "16p13.2", "inh": "AR",   "size_kda": 28,  "mechanism": "mannose-6P-to-mannose-1P-GDP-mannose-deficiency"},
    "MPI":     {"aa": 423,  "locus": "15q24.1", "inh": "AR",   "size_kda": 47,  "mechanism": "fructose-6P-mannose-6P-no-neurological-mannose-CURATIVE"},
    "ALG6":    {"aa": 507,  "locus": "1p31.3",  "inh": "AR",   "size_kda": 57,  "mechanism": "first-glucose-LLO-glucosyltransferase-milder-no-inverted-nipples"},
    "SRD5A3":  {"aa": 318,  "locus": "4q11",    "inh": "AR",   "size_kda": 35,  "mechanism": "polyprenol-reductase-dolichol-deficiency-ichthyosis-cerebellar"},
    "SLC35C1": {"aa": 364,  "locus": "11p11.2", "inh": "AR",   "size_kda": 40,  "mechanism": "GDP-fucose-transporter-Bombay-blood-group-LAD-II-fucose-responsive"},
    "PGM1":    {"aa": 562,  "locus": "1p31.3",  "inh": "AR",   "size_kda": 63,  "mechanism": "mixed-N-O-glycosylation-exercise-intolerance-mannose-galactose-RESPONSIVE"},
    "TMEM165": {"aa": 324,  "locus": "4q12",    "inh": "AR",   "size_kda": 36,  "mechanism": "Golgi-Mn-Ca-pump-manganese-responsive-type2-IEF"},
    "ATP6AP1": {"aa": 394,  "locus": "Xq28",    "inh": "XLR",  "size_kda": 45,  "mechanism": "V-ATPase-accessory-liver-dominant-copper-abnormal-XLR"},
}

PATHOGNOMONIC = {
    "PMM2":    "Inverted nipples + peau d'orange subcutaneous fat pads (infancy) PATHOGNOMONIC; cerebellar hypoplasia; transferrin IEF type 1; p.R141H never homozygous viable",
    "MPI":     "NO neurological involvement PATHOGNOMONIC (distinguishes from all other CDGs); mannose supplement CURATIVE for hepatopathy + enteropathy + coagulopathy; transferrin IEF type 1",
    "ALG6":    "No inverted nipples (distinguishes from PMM2); epilepsy more prominent (20-30%); no cerebellar hypoplasia in many; transferrin IEF type 1",
    "SRD5A3":  "Ichthyosis + cerebellar hypoplasia PATHOGNOMONIC combination; ocular coloboma 50-70%; polyprenol/dolichol ratio inverted; transferrin IEF type 1",
    "SLC35C1": "Bombay (h/h) blood group PATHOGNOMONIC — incompatible with all standard ABO types; CD15s absent on neutrophils; LAD-II phenotype; transferrin IEF type 2",
    "PGM1":    "Bifid uvula/cleft palate 50-60% + rhabdomyolysis + mixed type 1+2 IEF PATHOGNOMONIC combination; galactose + mannose supplementation RESPONSIVE; cardiomyopathy 20-30%",
    "TMEM165": "Transferrin IEF type 2 (Golgi); Mn²⁺ supplementation corrects biochemistry PATHOGNOMONIC response; moderate ID + hypotonia; skeletal dysplasia",
    "ATP6AP1": "X-linked liver disease dominant in males; low ceruloplasmin (copper defect) + hepatosplenomegaly; V-ATPase Golgi acidification failure; transferrin IEF type 2",
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _make_cohort(gene: str, seed: int) -> Dict[str, Any]:
    rng = _rng(seed)
    n = 40
    gene_meta = GENE_META[gene]

    if gene == "PMM2":
        pathogenic_frac = rng.uniform(0.90, 0.97)
    elif gene in ("MPI", "ALG6"):
        pathogenic_frac = rng.uniform(0.85, 0.94)
    elif gene in ("SRD5A3", "SLC35C1"):
        pathogenic_frac = rng.uniform(0.80, 0.92)
    else:
        pathogenic_frac = rng.uniform(0.78, 0.91)

    pathogenic = int(n * pathogenic_frac)
    vus = n - pathogenic

    if gene == "PMM2":
        mannose_response   = 0.0
        galactose_response = 0.0
        treatable          = False
        neurological_involvement = rng.uniform(0.85, 0.98)
    elif gene == "MPI":
        mannose_response   = rng.uniform(0.82, 0.95)
        galactose_response = 0.0
        treatable          = True
        neurological_involvement = 0.0  # NO neurological — PATHOGNOMONIC
    elif gene == "ALG6":
        mannose_response   = 0.0
        galactose_response = 0.0
        treatable          = False
        neurological_involvement = rng.uniform(0.70, 0.90)
    elif gene == "SRD5A3":
        mannose_response   = 0.0
        galactose_response = 0.0
        treatable          = False
        neurological_involvement = rng.uniform(0.80, 0.95)
    elif gene == "SLC35C1":
        mannose_response   = 0.0
        galactose_response = 0.0
        fucose_response    = rng.uniform(0.40, 0.75)  # variable by variant
        treatable          = True  # for infections, not ID
        neurological_involvement = rng.uniform(0.60, 0.85)
    elif gene == "PGM1":
        mannose_response   = rng.uniform(0.50, 0.75)
        galactose_response = rng.uniform(0.60, 0.85)
        treatable          = True
        neurological_involvement = rng.uniform(0.55, 0.80)
    elif gene == "TMEM165":
        mannose_response   = 0.0
        galactose_response = 0.0
        treatable          = True  # manganese responsive
        neurological_involvement = rng.uniform(0.70, 0.88)
    else:  # ATP6AP1
        mannose_response   = 0.0
        galactose_response = 0.0
        treatable          = True  # liver transplant
        neurological_involvement = rng.uniform(0.30, 0.60)  # variable

    return {
        "gene":              gene,
        "n":                 n,
        "pathogenic_n":      pathogenic,
        "vus_n":             vus,
        "locus":             gene_meta["locus"],
        "inheritance":       gene_meta["inh"],
        "aa":                gene_meta["aa"],
        "size_kda":          gene_meta["size_kda"],
        "mechanism_tag":     gene_meta["mechanism"],
        "pathognomonic":     PATHOGNOMONIC[gene],
        "treatable":         treatable,
        "neurological_frac": neurological_involvement if gene != "MPI" else 0.0,
    }


def generate_overview() -> Dict[str, Any]:
    cohorts = []
    for i, gene in enumerate(GENES):
        cohorts.append(_make_cohort(gene, SEED_BASE + i))

    summary_table = []
    for c in cohorts:
        summary_table.append({
            "gene":          c["gene"],
            "locus":         c["locus"],
            "aa":            c["aa"],
            "kda":           c["size_kda"],
            "inheritance":   c["inheritance"],
            "mechanism":     c["mechanism_tag"],
            "pathognomonic": c["pathognomonic"],
            "n":             c["n"],
            "treatable":     c["treatable"],
        })

    ief_groups = {
        "type1_pattern": {
            "genes":       ["PMM2", "MPI", "ALG6", "SRD5A3"],
            "description": "ER-level LLO assembly defects — asialo + disialo-transferrin elevated; type 1 pattern",
            "notes":       "PMM2 and MPI share type 1 IEF — enzyme assay MANDATORY to distinguish (MPI treatable!)",
        },
        "type2_pattern": {
            "genes":       ["SLC35C1", "TMEM165", "ATP6AP1"],
            "description": "Golgi-level processing defects — mono/trisialotransferrin elevated; type 2 pattern",
            "notes":       "Type 2 IEF = Golgi dysfunction; CD15s flowcytometry separates SLC35C1 (absent CD15s) from TMEM165/ATP6AP1",
        },
        "mixed_pattern": {
            "genes":       ["PGM1"],
            "description": "Mixed N+O glycosylation defect — may show both type 1 and type 2 components",
            "notes":       "PGM1 affects both LLO assembly (via UDP-glucose) and mucin O-glycosylation; bifid uvula + rhabdomyolysis",
        },
    }

    total_patients = sum(c["n"] for c in cohorts)
    treatable_genes = [c["gene"] for c in cohorts if c["treatable"]]

    return {
        "atlas_id":          ATLAS_ID,
        "title":             "Hereditary-CDG-N-Linked-Glycosylation-Atlas — 8-Gene Reference",
        "genes":             GENES,
        "total_patients":    total_patients,
        "summary_table":     summary_table,
        "ief_groups":        ief_groups,
        "treatable_genes":   treatable_genes,
        "seeds":             f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "diagnostic_triage": (
            "STEP 1: Transferrin IEF (serum transferrin isoelectric focusing) — first-line screen for ALL CDG types. "
            "Type 1 pattern (asialo + disialo elevated) → ER-assembly defect → PMM2/MPI/ALG6/SRD5A3. "
            "Type 2 pattern (mono + trisialo elevated) → Golgi processing defect → SLC35C1/TMEM165/ATP6AP1. "
            "Mixed type 1+2 → PGM1-CDG. "
            "STEP 2 (type 1): PMM2 enzyme assay (leucocytes) then MPI enzyme assay (if PMM2 normal — MPI is treatable!). "
            "STEP 3 (type 2): Blood group typing (Bombay = SLC35C1); CD15s flow cytometry (absent = SLC35C1). "
            "Transferrin IEF: only valid in patients >2 weeks old and NOT during active illness/sialylation changes."
        ),
    }


def generate_breakdown() -> Dict[str, Any]:
    cohorts = []
    for i, gene in enumerate(GENES):
        cohorts.append(_make_cohort(gene, SEED_BASE + i))

    gene_profiles = []
    for c in cohorts:
        rng = _rng(SEED_BASE + GENES.index(c["gene"]))
        gene = c["gene"]

        if gene == "PMM2":
            treatment_response = {
                "mannose_supplement": "NOT EFFECTIVE — PMM2 cannot use mannose efficiently",
                "substrate_reduction": "Not applicable",
                "dietary": "Physiotherapy + standard nutritional support",
                "monitoring": "Coagulation quarterly; echocardiogram; cerebellar MRI",
            }
            ief_type = "Type 1 (ER defect)"
            key_labs = "Transferrin IEF type 1; PMM2 enzyme activity low; p.R141H allele frequency check"
            cerebellar = rng.uniform(0.88, 0.96)
            coagulopathy = rng.uniform(0.70, 0.88)
            profile_extra = {
                "cerebellar_hypoplasia_pct": round(cerebellar * 100, 1),
                "coagulopathy_pct":          round(coagulopathy * 100, 1),
                "inverted_nipples_pct":      round(rng.uniform(0.60, 0.80) * 100, 1),
            }
        elif gene == "MPI":
            treatment_response = {
                "mannose_supplement": "CURATIVE — 1g/kg/day PO in 5-6 doses; hepatopathy/enteropathy/coagulopathy resolve",
                "substrate_reduction": "Not applicable",
                "dietary": "Oral mannose supplement; liver transplant if portal hypertension severe before treatment",
                "monitoring": "LFTs monthly on mannose; albumin; PT/PTT; glucose; liver ultrasound",
            }
            ief_type = "Type 1 (ER defect) — same as PMM2; enzyme assay distinguishes"
            key_labs = "Transferrin IEF type 1; MPI enzyme activity near-zero; PMM2 activity NORMAL"
            hepatic = rng.uniform(0.85, 0.98)
            enteropathy = rng.uniform(0.60, 0.82)
            profile_extra = {
                "hepatic_involvement_pct":    round(hepatic * 100, 1),
                "protein_losing_enteropathy": round(enteropathy * 100, 1),
                "neurological_involvement":   "0% — pathognomonic absence",
            }
        elif gene == "ALG6":
            treatment_response = {
                "mannose_supplement": "Not effective",
                "substrate_reduction": "Not applicable",
                "dietary": "Standard support; anti-epileptic drugs; physiotherapy",
                "monitoring": "Seizure diary; coagulation; developmental assessment",
            }
            ief_type = "Type 1 (ER defect)"
            key_labs = "Transferrin IEF type 1; ALG6 sequencing; no cerebellar hypoplasia in many"
            epilepsy = rng.uniform(0.20, 0.35)
            profile_extra = {
                "epilepsy_pct":         round(epilepsy * 100, 1),
                "cerebellar_hypo_pct":  round(rng.uniform(0.15, 0.40) * 100, 1),
                "inverted_nipples_pct": "Absent (distinguishes from PMM2)",
            }
        elif gene == "SRD5A3":
            treatment_response = {
                "mannose_supplement": "Not effective",
                "substrate_reduction": "Skin emollients for ichthyosis; retinoids investigated",
                "dietary": "Standard support",
                "monitoring": "Ophthalmology (coloboma); cerebellar MRI; dolichol/polyprenol ratio",
            }
            ief_type = "Type 1 (ER/dolichol defect)"
            key_labs = "Transferrin IEF type 1; plasma polyprenol/dolichol-18 ratio inverted; SRD5A3 sequencing"
            coloboma = rng.uniform(0.45, 0.72)
            profile_extra = {
                "ichthyosis_pct":           round(rng.uniform(0.75, 0.95) * 100, 1),
                "ocular_coloboma_pct":      round(coloboma * 100, 1),
                "cerebellar_hypoplasia_pct": round(rng.uniform(0.80, 0.95) * 100, 1),
            }
        elif gene == "SLC35C1":
            treatment_response = {
                "mannose_supplement": "Not applicable",
                "fucose_supplement": "Oral fucose 25 mg/kg/dose TID — RESPONSIVE in some variants (p.T308R); reduces infections, corrects leukocytosis; does NOT reverse blood group",
                "substrate_reduction": "Not applicable",
                "monitoring": "CD15s flow quarterly; blood group typing all family members; blood bank ALERT mandatory",
            }
            ief_type = "Type 2 (Golgi defect)"
            key_labs = "Transferrin IEF type 2; ABO blood typing: Bombay (anti-H positive); CD15s absent on neutrophils"
            leukocytosis = rng.uniform(0.80, 0.95)
            profile_extra = {
                "bombay_blood_group":         "100% — pathognomonic",
                "leukocytosis_pct":           round(leukocytosis * 100, 1),
                "CD15s_absent_pct":           round(rng.uniform(0.90, 0.98) * 100, 1),
                "fucose_responsive_variants": "~30-40% of cases (p.T308R: best response)",
            }
        elif gene == "PGM1":
            treatment_response = {
                "mannose_supplement": "Effective for N-glycosylation + hyperglycosylation; 0.5g/kg/day",
                "galactose_supplement": "D-galactose 1-1.5g/kg/day — corrects O-glycosylation; reduces cardiomyopathy",
                "combined": "Mannose + galactose: best combined response; monitor echocardiogram",
                "dietary": "Avoid prolonged fasting (hypoglycemia + rhabdomyolysis); cornstarch in GSD component",
                "monitoring": "Echo 6-monthly; CK; LFTs; transferrin IEF on treatment",
            }
            ief_type = "Mixed type 1 + type 2 (both N and O glycosylation)"
            key_labs = "Mixed IEF pattern; PGM1 enzyme assay; CK elevated; bifid uvula/cleft palate exam"
            bifid = rng.uniform(0.45, 0.65)
            cardiomyopathy = rng.uniform(0.18, 0.32)
            profile_extra = {
                "bifid_uvula_cleft_pct":    round(bifid * 100, 1),
                "cardiomyopathy_pct":       round(cardiomyopathy * 100, 1),
                "rhabdomyolysis_episodes":  round(rng.uniform(0.55, 0.80) * 100, 1),
                "mannose_response_pct":     round(rng.uniform(0.55, 0.78) * 100, 1),
                "galactose_response_pct":   round(rng.uniform(0.62, 0.85) * 100, 1),
            }
        elif gene == "TMEM165":
            treatment_response = {
                "manganese_supplement": "MnCl₂ or Mn-enriched formula — corrects transferrin IEF in responding patients; investigational",
                "substrate_reduction": "Not applicable",
                "dietary": "Mn-rich diet; specialist metabolic center",
                "monitoring": "Serum Mn levels; transferrin IEF response; liver function",
            }
            ief_type = "Type 2 (Golgi defect)"
            key_labs = "Transferrin IEF type 2; serum Mn low/dysregulated; TMEM165 sequencing"
            mn_response = rng.uniform(0.40, 0.70)
            profile_extra = {
                "manganese_responsive_pct":  round(mn_response * 100, 1),
                "skeletal_dysplasia_pct":    round(rng.uniform(0.25, 0.50) * 100, 1),
                "ief_type2_confirmed":       "100%",
            }
        else:  # ATP6AP1
            treatment_response = {
                "liver_transplant": "Curative for hepatic disease; corrects liver function + copper abnormality",
                "copper_supplement": "Copper supplementation if ceruloplasmin low; monitor serum copper",
                "antibiotics": "Prophylactic antibiotics for recurrent infections (immunodeficiency component)",
                "monitoring": "LFTs monthly; ceruloplasmin + copper quarterly; liver biopsy if fibrosis suspected",
            }
            ief_type = "Type 2 (Golgi V-ATPase defect)"
            key_labs = "Transferrin IEF type 2; ceruloplasmin low; hepatic copper elevated; liver biopsy EM (dilated ER/Golgi)"
            liver_transplant = rng.uniform(0.25, 0.45)
            profile_extra = {
                "males_affected_pct":         round(rng.uniform(0.85, 0.98) * 100, 1),
                "liver_transplant_needed_pct": round(liver_transplant * 100, 1),
                "low_ceruloplasmin_pct":       round(rng.uniform(0.60, 0.82) * 100, 1),
            }

        gene_profiles.append({
            "gene":              gene,
            "n":                 c["n"],
            "pathogenic_n":      c["pathogenic_n"],
            "vus_n":             c["vus_n"],
            "locus":             c["locus"],
            "inheritance":       c["inheritance"],
            "aa":                c["aa"],
            "size_kda":          c["size_kda"],
            "pathognomonic":     PATHOGNOMONIC[gene],
            "ief_pattern":       ief_type,
            "key_labs":          key_labs,
            "treatment":         treatment_response,
            "profile_extra":     profile_extra,
        })

    variant_spectrum = []
    for gene in GENES:
        rng2 = _rng(SEED_BASE + GENES.index(gene) + 100)
        if gene == "PMM2":
            variants = [
                {"variant": "p.R141H", "type": "missense", "allele_freq": "50% European alleles", "note": "Never homozygous viable — compound heterozyg only"},
                {"variant": "p.F119L", "type": "missense", "allele_freq": "10-15%", "note": "Second most common; R141H/F119L compound most common genotype"},
                {"variant": "p.D65Y",  "type": "missense", "allele_freq": "3-5%",  "note": "Severe allele"},
            ]
        elif gene == "MPI":
            variants = [
                {"variant": "p.R295H",  "type": "missense", "allele_freq": "most common", "note": "Variable phenotype"},
                {"variant": "splice",   "type": "splicing", "allele_freq": "variable",     "note": "Various splice variants reported"},
            ]
        elif gene == "SLC35C1":
            variants = [
                {"variant": "p.T308R", "type": "missense", "allele_freq": "most common", "note": "Fucose responsive — best response to oral fucose"},
                {"variant": "p.R147C", "type": "missense", "allele_freq": "~15%",        "note": "Arab founder variant — less fucose responsive"},
            ]
        elif gene == "PGM1":
            variants = [
                {"variant": "p.Q66H",  "type": "missense", "allele_freq": "common",   "note": "Reduces catalytic activity"},
                {"variant": "p.T337M", "type": "missense", "allele_freq": "variable", "note": "Structural destabilization"},
            ]
        else:
            variants = [
                {"variant": "various_missense", "type": "missense", "allele_freq": "multiple", "note": f"Multiple pathogenic variants described for {gene}"},
            ]
        variant_spectrum.append({"gene": gene, "notable_variants": variants})

    return {
        "atlas_id":          ATLAS_ID,
        "gene_profiles":     gene_profiles,
        "variant_spectrum":  variant_spectrum,
        "seeds":             f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "ief_triage_note":   (
            "Transferrin IEF is the universal CDG screen. "
            "Type 1 = ER defect (PMM2, MPI, ALG6, SRD5A3). "
            "Type 2 = Golgi defect (SLC35C1, TMEM165, ATP6AP1). "
            "Mixed = PGM1. "
            "IEF false-negatives: acute illness, galactosemia treatment. "
            "IEF invalid in first 2 weeks of life. "
            "CRITICAL: in type 1 IEF — ALWAYS check MPI enzyme (treatable!) before assuming PMM2."
        ),
    }


def generate_definitions() -> Dict[str, Any]:
    key_distinguishers = [
        {
            "feature":        "Transferrin IEF type 1 vs type 2",
            "significance":   "Primary CDG classification split",
            "type1_genes":    "PMM2, MPI, ALG6, SRD5A3 — ER-level LLO assembly defects",
            "type2_genes":    "SLC35C1, TMEM165, ATP6AP1 — Golgi-level processing defects",
            "mixed_genes":    "PGM1 — mixed N+O glycosylation",
            "note":           "IEF type 1 = asialo + disialo-transferrin elevated; type 2 = mono + trisialo elevated",
        },
        {
            "feature":        "PMM2 vs MPI — CRITICAL clinical distinction",
            "significance":   "MPI is TREATABLE with mannose; PMM2 is NOT — same IEF type 1 pattern",
            "pmm2":           "Cerebellar hypoplasia + inverted nipples + neurological; mannose FAILS",
            "mpi":            "NO neurological (pathognomonic); hepatic/enteropathy; mannose CURATIVE",
            "how_to_distinguish": "PMM2 enzyme assay: low in PMM2; normal in MPI-CDG. MPI enzyme assay: near-zero in MPI-CDG",
            "clinical_urgency": "ALWAYS rule out MPI in type 1 IEF without cerebellar hypoplasia — treatable condition",
        },
        {
            "feature":        "Bombay blood group (h/h) — SLC35C1 unique marker",
            "significance":   "Anti-H antibody present; incompatible with all standard ABO blood products",
            "clinical_action": "BLOOD BANK ALERT: identify Bombay donors in advance; autologous blood collection if elective surgery; contact national rare donor registry",
            "associated_features": "CD15s absent on neutrophils (flowcytometry); LAD-II phenotype; recurrent infections",
        },
        {
            "feature":        "PGM1 — bifid uvula/cleft palate + rhabdomyolysis combination",
            "significance":   "Essentially diagnostic when present together with IEF abnormality",
            "treatment":      "D-galactose + mannose supplementation RESPONSIVE; monitor cardiomyopathy",
            "exercise_risk":  "Avoid intense exercise; CK monitoring; myopathy component",
        },
        {
            "feature":        "ATP6AP1 — copper abnormality in CDG",
            "significance":   "Low ceruloplasmin + hepatic copper accumulation can mimic Wilson disease",
            "ddx_wilson":     "ATP6AP1-CDG: XLR males; transferrin IEF type 2; liver biopsy EM shows ER/Golgi dilation; ATP7B normal",
            "treatment":      "Liver transplant may be required and corrects copper metabolism",
        },
        {
            "feature":        "SRD5A3 — ichthyosis as CDG clue",
            "significance":   "Ichthyosis in CDG context should prompt SRD5A3 testing",
            "mechanism":      "Dolichol deficiency from polyprenol reductase deficiency; skin requires dolichol-linked glycosylation",
            "key_biochemistry": "Polyprenol/dolichol-18 ratio inverted in plasma; urine polyprenol elevated",
        },
    ]

    treatment_matrix = [
        {
            "intervention":    "Oral mannose supplementation",
            "indications":     "MPI-CDG (CDG-Ib) CURATIVE; partial benefit in PGM1-CDG",
            "mechanism":       "Bypasses MPI block — mannose → mannose-6-P (via hexokinase); restores GDP-mannose; corrects N-glycosylation",
            "dose":            "MPI-CDG: 1 g/kg/day PO divided 5-6 doses; PGM1: 0.5 g/kg/day combined with galactose",
            "contraindicated": "PMM2-CDG: NOT effective (PMM2 cannot process mannose-1-phosphate); do not use as empirical CDG treatment without MPI enzyme confirmation",
        },
        {
            "intervention":    "D-galactose supplementation",
            "indications":     "PGM1-CDG — corrects O-linked glycosylation defect; reduces cardiomyopathy progression",
            "mechanism":       "D-galactose → galactose-1-phosphate (GALT) → UDP-galactose; bypasses PGM1 block for O-glycosylation",
            "dose":            "1-1.5 g/kg/day in 3-4 divided doses; combine with mannose for N-glycosylation",
            "monitoring":      "Echocardiogram 6-monthly (cardiomyopathy regression); CK levels; LFTs; transferrin IEF on treatment",
        },
        {
            "intervention":    "Oral fucose supplementation",
            "indications":     "SLC35C1-CDG (CDG-IIc/LADII) — variant-dependent; p.T308R: best response",
            "mechanism":       "Increases intracellular GDP-fucose via salvage pathway; partially restores Golgi fucose transport",
            "dose":            "25 mg/kg/dose 3×/day; titrate to leukocyte count normalization",
            "limitations":     "Does NOT reverse Bombay blood group or intellectual disability; variable response by genotype; obtain specialist guidance",
        },
        {
            "intervention":    "Manganese supplementation",
            "indications":     "TMEM165-CDG — corrects Golgi Mn²⁺ deficiency; normalizes transferrin IEF in responsive patients",
            "mechanism":       "Exogenous Mn²⁺ partially compensates for defective Golgi Mn²⁺ pumping; restores Mn-dependent glycosyltransferase activity",
            "dose":            "MnCl₂ solution or Mn-enriched elemental formula; dosing at specialist metabolic centers only",
            "monitoring":      "Serum Mn levels; transferrin IEF; liver function (Mn excess risk)",
        },
        {
            "intervention":    "Liver transplantation",
            "indications":     "MPI-CDG: if portal hypertension severe before mannose started; ATP6AP1-CDG: liver failure in XLR males",
            "outcome":         "ATP6AP1-CDG: liver transplant corrects hepatic disease + ceruloplasmin/copper abnormality; does NOT correct extrahepatic glycosylation defect",
            "note":            "MPI-CDG liver transplant: mannose still required post-transplant for enteropathy component",
        },
    ]

    return {
        "atlas_id":              ATLAS_ID,
        "key_distinguishers":    key_distinguishers,
        "treatment_matrix":      treatment_matrix,
        "seeds":                 f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "primary_cascade_split": (
            "STEP 1: Transferrin IEF (serum) — universal CDG screen. "
            "Type 1 → ER defect: PMM2, MPI, ALG6, SRD5A3. "
            "Type 2 → Golgi defect: SLC35C1, TMEM165, ATP6AP1. "
            "Mixed → PGM1. "
            "STEP 2 (type 1): Check for inverted nipples + cerebellar (→ PMM2 likely); no neurological (→ MPI first; treatable!). "
            "  PMM2 enzyme assay (leucocytes): if low → PMM2-CDG; if normal → MPI enzyme assay → MPI-CDG. "
            "  Ichthyosis + coloboma → SRD5A3; polyprenol ratio inverted. "
            "  No distinguishing features → ALG6 (no cerebellar hypoplasia in many; epilepsy more prominent). "
            "STEP 3 (type 2): Blood group typing: Bombay → SLC35C1; CD15s flow absent → SLC35C1. "
            "  Male + liver disease + low ceruloplasmin → ATP6AP1 (XLR). "
            "  Moderate ID + Mn responsive → TMEM165. "
            "STEP 4 (mixed): Bifid uvula + rhabdomyolysis + bifid IEF → PGM1; start mannose + galactose."
        ),
        "ief_notes": {
            "type1_mechanism": "ER-level N-linked glycosylation defect — underglycosylated transferrin; more negative charge lost (fewer sialic acids) → shifted cathodally",
            "type2_mechanism": "Golgi-level processing defect — different sialic acid pattern; mono/trisialo transferrin elevated",
            "false_negatives": "Active illness, iron deficiency, pregnancy may affect IEF; sialylation varies with acute phase response",
            "age_validity":    "IEF only valid after 2 weeks of age — neonatal transferrin glycosylation immature",
            "confirmatory":    "IEF is screening only; enzyme activity + molecular confirmation mandatory before treatment decisions",
        },
        "cdg_registry": "CDG International Patient Registry (EIMD/CDG-HOPE); report new cases to national metabolic network",
    }
