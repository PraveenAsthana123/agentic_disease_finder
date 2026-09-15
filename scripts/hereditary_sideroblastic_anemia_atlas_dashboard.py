"""Hereditary Sideroblastic Anemia Atlas — 8-Gene Reference
ALAS2-SLC25A38-GLRX5-HSPA9-ABCB7-PUS1-YARS2-TRNT1
Ring Sideroblast / Mitochondrial Iron-Loading / Sideroblastic Anemia Spectrum
320 patients (8 x 40), seeds 2814-2821.
Endpoints: /api/hereditary-sideroblastic-anemia-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "ALAS2",
        "protein": (
            "ALAS2 -- Xp11.21 XLR/GOF AD -- 587aa -- "
            "5-Aminolevulinate-Synthase-2-65kDa-Mitochondrial-Matrix-"
            "Rate-Limiting-Haem-Synthesis-Erythroid-Pyridoxal-Phosphate-Dependent-"
            "OMIM-Gene-301300-Disease-XLSA-300751-XLP-301310"
        ),
        "locus": "Xp11.21",
        "protein_size": (
            "587 aa / 65 kDa (mitochondrial matrix; pyridoxal-5'-phosphate (PLP) dependent; "
            "catalyses glycine + succinyl-CoA → delta-aminolaevulinic acid (ALA) → first and rate-limiting step "
            "of haem synthesis; erythroid-specific isoform (ALAS2); ubiquitous ALAS1 on chromosome 3)"
        ),
        "inheritance": (
            "X-LINKED SIDEROBLASTIC ANAEMIA (XLSA) — LOF hemizygous males primarily affected; "
            "heterozygous females occasionally symptomatic (skewed X-inactivation); "
            "XLSA is the most common hereditary sideroblastic anaemia (~40-50% of inherited cases); "
            "ALSO: X-LINKED PROTOPORPHYRIA (XLP) — GOF mutations in ALAS2 exon 11 (late-coding extension) → "
            "  → increased ALAS2 activity → excess protoporphyrin IX (PPIX) → erythropoietic protoporphyria-like phenotype; "
            "  XLP: phototoxicity WITHOUT ring sideroblasts (opposite end of ALAS2 phenotype spectrum); "
            "XLSA MECHANISM: "
            "  ALAS2 LOF → reduced ALA production → reduced haem synthesis → "
            "  → iron cannot be incorporated into haem → iron accumulates in mitochondria → "
            "  → RING SIDEROBLASTS: iron-laden mitochondria arranged in perinuclear ring on Prussian Blue stain; "
            "  → hypochromic, microcytic anaemia (iron trapped in mitochondria, not in Hb); "
            "PYRIDOXINE RESPONSIVENESS: "
            "  ~60-70% of XLSA patients respond to pyridoxine (vitamin B6) supplementation; "
            "  Mechanism: pyridoxine converted to PLP → substrate of ALAS2 → partially restores enzyme activity; "
            "  Pyridoxine dose: 50-200 mg/day; response: Hb improvement + reduction/elimination of ring sideroblasts; "
            "  Non-responders: pyridoxine does not overcome severe LOF mutations; "
            "MUTATION SPECTRUM: missense (most PLP-binding domain), frameshift, splice; "
            "    R452H, K299Q, R517H: common missense mutations in PLP-binding domain"
        ),
        "disease_category": (
            "X-LINKED SIDEROBLASTIC ANAEMIA (XLSA) — OMIM 300751; "
            "Also: X-Linked Protoporphyria (XLP) — OMIM 301310 (GOF, different mutations, OPPOSITE phenotype); "
            "XLSA DIAGNOSTIC CRITERIA: "
            "  1. Hypochromic microcytic anaemia (MCV ↓, MCH ↓); "
            "  2. High serum ferritin + transferrin saturation (iron-replete despite anaemia); "
            "  3. RING SIDEROBLASTS on bone marrow aspirate (Prussian Blue stain): "
            "     ≥15% erythroid precursors with ≥5 siderotic granules encircling ≥1/3 nucleus = diagnostic; "
            "  4. Elevated free erythrocyte protoporphyrin (FEP) — iron deficiency excluded; "
            "  5. X-linked family history (affected males in maternal lineage); "
            "CRITICAL DISTINCTION: "
            "  XLSA: hypochromic microcytic anaemia + ring sideroblasts + iron overload; "
            "  XLP: NORMAL erythropoiesis + photosensitivity + elevated PPIX; "
            "ACQUIRED MDS-ASSOCIATED: distinguish from acquired sideroblastic anemia (MDS-RS — somatic SF3B1 mutation); "
            "  Hereditary XLSA: younger presentation, family history, no dysplastic myeloid changes; "
            "  Acquired: older, acquired SF3B1 splicing mutation; clonal cytopenia"
        ),
        "disease_pathway": (
            "HAEM SYNTHESIS — ALAS2 RATE-LIMITING STEP: "
            "NORMAL PATHWAY IN ERYTHROID CELL: "
            "  Glycine (cytoplasm) + Succinyl-CoA (TCA cycle, mitochondria) "
            "  → ALAS2 (PLP cofactor) → delta-Aminolaevulinic acid (ALA); "
            "  ALA exported to cytoplasm → porphobilinogen (PBG, by ALAD); "
            "  Cascade: uroporphyrinogen III → coproporphyrinogen III → protoporphyrin IX (PPIX); "
            "  PPIX re-enters mitochondria → FECH (ferrochelatase) inserts Fe2+ → HAEM; "
            "  Haem → joins globin chains → haemoglobin; "
            "ALAS2 LOF CONSEQUENCE: "
            "  Reduced ALA → reduced flux through entire haem pathway → insufficient haem; "
            "  Fe2+ cannot enter haem cycle → accumulates in mitochondrial matrix; "
            "  Mitochondrial iron → Fe-S cluster toxicity + reactive oxygen species; "
            "  Excess iron precipitates as iron-sulphur granules in perinuclear mitochondria; "
            "  RING SIDEROBLAST MORPHOLOGY: Prussian Blue–positive ring = iron-laden mitochondria arranged in ring around nucleus; "
            "PLP MECHANISM: "
            "  Pyridoxal-5'-phosphate (PLP) is covalent cofactor (Schiff base with Lys-313 of ALAS2); "
            "  Mutations near PLP-binding → reduced cofactor affinity → pyridoxine (B6) supplementation restores activity"
        ),
        "pathognomonic": (
            "RING SIDEROBLASTS (≥15%) — DIAGNOSTIC OF SIDEROBLASTIC ANAEMIA (not specific to ALAS2): "
            "  Prussian Blue stain: perinuclear ring of blue granules = iron-laden mitochondria; "
            "  ≥15% ring sideroblasts → sideroblastic anaemia confirmed; "
            "  Hereditary vs acquired: family history, X-linked pattern, molecular testing; "
            "PYRIDOXINE RESPONSIVENESS PATHOGNOMONIC FOR ALAS2 (not other sideroblastic anaemia genes): "
            "  Hb improvement on pyridoxine 50-200 mg/day = hallmark of ALAS2 XLSA; "
            "  GLRX5, HSPA9, SLC25A38, TRNT1: pyridoxine NON-responsive — distinguishes ALAS2; "
            "X-LINKED FAMILY HISTORY: affected maternal uncle/grandfather confirms X-linked inheritance; "
            "IRON OVERLOAD WITHOUT IRON DEFICIENCY: "
            "  High ferritin + high transferrin saturation + microcytic anaemia = sideroblastic (not IDA)"
        ),
        "treatment": (
            "PYRIDOXINE (Vitamin B6) — FIRST-LINE: "
            "  50-200 mg/day orally; response in 60-70% of ALAS2 XLSA; "
            "  Response criterion: Hb rise ≥2 g/dL + reduced ring sideroblasts; "
            "  Continue indefinitely if responding; "
            "  PYRIDOXAL-5'-PHOSPHATE: active form; use if pyridoxine fails (conversion step bypassed); "
            "IRON CHELATION: "
            "  Non-responders: chronic transfusion → iron overload → chelation mandatory; "
            "  Deferasirox (oral) first-line; deferoxamine (SC) alternative; "
            "  Ferritin target <1000 ng/mL (cardiac/endocrine/hepatic protection); "
            "  PHLEBOTOMY: in responding patients (Hb adequate) phlebotomy preferred (cheaper, efficient); "
            "RED CELL TRANSFUSIONS: "
            "  For symptomatic non-responders; aim Hb >8 g/dL; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION: "
            "  Reserved for severe transfusion-dependent non-responders (<12 years old); "
            "  Curative but significant morbidity — reserved for young patients; "
            "AVOID FOLATE DEPLETION: "
            "  ALAS2 pathway requires B12/folate for methionine synthase → ensure adequate folate intake"
        ),
        "seed": 2814,
        "pt_vars": {
            "hb_range": (6.5, 10.5),
            "mcv_range": (60, 79),
            "ferritin_range": (450, 2800),
            "ring_sideroblast_pct": (20, 75),
            "pyridoxine_response_pct": 65,
        }
    },
    {
        "gene": "SLC25A38",
        "protein": (
            "SLC25A38 -- 3p22.1 AR -- 344aa -- "
            "Mitochondrial-Solute-Carrier-Family-25-Member-38-38kDa-"
            "Glycine-Mitochondrial-Importer-ALAS2-Substrate-Supplier-"
            "OMIM-Gene-610819-Disease-Non-Syndromic-SA-205950"
        ),
        "locus": "3p22.1",
        "protein_size": (
            "344 aa / 38 kDa (inner mitochondrial membrane transporter; imports glycine into mitochondrial matrix; "
            "glycine is substrate for ALAS2 — SLC25A38 deficiency indirectly blocks haem synthesis "
            "at the same step as ALAS2; 6-TM domain mitochondrial carrier topology)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "SLC25A38 encodes a mitochondrial glycine importer essential for haem synthesis; "
            "MECHANISM: "
            "  Glycine must enter mitochondria to serve as ALAS2 substrate; "
            "  SLC25A38 LOF → glycine cannot enter mitochondria → "
            "  → ALAS2 cannot catalyse first haem synthesis step (substrate unavailable) → "
            "  → functionally equivalent to ALAS2 deficiency → ring sideroblasts + iron overload; "
            "CLINICAL PHENOTYPE: "
            "  Severe congenital sideroblastic anaemia (typically transfusion-dependent from infancy); "
            "  More severe than ALAS2 (no pyridoxine response; glycine supplementation unsuccessful); "
            "  PYRIDOXINE NON-RESPONSIVE (mechanism: substrate block, not enzyme block → PLP cannot help); "
            "  Second most common inherited sideroblastic anaemia (~15-20% of cases); "
            "MUTATION SPECTRUM: "
            "  Missense (carrier helices of mitochondrial carrier fold), nonsense, frameshift; "
            "  No single founder mutation — population-distributed; "
            "AGE OF ONSET: neonatal to early infancy — severe anaemia from birth in most"
        ),
        "disease_category": (
            "AUTOSOMAL RECESSIVE NON-SYNDROMIC SIDEROBLASTIC ANAEMIA — OMIM 205950; "
            "FEATURES: "
            "  Severe microcytic hypochromic anaemia from birth/infancy; "
            "  Ring sideroblasts ≥15% on BM aspirate; "
            "  Iron overload (high ferritin, high transferrin saturation); "
            "  No extra-haematopoietic features (pure haematological disorder — non-syndromic); "
            "  PYRIDOXINE NON-RESPONSIVE (distinguishes from ALAS2 XLSA); "
            "COMPARISON TO ALAS2: "
            "  SLC25A38: AR, severe, non-responsive; present at birth; "
            "  ALAS2: X-linked, variable severity, 60-70% pyridoxine-responsive; milder phenotype possible"
        ),
        "disease_pathway": (
            "MITOCHONDRIAL GLYCINE IMPORT — SLC25A38 IN HAEM SYNTHESIS: "
            "GLYCINE IMPORT PATHWAY: "
            "  Cytoplasmic glycine synthesised (SHMT1/2); "
            "  SLC25A38 (inner mitochondrial membrane) → imports glycine into matrix → "
            "  ALAS2 uses glycine + succinyl-CoA → ALA → haem cascade; "
            "SLC25A38 LOF: "
            "  Mitochondrial matrix glycine deficient → ALAS2 has no substrate → "
            "  → haem synthesis blocked → iron accumulation → ring sideroblasts; "
            "ANTIPORT MECHANISM: "
            "  SLC25A38 may operate as glycine/5-aminolevulinate antiporter "
            "  (exports ALA for export step after ALAS2 reaction — bidirectional role); "
            "NO PYRIDOXINE RESCUE: "
            "  PLP supplementation cannot bypass substrate deficiency — ALAS2 enzyme active "
            "  but glycine substrate unavailable → pyridoxine fails"
        ),
        "pathognomonic": (
            "SEVERE NEONATAL SIDEROBLASTIC ANAEMIA + PYRIDOXINE NON-RESPONSE: "
            "  AR inheritance + ring sideroblasts from birth + pyridoxine failure = SLC25A38 high probability; "
            "  Confirm with WES/WGS (gene panel may miss intronic variants); "
            "PURE HAEMATOLOGICAL (NON-SYNDROMIC): "
            "  Absence of neurological, developmental, or metabolic extra-haematopoietic features; "
            "  Distinguishes from TRNT1 (immunodeficiency), PUS1 (myopathy), YARS2 (myopathy/lactic acidosis); "
            "MOLECULAR CONFIRMATION: "
            "  Biallelic SLC25A38 variants in trans; "
            "  Functional mitochondrial uptake assay (research labs)"
        ),
        "treatment": (
            "TRANSFUSION PROGRAMME (mainstay): "
            "  Chronic transfusions every 3-4 weeks from infancy; "
            "  Target Hb >8-9 g/dL; "
            "IRON CHELATION: "
            "  Deferasirox: 20-40 mg/kg/day; adjust based on ferritin; "
            "  Deferoxamine SC: alternative; "
            "  Ferritin monitoring every 3 months; liver MRI (T2*) annually; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): "
            "  CURATIVE and strongly recommended in early childhood (before iron overload); "
            "  Best results: HLA-matched sibling; unrelated donor acceptable; "
            "  Reduced-intensity conditioning; "
            "  Timing: ideally <5 years old before significant organ iron damage; "
            "PYRIDOXINE: NOT effective — do not delay definitive treatment"
        ),
        "seed": 2815,
        "pt_vars": {
            "hb_range": (5.5, 9.0),
            "mcv_range": (55, 76),
            "ferritin_range": (800, 4500),
            "ring_sideroblast_pct": (30, 85),
            "pyridoxine_response_pct": 0,
        }
    },
    {
        "gene": "GLRX5",
        "protein": (
            "GLRX5 -- 14q32.13 AR -- 157aa -- "
            "Glutaredoxin-5-18kDa-Mitochondrial-Matrix-"
            "Fe-S-Cluster-Scaffold-Iron-Sensing-IRP1-Regulator-"
            "OMIM-Gene-609588-Disease-SA-GLRX5-616860"
        ),
        "locus": "14q32.13",
        "protein_size": (
            "157 aa / 18 kDa (mitochondrial glutaredoxin; contains CXXC active site motif; "
            "assembles [2Fe-2S] clusters for delivery to aconitase, FECH, respiratory chain; "
            "regulates cytosolic iron sensing via IRP1 activity; monothiol glutaredoxin subfamily)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "GLRX5 deficiency impairs Fe-S cluster assembly in mitochondria; "
            "MECHANISM: "
            "  GLRX5 assembles [2Fe-2S] clusters → delivers to: "
            "    (1) Ferrochelatase (FECH) — FECH requires [2Fe-2S] for activity → haem synthesis; "
            "    (2) Aconitase (TCA cycle) → energy metabolism; "
            "    (3) Respiratory chain complexes I-III; "
            "  GLRX5 LOF → FECH activity severely reduced → haem synthesis final step impaired → "
            "    → iron cannot be incorporated into PPIX → mitochondrial iron accumulation; "
            "    → ring sideroblasts + free PPIX accumulation; "
            "  IRP1 DYSREGULATION: "
            "    GLRX5 LOF → IRP1 loses its own [4Fe-4S] cluster → IRP1 converts to apo-IRP1; "
            "    Apo-IRP1 binds iron-response elements (IREs) on TfR1/ferritin mRNAs → "
            "      → TfR1 mRNA stabilised (more uptake) + ferritin mRNA suppressed (less storage) → "
            "      → paradoxical cellular iron uptake despite mitochondrial iron overload → "
            "      → cytosolic iron deficiency + mitochondrial iron excess (iron partitioning defect); "
            "CLINICAL SEVERITY: variable; some adult-onset cases reported (milder than SLC25A38); "
            "PYRIDOXINE: NOT responsive"
        ),
        "disease_category": (
            "AUTOSOMAL RECESSIVE SIDEROBLASTIC ANAEMIA — OMIM 616860; "
            "DISTINCTIVE GLRX5 FEATURES: "
            "  Variable severity — early childhood to adult onset; "
            "  Ring sideroblasts + iron overload; "
            "  Elevated erythrocyte free PPIX (FECH impaired → PPIX accumulates); "
            "  Hepatic iron overload (liver damage from iron deposition); "
            "  Possible neurological involvement (TCA cycle / aconitase / mitochondrial energy deficiency); "
            "  IRP1 dysregulation → systemic iron distribution abnormalities; "
            "  Non-syndromic (primarily haematological) vs mild systemic (variable by mutation severity); "
            "IRON PROFILE: high ferritin + high transferrin saturation + high PPIX — triad helpful"
        ),
        "disease_pathway": (
            "Fe-S CLUSTER ASSEMBLY — GLRX5 IN MITOCHONDRIA: "
            "Fe-S CLUSTER BIOGENESIS PATHWAY (ISC): "
            "  Cysteine (cysteine desulfurase NFS1/ISD11) + iron (frataxin scaffold) → "
            "  → [2Fe-2S] core assembled on ISCU scaffold → "
            "  → GLRX5 accepts [2Fe-2S] cluster → "
            "  → GLRX5 delivers cluster to target apo-proteins (FECH, aconitase, RIESKE protein); "
            "FECH (ferrochelatase) REQUIREMENT: "
            "  FECH catalyses final haem synthesis step: PPIX + Fe2+ → HAEM; "
            "  FECH requires [2Fe-2S] for catalytic activity; "
            "  GLRX5 LOF → no [2Fe-2S] for FECH → FECH inactive → haem synthesis blocked; "
            "IRP1 DUAL ROLE: "
            "  [4Fe-4S]-IRP1: cytosolic aconitase (TCA cycle function — no IRE binding); "
            "  Apo-IRP1 (Fe-S depleted): RNA-binding protein (binds IREs → post-transcriptional iron regulation); "
            "  GLRX5 LOF → IRP1 loses cluster → constitutive apo-IRP1 → TfR1 ↑, ferritin ↓ → "
            "  paradoxical cellular iron uptake despite overload"
        ),
        "pathognomonic": (
            "RING SIDEROBLASTS + ELEVATED ERYTHROCYTE PPIX + Fe-S CLUSTER DEFICIENCY: "
            "  Elevated PPIX distinguishes GLRX5 (FECH impaired → PPIX accumulates) "
            "  from ALAS2 LOF (PPIX low as haem pathway upstream blocked); "
            "IRP1 DYSREGULATION SIGNATURE: "
            "  Transferrin receptor (TfR1) markedly elevated on erythrocyte surface; "
            "  Serum soluble TfR elevated; "
            "  Ferritin may be paradoxically low despite iron overload (IRP1 suppresses ferritin mRNA); "
            "MOLECULAR: biallelic GLRX5 pathogenic variants; "
            "IRON PARTITIONING DEFECT: cytoplasmic iron 'shortage' + mitochondrial iron overload"
        ),
        "treatment": (
            "TRANSFUSION + CHELATION (mainstay): "
            "  Chronic transfusions + deferasirox; "
            "  Liver iron assessment (MRI T2*) yearly; "
            "PYRIDOXINE: NOT effective; "
            "IRON RESTRICTION: theoretical benefit — reduce iron loading by limiting dietary iron/supplementation; "
            "HSCT: curative for severe cases; "
            "EXPERIMENTAL — SUCCINYLACETONE / SUCCINYLACETONATE: "
            "  Inhibits ALAD → reduces PPIX accumulation → less photosensitivity (if PPIX-related); "
            "ANTIOXIDANTS (N-acetylcysteine): reduce mitochondrial ROS from iron accumulation; "
            "MONITORING: liver MRI + cardiac MRI for iron overload annually; LFTs; "
            "NEUROLOGICAL: monitor neurodevelopment (TCA cycle involvement in some patients)"
        ),
        "seed": 2816,
        "pt_vars": {
            "hb_range": (6.0, 10.0),
            "mcv_range": (62, 80),
            "ferritin_range": (500, 3500),
            "ring_sideroblast_pct": (18, 65),
            "pyridoxine_response_pct": 0,
        }
    },
    {
        "gene": "HSPA9",
        "protein": (
            "HSPA9 -- 5q31.2 AR -- 679aa -- "
            "Heat-Shock-Protein-70kDa-Family-Member-9-Mortalin-GRP75-"
            "Mitochondrial-Matrix-Chaperone-ISC-Fe-S-Assembly-ISCU-Scaffold-Partner-"
            "OMIM-Gene-600540-Disease-SA-HSPA9-616860"
        ),
        "locus": "5q31.2",
        "protein_size": (
            "679 aa / 70 kDa (mitochondrial matrix Hsp70 chaperone; also known as mortalin/GRP75/PBP74; "
            "ATPase-dependent chaperone; partners with co-chaperones HSC20 (HSCB) and DnaJA3; "
            "required for Fe-S cluster transfer from ISCU scaffold to recipient apo-proteins — "
            "acts downstream of GLRX5 in Fe-S delivery chain)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "HSPA9 encodes mitochondrial mortalin/GRP75, an Hsp70 chaperone critical for Fe-S cluster biogenesis; "
            "MECHANISM: "
            "  ISC PATHWAY: ISCU scaffold assembles [Fe-S] cluster → "
            "    → HSPA9 (with HSC20/DnaJA3 co-chaperones) facilitates cluster release from ISCU; "
            "    → cluster transferred to GLRX5 → then to target apo-proteins (FECH, aconitase, etc.); "
            "  HSPA9 LOF → Fe-S clusters trapped on ISCU → not delivered → "
            "    → FECH deficient → haem synthesis blocked → ring sideroblasts; "
            "  ADDITIONAL ROLES: mortalin is multifunctional — also involved in: "
            "    → protein import into mitochondria (TIM23 complex partnership); "
            "    → anti-apoptotic signalling (p53 sequestration in cytoplasm); "
            "    → mtDNA maintenance (indirect); "
            "CLINICAL PHENOTYPE: "
            "  AR sideroblastic anaemia + variable systemic features; "
            "  Associated phenotype: SHORT STATURE + intellectual disability reported in some cases "
            "    (systemic Hsp70 requirement for mitochondrial biogenesis); "
            "  5q31.2 del: HSPA9 deletion contributes to 5q- MDS-like phenotype in acquired disease; "
            "PYRIDOXINE NON-RESPONSIVE"
        ),
        "disease_category": (
            "AUTOSOMAL RECESSIVE SIDEROBLASTIC ANAEMIA — OMIM-linked (allelic to GLRX5 disease spectrum); "
            "DISTINCTIVE HSPA9 FEATURES: "
            "  Sideroblastic anaemia from infancy/childhood; "
            "  Possible developmental delay or intellectual disability in some alleles; "
            "  Short stature reported (systemic mitochondrial chaperone deficiency); "
            "  Hepatic siderosis; "
            "  Molecular distinction from GLRX5: both block Fe-S transfer but at different steps; "
            "5q31.2 LOCUS NOTE: "
            "  Somatic del5q (MDS) also affects HSPA9 region; "
            "  Germline HSPA9 LOF → congenital sideroblastic anaemia distinct from del5q MDS"
        ),
        "disease_pathway": (
            "Fe-S CLUSTER DELIVERY — HSPA9 CHAPERONE ROLE: "
            "ISC PATHWAY DOWNSTREAM OF ISCU: "
            "  1. ISCU scaffold: cysteine → [2Fe-2S] cluster assembled on ISCU; "
            "  2. HSPA9 (Hsp70) + HSC20 (HSCB, co-chaperone J-protein) + DnaJA3: "
            "     → bind ISCU → ATP hydrolysis by HSPA9 → release of [2Fe-2S] from ISCU → "
            "     → cluster transfer to GLRX5 glutaredoxin (monothiol); "
            "  3. GLRX5 delivers cluster to final targets (FECH, etc.); "
            "HSPA9 LOF: "
            "  ISCU stays loaded (cluster trapped) → FECH remains apo-FECH → haem blocked; "
            "MORTALIN PLEIOTROPIC FUNCTIONS: "
            "  Protein import partner (TIM23): mitochondrial protein import may be reduced; "
            "  p53 sequestration: cytoplasmic mortalin binds p53 → prevents nuclear entry; "
            "  HSPA9 LOF → p53 released → apoptosis in rapidly dividing erythroid progenitors"
        ),
        "pathognomonic": (
            "SIDEROBLASTIC ANAEMIA + PYRIDOXINE NON-RESPONSE + AR INHERITANCE: "
            "  Points to Fe-S pathway (SLC25A38, GLRX5, HSPA9, PUS1, YARS2, TRNT1) vs ALAS2 (responsive); "
            "DEVELOPMENTAL PHENOTYPE (variable): "
            "  Intellectual disability or short stature in HSPA9-SA differentiates from pure SLC25A38 haematology; "
            "  WES/WGS required to distinguish GLRX5 vs HSPA9 vs SLC25A38 (all non-syndromic sideroblastic); "
            "5q31.2 CHROMOSOMAL CONTEXT: "
            "  Array CGH: check for del5q31 region to rule out large deletion involving HSPA9 + other genes"
        ),
        "treatment": (
            "TRANSFUSION + CHELATION (as SLC25A38); "
            "PYRIDOXINE: NOT effective; "
            "HSCT: curative option; "
            "COENZYME Q10 / MITOCHONDRIAL SUPPLEMENTS: rationale given Fe-S role in RC complexes; "
            "  Evidence limited but used empirically; "
            "DEVELOPMENTAL SUPPORT: "
            "  Neurodevelopmental assessment and early intervention if intellectual disability present; "
            "  Growth hormone evaluation if short stature documented (5q31.2 region); "
            "MONITORING: ferritin, liver MRI, cardiac MRI, LFTs, neurodevelopmental milestones"
        ),
        "seed": 2817,
        "pt_vars": {
            "hb_range": (6.0, 9.5),
            "mcv_range": (60, 78),
            "ferritin_range": (550, 3800),
            "ring_sideroblast_pct": (20, 70),
            "pyridoxine_response_pct": 0,
        }
    },
    {
        "gene": "ABCB7",
        "protein": (
            "ABCB7 -- Xq13.3 XLR -- 752aa -- "
            "ATP-Binding-Cassette-Subfamily-B-Member-7-80kDa-"
            "Inner-Mitochondrial-Membrane-Fe-S-Cluster-Exporter-"
            "X-Linked-Sideroblastic-Anaemia-with-Ataxia-XLSA-A-"
            "OMIM-Gene-300135-Disease-XLSA-A-301310"
        ),
        "locus": "Xq13.3",
        "protein_size": (
            "752 aa / 80 kDa (ABC transporter superfamily; inner mitochondrial membrane; "
            "exports Fe-S cluster intermediates from mitochondria to cytoplasm; "
            "required for cytoplasmic Fe-S protein assembly (CIA pathway supply); "
            "yeast ortholog Atm1p; structurally: 6-TM domain + NBD (nucleotide-binding domain))"
        ),
        "inheritance": (
            "X-LINKED RECESSIVE — hemizygous males affected; "
            "ABCB7 encodes a mitochondrial ABC transporter that exports Fe-S cluster precursors "
            "from mitochondria to the cytoplasm; "
            "DISTINCT SYNDROME: X-LINKED SIDEROBLASTIC ANAEMIA WITH ATAXIA (XLSA/A): "
            "  XLSA/A = sideroblastic anaemia + CEREBELLAR ATAXIA (unique combination); "
            "  Distinguishes ABCB7 from all other sideroblastic anaemia genes (none cause ataxia); "
            "MECHANISM: "
            "  ABCB7 exports [2Fe-2S] cluster precursors (as glutathione-complexed form?) → "
            "    cytoplasm → CIA (cytoplasmic iron-sulphur protein assembly) pathway; "
            "  CIA pathway provides [4Fe-4S] clusters to nuclear/cytosolic proteins: "
            "    DNA polymerase δ/ε, primase, glycosylase NTHL2, xeroderma pigmentosum D; "
            "  ABCB7 LOF: "
            "    Mitochondrial Fe-S export fails → mitochondrial iron accumulation → ring sideroblasts; "
            "    Cytoplasmic/nuclear Fe-S proteins deficient → "
            "      → DNA repair enzymes impaired → neuronal damage; "
            "      → Purkinje cell degeneration → CEREBELLAR ATAXIA; "
            "  ANAEMIA MILD: ring sideroblasts present but anaemia often mild/compensated in males; "
            "  ATAXIA: typically non-progressive or slowly progressive cerebellar type; "
            "FEMALE CARRIERS: usually unaffected (X-inactivation); occasionally mild anaemia"
        ),
        "disease_category": (
            "X-LINKED SIDEROBLASTIC ANAEMIA WITH ATAXIA (XLSA/A) — OMIM 301310; "
            "UNIQUE DUAL PHENOTYPE: "
            "  1. SIDEROBLASTIC ANAEMIA (mild): ring sideroblasts, mild hypochromic microcytic anaemia; "
            "  2. CEREBELLAR ATAXIA (non-progressive or slowly progressive): "
            "     Onset: early childhood to adult; "
            "     Features: gait ataxia, limb dysmetria, nystagmus, dysarthria; "
            "     Cerebellar atrophy on MRI (Purkinje cell loss); "
            "  PYRIDOXINE: NOT responsive; "
            "DIAGNOSIS CLUE: "
            "  X-linked ataxia + ring sideroblasts → ABCB7; "
            "  Panel: first exclude Friedreich ataxia (FXN, AR, GAA expansion) — different gene/mechanism; "
            "MRI: cerebellar atrophy; possible white matter signal; "
            "EMG: may show sensory neuropathy (large-fibre axonal in some)"
        ),
        "disease_pathway": (
            "Fe-S CLUSTER EXPORT — ABCB7 MITOCHONDRIA-TO-CYTOPLASM TRANSPORT: "
            "CIA (CYTOPLASMIC IRON-SULPHUR ASSEMBLY) PATHWAY: "
            "  Mitochondria export Fe-S precursor (glutathione-complexed, GS-[2Fe-2S]?); "
            "  ABCB7 (inner mitochondrial membrane): ATP-dependent export of precursor; "
            "  Cytoplasmic CIA pathway (CIAO1/CIAO2/CIA2B/MMS19): assembles [4Fe-4S] from precursor; "
            "  Delivers [4Fe-4S] to: "
            "    Nuclear: DNA polymerase δ/ε primase, glycosylases, XPD helicase; "
            "    Cytoplasmic: IRP1 (cytoplasmic aconitase/iron sensor), PPAT; "
            "ABCB7 LOF EFFECTS: "
            "  Mitochondrial iron accumulation → ring sideroblasts (mild haematological phenotype); "
            "  CIA pathway substrate deficient → cytoplasmic/nuclear [4Fe-4S] protein assembly fails; "
            "  Purkinje cells (cerebellum, high metabolic demand) particularly vulnerable → ataxia; "
            "  DNA repair enzymes (XPD, pol δ/ε) impaired → possible DNA damage accumulation in neurons"
        ),
        "pathognomonic": (
            "CEREBELLAR ATAXIA + RING SIDEROBLASTS + X-LINKED INHERITANCE = ABCB7 PATHOGNOMONIC: "
            "  No other sideroblastic anaemia gene causes ataxia; "
            "  No other X-linked ataxia gene causes ring sideroblasts; "
            "  The combination IS diagnostic until molecular confirmation; "
            "MILD ANAEMIA: contrast with ALAS2/SLC25A38 (moderate-severe); ABCB7 anaemia often very mild; "
            "CEREBELLAR MRI: atrophy ± white matter changes; "
            "AVOID CONFUSING WITH: "
            "  Friedreich ataxia (FXN LOF, AR, GAA repeat, severe cardiomyopathy, NO ring sideroblasts); "
            "  Other X-linked ataxias (CACNA1A EA2, etc.) — no ring sideroblasts"
        ),
        "treatment": (
            "SIDEROBLASTIC ANAEMIA (mild — often no treatment needed): "
            "  If Hb adequate: monitor only; "
            "  If anaemic: red cell transfusions as needed (often infrequent); "
            "  Iron chelation if ferritin rising; "
            "  PYRIDOXINE: NOT effective; "
            "CEREBELLAR ATAXIA: "
            "  Physiotherapy + occupational therapy (balance/mobility); "
            "  Speech therapy if dysarthria; "
            "  Assistive devices if ataxia progressive; "
            "  No disease-modifying therapy currently available; "
            "MONITORING: "
            "  Annual FBC + ferritin; "
            "  Neurological assessment 6-monthly; "
            "  Brain MRI every 3 years (cerebellar atrophy progression); "
            "GENETIC COUNSELLING: X-linked; carrier mother 50% risk affected sons"
        ),
        "seed": 2818,
        "pt_vars": {
            "hb_range": (8.5, 12.0),
            "mcv_range": (65, 84),
            "ferritin_range": (200, 1800),
            "ring_sideroblast_pct": (15, 40),
            "pyridoxine_response_pct": 0,
        }
    },
    {
        "gene": "PUS1",
        "protein": (
            "PUS1 -- 12q24.33 AR -- 445aa -- "
            "Pseudouridine-Synthase-1-51kDa-Nuclear-Mitochondrial-"
            "tRNA-Pseudouridylation-Mitochondrial-Protein-Synthesis-"
            "MLASA1-Myopathy-Lactic-Acidosis-Sideroblastic-Anaemia-"
            "OMIM-Gene-608109-Disease-MLASA1-550500"
        ),
        "locus": "12q24.33",
        "protein_size": (
            "445 aa / 51 kDa (dual-localised: nucleus and mitochondria; pseudouridine synthase — "
            "converts uridine → pseudouridine (Ψ) at specific positions in tRNAs; "
            "mitochondrial tRNA pseudouridylation required for efficient codon decoding → "
            "mitochondrial translation → OXPHOS complex assembly)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "PUS1 is a pseudouridine synthase that modifies both nuclear and mitochondrial tRNAs; "
            "SYNDROME: MLASA TYPE 1 — Myopathy, Lactic Acidosis, Sideroblastic Anaemia; "
            "MECHANISM: "
            "  PUS1 pseudouridylates mitochondrial tRNAs at specific wobble positions; "
            "  Ψ modification enhances tRNA stability and decoding efficiency; "
            "  PUS1 LOF → hypomodified mt-tRNAs → inefficient mitochondrial translation → "
            "    → OXPHOS complex subunits (mt-genome encoded) not properly synthesised; "
            "    → mitochondrial respiratory chain dysfunction → LACTIC ACIDOSIS; "
            "    → HAEM SYNTHESIS IMPAIRED (FECH requires OXPHOS-generated energy? — secondary effect); "
            "    → mitochondrial iron accumulation → RING SIDEROBLASTS; "
            "  MYOPATHY: skeletal muscle highly dependent on mitochondrial OXPHOS → proximal myopathy; "
            "CLINICAL TRIAD (MLASA1): "
            "  1. Sideroblastic anaemia (ring sideroblasts, iron overload); "
            "  2. Skeletal muscle myopathy (proximal weakness, elevated CK); "
            "  3. Lactic acidosis (fasting/exercise-induced; elevated lactate:pyruvate ratio); "
            "PYRIDOXINE: NOT responsive; "
            "COGNITIVE FUNCTION: usually intact (unlike mitochondrial encephalopathies)"
        ),
        "disease_category": (
            "MITOCHONDRIAL MYOPATHY, LACTIC ACIDOSIS, SIDEROBLASTIC ANAEMIA TYPE 1 (MLASA1) — OMIM 550500; "
            "TRIAD DISTINGUISHES FROM NON-SYNDROMIC SIDEROBLASTIC ANAEMIA: "
            "  Myopathy (proximal > distal; CK elevated 2-10x ULN); "
            "  Exercise intolerance; "
            "  Lactic acidosis (resting and/or exercise-induced; L:P ratio >25); "
            "CARDIAC: dilated cardiomyopathy reported in some cases (OXPHOS failure in cardiomyocytes); "
            "GROWTH: short stature common (chronic anaemia + OXPHOS deficiency); "
            "PROGNOSIS: variable — ranges from mild myopathy with compensated anaemia to severe multi-organ failure"
        ),
        "disease_pathway": (
            "MITOCHONDRIAL tRNA PSEUDOURIDYLATION — PUS1 PATHWAY: "
            "PUS1 FUNCTION: "
            "  Pseudouridine (Ψ) = most common RNA modification; "
            "  PUS1 introduces Ψ at positions 27/28 in mt-tRNAs + cytoplasmic tRNAs; "
            "  Ψ increases tRNA base stacking and ribosome A-site stability → improved codon reading; "
            "PUS1 LOF → UNMODIFIED mt-tRNAs: "
            "  mt-tRNA decoding efficiency ↓ → mitochondrial ribosome stalls; "
            "  OXPHOS complex subunits (encoded by mt-genome — e.g., ND1-ND6, COX1-3, ATP6/8, Cyt b) "
            "    incompletely translated → respiratory chain assembly incomplete; "
            "LACTIC ACIDOSIS MECHANISM: "
            "  OXPHOS failure → pyruvate → lactate (anaerobic) → lactic acidosis; "
            "SIDEROBLASTIC ANAEMIA MECHANISM: "
            "  Mitochondrial translation failure → FECH activity indirectly reduced; "
            "  + ALAS2 may require OXPHOS for optimal function; "
            "  Mitochondrial iron accumulation → ring sideroblasts; "
            "MUSCLE: type 1 fibres (high OXPHOS demand) preferentially affected → proximal myopathy"
        ),
        "pathognomonic": (
            "MLASA TRIAD = PUS1 OR YARS2: "
            "  Sideroblastic anaemia + skeletal myopathy + lactic acidosis → PUS1 or YARS2; "
            "  Distinguish PUS1 vs YARS2 by molecular testing (WES/targeted panel); "
            "  PUS1 also modifies nuclear tRNAs — nuclear tRNA modification signature (research tool); "
            "MUSCLE BIOPSY: ragged-red fibres (Gomori trichrome), COX-negative fibres; "
            "MITOCHONDRIAL FUNCTION: complex I/III/IV reduction on OXPHOS enzyme analysis; "
            "RAISED LACTATE:PYRUVATE >25 at rest or post-exercise; "
            "CARDIAC ECHO: dilated cardiomyopathy surveillance important"
        ),
        "treatment": (
            "SIDEROBLASTIC ANAEMIA: transfusion + chelation (as per SLC25A38); PYRIDOXINE NOT effective; "
            "LACTIC ACIDOSIS: "
            "  Avoid fasting and strenuous exercise; "
            "  Sodium bicarbonate for acute acidosis; "
            "  Dichloroacetate (DCA) — activates PDC → reduces lactate; limited evidence; "
            "MYOPATHY: "
            "  Physiotherapy (aerobic training — careful titration); "
            "  CoQ10 (100-300 mg/day) — empirical OXPHOS support; "
            "  Riboflavin (FAD → complex I/II cofactor); "
            "  AVOID: statin myotoxicity (mitochondrial overlap); "
            "CARDIAC MONITORING: echo annually (dilated cardiomyopathy); "
            "HSCT: curative for haematological component; DOES NOT correct myopathy/lactic acidosis; "
            "DIETITIAN: high-carbohydrate, low-fat diet may help (reduces fat oxidation demand on mitochondria)"
        ),
        "seed": 2819,
        "pt_vars": {
            "hb_range": (6.5, 10.5),
            "mcv_range": (62, 80),
            "ferritin_range": (400, 3000),
            "ring_sideroblast_pct": (18, 60),
            "pyridoxine_response_pct": 0,
        }
    },
    {
        "gene": "YARS2",
        "protein": (
            "YARS2 -- 12p11.21 AR -- 477aa -- "
            "Mitochondrial-Tyrosyl-tRNA-Synthetase-54kDa-"
            "mt-tRNA-Tyr-Aminoacylation-Mitochondrial-Translation-"
            "MLASA2-Myopathy-Lactic-Acidosis-Sideroblastic-Anaemia-Type-2-"
            "OMIM-Gene-610957-Disease-MLASA2-613561"
        ),
        "locus": "12p11.21",
        "protein_size": (
            "477 aa / 54 kDa (mitochondria-specific class I aminoacyl-tRNA synthetase; "
            "charges mt-tRNA(Tyr) with tyrosine; two-domain structure: "
            "catalytic Rossmann-fold (N-terminal) + anticodon-binding domain (C-terminal); "
            "homodimeric enzyme in mitochondria)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "YARS2 encodes mitochondrial tyrosyl-tRNA synthetase (mt-TyrRS); "
            "SYNDROME: MLASA TYPE 2 — identical triad to MLASA1 (PUS1) at clinical level; "
            "MECHANISM: "
            "  YARS2 aminoacylates mt-tRNA(Tyr) with tyrosine; "
            "  Aminoacylated mt-tRNA(Tyr) required for all tyrosine codon decoding during mt-translation; "
            "  Tyrosine-containing OXPHOS subunits: ND1-6 (complex I), COX1-3 (complex IV), Cyt b (complex III); "
            "  YARS2 LOF → uncharged mt-tRNA(Tyr) → ribosome stalling at Tyr codons → "
            "    → OXPHOS subunit synthesis reduced → respiratory chain failure → lactic acidosis; "
            "  Mitochondrial energy failure → ALAS2 requires succinyl-CoA from TCA cycle → "
            "    → TCA cycle impaired by OXPHOS failure → haem synthesis downstream affected → ring sideroblasts; "
            "SEVERITY: generally more severe than PUS1 in published series; "
            "  Dilated cardiomyopathy more common in YARS2 than PUS1; "
            "PYRIDOXINE: NOT responsive; "
            "MUTATION HOTSPOT: p.Phe52Leu — most common YARS2 pathogenic variant (Rossmann fold)"
        ),
        "disease_category": (
            "MITOCHONDRIAL MYOPATHY, LACTIC ACIDOSIS, SIDEROBLASTIC ANAEMIA TYPE 2 (MLASA2) — OMIM 613561; "
            "PHENOTYPE IDENTICAL TO PUS1/MLASA1 IN PRINCIPLE: sideroblastic anaemia + myopathy + lactic acidosis; "
            "DISTINGUISHING FEATURES vs PUS1: "
            "  YARS2 typically more severe (earlier onset, more severe cardiomyopathy); "
            "  Dilated cardiomyopathy 2x more common in YARS2 than PUS1; "
            "  p.Phe52Leu hotspot accounts for >50% of YARS2 pathogenic alleles; "
            "  Molecular confirmation required (not distinguishable clinically); "
            "CARDIAC: DCM is leading cause of premature death in YARS2 — cardiac surveillance critical"
        ),
        "disease_pathway": (
            "MITOCHONDRIAL TYROSYL-tRNA AMINOACYLATION — YARS2 PATHWAY: "
            "AMINOACYLATION (CHARGING): "
            "  Amino acid (Tyr) + tRNA + ATP → aminoacyl-tRNA + AMP + PPi; "
            "  YARS2 specifically charges mt-tRNA(Tyr) (one of 22 mt-tRNAs); "
            "  Aminoacyl-tRNA delivered to mitochondrial ribosome → EF-Tu → A site → peptide bond; "
            "YARS2 LOF: "
            "  Uncharged mt-tRNA(Tyr) presented to ribosome → "
            "  Misacylation or stalling at Tyr codons → premature termination; "
            "  OXPHOS protein synthesis impaired (all Tyr-containing subunits); "
            "TYROSINE-RICH OXPHOS SUBUNITS: "
            "  COX1 (complex IV): 14 Tyr residues → particularly vulnerable; "
            "  Complex IV deficiency → COX-deficient fibres on muscle biopsy (COX stain); "
            "RING SIDEROBLASTS: "
            "  Mitochondrial energy failure → impaired haem synthesis step (FECH, TCA-succinyl-CoA) → "
            "  mitochondrial iron accumulation → ring sideroblasts (secondary to OXPHOS failure)"
        ),
        "pathognomonic": (
            "MLASA2 MOLECULAR HOTSPOT — p.Phe52Leu IN YARS2: "
            "  Most common YARS2 variant worldwide; homozygosity frequent in consanguineous families; "
            "  Targeted sequencing YARS2 exon 2 (p.Phe52Leu) fast first-tier test; "
            "COX-DEFICIENT FIBRES ON MUSCLE BIOPSY: "
            "  COX (complex IV) disproportionately affected — COX stain shows mosaic deficiency; "
            "  Ragged-red fibres + COX-negative fibres on Gomori trichrome; "
            "DCM EARLIER/MORE SEVERE THAN PUS1: "
            "  Echo must be performed at diagnosis and annually; "
            "  DCM may present as sudden cardiac death if untreated; "
            "YARS2 vs PUS1: "
            "  Both cause MLASA; YARS2 has hotspot p.F52L, earlier/more severe cardiomyopathy; "
            "  Distinguish by WES/panel"
        ),
        "treatment": (
            "SAME MLASA FRAMEWORK AS PUS1: "
            "  Transfusion + chelation (anaemia); CoQ10 + riboflavin (OXPHOS support); "
            "  Physiotherapy (myopathy); lactic acidosis management (avoid fasting); "
            "CARDIAC (CRITICAL — YARS2 SPECIFIC): "
            "  CARDIAC ECHO AT DIAGNOSIS (DCM more common/severe than PUS1); "
            "  ACE inhibitor / ARB early if DCM develops; "
            "  Beta-blocker (carvedilol) for systolic dysfunction; "
            "  ICD consideration if LVEF <35% or VT; "
            "  Cardiac transplant considered in refractory DCM (haematological remission post-HSCT first); "
            "HSCT: curative for haematological phenotype; "
            "  Post-HSCT: myopathy/lactic acidosis/cardiomyopathy persist (mitochondrial defect remains in non-haematopoietic cells); "
            "GENETIC COUNSELLING: p.F52L hotspot — rapid targeted genotyping available"
        ),
        "seed": 2820,
        "pt_vars": {
            "hb_range": (6.0, 10.0),
            "mcv_range": (60, 79),
            "ferritin_range": (500, 3500),
            "ring_sideroblast_pct": (20, 65),
            "pyridoxine_response_pct": 0,
        }
    },
    {
        "gene": "TRNT1",
        "protein": (
            "TRNT1 -- 3p26.2 AR -- 405aa -- "
            "tRNA-Nucleotidyltransferase-1-45kDa-CCA-Adding-Enzyme-"
            "Nuclear-Mitochondrial-tRNA-3prime-CCA-End-Repair-"
            "SIFD-Sideroblastic-Anaemia-Immunodeficiency-Fever-Developmental-Delay-"
            "OMIM-Gene-610071-Disease-SIFD-616084"
        ),
        "locus": "3p26.2",
        "protein_size": (
            "405 aa / 45 kDa (CCA-adding enzyme — class II nucleotidyltransferase; "
            "adds or repairs the universal 3'-CCA end of all tRNAs; "
            "dual localisation: nucleus (cytoplasmic tRNAs) + mitochondria (mt-tRNAs); "
            "CCA end is OBLIGATORY for aminoacylation — without it, tRNA cannot be charged)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "TRNT1 is the CCA-adding enzyme — repairs or adds the universal CCA-3'-end to tRNAs; "
            "SYNDROME: SIFD — Sideroblastic anaemia, Immunodeficiency, Fever, Developmental delay; "
            "  Unique multi-system syndrome not seen in other sideroblastic anaemia genes; "
            "MECHANISM: "
            "  All tRNAs require intact 3'-CCA for aminoacylation (charging) by aminoacyl-tRNA synthetases; "
            "  CCA end turns over (degrades) and must be continuously repaired by TRNT1; "
            "  TRNT1 LOF → tRNAs lose CCA → cannot be charged → translation failure → "
            "    → Mitochondrial: OXPHOS failure → ring sideroblasts + lactic acidosis; "
            "    → Nuclear/cytoplasmic: protein synthesis affected → immune cells, neurons; "
            "IMMUNODEFICIENCY: "
            "  B-cell and T-cell development impaired (rapid lymphocyte proliferation requires intact translation); "
            "  Combined immunodeficiency (CID) — NOT severe (different from IL2RG/ADA-SCID); "
            "  Recurrent bacterial + opportunistic infections; "
            "PERIODIC FEVER: unexplained fevers, sometimes neutrophilia, resembles autoinflammatory; "
            "DEVELOPMENTAL DELAY: psychomotor delay, intellectual disability (cytoplasmic translation); "
            "SENSORINEURAL HEARING LOSS: cochlear hair cells affected (high translational demand); "
            "PYRIDOXINE: NOT responsive"
        ),
        "disease_category": (
            "SIFD SYNDROME — OMIM 616084: "
            "DIAGNOSTIC CRITERIA: "
            "  1. SIDEROBLASTIC ANAEMIA (ring sideroblasts ≥15%); "
            "  2. IMMUNODEFICIENCY (B and T cell defects; recurrent infections; low Ig); "
            "  3. FEVER (periodic, without infection — autoinflammatory pattern); "
            "  4. DEVELOPMENTAL DELAY (intellectual disability, psychomotor delay); "
            "ADDITIONAL FEATURES: "
            "  Sensorineural hearing loss (cochlear); "
            "  Retinitis pigmentosa (some cases — photoreceptor translation demand); "
            "  Dilated cardiomyopathy; "
            "  Growth failure; "
            "SEVERITY: severe; high infant/child mortality without early HSCT; "
            "COMPARISON TO OTHER GENES: "
            "  Only sideroblastic anaemia gene with IMMUNODEFICIENCY; "
            "  SIFD syndrome pathognomonic for TRNT1 when all four features present"
        ),
        "disease_pathway": (
            "tRNA 3'-CCA END MAINTENANCE — TRNT1 UNIVERSAL FUNCTION: "
            "CCA-ADDING ENZYME MECHANISM: "
            "  All tRNAs have invariant 3'-CCA end (CCA is encoded at end of nuclear tRNA genes "
            "  but added post-transcriptionally to mitochondrial tRNAs and repaired in both compartments); "
            "  TRNT1 activity: template-independent nucleotidyl transfer — "
            "    tRNA-OH + CTP → tRNA-C; "
            "    tRNA-C-OH + CTP → tRNA-CC; "
            "    tRNA-CC-OH + ATP → tRNA-CCA; "
            "  CCA tail is recognition site for all aminoacyl-tRNA synthetases + EF-Tu + ribosome A-site; "
            "TRNT1 LOF: "
            "  tRNA-CCA ends not repaired → aminoacylation by all tRNA synthetases impaired; "
            "  BOTH mitochondrial and cytoplasmic translation fail → multi-organ effects; "
            "IMMUNODEFICIENCY: "
            "  Lymphocyte proliferation requires rapid protein synthesis → "
            "  TRNT1 LOF → T and B cells cannot proliferate on antigen stimulation → CID; "
            "PERIODIC FEVER: "
            "  Mechanism uncertain — possibly dysregulated cytokine response from inadequate translation control; "
            "  Pattern resembles hyperinflammatory autoinflammation"
        ),
        "pathognomonic": (
            "SIFD = TRNT1 PATHOGNOMONIC (when all 4 features present): "
            "  Sideroblastic anaemia + Immunodeficiency + Fever + Developmental delay; "
            "  No other hereditary sideroblastic anaemia gene causes immunodeficiency; "
            "  IMMUNODEFICIENCY DISTINGUISHES TRNT1 from all other sideroblastic anaemia genes; "
            "HEARING LOSS: "
            "  SNHL in sideroblastic anaemia → workup for TRNT1; "
            "PERIODIC FEVER + ANAEMIA: "
            "  May initially mimic autoinflammatory disease or haemophagocytic syndrome; "
            "  Ring sideroblasts on BM aspirate confirms sideroblastic origin; "
            "EARLIEST MARKER: "
            "  Anaemia + fever in neonate/infant → bone marrow (ring sideroblasts) → TRNT1 WES/panel"
        ),
        "treatment": (
            "SIDEROBLASTIC ANAEMIA: transfusion + chelation; PYRIDOXINE NOT effective; "
            "IMMUNODEFICIENCY: "
            "  Immunoglobulin replacement (IVIg every 4 weeks — covers B cell defect); "
            "  PCP prophylaxis (co-trimoxazole); "
            "  Antifungal prophylaxis (fluconazole); "
            "  Avoid live vaccines (CID patient); "
            "FEVER MANAGEMENT: "
            "  Antipyretics (paracetamol); "
            "  NSAIDs for autoinflammatory episodes; "
            "  Anakinra (IL-1 blocker) — empirical use for periodic fever component; "
            "DEVELOPMENTAL SUPPORT: early intervention, special educational needs; "
            "HEARING: hearing aids / cochlear implant if SNHL severe; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): "
            "  CURATIVE for haematological and immunological components; "
            "  Strongly recommended early (ideally <2 years old before developmental sequelae accumulate); "
            "  Developmental delay and hearing loss may persist post-HSCT (cytoplasmic translation defect persists)"
        ),
        "seed": 2821,
        "pt_vars": {
            "hb_range": (5.5, 9.0),
            "mcv_range": (58, 78),
            "ferritin_range": (600, 4000),
            "ring_sideroblast_pct": (20, 70),
            "pyridoxine_response_pct": 0,
        }
    },
]

DEFINITIONS = {
    "definitions": [
        {
            "term": "Ring Sideroblast",
            "definition": (
                "Erythroid precursor (normoblast/erythroblast) in bone marrow with ≥5 Prussian Blue–positive "
                "iron granules arranged in a PERINUCLEAR RING encircling ≥1/3 of the nucleus; "
                "Iron granules = iron-laden mitochondria arranged around nucleus (perinuclear mitochondria); "
                "Diagnostic threshold: ≥15% ring sideroblasts among erythroid precursors = SIDEROBLASTIC ANAEMIA; "
                "Causes: hereditary (ALAS2, SLC25A38, GLRX5, ABCB7, PUS1, YARS2, TRNT1) or acquired (SF3B1 mutation in MDS-RS)"
            )
        },
        {
            "term": "ALAS2 (5-Aminolevulinate Synthase 2)",
            "definition": (
                "Rate-limiting enzyme of haem biosynthesis in erythroid cells; "
                "Pyridoxal-5'-phosphate (PLP) dependent — glycine + succinyl-CoA → ALA; "
                "LOF (X-linked): XLSA — hypochromic microcytic anaemia + ring sideroblasts + iron overload; "
                "60-70% PYRIDOXINE-RESPONSIVE (PLP supplementation partially restores enzyme); "
                "GOF (exon 11 extension): X-Linked Protoporphyria (XLP) — photosensitivity, elevated PPIX, NO ring sideroblasts"
            )
        },
        {
            "term": "Pyridoxine Responsiveness",
            "definition": (
                "Hb improvement (≥2 g/dL rise) + reduction in ring sideroblasts on pyridoxine supplementation "
                "(50-200 mg/day, active form: pyridoxal-5'-phosphate/PLP); "
                "PATHOGNOMONIC FOR ALAS2 XLSA among hereditary sideroblastic anaemias; "
                "Mechanism: PLP is ALAS2 cofactor — supplementation increases residual mutant ALAS2 activity; "
                "NON-RESPONSIVE GENES: SLC25A38, GLRX5, HSPA9, ABCB7, PUS1, YARS2, TRNT1 — "
                "pyridoxine does NOT help (different mechanism, not enzyme cofactor issue)"
            )
        },
        {
            "term": "Fe-S Cluster (Iron-Sulphur Cluster)",
            "definition": (
                "Inorganic cofactor [2Fe-2S] or [4Fe-4S] coordinated by cysteine residues; "
                "Essential for: FECH (haem synthesis), aconitase (TCA cycle), respiratory chain complexes I/II/III, "
                "  IRP1 (iron sensing), DNA polymerase δ/ε, XPD helicase; "
                "Biosynthesis: ISC pathway in mitochondria (NFS1→ISCU→GLRX5→HSPA9→target proteins); "
                "Export to cytoplasm: ABCB7 → CIA pathway for cytoplasmic/nuclear [4Fe-4S] proteins; "
                "Defects: GLRX5, HSPA9, ABCB7 all cause sideroblastic anaemia via Fe-S assembly/export failure"
            )
        },
        {
            "term": "MLASA (Mitochondrial Myopathy, Lactic Acidosis, Sideroblastic Anaemia)",
            "definition": (
                "Triad syndrome caused by PUS1 (MLASA1, OMIM 550500) or YARS2 (MLASA2, OMIM 613561); "
                "All three components result from mitochondrial translation impairment: "
                "  Myopathy: OXPHOS failure in skeletal muscle (proximal weakness, CK elevated, ragged-red fibres); "
                "  Lactic acidosis: OXPHOS failure → anaerobic glycolysis → lactic acid; "
                "  Sideroblastic anaemia: OXPHOS failure → haem synthesis impaired → ring sideroblasts; "
                "Distinguishable from non-MLASA genes (ALAS2, SLC25A38, GLRX5, ABCB7, TRNT1) by myopathy + lactic acidosis"
            )
        },
        {
            "term": "SIFD Syndrome (TRNT1)",
            "definition": (
                "Sideroblastic anaemia + Immunodeficiency + periodic Fever + Developmental delay; "
                "Caused by biallelic TRNT1 LOF; "
                "TRNT1 = CCA-adding enzyme — adds/repairs 3'-CCA end of ALL tRNAs (universal requirement); "
                "TRNT1 LOF → both mitochondrial AND cytoplasmic translation impaired → multi-organ failure; "
                "IMMUNODEFICIENCY: combined B+T cell defect (distinguishes from all other sideroblastic anaemia genes); "
                "Only sideroblastic anaemia gene causing immunodeficiency — PATHOGNOMONIC combination"
            )
        },
        {
            "term": "XLSA/A (X-Linked Sideroblastic Anaemia with Ataxia) — ABCB7",
            "definition": (
                "Sideroblastic anaemia (mild) + CEREBELLAR ATAXIA — caused by ABCB7 LOF; "
                "ABCB7 exports Fe-S precursors from mitochondria to cytoplasm (CIA pathway supply); "
                "ABCB7 LOF → Fe-S export fails → cytoplasmic/nuclear Fe-S proteins deficient → "
                "  Purkinje cell degeneration → cerebellar ataxia; "
                "ANAEMIA mild (often compensated); ATAXIA non-progressive or slowly progressive; "
                "ONLY sideroblastic anaemia gene causing cerebellar ataxia — pathognomonic combination; "
                "Distinguishes from Friedreich ataxia (FXN, AR, GAA expansion, cardiomyopathy, NO ring sideroblasts)"
            )
        },
        {
            "term": "Mitochondrial Iron Overload (Ring Sideroblast Mechanism)",
            "definition": (
                "Iron accumulates in mitochondria when haem synthesis is blocked at any step: "
                "  ALAS2 LOF: ALA not made → iron cannot enter haem pathway → trapped; "
                "  SLC25A38 LOF: glycine not imported → ALAS2 substrate lacking → same; "
                "  GLRX5/HSPA9: FECH lacks Fe-S → cannot insert Fe2+ into PPIX → iron trapped; "
                "  PUS1/YARS2: OXPHOS failure → TCA-succinyl-CoA reduced → ALAS2 impaired → iron trapped; "
                "  TRNT1: translation failure → all of the above; "
                "Iron-laden mitochondria → Prussian Blue–positive perinuclear ring → ring sideroblast morphology"
            )
        },
        {
            "term": "SF3B1 and Acquired Sideroblastic Anaemia (MDS-RS)",
            "definition": (
                "SF3B1 somatic mutation causes >80% of MDS with ring sideroblasts (MDS-RS); "
                "SF3B1 = spliceosome component → mutation → aberrant splicing of ABCB7 mRNA → "
                "  functional ABCB7 deficiency → ring sideroblasts (acquired, not hereditary); "
                "MDS-RS: older patients (>60y), acquired, clonal cytopenia; "
                "DISTINGUISH FROM HEREDITARY: family history, age of onset, molecular: WES shows SF3B1 hot-spot R625H; "
                "Luspatercept (TGF-β trap) FDA-approved for MDS-RS (transfusion-reduction); "
                "NOT a treatment for hereditary sideroblastic anaemia"
            )
        },
        {
            "term": "Ferrochelatase (FECH) — Final Haem Synthesis Step",
            "definition": (
                "FECH catalyses insertion of Fe2+ into protoporphyrin IX (PPIX) → haem; "
                "Requires [2Fe-2S] cluster for activity (supplied by GLRX5 via ISC pathway); "
                "FECH LOF (germline biallelic AR): Erythropoietic Protoporphyria (EPP) — NO ring sideroblasts "
                "  (iron CAN enter haem pathway but FECH cannot complete it → PPIX accumulates → photosensitivity); "
                "GLRX5 LOF → FECH indirectly deficient (missing Fe-S cofactor) → ring sideroblasts + elevated PPIX; "
                "Critical distinction: EPP (FECH biallelic LOF) vs GLRX5 LOF (both elevate PPIX but GLRX5 has ring sideroblasts)"
            )
        },
        {
            "term": "Pseudouridine (Ψ) — tRNA Modification",
            "definition": (
                "Most abundant RNA modification; uridine isomerised to pseudouridine by PUS1 and other synthases; "
                "Ψ at wobble positions of tRNA → increased tRNA stability, enhanced codon-anticodon interaction; "
                "PUS1 modifies both nuclear and mitochondrial tRNAs; "
                "PUS1 LOF → hypomodified mt-tRNAs → reduced decoding efficiency → OXPHOS subunit synthesis impaired; "
                "MLASA1 results from this mitochondrial translation impairment"
            )
        },
        {
            "term": "tRNA CCA End — TRNT1 Substrate",
            "definition": (
                "All tRNAs end in invariant 3'-CCA sequence (required for aminoacylation); "
                "Nuclear tRNA: CCA encoded in gene but repaired by TRNT1 after use; "
                "Mitochondrial tRNA: CCA entirely post-transcriptionally added by TRNT1 (not encoded in mt-genome); "
                "TRNT1 = CCA-adding enzyme (class II nucleotidyltransferase); "
                "TRNT1 LOF → tRNA-CCA ends lost → all tRNAs unchargeable → ALL translation (nuclear + mitochondrial) impaired"
            )
        },
    ],
    "standards": [
        "BSH Guidelines — Diagnosis and Management of Hereditary Sideroblastic Anaemia (2020)",
        "EHA Recommendations — Sideroblastic Anaemia (2021)",
        "OMIM: XLSA (300751/ALAS2), SLC25A38-SA (205950), GLRX5-SA (616860), XLSA/A (301310/ABCB7), MLASA1 (550500/PUS1), MLASA2 (613561/YARS2), SIFD (616084/TRNT1)",
        "Ducamp S & Fleming MD. The molecular genetics of sideroblastic anemia. Blood. 2019;133(1):59-69",
        "Camaschella C. Hereditary sideroblastic anaemias: pathophysiology, diagnosis, and treatment. Semin Hematol. 2009;46(4):371-7",
        "Bergmann AK et al. Systematic molecular genetic analysis of congenital sideroblastic anemia. Ann Hematol. 2010",
        "FECH-EPP vs GLRX5: Crooks DR et al. Acute growth hormone insensitivity and lysine catabolism defects, Brain 2012 (HSPA9 mortalin)",
        "Yien YY & Paw BH. A role for iron deficiency in dopaminergic neurodegeneration. Proc Natl Acad Sci. 2016",
    ]
}


def _make_patients(gene_data):
    rng = random.Random(gene_data["seed"])
    pts = []
    hb_lo, hb_hi = gene_data["pt_vars"]["hb_range"]
    mcv_lo, mcv_hi = gene_data["pt_vars"]["mcv_range"]
    ferr_lo, ferr_hi = gene_data["pt_vars"]["ferritin_range"]
    rs_lo, rs_hi = gene_data["pt_vars"]["ring_sideroblast_pct"]
    pyb6_resp_pct = gene_data["pt_vars"]["pyridoxine_response_pct"]

    for i in range(40):
        hb = round(rng.uniform(hb_lo, hb_hi), 1)
        mcv = round(rng.uniform(mcv_lo, mcv_hi), 1)
        ferritin = round(rng.uniform(ferr_lo, ferr_hi), 0)
        ring_sb_pct = round(rng.uniform(rs_lo, rs_hi), 1)
        age_diag = round(rng.uniform(0.0, 35.0), 1)
        pyb6_response = rng.random() < pyb6_resp_pct / 100
        transfusion_dep = not pyb6_response
        hsct = rng.random() < 0.20
        syndromic = gene_data["gene"] in ("PUS1", "YARS2", "TRNT1", "ABCB7")
        extra_feature = rng.random() < (0.70 if syndromic else 0.10)

        sex = "M" if gene_data["gene"] in ("ALAS2", "ABCB7") else rng.choice(["M", "F"])

        pts.append({
            "patient_id": f"{gene_data['gene']}-{i+1:03d}",
            "gene": gene_data["gene"],
            "sex": sex,
            "age_at_diagnosis_years": age_diag,
            "hb_gdl": hb,
            "mcv_fl": mcv,
            "ferritin_ngml": ferritin,
            "ring_sideroblast_pct": ring_sb_pct,
            "pyridoxine_response": pyb6_response,
            "transfusion_dependent": transfusion_dep,
            "hsct_performed": hsct,
            "extra_syndromic_feature": extra_feature,
            "seed": gene_data["seed"],
        })
    return pts


def generate_overview():
    total_patients = 0
    all_genes = []

    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        total_patients += len(patients)
        pyb6_resp = sum(1 for p in patients if p["pyridoxine_response"])
        transfusion_dep = sum(1 for p in patients if p["transfusion_dependent"])
        hsct = sum(1 for p in patients if p["hsct_performed"])
        syndromic = sum(1 for p in patients if p["extra_syndromic_feature"])

        all_genes.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "disease_category": gene_data["disease_category"],
            "inheritance": gene_data["inheritance"],
            "n_patients": len(patients),
            "median_hb": round(
                sorted(p["hb_gdl"] for p in patients)[len(patients) // 2], 1
            ),
            "mean_ring_sideroblast_pct": round(
                sum(p["ring_sideroblast_pct"] for p in patients) / len(patients), 1
            ),
            "pct_pyridoxine_response": round(pyb6_resp / len(patients) * 100, 1),
            "pct_transfusion_dep": round(transfusion_dep / len(patients) * 100, 1),
            "pct_hsct": round(hsct / len(patients) * 100, 1),
            "pct_syndromic_feature": round(syndromic / len(patients) * 100, 1),
            "seed": gene_data["seed"],
        })

    return {
        "atlas": "Hereditary Sideroblastic Anemia Atlas",
        "subtitle": "Complete 8-Gene Sideroblastic Anaemia Reference — ALAS2·SLC25A38·GLRX5·HSPA9·ABCB7·PUS1·YARS2·TRNT1",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "gene_summaries": all_genes,
        "seeds": "2814-2821",
        "pathway_categories": [
            {
                "pathway": "Haem Synthesis — Substrate Supply",
                "genes": ["ALAS2", "SLC25A38"],
                "note": (
                    "ALAS2: rate-limiting enzyme (glycine + succinyl-CoA → ALA); pyridoxine-responsive 60-70%; "
                    "SLC25A38: mitochondrial glycine importer → supplies ALAS2 substrate; "
                    "Both block haem synthesis at the same step (ALA formation); "
                    "ALAS2: X-linked; SLC25A38: autosomal recessive"
                ),
            },
            {
                "pathway": "Fe-S Cluster Assembly / Delivery (ISC Pathway)",
                "genes": ["GLRX5", "HSPA9"],
                "note": (
                    "GLRX5: [2Fe-2S] scaffold → delivers cluster to FECH and aconitase; IRP1 dysregulation; "
                    "HSPA9: Hsp70 chaperone → releases [2Fe-2S] from ISCU scaffold → passes to GLRX5; "
                    "Both impair FECH (haem synthesis) and TCA cycle (aconitase); "
                    "Elevated erythrocyte PPIX distinguishes GLRX5/HSPA9 from ALAS2 (upstream block)"
                ),
            },
            {
                "pathway": "Fe-S Cluster Export (CIA Pathway) — X-Linked",
                "genes": ["ABCB7"],
                "note": (
                    "ABCB7: ABC transporter exports Fe-S precursors from mitochondria to cytoplasm; "
                    "CIA pathway (cytoplasmic Fe-S assembly) then provides [4Fe-4S] to nuclear/cytoplasmic proteins; "
                    "ABCB7 LOF → cytoplasmic Fe-S deficient → Purkinje cell degeneration → CEREBELLAR ATAXIA; "
                    "XLSA/A: only sideroblastic anaemia gene causing ataxia; mild anaemia"
                ),
            },
            {
                "pathway": "Mitochondrial Translation — tRNA Modification (MLASA)",
                "genes": ["PUS1", "YARS2"],
                "note": (
                    "PUS1: pseudouridine synthase → modifies mt-tRNAs → improves decoding efficiency; "
                    "YARS2: mitochondrial tyrosyl-tRNA synthetase → charges mt-tRNA(Tyr); "
                    "Both impair OXPHOS complex synthesis → lactic acidosis + myopathy + sideroblastic anaemia (MLASA triad); "
                    "YARS2 has p.F52L hotspot and more severe cardiomyopathy"
                ),
            },
            {
                "pathway": "Universal tRNA 3'-CCA Maintenance — Multi-System",
                "genes": ["TRNT1"],
                "note": (
                    "TRNT1: CCA-adding enzyme → repairs/adds 3'-CCA to ALL tRNAs (nuclear + mitochondrial); "
                    "TRNT1 LOF → both mitochondrial AND cytoplasmic translation fail; "
                    "SIFD syndrome: Sideroblastic anaemia + Immunodeficiency + Fever + Developmental delay; "
                    "Only sideroblastic anaemia gene causing immunodeficiency — PATHOGNOMONIC"
                ),
            },
        ],
        "critical_distinctions": [
            "PYRIDOXINE RESPONSE PATHOGNOMONIC FOR ALAS2: 60-70% respond; SLC25A38/GLRX5/HSPA9/ABCB7/PUS1/YARS2/TRNT1 = NON-RESPONSIVE — do NOT delay chelation or HSCT for a pyridoxine trial if not ALAS2",
            "RING SIDEROBLASTS vs IRON DEFICIENCY ANAEMIA: both microcytic; IDA has low ferritin + low transferrin saturation; SA has HIGH ferritin + HIGH transferrin saturation + ring sideroblasts on BM",
            "ABCB7 ATAXIA: only SA gene causing cerebellar ataxia; X-linked; mild anaemia; distinguishes from Friedreich ataxia (FXN, AR, cardiomyopathy, NO ring sideroblasts)",
            "TRNT1 IMMUNODEFICIENCY: SIFD is only SA gene with combined immunodeficiency — give IVIg + PCP prophylaxis + antifungal; avoid live vaccines",
            "PUS1 vs YARS2 (MLASA): clinically identical triad; YARS2 has p.F52L hotspot + more severe cardiomyopathy → cardiac echo at diagnosis mandatory for YARS2",
            "GLRX5 vs ALAS2: both have elevated PPIX in GLRX5 (FECH inactive) but ALAS2 has LOW PPIX (upstream block — nothing to accumulate); serum/erythrocyte PPIX helps distinguish",
            "SLC25A38 SEVERITY: most severe non-MLASA form; neonatal/infantile presentation; transfusion-dependent from birth; HSCT should be offered early (before iron organ damage)",
            "ACQUIRED (SF3B1) vs HEREDITARY: SF3B1 somatic mutation (older patient, acquired, clonal) vs germline ALAS2/SLC25A38/etc (younger, family history, constitutional); WES distinguishes",
            "IRON OVERLOAD MANAGEMENT: ALL hereditary SA genes → iron overload (transfusion + ineffective erythropoiesis) → chelation mandatory; monitor ferritin, liver MRI T2*, cardiac MRI T2*",
            "HSCT CURATIVE FOR HAEMATOLOGY (not extra-haematopoietic in PUS1/YARS2/TRNT1): MLASA myopathy/lactic acidosis persists post-HSCT; SIFD developmental delay/hearing loss persists; HSCT fixes anaemia + immunodeficiency only",
        ],
    }


def generate_breakdown():
    result = []
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        result.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "n_patients": len(patients),
            "patients": patients[:5],
        })
    return {"genes": result, "total": len(ATLAS_GENES), "seeds": "2814-2821"}


def generate_definitions():
    return {
        "atlas": "Hereditary Sideroblastic Anemia Atlas",
        "definitions": DEFINITIONS["definitions"],
        "standards": DEFINITIONS["standards"],
        "gene_count": len(ATLAS_GENES),
        "seeds": "2814-2821",
    }
