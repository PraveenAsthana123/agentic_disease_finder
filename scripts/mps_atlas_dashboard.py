#!/usr/bin/env python3
"""MPS-Atlas — Complete 8-Gene Mucopolysaccharidoses Atlas
IDUA · IDS · SGSH · NAGLU · GALNS · ARSB · GUSB · HGSNAT
320-patient aggregate cohort (8 × 40, seeds 912–919)

Mucopolysaccharidoses (MPS) facts:
  - Lysosomal storage disorders caused by deficiency of enzymes degrading glycosaminoglycans (GAGs).
  - GAGs accumulate in lysosomes → progressive multisystem disease.
  - Collective incidence ~1/25,000 live births; MPS I most common in most populations.
  - KEY TEACHING POINTS:
      IDUA (MPS I):   Most common MPS; three subtypes (Hurler severe, Hurler-Scheie intermediate,
                      Scheie mild); ERT (laronidase) does NOT cross BBB; HSCT effective in Hurler if <2 yr.
      IDS (MPS II):   X-linked Hunter syndrome; no corneal clouding (distinguishes from MPS I);
                      ERT (idursulfase) does not cross BBB; intrathecal formulations in trials.
      SGSH (MPS IIIA): Sanfilippo A — most common and most severe Sanfilippo; progressive
                       neuro dementia with behavioral features; NO approved ERT; gene therapy trials.
      NAGLU (MPS IIIB): Sanfilippo B — same phenotype as IIIA; tralesinidase alfa (IGFII-tagged ERT) trials.
      GALNS (MPS IVA): Morquio A — NORMAL intelligence; severe skeletal dysplasia; odontoid
                       hypoplasia → atlantoaxial instability → C-spine MRI MANDATORY before anesthesia.
      ARSB (MPS VI):  Maroteaux-Lamy — NORMAL intelligence (like Morquio A); ERT (galsulfase) effective;
                      somatic disease dominates; HSCT option in severe early-onset.
      GUSB (MPS VII): Sly syndrome — rarest classical MPS; hydrops fetalis in severe neonatal form;
                      ERT (vestronidase alfa) FDA-approved 2017; very variable phenotype.
      HGSNAT (MPS IIIC): Sanfilippo C — membrane-bound acetyltransferase (unique transmembrane enzyme);
                         same Sanfilippo behavioral dementia; no approved treatment; gene therapy in development.

COHORT: 8 × 40 = 320 patient slots (seeds 912–919; gene-specific seeds)
"""

import random

SEED_BASE = 912

# ── All 8 MPS Genes ───────────────────────────────────────────────────────────────
MPS_GENES = [
    # ── IDUA — α-L-iduronidase (MPS I: Hurler/Hurler-Scheie/Scheie) ─────────────
    {
        "gene": "IDUA", "alias": "IDUA — α-L-Iduronidase · MPS I (Hurler / Hurler-Scheie / Scheie)",
        "aa": "653 aa", "kDa": "74 kDa",
        "gene_class": "Lysosomal glycosidase (α-L-iduronidase, GH family 39)",
        "mps_subgroup": "Heparan/dermatan sulfate degradation (IDUA · IDS · SGSH · NAGLU · HGSNAT)",
        "locus": "4p16.3", "omim_gene": 607014,
        "phenotype": (
            "Most common MPS (1/100,000); three severity subtypes: Hurler (severe, MPS IH, OMIM #607015) "
            "— infantile onset, CNS regression, death <10 yr if untreated; Hurler-Scheie (intermediate, MPS IH/S) "
            "— variable CNS, survives to adulthood; Scheie (mild, MPS IS) — normal lifespan, no major CNS"
        ),
        "disease": (
            "IDUA (653aa, 74kDa) encodes α-L-iduronidase, the lysosomal hydrolase responsible for "
            "cleaving the terminal α-L-iduronic acid residues from dermatan sulfate and heparan sulfate. "
            "IDUA loss → dermatan sulfate + heparan sulfate accumulate in lysosomes of virtually all tissues. "
            "Clinical (Hurler, severe): coarse facies (gargoylism), macrocephaly, corneal clouding "
            "(dermatan sulfate in stroma — DISTINGUISHES from MPS II Hunter), hepatosplenomegaly, "
            "dysostosis multiplex (J-shaped sella, oar-shaped ribs, hip dysplasia, claw hand), "
            "cardiomyopathy/valvular disease, obstructive airway, progressive neurological regression "
            "(developmental plateau then decline, communicating hydrocephalus, perivascular GAG deposits). "
            "Urine: elevated dermatan sulfate + heparan sulfate on UAMG/DMB screening. "
            "Leukocyte IDUA enzyme activity <1% normal in Hurler. Genotype–phenotype: "
            "null/null (e.g., Q70X, W402X) → Hurler; missense/missense → milder forms. "
            "W402X is the most common pathogenic allele (50–60% of severe European alleles). "
            "Collective incidence: ~1/100,000; 1/26,000 (Ireland, founder effect). "
            "Newborn screening: urine GAG quantification + leukocyte enzyme + IDUA sequencing."
        ),
        "inheritance": (
            "Autosomal recessive. IDUA 4p16.3. Both severe (Hurler) and mild (Scheie) alleles exist. "
            "Compound heterozygosity common. Genotype broadly predicts severity (null = Hurler; "
            "missense = attenuated) but not perfectly — modifier genes influence CNS involvement."
        ),
        "hallmark": (
            "IDUA MPS I HALLMARKS: "
            "(1) CORNEAL CLOUDING — dermatan sulfate in corneal stroma; absent in MPS II (Hunter); "
            "present in MPS I, IV, VI — use slit lamp to distinguish; "
            "(2) HSCT WINDOW: HSCT (hematopoietic stem cell transplantation) ONLY effective if Hurler, "
            "<2 years of age, and DQ>70 — after 2 yr, neuro outcome poor even post-HSCT; "
            "(3) ERT (LARONIDASE/ALDURAZYME) DOES NOT CROSS BBB — reduces somatic burden "
            "(liver, spleen, airway, urine GAG) but CANNOT treat established CNS disease; "
            "(4) DYSOSTOSIS MULTIPLEX: J-shaped sella turcica, oar-shaped ribs, bullet-nosed phalanges, "
            "hip dysplasia — skeletal X-ray series (babygram) is standard screening; "
            "(5) OBSTRUCTIVE AIRWAY: tracheal narrowing + adenotonsillar hypertrophy → "
            "difficult intubation ALWAYS expected in Hurler patients — anaesthetic alert mandatory; "
            "(6) URINE GAG SPOT TEST: elevated DS+HS → quantitative urine GAG → leukocyte enzyme; "
            "(7) COMMUNICATING HYDROCEPHALUS: GAG in meninges → impaired CSF resorption; "
            "VP shunt may be needed; serial OFC measurement critical in infancy; "
            "(8) VALVULAR DISEASE: mitral/aortic regurgitation from GAG deposits in leaflets — "
            "progressive, requires cardiac surveillance every 1–2 yr"
        ),
        "key_ddx": (
            "IDUA DDx: "
            "(1) MPS II (IDS): X-linked; no corneal clouding; Boys only (rarely girls with skewed lyonization); "
            "confirm by IDS enzyme; "
            "(2) MPS VI (ARSB Maroteaux-Lamy): similar somatic but NORMAL intelligence; "
            "urine DS predominant (not HS); ARSB enzyme low; "
            "(3) MPS IV (GALNS Morquio A): skeletal dysplasia dominant; NORMAL intelligence; "
            "KS+CS6S (keratan sulfate + chondroitin-6-S) in urine; not DS/HS; "
            "(4) GM1 gangliosidosis (GLB1): coarse facies + neuro regression + cherry-red spot; "
            "beta-galactosidase low; no urine DS/HS; "
            "(5) Mucolipidosis II (I-cell disease, GNPTAB): similar to Hurler but urine GAG NORMAL; "
            "plasma lysosomal enzymes very high (misdirected to plasma); "
            "(6) Multiple sulfatase deficiency (SUMF1): multiple enzyme deficiencies + ichthyosis; "
            "confirm by SUMF1 sequencing and multiple sulfatase levels"
        ),
        "diet_treatment": (
            "ERT: Laronidase (Aldurazyme, BioMarin/Genzyme) 0.58 mg/kg IV weekly. "
            "Reduces urine GAG, liver/spleen volume, 6-min walk distance improves. "
            "Does NOT reverse established CNS disease. Pre-medicate with antihistamine/antipyretic. "
            "HSCT (Hurler only, <2 yr, DQ>70): allogeneic HSCT with full myeloablative conditioning; "
            "engraftment provides enzyme via microglial progenitors. Stabilizes CNS in early-treated. "
            "Combined ERT→HSCT protocol: ERT bridges to HSCT (reduces disease burden pre-transplant). "
            "Scheie/attenuated: ERT + symptom management (cardiac, ENT, orthopaedic); "
            "carpal tunnel release, VP shunt for hydrocephalus, valvular surgery as needed."
        ),
        "gene_therapy_status": (
            "Gene therapy: AAV5-IDUA intrathecal (UNC Chapel Hill) and AAV-IDUA IV trials active. "
            "Ex vivo HSC gene therapy (lentiviral IDUA correction) in IND-stage. "
            "Substrate reduction (SRT): not yet approved for MPS I."
        ),
        "critical_ci": (
            "NEVER delay HSCT past 2 years in Hurler (MPS IH): neuro window closes; "
            "laronidase ERT alone does not prevent CNS progression in severe form. "
            "Airway: always alert anaesthesia team — Hurler patients are difficult intubation; "
            "have video laryngoscope, LMA, and ENT backup ready."
        ),
        "nbs_marker": "Urine DS+HS elevation; leukocyte IDUA <1% (Hurler). NBS pilot programs: DBS IDUA enzyme assay.",
        "key_biomarker": "Urine dermatan sulfate + heparan sulfate (quantitative UAMG). Leukocyte IDUA enzyme activity.",
        "severity_spectrum": "Hurler (severe, null/null) → Hurler-Scheie (intermediate) → Scheie (mild, missense/missense)",
        "founder_variant": "W402X (European Hurler, 50–60%); Q70X (European Hurler); A75T (attenuated, common in UK)",
        "key_variants": ["p.W402X (null, Hurler)", "p.Q70X (null, Hurler)", "p.A75T (attenuated)", "p.P533R (attenuated)"],
    },

    # ── IDS — Iduronate-2-sulfatase (MPS II: Hunter syndrome) ─────────────────────
    {
        "gene": "IDS", "alias": "IDS — Iduronate-2-Sulfatase · MPS II (Hunter Syndrome, X-Linked)",
        "aa": "550 aa", "kDa": "62 kDa",
        "gene_class": "Lysosomal sulfatase (iduronate-2-sulfatase, SULF family)",
        "mps_subgroup": "Heparan/dermatan sulfate degradation (IDUA · IDS · SGSH · NAGLU · HGSNAT)",
        "locus": "Xq28", "omim_gene": 300823,
        "phenotype": (
            "X-linked MPS II; affects mostly males (hemizygous); rare carrier females with skewed lyonization; "
            "severe (neuronopathic, MPS IIA) and attenuated (MPS IIB) forms; ERT (idursulfase) available; "
            "hallmark: NO corneal clouding (distinguishes from MPS I/VI)"
        ),
        "disease": (
            "IDS (550aa, 62kDa) encodes iduronate-2-sulfatase, which cleaves 2-O-sulfate groups from "
            "iduronic acid residues in dermatan sulfate and heparan sulfate. IDS deficiency → DS+HS "
            "accumulation. IDS gene at Xq28; pseudogene IDS2 at Xq28 (10 kb proximal) — large "
            "inversions/deletions between IDS and IDS2 cause ~10–15% of severe Hunter cases. "
            "X-linked: virtually all patients male. Rare females: Turner (45,X), extreme skewed "
            "lyonization, or balanced X;autosome translocations. "
            "Clinical: similar to Hurler but lacks corneal clouding (IDS-deficient cornea does not "
            "accumulate sufficient DS for opacity). "
            "Severe (MPS IIA): onset 2–4 yr, progressive neurological decline, intellectual disability, "
            "aggressive behavior, hydrocephalus, hearing loss (mixed conductive + sensorineural). "
            "Pebbly ivory skin lesions over scapulae/upper arms (pathognomonic in some patients). "
            "Hepatosplenomegaly, dysostosis multiplex, cardiomyopathy/valve disease, airway disease. "
            "Attenuated (MPS IIB): normal to near-normal intelligence, somatic disease, long survival. "
            "Incidence: 1/150,000–170,000 males; higher in Korea/Japan (1/34,000). "
            "Urine: DS+HS elevated (similar ratio to MPS I). IDS leukocyte enzyme <1% in hemizygous severe."
        ),
        "inheritance": (
            "X-linked recessive. IDS at Xq28. Hemizygous males affected. "
            "Carrier females usually asymptomatic (enzyme 50% activity). "
            "~10–15% of severe Hunter: IDS–IDS2 inversion (large deletion/inversion). "
            "Sporadic (de novo) in ~30% of cases. Maternal carrier testing: molecular preferred "
            "over enzyme (50% activity overlap with normal females)."
        ),
        "hallmark": (
            "IDS HUNTER HALLMARKS: "
            "(1) NO CORNEAL CLOUDING — dermatan sulfate does NOT accumulate in corneal stroma in MPS II "
            "(IDS enzyme deficiency leaves cornea relatively spared); "
            "CRITICAL distinguishing feature from MPS I (Hurler), MPS VI (Maroteaux-Lamy), MPS IVB (GLB1); "
            "(2) X-LINKED — virtually all males; if a girl appears affected, check X-autosome translocation, "
            "45,X Turner, or extreme skewed lyonization; never assume MPS II in females without genetic proof; "
            "(3) PEBBLY IVORY SKIN LESIONS: peau d'orange textured plaques on scapulae/upper arms — "
            "pathognomonic when present (30–50% of MPS II patients); "
            "(4) ERT (IDURSULFASE/ELAPRASE): 0.5 mg/kg IV weekly; reduces urine GAG, liver/spleen; "
            "DOES NOT cross BBB; CNS disease in severe MPS II NOT addressed by IV ERT; "
            "(5) CNS-PENETRANT FORMULATIONS: idursulfase-IT (intrathecal) + IV in severe MPS IIA trials "
            "(CHAMPIONS trial); brain-penetrant idursulfase beta (Hunterase) — approved in Korea 2012; "
            "(6) HSCT CONTROVERSIAL in MPS II: less clearly beneficial than in Hurler; "
            "some CNS stabilisation reported in attenuated early-treated cases; not standard of care; "
            "(7) BEHAVIOURAL/NEURO FEATURES: severe form — autistic-like behaviours, hyperactivity, "
            "aggression, poor sleep; distinguishes from pure somatic attenuated form; "
            "(8) IDS–IDS2 INVERSIONS: require long-range PCR/MLPA not detected by standard sequencing"
        ),
        "key_ddx": (
            "IDS DDx: "
            "(1) MPS I (IDUA Hurler): corneal clouding (absent in Hunter); autosomal recessive (not X-linked); "
            "(2) MPS VI (ARSB Maroteaux-Lamy): normal intelligence + corneal clouding; ARSB enzyme low; "
            "(3) Sanfilippo (MPS III SGSH/NAGLU/HGSNAT): CNS-predominant, minimal somatic; urine HS only; "
            "(4) MPS IV (GALNS Morquio A): skeletal dysplasia + NORMAL intelligence; keratan sulfate; "
            "(5) Pseudohurler polydystrophy (ML III, GNPTAB): similar somatic, urine GAG NORMAL; "
            "plasma lysosomal enzymes elevated"
        ),
        "diet_treatment": (
            "ERT: Idursulfase (Elaprase, Shire/Takeda) 0.5 mg/kg IV weekly. "
            "Reduces urine DS+HS, liver/spleen volume, improves 6MWT in attenuated. "
            "Intrathecal (IT) ERT: idursulfase-IT via Ommaya or lumbar injection — experimental, "
            "phase 2/3 trials (CHAMPIONS study). Idursulfase beta (Hunterase, GreenCross): "
            "IV, approved Korea 2012, better CNS penetration than standard idursulfase. "
            "Symptomatic: VP shunt (hydrocephalus), ENT (adenotonsillectomy, grommets), "
            "carpal tunnel release, cardiac surveillance, hearing aids."
        ),
        "gene_therapy_status": (
            "Gene therapy: in utero AAV9-IDS (open-label phase 1); "
            "SB-913 (zinc-finger nuclease targeted IDS integration, Sangamo — phase 1/2 disappointing); "
            "lentiviral ex vivo HSC-IDS correction in preclinical/early clinical stage. "
            "SRT: miglustat (off-label, not standard). CNS: intrathecal ERT delivery remains primary strategy."
        ),
        "critical_ci": (
            "NEVER assume a girl with MPS-like features has MPS II without genetic confirmation (X-linked). "
            "Airway alert mandatory (same as MPS I) — difficult intubation in severe cases. "
            "Standard IV idursulfase does NOT treat CNS disease — do not reassure family otherwise."
        ),
        "nbs_marker": "Urine DS+HS elevation; leukocyte IDS <1% (hemizygous males). NBS: DBS IDS enzyme assay.",
        "key_biomarker": "Urine DS + HS (quantitative). Leukocyte IDS enzyme (fluorometric). Plasma heparin cofactor II-thrombin (HCII-T) complex.",
        "severity_spectrum": "MPS IIA (severe neuronopathic) → MPS IIB (attenuated somatic only)",
        "founder_variant": "IDS–IDS2 inversion (~13% of severe, Europe); c.1122C>T p.R468W (attenuated, common)",
        "key_variants": ["IDS–IDS2 large inversion (severe)", "p.R468W (attenuated)", "p.R468Q (attenuated)", "p.A85T"],
    },

    # ── SGSH — Heparan sulfamidase (MPS IIIA: Sanfilippo A) ─────────────────────────
    {
        "gene": "SGSH", "alias": "SGSH — Heparan Sulfamidase · MPS IIIA (Sanfilippo A)",
        "aa": "502 aa", "kDa": "56 kDa",
        "gene_class": "Lysosomal sulfamidase (heparan-N-sulfatase, N-sulfoglucosamine sulfohydrolase)",
        "mps_subgroup": "Heparan sulfate degradation — Sanfilippo group (SGSH · NAGLU · HGSNAT)",
        "locus": "17q25.3", "omim_gene": 605270,
        "phenotype": (
            "Most common and most severe Sanfilippo subtype; CNS-predominant with behavioral regression; "
            "minimal somatic features (mild hepatomegaly, mild coarsening); "
            "onset 2–6 yr progressive behavioral dementia; death usually 2nd decade; NO approved ERT"
        ),
        "disease": (
            "SGSH (502aa, 56kDa) encodes heparan sulfamidase (heparan-N-sulfatase), the first enzyme in "
            "the lysosomal stepwise degradation of heparan sulfate. SGSH cleaves the N-sulfate group "
            "from the non-reducing end glucosamine of heparan sulfate. SGSH deficiency → heparan sulfate "
            "accumulates exclusively in lysosomes → affects primarily brain (neurons). "
            "Clinical: normal or near-normal development to age 2–5 yr, then plateau and regression. "
            "BEHAVIORAL FEATURES PATHOGNOMONIC in Sanfilippo context: "
            "severe hyperactivity, aggression, sleep disturbance (irregular melatonin → absent sleep-wake cycle), "
            "autistic-like behaviors, self-injurious behavior, inconsolable episodes. "
            "This behavioral phase (age 3–7) transitions to severe ID, loss of speech/ambulation, "
            "then vegetative state and death (usually by age 20, some survive to 30s). "
            "Somatic features MINIMAL compared to other MPS: mild hepatomegaly, mild coarse facies, "
            "no corneal clouding, no dysostosis, no severe cardiac. "
            "This makes diagnosis MISSED or delayed — children referred to psychiatry before metabolic. "
            "Urine: heparan sulfate ONLY (not dermatan sulfate — unlike MPS I/II). "
            "Incidence: 1/70,000 (Sanfilippo A most common type in Europe/Australia)."
        ),
        "inheritance": (
            "Autosomal recessive. SGSH 17q25.3. Both parents carriers (~1/130–150 heterozygote frequency). "
            "No sex predilection. High allelic heterogeneity (>200 pathogenic variants). "
            "Genotype–phenotype: some correlation; early-severe variants → classical early-onset; "
            "residual activity variants → attenuated later-onset Sanfilippo A."
        ),
        "hallmark": (
            "SGSH SANFILIPPO A HALLMARKS: "
            "(1) CNS-PREDOMINANT / MINIMAL SOMATIC — Sanfilippo is unique among MPS in primarily affecting "
            "the brain with only mild somatic features; this leads to DIAGNOSTIC DELAY — "
            "average age of diagnosis is 4–5 years; "
            "(2) BEHAVIORAL DEMENTIA TRIAD: hyperactivity + aggression + sleep disturbance — "
            "all three together in a child with mild coarse facies should trigger urine GAG and enzyme testing; "
            "(3) URINE HEPARAN SULFATE ONLY — dermatan sulfate normal (unlike MPS I/II); "
            "urine UAMG/DMB screening may be falsely normal — quantitative HS assay required; "
            "(4) NO CORNEAL CLOUDING, NO SEVERE SKELETAL DISEASE — helps rule out MPS I/VI/IV; "
            "(5) NO APPROVED ERT — IV ERT cannot adequately penetrate CNS; intrathecal/IT ERT trials "
            "ongoing (SGSHex, LYS-SAF302); substrate reduction (SRT/genistein) — not approved, "
            "limited efficacy data; "
            "(6) GENE THERAPY: AAV5-SGSH intrathecal (Amsterdam, Lysosomal Therapeutics): "
            "early-phase trials show stabilisation in young patients (pre-symptomatic or early-stage); "
            "CRITICAL WINDOW: treat as early as possible before significant neuronal loss; "
            "(7) SLEEP MANAGEMENT: melatonin (up to 10 mg) + sedating antihistamines + "
            "behavioural strategies — key quality of life intervention; "
            "(8) ASPIRATON RISK in late disease: swallowing dysfunction → aspiration pneumonia "
            "is major cause of morbidity; PEG consideration as disease progresses"
        ),
        "key_ddx": (
            "SGSH DDx: "
            "(1) NAGLU (MPS IIIB): clinically identical Sanfilippo phenotype; urine HS only; "
            "distinguish by enzyme or gene panel; "
            "(2) HGSNAT (MPS IIIC): same Sanfilippo phenotype; HGSNAT is membrane-bound (unique); "
            "(3) GNS (MPS IIID): rare fourth subtype; same phenotype; N-acetylglucosamine-6-sulfatase; "
            "(4) Autism spectrum disorder: behavioral overlap; check urine HS in any atypical autism "
            "with coarsened features or regression beyond expected; "
            "(5) ADHD/behavioural disorder: hyperactivity first → delayed metabolic workup; "
            "urine HS spot test + quantitative HS should be reflex in unexplained regression"
        ),
        "diet_treatment": (
            "No approved disease-modifying therapy. Management: "
            "Behavioural: melatonin (sleep), clonidine/risperidone (aggression/hyperactivity — use with caution). "
            "Epilepsy: carbamazepine/levetiracetam for seizures (emerge mid-disease). "
            "Aspiration: dietary modification, PEG, chest physio. "
            "Clinical trials: LYS-SAF302 (AAV10-SGSH IT, phase 2/3 MPS IIIA), "
            "UX003-SGSH (intrathecal ERT), genistein SRT (minimal efficacy, not recommended outside trials)."
        ),
        "gene_therapy_status": (
            "ACTIVE gene therapy: AAV5-SGSH (Lysogene LYS-SAF302) intrathecal phase 2/3; "
            "AAV2/5-SGSH (Amsterdam UMC) phase 1/2 — early treated patients show stabilisation. "
            "This is considered the most promising treatment avenue. Early NBS identification critical "
            "to treat in presymptomatic window."
        ),
        "critical_ci": (
            "DO NOT withhold urine HS testing in behavioral children with any dysmorphic feature — "
            "missed Sanfilippo diagnosis is a common preventable tragedy. "
            "Genistein SRT: not recommended outside clinical trial — evidence insufficient. "
            "Melatonin and sleep management are QOL-critical — always address sleep."
        ),
        "nbs_marker": "Urine heparan sulfate only (no DS); leukocyte SGSH enzyme low. No universal NBS yet.",
        "key_biomarker": "Quantitative urine heparan sulfate (not routine UAMG/DMB). Leukocyte SGSH (sulfoamidase) enzyme.",
        "severity_spectrum": "Classical early-onset (most) → attenuated late-onset (rare, residual activity variants)",
        "founder_variant": "p.R245H (Dutch/European founder, attenuated); p.S66W (severe, European); p.R74C (severe)",
        "key_variants": ["p.R245H (attenuated, founder)", "p.S66W (severe)", "p.R74C (severe)", "p.P293R"],
    },

    # ── NAGLU — N-acetyl-α-D-glucosaminidase (MPS IIIB: Sanfilippo B) ─────────────
    {
        "gene": "NAGLU", "alias": "NAGLU — N-acetyl-α-D-Glucosaminidase · MPS IIIB (Sanfilippo B)",
        "aa": "743 aa", "kDa": "83 kDa",
        "gene_class": "Lysosomal glycosidase (N-acetyl-α-D-glucosaminidase, GH family 89)",
        "mps_subgroup": "Heparan sulfate degradation — Sanfilippo group (SGSH · NAGLU · HGSNAT)",
        "locus": "17q21.2", "omim_gene": 609701,
        "phenotype": (
            "MPS IIIB — Sanfilippo B; clinically identical to MPS IIIA (same behavioral dementia progression); "
            "second most common Sanfilippo type (after IIIA); slight later onset on average than IIIA; "
            "tralesinidase alfa (NAGLU+IGFII fusion) ERT in phase 2/3 trials"
        ),
        "disease": (
            "NAGLU (743aa, 83kDa) encodes N-acetyl-α-D-glucosaminidase, the second enzyme in the "
            "lysosomal degradation of heparan sulfate (acts after SGSH). NAGLU cleaves the terminal "
            "N-acetylglucosamine residue from the non-reducing end of heparan sulfate chains. "
            "NAGLU deficiency → heparan sulfate accumulation in lysosomes, predominantly in neurons, "
            "with identical CNS-predominant disease to MPS IIIA. "
            "Clinical: indistinguishable from MPS IIIA by examination — behavioral dementia with "
            "hyperactivity/aggression/sleep disturbance, progressive ID, minimal somatic features. "
            "On average, MPS IIIB may have slightly later onset and slower progression than IIIA, "
            "but this is not a reliable clinical distinction. Distinction requires enzyme assay: "
            "SGSH enzyme normal, NAGLU enzyme low in MPS IIIB. "
            "Urine: heparan sulfate only (same as IIIA). "
            "NAGLU has 743 aa and a unique GH89 family structure — larger than SGSH (502aa). "
            "IGF-II receptor (mannose-6-phosphate receptor) targeting: NAGLU lacks M6P tag naturally, "
            "preventing efficient lysosomal delivery via IV route; tralesinidase alfa adds IGFII domain "
            "to enable receptor-mediated delivery — key innovation in ERT for MPS IIIB. "
            "Incidence: ~1/200,000 (second to IIIA in most populations; IIIB more common in some regions)."
        ),
        "inheritance": (
            "Autosomal recessive. NAGLU 17q21.2. High allelic heterogeneity — >180 reported pathogenic variants. "
            "Genotype–phenotype: partial correlations. Y140C homozygotes — severe. "
            "Some missense variants associated with later-onset attenuated phenotype."
        ),
        "hallmark": (
            "NAGLU SANFILIPPO B HALLMARKS: "
            "(1) CLINICALLY IDENTICAL TO IIIA — cannot be distinguished clinically; "
            "requires enzyme assay (SGSH normal, NAGLU low) or molecular panel; "
            "(2) IGFII-RECEPTOR TRICK (TRALESINIDASE ALFA): NAGLU naturally lacks mannose-6-phosphate "
            "for lysosomal targeting; tralesinidase alfa (BioMarin BMN-250) adds IGF-II domain to "
            "NAGLU → binds M6P/IGFII receptor → receptor-mediated lysosomal delivery; "
            "phase 2 IT trial shows reduction in brain HS; "
            "(3) URINE HS ONLY (same as IIIA): quantitative urine HS required; "
            "(4) INTRATHECAL ROUTE NECESSARY: IV ERT insufficient for CNS; IT tralesinidase in trials; "
            "(5) BEHAVIORAL MANAGEMENT same as IIIA: melatonin, anti-epileptics, PEG in late disease; "
            "(6) NO NBS in most programs yet: diagnosis delayed average 4–6 years; "
            "(7) ATYPICAL FORM: rare attenuated MPS IIIB with mild ID and slower progression — "
            "important to identify early for any future gene therapy window; "
            "(8) SAME GENE THERAPY APPROACH AS IIIA: AAV-NAGLU intrathecal under development"
        ),
        "key_ddx": (
            "NAGLU DDx: "
            "(1) SGSH (MPS IIIA): identical phenotype; SGSH enzyme low, NAGLU normal in IIIA; "
            "(2) HGSNAT (MPS IIIC): same behavioral Sanfilippo phenotype; membrane enzyme; "
            "(3) GNS (MPS IIID): rarest; N-acetylglucosamine-6-sulfatase; same HS urine; "
            "(4) Autism/ADHD: always check urine HS in behavioral regression + mild coarse features"
        ),
        "diet_treatment": (
            "No approved disease-modifying therapy for MPS IIIB currently. "
            "Tralesinidase alfa (BMN-250): NAGLU-IGFII recombinant protein, intrathecal, phase 2/3 trial. "
            "Symptomatic: melatonin (sleep), AEDs (seizures), behavioral medications, PEG/aspiration care. "
            "Gene therapy: AAV9-NAGLU CNS delivery — preclinical and early clinical development."
        ),
        "gene_therapy_status": (
            "Gene therapy: AAV9-NAGLU intrathecal, early-phase clinical studies. "
            "ERT innovation: tralesinidase alfa (BMN-250) IGFII-NAGLU fusion — phase 2/3 IT trial active. "
            "SRT: no approved agent. Research: chaperone therapy for missense variants in development."
        ),
        "critical_ci": (
            "Do not confuse with MPS IIIA — same phenotype but different enzyme (NAGLU not SGSH). "
            "Requires full Sanfilippo enzyme panel (SGSH + NAGLU + HGSNAT + GNS) to type correctly. "
            "Tralesinidase alfa must be delivered intrathecally, not IV — "
            "IV route is insufficient for CNS penetration."
        ),
        "nbs_marker": "Urine heparan sulfate only. Leukocyte NAGLU enzyme low. No universal NBS program yet.",
        "key_biomarker": "Quantitative urine heparan sulfate. Leukocyte NAGLU enzyme (fluorometric with MU-GlcNAc substrate).",
        "severity_spectrum": "Classical severe (most) → attenuated mild ID (rare residual activity variants)",
        "founder_variant": "p.Y140C (severe, European); p.R297X (severe); p.R626X (severe, common in multiple populations)",
        "key_variants": ["p.Y140C (severe)", "p.R297X (severe)", "p.R626X (severe)", "p.E153K (attenuated)"],
    },

    # ── GALNS — Galactosamine-6-sulfatase (MPS IVA: Morquio A) ──────────────────────
    {
        "gene": "GALNS", "alias": "GALNS — Galactosamine-6-Sulfatase · MPS IVA (Morquio A)",
        "aa": "522 aa", "kDa": "59 kDa",
        "gene_class": "Lysosomal sulfatase (galactose-6-sulfatase, N-acetylgalactosamine-6-sulfatase)",
        "mps_subgroup": "Keratan sulfate / chondroitin-6-sulfate degradation (GALNS · GLB1-IVB)",
        "locus": "16q24.3", "omim_gene": 612222,
        "phenotype": (
            "MPS IVA — Morquio A; NORMAL intelligence (unlike Hurler/Hunter/Sanfilippo); "
            "severe skeletal dysplasia: odontoid hypoplasia + atlantoaxial instability = C-SPINE EMERGENCY; "
            "ERT: elosulfase alfa (Vimizim); incidence ~1/200,000–250,000"
        ),
        "disease": (
            "GALNS (522aa, 59kDa) encodes galactosamine-6-sulfatase (also called N-acetylgalactosamine-6-sulfatase), "
            "which cleaves 6-O-sulfate groups from galactose and N-acetylgalactosamine residues in "
            "keratan sulfate (KS) and chondroitin-6-sulfate (CS6S). "
            "GALNS deficiency → KS + CS6S accumulation, primarily in cartilage, bone, cornea, and heart. "
            "NEURONS are largely SPARED because KS in brain is different (non-sulfated) → intelligence NORMAL. "
            "Clinical: severe spondyloepiphyseal dysplasia — progressive from age 1–3 yr: "
            "short trunk short stature (most severe spondyloepiphyseal dysplasia), "
            "pectus carinatum/excavatum, genu valgum (knock-knees), wrist enlargement, "
            "hip dysplasia, ligamentous laxity. "
            "ODONTOID HYPOPLASIA → ATLANTOAXIAL INSTABILITY (AAI): "
            "underdeveloped odontoid process + lax transverse ligament → atlantoaxial subluxation "
            "at C1-C2 → cervical myelopathy / sudden death with minor trauma. "
            "CRITICAL: cervical MRI/CT BEFORE general anaesthesia or surgery. "
            "Corneal clouding (KS deposit) — present in most. "
            "Cardiac: aortic regurgitation, mitral valve disease. "
            "Hearing: mixed conductive + sensorineural loss. "
            "Urine: keratan sulfate + CS6S elevated. "
            "Diagnosis: urine KS quantification + leukocyte GALNS enzyme + GALNS sequencing."
        ),
        "inheritance": (
            "Autosomal recessive. GALNS 16q24.3. High allelic heterogeneity (>200 pathogenic variants). "
            "Genotype–phenotype: some correlation. p.M318R severe (null function). "
            "Residual activity missense → attenuated MPS IVA with longer ambulatory lifespan."
        ),
        "hallmark": (
            "GALNS MORQUIO A HALLMARKS: "
            "(1) NORMAL INTELLIGENCE — cardinal feature distinguishing Morquio A from Hurler/Hunter/Sanfilippo; "
            "these patients are cognitively intact and socially aware of their disease; "
            "(2) ATLANTOAXIAL INSTABILITY (AAI) — LIFE-THREATENING: "
            "odontoid hypoplasia + ligamentous laxity → C1-C2 subluxation → cervical cord compression; "
            "MANDATORY: cervical spine flexion-extension MRI/CT BEFORE ANY general anaesthesia; "
            "cervical surgical fusion (C1-C2 atlantoaxial fusion) in symptomatic or high-risk AAI; "
            "sudden death or quadriplegia from minor trauma if untreated; "
            "(3) URINE KERATAN SULFATE (KS) — not DS/HS; different from MPS I/II/III; "
            "KS decreases with age (problematic — may normalise in teens/adults); "
            "urine GALNS-specific oligosaccharides remain abnormal; "
            "(4) ERT: ELOSULFASE ALFA (VIMIZIM, BioMarin) 2 mg/kg IV weekly: "
            "improves 6MWT, respiratory endurance, reduces urine KS; "
            "does NOT correct skeletal deformity already established; "
            "(5) CORNEAL CLOUDING: present in most; different mechanism from MPS I (KS not DS); "
            "(6) NO HEPATOSPLENOMEGALY typically (cartilage/bone primary storage); "
            "(7) SHORT-TRUNK SHORT STATURE: characteristic — rib, vertebral, pelvic changes; "
            "(8) HIP/KNEE SURGICAL MANAGEMENT: progressive valgus deformity → orthopaedic interventions"
        ),
        "key_ddx": (
            "GALNS DDx: "
            "(1) GLB1 MPS IVB (β-galactosidase): Morquio B — similar skeletal to Morquio A but caused by GLB1; "
            "β-galactosidase low; GM1 gangliosidosis same gene (different alleles); "
            "(2) Spondyloepiphyseal dysplasia congenita (COL2A1): skeletal dysplasia, normal urine GAG; "
            "(3) Kniest dysplasia (COL2A1): severe SED, normal urine GAG; "
            "(4) Multiple epiphyseal dysplasia: less severe, normal urine GAG; "
            "(5) Spondylometaphyseal dysplasia: normal urine GAG; "
            "Urine keratan sulfate + leukocyte GALNS enzyme distinguishes Morquio A definitively"
        ),
        "diet_treatment": (
            "ERT: Elosulfase alfa (Vimizim, BioMarin) 2 mg/kg IV weekly (every 7 days). "
            "Reduces urine KS, improves 6MWT. Start as early as possible. "
            "Pre-medicate with antihistamine/antipyretic (high infusion reaction rate ~30%). "
            "Surgery: C1-C2 atlantoaxial fusion for symptomatic AAI (or high-risk on imaging); "
            "hip/knee orthopaedics; hearing aids; corneal transplant (rarely needed). "
            "No approved gene therapy yet. SRT: not applicable (KS is extracellular)."
        ),
        "gene_therapy_status": (
            "Gene therapy: AAV9-GALNS systemic — preclinical studies showing skeletal correction in mouse. "
            "Phase 1 clinical: not yet open. Substrate reduction: not applicable for KS. "
            "ERT remains standard of care."
        ),
        "critical_ci": (
            "NEVER administer general anaesthesia to a Morquio A patient without first obtaining "
            "flexion-extension cervical spine MRI — AAI with minor manipulation can cause quadriplegia or death. "
            "Alert surgeons, orthopaedists, and anaesthetists about AAI risk in every patient."
        ),
        "nbs_marker": "Urine keratan sulfate (decreases with age — may miss in adults). Leukocyte GALNS enzyme. DBS NBS programs being piloted.",
        "key_biomarker": "Urine keratan sulfate (KS); leukocyte GALNS enzyme; serum KS oligosaccharides (more stable with age).",
        "severity_spectrum": "Severe classical → attenuated (residual activity; later ambulatory loss)",
        "founder_variant": "p.M318R (severe, multiple populations); p.G301C (Middle Eastern); p.R386C (attenuated)",
        "key_variants": ["p.M318R (severe)", "p.G301C (Middle East)", "p.R386C (attenuated)", "p.L271P (severe)"],
    },

    # ── ARSB — Arylsulfatase B (MPS VI: Maroteaux-Lamy) ─────────────────────────────
    {
        "gene": "ARSB", "alias": "ARSB — Arylsulfatase B · MPS VI (Maroteaux-Lamy Syndrome)",
        "aa": "533 aa", "kDa": "61 kDa",
        "gene_class": "Lysosomal sulfatase (N-acetylgalactosamine-4-sulfatase, arylsulfatase B)",
        "mps_subgroup": "Dermatan sulfate degradation (ARSB · IDUA partial overlap)",
        "locus": "5q14.1", "omim_gene": 611542,
        "phenotype": (
            "MPS VI — Maroteaux-Lamy; NORMAL intelligence (distinguishes from MPS I/II); "
            "somatic disease: coarse facies, corneal clouding, dysostosis, hepatosplenomegaly, cardiac; "
            "ERT: galsulfase (Naglazyme) effective; urine dermatan sulfate ONLY (not HS)"
        ),
        "disease": (
            "ARSB (533aa, 61kDa) encodes arylsulfatase B (N-acetylgalactosamine-4-sulfatase), which "
            "cleaves 4-O-sulfate from N-acetylgalactosamine residues in dermatan sulfate. "
            "ARSB deficiency → dermatan sulfate accumulation only (NOT heparan sulfate — "
            "this explains why intelligence is NORMAL: neurons use HS not DS for lysosomal traffic; "
            "DS-laden cells are somatic — fibroblasts, Kupffer cells, chondrocytes, endothelium). "
            "Clinical: resembles Hurler (coarse facies, corneal clouding, hepatosplenomegaly, "
            "dysostosis multiplex) BUT INTELLIGENCE IS NORMAL to near-normal. "
            "Cardiac: mitral/aortic regurgitation, progressive cardiomyopathy — major cause of mortality. "
            "Airway: similar obstructive issues as Hurler — difficult intubation expected. "
            "Spinal cord compression: at thoracolumbar junction from GAG deposition and kyphosis. "
            "Corneal clouding: dermatan sulfate in stroma (present, as in MPS I). "
            "Urine: dermatan sulfate predominant (HS minimal/absent — unlike MPS I/II). "
            "Severity spectrum: severe (infantile onset, wheelchair by 20) to mild (adult-onset symptoms). "
            "Incidence: ~1/250,000–300,000."
        ),
        "inheritance": (
            "Autosomal recessive. ARSB 5q14.1. >130 reported pathogenic variants. "
            "Genotype–phenotype: severe null/null → classical; missense with residual activity → attenuated. "
            "ARSB complete loss → <1% enzyme; attenuated → 2–10% residual."
        ),
        "hallmark": (
            "ARSB MAROTEAUX-LAMY HALLMARKS: "
            "(1) NORMAL INTELLIGENCE — critical: Maroteaux-Lamy patients have NORMAL cognition; "
            "they understand their disease, participate in treatment decisions, hold jobs (attenuated form); "
            "confusion with Hurler (ID) is a clinical error; "
            "(2) CORNEAL CLOUDING PRESENT — this distinguishes from Hunter MPS II (no clouding); "
            "both Hurler (MPS I) and Maroteaux-Lamy (MPS VI) have corneal clouding + somatic disease; "
            "differentiate by intelligence (Hurler: impaired; Maroteaux-Lamy: NORMAL); "
            "(3) URINE DS ONLY (NOT HS) — DS predominant in urine; heparan sulfate normal or minimal; "
            "this reflects tissue selectivity (no neuronal HS accumulation); "
            "(4) ERT: GALSULFASE (NAGLAZYME, BioMarin) 1 mg/kg IV weekly: "
            "reduces urine DS, improves endurance (6MWT), respiratory function; "
            "ERT does NOT cross BBB (but CNS is spared in MPS VI); "
            "(5) CARDIAC: major disease burden — mitral + aortic valve disease, "
            "progressive cardiomyopathy; annual echo + ECG mandatory; "
            "surgical valve replacement in severe cases; "
            "(6) HSCT: considered in severe early-onset MPS VI (some centres); "
            "beneficial evidence weaker than MPS I Hurler; "
            "(7) AIRWAY ALERT: same obstructive issues as Hurler; difficult intubation; "
            "(8) SPINAL CORD: monitor for myelopathy (thoracolumbar kyphosis + cord compression)"
        ),
        "key_ddx": (
            "ARSB DDx: "
            "(1) IDUA MPS I Hurler: similar somatic + corneal clouding; "
            "BUT Hurler has intellectual disability; urine both DS+HS; IDUA enzyme low; "
            "(2) IDS MPS II Hunter: NO corneal clouding; X-linked; urine DS+HS; "
            "(3) GALNS MPS IV Morquio A: NORMAL intelligence + skeletal dysplasia but "
            "urine keratan sulfate (not DS); GALNS enzyme low; "
            "(4) GLB1 MPS IVB: rare skeletal Morquio-like; GM1 gangliosidosis alleles; "
            "(5) Multiple sulfatase deficiency (SUMF1): ARSB + multiple other sulfatases low; "
            "ichthyosis + neuro involvement; SUMF1 mutation"
        ),
        "diet_treatment": (
            "ERT: Galsulfase (Naglazyme, BioMarin) 1 mg/kg IV weekly. "
            "Reduces urine DS, liver/spleen, improves 6MWT and pulmonary function. "
            "Pre-medicate with antihistamine/antipyretic. "
            "HSCT: considered in severe early-onset before age 5 in some centres. "
            "Symptomatic: cardiac surveillance (annual echo), valve surgery, "
            "VP shunt (hydrocephalus — less common than MPS I), hearing aids, corneal transplant."
        ),
        "gene_therapy_status": (
            "Gene therapy: AAV9-ARSB systemic and intrathecal — preclinical. "
            "Phase 1 trials not yet open. ERT remains standard of care. "
            "CRISPR/base editing approaches in early research phase."
        ),
        "critical_ci": (
            "NEVER mistake MPS VI for MPS I Hurler — they are different diseases with different treatment "
            "(HSCT mandatory in Hurler <2yr; not so in MPS VI). Intelligence is NORMAL in MPS VI. "
            "Airway: difficult intubation alert mandatory — alert anaesthesia before any procedure."
        ),
        "nbs_marker": "Urine dermatan sulfate elevation; leukocyte ARSB <1%. NBS DBS ARSB enzyme assay piloted.",
        "key_biomarker": "Urine dermatan sulfate (quantitative DS; HS minimal). Leukocyte ARSB enzyme (4-MU-GalNAc4S substrate).",
        "severity_spectrum": "Severe classical (infantile onset, rapidly progressive) → attenuated (adult onset, mild somatic)",
        "founder_variant": "p.R152W (attenuated, Irish/European); p.H178L (severe); c.1533+1G>A splice (severe, various)",
        "key_variants": ["p.R152W (attenuated)", "p.H178L (severe)", "p.L269F (severe)", "c.1533+1G>A (splice, severe)"],
    },

    # ── GUSB — β-Glucuronidase (MPS VII: Sly Syndrome) ──────────────────────────────
    {
        "gene": "GUSB", "alias": "GUSB — β-Glucuronidase · MPS VII (Sly Syndrome)",
        "aa": "651 aa", "kDa": "74 kDa",
        "gene_class": "Lysosomal glycosidase (β-D-glucuronidase, GH family 79)",
        "mps_subgroup": "Multi-GAG degradation (DS + HS + CS) — GUSB",
        "locus": "7q11.21", "omim_gene": 611638,
        "phenotype": (
            "MPS VII — Sly syndrome; rarest classical MPS; most severe form = non-immune hydrops fetalis; "
            "HIGHLY variable phenotype (lethal neonatal hydrops → mild adult-onset); "
            "ERT: vestronidase alfa (Mepsevii, FDA 2017, first ERT for MPS VII); urine DS+HS+CS all elevated"
        ),
        "disease": (
            "GUSB (651aa, 74kDa) encodes β-D-glucuronidase, a lysosomal enzyme that cleaves terminal "
            "β-glucuronic acid from the non-reducing end of dermatan sulfate, heparan sulfate, "
            "AND chondroitin sulfate. GUSB deficiency → accumulation of DS + HS + CS (all three GAGs) "
            "in lysosomes — making urine biochemistry uniquely triply elevated. "
            "MPS VII is the RAREST MPS — incidence ~1/1,000,000–2,000,000. "
            "EXTREME PHENOTYPIC VARIABILITY: "
            "Severe neonatal: non-immune hydrops fetalis (NIHF) — GAG accumulation in utero; "
            "fetal ascites, pleural effusions, subcutaneous oedema, placentomegaly. "
            "Many NIHF cases from MPS VII are stillborn or die neonatally. "
            "If survive: Hurler-like course with intellectual disability, hepatosplenomegaly, "
            "dysostosis multiplex, corneal clouding, cardiac disease. "
            "Attenuated: childhood/adult onset — mild coarse facies, mild bone disease, "
            "slowly progressive; relatively preserved intellect. "
            "UNIQUE: MPS VII can recur as NIHF in successive pregnancies → MPS VII workup "
            "is MANDATORY in any unexplained non-immune hydrops fetalis. "
            "Diagnosis: triple-elevated urine GAG (DS+HS+CS) + leukocyte GUSB <1% + GUSB sequencing."
        ),
        "inheritance": (
            "Autosomal recessive. GUSB 7q11.21. Very rare — only ~200 reported cases. "
            "p.L176F hotspot in many non-European cases. Null/null → severe neonatal/Hurler-like. "
            "Missense with residual → attenuated."
        ),
        "hallmark": (
            "GUSB SLY SYNDROME HALLMARKS: "
            "(1) NON-IMMUNE HYDROPS FETALIS (NIHF) — SEVERE NEONATAL FORM: "
            "MPS VII is one of the TOP metabolic causes of non-immune hydrops fetalis; "
            "any unexplained NIHF must have MPS VII workup (enzyme + urine GAG on cord blood/neonatal sample); "
            "(2) TRIPLE GAG ELEVATION (DS + HS + CS) — uniquely trivalent urine GAG accumulation; "
            "this distinguishes MPS VII from all other MPS types on urine biochemistry; "
            "(3) ERT: VESTRONIDASE ALFA (MEPSEVII, Ultragenyx) — FIRST and only FDA-approved ERT for MPS VII "
            "(August 2017); 4 mg/kg IV every 2 weeks; "
            "reduces urine GAG, hepatosplenomegaly; "
            "(4) IN UTERO ERT: vestronidase alfa given in utero (intravascular/intraperitoneal "
            "to fetus) → first successful in-utero ERT for MPS VII; reduces NIHF severity; "
            "proof-of-concept for prenatal metabolic disease treatment; "
            "(5) EXTREME PHENOTYPIC VARIABILITY: same disease spans lethal NIHF → mild adult disease; "
            "(6) CORNEAL CLOUDING + SOMATIC DISEASE: Hurler-like in severe; "
            "(7) RECURRENT NIHF IN FAMILY: if one fetus had NIHF from MPS VII, "
            "subsequent pregnancies carry 25% risk — offer prenatal diagnosis; "
            "(8) SMALL CASE SERIES ONLY: ~200 cases worldwide — limited natural history data"
        ),
        "key_ddx": (
            "GUSB DDx: "
            "(1) Other causes of NIHF: cardiac (most common), chromosomal, haematological (Rh), "
            "twin-to-twin transfusion, alpha-thalassaemia; GUSB enzyme must be part of NIHF metabolic screen; "
            "(2) MPS I Hurler: urine DS+HS (not CS); corneal clouding; IDUA low; "
            "(3) MPS II Hunter: urine DS+HS; X-linked; no corneal clouding; "
            "(4) MPS VI Maroteaux-Lamy: urine DS only; normal intelligence; ARSB low; "
            "(5) Gaucher (GBA): NIHF possible in type 2 perinatal; ceramide not GAG; "
            "glucocerebrosidase low; "
            "(6) Sialidosis/Galactosialidosis: hydrops + cherry red spot; NEU1/CTSA; "
            "urine sialooligosaccharides"
        ),
        "diet_treatment": (
            "ERT: Vestronidase alfa (Mepsevii, Ultragenyx) 4 mg/kg IV every 2 weeks. "
            "Reduces urine GAG, hepatosplenomegaly, improves endurance. "
            "In utero ERT: for severe fetal NIHF form — intravascular fetal administration. "
            "Symptomatic: same multisystem management as Hurler (cardiac, ENT, orthopaedic, VP shunt). "
            "Prenatal diagnosis: CVS or amniocentesis + GUSB enzyme + sequencing in known carrier families."
        ),
        "gene_therapy_status": (
            "Gene therapy: AAV-GUSB preclinical (well-studied in the canine MPS VII model — "
            "the first lysosomal disease AAV gene therapy model). Phase 1 clinical not yet open. "
            "Vestronidase alfa remains the only approved therapy."
        ),
        "critical_ci": (
            "ALWAYS include MPS VII (GUSB) in the differential of non-immune hydrops fetalis — "
            "missed diagnosis = recurrent NIHF in subsequent pregnancies without carrier detection. "
            "In utero ERT for NIHF: requires highly specialised fetal medicine centre. "
            "Vestronidase alfa dosing: 4 mg/kg every 2 weeks (different interval from weekly ERT of other MPS)."
        ),
        "nbs_marker": "Urine DS+HS+CS all elevated (triple). Leukocyte GUSB <1%. Cord blood GUSB enzyme in NIHF workup.",
        "key_biomarker": "Urine DS + HS + CS (triple elevation — pathognomonic for MPS VII vs other MPS). Leukocyte GUSB enzyme.",
        "severity_spectrum": "Neonatal lethal (NIHF, null/null) → Hurler-like (severe infantile) → attenuated adult-onset",
        "founder_variant": "p.L176F (many ethnic groups); very few founder effects given extreme rarity",
        "key_variants": ["p.L176F (common in non-European cases)", "p.A619V (attenuated)", "p.G68E (severe)"],
    },

    # ── HGSNAT — Heparan-α-glucosaminide N-acetyltransferase (MPS IIIC: Sanfilippo C) ──
    {
        "gene": "HGSNAT", "alias": "HGSNAT — Heparan-α-glucosaminide N-acetyltransferase · MPS IIIC (Sanfilippo C)",
        "aa": "635 aa", "kDa": "73 kDa",
        "gene_class": "Lysosomal membrane acetyltransferase (transmembrane enzyme — unique; not a soluble hydrolase)",
        "mps_subgroup": "Heparan sulfate degradation — Sanfilippo group (SGSH · NAGLU · HGSNAT)",
        "locus": "8p11.21", "omim_gene": 610453,
        "phenotype": (
            "MPS IIIC — Sanfilippo C; same phenotype as IIIA/IIIB (behavioral dementia, minimal somatic); "
            "unique: HGSNAT is the ONLY membrane-bound acetyltransferase in the MPS pathway "
            "(not a soluble hydrolase); harder to assay than other Sanfilippo types; "
            "no approved ERT; gene therapy development"
        ),
        "disease": (
            "HGSNAT (635aa, 73kDa) encodes heparan-α-glucosaminide N-acetyltransferase, a lysosomal "
            "membrane-spanning enzyme that acetylates the free amino group on glucosamine residues "
            "before the next degradation step (NAGLU cleavage). "
            "HGSNAT is UNIQUE: it is a transmembrane acetyltransferase (not a soluble lysosomal hydrolase "
            "like SGSH, NAGLU, or GALNS). It requires acetyl-CoA from cytoplasm (via a membrane channel) "
            "and cannot be replaced by IV ERT (cannot be delivered as a recombinant secreted enzyme "
            "that gets taken up by mannose-6-phosphate receptor — transmembrane topology prevents this). "
            "This makes ERT inherently impractical for MPS IIIC, unlike most other MPS. "
            "Clinical: identical to MPS IIIA/B — progressive behavioral dementia. "
            "Onset age 2–6 yr: behavioral regression (hyperactivity, aggression, sleep disturbance), "
            "intellectual decline, then vegetative state and death in 2nd–3rd decade. "
            "Minimal somatic features (mild hepatomegaly, mild facies). "
            "Urine: heparan sulfate only (same as IIIA/IIIB). "
            "Enzyme assay: HGSNAT harder to assay (transmembrane enzyme — requires detergent, "
            "membrane preparation; acetylCoA substrate); less routinely available than SGSH/NAGLU. "
            "Incidence: ~1/1,500,000 (rarest Sanfilippo subtype, though regional variability exists). "
            "MPS IIIC has been under-diagnosed historically due to enzymatic assay difficulty — "
            "molecular panel now preferred for Sanfilippo typing."
        ),
        "inheritance": (
            "Autosomal recessive. HGSNAT 8p11.21. ~100 pathogenic variants reported. "
            "Variable severity. Some attenuated cases with residual acetyltransferase activity. "
            "p.R344C associated with attenuated / adult-onset presentation."
        ),
        "hallmark": (
            "HGSNAT SANFILIPPO C HALLMARKS: "
            "(1) MEMBRANE-BOUND ENZYME — UNIQUE IN MPS: HGSNAT is the only transmembrane lysosomal "
            "enzyme in the MPS degradation pathway; "
            "this fundamentally changes therapeutic strategy: "
            "standard IV ERT (secreted enzyme → M6P receptor uptake) DOES NOT WORK for HGSNAT "
            "because it cannot traffic to the correct membrane location; "
            "(2) CLINICALLY IDENTICAL TO MPS IIIA/IIIB: same behavioral dementia triad; "
            "cannot distinguish IIIA/B/C clinically — must use enzyme panel or molecular sequencing; "
            "(3) ENZYME ASSAY TECHNICALLY CHALLENGING: HGSNAT requires acetyl-CoA donor, "
            "detergent-solubilised membrane preparation; less available in standard metabolic labs — "
            "many centres go directly to genetic panel for Sanfilippo typing; "
            "(4) URINE HS ONLY — same as IIIA/IIIB (heparan sulfate; no dermatan sulfate); "
            "(5) GENE THERAPY IS THE ONLY VIABLE DISEASE-MODIFYING STRATEGY: "
            "AAV-HGSNAT needs to deliver the membrane enzyme gene intracellularly; "
            "intrathecal/intracranial AAV delivery in preclinical; "
            "(6) LATER ONSET on average than MPS IIIA: average age at diagnosis 4–7 yr; "
            "attenuated cases with p.R344C may present in adolescence/adulthood; "
            "(7) SLEEP/BEHAVIOUR management identical to IIIA/IIIB; "
            "(8) ATTENUATED FORM (MPS IIIC): p.R344C, p.P304L → later onset, slower course, "
            "some adults maintain functional life; "
        ),
        "key_ddx": (
            "HGSNAT DDx: "
            "(1) SGSH (MPS IIIA): clinically identical; SGSH enzyme low, HGSNAT normal; "
            "(2) NAGLU (MPS IIIB): same phenotype; NAGLU enzyme low, HGSNAT normal; "
            "(3) GNS (MPS IIID): fourth subtype; same HS urine; N-acetylglucosamine-6-sulfatase; "
            "(4) Autism/ADHD: behavioral overlap — check urine HS and full Sanfilippo enzyme panel; "
            "(5) Late-onset neurodegenerative conditions in attenuated IIIC: "
            "dementia workup should include MPS IIIC enzyme if unexplained adult-onset behavioral dementia"
        ),
        "diet_treatment": (
            "No approved disease-modifying therapy. ERT is NOT possible (transmembrane enzyme). "
            "Management: behavioural medications, melatonin (sleep), AEDs (seizures), PEG/aspiration care. "
            "Gene therapy (investigational): AAV9-HGSNAT intrathecal — preclinical stage. "
            "SRT: genistein (not recommended outside trial). "
            "Clinical trial watch: SanFilippo Research Foundation lists open trials. "
            "Consider natural history study enrollment for all MPS IIIC patients."
        ),
        "gene_therapy_status": (
            "Gene therapy: AAV9/PHP.B-HGSNAT intrathecal — mouse model correction achieved; "
            "IND application stage. No approved phase 1 clinical trial open as of 2026. "
            "Due to transmembrane nature, AAV must deliver full-length HGSNAT coding sequence "
            "for membrane integration — feasible with AAV9 (635aa fits within AAV payload). "
            "SRT and ERT both inherently inapplicable for MPS IIIC."
        ),
        "critical_ci": (
            "ERT (standard IV recombinant enzyme) is NOT applicable for MPS IIIC — "
            "do not plan IV ERT trials targeting HGSNAT; "
            "transmembrane topology prevents M6P-mediated delivery. "
            "Enzyme assay requires specialist lab — if assay not available, go directly to HGSNAT sequencing. "
            "Diagnosis delay is common — always include HGSNAT panel in Sanfilippo enzyme workup."
        ),
        "nbs_marker": "Urine heparan sulfate only. HGSNAT enzyme (membrane prep + acetyl-CoA assay) in specialist lab.",
        "key_biomarker": "Quantitative urine heparan sulfate. HGSNAT enzyme (acetyltransferase assay — technically demanding).",
        "severity_spectrum": "Classical severe (most) → attenuated adult-onset (p.R344C, p.P304L, rare)",
        "founder_variant": "p.R344C (attenuated, European); p.L18R (severe, various); no single dominant founder",
        "key_variants": ["p.R344C (attenuated, common)", "p.L18R (severe)", "p.P304L (attenuated)", "c.372-2A>G (splice, severe)"],
    },
]


def _simulate(gene_data, n=40):
    """Generate synthetic patient records for one MPS gene (seed = SEED_BASE + gene index)."""
    gene = gene_data["gene"]
    idx = next(i for i, g in enumerate(MPS_GENES) if g["gene"] == gene)
    rng = random.Random(SEED_BASE + idx)

    # Disease-specific age-at-diagnosis ranges (months)
    age_ranges = {
        "IDUA":   (6, 36),    # Hurler detected early; attenuated later
        "IDS":    (12, 48),   # Hunter onset 2–4 yr
        "SGSH":   (24, 60),   # Sanfilippo A 2–5 yr
        "NAGLU":  (24, 72),   # Sanfilippo B slightly later
        "GALNS":  (12, 48),   # Morquio A skeletal onset 1–4 yr
        "ARSB":   (12, 60),   # Maroteaux-Lamy
        "GUSB":   (0, 24),    # Sly: neonatal NIHF or early infantile
        "HGSNAT": (36, 84),   # Sanfilippo C tends later
    }

    # ERT available/approved (true = standard of care ERT exists)
    ert_available = {
        "IDUA":   True,   # laronidase
        "IDS":    True,   # idursulfase
        "SGSH":   False,  # no approved ERT
        "NAGLU":  False,  # tralesinidase alfa in trials only
        "GALNS":  True,   # elosulfase alfa
        "ARSB":   True,   # galsulfase
        "GUSB":   True,   # vestronidase alfa
        "HGSNAT": False,  # ERT inherently inapplicable
    }

    # Normal intelligence rates
    normal_iq_rate = {
        "IDUA":   0.15,   # Hurler ~0%; Scheie ~100%; mixed cohort
        "IDS":    0.40,   # attenuated form: normal; severe: impaired
        "SGSH":   0.00,   # all cognitive impairment in classical
        "NAGLU":  0.00,
        "GALNS":  0.95,   # Morquio A: almost always normal IQ
        "ARSB":   0.85,   # Maroteaux-Lamy: mostly normal IQ
        "GUSB":   0.25,   # severe form: impaired; attenuated: near-normal
        "HGSNAT": 0.00,   # Sanfilippo C: progressive dementia
    }

    # Corneal clouding rates
    corneal_rate = {
        "IDUA":   0.90,   # Hurler almost always; Scheie moderate
        "IDS":    0.05,   # Hunter: minimal/absent (KEY DISTINGUISHING FEATURE)
        "SGSH":   0.10,   # Sanfilippo: usually absent
        "NAGLU":  0.10,
        "GALNS":  0.75,   # Morquio A: present
        "ARSB":   0.85,   # Maroteaux-Lamy: present
        "GUSB":   0.60,   # Sly: variable
        "HGSNAT": 0.10,   # Sanfilippo C: usually absent
    }

    # Hepatosplenomegaly rates
    hepsplen_rate = {
        "IDUA":   0.95,
        "IDS":    0.90,
        "SGSH":   0.35,   # mild in Sanfilippo
        "NAGLU":  0.35,
        "GALNS":  0.25,   # Morquio A: not a primary feature
        "ARSB":   0.80,
        "GUSB":   0.80,
        "HGSNAT": 0.35,
    }

    # Atlantoaxial instability rates (Morquio A high)
    aai_rate = {
        "IDUA":   0.10,
        "IDS":    0.10,
        "SGSH":   0.00,
        "NAGLU":  0.00,
        "GALNS":  0.70,   # Morquio A: major feature
        "ARSB":   0.10,
        "GUSB":   0.15,
        "HGSNAT": 0.00,
    }

    # Hydrops fetalis rates (Sly syndrome/NIHF prominent)
    hydrops_rate = {
        "IDUA":   0.02,
        "IDS":    0.00,
        "SGSH":   0.00,
        "NAGLU":  0.00,
        "GALNS":  0.00,
        "ARSB":   0.02,
        "GUSB":   0.30,   # Sly syndrome: ~30% present with NIHF
        "HGSNAT": 0.00,
    }

    lo, hi = age_ranges.get(gene, (12, 60))
    patients = []
    for i in range(n):
        age_dx_mo = round(rng.uniform(lo, hi), 1)
        age_dx_y = round(age_dx_mo / 12, 2)
        ert = ert_available[gene]
        normal_iq = rng.random() < normal_iq_rate[gene]
        corneal = rng.random() < corneal_rate[gene]
        hepsplen = rng.random() < hepsplen_rate[gene]
        aai = rng.random() < aai_rate[gene]
        hydrops = rng.random() < hydrops_rate[gene]
        hsct_eligible = (
            gene == "IDUA" and age_dx_mo < 24 and not normal_iq and rng.random() < 0.60
        )
        patients.append({
            "patient_id": f"MPS-{gene}-{i+1:03d}",
            "gene": gene,
            "age_dx_months": age_dx_mo,
            "age_dx_y": age_dx_y,
            "ert_available": ert,
            "normal_iq": normal_iq,
            "corneal_clouding": corneal,
            "hepatosplenomegaly": hepsplen,
            "atlantoaxial_instability": aai,
            "hydrops_fetalis": hydrops,
            "hsct_eligible": hsct_eligible,
        })
    return patients


def _all_patients():
    result = []
    for g in MPS_GENES:
        result.extend(_simulate(g))
    return result


def _gene_stats(gene_data, patients):
    g = gene_data["gene"]
    pts = [p for p in patients if p["gene"] == g]
    n = len(pts)
    mean_age_dx_y = round(sum(p["age_dx_y"] for p in pts) / n, 2)
    pct_normal_iq = round(100 * sum(1 for p in pts if p["normal_iq"]) / n)
    pct_corneal = round(100 * sum(1 for p in pts if p["corneal_clouding"]) / n)
    pct_hepsplen = round(100 * sum(1 for p in pts if p["hepatosplenomegaly"]) / n)
    pct_aai = round(100 * sum(1 for p in pts if p["atlantoaxial_instability"]) / n)
    pct_hydrops = round(100 * sum(1 for p in pts if p["hydrops_fetalis"]) / n)
    pct_hsct = round(100 * sum(1 for p in pts if p["hsct_eligible"]) / n)
    return {
        "gene": g,
        "alias": gene_data["alias"],
        "aa": gene_data["aa"],
        "kDa": gene_data["kDa"],
        "gene_class": gene_data["gene_class"],
        "mps_subgroup": gene_data["mps_subgroup"],
        "locus": gene_data["locus"],
        "omim_gene": gene_data["omim_gene"],
        "phenotype": gene_data["phenotype"],
        "inheritance": gene_data["inheritance"],
        "hallmark": gene_data["hallmark"],
        "key_ddx": gene_data["key_ddx"],
        "diet_treatment": gene_data["diet_treatment"],
        "gene_therapy_status": gene_data["gene_therapy_status"],
        "critical_ci": gene_data["critical_ci"],
        "nbs_marker": gene_data["nbs_marker"],
        "key_biomarker": gene_data["key_biomarker"],
        "severity_spectrum": gene_data["severity_spectrum"],
        "founder_variant": gene_data["founder_variant"],
        "key_variants": gene_data["key_variants"],
        "n_patients": n,
        "mean_age_dx_y": mean_age_dx_y,
        "pct_normal_iq": pct_normal_iq,
        "pct_corneal_clouding": pct_corneal,
        "pct_hepatosplenomegaly": pct_hepsplen,
        "pct_atlantoaxial_instability": pct_aai,
        "pct_hydrops_fetalis": pct_hydrops,
        "pct_hsct_eligible": pct_hsct,
    }


# ── API endpoints ─────────────────────────────────────────────────────────────────
def get_overview():
    patients = _all_patients()
    gene_stats = [_gene_stats(g, patients) for g in MPS_GENES]
    return {
        "atlas": "MPS-Atlas",
        "full_name": "Complete 8-Gene Mucopolysaccharidoses Atlas",
        "n_genes": len(MPS_GENES),
        "n_patients": len(patients),
        "seeds": list(range(SEED_BASE, SEED_BASE + len(MPS_GENES))),
        "gene_subgroups": {
            "Heparan/dermatan sulfate degradation (IDUA · IDS · SGSH · NAGLU · HGSNAT)": [
                "IDUA", "IDS", "SGSH", "NAGLU", "HGSNAT"
            ],
            "Keratan sulfate / chondroitin-6-S degradation (GALNS)":  ["GALNS"],
            "Dermatan sulfate degradation (ARSB)":                     ["ARSB"],
            "Multi-GAG degradation DS+HS+CS (GUSB)":                  ["GUSB"],
        },
        "gene_summary": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "gene_class": g["gene_class"],
                "phenotype": g["phenotype"][:120] + "…",
                "mps_subgroup": g["mps_subgroup"],
                "mean_age_dx_y": gs["mean_age_dx_y"],
            }
            for g, gs in zip(MPS_GENES, gene_stats)
        ],
        "critical_clinical_rules": [
            "IDUA MPS I Hurler: HSCT effective ONLY if <2 years of age, DQ>70 — after 2 yr, neuro window closes. Laronidase ERT does NOT cross BBB.",
            "IDS MPS II Hunter: NO corneal clouding — cardinal distinguishing feature from MPS I Hurler. X-linked: if female, require genetic confirmation before diagnosing MPS II.",
            "GALNS MPS IVA Morquio A: NORMAL intelligence. Atlantoaxial instability (AAI) — cervical MRI MANDATORY before ANY general anaesthesia — risk of quadriplegia/death with minor neck manipulation.",
            "ARSB MPS VI Maroteaux-Lamy: NORMAL intelligence (unlike Hurler) — do not advise intellectual disability in these patients. Urine DS only (not HS), unlike MPS I/II.",
            "GUSB MPS VII Sly: ALWAYS include in workup for unexplained non-immune hydrops fetalis. Vestronidase alfa is approved ERT (every 2 weeks, not weekly like other MPS).",
            "SGSH/NAGLU/HGSNAT Sanfilippo (MPS III): NO approved ERT — do not promise families an IV enzyme replacement. Behavioural management (melatonin, AED) is the current standard.",
            "HGSNAT MPS IIIC: IV ERT inherently inapplicable — transmembrane enzyme cannot be delivered by M6P receptor pathway. Only gene therapy or intrathecal delivery can reach target.",
            "Urine GAG screening: DMB spot or UAMG may MISS Sanfilippo (HS-only, low sensitivity) — always do quantitative heparan sulfate if Sanfilippo suspected.",
        ],
        "nbs_note": (
            "MPS NBS programmes: DBS enzyme assays for MPS I (IDUA), II (IDS), IVA (GALNS), VI (ARSB) "
            "operational in some states/countries. Sanfilippo MPS III has no approved NBS assay "
            "(urine HS not detected on DBS). MPS VII (GUSB) NBS piloted in Taiwan and some US states."
        ),
        "total_patients": len(patients),
    }


def get_breakdown():
    patients = _all_patients()
    gene_stats = [_gene_stats(g, patients) for g in MPS_GENES]
    return {
        "atlas": "MPS-Atlas",
        "total": len(MPS_GENES),
        "total_patients": len(patients),
        "genes": gene_stats,
    }


def get_definitions():
    return {
        "mps_overview": {
            "full_name": "Mucopolysaccharidoses — 8-Gene Atlas",
            "genes_in_atlas": len(MPS_GENES),
            "collective_incidence": "~1/25,000 live births (all MPS combined); MPS I most common (~1/100,000)",
            "nbs_note": "DBS enzyme NBS available for MPS I/II/IVA/VI in select programs. Sanfilippo (III) lacks validated NBS.",
        },
        "definitions": [
            {
                "term": "Glycosaminoglycans (GAGs)",
                "definition": (
                    "Long unbranched polysaccharide chains composed of repeating disaccharide units "
                    "with hexuronic acid + hexosamine. Types: heparan sulfate (HS), dermatan sulfate (DS), "
                    "chondroitin sulfate (CS), keratan sulfate (KS), hyaluronic acid, heparin. "
                    "Normally degraded in lysosomes stepwise by specific sulfatases and glycosidases. "
                    "MPS = deficiency of one enzyme → accumulation of that GAG chain type."
                ),
            },
            {
                "term": "Dysostosis Multiplex",
                "definition": (
                    "Characteristic skeletal X-ray findings in MPS: J-shaped sella turcica, "
                    "oar-shaped ribs (spatula ribs), bullet-nosed/proximal pointing metacarpals, "
                    "hook-shaped vertebrae (L1–L2 beaking), hip dysplasia. "
                    "Caused by GAG accumulation in chondrocytes and ossification centres. "
                    "Seen in MPS I, II, VI, VII. Absent in Morquio A (different skeletal pattern) "
                    "and Sanfilippo III (minimal somatic)."
                ),
            },
            {
                "term": "Corneal Clouding (MPS)",
                "definition": (
                    "GAG (dermatan sulfate or keratan sulfate) deposition in corneal stroma → opacity. "
                    "PRESENT in: MPS I (IDUA), MPS IVA (GALNS), MPS VI (ARSB), MPS VII (GUSB). "
                    "ABSENT in: MPS II Hunter (IDS) — key distinguishing feature. "
                    "ABSENT in: MPS III Sanfilippo (SGSH/NAGLU/HGSNAT). "
                    "Best assessed by slit-lamp biomicroscopy."
                ),
            },
            {
                "term": "Atlantoaxial Instability (AAI) — MPS IVA",
                "definition": (
                    "Odontoid hypoplasia (underdeveloped dens) + lax transverse atlantal ligament → "
                    "C1-C2 subluxation risk. Unique to Morquio A (GALNS deficiency) among MPS, "
                    "due to KS accumulation in ligaments + cartilage. "
                    "CRITICAL: flexion-extension MRI/CT of cervical spine MANDATORY before anaesthesia. "
                    "Risk: cord compression → myelopathy or sudden death from minor trauma. "
                    "Treatment: C1-C2 posterior fusion in symptomatic or high-risk imaging."
                ),
            },
            {
                "term": "Non-Immune Hydrops Fetalis (NIHF) — MPS VII",
                "definition": (
                    "Excess fluid in ≥2 fetal compartments (ascites, pleural/pericardial effusion, "
                    "subcutaneous oedema) without Rh/blood-group isoimmunisation. "
                    "MPS VII (Sly, GUSB) is a recognised metabolic cause. "
                    "GAG accumulation in fetal tissues → protein loss, cardiac failure, lymphatic obstruction. "
                    "MPS VII workup is MANDATORY in unexplained NIHF. "
                    "In-utero ERT (vestronidase alfa) has rescued NIHF fetuses in reported cases."
                ),
            },
            {
                "term": "Laronidase (Aldurazyme) — MPS I ERT",
                "definition": (
                    "Recombinant human α-L-iduronidase (IDUA). 0.58 mg/kg IV weekly. "
                    "Reduces urine DS+HS, liver/spleen volume, improves respiratory function. "
                    "DOES NOT cross blood-brain barrier — cannot treat established Hurler CNS disease. "
                    "Used in Hurler as bridge to HSCT and in Hurler-Scheie/Scheie as long-term therapy."
                ),
            },
            {
                "term": "HSCT in MPS I Hurler — The 2-Year Window",
                "definition": (
                    "Allogeneic HSCT corrects MPS I Hurler if performed before age 2 years in patients "
                    "with DQ >70. Engrafted haematopoietic cells differentiate into microglia → "
                    "provide enzyme to brain via secretion/cross-correction. "
                    "After age 2: neuro correction incomplete. "
                    "Myeloablative conditioning (busulfan + cyclophosphamide). "
                    "ERT bridges patient to HSCT (reduces somatic burden). "
                    "HSCT does NOT correct established skeletal disease — "
                    "dysostosis, corneal clouding, cardiac valves continue to progress."
                ),
            },
            {
                "term": "Elosulfase Alfa (Vimizim) — MPS IVA ERT",
                "definition": (
                    "Recombinant human galactosamine-6-sulfatase (GALNS). 2 mg/kg IV weekly. "
                    "Reduces urine keratan sulfate, improves 6-minute walk test distance, "
                    "stair-climbing endurance. Does NOT correct established skeletal deformity. "
                    "Infusion reactions common (~30%) — pre-medicate with antihistamine/antipyretic. "
                    "Must continue long-term; stopping leads to GAG re-accumulation."
                ),
            },
            {
                "term": "Vestronidase Alfa (Mepsevii) — MPS VII ERT",
                "definition": (
                    "Recombinant human β-glucuronidase (GUSB). 4 mg/kg IV every 2 weeks. "
                    "First FDA-approved ERT for MPS VII (August 2017). "
                    "Dosing interval different from other MPS ERT (biweekly vs weekly). "
                    "In-utero fetal administration used for severe NIHF cases at specialised centres. "
                    "Reduces urine GAG (DS+HS+CS), hepatosplenomegaly."
                ),
            },
            {
                "term": "Urine GAG Quantification (MPS Diagnosis)",
                "definition": (
                    "First-line screening: urine dimethylmethylene blue (DMB) spot or UAMG total GAG. "
                    "Elevated → quantitative breakdown by type: DS, HS, KS, CS. "
                    "MPS I/II: DS+HS. MPS III: HS only. MPS IVA: KS+CS6S. MPS VI: DS only. MPS VII: DS+HS+CS. "
                    "IMPORTANT: DMB/UAMG may MISS Sanfilippo (HS-only, low screening sensitivity) "
                    "→ always use quantitative HS-specific assay when Sanfilippo is suspected."
                ),
            },
            {
                "term": "Sanfilippo Syndrome Behavioral Triad",
                "definition": (
                    "Characteristic of MPS III (SGSH/NAGLU/HGSNAT/GNS): "
                    "(1) Severe hyperactivity and short attention span; "
                    "(2) Aggressive/oppositional behaviour and emotional lability; "
                    "(3) Sleep disturbance — absent or severely fragmented sleep-wake cycle "
                    "(irregular melatonin secretion). "
                    "These three behavioural features combined with any mild dysmorphic feature "
                    "or coarsening should trigger urine HS + Sanfilippo enzyme panel."
                ),
            },
            {
                "term": "HGSNAT Transmembrane Enzyme — ERT Inapplicability",
                "definition": (
                    "HGSNAT is the only lysosomal enzyme in the MPS pathway that is a transmembrane "
                    "acetyltransferase rather than a soluble hydrolase. "
                    "Standard IV ERT works via mannose-6-phosphate (M6P) receptor-mediated endocytosis → "
                    "lysosome delivery of soluble enzyme. "
                    "HGSNAT cannot be delivered this way: transmembrane topology prevents M6P tagging "
                    "and receptor-mediated uptake into the correct membrane location. "
                    "Only viable approaches: gene therapy (AAV delivering HGSNAT gene) or "
                    "intrathecal delivery of gene vector directly to CNS."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== MPS-Atlas Overview ===")
    ov = get_overview()
    print(json.dumps({k: v for k, v in ov.items() if k not in ("gene_summary",)}, indent=2))
    bd = get_breakdown()
    print(f"\n=== MPS-Atlas Breakdown: {bd['total']} genes, {bd['total_patients']} patients ===")
    for g in bd["genes"]:
        print(f"  {g['gene']}: {g['n_patients']} pts, mean dx {g['mean_age_dx_y']}y, "
              f"normal IQ {g['pct_normal_iq']}%, corneal {g['pct_corneal_clouding']}%, "
              f"AAI {g['pct_atlantoaxial_instability']}%")
