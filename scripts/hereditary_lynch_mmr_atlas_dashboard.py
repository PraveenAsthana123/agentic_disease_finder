"""Hereditary Lynch Syndrome / MMR Atlas — 8-Gene Mismatch Repair Reference
MLH1-MSH2-MSH6-PMS2-EPCAM-MLH3-MSH3-PMS1 (Lynch syndrome, CMMRD, MMR-deficient polyposis)
320 patients (8 x 40), seeds 2718-2725.
Endpoints: /api/hereditary-lynch-mmr-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "MLH1",
        "seed_base": 2718,
        "protein": (
            "MLH1 -- 3p22.2 AD/AR -- 793aa -- MutL-Homolog-1-"
            "90kDa-Monomer-MutLalpha-Heterodimerization-Partner-"
            "OMIM-Gene-120436-Disease-Lynch-Syndrome-120435-CMMRD-619212"
        ),
        "locus": "3p22.2",
        "protein_size": "793 aa / 90 kDa (MutLα heterodimerization anchor; ATPase domain N-terminal)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (Lynch syndrome — monoallelic LOF, 50% germline penetrance); "
            "AUTOSOMAL RECESSIVE (CMMRD — constitutional MMR deficiency — biallelic LOF); "
            "MLH1 encodes MutL Homolog 1 — obligate heterodimerization anchor of MutLα (MLH1+PMS2); "
            "MutLα = primary MMR effector endonuclease (PMS2 carries the latent endonuclease, activated by MLH1); "
            "SOMATIC MLH1 SILENCING: promoter CpG methylation → ~15% of all sporadic CRC; "
            "BRAF V600E in MLH1-methylated sporadic CRC (NOT in Lynch) — key diagnostic discriminator; "
            "MLH1 germline: most common cause of Lynch syndrome (~35% of all Lynch families); "
            "EPIGENETIC TRAP: germline MLH1 promoter hypermethylation = constitutional epimutation "
            "(dominant negative, not inherited in Mendelian fashion); "
            "FOUNDER ALLELES: Finnish (c.1853-9G>A splice), Finnish (p.Glu578*), Ashkenazi Jewish "
            "(c.676C>T Arg226Ter), Northern European (large exon 16 deletion); "
            "PREVALENCE: Lynch syndrome 1 in 279 (all genes combined); MLH1 Lynch ~1 in 800"
        ),
        "disease_category": (
            "LYNCH SYNDROME / HNPCC TYPE I (OMIM 120435); "
            "CANCER SPECTRUM (MLH1 — most penetrant): "
            "  COLORECTAL CANCER (CRC): 52-82% lifetime risk (vs 4-5% general population); "
            "    onset median age 44 years (vs 70 sporadic); predominantly right-sided; "
            "    mucinous / signet-ring histology common; tumor-infiltrating lymphocytes (TILs) prominent; "
            "  ENDOMETRIAL CANCER (EC): 25-60% lifetime risk; often synchronous with ovarian; "
            "  GASTRIC CANCER: 6-13% (especially East Asian populations); "
            "  OVARIAN CANCER: 4-12%; "
            "  BILIARY TRACT: 2%; URINARY TRACT: 4-5%; BRAIN: 1-3% (Turcot); "
            "IMMUNOTHERAPY RESPONSE: MSI-H Lynch tumors → pembrolizumab (FDA 2017) — ORR 40%+ "
            "vs chemotherapy; PD-L1 often positive; TIL-rich microenvironment; "
            "BIALLELIC (CMMRD): "
            "  Childhood onset (median age 6-7 years); "
            "  Café-au-lait macules (CALM) ≥3 (NF1 clinical phenocopy); "
            "  Brain tumors (glioblastoma, DIPG, medulloblastoma); "
            "  Hematologic malignancies (T-cell lymphoma, leukemia); "
            "  Rapid multi-tumor development; "
            "COLONOSCOPY SURVEILLANCE: every 1-2 years from age 25 (MLH1/MSH2); "
            "ASPIRIN PROPHYLAXIS: 600mg/day reduces Lynch CRC risk 50% (CAPP2 trial, Burn 2011)"
        ),
        "disease_pathway": (
            "MISMATCH REPAIR (MMR) PATHWAY: "
            "STEP 1 — MISMATCH RECOGNITION: "
            "  MutSα (MSH2+MSH6): recognises base-base mismatches + single-nucleotide IDLs; "
            "  MutSβ (MSH2+MSH3): recognises 2-16 nucleotide insertion-deletion loops (IDLs); "
            "STEP 2 — VERIFICATION + RECRUITMENT: "
            "  MutLα (MLH1+PMS2) recruited by MutS; MLH1 ATPase coordinates strand discrimination; "
            "  PMS2 DQHA(X)2E(X)4E endonuclease activated by MLH1-ATP hydrolysis + PCNA + RFC; "
            "STEP 3 — INCISION: "
            "  PMS2 endonuclease nicks daughter strand → 5' or 3' of mismatch; "
            "  Strand discrimination via interaction with PCNA/RFC on nascent strand; "
            "STEP 4 — EXCISION: "
            "  EXOI (Exonuclease I) excises mismatch-containing fragment; "
            "  RPA protects ssDNA gap; "
            "STEP 5 — RESYNTHESIS + LIGATION: "
            "  Pol δ + PCNA fills gap; ligase I seals nick; "
            "MLH1 LOF → MutLα cannot nick → mismatch persists → accumulation of IDLs → "
            "  microsatellite instability (MSI-H) → hypermutator phenotype → Lynch carcinogenesis; "
            "SECOND-HIT MODEL: germline LOF + somatic loss of heterozygosity (LOH) at 3p22.2"
        ),
        "pathognomonic": (
            "IHC LOSS OF MLH1 + PMS2 PROTEINS TOGETHER: "
            "  MLH1 is required for PMS2 stability (PMS2 degraded without MLH1 partner); "
            "  LOSS MLH1+PMS2 → germline MLH1 mutation OR somatic MLH1 promoter methylation; "
            "  BRAF V600E ABSENCE: Lynch MLH1 tumors are BRAF V600E negative; "
            "  somatic methylation CRC: BRAF V600E present ~30% → NOT Lynch; "
            "  MLH1 METHYLATION TESTING (pyrosequencing/bisulfite): if methylated → somatic; "
            "MICROSATELLITE INSTABILITY (MSI-H): "
            "  PCR: instability at ≥2/5 standard Bethesda markers (BAT-25, BAT-26, D5S346, D2S123, D17S250); "
            "  NGS MSI: preferred (≥30 loci panel); "
            "  MLH1 Lynch tumors: MSI-H invariably (near 100%); "
            "GERMLINE CONFIRMATION: sequencing (SNVs/indels) + MLPA (large deletions/duplications); "
            "  MLH1 large deletions account for ~15% of pathogenic variants"
        ),
        "treatment": (
            "SURVEILLANCE PROTOCOL (MLH1/MSH2 — highest penetrance): "
            "  CRC: colonoscopy every 1-2 years from age 25 (or 5 years before youngest family CRC); "
            "  EC: annual transvaginal ultrasound + endometrial biopsy from age 30-35; "
            "  GASTRIC: endoscopy every 2-3 years (if family history or East Asian ancestry); "
            "CHEMOPREVENTION: "
            "  ASPIRIN 600mg/day — CAPP2 trial: 63% CRC risk reduction in Lynch carriers; "
            "  Optimal dose/duration: ongoing CAPP3 trial (100mg vs 300mg vs 600mg); "
            "RISK-REDUCING SURGERY: "
            "  Prophylactic hysterectomy + bilateral salpingo-oophorectomy (PBSO) after childbearing; "
            "  Consider subtotal colectomy for diagnosis of CRC (reduces metachronous CRC risk); "
            "IMMUNOTHERAPY (TUMOR TREATMENT): "
            "  Pembrolizumab (anti-PD-1): FDA 2017 — MSI-H/dMMR solid tumors; "
            "  KEYSTONE-158: ORR 33%; KEYNOTE-177: 1st line CRC pembrolizumab vs chemo (PFS benefit); "
            "  Lynch tumors: highly immunogenic → superior IO response; "
            "GERMLINE DISCLOSURE: cascade testing first-degree relatives (50% risk each)"
        ),
    },
    {
        "gene": "MSH2",
        "seed_base": 2719,
        "protein": (
            "MSH2 -- 2p21 AD/AR -- 934aa -- MutS-Homolog-2-"
            "100kDa-Monomer-MutSalpha-MSH2-MSH6-and-MutSbeta-MSH2-MSH3-"
            "OMIM-Gene-609309-Disease-Lynch-Syndrome-120435-Muir-Torre-158320"
        ),
        "locus": "2p21",
        "protein_size": "934 aa / 100 kDa (MutSα/MutSβ shared subunit; ATPase + mismatch-binding)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (Lynch syndrome — monoallelic LOF); "
            "MSH2 encodes the shared subunit of both MutSα (MSH2+MSH6) and MutSβ (MSH2+MSH3); "
            "MSH2 LOF → BOTH MutSα AND MutSβ inactivated → broadest MMR defect; "
            "EPCAM-CAUSED MSH2 SILENCING: EPCAM 3' deletion → read-through transcription → "
            "  MSH2 promoter CpG methylation (epigenetic silencing without MSH2 mutation); "
            "  IHC: MSH2+MSH6 lost; EPCAM protein NORMAL; MSH2 sequencing NORMAL; "
            "  Account for ~5-10% of apparent Lynch syndrome; "
            "SYNDROME OVERLAP: "
            "  MUIR-TORRE SYNDROME: Lynch + sebaceous gland neoplasms (sebaceous adenoma/carcinoma/epithelioma) "
            "    + visceral malignancies; almost exclusively MSH2/MLH1; sebaceous tumors = sentinel lesion; "
            "  TURCOT SYNDROME: Lynch variant with primary brain tumor (glioblastoma); MSH2/MLH1; "
            "FOUNDER ALLELES: "
            "  Newfoundland (large MSH2 deletion), Spanish (p.Ala636Pro), Dutch (exon 4-8 deletion via EPCAM); "
            "PREVALENCE: MSH2 Lynch ~1 in 800; one of most common causes"
        ),
        "disease_category": (
            "LYNCH SYNDROME (OMIM 120435) + MUIR-TORRE SYNDROME (OMIM 158320); "
            "CANCER SPECTRUM (MSH2 — broad, similar to MLH1): "
            "  CRC: 51-75% lifetime risk; predominantly right-sided; MSI-H invariably; "
            "  URINARY TRACT CANCER (UCC): 28% lifetime risk — HIGHEST of all Lynch genes; "
            "    upper urinary tract (renal pelvis + ureter); OBLIGATE Lynch surveillance target; "
            "  ENDOMETRIAL CANCER: 25-60% lifetime; "
            "  GASTRIC CANCER: 13% (highest gastric risk in Lynch); "
            "  OVARIAN: 11%; BILIARY: 5%; BRAIN: 2-4%; "
            "  SEBACEOUS GLAND NEOPLASMS (Muir-Torre): pathognomonic of MSH2/MLH1 Lynch; "
            "    sebaceous adenoma OR sebaceous carcinoma → referral for Lynch testing; "
            "IHC PATTERN: MSH2+MSH6 LOST TOGETHER (MSH6 degrades without MSH2 partner); "
            "  IMPORTANT EXCEPTION: EPCAM deletion → MSH2+MSH6 IHC lost, EPCAM intact; "
            "UROTHELIAL SURVEILLANCE: annual urinalysis + cytology from age 30-35 (MSH2 carriers)"
        ),
        "disease_pathway": (
            "MSH2 DUAL ROLE IN MMR: "
            "MutSα (MSH2+MSH6): "
            "  Recognises base-base mismatches (G/T, A/C) + 1-nt insertion-deletion loops; "
            "  Dominant MMR complex for single-base errors; "
            "MutSβ (MSH2+MSH3): "
            "  Recognises 2-16 nt insertion-deletion loops (IDLs) in microsatellite sequences; "
            "  Major role in dinucleotide repeat instability (D5S346, D2S123 in Bethesda panel); "
            "MSH2 LOF → BOTH complexes lost: "
            "  All mismatch types accumulate → broad MSI-H phenotype; "
            "  All 5 Bethesda markers typically unstable (vs MSH6 LOF which can show MSI-L); "
            "EPCAM-MSH2 EPIGENETIC MECHANISM: "
            "  EPCAM encodes epithelial cell adhesion molecule; its gene is immediately 3' upstream of MSH2; "
            "  EPCAM 3'-end genomic deletion (not truncating EPCAM protein) → "
            "    read-through transcription from EPCAM across MSH2 → "
            "    de novo CpG methylation of MSH2 promoter in MSH2-adjacent tissues → "
            "    MSH2 silencing in epithelial cells (colon, endometrium) → Lynch phenotype; "
            "    SOMATIC: methylation often tissue-specific (mosaic); "
            "  KEY: MSH2 sequence NORMAL; EPCAM sequence shows 3' deletion"
        ),
        "pathognomonic": (
            "SEBACEOUS GLAND NEOPLASM (Muir-Torre Sentinel): "
            "  Sebaceous adenoma or sebaceous carcinoma in patient <60 without prior Lynch diagnosis "
            "  → MANDATORY Lynch workup (>75% have Lynch syndrome); "
            "  All sebaceous tumors should undergo MMR IHC testing; "
            "IHC: MSH2 + MSH6 PROTEIN LOSS (together): "
            "  Both lost = MSH2 pathogenic variant OR EPCAM 3' deletion causing MSH2 methylation; "
            "  IMPORTANT: MSH6 alone lost → MSH6 mutation (MSH2 intact); "
            "UPPER URINARY TRACT TUMORS: renal pelvis/ureter transitional cell carcinoma "
            "  in Lynch-age patient → reflexive MMR testing (MSH2 most common cause); "
            "EPCAM WORKUP: MSH2 sequencing normal → test EPCAM for large 3' deletions (MLPA); "
            "  Confirm methylation of MSH2 promoter in tumor or blood (bisulfite sequencing)"
        ),
        "treatment": (
            "MSH2-SPECIFIC SURVEILLANCE: "
            "  CRC: colonoscopy every 1-2 years from age 25; "
            "  URINARY TRACT (HIGHEST RISK GENE): "
            "    Annual urinalysis + cytology from age 30; "
            "    Consider upper tract endoscopy (ureteroscopy) if cytology abnormal; "
            "  GASTRIC: endoscopy every 2-3 years; "
            "  EC/OVARY: annual pelvic ultrasound + endometrial biopsy from age 30; "
            "MUIR-TORRE MANAGEMENT: "
            "  Dermatologic annual survey; surgical excision of sebaceous lesions; "
            "  Visceral cancer surveillance identical to Lynch; "
            "CHEMOPREVENTION: Aspirin 600mg/day (CAPP2 data applies); "
            "RISK-REDUCING SURGERY: Hysterectomy + BSO after childbearing; "
            "IMMUNOTHERAPY: Pembrolizumab/nivolumab for MSI-H Lynch tumors; "
            "EPCAM-CAUSED CASES: Same Lynch surveillance; no risk difference from MSH2 point mutation; "
            "  Offspring at risk of inheriting EPCAM deletion (autosomal dominant)"
        ),
    },
    {
        "gene": "MSH6",
        "seed_base": 2720,
        "protein": (
            "MSH6 -- 2p16.3 AD -- 1360aa -- MutS-Homolog-6-"
            "160kDa-Monomer-MutSalpha-MSH2-MSH6-Mismatch-Binding-Subunit-"
            "OMIM-Gene-600678-Disease-Lynch-Syndrome-600678"
        ),
        "locus": "2p16.3",
        "protein_size": "1360 aa / 160 kDa (MutSα mismatch-binding subunit; PCNA-interacting PIP-box)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (Lynch syndrome — monoallelic LOF; penetrance lower than MLH1/MSH2); "
            "MSH6 encodes the mismatch-binding subunit of MutSα (MSH2+MSH6); "
            "MSH6 LOF → MutSα-specific defect (MutSβ = MSH2+MSH3 INTACT); "
            "CONSEQUENCE: predominantly single-nucleotide mismatches unrepaired (IDLs partially repaired by MutSβ); "
            "MSI PATTERN: MSI-L (microsatellite instability - low) OR MSI-H possible; "
            "  Standard Bethesda panel (mono/dinucleotide markers) may miss MSH6 Lynch (30% MSS!); "
            "  Mononucleotide markers (BAT-25, BAT-26) more sensitive for MSH6; "
            "ATTENUATED PHENOTYPE: later onset CRC (median 54 years vs 44 MLH1); lower CRC penetrance (25-40%); "
            "  ENDOMETRIAL CANCER: HIGHEST LIFETIME RISK of all Lynch genes (~70%); "
            "FOUNDER ALLELES: "
            "  Portuguese (p.Leu1027Ile), Dutch (p.Phe1088Leufs*5), US Ashkenazi (3 founder variants); "
            "PREVALENCE: MSH6 Lynch ~1 in 1,000 — may be more common than MLH1 in population-based series"
        ),
        "disease_category": (
            "LYNCH SYNDROME — ATTENUATED COLORECTAL / ENDOMETRIAL PREDOMINANT (OMIM 614350); "
            "CANCER SPECTRUM (MSH6 — attenuated + EC-dominant): "
            "  ENDOMETRIAL CANCER: 57-71% lifetime risk — HIGHEST of all Lynch genes; "
            "    onset median age 55 years; "
            "    EC PRESENTATION = index Lynch diagnosis in MSH6 families more often than CRC; "
            "  CRC: 25-40% lifetime risk (lower than MLH1/MSH2); median onset 54 years; "
            "    MSI-L or MSI-H (not invariably MSI-H as in MLH1/MSH2); "
            "  OVARIAN: 11-18%; URINARY TRACT: 9%; BILIARY: rare; "
            "IHC PATTERN: MSH6 ALONE LOST (MSH2 intact = MSH2 still forms MutSβ with MSH3); "
            "  KEY: MSH6 alone = MSH6 germline; both MSH2+MSH6 lost = MSH2 or EPCAM; "
            "MSI TESTING CAVEAT: "
            "  Standard Bethesda 5-marker panel: 30% MSH6 Lynch tumors = MSS or MSI-L; "
            "  Expanded mononucleotide marker panel required (or NGS-based MSI); "
            "  Never rule out Lynch on MSS result alone without IHC"
        ),
        "disease_pathway": (
            "MSH6 IN MutSα: "
            "  MSH6 N-terminal mismatch-binding domain (MBD): directly contacts mismatched base; "
            "  MSH6 PWWP domain + PIP-box: PCNA interaction → localizes to replication fork; "
            "  MSH2 ATPase domain drives conformational change after mismatch binding; "
            "MSH6 LOF CONSEQUENCES: "
            "  MutSα (MSH2+MSH6) eliminated → single-nucleotide mismatches not recognised; "
            "  MutSβ (MSH2+MSH3) INTACT → dinucleotide IDLs still repaired; "
            "  RESULT: partial MMR deficiency → predominantly base-substitution errors accumulate; "
            "  MICROSATELLITE PATTERN: "
            "    Mononucleotide repeats (BAT-25/BAT-26): unstable (sensitive); "
            "    Dinucleotide repeats (D5S346, D2S123): often stable (MutSβ intact); "
            "  EXPLAINS MSI-L pattern in MSH6 Lynch (unique among Lynch genes); "
            "  MUTATION SPECTRUM: POLE exonuclease domain mutations can phenocopy MSH6 deficiency "
            "(both show MSI-L, IHC may be equivocal — check POLE mutational signature); "
            "ASPIRIN MECHANISM: COX-2 inhibition reduces prostaglandin E2 → promotes mismatch-containing "
            "cell apoptosis → reduced Lynch adenoma progression"
        ),
        "pathognomonic": (
            "IHC: MSH6 PROTEIN LOSS ALONE (MSH2 INTACT): "
            "  MSH6 LOF; MSH2 forms MutSβ with MSH3 → MSH2 IHC preserved; "
            "  Pattern discriminates MSH6 pathogenic variant from MSH2 (which loses MSH2+MSH6 together); "
            "ENDOMETRIAL CANCER AS INDEX PRESENTATION: "
            "  EC in woman <50 OR EC with family history of CRC → IHC all MMR proteins; "
            "  MSH6-loss IHC in EC: 90%+ have germline MSH6 variant; "
            "MSI-L / EQUIVOCAL MSI ON BETHESDA PANEL: "
            "  Do NOT exclude Lynch — order expanded mononucleotide marker panel or NGS MSI; "
            "  MSH6 Lynch tumors have 30% false-negative rate on standard Bethesda; "
            "  IHC should be performed REGARDLESS of MSI result when Lynch suspected clinically; "
            "AMSTERDAM II CRITERIA NEGATIVE FAMILIES: "
            "  MSH6 Lynch commonly missed by Amsterdam criteria (lower CRC rate); "
            "  Bethesda revised criteria + IHC + germline testing recommended for all CRC/EC"
        ),
        "treatment": (
            "MSH6-SPECIFIC SURVEILLANCE (later onset, adjusted): "
            "  CRC: colonoscopy every 2-3 years from age 30-35 (later than MLH1/MSH2 given lower/later penetrance); "
            "  ENDOMETRIAL: Annual endometrial biopsy + pelvic ultrasound from age 30-35 (MANDATORY — highest EC risk); "
            "  OVARY: Pelvic ultrasound + CA-125 annually from age 30; "
            "  Urinary tract: annual urinalysis from age 30; "
            "RISK-REDUCING SURGERY: "
            "  Hysterectomy + BSO after childbearing — most important intervention for EC prevention; "
            "  DECISION: weigh high EC lifetime risk (70%) against surgical risk; "
            "CHEMOPREVENTION: Aspirin 600mg/day (CAPP2); aspirin data extrapolated from MLH1/MSH2; "
            "IMMUNOTHERAPY: Pembrolizumab/nivolumab for MSI-H Lynch cancers; "
            "  NOTE: MSI-L MSH6 tumors may be less responsive to checkpoint inhibitors; "
            "MSI TESTING IN TUMORS: always include IHC (do not rely on Bethesda MSI alone); "
            "FAMILY COUNSELLING: penetrance lower → individualized risk communication"
        ),
    },
    {
        "gene": "PMS2",
        "seed_base": 2721,
        "protein": (
            "PMS2 -- 7p22.1 AD/AR -- 862aa -- PMS1-Homolog-2-"
            "96kDa-Monomer-MutLalpha-Endonuclease-Subunit-DQHAXE-Motif-"
            "OMIM-Gene-600259-Disease-Lynch-Syndrome-614337-CMMRD-276300"
        ),
        "locus": "7p22.1",
        "protein_size": "862 aa / 96 kDa (MutLα endonuclease subunit; DQHA(X)2E(X)4E motif C-terminal)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (Lynch syndrome — monoallelic LOF; most attenuated); "
            "AUTOSOMAL RECESSIVE (CMMRD — constitutional mismatch repair deficiency — biallelic LOF); "
            "PMS2 encodes the endonuclease subunit of MutLα (MLH1+PMS2); "
            "PMS2 C-terminal DQHAXEXE motif = latent endonuclease, activated by PCNA + RFC + MLH1; "
            "PSEUDOGENE COMPLICATION: ~15 PMS2 pseudogenes (PMS2CL most problematic) → "
            "  standard sequencing may miss PMS2 variants (false negatives); "
            "  PMS2-specific long-range PCR + MLPA required; "
            "MONOALLELIC PHENOTYPE (Lynch): most attenuated Lynch gene — "
            "  CRC lifetime risk 15-20% (vs 52-82% for MLH1/MSH2); "
            "  Later onset (median 60s); "
            "BIALLELIC PHENOTYPE (CMMRD): "
            "  MOST COMMON biallelic Lynch cause (PMS2 most common CMMRD gene after MLH1); "
            "  Childhood brain tumors: glioblastoma, DIPG, medulloblastoma — median age 7 years; "
            "  Café-au-lait macules (CALMs): NF1 phenocopy — CRUCIAL DDx from NF1; "
            "  CRC at 2nd-3rd decade; Hematologic: T-lymphoma, lymphoma; "
            "  SPORE REVIEW: CMMRD registry (Vasen 2014): median survival 28 months without surveillance"
        ),
        "disease_category": (
            "LYNCH SYNDROME — ATTENUATED (OMIM 614337); CMMRD (OMIM 276300); "
            "MONOALLELIC — ATTENUATED LYNCH: "
            "  CRC: 15-20% lifetime risk (lowest of the 4 canonical Lynch genes); "
            "  EC: 15-35%; "
            "  Other cancers: low penetrance; "
            "  SURVEILLANCE: colonoscopy every 2-3 years from age 35-40; "
            "BIALLELIC — CMMRD (Constitutional MMR Deficiency): "
            "  BRAIN TUMORS: glioblastoma GBM (most common), diffuse intrinsic pontine glioma (DIPG), "
            "    medulloblastoma, ependymoma — median age diagnosis 6-7 years; "
            "  CRC / POLYPOSIS: colorectal polyposis + early CRC (median age 18-22 years); "
            "  HEMATOLOGIC: T-cell lymphoma, ALL, AML (often early childhood); "
            "  NF1 PHENOCOPY: ≥3 CALMs ± axillary freckling — NF1 sequencing NORMAL; "
            "    CRITICAL DDx: NF1 vs CMMRD in child with CALMs + brain tumor; "
            "    CMMRD: IHC loss of MMR in tumor + normal NF1 + positive germline PMS2 biallelic; "
            "  MICROSATELLITE: constitutional MSI in all tissues (germline MSI detectable in blood); "
            "  PROGNOSIS: without surveillance — median survival 28 months from diagnosis; "
            "  SURVEILLANCE CMMRD: annual MRI brain + annual colonoscopy from age 8; "
            "    Annual blood count for hematologic; dermatologic surveillance"
        ),
        "disease_pathway": (
            "PMS2 ENDONUCLEASE MECHANISM: "
            "  PMS2 C-terminal domain (residues 675-862) = latent endonuclease (DQHAXEXE motif + Zn2+ coordination); "
            "  Endonuclease activation requires: "
            "    1. Heterodimerization with MLH1 (N-terminal PMS2 + MLH1 C-terminal interaction); "
            "    2. PCNA binding (PIP-box on PMS2); "
            "    3. RFC (replication factor C) association; "
            "    4. MutSα/MutSβ signaling via ATPase exchange; "
            "  ACTIVATED: PMS2 nicks the daughter strand → signals EXOI to excise; "
            "PMS2 PSEUDOGENE PROBLEM: "
            "  PMS2CL (pseudogene on chr. 7) shares >98% identity with PMS2 exons 9-15; "
            "  Most clinical labs use PMS2-specific MLPA + long-range PCR; "
            "  Pathogenic variants identified only by long-range PCR + bidirectional sequencing; "
            "  Consequence: PMS2 Lynch historically under-ascertained; "
            "CMMRD MSI MECHANISM: "
            "  Biallelic PMS2 LOF → total MutLα loss → constitutive MSI in all cells; "
            "  Blood MSI detectable (unlike somatic-only MSI in sporadic cancers); "
            "  Constitutional MSI assay: normal lymphocytes can be tested for MSI"
        ),
        "pathognomonic": (
            "CMMRD — CAFÉ-AU-LAIT MACULES (CALMs) + BRAIN TUMOR IN CHILD: "
            "  NF1 phenocopy — ≥3 CALMs, no Lisch nodules, no NF1 mutation; "
            "  Brain tumor (GBM/DIPG) in child <10 + CALMs → CMMRD first in DDx; "
            "  CMMRD SOS-MMR IHC: glioblastoma IHC shows PMS2 loss (or PMS2+MLH1 loss); "
            "IHC: PMS2 ALONE LOST (MLH1 intact): "
            "  PMS2 LOF → PMS2 protein absent; MLH1 intact (MLH1 pairs with PMS1/MLH3 in backup complexes); "
            "  IMPORTANT: Unlike PMS2 loss, MLH1 loss → both MLH1+PMS2 lost (PMS2 requires MLH1); "
            "BLOOD MSI TEST (CMMRD ONLY): "
            "  Constitutional germline MSI detectable in blood cells → PATHOGNOMONIC of CMMRD; "
            "  Somatic (Lynch) cancers: MSI only in tumor tissue; blood MSI = CMMRD; "
            "PMS2 GERMLINE TESTING: "
            "  Requires PMS2-specific long-range PCR (not NGS alone → pseudogene contamination); "
            "  MLPA specific for PMS2 exons"
        ),
        "treatment": (
            "PMS2 MONOALLELIC SURVEILLANCE (ATTENUATED LYNCH): "
            "  CRC: colonoscopy every 2-3 years from age 35-40 (later onset, lower risk); "
            "  EC: annual endometrial biopsy from age 35; "
            "  Lower intensity surveillance justified by attenuated phenotype; "
            "CMMRD MANAGEMENT (HIGH URGENCY): "
            "  BRAIN TUMOR: annual MRI brain/spine from birth or at diagnosis; "
            "  CRC/POLYPOSIS: colonoscopy every 6-12 months from age 8 (bowel polyposis); "
            "  HEMATOLOGIC: annual CBC + LDH; blood film for lymphocytosis; "
            "IMMUNOTHERAPY IN CMMRD: "
            "  PD-1 inhibitors (pembrolizumab, nivolumab) — promising in CMMRD brain tumors; "
            "  MSI-H GBM in children → pembrolizumab compassionate use / clinical trials; "
            "  Leesha trial (2024): pembrolizumab in pediatric CMMRD solid tumors; "
            "GENETIC COUNSELLING: "
            "  PMS2 monoallelic carrier has 25% risk (if partner also carrier) of CMMRD child; "
            "  Partner PMS2 carrier testing before conception; "
            "  PGT (preimplantation genetic testing) for CMMRD families"
        ),
    },
    {
        "gene": "EPCAM",
        "seed_base": 2722,
        "protein": (
            "EPCAM -- 2p21 AD -- 314aa -- Epithelial-Cell-Adhesion-Molecule-"
            "35kDa-Monomer-Type-I-Transmembrane-EpCAM-CD326-"
            "OMIM-Gene-185535-Disease-Lynch-Syndrome-5-614350-Congenital-Tufting-Enteropathy-226730"
        ),
        "locus": "2p21",
        "protein_size": "314 aa / 35 kDa (type I transmembrane; thyroglobulin domains; signal peptide 23aa)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (Lynch syndrome — monoallelic 3' genomic deletion); "
            "EPCAM gene is located immediately 5' (upstream) of MSH2 on chromosome 2p21; "
            "MECHANISM: EPCAM 3' genomic deletions (not truncating variants) → "
            "  read-through transcription from EPCAM through MSH2 → "
            "  de novo CpG methylation of MSH2 promoter → tissue-specific MSH2 silencing; "
            "NOT AN MMR GENE: EPCAM protein has no role in DNA mismatch repair; "
            "  Lynch is caused by EPCAM deletion's epigenetic effect on MSH2, not EPCAM protein loss; "
            "BIALLELIC LOF (different mechanism): biallelic EPCAM LOF → "
            "  CONGENITAL TUFTING ENTEROPATHY (CTE): rare severe intestinal failure in neonates; "
            "  No MSH2 silencing (biallelic EPCAM protein loss → CTE, not Lynch); "
            "  CTE: villus tufts on biopsy (pathognomonic); total parenteral nutrition required; "
            "GERMLINE PREVALENCE: EPCAM deletions account for ~5-10% of Lynch not explained by MMR sequencing; "
            "  Dutch founder EPCAM deletion (exons 4-9 deletion with 3' extension → MSH2 methylation)"
        ),
        "disease_category": (
            "LYNCH SYNDROME TYPE 5 — MSH2-SILENCED BY EPCAM DELETION (OMIM 614350); "
            "CANCER SPECTRUM (EPCAM-Lynch — similar to MSH2 Lynch): "
            "  CRC: 75% lifetime risk (similar to MSH2-Lynch); predominantly right-sided; "
            "  EC: 12-25% (lower than canonical MSH2-Lynch — tissue-specificity of methylation matters); "
            "  GASTRIC: 11%; OVARIAN: 6%; URINARY TRACT: 10%; "
            "  TISSUE SPECIFICITY OF METHYLATION: colon mucosa (high risk) > endometrium (variable) "
            "  > bladder (variable); "
            "IHC PATTERN: MSH2 + MSH6 LOST TOGETHER (identical to MSH2 mutation); "
            "  EPCAM protein: NORMAL (tumors still express EPCAM — not the mechanism of silencing); "
            "  MSH2 sequencing: NORMAL (no mutation in MSH2 gene); "
            "  CONGENITAL TUFTING ENTEROPATHY (biallelic EPCAM LOF): "
            "    Neonatal onset severe diarrhea, intestinal failure; "
            "    Villus tufts on jejunal biopsy (pathognomonic); "
            "    TPN-dependent; bowel transplant in severe cases"
        ),
        "disease_pathway": (
            "EPCAM-TO-MSH2 EPIGENETIC SILENCING: "
            "STEP 1 — EPCAM 3' DELETION: "
            "  Genomic deletion removes 3' portion of EPCAM (including poly-A signal/3'UTR); "
            "  Truncation variant in EPCAM coding sequence (stop codon retained) → Lynch NOT caused; "
            "  Must be 3' regulatory deletion removing termination signal; "
            "STEP 2 — READ-THROUGH TRANSCRIPTION: "
            "  Without poly-A signal, RNA Pol II does not terminate at EPCAM; "
            "  Reads through into MSH2 gene (which is directly adjacent downstream); "
            "STEP 3 — MSH2 PROMOTER METHYLATION: "
            "  Read-through transcription induces de novo DNA methylation at MSH2 promoter CpGs; "
            "  Mechanism: transcription-associated methylation (RNAII passage triggers DNMT1 recruitment?); "
            "STEP 4 — TISSUE-SPECIFIC SILENCING: "
            "  MSH2 promoter methylated in specific tissues (colon, endometrium, gastric); "
            "  Other tissues (blood, fibroblasts) may show low-level methylation or none; "
            "  IHC in tumor: MSH2 protein absent; EPCAM protein present (EpCAM not silenced); "
            "NORMAL EPCAM FUNCTION: "
            "  EpCAM: epithelial cell adhesion molecule (E-cadherin modifier, Wnt/β-catenin signaling); "
            "  Overexpressed in many carcinomas (not related to Lynch mechanism)"
        ),
        "pathognomonic": (
            "IHC: MSH2+MSH6 LOST WITH NORMAL EPCAM IHC: "
            "  Lynch phenotype (MSH2+MSH6 loss) with no MSH2 germline mutation → "
            "  INVESTIGATE EPCAM for 3' deletion; "
            "  EPCAM MLPA: must use probes covering EPCAM 3' region; "
            "  Confirmation: MSH2 promoter methylation in blood or tumor; "
            "METHYLATION DETECTION: "
            "  Bisulfite pyrosequencing of MSH2 promoter in blood lymphocytes: "
            "    Germline EPCAM deletion carrier → mosaic MSH2 promoter methylation detectable; "
            "    Tumor tissue: high-level methylation (much higher than blood); "
            "NO MSH2 MUTATION FOUND ON SEQUENCING: "
            "  After MSH2 sequencing NORMAL + MLPA NORMAL → test EPCAM for large 3' deletions; "
            "  EPCAM testing required in Lynch diagnosis algorithm (NCCN, ESMO guidelines); "
            "CONGENITAL TUFTING ENTEROPATHY (biallelic EPCAM LOF): "
            "  Neonatal onset protracted diarrhea + failure to thrive; "
            "  EGD biopsy: villous tufts, teardrop enterocytes (pathognomonic); "
            "  EPCAM IHC: absent in intestine (vs Lynch where EPCAM is present)"
        ),
        "treatment": (
            "EPCAM-LYNCH SURVEILLANCE (same as MSH2-Lynch given similar cancer spectrum): "
            "  CRC: colonoscopy every 1-2 years from age 25; "
            "  EC: annual endometrial biopsy + transvaginal ultrasound from age 30-35; "
            "  GASTRIC: endoscopy every 2-3 years; "
            "  URINARY TRACT: annual urinalysis + cytology; "
            "RISK-REDUCING SURGERY: "
            "  Hysterectomy + BSO after childbearing; "
            "  Consider subtotal colectomy if CRC found; "
            "MOLECULAR TESTING ALGORITHM: "
            "  If Lynch suspected + MSH2/MSH6 IHC lost + MSH2 sequencing normal + MLPA normal: "
            "    STEP 1: EPCAM MLPA (3' end specific probes); "
            "    STEP 2: If EPCAM deletion found → MSH2 methylation confirmation; "
            "    STEP 3: Counsel as Lynch (MSH2-equivalent risk); "
            "IMMUNOTHERAPY: Pembrolizumab/nivolumab for MSI-H EPCAM-Lynch tumors; "
            "CONGENITAL TUFTING ENTEROPATHY: TPN support; bowel rehabilitation program; "
            "  Intestinal transplant for TPN-dependent cases; ciprofloxacin for bacterial overgrowth"
        ),
    },
    {
        "gene": "MLH3",
        "seed_base": 2723,
        "protein": (
            "MLH3 -- 14q24.3 AR/AD -- 1453aa -- MutL-Homolog-3-"
            "165kDa-Monomer-MutLgamma-MLH1-MLH3-Meiotic-MMR-"
            "OMIM-Gene-604395-Disease-Lynch-Syndrome-Modifier-Colorectal-Cancer"
        ),
        "locus": "14q24.3",
        "protein_size": "1453 aa / 165 kDa (MutLγ = MLH1+MLH3; latent endonuclease C-terminal)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic — polyposis/Lynch-like predisposition); "
            "AUTOSOMAL DOMINANT (monoallelic — modest modifier, low/moderate Lynch risk); "
            "MLH3 encodes MutL Homolog 3 — dimerizes with MLH1 to form MutLγ complex; "
            "MutLγ (MLH1+MLH3): "
            "  Primary role: MEIOTIC MMR — resolves mismatches in recombination intermediates; "
            "  Secondary role: Mitotic MMR backup — particularly for IDLs (redundant with MutLα); "
            "MLH3 LOF consequences: "
            "  Reduced meiotic crossing-over fidelity; "
            "  Partial MMR deficiency (MutLα intact — PMS2-MLH1); "
            "  Lower cancer risk than canonical Lynch genes; "
            "MALE INFERTILITY: MLH3 biallelic LOF → meiotic arrest → azoospermia in males "
            "(MLH3 role in spermatogenesis and oocyte maturation); "
            "PREVALENCE: rare pathogenic variants; ~1-3% of unexplained Lynch-suspected families; "
            "CANCER RISK: modest — CRC 15-30% lifetime; lower than MLH1/MSH2; "
            "IHC: MLH3 protein NOT routinely included in clinical MMR IHC panel (4-protein panel: MLH1/PMS2/MSH2/MSH6)"
        ),
        "disease_category": (
            "LYNCH SYNDROME MODIFIER / HEREDITARY CRC PREDISPOSITION (OMIM 604395); "
            "BIALLELIC MLH3 (RARER — polyposis/Lynch-like): "
            "  CRC: 30-50% lifetime; multiple adenomas (adenomatous polyposis); "
            "  Onset: 30-50 years (later than canonical Lynch); "
            "  No dominant endometrial/ovarian contribution; "
            "MONOALLELIC MLH3 (MODIFIER): "
            "  Modest Lynch-like CRC risk (possibly 20-30%); "
            "  Population variants (MLH3 Ile219Val — common polymorphism) — NOT pathogenic; "
            "MEIOTIC PHENOTYPE (biallelic males): "
            "  Azoospermia / severe oligospermia — MLH3 meiotic arrest; "
            "  Similar to MLH3 knockout mouse (infertile, tumor-prone); "
            "MSI PATTERN: "
            "  Variable — may show partial MSI or MSS (MutLα intact); "
            "  IHC: standard 4-protein panel (MLH1/PMS2/MSH2/MSH6) often NORMAL "
            "  (MLH3 not included, and MLH1+PMS2 intact); "
            "  Molecular: MLH3-specific germline sequencing required for diagnosis"
        ),
        "disease_pathway": (
            "MutLγ = MLH1 + MLH3: "
            "MEIOTIC MMR FUNCTION (primary): "
            "  During meiosis I: MutSγ (MSH4+MSH5) recognises Holliday junction intermediates; "
            "  MutLγ (MLH1+MLH3) resolves crossover intermediates; "
            "  MLH3 endonuclease (DQHAXEXE) makes nicks → crossover resolution; "
            "  MLH3 LOF → aberrant crossovers → meiotic arrest → infertility; "
            "MITOTIC MMR BACKUP (secondary): "
            "  MutLγ can substitute for MutLα (MLH1+PMS2) for IDL repair; "
            "  Lower efficiency → partial MMR in MLH3 LOF; "
            "  MutLα remains the primary mitotic MMR effector; "
            "MLH3 ENDONUCLEASE ACTIVATION: "
            "  Same mechanism as PMS2 (DQHAXEXE motif + Zn2+); "
            "  Activated by PCNA + RFC + MLH1; "
            "TUMOR DEVELOPMENT: "
            "  Partial MMR deficiency → accumulation of IDLs at microsatellites → instability; "
            "  Lower rate than canonical Lynch → explains attenuated cancer phenotype; "
            "  CRC adenoma → carcinoma sequence: microsatellite unstable pathway"
        ),
        "pathognomonic": (
            "STANDARD MMR IHC PANEL: OFTEN NORMAL (MLH1+PMS2+MSH2+MSH6 ALL EXPRESSED): "
            "  MLH3 is NOT in the standard clinical IHC panel; "
            "  Clinician must request MLH3 germline sequencing based on: "
            "    Lynch-like family history + normal IHC + MMR gene sequencing negative; "
            "    Male infertility + family CRC history (azoospermia clue); "
            "MSI VARIABLE: "
            "  Tumors may be MSS or partial MSI (MutLα still functional); "
            "  Mononucleotide repeat panel more sensitive (as with MSH6); "
            "MALE INFERTILITY CLUE: "
            "  Azoospermia/severe oligospermia in young male + family CRC history → "
            "  Consider biallelic MLH3 (OMIM 604395 + infertility OMIM 614809); "
            "GERMLINE TESTING: "
            "  MLH3 sequencing + MLPA in Lynch-suspected cases after canonical MMR genes negative; "
            "  Bioinformatic variant classification challenging (many VUS)"
        ),
        "treatment": (
            "MLH3 SURVEILLANCE (modest risk — individualized): "
            "  CRC: colonoscopy every 2-3 years from age 35-40 (lower risk than canonical Lynch); "
            "  Consider polypectomy surveillance schedule if adenomas found; "
            "  Less established evidence base than for MLH1/MSH2; "
            "MALE INFERTILITY (biallelic): "
            "  Testicular sperm extraction (TESE) may recover sperm for ICSI; "
            "  Preimplantation genetic testing to avoid transmitting biallelic MLH3; "
            "SURVEILLANCE STANDARD: "
            "  MLH3 variants are less established as Lynch-causing; "
            "  Multidisciplinary genetics team review for MLH3 VUS; "
            "  NCCN: MLH3 not in primary Lynch surveillance protocol (insufficient evidence); "
            "  European guidelines: MLH3 pathogenic variants → Lynch-lite surveillance; "
            "ASPIRIN: extrapolated from Lynch data; reasonable to offer; "
            "GENETIC COUNSELLING: "
            "  Distinguish pathogenic variant from common polymorphism (Ile219Val = NOT pathogenic); "
            "  Biallelic: offspring at risk for infertility + CRC predisposition"
        ),
    },
    {
        "gene": "MSH3",
        "seed_base": 2724,
        "protein": (
            "MSH3 -- 5q14.1 AR -- 1137aa -- MutS-Homolog-3-"
            "128kDa-Monomer-MutSbeta-MSH2-MSH3-IDL-Repair-"
            "OMIM-Gene-600887-Disease-DMMR-Polyposis-Glioblastoma-Predisposition"
        ),
        "locus": "5q14.1",
        "protein_size": "1137 aa / 128 kDa (MutSβ IDL-repair subunit; ATPase; no direct mismatch binding)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF required for polyposis phenotype); "
            "MSH3 encodes the IDL-repair subunit of MutSβ (MSH2+MSH3); "
            "MutSβ (MSH2+MSH3): recognises 2-16 nt insertion-deletion loops (IDLs) in microsatellites; "
            "MutSα (MSH2+MSH6) INTACT → base-substitution repair unaffected; "
            "MSH3 LOF CONSEQUENCES: "
            "  Dinucleotide repeat instability (not mononucleotide — that requires MutSα); "
            "  Specific MSI pattern: D5S346, D2S123 unstable (dinucleotide markers); "
            "  BAT-25, BAT-26 (mononucleotide markers) STABLE (MutSα intact); "
            "DISEASE: Adenomatous polyposis + CRC + rare glioblastoma predisposition; "
            "NOT CLASSIC LYNCH: no endometrial cancer excess; no typical Lynch extracolonic cancers; "
            "CANCER SPECTRUM DISTINCT: CRC + colorectal polyposis + glioblastoma (brain) + "
            "  duodenal/upper GI + eye (EGGD syndrome: eyes-gut-glioma-duodenum); "
            "PREVALENCE: very rare — <50 families reported; biallelic required → AR"
        ),
        "disease_category": (
            "DMMR POLYPOSIS / MSH3-ASSOCIATED HEREDITARY CRC (distinct from Lynch syndrome); "
            "CLINICAL FEATURES: "
            "  COLORECTAL POLYPOSIS: multiple adenomas (often 10-100+); CRC risk 60-80% by age 50; "
            "  GLIOBLASTOMA (GBM): 5-15% lifetime risk — high-grade glioma in adulthood; "
            "  DUODENAL ADENOMAS: duodenal polyposis ± duodenal adenocarcinoma; "
            "  UPPER GI: similar to attenuated FAP but MSI-H rather than APC-driven; "
            "EGGD SYNDROME: Eye + Gut + Glioma + Duodenal (proposed term for biallelic MSH3); "
            "IHC PATTERN: MSH3 PROTEIN ABSENT (MSH2 INTACT): "
            "  MSH2 still expressed (partners with MSH6 to form MutSα); "
            "  MSH2+MSH6 IHC: NORMAL; MLH1+PMS2: NORMAL; "
            "  MSH3 IHC: not routinely performed clinically → underdiagnosed; "
            "  NGS-based MSI: dinucleotide repeat instability (specific signature); "
            "MSI PATTERN: "
            "  Bethesda panel: MSI-L or MSS (BAT-25/BAT-26 mononucleotides STABLE); "
            "  Dinucleotide markers: UNSTABLE (D5S346, D2S123); "
            "  NGS microsatellite panel distinguishes MSH3-specific IDL signature"
        ),
        "disease_pathway": (
            "MutSβ = MSH2 + MSH3: "
            "IDL RECOGNITION: "
            "  MutSβ specifically recognises insertion-deletion loops of 2-16 nucleotides; "
            "  These arise from polymerase slippage at dinucleotide/tetranucleotide microsatellite repeats; "
            "  MSH3 provides the IDL-binding specificity; MSH2 provides ATPase for sliding; "
            "MSH3 LOF CONSEQUENCES: "
            "  2-16 nt IDLs NOT repaired → dinucleotide repeat instability accumulates; "
            "  Single-base mismatches: REPAIRED (MutSα = MSH2+MSH6 intact); "
            "  RESULT: specific MSI pattern (dinucleotide unstable, mononucleotide stable); "
            "COMPARE TO MSH6 LOF: "
            "  MSH6 LOF: mononucleotide unstable, dinucleotide stable (MutSβ intact); "
            "  MSH3 LOF: mononucleotide stable, dinucleotide unstable (MutSα intact); "
            "  These are RECIPROCAL IDL defects; "
            "GLIOBLASTOMA MECHANISM: "
            "  MSH3 LOF → IDL accumulation in EGFR microsatellite → EGFR amplification? "
            "  Dinucleotide repeat instability in glioblastoma driver genes → GBM susceptibility; "
            "COLORECTAL POLYPOSIS: "
            "  IDL repair failure → APC microsatellite instability → adenoma initiation; "
            "  Multiple adenomas → one undergoes second-hit → CRC"
        ),
        "pathognomonic": (
            "COLORECTAL POLYPOSIS + NORMAL STANDARD MMR IHC (MLH1/PMS2/MSH2/MSH6 ALL PRESENT): "
            "  Standard 4-protein IHC: NORMAL (MSH3 not included); "
            "  APC sequencing: NEGATIVE (not FAP); MUTYH: NEGATIVE (not MAP); "
            "  UNEXPLAINED POLYPOSIS: if standard testing negative → "
            "    MSH3 sequencing + MSH3-specific IHC in tumor; "
            "    Dinucleotide-specific MSI panel (not standard Bethesda); "
            "GLIOBLASTOMA IN YOUNG PATIENT + FAMILY CRC HISTORY: "
            "  GBM <50 + adenomatous polyposis → MSH3 biallelic in DDx; "
            "  MSH3 germline sequencing in tumor (MSI-L/specific signature) + germline; "
            "DINUCLEOTIDE-SPECIFIC MSI SIGNATURE: "
            "  NGS MSI panel: unstable at dinucleotide markers (D5S346, D2S123); "
            "  Mononucleotide (BAT-25, BAT-26): STABLE → discriminates from Lynch; "
            "  MSH3-specific mutational signature (COSMIC Signature 44 overlap?); "
            "IHC SPECIAL REQUEST: "
            "  MSH3 IHC (research grade, not routine): absent MSH3 confirms diagnosis"
        ),
        "treatment": (
            "MSH3 POLYPOSIS MANAGEMENT: "
            "  COLONOSCOPY: annual from age 25 (polypectomy for all adenomas; CRC prevention); "
            "  If >100 adenomas or uncontrollable: prophylactic subtotal colectomy + ileorectal anastomosis; "
            "GLIOBLASTOMA SURVEILLANCE: "
            "  Annual MRI brain from age 25-30 (if family history of GBM); "
            "  No established standard; extrapolated from CMMRD protocols; "
            "DUODENAL: "
            "  EGD (duodenoscopy) every 2-3 years from age 25-30; "
            "  Spigelman score for duodenal polyposis severity; "
            "IMMUNOTHERAPY IN TUMORS: "
            "  Dinucleotide-unstable tumors (MSH3-CRC): likely respond to checkpoint inhibitors; "
            "  PD-L1 checkpoint inhibitors active in dMMR regardless of which MMR protein lost; "
            "  MSH3 tumors: IHC MSI test negative on standard panel → "
            "    request extended MSI panel to confirm dMMR for pembrolizumab eligibility; "
            "GENETIC COUNSELLING: "
            "  AR inheritance: 25% risk each child if both parents carriers; "
            "  Carrier parents: no personal cancer risk elevation (monoallelic MSH3 = not pathogenic); "
            "  Sibling testing: all siblings tested for biallelic MSH3"
        ),
    },
    {
        "gene": "PMS1",
        "seed_base": 2725,
        "protein": (
            "PMS1 -- 2q31.1 AD -- 932aa -- PMS1-Homolog-1-"
            "103kDa-Monomer-MutLbeta-MLH1-PMS1-Mismatch-Repair-Modifier-"
            "OMIM-Gene-600258-Disease-Lynch-Syndrome-Modifier"
        ),
        "locus": "2q31.1",
        "protein_size": "932 aa / 103 kDa (MutLβ = MLH1+PMS1; no intrinsic endonuclease; modifier role)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (monoallelic — Lynch modifier; low/moderate cancer risk); "
            "PMS1 encodes PMS1 Homolog 1 — dimerizes with MLH1 to form MutLβ; "
            "MutLβ (MLH1+PMS1): "
            "  LACKS intrinsic endonuclease (PMS1 does not have the DQHAXEXE motif of PMS2/MLH3); "
            "  MutLβ function: primarily a regulatory complex; stimulates MutLα activity; "
            "  MutLβ competes with MutLα for MLH1 dimerization (titration regulatory role); "
            "PMS1 LOF consequence: "
            "  MutLβ absent → less competition for MLH1 with PMS2; "
            "  Net: modest MMR efficiency change; lower cancer risk than canonical Lynch; "
            "HISTORICAL CONTEXT: "
            "  PMS1 originally identified as Lynch gene in Fishel/Kolodner 1993 HNPCC studies; "
            "  Later studies showed lower penetrance than MLH1/MSH2; "
            "  Current evidence: PMS1 = low-moderate risk modifier; "
            "  NCCN does not recommend Lynch-intensity surveillance for monoallelic PMS1 alone "
            "(insufficient evidence for high penetrance); "
            "PREVALENCE: very rare confirmed pathogenic variants; many VUS; "
            "POPULATION RISK: 2-fold CRC risk increase vs general population"
        ),
        "disease_category": (
            "LYNCH SYNDROME MODIFIER — LOW PENETRANCE CRC PREDISPOSITION (OMIM 600258); "
            "CANCER SPECTRUM (PMS1 — modest): "
            "  CRC: ~20% lifetime risk (low-moderate; lower than canonical Lynch genes); "
            "  EC: modest increase (10-15%?); "
            "  Gastric, ovarian: low excess risk; "
            "IHC PATTERN: "
            "  PMS1 not included in standard MMR IHC panel; "
            "  MLH1+PMS2+MSH2+MSH6: all expressed (PMS1 absence does not destabilize these proteins); "
            "  Standard MMR IHC: NORMAL in PMS1 LOF families; "
            "MSI STATUS: "
            "  Variable — may show mild MSI or MSS; "
            "  MutLα (MLH1+PMS2) INTACT → primary MMR preserved; "
            "  PMS1 absence → reduced regulatory stimulation of MutLα; "
            "CLINICAL MANAGEMENT CHALLENGE: "
            "  PMS1 is difficult to classify as high-risk Lynch gene; "
            "  Many variants: VUS classification (uncertain significance); "
            "  Clinicians often extrapolate Lynch surveillance until evidence clearer"
        ),
        "disease_pathway": (
            "MutLβ = MLH1 + PMS1: "
            "STRUCTURE: "
            "  PMS1 C-terminal dimerizes with MLH1 C-terminal (same interface as PMS2); "
            "  PMS1 lacks endonuclease motif (distinct from PMS2 which has DQHAXEXE); "
            "  MutLβ: forms but cannot nick DNA (no intrinsic endonuclease); "
            "FUNCTION PROPOSED: "
            "  Regulatory/competitive role: MutLβ modulates MutLα activity; "
            "  MutLβ may stimulate EXO1-dependent excision pathway; "
            "  MutLβ has MLH1-dependent ATPase activity (regulatory not catalytic); "
            "PMS1 LOF MECHANISM: "
            "  Loss of MutLβ → subtle MMR efficiency reduction; "
            "  MutLα (MLH1+PMS2) compensates → partial correction; "
            "  Mild IDL accumulation over decades → modest CRC predisposition; "
            "COMPARE: "
            "  PMS2 LOF (MutLα endonuclease lost) → much greater MMR defect → higher Lynch risk; "
            "  PMS1 LOF (MutLβ regulatory role lost) → modest MMR change → lower cancer risk; "
            "MEIOTIC ROLE: PMS1 also has meiotic MMR function (minor); "
            "  PMS1 knockout mice: male infertility (reduced crossing-over efficiency)"
        ),
        "pathognomonic": (
            "STANDARD MMR IHC PANEL: NORMAL (all 4 MMR proteins expressed): "
            "  MLH1 present, PMS2 present, MSH2 present, MSH6 present; "
            "  PMS1 not in clinical IHC panel → underdiagnosed; "
            "UNEXPLAINED FAMILY HISTORY + NEGATIVE CANONICAL LYNCH TESTING: "
            "  After MLH1/MSH2/MSH6/PMS2/EPCAM all negative → "
            "    Consider PMS1 sequencing as expanded Lynch panel; "
            "    Many labs include PMS1 in comprehensive Lynch gene panel now; "
            "MSI VARIABILITY: "
            "  Some PMS1 LOF tumors: MSS or MSI-L; "
            "  IHC and MSI cannot diagnose PMS1 Lynch (both often normal); "
            "  Diagnosis = germline sequencing finding; "
            "VUS CHALLENGE: "
            "  PMS1 has many variants of uncertain significance; "
            "  Functional assays (yeast complementation, MMR activity assay) required for classification; "
            "POPULATION SCREENING IMPLICATION: "
            "  PMS1 cancer risk modest → germline testing positive result management challenging; "
            "  Multidisciplinary review with clinical geneticist + gastroenterologist"
        ),
        "treatment": (
            "PMS1 MANAGEMENT (individualized, extrapolated): "
            "  CRC: colonoscopy every 3-5 years from age 40-45 (similar to average-risk surveillance "
            "  + family history uplift; not full Lynch intensity); "
            "  EC: annual endometrial surveillance if family history; "
            "  NO CONSENSUS GUIDELINE for PMS1 alone (insufficient high-penetrance evidence); "
            "RISK STRATIFICATION: "
            "  Family history analysis: if multiple first-degree relatives with Lynch cancers → "
            "    escalate to Lynch-intensity surveillance despite PMS1-only finding; "
            "  Polygenic risk modifier: CRC PRS (polygenic risk score) to stratify individuals; "
            "ASPIRIN: reasonable to discuss (Lynch data); "
            "GENETIC COUNSELLING COMPLEXITY: "
            "  Penetrance estimates for PMS1 imprecise; "
            "  Patient communication: 'lower risk than full Lynch but real cancer risk increase'; "
            "  Family cascade testing: reasonable for first-degree relatives; "
            "RESEARCH CONTEXT: "
            "  PMS1 pathogenic variant carriers invited to join Lynch registries; "
            "  Functional studies (yeast MMR assays) to reclassify PMS1 VUS; "
            "IMMUNOTHERAPY: if tumor shows dMMR/MSI-H → pembrolizumab eligible "
            "(regardless of which MMR gene affected)"
        ),
    },
]


def _make_patients(gene_data, n=40):
    """Generate synthetic Lynch/MMR patient records for a gene."""
    rng = random.Random(gene_data["seed_base"])
    gene = gene_data["gene"]

    # Gene-specific clinical phenotype distributions
    GENE_PARAMS = {
        "MLH1": dict(
            onset_range=(28, 55), crc=0.85, ec=0.45, gastric=0.10, ovarian=0.08,
            biliary=0.03, urinary=0.05, brain=0.02, sebaceous=0.00, polyposis=0.30,
            cmmrd=0.00, cafe_au_lait=0.00, msi_h=0.97, immunotherapy=0.20,
        ),
        "MSH2": dict(
            onset_range=(32, 58), crc=0.78, ec=0.40, gastric=0.12, ovarian=0.10,
            biliary=0.04, urinary=0.28, brain=0.03, sebaceous=0.08, polyposis=0.25,
            cmmrd=0.00, cafe_au_lait=0.00, msi_h=0.95, immunotherapy=0.22,
        ),
        "MSH6": dict(
            onset_range=(38, 65), crc=0.32, ec=0.65, gastric=0.05, ovarian=0.14,
            biliary=0.02, urinary=0.09, brain=0.01, sebaceous=0.02, polyposis=0.10,
            cmmrd=0.00, cafe_au_lait=0.00, msi_h=0.70, immunotherapy=0.15,
        ),
        "PMS2": dict(
            onset_range=(15, 60), crc=0.22, ec=0.22, gastric=0.03, ovarian=0.04,
            biliary=0.01, urinary=0.04, brain=0.30, sebaceous=0.00, polyposis=0.35,
            cmmrd=0.25, cafe_au_lait=0.25, msi_h=0.85, immunotherapy=0.18,
        ),
        "EPCAM": dict(
            onset_range=(30, 56), crc=0.72, ec=0.18, gastric=0.10, ovarian=0.06,
            biliary=0.04, urinary=0.10, brain=0.02, sebaceous=0.00, polyposis=0.22,
            cmmrd=0.00, cafe_au_lait=0.00, msi_h=0.92, immunotherapy=0.18,
        ),
        "MLH3": dict(
            onset_range=(35, 62), crc=0.25, ec=0.08, gastric=0.04, ovarian=0.03,
            biliary=0.01, urinary=0.03, brain=0.04, sebaceous=0.00, polyposis=0.20,
            cmmrd=0.00, cafe_au_lait=0.00, msi_h=0.50, immunotherapy=0.10,
        ),
        "MSH3": dict(
            onset_range=(30, 58), crc=0.70, ec=0.04, gastric=0.06, ovarian=0.02,
            biliary=0.02, urinary=0.03, brain=0.12, sebaceous=0.00, polyposis=0.85,
            cmmrd=0.00, cafe_au_lait=0.00, msi_h=0.40, immunotherapy=0.12,
        ),
        "PMS1": dict(
            onset_range=(42, 68), crc=0.20, ec=0.12, gastric=0.03, ovarian=0.03,
            biliary=0.01, urinary=0.03, brain=0.02, sebaceous=0.00, polyposis=0.08,
            cmmrd=0.00, cafe_au_lait=0.00, msi_h=0.35, immunotherapy=0.08,
        ),
    }

    p = GENE_PARAMS.get(gene, GENE_PARAMS["MLH1"])
    patients = []
    for i in range(n):
        onset = round(rng.uniform(*p["onset_range"]), 1)
        age_current = round(onset + rng.uniform(2, 30), 1)
        patients.append({
            "patient_id": f"{gene}-{gene_data['seed_base']}-{i+1:02d}",
            "sex": rng.choice(["M", "F"]),
            "age_onset_years": onset,
            "age_current_years": min(age_current, 85.0),
            "colorectal_cancer": rng.random() < p["crc"],
            "endometrial_cancer": rng.random() < p["ec"],
            "gastric_cancer": rng.random() < p["gastric"],
            "ovarian_cancer": rng.random() < p["ovarian"],
            "urinary_tract_cancer": rng.random() < p["urinary"],
            "brain_tumor": rng.random() < p["brain"],
            "sebaceous_neoplasm": rng.random() < p["sebaceous"],
            "colorectal_polyposis": rng.random() < p["polyposis"],
            "cmmrd_phenotype": rng.random() < p["cmmrd"],
            "cafe_au_lait_macules": rng.random() < p["cafe_au_lait"],
            "msi_high": rng.random() < p["msi_h"],
            "immunotherapy_received": rng.random() < p["immunotherapy"],
        })
    return patients


def generate_overview():
    all_genes = []
    total_patients = 0

    GENE_PARAMS = {
        "MLH1": dict(onset=44, crc=85, ec=45, gastric=10, urinary=5, brain=2, sebaceous=0, polyposis=30, cmmrd=0, msi_h=97),
        "MSH2": dict(onset=46, crc=78, ec=40, gastric=12, urinary=28, brain=3, sebaceous=8, polyposis=25, cmmrd=0, msi_h=95),
        "MSH6": dict(onset=54, crc=32, ec=65, gastric=5, urinary=9, brain=1, sebaceous=2, polyposis=10, cmmrd=0, msi_h=70),
        "PMS2": dict(onset=50, crc=22, ec=22, gastric=3, urinary=4, brain=30, sebaceous=0, polyposis=35, cmmrd=25, msi_h=85),
        "EPCAM": dict(onset=45, crc=72, ec=18, gastric=10, urinary=10, brain=2, sebaceous=0, polyposis=22, cmmrd=0, msi_h=92),
        "MLH3": dict(onset=48, crc=25, ec=8, gastric=4, urinary=3, brain=4, sebaceous=0, polyposis=20, cmmrd=0, msi_h=50),
        "MSH3": dict(onset=42, crc=70, ec=4, gastric=6, urinary=3, brain=12, sebaceous=0, polyposis=85, cmmrd=0, msi_h=40),
        "PMS1": dict(onset=56, crc=20, ec=12, gastric=3, urinary=3, brain=2, sebaceous=0, polyposis=8, cmmrd=0, msi_h=35),
    }

    for gene_data in ATLAS_GENES:
        gene = gene_data["gene"]
        p = GENE_PARAMS[gene]
        patients = _make_patients(gene_data)
        total_patients += len(patients)
        all_genes.append({
            "gene": gene,
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "n_patients": len(patients),
            "mean_onset_years": p["onset"],
            "pct_crc": p["crc"],
            "pct_ec": p["ec"],
            "pct_gastric": p["gastric"],
            "pct_urinary": p["urinary"],
            "pct_brain": p["brain"],
            "pct_sebaceous": p["sebaceous"],
            "pct_polyposis": p["polyposis"],
            "pct_cmmrd": p["cmmrd"],
            "pct_msi_high": p["msi_h"],
        })

    return {
        "atlas": "Hereditary Lynch Syndrome / MMR Atlas",
        "subtitle": "Complete 8-Gene Mismatch Repair Reference",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "gene_summaries": all_genes,
        "seeds": "2718-2725",
        "pathway_categories": [
            {
                "pathway": "MutSα Complex (Base Mismatch + 1-nt IDL)",
                "genes": ["MSH2", "MSH6"],
                "note": (
                    "MutSα = MSH2+MSH6 heterocomplex; recognises single-base mismatches and 1-nt IDLs; "
                    "MSH6 binds mismatch directly; MSH2 provides ATPase; "
                    "MSH2 LOF → MutSα AND MutSβ both lost (shared subunit); "
                    "MSH6 LOF → only MutSα lost; MutSβ intact → partial MMR → MSI-L possible"
                ),
            },
            {
                "pathway": "MutSβ Complex (2-16 nt IDL / Dinucleotide Repeat Instability)",
                "genes": ["MSH2", "MSH3"],
                "note": (
                    "MutSβ = MSH2+MSH3 heterocomplex; recognises 2-16 nt insertion-deletion loops; "
                    "MSH3 LOF → dinucleotide repeat instability; mononucleotide stable (MutSα intact); "
                    "MSH3-specific: AR polyposis + glioblastoma; NOT classic Lynch cancers"
                ),
            },
            {
                "pathway": "MutLα Endonuclease Complex (Primary MMR Effector)",
                "genes": ["MLH1", "PMS2"],
                "note": (
                    "MutLα = MLH1+PMS2; PMS2 has DQHAXEXE endonuclease nicking daughter strand; "
                    "MLH1 anchors complex + activates PMS2 endonuclease; "
                    "MLH1 LOF → MLH1+PMS2 both lost on IHC (PMS2 degrades without MLH1); "
                    "PMS2 LOF → PMS2 alone lost (MLH1 intact, partners with PMS1/MLH3 in backup)"
                ),
            },
            {
                "pathway": "MutLγ + MutLβ (Meiotic/Backup MMR Complexes)",
                "genes": ["MLH1", "MLH3", "PMS1"],
                "note": (
                    "MutLγ = MLH1+MLH3 (meiotic MMR; IDL backup; MLH3 endonuclease active); "
                    "MutLβ = MLH1+PMS1 (regulatory modifier; no intrinsic endonuclease); "
                    "MLH3/PMS1 LOF → modest MMR efficiency change; lower cancer penetrance than canonical Lynch"
                ),
            },
            {
                "pathway": "Epigenetic MSH2 Silencing (EPCAM)",
                "genes": ["EPCAM"],
                "note": (
                    "EPCAM 3' deletion → read-through transcription → MSH2 promoter CpG methylation; "
                    "Lynch phenotype without MSH2 mutation; IHC: MSH2+MSH6 lost, EPCAM present; "
                    "Account for 5-10% Lynch not explained by MMR gene sequencing; "
                    "EPCAM biallelic LOF → Congenital Tufting Enteropathy (different mechanism)"
                ),
            },
        ],
        "critical_distinctions": [
            "MLH1+PMS2 IHC BOTH LOST: MLH1 germline OR somatic MLH1 methylation → BRAF V600E absent in Lynch but present in ~30% somatic-methylation CRC — KEY discriminator",
            "MSH2+MSH6 IHC BOTH LOST: MSH2 germline OR EPCAM 3' deletion → must test EPCAM by MLPA if MSH2 sequencing normal",
            "MSH6 IHC ALONE LOST: MSH6 germline — MSH2 still forms MutSβ with MSH3 → MSI-L possible on standard panel (30% false-negative) — do IHC regardless of MSI result",
            "PMS2 IHC ALONE LOST: PMS2 germline OR somatic — MLH1 intact (forms MutLβ/MutLγ with PMS1/MLH3) — CMMRD if biallelic (childhood brain tumors + CALMs + constitutional blood MSI)",
            "MSH3 POLYPOSIS: standard MMR IHC NORMAL; dinucleotide unstable MSI (not mononucleotide); AR — MSH3 not in standard 4-protein IHC panel → commonly missed",
            "EPCAM 3' DELETION: Lynch without MMR mutation — EPCAM-specific MLPA required; MSH2 methylation confirmation; MSH2+MSH6 IHC lost despite no MSH2 coding mutation",
            "CMMRD vs NF1: biallelic PMS2/MLH1 → ≥3 CALMs (NF1 phenocopy); NF1 sequencing NORMAL; blood MSI POSITIVE in CMMRD (pathognomonic); childhood brain tumors",
            "MSI-L/MSS IN LYNCH: MSH6 (30%) and MSH3 (60%) Lynch tumors may be MSS on Bethesda — never exclude Lynch on MSI result alone; IHC is more sensitive",
            "ASPIRIN PROPHYLAXIS: CAPP2 trial — 600mg/day aspirin reduces Lynch CRC risk 50% (MLH1/MSH2 carriers, 2-year treatment); mechanism via mismatch-cell apoptosis promotion",
            "IMMUNOTHERAPY ELIGIBILITY: all MSI-H/dMMR Lynch tumors → pembrolizumab (FDA 2017); even MSI-L Lynch tumors if IHC confirms dMMR — use IHC not MSI alone for IO eligibility",
        ],
    }


def generate_breakdown():
    genes_out = []
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        pct = lambda k: round(100 * sum(p[k] for p in patients) / len(patients))
        genes_out.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "n_patients": len(patients),
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "pct_crc": pct("colorectal_cancer"),
            "pct_ec": pct("endometrial_cancer"),
            "pct_gastric": pct("gastric_cancer"),
            "pct_ovarian": pct("ovarian_cancer"),
            "pct_urinary": pct("urinary_tract_cancer"),
            "pct_brain": pct("brain_tumor"),
            "pct_sebaceous": pct("sebaceous_neoplasm"),
            "pct_polyposis": pct("colorectal_polyposis"),
            "pct_cmmrd": pct("cmmrd_phenotype"),
            "pct_cafe_au_lait": pct("cafe_au_lait_macules"),
            "pct_msi_high": pct("msi_high"),
            "pct_immunotherapy": pct("immunotherapy_received"),
            "patients": patients[:40],
        })
    return {"genes": genes_out}


def generate_definitions():
    return {
        "glossary": {
            "Lynch Syndrome (HNPCC)": (
                "Hereditary nonpolyposis colorectal cancer (HNPCC); autosomal dominant MMR gene "
                "pathogenic variant (MLH1/MSH2/MSH6/PMS2/EPCAM); most common hereditary CRC syndrome "
                "(1:279 population); 50-85% CRC lifetime (MLH1/MSH2); diagnosed by Amsterdam criteria, "
                "Bethesda guidelines, IHC, MSI testing, germline sequencing"
            ),
            "CMMRD (Constitutional MMR Deficiency)": (
                "Biallelic MMR gene pathogenic variant (PMS2/MLH1/MSH2/MSH6); childhood onset; "
                "cafe-au-lait macules (NF1 phenocopy); brain tumors + CRC + hematologic malignancies; "
                "constitutional blood MSI detectable (pathognomonic); pembrolizumab promising; "
                "CMMRD-care consortium international registry"
            ),
            "Mismatch Repair (MMR)": (
                "Post-replication DNA repair pathway correcting base-base mismatches and insertion-deletion loops; "
                "MutS complexes (MutSα=MSH2+MSH6; MutSβ=MSH2+MSH3) recognize mismatches; "
                "MutL complexes (MutLα=MLH1+PMS2; MutLγ=MLH1+MLH3; MutLβ=MLH1+PMS1) coordinate repair; "
                "EXOI excises mismatch-containing strand; Pol δ fills gap; ligase seals"
            ),
            "MSI-H (Microsatellite Instability - High)": (
                "≥2/5 standard Bethesda markers unstable (BAT-25, BAT-26, D5S346, D2S123, D17S250); "
                "or ≥30% of microsatellite loci unstable on NGS panel; "
                "MMR-deficient tumors (dMMR); Lynch OR sporadic MLH1-methylated CRC; "
                "MSI-H = pembrolizumab-eligible (FDA 2017); not MSI = may miss MSH6/MSH3 Lynch"
            ),
            "MSI-L (Microsatellite Instability - Low)": (
                "1/5 Bethesda markers unstable; MSH6 Lynch: 30% tumors MSI-L or MSS (MutSα lost; "
                "MutSβ intact repairs dinucleotide repeats); MSH3 LOF: mononucleotide stable, "
                "dinucleotide unstable; NEVER exclude Lynch on MSI-L alone — IHC more sensitive"
            ),
            "dMMR (deficient MMR)": (
                "Loss of MMR protein expression on immunohistochemistry (IHC); equivalent to MSI-H "
                "in most contexts; used for pembrolizumab eligibility; "
                "IHC 4-panel (MLH1/PMS2/MSH2/MSH6) detects canonical Lynch and most sporadic dMMR; "
                "MSH3/MLH3/PMS1 not in standard panel → missed by routine IHC"
            ),
            "Amsterdam II Criteria": (
                "Clinical Lynch diagnosis (pre-molecular era): ≥3 relatives with Lynch cancers + "
                "spanning ≥2 generations + ≥1 case before age 50 + ≥1 first-degree relative of the other two; "
                "Specificity: 89%; Sensitivity: 61% — misses attenuated Lynch (MSH6/PMS2); "
                "Revised Bethesda guidelines more sensitive for molecular testing indication"
            ),
            "BRAF V600E": (
                "V600E somatic BRAF mutation found in ~30% of sporadic MLH1-methylated CRC (not Lynch); "
                "BRAF V600E ABSENT in Lynch MLH1 tumors; "
                "Reflexive BRAF V600E testing after MLH1+PMS2 IHC loss distinguishes Lynch from sporadic; "
                "If BRAF V600E positive → somatic methylation → not Lynch → no germline testing needed "
                "(unless young patient, family history, or MLH1 methylation negative)"
            ),
            "IHC (MMR Immunohistochemistry)": (
                "4-protein panel: MLH1, PMS2, MSH2, MSH6; loss of any protein = dMMR; "
                "INTERPRETATION RULES: "
                "MLH1+PMS2 lost → MLH1 mutation or methylation; "
                "PMS2 alone lost → PMS2 mutation; "
                "MSH2+MSH6 lost → MSH2 mutation or EPCAM deletion; "
                "MSH6 alone lost → MSH6 mutation; "
                "All present (pMMR) → MSH3/MLH3/PMS1 possible, or MSS non-Lynch CRC"
            ),
            "Pembrolizumab (Keytruda)": (
                "Anti-PD-1 checkpoint inhibitor; FDA 2017: first tumor-agnostic approval (MSI-H/dMMR solid tumors); "
                "Lynch CRC: KEYNOTE-177 — pembrolizumab superior to FOLFOX/FOLFIRI in 1st-line MSI-H CRC "
                "(PFS HR 0.60; OS benefit); ORR 43% vs 33%; recommended 1st-line MSI-H CRC; "
                "Lynch non-CRC: basket trials ongoing; CMMRD brain tumors: compassionate use"
            ),
            "EPCAM 3' Deletion": (
                "Genomic deletion removing EPCAM 3' poly-A signal/3'UTR → read-through → MSH2 methylation; "
                "NOT a truncating coding variant (those don't cause Lynch); "
                "Detection: MLPA specific to EPCAM 3' region; confirmation by MSH2 promoter bisulfite sequencing; "
                "Dutch founder deletion (exons 4-9 + 3' extension) — 10% of apparent Lynch"
            ),
            "CAPP2 Trial": (
                "Cancer Prevention Programme 2; Burn et al. 2011 Lancet; Lynch carriers randomized to "
                "aspirin 600mg/day vs placebo; 2-year treatment: 63% CRC risk reduction at 5-year follow-up; "
                "CAPP3 ongoing: dose-finding (100mg vs 300mg vs 600mg) — results awaited 2026; "
                "Mechanism: COX-2 inhibition → prostaglandin E2 reduction → mismatch-cell apoptosis promotion"
            ),
            "DMMR Polyposis (MSH3)": (
                "Distinct from Lynch syndrome; biallelic MSH3 → multiple adenomas (often >10); "
                "CRC + glioblastoma (EGGD: eye-gut-glioma-duodenum); "
                "IHC: standard 4-panel NORMAL (MSH3 absent but not tested); "
                "MSI: dinucleotide unstable, mononucleotide STABLE (MutSα intact) — unique signature; "
                "AR inheritance; carrier parents (monoallelic) have normal cancer risk"
            ),
            "Congenital Tufting Enteropathy (CTE)": (
                "Biallelic EPCAM LOF (different from Lynch EPCAM deletions); "
                "Neonatal onset protracted diarrhea + intestinal failure; "
                "EGD biopsy: villous tufts (teardrop enterocytes) — PATHOGNOMONIC; "
                "EPCAM IHC: absent in intestinal epithelium; "
                "No MSH2 silencing (EPCAM protein absent → not read-through mechanism); "
                "TPN-dependent; intestinal transplant in severe cases"
            ),
        },
        "standards": [
            "NCCN Guidelines v2.2025: Lynch Syndrome — genetic/familial high-risk assessment (colorectal)",
            "ESMO Clinical Practice Guidelines: Lynch Syndrome — Moller et al. 2022",
            "CAPP2 Trial: Burn et al. Lancet 2011 — aspirin 600mg CRC prevention in Lynch",
            "KEYNOTE-177: André et al. NEJM 2020 — pembrolizumab 1st-line MSI-H/dMMR CRC",
            "ICLinCS (International CMMRD Consortium): CMMRD surveillance protocol 2014 (Vasen)",
            "EPCAM Lynch: Ligtenberg et al. Nat Genet 2009 — EPCAM deletion causes MSH2 silencing",
            "MSH3 DMMR Polyposis: Sekine et al. Nature 2019; Adam et al. Am J Hum Genet 2021",
            "Amsterdam II Criteria: Vasen et al. Gastroenterology 1999",
            "Revised Bethesda Guidelines: Umar et al. JNCI 2004",
            "dMMR IHC interpretation: Rubio 2019 — 4-panel protocol (MLH1/PMS2/MSH2/MSH6)",
            "PMS2 pseudogene: Vaughn et al. Genet Med 2011 — long-range PCR required",
            "Muir-Torre: Singh et al. Fam Cancer 2017 — sebaceous tumor MMR testing mandatory",
        ],
    }
