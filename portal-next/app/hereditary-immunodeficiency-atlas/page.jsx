'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  BTK:    '#1565c0',  // deep blue — XLA, X-linked agammaglobulinemia
  RAG1:   '#b71c1c',  // deep red — Omenn/SCID, V(D)J recombination
  ADA:    '#e65100',  // deep orange — ADA-SCID, first gene therapy
  CYBB:   '#4a148c',  // deep purple — CGD gp91phox, X-linked
  WAS:    '#006064',  // dark cyan — Wiskott-Aldrich XL
  LRBA:   '#2e7d32',  // deep green — LRBA deficiency, abatacept
  CTLA4:  '#f57f17',  // amber — CTLA4 haploinsufficiency AD
  PIK3CD: '#4e342e',  // deep brown — APDS1 GOF, leniolisib
};

const GENE_DISEASE = {
  BTK:    'XLA XL Absent B Cells — Monthly IVIG Lifelong — No Live Vaccines EVER — Bronchiectasis Prevention — BTK Flow Cytometry Absent Monocyte BTK Confirm',
  RAG1:   'Omenn/SCID AR V(D)J Arrest — Spectrum SCID-Leaky-Omenn — TREC Newborn Screen — BCG Avoid ABSOLUTELY — HSCT Curative — Omenn Rash+Eosinophilia+High-IgE',
  ADA:    'ADA-SCID AR Metabolic dATP Lymphotoxic — Strimvelis EMA 2016 First Gene Therapy — PEG-ADA Enzyme Replacement — HSCT Curative — Skeletal Dysplasia Radiograph',
  CYBB:   'CGD XL gp91phox NADPH Oxidase Null — Aspergillus Staph Serratia Nocardia Burkholderia — DHR Assay Diagnostic — IFN-gamma 70pct Infection Reduction — Itraconazole+TMP-SMX Lifelong',
  WAS:    'Wiskott-Aldrich XL Triad Thrombocytopenia+Eczema+Immunodeficiency — Small Platelets PATHOGNOMONIC — WAS Score 1-5 Severity — Gene Therapy Trials — HSCT Curative',
  LRBA:   'LRBA Deficiency AR CVID+Autoimmunity+IBD+Organomegaly — Abatacept SPECIFIC Treatment — LRBA Recycles CTLA4 Endosomal — Western Blot LRBA Protein Absent — IVIG+Abatacept',
  CTLA4:  'CTLA4 Haploinsufficiency AD — CVID-Like Hypogammaglobulinemia+Lymphoproliferation+Granulomata — Abatacept SPECIFIC Replace CTLA4 Function — Sirolimus Lymphoproliferation — IVIG',
  PIK3CD: 'APDS1 AD GOF PI3K-delta Hyperactivation — Leniolisib FDA 2023 First Approved PI3Kdelta Inhibitor — EBV/CMV Herpesvirus Susceptibility — T-Cell Senescence — Idelalisib Trial',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /AVOID|CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|MISSES|EMERGENCY|NOT.Routine|OPPOSITE|PATHOGNOMONIC|LETHAL|FATAL|NEVER|ABSOLUTELY/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|PROTOCOL|STAT|FIRST|TRIAL|PHASE|ENROL|ELIGIBLE|Preferred|RESTRICTION|SCORE|Paradoxical|RECHECK|ARTIFACT|SPECIFIC/i.test(text);
  const bg = isCI ? '#b71c1c' : isWarning ? '#e65100' : '#1565c0';
  return (
    <div style={{
      background: bg, color: '#fff', borderRadius: 6, padding: '6px 12px',
      marginBottom: 8, fontSize: 13, lineHeight: 1.4,
    }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { aggregate_stats: s, top_alerts, genes } = data;

  const statRows = [
    ['Total patients', s.total_patients],
    ['Mean diagnostic delay (all genes)', `${s.mean_dx_delay_months} mo`],
    ['HSCT performed (any gene)', `${s.hsct_performed_pct}%`],
    ['IVIG given (any gene)', `${s.ivig_given_pct}%`],
    ['Live vaccines avoided', `${s.live_vaccine_avoided_pct}%`],
    ['Gene therapy given', `${s.gene_therapy_given_pct}%`],
    ['Antimicrobial prophylaxis', `${s.prophylaxis_given_pct}%`],
    // BTK
    ['BTK — IVIG prescribed lifelong', `${s.btk_ivig_pct}%`],
    ['BTK — live vaccine avoided', `${s.btk_live_vaccine_avoided_pct}%`],
    ['BTK — B cells absent on flow', `${s.btk_b_cells_absent_pct}%`],
    ['BTK — monocyte BTK flow performed', `${s.btk_monocyte_assay_pct}%`],
    ['BTK — bronchiectasis developed', `${s.btk_bronchiectasis_pct}%`],
    // RAG1
    ['RAG1 — HSCT performed', `${s.rag1_hsct_pct}%`],
    ['RAG1 — Omenn syndrome phenotype', `${s.rag1_omenn_pct}%`],
    ['RAG1 — TREC newborn screen positive', `${s.rag1_trec_pct}%`],
    ['RAG1 — BCG disease (disseminated)', `${s.rag1_bcg_disease_pct}%`],
    // ADA
    ['ADA — gene therapy (Strimvelis)', `${s.ada_gene_therapy_pct}%`],
    ['ADA — HSCT performed', `${s.ada_hsct_pct}%`],
    ['ADA — PEG-ADA enzyme replacement', `${s.ada_peg_ada_pct}%`],
    ['ADA — skeletal dysplasia on X-ray', `${s.ada_skeletal_pct}%`],
    ['ADA — dATP elevated (metabolite)', `${s.ada_datp_elevated_pct}%`],
    // CYBB
    ['CYBB — itraconazole+TMP-SMX prophylaxis', `${s.cybb_prophylaxis_pct}%`],
    ['CYBB — IFN-gamma prescribed', `${s.cybb_ifn_gamma_pct}%`],
    ['CYBB — Aspergillus infection', `${s.cybb_aspergillus_pct}%`],
    ['CYBB — DHR assay performed (diagnostic)', `${s.cybb_dhr_done_pct}%`],
    ['CYBB — HSCT performed', `${s.cybb_hsct_pct}%`],
    // WAS
    ['WAS — HSCT performed', `${s.was_hsct_pct}%`],
    ['WAS — splenectomy performed', `${s.was_splenectomy_pct}%`],
    ['WAS — small platelets (PATHOGNOMONIC)', `${s.was_small_platelets_pct}%`],
    ['WAS — eczema on presentation', `${s.was_eczema_pct}%`],
    ['WAS — intracranial haemorrhage', `${s.was_ich_pct}%`],
    ['WAS — autoimmunity', `${s.was_autoimmunity_pct}%`],
    // LRBA
    ['LRBA — abatacept prescribed', `${s.lrba_abatacept_pct}%`],
    ['LRBA — inflammatory bowel disease', `${s.lrba_ibd_pct}%`],
    ['LRBA — Western blot LRBA protein absent', `${s.lrba_western_blot_pct}%`],
    ['LRBA — organomegaly on presentation', `${s.lrba_organomegaly_pct}%`],
    ['LRBA — autoimmune cytopenia', `${s.lrba_autoimmune_cytopenia_pct}%`],
    // CTLA4
    ['CTLA4 — abatacept prescribed', `${s.ctla4_abatacept_pct}%`],
    ['CTLA4 — sirolimus for lymphoproliferation', `${s.ctla4_sirolimus_pct}%`],
    ['CTLA4 — lymphoproliferation', `${s.ctla4_lymphoproliferation_pct}%`],
    ['CTLA4 — granulomata', `${s.ctla4_granulomata_pct}%`],
    // PIK3CD
    ['PIK3CD — leniolisib (FDA 2023) prescribed', `${s.pik3cd_leniolisib_pct}%`],
    ['PIK3CD — EBV viraemia', `${s.pik3cd_ebv_viraemia_pct}%`],
    ['PIK3CD — CMV disease', `${s.pik3cd_cmv_disease_pct}%`],
    ['PIK3CD — T-cell senescence on flow', `${s.pik3cd_t_cell_senescence_pct}%`],
  ];

  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>
        {data.title || 'Hereditary Immunodeficiency Atlas'}
      </h2>
      <p style={{ color: '#555', marginBottom: 16 }}>
        {data.subtitle || '8 genes · 320 patients · seeds 1534–1541'}
      </p>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        {(genes || []).map(g => (
          <div key={g.gene} style={{
            background: GENE_COLORS[g.gene] || '#1565c0', color: '#fff', borderRadius: 8,
            padding: '10px 16px', minWidth: 120,
          }}>
            <div style={{ fontWeight: 700, fontSize: 15 }}>{g.gene}</div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>{g.locus} · {g.aa}</div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>{(g.inheritance || '').split('—')[0].trim()}</div>
            <div style={{ fontSize: 11, opacity: 0.9, marginTop: 4 }}>{g.n_patients} pts</div>
          </div>
        ))}
      </div>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Cohort Statistics</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, marginBottom: 24 }}>
        <tbody>
          {statRows.map(([label, val]) => (
            <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
              <td style={{ padding: '5px 8px', color: '#333' }}>{label}</td>
              <td style={{ padding: '5px 8px', fontWeight: 600, color: '#1565c0', textAlign: 'right' }}>{val}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>
        Critical Alerts ({(top_alerts || []).length})
      </h3>
      {(top_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;

  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Per-Gene Breakdown</h2>
      {breakdown.map(g => (
        <div key={g.gene} style={{
          border: `2px solid ${GENE_COLORS[g.gene] || '#1565c0'}`,
          borderRadius: 10, marginBottom: 24, overflow: 'hidden',
        }}>
          <div style={{
            background: GENE_COLORS[g.gene] || '#1565c0', color: '#fff',
            padding: '10px 16px',
          }}>
            <span style={{ fontWeight: 700, fontSize: 16 }}>{g.gene}</span>
            <span style={{ marginLeft: 12, fontSize: 13 }}>{g.protein}</span>
          </div>
          <div style={{ padding: 16 }}>
            <p style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>
              <strong>Locus:</strong> {g.locus} · <strong>Size:</strong> {g.aa} ({g.kDa}) ·
              <strong> OMIM gene:</strong> {g.omim_gene} · <strong>Disease:</strong> {g.omim_disease}
            </p>
            <p style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>
              <strong>Inheritance:</strong> {g.inheritance}
            </p>
            <p style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>
              <strong>Mean onset:</strong> {g.mean_onset_years} yr ·
              <strong> Mean dx delay:</strong> {g.mean_dx_delay_months} mo ·
              <strong> M:</strong> {g.sex_distribution?.M} / <strong>F:</strong> {g.sex_distribution?.F}
            </p>
            <p style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>{(g.alias || '').slice(0, 400)}…</p>
            <details style={{ marginTop: 8 }}>
              <summary style={{ cursor: 'pointer', fontSize: 12, color: '#1565c0', marginBottom: 6 }}>
                Gene-Class Mechanistic Detail
              </summary>
              <p style={{ fontSize: 12, color: '#444', marginTop: 6 }}>{g.gene_class}</p>
            </details>
            <div style={{ marginTop: 10 }}>
              <strong style={{ fontSize: 12 }}>Aetiology distribution:</strong>
              {Object.entries(g.etiology_counts || {}).map(([et, cnt]) => (
                <div key={et} style={{ fontSize: 11, color: '#555', marginTop: 2 }}>
                  {et}: <strong>{cnt}</strong>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 10 }}>
              {(g.key_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;

  const rows = breakdown.map(g => ({
    gene: g.gene,
    locus: g.locus,
    aa: g.aa,
    inh: (g.inheritance || '').split('—')[0].trim(),
    disease: GENE_DISEASE[g.gene] || '',
    pts: g.n_patients,
    onset: `${g.mean_onset_years}yr`,
  }));

  return (
    <div style={{ overflowX: 'auto' }}>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12 }}>Clinical Atlas Summary</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#1565c0', color: '#fff' }}>
            {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease / Key Rule', 'Pts', 'Onset'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.gene} style={{ background: i % 2 === 0 ? '#f8f8f8' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[r.gene] || '#1565c0' }}>{r.gene}</td>
              <td style={{ padding: '7px 10px' }}>{r.locus}</td>
              <td style={{ padding: '7px 10px' }}>{r.aa}</td>
              <td style={{ padding: '7px 10px' }}>{r.inh}</td>
              <td style={{ padding: '7px 10px', fontSize: 11 }}>{r.disease}</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{r.pts}</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{r.onset}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginTop: 24, marginBottom: 12 }}>
        Precision Treatment &amp; Investigation Matrix
      </h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#b71c1c', color: '#fff' }}>
            {['Gene', 'AVOID / Contraindicated', 'MANDATORY Investigation / Treatment', 'Special Rule'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[
            ['BTK', 'Live vaccines (BCG, MMR, OPV, varicella, yellow fever) ABSOLUTELY CONTRAINDICATED — fatal disseminated infection · Diagnostic delay from labelling as "recurrent infections" without immunoglobulin quantification · Isolating on Ig trough alone without chest CT for bronchiectasis', 'Monthly IVIG or SCIG LIFELONG (trough IgG >8 g/L) · B-cell enumeration by flow (absent CD19+ B cells) · Monocyte BTK protein flow cytometry to confirm diagnosis · Annual chest HRCT for bronchiectasis surveillance · Prophylactic antibiotics for Mycoplasma · Enteroviral surveillance (CNS viral meningitis risk)', 'BTK protein absent on monocyte flow (female carriers show bimodal) — fastest diagnostic confirmation; IVIG trough >8 g/L target to prevent bronchiectasis; Ibrutinib (BTK inhibitor for CLL) may RESTORE BTK protein production in some hypomorphic BTK variants — experimental'],
            ['RAG1', 'BCG vaccination (ABSOLUTELY CONTRAINDICATED — disseminated BCG disease fatal in SCID) · All live vaccines · Unirradiated, CMV-unscreened blood products · Missing Omenn syndrome (can mimic inflammatory/allergic conditions) · Delaying HSCT in complete SCID', 'TREC (T-cell receptor excision circle) newborn screening for early detection · Lymphocyte subsets (T, B, NK) — characterise immunophenotype · RAG1/RAG2 genetic panel · HSCT URGENTLY for complete SCID/Omenn · Irradiated, CMV-negative blood products ONLY · Strict reverse isolation until HSCT', 'RAG1 spectrum: null = complete SCID (no T/B/NK); hypomorphic = Omenn (maternal T cell expansion, erythroderma, lymphadenopathy, eosinophilia, elevated IgE) or combined immunodeficiency; TREC newborn screen detects early — do NOT wait for infections before diagnosis'],
            ['ADA', 'Live vaccines · Unirradiated blood products · PEG-ADA dose reduction without monitoring dATP levels · Missing skeletal dysplasia features (ADA-SCID has costochondral junction squaring, cupped ribs — unusual for an immunodeficiency)', 'dATP quantification in red blood cells (metabolite confirmation) · Skeletal radiograph (costochondral dysplasia) · HSCT (curative, preferred) OR Strimvelis gene therapy (EMA 2016 — autologous HSC gene therapy, EMA approved) OR PEG-ADA enzyme replacement (bridge to HSCT/gene therapy) · TREC newborn screen · Deoxyadenosine metabolite monitoring', 'Strimvelis (EMA 2016) was the FIRST approved gene therapy for a single-gene immune disorder — available as a curative option for ADA-SCID when HSCT not possible; PEG-ADA reverses metabolic block (dATP accumulation) but is not curative; skeletal dysplasia (cupped ribs, costochondral squaring) is PATHOGNOMONIC for ADA-SCID'],
            ['CYBB', 'Ignoring prophylactic antifungals / antibiotics in CGD (omission = fatal Aspergillus) · Delaying IFN-gamma therapy · Using oxidative bactericidal testing (nitro-blue tetrazolium NBT) without confirming by DHR flow · Missing catalase-positive organism susceptibility pattern', 'DHR (dihydrorhodamine) flow cytometry assay — gold standard diagnostic test for NADPH oxidase function · Itraconazole antifungal prophylaxis + TMP-SMX antibiotic prophylaxis LIFELONG · IFN-gamma (3x/week) reduces serious infections by 70% · HSCT for young severe cases · Genetic CYBB (XL gp91phox) panel + lyonisation testing in female carriers · CT chest/abdomen for granulomata', 'CYBB/CGD: NADPH oxidase complex = gp91phox (CYBB XL, 65%) + p22phox (CYBA, AR) + p47phox (NCF1, AR) + p67phox (NCF2, AR) + p40phox (NCF4, AR) — DHR assay tests ALL subtypes; DHR shows absent oxidative burst regardless of CGD type; catalase-positive organisms (Aspergillus, Staphylococcus, Burkholderia cepacia, Serratia, Nocardia) = signature susceptibility'],
            ['WAS', 'Splenectomy without HSCT evaluation (increases infection risk without cure) · Platelet transfusion for count alone without symptoms (small platelets function better than count suggests) · Missing WAS diagnosis in male with eczema + small platelets + infections', 'WAS gene sequencing + WAS protein western blot/flow · Platelet volume (MPV) — small platelets PATHOGNOMONIC · WAS clinical score 1-5 (determines urgency of HSCT) · HSCT curative for score ≥3 or autoimmunity/lymphoma · IVIG for hypogammaglobulinemia · Prophylactic TMP-SMX · Eczema management (topical) · WAS gene therapy (EAP trials, excellent results)', 'WAS score determines management: 1-2 (mild, infections only) = IVIG + surveillance; 3-5 (severe, autoimmunity/lymphoma risk) = HSCT URGENTLY; small platelet volume (MPV <7 fL) distinguishes WAS from ITP; WASp absent on flow in severe WAS; WAS gene therapy (Rocket trial, Orchard Therapeutics) shows curative potential like HSCT'],
            ['LRBA', 'Treating as "standard CVID" without testing for LRBA protein (misses specific abatacept therapy) · Missing IBD component (confused with Crohn\'s/UC) · Immunosuppressants that deplete T cells further (LRBA patients already have immune dysregulation)', 'LRBA Western blot protein expression (absent in LRBA deficiency — diagnostic) · Abatacept (CTLA4-Ig) — SPECIFIC treatment restoring CTLA4 surface expression — dramatic clinical response in organomegaly, IBD, autoimmunity · IVIG for hypogammaglobulinemia · Colonoscopy for IBD assessment · Screen siblings (AR, 25% recurrence)', 'LRBA stabilises intracellular CTLA4 transport: LRBA deficiency → CTLA4 misrouted to lysosomal degradation instead of recycled to surface → CTLA4 haploinsufficiency phenotype; abatacept (soluble CTLA4-Ig) BYPASSES the missing endosomal recycling — directly delivers CTLA4 function; distinguishable from CTLA4-HI by Western blot (LRBA protein absent) and AR vs AD inheritance'],
            ['CTLA4', 'Standard immunosuppression without specific abatacept consideration · Diagnosing as "seronegative autoimmune" without immunology referral · Missing the PID component (CVID-like hypogammaglobulinemia) in a patient with autoimmunity + lymphoproliferation', 'Abatacept (CTLA4-Ig fusion) — SPECIFIC treatment replacing CTLA4 haploinsufficiency · Sirolimus (mTOR inhibitor) for lymphoproliferation · IVIG for hypogammaglobulinemia · Lymphocyte subsets (CTLA4-HI: expanded effector T cells, low Tregs) · FDG-PET for lymphoproliferation/lymphoma surveillance · Family testing (AD, 50% inheritance risk)', 'CTLA4-HI: heterozygous CTLA4 loss-of-function → insufficient CTLA4 co-inhibition of T cell activation → autoreactive T cells + Treg dysfunction; CTLA4-Ig (abatacept, belatacept) provides the missing CTLA4 signal directly — response often dramatic; distinguish CTLA4-HI from LRBA deficiency by LRBA protein Western blot (normal in CTLA4-HI, absent in LRBA)'],
            ['PIK3CD', 'Missing EBV/CMV viraemia surveillance (leading cause of morbidity in APDS1) · Using idelalisib without monitoring for colitis/pneumonitis · Empiric immunosuppression that worsens T-cell senescence', 'Leniolisib (Joenja FDA 2023) — first specifically approved PI3Kδ inhibitor for APDS · EBV and CMV quantitative PCR monitoring (3-monthly) · CD8 T-cell senescence (CD57+CD28- population) on flow cytometry · IVIG for B-cell dysfunction/antibody failure · Anti-herpesvirus prophylaxis (valacyclovir) · mTOR inhibitor (rapamycin) as alternative/adjunct · HSCT as last resort', 'APDS1 (PIK3CD GOF) vs APDS2 (PIK3R1 LOF — regulatory subunit): both hyperactivate PI3Kδ; leniolisib (FDA 2023, Pharming) is approved for APDS (both types); PI3Kδ hyperactivation → constitutive AKT→S6K→mTOR signalling → T cell exhaustion, B cell class-switch failure, susceptibility to EBV/CMV herpesvirus → lymphoproliferation and lymphoma risk'],
          ].map(([gene, avoid, mandatory, special], i) => (
            <tr key={gene} style={{ background: i % 2 === 0 ? '#fce4ec' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#1565c0' }}>{gene}</td>
              <td style={{ padding: '7px 10px', color: '#b71c1c', fontWeight: 600 }}>{avoid}</td>
              <td style={{ padding: '7px 10px', color: '#1b5e20', fontWeight: 600 }}>{mandatory}</td>
              <td style={{ padding: '7px 10px', color: '#555' }}>{special}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.clinical_definitions || data.definitions || [];
  const crossDefs = data.cross_cutting_definitions || [];
  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Clinical Definitions</h2>
      {defs.map((d, i) => (
        <div key={i} style={{
          border: '1px solid #e0e0e0', borderRadius: 8,
          marginBottom: 16, padding: 16,
        }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#1565c0', marginBottom: 8 }}>
            {i + 1}. {d.term}
          </h3>
          <p style={{ fontSize: 13, color: '#444', lineHeight: 1.6 }}>{d.definition}</p>
        </div>
      ))}
      {crossDefs.length > 0 && (
        <>
          <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16, marginTop: 24 }}>Cross-Cutting PID Definitions</h2>
          {crossDefs.map((d, i) => (
            <div key={i} style={{
              border: '1px solid #bbdefb', borderRadius: 8,
              marginBottom: 16, padding: 16, background: '#f3f8ff',
            }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: '#1565c0', marginBottom: 8 }}>
                {i + 1}. {d.term}
              </h3>
              <p style={{ fontSize: 13, color: '#444', lineHeight: 1.6 }}>{d.definition}</p>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

export default function HereditaryImmunodeficiencyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-immunodeficiency-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if ((tab === 'Gene Table' || tab === 'Clinical Atlas') && !breakdown) {
      fetch(`${API}/api/hereditary-immunodeficiency-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if (tab === 'Definitions' && !definitions) {
      fetch(`${API}/api/hereditary-immunodeficiency-atlas/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [tab, breakdown, definitions]);

  return (
    <div style={{ padding: '1.5rem', fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 8 }}>
        <span style={{
          background: '#1565c0', color: '#fff', borderRadius: 6,
          padding: '4px 12px', fontSize: 12, fontWeight: 600,
        }}>
          Hereditary Immunodeficiency Atlas
        </span>
        <span style={{ marginLeft: 10, fontSize: 12, color: '#888' }}>
          8 genes · 320 patients · seeds 1534–1541
        </span>
      </div>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        Hereditary-Immunodeficiency-Atlas — Complete 8-Gene Hereditary Primary Immunodeficiency Reference
      </h1>
      <p style={{ fontSize: 13, color: '#666', marginBottom: 16 }}>
        BTK · RAG1 · ADA · CYBB · WAS · LRBA · CTLA4 · PIK3CD
        — XLA No Live Vaccines EVER, ADA-SCID Strimvelis Gene Therapy EMA 2016,
        CGD DHR Assay NADPH Oxidase, LRBA Abatacept SPECIFIC, Leniolisib FDA 2023 APDS
      </p>

      {error && (
        <div style={{ background: '#ffebee', border: '1px solid #ef9a9a', borderRadius: 6, padding: 12, marginBottom: 16 }}>
          Error: {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === t ? '#1565c0' : '#f0f0f0',
              color: tab === t ? '#fff' : '#333',
              fontWeight: tab === t ? 700 : 400,
              fontSize: 13,
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
