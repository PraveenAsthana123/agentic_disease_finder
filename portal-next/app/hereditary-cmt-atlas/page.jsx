'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cmt-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PMP22:  '#1565c0',  // deep blue    — CMT1A most common / HNPP
  MPZ:    '#2e7d32',  // deep green   — CMT1B multiple phenotypes
  GJB1:   '#6a1b9a',  // deep purple  — CMTX1 X-linked no male-to-male
  MFN2:   '#b71c1c',  // deep red     — CMT2A optic atrophy
  SH3TC2: '#00695c',  // deep teal    — CMT4C early scoliosis
  GDAP1:  '#e65100',  // deep orange  — CMT4A vocal cord palsy
  NEFL:   '#880e4f',  // deep pink    — CMT2E giant axons
  PRX:    '#37474f',  // dark slate   — CMT4F focally folded myelin
};

const GENE_INFO = {
  PMP22:  { full: 'PMP22 / PMP22-160aa', locus: '17p12',   size: '160 aa',   inh: 'AD (CNV)',  disease: 'CMT1A (dup) / HNPP (del) — UNIFORM NCV <38 m/s ALL NERVES PATHOGNOMONIC / MLPA MANDATORY / Most Common CMT 1:2500' },
  MPZ:    { full: 'MPZ / P0 248aa',      locus: '1q23.3',  size: '248 aa',   inh: 'AD',        disease: 'CMT1B — pThr124Met Late-Onset DISTINCTIVE / pSer44Phe Dejerine-Sottas Congenital Severe / Onion Bulbs Biopsy' },
  GJB1:   { full: 'GJB1 / Cx32 283aa',  locus: 'Xq13.1',  size: '283 aa',   inh: 'XLD',       disease: 'CMTX1 — NO Male-to-Male Transmission PATHOGNOMONIC / CNS WM Lesions Transient Fever PATHOGNOMONIC / Intermediate NCV' },
  MFN2:   { full: 'MFN2 / MFN2 741aa',  locus: '1p36.22', size: '741 aa',   inh: 'AD/deNovo', disease: 'CMT2A — OPTIC ATROPHY 20-30% PATHOGNOMONIC / Most Severe CMT2 Early Wheelchair / De Novo 30-40%' },
  SH3TC2: { full: 'SH3TC2 1288aa',      locus: '5q32',    size: '1288 aa',  inh: 'AR',        disease: 'CMT4C — EARLY SCOLIOSIS 50-70% PATHOGNOMONIC / Hearing Loss 40-50% / Most Common AR CMT Turkey/Pakistan/India' },
  GDAP1:  { full: 'GDAP1 358aa',        locus: '8q21.11', size: '358 aa',   inh: 'AR/AD',     disease: 'CMT4A (AR) / CMT2K (AD) — VOCAL CORD PARALYSIS 20-30% PATHOGNOMONIC AR / Laryngoscopy MANDATORY / N Africa Founders' },
  NEFL:   { full: 'NEFL / NF-L 543aa',  locus: '8p21.2',  size: '543 aa',   inh: 'AD/AR',     disease: 'CMT2E (AD) / CMT1F (AR) — GIANT AXONS Nerve Biopsy PATHOGNOMONIC / CSF Serum NF-L Elevated Biomarker' },
  PRX:    { full: 'PRX / Periaxin 1461aa', locus: '19q13.13', size: '1461 aa', inh: 'AR',     disease: 'CMT4F — FOCALLY FOLDED MYELIN Nerve Biopsy PATHOGNOMONIC / Sensory > Motor DISTINCTIVE / Romani/Pakistani Founders' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('ABSOLUTE') ? '#880e4f'
    : flag.includes('FIRST') || flag.includes('AFO') || flag.includes('TREATABLE') ? '#2e7d32'
    : flag.includes('MISS') || flag.includes('FAILED') ? '#c62828'
    : flag.includes('FOUNDER') || flag.includes('DISTINCTIVE') ? '#00695c'
    : flag.includes('FATAL') || flag.includes('RESPIRATORY') ? '#e65100'
    : flag.includes('DE-NOVO') || flag.includes('DDx') ? '#4a148c'
    : flag.includes('MLPA') || flag.includes('CNV') ? '#0d47a1'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditoryCMTAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, def]) => { setOverview(ov); setBreakdown(br); setDefinitions(def); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ color: '#1565c0', marginBottom: 4 }}>
        🧬 Hereditary CMT Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Charcot-Marie-Tooth Disease Atlas —
        PMP22 (CMT1A-dup/HNPP-del) · MPZ (CMT1B) · GJB1/Cx32 (CMTX1) · MFN2 (CMT2A/Optic-Atrophy) · SH3TC2 (CMT4C/Scoliosis) · GDAP1 (CMT4A/Vocal-Cord) · NEFL (CMT2E/Giant-Axons) · PRX (CMT4F/Focally-Folded-Myelin)
        &nbsp;|&nbsp; 320 patients · seeds 2142-2149
      </p>

      {/* Tab Bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '6px 6px 0 0', cursor: 'pointer',
            background: activeTab === t ? '#1565c0' : '#f5f5f5',
            color: activeTab === t ? '#fff' : '#333',
            fontWeight: activeTab === t ? 700 : 400, fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      {loading && <p style={{ color: '#888' }}>Loading CMT atlas data…</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* ── OVERVIEW ── */}
      {activeTab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 14, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#1565c0' },
              { label: 'Genes Covered', value: overview.genes?.length || 8, color: '#2e7d32' },
              { label: 'Seeds', value: overview.seeds, color: '#6a1b9a' },
              { label: 'Foot Drop', value: overview.all_foot_drop_patients, color: '#b71c1c' },
              { label: 'Demyelinating NCV', value: overview.all_demyelinating_patients, color: '#00695c' },
              { label: 'Wheelchair Users', value: overview.all_wheelchair_patients, color: '#e65100' },
              { label: 'Scoliosis', value: overview.all_scoliosis_patients, color: '#880e4f' },
              { label: 'Sensory Loss', value: overview.all_sensory_loss_patients, color: '#37474f' },
            ].map(kpi => (
              <div key={kpi.label} style={{
                background: '#fff', borderRadius: 8, padding: 16,
                boxShadow: '0 1px 4px rgba(0,0,0,0.12)', borderTop: `4px solid ${kpi.color}`,
              }}>
                <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>{kpi.label}</div>
                <div style={{ fontSize: 26, fontWeight: 700, color: kpi.color }}>{kpi.value}</div>
              </div>
            ))}
          </div>

          {/* Gene summary grid */}
          <h3 style={{ color: '#333', marginBottom: 10 }}>Gene-Level Highlights</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(280px,1fr))', gap: 14 }}>
            {[
              { gene: 'PMP22', label: 'Uniform NCV <38 m/s / HNPP Pressure Palsies', kpi: `Demyelinating: ${overview.pmp22_demyelinating_patients}/40`, kpi2: `Foot drop: ${overview.pmp22_foot_drop_patients}/40` },
              { gene: 'MPZ',   label: 'Multiple Phenotypes / pThr124Met Late-Onset', kpi: `Demyelinating: ${overview.mpz_demyelinating_patients}/40`, kpi2: `Scoliosis: ${overview.mpz_scoliosis_patients}/40` },
              { gene: 'GJB1',  label: 'No M→M Transmission / CNS WM Lesions (Fever)', kpi: `CNS WM lesions: ${overview.gjb1_cnx_wm_lesion_patients}/40`, kpi2: `X-linked: ${overview.gjb1_x_linked_patients}/40` },
              { gene: 'MFN2',  label: 'Optic Atrophy PATHOGNOMONIC (20-30%)', kpi: `Optic atrophy: ${overview.mfn2_optic_atrophy_patients}/40`, kpi2: `Wheelchair: ${overview.mfn2_wheelchair_patients}/40` },
              { gene: 'SH3TC2',label: 'Early Scoliosis PATHOGNOMONIC / Hearing Loss', kpi: `Scoliosis: ${overview.sh3tc2_scoliosis_patients}/40`, kpi2: `Hearing loss: ${overview.sh3tc2_hearing_loss_patients}/40` },
              { gene: 'GDAP1', label: 'Vocal Cord Palsy PATHOGNOMONIC (AR)', kpi: `Vocal fold palsy: ${overview.gdap1_vocal_fold_palsy_patients}/40`, kpi2: `Resp failure: ${overview.gdap1_respiratory_failure_patients}/40` },
              { gene: 'NEFL',  label: 'Giant Axons Nerve Biopsy PATHOGNOMONIC', kpi: `Giant axons: ${overview.nefl_giant_axons_patients}/40`, kpi2: `Wheelchair: ${overview.nefl_wheelchair_patients}/40` },
              { gene: 'PRX',   label: 'Focally Folded Myelin PATHOGNOMONIC / Sensory>Motor', kpi: `Focal myelin: ${overview.prx_focal_myelin_patients}/40`, kpi2: `Sensory predominant: ${overview.prx_sensory_predominant_patients}/40` },
            ].map(g => (
              <div key={g.gene} style={{
                background: '#fff', borderRadius: 8, padding: 14,
                borderLeft: `5px solid ${GENE_COLORS[g.gene]}`,
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene], marginBottom: 4 }}>{g.gene}</div>
                <div style={{ fontSize: 12, color: '#555', marginBottom: 6 }}>{g.label}</div>
                <div style={{ fontSize: 12, color: '#333' }}>{g.kpi}</div>
                <div style={{ fontSize: 12, color: '#333' }}>{g.kpi2}</div>
              </div>
            ))}
          </div>

          {/* Cross-atlas summary */}
          <div style={{ marginTop: 20, background: '#e3f2fd', borderRadius: 8, padding: 16 }}>
            <h4 style={{ color: '#1565c0', margin: '0 0 8px 0' }}>Cross-Atlas Summary (320 Patients)</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10 }}>
              {[
                { k: 'Optic Atrophy', v: overview.all_optic_atrophy_patients },
                { k: 'Hearing Loss', v: overview.all_hearing_loss_patients },
                { k: 'Vocal Fold Palsy', v: overview.all_vocal_fold_palsy_patients },
                { k: 'Scoliosis', v: overview.all_scoliosis_patients },
              ].map(x => (
                <div key={x.k} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#1565c0' }}>{x.v}</div>
                  <div style={{ fontSize: 11, color: '#555' }}>{x.k}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {activeTab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1565c0', color: '#fff' }}>
                {['Gene', 'Protein / Size', 'Locus', 'Inheritance', 'Disease / Key Features'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(GENE_INFO).map(([gene, info], i) => (
                <tr key={gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                  <td style={{ padding: '9px 12px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                  <td style={{ padding: '9px 12px' }}>{info.full}</td>
                  <td style={{ padding: '9px 12px', fontFamily: 'monospace' }}>{info.locus}</td>
                  <td style={{ padding: '9px 12px' }}>
                    <span style={{
                      background: info.inh === 'AR' ? '#e65100' : info.inh === 'XLD' ? '#6a1b9a' : info.inh.includes('deNovo') ? '#880e4f' : '#1565c0',
                      color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11,
                    }}>{info.inh}</span>
                  </td>
                  <td style={{ padding: '9px 12px', fontSize: 12, maxWidth: 420 }}>{info.disease}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* NCS Classification guide */}
          <div style={{ marginTop: 20, background: '#fff3e0', borderRadius: 8, padding: 16 }}>
            <h4 style={{ color: '#e65100', margin: '0 0 8px 0' }}>NCS-Based CMT Gene Prioritisation</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#e65100', color: '#fff' }}>
                  {['NCS Pattern', 'NCV Range', 'First Gene', 'Then Test'].map(h => (
                    <th key={h} style={{ padding: '7px 10px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  { pattern: 'Uniform demyelinating', ncv: '<38 m/s (all nerves)', first: 'PMP22 MLPA', then: 'MPZ → GJB1 → SH3TC2 → GDAP1 → NEFL' },
                  { pattern: 'Axonal', ncv: 'Normal NCV, low CMAP', first: 'MFN2', then: 'NEFL → GDAP1 (AD CMT2K) → others' },
                  { pattern: 'Intermediate (males)', ncv: '25-45 m/s', first: 'GJB1 (CMTX1)', then: 'Check pedigree: no M→M = X-linked confirmed' },
                  { pattern: 'Severe demyelinating AR', ncv: '<10-15 m/s', first: 'SH3TC2 / GDAP1 / PRX', then: 'Ethnicity guides order: Turkish → SH3TC2; N African → GDAP1; Romani → PRX' },
                ].map((row, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff8f0' : '#fff' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 600 }}>{row.pattern}</td>
                    <td style={{ padding: '7px 10px', fontFamily: 'monospace' }}>{row.ncv}</td>
                    <td style={{ padding: '7px 10px', color: '#1565c0', fontWeight: 600 }}>{row.first}</td>
                    <td style={{ padding: '7px 10px', color: '#555' }}>{row.then}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {activeTab === 'Clinical Atlas' && breakdown && (
        <div>
          {Object.entries(breakdown).map(([gene, data]) => (
            <div key={gene} style={{
              background: '#fff', borderRadius: 10, padding: 20, marginBottom: 20,
              borderLeft: `6px solid ${GENE_COLORS[gene]}`,
              boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                <div>
                  <h3 style={{ color: GENE_COLORS[gene], margin: '0 0 4px 0' }}>
                    {gene} — {GENE_INFO[gene]?.full}
                  </h3>
                  <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>
                    <strong>Locus:</strong> {data.locus} &nbsp;|&nbsp;
                    <strong>Inheritance:</strong> {data.inheritance?.split(';')[0]} &nbsp;|&nbsp;
                    <strong>Onset:</strong> {data.age_of_onset}
                  </div>
                </div>
                <div style={{ fontSize: 13, color: '#555' }}>
                  <strong>n = {data.patient_count}</strong> | mean onset {data.mean_onset_age} yr
                </div>
              </div>

              {/* Stats bar */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(130px,1fr))', gap: 8, marginBottom: 12 }}>
                {[
                  { label: 'Demyelinating', pct: data.demyelinating_pct },
                  { label: 'Foot Drop', pct: data.foot_drop_pct },
                  { label: 'Hand Wasting', pct: data.hand_wasting_pct },
                  { label: 'Sensory Loss', pct: data.sensory_loss_pct },
                  { label: 'Scoliosis', pct: data.scoliosis_pct },
                  { label: 'Hearing Loss', pct: data.hearing_loss_pct },
                  { label: 'Optic Atrophy', pct: data.optic_atrophy_pct },
                  { label: 'Vocal Fold Palsy', pct: data.vocal_fold_palsy_pct },
                  { label: 'CNS WM Lesions', pct: data.cnx_wm_lesion_pct },
                  { label: 'Giant Axons', pct: data.giant_axons_pct },
                  { label: 'Focal Myelin', pct: data.focal_myelin_pct },
                  { label: 'Wheelchair', pct: data.wheelchair_pct },
                ].filter(s => s.pct > 0).map(stat => (
                  <div key={stat.label} style={{
                    background: '#f5f5f5', borderRadius: 6, padding: '8px 10px', textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[gene] }}>{stat.pct}%</div>
                    <div style={{ fontSize: 11, color: '#666' }}>{stat.label}</div>
                  </div>
                ))}
              </div>

              {/* Pathognomonic */}
              <div style={{ background: '#ffebee', borderRadius: 6, padding: '8px 12px', marginBottom: 10 }}>
                <strong style={{ color: '#b71c1c', fontSize: 12 }}>⚠ PATHOGNOMONIC / KEY CLINICAL FEATURES:</strong>
                <p style={{ margin: '4px 0 0 0', fontSize: 12, color: '#333' }}>{data.pathognomonic}</p>
              </div>

              {/* Treatment */}
              <div style={{ background: '#e8f5e9', borderRadius: 6, padding: '8px 12px', marginBottom: 10 }}>
                <strong style={{ color: '#2e7d32', fontSize: 12 }}>💊 MANAGEMENT:</strong>
                <p style={{ margin: '4px 0 0 0', fontSize: 12, color: '#333' }}>{data.treatment}</p>
              </div>

              {/* Biomarker */}
              <div style={{ background: '#e3f2fd', borderRadius: 6, padding: '8px 12px', marginBottom: 10 }}>
                <strong style={{ color: '#1565c0', fontSize: 12 }}>🔬 KEY BIOMARKERS / NCS / GENETICS:</strong>
                <p style={{ margin: '4px 0 0 0', fontSize: 12, color: '#333' }}>{data.key_biomarker}</p>
              </div>

              {/* Critical flags */}
              <div>
                <strong style={{ fontSize: 12, color: '#444' }}>🚩 CRITICAL FLAGS:</strong>
                <div style={{ marginTop: 6 }}>
                  {(data.critical_flags || []).map(flag => <FLAG_BADGE key={flag} flag={flag} />)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {activeTab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ color: '#333', marginBottom: 12 }}>Gene Definitions</h3>
          <div style={{ marginBottom: 24 }}>
            {Object.entries(definitions.genes || {}).map(([gene, desc]) => (
              <div key={gene} style={{
                background: '#fff', borderRadius: 8, padding: 14, marginBottom: 10,
                borderLeft: `5px solid ${GENE_COLORS[gene] || '#37474f'}`,
                boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
              }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#333', marginBottom: 6 }}>{gene}</div>
                <div style={{ fontSize: 12, color: '#444', lineHeight: 1.6 }}>
                  {String(desc).split(' -- ').map((part, i) => (
                    <span key={i} style={{ marginRight: 6 }}>
                      {part}{i < String(desc).split(' -- ').length - 1 ? ' ·' : ''}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#333', marginBottom: 12 }}>CMT Glossary & Classification</h3>
          <div style={{ marginBottom: 24 }}>
            {Object.entries(definitions.glossary || {}).map(([term, def]) => (
              <div key={term} style={{
                background: '#f9f9f9', borderRadius: 8, padding: 14, marginBottom: 10,
                borderLeft: '4px solid #1565c0',
              }}>
                <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 6, fontSize: 14 }}>{term}</div>
                <div style={{ fontSize: 13, color: '#444', lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#333', marginBottom: 12 }}>Clinical Pearls</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {(definitions.clinical_pearls || []).map(pearl => (
              <FLAG_BADGE key={pearl} flag={pearl} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
