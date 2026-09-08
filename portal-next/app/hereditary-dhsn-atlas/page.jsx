'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-dhsn-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  HSPB1:   '#1565c0',  // deep blue    — dHMN2B adult-onset foot drop
  HSPB8:   '#2e7d32',  // deep green   — dHMN2A juvenile onset
  GARS1:   '#6a1b9a',  // deep purple  — dHMN5A upper-limb predominant
  DCTN1:   '#b71c1c',  // deep red     — dHMN7B vocal fold paralysis
  BICD2:   '#00695c',  // deep teal    — SMALED1 congenital contractures
  DYNC1H1: '#e65100',  // deep orange  — SMALED1 + intellectual disability
  IGHMBP2: '#880e4f',  // deep pink    — SMARD1 infantile respiratory failure
  TRPV4:   '#37474f',  // dark slate   — SMALED2 skeletal dysplasia
};

const GENE_INFO = {
  HSPB1:   { full: 'HSPB1 / HSP27 640aa',          locus: '7q11.23',  size: '640 aa',   inh: 'AD',       disease: 'dHMN2B — Foot Drop Steppage Gait / Intrinsic Hand Wasting / NO Sensory Loss DDx-CMT2F / Slow Progression' },
  HSPB8:   { full: 'HSPB8 / HSP22 196aa',          locus: '12q24.23', size: '196 aa',   inh: 'AD',       disease: 'dHMN2A — Juvenile Onset 10-25yr / pLys141Asn European Founder / Allelic CMT2L / Slow Progression' },
  GARS1:   { full: 'GARS1 / GlyRS 685aa',          locus: '7p14.3',   size: '685 aa',   inh: 'AD',       disease: 'dHMN5A/CMT2D — UPPER LIMB PREDOMINANT Thenar/Interosseous Wasting PATHOGNOMONIC / Phrenic Nerve 20-30%' },
  DCTN1:   { full: 'DCTN1 / p150glued 1278aa',     locus: '2p13.1',   size: '1278 aa',  inh: 'AD',       disease: 'dHMN7B — VOCAL FOLD PARALYSIS PATHOGNOMONIC / Laryngoscopy MANDATORY / pGly59Ser Founder / ALS14 severe' },
  BICD2:   { full: 'BICD2 / BicD2 820aa',          locus: '9q22.31',  size: '820 aa',   inh: 'AD/deNovo', disease: 'SMALED1 — CONGENITAL CONTRACTURES + HIP DISLOCATION PATHOGNOMONIC / Normal IQ / De Novo 40%' },
  DYNC1H1: { full: 'DYNC1H1 / Dynein-HC 4646aa',   locus: '14q32.31', size: '4646 aa',  inh: 'AD/deNovo', disease: 'SMALED1 — Lower Extremity SMA + INTELLECTUAL DISABILITY 30% PATHOGNOMONIC / Pachygyria / De Novo 60%' },
  IGHMBP2: { full: 'IGHMBP2 / RNA-Helicase 993aa', locus: '11q13.3',  size: '993 aa',   inh: 'AR',       disease: 'SMARD1/dHMN6 — INFANTILE RESPIRATORY FAILURE PATHOGNOMONIC / Diaphragm Palsy / NOT SMA1 / NIV MANDATORY' },
  TRPV4:   { full: 'TRPV4 / TRP-V4 871aa',         locus: '12q24.11', size: '871 aa',   inh: 'AD',       disease: 'SMALED2/CMT2C — SKELETAL DYSPLASIA + Motor Neuropathy PATHOGNOMONIC / Vocal Fold Palsy / pArg269His' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('ABSOLUTE') ? '#880e4f'
    : flag.includes('TREATABLE') || flag.includes('NIV') || flag.includes('AFO') ? '#2e7d32'
    : flag.includes('MISS') || flag.includes('NOT-SMA') ? '#b71c1c'
    : flag.includes('FOUNDER') || flag.includes('EUROPEAN') ? '#00695c'
    : flag.includes('FATAL') || flag.includes('RESPIRATORY') ? '#e65100'
    : flag.includes('DE-NOVO') || flag.includes('DDx') ? '#4a148c'
    : flag.includes('CONGENITAL') || flag.includes('INFANTILE') ? '#0d47a1'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryDHMNAtlasPage() {
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
        🧬 Hereditary dHMN Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Distal Hereditary Motor Neuropathy Atlas —
        HSPB1/HSP27 (dHMN2B/Adult-Foot-Drop) · HSPB8/HSP22 (dHMN2A/Juvenile) · GARS1 (dHMN5A/Upper-Limb-Predominant) · DCTN1 (dHMN7B/Vocal-Fold-Palsy) · BICD2 (SMALED1/Congenital) · DYNC1H1 (SMALED1/Intellect) · IGHMBP2 (SMARD1/Respiratory) · TRPV4 (SMALED2/Skeletal)
        &nbsp;|&nbsp; 320 patients · seeds 2134-2141
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: activeTab === tab ? '#1565c0' : '#e3f2fd',
            color: activeTab === tab ? '#fff' : '#1565c0', fontWeight: 600,
          }}>{tab}</button>
        ))}
      </div>

      {loading && <p style={{ color: '#888' }}>Loading atlas data…</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* ── OVERVIEW ── */}
      {activeTab === 'Overview' && overview && (
        <div>
          <h2 style={{ color: '#1565c0' }}>Atlas Overview — 320 Patients · 8 Genes · Seeds 2134-2141</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: 16 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#1565c0' },
              { label: 'All Foot Drop', value: overview.all_foot_drop_patients, color: '#1565c0' },
              { label: 'All Respiratory Failure', value: overview.all_respiratory_failure_patients, color: '#b71c1c' },
              { label: 'All Vocal Fold Palsy', value: overview.all_vocal_fold_palsy_patients, color: '#b71c1c' },
              { label: 'All Wheelchair', value: overview.all_wheelchair_patients, color: '#e65100' },
              { label: 'All Congenital Onset', value: overview.all_congenital_patients, color: '#00695c' },
              { label: 'GARS1 Upper-Limb Predominant', value: overview.gars1_upper_limb_predominant_patients, color: '#6a1b9a' },
              { label: 'GARS1 Respiratory Failure', value: overview.gars1_respiratory_failure_patients, color: '#6a1b9a' },
              { label: 'DCTN1 Vocal Fold Palsy', value: overview.dctn1_vocal_fold_palsy_patients, color: '#b71c1c' },
              { label: 'BICD2 Hip Dislocation', value: overview.bicd2_hip_dislocation_patients, color: '#00695c' },
              { label: 'DYNC1H1 Intellectual Disability', value: overview.dync1h1_intellectual_disability_patients, color: '#e65100' },
              { label: 'IGHMBP2 Diaphragm Palsy', value: overview.ighmbp2_diaphragm_palsy_patients, color: '#880e4f' },
              { label: 'TRPV4 Skeletal Dysplasia', value: overview.trpv4_skeletal_dysplasia_patients, color: '#37474f' },
              { label: 'TRPV4 Hearing Loss', value: overview.trpv4_hearing_loss_patients, color: '#37474f' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: 16, borderLeft: `4px solid ${color}` }}>
                <div style={{ fontSize: 13, color: '#777' }}>{label}</div>
                <div style={{ fontSize: 28, fontWeight: 700, color }}>{value}</div>
                <div style={{ fontSize: 12, color: '#999' }}>of 320 patients</div>
              </div>
            ))}
          </div>

          <h3 style={{ marginTop: 32, color: '#1565c0' }}>8 Genes at a Glance</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 12 }}>
            {Object.entries(GENE_INFO).map(([gene, info]) => (
              <div key={gene} style={{
                background: '#fff', border: `2px solid ${GENE_COLORS[gene]}`,
                borderRadius: 8, padding: 14,
              }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gene], fontSize: 15 }}>{info.full}</div>
                <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{info.locus} · {info.size} · {info.inh}</div>
                <div style={{ fontSize: 12, color: '#333', marginTop: 6 }}>{info.disease}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {activeTab === 'Gene Table' && breakdown && (
        <div>
          <h2 style={{ color: '#1565c0' }}>Per-Gene Clinical Breakdown</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#e3f2fd' }}>
                  {['Gene','Locus','Size','Inh.','Onset (yr)','Foot Drop %','Hand Wasting %','Sensory %','Upper-Limb %','Vocal Fold %','Resp Failure %','Congenital Contractures %','Intellect Dis %','Skeletal %','Wheelchair %'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #1565c0', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(breakdown).map(([gene, data], i) => (
                  <tr key={gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                    <td style={{ padding: '8px 10px' }}>{data.locus}</td>
                    <td style={{ padding: '8px 10px' }}>{data.protein_size}</td>
                    <td style={{ padding: '8px 10px' }}>{data.inheritance?.split(';')[0]}</td>
                    <td style={{ padding: '8px 10px' }}>{data.mean_onset_age}</td>
                    <td style={{ padding: '8px 10px' }}>{data.foot_drop_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{data.hand_wasting_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{data.sensory_loss_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{data.upper_limb_predominant_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{data.vocal_fold_palsy_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{data.respiratory_failure_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{data.congenital_contractures_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{data.intellectual_disability_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{data.skeletal_dysplasia_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{data.wheelchair_pct}%</td>
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
          <h2 style={{ color: '#1565c0' }}>Clinical Atlas — Pathognomonic Features, Treatments & Critical Flags</h2>
          {Object.entries(breakdown).map(([gene, data]) => (
            <div key={gene} style={{
              background: '#fff', border: `2px solid ${GENE_COLORS[gene]}`,
              borderRadius: 8, padding: 20, marginBottom: 20,
            }}>
              <h3 style={{ color: GENE_COLORS[gene], marginTop: 0 }}>
                {gene} — {GENE_INFO[gene]?.disease?.split(' — ')[0]}
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <strong>Pathognomonic:</strong>
                  <p style={{ fontSize: 13, color: '#444', marginTop: 4 }}>{data.pathognomonic}</p>
                </div>
                <div>
                  <strong>Treatment:</strong>
                  <p style={{ fontSize: 13, color: '#444', marginTop: 4 }}>{data.treatment}</p>
                </div>
                <div>
                  <strong>Key Biomarker:</strong>
                  <p style={{ fontSize: 13, color: '#444', marginTop: 4 }}>{data.key_biomarker}</p>
                </div>
                <div>
                  <strong>Inheritance:</strong>
                  <p style={{ fontSize: 13, color: '#444', marginTop: 4 }}>{data.inheritance}</p>
                </div>
              </div>
              <div style={{ marginTop: 10 }}>
                <strong>Critical Flags:</strong>
                <div style={{ marginTop: 6 }}>
                  {(data.critical_flags || []).map(f => <FLAG_BADGE key={f} flag={f} />)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {activeTab === 'Definitions' && definitions && (
        <div>
          <h2 style={{ color: '#1565c0' }}>Gene Definitions & Glossary</h2>
          <h3>Gene Proteins</h3>
          {Object.entries(definitions.genes || {}).map(([gene, desc]) => (
            <div key={gene} style={{
              background: '#f5f5f5', borderRadius: 6, padding: 12, marginBottom: 10,
              borderLeft: `4px solid ${GENE_COLORS[gene] || '#1565c0'}`,
            }}>
              <strong style={{ color: GENE_COLORS[gene] || '#1565c0' }}>{gene}</strong>
              <p style={{ fontSize: 12, color: '#555', marginTop: 4 }}>{desc}</p>
            </div>
          ))}
          <h3 style={{ marginTop: 24 }}>Glossary</h3>
          {Object.entries(definitions.glossary || {}).map(([term, text]) => (
            <div key={term} style={{ marginBottom: 16 }}>
              <strong style={{ color: '#1565c0' }}>{term}</strong>
              <p style={{ fontSize: 13, color: '#444', marginTop: 4 }}>{text}</p>
            </div>
          ))}
          <h3 style={{ marginTop: 24 }}>Surveillance Protocols</h3>
          {Object.entries(definitions.surveillance_protocols || {}).map(([gene, protocol]) => (
            <div key={gene} style={{
              background: '#e3f2fd', borderRadius: 6, padding: 12, marginBottom: 10,
            }}>
              <strong style={{ color: '#1565c0' }}>{gene}</strong>
              <p style={{ fontSize: 12, color: '#444', marginTop: 4, whiteSpace: 'pre-wrap' }}>{protocol}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
