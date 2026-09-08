'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-brain-malformation-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PAFAH1B1: '#1565c0',  // deep blue    — LIS1 lissencephaly posterior>anterior Miller-Dieker
  DCX:      '#2e7d32',  // deep green   — double cortex females band heterotopia
  TUBA1A:   '#b71c1c',  // deep red     — pachygyria cerebellar hypoplasia basal ganglia
  FLNA:     '#880e4f',  // deep magenta — PVNH bilateral normal IQ females lethal males
  ASPM:     '#e65100',  // deep orange  — MCPH5 most common AR microcephaly -7 to -10 SD
  CDK5RAP2: '#6a1b9a',  // deep purple  — MCPH3 milder microcephaly centrosomal
  ADGRG1:   '#004d40',  // deep teal    — BFPP strabismus hypotonia dysarthria
  RELN:     '#37474f',  // dark slate   — AR LCH cerebellar agenesis pathognomonic QT AD
};

const GENE_INFO = {
  PAFAH1B1: { full: 'LIS1 / PAFAH1B1', locus: '17p13.3', size: '380 aa', inh: 'AD de novo', disease: 'Lissencephaly Type 1 / Miller-Dieker Syndrome' },
  DCX:      { full: 'Doublecortin', locus: 'Xq22.3', size: '360 aa', inh: 'X-linked', disease: 'Lissencephaly (males) / Band Heterotopia (females)' },
  TUBA1A:   { full: 'Tubulin Alpha-1A', locus: '12q13.12', size: '451 aa', inh: 'AD de novo', disease: 'Pachygyria + Cerebellar Hypoplasia + CC Dysgenesis' },
  FLNA:     { full: 'Filamin A', locus: 'Xq28', size: '2647 aa', inh: 'XL dominant', disease: 'Bilateral PVNH — Epilepsy / Normal IQ Females' },
  ASPM:     { full: 'ASPM', locus: '1q31.3', size: '3477 aa', inh: 'AR', disease: 'Primary Microcephaly MCPH5 — OFC -7 to -10 SD' },
  CDK5RAP2: { full: 'CDK5RAP2', locus: '9q33.2', size: '1893 aa', inh: 'AR', disease: 'Primary Microcephaly MCPH3 — milder than ASPM' },
  ADGRG1:   { full: 'ADGRG1 / GPR56', locus: '16q21', size: '693 aa', inh: 'AR', disease: 'Bilateral Frontoparietal Polymicrogyria (BFPP)' },
  RELN:     { full: 'Reelin', locus: '7q22.1', size: '3461 aa', inh: 'AR/AD', disease: 'LCH (AR) / Autism+LQTS (AD)' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('LETHAL') ? '#880e4f'
    : flag.includes('DISTINGUISH') ? '#e65100'
    : flag.includes('AR') || flag.includes('XL') ? '#1565c0'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryBrainMalformationAtlasPage() {
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
        🧠 Hereditary Brain Malformation Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Atlas — Lissencephaly · Band Heterotopia · Pachygyria · PVNH · Polymicrogyria · Primary Microcephaly
        &nbsp;|&nbsp; PAFAH1B1 · DCX · TUBA1A · FLNA · ASPM · CDK5RAP2 · ADGRG1 · RELN
        &nbsp;|&nbsp; 320 patients · seeds 2046-2053
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: activeTab === t ? '#1565c0' : '#e3f2fd',
            color: activeTab === t ? '#fff' : '#1565c0', fontWeight: 600,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#888' }}>Loading atlas data…</div>}
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}

      {/* Overview Tab */}
      {activeTab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#1565c0' },
              { label: 'Lissencephaly', value: overview.lissencephaly_patients, color: '#b71c1c' },
              { label: 'Polymicrogyria (BFPP)', value: overview.polymicrogyria_patients, color: '#2e7d32' },
              { label: 'Primary Microcephaly', value: overview.primary_microcephaly_patients, color: '#e65100' },
              { label: 'Periventricular Heterotopia', value: overview.periventricular_heterotopia_patients, color: '#880e4f' },
              { label: 'With Seizures', value: overview.seizure_patients, color: '#6a1b9a' },
              { label: 'Cardiac Feature', value: overview.cardiac_feature_patients, color: '#004d40' },
              { label: 'Strabismus (BFPP)', value: overview.strabismus_patients, color: '#37474f' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                background: '#fff', border: `2px solid ${color}`, borderRadius: 10,
                padding: 16, textAlign: 'center',
              }}>
                <div style={{ fontSize: 32, fontWeight: 800, color }}>{value}</div>
                <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>
          <div style={{ background: '#e3f2fd', borderRadius: 10, padding: 16 }}>
            <h3 style={{ margin: '0 0 12px', color: '#1565c0' }}>Genes Covered</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {overview.genes.map(g => (
                <div key={g} style={{
                  background: GENE_COLORS[g] || '#555', color: '#fff',
                  padding: '6px 14px', borderRadius: 6, fontWeight: 700, fontSize: 13,
                }}>
                  {g} &nbsp;<span style={{ fontWeight: 400, fontSize: 11 }}>{GENE_INFO[g]?.locus}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {activeTab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1565c0', color: '#fff' }}>
                {['Gene', 'Full Name', 'Locus', 'Size', 'Inheritance', 'Disease / Syndrome', 'Patients'].map(h => (
                  <th key={h} style={{ padding: '10px 8px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, data], i) => (
                <tr key={gene} style={{ background: i % 2 === 0 ? '#fff' : '#f5f5f5' }}>
                  <td style={{ padding: '8px', fontWeight: 800, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                  <td style={{ padding: '8px' }}>{GENE_INFO[gene]?.full}</td>
                  <td style={{ padding: '8px', fontFamily: 'monospace' }}>{data.locus}</td>
                  <td style={{ padding: '8px', fontFamily: 'monospace' }}>{data.protein_size}</td>
                  <td style={{ padding: '8px' }}>{GENE_INFO[gene]?.inh}</td>
                  <td style={{ padding: '8px' }}>{GENE_INFO[gene]?.disease}</td>
                  <td style={{ padding: '8px', fontWeight: 700, textAlign: 'center' }}>{data.patient_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {activeTab === 'Clinical Atlas' && breakdown && (
        <div>
          {Object.entries(breakdown).map(([gene, data]) => (
            <div key={gene} style={{
              border: `2px solid ${GENE_COLORS[gene] || '#ccc'}`,
              borderRadius: 10, marginBottom: 24, overflow: 'hidden',
            }}>
              <div style={{
                background: GENE_COLORS[gene] || '#555', color: '#fff',
                padding: '12px 18px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                <span style={{ fontWeight: 800, fontSize: 16 }}>
                  {gene} — {GENE_INFO[gene]?.full}
                </span>
                <span style={{ fontSize: 13, opacity: 0.9 }}>
                  {data.locus} · {data.protein_size} · {GENE_INFO[gene]?.inh} · {data.patient_count} patients
                </span>
              </div>
              <div style={{ padding: 18 }}>
                <div style={{ marginBottom: 12 }}>
                  <strong style={{ color: '#b71c1c' }}>Pathognomonic:</strong>
                  <p style={{ margin: '4px 0', color: '#333', fontSize: 13 }}>{data.pathognomonic}</p>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <strong style={{ color: '#1565c0' }}>Age of Onset / Clinical:</strong>
                  <p style={{ margin: '4px 0', color: '#333', fontSize: 13 }}>{data.age_of_onset}</p>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <strong style={{ color: '#2e7d32' }}>Key Biomarkers / Workup:</strong>
                  <p style={{ margin: '4px 0', color: '#333', fontSize: 13 }}>{data.key_biomarker}</p>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <strong style={{ color: '#e65100' }}>Treatment:</strong>
                  <p style={{ margin: '4px 0', color: '#333', fontSize: 13 }}>{data.treatment}</p>
                </div>
                <div>
                  <strong style={{ color: '#880e4f' }}>Critical Flags:</strong>
                  <div style={{ marginTop: 6 }}>
                    {data.critical_flags.map(f => <FLAG_BADGE key={f} flag={f} />)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {activeTab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ color: '#1565c0' }}>Gene Proteins & Functions</h3>
          {Object.entries(definitions.genes).map(([gene, desc]) => (
            <div key={gene} style={{
              background: '#fff', border: `1px solid ${GENE_COLORS[gene] || '#ccc'}`,
              borderLeft: `6px solid ${GENE_COLORS[gene] || '#555'}`,
              borderRadius: 6, padding: '10px 14px', marginBottom: 10,
            }}>
              <strong style={{ color: GENE_COLORS[gene] || '#333' }}>{gene}</strong>
              <p style={{ margin: '4px 0', fontSize: 12, color: '#444' }}>{desc}</p>
            </div>
          ))}

          <h3 style={{ color: '#1565c0', marginTop: 24 }}>Glossary</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#e3f2fd' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', width: '28%' }}>Term</th>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Definition</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(definitions.glossary || {}).map(([term, def], i) => (
                <tr key={term} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 700, color: '#1565c0' }}>{term}</td>
                  <td style={{ padding: '8px 12px', color: '#444' }}>{def}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3 style={{ color: '#1565c0', marginTop: 24 }}>Surveillance Protocols</h3>
          {Object.entries(definitions.surveillance_protocols || {}).map(([gene, protocol]) => (
            <div key={gene} style={{
              background: '#fff', border: `1px solid ${GENE_COLORS[gene] || '#ccc'}`,
              borderLeft: `6px solid ${GENE_COLORS[gene] || '#555'}`,
              borderRadius: 6, padding: '10px 14px', marginBottom: 10,
            }}>
              <strong style={{ color: GENE_COLORS[gene] || '#333' }}>{gene}</strong>
              <p style={{ margin: '4px 0', fontSize: 12, color: '#444' }}>{protocol}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
