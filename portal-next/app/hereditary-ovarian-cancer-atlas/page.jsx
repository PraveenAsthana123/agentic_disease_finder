'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ovarian-cancer-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  BRCA1:  '#880e4f',  // deep magenta  — HBOC1 44% ovarian risk RRSO 35-40yr
  BRCA2:  '#1565c0',  // deep blue     — HBOC2 17% ovarian risk RRSO 40-45yr
  BRIP1:  '#e65100',  // deep orange   — FANCJ 11-13x ovarian RR NO breast risk
  RAD51C: '#2e7d32',  // deep green    — FANCO 5-6x ovarian RR NO breast risk
  RAD51D: '#004d40',  // deep teal     — 5-6x ovarian RR very low breast risk
  PALB2:  '#6a1b9a',  // deep purple   — FANCN 53% breast 3-5% ovarian PALB2
  MLH1:   '#b71c1c',  // deep red      — Lynch1 dMMR endometrioid NOT HGSOC
  MSH2:   '#37474f',  // dark slate    — Lynch2 EPCAM Muir-Torre sebaceous
};

const GENE_INFO = {
  BRCA1:  { full: 'BRCA1 / RING Domain Protein', locus: '17q21.31', size: '1863 aa', inh: 'AD', disease: 'HBOC1 — 44% Ovarian Risk / RRSO 35-40yr / Olaparib SOLO-1' },
  BRCA2:  { full: 'BRCA2 / FANCD1', locus: '13q12.3', size: '3418 aa', inh: 'AD', disease: 'HBOC2 — 17% Ovarian Risk / RRSO 40-45yr / Male Breast+Prostate Risk' },
  BRIP1:  { full: 'BRIP1 / FANCJ Helicase', locus: '17q23.2', size: '1249 aa', inh: 'AD', disease: 'HOCA3 — 11-13x Ovarian RR / NO Breast Risk / RRSO 45-50yr' },
  RAD51C: { full: 'RAD51C / FANCO Paralog', locus: '17q22', size: '376 aa', inh: 'AD', disease: 'HOCA4 — 5-6x Ovarian RR / NO Breast Risk / RRSO 45-50yr' },
  RAD51D: { full: 'RAD51D / HR Repair Paralog', locus: '17q12', size: '328 aa', inh: 'AD', disease: 'HOCA5 — 5-6x Ovarian RR / Very Low Breast Risk / RRSO 45-50yr' },
  PALB2:  { full: 'PALB2 / FANCN Bridge', locus: '16p12.2', size: '1186 aa', inh: 'AD', disease: 'HOCA6 — 53% Breast + 3-5% Ovarian / RRSO Timing Controversial' },
  MLH1:   { full: 'MLH1 / MutL Homolog 1', locus: '3p22.2', size: '756 aa', inh: 'AD', disease: 'Lynch 1 — Endometrioid OC / dMMR / Pembrolizumab FDA 2017' },
  MSH2:   { full: 'MSH2 / MutS Homolog 2', locus: '2p21', size: '934 aa', inh: 'AD', disease: 'Lynch 2 — EPCAM Deletion / Muir-Torre / dMMR / Pembrolizumab' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('CURATIVE') ? '#1565c0'
    : flag.includes('CI') || flag.includes('CONTRAINDICATED') || flag.includes('ABSOLUTELY') ? '#880e4f'
    : flag.includes('EMERGENCY') ? '#e65100'
    : flag.includes('DISTINGUISH') ? '#2e7d32'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryOvarianCancerAtlasPage() {
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
      <h1 style={{ color: '#880e4f', marginBottom: 4 }}>
        🎗️ Hereditary Ovarian Cancer Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Atlas — HBOC1 · HBOC2 · BRIP1/FANCJ · RAD51C/FANCO · RAD51D · PALB2/FANCN · Lynch MLH1 · Lynch MSH2
        &nbsp;|&nbsp; BRCA1 · BRCA2 · BRIP1 · RAD51C · RAD51D · PALB2 · MLH1 · MSH2
        &nbsp;|&nbsp; 320 patients · seeds 2062-2069
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: activeTab === t ? '#880e4f' : '#f5f5f5',
            color: activeTab === t ? '#fff' : '#333',
            fontWeight: activeTab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {loading && <p style={{ color: '#888' }}>Loading…</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* OVERVIEW TAB */}
      {activeTab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 12, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#880e4f' },
              { label: 'Ovarian Cancer', value: overview.ovarian_cancer_patients, color: '#b71c1c' },
              { label: 'Breast Cancer', value: overview.breast_cancer_patients, color: '#1565c0' },
              { label: 'HGSOC', value: overview.hgsoc_patients, color: '#880e4f' },
              { label: 'Endometrial Cancer', value: overview.endometrial_cancer_patients, color: '#6a1b9a' },
              { label: 'RRSO Done', value: overview.rrso_done_patients, color: '#2e7d32' },
              { label: 'PARP Inhibitor Use', value: overview.parp_inhibitor_patients, color: '#004d40' },
              { label: 'dMMR / MSI-H', value: overview.dmmr_msi_h_patients, color: '#37474f' },
              { label: 'Pembrolizumab', value: overview.pembrolizumab_patients, color: '#e65100' },
              { label: 'Synchronous Endo+OC', value: overview.synchronous_endometrial_ovarian, color: '#b71c1c' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ background: '#fafafa', border: `2px solid ${color}`, borderRadius: 8, padding: 14, textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color }}>{value}</div>
                <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#880e4f' }}>8 Genes — Hereditary Ovarian Cancer Atlas</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {(overview.genes || []).map(gene => (
              <div key={gene} style={{
                background: GENE_COLORS[gene] || '#555', color: '#fff',
                borderRadius: 8, padding: '8px 16px', fontWeight: 600,
              }}>
                {gene} — {GENE_INFO[gene]?.disease || ''}
              </div>
            ))}
          </div>

          <div style={{ marginTop: 24, background: '#fce4ec', borderRadius: 8, padding: 16 }}>
            <strong style={{ color: '#880e4f' }}>🎗️ Atlas Summary — seeds {overview.seeds}</strong>
            <p style={{ margin: '8px 0 0', color: '#555', fontSize: 14 }}>
              320-patient aggregate (8×40). BRCA1/BRCA2: high-grade serous OC (HGSOC); PARP inhibitors (olaparib SOLO-1/SOLO-2);
              RRSO 35-40yr (BRCA1) / 40-45yr (BRCA2). BRIP1/RAD51C/RAD51D: moderate OC risk 5-13x WITHOUT breast risk;
              RRSO 45-50yr. PALB2: high breast risk (53%) + moderate OC (3-5%); RRSO timing controversial.
              MLH1/MSH2: Lynch syndrome — endometrioid/clear cell OC (NOT HGSOC); dMMR/MSI-H; pembrolizumab FDA 2017;
              EPCAM deletion misses standard sequencing (MSH2). Universal MMR IHC on all endometrial + colorectal tumors.
            </p>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {activeTab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#880e4f', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Inh', 'Disease / Key Risk', 'Patients'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, data], i) => (
                <tr key={gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                  <td style={{ padding: '8px 12px' }}>{data.locus}</td>
                  <td style={{ padding: '8px 12px' }}>{data.protein_size}</td>
                  <td style={{ padding: '8px 12px' }}>{GENE_INFO[gene]?.inh || 'AD'}</td>
                  <td style={{ padding: '8px 12px', maxWidth: 320 }}>{GENE_INFO[gene]?.disease || ''}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700 }}>{data.patient_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {activeTab === 'Clinical Atlas' && breakdown && (
        <div>
          {Object.entries(breakdown).map(([gene, data]) => (
            <div key={gene} style={{
              border: `2px solid ${GENE_COLORS[gene] || '#ccc'}`,
              borderRadius: 10, padding: 18, marginBottom: 20,
            }}>
              <h3 style={{ color: GENE_COLORS[gene], margin: '0 0 8px' }}>
                {gene} — {data.locus} · {data.protein_size}
              </h3>
              <p style={{ fontSize: 13, color: '#444', margin: '0 0 10px' }}><strong>Inheritance:</strong> {data.inheritance}</p>
              <p style={{ fontSize: 13, color: '#444', margin: '0 0 10px' }}><strong>Age of Onset:</strong> {data.age_of_onset}</p>
              <p style={{ fontSize: 13, color: '#444', margin: '0 0 10px' }}><strong>Key Biomarker:</strong> {data.key_biomarker}</p>
              <p style={{ fontSize: 13, color: '#444', margin: '0 0 10px' }}><strong>Pathognomonic:</strong> {data.pathognomonic}</p>
              <p style={{ fontSize: 13, color: '#444', margin: '0 0 10px' }}><strong>Treatment:</strong> {data.treatment}</p>
              <div style={{ marginTop: 8 }}>
                <strong style={{ fontSize: 13 }}>Critical Flags:</strong>
                <div style={{ marginTop: 4 }}>
                  {(data.critical_flags || []).map(f => <FLAG_BADGE key={f} flag={f} />)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {activeTab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ color: '#880e4f' }}>Gene Definitions</h3>
          {Object.entries(definitions.genes || {}).map(([gene, def]) => (
            <div key={gene} style={{ marginBottom: 12, padding: 12, background: '#fafafa', borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS[gene] || '#ccc'}` }}>
              <strong style={{ color: GENE_COLORS[gene] }}>{gene}</strong>
              <p style={{ margin: '4px 0 0', fontSize: 13, color: '#555' }}>{def}</p>
            </div>
          ))}

          <h3 style={{ color: '#880e4f', marginTop: 24 }}>Glossary</h3>
          {Object.entries(definitions.glossary || {}).map(([term, def]) => (
            <div key={term} style={{ marginBottom: 10, padding: 10, background: '#fafafa', borderRadius: 6 }}>
              <strong style={{ color: '#37474f' }}>{term}</strong>
              <p style={{ margin: '4px 0 0', fontSize: 13, color: '#555' }}>{def}</p>
            </div>
          ))}

          <h3 style={{ color: '#880e4f', marginTop: 24 }}>Surveillance Protocols</h3>
          {Object.entries(definitions.surveillance_protocols || {}).map(([gene, protocol]) => (
            <div key={gene} style={{ marginBottom: 12, padding: 12, background: '#fce4ec', borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS[gene] || '#ccc'}` }}>
              <strong style={{ color: GENE_COLORS[gene] }}>{gene}</strong>
              <p style={{ margin: '4px 0 0', fontSize: 13, color: '#555' }}>{protocol}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
