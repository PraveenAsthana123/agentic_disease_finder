'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hepatic-disease-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ATP7B:    '#1565c0',  // deep blue    — Wilson disease copper ATPase KF rings
  HFE:      '#b71c1c',  // deep red     — hereditary haemochromatosis C282Y phlebotomy
  SERPINA1: '#e65100',  // deep orange  — alpha-1-AT deficiency PiZZ emphysema liver
  JAG1:     '#2e7d32',  // deep green   — Alagille syndrome butterfly vertebrae Notch
  ABCB11:   '#880e4f',  // deep magenta — PFIC2 BSEP high GGT childhood HCC
  ATP8B1:   '#6a1b9a',  // deep purple  — PFIC1 FIC1 low GGT extrahepatic
  SLC25A13: '#004d40',  // deep teal    — citrinemia NICCD CTLN2 protein preference
  NPC1:     '#37474f',  // dark slate   — NPC vertical gaze palsy miglustat
};

const GENE_INFO = {
  ATP7B:    { full: 'ATP7B / Wilson Protein', locus: '13q14.3', size: '1465 aa', inh: 'AR', disease: 'Wilson Disease — Kayser-Fleischer Rings / Chelation' },
  HFE:      { full: 'HFE / Haemochromatosis', locus: '6p21.3', size: '348 aa', inh: 'AR', disease: 'Hereditary Haemochromatosis — C282Y Phlebotomy' },
  SERPINA1: { full: 'Alpha-1-Antitrypsin', locus: '14q32.13', size: '418 aa', inh: 'AR', disease: 'AATD — PiZZ Emphysema + Cirrhosis (Smoking CI)' },
  JAG1:     { full: 'Jagged-1 / Notch Ligand', locus: '20p12.2', size: '1218 aa', inh: 'AD', disease: 'Alagille Syndrome — Butterfly Vertebrae + Paucity Bile Ducts' },
  ABCB11:   { full: 'BSEP / Bile Salt Export Pump', locus: '2q31.1', size: '1321 aa', inh: 'AR', disease: 'PFIC2 — HIGH GGT / Childhood HCC Risk' },
  ATP8B1:   { full: 'FIC1 / Phospholipid Flippase', locus: '18q21.31', size: '1251 aa', inh: 'AR', disease: 'PFIC1 / Byler — LOW GGT / Extrahepatic' },
  SLC25A13: { full: 'Citrin / Mitochondrial AGC2', locus: '7q21.3', size: '675 aa', inh: 'AR', disease: 'Citrinemia NICCD/CTLN2 — Protein Preference' },
  NPC1:     { full: 'NPC1 / Lysosomal Cholesterol', locus: '18q11.2', size: '1278 aa', inh: 'AR', disease: 'NPC — Vertical Gaze Palsy + Miglustat' },
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

export default function HereditaryHepaticDiseaseAtlasPage() {
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
        🫀 Hereditary Hepatic Disease Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Atlas — Wilson Disease · Haemochromatosis · AATD · Alagille · PFIC2 · PFIC1 · Citrinemia · NPC
        &nbsp;|&nbsp; ATP7B · HFE · SERPINA1 · JAG1 · ABCB11 · ATP8B1 · SLC25A13 · NPC1
        &nbsp;|&nbsp; 320 patients · seeds 2054-2061
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: activeTab === t ? '#1565c0' : '#e0e0e0',
            color: activeTab === t ? '#fff' : '#333', fontWeight: activeTab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {loading && <p style={{ color: '#888' }}>Loading…</p>}
      {error && <p style={{ color: '#c62828' }}>Error: {error}</p>}

      {/* Overview Tab */}
      {activeTab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#1565c0' },
              { label: 'Seeds', value: overview.seeds, color: '#37474f' },
              { label: 'Liver Disease', value: overview.liver_disease_patients, color: '#b71c1c' },
              { label: 'Cirrhosis', value: overview.cirrhosis_patients, color: '#880e4f' },
              { label: 'Transplant Patients', value: overview.transplant_patients, color: '#004d40' },
              { label: 'Neonatal Cholestasis', value: overview.neonatal_cholestasis_patients, color: '#e65100' },
              { label: 'Neurological', value: overview.neurological_patients, color: '#6a1b9a' },
              { label: 'Pulmonary', value: overview.pulmonary_patients, color: '#2e7d32' },
              { label: 'HCC Patients', value: overview.hcc_patients, color: '#c62828' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: 16, borderLeft: `4px solid ${color}` }}>
                <div style={{ fontSize: 28, fontWeight: 700, color }}>{value}</div>
                <div style={{ fontSize: 13, color: '#555' }}>{label}</div>
              </div>
            ))}
          </div>
          <h3 style={{ color: '#1565c0' }}>Genes in This Atlas</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            {overview.genes && overview.genes.map(gene => (
              <div key={gene} style={{
                background: GENE_COLORS[gene] || '#1565c0', color: '#fff',
                borderRadius: 8, padding: '10px 16px', minWidth: 160,
              }}>
                <div style={{ fontWeight: 700, fontSize: 16 }}>{gene}</div>
                <div style={{ fontSize: 12, opacity: 0.85 }}>{GENE_INFO[gene]?.disease}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {activeTab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1565c0', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease', 'Patients', 'Pathognomonic'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.values(breakdown).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                  <td style={{ padding: '10px 12px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#1565c0' }}>{g.gene}</td>
                  <td style={{ padding: '10px 12px' }}>{g.locus}</td>
                  <td style={{ padding: '10px 12px' }}>{g.protein_size}</td>
                  <td style={{ padding: '10px 12px' }}>{g.inheritance?.split(';')[0]}</td>
                  <td style={{ padding: '10px 12px' }}>{GENE_INFO[g.gene]?.disease}</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center' }}>{g.patient_count}</td>
                  <td style={{ padding: '10px 12px', fontSize: 11, maxWidth: 280 }}>{g.pathognomonic?.split(';')[0]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {activeTab === 'Clinical Atlas' && breakdown && (
        <div>
          {Object.values(breakdown).map(g => (
            <div key={g.gene} style={{
              marginBottom: 32, border: `2px solid ${GENE_COLORS[g.gene] || '#1565c0'}`,
              borderRadius: 10, padding: 20,
            }}>
              <h2 style={{ color: GENE_COLORS[g.gene] || '#1565c0', marginTop: 0 }}>
                {g.gene} — {GENE_INFO[g.gene]?.disease}
              </h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 12 }}>
                <div>
                  <b>Locus:</b> {g.locus} &nbsp;|&nbsp; <b>Size:</b> {g.protein_size} &nbsp;|&nbsp; <b>Inheritance:</b> {g.inheritance?.split(';')[0]}
                </div>
                <div><b>Patients:</b> {g.patient_count}</div>
              </div>
              <div style={{ marginBottom: 10 }}>
                <b style={{ color: '#b71c1c' }}>Pathognomonic:</b>
                <div style={{ marginTop: 4, fontSize: 13, color: '#333', lineHeight: 1.6 }}>
                  {g.pathognomonic?.split(';').map((s, i) => s.trim() && (
                    <div key={i} style={{ marginBottom: 4 }}>• {s.trim()}</div>
                  ))}
                </div>
              </div>
              <div style={{ marginBottom: 10 }}>
                <b style={{ color: '#2e7d32' }}>Key Biomarkers:</b>
                <div style={{ marginTop: 4, fontSize: 13, color: '#333', lineHeight: 1.6 }}>
                  {g.key_biomarker?.split(';').slice(0, 4).map((s, i) => s.trim() && (
                    <div key={i}>• {s.trim()}</div>
                  ))}
                </div>
              </div>
              <div style={{ marginBottom: 10 }}>
                <b style={{ color: '#1565c0' }}>Treatment:</b>
                <div style={{ marginTop: 4, fontSize: 13, color: '#333', lineHeight: 1.6 }}>
                  {g.treatment?.split(';').slice(0, 4).map((s, i) => s.trim() && (
                    <div key={i}>• {s.trim()}</div>
                  ))}
                </div>
              </div>
              <div>
                <b>Critical Flags:</b>
                <div style={{ marginTop: 6 }}>
                  {g.critical_flags?.map(f => <FLAG_BADGE key={f} flag={f} />)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {activeTab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ color: '#1565c0' }}>Gene Definitions</h3>
          {Object.entries(definitions.genes || {}).map(([gene, def]) => (
            <div key={gene} style={{ marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[gene] || '#1565c0'}`, paddingLeft: 16 }}>
              <b style={{ color: GENE_COLORS[gene] || '#1565c0' }}>{gene}:</b>
              <span style={{ fontSize: 13, color: '#444', marginLeft: 8 }}>{def}</span>
            </div>
          ))}
          <h3 style={{ color: '#1565c0', marginTop: 32 }}>Glossary</h3>
          {Object.entries(definitions.glossary || {}).map(([term, def]) => (
            <div key={term} style={{ marginBottom: 10, borderBottom: '1px solid #eee', paddingBottom: 10 }}>
              <b>{term}:</b> <span style={{ fontSize: 13, color: '#555' }}>{def}</span>
            </div>
          ))}
          <h3 style={{ color: '#1565c0', marginTop: 32 }}>Surveillance Protocols</h3>
          {Object.entries(definitions.surveillance_protocols || {}).map(([gene, proto]) => (
            <div key={gene} style={{ marginBottom: 14, borderLeft: `4px solid ${GENE_COLORS[gene] || '#1565c0'}`, paddingLeft: 16 }}>
              <b style={{ color: GENE_COLORS[gene] || '#1565c0' }}>{gene}:</b>
              <span style={{ fontSize: 13, color: '#444', marginLeft: 8 }}>{proto}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
