'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-dystonia-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  TOR1A:  '#1a237e',  // deep indigo  — DYT1 early-onset generalised GPi-DBS
  THAP1:  '#4a148c',  // deep purple  — DYT6 cranial-cervical-laryngeal spread
  GCH1:   '#1b5e20',  // deep green   — DRD levodopa MIRACULOUS diurnal fluctuation
  ATP1A3: '#e65100',  // deep orange  — AHC/RDP/CAPOS ATP1A3 triad
  KMT2B:  '#880e4f',  // deep pink    — DYT28 oculomotor GPi-DBS highly responsive
  ADCY5:  '#006064',  // deep teal    — nocturnal dyskinesia caffeine CI
  ANO3:   '#bf360c',  // deep burnt   — DYT24 craniocervical tremor BoNT-A
  GNAL:   '#37474f',  // dark slate   — DYT25 spasmodic dysphonia cranial
};

const GENE_INFO = {
  TOR1A:  { full: 'TOR1A / Torsin 1A AAA+ ATPase', locus: '9q34.11',  size: '332 aa',  inh: 'AD', disease: 'DYT-TOR1A (DYT1) — Early-Onset Generalised / GAG del / Penetrance 30% / GPi-DBS' },
  THAP1:  { full: 'THAP1 / THAP Zinc Finger TF',   locus: '8p11.21',  size: '213 aa',  inh: 'AD', disease: 'DYT-THAP1 (DYT6) — Mixed Onset / Cranial-Cervical-Laryngeal / BoNT-A' },
  GCH1:   { full: 'GCH1 / GTP Cyclohydrolase 1',   locus: '14q22.2',  size: '250 aa',  inh: 'AD', disease: 'DYT-GCH1 (DRD/Segawa) — Levodopa MIRACULOUS / Diurnal Fluctuation / Female 3:1' },
  ATP1A3: { full: 'ATP1A3 / Na+/K+-ATPase α3',     locus: '19q13.2',  size: '1013 aa', inh: 'AD', disease: 'ATP1A3 — AHC / RDP / CAPOS Triad / Flunarizine / Rostrocaudal Gradient' },
  KMT2B:  { full: 'KMT2B / MLL4 H3K4 Methylase',  locus: '19q13.12', size: '2715 aa', inh: 'AD', disease: 'DYT-KMT2B (DYT28) — Childhood Complex / Oculomotor / GPi-DBS Highly Responsive' },
  ADCY5:  { full: 'ADCY5 / Adenylate Cyclase 5',   locus: '3q21.3',   size: '1261 aa', inh: 'AD', disease: 'ADCY5-RMD — Nocturnal Dyskinesia / Caffeine CI / Clonazepam / Facial Hypotonia' },
  ANO3:   { full: 'ANO3 / Anoctamin 3 Cl Channel', locus: '11p14.3',  size: '981 aa',  inh: 'AD', disease: 'DYT-ANO3 (DYT24) — Adult Craniocervical / Tremor Prominent / BoNT-A' },
  GNAL:   { full: 'GNAL / Golf G-Protein α-L',     locus: '18p11.21', size: '381 aa',  inh: 'AD', disease: 'DYT-GNAL (DYT25) — Spasmodic Dysphonia / Cranial / Laryngeal BoNT-A' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('MIRACULOUS') || flag.includes('HIGHLY') ? '#1565c0'
    : flag.includes('CI') || flag.includes('CONTRAINDICATED') || flag.includes('ABSOLUTELY') || flag.includes('AVOID') ? '#880e4f'
    : flag.includes('EMERGENCY') ? '#e65100'
    : flag.includes('DISTINGUISH') || flag.includes('DDx') || flag.includes('CONTRAST') ? '#2e7d32'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryDystoniaAtlasPage() {
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
      <h1 style={{ color: '#1a237e', marginBottom: 4 }}>
        🧠 Hereditary Dystonia Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Atlas — DYT-TOR1A · DYT-THAP1 · DRD-GCH1 · ATP1A3-AHC/RDP/CAPOS · DYT-KMT2B · ADCY5-RMD · DYT-ANO3 · DYT-GNAL
        &nbsp;|&nbsp; TOR1A · THAP1 · GCH1 · ATP1A3 · KMT2B · ADCY5 · ANO3 · GNAL
        &nbsp;|&nbsp; 320 patients · seeds 2070-2077
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: activeTab === t ? '#1a237e' : '#e8eaf6',
            color: activeTab === t ? '#fff' : '#1a237e', fontWeight: activeTab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {loading && <p style={{ color: '#888' }}>Loading…</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* OVERVIEW TAB */}
      {activeTab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: 14, marginBottom: 28 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#1a237e' },
              { label: 'Generalised Dystonia', value: overview.generalised_dystonia_patients, color: '#4a148c' },
              { label: 'GPi-DBS Patients', value: overview.gpi_dbs_patients, color: '#1565c0' },
              { label: 'Levodopa-Responsive (DRD)', value: overview.levodopa_responsive_drd_patients, color: '#1b5e20' },
              { label: 'Botulinum Toxin Patients', value: overview.botulinum_toxin_patients, color: '#37474f' },
              { label: 'Nocturnal Dyskinesia (ADCY5)', value: overview.nocturnal_dyskinesia_patients, color: '#006064' },
              { label: 'Oculomotor Abnormality (KMT2B)', value: overview.oculomotor_abnormality_patients, color: '#880e4f' },
              { label: 'Misdiagnosed as CP (GCH1)', value: overview.misdiagnosed_cerebral_palsy, color: '#bf360c' },
              { label: 'Spasmodic Dysphonia (GNAL/THAP1)', value: overview.spasmodic_dysphonia_patients, color: '#6a1b9a' },
              { label: 'Tremor Prominent (ANO3)', value: overview.tremor_prominent_patients, color: '#bf360c' },
              { label: 'Flunarizine Use (AHC)', value: overview.flunarizine_use_ahc, color: '#e65100' },
              { label: 'Seeds', value: overview.seeds, color: '#455a64', isText: true },
            ].map(({ label, value, color, isText }) => (
              <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: '14px 18px', borderLeft: `4px solid ${color}` }}>
                <div style={{ fontSize: 12, color: '#777', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: isText ? 16 : 28, fontWeight: 700, color }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Gene colour legend */}
          <h3 style={{ color: '#1a237e', marginBottom: 12 }}>8-Gene Dystonia Spectrum</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
            {Object.entries(GENE_INFO).map(([gene, info]) => (
              <div key={gene} style={{
                background: GENE_COLORS[gene], color: '#fff',
                borderRadius: 6, padding: '8px 14px', minWidth: 180,
              }}>
                <div style={{ fontWeight: 700, fontSize: 15 }}>{gene}</div>
                <div style={{ fontSize: 11, opacity: 0.88 }}>{info.locus} · {info.size} · {info.inh}</div>
                <div style={{ fontSize: 11, opacity: 0.80, marginTop: 2 }}>{info.disease}</div>
              </div>
            ))}
          </div>

          {/* Key clinical rules */}
          <h3 style={{ color: '#1a237e', marginBottom: 8 }}>Key Clinical Rules — Hereditary Dystonia</h3>
          <div style={{ background: '#e8eaf6', borderRadius: 8, padding: 16 }}>
            <ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8 }}>
              <li><strong>LEVODOPA TRIAL IN ALL CHILDHOOD DYSTONIA</strong> — mandatory before any other treatment; GCH1/DRD responds miraculously; no downside to trial</li>
              <li><strong>DIURNAL FLUCTUATION (worse evening, better morning)</strong> = DRD/GCH1 pathognomonic — always trial levodopa immediately</li>
              <li><strong>DYT1/TOR1A PENETRANCE 30%</strong> — positive genetic test ≠ disease; counsel asymptomatic carriers carefully</li>
              <li><strong>KMT2B: CMA MANDATORY</strong> alongside sequencing — 30% of variants are microdeletions missed by sequencing alone</li>
              <li><strong>ADCY5: CAFFEINE ABSOLUTELY CONTRAINDICATED</strong> — adenosine blockade worsens cAMP GOF pathway; even 1 cup can trigger severe episodes</li>
              <li><strong>AHC (ATP1A3): EPISODES RESOLVE WITH SLEEP</strong> — pathognomonic; bilateral hemiplegia = respiratory emergency</li>
              <li><strong>TETRABENAZINE WORSENS PRIMARY DYSTONIA</strong> (TOR1A/THAP1/KMT2B) — only use in secondary/Huntington's chorea</li>
              <li><strong>GESTE ANTAGONISTE (sensory trick)</strong> = specific to focal dystonia; absent in Parkinson's cervical rigidity</li>
              <li><strong>SPASMODIC DYSPHONIA</strong> — BoNT-A laryngeal injection is standard of care; GNAL/THAP1 genetic workup if familial</li>
              <li><strong>GPi-DBS: earlier is better</strong> — especially in KMT2B (DYT28) and TOR1A (DYT1); do not delay for age</li>
            </ul>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {activeTab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Inh.', 'Patients', 'Disease / DYT', 'Key Treatment', 'Critical Flags'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, data], i) => (
                <tr key={gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                  <td style={{ padding: '9px 12px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                  <td style={{ padding: '9px 12px' }}>{data.locus}</td>
                  <td style={{ padding: '9px 12px' }}>{data.protein_size}</td>
                  <td style={{ padding: '9px 12px' }}>{data.inheritance.split(';')[0].split('(')[0].trim()}</td>
                  <td style={{ padding: '9px 12px', textAlign: 'center', fontWeight: 700 }}>{data.patient_count}</td>
                  <td style={{ padding: '9px 12px', maxWidth: 220, fontSize: 12 }}>{GENE_INFO[gene]?.disease || '—'}</td>
                  <td style={{ padding: '9px 12px', maxWidth: 200, fontSize: 12 }}>{data.treatment?.split(';')[0]?.replace(/\*\*/g, '') || '—'}</td>
                  <td style={{ padding: '9px 12px', maxWidth: 260 }}>
                    <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                      {(data.critical_flags || []).slice(0, 3).map(f => <FLAG_BADGE key={f} flag={f} />)}
                    </div>
                  </td>
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
              borderLeft: `5px solid ${GENE_COLORS[gene]}`,
              background: '#fafafa', borderRadius: 8, padding: 18, marginBottom: 20,
            }}>
              <h3 style={{ color: GENE_COLORS[gene], marginTop: 0, marginBottom: 6 }}>
                {gene} — {GENE_INFO[gene]?.full}
              </h3>
              <div style={{ fontSize: 12, color: '#777', marginBottom: 10 }}>
                {data.locus} · {data.protein_size} · {data.inheritance?.split(';')[0]}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                <div>
                  <div style={{ fontWeight: 600, color: '#444', marginBottom: 4 }}>Age of Onset</div>
                  <div style={{ fontSize: 13, color: '#555' }}>{data.age_of_onset}</div>
                </div>
                <div>
                  <div style={{ fontWeight: 600, color: '#444', marginBottom: 4 }}>Key Biomarker</div>
                  <div style={{ fontSize: 13, color: '#555' }}>{data.key_biomarker}</div>
                </div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, color: '#b71c1c', marginBottom: 4 }}>Pathognomonic Signs</div>
                <div style={{ fontSize: 13, color: '#555' }}>{data.pathognomonic}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, color: '#1565c0', marginBottom: 4 }}>Treatment</div>
                <div style={{ fontSize: 13, color: '#555' }}>{data.treatment}</div>
              </div>

              <div>
                <div style={{ fontWeight: 600, color: '#444', marginBottom: 6 }}>Critical Flags</div>
                <div style={{ display: 'flex', flexWrap: 'wrap' }}>
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
          <h3 style={{ color: '#1a237e', marginBottom: 12 }}>Gene Definitions</h3>
          {Object.entries(definitions.genes || {}).map(([gene, desc]) => (
            <div key={gene} style={{ marginBottom: 14, borderLeft: `4px solid ${GENE_COLORS[gene]}`, paddingLeft: 14 }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[gene], marginBottom: 2 }}>{gene}</div>
              <div style={{ fontSize: 13, color: '#555', lineHeight: 1.6 }}>{desc.replace(/--/g, '·')}</div>
            </div>
          ))}

          <h3 style={{ color: '#1a237e', marginTop: 28, marginBottom: 12 }}>Glossary</h3>
          {Object.entries(definitions.glossary || {}).map(([term, def]) => (
            <div key={term} style={{ marginBottom: 12, background: '#f5f5f5', borderRadius: 6, padding: '10px 14px' }}>
              <div style={{ fontWeight: 600, color: '#1a237e', marginBottom: 3 }}>{term}</div>
              <div style={{ fontSize: 13, color: '#555', lineHeight: 1.6 }}>{def}</div>
            </div>
          ))}

          <h3 style={{ color: '#1a237e', marginTop: 28, marginBottom: 12 }}>Surveillance Protocols</h3>
          {Object.entries(definitions.surveillance_protocols || {}).map(([gene, protocol]) => (
            <div key={gene} style={{ marginBottom: 12, borderLeft: `4px solid ${GENE_COLORS[gene.split(' ')[0]] || '#455a64'}`, paddingLeft: 14 }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[gene.split(' ')[0]] || '#455a64', marginBottom: 3 }}>{gene}</div>
              <div style={{ fontSize: 13, color: '#555', lineHeight: 1.6 }}>{protocol}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
