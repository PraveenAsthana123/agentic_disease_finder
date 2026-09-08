'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-myoclonus-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  CSTB:     '#1a237e',  // deep indigo   — ULD EPM1 dodecamer repeat
  EPM2A:    '#880e4f',  // deep pink      — Lafora type 1 laforin
  NHLRC1:   '#b71c1c',  // deep red       — Lafora type 2 malin
  SCARB2:   '#006064',  // deep teal      — AMRF renal failure
  GOSR2:    '#1b5e20',  // deep green     — Nord disease scoliosis
  KCNC1:    '#4a148c',  // deep purple    — EPM7 Kv3.1 giant SEPs
  PRICKLE1: '#e65100',  // deep orange    — EPM1B ULD-like
  KCTD7:    '#37474f',  // dark slate     — EPM3 infantile severe
};

const GENE_INFO = {
  CSTB:     { full: 'CSTB / Cystatin B 98aa',            locus: '21q22.3', size: '98 aa',  inh: 'AR', disease: 'EPM1 / Unverricht-Lundborg Disease — Dodecamer Repeat >30 / Piracetam Level A / NOT Fatal' },
  EPM2A:    { full: 'EPM2A / Laforin 331aa Phosphatase', locus: '6q24.3',  size: '331 aa', inh: 'AR', disease: 'Lafora Disease Type 1 — Lafora Bodies Skin Biopsy PATHOGNOMONIC / Occipital Seizures / Fatal 10-15yr' },
  NHLRC1:   { full: 'NHLRC1 / Malin 395aa E3 Ligase',   locus: '6p22.3',  size: '395 aa', inh: 'AR', disease: 'Lafora Disease Type 2 — Same Phenotype EPM2A / Mediterranean South Asian / Gene Distinguishes' },
  SCARB2:   { full: 'SCARB2 / LIMP2 478aa',              locus: '4q21.1',  size: '478 aa', inh: 'AR', disease: 'EPM4 / AMRF — Renal Failure PATHOGNOMONIC Co-Feature / Avoid NSAIDs / Hearing Loss' },
  GOSR2:    { full: 'GOSR2 / Golgi SNARE 235aa',         locus: '17q21.32',size: '235 aa', inh: 'AR', disease: 'EPM6 / Nord Disease — Scoliosis 100% PATHOGNOMONIC / Elevated CK / Early Onset 2-6yr' },
  KCNC1:    { full: 'KCNC1 / Kv3.1 585aa',              locus: '11p15.1', size: '585 aa', inh: 'AD', disease: 'EPM7 — R320H Dominant-Negative Finnish-Baltic / Giant SEPs PATHOGNOMONIC / CBZ ABSOLUTE CI' },
  PRICKLE1: { full: 'PRICKLE1 / Prickle-like 831aa',     locus: '12q12',   size: '831 aa', inh: 'AR', disease: 'EPM1B / ULD-Like — Slower Progression Than Lafora / Better Prognosis / Piracetam Levetiracetam' },
  KCTD7:    { full: 'KCTD7 / BTB-domain 289aa',          locus: '12q14.2', size: '289 aa', inh: 'AR', disease: 'EPM3 — Infantile Onset 1-2yr EARLIEST PME / Severe ID / NCL-Like / Exclude CLN2 First' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('LEVEL-A') || flag.includes('LEVEL-B') ? '#1565c0'
    : flag.includes('CONTRAINDICATED') || flag.includes('AVOID') || flag.includes('CI') || flag.includes('RISK') ? '#880e4f'
    : flag.includes('EMERGENCY') || flag.includes('FATAL') ? '#e65100'
    : flag.includes('DISTINGUISH') || flag.includes('FIRST') || flag.includes('FOCUS') ? '#2e7d32'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryMyoclonusAtlasPage() {
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
        ⚡ Hereditary Myoclonus Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Progressive Myoclonic Epilepsy (PME) Atlas — CSTB (ULD/EPM1) · EPM2A (Lafora-1) · NHLRC1 (Lafora-2) · SCARB2 (AMRF/EPM4) · GOSR2 (Nord/EPM6) · KCNC1 (EPM7) · PRICKLE1 (EPM1B) · KCTD7 (EPM3)
        &nbsp;|&nbsp; 320 patients · seeds 2086-2093
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
              { label: 'Lafora Bodies (EPM2A/NHLRC1)', value: overview.lafora_bodies_skin_biopsy_patients, color: '#880e4f' },
              { label: 'Occipital Seizures (Lafora)', value: overview.occipital_seizures_lafora_patients, color: '#b71c1c' },
              { label: 'Renal Failure (SCARB2)', value: overview.renal_failure_scarb2_patients, color: '#006064' },
              { label: 'Scoliosis (GOSR2)', value: overview.scoliosis_gosr2_patients, color: '#1b5e20' },
              { label: 'Giant SEPs (CSTB/KCNC1)', value: overview.giant_seps_patients, color: '#4a148c' },
              { label: 'Piracetam Patients', value: overview.piracetam_patients, color: '#1565c0' },
              { label: 'Photosensitive (PPR)', value: overview.photosensitive_patients, color: '#37474f' },
              { label: 'Severe ID (KCTD7)', value: overview.severe_id_kctd7_patients, color: '#e65100' },
              { label: 'Rapid Cognitive Decline (Lafora)', value: overview.rapid_cognitive_decline_lafora_patients, color: '#880e4f' },
              { label: 'Elevated CK (GOSR2)', value: overview.elevated_ck_gosr2_patients, color: '#1b5e20' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: 16, borderLeft: `4px solid ${color}` }}>
                <div style={{ fontSize: 28, fontWeight: 700, color }}>{value}</div>
                <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>

          <div style={{ background: '#fff3e0', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <strong style={{ color: '#e65100' }}>⚠ Critical PME Contraindications — ALL 8 Genes:</strong>
            <div style={{ marginTop: 8, color: '#333' }}>
              <strong>CBZ / OXC / PHT / LTG — ABSOLUTE CONTRAINDICATED in ALL PME</strong> (sodium channel blockade → NaV1.1 interneuron disinhibition → dramatic myoclonus worsening).
              GBP / PGB / VGB — HIGH RISK, avoid. Piracetam: Level A specifically for action myoclonus in ULD (CSTB/EPM1); Level C in KCNC1 (EPM7).
            </div>
          </div>

          <div style={{ background: '#e8f5e9', borderRadius: 8, padding: 16 }}>
            <strong style={{ color: '#1b5e20' }}>Gene Roster ({overview.genes?.length} genes):</strong>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
              {overview.genes?.map(g => (
                <span key={g} style={{ background: GENE_COLORS[g] || '#607d8b', color: '#fff', borderRadius: 4, padding: '4px 10px', fontSize: 13, fontWeight: 600 }}>{g}</span>
              ))}
            </div>
            <div style={{ marginTop: 10, color: '#555', fontSize: 13 }}>Seeds: {overview.seeds} · {overview.total_patients} total patients</div>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {activeTab === 'Gene Table' && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene', 'Full Name', 'Locus', 'Size', 'Inh', 'Disease / Key Facts'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(GENE_INFO).map(([gene, info], idx) => (
                <tr key={gene} style={{ background: idx % 2 === 0 ? '#f9f9f9' : '#fff', borderBottom: '1px solid #e0e0e0' }}>
                  <td style={{ padding: '10px 12px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                  <td style={{ padding: '10px 12px' }}>{info.full}</td>
                  <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>{info.locus}</td>
                  <td style={{ padding: '10px 12px' }}>{info.size}</td>
                  <td style={{ padding: '10px 12px' }}>
                    <span style={{ background: info.inh === 'AD' ? '#1565c0' : info.inh === 'AR' ? '#2e7d32' : '#880e4f', color: '#fff', borderRadius: 3, padding: '2px 6px', fontSize: 11 }}>{info.inh}</span>
                  </td>
                  <td style={{ padding: '10px 12px', color: '#333', fontSize: 12 }}>{info.disease}</td>
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
            <div key={gene} style={{ border: `2px solid ${GENE_COLORS[gene] || '#607d8b'}`, borderRadius: 10, marginBottom: 24, overflow: 'hidden' }}>
              <div style={{ background: GENE_COLORS[gene] || '#607d8b', color: '#fff', padding: '12px 20px' }}>
                <strong style={{ fontSize: 16 }}>{gene}</strong>
                <span style={{ marginLeft: 16, opacity: 0.85, fontSize: 13 }}>{data.locus} · {data.protein_size} · {data.inheritance?.split(';')[0]}</span>
                <span style={{ marginLeft: 16, background: 'rgba(255,255,255,0.2)', borderRadius: 4, padding: '2px 8px', fontSize: 12 }}>{data.patient_count} patients</span>
              </div>
              <div style={{ padding: '16px 20px' }}>
                <div style={{ marginBottom: 12 }}>
                  <strong style={{ color: '#333', fontSize: 13 }}>Age of Onset:</strong>
                  <span style={{ marginLeft: 8, color: '#555', fontSize: 13 }}>{data.age_of_onset}</span>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <strong style={{ color: '#b71c1c', fontSize: 13 }}>Pathognomonic / Key Features:</strong>
                  <p style={{ margin: '4px 0 0', color: '#444', fontSize: 13, whiteSpace: 'pre-line' }}>{data.pathognomonic}</p>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <strong style={{ color: '#1565c0', fontSize: 13 }}>Treatment:</strong>
                  <p style={{ margin: '4px 0 0', color: '#444', fontSize: 13, whiteSpace: 'pre-line' }}>{data.treatment}</p>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <strong style={{ color: '#2e7d32', fontSize: 13 }}>Key Biomarker:</strong>
                  <p style={{ margin: '4px 0 0', color: '#444', fontSize: 13 }}>{data.key_biomarker}</p>
                </div>
                <div>
                  <strong style={{ color: '#333', fontSize: 13 }}>Critical Flags:</strong>
                  <div style={{ marginTop: 6 }}>
                    {data.critical_flags?.map(f => <FLAG_BADGE key={f} flag={f} />)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {activeTab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ color: '#1a237e' }}>Gene Proteins</h3>
          {Object.entries(definitions.genes || {}).map(([gene, protein]) => (
            <div key={gene} style={{ marginBottom: 14, padding: 14, background: '#f5f5f5', borderRadius: 8, borderLeft: `4px solid ${GENE_COLORS[gene] || '#607d8b'}` }}>
              <strong style={{ color: GENE_COLORS[gene] || '#333' }}>{gene}:</strong>
              <span style={{ marginLeft: 8, color: '#444', fontSize: 13, wordBreak: 'break-word' }}>{protein}</span>
            </div>
          ))}

          <h3 style={{ color: '#1a237e', marginTop: 28 }}>Glossary</h3>
          {Object.entries(definitions.glossary || {}).map(([term, def]) => (
            <div key={term} style={{ marginBottom: 12, padding: 14, background: '#e8eaf6', borderRadius: 8 }}>
              <strong style={{ color: '#283593' }}>{term}:</strong>
              <p style={{ margin: '4px 0 0', color: '#444', fontSize: 13 }}>{def}</p>
            </div>
          ))}

          <h3 style={{ color: '#1a237e', marginTop: 28 }}>Surveillance Protocols</h3>
          {Object.entries(definitions.surveillance_protocols || {}).map(([gene, protocol]) => (
            <div key={gene} style={{ marginBottom: 12, padding: 14, background: '#e8f5e9', borderRadius: 8, borderLeft: '4px solid #2e7d32' }}>
              <strong style={{ color: '#1b5e20' }}>{gene}:</strong>
              <p style={{ margin: '4px 0 0', color: '#444', fontSize: 13 }}>{protocol}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
