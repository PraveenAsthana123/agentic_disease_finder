'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-myotonic-disorder-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  DMPK:    '#1565c0',  // deep blue    — DM1 multisystem anaesthesia risk
  CNBP:    '#2e7d32',  // deep green   — DM2 proximal myalgia
  CLCN1:   '#4a148c',  // deep purple  — Myotonia Congenita warm-up
  SCN4A:   '#e65100',  // deep orange  — Paramyotonia cold paradox
  CACNA1S: '#880e4f',  // deep pink    — HypoPP1 carbohydrate trigger
  KCNJ2:   '#b71c1c',  // deep red     — ATS triad cardiac
  RYR1:    '#37474f',  // dark grey    — MHS dantrolene life-saving
  ATP2A1:  '#00695c',  // deep teal    — Brody silent EMG
};

const GENE_INFO = {
  DMPK:    { full: 'DMPK / 639aa',    locus: '19q13.32', size: '639 aa',  inh: 'AD',    disease: 'DM1/Steinert — CTG repeat >50 / SUCCINYLCHOLINE ABSOLUTELY CI / Volatile anaesthetics EXTREME CAUTION / Annual Holter mandatory / Anticipation maternal bias congenital' },
  CNBP:    { full: 'CNBP / 347aa',    locus: '3q21.3',   size: '347 aa',  inh: 'AD',    disease: 'DM2/PROMM — CCTG repeat / PROXIMAL weakness DDx DM1 distal / Myalgia misdiagnosed fibromyalgia / NO congenital form / Same anaesthetic risk as DM1' },
  CLCN1:   { full: 'CLCN1 / 988aa',   locus: '7q34',     size: '988 aa',  inh: 'AD/AR', disease: 'Myotonia Congenita Thomsen AD / Becker AR — WARM-UP PHENOMENON PATHOGNOMONIC / NO systemic features / NO cardiac / Mexiletine FIRST-LINE level A' },
  SCN4A:   { full: 'SCN4A / 1836aa',  locus: '17q23.3',  size: '1836 aa', inh: 'AD',    disease: 'PMC/HyperPP2 — COLD WORSENS + Warm-up FAILS PATHOGNOMONIC DDx CLCN1 / Potassium triggers paralysis / Mexiletine PMC / Acetazolamide HyperPP2' },
  CACNA1S: { full: 'CACNA1S / 1873aa',locus: '1q32.1',   size: '1873 aa', inh: 'AD',    disease: 'HypoPP1 — CARBOHYDRATE + REST after exercise triggers PATHOGNOMONIC / GLUCOSE IV ABSOLUTELY CI / Acetazolamide FIRST-LINE / Allelic MHS5 anaesthetic risk' },
  KCNJ2:   { full: 'KCNJ2 / 427aa',   locus: '17q24.3',  size: '427 aa',  inh: 'AD',    disease: 'Andersen-Tawil Syndrome ATS/LQT7 — TRIAD: paralysis + bidirectional VT + dysmorphic PATHOGNOMONIC / Flecainide / ICD mandatory evaluation / QT-prolonging drugs CI' },
  RYR1:    { full: 'RYR1 / 5038aa',   locus: '19q13.2',  size: '5038 aa', inh: 'AR/AD', disease: 'MHS1/CCD — ALL VOLATILE ANAESTHETICS + SUCCINYLCHOLINE ABSOLUTELY CI / DANTROLENE 26 vials LIFE-SAVING mandatory / TIVA propofol only / Central Core Disease AR' },
  ATP2A1:  { full: 'ATP2A1 / 994aa',  locus: '16p11.2',  size: '994 aa',  inh: 'AR',    disease: 'Brody Myopathy — SILENT MYOTONIA EMG ELECTRICALLY SILENT despite stiffness PATHOGNOMONIC / DDx CLCN1 SCN4A / Verapamil anecdotal / SERCA1 Ca2+-pump deficiency' },
};

export default function HereditaryMyotonicDisorderAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = [
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ];
    Promise.all(endpoints)
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#555' }}>Loading Hereditary Myotonic Disorder Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;
  if (!overview) return null;

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui,sans-serif', maxWidth: 1200 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, color: '#1a237e', marginBottom: 4 }}>
        🧬 Hereditary Myotonic Disorder Atlas
      </h1>
      <p style={{ color: '#555', fontSize: 13, marginBottom: 18 }}>
        Complete 8-Gene Reference — DM1 · DM2 · Myotonia Congenita · Paramyotonia · HypoPP1 · ATS · MHS/CCD · Brody
        &nbsp;|&nbsp; {overview.total_patients} patients · Seeds {overview.seeds}
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #e3e8f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '6px 6px 0 0',
            background: tab === t ? '#1a237e' : '#f0f4ff',
            color: tab === t ? '#fff' : '#333',
            fontWeight: tab === t ? 700 : 400, cursor: 'pointer', fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && (
        <div>
          {/* Summary cards */}
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Alive', value: `${overview.alive_pct}%` },
              { label: 'Female', value: `${overview.female_pct}%` },
              { label: 'Treated', value: `${overview.treated_pct}%` },
              { label: 'Avg Age', value: overview.avg_age },
              { label: 'Avg Episodes/mo', value: overview.avg_attacks_per_month },
            ].map(c => (
              <div key={c.label} style={{
                background: '#f0f4ff', borderRadius: 8, padding: '14px 20px',
                minWidth: 110, textAlign: 'center', border: '1px solid #dce3f7',
              }}>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#1a237e' }}>{c.value}</div>
                <div style={{ fontSize: 11, color: '#555', marginTop: 2 }}>{c.label}</div>
              </div>
            ))}
          </div>

          {/* Clinical axioms */}
          <div style={{ background: '#fff8e1', borderRadius: 8, padding: '16px 20px', marginBottom: 24, border: '1px solid #ffe082' }}>
            <div style={{ fontWeight: 700, fontSize: 13, color: '#e65100', marginBottom: 8 }}>⚡ Clinical Axioms</div>
            {overview.clinical_axioms.map((a, i) => (
              <div key={i} style={{ fontSize: 12, color: '#333', marginBottom: 4, paddingLeft: 8 }}>• {a}</div>
            ))}
          </div>

          {/* Gene summary cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 14 }}>
            {Object.entries(overview.gene_summaries || {}).map(([gene, g]) => (
              <div key={gene} style={{
                border: `2px solid ${GENE_COLORS[gene] || '#999'}`,
                borderRadius: 8, padding: 14, background: '#fff',
              }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gene], fontSize: 15, marginBottom: 4 }}>{gene}</div>
                <div style={{ fontSize: 11, color: '#555', marginBottom: 8 }}>
                  {GENE_INFO[gene]?.locus} · {GENE_INFO[gene]?.size} · {GENE_INFO[gene]?.inh}
                </div>
                <div style={{ fontSize: 11, color: '#333', marginBottom: 8, fontStyle: 'italic' }}>
                  {GENE_INFO[gene]?.disease}
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {[
                    { label: 'n', value: g.n_patients },
                    { label: 'alive', value: `${g.alive_pct}%` },
                    { label: 'treated', value: `${g.treated_pct}%` },
                    { label: 'age', value: g.avg_age },
                  ].map(s => (
                    <span key={s.label} style={{
                      background: '#f5f5f5', borderRadius: 4, padding: '2px 6px',
                      fontSize: 11, color: '#333',
                    }}>{s.label}: <b>{s.value}</b></span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene', 'Protein / Size', 'Locus', 'Inh', 'Patients', 'Alive%', 'Treated%', 'Avg Age', 'Episodes/mo', 'Disease / Key Pearl'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown || {}).map(([gene, g], idx) => (
                <tr key={gene} style={{ background: idx % 2 === 0 ? '#f8f9ff' : '#fff', borderBottom: '1px solid #e3e8f0' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                  <td style={{ padding: '8px 10px' }}>{GENE_INFO[gene]?.full}</td>
                  <td style={{ padding: '8px 10px' }}>{g.locus}</td>
                  <td style={{ padding: '8px 10px' }}>{GENE_INFO[gene]?.inh}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center' }}>{g.n_patients}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center' }}>{g.alive_pct}%</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center' }}>{g.treated_pct}%</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center' }}>{g.avg_age}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center' }}>{g.avg_attacks_per_month}</td>
                  <td style={{ padding: '8px 10px', maxWidth: 280 }}>
                    <div style={{ fontStyle: 'italic', color: '#444' }}>{GENE_INFO[gene]?.disease?.slice(0, 120)}…</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && (
        <div>
          {Object.entries(breakdown || {}).map(([gene, g]) => (
            <div key={gene} style={{
              border: `2px solid ${GENE_COLORS[gene] || '#999'}`,
              borderRadius: 8, marginBottom: 20, overflow: 'hidden',
            }}>
              <div style={{ background: GENE_COLORS[gene], color: '#fff', padding: '10px 16px', fontWeight: 700, fontSize: 14 }}>
                {gene} — {GENE_INFO[gene]?.full} &nbsp;|&nbsp; {g.locus} &nbsp;|&nbsp; {GENE_INFO[gene]?.inh}
              </div>
              <div style={{ padding: 16, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#1a237e', marginBottom: 6 }}>🔑 Key Features</div>
                  {g.key_features?.map((f, i) => <div key={i} style={{ fontSize: 11, color: '#333', marginBottom: 3 }}>• {f}</div>)}
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#2e7d32', marginBottom: 6 }}>💊 Treatment</div>
                  {g.treatment?.map((t, i) => <div key={i} style={{ fontSize: 11, color: '#333', marginBottom: 3 }}>• {t}</div>)}
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#b71c1c', marginBottom: 6 }}>⛔ Contraindications</div>
                  {g.contraindications?.map((c, i) => <div key={i} style={{ fontSize: 11, color: '#b71c1c', marginBottom: 3 }}>• {c}</div>)}
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#e65100', marginBottom: 6, marginTop: 8 }}>💡 Critical Pearls</div>
                  {g.critical_pearls?.map((p, i) => <div key={i} style={{ fontSize: 11, color: '#444', marginBottom: 3 }}>• {p}</div>)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          {Object.entries(definitions).map(([section, entries]) => (
            <div key={section} style={{ background: '#f8f9ff', borderRadius: 8, padding: 16, border: '1px solid #e3e8f0' }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#1a237e', marginBottom: 10, textTransform: 'capitalize' }}>
                {section.replace(/_/g, ' ')}
              </div>
              {Object.entries(entries).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: 11, color: '#333' }}>{k.replace(/_/g, ' ')}: </span>
                  <span style={{ fontSize: 11, color: '#555' }}>{typeof v === 'string' ? v : JSON.stringify(v)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
