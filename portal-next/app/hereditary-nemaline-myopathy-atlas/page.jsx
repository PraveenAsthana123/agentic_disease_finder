'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-nemaline-myopathy-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  NEB:     '#1565c0',  // deep blue    — most common AR NM, nebulin giant protein
  ACTA1:   '#b71c1c',  // deep red     — de novo dominant, intranuclear rods, severe
  TPM2:    '#2e7d32',  // deep green   — allelic: NM + Cap Disease + DA2
  TPM3:    '#388e3c',  // green        — slow fiber selective, fiber type disproportion
  TNNT1:   '#e65100',  // deep orange  — Amish E180X lethal infantile founder
  CFL2:    '#6a1b9a',  // deep purple  — cores and rods, CFL2-specific biopsy
  KBTBD13: '#00695c',  // teal         — slow relaxation, electrically silent, Dutch founder
  LMOD3:   '#4a148c',  // dark purple  — severe congenital, DCM possible, thin filament elongation
};

const GENE_INFO = {
  NEB:     { full: 'NEB / 6669aa',     locus: '2q23.3',   size: '6669 aa / ~800 kDa', inh: 'AR',         disease: 'Typical/Severe NM — NEMALINE RODS Gomori PATHOGNOMONIC / most common AR NM 50% / NEB exon55 deletion Ashkenazi founder / CK normal–3× / respiratory failure critical' },
  ACTA1:   { full: 'ACTA1 / 375aa',   locus: '1q42.13',  size: '375 aa / 42 kDa',   inh: 'AD-de-novo/AR', disease: 'Typical to Severe/Congenital NM — INTRANUCLEAR RODS ACTA1-PATHOGNOMONIC / de novo dominant most severe / 2nd most common NM gene / salbutamol Level C / respiratory from birth' },
  TPM2:    { full: 'TPM2 / 284aa',    locus: '9p13.3',   size: '284 aa / 33 kDa',   inh: 'AD/AR',        disease: 'NM type4 + CAP DISEASE + DA2A/DA2B ALLELIC — GOF→Distal Arthrogryposis / LOF→Nemaline / CAP DISEASE thin filament caps biopsy PATHOGNOMONIC / fiber type disproportion' },
  TPM3:    { full: 'TPM3 / 285aa',    locus: '1q21.2',   size: '285 aa / 33 kDa',   inh: 'AD/AR',        disease: 'Typical NM type1 — SLOW TWITCH FIBER SELECTIVE / AD cap disease / AR typical NM / fiber type disproportion overlap / CK NORMAL / slower progression / exercise intolerance' },
  TNNT1:   { full: 'TNNT1 / 328aa',   locus: '19q13.42', size: '328 aa / 36 kDa',   inh: 'AR',           disease: 'Nemaline Myopathy Amish — E180X AMISH FOUNDER LETHAL INFANTILE / FATAL BY 2-3 YEARS Amish homozygous / E180K non-Amish milder / palliative care discussions critical' },
  CFL2:    { full: 'CFL2 / 166aa',    locus: '14q13.1',  size: '166 aa / 19 kDa',   inh: 'AR',           disease: 'NM type4 Cofilin-2 — CORES AND RODS biopsy PATHOGNOMONIC CFL2 / actin-depolymerizing factor / proximal > distal / respiratory moderate / childhood–adult onset' },
  KBTBD13: { full: 'KBTBD13 / 571aa', locus: '15q22.31', size: '571 aa / 64 kDa',   inh: 'AD',           disease: 'NM type6 — SLOW RELAXATION ELECTRICALLY SILENT PATHOGNOMONIC / NOT myotonia on EMG / EXERCISE INTOLERANCE / Dutch founder / CK NORMAL / mild course ambulant adulthood' },
  LMOD3:   { full: 'LMOD3 / 547aa',   locus: '3p14.1',   size: '547 aa / 64 kDa',   inh: 'AR',           disease: 'NM type10 — Thin Filament Elongation Factor / SEVERE CONGENITAL / DCM POSSIBLE mandatory echo / RODS+MINIMAL FILAMENTS biopsy / respiratory failure birth' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px', fontSize: 11, fontWeight: 700, marginRight: 4,
    }}>{text}</span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: '#fff', border: `2px solid ${color || '#e0e0e0'}`,
      borderRadius: 10, padding: '14px 18px', minWidth: 120, textAlign: 'center',
    }}>
      <div style={{ fontSize: 26, fontWeight: 800, color: color || '#333' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#888' }}>{sub}</div>}
    </div>
  );
}

function GeneCard({ gene, color, info, data }) {
  return (
    <div style={{
      border: `2px solid ${color}`, borderRadius: 10, padding: 16, marginBottom: 12,
      background: color + '08',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{ fontWeight: 800, fontSize: 18, color }}>{gene}</span>
        <Badge text={info.locus} color={color} />
        <Badge text={info.inh} color={color} />
        <Badge text={info.size} color="#555" />
      </div>
      <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6 }}>{info.disease}</div>
      {data && (
        <div style={{ display: 'flex', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: '#555' }}>Ambulant: <b>{data.ambulant_pct}%</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>NIV: <b>{data.niv_pct}%</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Cardiac: <b>{data.cardiac_pct}%</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Avg Onset: <b>{data.avg_onset}yr</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Avg CK: <b>{data.avg_ck} IU/L</b></span>
        </div>
      )}
    </div>
  );
}

export default function HNMAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, df]) => {
      setOverview(ov);
      setDefs(df);
      setLoading(false);
    }).catch(e => {
      setError(e.message);
      setLoading(false);
    });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#1565c0' }}>Loading Hereditary Nemaline Myopathy Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const kpis = overview?.kpis || {};
  const geneSummary = overview?.gene_summary || {};
  const pathognomonic = overview?.pathognomonic_features || {};
  const critical = overview?.critical_treatments || {};
  const distinctions = overview?.key_distinctions || [];

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto', padding: 24 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #1565c0 0%, #b71c1c 100%)', borderRadius: 14, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 11, letterSpacing: 2, opacity: 0.8, marginBottom: 4 }}>HEREDITARY DISEASE ATLAS</div>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800 }}>Hereditary Nemaline Myopathy Atlas</h1>
        <div style={{ fontSize: 14, opacity: 0.9, marginTop: 6 }}>
          Complete 8-Gene Nemaline Myopathy (NM) Spectrum Atlas · NEB · ACTA1 · TPM2 · TPM3 · TNNT1 · CFL2 · KBTBD13 · LMOD3
        </div>
        <div style={{ fontSize: 12, opacity: 0.75, marginTop: 4 }}>
          320-Patient Aggregate Cohort (8 × 40) · Seeds 2222–2229 · Thin Filament Myopathy Reference
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #e0e0e0', paddingBottom: 0 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '10px 20px', border: 'none', background: tab === i ? '#1565c0' : 'transparent',
            color: tab === i ? '#fff' : '#555', fontWeight: tab === i ? 700 : 400,
            borderRadius: '8px 8px 0 0', cursor: 'pointer', fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      {/* ── Overview ── */}
      {tab === 0 && (
        <div>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
            <StatCard label="Total Patients" value={kpis.total_patients} color="#1565c0" sub="8 × 40 cohort" />
            <StatCard label="Genes" value={overview?.gene_count} color="#b71c1c" sub="NEB to LMOD3" />
            <StatCard label="Ambulant" value={`${kpis.ambulant_pct}%`} color="#2e7d32" sub="walking" />
            <StatCard label="NIV / Vent" value={`${kpis.niv_pct}%`} color="#e65100" sub="respiratory support" />
            <StatCard label="Cardiac" value={`${kpis.cardiac_involvement_pct}%`} color="#6a1b9a" sub="involvement" />
            <StatCard label="Avg Onset" value={`${kpis.avg_onset_years}yr`} color="#00695c" sub="across cohort" />
            <StatCard label="Avg CK" value={`${kpis.avg_ck_iul}`} color="#4a148c" sub="IU/L — mostly normal" />
          </div>

          {/* Biopsy signature badges */}
          <div style={{ background: '#e8f5e9', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#2e7d32', marginBottom: 10, fontSize: 14 }}>
              ⚗ Biopsy Signature Features (8-Gene NM Spectrum)
            </div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              <span style={{ background: '#fff', border: '1px solid #2e7d32', borderRadius: 6, padding: '4px 10px', fontSize: 12 }}>
                Nemaline Rods: <b>{kpis.niv_pct}% NIV (respiratory)</b> — all genes
              </span>
              <span style={{ background: '#fff', border: '1px solid #b71c1c', borderRadius: 6, padding: '4px 10px', fontSize: 12 }}>
                Intranuclear Rods: <b>{kpis.intranuclear_rods_pct}%</b> — ACTA1 specific
              </span>
              <span style={{ background: '#fff', border: '1px solid #6a1b9a', borderRadius: 6, padding: '4px 10px', fontSize: 12 }}>
                Cores+Rods: <b>{kpis.cores_and_rods_pct}%</b> — CFL2 specific
              </span>
              <span style={{ background: '#fff', border: '1px solid #00695c', borderRadius: 6, padding: '4px 10px', fontSize: 12 }}>
                Slow Relaxation: <b>{kpis.slow_relaxation_pct}%</b> — KBTBD13 specific
              </span>
            </div>
          </div>

          {/* Pathognomonic features */}
          <div style={{ background: '#fff3e0', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#e65100', marginBottom: 10, fontSize: 14 }}>
              ⚡ Pathognomonic Features by Gene
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 8 }}>
              {Object.entries(pathognomonic).map(([gene, feat]) => (
                <div key={gene} style={{
                  background: '#fff', border: `1px solid ${GENE_COLORS[gene]}44`,
                  borderRadius: 8, padding: '8px 12px',
                }}>
                  <span style={{ fontWeight: 700, color: GENE_COLORS[gene], marginRight: 8 }}>{gene}</span>
                  <span style={{ fontSize: 12, color: '#333' }}>{feat}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Key distinctions */}
          <div style={{ background: '#e8eaf6', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 10, fontSize: 14 }}>
              🔑 Key Clinical Distinctions
            </div>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {distinctions.map((d, i) => (
                <li key={i} style={{ fontSize: 13, color: '#333', marginBottom: 4, lineHeight: 1.5 }}>{d}</li>
              ))}
            </ul>
          </div>

          {/* Critical treatments */}
          <div style={{ background: '#fce4ec', borderRadius: 10, padding: 16 }}>
            <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 10, fontSize: 14 }}>
              💊 Critical Treatment Priorities
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 8 }}>
              {Object.entries(critical).map(([gene, tx]) => (
                <div key={gene} style={{
                  background: '#fff', border: `1px solid ${GENE_COLORS[gene]}44`,
                  borderRadius: 8, padding: '8px 12px',
                }}>
                  <span style={{ fontWeight: 700, color: GENE_COLORS[gene], marginRight: 8 }}>{gene}</span>
                  <span style={{ fontSize: 12, color: '#333' }}>{tx}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Gene Table ── */}
      {tab === 1 && (
        <div>
          <h2 style={{ fontSize: 18, color: '#1565c0', marginBottom: 16 }}>8-Gene NM Reference Table</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#1565c0', color: '#fff' }}>
                  {['Gene', 'Locus', 'Size', 'Inheritance', 'Key Disease / Phenotype', 'Pathognomonic', 'Ambulant%', 'NIV%', 'Cardiac%'].map(h => (
                    <th key={h} style={{ padding: '10px 8px', textAlign: 'left', fontWeight: 700 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(GENE_INFO).map(([gene, info], idx) => {
                  const gdata = geneSummary[gene] || {};
                  const color = GENE_COLORS[gene];
                  return (
                    <tr key={gene} style={{ background: idx % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '8px', fontWeight: 800, color }}>{gene}</td>
                      <td style={{ padding: '8px', color: '#555' }}>{info.locus}</td>
                      <td style={{ padding: '8px', color: '#555' }}>{info.size}</td>
                      <td style={{ padding: '8px' }}><Badge text={info.inh} color={color} /></td>
                      <td style={{ padding: '8px', color: '#333', maxWidth: 280, fontSize: 12, lineHeight: 1.4 }}>{info.disease}</td>
                      <td style={{ padding: '8px', color: '#555', fontSize: 12 }}>{pathognomonic[gene] || '—'}</td>
                      <td style={{ padding: '8px', textAlign: 'center', color: '#2e7d32', fontWeight: 700 }}>{gdata.ambulant_pct ?? '—'}%</td>
                      <td style={{ padding: '8px', textAlign: 'center', color: '#e65100', fontWeight: 700 }}>{gdata.niv_pct ?? '—'}%</td>
                      <td style={{ padding: '8px', textAlign: 'center', color: '#b71c1c', fontWeight: 700 }}>{gdata.cardiac_pct ?? '—'}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 24 }}>
            <h3 style={{ color: '#1565c0', fontSize: 16 }}>Inheritance & Cohort Summary</h3>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {Object.entries(overview?.inheritance_map || {}).map(([gene, inh]) => (
                <div key={gene} style={{
                  background: GENE_COLORS[gene] + '11', border: `1px solid ${GENE_COLORS[gene]}44`,
                  borderRadius: 8, padding: '6px 14px', fontSize: 13,
                }}>
                  <span style={{ fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</span>
                  <span style={{ color: '#555', marginLeft: 6 }}>{inh}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Clinical Atlas ── */}
      {tab === 2 && (
        <div>
          <h2 style={{ fontSize: 18, color: '#1565c0', marginBottom: 16 }}>Clinical Atlas — Per-Gene Profiles</h2>
          {Object.entries(GENE_INFO).map(([gene, info]) => (
            <GeneCard
              key={gene}
              gene={gene}
              color={GENE_COLORS[gene]}
              info={info}
              data={geneSummary[gene]}
            />
          ))}

          {/* Diagnostic algorithm */}
          <div style={{ background: '#e8f5e9', borderRadius: 10, padding: 16, marginTop: 20 }}>
            <div style={{ fontWeight: 700, color: '#2e7d32', marginBottom: 10, fontSize: 14 }}>
              🔬 Diagnostic Algorithm — Nemaline Myopathy
            </div>
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {(defs?.diagnostic_algorithm || []).map((step, i) => (
                <li key={i} style={{ fontSize: 13, color: '#333', marginBottom: 6, lineHeight: 1.5 }}>{step}</li>
              ))}
            </ol>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 3 && (
        <div>
          <h2 style={{ fontSize: 18, color: '#1565c0', marginBottom: 16 }}>Definitions & Standards</h2>
          {Object.entries(defs?.glossary || {}).map(([term, def]) => (
            <div key={term} style={{
              background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
              padding: '14px 18px', marginBottom: 10,
            }}>
              <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 4 }}>{term}</div>
              <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6 }}>{def}</div>
            </div>
          ))}

          <div style={{ background: '#e3f2fd', borderRadius: 10, padding: 16, marginTop: 16 }}>
            <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 10 }}>📚 References</div>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs?.references || []).map((ref, i) => (
                <li key={i} style={{ fontSize: 12, color: '#333', marginBottom: 4 }}>{ref}</li>
              ))}
            </ul>
          </div>

          <div style={{ background: '#f3e5f5', borderRadius: 10, padding: 16, marginTop: 12 }}>
            <div style={{ fontWeight: 700, color: '#6a1b9a', marginBottom: 10 }}>📋 Standards</div>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs?.standards || []).map((std, i) => (
                <li key={i} style={{ fontSize: 12, color: '#333', marginBottom: 4 }}>{std}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
