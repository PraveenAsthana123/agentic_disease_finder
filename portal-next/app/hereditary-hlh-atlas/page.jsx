'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hlh-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PRF1:   '#b71c1c',  // deep red     — FHL2, most common genetic FHL (~30%), perforin zero
  UNC13D: '#1565c0',  // deep blue    — FHL3, ~25%, Munc13-4, MAS in JIA
  STX11:  '#0d47a1',  // navy         — FHL4, Kurdish/Turkish founder, normal CD107a paradox
  STXBP2: '#283593',  // dark indigo  — FHL5, earliest onset, IBD pathognomonic
  RAB27A: '#2e7d32',  // deep green   — GS2, silver hair + HLH, Rab27A melanosome/granule
  AP3B1:  '#6a1b9a',  // deep purple  — HPS-2, albinism + neutropenia + absent dense granules
  SH2D1A: '#e65100',  // burnt orange — XLP-1, EBV-selective, SAP, boys only, HSCT pre-EBV
  XIAP:   '#004d40',  // dark teal    — XLP-2, IBD + HLH, not EBV-exclusive, splenomegaly
};

const GENE_INFO = {
  PRF1:   { full: 'PRF1/Perforin-1 / 555aa', locus: '10q22.1', size: '555 aa / 67 kDa', inh: 'AR',    disease: 'FHL2 — MOST COMMON FHL (~30%); perforin pore absent; NK cytotoxicity ZERO; CD107a impaired; HLH-2004: dex + etoposide + ciclosporin; emapalumab for refractory; HSCT mandatory and curative' },
  UNC13D: { full: 'UNC13D/Munc13-4 / 2090aa', locus: '17q25.1', size: '2090 aa / 237 kDa', inh: 'AR', disease: 'FHL3 — 2nd most common FHL (~25%); Munc13-4 priming failure; CD107a ABSENT (key diagnostic); most common FHL gene in MAS-associated JIA; Kurdish/Turkish founder; HSCT mandatory' },
  STX11:  { full: 'STX11/Syntaxin-11 / 287aa', locus: '6q24.2', size: '287 aa / 33 kDa', inh: 'AR',   disease: 'FHL4 — Kurdish/Turkish founder pThr265Ile; UNIQUE: CD107a NORMAL but NK cytotoxicity ZERO — paradox; final SNARE fusion failure; must test cytotoxicity assay separately; HSCT mandatory' },
  STXBP2: { full: 'STXBP2/Munc18-2 / 593aa', locus: '19p13.2', size: '593 aa / 67 kDa', inh: 'AR',   disease: 'FHL5 — EARLIEST FHL onset (neonatal/prenatal); IBD (Crohn\'s-like colitis) PATHOGNOMONIC for FHL5; Munc18-2 stabilises STX11; HSCT mandatory; hypomorphic → adult IBD without HLH' },
  RAB27A: { full: 'RAB27A/Rab-27A / 221aa', locus: '15q21.3', size: '221 aa / 26 kDa', inh: 'AR',     disease: 'Griscelli Syndrome type 2 (GS2) — silver-grey hair FROM BIRTH + episodic HLH = PATHOGNOMONIC; large irregular melanin clumps on polarised hair microscopy (DDx GS1/GS3); perforin NORMAL (DDx FHL2); HSCT corrects HLH, NOT hair colour' },
  AP3B1:  { full: 'AP3B1/AP3-beta-1 / 1094aa', locus: '5q14.1', size: '1094 aa / 123 kDa', inh: 'AR', disease: 'HPS-2 — albinism + NEUTROPENIA (unique to HPS-2) + absent platelet dense granules TRIAD; EM shows dense granule absent; recurrent HLH from NK defect; NO ASPIRIN ever; DDAVP pre-procedure; HSCT for severe HLH' },
  SH2D1A: { full: 'SH2D1A/SAP / 128aa', locus: 'Xq25', size: '128 aa / 15 kDa', inh: 'XLR',          disease: 'XLP-1/Duncan Disease — EBV-SELECTIVE HLH; boys only (XLR); fulminant IM on first EBV exposure (60% fatal without HSCT); NKT cells ABSENT; dysgammaglobulinaemia post-EBV; lymphoma 30%; HSCT BEFORE EBV exposure mandatory' },
  XIAP:   { full: 'XIAP/BIRC4 / 497aa', locus: 'Xq25', size: '497 aa / 57 kDa', inh: 'XLR',           disease: 'XLP-2 — NOT EBV-exclusive; IBD (Crohn\'s-like colitis) + HLH PATHOGNOMONIC; splenomegaly >80%; NKT cells normal (DDx XLP-1); HSCT corrects BOTH HLH and IBD; live vaccines trigger HLH — AVOID' },
};

function GeneChip({ gene }) {
  return (
    <span style={{
      background: GENE_COLORS[gene] || '#555',
      color: '#fff',
      borderRadius: 4,
      padding: '2px 8px',
      fontSize: 12,
      fontWeight: 700,
      marginRight: 4,
      display: 'inline-block',
    }}>{gene}</span>
  );
}

function MetricCard({ label, value, sub }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 140, flex: '1 1 140px' }}>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#f1f5f9', fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HLHAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' || tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true);
    setError(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => {
        if (ep === 'overview') setOverview(d);
        else if (ep === 'breakdown') setBreakdown(d);
        else setDefinitions(d);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui,sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#7f1d1d,#1e1b4b)', padding: '24px 32px' }}>
        <div style={{ fontSize: 11, color: '#fca5a5', letterSpacing: 2, marginBottom: 6 }}>
          HEREDITARY DISEASE ATLAS · HAEMATOLOGICAL IMMUNOLOGY
        </div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#fff' }}>
          Hereditary-HLH-Lymphohistiocytosis-Atlas
        </h1>
        <div style={{ color: '#cbd5e1', marginTop: 6, fontSize: 14 }}>
          Complete 8-Gene Familial HLH & X-Linked Lymphoproliferative Disease Reference
          &nbsp;·&nbsp; 320 Patients · Seeds 2334–2341
        </div>
        {/* Gene chips */}
        <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {Object.keys(GENE_INFO).map(g => <GeneChip key={g} gene={g} />)}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #1e293b', background: '#0f172a', padding: '0 32px' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: 'none', border: 'none', color: tab === t ? '#f87171' : '#64748b',
            borderBottom: tab === t ? '2px solid #f87171' : '2px solid transparent',
            padding: '12px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
            fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '28px 32px', maxWidth: 1200 }}>
        {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
        {error && <div style={{ color: '#f87171' }}>Error: {error}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <h2 style={{ color: '#f8fafc', marginTop: 0 }}>{overview.atlas}</h2>
            <p style={{ color: '#94a3b8' }}>{overview.subtitle}</p>

            {/* Aggregate metrics */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 28 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40 cohort" />
              <MetricCard label="Avg Peak Ferritin" value={`${(overview.aggregate_metrics?.avg_ferritin_ug_L / 1000).toFixed(0)}k µg/L`} sub="hyperferritinaemia" />
              <MetricCard label="HSCT Completed" value={`${overview.aggregate_metrics?.hsct_completed_pct}%`} sub="curative in all FHL" />
              <MetricCard label="Haemophagocytosis" value={`${overview.aggregate_metrics?.haemophagocytosis_confirmed_pct}%`} sub="on BM biopsy" />
              <MetricCard label="EBV Triggered" value={`${overview.aggregate_metrics?.ebv_triggered_pct}%`} sub="all gene cohorts" />
              <MetricCard label="IBD Present" value={`${overview.aggregate_metrics?.ibd_present_pct}%`} sub="FHL5 + XLP-2 enriched" />
              <MetricCard label="Albinism Present" value={`${overview.aggregate_metrics?.albinism_present_pct}%`} sub="GS2 + HPS-2" />
            </div>

            {/* Gene summary cards */}
            <h3 style={{ color: '#cbd5e1', borderBottom: '1px solid #1e293b', paddingBottom: 8 }}>Per-Gene Summary</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {Object.entries(overview.gene_summary || {}).map(([gene, info]) => (
                <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <GeneChip gene={gene} />
                    <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{info.hlh_category}</span>
                    <span style={{ color: '#64748b', fontSize: 12 }}>· {info.locus} · {info.protein_size} · {info.inheritance}</span>
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: 13, marginBottom: 8 }}>{info.pathognomonic}</div>
                  <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, color: '#64748b' }}>
                    <span>Avg ferritin: <b style={{ color: '#f59e0b' }}>{(info.avg_ferritin / 1000).toFixed(0)}k µg/L</b></span>
                    <span>HSCT: <b style={{ color: '#34d399' }}>{info.hsct_rate_pct}%</b></span>
                    <span>Albinism: <b style={{ color: '#a78bfa' }}>{info.albinism_pct}%</b></span>
                    <span>IBD: <b style={{ color: '#fb923c' }}>{info.ibd_pct}%</b></span>
                    <span>EBV trigger: <b style={{ color: '#60a5fa' }}>{info.ebv_trigger_pct}%</b></span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <h2 style={{ color: '#f8fafc', marginTop: 0 }}>Gene Reference Table</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                    {['Gene', 'Locus', 'Size', 'Inh', 'Subtype', 'Key Pathognomonic', 'Albinism', 'IBD', 'EBV%', 'HSCT%'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.gene_breakdowns || []).map((row, i) => (
                    <tr key={row.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b', borderBottom: '1px solid #334155' }}>
                      <td style={{ padding: '9px 12px' }}><GeneChip gene={row.gene} /></td>
                      <td style={{ padding: '9px 12px', color: '#94a3b8' }}>{row.locus}</td>
                      <td style={{ padding: '9px 12px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{row.protein_size}</td>
                      <td style={{ padding: '9px 12px', color: '#fbbf24' }}>{row.inheritance?.split(';')[0]?.split('(')[0]?.trim()}</td>
                      <td style={{ padding: '9px 12px', color: '#e2e8f0', maxWidth: 160 }}>{row.hlh_category?.split('—')[0]?.trim()}</td>
                      <td style={{ padding: '9px 12px', color: '#94a3b8', maxWidth: 240, fontSize: 12 }}>{row.pathognomonic?.slice(0, 120)}…</td>
                      <td style={{ padding: '9px 12px', color: row.albinism_pct > 50 ? '#a78bfa' : '#475569' }}>{row.albinism_pct > 50 ? '✓' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: row.ibd_pct > 30 ? '#fb923c' : '#475569' }}>{row.ibd_pct > 30 ? '✓' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: '#60a5fa' }}>{row.ebv_trigger_pct}%</td>
                      <td style={{ padding: '9px 12px', color: '#34d399' }}>{row.hsct_completed_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <h2 style={{ color: '#f8fafc', marginTop: 0 }}>Clinical Atlas — Per-Gene Detail</h2>
            {(breakdown.gene_breakdowns || []).map(entry => (
              <div key={entry.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 20, borderLeft: `5px solid ${GENE_COLORS[entry.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <GeneChip gene={entry.gene} />
                  <span style={{ color: '#f1f5f9', fontWeight: 700, fontSize: 16 }}>
                    {GENE_INFO[entry.gene]?.full}
                  </span>
                  <span style={{ color: '#64748b', fontSize: 13 }}>· {entry.inheritance?.split(';')[0]}</span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                  <div>
                    <div style={{ color: '#64748b', fontSize: 11, marginBottom: 3 }}>HLH SUBTYPE</div>
                    <div style={{ color: '#fbbf24', fontWeight: 600 }}>{entry.hlh_category}</div>
                  </div>
                  <div>
                    <div style={{ color: '#64748b', fontSize: 11, marginBottom: 3 }}>ONSET</div>
                    <div style={{ color: '#e2e8f0' }}>{entry.onset_age}</div>
                  </div>
                </div>

                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                  <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>PATHOGNOMONIC / DIAGNOSTIC</div>
                  <div style={{ color: '#94a3b8', fontSize: 13 }}>{entry.pathognomonic}</div>
                </div>

                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                  <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>TREATMENT</div>
                  <div style={{ color: '#86efac', fontSize: 13 }}>{entry.treatment}</div>
                </div>

                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                  <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>KEY DDx</div>
                  <div style={{ color: '#fca5a5', fontSize: 13 }}>{entry.key_ddx}</div>
                </div>

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
                  {(entry.key_features || []).map((f, i) => (
                    <span key={i} style={{ background: '#1e3a5f', borderRadius: 4, padding: '3px 10px', fontSize: 12, color: '#93c5fd' }}>{f}</span>
                  ))}
                </div>

                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, color: '#64748b', marginTop: 4 }}>
                  <span>Albinism: <b style={{ color: entry.albinism_risk?.includes('PRESENT') ? '#a78bfa' : '#475569' }}>{entry.albinism_risk}</b></span>
                  <span>Bleeding: <b style={{ color: entry.bleeding_risk?.includes('HIGH') ? '#f87171' : '#94a3b8' }}>{entry.bleeding_risk}</b></span>
                  <span>EBV risk: <b style={{ color: entry.ebv_risk?.includes('EXTREME') ? '#f87171' : '#94a3b8' }}>{entry.ebv_risk?.slice(0, 50)}</b></span>
                  <span>IBD: <b style={{ color: entry.ibd_risk?.includes('HIGH') ? '#fb923c' : '#94a3b8' }}>{entry.ibd_risk}</b></span>
                </div>

                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, marginTop: 8, paddingTop: 8, borderTop: '1px solid #1e293b' }}>
                  <span style={{ color: '#64748b' }}>n={entry.n_patients} patients</span>
                  <span style={{ color: '#f59e0b' }}>Avg ferritin: {(entry.avg_ferritin_ug_L / 1000).toFixed(0)}k µg/L</span>
                  <span style={{ color: '#34d399' }}>HSCT: {entry.hsct_completed_pct}%</span>
                  <span style={{ color: '#60a5fa' }}>EBV trigger: {entry.ebv_trigger_pct}%</span>
                  <span style={{ color: '#fb923c' }}>IBD: {entry.ibd_pct}%</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            <h2 style={{ color: '#f8fafc', marginTop: 0 }}>HLH Clinical Definitions & Glossary</h2>

            {/* Gene definitions */}
            <h3 style={{ color: '#cbd5e1' }}>Gene Definitions</h3>
            {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
              <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <GeneChip gene={gene} />
                  <span style={{ color: '#f1f5f9', fontWeight: 600 }}>{entry.disease_name}</span>
                  <span style={{ color: '#64748b', fontSize: 12 }}>· {entry.locus} · {entry.protein_size} · {entry.inheritance}</span>
                </div>
                <div style={{ color: '#94a3b8', fontSize: 13, marginBottom: 6 }}><b style={{ color: '#fbbf24' }}>Pathognomonic:</b> {entry.pathognomonic?.slice(0, 250)}…</div>
                <div style={{ color: '#86efac', fontSize: 13, marginBottom: 6 }}><b>Treatment:</b> {entry.treatment}</div>
                <div style={{ color: '#64748b', fontSize: 12 }}><b>DDx:</b> {entry.key_ddx}</div>
              </div>
            ))}

            {/* Glossary */}
            <h3 style={{ color: '#cbd5e1', marginTop: 28 }}>HLH Glossary</h3>
            {Object.entries(definitions.hlh_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 10 }}>
                <div style={{ color: '#fbbf24', fontWeight: 600, marginBottom: 6 }}>{term}</div>
                <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
