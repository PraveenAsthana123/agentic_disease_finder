'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-sarcoglycanopathy-lgmd-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SGCA:   '#1565c0',  // deep blue    — α-SG LGMD-R3 most common European
  SGCB:   '#2e7d32',  // deep green   — β-SG LGMD-R4 C283Y founder
  SGCG:   '#b71c1c',  // deep red     — γ-SG LGMD-R5 most severe cardiac
  SGCD:   '#6a1b9a',  // deep purple  — δ-SG LGMD-R6 DCM 100%
  FKRP:   '#e65100',  // deep orange  — LGMD-R9 DCM 30-40% most common AR UK
  ANO5:   '#00695c',  // deep teal    — LGMD-R12 quads spared no cardiac
  TRIM32: '#37474f',  // dark grey    — LGMD-R8 sarcotubular aggregates
  TCAP:   '#880e4f',  // deep pink    — LGMD-R7 telethonin DCM rare
};

const GENE_INFO = {
  SGCA:   { full: 'SGCA / 387aa',   locus: '17q21.33', size: '387 aa',  inh: 'AR',   lgmd: 'LGMD-R3', disease: 'α-Sarcoglycanopathy — CK 10-70x ULN / childhood 5-15yr / IHC absent αSG secondary reduction βγδSG / cardiac rare / Gene therapy rAAVrh74-MCK-SGCA Phase 1-2 trials' },
  SGCB:   { full: 'SGCB / 318aa',   locus: '4q12',     size: '318 aa',  inh: 'AR',   lgmd: 'LGMD-R4', disease: 'β-Sarcoglycanopathy — C283Y founder N.Africa/M.East / DMD-like severe alleles / IHC absent βSG secondary reduction αγδSG / cardiac less common' },
  SGCG:   { full: 'SGCG / 291aa',   locus: '13q12.12', size: '291 aa',  inh: 'AR',   lgmd: 'LGMD-R5', disease: 'γ-Sarcoglycanopathy MOST SEVERE — del521T N.Africa / CK 20-100x ULN / onset 3-10yr / CARDIAC DCM every 6m echo / respiratory mandatory NIV' },
  SGCD:   { full: 'SGCD / 290aa',   locus: '5q33.3',   size: '290 aa',  inh: 'AR',   lgmd: 'LGMD-R6', disease: 'δ-Sarcoglycanopathy RAREST — DCM up to 100% PATHOGNOMONIC / CK 50-200x / DCM may PRECEDE weakness / gene panel in ALL unexplained DCM' },
  FKRP:   { full: 'FKRP / 495aa',   locus: '19q13.32', size: '495 aa',  inh: 'AR',   lgmd: 'LGMD-R9', disease: 'Fukutin-Related Protein — most common AR LGMD UK/Scandinavia / L276I founder / CARDIAC DCM 30-40% MANDATORY 6m echo / calf hypertrophy 85% / L276I/L276I mild → null/null MDC1C severe' },
  ANO5:   { full: 'ANO5 / 913aa',   locus: '11p14.3',  size: '913 aa',  inh: 'AR',   lgmd: 'LGMD-R12', disease: 'Anoctamin-5 — QUADRICEPS SPARED PATHOGNOMONIC DDx DYSF / posterior calf / R758C Dutch/Belgian / CK 5-100x / NO cardiac — critical DDx FKRP/SGCD / asymmetric onset' },
  TRIM32: { full: 'TRIM32 / 653aa', locus: '9q33.1',   size: '653 aa',  inh: 'AR',   lgmd: 'LGMD-R8', disease: 'Tripartite Motif-32 — SARCOTUBULAR AGGREGATES EM PATHOGNOMONIC / facial weakness / psychiatric features / CK 2-10x (lowest) / D487N Hutterite / BBS11 allelic' },
  TCAP:   { full: 'TCAP / 167aa',   locus: '17q12',    size: '167 aa',  inh: 'AR',   lgmd: 'LGMD-R7', disease: 'Telethonin/Titin-Cap — rarest LGMD / DCM significant / RIMMED VACUOLES biopsy / Z-disc titin-binding / Uruguay-Brazil-China founders / <50 patients worldwide' },
};

const CARDIAC_RISK = {
  SGCA: { risk: 'Low', freq: 'Annual', color: '#2e7d32' },
  SGCB: { risk: 'Low', freq: 'Annual', color: '#2e7d32' },
  SGCG: { risk: 'Moderate', freq: 'Every 6 months', color: '#e65100' },
  SGCD: { risk: 'High — DCM 100%', freq: 'Every 3-6 months', color: '#b71c1c' },
  FKRP: { risk: 'High — DCM 30-40%', freq: 'Every 6 months', color: '#b71c1c' },
  ANO5: { risk: 'None', freq: 'Not required', color: '#555' },
  TRIM32: { risk: 'Low', freq: 'Annual ECG', color: '#2e7d32' },
  TCAP: { risk: 'High — DCM significant', freq: 'Every 6 months', color: '#b71c1c' },
};

export default function HereditorySarcoglycanopathyLgmdAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#555' }}>Loading Hereditary Sarcoglycanopathy & LGMD Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;
  if (!overview) return null;

  const gs = overview.gene_summaries || {};

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui,sans-serif', maxWidth: 1200 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, color: '#1a237e', marginBottom: 4 }}>
        🧬 Hereditary Sarcoglycanopathy & LGMD Atlas
      </h1>
      <p style={{ color: '#555', fontSize: 13, marginBottom: 18 }}>
        Complete 8-Gene Reference — LGMD-R3 · R4 · R5 · R6 · R7 · R8 · R9 · R12
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
          {/* KPI row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Genes Covered', value: overview.n_genes },
              { label: 'Alive %', value: `${overview.alive_pct}%` },
              { label: 'Treated %', value: `${overview.treated_pct}%` },
            ].map(k => (
              <div key={k.label} style={{ background: '#f0f4ff', borderRadius: 10, padding: '14px 18px' }}>
                <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: '#1a237e' }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Gene cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 24 }}>
            {Object.entries(gs).map(([gene, d]) => (
              <div key={gene} style={{ background: '#fff', border: `2px solid ${GENE_COLORS[gene] || '#ccc'}`, borderRadius: 10, padding: 14 }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gene], fontSize: 15, marginBottom: 2 }}>{gene}</div>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 6 }}>{GENE_INFO[gene]?.lgmd} · {d.locus} · {d.protein_size}</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                  <div><div style={{ fontSize: 10, color: '#999' }}>Patients</div><div style={{ fontWeight: 600, fontSize: 15 }}>{d.n_patients}</div></div>
                  <div><div style={{ fontSize: 10, color: '#999' }}>Alive</div><div style={{ fontWeight: 600, fontSize: 15 }}>{d.alive_pct}%</div></div>
                  <div><div style={{ fontSize: 10, color: '#999' }}>Treated</div><div style={{ fontWeight: 600, fontSize: 15 }}>{d.treated_pct}%</div></div>
                  <div><div style={{ fontSize: 10, color: '#999' }}>Mean Onset</div><div style={{ fontWeight: 600, fontSize: 15 }}>{d.mean_onset_age}yr</div></div>
                </div>
                <div style={{ marginTop: 8, fontSize: 10, padding: '3px 6px', borderRadius: 4, display: 'inline-block',
                  background: CARDIAC_RISK[gene]?.risk === 'None' ? '#e8f5e9' : CARDIAC_RISK[gene]?.risk.startsWith('High') ? '#ffebee' : '#fff3e0',
                  color: CARDIAC_RISK[gene]?.risk === 'None' ? '#2e7d32' : CARDIAC_RISK[gene]?.risk.startsWith('High') ? '#b71c1c' : '#e65100',
                }}>
                  ❤️ {CARDIAC_RISK[gene]?.risk}
                </div>
              </div>
            ))}
          </div>

          {/* Key clinical facts */}
          <div style={{ background: '#fffde7', borderRadius: 10, padding: 16, marginBottom: 20, border: '1px solid #f9a825' }}>
            <div style={{ fontWeight: 700, fontSize: 14, color: '#f57f17', marginBottom: 10 }}>⚡ Key Clinical Facts</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {Object.entries(overview.key_clinical_facts || {}).map(([k, v]) => (
                <div key={k} style={{ fontSize: 12 }}>
                  <span style={{ fontWeight: 600, color: '#555' }}>{k.replace(/_/g, ' ')}: </span>
                  <span style={{ color: '#333' }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Gene</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>LGMD Class</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Locus</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Inh</th>
                <th style={{ padding: '10px 12px', textAlign: 'right' }}>N</th>
                <th style={{ padding: '10px 12px', textAlign: 'right' }}>Alive%</th>
                <th style={{ padding: '10px 12px', textAlign: 'right' }}>Onset yr</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Cardiac Risk</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Echo Freq</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, d], idx) => (
                <tr key={gene} style={{ background: idx % 2 === 0 ? '#f8faff' : '#fff', borderBottom: '1px solid #e3e8f0' }}>
                  <td style={{ padding: '9px 12px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                  <td style={{ padding: '9px 12px', fontSize: 11 }}>{GENE_INFO[gene]?.lgmd}</td>
                  <td style={{ padding: '9px 12px', fontFamily: 'monospace', fontSize: 11 }}>{d.locus}</td>
                  <td style={{ padding: '9px 12px', fontSize: 11 }}>AR</td>
                  <td style={{ padding: '9px 12px', textAlign: 'right' }}>{d.n_patients}</td>
                  <td style={{ padding: '9px 12px', textAlign: 'right' }}>{d.alive_pct}%</td>
                  <td style={{ padding: '9px 12px', textAlign: 'right' }}>{d.mean_age}yr</td>
                  <td style={{ padding: '9px 12px', fontSize: 11, color: CARDIAC_RISK[gene]?.color }}>{CARDIAC_RISK[gene]?.risk}</td>
                  <td style={{ padding: '9px 12px', fontSize: 11 }}>{CARDIAC_RISK[gene]?.freq}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Gene disease descriptions */}
          <div style={{ marginTop: 24 }}>
            {Object.entries(GENE_INFO).map(([gene, info]) => (
              <div key={gene} style={{ background: '#f8faff', borderLeft: `4px solid ${GENE_COLORS[gene]}`, padding: '10px 16px', marginBottom: 10, borderRadius: '0 8px 8px 0' }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gene], fontSize: 13 }}>{gene} — {info.lgmd} · {info.locus} · {info.size} · {info.inh}</div>
                <div style={{ fontSize: 12, color: '#444', marginTop: 4 }}>{info.disease}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {Object.entries(breakdown).map(([gene, d]) => (
            <div key={gene} style={{ marginBottom: 28, border: `1px solid ${GENE_COLORS[gene]}30`, borderRadius: 12, overflow: 'hidden' }}>
              <div style={{ background: GENE_COLORS[gene], color: '#fff', padding: '10px 16px', fontWeight: 700, fontSize: 14 }}>
                {gene} — {GENE_INFO[gene]?.lgmd} · {GENE_INFO[gene]?.locus} · {GENE_INFO[gene]?.size}
              </div>
              <div style={{ padding: 16, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#1a237e', marginBottom: 6 }}>Key Features</div>
                  {(d.key_features || []).map((f, i) => (
                    <div key={i} style={{ fontSize: 11, color: '#333', marginBottom: 4, paddingLeft: 8, borderLeft: '2px solid #e3e8f0' }}>{f}</div>
                  ))}
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#2e7d32', marginBottom: 6 }}>Treatment</div>
                  {(d.treatment || []).map((t, i) => (
                    <div key={i} style={{ fontSize: 11, color: '#333', marginBottom: 4, paddingLeft: 8, borderLeft: '2px solid #c8e6c9' }}>{t}</div>
                  ))}
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#b71c1c', marginBottom: 6 }}>Contraindications / Pearls</div>
                  {(d.contraindications || []).map((c, i) => (
                    <div key={i} style={{ fontSize: 11, color: '#b71c1c', marginBottom: 4, paddingLeft: 8, borderLeft: '2px solid #ffcdd2' }}>{c}</div>
                  ))}
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#e65100', marginTop: 8, marginBottom: 4 }}>Critical Pearls</div>
                  {(d.critical_pearls || []).map((p, i) => (
                    <div key={i} style={{ fontSize: 11, color: '#e65100', marginBottom: 4, paddingLeft: 8, borderLeft: '2px solid #ffe0b2' }}>{p}</div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          {/* LGMD Classification */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontWeight: 700, fontSize: 15, color: '#1a237e', marginBottom: 12 }}>LGMD 2017 Classification (EUROMYONET)</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {Object.entries(definitions.lgmd_classification_2017 || {}).map(([cls, desc]) => (
                <div key={cls} style={{ background: '#f0f4ff', borderRadius: 8, padding: '8px 12px', fontSize: 12 }}>
                  <span style={{ fontWeight: 700, color: '#1a237e' }}>{cls}: </span>{desc}
                </div>
              ))}
            </div>
          </div>

          {/* IHC Patterns */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontWeight: 700, fontSize: 15, color: '#1a237e', marginBottom: 12 }}>IHC Pattern Guide</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#e8eaf6' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Primary Gene Absent</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>IHC Pattern</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(definitions.ihc_pattern_table || {}).map(([k, v], i) => (
                  <tr key={k} style={{ background: i % 2 === 0 ? '#f8faff' : '#fff', borderBottom: '1px solid #e3e8f0' }}>
                    <td style={{ padding: '7px 12px', fontWeight: 600 }}>{k.replace(/_/g, ' ')}</td>
                    <td style={{ padding: '7px 12px' }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Cardiac Surveillance */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontWeight: 700, fontSize: 15, color: '#1a237e', marginBottom: 12 }}>Cardiac Surveillance Protocol</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fce4ec' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Gene</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Protocol</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(definitions.cardiac_surveillance_table || {}).map(([gene, v], i) => (
                  <tr key={gene} style={{ background: i % 2 === 0 ? '#fff9f9' : '#fff', borderBottom: '1px solid #f3e5e5' }}>
                    <td style={{ padding: '7px 12px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                    <td style={{ padding: '7px 12px' }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* DDx Table */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontWeight: 700, fontSize: 15, color: '#1a237e', marginBottom: 12 }}>Differential Diagnosis Table</div>
            {Object.entries(definitions.ddx_table || {}).map(([pair, desc]) => (
              <div key={pair} style={{ background: '#fff8e1', border: '1px solid #ffe082', borderRadius: 8, padding: '8px 14px', marginBottom: 8, fontSize: 12 }}>
                <span style={{ fontWeight: 700, color: '#f57f17' }}>{pair.replace(/_/g, ' ')}: </span>{desc}
              </div>
            ))}
          </div>

          {/* Founder Mutations */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontWeight: 700, fontSize: 15, color: '#1a237e', marginBottom: 12 }}>Founder Mutations by Population</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
              {Object.entries(definitions.founder_mutations || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#f3e5f5', borderRadius: 8, padding: '8px 12px', fontSize: 11 }}>
                  <div style={{ fontWeight: 700, color: '#6a1b9a' }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ color: '#444', marginTop: 2 }}>{v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Glossary */}
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, color: '#1a237e', marginBottom: 12 }}>Glossary</div>
            <div style={{ columns: 2, columnGap: 20 }}>
              {Object.entries(definitions.glossary || {}).map(([term, def]) => (
                <div key={term} style={{ breakInside: 'avoid', marginBottom: 10, fontSize: 12 }}>
                  <span style={{ fontWeight: 700, color: '#1a237e' }}>{term.replace(/_/g, ' ')}: </span>
                  <span style={{ color: '#444' }}>{def}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
