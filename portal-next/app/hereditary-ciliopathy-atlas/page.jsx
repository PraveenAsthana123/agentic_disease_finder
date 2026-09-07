'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  BBS1:    '#7c3aed',  // purple  — Bardet-Biedl; rod-cone; BBSome; M1L founder
  CEP290:  '#dc2626',  // red     — LCA10; Joubert; Meckel-lethal; deep intronic IVS26
  NPHP1:   '#0f766e',  // teal    — nephronophthisis; 2q13 deletion; MLPA required; child ESRD
  RPGR:    '#2563eb',  // blue    — XLRP; ORF15 hotspot; carrier female 20%; XLR
  AHI1:    '#ea580c',  // orange  — Joubert-3; 80% retinal; R830W Ashkenazi; AHI1 jouberin
  ALMS1:   '#ca8a04',  // amber   — Alström; cone-rod; cardiomyopathy 2 episodes; no polydactyly
  KIF7:    '#be185d',  // pink    — Hedgehog gatekeeper; hydrolethalus-to-Joubert spectrum; ACLS
  DYNC2H1: '#16a34a',  // green   — SRTD/Jeune; narrow thorax; VEPTR; IFT-A retrograde
};

const GENE_DISEASE = {
  BBS1:    'AR Bardet-Biedl-Syndrome-1 — BBS1-593aa — 11q13.2 — Rod-Cone-ALWAYS-First — Truncal-Obesity-Infancy — Postaxial-Polydactyly — Renal-Anomalies-50-100pct — M1L-Caucasian-Founder',
  CEP290:  'AR Joubert-JBTS5/LCA10/NPHP6/Meckel — CEP290-2479aa — 12q21.32 — c.2991+1655A>G-Deep-Intronic-IVS26-Most-Common-LCA-MISSED-by-Exome — Molar-Tooth-Sign-PATHOGNOMONIC',
  NPHP1:   'AR Nephronophthisis-Type1 — NPHP1-732aa — 2q13 — Homozygous-2q13-Deletion-85pct-MLPA-Required — Most-Common-Genetic-ESRD-Children — Polyuria-FIRST — Transplant-Non-Recurrent',
  RPGR:    'XLR X-linked-RP — RPGR-815aa — Xp11.3 — ORF15-Exon-70-75pct-XLRP-Hotspot-Specific-Sequencing — Female-Carriers-20pct-Symptomatic — Cone-Rod-Variant — Respiratory-Ciliary-Variant',
  AHI1:    'AR Joubert-JBTS3 — AHI1-1196aa — 6q23.3 — Highest-Retinal-80pct-All-JBTS-Genes — Nephronophthisis-30-40pct — R830W-Ashkenazi-Jewish-Founder-1in93-Carrier',
  ALMS1:   'AR Alström-Syndrome — ALMS1-4169aa — 2p13.1 — NO-Polydactyly-NO-Cognitive-Impairment — Cone-Rod-NOT-Rod-Cone — Cardiomyopathy-Infancy-RESOLVES-Then-Adolescence-RETURNS — T2DM-Childhood',
  KIF7:    'AR Acrocallosal/Hydrolethalus/JBTS12 — KIF7-1343aa — 15q26.1 — Hedgehog-Ciliary-Gatekeeper — Polydactyly-CC-Agenesis — Lethal-Hydrolethalus-to-Mild-JBTS12-Same-Gene',
  DYNC2H1: 'AR SRTD3-Jeune-ATD — DYNC2H1-4307aa — 11q22.3 — Most-Common-SRTD-45pct — Narrow-Thorax-Neonatal-Respiratory-Failure — VEPTR-Thoracic-Expansion — IFT-A-Retrograde-Motor',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#94a3b8' }}>Loading…</div>;
}

function ErrorBox({ msg }) {
  return (
    <div style={{ padding: '1rem', background: '#450a0a', borderRadius: 8, color: '#fca5a5', margin: '1rem 0' }}>
      Error: {msg}
    </div>
  );
}

function KPI({ label, value, color }) {
  return (
    <div style={{
      background: '#1e293b', borderRadius: 10, padding: '1rem 1.2rem',
      borderLeft: `4px solid ${color || '#6366f1'}`, minWidth: 160,
    }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#a5b4fc' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 6, padding: '2px 8px', fontSize: 11, fontWeight: 600,
    }}>{text}</span>
  );
}

export default function HereditoryCiliopathyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hereditary-ciliopathy-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-ciliopathy-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-ciliopathy-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const tabStyle = (t) => ({
    padding: '0.5rem 1.2rem', cursor: 'pointer', borderRadius: '6px 6px 0 0',
    fontWeight: tab === t ? 700 : 400,
    background: tab === t ? '#1e40af' : '#1e293b',
    color: tab === t ? '#fff' : '#94a3b8',
    border: 'none', fontSize: 13,
  });

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui,sans-serif', padding: '1.5rem' }}>
      <div style={{ maxWidth: 1280, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', margin: 0 }}>
            🔬 Hereditary Ciliopathy Atlas
          </h1>
          <p style={{ color: '#64748b', fontSize: 13, margin: '4px 0 0' }}>
            Complete 8-Gene Primary Cilia Disorder Atlas · BBS1 / CEP290 / NPHP1 / RPGR / AHI1 / ALMS1 / KIF7 / DYNC2H1 · 320-Patient Aggregate · Seeds 1950–1957
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 0, borderBottom: '2px solid #1e293b' }}>
          {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
        </div>

        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderTop: 'none', borderRadius: '0 0 10px 10px', padding: '1.5rem', minHeight: 400 }}>
          {loading && <Loading />}
          {error && <ErrorBox msg={error} />}

          {/* ── OVERVIEW ── */}
          {!loading && !error && tab === 'Overview' && overview && (
            <div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '1.5rem' }}>
                <KPI label="Total Patients" value={overview.total_patients} color="#6366f1" />
                <KPI label="Genes Covered" value={overview.genes_covered} color="#0ea5e9" />
                <KPI label="Seeds" value={overview.seeds} color="#10b981" />
                <KPI label="Inheritance" value={overview.inheritance_modes?.join(' · ')} color="#f59e0b" />
              </div>

              <h3 style={{ color: '#94a3b8', fontSize: 13, fontWeight: 700, marginBottom: 8 }}>KEY CLINICAL FACTS</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: '1.5rem' }}>
                {overview.key_clinical_facts?.map((f, i) => {
                  const gene = f.split(':')[0];
                  const color = GENE_COLORS[gene] || '#6366f1';
                  return (
                    <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: '0.6rem 1rem', borderLeft: `3px solid ${color}`, fontSize: 12 }}>
                      <span style={{ color, fontWeight: 700 }}>{gene}:</span>{' '}
                      <span style={{ color: '#cbd5e1' }}>{f.substring(gene.length + 2)}</span>
                    </div>
                  );
                })}
              </div>

              <h3 style={{ color: '#94a3b8', fontSize: 13, fontWeight: 700, marginBottom: 8 }}>GENE SUMMARY</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(260px,1fr))', gap: 12 }}>
                {overview.gene_summary?.map(gs => {
                  const color = GENE_COLORS[gs.gene] || '#6366f1';
                  return (
                    <div key={gs.gene} style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', borderLeft: `4px solid ${color}` }}>
                      <div style={{ fontWeight: 700, color, fontSize: 15 }}>{gs.gene}</div>
                      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{gs.alt_name}</div>
                      <div style={{ fontSize: 11, color: '#94a3b8' }}>
                        <span style={{ color: '#e2e8f0' }}>{gs.locus}</span> · <span style={{ color: '#e2e8f0' }}>{gs.protein_size}</span> · <span style={{ color: '#e2e8f0' }}>{gs.inheritance}</span>
                      </div>
                      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                        Patients: <span style={{ color }}>{gs.patient_count}</span>
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 }}>
                        {Object.entries(gs.severity_distribution || {}).map(([sev, cnt]) => (
                          <Badge key={sev} text={`${sev}: ${cnt}`} color={
                            sev === 'critical' ? '#ef4444' : sev === 'severe' ? '#f97316' : sev === 'moderate' ? '#eab308' : '#22c55e'
                          } />
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── GENE TABLE ── */}
          {!loading && !error && tab === 'Gene Table' && breakdown && (
            <div>
              <h3 style={{ color: '#94a3b8', fontSize: 13, fontWeight: 700, marginBottom: 12 }}>PER-GENE BREAKDOWN — 320 PATIENTS (8 × 40)</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#1e293b', color: '#64748b' }}>
                      {['Gene', 'Alt Name / Syndrome', 'Locus', 'Size', 'Inh.', 'N', 'Median Age', 'Severity Split'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #334155' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.values(breakdown.per_gene_breakdown || {}).map(gd => {
                      const color = GENE_COLORS[gd.gene] || '#6366f1';
                      const sevStr = Object.entries(gd.severity_counts || {}).map(([s, c]) => `${s}:${c}`).join(' · ');
                      return (
                        <tr key={gd.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                          <td style={{ padding: '8px 10px', fontWeight: 700, color }}>{gd.gene}</td>
                          <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{gd.alt_name}</td>
                          <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{gd.locus}</td>
                          <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{gd.protein_size}</td>
                          <td style={{ padding: '8px 10px' }}><Badge text={gd.inheritance} color={color} /></td>
                          <td style={{ padding: '8px 10px', color }}>{gd.patient_count}</td>
                          <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{gd.age_stats?.mean} yr</td>
                          <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 11 }}>{sevStr}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Sample patients */}
              <h3 style={{ color: '#94a3b8', fontSize: 13, fontWeight: 700, margin: '1.5rem 0 8px' }}>SAMPLE PATIENTS (first 3 per gene)</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 10 }}>
                {Object.values(breakdown.per_gene_breakdown || {}).flatMap(gd =>
                  (gd.sample_patients || []).map(p => {
                    const color = GENE_COLORS[p.gene] || '#6366f1';
                    return (
                      <div key={p.patient_id} style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem', borderLeft: `3px solid ${color}` }}>
                        <div style={{ fontWeight: 700, color, fontSize: 13 }}>{p.patient_id}</div>
                        <div style={{ fontSize: 11, color: '#94a3b8' }}>Age: {p.age_at_diagnosis} yr · {p.sex} · {p.severity}</div>
                        <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>Tx: {p.treatment}</div>
                        <div style={{ fontSize: 10, color: '#f59e0b', marginTop: 2 }}>⚑ {p.critical_flag}</div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}

          {/* ── CLINICAL ATLAS ── */}
          {!loading && !error && tab === 'Clinical Atlas' && breakdown && (
            <div>
              {Object.values(breakdown.per_gene_breakdown || {}).map(gd => {
                const color = GENE_COLORS[gd.gene] || '#6366f1';
                const disease = GENE_DISEASE[gd.gene] || '';
                return (
                  <div key={gd.gene} style={{ background: '#1e293b', borderRadius: 12, padding: '1.2rem', marginBottom: 14, borderLeft: `5px solid ${color}` }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, flexWrap: 'wrap' }}>
                      <div style={{ flex: 1, minWidth: 200 }}>
                        <div style={{ fontWeight: 800, color, fontSize: 17 }}>{gd.gene}</div>
                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{gd.alt_name}</div>
                        <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6, marginBottom: 6 }}>{disease}</div>
                        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                          <Badge text={gd.locus} color={color} />
                          <Badge text={gd.protein_size} color={color} />
                          <Badge text={gd.inheritance} color={color} />
                          <Badge text={`N=${gd.patient_count}`} color="#6366f1" />
                        </div>
                      </div>
                    </div>

                    <div style={{ marginTop: 10 }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 4 }}>PATHOGNOMONIC / DIAGNOSIS</div>
                      <div style={{ fontSize: 11, color: '#cbd5e1', background: '#0f172a', borderRadius: 6, padding: '0.5rem 0.8rem', lineHeight: 1.6 }}>
                        {gd.pathognomonic}
                      </div>
                    </div>

                    <div style={{ marginTop: 8 }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 4 }}>TREATMENT</div>
                      <div style={{ fontSize: 11, color: '#cbd5e1', background: '#0f172a', borderRadius: 6, padding: '0.5rem 0.8rem', lineHeight: 1.6 }}>
                        {gd.treatment}
                      </div>
                    </div>

                    <div style={{ marginTop: 8 }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: '#ef4444', marginBottom: 4 }}>⚑ CRITICAL FLAGS</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {(gd.critical_flags || []).map((f, i) => (
                          <div key={i} style={{ fontSize: 11, color: '#fca5a5', background: '#1f0a0a', borderRadius: 6, padding: '0.4rem 0.8rem', borderLeft: '3px solid #ef4444' }}>
                            {f}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* ── DEFINITIONS ── */}
          {!loading && !error && tab === 'Definitions' && definitions && (
            <div>
              <h3 style={{ color: '#94a3b8', fontSize: 13, fontWeight: 700, marginBottom: 12 }}>GENE DEFINITIONS & GLOSSARY</h3>

              {Object.values(definitions.gene_definitions || {}).map(gd => {
                const color = GENE_COLORS[gd.gene] || '#6366f1';
                return (
                  <div key={gd.gene} style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: 10, borderLeft: `4px solid ${color}` }}>
                    <div style={{ fontWeight: 700, color, fontSize: 14 }}>{gd.gene} — {gd.alt_name}</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{gd.protein_size} · {gd.locus} · {gd.inheritance}</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                      <span style={{ color: '#64748b', fontWeight: 700 }}>Protein: </span>{gd.protein}
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                      <span style={{ color: '#64748b', fontWeight: 700 }}>Age of Onset: </span>{gd.age_of_onset}
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                      <span style={{ color: '#64748b', fontWeight: 700 }}>Key Biomarker: </span>{gd.key_biomarker}
                    </div>
                  </div>
                );
              })}

              <h3 style={{ color: '#94a3b8', fontSize: 13, fontWeight: 700, margin: '1.5rem 0 8px' }}>CILIOPATHY GLOSSARY</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(300px,1fr))', gap: 8 }}>
                {Object.entries(definitions.glossary || {}).map(([term, def]) => (
                  <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem' }}>
                    <div style={{ fontWeight: 700, color: '#a5b4fc', fontSize: 12, marginBottom: 4 }}>{term.replace(/-/g, ' ')}</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{def}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
