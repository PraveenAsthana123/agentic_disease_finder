'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-dba-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'RPS19':  '#b71c1c',  // deep red   — most common (25%), DBA1, RPS19/16kDa/40S
  'RPL5':   '#1565c0',  // deep blue  — cleft palate PATHOGNOMONIC, highest cancer risk, 60S
  'RPL11':  '#7b1fa2',  // deep purple — thenar hypoplasia, 60S MDM2 partner
  'RPS26':  '#2e7d32',  // dark green  — 40S E-site, AML 12q link
  'RPL35A': '#e65100',  // deep orange — GU anomalies, 60S peptide exit tunnel
  'RPS17':  '#00695c',  // dark teal   — first RP after RPS19 discovered, 40S head domain
  'RPL26':  '#37474f',  // dark slate  — 17p13.1 adjacent to TP53, p53 mRNA IRES regulator
  'TSR2':   '#4527a0',  // deep indigo — X-linked, RPS26 chaperone, males only
};

const GENE_INFO = {
  'RPS19':  { full: 'RPS19 / Ribosomal Protein S19 / 145aa', locus: '19q13.2', size: '145 aa / 16 kDa (40S small subunit; nucleolar stress sensor)', inh: 'AD' },
  'RPL5':   { full: 'RPL5 / Ribosomal Protein L5 / 297aa', locus: '1p22.1', size: '297 aa / 34 kDa (60S large subunit; 5S rRNA scaffold; MDM2 binder)', inh: 'AD' },
  'RPL11':  { full: 'RPL11 / Ribosomal Protein L11 / 178aa', locus: '1p36.1', size: '178 aa / 20 kDa (60S large subunit; zinc-finger-like; MDM2 key partner)', inh: 'AD' },
  'RPS26':  { full: 'RPS26 / Ribosomal Protein S26 / 119aa', locus: '12q13.2', size: '119 aa / 13 kDa (40S small subunit; mRNA E-site; AML 12q link)', inh: 'AD' },
  'RPL35A': { full: 'RPL35A / Ribosomal Protein L35a / 110aa', locus: '3q29', size: '110 aa / 12 kDa (60S large subunit; peptide exit tunnel)', inh: 'AD' },
  'RPS17':  { full: 'RPS17 / Ribosomal Protein S17 / 135aa', locus: '15q25.2', size: '135 aa / 15 kDa (40S small subunit head domain; first non-RPS19 gene)', inh: 'AD' },
  'RPL26':  { full: 'RPL26 / Ribosomal Protein L26 / 145aa', locus: '17p13.1', size: '145 aa / 17 kDa (60S large subunit; p53 mRNA 5′-UTR binder; 17p adj TP53)', inh: 'AD' },
  'TSR2':   { full: 'TSR2 / TSR2 Ribosome Maturation Factor / 228aa', locus: 'Xp11.22', size: '228 aa / 26 kDa (RPS26 nuclear import chaperone; X-linked DBA only)', inh: 'XLR' },
};

function GeneChip({ gene, active, onClick }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span
      onClick={() => onClick && onClick(gene)}
      style={{
        background: col, color: '#fff', borderRadius: 4,
        padding: '3px 10px', fontSize: 12, fontWeight: 700,
        margin: '0 3px 4px 0', cursor: onClick ? 'pointer' : 'default',
        opacity: active === null || active === gene ? 1 : 0.45,
        border: active === gene ? '2px solid #fff' : '2px solid transparent',
        display: 'inline-block',
      }}
    >{gene}</span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryDbaAtlas() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [activeGene, setActiveGene] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const [ovRes, bkRes, dfRes] = await Promise.all([
          fetch(`${API}/api/${SLUG}/overview`),
          fetch(`${API}/api/${SLUG}/breakdown`),
          fetch(`${API}/api/${SLUG}/definitions`),
        ]);
        setOverview(await ovRes.json());
        setBreakdown(await bkRes.json());
        setDefs(await dfRes.json());
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const geneList = Object.keys(GENE_COLORS);

  if (loading) return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#94a3b8', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18 }}>
      Loading DBA Atlas…
    </div>
  );
  if (error) return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#ef4444', padding: 32, fontSize: 16 }}>
      Error: {error}
    </div>
  );

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1e1b4b 0%,#312e81 50%,#1e293b 100%)', padding: '28px 32px 20px', borderBottom: '1px solid #334155' }}>
        <div style={{ fontSize: 11, color: '#818cf8', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 6 }}>
          Hereditary Disease Atlas · Ribosomopathy
        </div>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, color: '#f1f5f9' }}>
          🧬 Hereditary Diamond-Blackfan Anaemia Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 6 }}>
          Complete 8-Gene DBA &amp; Ribosomopathy Reference — RPS19 · RPL5 · RPL11 · RPS26 · RPL35A · RPS17 · RPL26 · TSR2
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {geneList.map(g => <GeneChip key={g} gene={g} active={activeGene} onClick={g => setActiveGene(prev => prev === g ? null : g)} />)}
        </div>
        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginTop: 16 }}>
          {TABS.map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                background: tab === t ? '#4f46e5' : 'transparent',
                color: tab === t ? '#fff' : '#94a3b8',
                border: `1px solid ${tab === t ? '#6366f1' : '#334155'}`,
                borderRadius: 6, padding: '6px 16px', fontSize: 13, fontWeight: 600, cursor: 'pointer',
              }}
            >{t}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: '24px 32px' }}>
        {/* ─── OVERVIEW TAB ─── */}
        {tab === 'Overview' && overview && (
          <div>
            {/* Metric cards */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 28 }}>
              <MetricCard label="Total patients" value={overview.total_patients} sub="8 × 40 seeds" />
              <MetricCard label="Genes catalogued" value={overview.genes?.length} sub="40S + 60S + chaperone" />
              <MetricCard label="Most common gene" value="RPS19" sub="~25% of all DBA" />
              <MetricCard label="Seeds" value="2806–2813" sub="DBA atlas cohort" />
              <MetricCard label="eADA elevated" value="~82%" sub="pathognomonic when elevated" />
              <MetricCard label="Steroid response" value="40–60%" sub="prednisolone 2 mg/kg/day" />
            </div>

            {/* Gene summary table */}
            <div style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 28 }}>
              <h3 style={{ margin: '0 0 14px', color: '#f8fafc', fontSize: 16 }}>Gene Summary Table — 8-Gene DBA Reference</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#334155', color: '#94a3b8' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left' }}>Gene</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left' }}>Locus / Inh</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left' }}>Subunit</th>
                      <th style={{ padding: '8px 10px', textAlign: 'right' }}>Patients</th>
                      <th style={{ padding: '8px 10px', textAlign: 'right' }}>eADA ↑ %</th>
                      <th style={{ padding: '8px 10px', textAlign: 'right' }}>Steroid Resp %</th>
                      <th style={{ padding: '8px 10px', textAlign: 'right' }}>Median Hb (g/dL)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {overview.gene_summaries?.map((g, i) => {
                      const info = GENE_INFO[g.gene] || {};
                      const subunit = g.gene.startsWith('RPS') || g.gene === 'TSR2' ? '40S / Chaperone' : '60S';
                      const subunitDisplay = g.gene === 'TSR2' ? 'RPS26 Chaperone' : g.gene.startsWith('RPS') ? '40S Small' : '60S Large';
                      const show = activeGene === null || activeGene === g.gene;
                      return (
                        <tr
                          key={g.gene}
                          onClick={() => setActiveGene(prev => prev === g.gene ? null : g.gene)}
                          style={{ background: i % 2 === 0 ? '#162032' : '#1e293b', cursor: 'pointer', opacity: show ? 1 : 0.35, borderBottom: '1px solid #334155' }}
                        >
                          <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#fff' }}>{g.gene}</td>
                          <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.locus} · {info.inh || 'AD'}</td>
                          <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 12 }}>{subunitDisplay}</td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: '#38bdf8' }}>{g.n_patients}</td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: g.pct_eada_elevated >= 78 ? '#4ade80' : '#f59e0b' }}>
                            {g.pct_eada_elevated}%
                          </td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: g.pct_steroid_response >= 50 ? '#4ade80' : '#f59e0b' }}>
                            {g.pct_steroid_response}%
                          </td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: '#f87171' }}>
                            {g.median_hb}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Pathway categories */}
            <div style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 28 }}>
              <h3 style={{ margin: '0 0 14px', color: '#f8fafc', fontSize: 16 }}>Pathway Categories</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 14 }}>
                {overview.pathway_categories?.map((pc, i) => (
                  <div key={i} style={{ background: '#0f172a', borderRadius: 8, padding: 14, border: '1px solid #334155' }}>
                    <div style={{ fontWeight: 700, color: '#818cf8', fontSize: 13, marginBottom: 8 }}>{pc.pathway}</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
                      {pc.genes?.map(g => <GeneChip key={g} gene={g} active={null} />)}
                    </div>
                    <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>{pc.note}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Critical distinctions */}
            <div style={{ background: '#1e293b', borderRadius: 10, padding: 20 }}>
              <h3 style={{ margin: '0 0 14px', color: '#f8fafc', fontSize: 16 }}>⚠️ Critical Distinctions &amp; Clinical Pearls</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {overview.critical_distinctions?.map((d, i) => {
                  const [bold, ...rest] = d.split(':');
                  return (
                    <div key={i} style={{ background: '#0f172a', borderRadius: 6, padding: '10px 14px', borderLeft: '3px solid #4f46e5', fontSize: 13, color: '#cbd5e1', lineHeight: 1.6 }}>
                      <span style={{ fontWeight: 700, color: '#818cf8' }}>{bold}:</span>{rest.join(':')}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ─── GENE TABLE TAB ─── */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <div style={{ marginBottom: 16, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {geneList.map(g => <GeneChip key={g} gene={g} active={activeGene} onClick={g => setActiveGene(prev => prev === g ? null : g)} />)}
              {activeGene && <button onClick={() => setActiveGene(null)} style={{ background: '#334155', color: '#94a3b8', border: 'none', borderRadius: 4, padding: '3px 10px', fontSize: 12, cursor: 'pointer' }}>Clear</button>}
            </div>
            {breakdown.genes?.filter(g => activeGene === null || activeGene === g.gene).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 20, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                  <span style={{ fontSize: 22, fontWeight: 800, color: GENE_COLORS[g.gene] }}>{g.gene}</span>
                  <span style={{ fontSize: 13, color: '#94a3b8' }}>{GENE_INFO[g.gene]?.full}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(260px,1fr))', gap: 12, marginBottom: 14 }}>
                  <div style={{ background: '#0f172a', borderRadius: 6, padding: 10 }}>
                    <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 4 }}>Locus / Inheritance</div>
                    <div style={{ fontSize: 13, color: '#e2e8f0' }}>{g.locus} · {GENE_INFO[g.gene]?.inh}</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 6, padding: 10 }}>
                    <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 4 }}>Protein Size</div>
                    <div style={{ fontSize: 12, color: '#e2e8f0' }}>{g.protein_size?.slice(0, 100)}</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 6, padding: 10 }}>
                    <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 4 }}>Patients in Atlas</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: '#38bdf8' }}>{g.n_patients}</div>
                  </div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                  <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 6 }}>Disease Category</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{g.disease_category}</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                  <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 6 }}>Pathognomonic Signs</div>
                  <div style={{ fontSize: 12, color: '#fbbf24', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{g.pathognomonic}</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12 }}>
                  <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 6 }}>Treatment</div>
                  <div style={{ fontSize: 12, color: '#86efac', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{g.treatment}</div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ─── CLINICAL ATLAS TAB ─── */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <div style={{ marginBottom: 20, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {geneList.map(g => <GeneChip key={g} gene={g} active={activeGene} onClick={g => setActiveGene(prev => prev === g ? null : g)} />)}
              {activeGene && <button onClick={() => setActiveGene(null)} style={{ background: '#334155', color: '#94a3b8', border: 'none', borderRadius: 4, padding: '3px 10px', fontSize: 12, cursor: 'pointer' }}>Clear</button>}
            </div>
            {breakdown.genes?.filter(g => activeGene === null || activeGene === g.gene).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 20, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
                  <span style={{ fontSize: 20, fontWeight: 800, color: GENE_COLORS[g.gene] }}>{g.gene}</span>
                  <span style={{ fontSize: 12, color: '#64748b' }}>{g.locus}</span>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                  <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 6 }}>Inheritance &amp; Molecular Mechanism</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{g.inheritance}</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12 }}>
                  <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 6 }}>Disease Pathway — {g.gene} Mechanism</div>
                  <div style={{ fontSize: 12, color: '#a5b4fc', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{g.disease_pathway}</div>
                </div>
                {/* Sample patient data */}
                {g.patients?.length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 8 }}>Sample Patient Data (first 5 of {g.n_patients})</div>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                        <thead>
                          <tr style={{ background: '#334155', color: '#94a3b8' }}>
                            <th style={{ padding: '6px 8px', textAlign: 'left' }}>ID</th>
                            <th style={{ padding: '6px 8px', textAlign: 'center' }}>Sex</th>
                            <th style={{ padding: '6px 8px', textAlign: 'right' }}>Age Dx (yr)</th>
                            <th style={{ padding: '6px 8px', textAlign: 'right' }}>Hb (g/dL)</th>
                            <th style={{ padding: '6px 8px', textAlign: 'right' }}>MCV (fL)</th>
                            <th style={{ padding: '6px 8px', textAlign: 'right' }}>eADA</th>
                            <th style={{ padding: '6px 8px', textAlign: 'center' }}>Steroid Resp</th>
                            <th style={{ padding: '6px 8px', textAlign: 'center' }}>HSCT</th>
                          </tr>
                        </thead>
                        <tbody>
                          {g.patients.map((p, pi) => (
                            <tr key={pi} style={{ background: pi % 2 === 0 ? '#162032' : '#1e293b', borderBottom: '1px solid #334155' }}>
                              <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                              <td style={{ padding: '5px 8px', textAlign: 'center', color: p.sex === 'M' ? '#38bdf8' : '#f9a8d4' }}>{p.sex}</td>
                              <td style={{ padding: '5px 8px', textAlign: 'right', color: '#e2e8f0' }}>{p.age_at_diagnosis_years}</td>
                              <td style={{ padding: '5px 8px', textAlign: 'right', color: '#f87171', fontWeight: 600 }}>{p.hb_at_diagnosis_gdl}</td>
                              <td style={{ padding: '5px 8px', textAlign: 'right', color: '#fbbf24' }}>{p.mcv_fl}</td>
                              <td style={{ padding: '5px 8px', textAlign: 'right', color: p.eada_elevated ? '#4ade80' : '#94a3b8' }}>{p.eada_ugHb}</td>
                              <td style={{ padding: '5px 8px', textAlign: 'center' }}>
                                <span style={{ color: p.steroid_response ? '#4ade80' : '#ef4444', fontWeight: 700 }}>
                                  {p.steroid_response ? '✓' : '✗'}
                                </span>
                              </td>
                              <td style={{ padding: '5px 8px', textAlign: 'center' }}>
                                <span style={{ color: p.hsct_performed ? '#818cf8' : '#334155', fontWeight: 700 }}>
                                  {p.hsct_performed ? '✓' : '–'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* ─── DEFINITIONS TAB ─── */}
        {tab === 'Definitions' && defs && (
          <div>
            <div style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 16px', color: '#f8fafc', fontSize: 16 }}>DBA &amp; Ribosomopathy Glossary</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {defs.definitions?.map((d, i) => (
                  <div key={i} style={{ background: '#0f172a', borderRadius: 6, padding: '12px 16px', borderLeft: '3px solid #4f46e5' }}>
                    <div style={{ fontWeight: 700, color: '#818cf8', fontSize: 13, marginBottom: 6 }}>{d.term}</div>
                    <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>{d.definition}</div>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ background: '#1e293b', borderRadius: 10, padding: 20 }}>
              <h3 style={{ margin: '0 0 12px', color: '#f8fafc', fontSize: 15 }}>Clinical Standards &amp; References</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {defs.standards?.map((s, i) => (
                  <div key={i} style={{ background: '#0f172a', borderRadius: 4, padding: '8px 12px', fontSize: 12, color: '#94a3b8', borderLeft: '3px solid #334155' }}>
                    {s}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
