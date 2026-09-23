'use client';
import { useState, useEffect } from 'react';

const SLUG = 'hereditary-neuroblastoma-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ALK:    '#6366f1',
  PHOX2B: '#f59e0b',
  BARD1:  '#10b981',
  KIF1B:  '#3b82f6',
  NF1:    '#f97316',
  TP53:   '#ef4444',
  DICER1: '#8b5cf6',
  BRCA2:  '#ec4899',
};

const GENE_INFO = {
  ALK:    { full: 'Anaplastic Lymphoma Kinase',        locus: '2p23.2',   size: '1620aa', inh: 'AD GOF' },
  PHOX2B: { full: 'Paired-like Homeobox 2B',           locus: '4p13',     size: '314aa',  inh: 'AD LOF/poly' },
  BARD1:  { full: 'BRCA1-Associated RING Domain 1',    locus: '2q35',     size: '777aa',  inh: 'AD LOF' },
  KIF1B:  { full: 'Kinesin Family Member 1B',          locus: '1p36.22',  size: '1770aa', inh: 'AD LOF' },
  NF1:    { full: 'Neurofibromin 1',                   locus: '17q11.2',  size: '2839aa', inh: 'AD LOF' },
  TP53:   { full: 'Tumour Protein p53',                locus: '17p13.1',  size: '393aa',  inh: 'AD LOF' },
  DICER1: { full: 'Dicer 1, Ribonuclease III',         locus: '14q32.13', size: '1922aa', inh: 'AD LOF' },
  BRCA2:  { full: 'Breast Cancer Type 2 Susceptibility', locus: '13q12.3', size: '3418aa', inh: 'AD LOF' },
};

function Badge({ label, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 6, padding: '2px 10px', fontSize: 12, fontWeight: 700,
      display: 'inline-block', whiteSpace: 'nowrap'
    }}>{label}</span>
  );
}

function GeneBar({ gene, value, max }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</span>
        <span>{value}</span>
      </div>
      <div style={{ background: '#e5e7eb', borderRadius: 4, height: 8 }}>
        <div style={{ background: GENE_COLORS[gene], width: `${pct}%`, height: 8, borderRadius: 4, transition: 'width 0.4s' }} />
      </div>
    </div>
  );
}

export default function HereditaryNeuroblastomaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => {
      setError(e.message);
      setLoading(false);
    });
  }, []);

  if (loading) return (
    <div style={{ padding: 48, textAlign: 'center', color: '#6b7280' }}>
      Loading Hereditary Neuroblastoma Predisposition Atlas…
    </div>
  );
  if (error) return (
    <div style={{ padding: 48, textAlign: 'center', color: '#ef4444' }}>
      Error: {error}
    </div>
  );

  const genes = Object.keys(GENE_COLORS);
  const maxCR = overview ? Math.max(...genes.map(g => overview.gene_summaries?.[g]?.cr_rate_pct ?? 0)) : 100;

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 16px', fontFamily: 'system-ui,sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <span style={{ fontSize: 32 }}>🧬</span>
          <div>
            <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#111827' }}>
              Hereditary Neuroblastoma Predisposition Atlas
            </h1>
            <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
              8-Gene Reference Panel · 320-Patient Aggregate · Seeds 3270–3277
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {genes.map(g => <Badge key={g} label={g} color={GENE_COLORS[g]} />)}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '2px solid #e5e7eb', marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '10px 20px', border: 'none', background: 'none', cursor: 'pointer',
            fontWeight: tab === t ? 700 : 400, color: tab === t ? '#6366f1' : '#6b7280',
            borderBottom: tab === t ? '2px solid #6366f1' : '2px solid transparent',
            marginBottom: -2, fontSize: 14, transition: 'all 0.15s'
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          {/* KPI Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 16, marginBottom: 28 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#6366f1' },
              { label: 'Genes Covered', value: overview.genes_covered, color: '#10b981' },
              { label: 'Overall CR Rate', value: `${overview.overall_cr_rate_pct}%`, color: '#3b82f6' },
              { label: 'Radiation Avoided', value: `${overview.radiation_avoidance_pct}%`, color: '#f59e0b' },
              { label: 'Targeted Therapy', value: `${overview.targeted_therapy_pct}%`, color: '#8b5cf6' },
              { label: 'Familial Cases', value: `${overview.familial_pct}%`, color: '#ec4899' },
            ].map(k => (
              <div key={k.label} style={{
                background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12,
                padding: '16px 20px', boxShadow: '0 1px 4px rgba(0,0,0,0.06)'
              }}>
                <div style={{ fontSize: 24, fontWeight: 800, color: k.color }}>{k.value}</div>
                <div style={{ fontSize: 12, color: '#6b7280', marginTop: 2 }}>{k.label}</div>
              </div>
            ))}
          </div>

          {/* CR by Gene */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 28 }}>
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20 }}>
              <h3 style={{ margin: '0 0 16px', fontSize: 15, fontWeight: 700 }}>Complete Remission by Gene</h3>
              {genes.map(g => (
                <GeneBar key={g} gene={g} value={overview.gene_summaries?.[g]?.cr_rate_pct ?? 0} max={100} />
              ))}
            </div>
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20 }}>
              <h3 style={{ margin: '0 0 16px', fontSize: 15, fontWeight: 700 }}>INRG Risk Distribution</h3>
              {overview.risk_distribution && Object.entries(overview.risk_distribution).map(([risk, cnt]) => (
                <div key={risk} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontSize: 13, color: '#374151', fontWeight: 600 }}>{risk}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ background: '#e5e7eb', borderRadius: 4, height: 10, width: 120 }}>
                      <div style={{
                        background: risk.includes('High') ? '#ef4444' : risk.includes('Inter') ? '#f59e0b' : '#10b981',
                        width: `${Math.round((cnt / overview.total_patients) * 100)}%`,
                        height: 10, borderRadius: 4
                      }} />
                    </div>
                    <span style={{ fontSize: 12, color: '#6b7280', minWidth: 28 }}>{cnt}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Clinical Pearls */}
          {overview.clinical_pearls && (
            <div style={{ background: '#fffbeb', border: '1px solid #fcd34d', borderRadius: 12, padding: 20, marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 700, color: '#92400e' }}>
                ⚠ Critical Clinical Pearls
              </h3>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {overview.clinical_pearls.map((p, i) => (
                  <li key={i} style={{ fontSize: 13, color: '#78350f', marginBottom: 6 }}>{p}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Key Management Rules */}
          {overview.key_management_rules && (
            <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 12, padding: 20 }}>
              <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 700, color: '#991b1b' }}>
                🚫 Key Management Rules (Absolute Contraindications)
              </h3>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {overview.key_management_rules.map((r, i) => (
                  <li key={i} style={{ fontSize: 13, color: '#7f1d1d', marginBottom: 6 }}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && overview && (
        <div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f9fafb', borderBottom: '2px solid #e5e7eb' }}>
                  {['Gene', 'Full Name', 'Locus', 'Protein', 'Inheritance', 'Patients', 'CR Rate', 'Radiation Avoided', 'Key Targeted Tx'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700, color: '#374151', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {genes.map((g, i) => {
                  const gs = overview.gene_summaries?.[g] || {};
                  const gi = GENE_INFO[g];
                  return (
                    <tr key={g} style={{ borderBottom: '1px solid #f3f4f6', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                      <td style={{ padding: '10px 12px' }}>
                        <Badge label={g} color={GENE_COLORS[g]} />
                      </td>
                      <td style={{ padding: '10px 12px', color: '#374151' }}>{gi.full}</td>
                      <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: '#6b7280' }}>{gi.locus}</td>
                      <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: '#6b7280' }}>{gi.size}</td>
                      <td style={{ padding: '10px 12px', color: '#6b7280' }}>{gi.inh}</td>
                      <td style={{ padding: '10px 12px', fontWeight: 700, color: '#374151' }}>{gs.n_patients ?? 40}</td>
                      <td style={{ padding: '10px 12px', fontWeight: 700, color: '#10b981' }}>{gs.cr_rate_pct ?? '—'}%</td>
                      <td style={{ padding: '10px 12px', color: '#6b7280' }}>{gs.radiation_avoided_pct ?? '—'}%</td>
                      <td style={{ padding: '10px 12px', color: '#374151', maxWidth: 200, fontSize: 12 }}>
                        {gs.targeted_therapy ?? '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {genes.map(g => {
            const gd = breakdown.genes?.[g];
            if (!gd) return null;
            return (
              <div key={g} style={{ marginBottom: 32 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                  <Badge label={g} color={GENE_COLORS[g]} />
                  <span style={{ fontWeight: 700, color: '#374151' }}>{GENE_INFO[g].full}</span>
                  <span style={{ color: '#9ca3af', fontSize: 12 }}>{GENE_INFO[g].locus} · {GENE_INFO[g].size} · {GENE_INFO[g].inh}</span>
                </div>
                {/* Gene stats row */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(140px,1fr))', gap: 10, marginBottom: 14 }}>
                  {[
                    { label: 'CR Rate', value: `${gd.cr_rate_pct}%` },
                    { label: 'Radiation Avoided', value: `${gd.radiation_avoided_pct}%` },
                    { label: 'GTR Achieved', value: `${gd.gtr_pct}%` },
                    { label: 'Relapse Rate', value: `${gd.relapse_pct}%` },
                    { label: 'Targeted Therapy', value: `${gd.targeted_therapy_pct}%` },
                  ].map(s => (
                    <div key={s.label} style={{
                      background: '#fff', border: `1px solid ${GENE_COLORS[g]}44`,
                      borderRadius: 8, padding: '10px 14px'
                    }}>
                      <div style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[g] }}>{s.value}</div>
                      <div style={{ fontSize: 11, color: '#6b7280' }}>{s.label}</div>
                    </div>
                  ))}
                </div>
                {/* Patient sample table */}
                {gd.patients && gd.patients.length > 0 && (
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                      <thead>
                        <tr style={{ background: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                          {['Patient ID', 'Age (yr)', 'INRG Risk', 'Histology', 'CR', 'Radiation', 'Targeted Tx', 'Relapse', 'Variant'].map(h => (
                            <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#6b7280', whiteSpace: 'nowrap' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {gd.patients.slice(0, 10).map((p, i) => (
                          <tr key={p.patient_id} style={{ borderBottom: '1px solid #f3f4f6', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                            <td style={{ padding: '7px 10px', fontFamily: 'monospace', color: '#374151' }}>{p.patient_id}</td>
                            <td style={{ padding: '7px 10px' }}>{p.age_dx}</td>
                            <td style={{ padding: '7px 10px' }}>
                              <span style={{
                                color: p.inrg_risk?.includes('High') ? '#ef4444' : p.inrg_risk?.includes('Inter') ? '#f59e0b' : '#10b981',
                                fontWeight: 700, fontSize: 11
                              }}>{p.inrg_risk}</span>
                            </td>
                            <td style={{ padding: '7px 10px', color: '#374151' }}>{p.histology}</td>
                            <td style={{ padding: '7px 10px' }}>
                              <span style={{ color: p.cr_achieved ? '#10b981' : '#ef4444', fontWeight: 700 }}>
                                {p.cr_achieved ? 'Yes' : 'No'}
                              </span>
                            </td>
                            <td style={{ padding: '7px 10px' }}>
                              <span style={{ color: p.radiation_received ? '#f59e0b' : '#6b7280' }}>
                                {p.radiation_received ? 'Yes' : 'No'}
                              </span>
                            </td>
                            <td style={{ padding: '7px 10px', color: '#374151', fontSize: 11 }}>{p.targeted_therapy || '—'}</td>
                            <td style={{ padding: '7px 10px' }}>
                              <span style={{ color: p.relapse ? '#ef4444' : '#10b981', fontWeight: 700 }}>
                                {p.relapse ? 'Yes' : 'No'}
                              </span>
                            </td>
                            <td style={{ padding: '7px 10px', fontFamily: 'monospace', fontSize: 11, color: '#6b7280' }}>{p.variant || '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {gd.patients.length > 10 && (
                      <p style={{ fontSize: 12, color: '#9ca3af', padding: '6px 10px' }}>
                        Showing 10 of {gd.patients.length} patients
                      </p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          {/* Atlas-level definitions */}
          {definitions.atlas_definitions && (
            <div style={{ background: '#f0fdf4', border: '1px solid #86efac', borderRadius: 12, padding: 20, marginBottom: 24 }}>
              <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 700, color: '#166534' }}>
                Atlas Definitions
              </h3>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {definitions.atlas_definitions.map((d, i) => (
                  <li key={i} style={{ fontSize: 13, color: '#14532d', marginBottom: 6 }}>{d}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Per-gene definitions */}
          {genes.map(g => {
            const gdef = definitions.genes?.[g];
            if (!gdef) return null;
            return (
              <div key={g} style={{
                background: '#fff', border: `1px solid ${GENE_COLORS[g]}44`,
                borderRadius: 12, padding: 20, marginBottom: 16
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                  <Badge label={g} color={GENE_COLORS[g]} />
                  <span style={{ fontWeight: 700, color: '#374151', fontSize: 14 }}>{GENE_INFO[g].full}</span>
                  <span style={{ fontFamily: 'monospace', fontSize: 12, color: '#9ca3af' }}>
                    {GENE_INFO[g].locus} · {GENE_INFO[g].size} · {GENE_INFO[g].inh}
                  </span>
                </div>
                {gdef.summary && (
                  <p style={{ fontSize: 13, color: '#374151', marginBottom: 12, lineHeight: 1.6 }}>
                    {gdef.summary}
                  </p>
                )}
                {gdef.key_rules && gdef.key_rules.length > 0 && (
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#6b7280', marginBottom: 6 }}>
                      KEY RULES
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                      {gdef.key_rules.map((r, i) => (
                        <li key={i} style={{ fontSize: 13, color: '#374151', marginBottom: 4 }}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {gdef.pathognomonic_associations && gdef.pathognomonic_associations.length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#ef4444', marginBottom: 6 }}>
                      PATHOGNOMONIC ASSOCIATIONS
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {gdef.pathognomonic_associations.map((a, i) => (
                        <span key={i} style={{
                          background: '#fef2f2', color: '#991b1b', border: '1px solid #fca5a5',
                          borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700
                        }}>{a}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Cascade testing rule */}
          {definitions.cascade_testing_rule && (
            <div style={{ background: '#eff6ff', border: '1px solid #93c5fd', borderRadius: 12, padding: 20, marginTop: 8 }}>
              <h3 style={{ margin: '0 0 10px', fontSize: 15, fontWeight: 700, color: '#1e40af' }}>
                Cascade Testing Rule
              </h3>
              <p style={{ margin: 0, fontSize: 13, color: '#1e3a8a', lineHeight: 1.6 }}>
                {definitions.cascade_testing_rule}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
