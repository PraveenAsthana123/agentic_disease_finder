'use client';
import { useState, useEffect } from 'react';

const SLUG = 'hereditary-rhabdomyosarcoma-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  TP53:    '#ef4444',
  NF1:     '#f97316',
  DICER1:  '#8b5cf6',
  RB1:     '#3b82f6',
  HRAS:    '#10b981',
  BRCA2:   '#ec4899',
  PTPN11:  '#f59e0b',
  SMARCB1: '#6366f1',
};

const GENE_INFO = {
  TP53:    { full: 'Tumour Protein p53',                      locus: '17p13.1',  size: '393aa',  inh: 'AD LOF' },
  NF1:     { full: 'Neurofibromin 1',                         locus: '17q11.2',  size: '2839aa', inh: 'AD LOF' },
  DICER1:  { full: 'Dicer 1, Ribonuclease III',               locus: '14q32.13', size: '1922aa', inh: 'AD LOF' },
  RB1:     { full: 'Retinoblastoma 1',                        locus: '13q14.2',  size: '928aa',  inh: 'AD LOF' },
  HRAS:    { full: 'Harvey Ras Oncogene',                     locus: '11p15.5',  size: '189aa',  inh: 'AD GOF' },
  BRCA2:   { full: 'Breast Cancer Type 2 Susceptibility',     locus: '13q12.3',  size: '3418aa', inh: 'AD LOF / AR FA-D1' },
  PTPN11:  { full: 'Protein Tyrosine Phosphatase Non-Rec 11', locus: '12q24.13', size: '593aa',  inh: 'AD GOF' },
  SMARCB1: { full: 'SWI/SNF Matrix-Assoc Chromatin Reg B1',   locus: '22q11.23', size: '385aa',  inh: 'AD LOF' },
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

export default function HereditaryRhabdomyosarcomaAtlasPage() {
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
      Loading Hereditary Rhabdomyosarcoma Predisposition Atlas…
    </div>
  );
  if (error) return (
    <div style={{ padding: 48, textAlign: 'center', color: '#ef4444' }}>
      Error: {error}
    </div>
  );

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 16px', fontFamily: 'system-ui,sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <span style={{ fontSize: 32 }}>🧬</span>
          <div>
            <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#111827' }}>
              Hereditary Rhabdomyosarcoma Predisposition Atlas
            </h1>
            <p style={{ margin: '4px 0 0', fontSize: 14, color: '#6b7280' }}>
              Complete 8-Gene Reference: TP53 · NF1 · DICER1 · RB1 · HRAS · BRCA2 · PTPN11 · SMARCB1 &nbsp;|&nbsp;
              320-patient aggregate (8 × 40, seeds 3278–3285)
            </p>
          </div>
        </div>
        {/* Gene badges */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {genes.map(g => (
            <Badge key={g} label={`${g} ${GENE_INFO[g].locus}`} color={GENE_COLORS[g]} />
          ))}
        </div>
      </div>

      {/* Tab nav */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, borderBottom: '2px solid #e5e7eb' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', fontWeight: 600, fontSize: 14, border: 'none', cursor: 'pointer',
            background: 'none', borderBottom: tab === t ? '2px solid #7c3aed' : '2px solid transparent',
            color: tab === t ? '#7c3aed' : '#6b7280', marginBottom: -2,
          }}>{t}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'Overview' && overview && (
        <div>
          {/* KPI strip */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(130px,1fr))', gap: 12, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#7c3aed' },
              { label: 'CR Rate', value: `${overview.cr_pct}%`, color: '#10b981' },
              { label: 'Surgical Resection', value: `${overview.gtr_pct}%`, color: '#3b82f6' },
              { label: 'Targeted Therapy', value: `${overview.targeted_pct}%`, color: '#f59e0b' },
              { label: 'Radiation Used', value: `${overview.radiation_pct}%`, color: '#ef4444' },
              { label: 'Relapse Rate', value: `${overview.relapse_pct}%`, color: '#6b7280' },
              { label: 'Mean Age Dx (yr)', value: overview.mean_age_dx, color: '#6366f1' },
            ].map(k => (
              <div key={k.label} style={{ background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: 10, padding: '14px 16px' }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: k.color }}>{k.value}</div>
                <div style={{ fontSize: 11, color: '#6b7280', marginTop: 2 }}>{k.label}</div>
              </div>
            ))}
          </div>

          {/* Gene CR + radiation bars */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20 }}>
              <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 700, color: '#111827' }}>CR Rate by Gene (%)</h3>
              {(overview.gene_summaries || []).map(g => (
                <GeneBar key={g.gene} gene={g.gene} value={g.cr_pct} max={100} />
              ))}
            </div>
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20 }}>
              <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 700, color: '#111827' }}>Radiation Used by Gene (%)</h3>
              {(overview.gene_summaries || []).map(g => (
                <GeneBar key={g.gene} gene={g.gene} value={g.radiation_pct} max={100} />
              ))}
            </div>
          </div>

          {/* Top tumour types */}
          <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 20, marginBottom: 24 }}>
            <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 700, color: '#111827' }}>Top RMS Subtypes / Tumour Types</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {Object.entries(overview.top_tumor_types || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#f3f4f6', borderRadius: 8, padding: '8px 14px', fontSize: 13 }}>
                  <span style={{ fontWeight: 700 }}>{v}</span> <span style={{ color: '#6b7280' }}>{k}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Key management rules */}
          <div style={{ background: '#faf5ff', border: '1px solid #d8b4fe', borderRadius: 12, padding: 20, marginBottom: 24 }}>
            <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 700, color: '#6d28d9' }}>Key Management Rules</h3>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(overview.key_management_rules || []).map((r, i) => (
                <li key={i} style={{ fontSize: 13, color: '#374151', marginBottom: 6 }}>{r}</li>
              ))}
            </ul>
          </div>

          {/* Clinical pearls */}
          <div style={{ background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 12, padding: 20 }}>
            <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 700, color: '#92400e' }}>Clinical Pearls</h3>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(overview.clinical_pearls || []).map((p, i) => (
                <li key={i} style={{ fontSize: 13, color: '#374151', marginBottom: 6 }}>{p}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* ── Gene Table Tab ── */}
      {tab === 'Gene Table' && overview && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f3f4f6' }}>
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Patients', 'Mean Age', 'CR%', 'RT%', 'Targeted%', 'Cancer Risk'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700, color: '#374151', borderBottom: '2px solid #e5e7eb', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(overview.gene_summaries || []).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f9fafb' }}>
                  <td style={{ padding: '10px 12px', fontWeight: 800, color: GENE_COLORS[g.gene] }}>{g.gene}</td>
                  <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>{g.locus}</td>
                  <td style={{ padding: '10px 12px' }}>{GENE_INFO[g.gene]?.size}</td>
                  <td style={{ padding: '10px 12px', whiteSpace: 'nowrap' }}>{GENE_INFO[g.gene]?.inh}</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center' }}>{g.n_patients}</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center' }}>{g.mean_age_dx}</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center', color: '#10b981', fontWeight: 700 }}>{g.cr_pct}%</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center', color: '#ef4444', fontWeight: 700 }}>{g.radiation_pct}%</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center', color: '#f59e0b', fontWeight: 700 }}>{g.targeted_pct}%</td>
                  <td style={{ padding: '10px 12px', fontSize: 12, color: '#6b7280', maxWidth: 280 }}>{g.cancer_risk?.substring(0, 120)}…</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Clinical Atlas Tab ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {(breakdown.breakdown || []).map(gene => (
            <div key={gene.gene} style={{
              background: '#fff', border: `2px solid ${GENE_COLORS[gene.gene]}33`,
              borderRadius: 14, padding: 22, marginBottom: 20,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                <span style={{
                  background: GENE_COLORS[gene.gene], color: '#fff',
                  borderRadius: 8, padding: '4px 14px', fontWeight: 800, fontSize: 16
                }}>{gene.gene}</span>
                <span style={{ fontSize: 13, color: '#6b7280' }}>{gene.locus} · {GENE_INFO[gene.gene]?.size} · {gene.inheritance?.substring(0, 80)}</span>
              </div>

              {/* Stats row */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
                {[
                  { label: 'Patients', v: gene.n_patients, c: '#6366f1' },
                  { label: 'Mean Age', v: `${gene.mean_age_dx}yr`, c: '#374151' },
                  { label: 'CR%', v: `${gene.cr_pct}%`, c: '#10b981' },
                  { label: 'RT%', v: `${gene.radiation_pct}%`, c: '#ef4444' },
                  { label: 'Targeted%', v: `${gene.targeted_pct}%`, c: '#f59e0b' },
                  { label: 'Relapse%', v: `${gene.relapse_pct}%`, c: '#6b7280' },
                ].map(s => (
                  <div key={s.label} style={{ background: '#f9fafb', borderRadius: 8, padding: '6px 12px', fontSize: 12 }}>
                    <span style={{ fontWeight: 700, color: s.c }}>{s.v}</span>
                    <span style={{ color: '#9ca3af', marginLeft: 4 }}>{s.label}</span>
                  </div>
                ))}
              </div>

              {/* Pathognomonic */}
              <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 8, padding: '8px 14px', marginBottom: 12, fontSize: 13, color: '#991b1b' }}>
                <strong>PATHOGNOMONIC: </strong>{gene.pathognomonic}
              </div>

              {/* Top tumour types */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#6b7280', marginBottom: 6 }}>TOP TUMOUR TYPES</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {Object.entries(gene.top_tumor_types || {}).map(([t, n]) => (
                    <span key={t} style={{
                      background: GENE_COLORS[gene.gene] + '15',
                      color: GENE_COLORS[gene.gene],
                      border: `1px solid ${GENE_COLORS[gene.gene]}44`,
                      borderRadius: 6, padding: '2px 10px', fontSize: 12, fontWeight: 600
                    }}>{t} ({n})</span>
                  ))}
                </div>
              </div>

              {/* Top variants */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#6b7280', marginBottom: 6 }}>TOP PATHOGENIC VARIANTS</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {Object.entries(gene.top_variants || {}).map(([v, n]) => (
                    <span key={v} style={{
                      background: '#f3f4f6', color: '#374151',
                      border: '1px solid #d1d5db',
                      borderRadius: 6, padding: '2px 10px', fontSize: 11, fontFamily: 'monospace'
                    }}>{v} ({n})</span>
                  ))}
                </div>
              </div>

              {/* Treatment protocols */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#6b7280', marginBottom: 6 }}>TREATMENT PROTOCOLS</div>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {(gene.treatment_protocols || []).map((t, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#374151', marginBottom: 3 }}>{t}</li>
                  ))}
                </ul>
              </div>

              {/* Surveillance protocols */}
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#6b7280', marginBottom: 6 }}>SURVEILLANCE PROTOCOLS</div>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {(gene.surveillance_protocols || []).map((s, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#374151', marginBottom: 3 }}>{s}</li>
                  ))}
                </ul>
              </div>

              {/* Key distinctions */}
              {gene.key_distinctions && (
                <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {gene.key_distinctions.map((d, i) => (
                    <span key={i} style={{
                      background: '#ede9fe', color: '#5b21b6',
                      border: '1px solid #c4b5fd',
                      borderRadius: 4, padding: '1px 8px', fontSize: 10, fontWeight: 700
                    }}>{d}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          {/* Key rules */}
          {definitions.key_rules && Object.entries(definitions.key_rules).map(([k, v]) => (
            <div key={k} style={{ background: '#f0fdf4', border: '1px solid #86efac', borderRadius: 10, padding: 16, marginBottom: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 800, color: '#166534', marginBottom: 6 }}>{k.replace(/_/g, ' ')}</div>
              <div style={{ fontSize: 13, color: '#14532d', lineHeight: 1.6 }}>{v}</div>
            </div>
          ))}

          {/* Per-gene definitions */}
          {Object.entries(definitions.definitions || {}).map(([gene, gdef]) => {
            const summary = typeof gdef.protein_size === 'string' ? gdef.protein_size : '';
            const rules = gdef.key_distinctions || [];
            const pathAssoc = gdef.pathognomonic ? gdef.pathognomonic.split(';') : [];
            return (
              <div key={gene} style={{
                background: '#fff', border: `1px solid ${GENE_COLORS[gene] || '#e5e7eb'}44`,
                borderLeft: `4px solid ${GENE_COLORS[gene] || '#e5e7eb'}`,
                borderRadius: 10, padding: 18, marginBottom: 14,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <span style={{ fontWeight: 800, fontSize: 18, color: GENE_COLORS[gene] }}>{gene}</span>
                  <span style={{ fontSize: 12, color: '#6b7280' }}>{gdef.locus}</span>
                  <span style={{ fontSize: 12, color: '#6b7280' }}>·</span>
                  <span style={{ fontSize: 12, color: '#6b7280' }}>{GENE_INFO[gene]?.inh}</span>
                </div>
                <p style={{ margin: '0 0 10px', fontSize: 13, color: '#374151', lineHeight: 1.6 }}>
                  <strong>Cancer Risk: </strong>{gdef.cancer_risk}
                </p>
                <p style={{ margin: '0 0 10px', fontSize: 13, color: '#374151', lineHeight: 1.6 }}>
                  <strong>Surveillance: </strong>{gdef.surveillance_key}
                </p>
                {summary && (
                  <p style={{ margin: '0 0 10px', fontSize: 12, color: '#6b7280', lineHeight: 1.5, fontFamily: 'monospace', whiteSpace: 'pre-wrap', maxHeight: 180, overflowY: 'auto' }}>
                    {summary}
                  </p>
                )}
                {rules.length > 0 && (
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#6b7280', marginBottom: 6 }}>
                      KEY RULES
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                      {rules.map((r, i) => (
                        <li key={i} style={{ fontSize: 13, color: '#374151', marginBottom: 4 }}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {pathAssoc.length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#ef4444', marginBottom: 6 }}>
                      PATHOGNOMONIC ASSOCIATIONS
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {pathAssoc.map((a, i) => (
                        <span key={i} style={{
                          background: '#fef2f2', color: '#991b1b', border: '1px solid #fca5a5',
                          borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700
                        }}>{a.trim()}</span>
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
