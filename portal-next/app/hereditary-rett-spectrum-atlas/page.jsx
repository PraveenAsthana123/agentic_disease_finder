'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-rett-spectrum-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MECP2':   '#1565c0',  // deep blue   — Classic Rett; hand stereotypies + breathing irregularities PATHOGNOMONIC
  'CDKL5':   '#6a1b9a',  // deep purple — CDD; early-onset seizures <5 months PATHOGNOMONIC
  'FOXG1':   '#b71c1c',  // deep red    — Congenital Rett; no regression; dyskinesia + hypersalivation
  'MEF2C':   '#e65100',  // deep orange — Hyperkinesis; myeloid leukemia risk UNIQUE; 5q14.3 del
  'WDR45':   '#1b5e20',  // dark green  — BPAN; biphasic; iron accumulation GP+SN PATHOGNOMONIC; deferiprone
  'DDX3X':   '#004d40',  // dark teal   — Most common XL-ID in females; ASD 50%; CC anomalies
  'PURA':    '#f57f17',  // amber       — Excessive daytime sleepiness PATHOGNOMONIC; neonatal PICU
  'HNRNPH2': '#37474f',  // blue-grey   — Prominent forehead + hypertelorism + broad nasal tip PATHOGNOMONIC
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color,
      border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px',
      fontSize: 11, fontWeight: 600, marginRight: 4,
    }}>{text}</span>
  );
}

export default function HereditaryRettSpectrumAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [expandedGene, setExpandedGene] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const ep = tab === 'Definitions' ? 'definitions' : tab === 'Overview' ? 'overview' : 'breakdown';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(data => {
        if (tab === 'Overview') setOverview(data);
        else if (tab === 'Definitions') setDefinitions(data);
        else setBreakdown(data);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, marginBottom: 4 }}>
          🧬 Hereditary Rett-Spectrum Atlas
        </h1>
        <p style={{ color: '#555', marginBottom: 8 }}>
          Complete 8-Gene Reference: MECP2 · CDKL5 · FOXG1 · MEF2C · WDR45 · DDX3X · PURA · HNRNPH2
        </p>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <Badge text="320 Patients" color="#1565c0" />
          <Badge text="8 Genes" color="#6a1b9a" />
          <Badge text="Seeds 3054–3061" color="#37474f" />
          <Badge text="Rett-Spectrum" color="#b71c1c" />
          <Badge text="X-linked + AD de novo" color="#1b5e20" />
        </div>
      </div>

      {/* Gene colour chips */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
        {Object.entries(GENE_COLORS).map(([g, c]) => (
          <span key={g} style={{
            background: c + '18', border: `1px solid ${c}44`,
            borderRadius: 20, padding: '3px 12px',
            fontSize: 13, fontWeight: 600, color: c,
          }}>{g}</span>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
              style={{ cursor: 'pointer' }}
            >{t}</button>
          </li>
        ))}
      </ul>

      {loading && <div className="text-center py-5"><div className="spinner-border" /></div>}
      {error && <div className="alert alert-danger">{error}</div>}

      {/* Overview Tab */}
      {tab === 'Overview' && overview && !loading && (
        <div>
          <div className="row g-3 mb-4">
            {[
              { label: 'Total Genes', value: overview.total_genes, color: '#1565c0' },
              { label: 'Total Patients', value: overview.total_patients, color: '#6a1b9a' },
              { label: 'Seed Range', value: overview.seed_range, color: '#b71c1c' },
              { label: 'Cohort Design', value: '8 × 40', color: '#1b5e20' },
            ].map(s => (
              <div key={s.label} className="col-6 col-md-3">
                <div className="card border-0 h-100" style={{ background: s.color + '11' }}>
                  <div className="card-body text-center py-3">
                    <div style={{ fontSize: 28, fontWeight: 700, color: s.color }}>{s.value}</div>
                    <div style={{ fontSize: 12, color: '#666' }}>{s.label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Subtitle */}
          <div className="alert" style={{ background: '#e3f2fd', border: '1px solid #90caf9', borderRadius: 8 }}>
            <strong>{overview.subtitle}</strong>
          </div>

          {/* Inheritance / clinical summary */}
          <h5 className="mt-4 mb-3" style={{ fontWeight: 700 }}>Gene Inheritance & Key Clinical Rules</h5>
          <div className="row g-3 mb-4">
            {Object.entries(overview.inheritance_modes || {}).map(([gene, inh]) => (
              <div key={gene} className="col-12 col-md-6">
                <div className="card border-0 h-100" style={{ background: (GENE_COLORS[gene] || '#888') + '10', border: `1px solid ${(GENE_COLORS[gene] || '#888')}33` }}>
                  <div className="card-body py-2 px-3">
                    <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#333', marginBottom: 4, fontSize: 15 }}>{gene}</div>
                    <div style={{ fontSize: 12, color: '#444', lineHeight: 1.5 }}>{inh}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Key clinical rules */}
          <h5 className="mb-3" style={{ fontWeight: 700 }}>Clinical Action Rules</h5>
          <ul style={{ paddingLeft: 20 }}>
            {(overview.key_clinical_rules || []).map((r, i) => (
              <li key={i} style={{ marginBottom: 8, fontSize: 13, lineHeight: 1.6 }}>{r}</li>
            ))}
          </ul>

          {/* Panel note */}
          {overview.gene_panel_note && (
            <div className="alert mt-3" style={{ background: '#f3e5f5', border: '1px solid #ce93d8', fontSize: 13, borderRadius: 8 }}>
              <strong>Gene Panel Note:</strong> {overview.gene_panel_note}
            </div>
          )}

          {/* Gene loci table */}
          <h5 className="mt-4 mb-3" style={{ fontWeight: 700 }}>Gene Loci</h5>
          <table className="table table-sm table-bordered" style={{ fontSize: 13 }}>
            <thead className="table-light">
              <tr>
                <th>Gene</th>
                <th>Locus</th>
                <th>Color</th>
              </tr>
            </thead>
            <tbody>
              {(overview.genes || []).map(g => (
                <tr key={g}>
                  <td style={{ fontWeight: 700, color: GENE_COLORS[g] || '#333' }}>{g}</td>
                  <td>{overview.gene_loci?.[g] || '—'}</td>
                  <td>
                    <span style={{
                      display: 'inline-block', width: 16, height: 16,
                      borderRadius: 3, background: GENE_COLORS[g] || '#888',
                      verticalAlign: 'middle',
                    }} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && !loading && (
        <div>
          <h5 className="mb-3" style={{ fontWeight: 700 }}>Per-Gene Breakdown — {breakdown.count} genes, 40 patients each</h5>
          <div className="table-responsive">
            <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
              <thead className="table-dark">
                <tr>
                  <th>Gene</th><th>Locus</th><th>n</th><th>Severe%</th><th>Mod%</th>
                  <th>Mean IQ</th><th>Epilepsy%</th><th>SpeechAbsent%</th>
                  <th>Walk%</th><th>HandStereo%</th><th>BreathIrreg%</th>
                  <th>Regression%</th><th>Scoliosis%</th><th>Autism%</th>
                  <th>Gastrostomy%</th><th>CorpusCal%</th>
                  <th>Parkinson%</th><th>Myeloid%</th><th>Sleepy%</th><th>IronMRI%</th>
                  <th>MeanDxMo</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.genes || []).map(g => {
                  const color = GENE_COLORS[g.gene] || '#333';
                  return (
                    <tr key={g.gene} style={{ background: color + '09' }}>
                      <td style={{ fontWeight: 700, color }}>{g.gene}</td>
                      <td style={{ fontSize: 11 }}>{g.locus}</td>
                      <td>{g.n_patients}</td>
                      <td>{g.severe_pct}</td>
                      <td>{g.moderate_pct}</td>
                      <td>{g.mean_iq}</td>
                      <td style={{ fontWeight: g.epilepsy_pct > 70 ? 700 : 400, color: g.epilepsy_pct > 70 ? '#c62828' : 'inherit' }}>{g.epilepsy_pct}</td>
                      <td>{g.speech_absent_pct}</td>
                      <td>{g.independent_walk_pct}</td>
                      <td style={{ fontWeight: g.hand_stereo_pct > 80 ? 700 : 400, color: g.hand_stereo_pct > 80 ? '#1565c0' : 'inherit' }}>{g.hand_stereo_pct}</td>
                      <td style={{ fontWeight: g.breath_irreg_pct > 50 ? 700 : 400, color: g.breath_irreg_pct > 50 ? '#b71c1c' : 'inherit' }}>{g.breath_irreg_pct}</td>
                      <td>{g.regression_pct}</td>
                      <td>{g.scoliosis_pct}</td>
                      <td>{g.autism_pct}</td>
                      <td>{g.gastrostomy_pct}</td>
                      <td>{g.corpus_callosum_pct}</td>
                      <td style={{ fontWeight: g.parkinsonism_pct > 30 ? 700 : 400, color: g.parkinsonism_pct > 30 ? '#1b5e20' : 'inherit' }}>{g.parkinsonism_pct}</td>
                      <td style={{ fontWeight: g.myeloid_risk_pct > 0 ? 700 : 400, color: g.myeloid_risk_pct > 0 ? '#e65100' : 'inherit' }}>{g.myeloid_risk_pct}</td>
                      <td style={{ fontWeight: g.daytime_sleepiness_pct > 60 ? 700 : 400, color: g.daytime_sleepiness_pct > 60 ? '#f57f17' : 'inherit' }}>{g.daytime_sleepiness_pct}</td>
                      <td style={{ fontWeight: g.iron_accumulation_pct > 40 ? 700 : 400, color: g.iron_accumulation_pct > 40 ? '#1b5e20' : 'inherit' }}>{g.iron_accumulation_pct}</td>
                      <td>{g.mean_age_dx_mo}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Gene cards */}
          <h5 className="mt-4 mb-3" style={{ fontWeight: 700 }}>Gene Cards</h5>
          <div className="row g-3">
            {(breakdown.genes || []).map(g => {
              const color = GENE_COLORS[g.gene] || '#333';
              const isExp = expandedGene === g.gene;
              return (
                <div key={g.gene} className="col-12 col-md-6">
                  <div className="card border-0" style={{ border: `1px solid ${color}44`, background: color + '09' }}>
                    <div className="card-body py-2 px-3">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ fontWeight: 700, color, fontSize: 16 }}>{g.gene}</div>
                        <button className="btn btn-sm" style={{ fontSize: 11, padding: '2px 8px' }}
                          onClick={() => setExpandedGene(isExp ? null : g.gene)}>
                          {isExp ? 'Collapse' : 'Expand'}
                        </button>
                      </div>
                      <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>
                        {g.locus} · {g.disease_category} · {g.n_patients} pts
                      </div>
                      <div style={{ fontSize: 11, marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        <Badge text={`Epilepsy ${g.epilepsy_pct}%`} color="#c62828" />
                        <Badge text={`Speech absent ${g.speech_absent_pct}%`} color="#333" />
                        <Badge text={`Walk ${g.independent_walk_pct}%`} color="#1565c0" />
                        {g.parkinsonism_pct > 5 && <Badge text={`Parkinsonism ${g.parkinsonism_pct}%`} color="#1b5e20" />}
                        {g.myeloid_risk_pct > 0 && <Badge text={`Myeloid risk ${g.myeloid_risk_pct}%`} color="#e65100" />}
                        {g.daytime_sleepiness_pct > 50 && <Badge text={`EDS ${g.daytime_sleepiness_pct}%`} color="#f57f17" />}
                        {g.iron_accumulation_pct > 30 && <Badge text={`Iron MRI ${g.iron_accumulation_pct}%`} color="#1b5e20" />}
                      </div>
                      {isExp && (
                        <div style={{ marginTop: 10, fontSize: 12, borderTop: `1px solid ${color}33`, paddingTop: 8 }}>
                          <div><strong>Mutations seen:</strong> {g.sample_mutations?.join(', ')}</div>
                          <div style={{ marginTop: 6 }}><strong>Inheritance:</strong> {g.inheritance}</div>
                          <div style={{ marginTop: 6, fontSize: 11, color: '#555' }}>{g.protein}</div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && !loading && (
        <div>
          <h5 className="mb-3" style={{ fontWeight: 700 }}>Clinical Atlas — Rett-Spectrum Feature Comparison</h5>
          {(breakdown.genes || []).map(g => {
            const color = GENE_COLORS[g.gene] || '#333';
            return (
              <div key={g.gene} className="card border-0 mb-3" style={{ border: `1px solid ${color}33`, background: color + '06' }}>
                <div className="card-body">
                  <h6 style={{ color, fontWeight: 700, marginBottom: 8 }}>{g.gene} — {g.disease_category}</h6>
                  <div className="row g-2">
                    {[
                      ['Epilepsy', g.epilepsy_pct, '#c62828'],
                      ['Speech Absent', g.speech_absent_pct, '#333'],
                      ['Independent Walk', g.independent_walk_pct, '#1565c0'],
                      ['Hand Stereotypies', g.hand_stereo_pct, '#6a1b9a'],
                      ['Breathing Irregular', g.breath_irreg_pct, '#b71c1c'],
                      ['Regression', g.regression_pct, '#e65100'],
                      ['Scoliosis', g.scoliosis_pct, '#607d8b'],
                      ['Autism Features', g.autism_pct, '#795548'],
                      ['Gastrostomy', g.gastrostomy_pct, '#009688'],
                      ['Corpus Cal Anom', g.corpus_callosum_pct, '#455a64'],
                      ['Parkinsonism', g.parkinsonism_pct, '#1b5e20'],
                      ['Myeloid Risk', g.myeloid_risk_pct, '#e65100'],
                      ['Daytime Sleepiness', g.daytime_sleepiness_pct, '#f57f17'],
                      ['Iron on MRI', g.iron_accumulation_pct, '#1b5e20'],
                    ].map(([label, pct, barColor]) => (
                      <div key={label} className="col-12 col-md-6">
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                          <span style={{ width: 130, fontSize: 11, color: '#555', flexShrink: 0 }}>{label}</span>
                          <div style={{
                            height: 14, width: `${pct}%`, maxWidth: '100%',
                            background: barColor + 'cc', borderRadius: 3, minWidth: 2,
                            transition: 'width 0.4s',
                          }} />
                          <span style={{ fontSize: 11, color: barColor, fontWeight: 600, whiteSpace: 'nowrap' }}>{pct}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginTop: 8, fontSize: 11, color: '#666' }}>
                    Mean IQ {g.mean_iq} · Mean Dx {g.mean_age_dx_mo}mo · Severe {g.severe_pct}% · Moderate {g.moderate_pct}%
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && !loading && (
        <div>
          <h5 className="mb-3" style={{ fontWeight: 700 }}>Clinical Definitions — {definitions.count} entries</h5>
          {(definitions.definitions || []).map((d, i) => (
            <div key={i} className="card border-0 mb-4" style={{ border: '1px solid #e0e0e0', borderRadius: 8 }}>
              <div className="card-body">
                <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 8, color: '#1565c0' }}>{d.term}</div>
                <div style={{ marginBottom: 8 }}>
                  {(d.genes || []).map(g => (
                    <Badge key={g} text={g} color={GENE_COLORS[g] || '#888'} />
                  ))}
                </div>
                <pre style={{
                  whiteSpace: 'pre-wrap', fontFamily: 'inherit',
                  fontSize: 12, lineHeight: 1.7, color: '#333',
                  background: '#f8f9fa', borderRadius: 6, padding: '12px 14px',
                  border: '1px solid #dee2e6', margin: 0,
                }}>{d.definition}</pre>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
