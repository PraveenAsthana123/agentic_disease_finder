'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const RESP_COLORS = {
  'Drug-Resistant': '#ef4444',
  'Partial Response': '#f59e0b',
  'Partial': '#f59e0b',
  'Drug-Responsive': '#22c55e',
  'Newly Diagnosed': '#6b7280',
  'Pending': '#6b7280',
};

const GENE_COLORS = { SCN1A: '#8b5cf6', KCNQ2: '#3b82f6', Familial: '#10b981' };

function StatCard({ label, value, sub, color = '#6366f1' }) {
  return (
    <div className="col-6 col-md mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h4 mb-0 fw-bold" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Badge({ label, colorMap }) {
  const bg = (colorMap && colorMap[label]) || '#6b7280';
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 4,
      fontSize: '0.75rem', fontWeight: 600, color: '#fff', backgroundColor: bg,
    }}>{label}</span>
  );
}

function MiniBar({ value, total, color }) {
  const pct = total ? Math.min(100, (value / total) * 100) : 0;
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 10, minWidth: 80 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color || '#6366f1' }} />
      </div>
      <span className="small fw-bold">{value}</span>
    </div>
  );
}

export default function GeneticEpilepsyDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [patSort, setPatSort] = useState('patient_id');
  const [patDir, setPatDir] = useState(1);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/genetic-epilepsy/overview`).then(r => r.json()),
      fetch(`${API}/api/genetic-epilepsy/breakdown`).then(r => r.json()),
      fetch(`${API}/api/genetic-epilepsy/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading genetic epilepsy data…</div>;

  const kpi = ov.kpis || {};
  const totalPts = kpi.total_genetic_cases || 0;

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'genes', label: 'Gene Profiles' },
    { id: 'patients', label: 'Per Patient' },
    { id: 'pgx', label: 'SCN1A PGx' },
    { id: 'defs', label: 'Definitions' },
  ];

  /* sortable patient table */
  const sortedPats = bd
    ? [...bd.per_patient].sort((a, b) => {
        const av = a[patSort] ?? '';
        const bv = b[patSort] ?? '';
        return patDir * (av < bv ? -1 : av > bv ? 1 : 0);
      })
    : [];

  const thSort = (col) => {
    if (patSort === col) setPatDir(d => -d);
    else { setPatSort(col); setPatDir(1); }
  };

  const arrow = (col) => patSort === col ? (patDir > 0 ? ' ▲' : ' ▼') : '';

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center justify-content-between mb-3 flex-wrap gap-2">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: '#7c3aed' }}>🧬 Genetic Epilepsy Syndromes</h4>
          <small className="text-muted">SCN1A · KCNQ2 · Familial — {totalPts} genetically-linked patients</small>
        </div>
      </div>

      {/* KPI row */}
      <div className="row g-2 mb-3">
        <StatCard label="Genetic Cases" value={kpi.total_genetic_cases} color="#7c3aed" />
        <StatCard label="Distinct Genes" value={kpi.distinct_genes} color="#3b82f6" />
        <StatCard label="Drug-Resistant" value={kpi.drug_resistant_count} color="#ef4444" sub="failed ≥2 AEDs" />
        <StatCard label="Drug-Responsive" value={kpi.drug_responsive_count} color="#22c55e" />
        <StatCard label="Pediatric Onset" value={kpi.pediatric_onset_count} color="#f59e0b" sub="age ≤18 y" />
        <StatCard label="Avg Onset Age" value={`${kpi.avg_age_at_onset} y`} color="#6b7280" />
        <StatCard label="SCN1A PGx Records" value={kpi.scn1a_pgx_records} color="#8b5cf6" sub="drug-gene links" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Gene Distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Gene Distribution</div>
              <div className="card-body">
                {(ov.gene_distribution || []).map(g => (
                  <div key={g.gene} className="mb-3">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="fw-semibold" style={{ color: GENE_COLORS[g.gene] || '#555' }}>{g.gene}</span>
                      <span className="small text-muted">{g.count} ({g.pct}%)</span>
                    </div>
                    <MiniBar value={g.count} total={totalPts} color={GENE_COLORS[g.gene]} />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Drug Response */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Drug Response Distribution</div>
              <div className="card-body">
                {(ov.drug_response_distribution || []).map(d => (
                  <div key={d.response} className="mb-3">
                    <div className="d-flex justify-content-between mb-1">
                      <Badge label={d.response} colorMap={RESP_COLORS} />
                      <span className="small text-muted">{d.count}/{totalPts}</span>
                    </div>
                    <MiniBar value={d.count} total={totalPts} color={RESP_COLORS[d.response]} />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Syndrome Distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Syndrome Distribution</div>
              <div className="card-body">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Syndrome</th><th className="text-end">n</th></tr></thead>
                  <tbody>
                    {(ov.syndrome_distribution || []).map(s => (
                      <tr key={s.syndrome}>
                        <td className="small">{s.syndrome}</td>
                        <td className="text-end"><span className="badge bg-secondary">{s.count}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Gene × Response Matrix */}
          {bd && bd.gene_response_matrix && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Gene × Drug Response Matrix</div>
                <div className="card-body p-0 table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Gene</th>
                        {['Drug-Resistant', 'Partial Response', 'Partial', 'Drug-Responsive', 'Newly Diagnosed', 'Pending'].map(h => (
                          <th key={h} className="text-center small">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {bd.gene_response_matrix.map(row => (
                        <tr key={row.gene}>
                          <td className="fw-semibold" style={{ color: GENE_COLORS[row.gene] || '#333' }}>{row.gene}</td>
                          {['Drug-Resistant', 'Partial Response', 'Partial', 'Drug-Responsive', 'Newly Diagnosed', 'Pending'].map(h => (
                            <td key={h} className="text-center">
                              {row[h] ? (
                                <span className="badge" style={{ backgroundColor: RESP_COLORS[h] || '#999' }}>{row[h]}</span>
                              ) : <span className="text-muted">—</span>}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Onset zone */}
          {bd && bd.onset_distribution && (
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">EEG Onset Zone (Genetic Cases)</div>
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0">
                    <thead><tr><th>Onset Zone</th><th className="text-end">n</th></tr></thead>
                    <tbody>
                      {bd.onset_distribution.map(o => (
                        <tr key={o.onset_zone}>
                          <td className="small">{o.onset_zone}</td>
                          <td className="text-end"><span className="badge bg-info">{o.count}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* GENE PROFILES */}
      {tab === 'genes' && defs && (
        <div className="row g-3">
          {(defs.genes || []).map(g => (
            <div key={g.gene} className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ backgroundColor: GENE_COLORS[g.gene] || '#6b7280', color: '#fff' }}>
                  {g.gene} — {g.full_name}
                </div>
                <div className="card-body">
                  <p className="small mb-1"><strong>Inheritance:</strong> {g.inheritance}</p>
                  <p className="small mb-2"><strong>Associated Syndromes:</strong></p>
                  <ul className="small ps-3 mb-2">
                    {(g.associated_syndromes || []).map(s => <li key={s}>{s}</li>)}
                  </ul>
                  <div className="alert alert-light p-2 mb-0 small">
                    <strong>Clinical Note:</strong> {g.clinical_note}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Drug response tiers */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Drug Response Tier Criteria</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light"><tr><th>Tier</th><th>ILAE Definition</th><th>Management</th></tr></thead>
                  <tbody>
                    {(defs.drug_response_tiers || []).map(t => (
                      <tr key={t.tier}>
                        <td><Badge label={t.tier} colorMap={RESP_COLORS} /></td>
                        <td className="small">{t.definition}</td>
                        <td className="small text-muted">{t.typical_management}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Syndromes */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Common Genetic Epilepsy Syndromes</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light"><tr><th>Syndrome</th><th>Typical Onset</th><th>Long-term Prognosis</th></tr></thead>
                  <tbody>
                    {(defs.syndromes || []).map(s => (
                      <tr key={s.name}>
                        <td className="small fw-semibold">{s.name}</td>
                        <td className="small">{s.onset}</td>
                        <td className="small text-muted">{s.prognosis}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* PER PATIENT */}
      {tab === 'patients' && bd && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Per-Patient Genetic Profile ({sortedPats.length} patients)</div>
          <div className="card-body p-0 table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  {[
                    ['patient_id', 'Patient'],
                    ['gene', 'Gene'],
                    ['syndrome', 'Syndrome'],
                    ['age_at_onset', 'Onset Age'],
                    ['onset_zone', 'Onset Zone'],
                    ['drug_response', 'Drug Response'],
                    ['seizure_frequency', 'Frequency'],
                    ['surgery_candidacy', 'Surgery'],
                  ].map(([col, label]) => (
                    <th key={col} style={{ cursor: 'pointer' }} onClick={() => thSort(col)}>
                      {label}{arrow(col)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedPats.map(p => (
                  <tr key={p.patient_id}>
                    <td className="small fw-semibold">{p.patient_id}</td>
                    <td><Badge label={p.gene} colorMap={GENE_COLORS} /></td>
                    <td className="small">{p.syndrome}</td>
                    <td className="text-center">{p.age_at_onset ?? '—'}</td>
                    <td className="small">{p.onset_zone}</td>
                    <td>
                      <span className={`badge bg-${p.drug_response_class}`}>
                        {p.drug_response}
                      </span>
                    </td>
                    <td className="small">{p.seizure_frequency}</td>
                    <td className="small text-muted">{p.surgery_candidacy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* SCN1A PGx */}
      {tab === 'pgx' && bd && (
        <div>
          <div className="alert alert-info mb-3 small">
            <strong>SCN1A Pharmacogenomics ({bd.scn1a_pgx_records?.length || 0} records)</strong> —
            SCN1A variants affect sodium-channel drug metabolism. Sodium-channel blockers
            (phenytoin, carbamazepine, lamotrigine) may be contraindicated in loss-of-function SCN1A variants (Dravet syndrome).
          </div>
          <div className="card shadow-sm">
            <div className="card-body p-0 table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Variant</th>
                    <th>Allele Function</th>
                    <th>Metabolizer Status</th>
                    <th>Clinical Significance</th>
                    <th>Affected Drugs</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.scn1a_pgx_records || []).map((r, i) => (
                    <tr key={i}>
                      <td className="small fw-semibold">{r.patient_id}</td>
                      <td className="small font-monospace">{r.variant}</td>
                      <td className="small">{r.allele_function}</td>
                      <td className="small">{r.metabolizer_status}</td>
                      <td className="small">
                        <span className={`badge ${r.clinical_significance?.includes('High') ? 'bg-danger' : r.clinical_significance?.includes('Moderate') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                          {r.clinical_significance}
                        </span>
                      </td>
                      <td className="small text-muted">{r.affected_drugs}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS */}
      {tab === 'defs' && defs && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">About This Dashboard</div>
              <div className="card-body small">{defs.description}</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Metric Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Metric</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(defs.metrics || []).map(m => (
                      <tr key={m.metric}>
                        <td className="small fw-semibold">{m.metric}</td>
                        <td className="small text-muted">{m.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">References</div>
              <div className="card-body">
                <ol className="small ps-3 mb-0">
                  {(defs.references || []).map((r, i) => <li key={i} className="mb-1">{r}</li>)}
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
