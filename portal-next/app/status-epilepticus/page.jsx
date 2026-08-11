'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-2">
      <div className={`card border-${color || 'danger'} text-center h-100`}>
        <div className="card-body py-2 px-1">
          <div className={`h4 fw-bold mb-0 text-${color || 'danger'}`}>{value ?? '—'}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.68rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ items, labelKey = 'label', countKey = 'count', colorClass = 'danger' }) {
  const mx = Math.max(...(items || []).map(i => i[countKey] || 0), 1);
  return (
    <div>
      {(items || []).map((it, i) => {
        const val = it[countKey] ?? 0;
        const label = it[labelKey] || '?';
        const pct = Math.round((val / mx) * 100);
        return (
          <div key={i} className="d-flex align-items-center mb-1 gap-2">
            <div className="text-end small text-muted" style={{ width: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.75rem' }}>
              {label}
            </div>
            <div className="flex-grow-1">
              <div className="progress" style={{ height: 16 }}>
                <div className={`progress-bar bg-${colorClass}`} style={{ width: `${pct}%` }}>
                  <span className="small px-1">{val}</span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function StageBadge({ stage }) {
  const color = stage?.includes('Refractory') ? 'danger'
    : stage?.includes('Stage 2') ? 'warning'
    : 'success';
  return <span className={`badge bg-${color}`}>{stage || '—'}</span>;
}

function BoolBadge({ val, trueLabel = 'Yes', falseLabel = 'No' }) {
  return <span className={`badge bg-${val ? 'success' : 'secondary'}`}>{val ? trueLabel : falseLabel}</span>;
}

export default function StatusEpilepticus() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    const load = async (path, setter) => {
      try {
        const r = await fetch(`${API}${path}`);
        if (!r.ok) throw new Error(`${r.status}`);
        setter(await r.json());
      } catch (e) { setErr(e.message); }
    };
    load('/api/status-epilepticus/overview', setOv);
    load('/api/status-epilepticus/breakdown', setBk);
    load('/api/status-epilepticus/definitions', setDf);
  }, []);

  const kpis = ov?.kpis || {};

  return (
    <div className="container-fluid py-3">
      <h3 className="fw-bold mb-1">🚨 Status Epilepticus Dashboard</h3>
      <p className="text-muted small mb-3">
        Emergency seizure management · NCS 3-stage treatment algorithm · ICU outcomes
        · ILAE 2015 SE definition · real hospitalization data
      </p>

      {err && <div className="alert alert-danger small">API error: {err}</div>}

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {['overview', 'per-patient', 'trends', 'treatment', 'definitions'].map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
              {t === 'overview' ? '📊 Overview'
                : t === 'per-patient' ? '🏥 Per Patient'
                : t === 'trends' ? '📈 Trends'
                : t === 'treatment' ? '💉 Treatment'
                : '📖 Definitions'}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="SE Direct Admissions" value={kpis.se_direct_admissions} color="danger" />
            <KPI label="Seizure Clusters" value={kpis.seizure_cluster_admissions} color="warning" />
            <KPI label="Total Emergency" value={kpis.emergency_admissions} color="danger" />
            <KPI label="Seizure-Free Discharge" value={kpis.seizure_free_at_discharge != null ? `${kpis.seizure_free_pct}%` : '—'} color="success" sub={`${kpis.seizure_free_at_discharge ?? '—'} patients`} />
            <KPI label="30d Readmission" value={kpis.readmission_30d} color="warning" />
            <KPI label="Avg LOS (days)" value={kpis.avg_los_days} color="info" />
            <KPI label="Avg Cost (USD)" value={kpis.avg_cost_usd != null ? `$${kpis.avg_cost_usd.toLocaleString()}` : '—'} color="secondary" />
            <KPI label="Total Admissions" value={kpis.total_admissions} color="dark" />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small">🏥 Admission Reason Breakdown</div>
                <div className="card-body">
                  <Bar
                    items={(ov?.admission_reason_breakdown || []).map(i => ({ label: i.reason, count: i.count }))}
                    colorClass="danger"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small">⚡ NCS Stage Distribution</div>
                <div className="card-body">
                  <Bar
                    items={(ov?.ncs_stage_distribution || []).map(i => ({ label: i.stage, count: i.count }))}
                    colorClass="warning"
                  />
                  <div className="text-muted mt-2" style={{ fontSize: '0.72rem' }}>
                    Stage 1: Early SE · Stage 2: Established · Stage 3: Refractory (ICU)
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small">🏢 Ward Distribution (SE admissions)</div>
                <div className="card-body">
                  <Bar
                    items={(ov?.ward_distribution || []).map(i => ({ label: i.ward, count: i.count }))}
                    colorClass="primary"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small">🏠 Discharge Disposition</div>
                <div className="card-body">
                  <Bar
                    items={(ov?.discharge_disposition || []).map(i => ({ label: i.disposition, count: i.count }))}
                    colorClass="success"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small">🧬 Etiology Breakdown (SE patients)</div>
                <div className="card-body">
                  <Bar
                    items={(ov?.etiology_breakdown || []).map(i => ({ label: i.etiology, count: i.count }))}
                    colorClass="info"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small">⚠️ Complications</div>
                <div className="card-body">
                  {(ov?.complications || []).length === 0
                    ? <div className="text-muted small">No complications recorded in SE admissions</div>
                    : <Bar
                        items={(ov?.complications || []).map(i => ({ label: i.complication, count: i.count }))}
                        colorClass="danger"
                      />
                  }
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Per Patient ── */}
      {tab === 'per-patient' && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-4">
              <div className="card">
                <div className="card-header fw-bold small">📅 Length-of-Stay (LOS)</div>
                <div className="card-body">
                  <Bar
                    items={(bk?.los_histogram || []).map(i => ({ label: i.bucket, count: i.count }))}
                    colorClass="warning"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card">
                <div className="card-header fw-bold small">🩺 Drug-Resistant in SE</div>
                <div className="card-body">
                  {bk?.drug_resistant_in_se && (
                    <div>
                      <div className="h3 text-danger fw-bold">{bk.drug_resistant_in_se.pct}%</div>
                      <div className="small text-muted">{bk.drug_resistant_in_se.count} of {bk.drug_resistant_in_se.total} SE patients are drug-resistant</div>
                      <div className="mt-2 text-muted" style={{ fontSize: '0.72rem' }}>
                        DRE patients face higher SE recurrence — escalate to epilepsy surgery evaluation
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card">
                <div className="card-header fw-bold small">🏦 Insurance Type</div>
                <div className="card-body">
                  <Bar
                    items={(bk?.insurance_breakdown || []).map(i => ({ label: i.type, count: i.count }))}
                    colorClass="secondary"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header fw-bold small">🏥 Per-Patient SE Admission Table</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0 small">
                  <thead className="table-dark">
                    <tr>
                      <th>Patient</th>
                      <th>Reason</th>
                      <th>NCS Stage</th>
                      <th>Ward</th>
                      <th>LOS (d)</th>
                      <th>Seizure-Free</th>
                      <th>DRE</th>
                      <th>Readmit 30d</th>
                      <th>Disposition</th>
                      <th>Cost ($)</th>
                      <th>Syndrome</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bk?.patients || []).map((r, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{r.patient_id}</td>
                        <td>{r.admission_reason}</td>
                        <td><StageBadge stage={r.ncs_stage} /></td>
                        <td>{r.ward}</td>
                        <td>{r.los_days}</td>
                        <td><BoolBadge val={r.seizure_free_at_discharge} trueLabel="Yes" falseLabel="No" /></td>
                        <td><BoolBadge val={r.drug_resistant} trueLabel="DRE" falseLabel="—" /></td>
                        <td><BoolBadge val={r.readmission_30d} trueLabel="Yes" falseLabel="No" /></td>
                        <td>{r.discharge_disposition}</td>
                        <td>{r.cost_usd ? r.cost_usd.toLocaleString() : '—'}</td>
                        <td className="text-muted">{r.syndrome || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Trends ── */}
      {tab === 'trends' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-bold small">📅 Monthly SE Admissions Trend</div>
              <div className="card-body">
                {(bk?.monthly_trend || []).length === 0
                  ? <div className="text-muted small">No trend data available</div>
                  : <Bar
                      items={(bk.monthly_trend || []).map(i => ({ label: i.month, count: i.count }))}
                      colorClass="danger"
                    />
                }
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-bold small">🧬 Syndrome Breakdown in SE Patients</div>
              <div className="card-body">
                <Bar
                  items={(bk?.syndrome_breakdown || []).map(i => ({ label: i.syndrome || 'Unknown', count: i.count }))}
                  colorClass="info"
                />
              </div>
            </div>
          </div>
          <div className="col-md-12">
            <div className="card border-warning">
              <div className="card-header fw-bold small text-warning">💡 Cost Summary</div>
              <div className="card-body">
                {ov?.cost_summary && (
                  <div className="row text-center">
                    <div className="col-4">
                      <div className="h4 fw-bold text-danger">${(ov.cost_summary.total_se_cost_usd || 0).toLocaleString()}</div>
                      <div className="small text-muted">Total SE Admission Cost</div>
                    </div>
                    <div className="col-4">
                      <div className="h4 fw-bold text-warning">${(ov.cost_summary.avg_se_cost_usd || 0).toLocaleString()}</div>
                      <div className="small text-muted">Average Cost per SE Admission</div>
                    </div>
                    <div className="col-4">
                      <div className="h4 fw-bold text-info">{ov.cost_summary.se_admissions_n}</div>
                      <div className="small text-muted">SE-Related Admissions</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Treatment ── */}
      {tab === 'treatment' && df && (
        <div className="row g-3">
          <div className="col-12">
            <div className="alert alert-danger small fw-bold">
              🚨 SE is a neurological emergency. Early IV benzodiazepine within 5 minutes improves outcomes. Follow NCS/ILAE protocol.
            </div>
          </div>
          {(df.ncs_treatment_stages || []).map((s, i) => (
            <div key={i} className="col-md-4">
              <div className={`card border-${i === 0 ? 'success' : i === 1 ? 'warning' : 'danger'} h-100`}>
                <div className={`card-header fw-bold small text-${i === 0 ? 'success' : i === 1 ? 'warning' : 'danger'}`}>
                  {s.stage}
                </div>
                <div className="card-body">
                  <div className="small fw-bold mb-1">Agents:</div>
                  <div className="small text-muted mb-2">{s.agents}</div>
                  <div className="small fw-bold mb-1">Target:</div>
                  <div className="small text-success">{s.target}</div>
                </div>
              </div>
            </div>
          ))}
          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-bold small">⚠️ Prognostic Factors</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0 small">
                  <thead className="table-dark"><tr><th>Factor</th><th>Clinical Impact</th></tr></thead>
                  <tbody>
                    {(df.prognostic_factors || []).map((f, i) => (
                      <tr key={i}><td className="fw-bold">{f.factor}</td><td className="text-muted">{f.impact}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-bold small">💡 Clinical Pearls</div>
              <div className="card-body">
                <ul className="small mb-0">
                  {(df.clinical_pearls || []).map((p, i) => (
                    <li key={i} className="mb-1 text-muted">{p}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && df && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card border-danger">
              <div className="card-header fw-bold small text-danger">📖 {df.term}</div>
              <div className="card-body">
                <p className="small mb-1">{df.definition}</p>
                <div className="text-muted small"><strong>Incidence:</strong> {df.incidence}</div>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold small">🔬 SE Types</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0 small">
                  <thead className="table-dark"><tr><th>Type</th><th>Description</th></tr></thead>
                  <tbody>
                    {(df.se_types || []).map((t, i) => (
                      <tr key={i}><td className="fw-bold">{t.type}</td><td className="text-muted">{t.description}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold small">🔤 Abbreviations</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0 small">
                  <thead className="table-dark"><tr><th>Abbr.</th><th>Meaning</th></tr></thead>
                  <tbody>
                    {Object.entries(df.abbreviations || {}).map(([k, v], i) => (
                      <tr key={i}><td className="fw-bold">{k}</td><td className="text-muted">{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-bold small">📚 References</div>
              <div className="card-body">
                <ul className="small mb-0">
                  {(df.references || []).map((r, i) => (
                    <li key={i} className="text-muted">{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
