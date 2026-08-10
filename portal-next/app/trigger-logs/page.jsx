'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TRIGGER_COLORS = {
  sleep_deprivation: 'primary',
  photosensitivity: 'warning',
  missed_medication: 'danger',
  stress: 'info',
  hormonal_changes: 'secondary',
  alcohol: 'dark',
  illness: 'success',
  fatigue: 'warning',
  dehydration: 'info',
};
function triggerColor(t) { return TRIGGER_COLORS[t] || 'secondary'; }

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 mb-1 text-${color || 'primary'}`}>{value}</div>
          <div className="text-muted small fw-semibold">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function HBar({ items, colorFn, maxWidth = 100 }) {
  if (!items || !items.length) return null;
  const mx = Math.max(...items.map(i => i.value));
  return (
    <div>
      {items.map((it, i) => (
        <div key={i} className="d-flex align-items-center mb-2">
          <div className="text-end me-2 small" style={{ width: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {it.label}
          </div>
          <div className="flex-grow-1">
            <div className="progress" style={{ height: 20 }}>
              <div
                className={`progress-bar bg-${colorFn ? colorFn(it) : it.color || 'primary'}`}
                style={{ width: `${mx ? ((it.value / mx) * maxWidth) : 0}%` }}
              >
                <span className="small px-1">{it.value}{it.pct !== undefined ? ` (${it.pct}%)` : ''}</span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function MonthChart({ months }) {
  if (!months || !months.length) return null;
  const maxSeiz = Math.max(...months.map(m => m.seizures));
  const maxLogs = Math.max(...months.map(m => m.total_logs));
  return (
    <div className="table-responsive">
      <table className="table table-sm table-bordered">
        <thead className="table-light">
          <tr>
            <th>Month</th>
            <th>Total Logs</th>
            <th>Seizure Days</th>
            <th>Seizure Rate</th>
            <th>Trend</th>
          </tr>
        </thead>
        <tbody>
          {months.map((m, i) => {
            const rate = m.total_logs > 0 ? ((m.seizures / m.total_logs) * 100).toFixed(1) : 0;
            return (
              <tr key={i}>
                <td><strong>{m.month}</strong></td>
                <td>{m.total_logs}</td>
                <td><span className={`badge bg-${m.seizures > 10 ? 'danger' : m.seizures > 5 ? 'warning' : 'success'}`}>{m.seizures}</span></td>
                <td>{rate}%</td>
                <td>
                  <div className="progress" style={{ height: 14 }}>
                    <div className="progress-bar bg-danger" style={{ width: `${maxSeiz ? (m.seizures / maxSeiz) * 100 : 0}%` }} />
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function TriggerLogsDashboard() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [sortCol, setSortCol] = useState('seizure_rate');
  const [sortDir, setSortDir] = useState('desc');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/trigger-logs/overview`).then(r => r.json()),
      fetch(`${API}/api/trigger-logs/breakdown`).then(r => r.json()),
      fetch(`${API}/api/trigger-logs/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpis = overview.kpis || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'lifestyle', label: 'Lifestyle Analysis' },
    { id: 'highrisk', label: 'High-Risk Days' },
    { id: 'patients', label: 'Per Patient' },
    { id: 'recent', label: 'Recent Logs' },
    { id: 'defs', label: 'Definitions' },
  ];

  // Per-patient sorting
  const patients = [...(breakdown?.per_patient || [])];
  patients.sort((a, b) => {
    const av = a[sortCol] ?? 0, bv = b[sortCol] ?? 0;
    return sortDir === 'asc' ? av - bv : bv - av;
  });
  function toggleSort(col) {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('desc'); }
  }
  function SortIcon({ col }) {
    if (sortCol !== col) return <span className="text-muted ms-1">⇅</span>;
    return <span className="ms-1">{sortDir === 'asc' ? '▲' : '▼'}</span>;
  }

  const trigItems = (overview.primary_trigger_distribution || []).map(t => ({
    label: t.trigger.replace(/_/g, ' '),
    value: t.count,
    pct: t.pct,
    color: triggerColor(t.trigger),
  }));

  const sleepItems = (overview.sleep_quality_distribution || []).map(s => ({
    label: s.quality.replace(/_/g, ' '),
    value: s.count,
    color: s.quality === 'good' ? 'success' : s.quality === 'fair' ? 'info' : s.quality === 'poor' ? 'warning' : 'danger',
  }));

  return (
    <div>
      <h3>&#x1f4c5; Trigger Logs &amp; Lifestyle Diary</h3>
      <p className="text-muted">
        Daily lifestyle diary with seizure trigger analysis — real <code>trigger_logs</code> table, {kpis.total_logs} entries, {kpis.total_patients} patients.
      </p>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Total Logs" value={kpis.total_logs} color="primary" />
        <KPI label="Patients" value={kpis.total_patients} color="info" />
        <KPI label="Seizure Rate" value={`${kpis.seizure_rate_pct}%`} color="danger" sub="days with seizure" />
        <KPI label="Seizure Events" value={kpis.total_seizure_events} color="warning" />
        <KPI label="Avg Sleep" value={`${kpis.avg_sleep_hours}h`} color="success" sub="per log day" />
        <KPI label="Med Adherence" value={`${kpis.medication_adherence_pct}%`} color="secondary" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div>
          <div className="row">
            <div className="col-md-6 mb-4">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Primary Trigger Distribution</div>
                <div className="card-body">
                  <HBar items={trigItems} colorFn={it => triggerColor(it.label.replace(/ /g, '_'))} />
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-4">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Sleep Quality Distribution</div>
                <div className="card-body">
                  <div className="row text-center">
                    {sleepItems.map((s, i) => (
                      <div key={i} className="col-3">
                        <div className={`badge bg-${s.color} fs-6 mb-1`}>{s.value}</div>
                        <div className="small text-capitalize text-muted">{s.label}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Monthly Seizure Trend</div>
            <div className="card-body">
              <MonthChart months={overview.seizure_by_month} />
            </div>
          </div>
        </div>
      )}

      {/* LIFESTYLE ANALYSIS TAB */}
      {tab === 'lifestyle' && (
        <div>
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Lifestyle Factors: With vs Without Seizure</div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>Factor</th>
                      <th className="text-danger">With Seizure</th>
                      <th className="text-success">Without Seizure</th>
                      <th>Difference</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(overview.lifestyle_averages || []).map((f, i) => {
                      const diff = (f.with_seizure - f.without_seizure).toFixed(2);
                      const worse = parseFloat(diff) > 0;
                      return (
                        <tr key={i}>
                          <td><strong>{f.factor}</strong></td>
                          <td className="text-danger">{f.with_seizure.toFixed(2)}</td>
                          <td className="text-success">{f.without_seizure.toFixed(2)}</td>
                          <td>
                            <span className={`badge bg-${worse ? 'danger' : 'success'}`}>
                              {worse ? '+' : ''}{diff}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Stress Level vs Seizure Rate</div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-bordered">
                  <thead className="table-light">
                    <tr><th>Stress Level</th><th>Log Count</th><th>Seizure %</th><th>Risk Bar</th></tr>
                  </thead>
                  <tbody>
                    {(overview.stress_vs_seizure || []).map((s, i) => (
                      <tr key={i}>
                        <td><span className={`badge bg-${s.stress_level >= 7 ? 'danger' : s.stress_level >= 4 ? 'warning' : 'success'}`}>{s.stress_level}/10</span></td>
                        <td>{s.count}</td>
                        <td>{s.seizure_pct.toFixed(1)}%</td>
                        <td>
                          <div className="progress" style={{ height: 16 }}>
                            <div className={`progress-bar bg-${s.seizure_pct > 30 ? 'danger' : s.seizure_pct > 20 ? 'warning' : 'success'}`}
                              style={{ width: `${s.seizure_pct}%` }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* HIGH-RISK DAYS TAB */}
      {tab === 'highrisk' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">High-Risk Days <span className="badge bg-danger ms-2">{breakdown?.high_risk_days?.length || 0}</span></div>
          <div className="card-body">
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Date</th>
                    <th>Primary Trigger</th>
                    <th>Sleep (h)</th>
                    <th>Stress</th>
                    <th>Fatigue</th>
                    <th>Missed Doses</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.high_risk_days || []).map((d, i) => (
                    <tr key={i} className={d.missed_doses > 0 ? 'table-warning' : ''}>
                      <td><code>{d.patient_id}</code></td>
                      <td>{d.log_date}</td>
                      <td><span className={`badge bg-${triggerColor(d.primary_trigger)}`}>{(d.primary_trigger || '—').replace(/_/g, ' ')}</span></td>
                      <td className={d.sleep_hours < 5 ? 'text-danger fw-bold' : ''}>{d.sleep_hours}h</td>
                      <td><span className={`badge bg-${d.stress_level >= 7 ? 'danger' : d.stress_level >= 4 ? 'warning' : 'success'}`}>{d.stress_level}/10</span></td>
                      <td>{d.fatigue_level}/10</td>
                      <td>{d.missed_doses > 0 ? <span className="badge bg-danger">{d.missed_doses}</span> : <span className="badge bg-success">0</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* PER PATIENT TAB */}
      {tab === 'patients' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Per-Patient Summary <span className="badge bg-secondary ms-2">{patients.length} patients</span></div>
          <div className="card-body">
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('total_logs')}>Logs <SortIcon col="total_logs" /></th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('seizures')}>Seizure Days <SortIcon col="seizures" /></th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('seizure_rate')}>Rate % <SortIcon col="seizure_rate" /></th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('avg_sleep')}>Avg Sleep <SortIcon col="avg_sleep" /></th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('avg_stress')}>Avg Stress <SortIcon col="avg_stress" /></th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('avg_mood')}>Avg Mood <SortIcon col="avg_mood" /></th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('adherence_pct')}>Adherence % <SortIcon col="adherence_pct" /></th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map((p, i) => (
                    <tr key={i}>
                      <td><code>{p.patient_id}</code></td>
                      <td>{p.total_logs}</td>
                      <td><span className={`badge bg-${p.seizures > 3 ? 'danger' : p.seizures > 1 ? 'warning' : 'success'}`}>{p.seizures}</span></td>
                      <td>
                        <span className={`badge bg-${p.seizure_rate > 40 ? 'danger' : p.seizure_rate > 20 ? 'warning' : 'success'}`}>
                          {p.seizure_rate.toFixed(1)}%
                        </span>
                      </td>
                      <td className={p.avg_sleep < 6 ? 'text-danger' : ''}>{p.avg_sleep.toFixed(1)}h</td>
                      <td><span className={`badge bg-${p.avg_stress >= 6 ? 'danger' : p.avg_stress >= 4 ? 'warning' : 'success'}`}>{p.avg_stress.toFixed(1)}</span></td>
                      <td>{p.avg_mood.toFixed(1)}</td>
                      <td>
                        <div className="d-flex align-items-center gap-1">
                          <div className="progress flex-grow-1" style={{ height: 14 }}>
                            <div className={`progress-bar bg-${p.adherence_pct >= 90 ? 'success' : p.adherence_pct >= 70 ? 'warning' : 'danger'}`}
                              style={{ width: `${p.adherence_pct}%` }} />
                          </div>
                          <span className="small">{p.adherence_pct.toFixed(0)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* RECENT LOGS TAB */}
      {tab === 'recent' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Recent Diary Entries <span className="badge bg-secondary ms-2">{breakdown?.recent_logs?.length || 0}</span></div>
          <div className="card-body">
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Date</th>
                    <th>Sleep (h)</th>
                    <th>Sleep Quality</th>
                    <th>Stress</th>
                    <th>Mood</th>
                    <th>Seizure?</th>
                    <th>Primary Trigger</th>
                    <th>Med Adherence</th>
                    <th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.recent_logs || []).map((l, i) => (
                    <tr key={i} className={l.seizure_occurred ? 'table-danger' : ''}>
                      <td><code>{l.patient_id}</code></td>
                      <td>{l.log_date}</td>
                      <td className={l.sleep_hours < 5 ? 'text-danger fw-bold' : ''}>{l.sleep_hours}h</td>
                      <td><span className={`badge bg-${l.sleep_quality === 'good' ? 'success' : l.sleep_quality === 'fair' ? 'info' : l.sleep_quality === 'poor' ? 'warning' : 'danger'}`}>{l.sleep_quality}</span></td>
                      <td><span className={`badge bg-${l.stress_level >= 7 ? 'danger' : l.stress_level >= 4 ? 'warning' : 'success'}`}>{l.stress_level}/10</span></td>
                      <td>{l.mood_score}/10</td>
                      <td>{l.seizure_occurred ? <span className="badge bg-danger">Yes</span> : <span className="badge bg-success">No</span>}</td>
                      <td>{l.primary_trigger ? <span className={`badge bg-${triggerColor(l.primary_trigger)}`}>{l.primary_trigger.replace(/_/g, ' ')}</span> : '—'}</td>
                      <td>{l.medication_adherence ? <span className="badge bg-success">Yes</span> : <span className="badge bg-danger">No</span>}</td>
                      <td className="text-muted small" style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{l.notes || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'defs' && (
        <div>
          <h5 className="mb-3">Seizure Trigger Glossary</h5>
          {defs?.trigger_descriptions && Object.entries(defs.trigger_descriptions).map(([k, v]) => (
            <div key={k} className="card shadow-sm mb-2">
              <div className="card-body py-2">
                <span className={`badge bg-${triggerColor(k)} me-2`}>{k.replace(/_/g, ' ')}</span>
                <span className="small">{v}</span>
              </div>
            </div>
          ))}
          {defs?.lifestyle_definitions && (
            <div className="mt-4">
              <h5>Lifestyle Metric Definitions</h5>
              {Object.entries(defs.lifestyle_definitions).map(([k, v]) => (
                <div key={k} className="card shadow-sm mb-2">
                  <div className="card-body py-2">
                    <strong className="me-2">{k.replace(/_/g, ' ')}</strong>
                    <span className="small text-muted">{v}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
