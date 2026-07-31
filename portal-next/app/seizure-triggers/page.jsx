'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Log Details' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function DistBar({ items, colorFn }) {
  const total = items.reduce((a, b) => a + (b.count || b.rate || 0), 0);
  return (
    <table className="table table-sm mb-0">
      <tbody>
        {items.map((item, i) => {
          const label = item.trigger || item.type || item.quality || '';
          const val = item.count != null ? item.count : item.rate;
          const pct = item.rate != null ? item.rate : (total > 0 ? ((val / total) * 100).toFixed(1) : 0);
          const color = colorFn ? colorFn(label) : 'primary';
          return (
            <tr key={i}>
              <td className="text-nowrap small fw-semibold" style={{ width: '40%' }}>{label.replace(/_/g, ' ')}</td>
              <td style={{ width: '45%' }}>
                <div className="progress" style={{ height: 10 }}>
                  <div className={`progress-bar bg-${color}`} style={{ width: `${Math.min(pct, 100)}%` }} />
                </div>
              </td>
              <td className="small text-end">
                {item.count != null && <>{item.count} </>}
                {item.rate != null && <span className="text-muted">({item.rate}%)</span>}
                {item.count != null && item.rate == null && <span className="text-muted">({pct}%)</span>}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const trigColor = t => {
    const m = { fatigue: 'danger', missed_medication: 'danger', stress: 'warning', photosensitivity: 'warning',
      sleep_deprivation: 'info', illness: 'secondary', hormonal_changes: 'primary', dehydration: 'success', alcohol: 'dark' };
    return m[t] || 'primary';
  };
  const sleepColor = q => ({ very_poor: 'danger', poor: 'warning', fair: 'info', good: 'success' }[q] || 'secondary');
  const typeColor = () => 'info';

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Logs" value={data.total_logs} color="primary" sub="trigger log entries" />
        <KPI label="Patients" value={data.total_patients} color="info" sub="unique patients" />
        <KPI label="Seizure Events" value={data.seizure_count} color="danger" sub={`${data.seizure_rate}% seizure rate`} />
        <KPI label="Avg Sleep" value={data.avg_sleep_hours != null ? `${data.avg_sleep_hours}h` : '—'} color="warning" sub="hours per night" />
      </div>
      <div className="row mb-4">
        <KPI label="Med Adherence" value={data.medication_adherence_rate != null ? `${data.medication_adherence_rate}%` : '—'} color={data.medication_adherence_rate >= 80 ? 'success' : 'danger'} sub="days with full adherence" />
      </div>

      <div className="row mb-3">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Trigger Distribution</div>
            <div className="card-body p-2">
              <DistBar items={data.trigger_distribution || []} colorFn={trigColor} />
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Trigger → Seizure Rate</div>
            <div className="card-body p-2">
              <DistBar items={data.trigger_seizure_rate || []} colorFn={trigColor} />
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-3">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Sleep Quality Distribution</div>
            <div className="card-body p-2">
              <DistBar items={data.sleep_quality_distribution || []} colorFn={sleepColor} />
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Sleep Quality → Seizure Rate</div>
            <div className="card-body p-2">
              <DistBar items={data.sleep_vs_seizure || []} colorFn={sleepColor} />
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-3">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Seizure Type Distribution</div>
            <div className="card-body p-2">
              <DistBar items={data.seizure_type_distribution || []} colorFn={typeColor} />
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Risk Factor Comparison (Seizure vs No Seizure)</div>
            <div className="card-body p-2">
              <table className="table table-sm table-striped mb-0">
                <thead>
                  <tr>
                    <th>Factor</th>
                    <th className="text-end">With Seizure</th>
                    <th className="text-end">Without Seizure</th>
                    <th className="text-end">Diff</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.risk_comparison || []).map((r, i) => {
                    const diff = (r.with_seizure - r.without_seizure).toFixed(1);
                    const worse = parseFloat(diff) > 0 && ['Stress Level', 'Fatigue Level', 'Caffeine (mg)'].includes(r.factor)
                      || parseFloat(diff) < 0 && ['Sleep Hours', 'Mood Score'].includes(r.factor);
                    return (
                      <tr key={i}>
                        <td className="fw-semibold small">{r.factor}</td>
                        <td className="text-end">{r.with_seizure}</td>
                        <td className="text-end">{r.without_seizure}</td>
                        <td className={`text-end fw-bold ${worse ? 'text-danger' : 'text-success'}`}>
                          {parseFloat(diff) > 0 ? '+' : ''}{diff}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {(data.monthly_trend || []).length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Monthly Trend</div>
          <div className="card-body p-2">
            <table className="table table-sm table-striped mb-0">
              <thead>
                <tr>
                  <th>Month</th>
                  <th className="text-end">Total Logs</th>
                  <th className="text-end">Seizures</th>
                  <th className="text-end">Rate</th>
                </tr>
              </thead>
              <tbody>
                {data.monthly_trend.map(m => {
                  const rate = m.total_logs > 0 ? ((m.seizures / m.total_logs) * 100).toFixed(1) : 0;
                  return (
                    <tr key={m.month}>
                      <td>{m.month}</td>
                      <td className="text-end">{m.total_logs}</td>
                      <td className="text-end"><span className={m.seizures > 0 ? 'text-danger fw-bold' : ''}>{m.seizures}</span></td>
                      <td className="text-end">{rate}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const logs = data.logs || [];
  const [filter, setFilter] = useState('all');
  const filtered = filter === 'seizure' ? logs.filter(l => l.seizure_occurred)
    : filter === 'no_seizure' ? logs.filter(l => !l.seizure_occurred) : logs;

  return (
    <div>
      <div className="d-flex gap-2 mb-3">
        <button className={`btn btn-sm ${filter === 'all' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setFilter('all')}>All ({logs.length})</button>
        <button className={`btn btn-sm ${filter === 'seizure' ? 'btn-danger' : 'btn-outline-danger'}`} onClick={() => setFilter('seizure')}>Seizure Days ({logs.filter(l => l.seizure_occurred).length})</button>
        <button className={`btn btn-sm ${filter === 'no_seizure' ? 'btn-success' : 'btn-outline-success'}`} onClick={() => setFilter('no_seizure')}>No Seizure ({logs.filter(l => !l.seizure_occurred).length})</button>
      </div>

      <div className="card">
        <div className="card-header fw-semibold">Trigger Logs ({filtered.length})</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0" style={{ fontSize: '0.8rem' }}>
              <thead>
                <tr>
                  <th>Patient</th>
                  <th>Date</th>
                  <th>Trigger</th>
                  <th className="text-center">Seizure</th>
                  <th>Type</th>
                  <th className="text-end">Duration</th>
                  <th className="text-end">Sleep</th>
                  <th>Quality</th>
                  <th className="text-end">Stress</th>
                  <th className="text-end">Fatigue</th>
                  <th className="text-center">Med Adh</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 100).map((l, i) => (
                  <tr key={i} className={l.seizure_occurred ? 'table-danger' : ''}>
                    <td className="fw-semibold">{l.patient_id}</td>
                    <td className="text-nowrap">{l.log_date}</td>
                    <td>{(l.primary_trigger || '—').replace(/_/g, ' ')}</td>
                    <td className="text-center">
                      {l.seizure_occurred ? <span className="badge bg-danger">Yes</span> : <span className="badge bg-success">No</span>}
                    </td>
                    <td className="small">{l.seizure_type || '—'}</td>
                    <td className="text-end">{l.seizure_duration_sec != null ? `${l.seizure_duration_sec}s` : '—'}</td>
                    <td className="text-end">{l.sleep_hours}h</td>
                    <td>
                      <span className={`badge bg-${l.sleep_quality === 'good' ? 'success' : l.sleep_quality === 'fair' ? 'info' : l.sleep_quality === 'poor' ? 'warning' : 'danger'}`}>
                        {(l.sleep_quality || '—').replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="text-end">{l.stress_level}/10</td>
                    <td className="text-end">{l.fatigue_level}/10</td>
                    <td className="text-center">
                      {l.medication_adherence ? <span className="badge bg-success">Yes</span> : <span className="badge bg-danger">No</span>}
                    </td>
                    <td className="small text-muted" style={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{l.notes || ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        {filtered.length > 100 && <div className="card-footer text-muted small">Showing first 100 of {filtered.length} logs</div>}
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div className="card">
      <div className="card-header fw-semibold">{data.title || 'Seizure Trigger Logs'} — Definitions</div>
      <div className="card-body p-0">
        <table className="table table-sm table-striped mb-0">
          <thead><tr><th>Concept</th><th>Description</th></tr></thead>
          <tbody>
            {(data.concepts || data.metrics || []).map((m, i) => (
              <tr key={i}>
                <td className="fw-semibold text-nowrap">{m.name}</td>
                <td className="small">{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function SeizureTriggerLogsDashboard() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/seizure-triggers/overview`).then(r => r.json()).then(setOverview).catch(() => setOverview({ error: 'Failed to load overview' }));
    fetch(`${API}/api/seizure-triggers/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setBreakdown({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/seizure-triggers/definitions`).then(r => r.json()).then(setDefinitions).catch(() => setDefinitions({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-3">⚡ Seizure Trigger Logs Dashboard</h4>
      <p className="text-muted small mb-3">
        Daily lifestyle and trigger logs for 40 patients — sleep, stress, caffeine, medication adherence, and seizure outcomes.
        Identifies modifiable risk factors correlated with seizure occurrence.
      </p>
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>
      {tab === 'overview' && <OverviewPanel data={overview} />}
      {tab === 'breakdown' && <BreakdownPanel data={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
