'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'patients',    label: 'Per Patient' },
  { id: 'definitions', label: 'Definitions' },
];

const ACTIVITY_COLOR = {
  medication_reminders: '#6366f1',
  meals_logged:         '#22c55e',
  exercise_logged:      '#f59e0b',
  sleep_logged:         '#06b6d4',
  mood_logged:          '#8b5cf6',
  seizure_logged:       '#ef4444',
};

const ACTIVITY_LABEL = {
  medication_reminders: 'Medication Reminders',
  meals_logged:         'Meals Logged',
  exercise_logged:      'Exercise Logged',
  sleep_logged:         'Sleep Logged',
  mood_logged:          'Mood Logged',
  seizure_logged:       'Seizures Logged',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 fw-bold mb-1 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ pct, color }) {
  const w = Math.min(100, Math.max(0, pct || 0));
  return (
    <div className="progress" style={{ height: 10, borderRadius: 6 }}>
      <div className="progress-bar" style={{ width: `${w}%`, backgroundColor: color || '#3b82f6', borderRadius: 6 }} />
    </div>
  );
}

function CompletionBadge({ pct }) {
  const p = pct || 0;
  const cls = p >= 75 ? 'success' : p >= 50 ? 'info' : p >= 25 ? 'warning' : 'danger';
  return <span className={`badge bg-${cls}`}>{p.toFixed(1)}%</span>;
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;

  const actRates   = ov.activity_rates     || {};
  const actTotals  = ov.activity_totals    || {};
  const compDist   = ov.completion_distribution || {};
  const trend      = ov.daily_trend        || [];

  const distEntries = Object.entries(compDist);
  const distTotal   = distEntries.reduce((s, [, v]) => s + v, 0);

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Total Plans"    value={ov.total_plans?.toLocaleString()}                  color="primary"   sub="30 days × 30 patients" />
        <KPI label="Patients"       value={ov.total_patients}                                 color="info"      sub="enrolled in daily tracking" />
        <KPI label="Avg Completion" value={`${(ov.avg_completion_pct || 0).toFixed(1)}%`}    color="success"   sub="daily plan adherence" />
        <KPI label="Date Range"     value={`${ov.date_range?.start} → ${ov.date_range?.end}`} color="secondary" sub="tracking window" />
      </div>

      {/* Activity Rates */}
      <div className="row mb-4">
        <div className="col-12 col-md-7 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Activity Logging Rates</div>
            <div className="card-body">
              {Object.keys(actRates).map(k => (
                <div key={k} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{ACTIVITY_LABEL[k] || k}</span>
                    <span className="fw-semibold">{(actRates[k] || 0).toFixed(1)}%
                      <span className="text-muted fw-normal ms-1">({(actTotals[k] || 0).toLocaleString()} events)</span>
                    </span>
                  </div>
                  <Bar pct={actRates[k]} color={ACTIVITY_COLOR[k]} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Completion Distribution */}
        <div className="col-12 col-md-5 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Completion Distribution</div>
            <div className="card-body">
              {distEntries.map(([bucket, count]) => (
                <div key={bucket} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{bucket}</span>
                    <span className="fw-semibold">{count} plans ({distTotal ? ((count / distTotal) * 100).toFixed(0) : 0}%)</span>
                  </div>
                  <Bar pct={distTotal ? (count / distTotal) * 100 : 0} color={
                    bucket.startsWith('76') ? '#22c55e' : bucket.startsWith('51') ? '#06b6d4' :
                    bucket.startsWith('26') ? '#f59e0b' : '#ef4444'
                  } />
                </div>
              ))}
              <div className="mt-3 p-2 rounded" style={{ background: '#f8f9fa', fontSize: '0.8rem' }}>
                <div className="d-flex justify-content-between">
                  <span>Min completion:</span><span className="fw-semibold">{ov.min_completion_pct}%</span>
                </div>
                <div className="d-flex justify-content-between">
                  <span>Max completion:</span><span className="fw-semibold">{ov.max_completion_pct}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Daily Trend */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Daily Completion Trend (30-day window)</div>
        <div className="card-body">
          {trend.length === 0 ? (
            <div className="text-muted">No trend data.</div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 120, minWidth: trend.length * 20 }}>
                {trend.map((d, i) => {
                  const h = Math.round((d.avg_completion / 100) * 100);
                  return (
                    <div key={i} title={`${d.date}: ${d.avg_completion?.toFixed(1)}%`}
                      style={{ flex: 1, minWidth: 16, height: `${h}%`, backgroundColor: '#6366f1', borderRadius: '3px 3px 0 0', opacity: 0.85 }}
                    />
                  );
                })}
              </div>
              <div className="d-flex justify-content-between text-muted mt-1" style={{ fontSize: '0.72rem' }}>
                <span>{trend[0]?.date}</span>
                <span>avg completion %</span>
                <span>{trend[trend.length - 1]?.date}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PatientsPanel({ bk }) {
  const [sort, setSort] = useState('avg_completion');
  const [dir,  setDir]  = useState(-1);

  if (!bk) return <div className="text-muted p-3">Loading…</div>;
  const patients = [...(bk.per_patient || [])].sort((a, b) => dir * ((a[sort] || 0) - (b[sort] || 0)));

  const toggle = col => {
    if (sort === col) setDir(d => -d);
    else { setSort(col); setDir(-1); }
  };
  const hdr = col => (
    <th className="small" style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => toggle(col)}>
      {col === sort ? (dir === -1 ? '▼ ' : '▲ ') : ''}{ACTIVITY_LABEL[col] || col}
    </th>
  );

  return (
    <div>
      <div className="mb-2 text-muted small">{patients.length} patients — click column header to sort</div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-light">
            <tr>
              <th className="small">Patient</th>
              {hdr('avg_completion')}
              {hdr('total_med_reminders')}
              {hdr('total_meals')}
              {hdr('total_exercise')}
              {hdr('total_sleep')}
              {hdr('total_mood')}
              {hdr('total_seizure')}
              <th className="small">Window</th>
            </tr>
          </thead>
          <tbody>
            {patients.map(p => (
              <tr key={p.patient_id}>
                <td><span className="badge bg-secondary">{p.patient_id}</span></td>
                <td><CompletionBadge pct={p.avg_completion} /></td>
                <td className="text-center">{p.total_med_reminders}</td>
                <td className="text-center">{p.total_meals}</td>
                <td className="text-center">{p.total_exercise}</td>
                <td className="text-center">{p.total_sleep}</td>
                <td className="text-center">{p.total_mood}</td>
                <td className="text-center">{p.total_seizure}</td>
                <td className="text-muted small">{p.first_plan} → {p.last_plan}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsPanel({ def }) {
  if (!def) return <div className="text-muted p-3">Loading…</div>;
  const fields   = def.fields   || {};
  const metrics  = def.metrics  || {};
  const sources  = def.sources  || {};

  return (
    <div>
      <div className="row">
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Field Definitions</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Field</th><th>Description</th></tr></thead>
                <tbody>
                  {Object.entries(fields).map(([k, v]) => (
                    <tr key={k}><td className="fw-semibold small">{k}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Metrics</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Metric</th><th>Definition</th></tr></thead>
                <tbody>
                  {Object.entries(metrics).map(([k, v]) => (
                    <tr key={k}><td className="fw-semibold small">{k}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          {Object.keys(sources).length > 0 && (
            <div className="card shadow-sm mt-3">
              <div className="card-header fw-semibold">Data Sources</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Source</th><th>Details</th></tr></thead>
                  <tbody>
                    {Object.entries(sources).map(([k, v]) => (
                      <tr key={k}><td className="fw-semibold small">{k}</td><td className="small">{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function DailyPlansPage() {
  const [tab, setTab] = useState('overview');
  const [ov,  setOv]  = useState(null);
  const [bk,  setBk]  = useState(null);
  const [def, setDef] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/daily-plans/overview`).then(r => r.json()),
      fetch(`${API}/api/daily-plans/breakdown`).then(r => r.json()),
      fetch(`${API}/api/daily-plans/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDef(d); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '1.6rem' }}>&#x1f4c5;</span>
        <div>
          <h4 className="mb-0 fw-bold">Daily Plans Dashboard</h4>
          <div className="text-muted small">
            {ov
              ? `${ov.total_plans?.toLocaleString()} plans · ${ov.total_patients} patients · ${ov.date_range?.start} → ${ov.date_range?.end} · avg ${(ov.avg_completion_pct || 0).toFixed(1)}% completion`
              : 'Loading…'}
          </div>
        </div>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel     ov={ov} />}
      {tab === 'patients'    && <PatientsPanel      bk={bk} />}
      {tab === 'definitions' && <DefinitionsPanel   def={def} />}
    </div>
  );
}
