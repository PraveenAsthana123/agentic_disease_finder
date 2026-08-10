'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
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

function HBar({ items, maxWidth = 100 }) {
  if (!items || !items.length) return null;
  const mx = Math.max(...items.map(i => i.value));
  return (
    <div>
      {items.map((it, i) => (
        <div key={i} className="d-flex align-items-center mb-2">
          <div className="text-end me-2 small text-capitalize" style={{ width: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {it.label}
          </div>
          <div className="flex-grow-1">
            <div className="progress" style={{ height: 20 }}>
              <div
                className={`progress-bar bg-${it.color || 'primary'}`}
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

function CheckBadge({ val }) {
  return val
    ? <span className="badge bg-success">&#x2713; Yes</span>
    : <span className="badge bg-secondary">&#x2715; No</span>;
}

function CoopBadge({ level }) {
  const colors = { excellent: 'success', good: 'primary', fair: 'warning', poor: 'danger' };
  return <span className={`badge bg-${colors[level] || 'secondary'} text-capitalize`}>{level}</span>;
}

function StateBadge({ state }) {
  const colors = { awake: 'success', drowsy: 'warning', asleep: 'info' };
  return <span className={`badge bg-${colors[state] || 'secondary'} text-capitalize`}>{state}</span>;
}

export default function RecordingConditionsDashboard() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [sortCol, setSortCol] = useState('activations_completed');
  const [sortDir, setSortDir] = useState('desc');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/recording-conditions/overview`).then(r => r.json()),
      fetch(`${API}/api/recording-conditions/breakdown`).then(r => r.json()),
      fetch(`${API}/api/recording-conditions/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Per Patient' },
    { id: 'defs', label: 'Definitions' },
  ];

  const patients = [...(breakdown?.patients || [])];
  patients.sort((a, b) => {
    const av = a[sortCol] ?? 0, bv = b[sortCol] ?? 0;
    if (typeof av === 'boolean') return sortDir === 'asc' ? (av ? 1 : 0) - (bv ? 1 : 0) : (bv ? 1 : 0) - (av ? 1 : 0);
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

  const activationItems = [
    { label: 'Eyes Open/Closed', value: overview.activation_rates?.eyes_open_pct ?? 0, pct: overview.activation_rates?.eyes_open_pct, color: 'primary' },
    { label: 'Hyperventilation', value: overview.activation_rates?.hyperventilation_pct ?? 0, pct: overview.activation_rates?.hyperventilation_pct, color: 'info' },
    { label: 'Photic Stimulation', value: overview.activation_rates?.photic_stimulation_pct ?? 0, pct: overview.activation_rates?.photic_stimulation_pct, color: 'warning' },
    { label: 'Sleep Recording', value: overview.activation_rates?.sleep_recorded_pct ?? 0, pct: overview.activation_rates?.sleep_recorded_pct, color: 'success' },
  ];

  const stateItems = Object.entries(overview.patient_state_distribution || {}).map(([k, v]) => ({
    label: k,
    value: v,
    color: k === 'awake' ? 'success' : k === 'drowsy' ? 'warning' : 'info',
  }));

  const coopItems = Object.entries(overview.cooperation_distribution || {}).map(([k, v]) => ({
    label: k,
    value: v,
    color: k === 'excellent' ? 'success' : k === 'good' ? 'primary' : k === 'fair' ? 'warning' : 'danger',
  }));

  const completeCount = patients.filter(p => p.protocol_complete).length;

  return (
    <div>
      <h3>&#x1f4f9; EEG Recording Conditions</h3>
      <p className="text-muted">
        Activation procedures, patient state, and cooperation ratings for all EEG studies — real <code>recording_conditions</code> table, {overview.total_recordings} recordings, {patients.length} patients.
      </p>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Total Recordings" value={overview.total_recordings} color="primary" />
        <KPI label="Protocol Complete" value={`${overview.protocol_completeness?.toFixed(1)}%`} color={overview.protocol_completeness >= 60 ? 'success' : 'warning'} sub={`${completeCount} of ${patients.length} patients`} />
        <KPI label="Excellent/Good Coop" value={`${overview.quality_summary?.excellent_good_pct}%`} color="success" sub="cooperation quality" />
        <KPI label="Sleep Recorded" value={`${overview.activation_rates?.sleep_recorded_pct}%`} color="info" sub="of recordings" />
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
                <div className="card-header fw-semibold">Activation Procedure Rates</div>
                <div className="card-body">
                  <HBar items={activationItems} />
                  <div className="text-muted small mt-2">% of recordings where each activation procedure was performed</div>
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-4">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Patient State Distribution</div>
                <div className="card-body">
                  <div className="row text-center mb-3">
                    {stateItems.map((s, i) => (
                      <div key={i} className="col">
                        <div className={`badge bg-${s.color} fs-5 mb-1 d-block`}>{s.value}</div>
                        <div className="small text-capitalize text-muted">{s.label}</div>
                      </div>
                    ))}
                  </div>
                  <HBar items={stateItems} />
                </div>
              </div>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6 mb-4">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Cooperation Quality</div>
                <div className="card-body">
                  <div className="row text-center mb-3">
                    {coopItems.map((c, i) => (
                      <div key={i} className="col">
                        <div className={`badge bg-${c.color} fs-5 mb-1 d-block`}>{c.value}</div>
                        <div className="small text-capitalize text-muted">{c.label}</div>
                      </div>
                    ))}
                  </div>
                  <HBar items={coopItems} />
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-4">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Protocol Completeness</div>
                <div className="card-body">
                  <div className="text-center mb-3">
                    <div className="display-6 fw-bold text-primary">{overview.protocol_completeness?.toFixed(1)}%</div>
                    <div className="text-muted small">of recordings completed all 4 activation procedures</div>
                  </div>
                  <div className="progress mb-2" style={{ height: 24 }}>
                    <div
                      className={`progress-bar bg-${overview.protocol_completeness >= 60 ? 'success' : overview.protocol_completeness >= 40 ? 'warning' : 'danger'}`}
                      style={{ width: `${overview.protocol_completeness}%` }}
                    >
                      {overview.protocol_completeness?.toFixed(1)}%
                    </div>
                  </div>
                  <div className="text-muted small">
                    ACNS/ILAE minimum standard: complete all 4 procedures (eyes open/closed, HV, PS, sleep)
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* PER PATIENT TAB */}
      {tab === 'patients' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">
            Per-Patient Recording Conditions
            <span className="badge bg-secondary ms-2">{patients.length} patients</span>
          </div>
          <div className="card-body">
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Eyes Open</th>
                    <th>Hypervent.</th>
                    <th>Photic Stim</th>
                    <th>Sleep Rec.</th>
                    <th>State</th>
                    <th>Cooperation</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('activations_completed')}>
                      Activations <SortIcon col="activations_completed" />
                    </th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('protocol_complete')}>
                      Protocol <SortIcon col="protocol_complete" />
                    </th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map((p, i) => (
                    <tr key={i} className={p.protocol_complete ? 'table-success' : ''}>
                      <td><code>{p.patient_id}</code></td>
                      <td><CheckBadge val={p.eyes_open} /></td>
                      <td><CheckBadge val={p.hyperventilation} /></td>
                      <td><CheckBadge val={p.photic_stimulation} /></td>
                      <td><CheckBadge val={p.sleep_recorded} /></td>
                      <td><StateBadge state={p.patient_state} /></td>
                      <td><CoopBadge level={p.cooperation} /></td>
                      <td>
                        <div className="d-flex align-items-center gap-1">
                          <div className="progress flex-grow-1" style={{ height: 14, minWidth: 50 }}>
                            <div
                              className={`progress-bar bg-${p.activations_completed === 4 ? 'success' : p.activations_completed >= 2 ? 'warning' : 'danger'}`}
                              style={{ width: `${(p.activations_completed / 4) * 100}%` }}
                            />
                          </div>
                          <span className="small">{p.activations_completed}/4</span>
                        </div>
                      </td>
                      <td>
                        {p.protocol_complete
                          ? <span className="badge bg-success">&#x2713; Complete</span>
                          : <span className="badge bg-warning text-dark">Partial</span>}
                      </td>
                      <td className="text-muted small">{p.created_at?.split(' ')[0] || '—'}</td>
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
          {defs?.description && (
            <div className="alert alert-info mb-4">
              <strong>{defs.title}</strong><br />
              <span className="small">{defs.description}</span>
            </div>
          )}
          <div className="row">
            {(defs?.terms || []).map((t, i) => (
              <div key={i} className="col-md-6 mb-3">
                <div className="card shadow-sm h-100">
                  <div className="card-body">
                    <h6 className="card-title text-primary">{t.term}</h6>
                    <p className="card-text small text-muted mb-0">{t.definition}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
