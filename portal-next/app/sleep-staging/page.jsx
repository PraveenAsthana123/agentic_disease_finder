'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'patients',    label: 'Per Patient' },
  { id: 'definitions', label: 'Definitions' },
];

const STAGE_COLOR = {
  n1_pct:  '#a78bfa',
  n2_pct:  '#6366f1',
  n3_pct:  '#1d4ed8',
  rem_pct: '#22c55e',
};

const STAGE_LABEL = {
  n1_pct:  'N1 (Light)',
  n2_pct:  'N2 (Core)',
  n3_pct:  'N3 (Deep / SWS)',
  rem_pct: 'REM',
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

function Bar({ pct, color, max }) {
  const w = Math.min(100, Math.max(0, max ? (pct / max) * 100 : pct || 0));
  return (
    <div className="progress" style={{ height: 10, borderRadius: 6 }}>
      <div className="progress-bar" style={{ width: `${w}%`, backgroundColor: color || '#3b82f6', borderRadius: 6 }} />
    </div>
  );
}

function EfficiencyBadge({ eff }) {
  const e = eff || 0;
  const cls = e >= 90 ? 'success' : e >= 85 ? 'info' : e >= 70 ? 'warning' : 'danger';
  return <span className={`badge bg-${cls}`}>{e.toFixed(1)}%</span>;
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;

  const avgs       = ov.averages              || {};
  const studyTypes = ov.study_type_distribution || [];
  const effDist    = ov.efficiency_distribution || [];
  const sleepLink  = ov.seizure_sleep_link     || {};

  const maxType = Math.max(...studyTypes.map(t => t.count), 1);

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Total Studies"    value={ov.total_studies}                                  color="primary"   sub="polysomnography records" />
        <KPI label="Patients"         value={ov.total_patients}                                  color="info"      sub="with sleep study data" />
        <KPI label="Avg Sleep Eff."   value={`${(avgs.sleep_efficiency || 0).toFixed(1)}%`}    color="success"   sub="normal ≥85%" />
        <KPI label="Avg TST"          value={`${(avgs.tst_minutes || 0).toFixed(0)} min`}       color="secondary" sub="total sleep time" />
      </div>

      {/* Sleep Architecture + Study Types */}
      <div className="row mb-4">
        {/* Sleep Stage Distribution */}
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Average Sleep Architecture</div>
            <div className="card-body">
              {Object.keys(STAGE_LABEL).map(k => (
                <div key={k} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{STAGE_LABEL[k]}</span>
                    <span className="fw-semibold">{(avgs[k] || 0).toFixed(1)}%
                      <span className="text-muted fw-normal ms-1">of TST</span>
                    </span>
                  </div>
                  <Bar pct={avgs[k]} color={STAGE_COLOR[k]} />
                </div>
              ))}
              {/* Extra metrics */}
              <div className="mt-3 p-2 rounded" style={{ background: '#f8f9fa', fontSize: '0.8rem' }}>
                <div className="d-flex justify-content-between mb-1">
                  <span>SOL (onset latency):</span><span className="fw-semibold">{(avgs.sol_minutes || 0).toFixed(1)} min</span>
                </div>
                <div className="d-flex justify-content-between mb-1">
                  <span>REM Latency:</span><span className="fw-semibold">{(avgs.rem_latency_minutes || 0).toFixed(1)} min</span>
                </div>
                <div className="d-flex justify-content-between mb-1">
                  <span>WASO:</span><span className="fw-semibold">{(avgs.waso_minutes || 0).toFixed(1)} min</span>
                </div>
                <div className="d-flex justify-content-between mb-1">
                  <span>Arousal Index:</span><span className="fw-semibold">{(avgs.arousal_index || 0).toFixed(1)}/h</span>
                </div>
                <div className="d-flex justify-content-between">
                  <span>Spindle Density:</span><span className="fw-semibold">{(avgs.spindle_density || 0).toFixed(1)}/min</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Study Type Distribution */}
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Study Type Distribution</div>
            <div className="card-body">
              {studyTypes.map(st => (
                <div key={st.type} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{st.type}</span>
                    <span className="fw-semibold">{st.count} studies</span>
                  </div>
                  <Bar pct={st.count} color="#6366f1" max={maxType} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Sleep Efficiency Distribution + Seizure-Sleep Link */}
      <div className="row mb-4">
        {/* Efficiency Distribution */}
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Sleep Efficiency Distribution</div>
            <div className="card-body">
              {effDist.map(e => {
                const color = e.range === '≥95%' ? '#22c55e' : e.range.startsWith('90') ? '#06b6d4' :
                              e.range.startsWith('85') ? '#6366f1' : e.range.startsWith('80') ? '#f59e0b' :
                              e.range.startsWith('70') ? '#f97316' : '#ef4444';
                const total = effDist.reduce((s, x) => s + x.count, 0);
                return (
                  <div key={e.range} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{e.range}</span>
                      <span className="fw-semibold">{e.count} ({total ? ((e.count / total) * 100).toFixed(0) : 0}%)</span>
                    </div>
                    <Bar pct={total ? (e.count / total) * 100 : 0} color={color} />
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Seizure-Sleep Link */}
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Seizure–Sleep Interaction</div>
            <div className="card-body">
              <div className="row text-center mb-3">
                <div className="col-6 mb-3">
                  <div className="h4 fw-bold text-danger">{sleepLink.total_seizures_during_sleep ?? '—'}</div>
                  <div className="text-muted small">Seizures during sleep</div>
                </div>
                <div className="col-6 mb-3">
                  <div className="h4 fw-bold text-warning">{sleepLink.studies_with_seizures ?? '—'}</div>
                  <div className="text-muted small">Studies with seizures</div>
                </div>
                <div className="col-6">
                  <div className="h4 fw-bold text-info">{sleepLink.ied_nrem_activation_count ?? '—'}</div>
                  <div className="text-muted small">IED–NREM activations</div>
                </div>
                <div className="col-6">
                  <div className="h4 fw-bold text-secondary">{sleepLink.osa_prevalence ?? '—'}%</div>
                  <div className="text-muted small">OSA prevalence</div>
                </div>
              </div>
              <div className="p-2 rounded" style={{ background: '#fef2f2', fontSize: '0.78rem', borderLeft: '3px solid #ef4444' }}>
                <strong>Clinical note:</strong> IEDs increase 2–5× during NREM.
                REM has protective anti-seizure effect. OSA treatment can reduce seizure frequency up to 50%.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientsPanel({ bk }) {
  const [sort, setSort] = useState('avg_sleep_efficiency');
  const [dir,  setDir]  = useState(-1);

  if (!bk) return <div className="text-muted p-3">Loading…</div>;
  const patients = [...(bk.patients || [])].sort((a, b) => dir * ((a[sort] || 0) - (b[sort] || 0)));

  const toggle = col => {
    if (sort === col) setDir(d => -d);
    else { setSort(col); setDir(-1); }
  };
  const Hdr = ({ col, label }) => (
    <th className="small" style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => toggle(col)}>
      {sort === col ? (dir === -1 ? '▼ ' : '▲ ') : ''}{label}
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
              <Hdr col="studies"              label="Studies" />
              <Hdr col="avg_sleep_efficiency" label="Avg Efficiency" />
              <Hdr col="avg_tst_minutes"      label="Avg TST (min)" />
              <Hdr col="avg_rem_pct"          label="REM %" />
              <Hdr col="avg_n3_pct"           label="N3 %" />
              <Hdr col="avg_arousal_index"    label="Arousal Idx" />
              <Hdr col="total_seizures_sleep" label="Seizures (sleep)" />
              <th className="small">Flags</th>
              <th className="small">Latest</th>
            </tr>
          </thead>
          <tbody>
            {patients.map(p => (
              <tr key={p.patient_id}>
                <td><span className="badge bg-secondary">{p.patient_id}</span></td>
                <td className="text-center">{p.studies}</td>
                <td><EfficiencyBadge eff={p.avg_sleep_efficiency} /></td>
                <td className="text-center">{(p.avg_tst_minutes || 0).toFixed(0)}</td>
                <td className="text-center">{(p.avg_rem_pct || 0).toFixed(1)}%</td>
                <td className="text-center">{(p.avg_n3_pct || 0).toFixed(1)}%</td>
                <td className="text-center">{(p.avg_arousal_index || 0).toFixed(1)}</td>
                <td className="text-center">
                  {p.total_seizures_sleep > 0
                    ? <span className="badge bg-danger">{p.total_seizures_sleep}</span>
                    : <span className="text-muted">0</span>}
                </td>
                <td>
                  {(p.flags || []).length > 0
                    ? (p.flags || []).map(f => (
                        <span key={f} className="badge bg-warning text-dark me-1" style={{ fontSize: '0.7rem' }}>{f.replace(/_/g, ' ')}</span>
                      ))
                    : <span className="text-muted small">—</span>}
                </td>
                <td className="text-muted small">{p.latest_study}</td>
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
  const stages   = def.sleep_stages              || [];
  const metrics  = def.key_metrics               || [];
  const epilepsy = def.epilepsy_sleep_interaction || [];

  return (
    <div>
      {/* Sleep Stages */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Sleep Stage Definitions (AASM 2007)</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr>
                  <th>Stage</th>
                  <th>EEG Pattern</th>
                  <th>Key Features</th>
                  <th>Normal Duration</th>
                </tr>
              </thead>
              <tbody>
                {stages.map(s => (
                  <tr key={s.stage}>
                    <td className="fw-semibold small">{s.stage}</td>
                    <td className="small">{s.eeg_pattern}</td>
                    <td className="small">{s.key_features}</td>
                    <td className="small text-muted">{s.duration_normal}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Key Sleep Metrics</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr><th>Metric</th><th>Definition</th><th>Normal (adult)</th></tr>
              </thead>
              <tbody>
                {metrics.map(m => (
                  <tr key={m.metric}>
                    <td className="fw-semibold small">{m.metric}</td>
                    <td className="small">{m.definition}</td>
                    <td className="small text-muted">{m.normal_adult}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Epilepsy-Sleep Interaction */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Epilepsy–Sleep Interaction</div>
        <div className="card-body">
          <div className="row">
            {epilepsy.map(e => (
              <div key={e.topic} className="col-12 col-md-6 mb-3">
                <div className="p-3 rounded border" style={{ background: '#f8faff' }}>
                  <div className="fw-semibold small mb-1">{e.topic}</div>
                  <div className="text-muted" style={{ fontSize: '0.82rem' }}>{e.description}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function SleepStagingPage() {
  const [tab, setTab] = useState('overview');
  const [ov,  setOv]  = useState(null);
  const [bk,  setBk]  = useState(null);
  const [def, setDef] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/sleep-staging/overview`).then(r => r.json()),
      fetch(`${API}/api/sleep-staging/breakdown`).then(r => r.json()),
      fetch(`${API}/api/sleep-staging/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDef(d); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '1.6rem' }}>&#x1f4a4;</span>
        <div>
          <h4 className="mb-0 fw-bold">Sleep Staging Dashboard</h4>
          <div className="text-muted small">
            {ov
              ? `${ov.total_studies} studies · ${ov.total_patients} patients · avg efficiency ${(ov.averages?.sleep_efficiency || 0).toFixed(1)}% · avg TST ${(ov.averages?.tst_minutes || 0).toFixed(0)} min`
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

      {tab === 'overview'    && <OverviewPanel    ov={ov} />}
      {tab === 'patients'    && <PatientsPanel     bk={bk} />}
      {tab === 'definitions' && <DefinitionsPanel  def={def} />}
    </div>
  );
}
