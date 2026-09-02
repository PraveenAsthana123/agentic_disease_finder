'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STAGE_META = {
  W:   { label: 'Wake',      color: '#ef4444', icon: '☀️' },
  N1:  { label: 'N1 Light',  color: '#f59e0b', icon: '🌙' },
  N2:  { label: 'N2 Core',   color: '#3b82f6', icon: '💤' },
  N3:  { label: 'Deep (N3)', color: '#6366f1', icon: '🌊' },
  REM: { label: 'REM',       color: '#22c55e', icon: '👁️' },
};

function KPI({ label, value, unit, color, icon, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className={`card border-${color || 'primary'} border-2 h-100`}>
        <div className="card-body text-center py-2 px-2">
          <div style={{ fontSize: '1.3rem' }}>{icon}</div>
          <div className={`fw-bold fs-5 text-${color || 'primary'}`}>
            {value ?? '—'}{unit && <span className="fs-6 fw-normal ms-1">{unit}</span>}
          </div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function StageBar({ stageKey, data, totalMin }) {
  const meta = STAGE_META[stageKey] || { label: stageKey, color: '#6b7280', icon: '?' };
  const pct = totalMin > 0 ? (data.minutes / totalMin) * 100 : 0;
  return (
    <div className="mb-3">
      <div className="d-flex justify-content-between align-items-center mb-1">
        <span className="fw-semibold small">
          {meta.icon} {meta.label}
        </span>
        <span className="text-muted small">
          {data.minutes} min &nbsp;·&nbsp; {data.pct_of_sleep?.toFixed(1)}%
        </span>
      </div>
      <div className="progress" style={{ height: 18, borderRadius: 4 }}>
        <div
          className="progress-bar"
          role="progressbar"
          style={{ width: `${pct}%`, backgroundColor: meta.color }}
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}

function RecordingRow({ rec, idx, onSelect, selected }) {
  const name = rec.hypnogram ? rec.hypnogram.split('/').pop() : '?';
  return (
    <tr
      className={selected ? 'table-primary' : ''}
      style={{ cursor: 'pointer' }}
      onClick={() => onSelect(rec.hypnogram)}
    >
      <td className="text-muted small">{idx + 1}</td>
      <td className="small font-monospace">{name}</td>
      <td>
        <span className={`badge bg-${rec.dataset === 'cassette' ? 'info' : 'secondary'}`}>
          {rec.dataset}
        </span>
      </td>
      <td className="text-center">
        {selected ? <span className="badge bg-success">selected</span> : null}
      </td>
    </tr>
  );
}

export default function EegVizSleepPage() {
  const [tab, setTab] = useState('overview');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);
  const [selectedHyp, setSelectedHyp] = useState(null);
  const [archLoading, setArchLoading] = useState(false);

  const fetchSleep = (hypnogram) => {
    const url = hypnogram
      ? `${API}/api/eeg-viz/sleep?hypnogram=${encodeURIComponent(hypnogram)}`
      : `${API}/api/eeg-viz/sleep`;
    setArchLoading(true);
    fetch(url)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(d => { setData(d); setLoading(false); setArchLoading(false); })
      .catch(e => { setErr(String(e)); setLoading(false); setArchLoading(false); });
  };

  useEffect(() => { fetchSleep(null); }, []);

  const handleSelect = (hyp) => {
    setSelectedHyp(hyp);
    fetchSleep(hyp);
  };

  if (loading) return (
    <div className="d-flex align-items-center justify-content-center" style={{ minHeight: '60vh' }}>
      <div className="text-center">
        <div className="spinner-border text-info mb-3" />
        <div className="text-muted">Loading sleep architecture…</div>
      </div>
    </div>
  );

  if (err) return (
    <div className="container py-5">
      <div className="alert alert-danger">Backend error: {err}</div>
    </div>
  );

  const stages = data?.stages || {};
  const totalMin = data?.total_sleep_time_min || 0;
  const recordings = data?.recordings || [];

  return (
    <div className="d-flex flex-column" style={{ minHeight: '100vh', backgroundColor: '#0f172a' }}>
      {/* Header */}
      <div className="bg-dark border-bottom border-secondary px-4 py-3 d-flex align-items-center gap-3">
        <Link href="/eeg-viz" className="btn btn-sm btn-outline-secondary">← EEG Viz</Link>
        <div>
          <h5 className="mb-0 text-white fw-bold">💤 Sleep State Dashboard</h5>
          <div className="text-muted small">
            Sleep-EDFx · {data?.n_total ?? '?'} recordings · Expert-scored hypnograms · MNE
          </div>
        </div>
        <div className="ms-auto d-flex gap-2 align-items-center">
          {data?.quality === 'PASS'
            ? <span className="badge bg-success">Quality: PASS</span>
            : <span className="badge bg-warning text-dark">Quality: REVIEW</span>}
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-dark px-4 border-bottom border-secondary">
        <ul className="nav nav-tabs border-0">
          {['overview', 'architecture', 'recordings', 'quality'].map(t => (
            <li key={t} className="nav-item">
              <button
                className={`nav-link ${tab === t ? 'active text-white fw-semibold border-info border-bottom border-2' : 'text-secondary'}`}
                style={{ background: 'none', border: 'none', borderBottom: tab === t ? '2px solid #06b6d4' : 'none' }}
                onClick={() => setTab(t)}
              >
                {t === 'overview' ? '📊 Overview' :
                 t === 'architecture' ? '🌙 Architecture' :
                 t === 'recordings' ? '📁 Recordings' : '✅ Quality'}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* Body */}
      <div className="flex-grow-1 p-4" style={{ backgroundColor: '#0f172a' }}>

        {/* ── OVERVIEW ── */}
        {tab === 'overview' && (
          <>
            <div className="row mb-3">
              <KPI label="Total Sleep Time" value={data?.total_sleep_time_min?.toFixed(0)} unit="min"
                   color="info" icon="💤" sub={`${(totalMin / 60).toFixed(1)} hrs`} />
              <KPI label="Time in Bed" value={data?.time_in_bed_min?.toFixed(0)} unit="min"
                   color="primary" icon="🛏️" />
              <KPI label="Sleep Efficiency" value={data?.sleep_efficiency_pct?.toFixed(1)} unit="%"
                   color={data?.sleep_efficiency_pct >= 85 ? 'success' : 'warning'} icon="⚡" />
              <KPI label="Stage Transitions" value={data?.stage_transitions}
                   color="secondary" icon="🔄" sub="wake/stage switches" />
            </div>
            <div className="row mb-3">
              <KPI label="REM Sleep" value={data?.rem_pct?.toFixed(1)} unit="%"
                   color="success" icon="👁️" sub="norm 20–25%" />
              <KPI label="Deep Sleep (N3)" value={data?.deep_sleep_pct?.toFixed(1)} unit="%"
                   color="info" icon="🌊" sub="norm 15–25%" />
              <KPI label="Recordings" value={data?.n_total}
                   color="secondary" icon="📁" sub="Sleep-EDFx cassette+telemetry" />
              <KPI label="Source" value="Sleep-EDFx"
                   color="dark" icon="📊" sub="PhysioNet real hypnograms" />
            </div>

            <div className="card bg-dark border-secondary mb-3">
              <div className="card-header text-white fw-semibold">Sleep Stage Distribution</div>
              <div className="card-body">
                {Object.entries(stages).map(([k, v]) => (
                  <StageBar key={k} stageKey={k} data={v} totalMin={totalMin} />
                ))}
              </div>
            </div>

            {data?.flags?.length > 0 && (
              <div className="alert alert-info small mb-0">
                <strong>Clinical Flags:</strong> {data.flags.join(' · ')}
              </div>
            )}
          </>
        )}

        {/* ── ARCHITECTURE ── */}
        {tab === 'architecture' && (
          <>
            {archLoading && (
              <div className="text-center py-4">
                <div className="spinner-border spinner-border-sm text-info me-2" />
                <span className="text-muted">Loading architecture…</span>
              </div>
            )}
            {!archLoading && (
              <>
                <div className="row mb-4">
                  {Object.entries(stages).map(([k, v]) => {
                    const meta = STAGE_META[k] || {};
                    return (
                      <div key={k} className="col-6 col-md-2 mb-3">
                        <div className="card border-0 shadow-sm h-100 text-center"
                             style={{ backgroundColor: '#1e293b' }}>
                          <div className="card-body py-3">
                            <div style={{ fontSize: '1.5rem' }}>{meta.icon || '?'}</div>
                            <div className="fw-bold text-white fs-5">{v.pct_of_sleep?.toFixed(1)}%</div>
                            <div className="text-muted small">{meta.label || k}</div>
                            <div className="text-muted" style={{ fontSize: '0.65rem' }}>{v.minutes} min</div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="card bg-dark border-secondary mb-3">
                  <div className="card-header text-white fw-semibold">Stage-by-Stage Breakdown</div>
                  <div className="card-body">
                    <div className="table-responsive">
                      <table className="table table-dark table-sm table-hover mb-0">
                        <thead>
                          <tr>
                            <th>Stage</th>
                            <th>Duration (min)</th>
                            <th>% of Sleep</th>
                            <th>Normative Range</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[
                            { k: 'W',   norm: '5–10%',   lo: 5,  hi: 15 },
                            { k: 'N1',  norm: '5–10%',   lo: 5,  hi: 15 },
                            { k: 'N2',  norm: '45–55%',  lo: 40, hi: 60 },
                            { k: 'N3',  norm: '15–25%',  lo: 15, hi: 30 },
                            { k: 'REM', norm: '20–25%',  lo: 18, hi: 30 },
                          ].map(({ k, norm, lo, hi }) => {
                            const s = stages[k];
                            if (!s) return null;
                            const pct = s.pct_of_sleep;
                            const ok = pct >= lo && pct <= hi;
                            return (
                              <tr key={k}>
                                <td>
                                  <span style={{ color: STAGE_META[k]?.color }}>
                                    {STAGE_META[k]?.icon} {STAGE_META[k]?.label || k}
                                  </span>
                                </td>
                                <td>{s.minutes}</td>
                                <td>{pct?.toFixed(1)}%</td>
                                <td className="text-muted small">{norm}</td>
                                <td>
                                  <span className={`badge bg-${ok ? 'success' : 'warning'}`}>
                                    {ok ? 'Normal' : 'Atypical'}
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

                <div className="row">
                  <div className="col-md-4 mb-3">
                    <div className="card bg-dark border-secondary h-100">
                      <div className="card-body text-center">
                        <div className="text-muted small mb-1">Sleep Efficiency</div>
                        <div className="display-6 fw-bold text-info">
                          {data?.sleep_efficiency_pct?.toFixed(1)}%
                        </div>
                        <div className="text-muted small">target ≥ 85%</div>
                      </div>
                    </div>
                  </div>
                  <div className="col-md-4 mb-3">
                    <div className="card bg-dark border-secondary h-100">
                      <div className="card-body text-center">
                        <div className="text-muted small mb-1">Stage Transitions</div>
                        <div className="display-6 fw-bold text-warning">
                          {data?.stage_transitions}
                        </div>
                        <div className="text-muted small">sleep fragmentation index</div>
                      </div>
                    </div>
                  </div>
                  <div className="col-md-4 mb-3">
                    <div className="card bg-dark border-secondary h-100">
                      <div className="card-body text-center">
                        <div className="text-muted small mb-1">Selected Recording</div>
                        <div className="small font-monospace text-success text-break">
                          {selectedHyp ? selectedHyp.split('/').pop() : data?.hypnogram || '—'}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </>
        )}

        {/* ── RECORDINGS ── */}
        {tab === 'recordings' && (
          <div className="card bg-dark border-secondary">
            <div className="card-header text-white fw-semibold d-flex justify-content-between">
              <span>📁 Sleep-EDFx Recordings ({data?.n_total} total, showing first 20)</span>
              <span className="text-muted small">Click a row to load its architecture</span>
            </div>
            <div className="card-body p-0">
              <div className="table-responsive" style={{ maxHeight: 480 }}>
                <table className="table table-dark table-sm table-hover mb-0">
                  <thead className="sticky-top" style={{ background: '#1e293b' }}>
                    <tr>
                      <th>#</th>
                      <th>Hypnogram File</th>
                      <th>Dataset</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {recordings.map((rec, i) => (
                      <RecordingRow
                        key={i} idx={i} rec={rec}
                        selected={selectedHyp === rec.hypnogram}
                        onSelect={handleSelect}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ── QUALITY ── */}
        {tab === 'quality' && (
          <>
            <div className={`alert alert-${data?.quality === 'PASS' ? 'success' : 'warning'} mb-4`}>
              <strong>Overall Quality:</strong> {data?.quality} &nbsp;·&nbsp;
              {data?.flags?.join(' · ') || 'No flags raised'}
            </div>
            <div className="card bg-dark border-secondary mb-3">
              <div className="card-header text-white fw-semibold">Quality Metrics</div>
              <div className="card-body">
                <table className="table table-dark table-sm mb-0">
                  <tbody>
                    <tr>
                      <td className="text-muted">Sleep Efficiency</td>
                      <td className={`fw-bold text-${data?.sleep_efficiency_pct >= 85 ? 'success' : 'warning'}`}>
                        {data?.sleep_efficiency_pct?.toFixed(1)}%
                        <span className="text-muted fw-normal ms-2 small">(target ≥85%)</span>
                      </td>
                    </tr>
                    <tr>
                      <td className="text-muted">REM %</td>
                      <td className={`fw-bold text-${data?.rem_pct >= 18 && data?.rem_pct <= 30 ? 'success' : 'warning'}`}>
                        {data?.rem_pct?.toFixed(1)}%
                        <span className="text-muted fw-normal ms-2 small">(norm 18–30%)</span>
                      </td>
                    </tr>
                    <tr>
                      <td className="text-muted">Deep Sleep (N3) %</td>
                      <td className={`fw-bold text-${data?.deep_sleep_pct >= 15 && data?.deep_sleep_pct <= 30 ? 'success' : 'warning'}`}>
                        {data?.deep_sleep_pct?.toFixed(1)}%
                        <span className="text-muted fw-normal ms-2 small">(norm 15–30%)</span>
                      </td>
                    </tr>
                    <tr>
                      <td className="text-muted">Stage Transitions</td>
                      <td className="fw-bold text-white">
                        {data?.stage_transitions}
                        <span className="text-muted fw-normal ms-2 small">(fragmentation indicator)</span>
                      </td>
                    </tr>
                    <tr>
                      <td className="text-muted">Source</td>
                      <td className="small text-info">
                        Sleep-EDFx hypnogram (expert-scored) · PhysioNet · MNE staging
                      </td>
                    </tr>
                    <tr>
                      <td className="text-muted">Clinical Disclaimer</td>
                      <td className="small text-warning">
                        Screening-grade only. Not validated for clinical diagnosis.
                        Consult a sleep specialist for clinical interpretation.
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div className="alert alert-dark border-secondary small mb-0">
              <strong>Data Provenance:</strong> Sleep-EDFx Cassette + Telemetry subsets from PhysioNet (Goldberger et al. 2000).
              Hypnograms are expert-scored polysomnography annotations. Stage proportions computed via MNE annotation
              parsing from real EDF hypnogram files. Normative ranges from AASM guidelines.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
