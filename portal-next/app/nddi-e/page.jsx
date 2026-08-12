'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const LEVEL_COLOR = { Normal: 'success', Borderline: 'warning', 'Screen+': 'danger' };
const TREND_COLOR = { improving: 'success', worsening: 'danger', stable: 'secondary' };
const TREND_ICON  = { improving: '↓ Improving', worsening: '↑ Worsening', stable: '→ Stable' };

const SUBSCALE_COLOR = {
  functional: 'primary', hopelessness: 'danger', cognitive: 'warning',
  suicidality: 'danger', emotional: 'info', anhedonia: 'secondary',
};

export default function NDDIEDashboard() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [search, setSearch] = useState('');
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/nddi-e/overview`).then(r => r.json()),
      fetch(`${API}/api/nddi-e/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nddi-e/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading NDDI-E data…</div>;

  const TABS = [
    { id: 'overview',    label: '📊 Overview' },
    { id: 'patients',    label: '👤 Per Patient' },
    { id: 'items',       label: '📋 Item Analysis' },
    { id: 'log',         label: '📝 Assessment Log' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const sevDist = ov.severity_distribution || {};
  const total   = ov.total_assessments || 1;
  const patients = (bd?.patient_summary || []).filter(p =>
    !search ||
    p.patient_id.toLowerCase().includes(search.toLowerCase()) ||
    (p.latest_level || '').toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="p-3">
      <h3>🧠 NDDI-E Dashboard</h3>
      <p className="text-muted">
        Neurological Disorders Depression Inventory for Epilepsy —{' '}
        {ov.total_assessments} assessments · {ov.unique_patients} patients ·{' '}
        avg score {ov.avg_score} / 24 · {ov.screen_positive_pct}% screen positive (≥15)
      </p>

      {ov.suicidality_flag_count > 0 && (
        <div className="alert alert-danger py-2 d-flex align-items-center gap-2 mb-3">
          <span className="fs-5">⚠️</span>
          <span>
            <strong>{ov.suicidality_flag_count} patient{ov.suicidality_flag_count > 1 ? 's' : ''}</strong>{' '}
            ({ov.suicidality_flag_pct}%) flagged on Item 4 (death wish ≥ Often) — suicidality protocol applies.
          </span>
        </div>
      )}

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI row */}
          <div className="row mb-3">
            {[
              ['Total Assessments', ov.total_assessments, 'primary'],
              ['Unique Patients',   ov.unique_patients,    'info'],
              ['Avg Score',         `${ov.avg_score} / 24`, 'warning'],
              ['Screen Positive',   `${ov.screen_positive_count} (${ov.screen_positive_pct}%)`, 'danger'],
              ['Suicidality Flags', ov.suicidality_flag_count, 'danger'],
              ['Min / Max Score',   `${ov.min_score} / ${ov.max_score}`, 'secondary'],
            ].map(([label, val, c]) => (
              <div key={label} className="col-6 col-md-2 mb-2">
                <div className="card shadow-sm h-100">
                  <div className="card-body text-center py-2">
                    <div className={`h5 mb-0 text-${c}`}>{val}</div>
                    <div className="text-muted small">{label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row mb-3">
            {/* Severity distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Severity Distribution</h6>
                  {['Normal', 'Borderline', 'Screen+'].map(lvl => {
                    const cnt = sevDist[lvl] || 0;
                    const pct = ((cnt / total) * 100).toFixed(0);
                    return (
                      <div key={lvl} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span className={`badge bg-${LEVEL_COLOR[lvl] || 'secondary'}`}>{lvl}</span>
                          <span>{cnt} ({pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: 10 }}>
                          <div className={`progress-bar bg-${LEVEL_COLOR[lvl]}`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                  <div className="mt-3 small text-muted">Cutoff: ≥15 = Screen positive</div>
                </div>
              </div>
            </div>

            {/* Score histogram */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Score Histogram (6–24)</h6>
                  {(ov.score_histogram || []).map(({ bin, count }) => {
                    const pct = ((count / total) * 100).toFixed(0);
                    const isPos = parseInt(bin) >= 15;
                    return (
                      <div key={bin} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span className="font-monospace">{bin}</span>
                          <span>{count}</span>
                        </div>
                        <div className="progress" style={{ height: 10 }}>
                          <div className={`progress-bar ${isPos ? 'bg-danger' : 'bg-success'}`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Monthly trend */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Monthly Assessments</h6>
                  {(ov.monthly_trend || []).map(({ month, assessments, avg_score }) => (
                    <div key={month} className="d-flex justify-content-between border-bottom py-2">
                      <span className="small">{month}</span>
                      <div className="text-end">
                        <span className="badge bg-primary me-1">{assessments}</span>
                        <span className="text-muted small">avg {avg_score}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Score gauge */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6>Average Score — {ov.avg_score} / 24</h6>
              <div className="progress" style={{ height: 28 }}>
                <div
                  className={`progress-bar ${ov.avg_score >= 15 ? 'bg-danger' : ov.avg_score >= 13 ? 'bg-warning' : 'bg-success'}`}
                  style={{ width: `${((ov.avg_score || 0) / 24) * 100}%` }}
                >
                  {ov.avg_score}
                </div>
              </div>
              <div className="d-flex justify-content-between small text-muted mt-1">
                <span>6 — Normal (≤12)</span>
                <span>13–14 — Borderline</span>
                <span>≥15 — Screen+ — 24</span>
              </div>
            </div>
          </div>

          {/* Item averages summary */}
          <div className="card shadow-sm">
            <div className="card-body">
              <h6>Item Endorsement Rates (avg 1–4)</h6>
              <div className="row">
                {(ov.item_averages || []).map(({ item, label, subscale, avg_score }) => (
                  <div key={item} className="col-md-6 mb-2">
                    <div className="d-flex align-items-center gap-2 mb-1">
                      <span className={`badge bg-${SUBSCALE_COLOR[subscale] || 'secondary'}`}>{item}</span>
                      <span className="small flex-grow-1">{label}</span>
                      <span className="text-muted small fw-bold">{avg_score?.toFixed(2)}</span>
                    </div>
                    <div className="progress" style={{ height: 8 }}>
                      <div
                        className={`progress-bar ${avg_score >= 3 ? 'bg-danger' : avg_score >= 2 ? 'bg-warning' : 'bg-success'}`}
                        style={{ width: `${((avg_score || 0) / 4) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-muted small mt-2 mb-0">1 = Never · 2 = Sometimes · 3 = Often · 4 = Always</p>
            </div>
          </div>
        </div>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && (
        <div>
          <input
            className="form-control mb-3"
            style={{ maxWidth: 240 }}
            placeholder="Search patient / level…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Assessments</th>
                  <th>Avg Score</th>
                  <th>Latest</th>
                  <th>Level</th>
                  <th>Suicidality</th>
                  <th>Trend</th>
                  <th>First Date</th>
                  <th>Latest Date</th>
                </tr>
              </thead>
              <tbody>
                {patients.map(p => (
                  <tr key={p.patient_id}>
                    <td><span className="badge bg-secondary">{p.patient_id}</span></td>
                    <td className="text-center">{p.assessments}</td>
                    <td>
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{ height: 10, minWidth: 50 }}>
                          <div
                            className={`progress-bar bg-${LEVEL_COLOR[p.latest_level] || 'secondary'}`}
                            style={{ width: `${((p.avg_score || 0) / 24) * 100}%` }}
                          />
                        </div>
                        <span className="small">{p.avg_score}</span>
                      </div>
                    </td>
                    <td className="text-center fw-bold">{p.latest_score}</td>
                    <td>
                      <span className={`badge bg-${LEVEL_COLOR[p.latest_level] || 'secondary'}`}>
                        {p.latest_level}
                      </span>
                    </td>
                    <td className="text-center">
                      {p.suicidality_positive
                        ? <span className="badge bg-danger">⚠ Positive</span>
                        : <span className="badge bg-success">—</span>}
                    </td>
                    <td>
                      <span className={`badge bg-${TREND_COLOR[p.trend] || 'secondary'}`}>
                        {TREND_ICON[p.trend] || p.trend}
                      </span>
                    </td>
                    <td className="small text-muted">{p.first_date?.slice(0, 10) || '—'}</td>
                    <td className="small text-muted">{p.latest_date?.slice(0, 10) || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-muted small">{patients.length} patients shown</div>
          </div>
        </div>
      )}

      {/* ── ITEM ANALYSIS ── */}
      {tab === 'items' && (
        <div>
          <div className="row">
            {(bd?.item_averages || []).map(({ item, label, subscale, avg_score, response_distribution }) => (
              <div key={item} className="col-md-6 mb-3">
                <div className="card shadow-sm h-100">
                  <div className="card-body">
                    <div className="d-flex align-items-center gap-2 mb-2">
                      <span className={`badge bg-${SUBSCALE_COLOR[subscale] || 'secondary'}`}>{item.toUpperCase()}</span>
                      {subscale === 'suicidality' && <span className="badge bg-danger">⚠ Suicidality</span>}
                      <span className="small text-muted">{subscale}</span>
                    </div>
                    <p className="mb-2 fw-semibold small">{label}</p>
                    <div className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>Avg Score</span>
                        <span className="fw-bold">{avg_score?.toFixed(2)} / 4</span>
                      </div>
                      <div className="progress" style={{ height: 12 }}>
                        <div
                          className={`progress-bar ${avg_score >= 3 ? 'bg-danger' : avg_score >= 2 ? 'bg-warning' : 'bg-success'}`}
                          style={{ width: `${((avg_score || 0) / 4) * 100}%` }}
                        />
                      </div>
                    </div>
                    <div className="small text-muted mt-2">Response distribution:</div>
                    {Object.entries(response_distribution || {}).map(([resp, cnt]) => (
                      <div key={resp} className="d-flex justify-content-between small py-1 border-bottom">
                        <span>{resp}</span>
                        <span className="fw-semibold">{cnt}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── ASSESSMENT LOG ── */}
      {tab === 'log' && (
        <div>
          <h6 className="mb-3">Assessment Log ({bd?.assessment_log?.length || 0} records)</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr>
                  <th>ID</th><th>Patient</th><th>Score</th><th>Level</th>
                  <th>Interpretation</th><th>Alert</th><th>Examiner</th><th>Date</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.assessment_log || []).map(a => (
                  <tr key={a.id}>
                    <td className="small text-muted">{a.id}</td>
                    <td><span className="badge bg-secondary">{a.patient_id}</span></td>
                    <td className="fw-bold">{a.score}</td>
                    <td>
                      <span className={`badge bg-${LEVEL_COLOR[a.level] || 'secondary'}`}>{a.level}</span>
                    </td>
                    <td className="small">{a.interpretation}</td>
                    <td className="small">
                      {a.alert
                        ? <span className="text-danger">{a.alert}</span>
                        : <span className="text-muted">—</span>}
                    </td>
                    <td className="small text-muted">{a.examiner}</td>
                    <td className="small text-muted">{a.date?.slice(0, 10) || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h5>{defs.title}</h5>
              <p className="text-muted">{defs.description}</p>
              <div className="d-flex flex-wrap gap-3 small">
                <span>Developer: <strong>{defs.developer}</strong></span>
                <span>Admin: <strong>{defs.administration}</strong></span>
                <span>Range: <strong>{defs.score_range?.min}–{defs.score_range?.max}</strong></span>
                <span>Cutoff: <strong className="text-danger">≥{defs.cutoff}</strong></span>
              </div>
            </div>
          </div>

          <div className="row mb-3">
            <div className="col-md-5 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Severity Thresholds</h6>
                  {(defs.severity_thresholds || []).map(({ level, min, max, description }) => (
                    <div key={level} className="mb-3 pb-2 border-bottom">
                      <div className="d-flex align-items-center gap-2 mb-1">
                        <span className={`badge bg-${LEVEL_COLOR[level] || 'secondary'}`}>{level}</span>
                        <span className="small text-muted">{min}–{max}</span>
                      </div>
                      <div className="text-muted small">{description}</div>
                    </div>
                  ))}
                  <div className="alert alert-danger py-2 mt-2 small">
                    <strong>Suicidality Protocol:</strong> {defs.suicidality_protocol}
                  </div>
                </div>
              </div>
            </div>

            <div className="col-md-7 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Scale Items (6-item, each 1–4)</h6>
                  {(defs.items || []).map(({ id, number, label, subscale, clinical_note }) => (
                    <div key={id} className="d-flex gap-2 mb-2 pb-2 border-bottom align-items-start">
                      <span className={`badge bg-${SUBSCALE_COLOR[subscale] || 'secondary'} mt-1`}>{id}</span>
                      <div>
                        <div className="small fw-semibold">{label}</div>
                        <div className="text-muted" style={{ fontSize: '0.72rem' }}>
                          {subscale}
                          {clinical_note && <span className="text-danger ms-2">⚠ {clinical_note}</span>}
                        </div>
                      </div>
                    </div>
                  ))}
                  <div className="mt-2 small text-muted">
                    Response: 1=Never · 2=Sometimes · 3=Often · 4=Always
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Advantages Over PHQ-9 in Epilepsy</h6>
                  {(defs.advantages_over_phq9 || []).map((adv, i) => (
                    <div key={i} className="small text-muted mb-2">✓ {adv}</div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Clinical Uses</h6>
                  {(defs.clinical_use || []).map((use, i) => (
                    <div key={i} className="small text-muted mb-2">• {use}</div>
                  ))}
                  <h6 className="mt-3">Regulatory Context</h6>
                  <p className="small text-muted">{defs.regulatory_context}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-body">
              <h6>References</h6>
              {(defs.references || []).map((ref, i) => (
                <div key={i} className="small text-muted mb-2">• {ref}</div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
