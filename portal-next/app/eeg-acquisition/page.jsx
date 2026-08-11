'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const recColor = t =>
  t === 'LTM'        ? 'primary' :
  t === 'ambulatory' ? 'info'    :
  t === 'routine'    ? 'success' : 'warning';   // video_eeg

const impColor = g =>
  g === 'Good' ? 'success' : g === 'Fair' ? 'warning' : 'danger';

const qualColor = g =>
  g === 'Good' ? 'success' : g === 'Fair' ? 'warning' : 'danger';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 col-xl-2 mb-2">
      <div className="card shadow-sm h-100 border-0">
        <div className="card-body text-center py-2 px-1">
          <div className={`h3 mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.72rem' }}>{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ items, colorFn, valueKey = 'count', labelKey = 'label', title }) {
  if (!items || !items.length) return null;
  const mx = Math.max(...items.map(i => i[valueKey] ?? 0));
  return (
    <div className="mb-3">
      {title && <div className="fw-semibold small mb-1">{title}</div>}
      {items.map((it, i) => (
        <div key={i} className="d-flex align-items-center mb-1">
          <div className="text-end me-2 text-muted small" style={{ width: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {it[labelKey]}
          </div>
          <div className="flex-grow-1">
            <div className="progress" style={{ height: 18 }}>
              <div
                className={`progress-bar bg-${colorFn ? colorFn(it[labelKey]) : 'primary'}`}
                style={{ width: `${mx ? ((it[valueKey] / mx) * 100) : 0}%` }}
              >
                <span className="small">{it[valueKey]}</span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function EEGAcquisitionPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [sortCol, setSortCol] = useState('patient_id');
  const [sortDir, setSortDir] = useState('asc');

  useEffect(() => {
    fetch(`${API}/api/eeg-acquisition/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/eeg-acquisition/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/eeg-acquisition/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'channel',     label: 'Channel Quality' },
    { id: 'patients',    label: 'Per Patient' },
    { id: 'definitions', label: 'Definitions' },
  ];

  // Derived distributions for bar charts
  const recDist = Object.entries(ov.recording_type_distribution || {})
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);

  const montageDist = Object.entries(ov.montage_distribution || {})
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);

  const srDist = Object.entries(ov.sampling_rate_distribution || {})
    .map(([label, count]) => ({ label: `${label} Hz`, count }))
    .sort((a, b) => b.count - a.count);

  const impGrades = ov.channel_quality_summary?.impedance_grade_distribution || {};
  const qualGrades = ov.channel_quality_summary?.quality_grade_distribution || {};

  const impDist = Object.entries(impGrades).map(([label, count]) => ({ label, count }));
  const qualDist = Object.entries(qualGrades).map(([label, count]) => ({ label, count }));

  // Monthly trend
  const monthlyTrend = ov.monthly_trend || [];

  // Per-patient sorting
  const patients = bd?.per_patient_summary || [];
  const sorted = [...patients].sort((a, b) => {
    const va = a[sortCol] ?? '';
    const vb = b[sortCol] ?? '';
    if (typeof va === 'number') return sortDir === 'asc' ? va - vb : vb - va;
    return sortDir === 'asc' ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  });

  const toggleSort = col => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('asc'); }
  };
  const sortIcon = col => sortCol === col ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';

  // Recording type detail from breakdown
  const recDetail = bd?.recording_type_detail || [];

  // Channel detail map (per patient)
  const channelMap = bd?.channel_detail || {};

  return (
    <div>
      <h3>&#x1f4f6; EEG Acquisition Dashboard</h3>
      <p className="text-muted small">
        Real clinical.db data: {ov.total_studies} EEG studies across {ov.total_patients} patients —
        recording type, montage, sampling rate, duration, impedance, SNR, and per-channel quality.
        IFCN 2017 + ACNS 2021 quality standards.
      </p>

      {/* KPI Cards */}
      <div className="row mb-3">
        <KPI label="Total Studies"     value={ov.total_studies}            color="primary" />
        <KPI label="Patients"          value={ov.total_patients}           color="info" />
        <KPI label="Avg Duration"      value={`${Math.round(ov.duration_stats?.avg ?? 0)} min`} color="dark" sub={`${Math.round((ov.duration_stats?.avg ?? 0)/60 * 10)/10} hrs`} />
        <KPI label="Total Channels"    value={ov.channel_quality_summary?.total_channels} color="secondary" />
        <KPI label="Avg Impedance"     value={`${ov.avg_impedance_kohm} kΩ`} color={ov.avg_impedance_kohm > 10 ? 'danger' : ov.avg_impedance_kohm > 5 ? 'warning' : 'success'} sub="target <5 kΩ" />
        <KPI label="Avg SNR"           value={`${ov.avg_snr_db} dB`}      color={ov.avg_snr_db >= 20 ? 'success' : ov.avg_snr_db >= 10 ? 'warning' : 'danger'} sub="target ≥20 dB" />
        <KPI label="Good Impedance"    value={`${ov.pct_good_impedance}%`} color={ov.pct_good_impedance >= 60 ? 'success' : ov.pct_good_impedance >= 30 ? 'warning' : 'danger'} />
        <KPI label="Good Quality"      value={`${ov.pct_good_quality}%`}   color={ov.pct_good_quality >= 70 ? 'success' : ov.pct_good_quality >= 40 ? 'warning' : 'danger'} />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          <div className="row">
            {/* Recording Type Distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-body">
                  <h6 className="card-title">Recording Types</h6>
                  <Bar
                    items={recDist}
                    labelKey="label"
                    valueKey="count"
                    colorFn={recColor}
                  />
                  <div className="mt-2">
                    {recDist.map(r => (
                      <span key={r.label} className={`badge bg-${recColor(r.label)} me-1 mb-1`}>
                        {r.label}: {r.count}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Montage Distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-body">
                  <h6 className="card-title">Montage Type</h6>
                  <Bar items={montageDist} labelKey="label" valueKey="count" colorFn={() => 'info'} />
                  <div className="mt-2">
                    {montageDist.map(m => (
                      <span key={m.label} className="badge bg-info text-dark me-1 mb-1">
                        {m.label}: {m.count}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Sampling Rate Distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-body">
                  <h6 className="card-title">Sampling Rate</h6>
                  <Bar items={srDist} labelKey="label" valueKey="count" colorFn={() => 'secondary'} />
                  <div className="mt-2">
                    {srDist.map(s => (
                      <span key={s.label} className="badge bg-secondary me-1 mb-1">
                        {s.label}: {s.count}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Duration Stats */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">Recording Duration Stats</h6>
              <div className="row">
                <div className="col-md-4 text-center">
                  <div className="h4 text-primary">{Math.round(ov.duration_stats?.avg ?? 0)} min</div>
                  <div className="text-muted small">Average Duration</div>
                </div>
                <div className="col-md-4 text-center">
                  <div className="h4 text-success">{ov.duration_stats?.min ?? 0} min</div>
                  <div className="text-muted small">Minimum Duration</div>
                </div>
                <div className="col-md-4 text-center">
                  <div className="h4 text-info">{ov.duration_stats?.max ?? 0} min</div>
                  <div className="text-muted small">Maximum Duration</div>
                </div>
              </div>
            </div>
          </div>

          {/* Recording Type Detail Table */}
          {recDetail.length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title">Recording Type Summary</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Type</th>
                        <th>Count</th>
                        <th>Avg Duration (min)</th>
                        <th>Avg Impedance (kΩ)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recDetail.map(r => (
                        <tr key={r.recording_type}>
                          <td><span className={`badge bg-${recColor(r.recording_type)}`}>{r.recording_type}</span></td>
                          <td>{r.count}</td>
                          <td>{Math.round(r.avg_duration_min)}</td>
                          <td>
                            <span className={`text-${r.avg_impedance_kohm > 10 ? 'danger' : r.avg_impedance_kohm > 5 ? 'warning' : 'success'} fw-bold`}>
                              {r.avg_impedance_kohm?.toFixed(1)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Monthly Trend */}
          {monthlyTrend.length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title">Monthly Recording Trend</h6>
                <div className="d-flex align-items-end gap-1" style={{ height: 80 }}>
                  {(() => {
                    const maxCnt = Math.max(...monthlyTrend.map(m => m.cnt));
                    return monthlyTrend.map((m, i) => (
                      <div key={i} className="d-flex flex-column align-items-center flex-grow-1">
                        <div
                          className="bg-primary rounded-top"
                          style={{ width: '100%', height: `${maxCnt ? (m.cnt / maxCnt) * 60 : 4}px` }}
                          title={`${m.month}: ${m.cnt}`}
                        />
                        <div className="text-muted" style={{ fontSize: '0.55rem', transform: 'rotate(-45deg)', whiteSpace: 'nowrap' }}>
                          {m.month?.slice(5)}
                        </div>
                      </div>
                    ));
                  })()}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── CHANNEL QUALITY ── */}
      {tab === 'channel' && (
        <div>
          <div className="row mb-3">
            {/* Impedance Grade Distribution */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-body">
                  <h6 className="card-title">Impedance Grade Distribution</h6>
                  <p className="text-muted small">Target: Good (&lt;5 kΩ) | Fair (5–10 kΩ) | Poor (&gt;10 kΩ)</p>
                  <Bar
                    items={impDist}
                    labelKey="label"
                    valueKey="count"
                    colorFn={impColor}
                  />
                  <div className="mt-2 d-flex gap-2 flex-wrap">
                    {impDist.map(g => (
                      <span key={g.label} className={`badge bg-${impColor(g.label)}`}>
                        {g.label}: {g.count} ({Math.round(g.count / (ov.channel_quality_summary?.total_channels || 1) * 100)}%)
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Quality Grade Distribution */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-body">
                  <h6 className="card-title">Overall Channel Quality</h6>
                  <p className="text-muted small">Good = low impedance + high SNR + low artifact burden</p>
                  <Bar
                    items={qualDist}
                    labelKey="label"
                    valueKey="count"
                    colorFn={qualColor}
                  />
                  <div className="mt-2 d-flex gap-2 flex-wrap">
                    {qualDist.map(g => (
                      <span key={g.label} className={`badge bg-${qualColor(g.label)}`}>
                        {g.label}: {g.count} ({Math.round(g.count / (ov.channel_quality_summary?.total_channels || 1) * 100)}%)
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Poor-impedance patients */}
          {Object.keys(channelMap).length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title">Patients with Poor-Impedance Channels</h6>
                <p className="text-muted small">Channels with impedance &gt;10 kΩ flagged for re-gelling or electrode replacement.</p>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Patient</th>
                        <th>Channel</th>
                        <th>Impedance (kΩ)</th>
                        <th>SNR (dB)</th>
                        <th>Grade</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(channelMap).flatMap(([pid, chs]) =>
                        chs.map((ch, i) => (
                          <tr key={`${pid}-${i}`}>
                            <td><code>{pid}</code></td>
                            <td>{ch.channel}</td>
                            <td className="text-danger fw-bold">{ch.impedance_kohm?.toFixed(1)}</td>
                            <td className={ch.snr_db < 10 ? 'text-danger' : ch.snr_db < 20 ? 'text-warning' : 'text-success'}>
                              {ch.snr_db?.toFixed(1)}
                            </td>
                            <td><span className={`badge bg-${impColor(ch.impedance_grade)}`}>{ch.impedance_grade}</span></td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Quality alert */}
          {ov.pct_good_impedance < 50 && (
            <div className="alert alert-warning">
              <strong>⚠️ Impedance Alert:</strong> Only {ov.pct_good_impedance}% of channels have good impedance (&lt;5 kΩ).
              Average across studies is {ov.avg_impedance_kohm} kΩ — above the 5 kΩ target (IFCN 2017).
              Consider protocol review: electrode preparation, abrasion technique, and gel application.
            </div>
          )}
        </div>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && (
        <div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  {[
                    ['patient_id',    'Patient'],
                    ['recording_type','Type'],
                    ['duration_min',  'Duration (min)'],
                    ['sampling_rate', 'Sample Rate (Hz)'],
                    ['montage',       'Montage'],
                    ['channels_good', 'Good Ch'],
                    ['channels_fair', 'Fair Ch'],
                    ['channels_poor', 'Poor Ch'],
                  ].map(([col, label]) => (
                    <th key={col} style={{ cursor: 'pointer', userSelect: 'none' }}
                        onClick={() => toggleSort(col)}>
                      {label}{sortIcon(col)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map(p => (
                  <tr key={p.patient_id}>
                    <td><code>{p.patient_id}</code></td>
                    <td><span className={`badge bg-${recColor(p.recording_type)}`}>{p.recording_type}</span></td>
                    <td>{p.duration_min}</td>
                    <td>{p.sampling_rate}</td>
                    <td><span className="badge bg-secondary">{p.montage}</span></td>
                    <td className="text-success fw-bold">{p.channels_good}</td>
                    <td className="text-warning fw-bold">{p.channels_fair}</td>
                    <td className={`fw-bold${p.channels_poor > 0 ? ' text-danger' : ' text-success'}`}>
                      {p.channels_poor}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-muted small mt-1">
            {sorted.length} studies. Click column headers to sort.
            Poor channels = impedance &gt;10 kΩ — flag for re-gelling.
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="row">
            {/* Glossary */}
            <div className="col-md-8 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-body">
                  <h6 className="card-title">Glossary</h6>
                  <dl className="row mb-0">
                    {(defs.glossary || []).map((g, i) => (
                      <div key={i} className="col-12 mb-2">
                        <dt className="fw-semibold">{g.term}</dt>
                        <dd className="text-muted small mb-0">{g.definition}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </div>
            </div>

            {/* Channel Regions + References */}
            <div className="col-md-4 mb-3">
              {defs.channel_regions && (
                <div className="card shadow-sm border-0 mb-3">
                  <div className="card-body">
                    <h6 className="card-title">10-20 Channel Regions</h6>
                    {Object.entries(defs.channel_regions).map(([region, channels]) => (
                      <div key={region} className="mb-2">
                        <div className="fw-semibold small">{region}</div>
                        <div>{(channels || []).map(ch => (
                          <span key={ch} className="badge bg-light text-dark border me-1 mb-1">{ch}</span>
                        ))}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {defs.quality_standards && (
                <div className="card shadow-sm border-0 mb-3">
                  <div className="card-body">
                    <h6 className="card-title">Quality Standards</h6>
                    <ul className="list-unstyled small mb-0">
                      {Object.entries(defs.quality_standards).map(([k, v]) => (
                        <li key={k} className="mb-1">
                          <strong>{k.replace(/_/g,' ')}:</strong>{' '}
                          <span className="text-muted">{typeof v === 'object' ? JSON.stringify(v) : v}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {defs.references && (
                <div className="card shadow-sm border-0">
                  <div className="card-body">
                    <h6 className="card-title">References</h6>
                    <ol className="small mb-0">
                      {defs.references.map((r, i) => (
                        <li key={i} className="mb-1 text-muted">{r}</li>
                      ))}
                    </ol>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
