'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const LAT_COLOR = l =>
  l === 'Left' ? 'primary' : l === 'Right' ? 'danger' : l === 'Bilateral' ? 'warning' : 'secondary';

const CAT_COLOR = c => ({
  Aura: 'info', Motor: 'danger', Dialeptic: 'warning', Autonomic: 'secondary',
  Language: 'primary', Automatism: 'success', Postictal: 'dark', Other: 'light',
}[c] || 'secondary');

export default function VideoCorrelationPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetch(`${API}/api/video-correlation/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/video-correlation/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/video-correlation/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'matrix',     label: 'Onset Zone Matrix' },
    { id: 'patients',   label: 'Per Patient' },
    { id: 'definitions', label: 'Definitions & Rules' },
  ];

  const filteredPts = (bd?.per_patient || []).filter(p =>
    !search || p.patient_id.toLowerCase().includes(search.toLowerCase()) ||
    (p.semiology_signs || []).some(s => s.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div>
      <h3>🎥 Video-EEG Semiology Correlation Dashboard</h3>
      <p className="text-muted small">{ov.description}</p>

      {/* KPI Cards */}
      <div className="row mb-3">
        {[
          { label: 'Patients',           value: ov.kpis.total_patients,           color: 'primary' },
          { label: 'Behavioral Events',  value: ov.kpis.total_behavioral_events,  color: 'danger' },
          { label: 'Semiology Categories', value: ov.kpis.semiology_categories,   color: 'info' },
          { label: 'Lateralized',        value: ov.kpis.lateralized_patients,     color: 'warning' },
          { label: 'EEG-Semio Concordant', value: ov.kpis.semiology_eeg_concordant, color: 'success' },
          { label: 'Video Frames',       value: ov.kpis.video_frames_available,   color: 'secondary' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`}
                    onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div className="row">

          {/* Semiology Category Distribution */}
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Semiology Category Distribution</div>
              <div className="card-body">
                {(ov.semiology_category_distribution || []).map((item, i) => {
                  const total = ov.kpis.total_behavioral_events;
                  const pct = total ? Math.round(item.value / total * 100) : 0;
                  return (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <span className="small" style={{ minWidth: 100 }}>{item.name}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{ height: 18 }}>
                          <div className={`progress-bar bg-${CAT_COLOR(item.name)}`}
                               style={{ width: `${pct}%` }}>{item.value}</div>
                        </div>
                      </div>
                      <span className="small text-muted">{pct}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Lateralization Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">EEG Lateralization</div>
              <div className="card-body">
                {(ov.lateralization_distribution || []).map((item, i) => {
                  const total = ov.kpis.total_patients;
                  const pct = total ? Math.round(item.value / total * 100) : 0;
                  return (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <span className={`badge bg-${LAT_COLOR(item.name)} me-2`} style={{ minWidth: 90 }}>
                        {item.name}
                      </span>
                      <div className="flex-grow-1 mx-1">
                        <div className="progress" style={{ height: 16 }}>
                          <div className={`progress-bar bg-${LAT_COLOR(item.name)}`}
                               style={{ width: `${pct}%` }}>{item.value}</div>
                        </div>
                      </div>
                      <span className="small text-muted">{pct}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Concordance Summary */}
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm border-success">
              <div className="card-header fw-bold text-success">Semiology-EEG Concordance</div>
              <div className="card-body text-center">
                <div className="display-5 fw-bold text-success">
                  {ov.concordance_summary?.concordance_rate_pct}%
                </div>
                <div className="text-muted small mb-2">concordance rate</div>
                <div className="small">
                  <span className="badge bg-success me-1">{ov.concordance_summary?.concordant_lat_semio_pairs} concordant</span>
                  <span className="badge bg-danger">{ov.concordance_summary?.discordant_lat_semio_pairs} discordant</span>
                </div>
                <div className="text-muted mt-2" style={{ fontSize: '0.72rem' }}>
                  {ov.concordance_summary?.note}
                </div>
              </div>
            </div>
          </div>

          {/* Top Semiology Signs */}
          <div className="col-md-7 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Top Behavioral Signs (All Patients)</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light"><tr>
                    <th>#</th><th>Semiology Sign</th><th>Count</th><th>Freq</th>
                  </tr></thead>
                  <tbody>
                    {(ov.top_semiology_signs || []).map((s, i) => (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        <td>{s.name}</td>
                        <td><span className="badge bg-danger">{s.count}</span></td>
                        <td>
                          <div className="progress" style={{ height: 10, minWidth: 80 }}>
                            <div className="progress-bar bg-danger"
                                 style={{ width: `${Math.round(s.count / ov.kpis.total_patients * 100)}%` }} />
                          </div>
                          <span className="text-muted small">
                            {Math.round(s.count / ov.kpis.total_patients * 100)}% pts
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Video Frame Status */}
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Video Frame Availability</div>
              <div className="card-body">
                {ov.video_frame_status && (
                  <div>
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small">Videos Found</span>
                      <strong>{ov.video_frame_status.videos_found}</strong>
                    </div>
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small">Processed</span>
                      <strong className="text-success">{ov.video_frame_status.processed}</strong>
                    </div>
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small">Total Frames</span>
                      <strong className="text-primary">{ov.video_frame_status.total_frames}</strong>
                    </div>
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small">FFmpeg Available</span>
                      <span className={`badge bg-${ov.video_frame_status.ffmpeg_available ? 'success' : 'danger'}`}>
                        {ov.video_frame_status.ffmpeg_available ? 'Yes' : 'No'}
                      </span>
                    </div>
                    {(ov.video_frame_status.results || []).map((r, i) => (
                      <div key={i} className="alert alert-info py-1 px-2 mt-2 small">
                        <strong>{r.video}</strong>: {r.frames} frames
                        {r.ok ? ' ✅' : ' ❌'}
                      </div>
                    ))}
                    <div className="text-muted small mt-2">
                      Note: Most seizure semiology is annotated from EEG+clinical records.
                      Prospective video upload will expand frame-level behavioral tagging.
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

        </div>
      )}

      {/* ── Onset Zone Matrix Tab ── */}
      {tab === 'matrix' && bd && (
        <div>
          <h5 className="mb-3">Onset Zone × Semiology Category Matrix</h5>
          <div className="table-responsive">
            <table className="table table-sm table-bordered table-hover">
              <thead className="table-dark">
                <tr>
                  <th>Onset Zone</th>
                  {(bd.onset_zone_semiology_matrix?.columns || []).filter(c => c !== 'onset_zone').map(c => (
                    <th key={c} className="text-center">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.onset_zone_semiology_matrix?.rows || []).map((row, i) => (
                  <tr key={i}>
                    <td className="fw-bold small">{row.onset_zone}</td>
                    {(bd.onset_zone_semiology_matrix?.columns || []).filter(c => c !== 'onset_zone').map(c => (
                      <td key={c} className="text-center">
                        {row[c] > 0
                          ? <span className={`badge bg-${CAT_COLOR(c)}`}>{row[c]}</span>
                          : <span className="text-muted">–</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h5 className="mt-4 mb-3">Semiology × Lateralization Heatmap (Top 15)</h5>
          <div className="table-responsive">
            <table className="table table-sm table-bordered">
              <thead className="table-light">
                <tr>
                  <th>Semiology Sign</th>
                  <th className="text-center">Left</th>
                  <th className="text-center">Right</th>
                  <th className="text-center">Bilateral</th>
                  <th className="text-center">Non-lateralized</th>
                  <th className="text-center">Total</th>
                </tr>
              </thead>
              <tbody>
                {(bd.semiology_lateralization_heatmap || []).map((row, i) => (
                  <tr key={i}>
                    <td className="small">{row.semiology}</td>
                    <td className="text-center"><span className={row.Left ? 'badge bg-primary' : 'text-muted'}>{row.Left || '–'}</span></td>
                    <td className="text-center"><span className={row.Right ? 'badge bg-danger' : 'text-muted'}>{row.Right || '–'}</span></td>
                    <td className="text-center"><span className={row.Bilateral ? 'badge bg-warning text-dark' : 'text-muted'}>{row.Bilateral || '–'}</span></td>
                    <td className="text-center"><span className={row['Non-lateralized'] ? 'badge bg-secondary' : 'text-muted'}>{row['Non-lateralized'] || '–'}</span></td>
                    <td className="text-center fw-bold">{row.total}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Per Patient Tab ── */}
      {tab === 'patients' && bd && (
        <div>
          <div className="mb-3">
            <input
              className="form-control"
              placeholder="Search patient or semiology sign..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{ maxWidth: 360 }}
            />
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Signs</th>
                  <th>Primary Category</th>
                  <th>Lateralization</th>
                  <th>Onset Zone</th>
                  <th>Semiology</th>
                </tr>
              </thead>
              <tbody>
                {filteredPts.map((p, i) => (
                  <tr key={i}>
                    <td className="fw-bold small">{p.patient_id}</td>
                    <td className="text-center">
                      <span className="badge bg-secondary">{p.n_semiology_signs}</span>
                    </td>
                    <td>
                      <span className={`badge bg-${CAT_COLOR(p.primary_category)}`}>
                        {p.primary_category}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${LAT_COLOR(p.lateralization)}`}>
                        {p.lateralization}
                      </span>
                    </td>
                    <td className="small">{p.onset_zone}</td>
                    <td>
                      <div className="d-flex flex-wrap gap-1">
                        {(p.semiology_signs || []).map((s, j) => (
                          <span key={j} className="badge bg-light text-dark border small"
                                style={{ fontSize: '0.7rem' }}>{s}</span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-muted small">{filteredPts.length} patients shown</div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Semiology Categories</div>
              <div className="card-body">
                {Object.entries(defs.semiology_categories || {}).map(([cat, desc]) => (
                  <div key={cat} className="mb-2">
                    <span className={`badge bg-${CAT_COLOR(cat)} me-2`}>{cat}</span>
                    <span className="small">{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Lateralization Rules (Lüders 1998)</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr>
                    <th>Semiology Sign</th><th>Localizes To</th>
                  </tr></thead>
                  <tbody>
                    {(defs.lateralization_semiology_rules || []).map((r, i) => (
                      <tr key={i}>
                        <td className="small">{r.sign}</td>
                        <td><span className="badge bg-info text-dark">{r.localizes_to}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Concordance Methodology</div>
              <div className="card-body">
                <p className="small mb-2">{defs.concordance_methodology}</p>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">References</div>
              <div className="card-body">
                <ol className="small mb-0">
                  {(defs.references || []).map((r, i) => <li key={i}>{r}</li>)}
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
