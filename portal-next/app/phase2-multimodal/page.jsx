'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const tierColor = t => ({
  'Complete': 'success',
  'Complete (4/4)': 'success',
  '3 of 4': 'info',
  '2 of 4': 'warning',
  '1 of 4': 'danger',
  'None': 'secondary',
}[t] || 'secondary');

const pct = (n, total) => total ? ((n / total) * 100).toFixed(1) : '0.0';

const Bar = ({value, max, color = 'primary'}) => (
  <div className="progress" style={{height: 8}}>
    <div
      className={`progress-bar bg-${color}`}
      style={{width: `${max ? (value / max) * 100 : 0}%`}}
    />
  </div>
);

export default function Phase2MultimodalDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/phase2-multimodal/overview`).then(r => r.json()),
      fetch(`${API}/api/phase2-multimodal/breakdown`).then(r => r.json()),
      fetch(`${API}/api/phase2-multimodal/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => {setOv(o); setBd(b); setDefs(d);})
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Phase 2 multimodal data…</div>;

  const TABS = [
    {id: 'overview',    label: 'Overview'},
    {id: 'coverage',    label: 'Coverage Map'},
    {id: 'patients',    label: 'Per Patient'},
    {id: 'video',       label: 'Patient Video'},
    {id: 'eeg',         label: 'Video-EEG'},
    {id: 'definitions', label: 'Definitions'},
  ];

  const modColors = ['primary', 'info', 'success', 'warning'];

  return (
    <div className="p-3">
      <h3>&#x1f4ca; Phase 2 Multimodal Coverage</h3>
      <p className="text-muted mb-1">
        Cross-modality data coverage for Phase 2 &mdash; patient video + video-EEG + MRI + neuropsych
        &mdash; {ov.total_patients} patients, {ov.complete_phase2} complete ({ov.complete_phase2_pct}%)
      </p>
      <span className="badge bg-success me-1">Phase 2</span>
      <span className="badge bg-secondary me-1">{ov.total_patients} patients</span>
      <span className="badge bg-primary me-1">{ov.complete_phase2_pct}% complete</span>
      <span className="badge bg-info">{ov.video_sessions_total} video sessions</span>

      <ul className="nav nav-tabs mt-3 mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI cards */}
          <div className="row mb-4">
            {[
              ['Total Patients', ov.total_patients, 'secondary', '&#x1f465;'],
              ['Complete (all 4)', ov.complete_phase2, 'success', '&#x2705;'],
              ['3 of 4 Modalities', ov.three_modalities, 'info', '&#x1f4ca;'],
              ['2 of 4 Modalities', ov.two_modalities, 'warning', '&#x26a0;&#xfe0f;'],
              ['Video Sessions', ov.video_sessions_total, 'primary', '&#x1f4f9;'],
              ['Neuropsych Avg MoCA', ov.neuropsych_avg_moca?.toFixed(1), 'info', '&#x1f9e0;'],
            ].map(([label, val, color, icon]) => (
              <div key={label} className="col-6 col-md-4 col-xl-2 mb-3">
                <div className={`card border-${color} h-100`}>
                  <div className="card-body text-center p-2">
                    <div style={{fontSize: 22}} dangerouslySetInnerHTML={{__html: icon}} />
                    <div className={`fs-4 fw-bold text-${color}`}>{val}</div>
                    <div className="small text-muted">{label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Per-modality coverage bars */}
          <h5>Modality Coverage Across {ov.total_patients} Patients</h5>
          <div className="table-responsive mb-4">
            <table className="table table-sm table-bordered">
              <thead className="table-dark">
                <tr>
                  <th>Modality</th>
                  <th>Source</th>
                  <th>Patients</th>
                  <th>Coverage %</th>
                  <th style={{minWidth: 160}}>Bar</th>
                </tr>
              </thead>
              <tbody>
                {(ov.modality_coverage || []).map((m, i) => (
                  <tr key={m.code}>
                    <td><strong>{m.modality}</strong> <span className="badge bg-secondary">{m.code}</span></td>
                    <td><code className="small">{m.source}</code></td>
                    <td>{m.count}</td>
                    <td><span className={`badge bg-${modColors[i]}`}>{m.pct}%</span></td>
                    <td><Bar value={m.count} max={ov.total_patients} color={modColors[i]} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Completeness pyramid */}
          <div className="row mb-4">
            <div className="col-md-6">
              <h5>Completeness Distribution</h5>
              {[
                ['Complete (4/4)', ov.complete_phase2, 'success'],
                ['3 of 4', ov.three_modalities, 'info'],
                ['2 of 4', ov.two_modalities, 'warning'],
                ['1 of 4', ov.one_modality, 'danger'],
                ['None', ov.no_phase2_data, 'secondary'],
              ].map(([label, count, color]) => (
                <div key={label} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className={`badge bg-${color}`}>{label}</span>
                    <span className="small text-muted">{count} patients ({pct(count, ov.total_patients)}%)</span>
                  </div>
                  <Bar value={count} max={ov.total_patients} color={color} />
                </div>
              ))}
            </div>
            <div className="col-md-6">
              <h5>MRI Lesion Types</h5>
              <table className="table table-sm table-hover">
                <thead><tr><th>Lesion Type</th><th>Count</th></tr></thead>
                <tbody>
                  {Object.entries(ov.mri_lesion_type_distribution || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td><span className="badge bg-success">{v}</span></td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* EEG type distribution */}
          <div className="row">
            <div className="col-md-6">
              <h5>EEG Study Types ({ov.eeg_total_studies} studies)</h5>
              <table className="table table-sm table-hover">
                <thead><tr><th>Recording Type</th><th>Count</th></tr></thead>
                <tbody>
                  {Object.entries(ov.eeg_type_distribution || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td><span className="badge bg-info">{v}</span></td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="col-md-6">
              <h5>Video Recording Quality</h5>
              <table className="table table-sm table-hover">
                <thead><tr><th>Quality</th><th>Sessions</th></tr></thead>
                <tbody>
                  {Object.entries(ov.video_quality_distribution || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td><span className="badge bg-primary">{v}</span></td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── COVERAGE MAP ── */}
      {tab === 'coverage' && bd && (
        <div>
          <h5>Modality Co-occurrence Pairs</h5>
          <p className="text-muted small">Number of patients who have BOTH modalities simultaneously.</p>
          <div className="row mb-4">
            {(bd.cooccurrence_pairs || []).map(p => (
              <div key={`${p.a}-${p.b}`} className="col-6 col-md-4 col-xl-2 mb-3">
                <div className="card text-center border-primary">
                  <div className="card-body p-2">
                    <div className="fs-4 fw-bold text-primary">{p.count}</div>
                    <div className="small">{p.a} ∩ {p.b}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <h5>Completeness Tier Distribution</h5>
          <div className="table-responsive">
            <table className="table table-sm table-bordered">
              <thead className="table-dark">
                <tr><th>Tier</th><th>Patients</th><th>%</th><th>Coverage Bar</th></tr>
              </thead>
              <tbody>
                {Object.entries(bd.completeness_distribution || {}).map(([tier, count]) => (
                  <tr key={tier}>
                    <td><span className={`badge bg-${tierColor(tier)}`}>{tier}</span></td>
                    <td>{count}</td>
                    <td>{pct(count, ov.total_patients)}%</td>
                    <td style={{minWidth: 200}}><Bar value={count} max={ov.total_patients} color={tierColor(tier)} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h5 className="mt-4">Video Session Monthly Trend</h5>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead><tr><th>Month</th><th>Sessions</th><th>Seizure Events</th><th>Total Hours</th></tr></thead>
              <tbody>
                {(bd.video_monthly_trend || []).map(r => (
                  <tr key={r.month}>
                    <td>{r.month}</td>
                    <td>{r.sessions}</td>
                    <td>{r.seizure_events}</td>
                    <td>{r.total_hours?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && bd && (
        <div>
          <h5>Per-Patient Phase 2 Coverage Matrix ({bd.per_patient?.length} patients)</h5>
          <div className="table-responsive">
            <table className="table table-sm table-bordered table-hover">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>&#x1f4f9; Video</th>
                  <th>&#x1f4e1; EEG</th>
                  <th>&#x1fa7b; MRI</th>
                  <th>&#x1f9e0; Neuropsych</th>
                  <th>Modalities</th>
                  <th>Tier</th>
                </tr>
              </thead>
              <tbody>
                {(bd.per_patient || []).map(r => (
                  <tr key={r.patient_id} className={r.complete ? 'table-success' : ''}>
                    <td><code>{r.patient_id}</code></td>
                    <td className="text-center">{r.video ? '✅' : '–'}</td>
                    <td className="text-center">{r.eeg ? '✅' : '–'}</td>
                    <td className="text-center">{r.mri ? '✅' : '–'}</td>
                    <td className="text-center">{r.neuropsych ? '✅' : '–'}</td>
                    <td className="text-center fw-bold">{r.modality_count}/4</td>
                    <td><span className={`badge bg-${tierColor(r.tier)}`}>{r.tier}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── PATIENT VIDEO ── */}
      {tab === 'video' && (
        <div>
          <div className="row mb-3">
            {[
              ['Total Sessions', ov.video_sessions_total, 'primary'],
              ['Patients w/ Video', (ov.modality_coverage || [])[0]?.count, 'info'],
              ['Avg Duration (h)', ov.video_avg_duration_hours?.toFixed(1), 'success'],
              ['Total Seizure Events', ov.video_total_seizure_events, 'danger'],
            ].map(([label, val, color]) => (
              <div key={label} className="col-6 col-md-3 mb-3">
                <div className={`card border-${color}`}>
                  <div className="card-body text-center p-2">
                    <div className={`fs-4 fw-bold text-${color}`}>{val}</div>
                    <div className="small text-muted">{label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <h5>Video Quality Distribution</h5>
          <table className="table table-sm table-hover mb-4">
            <thead><tr><th>Quality</th><th>Sessions</th><th>%</th></tr></thead>
            <tbody>
              {Object.entries(ov.video_quality_distribution || {}).map(([k, v]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td>{v}</td>
                  <td>{pct(v, ov.video_sessions_total)}%</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h5>Monthly Video Session Trend</h5>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead><tr><th>Month</th><th>Sessions</th><th>Seizure Events</th><th>Total Hours</th></tr></thead>
              <tbody>
                {(bd?.video_monthly_trend || []).map(r => (
                  <tr key={r.month}>
                    <td>{r.month}</td>
                    <td>{r.sessions}</td>
                    <td className={r.seizure_events > 0 ? 'text-danger fw-bold' : ''}>{r.seizure_events}</td>
                    <td>{r.total_hours?.toFixed(1)} h</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── VIDEO-EEG ── */}
      {tab === 'eeg' && bd && (
        <div>
          <div className="row mb-3">
            {[
              ['EEG Studies', ov.eeg_total_studies, 'info'],
              ['Patients w/ EEG', (ov.modality_coverage || [])[1]?.count, 'primary'],
              ['Avg MoCA', ov.neuropsych_avg_moca?.toFixed(1), 'success'],
              ['Avg MMSE', ov.neuropsych_avg_mmse?.toFixed(1), 'warning'],
            ].map(([label, val, color]) => (
              <div key={label} className="col-6 col-md-3 mb-3">
                <div className={`card border-${color}`}>
                  <div className="card-body text-center p-2">
                    <div className={`fs-4 fw-bold text-${color}`}>{val}</div>
                    <div className="small text-muted">{label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <h5>EEG Study Details ({ov.eeg_total_studies} studies)</h5>
          <div className="table-responsive mb-4">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr><th>Patient</th><th>Recording Type</th><th>Duration (min)</th><th>Sampling (Hz)</th><th>Montage</th></tr>
              </thead>
              <tbody>
                {(bd.eeg_detail || []).map((r, i) => (
                  <tr key={i}>
                    <td><code>{r.patient_id}</code></td>
                    <td><span className="badge bg-info">{r.recording_type}</span></td>
                    <td>{r.duration_min ?? '—'}</td>
                    <td>{r.sampling_rate ?? '—'}</td>
                    <td>{r.montage ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h5>Neuropsych Summary (Phase 2 — {bd.neuropsych_summary?.length} assessments)</h5>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr><th>Patient</th><th>MoCA</th><th>MMSE</th><th>PHQ-9</th><th>GAD-7</th><th>Memory</th><th>Attention</th><th>Executive</th></tr>
              </thead>
              <tbody>
                {(bd.neuropsych_summary || []).map((r, i) => (
                  <tr key={i}>
                    <td><code>{r.patient_id}</code></td>
                    <td className={r.moca != null && r.moca < 26 ? 'text-danger' : ''}>{r.moca ?? '—'}</td>
                    <td className={r.mmse != null && r.mmse < 24 ? 'text-danger' : ''}>{r.mmse ?? '—'}</td>
                    <td className={r.phq9 != null && r.phq9 >= 10 ? 'text-warning' : ''}>{r.phq9 ?? '—'}</td>
                    <td className={r.gad7 != null && r.gad7 >= 10 ? 'text-warning' : ''}>{r.gad7 ?? '—'}</td>
                    <td>{r.memory_index ?? '—'}</td>
                    <td>{r.attention_index ?? '—'}</td>
                    <td>{r.executive_index ?? '—'}</td>
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
          <h5>Phase 2 Context</h5>
          <div className="alert alert-info">
            <strong>Phase {defs.phase_context?.phase}: {defs.phase_context?.name}</strong><br />
            <span className="small text-muted">Predecessor: {defs.phase_context?.predecessor}</span><br />
            <span className="small text-muted">Successor: {defs.phase_context?.successor}</span><br />
            <span className="small">{defs.phase_context?.goal}</span>
          </div>

          <h5>Modality Definitions</h5>
          {(defs.modalities || []).map(m => (
            <div key={m.code} className="card mb-3">
              <div className="card-header">
                <strong>{m.name}</strong> <span className="badge bg-secondary">{m.code}</span>
                <span className="ms-2 small text-muted">Table: <code>{m.source_table}</code></span>
              </div>
              <div className="card-body p-2">
                <p className="mb-1">{m.description}</p>
                <p className="mb-1 small"><strong>Key fields:</strong> {(m.key_fields || []).join(', ')}</p>
                <p className="mb-0 small text-success"><strong>AI potential:</strong> {m.ai_potential}</p>
              </div>
            </div>
          ))}

          <h5>Completeness Tiers</h5>
          <table className="table table-sm table-hover">
            <thead><tr><th>Tier</th><th>Description</th></tr></thead>
            <tbody>
              {(defs.completeness_tiers || []).map(t => (
                <tr key={t.tier}>
                  <td><span className={`badge bg-${tierColor(t.tier)}`}>{t.tier}</span></td>
                  <td>{t.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
