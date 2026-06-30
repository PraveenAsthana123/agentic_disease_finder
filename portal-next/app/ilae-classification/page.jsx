'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const ONSET_COLORS = { focal: '#e74c3c', generalized: '#3498db', unknown: '#95a5a6' };
const CONF_COLORS  = { high: 'success', moderate: 'info', low: 'warning' };

export default function ILAEClassificationPage() {
  const [data, setData]           = useState(null);
  const [tab, setTab]             = useState('overview');
  const [expandedSubj, setExpSub] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/ilae-classification`).then(r => r.json()).then(setData).catch(() => {});
  }, []);

  if (!data) return <div className="p-4"><div className="spinner-border text-primary" /></div>;
  if (!data.available) return <div className="p-4 alert alert-warning">{data.error || 'Data unavailable'}</div>;

  const { total_seizures, total_subjects, onset_distribution, subtype_distribution,
          confidence_distribution, awareness_distribution, subjects, classifications, taxonomy } = data;

  const focalCount = onset_distribution.find(o => o.type === 'focal')?.count || 0;
  const genCount   = onset_distribution.find(o => o.type === 'generalized')?.count || 0;
  const unkCount   = onset_distribution.find(o => o.type === 'unknown')?.count || 0;

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'subtypes',   label: 'Subtypes & Features' },
    { id: 'patients',   label: 'Subject Detail' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>ILAE 2017 Seizure Classification</h3>
      <p className="text-muted">{data.reference} — Data: {data.data_source}</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Seizures',  value: total_seizures, color: 'primary' },
          { label: 'Subjects',        value: total_subjects, color: 'secondary' },
          { label: 'Focal Onset',     value: focalCount,     color: 'danger' },
          { label: 'Generalized',     value: genCount,       color: 'info' },
          { label: 'Unknown Onset',   value: unkCount,       color: 'secondary' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md mb-2">
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
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ──────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Onset Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Onset Type Distribution</div>
              <div className="card-body">
                {onset_distribution.map(o => (
                  <div key={o.type} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="fw-semibold" style={{minWidth:140}}>{o.label}</span>
                    <div className="d-flex align-items-center" style={{width:'55%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className="progress-bar" style={{width:`${o.percent}%`, backgroundColor: ONSET_COLORS[o.type] || '#999'}} />
                      </div>
                      <span className="fw-bold">{o.count} ({o.percent}%)</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Confidence Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Classification Confidence</div>
              <div className="card-body">
                {confidence_distribution.map(c => (
                  <div key={c.level} className="d-flex justify-content-between align-items-center mb-2">
                    <span><span className={`badge bg-${CONF_COLORS[c.level] || 'secondary'} me-2`}>{c.level}</span></span>
                    <div className="d-flex align-items-center" style={{width:'55%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className={`progress-bar bg-${CONF_COLORS[c.level] || 'secondary'}`}
                             style={{width:`${total_seizures ? (c.count/total_seizures*100) : 0}%`}} />
                      </div>
                      <span className="fw-bold">{c.count}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Awareness (focal only) */}
          {awareness_distribution && awareness_distribution.length > 0 && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Focal Awareness Level</div>
                <div className="card-body">
                  {awareness_distribution.map(a => (
                    <div key={a.level} className="d-flex justify-content-between align-items-center mb-2">
                      <span className="fw-semibold text-capitalize">{a.level.replace('_', ' ')}</span>
                      <div className="d-flex align-items-center" style={{width:'55%'}}>
                        <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                          <div className="progress-bar bg-danger" style={{width:`${focalCount ? (a.count/focalCount*100) : 0}%`}} />
                        </div>
                        <span className="fw-bold">{a.count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Per-subject onset summary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Per-Subject Dominant Onset</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Subject</th><th>Seizures</th><th>Dominant Type</th><th>Distribution</th></tr></thead>
                  <tbody>
                    {subjects.map(s => (
                      <tr key={s.subject}>
                        <td className="fw-semibold">{s.subject}</td>
                        <td>{s.total_seizures}</td>
                        <td><span className="badge" style={{backgroundColor: ONSET_COLORS[s.dominant_onset_type] || '#999'}}>{s.dominant_onset_type}</span></td>
                        <td className="small">{Object.entries(s.onset_type_counts).map(([k,v]) => `${k}:${v}`).join(', ')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Subtypes & Features Tab ───────────────────────────── */}
      {tab === 'subtypes' && (
        <div className="row">
          {/* Subtype Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Seizure Subtype Distribution (Level 3)</div>
              <div className="card-body">
                {subtype_distribution.map(s => (
                  <div key={s.subtype} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="fw-semibold text-capitalize" style={{minWidth:120}}>{s.subtype}</span>
                    <div className="d-flex align-items-center" style={{width:'55%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className="progress-bar bg-primary" style={{width:`${s.percent}%`}} />
                      </div>
                      <span className="fw-bold">{s.count} ({s.percent}%)</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Feature Summary Table */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">EEG Feature Summary (All Seizures)</div>
              <div className="card-body p-0" style={{maxHeight:400, overflowY:'auto'}}>
                <table className="table table-sm table-hover mb-0" style={{fontSize:'0.82rem'}}>
                  <thead className="table-light"><tr>
                    <th>Subject</th><th>File</th><th>Dur(s)</th>
                    <th>Spread</th><th>Lat.Idx</th><th>Dom.Hz</th><th>Onset</th><th>Subtype</th><th>Conf.</th>
                  </tr></thead>
                  <tbody>
                    {classifications.map((c, i) => (
                      <tr key={i}>
                        <td>{c.subject}</td>
                        <td className="text-truncate" style={{maxWidth:100}} title={c.file}>{c.file}</td>
                        <td>{c.duration_sec}</td>
                        <td>{c.features.channel_spread.toFixed(2)}</td>
                        <td>{c.features.lateralization_index.toFixed(2)}</td>
                        <td>{c.features.dominant_freq_hz}</td>
                        <td><span className="badge" style={{backgroundColor: ONSET_COLORS[c.classification.onset_type] || '#999', fontSize:'0.7rem'}}>{c.classification.onset_type}</span></td>
                        <td className="text-capitalize">{c.classification.level3_subtype}</td>
                        <td><span className={`badge bg-${CONF_COLORS[c.classification.confidence] || 'secondary'}`} style={{fontSize:'0.7rem'}}>{c.classification.confidence}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Channel Spread vs Lateralization scatter (text-based) */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Classification Reasoning</div>
              <div className="card-body p-0" style={{maxHeight:350, overflowY:'auto'}}>
                <table className="table table-sm mb-0" style={{fontSize:'0.82rem'}}>
                  <thead className="table-light"><tr><th>#</th><th>Subject</th><th>Classification</th><th>Reasoning</th></tr></thead>
                  <tbody>
                    {classifications.map((c, i) => (
                      <tr key={i}>
                        <td>{i+1}</td>
                        <td>{c.subject}</td>
                        <td className="text-capitalize">{c.classification.onset_type} → {c.classification.level3_subtype}</td>
                        <td className="small">{c.classification.reasoning}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Subject Detail Tab ────────────────────────────────── */}
      {tab === 'patients' && (
        <div>
          {subjects.map(s => {
            const subjCls = classifications.filter(c => c.subject === s.subject);
            const isExpanded = expandedSubj === s.subject;
            return (
              <div key={s.subject} className="card mb-3 shadow-sm">
                <div className="card-header d-flex justify-content-between align-items-center"
                     style={{cursor:'pointer'}} onClick={() => setExpSub(isExpanded ? null : s.subject)}>
                  <span className="fw-bold">{s.subject}</span>
                  <span>
                    <span className="badge" style={{backgroundColor: ONSET_COLORS[s.dominant_onset_type] || '#999'}} >{s.dominant_onset_type}</span>
                    <span className="ms-2 text-muted small">{s.total_seizures} seizures</span>
                    <span className="ms-2">{isExpanded ? '▲' : '▼'}</span>
                  </span>
                </div>
                {isExpanded && (
                  <div className="card-body p-0">
                    <table className="table table-sm table-striped mb-0" style={{fontSize:'0.82rem'}}>
                      <thead className="table-light"><tr>
                        <th>File</th><th>Start</th><th>End</th><th>Dur(s)</th>
                        <th>Spread</th><th>Lat.</th><th>Hz</th><th>Amp(µV)</th>
                        <th>Onset</th><th>Awareness</th><th>Subtype</th><th>Conf.</th>
                      </tr></thead>
                      <tbody>
                        {subjCls.map((c, i) => (
                          <tr key={i}>
                            <td title={c.file}>{c.file}</td>
                            <td>{c.start_sec}s</td>
                            <td>{c.end_sec}s</td>
                            <td>{c.duration_sec}</td>
                            <td>{c.features.channel_spread.toFixed(3)}</td>
                            <td>{c.features.lateralization_index.toFixed(3)}</td>
                            <td>{c.features.dominant_freq_hz}</td>
                            <td>{c.features.mean_amplitude_uv.toFixed(1)}</td>
                            <td><span className="badge" style={{backgroundColor: ONSET_COLORS[c.classification.onset_type] || '#999', fontSize:'0.7rem'}}>{c.classification.onset_type}</span></td>
                            <td className="text-capitalize">{c.classification.level2?.replace('_',' ') || '—'}</td>
                            <td className="text-capitalize">{c.classification.level3_subtype}</td>
                            <td><span className={`badge bg-${CONF_COLORS[c.classification.confidence] || 'secondary'}`} style={{fontSize:'0.7rem'}}>{c.classification.confidence}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {/* Per-subject onset breakdown bar */}
                    <div className="p-3">
                      <strong>Onset Breakdown:</strong>
                      <div className="progress mt-1" style={{height:'24px'}}>
                        {Object.entries(s.onset_type_counts).map(([type, cnt]) => (
                          <div key={type} className="progress-bar" style={{
                            width: `${(cnt/s.total_seizures*100)}%`,
                            backgroundColor: ONSET_COLORS[type] || '#999'
                          }} title={`${type}: ${cnt}`}>
                            {type} ({cnt})
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Definitions Tab ───────────────────────────────────── */}
      {tab === 'definitions' && (
        <div className="row">
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">ILAE 2017 Seizure Classification Taxonomy</div>
              <div className="card-body">
                <p className="text-muted small mb-3">Reference: {data.reference}</p>
                {Object.entries(taxonomy).map(([key, val]) => (
                  <div key={key} className="mb-4">
                    <h5><span className="badge" style={{backgroundColor: ONSET_COLORS[key] || '#999'}}>{val.label}</span></h5>
                    <p className="small">{val.description}</p>
                    {val.awareness && (
                      <div className="mb-2">
                        <strong className="small">Awareness Levels:</strong>
                        <ul className="small mb-1">
                          {Object.entries(val.awareness).map(([ak, av]) => <li key={ak}><strong className="text-capitalize">{ak}:</strong> {av}</li>)}
                        </ul>
                      </div>
                    )}
                    <div className="row">
                      <div className="col-md-6">
                        <strong className="small">Motor Subtypes:</strong>
                        <ul className="small mb-1">{val.motor_subtypes.map(s => <li key={s} className="text-capitalize">{s}</li>)}</ul>
                      </div>
                      <div className="col-md-6">
                        <strong className="small">Non-Motor Subtypes:</strong>
                        <ul className="small mb-1">{val.non_motor_subtypes.map(s => <li key={s} className="text-capitalize">{s}</li>)}</ul>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* EEG Feature Definitions */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">EEG Feature Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Feature</th><th>Description</th><th>Classification Role</th></tr></thead>
                  <tbody>
                    {[
                      { name: 'Channel Spread', desc: 'Fraction of EEG channels showing ictal activity (0-1)', role: '<0.40 → Focal, ≥0.40 → Generalized' },
                      { name: 'Lateralization Index', desc: 'Left-right hemisphere asymmetry (0=symmetric, 1=fully lateralized)', role: '>0.2 with short duration → Focal Aware' },
                      { name: 'Duration (sec)', desc: 'Seizure length in seconds', role: '<30s focal + lateralized → Aware; >60s generalized → Tonic-Clonic' },
                      { name: 'Dominant Frequency (Hz)', desc: 'Peak frequency during ictal period', role: '2.5-4 Hz spike-wave → Absence; >12 Hz → Tonic' },
                      { name: 'Mean Amplitude (µV)', desc: 'Average absolute amplitude across channels', role: 'High-voltage bilateral → Generalized' },
                      { name: 'Spike-Wave', desc: 'Presence of 2.5-4 Hz spike-wave pattern', role: 'Classic marker for absence seizures' },
                    ].map(f => (
                      <tr key={f.name}><td className="fw-semibold">{f.name}</td><td className="small">{f.desc}</td><td className="small">{f.role}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Confidence Levels */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Confidence Levels</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Level</th><th>Criteria</th></tr></thead>
                  <tbody>
                    <tr><td><span className="badge bg-success">High</span></td><td className="small">Clear spike-wave with bilateral spread — classic absence pattern</td></tr>
                    <tr><td><span className="badge bg-info">Moderate</span></td><td className="small">Clear focal or generalized features; consistent EEG indicators</td></tr>
                    <tr><td><span className="badge bg-warning">Low</span></td><td className="small">Borderline spread (0.35-0.45), minimal lateralization, or ambiguous features</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
